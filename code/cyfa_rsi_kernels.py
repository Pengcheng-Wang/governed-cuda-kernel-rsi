# -*- coding: utf-8 -*-
"""
Round-22 RSI Kernel Suite (`E148` / `v139`):
Cyclic Flow Attention (`CyFA: Linear Sequence Modeling with Relative-Time-Partitioned Memory`,
arXiv:2609.36259, Yixiao Chen / FLA Team; `fla/ops/cyfa` & `fla/layers/cyfa.py`).

Key SM80 (`NVIDIA A100-SXM4-80GB`) Innovations:
1. Strip-Tiled Zero-Spill Cyclic Modal Readout Forward & 2-Pass Backward:
   - `_cyfa_modal_readout_fwd_rsi_kernel` (`BT=16, BC=64/32, num_warps=4, num_stages=1`)
   - `_cyfa_readout_bwd_pass1_rsi_kernel` (`BT=16, BC=32, num_warps=4, num_stages=1`)
   - `_cyfa_readout_bwd_pass2_rsi_kernel` (`BT=32, BR=64, BC=64, num_warps=4, num_stages=1`)
   - `_cyfa_readout_table_grad_rsi_kernel` + `_cyfa_readout_table_grad_reduce_rsi_kernel`
   Eliminates the `[128, 128]` FP32 table register residency and 14 simultaneous modal tiles
   that caused upstream `_chunk_cyfa_fused_readout_bwd_kernel` (`255` regs, `4,248` max spills,
   `31,638` total spills, `147,456` B SMEM) and `_chunk_cyfa_readout_table_grad_kernel`
   (`255` regs, `1,440` max spills, `5,918` total spills) to spill to local memory.
2. Live-Range-Decoupled `BR=32` Sub-Tiled Scalar-Decay Recurrence Kernels + Saved Forward States:
   - `_cyfa_chunk_fwd_h_rsi_kernel` (`BT=64, BK=64/32, BV=64/32`)
   - `_cyfa_chunk_fwd_o_rsi_kernel` (`BT=64, BR=32` two-loop intra/inter live-range split)
   - `_cyfa_chunk_bwd_dh_rsi_kernel` (`BT=64, BK=64/32, BV=32`)
   - `_cyfa_chunk_bwd_dqk_rsi_kernel` (`BT=64, BR=32` sequential `dq` -> `dk` lifecycle + atomic-free `dg_part`)
   - `_cyfa_chunk_bwd_dv_rsi_kernel` (`BT=64, BR=32` two-loop intra/inter live-range split)
   - `_cyfa_bwd_dg_reduce_rev_cumsum_rsi_kernel` (fuses dual Pass-1/Pass-2 `dg_part` + `dg_last_part` reduction + reverse cumsum)
   Eliminates all `20,614` spills and global `tl.atomic_add` calls of upstream `_chunk_cyfa_bwd_dqk_kernel`,
   all `30` spills of `chunk_bwd_kernel_dv`, and cuts `chunk_fwd_h` calls per step from `4` to `2`
   by saving `(q_norm, k_norm, write, h_k, h_v)` at `checkpoint_level == 0`.
3. Fused Clock & Decay Scans + Zero-Spill Q/K RMSNorm & Cardinal Orbit Write Kernels:
   - `_cyfa_qk_rmsnorm_fwd_rsi_kernel` (`STORE_Q=True` always, eliminating backward RMSNorm recomputation)
   - `_cyfa_qk_rmsnorm_bwd_rsi_kernel` (`(n_splits, 2)` Q/K-split 1D weight accumulator: `80–128` regs vs `255`)
   - `_cyfa_clock_and_decay_cumsum_fwd_rsi_kernel` (fuses `g` chunk-local cumsum + `delta` global clock scan + `initial_lambda`)
   - `_cyfa_clock_rev_cumsum_bwd_rsi_kernel` (reverse global clock scan with fused `dlt` & `dlambda0`)
   - `_cyfa_write_fwd_rsi_kernel` & `_cyfa_write_bwd_rsi_kernel` (`BT=16`, fusing `dlambda_readout_part` reduction)
4. Superfused `CyclicFlowAttentionBlockRSI` (`Stages A0 -> A4`) & 8-GPU NVSwitch `CyFA-CP = 8` (`Ulysses` vs `Ring`).
"""

from __future__ import annotations

import math
from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.convolution import causal_conv1d
from fla.ops.cyfa.chunk import chunk_cyfa
from fla.ops.cyfa.naive import naive_recurrent_cyfa
from fla.ops.cyfa.utils import build_readout_table
from fla.layers.cyfa import CyclicFlowAttention

RCP_LN2 = tl.constexpr(1.4426950408889634)
TWO_PI = tl.constexpr(6.283185307179586)


def _extract_compiled_kernels(kernel_obj):
    out = []
    seen = set()

    def _walk(obj, depth=0):
        if obj is None or depth > 6 or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "n_regs") and hasattr(obj, "n_spills"):
            out.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v, depth + 1)
            return
        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                _walk(v, depth + 1)
            return
        for attr in ("device_caches", "cache", "kernel", "kernels", "fn", "jit_function", "best_config"):
            if hasattr(obj, attr):
                _walk(getattr(obj, attr), depth + 1)

    _walk(kernel_obj)
    return out


@triton.jit
def _cyfa_cos_fast(x):
    return libdevice.fast_cosf(x)


@triton.jit
def _cyfa_sin_fast(x):
    return libdevice.fast_sinf(x)


@triton.jit
def _cyfa_modal_omega(o_m, M: tl.constexpr):
    pair = (o_m // 2).to(tl.float32)
    return pair * (6.283185307179586 / (M - 1))


@triton.jit
def _cyfa_modal_phase(lam, o_m, M: tl.constexpr):
    period: tl.constexpr = M - 1
    lam = lam.to(tl.float32)
    pair_i = (o_m // 2).to(tl.int32)
    whole = tl.floor(lam)
    fraction = lam - whole
    whole_mod = whole.to(tl.int32) % period
    whole_mod = tl.where(whole_mod < 0, whole_mod + period, whole_mod)
    coordinate = ((whole_mod * pair_i) % period).to(tl.float32)
    coordinate += fraction * pair_i.to(tl.float32)
    coordinate = tl.where(coordinate > period * 0.5, coordinate - period, coordinate)
    return coordinate * (6.283185307179586 / period)


@triton.jit
def _cyfa_cardinal_orbit_val(lam, o_m, M: tl.constexpr):
    is_cos = (o_m % 2) == 0
    phase = _cyfa_modal_phase(lam, o_m, M)
    scale = tl.sqrt(tl.where((o_m // 2) == 0, 1.0, 2.0) / (M - 1))
    return scale * tl.where(is_cos, _cyfa_cos_fast(phase), -_cyfa_sin_fast(phase))


@triton.jit
def _cyfa_cardinal_orbit_deriv(lam, o_m, M: tl.constexpr):
    is_cos = (o_m % 2) == 0
    omega = _cyfa_modal_omega(o_m, M)
    phase = _cyfa_modal_phase(lam, o_m, M)
    scale = tl.sqrt(tl.where((o_m // 2) == 0, 1.0, 2.0) / (M - 1))
    return -scale * omega * tl.where(is_cos, _cyfa_sin_fast(phase), _cyfa_cos_fast(phase))


# ============================================================================
# 1. Zero-Spill Q/K RMSNorm Forward & Backward Kernels
# ============================================================================

@triton.jit(do_not_specialize=['N'])
def _cyfa_qk_rmsnorm_fwd_rsi_kernel(
    q, k, q_weight, k_weight, q_out, k_out, q_rstd, k_rstd,
    q_eps, k_eps, N,
    D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    o_n = i_n * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_n = o_n < N
    m_d = o_d < D
    mask = m_n[:, None] & m_d[None, :]
    offsets = o_n[:, None] * D + o_d[None, :]

    b_q = tl.load(q + offsets, mask=mask, other=0.0).to(tl.float32)
    b_q_rstd = tl.rsqrt(tl.sum(b_q * b_q, axis=1) / D + q_eps)
    b_qw = tl.load(q_weight + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_q_norm = b_q * b_q_rstd[:, None] * b_qw[None, :]
    tl.store(q_out + offsets, b_q_norm.to(q_out.dtype.element_ty), mask=mask)
    tl.store(q_rstd + o_n, b_q_rstd, mask=m_n)

    b_k = tl.load(k + offsets, mask=mask, other=0.0).to(tl.float32)
    b_k_rstd = tl.rsqrt(tl.sum(b_k * b_k, axis=1) / D + k_eps)
    b_kw = tl.load(k_weight + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_k_norm = b_k * b_k_rstd[:, None] * b_kw[None, :]
    tl.store(k_out + offsets, b_k_norm.to(k_out.dtype.element_ty), mask=mask)
    tl.store(k_rstd + o_n, b_k_rstd, mask=m_n)


def cyfa_qk_rmsnorm_fwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    n_rows = q.numel() // q.shape[-1]
    d = q.shape[-1]
    q_rstd = torch.empty(n_rows, device=q.device, dtype=torch.float32)
    k_rstd = torch.empty(n_rows, device=q.device, dtype=torch.float32)
    bt = 16 if d >= 256 else 32
    _cyfa_qk_rmsnorm_fwd_rsi_kernel[(triton.cdiv(n_rows, bt),)](
        q, k, q_weight, k_weight, q_out, k_out, q_rstd, k_rstd,
        float(q_eps), float(k_eps), N=n_rows, D=d, BT=bt, BD=triton.next_power_of_2(d),
        num_warps=4, num_stages=1,
    )
    return q_out, k_out, q_rstd, k_rstd


@triton.jit(do_not_specialize=['N'])
def _cyfa_qk_rmsnorm_bwd_rsi_kernel(
    q, k, dq_out, dk_out, q_weight, k_weight, q_rstd, k_rstd,
    dq, dk, dweight,
    N,
    D: tl.constexpr, BS: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
):
    i_s = tl.program_id(0).to(tl.int64)
    i_qk = tl.program_id(1).to(tl.int64)
    o_d = tl.arange(0, BD)
    m_d = o_d < D

    x_ptr = q if i_qk == 0 else k
    dx_out_ptr = dq_out if i_qk == 0 else dk_out
    w_ptr = q_weight if i_qk == 0 else k_weight
    rstd_ptr = q_rstd if i_qk == 0 else k_rstd
    dx_ptr = dq if i_qk == 0 else dk

    b_w = tl.load(w_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_dw = tl.zeros((BD,), dtype=tl.float32)
    o_n_local = tl.arange(0, BT)

    for i_n in range(i_s * BS, i_s * BS + BS, BT):
        o_n = i_n + o_n_local
        m_n = o_n < N
        m_acc = o_n < tl.minimum((i_s + 1) * BS, N)
        mask = m_n[:, None] & m_d[None, :]
        acc_mask = m_acc[:, None] & m_d[None, :]
        offsets = o_n_local[:, None] * D + o_d[None, :]

        b_x = tl.load((x_ptr + i_n * D) + offsets, mask=mask, other=0.0).to(tl.float32)
        b_dx_out = tl.load((dx_out_ptr + i_n * D) + offsets, mask=mask, other=0.0).to(tl.float32)
        b_rstd = tl.load((rstd_ptr + i_n) + o_n_local, mask=m_n, other=0.0)
        b_x_hat = tl.where(m_d[None, :], b_x * b_rstd[:, None], 0.0)
        b_wdy = b_dx_out * b_w[None, :]
        b_proj = tl.sum(b_x_hat * b_wdy, axis=1) / D
        b_dx = (b_wdy - b_x_hat * b_proj[:, None]) * b_rstd[:, None]
        tl.store((dx_ptr + i_n * D) + offsets, b_dx.to(dx_ptr.dtype.element_ty), mask=mask)
        b_dw += tl.sum(tl.where(acc_mask, b_dx_out * b_x_hat, 0.0), axis=0)

    n_splits = tl.num_programs(0).to(tl.int64)
    tl.store(dweight + (i_qk * n_splits + i_s) * D + o_d, b_dw, mask=m_d)


def cyfa_qk_rmsnorm_bwd_rsi(
    dq_out: torch.Tensor,
    dk_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    dq_out = dq_out.contiguous()
    dk_out = dk_out.contiguous()
    n_rows = q.numel() // q.shape[-1]
    d_head = q.shape[-1]
    n_splits = min(108, n_rows)
    block_rows = triton.cdiv(n_rows, n_splits)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dweight = torch.empty((2, n_splits, d_head), device=q.device, dtype=torch.float32)
    _cyfa_qk_rmsnorm_bwd_rsi_kernel[(n_splits, 2)](
        q, k, dq_out, dk_out, q_weight, k_weight, q_rstd, k_rstd,
        dq, dk, dweight,
        N=n_rows, D=d_head, BS=block_rows, BT=16, BD=triton.next_power_of_2(d_head),
        num_warps=4, num_stages=2,
    )
    dq_w, dk_w = dweight.sum(dim=1).to(q_weight.dtype).unbind(0)
    return dq, dk, dq_w, dk_w


# ============================================================================
# 2. Fused BT=256 (4x64) Clock & Decay Cumsum Forward + Reverse Clock Backward
# ============================================================================

@triton.jit(do_not_specialize=['T'])
def _cyfa_clock_and_decay_cumsum_fwd_rsi_kernel(
    g_in, delta_in, initial_lambda,
    g_cumsum_out, lambdas_out, final_lambda_out,
    T,
    H: tl.constexpr, BT: tl.constexpr,
    HAS_INIT_LAMBDA: tl.constexpr, HAS_FINAL_LAMBDA: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    running_clock = tl.full((), 0.0, tl.float32)
    if HAS_INIT_LAMBDA:
        running_clock = tl.load(initial_lambda + i_bh).to(tl.float32)

    nt = tl.cdiv(T, BT)
    for i_t in range(0, nt):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        idx = (bos + o_t) * H + i_h
        b_g = tl.load(g_in + idx, mask=m_t, other=0.0).to(tl.float32)
        b_d = tl.load(delta_in + idx, mask=m_t, other=0.0).to(tl.float32)

        b_g_2d = tl.reshape(b_g, (BT // 64, 64))
        b_g_cs = tl.reshape(tl.cumsum(b_g_2d, 1), (BT,)) * RCP_LN2
        b_lam = tl.cumsum(b_d, 0) + running_clock
        running_clock += tl.sum(b_d, 0)

        tl.store(g_cumsum_out + idx, b_g_cs, mask=m_t)
        tl.store(lambdas_out + idx, b_lam, mask=m_t)

    if HAS_FINAL_LAMBDA:
        tl.store(final_lambda_out + i_bh, running_clock)


@triton.jit(do_not_specialize=['T'])
def _cyfa_clock_rev_cumsum_bwd_rsi_kernel(
    phase_force, dlt, ddelta_out, dlambda0_out,
    T,
    H: tl.constexpr, BT: tl.constexpr,
    HAS_DLT: tl.constexpr, HAS_DLAMBDA0: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    running = tl.full((), 0.0, tl.float32)
    if HAS_DLT:
        running = tl.load(dlt + i_bh).to(tl.float32)

    nt = tl.cdiv(T, BT)
    for i_t in range(nt - 1, -1, -1):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        idx = (bos + o_t) * H + i_h
        vals = tl.load(phase_force + idx, mask=m_t, other=0.0).to(tl.float32)
        chunk_sum = tl.sum(vals, 0)
        rev_prefix = chunk_sum - tl.cumsum(vals, 0) + vals + running
        running += chunk_sum
        tl.store(ddelta_out + idx, rev_prefix.to(ddelta_out.dtype.element_ty), mask=m_t)

    if HAS_DLAMBDA0:
        tl.store(dlambda0_out + i_bh, running)


# ============================================================================
# 3. Zero-Spill Cardinal Orbit Write Forward & Fused Write+Readout Clock Backward
# ============================================================================

@triton.jit(do_not_specialize=['T'])
def _cyfa_write_fwd_rsi_kernel(
    lambdas, beta, write,
    T,
    H: tl.constexpr, M: tl.constexpr, BT: tl.constexpr, BM: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_m = tl.arange(0, BM)
    m_t = o_t < T
    m_m = o_m < M

    b_lam = tl.load(lambdas + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_beta = tl.load(beta + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    w_val = b_beta[:, None] * _cyfa_cardinal_orbit_val(b_lam[:, None], o_m[None, :], M)
    p = ((i_b * T + o_t[:, None]) * H + i_h) * M + o_m[None, :]
    tl.store(write + p, w_val.to(write.dtype.element_ty), mask=m_t[:, None] & m_m[None, :])


@triton.jit(do_not_specialize=['T'])
def _cyfa_write_bwd_rsi_kernel(
    dwrite_a, dwrite_b, lambdas, beta, dlambda_readout_part,
    phase_force_out, dbeta_out,
    T,
    H: tl.constexpr, M: tl.constexpr, BT: tl.constexpr, BM: tl.constexpr, NC_READOUT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_m = tl.arange(0, BM)
    m_t = o_t < T
    mask = m_t[:, None] & (o_m[None, :] < M)

    p = ((i_b * T + o_t[:, None]) * H + i_h) * M + o_m[None, :]
    dw = tl.load(dwrite_a + p, mask=mask, other=0.0).to(tl.float32)
    dw += tl.load(dwrite_b + p, mask=mask, other=0.0).to(tl.float32)

    offset = (i_b * T + o_t) * H + i_h
    b_lam = tl.load(lambdas + offset, mask=m_t, other=0.0).to(tl.float32)
    b_beta = tl.load(beta + offset, mask=m_t, other=0.0).to(tl.float32)

    endpoint = _cyfa_cardinal_orbit_val(b_lam[:, None], o_m[None, :], M)
    tangent = _cyfa_cardinal_orbit_deriv(b_lam[:, None], o_m[None, :], M)
    d_phase = tl.sum(tl.where(mask, dw * b_beta[:, None] * tangent, 0.0), axis=1)
    d_write = tl.sum(tl.where(mask, dw * endpoint, 0.0), axis=1)

    n_b = tl.num_programs(1).to(tl.int64) // H
    n_total = n_b * T * H
    for i_c in range(0, NC_READOUT):
        d_phase += tl.load(dlambda_readout_part + i_c * n_total + offset, mask=m_t, other=0.0).to(tl.float32)

    tl.store(phase_force_out + offset, d_phase, mask=m_t)
    tl.store(dbeta_out + offset, d_write.to(dbeta_out.dtype.element_ty), mask=m_t)


# ============================================================================
# 4. Zero-Spill Strip-Tiled Cyclic Modal Readout Forward, Backward, & Table Grad
# ============================================================================

@triton.jit(do_not_specialize=['N'])
def _cyfa_modal_readout_fwd_rsi_kernel(
    raw_logits, lambdas, table, weights,
    N,
    H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
    BT: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_blk = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_n_local = tl.arange(0, BT)
    o_n = i_blk * BT + o_n_local
    mask_n = o_n < N
    o_r = tl.arange(0, BM)
    mask_r = o_r < C

    token_head_base = i_blk * BT * H + i_h
    b_lambda = tl.load(lambdas + token_head_base + o_n_local * H, mask=mask_n, other=0.0).to(tl.float32)

    logits = tl.zeros((BT, BM), dtype=tl.float32)
    for i_c in range(0, tl.cdiv(M, BC)):
        o_j = i_c * BC + tl.arange(0, BC)
        mask_j = o_j < M
        p_x = (token_head_base * M) + o_n_local[:, None] * (H * M) + o_j[None, :]
        b_x = tl.load(raw_logits + p_x, mask=mask_n[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
        x_even, x_odd = tl.split(tl.reshape(b_x, (BT, BC // 2, 2)))
        o_even = i_c * BC + 2 * tl.arange(0, BC // 2)
        phase = _cyfa_modal_phase(b_lambda[:, None], o_even[None, :], M)
        c = _cyfa_cos_fast(phase)
        s = _cyfa_sin_fast(phase)
        z = tl.reshape(tl.join(c * x_even - s * x_odd, s * x_even + c * x_odd), (BT, BC))
        z = tl.where(mask_n[:, None] & mask_j[None, :], z, 0.0)
        w = tl.load(
            table + i_h * C * M + o_r[:, None] * M + o_j[None, :],
            mask=mask_r[:, None] & mask_j[None, :],
            other=0.0,
        )
        if DOT_IEEE:
            logits += tl.dot(z, tl.trans(w.to(tl.float32)), input_precision='ieee')
        else:
            logits += tl.dot(z.to(raw_logits.dtype.element_ty), tl.trans(w.to(raw_logits.dtype.element_ty)))

    logits = tl.where(mask_r[None, :], logits, -float('inf'))
    max_logits = tl.max(logits, axis=1)
    probs = tl.where(mask_r[None, :], tl.exp(logits - max_logits[:, None]), 0.0)
    denom = tl.sum(probs, axis=1)
    probs *= tl.where(denom > 0.0, 1.0 / denom, 0.0)[:, None]
    p_dt = probs.to(weights.dtype.element_ty)

    for i_c in range(0, tl.cdiv(M, BC)):
        o_j = i_c * BC + tl.arange(0, BC)
        mask_j = o_j < M
        w = tl.load(
            table + i_h * C * M + o_r[:, None] * M + o_j[None, :],
            mask=mask_r[:, None] & mask_j[None, :],
            other=0.0,
        )
        if DOT_IEEE:
            u = tl.dot(probs, w.to(tl.float32), input_precision='ieee')
        else:
            u = tl.dot(p_dt, w.to(weights.dtype.element_ty))
        u_even, u_odd = tl.split(tl.reshape(u, (BT, BC // 2, 2)))
        o_even = i_c * BC + 2 * tl.arange(0, BC // 2)
        phase = _cyfa_modal_phase(b_lambda[:, None], o_even[None, :], M)
        c = _cyfa_cos_fast(phase)
        s = _cyfa_sin_fast(phase)
        y = tl.reshape(tl.join(c * u_even + s * u_odd, -s * u_even + c * u_odd), (BT, BC))
        p_x = (token_head_base * M) + o_n_local[:, None] * (H * M) + o_j[None, :]
        tl.store(weights + p_x, y.to(weights.dtype.element_ty), mask=mask_n[:, None] & mask_j[None, :])


@triton.jit(do_not_specialize=['N'])
def _cyfa_readout_bwd_pass1_rsi_kernel(
    lambdas, table, raw_logits, dweights,
    probs, dlogits,
    N,
    H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
    BT: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_blk = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_n_local = tl.arange(0, BT)
    o_n = i_blk * BT + o_n_local
    mask_n = o_n < N
    o_r = tl.arange(0, BM)
    mask_r = o_r < C

    token_head_base = i_blk * BT * H + i_h
    lambdas += token_head_base
    raw_logits += token_head_base * M
    dweights += token_head_base * M
    probs += token_head_base * C
    dlogits += token_head_base * C

    b_lambda = tl.load(lambdas + o_n_local * H, mask=mask_n, other=0.0).to(tl.float32)
    scores = tl.zeros((BT, BM), dtype=tl.float32)
    dp = tl.zeros((BT, BM), dtype=tl.float32)

    for i_c in range(0, tl.cdiv(M, BC)):
        o_j = i_c * BC + tl.arange(0, BC)
        mask_j = o_j < M
        p_x = o_n_local[:, None] * (H * M) + o_j[None, :]
        raw = tl.load(raw_logits + p_x, mask=mask_n[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
        dw = tl.load(dweights + p_x, mask=mask_n[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
        raw_even, raw_odd = tl.split(tl.reshape(raw, (BT, BC // 2, 2)))
        dw_even, dw_odd = tl.split(tl.reshape(dw, (BT, BC // 2, 2)))
        o_even = i_c * BC + 2 * tl.arange(0, BC // 2)
        phase = _cyfa_modal_phase(b_lambda[:, None], o_even[None, :], M)
        c = _cyfa_cos_fast(phase)
        s = _cyfa_sin_fast(phase)
        raw_z = tl.reshape(tl.join(c * raw_even - s * raw_odd, s * raw_even + c * raw_odd), (BT, BC))
        dw_z = tl.reshape(tl.join(c * dw_even - s * dw_odd, s * dw_even + c * dw_odd), (BT, BC))
        w = tl.load(
            table + i_h * C * M + o_r[:, None] * M + o_j[None, :],
            mask=mask_r[:, None] & mask_j[None, :],
            other=0.0,
        )
        if DOT_IEEE:
            scores += tl.dot(raw_z, tl.trans(w.to(tl.float32)), input_precision='ieee')
            dp += tl.dot(dw_z, tl.trans(w.to(tl.float32)), input_precision='ieee')
        else:
            w_dt = w.to(raw_logits.dtype.element_ty)
            scores += tl.dot(raw_z.to(raw_logits.dtype.element_ty), tl.trans(w_dt))
            dp += tl.dot(dw_z.to(dweights.dtype.element_ty), tl.trans(w_dt))

    scores = tl.where(mask_r[None, :], scores, -float('inf'))
    max_scores = tl.max(scores, axis=1)
    p = tl.where(mask_r[None, :], tl.exp(scores - max_scores[:, None]), 0.0)
    denom = tl.sum(p, axis=1)
    p *= tl.where(denom > 0.0, 1.0 / denom, 0.0)[:, None]
    correction = tl.sum(p * dp, axis=1)
    dl = p * (dp - correction[:, None])
    p_probs = o_n_local[:, None] * (H * C) + o_r[None, :]
    tl.store(probs + p_probs, p.to(probs.dtype.element_ty), mask=mask_n[:, None] & mask_r[None, :])
    tl.store(dlogits + p_probs, dl.to(dlogits.dtype.element_ty), mask=mask_n[:, None] & mask_r[None, :])


@triton.jit(do_not_specialize=['N'])
def _cyfa_readout_bwd_pass2_rsi_kernel(
    lambdas, table, raw_logits, dweights,
    probs, dlogits, draw_logits, dlambda_part,
    N,
    H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
    BT: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_blk = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_h = tl.program_id(2).to(tl.int64)
    o_n_local = tl.arange(0, BT)
    o_n = i_blk * BT + o_n_local
    mask_n = o_n < N
    o_j = i_c * BC + tl.arange(0, BC)
    mask_j = o_j < M

    token_head_base = i_blk * BT * H + i_h
    b_lambda = tl.load(lambdas + token_head_base + o_n_local * H, mask=mask_n, other=0.0).to(tl.float32)

    coeff = tl.zeros((BT, BC), dtype=tl.float32)
    grad = tl.zeros((BT, BC), dtype=tl.float32)
    for i_r in range(0, tl.cdiv(C, BR)):
        o_r = i_r * BR + tl.arange(0, BR)
        mask_r = o_r < C
        p_pr = (token_head_base * C) + o_n_local[:, None] * (H * C) + o_r[None, :]
        b_p = tl.load(probs + p_pr, mask=mask_n[:, None] & mask_r[None, :], other=0.0)
        b_dl = tl.load(dlogits + p_pr, mask=mask_n[:, None] & mask_r[None, :], other=0.0)
        w = tl.load(
            table + i_h * C * M + o_r[:, None] * M + o_j[None, :],
            mask=mask_r[:, None] & mask_j[None, :],
            other=0.0,
        )
        if DOT_IEEE:
            coeff += tl.dot(b_p.to(tl.float32), w.to(tl.float32), input_precision='ieee')
            grad += tl.dot(b_dl.to(tl.float32), w.to(tl.float32), input_precision='ieee')
        else:
            w_low = w.to(probs.dtype.element_ty)
            coeff += tl.dot(b_p, w_low)
            grad += tl.dot(b_dl, w_low)

    p_x = (token_head_base * M) + o_n_local[:, None] * (H * M) + o_j[None, :]
    raw = tl.load(raw_logits + p_x, mask=mask_n[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
    dw = tl.load(dweights + p_x, mask=mask_n[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
    raw_even, raw_odd = tl.split(tl.reshape(raw, (BT, BC // 2, 2)))
    dw_even, dw_odd = tl.split(tl.reshape(dw, (BT, BC // 2, 2)))
    coeff_even, coeff_odd = tl.split(tl.reshape(coeff, (BT, BC // 2, 2)))
    grad_even, grad_odd = tl.split(tl.reshape(grad, (BT, BC // 2, 2)))

    o_even = i_c * BC + 2 * tl.arange(0, BC // 2)
    mask_p = (o_even + 1) < M
    omega = _cyfa_modal_omega(o_even, M)
    phase = _cyfa_modal_phase(b_lambda[:, None], o_even[None, :], M)
    c = _cyfa_cos_fast(phase)
    s = _cyfa_sin_fast(phase)

    dx_even = c * grad_even + s * grad_odd
    dx_odd = -s * grad_even + c * grad_odd
    dx = tl.reshape(tl.join(dx_even, dx_odd), (BT, BC))
    tl.store(draw_logits + p_x, dx.to(draw_logits.dtype.element_ty), mask=mask_n[:, None] & mask_j[None, :])

    force_v = coeff_even * (-s * dw_even - c * dw_odd) + coeff_odd * (c * dw_even - s * dw_odd)
    force_k = grad_even * (-s * raw_even - c * raw_odd) + grad_odd * (c * raw_even - s * raw_odd)
    force = tl.sum(tl.where(mask_n[:, None] & mask_p[None, :], omega[None, :] * (force_v + force_k), 0.0), axis=1)
    tl.store(dlambda_part + i_c * (N * H) + token_head_base + o_n_local * H, force, mask=mask_n)


@triton.jit(do_not_specialize=['T'])
def _cyfa_readout_table_grad_rsi_kernel(
    lambdas, left_a, right_a, left_b, right_b, dtable,
    T,
    H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
    BN: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_r = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_hn = tl.program_id(2).to(tl.int64)
    i_h = i_hn % H
    i_n = i_hn // H
    o_r = i_r * BR + tl.arange(0, BR)
    o_c = i_c * BC + tl.arange(0, BC)
    m_r = o_r < C
    m_c = o_c < M
    acc = tl.zeros((BR, BC), tl.float32)
    n_splits = tl.num_programs(2).to(tl.int64) // H
    o_n_local = tl.arange(0, BN)
    o_even = i_c * BC + 2 * tl.arange(0, BC // 2)

    for start in range(i_n * BN, T, n_splits * BN):
        o_n = start + o_n_local
        token_head_base = start * H + i_h
        m_n = o_n < T
        b_lambda = tl.load(lambdas + token_head_base + o_n_local * H, mask=m_n, other=0.0).to(tl.float32)
        phase = _cyfa_modal_phase(b_lambda[:, None], o_even[None, :], M)
        co = _cyfa_cos_fast(phase)
        si = _cyfa_sin_fast(phase)

        p_l = o_n_local[:, None] * (H * C) + o_r[None, :]
        p_x = o_n_local[:, None] * (H * M) + o_c[None, :]

        l_a = tl.load((left_a + token_head_base * C) + p_l, mask=m_n[:, None] & m_r[None, :], other=0.0)
        x_a = tl.load((right_a + token_head_base * M) + p_x, mask=m_n[:, None] & m_c[None, :], other=0.0).to(tl.float32)
        a_even, a_odd = tl.split(tl.reshape(x_a, (BN, BC // 2, 2)))
        z_a = tl.reshape(tl.join(co * a_even - si * a_odd, si * a_even + co * a_odd), (BN, BC))
        if DOT_IEEE:
            acc += tl.dot(tl.trans(l_a.to(tl.float32)), z_a, input_precision='ieee')
        else:
            acc += tl.dot(tl.trans(l_a.to(right_a.dtype.element_ty)), z_a.to(right_a.dtype.element_ty))

        l_b = tl.load((left_b + token_head_base * C) + p_l, mask=m_n[:, None] & m_r[None, :], other=0.0)
        x_b = tl.load((right_b + token_head_base * M) + p_x, mask=m_n[:, None] & m_c[None, :], other=0.0).to(tl.float32)
        b_even, b_odd = tl.split(tl.reshape(x_b, (BN, BC // 2, 2)))
        z_b = tl.reshape(tl.join(co * b_even - si * b_odd, si * b_even + co * b_odd), (BN, BC))
        if DOT_IEEE:
            acc += tl.dot(tl.trans(l_b.to(tl.float32)), z_b, input_precision='ieee')
        else:
            acc += tl.dot(tl.trans(l_b.to(right_b.dtype.element_ty)), z_b.to(right_b.dtype.element_ty))

    tl.store(
        dtable + (i_n * H + i_h) * C * M + o_r[:, None] * M + o_c[None, :],
        acc,
        mask=m_r[:, None] & m_c[None, :],
    )


@triton.jit(do_not_specialize=['S'])
def _cyfa_readout_table_grad_reduce_rsi_kernel(
    partials, output, S, SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    o_x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    o_s = tl.arange(0, 32)
    x = tl.load(
        partials + o_s[:, None] * SIZE + o_x[None, :],
        mask=(o_s[:, None] < S) & (o_x[None, :] < SIZE),
        other=0.0,
    )
    tl.store(output + o_x, tl.sum(x, axis=0), mask=o_x < SIZE)


# ============================================================================
# 5. Live-Range-Decoupled Zero-Spill Scalar-Decay Recurrence Kernels
# ============================================================================

@triton.jit(do_not_specialize=['T'])
def _cyfa_chunk_fwd_h_rsi_kernel(
    k, v, g, h, h0, ht,
    T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, OUTPUT_FINAL_STATE: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    nt = tl.cdiv(T, BT)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V

    b_h = tl.zeros((BV, BK), dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + (i_b * H + i_h).to(tl.int64) * K * V + o_k[None, :] * V + o_v[:, None]
        b_h = tl.load(p_h0, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)

    bos = i_b * T
    k_base = k + (bos * H + i_h) * K
    v_base = v + (bos * H + i_h) * V
    g_base = g + bos * H + i_h

    for i_t in range(0, nt):
        i_tg = i_b * nt + i_t
        p_h = h + ((i_tg * H + i_h).to(tl.int64) * V + o_v[:, None]) * K + o_k[None, :]
        tl.store(p_h, b_h.to(h.dtype.element_ty), mask=m_v[:, None] & m_k[None, :])

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        last_idx = tl.minimum(i_t * BT + BT, T) - 1
        b_g_last = tl.load(g_base + last_idx * H)
        b_g = tl.load(g_base + o_t * H, mask=m_t, other=0.0)
        b_h *= tl.exp2(b_g_last)

        p_k = k_base + o_t[:, None] * (H * K) + o_k[None, :]
        p_v = v_base + o_t[:, None] * (H * V) + o_v[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)
        decay = tl.where(m_t, tl.exp2(b_g_last - b_g), 0.0)
        if DOT_IEEE:
            b_k_d = b_k.to(tl.float32) * decay[:, None]
            b_h += tl.dot(tl.trans(b_v.to(tl.float32)), b_k_d, input_precision='ieee')
        else:
            b_k_d = (b_k.to(tl.float32) * decay[:, None]).to(b_k.dtype)
            b_h += tl.dot(tl.trans(b_v), b_k_d)

    if OUTPUT_FINAL_STATE:
        p_ht = ht + (i_b * H + i_h).to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, tl.trans(b_h).to(ht.dtype.element_ty), mask=m_k[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['T'])
def _cyfa_chunk_fwd_o_rsi_kernel(
    q, k, v, h, g, o,
    scale, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_tr = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    n_r = BT // BR
    i_t = i_tr // n_r
    i_r = i_tr % n_r
    i_b = i_bh // H
    i_h = i_bh % H
    nt = tl.cdiv(T, BT)
    i_tg = i_b * nt + i_t
    bos = i_b * T

    o_r = i_t * BT + i_r * BR + tl.arange(0, BR)
    o_s = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_r = o_r < T
    m_s = o_s < T
    m_v = o_v < V

    b_g_r = tl.load(g + (bos + o_r) * H + i_h, mask=m_r, other=0.0)
    b_g_s = tl.load(g + (bos + o_s) * H + i_h, mask=m_s, other=0.0)
    q_base = q + (bos * H + i_h) * K
    k_base = k + (bos * H + i_h) * K
    h_base = h + (i_tg * H + i_h).to(tl.int64) * V * K

    b_A = tl.zeros((BR, BT), dtype=tl.float32)
    for i_k in range(0, tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_q = tl.load(q_base + o_r[:, None] * (H * K) + o_k[None, :], mask=m_r[:, None] & m_k[None, :], other=0.0)
        b_k = tl.load(k_base + o_s[:, None] * (H * K) + o_k[None, :], mask=m_s[:, None] & m_k[None, :], other=0.0)
        if DOT_IEEE:
            b_A += tl.dot(b_q.to(tl.float32), tl.trans(b_k.to(tl.float32)), input_precision='ieee')
        else:
            b_A += tl.dot(b_q, tl.trans(b_k))

    m_A = (o_r[:, None] >= o_s[None, :]) & m_r[:, None] & m_s[None, :]
    b_A = tl.where(m_A, b_A * tl.exp2(b_g_r[:, None] - b_g_s[None, :]), 0.0)
    b_v = tl.load(v + (bos * H + i_h) * V + o_s[:, None] * (H * V) + o_v[None, :], mask=m_s[:, None] & m_v[None, :], other=0.0)
    if DOT_IEEE:
        b_o = tl.dot(b_A, b_v.to(tl.float32), input_precision='ieee')
    else:
        b_o = tl.dot(b_A.to(b_v.dtype), b_v)

    b_o_inter = tl.zeros((BR, BV), dtype=tl.float32)
    for i_k in range(0, tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_q = tl.load(q_base + o_r[:, None] * (H * K) + o_k[None, :], mask=m_r[:, None] & m_k[None, :], other=0.0)
        b_h = tl.load(h_base + o_v[:, None] * K + o_k[None, :], mask=m_v[:, None] & m_k[None, :], other=0.0)
        if DOT_IEEE:
            b_o_inter += tl.dot(b_q.to(tl.float32), tl.trans(b_h.to(tl.float32)), input_precision='ieee')
        else:
            b_o_inter += tl.dot(b_q, tl.trans(b_h.to(b_q.dtype)))

    b_o = (b_o_inter * tl.exp2(b_g_r)[:, None] + b_o) * scale
    p_o = o + (bos * H + i_h) * V + o_r[:, None] * (H * V) + o_v[None, :]
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=m_r[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['T'])
def _cyfa_chunk_bwd_dh_rsi_kernel(
    q, g, do, dh, dht, dh0,
    scale, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_FINAL_STATE_GRAD: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_k = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    nt = tl.cdiv(T, BT)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V

    b_dh = tl.zeros((BK, BV), dtype=tl.float32)
    if USE_FINAL_STATE_GRAD:
        p_dht = dht + (i_b * H + i_h).to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        b_dh = tl.load(p_dht, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)

    bos = i_b * T
    q_base = q + (bos * H + i_h) * K
    do_base = do + (bos * H + i_h) * V
    g_base = g + bos * H + i_h

    for i_t in range(nt - 1, -1, -1):
        i_tg = i_b * nt + i_t
        p_dh = dh + ((i_tg * H + i_h).to(tl.int64) * K + o_k[:, None]) * V + o_v[None, :]
        tl.store(p_dh, b_dh.to(dh.dtype.element_ty), mask=m_k[:, None] & m_v[None, :])

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        last_idx = tl.minimum(i_t * BT + BT, T) - 1
        b_g_last = tl.load(g_base + last_idx * H)
        b_g = tl.load(g_base + o_t * H, mask=m_t, other=0.0)
        b_dh *= tl.exp2(b_g_last)

        p_q = q_base + o_t[:, None] * (H * K) + o_k[None, :]
        p_do = do_base + o_t[:, None] * (H * V) + o_v[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)
        q_scale = (tl.exp2(b_g) * scale)[:, None]
        if DOT_IEEE:
            b_q_s = b_q.to(tl.float32) * q_scale
            b_dh += tl.dot(tl.trans(b_q_s), b_do.to(tl.float32), input_precision='ieee')
        else:
            b_q_s = (b_q.to(tl.float32) * q_scale).to(b_q.dtype)
            b_dh += tl.dot(tl.trans(b_q_s), b_do)

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + (i_b * H + i_h).to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(dh0.dtype.element_ty), mask=m_k[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['T'])
def _cyfa_chunk_bwd_dqk_rsi_kernel(
    q, k, v, g, h, do, dh, dq, dk, dg_part, dg_last_part,
    scale, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_k = tl.program_id(0).to(tl.int64)
    i_tr = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    n_r = BT // BR
    i_t = i_tr // n_r
    i_r = i_tr % n_r
    i_b = i_bh // H
    i_h = i_bh % H
    nt = tl.cdiv(T, BT)
    i_tg = i_b * nt + i_t
    bos = i_b * T

    o_r = i_t * BT + i_r * BR + tl.arange(0, BR)
    o_s = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_r = o_r < T
    m_s = o_s < T
    m_k = o_k < K

    v_base = v + (bos * H + i_h) * V
    do_base = do + (bos * H + i_h) * V
    q_base = q + (bos * H + i_h) * K
    k_base = k + (bos * H + i_h) * K
    h_base = h + (i_tg * H + i_h).to(tl.int64) * V * K
    dh_base = dh + (i_tg * H + i_h).to(tl.int64) * K * V

    b_g_r = tl.load(g + (bos + o_r) * H + i_h, mask=m_r, other=0.0)
    b_g_s = tl.load(g + (bos + o_s) * H + i_h, mask=m_s, other=0.0)
    last_idx = tl.minimum(i_t * BT + BT, T) - 1
    b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

    # Half 1: dq lifecycle
    b_ds_q = tl.zeros((BR, BT), dtype=tl.float32)
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_do_r = tl.load(do_base + o_r[:, None] * (H * V) + o_v[None, :], mask=m_r[:, None] & m_v[None, :], other=0.0)
        b_v_s = tl.load(v_base + o_s[:, None] * (H * V) + o_v[None, :], mask=m_s[:, None] & m_v[None, :], other=0.0)
        if DOT_IEEE:
            b_ds_q += tl.dot(b_do_r.to(tl.float32), tl.trans(b_v_s.to(tl.float32)), input_precision='ieee')
        else:
            b_ds_q += tl.dot(b_do_r, tl.trans(b_v_s))

    m_A_q = (o_r[:, None] >= o_s[None, :]) & m_r[:, None] & m_s[None, :]
    b_ds_q = tl.where(m_A_q, b_ds_q * tl.exp2(b_g_r[:, None] - b_g_s[None, :]) * scale, 0.0)
    b_k_s = tl.load(k_base + o_s[:, None] * (H * K) + o_k[None, :], mask=m_s[:, None] & m_k[None, :], other=0.0)
    if DOT_IEEE:
        b_dq_intra = tl.dot(b_ds_q, b_k_s.to(tl.float32), input_precision='ieee')
    else:
        b_dq_intra = tl.dot(b_ds_q.to(b_k_s.dtype), b_k_s)

    b_dq = tl.zeros((BR, BK), dtype=tl.float32)
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_do_r = tl.load(do_base + o_r[:, None] * (H * V) + o_v[None, :], mask=m_r[:, None] & m_v[None, :], other=0.0)
        b_h = tl.load(h_base + o_v[:, None] * K + o_k[None, :], mask=m_v[:, None] & m_k[None, :], other=0.0)
        if DOT_IEEE:
            b_dq += tl.dot(b_do_r.to(tl.float32), b_h.to(tl.float32), input_precision='ieee')
        else:
            b_dq += tl.dot(b_do_r, b_h.to(b_do_r.dtype))

    b_dq = b_dq * (tl.exp2(b_g_r)[:, None] * scale) + b_dq_intra
    b_q_r = tl.load(q_base + o_r[:, None] * (H * K) + o_k[None, :], mask=m_r[:, None] & m_k[None, :], other=0.0)
    force = tl.sum(b_dq * b_q_r.to(tl.float32), axis=1)
    p_dq = dq + (bos * H + i_h) * K + o_r[:, None] * (H * K) + o_k[None, :]
    tl.store(p_dq, b_dq.to(dq.dtype.element_ty), mask=m_r[:, None] & m_k[None, :])

    # Half 2: dk lifecycle
    b_ds_k = tl.zeros((BR, BT), dtype=tl.float32)
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_v_r = tl.load(v_base + o_r[:, None] * (H * V) + o_v[None, :], mask=m_r[:, None] & m_v[None, :], other=0.0)
        b_do_s = tl.load(do_base + o_s[:, None] * (H * V) + o_v[None, :], mask=m_s[:, None] & m_v[None, :], other=0.0)
        if DOT_IEEE:
            b_ds_k += tl.dot(b_v_r.to(tl.float32), tl.trans(b_do_s.to(tl.float32)), input_precision='ieee')
        else:
            b_ds_k += tl.dot(b_v_r, tl.trans(b_do_s))

    m_A_k = (o_r[:, None] <= o_s[None, :]) & m_r[:, None] & m_s[None, :]
    b_ds_k = tl.where(m_A_k, b_ds_k * tl.exp2(b_g_s[None, :] - b_g_r[:, None]) * scale, 0.0)
    b_q_s = tl.load(q_base + o_s[:, None] * (H * K) + o_k[None, :], mask=m_s[:, None] & m_k[None, :], other=0.0)
    if DOT_IEEE:
        b_dk_intra = tl.dot(b_ds_k, b_q_s.to(tl.float32), input_precision='ieee')
    else:
        b_dk_intra = tl.dot(b_ds_k.to(b_q_s.dtype), b_q_s)

    b_dk = tl.zeros((BR, BK), dtype=tl.float32)
    b_dg_last = tl.zeros((1,), dtype=tl.float32)
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_v_r = tl.load(v_base + o_r[:, None] * (H * V) + o_v[None, :], mask=m_r[:, None] & m_v[None, :], other=0.0)
        b_dh = tl.load(dh_base + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0)
        if i_r == 0:
            b_h = tl.load(h_base + o_v[:, None] * K + o_k[None, :], mask=m_v[:, None] & m_k[None, :], other=0.0)
            b_dg_last += tl.sum(b_h.to(tl.float32) * tl.trans(b_dh.to(tl.float32)))
        if DOT_IEEE:
            b_dk += tl.dot(b_v_r.to(tl.float32), tl.trans(b_dh.to(tl.float32)), input_precision='ieee')
        else:
            b_dk += tl.dot(b_v_r, tl.trans(b_dh.to(b_v_r.dtype)))

    b_dg_last *= tl.exp2(b_g_last)
    b_dk *= tl.where(m_r, tl.exp2(-b_g_r + b_g_last), 0.0)[:, None]
    b_k_r = tl.load(k_base + o_r[:, None] * (H * K) + o_k[None, :], mask=m_r[:, None] & m_k[None, :], other=0.0)
    b_dg_last += tl.sum(b_dk * b_k_r.to(tl.float32))
    b_dk += b_dk_intra

    force -= tl.sum(b_dk * b_k_r.to(tl.float32), axis=1)
    n_b = tl.num_programs(2).to(tl.int64) // H
    tl.store(dg_part + i_k * (n_b * T * H) + (bos + o_r) * H + i_h, force, mask=m_r)
    tl.store(dg_last_part + ((i_k * n_b + i_b) * (nt * n_r) + i_tr) * H + i_h + tl.arange(0, 1), b_dg_last)
    p_dk = dk + (bos * H + i_h) * K + o_r[:, None] * (H * K) + o_k[None, :]
    tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_r[:, None] & m_k[None, :])


@triton.jit(do_not_specialize=['T'])
def _cyfa_chunk_bwd_dv_rsi_kernel(
    q, k, g, do, dh, dv,
    scale, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_tr = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    n_r = BT // BR
    i_t = i_tr // n_r
    i_r = i_tr % n_r
    i_b = i_bh // H
    i_h = i_bh % H
    nt = tl.cdiv(T, BT)
    i_tg = i_b * nt + i_t
    bos = i_b * T

    o_r = i_t * BT + i_r * BR + tl.arange(0, BR)
    o_s = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_r = o_r < T
    m_s = o_s < T
    m_v = o_v < V

    b_g_r = tl.load(g + (bos + o_r) * H + i_h, mask=m_r, other=0.0)
    b_g_s = tl.load(g + (bos + o_s) * H + i_h, mask=m_s, other=0.0)
    last_idx = tl.minimum(i_t * BT + BT, T) - 1
    b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

    q_base = q + (bos * H + i_h) * K
    k_base = k + (bos * H + i_h) * K
    dh_base = dh + (i_tg * H + i_h).to(tl.int64) * K * V

    b_A = tl.zeros((BR, BT), dtype=tl.float32)
    for i_k in range(0, tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_k = tl.load(k_base + o_r[:, None] * (H * K) + o_k[None, :], mask=m_r[:, None] & m_k[None, :], other=0.0)
        b_q = tl.load(q_base + o_s[:, None] * (H * K) + o_k[None, :], mask=m_s[:, None] & m_k[None, :], other=0.0)
        if DOT_IEEE:
            b_A += tl.dot(b_k.to(tl.float32), tl.trans(b_q.to(tl.float32)), input_precision='ieee')
        else:
            b_A += tl.dot(b_k, tl.trans(b_q))

    m_A = (o_r[:, None] <= o_s[None, :]) & m_r[:, None] & m_s[None, :]
    b_A = tl.where(m_A, b_A * tl.exp2(b_g_s[None, :] - b_g_r[:, None]) * scale, 0.0)
    b_do = tl.load(do + (bos * H + i_h) * V + o_s[:, None] * (H * V) + o_v[None, :], mask=m_s[:, None] & m_v[None, :], other=0.0)
    if DOT_IEEE:
        b_dv_intra = tl.dot(b_A, b_do.to(tl.float32), input_precision='ieee')
    else:
        b_dv_intra = tl.dot(b_A.to(b_do.dtype), b_do)

    b_dv = tl.zeros((BR, BV), dtype=tl.float32)
    for i_k in range(0, tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_k = tl.load(k_base + o_r[:, None] * (H * K) + o_k[None, :], mask=m_r[:, None] & m_k[None, :], other=0.0)
        b_dh = tl.load(dh_base + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0)
        if DOT_IEEE:
            b_dv += tl.dot(b_k.to(tl.float32), b_dh.to(tl.float32), input_precision='ieee')
        else:
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    b_dv = b_dv * tl.where(m_r, tl.exp2(-b_g_r + b_g_last), 0.0)[:, None] + b_dv_intra
    p_dv = dv + (bos * H + i_h) * V + o_r[:, None] * (H * V) + o_v[None, :]
    tl.store(p_dv, b_dv.to(dv.dtype.element_ty), mask=m_r[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['T'])
def _cyfa_bwd_dg_reduce_rev_cumsum_rsi_kernel(
    dg_part_v, dg_last_part_v,
    dg_part_k, dg_last_part_k,
    dg_out,
    T,
    H: tl.constexpr, BT: tl.constexpr, NR: tl.constexpr,
    NK_V: tl.constexpr, NK_K: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    n_b = tl.num_programs(1).to(tl.int64) // H
    nt = tl.cdiv(T, BT)
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    last_idx = tl.minimum(i_t * BT + BT, T) - 1
    stride_th = n_b * T * H

    force = tl.zeros((BT,), dtype=tl.float32)
    last_total = tl.zeros((1,), dtype=tl.float32)

    for i_kv in range(0, NK_V):
        force += tl.load(dg_part_v + i_kv * stride_th + (bos + o_t) * H + i_h, mask=m_t, other=0.0)
        for i_r in range(0, NR):
            i_tr = i_t * NR + i_r
            last_total += tl.load(dg_last_part_v + ((i_kv * n_b + i_b) * (nt * NR) + i_tr) * H + i_h + tl.arange(0, 1))

    for i_kk in range(0, NK_K):
        force += tl.load(dg_part_k + i_kk * stride_th + (bos + o_t) * H + i_h, mask=m_t, other=0.0)
        for i_r in range(0, NR):
            i_tr = i_t * NR + i_r
            last_total += tl.load(dg_last_part_k + ((i_kk * n_b + i_b) * (nt * NR) + i_tr) * H + i_h + tl.arange(0, 1))

    force += tl.where(o_t == last_idx, last_total, 0.0)
    force = tl.where(m_t, force, 0.0)
    chunk_sum = tl.sum(force, axis=0)
    dg = tl.where(m_t, chunk_sum - tl.cumsum(force, axis=0) + force, 0.0)
    tl.store(dg_out + (bos + o_t) * H + i_h, dg.to(dg_out.dtype.element_ty), mask=m_t)


# ============================================================================
# 6. Helper Wrappers & Autograd Function `ChunkCyFARSI`
# ============================================================================

def _cyfa_scalar_decay_fwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    save_h: bool,
    mutate_mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    bsz, seq_len, n_heads, d_k = q.shape
    d_v = v.shape[-1]
    bt = 64
    nt = triton.cdiv(seq_len, bt)
    ieee = q.dtype == torch.float32
    bk_h = 32 if ieee else min(64, triton.next_power_of_2(d_k))
    bv_h = 32
    br_o = 32 if ieee else 64
    bk_o = 32 if ieee else min(64, triton.next_power_of_2(d_k))
    bv_o = 32 if ieee else min(64, triton.next_power_of_2(d_v))
    nw_o = 8
    ns_h = 1 if ieee else 3
    ns_o = 1 if ieee else 2

    h = torch.empty((bsz, nt, n_heads, d_v, d_k), device=q.device, dtype=q.dtype)
    ht = torch.empty((bsz, n_heads, d_k, d_v), device=q.device, dtype=torch.float32) if output_final_state else None

    _cyfa_chunk_fwd_h_rsi_kernel[(triton.cdiv(d_v, bv_h), triton.cdiv(d_k, bk_h), bsz * n_heads)](
        k, v, g_cumsum, h, initial_state, ht,
        T=seq_len, H=n_heads, K=d_k, V=d_v, BT=bt, BK=bk_h, BV=bv_h,
        USE_INITIAL_STATE=initial_state is not None,
        OUTPUT_FINAL_STATE=output_final_state,
        DOT_IEEE=ieee,
        num_warps=4, num_stages=ns_h,
    )

    out = torch.empty((bsz, seq_len, n_heads, d_v), device=q.device, dtype=q.dtype)
    _cyfa_chunk_fwd_o_rsi_kernel[(triton.cdiv(d_v, bv_o), nt * (bt // br_o), bsz * n_heads)](
        q, k, v, h, g_cumsum, out,
        float(scale), T=seq_len, H=n_heads, K=d_k, V=d_v,
        BT=bt, BR=br_o, BK=bk_o, BV=bv_o, DOT_IEEE=ieee,
        num_warps=nw_o, num_stages=ns_o,
    )
    return out, (h if save_h else None), ht


def _cyfa_scalar_decay_bwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    h: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    bsz, seq_len, n_heads, d_k = q.shape
    d_v = v.shape[-1]
    bt = 64
    br = 32
    n_r = bt // br
    nt = triton.cdiv(seq_len, bt)
    ieee = q.dtype == torch.float32
    bk_dh = 32
    bv_dh = 32 if ieee else min(64, triton.next_power_of_2(d_v))
    bk = 32 if ieee else min(64, triton.next_power_of_2(d_k))
    bv = 32
    br_dv = 32 if ieee else 64
    bk_dv = 32 if ieee else min(64, triton.next_power_of_2(d_k))
    bv_dv = 32 if ieee else min(64, triton.next_power_of_2(d_v))
    nw_bwd = 8 if ieee else 4
    nw_dv = 8
    ns_dh = 1 if ieee else 3
    ns_dv = 1 if ieee else 2
    nk_dh = triton.cdiv(d_k, bk_dh)
    nv_dh = triton.cdiv(d_v, bv_dh)
    nk = triton.cdiv(d_k, bk)
    nv = triton.cdiv(d_v, bv)
    nv_dv = triton.cdiv(d_v, bv_dv)

    if h is None:
        h = torch.empty((bsz, nt, n_heads, d_v, d_k), device=q.device, dtype=q.dtype)
        _cyfa_chunk_fwd_h_rsi_kernel[(nv, nk, bsz * n_heads)](
            k, v, g_cumsum, h, initial_state, None,
            T=seq_len, H=n_heads, K=d_k, V=d_v, BT=bt, BK=bk, BV=bv,
            USE_INITIAL_STATE=initial_state is not None,
            OUTPUT_FINAL_STATE=False,
            DOT_IEEE=ieee,
            num_warps=4, num_stages=ns_dh,
        )

    dh = torch.empty((bsz, nt, n_heads, d_k, d_v), device=q.device, dtype=q.dtype)
    dh0 = torch.empty((bsz, n_heads, d_k, d_v), device=q.device, dtype=torch.float32) if initial_state is not None else None
    dht_c = dht.contiguous() if dht is not None else None
    _cyfa_chunk_bwd_dh_rsi_kernel[(nk_dh, nv_dh, bsz * n_heads)](
        q, g_cumsum, do, dh, dht_c, dh0,
        float(scale), T=seq_len, H=n_heads, K=d_k, V=d_v, BT=bt, BK=bk_dh, BV=bv_dh,
        USE_FINAL_STATE_GRAD=dht is not None,
        USE_INITIAL_STATE=initial_state is not None,
        DOT_IEEE=ieee,
        num_warps=4, num_stages=ns_dh,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg_part = torch.empty((nk, bsz, seq_len, n_heads), device=q.device, dtype=torch.float32)
    dg_last_part = torch.empty((nk, bsz, nt * n_r, n_heads), device=q.device, dtype=torch.float32)
    _cyfa_chunk_bwd_dqk_rsi_kernel[(nk, nt * n_r, bsz * n_heads)](
        q, k, v, g_cumsum, h, do, dh, dq, dk, dg_part, dg_last_part,
        float(scale), T=seq_len, H=n_heads, K=d_k, V=d_v,
        BT=bt, BR=br, BK=bk, BV=bv, DOT_IEEE=ieee,
        num_warps=nw_bwd, num_stages=1,
    )

    dv = torch.empty_like(v)
    _cyfa_chunk_bwd_dv_rsi_kernel[(nv_dv, nt * (bt // br_dv), bsz * n_heads)](
        q, k, g_cumsum, do, dh, dv,
        float(scale), T=seq_len, H=n_heads, K=d_k, V=d_v,
        BT=bt, BR=br_dv, BK=bk_dv, BV=bv_dv, DOT_IEEE=ieee,
        num_warps=nw_dv, num_stages=ns_dv,
    )
    return dq, dk, dv, dg_part, dg_last_part, dh0


class ChunkCyFARSIFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        delta: torch.Tensor,
        beta: torch.Tensor,
        readout_table: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        q_norm_eps: float,
        k_norm_eps: float,
        scale: float,
        initial_k: torch.Tensor | None,
        initial_v: torch.Tensor | None,
        initial_lambda: torch.Tensor | None,
        output_final_state: bool,
        checkpoint_level: int,
        mutate_mode: int,
    ):
        bsz, seq_len, n_heads, d_k = q.shape
        num_slots = readout_table.shape[-1]
        readout_slots = readout_table.shape[-2]
        ieee = q.dtype == torch.float32

        # 1. Fused Q/K RMSNorm storing both q_norm and k_norm
        q_norm, k_norm, q_rstd, k_rstd = cyfa_qk_rmsnorm_fwd_rsi(
            q, k, q_norm_weight, k_norm_weight, q_norm_eps, k_norm_eps
        )

        # 2. Fused BT=256 (4x64) chunk-local decay cumsum + global clock cumsum
        g_cumsum = torch.empty((bsz, seq_len, n_heads), device=q.device, dtype=torch.float32)
        lambdas = torch.empty((bsz, seq_len, n_heads), device=q.device, dtype=torch.float32)
        final_lambda = torch.empty((bsz, n_heads), device=q.device, dtype=torch.float32) if output_final_state else None

        delta_eff = torch.zeros_like(delta) if mutate_mode == 1 else delta
        _cyfa_clock_and_decay_cumsum_fwd_rsi_kernel[(bsz * n_heads,)](
            g.contiguous(), delta_eff.contiguous(),
            initial_lambda.contiguous() if initial_lambda is not None else None,
            g_cumsum, lambdas, final_lambda,
            T=seq_len, H=n_heads, BT=256,
            HAS_INIT_LAMBDA=initial_lambda is not None,
            HAS_FINAL_LAMBDA=output_final_state,
            num_warps=4, num_stages=1,
        )

        # 3. Cardinal Fourier orbit write vector
        write = torch.empty((bsz, seq_len, n_heads, num_slots), device=q.device, dtype=q.dtype)
        _cyfa_write_fwd_rsi_kernel[(triton.cdiv(seq_len, 32), bsz * n_heads)](
            lambdas, beta.contiguous(), write,
            T=seq_len, H=n_heads, M=num_slots, BT=32, BM=triton.next_power_of_2(num_slots),
            num_warps=4, num_stages=1,
        )

        # 4. Pass-1 Key-Memory Readout (`raw_logits`) + Strip-Tiled Cyclic Modal Readout (`weights`)
        save_states = checkpoint_level == 0
        raw_logits, h_k, hkt = _cyfa_scalar_decay_fwd_rsi(
            q=q_norm, k=k_norm, v=write, g_cumsum=g_cumsum, scale=scale,
            initial_state=initial_k, output_final_state=output_final_state,
            save_h=save_states,
        )

        weights = torch.empty_like(write)
        n_tokens = bsz * seq_len
        bt_ro = 16 if ieee else 32
        bc_fwd = 32 if ieee else min(64, triton.next_power_of_2(num_slots))
        nw_ro = 4 if ieee else 8
        table_c = readout_table.contiguous()
        lam_readout = torch.zeros_like(lambdas) if mutate_mode == 2 else lambdas
        _cyfa_modal_readout_fwd_rsi_kernel[(triton.cdiv(n_tokens, bt_ro), n_heads)](
            raw_logits, lam_readout, table_c, weights,
            N=n_tokens, H=n_heads, M=num_slots, C=readout_slots,
            BT=bt_ro, BM=triton.next_power_of_2(num_slots), BC=bc_fwd,
            DOT_IEEE=ieee, num_warps=nw_ro, num_stages=1,
        )

        # 5. Pass-2 Value-Memory Readout (`out`)
        out, h_v, hvt = _cyfa_scalar_decay_fwd_rsi(
            q=weights, k=write, v=v.contiguous(), g_cumsum=g_cumsum, scale=1.0,
            initial_state=initial_v, output_final_state=output_final_state,
            save_h=save_states,
        )

        ctx.save_for_backward(
            q, k, v, q_norm, k_norm, q_norm_weight, k_norm_weight,
            q_rstd, k_rstd, delta, beta, g_cumsum, lambdas,
            readout_table, raw_logits if save_states else None,
            weights, write if save_states else None,
            h_k, h_v, initial_k, initial_v, initial_lambda,
        )
        ctx.q_norm_eps = float(q_norm_eps)
        ctx.k_norm_eps = float(k_norm_eps)
        ctx.scale = float(scale)
        ctx.g_dtype = g.dtype
        ctx.mutate_mode = int(mutate_mode)
        return out, hkt, hvt, final_lambda

    @staticmethod
    def backward(ctx, do, dhkt=None, dhvt=None, dlt=None):
        (
            q_raw, k_raw, v, q_norm, k_norm, q_norm_weight, k_norm_weight,
            q_rstd, k_rstd, delta, beta, g_cumsum, lambdas,
            readout_table, raw_logits, weights, write,
            h_k, h_v, initial_k, initial_v, initial_lambda,
        ) = ctx.saved_tensors

        bsz, seq_len, n_heads, d_k = q_raw.shape
        num_slots = readout_table.shape[-1]
        readout_slots = readout_table.shape[-2]
        ieee = q_raw.dtype == torch.float32
        scale = ctx.scale

        if write is None:
            write = torch.empty((bsz, seq_len, n_heads, num_slots), device=q_raw.device, dtype=q_raw.dtype)
            _cyfa_write_fwd_rsi_kernel[(triton.cdiv(seq_len, 16), bsz * n_heads)](
                lambdas, beta.contiguous(), write,
                T=seq_len, H=n_heads, M=num_slots, BT=16, BM=triton.next_power_of_2(num_slots),
                num_warps=4, num_stages=1,
            )

        # 1. Pass-2 Value-Memory Backward
        dweights, dwrite_v, dv, dg_part_v, dg_last_part_v, dhv0 = _cyfa_scalar_decay_bwd_rsi(
            q=weights, k=write, v=v.contiguous(), g_cumsum=g_cumsum,
            h=h_v, initial_state=initial_v, do=do.contiguous(), dht=dhvt, scale=1.0,
        )

        if raw_logits is None:
            raw_logits, h_k, _ = _cyfa_scalar_decay_fwd_rsi(
                q=q_norm, k=k_norm, v=write, g_cumsum=g_cumsum, scale=scale,
                initial_state=initial_k, output_final_state=False, save_h=True,
            )

        # 2. Strip-Tiled 2-Pass Readout Backward
        n_tokens = bsz * seq_len
        bm = triton.next_power_of_2(num_slots)
        probs = torch.empty((bsz, seq_len, n_heads, readout_slots), device=q_raw.device, dtype=q_raw.dtype)
        dlogits = torch.empty_like(probs)
        draw_logits = torch.empty_like(raw_logits)
        bt_p1 = 16 if ieee else 32
        bc_p1 = 32 if ieee else min(64, bm)
        nw_p1 = 4 if ieee else 8
        bt_p2 = 16 if ieee else 32
        br_p2 = 32 if ieee else min(64, bm)
        bc_p2 = 32 if ieee else min(64, bm)
        nc_readout = triton.cdiv(num_slots, bc_p2)
        dlambda_readout_part = torch.empty((nc_readout, bsz, seq_len, n_heads), device=q_raw.device, dtype=torch.float32)
        table_work = readout_table.contiguous()

        _cyfa_readout_bwd_pass1_rsi_kernel[(triton.cdiv(n_tokens, bt_p1), n_heads)](
            lambdas, table_work, raw_logits.contiguous(), dweights.contiguous(),
            probs, dlogits,
            N=n_tokens, H=n_heads, M=num_slots, C=readout_slots,
            BT=bt_p1, BM=bm, BC=bc_p1, DOT_IEEE=ieee,
            num_warps=nw_p1, num_stages=1,
        )
        _cyfa_readout_bwd_pass2_rsi_kernel[(triton.cdiv(n_tokens, bt_p2), nc_readout, n_heads)](
            lambdas, table_work, raw_logits.contiguous(), dweights.contiguous(),
            probs, dlogits, draw_logits, dlambda_readout_part,
            N=n_tokens, H=n_heads, M=num_slots, C=readout_slots,
            BT=bt_p2, BR=br_p2, BC=bc_p2, DOT_IEEE=ieee,
            num_warps=4, num_stages=1,
        )

        # 3. Readout Table Gradient
        if ctx.needs_input_grad[6]:
            bn_tg = 32 if ieee else 64
            bc_tg = 32 if ieee else min(64, bm)
            nw_tg = 4 if ieee else 8
            n_splits = min(32, triton.cdiv(n_tokens, bn_tg))
            partials = torch.empty((n_splits, n_heads, readout_slots, num_slots), device=q_raw.device, dtype=torch.float32)
            dreadout_table = torch.empty_like(readout_table, dtype=torch.float32)
            _cyfa_readout_table_grad_rsi_kernel[(triton.cdiv(readout_slots, 32), triton.cdiv(num_slots, bc_tg), n_heads * n_splits)](
                lambdas, probs, dweights, dlogits, raw_logits, partials,
                T=n_tokens, H=n_heads, M=num_slots, C=readout_slots,
                BN=bn_tg, BR=32, BC=bc_tg, DOT_IEEE=ieee,
                num_warps=nw_tg, num_stages=1,
            )
            _cyfa_readout_table_grad_reduce_rsi_kernel[(triton.cdiv(dreadout_table.numel(), 128),)](
                partials, dreadout_table, S=n_splits, SIZE=dreadout_table.numel(), BLOCK=128,
                num_warps=4, num_stages=1,
            )
        else:
            dreadout_table = None

        # 4. Pass-1 Key-Memory Backward
        dq_norm, dk_norm, dwrite_k, dg_part_k, dg_last_part_k, dhk0 = _cyfa_scalar_decay_bwd_rsi(
            q=q_norm, k=k_norm, v=write, g_cumsum=g_cumsum,
            h=h_k, initial_state=initial_k, do=draw_logits, dht=dhkt, scale=scale,
        )
        if ctx.mutate_mode == 3:
            dg_part_k.zero_()
            dg_last_part_k.zero_()

        # 5. Fused Write Backward + Readout Clock Force Reduction + Reverse Clock Scan
        phase_force = torch.empty((bsz, seq_len, n_heads), device=q_raw.device, dtype=torch.float32)
        dbeta = torch.empty_like(beta)
        _cyfa_write_bwd_rsi_kernel[(triton.cdiv(seq_len, 16), bsz * n_heads)](
            dwrite_k.contiguous(), dwrite_v.contiguous(), lambdas, beta.contiguous(), dlambda_readout_part,
            phase_force, dbeta,
            T=seq_len, H=n_heads, M=num_slots, BT=16, BM=bm, NC_READOUT=nc_readout,
            num_warps=4, num_stages=1,
        )

        ddelta = torch.empty_like(delta)
        dlambda0 = torch.empty((bsz, n_heads), device=q_raw.device, dtype=torch.float32) if initial_lambda is not None else None
        _cyfa_clock_rev_cumsum_bwd_rsi_kernel[(bsz * n_heads,)](
            phase_force, dlt.contiguous() if dlt is not None else None,
            ddelta, dlambda0,
            T=seq_len, H=n_heads, BT=256,
            HAS_DLT=dlt is not None,
            HAS_DLAMBDA0=initial_lambda is not None,
            num_warps=4, num_stages=1,
        )

        # 6. Fused Dual `dg` Reduction + Reverse Chunk-Local Cumsum
        dg = torch.empty((bsz, seq_len, n_heads), device=q_raw.device, dtype=ctx.g_dtype)
        nt = triton.cdiv(seq_len, 64)
        _cyfa_bwd_dg_reduce_rev_cumsum_rsi_kernel[(nt, bsz * n_heads)](
            dg_part_v, dg_last_part_v, dg_part_k, dg_last_part_k, dg,
            T=seq_len, H=n_heads, BT=64, NR=2,
            NK_V=dg_part_v.shape[0], NK_K=dg_part_k.shape[0],
            num_warps=4, num_stages=1,
        )

        # 7. Zero-Spill Q/K RMSNorm Backward
        dq, dk, dq_norm_w, dk_norm_w = cyfa_qk_rmsnorm_bwd_rsi(
            dq_norm, dk_norm, q_raw, k_raw, q_norm_weight, k_norm_weight, q_rstd, k_rstd
        )

        return (
            dq, dk, dv, dg, ddelta, dbeta,
            None if dreadout_table is None else dreadout_table.to(readout_table.dtype),
            dq_norm_w, dk_norm_w,
            None, None, None,
            dhk0, dhv0, dlambda0,
            None, None, None,
        )


def chunk_cyfa_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    delta: torch.Tensor,
    beta: torch.Tensor,
    readout: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_eps: float = 1e-5,
    k_norm_eps: float = 1e-5,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    checkpoint_level: int = 0,
    precomputed_readout_table: torch.Tensor | None = None,
    mutate_mode: int = 0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    bsz, _, n_heads, d_k = q.shape
    d_v = v.shape[-1]
    if scale is None:
        scale = d_k ** -0.5
    if precomputed_readout_table is not None:
        readout_table = precomputed_readout_table
    else:
        readout_table = build_readout_table(readout)
    num_slots = readout_table.shape[-1]

    if initial_state is None:
        initial_k = initial_v = initial_lambda = None
    else:
        k0, v0, l0 = initial_state
        initial_k = k0.to(device=q.device, dtype=torch.float32).contiguous()
        initial_v = v0.to(device=q.device, dtype=torch.float32).contiguous()
        initial_lambda = l0.to(device=q.device, dtype=torch.float32).contiguous()

    eff_ckpt = 0 if (
        checkpoint_level == 0
        and torch.is_grad_enabled()
        and any(t.requires_grad for t in (q, k, v, g, delta, beta, readout_table))
    ) else 1

    out, fk, fv, fl = ChunkCyFARSIFunction.apply(
        q, k, v, g, delta, beta, readout_table,
        q_norm_weight, k_norm_weight, float(q_norm_eps), float(k_norm_eps), float(scale),
        initial_k, initial_v, initial_lambda, bool(output_final_state), eff_ckpt, int(mutate_mode),
    )
    final_state = (fk, fv, fl) if output_final_state else None
    return out, final_state


# ============================================================================
# 7. Superfused `CyclicFlowAttentionBlockRSI` (`Stages A0 -> A4`)
# ============================================================================

@triton.jit(do_not_specialize=['T'])
def _cyfa_gate_prep_fwd_rsi_kernel(
    ab_in, d_in, dt_bias, A_log,
    g_out, beta_out, delta_out,
    T,
    H: tl.constexpr, BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_dt_bias = tl.load(dt_bias + i_h).to(tl.float32)
    b_A_exp = tl.exp(tl.load(A_log + i_h).to(tl.float32))

    base_t = i_b * T + o_t
    b_a = tl.load(ab_in + base_t * (2 * H) + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_b = tl.load(ab_in + base_t * (2 * H) + H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_d = tl.load(d_in + base_t * H + i_h, mask=m_t, other=0.0).to(tl.float32)

    x_sp = b_a + b_dt_bias
    sp = tl.where(x_sp > 20.0, x_sp, tl.log(1.0 + tl.exp(x_sp)))
    b_g = -b_A_exp * sp
    b_beta = tl.sigmoid(b_b)
    b_delta = tl.sigmoid(b_d)

    out_idx = base_t * H + i_h
    tl.store(g_out + out_idx, b_g, mask=m_t)
    tl.store(beta_out + out_idx, b_beta, mask=m_t)
    tl.store(delta_out + out_idx, b_delta, mask=m_t)


@triton.jit(do_not_specialize=['T'])
def _cyfa_gate_prep_bwd_rsi_kernel(
    ab_in, d_in, dt_bias, A_log,
    dg_in, dbeta_in, ddelta_in,
    dab_out, dd_out, d_dt_bias_part, d_A_log_part,
    T,
    H: tl.constexpr, BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_dt_bias = tl.load(dt_bias + i_h).to(tl.float32)
    b_A_exp = tl.exp(tl.load(A_log + i_h).to(tl.float32))

    base_t = i_b * T + o_t
    idx = base_t * H + i_h
    b_a = tl.load(ab_in + base_t * (2 * H) + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_b = tl.load(ab_in + base_t * (2 * H) + H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_d = tl.load(d_in + idx, mask=m_t, other=0.0).to(tl.float32)

    b_dg = tl.load(dg_in + idx, mask=m_t, other=0.0).to(tl.float32)
    b_dbeta = tl.load(dbeta_in + idx, mask=m_t, other=0.0).to(tl.float32)
    b_ddelta = tl.load(ddelta_in + idx, mask=m_t, other=0.0).to(tl.float32)

    x_sp = b_a + b_dt_bias
    sp = tl.where(x_sp > 20.0, x_sp, tl.log(1.0 + tl.exp(x_sp)))
    sig_sp = tl.sigmoid(x_sp)
    b_g = -b_A_exp * sp

    da = tl.where(m_t, -b_A_exp * sig_sp * b_dg, 0.0)
    dA = tl.where(m_t, b_g * b_dg, 0.0)

    sig_b = tl.sigmoid(b_b)
    db = tl.where(m_t, b_dbeta * sig_b * (1.0 - sig_b), 0.0)

    sig_d = tl.sigmoid(b_d)
    dd = tl.where(m_t, b_ddelta * sig_d * (1.0 - sig_d), 0.0)

    tl.store(dab_out + base_t * (2 * H) + i_h, da.to(dab_out.dtype.element_ty), mask=m_t)
    tl.store(dab_out + base_t * (2 * H) + H + i_h, db.to(dab_out.dtype.element_ty), mask=m_t)
    tl.store(dd_out + idx, dd.to(dd_out.dtype.element_ty), mask=m_t)

    n_chunks = tl.num_programs(0).to(tl.int64)
    part_idx = (i_bh * n_chunks + i_t)
    tl.store(d_dt_bias_part + part_idx, tl.sum(da, 0))
    tl.store(d_A_log_part + part_idx, tl.sum(dA, 0))


class _CyFAGatePrepFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ab: torch.Tensor, d: torch.Tensor, dt_bias: torch.Tensor, A_log: torch.Tensor):
        bsz, seq_len, two_h = ab.shape
        n_heads = two_h // 2
        ab_c = ab.contiguous()
        d_c = d.contiguous()
        g_out = torch.empty((bsz, seq_len, n_heads), device=ab.device, dtype=torch.float32)
        beta_out = torch.empty((bsz, seq_len, n_heads), device=ab.device, dtype=torch.float32)
        delta_out = torch.empty((bsz, seq_len, n_heads), device=ab.device, dtype=torch.float32)
        nt = triton.cdiv(seq_len, 64)
        _cyfa_gate_prep_fwd_rsi_kernel[(nt, bsz * n_heads)](
            ab_c, d_c, dt_bias, A_log, g_out, beta_out, delta_out,
            T=seq_len, H=n_heads, BT=64, num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(ab_c, d_c, dt_bias, A_log)
        return g_out, beta_out, delta_out

    @staticmethod
    def backward(ctx, dg, dbeta, ddelta):
        ab_c, d_c, dt_bias, A_log = ctx.saved_tensors
        bsz, seq_len, two_h = ab_c.shape
        n_heads = two_h // 2
        nt = triton.cdiv(seq_len, 64)
        dab = torch.empty_like(ab_c)
        dd = torch.empty_like(d_c)
        d_dt_part = torch.empty((bsz, n_heads, nt), device=ab_c.device, dtype=torch.float32)
        d_A_part = torch.empty((bsz, n_heads, nt), device=ab_c.device, dtype=torch.float32)
        _cyfa_gate_prep_bwd_rsi_kernel[(nt, bsz * n_heads)](
            ab_c, d_c, dt_bias, A_log,
            dg.contiguous(), dbeta.contiguous(), ddelta.contiguous(),
            dab, dd, d_dt_part, d_A_part,
            T=seq_len, H=n_heads, BT=64, num_warps=4, num_stages=1,
        )
        d_dt_bias = d_dt_part.sum(dim=(0, 2)).to(dt_bias.dtype)
        d_A_log = d_A_part.sum(dim=(0, 2)).to(A_log.dtype)
        return dab, dd, d_dt_bias, d_A_log


def cyfa_gate_prep_rsi(ab: torch.Tensor, d: torch.Tensor, dt_bias: torch.Tensor, A_log: torch.Tensor):
    return _CyFAGatePrepFn.apply(ab, d, dt_bias, A_log)


class CyclicFlowAttentionBlockRSI(nn.Module):
    """
    5-Stage Progressive RSI Optimization of `CyclicFlowAttention` (`arXiv:2609.36259`):
      - `stage="A0"`: Upstream `CyclicFlowAttention` (`fla.ops.cyfa.chunk.chunk_cyfa` + unfused periphery)
      - `stage="A1"`: Zero-spill `chunk_cyfa_rsi` recurrence + strip-tiled modal readout (`0` spills across all 14 kernels)
      - `stage="A2"`: `A1` + Fused QKV `ShortConvolution` (`1` conv call instead of `3`)
      - `stage="A3"`: `A2` + Fused `(a, b)` gate projection & Triton `_cyfa_gate_prep_fwd_bwd_rsi_kernel`
      - `stage="A4"`: `A3` + Fused post-scale `o_norm` (`scale = num_readout_slots` folded into Pass-2 `chunk_fwd_o` so `o * num_readout_slots` allocation is eliminated)
    """

    def __init__(self, ref_layer: CyclicFlowAttention, stage: str = "A4"):
        super().__init__()
        assert stage in ("A0", "A1", "A2", "A3", "A4")
        self.ref = ref_layer
        self.stage = stage
        self.hidden_size = ref_layer.hidden_size
        self.num_heads = ref_layer.num_heads
        self.head_k_dim = ref_layer.head_k_dim
        self.head_v_dim = ref_layer.head_v_dim
        self.key_dim = ref_layer.key_dim
        self.value_dim = ref_layer.value_dim
        self.num_slots = ref_layer.num_slots
        self.num_readout_slots = ref_layer.num_readout_slots
        self.use_short_conv = ref_layer.use_short_conv

        if self.use_short_conv:
            total_qkv = 2 * self.key_dim + self.value_dim
            self.qkv_conv1d = ShortConvolution(
                hidden_size=total_qkv,
                kernel_size=ref_layer.conv_size,
                bias=ref_layer.conv_bias,
                activation=None,
            ).to(device=ref_layer.q_proj.weight.device, dtype=ref_layer.q_proj.weight.dtype)
            with torch.no_grad():
                w_cat = torch.cat(
                    [ref_layer.q_conv1d.weight, ref_layer.k_conv1d.weight, ref_layer.v_conv1d.weight], dim=0
                )
                self.qkv_conv1d.weight.copy_(w_cat)
                if ref_layer.conv_bias:
                    b_cat = torch.cat(
                        [ref_layer.q_conv1d.bias, ref_layer.k_conv1d.bias, ref_layer.v_conv1d.bias], dim=0
                    )
                    self.qkv_conv1d.bias.copy_(b_cat)

    def _sync_qkv_conv_weights(self):
        if self.use_short_conv and hasattr(self, "qkv_conv1d"):
            with torch.no_grad():
                w_cat = torch.cat(
                    [self.ref.q_conv1d.weight, self.ref.k_conv1d.weight, self.ref.v_conv1d.weight], dim=0
                )
                self.qkv_conv1d.weight.copy_(w_cat)

    def _copy_qkv_conv_grads_back(self):
        if self.use_short_conv and hasattr(self, "qkv_conv1d") and self.qkv_conv1d.weight.grad is not None:
            gq, gk, gv = torch.split(
                self.qkv_conv1d.weight.grad, [self.key_dim, self.key_dim, self.value_dim], dim=0
            )
            self.ref.q_conv1d.weight.grad = gq.clone()
            self.ref.k_conv1d.weight.grad = gk.clone()
            self.ref.v_conv1d.weight.grad = gv.clone()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.stage == "A0":
            out, _, _ = self.ref(hidden_states)
            return out

        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(
            hidden_states,
            torch.cat((self.ref.q_proj.weight, self.ref.k_proj.weight, self.ref.v_proj.weight), dim=0),
        )
        if self.use_short_conv:
            if self.stage in ("A2", "A3", "A4"):
                qkv_conv, _ = self.qkv_conv1d(x=qkv, cache=None, output_final_state=False)
                q, k, v = torch.split(qkv_conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            else:
                q_p, k_p, v_p = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
                q, _ = self.ref.q_conv1d(x=q_p, cache=None, output_final_state=False)
                k, _ = self.ref.k_conv1d(x=k_p, cache=None, output_final_state=False)
                v, _ = self.ref.v_conv1d(x=v_p, cache=None, output_final_state=False)
        else:
            q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_k_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_k_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_v_dim)

        if self.stage in ("A3", "A4"):
            ab_w = torch.cat((self.ref.a_proj.weight, self.ref.b_proj.weight), dim=0)
            ab_bias = torch.cat(
                (torch.zeros_like(self.ref.b_proj.bias), self.ref.b_proj.bias), dim=0
            )
            ab = F.linear(hidden_states, ab_w, ab_bias)
            d_raw = self.ref.d_proj(hidden_states.detach())
            g, beta, delta = cyfa_gate_prep_rsi(ab, d_raw, self.ref.dt_bias, self.ref.A_log)
        else:
            a = self.ref.a_proj(hidden_states)
            beta = self.ref.b_proj(hidden_states).float().sigmoid()
            delta = self.ref.d_proj(hidden_states.detach()).float().sigmoid()
            decay = F.softplus(a.float() + self.ref.dt_bias.float())
            g = -self.ref.A_log.float().exp() * decay

        readout_table = build_readout_table(self.ref.readout)
        if self.stage == "A4":
            # Fold `num_readout_slots` scaling into `v` or `o_norm`?
            # Wait: `RMSNorm(o * num_readout_slots, eps)` has `rstd = 1 / sqrt(mean((o * S)^2) + eps)`.
            # Passing `o * self.num_readout_slots` vs folding `num_readout_slots` into Pass-2 `v` scale:
            # Since `o` is linear in Pass-2 `scale`, setting Pass-2 `scale = float(self.num_readout_slots)`
            # produces `o_scaled = o * self.num_readout_slots` directly inside `_cyfa_chunk_fwd_o_rsi_kernel`
            # with zero extra tensor allocation!
            o_scaled, _, _, _ = _ChunkCyFAStageA4Function.apply(
                q, k, v, g, delta, beta, readout_table,
                self.ref.q_norm.weight, self.ref.k_norm.weight,
                float(self.ref.q_norm.eps), float(self.ref.k_norm.eps),
                float(self.head_k_dim ** -0.5), float(self.num_readout_slots),
            )
            g_gate = self.ref.g_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_v_dim)
            o = self.ref.o_norm(o_scaled, g_gate)
        else:
            o, _ = chunk_cyfa_rsi(
                q=q, k=k, v=v, g=g, delta=delta, beta=beta,
                readout=self.ref.readout,
                q_norm_weight=self.ref.q_norm.weight,
                k_norm_weight=self.ref.k_norm.weight,
                q_norm_eps=self.ref.q_norm.eps,
                k_norm_eps=self.ref.k_norm.eps,
                scale=self.head_k_dim ** -0.5,
                precomputed_readout_table=readout_table,
            )
            g_gate = self.ref.g_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_v_dim)
            o = self.ref.o_norm(o * self.num_readout_slots, g_gate)

        o = o.reshape(bsz, seq_len, self.value_dim)
        return self.ref.o_proj(o)


class _ChunkCyFAStageA4Function(torch.autograd.Function):
    """Stage A4 wrapper folding `post_scale = num_readout_slots` directly into Pass-2 value readout."""

    @staticmethod
    def forward(
        ctx,
        q, k, v, g, delta, beta, readout_table,
        q_norm_weight, k_norm_weight, q_norm_eps, k_norm_eps, scale, post_scale,
    ):
        bsz, seq_len, n_heads, d_k = q.shape
        num_slots = readout_table.shape[-1]
        readout_slots = readout_table.shape[-2]
        ieee = q.dtype == torch.float32

        q_norm, k_norm, q_rstd, k_rstd = cyfa_qk_rmsnorm_fwd_rsi(
            q, k, q_norm_weight, k_norm_weight, q_norm_eps, k_norm_eps
        )
        g_cumsum = torch.empty((bsz, seq_len, n_heads), device=q.device, dtype=torch.float32)
        lambdas = torch.empty((bsz, seq_len, n_heads), device=q.device, dtype=torch.float32)
        _cyfa_clock_and_decay_cumsum_fwd_rsi_kernel[(bsz * n_heads,)](
            g.contiguous(), delta.contiguous(), None, g_cumsum, lambdas, None,
            T=seq_len, H=n_heads, BT=256, HAS_INIT_LAMBDA=False, HAS_FINAL_LAMBDA=False,
            num_warps=4, num_stages=1,
        )
        write = torch.empty((bsz, seq_len, n_heads, num_slots), device=q.device, dtype=q.dtype)
        _cyfa_write_fwd_rsi_kernel[(triton.cdiv(seq_len, 32), bsz * n_heads)](
            lambdas, beta.contiguous(), write,
            T=seq_len, H=n_heads, M=num_slots, BT=32, BM=triton.next_power_of_2(num_slots),
            num_warps=4, num_stages=1,
        )
        raw_logits, h_k, _ = _cyfa_scalar_decay_fwd_rsi(
            q=q_norm, k=k_norm, v=write, g_cumsum=g_cumsum, scale=scale,
            initial_state=None, output_final_state=False, save_h=True,
        )
        weights = torch.empty_like(write)
        n_tokens = bsz * seq_len
        bt_ro = 16 if ieee else 32
        bc_fwd = 32 if ieee else min(64, triton.next_power_of_2(num_slots))
        nw_ro = 4 if ieee else 8
        table_work = readout_table.contiguous()
        _cyfa_modal_readout_fwd_rsi_kernel[(triton.cdiv(n_tokens, bt_ro), n_heads)](
            raw_logits, lambdas, table_work, weights,
            N=n_tokens, H=n_heads, M=num_slots, C=readout_slots,
            BT=bt_ro, BM=triton.next_power_of_2(num_slots), BC=bc_fwd,
            DOT_IEEE=ieee, num_warps=nw_ro, num_stages=1,
        )
        out_scaled, h_v, _ = _cyfa_scalar_decay_fwd_rsi(
            q=weights, k=write, v=v.contiguous(), g_cumsum=g_cumsum, scale=post_scale,
            initial_state=None, output_final_state=False, save_h=True,
        )
        ctx.save_for_backward(
            q, k, v, q_norm, k_norm, q_norm_weight, k_norm_weight,
            q_rstd, k_rstd, delta, beta, g_cumsum, lambdas,
            readout_table, raw_logits, weights, write, h_k, h_v,
        )
        ctx.scale = float(scale)
        ctx.post_scale = float(post_scale)
        ctx.g_dtype = g.dtype
        return out_scaled, None, None, None

    @staticmethod
    def backward(ctx, do_scaled, _1=None, _2=None, _3=None):
        (
            q_raw, k_raw, v, q_norm, k_norm, q_norm_weight, k_norm_weight,
            q_rstd, k_rstd, delta, beta, g_cumsum, lambdas,
            readout_table, raw_logits, weights, write, h_k, h_v,
        ) = ctx.saved_tensors
        bsz, seq_len, n_heads, _ = q_raw.shape
        num_slots = readout_table.shape[-1]
        readout_slots = readout_table.shape[-2]
        ieee = q_raw.dtype == torch.float32

        dweights, dwrite_v, dv, dg_part_v, dg_last_part_v, _ = _cyfa_scalar_decay_bwd_rsi(
            q=weights, k=write, v=v.contiguous(), g_cumsum=g_cumsum,
            h=h_v, initial_state=None, do=do_scaled.contiguous(), dht=None, scale=ctx.post_scale,
        )

        n_tokens = bsz * seq_len
        bm = triton.next_power_of_2(num_slots)
        probs = torch.empty((bsz, seq_len, n_heads, readout_slots), device=q_raw.device, dtype=q_raw.dtype)
        dlogits = torch.empty_like(probs)
        draw_logits = torch.empty_like(raw_logits)
        bt_p1 = 16 if ieee else 32
        bc_p1 = 32 if ieee else min(64, bm)
        nw_p1 = 4 if ieee else 8
        bt_p2 = 16 if ieee else 32
        br_p2 = 32 if ieee else min(64, bm)
        bc_p2 = 32 if ieee else min(64, bm)
        nc_readout = triton.cdiv(num_slots, bc_p2)
        dlambda_readout_part = torch.empty((nc_readout, bsz, seq_len, n_heads), device=q_raw.device, dtype=torch.float32)
        table_work = readout_table.contiguous()

        _cyfa_readout_bwd_pass1_rsi_kernel[(triton.cdiv(n_tokens, bt_p1), n_heads)](
            lambdas, table_work, raw_logits, dweights,
            probs, dlogits,
            N=n_tokens, H=n_heads, M=num_slots, C=readout_slots,
            BT=bt_p1, BM=bm, BC=bc_p1, DOT_IEEE=ieee, num_warps=nw_p1, num_stages=1,
        )
        _cyfa_readout_bwd_pass2_rsi_kernel[(triton.cdiv(n_tokens, bt_p2), nc_readout, n_heads)](
            lambdas, table_work, raw_logits, dweights,
            probs, dlogits, draw_logits, dlambda_readout_part,
            N=n_tokens, H=n_heads, M=num_slots, C=readout_slots,
            BT=bt_p2, BR=br_p2, BC=bc_p2, DOT_IEEE=ieee, num_warps=4, num_stages=1,
        )

        bn_tg = 32 if ieee else 64
        bc_tg = 32 if ieee else min(64, bm)
        nw_tg = 4 if ieee else 8
        n_splits = min(32, triton.cdiv(n_tokens, bn_tg))
        partials = torch.empty((n_splits, n_heads, readout_slots, num_slots), device=q_raw.device, dtype=torch.float32)
        dreadout_table = torch.empty_like(readout_table, dtype=torch.float32)
        _cyfa_readout_table_grad_rsi_kernel[(triton.cdiv(readout_slots, 32), triton.cdiv(num_slots, bc_tg), n_heads * n_splits)](
            lambdas, probs, dweights, dlogits, raw_logits, partials,
            T=n_tokens, H=n_heads, M=num_slots, C=readout_slots,
            BN=bn_tg, BR=32, BC=bc_tg, DOT_IEEE=ieee, num_warps=nw_tg, num_stages=1,
        )
        _cyfa_readout_table_grad_reduce_rsi_kernel[(triton.cdiv(dreadout_table.numel(), 128),)](
            partials, dreadout_table, S=n_splits, SIZE=dreadout_table.numel(), BLOCK=128,
            num_warps=4, num_stages=1,
        )

        dq_norm, dk_norm, dwrite_k, dg_part_k, dg_last_part_k, _ = _cyfa_scalar_decay_bwd_rsi(
            q=q_norm, k=k_norm, v=write, g_cumsum=g_cumsum,
            h=h_k, initial_state=None, do=draw_logits, dht=None, scale=ctx.scale,
        )

        phase_force = torch.empty((bsz, seq_len, n_heads), device=q_raw.device, dtype=torch.float32)
        dbeta = torch.empty_like(beta)
        _cyfa_write_bwd_rsi_kernel[(triton.cdiv(seq_len, 16), bsz * n_heads)](
            dwrite_k, dwrite_v, lambdas, beta.contiguous(), dlambda_readout_part,
            phase_force, dbeta,
            T=seq_len, H=n_heads, M=num_slots, BT=16, BM=bm, NC_READOUT=nc_readout,
            num_warps=4, num_stages=1,
        )

        ddelta = torch.empty_like(delta)
        _cyfa_clock_rev_cumsum_bwd_rsi_kernel[(bsz * n_heads,)](
            phase_force, None, ddelta, None,
            T=seq_len, H=n_heads, BT=256, HAS_DLT=False, HAS_DLAMBDA0=False,
            num_warps=4, num_stages=1,
        )

        dg = torch.empty((bsz, seq_len, n_heads), device=q_raw.device, dtype=ctx.g_dtype)
        nt = triton.cdiv(seq_len, 64)
        _cyfa_bwd_dg_reduce_rev_cumsum_rsi_kernel[(nt, bsz * n_heads)](
            dg_part_v, dg_last_part_v, dg_part_k, dg_last_part_k, dg,
            T=seq_len, H=n_heads, BT=64, NR=2,
            NK_V=dg_part_v.shape[0], NK_K=dg_part_k.shape[0],
            num_warps=4, num_stages=1,
        )

        dq, dk, dq_norm_w, dk_norm_w = cyfa_qk_rmsnorm_bwd_rsi(
            dq_norm, dk_norm, q_raw, k_raw, q_norm_weight, k_norm_weight, q_rstd, k_rstd
        )
        return (
            dq, dk, dv, dg, ddelta, dbeta, dreadout_table.to(readout_table.dtype),
            dq_norm_w, dk_norm_w, None, None, None, None,
        )


# ============================================================================
# 8. 8-GPU NVSwitch Context Parallelism (`CyFA-CP = 8`: Ulysses vs Ring)
# ============================================================================

class _UlyssesAllToAllSeq2Head(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        world_size = dist.get_world_size(group)
        bsz, t_local, n_heads, dim = x.shape
        assert n_heads % world_size == 0
        h_local = n_heads // world_size
        # [B, T_local, P, H_local, D] -> [P, B, T_local, H_local, D]
        send_buf = x.view(bsz, t_local, world_size, h_local, dim).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        # [P, B, T_local, H_local, D] -> [B, P * T_local, H_local, D]
        return recv_buf.permute(1, 0, 2, 3, 4).reshape(bsz, world_size * t_local, h_local, dim).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        world_size = dist.get_world_size(group)
        bsz, t_global, h_local, dim = grad_out.shape
        t_local = t_global // world_size
        send_buf = grad_out.view(bsz, world_size, t_local, h_local, dim).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 2, 0, 3, 4).reshape(bsz, t_local, world_size * h_local, dim).contiguous(), None


class _UlyssesAllToAllHead2Seq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        world_size = dist.get_world_size(group)
        bsz, t_global, h_local, dim = x.shape
        t_local = t_global // world_size
        send_buf = x.view(bsz, world_size, t_local, h_local, dim).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 2, 0, 3, 4).reshape(bsz, t_local, world_size * h_local, dim).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        world_size = dist.get_world_size(group)
        bsz, t_local, n_heads, dim = grad_out.shape
        h_local = n_heads // world_size
        send_buf = grad_out.view(bsz, t_local, world_size, h_local, dim).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 0, 2, 3, 4).reshape(bsz, world_size * t_local, h_local, dim).contiguous(), None


def cyfa_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    delta_local: torch.Tensor,
    beta_local: torch.Tensor,
    readout: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_eps: float = 1e-5,
    k_norm_eps: float = 1e-5,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    bsz, t_local, n_heads, d_k = q_local.shape
    d_v = v_local.shape[-1]
    h_local = n_heads // world_size

    # Pack (g, delta, beta) into [B, T_local, H, 3] for a single FP32 collective
    gdb_local = torch.stack((g_local.float(), delta_local.float(), beta_local.float()), dim=-1)
    gdb_all = _UlyssesAllToAllSeq2Head.apply(gdb_local, group)
    g_all, delta_all, beta_all = gdb_all.unbind(dim=-1)

    # Pack (q, k, v) when d_k == d_v into [B, T_local, H, 2*d_k + d_v]
    qkv_local = torch.cat((q_local, k_local, v_local), dim=-1)
    qkv_all = _UlyssesAllToAllSeq2Head.apply(qkv_local, group)
    q_all, k_all, v_all = torch.split(qkv_all, [d_k, d_k, d_v], dim=-1)

    readout_local = readout[rank * h_local : (rank + 1) * h_local]
    o_all, _ = chunk_cyfa_rsi(
        q=q_all.contiguous(),
        k=k_all.contiguous(),
        v=v_all.contiguous(),
        g=g_all.contiguous(),
        delta=delta_all.contiguous(),
        beta=beta_all.contiguous(),
        readout=readout_local,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        q_norm_eps=q_norm_eps,
        k_norm_eps=k_norm_eps,
        scale=d_k ** -0.5,
    )
    return _UlyssesAllToAllHead2Seq.apply(o_all, group)


class _CyFARingStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_local, k_local, v_local, g_local, delta_local, beta_local,
        readout, q_norm_weight, k_norm_weight, q_norm_eps, k_norm_eps,
        group: dist.ProcessGroup,
    ):
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        bsz, _, n_heads, d_k = q_local.shape
        d_v = v_local.shape[-1]
        num_slots = readout.shape[-1] + 1

        if rank > 0:
            hk0 = torch.empty((bsz, n_heads, d_k, num_slots), device=q_local.device, dtype=torch.float32)
            hv0 = torch.empty((bsz, n_heads, num_slots, d_v), device=q_local.device, dtype=torch.float32)
            fl0 = torch.empty((bsz, n_heads), device=q_local.device, dtype=torch.float32)
            dist.recv(hk0, src=rank - 1, group=group)
            dist.recv(hv0, src=rank - 1, group=group)
            dist.recv(fl0, src=rank - 1, group=group)
            init_state = (hk0, hv0, fl0)
        else:
            hk0 = hv0 = fl0 = None
            init_state = None

        with torch.enable_grad():
            q_d = q_local.detach().requires_grad_(True)
            k_d = k_local.detach().requires_grad_(True)
            v_d = v_local.detach().requires_grad_(True)
            g_d = g_local.detach().requires_grad_(True)
            d_d = delta_local.detach().requires_grad_(True)
            b_d = beta_local.detach().requires_grad_(True)
            r_d = readout.detach().requires_grad_(True)
            qw_d = q_norm_weight.detach().requires_grad_(True)
            kw_d = k_norm_weight.detach().requires_grad_(True)
            if init_state is not None:
                hk0_d = hk0.detach().requires_grad_(True)
                hv0_d = hv0.detach().requires_grad_(True)
                fl0_d = fl0.detach().requires_grad_(True)
                init_d = (hk0_d, hv0_d, fl0_d)
            else:
                hk0_d = hv0_d = fl0_d = None
                init_d = None

            o_local, final_state = chunk_cyfa_rsi(
                q=q_d, k=k_d, v=v_d, g=g_d, delta=d_d, beta=b_d,
                readout=r_d, q_norm_weight=qw_d, k_norm_weight=kw_d,
                q_norm_eps=q_norm_eps, k_norm_eps=k_norm_eps,
                scale=d_k ** -0.5, initial_state=init_d, output_final_state=True,
            )

        hkt, hvt, flt = final_state
        if rank < world_size - 1:
            dist.send(hkt.detach().contiguous(), dst=rank + 1, group=group)
            dist.send(hvt.detach().contiguous(), dst=rank + 1, group=group)
            dist.send(flt.detach().contiguous(), dst=rank + 1, group=group)

        ctx.group = group
        ctx.saved_graph = (o_local, hkt, hvt, flt, q_d, k_d, v_d, g_d, d_d, b_d, r_d, qw_d, kw_d, hk0_d, hv0_d, fl0_d)
        return o_local.detach()

    @staticmethod
    def backward(ctx, do_local):
        group = ctx.group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        o_local, hkt, hvt, flt, q_d, k_d, v_d, g_d, d_d, b_d, r_d, qw_d, kw_d, hk0_d, hv0_d, fl0_d = ctx.saved_graph

        if rank < world_size - 1:
            dhkt = torch.empty_like(hkt)
            dhvt = torch.empty_like(hvt)
            dflt = torch.empty_like(flt)
            dist.recv(dhkt, src=rank + 1, group=group)
            dist.recv(dhvt, src=rank + 1, group=group)
            dist.recv(dflt, src=rank + 1, group=group)
            torch.autograd.backward(
                [o_local, hkt, hvt, flt],
                [do_local, dhkt, dhvt, dflt],
            )
        else:
            torch.autograd.backward([o_local], [do_local])

        if rank > 0:
            dist.send(hk0_d.grad.contiguous(), dst=rank - 1, group=group)
            dist.send(hv0_d.grad.contiguous(), dst=rank - 1, group=group)
            dist.send(fl0_d.grad.contiguous(), dst=rank - 1, group=group)

        dr = r_d.grad.clone()
        dqw = qw_d.grad.clone()
        dkw = kw_d.grad.clone()
        dist.all_reduce(dr, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(dqw, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(dkw, op=dist.ReduceOp.SUM, group=group)
        return q_d.grad, k_d.grad, v_d.grad, g_d.grad, d_d.grad, b_d.grad, dr, dqw, dkw, None, None, None


def cyfa_cp8_ring_baseline(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    delta_local: torch.Tensor,
    beta_local: torch.Tensor,
    readout: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_eps: float = 1e-5,
    k_norm_eps: float = 1e-5,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    if group is None:
        group = dist.group.WORLD
    return _CyFARingStepFunction.apply(
        q_local, k_local, v_local, g_local, delta_local, beta_local,
        readout, q_norm_weight, k_norm_weight, q_norm_eps, k_norm_eps, group,
    )


RSI_KERNEL_OBJECTS = [
    ("_cyfa_qk_rmsnorm_fwd_rsi_kernel", _cyfa_qk_rmsnorm_fwd_rsi_kernel),
    ("_cyfa_qk_rmsnorm_bwd_rsi_kernel", _cyfa_qk_rmsnorm_bwd_rsi_kernel),
    ("_cyfa_clock_and_decay_cumsum_fwd_rsi_kernel", _cyfa_clock_and_decay_cumsum_fwd_rsi_kernel),
    ("_cyfa_clock_rev_cumsum_bwd_rsi_kernel", _cyfa_clock_rev_cumsum_bwd_rsi_kernel),
    ("_cyfa_write_fwd_rsi_kernel", _cyfa_write_fwd_rsi_kernel),
    ("_cyfa_write_bwd_rsi_kernel", _cyfa_write_bwd_rsi_kernel),
    ("_cyfa_modal_readout_fwd_rsi_kernel", _cyfa_modal_readout_fwd_rsi_kernel),
    ("_cyfa_readout_bwd_pass1_rsi_kernel", _cyfa_readout_bwd_pass1_rsi_kernel),
    ("_cyfa_readout_bwd_pass2_rsi_kernel", _cyfa_readout_bwd_pass2_rsi_kernel),
    ("_cyfa_readout_table_grad_rsi_kernel", _cyfa_readout_table_grad_rsi_kernel),
    ("_cyfa_readout_table_grad_reduce_rsi_kernel", _cyfa_readout_table_grad_reduce_rsi_kernel),
    ("_cyfa_chunk_fwd_h_rsi_kernel", _cyfa_chunk_fwd_h_rsi_kernel),
    ("_cyfa_chunk_fwd_o_rsi_kernel", _cyfa_chunk_fwd_o_rsi_kernel),
    ("_cyfa_chunk_bwd_dh_rsi_kernel", _cyfa_chunk_bwd_dh_rsi_kernel),
    ("_cyfa_chunk_bwd_dqk_rsi_kernel", _cyfa_chunk_bwd_dqk_rsi_kernel),
    ("_cyfa_chunk_bwd_dv_rsi_kernel", _cyfa_chunk_bwd_dv_rsi_kernel),
    ("_cyfa_bwd_dg_reduce_rev_cumsum_rsi_kernel", _cyfa_bwd_dg_reduce_rev_cumsum_rsi_kernel),
    ("_cyfa_gate_prep_fwd_rsi_kernel", _cyfa_gate_prep_fwd_rsi_kernel),
    ("_cyfa_gate_prep_bwd_rsi_kernel", _cyfa_gate_prep_bwd_rsi_kernel),
]


def audit_cyfa_rsi_cubin_kernels() -> dict:
    per_kernel = {}
    total_specs = 0
    spilling_specs = 0
    total_spills = 0
    max_regs_all = 0
    max_spills_all = 0
    max_smem_all = 0

    for name, kobj in RSI_KERNEL_OBJECTS:
        compiled = _extract_compiled_kernels(kobj)
        n_spilling = sum(1 for c in compiled if int(getattr(c, "n_spills", 0)) > 0)
        spills_sum = sum(int(getattr(c, "n_spills", 0)) for c in compiled)
        m_regs = max((int(getattr(c, "n_regs", 0)) for c in compiled), default=0)
        m_spills = max((int(getattr(c, "n_spills", 0)) for c in compiled), default=0)
        m_smem = max(
            (int(getattr(c.metadata, "shared", 0)) for c in compiled if hasattr(c, "metadata")),
            default=0,
        )
        per_kernel[name] = {
            "specializations": len(compiled),
            "spilling_specializations": int(n_spilling),
            "max_regs": int(m_regs),
            "max_spills": int(m_spills),
            "total_spills": int(spills_sum),
            "max_shared_bytes": int(m_smem),
        }
        total_specs += len(compiled)
        spilling_specs += int(n_spilling)
        total_spills += int(spills_sum)
        max_regs_all = max(max_regs_all, int(m_regs))
        max_spills_all = max(max_spills_all, int(m_spills))
        max_smem_all = max(max_smem_all, int(m_smem))

    return {
        "kernel_count": len(RSI_KERNEL_OBJECTS),
        "total_specializations": int(total_specs),
        "spilling_specializations": int(spilling_specs),
        "total_spills": int(total_spills),
        "max_regs": int(max_regs_all),
        "max_spills": int(max_spills_all),
        "max_shared_bytes": int(max_smem_all),
        "all_zero_spill": int(total_spills) == 0,
        "per_kernel": per_kernel,
    }

