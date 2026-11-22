#!/usr/bin/env python3
"""
E134 (v125, Round-8 RSI): PaTH Attention & PaTH-FoX Lock-Free UT + K-Duality
Streaming Kernels and Distributed Ring PaTH-CP Context Parallelism on SM80 A100.

Primary Literature & Official Reference Grounding:
  - Songlin Yang, Yikang Shen, Kaiyue Wen, Shawn Tan, Mayank Mishra, Liliang Ren,
    Rameswar Panda, Yoon Kim (MIT, MIT-IBM Watson AI Lab, Harvard, Microsoft),
    "PaTH Attention: Position Encoding via Accumulating Householder Transformations,"
    arXiv:2505.16381v2 (NeurIPS 2025 / Feb 2026).
  - Official reference implementation audited on node-0:
    fla-org/flash-linear-attention/fla/ops/path_attn/ (12 kernel files) and
    fla/layers/path_attn.py.

Bottlenecks Identified in Official fla-org/flash-linear-attention on SM80 A100:
  1. Register Spills at K = 128: Official `chunk_cumprod_householder_bwd_kernel`
     holds three dense [K, K] = [128, 128] FP32 transition matrices (`b_dhc`,
     `b_dh`, `b_hc`) simultaneously in a single CTA, compiling to `n_regs = 32`
     with `n_spills = 3,668` local-memory register spills on SM80 A100.
  2. Quadratic O(T^2 / S) HBM Workspace (`q_new_large`): Official
     `chunk_cumprod_householder_fwd` allocates `q_new_large` of shape
     `[B, T, num_splits, HQ, K]`, storing transformed queries at every split
     boundary (32 copies of Q at T = 16,384).
  3. Unfused Frontend & Separate FoX Cumsum Launches: Unfused causal ShortConv +
     FP32 L2Norm + beta sigmoid and separate FoX gate prefix scans.

Round-8 RSI Architectural Innovations Implemented Here:
  1. Fused Causal ShortConv(k=3, SiLU) + eps-Guarded FP32 L2Norm + Beta Sigmoid
     Forward & Backward (`_path_frontend_conv3_silu_l2norm_fwd/bwd_kernel`):
     Zero-NaN radial projection Jacobian `(dw - (dw . w_hat) w_hat) / r`.
  2. Fused FoX LogSigmoid Gate + Chunked Inclusive/Reverse FP32 Cumsum
     (`_path_fox_gate_fwd_cumsum_kernel`, `_path_fox_gate_bwd_rev_cumsum_kernel`).
  3. Lock-Free Streaming UT + Query-Key Duality 1-GPU PaTH-FoX Forward & Backward:
     - Zero `[128, 128]` dense transition matrices in 1-GPU kernels (`0` cubin
       spills at both K = 64 and K = 128).
     - Zero `q_new_large` (`O(T)` strictly linear HBM workspace).
     - Zero `tl.atomic_add` across all forward and backward passes (Query-Owner
       inter-chunk dQ + Key-Duality inter-chunk dK/dV/dW/dU with bounded `NW_K`
       private worker slots + UT Intra-Chunk backward with in-SRAM GQA loop).
     - Native unpadded ragged `cu_seqlens` varlen batching via per-chunk metadata.
  4. Strip-Mined Ring PaTH-CP Distributed Context Parallelism (`CP = 8` on
     NVSwitch): `[32, K]` row-strip shard Householder defect kernels
     (`_path_cp_shard_defect_fwd/bwd_kernel`, `0` cubin spills at K = 128) with
     exact online-softmax cross-rank ring forward and reverse-ring adjoint.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

for _fla_cand in ("/tmp/fla_repo", "/tmp/flash-linear-attention"):
    if os.path.isdir(_fla_cand) and _fla_cand not in sys.path:
        sys.path.insert(0, _fla_cand)

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl


# =============================================================================
# 1. Chunk Metadata Builder for Fixed-Length and Ragged Unpadded `cu_seqlens`
# =============================================================================

_CHUNK_META_CACHE: Dict[Tuple[Any, ...], Tuple[torch.Tensor, ...]] = {}


def build_chunk_meta(
    B: int,
    T: int,
    device: torch.device,
    cu_seqlens: Optional[torch.Tensor] = None,
    BT: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build per-chunk metadata arrays of length `NT_total`:
      - chunk_bos[c]: global token start `bos` of the sequence owning chunk `c`
      - chunk_eos[c]: global token end `eos` of the sequence owning chunk `c`
      - chunk_first_c[c]: first chunk index belonging to the same sequence
      - chunk_last_c[c]: one-past-last chunk index belonging to the same sequence
      - chunk_local_c[c]: 0-based chunk offset within that sequence (`c - chunk_first_c[c]`)
    Supports both dense batches (`cu_seqlens is None`) and arbitrary ragged
    packed batches (`cu_seqlens` of shape `[N_seq + 1]`, `B == 1`) even when
    individual sequence lengths are not multiples of `BT = 32`.
    """
    if cu_seqlens is None:
        key = ("dense", B, T, BT, device.index)
        if key in _CHUNK_META_CACHE:
            bos, eos, fc, lc, loc = _CHUNK_META_CACHE[key]
            return bos, eos, fc, lc, loc, int(bos.numel())
        nt = triton.cdiv(T, BT)
        bos = torch.zeros((nt,), dtype=torch.int32, device=device)
        eos = torch.full((nt,), T, dtype=torch.int32, device=device)
        fc = torch.zeros((nt,), dtype=torch.int32, device=device)
        lc = torch.full((nt,), nt, dtype=torch.int32, device=device)
        loc = torch.arange(nt, dtype=torch.int32, device=device)
        _CHUNK_META_CACHE[key] = (bos, eos, fc, lc, loc)
        return bos, eos, fc, lc, loc, nt

    assert B == 1, "Packed varlen `cu_seqlens` requires B == 1 (tokens packed on axis 1)."
    cu_cpu = cu_seqlens.detach().to( dtype=torch.int32, device="cpu").tolist()
    key = ("varlen", tuple(cu_cpu), BT, device.index)
    if key in _CHUNK_META_CACHE:
        bos, eos, fc, lc, loc = _CHUNK_META_CACHE[key]
        return bos, eos, fc, lc, loc, int(bos.numel())

    bos_list: List[int] = []
    eos_list: List[int] = []
    fc_list: List[int] = []
    lc_list: List[int] = []
    loc_list: List[int] = []
    cur_c = 0
    for s_idx in range(len(cu_cpu) - 1):
        b_tok = int(cu_cpu[s_idx])
        e_tok = int(cu_cpu[s_idx + 1])
        slen = max(0, e_tok - b_tok)
        n_c = (slen + BT - 1) // BT
        for cl in range(n_c):
            bos_list.append(b_tok)
            eos_list.append(e_tok)
            fc_list.append(cur_c)
            lc_list.append(cur_c + n_c)
            loc_list.append(cl)
        cur_c += n_c

    bos = torch.tensor(bos_list, dtype=torch.int32, device=device)
    eos = torch.tensor(eos_list, dtype=torch.int32, device=device)
    fc = torch.tensor(fc_list, dtype=torch.int32, device=device)
    lc = torch.tensor(lc_list, dtype=torch.int32, device=device)
    loc = torch.tensor(loc_list, dtype=torch.int32, device=device)
    _CHUNK_META_CACHE[key] = (bos, eos, fc, lc, loc)
    return bos, eos, fc, lc, loc, cur_c


# =============================================================================
# 2. Fused Causal ShortConv(k=3, SiLU) + FP32 L2Norm + Beta Sigmoid Kernels
# =============================================================================

@triton.jit
def _path_frontend_conv3_silu_l2norm_fwd_kernel(
    U_raw,
    Conv_W,
    Beta_raw,
    W_unit,
    Beta_out,
    Pre_conv,
    Inv_r,
    Chunk_bos,
    Chunk_eos,
    Chunk_local_c,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BETA_SCALE: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_idx = bos + loc_c * BT + offs_t
    m_valid = tok_idx < eos

    offs_k = tl.arange(0, K)
    base_bh = b * T * H + h

    # Load depthwise conv weights [H, 3, K]
    cw_base = h * 3 * K + offs_k
    w0 = tl.load(Conv_W + cw_base + 0 * K).to(tl.float32)
    w1 = tl.load(Conv_W + cw_base + 1 * K).to(tl.float32)
    w2 = tl.load(Conv_W + cw_base + 2 * K).to(tl.float32)

    # Causal positions tok_idx, tok_idx - 1, tok_idx - 2 within [bos, eos)
    u0_ptrs = U_raw + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :]
    u0 = tl.load(u0_ptrs, mask=m_valid[:, None], other=0.0).to(tl.float32)

    t_m1 = tok_idx - 1
    m_m1 = m_valid & (t_m1 >= bos)
    u1_ptrs = U_raw + (base_bh + tl.maximum(t_m1, bos)[:, None] * H) * K + offs_k[None, :]
    u1 = tl.load(u1_ptrs, mask=m_m1[:, None], other=0.0).to(tl.float32)

    t_m2 = tok_idx - 2
    m_m2 = m_valid & (t_m2 >= bos)
    u2_ptrs = U_raw + (base_bh + tl.maximum(t_m2, bos)[:, None] * H) * K + offs_k[None, :]
    u2 = tl.load(u2_ptrs, mask=m_m2[:, None], other=0.0).to(tl.float32)

    z = u0 * w0[None, :] + u1 * w1[None, :] + u2 * w2[None, :]
    sig_z = tl.sigmoid(z)
    act = z * sig_z

    r2 = tl.sum(act * act, axis=1)
    inv_r = tl.where(r2 > EPS, tl.rsqrt(tl.maximum(r2, EPS)), 0.0)
    w_hat = act * inv_r[:, None]

    # Store Pre_conv, Inv_r, W_unit
    pre_ptrs = Pre_conv + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :]
    tl.store(pre_ptrs, z, mask=m_valid[:, None])
    w_ptrs = W_unit + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :]
    tl.store(w_ptrs, w_hat.to(W_unit.dtype.element_ty), mask=m_valid[:, None])

    ir_ptrs = Inv_r + b * T * H + tok_idx * H + h
    tl.store(ir_ptrs, inv_r, mask=m_valid)

    # Beta = BETA_SCALE * sigmoid(Beta_raw)
    bt_ptrs = Beta_raw + b * T * H + tok_idx * H + h
    bt_in = tl.load(bt_ptrs, mask=m_valid, other=0.0).to(tl.float32)
    bt_out = BETA_SCALE * tl.sigmoid(bt_in)
    bto_ptrs = Beta_out + b * T * H + tok_idx * H + h
    tl.store(bto_ptrs, bt_out.to(Beta_out.dtype.element_ty), mask=m_valid)


@triton.jit
def _path_frontend_conv3_silu_l2norm_bwd_kernel(
    U_raw,
    Conv_W,
    Beta_raw,
    Pre_conv,
    Inv_r,
    dW_unit,
    dBeta_out,
    dZ_buf,
    dBeta_raw,
    Chunk_bos,
    Chunk_eos,
    Chunk_local_c,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BETA_SCALE: tl.constexpr,
):
    """Stage 1 of frontend backward: radial L2Norm + SiLU backward -> dZ_buf and dBeta_raw."""
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_idx = bos + loc_c * BT + offs_t
    m_valid = tok_idx < eos
    offs_k = tl.arange(0, K)
    base_bh = b * T * H + h

    z_ptrs = Pre_conv + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :]
    z = tl.load(z_ptrs, mask=m_valid[:, None], other=0.0).to(tl.float32)
    inv_r = tl.load(Inv_r + b * T * H + tok_idx * H + h, mask=m_valid, other=0.0).to(tl.float32)

    sig_z = tl.sigmoid(z)
    act = z * sig_z
    w_hat = act * inv_r[:, None]

    dw_ptrs = dW_unit + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :]
    dw = tl.load(dw_ptrs, mask=m_valid[:, None], other=0.0).to(tl.float32)

    dot = tl.sum(dw * w_hat, axis=1)
    d_act = (dw - dot[:, None] * w_hat) * inv_r[:, None]
    dsilu = sig_z * (1.0 + z * (1.0 - sig_z))
    dz = tl.where(m_valid[:, None], d_act * dsilu, 0.0)

    dz_ptrs = dZ_buf + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :]
    tl.store(dz_ptrs, dz, mask=m_valid[:, None])

    bt_ptrs = Beta_raw + b * T * H + tok_idx * H + h
    bt_in = tl.load(bt_ptrs, mask=m_valid, other=0.0).to(tl.float32)
    dbt_out = tl.load(dBeta_out + b * T * H + tok_idx * H + h, mask=m_valid, other=0.0).to(tl.float32)
    s_bt = tl.sigmoid(bt_in)
    dbt_in = dbt_out * BETA_SCALE * s_bt * (1.0 - s_bt)
    tl.store(dBeta_raw + b * T * H + tok_idx * H + h, dbt_in.to(dBeta_raw.dtype.element_ty), mask=m_valid)


@triton.jit
def _path_frontend_conv3_gather_du_bwd_kernel(
    U_raw,
    Conv_W,
    dZ_buf,
    dU_raw,
    dConv_W_part,
    Chunk_bos,
    Chunk_eos,
    Chunk_local_c,
    NT_total,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    """Stage 2 of frontend backward: atomic-free causal gather for dU_raw and per-chunk dConv_W_part."""
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_idx = bos + loc_c * BT + offs_t
    m_valid = tok_idx < eos
    offs_k = tl.arange(0, K)
    base_bh = b * T * H + h

    cw_base = h * 3 * K + offs_k
    w0 = tl.load(Conv_W + cw_base + 0 * K).to(tl.float32)
    w1 = tl.load(Conv_W + cw_base + 1 * K).to(tl.float32)
    w2 = tl.load(Conv_W + cw_base + 2 * K).to(tl.float32)

    dz0 = tl.load(dZ_buf + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :], mask=m_valid[:, None], other=0.0).to(tl.float32)

    t_p1 = tok_idx + 1
    m_p1 = m_valid & (t_p1 < eos)
    dz1 = tl.load(
        dZ_buf + (base_bh + tl.minimum(t_p1, eos - 1)[:, None] * H) * K + offs_k[None, :],
        mask=m_p1[:, None],
        other=0.0,
    ).to(tl.float32)

    t_p2 = tok_idx + 2
    m_p2 = m_valid & (t_p2 < eos)
    dz2 = tl.load(
        dZ_buf + (base_bh + tl.minimum(t_p2, eos - 1)[:, None] * H) * K + offs_k[None, :],
        mask=m_p2[:, None],
        other=0.0,
    ).to(tl.float32)

    du = dz0 * w0[None, :] + dz1 * w1[None, :] + dz2 * w2[None, :]
    du_ptrs = dU_raw + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :]
    tl.store(du_ptrs, du.to(dU_raw.dtype.element_ty), mask=m_valid[:, None])

    # Load u0, u_{-1}, u_{-2} to form per-chunk dConv_W_part without atomics
    u0 = tl.load(U_raw + (base_bh + tok_idx[:, None] * H) * K + offs_k[None, :], mask=m_valid[:, None], other=0.0).to(tl.float32)
    t_m1 = tok_idx - 1
    m_m1 = m_valid & (t_m1 >= bos)
    u1 = tl.load(U_raw + (base_bh + tl.maximum(t_m1, bos)[:, None] * H) * K + offs_k[None, :], mask=m_m1[:, None], other=0.0).to(tl.float32)
    t_m2 = tok_idx - 2
    m_m2 = m_valid & (t_m2 >= bos)
    u2 = tl.load(U_raw + (base_bh + tl.maximum(t_m2, bos)[:, None] * H) * K + offs_k[None, :], mask=m_m2[:, None], other=0.0).to(tl.float32)

    dw0 = tl.sum(dz0 * u0, axis=0)
    dw1 = tl.sum(dz0 * u1, axis=0)
    dw2 = tl.sum(dz0 * u2, axis=0)

    part_base = ((pid_bh * NT_total + pid_c) * 3) * K + offs_k
    tl.store(dConv_W_part + part_base + 0 * K, dw0)
    tl.store(dConv_W_part + part_base + 1 * K, dw1)
    tl.store(dConv_W_part + part_base + 2 * K, dw2)


class PaTHFrontendFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_raw: torch.Tensor,
        conv_w: torch.Tensor,
        beta_raw: torch.Tensor,
        beta_scale: float = 2.0,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = u_raw.shape
        BT = 32
        bos, eos, fc, lc, loc, nt_total = build_chunk_meta(B, T, u_raw.device, cu_seqlens, BT=BT)
        w_unit = torch.empty_like(u_raw)
        beta_out = torch.empty_like(beta_raw)
        pre_conv = torch.empty((B, T, H, K), dtype=torch.float32, device=u_raw.device)
        inv_r = torch.empty((B, T, H), dtype=torch.float32, device=u_raw.device)

        grid = (nt_total, B * H)
        _path_frontend_conv3_silu_l2norm_fwd_kernel[grid](
            u_raw, conv_w, beta_raw, w_unit, beta_out, pre_conv, inv_r,
            bos, eos, loc, T,
            H=H, K=K, BT=BT, BETA_SCALE=float(beta_scale), EPS=1e-12,
            num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(u_raw, conv_w, beta_raw, pre_conv, inv_r, bos, eos, loc)
        ctx.beta_scale = float(beta_scale)
        ctx.nt_total = nt_total
        return w_unit, beta_out

    @staticmethod
    def backward(ctx, dw_unit: torch.Tensor, dbeta_out: torch.Tensor):
        u_raw, conv_w, beta_raw, pre_conv, inv_r, bos, eos, loc = ctx.saved_tensors
        B, T, H, K = u_raw.shape
        BT = 32
        nt_total = ctx.nt_total
        dz_buf = torch.empty((B, T, H, K), dtype=torch.float32, device=u_raw.device)
        dbeta_raw = torch.empty_like(beta_raw)
        du_raw = torch.empty_like(u_raw)
        dconv_part = torch.empty((B, H, nt_total, 3, K), dtype=torch.float32, device=u_raw.device)

        grid = (nt_total, B * H)
        _path_frontend_conv3_silu_l2norm_bwd_kernel[grid](
            u_raw, conv_w, beta_raw, pre_conv, inv_r, dw_unit.contiguous(), dbeta_out.contiguous(),
            dz_buf, dbeta_raw, bos, eos, loc, T,
            H=H, K=K, BT=BT, BETA_SCALE=ctx.beta_scale,
            num_warps=4, num_stages=1,
        )
        _path_frontend_conv3_gather_du_bwd_kernel[grid](
            u_raw, conv_w, dz_buf, du_raw, dconv_part, bos, eos, loc, nt_total, T,
            H=H, K=K, BT=BT,
            num_warps=4, num_stages=1,
        )
        dconv_w = dconv_part.sum(dim=(0, 2)).to(conv_w.dtype)
        return du_raw, dconv_w, dbeta_raw, None, None


def path_frontend_conv3_silu_l2norm(
    u_raw: torch.Tensor,
    conv_w: torch.Tensor,
    beta_raw: torch.Tensor,
    beta_scale: float = 2.0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return PaTHFrontendFunction.apply(u_raw.contiguous(), conv_w.contiguous(), beta_raw.contiguous(), beta_scale, cu_seqlens)


# =============================================================================
# 3. Fused FoX Gate Cumsum Forward & Reverse-Cumsum Backward Kernels
# =============================================================================

@triton.jit
def _path_fox_gate_fwd_cumsum_kernel(
    G_in,
    C_out,
    Chunk_bos,
    Chunk_eos,
    Chunk_first_c,
    Chunk_last_c,
    T,
    HQ: tl.constexpr,
    BT: tl.constexpr,
    APPLY_LOGSIGMOID: tl.constexpr,
):
    """Per-(sequence, batch, query_head) exact FP32 inclusive prefix sum across chunks."""
    pid_seq = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HQ
    hq = pid_bh % HQ

    # pid_seq indexes into unique sequence starts via a chunk scan or direct sequence table
    bos = tl.load(Chunk_bos + pid_seq)
    eos = tl.load(Chunk_eos + pid_seq)
    fc = tl.load(Chunk_first_c + pid_seq)
    lc = tl.load(Chunk_last_c + pid_seq)

    offs_t = tl.arange(0, BT)
    tril_m = offs_t[:, None] >= offs_t[None, :]
    running = tl.zeros([], dtype=tl.float32)

    for c in range(fc, lc):
        loc_c = c - fc
        tok_idx = bos + loc_c * BT + offs_t
        m_valid = tok_idx < eos
        ptrs = b * T * HQ + tok_idx * HQ + hq
        g_val = tl.load(G_in + ptrs, mask=m_valid, other=0.0).to(tl.float32)
        if APPLY_LOGSIGMOID:
            # logsigmoid(x) = -softplus(-x) = -log(1 + exp(-|x|)) + min(x, 0)
            g_val = tl.minimum(g_val, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(g_val)))
            g_val = tl.where(m_valid, g_val, 0.0)
        # Inclusive prefix sum within chunk + running total
        c_chunk = tl.sum(tl.where(tril_m, g_val[None, :], 0.0), axis=1) + running
        tl.store(C_out + ptrs, c_chunk, mask=m_valid)
        running += tl.sum(g_val, axis=0)


@triton.jit
def _path_fox_gate_bwd_rev_cumsum_kernel(
    dC_in,
    G_raw,
    dG_out,
    Chunk_bos,
    Chunk_eos,
    Chunk_first_c,
    Chunk_last_c,
    T,
    HQ: tl.constexpr,
    BT: tl.constexpr,
    APPLY_LOGSIGMOID: tl.constexpr,
):
    """Per-(sequence, batch, query_head) exact FP32 reverse inclusive prefix sum across chunks."""
    pid_seq = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HQ
    hq = pid_bh % HQ

    bos = tl.load(Chunk_bos + pid_seq)
    eos = tl.load(Chunk_eos + pid_seq)
    fc = tl.load(Chunk_first_c + pid_seq)
    lc = tl.load(Chunk_last_c + pid_seq)

    offs_t = tl.arange(0, BT)
    triu_m = offs_t[:, None] <= offs_t[None, :]
    running = tl.zeros([], dtype=tl.float32)
    n_chunks = lc - fc

    for step in range(0, n_chunks):
        c = lc - 1 - step
        loc_c = c - fc
        tok_idx = bos + loc_c * BT + offs_t
        m_valid = tok_idx < eos
        ptrs = b * T * HQ + tok_idx * HQ + hq
        dc_val = tl.load(dC_in + ptrs, mask=m_valid, other=0.0).to(tl.float32)
        dg_chunk = tl.sum(tl.where(triu_m, dc_val[None, :], 0.0), axis=1) + running
        if APPLY_LOGSIGMOID:
            g_raw = tl.load(G_raw + ptrs, mask=m_valid, other=0.0).to(tl.float32)
            dg_chunk = dg_chunk * tl.sigmoid(-g_raw)
        tl.store(dG_out + ptrs, dg_chunk.to(dG_out.dtype.element_ty), mask=m_valid)
        running += tl.sum(dc_val, axis=0)


_SEQ_META_CACHE: Dict[Tuple[int, int, Optional[int]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}


def _get_seq_rep_meta(
    bos: torch.Tensor,
    eos: torch.Tensor,
    fc: torch.Tensor,
    lc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return cached 1-entry-per-sequence metadata without CPU-GPU sync on hot path."""
    key = (int(fc.data_ptr()), int(fc.numel()), fc.device.index)
    cached = _SEQ_META_CACHE.get(key)
    if cached is not None:
        return cached
    idx = torch.arange(fc.numel(), device=fc.device, dtype=torch.int32)
    rep_chunks = idx[idx == fc]
    s_bos = bos[rep_chunks].contiguous()
    s_eos = eos[rep_chunks].contiguous()
    s_fc = fc[rep_chunks].contiguous()
    s_lc = lc[rep_chunks].contiguous()
    n_seqs = int(rep_chunks.numel())
    res = (s_bos, s_eos, s_fc, s_lc, n_seqs)
    if len(_SEQ_META_CACHE) > 256:
        _SEQ_META_CACHE.clear()
    _SEQ_META_CACHE[key] = res
    return res


def path_fox_gate_fwd_cumsum_raw(
    g: torch.Tensor,
    bos: torch.Tensor,
    eos: torch.Tensor,
    fc: torch.Tensor,
    lc: torch.Tensor,
    apply_logsigmoid: bool = False,
) -> torch.Tensor:
    B, T, HQ = g.shape
    c_out = torch.empty((B, T, HQ), dtype=torch.float32, device=g.device)
    s_bos, s_eos, s_fc, s_lc, n_seqs = _get_seq_rep_meta(bos, eos, fc, lc)
    _path_fox_gate_fwd_cumsum_kernel[(n_seqs, B * HQ)](
        g, c_out,
        s_bos, s_eos, s_fc, s_lc,
        T, HQ=HQ, BT=32, APPLY_LOGSIGMOID=apply_logsigmoid,
        num_warps=2, num_stages=1,
    )
    return c_out


def path_fox_gate_bwd_rev_cumsum_raw(
    dc: torch.Tensor,
    g_ref: torch.Tensor,
    bos: torch.Tensor,
    eos: torch.Tensor,
    fc: torch.Tensor,
    lc: torch.Tensor,
    apply_logsigmoid: bool = False,
) -> torch.Tensor:
    B, T, HQ = dc.shape
    dg_out = torch.empty_like(g_ref)
    s_bos, s_eos, s_fc, s_lc, n_seqs = _get_seq_rep_meta(bos, eos, fc, lc)
    _path_fox_gate_bwd_rev_cumsum_kernel[(n_seqs, B * HQ)](
        dc, g_ref, dg_out,
        s_bos, s_eos, s_fc, s_lc,
        T, HQ=HQ, BT=32, APPLY_LOGSIGMOID=apply_logsigmoid,
        num_warps=2, num_stages=1,
    )
    return dg_out


# =============================================================================
# 4. Lock-Free UT + Query-Key Duality PaTH-FoX Forward & Backward Kernels
# =============================================================================

@triton.jit
def _path_ut_solve_tril_32x32_fwd_kernel(
    W,
    Beta,
    A_inv,
    Chunk_bos,
    Chunk_eos,
    Chunk_local_c,
    NT_total,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    """
    Compute exact FP32 unit-lower-triangular inverse:
      A_inv[b, c, h] = (I_32 + strictLower(diag(beta_c) @ W_c @ W_c^T))^{-1}
    Stores A_inv in [B, NT_total, H, BT, BT] with 0 register spills.
    """
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_idx = bos + loc_c * BT + offs_t
    m_valid = tok_idx < eos
    offs_k = tl.arange(0, K)

    w_ptrs = W + (b * T * H + tok_idx[:, None] * H + h) * K + offs_k[None, :]
    b_w = tl.load(w_ptrs, mask=m_valid[:, None], other=0.0)
    bt_ptrs = Beta + b * T * H + tok_idx * H + h
    b_beta = tl.load(bt_ptrs, mask=m_valid, other=0.0).to(tl.float32)

    b_wb = (b_w.to(tl.float32) * b_beta[:, None]).to(b_w.dtype)
    m_strict = offs_t[:, None] > offs_t[None, :]
    b_L = tl.where(m_strict, tl.dot(b_wb, tl.trans(b_w), allow_tf32=False), 0.0)

    # Exact forward substitution for unit lower triangular inverse of (I + b_L)
    b_A = -b_L
    for i in range(1, BT):
        mask_row = offs_t == i
        row_l = tl.sum(tl.where(mask_row[:, None], b_L, 0.0), axis=0)
        new_row = -row_l - tl.sum(row_l[:, None] * b_A, axis=0)
        b_A = tl.where(mask_row[:, None], new_row[None, :], b_A)
    b_A = b_A + (offs_t[:, None] == offs_t[None, :]).to(tl.float32)

    a_base = ((b * NT_total + pid_c) * H + h) * BT * BT
    tl.store(A_inv + a_base + offs_t[:, None] * BT + offs_t[None, :], b_A)


@triton.jit
def _path_intra_chunk_ut_fwd_kernel(
    Q,
    K_in,
    V,
    W,
    Beta,
    C_cum,
    A_inv,
    Q_arrow,
    K_arrow,
    U_out,
    O_intra,
    M_intra,
    L_intra,
    Chunk_bos,
    Chunk_eos,
    Chunk_local_c,
    NT_total,
    T,
    scale,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
):
    """
    Fused Intra-Chunk Compact WY/UT Transformation + Intra-Chunk PaTH-FoX Attention:
      - T_c = A_inv[c] * beta_c[None, :]
      - U_c = T_c @ W_c
      - K_arrow_c = K_c - (T_c @ strictLower(W_c @ K_c^T))^T @ W_c
      - For each GQA head g_idx in 0..G-1:
          Q_arrow_c = Q_c - tril(Q_c @ W_c^T) @ U_c
          B_raw = tril(Q_c @ K_c^T - (tril(Q_c @ W_c^T) @ T_c) @ strictLower(W_c @ K_c^T))
          Z_intra = scale * B_raw + (c_i - c_j)
          Compute online softmax initials (O_intra, M_intra, L_intra).
    All tiles are [32, K] or [32, 32] -> 0 register spills at K = 64 and K = 128.
    """
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_idx = bos + loc_c * BT + offs_t
    m_valid = tok_idx < eos
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V_DIM)

    m_tril = offs_t[:, None] >= offs_t[None, :]
    m_strict = offs_t[:, None] > offs_t[None, :]

    w_ptrs = W + (b * T * H + tok_idx[:, None] * H + h) * K + offs_k[None, :]
    b_w = tl.load(w_ptrs, mask=m_valid[:, None], other=0.0)
    k_ptrs = K_in + (b * T * H + tok_idx[:, None] * H + h) * K + offs_k[None, :]
    b_k = tl.load(k_ptrs, mask=m_valid[:, None], other=0.0)
    v_ptrs = V + (b * T * H + tok_idx[:, None] * H + h) * V_DIM + offs_v[None, :]
    b_v = tl.load(v_ptrs, mask=m_valid[:, None], other=0.0)
    b_beta = tl.load(Beta + b * T * H + tok_idx * H + h, mask=m_valid, other=0.0).to(tl.float32)

    a_base = ((b * NT_total + pid_c) * H + h) * BT * BT
    b_A = tl.load(A_inv + a_base + offs_t[:, None] * BT + offs_t[None, :]).to(tl.float32)
    b_T = (b_A * b_beta[None, :]).to(b_w.dtype)

    b_u = tl.dot(b_T, b_w, allow_tf32=False)
    u_base = ((b * NT_total + pid_c) * H + h) * BT * K
    tl.store(U_out + u_base + offs_t[:, None] * K + offs_k[None, :], b_u.to(U_out.dtype.element_ty))

    m_wbk = tl.where(m_strict, tl.dot(b_w, tl.trans(b_k), allow_tf32=False), 0.0).to(b_w.dtype)
    n_twbk = tl.dot(b_T, m_wbk, allow_tf32=False).to(b_w.dtype)
    b_k_arrow = b_k.to(tl.float32) - tl.dot(tl.trans(n_twbk), b_w, allow_tf32=False)
    tl.store(K_arrow + u_base + offs_t[:, None] * K + offs_k[None, :], b_k_arrow.to(K_arrow.dtype.element_ty))

    for g_idx in range(0, G):
        hq = h * G + g_idx
        q_ptrs = Q + (b * T * HQ + tok_idx[:, None] * HQ + hq) * K + offs_k[None, :]
        b_q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

        m_qw = tl.where(m_tril, tl.dot(b_q, tl.trans(b_w), allow_tf32=False), 0.0).to(b_w.dtype)
        n_qwt = tl.dot(m_qw, b_T, allow_tf32=False).to(b_w.dtype)
        b_q_arrow = b_q.to(tl.float32) - tl.dot(n_qwt, b_w, allow_tf32=False)

        qa_base = ((b * NT_total + pid_c) * HQ + hq) * BT * K
        tl.store(Q_arrow + qa_base + offs_t[:, None] * K + offs_k[None, :], b_q_arrow.to(Q_arrow.dtype.element_ty))

        b_raw = tl.dot(b_q, tl.trans(b_k), allow_tf32=False) - tl.dot(n_qwt, m_wbk, allow_tf32=False)
        z = b_raw * scale
        if USE_G:
            c_q = tl.load(C_cum + b * T * HQ + tok_idx * HQ + hq, mask=m_valid, other=0.0).to(tl.float32)
            z = z + (c_q[:, None] - c_q[None, :])

        m_causal_valid = m_tril & m_valid[:, None] & m_valid[None, :]
        z = tl.where(m_causal_valid, z, -float("inf"))
        m_row = tl.max(z, axis=1)
        # Guard empty/invalid rows against -inf - (-inf) = NaN
        m_row_safe = tl.where(m_valid, m_row, 0.0)
        p = tl.where(m_causal_valid, tl.exp(z - m_row_safe[:, None]), 0.0)
        l_row = tl.sum(p, axis=1)
        o_acc = tl.dot(p.to(b_v.dtype), b_v, allow_tf32=False)

        o_base = ((b * NT_total + pid_c) * HQ + hq) * BT * V_DIM
        tl.store(O_intra + o_base + offs_t[:, None] * V_DIM + offs_v[None, :], o_acc)
        ml_base = ((b * NT_total + pid_c) * HQ + hq) * BT + offs_t
        tl.store(M_intra + ml_base, m_row_safe)
        tl.store(L_intra + ml_base, tl.where(m_valid, l_row, 1.0))


@triton.jit
def _path_inter_chunk_streaming_fwd_kernel(
    Q_arrow,
    K_arrow,
    V,
    W,
    U_buf,
    C_cum,
    O_intra,
    M_intra,
    L_intra,
    O_out,
    LSE_out,
    X_init_out,
    Chunk_bos,
    Chunk_eos,
    Chunk_first_c,
    Chunk_local_c,
    NT_total,
    T,
    scale,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    SAVE_X_INIT: tl.constexpr,
):
    """
    Zero-`q_new_large` Streaming Inter-Chunk PaTH-FoX Forward Kernel:
    Propagates `b_q` right-to-left in registers from chunk `pid_c - 1` down to `fc`
    via `b_q <- b_q - (b_q @ W_{ck}^T) @ U_{ck}`, merging online softmax states.
    """
    pid_c = NT_total - 1 - tl.program_id(0)
    pid_bhq = tl.program_id(1)
    b = pid_bhq // HQ
    hq = pid_bhq % HQ
    h = hq // G

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    fc = tl.load(Chunk_first_c + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_q = bos + loc_c * BT + offs_t
    m_q_valid = tok_q < eos
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V_DIM)

    qa_base = ((b * NT_total + pid_c) * HQ + hq) * BT * K
    b_q = tl.load(Q_arrow + qa_base + offs_t[:, None] * K + offs_k[None, :])

    o_base = ((b * NT_total + pid_c) * HQ + hq) * BT * V_DIM
    b_o = tl.load(O_intra + o_base + offs_t[:, None] * V_DIM + offs_v[None, :]).to(tl.float32)
    ml_base = ((b * NT_total + pid_c) * HQ + hq) * BT + offs_t
    b_m = tl.load(M_intra + ml_base).to(tl.float32)
    b_l = tl.load(L_intra + ml_base).to(tl.float32)

    if USE_G:
        c_q = tl.load(C_cum + b * T * HQ + tok_q * HQ + hq, mask=m_q_valid, other=0.0).to(tl.float32)
    else:
        c_q = tl.zeros([BT], dtype=tl.float32)

    n_prev = pid_c - fc
    for step in range(0, n_prev):
        ck = pid_c - 1 - step
        loc_k = ck - fc
        tok_k = bos + loc_k * BT + offs_t
        m_k_valid = tok_k < eos

        ku_base = ((b * NT_total + ck) * H + h) * BT * K
        b_k_arr = tl.load(K_arrow + ku_base + offs_t[:, None] * K + offs_k[None, :])
        v_ptrs = V + (b * T * H + tok_k[:, None] * H + h) * V_DIM + offs_v[None, :]
        b_v = tl.load(v_ptrs, mask=m_k_valid[:, None], other=0.0)

        z = tl.dot(b_q, tl.trans(b_k_arr), allow_tf32=False) * scale
        if USE_G:
            c_k = tl.load(C_cum + b * T * HQ + tok_k * HQ + hq, mask=m_k_valid, other=0.0).to(tl.float32)
            z = z + (c_q[:, None] - c_k[None, :])
        m_pair = m_q_valid[:, None] & m_k_valid[None, :]
        z = tl.where(m_pair, z, -float("inf"))

        m_new = tl.maximum(b_m, tl.max(z, axis=1))
        alpha_m = tl.exp(b_m - m_new)
        p = tl.where(m_pair, tl.exp(z - m_new[:, None]), 0.0)
        b_l = b_l * alpha_m + tl.sum(p, axis=1)
        b_o = b_o * alpha_m[:, None] + tl.dot(p.to(b_v.dtype), b_v, allow_tf32=False)
        b_m = m_new

        # Propagate b_q across chunk ck: b_q <- b_q - (b_q @ W_{ck}^T) @ U_{ck}
        w_ptrs = W + (b * T * H + tok_k[:, None] * H + h) * K + offs_k[None, :]
        b_w = tl.load(w_ptrs, mask=m_k_valid[:, None], other=0.0)
        b_u = tl.load(U_buf + ku_base + offs_t[:, None] * K + offs_k[None, :])
        proj = tl.dot(b_q, tl.trans(b_w), allow_tf32=False).to(b_u.dtype)
        b_q = (b_q.to(tl.float32) - tl.dot(proj, b_u, allow_tf32=False)).to(b_q.dtype)

    if SAVE_X_INIT:
        x_ptrs = X_init_out + (b * T * HQ + tok_q[:, None] * HQ + hq) * K + offs_k[None, :]
        tl.store(x_ptrs, b_q.to(X_init_out.dtype.element_ty), mask=m_q_valid[:, None])

    b_o_final = b_o / b_l[:, None]
    b_lse = b_m + tl.log(b_l)
    o_ptrs = O_out + (b * T * HQ + tok_q[:, None] * HQ + hq) * V_DIM + offs_v[None, :]
    tl.store(o_ptrs, b_o_final.to(O_out.dtype.element_ty), mask=m_q_valid[:, None])
    lse_ptrs = LSE_out + b * T * HQ + tok_q * HQ + hq
    tl.store(lse_ptrs, b_lse, mask=m_q_valid)


@triton.jit
def _path_bwd_preprocess_delta_kernel(
    O,
    dO,
    Delta,
    N_rows,
    V_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N_rows:
        return
    offs_v = tl.arange(0, V_DIM)
    o = tl.load(O + pid * V_DIM + offs_v).to(tl.float32)
    do = tl.load(dO + pid * V_DIM + offs_v).to(tl.float32)
    tl.store(Delta + pid, tl.sum(o * do, axis=0))


@triton.jit
def _path_inter_chunk_query_worker_bwd_kernel(
    Q_arrow,
    K_arrow,
    V,
    W,
    U_buf,
    C_cum,
    dO,
    LSE,
    Delta,
    X_ws,
    dQ_arrow,
    dC_q_inter,
    B_init_in,
    Chunk_bos,
    Chunk_eos,
    Chunk_first_c,
    Chunk_local_c,
    NT_total,
    T,
    scale,
    NW_Q: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    HAS_B_INIT: tl.constexpr,
):
    """
    Lock-Free (0 atomics, O(T) linear HBM!) Query-Worker Inter-Chunk Backward Kernel:
    Grid = (NW_Q, B * HQ). Worker `pid_w` processes query chunks `cq in range(pid_w, NT_total, NW_Q)`
    using its private linear O(T) workspace `X_ws[pid_w, pid_bhq, NT_total, BT, K]`.
    """
    pid_w = tl.program_id(0)
    pid_bhq = tl.program_id(1)
    b = pid_bhq // HQ
    hq = pid_bhq % HQ
    h = hq // G

    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V_DIM)

    ws_bhq_base = (pid_w * (tl.num_programs(1)) + pid_bhq) * NT_total * BT * K

    n_stripes = tl.cdiv(NT_total, NW_Q)
    for s in range(0, n_stripes):
        cq = s * NW_Q + tl.where((s & 1) == 0, pid_w, NW_Q - 1 - pid_w)
        if cq < NT_total:
            bos = tl.load(Chunk_bos + cq)
            eos = tl.load(Chunk_eos + cq)
            fc = tl.load(Chunk_first_c + cq)
            loc_q = tl.load(Chunk_local_c + cq)
            tok_q = bos + loc_q * BT + offs_t
            m_q_valid = tok_q < eos

            qa_base = ((b * NT_total + cq) * HQ + hq) * BT * K
            b_q = tl.load(Q_arrow + qa_base + offs_t[:, None] * K + offs_k[None, :])

            n_prev = cq - fc
            # Pass 1: right-to-left state reconstruction into private linear workspace X_ws[pid_w]
            for step in range(0, n_prev):
                ck = cq - 1 - step
                loc_k = ck - fc
                tok_k = bos + loc_k * BT + offs_t
                m_k_valid = tok_k < eos

                ws_ptr = X_ws + ws_bhq_base + ck * BT * K + offs_t[:, None] * K + offs_k[None, :]
                tl.store(ws_ptr, b_q)

                if ck > fc:
                    ku_base = ((b * NT_total + ck) * H + h) * BT * K
                    w_ptrs = W + (b * T * H + tok_k[:, None] * H + h) * K + offs_k[None, :]
                    b_w = tl.load(w_ptrs, mask=m_k_valid[:, None], other=0.0)
                    b_u = tl.load(U_buf + ku_base + offs_t[:, None] * K + offs_k[None, :])
                    proj = tl.dot(b_q, tl.trans(b_w), allow_tf32=False).to(b_u.dtype)
                    b_q = (b_q.to(tl.float32) - tl.dot(proj, b_u, allow_tf32=False)).to(b_q.dtype)

            # Pass 2: left-to-right adjoint propagation
            do_ptrs = dO + (b * T * HQ + tok_q[:, None] * HQ + hq) * V_DIM + offs_v[None, :]
            b_do = tl.load(do_ptrs, mask=m_q_valid[:, None], other=0.0)
            b_lse = tl.load(LSE + b * T * HQ + tok_q * HQ + hq, mask=m_q_valid, other=0.0).to(tl.float32)
            b_delta = tl.load(Delta + b * T * HQ + tok_q * HQ + hq, mask=m_q_valid, other=0.0).to(tl.float32)

            if USE_G:
                c_q = tl.load(C_cum + b * T * HQ + tok_q * HQ + hq, mask=m_q_valid, other=0.0).to(tl.float32)
            else:
                c_q = tl.zeros([BT], dtype=tl.float32)

            if HAS_B_INIT:
                bi_ptrs = B_init_in + (b * T * HQ + tok_q[:, None] * HQ + hq) * K + offs_k[None, :]
                b_dq = tl.load(bi_ptrs, mask=m_q_valid[:, None], other=0.0).to(tl.float32)
            else:
                b_dq = tl.zeros([BT, K], dtype=tl.float32)

            b_dc_q = tl.zeros([BT], dtype=tl.float32)

            for ck in range(fc, cq):
                loc_k = ck - fc
                tok_k = bos + loc_k * BT + offs_t
                m_k_valid = tok_k < eos

                ku_base = ((b * NT_total + ck) * H + h) * BT * K
                if ck > fc or HAS_B_INIT:
                    w_ptrs = W + (b * T * H + tok_k[:, None] * H + h) * K + offs_k[None, :]
                    b_w = tl.load(w_ptrs, mask=m_k_valid[:, None], other=0.0)
                    b_u = tl.load(U_buf + ku_base + offs_t[:, None] * K + offs_k[None, :])
                    proj_u = tl.dot(b_dq.to(b_u.dtype), tl.trans(b_u), allow_tf32=False).to(b_w.dtype)
                    b_dq = b_dq - tl.dot(proj_u, b_w, allow_tf32=False)

                ws_ptr = X_ws + ws_bhq_base + ck * BT * K + offs_t[:, None] * K + offs_k[None, :]
                b_x = tl.load(ws_ptr)
                b_k_arr = tl.load(K_arrow + ku_base + offs_t[:, None] * K + offs_k[None, :])
                v_ptrs = V + (b * T * H + tok_k[:, None] * H + h) * V_DIM + offs_v[None, :]
                b_v = tl.load(v_ptrs, mask=m_k_valid[:, None], other=0.0)

                z = tl.dot(b_x, tl.trans(b_k_arr), allow_tf32=False) * scale
                if USE_G:
                    c_k = tl.load(C_cum + b * T * HQ + tok_k * HQ + hq, mask=m_k_valid, other=0.0).to(tl.float32)
                    z = z + (c_q[:, None] - c_k[None, :])
                m_pair = m_q_valid[:, None] & m_k_valid[None, :]
                p = tl.where(m_pair, tl.exp(z - b_lse[:, None]), 0.0)
                dp = tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
                dz = p * (dp - b_delta[:, None])

                if USE_G:
                    b_dc_q += tl.sum(dz, axis=1)

                b_dq += tl.dot((dz * scale).to(b_k_arr.dtype), b_k_arr, allow_tf32=False)

            tl.store(dQ_arrow + qa_base + offs_t[:, None] * K + offs_k[None, :], b_dq.to(dQ_arrow.dtype.element_ty))
            if USE_G:
                tl.store(dC_q_inter + b * T * HQ + tok_q * HQ + hq, b_dc_q, mask=m_q_valid)


@triton.jit
def _path_inter_chunk_key_duality_worker_bwd_kernel(
    Q_arrow,
    K_arrow,
    V,
    W,
    U_buf,
    C_cum,
    dO,
    LSE,
    Delta,
    Y_ws,
    dK_arrow,
    dV_inter,
    dC_k_inter,
    dW_priv,
    dU_priv,
    Chunk_bos,
    Chunk_eos,
    Chunk_first_c,
    Chunk_last_c,
    Chunk_local_c,
    NT_total,
    T,
    scale,
    NW_K: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
):
    """
    Lock-Free (0 atomics, 0 [128, 128] matrices, O(T) linear HBM!) Key-Duality
    Inter-Chunk Backward Kernel:
    Grid = (NW_K, B * H). Worker `pid_w` processes key chunks `ck in range(pid_w, NT_total, NW_K)`:
      1. Forward key propagation `K_tilde <- K_tilde - (K_tilde @ U_m^T) @ W_m` for m = ck+1 .. lc-1
         stored in private linear workspace `Y_ws[pid_w, pid_bh, m, 32, K]`.
      2. Reverse adjoint propagation for m = lc-1 .. ck+1:
         - Accumulates transition gradients into private slice `dW_priv[pid_w, pid_bh, m]`,
           `dU_priv[pid_w, pid_bh, m]` (zero atomics!)
         - Loops over G query heads in SRAM to accumulate `b_dv`, `b_dk`, and `dC_k_inter`.
    """
    pid_w = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V_DIM)

    ws_bh_base = (pid_w * (tl.num_programs(1)) + pid_bh) * NT_total * BT * K

    n_stripes = tl.cdiv(NT_total, NW_K)
    for s in range(0, n_stripes):
        ck = s * NW_K + tl.where((s & 1) == 0, pid_w, NW_K - 1 - pid_w)
        if ck < NT_total:
            bos = tl.load(Chunk_bos + ck)
            eos = tl.load(Chunk_eos + ck)
            fc = tl.load(Chunk_first_c + ck)
            lc = tl.load(Chunk_last_c + ck)
            loc_k = tl.load(Chunk_local_c + ck)
            tok_k = bos + loc_k * BT + offs_t
            m_k_valid = tok_k < eos

            ku_ck_base = ((b * NT_total + ck) * H + h) * BT * K
            b_kt = tl.load(K_arrow + ku_ck_base + offs_t[:, None] * K + offs_k[None, :])

            # Pass 1: left-to-right key propagation Y_{ck -> m}
            for m in range(ck + 1, lc):
                loc_m = m - fc
                tok_m = bos + loc_m * BT + offs_t
                m_m_valid = tok_m < eos

                ws_ptr = Y_ws + ws_bh_base + m * BT * K + offs_t[:, None] * K + offs_k[None, :]
                tl.store(ws_ptr, b_kt)

                if m < lc - 1:
                    ku_m_base = ((b * NT_total + m) * H + h) * BT * K
                    w_ptrs = W + (b * T * H + tok_m[:, None] * H + h) * K + offs_k[None, :]
                    b_w = tl.load(w_ptrs, mask=m_m_valid[:, None], other=0.0)
                    b_u = tl.load(U_buf + ku_m_base + offs_t[:, None] * K + offs_k[None, :])
                    proj = tl.dot(b_kt, tl.trans(b_u), allow_tf32=False).to(b_w.dtype)
                    b_kt = (b_kt.to(tl.float32) - tl.dot(proj, b_w, allow_tf32=False)).to(b_kt.dtype)

            # Pass 2: right-to-left adjoint propagation
            v_ptrs = V + (b * T * H + tok_k[:, None] * H + h) * V_DIM + offs_v[None, :]
            b_v = tl.load(v_ptrs, mask=m_k_valid[:, None], other=0.0)

            b_dk = tl.zeros([BT, K], dtype=tl.float32)
            b_dv = tl.zeros([BT, V_DIM], dtype=tl.float32)

            n_after = lc - (ck + 1)
            for step in range(0, n_after):
                m = lc - 1 - step
                loc_m = m - fc
                tok_m = bos + loc_m * BT + offs_t
                m_m_valid = tok_m < eos

                ws_ptr = Y_ws + ws_bh_base + m * BT * K + offs_t[:, None] * K + offs_k[None, :]
                b_y = tl.load(ws_ptr)

                if m < lc - 1:
                    ku_m_base = ((b * NT_total + m) * H + h) * BT * K
                    w_ptrs = W + (b * T * H + tok_m[:, None] * H + h) * K + offs_k[None, :]
                    b_w = tl.load(w_ptrs, mask=m_m_valid[:, None], other=0.0)
                    b_u = tl.load(U_buf + ku_m_base + offs_t[:, None] * K + offs_k[None, :])

                    # s_gw = b_dk @ W_m^T ([32, 32]), s_yu = b_y @ U_m^T ([32, 32])
                    s_gw = tl.dot(b_dk.to(b_w.dtype), tl.trans(b_w), allow_tf32=False)
                    s_yu = tl.dot(b_y, tl.trans(b_u), allow_tf32=False)

                    priv_ptr = ws_bh_base + m * BT * K + offs_t[:, None] * K + offs_k[None, :]
                    cur_du = tl.load(dU_priv + priv_ptr)
                    cur_dw = tl.load(dW_priv + priv_ptr)
                    cur_du = cur_du - tl.dot(tl.trans(s_gw.to(b_y.dtype)), b_y, allow_tf32=False)
                    cur_dw = cur_dw - tl.dot(tl.trans(s_yu.to(b_w.dtype)), b_dk.to(b_w.dtype), allow_tf32=False)
                    tl.store(dU_priv + priv_ptr, cur_du)
                    tl.store(dW_priv + priv_ptr, cur_dw)

                    b_dk = b_dk - tl.dot(s_gw.to(b_u.dtype), b_u, allow_tf32=False)

                for g_idx in range(0, G):
                    hq = h * G + g_idx
                    qa_m_base = ((b * NT_total + m) * HQ + hq) * BT * K
                    b_qm = tl.load(Q_arrow + qa_m_base + offs_t[:, None] * K + offs_k[None, :])

                    do_ptrs = dO + (b * T * HQ + tok_m[:, None] * HQ + hq) * V_DIM + offs_v[None, :]
                    b_do = tl.load(do_ptrs, mask=m_m_valid[:, None], other=0.0)
                    b_lse = tl.load(LSE + b * T * HQ + tok_m * HQ + hq, mask=m_m_valid, other=0.0).to(tl.float32)
                    b_delta = tl.load(Delta + b * T * HQ + tok_m * HQ + hq, mask=m_m_valid, other=0.0).to(tl.float32)

                    z = tl.dot(b_qm, tl.trans(b_y), allow_tf32=False) * scale
                    if USE_G:
                        c_m = tl.load(C_cum + b * T * HQ + tok_m * HQ + hq, mask=m_m_valid, other=0.0).to(tl.float32)
                        c_k = tl.load(C_cum + b * T * HQ + tok_k * HQ + hq, mask=m_k_valid, other=0.0).to(tl.float32)
                        z = z + (c_m[:, None] - c_k[None, :])
                    m_pair = m_m_valid[:, None] & m_k_valid[None, :]
                    p = tl.where(m_pair, tl.exp(z - b_lse[:, None]), 0.0)
                    b_dv += tl.dot(tl.trans(p.to(b_do.dtype)), b_do, allow_tf32=False)

                    dp = tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
                    dz = p * (dp - b_delta[:, None])
                    b_dk += tl.dot(tl.trans((dz * scale).to(b_qm.dtype)), b_qm, allow_tf32=False)

                    if USE_G:
                        dc_k_ptr = dC_k_inter + b * T * HQ + tok_k * HQ + hq
                        prev_dc = tl.load(dc_k_ptr, mask=m_k_valid, other=0.0).to(tl.float32)
                        tl.store(dc_k_ptr, prev_dc - tl.sum(dz, axis=0), mask=m_k_valid)

            tl.store(dK_arrow + ku_ck_base + offs_t[:, None] * K + offs_k[None, :], b_dk.to(dK_arrow.dtype.element_ty))
            dv_base = ((b * NT_total + ck) * H + h) * BT * V_DIM
            tl.store(dV_inter + dv_base + offs_t[:, None] * V_DIM + offs_v[None, :], b_dv.to(dV_inter.dtype.element_ty))


@triton.jit
def _path_reduce_dw_du_priv_kernel(
    dW_priv,
    dU_priv,
    dW_inter,
    dU_inter,
    NT_total,
    B_H,
    NW_K: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    """Deterministic atomic-free reduction across the `NW_K` private key-worker slots."""
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, K)
    cell_offs = offs_t[:, None] * K + offs_k[None, :]

    acc_dw = tl.zeros([BT, K], dtype=tl.float32)
    acc_du = tl.zeros([BT, K], dtype=tl.float32)

    for w in range(0, NW_K):
        priv_base = ((w * B_H + pid_bh) * NT_total + pid_c) * BT * K
        acc_dw += tl.load(dW_priv + priv_base + cell_offs)
        acc_du += tl.load(dU_priv + priv_base + cell_offs)

    # Note: dW_inter and dU_inter are indexed as [B, NT_total, H, BT, K] -> ((b * NT_total + pid_c) * H + h)
    # Wait: let's match the exact [B, NT_total, H, BT, K] layout!
    out_base = (pid_bh * NT_total + pid_c) * BT * K
    tl.store(dW_inter + out_base + cell_offs, acc_dw)
    tl.store(dU_inter + out_base + cell_offs, acc_du)


@triton.jit
def _path_intra_chunk_score_dv_bwd_kernel(
    Q,
    K_in,
    V,
    W,
    Beta,
    C_cum,
    A_inv,
    dO,
    LSE,
    Delta,
    dV_inter,
    dC_q_inter,
    dC_k_inter,
    P_intra_buf,
    dB_raw_out,
    dV_out,
    dC_total_out,
    Chunk_bos,
    Chunk_eos,
    Chunk_local_c,
    NT_total,
    T,
    scale,
    G_rt,
    K_rt,
    V_rt,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
):
    """
    Stage 1A of Lock-Free Intra-Chunk Backward (0 atomics, 0 spills at K=64 and K=128):
    Pass 1: Recomputes intra-chunk causal scores `P` by streaming [32, 64] K-slices (`K_rt`),
            computes `dp` by streaming [32, 64] V-slices (`V_rt`), writes `P_intra_buf` (L1/L2)
            and `dB_raw_out = dz * scale` ([B, NT_total, HQ, 32, 32]).
    Pass 2: Streams [32, 64] V-slices (`V_rt`) after Pass 1 registers (`b_T`, `m_wbk`, `b_raw`, `dp`)
            are dead, accumulating `dV` with only a single [32, 64] FP32 accumulator live at a time.
    """
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_idx = bos + loc_c * BT + offs_t
    m_valid = tok_idx < eos
    BK: tl.constexpr = 64
    BV: tl.constexpr = 64
    offs_bk = tl.arange(0, BK)
    offs_bv = tl.arange(0, BV)
    m_tril = offs_t[:, None] >= offs_t[None, :]
    m_strict = offs_t[:, None] > offs_t[None, :]

    b_beta = tl.load(Beta + b * T * H + tok_idx * H + h, mask=m_valid, other=0.0).to(tl.float32)

    a_base = ((b * NT_total + pid_c) * H + h) * BT * BT
    b_A = tl.load(A_inv + a_base + offs_t[:, None] * BT + offs_t[None, :]).to(tl.float32)
    b_T = (b_A * b_beta[None, :]).to(Q.dtype.element_ty)

    m_wbk_acc = tl.zeros([BT, BT], dtype=tl.float32)
    for k_off in range(0, K_rt, BK):
        col_k = k_off + offs_bk
        b_w_s = tl.load(W + (b * T * H + tok_idx[:, None] * H + h) * K + col_k[None, :], mask=m_valid[:, None], other=0.0)
        b_k_s = tl.load(K_in + (b * T * H + tok_idx[:, None] * H + h) * K + col_k[None, :], mask=m_valid[:, None], other=0.0)
        m_wbk_acc += tl.dot(b_w_s, tl.trans(b_k_s), allow_tf32=False)
    m_wbk = tl.where(m_strict, m_wbk_acc, 0.0).to(Q.dtype.element_ty)

    v_row_off = (b * T * H + tok_idx[:, None] * H + h) * V_DIM
    dv_base = ((b * NT_total + pid_c) * H + h) * BT * V_DIM

    for g_idx in range(0, G_rt):
        hq = h * G_rt + g_idx
        m_qw_acc = tl.zeros([BT, BT], dtype=tl.float32)
        for k_off in range(0, K_rt, BK):
            col_k = k_off + offs_bk
            b_q_s = tl.load(Q + (b * T * HQ + tok_idx[:, None] * HQ + hq) * K + col_k[None, :], mask=m_valid[:, None], other=0.0)
            b_w_s = tl.load(W + (b * T * H + tok_idx[:, None] * H + h) * K + col_k[None, :], mask=m_valid[:, None], other=0.0)
            m_qw_acc += tl.dot(b_q_s, tl.trans(b_w_s), allow_tf32=False)

        m_qw = tl.where(m_tril, m_qw_acc, 0.0).to(Q.dtype.element_ty)
        n_qwt = tl.dot(m_qw, b_T, allow_tf32=False).to(Q.dtype.element_ty)
        b_raw = -tl.dot(n_qwt, m_wbk, allow_tf32=False)

        for k_off in range(0, K_rt, BK):
            col_k = k_off + offs_bk
            b_q_s = tl.load(Q + (b * T * HQ + tok_idx[:, None] * HQ + hq) * K + col_k[None, :], mask=m_valid[:, None], other=0.0)
            b_k_s = tl.load(K_in + (b * T * H + tok_idx[:, None] * H + h) * K + col_k[None, :], mask=m_valid[:, None], other=0.0)
            b_raw += tl.dot(b_q_s, tl.trans(b_k_s), allow_tf32=False)

        b_lse = tl.load(LSE + b * T * HQ + tok_idx * HQ + hq, mask=m_valid, other=0.0).to(tl.float32)
        b_delta = tl.load(Delta + b * T * HQ + tok_idx * HQ + hq, mask=m_valid, other=0.0).to(tl.float32)
        z = b_raw * scale
        if USE_G:
            c_q = tl.load(C_cum + b * T * HQ + tok_idx * HQ + hq, mask=m_valid, other=0.0).to(tl.float32)
            z = z + (c_q[:, None] - c_q[None, :])
        m_causal_valid = m_tril & m_valid[:, None] & m_valid[None, :]
        p = tl.where(m_causal_valid, tl.exp(z - b_lse[:, None]), 0.0)
        p_bf = p.to(V.dtype.element_ty)

        db_base = ((b * NT_total + pid_c) * HQ + hq) * BT * BT
        tl.store(P_intra_buf + db_base + offs_t[:, None] * BT + offs_t[None, :], p_bf)

        dp = tl.zeros([BT, BT], dtype=tl.float32)
        for v_off in range(0, V_rt, BV):
            col_v = v_off + offs_bv
            b_do = tl.load(dO + (b * T * HQ + tok_idx[:, None] * HQ + hq) * V_DIM + col_v[None, :], mask=m_valid[:, None], other=0.0)
            b_v = tl.load(V + v_row_off + col_v[None, :], mask=m_valid[:, None], other=0.0)
            dp += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)

        dz = tl.where(m_causal_valid, p * (dp - b_delta[:, None]), 0.0)
        if USE_G:
            dc_q = tl.load(dC_q_inter + b * T * HQ + tok_idx * HQ + hq, mask=m_valid, other=0.0).to(tl.float32)
            dc_k = tl.load(dC_k_inter + b * T * HQ + tok_idx * HQ + hq, mask=m_valid, other=0.0).to(tl.float32)
            dc_tot = dc_q + dc_k + tl.sum(dz, axis=1) - tl.sum(dz, axis=0)
            tl.store(dC_total_out + b * T * HQ + tok_idx * HQ + hq, dc_tot, mask=m_valid)

        tl.store(dB_raw_out + db_base + offs_t[:, None] * BT + offs_t[None, :], dz * scale)

    for v_off in range(0, V_rt, BV):
        col_v = v_off + offs_bv
        b_dv = tl.load(dV_inter + dv_base + offs_t[:, None] * V_DIM + col_v[None, :]).to(tl.float32)
        for g_idx in range(0, G_rt):
            hq = h * G_rt + g_idx
            db_base = ((b * NT_total + pid_c) * HQ + hq) * BT * BT
            p_bf = tl.load(P_intra_buf + db_base + offs_t[:, None] * BT + offs_t[None, :])
            b_do = tl.load(dO + (b * T * HQ + tok_idx[:, None] * HQ + hq) * V_DIM + col_v[None, :], mask=m_valid[:, None], other=0.0)
            b_dv += tl.dot(tl.trans(p_bf), b_do, allow_tf32=False)
        tl.store(dV_out + v_row_off + col_v[None, :], b_dv.to(dV_out.dtype.element_ty), mask=m_valid[:, None])


@triton.jit
def _path_intra_chunk_ut_matrices_bwd_kernel(
    Q,
    K_in,
    W,
    Beta,
    A_inv,
    U_buf,
    dB_raw_in,
    dQ_arrow,
    dK_arrow,
    dU_inter,
    dM_qw_out,
    N_qwt_out,
    dM_wbk_out,
    N_twbk_out,
    dL_out,
    dBeta_out,
    Chunk_bos,
    Chunk_eos,
    Chunk_local_c,
    NT_total,
    T,
    G_rt,
    K_rt,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    """
    Stage 1B of Lock-Free Intra-Chunk Backward (0 atomics, 0 spills at K=64 and K=128):
    Computes all [32, 32] UT Gram/triangular reversal matrices (`dM_qw`, `N_qwt`, `dM_wbk`,
    `N_twbk`, `dL_strict`) and `dBeta_out` by streaming [32, 64] K-slices via runtime `K_rt`
    loops so no [32, 128] tile is ever held in registers.
    """
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_idx = bos + loc_c * BT + offs_t
    m_valid = tok_idx < eos
    BK: tl.constexpr = 64
    offs_bk = tl.arange(0, BK)
    m_tril = offs_t[:, None] >= offs_t[None, :]
    m_strict = offs_t[:, None] > offs_t[None, :]

    ku_base = ((b * NT_total + pid_c) * H + h) * BT * K
    inter_base = (pid_bh * NT_total + pid_c) * BT * K
    a_base = ((b * NT_total + pid_c) * H + h) * BT * BT

    m_wbk_acc = tl.zeros([BT, BT], dtype=tl.float32)
    b_dN_twbk = tl.zeros([BT, BT], dtype=tl.float32)
    du_wt_acc = tl.zeros([BT, BT], dtype=tl.float32)
    for kb in range(0, K_rt, BK):
        ok = kb + offs_bk
        b_w_s = tl.load(W + (b * T * H + tok_idx[:, None] * H + h) * K + ok[None, :], mask=m_valid[:, None], other=0.0)
        b_k_s = tl.load(K_in + (b * T * H + tok_idx[:, None] * H + h) * K + ok[None, :], mask=m_valid[:, None], other=0.0)
        m_wbk_acc += tl.dot(b_w_s, tl.trans(b_k_s), allow_tf32=False)
        b_dk_arr_s = tl.load(dK_arrow + ku_base + offs_t[:, None] * K + ok[None, :]).to(b_w_s.dtype)
        b_dN_twbk -= tl.dot(b_w_s, tl.trans(b_dk_arr_s), allow_tf32=False)
        b_du_s = tl.load(dU_inter + inter_base + offs_t[:, None] * K + ok[None, :]).to(b_w_s.dtype)
        du_wt_acc += tl.dot(b_du_s, tl.trans(b_w_s), allow_tf32=False)

    b_beta0 = tl.load(Beta + b * T * H + tok_idx * H + h, mask=m_valid, other=0.0).to(tl.float32)
    b_A0 = tl.load(A_inv + a_base + offs_t[:, None] * BT + offs_t[None, :]).to(tl.float32)
    b_T = (b_A0 * b_beta0[None, :]).to(W.dtype.element_ty)

    m_wbk = tl.where(m_strict, m_wbk_acc, 0.0).to(W.dtype.element_ty)
    n_twbk = tl.dot(b_T, m_wbk, allow_tf32=False)
    tl.store(N_twbk_out + a_base + offs_t[:, None] * BT + offs_t[None, :], n_twbk)

    b_dM_wbk = tl.dot(tl.trans(b_T), b_dN_twbk.to(b_T.dtype), allow_tf32=False)
    b_dT = tl.dot(b_dN_twbk.to(m_wbk.dtype), tl.trans(m_wbk), allow_tf32=False) + du_wt_acc

    for g_idx in range(0, G_rt):
        hq = h * G_rt + g_idx
        qa_base = ((b * NT_total + pid_c) * HQ + hq) * BT * K
        db_base = ((b * NT_total + pid_c) * HQ + hq) * BT * BT

        m_qw_acc = tl.zeros([BT, BT], dtype=tl.float32)
        for kb in range(0, K_rt, BK):
            ok = kb + offs_bk
            b_q_s = tl.load(Q + (b * T * HQ + tok_idx[:, None] * HQ + hq) * K + ok[None, :], mask=m_valid[:, None], other=0.0)
            b_w_s = tl.load(W + (b * T * H + tok_idx[:, None] * H + h) * K + ok[None, :], mask=m_valid[:, None], other=0.0)
            m_qw_acc += tl.dot(b_q_s, tl.trans(b_w_s), allow_tf32=False)

        dqa_ut_acc = tl.zeros([BT, BT], dtype=tl.float32)
        dqa_wt_acc = tl.zeros([BT, BT], dtype=tl.float32)
        for kb in range(0, K_rt, BK):
            ok = kb + offs_bk
            b_dq_arr_s = tl.load(dQ_arrow + qa_base + offs_t[:, None] * K + ok[None, :]).to(W.dtype.element_ty)
            b_u_s = tl.load(U_buf + ku_base + offs_t[:, None] * K + ok[None, :])
            dqa_ut_acc += tl.dot(b_dq_arr_s, tl.trans(b_u_s), allow_tf32=False)
            b_w_s = tl.load(W + (b * T * H + tok_idx[:, None] * H + h) * K + ok[None, :], mask=m_valid[:, None], other=0.0)
            dqa_wt_acc += tl.dot(b_dq_arr_s, tl.trans(b_w_s), allow_tf32=False)

        db_raw = tl.load(dB_raw_in + db_base + offs_t[:, None] * BT + offs_t[None, :]).to(W.dtype.element_ty)
        m_qw = tl.where(m_tril, m_qw_acc, 0.0).to(W.dtype.element_ty)
        n_qwt = tl.dot(m_qw, b_T, allow_tf32=False)
        tl.store(N_qwt_out + db_base + offs_t[:, None] * BT + offs_t[None, :], n_qwt)

        b_dN_qwt = -tl.dot(db_raw, tl.trans(m_wbk), allow_tf32=False)
        b_dM_wbk -= tl.dot(tl.trans(n_qwt.to(W.dtype.element_ty)), db_raw, allow_tf32=False)
        b_dM_qw = tl.where(m_tril, -dqa_ut_acc + tl.dot(b_dN_qwt.to(b_T.dtype), tl.trans(b_T), allow_tf32=False), 0.0)
        b_dT += tl.dot(tl.trans(m_qw), (b_dN_qwt - dqa_wt_acc).to(m_qw.dtype), allow_tf32=False)
        tl.store(dM_qw_out + db_base + offs_t[:, None] * BT + offs_t[None, :], b_dM_qw)

    b_dM_wbk = tl.where(m_strict, b_dM_wbk, 0.0)
    tl.store(dM_wbk_out + a_base + offs_t[:, None] * BT + offs_t[None, :], b_dM_wbk)

    b_beta = tl.load(Beta + b * T * H + tok_idx * H + h, mask=m_valid, other=0.0).to(tl.float32)
    b_A2 = tl.load(A_inv + a_base + offs_t[:, None] * BT + offs_t[None, :]).to(tl.float32)
    b_dbeta = tl.sum(b_dT * b_A2, axis=0)
    b_dA = tl.where(m_tril, b_dT * b_beta[None, :], 0.0)
    b_AT = tl.trans(b_A2).to(W.dtype.element_ty)
    tmp_a = tl.dot(b_AT, b_dA.to(W.dtype.element_ty), allow_tf32=False).to(W.dtype.element_ty)
    b_dL = tl.where(m_strict, -tl.dot(tmp_a, b_AT, allow_tf32=False), 0.0)
    tl.store(dL_out + a_base + offs_t[:, None] * BT + offs_t[None, :], b_dL)

    for kb in range(0, K_rt, BK):
        ok = kb + offs_bk
        b_w_s = tl.load(W + (b * T * H + tok_idx[:, None] * H + h) * K + ok[None, :], mask=m_valid[:, None], other=0.0)
        b_dwb_s = tl.dot(b_dL.to(b_w_s.dtype), b_w_s, allow_tf32=False)
        b_dbeta += tl.sum(b_dwb_s * b_w_s.to(tl.float32), axis=1)

    tl.store(dBeta_out + b * T * H + tok_idx * H + h, b_dbeta.to(dBeta_out.dtype.element_ty), mask=m_valid)


@triton.jit
def _path_intra_chunk_ut_vectors_bwd_kernel(
    Q,
    K_in,
    W,
    Beta,
    A_inv,
    dB_raw_in,
    dM_qw_in,
    N_qwt_in,
    dM_wbk_in,
    N_twbk_in,
    dL_in,
    dQ_arrow,
    dK_arrow,
    dW_inter,
    dU_inter,
    dQ_out,
    dK_out,
    dW_out,
    Chunk_bos,
    Chunk_eos,
    Chunk_local_c,
    NT_total,
    T,
    G_rt,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    """
    Stage 1C of Lock-Free Intra-Chunk Backward (0 atomics, 0 spills, 168 regs at K=128):
    Grid = (NT_total, B * H, K // 64). Each program computes one [32, 64] column slice of
    `dQ_out`, `dK_out`, and `dW_out`. Uses associative `N_qwt^T @ dQ_arrow` elimination so
    `b_du` is not accumulated across G, and sequences `dK_out` before `dW_out` so `b_dk`
    and `b_dw` never overlap in live registers.
    """
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_k = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H

    bos = tl.load(Chunk_bos + pid_c)
    eos = tl.load(Chunk_eos + pid_c)
    loc_c = tl.load(Chunk_local_c + pid_c)

    offs_t = tl.arange(0, BT)
    tok_idx = bos + loc_c * BT + offs_t
    m_valid = tok_idx < eos
    BK: tl.constexpr = 64
    offs_k = pid_k * BK + tl.arange(0, BK)

    ku_base = ((b * NT_total + pid_c) * H + h) * BT * K
    a_base = ((b * NT_total + pid_c) * H + h) * BT * BT

    # Phase 1: Compute dQ_out and dK_out ONLY (b_dw is not live yet)
    b_w = tl.load(W + (b * T * H + tok_idx[:, None] * H + h) * K + offs_k[None, :], mask=m_valid[:, None], other=0.0)
    b_k = tl.load(K_in + (b * T * H + tok_idx[:, None] * H + h) * K + offs_k[None, :], mask=m_valid[:, None], other=0.0)
    b_dk = tl.load(dK_arrow + ku_base + offs_t[:, None] * K + offs_k[None, :]).to(tl.float32)

    for g_idx in range(0, G_rt):
        hq = h * G_rt + g_idx
        qa_base = ((b * NT_total + pid_c) * HQ + hq) * BT * K
        db_base = ((b * NT_total + pid_c) * HQ + hq) * BT * BT
        b_dq_arr = tl.load(dQ_arrow + qa_base + offs_t[:, None] * K + offs_k[None, :]).to(tl.float32)
        db_raw = tl.load(dB_raw_in + db_base + offs_t[:, None] * BT + offs_t[None, :]).to(b_k.dtype)
        b_dM_qw = tl.load(dM_qw_in + db_base + offs_t[:, None] * BT + offs_t[None, :]).to(b_w.dtype)
        b_dq = b_dq_arr + tl.dot(db_raw, b_k, allow_tf32=False) + tl.dot(b_dM_qw, b_w, allow_tf32=False)
        tl.store(dQ_out + (b * T * HQ + tok_idx[:, None] * HQ + hq) * K + offs_k[None, :], b_dq.to(dQ_out.dtype.element_ty), mask=m_valid[:, None])

        b_q = tl.load(Q + (b * T * HQ + tok_idx[:, None] * HQ + hq) * K + offs_k[None, :], mask=m_valid[:, None], other=0.0)
        b_dk += tl.dot(tl.trans(db_raw), b_q, allow_tf32=False)

    b_dM_wbk = tl.load(dM_wbk_in + a_base + offs_t[:, None] * BT + offs_t[None, :]).to(b_w.dtype)
    b_dk += tl.dot(tl.trans(b_dM_wbk), b_w, allow_tf32=False)
    tl.store(dK_out + (b * T * H + tok_idx[:, None] * H + h) * K + offs_k[None, :], b_dk.to(dK_out.dtype.element_ty), mask=m_valid[:, None])

    # Phase 2: b_dk is dead; load b_dw and compute dW_out
    inter_base = (pid_bh * NT_total + pid_c) * BT * K
    b_dw = tl.load(dW_inter + inter_base + offs_t[:, None] * K + offs_k[None, :]).to(tl.float32)
    b_dw += tl.dot(b_dM_wbk, b_k, allow_tf32=False)

    for g_idx in range(0, G_rt):
        hq = h * G_rt + g_idx
        qa_base = ((b * NT_total + pid_c) * HQ + hq) * BT * K
        db_base = ((b * NT_total + pid_c) * HQ + hq) * BT * BT
        b_q = tl.load(Q + (b * T * HQ + tok_idx[:, None] * HQ + hq) * K + offs_k[None, :], mask=m_valid[:, None], other=0.0)
        b_dM_qw = tl.load(dM_qw_in + db_base + offs_t[:, None] * BT + offs_t[None, :]).to(b_w.dtype)
        b_dw += tl.dot(tl.trans(b_dM_qw), b_q, allow_tf32=False)
        b_dq_arr = tl.load(dQ_arrow + qa_base + offs_t[:, None] * K + offs_k[None, :]).to(b_w.dtype)
        n_qwt = tl.load(N_qwt_in + db_base + offs_t[:, None] * BT + offs_t[None, :]).to(b_w.dtype)
        b_dw -= tl.dot(tl.trans(n_qwt), b_dq_arr, allow_tf32=False)

    b_dk_arr = tl.load(dK_arrow + ku_base + offs_t[:, None] * K + offs_k[None, :]).to(b_w.dtype)
    n_twbk = tl.load(N_twbk_in + a_base + offs_t[:, None] * BT + offs_t[None, :]).to(b_w.dtype)
    b_dw -= tl.dot(n_twbk, b_dk_arr, allow_tf32=False)

    b_du = tl.load(dU_inter + inter_base + offs_t[:, None] * K + offs_k[None, :]).to(b_w.dtype)
    b_beta = tl.load(Beta + b * T * H + tok_idx * H + h, mask=m_valid, other=0.0).to(tl.float32)
    b_A = tl.load(A_inv + a_base + offs_t[:, None] * BT + offs_t[None, :]).to(tl.float32)
    b_T = (b_A * b_beta[None, :]).to(b_w.dtype)
    b_dw += tl.dot(tl.trans(b_T), b_du, allow_tf32=False)

    b_dL = tl.load(dL_in + a_base + offs_t[:, None] * BT + offs_t[None, :]).to(b_w.dtype)
    b_wb = (b_w.to(tl.float32) * b_beta[:, None]).to(b_w.dtype)
    b_dwb = tl.dot(b_dL, b_w, allow_tf32=False)
    b_dw += tl.dot(tl.trans(b_dL), b_wb, allow_tf32=False) + b_dwb * b_beta[:, None]
    tl.store(dW_out + (b * T * H + tok_idx[:, None] * H + h) * K + offs_k[None, :], b_dw.to(dW_out.dtype.element_ty), mask=m_valid[:, None])


def _run_path_fox_ut_duality_fwd_raw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor],
    scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
    save_x_init: bool = False,
):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V_DIM = v.shape[3]
    G = HQ // H
    BT = 32
    bos, eos, fc, lc, loc, nt_total = build_chunk_meta(B, T, q.device, cu_seqlens, BT=BT)

    use_g = g is not None
    if use_g:
        c_cum = path_fox_gate_fwd_cumsum_raw(g, bos, eos, fc, lc, apply_logsigmoid=False)
    else:
        c_cum = torch.empty((1,), dtype=torch.float32, device=q.device)

    A_inv = torch.empty((B, nt_total, H, BT, BT), dtype=torch.float32, device=q.device)
    Q_arrow = torch.empty((B, nt_total, HQ, BT, K), dtype=q.dtype, device=q.device)
    K_arrow = torch.empty((B, nt_total, H, BT, K), dtype=k.dtype, device=q.device)
    U_buf = torch.empty((B, nt_total, H, BT, K), dtype=w.dtype, device=q.device)
    O_intra = torch.empty((B, nt_total, HQ, BT, V_DIM), dtype=torch.float32, device=q.device)
    M_intra = torch.empty((B, nt_total, HQ, BT), dtype=torch.float32, device=q.device)
    L_intra = torch.empty((B, nt_total, HQ, BT), dtype=torch.float32, device=q.device)
    O_out = torch.empty((B, T, HQ, V_DIM), dtype=v.dtype, device=q.device)
    LSE_out = torch.empty((B, T, HQ), dtype=torch.float32, device=q.device)
    X_init = torch.empty_like(q) if save_x_init else torch.empty((1,), dtype=q.dtype, device=q.device)

    _path_ut_solve_tril_32x32_fwd_kernel[(nt_total, B * H)](
        w, beta, A_inv, bos, eos, loc, nt_total, T,
        H=H, K=K, BT=BT,
        num_warps=4, num_stages=1,
    )
    _path_intra_chunk_ut_fwd_kernel[(nt_total, B * H)](
        q, k, v, w, beta, c_cum, A_inv, Q_arrow, K_arrow, U_buf,
        O_intra, M_intra, L_intra, bos, eos, loc, nt_total, T, float(scale),
        HQ=HQ, H=H, G=G, K=K, V_DIM=V_DIM, BT=BT, USE_G=use_g,
        num_warps=4, num_stages=1,
    )
    _path_inter_chunk_streaming_fwd_kernel[(nt_total, B * HQ)](
        Q_arrow, K_arrow, v, w, U_buf, c_cum, O_intra, M_intra, L_intra,
        O_out, LSE_out, X_init, bos, eos, fc, loc, nt_total, T, float(scale),
        HQ=HQ, H=H, G=G, K=K, V_DIM=V_DIM, BT=BT, USE_G=use_g, SAVE_X_INIT=save_x_init,
        num_warps=4, num_stages=1,
    )
    return O_out, LSE_out, A_inv, Q_arrow, K_arrow, U_buf, c_cum, O_intra, M_intra, L_intra, X_init, bos, eos, fc, lc, loc, nt_total


def _run_path_fox_ut_duality_bwd_raw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor],
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    A_inv: torch.Tensor,
    Q_arrow: torch.Tensor,
    K_arrow: torch.Tensor,
    U_buf: torch.Tensor,
    c_cum: torch.Tensor,
    bos: torch.Tensor,
    eos: torch.Tensor,
    fc: torch.Tensor,
    lc: torch.Tensor,
    loc: torch.Tensor,
    nt_total: int,
    scale: float,
    b_init_ext: Optional[torch.Tensor] = None,
    dk_arrow_ext: Optional[torch.Tensor] = None,
    dv_ext: Optional[torch.Tensor] = None,
    dw_ext: Optional[torch.Tensor] = None,
    du_ext: Optional[torch.Tensor] = None,
    dc_ext: Optional[torch.Tensor] = None,
):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V_DIM = v.shape[3]
    G = HQ // H
    BT = 32
    use_g = g is not None

    n_rows = B * T * HQ
    delta = torch.empty((B, T, HQ), dtype=torch.float32, device=q.device)
    _path_bwd_preprocess_delta_kernel[(n_rows,)](
        o, do, delta, n_rows, V_DIM=V_DIM, num_warps=2, num_stages=1
    )

    nw_q = min(nt_total, max(8, 128 // max(1, B * HQ)))
    nw_k = min(nt_total, 16)

    dQ_arrow = torch.empty((B, nt_total, HQ, BT, K), dtype=torch.float32, device=q.device)
    dK_arrow = torch.empty((B, nt_total, H, BT, K), dtype=torch.float32, device=q.device)
    dV_inter = torch.empty((B, nt_total, H, BT, V_DIM), dtype=torch.float32, device=q.device)
    dW_inter = torch.empty((B * H, nt_total, BT, K), dtype=torch.float32, device=q.device)
    dU_inter = torch.empty((B * H, nt_total, BT, K), dtype=torch.float32, device=q.device)

    if use_g:
        dC_q_inter = torch.zeros((B, T, HQ), dtype=torch.float32, device=q.device)
        dC_k_inter = dc_ext.clone() if dc_ext is not None else torch.zeros((B, T, HQ), dtype=torch.float32, device=q.device)
        dC_total = torch.empty((B, T, HQ), dtype=torch.float32, device=q.device)
    else:
        dC_q_inter = torch.empty((1,), dtype=torch.float32, device=q.device)
        dC_k_inter = torch.empty((1,), dtype=torch.float32, device=q.device)
        dC_total = torch.empty((1,), dtype=torch.float32, device=q.device)

    has_b_init = b_init_ext is not None
    b_init_arg = b_init_ext if has_b_init else torch.empty((1,), dtype=torch.float32, device=q.device)

    X_ws = torch.empty((nw_q, B * HQ, nt_total, BT, K), dtype=q.dtype, device=q.device)
    _path_inter_chunk_query_worker_bwd_kernel[(nw_q, B * HQ)](
        Q_arrow, K_arrow, v, w, U_buf, c_cum, do, lse, delta,
        X_ws, dQ_arrow, dC_q_inter, b_init_arg,
        bos, eos, fc, loc, nt_total, T, float(scale),
        NW_Q=nw_q, HQ=HQ, H=H, G=G, K=K, V_DIM=V_DIM, BT=BT,
        USE_G=use_g, HAS_B_INIT=has_b_init,
        num_warps=4, num_stages=1,
    )
    del X_ws

    Y_ws = torch.empty((nw_k, B * H, nt_total, BT, K), dtype=k.dtype, device=q.device)
    dW_priv = torch.zeros((nw_k, B * H, nt_total, BT, K), dtype=torch.float32, device=q.device)
    dU_priv = torch.zeros((nw_k, B * H, nt_total, BT, K), dtype=torch.float32, device=q.device)
    _path_inter_chunk_key_duality_worker_bwd_kernel[(nw_k, B * H)](
        Q_arrow, K_arrow, v, w, U_buf, c_cum, do, lse, delta,
        Y_ws, dK_arrow, dV_inter, dC_k_inter, dW_priv, dU_priv,
        bos, eos, fc, lc, loc, nt_total, T, float(scale),
        NW_K=nw_k, HQ=HQ, H=H, G=G, K=K, V_DIM=V_DIM, BT=BT, USE_G=use_g,
        num_warps=8, num_stages=1,
    )
    _path_reduce_dw_du_priv_kernel[(nt_total, B * H)](
        dW_priv, dU_priv, dW_inter, dU_inter, nt_total, B * H,
        NW_K=nw_k, K=K, BT=BT,
        num_warps=4, num_stages=1,
    )
    del Y_ws, dW_priv, dU_priv

    if dk_arrow_ext is not None:
        dK_arrow = dK_arrow + dk_arrow_ext
    if dv_ext is not None:
        dV_inter = dV_inter + dv_ext
    if dw_ext is not None:
        dW_inter = dW_inter + dw_ext
    if du_ext is not None:
        dU_inter = dU_inter + du_ext

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dw = torch.empty_like(w)
    dbeta = torch.empty_like(beta)

    P_intra = torch.empty((B, nt_total, HQ, BT, BT), dtype=v.dtype, device=q.device)
    dB_raw = torch.empty((B, nt_total, HQ, BT, BT), dtype=torch.float32, device=q.device)
    dM_qw = torch.empty((B, nt_total, HQ, BT, BT), dtype=torch.float32, device=q.device)
    N_qwt = torch.empty((B, nt_total, HQ, BT, BT), dtype=torch.float32, device=q.device)
    dM_wbk = torch.empty((B, nt_total, H, BT, BT), dtype=torch.float32, device=q.device)
    N_twbk = torch.empty((B, nt_total, H, BT, BT), dtype=torch.float32, device=q.device)
    dL_buf = torch.empty((B, nt_total, H, BT, BT), dtype=torch.float32, device=q.device)

    _path_intra_chunk_score_dv_bwd_kernel[(nt_total, B * H)](
        q, k, v, w, beta, c_cum, A_inv, do, lse, delta,
        dV_inter, dC_q_inter, dC_k_inter,
        P_intra, dB_raw, dv, dC_total,
        bos, eos, loc, nt_total, T, float(scale), int(G), int(K), int(V_DIM),
        HQ=HQ, H=H, K=K, V_DIM=V_DIM, BT=BT, USE_G=use_g,
        num_warps=8, num_stages=1,
    )
    _path_intra_chunk_ut_matrices_bwd_kernel[(nt_total, B * H)](
        q, k, w, beta, A_inv, U_buf, dB_raw,
        dQ_arrow, dK_arrow, dU_inter,
        dM_qw, N_qwt, dM_wbk, N_twbk, dL_buf, dbeta,
        bos, eos, loc, nt_total, T, int(G), int(K),
        HQ=HQ, H=H, K=K, BT=BT,
        num_warps=8, num_stages=1,
    )
    _path_intra_chunk_ut_vectors_bwd_kernel[(nt_total, B * H, K // 64)](
        q, k, w, beta, A_inv, dB_raw,
        dM_qw, N_qwt, dM_wbk, N_twbk, dL_buf,
        dQ_arrow, dK_arrow, dW_inter, dU_inter,
        dq, dk, dw,
        bos, eos, loc, nt_total, T, int(G),
        HQ=HQ, H=H, K=K, BT=BT,
        num_warps=8, num_stages=1,
    )

    if use_g:
        dg = path_fox_gate_bwd_rev_cumsum_raw(dC_total, g, bos, eos, fc, lc, apply_logsigmoid=False)
    else:
        dg = None
    return dq, dk, dv, dw, dbeta, dg


class PaTHFoXUTDualityAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        beta: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scale is None:
            scale = q.shape[-1] ** -0.5
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        ctx.orig_w_dtype = w.dtype
        w_c = (w if w.dtype == q.dtype else w.to(q.dtype)).contiguous()
        beta_c = beta.contiguous()
        g_c = g.contiguous() if g is not None else None

        (
            O_out, LSE_out, A_inv, Q_arrow, K_arrow, U_buf, c_cum,
            _O_intra, _M_intra, _L_intra, _X_init, bos, eos, fc, lc, loc, nt_total
        ) = _run_path_fox_ut_duality_fwd_raw(
            q_c, k_c, v_c, w_c, beta_c, g_c, float(scale), cu_seqlens=cu_seqlens, save_x_init=False
        )
        ctx.save_for_backward(
            q_c, k_c, v_c, w_c, beta_c,
            g_c if g_c is not None else torch.empty((0,), device=q.device),
            O_out, LSE_out, A_inv, Q_arrow, K_arrow, U_buf, c_cum,
            bos, eos, fc, lc, loc,
        )
        ctx.has_g = g_c is not None
        ctx.scale = float(scale)
        ctx.nt_total = nt_total
        return O_out, LSE_out

    @staticmethod
    def backward(ctx, do: torch.Tensor, dlse: Optional[torch.Tensor] = None):
        (
            q_c, k_c, v_c, w_c, beta_c, g_saved,
            O_out, LSE_out, A_inv, Q_arrow, K_arrow, U_buf, c_cum,
            bos, eos, fc, lc, loc,
        ) = ctx.saved_tensors
        g_c = g_saved if ctx.has_g else None
        dq, dk, dv, dw, dbeta, dg = _run_path_fox_ut_duality_bwd_raw(
            q_c, k_c, v_c, w_c, beta_c, g_c,
            O_out, do.contiguous(), LSE_out,
            A_inv, Q_arrow, K_arrow, U_buf, c_cum,
            bos, eos, fc, lc, loc, ctx.nt_total, ctx.scale,
        )
        if dw.dtype != ctx.orig_w_dtype:
            dw = dw.to(ctx.orig_w_dtype)
        return dq, dk, dv, dw, dbeta, dg, None, None


def path_fox_ut_duality_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return PaTHFoXUTDualityAttentionFunction.apply(q, k, v, w, beta, g, scale, cu_seqlens)


# =============================================================================
# 5. Strip-Mined Ring PaTH-CP Distributed Context Parallelism Kernels (0 Spills)
# =============================================================================

@triton.jit
def _path_cp_shard_defect_fwd_kernel(
    K_arrow,
    W,
    U_buf,
    F_shard_out,
    F_suf_out,
    K_hat_out,
    NT_shard,
    T_shard,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
):
    """
    Strip-mined [BR=32, K] right-to-left shard Householder defect accumulator:
      F_{next} = F_{cur} + (I_K - F_{cur}) @ W_{nt}^T @ U_{nt}
      K_hat[nt] = K_arrow[nt] @ (I_K - F_{suf}[nt]^T)
    Grid = (K // BR, B * H). Holds only one [32, K] row strip of F in registers -> 0 spills at K=128!
    """
    pid_strip = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_r = pid_strip * BR + tl.arange(0, BR)
    offs_k = tl.arange(0, K)
    offs_t = tl.arange(0, BT)

    eye_strip = (offs_r[:, None] == offs_k[None, :]).to(tl.float32)
    b_F = tl.zeros([BR, K], dtype=tl.float32)

    for step in range(0, NT_shard):
        nt = NT_shard - 1 - step
        tok_t = nt * BT + offs_t
        ku_base = ((b * NT_shard + nt) * H + h) * BT * K

        # Save suffix defect strip F_suf[b, nt, h, offs_r, offs_k]
        fsuf_base = ((b * NT_shard + nt) * H + h) * K * K
        tl.store(F_suf_out + fsuf_base + offs_r[:, None] * K + offs_k[None, :], b_F)

        # Update F strip: b_F <- b_F + ((eye_strip - b_F) @ W_{nt}^T) @ U_{nt}
        w_ptrs = W + (b * T_shard * H + tok_t[:, None] * H + h) * K + offs_k[None, :]
        b_w = tl.load(w_ptrs)
        b_u = tl.load(U_buf + ku_base + offs_t[:, None] * K + offs_k[None, :])
        p_strip = (eye_strip - b_F).to(b_w.dtype)
        proj = tl.dot(p_strip, tl.trans(b_w), allow_tf32=False).to(b_u.dtype)
        b_F = b_F + tl.dot(proj, b_u, allow_tf32=False)

    f_base = pid_bh * K * K
    tl.store(F_shard_out + f_base + offs_r[:, None] * K + offs_k[None, :], b_F)


@triton.jit
def _path_cp_apply_khat_fwd_kernel(
    K_arrow,
    F_suf,
    K_hat_out,
    NT_shard,
    T_shard,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
):
    """
    Compute K_hat[nt] = K_arrow[nt] - sum_{strip} K_arrow[:, strip] @ F_suf[nt, strip, :]^T
    Wait: K_hat = K_arrow @ (I - F_suf^T), so K_hat[t, r] = K_arrow[t, r] - dot(K_arrow[t, :], F_suf[r, :]).
    Strip-mined over output feature strip `r in [pid_strip*32 .. (pid_strip+1)*32]` -> 0 spills!
    """
    pid_nt = tl.program_id(0)
    pid_strip = tl.program_id(1)
    pid_bh = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H

    offs_t = tl.arange(0, BT)
    offs_r = pid_strip * BR + tl.arange(0, BR)
    offs_k = tl.arange(0, K)

    ku_base = ((b * NT_shard + pid_nt) * H + h) * BT * K
    b_k_all = tl.load(K_arrow + ku_base + offs_t[:, None] * K + offs_k[None, :])
    b_k_strip = tl.load(K_arrow + ku_base + offs_t[:, None] * K + offs_r[None, :]).to(tl.float32)

    fsuf_base = ((b * NT_shard + pid_nt) * H + h) * K * K
    b_fsuf_strip = tl.load(F_suf + fsuf_base + offs_r[:, None] * K + offs_k[None, :]).to(b_k_all.dtype)

    b_khat_strip = b_k_strip - tl.dot(b_k_all, tl.trans(b_fsuf_strip), allow_tf32=False)
    tok_t = pid_nt * BT + offs_t
    kh_ptrs = K_hat_out + (b * T_shard * H + tok_t[:, None] * H + h) * K + offs_r[None, :]
    tl.store(kh_ptrs, b_khat_strip.to(K_hat_out.dtype.element_ty))


@triton.jit
def _path_cp_shard_defect_bwd_kernel(
    K_arrow,
    W,
    U_buf,
    F_suf,
    dK_hat_ext,
    dF_shard_ext,
    dK_arrow_ext,
    dW_part_strip,
    dU_part_strip,
    NT_shard,
    T_shard,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
):
    """
    Strip-mined [BR=32, K] backward for shard Householder defect F_shard and K_hat:
    Grid = (K // BR, B * H). 0 spills at K = 128, 0 atomics!
    """
    pid_strip = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_r = pid_strip * BR + tl.arange(0, BR)
    offs_k = tl.arange(0, K)
    offs_t = tl.arange(0, BT)
    eye_strip = (offs_r[:, None] == offs_k[None, :]).to(tl.float32)

    f_base = pid_bh * K * K
    b_dF = tl.load(dF_shard_ext + f_base + offs_r[:, None] * K + offs_k[None, :]).to(tl.float32)

    for nt in range(0, NT_shard):
        tok_t = nt * BT + offs_t
        ku_base = ((b * NT_shard + nt) * H + h) * BT * K
        fsuf_base = ((b * NT_shard + nt) * H + h) * K * K

        b_F_prev = tl.load(F_suf + fsuf_base + offs_r[:, None] * K + offs_k[None, :]).to(tl.float32)
        b_P_prev = (eye_strip - b_F_prev).to(W.dtype.element_ty)

        w_ptrs = W + (b * T_shard * H + tok_t[:, None] * H + h) * K + offs_k[None, :]
        b_w = tl.load(w_ptrs)
        b_u = tl.load(U_buf + ku_base + offs_t[:, None] * K + offs_k[None, :])

        # Forward was: A_strip = P_prev @ W^T ([BR, BT]), F_next = F_prev + A_strip @ U
        b_A = tl.dot(b_P_prev, tl.trans(b_w), allow_tf32=False)
        b_dA = tl.dot(b_dF.to(b_u.dtype), tl.trans(b_u), allow_tf32=False)
        b_du_strip = tl.dot(tl.trans(b_A.to(b_w.dtype)), b_dF.to(b_w.dtype), allow_tf32=False)
        b_dw_strip = tl.dot(tl.trans(b_dA.to(b_w.dtype)), b_P_prev, allow_tf32=False)

        part_base = (((pid_strip * tl.num_programs(1) + pid_bh) * NT_shard + nt) * BT) * K
        tl.store(dU_part_strip + part_base + offs_t[:, None] * K + offs_k[None, :], b_du_strip)
        tl.store(dW_part_strip + part_base + offs_t[:, None] * K + offs_k[None, :], b_dw_strip)

        # Propagate b_dF to previous suffix step + add contribution from K_hat[nt, :, offs_r]
        b_dF = b_dF - tl.dot(b_dA.to(b_w.dtype), b_w, allow_tf32=False)
        dkh_ptrs = dK_hat_ext + (b * T_shard * H + tok_t[:, None] * H + h) * K + offs_r[None, :]
        b_dkh_strip = tl.load(dkh_ptrs)
        b_k_all = tl.load(K_arrow + ku_base + offs_t[:, None] * K + offs_k[None, :])
        b_dF = b_dF - tl.dot(tl.trans(b_dkh_strip.to(b_k_all.dtype)), b_k_all, allow_tf32=False)


def _compute_shard_summary_triton(
    K_arrow: torch.Tensor,
    w_local: torch.Tensor,
    U_buf: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run strip-mined shard Householder defect + K_hat kernels (0 spills at K=128)."""
    B, nt_shard, H, BT, K = K_arrow.shape
    T_shard = nt_shard * BT
    BR = 32
    n_strips = K // BR
    F_shard = torch.empty((B, H, K, K), dtype=torch.float32, device=w_local.device)
    F_suf = torch.empty((B, nt_shard, H, K, K), dtype=torch.float32, device=w_local.device)
    K_hat = torch.empty((B, T_shard, H, K), dtype=K_arrow.dtype, device=w_local.device)

    _path_cp_shard_defect_fwd_kernel[(n_strips, B * H)](
        K_arrow, w_local, U_buf, F_shard, F_suf, K_hat, nt_shard, T_shard,
        H=H, K=K, BT=BT, BR=BR, num_warps=4, num_stages=1,
    )
    _path_cp_apply_khat_fwd_kernel[(nt_shard, n_strips, B * H)](
        K_arrow, F_suf, K_hat, nt_shard, T_shard,
        H=H, K=K, BT=BT, BR=BR, num_warps=4, num_stages=1,
    )
    return F_shard, F_suf, K_hat


def _backward_shard_summary_triton(
    K_arrow: torch.Tensor,
    w_local: torch.Tensor,
    U_buf: torch.Tensor,
    F_suf: torch.Tensor,
    dK_hat_ext: torch.Tensor,
    dF_shard_ext: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse strip-mined shard summary into (dK_arrow_ext, dW_ext, dU_ext) with 0 spills & 0 atomics."""
    B, nt_shard, H, BT, K = K_arrow.shape
    T_shard = nt_shard * BT
    BR = 32
    n_strips = K // BR
    dW_part_strip = torch.empty((n_strips, B * H, nt_shard, BT, K), dtype=torch.float32, device=w_local.device)
    dU_part_strip = torch.empty((n_strips, B * H, nt_shard, BT, K), dtype=torch.float32, device=w_local.device)
    dK_arrow_ext = torch.empty((B, nt_shard, H, BT, K), dtype=torch.float32, device=w_local.device)

    _path_cp_shard_defect_bwd_kernel[(n_strips, B * H)](
        K_arrow, w_local, U_buf, F_suf, dK_hat_ext.contiguous(), dF_shard_ext.contiguous(),
        dK_arrow_ext, dW_part_strip, dU_part_strip, nt_shard, T_shard,
        H=H, K=K, BT=BT, BR=BR, num_warps=4, num_stages=1,
    )
    # Also dK_arrow_ext from K_hat = K_arrow @ (I - F_suf^T) => dK_arrow_ext = dK_hat_ext @ (I - F_suf)
    dkh_4d = dK_hat_ext.view(B, nt_shard, BT, H, K).permute(0, 1, 3, 2, 4).to(torch.float32)
    eye = torch.eye(K, dtype=torch.float32, device=w_local.device)
    dK_arrow_ext = torch.matmul(dkh_4d, eye - F_suf)
    dW_ext = dW_part_strip.sum(dim=0)
    dU_ext = dU_part_strip.sum(dim=0)
    return dK_arrow_ext, dW_ext, dU_ext


@triton.jit
def _path_cp_xinit_prefix_ws_kernel(
    W,           # [B, T_shard, H, K]
    U_buf,       # [B, nt_shard, H, BT, K]
    Pt_ws,       # [n_strips, B * H, nt_shard - 1, BR, K] W.dtype
    At_ws,       # [n_strips, B * H, nt_shard - 1, BR, BT] W.dtype
    dW_strip,    # [n_strips, B * H, nt_shard, BT, K] float32
    dU_strip,    # [n_strips, B * H, nt_shard, BT, K] float32
    nt_shard,
    T_shard,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
):
    """
    Pass 1 of strip-mined [BR=32, K] 0-spill prefix Householder workspace recurrence:
    P_pre[0] = I, P_pre[m+1] = (I - W_m^T U_m) P_pre[m] in transposed strip form.
    Saves both Pt_ws and At_ws in W.dtype so Pass 2 avoids recomputing At and stays well below 255 regs (`0` spills).
    """
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    B_H = tl.num_programs(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_r = pid_s * BR + tl.arange(0, BR)
    offs_r_loc = tl.arange(0, BR)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, K)

    b_Pt = (offs_r[:, None] == offs_k[None, :]).to(tl.float32)  # [BR, K]
    for m in range(0, nt_shard - 1):
        ws_idx = (pid_s * B_H + pid_bh) * (nt_shard - 1) + m
        b_Pt_w = b_Pt.to(W.dtype.element_ty)
        tl.store(Pt_ws + ws_idx * BR * K + offs_r_loc[:, None] * K + offs_k[None, :], b_Pt_w)

        tok_t = m * BT + offs_t
        w_ptrs = W + (b * T_shard * H + tok_t[:, None] * H + h) * K + offs_k[None, :]
        b_w = tl.load(w_ptrs)  # [BT, K]

        u_base = ((b * nt_shard + m) * H + h) * BT * K
        b_u = tl.load(U_buf + u_base + offs_t[:, None] * K + offs_k[None, :])  # [BT, K]

        b_At = tl.dot(b_Pt_w, tl.trans(b_u), allow_tf32=False).to(b_w.dtype)
        tl.store(At_ws + ws_idx * BR * BT + offs_r_loc[:, None] * BT + offs_t[None, :], b_At)
        b_Pt = b_Pt - tl.dot(b_At, b_w, allow_tf32=False)

    last_base = ((pid_s * B_H + pid_bh) * nt_shard + (nt_shard - 1)) * BT * K
    b_zero = tl.zeros([BT, K], dtype=tl.float32)
    tl.store(dW_strip + last_base + offs_t[:, None] * K + offs_k[None, :], b_zero)
    tl.store(dU_strip + last_base + offs_t[:, None] * K + offs_k[None, :], b_zero)


@triton.jit
def _path_cp_xinit_prefix_bwd_kernel(
    W,             # [B, T_shard, H, K]
    U_buf,         # [B, nt_shard, H, BT, K]
    dP_contrib_T,  # [B, nt_shard - 1, H, K, K] float32 (already transposed: dP_contrib^T)
    Pt_ws,         # [n_strips, B * H, nt_shard - 1, BR, K] W.dtype
    At_ws,         # [n_strips, B * H, nt_shard - 1, BR, BT] W.dtype
    dW_strip,      # [n_strips, B * H, nt_shard, BT, K] float32
    dU_strip,      # [n_strips, B * H, nt_shard, BT, K] float32
    nt_shard,
    T_shard,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
):
    """
    Pass 2 of strip-mined [BR=32, K] 0-spill prefix Householder backward recurrence for X_init[cq] = Q_arrow[cq] @ P_pre[cq].
    Each program (pid_s, pid_bh) tracks transposed column strip b_dPt = (dP_run[:, offs_r])^T [BR, K]
    completely independently across strips (`0` spills, `0` atomics).
    """
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    B_H = tl.num_programs(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_r = pid_s * BR + tl.arange(0, BR)
    offs_r_loc = tl.arange(0, BR)
    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, K)

    b_dPt = tl.zeros([BR, K], dtype=tl.float32)
    for step in range(0, nt_shard - 1):
        m = (nt_shard - 2) - step
        dp_base = ((b * (nt_shard - 1) + m) * H + h) * K * K
        b_dp_c = tl.load(dP_contrib_T + dp_base + offs_r[:, None] * K + offs_k[None, :]).to(tl.float32)
        b_dPt = b_dPt + b_dp_c

        ws_idx = (pid_s * B_H + pid_bh) * (nt_shard - 1) + m
        out_base = ((pid_s * B_H + pid_bh) * nt_shard + m) * BT * K
        b_dPt_w = b_dPt.to(W.dtype.element_ty)

        b_At = tl.load(At_ws + ws_idx * BR * BT + offs_r_loc[:, None] * BT + offs_t[None, :])
        b_dw = -tl.dot(tl.trans(b_At), b_dPt_w, allow_tf32=False)
        tl.store(dW_strip + out_base + offs_t[:, None] * K + offs_k[None, :], b_dw)

        tok_t = m * BT + offs_t
        w_ptrs = W + (b * T_shard * H + tok_t[:, None] * H + h) * K + offs_k[None, :]
        b_w = tl.load(w_ptrs)
        b_dAt = -tl.dot(b_dPt_w, tl.trans(b_w), allow_tf32=False).to(b_w.dtype)

        b_Pt_m = tl.load(Pt_ws + ws_idx * BR * K + offs_r_loc[:, None] * K + offs_k[None, :])
        b_du = tl.dot(tl.trans(b_dAt), b_Pt_m, allow_tf32=False)
        tl.store(dU_strip + out_base + offs_t[:, None] * K + offs_k[None, :], b_du)

        u_base = ((b * nt_shard + m) * H + h) * BT * K
        b_u = tl.load(U_buf + u_base + offs_t[:, None] * K + offs_k[None, :])
        b_dPt = b_dPt + tl.dot(b_dAt, b_u, allow_tf32=False)


@triton.jit
def _path_cp_cross_softmax_fwd_kernel(
    Z_raw,       # [rank, B, H, Lq * G, L] in mm_dtype (modified in-place to normalized probs)
    LSE_loc,     # [B, H, Lq, G] float32
    CQ_lg,       # [B, H, Lq, G] float32
    CK_lg,       # [rank, B, H, G, L] float32
    LSE_out,     # [B, H, Lq, G] float32
    Alpha_out,   # [B, H, Lq, G] float32
    scale,
    rank_rt,
    Lq_rt,
    L_rt,
    B_H_rt,
    G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_G: tl.constexpr,
):
    """
    Fused cross-shard online softmax forward kernel over Z_raw [rank, B, H, Lq * G, L].
    Eliminates 8 PyTorch elementwise/reduction/copy kernels per query block with 0 spills.
    """
    pid = tl.program_id(0)
    n_rows = B_H_rt * Lq_rt * G
    lse_loc_val = tl.load(LSE_loc + pid).to(tl.float32)
    m_cur = lse_loc_val
    l_cur = 1.0
    if USE_G:
        cq_val = tl.load(CQ_lg + pid).to(tl.float32)
        g_idx = pid % G
        bh = pid // (Lq_rt * G)
    else:
        cq_val = 0.0
        g_idx = 0
        bh = 0

    offs_n = tl.arange(0, BLOCK_N)
    for p in range(0, rank_rt):
        z_row_base = (p * n_rows + pid) * L_rt
        ck_row_base = ((p * B_H_rt + bh) * G + g_idx) * L_rt
        for n0 in range(0, L_rt, BLOCK_N):
            col = n0 + offs_n
            m_n = col < L_rt
            z_val = tl.load(Z_raw + z_row_base + col, mask=m_n, other=0.0).to(tl.float32) * scale
            if USE_G:
                ck_val = tl.load(CK_lg + ck_row_base + col, mask=m_n, other=0.0).to(tl.float32)
                z_val = z_val + (cq_val - ck_val)
            z_masked = tl.where(m_n, z_val, -float("inf"))
            m_tile = tl.max(z_masked, axis=0)
            m_next = tl.maximum(m_cur, m_tile)
            l_cur = l_cur * tl.exp(m_cur - m_next) + tl.sum(tl.where(m_n, tl.exp(z_val - m_next), 0.0), axis=0)
            m_cur = m_next

    lse_final = m_cur + tl.log(l_cur)
    inv_l = 1.0 / l_cur
    alpha_val = tl.exp(lse_loc_val - m_cur) * inv_l
    tl.store(LSE_out + pid, lse_final)
    tl.store(Alpha_out + pid, alpha_val)

    for p in range(0, rank_rt):
        z_row_base = (p * n_rows + pid) * L_rt
        ck_row_base = ((p * B_H_rt + bh) * G + g_idx) * L_rt
        for n0 in range(0, L_rt, BLOCK_N):
            col = n0 + offs_n
            m_n = col < L_rt
            z_val = tl.load(Z_raw + z_row_base + col, mask=m_n, other=0.0).to(tl.float32) * scale
            if USE_G:
                ck_val = tl.load(CK_lg + ck_row_base + col, mask=m_n, other=0.0).to(tl.float32)
                z_val = z_val + (cq_val - ck_val)
            prob = tl.where(m_n, tl.exp(z_val - m_cur) * inv_l, 0.0)
            tl.store(Z_raw + z_row_base + col, prob.to(Z_raw.dtype.element_ty), mask=m_n)


@triton.jit
def _path_cp_cross_softmax_bwd_kernel(
    Z_raw,        # [rank, B, H, Lq * G, L] in mm_dtype (in: Q @ K^T, out: prob in mm_dtype)
    dP_raw,       # [rank, B, H, Lq * G, L] in mm_dtype (in: dO @ V^T, out: unscaled dz in mm_dtype)
    LSE_lg,       # [B, H, Lq, G] float32
    Delta_lg,     # [B, H, Lq, G] float32
    CQ_lg,        # [B, H, Lq, G] float32
    CK_lg,        # [rank, B, H, G, L] float32
    dC_self_out,  # [B, H, Lq, G] float32
    scale,
    rank_rt,
    Lq_rt,
    L_rt,
    B_H_rt,
    G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_G: tl.constexpr,
):
    """
    Fused cross-shard softmax backward kernel over Z_raw and dP_raw [rank, B, H, Lq * G, L].
    Recomputes prob in-place in Z_raw, unscaled dz = prob * (dp - delta) in-place in dP_raw,
    and accumulates dC_self_out in 1 pass (0 spills).
    """
    pid = tl.program_id(0)
    n_rows = B_H_rt * Lq_rt * G
    lse_val = tl.load(LSE_lg + pid).to(tl.float32)
    delta_val = tl.load(Delta_lg + pid).to(tl.float32)
    if USE_G:
        cq_val = tl.load(CQ_lg + pid).to(tl.float32)
        g_idx = pid % G
        bh = pid // (Lq_rt * G)
    else:
        cq_val = 0.0
        g_idx = 0
        bh = 0

    dc_self_acc = 0.0
    offs_n = tl.arange(0, BLOCK_N)
    for p in range(0, rank_rt):
        z_row_base = (p * n_rows + pid) * L_rt
        ck_row_base = ((p * B_H_rt + bh) * G + g_idx) * L_rt
        for n0 in range(0, L_rt, BLOCK_N):
            col = n0 + offs_n
            m_n = col < L_rt
            z_val = tl.load(Z_raw + z_row_base + col, mask=m_n, other=0.0).to(tl.float32) * scale
            if USE_G:
                ck_val = tl.load(CK_lg + ck_row_base + col, mask=m_n, other=0.0).to(tl.float32)
                z_val = z_val + (cq_val - ck_val)
            prob = tl.where(m_n, tl.exp(z_val - lse_val), 0.0)
            dp_val = tl.load(dP_raw + z_row_base + col, mask=m_n, other=0.0).to(tl.float32)
            dz = prob * (dp_val - delta_val)
            if USE_G:
                dc_self_acc += tl.sum(dz, axis=0)
            tl.store(Z_raw + z_row_base + col, prob.to(Z_raw.dtype.element_ty), mask=m_n)
            tl.store(dP_raw + z_row_base + col, dz.to(dP_raw.dtype.element_ty), mask=m_n)

    if USE_G:
        tl.store(dC_self_out + pid, dc_self_acc)


class RingPaTHCPAttentionFunction(torch.autograd.Function):
    """
    Distributed 8-GPU NVSwitch Ring PaTH-CP Context-Parallel Attention:
    Combines local lock-free UT + K-duality intra-shard execution with strip-mined
    shard Householder defect summaries (F_shard, K_hat, X_init) and online-softmax
    cross-rank ring aggregation.
    """

    @staticmethod
    def forward(
        ctx,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        w_local: torch.Tensor,
        beta_local: torch.Tensor,
        g_local: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scale is None:
            scale = q_local.shape[-1] ** -0.5
        q_c = q_local.contiguous()
        k_c = k_local.contiguous()
        v_c = v_local.contiguous()
        ctx.orig_w_dtype = w_local.dtype
        w_c = (w_local if w_local.dtype == q_local.dtype else w_local.to(q_local.dtype)).contiguous()
        beta_c = beta_local.contiguous()
        g_c = g_local.contiguous() if g_local is not None else None

        B, L, HQ, K = q_c.shape
        H = k_c.shape[2]
        V_DIM = v_c.shape[3]
        G = HQ // H

        (
            O_loc, LSE_loc, A_inv, Q_arrow, K_arrow, U_buf, c_loc,
            _O_intra, _M_intra, _L_intra, X_init, bos, eos, fc, lc, loc, nt_shard
        ) = _run_path_fox_ut_duality_fwd_raw(
            q_c, k_c, v_c, w_c, beta_c, g_c, float(scale), cu_seqlens=None, save_x_init=True
        )
        F_shard, F_suf, K_hat = _compute_shard_summary_triton(K_arrow, w_c, U_buf)

        world_size = dist.get_world_size(cp_group) if (cp_group is not None or dist.is_initialized()) else 1
        rank = dist.get_rank(cp_group) if (cp_group is not None or dist.is_initialized()) else 0

        use_g = g_c is not None
        if use_g:
            g_tot = c_loc[:, -1:, :]  # [B, 1, HQ]
            g_tot_all = torch.empty((world_size, B, 1, HQ), dtype=torch.float32, device=q_c.device)
            if world_size > 1:
                dist.all_gather_into_tensor(g_tot_all, g_tot.contiguous(), group=cp_group)
            else:
                g_tot_all[0].copy_(g_tot)
            prefix_g = torch.cumsum(g_tot_all, dim=0) - g_tot_all
            c_global_loc = c_loc + prefix_g[rank]
            c_all = torch.empty((world_size, B, L, HQ), dtype=torch.float32, device=q_c.device)
            if world_size > 1:
                dist.all_gather_into_tensor(c_all, c_global_loc.contiguous(), group=cp_group)
            else:
                c_all[0].copy_(c_global_loc)
        else:
            c_global_loc = torch.empty((1,), dtype=torch.float32, device=q_c.device)
            c_all = torch.empty((1,), dtype=torch.float32, device=q_c.device)

        # Gather compact shard summaries (K_hat, v_c, F_shard) across CP ring ranks
        K_hat_all = torch.empty((world_size, B, L, H, K), dtype=K_hat.dtype, device=q_c.device)
        V_all = torch.empty((world_size, B, L, H, V_DIM), dtype=v_c.dtype, device=q_c.device)
        F_all = torch.empty((world_size, B, H, K, K), dtype=torch.float32, device=q_c.device)
        if world_size > 1:
            dist.all_gather_into_tensor(K_hat_all, K_hat.contiguous(), group=cp_group)
            dist.all_gather_into_tensor(V_all, v_c.contiguous(), group=cp_group)
            dist.all_gather_into_tensor(F_all, F_shard.contiguous(), group=cp_group)
        else:
            K_hat_all[0].copy_(K_hat)
            V_all[0].copy_(v_c)
            F_all[0].copy_(F_shard)

        # Cross-rank right-to-left ring merge for p = rank - 1 .. 0
        # Use q_c.dtype (BF16 TensorCores in benchmark; FP32 in Gate G9 parity) for batched cross-rank GEMMs
        mm_dtype = q_c.dtype if q_c.dtype in (torch.bfloat16, torch.float16) else torch.float32
        eye = torch.eye(K, dtype=mm_dtype, device=q_c.device)

        if rank > 0:
            x_cur = X_init.view(B, L, H, G, K).permute(0, 2, 1, 3, 4).contiguous().view(B, H, L * G, K).to(mm_dtype)
            x_list: List[Optional[torch.Tensor]] = [None] * rank
            for p in range(rank - 1, -1, -1):
                x_list[p] = x_cur.view(B, H, L, G, K)
                if p > 0:
                    x_cur = torch.matmul(x_cur, eye - F_all[p].to(mm_dtype))
            x_stack = torch.stack(x_list, dim=0)  # [rank, B, H, L, G, K]
            kh_stack_t = K_hat_all[:rank].permute(0, 1, 3, 4, 2).contiguous().to(mm_dtype)  # [rank, B, H, K, L]
            v_stack = V_all[:rank].permute(0, 1, 3, 2, 4).contiguous().to(mm_dtype)         # [rank, B, H, L, V_DIM]

            lse_loc_lg = LSE_loc.view(B, L, H, G).permute(0, 2, 1, 3).contiguous()
            o_loc_lg = O_loc.to(torch.float32).view(B, L, H, G, V_DIM).permute(0, 2, 1, 3, 4).contiguous()
            if use_g:
                cq_lg = c_global_loc.view(B, L, H, G).permute(0, 2, 1, 3).contiguous()
                ck_lg = c_all[:rank].view(rank, B, L, H, G).permute(0, 1, 3, 4, 2).contiguous()

            o_final_lg = torch.empty((B, H, L, G, V_DIM), dtype=v_c.dtype, device=q_c.device)
            lse_final_lg = torch.empty((B, H, L, G), dtype=torch.float32, device=q_c.device)
            alpha_lg = torch.empty((B, H, L, G), dtype=torch.float32, device=q_c.device)
            dummy_g = torch.empty((1,), dtype=torch.float32, device=q_c.device)
            BLOCK_N_CP = 128 if L >= 128 else 64
            B_Q = 512
            for q0 in range(0, L, B_Q):
                q1 = min(L, q0 + B_Q)
                Lq = q1 - q0
                x_blk = x_stack if (q0 == 0 and q1 == L) else x_stack[:, :, :, q0:q1, :, :].contiguous()
                x_blk_flat = x_blk.view(rank, B, H, Lq * G, K)
                z_raw = torch.matmul(x_blk_flat, kh_stack_t)  # [rank, B, H, Lq * G, L] in mm_dtype
                lse_q = lse_loc_lg if (q0 == 0 and q1 == L) else lse_loc_lg[:, :, q0:q1, :].contiguous()
                lse_out_q = lse_final_lg if (q0 == 0 and q1 == L) else torch.empty((B, H, Lq, G), dtype=torch.float32, device=q_c.device)
                alpha_out_q = alpha_lg if (q0 == 0 and q1 == L) else torch.empty((B, H, Lq, G), dtype=torch.float32, device=q_c.device)
                cq_q = (cq_lg if (q0 == 0 and q1 == L) else cq_lg[:, :, q0:q1, :].contiguous()) if use_g else dummy_g
                ck_q = ck_lg if use_g else dummy_g

                n_rows_q = B * H * Lq * G
                _path_cp_cross_softmax_fwd_kernel[(n_rows_q,)](
                    z_raw, lse_q, cq_q, ck_q, lse_out_q, alpha_out_q,
                    float(scale), rank, Lq, L, B * H,
                    G=G, BLOCK_N=BLOCK_N_CP, USE_G=use_g,
                    num_warps=4, num_stages=1,
                )
                if not (q0 == 0 and q1 == L):
                    lse_final_lg[:, :, q0:q1, :] = lse_out_q
                o_cross_blk = torch.matmul(z_raw, v_stack).to(torch.float32).view(rank, B, H, Lq, G, V_DIM).sum(dim=0)
                o_final_lg[:, :, q0:q1, :, :] = (
                    o_loc_lg[:, :, q0:q1, :, :] * alpha_out_q.unsqueeze(-1) + o_cross_blk
                ).to(v_c.dtype)
                del z_raw, o_cross_blk

            o_final = o_final_lg.permute(0, 2, 1, 3, 4).contiguous().view(B, L, HQ, V_DIM)
            lse_final = lse_final_lg.permute(0, 2, 1, 3).contiguous().view(B, L, HQ)
        else:
            o_final = O_loc.to(v_c.dtype)
            lse_final = LSE_loc.clone()

        ctx.save_for_backward(
            q_c, k_c, v_c, w_c, beta_c,
            g_c if g_c is not None else torch.empty((0,), device=q_c.device),
            o_final, lse_final, A_inv, Q_arrow, K_arrow, U_buf, c_loc,
            X_init, F_suf, K_hat_all, V_all, F_all, c_global_loc, c_all,
            bos, eos, fc, lc, loc,
        )
        ctx.has_g = use_g
        ctx.scale = float(scale)
        ctx.nt_shard = nt_shard
        ctx.cp_group = cp_group
        ctx.world_size = world_size
        ctx.rank = rank
        return o_final, lse_final

    @staticmethod
    def backward(ctx, do_local: torch.Tensor, dlse_local: Optional[torch.Tensor] = None):
        (
            q_c, k_c, v_c, w_c, beta_c, g_saved,
            o_final, lse_final, A_inv, Q_arrow, K_arrow, U_buf, c_loc,
            X_init, F_suf, K_hat_all, V_all, F_all, c_global_loc, c_all,
            bos, eos, fc, lc, loc,
        ) = ctx.saved_tensors
        g_c = g_saved if ctx.has_g else None
        use_g = ctx.has_g
        scale = ctx.scale
        world_size = ctx.world_size
        rank = ctx.rank
        cp_group = ctx.cp_group

        B, L, HQ, K = q_c.shape
        H = k_c.shape[2]
        V_DIM = v_c.shape[3]
        G = HQ // H
        mm_dtype = q_c.dtype if q_c.dtype in (torch.bfloat16, torch.float16) else torch.float32
        do_f = do_local.contiguous().to(torch.float32)
        delta_loc = (o_final.to(torch.float32) * do_f).sum(dim=-1)  # [B, L, HQ]

        eye_mm = torch.eye(K, dtype=mm_dtype, device=q_c.device)

        dK_hat_send = torch.zeros((world_size, B, L, H, K), dtype=torch.float32, device=q_c.device)
        dV_send = torch.zeros((world_size, B, L, H, V_DIM), dtype=torch.float32, device=q_c.device)
        dF_send = torch.zeros((world_size, B, H, K, K), dtype=torch.float32, device=q_c.device)
        dc_send = torch.zeros((world_size, B, L, HQ), dtype=torch.float32, device=q_c.device) if use_g else None

        B_run = torch.zeros((B, H, L * G, K), dtype=mm_dtype, device=q_c.device)
        dc_self_cross = torch.zeros((B, L, HQ), dtype=torch.float32, device=q_c.device) if use_g else None

        if rank > 0:
            x_cur = X_init.view(B, L, H, G, K).permute(0, 2, 1, 3, 4).contiguous().view(B, H, L * G, K).to(mm_dtype)
            x_list: List[Optional[torch.Tensor]] = [None] * rank
            for p in range(rank - 1, -1, -1):
                x_list[p] = x_cur.view(B, H, L, G, K)
                if p > 0:
                    x_cur = torch.matmul(x_cur, eye_mm - F_all[p].to(mm_dtype))
            x_stack = torch.stack(x_list, dim=0)  # [rank, B, H, L, G, K]
            kh_stack = K_hat_all[:rank].permute(0, 1, 3, 2, 4).contiguous().to(mm_dtype)    # [rank, B, H, L, K]
            kh_stack_t = kh_stack.transpose(-1, -2).contiguous()                            # [rank, B, H, K, L]
            v_stack_t = V_all[:rank].permute(0, 1, 3, 4, 2).contiguous().to(mm_dtype)       # [rank, B, H, V_DIM, L]
            do_lg_all = do_f.view(B, L, H, G, V_DIM).permute(0, 2, 1, 3, 4).contiguous().to(mm_dtype)

            lse_lg = lse_final.view(B, L, H, G).permute(0, 2, 1, 3).contiguous()
            delta_lg = delta_loc.view(B, L, H, G).permute(0, 2, 1, 3).contiguous()
            if use_g:
                cq_lg = c_global_loc.view(B, L, H, G).permute(0, 2, 1, 3).contiguous()
                ck_lg = c_all[:rank].view(rank, B, L, H, G).permute(0, 1, 3, 4, 2).contiguous()
                dc_self_lg = torch.zeros((B, H, L, G), dtype=torch.float32, device=q_c.device)
                dc_send_hg = torch.zeros((rank, B, H, G, L), dtype=torch.float32, device=q_c.device)

            dV_stack = torch.zeros((rank, B, H, L, V_DIM), dtype=torch.float32, device=q_c.device)
            dK_hat_stack = torch.zeros((rank, B, H, L, K), dtype=torch.float32, device=q_c.device)
            B_contrib = torch.empty((rank, B, H, L, G, K), dtype=mm_dtype, device=q_c.device)
            kh_stack_scaled = (kh_stack.to(torch.float32) * scale).to(mm_dtype).contiguous()
            dummy_g = torch.empty((1,), dtype=torch.float32, device=q_c.device)
            BLOCK_N_CP = 128 if L >= 128 else 64

            B_Q = 512
            for q0 in range(0, L, B_Q):
                q1 = min(L, q0 + B_Q)
                Lq = q1 - q0
                x_blk = (x_stack if (q0 == 0 and q1 == L) else x_stack[:, :, :, q0:q1, :, :].contiguous()).view(rank, B, H, Lq * G, K)
                do_blk = (do_lg_all if (q0 == 0 and q1 == L) else do_lg_all[:, :, q0:q1, :, :].contiguous()).view(1, B, H, Lq * G, V_DIM)

                z_raw = torch.matmul(x_blk, kh_stack_t)    # [rank, B, H, Lq * G, L] in mm_dtype
                dp_raw = torch.matmul(do_blk, v_stack_t)   # [rank, B, H, Lq * G, L] in mm_dtype

                lse_q = lse_lg if (q0 == 0 and q1 == L) else lse_lg[:, :, q0:q1, :].contiguous()
                delta_q = delta_lg if (q0 == 0 and q1 == L) else delta_lg[:, :, q0:q1, :].contiguous()
                cq_q = (cq_lg if (q0 == 0 and q1 == L) else cq_lg[:, :, q0:q1, :].contiguous()) if use_g else dummy_g
                ck_q = ck_lg if use_g else dummy_g
                dc_self_q = (dc_self_lg if (q0 == 0 and q1 == L) else torch.empty((B, H, Lq, G), dtype=torch.float32, device=q_c.device)) if use_g else dummy_g

                n_rows_q = B * H * Lq * G
                _path_cp_cross_softmax_bwd_kernel[(n_rows_q,)](
                    z_raw, dp_raw, lse_q, delta_q, cq_q, ck_q, dc_self_q,
                    float(scale), rank, Lq, L, B * H,
                    G=G, BLOCK_N=BLOCK_N_CP, USE_G=use_g,
                    num_warps=4, num_stages=1,
                )
                if use_g:
                    if not (q0 == 0 and q1 == L):
                        dc_self_lg[:, :, q0:q1, :] = dc_self_q
                    dc_send_hg.sub_(dp_raw.view(rank, B, H, Lq, G, L).sum(dim=3, dtype=torch.float32))

                dV_stack.add_(torch.matmul(z_raw.transpose(-1, -2), do_blk).to(torch.float32))
                del z_raw
                B_contrib[:, :, :, q0:q1, :, :] = torch.matmul(dp_raw, kh_stack_scaled).view(rank, B, H, Lq, G, K)
                dK_hat_stack.add_(torch.matmul(dp_raw.transpose(-1, -2), x_blk).to(torch.float32), alpha=scale)
                del dp_raw

            dV_send[:rank] = dV_stack.permute(0, 1, 3, 2, 4)
            dK_hat_send[:rank] = dK_hat_stack.permute(0, 1, 3, 2, 4)
            if use_g:
                dc_self_cross = dc_self_lg.permute(0, 2, 1, 3).contiguous().view(B, L, HQ)
                dc_send[:rank] = dc_send_hg.permute(0, 1, 4, 2, 3).contiguous().view(rank, B, L, HQ)

            x_stack_flat = x_stack.view(rank, B, H, L * G, K)
            B_contrib_flat = B_contrib.view(rank, B, H, L * G, K)
            for p in range(0, rank):
                if p > 0:
                    dF_send[p] = -torch.matmul(x_stack_flat[p].transpose(-1, -2), B_run).to(torch.float32)
                    B_run = torch.matmul(B_run, (eye_mm - F_all[p].to(mm_dtype)).transpose(-1, -2))
                B_run = B_run + B_contrib_flat[p]
            del x_stack, x_stack_flat, B_contrib, B_contrib_flat, dV_stack, dK_hat_stack

        # Reduce-scatter cross-rank gradients back to each owning shard
        dK_hat_ext = torch.empty((B, L, H, K), dtype=torch.float32, device=q_c.device)
        dV_ext_tok = torch.empty((B, L, H, V_DIM), dtype=torch.float32, device=q_c.device)
        dF_ext = torch.empty((B, H, K, K), dtype=torch.float32, device=q_c.device)
        if world_size > 1:
            dist.reduce_scatter_tensor(dK_hat_ext, dK_hat_send, group=cp_group)
            dist.reduce_scatter_tensor(dV_ext_tok, dV_send, group=cp_group)
            dist.reduce_scatter_tensor(dF_ext, dF_send, group=cp_group)
        else:
            dK_hat_ext.copy_(dK_hat_send[0])
            dV_ext_tok.copy_(dV_send[0])
            dF_ext.copy_(dF_send[0])

        if use_g:
            dc_recv = torch.empty((B, L, HQ), dtype=torch.float32, device=q_c.device)
            if world_size > 1:
                dist.reduce_scatter_tensor(dc_recv, dc_send, group=cp_group)
            else:
                dc_recv.copy_(dc_send[0])
            dc_ext_local = dc_self_cross + dc_recv
        else:
            dc_ext_local = None

        # Reverse shard summary (F_shard, K_hat) into (dK_arrow_ext, dW_ext, dU_ext)
        dK_arrow_ext, dW_ext, dU_ext = _backward_shard_summary_triton(
            K_arrow, w_c, U_buf, F_suf, dK_hat_ext, dF_ext
        )
        b_init_ext = B_run.to(torch.float32).view(B, H, L, G, K).permute(0, 2, 1, 3, 4).contiguous().view(B, L, HQ, K)
        if rank > 0 and ctx.nt_shard > 1:
            # X_init[cq] = Q_arrow[cq] @ P_pre[cq], where P_pre[0] = I, P_pre[m+1] = (I - W_m^T U_m) P_pre[m].
            # Fused strip-mined [BR=32, K] 0-spill Triton kernel replacing 30-step Python loop (90 bmm launches).
            nt_s = ctx.nt_shard
            BR = 32
            n_strips = K // BR
            q_arr_hg = Q_arrow.view(B, nt_s, H, G, 32, K).contiguous().view(B, nt_s, H, G * 32, K).to(mm_dtype)
            b_init_hg = B_run.view(B, H, nt_s, 32, G, K).permute(0, 2, 1, 4, 3, 5).contiguous().view(B, nt_s, H, G * 32, K).to(mm_dtype)
            # Notice: dP_contrib^T = (Q_arrow^T @ B_init)^T = B_init^T @ Q_arrow!
            dP_contrib_T = torch.matmul(b_init_hg[:, 1:].transpose(-1, -2), q_arr_hg[:, 1:]).to(torch.float32).contiguous()  # [B, nt_s-1, H, K, K]
            Pt_ws = torch.empty((n_strips, B * H, nt_s - 1, BR, K), dtype=w_c.dtype, device=q_c.device)
            At_ws = torch.empty((n_strips, B * H, nt_s - 1, BR, 32), dtype=w_c.dtype, device=q_c.device)
            dW_xinit_strip = torch.empty((n_strips, B * H, nt_s, 32, K), dtype=torch.float32, device=q_c.device)
            dU_xinit_strip = torch.empty((n_strips, B * H, nt_s, 32, K), dtype=torch.float32, device=q_c.device)
            _path_cp_xinit_prefix_ws_kernel[(n_strips, B * H)](
                w_c, U_buf, Pt_ws, At_ws, dW_xinit_strip, dU_xinit_strip,
                nt_s, L, H=H, K=K, BT=32, BR=BR, num_warps=4, num_stages=1,
            )
            _path_cp_xinit_prefix_bwd_kernel[(n_strips, B * H)](
                w_c, U_buf, dP_contrib_T, Pt_ws, At_ws, dW_xinit_strip, dU_xinit_strip,
                nt_s, L, H=H, K=K, BT=32, BR=BR, num_warps=4, num_stages=1,
            )
            dW_ext = dW_ext + dW_xinit_strip.sum(dim=0)
            dU_ext = dU_ext + dU_xinit_strip.sum(dim=0)
            del Pt_ws, At_ws, dW_xinit_strip, dU_xinit_strip, dP_contrib_T

        dV_ext_chunk = dV_ext_tok.view(B, ctx.nt_shard, 32, H, V_DIM).permute(0, 1, 3, 2, 4).contiguous()

        dq_loc, dk_loc, dv_loc, dw_loc, dbeta_loc, dg_loc = _run_path_fox_ut_duality_bwd_raw(
            q_c, k_c, v_c, w_c, beta_c, g_c,
            o_final, do_local.contiguous(), lse_final,
            A_inv, Q_arrow, K_arrow, U_buf, c_loc,
            bos, eos, fc, lc, loc, ctx.nt_shard, scale,
            b_init_ext=b_init_ext,
            dk_arrow_ext=dK_arrow_ext,
            dv_ext=dV_ext_chunk,
            dw_ext=dW_ext,
            du_ext=dU_ext,
            dc_ext=dc_ext_local,
        )

        # In global cumsum c_global = c_loc + sum_{p < rank} sum_t g_p[t],
        # any dc_ext_local[t] on rank r also flows to every token of every earlier shard p < r!
        if use_g and world_size > 1:
            dc_sum_loc = dc_ext_local.sum(dim=1, keepdim=True)  # [B, 1, HQ]
            dc_sum_all = torch.empty((world_size, B, 1, HQ), dtype=torch.float32, device=q_c.device)
            dist.all_gather_into_tensor(dc_sum_all, dc_sum_loc.contiguous(), group=cp_group)
            suffix_dc = dc_sum_all[rank + 1 :].sum(dim=0) if rank + 1 < world_size else torch.zeros_like(dc_sum_loc)
            dg_loc = (dg_loc.to(torch.float32) + suffix_dc).to(g_c.dtype)

        if dw_loc.dtype != ctx.orig_w_dtype:
            dw_loc = dw_loc.to(ctx.orig_w_dtype)

        return dq_loc, dk_loc, dv_loc, dw_loc, dbeta_loc, dg_loc, None, None


def ring_path_cp_attention(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    w_local: torch.Tensor,
    beta_local: torch.Tensor,
    g_local: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return RingPaTHCPAttentionFunction.apply(q_local, k_local, v_local, w_local, beta_local, g_local, scale, cp_group)


# =============================================================================
# 6. Exact FP64 Token-by-Token Reference Oracle & Official/Naive Baselines
# =============================================================================

def path_frontend_exact_oracle_fp64(
    u_raw: torch.Tensor,
    conv_w: torch.Tensor,
    beta_raw: torch.Tensor,
    beta_scale: float = 2.0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact FP64 mathematical specification of the PaTH Householder reflector frontend:
      z_t = sum_{d=0}^2 conv_w[h, d] * u_{t-d}   (masked at sequence BOS boundary)
      y_t = silu(z_t) = z_t * sigmoid(z_t)
      w_t = y_t / sqrt(||y_t||_2^2 + 1e-12)
      beta_t = beta_scale * sigmoid(beta_raw_t)
    """
    B, T, H, K = u_raw.shape
    ud = u_raw.to(torch.float64)
    cwd = conv_w.to(torch.float64)
    btd = beta_raw.to(torch.float64)

    if cu_seqlens is None:
        seq_ranges = [(0, T)]
    else:
        cu_list = cu_seqlens.detach().cpu().tolist()
        seq_ranges = [(int(cu_list[s]), int(cu_list[s + 1])) for s in range(len(cu_list) - 1)]

    w_chunks: List[torch.Tensor] = []
    for bos_t, eos_t in seq_ranges:
        u0 = ud[:, bos_t:eos_t]  # [B, L, H, K]
        u1 = F.pad(u0[:, :-1], (0, 0, 0, 0, 1, 0)) if (eos_t - bos_t) > 1 else torch.zeros_like(u0)
        u2 = F.pad(u0[:, :-2], (0, 0, 0, 0, 2, 0)) if (eos_t - bos_t) > 2 else torch.zeros_like(u0)
        z = (
            u0 * cwd[:, 0, :].unsqueeze(0).unsqueeze(0)
            + u1 * cwd[:, 1, :].unsqueeze(0).unsqueeze(0)
            + u2 * cwd[:, 2, :].unsqueeze(0).unsqueeze(0)
        )
        y = F.silu(z)
        inv_r = torch.rsqrt((y * y).sum(dim=-1, keepdim=True) + 1e-12)
        w_chunks.append(y * inv_r)

    w_out = torch.cat(w_chunks, dim=1)
    beta_out = float(beta_scale) * torch.sigmoid(btd)
    return w_out, beta_out


def path_frontend_naive_eager(
    u_raw: torch.Tensor,
    conv_w: torch.Tensor,
    beta_raw: torch.Tensor,
    beta_scale: float = 2.0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unfused Eager PyTorch frontend (Track 2 Baseline 1): separate depthwise Conv1d(k=3) + SiLU + L2-norm + Sigmoid.
    """
    B, T, H, K = u_raw.shape
    uf = u_raw.to(torch.float32)
    cwf = conv_w.to(torch.float32)
    btf = beta_raw.to(torch.float32)

    if cu_seqlens is None:
        u1 = F.pad(uf[:, :-1], (0, 0, 0, 0, 1, 0))
        u2 = F.pad(uf[:, :-2], (0, 0, 0, 0, 2, 0))
        z = (
            uf * cwf[:, 0, :].unsqueeze(0).unsqueeze(0)
            + u1 * cwf[:, 1, :].unsqueeze(0).unsqueeze(0)
            + u2 * cwf[:, 2, :].unsqueeze(0).unsqueeze(0)
        )
    else:
        cu_list = cu_seqlens.detach().cpu().tolist()
        z_parts = []
        for s in range(len(cu_list) - 1):
            bos_t, eos_t = int(cu_list[s]), int(cu_list[s + 1])
            u0 = uf[:, bos_t:eos_t]
            u1 = F.pad(u0[:, :-1], (0, 0, 0, 0, 1, 0)) if (eos_t - bos_t) > 1 else torch.zeros_like(u0)
            u2 = F.pad(u0[:, :-2], (0, 0, 0, 0, 2, 0)) if (eos_t - bos_t) > 2 else torch.zeros_like(u0)
            z_parts.append(
                u0 * cwf[:, 0, :].unsqueeze(0).unsqueeze(0)
                + u1 * cwf[:, 1, :].unsqueeze(0).unsqueeze(0)
                + u2 * cwf[:, 2, :].unsqueeze(0).unsqueeze(0)
            )
        z = torch.cat(z_parts, dim=1)

    y = F.silu(z)
    w_unit = (y * torch.rsqrt((y * y).sum(dim=-1, keepdim=True) + 1e-12)).to(u_raw.dtype)
    beta_out = (float(beta_scale) * torch.sigmoid(btf)).to(beta_raw.dtype)
    return w_unit, beta_out


def path_fox_exact_oracle_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact FP64 mathematical specification of PaTH-FoX (Yang et al., arXiv:2505.16381v2):
      For j <= i within the same sequence:
        P_{j+1:i} = H_{j+1} H_{j+2} ... H_i   (where H_t = I_K - beta_t w_t w_t^T)
        Z_{i, j}  = scale * (k_j^T P_{j+1:i} q_i) + (c_i - c_j)
    Differentiable in FP64 for exact autograd and O(h^2) Richardson extrapolation.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V_DIM = v.shape[3]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5

    qd = q.to(torch.float64)
    kd = k.to(torch.float64)
    vd = v.to(torch.float64)
    wd = w.to(torch.float64)
    bd = beta.to(torch.float64)
    gd = g.to(torch.float64) if g is not None else None

    if cu_seqlens is None:
        seq_ranges = [(0, T)]
    else:
        cu_list = cu_seqlens.detach().cpu().tolist()
        seq_ranges = [(int(cu_list[s]), int(cu_list[s + 1])) for s in range(len(cu_list) - 1)]

    o_chunks: List[torch.Tensor] = []
    lse_chunks: List[torch.Tensor] = []

    for bos_t, eos_t in seq_ranges:
        L = eos_t - bos_t
        q_s = qd[:, bos_t:eos_t].view(B, L, H, G, K).permute(0, 2, 3, 1, 4)  # [B, H, G, L, K]
        k_s = kd[:, bos_t:eos_t].permute(0, 2, 1, 3)                         # [B, H, L, K]
        v_s = vd[:, bos_t:eos_t].permute(0, 2, 1, 3)                         # [B, H, L, V_DIM]
        w_s = wd[:, bos_t:eos_t].permute(0, 2, 1, 3)                         # [B, H, L, K]
        b_s = bd[:, bos_t:eos_t].permute(0, 2, 1)                            # [B, H, L]

        # Build exact transformed queries q_tilde[i, j] = q_i @ H_i @ ... @ H_{j+1} for j <= i
        score_rows: List[torch.Tensor] = []
        for i in range(L):
            qi = q_s[:, :, :, i, :]  # [B, H, G, K]
            row_scores: List[torch.Tensor] = []
            for j in range(i, -1, -1):
                kj = k_s[:, :, j, :]  # [B, H, K]
                s_ij = (qi * kj.unsqueeze(2)).sum(dim=-1) * float(scale)  # [B, H, G]
                row_scores.append(s_ij)
                wj = w_s[:, :, j, :]
                bj = b_s[:, :, j]
                proj = (qi * wj.unsqueeze(2)).sum(dim=-1, keepdim=True)
                qi = qi - bj.unsqueeze(-1).unsqueeze(-1) * proj * wj.unsqueeze(2)
            # row_scores is in order j = i, i-1, ..., 0 -> reverse to j = 0 .. i
            row_scores_rev = torch.stack(row_scores[::-1], dim=-1)  # [B, H, G, i+1]
            if i + 1 < L:
                pad = torch.full((B, H, G, L - (i + 1)), -float("inf"), dtype=torch.float64, device=q.device)
                row_full = torch.cat([row_scores_rev, pad], dim=-1)
            else:
                row_full = row_scores_rev
            score_rows.append(row_full)

        Z = torch.stack(score_rows, dim=-2)  # [B, H, G, L, L]
        if gd is not None:
            c_s = torch.cumsum(gd[:, bos_t:eos_t], dim=1).view(B, L, H, G).permute(0, 2, 3, 1)  # [B, H, G, L]
            causal_m = torch.tril(torch.ones((L, L), dtype=torch.bool, device=q.device))
            Z = torch.where(causal_m, Z + (c_s.unsqueeze(-1) - c_s.unsqueeze(-2)), Z)

        lse_s = torch.logsumexp(Z, dim=-1)  # [B, H, G, L]
        P_attn = torch.exp(Z - lse_s.unsqueeze(-1))
        o_s = torch.matmul(P_attn, v_s.unsqueeze(2))  # [B, H, G, L, V_DIM]
        o_chunks.append(o_s.permute(0, 3, 1, 2, 4).reshape(B, L, HQ, V_DIM))
        lse_chunks.append(lse_s.permute(0, 3, 1, 2).reshape(B, L, HQ))

    return torch.cat(o_chunks, dim=1), torch.cat(lse_chunks, dim=1)


def path_fox_naive_chunk_eager_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive Eager PyTorch Chunk-Householder Baseline (Track 1 Baseline 1):
    Explicitly materializes dense [K, K] Householder reflector products per chunk
    `(I - W_c^T T_c W_c)` (`P_chunk`), intermediate lag-propagated query states, and
    the full causal score/probability matrix in HBM.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V_DIM = v.shape[3]
    G = HQ // H
    BT = 64 if (T % 64 == 0) else 32
    NT = T // BT
    if scale is None:
        scale = K ** -0.5

    qf = q.view(B, NT, BT, H, G, K).permute(0, 3, 1, 4, 2, 5).to(torch.float32)  # [B, H, NT, G, BT, K]
    kf = k.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4).to(torch.float32)        # [B, H, NT, BT, K]
    vf = v.view(B, NT, BT, H, V_DIM).permute(0, 3, 1, 2, 4).to(torch.float32)    # [B, H, NT, BT, V_DIM]
    wf = w.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4).to(torch.float32)        # [B, H, NT, BT, K]
    bf = beta.view(B, NT, BT, H).permute(0, 3, 1, 2).to(torch.float32)           # [B, H, NT, BT]

    idx = torch.arange(BT, device=q.device)
    m_tril = idx[:, None] >= idx[None, :]
    m_strict = idx[:, None] > idx[None, :]
    eye_bt = torch.eye(BT, dtype=torch.float32, device=q.device)
    eye_k = torch.eye(K, dtype=torch.float32, device=q.device)

    L_unit = eye_bt + torch.where(m_strict, torch.matmul(wf * bf.unsqueeze(-1), wf.transpose(-1, -2)), 0.0)
    A_inv = torch.linalg.solve_triangular(L_unit, eye_bt.expand_as(L_unit), upper=False, unitriangular=True)
    T_mat = A_inv * bf.unsqueeze(-2)
    U_mat = torch.matmul(T_mat, wf)

    # Materialize dense [B, H, NT, K, K] chunk transition matrices P_c = I_K - W_c^T U_c
    P_chunk = eye_k - torch.matmul(wf.transpose(-1, -2), U_mat)
    M_wbk = torch.where(m_strict, torch.matmul(wf, kf.transpose(-1, -2)), 0.0)
    K_arrow = kf - torch.matmul(torch.matmul(T_mat, M_wbk).transpose(-1, -2), wf)

    M_qw = torch.where(m_tril, torch.matmul(qf, wf.unsqueeze(3).transpose(-1, -2)), 0.0)
    N_qwt = torch.matmul(M_qw, T_mat.unsqueeze(3))
    Q_arrow = qf - torch.matmul(N_qwt, wf.unsqueeze(3))

    if g is not None:
        cf = torch.cumsum(g.to(torch.float32), dim=1).view(B, NT, BT, H, G).permute(0, 3, 1, 4, 2)
    else:
        cf = None

    # Intra-chunk scores across all NT chunks simultaneously: [B, H, NT, G, BT, BT]
    b_raw = torch.matmul(qf, kf.unsqueeze(3).transpose(-1, -2)) - torch.matmul(N_qwt, M_wbk.unsqueeze(3))
    z_diag = torch.where(m_tril, b_raw * float(scale), -float("inf"))
    if cf is not None:
        z_diag = torch.where(m_tril, z_diag + (cf.unsqueeze(-1) - cf.unsqueeze(-2)), -float("inf"))

    # Materialize dense [B, H, G, NT, NT, BT, BT] causal block score matrix in HBM
    Z_blocks = torch.full((B, H, G, NT, NT, BT, BT), -float("inf"), dtype=torch.float32, device=q.device)
    diag_idx = torch.arange(NT, device=q.device)
    Z_blocks[:, :, :, diag_idx, diag_idx] = z_diag.permute(0, 1, 3, 2, 4, 5)

    if NT > 1:
        q_state = Q_arrow[:, :, 1:]  # [B, H, NT-1, G, BT, K] for cq = 1 .. NT-1
        for lag in range(1, NT):
            n_act = NT - lag
            k_lag = K_arrow[:, :, :n_act].unsqueeze(3)  # [B, H, n_act, 1, BT, K]
            z_lag = torch.matmul(q_state, k_lag.transpose(-1, -2)) * float(scale)  # [B, H, n_act, G, BT, BT]
            if cf is not None:
                c_q_lag = cf[:, :, lag:]    # [B, H, n_act, G, BT]
                c_k_lag = cf[:, :, :n_act]  # [B, H, n_act, G, BT]
                z_lag = z_lag + (c_q_lag.unsqueeze(-1) - c_k_lag.unsqueeze(-2))
            cq_idx = torch.arange(lag, NT, device=q.device)
            ck_idx = torch.arange(0, n_act, device=q.device)
            Z_blocks[:, :, :, cq_idx, ck_idx] = z_lag.permute(0, 1, 3, 2, 4, 5)
            if lag + 1 < NT:
                # Advance q_state for cq = lag+1 .. NT-1 through P_chunk[cq - lag]^T where cq - lag = 1 .. n_act-1
                p_trans = P_chunk[:, :, 1:n_act].unsqueeze(3).transpose(-1, -2)
                q_state = torch.matmul(q_state[:, :, 1:], p_trans)

    Z_full = Z_blocks.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, H, G, T, T)
    v_full = vf.permute(0, 1, 2, 3, 4).reshape(B, H, 1, T, V_DIM)
    lse_full = torch.logsumexp(Z_full, dim=-1)
    p_full = torch.softmax(Z_full, dim=-1)
    o_full = torch.matmul(p_full, v_full)

    o_out = o_full.permute(0, 3, 1, 2, 4).reshape(B, T, HQ, V_DIM).to(v.dtype)
    lse_out = lse_full.permute(0, 3, 1, 2).reshape(B, T, HQ)
    return o_out, lse_out


def official_fla_parallel_path_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direct wrapper around official `fla-org/flash-linear-attention` `parallel_path_attn`
    installed on node-0 (`/tmp/fla_repo/fla/ops/path_attn/`).
    """
    import sys
    if "/tmp/fla_repo" not in sys.path:
        sys.path.insert(0, "/tmp/fla_repo")
    from fla.ops.path_attn.parallel import parallel_path_attn
    eff_scale = (k.shape[-1] ** -0.5) if scale is None else float(scale)
    w_f32 = w if w.dtype == torch.float32 else w.to(torch.float32)
    beta_f32 = beta if beta.dtype == torch.float32 else beta.to(torch.float32)
    # In fla's intra_chunk_preprocess_fwd_kernel / parallel_path_fwd_kernel,
    # g_cumsum is added to unscaled dot products before multiplying by `scale`.
    # Dividing `g` by `eff_scale` aligns fla with Eq. (5) natural-log forget gate
    # logits `scale * (q @ P @ k^T) + (c_i - c_j)` in both forward and backward (dg).
    g_f32 = (g.to(torch.float32) / eff_scale) if g is not None else None
    return parallel_path_attn(
        q=q, k=k, v=v, w=w_f32, beta=beta_f32, g=g_f32, scale=eff_scale, cu_seqlens=cu_seqlens
    )


# =============================================================================
# 7. Cubin Register & Spill Introspection Audit (Gate G4 / Rule 30)
# =============================================================================

def audit_round8_triton_cubin_spills(device: torch.device) -> Dict[str, Any]:
    """
    Compile and warm up all Round-8 RSI Triton kernels across both K = 64 and K = 128,
    extracting `.n_regs` and `.n_spills` from every compiled cubin, and comparing
    directly against official `fla-org` `chunk_cumprod_householder_bwd_kernel` at K = 128.
    """
    B, T, HQ, H = 1, 256, 8, 2
    for K_dim in (64, 128):
        u_raw = torch.randn((B, T, H, K_dim), dtype=torch.bfloat16, device=device, requires_grad=True)
        conv_w = torch.randn((H, 3, K_dim), dtype=torch.bfloat16, device=device, requires_grad=True)
        bt_raw = torch.randn((B, T, H), dtype=torch.bfloat16, device=device, requires_grad=True)
        q = torch.randn((B, T, HQ, K_dim), dtype=torch.bfloat16, device=device, requires_grad=True)
        k = torch.randn((B, T, H, K_dim), dtype=torch.bfloat16, device=device, requires_grad=True)
        v = torch.randn((B, T, H, K_dim), dtype=torch.bfloat16, device=device, requires_grad=True)
        g = F.logsigmoid(torch.randn((B, T, HQ), dtype=torch.float32, device=device)).to(torch.bfloat16).detach().requires_grad_(True)

        w_unit, beta = path_frontend_conv3_silu_l2norm(u_raw, conv_w, bt_raw)
        o, _ = path_fox_ut_duality_attention(q, k, v, w_unit, beta, g=g)
        loss = o.sum()
        loss.backward()

        # Warm up Ring PaTH-CP shard defect + cross-shard softmax + xinit prefix kernels across K=64 and K=128
        nt_s = T // 32
        w_s = w_unit.detach().contiguous()
        u_s = torch.randn((B, nt_s, H, 32, K_dim), dtype=torch.bfloat16, device=device)
        k_arr_s = torch.randn((B, nt_s, H, 32, K_dim), dtype=torch.bfloat16, device=device)
        f_shard_s, f_suf_s, k_hat_s = _compute_shard_summary_triton(k_arr_s, w_s, u_s)
        dk_hat_s = torch.randn_like(k_hat_s)
        df_shard_s = torch.randn_like(f_shard_s)
        _backward_shard_summary_triton(k_arr_s, w_s, u_s, f_suf_s, dk_hat_s, df_shard_s)

        n_strips_s = K_dim // 32
        dp_c_s = torch.randn((B, nt_s - 1, H, K_dim, K_dim), dtype=torch.float32, device=device)
        pt_ws_s = torch.empty((n_strips_s, B * H, nt_s - 1, 32, K_dim), dtype=torch.bfloat16, device=device)
        at_ws_s = torch.empty((n_strips_s, B * H, nt_s - 1, 32, 32), dtype=torch.bfloat16, device=device)
        dw_x_s = torch.empty((n_strips_s, B * H, nt_s, 32, K_dim), dtype=torch.float32, device=device)
        du_x_s = torch.empty((n_strips_s, B * H, nt_s, 32, K_dim), dtype=torch.float32, device=device)
        _path_cp_xinit_prefix_ws_kernel[(n_strips_s, B * H)](
            w_s, u_s, pt_ws_s, at_ws_s, dw_x_s, du_x_s,
            nt_s, T, H=H, K=K_dim, BT=32, BR=32, num_warps=4, num_stages=1,
        )
        _path_cp_xinit_prefix_bwd_kernel[(n_strips_s, B * H)](
            w_s, u_s, dp_c_s, pt_ws_s, at_ws_s, dw_x_s, du_x_s,
            nt_s, T, H=H, K=K_dim, BT=32, BR=32, num_warps=4, num_stages=1,
        )

        G_s = HQ // H
        z_raw_s = torch.randn((2, B, H, T * G_s, T), dtype=torch.bfloat16, device=device)
        dp_raw_s = torch.randn((2, B, H, T * G_s, T), dtype=torch.bfloat16, device=device)
        lse_s = torch.randn((B, H, T, G_s), dtype=torch.float32, device=device)
        cq_s = torch.randn((B, H, T, G_s), dtype=torch.float32, device=device)
        ck_s = torch.randn((2, B, H, G_s, T), dtype=torch.float32, device=device)
        lse_out_s = torch.empty_like(lse_s)
        alpha_out_s = torch.empty_like(lse_s)
        dc_self_s = torch.empty_like(lse_s)
        n_rows_s = B * H * T * G_s
        _path_cp_cross_softmax_fwd_kernel[(n_rows_s,)](
            z_raw_s, lse_s, cq_s, ck_s, lse_out_s, alpha_out_s,
            float(K_dim ** -0.5), 2, T, T, B * H,
            G=G_s, BLOCK_N=128, USE_G=True, num_warps=4, num_stages=1,
        )
        _path_cp_cross_softmax_bwd_kernel[(n_rows_s,)](
            z_raw_s, dp_raw_s, lse_s, cq_s, cq_s, ck_s, dc_self_s,
            float(K_dim ** -0.5), 2, T, T, B * H,
            G=G_s, BLOCK_N=128, USE_G=True, num_warps=4, num_stages=1,
        )
    torch.cuda.synchronize(device)

    def _collect_compiled(kobj):
        if not hasattr(kobj, "device_caches") and hasattr(kobj, "fn"):
            kobj = kobj.fn
        found = []
        for dev_entry in getattr(kobj, "device_caches", {}).values():
            cdict = dev_entry[0] if isinstance(dev_entry, tuple) and len(dev_entry) > 0 else dev_entry
            if isinstance(cdict, dict):
                for comp in cdict.values():
                    if hasattr(comp, "n_spills"):
                        found.append(comp)
        if not found:
            for cv in getattr(kobj, "cache", {}).values():
                entries = list(cv.values()) if isinstance(cv, dict) else [cv]
                for comp in entries:
                    if hasattr(comp, "n_spills"):
                        found.append(comp)
        return found

    round8_kernels = [
        ("_path_frontend_conv3_silu_l2norm_fwd_kernel", _path_frontend_conv3_silu_l2norm_fwd_kernel),
        ("_path_frontend_conv3_silu_l2norm_bwd_kernel", _path_frontend_conv3_silu_l2norm_bwd_kernel),
        ("_path_frontend_conv3_gather_du_bwd_kernel", _path_frontend_conv3_gather_du_bwd_kernel),
        ("_path_fox_gate_fwd_cumsum_kernel", _path_fox_gate_fwd_cumsum_kernel),
        ("_path_fox_gate_bwd_rev_cumsum_kernel", _path_fox_gate_bwd_rev_cumsum_kernel),
        ("_path_ut_solve_tril_32x32_fwd_kernel", _path_ut_solve_tril_32x32_fwd_kernel),
        ("_path_intra_chunk_ut_fwd_kernel", _path_intra_chunk_ut_fwd_kernel),
        ("_path_inter_chunk_streaming_fwd_kernel", _path_inter_chunk_streaming_fwd_kernel),
        ("_path_bwd_preprocess_delta_kernel", _path_bwd_preprocess_delta_kernel),
        ("_path_inter_chunk_query_worker_bwd_kernel", _path_inter_chunk_query_worker_bwd_kernel),
        ("_path_inter_chunk_key_duality_worker_bwd_kernel", _path_inter_chunk_key_duality_worker_bwd_kernel),
        ("_path_reduce_dw_du_priv_kernel", _path_reduce_dw_du_priv_kernel),
        ("_path_intra_chunk_score_dv_bwd_kernel", _path_intra_chunk_score_dv_bwd_kernel),
        ("_path_intra_chunk_ut_matrices_bwd_kernel", _path_intra_chunk_ut_matrices_bwd_kernel),
        ("_path_intra_chunk_ut_vectors_bwd_kernel", _path_intra_chunk_ut_vectors_bwd_kernel),
        ("_path_cp_shard_defect_fwd_kernel", _path_cp_shard_defect_fwd_kernel),
        ("_path_cp_apply_khat_fwd_kernel", _path_cp_apply_khat_fwd_kernel),
        ("_path_cp_shard_defect_bwd_kernel", _path_cp_shard_defect_bwd_kernel),
        ("_path_cp_xinit_prefix_ws_kernel", _path_cp_xinit_prefix_ws_kernel),
        ("_path_cp_xinit_prefix_bwd_kernel", _path_cp_xinit_prefix_bwd_kernel),
        ("_path_cp_cross_softmax_fwd_kernel", _path_cp_cross_softmax_fwd_kernel),
        ("_path_cp_cross_softmax_bwd_kernel", _path_cp_cross_softmax_bwd_kernel),
    ]

    kernel_report: Dict[str, Dict[str, int]] = {}
    total_round8_spills = 0
    total_compiled_specializations = 0

    for kname, kfn in round8_kernels:
        compiled_entries = _collect_compiled(kfn)
        max_regs = 0
        max_spills = 0
        for comp in compiled_entries:
            max_regs = max(max_regs, int(getattr(comp, "n_regs", 0) or 0))
            max_spills = max(max_spills, int(getattr(comp, "n_spills", 0) or 0))
        n_specs = len(compiled_entries)
        total_round8_spills += max_spills
        total_compiled_specializations += n_specs
        kernel_report[kname] = {
            "specializations": n_specs,
            "max_n_regs": max_regs,
            "max_n_spills": max_spills,
        }

    # Also measure official fla-org cubin spills at K = 64 and K = 128
    official_fla_spills: Dict[str, Any] = {}
    try:
        from fla.ops.path_attn.cumprod_householder_bwd import chunk_cumprod_householder_bwd_kernel
        for K_dim in (64, 128):
            q_f = torch.randn((1, 1024, 8, K_dim), dtype=torch.bfloat16, device=device, requires_grad=True)
            k_f = torch.randn((1, 1024, 2, K_dim), dtype=torch.bfloat16, device=device, requires_grad=True)
            v_f = torch.randn((1, 1024, 2, K_dim), dtype=torch.bfloat16, device=device, requires_grad=True)
            w_f = F.normalize(torch.randn((1, 1024, 2, K_dim), dtype=torch.float32, device=device), dim=-1).detach().requires_grad_(True)
            b_f = (torch.sigmoid(torch.randn((1, 1024, 2), dtype=torch.float32, device=device)) * 1.8).detach().requires_grad_(True)
            g_f = F.logsigmoid(torch.randn((1, 1024, 8), dtype=torch.float32, device=device)).detach().requires_grad_(True)
            o_f, _ = official_fla_parallel_path_attn(q_f, k_f, v_f, w_f, b_f, g=g_f)
            o_f.sum().backward()

        fla_max_spills = 0
        fla_max_regs = 0
        for comp in _collect_compiled(chunk_cumprod_householder_bwd_kernel):
            fla_max_spills = max(fla_max_spills, int(getattr(comp, "n_spills", 0) or 0))
            fla_max_regs = max(fla_max_regs, int(getattr(comp, "n_regs", 0) or 0))
        official_fla_spills["chunk_cumprod_householder_bwd_kernel"] = {
            "max_n_regs": fla_max_regs,
            "max_n_spills": fla_max_spills,
        }
    except Exception as exc:
        official_fla_spills["error"] = str(exc)

    return {
        "total_round8_spills": total_round8_spills,
        "total_compiled_specializations": total_compiled_specializations,
        "all_round8_zero_spills": (total_round8_spills == 0 and all(v["specializations"] > 0 for v in kernel_report.values())),
        "round8_kernels": kernel_report,
        "official_fla_reference": official_fla_spills,
    }
