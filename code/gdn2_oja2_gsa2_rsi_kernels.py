#!/usr/bin/env python3
"""
Round-20 RSI (E146, v137): Zero-Spill SM80 Triton Kernels for Gated DeltaNet-2 (GDN-2),
Gated Oja Rule 2 (Oja-2), and Hierarchical Gated Slot Attention 2 (GSA-2)
=======================================================================================

Targets:
  1. Gated DeltaNet-2 (`NVlabs/GatedDeltaNet-2`, `fla/ops/gdn2` & `fla/layers/gdn2.py`):
     Left-acting recurrence on S_t in R^{K x V} with channel-wise erase gate b_t in R^K,
     channel-wise write gate w_t in R^V, and channel-wise decay g_t in R^K:
       S_t = (I_K - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
  2. Gated Oja Rule 2 (`fla/ops/gated_oja_rule2`):
     Right-acting slot-memory dual on S_t in R^{K x M} (M in {64, 128, 256}) with per-slot
     erase gate b_t in R^M, per-channel write gate c_t in R^K, L2-normalized slot code
     v_t in R^M, and per-slot decay gv_t in [-5, 0]^M:
       S_t = S_{t-1} Diag(exp(gv_t)) (I_M - (b_t * v_t) v_t^T) + (c_t * k_t) v_t^T
  3. Gated Slot Attention 2 (`fla/layers/gsa2.py`):
     Hierarchical two-level cascade feeding Level-1 Oja-2 slot readout q_t^(2) = act(o_t^(1))
     and slot code k_t^(2) = v_t into Level-2 GDN-2 value memory.

All Triton kernels in this module are engineered for NVIDIA A100-SXM4-80GB (SM80) with:
  - 0 cubin register spills (max_regs <= 168 across 100% of compiled specializations)
  - Span-checked midpoint-anchored 16x16 Tensor Core sub-chunking (eliminating all five
    16-step scalar SIMT diagonal loops in GDN-2 and Oja-2)
  - Two-pass state-split WY backward adjoints (eliminating 206 spills in GDN-2 and 132 spills in Oja-2)
  - State-invariant BK * M = 4096 Oja-2 recurrence (BK = 64, 32, 16 for M = 64, 128, 256)
  - Saved forward states (eliminating redundant backward WY and fwd_h recomputations)
  - Superfused GDN2BlockRSI & GSA2BlockRSI (Stages A0..A4) and 8-GPU NVSwitch Ulysses CP=8
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

RCP_LN2: float = 1.4426950408889634


# ======================================================================================
# 1. Exact FP64 Reference Oracles (GDN-2, Oja-2, and Two-Level GSA-2)
# ======================================================================================

def gdn2_recurrent_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable FP64 token-by-token reference oracle for GDN-2."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    qf = q.double()
    kf = k.double()
    vf = v.double()
    gf = g.double()
    bf = b.double()
    wf = w.double()
    if use_qk_l2norm:
        qf = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6)
        kf = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)

    S = (
        initial_state.double().clone()
        if initial_state is not None
        else torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    )
    outs = []
    for t in range(T):
        qt = qf[:, t]                          # [B, H, K]
        kt = kf[:, t]                          # [B, H, K]
        vt = vf[:, t]                          # [B, H, V]
        gt = torch.exp(gf[:, t])               # [B, H, K]
        bt = bf[:, t]                          # [B, H, K]
        wt = wf[:, t]                          # [B, H, V]
        S_dec = S * gt.unsqueeze(-1)           # [B, H, K, V]
        proj = torch.einsum("bhk,bhkv->bhv", bt * kt, S_dec)
        v_new = wt * vt - proj
        S = S_dec + torch.einsum("bhk,bhv->bhkv", kt, v_new)
        ot = scale * torch.einsum("bhk,bhkv->bhv", qt, S)
        outs.append(ot)
    o = torch.stack(outs, dim=1)
    return o, S


def oja2_recurrent_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    use_q_l2norm: bool = False,
    use_k_l2norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable FP64 token-by-token reference oracle for Gated Oja Rule 2."""
    B, T, H, K = q.shape
    M = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    qf = q.double()
    kf = k.double()
    vf = v.double()
    gvf = gv.double()
    bf = b.double()
    cf = c.double()
    if use_q_l2norm:
        qf = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6)
    if use_k_l2norm:
        kf = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)

    S = (
        initial_state.double().clone()
        if initial_state is not None
        else torch.zeros(B, H, K, M, dtype=torch.float64, device=q.device)
    )
    outs = []
    for t in range(T):
        qt = qf[:, t]                          # [B, H, K]
        kt = kf[:, t]                          # [B, H, K]
        vt = vf[:, t]                          # [B, H, M]
        gvt = torch.exp(gvf[:, t])             # [B, H, M]
        bt = bf[:, t]                          # [B, H, M]
        ct = cf[:, t]                          # [B, H, K]
        S_dec = S * gvt.unsqueeze(-2)          # [B, H, K, M]
        proj_k = torch.einsum("bhm,bhkm->bhk", bt * vt, S_dec)
        k_new = ct * kt - proj_k
        S = S_dec + torch.einsum("bhk,bhm->bhkm", k_new, vt)
        ot = scale * torch.einsum("bhk,bhkm->bhm", qt, S)
        outs.append(ot)
    o = torch.stack(outs, dim=1)
    return o, S


# ======================================================================================
# 2. Shared Zero-Spill Utility Kernels (Chunk Cumsum, L2Norm, 64x64 Block Solve-Tril)
# ======================================================================================

@triton.jit(do_not_specialize=["T"])
def _chunk_cumsum_fwd_kernel(
    g_ptr,
    g_out_ptr,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_td = (o_t[:, None] < T) & (o_d[None, :] < D)
    base = (bos * H + i_h) * D
    p_g = g_ptr + base + o_t[:, None] * (H * D) + o_d[None, :]
    b_g = tl.load(p_g, mask=m_td, other=0.0).to(tl.float32) * scale
    b_cs = tl.cumsum(b_g, axis=0)
    p_out = g_out_ptr + base + o_t[:, None] * (H * D) + o_d[None, :]
    tl.store(p_out, b_cs, mask=m_td)


@triton.jit(do_not_specialize=["T"])
def _chunk_cumsum_rev_kernel(
    dg_ptr,
    dg_out_ptr,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_td = (o_t[:, None] < T) & (o_d[None, :] < D)
    base = (bos * H + i_h) * D
    p_dg = dg_ptr + base + o_t[:, None] * (H * D) + o_d[None, :]
    b_dg = tl.load(p_dg, mask=m_td, other=0.0).to(tl.float32)
    b_tot = tl.sum(b_dg, axis=0)[None, :]
    b_cs = (b_tot - tl.cumsum(b_dg, axis=0) + b_dg) * scale
    p_out = dg_out_ptr + base + o_t[:, None] * (H * D) + o_d[None, :]
    tl.store(p_out, b_cs, mask=m_td)


def chunk_local_cumsum_rsi(g: torch.Tensor, scale: float = 1.0, chunk_size: int = 64) -> torch.Tensor:
    B, T, H, D = g.shape
    BT = chunk_size
    BD = 64
    NT = triton.cdiv(T, BT)
    ND = triton.cdiv(D, BD)
    out = torch.empty_like(g, dtype=torch.float32)
    _chunk_cumsum_fwd_kernel[(ND, NT, B * H)](
        g, out, scale, T, H=H, D=D, BT=BT, BD=BD, num_warps=4, num_stages=1
    )
    return out


def chunk_local_cumsum_rev_rsi(dg: torch.Tensor, scale: float = 1.0, chunk_size: int = 64) -> torch.Tensor:
    B, T, H, D = dg.shape
    BT = chunk_size
    BD = 64
    NT = triton.cdiv(T, BT)
    ND = triton.cdiv(D, BD)
    out = torch.empty_like(dg, dtype=torch.float32)
    _chunk_cumsum_rev_kernel[(ND, NT, B * H)](
        dg, out, scale, T, H=H, D=D, BT=BT, BD=BD, num_warps=4, num_stages=1
    )
    return out


@triton.jit(do_not_specialize=["N_ROWS"])
def _l2norm_fwd_kernel(
    x_ptr,
    y_ptr,
    rstd_ptr,
    eps,
    N_ROWS,
    D: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
):
    i_b = tl.program_id(0).to(tl.int64)
    o_r = i_b * BR + tl.arange(0, BR)
    m_r = o_r < N_ROWS
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_rd = m_r[:, None] & m_d[None, :]
    b_x = tl.load(x_ptr + o_r[:, None] * D + o_d[None, :], mask=m_rd, other=0.0).to(tl.float32)
    b_rstd = tl.rsqrt(tl.sum(b_x * b_x, axis=1) + eps)
    tl.store(rstd_ptr + o_r, b_rstd, mask=m_r)
    b_y = b_x * b_rstd[:, None]
    tl.store(y_ptr + o_r[:, None] * D + o_d[None, :], b_y.to(y_ptr.dtype.element_ty), mask=m_rd)


@triton.jit(do_not_specialize=["N_ROWS"])
def _l2norm_bwd_kernel(
    x_ptr,
    dy_ptr,
    rstd_ptr,
    dx_ptr,
    N_ROWS,
    D: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
):
    i_b = tl.program_id(0).to(tl.int64)
    o_r = i_b * BR + tl.arange(0, BR)
    m_r = o_r < N_ROWS
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_rd = m_r[:, None] & m_d[None, :]
    b_x = tl.load(x_ptr + o_r[:, None] * D + o_d[None, :], mask=m_rd, other=0.0).to(tl.float32)
    b_dy = tl.load(dy_ptr + o_r[:, None] * D + o_d[None, :], mask=m_rd, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd_ptr + o_r, mask=m_r, other=0.0).to(tl.float32)[:, None]
    b_y = b_x * b_rstd
    b_dot = tl.sum(b_dy * b_y, axis=1)[:, None]
    b_dx = (b_dy - b_y * b_dot) * b_rstd
    tl.store(dx_ptr + o_r[:, None] * D + o_d[None, :], b_dx.to(dx_ptr.dtype.element_ty), mask=m_rd)


def l2norm_fwd_rsi(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    D = orig_shape[-1]
    BD = triton.next_power_of_2(D)
    BR = 32
    n_rows = x.numel() // D
    y = torch.empty_like(x)
    rstd = torch.empty(orig_shape[:-1], dtype=torch.float32, device=x.device)
    _l2norm_fwd_kernel[(triton.cdiv(n_rows, BR),)](
        x, y, rstd, eps, n_rows, D=D, BR=BR, BD=BD, num_warps=4, num_stages=1
    )
    return y, rstd


def l2norm_bwd_rsi(x: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]
    BD = triton.next_power_of_2(D)
    BR = 32
    n_rows = x.numel() // D
    dx = torch.empty_like(x)
    _l2norm_bwd_kernel[(triton.cdiv(n_rows, BR),)](
        x, dy, rstd, dx, n_rows, D=D, BR=BR, BD=BD, num_warps=4, num_stages=1
    )
    return dx


@triton.jit(do_not_specialize=["T"])
def _merge_16x16_to_64x64_inverse_kernel(
    Akkd_ptr,
    Akk_off_ptr,
    Akk_out_ptr,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill 64x64 block-lower-triangular inverse merger from 16x16 diagonal inverses and off-diagonal blocks."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    i_tc0 = i_t * BT
    if i_tc0 >= T:
        return
    i_tc1 = i_tc0 + BC
    i_tc2 = i_tc0 + 2 * BC
    i_tc3 = i_tc0 + 3 * BC

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    Akkd = Akkd_ptr + (bos * H + i_h) * BC
    Akk_off = Akk_off_ptr + (bos * H + i_h) * BT
    Akk_out = Akk_out_ptr + (bos * H + i_h) * BT

    b_Ai00 = tl.load(Akkd + (i_tc0 + o_i)[:, None] * (H * BC) + o_i[None, :], mask=m_tc0[:, None], other=0.0).to(tl.float32)
    b_Ai11 = tl.load(Akkd + (i_tc1 + o_i)[:, None] * (H * BC) + o_i[None, :], mask=m_tc1[:, None], other=0.0).to(tl.float32)
    b_Ai22 = tl.load(Akkd + (i_tc2 + o_i)[:, None] * (H * BC) + o_i[None, :], mask=m_tc2[:, None], other=0.0).to(tl.float32)
    b_Ai33 = tl.load(Akkd + (i_tc3 + o_i)[:, None] * (H * BC) + o_i[None, :], mask=m_tc3[:, None], other=0.0).to(tl.float32)

    b_Akk10 = tl.load(Akk_off + (i_tc1 + o_i)[:, None] * (H * BT) + o_i[None, :], mask=m_tc1[:, None], other=0.0).to(tl.float32)
    b_Akk20 = tl.load(Akk_off + (i_tc2 + o_i)[:, None] * (H * BT) + o_i[None, :], mask=m_tc2[:, None], other=0.0).to(tl.float32)
    b_Akk21 = tl.load(Akk_off + (i_tc2 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :], mask=m_tc2[:, None], other=0.0).to(tl.float32)
    b_Akk30 = tl.load(Akk_off + (i_tc3 + o_i)[:, None] * (H * BT) + o_i[None, :], mask=m_tc3[:, None], other=0.0).to(tl.float32)
    b_Akk31 = tl.load(Akk_off + (i_tc3 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :], mask=m_tc3[:, None], other=0.0).to(tl.float32)
    b_Akk32 = tl.load(Akk_off + (i_tc3 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :], mask=m_tc3[:, None], other=0.0).to(tl.float32)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10, input_precision=DOT_PREC), b_Ai00, input_precision=DOT_PREC)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_Akk21, input_precision=DOT_PREC), b_Ai11, input_precision=DOT_PREC)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_Akk32, input_precision=DOT_PREC), b_Ai22, input_precision=DOT_PREC)

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=DOT_PREC) + tl.dot(b_Akk21, b_Ai10, input_precision=DOT_PREC),
        input_precision=DOT_PREC,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=DOT_PREC) + tl.dot(b_Akk32, b_Ai21, input_precision=DOT_PREC),
        input_precision=DOT_PREC,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=DOT_PREC)
        + tl.dot(b_Akk31, b_Ai10, input_precision=DOT_PREC)
        + tl.dot(b_Akk32, b_Ai20, input_precision=DOT_PREC),
        input_precision=DOT_PREC,
    )

    # Zero upper-triangular blocks explicitly so Akk_out needs no prior zero-fill
    b_z = tl.zeros([BC, BC], dtype=Akk_out.dtype.element_ty)
    tl.store(Akk_out + (i_tc0 + o_i)[:, None] * (H * BT) + o_i[None, :], b_Ai00.to(Akk_out.dtype.element_ty), mask=m_tc0[:, None])
    tl.store(Akk_out + (i_tc0 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :], b_z, mask=m_tc0[:, None])
    tl.store(Akk_out + (i_tc0 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :], b_z, mask=m_tc0[:, None])
    tl.store(Akk_out + (i_tc0 + o_i)[:, None] * (H * BT) + (3 * BC + o_i)[None, :], b_z, mask=m_tc0[:, None])

    tl.store(Akk_out + (i_tc1 + o_i)[:, None] * (H * BT) + o_i[None, :], b_Ai10.to(Akk_out.dtype.element_ty), mask=m_tc1[:, None])
    tl.store(Akk_out + (i_tc1 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :], b_Ai11.to(Akk_out.dtype.element_ty), mask=m_tc1[:, None])
    tl.store(Akk_out + (i_tc1 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :], b_z, mask=m_tc1[:, None])
    tl.store(Akk_out + (i_tc1 + o_i)[:, None] * (H * BT) + (3 * BC + o_i)[None, :], b_z, mask=m_tc1[:, None])

    tl.store(Akk_out + (i_tc2 + o_i)[:, None] * (H * BT) + o_i[None, :], b_Ai20.to(Akk_out.dtype.element_ty), mask=m_tc2[:, None])
    tl.store(Akk_out + (i_tc2 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :], b_Ai21.to(Akk_out.dtype.element_ty), mask=m_tc2[:, None])
    tl.store(Akk_out + (i_tc2 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :], b_Ai22.to(Akk_out.dtype.element_ty), mask=m_tc2[:, None])
    tl.store(Akk_out + (i_tc2 + o_i)[:, None] * (H * BT) + (3 * BC + o_i)[None, :], b_z, mask=m_tc2[:, None])

    tl.store(Akk_out + (i_tc3 + o_i)[:, None] * (H * BT) + o_i[None, :], b_Ai30.to(Akk_out.dtype.element_ty), mask=m_tc3[:, None])
    tl.store(Akk_out + (i_tc3 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :], b_Ai31.to(Akk_out.dtype.element_ty), mask=m_tc3[:, None])
    tl.store(Akk_out + (i_tc3 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :], b_Ai32.to(Akk_out.dtype.element_ty), mask=m_tc3[:, None])
    tl.store(Akk_out + (i_tc3 + o_i)[:, None] * (H * BT) + (3 * BC + o_i)[None, :], b_Ai33.to(Akk_out.dtype.element_ty), mask=m_tc3[:, None])


# ======================================================================================
# 3. Zero-Spill Gated DeltaNet-2 (GDN-2) Forward & Backward Kernels
# ======================================================================================

@triton.jit(do_not_specialize=["T"])
def _gdn2_fwd_sub_chunk_diag_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    b_ptr,
    Aqk_ptr,
    Akkd_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Span-checked midpoint-anchored 16x16 diagonal sub-chunk builder + in-register 16x16 inverse."""
    i_t, i_i, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    i_last = min(T, i_ti + BC) - 1

    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    m_c = o_c < T

    q_base = q_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    g_base = g_ptr + (bos * H + i_h) * K
    b_base = b_ptr + (bos * H + i_h) * K
    Aqk_base = Aqk_ptr + (bos * H + i_h) * BT
    Akkd_base = Akkd_ptr + (bos * H + i_h) * BC

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_kc = m_c[:, None] & m_k[None, :]

        b_q = tl.load(q_base + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.0)
        b_k = tl.load(k_base + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.0)
        b_g = tl.load(g_base + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.0).to(tl.float32)
        b_b = tl.load(b_base + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.0)

        b_g_first = tl.load(g_base + i_ti * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_g_last = tl.load(g_base + i_last * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_span = tl.max(tl.abs(b_g_first - b_g_last), axis=0)

        if b_span <= 100.0:
            b_g_mid = 0.5 * (b_g_first + b_g_last)[None, :]
            b_gm = tl.where(m_kc, b_g - b_g_mid, 0.0)
            b_gq = tl.where(m_kc, tl.exp2(b_gm), 0.0)
            b_gk = tl.where(m_kc, tl.exp2(-b_gm), 0.0)
            b_bk = b_b.to(tl.float32) * b_k.to(tl.float32)
            if ALLOW_TF32:
                b_qg_d = (b_q.to(tl.float32) * b_gq).to(b_q.dtype)
                b_bkg_d = (b_bk * b_gq).to(b_k.dtype)
                b_kgt_d = tl.trans((b_k.to(tl.float32) * b_gk).to(b_k.dtype))
                b_Aqk = tl.dot(b_qg_d, b_kgt_d, b_Aqk)
                b_Akk = tl.dot(b_bkg_d, b_kgt_d, b_Akk)
            else:
                b_kgt = tl.trans(b_k.to(tl.float32) * b_gk)
                b_Aqk = tl.dot(b_q.to(tl.float32) * b_gq, b_kgt, b_Aqk, input_precision=DOT_PREC)
                b_Akk = tl.dot(b_bk * b_gq, b_kgt, b_Akk, input_precision=DOT_PREC)
        else:
            b_bk = b_b.to(tl.float32) * b_k.to(tl.float32)
            b_qf = b_q.to(tl.float32)
            for j in range(0, min(BC, T - i_ti)):
                b_kj = tl.load(k_base + (i_ti + j) * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_gj = tl.load(g_base + (i_ti + j) * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
                m_row = (o_i >= j) & m_c
                b_diff = tl.where(m_row[:, None] & m_k[None, :], b_g - b_gj[None, :], 0.0)
                b_kgj = tl.where(m_row[:, None] & m_k[None, :], b_kj[None, :] * tl.exp2(b_diff), 0.0)
                b_aqk_j = tl.sum(b_qf * b_kgj, axis=1)
                b_akk_j = tl.sum(b_bk * b_kgj, axis=1)
                b_Aqk = tl.where((o_i == j)[None, :] & (o_i[:, None] >= j), b_aqk_j[:, None], b_Aqk)
                b_Akk = tl.where((o_i == j)[None, :] & (o_i[:, None] > j), b_akk_j[:, None], b_Akk)

    m_Aqk = (o_i[:, None] >= o_i[None, :]) & m_c[:, None] & ((i_ti + o_i[None, :]) < T)
    m_Akk = (o_i[:, None] > o_i[None, :]) & m_c[:, None] & ((i_ti + o_i[None, :]) < T)
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    # Zero-fill entire row strip in Aqk on i_i == 0 or store sub-chunk diagonal
    tl.store(
        Aqk_base + o_c[:, None] * (H * BT) + (i_i * BC + o_i)[None, :],
        b_Aqk.to(Aqk_ptr.dtype.element_ty),
        mask=m_c[:, None],
    )
    # Zero upper-triangular sub-chunk blocks of Aqk
    b_z_aqk = tl.zeros([BC, BC], dtype=Aqk_ptr.dtype.element_ty)
    for j_up in range(i_i + 1, 4):
        tl.store(
            Aqk_base + o_c[:, None] * (H * BT) + (j_up * BC + o_i)[None, :],
            b_z_aqk,
            mask=m_c[:, None],
        )

    # Store raw strictly-lower b_Akk into Akkd, then invert (I + b_Akk)^{-1}
    tl.store(
        Akkd_base + o_c[:, None] * (H * BC) + o_i[None, :],
        b_Akk.to(Akkd_ptr.dtype.element_ty),
        mask=m_c[:, None],
    )
    tl.debug_barrier()

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akkd_base + (i_ti + i) * (H * BC) + o_i).to(tl.float32)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(
        Akkd_base + o_c[:, None] * (H * BC) + o_i[None, :],
        b_Ai.to(Akkd_ptr.dtype.element_ty),
        mask=m_c[:, None],
    )


@triton.jit(do_not_specialize=["T"])
def _gdn2_fwd_inter_offdiag_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    b_ptr,
    Aqk_ptr,
    Akk_off_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Computes off-diagonal 16x16 blocks for sub-chunk row i_r in {1, 2, 3} with 0 register spills."""
    i_t, i_r_idx, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_r = i_r_idx + 1  # sub-chunk row in {1, 2, 3}
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    i_tr = i_t * BT + i_r * BC
    if i_tr >= T:
        return

    o_i = tl.arange(0, BC)
    m_tr = (i_tr + o_i) < T

    q_base = q_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    g_base = g_ptr + (bos * H + i_h) * K
    b_base = b_ptr + (bos * H + i_h) * K
    Aqk_base = Aqk_ptr + (bos * H + i_h) * BT
    Akk_base = Akk_off_ptr + (bos * H + i_h) * BT

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    for i_c in range(0, 3):
        if i_c < i_r:
            i_tc = i_t * BT + i_c * BC
            m_tc = (i_tc + o_i) < T
            b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                m_rk = m_tr[:, None] & m_k[None, :]
                m_ck = m_tc[:, None] & m_k[None, :]

                b_qr = tl.load(q_base + (i_tr + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_rk, other=0.0)
                b_kr = tl.load(k_base + (i_tr + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_rk, other=0.0)
                b_gr = tl.load(g_base + (i_tr + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_rk, other=0.0).to(tl.float32)
                b_br = tl.load(b_base + (i_tr + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_rk, other=0.0)

                b_kc = tl.load(k_base + (i_tc + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_ck, other=0.0)
                b_gc = tl.load(g_base + (i_tc + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_ck, other=0.0).to(tl.float32)
                b_gn = tl.load(g_base + i_tr * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]

                b_gqn = tl.where(m_rk, tl.exp2(b_gr - b_gn), 0.0)
                b_gkn = tl.where(m_ck, tl.exp2(b_gn - b_gc), 0.0)

                if ALLOW_TF32:
                    b_qg = (b_qr.to(tl.float32) * b_gqn).to(b_qr.dtype)
                    b_bkg = (b_br.to(tl.float32) * b_kr.to(tl.float32) * b_gqn).to(b_kr.dtype)
                    b_kgt = tl.trans((b_kc.to(tl.float32) * b_gkn).to(b_kc.dtype))
                    b_Aqk = tl.dot(b_qg, b_kgt, b_Aqk)
                    b_Akk = tl.dot(b_bkg, b_kgt, b_Akk)
                else:
                    b_qg = b_qr.to(tl.float32) * b_gqn
                    b_bkg = b_br.to(tl.float32) * b_kr.to(tl.float32) * b_gqn
                    b_kgt = tl.trans(b_kc.to(tl.float32) * b_gkn)
                    b_Aqk = tl.dot(b_qg, b_kgt, b_Aqk, input_precision=DOT_PREC)
                    b_Akk = tl.dot(b_bkg, b_kgt, b_Akk, input_precision=DOT_PREC)

            tl.store(
                Aqk_base + (i_tr + o_i)[:, None] * (H * BT) + (i_c * BC + o_i)[None, :],
                (b_Aqk * scale).to(Aqk_ptr.dtype.element_ty),
                mask=m_tr[:, None],
            )
            tl.store(
                Akk_base + (i_tr + o_i)[:, None] * (H * BT) + (i_c * BC + o_i)[None, :],
                b_Akk.to(Akk_off_ptr.dtype.element_ty),
                mask=m_tr[:, None],
            )


@triton.jit(do_not_specialize=["T"])
def _gdn2_wy_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    w_gate_ptr,
    A_ptr,
    gk_ptr,
    w_out_ptr,
    u_out_ptr,
    qg_out_ptr,
    kg_out_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill WY forward producing w = A @ (b * 2^gk * k), u = A @ (w_gate * v), qg, and kg."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    p_A = A_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        base_v = (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(v_ptr + base_v, mask=m_v, other=0.0)
        b_wg = tl.load(w_gate_ptr + base_v, mask=m_v, other=0.0)
        if ALLOW_TF32:
            b_vb = (b_v.to(tl.float32) * b_wg.to(tl.float32)).to(b_A.dtype)
            b_u = tl.dot(b_A, b_vb)
        else:
            b_vb = b_v.to(tl.float32) * b_wg.to(tl.float32)
            b_u = tl.dot(b_A.to(tl.float32), b_vb, input_precision=DOT_PREC)
        tl.store(u_out_ptr + base_v, b_u.to(u_out_ptr.dtype.element_ty), mask=m_v)

    last_idx = min(i_t * BT + BT, T) - 1
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_kk = m_t[:, None] & m_k[None, :]
        base_k = (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(k_ptr + base_k, mask=m_kk, other=0.0)
        b_b = tl.load(b_ptr + base_k, mask=m_kk, other=0.0)
        b_gk = tl.load(gk_ptr + base_k, mask=m_kk, other=0.0).to(tl.float32)
        b_exp_gk = tl.where(m_kk, tl.exp2(b_gk), 0.0)

        b_q = tl.load(q_ptr + base_k, mask=m_kk, other=0.0)
        b_qg = b_q.to(tl.float32) * b_exp_gk
        tl.store(qg_out_ptr + base_k, b_qg.to(qg_out_ptr.dtype.element_ty), mask=m_kk)

        b_gn = tl.load(gk_ptr + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
        b_kg = b_k.to(tl.float32) * tl.where(m_kk, tl.exp2(b_gn - b_gk), 0.0)
        tl.store(kg_out_ptr + base_k, b_kg.to(kg_out_ptr.dtype.element_ty), mask=m_kk)

        b_kb = b_k.to(tl.float32) * b_b.to(tl.float32) * b_exp_gk
        if ALLOW_TF32:
            b_w = tl.dot(b_A, b_kb.to(b_A.dtype))
        else:
            b_w = tl.dot(b_A.to(tl.float32), b_kb, input_precision=DOT_PREC)
        tl.store(w_out_ptr + base_k, b_w.to(w_out_ptr.dtype.element_ty), mask=m_kk)


@triton.jit(do_not_specialize=["T"])
def _gdn2_fwd_h_kernel(
    kg_ptr,
    w_ptr,
    u_ptr,
    gk_ptr,
    h_ptr,
    v_new_ptr,
    h0_ptr,
    ht_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill inter-chunk state recurrence for GDN-2 (BV=32, K in {64, 128})."""
    i_v, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        h0_base = h0_ptr + i_bh * K * V
        b_h1 += tl.load(h0_base + o_k1[:, None] * V + o_v[None, :], mask=m_k1[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            b_h2 += tl.load(h0_base + o_k2[:, None] * V + o_v[None, :], mask=m_k2[:, None] & m_v[None, :], other=0.0).to(tl.float32)

    kg_base = kg_ptr + (bos * H + i_h) * K
    w_base = w_ptr + (bos * H + i_h) * K
    u_base = u_ptr + (bos * H + i_h) * V
    gk_base = gk_ptr + (bos * H + i_h) * K
    v_new_base = v_new_ptr + (bos * H + i_h) * V
    h_base = h_ptr + (i_b * NT * H + i_h) * K * V
    stride_h = H * K * V

    for i_t in range(0, NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]

        # Store chunk-start state h_c
        tl.store(
            h_base + i_t * stride_h + o_k1[:, None] * V + o_v[None, :],
            b_h1.to(h_ptr.dtype.element_ty),
            mask=m_k1[:, None] & m_v[None, :],
        )
        if K > 64:
            tl.store(
                h_base + i_t * stride_h + o_k2[:, None] * V + o_v[None, :],
                b_h2.to(h_ptr.dtype.element_ty),
                mask=m_k2[:, None] & m_v[None, :],
            )

        b_w1 = tl.load(w_base + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if ALLOW_TF32:
            b_proj = tl.dot(b_w1, b_h1.to(b_w1.dtype))
        else:
            b_proj = tl.dot(b_w1.to(tl.float32), b_h1, input_precision=DOT_PREC)

        if K > 64:
            b_w2 = tl.load(w_base + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if ALLOW_TF32:
                b_proj = tl.dot(b_w2, b_h2.to(b_w2.dtype), b_proj)
            else:
                b_proj = tl.dot(b_w2.to(tl.float32), b_h2, b_proj, input_precision=DOT_PREC)

        b_u = tl.load(u_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)
        b_v_new = b_u - b_proj
        tl.store(v_new_base + o_t[:, None] * (H * V) + o_v[None, :], b_v_new.to(v_new_ptr.dtype.element_ty), mask=m_tv)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_gn1 = tl.load(gk_base + last_idx * (H * K) + o_k1, mask=m_k1, other=0.0).to(tl.float32)
        b_h1 *= tl.exp2(b_gn1)[:, None]

        b_kg1 = tl.load(kg_base + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if ALLOW_TF32:
            b_h1 = tl.dot(tl.trans(b_kg1), b_v_new.to(b_kg1.dtype), b_h1)
        else:
            b_h1 = tl.dot(tl.trans(b_kg1.to(tl.float32)), b_v_new, b_h1, input_precision=DOT_PREC)

        if K > 64:
            b_gn2 = tl.load(gk_base + last_idx * (H * K) + o_k2, mask=m_k2, other=0.0).to(tl.float32)
            b_h2 *= tl.exp2(b_gn2)[:, None]
            b_kg2 = tl.load(kg_base + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if ALLOW_TF32:
                b_h2 = tl.dot(tl.trans(b_kg2), b_v_new.to(b_kg2.dtype), b_h2)
            else:
                b_h2 = tl.dot(tl.trans(b_kg2.to(tl.float32)), b_v_new, b_h2, input_precision=DOT_PREC)

    if STORE_FINAL_STATE:
        ht_base = ht_ptr + i_bh * K * V
        tl.store(ht_base + o_k1[:, None] * V + o_v[None, :], b_h1.to(ht_ptr.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(ht_base + o_k2[:, None] * V + o_v[None, :], b_h2.to(ht_ptr.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=["T"])
def _gdn2_fwd_o_kernel(
    qg_ptr,
    v_new_ptr,
    Aqk_ptr,
    h_ptr,
    o_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill output kernel o = scale * (qg @ h) + Aqk @ v_new using precomputed qg = q * 2^g."""
    i_v, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_tv = m_t[:, None] & (o_v[None, :] < V)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    qg_base = qg_ptr + (bos * H + i_h) * K
    h_base = h_ptr + (i_tg * H + i_h) * K * V

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        m_kv = (o_k[:, None] < K) & (o_v[None, :] < V)
        b_qg = tl.load(qg_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        b_h = tl.load(h_base + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.0)
        if ALLOW_TF32:
            b_o = tl.dot(b_qg, b_h.to(b_qg.dtype), b_o)
        else:
            b_o = tl.dot(b_qg.to(tl.float32), b_h.to(tl.float32), b_o, input_precision=DOT_PREC)

    b_o *= scale

    o_A = tl.arange(0, BT)
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    b_A = tl.load(Aqk_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], mask=m_A, other=0.0)
    b_v = tl.load(v_new_ptr + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)
    m_tril = o_A[:, None] >= o_A[None, :]
    b_A = tl.where(m_tril, b_A, 0.0)
    if ALLOW_TF32:
        b_o = tl.dot(b_A.to(b_v.dtype), b_v, b_o)
    else:
        b_o = tl.dot(b_A.to(tl.float32), b_v.to(tl.float32), b_o, input_precision=DOT_PREC)

    tl.store(o_ptr + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :], b_o.to(o_ptr.dtype.element_ty), mask=m_tv)


@triton.jit(do_not_specialize=["T"])
def _gdn2_bwd_dAv_kernel(
    v_new_ptr,
    do_ptr,
    Aqk_ptr,
    dAqk_ptr,
    dv_new_ptr,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill backward for dAqk = tril(scale * do @ v_new^T) and dv_new = Aqk^T @ do."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    p_AT = Aqk_ptr + (bos * H + i_h) * BT + o_A[:, None] + o_t[None, :] * (H * BT)
    b_AT = tl.load(p_AT, mask=m_AT, other=0.0)
    m_upper = (o_A[:, None] <= o_A[None, :]) & m_AT
    b_AT = tl.where(m_upper, b_AT, 0.0)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    v_base = v_new_ptr + (bos * H + i_h) * V
    do_base = do_ptr + (bos * H + i_h) * V
    dv_base = dv_new_ptr + (bos * H + i_h) * V

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        b_v = tl.load(v_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)
        b_do = tl.load(do_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)
        if ALLOW_TF32:
            b_dA = tl.dot(b_do, tl.trans(b_v), b_dA)
            b_dv = tl.dot(b_AT.to(b_do.dtype), b_do)
        else:
            b_dA = tl.dot(b_do.to(tl.float32), tl.trans(b_v.to(tl.float32)), b_dA, input_precision=DOT_PREC)
            b_dv = tl.dot(b_AT.to(tl.float32), b_do.to(tl.float32), input_precision=DOT_PREC)
        tl.store(dv_base + o_t[:, None] * (H * V) + o_v[None, :], b_dv.to(dv_new_ptr.dtype.element_ty), mask=m_tv)

    m_tril = (o_A[:, None] >= o_A[None, :]) & m_t[:, None] & m_t[None, :]
    b_dA = tl.where(m_tril, b_dA * scale, 0.0)
    tl.store(
        dAqk_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :],
        b_dA.to(dAqk_ptr.dtype.element_ty),
        mask=m_t[:, None] & (o_A[None, :] < BT),
    )


@triton.jit(do_not_specialize=["T"])
def _gdn2_bwd_dhu_kernel(
    qg_ptr,
    kg_ptr,
    w_ptr,
    gk_ptr,
    do_ptr,
    dv_ptr,
    dh_ptr,
    dv_out_ptr,
    dh0_ptr,
    dht_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRAD: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill reverse chunk state adjoint for GDN-2 (BV=32, K in {64, 128})."""
    i_v, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_FINAL_STATE_GRAD:
        dht_base = dht_ptr + i_bh * K * V
        b_dh1 += tl.load(dht_base + o_k1[:, None] * V + o_v[None, :], mask=m_k1[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            b_dh2 += tl.load(dht_base + o_k2[:, None] * V + o_v[None, :], mask=m_k2[:, None] & m_v[None, :], other=0.0).to(tl.float32)

    qg_base = qg_ptr + (bos * H + i_h) * K
    kg_base = kg_ptr + (bos * H + i_h) * K
    w_base = w_ptr + (bos * H + i_h) * K
    gk_base = gk_ptr + (bos * H + i_h) * K
    do_base = do_ptr + (bos * H + i_h) * V
    dv_base = dv_ptr + (bos * H + i_h) * V
    dv_out_base = dv_out_ptr + (bos * H + i_h) * V
    dh_base = dh_ptr + (i_b * NT * H + i_h) * K * V
    stride_h = H * K * V

    for i_t in range(NT - 1, -1, -1):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]

        tl.store(
            dh_base + i_t * stride_h + o_k1[:, None] * V + o_v[None, :],
            b_dh1.to(dh_ptr.dtype.element_ty),
            mask=m_k1[:, None] & m_v[None, :],
        )
        if K > 64:
            tl.store(
                dh_base + i_t * stride_h + o_k2[:, None] * V + o_v[None, :],
                b_dh2.to(dh_ptr.dtype.element_ty),
                mask=m_k2[:, None] & m_v[None, :],
            )

        b_kg1 = tl.load(kg_base + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_dv = tl.load(dv_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)
        if ALLOW_TF32:
            b_dv = tl.dot(b_kg1, b_dh1.to(b_kg1.dtype), b_dv)
        else:
            b_dv = tl.dot(b_kg1.to(tl.float32), b_dh1, b_dv, input_precision=DOT_PREC)

        if K > 64:
            b_kg2 = tl.load(kg_base + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if ALLOW_TF32:
                b_dv = tl.dot(b_kg2, b_dh2.to(b_kg2.dtype), b_dv)
            else:
                b_dv = tl.dot(b_kg2.to(tl.float32), b_dh2, b_dv, input_precision=DOT_PREC)

        tl.store(dv_out_base + o_t[:, None] * (H * V) + o_v[None, :], b_dv.to(dv_out_ptr.dtype.element_ty), mask=m_tv)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_gn1 = tl.load(gk_base + last_idx * (H * K) + o_k1, mask=m_k1, other=0.0).to(tl.float32)
        b_dh1 *= tl.exp2(b_gn1)[:, None]

        b_do = tl.load(do_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)
        b_qg1 = tl.load(qg_base + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_w1 = tl.load(w_base + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if ALLOW_TF32:
            b_dh1 += tl.dot(tl.trans(b_qg1), b_do.to(b_qg1.dtype)) * scale - tl.dot(tl.trans(b_w1), b_dv.to(b_w1.dtype))
        else:
            b_dh1 += (
                tl.dot(tl.trans(b_qg1.to(tl.float32)), b_do.to(tl.float32), input_precision=DOT_PREC) * scale
                - tl.dot(tl.trans(b_w1.to(tl.float32)), b_dv, input_precision=DOT_PREC)
            )

        if K > 64:
            b_gn2 = tl.load(gk_base + last_idx * (H * K) + o_k2, mask=m_k2, other=0.0).to(tl.float32)
            b_dh2 *= tl.exp2(b_gn2)[:, None]
            b_qg2 = tl.load(qg_base + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_w2 = tl.load(w_base + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if ALLOW_TF32:
                b_dh2 += tl.dot(tl.trans(b_qg2), b_do.to(b_qg2.dtype)) * scale - tl.dot(tl.trans(b_w2), b_dv.to(b_w2.dtype))
            else:
                b_dh2 += (
                    tl.dot(tl.trans(b_qg2.to(tl.float32)), b_do.to(tl.float32), input_precision=DOT_PREC) * scale
                    - tl.dot(tl.trans(b_w2.to(tl.float32)), b_dv, input_precision=DOT_PREC)
                )

    if USE_INITIAL_STATE:
        dh0_base = dh0_ptr + i_bh * K * V
        tl.store(dh0_base + o_k1[:, None] * V + o_v[None, :], b_dh1, mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(dh0_base + o_k2[:, None] * V + o_v[None, :], b_dh2, mask=m_k2[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=["T"])
def _gdn2_bwd_state_dqkw_pass1_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    v_new_ptr,
    do_ptr,
    dv_ptr,
    h_ptr,
    dh_ptr,
    dq_ptr,
    dk_state_ptr,
    dw_flow_ptr,
    dg_state_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Pass 1 of State-Split GDN-2 WY Backward: accumulates dq, dk_state, dw_flow, dg_state over V tiles (0 spills)."""
    i_k, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    last_idx = min(T, i_t * BT + BT) - 1
    m_last = o_t == last_idx

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw_flow = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    v_new_base = v_new_ptr + (bos * H + i_h) * V
    do_base = do_ptr + (bos * H + i_h) * V
    dv_base = dv_ptr + (bos * H + i_h) * V
    h_base = h_ptr + (i_tg * H + i_h) * K * V
    dh_base = dh_ptr + (i_tg * H + i_h) * K * V

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_vv = m_t[:, None] & (o_v[None, :] < V)
        m_hv = (o_v[:, None] < V) & m_k[None, :]

        b_v_new = tl.load(v_new_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_vv, other=0.0)
        b_do = tl.load(do_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_vv, other=0.0)
        b_dv = tl.load(dv_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_vv, other=0.0)
        b_h_t = tl.load(h_base + o_v[:, None] + o_k[None, :] * V, mask=m_hv, other=0.0)
        b_dh_t = tl.load(dh_base + o_v[:, None] + o_k[None, :] * V, mask=m_hv, other=0.0)

        b_dgk += tl.sum(b_h_t.to(tl.float32) * b_dh_t.to(tl.float32), axis=0)
        if ALLOW_TF32:
            b_dq = tl.dot(b_do, b_h_t.to(b_do.dtype), b_dq)
            b_dk = tl.dot(b_v_new, b_dh_t.to(b_v_new.dtype), b_dk)
            b_dw_flow = tl.dot(b_dv.to(b_v_new.dtype), b_h_t.to(b_v_new.dtype), b_dw_flow)
        else:
            b_dq = tl.dot(b_do.to(tl.float32), b_h_t.to(tl.float32), b_dq, input_precision=DOT_PREC)
            b_dk = tl.dot(b_v_new.to(tl.float32), b_dh_t.to(tl.float32), b_dk, input_precision=DOT_PREC)
            b_dw_flow = tl.dot(b_dv.to(tl.float32), b_h_t.to(tl.float32), b_dw_flow, input_precision=DOT_PREC)

    base_k = (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(q_ptr + base_k, mask=m_tk, other=0.0).to(tl.float32)
    b_k = tl.load(k_ptr + base_k, mask=m_tk, other=0.0).to(tl.float32)
    b_g = tl.load(g_ptr + base_k, mask=m_tk, other=0.0).to(tl.float32)
    b_gn = tl.load(g_ptr + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]

    b_gk_exp = tl.where(m_tk, tl.exp2(b_g), 0.0)
    b_dgk_row = b_dgk[None, :] * tl.exp2(b_gn)
    b_dq = b_dq * b_gk_exp * scale
    b_dk = b_dk * tl.where(m_tk, tl.exp2(b_gn - b_g), 0.0)
    b_kdk = b_k * b_dk
    b_dgk_row += tl.sum(b_kdk, axis=0)[None, :]
    b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk_row

    tl.store(dq_ptr + base_k, b_dq, mask=m_tk)
    tl.store(dk_state_ptr + base_k, b_dk, mask=m_tk)
    tl.store(dw_flow_ptr + base_k, (-b_dw_flow).to(dw_flow_ptr.dtype.element_ty), mask=m_tk)
    tl.store(dg_state_ptr + base_k, b_dg, mask=m_tk)


@triton.jit(do_not_specialize=["T"])
def _bwd_dA_triple_product_kernel(
    A_ptr,
    dA_in_ptr,
    dA_out_ptr,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill triple product dAkk = -A^T @ tril(dA_in, -1) @ A^T shared by GDN-2 and Oja-2."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    m_dA = m_t[:, None] & (o_A[None, :] < BT)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    p_A = A_ptr + (bos * H + i_h) * BT + o_A[:, None] + o_t[None, :] * (H * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.load(dA_in_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], mask=m_dA, other=0.0).to(tl.float32)

    m_strict = (o_t[:, None] > o_t[None, :]) & m_t[:, None] & m_t[None, :]
    b_dA = tl.where(m_strict, b_dA, 0.0)
    if ALLOW_TF32:
        b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
        b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    else:
        b_dA = tl.dot(b_dA, b_A.to(tl.float32), input_precision=DOT_PREC)
        b_dA = tl.dot(b_A.to(tl.float32), b_dA, input_precision=DOT_PREC)
    b_dA = tl.where(m_strict, -b_dA, 0.0)
    tl.store(dA_out_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], b_dA.to(dA_out_ptr.dtype.element_ty), mask=m_dA)


@triton.jit(do_not_specialize=["T"])
def _gdn2_bwd_wy_v_pass2a_kernel(
    v_ptr,
    w_gate_ptr,
    A_ptr,
    dv_ptr,
    dv2_ptr,
    dw_ptr,
    dA_v_ptr,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Pass 2a of State-Split GDN-2 WY Backward: applies A^T to dv and accumulates dA_v over V tiles (0 spills)."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    m_dA = m_t[:, None] & (o_A[None, :] < BT)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    p_A = A_ptr + (bos * H + i_h) * BT + o_A[:, None] + o_t[None, :] * (H * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_vv = m_t[:, None] & (o_v[None, :] < V)
        base_v = (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(v_ptr + base_v, mask=m_vv, other=0.0)
        b_wg = tl.load(w_gate_ptr + base_v, mask=m_vv, other=0.0)
        b_dv = tl.load(dv_ptr + base_v, mask=m_vv, other=0.0)
        b_vwg = b_v.to(tl.float32) * b_wg.to(tl.float32)
        if ALLOW_TF32:
            b_dA = tl.dot(b_dv, tl.trans(b_vwg.to(b_dv.dtype)), b_dA)
            b_dvb = tl.dot(b_A, b_dv.to(b_A.dtype))
        else:
            b_dA = tl.dot(b_dv.to(tl.float32), tl.trans(b_vwg), b_dA, input_precision=DOT_PREC)
            b_dvb = tl.dot(b_A.to(tl.float32), b_dv.to(tl.float32), input_precision=DOT_PREC)
        tl.store(dv2_ptr + base_v, (b_dvb * b_wg.to(tl.float32)).to(dv2_ptr.dtype.element_ty), mask=m_vv)
        tl.store(dw_ptr + base_v, b_dvb * b_v.to(tl.float32), mask=m_vv)

    tl.store(dA_v_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], b_dA, mask=m_dA)


@triton.jit(do_not_specialize=["T"])
def _gdn2_bwd_wy_k_pass2b_kernel(
    k_ptr,
    g_ptr,
    b_ptr,
    A_ptr,
    dk_state_ptr,
    dw_flow_ptr,
    dg_state_ptr,
    dA_v_ptr,
    dk_ptr,
    dg_ptr,
    db_ptr,
    dA_raw_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Pass 2b of State-Split GDN-2 WY Backward: applies A^T to dw_flow and accumulates dA_raw over K tiles (0 spills)."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    m_dA = m_t[:, None] & (o_A[None, :] < BT)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    p_A = A_ptr + (bos * H + i_h) * BT + o_A[:, None] + o_t[None, :] * (H * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.load(dA_v_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], mask=m_dA, other=0.0).to(tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_kk = m_t[:, None] & (o_k[None, :] < K)
        base_k = (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]

        b_k = tl.load(k_ptr + base_k, mask=m_kk, other=0.0).to(tl.float32)
        b_g = tl.load(g_ptr + base_k, mask=m_kk, other=0.0).to(tl.float32)
        b_b = tl.load(b_ptr + base_k, mask=m_kk, other=0.0).to(tl.float32)
        b_dw_flow = tl.load(dw_flow_ptr + base_k, mask=m_kk, other=0.0)

        b_gk_exp = tl.where(m_kk, tl.exp2(b_g), 0.0)
        b_gb = b_gk_exp * b_b
        b_kg = b_k * b_gk_exp
        b_kgb = b_kg * b_b

        if ALLOW_TF32:
            b_dA = tl.dot(b_dw_flow.to(b_A.dtype), tl.trans(b_kgb.to(b_A.dtype)), b_dA)
            b_dkgb = tl.dot(b_A, b_dw_flow.to(b_A.dtype))
        else:
            b_dA = tl.dot(b_dw_flow.to(tl.float32), tl.trans(b_kgb), b_dA, input_precision=DOT_PREC)
            b_dkgb = tl.dot(b_A.to(tl.float32), b_dw_flow.to(tl.float32), input_precision=DOT_PREC)

        tl.store(db_ptr + base_k, b_dkgb * b_kg, mask=m_kk)

        b_dk = tl.load(dk_state_ptr + base_k, mask=m_kk, other=0.0).to(tl.float32) + b_dkgb * b_gb
        b_dg = tl.load(dg_state_ptr + base_k, mask=m_kk, other=0.0).to(tl.float32) + b_kg * b_dkgb * b_b

        tl.store(dk_ptr + base_k, b_dk, mask=m_kk)
        tl.store(dg_ptr + base_k, b_dg, mask=m_kk)

    tl.store(dA_raw_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], b_dA, mask=m_dA)


@triton.jit(do_not_specialize=["T"])
def _gdn2_bwd_intra_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    b_ptr,
    dAqk_ptr,
    dAkk_ptr,
    dq_ptr,
    dq_out_ptr,
    dk_ptr,
    dk_out_ptr,
    dg_ptr,
    dg_out_ptr,
    db_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill intra-chunk backward for GDN-2 using span-checked midpoint-anchored 16x16 Tensor Core dots."""
    i_kc, i_t = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    i_last = min(T, i_ti + BC) - 1

    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    o_k = i_k * BK + tl.arange(0, BK)
    m_c = o_c < T
    m_k = o_k < K
    m_kc = m_c[:, None] & m_k[None, :]

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    q_base = q_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    g_base = g_ptr + (bos * H + i_h) * K
    b_base = b_ptr + (bos * H + i_h) * K
    dAqk_base = dAqk_ptr + (bos * H + i_h) * BT
    dAkk_base = dAkk_ptr + (bos * H + i_h) * BT

    b_q = tl.load(q_base + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.0)
    b_k = tl.load(k_base + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.0)
    b_g = tl.load(g_base + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.0).to(tl.float32)
    b_b = tl.load(b_base + o_c[:, None] * (H * K) + o_k[None, :], mask=m_kc, other=0.0)

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

    # 1. Off-diagonal lower blocks (i_j < i_i)
    if i_i > 0:
        b_gn = tl.load(g_base + i_ti * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dj = m_c[:, None] & ((i_j * BC + o_i)[None, :] < BT)

            b_kj = tl.load(k_base + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.0)
            b_gk = tl.load(g_base + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.0).to(tl.float32)
            b_kg = b_kj.to(tl.float32) * tl.where(m_jk, tl.exp2(b_gn - b_gk), 0.0)
            b_dAqk = tl.load(dAqk_base + o_c[:, None] * (H * BT) + (i_j * BC + o_i)[None, :], mask=m_dj, other=0.0)
            b_dAkk = tl.load(dAkk_base + o_c[:, None] * (H * BT) + (i_j * BC + o_i)[None, :], mask=m_dj, other=0.0)
            if ALLOW_TF32:
                b_kg_d = b_kg.to(b_kj.dtype)
                b_dq2 = tl.dot(b_dAqk.to(b_kj.dtype), b_kg_d, b_dq2)
                b_dk2 = tl.dot(b_dAkk.to(b_kj.dtype), b_kg_d, b_dk2)
            else:
                b_dq2 = tl.dot(b_dAqk.to(tl.float32), b_kg, b_dq2, input_precision=DOT_PREC)
                b_dk2 = tl.dot(b_dAkk.to(tl.float32), b_kg, b_dk2, input_precision=DOT_PREC)
        b_gqn = tl.where(m_kc, tl.exp2(b_g - b_gn), 0.0)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    # 2. Diagonal sub-chunk (i_j == i_i) for dq2, dk2, and dkt
    b_g_first = tl.load(g_base + i_ti * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_g_last = tl.load(g_base + i_last * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_span = tl.max(tl.abs(b_g_first - b_g_last), axis=0)

    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
    if b_span <= 100.0:
        b_g_mid = 0.5 * (b_g_first + b_g_last)[None, :]
        b_gm = tl.where(m_kc, b_g - b_g_mid, 0.0)
        b_exp_pos = tl.where(m_kc, tl.exp2(b_gm), 0.0)
        b_exp_neg = tl.where(m_kc, tl.exp2(-b_gm), 0.0)

        m_dd = m_c[:, None] & ((i_i * BC + o_i)[None, :] < BT)
        b_dAqk_d = tl.load(dAqk_base + o_c[:, None] * (H * BT) + (i_i * BC + o_i)[None, :], mask=m_dd, other=0.0).to(tl.float32)
        b_dAkk_d = tl.load(dAkk_base + o_c[:, None] * (H * BT) + (i_i * BC + o_i)[None, :], mask=m_dd, other=0.0).to(tl.float32)
        m_tril = (o_i[:, None] >= o_i[None, :]) & m_c[:, None] & m_c[None, :]
        m_strict = (o_i[:, None] > o_i[None, :]) & m_c[:, None] & m_c[None, :]
        b_dAqk_d = tl.where(m_tril, b_dAqk_d, 0.0)
        b_dAkk_d = tl.where(m_strict, b_dAkk_d, 0.0)

        b_k_neg = b_k.to(tl.float32) * b_exp_neg
        b_dq2 += tl.dot(b_dAqk_d, b_k_neg, input_precision=DOT_PREC) * b_exp_pos
        b_dk2 += tl.dot(b_dAkk_d, b_k_neg, input_precision=DOT_PREC) * b_exp_pos

        b_q_pos = b_q.to(tl.float32) * b_exp_pos
        b_kb_pos = b_k.to(tl.float32) * b_b.to(tl.float32) * b_exp_pos
        b_dkt += (
            tl.dot(tl.trans(b_dAqk_d), b_q_pos, input_precision=DOT_PREC)
            + tl.dot(tl.trans(b_dAkk_d), b_kb_pos, input_precision=DOT_PREC)
        ) * b_exp_neg
    else:
        o_dA_row = o_c * (H * BT) + i_i * BC
        o_dA_col = i_ti * (H * BT) + i_i * BC + o_i
        for j in range(0, min(BC, T - i_ti)):
            b_dAqk_r = tl.load(dAqk_base + o_dA_row + j, mask=m_c, other=0.0).to(tl.float32)
            b_dAkk_r = tl.load(dAkk_base + o_dA_row + j, mask=m_c, other=0.0).to(tl.float32)
            b_qj = tl.load(q_base + (i_ti + j) * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_kj = tl.load(k_base + (i_ti + j) * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_bj = tl.load(b_base + (i_ti + j) * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_gkj = tl.load(g_base + (i_ti + j) * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)

            m_ge = (o_i >= j)[:, None] & m_kc
            b_gqk = tl.where(m_ge, tl.exp2(b_g - b_gkj[None, :]), 0.0)
            b_dq2 += tl.where(m_ge, b_dAqk_r[:, None] * b_kj[None, :] * b_gqk, 0.0)
            b_dk2 += tl.where((o_i > j)[:, None] & m_kc, b_dAkk_r[:, None] * b_kj[None, :] * b_gqk, 0.0)

            b_dAqk_c = tl.load(dAqk_base + o_dA_col + j * (H * BT), mask=m_c, other=0.0).to(tl.float32)
            b_dAkk_c = tl.load(dAkk_base + o_dA_col + j * (H * BT), mask=m_c, other=0.0).to(tl.float32)
            m_le = (o_i <= j)[:, None] & m_kc
            b_gkq = tl.where(m_le, tl.exp2(b_gkj[None, :] - b_g), 0.0)
            b_dkt += tl.where(m_le, b_dAqk_c[:, None] * b_qj[None, :] * b_gkq, 0.0)
            b_dkt += tl.where((o_i < j)[:, None] & m_kc, b_dAkk_c[:, None] * (b_kj * b_bj)[None, :] * b_gkq, 0.0)

    # Fold channel-wise erase gate b into db and dk2
    b_db_tile = b_dk2 * b_k.to(tl.float32)
    b_dk2 = b_dk2 * b_b.to(tl.float32)

    base_out = (bos * H + i_h) * K + o_c[:, None] * (H * K) + o_k[None, :]
    b_dg2 = b_q.to(tl.float32) * b_dq2
    b_dq_tot = b_dq2 + tl.load(dq_ptr + base_out, mask=m_kc, other=0.0).to(tl.float32)
    tl.store(dq_out_ptr + base_out, b_dq_tot.to(dq_out_ptr.dtype.element_ty), mask=m_kc)

    b_db_tot = b_db_tile + tl.load(db_ptr + base_out, mask=m_kc, other=0.0).to(tl.float32)
    tl.store(db_ptr + base_out, b_db_tot.to(db_ptr.dtype.element_ty), mask=m_kc)

    # 3. Off-diagonal upper blocks (i_j > i_i) for dkt
    NC_valid = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC_valid - 1:
        b_gn_u = tl.load(g_base + i_last * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
        b_dkt_off = tl.zeros([BC, BK], dtype=tl.float32)
        for i_j in range(i_i + 1, NC_valid):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dj = ((i_i * BC + o_i)[:, None] < BT) & m_j[None, :]

            b_qj = tl.load(q_base + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.0)
            b_kj = tl.load(k_base + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.0)
            b_bj = tl.load(b_base + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.0)
            b_gk = tl.load(g_base + o_j[:, None] * (H * K) + o_k[None, :], mask=m_jk, other=0.0).to(tl.float32)
            b_dAqk_u = tl.load(dAqk_base + (i_i * BC + o_i)[:, None] + o_j[None, :] * (H * BT), mask=m_dj, other=0.0)
            b_dAkk_u = tl.load(dAkk_base + (i_i * BC + o_i)[:, None] + o_j[None, :] * (H * BT), mask=m_dj, other=0.0)

            b_gkn = tl.where(m_jk, tl.exp2(b_gk - b_gn_u), 0.0)
            b_qg = b_qj.to(tl.float32) * b_gkn
            b_kbg = b_kj.to(tl.float32) * b_bj.to(tl.float32) * b_gkn
            if ALLOW_TF32:
                b_dkt_off = tl.dot(b_dAqk_u.to(b_qj.dtype), b_qg.to(b_qj.dtype), b_dkt_off)
                b_dkt_off = tl.dot(b_dAkk_u.to(b_kj.dtype), b_kbg.to(b_kj.dtype), b_dkt_off)
            else:
                b_dkt_off = tl.dot(b_dAqk_u.to(tl.float32), b_qg, b_dkt_off, input_precision=DOT_PREC)
                b_dkt_off = tl.dot(b_dAkk_u.to(tl.float32), b_kbg, b_dkt_off, input_precision=DOT_PREC)
        b_dkt += b_dkt_off * tl.where(m_kc, tl.exp2(b_gn_u - b_g), 0.0)

    b_dg2 += (b_dk2 - b_dkt) * b_k.to(tl.float32) + tl.load(dg_ptr + base_out, mask=m_kc, other=0.0).to(tl.float32)
    b_dk_tot = b_dk2 + b_dkt + tl.load(dk_ptr + base_out, mask=m_kc, other=0.0).to(tl.float32)

    tl.store(dk_out_ptr + base_out, b_dk_tot.to(dk_out_ptr.dtype.element_ty), mask=m_kc)
    tl.store(dg_out_ptr + base_out, b_dg2, mask=m_kc)


class _ChunkGDN2RSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
    ):
        B, T, H, K = q.shape
        V = v.shape[-1]
        BT = 64
        BC = 16
        BK = 64
        BV = 64
        NT = triton.cdiv(T, BT)
        allow_tf32 = q.dtype != torch.float32

        q_raw, k_raw = q, k
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd_rsi(q)
            k, k_rstd = l2norm_fwd_rsi(k)

        g_cumsum = chunk_local_cumsum_rsi(g, scale=RCP_LN2, chunk_size=BT)

        Aqk = torch.empty(B, T, H, BT, device=q.device, dtype=q.dtype)
        Akkd = torch.empty(B, T, H, BC, device=q.device, dtype=torch.float32)
        Akk_off = torch.empty(B, T, H, BT, device=q.device, dtype=k.dtype)
        Akk = torch.empty(B, T, H, BT, device=q.device, dtype=k.dtype)

        _gdn2_fwd_sub_chunk_diag_kernel[(NT, 4, B * H)](
            q, k, g_cumsum, b, Aqk, Akkd, scale, T,
            H=H, K=K, BT=BT, BC=BC, BK=BK, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )
        _gdn2_fwd_inter_offdiag_kernel[(NT, 3, B * H)](
            q, k, g_cumsum, b, Aqk, Akk_off, scale, T,
            H=H, K=K, BT=BT, BC=BC, BK=BK, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )
        _merge_16x16_to_64x64_inverse_kernel[(NT, B * H)](
            Akkd, Akk_off, Akk, T,
            H=H, BT=BT, BC=BC, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        w_wy = torch.empty_like(k)
        u_wy = torch.empty_like(v)
        qg = torch.empty_like(q)
        kg = torch.empty_like(k)
        _gdn2_wy_fwd_kernel[(NT, B * H)](
            q, k, v, b, w, Akk, g_cumsum, w_wy, u_wy, qg, kg, T,
            H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        BV_H = 32
        NV_H = triton.cdiv(V, BV_H)
        h = torch.empty(B, NT, H, K, V, device=q.device, dtype=k.dtype)
        v_new = torch.empty_like(v)
        final_state = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if output_final_state else None
        _gdn2_fwd_h_kernel[(NV_H, B * H)](
            kg, w_wy, u_wy, g_cumsum, h, v_new, initial_state, final_state, T,
            H=H, K=K, V=V, BT=BT, BV=BV_H,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        NV_O = triton.cdiv(V, BV)
        o = torch.empty_like(v)
        _gdn2_fwd_o_kernel[(NV_O, NT, B * H)](
            qg, v_new, Aqk, h, o, scale, T,
            H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        ctx.save_for_backward(
            q_raw, k_raw, q, k, v, g_cumsum, b, w, Aqk, Akk, w_wy, qg, kg, v_new, h, initial_state, q_rstd, k_rstd
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o, final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        (
            q_raw, k_raw, q, k, v, g_cumsum, b, w, Aqk, Akk, w_wy, qg, kg, v_new, h, initial_state, q_rstd, k_rstd
        ) = ctx.saved_tensors
        scale = ctx.scale
        B, T, H, K = q.shape
        V = v.shape[-1]
        BT = 64
        BC = 16
        BK = 64
        BV = 64
        NT = triton.cdiv(T, BT)
        allow_tf32 = q.dtype != torch.float32
        do = do.contiguous()

        dAqk = torch.empty(B, T, H, BT, device=q.device, dtype=q.dtype)
        dv_new = torch.empty_like(v)
        _gdn2_bwd_dAv_kernel[(NT, B * H)](
            v_new, do, Aqk, dAqk, dv_new, scale, T,
            H=H, V=V, BT=BT, BV=32, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        BV_H = 16
        NV_H = triton.cdiv(V, BV_H)
        dh = torch.empty(B, NT, H, K, V, device=q.device, dtype=q.dtype)
        dv_total = torch.empty_like(v)
        dh0 = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if initial_state is not None else None
        _gdn2_bwd_dhu_kernel[(NV_H, B * H)](
            qg, kg, w_wy, g_cumsum, do, dv_new, dh, dv_total, dh0, dht, scale, T,
            H=H, K=K, V=V, BT=BT, BV=BV_H,
            USE_INITIAL_STATE=initial_state is not None,
            USE_FINAL_STATE_GRAD=dht is not None,
            ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        BK_S = 32
        NK_S = triton.cdiv(K, BK_S)
        dq = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
        dk_state = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
        dw_flow = torch.empty(B, T, H, K, device=q.device, dtype=q.dtype)
        dg_state = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
        _gdn2_bwd_state_dqkw_pass1_kernel[(NK_S, NT, B * H)](
            q, k, g_cumsum, v_new, do, dv_total, h, dh, dq, dk_state, dw_flow, dg_state, scale, T,
            H=H, K=K, V=V, BT=BT, BK=BK_S, BV=32, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        dk = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        dg = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
        db = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
        dw = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
        dA_v = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
        dA_raw = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
        dAkk = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
        _gdn2_bwd_wy_v_pass2a_kernel[(NT, B * H)](
            v, w, Akk, dv_total, dv, dw, dA_v, T,
            H=H, V=V, BT=BT, BV=16, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )
        _gdn2_bwd_wy_k_pass2b_kernel[(NT, B * H)](
            k, g_cumsum, b, Akk, dk_state, dw_flow, dg_state, dA_v,
            dk, dg, db, dA_raw, T,
            H=H, K=K, BT=BT, BK=16, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )
        _bwd_dA_triple_product_kernel[(NT, B * H)](
            Akk, dA_raw, dAkk, T,
            H=H, BT=BT, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        NK = triton.cdiv(K, BK)
        NC = BT // BC
        dq_out = torch.empty_like(q)
        dk_out = torch.empty_like(k)
        dg_pre = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
        _gdn2_bwd_intra_kernel[(NK * NC, NT, B * H)](
            q, k, g_cumsum, b, dAqk, dAkk, dq, dq_out, dk, dk_out, dg, dg_pre, db, T,
            H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        dg_out = chunk_local_cumsum_rev_rsi(dg_pre, scale=1.0, chunk_size=BT)

        if ctx.use_qk_l2norm_in_kernel:
            dq_out = l2norm_bwd_rsi(q_raw, q_rstd, dq_out)
            dk_out = l2norm_bwd_rsi(k_raw, k_rstd, dk_out)

        return (
            dq_out,
            dk_out,
            dv,
            dg_out,
            db.to(b.dtype),
            dw.to(w.dtype),
            None,
            dh0.to(initial_state.dtype) if dh0 is not None else None,
            None,
            None,
        )


def chunk_gdn2_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _ChunkGDN2RSIFunction.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        b.contiguous(),
        w.contiguous(),
        scale,
        initial_state.contiguous() if initial_state is not None else None,
        output_final_state,
        use_qk_l2norm_in_kernel,
    )


# ======================================================================================
# 4. Zero-Spill Gated Oja Rule 2 (Oja-2) Forward & Backward Kernels (M in {64, 128, 256})
# ======================================================================================

@triton.jit(do_not_specialize=["T"])
def _oja2_kkt_sub_chunk_diag_kernel(
    v_ptr,
    gv_ptr,
    b_ptr,
    Avvd_ptr,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Midpoint-anchored 16x16 diagonal sub-chunk builder + in-register 16x16 inverse for Oja-2 slot Gram matrix."""
    i_t, i_i, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    i_last = min(T, i_ti + BC) - 1

    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    m_c = o_c < T

    v_base = v_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    b_base = b_ptr + (bos * H + i_h) * M
    Avvd_base = Avvd_ptr + (bos * H + i_h) * BC

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_Avv = tl.zeros([BC, BC], dtype=tl.float32)
    for i_m in range(tl.cdiv(M, BM)):
        o_m = i_m * BM + tl.arange(0, BM)
        m_m = o_m < M
        m_mc = m_c[:, None] & m_m[None, :]

        b_v = tl.load(v_base + o_c[:, None] * (H * M) + o_m[None, :], mask=m_mc, other=0.0).to(tl.float32)
        b_gv = tl.load(gv_base + o_c[:, None] * (H * M) + o_m[None, :], mask=m_mc, other=0.0).to(tl.float32)
        b_b = tl.load(b_base + o_c[:, None] * (H * M) + o_m[None, :], mask=m_mc, other=0.0).to(tl.float32)

        b_g_first = tl.load(gv_base + i_ti * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
        b_g_last = tl.load(gv_base + i_last * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
        b_span = tl.max(tl.abs(b_g_first - b_g_last), axis=0)

        if b_span <= 80.0:
            b_g_mid = 0.5 * (b_g_first + b_g_last)[None, :]
            b_gm = tl.where(m_mc, b_gv - b_g_mid, 0.0)
            b_exp_pos = tl.where(m_mc, tl.exp(b_gm), 0.0)
            b_exp_neg = tl.where(m_mc, tl.exp(-b_gm), 0.0)
            b_bv_pos = b_b * b_v * b_exp_pos
            b_vt_neg = tl.trans(b_v * b_exp_neg)
            b_Avv = tl.dot(b_bv_pos, b_vt_neg, b_Avv, input_precision=DOT_PREC)
        else:
            b_bv = b_b * b_v
            for j in range(0, min(BC, T - i_ti)):
                b_vj = tl.load(v_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
                b_gj = tl.load(gv_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
                m_row = (o_i > j) & m_c
                b_diff = tl.where(m_row[:, None] & m_m[None, :], b_gv - b_gj[None, :], 0.0)
                b_vgj = tl.where(m_row[:, None] & m_m[None, :], b_vj[None, :] * tl.exp(b_diff), 0.0)
                b_avv_j = tl.sum(b_bv * b_vgj, axis=1)
                b_Avv = tl.where((o_i == j)[None, :] & (o_i[:, None] > j), b_avv_j[:, None], b_Avv)

    m_strict = (o_i[:, None] > o_i[None, :]) & m_c[:, None] & ((i_ti + o_i[None, :]) < T)
    m_I = o_i[:, None] == o_i[None, :]
    b_Avv = tl.where(m_strict, b_Avv, 0.0)

    tl.store(Avvd_base + o_c[:, None] * (H * BC) + o_i[None, :], b_Avv, mask=m_c[:, None])
    tl.debug_barrier()

    b_Ai = -b_Avv
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Avvd_base + (i_ti + i) * (H * BC) + o_i).to(tl.float32)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(Avvd_base + o_c[:, None] * (H * BC) + o_i[None, :], b_Ai, mask=m_c[:, None])


@triton.jit(do_not_specialize=["T"])
def _oja2_kkt_offdiag_kernel(
    v_ptr,
    gv_ptr,
    b_ptr,
    Avv_off_ptr,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill off-diagonal 16x16 blocks for Oja-2 slot Gram matrix."""
    i_t, i_r_idx, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_r = i_r_idx + 1
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    i_tr = i_t * BT + i_r * BC
    if i_tr >= T:
        return

    o_i = tl.arange(0, BC)
    m_tr = (i_tr + o_i) < T

    v_base = v_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    b_base = b_ptr + (bos * H + i_h) * M
    Avv_base = Avv_off_ptr + (bos * H + i_h) * BT

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    for i_c in range(0, 3):
        if i_c < i_r:
            i_tc = i_t * BT + i_c * BC
            m_tc = (i_tc + o_i) < T
            b_Avv = tl.zeros([BC, BC], dtype=tl.float32)
            for i_m in range(tl.cdiv(M, BM)):
                o_m = i_m * BM + tl.arange(0, BM)
                m_m = o_m < M
                m_rm = m_tr[:, None] & m_m[None, :]
                m_cm = m_tc[:, None] & m_m[None, :]

                b_vr = tl.load(v_base + (i_tr + o_i)[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
                b_gr = tl.load(gv_base + (i_tr + o_i)[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
                b_br = tl.load(b_base + (i_tr + o_i)[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)

                b_vc = tl.load(v_base + (i_tc + o_i)[:, None] * (H * M) + o_m[None, :], mask=m_cm, other=0.0).to(tl.float32)
                b_gc = tl.load(gv_base + (i_tc + o_i)[:, None] * (H * M) + o_m[None, :], mask=m_cm, other=0.0).to(tl.float32)
                b_gn = tl.load(gv_base + i_tr * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)[None, :]

                b_bvr = b_br * b_vr * tl.where(m_rm, tl.exp(b_gr - b_gn), 0.0)
                b_vct = tl.trans(b_vc * tl.where(m_cm, tl.exp(b_gn - b_gc), 0.0))
                b_Avv = tl.dot(b_bvr, b_vct, b_Avv, input_precision=DOT_PREC)

            tl.store(
                Avv_base + (i_tr + o_i)[:, None] * (H * BT) + (i_c * BC + o_i)[None, :],
                b_Avv.to(Avv_off_ptr.dtype.element_ty),
                mask=m_tr[:, None],
            )


@triton.jit(do_not_specialize=["T"])
def _oja2_wy_fwd_kernel(
    k_ptr,
    v_ptr,
    vg_ptr,
    b_gate_ptr,
    c_gate_ptr,
    w_ptr,
    u_ptr,
    A_ptr,
    gv_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill WY forward for Oja-2: w = A @ (b * v * exp(gv)), u = A @ (c * k), vg = v * exp(gv_last - gv)."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = m_t[:, None] & (o_A[None, :] < BT)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    p_A = A_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    last_idx = min(i_t * BT + BT, T) - 1

    for i_m in range(tl.cdiv(M, BM)):
        o_m = i_m * BM + tl.arange(0, BM)
        m_m = o_m < M
        m_tm = m_t[:, None] & m_m[None, :]
        base_m = (bos * H + i_h) * M + o_t[:, None] * (H * M) + o_m[None, :]

        b_v = tl.load(v_ptr + base_m, mask=m_tm, other=0.0).to(tl.float32)
        b_bg = tl.load(b_gate_ptr + base_m, mask=m_tm, other=0.0).to(tl.float32)
        b_gv = tl.load(gv_ptr + base_m, mask=m_tm, other=0.0).to(tl.float32)
        b_gn = tl.load(gv_ptr + ((bos + last_idx) * H + i_h) * M + o_m, mask=m_m, other=0.0).to(tl.float32)[None, :]

        b_vg = b_v * tl.where(m_tm, tl.exp(b_gn - b_gv), 0.0)
        tl.store(vg_ptr + base_m, b_vg.to(vg_ptr.dtype.element_ty), mask=m_tm)

        b_vb = b_v * b_bg * tl.where(m_tm, tl.exp(b_gv), 0.0)
        if ALLOW_TF32:
            b_w = tl.dot(b_A, b_vb.to(b_A.dtype))
        else:
            b_w = tl.dot(b_A.to(tl.float32), b_vb, input_precision=DOT_PREC)
        tl.store(w_ptr + base_m, b_w.to(w_ptr.dtype.element_ty), mask=m_tm)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        base_k = (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(k_ptr + base_k, mask=m_tk, other=0.0)
        b_cg = tl.load(c_gate_ptr + base_k, mask=m_tk, other=0.0)
        b_kb = (b_k.to(tl.float32) * b_cg.to(tl.float32))
        if ALLOW_TF32:
            b_u = tl.dot(b_A, b_kb.to(b_A.dtype))
        else:
            b_u = tl.dot(b_A.to(tl.float32), b_kb, input_precision=DOT_PREC)
        tl.store(u_ptr + base_k, b_u.to(u_ptr.dtype.element_ty), mask=m_tk)


@triton.jit(do_not_specialize=["T"])
def _oja2_fwd_h_kernel(
    vg_ptr,
    u_ptr,
    w_ptr,
    k_new_ptr,
    gv_ptr,
    h_ptr,
    h0_ptr,
    ht_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """State-invariant BK * M = 4096 Oja-2 inter-chunk recurrence (BK=64 for M=64, BK=32 for M=128, BK=16 for M=256; 0 spills)."""
    i_k, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    o_m1 = tl.arange(0, 64)
    m_m1 = o_m1 < M

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_h1 = tl.zeros([BK, 64], dtype=tl.float32)
    if M > 64:
        b_h2 = tl.zeros([BK, 64], dtype=tl.float32)
    if M > 128:
        b_h3 = tl.zeros([BK, 64], dtype=tl.float32)
    if M > 192:
        b_h4 = tl.zeros([BK, 64], dtype=tl.float32)

    if USE_INITIAL_STATE:
        h0_base = h0_ptr + i_bh * K * M
        b_h1 += tl.load(h0_base + o_k[:, None] * M + o_m1[None, :], mask=m_k[:, None] & m_m1[None, :], other=0.0).to(tl.float32)
        if M > 64:
            b_h2 += tl.load(h0_base + o_k[:, None] * M + (64 + o_m1)[None, :], mask=m_k[:, None] & ((64 + o_m1)[None, :] < M), other=0.0).to(tl.float32)
        if M > 128:
            b_h3 += tl.load(h0_base + o_k[:, None] * M + (128 + o_m1)[None, :], mask=m_k[:, None] & ((128 + o_m1)[None, :] < M), other=0.0).to(tl.float32)
        if M > 192:
            b_h4 += tl.load(h0_base + o_k[:, None] * M + (192 + o_m1)[None, :], mask=m_k[:, None] & ((192 + o_m1)[None, :] < M), other=0.0).to(tl.float32)

    vg_base = vg_ptr + (bos * H + i_h) * M
    w_base = w_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    u_base = u_ptr + (bos * H + i_h) * K
    k_new_base = k_new_ptr + (bos * H + i_h) * K
    h_base = h_ptr + (i_b * NT * H + i_h) * K * M
    stride_h = H * K * M

    for i_t in range(0, NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tk = m_t[:, None] & m_k[None, :]

        tl.store(h_base + i_t * stride_h + o_k[:, None] * M + o_m1[None, :], b_h1.to(h_ptr.dtype.element_ty), mask=m_k[:, None] & m_m1[None, :])
        if M > 64:
            tl.store(h_base + i_t * stride_h + o_k[:, None] * M + (64 + o_m1)[None, :], b_h2.to(h_ptr.dtype.element_ty), mask=m_k[:, None] & ((64 + o_m1)[None, :] < M))
        if M > 128:
            tl.store(h_base + i_t * stride_h + o_k[:, None] * M + (128 + o_m1)[None, :], b_h3.to(h_ptr.dtype.element_ty), mask=m_k[:, None] & ((128 + o_m1)[None, :] < M))
        if M > 192:
            tl.store(h_base + i_t * stride_h + o_k[:, None] * M + (192 + o_m1)[None, :], b_h4.to(h_ptr.dtype.element_ty), mask=m_k[:, None] & ((192 + o_m1)[None, :] < M))

        b_w = tl.load(w_base + o_t[:, None] * (H * M) + o_m1[None, :], mask=m_t[:, None] & m_m1[None, :], other=0.0)
        b_k = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype), input_precision=DOT_PREC)
        if M > 64:
            b_w = tl.load(w_base + o_t[:, None] * (H * M) + (64 + o_m1)[None, :], mask=m_t[:, None] & ((64 + o_m1)[None, :] < M), other=0.0)
            b_k = tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype), b_k, input_precision=DOT_PREC)
        if M > 128:
            b_w = tl.load(w_base + o_t[:, None] * (H * M) + (128 + o_m1)[None, :], mask=m_t[:, None] & ((128 + o_m1)[None, :] < M), other=0.0)
            b_k = tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype), b_k, input_precision=DOT_PREC)
        if M > 192:
            b_w = tl.load(w_base + o_t[:, None] * (H * M) + (192 + o_m1)[None, :], mask=m_t[:, None] & ((192 + o_m1)[None, :] < M), other=0.0)
            b_k = tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype), b_k, input_precision=DOT_PREC)

        b_u = tl.load(u_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
        b_k = b_u - b_k
        tl.store(k_new_base + o_t[:, None] * (H * K) + o_k[None, :], b_k.to(k_new_ptr.dtype.element_ty), mask=m_tk)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_gk_last = tl.load(gv_base + last_idx * (H * M) + o_m1, mask=m_m1, other=0.0).to(tl.float32)
        b_h1 *= tl.exp(b_gk_last)[None, :]
        b_kt = tl.trans(b_k.to(vg_ptr.dtype.element_ty))

        b_v = tl.load(vg_base + o_t[:, None] * (H * M) + o_m1[None, :], mask=m_t[:, None] & m_m1[None, :], other=0.0)
        b_h1 = tl.dot(b_kt, b_v, b_h1, input_precision=DOT_PREC)
        if M > 64:
            b_gk_last = tl.load(gv_base + last_idx * (H * M) + (64 + o_m1), mask=(64 + o_m1) < M, other=0.0).to(tl.float32)
            b_h2 *= tl.exp(b_gk_last)[None, :]
            b_v = tl.load(vg_base + o_t[:, None] * (H * M) + (64 + o_m1)[None, :], mask=m_t[:, None] & ((64 + o_m1)[None, :] < M), other=0.0)
            b_h2 = tl.dot(b_kt, b_v, b_h2, input_precision=DOT_PREC)
        if M > 128:
            b_gk_last = tl.load(gv_base + last_idx * (H * M) + (128 + o_m1), mask=(128 + o_m1) < M, other=0.0).to(tl.float32)
            b_h3 *= tl.exp(b_gk_last)[None, :]
            b_v = tl.load(vg_base + o_t[:, None] * (H * M) + (128 + o_m1)[None, :], mask=m_t[:, None] & ((128 + o_m1)[None, :] < M), other=0.0)
            b_h3 = tl.dot(b_kt, b_v, b_h3, input_precision=DOT_PREC)
        if M > 192:
            b_gk_last = tl.load(gv_base + last_idx * (H * M) + (192 + o_m1), mask=(192 + o_m1) < M, other=0.0).to(tl.float32)
            b_h4 *= tl.exp(b_gk_last)[None, :]
            b_v = tl.load(vg_base + o_t[:, None] * (H * M) + (192 + o_m1)[None, :], mask=m_t[:, None] & ((192 + o_m1)[None, :] < M), other=0.0)
            b_h4 = tl.dot(b_kt, b_v, b_h4, input_precision=DOT_PREC)

    if STORE_FINAL_STATE:
        ht_base = ht_ptr + i_bh * K * M
        tl.store(ht_base + o_k[:, None] * M + o_m1[None, :], b_h1.to(ht_ptr.dtype.element_ty), mask=m_k[:, None] & m_m1[None, :])
        if M > 64:
            tl.store(ht_base + o_k[:, None] * M + (64 + o_m1)[None, :], b_h2.to(ht_ptr.dtype.element_ty), mask=m_k[:, None] & ((64 + o_m1)[None, :] < M))
        if M > 128:
            tl.store(ht_base + o_k[:, None] * M + (128 + o_m1)[None, :], b_h3.to(ht_ptr.dtype.element_ty), mask=m_k[:, None] & ((128 + o_m1)[None, :] < M))
        if M > 192:
            tl.store(ht_base + o_k[:, None] * M + (192 + o_m1)[None, :], b_h4.to(ht_ptr.dtype.element_ty), mask=m_k[:, None] & ((192 + o_m1)[None, :] < M))


@triton.jit(do_not_specialize=["T"])
def _oja2_fwd_o_inter_kernel(
    q_ptr,
    k_new_ptr,
    h_ptr,
    gv_ptr,
    o_ptr,
    Aqk_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill Oja-2 inter-chunk readout o_inter = scale * (q @ h) * exp(gv) and Aqk = tril(scale * q @ k_new^T, 0)."""
    i_m, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t

    o_t = i_t * BT + tl.arange(0, BT)
    o_m = i_m * BM + tl.arange(0, BM)
    m_t = o_t < T
    m_tm = m_t[:, None] & (o_m[None, :] < M)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_o = tl.zeros([BT, BM], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    q_base = q_ptr + (bos * H + i_h) * K
    k_base = k_new_ptr + (bos * H + i_h) * K
    h_base = h_ptr + (i_tg * H + i_h) * K * M

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        m_kt = (o_k[:, None] < K) & m_t[None, :]
        m_km = (o_k[:, None] < K) & (o_m[None, :] < M)

        b_q = tl.load(q_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        b_q = (b_q * scale).to(b_q.dtype)
        b_h = tl.load(h_base + o_k[:, None] * M + o_m[None, :], mask=m_km, other=0.0)
        if ALLOW_TF32:
            b_o = tl.dot(b_q.to(b_h.dtype), b_h, b_o)
        else:
            b_o = tl.dot(b_q.to(tl.float32), b_h.to(tl.float32), b_o, input_precision=DOT_PREC)

        if i_m == 0:
            b_kt = tl.load(k_base + o_k[:, None] + o_t[None, :] * (H * K), mask=m_kt, other=0.0)
            if ALLOW_TF32:
                b_A = tl.dot(b_q, b_kt.to(b_q.dtype), b_A)
            else:
                b_A = tl.dot(b_q.to(tl.float32), b_kt.to(tl.float32), b_A, input_precision=DOT_PREC)

    b_gv = tl.load(gv_ptr + (bos * H + i_h) * M + o_t[:, None] * (H * M) + o_m[None, :], mask=m_tm, other=0.0).to(tl.float32)
    b_o = b_o * tl.where(m_tm, tl.exp(b_gv), 0.0)
    tl.store(o_ptr + (bos * H + i_h) * M + o_t[:, None] * (H * M) + o_m[None, :], b_o.to(o_ptr.dtype.element_ty), mask=m_tm)

    if i_m == 0:
        o_A = tl.arange(0, BT)
        m_s = (o_A[:, None] >= o_A[None, :]) & m_t[:, None] & m_t[None, :]
        b_A = tl.where(m_s, b_A, 0.0)
        tl.store(
            Aqk_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :],
            b_A.to(Aqk_ptr.dtype.element_ty),
            mask=m_t[:, None] & (o_A[None, :] < BT),
        )


@triton.jit(do_not_specialize=["T"])
def _oja2_fwd_o_intra_kernel(
    v_ptr,
    gv_ptr,
    o_ptr,
    Aqk_ptr,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    NC: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Midpoint-anchored 16x16 Tensor Core intra-chunk readout for Oja-2 (eliminates 16-step scalar loop)."""
    i_m, i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    i_last = min(T, i_ti + BC) - 1

    o_i = tl.arange(0, BC)
    o_r = i_ti + o_i
    o_m = i_m * BM + tl.arange(0, BM)
    m_r = o_r < T
    m_m = o_m < M
    m_rm = m_r[:, None] & m_m[None, :]

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    v_base = v_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    Aqk_base = Aqk_ptr + (bos * H + i_h) * BT
    o_base = o_ptr + (bos * H + i_h) * M

    b_gn = tl.load(gv_base + i_ti * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)[None, :]
    b_o = tl.zeros([BC, BM], dtype=tl.float32)

    for i_j in range(0, i_i):
        o_j = i_t * BT + i_j * BC + o_i
        o_Aj = i_j * BC + o_i
        m_jm = (o_j[:, None] < T) & m_m[None, :]
        m_A = m_r[:, None] & (o_Aj[None, :] < BT)

        b_v = tl.load(v_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0).to(tl.float32)
        b_gv = tl.load(gv_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0).to(tl.float32)
        b_vg = b_v * tl.where(m_jm, tl.exp(b_gn - b_gv), 0.0)
        b_A = tl.load(Aqk_base + o_r[:, None] * (H * BT) + o_Aj[None, :], mask=m_A, other=0.0)
        if ALLOW_TF32:
            b_o = tl.dot(b_A, b_vg.to(b_A.dtype), b_o)
        else:
            b_o = tl.dot(b_A.to(tl.float32), b_vg, b_o, input_precision=DOT_PREC)

    b_g = tl.load(gv_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
    b_o *= tl.where(m_rm, tl.exp(b_g - b_gn), 0.0)

    # Diagonal sub-chunk i_j == i_i via midpoint-anchored 16x16 Tensor Core dot
    b_g_last = tl.load(gv_base + i_last * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)[None, :]
    b_span = tl.max(tl.abs(b_gn - b_g_last), axis=1)
    if tl.sum(b_span, axis=0) <= 80.0:
        b_g_mid = 0.5 * (b_gn + b_g_last)
        b_gm = tl.where(m_rm, b_g - b_g_mid, 0.0)
        b_exp_pos = tl.where(m_rm, tl.exp(b_gm), 0.0)
        b_exp_neg = tl.where(m_rm, tl.exp(-b_gm), 0.0)

        o_Ad = i_i * BC + o_i
        m_Ad = m_r[:, None] & (o_Ad[None, :] < BT)
        b_Ad = tl.load(Aqk_base + o_r[:, None] * (H * BT) + o_Ad[None, :], mask=m_Ad, other=0.0).to(tl.float32)
        m_tril = (o_i[:, None] >= o_i[None, :]) & m_r[:, None] & m_r[None, :]
        b_Ad = tl.where(m_tril, b_Ad, 0.0)

        b_vd = tl.load(v_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
        b_o += tl.dot(b_Ad, b_vd * b_exp_neg, input_precision=DOT_PREC) * b_exp_pos
    else:
        o_A_row = o_r * (H * BT) + i_i * BC
        for j in range(0, min(BC, T - i_ti)):
            b_Aj = tl.load(Aqk_base + o_A_row + j, mask=m_r, other=0.0).to(tl.float32)
            b_vj = tl.load(v_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            b_gvj = tl.load(gv_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            m_ge = (o_i >= j)[:, None] & m_rm
            b_vg = tl.where(m_ge, b_vj[None, :] * tl.exp(b_g - b_gvj[None, :]), 0.0)
            b_o += tl.where(m_ge, b_Aj[:, None] * b_vg, 0.0)

    p_o = o_base + o_r[:, None] * (H * M) + o_m[None, :]
    b_o += tl.load(p_o, mask=m_rm, other=0.0).to(tl.float32)
    tl.store(p_o, b_o.to(o_ptr.dtype.element_ty), mask=m_rm)


@triton.jit(do_not_specialize=["T"])
def _oja2_bwd_dA_kernel(
    v_ptr,
    gv_ptr,
    do_ptr,
    dAqk_ptr,
    scale,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    NC: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """In-kernel M-reduced dAqk backward for Oja-2 with midpoint-anchored 16x16 Tensor Core diagonal (no 5D .sum(0))."""
    i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_ij = i_c // (NC * NC), i_c % (NC * NC)
    i_i, i_j = i_ij // NC, i_ij % NC
    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_i = tl.arange(0, BC)
    o_r = i_ti + o_i
    o_Aj = i_j * BC + o_i
    m_r = o_r < T

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    v_base = v_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    do_base = do_ptr + (bos * H + i_h) * M
    dAqk_base = dAqk_ptr + (bos * H + i_h) * BT

    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        o_kj = i_t * BT + i_j * BC + o_i
        m_kj = o_kj < T
        for i_m in range(tl.cdiv(M, BM)):
            o_m = i_m * BM + tl.arange(0, BM)
            m_m = o_m < M
            m_rm = m_r[:, None] & m_m[None, :]
            m_mj = m_m[:, None] & m_kj[None, :]

            b_gn = tl.load(gv_base + i_ti * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            b_g = tl.load(gv_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
            b_do = tl.load(do_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0)
            b_do_g = b_do.to(tl.float32) * tl.where(m_rm, tl.exp(b_g - b_gn[None, :]), 0.0) * scale

            b_vt = tl.load(v_base + o_m[:, None] + o_kj[None, :] * (H * M), mask=m_mj, other=0.0).to(tl.float32)
            b_gvt = tl.load(gv_base + o_m[:, None] + o_kj[None, :] * (H * M), mask=m_mj, other=0.0).to(tl.float32)
            b_vgt = b_vt * tl.where(m_mj, tl.exp(b_gn[:, None] - b_gvt), 0.0)
            if ALLOW_TF32:
                b_dA = tl.dot(b_do_g.to(b_do.dtype), b_vgt.to(b_do.dtype), b_dA)
            else:
                b_dA = tl.dot(b_do_g, b_vgt, b_dA, input_precision=DOT_PREC)
    elif i_i == i_j:
        i_last = min(T, i_ti + BC) - 1
        for i_m in range(tl.cdiv(M, BM)):
            o_m = i_m * BM + tl.arange(0, BM)
            m_m = o_m < M
            m_rm = m_r[:, None] & m_m[None, :]

            b_g = tl.load(gv_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
            b_do = tl.load(do_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32) * scale
            b_v = tl.load(v_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)

            b_g_first = tl.load(gv_base + i_ti * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            b_g_last = tl.load(gv_base + i_last * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            b_span = tl.max(tl.abs(b_g_first - b_g_last), axis=0)
            if b_span <= 80.0:
                b_g_mid = 0.5 * (b_g_first + b_g_last)[None, :]
                b_gm = tl.where(m_rm, b_g - b_g_mid, 0.0)
                b_do_pos = b_do * tl.where(m_rm, tl.exp(b_gm), 0.0)
                b_vt_neg = tl.trans(b_v * tl.where(m_rm, tl.exp(-b_gm), 0.0))
                b_dA = tl.dot(b_do_pos, b_vt_neg, b_dA, input_precision=DOT_PREC)
            else:
                for j in range(0, min(BC, T - i_ti)):
                    b_vj = tl.load(v_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
                    b_gvj = tl.load(gv_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
                    m_ge = (o_i >= j)[:, None] & m_rm
                    b_dAj = tl.sum(tl.where(m_ge, b_do * b_vj[None, :] * tl.exp(b_g - b_gvj[None, :]), 0.0), axis=1)
                    b_dA = tl.where((o_i == j)[None, :] & (o_i[:, None] >= j), b_dA + b_dAj[:, None], b_dA)

        m_tril = (o_i[:, None] >= o_i[None, :]) & m_r[:, None] & m_r[None, :]
        b_dA = tl.where(m_tril, b_dA, 0.0)

    tl.store(
        dAqk_base + o_r[:, None] * (H * BT) + o_Aj[None, :],
        b_dA.to(dAqk_ptr.dtype.element_ty),
        mask=m_r[:, None] & (o_Aj[None, :] < BT),
    )


@triton.jit(do_not_specialize=["T"])
def _oja2_bwd_dqk_kernel(
    q_ptr,
    k_new_ptr,
    h_ptr,
    gv_ptr,
    dAqk_ptr,
    do_ptr,
    dq_ptr,
    dk_new_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill Oja-2 backward for dq and dk_new using saved Aqk from forward (eliminates 5D A.sum(0))."""
    i_k, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_tk = m_t[:, None] & (o_k[None, :] < K)
    m_AT = m_t[:, None] & (o_A[None, :] < BT)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    base_k = (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(q_ptr + base_k, mask=m_tk, other=0.0)
    b_k = tl.load(k_new_ptr + base_k, mask=m_tk, other=0.0)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    h_base = h_ptr + (i_tg * H + i_h) * K * M
    do_base = do_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M

    for i_m in range(tl.cdiv(M, BM)):
        o_m = i_m * BM + tl.arange(0, BM)
        m_tm = m_t[:, None] & (o_m[None, :] < M)
        m_hm = (o_m[:, None] < M) & (o_k[None, :] < K)
        b_h_t = tl.load(h_base + o_m[:, None] + o_k[None, :] * M, mask=m_hm, other=0.0)
        b_do = tl.load(do_base + o_t[:, None] * (H * M) + o_m[None, :], mask=m_tm, other=0.0)
        b_gv = tl.load(gv_base + o_t[:, None] * (H * M) + o_m[None, :], mask=m_tm, other=0.0).to(tl.float32)
        b_do_g = b_do.to(tl.float32) * tl.where(m_tm, tl.exp(b_gv), 0.0) * scale
        if ALLOW_TF32:
            b_dq = tl.dot(b_do_g.to(b_do.dtype), b_h_t.to(b_do.dtype), b_dq)
        else:
            b_dq = tl.dot(b_do_g, b_h_t.to(tl.float32), b_dq, input_precision=DOT_PREC)

    b_dA = tl.load(dAqk_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], mask=m_AT, other=0.0)
    if ALLOW_TF32:
        b_dq = tl.dot(b_dA.to(b_k.dtype), b_k, b_dq)
        b_dk = tl.dot(tl.trans(b_dA).to(b_q.dtype), b_q)
    else:
        b_dq = tl.dot(b_dA.to(tl.float32), b_k.to(tl.float32), b_dq, input_precision=DOT_PREC)
        b_dk = tl.dot(tl.trans(b_dA.to(tl.float32)), b_q.to(tl.float32), input_precision=DOT_PREC)

    tl.store(dq_ptr + base_k, b_dq.to(dq_ptr.dtype.element_ty), mask=m_tk)
    tl.store(dk_new_ptr + base_k, b_dk.to(dk_new_ptr.dtype.element_ty), mask=m_tk)


@triton.jit(do_not_specialize=["T"])
def _oja2_bwd_dhu_kernel(
    q_ptr,
    vg_ptr,
    w_ptr,
    gv_ptr,
    dht_ptr,
    dh0_ptr,
    do_ptr,
    dh_ptr,
    dk_ptr,
    dk2_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRAD: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """State-invariant BK * M = 4096 Oja-2 reverse chunk state adjoint (0 spills across M in {64, 128, 256})."""
    i_k, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    o_m1 = tl.arange(0, 64)
    m_m1 = o_m1 < M

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    b_dh1 = tl.zeros([BK, 64], dtype=tl.float32)
    if M > 64:
        b_dh2 = tl.zeros([BK, 64], dtype=tl.float32)
    if M > 128:
        b_dh3 = tl.zeros([BK, 64], dtype=tl.float32)
    if M > 192:
        b_dh4 = tl.zeros([BK, 64], dtype=tl.float32)

    if USE_FINAL_STATE_GRAD:
        dht_base = dht_ptr + i_bh * K * M
        b_dh1 += tl.load(dht_base + o_k[:, None] * M + o_m1[None, :], mask=m_k[:, None] & m_m1[None, :], other=0.0).to(tl.float32)
        if M > 64:
            b_dh2 += tl.load(dht_base + o_k[:, None] * M + (64 + o_m1)[None, :], mask=m_k[:, None] & ((64 + o_m1)[None, :] < M), other=0.0).to(tl.float32)
        if M > 128:
            b_dh3 += tl.load(dht_base + o_k[:, None] * M + (128 + o_m1)[None, :], mask=m_k[:, None] & ((128 + o_m1)[None, :] < M), other=0.0).to(tl.float32)
        if M > 192:
            b_dh4 += tl.load(dht_base + o_k[:, None] * M + (192 + o_m1)[None, :], mask=m_k[:, None] & ((192 + o_m1)[None, :] < M), other=0.0).to(tl.float32)

    q_base = q_ptr + (bos * H + i_h) * K
    vg_base = vg_ptr + (bos * H + i_h) * M
    w_base = w_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    do_base = do_ptr + (bos * H + i_h) * M
    dk_base = dk_ptr + (bos * H + i_h) * K
    dk2_base = dk2_ptr + (bos * H + i_h) * K
    dh_base = dh_ptr + (i_b * NT * H + i_h) * K * M
    stride_h = H * K * M

    for i_t in range(NT - 1, -1, -1):
        tl.store(dh_base + i_t * stride_h + o_k[:, None] * M + o_m1[None, :], b_dh1.to(dh_ptr.dtype.element_ty), mask=m_k[:, None] & m_m1[None, :])
        if M > 64:
            tl.store(dh_base + i_t * stride_h + o_k[:, None] * M + (64 + o_m1)[None, :], b_dh2.to(dh_ptr.dtype.element_ty), mask=m_k[:, None] & ((64 + o_m1)[None, :] < M))
        if M > 128:
            tl.store(dh_base + i_t * stride_h + o_k[:, None] * M + (128 + o_m1)[None, :], b_dh3.to(dh_ptr.dtype.element_ty), mask=m_k[:, None] & ((128 + o_m1)[None, :] < M))
        if M > 192:
            tl.store(dh_base + i_t * stride_h + o_k[:, None] * M + (192 + o_m1)[None, :], b_dh4.to(dh_ptr.dtype.element_ty), mask=m_k[:, None] & ((192 + o_m1)[None, :] < M))

        for i_s in range(0, 4):
            o_t = i_t * BT + i_s * 16 + tl.arange(0, 16)
            m_t = o_t < T
            m_tk = m_t[:, None] & m_k[None, :]

            b_v = tl.load(vg_base + o_t[:, None] * (H * M) + o_m1[None, :], mask=m_t[:, None] & m_m1[None, :], other=0.0)
            b_dk = tl.dot(b_v, tl.trans(b_dh1).to(b_v.dtype), input_precision=DOT_PREC)
            if M > 64:
                b_v = tl.load(vg_base + o_t[:, None] * (H * M) + (64 + o_m1)[None, :], mask=m_t[:, None] & ((64 + o_m1)[None, :] < M), other=0.0)
                b_dk = tl.dot(b_v, tl.trans(b_dh2).to(b_v.dtype), b_dk, input_precision=DOT_PREC)
            if M > 128:
                b_v = tl.load(vg_base + o_t[:, None] * (H * M) + (128 + o_m1)[None, :], mask=m_t[:, None] & ((128 + o_m1)[None, :] < M), other=0.0)
                b_dk = tl.dot(b_v, tl.trans(b_dh3).to(b_v.dtype), b_dk, input_precision=DOT_PREC)
            if M > 192:
                b_v = tl.load(vg_base + o_t[:, None] * (H * M) + (192 + o_m1)[None, :], mask=m_t[:, None] & ((192 + o_m1)[None, :] < M), other=0.0)
                b_dk = tl.dot(b_v, tl.trans(b_dh4).to(b_v.dtype), b_dk, input_precision=DOT_PREC)

            b_dk += tl.load(dk_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
            tl.store(dk2_base + o_t[:, None] * (H * K) + o_k[None, :], b_dk.to(dk2_ptr.dtype.element_ty), mask=m_tk)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_gv_last = tl.load(gv_base + last_idx * (H * M) + o_m1, mask=m_m1, other=0.0).to(tl.float32)
        b_dh1 *= tl.exp(b_gv_last)[None, :]
        if M > 64:
            b_gv_last = tl.load(gv_base + last_idx * (H * M) + (64 + o_m1), mask=(64 + o_m1) < M, other=0.0).to(tl.float32)
            b_dh2 *= tl.exp(b_gv_last)[None, :]
        if M > 128:
            b_gv_last = tl.load(gv_base + last_idx * (H * M) + (128 + o_m1), mask=(128 + o_m1) < M, other=0.0).to(tl.float32)
            b_dh3 *= tl.exp(b_gv_last)[None, :]
        if M > 192:
            b_gv_last = tl.load(gv_base + last_idx * (H * M) + (192 + o_m1), mask=(192 + o_m1) < M, other=0.0).to(tl.float32)
            b_dh4 *= tl.exp(b_gv_last)[None, :]

        for i_s in range(0, 4):
            o_t = i_t * BT + i_s * 16 + tl.arange(0, 16)
            m_t = o_t < T
            m_tm1 = m_t[:, None] & m_m1[None, :]
            b_qt = tl.load(q_base + o_k[:, None] + o_t[None, :] * (H * K), mask=m_k[:, None] & m_t[None, :], other=0.0)
            b_dkt = tl.load(dk2_base + o_k[:, None] + o_t[None, :] * (H * K), mask=m_k[:, None] & m_t[None, :], other=0.0)

            b_do = tl.load(do_base + o_t[:, None] * (H * M) + o_m1[None, :], mask=m_tm1, other=0.0)
            b_w = tl.load(w_base + o_t[:, None] * (H * M) + o_m1[None, :], mask=m_tm1, other=0.0)
            b_gv = tl.load(gv_base + o_t[:, None] * (H * M) + o_m1[None, :], mask=m_tm1, other=0.0).to(tl.float32)
            b_do = (b_do.to(tl.float32) * tl.where(m_tm1, tl.exp(b_gv), 0.0)).to(b_qt.dtype)
            if ALLOW_TF32:
                b_dh1 += tl.dot(b_qt, b_do) * scale - tl.dot(b_dkt.to(b_w.dtype), b_w, input_precision=DOT_PREC)
            else:
                b_dh1 += tl.dot(b_qt.to(tl.float32), b_do.to(tl.float32), input_precision=DOT_PREC) * scale - tl.dot(b_dkt.to(tl.float32), b_w.to(tl.float32), input_precision=DOT_PREC)

            if M > 64:
                o_m2 = 64 + o_m1
                m_tm2 = m_t[:, None] & (o_m2[None, :] < M)
                b_do = tl.load(do_base + o_t[:, None] * (H * M) + o_m2[None, :], mask=m_tm2, other=0.0)
                b_w = tl.load(w_base + o_t[:, None] * (H * M) + o_m2[None, :], mask=m_tm2, other=0.0)
                b_gv = tl.load(gv_base + o_t[:, None] * (H * M) + o_m2[None, :], mask=m_tm2, other=0.0).to(tl.float32)
                b_do = (b_do.to(tl.float32) * tl.where(m_tm2, tl.exp(b_gv), 0.0)).to(b_qt.dtype)
                if ALLOW_TF32:
                    b_dh2 += tl.dot(b_qt, b_do) * scale - tl.dot(b_dkt.to(b_w.dtype), b_w, input_precision=DOT_PREC)
                else:
                    b_dh2 += tl.dot(b_qt.to(tl.float32), b_do.to(tl.float32), input_precision=DOT_PREC) * scale - tl.dot(b_dkt.to(tl.float32), b_w.to(tl.float32), input_precision=DOT_PREC)

            if M > 128:
                o_m3 = 128 + o_m1
                m_tm3 = m_t[:, None] & (o_m3[None, :] < M)
                b_do = tl.load(do_base + o_t[:, None] * (H * M) + o_m3[None, :], mask=m_tm3, other=0.0)
                b_w = tl.load(w_base + o_t[:, None] * (H * M) + o_m3[None, :], mask=m_tm3, other=0.0)
                b_gv = tl.load(gv_base + o_t[:, None] * (H * M) + o_m3[None, :], mask=m_tm3, other=0.0).to(tl.float32)
                b_do = (b_do.to(tl.float32) * tl.where(m_tm3, tl.exp(b_gv), 0.0)).to(b_qt.dtype)
                if ALLOW_TF32:
                    b_dh3 += tl.dot(b_qt, b_do) * scale - tl.dot(b_dkt.to(b_w.dtype), b_w, input_precision=DOT_PREC)
                else:
                    b_dh3 += tl.dot(b_qt.to(tl.float32), b_do.to(tl.float32), input_precision=DOT_PREC) * scale - tl.dot(b_dkt.to(tl.float32), b_w.to(tl.float32), input_precision=DOT_PREC)

            if M > 192:
                o_m4 = 192 + o_m1
                m_tm4 = m_t[:, None] & (o_m4[None, :] < M)
                b_do = tl.load(do_base + o_t[:, None] * (H * M) + o_m4[None, :], mask=m_tm4, other=0.0)
                b_w = tl.load(w_base + o_t[:, None] * (H * M) + o_m4[None, :], mask=m_tm4, other=0.0)
                b_gv = tl.load(gv_base + o_t[:, None] * (H * M) + o_m4[None, :], mask=m_tm4, other=0.0).to(tl.float32)
                b_do = (b_do.to(tl.float32) * tl.where(m_tm4, tl.exp(b_gv), 0.0)).to(b_qt.dtype)
                if ALLOW_TF32:
                    b_dh4 += tl.dot(b_qt, b_do) * scale - tl.dot(b_dkt.to(b_w.dtype), b_w, input_precision=DOT_PREC)
                else:
                    b_dh4 += tl.dot(b_qt.to(tl.float32), b_do.to(tl.float32), input_precision=DOT_PREC) * scale - tl.dot(b_dkt.to(tl.float32), b_w.to(tl.float32), input_precision=DOT_PREC)

    if USE_INITIAL_STATE:
        dh0_base = dh0_ptr + i_bh * K * M
        tl.store(dh0_base + o_k[:, None] * M + o_m1[None, :], b_dh1, mask=m_k[:, None] & m_m1[None, :])
        if M > 64:
            tl.store(dh0_base + o_k[:, None] * M + (64 + o_m1)[None, :], b_dh2, mask=m_k[:, None] & ((64 + o_m1)[None, :] < M))
        if M > 128:
            tl.store(dh0_base + o_k[:, None] * M + (128 + o_m1)[None, :], b_dh3, mask=m_k[:, None] & ((128 + o_m1)[None, :] < M))
        if M > 192:
            tl.store(dh0_base + o_k[:, None] * M + (192 + o_m1)[None, :], b_dh4, mask=m_k[:, None] & ((192 + o_m1)[None, :] < M))


@triton.jit(do_not_specialize=["T"])
def _oja2_bwd_dvwg_h_kernel(
    k_new_ptr,
    v_ptr,
    gv_ptr,
    h_ptr,
    dh_ptr,
    dk_new2_ptr,
    dw_ptr,
    dv_ptr,
    dgv_last_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Zero-spill Oja-2 backward for dv, dw, and dgv_last from chunk states h and dh."""
    i_m, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t

    o_t = i_t * BT + tl.arange(0, BT)
    o_m = i_m * BM + tl.arange(0, BM)
    m_t = o_t < T
    m_m = o_m < M
    m_tm = m_t[:, None] & m_m[None, :]

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    k_base = k_new_ptr + (bos * H + i_h) * K
    dk_base = dk_new2_ptr + (bos * H + i_h) * K
    v_base = v_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    h_base = h_ptr + (i_tg * H + i_h) * K * M
    dh_base = dh_ptr + (i_tg * H + i_h) * K * M

    b_dvg = tl.zeros([BT, BM], dtype=tl.float32)
    b_dw = tl.zeros([BT, BM], dtype=tl.float32)
    b_dgv_last = tl.zeros([BM], dtype=tl.float32)

    last_idx = min(T, i_t * BT + BT) - 1
    b_gn = tl.load(gv_base + last_idx * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
    b_gv = tl.load(gv_base + o_t[:, None] * (H * M) + o_m[None, :], mask=m_tm, other=0.0).to(tl.float32)
    b_exp_gn = tl.exp(b_gn)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        m_km = (o_k[:, None] < K) & m_m[None, :]

        b_k = tl.load(k_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        b_dk = tl.load(dk_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        b_h = tl.load(h_base + o_k[:, None] * M + o_m[None, :], mask=m_km, other=0.0)
        b_dh = tl.load(dh_base + o_k[:, None] * M + o_m[None, :], mask=m_km, other=0.0)

        if ALLOW_TF32:
            b_dvg = tl.dot(b_k, b_dh.to(b_k.dtype), b_dvg)
            b_dw = tl.dot(b_dk.to(b_k.dtype), b_h.to(b_k.dtype), b_dw)
        else:
            b_dvg = tl.dot(b_k.to(tl.float32), b_dh.to(tl.float32), b_dvg, input_precision=DOT_PREC)
            b_dw = tl.dot(b_dk.to(tl.float32), b_h.to(tl.float32), b_dw, input_precision=DOT_PREC)
        b_dgv_last += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32), axis=0) * b_exp_gn

    b_dv = b_dvg * tl.where(m_tm, tl.exp(b_gn[None, :] - b_gv), 0.0)
    base_m = (bos * H + i_h) * M + o_t[:, None] * (H * M) + o_m[None, :]
    b_v = tl.load(v_base + o_t[:, None] * (H * M) + o_m[None, :], mask=m_tm, other=0.0).to(tl.float32)
    b_dgv_last += tl.sum(b_dv * b_v, axis=0)

    m_last = o_t == last_idx
    b_dgv_out = tl.where(m_last[:, None] & m_tm, b_dgv_last[None, :], 0.0)

    tl.store(dv_ptr + base_m, b_dv.to(dv_ptr.dtype.element_ty), mask=m_tm)
    tl.store(dw_ptr + base_m, (-b_dw).to(dw_ptr.dtype.element_ty), mask=m_tm)
    tl.store(dgv_last_ptr + base_m, b_dgv_out.to(dgv_last_ptr.dtype.element_ty), mask=m_tm)


@triton.jit(do_not_specialize=["T"])
def _oja2_bwd_dv_o_kernel(
    v_ptr,
    gv_ptr,
    o_ptr,
    Aqk_ptr,
    do_ptr,
    dv_in_ptr,
    dv_out_ptr,
    dgv1_ptr,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    NC: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Midpoint-anchored 16x16 Tensor Core backward for dv_o and dgv1 in Oja-2 (eliminates 16-step scalar loop)."""
    i_m, i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    i_last = min(T, i_ti + BC) - 1

    o_i = tl.arange(0, BC)
    o_r = i_ti + o_i
    o_m = i_m * BM + tl.arange(0, BM)
    m_r = o_r < T
    m_m = o_m < M
    m_rm = m_r[:, None] & m_m[None, :]

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    v_base = v_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    o_base = o_ptr + (bos * H + i_h) * M
    do_base = do_ptr + (bos * H + i_h) * M
    Aqk_base = Aqk_ptr + (bos * H + i_h) * BT

    b_gn = tl.load(gv_base + i_last * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)[None, :]
    b_gv = tl.load(gv_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
    b_dvg = tl.zeros([BC, BM], dtype=tl.float32)

    NC_valid = min(NC, tl.cdiv(T - i_t * BT, BC))
    for i_j in range(i_i + 1, NC_valid):
        o_j = i_t * BT + i_j * BC + o_i
        o_Ai = i_i * BC + o_i
        m_jm = (o_j[:, None] < T) & m_m[None, :]
        m_A = (o_Ai[:, None] < BT) & (o_j[None, :] < T)

        b_g = tl.load(gv_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0).to(tl.float32)
        b_do = tl.load(do_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0)
        b_do_g = b_do.to(tl.float32) * tl.where(m_jm, tl.exp(b_g - b_gn), 0.0)
        b_A = tl.load(Aqk_base + o_Ai[:, None] + o_j[None, :] * (H * BT), mask=m_A, other=0.0)
        if ALLOW_TF32:
            b_dvg = tl.dot(b_A, b_do_g.to(b_A.dtype), b_dvg)
        else:
            b_dvg = tl.dot(b_A.to(tl.float32), b_do_g, b_dvg, input_precision=DOT_PREC)

    b_dv = b_dvg * tl.where(m_rm, tl.exp(b_gn - b_gv), 0.0)

    # Diagonal sub-chunk i_j == i_i via midpoint-anchored 16x16 Tensor Core dot
    b_g_first = tl.load(gv_base + i_ti * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)[None, :]
    b_span = tl.max(tl.abs(b_g_first - b_gn), axis=1)
    if tl.sum(b_span, axis=0) <= 80.0:
        b_g_mid = 0.5 * (b_g_first + b_gn)
        b_gm = tl.where(m_rm, b_gv - b_g_mid, 0.0)
        b_exp_pos = tl.where(m_rm, tl.exp(b_gm), 0.0)
        b_exp_neg = tl.where(m_rm, tl.exp(-b_gm), 0.0)

        o_Ai = i_i * BC + o_i
        m_Ad = (o_Ai[:, None] < BT) & m_r[None, :]
        b_AT = tl.load(Aqk_base + o_Ai[:, None] + o_r[None, :] * (H * BT), mask=m_Ad, other=0.0).to(tl.float32)
        m_upper = (o_i[:, None] <= o_i[None, :]) & m_r[:, None] & m_r[None, :]
        b_AT = tl.where(m_upper, b_AT, 0.0)

        b_do_d = tl.load(do_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
        b_dv += tl.dot(b_AT, b_do_d * b_exp_pos, input_precision=DOT_PREC) * b_exp_neg
    else:
        o_c_idx = i_i * BC + o_i
        for j in range(0, min(BC, T - i_ti)):
            b_A = tl.load(Aqk_base + (i_ti + j) * (H * BT) + o_c_idx, mask=m_r, other=0.0).to(tl.float32)
            b_g = tl.load(gv_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            b_do = tl.load(do_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            m_le = (o_i <= j)[:, None] & m_rm
            b_dv += tl.where(m_le, tl.exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.0)

    base_out = (bos * H + i_h) * M + o_r[:, None] * (H * M) + o_m[None, :]
    b_o = tl.load(o_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
    b_v = tl.load(v_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
    b_do_r = tl.load(do_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
    b_dv_tot = b_dv + tl.load(dv_in_ptr + base_out, mask=m_rm, other=0.0).to(tl.float32)
    b_dg = b_o * b_do_r - b_v * b_dv_tot

    tl.store(dv_out_ptr + base_out, b_dv_tot.to(dv_out_ptr.dtype.element_ty), mask=m_rm)
    tl.store(dgv1_ptr + base_out, b_dg.to(dgv1_ptr.dtype.element_ty), mask=m_rm)


@triton.jit(do_not_specialize=["T"])
def _oja2_bwd_wy_v_pass1_kernel(
    v_ptr,
    b_gate_ptr,
    gv_ptr,
    A_ptr,
    dw_ptr,
    dv1_ptr,
    db_gate_ptr,
    dgv2_ptr,
    dA_v_ptr,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Pass 1 of Two-Pass Zero-Spill Oja-2 WY Backward: w path (w = A @ (b * v * exp(gv)))."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    m_AT2 = m_t[:, None] & (o_A[None, :] < BT)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    p_A = A_ptr + (bos * H + i_h) * BT + o_A[:, None] + o_t[None, :] * (H * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_m in range(tl.cdiv(M, BM)):
        o_m = i_m * BM + tl.arange(0, BM)
        m_tm = m_t[:, None] & (o_m[None, :] < M)
        base_m = (bos * H + i_h) * M + o_t[:, None] * (H * M) + o_m[None, :]

        b_v = tl.load(v_ptr + base_m, mask=m_tm, other=0.0).to(tl.float32)
        b_bg = tl.load(b_gate_ptr + base_m, mask=m_tm, other=0.0).to(tl.float32)
        b_gv_exp = tl.where(m_tm, tl.exp(tl.load(gv_ptr + base_m, mask=m_tm, other=0.0).to(tl.float32)), 0.0)
        b_vbg = b_bg * b_v * b_gv_exp
        b_dw = tl.load(dw_ptr + base_m, mask=m_tm, other=0.0)

        if ALLOW_TF32:
            b_dA = tl.dot(b_dw, tl.trans(b_vbg).to(b_dw.dtype), b_dA, input_precision=DOT_PREC)
            b_dvbg = tl.dot(b_A, b_dw.to(b_A.dtype), input_precision=DOT_PREC)
        else:
            b_dA = tl.dot(b_dw.to(tl.float32), tl.trans(b_vbg), b_dA, input_precision=DOT_PREC)
            b_dvbg = tl.dot(b_A.to(tl.float32), b_dw.to(tl.float32), input_precision=DOT_PREC)

        b_dv = b_dvbg * b_bg * b_gv_exp
        b_dbg = b_dvbg * b_v * b_gv_exp
        b_dgv = b_dvbg * b_vbg

        tl.store(dv1_ptr + base_m, b_dv.to(dv1_ptr.dtype.element_ty), mask=m_tm)
        tl.store(db_gate_ptr + base_m, b_dbg.to(db_gate_ptr.dtype.element_ty), mask=m_tm)
        tl.store(dgv2_ptr + base_m, b_dgv.to(dgv2_ptr.dtype.element_ty), mask=m_tm)

    tl.store(dA_v_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], b_dA, mask=m_AT2)


@triton.jit(do_not_specialize=["T"])
def _oja2_bwd_wy_k_pass2_kernel(
    k_ptr,
    c_gate_ptr,
    A_ptr,
    du_ptr,
    dA_v_ptr,
    dk_ptr,
    dc_gate_ptr,
    dA_raw_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Pass 2 of Two-Pass Zero-Spill Oja-2 WY Backward: u path (u = A @ (c * k)) accumulating dA_raw."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    m_AT2 = m_t[:, None] & (o_A[None, :] < BT)

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    p_A = A_ptr + (bos * H + i_h) * BT + o_A[:, None] + o_t[None, :] * (H * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.load(dA_v_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], mask=m_AT2, other=0.0).to(tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        base_k = (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]

        b_k = tl.load(k_ptr + base_k, mask=m_tk, other=0.0)
        b_cg = tl.load(c_gate_ptr + base_k, mask=m_tk, other=0.0)
        b_kb = (b_cg * b_k).to(b_k.dtype)
        b_du = tl.load(du_ptr + base_k, mask=m_tk, other=0.0)

        if ALLOW_TF32:
            b_dA = tl.dot(b_du, tl.trans(b_kb), b_dA)
            b_dkb = tl.dot(b_A, b_du.to(b_A.dtype))
        else:
            b_dA = tl.dot(b_du.to(tl.float32), tl.trans(b_kb.to(tl.float32)), b_dA, input_precision=DOT_PREC)
            b_dkb = tl.dot(b_A.to(tl.float32), b_du.to(tl.float32), input_precision=DOT_PREC)

        b_dk = b_dkb * b_cg.to(tl.float32)
        b_dcg = b_dkb * b_k.to(tl.float32)
        tl.store(dk_ptr + base_k, b_dk.to(dk_ptr.dtype.element_ty), mask=m_tk)
        tl.store(dc_gate_ptr + base_k, b_dcg.to(dc_gate_ptr.dtype.element_ty), mask=m_tk)

    tl.store(dA_raw_ptr + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :], b_dA, mask=m_AT2)


@triton.jit(do_not_specialize=["T"])
def _oja2_bwd_kkt_kernel(
    v_ptr,
    gv_ptr,
    b_ptr,
    dAvv_ptr,
    dv_in_ptr,
    dv_out_ptr,
    dgv3_ptr,
    db_in_ptr,
    db_out_ptr,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    NC: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Midpoint-anchored 16x16 Tensor Core backward for Oja-2 KKT (eliminates 16-step scalar loop & 5D db.sum(0))."""
    i_mc, i_t = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_m, i_i = i_mc // NC, i_mc % NC

    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    i_last = min(T, i_ti + BC) - 1

    o_i = tl.arange(0, BC)
    o_r = i_ti + o_i
    o_m = i_m * BM + tl.arange(0, BM)
    m_r = o_r < T
    m_m = o_m < M
    m_rm = m_r[:, None] & m_m[None, :]

    DOT_PREC: tl.constexpr = "tf32" if ALLOW_TF32 else "ieee"

    v_base = v_ptr + (bos * H + i_h) * M
    gv_base = gv_ptr + (bos * H + i_h) * M
    b_base = b_ptr + (bos * H + i_h) * M
    dAvv_base = dAvv_ptr + (bos * H + i_h) * BT

    b_v = tl.load(v_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
    b_g = tl.load(gv_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)
    b_b = tl.load(b_base + o_r[:, None] * (H * M) + o_m[None, :], mask=m_rm, other=0.0).to(tl.float32)

    b_dv = tl.zeros([BC, BM], dtype=tl.float32)
    if i_i > 0:
        b_gn = tl.load(gv_base + i_ti * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + o_i
            o_dAj = i_j * BC + o_i
            m_jm = (o_j[:, None] < T) & m_m[None, :]
            m_dA = m_r[:, None] & (o_dAj[None, :] < BT)

            b_vj = tl.load(v_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0).to(tl.float32)
            b_gj = tl.load(gv_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0).to(tl.float32)
            b_vg = b_vj * tl.where(m_jm, tl.exp(b_gn - b_gj), 0.0)
            b_dA = tl.load(dAvv_base + o_r[:, None] * (H * BT) + o_dAj[None, :], mask=m_dA, other=0.0).to(tl.float32)
            b_dv += tl.dot(b_dA, b_vg, input_precision=DOT_PREC) * tl.where(m_rm, tl.exp(b_g - b_gn), 0.0)

    # Diagonal sub-chunk i_j == i_i via midpoint-anchored 16x16 Tensor Core dot
    b_g_first = tl.load(gv_base + i_ti * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
    b_g_last = tl.load(gv_base + i_last * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
    b_span = tl.max(tl.abs(b_g_first - b_g_last), axis=0)

    b_dvt = tl.zeros([BC, BM], dtype=tl.float32)
    if b_span <= 80.0:
        b_g_mid = 0.5 * (b_g_first + b_g_last)[None, :]
        b_gm = tl.where(m_rm, b_g - b_g_mid, 0.0)
        b_exp_pos = tl.where(m_rm, tl.exp(b_gm), 0.0)
        b_exp_neg = tl.where(m_rm, tl.exp(-b_gm), 0.0)

        m_dd = m_r[:, None] & ((i_i * BC + o_i)[None, :] < BT)
        b_dA_d = tl.load(dAvv_base + o_r[:, None] * (H * BT) + (i_i * BC + o_i)[None, :], mask=m_dd, other=0.0).to(tl.float32)
        m_strict = (o_i[:, None] > o_i[None, :]) & m_r[:, None] & m_r[None, :]
        b_dA_d = tl.where(m_strict, b_dA_d, 0.0)

        b_dv += tl.dot(b_dA_d, b_v * b_exp_neg, input_precision=DOT_PREC) * b_exp_pos
        b_dvt += tl.dot(tl.trans(b_dA_d), b_v * b_b * b_exp_pos, input_precision=DOT_PREC) * b_exp_neg
    else:
        o_dA_row = o_r * (H * BT) + i_i * BC
        o_dA_col = i_ti * (H * BT) + i_i * BC + o_i
        for j in range(0, min(BC, T - i_ti)):
            b_dA_r = tl.load(dAvv_base + o_dA_row + j, mask=m_r, other=0.0).to(tl.float32)
            b_vj = tl.load(v_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            b_bj = tl.load(b_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)
            b_gj = tl.load(gv_base + (i_ti + j) * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)

            m_gt = (o_i > j)[:, None] & m_rm
            b_dv += tl.where(m_gt, b_dA_r[:, None] * b_vj[None, :] * tl.exp(b_g - b_gj[None, :]), 0.0)

            b_dA_c = tl.load(dAvv_base + o_dA_col + j * (H * BT), mask=m_r, other=0.0).to(tl.float32)
            m_lt = (o_i < j)[:, None] & m_rm
            b_dvt += tl.where(m_lt, b_dA_c[:, None] * (b_vj * b_bj)[None, :] * tl.exp(b_gj[None, :] - b_g), 0.0)

    b_db_tile = b_dv * b_v
    b_dv *= b_b

    base_out = (bos * H + i_h) * M + o_r[:, None] * (H * M) + o_m[None, :]
    b_db_tot = b_db_tile + tl.load(db_in_ptr + base_out, mask=m_rm, other=0.0).to(tl.float32)
    tl.store(db_out_ptr + base_out, b_db_tot.to(db_out_ptr.dtype.element_ty), mask=m_rm)

    NC_valid = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC_valid - 1:
        b_gn_u = tl.load(gv_base + i_last * (H * M) + o_m, mask=m_m, other=0.0).to(tl.float32)[None, :]
        b_dvt_off = tl.zeros([BC, BM], dtype=tl.float32)
        for i_j in range(i_i + 1, NC_valid):
            o_j = i_t * BT + i_j * BC + o_i
            o_dAi = i_i * BC + o_i
            m_jm = (o_j[:, None] < T) & m_m[None, :]
            m_dA = (o_dAi[:, None] < BT) & (o_j[None, :] < T)

            b_bj = tl.load(b_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0).to(tl.float32)
            b_vj = tl.load(v_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0).to(tl.float32)
            b_gj = tl.load(gv_base + o_j[:, None] * (H * M) + o_m[None, :], mask=m_jm, other=0.0).to(tl.float32)
            b_vbg = b_vj * b_bj * tl.where(m_jm, tl.exp(b_gj - b_gn_u), 0.0)
            b_dA_u = tl.load(dAvv_base + o_dAi[:, None] + o_j[None, :] * (H * BT), mask=m_dA, other=0.0).to(tl.float32)
            b_dvt_off = tl.dot(b_dA_u, b_vbg, b_dvt_off, input_precision=DOT_PREC)
        b_dvt += b_dvt_off * tl.where(m_rm, tl.exp(b_gn_u - b_g), 0.0)

    b_dg = (b_dv - b_dvt) * b_v
    b_dv_tot = b_dv + b_dvt + tl.load(dv_in_ptr + base_out, mask=m_rm, other=0.0).to(tl.float32)

    tl.store(dv_out_ptr + base_out, b_dv_tot.to(dv_out_ptr.dtype.element_ty), mask=m_rm)
    tl.store(dgv3_ptr + base_out, b_dg.to(dgv3_ptr.dtype.element_ty), mask=m_rm)


class _ChunkOja2RSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gv: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        use_q_l2norm: bool,
        use_k_l2norm: bool,
    ):
        B, T, H, K = q.shape
        M = v.shape[-1]
        BT = 64
        BC = 16
        BK = 64
        BM = 64
        NT = triton.cdiv(T, BT)
        allow_tf32 = q.dtype != torch.float32

        q_raw, k_raw = q, k
        q_rstd, k_rstd = None, None
        if use_q_l2norm:
            q, q_rstd = l2norm_fwd_rsi(q)
        if use_k_l2norm:
            k, k_rstd = l2norm_fwd_rsi(k)

        gv_cumsum = chunk_local_cumsum_rsi(gv, scale=1.0, chunk_size=BT)

        Avvd = torch.empty(B, T, H, BC, device=q.device, dtype=torch.float32)
        Avv_off = torch.empty(B, T, H, BT, device=q.device, dtype=k.dtype)
        Avv = torch.empty(B, T, H, BT, device=q.device, dtype=k.dtype)

        _oja2_kkt_sub_chunk_diag_kernel[(NT, 4, B * H)](
            v, gv_cumsum, b, Avvd, T,
            H=H, M=M, BT=BT, BC=BC, BM=BM, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )
        _oja2_kkt_offdiag_kernel[(NT, 3, B * H)](
            v, gv_cumsum, b, Avv_off, T,
            H=H, M=M, BT=BT, BC=BC, BM=BM, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )
        _merge_16x16_to_64x64_inverse_kernel[(NT, B * H)](
            Avvd, Avv_off, Avv, T,
            H=H, BT=BT, BC=BC, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        w = torch.empty(B, T, H, M, device=q.device, dtype=q.dtype)
        u = torch.empty_like(k)
        vg = torch.empty(B, T, H, M, device=q.device, dtype=q.dtype)
        _oja2_wy_fwd_kernel[(NT, B * H)](
            k, v, vg, b, c, w, u, Avv, gv_cumsum, T,
            H=H, K=K, M=M, BT=BT, BK=32, BM=32, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        # State-invariant BK_H = 16 with num_warps=8 so every CTA holds <= 16 FP32 state regs/thread
        BK_H = 16
        NK_H = triton.cdiv(K, BK_H)
        h = torch.empty(B, NT, H, K, M, device=q.device, dtype=q.dtype)
        k_new = torch.empty_like(u)
        final_state = torch.empty(B, H, K, M, device=q.device, dtype=torch.float32) if output_final_state else None
        _oja2_fwd_h_kernel[(NK_H, B * H)](
            vg, u, w, k_new, gv_cumsum, h, initial_state, final_state, T,
            H=H, K=K, M=M, BT=BT, BK=BK_H,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        NM = triton.cdiv(M, BM)
        NC = BT // BC
        o = torch.empty(B, T, H, M, device=q.device, dtype=q.dtype)
        Aqk = torch.empty(B, T, H, BT, device=q.device, dtype=q.dtype)
        _oja2_fwd_o_inter_kernel[(NM, NT, B * H)](
            q, k_new, h, gv_cumsum, o, Aqk, scale, T,
            H=H, K=K, M=M, BT=BT, BK=32, BM=BM, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )
        _oja2_fwd_o_intra_kernel[(NM, NT * NC, B * H)](
            v, gv_cumsum, o, Aqk, T,
            H=H, M=M, BT=BT, BC=BC, BM=BM, NC=NC, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        ctx.save_for_backward(
            q_raw, k_raw, q, k, v, gv_cumsum, b, c, Avv, Aqk, w, vg, k_new, h, o, initial_state, q_rstd, k_rstd
        )
        ctx.scale = scale
        ctx.use_q_l2norm = use_q_l2norm
        ctx.use_k_l2norm = use_k_l2norm
        return o.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        (
            q_raw, k_raw, q, k, v, gv_cumsum, b, c, Avv, Aqk, w, vg, k_new, h, o, initial_state, q_rstd, k_rstd
        ) = ctx.saved_tensors
        scale = ctx.scale
        B, T, H, K = q.shape
        M = v.shape[-1]
        BT = 64
        BC = 16
        BK = 64
        BM = 64
        NT = triton.cdiv(T, BT)
        NC = BT // BC
        NM = triton.cdiv(M, BM)
        NK = triton.cdiv(K, BK)
        allow_tf32 = q.dtype != torch.float32
        do = do.contiguous()

        dAqk = torch.empty(B, T, H, BT, device=q.device, dtype=q.dtype)
        _oja2_bwd_dA_kernel[(NT * NC * NC, B * H)](
            v, gv_cumsum, do, dAqk, scale, T,
            H=H, M=M, BT=BT, BC=BC, BM=BM, NC=NC, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        dq = torch.empty_like(q)
        dk_new = torch.empty_like(k_new)
        _oja2_bwd_dqk_kernel[(NK, NT, B * H)](
            q, k_new, h, gv_cumsum, dAqk, do, dq, dk_new, scale, T,
            H=H, K=K, M=M, BT=BT, BK=BK, BM=32, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        BK_H = 16
        NK_H = triton.cdiv(K, BK_H)
        dh = torch.empty(B, NT, H, K, M, device=q.device, dtype=q.dtype)
        dh0 = torch.empty(B, H, K, M, device=q.device, dtype=torch.float32) if initial_state is not None else None
        dk_new2 = torch.empty_like(dk_new)
        _oja2_bwd_dhu_kernel[(NK_H, B * H)](
            q, vg, w, gv_cumsum, dht, dh0, do, dh, dk_new, dk_new2, scale, T,
            H=H, K=K, M=M, BT=BT, BK=BK_H,
            USE_INITIAL_STATE=initial_state is not None,
            USE_FINAL_STATE_GRAD=dht is not None,
            ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        BM_S = 32
        NM_S = triton.cdiv(M, BM_S)
        dv_h = torch.empty(B, T, H, M, device=q.device, dtype=torch.float32)
        dw = torch.empty(B, T, H, M, device=q.device, dtype=q.dtype)
        dgv_last = torch.empty_like(gv_cumsum)
        _oja2_bwd_dvwg_h_kernel[(NM_S, NT, B * H)](
            k_new, v, gv_cumsum, h, dh, dk_new2, dw, dv_h, dgv_last, T,
            H=H, K=K, M=M, BT=BT, BK=32, BM=BM_S, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        dv_o = torch.empty(B, T, H, M, device=q.device, dtype=torch.float32)
        dgv1 = torch.empty_like(gv_cumsum)
        _oja2_bwd_dv_o_kernel[(NM, NT * NC, B * H)](
            v, gv_cumsum, o, Aqk, do, dv_h, dv_o, dgv1, T,
            H=H, M=M, BT=BT, BC=BC, BM=BM, NC=NC, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        dv1 = torch.empty(B, T, H, M, device=q.device, dtype=torch.float32)
        db_gate = torch.empty(B, T, H, M, device=q.device, dtype=torch.float32)
        dgv2 = torch.empty_like(gv_cumsum)
        dA_v = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
        _oja2_bwd_wy_v_pass1_kernel[(NT, B * H)](
            v, b, gv_cumsum, Avv, dw, dv1, db_gate, dgv2, dA_v, T,
            H=H, M=M, BT=BT, BM=16, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        dk = torch.empty_like(k)
        dc = torch.empty_like(c)
        dA_raw = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
        dAvv = torch.empty(B, T, H, BT, device=q.device, dtype=Avv.dtype)
        _oja2_bwd_wy_k_pass2_kernel[(NT, B * H)](
            k, c, Avv, dk_new2, dA_v, dk, dc, dA_raw, T,
            H=H, K=K, BT=BT, BK=16, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )
        _bwd_dA_triple_product_kernel[(NT, B * H)](
            Avv, dA_raw, dAvv, T,
            H=H, BT=BT, ALLOW_TF32=allow_tf32,
            num_warps=8, num_stages=1,
        )

        dv_pre = dv_o + dv1
        dv_out = torch.empty(B, T, H, M, device=q.device, dtype=v.dtype)
        dgv3 = torch.empty_like(gv_cumsum)
        db_out = torch.empty_like(b)
        _oja2_bwd_kkt_kernel[(NM * NC, NT, B * H)](
            v, gv_cumsum, b, dAvv, dv_pre, dv_out, dgv3, db_gate, db_out, T,
            H=H, M=M, BT=BT, BC=BC, BM=BM, NC=NC, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        dgv_sum = dgv_last + dgv1 + dgv2 + dgv3
        dgv = chunk_local_cumsum_rev_rsi(dgv_sum, scale=1.0, chunk_size=BT)

        if ctx.use_q_l2norm:
            dq = l2norm_bwd_rsi(q_raw, q_rstd, dq)
        if ctx.use_k_l2norm:
            dk = l2norm_bwd_rsi(k_raw, k_rstd, dk)

        return (
            dq,
            dk,
            dv_out,
            dgv.to(gv_cumsum.dtype),
            db_out,
            dc,
            None,
            dh0.to(initial_state.dtype) if dh0 is not None else None,
            None,
            None,
            None,
        )


def chunk_gated_oja_rule2_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_q_l2norm: bool = False,
    use_k_l2norm: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _ChunkOja2RSIFunction.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        gv.contiguous(),
        b.contiguous(),
        c.contiguous(),
        scale,
        initial_state.contiguous() if initial_state is not None else None,
        output_final_state,
        use_q_l2norm,
        use_k_l2norm,
    )


# ======================================================================================
# 5. Superfused Peripheral Kernels (Multi-Branch ShortConv1d, Slot L2Norm+Gates, Bridge)
# ======================================================================================

@triton.jit(do_not_specialize=["T"])
def _short_conv1d_k4_silu_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Zero-spill causal depthwise Conv1d (kernel_size=4, bias=False) + SiLU."""
    i_d, i_t, i_b = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    w0 = tl.load(w_ptr + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)[None, :]
    w1 = tl.load(w_ptr + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)[None, :]
    w2 = tl.load(w_ptr + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)[None, :]
    w3 = tl.load(w_ptr + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)[None, :]

    base_b = i_b * T * D
    x0 = tl.load(x_ptr + base_b + (o_t - 3)[:, None] * D + o_d[None, :], mask=((o_t >= 3) & m_t)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base_b + (o_t - 2)[:, None] * D + o_d[None, :], mask=((o_t >= 2) & m_t)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base_b + (o_t - 1)[:, None] * D + o_d[None, :], mask=((o_t >= 1) & m_t)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + base_b + o_t[:, None] * D + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)

    acc = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
    sig = tl.sigmoid(acc)
    y = acc * sig
    tl.store(y_ptr + base_b + o_t[:, None] * D + o_d[None, :], y.to(y_ptr.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])


@triton.jit(do_not_specialize=["T"])
def _short_conv1d_k4_silu_bwd_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    dx_ptr,
    dw_partial_ptr,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Zero-spill backward for causal depthwise Conv1d (kernel_size=4) + SiLU."""
    i_d, i_t, i_b = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    NT = tl.cdiv(T, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    w0 = tl.load(w_ptr + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)[None, :]
    w1 = tl.load(w_ptr + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)[None, :]
    w2 = tl.load(w_ptr + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)[None, :]
    w3 = tl.load(w_ptr + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)[None, :]

    base_b = i_b * T * D

    xm3 = tl.load(x_ptr + base_b + (o_t - 3)[:, None] * D + o_d[None, :], mask=((o_t >= 3) & m_t)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    xm2 = tl.load(x_ptr + base_b + (o_t - 2)[:, None] * D + o_d[None, :], mask=((o_t >= 2) & m_t)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    xm1 = tl.load(x_ptr + base_b + (o_t - 1)[:, None] * D + o_d[None, :], mask=((o_t >= 1) & m_t)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + base_b + o_t[:, None] * D + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    xp1 = tl.load(x_ptr + base_b + (o_t + 1)[:, None] * D + o_d[None, :], mask=((o_t + 1) < T)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    xp2 = tl.load(x_ptr + base_b + (o_t + 2)[:, None] * D + o_d[None, :], mask=((o_t + 2) < T)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    xp3 = tl.load(x_ptr + base_b + (o_t + 3)[:, None] * D + o_d[None, :], mask=((o_t + 3) < T)[:, None] & m_d[None, :], other=0.0).to(tl.float32)

    dy0 = tl.load(dy_ptr + base_b + o_t[:, None] * D + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    dy1 = tl.load(dy_ptr + base_b + (o_t + 1)[:, None] * D + o_d[None, :], mask=((o_t + 1) < T)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    dy2 = tl.load(dy_ptr + base_b + (o_t + 2)[:, None] * D + o_d[None, :], mask=((o_t + 2) < T)[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    dy3 = tl.load(dy_ptr + base_b + (o_t + 3)[:, None] * D + o_d[None, :], mask=((o_t + 3) < T)[:, None] & m_d[None, :], other=0.0).to(tl.float32)

    acc0 = xm3 * w0 + xm2 * w1 + xm1 * w2 + x0 * w3
    sig0 = tl.sigmoid(acc0)
    g0 = tl.where(m_t[:, None] & m_d[None, :], dy0 * sig0 * (1.0 + acc0 * (1.0 - sig0)), 0.0)

    acc1 = xm2 * w0 + xm1 * w1 + x0 * w2 + xp1 * w3
    sig1 = tl.sigmoid(acc1)
    g1 = tl.where(((o_t + 1) < T)[:, None] & m_d[None, :], dy1 * sig1 * (1.0 + acc1 * (1.0 - sig1)), 0.0)

    acc2 = xm1 * w0 + x0 * w1 + xp1 * w2 + xp2 * w3
    sig2 = tl.sigmoid(acc2)
    g2 = tl.where(((o_t + 2) < T)[:, None] & m_d[None, :], dy2 * sig2 * (1.0 + acc2 * (1.0 - sig2)), 0.0)

    acc3 = x0 * w0 + xp1 * w1 + xp2 * w2 + xp3 * w3
    sig3 = tl.sigmoid(acc3)
    g3 = tl.where(((o_t + 3) < T)[:, None] & m_d[None, :], dy3 * sig3 * (1.0 + acc3 * (1.0 - sig3)), 0.0)

    dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0
    tl.store(dx_ptr + base_b + o_t[:, None] * D + o_d[None, :], dx.to(dx_ptr.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])

    dw0 = tl.sum(g0 * xm3, axis=0)
    dw1 = tl.sum(g0 * xm2, axis=0)
    dw2 = tl.sum(g0 * xm1, axis=0)
    dw3 = tl.sum(g0 * x0, axis=0)
    i_bg = i_b * NT + i_t
    p_dw = dw_partial_ptr + i_bg * (D * 4) + o_d * 4
    tl.store(p_dw + 0, dw0, mask=m_d)
    tl.store(p_dw + 1, dw1, mask=m_d)
    tl.store(p_dw + 2, dw2, mask=m_d)
    tl.store(p_dw + 3, dw3, mask=m_d)


class _FusedShortConv1dSiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        BT, BD = 64, 64
        w_flat = weight.squeeze(1).contiguous()
        y = torch.empty_like(x)
        _short_conv1d_k4_silu_fwd_kernel[(triton.cdiv(D, BD), triton.cdiv(T, BT), B)](
            x, w_flat, y, T, D=D, BT=BT, BD=BD, num_warps=4, num_stages=1
        )
        ctx.save_for_backward(x, w_flat)
        ctx.weight_shape = weight.shape
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w_flat = ctx.saved_tensors
        B, T, D = x.shape
        BT, BD = 32, 32
        NT = triton.cdiv(T, BT)
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dw_partial = torch.empty(B * NT, D, 4, device=x.device, dtype=torch.float32)
        _short_conv1d_k4_silu_bwd_kernel[(triton.cdiv(D, BD), NT, B)](
            x, w_flat, dy, dx, dw_partial, T, D=D, BT=BT, BD=BD, num_warps=8, num_stages=1
        )
        dw = dw_partial.sum(dim=0).to(w_flat.dtype).view(ctx.weight_shape)
        return dx, dw


def fused_short_conv1d_silu_rsi(x: torch.Tensor, conv_mod: nn.Module) -> torch.Tensor:
    return _FusedShortConv1dSiLUFn.apply(x.contiguous(), conv_mod.weight)


@triton.jit(do_not_specialize=["N_ROWS"])
def _gsa2_slot_norm_decay_gates_fwd_kernel(
    w_in_ptr,
    gv_raw_ptr,
    b1_raw_ptr,
    A1_log_ptr,
    dt1_bias_ptr,
    w_out_ptr,
    w_inv_denom_ptr,
    gv_out_ptr,
    b1_out_ptr,
    w_scale,
    w_l2norm_eps,
    oja_lower_bound,
    N_ROWS,
    H: tl.constexpr,
    M: tl.constexpr,
    BM: tl.constexpr,
):
    """Superfused GSA-2 Level-1 slot prep: exact FP32 w / (w.norm(p=2) + eps), bounded Oja decay gv, and b1 = sigmoid(b1_raw)."""
    i_r = tl.program_id(0).to(tl.int64)
    if i_r >= N_ROWS:
        return
    i_h = i_r % H
    o_m = tl.arange(0, BM)
    m_m = o_m < M
    base = i_r * M + o_m

    b_w_in = tl.load(w_in_ptr + base, mask=m_m, other=0.0).to(tl.float32) * w_scale
    b_norm = tl.sqrt(tl.sum(b_w_in * b_w_in, axis=0))
    b_inv_denom = 1.0 / (b_norm + w_l2norm_eps)
    b_w_out = b_w_in * b_inv_denom
    tl.store(w_out_ptr + base, b_w_out, mask=m_m)
    tl.store(w_inv_denom_ptr + i_r * 2 + 0, b_norm)
    tl.store(w_inv_denom_ptr + i_r * 2 + 1, b_inv_denom)

    b_gv_raw = tl.load(gv_raw_ptr + base, mask=m_m, other=0.0).to(tl.float32)
    b_A1 = tl.exp(tl.load(A1_log_ptr + i_h).to(tl.float32))
    b_dt1 = tl.load(dt1_bias_ptr + i_h * M + o_m, mask=m_m, other=0.0).to(tl.float32)
    b_sig_gv = tl.sigmoid(b_A1 * (b_gv_raw + b_dt1))
    b_gv = -oja_lower_bound * b_sig_gv
    tl.store(gv_out_ptr + base, b_gv, mask=m_m)

    b_b1_raw = tl.load(b1_raw_ptr + base, mask=m_m, other=0.0).to(tl.float32)
    b_b1 = tl.sigmoid(b_b1_raw)
    tl.store(b1_out_ptr + base, b_b1.to(b1_out_ptr.dtype.element_ty), mask=m_m)


@triton.jit(do_not_specialize=["N_ROWS"])
def _gsa2_slot_norm_decay_gates_bwd_kernel(
    w_in_ptr,
    gv_raw_ptr,
    b1_raw_ptr,
    A1_log_ptr,
    dt1_bias_ptr,
    w_inv_denom_ptr,
    dw_out_ptr,
    dgv_out_ptr,
    db1_out_ptr,
    dw_in_ptr,
    dgv_raw_ptr,
    db1_raw_ptr,
    dA1_partial_ptr,
    ddt1_partial_ptr,
    w_scale,
    oja_lower_bound,
    N_ROWS,
    H: tl.constexpr,
    M: tl.constexpr,
    BM: tl.constexpr,
):
    """Exact backward for superfused GSA-2 Level-1 slot prep (preserving exact w / (||w||_2 + eps) Jacobian)."""
    i_r = tl.program_id(0).to(tl.int64)
    if i_r >= N_ROWS:
        return
    i_h = i_r % H
    o_m = tl.arange(0, BM)
    m_m = o_m < M
    base = i_r * M + o_m

    b_w_in = tl.load(w_in_ptr + base, mask=m_m, other=0.0).to(tl.float32) * w_scale
    b_norm = tl.load(w_inv_denom_ptr + i_r * 2 + 0).to(tl.float32)
    b_inv_denom = tl.load(w_inv_denom_ptr + i_r * 2 + 1).to(tl.float32)
    b_dw = tl.load(dw_out_ptr + base, mask=m_m, other=0.0).to(tl.float32)
    b_dot = tl.sum(b_dw * b_w_in, axis=0)
    b_safe_norm = tl.where(b_norm > 0.0, b_norm, 1.0)
    b_dw_in = (b_dw * b_inv_denom - b_w_in * (b_dot * b_inv_denom * b_inv_denom / b_safe_norm)) * w_scale
    tl.store(dw_in_ptr + base, b_dw_in.to(dw_in_ptr.dtype.element_ty), mask=m_m)

    b_gv_raw = tl.load(gv_raw_ptr + base, mask=m_m, other=0.0).to(tl.float32)
    b_A1 = tl.exp(tl.load(A1_log_ptr + i_h).to(tl.float32))
    b_dt1 = tl.load(dt1_bias_ptr + i_h * M + o_m, mask=m_m, other=0.0).to(tl.float32)
    b_x1 = b_gv_raw + b_dt1
    b_sig = tl.sigmoid(b_A1 * b_x1)
    b_dgv = tl.load(dgv_out_ptr + base, mask=m_m, other=0.0).to(tl.float32)
    b_dz = -oja_lower_bound * b_dgv * b_sig * (1.0 - b_sig)
    b_dgv_raw = b_dz * b_A1
    tl.store(dgv_raw_ptr + base, b_dgv_raw.to(dgv_raw_ptr.dtype.element_ty), mask=m_m)
    tl.store(ddt1_partial_ptr + base, b_dgv_raw, mask=m_m)
    b_dA1 = tl.sum(b_dz * b_x1, axis=0) * b_A1
    tl.store(dA1_partial_ptr + i_r, b_dA1)

    b_b1_raw = tl.load(b1_raw_ptr + base, mask=m_m, other=0.0).to(tl.float32)
    b_sig_b1 = tl.sigmoid(b_b1_raw)
    b_db1 = tl.load(db1_out_ptr + base, mask=m_m, other=0.0).to(tl.float32)
    b_db1_raw = b_db1 * b_sig_b1 * (1.0 - b_sig_b1)
    tl.store(db1_raw_ptr + base, b_db1_raw.to(db1_raw_ptr.dtype.element_ty), mask=m_m)


class _GSA2SlotPrepFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w_in: torch.Tensor,
        gv_raw: torch.Tensor,
        b1_raw: torch.Tensor,
        A1_log: torch.Tensor,
        dt1_bias: torch.Tensor,
        w_scale: float,
        w_l2norm_eps: float,
        oja_lower_bound: float,
    ):
        B, T, H, M = w_in.shape
        BM = triton.next_power_of_2(M)
        n_rows = B * T * H
        w_out = torch.empty(B, T, H, M, device=w_in.device, dtype=torch.float32)
        w_inv_denom = torch.empty(n_rows, 2, device=w_in.device, dtype=torch.float32)
        gv_out = torch.empty(B, T, H, M, device=w_in.device, dtype=torch.float32)
        b1_out = torch.empty_like(b1_raw)
        _gsa2_slot_norm_decay_gates_fwd_kernel[(n_rows,)](
            w_in, gv_raw, b1_raw, A1_log, dt1_bias, w_out, w_inv_denom, gv_out, b1_out,
            w_scale, w_l2norm_eps, oja_lower_bound, n_rows, H=H, M=M, BM=BM,
            num_warps=2, num_stages=1,
        )
        ctx.save_for_backward(w_in, gv_raw, b1_raw, A1_log, dt1_bias, w_inv_denom)
        ctx.w_scale = w_scale
        ctx.oja_lower_bound = oja_lower_bound
        return w_out, gv_out, b1_out

    @staticmethod
    def backward(ctx, dw_out: torch.Tensor, dgv_out: torch.Tensor, db1_out: torch.Tensor):
        w_in, gv_raw, b1_raw, A1_log, dt1_bias, w_inv_denom = ctx.saved_tensors
        B, T, H, M = w_in.shape
        BM = triton.next_power_of_2(M)
        n_rows = B * T * H
        dw_in = torch.empty_like(w_in)
        dgv_raw = torch.empty_like(gv_raw)
        db1_raw = torch.empty_like(b1_raw)
        dA1_partial = torch.empty(B * T, H, device=w_in.device, dtype=torch.float32)
        ddt1_partial = torch.empty(B * T, H * M, device=w_in.device, dtype=torch.float32)
        _gsa2_slot_norm_decay_gates_bwd_kernel[(n_rows,)](
            w_in, gv_raw, b1_raw, A1_log, dt1_bias, w_inv_denom,
            dw_out.contiguous(), dgv_out.contiguous(), db1_out.contiguous(),
            dw_in, dgv_raw, db1_raw, dA1_partial, ddt1_partial,
            ctx.w_scale, ctx.oja_lower_bound, n_rows, H=H, M=M, BM=BM,
            num_warps=2, num_stages=1,
        )
        dA1_log = dA1_partial.sum(dim=0).to(A1_log.dtype)
        ddt1_bias = ddt1_partial.sum(dim=0).to(dt1_bias.dtype)
        return dw_in, dgv_raw, db1_raw, dA1_log, ddt1_bias, None, None, None


@triton.jit(do_not_specialize=["N_ROWS"])
def _gdn2_mamba_gate_b_fwd_kernel(
    g_raw_ptr,
    b_raw_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_out_ptr,
    b_out_ptr,
    N_ROWS,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    """Superfused GDN-2 gate kernel: g = -exp(A_log) * softplus(g_raw + dt_bias) and b = sigmoid(b_raw)."""
    i_r = tl.program_id(0).to(tl.int64)
    if i_r >= N_ROWS:
        return
    i_h = i_r % H
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    base = i_r * K + o_k

    b_g_raw = tl.load(g_raw_ptr + base, mask=m_k, other=0.0).to(tl.float32)
    b_A = tl.exp(tl.load(A_log_ptr + i_h).to(tl.float32))
    b_dt = tl.load(dt_bias_ptr + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_x = b_g_raw + b_dt
    b_sp = tl.where(b_x > 20.0, b_x, tl.log(1.0 + tl.exp(b_x)))
    b_g = -b_A * b_sp
    tl.store(g_out_ptr + base, b_g, mask=m_k)

    b_b_raw = tl.load(b_raw_ptr + base, mask=m_k, other=0.0).to(tl.float32)
    b_b = tl.sigmoid(b_b_raw)
    tl.store(b_out_ptr + base, b_b.to(b_out_ptr.dtype.element_ty), mask=m_k)


@triton.jit(do_not_specialize=["N_ROWS"])
def _gdn2_mamba_gate_b_bwd_kernel(
    g_raw_ptr,
    b_raw_ptr,
    A_log_ptr,
    dt_bias_ptr,
    dg_out_ptr,
    db_out_ptr,
    dg_raw_ptr,
    db_raw_ptr,
    dA_partial_ptr,
    ddt_partial_ptr,
    N_ROWS,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    """Exact backward for superfused GDN-2 Mamba gate + b sigmoid."""
    i_r = tl.program_id(0).to(tl.int64)
    if i_r >= N_ROWS:
        return
    i_h = i_r % H
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    base = i_r * K + o_k

    b_g_raw = tl.load(g_raw_ptr + base, mask=m_k, other=0.0).to(tl.float32)
    b_A = tl.exp(tl.load(A_log_ptr + i_h).to(tl.float32))
    b_dt = tl.load(dt_bias_ptr + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_x = b_g_raw + b_dt
    b_sp = tl.where(b_x > 20.0, b_x, tl.log(1.0 + tl.exp(b_x)))
    b_sig_x = tl.sigmoid(b_x)
    b_dg = tl.load(dg_out_ptr + base, mask=m_k, other=0.0).to(tl.float32)
    b_dg_raw = -b_A * b_dg * b_sig_x
    tl.store(dg_raw_ptr + base, b_dg_raw.to(dg_raw_ptr.dtype.element_ty), mask=m_k)
    tl.store(ddt_partial_ptr + base, b_dg_raw, mask=m_k)
    b_dA = tl.sum(-b_A * b_sp * b_dg, axis=0)
    tl.store(dA_partial_ptr + i_r, b_dA)

    b_b_raw = tl.load(b_raw_ptr + base, mask=m_k, other=0.0).to(tl.float32)
    b_sig_b = tl.sigmoid(b_b_raw)
    b_db = tl.load(db_out_ptr + base, mask=m_k, other=0.0).to(tl.float32)
    b_db_raw = b_db * b_sig_b * (1.0 - b_sig_b)
    tl.store(db_raw_ptr + base, b_db_raw.to(db_raw_ptr.dtype.element_ty), mask=m_k)


class _GDN2GateBFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g_raw: torch.Tensor, b_raw: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
        B, T, H, K = g_raw.shape
        BK = triton.next_power_of_2(K)
        n_rows = B * T * H
        g_out = torch.empty(B, T, H, K, device=g_raw.device, dtype=torch.float32)
        b_out = torch.empty_like(b_raw)
        _gdn2_mamba_gate_b_fwd_kernel[(n_rows,)](
            g_raw, b_raw, A_log, dt_bias, g_out, b_out, n_rows, H=H, K=K, BK=BK,
            num_warps=2, num_stages=1,
        )
        ctx.save_for_backward(g_raw, b_raw, A_log, dt_bias)
        return g_out, b_out

    @staticmethod
    def backward(ctx, dg_out: torch.Tensor, db_out: torch.Tensor):
        g_raw, b_raw, A_log, dt_bias = ctx.saved_tensors
        B, T, H, K = g_raw.shape
        BK = triton.next_power_of_2(K)
        n_rows = B * T * H
        dg_raw = torch.empty_like(g_raw)
        db_raw = torch.empty_like(b_raw)
        dA_partial = torch.empty(B * T, H, device=g_raw.device, dtype=torch.float32)
        ddt_partial = torch.empty(B * T, H * K, device=g_raw.device, dtype=torch.float32)
        _gdn2_mamba_gate_b_bwd_kernel[(n_rows,)](
            g_raw, b_raw, A_log, dt_bias, dg_out.contiguous(), db_out.contiguous(),
            dg_raw, db_raw, dA_partial, ddt_partial, n_rows, H=H, K=K, BK=BK,
            num_warps=2, num_stages=1,
        )
        return dg_raw, db_raw, dA_partial.sum(dim=0).to(A_log.dtype), ddt_partial.sum(dim=0).to(dt_bias.dtype)


# ======================================================================================
# 6. 5-Stage Progressive Layer Wrappers: GDN2BlockRSI and GSA2BlockRSI (Stages A0..A4)
# ======================================================================================

class GDN2BlockRSI(nn.Module):
    """Wraps an existing fla.layers.gdn2.GatedDeltaNet2 module across progressive RSI stages A0..A4."""

    def __init__(self, base_layer: nn.Module, stage: str = "A4"):
        super().__init__()
        assert stage in ("A0", "A1", "A2", "A3", "A4")
        self.base = base_layer
        self.stage = stage

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.stage in ("A0", "A1"):
            out, _, _ = self.base(hidden_states)
            return out

        B, T, _ = hidden_states.shape
        H = self.base.num_heads
        HV = self.base.num_v_heads
        K = self.base.head_k_dim
        V = self.base.head_v_dim

        q_raw = self.base.q_proj(hidden_states)
        k_raw = self.base.k_proj(hidden_states)
        v_raw = self.base.v_proj(hidden_states)

        if self.base.use_short_conv:
            if self.stage in ("A3", "A4"):
                q = fused_short_conv1d_silu_rsi(q_raw, self.base.q_conv1d)
                k = fused_short_conv1d_silu_rsi(k_raw, self.base.k_conv1d)
                v = fused_short_conv1d_silu_rsi(v_raw, self.base.v_conv1d)
            else:
                q, _ = self.base.q_conv1d(x=q_raw, cache=None, output_final_state=False)
                k, _ = self.base.k_conv1d(x=k_raw, cache=None, output_final_state=False)
                v, _ = self.base.v_conv1d(x=v_raw, cache=None, output_final_state=False)
        else:
            q, k, v = F.silu(q_raw), F.silu(k_raw), F.silu(v_raw)

        q = q.view(B, T, H, K)
        k = k.view(B, T, H, K)
        v = v.view(B, T, HV, V)

        f_raw = self.base.f_proj(hidden_states).view(B, T, H, K)
        b_raw = self.base.b_proj(hidden_states).view(B, T, H, K)
        w = torch.sigmoid(self.base.w_proj(hidden_states).view(B, T, HV, V))

        if self.stage == "A4":
            g, b = _GDN2GateBFn.apply(f_raw, b_raw, self.base.A_log, self.base.dt_bias)
        else:
            A = -self.base.A_log.float().exp().view(1, 1, H, 1)
            dt_b = self.base.dt_bias.float().view(1, 1, H, K)
            g = A * F.softplus(f_raw.float() + dt_b)
            b = torch.sigmoid(b_raw)

        if HV > H:
            q = q.repeat_interleave(HV // H, dim=2)
            k = k.repeat_interleave(HV // H, dim=2)
            g = g.repeat_interleave(HV // H, dim=2)
            b = b.repeat_interleave(HV // H, dim=2)

        o, _ = chunk_gdn2_rsi(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            use_qk_l2norm_in_kernel=getattr(self.base, "use_qk_l2norm_in_kernel", True),
        )
        g_out = self.base.g_proj(hidden_states).view(B, T, HV, V)
        o = self.base.o_norm(o, g_out)
        return self.base.o_proj(o.view(B, T, HV * V))


class GSA2BlockRSI(nn.Module):
    """Wraps an existing fla.layers.gsa2.GatedSlotAttention2 module across progressive RSI stages A0..A4."""

    def __init__(self, base_layer: nn.Module, stage: str = "A4"):
        super().__init__()
        assert stage in ("A0", "A1", "A2", "A3", "A4")
        self.base = base_layer
        self.stage = stage

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.stage == "A0":
            out, _, _ = self.base(hidden_states)
            return out

        B, T, _ = hidden_states.shape
        H = self.base.num_heads
        HV = self.base.num_v_heads
        K = self.base.head_k_dim
        V = self.base.head_v_dim
        M = self.base.num_slots
        D2 = self.base.level2_dim

        q_raw = self.base.q_proj(hidden_states)
        k_raw = self.base.k_proj(hidden_states)
        v_raw = self.base.v_proj(hidden_states)
        w_raw = self.base.w_up(self.base.w_down(hidden_states))

        if self.base.use_short_conv:
            if self.stage in ("A3", "A4"):
                q = fused_short_conv1d_silu_rsi(q_raw, self.base.q_conv1d)
                k = fused_short_conv1d_silu_rsi(k_raw, self.base.k_conv1d)
                v = fused_short_conv1d_silu_rsi(v_raw, self.base.v_conv1d)
            else:
                q, _ = self.base.q_conv1d(x=q_raw, cache=None, output_final_state=False)
                k, _ = self.base.k_conv1d(x=k_raw, cache=None, output_final_state=False)
                v, _ = self.base.v_conv1d(x=v_raw, cache=None, output_final_state=False)
        else:
            q, k, v = F.silu(q_raw), F.silu(k_raw), F.silu(v_raw)

        if self.base.use_w_conv:
            if self.stage in ("A3", "A4"):
                w_conv = fused_short_conv1d_silu_rsi(w_raw, self.base.w_conv1d)
            else:
                w_conv, _ = self.base.w_conv1d(x=w_raw, cache=None, output_final_state=False)
        else:
            w_conv = w_raw

        gv_raw = self.base.oja_gv_proj(hidden_states).view(B, T, H, M)
        b1_raw = self.base.oja_b_proj(hidden_states).view(B, T, H, M)
        c1 = torch.sigmoid(self.base.oja_c_proj(hidden_states).view(B, T, H, K))
        g2_raw = self.base.gdn_g_proj(hidden_states).view(B, T, HV, D2)
        b2_raw = self.base.gdn_b_proj(hidden_states).view(B, T, HV, D2)
        w2 = torch.sigmoid(self.base.gdn_w_proj(hidden_states).view(B, T, HV, V))

        q = q.view(B, T, H, K)
        k = k.view(B, T, H, K)
        v = v.view(B, T, HV, V)
        w_conv = w_conv.view(B, T, H, M)

        if self.stage == "A4" and self.base.use_w_l2norm and self.base.w_clip is None:
            w, gv, b1 = _GSA2SlotPrepFn.apply(
                w_conv,
                gv_raw,
                b1_raw,
                self.base.A1_log,
                self.base.dt1_bias,
                float(self.base.w_scale),
                float(self.base.w_l2norm_eps),
                float(self.base.oja_lower_bound),
            )
        else:
            w = w_conv * self.base.w_scale if self.base.w_scale != 1.0 else w_conv
            if self.base.use_w_l2norm:
                w = w.float()
                w = w / (w.norm(p=2, dim=-1, keepdim=True) + self.base.w_l2norm_eps)
            if self.base.w_clip is not None:
                w = w.clamp(min=-self.base.w_clip, max=self.base.w_clip)
            gv = self.base._compute_oja_decay(gv_raw)
            b1 = torch.sigmoid(b1_raw)

        if self.stage == "A1":
            from fla.ops.gated_oja_rule2 import chunk_gated_oja_rule2
            from fla.ops.gdn2 import chunk_gdn2
            o1, _ = chunk_gated_oja_rule2(
                q=q,
                k=k,
                v=w,
                gv=gv,
                b=b1,
                c=c1,
                scale=self.base.oja_scale,
                initial_state=None,
                output_final_state=False,
                use_q_l2norm=self.base.use_oja_q_l2norm,
                use_k_l2norm=self.base.use_oja_k_l2norm,
            )
            q2 = self.base._apply_oja_out_act(o1)
            k2 = w
            if self.base.gdn_q_proj is not None:
                q2 = self.base.gdn_q_proj(q2)
                k2 = self.base.gdn_k_proj(k2.to(dtype=self.base.gdn_k_proj.weight.dtype))
            k2 = k2.to(dtype=q2.dtype)
            if HV > H:
                q2 = q2.repeat_interleave(HV // H, dim=2)
                k2 = k2.repeat_interleave(HV // H, dim=2)
            o, _ = chunk_gdn2(
                q=q2,
                k=k2,
                v=v,
                g=g2_raw,
                b=torch.sigmoid(b2_raw),
                w=w2,
                scale=self.base.gdn_scale,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=self.base.use_gdn_qk_l2norm_in_kernel,
                use_gate_in_kernel=True,
                A_log=self.base.A2_log,
                dt_bias=self.base.dt2_bias,
            )
        else:
            o1, _ = chunk_gated_oja_rule2_rsi(
                q=q,
                k=k,
                v=w,
                gv=gv,
                b=b1,
                c=c1,
                scale=self.base.oja_scale,
                initial_state=None,
                output_final_state=False,
                use_q_l2norm=self.base.use_oja_q_l2norm,
                use_k_l2norm=self.base.use_oja_k_l2norm,
            )
            q2 = self.base._apply_oja_out_act(o1)
            k2 = w
            if self.base.gdn_q_proj is not None:
                q2 = self.base.gdn_q_proj(q2)
                k2 = self.base.gdn_k_proj(k2.to(dtype=self.base.gdn_k_proj.weight.dtype))
            k2 = k2.to(dtype=q2.dtype)
            if HV > H:
                q2 = q2.repeat_interleave(HV // H, dim=2)
                k2 = k2.repeat_interleave(HV // H, dim=2)

            if self.stage == "A4":
                g2, b2 = _GDN2GateBFn.apply(g2_raw, b2_raw, self.base.A2_log, self.base.dt2_bias)
            else:
                A2 = -self.base.A2_log.float().exp().view(1, 1, HV, 1)
                dt2 = self.base.dt2_bias.float().view(1, 1, HV, D2)
                g2 = A2 * F.softplus(g2_raw.float() + dt2)
                b2 = torch.sigmoid(b2_raw)

            o, _ = chunk_gdn2_rsi(
                q=q2,
                k=k2,
                v=v,
                g=g2,
                b=b2,
                w=w2,
                scale=self.base.gdn_scale,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=self.base.use_gdn_qk_l2norm_in_kernel,
            )

        g_out = self.base.g_proj(hidden_states).view(B, T, HV, V)
        o = self.base.o_norm(o, g_out)
        return self.base.o_proj(o.view(B, T, HV * V))


# ======================================================================================
# 7. 8-GPU NVSwitch Context Parallelism: Zero-Copy Ulysses GSA2-CP=8 vs Dual-State Ring
# ======================================================================================

class _UlyssesAllToAllSeq2Head(torch.autograd.Function):
    """Zero-copy B=1 sequence-to-head all_to_all_single collective over 8x A100 NVSwitch."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        B, T_local, H, D = x.shape
        H_local = H // world_size
        ctx.group = group
        ctx.shape_in = (B, T_local, H, D)
        send_buf = x.view(B, T_local, world_size, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        if B == 1:
            return recv_buf.view(1, world_size * T_local, H_local, D)
        return recv_buf.permute(1, 0, 2, 3, 4).contiguous().view(B, world_size * T_local, H_local, D)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        B, T_local, H, D = ctx.shape_in
        world_size = dist.get_world_size(group)
        H_local = H // world_size
        if B == 1:
            send_buf = grad_out.contiguous().view(world_size, 1, T_local, H_local, D)
        else:
            send_buf = grad_out.view(B, world_size, T_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        grad_in = recv_buf.permute(1, 2, 0, 3, 4).contiguous().view(B, T_local, H, D)
        return grad_in, None


class _UlyssesAllToAllHead2Seq(torch.autograd.Function):
    """Zero-copy B=1 head-to-sequence all_to_all_single collective over 8x A100 NVSwitch."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        B, T_total, H_local, D = x.shape
        T_local = T_total // world_size
        ctx.group = group
        ctx.shape_in = (B, T_total, H_local, D)
        if B == 1:
            send_buf = x.contiguous().view(world_size, 1, T_local, H_local, D)
        else:
            send_buf = x.view(B, world_size, T_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 2, 0, 3, 4).contiguous().view(B, T_local, world_size * H_local, D)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        B, T_total, H_local, D = ctx.shape_in
        world_size = dist.get_world_size(group)
        T_local = T_total // world_size
        send_buf = grad_out.view(B, T_local, world_size, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        if B == 1:
            return recv_buf.view(1, T_total, H_local, D), None
        return recv_buf.permute(1, 0, 2, 3, 4).contiguous().view(B, T_total, H_local, D), None


def gsa2_cp8_ulysses_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_slot: torch.Tensor,
    gv: torch.Tensor,
    b1: torch.Tensor,
    c1: torch.Tensor,
    g2: torch.Tensor,
    b2: torch.Tensor,
    w2: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    K = q.shape[-1]
    M = w_slot.shape[-1]
    D2 = g2.shape[-1]
    V = v.shape[-1]

    bf16_pack = torch.cat([q, k, v, b1, c1, b2, w2], dim=-1)
    fp32_pack = torch.cat([w_slot, gv, g2], dim=-1)

    bf16_full = _UlyssesAllToAllSeq2Head.apply(bf16_pack, group)
    fp32_full = _UlyssesAllToAllSeq2Head.apply(fp32_pack, group)

    q_f, k_f, v_f, b1_f, c1_f, b2_f, w2_f = torch.split(bf16_full, [K, K, V, M, K, D2, V], dim=-1)
    w_slot_f, gv_f, g2_f = torch.split(fp32_full, [M, M, D2], dim=-1)

    o1_f, _ = chunk_gated_oja_rule2_rsi(
        q=q_f, k=k_f, v=w_slot_f, gv=gv_f, b=b1_f, c=c1_f, use_q_l2norm=True, use_k_l2norm=True
    )
    q2_f = F.silu(o1_f)
    k2_f = w_slot_f.to(q2_f.dtype)
    o2_f, _ = chunk_gdn2_rsi(
        q=q2_f, k=k2_f, v=v_f, g=g2_f, b=b2_f, w=w2_f, use_qk_l2norm_in_kernel=True
    )
    return _UlyssesAllToAllHead2Seq.apply(o2_f, group)


class _GSA2RingCP8Function(torch.autograd.Function):
    """8-GPU Sequential Dual-State Ring Baseline propagating S^(1) in R^{K x M} and S^(2) in R^{D2 x V}."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w_slot: torch.Tensor,
        gv: torch.Tensor,
        b1: torch.Tensor,
        c1: torch.Tensor,
        g2: torch.Tensor,
        b2: torch.Tensor,
        w2: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        B, T_local, H, K = q.shape
        M = w_slot.shape[-1]
        D2 = g2.shape[-1]
        V = v.shape[-1]

        s1_init_list = [None] * world_size
        s2_init_list = [None] * world_size
        o_local = None

        for step in range(world_size):
            if rank == step:
                s1_in = s1_init_list[step]
                s2_in = s2_init_list[step]
                o1, s1_out = chunk_gated_oja_rule2_rsi(
                    q=q, k=k, v=w_slot, gv=gv, b=b1, c=c1,
                    initial_state=s1_in, output_final_state=True,
                    use_q_l2norm=True, use_k_l2norm=True,
                )
                q2 = F.silu(o1)
                k2 = w_slot.to(q2.dtype)
                o2, s2_out = chunk_gdn2_rsi(
                    q=q2, k=k2, v=v, g=g2, b=b2, w=w2,
                    initial_state=s2_in, output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                o_local = o2
                if step + 1 < world_size:
                    dist.send(s1_out.contiguous(), dst=step + 1, group=group)
                    dist.send(s2_out.contiguous(), dst=step + 1, group=group)
            elif rank == step + 1:
                s1_recv = torch.empty(B, H, K, M, device=q.device, dtype=torch.float32)
                s2_recv = torch.empty(B, H, D2, V, device=q.device, dtype=torch.float32)
                dist.recv(s1_recv, src=step, group=group)
                dist.recv(s2_recv, src=step, group=group)
                s1_init_list[step + 1] = s1_recv
                s2_init_list[step + 1] = s2_recv
            dist.barrier(group=group)

        ctx.save_for_backward(
            q, k, v, w_slot, gv, b1, c1, g2, b2, w2,
            s1_init_list[rank] if s1_init_list[rank] is not None else torch.empty(0, device=q.device),
            s2_init_list[rank] if s2_init_list[rank] is not None else torch.empty(0, device=q.device),
        )
        ctx.group = group
        ctx.has_init = rank > 0
        return o_local

    @staticmethod
    def backward(ctx, do_local: torch.Tensor):
        q, k, v, w_slot, gv, b1, c1, g2, b2, w2, s1_saved, s2_saved = ctx.saved_tensors
        group = ctx.group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        B, T_local, H, K = q.shape
        M = w_slot.shape[-1]
        D2 = g2.shape[-1]
        V = v.shape[-1]

        ds1_next = None
        ds2_next = None
        grads = None

        for step in range(world_size - 1, -1, -1):
            if rank == step:
                with torch.enable_grad():
                    q_ = q.detach().requires_grad_(True)
                    k_ = k.detach().requires_grad_(True)
                    v_ = v.detach().requires_grad_(True)
                    w_slot_ = w_slot.detach().requires_grad_(True)
                    gv_ = gv.detach().requires_grad_(True)
                    b1_ = b1.detach().requires_grad_(True)
                    c1_ = c1.detach().requires_grad_(True)
                    g2_ = g2.detach().requires_grad_(True)
                    b2_ = b2.detach().requires_grad_(True)
                    w2_ = w2.detach().requires_grad_(True)
                    s1_in = s1_saved.detach().requires_grad_(True) if ctx.has_init else None
                    s2_in = s2_saved.detach().requires_grad_(True) if ctx.has_init else None

                    o1, s1_out = chunk_gated_oja_rule2_rsi(
                        q=q_, k=k_, v=w_slot_, gv=gv_, b=b1_, c=c1_,
                        initial_state=s1_in, output_final_state=True,
                        use_q_l2norm=True, use_k_l2norm=True,
                    )
                    q2 = F.silu(o1)
                    k2 = w_slot_.to(q2.dtype)
                    o2, s2_out = chunk_gdn2_rsi(
                        q=q2, k=k2, v=v_, g=g2_, b=b2_, w=w2_,
                        initial_state=s2_in, output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    outputs = [o2]
                    grad_outputs = [do_local]
                    if ds1_next is not None:
                        outputs.append(s1_out)
                        grad_outputs.append(ds1_next)
                    if ds2_next is not None:
                        outputs.append(s2_out)
                        grad_outputs.append(ds2_next)
                    torch.autograd.backward(outputs, grad_outputs)

                grads = (
                    q_.grad, k_.grad, v_.grad, w_slot_.grad, gv_.grad,
                    b1_.grad, c1_.grad, g2_.grad, b2_.grad, w2_.grad, None
                )
                if step > 0:
                    dist.send(s1_in.grad.contiguous(), dst=step - 1, group=group)
                    dist.send(s2_in.grad.contiguous(), dst=step - 1, group=group)
            elif rank == step - 1:
                ds1_next = torch.empty(B, H, K, M, device=q.device, dtype=torch.float32)
                ds2_next = torch.empty(B, H, D2, V, device=q.device, dtype=torch.float32)
                dist.recv(ds1_next, src=step, group=group)
                dist.recv(ds2_next, src=step, group=group)
            dist.barrier(group=group)

        return grads


def gsa2_cp8_ring_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_slot: torch.Tensor,
    gv: torch.Tensor,
    b1: torch.Tensor,
    c1: torch.Tensor,
    g2: torch.Tensor,
    b2: torch.Tensor,
    w2: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    return _GSA2RingCP8Function.apply(q, k, v, w_slot, gv, b1, c1, g2, b2, w2, group)


# ======================================================================================
# 8. SM80 Cubin Register, Spill & Shared-Memory Audit Helper
# ======================================================================================

RSI_KERNEL_REGISTRY = [
    ("_chunk_cumsum_fwd_kernel", _chunk_cumsum_fwd_kernel),
    ("_chunk_cumsum_rev_kernel", _chunk_cumsum_rev_kernel),
    ("_l2norm_fwd_kernel", _l2norm_fwd_kernel),
    ("_l2norm_bwd_kernel", _l2norm_bwd_kernel),
    ("_merge_16x16_to_64x64_inverse_kernel", _merge_16x16_to_64x64_inverse_kernel),
    ("_gdn2_fwd_sub_chunk_diag_kernel", _gdn2_fwd_sub_chunk_diag_kernel),
    ("_gdn2_fwd_inter_offdiag_kernel", _gdn2_fwd_inter_offdiag_kernel),
    ("_gdn2_wy_fwd_kernel", _gdn2_wy_fwd_kernel),
    ("_gdn2_fwd_h_kernel", _gdn2_fwd_h_kernel),
    ("_gdn2_fwd_o_kernel", _gdn2_fwd_o_kernel),
    ("_gdn2_bwd_dAv_kernel", _gdn2_bwd_dAv_kernel),
    ("_gdn2_bwd_dhu_kernel", _gdn2_bwd_dhu_kernel),
    ("_gdn2_bwd_state_dqkw_pass1_kernel", _gdn2_bwd_state_dqkw_pass1_kernel),
    ("_gdn2_bwd_wy_v_pass2a_kernel", _gdn2_bwd_wy_v_pass2a_kernel),
    ("_gdn2_bwd_wy_k_pass2b_kernel", _gdn2_bwd_wy_k_pass2b_kernel),
    ("_bwd_dA_triple_product_kernel", _bwd_dA_triple_product_kernel),
    ("_gdn2_bwd_intra_kernel", _gdn2_bwd_intra_kernel),
    ("_oja2_kkt_sub_chunk_diag_kernel", _oja2_kkt_sub_chunk_diag_kernel),
    ("_oja2_kkt_offdiag_kernel", _oja2_kkt_offdiag_kernel),
    ("_oja2_wy_fwd_kernel", _oja2_wy_fwd_kernel),
    ("_oja2_fwd_h_kernel", _oja2_fwd_h_kernel),
    ("_oja2_fwd_o_inter_kernel", _oja2_fwd_o_inter_kernel),
    ("_oja2_fwd_o_intra_kernel", _oja2_fwd_o_intra_kernel),
    ("_oja2_bwd_dA_kernel", _oja2_bwd_dA_kernel),
    ("_oja2_bwd_dqk_kernel", _oja2_bwd_dqk_kernel),
    ("_oja2_bwd_dhu_kernel", _oja2_bwd_dhu_kernel),
    ("_oja2_bwd_dvwg_h_kernel", _oja2_bwd_dvwg_h_kernel),
    ("_oja2_bwd_dv_o_kernel", _oja2_bwd_dv_o_kernel),
    ("_oja2_bwd_wy_v_pass1_kernel", _oja2_bwd_wy_v_pass1_kernel),
    ("_oja2_bwd_wy_k_pass2_kernel", _oja2_bwd_wy_k_pass2_kernel),
    ("_oja2_bwd_kkt_kernel", _oja2_bwd_kkt_kernel),
    ("_short_conv1d_k4_silu_fwd_kernel", _short_conv1d_k4_silu_fwd_kernel),
    ("_short_conv1d_k4_silu_bwd_kernel", _short_conv1d_k4_silu_bwd_kernel),
    ("_gsa2_slot_norm_decay_gates_fwd_kernel", _gsa2_slot_norm_decay_gates_fwd_kernel),
    ("_gsa2_slot_norm_decay_gates_bwd_kernel", _gsa2_slot_norm_decay_gates_bwd_kernel),
    ("_gdn2_mamba_gate_b_fwd_kernel", _gdn2_mamba_gate_b_fwd_kernel),
    ("_gdn2_mamba_gate_b_bwd_kernel", _gdn2_mamba_gate_b_bwd_kernel),
]


def _extract_compiled_kernels(jit_fn) -> List[Any]:
    visited = set()
    stack = [jit_fn]
    compiled = []
    while stack:
        cur = stack.pop()
        if id(cur) in visited:
            continue
        visited.add(id(cur))
        for cname in ("cache", "device_caches"):
            cobj = getattr(cur, cname, None)
            if isinstance(cobj, dict):
                for v in cobj.values():
                    if hasattr(v, "n_regs"):
                        compiled.append(v)
                    elif isinstance(v, (dict, tuple, list)):
                        items = v.values() if isinstance(v, dict) else v
                        for vv in items:
                            if hasattr(vv, "n_regs"):
                                compiled.append(vv)
                            elif isinstance(vv, dict):
                                for vvv in vv.values():
                                    if hasattr(vvv, "n_regs"):
                                        compiled.append(vvv)
        for attr in ("fn", "Autotuner", "heuristics"):
            if hasattr(cur, attr):
                stack.append(getattr(cur, attr))
    return compiled


def clear_rsi_kernel_caches() -> None:
    for _, kfn in RSI_KERNEL_REGISTRY:
        for cname in ("cache", "device_caches"):
            cobj = getattr(kfn, cname, None)
            if isinstance(cobj, dict):
                cobj.clear()


def audit_rsi_cubin_kernels() -> Dict[str, Any]:
    per_kernel = {}
    total_specs = 0
    spilling_specs = 0
    max_regs_global = 0
    max_spills_global = 0
    max_smem_global = 0

    for kname, kfn in RSI_KERNEL_REGISTRY:
        compiled = _extract_compiled_kernels(kfn)
        n_specs = len(compiled)
        n_spills = sum(1 for c in compiled if getattr(c, "n_spills", 0) > 0)
        max_regs = max((getattr(c, "n_regs", 0) for c in compiled), default=0)
        max_spills = max((getattr(c, "n_spills", 0) for c in compiled), default=0)
        max_smem = max((getattr(c.metadata, "shared", 0) for c in compiled if hasattr(c, "metadata")), default=0)
        per_kernel[kname] = {
            "specializations": n_specs,
            "spilling_specializations": n_spills,
            "max_regs": int(max_regs),
            "max_spills": int(max_spills),
            "max_shared_bytes": int(max_smem),
        }
        total_specs += n_specs
        spilling_specs += n_spills
        max_regs_global = max(max_regs_global, int(max_regs))
        max_spills_global = max(max_spills_global, int(max_spills))
        max_smem_global = max(max_smem_global, int(max_smem))

    return {
        "total_kernels": len(RSI_KERNEL_REGISTRY),
        "compiled_kernels": sum(1 for v in per_kernel.values() if v["specializations"] > 0),
        "total_specializations": int(total_specs),
        "spilling_specializations": int(spilling_specs),
        "max_regs": int(max_regs_global),
        "max_spills": int(max_spills_global),
        "max_shared_bytes": int(max_smem_global),
        "zero_spill_verified": bool(spilling_specs == 0 and max_regs_global <= 168 and total_specs > 0),
        "per_kernel": per_kernel,
    }

