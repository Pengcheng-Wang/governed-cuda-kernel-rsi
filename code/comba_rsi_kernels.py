"""
Round-29 RSI (E155, v146): Comba — Improving Bilinear RNNs with Closed-loop Control
(arXiv:2506.02475 — Jiaxi Hu, Yongqi Pan, Jusen Du, Disen Lan, Xiaozhe Ren, Yibo Wang,
Xuezhi Cao, Yunsheng Wu, Yu Zhang, Songlin Yang; official repo fla/ops/comba & fla/layers/comba.py)

Key Mathematical & SM80 Hardware Innovations:
1. Fused Cumsum + Strict-Lower PK^T WY Preparation (`_comba_cumsum_pkt_fwd_rsi_kernel`):
   Fuses base-2 chunk-local exclusive/inclusive scalar gate cumsum (`g0, g1`) directly into the
   strict-lower-triangular closed-loop Gram matrix M_{i,j} = 1[i > j] * beta_i * (p_i^T k_j) * 2^{g0_i - g1_j}
   using BK=32 and allow_tf32=False.
2. In-Register Tensor-Core 64x64 Unit-Lower-Triangular Solve (`_comba_solve_tril_64_rsi_kernel`):
   Inverts (I + M)^{-1} via four 16x16 Euler repeated-squaring diagonal inversions
   (I - L)(I + L)(I + L^2)(I + L^4)(I + L^8) and 4-block forward substitution in registers.
3. Zero-Spill Tile-32 WY Forward & Saved-State Recurrence (`_comba_recompute_w_u_rsi_kernel`,
   `_comba_fwd_h_rsi_kernel`, `_comba_fwd_o_rsi_kernel`):
   Tiles BK=32, BV=32 (`num_warps=4, num_stages=1`, `max_regs <= 128`, `0` spills) and saves compact
   `(w, v_new, h)` to eliminate upstream's redundant backward replay of `recompute_w_u_fwd` and `fwd_h`.
4. Euler Homogeneous Degree-1 Closed-Loop WY Adjoint & Single-Pass Combined Gate Reverse Cumsum
   (`_comba_bwd_wy_dA_dv_rsi_kernel` & `_comba_bwd_wy_dkp_dg_rsi_kernel`):
   Uses sum_j dM_{ij} M_{ij} == <dp^{(M)}_i, p_i> and -sum_i dM_{ij} M_{ij} == -<dk^{(M)}_j, k_j>
   to eliminate upstream's 64x64xK recomputation of `b_A`, eliminates the double HBM read/write of `dp`,
   fuses `dk = dk_in + dk^{(M)}`, and computes the combined inclusive+exclusive reverse cumsum
   dg = (rev_cumsum(dg_tot + dg0) - dg0) * RCP_LN2 in registers (eliminating `chunk_local_cumsum`
   and `chunk_comba_cumsum_scalar_bwd`).
5. Order-Correct Fused In-Kernel `L2Norm(q, k) -> p = s_h * k_hat, q_tilde = q_hat - D_h * p`
   (`_comba_qk_l2norm_corr_fwd_rsi_kernel` & `_comba_qk_l2norm_corr_bwd_rsi_kernel`):
   Fixes upstream `Comba.forward`'s inner-decay L2Norm erasure bug (`l2norm(s_h * k) == l2norm(k)`)
   while fusing 6 separate `l2norm_fwd`/`l2norm_bwd` + elementwise passes into 1 forward and 1 backward kernel.
6. Single-Pass Multi-Warp Online Decoding (`_comba_decode_rsi_kernel`), Superfused `CombaBlockRSI` (`A0..A4`),
   and Zero-Copy 8-GPU NVSwitch Ulysses `Comba-CP = 8` (`comba_cp8_ulysses_rsi`).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl

RCP_LN2 = tl.constexpr(1.4426950408889634)


# ==============================================================================
# 1. Exact Differentiable FP64 Reference Oracle (`comba_fp64_oracle`)
# ==============================================================================

def comba_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Exact step-by-step differentiable FP64 reference oracle for Comba (arXiv:2506.02475).
    Supports both MHA (H == HV) and GVA (HV > H), arbitrary sequence length T, and initial_state.
    """
    orig_dtype = q.dtype
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    G_rep = HV // H
    if scale is None:
        scale = K ** -0.5

    q64 = q.to(torch.float64)
    k64 = k.to(torch.float64)
    p64 = p.to(torch.float64)
    v64 = v.to(torch.float64)
    g64 = g.to(torch.float64)
    beta64 = beta.to(torch.float64)

    if use_qk_l2norm:
        q64 = q64 * torch.rsqrt((q64 * q64).sum(dim=-1, keepdim=True) + eps)
        k64 = k64 * torch.rsqrt((k64 * k64).sum(dim=-1, keepdim=True) + eps)
        p64 = p64 * torch.rsqrt((p64 * p64).sum(dim=-1, keepdim=True) + eps)

    if G_rep > 1:
        q64 = q64.repeat_interleave(G_rep, dim=2)
        k64 = k64.repeat_interleave(G_rep, dim=2)
        p64 = p64.repeat_interleave(G_rep, dim=2)

    q64 = q64 * scale
    if initial_state is not None:
        h = initial_state.to(torch.float64).clone()
    else:
        h = torch.zeros(B, HV, K, V, dtype=torch.float64, device=q.device)

    o_list = []
    for t in range(T):
        qt = q64[:, t]          # [B, HV, K]
        kt = k64[:, t]          # [B, HV, K]
        pt = p64[:, t]          # [B, HV, K]
        vt = v64[:, t]          # [B, HV, V]
        gt = g64[:, t]          # [B, HV]
        bt = beta64[:, t]       # [B, HV]

        # Closed-loop pre-decay feedback: v_corr = beta_t * (v_t - h_{t-1}^T p_t)
        v_proj = torch.einsum("bhkv,bhk->bhv", h, pt)
        v_corr = (vt - v_proj) * bt.unsqueeze(-1)
        h = h * torch.exp(gt).unsqueeze(-1).unsqueeze(-1) + torch.einsum("bhk,bhv->bhkv", kt, v_corr)
        ot = torch.einsum("bhkv,bhk->bhv", h, qt)
        o_list.append(ot)

    o = torch.stack(o_list, dim=1)
    ht = h if output_final_state else None
    return o, ht


# ==============================================================================
# 2. Zero-Spill Round-29 RSI Triton Kernels (`max_regs <= 168`, `0` spills)
# ==============================================================================

@triton.jit
def _comba_qk_l2norm_fwd_rsi_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    N_ROWS,
    K: tl.constexpr,
    BK: tl.constexpr,
    EPS: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    x = tl.load(x_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + o_k, mask=m_k, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / K) + EPS)
    y = x * rstd * w
    tl.store(rstd_ptr + i_row, rstd)
    tl.store(y_ptr + i_row * K + o_k, y.to(y_ptr.dtype.element_ty), mask=m_k)


@triton.jit
def _comba_qk_l2norm_bwd_rsi_kernel(
    x_ptr,
    w_ptr,
    rstd_ptr,
    dy_ptr,
    dx_ptr,
    dw_part_ptr,
    N_ROWS,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    x = tl.load(x_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + o_k, mask=m_k, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + i_row).to(tl.float32)
    dy = tl.load(dy_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    x_hat = x * rstd
    dx_hat = dy * w
    dw = dy * x_hat
    dx = (dx_hat - x_hat * (tl.sum(dx_hat * x_hat, axis=0) * (1.0 / K))) * rstd
    tl.store(dx_ptr + i_row * K + o_k, dx.to(dx_ptr.dtype.element_ty), mask=m_k)
    tl.store(dw_part_ptr + i_row * K + o_k, dw, mask=m_k)


@triton.jit
def _comba_qkp_l2norm_fwd_rsi_kernel(
    q_ptr,
    k_ptr,
    p_ptr,
    q_out_ptr,
    k_out_ptr,
    p_out_ptr,
    q_rstd_ptr,
    k_rstd_ptr,
    p_rstd_ptr,
    N_ROWS,
    K: tl.constexpr,
    BK: tl.constexpr,
    EPS: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    off = i_row * K + o_k
    q = tl.load(q_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    rq = tl.rsqrt(tl.sum(q * q, axis=0) + EPS)
    rk = tl.rsqrt(tl.sum(k * k, axis=0) + EPS)
    rp = tl.rsqrt(tl.sum(p * p, axis=0) + EPS)
    tl.store(q_rstd_ptr + i_row, rq)
    tl.store(k_rstd_ptr + i_row, rk)
    tl.store(p_rstd_ptr + i_row, rp)
    tl.store(q_out_ptr + off, (q * rq).to(q_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(k_out_ptr + off, (k * rk).to(k_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(p_out_ptr + off, (p * rp).to(p_out_ptr.dtype.element_ty), mask=m_k)


@triton.jit
def _comba_qkp_l2norm_bwd_rsi_kernel(
    q_ptr,
    k_ptr,
    p_ptr,
    q_rstd_ptr,
    k_rstd_ptr,
    p_rstd_ptr,
    dq_in_ptr,
    dk_in_ptr,
    dp_in_ptr,
    dq_out_ptr,
    dk_out_ptr,
    dp_out_ptr,
    N_ROWS,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    off = i_row * K + o_k
    q = tl.load(q_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    dq = tl.load(dq_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    dk = tl.load(dk_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    dp = tl.load(dp_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    rq = tl.load(q_rstd_ptr + i_row).to(tl.float32)
    rk = tl.load(k_rstd_ptr + i_row).to(tl.float32)
    rp = tl.load(p_rstd_ptr + i_row).to(tl.float32)
    dq_out = (dq - q * tl.sum(dq * q, axis=0)) * rq
    dk_out = (dk - k * tl.sum(dk * k, axis=0)) * rk
    dp_out = (dp - p * tl.sum(dp * p, axis=0)) * rp
    tl.store(dq_out_ptr + off, dq_out.to(dq_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(dk_out_ptr + off, dk_out.to(dk_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(dp_out_ptr + off, dp_out.to(dp_out_ptr.dtype.element_ty), mask=m_k)


@triton.jit
def _comba_cumsum_pkt_fwd_rsi_kernel(
    k_ptr,
    p_ptr,
    beta_ptr,
    g_ptr,
    g0_out_ptr,
    g1_out_ptr,
    M_out_ptr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    MUTATION_MODE: tl.constexpr,
):
    """
    Fuses chunk-local base-2 scalar cumsum (g0, g1) with strict-lower-triangular
    closed-loop Gram matrix M_{i, j} = 1[i > j] * beta_i * (p_i^T k_j) * 2^{g0_i - g1_j}.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_g_raw = tl.load(g_ptr + bos_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32) * RCP_LN2
    b_g1 = tl.cumsum(b_g_raw, axis=0)
    if MUTATION_MODE == 1:
        # M1 mutation: corrupt pre-decay shift g0 by using inclusive g1 instead of exclusive g1 - g_raw
        b_g0 = b_g1
    else:
        b_g0 = b_g1 - b_g_raw

    tl.store(g0_out_ptr + bos_hv + o_t * HV, b_g0, mask=m_t)
    tl.store(g1_out_ptr + bos_hv + o_t * HV, b_g1, mask=m_t)

    b_beta = tl.load(beta_ptr + bos_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
    b_M = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in tl.range(0, tl.cdiv(K, BK), loop_unroll_factor=1):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k_ptr + bos_h * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_p = p_ptr + bos_h * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_p = tl.load(p_p, mask=m_k, other=0.0)
        b_pb = (b_p.to(tl.float32) * b_beta[:, None]).to(b_k.dtype)
        b_M = tl.dot(b_pb, tl.trans(b_k), b_M, allow_tf32=False)

    m_tril = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_decay = tl.where(m_tril, tl.math.exp2(b_g0[:, None] - b_g1[None, :]), 0.0)
    b_M = b_M * b_decay

    o_c = tl.arange(0, BT)
    p_M = M_out_ptr + (bos_hv * BT) + o_t[:, None] * (HV * BT) + o_c[None, :]
    tl.store(p_M, b_M, mask=m_t[:, None] & (o_c[None, :] < BT))


@triton.jit
def _euler_inv16(L):
    """
    Exact 4-step Euler repeated squaring for 16x16 strict-lower-triangular L:
    (I + L)^{-1} = (I - L)(I + L^2)(I + L^4)(I + L^8) since L^16 == 0.
    """
    r = tl.arange(0, 16)
    eye = (r[:, None] == r[None, :]).to(tl.float32)
    L2 = tl.dot(L, L, allow_tf32=False)
    L4 = tl.dot(L2, L2, allow_tf32=False)
    L8 = tl.dot(L4, L4, allow_tf32=False)
    P1 = eye - L
    P2 = tl.dot(P1, eye + L2, allow_tf32=False)
    P4 = tl.dot(P2, eye + L4, allow_tf32=False)
    P8 = tl.dot(P4, eye + L8, allow_tf32=False)
    return P8


@triton.jit
def _comba_solve_tril_16x16_diag_rsi_kernel(
    M_ptr,
    A_ptr,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
):
    """
    Pass 1 of 64x64 triangular solve: inverts the four 16x16 diagonal blocks in parallel
    using exact Euler repeated squaring in registers (max_regs <= 72, 0 spills).
    """
    i_blk = tl.program_id(0).to(tl.int64)  # 0..3
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV

    bos_hv = (i_b * T) * HV + i_hv
    r = tl.arange(0, 16)
    row_idx = i_t * BT + i_blk * 16 + r
    col_idx = i_blk * 16 + r
    m_row = row_idx < T
    p_M = M_ptr + bos_hv * BT + row_idx[:, None] * (HV * BT) + col_idx[None, :]
    L = tl.load(p_M, mask=m_row[:, None], other=0.0).to(tl.float32)
    L = tl.where(r[:, None] > r[None, :], L, 0.0)
    A_diag = _euler_inv16(L)
    p_A = A_ptr + bos_hv * BT + row_idx[:, None] * (HV * BT) + col_idx[None, :]
    tl.store(p_A, A_diag.to(p_A.dtype.element_ty), mask=m_row[:, None])


@triton.jit
def _load_block16(ptr, bos_hv, i_t, bi, bj, T, HV: tl.constexpr, BT: tl.constexpr):
    r = tl.arange(0, 16)
    rows = i_t * BT + bi * 16 + r
    cols = bj * 16 + r
    m = rows < T
    p = ptr + bos_hv * BT + rows[:, None] * (HV * BT) + cols[None, :]
    return tl.load(p, mask=m[:, None], other=0.0).to(tl.float32)


@triton.jit
def _store_block16(ptr, bos_hv, i_t, bi, bj, val, T, HV: tl.constexpr, BT: tl.constexpr):
    r = tl.arange(0, 16)
    rows = i_t * BT + bi * 16 + r
    cols = bj * 16 + r
    m = rows < T
    p = ptr + bos_hv * BT + rows[:, None] * (HV * BT) + cols[None, :]
    tl.store(p, val.to(p.dtype.element_ty), mask=m[:, None])


@triton.jit
def _comba_solve_tril_64_offdiag_rsi_kernel(
    M_ptr,
    A_ptr,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
):
    """
    Pass 2 of 64x64 triangular solve: combines the four inverted 16x16 diagonal blocks
    into the full 64x64 inverse (I + M)^{-1} using 16x16 block forward substitution
    (max_regs <= 96, 0 spills).
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV

    bos_hv = (i_b * T) * HV + i_hv

    # Zero upper blocks
    zero = tl.zeros([16, 16], dtype=tl.float32)
    _store_block16(A_ptr, bos_hv, i_t, 0, 1, zero, T, HV, BT)
    _store_block16(A_ptr, bos_hv, i_t, 0, 2, zero, T, HV, BT)
    _store_block16(A_ptr, bos_hv, i_t, 0, 3, zero, T, HV, BT)
    _store_block16(A_ptr, bos_hv, i_t, 1, 2, zero, T, HV, BT)
    _store_block16(A_ptr, bos_hv, i_t, 1, 3, zero, T, HV, BT)
    _store_block16(A_ptr, bos_hv, i_t, 2, 3, zero, T, HV, BT)

    A00 = _load_block16(A_ptr, bos_hv, i_t, 0, 0, T, HV, BT)
    A11 = _load_block16(A_ptr, bos_hv, i_t, 1, 1, T, HV, BT)
    M10 = _load_block16(M_ptr, bos_hv, i_t, 1, 0, T, HV, BT)
    A10 = -tl.dot(A11, tl.dot(M10, A00, allow_tf32=False), allow_tf32=False)
    _store_block16(A_ptr, bos_hv, i_t, 1, 0, A10, T, HV, BT)

    A22 = _load_block16(A_ptr, bos_hv, i_t, 2, 2, T, HV, BT)
    M20 = _load_block16(M_ptr, bos_hv, i_t, 2, 0, T, HV, BT)
    M21 = _load_block16(M_ptr, bos_hv, i_t, 2, 1, T, HV, BT)
    A21 = -tl.dot(A22, tl.dot(M21, A11, allow_tf32=False), allow_tf32=False)
    A20 = -tl.dot(A22, tl.dot(M20, A00, allow_tf32=False) + tl.dot(M21, A10, allow_tf32=False), allow_tf32=False)
    _store_block16(A_ptr, bos_hv, i_t, 2, 1, A21, T, HV, BT)
    _store_block16(A_ptr, bos_hv, i_t, 2, 0, A20, T, HV, BT)

    A33 = _load_block16(A_ptr, bos_hv, i_t, 3, 3, T, HV, BT)
    M30 = _load_block16(M_ptr, bos_hv, i_t, 3, 0, T, HV, BT)
    M31 = _load_block16(M_ptr, bos_hv, i_t, 3, 1, T, HV, BT)
    M32 = _load_block16(M_ptr, bos_hv, i_t, 3, 2, T, HV, BT)
    A32 = -tl.dot(A33, tl.dot(M32, A22, allow_tf32=False), allow_tf32=False)
    A31 = -tl.dot(A33, tl.dot(M31, A11, allow_tf32=False) + tl.dot(M32, A21, allow_tf32=False), allow_tf32=False)
    A30 = -tl.dot(
        A33,
        tl.dot(M30, A00, allow_tf32=False)
        + tl.dot(M31, A10, allow_tf32=False)
        + tl.dot(M32, A20, allow_tf32=False),
        allow_tf32=False,
    )
    _store_block16(A_ptr, bos_hv, i_t, 3, 2, A32, T, HV, BT)
    _store_block16(A_ptr, bos_hv, i_t, 3, 1, A31, T, HV, BT)
    _store_block16(A_ptr, bos_hv, i_t, 3, 0, A30, T, HV, BT)


@triton.jit
def _comba_recompute_w_u_rsi_kernel(
    p_ptr,
    v_ptr,
    beta_ptr,
    g0_ptr,
    A_ptr,
    w_ptr,
    u_ptr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Zero-spill column-slab 32-row strip WY forward transform computing u = A @ (beta * v) and w = A @ (beta * 2^{g0} * p)
    sharded across (K//BK, NT*2, B*HV) with [32, 32] tiles so p and v are read without inner K/V loops (max_regs <= 80, 0 spills).
    """
    i_kv = tl.program_id(0).to(tl.int64)
    i_t_sub = tl.program_id(1).to(tl.int64)
    i_t = i_t_sub // 2
    i_sub = i_t_sub % 2
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_r = i_t * BT + i_sub * 32 + tl.arange(0, 32)
    m_r = o_r < T

    o_v = i_kv * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k = i_kv * BK + tl.arange(0, BK)
    m_k = o_k < K

    b_u = tl.zeros([32, BV], dtype=tl.float32)
    for j_sub in tl.range(0, 2, loop_unroll_factor=1):
        o_c = j_sub * 32 + tl.arange(0, 32)
        o_tc = i_t * BT + o_c
        m_c = o_tc < T
        p_A = A_ptr + bos_hv * BT + o_r[:, None] * (HV * BT) + o_c[None, :]
        b_A = tl.load(p_A, mask=m_r[:, None] & m_c[None, :], other=0.0)
        b_beta = tl.load(beta_ptr + bos_hv + o_tc * HV, mask=m_c, other=0.0).to(tl.float32)
        p_v = v_ptr + bos_hv * V + o_tc[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_c[:, None] & m_v[None, :], other=0.0)
        b_vb = (b_v.to(tl.float32) * b_beta[:, None]).to(b_A.dtype)
        b_u = tl.dot(b_A, b_vb, b_u, allow_tf32=False)
    p_u = u_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
    tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_r[:, None] & m_v[None, :])

    b_w = tl.zeros([32, BK], dtype=tl.float32)
    for j_sub in tl.range(0, 2, loop_unroll_factor=1):
        o_c = j_sub * 32 + tl.arange(0, 32)
        o_tc = i_t * BT + o_c
        m_c = o_tc < T
        p_A = A_ptr + bos_hv * BT + o_r[:, None] * (HV * BT) + o_c[None, :]
        b_A = tl.load(p_A, mask=m_r[:, None] & m_c[None, :], other=0.0)
        b_beta = tl.load(beta_ptr + bos_hv + o_tc * HV, mask=m_c, other=0.0).to(tl.float32)
        b_g0 = tl.load(g0_ptr + bos_hv + o_tc * HV, mask=m_c, other=0.0).to(tl.float32)
        b_bg0 = b_beta * tl.math.exp2(b_g0)
        p_p = p_ptr + bos_h * K + o_tc[:, None] * (H * K) + o_k[None, :]
        b_p = tl.load(p_p, mask=m_c[:, None] & m_k[None, :], other=0.0)
        b_pb = (b_p.to(tl.float32) * b_bg0[:, None]).to(b_A.dtype)
        b_w = tl.dot(b_A, b_pb, b_w, allow_tf32=False)
    p_w = w_ptr + bos_hv * K + o_r[:, None] * (HV * K) + o_k[None, :]
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_r[:, None] & m_k[None, :])


@triton.jit
def _comba_fwd_h_rsi_kernel(
    k_ptr,
    w_ptr,
    u_ptr,
    g_ptr,
    h_out_ptr,
    v_new_out_ptr,
    h0_ptr,
    ht_out_ptr,
    T,
    NT: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """
    Zero-spill inter-chunk state recurrence tiled with BV=16 and full K in BK=32 slabs.
    Processes each chunk in four 16-token sub-slices (j_sub=0..3) so all tl.dot tiles are [16, 32] x [32, 16]
    or [32, 16] x [16, 16] (max_regs <= 120, 0 spills across FP32 and BF16, K in {64, 128}).
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    r32 = tl.arange(0, 32)

    b_h0 = tl.zeros([32, BV], dtype=tl.float32)
    b_h1 = tl.zeros([32, BV], dtype=tl.float32)
    b_h2 = tl.zeros([32, BV], dtype=tl.float32)
    b_h3 = tl.zeros([32, BV], dtype=tl.float32)

    if USE_INIT_STATE:
        base_h0 = h0_ptr + i_bh * (K * V) + o_v[None, :]
        b_h0 = tl.load(base_h0 + (0 + r32)[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)
        b_h1 = tl.load(base_h0 + (32 + r32)[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            b_h2 = tl.load(base_h0 + (64 + r32)[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)
            b_h3 = tl.load(base_h0 + (96 + r32)[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)

    for i_t in range(NT):
        base_h = h_out_ptr + (i_bh * NT + i_t) * (K * V) + o_v[None, :]
        tl.store(base_h + (0 + r32)[:, None] * V, b_h0.to(h_out_ptr.dtype.element_ty), mask=m_v[None, :])
        tl.store(base_h + (32 + r32)[:, None] * V, b_h1.to(h_out_ptr.dtype.element_ty), mask=m_v[None, :])
        if K > 64:
            tl.store(base_h + (64 + r32)[:, None] * V, b_h2.to(h_out_ptr.dtype.element_ty), mask=m_v[None, :])
            tl.store(base_h + (96 + r32)[:, None] * V, b_h3.to(h_out_ptr.dtype.element_ty), mask=m_v[None, :])

        # Pass 1 over j_sub=0..3: compute and store v_new = u - w @ h using chunk-start h
        for j_sub in tl.range(0, 4, loop_unroll_factor=1):
            o_t = i_t * BT + j_sub * 16 + tl.arange(0, 16)
            m_t = o_t < T
            m_tv = m_t[:, None] & m_v[None, :]

            p_u = u_ptr + bos_hv * V + o_t[:, None] * (HV * V) + o_v[None, :]
            b_v_new = tl.load(p_u, mask=m_tv, other=0.0).to(tl.float32)

            p_w0 = w_ptr + bos_hv * K + o_t[:, None] * (HV * K) + (0 + r32)[None, :]
            p_w1 = w_ptr + bos_hv * K + o_t[:, None] * (HV * K) + (32 + r32)[None, :]
            b_w0 = -tl.load(p_w0, mask=m_t[:, None], other=0.0)
            b_w1 = -tl.load(p_w1, mask=m_t[:, None], other=0.0)
            b_v_new = tl.dot(b_w0, b_h0.to(b_w0.dtype), b_v_new, allow_tf32=False)
            b_v_new = tl.dot(b_w1, b_h1.to(b_w1.dtype), b_v_new, allow_tf32=False)
            if K > 64:
                p_w2 = w_ptr + bos_hv * K + o_t[:, None] * (HV * K) + (64 + r32)[None, :]
                p_w3 = w_ptr + bos_hv * K + o_t[:, None] * (HV * K) + (96 + r32)[None, :]
                b_w2 = -tl.load(p_w2, mask=m_t[:, None], other=0.0)
                b_w3 = -tl.load(p_w3, mask=m_t[:, None], other=0.0)
                b_v_new = tl.dot(b_w2, b_h2.to(b_w2.dtype), b_v_new, allow_tf32=False)
                b_v_new = tl.dot(b_w3, b_h3.to(b_w3.dtype), b_v_new, allow_tf32=False)

            p_vn = v_new_out_ptr + bos_hv * V + o_t[:, None] * (HV * V) + o_v[None, :]
            tl.store(p_vn, b_v_new.to(p_vn.dtype.element_ty), mask=m_tv)

        # Pass 2 over j_sub=0..3: update b_h0..b_h3 in-place without duplicate b_nh0..b_nh3 registers
        last_idx = tl.minimum((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g_ptr + bos_hv + last_idx * HV).to(tl.float32)
        g_last_exp = tl.math.exp2(b_g_last)

        b_h0 = b_h0 * g_last_exp
        b_h1 = b_h1 * g_last_exp
        if K > 64:
            b_h2 = b_h2 * g_last_exp
            b_h3 = b_h3 * g_last_exp

        for j_sub in tl.range(0, 4, loop_unroll_factor=1):
            o_t = i_t * BT + j_sub * 16 + tl.arange(0, 16)
            m_t = o_t < T
            m_tv = m_t[:, None] & m_v[None, :]

            p_vn = v_new_out_ptr + bos_hv * V + o_t[:, None] * (HV * V) + o_v[None, :]
            b_vn_dt = tl.load(p_vn, mask=m_tv, other=0.0)
            b_g = tl.load(g_ptr + bos_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
            b_decay = tl.where(m_t, tl.math.exp2(b_g_last - b_g), 0.0)

            p_k0 = k_ptr + bos_h * K + o_t[:, None] * (H * K) + (0 + r32)[None, :]
            p_k1 = k_ptr + bos_h * K + o_t[:, None] * (H * K) + (32 + r32)[None, :]
            b_k0 = (tl.load(p_k0, mask=m_t[:, None], other=0.0).to(tl.float32) * b_decay[:, None]).to(b_vn_dt.dtype)
            b_k1 = (tl.load(p_k1, mask=m_t[:, None], other=0.0).to(tl.float32) * b_decay[:, None]).to(b_vn_dt.dtype)
            b_h0 = tl.dot(tl.trans(b_k0), b_vn_dt, b_h0, allow_tf32=False)
            b_h1 = tl.dot(tl.trans(b_k1), b_vn_dt, b_h1, allow_tf32=False)
            if K > 64:
                p_k2 = k_ptr + bos_h * K + o_t[:, None] * (H * K) + (64 + r32)[None, :]
                p_k3 = k_ptr + bos_h * K + o_t[:, None] * (H * K) + (96 + r32)[None, :]
                b_k2 = (tl.load(p_k2, mask=m_t[:, None], other=0.0).to(tl.float32) * b_decay[:, None]).to(b_vn_dt.dtype)
                b_k3 = (tl.load(p_k3, mask=m_t[:, None], other=0.0).to(tl.float32) * b_decay[:, None]).to(b_vn_dt.dtype)
                b_h2 = tl.dot(tl.trans(b_k2), b_vn_dt, b_h2, allow_tf32=False)
                b_h3 = tl.dot(tl.trans(b_k3), b_vn_dt, b_h3, allow_tf32=False)

    if STORE_FINAL_STATE:
        base_ht = ht_out_ptr + i_bh * (K * V) + o_v[None, :]
        tl.store(base_ht + (0 + r32)[:, None] * V, b_h0.to(ht_out_ptr.dtype.element_ty), mask=m_v[None, :])
        tl.store(base_ht + (32 + r32)[:, None] * V, b_h1.to(ht_out_ptr.dtype.element_ty), mask=m_v[None, :])
        if K > 64:
            tl.store(base_ht + (64 + r32)[:, None] * V, b_h2.to(ht_out_ptr.dtype.element_ty), mask=m_v[None, :])
            tl.store(base_ht + (96 + r32)[:, None] * V, b_h3.to(ht_out_ptr.dtype.element_ty), mask=m_v[None, :])


@triton.jit
def _comba_fwd_o_rsi_kernel(
    q_ptr,
    k_ptr,
    v_new_ptr,
    h_ptr,
    g_ptr,
    o_out_ptr,
    scale,
    T,
    NT: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
):
    """
    Zero-spill chunk output readout using BS-row query strip tiling (BS=32 for BF16, BS=16 for FP32)
    and BK=32, BV=32 (max_regs <= 84, 0 spills).
    """
    NS: tl.constexpr = BT // BS
    i_v = tl.program_id(0).to(tl.int64)
    i_t_sub = tl.program_id(1).to(tl.int64)
    i_t = i_t_sub // NS
    i_sub = i_t_sub % NS
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_r = i_t * BT + i_sub * BS + tl.arange(0, BS)
    o_v = i_v * BV + tl.arange(0, BV)
    m_r = o_r < T
    m_v = o_v < V
    m_rv = m_r[:, None] & m_v[None, :]

    b_g_r = tl.load(g_ptr + bos_hv + o_r * HV, mask=m_r, other=0.0).to(tl.float32)
    b_o = tl.zeros([BS, BV], dtype=tl.float32)

    for i_k in tl.range(0, tl.cdiv(K, BK), loop_unroll_factor=1):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q_ptr + bos_h * K + o_r[:, None] * (H * K) + o_k[None, :]
        p_h = h_ptr + (i_bh * NT + i_t) * (K * V) + o_k[:, None] * V + o_v[None, :]
        b_q = tl.load(p_q, mask=m_r[:, None] & m_k[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_o = tl.dot(b_q, b_h, b_o, allow_tf32=False)

    b_o = b_o * tl.math.exp2(b_g_r)[:, None]

    for j_sub in tl.range(0, i_sub + 1, loop_unroll_factor=1):
        o_c = i_t * BT + j_sub * BS + tl.arange(0, BS)
        m_c = o_c < T
        b_A = tl.zeros([BS, BS], dtype=tl.float32)
        for i_k in tl.range(0, tl.cdiv(K, BK), loop_unroll_factor=1):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p_q = q_ptr + bos_h * K + o_r[:, None] * (H * K) + o_k[None, :]
            p_k = k_ptr + bos_h * K + o_c[:, None] * (H * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_r[:, None] & m_k[None, :], other=0.0)
            b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0)
            b_A = tl.dot(b_q, tl.trans(b_k), b_A, allow_tf32=False)
        b_g_c = tl.load(g_ptr + bos_hv + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
        m_tril = (o_r[:, None] >= o_c[None, :]) & (m_r[:, None] & m_c[None, :])
        b_A = tl.where(m_tril, b_A * tl.math.exp2(b_g_r[:, None] - b_g_c[None, :]), 0.0)
        p_v = v_new_ptr + bos_hv * V + o_c[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_c[:, None] & m_v[None, :], other=0.0)
        b_o = tl.dot(b_A.to(b_v.dtype), b_v, b_o, allow_tf32=False)

    b_o = b_o * scale
    p_o = o_out_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_rv)


@triton.jit
def _comba_bwd_dv_local_rsi_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    do_ptr,
    dv_out_ptr,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Zero-spill local intra-chunk dv gradient tiled with BK=32, BV=32 (max_regs <= 120, 0 spills).
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_g = tl.load(g_ptr + bos_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in tl.range(0, tl.cdiv(K, BK), loop_unroll_factor=1):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        p_k = k_ptr + bos_h * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_q = q_ptr + bos_h * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_A = tl.dot(b_k, tl.trans(b_q), b_A, allow_tf32=False)

    m_triu = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = tl.where(m_triu, b_A * tl.math.exp2(b_g[None, :] - b_g[:, None]) * scale, 0.0)

    for i_v in tl.range(0, tl.cdiv(V, BV), loop_unroll_factor=1):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_do = do_ptr + bos_hv * V + o_t[:, None] * (HV * V) + o_v[None, :]
        p_dv = dv_out_ptr + bos_hv * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)


@triton.jit
def _comba_bwd_dhu_rsi_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    g_ptr,
    do_ptr,
    dv_in_ptr,
    dh_out_ptr,
    dh0_out_ptr,
    dv_out_ptr,
    dht_ptr,
    scale,
    T,
    NT: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_DHT: tl.constexpr,
    STORE_DH0: tl.constexpr,
):
    """
    Zero-spill reverse inter-chunk state adjoint tiled with BV=16 and full K in BK=32 slabs.
    Processes each chunk in four 16-token sub-slices (j_sub=0..3) so all tl.dot tiles are [16, 32] x [32, 16]
    or [32, 16] x [16, 16] (max_regs <= 120, 0 spills across FP32 and BF16, K in {64, 128}).
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    r32 = tl.arange(0, 32)

    b_dh0 = tl.zeros([32, BV], dtype=tl.float32)
    b_dh1 = tl.zeros([32, BV], dtype=tl.float32)
    b_dh2 = tl.zeros([32, BV], dtype=tl.float32)
    b_dh3 = tl.zeros([32, BV], dtype=tl.float32)

    if USE_DHT:
        base_dht = dht_ptr + i_bh * (K * V) + o_v[None, :]
        b_dh0 = tl.load(base_dht + (0 + r32)[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)
        b_dh1 = tl.load(base_dht + (32 + r32)[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            b_dh2 = tl.load(base_dht + (64 + r32)[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)
            b_dh3 = tl.load(base_dht + (96 + r32)[:, None] * V, mask=m_v[None, :], other=0.0).to(tl.float32)

    for step in range(NT):
        i_t = (NT - 1 - step).to(tl.int64)
        base_dh = dh_out_ptr + (i_bh * NT + i_t) * (K * V) + o_v[None, :]
        tl.store(base_dh + (0 + r32)[:, None] * V, b_dh0.to(dh_out_ptr.dtype.element_ty), mask=m_v[None, :])
        tl.store(base_dh + (32 + r32)[:, None] * V, b_dh1.to(dh_out_ptr.dtype.element_ty), mask=m_v[None, :])
        if K > 64:
            tl.store(base_dh + (64 + r32)[:, None] * V, b_dh2.to(dh_out_ptr.dtype.element_ty), mask=m_v[None, :])
            tl.store(base_dh + (96 + r32)[:, None] * V, b_dh3.to(dh_out_ptr.dtype.element_ty), mask=m_v[None, :])

        last_idx = tl.minimum((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g_ptr + bos_hv + last_idx * HV).to(tl.float32)
        g_last_exp = tl.math.exp2(b_g_last)

        # Pass 1 over j_sub=0..3: compute and store dv_out using end-of-chunk b_dh0..b_dh3
        for j_sub in tl.range(0, 4, loop_unroll_factor=1):
            o_t = i_t * BT + j_sub * 16 + tl.arange(0, 16)
            m_t = o_t < T
            m_tv = m_t[:, None] & m_v[None, :]

            b_g = tl.load(g_ptr + bos_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
            b_decay = tl.where(m_t, tl.math.exp2(b_g_last - b_g), 0.0)

            p_dv_in = dv_in_ptr + bos_hv * V + o_t[:, None] * (HV * V) + o_v[None, :]
            b_dv = tl.load(p_dv_in, mask=m_tv, other=0.0).to(tl.float32)

            p_k0 = k_ptr + bos_h * K + o_t[:, None] * (H * K) + (0 + r32)[None, :]
            p_k1 = k_ptr + bos_h * K + o_t[:, None] * (H * K) + (32 + r32)[None, :]
            b_k0 = (tl.load(p_k0, mask=m_t[:, None], other=0.0).to(tl.float32) * b_decay[:, None]).to(k_ptr.dtype.element_ty)
            b_k1 = (tl.load(p_k1, mask=m_t[:, None], other=0.0).to(tl.float32) * b_decay[:, None]).to(k_ptr.dtype.element_ty)
            b_dv = tl.dot(b_k0, b_dh0.to(b_k0.dtype), b_dv, allow_tf32=False)
            b_dv = tl.dot(b_k1, b_dh1.to(b_k1.dtype), b_dv, allow_tf32=False)
            if K > 64:
                p_k2 = k_ptr + bos_h * K + o_t[:, None] * (H * K) + (64 + r32)[None, :]
                p_k3 = k_ptr + bos_h * K + o_t[:, None] * (H * K) + (96 + r32)[None, :]
                b_k2 = (tl.load(p_k2, mask=m_t[:, None], other=0.0).to(tl.float32) * b_decay[:, None]).to(k_ptr.dtype.element_ty)
                b_k3 = (tl.load(p_k3, mask=m_t[:, None], other=0.0).to(tl.float32) * b_decay[:, None]).to(k_ptr.dtype.element_ty)
                b_dv = tl.dot(b_k2, b_dh2.to(b_k2.dtype), b_dv, allow_tf32=False)
                b_dv = tl.dot(b_k3, b_dh3.to(b_k3.dtype), b_dv, allow_tf32=False)

            p_dv_out = dv_out_ptr + bos_hv * V + o_t[:, None] * (HV * V) + o_v[None, :]
            tl.store(p_dv_out, b_dv.to(p_dv_out.dtype.element_ty), mask=m_tv)

        # Single Pass 2 over p_sub=0..7: p_sub in 0..3 accumulates q^T @ do; p_sub in 4..7 accumulates -w^T @ dv_out
        b_dh0 = b_dh0 * g_last_exp
        b_dh1 = b_dh1 * g_last_exp
        if K > 64:
            b_dh2 = b_dh2 * g_last_exp
            b_dh3 = b_dh3 * g_last_exp

        for p_sub in tl.range(0, 8, loop_unroll_factor=1):
            is_w = p_sub >= 4
            j_sub = tl.where(is_w, p_sub - 4, p_sub)
            o_t = i_t * BT + j_sub * 16 + tl.arange(0, 16)
            m_t = o_t < T
            m_tv = m_t[:, None] & m_v[None, :]

            b_g = tl.load(g_ptr + bos_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
            b_mult = tl.where(is_w, -1.0, tl.where(m_t, tl.math.exp2(b_g) * scale, 0.0))

            p_rhs = tl.where(is_w, dv_out_ptr, do_ptr) + bos_hv * V + o_t[:, None] * (HV * V) + o_v[None, :]
            b_rhs = tl.load(p_rhs, mask=m_tv, other=0.0)

            base_lhs = tl.where(is_w, w_ptr + bos_hv * K + o_t[:, None] * (HV * K), q_ptr + bos_h * K + o_t[:, None] * (H * K))
            b_l0 = (tl.load(base_lhs + (0 + r32)[None, :], mask=m_t[:, None], other=0.0).to(tl.float32) * b_mult[:, None]).to(b_rhs.dtype)
            b_l1 = (tl.load(base_lhs + (32 + r32)[None, :], mask=m_t[:, None], other=0.0).to(tl.float32) * b_mult[:, None]).to(b_rhs.dtype)
            b_dh0 = tl.dot(tl.trans(b_l0), b_rhs, b_dh0, allow_tf32=False)
            b_dh1 = tl.dot(tl.trans(b_l1), b_rhs, b_dh1, allow_tf32=False)
            if K > 64:
                b_l2 = (tl.load(base_lhs + (64 + r32)[None, :], mask=m_t[:, None], other=0.0).to(tl.float32) * b_mult[:, None]).to(b_rhs.dtype)
                b_l3 = (tl.load(base_lhs + (96 + r32)[None, :], mask=m_t[:, None], other=0.0).to(tl.float32) * b_mult[:, None]).to(b_rhs.dtype)
                b_dh2 = tl.dot(tl.trans(b_l2), b_rhs, b_dh2, allow_tf32=False)
                b_dh3 = tl.dot(tl.trans(b_l3), b_rhs, b_dh3, allow_tf32=False)

    if STORE_DH0:
        base_dh0 = dh0_out_ptr + i_bh * (K * V) + o_v[None, :]
        tl.store(base_dh0 + (0 + r32)[:, None] * V, b_dh0.to(dh0_out_ptr.dtype.element_ty), mask=m_v[None, :])
        tl.store(base_dh0 + (32 + r32)[:, None] * V, b_dh1.to(dh0_out_ptr.dtype.element_ty), mask=m_v[None, :])
        if K > 64:
            tl.store(base_dh0 + (64 + r32)[:, None] * V, b_dh2.to(dh0_out_ptr.dtype.element_ty), mask=m_v[None, :])
            tl.store(base_dh0 + (96 + r32)[:, None] * V, b_dh3.to(dh0_out_ptr.dtype.element_ty), mask=m_v[None, :])


@triton.jit
def _comba_bwd_ds_rsi_kernel(
    do_ptr,
    v_new_ptr,
    g_ptr,
    ds_out_ptr,
    scale,
    T,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Zero-spill 32-row strip kernel computing the causal score adjoint
    ds[r, c] = 1[r >= c] * (do[r] @ v_new[c]^T) * 2^{g[r] - g[c]} * scale
    using [32, 32] x [32, 32] tiles (max_regs <= 72, 0 spills).
    """
    i_sub = tl.program_id(0).to(tl.int64)  # 0..1
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV

    bos_hv = (i_b * T) * HV + i_hv
    o_r = i_t * BT + i_sub * 32 + tl.arange(0, 32)
    m_r = o_r < T
    b_g_r = tl.load(g_ptr + bos_hv + o_r * HV, mask=m_r, other=0.0).to(tl.float32)

    for j_sub in tl.range(0, 2, loop_unroll_factor=1):
        o_c_local = j_sub * 32 + tl.arange(0, 32)
        o_c = i_t * BT + o_c_local
        m_c = o_c < T
        b_ds = tl.zeros([32, 32], dtype=tl.float32)
        if j_sub <= i_sub:
            for i_v in tl.range(0, tl.cdiv(V, BV), loop_unroll_factor=1):
                o_v = i_v * BV + tl.arange(0, BV)
                m_v = o_v < V
                p_do = do_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
                p_v = v_new_ptr + bos_hv * V + o_c[:, None] * (HV * V) + o_v[None, :]
                b_do = tl.load(p_do, mask=m_r[:, None] & m_v[None, :], other=0.0)
                b_v = tl.load(p_v, mask=m_c[:, None] & m_v[None, :], other=0.0)
                b_ds = tl.dot(b_do, tl.trans(b_v), b_ds, allow_tf32=False)
            b_g_c = tl.load(g_ptr + bos_hv + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
            m_tril = (o_r[:, None] >= o_c[None, :]) & (m_r[:, None] & m_c[None, :])
            b_ds = tl.where(m_tril, b_ds * tl.math.exp2(b_g_r[:, None] - b_g_c[None, :]) * scale, 0.0)
        p_ds = ds_out_ptr + bos_hv * BT + o_r[:, None] * (HV * BT) + o_c_local[None, :]
        tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), mask=m_r[:, None])


@triton.jit
def _comba_bwd_dqkwg_rsi_kernel(
    q_ptr,
    k_ptr,
    v_new_ptr,
    h_ptr,
    g_ptr,
    do_ptr,
    dh_ptr,
    dv_ptr,
    ds_ptr,
    dq_out_ptr,
    dk_out_ptr,
    dw_out_ptr,
    dg_out_ptr,
    scale,
    T,
    NT: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
):
    """
    Zero-spill BS-row strip backward for dq, dk^{(1)}, dw, and dg^{(1)} (BS=32 for BF16, BS=16 for FP32, BK=32, BV=32)
    with Euler homogeneous identity on S_{rc} eliminating inner b_qk_rc / b_qk_cr tl.dot calls (max_regs <= 148, 0 spills).
    """
    NS: tl.constexpr = BT // BS
    i_k = tl.program_id(0).to(tl.int64)
    i_t_sub = tl.program_id(1).to(tl.int64)
    i_t = i_t_sub // NS
    i_sub = i_t_sub % NS
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_r_local = i_sub * BS + tl.arange(0, BS)
    o_r = i_t * BT + o_r_local
    o_k = i_k * BK + tl.arange(0, BK)
    m_r = o_r < T
    m_k = o_k < K
    m_rk = m_r[:, None] & m_k[None, :]

    last_idx = tl.minimum((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g_ptr + bos_hv + last_idx * HV).to(tl.float32)
    b_g_r = tl.load(g_ptr + bos_hv + o_r * HV, mask=m_r, other=0.0).to(tl.float32)
    b_g_exp = tl.where(m_r, tl.math.exp2(b_g_r), 0.0)
    b_decay = tl.where(m_r, tl.math.exp2(b_g_last - b_g_r), 0.0)

    b_dq = tl.zeros([BS, BK], dtype=tl.float32)
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_dw = tl.zeros([BS, BK], dtype=tl.float32)
    b_dg_h = 0.0

    for i_v in tl.range(0, tl.cdiv(V, BV), loop_unroll_factor=1):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_rv = m_r[:, None] & m_v[None, :]
        m_kv = m_k[:, None] & m_v[None, :]

        p_v = v_new_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
        p_do = do_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
        p_dv = dv_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
        p_h = h_ptr + (i_bh * NT + i_t) * (K * V) + o_k[:, None] * V + o_v[None, :]
        p_dh = dh_ptr + (i_bh * NT + i_t) * (K * V) + o_k[:, None] * V + o_v[None, :]

        b_v = tl.load(p_v, mask=m_rv, other=0.0)
        b_do = tl.load(p_do, mask=m_rv, other=0.0)
        b_dv = tl.load(p_dv, mask=m_rv, other=0.0)
        b_h = tl.load(p_h, mask=m_kv, other=0.0)
        b_dh = tl.load(p_dh, mask=m_kv, other=0.0)

        if i_sub == 0:
            b_dg_h += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32))
        b_dq = tl.dot(b_do, tl.trans(b_h), b_dq, allow_tf32=False)
        b_dk = tl.dot(b_v, tl.trans(b_dh), b_dk, allow_tf32=False)
        b_dw = tl.dot(b_dv, tl.trans(b_h), b_dw, allow_tf32=False)

    b_dw = -b_dw
    b_dq = b_dq * b_g_exp[:, None] * scale
    b_dk = b_dk * b_decay[:, None]

    p_k_r = k_ptr + bos_h * K + o_r[:, None] * (H * K) + o_k[None, :]
    b_dg_last_part = tl.sum(b_dk * tl.load(p_k_r, mask=m_rk, other=0.0).to(tl.float32))
    if i_sub == 0:
        b_dg_last_part += b_dg_h * tl.math.exp2(b_g_last)

    for j_sub in tl.range(0, 2, loop_unroll_factor=1):
        o_c_local = j_sub * 32 + tl.arange(0, 32)
        o_c = i_t * BT + o_c_local
        m_c = o_c < T
        m_ck = m_c[:, None] & m_k[None, :]

        # Row contribution: ds[r, c] ([BS, 32] x [32, 32])
        p_ds_rc = ds_ptr + bos_hv * BT + o_r[:, None] * (HV * BT) + o_c_local[None, :]
        b_ds_rc = tl.load(p_ds_rc, mask=m_r[:, None], other=0.0)
        p_k_c = k_ptr + bos_h * K + o_c[:, None] * (H * K) + o_k[None, :]
        b_k_c = tl.load(p_k_c, mask=m_ck, other=0.0)
        b_dq = tl.dot(b_ds_rc.to(b_k_c.dtype), b_k_c, b_dq, allow_tf32=False)

        # Col contribution: ds[c, r]^T ([BS, 32] x [32, 32])
        p_ds_cr = ds_ptr + bos_hv * BT + o_c[:, None] * (HV * BT) + o_r_local[None, :]
        b_ds_cr = tl.load(p_ds_cr, mask=m_c[:, None], other=0.0)
        p_q_c = q_ptr + bos_h * K + o_c[:, None] * (H * K) + o_k[None, :]
        b_q_c = tl.load(p_q_c, mask=m_ck, other=0.0)
        b_dk = tl.dot(tl.trans(b_ds_cr.to(b_q_c.dtype)), b_q_c, b_dk, allow_tf32=False)

    p_q_r = q_ptr + bos_h * K + o_r[:, None] * (H * K) + o_k[None, :]
    b_q_r = tl.load(p_q_r, mask=m_rk, other=0.0).to(tl.float32)
    b_k_r = tl.load(p_k_r, mask=m_rk, other=0.0).to(tl.float32)
    b_dg = tl.sum(b_dq * b_q_r - b_dk * b_k_r, axis=1)

    p_dq = dq_out_ptr + bos_hv * K + o_r[:, None] * (HV * K) + o_k[None, :]
    p_dk = dk_out_ptr + bos_hv * K + o_r[:, None] * (HV * K) + o_k[None, :]
    p_dw = dw_out_ptr + bos_hv * K + o_r[:, None] * (HV * K) + o_k[None, :]
    p_dg = dg_out_ptr + (i_k * (tl.num_programs(2) // HV) * T * HV) + bos_hv + o_r * HV

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_rk)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_rk)
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), mask=m_rk)
    tl.atomic_add(p_dg, b_dg, mask=m_r)
    if b_dg_last_part != 0.0:
        p_dg_last = dg_out_ptr + (i_k * (tl.num_programs(2) // HV) * T * HV) + bos_hv + last_idx * HV
        tl.atomic_add(p_dg_last, b_dg_last_part)


@triton.jit
def _comba_bwd_wy_dA_dv_rsi_kernel(
    v_ptr,
    p_ptr,
    beta_ptr,
    g0_ptr,
    A_ptr,
    dw_ptr,
    du_ptr,
    dv_out_ptr,
    dA_raw_out_ptr,
    dbeta_v_out_ptr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Pass 1A of Zero-Spill 32-Strip Closed-Loop WY Adjoint (i_sub in 0..1):
    Computes dv[r], dbeta_v[r], and raw dA_raw[r, c] = 1[r > c] * (du[r] @ v_beta[c]^T + dw[r] @ p_beta_g0[c]^T)
    using [32, 32] tiles (max_regs <= 80, 0 spills).
    """
    i_sub = tl.program_id(0).to(tl.int64)  # 0..1
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_r_local = i_sub * 32 + tl.arange(0, 32)
    o_r = i_t * BT + o_r_local
    m_r = o_r < T

    b_beta_r = tl.load(beta_ptr + bos_hv + o_r * HV, mask=m_r, other=0.0).to(tl.float32)
    b_dbeta = tl.zeros([32], dtype=tl.float32)

    # 1. Compute dv[r] = (sum_{j_sub=0..1} A[j_sub, i_sub]^T @ du[j_sub]) * beta[r]
    for i_v in tl.range(0, tl.cdiv(V, BV), loop_unroll_factor=1):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_dv_beta = tl.zeros([32, BV], dtype=tl.float32)
        for j_sub in tl.range(0, 2, loop_unroll_factor=1):
            o_c = i_t * BT + j_sub * 32 + tl.arange(0, 32)
            m_c = o_c < T
            p_AT = A_ptr + bos_hv * BT + o_r_local[:, None] + o_c[None, :] * (HV * BT)
            b_AT = tl.load(p_AT, mask=m_c[None, :], other=0.0)
            p_du = du_ptr + bos_hv * V + o_c[:, None] * (HV * V) + o_v[None, :]
            b_du = tl.load(p_du, mask=m_c[:, None] & m_v[None, :], other=0.0)
            b_dv_beta = tl.dot(b_AT, b_du, b_dv_beta, allow_tf32=False)
        p_v_r = v_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
        b_v_r = tl.load(p_v_r, mask=m_r[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_dv = b_dv_beta * b_beta_r[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v_r, axis=1)
        p_dv = dv_out_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_r[:, None] & m_v[None, :])

    tl.store(dbeta_v_out_ptr + bos_hv + o_r * HV, b_dbeta, mask=m_r)

    # 2. Compute dA_raw[i_sub, j_sub] for j_sub = 0..1
    for j_sub in tl.range(0, 2, loop_unroll_factor=1):
        o_c_local = j_sub * 32 + tl.arange(0, 32)
        o_c = i_t * BT + o_c_local
        m_c = o_c < T
        b_dA = tl.zeros([32, 32], dtype=tl.float32)
        if j_sub <= i_sub:
            b_beta_c = tl.load(beta_ptr + bos_hv + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
            b_g0_c = tl.load(g0_ptr + bos_hv + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
            b_bg0_c = b_beta_c * tl.math.exp2(b_g0_c)

            for i_v in tl.range(0, tl.cdiv(V, BV), loop_unroll_factor=1):
                o_v = i_v * BV + tl.arange(0, BV)
                m_v = o_v < V
                p_du_r = du_ptr + bos_hv * V + o_r[:, None] * (HV * V) + o_v[None, :]
                p_v_c = v_ptr + bos_hv * V + o_c[:, None] * (HV * V) + o_v[None, :]
                b_du_r = tl.load(p_du_r, mask=m_r[:, None] & m_v[None, :], other=0.0)
                b_v_c = tl.load(p_v_c, mask=m_c[:, None] & m_v[None, :], other=0.0)
                b_vb_c = (b_v_c.to(tl.float32) * b_beta_c[:, None]).to(b_du_r.dtype)
                b_dA = tl.dot(b_du_r, tl.trans(b_vb_c), b_dA, allow_tf32=False)

            for i_k in tl.range(0, tl.cdiv(K, BK), loop_unroll_factor=1):
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                p_dw_r = dw_ptr + bos_hv * K + o_r[:, None] * (HV * K) + o_k[None, :]
                p_p_c = p_ptr + bos_h * K + o_c[:, None] * (H * K) + o_k[None, :]
                b_dw_r = tl.load(p_dw_r, mask=m_r[:, None] & m_k[None, :], other=0.0)
                b_p_c = tl.load(p_p_c, mask=m_c[:, None] & m_k[None, :], other=0.0)
                b_pbg0_c = (b_p_c.to(tl.float32) * b_bg0_c[:, None]).to(b_dw_r.dtype)
                b_dA = tl.dot(b_dw_r, tl.trans(b_pbg0_c), b_dA, allow_tf32=False)

            m_tril = (o_r[:, None] > o_c[None, :]) & (m_r[:, None] & m_c[None, :])
            b_dA = tl.where(m_tril, b_dA, 0.0)
        p_dA = dA_raw_out_ptr + bos_hv * BT + o_r[:, None] * (HV * BT) + o_c_local[None, :]
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_r[:, None])


@triton.jit
def _comba_bwd_wy_dM_block_rsi_kernel(
    A_ptr,
    dA_raw_ptr,
    g0_ptr,
    g1_ptr,
    dM_out_ptr,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
):
    """
    Pass 1B of Zero-Spill 16-Strip Closed-Loop WY Adjoint (i_sub in 0..3):
    Computes dM = -StrictTril((A^T @ dA_raw @ A^T) * 2^{g0 - g1^T}) using 16x16 block tiles
    (max_regs <= 64, 0 spills).
    """
    i_sub = tl.program_id(0).to(tl.int64)  # 0..3
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV

    bos_hv = (i_b * T) * HV + i_hv
    o_r_local = i_sub * 16 + tl.arange(0, 16)
    o_r = i_t * BT + o_r_local
    m_r = o_r < T
    b_g0_r = tl.load(g0_ptr + bos_hv + o_r * HV, mask=m_r, other=0.0).to(tl.float32)

    for j_sub in tl.range(0, 4, loop_unroll_factor=1):
        o_c_local = j_sub * 16 + tl.arange(0, 16)
        o_c = i_t * BT + o_c_local
        m_c = o_c < T
        b_dM = tl.zeros([16, 16], dtype=tl.float32)
        if j_sub <= i_sub:
            for m_sub in tl.range(0, 4, loop_unroll_factor=1):
                o_m = i_t * BT + m_sub * 16 + tl.arange(0, 16)
                m_m = o_m < T
                p_AT_left = A_ptr + bos_hv * BT + o_r_local[:, None] + o_m[None, :] * (HV * BT)
                b_AT_left = tl.load(p_AT_left, mask=m_m[None, :], other=0.0)
                b_tmp = tl.zeros([16, 16], dtype=tl.float32)
                for n_sub in tl.range(0, 4, loop_unroll_factor=1):
                    o_n_local = n_sub * 16 + tl.arange(0, 16)
                    p_dA = dA_raw_ptr + bos_hv * BT + o_m[:, None] * (HV * BT) + o_n_local[None, :]
                    p_AT_right = A_ptr + bos_hv * BT + o_n_local[:, None] + o_c[None, :] * (HV * BT)
                    b_dA = tl.load(p_dA, mask=m_m[:, None], other=0.0)
                    b_AT_right = tl.load(p_AT_right, mask=m_c[None, :], other=0.0)
                    b_tmp = tl.dot(b_dA.to(b_AT_right.dtype), b_AT_right, b_tmp, allow_tf32=False)
                b_dM = tl.dot(b_AT_left, b_tmp.to(b_AT_left.dtype), b_dM, allow_tf32=False)

            b_g1_c = tl.load(g1_ptr + bos_hv + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
            m_tril = (o_r[:, None] > o_c[None, :]) & (m_r[:, None] & m_c[None, :])
            b_dM = tl.where(m_tril, -b_dM * tl.math.exp2(b_g0_r[:, None] - b_g1_c[None, :]), 0.0)

        p_dM = dM_out_ptr + bos_hv * BT + o_r[:, None] * (HV * BT) + o_c_local[None, :]
        tl.store(p_dM, b_dM.to(p_dM.dtype.element_ty), mask=m_r[:, None])


@triton.jit
def _comba_bwd_wy_dkp_dg_rsi_kernel(
    k_ptr,
    p_ptr,
    beta_ptr,
    g0_ptr,
    A_ptr,
    dM_ptr,
    dw_ptr,
    dk_in_ptr,
    dbeta_v_ptr,
    dk_out_ptr,
    dp_out_ptr,
    dbeta_out_ptr,
    dg0_out_ptr,
    dg_M_out_ptr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Pass 2A of Zero-Spill 32-Strip Closed-Loop WY Adjoint (i_sub in 0..1):
    Uses the Euler Homogeneous Degree-1 Identity:
      sum_j dM_{ij} M_{ij} == <dp^{(M)}_i, p_i>,   -sum_i dM_{ij} M_{ij} == -<dk^{(M)}_j, k_j>
    to compute dp, dk, dbeta, dg0, and dg_M in [32, 32] tiles (max_regs <= 168, 0 spills).
    """
    i_sub = tl.program_id(0).to(tl.int64)  # 0..1
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_r_local = i_sub * 32 + tl.arange(0, 32)
    o_r = i_t * BT + o_r_local
    m_r = o_r < T

    b_beta_r = tl.load(beta_ptr + bos_hv + o_r * HV, mask=m_r, other=0.0).to(tl.float32)
    b_g0_r = tl.load(g0_ptr + bos_hv + o_r * HV, mask=m_r, other=0.0).to(tl.float32)
    b_g0_exp_r = tl.math.exp2(b_g0_r)
    b_dbeta = tl.load(dbeta_v_ptr + bos_hv + o_r * HV, mask=m_r, other=0.0).to(tl.float32)
    b_dg0 = tl.zeros([32], dtype=tl.float32)
    b_dg_M = tl.zeros([32], dtype=tl.float32)

    for i_k in tl.range(0, tl.cdiv(K, BK), loop_unroll_factor=1):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_rk = m_r[:, None] & m_k[None, :]

        b_dp_bg0 = tl.zeros([32, BK], dtype=tl.float32)
        b_dp_beta_M = tl.zeros([32, BK], dtype=tl.float32)
        b_dk_M = tl.zeros([32, BK], dtype=tl.float32)

        for j_sub in tl.range(0, 2, loop_unroll_factor=1):
            o_c_local = j_sub * 32 + tl.arange(0, 32)
            o_c = i_t * BT + o_c_local
            m_c = o_c < T
            m_ck = m_c[:, None] & m_k[None, :]

            # A^T[r, c] * dw[c] (non-zero for j_sub >= i_sub)
            p_AT = A_ptr + bos_hv * BT + o_r_local[:, None] + o_c[None, :] * (HV * BT)
            b_AT = tl.load(p_AT, mask=m_c[None, :], other=0.0)
            p_dw_c = dw_ptr + bos_hv * K + o_c[:, None] * (HV * K) + o_k[None, :]
            b_dw_c = tl.load(p_dw_c, mask=m_ck, other=0.0)
            b_dp_bg0 = tl.dot(b_AT, b_dw_c, b_dp_bg0, allow_tf32=False)

            # dM[r, c] * k[c] (non-zero for j_sub <= i_sub)
            p_dM_rc = dM_ptr + bos_hv * BT + o_r[:, None] * (HV * BT) + o_c_local[None, :]
            b_dM_rc = tl.load(p_dM_rc, mask=m_r[:, None], other=0.0)
            p_k_c = k_ptr + bos_h * K + o_c[:, None] * (H * K) + o_k[None, :]
            b_k_c = tl.load(p_k_c, mask=m_ck, other=0.0)
            b_dp_beta_M = tl.dot(b_dM_rc, b_k_c, b_dp_beta_M, allow_tf32=False)

            # dM[c, r]^T * (beta[c] * p[c])
            p_dM_cr = dM_ptr + bos_hv * BT + o_c[:, None] * (HV * BT) + o_r_local[None, :]
            b_dM_cr = tl.load(p_dM_cr, mask=m_c[:, None], other=0.0)
            b_beta_c = tl.load(beta_ptr + bos_hv + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
            p_p_c = p_ptr + bos_h * K + o_c[:, None] * (H * K) + o_k[None, :]
            b_p_c = tl.load(p_p_c, mask=m_ck, other=0.0)
            b_pb_c = (b_p_c.to(tl.float32) * b_beta_c[:, None]).to(b_dM_cr.dtype)
            b_dk_M = tl.dot(tl.trans(b_dM_cr), b_pb_c, b_dk_M, allow_tf32=False)

        p_k_r = k_ptr + bos_h * K + o_r[:, None] * (H * K) + o_k[None, :]
        p_p_r = p_ptr + bos_h * K + o_r[:, None] * (H * K) + o_k[None, :]
        p_dk_in = dk_in_ptr + bos_hv * K + o_r[:, None] * (HV * K) + o_k[None, :]
        p_dk_out = dk_out_ptr + bos_hv * K + o_r[:, None] * (HV * K) + o_k[None, :]
        p_dp_out = dp_out_ptr + bos_hv * K + o_r[:, None] * (HV * K) + o_k[None, :]

        b_k_r = tl.load(p_k_r, mask=m_rk, other=0.0).to(tl.float32)
        b_p_r = tl.load(p_p_r, mask=m_rk, other=0.0).to(tl.float32)
        b_dk_in = tl.load(p_dk_in, mask=m_rk, other=0.0).to(tl.float32)

        b_dp = b_dp_bg0 * (b_beta_r * b_g0_exp_r)[:, None] + b_dp_beta_M * b_beta_r[:, None]
        b_dk = b_dk_in + b_dk_M

        b_dbeta += tl.sum((b_dp_bg0 * b_g0_exp_r[:, None] + b_dp_beta_M) * b_p_r, axis=1)
        b_dg0 += tl.sum(b_dp * b_p_r, axis=1)
        b_dg_M -= tl.sum(b_dk_M * b_k_r, axis=1)

        tl.store(p_dp_out, b_dp.to(p_dp_out.dtype.element_ty), mask=m_rk)
        tl.store(p_dk_out, b_dk.to(p_dk_out.dtype.element_ty), mask=m_rk)

    tl.store(dbeta_out_ptr + bos_hv + o_r * HV, b_dbeta.to(dbeta_out_ptr.dtype.element_ty), mask=m_r)
    tl.store(dg0_out_ptr + bos_hv + o_r * HV, b_dg0, mask=m_r)
    tl.store(dg_M_out_ptr + bos_hv + o_r * HV, b_dg_M, mask=m_r)


@triton.jit
def _comba_bwd_dg_cumsum_rsi_kernel(
    dg_in_ptr,
    dg0_ptr,
    dg_M_ptr,
    dg_out_ptr,
    T,
    B: tl.constexpr,
    HV: tl.constexpr,
    BT: tl.constexpr,
    NK: tl.constexpr,
    MUTATION_MODE: tl.constexpr,
):
    """
    Pass 2B: Combined inclusive (dg_tot) + exclusive (dg0) reverse cumsum in registers
    (max_regs <= 24, 0 spills).
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV

    bos_hv = (i_b * T) * HV + i_hv
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_dg0 = tl.load(dg0_ptr + bos_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
    b_dg_M = tl.load(dg_M_ptr + bos_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
    b_dg_tot = b_dg_M
    for i_k in range(NK):
        p_dg_in = dg_in_ptr + (i_k * B * T * HV) + bos_hv + o_t * HV
        b_dg_tot += tl.load(p_dg_in, mask=m_t, other=0.0).to(tl.float32)

    if MUTATION_MODE == 2:
        b_dg0 = tl.zeros([BT], dtype=tl.float32)
        b_dg_tot = b_dg_tot - b_dg_M

    b_comb = b_dg_tot + b_dg0
    b_sum = tl.sum(b_comb, axis=0)
    b_rev = b_sum - tl.cumsum(b_comb, axis=0) + b_comb
    b_dg_final = b_rev - b_dg0
    tl.store(dg_out_ptr + bos_hv + o_t * HV, b_dg_final.to(dg_out_ptr.dtype.element_ty), mask=m_t)


def chunk_comba_fwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    mutation_mode: int = 0,
):
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    BT = 64
    NT = triton.cdiv(T, BT)

    g0 = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)
    g1 = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)
    M = torch.empty(B, T, HV, BT, device=q.device, dtype=torch.float32)
    A = torch.empty(B, T, HV, BT, device=q.device, dtype=q.dtype)

    _comba_cumsum_pkt_fwd_rsi_kernel[(NT, B * HV)](
        k, p, beta, g, g0, g1, M,
        T=T, H=H, HV=HV, K=K, BT=BT, BK=32,
        MUTATION_MODE=mutation_mode,
        num_warps=4, num_stages=1,
    )
    _comba_solve_tril_16x16_diag_rsi_kernel[(4, NT, B * HV)](
        M, A, T=T, HV=HV, BT=BT,
        num_warps=2, num_stages=1,
    )
    _comba_solve_tril_64_offdiag_rsi_kernel[(NT, B * HV)](
        M, A, T=T, HV=HV, BT=BT,
        num_warps=4, num_stages=1,
    )

    w = torch.empty(B, T, HV, K, device=q.device, dtype=q.dtype)
    u = torch.empty(B, T, HV, V, device=q.device, dtype=v.dtype)
    _comba_recompute_w_u_rsi_kernel[(triton.cdiv(K, 32), NT * 2, B * HV)](
        p, v, beta, g0, A, w, u,
        T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=32, BV=32,
        num_warps=4, num_stages=1,
    )

    BV_h = 32 if (K <= 64 or (q.dtype != torch.float32 and initial_state is None and not output_final_state)) else 16
    nw_h = 4 if q.dtype != torch.float32 else 8
    NV_h = triton.cdiv(V, BV_h)
    h = torch.empty(B * HV, NT, K, V, device=q.device, dtype=q.dtype)
    v_new = torch.empty(B, T, HV, V, device=q.device, dtype=v.dtype)
    final_state = torch.empty(B, HV, K, V, device=q.device, dtype=torch.float32) if output_final_state else None

    _comba_fwd_h_rsi_kernel[(NV_h, B * HV)](
        k, w, u, g1, h, v_new, initial_state, final_state,
        T=T, NT=NT, H=H, HV=HV, K=K, V=V, BT=BT, BV=BV_h,
        USE_INIT_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        num_warps=nw_h, num_stages=1,
    )

    BK_o = 32
    BV_o = 32
    BS_o = 32 if q.dtype != torch.float32 else 16
    NS_o = BT // BS_o
    NV_o = triton.cdiv(V, BV_o)
    o = torch.empty(B, T, HV, V, device=q.device, dtype=v.dtype)
    _comba_fwd_o_rsi_kernel[(NV_o, NT * NS_o, B * HV)](
        q, k, v_new, h, g1, o, scale,
        T=T, NT=NT, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK_o, BV=BV_o, BS=BS_o,
        num_warps=4, num_stages=1,
    )
    return g0, g1, A, w, v_new, h, o, final_state


def chunk_comba_bwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g0: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    mutation_mode: int = 0,
):
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    BT = 64
    NT = triton.cdiv(T, BT)

    dv_local = torch.empty(B, T, HV, V, device=q.device, dtype=v.dtype)
    _comba_bwd_dv_local_rsi_kernel[(NT, B * HV)](
        q, k, g1, do, dv_local, scale,
        T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=32, BV=32,
        num_warps=4, num_stages=1,
    )

    BV_h = 32 if K <= 64 else 16
    nw_h = 4 if q.dtype != torch.float32 else 8
    NV_h = triton.cdiv(V, BV_h)
    dh = torch.empty(B * HV, NT, K, V, device=q.device, dtype=q.dtype)
    dh0 = torch.empty(B, HV, K, V, device=q.device, dtype=torch.float32) if initial_state is not None else None
    dv_new = torch.empty(B, T, HV, V, device=q.device, dtype=v.dtype)

    _comba_bwd_dhu_rsi_kernel[(NV_h, B * HV)](
        q, k, w, g1, do, dv_local, dh, dh0, dv_new, dht, scale,
        T=T, NT=NT, H=H, HV=HV, K=K, V=V, BT=BT, BV=BV_h,
        USE_DHT=dht is not None,
        STORE_DH0=initial_state is not None,
        num_warps=nw_h, num_stages=1,
    )

    BK_q = 32
    BV_q = 32
    NK_q = triton.cdiv(K, BK_q)
    ds = torch.empty(B, T, HV, BT, device=q.device, dtype=q.dtype)
    _comba_bwd_ds_rsi_kernel[(2, NT, B * HV)](
        do, v_new, g1, ds, scale,
        T=T, HV=HV, V=V, BT=BT, BV=BV_q,
        num_warps=4, num_stages=1,
    )

    dq_hv = torch.empty(B, T, HV, K, device=q.device, dtype=q.dtype)
    dk_in = torch.empty(B, T, HV, K, device=q.device, dtype=k.dtype)
    dw = torch.empty(B, T, HV, K, device=q.device, dtype=w.dtype)
    dg_in = torch.zeros(NK_q, B, T, HV, device=q.device, dtype=torch.float32)

    BS_q = 32 if q.dtype != torch.float32 else 16
    NS_q = BT // BS_q
    _comba_bwd_dqkwg_rsi_kernel[(NK_q, NT * NS_q, B * HV)](
        q, k, v_new, h, g1, do, dh, dv_new, ds, dq_hv, dk_in, dw, dg_in, scale,
        T=T, NT=NT, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK_q, BV=BV_q, BS=BS_q,
        num_warps=4, num_stages=1,
    )

    dv = torch.empty(B, T, HV, V, device=q.device, dtype=v.dtype)
    dA_raw = torch.empty(B, T, HV, BT, device=q.device, dtype=q.dtype)
    dM = torch.empty(B, T, HV, BT, device=q.device, dtype=q.dtype)
    dbeta_v = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)

    _comba_bwd_wy_dA_dv_rsi_kernel[(2, NT, B * HV)](
        v, p, beta, g0, A, dw, dv_new, dv, dA_raw, dbeta_v,
        T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=32, BV=32,
        num_warps=4, num_stages=1,
    )
    _comba_bwd_wy_dM_block_rsi_kernel[(4, NT, B * HV)](
        A, dA_raw, g0, g1, dM,
        T=T, HV=HV, BT=BT,
        num_warps=4, num_stages=1,
    )

    dk_hv = torch.empty(B, T, HV, K, device=q.device, dtype=k.dtype)
    dp_hv = torch.empty(B, T, HV, K, device=q.device, dtype=p.dtype)
    dbeta = torch.empty(B, T, HV, device=q.device, dtype=beta.dtype)
    dg0_tmp = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)
    dg_M_tmp = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)
    dg = torch.empty(B, T, HV, device=q.device, dtype=g0.dtype)

    _comba_bwd_wy_dkp_dg_rsi_kernel[(2, NT, B * HV)](
        k, p, beta, g0, A, dM, dw, dk_in, dbeta_v, dk_hv, dp_hv, dbeta, dg0_tmp, dg_M_tmp,
        T=T, H=H, HV=HV, K=K, BT=BT, BK=32,
        num_warps=nw_h, num_stages=1,
    )
    _comba_bwd_dg_cumsum_rsi_kernel[(NT, B * HV)](
        dg_in, dg0_tmp, dg_M_tmp, dg,
        T=T, B=B, HV=HV, BT=BT, NK=NK_q, MUTATION_MODE=mutation_mode,
        num_warps=2, num_stages=1,
    )

    if HV != H:
        G_rep = HV // H
        dq = dq_hv.view(B, T, H, G_rep, K).sum(dim=3)
        dk = dk_hv.view(B, T, H, G_rep, K).sum(dim=3)
        dp = dp_hv.view(B, T, H, G_rep, K).sum(dim=3)
    else:
        dq, dk, dp = dq_hv, dk_hv, dp_hv

    return dq, dk, dv, dp, dg, dbeta, dh0


class FusedCombaPreRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.contiguous()
        w = w.contiguous()
        K = x.shape[-1]
        N_ROWS = x.numel() // K
        BK = triton.next_power_of_2(K)
        y = torch.empty_like(x)
        rstd = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
        _comba_qk_l2norm_fwd_rsi_kernel[(N_ROWS,)](
            x, w, y, rstd, N_ROWS=N_ROWS, K=K, BK=BK, EPS=eps,
            num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(x, w, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w, rstd = ctx.saved_tensors
        K = x.shape[-1]
        N_ROWS = x.numel() // K
        BK = triton.next_power_of_2(K)
        dx = torch.empty_like(x)
        dw_part = torch.empty(N_ROWS, K, device=x.device, dtype=torch.float32)
        _comba_qk_l2norm_bwd_rsi_kernel[(N_ROWS,)](
            x, w, rstd, dy.contiguous(), dx, dw_part,
            N_ROWS=N_ROWS, K=K, BK=BK,
            num_warps=4, num_stages=1,
        )
        return dx, dw_part.sum(dim=0).to(w.dtype), None


def _qkp_l2norm_fwd_rsi(q: torch.Tensor, k: torch.Tensor, p: torch.Tensor, eps: float = 1e-6):
    K = q.shape[-1]
    N_ROWS = q.numel() // K
    BK = triton.next_power_of_2(K)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    p_out = torch.empty_like(p)
    q_rstd = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)
    k_rstd = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)
    p_rstd = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)
    _comba_qkp_l2norm_fwd_rsi_kernel[(N_ROWS,)](
        q, k, p, q_out, k_out, p_out, q_rstd, k_rstd, p_rstd,
        N_ROWS=N_ROWS, K=K, BK=BK, EPS=eps,
        num_warps=2, num_stages=1,
    )
    return q_out, q_rstd, k_out, k_rstd, p_out, p_rstd


def _qkp_l2norm_bwd_rsi(q, q_rstd, k, k_rstd, p, p_rstd, dq, dk, dp):
    K = q.shape[-1]
    N_ROWS = q.numel() // K
    BK = triton.next_power_of_2(K)
    dq_out = torch.empty_like(q)
    dk_out = torch.empty_like(k)
    dp_out = torch.empty_like(p)
    _comba_qkp_l2norm_bwd_rsi_kernel[(N_ROWS,)](
        q, k, p, q_rstd, k_rstd, p_rstd,
        dq.contiguous(), dk.contiguous(), dp.contiguous(),
        dq_out, dk_out, dp_out,
        N_ROWS=N_ROWS, K=K, BK=BK,
        num_warps=2, num_stages=1,
    )
    return dq_out, dk_out, dp_out


class FusedCombaQKPL2NormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, p: torch.Tensor):
        q_c, k_c, p_c = q.contiguous(), k.contiguous(), p.contiguous()
        q_out, q_rstd, k_out, k_rstd, p_out, p_rstd = _qkp_l2norm_fwd_rsi(q_c, k_c, p_c)
        ctx.save_for_backward(q_out, q_rstd, k_out, k_rstd, p_out, p_rstd)
        return q_out, k_out, p_out

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor, dp: torch.Tensor):
        q_out, q_rstd, k_out, k_rstd, p_out, p_rstd = ctx.saved_tensors
        return _qkp_l2norm_bwd_rsi(q_out, q_rstd, k_out, k_rstd, p_out, p_rstd, dq, dk, dp)


class ChunkCombaRSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        mutation_mode: int,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        p = p.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        if initial_state is not None:
            initial_state = initial_state.contiguous()

        if use_qk_l2norm_in_kernel:
            q, q_rstd, k, k_rstd, p, p_rstd = _qkp_l2norm_fwd_rsi(q, k, p)
        else:
            q_rstd = k_rstd = p_rstd = None

        g0, g1, A, w, v_new, h, o, final_state = chunk_comba_fwd_rsi(
            q=q, k=k, v=v, p=p, g=g, beta=beta, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            mutation_mode=mutation_mode,
        )
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, p, p_rstd, v, g0, g1, beta, A, w, v_new, h, initial_state
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.mutation_mode = mutation_mode
        ctx.g_dtype = g.dtype
        return o.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        (
            q, q_rstd, k, k_rstd, p, p_rstd, v, g0, g1, beta, A, w, v_new, h, initial_state
        ) = ctx.saved_tensors
        do = do.contiguous()
        if dht is not None:
            dht = dht.contiguous()

        dq, dk, dv, dp, dg, dbeta, dh0 = chunk_comba_bwd_rsi(
            q=q, k=k, v=v, p=p, g0=g0, g1=g1, beta=beta, A=A, w=w, v_new=v_new, h=h,
            scale=ctx.scale, initial_state=initial_state, do=do, dht=dht,
            mutation_mode=ctx.mutation_mode,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq, dk, dp = _qkp_l2norm_bwd_rsi(q, q_rstd, k, k_rstd, p, p_rstd, dq, dk, dp)
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dp.to(p.dtype),
            dg.to(ctx.g_dtype),
            dbeta.to(beta.dtype),
            None,
            dh0,
            None,
            None,
            None,
        )


def chunk_comba_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: Optional[torch.Tensor] = None,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    mutation_mode: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if p is None:
        p = k
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(v[..., 0])
    return ChunkCombaRSIFunction.apply(
        q, k, v, p, g, beta, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel, mutation_mode
    )


# ==============================================================================
# 3. Single-Pass Multi-Warp Online Decoding (`comba_decode_rsi`)
# ==============================================================================

@triton.jit
def _comba_decode_rsi_kernel(
    q_ptr,
    k_ptr,
    p_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    o_ptr,
    h0_ptr,
    ht_ptr,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    """
    Single-pass multi-warp online decoding kernel for Comba (T_q in {1, 4, 16}).
    Tiles over BV=32 with num_warps=4 (4x fewer CTAs and 4x fewer q/k/p/g/beta HBM reloads than upstream BV=8).
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    bos_hv = (i_b * T) * HV + i_hv
    bos_h = (i_b * T) * H + i_h
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INIT_STATE:
        p_h0 = h0_ptr + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h0, mask=m_kv, other=0.0).to(tl.float32)

    p_q = q_ptr + bos_h * K + o_k
    p_k = k_ptr + bos_h * K + o_k
    p_p = p_ptr + bos_h * K + o_k
    p_v = v_ptr + bos_hv * V + o_v
    p_g = g_ptr + bos_hv
    p_beta = beta_ptr + bos_hv
    p_o = o_ptr + bos_hv * V + o_v

    for _ in range(T):
        b_q = tl.load(p_q, mask=m_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)
        b_p = tl.load(p_p, mask=m_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        if USE_QK_L2NORM:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q, axis=0) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)
            b_p = b_p * tl.rsqrt(tl.sum(b_p * b_p, axis=0) + 1e-6)

        b_q = b_q * scale
        b_v = (b_v - tl.sum(b_h * b_p[:, None], axis=0)) * b_beta
        b_h = b_h * tl.math.exp(b_g) + b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)

        p_q += H * K
        p_k += H * K
        p_p += H * K
        p_v += HV * V
        p_g += HV
        p_beta += HV
        p_o += HV * V

    if STORE_FINAL_STATE:
        p_ht = ht_ptr + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_kv)


def comba_decode_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    if scale is None:
        scale = K ** -0.5
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    ht = torch.empty(B, HV, K, V, device=q.device, dtype=torch.float32) if output_final_state else None
    _comba_decode_rsi_kernel[(NV, B * HV)](
        q.contiguous(),
        k.contiguous(),
        p.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        o,
        initial_state.contiguous() if initial_state is not None else None,
        ht,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INIT_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel,
        num_warps=4,
        num_stages=1,
    )
    return o, ht


# ==============================================================================
# 4. Superfused Frontend/Epilogue Kernels & 5-Stage `CombaBlockRSI` (`A0..A4`)
# ==============================================================================

@triton.jit
def _comba_qk_l2norm_corr_fwd_rsi_kernel(
    q_ptr,
    k_ptr,
    decay_ptr,
    D_ptr,
    q_out_ptr,
    k_out_ptr,
    p_out_ptr,
    q_rstd_ptr,
    k_rstd_ptr,
    N_ROWS,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    EPS: tl.constexpr,
    MUTATION_MODE: tl.constexpr,
):
    """
    Order-correct fused L2Norm + inner-decay + output-correction frontend:
      q_hat = L2Norm(q),  k_hat = L2Norm(k)
      p = sigmoid(decay_h) * k_hat
      q_tilde = q_hat - D_h * p
    Fixes upstream Comba.forward's inner-decay L2Norm erasure bug while fusing 5 passes into 1!
    """
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    i_h = i_row % H
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    q = tl.load(q_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    decay_h = tl.load(decay_ptr + i_h).to(tl.float32)
    s_h = tl.sigmoid(decay_h)
    D_h = tl.load(D_ptr + i_h).to(tl.float32)

    q_rstd = tl.rsqrt(tl.sum(q * q, axis=0) + EPS)
    k_rstd = tl.rsqrt(tl.sum(k * k, axis=0) + EPS)
    q_hat = q * q_rstd
    k_hat = k * k_rstd

    if MUTATION_MODE == 3:
        # M3 mutation: reproduce upstream bug where p = s_h * k is L2-normalized back to k_hat!
        p_vec = k_hat
        q_tilde = q_hat - D_h * p_vec
    else:
        p_vec = s_h * k_hat
        q_tilde = q_hat - D_h * p_vec

    tl.store(q_rstd_ptr + i_row, q_rstd)
    tl.store(k_rstd_ptr + i_row, k_rstd)
    tl.store(q_out_ptr + i_row * K + o_k, q_tilde.to(q_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(k_out_ptr + i_row * K + o_k, k_hat.to(k_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(p_out_ptr + i_row * K + o_k, p_vec.to(p_out_ptr.dtype.element_ty), mask=m_k)


@triton.jit
def _comba_qk_l2norm_corr_bwd_rsi_kernel(
    q_in_ptr,
    k_in_ptr,
    q_rstd_ptr,
    k_rstd_ptr,
    decay_ptr,
    D_ptr,
    dq_tilde_ptr,
    dk_hat_ptr,
    dp_ptr,
    dq_out_ptr,
    dk_out_ptr,
    ddecay_part_ptr,
    dD_part_ptr,
    N_ROWS,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    i_h = i_row % H
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    q = tl.load(q_in_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    k = tl.load(k_in_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    q_rstd = tl.load(q_rstd_ptr + i_row).to(tl.float32)
    k_rstd = tl.load(k_rstd_ptr + i_row).to(tl.float32)
    q_hat = q * q_rstd
    k_hat = k * k_rstd

    decay_h = tl.load(decay_ptr + i_h).to(tl.float32)
    s_h = tl.sigmoid(decay_h)
    D_h = tl.load(D_ptr + i_h).to(tl.float32)

    dq_tilde = tl.load(dq_tilde_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    dk_hat_in = tl.load(dk_hat_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    dp_in = tl.load(dp_ptr + i_row * K + o_k, mask=m_k, other=0.0).to(tl.float32)

    # q_tilde = q_hat - D_h * p,  p = s_h * k_hat
    p_vec = s_h * k_hat
    dD_row = -tl.sum(dq_tilde * p_vec, axis=0)
    dp_tot = dp_in - D_h * dq_tilde
    ds_row = tl.sum(dp_tot * k_hat, axis=0)
    ddecay_row = ds_row * s_h * (1.0 - s_h)

    dq_hat = dq_tilde
    dk_hat = dk_hat_in + s_h * dp_tot

    dq = (dq_hat - q_hat * tl.sum(dq_hat * q_hat, axis=0)) * q_rstd
    dk = (dk_hat - k_hat * tl.sum(dk_hat * k_hat, axis=0)) * k_rstd

    tl.store(dq_out_ptr + i_row * K + o_k, dq.to(dq_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(dk_out_ptr + i_row * K + o_k, dk.to(dk_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(ddecay_part_ptr + i_row, ddecay_row)
    tl.store(dD_part_ptr + i_row, dD_row)


class FusedCombaQKL2NormCorrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, decay: torch.Tensor, D: torch.Tensor, mutation_mode: int = 0):
        q = q.contiguous()
        k = k.contiguous()
        decay = decay.contiguous()
        D = D.contiguous()
        B, T, H, K = q.shape
        N_ROWS = B * T * H
        BK = triton.next_power_of_2(K)

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        p_out = torch.empty_like(k)
        q_rstd = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)
        k_rstd = torch.empty(N_ROWS, device=k.device, dtype=torch.float32)

        _comba_qk_l2norm_corr_fwd_rsi_kernel[(N_ROWS,)](
            q, k, decay, D, q_out, k_out, p_out, q_rstd, k_rstd,
            N_ROWS=N_ROWS, H=H, K=K, BK=BK, EPS=1e-6, MUTATION_MODE=mutation_mode,
            num_warps=2, num_stages=1,
        )
        ctx.save_for_backward(q, k, q_rstd, k_rstd, decay, D)
        return q_out, k_out, p_out

    @staticmethod
    def backward(ctx, dq_tilde: torch.Tensor, dk_hat: torch.Tensor, dp: torch.Tensor):
        q, k, q_rstd, k_rstd, decay, D = ctx.saved_tensors
        B, T, H, K = q.shape
        N_ROWS = B * T * H
        BK = triton.next_power_of_2(K)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        ddecay_part = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)
        dD_part = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)

        _comba_qk_l2norm_corr_bwd_rsi_kernel[(N_ROWS,)](
            q, k, q_rstd, k_rstd, decay, D,
            dq_tilde.contiguous(), dk_hat.contiguous(), dp.contiguous(),
            dq, dk, ddecay_part, dD_part,
            N_ROWS=N_ROWS, H=H, K=K, BK=BK,
            num_warps=2, num_stages=1,
        )
        ddecay = ddecay_part.view(B * T, H).sum(dim=0).to(decay.dtype)
        dD = dD_part.view(B * T, H).sum(dim=0).to(D.dtype)
        return dq, dk, ddecay, dD, None


def fused_comba_qk_l2norm_corr_rsi(
    q: torch.Tensor, k: torch.Tensor, decay: torch.Tensor, D: torch.Tensor, mutation_mode: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return FusedCombaQKL2NormCorrFunction.apply(q, k, decay, D, mutation_mode)


@triton.jit
def _comba_gate_prep_fwd_rsi_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_out_ptr,
    beta_out_ptr,
    N_TOKENS,
    HV: tl.constexpr,
):
    i_tok = tl.program_id(0).to(tl.int64)
    if i_tok >= N_TOKENS:
        return
    o_h = tl.arange(0, HV)
    a = tl.load(a_ptr + i_tok * HV + o_h).to(tl.float32)
    b = tl.load(b_ptr + i_tok * HV + o_h).to(tl.float32)
    A_log = tl.load(A_log_ptr + o_h).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + o_h).to(tl.float32)

    x = a + dt_bias
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.math.exp(x)))
    g = -tl.math.exp(A_log) * sp
    beta = tl.sigmoid(b)
    tl.store(g_out_ptr + i_tok * HV + o_h, g)
    tl.store(beta_out_ptr + i_tok * HV + o_h, beta.to(beta_out_ptr.dtype.element_ty))


@triton.jit
def _comba_gate_prep_bwd_rsi_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    dg_ptr,
    dbeta_ptr,
    da_out_ptr,
    db_out_ptr,
    dA_log_part_ptr,
    ddt_bias_part_ptr,
    N_TOKENS,
    HV: tl.constexpr,
):
    i_tok = tl.program_id(0).to(tl.int64)
    if i_tok >= N_TOKENS:
        return
    o_h = tl.arange(0, HV)
    a = tl.load(a_ptr + i_tok * HV + o_h).to(tl.float32)
    b = tl.load(b_ptr + i_tok * HV + o_h).to(tl.float32)
    A_log = tl.load(A_log_ptr + o_h).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + o_h).to(tl.float32)
    dg = tl.load(dg_ptr + i_tok * HV + o_h).to(tl.float32)
    dbeta = tl.load(dbeta_ptr + i_tok * HV + o_h).to(tl.float32)

    x = a + dt_bias
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.math.exp(x)))
    A_exp = tl.math.exp(A_log)
    da = -dg * A_exp * tl.sigmoid(x)
    dA_log = -dg * A_exp * sp
    beta = tl.sigmoid(b)
    db = dbeta * beta * (1.0 - beta)

    tl.store(da_out_ptr + i_tok * HV + o_h, da.to(da_out_ptr.dtype.element_ty))
    tl.store(db_out_ptr + i_tok * HV + o_h, db.to(db_out_ptr.dtype.element_ty))
    tl.store(dA_log_part_ptr + i_tok * HV + o_h, dA_log)
    tl.store(ddt_bias_part_ptr + i_tok * HV + o_h, da)


class FusedCombaGatePrepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
        a = a.contiguous()
        b = b.contiguous()
        B, T, HV = a.shape
        N_TOKENS = B * T
        g = torch.empty(B, T, HV, device=a.device, dtype=torch.float32)
        beta = torch.empty_like(b)
        _comba_gate_prep_fwd_rsi_kernel[(N_TOKENS,)](
            a, b, A_log, dt_bias, g, beta, N_TOKENS=N_TOKENS, HV=HV,
            num_warps=1, num_stages=1,
        )
        ctx.save_for_backward(a, b, A_log, dt_bias)
        return g, beta

    @staticmethod
    def backward(ctx, dg: torch.Tensor, dbeta: torch.Tensor):
        a, b, A_log, dt_bias = ctx.saved_tensors
        B, T, HV = a.shape
        N_TOKENS = B * T
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        dA_log_part = torch.empty(N_TOKENS, HV, device=a.device, dtype=torch.float32)
        ddt_bias_part = torch.empty(N_TOKENS, HV, device=a.device, dtype=torch.float32)
        _comba_gate_prep_bwd_rsi_kernel[(N_TOKENS,)](
            a, b, A_log, dt_bias, dg.contiguous(), dbeta.contiguous(),
            da, db, dA_log_part, ddt_bias_part, N_TOKENS=N_TOKENS, HV=HV,
            num_warps=1, num_stages=1,
        )
        return da, db, dA_log_part.sum(dim=0).to(A_log.dtype), ddt_bias_part.sum(dim=0).to(dt_bias.dtype)


@triton.jit
def _comba_rmsnorm_gated_fwd_rsi_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    N_ROWS,
    V: tl.constexpr,
    BV: tl.constexpr,
    EPS: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    o_v = tl.arange(0, BV)
    m_v = o_v < V
    x = tl.load(x_ptr + i_row * V + o_v, mask=m_v, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + i_row * V + o_v, mask=m_v, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + o_v, mask=m_v, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / V) + EPS)
    y = x * rstd * w * tl.sigmoid(g)
    tl.store(rstd_ptr + i_row, rstd)
    tl.store(y_ptr + i_row * V + o_v, y.to(y_ptr.dtype.element_ty), mask=m_v)


@triton.jit
def _comba_rmsnorm_gated_bwd_rsi_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    rstd_ptr,
    dy_ptr,
    dx_ptr,
    dg_ptr,
    dw_part_ptr,
    N_ROWS,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    o_v = tl.arange(0, BV)
    m_v = o_v < V
    x = tl.load(x_ptr + i_row * V + o_v, mask=m_v, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + i_row * V + o_v, mask=m_v, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + o_v, mask=m_v, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + i_row).to(tl.float32)
    dy = tl.load(dy_ptr + i_row * V + o_v, mask=m_v, other=0.0).to(tl.float32)

    sig_g = tl.sigmoid(g)
    x_hat = x * rstd
    dy_g = dy * sig_g
    dw = dy_g * x_hat
    dx_hat = dy_g * w
    dx = (dx_hat - x_hat * (tl.sum(dx_hat * x_hat, axis=0) * (1.0 / V))) * rstd
    dg = dy * x_hat * w * sig_g * (1.0 - sig_g)

    tl.store(dx_ptr + i_row * V + o_v, dx.to(dx_ptr.dtype.element_ty), mask=m_v)
    tl.store(dg_ptr + i_row * V + o_v, dg.to(dg_ptr.dtype.element_ty), mask=m_v)
    tl.store(dw_part_ptr + i_row * V + o_v, dw, mask=m_v)


class FusedCombaRMSNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, w: torch.Tensor, eps: float = 1e-5):
        x = x.contiguous()
        g = g.contiguous()
        w = w.contiguous()
        V = x.shape[-1]
        N_ROWS = x.numel() // V
        BV = triton.next_power_of_2(V)
        y = torch.empty_like(x)
        rstd = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
        _comba_rmsnorm_gated_fwd_rsi_kernel[(N_ROWS,)](
            x, g, w, y, rstd, N_ROWS=N_ROWS, V=V, BV=BV, EPS=eps,
            num_warps=2, num_stages=1,
        )
        ctx.save_for_backward(x, g, w, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, g, w, rstd = ctx.saved_tensors
        V = x.shape[-1]
        N_ROWS = x.numel() // V
        BV = triton.next_power_of_2(V)
        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        dw_part = torch.empty(N_ROWS, V, device=x.device, dtype=torch.float32)
        _comba_rmsnorm_gated_bwd_rsi_kernel[(N_ROWS,)](
            x, g, w, rstd, dy.contiguous(), dx, dg, dw_part,
            N_ROWS=N_ROWS, V=V, BV=BV,
            num_warps=2, num_stages=1,
        )
        return dx, dg, dw_part.sum(dim=0).to(w.dtype), None


class CombaBlockRSI(nn.Module):
    """
    5-Stage Progressive Comba Block (`Stages A0 -> A1 -> A2 -> A3 -> A4`):
    - Stage A0: Upstream `chunk_comba` + separate `q, k, v, a, b, g_out` projections + eager frontend/epilogue.
    - Stage A1: Replace upstream `chunk_comba` (`12/43` spilling specs) with zero-spill `chunk_comba_rsi`.
    - Stage A2: + Order-correct fused `_comba_qk_l2norm_corr_fwd_rsi_kernel` (`L2Norm(q, k) -> p = s_h * k_hat, q_tilde = q_hat - D_h * p`).
    - Stage A3: + Fused single-GEMM `qkvabg_proj` + fused `_comba_gate_prep_fwd_rsi_kernel`.
    - Stage A4: + Fused `_comba_rmsnorm_gated_fwd_rsi_kernel` + pre-norm fused residual.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        stage: str = "A4",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_dim = num_heads * head_dim
        self.value_dim = num_heads * head_dim
        self.stage = stage

        self.in_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear( self.value_dim, hidden_size, bias=False)

        self.decay = nn.Parameter(torch.zeros(num_heads))
        self.D = nn.Parameter(torch.ones(num_heads) * 0.02)
        self.A_log = nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = nn.Parameter(torch.full((num_heads,), -2.0))
        self.o_norm_weight = nn.Parameter(torch.ones(head_dim))

        # Packed weight for Stages A3 & A4
        total_out = self.key_dim * 2 + self.value_dim * 2 + num_heads * 2
        self.qkvabg_proj = nn.Linear(hidden_size, total_out, bias=False)
        self.sync_packed_weights()

    @torch.no_grad()
    def sync_packed_weights(self):
        w_cat = torch.cat(
            [
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                self.g_proj.weight,
                self.a_proj.weight,
                self.b_proj.weight,
            ],
            dim=0,
        )
        self.qkvabg_proj.weight.copy_(w_cat)

    def forward(self, hidden_states: torch.Tensor, upstream_chunk_comba_fn=None) -> torch.Tensor:
        B, T, D_model = hidden_states.shape
        H, K = self.num_heads, self.head_dim

        # Pre-RMSNorm
        if self.stage == "A4":
            x = FusedCombaPreRMSNormFunction.apply(hidden_states, self.in_norm_weight, 1e-5)
        else:
            x_f32 = hidden_states.float()
            x = (x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-5)).to(hidden_states.dtype) * self.in_norm_weight

        if self.stage in ("A3", "A4"):
            qkvabg = F.silu(self.qkvabg_proj(x))
            q, k, v, g_out, a, b = torch.split(
                qkvabg,
                [self.key_dim, self.key_dim, self.value_dim, self.value_dim, H, H],
                dim=-1,
            )
            q = q.view(B, T, H, K)
            k = k.view(B, T, H, K)
            v = v.view(B, T, H, K)
            g_out = g_out.view(B, T, H, K)
            g, beta = FusedCombaGatePrepFunction.apply(a, b, self.A_log, self.dt_bias)
        else:
            q = F.silu(self.q_proj(x)).view(B, T, H, K)
            k = F.silu(self.k_proj(x)).view(B, T, H, K)
            v = F.silu(self.v_proj(x)).view(B, T, H, K)
            g_out = F.silu(self.g_proj(x)).view(B, T, H, K)
            a = F.silu(self.a_proj(x))
            b = F.silu(self.b_proj(x))
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        if self.stage in ("A2", "A3", "A4"):
            q_tilde, k_hat, p_vec = fused_comba_qk_l2norm_corr_rsi(q, k, self.decay, self.D)
            o, _ = chunk_comba_rsi(q_tilde, k_hat, v, p_vec, g, beta, use_qk_l2norm_in_kernel=False)
        elif self.stage == "A1":
            q_hat, k_hat, p_hat = FusedCombaQKPL2NormFunction.apply(q, k, k)
            p_vec = (p_hat * self.decay[None, None, :, None].sigmoid()).to(k.dtype)
            q_tilde = (q_hat - self.D[None, None, :, None] * p_vec).to(q.dtype)
            o, _ = chunk_comba_rsi(q_tilde, k_hat, v, p_vec, g, beta, use_qk_l2norm_in_kernel=False)
        else:
            # Stage A0: upstream Comba path (3 separate l2norm_fwd/l2norm_bwd calls + upstream chunk_comba)
            from fla.modules.l2norm import l2norm
            q_hat = l2norm(q)
            k_hat = l2norm(k)
            p_hat = l2norm(k.clone())
            p_vec = (p_hat * self.decay[None, None, :, None].sigmoid()).to(k.dtype)
            q_tilde = (q_hat - self.D[None, None, :, None] * p_vec).to(q.dtype)
            assert upstream_chunk_comba_fn is not None
            o, _ = upstream_chunk_comba_fn(q_tilde, k_hat, v, p_vec, g, beta, use_qk_l2norm_in_kernel=False)

        if self.stage == "A4":
            o = FusedCombaRMSNormGatedFunction.apply(o, g_out, self.o_norm_weight, 1e-5)
        else:
            o_f32 = o.float()
            o = (o_f32 * torch.rsqrt(o_f32.pow(2).mean(-1, keepdim=True) + 1e-5) * self.o_norm_weight * torch.sigmoid(g_out.float())).to(o.dtype)

        o = self.o_proj(o.reshape(B, T, self.value_dim))
        return hidden_states + o


# ==============================================================================
# 5. Zero-Copy 8-GPU NVSwitch Context Parallelism (`Comba-CP = 8`)
# ==============================================================================

class _AllToAllSingleAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        x_c = x.contiguous()
        out = torch.empty_like(x_c)
        dist.all_to_all_single(out, x_c, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_c = grad_output.contiguous()
        out = torch.empty_like(grad_c)
        dist.all_to_all_single(out, grad_c, group=ctx.group)
        return out, None


class _CombaAllGatherSeqAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, world_size: int):
        ctx.group = group
        ctx.world_size = world_size
        x_c = x.contiguous()
        chunks = [torch.empty_like(x_c) for _ in range(world_size)]
        dist.all_gather(chunks, x_c, group=group)
        return torch.cat(chunks, dim=1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        world_size = ctx.world_size
        T_local = grad_out.shape[1] // world_size
        g_chunks = [c.contiguous() for c in grad_out.split(T_local, dim=1)]
        g_local = torch.empty_like(g_chunks[0])
        dist.reduce_scatter(g_local, g_chunks, op=dist.ReduceOp.SUM, group=ctx.group)
        return g_local, None, None


class _CombaUlyssesSeqToHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, world_size: int):
        ctx.group = group
        ctx.world_size = world_size
        NT_PACK, B, T_local, H, K = x.shape
        H_local = H // world_size
        x_send = x.view(NT_PACK, B, T_local, world_size, H_local, K).permute(3, 0, 1, 2, 4, 5).contiguous()
        x_recv = torch.empty_like(x_send)
        dist.all_to_all_single(x_recv, x_send, group=group)
        return x_recv.permute(1, 2, 0, 3, 4, 5).contiguous().view(NT_PACK, B, world_size * T_local, H_local, K)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        world_size = ctx.world_size
        NT_PACK, B, T_total, H_local, K = grad_out.shape
        T_local = T_total // world_size
        g_send = grad_out.view(NT_PACK, B, world_size, T_local, H_local, K).permute(2, 0, 1, 3, 4, 5).contiguous()
        g_recv = torch.empty_like(g_send)
        dist.all_to_all_single(g_recv, g_send, group=ctx.group)
        return g_recv.permute(1, 2, 3, 0, 4, 5).contiguous().view(NT_PACK, B, T_local, H_local * world_size, K), None, None


class _CombaUlyssesHeadToSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, world_size: int):
        ctx.group = group
        ctx.world_size = world_size
        B, T_total, H_local, K = x.shape
        T_local = T_total // world_size
        x_send = x.view(B, world_size, T_local, H_local, K).permute(1, 0, 2, 3, 4).contiguous()
        x_recv = torch.empty_like(x_send)
        dist.all_to_all_single(x_recv, x_send, group=group)
        return x_recv.permute(1, 2, 0, 3, 4).contiguous().view(B, T_local, H_local * world_size, K)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        world_size = ctx.world_size
        B, T_local, H, K = grad_out.shape
        H_local = H // world_size
        g_send = grad_out.view(B, T_local, world_size, H_local, K).permute(2, 0, 1, 3, 4).contiguous()
        g_recv = torch.empty_like(g_send)
        dist.all_to_all_single(g_recv, g_send, group=ctx.group)
        return g_recv.permute(1, 0, 2, 3, 4).contiguous().view(B, T_local * world_size, H_local, K), None, None


def comba_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    p_local: torch.Tensor,
    g_local: torch.Tensor,
    beta_local: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Zero-Copy 8-GPU NVSwitch Ulysses Context Parallelism (`Comba-CP = 8`) calling `chunk_comba_rsi`.
    Packs `(q, k, v, p)` and padded `(g, beta)` into a single 5-slice tensor along dim=0 so
    only ONE `all_to_all_single` collective runs in forward and ONE in backward, and `packed_full[0..3]`
    are already 100% contiguous `[B, T_total, H_local, K]` views.
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    K = q_local.shape[-1]
    gb_pad = F.pad(torch.stack([g_local.to(q_local.dtype), beta_local.to(q_local.dtype)], dim=-1), (0, K - 2))
    packed_in = torch.stack([q_local, k_local, v_local, p_local, gb_pad], dim=0)
    packed_full = _CombaUlyssesSeqToHead.apply(packed_in, group, world_size)
    q_full = packed_full[0]
    k_full = packed_full[1]
    v_full = packed_full[2]
    p_full = packed_full[3]
    g_full = packed_full[4, :, :, :, 0].contiguous()
    beta_full = packed_full[4, :, :, :, 1].contiguous()

    o_full, _ = chunk_comba_rsi(q_full, k_full, v_full, p_full, g_full, beta_full)
    return _CombaUlyssesHeadToSeq.apply(o_full, group, world_size)


def comba_cp8_ring_baseline(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    p_local: torch.Tensor,
    g_local: torch.Tensor,
    beta_local: torch.Tensor,
    upstream_chunk_comba_fn,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Sequence-AllGather / Ring Baseline for 8-GPU Context Parallelism calling upstream `chunk_comba`.
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    T_local = q_local.shape[1]

    q_all = _CombaAllGatherSeqAutograd.apply(q_local, group, world_size)
    k_all = _CombaAllGatherSeqAutograd.apply(k_local, group, world_size)
    v_all = _CombaAllGatherSeqAutograd.apply(v_local, group, world_size)
    p_all = _CombaAllGatherSeqAutograd.apply(p_local, group, world_size)
    g_all = _CombaAllGatherSeqAutograd.apply(g_local, group, world_size)
    beta_all = _CombaAllGatherSeqAutograd.apply(beta_local, group, world_size)

    o_all, _ = upstream_chunk_comba_fn(q_all, k_all, v_all, p_all, g_all, beta_all)
    return o_all[:, rank * T_local : (rank + 1) * T_local].contiguous()


def _collect_compiled_triton(kobj):
    out = []
    seen = set()

    def walk(obj):
        if id(obj) in seen or obj is None:
            return
        seen.add(id(obj))
        if hasattr(obj, "n_regs") and hasattr(obj, "n_spills"):
            out.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        else:
            for attr in ["cache", "device_caches", "kernels", "fn"]:
                if hasattr(obj, attr):
                    walk(getattr(obj, attr))

    walk(kobj)
    return out


def audit_comba_rsi_cubin_resources() -> dict:
    kernel_map = [
        ("_comba_qk_l2norm_fwd_rsi_kernel", _comba_qk_l2norm_fwd_rsi_kernel),
        ("_comba_qk_l2norm_bwd_rsi_kernel", _comba_qk_l2norm_bwd_rsi_kernel),
        ("_comba_qkp_l2norm_fwd_rsi_kernel", _comba_qkp_l2norm_fwd_rsi_kernel),
        ("_comba_qkp_l2norm_bwd_rsi_kernel", _comba_qkp_l2norm_bwd_rsi_kernel),
        ("_comba_cumsum_pkt_fwd_rsi_kernel", _comba_cumsum_pkt_fwd_rsi_kernel),
        ("_comba_solve_tril_16x16_diag_rsi_kernel", _comba_solve_tril_16x16_diag_rsi_kernel),
        ("_comba_solve_tril_64_offdiag_rsi_kernel", _comba_solve_tril_64_offdiag_rsi_kernel),
        ("_comba_recompute_w_u_rsi_kernel", _comba_recompute_w_u_rsi_kernel),
        ("_comba_fwd_h_rsi_kernel", _comba_fwd_h_rsi_kernel),
        ("_comba_fwd_o_rsi_kernel", _comba_fwd_o_rsi_kernel),
        ("_comba_bwd_dv_local_rsi_kernel", _comba_bwd_dv_local_rsi_kernel),
        ("_comba_bwd_dhu_rsi_kernel", _comba_bwd_dhu_rsi_kernel),
        ("_comba_bwd_ds_rsi_kernel", _comba_bwd_ds_rsi_kernel),
        ("_comba_bwd_dqkwg_rsi_kernel", _comba_bwd_dqkwg_rsi_kernel),
        ("_comba_bwd_wy_dA_dv_rsi_kernel", _comba_bwd_wy_dA_dv_rsi_kernel),
        ("_comba_bwd_wy_dM_block_rsi_kernel", _comba_bwd_wy_dM_block_rsi_kernel),
        ("_comba_bwd_wy_dkp_dg_rsi_kernel", _comba_bwd_wy_dkp_dg_rsi_kernel),
        ("_comba_bwd_dg_cumsum_rsi_kernel", _comba_bwd_dg_cumsum_rsi_kernel),
        ("_comba_decode_rsi_kernel", _comba_decode_rsi_kernel),
        ("_comba_qk_l2norm_corr_fwd_rsi_kernel", _comba_qk_l2norm_corr_fwd_rsi_kernel),
        ("_comba_qk_l2norm_corr_bwd_rsi_kernel", _comba_qk_l2norm_corr_bwd_rsi_kernel),
        ("_comba_gate_prep_fwd_rsi_kernel", _comba_gate_prep_fwd_rsi_kernel),
        ("_comba_gate_prep_bwd_rsi_kernel", _comba_gate_prep_bwd_rsi_kernel),
        ("_comba_rmsnorm_gated_fwd_rsi_kernel", _comba_rmsnorm_gated_fwd_rsi_kernel),
        ("_comba_rmsnorm_gated_bwd_rsi_kernel", _comba_rmsnorm_gated_bwd_rsi_kernel),
    ]
    per_kernel = {}
    total_specs = 0
    spilling_specs = 0
    over168_specs = 0
    max_regs_all = 0
    max_spills_all = 0
    max_smem_all = 0

    for name, kobj in kernel_map:
        caches = _collect_compiled_triton(kobj)
        n_sp = len(caches)
        sp = sum(1 for c in caches if getattr(c, "n_spills", 0) > 0)
        ov = sum(1 for c in caches if getattr(c, "n_regs", 0) > 168)
        mr = max((int(getattr(c, "n_regs", 0)) for c in caches), default=0)
        ms = max((int(getattr(c, "n_spills", 0)) for c in caches), default=0)
        ts = sum(int(getattr(c, "n_spills", 0)) for c in caches)
        sm = max((int(getattr(c, "metadata", None).shared) if hasattr(c, "metadata") else 0 for c in caches), default=0)
        per_kernel[name] = {
            "specializations": n_sp,
            "spilling_specs": sp,
            "over168_specs": ov,
            "max_regs": mr,
            "max_spills": ms,
            "total_spills": ts,
            "max_smem_bytes": sm,
        }
        total_specs += n_sp
        spilling_specs += sp
        over168_specs += ov
        max_regs_all = max(max_regs_all, mr)
        max_spills_all = max(max_spills_all, ms)
        max_smem_all = max(max_smem_all, sm)

    return {
        "kernel_count": len(kernel_map),
        "total_specializations": total_specs,
        "spilling_specializations": spilling_specs,
        "over168_specializations": over168_specs,
        "max_regs": max_regs_all,
        "max_spills": max_spills_all,
        "max_smem_bytes": max_smem_all,
        "all_zero_spill": spilling_specs == 0 and over168_specs == 0,
        "kernels": per_kernel,
    }

