#!/usr/bin/env python3
"""
E147 (v138, Round-21 RSI): Zero-Spill SM80 Triton Kernels for Gated DeltaProduct (GDP)
=======================================================================================

Covers:
  - Gated DeltaProduct (GDP): Multi-Householder Product Linear Recurrences for Rotational
    and Parity State Tracking (`arXiv:2502.10297`, Siems et al.; official repository
    `fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_product` &
    `fla/layers/gated_deltaproduct.py`).

Key Algorithmic & Hardware Innovations (Round-21 RSI):
  1. Rectangular Query-Compacted `[BQ, BDP]` Chunking (`BQ = 64 // NH` for `NH in {1, 2, 4}`,
     `BQ = 16, BDP = 48` for `NH = 3`):
     - Every chunk boundary aligns strictly with a whole number `BQ` of real tokens
       (`BDP = BQ * NH <= 64` virtual Householder steps).
     - Because `g_t` is applied only at sub-step `j = 0` of each real token `m`, the chunk-local
       cumulative log-decay at virtual step `r = m * NH + j` is identically `g_cumsum[m] =
       g_cumsum[r // NH]`, eliminating upstream's `[B, T * NH, H]` `g_interleaved` allocation
       and duplicate `chunk_local_cumsum` launch.
     - Intra-chunk readout (`_gdp_fwd_o_rect_kernel`) and backward (`_gdp_bwd_dv_local_rect_kernel`,
       `_gdp_bwd_dhu_rect_kernel`, `_gdp_bwd_state_dqkw_pass1_rect_kernel`) operate directly on
       dense `[BQ_PAD, BK]` real queries `q` and `[BQ_PAD, BV]` output gradients `do` against
       `[64, BK]` virtual keys `k`, cutting query-side Tensor Core MMA FLOPs and HBM traffic by
       `2.0x` (`NH=2`), `3.0x` (`NH=3`), and `4.0x` (`NH=4`) and solving upstream's
       `# TODO: optimize the backward pass like the forward pass` (`chunk.py:204`).
  2. Saved `(w, v_new, h)` Forward States & 3-Pass Split Zero-Spill WY Backward:
     - Eliminates upstream's backward recomputation of `recompute_w_u_fwd` and `chunk_fwd_h`.
     - Splits WY backward into Pass 2A (`dv`, `db_v`, `dA_v`), Pass 2B (`dk_wy_w`, `db_w`,
       `dg_dp_w`, `dA_k`), shared `-Ai^T @ strictTril(dA) @ Ai^T`, and Pass 2C (`dk_A`, `db_A`,
       `dg_dp_A`), plus fused `NH`-substep gate reduction + reverse cumsum (`_gdp_bwd_reduce_dg_rev_cumsum_kernel`),
       achieving `0` SM80 cubin register spills (`max_regs <= 168`) across all 22 Triton kernels.
  3. Superfused `GatedDeltaProductBlockRSI` (`Stages A0..A4`) & Zero-Copy 8-GPU NVSwitch Ulysses
     `GDP-CP = 8` (`gdp_cp8_ulysses_rsi` vs `gdp_cp8_ring_baseline`).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ==============================================================================
# Section 1: Exact FP64 Recurrent Oracle (Supports FP32 & BF16 Verification)
# ==============================================================================

def gdp_recurrent_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int = 2,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Exact FP64 recurrent oracle for Gated DeltaProduct (`arXiv:2502.10297`):
      S_{t,0} = exp(g_t) * S_{t-1, NH}
      S_{t,j} = (I - beta_{t,j} k_{t,j} k_{t,j}^T) S_{t,j-1} + beta_{t,j} k_{t,j} v_{t,j}^T  (j=1..NH)
      o_t     = scale * S_{t,NH}^T q_t
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    nh = int(num_householder)
    assert k.shape == (B, T * nh, H, K)
    assert v.shape == (B, T * nh, H, V)
    assert beta.shape == (B, T * nh, H)
    assert g.shape == (B, T, H)
    if scale is None:
        scale = K ** -0.5

    q64 = q.to(torch.float64) * float(scale)
    k64 = k.to(torch.float64)
    v64 = v.to(torch.float64)
    g64 = g.to(torch.float64)
    b64 = beta.to(torch.float64)

    if initial_state is not None:
        h = initial_state.to(torch.float64).clone()
    else:
        h = torch.zeros((B, H, K, V), dtype=torch.float64, device=q.device)

    outs = []
    for t in range(T):
        decay = torch.exp(g64[:, t, :]).unsqueeze(-1).unsqueeze(-1)
        h = h * decay
        for j in range(nh):
            idx = t * nh + j
            k_ij = k64[:, idx]
            v_ij = v64[:, idx]
            b_ij = b64[:, idx]
            proj = torch.einsum("bhkv,bhk->bhv", h, k_ij)
            h = h + torch.einsum("bhk,bhv->bhkv", k_ij * b_ij.unsqueeze(-1), v_ij - proj)
        o_t = torch.einsum("bhkv,bhk->bhv", h, q64[:, t])
        outs.append(o_t)

    o = torch.stack(outs, dim=1)
    ht = h if output_final_state else None
    return o, ht


# ==============================================================================
# Section 2: Chunk-Local Cumsum & Shared Inverse-Triangular Adjoint Kernels
# ==============================================================================

def _gdp_chunk_params(num_householder: int) -> Tuple[int, int, int]:
    nh = int(num_householder)
    if nh == 1:
        return 64, 64, 64
    elif nh == 2:
        return 32, 64, 32
    elif nh == 3:
        return 22, 64, 32
    elif nh == 4:
        return 16, 64, 16
    else:
        raise ValueError(f"Unsupported num_householder={nh}; expected 1, 2, 3, or 4.")


@triton.jit
def _gdp_local_cumsum_fwd_kernel(
    g_ptr,
    out_ptr,
    T_Q,
    T_DP,
    H: tl.constexpr,
    NH: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    dp_idx = i_t * BT + tl.arange(0, BT)
    m_dp = dp_idx < T_DP
    is_g0 = ((dp_idx % NH) == 0) & m_dp
    q_idx = dp_idx // NH
    b_g = tl.load(g_ptr + (i_b * T_Q + q_idx) * H + i_h, mask=is_g0 & (q_idx < T_Q), other=0.0).to(tl.float32)
    b_cum = tl.cumsum(b_g, axis=0)
    tl.store(out_ptr + (i_b * T_DP + dp_idx) * H + i_h, b_cum, mask=m_dp)


def gdp_chunk_local_cumsum_fwd(g: torch.Tensor, num_householder: int, bt: int = 64) -> torch.Tensor:
    B, T_Q, H = g.shape
    nh = int(num_householder)
    T_DP = T_Q * nh
    NT = triton.cdiv(T_DP, bt)
    out = torch.empty((B, T_DP, H), dtype=torch.float32, device=g.device)
    _gdp_local_cumsum_fwd_kernel[(NT, B * H)](
        g, out, T_Q, T_DP, H=H, NH=nh, BT=bt, num_warps=2, num_stages=1
    )
    return out


@triton.jit
def _bwd_dA_triple_product_kernel(
    Ai_ptr,
    dAi_ptr,
    dA_out_ptr,
    T_DP,
    H: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Computes dA = strictTril(-Ai^T @ strictTril(dAi) @ Ai^T) for [BT, BT] chunk (active BDP <= 64).
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    r_idx = tl.arange(0, BT)
    c_idx = tl.arange(0, BT)
    t_r = i_t * BDP + r_idx
    t_c = i_t * BDP + c_idx
    m_r = (r_idx < BDP) & (t_r < T_DP)
    m_c = (c_idx < BDP) & (t_c < T_DP)
    m_rc = m_r[:, None] & m_c[None, :]
    strict_tril = (r_idx[:, None] > c_idx[None, :]) & m_rc

    base_mat = ((i_b * T_DP + t_r[:, None]) * H + i_h) * BT + c_idx[None, :]
    p_Ai_t = Ai_ptr + ((i_b * T_DP + t_c[None, :]) * H + i_h) * BT + r_idx[:, None]
    b_Ai_t = tl.load(p_Ai_t, mask=m_c[None, :] & m_r[:, None], other=0.0)
    b_dA = tl.load(dAi_ptr + base_mat, mask=strict_tril, other=0.0).to(tl.float32)

    if USE_TF32:
        b_dA = tl.dot(b_dA.to(tl.bfloat16), b_Ai_t.to(tl.bfloat16))
        b_dA = tl.dot(b_Ai_t.to(tl.bfloat16), b_dA.to(tl.bfloat16))
    else:
        b_dA = tl.dot(b_dA, b_Ai_t.to(tl.float32), input_precision="ieee")
        b_dA = tl.dot(b_Ai_t.to(tl.float32), b_dA, input_precision="ieee")

    b_dA = tl.where(strict_tril, -b_dA, 0.0)
    tl.store(dA_out_ptr + base_mat, b_dA, mask=m_rc)


# ==============================================================================
# Section 3: Forward Kernels for Gated DeltaProduct (Zero-Spill SM80)
# ==============================================================================

@triton.jit
def _gdp_fwd_kkt_subchunk_kernel(
    k_ptr,
    g_cum_ptr,
    beta_ptr,
    A_ptr,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Computes strictly lower-triangular A_rc = beta_r * (k_r . k_c) * exp(g_cum[r//NH] - g_cum[c//NH])
    for virtual chunk i_t of length BDP <= 64.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    r_idx = tl.arange(0, BT)
    c_idx = tl.arange(0, BT)
    dp_r = i_t * BDP + r_idx
    dp_c = i_t * BDP + c_idx
    m_r = (r_idx < BDP) & (dp_r < T_DP)
    m_c = (c_idx < BDP) & (dp_c < T_DP)

    b_g = tl.load(g_cum_ptr + (i_b * T_DP + dp_r) * H + i_h, mask=m_r, other=0.0).to(tl.float32)
    b_beta = tl.load(beta_ptr + (i_b * T_DP + dp_r) * H + i_h, mask=m_r, other=0.0).to(tl.float32)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_kr = k_ptr + ((i_b * T_DP + dp_r[:, None]) * H + i_h) * K + o_k[None, :]
        p_kc = k_ptr + ((i_b * T_DP + dp_c[None, :]) * H + i_h) * K + o_k[:, None]
        b_kr = tl.load(p_kr, mask=m_r[:, None] & m_k[None, :], other=0.0)
        b_kc = tl.load(p_kc, mask=m_k[:, None] & m_c[None, :], other=0.0)
        if USE_TF32:
            b_A = tl.dot(b_kr.to(tl.bfloat16), b_kc.to(tl.bfloat16), b_A)
        else:
            b_A = tl.dot(b_kr.to(tl.float32), b_kc.to(tl.float32), b_A, input_precision="ieee")

    strict_tril = (r_idx[:, None] > c_idx[None, :]) & m_r[:, None] & m_c[None, :]
    diff_g = tl.where(strict_tril, b_g[:, None] - b_g[None, :], 0.0)
    b_A = tl.where(strict_tril, b_A * b_beta[:, None] * tl.exp(diff_g), 0.0)

    p_A = A_ptr + ((i_b * T_DP + dp_r[:, None]) * H + i_h) * BT + c_idx[None, :]
    tl.store(p_A, b_A, mask=m_r[:, None] & (c_idx[None, :] < BT))


@triton.jit
def _gdp_fwd_solve_tril_16x16_kernel(
    A_ptr,
    Ai_ptr,
    T_DP,
    H: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Inverts (I + A)^{-1} for chunk i_t of active length BDP <= 64 via 16x16 block substitution.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_16 = tl.arange(0, 16)
    r_16 = o_16[:, None]
    c_16 = o_16[None, :]
    tril_mask = r_16 > c_16
    eye_16 = (r_16 == c_16).to(tl.float32)

    base_chunk = ((i_b * T_DP + i_t * BDP) * H + i_h) * BT
    stride_row = H * BT

    # Step 0: Zero upper-triangular 16x16 blocks so `Ai` requires no separate `torch.zeros` memset
    z_16 = tl.zeros([16, 16], dtype=Ai_ptr.dtype.element_ty)
    for r_blk in range(4):
        row_off = r_blk * 16 + r_16
        m_row = ((i_t * BDP + row_off) < T_DP) & (row_off < BDP)
        for c_blk in range(r_blk + 1, 4):
            col_off = c_blk * 16 + c_16
            tl.store(Ai_ptr + base_chunk + row_off * stride_row + col_off, z_16, mask=m_row)

    # Step 1: Invert up to 4 diagonal 16x16 blocks via Euler repeated squaring (I+N)(I+N^2)(I+N^4)(I+N^8)
    DOT_PREC: tl.constexpr = "tf32" if USE_TF32 else "ieee"
    for blk in range(4):
        row_off = blk * 16 + r_16
        col_off = blk * 16 + c_16
        valid = (row_off < BDP) & ((i_t * BDP + row_off) < T_DP) & (col_off < BDP) & ((i_t * BDP + col_off) < T_DP)
        b_L = tl.load(A_ptr + base_chunk + row_off * stride_row + col_off, mask=valid & tril_mask, other=0.0).to(tl.float32)
        b_N = -b_L
        b_N2 = tl.dot(b_N, b_N, input_precision=DOT_PREC)
        b_N4 = tl.dot(b_N2, b_N2, input_precision=DOT_PREC)
        b_N8 = tl.dot(b_N4, b_N4, input_precision=DOT_PREC)
        b_P2 = eye_16 + b_N + b_N2 + tl.dot(b_N, b_N2, input_precision=DOT_PREC)
        b_P4 = b_P2 + tl.dot(b_N4, b_P2, input_precision=DOT_PREC)
        b_inv = b_P4 + tl.dot(b_N8, b_P4, input_precision=DOT_PREC)
        b_inv = tl.where(r_16 >= c_16, b_inv, 0.0)
        m_store = ((i_t * BDP + row_off) < T_DP) & (row_off < BDP) & (col_off < BT)
        tl.store(Ai_ptr + base_chunk + row_off * stride_row + col_off, b_inv.to(Ai_ptr.dtype.element_ty), mask=m_store)

    # Load inverted diagonal blocks
    m0 = ((i_t * BDP + r_16) < T_DP) & (r_16 < BDP)
    m1 = ((i_t * BDP + 16 + r_16) < T_DP) & ((16 + r_16) < BDP)
    m2 = ((i_t * BDP + 32 + r_16) < T_DP) & ((32 + r_16) < BDP)
    m3 = ((i_t * BDP + 48 + r_16) < T_DP) & ((48 + r_16) < BDP)

    D0 = tl.load(Ai_ptr + base_chunk + r_16 * stride_row + c_16, mask=m0, other=0.0).to(tl.float32)
    D1 = tl.load(Ai_ptr + base_chunk + (16 + r_16) * stride_row + (16 + c_16), mask=m1, other=0.0).to(tl.float32)
    D2 = tl.load(Ai_ptr + base_chunk + (32 + r_16) * stride_row + (32 + c_16), mask=m2, other=0.0).to(tl.float32)

    # Row 1: X10 = -D1 @ A10 @ D0
    A10 = tl.load(A_ptr + base_chunk + (16 + r_16) * stride_row + c_16, mask=m1, other=0.0).to(tl.float32)
    X10 = -tl.dot(D1, tl.dot(A10, D0, input_precision=DOT_PREC), input_precision=DOT_PREC)
    tl.store(Ai_ptr + base_chunk + (16 + r_16) * stride_row + c_16, X10.to(Ai_ptr.dtype.element_ty), mask=m1)

    # Row 2: X21 = -D2 @ A21 @ D1, X20 = -D2 @ (A20 @ D0 + A21 @ X10)
    A20 = tl.load(A_ptr + base_chunk + (32 + r_16) * stride_row + c_16, mask=m2, other=0.0).to(tl.float32)
    A21 = tl.load(A_ptr + base_chunk + (32 + r_16) * stride_row + (16 + c_16), mask=m2, other=0.0).to(tl.float32)
    X21 = -tl.dot(D2, tl.dot(A21, D1, input_precision=DOT_PREC), input_precision=DOT_PREC)
    X20 = -tl.dot(D2, tl.dot(A20, D0, input_precision=DOT_PREC) + tl.dot(A21, X10, input_precision=DOT_PREC), input_precision=DOT_PREC)
    tl.store(Ai_ptr + base_chunk + (32 + r_16) * stride_row + c_16, X20.to(Ai_ptr.dtype.element_ty), mask=m2)
    tl.store(Ai_ptr + base_chunk + (32 + r_16) * stride_row + (16 + c_16), X21.to(Ai_ptr.dtype.element_ty), mask=m2)

    if BDP > 48:
        D3 = tl.load(Ai_ptr + base_chunk + (48 + r_16) * stride_row + (48 + c_16), mask=m3, other=0.0).to(tl.float32)
        A30 = tl.load(A_ptr + base_chunk + (48 + r_16) * stride_row + c_16, mask=m3, other=0.0).to(tl.float32)
        A31 = tl.load(A_ptr + base_chunk + (48 + r_16) * stride_row + (16 + c_16), mask=m3, other=0.0).to(tl.float32)
        A32 = tl.load(A_ptr + base_chunk + (48 + r_16) * stride_row + (32 + c_16), mask=m3, other=0.0).to(tl.float32)
        X32 = -tl.dot(D3, tl.dot(A32, D2, input_precision=DOT_PREC), input_precision=DOT_PREC)
        X31 = -tl.dot(D3, tl.dot(A31, D1, input_precision=DOT_PREC) + tl.dot(A32, X21, input_precision=DOT_PREC), input_precision=DOT_PREC)
        X30 = -tl.dot(
            D3,
            tl.dot(A30, D0, input_precision=DOT_PREC)
            + tl.dot(A31, X10, input_precision=DOT_PREC)
            + tl.dot(A32, X20, input_precision=DOT_PREC),
            input_precision=DOT_PREC,
        )
        tl.store(Ai_ptr + base_chunk + (48 + r_16) * stride_row + c_16, X30.to(Ai_ptr.dtype.element_ty), mask=m3)
        tl.store(Ai_ptr + base_chunk + (48 + r_16) * stride_row + (16 + c_16), X31.to(Ai_ptr.dtype.element_ty), mask=m3)
        tl.store(Ai_ptr + base_chunk + (48 + r_16) * stride_row + (32 + c_16), X32.to(Ai_ptr.dtype.element_ty), mask=m3)


@triton.jit
def _gdp_fwd_wy_wu_kernel(
    k_ptr,
    v_ptr,
    beta_ptr,
    g_cum_ptr,
    Ai_ptr,
    w_ptr,
    u_ptr,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Computes WY representations w = Ai @ (k * beta * exp(g_cum[r//NH])) and u = Ai @ (v * beta).
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_dp = tl.arange(0, BT)
    dp_idx = i_t * BDP + o_dp
    m_dp = (o_dp < BDP) & (dp_idx < T_DP)

    b_beta = tl.load(beta_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    b_g = tl.load(g_cum_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    b_kb_scale = b_beta * tl.exp(b_g)

    p_Ai = Ai_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * BT + o_dp[None, :]
    b_Ai = tl.load(p_Ai, mask=m_dp[:, None] & m_dp[None, :], other=0.0)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = k_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]
        b_k = tl.load(p_k, mask=m_dp[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_kb = b_k * b_kb_scale[:, None]
        if USE_TF32:
            b_w = tl.dot(b_Ai.to(tl.bfloat16), b_kb.to(tl.bfloat16))
        else:
            b_w = tl.dot(b_Ai.to(tl.float32), b_kb, input_precision="ieee")
        p_w = w_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]
        tl.store(p_w, b_w.to(w_ptr.dtype.element_ty), mask=m_dp[:, None] & m_k[None, :])

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_v = v_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        b_v = tl.load(p_v, mask=m_dp[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_vb = b_v * b_beta[:, None]
        if USE_TF32:
            b_u = tl.dot(b_Ai.to(tl.bfloat16), b_vb.to(tl.bfloat16))
        else:
            b_u = tl.dot(b_Ai.to(tl.float32), b_vb, input_precision="ieee")
        p_u = u_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        tl.store(p_u, b_u.to(u_ptr.dtype.element_ty), mask=m_dp[:, None] & m_v[None, :])


@triton.jit
def _gdp_fwd_h_kernel(
    k_ptr,
    w_ptr,
    u_ptr,
    g_cum_ptr,
    h_ptr,
    h0_ptr,
    ht_ptr,
    v_new_ptr,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Zero-spill forward state scan over NT = cdiv(T_Q, BQ) chunks, storing h [B, NT, H, K, V]
    and v_new [B, T_DP, H, V].
    """
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T_DP, BDP)

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    m_h1 = m_k1[:, None] & m_v[None, :]

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        o_k2 = 64 + tl.arange(0, 64)
        m_k2 = o_k2 < K
        m_h2 = m_k2[:, None] & m_v[None, :]
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        base_h0 = (i_b * H + i_h) * K * V
        b_h1 += tl.load(h0_ptr + base_h0 + o_k1[:, None] * V + o_v[None, :], mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            b_h2 += tl.load(h0_ptr + base_h0 + o_k2[:, None] * V + o_v[None, :], mask=m_h2, other=0.0).to(tl.float32)

    o_dp = tl.arange(0, BT)
    for i_t in range(NT):
        base_h = ((i_b * NT + i_t) * H + i_h) * K * V
        tl.store(h_ptr + base_h + o_k1[:, None] * V + o_v[None, :], b_h1.to(h_ptr.dtype.element_ty), mask=m_h1)
        if K > 64:
            tl.store(h_ptr + base_h + o_k2[:, None] * V + o_v[None, :], b_h2.to(h_ptr.dtype.element_ty), mask=m_h2)

        dp_idx = i_t * BDP + o_dp
        m_dp = (o_dp < BDP) & (dp_idx < T_DP)

        p_u = u_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        b_v_new = tl.load(p_u, mask=m_dp[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        p_w1 = w_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k1[None, :]
        b_w1 = tl.load(p_w1, mask=m_dp[:, None] & m_k1[None, :], other=0.0)
        if USE_TF32:
            b_v_new -= tl.dot(b_w1.to(tl.bfloat16), b_h1.to(tl.bfloat16))
        else:
            b_v_new -= tl.dot(b_w1.to(tl.float32), b_h1, input_precision="ieee")

        if K > 64:
            p_w2 = w_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k2[None, :]
            b_w2 = tl.load(p_w2, mask=m_dp[:, None] & m_k2[None, :], other=0.0)
            if USE_TF32:
                b_v_new -= tl.dot(b_w2.to(tl.bfloat16), b_h2.to(tl.bfloat16))
            else:
                b_v_new -= tl.dot(b_w2.to(tl.float32), b_h2, input_precision="ieee")

        p_vn = v_new_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        tl.store(p_vn, b_v_new.to(v_new_ptr.dtype.element_ty), mask=m_dp[:, None] & m_v[None, :])

        last_dp = tl.minimum((i_t + 1) * BDP, T_DP) - 1
        b_g_last = tl.load(g_cum_ptr + (i_b * T_DP + last_dp) * H + i_h).to(tl.float32)
        b_g_dp = tl.load(g_cum_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
        b_decay = tl.where(m_dp, tl.exp(b_g_last - b_g_dp), 0.0)
        b_vn_decayed = b_v_new * b_decay[:, None]

        b_g_last_exp = tl.exp(b_g_last)
        b_h1 = b_h1 * b_g_last_exp
        p_k1 = k_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * K + o_k1[:, None]
        b_k1 = tl.load(p_k1, mask=m_k1[:, None] & m_dp[None, :], other=0.0)
        if USE_TF32:
            b_h1 = tl.dot(b_k1.to(tl.bfloat16), b_vn_decayed.to(tl.bfloat16), b_h1)
        else:
            b_h1 = tl.dot(b_k1.to(tl.float32), b_vn_decayed, b_h1, input_precision="ieee")

        if K > 64:
            b_h2 = b_h2 * b_g_last_exp
            p_k2 = k_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * K + o_k2[:, None]
            b_k2 = tl.load(p_k2, mask=m_k2[:, None] & m_dp[None, :], other=0.0)
            if USE_TF32:
                b_h2 = tl.dot(b_k2.to(tl.bfloat16), b_vn_decayed.to(tl.bfloat16), b_h2)
            else:
                b_h2 = tl.dot(b_k2.to(tl.float32), b_vn_decayed, b_h2, input_precision="ieee")

    if STORE_FINAL_STATE:
        base_ht = (i_b * H + i_h) * K * V
        tl.store(ht_ptr + base_ht + o_k1[:, None] * V + o_v[None, :], b_h1, mask=m_h1)
        if K > 64:
            tl.store(ht_ptr + base_ht + o_k2[:, None] * V + o_v[None, :], b_h2, mask=m_h2)


@triton.jit
def _gdp_fwd_o_rect_kernel(
    q_ptr,
    k_ptr,
    v_new_ptr,
    h_ptr,
    g_cum_ptr,
    o_ptr,
    scale,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BQ_PAD: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Rectangular Query-Compacted `[BQ_PAD, 64]` Readout Kernel (0 Spills vs Upstream's 46 Spills).
    Loads `q` (`[BQ_PAD, BK]`) ONCE per `i_k` tile and computes a single `[BQ_PAD, 64]` attention
    matrix `b_Aqk` across all `NH` Householder updates of the chunk.
    """
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T_DP, BDP)

    o_q = tl.arange(0, BQ_PAD)
    o_dp = tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)

    r_start = i_t * BDP
    first_off = ((NH - 1) - (r_start % NH)) % NH
    q_dp_off = first_off + o_q * NH
    q_idx = (r_start + q_dp_off) // NH
    dp_idx = r_start + o_dp
    m_q = (q_dp_off < BDP) & (q_idx < T_Q)
    m_dp = (o_dp < BDP) & (dp_idx < T_DP)
    m_v = o_v < V

    b_o = tl.zeros([BQ_PAD, BV], dtype=tl.float32)
    b_Aqk = tl.zeros([BQ_PAD, BT], dtype=tl.float32)

    base_h = ((i_b * NT + i_t) * H + i_h) * K * V
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * K + o_k[None, :]
        p_k = k_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * K + o_k[:, None]
        p_h = h_ptr + base_h + o_k[:, None] * V + o_v[None, :]

        b_q = tl.load(p_q, mask=m_q[:, None] & m_k[None, :], other=0.0)
        b_k = tl.load(p_k, mask=m_k[:, None] & m_dp[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)

        if USE_TF32:
            b_o = tl.dot(b_q.to(tl.bfloat16), b_h.to(tl.bfloat16), b_o)
            b_Aqk = tl.dot(b_q.to(tl.bfloat16), b_k.to(tl.bfloat16), b_Aqk)
        else:
            b_o = tl.dot(b_q.to(tl.float32), b_h.to(tl.float32), b_o, input_precision="ieee")
            b_Aqk = tl.dot(b_q.to(tl.float32), b_k.to(tl.float32), b_Aqk, input_precision="ieee")

    b_g_q = tl.load(g_cum_ptr + (i_b * T_DP + r_start + q_dp_off) * H + i_h, mask=m_q, other=0.0).to(tl.float32)
    b_g_dp = tl.load(g_cum_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)

    mask_rect = (q_dp_off[:, None] >= o_dp[None, :]) & m_q[:, None] & m_dp[None, :]
    diff_g = tl.where(mask_rect, b_g_q[:, None] - b_g_dp[None, :], 0.0)
    b_Aqk = tl.where(mask_rect, b_Aqk * tl.exp(diff_g), 0.0)

    tl.debug_barrier()
    p_vn = v_new_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
    b_vn = tl.load(p_vn, mask=m_dp[:, None] & m_v[None, :], other=0.0)

    b_o = b_o * tl.exp(b_g_q)[:, None]
    if USE_TF32:
        b_o = tl.dot(b_Aqk.to(tl.bfloat16), b_vn.to(tl.bfloat16), b_o)
    else:
        b_o = tl.dot(b_Aqk, b_vn.to(tl.float32), b_o, input_precision="ieee")

    b_o = b_o * scale
    p_o = o_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * V + o_v[None, :]
    tl.store(p_o, b_o.to(o_ptr.dtype.element_ty), mask=m_q[:, None] & m_v[None, :])


# ==============================================================================
# Section 4: Native Stride-NH Backward Kernels (Solving Upstream's # TODO)
# ==============================================================================

@triton.jit
def _gdp_bwd_dv_local_rect_kernel(
    q_ptr,
    k_ptr,
    v_new_ptr,
    g_cum_ptr,
    do_ptr,
    dv_local_ptr,
    ds_ptr,
    scale,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BQ_PAD: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Computes dv_local = Aqk^T @ do and ds = tril_rect(do @ v_new^T * exp(g_q - g_dp) * scale)
    once per chunk using rectangular `[BQ_PAD, 64]` tiles without zero-padding q or do.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_q = tl.arange(0, BQ_PAD)
    o_dp = tl.arange(0, BT)
    r_start = i_t * BDP
    first_off = ((NH - 1) - (r_start % NH)) % NH
    q_dp_off = first_off + o_q * NH
    q_idx = (r_start + q_dp_off) // NH
    dp_idx = r_start + o_dp
    m_q = (q_dp_off < BDP) & (q_idx < T_Q)
    m_dp = (o_dp < BDP) & (dp_idx < T_DP)

    b_Aqk = tl.zeros([BQ_PAD, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * K + o_k[None, :]
        p_k = k_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * K + o_k[:, None]
        b_q = tl.load(p_q, mask=m_q[:, None] & m_k[None, :], other=0.0)
        b_k = tl.load(p_k, mask=m_k[:, None] & m_dp[None, :], other=0.0)
        if USE_TF32:
            b_Aqk = tl.dot(b_q.to(tl.bfloat16), b_k.to(tl.bfloat16), b_Aqk)
        else:
            b_Aqk = tl.dot(b_q.to(tl.float32), b_k.to(tl.float32), b_Aqk, input_precision="ieee")

    b_g_q = tl.load(g_cum_ptr + (i_b * T_DP + r_start + q_dp_off) * H + i_h, mask=m_q, other=0.0).to(tl.float32)
    b_g_dp = tl.load(g_cum_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)

    mask_rect = (q_dp_off[:, None] >= o_dp[None, :]) & m_q[:, None] & m_dp[None, :]
    diff_g = tl.where(mask_rect, b_g_q[:, None] - b_g_dp[None, :], 0.0)
    b_scale_rect = tl.where(mask_rect, tl.exp(diff_g) * scale, 0.0)
    b_Aqk_t = tl.trans(b_Aqk * b_scale_rect)

    tl.debug_barrier()
    b_ds = tl.zeros([BQ_PAD, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_do = do_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * V + o_v[None, :]
        p_vn_t = v_new_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * V + o_v[:, None]
        b_do = tl.load(p_do, mask=m_q[:, None] & m_v[None, :], other=0.0)
        b_vn_t = tl.load(p_vn_t, mask=m_v[:, None] & m_dp[None, :], other=0.0)
        if USE_TF32:
            b_dv = tl.dot(b_Aqk_t.to(tl.bfloat16), b_do.to(tl.bfloat16))
            b_ds = tl.dot(b_do.to(tl.bfloat16), b_vn_t.to(tl.bfloat16), b_ds)
        else:
            b_dv = tl.dot(b_Aqk_t, b_do.to(tl.float32), input_precision="ieee")
            b_ds = tl.dot(b_do.to(tl.float32), b_vn_t.to(tl.float32), b_ds, input_precision="ieee")
        p_dv = dv_local_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        tl.store(p_dv, b_dv.to(dv_local_ptr.dtype.element_ty), mask=m_dp[:, None] & m_v[None, :])

    b_ds = b_ds * b_scale_rect
    p_ds = ds_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * BT + o_dp[None, :]
    tl.store(p_ds, b_ds.to(ds_ptr.dtype.element_ty), mask=m_q[:, None] & (o_dp[None, :] < BT))


@triton.jit
def _gdp_bwd_dhu_rect_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    g_cum_ptr,
    do_ptr,
    dv_local_ptr,
    dht_ptr,
    dh0_ptr,
    dh_ptr,
    dv_new_ptr,
    scale,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BQ_PAD: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRAD: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Zero-spill reverse state scan with rectangular `[BQ_PAD, BV]` `q, do` update.
    """
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T_DP, BDP)

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    m_h1 = m_k1[:, None] & m_v[None, :]

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        o_k2 = 64 + tl.arange(0, 64)
        m_k2 = o_k2 < K
        m_h2 = m_k2[:, None] & m_v[None, :]
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_FINAL_STATE_GRAD:
        base_dht = (i_b * H + i_h) * K * V
        b_dh1 += tl.load(dht_ptr + base_dht + o_k1[:, None] * V + o_v[None, :], mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            b_dh2 += tl.load(dht_ptr + base_dht + o_k2[:, None] * V + o_v[None, :], mask=m_h2, other=0.0).to(tl.float32)

    o_q = tl.arange(0, BQ_PAD)
    o_dp = tl.arange(0, BT)

    for i_t in range(NT - 1, -1, -1):
        base_dh = ((i_b * NT + i_t) * H + i_h) * K * V
        tl.store(dh_ptr + base_dh + o_k1[:, None] * V + o_v[None, :], b_dh1.to(dh_ptr.dtype.element_ty), mask=m_h1)
        if K > 64:
            tl.store(dh_ptr + base_dh + o_k2[:, None] * V + o_v[None, :], b_dh2.to(dh_ptr.dtype.element_ty), mask=m_h2)

        r_start = i_t * BDP
        first_off = ((NH - 1) - (r_start % NH)) % NH
        q_dp_off = first_off + o_q * NH
        q_idx = (r_start + q_dp_off) // NH
        dp_idx = r_start + o_dp
        m_q = (q_dp_off < BDP) & (q_idx < T_Q)
        m_dp = (o_dp < BDP) & (dp_idx < T_DP)

        last_dp = tl.minimum((i_t + 1) * BDP, T_DP) - 1
        b_g_last = tl.load(g_cum_ptr + (i_b * T_DP + last_dp) * H + i_h).to(tl.float32)
        b_g_dp = tl.load(g_cum_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
        b_decay_end = tl.where(m_dp, tl.exp(b_g_last - b_g_dp), 0.0)

        p_k1 = k_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k1[None, :]
        b_k1 = tl.load(p_k1, mask=m_dp[:, None] & m_k1[None, :], other=0.0)
        if USE_TF32:
            b_dv = tl.dot(b_k1.to(tl.bfloat16), b_dh1.to(tl.bfloat16))
        else:
            b_dv = tl.dot(b_k1.to(tl.float32), b_dh1, input_precision="ieee")

        if K > 64:
            p_k2 = k_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k2[None, :]
            b_k2 = tl.load(p_k2, mask=m_dp[:, None] & m_k2[None, :], other=0.0)
            if USE_TF32:
                b_dv = tl.dot(b_k2.to(tl.bfloat16), b_dh2.to(tl.bfloat16), b_dv)
            else:
                b_dv = tl.dot(b_k2.to(tl.float32), b_dh2, b_dv, input_precision="ieee")

        p_dvl = dv_local_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        b_dvl = tl.load(p_dvl, mask=m_dp[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_dv = b_dv * b_decay_end[:, None] + b_dvl
        p_dvn = dv_new_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        tl.store(p_dvn, b_dv.to(dv_new_ptr.dtype.element_ty), mask=m_dp[:, None] & m_v[None, :])

        b_g_q = tl.load(g_cum_ptr + (i_b * T_DP + r_start + q_dp_off) * H + i_h, mask=m_q, other=0.0).to(tl.float32)
        b_q_scale = tl.where(m_q, tl.exp(b_g_q) * scale, 0.0)
        p_do = do_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * V + o_v[None, :]
        b_do_scaled = tl.load(p_do, mask=m_q[:, None] & m_v[None, :], other=0.0).to(tl.float32) * b_q_scale[:, None]
        b_ndv = -b_dv

        b_g_last_exp = tl.exp(b_g_last)
        b_dh1 = b_dh1 * b_g_last_exp
        p_q1 = q_ptr + ((i_b * T_Q + q_idx[None, :]) * H + i_h) * K + o_k1[:, None]
        b_q1 = tl.load(p_q1, mask=m_k1[:, None] & m_q[None, :], other=0.0)
        if USE_TF32:
            b_dh1 = tl.dot(b_q1.to(tl.bfloat16), b_do_scaled.to(tl.bfloat16), b_dh1)
        else:
            b_dh1 = tl.dot(b_q1.to(tl.float32), b_do_scaled, b_dh1, input_precision="ieee")

        p_w1 = w_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * K + o_k1[:, None]
        b_w1 = tl.load(p_w1, mask=m_k1[:, None] & m_dp[None, :], other=0.0)
        if USE_TF32:
            b_dh1 = tl.dot(b_w1.to(tl.bfloat16), b_ndv.to(tl.bfloat16), b_dh1)
        else:
            b_dh1 = tl.dot(b_w1.to(tl.float32), b_ndv, b_dh1, input_precision="ieee")

        if K > 64:
            b_dh2 = b_dh2 * b_g_last_exp
            p_q2 = q_ptr + ((i_b * T_Q + q_idx[None, :]) * H + i_h) * K + o_k2[:, None]
            b_q2 = tl.load(p_q2, mask=m_k2[:, None] & m_q[None, :], other=0.0)
            if USE_TF32:
                b_dh2 = tl.dot(b_q2.to(tl.bfloat16), b_do_scaled.to(tl.bfloat16), b_dh2)
            else:
                b_dh2 = tl.dot(b_q2.to(tl.float32), b_do_scaled, b_dh2, input_precision="ieee")

            p_w2 = w_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * K + o_k2[:, None]
            b_w2 = tl.load(p_w2, mask=m_k2[:, None] & m_dp[None, :], other=0.0)
            if USE_TF32:
                b_dh2 = tl.dot(b_w2.to(tl.bfloat16), b_ndv.to(tl.bfloat16), b_dh2)
            else:
                b_dh2 = tl.dot(b_w2.to(tl.float32), b_ndv, b_dh2, input_precision="ieee")

    if USE_INITIAL_STATE:
        base_dh0 = (i_b * H + i_h) * K * V
        tl.store(dh0_ptr + base_dh0 + o_k1[:, None] * V + o_v[None, :], b_dh1, mask=m_h1)
        if K > 64:
            tl.store(dh0_ptr + base_dh0 + o_k2[:, None] * V + o_v[None, :], b_dh2, mask=m_h2)


@triton.jit
def _gdp_bwd_state_dqkw_pass1_rect_kernel(
    q_ptr,
    k_ptr,
    v_new_ptr,
    g_cum_ptr,
    h_ptr,
    do_ptr,
    dh_ptr,
    dv_new_ptr,
    ds_ptr,
    dq_ptr,
    dk1_ptr,
    dw_ptr,
    dg_q_part_ptr,
    dg_dp1_part_ptr,
    scale,
    B,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BQ_PAD: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Pass 1 of Native Stride-NH Backward:
    Computes `dq` (`[B, T_Q, H, K]`), `dk_1` (`[B, T_DP, H, K]`), `dw` (`[B, T_DP, H, K]`),
    and partial gate gradients `dg_q_part` (`[NK, B, T_Q, H]`) and `dg_dp1_part` (`[NK, B, T_DP, H]`)
    using precomputed rectangular `ds` (`[B, T_Q, H, BT]`).
    """
    i_k = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T_DP, BDP)

    o_q = tl.arange(0, BQ_PAD)
    o_dp = tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)

    r_start = i_t * BDP
    first_off = ((NH - 1) - (r_start % NH)) % NH
    q_dp_off = first_off + o_q * NH
    q_idx = (r_start + q_dp_off) // NH
    dp_idx = r_start + o_dp
    m_q = (q_dp_off < BDP) & (q_idx < T_Q)
    m_dp = (o_dp < BDP) & (dp_idx < T_DP)
    m_k = o_k < K

    b_dq = tl.zeros([BQ_PAD, BK], dtype=tl.float32)
    b_dk_state = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)

    base_h = ((i_b * NT + i_t) * H + i_h) * K * V
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V

        p_do = do_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * V + o_v[None, :]
        p_vn = v_new_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        p_dvn = dv_new_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        p_h_t = h_ptr + base_h + o_k[None, :] * V + o_v[:, None]
        p_dh_t = dh_ptr + base_h + o_k[None, :] * V + o_v[:, None]

        b_do = tl.load(p_do, mask=m_q[:, None] & m_v[None, :], other=0.0)
        b_vn = tl.load(p_vn, mask=m_dp[:, None] & m_v[None, :], other=0.0)
        b_dvn = tl.load(p_dvn, mask=m_dp[:, None] & m_v[None, :], other=0.0)
        b_h_t = tl.load(p_h_t, mask=m_v[:, None] & m_k[None, :], other=0.0)
        b_dh_t = tl.load(p_dh_t, mask=m_v[:, None] & m_k[None, :], other=0.0)

        b_dg_last += tl.sum(b_h_t.to(tl.float32) * b_dh_t.to(tl.float32))
        if USE_TF32:
            b_dq = tl.dot(b_do.to(tl.bfloat16), b_h_t.to(tl.bfloat16), b_dq)
            b_dk_state = tl.dot(b_vn.to(tl.bfloat16), b_dh_t.to(tl.bfloat16), b_dk_state)
            b_dw = tl.dot(b_dvn.to(tl.bfloat16), b_h_t.to(tl.bfloat16), b_dw)
        else:
            b_dq = tl.dot(b_do.to(tl.float32), b_h_t.to(tl.float32), b_dq, input_precision="ieee")
            b_dk_state = tl.dot(b_vn.to(tl.float32), b_dh_t.to(tl.float32), b_dk_state, input_precision="ieee")
            b_dw = tl.dot(b_dvn.to(tl.float32), b_h_t.to(tl.float32), b_dw, input_precision="ieee")

    b_dw = -b_dw
    p_dw = dw_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]
    tl.store(p_dw, b_dw.to(dw_ptr.dtype.element_ty), mask=m_dp[:, None] & m_k[None, :])

    tl.debug_barrier()
    p_q = q_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * K + o_k[None, :]
    p_k = k_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]
    b_q = tl.load(p_q, mask=m_q[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_k = tl.load(p_k, mask=m_dp[:, None] & m_k[None, :], other=0.0).to(tl.float32)

    b_g_q = tl.load(g_cum_ptr + (i_b * T_DP + r_start + q_dp_off) * H + i_h, mask=m_q, other=0.0).to(tl.float32)
    b_g_dp = tl.load(g_cum_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    last_dp = tl.minimum((i_t + 1) * BDP, T_DP) - 1
    b_g_last = tl.load(g_cum_ptr + (i_b * T_DP + last_dp) * H + i_h).to(tl.float32)

    b_dg_last = b_dg_last * tl.exp(b_g_last)
    b_dq = b_dq * (tl.exp(b_g_q) * scale)[:, None]
    b_decay_end = tl.where(m_dp, tl.exp(b_g_last - b_g_dp), 0.0)
    b_dk_state = b_dk_state * b_decay_end[:, None]
    b_dg_last += tl.sum(b_dk_state * b_k)

    p_ds = ds_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * BT + o_dp[None, :]
    b_ds = tl.load(p_ds, mask=m_q[:, None] & m_dp[None, :], other=0.0).to(tl.float32)

    if USE_TF32:
        b_dq += tl.dot(b_ds.to(tl.bfloat16), b_k.to(tl.bfloat16))
        b_dk_intra = tl.dot(tl.trans(b_ds.to(tl.bfloat16)), b_q.to(tl.bfloat16))
    else:
        b_dq += tl.dot(b_ds, b_k, input_precision="ieee")
        b_dk_intra = tl.dot(tl.trans(b_ds), b_q, input_precision="ieee")

    b_dk1 = b_dk_state + b_dk_intra
    p_dq = dq_ptr + ((i_b * T_Q + q_idx[:, None]) * H + i_h) * K + o_k[None, :]
    p_dk1 = dk1_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]
    tl.store(p_dq, b_dq.to(dq_ptr.dtype.element_ty), mask=m_q[:, None] & m_k[None, :])
    tl.store(p_dk1, b_dk1, mask=m_dp[:, None] & m_k[None, :])

    b_dg_q = tl.sum(b_dq * b_q, axis=1)
    b_dg_dp1 = -tl.sum(b_dk1 * b_k, axis=1) + tl.where(dp_idx == last_dp, b_dg_last, 0.0)

    p_dg_q = dg_q_part_ptr + ((i_k * (B * T_Q) + i_b * T_Q + q_idx) * H + i_h)
    p_dg_dp1 = dg_dp1_part_ptr + ((i_k * (B * T_DP) + i_b * T_DP + dp_idx) * H + i_h)
    tl.store(p_dg_q, b_dg_q, mask=m_q)
    tl.store(p_dg_dp1, b_dg_dp1, mask=m_dp)


@triton.jit
def _gdp_bwd_wy_v_pass2a_kernel(
    v_ptr,
    beta_ptr,
    Ai_ptr,
    du_ptr,
    dv_out_ptr,
    db_v_ptr,
    dAi_v_ptr,
    T_DP,
    H: tl.constexpr,
    V: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Pass 2A of WY Backward: loops over V tiles computing dAi_v = du @ (v * beta)^T,
    dvb = Ai^T @ du, dv = dvb * beta, and db_v = sum(dvb * v, axis=1).
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_dp = tl.arange(0, BT)
    dp_idx = i_t * BDP + o_dp
    m_dp = (o_dp < BDP) & (dp_idx < T_DP)

    b_beta = tl.load(beta_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    p_Ai_t = Ai_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * BT + o_dp[:, None]
    b_Ai_t = tl.load(p_Ai_t, mask=m_dp[:, None] & m_dp[None, :], other=0.0)

    b_dAi = tl.zeros([BT, BT], dtype=tl.float32)
    b_db_v = tl.zeros([BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_v_t = v_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * V + o_v[:, None]
        p_du = du_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        b_v_t = tl.load(p_v_t, mask=m_v[:, None] & m_dp[None, :], other=0.0)
        b_du = tl.load(p_du, mask=m_dp[:, None] & m_v[None, :], other=0.0)

        if USE_TF32:
            b_dAi = tl.dot(b_du.to(tl.bfloat16), b_v_t.to(tl.bfloat16), b_dAi)
            b_dvb = tl.dot(b_Ai_t.to(tl.bfloat16), b_du.to(tl.bfloat16))
        else:
            b_dAi = tl.dot(b_du.to(tl.float32), b_v_t.to(tl.float32), b_dAi, input_precision="ieee")
            b_dvb = tl.dot(b_Ai_t.to(tl.float32), b_du.to(tl.float32), input_precision="ieee")

        b_dv = b_dvb * b_beta[:, None]
        b_db_v += tl.sum(b_dvb * tl.trans(b_v_t.to(tl.float32)), axis=1)
        p_dv = dv_out_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * V + o_v[None, :]
        tl.store(p_dv, b_dv.to(dv_out_ptr.dtype.element_ty), mask=m_dp[:, None] & m_v[None, :])

    b_dAi = b_dAi * b_beta[None, :]
    tl.store(db_v_ptr + (i_b * T_DP + dp_idx) * H + i_h, b_db_v, mask=m_dp)
    p_dAi = dAi_v_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * BT + o_dp[None, :]
    tl.store(p_dAi, b_dAi, mask=m_dp[:, None] & (o_dp[None, :] < BT))


@triton.jit
def _gdp_bwd_wy_k_pass2b_kernel(
    k_ptr,
    beta_ptr,
    g_cum_ptr,
    Ai_ptr,
    dw_ptr,
    dk1_ptr,
    db_v_ptr,
    dg_dp_wy_ptr,
    dAi_io_ptr,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Pass 2B of WY Backward: loops over K tiles accumulating dAi_k into dAi_io,
    adds dk_wy_w into dk1_ptr, adds db_w into db_v_ptr, and writes dg_dp_w into dg_dp_wy_ptr.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_dp = tl.arange(0, BT)
    dp_idx = i_t * BDP + o_dp
    m_dp = (o_dp < BDP) & (dp_idx < T_DP)

    b_beta = tl.load(beta_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    b_g = tl.load(g_cum_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    b_g_exp = tl.exp(b_g)
    b_kb_scale = b_beta * b_g_exp

    p_Ai_t = Ai_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * BT + o_dp[:, None]
    b_Ai_t = tl.load(p_Ai_t, mask=m_dp[:, None] & m_dp[None, :], other=0.0)

    b_dAi_k = tl.zeros([BT, BT], dtype=tl.float32)
    b_db_w = tl.load(db_v_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    b_dg_w = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k_t = k_ptr + ((i_b * T_DP + dp_idx[None, :]) * H + i_h) * K + o_k[:, None]
        p_dw = dw_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]
        p_dk1 = dk1_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]

        b_k_t = tl.load(p_k_t, mask=m_k[:, None] & m_dp[None, :], other=0.0)
        b_dw = tl.load(p_dw, mask=m_dp[:, None] & m_k[None, :], other=0.0)

        if USE_TF32:
            b_dAi_k = tl.dot(b_dw.to(tl.bfloat16), b_k_t.to(tl.bfloat16), b_dAi_k)
            b_dkb = tl.dot(b_Ai_t.to(tl.bfloat16), b_dw.to(tl.bfloat16))
        else:
            b_dAi_k = tl.dot(b_dw.to(tl.float32), b_k_t.to(tl.float32), b_dAi_k, input_precision="ieee")
            b_dkb = tl.dot(b_Ai_t.to(tl.float32), b_dw.to(tl.float32), input_precision="ieee")

        b_k = tl.trans(b_k_t.to(tl.float32))
        b_dot_k = tl.sum(b_dkb * b_k, axis=1)
        b_db_w += b_dot_k * b_g_exp
        b_dg_w += b_dot_k * b_kb_scale
        b_dk1 = tl.load(p_dk1, mask=m_dp[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_dk1 += b_dkb * b_kb_scale[:, None]
        tl.store(p_dk1, b_dk1, mask=m_dp[:, None] & m_k[None, :])

    tl.debug_barrier()
    p_dAi = dAi_io_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * BT + o_dp[None, :]
    b_dAi = tl.load(p_dAi, mask=m_dp[:, None] & m_dp[None, :], other=0.0).to(tl.float32)
    b_dAi += b_dAi_k * b_kb_scale[None, :]
    tl.store(db_v_ptr + (i_b * T_DP + dp_idx) * H + i_h, b_db_w, mask=m_dp)
    tl.store(dg_dp_wy_ptr + (i_b * T_DP + dp_idx) * H + i_h, b_dg_w, mask=m_dp)
    tl.store(p_dAi, b_dAi, mask=m_dp[:, None] & (o_dp[None, :] < BT))


@triton.jit
def _gdp_bwd_wy_dA_pass2c_kernel(
    k_ptr,
    beta_ptr,
    g_cum_ptr,
    dA_ptr,
    dk1_ptr,
    db_io_ptr,
    dk_out_ptr,
    db_out_ptr,
    dg_dp_wy_ptr,
    T_Q,
    T_DP,
    H: tl.constexpr,
    K: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BDP: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Pass 2C of WY Backward: given dA = strictTril(-Ai^T @ strictTril(dAi) @ Ai^T),
    computes dk_A, db_A, and dg_dp_A using the 1-matrix identity without forming K @ K^T.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_dp = tl.arange(0, BT)
    dp_idx = i_t * BDP + o_dp
    m_dp = (o_dp < BDP) & (dp_idx < T_DP)

    b_beta = tl.load(beta_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    b_g = tl.load(g_cum_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)

    strict_tril = (o_dp[:, None] > o_dp[None, :]) & m_dp[:, None] & m_dp[None, :]
    p_dA = dA_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * BT + o_dp[None, :]
    b_dA = tl.load(p_dA, mask=strict_tril, other=0.0).to(tl.float32)

    diff_g = tl.where(strict_tril, b_g[:, None] - b_g[None, :], 0.0)
    b_dA_exp = tl.where(strict_tril, b_dA * tl.exp(diff_g), 0.0)
    b_dA_scaled_t = tl.trans(b_dA_exp * b_beta[:, None])

    b_db_A = tl.zeros([BT], dtype=tl.float32)
    b_dg_A = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = k_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]
        p_dk1 = dk1_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]
        p_dk_out = dk_out_ptr + ((i_b * T_DP + dp_idx[:, None]) * H + i_h) * K + o_k[None, :]

        b_k = tl.load(p_k, mask=m_dp[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_dk1 = tl.load(p_dk1, mask=m_dp[:, None] & m_k[None, :], other=0.0).to(tl.float32)

        if USE_TF32:
            b_u_row = tl.dot(b_dA_exp.to(tl.bfloat16), b_k.to(tl.bfloat16))
            b_dk_col = tl.dot(b_dA_scaled_t.to(tl.bfloat16), b_k.to(tl.bfloat16))
        else:
            b_u_row = tl.dot(b_dA_exp, b_k, input_precision="ieee")
            b_dk_col = tl.dot(b_dA_scaled_t, b_k, input_precision="ieee")

        b_dk_row = b_u_row * b_beta[:, None]
        b_db_A += tl.sum(b_u_row * b_k, axis=1)
        b_dg_A += tl.sum((b_dk_row - b_dk_col) * b_k, axis=1)
        b_dk_final = b_dk1 + b_dk_row + b_dk_col
        tl.store(p_dk_out, b_dk_final.to(dk_out_ptr.dtype.element_ty), mask=m_dp[:, None] & m_k[None, :])

    b_db_prev = tl.load(db_io_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    b_db_final = b_db_prev + b_db_A
    tl.store(db_out_ptr + (i_b * T_DP + dp_idx) * H + i_h, b_db_final.to(db_out_ptr.dtype.element_ty), mask=m_dp)

    b_dg_prev = tl.load(dg_dp_wy_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    tl.store(dg_dp_wy_ptr + (i_b * T_DP + dp_idx) * H + i_h, b_dg_prev + b_dg_A, mask=m_dp)


@triton.jit
def _gdp_bwd_reduce_dg_rev_cumsum_clean_kernel(
    dg_q_part_ptr,
    dg_dp1_part_ptr,
    dg_dp_wy_ptr,
    dg_out_ptr,
    B,
    T_Q,
    T_DP,
    H: tl.constexpr,
    NK: tl.constexpr,
    NH: tl.constexpr,
    BQ: tl.constexpr,
    BQ_PAD: tl.constexpr,
):
    """
    Universal BT=64 reverse cumsum over virtual chunk [i_t * 64 .. i_t * 64 + 63],
    storing the total reverse-accumulated gate gradient at each g0 step (dp_idx % NH == 0).
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    dp_idx = i_t * 64 + tl.arange(0, 64)
    m_dp = dp_idx < T_DP
    is_q = ((dp_idx % NH) == (NH - 1)) & m_dp
    is_g0 = ((dp_idx % NH) == 0) & m_dp
    q_idx = dp_idx // NH

    b_dg = tl.load(dg_dp_wy_ptr + (i_b * T_DP + dp_idx) * H + i_h, mask=m_dp, other=0.0).to(tl.float32)
    for ik in range(NK):
        off_dp1 = ((ik * B + i_b) * T_DP + dp_idx) * H + i_h
        off_q = ((ik * B + i_b) * T_Q + q_idx) * H + i_h
        b_dg += tl.load(dg_dp1_part_ptr + off_dp1, mask=m_dp, other=0.0).to(tl.float32)
        b_dg += tl.load(dg_q_part_ptr + off_q, mask=is_q, other=0.0).to(tl.float32)

    b_dg = tl.where(m_dp, b_dg, 0.0)
    b_total = tl.sum(b_dg, axis=0)
    b_prefix = tl.cumsum(b_dg, axis=0)
    b_rev = b_total - b_prefix + b_dg

    off_out = (i_b * T_Q + q_idx) * H + i_h
    tl.store(dg_out_ptr + off_out, b_rev.to(dg_out_ptr.dtype.element_ty), mask=is_g0)


# ==============================================================================
# Section 5: Superfused Periphery Kernels (L2Norm, ShortConv1d+SiLU, Beta+G, RMSNormGated)
# ==============================================================================

@triton.jit
def _gdp_l2norm_fwd_kernel(
    x_ptr,
    y_ptr,
    rstd_ptr,
    eps,
    N_ROWS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    i_blk = tl.program_id(0)
    o_r = i_blk * BR + tl.arange(0, BR)
    o_d = tl.arange(0, BD)
    m_r = o_r < N_ROWS
    m_d = o_d < D
    mask = m_r[:, None] & m_d[None, :]
    b_x = tl.load(x_ptr + o_r[:, None] * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1) + eps
    b_rstd = tl.rsqrt(b_var)
    b_y = b_x * b_rstd[:, None]
    tl.store(y_ptr + o_r[:, None] * D + o_d[None, :], b_y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + o_r, b_rstd, mask=m_r)


@triton.jit
def _gdp_l2norm_bwd_kernel(
    x_ptr,
    rstd_ptr,
    dy_ptr,
    dx_ptr,
    N_ROWS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    i_blk = tl.program_id(0)
    o_r = i_blk * BR + tl.arange(0, BR)
    o_d = tl.arange(0, BD)
    m_r = o_r < N_ROWS
    m_d = o_d < D
    mask = m_r[:, None] & m_d[None, :]
    b_x = tl.load(x_ptr + o_r[:, None] * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
    b_dy = tl.load(dy_ptr + o_r[:, None] * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd_ptr + o_r, mask=m_r, other=0.0).to(tl.float32)
    b_y = b_x * b_rstd[:, None]
    b_proj = tl.sum(b_dy * b_y, axis=1)
    b_dx = (b_dy - b_y * b_proj[:, None]) * b_rstd[:, None]
    tl.store(dx_ptr + o_r[:, None] * D + o_d[None, :], b_dx.to(dx_ptr.dtype.element_ty), mask=mask)


def gdp_l2norm_fwd(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    x_c = x.contiguous()
    D = x_c.shape[-1]
    N_ROWS = x_c.numel() // D
    BD = max(triton.next_power_of_2(D), 16)
    BR = 32
    y = torch.empty_like(x_c)
    rstd = torch.empty((N_ROWS,), dtype=torch.float32, device=x.device)
    _gdp_l2norm_fwd_kernel[(triton.cdiv(N_ROWS, BR),)](
        x_c, y, rstd, eps, N_ROWS, D=D, BD=BD, BR=BR, num_warps=4, num_stages=1
    )
    return y, rstd.view(x_c.shape[:-1])


def gdp_l2norm_bwd(x: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    dy_c = dy.contiguous()
    D = x_c.shape[-1]
    N_ROWS = x_c.numel() // D
    BD = max(triton.next_power_of_2(D), 16)
    BR = 32
    dx = torch.empty_like(x_c)
    _gdp_l2norm_bwd_kernel[(triton.cdiv(N_ROWS, BR),)](
        x_c, rstd.contiguous(), dy_c, dx, N_ROWS, D=D, BD=BD, BR=BR, num_warps=4, num_stages=1
    )
    return dx


@triton.jit
def _gdp_short_conv1d_k4_silu_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0)
    i_t = tl.program_id(1)
    i_b = tl.program_id(2)

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t * BT + tl.arange(0, BT)
    m_d = o_d < D
    m_t = o_t < T

    w0 = tl.load(w_ptr + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)

    base_b = i_b * T * D
    t0 = o_t - 3
    t1 = o_t - 2
    t2 = o_t - 1
    t3 = o_t

    x0 = tl.load(x_ptr + base_b + t0[:, None] * D + o_d[None, :], mask=(t0[:, None] >= 0) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base_b + t1[:, None] * D + o_d[None, :], mask=(t1[:, None] >= 0) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base_b + t2[:, None] * D + o_d[None, :], mask=(t2[:, None] >= 0) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + base_b + t3[:, None] * D + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)

    acc = x0 * w0[None, :] + x1 * w1[None, :] + x2 * w2[None, :] + x3 * w3[None, :]
    sig = tl.sigmoid(acc)
    out = acc * sig
    tl.store(y_ptr + base_b + o_t[:, None] * D + o_d[None, :], out.to(y_ptr.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])


@triton.jit
def _gdp_short_conv1d_k4_silu_bwd_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    dx_ptr,
    dw_part_ptr,
    B,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0)
    i_t = tl.program_id(1)
    i_b = tl.program_id(2)
    NT = tl.cdiv(T, BT)

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t * BT + tl.arange(0, BT)
    m_d = o_d < D
    m_t = o_t < T

    w0 = tl.load(w_ptr + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)

    base_b = i_b * T * D
    xm3 = tl.load(x_ptr + base_b + (o_t - 3)[:, None] * D + o_d[None, :], mask=((o_t - 3)[:, None] >= 0) & ((o_t - 3)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)
    xm2 = tl.load(x_ptr + base_b + (o_t - 2)[:, None] * D + o_d[None, :], mask=((o_t - 2)[:, None] >= 0) & ((o_t - 2)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)
    xm1 = tl.load(x_ptr + base_b + (o_t - 1)[:, None] * D + o_d[None, :], mask=((o_t - 1)[:, None] >= 0) & ((o_t - 1)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + base_b + o_t[:, None] * D + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    xp1 = tl.load(x_ptr + base_b + (o_t + 1)[:, None] * D + o_d[None, :], mask=((o_t + 1)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)
    xp2 = tl.load(x_ptr + base_b + (o_t + 2)[:, None] * D + o_d[None, :], mask=((o_t + 2)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)
    xp3 = tl.load(x_ptr + base_b + (o_t + 3)[:, None] * D + o_d[None, :], mask=((o_t + 3)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)

    dy0 = tl.load(dy_ptr + base_b + o_t[:, None] * D + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    dyp1 = tl.load(dy_ptr + base_b + (o_t + 1)[:, None] * D + o_d[None, :], mask=((o_t + 1)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)
    dyp2 = tl.load(dy_ptr + base_b + (o_t + 2)[:, None] * D + o_d[None, :], mask=((o_t + 2)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)
    dyp3 = tl.load(dy_ptr + base_b + (o_t + 3)[:, None] * D + o_d[None, :], mask=((o_t + 3)[:, None] < T) & m_d[None, :], other=0.0).to(tl.float32)

    acc0 = xm3 * w0[None, :] + xm2 * w1[None, :] + xm1 * w2[None, :] + x0 * w3[None, :]
    sig0 = tl.sigmoid(acc0)
    dz0 = tl.where(m_t[:, None], dy0 * sig0 * (1.0 + acc0 * (1.0 - sig0)), 0.0)

    acc1 = xm2 * w0[None, :] + xm1 * w1[None, :] + x0 * w2[None, :] + xp1 * w3[None, :]
    sig1 = tl.sigmoid(acc1)
    dz1 = tl.where((o_t + 1)[:, None] < T, dyp1 * sig1 * (1.0 + acc1 * (1.0 - sig1)), 0.0)

    acc2 = xm1 * w0[None, :] + x0 * w1[None, :] + xp1 * w2[None, :] + xp2 * w3[None, :]
    sig2 = tl.sigmoid(acc2)
    dz2 = tl.where((o_t + 2)[:, None] < T, dyp2 * sig2 * (1.0 + acc2 * (1.0 - sig2)), 0.0)

    acc3 = x0 * w0[None, :] + xp1 * w1[None, :] + xp2 * w2[None, :] + xp3 * w3[None, :]
    sig3 = tl.sigmoid(acc3)
    dz3 = tl.where((o_t + 3)[:, None] < T, dyp3 * sig3 * (1.0 + acc3 * (1.0 - sig3)), 0.0)

    dx = dz0 * w3[None, :] + dz1 * w2[None, :] + dz2 * w1[None, :] + dz3 * w0[None, :]
    tl.store(dx_ptr + base_b + o_t[:, None] * D + o_d[None, :], dx.to(dx_ptr.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])

    dw0 = tl.sum(dz0 * xm3, axis=0)
    dw1 = tl.sum(dz0 * xm2, axis=0)
    dw2 = tl.sum(dz0 * xm1, axis=0)
    dw3 = tl.sum(dz0 * x0, axis=0)

    part_idx = i_b * NT + i_t
    base_dw = part_idx * D * 4 + o_d * 4
    tl.store(dw_part_ptr + base_dw + 0, dw0, mask=m_d)
    tl.store(dw_part_ptr + base_dw + 1, dw1, mask=m_d)
    tl.store(dw_part_ptr + base_dw + 2, dw2, mask=m_d)
    tl.store(dw_part_ptr + base_dw + 3, dw3, mask=m_d)


class _GDPShortConv1dSiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_c = x.contiguous()
        w_c = weight.squeeze(1).contiguous()
        y = torch.empty_like(x_c)
        BT, BD = 32, 32
        grid = (triton.cdiv(D, BD), triton.cdiv(T, BT), B)
        _gdp_short_conv1d_k4_silu_fwd_kernel[grid](
            x_c, w_c, y, T, D=D, BT=BT, BD=BD, num_warps=4, num_stages=1
        )
        ctx.save_for_backward(x_c, w_c)
        ctx.weight_shape = weight.shape
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, w_c = ctx.saved_tensors
        B, T, D = x_c.shape
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        BT, BD = 32, 32
        NT = triton.cdiv(T, BT)
        dw_part = torch.empty((B * NT, D, 4), dtype=torch.float32, device=x_c.device)
        grid = (triton.cdiv(D, BD), NT, B)
        _gdp_short_conv1d_k4_silu_bwd_kernel[grid](
            x_c, w_c, dy_c, dx, dw_part, B, T, D=D, BT=BT, BD=BD, num_warps=8, num_stages=1
        )
        dw = dw_part.sum(dim=0).to(w_c.dtype).view(ctx.weight_shape)
        return dx, dw


@triton.jit
def _gdp_fused_beta_g_fwd_kernel(
    b_raw_ptr,
    a_raw_ptr,
    A_log_ptr,
    dt_bias_ptr,
    beta_out_ptr,
    g_out_ptr,
    beta_scale,
    N_TOK,
    H: tl.constexpr,
    NH: tl.constexpr,
    BT: tl.constexpr,
):
    """
    Fuses:
      beta = sigmoid(b_raw) * beta_scale   (shape [B, T, NH * H] -> [B, T * NH, H])
      g    = -exp(A_log) * softplus(a_raw + dt_bias)  (shape [B, T, H])
    """
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < N_TOK

    b_A = tl.exp(tl.load(A_log_ptr + i_h).to(tl.float32))
    b_dtb = tl.load(dt_bias_ptr + i_h).to(tl.float32)
    b_a = tl.load(a_raw_ptr + o_t * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    x_sp = b_a + b_dtb
    sp = tl.where(x_sp > 20.0, x_sp, tl.log(1.0 + tl.exp(x_sp)))
    b_g = -b_A * sp
    tl.store(g_out_ptr + o_t * H + i_h, b_g, mask=m_t)

    for j in range(NH):
        # In GatedDeltaProduct.forward: rearrange(beta, '... t (n h) -> ... (t n) h', n=NH)
        b_val = tl.load(b_raw_ptr + o_t * (NH * H) + j * H + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_beta = tl.sigmoid(b_val) * beta_scale
        tl.store(beta_out_ptr + (o_t * NH + j) * H + i_h, b_beta.to(beta_out_ptr.dtype.element_ty), mask=m_t)


@triton.jit
def _gdp_fused_beta_g_bwd_kernel(
    b_raw_ptr,
    a_raw_ptr,
    A_log_ptr,
    dt_bias_ptr,
    dbeta_ptr,
    dg_ptr,
    db_raw_ptr,
    da_raw_ptr,
    dA_part_ptr,
    ddtb_part_ptr,
    beta_scale,
    N_TOK,
    H: tl.constexpr,
    NH: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < N_TOK

    b_A = tl.exp(tl.load(A_log_ptr + i_h).to(tl.float32))
    b_dtb = tl.load(dt_bias_ptr + i_h).to(tl.float32)
    b_a = tl.load(a_raw_ptr + o_t * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_dg = tl.load(dg_ptr + o_t * H + i_h, mask=m_t, other=0.0).to(tl.float32)

    x_sp = b_a + b_dtb
    sp = tl.where(x_sp > 20.0, x_sp, tl.log(1.0 + tl.exp(x_sp)))
    sig_sp = tl.sigmoid(x_sp)

    b_da = tl.where(m_t, -b_dg * b_A * sig_sp, 0.0)
    b_dA_log = tl.sum(tl.where(m_t, -b_dg * b_A * sp, 0.0), axis=0)
    b_ddtb = tl.sum(b_da, axis=0)

    tl.store(da_raw_ptr + o_t * H + i_h, b_da.to(da_raw_ptr.dtype.element_ty), mask=m_t)
    tl.store(dA_part_ptr + i_t * H + i_h, b_dA_log)
    tl.store(ddtb_part_ptr + i_t * H + i_h, b_ddtb)

    for j in range(NH):
        b_val = tl.load(b_raw_ptr + o_t * (NH * H) + j * H + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_dbeta = tl.load(dbeta_ptr + (o_t * NH + j) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
        sig_b = tl.sigmoid(b_val)
        b_db = b_dbeta * beta_scale * sig_b * (1.0 - sig_b)
        tl.store(db_raw_ptr + o_t * (NH * H) + j * H + i_h, b_db.to(db_raw_ptr.dtype.element_ty), mask=m_t)


class _GDPFusedBetaGFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        b_raw: torch.Tensor,
        a_raw: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_householder: int,
        allow_neg_eigval: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = a_raw.shape
        nh = int(num_householder)
        beta_scale = 2.0 if allow_neg_eigval else 1.0
        b_c = b_raw.contiguous()
        a_c = a_raw.contiguous()
        N_TOK = B * T
        BT = 64
        NT = triton.cdiv(N_TOK, BT)
        beta_out = torch.empty((B, T * nh, H), dtype=b_c.dtype, device=b_c.device)
        g_out = torch.empty((B, T, H), dtype=torch.float32, device=a_c.device)
        _gdp_fused_beta_g_fwd_kernel[(NT, H)](
            b_c, a_c, A_log, dt_bias, beta_out, g_out, beta_scale, N_TOK, H=H, NH=nh, BT=BT, num_warps=2, num_stages=1
        )
        ctx.save_for_backward(b_c, a_c, A_log, dt_bias)
        ctx.nh = nh
        ctx.beta_scale = beta_scale
        return beta_out, g_out

    @staticmethod
    def backward(ctx, dbeta: torch.Tensor, dg: torch.Tensor):
        b_c, a_c, A_log, dt_bias = ctx.saved_tensors
        B, T, H = a_c.shape
        nh = ctx.nh
        N_TOK = B * T
        BT = 64
        NT = triton.cdiv(N_TOK, BT)
        db_raw = torch.empty_like(b_c)
        da_raw = torch.empty_like(a_c)
        dA_part = torch.empty((NT, H), dtype=torch.float32, device=a_c.device)
        ddtb_part = torch.empty((NT, H), dtype=torch.float32, device=a_c.device)
        _gdp_fused_beta_g_bwd_kernel[(NT, H)](
            b_c,
            a_c,
            A_log,
            dt_bias,
            dbeta.contiguous(),
            dg.contiguous(),
            db_raw,
            da_raw,
            dA_part,
            ddtb_part,
            ctx.beta_scale,
            N_TOK,
            H=H,
            NH=nh,
            BT=BT,
            num_warps=2,
            num_stages=1,
        )
        return db_raw, da_raw, dA_part.sum(dim=0).to(A_log.dtype), ddtb_part.sum(dim=0).to(dt_bias.dtype), None, None


@triton.jit
def _gdp_rmsnorm_gated_fwd_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    eps,
    N_ROWS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    i_blk = tl.program_id(0)
    o_r = i_blk * BR + tl.arange(0, BR)
    o_d = tl.arange(0, BD)
    m_r = o_r < N_ROWS
    m_d = o_d < D
    mask = m_r[:, None] & m_d[None, :]
    b_x = tl.load(x_ptr + o_r[:, None] * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
    b_g = tl.load(g_ptr + o_r[:, None] * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
    b_w = tl.load(w_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)

    b_var = tl.sum(b_x * b_x, axis=1) / D + eps
    b_rstd = tl.rsqrt(b_var)
    b_x_hat = b_x * b_rstd[:, None]
    b_silu_g = b_g * tl.sigmoid(b_g)
    b_y = b_x_hat * b_w[None, :] * b_silu_g
    tl.store(y_ptr + o_r[:, None] * D + o_d[None, :], b_y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + o_r, b_rstd, mask=m_r)


@triton.jit
def _gdp_rmsnorm_gated_bwd_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    rstd_ptr,
    dy_ptr,
    dx_ptr,
    dg_ptr,
    dw_part_ptr,
    N_ROWS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    i_blk = tl.program_id(0)
    o_r = i_blk * BR + tl.arange(0, BR)
    o_d = tl.arange(0, BD)
    m_r = o_r < N_ROWS
    m_d = o_d < D
    mask = m_r[:, None] & m_d[None, :]

    b_x = tl.load(x_ptr + o_r[:, None] * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
    b_g = tl.load(g_ptr + o_r[:, None] * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
    b_dy = tl.load(dy_ptr + o_r[:, None] * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
    b_w = tl.load(w_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd_ptr + o_r, mask=m_r, other=0.0).to(tl.float32)

    b_x_hat = b_x * b_rstd[:, None]
    sig_g = tl.sigmoid(b_g)
    silu_g = b_g * sig_g
    dsilu_g = sig_g * (1.0 + b_g * (1.0 - sig_g))

    b_dg = b_dy * b_x_hat * b_w[None, :] * dsilu_g
    b_dy_norm = b_dy * silu_g * b_w[None, :]
    b_proj = tl.sum(b_dy_norm * b_x_hat, axis=1) / D
    b_dx = (b_dy_norm - b_x_hat * b_proj[:, None]) * b_rstd[:, None]
    b_dw = tl.sum(tl.where(mask, b_dy * silu_g * b_x_hat, 0.0), axis=0)

    tl.store(dx_ptr + o_r[:, None] * D + o_d[None, :], b_dx.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dg_ptr + o_r[:, None] * D + o_d[None, :], b_dg.to(dg_ptr.dtype.element_ty), mask=mask)
    tl.store(dw_part_ptr + i_blk * D + o_d, b_dw, mask=m_d)


class _GDPRMSNormGatedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x_c = x.contiguous()
        g_c = g.contiguous()
        D = x_c.shape[-1]
        N_ROWS = x_c.numel() // D
        BD = max(triton.next_power_of_2(D), 16)
        BR = 16
        y = torch.empty_like(x_c)
        rstd = torch.empty((N_ROWS,), dtype=torch.float32, device=x.device)
        _gdp_rmsnorm_gated_fwd_kernel[(triton.cdiv(N_ROWS, BR),)](
            x_c, g_c, weight, y, rstd, eps, N_ROWS, D=D, BD=BD, BR=BR, num_warps=4, num_stages=1
        )
        ctx.save_for_backward(x_c, g_c, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, g_c, weight, rstd = ctx.saved_tensors
        dy_c = dy.contiguous()
        D = x_c.shape[-1]
        N_ROWS = x_c.numel() // D
        BD = max(triton.next_power_of_2(D), 16)
        BR = 16
        N_BLK = triton.cdiv(N_ROWS, BR)
        dx = torch.empty_like(x_c)
        dg = torch.empty_like(g_c)
        dw_part = torch.empty((N_BLK, D), dtype=torch.float32, device=x_c.device)
        _gdp_rmsnorm_gated_bwd_kernel[(N_BLK,)](
            x_c, g_c, weight, rstd, dy_c, dx, dg, dw_part, N_ROWS, D=D, BD=BD, BR=BR, num_warps=4, num_stages=1
        )
        return dx, dg, dw_part.sum(dim=0).to(weight.dtype), None


# ==============================================================================
# Section 6: Autograd Function & Public API for `chunk_gated_delta_product_rsi`
# ==============================================================================

class _ChunkGatedDeltaProductRSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        num_householder: int,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if use_qk_l2norm_in_kernel:
            q_orig, k_orig = q, k
            q, q_rstd = gdp_l2norm_fwd(q)
            k, k_rstd = gdp_l2norm_fwd(k)
        else:
            q_orig, k_orig, q_rstd, k_rstd = None, None, None, None

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()

        B, T_Q, H, K = q.shape
        V = v.shape[-1]
        nh = int(num_householder)
        T_DP = T_Q * nh
        BQ, BDP, BQ_PAD = _gdp_chunk_params(nh)
        BT = 64
        NT = triton.cdiv(T_DP, BDP)
        use_tf32 = q.dtype in (torch.bfloat16, torch.float16)

        g_cum = gdp_chunk_local_cumsum_fwd(g, num_householder=nh, bt=BT)
        A = torch.empty((B, T_DP, H, BT), dtype=torch.float32, device=q.device)
        Ai = torch.empty((B, T_DP, H, BT), dtype=k.dtype, device=q.device)

        nw_fast = 4 if use_tf32 else 8
        ns_fast = 2 if use_tf32 else 1
        bk_fast = 64 if use_tf32 else 32
        bv_fast = 64 if use_tf32 else 32

        _gdp_fwd_kkt_subchunk_kernel[(NT, B * H)](
            k, g_cum, beta, A, T_Q, T_DP,
            H=H, K=K, NH=nh, BQ=BQ, BDP=BDP, BT=BT, BK=bk_fast, USE_TF32=use_tf32,
            num_warps=nw_fast, num_stages=ns_fast,
        )
        _gdp_fwd_solve_tril_16x16_kernel[(NT, B * H)](
            A, Ai, T_DP, H=H, BDP=BDP, BT=BT, USE_TF32=use_tf32,
            num_warps=4, num_stages=1,
        )

        w = torch.empty_like(k)
        u = torch.empty_like(v)
        _gdp_fwd_wy_wu_kernel[(NT, B * H)](
            k, v, beta, g_cum, Ai, w, u, T_Q, T_DP,
            H=H, K=K, V=V, NH=nh, BQ=BQ, BDP=BDP, BT=BT, BK=bk_fast, BV=bv_fast, USE_TF32=use_tf32,
            num_warps=nw_fast, num_stages=ns_fast,
        )

        h = torch.empty((B, NT, H, K, V), dtype=k.dtype, device=q.device)
        v_new = torch.empty_like(v)
        ht = torch.empty((B, H, K, V), dtype=torch.float32, device=q.device) if output_final_state else None
        h0_c = initial_state.contiguous() if initial_state is not None else None

        BV_H = 32
        _gdp_fwd_h_kernel[(triton.cdiv(V, BV_H), B * H)](
            k, w, u, g_cum, h, h0_c, ht, v_new, T_Q, T_DP,
            H=H, K=K, V=V, NH=nh, BQ=BQ, BDP=BDP, BT=BT, BV=BV_H,
            USE_INITIAL_STATE=h0_c is not None,
            STORE_FINAL_STATE=output_final_state,
            USE_TF32=use_tf32,
            num_warps=nw_fast, num_stages=ns_fast,
        )

        o = torch.empty((B, T_Q, H, V), dtype=q.dtype, device=q.device)
        BV_O = 32 if (nh == 1 or not use_tf32) else 64
        nw_o = 8 if (nh == 1 or not use_tf32) else 4
        _gdp_fwd_o_rect_kernel[(triton.cdiv(V, BV_O), NT, B * H)](
            q, k, v_new, h, g_cum, o, float(scale), T_Q, T_DP,
            H=H, K=K, V=V, NH=nh, BQ=BQ, BQ_PAD=BQ_PAD, BDP=BDP, BT=BT, BK=bk_fast, BV=BV_O, USE_TF32=use_tf32,
            num_warps=nw_o, num_stages=ns_fast,
        )

        ctx.save_for_backward(q, k, v, g_cum, beta, Ai, w, v_new, h, h0_c, q_orig, k_orig, q_rstd, k_rstd)
        ctx.nh = nh
        ctx.scale = float(scale)
        ctx.g_dtype = g.dtype
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o, ht

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        q, k, v, g_cum, beta, Ai, w, v_new, h, h0_c, q_orig, k_orig, q_rstd, k_rstd = ctx.saved_tensors
        do = do.contiguous()
        dht_c = dht.contiguous() if dht is not None else None

        B, T_Q, H, K = q.shape
        V = v.shape[-1]
        nh = ctx.nh
        T_DP = T_Q * nh
        BQ, BDP, BQ_PAD = _gdp_chunk_params(nh)
        BT = 64
        NT = triton.cdiv(T_DP, BDP)
        use_tf32 = q.dtype in (torch.bfloat16, torch.float16)
        nw_fast = 4 if use_tf32 else 8
        ns_fast = 2 if use_tf32 else 1
        bk_fast = 64 if use_tf32 else 32

        dv_local = torch.empty_like(v)
        ds = torch.empty((B, T_Q, H, BT), dtype=q.dtype, device=q.device)
        nw_dvl = 8 if (nh == 1 or not use_tf32) else 4
        bv_dvl = 16 if (nh == 1 or not use_tf32) else 32
        _gdp_bwd_dv_local_rect_kernel[(NT, B * H)](
            q, k, v_new, g_cum, do, dv_local, ds, ctx.scale, T_Q, T_DP,
            H=H, K=K, V=V, NH=nh, BQ=BQ, BQ_PAD=BQ_PAD, BDP=BDP, BT=BT,
            BK=bk_fast, BV=bv_dvl, USE_TF32=use_tf32,
            num_warps=nw_dvl, num_stages=1,
        )

        dh = torch.empty_like(h)
        dv_new = torch.empty_like(v)
        dh0 = torch.empty((B, H, K, V), dtype=torch.float32, device=q.device) if h0_c is not None else None
        BV_H = 32 if (nh > 1 and use_tf32) else 16
        nw_dhu = 4 if (nh > 1 and use_tf32) else 8
        ns_dhu = 2 if (nh > 1 and use_tf32) else 1
        _gdp_bwd_dhu_rect_kernel[(triton.cdiv(V, BV_H), B * H)](
            q, k, w, g_cum, do, dv_local, dht_c, dh0, dh, dv_new, ctx.scale, T_Q, T_DP,
            H=H, K=K, V=V, NH=nh, BQ=BQ, BQ_PAD=BQ_PAD, BDP=BDP, BT=BT, BV=BV_H,
            USE_INITIAL_STATE=h0_c is not None,
            USE_FINAL_STATE_GRAD=dht_c is not None,
            USE_TF32=use_tf32,
            num_warps=nw_dhu, num_stages=ns_dhu,
        )

        BK_P1 = 32
        BV_P1 = 32
        NK = triton.cdiv(K, BK_P1)
        dq = torch.empty_like(q)
        dk1 = torch.empty((B, T_DP, H, K), dtype=torch.float32, device=q.device)
        dw = torch.empty_like(k)
        dg_q_part = torch.empty((NK, B, T_Q, H), dtype=torch.float32, device=q.device)
        dg_dp1_part = torch.empty((NK, B, T_DP, H), dtype=torch.float32, device=q.device)

        _gdp_bwd_state_dqkw_pass1_rect_kernel[(NK, NT, B * H)](
            q, k, v_new, g_cum, h, do, dh, dv_new, ds, dq, dk1, dw, dg_q_part, dg_dp1_part, ctx.scale, B, T_Q, T_DP,
            H=H, K=K, V=V, NH=nh, BQ=BQ, BQ_PAD=BQ_PAD, BDP=BDP, BT=BT, BK=BK_P1, BV=BV_P1, USE_TF32=use_tf32,
            num_warps=4, num_stages=1,
        )

        dv = torch.empty_like(v)
        db_tmp = torch.empty((B, T_DP, H), dtype=torch.float32, device=q.device)
        dAi = torch.empty((B, T_DP, H, BT), dtype=torch.float32, device=q.device)
        _gdp_bwd_wy_v_pass2a_kernel[(NT, B * H)](
            v, beta, Ai, dv_new, dv, db_tmp, dAi, T_DP,
            H=H, V=V, BDP=BDP, BT=BT, BV=32 if use_tf32 else 16, USE_TF32=use_tf32,
            num_warps=nw_fast, num_stages=1,
        )

        dg_dp_wy = torch.empty((B, T_DP, H), dtype=torch.float32, device=q.device)
        _gdp_bwd_wy_k_pass2b_kernel[(NT, B * H)](
            k, beta, g_cum, Ai, dw, dk1, db_tmp, dg_dp_wy, dAi, T_Q, T_DP,
            H=H, K=K, NH=nh, BQ=BQ, BDP=BDP, BT=BT, BK=16, USE_TF32=use_tf32,
            num_warps=8, num_stages=1,
        )

        dA = torch.empty_like(dAi)
        _bwd_dA_triple_product_kernel[(NT, B * H)](
            Ai, dAi, dA, T_DP, H=H, BDP=BDP, BT=BT, USE_TF32=use_tf32,
            num_warps=nw_fast, num_stages=1,
        )

        dk = torch.empty_like(k)
        dbeta = torch.empty_like(beta)
        _gdp_bwd_wy_dA_pass2c_kernel[(NT, B * H)](
            k, beta, g_cum, dA, dk1, db_tmp, dk, dbeta, dg_dp_wy, T_Q, T_DP,
            H=H, K=K, NH=nh, BQ=BQ, BDP=BDP, BT=BT, BK=32 if use_tf32 else 16, USE_TF32=use_tf32,
            num_warps=nw_fast, num_stages=1,
        )

        dg = torch.empty((B, T_Q, H), dtype=ctx.g_dtype, device=q.device)
        _gdp_bwd_reduce_dg_rev_cumsum_clean_kernel[(NT, B * H)](
            dg_q_part, dg_dp1_part, dg_dp_wy, dg, B, T_Q, T_DP,
            H=H, NK=NK, NH=nh, BQ=BQ, BQ_PAD=BQ_PAD,
            num_warps=2, num_stages=1,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = gdp_l2norm_bwd(q_orig, q_rstd, dq)
            dk = gdp_l2norm_bwd(k_orig, k_rstd, dk)

        return dq, dk, dv, dg, dbeta, None, None, dh0, None, None


def chunk_gated_delta_product_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int = 2,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _ChunkGatedDeltaProductRSIFunction.apply(
        q, k, v, g, beta, int(num_householder), float(scale), initial_state, output_final_state, use_qk_l2norm_in_kernel
    )


# ==============================================================================
# Section 7: 5-Stage Progressive Layer Ablation (`GatedDeltaProductBlockRSI`)
# ==============================================================================

class GatedDeltaProductBlockRSI(nn.Module):
    """
    5-Stage Progressive Ablation Block for Gated DeltaProduct (`arXiv:2502.10297`):
      - Stage A0: Upstream `GatedDeltaProduct` (`unpad_hidden_states` + `chunk_gated_delta_product`
                  with $N_h \times$ zero-padded backward & spilling kernels).
      - Stage A1: Native dense batching (`cu_seqlens=None` fast path) with upstream `chunk_gated_delta_product`.
      - Stage A2: Round-21 RSI Zero-Spill Rectangular `chunk_gated_delta_product_rsi` (eliminates
                  $N_h \times$ zero-padded `q_new`/`do_new`/`g_interleaved` and backward recomputations).
      - Stage A3: Stage A2 + Fused `ShortConv1d(W=4)+SiLU` (`62` regs, `0` spills) + Vectorized `L2Norm`.
      - Stage A4: Stage A3 + Superfused `(beta, g)` gate activation (`_GDPFusedBetaGFn`) +
                  Superfused `FusedRMSNormGated` (`_GDPRMSNormGatedFn`).
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_v: float = 2.0,
        head_dim: int = 128,
        num_heads: int = 8,
        num_householder: int = 3,
        allow_neg_eigval: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_householder = int(num_householder)
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim * self.num_householder, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim * self.num_householder, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads * self.num_householder, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.q_conv_weight = nn.Parameter(torch.randn(self.key_dim, 1, 4) * 0.02)
        self.k_conv_weight = nn.Parameter(torch.randn(self.key_dim * self.num_householder, 1, 4) * 0.02)
        self.v_conv_weight = nn.Parameter(torch.randn(self.value_dim * self.num_householder, 1, 4) * 0.02)
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))

    @staticmethod
    def _eager_short_conv1d_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        y_t = F.conv1d(F.pad(x_t, (3, 0)), weight, groups=x.shape[-1])
        return F.silu(y_t.transpose(1, 2))

    @staticmethod
    def _eager_rmsnorm_gated(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x_f = x.float()
        g_f = g.float()
        rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x_f * rstd * weight.float() * F.silu(g_f)).to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, stage: str = "A4") -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H = self.num_heads
        nh = self.num_householder

        q_raw = self.q_proj(hidden_states)
        k_raw = self.k_proj(hidden_states)
        v_raw = self.v_proj(hidden_states)
        b_raw = self.b_proj(hidden_states)
        a_raw = self.a_proj(hidden_states)
        g_out_raw = self.g_proj(hidden_states).view(B, T, H, self.head_v_dim)

        if stage in ("A0", "A1", "A2"):
            q_c = self._eager_short_conv1d_silu(q_raw, self.q_conv_weight)
            k_c = self._eager_short_conv1d_silu(k_raw, self.k_conv_weight)
            v_c = self._eager_short_conv1d_silu(v_raw, self.v_conv_weight)
        else:
            q_c = _GDPShortConv1dSiLUFn.apply(q_raw, self.q_conv_weight)
            k_c = _GDPShortConv1dSiLUFn.apply(k_raw, self.k_conv_weight)
            v_c = _GDPShortConv1dSiLUFn.apply(v_raw, self.v_conv_weight)

        q = q_c.view(B, T, H, self.head_k_dim)
        # Match upstream GatedDeltaProduct.forward rearrange: '... t (n h d) -> ... (t n) h d'
        k = k_c.view(B, T, nh, H, self.head_k_dim).reshape(B, T * nh, H, self.head_k_dim)
        v = v_c.view(B, T, nh, H, self.head_v_dim).reshape(B, T * nh, H, self.head_v_dim)

        if stage in ("A0", "A1", "A2", "A3"):
            beta = b_raw.sigmoid()
            if self.allow_neg_eigval:
                beta = beta * 2.0
            beta = beta.view(B, T, nh, H).reshape(B, T * nh, H)
            g = -self.A_log.float().exp() * F.softplus(a_raw.float() + self.dt_bias.float())
        else:
            beta, g = _GDPFusedBetaGFn.apply(
                b_raw, a_raw, self.A_log, self.dt_bias, nh, self.allow_neg_eigval
            )

        if stage in ("A0", "A1"):
            from fla.ops.gated_delta_product import chunk_gated_delta_product
            if stage == "A0":
                cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=hidden_states.device)
                o, _ = chunk_gated_delta_product(
                    q=q.reshape(1, B * T, H, self.head_k_dim),
                    k=k.reshape(1, B * T * nh, H, self.head_k_dim),
                    v=v.reshape(1, B * T * nh, H, self.head_v_dim),
                    g=g.to(q.dtype).reshape(1, B * T, H),
                    beta=beta.reshape(1, B * T * nh, H),
                    num_householder=nh,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
                o = o.view(B, T, H, self.head_v_dim)
            else:
                o, _ = chunk_gated_delta_product(
                    q=q,
                    k=k,
                    v=v,
                    g=g.to(q.dtype),
                    beta=beta,
                    num_householder=nh,
                    use_qk_l2norm_in_kernel=True,
                )
        else:
            o, _ = chunk_gated_delta_product_rsi(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                num_householder=nh,
                use_qk_l2norm_in_kernel=True,
            )

        if stage in ("A0", "A1", "A2", "A3"):
            o_normed = self._eager_rmsnorm_gated(o, g_out_raw, self.o_norm_weight, self.norm_eps)
        else:
            o_normed = _GDPRMSNormGatedFn.apply(o, g_out_raw, self.o_norm_weight, self.norm_eps)

        return self.o_proj(o_normed.reshape(B, T, self.value_dim))


# ==============================================================================
# Section 8: 8-GPU NVSwitch Context Parallelism (`GDP-CP = 8` Ulysses vs Ring)
# ==============================================================================

class _AllToAllSeqToHeadFn(torch.autograd.Function):
    """
    Zero-copy NVSwitch all_to_all_single transposing [B, T_local, H, D] -> [B, T_total, H_local, D].
    """

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


class _AllToAllHeadToSeqFn(torch.autograd.Function):
    """
    Zero-copy NVSwitch all_to_all_single transposing [B, T_total, H_local, D] -> [B, T_local, H, D].
    """

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


def gdp_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    beta_local: torch.Tensor,
    num_householder: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Zero-copy NVSwitch Ulysses Context Parallelism (`GDP-CP = 8`) for Gated DeltaProduct.
    Packs real-token `(q, g)` and virtual-step `(k, v, beta)` into two fused all_to_all_single collectives.
    """
    K = q_local.shape[-1]
    V = v_local.shape[-1]
    qg_pack = torch.cat([q_local, g_local.to(q_local.dtype).unsqueeze(-1)], dim=-1)
    kvb_pack = torch.cat([k_local, v_local, beta_local.to(k_local.dtype).unsqueeze(-1)], dim=-1)

    qg_full = _AllToAllSeqToHeadFn.apply(qg_pack, group)
    kvb_full = _AllToAllSeqToHeadFn.apply(kvb_pack, group)

    q_full, g_full_u = torch.split(qg_full, [K, 1], dim=-1)
    k_full, v_full, beta_full_u = torch.split(kvb_full, [K, V, 1], dim=-1)
    g_full = g_full_u.squeeze(-1).contiguous()
    beta_full = beta_full_u.squeeze(-1).contiguous()

    o_full, _ = chunk_gated_delta_product_rsi(
        q=q_full.contiguous(),
        k=k_full.contiguous(),
        v=v_full.contiguous(),
        g=g_full,
        beta=beta_full,
        num_householder=num_householder,
        use_qk_l2norm_in_kernel=True,
    )
    return _AllToAllHeadToSeqFn.apply(o_full, group)


class _GDPRingCP8Function(torch.autograd.Function):
    """
    8-GPU Sequential Ring State-Passing Baseline (`GDP-Ring-CP = 8`) for Gated DeltaProduct.
    """

    @staticmethod
    def forward(
        ctx,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        g_local: torch.Tensor,
        beta_local: torch.Tensor,
        num_householder: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        B, _, H, K = q_local.shape
        V = v_local.shape[-1]

        h_init_list = [None] * world_size
        o_local = None

        for step in range(world_size):
            if rank == step:
                h_in = h_init_list[step]
                o_step, h_out = chunk_gated_delta_product_rsi(
                    q=q_local,
                    k=k_local,
                    v=v_local,
                    g=g_local,
                    beta=beta_local,
                    num_householder=num_householder,
                    initial_state=h_in,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                o_local = o_step
                if step + 1 < world_size:
                    dist.send(h_out.contiguous(), dst=step + 1, group=group)
            elif rank == step + 1:
                h_recv = torch.empty((B, H, K, V), device=q_local.device, dtype=torch.float32)
                dist.recv(h_recv, src=step, group=group)
                h_init_list[step + 1] = h_recv

        ctx.save_for_backward(
            q_local,
            k_local,
            v_local,
            g_local,
            beta_local,
            h_init_list[rank] if h_init_list[rank] is not None else torch.empty(0, device=q_local.device),
        )
        ctx.nh = num_householder
        ctx.group = group
        ctx.has_init = rank > 0
        return o_local

    @staticmethod
    def backward(ctx, do_local: torch.Tensor):
        q_local, k_local, v_local, g_local, beta_local, h_saved = ctx.saved_tensors
        group = ctx.group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        B, _, H, K = q_local.shape
        V = v_local.shape[-1]

        dh_next = None
        grads = None

        for step in range(world_size - 1, -1, -1):
            if rank == step:
                with torch.enable_grad():
                    q_ = q_local.detach().requires_grad_(True)
                    k_ = k_local.detach().requires_grad_(True)
                    v_ = v_local.detach().requires_grad_(True)
                    g_ = g_local.detach().requires_grad_(True)
                    b_ = beta_local.detach().requires_grad_(True)
                    h_in = h_saved.detach().requires_grad_(True) if ctx.has_init else None

                    o_step, h_out = chunk_gated_delta_product_rsi(
                        q=q_,
                        k=k_,
                        v=v_,
                        g=g_,
                        beta=b_,
                        num_householder=ctx.nh,
                        initial_state=h_in,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    loss = (o_step.float() * do_local.float()).sum()
                    if dh_next is not None:
                        loss = loss + (h_out.float() * dh_next.float()).sum()
                    inputs = (q_, k_, v_, g_, b_) + ((h_in,) if ctx.has_init else ())
                    all_g = torch.autograd.grad(loss, inputs)
                    grads = all_g[:5]
                    if step > 0:
                        dh_prev = all_g[5].contiguous()
                        dist.send(dh_prev, dst=step - 1, group=group)
            elif rank == step - 1:
                dh_recv = torch.empty((B, H, K, V), device=q_local.device, dtype=torch.float32)
                dist.recv(dh_recv, src=step, group=group)
                dh_next = dh_recv

        return (*grads, None, None)


def gdp_cp8_ring_baseline(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    beta_local: torch.Tensor,
    num_householder: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    return _GDPRingCP8Function.apply(
        q_local, k_local, v_local, g_local, beta_local, num_householder, group
    )


# ==============================================================================
# Section 9: SM80 Cubin Register & Spill Audit Utilities
# ==============================================================================

ALL_GDP_RSI_KERNELS = [
    ("_gdp_local_cumsum_fwd_kernel", _gdp_local_cumsum_fwd_kernel),
    ("_bwd_dA_triple_product_kernel", _bwd_dA_triple_product_kernel),
    ("_gdp_fwd_kkt_subchunk_kernel", _gdp_fwd_kkt_subchunk_kernel),
    ("_gdp_fwd_solve_tril_16x16_kernel", _gdp_fwd_solve_tril_16x16_kernel),
    ("_gdp_fwd_wy_wu_kernel", _gdp_fwd_wy_wu_kernel),
    ("_gdp_fwd_h_kernel", _gdp_fwd_h_kernel),
    ("_gdp_fwd_o_rect_kernel", _gdp_fwd_o_rect_kernel),
    ("_gdp_bwd_dv_local_rect_kernel", _gdp_bwd_dv_local_rect_kernel),
    ("_gdp_bwd_dhu_rect_kernel", _gdp_bwd_dhu_rect_kernel),
    ("_gdp_bwd_state_dqkw_pass1_rect_kernel", _gdp_bwd_state_dqkw_pass1_rect_kernel),
    ("_gdp_bwd_wy_v_pass2a_kernel", _gdp_bwd_wy_v_pass2a_kernel),
    ("_gdp_bwd_wy_k_pass2b_kernel", _gdp_bwd_wy_k_pass2b_kernel),
    ("_gdp_bwd_wy_dA_pass2c_kernel", _gdp_bwd_wy_dA_pass2c_kernel),
    ("_gdp_bwd_reduce_dg_rev_cumsum_clean_kernel", _gdp_bwd_reduce_dg_rev_cumsum_clean_kernel),
    ("_gdp_l2norm_fwd_kernel", _gdp_l2norm_fwd_kernel),
    ("_gdp_l2norm_bwd_kernel", _gdp_l2norm_bwd_kernel),
    ("_gdp_short_conv1d_k4_silu_fwd_kernel", _gdp_short_conv1d_k4_silu_fwd_kernel),
    ("_gdp_short_conv1d_k4_silu_bwd_kernel", _gdp_short_conv1d_k4_silu_bwd_kernel),
    ("_gdp_fused_beta_g_fwd_kernel", _gdp_fused_beta_g_fwd_kernel),
    ("_gdp_fused_beta_g_bwd_kernel", _gdp_fused_beta_g_bwd_kernel),
    ("_gdp_rmsnorm_gated_fwd_kernel", _gdp_rmsnorm_gated_fwd_kernel),
    ("_gdp_rmsnorm_gated_bwd_kernel", _gdp_rmsnorm_gated_bwd_kernel),
]


def _extract_compiled_kernels(kobj: Any) -> List[Any]:
    compiled = []
    seen = set()

    def _walk(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "n_regs") and hasattr(obj, "n_spills"):
            compiled.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
            return
        if hasattr(obj, "cache"):
            _walk(getattr(obj, "cache"))
        if hasattr(obj, "device_caches"):
            _walk(getattr(obj, "device_caches"))
        if hasattr(obj, "fn"):
            _walk(getattr(obj, "fn"))

    _walk(kobj)
    return compiled


def clear_gdp_rsi_kernel_caches() -> None:
    for _, kobj in ALL_GDP_RSI_KERNELS:
        if hasattr(kobj, "cache") and isinstance(kobj.cache, dict):
            kobj.cache.clear()
        if hasattr(kobj, "device_caches") and isinstance(kobj.device_caches, dict):
            for dc in kobj.device_caches.values():
                if isinstance(dc, tuple) and len(dc) > 0 and isinstance(dc[0], dict):
                    dc[0].clear()
                elif isinstance(dc, dict):
                    dc.clear()


def audit_gdp_rsi_cubin_kernels(max_allowed_regs: int = 168) -> Dict[str, Any]:
    per_kernel = {}
    total_specs = 0
    spilling_specs = 0
    over_reg_specs = 0
    max_regs = 0
    max_spills = 0
    max_shared = 0

    for kname, kobj in ALL_GDP_RSI_KERNELS:
        cks = _extract_compiled_kernels(kobj)
        specs = []
        for ck in cks:
            nr = int(ck.n_regs)
            ns = int(ck.n_spills)
            smem = int(getattr(ck.metadata, "shared", 0))
            specs.append({"n_regs": nr, "n_spills": ns, "shared_bytes": smem})
            total_specs += 1
            if ns > 0:
                spilling_specs += 1
            if nr > max_allowed_regs:
                over_reg_specs += 1
            max_regs = max(max_regs, nr)
            max_spills = max(max_spills, ns)
            max_shared = max(max_shared, smem)
        per_kernel[kname] = {
            "num_specializations": len(specs),
            "max_regs": max((s["n_regs"] for s in specs), default=0),
            "max_spills": max((s["n_spills"] for s in specs), default=0),
            "max_shared_bytes": max((s["shared_bytes"] for s in specs), default=0),
            "specializations": specs,
        }

    compiled_kernels = sum(1 for v in per_kernel.values() if v["num_specializations"] > 0)
    return {
        "total_kernels": len(ALL_GDP_RSI_KERNELS),
        "compiled_kernels": compiled_kernels,
        "total_specializations": total_specs,
        "spilling_specializations": spilling_specs,
        "over_reg_specializations": over_reg_specs,
        "max_regs": max_regs,
        "max_allowed_regs": max_allowed_regs,
        "max_spills": max_spills,
        "max_shared_bytes": max_shared,
        "all_zero_spill": (spilling_specs == 0 and over_reg_specs == 0 and compiled_kernels == len(ALL_GDP_RSI_KERNELS)),
        "per_kernel": per_kernel,
    }
