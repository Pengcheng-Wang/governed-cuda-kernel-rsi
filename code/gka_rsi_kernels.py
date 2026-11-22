"""
Round-30 RSI (E156 / v147): Zero-Spill Triton Kernels for Gated KalmaNet (GKA)
Reference: "Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression" (arXiv:2511.21016)
           Official upstream baseline: fla/ops/gka & fla/layers/gka.py

Key Architectural & Mathematical Innovations:
1. Unified Adjoint Gramian State Theorem:
    Combines the direct key-Gramian backward recurrence `dH_{[c]}` (`chunk_bwd_dh`) and the
   Frobenius-ridge `lambda_t = ridge_ratio * ||H_t||_F` inter-chunk backward recurrence `B_{hs, [c]}`
   (`chunk_gka_bwd_kernel_dk_inter`) into a single column-slabbed `[K, BV]` recurrence
   `dH_tilde_{[c]} = dH_{[c]} + 0.5 * B_{hs, [c]}`, eliminating `hs [B, NT, H, K, K]` and
   `chunk_gka_bwd_kernel_dk_inter` while yielding exact initial-state gradient `dh0_kk = -dH_tilde_{[0]}`.
2. Rank-1 Symmetric Completion of W_2 & Euler Quadratic Gramian Identity:
   Replaces the `[64, 64] x [64, 64]` matrix multiplication `W_2 = (M * w) @ M^T` with the exact
   1D reverse-cumsum rank-1 completion `W_{2, j, l} = 1[j <= l] 2^{-g_j} w_{1, l} + 1[j > l] w_{1, j} 2^{-g_l}`,
   and replaces the redundant `O(BT^2 K)` GEMM loop for `sum_d ((K K^T * W_2) K)_{t,d} K_{t,d}` with
   `sum_j (K K^T * W_2 * K K^T)_{t,j}`.
3. In-Flight Frobenius Norm Projection & Runtime-Loop Strip-Tiled Chebyshev Solver:
   Accumulates `||H_{[c]}||_F^2` and `k_j^T H_{[c]} k_j` inside `_gka_fwd_h_kk_fro_part_rsi_kernel`
   while each `[K, BV]` slab of `H_{[c]}` is already in registers, and solves `(H_t + lambda_t I) x_t = q_t`
   using a runtime-loop strip-tiled Tensor-Core kernel (`BF16`) and 2-slab 1D-vector IEEE-754 kernel (`FP32`)
   with scalar Chebyshev momentum `rho_sq4 = (2 * (2 * ridge_ratio + 1))^-2`.
4. Self-Contained Zero-Spill GLA Readout/Backward & 100% Elimination of `SimpleGLABackwardGuard` Crash:
   Replaces upstream `chunk_simple_gla` (which crashes with illegal memory access in 16-bit backward when
   `K < 64` or `V < 32` and spills up to 120 registers) with strip-tiled zero-spill readout and backward
   kernels that fuse `q_mix = q + alpha * (x - q)` and its adjoint in-register.
5. 2-Slab Zero-Spill Online Decoding (`gka_decode_rsi`), 5-Stage Superfused `GatedKalmaNetBlockRSI`,
   and Zero-Copy `dim=0` Packed 8-GPU NVSwitch Ulysses Context Parallelism (`gka_cp8_ulysses_rsi`).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

RCP_LN2: float = 1.4426950408889634


# ==============================================================================
# 0. Differentiable FP64 Oracles (`exact` and `chebyshev`)
# ==============================================================================

def _cheb_solve_fp64(
    h_kk: torch.Tensor,
    qt: torch.Tensor,
    fro: torch.Tensor,
    ridge_ratio: float,
    num_iter: Optional[int],
    exact_solve: bool = False,
) -> torch.Tensor:
    if exact_solve or num_iter is None:
        K = qt.shape[-1]
        eye = torch.eye(K, device=qt.device, dtype=torch.float64)
        lamb = ridge_ratio * fro
        return torch.linalg.solve(h_kk + lamb[..., None, None] * eye, qt.unsqueeze(-1)).squeeze(-1)
    rho_sq4 = (1.0 / (2.0 * (2.0 * ridge_ratio + 1.0))) ** 2
    step_scale = 2.0 / (2.0 * ridge_ratio + 1.0)
    lamb = ridge_ratio * fro
    step = (step_scale / fro)[..., None]
    x_prev = torch.zeros_like(qt)
    xt = step * qt
    w = 2.0
    for _ in range(num_iter):
        w = 1.0 / (1.0 - rho_sq4 * w)
        res = torch.einsum("bhij,bhj->bhi", h_kk, xt) + lamb[..., None] * xt - qt
        d = step * w * res + (w - 1.0) * x_prev
        x_prev = xt
        xt = w * xt - d
    return xt


class _GKAOracleChebIFTFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h_kk: torch.Tensor,
        qt: torch.Tensor,
        fro: torch.Tensor,
        ridge_ratio: float,
        num_iter: Optional[int],
        exact_solve: bool,
    ) -> torch.Tensor:
        xt = _cheb_solve_fp64(h_kk, qt, fro, ridge_ratio, num_iter, exact_solve=exact_solve)
        ctx.save_for_backward(h_kk, xt, fro)
        ctx.ridge_ratio = ridge_ratio
        ctx.num_iter = num_iter
        ctx.exact_solve = exact_solve
        return xt

    @staticmethod
    def backward(ctx, dxt: torch.Tensor):
        h_kk, xt, fro = ctx.saved_tensors
        dqt = _cheb_solve_fp64(
            h_kk, dxt, fro, ctx.ridge_ratio, ctx.num_iter, exact_solve=ctx.exact_solve
        )
        dh_kk = -dqt.unsqueeze(-1) * xt.unsqueeze(-2)
        dfro = -ctx.ridge_ratio * (dqt * xt).sum(dim=-1)
        return dh_kk, dqt, dfro, None, None, None


def gka_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    ridge_ratio: float = 0.02,
    num_iter: Optional[int] = 30,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    exact_solve: bool = False,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Token-by-token differentiable FP64 reference for Gated KalmaNet.
    - Uses `_GKAOracleChebIFTFunction` so forward executes `num_iter` Chebyshev semi-iterative steps
      (or `torch.linalg.solve` when `exact_solve=True`), and backward executes the exact Implicit Function
      Theorem (IFT) Chebyshev adjoint `(H_t + lamb_t I) dq_solve = dx` specified by Gated KalmaNet (arXiv:2511.21016).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q64 = q.to(torch.float64)
    k64 = k.to(torch.float64)
    v64 = v.to(torch.float64)
    g64 = g.to(torch.float64) if g is not None else None
    gk64 = gk.to(torch.float64) if gk is not None else None
    alpha64 = alpha.to(torch.float64) if alpha is not None else None

    if initial_state is not None and initial_state[0] is not None:
        h_kk = initial_state[0].to(torch.float64)
    else:
        h_kk = torch.zeros(B, H, K, K, device=q.device, dtype=torch.float64)

    if initial_state is not None and initial_state[1] is not None:
        h_kv = initial_state[1].to(torch.float64)
    else:
        h_kv = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float64)

    outs = []
    for t in range(T):
        qt = q64[:, t]
        kt = k64[:, t]
        vt = v64[:, t]
        if gk64 is not None:
            h_kk = h_kk * torch.exp(gk64[:, t])[..., None, None]
        if g64 is not None:
            h_kv = h_kv * torch.exp(g64[:, t])[..., None, None]
        h_kk = h_kk + kt.unsqueeze(-1) * kt.unsqueeze(-2)
        h_kv = h_kv + kt.unsqueeze(-1) * vt.unsqueeze(-2)

        fro = torch.linalg.matrix_norm(h_kk, ord="fro")
        xt = _GKAOracleChebIFTFunction.apply(h_kk, qt, fro, ridge_ratio, num_iter, exact_solve)

        q_mix = xt if alpha64 is None else (qt + alpha64[:, t, :, None] * (xt - qt))
        ot = scale * torch.einsum("bhk,bhkv->bhv", q_mix, h_kv)
        outs.append(ot)

    o = torch.stack(outs, dim=1)
    final_state = (h_kk, h_kv) if output_final_state else None
    return o, final_state


# ==============================================================================
# 1. Forward Kernels (`chunk_gka_fwd_rsi`)
# ==============================================================================

@triton.jit
def _gka_local_cumsum_fwd_rsi_kernel(
    g_in_ptr,
    g_out_ptr,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    SCALE: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    b_g = tl.load(g_in_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_cs = tl.cumsum(b_g, axis=0) * SCALE
    tl.store(g_out_ptr + (bos + o_t) * H + i_h, b_cs, mask=m_t)


@triton.jit
def _gka_fwd_h_kk_fro_part_rsi_kernel(
    k_ptr,
    gk_ptr,
    h_out_ptr,
    h0_ptr,
    ht_ptr,
    khk_part_ptr,
    h_norm_part_ptr,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IEEE_DOT: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"
    NS: tl.constexpr = BT // BS

    b_h = tl.zeros([K, BV], dtype=tl.float32)
    if USE_INIT_STATE:
        b_h += tl.load(h0_ptr + i_bh * K * K + o_k[:, None] * K + o_v[None, :]).to(tl.float32)

    k_base = k_ptr + (bos * H + i_h) * K

    for i_c in range(NT):
        i_tg = i_b * NT + i_c
        h_off = (i_tg * H + i_h) * K * K
        b_h_store = b_h.to(h_out_ptr.dtype.element_ty)
        tl.store(h_out_ptr + h_off + o_k[:, None] * K + o_v[None, :], b_h_store)

        h_norm_slab = tl.sum(b_h * b_h)
        tl.store(h_norm_part_ptr + (i_tg * H + i_h) * NV + i_v, h_norm_slab)

        i_last = tl.minimum(BT, T - i_c * BT) - 1
        if USE_GK:
            b_g_last = tl.load(gk_ptr + (bos + i_c * BT + i_last) * H + i_h).to(tl.float32)
            b_h_next = b_h * tl.math.exp2(b_g_last)
        else:
            b_g_last = 0.0
            b_h_next = b_h

        for i_s in range(NS):
            o_t = i_c * BT + i_s * BS + tl.arange(0, BS)
            m_t = o_t < T
            b_k = tl.load(k_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_k_v = tl.load(k_base + o_t[:, None] * (H * K) + o_v[None, :], mask=m_t[:, None], other=0.0)

            b_kh = tl.dot(b_k, b_h_store.to(b_k.dtype), input_precision=DOT_PREC)
            b_khk = tl.sum(b_kh * b_k_v.to(tl.float32), axis=1)
            tl.store(khk_part_ptr + ((bos + o_t) * H + i_h) * NV + i_v, b_khk, mask=m_t)

            if USE_GK:
                b_g = tl.load(gk_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
                b_k_scaled = (b_k.to(tl.float32) * tl.math.exp2(b_g_last - b_g)[:, None]).to(b_k.dtype)
            else:
                b_k_scaled = b_k
            b_h_next += tl.dot(tl.trans(b_k_scaled), b_k_v, input_precision=DOT_PREC)

        b_h = b_h_next

    if STORE_FINAL_STATE:
        tl.store(ht_ptr + i_bh * K * K + o_k[:, None] * K + o_v[None, :], b_h.to(ht_ptr.dtype.element_ty))


@triton.jit
def _gka_fro_full_bf16_rsi_kernel(
    k_ptr,
    gk_ptr,
    khk_part_ptr,
    h_norm_part_ptr,
    fro_ptr,
    T,
    NT,
    NK_rt,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NV: tl.constexpr,
    USE_GK: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_tg = i_b * NT + i_t
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_s = (o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :]

    if USE_GK:
        b_gk = tl.load(gk_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_m = tl.where(m_s, tl.math.exp2(b_gk[:, None] - b_gk[None, :]), 0.0)
        b_gh = tl.math.exp2(b_gk)
    else:
        b_m = tl.where(m_s, 1.0, 0.0)
        b_gh = tl.full([BT], 1.0, dtype=tl.float32)

    k_base = k_ptr + (bos * H + i_h) * K
    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(NK_rt):
        o_k = i_k * BK + tl.arange(0, BK)
        b_k = tl.load(k_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None], other=0.0)
        b_kk += tl.dot(b_k, tl.trans(b_k))

    h_norm_sq = 0.0
    b_khk = tl.zeros([BT], dtype=tl.float32)
    for i_v in range(NV):
        h_norm_sq += tl.load(h_norm_part_ptr + (i_tg * H + i_h) * NV + i_v).to(tl.float32)
        b_khk += tl.load(khk_part_ptr + ((bos + o_t) * H + i_h) * NV + i_v, mask=m_t, other=0.0).to(tl.float32)

    b_m_cast = b_m.to(k_ptr.dtype.element_ty)
    b_kk_sq = (b_kk * b_kk).to(k_ptr.dtype.element_ty)
    b_fro = tl.sum(tl.dot(b_m_cast, b_kk_sq) * b_m, axis=1)
    b_fro += b_gh * (b_gh * h_norm_sq) + 2.0 * b_gh * tl.sum(b_m * b_khk[None, :], axis=1)
    b_fro = tl.sqrt(tl.maximum(b_fro, 0.0))
    b_fro = tl.where(m_t, b_fro, 1.0)
    tl.store(fro_ptr + (bos + o_t) * H + i_h, b_fro.to(fro_ptr.dtype.element_ty), mask=m_t)


@triton.jit
def _gka_fro_strip_rsi_kernel(
    k_ptr,
    gk_ptr,
    khk_part_ptr,
    h_norm_part_ptr,
    fro_ptr,
    T,
    NT,
    NK_rt,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    NV: tl.constexpr,
    USE_GK: tl.constexpr,
    IEEE_DOT: tl.constexpr,
):
    i_ts = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NS: tl.constexpr = BT // BS
    i_t = i_ts // NS
    i_s = i_ts % NS
    i_b, i_h = i_bh // H, i_bh % H
    i_tg = i_b * NT + i_t
    bos = i_b * T
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"

    o_t_strip = i_t * BT + i_s * BS + tl.arange(0, BS)
    o_t_chunk = i_t * BT + tl.arange(0, BT)
    m_ts = o_t_strip < T
    m_tc = o_t_chunk < T
    m_s = (o_t_strip[:, None] >= o_t_chunk[None, :]) & m_ts[:, None] & m_tc[None, :]

    if USE_GK:
        b_gk_s = tl.load(gk_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
        b_gk_c = tl.load(gk_ptr + (bos + o_t_chunk) * H + i_h, mask=m_tc, other=0.0).to(tl.float32)
        b_m = tl.where(m_s, tl.math.exp2(b_gk_s[:, None] - b_gk_c[None, :]), 0.0)
        b_gh = tl.math.exp2(b_gk_s)
    else:
        b_m = tl.where(m_s, 1.0, 0.0)
        b_gh = tl.full([BS], 1.0, dtype=tl.float32)

    h_norm_sq = 0.0
    b_khk_c = tl.zeros([BT], dtype=tl.float32)
    for i_v in range(NV):
        h_norm_sq += tl.load(h_norm_part_ptr + (i_tg * H + i_h) * NV + i_v).to(tl.float32)
        b_khk_c += tl.load(khk_part_ptr + ((bos + o_t_chunk) * H + i_h) * NV + i_v, mask=m_tc, other=0.0).to(tl.float32)
    b_khk_sum = tl.sum(b_m * b_khk_c[None, :], axis=1)

    k_base = k_ptr + (bos * H + i_h) * K
    b_fro_intra = tl.zeros([BS], dtype=tl.float32)

    for s_c in range(NS):
        o_t_j = i_t * BT + s_c * BS + tl.arange(0, BS)
        m_tj = o_t_j < T
        b_kk_sub = tl.zeros([BS, BT], dtype=tl.float32)
        for i_k in range(NK_rt):
            o_k = i_k * BK + tl.arange(0, BK)
            b_kj = tl.load(k_base + o_t_j[:, None] * (H * K) + o_k[None, :], mask=m_tj[:, None], other=0.0)
            b_kl = tl.load(k_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None], other=0.0)
            b_kk_sub += tl.dot(b_kj, tl.trans(b_kl), input_precision=DOT_PREC)
        b_kk_sq = (b_kk_sub * b_kk_sub).to(k_ptr.dtype.element_ty)
        b_v_jt = tl.dot(b_kk_sq, tl.trans(b_m.to(k_ptr.dtype.element_ty)), input_precision=DOT_PREC)
        m_sj = (o_t_strip[:, None] >= o_t_j[None, :]) & m_ts[:, None] & m_tj[None, :]
        if USE_GK:
            b_gk_j = tl.load(gk_ptr + (bos + o_t_j) * H + i_h, mask=m_tj, other=0.0).to(tl.float32)
            b_m_sub = tl.where(m_sj, tl.math.exp2(b_gk_s[:, None] - b_gk_j[None, :]), 0.0)
        else:
            b_m_sub = tl.where(m_sj, 1.0, 0.0)
        b_fro_intra += tl.sum(tl.trans(b_v_jt) * b_m_sub, axis=1)

    b_fro = b_fro_intra + b_gh * (b_gh * h_norm_sq) + 2.0 * b_gh * b_khk_sum
    b_fro = tl.sqrt(tl.maximum(b_fro, 0.0))
    b_fro = tl.where(m_ts, b_fro, 1.0)
    tl.store(fro_ptr + (bos + o_t_strip) * H + i_h, b_fro.to(fro_ptr.dtype.element_ty), mask=m_ts)


@triton.jit
def _gka_solve_fwd_bf16_rsi_kernel(
    q_ptr,
    k_ptr,
    h_ptr,
    gk_ptr,
    fro_ptr,
    x_ptr,
    ridge_ratio,
    num_iter,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    MUTATION_MODE: tl.constexpr,
):
    i_ts = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NS: tl.constexpr = BT // BS
    i_t = i_ts // NS
    i_s = i_ts % NS
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T

    q_base = q_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    x_base = x_ptr + (bos * H + i_h) * K
    fro_base = fro_ptr + bos * H + i_h
    h_base = h_ptr + (i_tg * H + i_h) * K * K

    o_t_strip = i_t * BT + i_s * BS + tl.arange(0, BS)
    m_ts = o_t_strip < T
    o_k = tl.arange(0, BK)

    b_q = tl.load(q_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None], other=0.0)
    b_h = tl.load(h_base + o_k[:, None] * K + o_k[None, :]).to(b_q.dtype)
    b_fro = tl.load(fro_base + o_t_strip * H, mask=m_ts, other=1.0).to(tl.float32)

    if BS == 32:
        # Sub-Chunk Strip-State Folding for K <= 64 (NS = 2): fold strip 0 into b_h once before Chebyshev loop
        g_prev = 0.0
        if i_s == 1:
            o_t0 = i_t * BT + tl.arange(0, BS)
            m_t0 = o_t0 < T
            b_k0 = tl.load(k_base + o_t0[:, None] * (H * K) + o_k[None, :], mask=m_t0[:, None], other=0.0)
            if USE_GK:
                last_prev = i_t * BT + BS - 1
                g_prev = tl.load(gk_ptr + (bos + last_prev) * H + i_h).to(tl.float32)
                b_gk0 = tl.load(gk_ptr + (bos + o_t0) * H + i_h, mask=m_t0, other=0.0).to(tl.float32)
                b_k0_s = (b_k0.to(tl.float32) * tl.math.exp2(g_prev - b_gk0)[:, None]).to(b_q.dtype)
                b_h = (b_h.to(tl.float32) * tl.math.exp2(g_prev) + tl.dot(tl.trans(b_k0_s), b_k0)).to(b_q.dtype)
            else:
                b_h = (b_h.to(tl.float32) + tl.dot(tl.trans(b_k0), b_k0)).to(b_q.dtype)

        b_k = tl.load(k_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None], other=0.0)
        m_s = (o_t_strip[:, None] >= o_t_strip[None, :]) & m_ts[:, None] & m_ts[None, :]
        if USE_GK:
            b_gk_s = tl.load(gk_ptr + (bos * H + i_h) + o_t_strip * H, mask=m_ts, other=0.0).to(tl.float32)
            b_m = tl.where(m_s, tl.math.exp2(b_gk_s[:, None] - b_gk_s[None, :]), 0.0).to(b_q.dtype)
            b_gh = tl.math.exp2(b_gk_s - g_prev)
        else:
            b_m = m_s.to(b_q.dtype)
            b_gh = tl.full([BS], 1.0, dtype=tl.float32)
    else:
        o_t_chunk = i_t * BT + tl.arange(0, BT)
        m_tc = o_t_chunk < T
        b_k = tl.load(k_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None], other=0.0)
        m_s = (o_t_strip[:, None] >= o_t_chunk[None, :]) & m_ts[:, None] & m_tc[None, :]
        if USE_GK:
            b_gk_s = tl.load(gk_ptr + (bos * H + i_h) + o_t_strip * H, mask=m_ts, other=0.0).to(tl.float32)
            b_gk_c = tl.load(gk_ptr + (bos * H + i_h) + o_t_chunk * H, mask=m_tc, other=0.0).to(tl.float32)
            b_m = tl.where(m_s, tl.math.exp2(b_gk_s[:, None] - b_gk_c[None, :]), 0.0).to(b_q.dtype)
            b_gh = tl.math.exp2(b_gk_s)
        else:
            b_m = m_s.to(b_q.dtype)
            b_gh = tl.full([BS], 1.0, dtype=tl.float32)

    rho_sq4 = 1.0 / (2.0 * (2.0 * ridge_ratio + 1.0))
    rho_sq4 = rho_sq4 * rho_sq4
    if MUTATION_MODE == 1:
        rho_sq4 = 0.0
    step_scale = 2.0 / (2.0 * ridge_ratio + 1.0)
    b_lamb = (ridge_ratio * b_fro)[:, None]
    if MUTATION_MODE == 2:
        b_lamb = tl.zeros([BS, 1], dtype=tl.float32)
    b_step = (step_scale / b_fro)[:, None]

    b_x_prev = tl.zeros([BS, BK], dtype=tl.float32)
    b_x = b_step * b_q.to(tl.float32)
    w_s = 2.0
    for _ in range(num_iter):
        w_s = 1.0 / (1.0 - rho_sq4 * w_s)
        b_r = (
            tl.dot((b_x * b_gh[:, None]).to(b_q.dtype), b_h)
            + tl.dot((tl.dot(b_x.to(b_q.dtype), tl.trans(b_k)) * b_m).to(b_q.dtype), b_k)
            + b_lamb * b_x
            - b_q.to(tl.float32)
        )
        b_d = b_step * w_s * b_r + (w_s - 1.0) * b_x_prev
        b_x_prev = b_x
        b_x = w_s * b_x - b_d

    tl.store(
        x_base + o_t_strip[:, None] * (H * K) + o_k[None, :],
        b_x.to(x_ptr.dtype.element_ty),
        mask=m_ts[:, None],
    )


@triton.jit
def _gka_solve_fwd_fp32_rsi_kernel(
    q_ptr,
    k_ptr,
    h_ptr,
    gk_ptr,
    fro_ptr,
    alpha_ptr,
    x_ptr,
    q_mix_ptr,
    ridge_ratio,
    num_iter,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    MUTATION_MODE: tl.constexpr,
):
    i_t_tok = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    if i_t_tok >= T:
        return
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    i_c = i_t_tok // BT
    i_sub = i_t_tok % BT
    i_tg = i_b * NT + i_c
    bos = i_b * T

    o_k = tl.arange(0, BK)
    q_off = ((bos + i_t_tok) * H + i_h) * K + o_k
    b_q = tl.load(q_ptr + q_off).to(tl.float32)
    fro_val = tl.load(fro_ptr + (bos + i_t_tok) * H + i_h).to(tl.float32)
    gk_t = tl.load(gk_ptr + (bos + i_t_tok) * H + i_h).to(tl.float32) if USE_GK else 0.0
    gh_t = tl.math.exp2(gk_t) if USE_GK else 1.0

    HALF_K: tl.constexpr = BK // 2
    o_r0 = tl.arange(0, HALF_K)
    o_r1 = HALF_K + tl.arange(0, HALF_K)
    h_base = h_ptr + (i_tg * H + i_h) * K * K
    b_h0 = tl.load(h_base + o_r0[:, None] * K + o_k[None, :]).to(tl.float32) * gh_t
    b_h1 = tl.load(h_base + o_r1[:, None] * K + o_k[None, :]).to(tl.float32) * gh_t

    k_chunk_base = k_ptr + ((bos + i_c * BT) * H + i_h) * K
    gk_chunk_base = gk_ptr + (bos + i_c * BT) * H + i_h
    for j in range(i_sub + 1):
        b_kj = tl.load(k_chunk_base + j * (H * K) + o_k).to(tl.float32)
        if USE_GK:
            gk_j = tl.load(gk_chunk_base + j * H).to(tl.float32)
            m_tj = tl.math.exp2(gk_t - gk_j)
        else:
            m_tj = 1.0
        b_kj0 = tl.load(k_chunk_base + j * (H * K) + o_r0).to(tl.float32) * m_tj
        b_kj1 = tl.load(k_chunk_base + j * (H * K) + o_r1).to(tl.float32) * m_tj
        b_h0 += b_kj0[:, None] * b_kj[None, :]
        b_h1 += b_kj1[:, None] * b_kj[None, :]

    rho_sq4 = 1.0 / (2.0 * (2.0 * ridge_ratio + 1.0))
    rho_sq4 = rho_sq4 * rho_sq4
    if MUTATION_MODE == 1:
        rho_sq4 = 0.0
    lamb_val = ridge_ratio * fro_val
    if MUTATION_MODE == 2:
        lamb_val = 0.0
    step_val = (2.0 / (2.0 * ridge_ratio + 1.0)) / fro_val

    b_x_prev = tl.zeros([BK], dtype=tl.float32)
    b_x = step_val * b_q
    w_s = 2.0

    for _ in range(num_iter):
        w_s = 1.0 / (1.0 - rho_sq4 * w_s)
        b_x_2d = tl.reshape(b_x, [2, HALF_K])
        b_x0 = tl.sum(b_x_2d * (tl.arange(0, 2)[:, None] == 0), axis=0)
        b_x1 = tl.sum(b_x_2d * (tl.arange(0, 2)[:, None] == 1), axis=0)
        b_hx = tl.sum(b_h0 * b_x0[:, None], axis=0) + tl.sum(b_h1 * b_x1[:, None], axis=0)
        b_r = b_hx + lamb_val * b_x - b_q
        b_d = step_val * w_s * b_r + (w_s - 1.0) * b_x_prev
        b_x_prev = b_x
        b_x = w_s * b_x - b_d

    tl.store(x_ptr + q_off, b_x.to(x_ptr.dtype.element_ty))
    if USE_ALPHA:
        alpha_val = tl.load(alpha_ptr + (bos + i_t_tok) * H + i_h).to(tl.float32)
        if MUTATION_MODE == 3:
            alpha_val = 1.0 - alpha_val
        b_q_mix = b_q + alpha_val * (b_x - b_q)
        tl.store(q_mix_ptr + q_off, b_q_mix.to(q_mix_ptr.dtype.element_ty))


@triton.jit
def _gka_fwd_h_kv_rsi_kernel(
    k_ptr,
    v_ptr,
    g_ptr,
    h_kv_out_ptr,
    h0_kv_ptr,
    ht_kv_ptr,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IEEE_DOT: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"
    NS: tl.constexpr = BT // BS

    b_h = tl.zeros([K, BV], dtype=tl.float32)
    if USE_INIT_STATE:
        b_h += tl.load(h0_kv_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)

    k_base = k_ptr + (bos * H + i_h) * K
    v_base = v_ptr + (bos * H + i_h) * V

    for i_c in range(NT):
        i_tg = i_b * NT + i_c
        h_off = (i_tg * H + i_h) * K * V
        tl.store(h_kv_out_ptr + h_off + o_k[:, None] * V + o_v[None, :], b_h.to(h_kv_out_ptr.dtype.element_ty), mask=m_v[None, :])

        i_last = tl.minimum(BT, T - i_c * BT) - 1
        if USE_G:
            b_g_last = tl.load(g_ptr + (bos + i_c * BT + i_last) * H + i_h).to(tl.float32)
            b_h = b_h * tl.math.exp2(b_g_last)
        else:
            b_g_last = 0.0

        for i_s in range(NS):
            o_t = i_c * BT + i_s * BS + tl.arange(0, BS)
            m_t = o_t < T
            b_k = tl.load(k_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_v = tl.load(v_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.0)
            if USE_G:
                b_g = tl.load(g_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
                b_k_scaled = (b_k.to(tl.float32) * tl.math.exp2(b_g_last - b_g)[:, None]).to(b_k.dtype)
            else:
                b_k_scaled = b_k
            b_h += tl.dot(tl.trans(b_k_scaled), b_v, input_precision=DOT_PREC)

    if STORE_FINAL_STATE:
        tl.store(ht_kv_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :], b_h.to(ht_kv_ptr.dtype.element_ty), mask=m_v[None, :])


@triton.jit
def _gka_fwd_o_rsi_kernel(
    q_ptr,
    x_ptr,
    alpha_ptr,
    q_mix_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    h_kv_ptr,
    o_ptr,
    scale,
    T,
    NT,
    NK_rt,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    IEEE_DOT: tl.constexpr,
    MUTATION_MODE: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_ts = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NS: tl.constexpr = BT // BS
    i_t = i_ts // NS
    i_s = i_ts % NS
    i_b, i_h = i_bh // H, i_bh % H
    i_tg = i_b * NT + i_t
    bos = i_b * T
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"

    o_t_strip = i_t * BT + i_s * BS + tl.arange(0, BS)
    o_t_chunk = i_t * BT + tl.arange(0, BT)
    m_ts = o_t_strip < T
    m_tc = o_t_chunk < T
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    q_raw_base = q_ptr + (bos * H + i_h) * K
    x_base = x_ptr + (bos * H + i_h) * K
    qm_base = q_mix_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    v_base = v_ptr + (bos * H + i_h) * V
    o_base = o_ptr + (bos * H + i_h) * V
    h_base = h_kv_ptr + (i_tg * H + i_h) * K * V

    if USE_ALPHA:
        b_alpha = tl.load(alpha_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
        if MUTATION_MODE == 3:
            b_alpha = 1.0 - b_alpha
    else:
        b_alpha = tl.zeros([BS], dtype=tl.float32)

    b_o = tl.zeros([BS, BV], dtype=tl.float32)
    b_A = tl.zeros([BS, BT], dtype=tl.float32)

    for i_k in range(NK_rt):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        if USE_ALPHA:
            b_q_raw = tl.load(q_raw_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_x_s = tl.load(x_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_q = (b_q_raw + b_alpha[:, None] * (b_x_s - b_q_raw)).to(k_ptr.dtype.element_ty)
            if i_v == 0:
                tl.store(qm_base + o_t_strip[:, None] * (H * K) + o_k[None, :], b_q, mask=m_ts[:, None] & m_k[None, :])
        else:
            b_q = tl.load(qm_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None] & m_k[None, :], other=0.0)
        b_k = tl.load(k_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None] & m_k[None, :], other=0.0)
        b_h = tl.load(h_base + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_o += tl.dot(b_q, b_h.to(b_q.dtype), input_precision=DOT_PREC)
        b_A += tl.dot(b_q, tl.trans(b_k), input_precision=DOT_PREC)

    m_s = (o_t_strip[:, None] >= o_t_chunk[None, :]) & m_ts[:, None] & m_tc[None, :]
    if USE_G:
        b_g_s = tl.load(g_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
        b_g_c = tl.load(g_ptr + (bos + o_t_chunk) * H + i_h, mask=m_tc, other=0.0).to(tl.float32)
        b_o = b_o * tl.math.exp2(b_g_s)[:, None]
        b_A = tl.where(m_s, b_A * tl.math.exp2(b_g_s[:, None] - b_g_c[None, :]), 0.0)
    else:
        b_A = tl.where(m_s, b_A, 0.0)

    b_v = tl.load(v_base + o_t_chunk[:, None] * (H * V) + o_v[None, :], mask=m_tc[:, None] & m_v[None, :], other=0.0)
    b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v, input_precision=DOT_PREC)) * scale
    tl.store(
        o_base + o_t_strip[:, None] * (H * V) + o_v[None, :],
        b_o.to(o_ptr.dtype.element_ty),
        mask=m_ts[:, None] & m_v[None, :],
    )


# ==============================================================================
# 2. Backward Kernels (`chunk_gka_bwd_rsi`)
# ==============================================================================

@triton.jit
def _gka_bwd_dh_kv_rsi_kernel(
    q_mix_ptr,
    do_ptr,
    g_ptr,
    dht_kv_ptr,
    dh_kv_out_ptr,
    dh0_kv_ptr,
    scale,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_DHT: tl.constexpr,
    STORE_DH0: tl.constexpr,
    IEEE_DOT: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"
    NS: tl.constexpr = BT // BS

    b_dh = tl.zeros([K, BV], dtype=tl.float32)
    if USE_DHT:
        b_dh += tl.load(dht_kv_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)

    q_base = q_mix_ptr + (bos * H + i_h) * K
    do_base = do_ptr + (bos * H + i_h) * V

    for i_c in range(NT - 1, -1, -1):
        i_tg = i_b * NT + i_c
        h_off = (i_tg * H + i_h) * K * V
        tl.store(dh_kv_out_ptr + h_off + o_k[:, None] * V + o_v[None, :], b_dh.to(dh_kv_out_ptr.dtype.element_ty), mask=m_v[None, :])

        i_last = tl.minimum(BT, T - i_c * BT) - 1
        if USE_G:
            b_g_last = tl.load(g_ptr + (bos + i_c * BT + i_last) * H + i_h).to(tl.float32)
            b_dh = b_dh * tl.math.exp2(b_g_last)

        for i_s in range(NS):
            o_t = i_c * BT + i_s * BS + tl.arange(0, BS)
            m_t = o_t < T
            b_q = tl.load(q_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_do = tl.load(do_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.0)
            if USE_G:
                b_g = tl.load(g_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
                b_q_scaled = (b_q.to(tl.float32) * (tl.math.exp2(b_g) * scale)[:, None]).to(b_q.dtype)
            else:
                b_q_scaled = (b_q.to(tl.float32) * scale).to(b_q.dtype)
            b_dh += tl.dot(tl.trans(b_q_scaled), b_do, input_precision=DOT_PREC)

    if STORE_DH0:
        tl.store(dh0_kv_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :], b_dh.to(dh0_kv_ptr.dtype.element_ty), mask=m_v[None, :])


@triton.jit
def _gka_bwd_dv_gla_rsi_kernel(
    q_mix_ptr,
    k_ptr,
    g_ptr,
    do_ptr,
    dh_kv_ptr,
    dv_ptr,
    scale,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IEEE_DOT: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_ts = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NS: tl.constexpr = BT // BS
    i_t = i_ts // NS
    i_s = i_ts % NS
    i_b, i_h = i_bh // H, i_bh % H
    i_tg = i_b * NT + i_t
    bos = i_b * T
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"

    o_t_strip = i_t * BT + i_s * BS + tl.arange(0, BS)
    o_t_chunk = i_t * BT + tl.arange(0, BT)
    m_ts = o_t_strip < T
    m_tc = o_t_chunk < T
    i_last = tl.minimum(BT, T - i_t * BT) - 1
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    q_base = q_mix_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    do_base = do_ptr + (bos * H + i_h) * V
    dv_base = dv_ptr + (bos * H + i_h) * V
    dh_base = dh_kv_ptr + (i_tg * H + i_h) * K * V

    b_A = tl.zeros([BS, BT], dtype=tl.float32)
    b_dv_inter = tl.zeros([BS, BV], dtype=tl.float32)

    NK: tl.constexpr = tl.cdiv(K, BK)
    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_k_s = tl.load(k_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None] & m_k[None, :], other=0.0)
        b_q_c = tl.load(q_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None] & m_k[None, :], other=0.0)
        b_dh = tl.load(dh_base + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_A += tl.dot(b_k_s, tl.trans(b_q_c), input_precision=DOT_PREC)
        b_dv_inter += tl.dot(b_k_s, b_dh.to(b_k_s.dtype), input_precision=DOT_PREC)

    m_s = (o_t_strip[:, None] <= o_t_chunk[None, :]) & m_ts[:, None] & m_tc[None, :]
    if USE_G:
        b_g_s = tl.load(g_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
        b_g_c = tl.load(g_ptr + (bos + o_t_chunk) * H + i_h, mask=m_tc, other=0.0).to(tl.float32)
        b_g_last = tl.load(g_ptr + (bos + i_t * BT + i_last) * H + i_h).to(tl.float32)
        b_dv_inter = b_dv_inter * tl.math.exp2(b_g_last - b_g_s)[:, None]
        b_A = tl.where(m_s, b_A * tl.math.exp2(b_g_c[None, :] - b_g_s[:, None]) * scale, 0.0)
    else:
        b_A = tl.where(m_s, b_A * scale, 0.0)

    b_do = tl.load(do_base + o_t_chunk[:, None] * (H * V) + o_v[None, :], mask=m_tc[:, None] & m_v[None, :], other=0.0)
    b_dv = b_dv_inter + tl.dot(b_A.to(b_do.dtype), b_do, input_precision=DOT_PREC)
    tl.store(
        dv_base + o_t_strip[:, None] * (H * V) + o_v[None, :],
        b_dv.to(dv_ptr.dtype.element_ty),
        mask=m_ts[:, None] & m_v[None, :],
    )


@triton.jit
def _gka_bwd_dqkg_gla_rsi_kernel(
    q_ptr,
    x_ptr,
    q_mix_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    alpha_ptr,
    h_kv_ptr,
    dh_kv_ptr,
    do_ptr,
    dx_ptr,
    dq_alpha_ptr,
    dk_gla_ptr,
    dg_gla_ptr,
    dg_gla_last_ptr,
    dalpha_part_ptr,
    scale,
    T,
    NT,
    NV_rt,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    USE_G: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    IEEE_DOT: tl.constexpr,
    MUTATION_MODE: tl.constexpr,
):
    i_k = tl.program_id(0).to(tl.int64)
    i_ts = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NS: tl.constexpr = BT // BS
    i_t = i_ts // NS
    i_s = i_ts % NS
    i_b, i_h = i_bh // H, i_bh % H
    i_tg = i_b * NT + i_t
    bos = i_b * T
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"

    o_t_strip = i_t * BT + i_s * BS + tl.arange(0, BS)
    o_t_chunk = i_t * BT + tl.arange(0, BT)
    m_ts = o_t_strip < T
    m_tc = o_t_chunk < T
    i_last = tl.minimum(BT, T - i_t * BT) - 1
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q_raw_base = q_ptr + (bos * H + i_h) * K
    x_base = x_ptr + (bos * H + i_h) * K
    qm_base = q_mix_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    v_base = v_ptr + (bos * H + i_h) * V
    do_base = do_ptr + (bos * H + i_h) * V
    h_base = h_kv_ptr + (i_tg * H + i_h) * K * V
    dh_base = dh_kv_ptr + (i_tg * H + i_h) * K * V

    b_ds_rc = tl.zeros([BS, BT], dtype=tl.float32)
    b_ds_cr = tl.zeros([BS, BT], dtype=tl.float32)
    b_dq_inter = tl.zeros([BS, BK], dtype=tl.float32)
    b_dk_inter = tl.zeros([BS, BK], dtype=tl.float32)
    dh_h_dot = 0.0

    for i_v in range(NV_rt):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_do_s = tl.load(do_base + o_t_strip[:, None] * (H * V) + o_v[None, :], mask=m_ts[:, None] & m_v[None, :], other=0.0)
        b_v_s = tl.load(v_base + o_t_strip[:, None] * (H * V) + o_v[None, :], mask=m_ts[:, None] & m_v[None, :], other=0.0)
        b_do_c = tl.load(do_base + o_t_chunk[:, None] * (H * V) + o_v[None, :], mask=m_tc[:, None] & m_v[None, :], other=0.0)
        b_v_c = tl.load(v_base + o_t_chunk[:, None] * (H * V) + o_v[None, :], mask=m_tc[:, None] & m_v[None, :], other=0.0)

        b_h = tl.load(h_base + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_dh = tl.load(dh_base + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0)

        b_ds_rc += tl.dot(b_do_s, tl.trans(b_v_c), input_precision=DOT_PREC)
        b_ds_cr += tl.dot(b_v_s, tl.trans(b_do_c), input_precision=DOT_PREC)
        b_dq_inter += tl.dot(b_do_s, tl.trans(b_h.to(b_do_s.dtype)), input_precision=DOT_PREC)
        b_dk_inter += tl.dot(b_v_s, tl.trans(b_dh.to(b_v_s.dtype)), input_precision=DOT_PREC)
        if i_s == 0:
            dh_h_dot += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32))

    m_rc = (o_t_strip[:, None] >= o_t_chunk[None, :]) & m_ts[:, None] & m_tc[None, :]
    m_cr = (o_t_strip[:, None] <= o_t_chunk[None, :]) & m_ts[:, None] & m_tc[None, :]

    if USE_G:
        b_g_s = tl.load(g_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
        b_g_c = tl.load(g_ptr + (bos + o_t_chunk) * H + i_h, mask=m_tc, other=0.0).to(tl.float32)
        b_g_last = tl.load(g_ptr + (bos + i_t * BT + i_last) * H + i_h).to(tl.float32)
        b_gh = tl.math.exp2(b_g_s)
        b_gl = tl.math.exp2(b_g_last - b_g_s)
        b_ds_rc = tl.where(m_rc, b_ds_rc * tl.math.exp2(b_g_s[:, None] - b_g_c[None, :]) * scale, 0.0)
        b_ds_cr = tl.where(m_cr, b_ds_cr * tl.math.exp2(b_g_c[None, :] - b_g_s[:, None]) * scale, 0.0)
        b_dq_inter = b_dq_inter * (b_gh * scale)[:, None]
        b_dk_inter = b_dk_inter * b_gl[:, None]
    else:
        b_g_last = 0.0
        b_ds_rc = tl.where(m_rc, b_ds_rc * scale, 0.0)
        b_ds_cr = tl.where(m_cr, b_ds_cr * scale, 0.0)
        b_dq_inter = b_dq_inter * scale

    b_k_c = tl.load(k_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None] & m_k[None, :], other=0.0)
    b_dq_intra = tl.dot(b_ds_rc.to(b_k_c.dtype), b_k_c, input_precision=DOT_PREC)
    b_qm_c = tl.load(qm_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None] & m_k[None, :], other=0.0)
    b_dk_intra = tl.dot(b_ds_cr.to(b_qm_c.dtype), b_qm_c, input_precision=DOT_PREC)
    b_qm_s = tl.load(qm_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None] & m_k[None, :], other=0.0)
    b_k_s = tl.load(k_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None] & m_k[None, :], other=0.0)

    b_dq_mix = b_dq_inter + b_dq_intra
    b_dk_gla = b_dk_inter + b_dk_intra

    tl.store(
        dk_gla_ptr + (bos * H + i_h) * K + o_t_strip[:, None] * (H * K) + o_k[None, :],
        b_dk_gla.to(dk_gla_ptr.dtype.element_ty),
        mask=m_ts[:, None] & m_k[None, :],
    )

    if USE_ALPHA:
        b_q_raw = tl.load(q_raw_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_x = tl.load(x_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_alpha = tl.load(alpha_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
        if MUTATION_MODE == 3:
            b_alpha = 1.0 - b_alpha
        b_dx = b_dq_mix * b_alpha[:, None]
        b_dq_a = b_dq_mix * (1.0 - b_alpha[:, None])
        b_dalpha = tl.sum(b_dq_mix * (b_x - b_q_raw), axis=1)
        if MUTATION_MODE == 3:
            b_dalpha = -b_dalpha
        tl.store(
            dx_ptr + (bos * H + i_h) * K + o_t_strip[:, None] * (H * K) + o_k[None, :],
            b_dx.to(dx_ptr.dtype.element_ty),
            mask=m_ts[:, None] & m_k[None, :],
        )
        tl.store(
            dq_alpha_ptr + (bos * H + i_h) * K + o_t_strip[:, None] * (H * K) + o_k[None, :],
            b_dq_a.to(dq_alpha_ptr.dtype.element_ty),
            mask=m_ts[:, None] & m_k[None, :],
        )
        tl.store(dalpha_part_ptr + ((bos + o_t_strip) * H + i_h) * NK + i_k, b_dalpha, mask=m_ts)
    else:
        tl.store(
            dx_ptr + (bos * H + i_h) * K + o_t_strip[:, None] * (H * K) + o_k[None, :],
            b_dq_mix.to(dx_ptr.dtype.element_ty),
            mask=m_ts[:, None] & m_k[None, :],
        )

    if USE_G:
        # Euler Homogeneous Degree-1 Identity on S_rc: sum_c dS_rc S_rc - sum_c dS_cr S_cr = <dq_intra, qm_s> - <dk_intra, k_s>
        b_dg_s = (
            tl.sum(b_dq_mix * b_qm_s.to(tl.float32), axis=1)
            - tl.sum(b_dk_gla * b_k_s.to(tl.float32), axis=1)
        )
        b_dg_last = tl.sum(b_dk_inter * b_k_s.to(tl.float32), axis=0)
        b_dg_last_val = tl.sum(b_dg_last, axis=0)
        if i_s == 0:
            b_dg_last_val += dh_h_dot * tl.math.exp2(b_g_last)
        tl.store(dg_gla_ptr + ((bos + o_t_strip) * H + i_h) * NK + i_k, b_dg_s, mask=m_ts)
        tl.store(dg_gla_last_ptr + (((i_b * NT + i_t) * H + i_h) * NS + i_s) * NK + i_k, b_dg_last_val)


@triton.jit
def _gka_bwd_w1_vec_rsi_kernel(
    dq_ptr,
    x_ptr,
    fro_ptr,
    gk_ptr,
    w1_ptr,
    r_tilde_ptr,
    w_gh2_sum_ptr,
    dg_intra_ptr,
    ridge_ratio,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    MUTATION_MODE: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_s = (o_t[:, None] <= o_t[None, :]) & m_t[:, None] & m_t[None, :]

    dq_base = dq_ptr + (bos * H + i_h) * K
    x_base = x_ptr + (bos * H + i_h) * K
    b_dq_x = tl.zeros([BT], dtype=tl.float32)
    NK: tl.constexpr = tl.cdiv(K, BK)
    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        b_dq = tl.load(dq_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None], other=0.0).to(tl.float32)
        b_x = tl.load(x_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None], other=0.0).to(tl.float32)
        b_dq_x += tl.sum(b_dq * b_x, axis=1)

    b_fro = tl.load(fro_ptr + (bos + o_t) * H + i_h, mask=m_t, other=1.0).to(tl.float32)
    eff_ridge = 0.0 if MUTATION_MODE == 2 else ridge_ratio
    b_tmp = eff_ridge * b_dq_x
    b_w = tl.where(m_t, 2.0 * b_tmp / b_fro, 0.0)

    if USE_GK:
        b_g = tl.load(gk_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_gh = tl.math.exp2(b_g)
        b_m = tl.where(m_s, tl.math.exp2(b_g[None, :] - b_g[:, None]), 0.0)
        tl.store(dg_intra_ptr + (bos + o_t) * H + i_h, b_tmp * b_fro, mask=m_t)
    else:
        b_gh = tl.full([BT], 1.0, dtype=tl.float32)
        b_m = tl.where(m_s, 1.0, 0.0)

    w_gh2_sum = tl.sum(b_gh * b_gh * b_w, axis=0)
    b_w1 = tl.sum(b_m * (b_gh * b_w)[None, :], axis=1)
    b_r_tilde = tl.sum((b_m * b_m) * b_w[None, :], axis=1)
    tl.store(w1_ptr + (bos + o_t) * H + i_h, b_w1, mask=m_t)
    tl.store(r_tilde_ptr + (bos + o_t) * H + i_h, b_r_tilde, mask=m_t)
    tl.store(w_gh2_sum_ptr + (i_b * NT + i_t) * H + i_h, w_gh2_sum)


@triton.jit
def _gka_bwd_intra_strip_rsi_kernel(
    k_ptr,
    x_ptr,
    dq_ptr,
    gk_ptr,
    r_tilde_ptr,
    dk_intra_ptr,
    dg_intra_ptr,
    dg_col_part_ptr,
    T,
    NK_rt,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    USE_GK: tl.constexpr,
    IEEE_DOT: tl.constexpr,
):
    i_ts = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_t = i_ts // NS
    i_s = i_ts % NS
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"

    o_t_strip = i_t * BT + i_s * BS + tl.arange(0, BS)
    o_t_chunk = i_t * BT + tl.arange(0, BT)
    m_ts = o_t_strip < T
    m_tc = o_t_chunk < T

    k_base = k_ptr + (bos * H + i_h) * K
    x_base = x_ptr + (bos * H + i_h) * K
    dq_base = dq_ptr + (bos * H + i_h) * K
    dk_base = dk_intra_ptr + (bos * H + i_h) * K

    b_kx = tl.zeros([BS, BT], dtype=tl.float32)
    b_ds = tl.zeros([BS, BT], dtype=tl.float32)
    b_kk = tl.zeros([BS, BT], dtype=tl.float32)

    for i_k in range(NK_rt):
        o_k = i_k * BK + tl.arange(0, BK)
        b_k_s = tl.load(k_base + o_t_strip[:, None] * (H * K) + o_k[None, :], mask=m_ts[:, None], other=0.0)
        b_k_c = tl.load(k_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None], other=0.0)
        b_x_c = tl.load(x_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None], other=0.0)
        b_dq_c = tl.load(dq_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None], other=0.0)

        b_kx += tl.dot(b_k_s, tl.trans(b_x_c), input_precision=DOT_PREC)
        b_ds += tl.dot(b_k_s, tl.trans(b_dq_c), input_precision=DOT_PREC)
        b_kk += tl.dot(b_k_s, tl.trans(b_k_c), input_precision=DOT_PREC)

    m_upper = (o_t_strip[:, None] <= o_t_chunk[None, :]) & m_ts[:, None] & m_tc[None, :]
    m_strict_lower = (o_t_strip[:, None] > o_t_chunk[None, :]) & m_ts[:, None] & m_tc[None, :]

    if USE_GK:
        b_g_s = tl.load(gk_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
        b_g_c = tl.load(gk_ptr + (bos + o_t_chunk) * H + i_h, mask=m_tc, other=0.0).to(tl.float32)
        b_m_up = tl.where(m_upper, tl.math.exp2(b_g_c[None, :] - b_g_s[:, None]), 0.0)
        b_m_lo = tl.where(m_strict_lower, tl.math.exp2(b_g_s[:, None] - b_g_c[None, :]), 0.0)
    else:
        b_m_up = tl.where(m_upper, 1.0, 0.0)
        b_m_lo = tl.where(m_strict_lower, 1.0, 0.0)

    b_rt_s = tl.load(r_tilde_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
    b_rt_c = tl.load(r_tilde_ptr + (bos + o_t_chunk) * H + i_h, mask=m_tc, other=0.0).to(tl.float32)

    b_w2 = b_m_up * b_rt_c[None, :] + b_m_lo * b_rt_s[:, None]
    b_w2 = tl.where(m_ts[:, None] & m_tc[None, :], b_w2, 0.0)

    b_s = b_kx * b_m_up
    b_dsm = b_ds * b_m_up
    b_kkw2 = b_kk * b_w2

    if USE_GK:
        b_tk_dot = tl.sum(b_kkw2 * b_kk, axis=1)
        b_dm = b_s * b_ds
        b_dm_row_sum = tl.sum(b_dm, axis=1)
        b_dm_col_part = tl.sum(b_dm, axis=0)
        tl.store(dg_col_part_ptr + ((bos + o_t_chunk) * H + i_h) * NS + i_s, b_dm_col_part, mask=m_tc)
        b_dg_prev = tl.load(dg_intra_ptr + (bos + o_t_strip) * H + i_h, mask=m_ts, other=0.0).to(tl.float32)
        b_dg = b_dg_prev - b_dm_row_sum - 0.5 * b_tk_dot
        tl.store(dg_intra_ptr + (bos + o_t_strip) * H + i_h, b_dg, mask=m_ts)

    b_s_cast = b_s.to(k_ptr.dtype.element_ty)
    b_dsm_cast = b_dsm.to(k_ptr.dtype.element_ty)
    b_kkw2_cast = b_kkw2.to(k_ptr.dtype.element_ty)

    for i_k in range(NK_rt):
        o_k = i_k * BK + tl.arange(0, BK)
        b_k_c = tl.load(k_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None], other=0.0)
        b_x_c = tl.load(x_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None], other=0.0)
        b_dq_c = tl.load(dq_base + o_t_chunk[:, None] * (H * K) + o_k[None, :], mask=m_tc[:, None], other=0.0)
        b_dk = (
            tl.dot(b_s_cast, b_dq_c, input_precision=DOT_PREC)
            + tl.dot(b_dsm_cast, b_x_c, input_precision=DOT_PREC)
            + tl.dot(b_kkw2_cast, b_k_c, input_precision=DOT_PREC)
        )
        tl.store(
            dk_base + o_t_strip[:, None] * (H * K) + o_k[None, :],
            (-b_dk).to(dk_intra_ptr.dtype.element_ty),
            mask=m_ts[:, None],
        )


@triton.jit
def _gka_bwd_dh_unified_rsi_kernel(
    dq_ptr,
    x_ptr,
    k_ptr,
    h_ptr,
    gk_ptr,
    w1_ptr,
    w_gh2_sum_ptr,
    dht_ptr,
    dh_out_ptr,
    dh0_ptr,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_DHT: tl.constexpr,
    STORE_DH0: tl.constexpr,
    IEEE_DOT: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"
    NS: tl.constexpr = BT // BS

    b_dh = tl.zeros([K, BV], dtype=tl.float32)
    if USE_DHT:
        b_dh -= tl.load(dht_ptr + i_bh * K * K + o_k[:, None] * K + o_v[None, :]).to(tl.float32)

    dq_base = dq_ptr + (bos * H + i_h) * K
    x_base = x_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    w1_base = w1_ptr + bos * H + i_h

    for i_c in range(NT - 1, -1, -1):
        i_tg = i_b * NT + i_c
        h_off = (i_tg * H + i_h) * K * K
        tl.store(dh_out_ptr + h_off + o_k[:, None] * K + o_v[None, :], b_dh.to(dh_out_ptr.dtype.element_ty))

        i_last = tl.minimum(BT, T - i_c * BT) - 1
        if USE_GK:
            b_g_last = tl.load(gk_ptr + (bos + i_c * BT + i_last) * H + i_h).to(tl.float32)
            b_dh = b_dh * tl.math.exp2(b_g_last)

        b_h_slab = tl.load(h_ptr + h_off + o_k[:, None] * K + o_v[None, :]).to(tl.float32)
        w_gh2 = tl.load(w_gh2_sum_ptr + i_tg * H + i_h).to(tl.float32)
        b_dh += 0.5 * w_gh2 * b_h_slab

        for i_s in range(NS):
            o_t = i_c * BT + i_s * BS + tl.arange(0, BS)
            m_t = o_t < T
            b_dq = tl.load(dq_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_x_v = tl.load(x_base + o_t[:, None] * (H * K) + o_v[None, :], mask=m_t[:, None], other=0.0)
            b_k = tl.load(k_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_k_v = tl.load(k_base + o_t[:, None] * (H * K) + o_v[None, :], mask=m_t[:, None], other=0.0)
            b_w1 = tl.load(w1_base + o_t * H, mask=m_t, other=0.0).to(tl.float32)
            if USE_GK:
                b_g = tl.load(gk_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
                b_dq_scaled = (b_dq.to(tl.float32) * tl.math.exp2(b_g)[:, None]).to(b_dq.dtype)
            else:
                b_dq_scaled = b_dq
            b_k_w1 = (b_k.to(tl.float32) * (0.5 * b_w1)[:, None]).to(b_k.dtype)
            b_dh += (
                tl.dot(tl.trans(b_dq_scaled), b_x_v, input_precision=DOT_PREC)
                + tl.dot(tl.trans(b_k_w1), b_k_v, input_precision=DOT_PREC)
            )

    if STORE_DH0:
        tl.store(dh0_ptr + i_bh * K * K + o_k[:, None] * K + o_v[None, :], (-b_dh).to(dh0_ptr.dtype.element_ty))


@triton.jit
def _gka_bwd_dk_dg_inter_strip_rsi_kernel(
    k_ptr,
    x_ptr,
    dq_ptr,
    h_ptr,
    dh_ptr,
    gk_ptr,
    w1_ptr,
    dk_solve_ptr,
    dk_gla_ptr,
    dk_out_ptr,
    dg_slabs_ptr,
    dg_last_part_ptr,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    NK: tl.constexpr,
    USE_GK: tl.constexpr,
    IEEE_DOT: tl.constexpr,
):
    i_k0 = tl.program_id(0).to(tl.int64)
    i_ts = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NS: tl.constexpr = BT // BS
    i_t = i_ts // NS
    i_s = i_ts % NS
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    i_tg = i_b * NT + i_t
    DOT_PREC: tl.constexpr = "ieee" if IEEE_DOT else "tf32"

    o_t = i_t * BT + i_s * BS + tl.arange(0, BS)
    m_t = o_t < T
    i_last = tl.minimum(BT, T - i_t * BT) - 1
    o_k0 = i_k0 * BK + tl.arange(0, BK)

    k_base = k_ptr + (bos * H + i_h) * K
    x_base = x_ptr + (bos * H + i_h) * K
    dq_base = dq_ptr + (bos * H + i_h) * K
    h_base = h_ptr + (i_tg * H + i_h) * K * K
    dh_base = dh_ptr + (i_tg * H + i_h) * K * K

    if USE_GK:
        b_g = tl.load(gk_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_g_last = tl.load(gk_ptr + (bos + i_t * BT + i_last) * H + i_h).to(tl.float32)
        b_gh = tl.math.exp2(b_g)
        b_gl = tl.math.exp2(b_g_last - b_g)
    else:
        b_g_last = 0.0
        b_gh = tl.full([BS], 1.0, dtype=tl.float32)
        b_gl = tl.full([BS], 1.0, dtype=tl.float32)

    b_w1 = tl.load(w1_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)

    b_dk_dh = tl.zeros([BS, BK], dtype=tl.float32)
    b_dk_h = tl.zeros([BS, BK], dtype=tl.float32)
    b_dq_h = tl.zeros([BS, BK], dtype=tl.float32)
    dh_h_dot = 0.0

    for i_k1 in range(NK):
        o_k1 = i_k1 * BK + tl.arange(0, BK)
        b_k1 = tl.load(k_base + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
        b_dq1 = tl.load(dq_base + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)

        b_h_blk = tl.load(h_base + o_k1[:, None] * K + o_k0[None, :])
        b_dh_blk = tl.load(dh_base + o_k1[:, None] * K + o_k0[None, :]).to(tl.float32)
        b_dh_blk_T = tl.load(dh_base + o_k0[:, None] * K + o_k1[None, :]).to(tl.float32)
        b_dh_sym = (b_dh_blk + tl.trans(b_dh_blk_T)).to(b_k1.dtype)

        b_k1_gl = (b_k1.to(tl.float32) * b_gl[:, None]).to(b_k1.dtype)
        b_k1_w1 = (b_k1.to(tl.float32) * b_w1[:, None]).to(b_k1.dtype)

        b_dk_dh += tl.dot(b_k1_gl, b_dh_sym, input_precision=DOT_PREC)
        b_dk_h += tl.dot(b_k1_w1, b_h_blk.to(b_k1.dtype), input_precision=DOT_PREC)
        b_dq_h += tl.dot(b_dq1, b_h_blk.to(b_dq1.dtype), input_precision=DOT_PREC)
        if i_s == 0:
            dh_h_dot += tl.sum(b_dh_blk * b_h_blk.to(tl.float32))

    off_tk = (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k0[None, :]
    b_dk_intra = tl.load(dk_solve_ptr + off_tk, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_dk_gla = tl.load(dk_gla_ptr + off_tk, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_dk_total = b_dk_intra - b_dk_dh - b_dk_h + b_dk_gla
    tl.store(dk_out_ptr + off_tk, b_dk_total.to(dk_out_ptr.dtype.element_ty), mask=m_t[:, None])

    if USE_GK:
        b_k0 = tl.load(k_base + o_t[:, None] * (H * K) + o_k0[None, :], mask=m_t[:, None], other=0.0)
        b_x0 = tl.load(x_base + o_t[:, None] * (H * K) + o_k0[None, :], mask=m_t[:, None], other=0.0)
        b_dgl = 0.5 * tl.sum(b_dk_dh * b_k0.to(tl.float32), axis=1)
        b_dg_slab = (
            tl.sum(b_dq_h * b_gh[:, None] * b_x0.to(tl.float32), axis=1)
            - b_dgl
            - 0.5 * tl.sum(b_dk_h * b_k0.to(tl.float32), axis=1)
        )
        b_dg_last = tl.sum(b_dgl, axis=0)
        if i_s == 0:
            b_dg_last += dh_h_dot * tl.math.exp2(b_g_last)
        tl.store(dg_slabs_ptr + ((bos + o_t) * H + i_h) * NK + i_k0, b_dg_slab, mask=m_t)
        tl.store(dg_last_part_ptr + (((i_b * NT + i_t) * H + i_h) * NS + i_s) * NK + i_k0, b_dg_last)


@triton.jit
def _gka_bwd_dg_rev_cumsum_rsi_kernel(
    dg_intra_ptr,
    dg_col_part_ptr,
    dg_slabs_ptr,
    dg_last_part_ptr,
    dg_gla_ptr,
    dg_gla_last_ptr,
    dg_out_ptr,
    dgk_out_ptr,
    T,
    NT,
    H: tl.constexpr,
    BT: tl.constexpr,
    NK_KK: tl.constexpr,
    NK_GLA: tl.constexpr,
    NS_INTRA: tl.constexpr,
    NS_INTER: tl.constexpr,
    NS_GLA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_G: tl.constexpr,
    SAME_GATE: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    i_last = tl.minimum(BT, T - i_t * BT) - 1
    m_last = tl.arange(0, BT) == i_last

    b_dg_kk = tl.zeros([BT], dtype=tl.float32)
    if USE_GK:
        b_dg_base = tl.load(dg_intra_ptr + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
        for i_s in range(NS_INTRA):
            b_dg_base += tl.load(dg_col_part_ptr + ((bos + o_t) * H + i_h) * NS_INTRA + i_s, mask=m_t, other=0.0).to(tl.float32)
        for i_k in range(NK_KK):
            b_dg_base += tl.load(dg_slabs_ptr + ((bos + o_t) * H + i_h) * NK_KK + i_k, mask=m_t, other=0.0).to(tl.float32)
        dg_kk_last = 0.0
        for i_s in range(NS_INTER):
            for i_k in range(NK_KK):
                dg_kk_last += tl.load(dg_last_part_ptr + (((i_b * NT + i_t) * H + i_h) * NS_INTER + i_s) * NK_KK + i_k).to(tl.float32)
        b_dg_base = tl.where(m_last, b_dg_base + dg_kk_last, b_dg_base)
        b_dg_kk = -b_dg_base

    b_dg_gla = tl.zeros([BT], dtype=tl.float32)
    if USE_G:
        for i_k in range(NK_GLA):
            b_dg_gla += tl.load(dg_gla_ptr + ((bos + o_t) * H + i_h) * NK_GLA + i_k, mask=m_t, other=0.0).to(tl.float32)
        dg_g_last = 0.0
        for i_s in range(NS_GLA):
            for i_k in range(NK_GLA):
                dg_g_last += tl.load(dg_gla_last_ptr + (((i_b * NT + i_t) * H + i_h) * NS_GLA + i_s) * NK_GLA + i_k).to(tl.float32)
        b_dg_gla = tl.where(m_last, b_dg_gla + dg_g_last, b_dg_gla)

    if SAME_GATE:
        b_tot = tl.where(m_t, b_dg_kk + b_dg_gla, 0.0)
        b_sum = tl.sum(b_tot, axis=0)
        b_rev = b_sum - tl.cumsum(b_tot, axis=0) + b_tot
        tl.store(dg_out_ptr + (bos + o_t) * H + i_h, b_rev.to(dg_out_ptr.dtype.element_ty), mask=m_t)
    else:
        if USE_G:
            b_tot_g = tl.where(m_t, b_dg_gla, 0.0)
            b_sum_g = tl.sum(b_tot_g, axis=0)
            b_rev_g = b_sum_g - tl.cumsum(b_tot_g, axis=0) + b_tot_g
            tl.store(dg_out_ptr + (bos + o_t) * H + i_h, b_rev_g.to(dg_out_ptr.dtype.element_ty), mask=m_t)
        if USE_GK:
            b_tot_gk = tl.where(m_t, b_dg_kk, 0.0)
            b_sum_gk = tl.sum(b_tot_gk, axis=0)
            b_rev_gk = b_sum_gk - tl.cumsum(b_tot_gk, axis=0) + b_tot_gk
            tl.store(dgk_out_ptr + (bos + o_t) * H + i_h, b_rev_gk.to(dgk_out_ptr.dtype.element_ty), mask=m_t)


# ==============================================================================
# 3. Autograd Function & Python Wrapper (`chunk_gka_rsi`)
# ==============================================================================

def chunk_gka_fwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    alpha: Optional[torch.Tensor],
    scale: float,
    ridge_ratio: float,
    num_iter: int,
    h0_kk: Optional[torch.Tensor],
    h0_kv: Optional[torch.Tensor],
    output_final_state: bool,
    mutation_mode: int = 0,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)
    ieee_dot = q.dtype == torch.float32

    same_gate = (g is not None) and (gk is not None) and (g.data_ptr() == gk.data_ptr())
    g_cs = None
    gk_cs = None
    if g is not None:
        g_cs = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
        _gka_local_cumsum_fwd_rsi_kernel[(NT, B * H)](
            g, g_cs, T=T, H=H, BT=BT, SCALE=RCP_LN2, num_warps=2, num_stages=1
        )
    if gk is not None:
        if same_gate:
            gk_cs = g_cs
        else:
            gk_cs = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
            _gka_local_cumsum_fwd_rsi_kernel[(NT, B * H)](
                gk, gk_cs, T=T, H=H, BT=BT, SCALE=RCP_LN2, num_warps=2, num_stages=1
            )

    BV_kk = 32 if (K <= 64 or not ieee_dot) else 16
    BS_kk = 64 if not ieee_dot else 16
    NV_kk = K // BV_kk
    h_kk = torch.empty(B, NT, H, K, K, device=q.device, dtype=q.dtype)
    ht_kk = torch.empty(B, H, K, K, device=q.device, dtype=torch.float32) if output_final_state else None
    khk_part = torch.empty(B, T, H, NV_kk, device=q.device, dtype=torch.float32)
    h_norm_part = torch.empty(B, NT, H, NV_kk, device=q.device, dtype=torch.float32)

    _gka_fwd_h_kk_fro_part_rsi_kernel[(NV_kk, B * H)](
        k, gk_cs, h_kk, h0_kk, ht_kk, khk_part, h_norm_part, T, NT,
        H=H, K=K, BT=BT, BS=BS_kk, BV=BV_kk, NV=NV_kk,
        USE_GK=gk_cs is not None,
        USE_INIT_STATE=h0_kk is not None,
        STORE_FINAL_STATE=output_final_state,
        IEEE_DOT=ieee_dot,
        num_warps=4, num_stages=1,
    )

    BK_fro = 32 if K >= 32 else 16
    NK_fro = K // BK_fro
    fro = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    if not ieee_dot:
        _gka_fro_full_bf16_rsi_kernel[(NT, B * H)](
            k, gk_cs, khk_part, h_norm_part, fro, T, NT, NK_fro,
            H=H, K=K, BT=BT, BK=BK_fro, NV=NV_kk,
            USE_GK=gk_cs is not None,
            num_warps=8, num_stages=1,
        )
    else:
        BS_fro = 16
        _gka_fro_strip_rsi_kernel[(NT * (BT // BS_fro), B * H)](
            k, gk_cs, khk_part, h_norm_part, fro, T, NT, NK_fro,
            H=H, K=K, BT=BT, BS=BS_fro, BK=BK_fro, NV=NV_kk,
            USE_GK=gk_cs is not None,
            IEEE_DOT=ieee_dot,
            num_warps=8, num_stages=1,
        )

    x = torch.empty_like(q)
    q_mix = torch.empty_like(q) if alpha is not None else x
    if not ieee_dot:
        BS_solve = 32 if K <= 64 else 16
        _gka_solve_fwd_bf16_rsi_kernel[(NT * (BT // BS_solve), B * H)](
            q, k, h_kk, gk_cs, fro, x, ridge_ratio, num_iter, T,
            H=H, K=K, BT=BT, BS=BS_solve, BK=K,
            USE_GK=gk_cs is not None,
            MUTATION_MODE=mutation_mode,
            num_warps=4, num_stages=1,
        )
    else:
        _gka_solve_fwd_fp32_rsi_kernel[(T, B * H)](
            q, k, h_kk, gk_cs, fro, alpha, x, q_mix, ridge_ratio, num_iter, T,
            H=H, K=K, BT=BT, BK=K,
            USE_GK=gk_cs is not None,
            USE_ALPHA=alpha is not None,
            MUTATION_MODE=mutation_mode,
            num_warps=4 if K <= 64 else 8, num_stages=1,
        )

    BV_kv = min(32 if (K <= 64 or not ieee_dot) else 16, V)
    BS_kv = 64 if not ieee_dot else 16
    NV_kv = triton.cdiv(V, BV_kv)
    h_kv = torch.empty(B, NT, H, K, V, device=q.device, dtype=q.dtype)
    ht_kv = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if output_final_state else None
    _gka_fwd_h_kv_rsi_kernel[(NV_kv, B * H)](
        k, v, g_cs, h_kv, h0_kv, ht_kv, T, NT,
        H=H, K=K, V=V, BT=BT, BS=BS_kv, BV=BV_kv,
        USE_G=g_cs is not None,
        USE_INIT_STATE=h0_kv is not None,
        STORE_FINAL_STATE=output_final_state,
        IEEE_DOT=ieee_dot,
        num_warps=4, num_stages=1,
    )

    BS_o = 32 if not ieee_dot else 16
    BK_o = 64 if (not ieee_dot and K == 64 and V == 64) else min(32, K)
    BV_o = 64 if (not ieee_dot and K == 64 and V == 64) else min(32, V)
    NK_o = triton.cdiv(K, BK_o)
    NV_o = triton.cdiv(V, BV_o)
    o = torch.empty_like(v)
    _gka_fwd_o_rsi_kernel[(NV_o, NT * (BT // BS_o), B * H)](
        q, x, alpha if alpha is not None else q, q_mix, k, v, g_cs, h_kv, o, scale, T, NT, NK_o,
        H=H, K=K, V=V, BT=BT, BS=BS_o, BK=BK_o, BV=BV_o,
        USE_G=g_cs is not None,
        USE_ALPHA=(alpha is not None) and (not ieee_dot),
        IEEE_DOT=ieee_dot,
        MUTATION_MODE=mutation_mode,
        num_warps=4, num_stages=1,
    )

    return o, ht_kk, ht_kv, g_cs, gk_cs, h_kk, h_kv, x, q_mix, fro


def chunk_gka_bwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: Optional[torch.Tensor],
    gk_cs: Optional[torch.Tensor],
    alpha: Optional[torch.Tensor],
    h_kk: torch.Tensor,
    h_kv: torch.Tensor,
    x: torch.Tensor,
    q_mix: torch.Tensor,
    fro: torch.Tensor,
    h0_kk: Optional[torch.Tensor],
    h0_kv: Optional[torch.Tensor],
    do: torch.Tensor,
    dht_kk: Optional[torch.Tensor],
    dht_kv: Optional[torch.Tensor],
    scale: float,
    ridge_ratio: float,
    num_iter: int,
    same_gate: bool,
    mutation_mode: int = 0,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)
    ieee_dot = q.dtype == torch.float32

    BV_kv = min(32 if (K <= 64 or not ieee_dot) else 16, V)
    BS_kv = 64 if not ieee_dot else 16
    NV_kv = triton.cdiv(V, BV_kv)
    dh_kv = torch.empty_like(h_kv)
    dh0_kv = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if h0_kv is not None else None
    _gka_bwd_dh_kv_rsi_kernel[(NV_kv, B * H)](
        q_mix, do, g_cs, dht_kv, dh_kv, dh0_kv, scale, T, NT,
        H=H, K=K, V=V, BT=BT, BS=BS_kv, BV=BV_kv,
        USE_G=g_cs is not None,
        USE_DHT=dht_kv is not None,
        STORE_DH0=h0_kv is not None,
        IEEE_DOT=ieee_dot,
        num_warps=4, num_stages=1,
    )

    BS_dv = 32 if not ieee_dot else 16
    NS_dv = BT // BS_dv
    BK_dv = min(64 if not ieee_dot else 32, K)
    BV_dv = min(64 if not ieee_dot else 32, V)
    NV_dv = triton.cdiv(V, BV_dv)

    dv = torch.empty_like(v)
    _gka_bwd_dv_gla_rsi_kernel[(NV_dv, NT * NS_dv, B * H)](
        q_mix, k, g_cs, do, dh_kv, dv, scale, T, NT,
        H=H, K=K, V=V, BT=BT, BS=BS_dv, BK=BK_dv, BV=BV_dv,
        USE_G=g_cs is not None,
        IEEE_DOT=ieee_dot,
        num_warps=4, num_stages=1,
    )

    BS_gla = 16
    NS_gla = BT // BS_gla
    BK_gla = min(64 if not ieee_dot else 32, K)
    BV_gla = min(64 if not ieee_dot else 32, V)
    NK_gla = triton.cdiv(K, BK_gla)
    NV_gla = triton.cdiv(V, BV_gla)
    dx = torch.empty_like(q)
    dq_alpha = torch.empty_like(q) if alpha is not None else None
    dk_gla = torch.empty_like(k)
    dg_gla = torch.empty(B, T, H, NK_gla, device=q.device, dtype=torch.float32) if g_cs is not None else None
    dg_gla_last = torch.empty(B, NT, H, NS_gla, NK_gla, device=q.device, dtype=torch.float32) if g_cs is not None else None
    dalpha_part = torch.empty(B, T, H, NK_gla, device=q.device, dtype=torch.float32) if alpha is not None else None

    _gka_bwd_dqkg_gla_rsi_kernel[(NK_gla, NT * NS_gla, B * H)](
        q, x, q_mix, k, v, g_cs, alpha, h_kv, dh_kv, do,
        dx, dq_alpha if alpha is not None else dx, dk_gla,
        dg_gla if g_cs is not None else dx,
        dg_gla_last if g_cs is not None else dx,
        dalpha_part if alpha is not None else dx,
        scale, T, NT, NV_gla,
        H=H, K=K, V=V, BT=BT, BS=BS_gla, BK=BK_gla, BV=BV_gla, NK=NK_gla,
        USE_G=g_cs is not None,
        USE_ALPHA=alpha is not None,
        IEEE_DOT=ieee_dot,
        MUTATION_MODE=mutation_mode,
        num_warps=4 if (not ieee_dot and K <= 64 and V <= 64) else 8, num_stages=1,
    )
    dalpha = dalpha_part.sum(dim=-1) if alpha is not None else None

    # Adjoint Chebyshev solve: (H_t + lamb_t I) dq_solve = dx
    dq_solve = torch.empty_like(q)
    if not ieee_dot:
        BS_solve = 32 if K <= 64 else 16
        _gka_solve_fwd_bf16_rsi_kernel[(NT * (BT // BS_solve), B * H)](
            dx, k, h_kk, gk_cs, fro, dq_solve, ridge_ratio, num_iter, T,
            H=H, K=K, BT=BT, BS=BS_solve, BK=K,
            USE_GK=gk_cs is not None,
            MUTATION_MODE=mutation_mode,
            num_warps=4, num_stages=1,
        )
    else:
        _gka_solve_fwd_fp32_rsi_kernel[(T, B * H)](
            dx, k, h_kk, gk_cs, fro, None, dq_solve, dq_solve, ridge_ratio, num_iter, T,
            H=H, K=K, BT=BT, BK=K,
            USE_GK=gk_cs is not None,
            USE_ALPHA=False,
            MUTATION_MODE=mutation_mode,
            num_warps=4 if K <= 64 else 8, num_stages=1,
        )

    BK_w1 = min(32, K)
    w1 = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    r_tilde = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    w_gh2_sum = torch.empty(B, NT, H, device=q.device, dtype=torch.float32)
    dg_intra = torch.empty(B, T, H, device=q.device, dtype=torch.float32) if gk_cs is not None else None

    _gka_bwd_w1_vec_rsi_kernel[(NT, B * H)](
        dq_solve, x, fro, gk_cs, w1, r_tilde, w_gh2_sum,
        dg_intra if gk_cs is not None else w1,
        ridge_ratio, T, NT,
        H=H, K=K, BT=BT, BK=BK_w1,
        USE_GK=gk_cs is not None,
        MUTATION_MODE=mutation_mode,
        num_warps=4, num_stages=1,
    )

    BS_intra = 32 if (not ieee_dot and K >= 64) else 16
    NS_intra = BT // BS_intra
    BK_intra = min(64 if (not ieee_dot and K >= 64) else 32, K)
    NK_intra = K // BK_intra
    dk_solve = torch.empty_like(k)
    dg_col_part = torch.empty(B, T, H, NS_intra, device=q.device, dtype=torch.float32) if gk_cs is not None else None
    _gka_bwd_intra_strip_rsi_kernel[(NT * NS_intra, B * H)](
        k, x, dq_solve, gk_cs, r_tilde, dk_solve,
        dg_intra if gk_cs is not None else w1,
        dg_col_part if gk_cs is not None else w1,
        T, NK_intra,
        H=H, K=K, BT=BT, BS=BS_intra, BK=BK_intra, NS=NS_intra,
        USE_GK=gk_cs is not None,
        IEEE_DOT=ieee_dot,
        num_warps=8, num_stages=1,
    )

    BV_dh = min(32 if K <= 64 else 16, K)
    BS_dh = 32 if not ieee_dot else 16
    NV_dh = K // BV_dh
    dh_uni = torch.empty_like(h_kk)
    dh0_kk = torch.empty(B, H, K, K, device=q.device, dtype=torch.float32) if h0_kk is not None else None
    _gka_bwd_dh_unified_rsi_kernel[(NV_dh, B * H)](
        dq_solve, x, k, h_kk, gk_cs, w1, w_gh2_sum, dht_kk, dh_uni, dh0_kk, T, NT,
        H=H, K=K, BT=BT, BS=BS_dh, BV=BV_dh,
        USE_GK=gk_cs is not None,
        USE_DHT=dht_kk is not None,
        STORE_DH0=h0_kk is not None,
        IEEE_DOT=ieee_dot,
        num_warps=4, num_stages=1,
    )

    BS_inter = 32 if not ieee_dot else 16
    NS_inter = BT // BS_inter
    BK_kk = 64 if (not ieee_dot and K == 64) else min(32, K)
    NK_kk = K // BK_kk
    dk_out = torch.empty_like(k)
    dg_slabs = torch.empty(B, T, H, NK_kk, device=q.device, dtype=torch.float32) if gk_cs is not None else None
    dg_last_part = torch.empty(B, NT, H, NS_inter, NK_kk, device=q.device, dtype=torch.float32) if gk_cs is not None else None
    _gka_bwd_dk_dg_inter_strip_rsi_kernel[(NK_kk, NT * NS_inter, B * H)](
        k, x, dq_solve, h_kk, dh_uni, gk_cs, w1, dk_solve, dk_gla, dk_out,
        dg_slabs if gk_cs is not None else w1,
        dg_last_part if gk_cs is not None else w1,
        T, NT,
        H=H, K=K, BT=BT, BS=BS_inter, BK=BK_kk, NK=NK_kk,
        USE_GK=gk_cs is not None,
        IEEE_DOT=ieee_dot,
        num_warps=4, num_stages=1,
    )

    dq_out = (dq_solve + dq_alpha) if alpha is not None else dq_solve
    dg_out = torch.empty(B, T, H, device=q.device, dtype=torch.float32) if g_cs is not None else None
    dgk_out = (
        dg_out
        if same_gate
        else (torch.empty(B, T, H, device=q.device, dtype=torch.float32) if gk_cs is not None else None)
    )
    if (g_cs is not None) or (gk_cs is not None):
        dummy = w1
        _gka_bwd_dg_rev_cumsum_rsi_kernel[(NT, B * H)](
            dg_intra if gk_cs is not None else dummy,
            dg_col_part if gk_cs is not None else dummy,
            dg_slabs if gk_cs is not None else dummy,
            dg_last_part if gk_cs is not None else dummy,
            dg_gla if g_cs is not None else dummy,
            dg_gla_last if g_cs is not None else dummy,
            dg_out if g_cs is not None else dummy,
            dgk_out if gk_cs is not None else dummy,
            T, NT,
            H=H, BT=BT, NK_KK=NK_kk, NK_GLA=NK_gla, NS_INTRA=NS_intra, NS_INTER=NS_inter, NS_GLA=NS_gla,
            USE_GK=gk_cs is not None,
            USE_G=g_cs is not None,
            SAME_GATE=same_gate,
            num_warps=4, num_stages=1,
        )

    return dq_out, dk_out, dv, dg_out, dgk_out, dalpha, dh0_kk, dh0_kv


class ChunkGKARSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        gk: Optional[torch.Tensor],
        alpha: Optional[torch.Tensor],
        scale: float,
        ridge_ratio: float,
        num_iter: int,
        h0_kk: Optional[torch.Tensor],
        h0_kv: Optional[torch.Tensor],
        output_final_state: bool,
        mutation_mode: int,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous() if g is not None else None
        gk = gk.contiguous() if gk is not None else None
        alpha = alpha.contiguous() if alpha is not None else None
        h0_kk = h0_kk.contiguous() if h0_kk is not None else None
        h0_kv = h0_kv.contiguous() if h0_kv is not None else None

        same_gate = (g is not None) and (gk is not None) and (g.data_ptr() == gk.data_ptr())
        o, ht_kk, ht_kv, g_cs, gk_cs, h_kk, h_kv, x, q_mix, fro = chunk_gka_fwd_rsi(
            q=q, k=k, v=v, g=g, gk=gk, alpha=alpha,
            scale=scale, ridge_ratio=ridge_ratio, num_iter=num_iter,
            h0_kk=h0_kk, h0_kv=h0_kv, output_final_state=output_final_state,
            mutation_mode=mutation_mode,
        )
        ctx.save_for_backward(q, k, v, g_cs, gk_cs, alpha, h_kk, h_kv, x, q_mix, fro, h0_kk, h0_kv)
        ctx.scale = scale
        ctx.ridge_ratio = ridge_ratio
        ctx.num_iter = num_iter
        ctx.same_gate = same_gate
        ctx.mutation_mode = mutation_mode
        ctx.g_dtype = g.dtype if g is not None else None
        ctx.gk_dtype = gk.dtype if gk is not None else None
        ctx.alpha_dtype = alpha.dtype if alpha is not None else None
        return o, ht_kk, ht_kv

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht_kk: Optional[torch.Tensor], dht_kv: Optional[torch.Tensor]):
        q, k, v, g_cs, gk_cs, alpha, h_kk, h_kv, x, q_mix, fro, h0_kk, h0_kv = ctx.saved_tensors
        do = do.contiguous()
        dht_kk = dht_kk.contiguous() if dht_kk is not None else None
        dht_kv = dht_kv.contiguous() if dht_kv is not None else None

        dq, dk, dv, dg, dgk, dalpha, dh0_kk, dh0_kv = chunk_gka_bwd_rsi(
            q=q, k=k, v=v, g_cs=g_cs, gk_cs=gk_cs, alpha=alpha,
            h_kk=h_kk, h_kv=h_kv, x=x, q_mix=q_mix, fro=fro,
            h0_kk=h0_kk, h0_kv=h0_kv, do=do, dht_kk=dht_kk, dht_kv=dht_kv,
            scale=ctx.scale, ridge_ratio=ctx.ridge_ratio, num_iter=ctx.num_iter,
            same_gate=ctx.same_gate, mutation_mode=ctx.mutation_mode,
        )
        if ctx.same_gate:
            # When the caller passes the same tensor `g` for both `g` and `gk`, return half in each slot
            # or return full in `dg` and zero in `dgk` so PyTorch's autograd accumulation into `g.grad` is exact!
            dg_ret = (dg * 0.5).to(ctx.g_dtype)
            dgk_ret = (dg * 0.5).to(ctx.gk_dtype)
        else:
            dg_ret = dg.to(ctx.g_dtype) if dg is not None else None
            dgk_ret = dgk.to(ctx.gk_dtype) if dgk is not None else None
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg_ret,
            dgk_ret,
            dalpha.to(ctx.alpha_dtype) if dalpha is not None else None,
            None,
            None,
            None,
            dh0_kk,
            dh0_kv,
            None,
            None,
        )


def chunk_gka_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    mutation_mode: int = 0,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    h0_kk, h0_kv = initial_state if initial_state is not None else (None, None)
    o, ht_kk, ht_kv = ChunkGKARSIFunction.apply(
        q, k, v, g, gk, alpha, scale, ridge_ratio, num_iter, h0_kk, h0_kv, output_final_state, mutation_mode
    )
    final_state = (ht_kk, ht_kv) if output_final_state else None
    return o, final_state


# ==============================================================================
# 4. Online Decoding Kernels (`gka_decode_rsi`, Track 2)
# ==============================================================================

@triton.jit
def _gka_decode_solve_rsi_kernel(
    q_ptr,
    k_ptr,
    gk_ptr,
    alpha_ptr,
    q_mix_ptr,
    h0_kk_ptr,
    ht_kk_ptr,
    ridge_ratio,
    num_iter,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    HALF_K: tl.constexpr = BK // 2
    o_r0 = tl.arange(0, HALF_K)
    o_r1 = HALF_K + tl.arange(0, HALF_K)

    if USE_INIT_STATE:
        h0_base = h0_kk_ptr + i_bh * K * K
        b_h00 = tl.load(h0_base + o_r0[:, None] * K + o_r0[None, :]).to(tl.float32)
        b_h01 = tl.load(h0_base + o_r0[:, None] * K + o_r1[None, :]).to(tl.float32)
        b_h11 = tl.load(h0_base + o_r1[:, None] * K + o_r1[None, :]).to(tl.float32)
    else:
        b_h00 = tl.zeros([HALF_K, HALF_K], dtype=tl.float32)
        b_h01 = tl.zeros([HALF_K, HALF_K], dtype=tl.float32)
        b_h11 = tl.zeros([HALF_K, HALF_K], dtype=tl.float32)

    rho_sq4 = 1.0 / (2.0 * (2.0 * ridge_ratio + 1.0))
    rho_sq4 = rho_sq4 * rho_sq4
    step_scale = 2.0 / (2.0 * ridge_ratio + 1.0)

    for i_t in range(T):
        tok_off = ((bos + i_t) * H + i_h) * K
        b_q0 = tl.load(q_ptr + tok_off + o_r0).to(tl.float32)
        b_q1 = tl.load(q_ptr + tok_off + o_r1).to(tl.float32)
        b_k0 = tl.load(k_ptr + tok_off + o_r0).to(tl.float32)
        b_k1 = tl.load(k_ptr + tok_off + o_r1).to(tl.float32)
        if USE_GK:
            gk_val = tl.load(gk_ptr + (bos + i_t) * H + i_h).to(tl.float32)
            dec = tl.exp(gk_val)
            b_h00 = b_h00 * dec
            b_h01 = b_h01 * dec
            b_h11 = b_h11 * dec
        b_h00 += b_k0[:, None] * b_k0[None, :]
        b_h01 += b_k0[:, None] * b_k1[None, :]
        b_h11 += b_k1[:, None] * b_k1[None, :]

        fro_val = tl.sqrt(tl.sum(b_h00 * b_h00) + 2.0 * tl.sum(b_h01 * b_h01) + tl.sum(b_h11 * b_h11))
        lamb_val = ridge_ratio * fro_val
        step_val = step_scale / fro_val

        b_xp0 = tl.zeros([HALF_K], dtype=tl.float32)
        b_xp1 = tl.zeros([HALF_K], dtype=tl.float32)
        b_x0 = step_val * b_q0
        b_x1 = step_val * b_q1
        w_s = 2.0
        for _ in range(num_iter):
            w_s = 1.0 / (1.0 - rho_sq4 * w_s)
            b_hx0 = tl.sum(b_h00 * b_x0[None, :], axis=1) + tl.sum(b_h01 * b_x1[None, :], axis=1)
            b_hx1 = tl.sum(b_h01 * b_x0[:, None], axis=0) + tl.sum(b_h11 * b_x1[None, :], axis=1)
            b_r0 = b_hx0 + lamb_val * b_x0 - b_q0
            b_r1 = b_hx1 + lamb_val * b_x1 - b_q1
            b_d0 = step_val * w_s * b_r0 + (w_s - 1.0) * b_xp0
            b_d1 = step_val * w_s * b_r1 + (w_s - 1.0) * b_xp1
            b_xp0 = b_x0
            b_xp1 = b_x1
            b_x0 = w_s * b_x0 - b_d0
            b_x1 = w_s * b_x1 - b_d1

        if USE_ALPHA:
            alpha_val = tl.load(alpha_ptr + (bos + i_t) * H + i_h).to(tl.float32)
            b_qm0 = b_q0 + alpha_val * (b_x0 - b_q0)
            b_qm1 = b_q1 + alpha_val * (b_x1 - b_q1)
        else:
            b_qm0 = b_x0
            b_qm1 = b_x1
        tl.store(q_mix_ptr + tok_off + o_r0, b_qm0.to(q_mix_ptr.dtype.element_ty))
        tl.store(q_mix_ptr + tok_off + o_r1, b_qm1.to(q_mix_ptr.dtype.element_ty))

    if STORE_FINAL_STATE:
        ht_base = ht_kk_ptr + i_bh * K * K
        tl.store(ht_base + o_r0[:, None] * K + o_r0[None, :], b_h00.to(ht_kk_ptr.dtype.element_ty))
        tl.store(ht_base + o_r0[:, None] * K + o_r1[None, :], b_h01.to(ht_kk_ptr.dtype.element_ty))
        tl.store(ht_base + o_r1[:, None] * K + o_r0[None, :], tl.trans(b_h01).to(ht_kk_ptr.dtype.element_ty))
        tl.store(ht_base + o_r1[:, None] * K + o_r1[None, :], b_h11.to(ht_kk_ptr.dtype.element_ty))


@triton.jit
def _gka_decode_readout_rsi_kernel(
    q_mix_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    o_ptr,
    h0_kv_ptr,
    ht_kv_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INIT_STATE:
        b_h += tl.load(h0_kv_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)

    for i_t in range(T):
        tok_k = ((bos + i_t) * H + i_h) * K + o_k
        tok_v = ((bos + i_t) * H + i_h) * V + o_v
        b_q = tl.load(q_mix_ptr + tok_k).to(tl.float32)
        b_k = tl.load(k_ptr + tok_k).to(tl.float32)
        b_v = tl.load(v_ptr + tok_v, mask=m_v, other=0.0).to(tl.float32)
        if USE_G:
            g_val = tl.load(g_ptr + (bos + i_t) * H + i_h).to(tl.float32)
            b_h = b_h * tl.exp(g_val)
        b_h += b_k[:, None] * b_v[None, :]
        b_o = scale * tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(o_ptr + tok_v, b_o.to(o_ptr.dtype.element_ty), mask=m_v)

    if STORE_FINAL_STATE:
        tl.store(ht_kv_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :], b_h.to(ht_kv_ptr.dtype.element_ty), mask=m_v[None, :])


def gka_decode_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = True,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    if T >= 16 and K >= 128 and (B * T >= 64):
        return chunk_gka_rsi(
            q=q, k=k, v=v, g=g, gk=gk, alpha=alpha, scale=scale,
            ridge_ratio=ridge_ratio, num_iter=num_iter,
            initial_state=initial_state, output_final_state=output_final_state,
        )
    h0_kk, h0_kv = initial_state if initial_state is not None else (None, None)

    q_mix = torch.empty_like(q)
    ht_kk = torch.empty(B, H, K, K, device=q.device, dtype=torch.float32) if output_final_state else None
    ht_kv = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if output_final_state else None
    o = torch.empty_like(v)

    _gka_decode_solve_rsi_kernel[(B * H,)](
        q, k, gk, alpha, q_mix, h0_kk, ht_kk,
        ridge_ratio, num_iter, T,
        H=H, K=K, BK=K,
        USE_GK=gk is not None,
        USE_ALPHA=alpha is not None,
        USE_INIT_STATE=h0_kk is not None,
        STORE_FINAL_STATE=output_final_state,
        num_warps=4 if K <= 64 else 8, num_stages=1,
    )
    BV = min(32, V)
    NV = triton.cdiv(V, BV)
    _gka_decode_readout_rsi_kernel[(NV, B * H)](
        q_mix, k, v, g, o, h0_kv, ht_kv,
        scale, T,
        H=H, K=K, V=V, BK=K, BV=BV,
        USE_G=g is not None,
        USE_INIT_STATE=h0_kv is not None,
        STORE_FINAL_STATE=output_final_state,
        num_warps=4, num_stages=1,
    )
    final_state = (ht_kk, ht_kv) if output_final_state else None
    return o, final_state


# ==============================================================================
# 5. Superfused Frontend/Epilogue & 5-Stage `GatedKalmaNetBlockRSI` (Track 3 & 5)
# ==============================================================================

@triton.jit
def _gka_qkv_l2norm_beta_fwd_rsi_kernel(
    q_in_ptr,
    k_in_ptr,
    v_in_ptr,
    b_raw_ptr,
    q_out_ptr,
    q_rstd_ptr,
    k_out_ptr,
    k_rstd_ptr,
    v_out_ptr,
    beta_out_ptr,
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

    q = tl.load(q_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    k = tl.load(k_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    v = tl.load(v_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    b_raw = tl.load(b_raw_ptr + i_row).to(tl.float32)

    q_rstd = tl.rsqrt(tl.sum(q * q, axis=0) + EPS)
    k_rstd = tl.rsqrt(tl.sum(k * k, axis=0) + EPS)
    beta = tl.sigmoid(b_raw) + 1e-6

    q_hat = q * q_rstd
    k_tilde = (k * k_rstd) * beta
    v_tilde = v * beta

    tl.store(q_rstd_ptr + i_row, q_rstd)
    tl.store(k_rstd_ptr + i_row, k_rstd)
    tl.store(beta_out_ptr + i_row, beta)
    tl.store(q_out_ptr + off, q_hat.to(q_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(k_out_ptr + off, k_tilde.to(k_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(v_out_ptr + off, v_tilde.to(v_out_ptr.dtype.element_ty), mask=m_k)


@triton.jit
def _gka_qkv_l2norm_beta_bwd_rsi_kernel(
    q_in_ptr,
    q_rstd_ptr,
    k_in_ptr,
    k_rstd_ptr,
    v_in_ptr,
    b_raw_ptr,
    dq_hat_ptr,
    dk_tilde_ptr,
    dv_tilde_ptr,
    dq_in_ptr,
    dk_in_ptr,
    dv_in_ptr,
    db_raw_ptr,
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

    q = tl.load(q_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    k = tl.load(k_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    v = tl.load(v_in_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    q_rstd = tl.load(q_rstd_ptr + i_row).to(tl.float32)
    k_rstd = tl.load(k_rstd_ptr + i_row).to(tl.float32)
    b_raw = tl.load(b_raw_ptr + i_row).to(tl.float32)

    dq_hat = tl.load(dq_hat_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    dk_tilde = tl.load(dk_tilde_ptr + off, mask=m_k, other=0.0).to(tl.float32)
    dv_tilde = tl.load(dv_tilde_ptr + off, mask=m_k, other=0.0).to(tl.float32)

    sig_b = tl.sigmoid(b_raw)
    beta = sig_b + 1e-6
    q_hat = q * q_rstd
    k_hat = k * k_rstd

    dbeta = tl.sum(dk_tilde * k_hat + dv_tilde * v, axis=0)
    db_raw = dbeta * sig_b * (1.0 - sig_b)

    dk_hat = dk_tilde * beta
    dq_in = (dq_hat - q_hat * tl.sum(dq_hat * q_hat, axis=0)) * q_rstd
    dk_in = (dk_hat - k_hat * tl.sum(dk_hat * k_hat, axis=0)) * k_rstd
    dv_in = dv_tilde * beta

    tl.store(dq_in_ptr + off, dq_in.to(dq_in_ptr.dtype.element_ty), mask=m_k)
    tl.store(dk_in_ptr + off, dk_in.to(dk_in_ptr.dtype.element_ty), mask=m_k)
    tl.store(dv_in_ptr + off, dv_in.to(dv_in_ptr.dtype.element_ty), mask=m_k)
    tl.store(db_raw_ptr + i_row, db_raw.to(db_raw_ptr.dtype.element_ty))


class FusedGKAQKVNormBetaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b_raw: torch.Tensor, eps: float = 1e-6):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        b_raw = b_raw.contiguous()
        K = q.shape[-1]
        N_ROWS = q.numel() // K
        BK = triton.next_power_of_2(K)
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        v_out = torch.empty_like(v)
        q_rstd = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)
        k_rstd = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)
        beta_out = torch.empty(N_ROWS, device=q.device, dtype=torch.float32)
        _gka_qkv_l2norm_beta_fwd_rsi_kernel[(N_ROWS,)](
            q, k, v, b_raw, q_out, q_rstd, k_out, k_rstd, v_out, beta_out,
            N_ROWS=N_ROWS, K=K, BK=BK, EPS=eps, num_warps=2, num_stages=1,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, b_raw)
        return q_out, k_out, v_out

    @staticmethod
    def backward(ctx, dq_out: torch.Tensor, dk_out: torch.Tensor, dv_out: torch.Tensor):
        q, q_rstd, k, k_rstd, v, b_raw = ctx.saved_tensors
        K = q.shape[-1]
        N_ROWS = q.numel() // K
        BK = triton.next_power_of_2(K)
        dq_in = torch.empty_like(q)
        dk_in = torch.empty_like(k)
        dv_in = torch.empty_like(v)
        db_raw = torch.empty_like(b_raw)
        _gka_qkv_l2norm_beta_bwd_rsi_kernel[(N_ROWS,)](
            q, q_rstd, k, k_rstd, v, b_raw,
            dq_out.contiguous(), dk_out.contiguous(), dv_out.contiguous(),
            dq_in, dk_in, dv_in, db_raw,
            N_ROWS=N_ROWS, K=K, BK=BK, num_warps=2, num_stages=1,
        )
        return dq_in, dk_in, dv_in, db_raw, None


@triton.jit
def _gka_gate_prep_fwd_rsi_kernel(
    alpha_raw_ptr,
    a_raw_ptr,
    A_log_ptr,
    dt_bias_ptr,
    alpha_out_ptr,
    g_out_ptr,
    N_TOKENS,
    H: tl.constexpr,
    BH: tl.constexpr,
):
    i_tok = tl.program_id(0).to(tl.int64)
    if i_tok >= N_TOKENS:
        return
    o_h = tl.arange(0, BH)
    m_h = o_h < H
    off = i_tok * H + o_h

    alpha_raw = tl.load(alpha_raw_ptr + off, mask=m_h, other=0.0).to(tl.float32)
    a_raw = tl.load(a_raw_ptr + off, mask=m_h, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + o_h, mask=m_h, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + o_h, mask=m_h, other=0.0).to(tl.float32)

    alpha = tl.sigmoid(alpha_raw)
    x = a_raw + dt_bias
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = -tl.exp(A_log) * sp

    tl.store(alpha_out_ptr + off, alpha, mask=m_h)
    tl.store(g_out_ptr + off, g, mask=m_h)


@triton.jit
def _gka_gate_prep_bwd_rsi_kernel(
    alpha_raw_ptr,
    a_raw_ptr,
    A_log_ptr,
    dt_bias_ptr,
    dalpha_ptr,
    dg_ptr,
    dalpha_raw_ptr,
    da_raw_ptr,
    dA_log_part_ptr,
    ddt_bias_part_ptr,
    N_TOKENS,
    H: tl.constexpr,
    BH: tl.constexpr,
):
    i_tok = tl.program_id(0).to(tl.int64)
    if i_tok >= N_TOKENS:
        return
    o_h = tl.arange(0, BH)
    m_h = o_h < H
    off = i_tok * H + o_h

    alpha_raw = tl.load(alpha_raw_ptr + off, mask=m_h, other=0.0).to(tl.float32)
    a_raw = tl.load(a_raw_ptr + off, mask=m_h, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + o_h, mask=m_h, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + o_h, mask=m_h, other=0.0).to(tl.float32)
    dalpha = tl.load(dalpha_ptr + off, mask=m_h, other=0.0).to(tl.float32)
    dg = tl.load(dg_ptr + off, mask=m_h, other=0.0).to(tl.float32)

    sig_a = tl.sigmoid(alpha_raw)
    dalpha_raw = dalpha * sig_a * (1.0 - sig_a)

    x = a_raw + dt_bias
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    exp_A = tl.exp(A_log)
    g = -exp_A * sp
    da = dg * (-exp_A * tl.sigmoid(x))
    dA = dg * g

    tl.store(dalpha_raw_ptr + off, dalpha_raw.to(dalpha_raw_ptr.dtype.element_ty), mask=m_h)
    tl.store(da_raw_ptr + off, da.to(da_raw_ptr.dtype.element_ty), mask=m_h)
    tl.store(dA_log_part_ptr + off, dA, mask=m_h)
    tl.store(ddt_bias_part_ptr + off, da, mask=m_h)


class FusedGKAGatePrepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_raw: torch.Tensor, a_raw: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
        alpha_raw = alpha_raw.contiguous()
        a_raw = a_raw.contiguous()
        B, T, H = alpha_raw.shape
        N_TOKENS = B * T
        BH = max(triton.next_power_of_2(H), 16)
        alpha_out = torch.empty(B, T, H, device=alpha_raw.device, dtype=torch.float32)
        g_out = torch.empty(B, T, H, device=alpha_raw.device, dtype=torch.float32)
        _gka_gate_prep_fwd_rsi_kernel[(N_TOKENS,)](
            alpha_raw, a_raw, A_log, dt_bias, alpha_out, g_out,
            N_TOKENS=N_TOKENS, H=H, BH=BH, num_warps=1, num_stages=1,
        )
        ctx.save_for_backward(alpha_raw, a_raw, A_log, dt_bias)
        return alpha_out, g_out

    @staticmethod
    def backward(ctx, dalpha: torch.Tensor, dg: torch.Tensor):
        alpha_raw, a_raw, A_log, dt_bias = ctx.saved_tensors
        B, T, H = alpha_raw.shape
        N_TOKENS = B * T
        BH = max(triton.next_power_of_2(H), 16)
        dalpha_raw = torch.empty_like(alpha_raw)
        da_raw = torch.empty_like(a_raw)
        dA_part = torch.empty(N_TOKENS, H, device=alpha_raw.device, dtype=torch.float32)
        ddt_part = torch.empty(N_TOKENS, H, device=alpha_raw.device, dtype=torch.float32)
        _gka_gate_prep_bwd_rsi_kernel[(N_TOKENS,)](
            alpha_raw, a_raw, A_log, dt_bias, dalpha.contiguous(), dg.contiguous(),
            dalpha_raw, da_raw, dA_part, ddt_part,
            N_TOKENS=N_TOKENS, H=H, BH=BH, num_warps=1, num_stages=1,
        )
        return dalpha_raw, da_raw, dA_part.sum(dim=0).to(A_log.dtype), ddt_part.sum(dim=0).to(dt_bias.dtype)


@triton.jit
def _gka_rmsnorm_fwd_rsi_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    N_ROWS,
    D: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    x = tl.load(x_ptr + i_row * D + o_d, mask=m_d, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + EPS)
    y = (x * rstd) * w
    tl.store(rstd_ptr + i_row, rstd)
    tl.store(y_ptr + i_row * D + o_d, y.to(y_ptr.dtype.element_ty), mask=m_d)


@triton.jit
def _gka_rmsnorm_bwd_rsi_kernel(
    x_ptr,
    w_ptr,
    rstd_ptr,
    dy_ptr,
    dx_ptr,
    dw_part_ptr,
    N_ROWS,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if i_row >= N_ROWS:
        return
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    x = tl.load(x_ptr + i_row * D + o_d, mask=m_d, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + i_row).to(tl.float32)
    dy = tl.load(dy_ptr + i_row * D + o_d, mask=m_d, other=0.0).to(tl.float32)

    x_hat = x * rstd
    dw = dy * x_hat
    dx_hat = dy * w
    dx = (dx_hat - x_hat * (tl.sum(dx_hat * x_hat, axis=0) * (1.0 / D))) * rstd
    tl.store(dx_ptr + i_row * D + o_d, dx.to(dx_ptr.dtype.element_ty), mask=m_d)
    tl.store(dw_part_ptr + i_row * D + o_d, dw, mask=m_d)


class FusedGKARMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        x = x.contiguous()
        w = w.contiguous()
        D = x.shape[-1]
        N_ROWS = x.numel() // D
        BD = triton.next_power_of_2(D)
        y = torch.empty_like(x)
        rstd = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
        _gka_rmsnorm_fwd_rsi_kernel[(N_ROWS,)](
            x, w, y, rstd, N_ROWS=N_ROWS, D=D, BD=BD, EPS=eps, num_warps=4, num_stages=1
        )
        ctx.save_for_backward(x, w, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w, rstd = ctx.saved_tensors
        D = x.shape[-1]
        N_ROWS = x.numel() // D
        BD = triton.next_power_of_2(D)
        dx = torch.empty_like(x)
        dw_part = torch.empty(N_ROWS, D, device=x.device, dtype=torch.float32)
        _gka_rmsnorm_bwd_rsi_kernel[(N_ROWS,)](
            x, w, rstd, dy.contiguous(), dx, dw_part, N_ROWS=N_ROWS, D=D, BD=BD, num_warps=4, num_stages=1
        )
        return dx, dw_part.sum(dim=0).to(w.dtype), None


@triton.jit
def _gka_rmsnorm_gated_fwd_rsi_kernel(
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
    silu_g = g * tl.sigmoid(g)
    y = (x * rstd) * w * silu_g
    tl.store(rstd_ptr + i_row, rstd)
    tl.store(y_ptr + i_row * V + o_v, y.to(y_ptr.dtype.element_ty), mask=m_v)


@triton.jit
def _gka_rmsnorm_gated_bwd_rsi_kernel(
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
    silu_g = g * sig_g
    dsilu_g = sig_g * (1.0 + g * (1.0 - sig_g))
    x_hat = x * rstd
    dy_g = dy * silu_g
    dw = dy_g * x_hat
    dx_hat = dy_g * w
    dx = (dx_hat - x_hat * (tl.sum(dx_hat * x_hat, axis=0) * (1.0 / V))) * rstd
    dg = dy * x_hat * w * dsilu_g

    tl.store(dx_ptr + i_row * V + o_v, dx.to(dx_ptr.dtype.element_ty), mask=m_v)
    tl.store(dg_ptr + i_row * V + o_v, dg.to(dg_ptr.dtype.element_ty), mask=m_v)
    tl.store(dw_part_ptr + i_row * V + o_v, dw, mask=m_v)


class FusedGKARMSNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        x = x.contiguous()
        g = g.contiguous()
        w = w.contiguous()
        V = x.shape[-1]
        N_ROWS = x.numel() // V
        BV = triton.next_power_of_2(V)
        y = torch.empty_like(x)
        rstd = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
        _gka_rmsnorm_gated_fwd_rsi_kernel[(N_ROWS,)](
            x, g, w, y, rstd, N_ROWS=N_ROWS, V=V, BV=BV, EPS=eps, num_warps=2, num_stages=1
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
        _gka_rmsnorm_gated_bwd_rsi_kernel[(N_ROWS,)](
            x, g, w, rstd, dy.contiguous(), dx, dg, dw_part,
            N_ROWS=N_ROWS, V=V, BV=BV, num_warps=2, num_stages=1
        )
        return dx, dg, dw_part.sum(dim=0).to(w.dtype), None


class GatedKalmaNetBlockRSI(nn.Module):
    """
    5-Stage Progressive Gated KalmaNet Block (`Stages A0 -> A1 -> A2 -> A3 -> A4`):
    - Stage A0: Upstream `chunk_gka` (`24/44` spilling specs, `52,528` spills) + 7 separate `nn.Linear` projections + eager frontend/epilogue.
    - Stage A1: Replace upstream `chunk_gka` with zero-spill `chunk_gka_rsi` (`0` spills, `max_regs <= 168`).
    - Stage A2: + Fused `_gka_qkv_l2norm_beta_fwd_rsi_kernel` (`q_hat = l2_norm(q), k_tilde = beta * l2_norm(k), v_tilde = beta * v`).
    - Stage A3: + Single packed `qkvbaag_proj` GEMM + fused `_gka_gate_prep_fwd_rsi_kernel` (`alpha, g`).
    - Stage A4: + Fused `_gka_rmsnorm_fwd_rsi_kernel` pre-norm + fused `_gka_rmsnorm_gated_fwd_rsi_kernel` epilogue.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        ridge_ratio: float = 0.02,
        num_iter: int = 16,
        stage: str = "A4",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_dim = num_heads * head_dim
        self.value_dim = num_heads * head_dim
        self.ridge_ratio = ridge_ratio
        self.num_iter = num_iter
        self.scale = head_dim ** -0.5
        self.stage = stage

        self.in_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.alpha_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.A_log = nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = nn.Parameter(torch.full((num_heads,), -2.0))
        self.o_norm_weight = nn.Parameter(torch.ones(head_dim))

        total_out = self.key_dim * 2 + self.value_dim * 2 + num_heads * 3
        self.qkvbaag_proj = nn.Linear(hidden_size, total_out, bias=False)
        self.sync_packed_weights()

    @torch.no_grad()
    def sync_packed_weights(self):
        w_cat = torch.cat(
            [
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                self.g_proj.weight,
                self.b_proj.weight,
                self.alpha_proj.weight,
                self.a_proj.weight,
            ],
            dim=0,
        )
        self.qkvbaag_proj.weight.copy_(w_cat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, K = self.num_heads, self.head_dim

        if self.stage in ("A0", "A1", "A2"):
            x_f = x.float()
            rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            h_in = (x_f * rstd * self.in_norm_weight.float()).to(x.dtype)
        else:
            h_in = FusedGKARMSNormFunction.apply(x, self.in_norm_weight, 1e-6)

        if self.stage in ("A0", "A1", "A2"):
            q_raw = self.q_proj(h_in).view(B, T, H, K)
            k_raw = self.k_proj(h_in).view(B, T, H, K)
            v_raw = self.v_proj(h_in).view(B, T, H, K)
            g_out = self.g_proj(h_in).view(B, T, H, K)
            b_raw = self.b_proj(h_in)
            alpha_raw = self.alpha_proj(h_in)
            a_raw = self.a_proj(h_in)
        else:
            packed = self.qkvbaag_proj(h_in)
            kd = self.key_dim
            q_raw = packed[..., :kd].view(B, T, H, K)
            k_raw = packed[..., kd : 2 * kd].view(B, T, H, K)
            v_raw = packed[..., 2 * kd : 3 * kd].view(B, T, H, K)
            g_out = packed[..., 3 * kd : 4 * kd].view(B, T, H, K)
            b_raw = packed[..., 4 * kd : 4 * kd + H]
            alpha_raw = packed[..., 4 * kd + H : 4 * kd + 2 * H]
            a_raw = packed[..., 4 * kd + 2 * H : 4 * kd + 3 * H]

        if self.stage in ("A0", "A1"):
            qf = q_raw.float()
            kf = k_raw.float()
            q = (qf * torch.rsqrt(qf.pow(2).sum(dim=-1, keepdim=True) + 1e-6)).to(x.dtype)
            k_norm = (kf * torch.rsqrt(kf.pow(2).sum(dim=-1, keepdim=True) + 1e-6)).to(x.dtype)
            beta = torch.sigmoid(b_raw)[..., None] + 1e-6
            k = (beta * k_norm).to(x.dtype)
            v = (beta * v_raw).to(x.dtype)
        else:
            q, k, v = FusedGKAQKVNormBetaFunction.apply(q_raw, k_raw, v_raw, b_raw, 1e-6)

        if self.stage in ("A0", "A1", "A2"):
            alpha = torch.sigmoid(alpha_raw.float())
            g = -self.A_log.float().exp() * F.softplus(a_raw.float() + self.dt_bias.float())
        else:
            alpha, g = FusedGKAGatePrepFunction.apply(alpha_raw, a_raw, self.A_log, self.dt_bias)

        if self.stage == "A0":
            from fla.ops.gka import chunk_gka
            o, _ = chunk_gka(
                q=q, k=k, v=v, g=g, gk=g, alpha=alpha,
                scale=self.scale, ridge_ratio=self.ridge_ratio, num_iter=self.num_iter,
            )
        else:
            o, _ = chunk_gka_rsi(
                q=q, k=k, v=v, g=g, gk=g, alpha=alpha,
                scale=self.scale, ridge_ratio=self.ridge_ratio, num_iter=self.num_iter,
            )

        if self.stage in ("A0", "A1", "A2", "A3"):
            of = o.float()
            o_rstd = torch.rsqrt(of.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            o_normed = (of * o_rstd * self.o_norm_weight.float() * F.silu(g_out.float())).to(x.dtype)
        else:
            o_normed = FusedGKARMSNormGatedFunction.apply(o, g_out, self.o_norm_weight, 1e-6)

        return x + self.o_proj(o_normed.reshape(B, T, D))


# ==============================================================================
# 6. 8-GPU NVSwitch Context Parallelism (`GKA-CP = 8`, Track 4)
# ==============================================================================

class _GKAUlyssesSeqToHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed_local: torch.Tensor, group: dist.ProcessGroup, world_size: int):
        ctx.group = group
        ctx.world_size = world_size
        N_PACK, B, T_local, H, K = packed_local.shape
        H_local = H // world_size
        send_buf = (
            packed_local.view(N_PACK, B, T_local, world_size, H_local, K)
            .permute(3, 0, 1, 2, 4, 5)
            .contiguous()
        )
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        packed_full = (
            recv_buf.permute(1, 2, 0, 3, 4, 5)
            .contiguous()
            .view(N_PACK, B, world_size * T_local, H_local, K)
        )
        return packed_full

    @staticmethod
    def backward(ctx, dpacked_full: torch.Tensor):
        group = ctx.group
        world_size = ctx.world_size
        N_PACK, B, T_total, H_local, K = dpacked_full.shape
        T_local = T_total // world_size
        send_buf = (
            dpacked_full.view(N_PACK, B, world_size, T_local, H_local, K)
            .permute(2, 0, 1, 3, 4, 5)
            .contiguous()
        )
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        dpacked_local = (
            recv_buf.permute(1, 2, 3, 0, 4, 5)
            .contiguous()
            .view(N_PACK, B, T_local, world_size * H_local, K)
        )
        return dpacked_local, None, None


class _GKAUlyssesHeadToSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o_full: torch.Tensor, group: dist.ProcessGroup, world_size: int):
        ctx.group = group
        ctx.world_size = world_size
        B, T_total, H_local, V = o_full.shape
        T_local = T_total // world_size
        send_buf = o_full.view(B, world_size, T_local, H_local, V).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        o_local = recv_buf.permute(1, 2, 0, 3, 4).contiguous().view(B, T_local, world_size * H_local, V)
        return o_local

    @staticmethod
    def backward(ctx, do_local: torch.Tensor):
        group = ctx.group
        world_size = ctx.world_size
        B, T_local, H, V = do_local.shape
        H_local = H // world_size
        send_buf = do_local.view(B, T_local, world_size, H_local, V).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        do_full = recv_buf.permute(1, 0, 2, 3, 4).contiguous().view(B, world_size * T_local, H_local, V)
        return do_full, None, None


def gka_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    alpha_local: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int = 8,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
) -> torch.Tensor:
    B, T_local, H, K = q_local.shape
    ga_pad = torch.zeros_like(q_local)
    ga_pad[..., 0] = g_local.to(q_local.dtype)
    ga_pad[..., 1] = alpha_local.to(q_local.dtype)
    packed_local = torch.stack([q_local, k_local, v_local, ga_pad], dim=0)
    packed_full = _GKAUlyssesSeqToHead.apply(packed_local, group, world_size)
    q_full = packed_full[0]
    k_full = packed_full[1]
    v_full = packed_full[2]
    g_full = packed_full[3, ..., 0].float()
    alpha_full = packed_full[3, ..., 1].float()
    o_full, _ = chunk_gka_rsi(
        q=q_full, k=k_full, v=v_full, g=g_full, gk=g_full, alpha=alpha_full,
        ridge_ratio=ridge_ratio, num_iter=num_iter,
    )
    return _GKAUlyssesHeadToSeq.apply(o_full, group, world_size)


class _GKARingAllGatherSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_local: torch.Tensor, group: dist.ProcessGroup, world_size: int, rank: int):
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        x_contig = x_local.contiguous()
        gather_list = [torch.empty_like(x_contig) for _ in range(world_size)]
        dist.all_gather(gather_list, x_contig, group=group)
        return torch.cat(gather_list, dim=1)

    @staticmethod
    def backward(ctx, dx_full: torch.Tensor):
        group = ctx.group
        world_size = ctx.world_size
        T_local = dx_full.shape[1] // world_size
        dx_chunks = list(dx_full.split(T_local, dim=1))
        dx_chunks = [c.contiguous() for c in dx_chunks]
        dx_local = torch.empty_like(dx_chunks[0])
        dist.reduce_scatter(dx_local, dx_chunks, op=dist.ReduceOp.SUM, group=group)
        return dx_local, None, None, None


def gka_cp8_ring_baseline(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    alpha_local: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int = 8,
    rank: int = 0,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
) -> torch.Tensor:
    from fla.ops.gka import chunk_gka

    T_local = q_local.shape[1]
    q_full = _GKARingAllGatherSeq.apply(q_local, group, world_size, rank)
    k_full = _GKARingAllGatherSeq.apply(k_local, group, world_size, rank)
    v_full = _GKARingAllGatherSeq.apply(v_local, group, world_size, rank)
    g_full = _GKARingAllGatherSeq.apply(g_local, group, world_size, rank)
    alpha_full = _GKARingAllGatherSeq.apply(alpha_local, group, world_size, rank)
    o_full, _ = chunk_gka(
        q=q_full, k=k_full, v=v_full, g=g_full, gk=g_full, alpha=alpha_full,
        ridge_ratio=ridge_ratio, num_iter=num_iter,
    )
    return o_full[:, rank * T_local : (rank + 1) * T_local].contiguous()


# ==============================================================================
# 7. SM80 Cubin Resource Audit (`audit_gka_rsi_cubin_resources`)
# ==============================================================================

def _collect_compiled_triton(kobj):
    out = []
    seen = set()

    def walk(obj):
        if id(obj) in seen:
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


def audit_gka_rsi_cubin_resources() -> dict:
    kernel_map = [
        ("_gka_local_cumsum_fwd_rsi_kernel", _gka_local_cumsum_fwd_rsi_kernel),
        ("_gka_fwd_h_kk_fro_part_rsi_kernel", _gka_fwd_h_kk_fro_part_rsi_kernel),
        ("_gka_fro_full_bf16_rsi_kernel", _gka_fro_full_bf16_rsi_kernel),
        ("_gka_fro_strip_rsi_kernel", _gka_fro_strip_rsi_kernel),
        ("_gka_solve_fwd_bf16_rsi_kernel", _gka_solve_fwd_bf16_rsi_kernel),
        ("_gka_solve_fwd_fp32_rsi_kernel", _gka_solve_fwd_fp32_rsi_kernel),
        ("_gka_fwd_h_kv_rsi_kernel", _gka_fwd_h_kv_rsi_kernel),
        ("_gka_fwd_o_rsi_kernel", _gka_fwd_o_rsi_kernel),
        ("_gka_bwd_dh_kv_rsi_kernel", _gka_bwd_dh_kv_rsi_kernel),
        ("_gka_bwd_dv_gla_rsi_kernel", _gka_bwd_dv_gla_rsi_kernel),
        ("_gka_bwd_dqkg_gla_rsi_kernel", _gka_bwd_dqkg_gla_rsi_kernel),
        ("_gka_bwd_w1_vec_rsi_kernel", _gka_bwd_w1_vec_rsi_kernel),
        ("_gka_bwd_intra_strip_rsi_kernel", _gka_bwd_intra_strip_rsi_kernel),
        ("_gka_bwd_dh_unified_rsi_kernel", _gka_bwd_dh_unified_rsi_kernel),
        ("_gka_bwd_dk_dg_inter_strip_rsi_kernel", _gka_bwd_dk_dg_inter_strip_rsi_kernel),
        ("_gka_bwd_dg_rev_cumsum_rsi_kernel", _gka_bwd_dg_rev_cumsum_rsi_kernel),
        ("_gka_decode_solve_rsi_kernel", _gka_decode_solve_rsi_kernel),
        ("_gka_decode_readout_rsi_kernel", _gka_decode_readout_rsi_kernel),
        ("_gka_qkv_l2norm_beta_fwd_rsi_kernel", _gka_qkv_l2norm_beta_fwd_rsi_kernel),
        ("_gka_qkv_l2norm_beta_bwd_rsi_kernel", _gka_qkv_l2norm_beta_bwd_rsi_kernel),
        ("_gka_gate_prep_fwd_rsi_kernel", _gka_gate_prep_fwd_rsi_kernel),
        ("_gka_gate_prep_bwd_rsi_kernel", _gka_gate_prep_bwd_rsi_kernel),
        ("_gka_rmsnorm_fwd_rsi_kernel", _gka_rmsnorm_fwd_rsi_kernel),
        ("_gka_rmsnorm_bwd_rsi_kernel", _gka_rmsnorm_bwd_rsi_kernel),
        ("_gka_rmsnorm_gated_fwd_rsi_kernel", _gka_rmsnorm_gated_fwd_rsi_kernel),
        ("_gka_rmsnorm_gated_bwd_rsi_kernel", _gka_rmsnorm_gated_bwd_rsi_kernel),
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
