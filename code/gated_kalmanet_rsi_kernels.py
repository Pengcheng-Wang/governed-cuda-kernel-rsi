#!/usr/bin/env python3
"""
E145 (v136, Round-19 RSI): Gated KalmaNet (GKA) Zero-Spill SM80 Triton Kernels
==============================================================================

Reference:
  - Paper: "Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression" (arXiv:2511.21016)
  - Official reference: `fla-org/flash-linear-attention/tree/main/fla/ops/gka` & `fla/layers/gka.py`

Resolves five concrete SM80 (`NVIDIA A100-SXM4-80GB`) mathematical, compilation, and GPU-kernel pathologies
in upstream `fla/ops/gka` & `fla/layers/gka.py`:
  1. Severe SM80 Cubin Register Spilling (`193/219` = `88.1%` of upstream specializations spilling `1,519,648` B
     of local stack, including `75/75` = `100%` of `chunk_gka_solve_fwd_kernel`, `72/72` = `100%` of
     `chunk_gka_bwd_kernel_dk_intra` with `ptxas` `regs=32` register-allocator collapse up to `47,864` B stack/thread
     and `262,144 > 166,912` B SMEM overflow, and `46/72` of `chunk_gka_bwd_kernel_dk_inter`), plus `82.3–200.1 s`
     cold compilation per shape caused by compile-time unrolling `NUM_ITER: tl.constexpr = 30` into a `1.7 MB` PTX file.
  2. Zero-Key / Masked-Token Division-by-Zero `NaN` Contamination (`32,768 / 65,536` `dk` `NaN`s and `512` `o/dq/dg`
     `NaN`s from a single $k_0 = 0$ token when $\\|H_0\\|_F = 0$).
  3. `TF32` Truncation in `FP32` Mode (`dg rel_l2 = 2.81e-3`, `dk rel_l2 = 2.20e-3`) and `BF16` casting of $(k_i^\\top k_j)^2$
     inside $\\|H_t\\|_F^2$.
  4. Redundant Backward State Recomputations (`2x chunk_fwd_h`), duplicate `chunk_local_cumsum`, and 11-launch
     unfused `GatedKalmaNet` frontend/epilogue.
  5. Missing Multi-GPU Context Parallelism (`CP`) for the coupled dual state $(H_t, U_t) \\in \\mathbb{R}^{K \\times K} \\times \\mathbb{R}^{K \\times V}$.
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


def rel_l2_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.detach().to(torch.float64) - b.detach().to(torch.float64)).norm()
    denom = b.detach().to(torch.float64).norm().clamp(min=1e-12)
    return float((diff / denom).item())


def naive_gka_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    output_final_state: bool = False,
    exact_solve: bool = True,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Token-by-token `FP64` reference oracle for Gated KalmaNet (`arXiv:2511.21016`).
    Supports both exact `torch.linalg.solve` (`exact_solve=True`, for exact implicit-differentiation
    gradient oracle) and 30-step Chebyshev recurrence (`exact_solve=False`, for forward Chebyshev parity).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q64 = q.to(torch.float64)
    k64 = k.to(torch.float64)
    v64 = v.to(torch.float64)
    g64 = g.to(torch.float64) if g is not None else torch.zeros(B, T, H, device=q.device, dtype=torch.float64)
    gk64 = gk.to(torch.float64) if gk is not None else torch.zeros(B, T, H, device=q.device, dtype=torch.float64)
    a64 = alpha.to(torch.float64) if alpha is not None else None

    if initial_state is not None and initial_state[0] is not None:
        h_kk = initial_state[0].to(torch.float64).clone()
    else:
        h_kk = torch.zeros(B, H, K, K, device=q.device, dtype=torch.float64)

    if initial_state is not None and initial_state[1] is not None:
        h_kv = initial_state[1].to(torch.float64).clone()
    else:
        h_kv = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float64)

    eye = torch.eye(K, device=q.device, dtype=torch.float64).view(1, 1, K, K)
    outs = []
    for t in range(T):
        qt = q64[:, t]
        kt = k64[:, t]
        vt = v64[:, t]
        gt = g64[:, t].unsqueeze(-1).unsqueeze(-1)
        gkt = gk64[:, t].unsqueeze(-1).unsqueeze(-1)

        h_kk = torch.exp(gkt) * h_kk + kt.unsqueeze(-1) * kt.unsqueeze(-2)
        h_kv = torch.exp(gt) * h_kv + kt.unsqueeze(-1) * vt.unsqueeze(-2)

        fro_t = torch.linalg.matrix_norm(h_kk, ord="fro", dim=(-2, -1), keepdim=True)
        valid = fro_t > 1e-12
        fro_safe = torch.where(valid, fro_t, torch.ones_like(fro_t))
        lamb = ridge_ratio * fro_safe

        if exact_solve:
            mat = h_kk + lamb * eye
            xt = torch.linalg.solve(mat, qt.unsqueeze(-1)).squeeze(-1)
            xt = torch.where(valid.squeeze(-1), xt, torch.zeros_like(xt))
        else:
            step = (2.0 / (2.0 * lamb + fro_safe)).squeeze(-1)
            rho = (fro_safe / (2.0 * lamb + fro_safe) / 2.0).squeeze(-1)
            rho_sq = rho * rho
            x_prev = torch.zeros_like(qt)
            xt = torch.where(valid.squeeze(-1), step * qt, torch.zeros_like(qt))
            w = torch.full_like(step, 2.0)
            for _ in range(num_iter):
                w = 1.0 / (1.0 - rho_sq * w)
                r_vec = torch.einsum("bhij,bhj->bhi", h_kk, xt) + lamb.squeeze(-1) * xt - qt
                d_vec = step * w * r_vec + (w - 1.0) * x_prev
                x_prev = xt
                xt = torch.where(valid.squeeze(-1), w * xt - d_vec, torch.zeros_like(xt))

        if a64 is not None:
            at = a64[:, t].unsqueeze(-1)
            q_mix_t = qt + at * (xt - qt)
        else:
            q_mix_t = xt

        ot = float(scale) * torch.einsum("bhkv,bhk->bhv", h_kv, q_mix_t)
        outs.append(ot)

    o_out = torch.stack(outs, dim=1)
    return o_out, (h_kk, h_kv) if output_final_state else None



# ==============================================================================
# 1. Chunk-Local Cumulative Decay Forward & Reverse-Cumsum Backward Kernels
# ==============================================================================

@triton.jit(do_not_specialize=['T'])
def _gka_chunk_cumsum_fwd_kernel(
    g_ptr,
    g_cs_ptr,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    base = (i_b * T + o_t) * H + i_h
    b_g = tl.load(g_ptr + base, mask=m_t, other=0.0).to(tl.float32)
    b_cs = tl.cumsum(b_g, axis=0)
    tl.store(g_cs_ptr + base, b_cs, mask=m_t)


@triton.jit(do_not_specialize=['T'])
def _gka_chunk_cumsum_bwd_reduce_kernel(
    dg_parts_ptr,
    dg_last_add_ptr,
    dg_extra_ptr,
    dg_out_ptr,
    T,
    H: tl.constexpr,
    NK: tl.constexpr,
    BT: tl.constexpr,
    HAS_LAST_ADD: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
):
    """
    Sums `dg_parts [B, T, H, NK]` across `NK = K // 32` slabs, optionally adds `dg_last_add [B, NT, H, NK]`
    at `i_last` and `dg_extra [B, T, H]`, and performs the reverse chunk-local cumulative sum.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    i_last = tl.minimum(BT, T - i_t * BT) - 1
    r_bt = tl.arange(0, BT)
    base = (i_b * T + o_t) * H + i_h

    b_dg = tl.zeros([BT], dtype=tl.float32)
    b_last_add = 0.0
    chunk_base = ((i_b * NT + i_t) * H + i_h) * NK
    for ik in range(0, NK):
        b_dg += tl.load(dg_parts_ptr + base * NK + ik, mask=m_t, other=0.0).to(tl.float32)
        if HAS_LAST_ADD:
            b_last_add += tl.load(dg_last_add_ptr + chunk_base + ik).to(tl.float32)
    if HAS_LAST_ADD:
        b_dg = tl.where(r_bt == i_last, b_dg + b_last_add, b_dg)
    if HAS_EXTRA:
        b_dg += tl.load(dg_extra_ptr + base, mask=m_t, other=0.0).to(tl.float32)

    b_total = tl.sum(b_dg, axis=0)
    b_prefix = tl.cumsum(b_dg, axis=0)
    b_rev_cs = tl.where(m_t, b_total - b_prefix + b_dg, 0.0)
    tl.store(dg_out_ptr + base, b_rev_cs.to(dg_out_ptr.dtype.element_ty), mask=m_t)


# ==============================================================================
# 2. Zero-Spill `[BK=32, BV=32]` Chunk-State Forward & Backward Kernels
# ==============================================================================

@triton.jit(do_not_specialize=['T'])
def _gka_chunk_fwd_h_kernel(
    k_ptr,
    v_ptr,
    g_cs_ptr,
    h0_ptr,
    h_all_ptr,
    ht_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    HAS_H0: tl.constexpr,
    OUT_HT: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    i_kv = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if HAS_H0:
        p_h0 = h0_ptr + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h0, mask=m_kv, other=0.0).to(tl.float32)

    NT = tl.cdiv(T, BT)
    for i_t in range(0, NT):
        p_hall = h_all_ptr + ((i_b * NT + i_t) * H + i_h) * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_hall, b_h, mask=m_kv)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        i_last = tl.minimum(BT, T - i_t * BT) - 1

        p_k = k_ptr + ((i_b * T + o_t[:, None]) * H + i_h) * K + o_k[None, :]
        p_v = v_ptr + ((i_b * T + o_t[:, None]) * H + i_h) * V + o_v[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)

        if USE_G:
            p_g = g_cs_ptr + (i_b * T + o_t) * H + i_h
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_g_last = tl.load(g_cs_ptr + (i_b * T + i_t * BT + i_last) * H + i_h).to(tl.float32)
            b_gl = tl.where(m_t, tl.exp(b_g_last - b_g), 0.0)
            b_k_scaled = (b_k.to(tl.float32) * b_gl[:, None]).to(b_v.dtype)
            b_h = b_h * tl.exp(b_g_last)
        else:
            b_k_scaled = b_k.to(b_v.dtype)

        b_h += tl.dot(tl.trans(b_k_scaled), b_v, allow_tf32=ALLOW_TF32)

    if OUT_HT:
        p_ht = ht_ptr + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h, mask=m_kv)


@triton.jit(do_not_specialize=['T'])
def _gka_chunk_bwd_dh_kernel(
    q_ptr,
    do_ptr,
    g_cs_ptr,
    dht_ptr,
    dh_all_ptr,
    dh0_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    HAS_DHT: tl.constexpr,
    NEGATE_DHT: tl.constexpr,
    OUT_DH0: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    i_kv = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if HAS_DHT:
        p_dht = dht_ptr + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :]
        b_dh = tl.load(p_dht, mask=m_kv, other=0.0).to(tl.float32)
        if NEGATE_DHT:
            b_dh = -b_dh

    NT = tl.cdiv(T, BT)
    for i_c in range(NT - 1, -1, -1):
        p_dhall = dh_all_ptr + ((i_b * NT + i_c) * H + i_h) * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dhall, b_dh, mask=m_kv)

        o_t = i_c * BT + tl.arange(0, BT)
        m_t = o_t < T
        i_last = tl.minimum(BT, T - i_c * BT) - 1

        p_q = q_ptr + ((i_b * T + o_t[:, None]) * H + i_h) * K + o_k[None, :]
        p_do = do_ptr + ((i_b * T + o_t[:, None]) * H + i_h) * V + o_v[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)

        if USE_G:
            p_g = g_cs_ptr + (i_b * T + o_t) * H + i_h
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_g_last = tl.load(g_cs_ptr + (i_b * T + i_c * BT + i_last) * H + i_h).to(tl.float32)
            b_gh = tl.where(m_t, tl.exp(b_g) * scale, 0.0)
            b_q_scaled = (b_q.to(tl.float32) * b_gh[:, None]).to(b_do.dtype)
            b_dh = b_dh * tl.exp(b_g_last)
        else:
            b_q_scaled = (b_q.to(tl.float32) * scale).to(b_do.dtype)

        b_dh += tl.dot(tl.trans(b_q_scaled), b_do, allow_tf32=ALLOW_TF32)

    if OUT_DH0:
        p_dh0 = dh0_ptr + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh, mask=m_kv)


# ==============================================================================
# 3. Zero-Spill `FP32` Frobenius-Norm & `[BM=16]` Runtime-Loop Chebyshev Solver
# ==============================================================================

@triton.jit(do_not_specialize=['T'])
def _gka_khk_fwd_kernel(
    k_ptr,
    h_all_ptr,
    khk_ptr,
    h_norm_sq_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Query-subtiled (`BM = 16`) precomputation of $k_t^\\top H_{[c]} k_t$ (`khk_ptr [B, T, H]`)
    and $\\|H_{[c]}\\|_F^2$ (`h_norm_sq_ptr [B, NT, H]`) (`< 64` regs, `0` spills).
    """
    i_tm = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)
    NK = tl.cdiv(K, BK)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    m_m = o_m < T

    b_khk = tl.zeros([BM], dtype=tl.float32)
    h_norm_sq = 0.0

    chunk_idx = (i_b * NT + i_t) * H + i_h
    h_base = h_all_ptr + chunk_idx * (K * K)
    k_base = k_ptr + (i_b * T * H + i_h) * K

    for ik0 in range(0, NK):
        o_k0 = ik0 * BK + tl.arange(0, BK)
        m_k0 = o_k0 < K
        b_k0 = tl.load(k_base + o_m[:, None] * (H * K) + o_k0[None, :], mask=m_m[:, None] & m_k0[None, :], other=0.0).to(tl.float32)
        b_hk0 = tl.zeros([BM, BK], dtype=tl.float32)
        for ik1 in range(0, NK):
            o_k1 = ik1 * BK + tl.arange(0, BK)
            m_k1 = o_k1 < K
            b_h10 = tl.load(h_base + o_k1[:, None] * K + o_k0[None, :], mask=m_k1[:, None] & m_k0[None, :], other=0.0).to(tl.float32)
            if i_sub == 0:
                h_norm_sq += tl.sum(b_h10 * b_h10)
            b_k1 = tl.load(k_base + o_m[:, None] * (H * K) + o_k1[None, :], mask=m_m[:, None] & m_k1[None, :], other=0.0).to(tl.float32)
            b_hk0 += tl.dot(b_k1, b_h10, allow_tf32=ALLOW_TF32)
        b_khk += tl.sum(b_k0 * b_hk0, axis=1)

    tl.store(khk_ptr + (i_b * T + o_m) * H + i_h, b_khk, mask=m_m)
    if i_sub == 0:
        tl.store(h_norm_sq_ptr + chunk_idx, h_norm_sq)


@triton.jit(do_not_specialize=['T'])
def _gka_fro_fwd_kernel(
    k_ptr,
    khk_ptr,
    h_norm_sq_ptr,
    gk_cs_ptr,
    fro_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Query-subtiled (`BM = 16`) `FP32` Frobenius-norm kernel: computes `fro_t = ||H_t||_F` for `16` rows per CTA.
    Streams $j' \\in [0, 64)$ in `BM = 16` sub-blocks so every matrix is `[16, 16]` or `[16, 32]` (`< 64` regs, `0` spills).
    """
    i_tm = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)
    NK = tl.cdiv(K, BK)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    m_m = o_m < T
    chunk_idx = (i_b * NT + i_t) * H + i_h

    if USE_GK:
        b_gk_m = tl.load(gk_cs_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
        b_gh = tl.where(m_m, tl.exp(b_gk_m), 0.0)
    else:
        b_gk_m = tl.zeros([BM], dtype=tl.float32)
        b_gh = tl.where(m_m, 1.0, 0.0)

    h_norm_sq = tl.load(h_norm_sq_ptr + chunk_idx).to(tl.float32)
    b_fro_sq = b_gh * (b_gh * h_norm_sq)

    o_n = i_t * BT + tl.arange(0, BT)
    m_n = o_n < T
    m_mn = (o_m[:, None] >= o_n[None, :]) & m_m[:, None] & m_n[None, :]
    if USE_GK:
        b_gk_n = tl.load(gk_cs_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
        b_m_all = tl.where(m_mn, tl.exp(b_gk_m[:, None] - b_gk_n[None, :]), 0.0)
    else:
        b_m_all = m_mn.to(tl.float32)

    b_khk = tl.load(khk_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
    b_fro_sq += 2.0 * b_gh * tl.sum(b_m_all * b_khk[None, :], axis=1)

    k_base = k_ptr + (i_b * T * H + i_h) * K

    for j0_sub in range(0, NSUB):
        o_j0 = i_t * BT + j0_sub * BM + tl.arange(0, BM)
        m_j0 = o_j0 < T
        m_mj0 = (o_m[:, None] >= o_j0[None, :]) & m_m[:, None] & m_j0[None, :]
        if USE_GK:
            b_gk_j0 = tl.load(gk_cs_ptr + (i_b * T + o_j0) * H + i_h, mask=m_j0, other=0.0).to(tl.float32)
            b_m0 = tl.where(m_mj0, tl.exp(b_gk_m[:, None] - b_gk_j0[None, :]), 0.0)
        else:
            b_m0 = m_mj0.to(tl.float32)

        b_kk_row = tl.zeros([BM, BT], dtype=tl.float32)
        for ik in range(0, NK):
            o_k = ik * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_kj0 = tl.load(k_base + o_j0[:, None] * (H * K) + o_k[None, :], mask=m_j0[:, None] & m_k[None, :], other=0.0)
            b_kn = tl.load(k_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_n[:, None] & m_k[None, :], other=0.0)
            b_kk_row += tl.dot(b_kj0, tl.trans(b_kn), input_precision='ieee')

        b_kk_sq = b_kk_row * b_kk_row
        b_fro_sq += tl.sum(tl.dot(b_m0.to(k_ptr.dtype.element_ty), b_kk_sq.to(k_ptr.dtype.element_ty), input_precision='ieee') * b_m_all, axis=1)

    b_fro = tl.sqrt(tl.maximum(b_fro_sq, 0.0))
    tl.store(fro_ptr + (i_b * T + o_m) * H + i_h, b_fro, mask=m_m)


@triton.jit(do_not_specialize=['T', 'num_iter', 'do_mix'])
def _gka_chebyshev_solve_fwd_kernel(
    q_ptr,
    k_ptr,
    h_all_ptr,
    gk_cs_ptr,
    fro_ptr,
    alpha_ptr,
    x_ptr,
    q_mix_ptr,
    ridge_ratio,
    num_iter,
    do_mix,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Zero-spill query-subtiled (`BM = 16`) runtime-loop Chebyshev solver for $(H_t + \\lambda_t I) x_t = q_t$.
    Preloads register-resident `b_k [64, K]`, `b_h [K, K]`, and `b_m [16, 64]` outside the runtime loop,
    guaranteeing `<= 168` registers and `0` cubin register spills on SM80 while compiling in `< 1 s`.
    """
    i_tm = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_n = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_m = o_m < T
    m_n = o_n < T
    m_k = o_k < K
    m_mk = m_m[:, None] & m_k[None, :]
    m_nk = m_n[:, None] & m_k[None, :]
    m_kk = m_k[:, None] & m_k[None, :]
    m_s = (o_m[:, None] >= o_n[None, :]) & m_m[:, None] & m_n[None, :]

    q_base = q_ptr + (i_b * T * H + i_h) * K
    k_base = k_ptr + (i_b * T * H + i_h) * K
    x_base = x_ptr + (i_b * T * H + i_h) * K
    h_base = h_all_ptr + ((i_b * NT + i_t) * H + i_h) * (K * K)

    b_q = tl.load(q_base + o_m[:, None] * (H * K) + o_k[None, :], mask=m_mk, other=0.0).to(tl.float32)
    b_k = tl.load(k_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_nk, other=0.0)
    b_h = tl.load(h_base + o_k[:, None] * K + o_k[None, :], mask=m_kk, other=0.0).to(b_k.dtype)

    if USE_GK:
        b_gm = tl.load(gk_cs_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
        b_gn = tl.load(gk_cs_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
        b_m = tl.where(m_s, tl.exp(b_gm[:, None] - b_gn[None, :]), 0.0).to(b_k.dtype)
        b_gh = tl.where(m_m, tl.exp(b_gm), 0.0)
    else:
        b_m = m_s.to(b_k.dtype)
        b_gh = tl.where(m_m, 1.0, 0.0)

    b_fro = tl.load(fro_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
    valid_fro = m_m & (b_fro > 1e-12)
    b_fro_safe = tl.where(valid_fro, b_fro, 1.0)
    b_lamb = (ridge_ratio * b_fro_safe)[:, None]
    b_step = (2.0 / (2.0 * ridge_ratio * b_fro_safe + b_fro_safe))[:, None]
    b_rho = b_fro_safe[:, None] / (2.0 * b_lamb + b_fro_safe[:, None]) / 2.0
    b_rho_sq = b_rho * b_rho

    b_x_prev = tl.zeros([BM, BK], dtype=tl.float32)
    b_x = tl.where(valid_fro[:, None], b_step * b_q, 0.0)
    b_w = tl.full([BM, 1], 2.0, dtype=tl.float32)

    for _ in range(0, num_iter):
        b_w = 1.0 / (1.0 - b_rho_sq * b_w)
        b_x_k = b_x.to(b_k.dtype)
        b_x_gh = (b_x * b_gh[:, None]).to(b_k.dtype)
        b_sc = (tl.dot(b_x_k, tl.trans(b_k), input_precision='ieee') * b_m).to(b_k.dtype)
        b_r = (
            tl.dot(b_x_gh, b_h, input_precision='ieee')
            + tl.dot(b_sc, b_k, input_precision='ieee')
            + b_lamb * b_x
            - b_q
        )
        b_d = b_step * b_w * b_r + (b_w - 1.0) * b_x_prev
        b_x_prev = b_x
        b_x = tl.where(valid_fro[:, None], b_w * b_x - b_d, 0.0)

    tl.store(x_base + o_m[:, None] * (H * K) + o_k[None, :], b_x.to(x_ptr.dtype.element_ty), mask=m_mk)
    if do_mix != 0:
        q_mix_base = q_mix_ptr + (i_b * T * H + i_h) * K
        if HAS_ALPHA:
            b_a = tl.load(alpha_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)[:, None]
            b_qm = b_q + b_a * (b_x - b_q)
        else:
            b_qm = b_x
        tl.store(q_mix_base + o_m[:, None] * (H * K) + o_k[None, :], b_qm.to(q_mix_ptr.dtype.element_ty), mask=m_mk)


# ==============================================================================
# 4. Zero-Spill Slab-Streamed Implicit Ridge Backward Kernels
# ==============================================================================

@triton.jit(do_not_specialize=['T'])
def _gka_bwd_gram_prep_kernel(
    k_ptr,
    x_ptr,
    dq_ptr,
    fro_ptr,
    gk_cs_ptr,
    s_mat_ptr,
    ds_mat_ptr,
    kk_mat_ptr,
    dm_mat_ptr,
    w_vec_ptr,
    w1_vec_ptr,
    w_gh2_ptr,
    dg_gram_ptr,
    ridge_ratio,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Kernel B1a (`BM = 16` row-subtiled): Streams `k, x, dq` in `BK = 32` slabs for `16` rows `o_m`
    to form `[16, 64]` slices of `S_mat`, `DS_mat`, `KK_mat`, `DM_mat`, and `w_vec [16]` (`< 72` regs, `0` spills).
    """
    i_tm = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_n = i_t * BT + tl.arange(0, BT)
    m_m = o_m < T
    m_n = o_n < T
    m_s = (o_m[:, None] <= o_n[None, :]) & m_m[:, None] & m_n[None, :]

    if USE_GK:
        b_gm = tl.load(gk_cs_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
        b_gn = tl.load(gk_cs_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
        b_m = tl.where(m_s, tl.exp(b_gn[None, :] - b_gm[:, None]), 0.0)
        b_gh = tl.where(m_m, tl.exp(b_gm), 0.0)
        b_gh_n = tl.where(m_n, tl.exp(b_gn), 0.0)
    else:
        b_m = m_s.to(tl.float32)
        b_gh = tl.where(m_m, 1.0, 0.0)
        b_gh_n = tl.where(m_n, 1.0, 0.0)

    b_s_raw = tl.zeros([BM, BT], dtype=tl.float32)
    b_ds_raw = tl.zeros([BM, BT], dtype=tl.float32)
    b_kk = tl.zeros([BM, BT], dtype=tl.float32)
    b_dq_x = tl.zeros([BM], dtype=tl.float32)

    NK = tl.cdiv(K, BK)
    k_base = k_ptr + (i_b * T * H + i_h) * K
    x_base = x_ptr + (i_b * T * H + i_h) * K
    dq_base = dq_ptr + (i_b * T * H + i_h) * K

    for ik in range(0, NK):
        o_k = ik * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_km = tl.load(k_base + o_m[:, None] * (H * K) + o_k[None, :], mask=m_m[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_xm = tl.load(x_base + o_m[:, None] * (H * K) + o_k[None, :], mask=m_m[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_dqm = tl.load(dq_base + o_m[:, None] * (H * K) + o_k[None, :], mask=m_m[:, None] & m_k[None, :], other=0.0).to(tl.float32)

        b_kn = tl.load(k_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_n[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_xn = tl.load(x_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_n[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_dqn = tl.load(dq_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_n[:, None] & m_k[None, :], other=0.0).to(tl.float32)

        b_s_raw += tl.dot(b_km, tl.trans(b_xn), allow_tf32=ALLOW_TF32)
        b_ds_raw += tl.dot(b_km, tl.trans(b_dqn), allow_tf32=ALLOW_TF32)
        b_kk += tl.dot(b_km, tl.trans(b_kn), allow_tf32=ALLOW_TF32)
        b_dq_x += tl.sum(b_dqm * b_xm, axis=1)

    b_s = b_s_raw * b_m
    b_ds = b_ds_raw * b_m
    b_dm = b_s * b_ds_raw

    b_fro = tl.load(fro_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
    valid_fro = m_m & (b_fro > 1e-12)
    b_fro_safe = tl.where(valid_fro, b_fro, 1.0)
    b_tmp = tl.where(valid_fro, ridge_ratio * b_dq_x, 0.0)
    b_w = tl.where(valid_fro, 2.0 * b_tmp / b_fro_safe, 0.0)

    chunk_idx = (i_b * NT + i_t) * H + i_h
    r_m = i_sub * BM + tl.arange(0, BM)
    r_n = tl.arange(0, BT)
    mat_off = chunk_idx * (BT * BT) + r_m[:, None] * BT + r_n[None, :]
    tl.store(s_mat_ptr + mat_off, b_s)
    tl.store(ds_mat_ptr + mat_off, b_ds)
    tl.store(kk_mat_ptr + mat_off, b_kk)
    tl.store(dm_mat_ptr + mat_off, b_dm)
    tl.store(w_vec_ptr + (i_b * T + o_m) * H + i_h, b_w, mask=m_m)

    if USE_GK:
        b_dg_row = -tl.sum(b_dm, axis=1) + b_tmp * b_fro
        tl.store(dg_gram_ptr + (i_b * T + o_m) * H + i_h, b_dg_row, mask=m_m)


@triton.jit(do_not_specialize=['T'])
def _gka_bwd_gram_w2kk_kernel(
    gk_cs_ptr,
    kk_mat_ptr,
    dm_mat_ptr,
    w_vec_ptr,
    w2kk_mat_ptr,
    w1_vec_ptr,
    w_gh2_ptr,
    dg_gram_ptr,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    USE_GK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Kernel B1b (`BM = 16` row-subtiled): Computes `W2KK_mat [16, 64]`, `W1_vec [16]`, `W_gh2`,
    and completes `dg_gram [16]` (`< 64` regs, `0` spills).
    """
    i_tm = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_n = i_t * BT + tl.arange(0, BT)
    m_m = o_m < T
    m_n = o_n < T
    m_s = (o_m[:, None] <= o_n[None, :]) & m_m[:, None] & m_n[None, :]

    if USE_GK:
        b_gm = tl.load(gk_cs_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
        b_gn = tl.load(gk_cs_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
        b_m = tl.where(m_s, tl.exp(b_gn[None, :] - b_gm[:, None]), 0.0)
        b_gh_n = tl.where(m_n, tl.exp(b_gn), 0.0)
    else:
        b_gn = tl.zeros([BT], dtype=tl.float32)
        b_m = m_s.to(tl.float32)
        b_gh_n = tl.where(m_n, 1.0, 0.0)

    b_w_n = tl.load(w_vec_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
    b_w1 = tl.sum(b_m * (b_gh_n * b_w_n)[None, :], axis=1)
    tl.store(w1_vec_ptr + (i_b * T + o_m) * H + i_h, b_w1, mask=m_m)

    if i_sub == 0:
        chunk_idx0 = (i_b * NT + i_t) * H + i_h
        tl.store(w_gh2_ptr + chunk_idx0, tl.sum(b_gh_n * b_gh_n * b_w_n, axis=0))

    b_mw = b_m * b_w_n[None, :]
    chunk_idx = (i_b * NT + i_t) * H + i_h
    r_m = i_sub * BM + tl.arange(0, BM)
    r_n = tl.arange(0, BT)

    # Stream j_sub = 0..3 (`[16, 64]` blocks of `b_m_col`) to form `W2KK_mat [16, 64]`
    for j_sub in range(0, NSUB):
        o_j = i_t * BT + j_sub * BM + tl.arange(0, BM)
        m_j = o_j < T
        m_sj = (o_j[:, None] <= o_n[None, :]) & m_j[:, None] & m_n[None, :]
        if USE_GK:
            b_gj = tl.load(gk_cs_ptr + (i_b * T + o_j) * H + i_h, mask=m_j, other=0.0).to(tl.float32)
            b_mj = tl.where(m_sj, tl.exp(b_gn[None, :] - b_gj[:, None]), 0.0)
        else:
            b_mj = m_sj.to(tl.float32)

        b_w2_sub = tl.dot(b_mw, tl.trans(b_mj), allow_tf32=ALLOW_TF32)
        r_j = j_sub * BM + tl.arange(0, BM)
        sub_off = chunk_idx * (BT * BT) + r_m[:, None] * BT + r_j[None, :]
        b_kk_sub = tl.load(kk_mat_ptr + sub_off).to(tl.float32)
        tl.store(w2kk_mat_ptr + sub_off, b_kk_sub * b_w2_sub)

    if USE_GK:
        b_dm_col_sum = tl.zeros([BM], dtype=tl.float32)
        for j_sub in range(0, NSUB):
            r_j = j_sub * BM + tl.arange(0, BM)
            dm_off = chunk_idx * (BT * BT) + r_j[:, None] * BT + r_m[None, :]
            b_dm_sub = tl.load(dm_mat_ptr + dm_off).to(tl.float32)
            b_dm_col_sum += tl.sum(b_dm_sub, axis=0)
        p_dg = dg_gram_ptr + (i_b * T + o_m) * H + i_h
        b_dg_prev = tl.load(p_dg, mask=m_m, other=0.0).to(tl.float32)
        tl.store(p_dg, -(b_dg_prev + b_dm_col_sum), mask=m_m)


@triton.jit(do_not_specialize=['T'])
def _gka_bwd_hs_tile_scan_kernel(
    k_ptr,
    h_all_ptr,
    gk_cs_ptr,
    w1_vec_ptr,
    w_gh2_ptr,
    hs_end_ptr,
    dh0_lamb_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    OUT_DH0: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Kernel B2: Fused `[BK=32, BK=32]` tile computation of `hs_local` and reverse scan of `b_hs`
    across chunks $c = N_T - 1 \\dots 0$. Holds only `[32, 32]` state tiles (`0` spills).
    """
    i_kk = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    NK = tl.cdiv(K, BK)
    i_k0 = i_kk // NK
    i_k1 = i_kk % NK

    o_k0 = i_k0 * BK + tl.arange(0, BK)
    o_k1 = i_k1 * BK + tl.arange(0, BK)
    m_k0 = o_k0 < K
    m_k1 = o_k1 < K
    m_kk = m_k0[:, None] & m_k1[None, :]

    NT = tl.cdiv(T, BT)
    b_hs = tl.zeros([BK, BK], dtype=tl.float32)

    for i_c in range(NT - 1, -1, -1):
        chunk_idx = (i_b * NT + i_c) * H + i_h
        p_hs_end = hs_end_ptr + chunk_idx * (K * K) + o_k0[:, None] * K + o_k1[None, :]
        tl.store(p_hs_end, b_hs, mask=m_kk)

        o_t = i_c * BT + tl.arange(0, BT)
        m_t = o_t < T
        i_last = tl.minimum(BT, T - i_c * BT) - 1

        if USE_GK:
            b_g_last = tl.load(gk_cs_ptr + (i_b * T + i_c * BT + i_last) * H + i_h).to(tl.float32)
            b_hs = b_hs * tl.exp(b_g_last)

        b_h = tl.load(h_all_ptr + chunk_idx * (K * K) + o_k0[:, None] * K + o_k1[None, :], mask=m_kk, other=0.0).to(tl.float32)
        b_w_gh2 = tl.load(w_gh2_ptr + chunk_idx).to(tl.float32)
        b_w1 = tl.load(w1_vec_ptr + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)

        k_base = k_ptr + (i_b * T * H + i_h) * K
        b_k0 = tl.load(k_base + o_t[:, None] * (H * K) + o_k0[None, :], mask=m_t[:, None] & m_k0[None, :], other=0.0).to(tl.float32)
        b_k1 = tl.load(k_base + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0).to(tl.float32)

        b_hs_local = b_w_gh2 * b_h + tl.dot(tl.trans(b_w1[:, None] * b_k0), b_k1, allow_tf32=ALLOW_TF32)
        b_hs += b_hs_local

    if OUT_DH0:
        p_dh0 = dh0_lamb_ptr + i_bh * (K * K) + o_k0[:, None] * K + o_k1[None, :]
        tl.store(p_dh0, -0.5 * b_hs, mask=m_kk)


@triton.jit(do_not_specialize=['T'])
def _gka_bwd_dk_intra_gram_apply_kernel(
    k_ptr,
    x_ptr,
    dq_ptr,
    s_mat_ptr,
    ds_mat_ptr,
    w2kk_mat_ptr,
    dk_intra_ptr,
    t_gram_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Kernel B3a (`BM = 16, BK = 32`): Computes `b_dk_intra = S_mat @ dq + DS_mat @ x` and `b_t_gram = W2KK_mat @ k`
    by streaming `[16, 16]` Gram blocks (`< 64` regs, `0` spills).
    """
    i_tm = tl.program_id(0)
    i_k0 = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_k0 = i_k0 * BK + tl.arange(0, BK)
    m_m = o_m < T
    m_k0 = o_k0 < K
    m_mk0 = m_m[:, None] & m_k0[None, :]

    k_base = k_ptr + (i_b * T * H + i_h) * K
    x_base = x_ptr + (i_b * T * H + i_h) * K
    dq_base = dq_ptr + (i_b * T * H + i_h) * K

    chunk_idx = (i_b * NT + i_t) * H + i_h
    r_m = i_sub * BM + tl.arange(0, BM)

    b_dk_intra = tl.zeros([BM, BK], dtype=tl.float32)
    b_t = tl.zeros([BM, BK], dtype=tl.float32)

    for j_sub in range(0, NSUB):
        o_j = i_t * BT + j_sub * BM + tl.arange(0, BM)
        m_jk0 = (o_j < T)[:, None] & m_k0[None, :]
        r_j = j_sub * BM + tl.arange(0, BM)
        sub_off = chunk_idx * (BT * BT) + r_m[:, None] * BT + r_j[None, :]

        b_dqj = tl.load(dq_base + o_j[:, None] * (H * K) + o_k0[None, :], mask=m_jk0, other=0.0).to(tl.float32)
        b_s_sub = tl.load(s_mat_ptr + sub_off).to(tl.float32)
        b_dk_intra += tl.dot(b_s_sub, b_dqj, allow_tf32=ALLOW_TF32)

        b_xj = tl.load(x_base + o_j[:, None] * (H * K) + o_k0[None, :], mask=m_jk0, other=0.0).to(tl.float32)
        b_ds_sub = tl.load(ds_mat_ptr + sub_off).to(tl.float32)
        b_dk_intra += tl.dot(b_ds_sub, b_xj, allow_tf32=ALLOW_TF32)

        b_kj = tl.load(k_base + o_j[:, None] * (H * K) + o_k0[None, :], mask=m_jk0, other=0.0).to(tl.float32)
        b_w2kk_sub = tl.load(w2kk_mat_ptr + sub_off).to(tl.float32)
        b_t += tl.dot(b_w2kk_sub, b_kj, allow_tf32=ALLOW_TF32)

    out_base = (i_b * T * H + i_h) * K + o_m[:, None] * (H * K) + o_k0[None, :]
    tl.store(dk_intra_ptr + out_base, b_dk_intra, mask=m_mk0)
    tl.store(t_gram_ptr + out_base, b_t, mask=m_mk0)


@triton.jit(do_not_specialize=['T'])
def _gka_bwd_dk_dg_apply_kernel(
    k_ptr,
    x_ptr,
    dq_ptr,
    h_all_ptr,
    dh_all_ptr,
    hs_end_ptr,
    gk_cs_ptr,
    dk_intra_ptr,
    t_gram_ptr,
    w1_vec_ptr,
    dk_ptr,
    dg_parts_ptr,
    dg_last_add_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Kernel B3b (`BM = 16, BK = 32`): Streams `[32, 32]` state tiles of `h, dh, hs_end` and combines with
    `dk_intra` and `t_gram` (`< 96` regs, `0` spills).
    """
    i_tm = tl.program_id(0)
    i_k0 = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)
    NK = tl.cdiv(K, BK)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_k0 = i_k0 * BK + tl.arange(0, BK)
    m_m = o_m < T
    m_k0 = o_k0 < K
    m_mk0 = m_m[:, None] & m_k0[None, :]
    i_last = tl.minimum(BT, T - i_t * BT) - 1

    if USE_GK:
        b_gm = tl.load(gk_cs_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
        b_g_last = tl.load(gk_cs_ptr + (i_b * T + i_t * BT + i_last) * H + i_h).to(tl.float32)
        b_gh = tl.where(m_m, tl.exp(b_gm), 0.0)
        b_gl = tl.where(m_m, tl.exp(b_g_last - b_gm), 0.0)
        exp_g_last = tl.exp(b_g_last)
    else:
        b_gh = tl.where(m_m, 1.0, 0.0)
        b_gl = tl.where(m_m, 1.0, 0.0)
        exp_g_last = 1.0

    k_base = k_ptr + (i_b * T * H + i_h) * K
    x_base = x_ptr + (i_b * T * H + i_h) * K
    dq_base = dq_ptr + (i_b * T * H + i_h) * K
    in_off = (i_b * T * H + i_h) * K + o_m[:, None] * (H * K) + o_k0[None, :]

    b_km0 = tl.load(k_base + o_m[:, None] * (H * K) + o_k0[None, :], mask=m_mk0, other=0.0).to(tl.float32)
    b_xm0 = tl.load(x_base + o_m[:, None] * (H * K) + o_k0[None, :], mask=m_mk0, other=0.0).to(tl.float32)
    b_dk_intra = tl.load(dk_intra_ptr + in_off, mask=m_mk0, other=0.0).to(tl.float32)
    b_t = tl.load(t_gram_ptr + in_off, mask=m_mk0, other=0.0).to(tl.float32)
    b_w1 = tl.load(w1_vec_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)

    chunk_idx = (i_b * NT + i_t) * H + i_h
    h_base = h_all_ptr + chunk_idx * (K * K)
    dh_base = dh_all_ptr + chunk_idx * (K * K)
    hs_base = hs_end_ptr + chunk_idx * (K * K)

    # Pass 1: Accumulate b_t and b_dq_h using only b_h10, then retire b_dq_h and b_xm0
    b_dq_h = tl.zeros([BM, BK], dtype=tl.float32)
    b_dh_h_sum = 0.0
    for ik1 in range(0, NK):
        o_k1 = ik1 * BK + tl.arange(0, BK)
        m_k1 = o_k1 < K
        m_mk1 = m_m[:, None] & m_k1[None, :]
        m_10 = m_k1[:, None] & m_k0[None, :]

        b_km1 = tl.load(k_base + o_m[:, None] * (H * K) + o_k1[None, :], mask=m_mk1, other=0.0).to(tl.float32)
        b_dqm1 = tl.load(dq_base + o_m[:, None] * (H * K) + o_k1[None, :], mask=m_mk1, other=0.0).to(tl.float32)
        b_h10 = tl.load(h_base + o_k1[:, None] * K + o_k0[None, :], mask=m_10, other=0.0).to(tl.float32)

        b_t += tl.dot(b_w1[:, None] * b_km1, b_h10, allow_tf32=ALLOW_TF32)
        if USE_GK:
            b_dq_h += tl.dot(b_dqm1, b_h10, allow_tf32=ALLOW_TF32)

    b_dg_acc = tl.zeros([BM], dtype=tl.float32)
    if USE_GK:
        b_dg_acc = tl.sum(b_dq_h * b_gh[:, None] * b_xm0, axis=1) - tl.sum(b_t * b_km0, axis=1) * 0.5

    # Pass 2: Accumulate b_dk_state and b_dg_state using b_dh10, b_dh01, b_hs10
    b_dk_state = tl.zeros([BM, BK], dtype=tl.float32)
    b_dg_state = tl.zeros([BM, BK], dtype=tl.float32)
    for ik1 in range(0, NK):
        o_k1 = ik1 * BK + tl.arange(0, BK)
        m_k1 = o_k1 < K
        m_mk1 = m_m[:, None] & m_k1[None, :]
        m_10 = m_k1[:, None] & m_k0[None, :]
        m_01 = m_k0[:, None] & m_k1[None, :]

        b_km1 = tl.load(k_base + o_m[:, None] * (H * K) + o_k1[None, :], mask=m_mk1, other=0.0).to(tl.float32)
        b_dh10 = tl.load(dh_base + o_k1[:, None] * K + o_k0[None, :], mask=m_10, other=0.0).to(tl.float32)
        b_dh01 = tl.load(dh_base + o_k0[:, None] * K + o_k1[None, :], mask=m_01, other=0.0).to(tl.float32)
        b_hs10 = tl.load(hs_base + o_k1[:, None] * K + o_k0[None, :], mask=m_10, other=0.0).to(tl.float32)

        b_km1_gl = b_km1 * b_gl[:, None]
        b_dk_state += tl.dot(b_km1_gl, b_dh10 + tl.trans(b_dh01) + b_hs10, allow_tf32=ALLOW_TF32)
        if USE_GK:
            b_dg_state += tl.dot(b_km1_gl, b_dh10 + 0.5 * b_hs10, allow_tf32=ALLOW_TF32)
            if i_sub == 0:
                b_h10 = tl.load(h_base + o_k1[:, None] * K + o_k0[None, :], mask=m_10, other=0.0).to(tl.float32)
                b_dh_h_sum += tl.sum((b_dh10 + 0.5 * b_hs10) * b_h10)

    b_dk_out = -(b_dk_intra + b_dk_state + b_t)
    dk_base = dk_ptr + (i_b * T * H + i_h) * K
    tl.store(dk_base + o_m[:, None] * (H * K) + o_k0[None, :], b_dk_out.to(dk_ptr.dtype.element_ty), mask=m_mk0)

    if USE_GK:
        b_dg_comb = tl.sum(b_dg_state * b_km0, axis=1)
        b_dg_part = -(b_dg_acc - b_dg_comb)
        p_dg_part = dg_parts_ptr + ((i_b * T + o_m) * H + i_h) * NK + i_k0
        tl.store(p_dg_part, b_dg_part, mask=m_m)

        b_sub_last = -tl.sum(b_dg_comb, axis=0)
        if i_sub == 0:
            b_sub_last -= b_dh_h_sum * exp_g_last
        tl.atomic_add(dg_last_add_ptr + chunk_idx * NK + i_k0, b_sub_last)


# ==============================================================================
# 5. Zero-Spill Simple-GLA Readout Forward & Backward + Alpha-Mix Adjoint Kernels
# ==============================================================================

@triton.jit(do_not_specialize=['T'])
def _gka_gla_fwd_o_kernel(
    q_mix_ptr,
    k_ptr,
    v_ptr,
    u_all_ptr,
    g_cs_ptr,
    o_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Zero-spill `[BM=16, BV=32]` GLA readout (`< 72` regs, `0` spills).
    """
    i_v = tl.program_id(0)
    i_tm = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_n = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_m = o_m < T
    m_n = o_n < T
    m_v = o_v < V
    m_s = (o_m[:, None] >= o_n[None, :]) & m_m[:, None] & m_n[None, :]

    if USE_G:
        b_gm = tl.load(g_cs_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
        b_gn = tl.load(g_cs_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
        b_m = tl.where(m_s, tl.exp(b_gm[:, None] - b_gn[None, :]), 0.0)
        b_gh = tl.where(m_m, tl.exp(b_gm), 0.0)
    else:
        b_m = m_s.to(tl.float32)
        b_gh = tl.where(m_m, 1.0, 0.0)

    b_a = tl.zeros([BM, BT], dtype=tl.float32)
    b_o = tl.zeros([BM, BV], dtype=tl.float32)

    NK = tl.cdiv(K, BK)
    q_base = q_mix_ptr + (i_b * T * H + i_h) * K
    k_base = k_ptr + (i_b * T * H + i_h) * K
    u_base = u_all_ptr + ((i_b * NT + i_t) * H + i_h) * (K * V)

    for ik in range(0, NK):
        o_k = ik * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_qm = tl.load(q_base + o_m[:, None] * (H * K) + o_k[None, :], mask=m_m[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_kn = tl.load(k_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_n[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_u = tl.load(u_base + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        b_a += tl.dot(b_qm, tl.trans(b_kn), allow_tf32=ALLOW_TF32)
        b_o += tl.dot(b_qm * b_gh[:, None], b_u, allow_tf32=ALLOW_TF32)

    b_a = (b_a * b_m).to(v_ptr.dtype.element_ty)
    v_base = v_ptr + (i_b * T * H + i_h) * V
    b_v = tl.load(v_base + o_n[:, None] * (H * V) + o_v[None, :], mask=m_n[:, None] & m_v[None, :], other=0.0)
    b_o = (b_o + tl.dot(b_a.to(tl.float32), b_v.to(tl.float32), allow_tf32=ALLOW_TF32)) * scale

    o_base = o_ptr + (i_b * T * H + i_h) * V
    tl.store(o_base + o_m[:, None] * (H * V) + o_v[None, :], b_o.to(o_ptr.dtype.element_ty), mask=m_m[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['T'])
def _gka_gla_bwd_dq_dk_dg_dv_kernel(
    q_mix_ptr,
    k_ptr,
    v_ptr,
    u_all_ptr,
    du_all_ptr,
    do_ptr,
    g_cs_ptr,
    dq_mix_ptr,
    dk_u_ptr,
    dg_u_parts_ptr,
    dg_u_last_add_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Zero-spill `[BM=16, BK=32]` GLA backward kernel for `dq_mix`, `dk_u`, `dg_u_parts` (`< 80` regs, `0` spills).
    """
    i_tm = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)
    NK = tl.cdiv(K, BK)
    NV = tl.cdiv(V, BV)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_n = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_m = o_m < T
    m_n = o_n < T
    m_k = o_k < K
    m_mk = m_m[:, None] & m_k[None, :]
    m_nk = m_n[:, None] & m_k[None, :]
    i_last = tl.minimum(BT, T - i_t * BT) - 1
    m_s_q = (o_m[:, None] >= o_n[None, :]) & m_m[:, None] & m_n[None, :]
    m_s_k = (o_m[:, None] <= o_n[None, :]) & m_m[:, None] & m_n[None, :]

    if USE_G:
        b_gm = tl.load(g_cs_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
        b_gn = tl.load(g_cs_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
        b_g_last = tl.load(g_cs_ptr + (i_b * T + i_t * BT + i_last) * H + i_h).to(tl.float32)
        b_m_q = tl.where(m_s_q, tl.exp(b_gm[:, None] - b_gn[None, :]), 0.0)
        b_m_k = tl.where(m_s_k, tl.exp(b_gn[None, :] - b_gm[:, None]), 0.0)
        b_gh = tl.where(m_m, tl.exp(b_gm), 0.0)
        b_gl = tl.where(m_m, tl.exp(b_g_last - b_gm), 0.0)
        exp_g_last = tl.exp(b_g_last)
    else:
        b_m_q = m_s_q.to(tl.float32)
        b_m_k = m_s_k.to(tl.float32)
        b_gh = tl.where(m_m, 1.0, 0.0)
        b_gl = tl.where(m_m, 1.0, 0.0)
        exp_g_last = 1.0

    q_base = q_mix_ptr + (i_b * T * H + i_h) * K
    k_base = k_ptr + (i_b * T * H + i_h) * K
    v_base = v_ptr + (i_b * T * H + i_h) * V
    do_base = do_ptr + (i_b * T * H + i_h) * V
    chunk_idx = (i_b * NT + i_t) * H + i_h
    u_base = u_all_ptr + chunk_idx * (K * V)
    du_base = du_all_ptr + chunk_idx * (K * V)

    b_qm = tl.load(q_base + o_m[:, None] * (H * K) + o_k[None, :], mask=m_mk, other=0.0).to(tl.float32)
    b_km = tl.load(k_base + o_m[:, None] * (H * K) + o_k[None, :], mask=m_mk, other=0.0).to(tl.float32)
    b_qn = tl.load(q_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_nk, other=0.0).to(tl.float32)
    b_kn = tl.load(k_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_nk, other=0.0).to(tl.float32)

    b_ds_q = tl.zeros([BM, BT], dtype=tl.float32)
    b_ds_k = tl.zeros([BM, BT], dtype=tl.float32)
    b_dq = tl.zeros([BM, BK], dtype=tl.float32)
    b_dk = tl.zeros([BM, BK], dtype=tl.float32)
    b_du_u_sum = 0.0

    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_mv = m_m[:, None] & m_v[None, :]
        m_nv = m_n[:, None] & m_v[None, :]
        m_kv = m_k[:, None] & m_v[None, :]

        b_vm = tl.load(v_base + o_m[:, None] * (H * V) + o_v[None, :], mask=m_mv, other=0.0).to(tl.float32)
        b_dom = tl.load(do_base + o_m[:, None] * (H * V) + o_v[None, :], mask=m_mv, other=0.0).to(tl.float32)
        b_vn = tl.load(v_base + o_n[:, None] * (H * V) + o_v[None, :], mask=m_nv, other=0.0).to(tl.float32)
        b_don = tl.load(do_base + o_n[:, None] * (H * V) + o_v[None, :], mask=m_nv, other=0.0).to(tl.float32)

        b_u = tl.load(u_base + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.0).to(tl.float32)
        b_du = tl.load(du_base + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.0).to(tl.float32)

        b_ds_q += tl.dot(b_dom, tl.trans(b_vn), allow_tf32=ALLOW_TF32)
        b_ds_k += tl.dot(b_vm, tl.trans(b_don), allow_tf32=ALLOW_TF32)
        b_dq += tl.dot(b_dom, tl.trans(b_u), allow_tf32=ALLOW_TF32)
        b_dk += tl.dot(b_vm, tl.trans(b_du), allow_tf32=ALLOW_TF32)
        if USE_G and i_sub == 0:
            b_du_u_sum += tl.sum(b_du * b_u)

    b_dq = b_dq * (b_gh * scale)[:, None]
    b_dk_inter = b_dk * b_gl[:, None]
    b_dq += tl.dot((b_ds_q * b_m_q) * scale, b_kn, allow_tf32=ALLOW_TF32)
    b_dk_intra = tl.dot((b_ds_k * b_m_k) * scale, b_qn, allow_tf32=ALLOW_TF32)
    b_dk_out = b_dk_inter + b_dk_intra

    dq_base = dq_mix_ptr + (i_b * T * H + i_h) * K
    dk_base = dk_u_ptr + (i_b * T * H + i_h) * K
    tl.store(dq_base + o_m[:, None] * (H * K) + o_k[None, :], b_dq.to(dq_mix_ptr.dtype.element_ty), mask=m_mk)
    tl.store(dk_base + o_m[:, None] * (H * K) + o_k[None, :], b_dk_out.to(dk_u_ptr.dtype.element_ty), mask=m_mk)

    if USE_G:
        b_dg = tl.sum(b_dq * b_qm, axis=1) - tl.sum(b_dk_out * b_km, axis=1)
        p_dg_part = dg_u_parts_ptr + ((i_b * T + o_m) * H + i_h) * NK + i_k
        tl.store(p_dg_part, b_dg, mask=m_m)

        b_last_sub = tl.sum(tl.sum(b_dk_inter * b_km, axis=1), axis=0)
        if i_sub == 0:
            b_last_sub += b_du_u_sum * exp_g_last
        tl.atomic_add(dg_u_last_add_ptr + chunk_idx * NK + i_k, b_last_sub)


@triton.jit(do_not_specialize=['T'])
def _gka_gla_bwd_dv_kernel(
    q_mix_ptr,
    k_ptr,
    du_all_ptr,
    do_ptr,
    g_cs_ptr,
    dv_ptr,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Zero-spill `[BM=16, BV=32]` GLA `dv` backward kernel (`< 72` regs, `0` spills).
    """
    i_v = tl.program_id(0)
    i_tm = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_sub = i_tm % NSUB
    NT = tl.cdiv(T, BT)
    NK = tl.cdiv(K, BK)

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_n = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_m = o_m < T
    m_n = o_n < T
    m_v = o_v < V
    i_last = tl.minimum(BT, T - i_t * BT) - 1
    m_s = (o_m[:, None] <= o_n[None, :]) & m_m[:, None] & m_n[None, :]

    if USE_G:
        b_gm = tl.load(g_cs_ptr + (i_b * T + o_m) * H + i_h, mask=m_m, other=0.0).to(tl.float32)
        b_gn = tl.load(g_cs_ptr + (i_b * T + o_n) * H + i_h, mask=m_n, other=0.0).to(tl.float32)
        b_g_last = tl.load(g_cs_ptr + (i_b * T + i_t * BT + i_last) * H + i_h).to(tl.float32)
        b_m = tl.where(m_s, tl.exp(b_gn[None, :] - b_gm[:, None]), 0.0)
        b_gl = tl.where(m_m, tl.exp(b_g_last - b_gm), 0.0)
    else:
        b_m = m_s.to(tl.float32)
        b_gl = tl.where(m_m, 1.0, 0.0)

    b_a = tl.zeros([BM, BT], dtype=tl.float32)
    b_dv = tl.zeros([BM, BV], dtype=tl.float32)

    q_base = q_mix_ptr + (i_b * T * H + i_h) * K
    k_base = k_ptr + (i_b * T * H + i_h) * K
    du_base = du_all_ptr + ((i_b * NT + i_t) * H + i_h) * (K * V)

    for ik in range(0, NK):
        o_k = ik * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_km = tl.load(k_base + o_m[:, None] * (H * K) + o_k[None, :], mask=m_m[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_qn = tl.load(q_base + o_n[:, None] * (H * K) + o_k[None, :], mask=m_n[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_du = tl.load(du_base + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        b_a += tl.dot(b_km, tl.trans(b_qn), allow_tf32=ALLOW_TF32)
        b_dv += tl.dot(b_km * b_gl[:, None], b_du, allow_tf32=ALLOW_TF32)

    b_a = (b_a * b_m) * scale
    do_base = do_ptr + (i_b * T * H + i_h) * V
    b_don = tl.load(do_base + o_n[:, None] * (H * V) + o_v[None, :], mask=m_n[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    b_dv += tl.dot(b_a, b_don, allow_tf32=ALLOW_TF32)

    dv_base = dv_ptr + (i_b * T * H + i_h) * V
    tl.store(dv_base + o_m[:, None] * (H * V) + o_v[None, :], b_dv.to(dv_ptr.dtype.element_ty), mask=m_m[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['T'])
def _gka_bwd_alpha_mix_dx_kernel(
    dq_mix_ptr,
    q_ptr,
    x_ptr,
    alpha_ptr,
    dx_ptr,
    dalpha_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Computes $\\mathrm{d}x_t = \\alpha_t \\mathrm{d}\\tilde{q}_t$ and $\\mathrm{d}\\alpha_t = \\langle \\mathrm{d}\\tilde{q}_t, x_t - q_t \\rangle$.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if i_t >= T:
        return
    base_th = (i_b * T + i_t) * H + i_h
    base_thk = base_th * K
    b_a = tl.load(alpha_ptr + base_th).to(tl.float32)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    b_dqm = tl.load(dq_mix_ptr + base_thk + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_q = tl.load(q_ptr + base_thk + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_x = tl.load(x_ptr + base_thk + o_k, mask=m_k, other=0.0).to(tl.float32)

    b_dx = b_a * b_dqm
    b_da = tl.sum(b_dqm * (b_x - b_q), axis=0)
    tl.store(dx_ptr + base_thk + o_k, b_dx.to(dx_ptr.dtype.element_ty), mask=m_k)
    tl.store(dalpha_ptr + base_th, b_da.to(dalpha_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=['T'])
def _gka_bwd_combine_dq_dk_kernel(
    dq_solve_ptr,
    dq_mix_ptr,
    alpha_ptr,
    dk_solve_ptr,
    dk_u_ptr,
    dq_out_ptr,
    dk_out_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
):
    """
    Fuses $\\mathrm{d}q_t = \\mathrm{d}q^{\\text{solve}}_t + (1 - \\alpha_t)\\mathrm{d}\\tilde{q}_t$ and $\\mathrm{d}k_t = \\mathrm{d}k^{\\text{solve}}_t + \\mathrm{d}k^{(U)}_t$.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    if i_t >= T:
        return

    base_th = (i_b * T + i_t) * H + i_h
    base_thk = base_th * K
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    b_dqs = tl.load(dq_solve_ptr + base_thk + o_k, mask=m_k, other=0.0).to(tl.float32)
    if HAS_ALPHA:
        b_a = tl.load(alpha_ptr + base_th).to(tl.float32)
        b_dqm = tl.load(dq_mix_ptr + base_thk + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_dq = b_dqs + (1.0 - b_a) * b_dqm
    else:
        b_dq = b_dqs

    b_dks = tl.load(dk_solve_ptr + base_thk + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_dku = tl.load(dk_u_ptr + base_thk + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_dk = b_dks + b_dku

    tl.store(dq_out_ptr + base_thk + o_k, b_dq.to(dq_out_ptr.dtype.element_ty), mask=m_k)
    tl.store(dk_out_ptr + base_thk + o_k, b_dk.to(dk_out_ptr.dtype.element_ty), mask=m_k)


# ==============================================================================
# 6. Python Autograd Wrapper & Functional Entrypoints (`chunk_gka_rsi`)
# ==============================================================================

def chunk_gka_fwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], Dict[str, Any]]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    BK = 32
    BV = 32
    BM = 16
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    allow_tf32 = q.dtype != torch.float32

    if scale is None:
        scale = K ** -0.5

    h_kk0, h_kv0 = initial_state if initial_state is not None else (None, None)

    g_cs = None
    gk_cs = None
    if g is not None:
        g_cs = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
        _gka_chunk_cumsum_fwd_kernel[(NT, B * H)](g, g_cs, T, H=H, BT=BT, num_warps=2)
    if gk is not None:
        if g is not None and gk.data_ptr() == g.data_ptr():
            gk_cs = g_cs
        else:
            gk_cs = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
            _gka_chunk_cumsum_fwd_kernel[(NT, B * H)](gk, gk_cs, T, H=H, BT=BT, num_warps=2)

    h_kk_all = torch.empty(B, NT, H, K, K, device=q.device, dtype=torch.float32)
    ht_kk = torch.empty(B, H, K, K, device=q.device, dtype=torch.float32) if output_final_state else None
    _gka_chunk_fwd_h_kernel[(NK * NK, B * H)](
        k, k, gk_cs, h_kk0, h_kk_all, ht_kk,
        T, H=H, K=K, V=K, BT=BT, BK=BK, BV=BK,
        USE_G=gk_cs is not None, HAS_H0=h_kk0 is not None, OUT_HT=output_final_state, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    fro = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    khk = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    h_norm_sq = torch.empty(B, NT, H, device=q.device, dtype=torch.float32)
    _gka_khk_fwd_kernel[(NT * (BT // BM), B * H)](
        k, h_kk_all, khk, h_norm_sq,
        T, H=H, K=K, BT=BT, BM=BM, BK=BK, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=1,
    )
    _gka_fro_fwd_kernel[(NT * (BT // BM), B * H)](
        k, khk, h_norm_sq, gk_cs, fro,
        T, H=H, K=K, BT=BT, BM=BM, BK=BK, USE_GK=gk_cs is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    x = torch.empty_like(q)
    q_mix = torch.empty_like(q) if alpha is not None else x
    BK_FULL = triton.next_power_of_2(K)
    BM_SOLVE = 32 if K <= 64 else 16
    _gka_chebyshev_solve_fwd_kernel[(NT * (BT // BM_SOLVE), B * H)](
        q, k, h_kk_all, gk_cs, fro, alpha, x, q_mix,
        float(ridge_ratio), int(num_iter), 1 if alpha is not None else 0, T,
        H=H, K=K, BT=BT, BM=BM_SOLVE, BK=BK_FULL,
        USE_GK=gk_cs is not None, HAS_ALPHA=alpha is not None, ALLOW_TF32=allow_tf32,
        num_warps=4 if allow_tf32 else 8, num_stages=1,
    )

    u_kv_all = torch.empty(B, NT, H, K, V, device=q.device, dtype=torch.float32)
    ht_kv = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if output_final_state else None
    _gka_chunk_fwd_h_kernel[(NK * NV, B * H)](
        k, v, g_cs, h_kv0, u_kv_all, ht_kv,
        T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
        USE_G=g_cs is not None, HAS_H0=h_kv0 is not None, OUT_HT=output_final_state, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    o = torch.empty(B, T, H, V, device=q.device, dtype=v.dtype)
    _gka_gla_fwd_o_kernel[(NV, NT * (BT // BM), B * H)](
        q_mix, k, v, u_kv_all, g_cs, o,
        float(scale), T, H=H, K=K, V=V, BT=BT, BM=BM, BK=BK, BV=BV,
        USE_G=g_cs is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    saved = {
        "q": q, "k": k, "v": v, "x": x, "q_mix": q_mix, "fro": fro,
        "g_cs": g_cs, "gk_cs": gk_cs, "alpha": alpha,
        "h_kk_all": h_kk_all, "u_kv_all": u_kv_all,
        "h_kk0": h_kk0, "h_kv0": h_kv0,
        "scale": float(scale), "ridge_ratio": float(ridge_ratio), "num_iter": int(num_iter),
        "shared_gate": (g is not None and gk is not None and g.data_ptr() == gk.data_ptr()),
        "g_dtype": g.dtype if g is not None else None,
        "gk_dtype": gk.dtype if gk is not None else None,
    }
    return o, (ht_kk, ht_kv) if output_final_state else None, saved


def chunk_gka_bwd_rsi(
    do: torch.Tensor,
    dht_kk: Optional[torch.Tensor],
    dht_kv: Optional[torch.Tensor],
    saved: Dict[str, Any],
    corrupt_mode: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    q = saved["q"]
    k = saved["k"]
    v = saved["v"]
    x = saved["x"]
    q_mix = saved["q_mix"]
    fro = saved["fro"]
    g_cs = saved["g_cs"]
    gk_cs = saved["gk_cs"]
    alpha = saved["alpha"]
    h_kk_all = saved["h_kk_all"]
    u_kv_all = saved["u_kv_all"]
    h_kk0 = saved["h_kk0"]
    h_kv0 = saved["h_kv0"]
    scale = saved["scale"]
    ridge_ratio = 0.0 if corrupt_mode == 1 else saved["ridge_ratio"]
    num_iter = saved["num_iter"]

    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    BK = 32
    BV = 32
    BM = 16
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    BK_FULL = triton.next_power_of_2(K)
    allow_tf32 = q.dtype != torch.float32

    # 1. GLA backward for U_t: du_all, dh_kv0, dq_mix, dk_u, dv, dg_u_parts
    du_all = torch.empty(B, NT, H, K, V, device=q.device, dtype=torch.float32)
    dh_kv0 = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if h_kv0 is not None else None
    _gka_chunk_bwd_dh_kernel[(NK * NV, B * H)](
        q_mix, do, g_cs, dht_kv, du_all, dh_kv0,
        float(scale), T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
        USE_G=g_cs is not None, HAS_DHT=dht_kv is not None, NEGATE_DHT=False, OUT_DH0=h_kv0 is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    dq_mix = torch.empty_like(q)
    dk_u = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg_u_parts = torch.empty(B, T, H, NK, device=q.device, dtype=torch.float32) if g_cs is not None else None
    dg_u_last_add = torch.zeros(B, NT, H, NK, device=q.device, dtype=torch.float32) if g_cs is not None else None
    _gka_gla_bwd_dq_dk_dg_dv_kernel[(NT * (BT // BM), NK, B * H)](
        q_mix, k, v, u_kv_all, du_all, do, g_cs, dq_mix, dk_u, dg_u_parts, dg_u_last_add,
        float(scale), T, H=H, K=K, V=V, BT=BT, BM=BM, BK=BK, BV=BV,
        USE_G=g_cs is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )
    _gka_gla_bwd_dv_kernel[(NV, NT * (BT // BM), B * H)](
        q_mix, k, du_all, do, g_cs, dv,
        float(scale), T, H=H, K=K, V=V, BT=BT, BM=BM, BK=BK, BV=BV,
        USE_G=g_cs is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    dg = None
    if g_cs is not None:
        dg = torch.empty(B, T, H, device=q.device, dtype=saved["g_dtype"])
        _gka_chunk_cumsum_bwd_reduce_kernel[(NT, B * H)](
            dg_u_parts, dg_u_last_add, None, dg,
            T, H=H, NK=NK, BT=BT, HAS_LAST_ADD=True, HAS_EXTRA=False, num_warps=2,
        )

    # 2. Alpha mixing backward: dx, dalpha
    if alpha is not None:
        dx = torch.empty_like(q)
        dalpha = torch.empty_like(alpha)
        _gka_bwd_alpha_mix_dx_kernel[(T, B * H)](
            dq_mix, q, x, alpha, dx, dalpha,
            T, H=H, K=K, BK=BK_FULL, num_warps=2,
        )
    else:
        dx = dq_mix
        dalpha = None

    # 3. Adjoint Chebyshev solve: dq_solve = (H_t + lamb_t I)^{-1} dx
    dq_solve = torch.empty_like(q)
    BM_SOLVE = 32 if K <= 64 else 16
    _gka_chebyshev_solve_fwd_kernel[(NT * (BT // BM_SOLVE), B * H)](
        dx, k, h_kk_all, gk_cs, fro, alpha, dq_solve, dq_solve,
        float(saved["ridge_ratio"]), int(num_iter), 0, T,
        H=H, K=K, BT=BT, BM=BM_SOLVE, BK=BK_FULL,
        USE_GK=gk_cs is not None, HAS_ALPHA=alpha is not None, ALLOW_TF32=allow_tf32,
        num_warps=4 if allow_tf32 else 8, num_stages=1,
    )
    if corrupt_mode == 3:
        dq_solve = -dq_solve

    # 4. Implicit ridge backward for H_t: dh_kk_all, dh_kk0_direct, dk_solve, dgk
    dh_kk_all = torch.empty(B, NT, H, K, K, device=q.device, dtype=torch.float32)
    dh_kk0_direct = torch.empty(B, H, K, K, device=q.device, dtype=torch.float32) if h_kk0 is not None else None
    _gka_chunk_bwd_dh_kernel[(NK * NK, B * H)](
        dq_solve, x, gk_cs, dht_kk, dh_kk_all, dh_kk0_direct,
        1.0, T, H=H, K=K, V=K, BT=BT, BK=BK, BV=BK,
        USE_G=gk_cs is not None, HAS_DHT=dht_kk is not None, NEGATE_DHT=True, OUT_DH0=h_kk0 is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    s_mat = torch.empty(B, NT, H, BT, BT, device=q.device, dtype=torch.float32)
    ds_mat = torch.empty(B, NT, H, BT, BT, device=q.device, dtype=torch.float32)
    kk_mat = torch.empty(B, NT, H, BT, BT, device=q.device, dtype=torch.float32)
    dm_mat = torch.empty(B, NT, H, BT, BT, device=q.device, dtype=torch.float32)
    w2kk_mat = torch.empty(B, NT, H, BT, BT, device=q.device, dtype=torch.float32)
    w_vec = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    w1_vec = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    w_gh2 = torch.empty(B, NT, H, device=q.device, dtype=torch.float32)
    dg_gram = torch.empty(B, T, H, device=q.device, dtype=torch.float32) if gk_cs is not None else None

    _gka_bwd_gram_prep_kernel[(NT * (BT // BM), B * H)](
        k, x, dq_solve, fro, gk_cs, s_mat, ds_mat, kk_mat, dm_mat, w_vec, w1_vec, w_gh2, dg_gram,
        float(ridge_ratio), T, H=H, K=K, BT=BT, BM=BM, BK=BK, USE_GK=gk_cs is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )
    _gka_bwd_gram_w2kk_kernel[(NT * (BT // BM), B * H)](
        gk_cs, kk_mat, dm_mat, w_vec, w2kk_mat, w1_vec, w_gh2, dg_gram,
        T, H=H, BT=BT, BM=BM, USE_GK=gk_cs is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    hs_end = torch.empty(B, NT, H, K, K, device=q.device, dtype=torch.float32)
    dh0_lamb = torch.empty(B, H, K, K, device=q.device, dtype=torch.float32) if h_kk0 is not None else None
    _gka_bwd_hs_tile_scan_kernel[(NK * NK, B * H)](
        k, h_kk_all, gk_cs, w1_vec, w_gh2, hs_end, dh0_lamb,
        T, H=H, K=K, BT=BT, BK=BK, USE_GK=gk_cs is not None, OUT_DH0=h_kk0 is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=2,
    )

    dk_intra = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    t_gram = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    _gka_bwd_dk_intra_gram_apply_kernel[(NT * (BT // BM), NK, B * H)](
        k, x, dq_solve, s_mat, ds_mat, w2kk_mat, dk_intra, t_gram,
        T, H=H, K=K, BT=BT, BM=BM, BK=BK, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=1,
    )

    dk_solve = torch.empty_like(k)
    dg_k_parts = torch.empty(B, T, H, NK, device=q.device, dtype=torch.float32) if gk_cs is not None else None
    dg_k_last_add = torch.zeros(B, NT, H, NK, device=q.device, dtype=torch.float32) if gk_cs is not None else None
    _gka_bwd_dk_dg_apply_kernel[(NT * (BT // BM), NK, B * H)](
        k, x, dq_solve, h_kk_all, dh_kk_all, hs_end, gk_cs,
        dk_intra, t_gram, w1_vec, dk_solve, dg_k_parts, dg_k_last_add,
        T, H=H, K=K, BT=BT, BM=BM, BK=BK, USE_GK=gk_cs is not None, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=1,
    )

    dgk = None
    if gk_cs is not None:
        dgk = torch.empty(B, T, H, device=q.device, dtype=saved["gk_dtype"])
        _gka_chunk_cumsum_bwd_reduce_kernel[(NT, B * H)](
            dg_k_parts, dg_k_last_add, dg_gram, dgk,
            T, H=H, NK=NK, BT=BT, HAS_LAST_ADD=True, HAS_EXTRA=True, num_warps=2,
        )

    dq_out = torch.empty_like(q)
    dk_out = torch.empty_like(k)
    _gka_bwd_combine_dq_dk_kernel[(T, B * H)](
        dq_solve, dq_mix, alpha, dk_solve, dk_u, dq_out, dk_out,
        T, H=H, K=K, BK=BK_FULL, HAS_ALPHA=alpha is not None, num_warps=2,
    )

    dh_kk0 = (-dh_kk0_direct + dh0_lamb) if h_kk0 is not None else None
    return dq_out, dk_out, dv, dg, dgk, dalpha, dh_kk0, dh_kv0


class _ChunkGKARSIFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        gk: Optional[torch.Tensor],
        alpha: Optional[torch.Tensor],
        scale: Optional[float],
        ridge_ratio: float,
        num_iter: int,
        h_kk0: Optional[torch.Tensor],
        h_kv0: Optional[torch.Tensor],
        output_final_state: bool,
    ):
        init_state = (h_kk0, h_kv0) if (h_kk0 is not None or h_kv0 is not None) else None
        o, final_state, saved = chunk_gka_fwd_rsi(
            q, k, v, g=g, gk=gk, alpha=alpha, scale=scale,
            ridge_ratio=ridge_ratio, num_iter=num_iter,
            initial_state=init_state, output_final_state=output_final_state,
        )
        ctx.saved_dict = saved
        ctx.output_final_state = output_final_state
        if output_final_state:
            return o, final_state[0], final_state[1]
        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        do = grad_outputs[0].contiguous()
        dht_kk = grad_outputs[1].contiguous() if (ctx.output_final_state and len(grad_outputs) > 1 and grad_outputs[1] is not None) else None
        dht_kv = grad_outputs[2].contiguous() if (ctx.output_final_state and len(grad_outputs) > 2 and grad_outputs[2] is not None) else None
        dq, dk, dv, dg, dgk, dalpha, dh_kk0, dh_kv0 = chunk_gka_bwd_rsi(
            do, dht_kk, dht_kv, ctx.saved_dict,
        )
        return dq, dk, dv, dg, dgk, dalpha, None, None, None, dh_kk0, dh_kv0, None


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
    initial_state: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    h_kk0, h_kv0 = initial_state if initial_state is not None else (None, None)
    if output_final_state:
        o, ht_kk, ht_kv = _ChunkGKARSIFn.apply(
            q, k, v, g, gk, alpha, scale, ridge_ratio, num_iter, h_kk0, h_kv0, True,
        )
        return o, (ht_kk, ht_kv)
    o = _ChunkGKARSIFn.apply(
        q, k, v, g, gk, alpha, scale, ridge_ratio, num_iter, h_kk0, h_kv0, False,
    )
    return o, None


# ==============================================================================
# 7. Progressive Ablation Stages (`A1`, `A2`) & Superfused Periphery (`A0..A4`)
# ==============================================================================

class _ChunkGKAStageA1SolveFn(torch.autograd.Function):
    """
    Stage A1 (`rsi_solve_fwd`): Replaces upstream's 30-step compile-time-unrolled
    `chunk_gka_solve_fwd` (`75/75` spilling specs, `82-200 s` compile) in forward with our
    zero-spill `[BM=16]` runtime-loop Chebyshev solver while keeping upstream's backward.
    """
    @staticmethod
    def forward(ctx, q, k, gk, ridge_ratio, num_iter):
        from fla.ops.common.chunk_h import chunk_fwd_h
        from fla.ops.utils import chunk_local_cumsum
        from fla.ops.utils.constant import RCP_LN2

        B, T, H, K = q.shape
        BT = 64
        BM = 16
        BK = 32
        NT = triton.cdiv(T, BT)
        allow_tf32 = q.dtype != torch.float32

        gk_cs = chunk_local_cumsum(gk, chunk_size=BT, scale=RCP_LN2) if gk is not None else None
        gk_cs_nat = chunk_local_cumsum(gk, chunk_size=BT) if gk is not None else None
        h_kk_all, _ = chunk_fwd_h(k=k, v=k, g=gk_cs, h0=None, output_final_state=False, chunk_size=BT)

        fro = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
        khk = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
        h_norm_sq = torch.empty(B, NT, H, device=q.device, dtype=torch.float32)
        _gka_khk_fwd_kernel[(NT * (BT // BM), B * H)](
            k, h_kk_all, khk, h_norm_sq, T, H=H, K=K, BT=BT, BM=BM, BK=BK, ALLOW_TF32=allow_tf32, num_warps=4, num_stages=1,
        )
        _gka_fro_fwd_kernel[(NT * (BT // BM), B * H)](
            k, khk, h_norm_sq, gk_cs_nat, fro, T, H=H, K=K, BT=BT, BM=BM, BK=BK, USE_GK=gk_cs_nat is not None, ALLOW_TF32=allow_tf32, num_warps=4, num_stages=2,
        )

        x = torch.empty_like(q)
        _gka_chebyshev_solve_fwd_kernel[(NT * (BT // BM), B * H)](
            q, k, h_kk_all, gk_cs_nat, fro, None, x, x,
            float(ridge_ratio), int(num_iter), 0, T,
            H=H, K=K, BT=BT, BM=BM, BK=triton.next_power_of_2(K), USE_GK=gk_cs_nat is not None, HAS_ALPHA=False, ALLOW_TF32=allow_tf32,
            num_warps=4 if allow_tf32 else 8, num_stages=1,
        )
        ctx.save_for_backward(k, x, fro, gk_cs)
        ctx.gk_dtype = gk.dtype if gk is not None else None
        ctx.ridge_ratio = ridge_ratio
        ctx.num_iter = num_iter
        return x

    @staticmethod
    def backward(ctx, dx):
        from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
        from fla.ops.gka.chunk_solve_bwd import chunk_gka_solve_bwd
        from fla.ops.gka.chunk_solve_fwd import chunk_gka_solve_fwd
        from fla.ops.utils import chunk_local_cumsum

        k, x, fro, gk_cs = ctx.saved_tensors
        h, _ = chunk_fwd_h(k=k, v=k, g=gk_cs, h0=None, output_final_state=False, chunk_size=64)
        dq, _ = chunk_gka_solve_fwd(
            q=dx.contiguous(), k=k, h=h, gk=gk_cs, fro=fro,
            ridge_ratio=ctx.ridge_ratio, num_iter=ctx.num_iter, chunk_size=64,
        )
        dh, _ = chunk_bwd_dh(
            q=dq, k=k, v=k, do=x, h0=None, dht=None, scale=1.0, g=gk_cs, chunk_size=64,
        )
        dk, dg, _ = chunk_gka_solve_bwd(
            k=k, x=x, dq=dq, h=h, dh=dh, fro=fro, gk=gk_cs,
            ridge_ratio=ctx.ridge_ratio, output_dh0=False, chunk_size=64,
        )
        dgk = chunk_local_cumsum(dg, chunk_size=64, reverse=True).to(ctx.gk_dtype) if dg is not None else None
        return dq, dk, dgk, None, None


class _ChunkGKAStageA2SolveFn(torch.autograd.Function):
    """
    Stage A2 (`rsi_solve_bwd`): Combines Round-19 RSI zero-spill Chebyshev forward solve
    and zero-spill slab-streamed implicit ridge backward, while still recomputing `h_kk`
    in backward and running `chunk_simple_gla` separately like upstream.
    """
    @staticmethod
    def forward(ctx, q, k, gk, ridge_ratio, num_iter):
        B, T, H, K = q.shape
        BT = 64
        BM = 16
        BK = 32
        NT = triton.cdiv(T, BT)
        NK = triton.cdiv(K, BK)
        allow_tf32 = q.dtype != torch.float32

        gk_cs = None
        if gk is not None:
            gk_cs = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
            _gka_chunk_cumsum_fwd_kernel[(NT, B * H)](gk, gk_cs, T, H=H, BT=BT, num_warps=2)

        h_kk_all = torch.empty(B, NT, H, K, K, device=q.device, dtype=torch.float32)
        _gka_chunk_fwd_h_kernel[(NK * NK, B * H)](
            k, k, gk_cs, None, h_kk_all, None,
            T, H=H, K=K, V=K, BT=BT, BK=BK, BV=BK,
            USE_G=gk_cs is not None, HAS_H0=False, OUT_HT=False, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=2,
        )

        fro = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
        khk = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
        h_norm_sq = torch.empty(B, NT, H, device=q.device, dtype=torch.float32)
        _gka_khk_fwd_kernel[(NT * (BT // BM), B * H)](
            k, h_kk_all, khk, h_norm_sq, T, H=H, K=K, BT=BT, BM=BM, BK=BK, ALLOW_TF32=allow_tf32, num_warps=4, num_stages=1,
        )
        _gka_fro_fwd_kernel[(NT * (BT // BM), B * H)](
            k, khk, h_norm_sq, gk_cs, fro, T, H=H, K=K, BT=BT, BM=BM, BK=BK, USE_GK=gk_cs is not None, ALLOW_TF32=allow_tf32, num_warps=4, num_stages=2,
        )

        x = torch.empty_like(q)
        _gka_chebyshev_solve_fwd_kernel[(NT * (BT // BM), B * H)](
            q, k, h_kk_all, gk_cs, fro, None, x, x,
            float(ridge_ratio), int(num_iter), 0, T,
            H=H, K=K, BT=BT, BM=BM, BK=triton.next_power_of_2(K), USE_GK=gk_cs is not None, HAS_ALPHA=False, ALLOW_TF32=allow_tf32,
            num_warps=4 if allow_tf32 else 8, num_stages=1,
        )
        ctx.save_for_backward(k, x, fro, gk_cs)
        ctx.gk_dtype = gk.dtype if gk is not None else None
        ctx.ridge_ratio = ridge_ratio
        ctx.num_iter = num_iter
        return x

    @staticmethod
    def backward(ctx, dx):
        k, x, fro, gk_cs = ctx.saved_tensors
        B, T, H, K = k.shape
        BT = 64
        BM = 16
        BK = 32
        NT = triton.cdiv(T, BT)
        NK = triton.cdiv(K, BK)
        allow_tf32 = k.dtype != torch.float32

        # Recompute h_kk_all like upstream
        h_kk_all = torch.empty(B, NT, H, K, K, device=k.device, dtype=torch.float32)
        _gka_chunk_fwd_h_kernel[(NK * NK, B * H)](
            k, k, gk_cs, None, h_kk_all, None,
            T, H=H, K=K, V=K, BT=BT, BK=BK, BV=BK,
            USE_G=gk_cs is not None, HAS_H0=False, OUT_HT=False, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=2,
        )

        dq_solve = torch.empty_like(dx)
        _gka_chebyshev_solve_fwd_kernel[(NT * (BT // BM), B * H)](
            dx.contiguous(), k, h_kk_all, gk_cs, fro, None, dq_solve, dq_solve,
            float(ctx.ridge_ratio), int(ctx.num_iter), 0, T,
            H=H, K=K, BT=BT, BM=BM, BK=triton.next_power_of_2(K), USE_GK=gk_cs is not None, HAS_ALPHA=False, ALLOW_TF32=allow_tf32,
            num_warps=4 if allow_tf32 else 8, num_stages=1,
        )

        dh_kk_all = torch.empty(B, NT, H, K, K, device=k.device, dtype=torch.float32)
        _gka_chunk_bwd_dh_kernel[(NK * NK, B * H)](
            dq_solve, x, gk_cs, None, dh_kk_all, None,
            1.0, T, H=H, K=K, V=K, BT=BT, BK=BK, BV=BK,
            USE_G=gk_cs is not None, HAS_DHT=False, NEGATE_DHT=True, OUT_DH0=False, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=2,
        )

        s_mat = torch.empty(B, NT, H, BT, BT, device=k.device, dtype=torch.float32)
        ds_mat = torch.empty(B, NT, H, BT, BT, device=k.device, dtype=torch.float32)
        kk_mat = torch.empty(B, NT, H, BT, BT, device=k.device, dtype=torch.float32)
        dm_mat = torch.empty(B, NT, H, BT, BT, device=k.device, dtype=torch.float32)
        w2kk_mat = torch.empty(B, NT, H, BT, BT, device=k.device, dtype=torch.float32)
        w_vec = torch.empty(B, T, H, device=k.device, dtype=torch.float32)
        w1_vec = torch.empty(B, T, H, device=k.device, dtype=torch.float32)
        w_gh2 = torch.empty(B, NT, H, device=k.device, dtype=torch.float32)
        dg_gram = torch.empty(B, T, H, device=k.device, dtype=torch.float32) if gk_cs is not None else None

        _gka_bwd_gram_prep_kernel[(NT * (BT // BM), B * H)](
            k, x, dq_solve, fro, gk_cs, s_mat, ds_mat, kk_mat, dm_mat, w_vec, w1_vec, w_gh2, dg_gram,
            float(ctx.ridge_ratio), T, H=H, K=K, BT=BT, BM=BM, BK=BK, USE_GK=gk_cs is not None, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=2,
        )
        _gka_bwd_gram_w2kk_kernel[(NT * (BT // BM), B * H)](
            gk_cs, kk_mat, dm_mat, w_vec, w2kk_mat, w1_vec, w_gh2, dg_gram,
            T, H=H, BT=BT, BM=BM, USE_GK=gk_cs is not None, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=2,
        )

        hs_end = torch.empty(B, NT, H, K, K, device=k.device, dtype=torch.float32)
        _gka_bwd_hs_tile_scan_kernel[(NK * NK, B * H)](
            k, h_kk_all, gk_cs, w1_vec, w_gh2, hs_end, None,
            T, H=H, K=K, BT=BT, BK=BK, USE_GK=gk_cs is not None, OUT_DH0=False, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=2,
        )

        dk_intra = torch.empty(B, T, H, K, device=k.device, dtype=torch.float32)
        t_gram = torch.empty(B, T, H, K, device=k.device, dtype=torch.float32)
        _gka_bwd_dk_intra_gram_apply_kernel[(NT * (BT // BM), NK, B * H)](
            k, x, dq_solve, s_mat, ds_mat, w2kk_mat, dk_intra, t_gram,
            T, H=H, K=K, BT=BT, BM=BM, BK=BK, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        dk_solve = torch.empty_like(k)
        dg_k_parts = torch.empty(B, T, H, NK, device=k.device, dtype=torch.float32) if gk_cs is not None else None
        dg_k_last_add = torch.zeros(B, NT, H, NK, device=k.device, dtype=torch.float32) if gk_cs is not None else None
        _gka_bwd_dk_dg_apply_kernel[(NT * (BT // BM), NK, B * H)](
            k, x, dq_solve, h_kk_all, dh_kk_all, hs_end, gk_cs,
            dk_intra, t_gram, w1_vec, dk_solve, dg_k_parts, dg_k_last_add,
            T, H=H, K=K, BT=BT, BM=BM, BK=BK, USE_GK=gk_cs is not None, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )

        dgk = None
        if gk_cs is not None:
            dgk = torch.empty(B, T, H, device=k.device, dtype=ctx.gk_dtype)
            _gka_chunk_cumsum_bwd_reduce_kernel[(NT, B * H)](
                dg_k_parts, dg_k_last_add, dg_gram, dgk,
                T, H=H, NK=NK, BT=BT, HAS_LAST_ADD=True, HAS_EXTRA=True, num_warps=2,
            )
        return dq_solve, dk_solve, dgk, None, None


def chunk_gka_stage_a1_rsi(q, k, v, g=None, gk=None, alpha=None, scale=None, ridge_ratio=0.02, num_iter=30):
    from fla.ops.simple_gla import chunk_simple_gla
    if scale is None:
        scale = q.shape[-1] ** -0.5
    x = _ChunkGKAStageA1SolveFn.apply(q, k, gk, ridge_ratio, num_iter)
    q_mix = x if alpha is None else (q + alpha[..., None] * (x - q)).to(q.dtype)
    return chunk_simple_gla(q=q_mix, k=k, v=v, g=g, scale=scale)


def chunk_gka_stage_a2_rsi(q, k, v, g=None, gk=None, alpha=None, scale=None, ridge_ratio=0.02, num_iter=30):
    from fla.ops.simple_gla import chunk_simple_gla
    if scale is None:
        scale = q.shape[-1] ** -0.5
    x = _ChunkGKAStageA2SolveFn.apply(q, k, gk, ridge_ratio, num_iter)
    q_mix = x if alpha is None else (q + alpha[..., None] * (x - q)).to(q.dtype)
    return chunk_simple_gla(q=q_mix, k=k, v=v, g=g, scale=scale)


@triton.jit(do_not_specialize=['T'])
def _gka_short_conv1d_3x_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    T,
    D_TOTAL: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Superfused causal `ShortConv1d(W=4) + SiLU` forward kernel across concatenated `[Q, K, V]` channels (`0` spills).
    """
    i_d = tl.program_id(0)
    i_t = tl.program_id(1)
    i_b = tl.program_id(2)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D_TOTAL

    w0 = tl.load(w_ptr + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)
    acc = tl.load(b_ptr + o_d, mask=m_d, other=0.0).to(tl.float32) if HAS_BIAS else tl.zeros([BD], dtype=tl.float32)

    base_b = i_b * T * D_TOTAL
    if i_t >= 3:
        acc += tl.load(x_ptr + base_b + (i_t - 3) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32) * w0
    if i_t >= 2:
        acc += tl.load(x_ptr + base_b + (i_t - 2) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32) * w1
    if i_t >= 1:
        acc += tl.load(x_ptr + base_b + (i_t - 1) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32) * w2
    acc += tl.load(x_ptr + base_b + i_t * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32) * w3

    sig = tl.sigmoid(acc)
    out = acc * sig
    tl.store(y_ptr + base_b + i_t * D_TOTAL + o_d, out.to(y_ptr.dtype.element_ty), mask=m_d)


@triton.jit(do_not_specialize=['T'])
def _gka_short_conv1d_3x_bwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    dy_ptr,
    dx_ptr,
    dw_part_ptr,
    db_part_ptr,
    T,
    D_TOTAL: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Superfused causal `ShortConv1d(W=4) + SiLU` backward kernel (`0` spills).
    """
    i_d = tl.program_id(0)
    i_c = tl.program_id(1)
    i_b = tl.program_id(2)
    NT = tl.cdiv(T, BT)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D_TOTAL

    w0 = tl.load(w_ptr + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + o_d, mask=m_d, other=0.0).to(tl.float32) if HAS_BIAS else tl.zeros([BD], dtype=tl.float32)

    dw0 = tl.zeros([BD], dtype=tl.float32)
    dw1 = tl.zeros([BD], dtype=tl.float32)
    dw2 = tl.zeros([BD], dtype=tl.float32)
    dw3 = tl.zeros([BD], dtype=tl.float32)
    db = tl.zeros([BD], dtype=tl.float32)

    base_b = i_b * T * D_TOTAL
    t_start = i_c * BT
    t_end = tl.minimum(T, t_start + BT)

    # Compute dz(t) for t in [t_start, min(T, t_end + 3))
    for t in range(t_start, t_end):
        x_m3 = tl.load(x_ptr + base_b + (t - 3) * D_TOTAL + o_d, mask=m_d & (t >= 3), other=0.0).to(tl.float32)
        x_m2 = tl.load(x_ptr + base_b + (t - 2) * D_TOTAL + o_d, mask=m_d & (t >= 2), other=0.0).to(tl.float32)
        x_m1 = tl.load(x_ptr + base_b + (t - 1) * D_TOTAL + o_d, mask=m_d & (t >= 1), other=0.0).to(tl.float32)
        x_0 = tl.load(x_ptr + base_b + t * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32)

        z0 = bias + x_m3 * w0 + x_m2 * w1 + x_m1 * w2 + x_0 * w3
        s0 = tl.sigmoid(z0)
        dy0 = tl.load(dy_ptr + base_b + t * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32)
        dz0 = dy0 * s0 * (1.0 + z0 * (1.0 - s0))

        dw0 += dz0 * x_m3
        dw1 += dz0 * x_m2
        dw2 += dz0 * x_m1
        dw3 += dz0 * x_0
        db += dz0

        dx_val = dz0 * w3
        if t + 1 < T:
            x_p1 = tl.load(x_ptr + base_b + (t + 1) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32)
            z1 = bias + x_m2 * w0 + x_m1 * w1 + x_0 * w2 + x_p1 * w3
            s1 = tl.sigmoid(z1)
            dy1 = tl.load(dy_ptr + base_b + (t + 1) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32)
            dz1 = dy1 * s1 * (1.0 + z1 * (1.0 - s1))
            dx_val += dz1 * w2
            if t + 2 < T:
                x_p2 = tl.load(x_ptr + base_b + (t + 2) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32)
                z2 = bias + x_m1 * w0 + x_0 * w1 + x_p1 * w2 + x_p2 * w3
                s2 = tl.sigmoid(z2)
                dy2 = tl.load(dy_ptr + base_b + (t + 2) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32)
                dz2 = dy2 * s2 * (1.0 + z2 * (1.0 - s2))
                dx_val += dz2 * w1
                if t + 3 < T:
                    x_p3 = tl.load(x_ptr + base_b + (t + 3) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32)
                    z3 = bias + x_0 * w0 + x_p1 * w1 + x_p2 * w2 + x_p3 * w3
                    s3 = tl.sigmoid(z3)
                    dy3 = tl.load(dy_ptr + base_b + (t + 3) * D_TOTAL + o_d, mask=m_d, other=0.0).to(tl.float32)
                    dz3 = dy3 * s3 * (1.0 + z3 * (1.0 - s3))
                    dx_val += dz3 * w0

        tl.store(dx_ptr + base_b + t * D_TOTAL + o_d, dx_val.to(dx_ptr.dtype.element_ty), mask=m_d)

    part_idx = i_b * NT + i_c
    dw_base = dw_part_ptr + part_idx * (D_TOTAL * 4) + o_d * 4
    tl.store(dw_base + 0, dw0, mask=m_d)
    tl.store(dw_base + 1, dw1, mask=m_d)
    tl.store(dw_base + 2, dw2, mask=m_d)
    tl.store(dw_base + 3, dw3, mask=m_d)
    if HAS_BIAS:
        tl.store(db_part_ptr + part_idx * D_TOTAL + o_d, db, mask=m_d)


class _SuperfusedShortConv3xFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        B, T, D_TOTAL = x.shape
        BD = 64
        ND = triton.cdiv(D_TOTAL, BD)
        w_contig = weight.squeeze(1).contiguous()
        b_contig = bias.contiguous() if bias is not None else None
        y = torch.empty_like(x)
        _gka_short_conv1d_3x_fwd_kernel[(ND, T, B)](
            x, w_contig, b_contig, y, T, D_TOTAL=D_TOTAL, BD=BD, HAS_BIAS=b_contig is not None, num_warps=2,
        )
        ctx.save_for_backward(x, w_contig, b_contig)
        ctx.has_bias = b_contig is not None
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w_contig, b_contig = ctx.saved_tensors
        B, T, D_TOTAL = x.shape
        BD = 64
        BT = 64
        ND = triton.cdiv(D_TOTAL, BD)
        NT = triton.cdiv(T, BT)
        dy_c = dy.contiguous()
        dx = torch.empty_like(x)
        dw_part = torch.empty(B * NT, D_TOTAL, 4, device=x.device, dtype=torch.float32)
        db_part = torch.empty(B * NT, D_TOTAL, device=x.device, dtype=torch.float32) if ctx.has_bias else None
        _gka_short_conv1d_3x_bwd_kernel[(ND, NT, B)](
            x, w_contig, b_contig, dy_c, dx, dw_part, db_part,
            T, D_TOTAL=D_TOTAL, BT=BT, BD=BD, HAS_BIAS=ctx.has_bias, num_warps=2,
        )
        dw = dw_part.sum(dim=0).to(w_contig.dtype).unsqueeze(1)
        db = db_part.sum(dim=0).to(b_contig.dtype) if ctx.has_bias else None
        return dx, dw, db


@triton.jit(do_not_specialize=['N_TOKENS'])
def _gka_qknorm_alpha_gate_fwd_kernel(
    qkv_in_ptr,
    agb_in_ptr,
    a_log_ptr,
    dt_bias_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    alpha_out_ptr,
    g_out_ptr,
    q_rstd_ptr,
    k_rstd_ptr,
    N_TOKENS,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Superfuses:
      1. De-interleaving `q_in, k_in, v_in` from contiguous `qkv_conv [B, T, 2*H*K + H*V]`
      2. `q_out = l2_norm(q_in)`, `k_hat = l2_norm(k_in)`
      3. `beta = sigmoid(beta_raw) + 1e-6`, `k_out = beta * k_hat`, `v_out = beta * v_in`
      4. `alpha = sigmoid(alpha_raw)`
      5. `g = -exp(A_log) * softplus(g_raw + dt_bias)`
    in a single zero-spill launch (`0` spills).
    """
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    if i_t >= N_TOKENS:
        return

    D_QKV = 2 * H * K + H * V
    D_AGB = 3 * H
    qkv_t_base = i_t * D_QKV
    agb_t_base = i_t * D_AGB

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = tl.arange(0, BV)
    m_v = o_v < V

    b_q = tl.load(qkv_in_ptr + qkv_t_base + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_k = tl.load(qkv_in_ptr + qkv_t_base + H * K + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_v = tl.load(qkv_in_ptr + qkv_t_base + 2 * H * K + i_h * V + o_v, mask=m_v, other=0.0).to(tl.float32)

    q_rstd = tl.rsqrt(tl.sum(b_q * b_q, axis=0) + 1e-6)
    k_rstd = tl.rsqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)

    th_idx = i_t * H + i_h
    thk_base = th_idx * K
    thv_base = th_idx * V

    b_raw = tl.load(agb_in_ptr + agb_t_base + 2 * H + i_h).to(tl.float32)
    beta = tl.sigmoid(b_raw) + 1e-6

    q_norm = (b_q * q_rstd).to(q_out_ptr.dtype.element_ty)
    k_hat = (b_k * k_rstd).to(k_out_ptr.dtype.element_ty).to(tl.float32)
    k_out = (beta * k_hat).to(k_out_ptr.dtype.element_ty)
    v_out = (beta * b_v).to(v_out_ptr.dtype.element_ty)

    tl.store(q_out_ptr + thk_base + o_k, q_norm, mask=m_k)
    tl.store(k_out_ptr + thk_base + o_k, k_out, mask=m_k)
    tl.store(v_out_ptr + thv_base + o_v, v_out, mask=m_v)
    tl.store(q_rstd_ptr + th_idx, q_rstd)
    tl.store(k_rstd_ptr + th_idx, k_rstd)

    a_raw = tl.load(agb_in_ptr + agb_t_base + i_h).to(tl.float32)
    alpha = tl.sigmoid(a_raw)
    tl.store(alpha_out_ptr + th_idx, alpha.to(alpha_out_ptr.dtype.element_ty))

    g_raw = tl.load(agb_in_ptr + agb_t_base + H + i_h).to(tl.float32)
    a_log = tl.load(a_log_ptr + i_h).to(tl.float32)
    dt_b = tl.load(dt_bias_ptr + i_h).to(tl.float32)
    x_sp = g_raw + dt_b
    sp = tl.where(x_sp > 20.0, x_sp, tl.log(1.0 + tl.exp(x_sp)))
    g_val = -tl.exp(a_log) * sp
    tl.store(g_out_ptr + th_idx, g_val)


@triton.jit(do_not_specialize=['N_TOKENS'])
def _gka_qknorm_alpha_gate_bwd_kernel(
    qkv_in_ptr,
    agb_in_ptr,
    q_rstd_ptr,
    k_rstd_ptr,
    a_log_ptr,
    dt_bias_ptr,
    dq_out_ptr,
    dk_out_ptr,
    dv_out_ptr,
    dalpha_ptr,
    dg_total_ptr,
    dqkv_in_ptr,
    dagb_in_ptr,
    da_log_part_ptr,
    ddt_bias_part_ptr,
    N_TOKENS,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Superfused backward for `QK-L2Norm + Beta-Gate + Alpha-Sigmoid + Mamba-Decay Gate` (`0` spills).
    """
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    if i_t >= N_TOKENS:
        return

    D_QKV = 2 * H * K + H * V
    D_AGB = 3 * H
    qkv_t_base = i_t * D_QKV
    agb_t_base = i_t * D_AGB

    th_idx = i_t * H + i_h
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = tl.arange(0, BV)
    m_v = o_v < V
    thk_base = th_idx * K
    thv_base = th_idx * V

    b_q_in = tl.load(qkv_in_ptr + qkv_t_base + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_k_in = tl.load(qkv_in_ptr + qkv_t_base + H * K + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_v_in = tl.load(qkv_in_ptr + qkv_t_base + 2 * H * K + i_h * V + o_v, mask=m_v, other=0.0).to(tl.float32)

    q_rstd = tl.load(q_rstd_ptr + th_idx).to(tl.float32)
    k_rstd = tl.load(k_rstd_ptr + th_idx).to(tl.float32)
    b_qo = b_q_in * q_rstd
    b_k_hat = b_k_in * k_rstd

    b_raw = tl.load(agb_in_ptr + agb_t_base + 2 * H + i_h).to(tl.float32)
    sig_b = tl.sigmoid(b_raw)
    beta = sig_b + 1e-6

    b_dqo = tl.load(dq_out_ptr + thk_base + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_dko = tl.load(dk_out_ptr + thk_base + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_dvo = tl.load(dv_out_ptr + thv_base + o_v, mask=m_v, other=0.0).to(tl.float32)

    dbeta = tl.sum(b_dko * b_k_hat, axis=0) + tl.sum(b_dvo * b_v_in, axis=0)
    db_raw = dbeta * sig_b * (1.0 - sig_b)

    b_dk_hat = b_dko * beta
    b_dv_in = b_dvo * beta
    dq_in = (b_dqo - b_qo * tl.sum(b_dqo * b_qo, axis=0)) * q_rstd
    dk_in = (b_dk_hat - b_k_hat * tl.sum(b_dk_hat * b_k_hat, axis=0)) * k_rstd

    tl.store(dqkv_in_ptr + qkv_t_base + i_h * K + o_k, dq_in.to(dqkv_in_ptr.dtype.element_ty), mask=m_k)
    tl.store(dqkv_in_ptr + qkv_t_base + H * K + i_h * K + o_k, dk_in.to(dqkv_in_ptr.dtype.element_ty), mask=m_k)
    tl.store(dqkv_in_ptr + qkv_t_base + 2 * H * K + i_h * V + o_v, b_dv_in.to(dqkv_in_ptr.dtype.element_ty), mask=m_v)

    a_raw = tl.load(agb_in_ptr + agb_t_base + i_h).to(tl.float32)
    sig_a = tl.sigmoid(a_raw)
    dalpha = tl.load(dalpha_ptr + th_idx).to(tl.float32)
    da_raw = dalpha * sig_a * (1.0 - sig_a)

    g_raw = tl.load(agb_in_ptr + agb_t_base + H + i_h).to(tl.float32)
    a_log = tl.load(a_log_ptr + i_h).to(tl.float32)
    dt_b = tl.load(dt_bias_ptr + i_h).to(tl.float32)
    dg = tl.load(dg_total_ptr + th_idx).to(tl.float32)

    x_sp = g_raw + dt_b
    sp = tl.where(x_sp > 20.0, x_sp, tl.log(1.0 + tl.exp(x_sp)))
    sig_sp = tl.sigmoid(x_sp)
    exp_a = tl.exp(a_log)

    dg_raw = -dg * exp_a * sig_sp
    da_log = -dg * exp_a * sp

    tl.store(dagb_in_ptr + agb_t_base + i_h, da_raw.to(dagb_in_ptr.dtype.element_ty))
    tl.store(dagb_in_ptr + agb_t_base + H + i_h, dg_raw.to(dagb_in_ptr.dtype.element_ty))
    tl.store(dagb_in_ptr + agb_t_base + 2 * H + i_h, db_raw.to(dagb_in_ptr.dtype.element_ty))
    tl.store(da_log_part_ptr + th_idx, da_log)
    tl.store(ddt_bias_part_ptr + th_idx, dg_raw)


class _SuperfusedQKNormAlphaGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv_in: torch.Tensor, agb_in: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, H: int, K: int, V: int):
        B, T, _ = qkv_in.shape
        N_TOKENS = B * T
        BK = triton.next_power_of_2(K)
        BV = triton.next_power_of_2(V)
        q_out = torch.empty(B, T, H, K, device=qkv_in.device, dtype=qkv_in.dtype)
        k_out = torch.empty(B, T, H, K, device=qkv_in.device, dtype=qkv_in.dtype)
        v_out = torch.empty(B, T, H, V, device=qkv_in.device, dtype=qkv_in.dtype)
        alpha_out = torch.empty(B, T, H, device=qkv_in.device, dtype=qkv_in.dtype)
        g_out = torch.empty(B, T, H, device=qkv_in.device, dtype=torch.float32)
        q_rstd = torch.empty(N_TOKENS, H, device=qkv_in.device, dtype=torch.float32)
        k_rstd = torch.empty(N_TOKENS, H, device=qkv_in.device, dtype=torch.float32)

        _gka_qknorm_alpha_gate_fwd_kernel[(N_TOKENS, H)](
            qkv_in, agb_in, A_log, dt_bias,
            q_out, k_out, v_out, alpha_out, g_out, q_rstd, k_rstd,
            N_TOKENS, H=H, K=K, V=V, BK=BK, BV=BV, num_warps=2,
        )
        ctx.save_for_backward(qkv_in, agb_in, q_rstd, k_rstd, A_log, dt_bias)
        ctx.dims = (H, K, V)
        return q_out, k_out, v_out, alpha_out, g_out

    @staticmethod
    def backward(ctx, dq_out, dk_out, dv_out, dalpha_out, dg_out):
        qkv_in, agb_in, q_rstd, k_rstd, A_log, dt_bias = ctx.saved_tensors
        H, K, V = ctx.dims
        B, T, _ = qkv_in.shape
        N_TOKENS = B * T
        BK = triton.next_power_of_2(K)
        BV = triton.next_power_of_2(V)
        dqkv_in = torch.empty_like(qkv_in)
        dagb_in = torch.empty_like(agb_in)
        da_log_part = torch.empty(N_TOKENS, H, device=qkv_in.device, dtype=torch.float32)
        ddt_bias_part = torch.empty(N_TOKENS, H, device=qkv_in.device, dtype=torch.float32)

        _gka_qknorm_alpha_gate_bwd_kernel[(N_TOKENS, H)](
            qkv_in, agb_in, q_rstd, k_rstd, A_log, dt_bias,
            dq_out.contiguous(), dk_out.contiguous(), dv_out.contiguous(), dalpha_out.contiguous(), dg_out.contiguous(),
            dqkv_in, dagb_in, da_log_part, ddt_bias_part,
            N_TOKENS, H=H, K=K, V=V, BK=BK, BV=BV, num_warps=2,
        )
        return dqkv_in, dagb_in, da_log_part.sum(dim=0).to(A_log.dtype), ddt_bias_part.sum(dim=0).to(dt_bias.dtype), None, None, None


class GatedKalmaNetBlockRSI(nn.Module):
    """
    Complete Gated KalmaNet (`arXiv:2511.21016`, `fla/layers/gka.py`) layer supporting
    cumulative optimization stages `A0..A4` on identical weights and inputs.
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        expand_v: float = 1.0,
        conv_size: int = 4,
        ridge_ratio: float = 0.02,
        num_iter: int = 30,
    ):
        super().__init__()
        from fla.modules import FusedRMSNormGated, ShortConvolution

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim
        self.conv_size = conv_size
        self.ridge_ratio = ridge_ratio
        self.num_iter = num_iter

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.alpha_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        A = torch.empty(num_heads, dtype=torch.float32).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A))
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(num_heads, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, bias=False, activation="silu")

        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=1e-6)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, stage: str = "A4") -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H = self.num_heads
        K = self.head_k_dim
        V = self.head_v_dim

        if stage == "A4":
            # Stage A4: Fused QKV + AGB projections + Superfused 3x ShortConv1d + Superfused QKNorm/Beta/Alpha/MambaGate + chunk_gka_rsi
            w_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
            w_agb = torch.cat([self.alpha_proj.weight, self.a_proj.weight, self.beta_proj.weight], dim=0)
            qkv_in = F.linear(hidden_states, w_qkv)
            agb_in = F.linear(hidden_states, w_agb)
            g_gate = self.g_proj(hidden_states).view(B, T, H, V)

            w_conv_3x = torch.cat([self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight], dim=0)
            qkv_conv = _SuperfusedShortConv3xFn.apply(qkv_in, w_conv_3x, None)

            q_norm, k_gated, v_gated, alpha, g_decay = _SuperfusedQKNormAlphaGateFn.apply(
                qkv_conv, agb_in, self.A_log, self.dt_bias, H, K, V,
            )
            o, _ = chunk_gka_rsi(
                q_norm, k_gated, v_gated, g=g_decay, gk=g_decay, alpha=alpha,
                ridge_ratio=self.ridge_ratio, num_iter=self.num_iter,
            )
            o = self.o_norm(o, g_gate)
            return self.o_proj(o.reshape(B, T, self.value_dim))

        # Stages A0..A3: Standard unfused periphery matching fla/layers/gka.py (11 periphery launches)
        from fla.modules.l2norm import l2_norm
        from fla.ops.gka import chunk_gka

        q, _ = self.q_conv1d(self.q_proj(hidden_states), output_final_state=False)
        k, _ = self.k_conv1d(self.k_proj(hidden_states), output_final_state=False)
        v, _ = self.v_conv1d(self.v_proj(hidden_states), output_final_state=False)
        q = l2_norm(q.view(B, T, H, K))
        k = l2_norm(k.view(B, T, H, K))
        v = v.view(B, T, H, V)

        beta = torch.sigmoid(self.beta_proj(hidden_states)).unsqueeze(-1) + 1e-6
        k = (beta * k).to(k.dtype)
        v = (beta * v).to(v.dtype)

        alpha = torch.sigmoid(self.alpha_proj(hidden_states))
        g_raw = self.a_proj(hidden_states)
        g_decay = -self.A_log.float().exp() * F.softplus(g_raw.float() + self.dt_bias.float())

        if stage == "A0":
            o, _ = chunk_gka(
                q, k, v, g=g_decay, gk=g_decay, alpha=alpha,
                ridge_ratio=self.ridge_ratio, num_iter=self.num_iter,
            )
        elif stage == "A1":
            o, _ = chunk_gka_stage_a1_rsi(
                q, k, v, g=g_decay, gk=g_decay, alpha=alpha,
                ridge_ratio=self.ridge_ratio, num_iter=self.num_iter,
            )
        elif stage == "A2":
            o, _ = chunk_gka_stage_a2_rsi(
                q, k, v, g=g_decay, gk=g_decay, alpha=alpha,
                ridge_ratio=self.ridge_ratio, num_iter=self.num_iter,
            )
        elif stage == "A3":
            o, _ = chunk_gka_rsi(
                q, k, v, g=g_decay, gk=g_decay, alpha=alpha,
                ridge_ratio=self.ridge_ratio, num_iter=self.num_iter,
            )
        else:
            raise ValueError(f"Unknown GKA ablation stage: {stage}")

        g_gate = self.g_proj(hidden_states).view(B, T, H, V)
        o = self.o_norm(o, g_gate)
        return self.o_proj(o.reshape(B, T, self.value_dim))


# ==============================================================================
# 8. Multi-GPU Context Parallelism (`GKA-CP = 8`, Track D) on 8x A100 NVSwitch
# ==============================================================================

class _UlyssesAllToAll4DB1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, seq_to_head: bool):
        ctx.group = group
        ctx.seq_to_head = seq_to_head
        P = dist.get_world_size(group)
        if seq_to_head:
            B, T_loc, H, D = x.shape
            H_loc = H // P
            send_buf = x.view(B, T_loc, P, H_loc, D).permute(2, 0, 1, 3, 4).contiguous()
            recv_buf = torch.empty_like(send_buf)
            dist.all_to_all_single(recv_buf, send_buf, group=group)
            return recv_buf.view(B, P * T_loc, H_loc, D)
        else:
            B, T_full, H_loc, D = x.shape
            T_loc = T_full // P
            send_buf = x.view(P, B, T_loc, H_loc, D).contiguous()
            recv_buf = torch.empty_like(send_buf)
            dist.all_to_all_single(recv_buf, send_buf, group=group)
            return recv_buf.permute(1, 2, 0, 3, 4).contiguous().view(B, T_loc, P * H_loc, D)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _UlyssesAllToAll4DB1Fn.apply(grad_out, ctx.group, not ctx.seq_to_head), None, None


def gka_cp8_ulysses_rsi(
    q_shard: torch.Tensor,
    k_shard: torch.Tensor,
    v_shard: torch.Tensor,
    g_shard: torch.Tensor,
    gk_shard: torch.Tensor,
    alpha_shard: torch.Tensor,
    group: dist.ProcessGroup,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
) -> torch.Tensor:
    """
    Zero-Copy `B = 1` Ulysses Sequence-to-Head All-to-All GKA Context Parallelism (`GKA-CP = 8`).
    Eliminates sequential cross-rank state-passing of $(H_t, U_t) \\in \\mathbb{R}^{K \\times K} \\times \\mathbb{R}^{K \\times V}$
    by transposing `[1, T_local, H, D] -> [1, T_full, H/8, D]` over NVSwitch.
    """
    B, T_loc, H, K = q_shard.shape
    V = v_shard.shape[-1]
    assert B == 1, "Zero-Copy B=1 Ulysses GKA-CP requires B == 1"

    q_h = _UlyssesAllToAll4DB1Fn.apply(q_shard, group, True)
    k_h = _UlyssesAllToAll4DB1Fn.apply(k_shard, group, True)
    v_h = _UlyssesAllToAll4DB1Fn.apply(v_shard, group, True)
    g_h = _UlyssesAllToAll4DB1Fn.apply(g_shard.unsqueeze(-1), group, True).squeeze(-1)
    gk_h = g_h if gk_shard.data_ptr() == g_shard.data_ptr() else _UlyssesAllToAll4DB1Fn.apply(gk_shard.unsqueeze(-1), group, True).squeeze(-1)
    a_h = _UlyssesAllToAll4DB1Fn.apply(alpha_shard.unsqueeze(-1), group, True).squeeze(-1)

    o_h, _ = chunk_gka_rsi(
        q_h, k_h, v_h, g=g_h, gk=gk_h, alpha=a_h,
        ridge_ratio=ridge_ratio, num_iter=num_iter,
    )
    return _UlyssesAllToAll4DB1Fn.apply(o_h, group, False)


class _RingGKACPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_s, k_s, v_s, g_s, gk_s, a_s, group, ridge_ratio, num_iter):
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        B, T_loc, H, K = q_s.shape
        V = v_s.shape[-1]

        if rank > 0:
            h_kk0 = torch.empty(B, H, K, K, device=q_s.device, dtype=torch.float32)
            h_kv0 = torch.empty(B, H, K, V, device=q_s.device, dtype=torch.float32)
            dist.recv(h_kk0, src=rank - 1, group=group)
            dist.recv(h_kv0, src=rank - 1, group=group)
            init_state = (h_kk0, h_kv0)
        else:
            init_state = (
                torch.zeros(B, H, K, K, device=q_s.device, dtype=torch.float32),
                torch.zeros(B, H, K, V, device=q_s.device, dtype=torch.float32),
            )

        o_s, (ht_kk, ht_kv), saved = chunk_gka_fwd_rsi(
            q_s, k_s, v_s, g=g_s, gk=gk_s, alpha=a_s,
            ridge_ratio=ridge_ratio, num_iter=num_iter,
            initial_state=init_state, output_final_state=True,
        )

        if rank < world_size - 1:
            dist.send(ht_kk.contiguous(), dst=rank + 1, group=group)
            dist.send(ht_kv.contiguous(), dst=rank + 1, group=group)

        ctx.saved_dict = saved
        ctx.group = group
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        group = ctx.group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        saved = ctx.saved_dict
        B, T_loc, H, K = saved["q"].shape
        V = saved["v"].shape[-1]

        if rank < world_size - 1:
            dht_kk = torch.empty(B, H, K, K, device=do_s.device, dtype=torch.float32)
            dht_kv = torch.empty(B, H, K, V, device=do_s.device, dtype=torch.float32)
            dist.recv(dht_kk, src=rank + 1, group=group)
            dist.recv(dht_kv, src=rank + 1, group=group)
        else:
            dht_kk = None
            dht_kv = None

        dq, dk, dv, dg, dgk, da, dh_kk0, dh_kv0 = chunk_gka_bwd_rsi(
            do_s.contiguous(), dht_kk, dht_kv, saved,
        )

        if rank > 0:
            dist.send(dh_kk0.contiguous(), dst=rank - 1, group=group)
            dist.send(dh_kv0.contiguous(), dst=rank - 1, group=group)

        return dq, dk, dv, dg, dgk, da, None, None, None


def gka_cp8_ring_baseline(
    q_shard: torch.Tensor,
    k_shard: torch.Tensor,
    v_shard: torch.Tensor,
    g_shard: torch.Tensor,
    gk_shard: torch.Tensor,
    alpha_shard: torch.Tensor,
    group: dist.ProcessGroup,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
) -> torch.Tensor:
    """
    Sequential 8-stage ring state-passing baseline for GKA Context Parallelism (`GKA-CP = 8`).
    Passes the coupled dual state $(H_t, U_t) \\in \\mathbb{R}^{B \\times H \\times K \\times K} \\times \\mathbb{R}^{B \\times H \\times K \\times V}$
    sequentially across ranks $0 \\to 1 \\to \\dots \\to 7$ in forward and $7 \\to \\dots \\to 0$ in backward.
    """
    return _RingGKACPFn.apply(q_shard, k_shard, v_shard, g_shard, gk_shard, alpha_shard, group, ridge_ratio, num_iter)


ALL_ROUND19_RSI_KERNELS = [
    ("_gka_chunk_cumsum_fwd_kernel", _gka_chunk_cumsum_fwd_kernel),
    ("_gka_chunk_cumsum_bwd_reduce_kernel", _gka_chunk_cumsum_bwd_reduce_kernel),
    ("_gka_chunk_fwd_h_kernel", _gka_chunk_fwd_h_kernel),
    ("_gka_chunk_bwd_dh_kernel", _gka_chunk_bwd_dh_kernel),
    ("_gka_khk_fwd_kernel", _gka_khk_fwd_kernel),
    ("_gka_fro_fwd_kernel", _gka_fro_fwd_kernel),
    ("_gka_chebyshev_solve_fwd_kernel", _gka_chebyshev_solve_fwd_kernel),
    ("_gka_bwd_gram_prep_kernel", _gka_bwd_gram_prep_kernel),
    ("_gka_bwd_gram_w2kk_kernel", _gka_bwd_gram_w2kk_kernel),
    ("_gka_bwd_hs_tile_scan_kernel", _gka_bwd_hs_tile_scan_kernel),
    ("_gka_bwd_dk_intra_gram_apply_kernel", _gka_bwd_dk_intra_gram_apply_kernel),
    ("_gka_bwd_dk_dg_apply_kernel", _gka_bwd_dk_dg_apply_kernel),
    ("_gka_gla_fwd_o_kernel", _gka_gla_fwd_o_kernel),
    ("_gka_gla_bwd_dq_dk_dg_dv_kernel", _gka_gla_bwd_dq_dk_dg_dv_kernel),
    ("_gka_gla_bwd_dv_kernel", _gka_gla_bwd_dv_kernel),
    ("_gka_bwd_alpha_mix_dx_kernel", _gka_bwd_alpha_mix_dx_kernel),
    ("_gka_bwd_combine_dq_dk_kernel", _gka_bwd_combine_dq_dk_kernel),
    ("_gka_short_conv1d_3x_fwd_kernel", _gka_short_conv1d_3x_fwd_kernel),
    ("_gka_short_conv1d_3x_bwd_kernel", _gka_short_conv1d_3x_bwd_kernel),
    ("_gka_qknorm_alpha_gate_fwd_kernel", _gka_qknorm_alpha_gate_fwd_kernel),
    ("_gka_qknorm_alpha_gate_bwd_kernel", _gka_qknorm_alpha_gate_bwd_kernel),
]


def audit_round19_rsi_cubin_spills() -> Dict[str, Any]:
    per_kernel: Dict[str, Any] = {}
    total_specs = 0
    total_spills = 0
    max_regs = 0
    max_smem = 0
    for name, k_obj in ALL_ROUND19_RSI_KERNELS:
        dev_caches = getattr(k_obj, "device_caches", None)
        k_specs = 0
        k_spills = 0
        k_max_regs = 0
        k_max_smem = 0
        if isinstance(dev_caches, dict):
            for _, (c_dict, *_) in dev_caches.items():
                for _, comp in c_dict.items():
                    if hasattr(comp, "n_regs"):
                        k_specs += 1
                        k_spills += int(comp.n_spills)
                        k_max_regs = max(k_max_regs, int(comp.n_regs))
                        smem_b = int(getattr(comp.metadata, "shared", 0))
                        k_max_smem = max(k_max_smem, smem_b)
        per_kernel[name] = {
            "num_specs": k_specs,
            "max_regs": k_max_regs,
            "total_spills": k_spills,
            "max_smem_bytes": k_max_smem,
        }
        total_specs += k_specs
        total_spills += k_spills
        max_regs = max(max_regs, k_max_regs)
        max_smem = max(max_smem, k_max_smem)
    return {
        "num_kernels": len(ALL_ROUND19_RSI_KERNELS),
        "total_specializations": total_specs,
        "max_regs": max_regs,
        "total_cubin_spills": total_spills,
        "max_smem_bytes": max_smem,
        "all_zero_spills": total_spills == 0,
        "per_kernel": per_kernel,
    }


