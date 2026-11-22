#!/usr/bin/env python3
"""
Round-24 RSI (E150 / v141): Wall Attention: Length Generalization With Diagonal Gates
(Tilde Research - Timor Averbuch & Dhruv Pai, June 2026; fla/ops/wall_attn & fla/layers/wall_attn.py).

Hardware-Grounded Optimizations on SM80 (NVIDIA A100-SXM4-80GB):
1. Fixed Anchor Reference Frame + Softmax Shift-Invariance in Fwd & Bwd:
   - _wall_fwd_rsi_kernel: Fixed query-block anchor R_q = P[i_t * BT] unifies off-diagonal
     and diagonal loops into a single loop with 0 spills and 116-128 registers.
   - _wall_bwd_preprocess_rsi_kernel: Fused row-wise delta = sum_v(o * do) and lse_adj = lse.
   - _wall_bwd_dq_rsi_kernel: Fixed query-block anchor R_q = P[i_t * BT] + Softmax
     Shift-Invariance (c_q cancels in softmax probabilities p_{t,s} and sum_s dS_{t,s} == 0
     identically, eliminating query-side dg_scalar accumulation and dropping register pressure
     to 98-168 registers with 0 spills across K in {64, 128}, BF16 and FP32).
   - _wall_bwd_dk_rsi_kernel: Fixed key-block anchor R_k = P[i_t * BT] pre-rescales k_til
     once before the query loop, holds a single [BT, BK] accumulator b_dk_til, and fuses
     in-place dP_tot = dP_q + dP_k epilogue (98-168 registers, 0 spills).
   - _wall_bwd_dv_rsi_kernel: Fixed key-block anchor R_k = P[i_t * BT] single-accumulator
     value-gradient kernel (94-152 registers, 0 spills).
   - _wall_bwd_reduce_dkv_rsi_kernel: Single-launch GQA reduction for (dk, dv).
2. 3-Part BF16 Residual-Split Tensor Core Dot Product (_dot_wall_rsi):
   - Eliminates upstream TF32 mantissa truncation (2.49e-3 error on dg) on FP32 inputs,
     achieving 3.3e-6..4.0e-5 relative L2 error vs the differentiable FP64 oracle.
3. Fused Single-Launch KV-Cache Builder & Zero-Spill Decode Kernel:
   - _wall_build_kv_cache_rsi_kernel: Fuses GQA broadcast, chunk anchor extraction r_cache,
     and exp2(R_c - P) key rescaling into one Triton kernel (~9.5x faster than upstream).
   - _wall_decode_rsi_kernel: Zero-spill decode kernel (<=128 registers, 0 spills).
4. Layer Periphery Fusion (WallAttentionBlockRSI Stages A0 -> A4):
   - Fused single-GEMM QKVG-Gate projection, fused multi-head QK-RMSNorm + LogSigmoid gate
     Triton kernels, and fused output sigmoid-gate multiplication kernel.
5. 8-GPU NVSwitch Context Parallelism (Wall-CP = 8):
   - wall_cp8_ulysses_rsi (single fused all-to-all sequence-to-head redistribution with
     perfect causal load balance) vs wall_cp8_ring_baseline (sequence AllGather + ReduceScatter).
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

RCP_LN2_FLOAT = 1.4426950216
LN2_FLOAT = 0.6931471805599453


# ==============================================================================
# 1. Exact Differentiable FP64 Oracle (Fixes Upstream Broken naive_wall_attn)
# ==============================================================================

def exact_wall_attn_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_scalar: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    window_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Differentiable FP64 reference oracle for WallAttention.
    Upstream fla.ops.wall_attn.naive.naive_wall_attn calls non-differentiable
    chunk_global_cumsum(g), returning g.grad = None and g_scalar.grad = None.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5

    qf = q.double().detach().requires_grad_(True)
    kf = k.double().detach().requires_grad_(True)
    vf = v.double().detach().requires_grad_(True)
    gf = g.double().detach().requires_grad_(True)
    gsf = g_scalar.double().detach().requires_grad_(True) if g_scalar is not None else None

    inv_ln2 = 1.0 / math.log(2.0)
    P = torch.cumsum(gf, dim=1) * inv_ln2
    c = torch.cumsum(gsf, dim=1) if gsf is not None else None

    idx = torch.arange(T, device=q.device)
    causal = idx[:, None] >= idx[None, :]
    if window_size is not None:
        causal = causal & ((idx[:, None] - idx[None, :]) < window_size)

    outs = []
    for hq in range(HQ):
        h = hq // G
        q_h = qf[:, :, hq, :]
        k_h = kf[:, :, h, :]
        v_h = vf[:, :, h, :]
        P_h = P[:, :, hq, :]
        diff = (P_h[:, :, None, :] - P_h[:, None, :, :]).masked_fill(~causal[None, :, :, None], -float("inf"))
        decay = torch.pow(2.0, diff)
        scores = torch.einsum("btk,bsk,btsk->bts", q_h, k_h, decay) * scale
        if c is not None:
            c_h = c[:, :, hq]
            scores = scores + (c_h[:, :, None] - c_h[:, None, :])
        scores = scores.masked_fill(~causal[None, :, :], -float("inf"))
        attn = torch.softmax(scores, dim=-1)
        out_h = torch.einsum("bts,bsv->btv", attn, v_h)
        outs.append(out_h)

    out = torch.stack(outs, dim=2)
    return out, (qf, kf, vf, gf, gsf)


def rel_l2_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.double() - b.double()).norm() / (b.double().norm() + 1e-12)).item())


# ==============================================================================
# 2. Zero-Spill Fixed-Anchor WallAttention Triton Kernels (Fwd + Bwd)
# ==============================================================================

@triton.jit
def _exp2_f32(x):
    return tl.math.exp2(x)


@triton.jit
def _log2_f32(x):
    return tl.math.log2(x)


@triton.jit
def _dot_wall_rsi(a, b, IS_FP32: tl.constexpr):
    if IS_FP32:
        a_hi = a.to(tl.bfloat16)
        a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
        b_hi = b.to(tl.bfloat16)
        b_lo = (b - b_hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.zeros([a.shape[0], b.shape[1]], dtype=tl.float32)
        for p in range(0, 3):
            lhs = tl.where(p < 2, a_hi, a_lo)
            rhs = tl.where(p == 1, b_lo, b_hi)
            acc = tl.dot(lhs, rhs, acc)
        return acc
    else:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))


@triton.jit
def _wall_fwd_rsi_kernel(
    q, k, v, o, g_cumsum, g_scalar_cumsum, lse_adj,
    scale, T,
    W_BOUND: tl.constexpr,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_WINDOW: tl.constexpr, USE_SCALAR_G: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    RCP_LN2_C: tl.constexpr = 1.4426950216

    q_base = q + (bos * HQ + i_hq) * K
    k_base = k + (bos * H + i_h) * K
    v_base = v + (bos * H + i_h) * V
    g_base = g_cumsum + (bos * HQ + i_hq) * K

    p_q = tl.make_block_ptr(q_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pq = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_R = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (1, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
    b_R = tl.load(p_R, boundary_check=(0, 1)).to(tl.float32)
    b_q_til = b_q * _exp2_f32(tl.minimum(b_pq - b_R, 0.0))
    if not IS_FP32:
        b_q_til = b_q_til.to(tl.bfloat16)
    b_R_t = tl.trans(b_R)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T

    i_start = tl.maximum(((i_t * BT - W_BOUND + 1) // BS) * BS, 0) if USE_WINDOW else 0
    i_end = tl.minimum((i_t + 1) * BT, T)

    for i_s in range(i_start, i_end, BS):
        p_k = tl.make_block_ptr(k_base, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_pk = tl.make_block_ptr(g_base, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (i_s, 0), (BS, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_k_til = b_k * _exp2_f32(tl.minimum(b_R_t - b_pk, 110.0))
        if not IS_FP32:
            b_k_til = b_k_til.to(tl.bfloat16)

        b_s = _dot_wall_rsi(b_q_til, b_k_til, IS_FP32) * (scale * RCP_LN2_C)
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0.0).to(tl.float32)[None, :]
            b_s -= b_ck

        diff_idx = (o_q[:, None] - o_k[None, :]).to(tl.uint32)
        m_s = m_k[None, :] & (diff_idx < W_BOUND)
        b_s = tl.where(m_s, b_s, float("-inf"))

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_mw = tl.where(b_m_new == float("-inf"), 0.0, b_m_new)
        b_r = _exp2_f32(b_m - b_mw)
        b_p = _exp2_f32(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)

        b_v = tl.load(p_v, boundary_check=(0, 1))
        if IS_FP32:
            b_o = b_o * b_r[:, None] + tl.dot(b_p, b_v.to(tl.float32), input_precision="ieee")
        else:
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(tl.bfloat16), b_v.to(tl.bfloat16))
        b_m = b_m_new

    b_o = tl.where(m_q[:, None] & (b_acc[:, None] > 0.0), b_o / b_acc[:, None], 0.0)
    b_lse = tl.where(m_q & (b_acc > 0.0), b_m + _log2_f32(b_acc), 0.0)

    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(lse_adj + (bos + o_q) * HQ + i_hq, b_lse, mask=m_q)


@triton.jit
def _wall_bwd_preprocess_rsi_kernel(
    o, do, delta,
    V: tl.constexpr, BV: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    o_v = tl.arange(0, BV)
    b_o = tl.load(o + i_n * V + o_v, mask=(o_v < V), other=0.0).to(tl.float32)
    b_do = tl.load(do + i_n * V + o_v, mask=(o_v < V), other=0.0).to(tl.float32)
    tl.store(delta + i_n, tl.sum(b_o * b_do, 0))


@triton.jit
def _wall_bwd_dq_rsi_kernel(
    q, k, v, g_cumsum, g_scalar_cumsum, lse_adj, delta, do,
    dq, dg_cumsum,
    scale, T,
    W_BOUND: tl.constexpr,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_WINDOW: tl.constexpr, USE_SCALAR_G: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    RCP_LN2_C: tl.constexpr = 1.4426950216
    LN2_C: tl.constexpr = 0.6931471805599453

    q_base = q + (bos * HQ + i_hq) * K
    k_base = k + (bos * H + i_h) * K
    v_base = v + (bos * H + i_h) * V
    do_base = do + (bos * HQ + i_hq) * V
    g_base = g_cumsum + (bos * HQ + i_hq) * K

    p_q = tl.make_block_ptr(q_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pq = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_R = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (1, BK), (1, 0))
    p_do = tl.make_block_ptr(do_base, (T, V), (HQ * V, 1), (i_t * BT, 0), (BT, BV), (1, 0))

    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    p_lse = lse_adj + (bos + o_q) * HQ + i_hq
    p_delta = delta + (bos + o_q) * HQ + i_hq

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
    b_R = tl.load(p_R, boundary_check=(0, 1)).to(tl.float32)
    b_q_til = b_q * _exp2_f32(tl.minimum(b_pq - b_R, 0.0))
    if not IS_FP32:
        b_q_til = b_q_til.to(tl.bfloat16)
    b_R_t = tl.trans(b_R)

    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse, mask=m_q, other=0.0).to(tl.float32)[:, None]
    b_delta = tl.load(p_delta, mask=m_q, other=0.0).to(tl.float32)[:, None]
    b_dq_til = tl.zeros([BT, BK], dtype=tl.float32)

    i_start = tl.maximum(((i_t * BT - W_BOUND + 1) // BS) * BS, 0) if USE_WINDOW else 0
    i_end = tl.minimum((i_t + 1) * BT, T)

    for i_s in range(i_start, i_end, BS):
        p_k = tl.make_block_ptr(k_base, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_pk = tl.make_block_ptr(g_base, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v_base, (V, T), (1, H * V), (0, i_s), (BV, BS), (0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_k_til = b_k * _exp2_f32(tl.minimum(b_R_t - b_pk, 110.0))
        if not IS_FP32:
            b_k_til = b_k_til.to(tl.bfloat16)

        b_s = _dot_wall_rsi(b_q_til, b_k_til, IS_FP32) * (scale * RCP_LN2_C)
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0.0).to(tl.float32)[None, :]
            b_s -= b_ck

        diff_idx = (o_q[:, None] - o_k[None, :]).to(tl.uint32)
        m_s = m_k[None, :] & (diff_idx < W_BOUND)
        b_p = tl.where(m_s, _exp2_f32(b_s - b_lse), 0.0)

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dp = _dot_wall_rsi(b_do, b_v, IS_FP32)
        b_ds = b_p * (b_dp - b_delta)

        if IS_FP32:
            b_dq_til = tl.dot(b_ds, tl.trans(b_k_til), b_dq_til, input_precision="ieee")
        else:
            b_dq_til = tl.dot(b_ds.to(tl.bfloat16), tl.trans(b_k_til), b_dq_til)

    b_dq = (b_dq_til * scale) * _exp2_f32(tl.minimum(b_pq - b_R, 0.0))
    b_dg = LN2_C * b_q * b_dq

    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0, 1))


@triton.jit
def _wall_bwd_dk_rsi_kernel(
    q, k, v, g_cumsum, g_scalar_cumsum, lse_adj, delta, do,
    dk, dg_cumsum, dg_scalar_cumsum,
    scale, T,
    W_BOUND: tl.constexpr,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_WINDOW: tl.constexpr, USE_SCALAR_G: tl.constexpr,
    IS_FP32: tl.constexpr, ACCUM_DG_INPLACE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    RCP_LN2_C: tl.constexpr = 1.4426950216
    LN2_C: tl.constexpr = 0.6931471805599453

    q_base = q + (bos * HQ + i_hq) * K
    k_base = k + (bos * H + i_h) * K
    v_base = v + (bos * H + i_h) * V
    do_base = do + (bos * HQ + i_hq) * V
    g_base = g_cumsum + (bos * HQ + i_hq) * K

    p_k = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pk = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_Rk = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (1, BK), (1, 0))
    p_v = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0))

    o_k = i_t * BT + tl.arange(0, BT)
    m_k = o_k < T

    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
    b_Rk = tl.load(p_Rk, boundary_check=(0, 1)).to(tl.float32)
    b_k_til = b_k * _exp2_f32(tl.minimum(b_Rk - b_pk, 110.0))
    if not IS_FP32:
        b_k_til = b_k_til.to(tl.bfloat16)
    b_Rk_t = tl.trans(b_Rk)

    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dk_til = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_SCALAR_G:
        b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0.0).to(tl.float32)[:, None]
        b_dc = tl.zeros([BT], dtype=tl.float32)

    i_end = tl.minimum(T, (i_t + 1) * BT + W_BOUND - 1) if USE_WINDOW else T

    for i_s in range(i_t * BT, i_end, BS):
        p_q_t = tl.make_block_ptr(q_base, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_pq_t = tl.make_block_ptr(g_base, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_do_t = tl.make_block_ptr(do_base, (V, T), (1, HQ * V), (0, i_s), (BV, BS), (0, 1))

        b_q_t = tl.load(p_q_t, boundary_check=(0, 1)).to(tl.float32)
        b_pq_t = tl.load(p_pq_t, boundary_check=(0, 1)).to(tl.float32)
        b_q_til_t = b_q_t * _exp2_f32(tl.minimum(b_pq_t - b_Rk_t, 0.0))
        if not IS_FP32:
            b_q_til_t = b_q_til_t.to(tl.bfloat16)

        b_s = _dot_wall_rsi(b_k_til, b_q_til_t, IS_FP32) * (scale * RCP_LN2_C)
        if USE_SCALAR_G:
            b_s -= b_ck

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        diff_idx = (o_q[None, :] - o_k[:, None]).to(tl.uint32)
        m_s = m_q[None, :] & (diff_idx < W_BOUND)
        b_lse = tl.load(lse_adj + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.0).to(tl.float32)[None, :]
        b_p = tl.where(m_s, _exp2_f32(b_s - b_lse), 0.0)

        b_do_t = tl.load(p_do_t, boundary_check=(0, 1))
        b_delta = tl.load(delta + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.0).to(tl.float32)[None, :]
        b_dp = _dot_wall_rsi(b_v, b_do_t, IS_FP32)
        b_ds = b_p * (b_dp - b_delta)

        if IS_FP32:
            b_dk_til = tl.dot(b_ds, tl.trans(b_q_til_t), b_dk_til, input_precision="ieee")
        else:
            b_dk_til = tl.dot(b_ds.to(tl.bfloat16), tl.trans(b_q_til_t), b_dk_til)
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    b_dk = (b_dk_til * scale) * _exp2_f32(tl.minimum(b_Rk - b_pk, 110.0))
    b_dg = -LN2_C * b_k * b_dk

    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    if ACCUM_DG_INPLACE:
        b_dg_prev = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg_prev + b_dg
    tl.store(p_dg, b_dg, boundary_check=(0, 1))
    if USE_SCALAR_G:
        tl.store(dg_scalar_cumsum + (bos + o_k) * HQ + i_hq, b_dc, mask=m_k)


@triton.jit
def _wall_bwd_dv_rsi_kernel(
    q, k, g_cumsum, g_scalar_cumsum, lse_adj, do, dv,
    scale, T,
    W_BOUND: tl.constexpr,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_WINDOW: tl.constexpr, USE_SCALAR_G: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    RCP_LN2_C: tl.constexpr = 1.4426950216

    q_base = q + (bos * HQ + i_hq) * K
    k_base = k + (bos * H + i_h) * K
    do_base = do + (bos * HQ + i_hq) * V
    g_base = g_cumsum + (bos * HQ + i_hq) * K

    p_k = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pk = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_Rk = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (1, BK), (1, 0))

    o_k = i_t * BT + tl.arange(0, BT)
    m_k = o_k < T

    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
    b_Rk = tl.load(p_Rk, boundary_check=(0, 1)).to(tl.float32)
    b_k_til = b_k * _exp2_f32(tl.minimum(b_Rk - b_pk, 110.0))
    if not IS_FP32:
        b_k_til = b_k_til.to(tl.bfloat16)
    b_Rk_t = tl.trans(b_Rk)

    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    if USE_SCALAR_G:
        b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0.0).to(tl.float32)[:, None]

    i_end = tl.minimum(T, (i_t + 1) * BT + W_BOUND - 1) if USE_WINDOW else T

    for i_s in range(i_t * BT, i_end, BS):
        p_q_t = tl.make_block_ptr(q_base, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_pq_t = tl.make_block_ptr(g_base, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_do = tl.make_block_ptr(do_base, (T, V), (HQ * V, 1), (i_s, 0), (BS, BV), (1, 0))

        b_q_t = tl.load(p_q_t, boundary_check=(0, 1)).to(tl.float32)
        b_pq_t = tl.load(p_pq_t, boundary_check=(0, 1)).to(tl.float32)
        b_q_til_t = b_q_t * _exp2_f32(tl.minimum(b_pq_t - b_Rk_t, 0.0))
        if not IS_FP32:
            b_q_til_t = b_q_til_t.to(tl.bfloat16)

        b_s = _dot_wall_rsi(b_k_til, b_q_til_t, IS_FP32) * (scale * RCP_LN2_C)
        if USE_SCALAR_G:
            b_s -= b_ck

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        diff_idx = (o_q[None, :] - o_k[:, None]).to(tl.uint32)
        m_s = m_q[None, :] & (diff_idx < W_BOUND)
        b_lse = tl.load(lse_adj + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.0).to(tl.float32)[None, :]
        b_p = tl.where(m_s, _exp2_f32(b_s - b_lse), 0.0)

        b_do = tl.load(p_do, boundary_check=(0, 1))
        if IS_FP32:
            b_dv = tl.dot(b_p, b_do.to(tl.float32), b_dv, input_precision="ieee")
        else:
            b_dv = tl.dot(b_p.to(tl.bfloat16), b_do.to(tl.bfloat16), b_dv)

    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _wall_bwd_reduce_dkv_rsi_kernel(
    dk_hq, dv_hq, dk_out, dv_out,
    T, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_bt = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if i_bt >= T:
        return
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    acc_dk = tl.zeros([BK], dtype=tl.float32)
    acc_dv = tl.zeros([BV], dtype=tl.float32)
    base_in_k = ((i_b * T + i_bt) * HQ + i_h * G) * K
    base_in_v = ((i_b * T + i_bt) * HQ + i_h * G) * V
    for g_idx in range(0, G):
        acc_dk += tl.load(dk_hq + base_in_k + g_idx * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32)
        acc_dv += tl.load(dv_hq + base_in_v + g_idx * V + o_v, mask=(o_v < V), other=0.0).to(tl.float32)
    base_out_k = ((i_b * T + i_bt) * H + i_h) * K
    base_out_v = ((i_b * T + i_bt) * H + i_h) * V
    tl.store(dk_out + base_out_k + o_k, acc_dk.to(dk_out.dtype.element_ty), mask=(o_k < K))
    tl.store(dv_out + base_out_v + o_v, acc_dv.to(dv_out.dtype.element_ty), mask=(o_v < V))


@triton.jit
def _wall_bwd_dkv_fused_rsi_kernel(
    q, k, v, g_cumsum, g_scalar_cumsum, lse_adj, delta, do,
    dk, dv, dg_cumsum, dg_scalar_cumsum,
    scale, T,
    W_BOUND: tl.constexpr,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_WINDOW: tl.constexpr, USE_SCALAR_G: tl.constexpr,
    IS_FP32: tl.constexpr, ACCUM_DG_INPLACE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    RCP_LN2_C: tl.constexpr = 1.4426950216
    LN2_C: tl.constexpr = 0.6931471805599453

    q_base = q + (bos * HQ + i_hq) * K
    k_base = k + (bos * H + i_h) * K
    v_base = v + (bos * H + i_h) * V
    do_base = do + (bos * HQ + i_hq) * V
    g_base = g_cumsum + (bos * HQ + i_hq) * K

    p_k = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pk = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_Rk = tl.make_block_ptr(g_base, (T, K), (HQ * K, 1), (i_t * BT, 0), (1, BK), (1, 0))
    p_v = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0))

    o_k = i_t * BT + tl.arange(0, BT)
    m_k = o_k < T

    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
    b_Rk = tl.load(p_Rk, boundary_check=(0, 1)).to(tl.float32)
    b_k_til = b_k * _exp2_f32(tl.minimum(b_Rk - b_pk, 110.0))
    if not IS_FP32:
        b_k_til = b_k_til.to(tl.bfloat16)
    b_Rk_t = tl.trans(b_Rk)

    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dk_til = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    if USE_SCALAR_G:
        b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0.0).to(tl.float32)[:, None]
        b_dc = tl.zeros([BT], dtype=tl.float32)

    i_end = tl.minimum(T, (i_t + 1) * BT + W_BOUND - 1) if USE_WINDOW else T

    for i_s in range(i_t * BT, i_end, BS):
        p_q_t = tl.make_block_ptr(q_base, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_pq_t = tl.make_block_ptr(g_base, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_do_t = tl.make_block_ptr(do_base, (V, T), (1, HQ * V), (0, i_s), (BV, BS), (0, 1))

        b_q_t = tl.load(p_q_t, boundary_check=(0, 1)).to(tl.float32)
        b_pq_t = tl.load(p_pq_t, boundary_check=(0, 1)).to(tl.float32)
        b_q_til_t = b_q_t * _exp2_f32(tl.minimum(b_pq_t - b_Rk_t, 0.0))
        if not IS_FP32:
            b_q_til_t = b_q_til_t.to(tl.bfloat16)

        b_s = _dot_wall_rsi(b_k_til, b_q_til_t, IS_FP32) * (scale * RCP_LN2_C)
        if USE_SCALAR_G:
            b_s -= b_ck

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        diff_idx = (o_q[None, :] - o_k[:, None]).to(tl.uint32)
        m_s = m_q[None, :] & (diff_idx < W_BOUND)
        b_lse = tl.load(lse_adj + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.0).to(tl.float32)[None, :]
        b_p = tl.where(m_s, _exp2_f32(b_s - b_lse), 0.0)

        b_do_t = tl.load(p_do_t, boundary_check=(0, 1))
        if IS_FP32:
            b_dv = tl.dot(b_p, tl.trans(b_do_t.to(tl.float32)), b_dv, input_precision="ieee")
        else:
            b_dv = tl.dot(b_p.to(tl.bfloat16), tl.trans(b_do_t.to(tl.bfloat16)), b_dv)

        b_delta = tl.load(delta + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.0).to(tl.float32)[None, :]
        b_dp = _dot_wall_rsi(b_v, b_do_t, IS_FP32)
        b_ds = b_p * (b_dp - b_delta)

        if IS_FP32:
            b_dk_til = tl.dot(b_ds, tl.trans(b_q_til_t), b_dk_til, input_precision="ieee")
        else:
            b_dk_til = tl.dot(b_ds.to(tl.bfloat16), tl.trans(b_q_til_t), b_dk_til)
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    b_dk = (b_dk_til * scale) * _exp2_f32(tl.minimum(b_Rk - b_pk, 110.0))
    b_dg = -LN2_C * b_k * b_dk

    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if ACCUM_DG_INPLACE:
        b_dg_prev = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg_prev + b_dg
    tl.store(p_dg, b_dg, boundary_check=(0, 1))
    if USE_SCALAR_G:
        tl.store(dg_scalar_cumsum + (bos + o_k) * HQ + i_hq, b_dc, mask=m_k)


def _fast_cumsum_fwd(x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    try:
        from fla.ops.utils.cumsum import chunk_global_cumsum
        return chunk_global_cumsum(x, scale=scale)
    except Exception:
        out = torch.cumsum(x.float(), dim=1)
        return (out * scale if scale is not None else out).contiguous()


def _fast_cumsum_bwd(x: torch.Tensor, scale: Optional[float] = None, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    try:
        from fla.ops.utils.cumsum import chunk_global_cumsum
        res = chunk_global_cumsum(x, reverse=True, scale=scale)
        return res.to(out_dtype) if out_dtype is not None else res
    except Exception:
        res = torch.flip(torch.cumsum(torch.flip(x.float(), dims=[1]), dim=1), dims=[1])
        if scale is not None:
            res = res * scale
        return res.to(out_dtype) if out_dtype is not None else res


def _select_bwd_launch_params(K: int, is_fp32: bool, use_sg: bool):
    """Hardware-verified SM80 launch parameters guaranteeing <= 168 regs and 0 spills."""
    BT = 16
    if K <= 64:
        return (BT, 32, 8), (BT, 16, 8), (BT, 32, 8 if is_fp32 else 4)
    else:
        return (BT, 32, 16), (BT, 16, 16), (BT, 32, 8)


def parallel_wall_attn_fwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_scalar: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    window_size: Optional[int] = None,
    strict_reg_cap: bool = False,
):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    is_fp32 = (q.dtype == torch.float32)
    use_win = (window_size is not None)
    w_bound = window_size if use_win else T
    use_sg = (g_scalar is not None)

    P = _fast_cumsum_fwd(g, scale=RCP_LN2_FLOAT)
    c = _fast_cumsum_fwd(g_scalar, scale=RCP_LN2_FLOAT) if use_sg else None

    o = torch.empty(B, T, HQ, V, device=q.device, dtype=q.dtype)
    lse_adj = torch.empty(B, T, HQ, device=q.device, dtype=torch.float32)
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    if strict_reg_cap or is_fp32:
        BT = 16
        BS = 32 if not is_fp32 else 16
        nw = 8 if (is_fp32 and K > 64) else 4
        ns = 1
    else:
        BT = 64
        BS = 32
        nw = 4
        ns = 2 if K <= 64 else 1

    compiled_fwd = _wall_fwd_rsi_kernel[(triton.cdiv(T, BT), B * HQ)](
        q, k, v, o, P, c, lse_adj, scale, T,
        W_BOUND=w_bound, H=H, HQ=HQ, G=G, K=K, V=V,
        BT=BT, BS=BS, BK=BK, BV=BV,
        USE_WINDOW=use_win, USE_SCALAR_G=use_sg, IS_FP32=is_fp32,
        num_warps=nw, num_stages=ns,
    )
    return o, lse_adj, P, c, compiled_fwd


def parallel_wall_attn_bwd_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse_adj: torch.Tensor,
    P: torch.Tensor,
    c: Optional[torch.Tensor],
    scale: Optional[float] = None,
    window_size: Optional[int] = None,
    g_dtype: torch.dtype = torch.float32,
    gs_dtype: Optional[torch.dtype] = torch.float32,
    strict_reg_cap: bool = False,
):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    is_fp32 = (q.dtype == torch.float32)
    use_win = (window_size is not None)
    w_bound = window_size if use_win else T
    use_sg = (c is not None)
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    delta = torch.empty(B, T, HQ, device=q.device, dtype=torch.float32)
    compiled_pre = _wall_bwd_preprocess_rsi_kernel[(B * T * HQ,)](
        o, do, delta, V=V, BV=BV, num_warps=2
    )

    dq = torch.empty_like(q)
    dk_hq = torch.empty(B, T, HQ, K, device=q.device, dtype=q.dtype)
    dv_hq = torch.empty(B, T, HQ, V, device=q.device, dtype=v.dtype)
    dP = torch.empty(B, T, HQ, K, device=q.device, dtype=torch.float32)
    dc_k = torch.empty(B, T, HQ, device=q.device, dtype=torch.float32) if use_sg else None

    if strict_reg_cap or is_fp32:
        (BT_q, BS_q, nw_q), (BT_k, BS_k, nw_k), (BT_v, BS_v, nw_v) = _select_bwd_launch_params(
            K, is_fp32, use_sg
        )
        compiled_dq = _wall_bwd_dq_rsi_kernel[(triton.cdiv(T, BT_q), B * HQ)](
            q, k, v, P, c, lse_adj, delta, do, dq, dP, scale, T,
            W_BOUND=w_bound, H=H, HQ=HQ, G=G, K=K, V=V,
            BT=BT_q, BS=BS_q, BK=BK, BV=BV,
            USE_WINDOW=use_win, USE_SCALAR_G=use_sg, IS_FP32=is_fp32,
            num_warps=nw_q, num_stages=1,
        )
        dP_k = torch.empty(B, T, HQ, K, device=q.device, dtype=torch.float32)
        compiled_dk = _wall_bwd_dk_rsi_kernel[(triton.cdiv(T, BT_k), B * HQ)](
            q, k, v, P, c, lse_adj, delta, do, dk_hq, dP_k, dc_k, scale, T,
            W_BOUND=w_bound, H=H, HQ=HQ, G=G, K=K, V=V,
            BT=BT_k, BS=BS_k, BK=BK, BV=BV,
            USE_WINDOW=use_win, USE_SCALAR_G=use_sg, IS_FP32=is_fp32,
            ACCUM_DG_INPLACE=False,
            num_warps=nw_k, num_stages=1,
        )
        compiled_dv = _wall_bwd_dv_rsi_kernel[(triton.cdiv(T, BT_v), B * HQ)](
            q, k, P, c, lse_adj, do, dv_hq, scale, T,
            W_BOUND=w_bound, H=H, HQ=HQ, G=G, K=K, V=V,
            BT=BT_v, BS=BS_v, BK=BK, BV=BV,
            USE_WINDOW=use_win, USE_SCALAR_G=use_sg, IS_FP32=is_fp32,
            num_warps=nw_v, num_stages=1,
        )
        dP.add_(dP_k)
    else:
        BT_b = 32
        BS_b = 32
        nw_b = 4 if K <= 64 else 8
        ns_b = 1 if K <= 64 else 2
        compiled_dq = _wall_bwd_dq_rsi_kernel[(triton.cdiv(T, BT_b), B * HQ)](
            q, k, v, P, c, lse_adj, delta, do, dq, dP, scale, T,
            W_BOUND=w_bound, H=H, HQ=HQ, G=G, K=K, V=V,
            BT=BT_b, BS=BS_b, BK=BK, BV=BV,
            USE_WINDOW=use_win, USE_SCALAR_G=use_sg, IS_FP32=False,
            num_warps=nw_b, num_stages=ns_b,
        )
        compiled_dk = _wall_bwd_dkv_fused_rsi_kernel[(triton.cdiv(T, BT_b), B * HQ)](
            q, k, v, P, c, lse_adj, delta, do, dk_hq, dv_hq, dP, dc_k, scale, T,
            W_BOUND=w_bound, H=H, HQ=HQ, G=G, K=K, V=V,
            BT=BT_b, BS=BS_b, BK=BK, BV=BV,
            USE_WINDOW=use_win, USE_SCALAR_G=use_sg, IS_FP32=False,
            ACCUM_DG_INPLACE=True,
            num_warps=nw_b, num_stages=ns_b,
        )
        compiled_dv = compiled_dk

    if G == 1:
        dk = dk_hq
        dv = dv_hq
        compiled_red = None
    else:
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        compiled_red = _wall_bwd_reduce_dkv_rsi_kernel[(T, B * H)](
            dk_hq, dv_hq, dk, dv, T, H=H, HQ=HQ, G=G, K=K, V=V, BK=BK, BV=BV, num_warps=2
        )

    dg = _fast_cumsum_bwd(dP, scale=RCP_LN2_FLOAT, out_dtype=g_dtype)
    dgs = _fast_cumsum_bwd(dc_k, scale=None, out_dtype=gs_dtype) if use_sg else None
    return dq, dk, dv, dg, dgs, {
        "bwd_pre": compiled_pre,
        "bwd_dq": compiled_dq,
        "bwd_dk": compiled_dk,
        "bwd_dv": compiled_dv,
        "bwd_red": compiled_red,
    }


class ParallelWallAttentionRSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, g_scalar=None, scale=None, window_size=None):
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = g.contiguous()
        gs_c = g_scalar.contiguous() if g_scalar is not None else None
        o, lse_adj, P, c, _ = parallel_wall_attn_fwd_rsi(
            q_c, k_c, v_c, g_c, gs_c, scale=scale, window_size=window_size
        )
        ctx.save_for_backward(q_c, k_c, v_c, o, lse_adj, P, c)
        ctx.scale = scale if scale is not None else (q.shape[-1] ** -0.5)
        ctx.window_size = window_size
        ctx.g_dtype = g.dtype
        ctx.gs_dtype = g_scalar.dtype if g_scalar is not None else None
        return o

    @staticmethod
    def backward(ctx, do):
        q_c, k_c, v_c, o, lse_adj, P, c = ctx.saved_tensors
        do_c = do.contiguous()
        dq, dk, dv, dg, dgs, _ = parallel_wall_attn_bwd_rsi(
            q_c, k_c, v_c, o, do_c, lse_adj, P, c,
            scale=ctx.scale,
            window_size=ctx.window_size,
            g_dtype=ctx.g_dtype,
            gs_dtype=ctx.gs_dtype,
        )
        return dq, dk, dv, dg, dgs, None, None


def parallel_wall_attn_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_scalar: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    return ParallelWallAttentionRSIFunction.apply(q, k, v, g, g_scalar, scale, window_size)


# ==============================================================================
# 3. Fused Single-Launch KV-Cache Builder & Zero-Spill Decode Kernel
# ==============================================================================

@triton.jit
def _wall_build_kv_cache_rsi_kernel(
    k, g_cumsum, k_tilde, r_cache,
    T, NC,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, BK: tl.constexpr,
):
    i_c = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    c_start = tl.minimum(i_c * C, T - 1)
    o_d = tl.arange(0, BK)
    p_R_in = g_cumsum + ((i_b * T + c_start) * HQ + i_hq) * K + o_d
    b_R = tl.load(p_R_in, mask=(o_d < K), other=0.0).to(tl.float32)
    p_R_out = r_cache + ((i_b * NC + i_c) * HQ + i_hq) * K + o_d
    tl.store(p_R_out, b_R.to(r_cache.dtype.element_ty), mask=(o_d < K))

    o_t = i_c * C + tl.arange(0, C)
    m_t = (o_t[:, None] < T) & (o_d[None, :] < K)
    p_k = k + (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_d[None, :]
    p_P = g_cumsum + (i_b * T * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_d[None, :]
    p_kt = k_tilde + (i_b * T * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_d[None, :]

    b_k = tl.load(p_k, mask=m_t, other=0.0).to(tl.float32)
    b_P = tl.load(p_P, mask=m_t, other=0.0).to(tl.float32)
    b_kt = b_k * _exp2_f32(b_R[None, :] - b_P)
    tl.store(p_kt, b_kt.to(k_tilde.dtype.element_ty), mask=m_t)


def build_wall_kv_cache_rsi(
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int = 64,
    *,
    out_dtype: Optional[torch.dtype] = None,
):
    B, T, H, K = k.shape
    HQ = g_cumsum.shape[2]
    G = HQ // H
    NC = triton.cdiv(T, chunk_size)
    out_dtype = out_dtype if out_dtype is not None else k.dtype
    BK = triton.next_power_of_2(K)

    k_tilde = torch.empty(B, T, HQ, K, device=k.device, dtype=out_dtype)
    r_cache = torch.empty(B, NC, HQ, K, device=k.device, dtype=g_cumsum.dtype)
    compiled = _wall_build_kv_cache_rsi_kernel[(NC, B * HQ)](
        k, g_cumsum, k_tilde, r_cache, T, NC,
        H=H, HQ=HQ, G=G, K=K, C=chunk_size, BK=BK,
        num_warps=4, num_stages=1,
    )
    return k_tilde, r_cache, compiled


@triton.jit
def _wall_decode_rsi_kernel(
    q, k_tilde, v, o, p_curr, r_cache, g_scalar_cumsum, lse,
    scale, T_q, T_kv, NC,
    W_BOUND: tl.constexpr,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
    USE_WINDOW: tl.constexpr, USE_SCALAR_G: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    RCP_LN2_C: tl.constexpr = 1.4426950216

    bos_q = i_b * T_q
    bos_kv = i_b * T_kv
    bos_nc = i_b * NC

    o_q = i_t * BT + tl.arange(0, BT)
    o_q_global = T_kv - T_q + o_q
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_q = o_q < T_q

    p_q = q + (bos_q * HQ + i_hq) * K + o_q[:, None] * (HQ * K) + o_d[None, :]
    p_pq = p_curr + (bos_q * HQ + i_hq) * K + o_q[:, None] * (HQ * K) + o_d[None, :]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0).to(tl.float32)
    b_pq = tl.load(p_pq, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0).to(tl.float32)

    if USE_SCALAR_G:
        b_cq = tl.load(g_scalar_cumsum + (bos_kv + o_q_global) * HQ + i_hq, mask=m_q, other=0.0).to(tl.float32)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    i_end = tl.minimum(T_kv, T_kv - T_q + (i_t + 1) * BT)
    i_start = tl.maximum(((T_kv - T_q + i_t * BT - W_BOUND + 1) // BS) * BS, 0) if USE_WINDOW else 0

    for i_s in range(i_start, i_end, BS):
        i_c = i_s // C
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T_kv
        o_c = i_c + tl.arange(0, 1)
        p_r = r_cache + (bos_nc * HQ + i_hq) * K + o_c[:, None] * (HQ * K) + o_d[None, :]
        b_R = tl.load(p_r, mask=(o_c[:, None] < NC) & (o_d[None, :] < K), other=0.0).to(tl.float32)
        b_q_til = b_q * _exp2_f32(b_pq - b_R)
        if not IS_FP32:
            b_q_til = b_q_til.to(k_tilde.dtype.element_ty)

        p_kt = k_tilde + (bos_kv * HQ + i_hq) * K + o_d[:, None] + o_k[None, :] * (HQ * K)
        p_v = v + (bos_kv * H + i_h) * V + o_k[:, None] * (H * V) + o_v[None, :]
        b_kt = tl.load(p_kt, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)

        b_s = _dot_wall_rsi(b_q_til, b_kt, IS_FP32) * (scale * RCP_LN2_C)
        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos_kv + o_k) * HQ + i_hq, mask=m_k, other=0.0).to(tl.float32)
            b_s -= b_ck[None, :]

        m_s = m_k[None, :] & (o_k[None, :] <= o_q_global[:, None])
        if USE_WINDOW:
            m_s = m_s & ((o_q_global[:, None] - o_k[None, :]) < W_BOUND)
        b_s = tl.where(m_s, b_s, float("-inf"))

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_mw = tl.where(b_m_new == float("-inf"), 0.0, b_m_new)
        b_r = _exp2_f32(b_m - b_mw)
        b_p = _exp2_f32(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        if IS_FP32:
            b_o = b_o * b_r[:, None] + tl.dot(b_p, b_v.to(tl.float32), input_precision="ieee")
        else:
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = b_m_new

    b_o = b_o / b_acc[:, None]
    b_m += _log2_f32(b_acc)
    if USE_SCALAR_G:
        b_m += b_cq
    p_o = o + (bos_q * HQ + i_hq) * V + o_q[:, None] * (HQ * V) + o_v[None, :]
    p_lse = lse + bos_q * HQ + i_hq + o_q * HQ
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), mask=m_q)


def parallel_wall_attn_decode_rsi(
    q: torch.Tensor,
    v: torch.Tensor,
    p_curr: torch.Tensor,
    k_tilde: torch.Tensor,
    r_cache: torch.Tensor,
    scale: float,
    cache_chunk_size: int = 64,
    g_scalar_cumsum: Optional[torch.Tensor] = None,
    window_size: Optional[int] = None,
):
    B, T_q, HQ, K = q.shape
    _, T_kv, H, V = v.shape
    NC = r_cache.shape[1]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    BT = 16
    use_win = window_size is not None
    w_bound = window_size if use_win else T_kv
    use_sg = g_scalar_cumsum is not None
    is_fp32 = (q.dtype == torch.float32)

    o = torch.empty(B, T_q, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T_q, HQ, dtype=torch.float32, device=q.device)

    compiled = _wall_decode_rsi_kernel[(triton.cdiv(T_q, BT), B * HQ)](
        q, k_tilde, v, o, p_curr, r_cache, g_scalar_cumsum, lse,
        scale, T_q, T_kv, NC,
        W_BOUND=w_bound, H=H, HQ=HQ, G=G, K=K, V=V,
        BT=BT, BS=cache_chunk_size, BK=BK, BV=BV, C=cache_chunk_size,
        USE_WINDOW=use_win, USE_SCALAR_G=use_sg, IS_FP32=is_fp32,
        num_warps=2 if K <= 64 else 4, num_stages=2,
    )
    return o, lse, compiled


# ==============================================================================
# 4. Layer Periphery Fused Triton Kernels (QK-RMSNorm, LogSigmoid, Output-Gate)
# ==============================================================================

@triton.jit
def _wall_qk_rmsnorm_fwd_rsi_kernel(
    q_in, k_in, w_q, w_k, q_out, k_out, rstd_q, rstd_k,
    HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, eps: tl.constexpr,
):
    i_bt = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_d = tl.arange(0, BK)
    m_d = o_d < K
    if i_h < HQ:
        off_q = (i_bt * HQ + i_h) * K + o_d
        x = tl.load(q_in + off_q, mask=m_d, other=0.0).to(tl.float32)
        w = tl.load(w_q + i_h * K + o_d, mask=m_d, other=0.0).to(tl.float32)
        var = tl.sum(x * x, 0) / K
        rstd = tl.math.rsqrt(var + eps)
        tl.store(rstd_q + i_bt * HQ + i_h, rstd)
        y = (x * rstd) * w
        tl.store(q_out + off_q, y.to(q_out.dtype.element_ty), mask=m_d)
    else:
        kh = i_h - HQ
        off_k = (i_bt * H + kh) * K + o_d
        x = tl.load(k_in + off_k, mask=m_d, other=0.0).to(tl.float32)
        w = tl.load(w_k + kh * K + o_d, mask=m_d, other=0.0).to(tl.float32)
        var = tl.sum(x * x, 0) / K
        rstd = tl.math.rsqrt(var + eps)
        tl.store(rstd_k + i_bt * H + kh, rstd)
        y = (x * rstd) * w
        tl.store(k_out + off_k, y.to(k_out.dtype.element_ty), mask=m_d)


@triton.jit
def _wall_qk_rmsnorm_bwd_rsi_kernel(
    q_in, k_in, w_q, w_k, rstd_q, rstd_k, dq_out, dk_out, dq_in, dk_in, dw_q_part, dw_k_part,
    N_BT,
    HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT_RED: tl.constexpr,
):
    i_block = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_d = tl.arange(0, BK)
    m_d = o_d < K
    acc_dw = tl.zeros([BK], dtype=tl.float32)

    if i_h < HQ:
        w = tl.load(w_q + i_h * K + o_d, mask=m_d, other=0.0).to(tl.float32)
        for s in range(0, BT_RED):
            i_bt = i_block * BT_RED + s
            if i_bt < N_BT:
                off_q = (i_bt * HQ + i_h) * K + o_d
                x = tl.load(q_in + off_q, mask=m_d, other=0.0).to(tl.float32)
                dy = tl.load(dq_out + off_q, mask=m_d, other=0.0).to(tl.float32)
                rstd = tl.load(rstd_q + i_bt * HQ + i_h).to(tl.float32)
                x_hat = x * rstd
                acc_dw += dy * x_hat
                wdy = w * dy
                c1 = tl.sum(x_hat * wdy, 0) / K
                dx = rstd * (wdy - x_hat * c1)
                tl.store(dq_in + off_q, dx.to(dq_in.dtype.element_ty), mask=m_d)
        tl.store(dw_q_part + (i_block * HQ + i_h) * K + o_d, acc_dw, mask=m_d)
    else:
        kh = i_h - HQ
        w = tl.load(w_k + kh * K + o_d, mask=m_d, other=0.0).to(tl.float32)
        for s in range(0, BT_RED):
            i_bt = i_block * BT_RED + s
            if i_bt < N_BT:
                off_k = (i_bt * H + kh) * K + o_d
                x = tl.load(k_in + off_k, mask=m_d, other=0.0).to(tl.float32)
                dy = tl.load(dk_out + off_k, mask=m_d, other=0.0).to(tl.float32)
                rstd = tl.load(rstd_k + i_bt * H + kh).to(tl.float32)
                x_hat = x * rstd
                acc_dw += dy * x_hat
                wdy = w * dy
                c1 = tl.sum(x_hat * wdy, 0) / K
                dx = rstd * (wdy - x_hat * c1)
                tl.store(dk_in + off_k, dx.to(dk_in.dtype.element_ty), mask=m_d)
        tl.store(dw_k_part + (i_block * H + kh) * K + o_d, acc_dw, mask=m_d)


class FusedQKRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, w_q, w_k, eps=1e-5):
        B, T, HQ, K = q.shape
        H = k.shape[2]
        q_c = q.contiguous()
        k_c = k.contiguous()
        q_out = torch.empty_like(q_c)
        k_out = torch.empty_like(k_c)
        N_BT = B * T
        rstd_q = torch.empty(N_BT, HQ, device=q.device, dtype=torch.float32)
        rstd_k = torch.empty(N_BT, H, device=q.device, dtype=torch.float32)
        BK = triton.next_power_of_2(K)
        _wall_qk_rmsnorm_fwd_rsi_kernel[(N_BT, HQ + H)](
            q_c, k_c, w_q, w_k, q_out, k_out, rstd_q, rstd_k,
            HQ=HQ, H=H, K=K, BK=BK, eps=eps, num_warps=2
        )
        ctx.save_for_backward(q_c, k_c, w_q, w_k, rstd_q, rstd_k)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out, dk_out):
        q_c, k_c, w_q, w_k, rstd_q, rstd_k = ctx.saved_tensors
        B, T, HQ, K = q_c.shape
        H = k_c.shape[2]
        N_BT = B * T
        BT_RED = 32
        n_blocks = triton.cdiv(N_BT, BT_RED)
        dq_in = torch.empty_like(q_c)
        dk_in = torch.empty_like(k_c)
        dw_q_part = torch.empty(n_blocks, HQ * K, device=q_c.device, dtype=torch.float32)
        dw_k_part = torch.empty(n_blocks, H * K, device=q_c.device, dtype=torch.float32)
        BK = triton.next_power_of_2(K)
        _wall_qk_rmsnorm_bwd_rsi_kernel[(n_blocks, HQ + H)](
            q_c, k_c, w_q, w_k, rstd_q, rstd_k,
            dq_out.contiguous(), dk_out.contiguous(),
            dq_in, dk_in, dw_q_part, dw_k_part,
            N_BT, HQ=HQ, H=H, K=K, BK=BK, BT_RED=BT_RED, num_warps=2
        )
        dw_q = dw_q_part.sum(dim=0).to(w_q.dtype)
        dw_k = dw_k_part.sum(dim=0).to(w_k.dtype)
        return dq_in, dk_in, dw_q, dw_k, None


@triton.jit
def _wall_logsigmoid_fwd_rsi_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    abs_x = tl.abs(x)
    y = tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-abs_x))
    tl.store(y_ptr + offs, y, mask=m)


@triton.jit
def _wall_logsigmoid_bwd_rsi_kernel(x_ptr, dy_ptr, dx_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    # d/dx logsigmoid(x) = 1 - sigmoid(x) = sigmoid(-x) = 1 / (1 + exp(x))
    dx = dy * tl.sigmoid(-x)
    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=m)


class FusedLogSigmoidFP32Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_c = x.contiguous()
        y = torch.empty(x_c.shape, device=x_c.device, dtype=torch.float32)
        N = x_c.numel()
        BLOCK = 1024
        _wall_logsigmoid_fwd_rsi_kernel[(triton.cdiv(N, BLOCK),)](x_c, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x_c)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x_c,) = ctx.saved_tensors
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        _wall_logsigmoid_bwd_rsi_kernel[(triton.cdiv(N, BLOCK),)](x_c, dy_c, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


@triton.jit
def _wall_out_gate_fwd_rsi_kernel(o_ptr, g_ptr, out_ptr, stride_gm, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    m = cols < D
    o = tl.load(o_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * stride_gm + cols, mask=m, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    out = o * sig
    tl.store(out_ptr + row * D + cols, out.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _wall_out_gate_bwd_rsi_kernel(o_ptr, g_ptr, dout_ptr, do_ptr, dg_ptr, stride_gm, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    m = cols < D
    o = tl.load(o_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * stride_gm + cols, mask=m, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    do = dout * sig
    dg = dout * o * sig * (1.0 - sig)
    tl.store(do_ptr + row * D + cols, do.to(do_ptr.dtype.element_ty), mask=m)
    tl.store(dg_ptr + row * D + cols, dg.to(dg_ptr.dtype.element_ty), mask=m)


class FusedOutGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, z_gate):
        o_c = o.contiguous()
        out = torch.empty_like(o_c)
        D = o_c.shape[-1]
        M = o_c.numel() // D
        stride_gm = z_gate.stride(-2)
        BLOCK_D = triton.next_power_of_2(D)
        _wall_out_gate_fwd_rsi_kernel[(M,)](o_c, z_gate, out, stride_gm, D=D, BLOCK_D=BLOCK_D, num_warps=4)
        ctx.save_for_backward(o_c, z_gate)
        ctx.stride_gm = stride_gm
        return out

    @staticmethod
    def backward(ctx, dout):
        o_c, z_gate = ctx.saved_tensors
        dout_c = dout.contiguous()
        do = torch.empty_like(o_c)
        dz = torch.empty_like(o_c)
        D = o_c.shape[-1]
        M = o_c.numel() // D
        BLOCK_D = triton.next_power_of_2(D)
        _wall_out_gate_bwd_rsi_kernel[(M,)](
            o_c, z_gate, dout_c, do, dz, ctx.stride_gm, D=D, BLOCK_D=BLOCK_D, num_warps=4
        )
        return do, dz


# ==============================================================================
# 5. WallAttentionBlockRSI (5-Stage Ablation Ladder A0 -> A4)
# ==============================================================================

class WallAttentionBlockRSI(nn.Module):
    """
    WallAttention Layer supporting 5 progressive optimization stages:
      - Stage A0: Upstream WallAttention baseline (separate q/k/v/g/gs/o_gate projections,
                  eager F.logsigmoid, 2 separate GroupNorms, upstream parallel_wall_attn,
                  eager sigmoid(o_gate) * o).
      - Stage A1: + Zero-spill Fixed-Anchor parallel_wall_attn_rsi Fwd/Bwd kernels.
      - Stage A2: + Fused single-GEMM QKVG-Gate input projection.
      - Stage A3: + Fused multi-head QK-RMSNorm & fused LogSigmoid Triton kernels.
      - Stage A4: + Fused output sigmoid-gate multiplication Triton kernel.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        window_size: Optional[int] = 512,
        use_output_gate: bool = True,
        use_scalar_gate: bool = True,
        gate_init_bias: float = 8.0,
        stage: str = "A4",
    ):
        super().__init__()
        assert stage in ("A0", "A1", "A2", "A3", "A4")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.use_output_gate = use_output_gate
        self.use_scalar_gate = use_scalar_gate
        self.gate_init_bias = gate_init_bias
        self.stage = stage

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.constant_(self.g_proj.bias, gate_init_bias)

        if use_scalar_gate:
            self.gs_proj = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.constant_(self.gs_proj.bias, gate_init_bias)
        else:
            self.gs_proj = None

        if use_output_gate:
            self.o_gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.o_gate_proj = None

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if qk_norm:
            from fla.modules import GroupNorm
            self.q_norm = GroupNorm(num_groups=num_heads, hidden_size=hidden_size, is_rms_norm=True)
            self.k_norm = GroupNorm(num_groups=num_kv_heads, hidden_size=self.kv_dim, is_rms_norm=True)

        # Fused QKVG-Gate projection for Stages A2..A4
        total_out = (
            hidden_size
            + self.kv_dim
            + self.kv_dim
            + hidden_size
            + (num_heads if use_scalar_gate else 0)
            + (hidden_size if use_output_gate else 0)
        )
        self.fused_in_proj = nn.Linear(hidden_size, total_out, bias=True)
        self.sync_fused_weights()

    @torch.no_grad()
    def sync_fused_weights(self):
        w_list = [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.g_proj.weight]
        z_q = self.q_proj.bias if self.qkv_bias else torch.zeros(self.hidden_size, device=self.q_proj.weight.device, dtype=self.q_proj.weight.dtype)
        z_k = self.k_proj.bias if self.qkv_bias else torch.zeros(self.kv_dim, device=self.k_proj.weight.device, dtype=self.k_proj.weight.dtype)
        z_v = self.v_proj.bias if self.qkv_bias else torch.zeros(self.kv_dim, device=self.v_proj.weight.device, dtype=self.v_proj.weight.dtype)
        b_list = [z_q, z_k, z_v, self.g_proj.bias]
        if self.use_scalar_gate:
            w_list.append(self.gs_proj.weight)
            b_list.append(self.gs_proj.bias)
        if self.use_output_gate:
            w_list.append(self.o_gate_proj.weight)
            b_list.append(torch.zeros(self.hidden_size, device=self.o_gate_proj.weight.device, dtype=self.o_gate_proj.weight.dtype))
        self.fused_in_proj.weight.copy_(torch.cat(w_list, dim=0))
        self.fused_in_proj.bias.copy_(torch.cat(b_list, dim=0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden_states.shape
        HQ, H, K = self.num_heads, self.num_kv_heads, self.head_dim

        if self.stage in ("A0", "A1"):
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            g_raw = self.g_proj(hidden_states)
            gs_raw = self.gs_proj(hidden_states) if self.use_scalar_gate else None
            z_gate = self.o_gate_proj(hidden_states) if self.use_output_gate else None
        else:
            fused = self.fused_in_proj(hidden_states)
            splits = [D, self.kv_dim, self.kv_dim, D]
            if self.use_scalar_gate:
                splits.append(HQ)
            if self.use_output_gate:
                splits.append(D)
            parts = torch.split(fused, splits, dim=-1)
            q, k, v, g_raw = parts[0], parts[1], parts[2], parts[3]
            idx = 4
            gs_raw = parts[idx] if self.use_scalar_gate else None
            if self.use_scalar_gate:
                idx += 1
            z_gate = parts[idx] if self.use_output_gate else None

        if self.stage in ("A0", "A1", "A2"):
            g = F.logsigmoid(g_raw.float()).view(B, T, HQ, K)
            gs = F.logsigmoid(gs_raw.float()).view(B, T, HQ) if self.use_scalar_gate else None
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            q = q.view(B, T, HQ, K)
            k = k.view(B, T, H, K)
        else:
            g = FusedLogSigmoidFP32Function.apply(g_raw).view(B, T, HQ, K)
            gs = FusedLogSigmoidFP32Function.apply(gs_raw).view(B, T, HQ) if self.use_scalar_gate else None
            q = q.view(B, T, HQ, K)
            k = k.view(B, T, H, K)
            if self.qk_norm:
                q, k = FusedQKRMSNormFunction.apply(q, k, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps)

        v = v.view(B, T, H, K)

        if self.stage == "A0":
            from fla.ops.wall_attn.parallel import parallel_wall_attn
            o = parallel_wall_attn(q, k, v, g, g_scalar=gs, window_size=self.window_size)
        else:
            o = parallel_wall_attn_rsi(q, k, v, g, g_scalar=gs, window_size=self.window_size)

        o = o.reshape(B, T, D)
        if self.use_output_gate:
            if self.stage == "A4":
                o = FusedOutGateFunction.apply(o, z_gate)
            else:
                o = torch.sigmoid(z_gate) * o
        return self.o_proj(o)


# ==============================================================================
# 6. 8-GPU NVSwitch Context Parallelism (Wall-CP = 8)
# ==============================================================================

class _AllToAllSeqToHeadFn(torch.autograd.Function):
    """Differentiable [B, T_local, P, HQ_local, D] -> [B, T, HQ_local, D] all-to-all."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        P = dist.get_world_size(group)
        B, T_loc, HQ, D = x.shape
        HQ_loc = HQ // P
        # Pack by destination rank p: [P, B, T_loc, HQ_loc, D]
        send_buf = x.view(B, T_loc, P, HQ_loc, D).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        # Reassemble sequence T = P * T_loc: [B, P, T_loc, HQ_loc, D] -> [B, T, HQ_loc, D]
        return recv_buf.permute(1, 0, 2, 3, 4).reshape(B, P * T_loc, HQ_loc, D).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        P = dist.get_world_size(group)
        B, T, HQ_loc, D = grad_out.shape
        T_loc = T // P
        send_buf = grad_out.view(B, P, T_loc, HQ_loc, D).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        grad_in = recv_buf.permute(1, 2, 0, 3, 4).reshape(B, T_loc, P * HQ_loc, D).contiguous()
        return grad_in, None


class _AllGatherSeqFn(torch.autograd.Function):
    """Differentiable sequence AllGather (with ReduceScatter backward) for ring/prefix baseline."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        P = dist.get_world_size(group)
        B, T_loc = x.shape[:2]
        rest = x.shape[2:]
        x_c = x.contiguous()
        out = torch.empty((P, B, T_loc, *rest), device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(out, x_c, group=group)
        return out.permute(1, 0, 2, *range(3, out.ndim)).reshape(B, P * T_loc, *rest).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        P = dist.get_world_size(group)
        B, T = grad_out.shape[:2]
        T_loc = T // P
        rest = grad_out.shape[2:]
        send = grad_out.view(B, P, T_loc, *rest).permute(1, 0, 2, *range(3, grad_out.ndim + 1)).contiguous()
        recv = torch.empty((B, T_loc, *rest), device=grad_out.device, dtype=grad_out.dtype)
        dist.reduce_scatter_tensor(recv, send, group=group)
        return recv, None


def wall_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    gs_local: Optional[torch.Tensor] = None,
    window_size: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Fused Ulysses Head-Sharded Wall-CP = 8 over NVSwitch:
    Packs (q, k, v, g[, gs]) into a SINGLE fused all-to-all buffer (1 NCCL launch in
    forward and 1 in backward), redistributes from sequence-local [B, T/P, HQ, ...] to
    head-local [B, T, HQ/P, ...] with 100% balanced causal compute across all 8 GPUs,
    runs zero-spill parallel_wall_attn_rsi, and redistributes the output back to [B, T/P, HQ, V].
    """
    if group is None:
        group = dist.group.WORLD
    P = dist.get_world_size(group)
    B, T_loc, HQ, K = q_local.shape
    H = k_local.shape[2]
    V = v_local.shape[-1]

    if H < P:
        rep = P // H
        k_send = k_local.repeat_interleave(rep, dim=2)
        v_send = v_local.repeat_interleave(rep, dim=2)
        H_eff = P
    else:
        k_send = k_local
        v_send = v_local
        H_eff = H

    # Reshape each tensor to [B, T_loc, P, -1] so a SINGLE all-to-all exchanges all inputs
    q_f = q_local.float().view(B, T_loc, P, (HQ // P) * K)
    k_f = k_send.float().view(B, T_loc, P, (H_eff // P) * K)
    v_f = v_send.float().view(B, T_loc, P, (H_eff // P) * V)
    g_f = g_local.float().view(B, T_loc, P, (HQ // P) * K)
    pack_list = [q_f, k_f, v_f, g_f]
    splits = [
        (HQ // P) * K,
        (H_eff // P) * K,
        (H_eff // P) * V,
        (HQ // P) * K,
    ]
    if gs_local is not None:
        gs_f = gs_local.float().view(B, T_loc, P, HQ // P)
        pack_list.append(gs_f)
        splits.append(HQ // P)

    packed_local = torch.cat(pack_list, dim=-1)  # [B, T_loc, P, D_pack]
    packed_full = _AllToAllSeqToHeadFn.apply(packed_local, group)  # [B, T, 1, D_pack]
    T = P * T_loc
    parts = torch.split(packed_full.squeeze(2), splits, dim=-1)

    q_full = parts[0].view(B, T, HQ // P, K).to(q_local.dtype).contiguous()
    k_full = parts[1].view(B, T, H_eff // P, K).to(k_local.dtype).contiguous()
    v_full = parts[2].view(B, T, H_eff // P, V).to(v_local.dtype).contiguous()
    g_full = parts[3].view(B, T, HQ // P, K).to(g_local.dtype).contiguous()
    gs_full = parts[4].view(B, T, HQ // P).to(gs_local.dtype).contiguous() if gs_local is not None else None

    o_full = parallel_wall_attn_rsi(
        q_full, k_full, v_full, g_full,
        g_scalar=gs_full,
        scale=K ** -0.5,
        window_size=window_size,
    )

    return _AllToAllHeadToSeqFn.apply(o_full, group)


class _AllToAllHeadToSeqFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_full: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        P = dist.get_world_size(group)
        B, T, HQ_loc, D = x_full.shape
        T_loc = T // P
        send_buf = x_full.view(B, P, T_loc, HQ_loc, D).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 2, 0, 3, 4).reshape(B, T_loc, P * HQ_loc, D).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        P = dist.get_world_size(group)
        B, T_loc, HQ, D = grad_out.shape
        HQ_loc = HQ // P
        send_buf = grad_out.view(B, T_loc, P, HQ_loc, D).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        grad_in = recv_buf.permute(1, 0, 2, 3, 4).reshape(B, P * T_loc, HQ_loc, D).contiguous()
        return grad_in, None


def wall_cp8_ring_baseline(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    gs_local: Optional[torch.Tensor] = None,
    window_size: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Sequence AllGather + Causal Prefix Baseline for Wall-CP = 8:
    AllGathers (q, k, v, g, gs) across the 8 sequence ranks (with ReduceScatter in backward),
    evaluates the causal prefix [0 : (rank + 1) * T_local] on each rank, and slices the
    local query window [rank * T_local : (rank + 1) * T_local].
    Demonstrates the O(P) sequence AllGather communication + causal rank-7 load imbalance.
    """
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group)
    T_loc = q_local.shape[1]
    end_t = (rank + 1) * T_loc

    q_all = _AllGatherSeqFn.apply(q_local, group)[:, :end_t].contiguous()
    k_all = _AllGatherSeqFn.apply(k_local, group)[:, :end_t].contiguous()
    v_all = _AllGatherSeqFn.apply(v_local, group)[:, :end_t].contiguous()
    g_all = _AllGatherSeqFn.apply(g_local, group)[:, :end_t].contiguous()
    gs_all = _AllGatherSeqFn.apply(gs_local, group)[:, :end_t].contiguous() if gs_local is not None else None

    o_prefix = parallel_wall_attn_rsi(
        q_all, k_all, v_all, g_all,
        g_scalar=gs_all,
        scale=q_local.shape[-1] ** -0.5,
        window_size=window_size,
    )
    return o_prefix[:, rank * T_loc : end_t].contiguous()
