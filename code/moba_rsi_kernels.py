#!/usr/bin/env python3
"""
Round-32 RSI (E158 / v149): Mixture of Block Attention (MoBA) Zero-Spill SM80 Kernels
Paper: "MoBA: Mixture of Block Attention for Long-Context LLMs" (arXiv:2502.13189, Moonshot AI / Kimi Team)
Upstream Reference: fla-org/flash-linear-attention/tree/main/fla/ops/moba & fla/layers/moba.py

Resolves 5 concrete upstream SM80 hardware, algorithmic, and numerical pathologies:
  1. 0 Triton Kernels + Two-Stream _flash_attn_varlen + 12-Op Eager LSE Merge + Host-Sync .nonzero() + (topk-1)x Q / 2x KV HBM Duplication
  2. Deterministic GQA RuntimeError Crash in MoBA.forward When num_kv_heads < num_heads
  3. Deterministic FP32 RuntimeError Crash & BF16 Mean + TF32 einsum Top-k Gate Boundary Flips
  4. Dense Full-History Attention Fallback in MoBA Decoding
  5. Unfused MoBA Layer Periphery & Missing 8-GPU NVSwitch Context Parallelism (MoBA-CP = 8)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# 1. Exact FP32 Block-Centroid Reduction & In-Register Top-(k-1) Block-Bitmask
# ============================================================================

@triton.jit
def _moba_block_centroid_fwd_rsi_kernel(
    k,
    k_mean,
    T,
    HKV: tl.constexpr,
    K: tl.constexpr,
    NC: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BS: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HKV
    i_hkv = i_bh % HKV

    o_d = tl.arange(0, K)
    b_sum = tl.zeros([K], dtype=tl.float32)
    c_start = i_c * CHUNK_SIZE
    c_end = tl.minimum(c_start + CHUNK_SIZE, T)
    valid_cnt = tl.maximum(c_end - c_start, 1).to(tl.float32)

    k_base = k + i_b * (T * HKV * K) + i_hkv * K
    for s_off in range(0, CHUNK_SIZE, BS):
        o_t = c_start + s_off + tl.arange(0, BS)
        m_t = o_t < c_end
        b_k = tl.load(k_base + o_t[:, None] * (HKV * K) + o_d[None, :], mask=m_t[:, None], other=0.0).to(tl.float32)
        b_sum += tl.sum(b_k, axis=0)

    b_mean = b_sum / valid_cnt
    out_ptr = k_mean + (i_b * HKV + i_hkv) * (NC * K) + i_c * K + o_d
    tl.store(out_ptr, b_mean)


@triton.jit
def _bitwise_or_u64(a, b):
    return a | b


@triton.jit
def _moba_gate_topk_bitmask_rsi_kernel(
    q,
    k_mean,
    q_block_mask,
    qtile_block_mask,
    T,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    NC: tl.constexpr,
    NC_PAD: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BM: tl.constexpr,
    TOPK_HIST: tl.constexpr,
    IS_FP32: tl.constexpr,
    MUTATE_MODE: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_hkv = i_hq // G

    o_m = i_m * BM + tl.arange(0, BM)
    m_m = o_m < T
    o_d = tl.arange(0, K)
    o_c = tl.arange(0, NC_PAD)
    c_q = o_m // CHUNK_SIZE

    q_ptr = q + i_b * (T * HQ * K) + o_m[:, None] * (HQ * K) + i_hq * K + o_d[None, :]
    kc_ptr = k_mean + (i_b * HKV + i_hkv) * (NC * K) + o_c[None, :] * K + o_d[:, None]

    b_q = tl.load(q_ptr, mask=m_m[:, None], other=0.0).to(tl.float32)
    b_kc = tl.load(kc_ptr, mask=o_c[None, :] < NC, other=0.0).to(tl.float32)

    # Always evaluate the compact [BM, K] x [K, NC_PAD] block-centroid gate in IEEE FP32
    # so BF16 inputs never suffer top-k block boundary flips against the FP64 oracle.
    b_scores = tl.dot(b_q, b_kc, input_precision="ieee")

    # Strictly prior blocks: c_k < c_q and c_k < NC
    valid_prior = (o_c[None, :] < c_q[:, None]) & (o_c[None, :] < NC) & m_m[:, None]
    # Add tiny index-monotonic tie-breaker so ties deterministically pick the lowest block index
    b_scores = tl.where(valid_prior, b_scores - o_c[None, :].to(tl.float32) * 1e-7, float("-inf"))

    b_qmask = tl.zeros([BM], dtype=tl.uint64)
    if MUTATE_MODE != 1:
        for _ in range(TOPK_HIST):
            max_v = tl.max(b_scores, axis=1)
            best_c = tl.argmax(b_scores, axis=1)
            has_valid = max_v > -1e30
            bit = tl.where(has_valid, (tl.full([BM], 1, dtype=tl.uint64) << best_c.to(tl.uint64)), 0)
            b_qmask = b_qmask | bit
            b_scores = tl.where(o_c[None, :] == best_c[:, None], float("-inf"), b_scores)

    # Unify historical blocks (c_k < c_q) and local causal block (c_k == c_q) by setting bit c_q
    local_bit = tl.where(m_m, (tl.full([BM], 1, dtype=tl.uint64) << c_q.to(tl.uint64)), 0)
    b_qmask = b_qmask | local_bit

    tile_mask = tl.reduce(b_qmask, axis=0, combine_fn=_bitwise_or_u64)
    NT_M = tl.cdiv(T, BM)
    tl.store(q_block_mask + (i_b * HQ + i_hq) * T + o_m, b_qmask, mask=m_m)
    tl.store(qtile_block_mask + (i_b * HQ + i_hq) * NT_M + i_m, tile_mask)


# ============================================================================
# 2. Single-Pass Block-Bitmask Tensor-Core Flash-MoBA Forward Kernel
# ============================================================================

@triton.jit
def _moba_fwd_rsi_kernel(
    q,
    k,
    v,
    q_block_mask,
    qtile_block_mask,
    o,
    lse,
    T,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IS_FP32: tl.constexpr,
    MUTATE_MODE: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_hkv = i_hq // G

    NT_M = tl.cdiv(T, BM)
    o_m = i_m * BM + tl.arange(0, BM)
    m_m = o_m < T
    o_d = tl.arange(0, K)
    o_v = tl.arange(0, V)

    q_ptr = q + i_b * (T * HQ * K) + o_m[:, None] * (HQ * K) + i_hq * K + o_d[None, :]
    b_q = tl.load(q_ptr, mask=m_m[:, None], other=0.0)
    b_qmask = tl.load(q_block_mask + (i_b * HQ + i_hq) * T + o_m, mask=m_m, other=0).to(tl.uint64)
    tile_mask = tl.load(qtile_block_mask + (i_b * HQ + i_hq) * NT_M + i_m).to(tl.uint64)

    scale: tl.constexpr = 1.0 / math.sqrt(float(K))
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale_log2: tl.constexpr = scale * RCP_LN2

    b_m = tl.full([BM], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BM], dtype=tl.float32)
    b_o = tl.zeros([BM, V], dtype=tl.float32)

    k_base = k + i_b * (T * HKV * K) + i_hkv * K
    v_base = v + i_b * (T * HKV * V) + i_hkv * V

    # All BM queries in tile i_m belong to the same MoBA block c_q (since CHUNK_SIZE % BM == 0).
    # Because bit c_q is set in b_qmask/tile_mask and every historical key in c_k < c_q satisfies
    # o_n < c_q * CHUNK_SIZE <= o_m, a single unified loop over c_k in [0, c_q + 1) handles both
    # selected historical blocks and the local causal block!
    c_q = (i_m * BM) // CHUNK_SIZE
    q_tile_end = tl.minimum((i_m + 1) * BM, T)
    if MUTATE_MODE == 2:
        q_tile_end = tl.minimum((c_q + 1) * CHUNK_SIZE, T)

    for c_k in range(0, c_q + 1):
        active_c = ((tile_mask >> c_k.to(tl.uint64)) & 1) != 0
        if active_c:
            row_sel = ((b_qmask >> c_k.to(tl.uint64)) & 1) != 0
            n_block_start = c_k * CHUNK_SIZE
            n_block_end = tl.minimum(n_block_start + CHUNK_SIZE, q_tile_end)
            for n_start in range(n_block_start, n_block_end, BN):
                o_n = n_start + tl.arange(0, BN)
                m_n = o_n < T
                b_k = tl.load(k_base + o_n[:, None] * (HKV * K) + o_d[None, :], mask=m_n[:, None], other=0.0)
                b_v = tl.load(v_base + o_n[:, None] * (HKV * V) + o_v[None, :], mask=m_n[:, None], other=0.0)
                if IS_FP32:
                    b_s = tl.dot(b_q, tl.trans(b_k), input_precision="ieee") * scale_log2
                else:
                    b_s = tl.dot(b_q, tl.trans(b_k)) * scale_log2
                if MUTATE_MODE == 2:
                    m_valid = row_sel[:, None] & m_n[None, :]
                else:
                    m_valid = row_sel[:, None] & (o_m[:, None] >= o_n[None, :]) & m_n[None, :]
                b_s = tl.where(m_valid, b_s, float("-inf"))
                b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
                b_mw = tl.where(b_m_new == float("-inf"), 0.0, b_m_new)
                b_r = tl.math.exp2(b_m - b_mw)
                b_p = tl.math.exp2(b_s - b_mw[:, None])
                b_acc = b_acc * b_r + tl.sum(b_p, axis=1)
                if IS_FP32:
                    b_o = b_o * b_r[:, None] + tl.dot(b_p, b_v, input_precision="ieee")
                else:
                    b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
                b_m = b_m_new

    b_o = b_o / b_acc[:, None]
    b_lse = b_m + tl.math.log2(b_acc)

    o_ptr = o + i_b * (T * HQ * V) + o_m[:, None] * (HQ * V) + i_hq * V + o_v[None, :]
    tl.store(o_ptr, b_o.to(o.dtype.element_ty), mask=m_m[:, None])
    tl.store(lse + (i_b * HQ + i_hq) * T + o_m, b_lse, mask=m_m)


# ============================================================================
# 3. Zero-Atomic, Zero-Spill Split dq / dkv Backward Kernels
# ============================================================================

@triton.jit
def _moba_bwd_prep_delta_rsi_kernel(
    o,
    do,
    delta,
    T,
    HQ: tl.constexpr,
    V: tl.constexpr,
    BM: tl.constexpr,
    MUTATE_MODE: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ

    o_m = i_m * BM + tl.arange(0, BM)
    m_m = o_m < T
    o_v = tl.arange(0, V)
    off = i_b * (T * HQ * V) + o_m[:, None] * (HQ * V) + i_hq * V + o_v[None, :]
    b_o = tl.load(o + off, mask=m_m[:, None], other=0.0).to(tl.float32)
    b_do = tl.load(do + off, mask=m_m[:, None], other=0.0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do, axis=1)
    if MUTATE_MODE == 3:
        b_delta = -b_delta
    tl.store(delta + (i_b * HQ + i_hq) * T + o_m, b_delta, mask=m_m)


@triton.jit
def _moba_bwd_dq_rsi_kernel(
    q,
    k,
    v,
    do,
    lse,
    delta,
    q_block_mask,
    qtile_block_mask,
    dq,
    T,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_hkv = i_hq // G

    NT_M = tl.cdiv(T, BM)
    o_m = i_m * BM + tl.arange(0, BM)
    m_m = o_m < T
    o_d = tl.arange(0, K)
    o_v = tl.arange(0, V)

    q_off = i_b * (T * HQ * K) + o_m[:, None] * (HQ * K) + i_hq * K + o_d[None, :]
    do_off = i_b * (T * HQ * V) + o_m[:, None] * (HQ * V) + i_hq * V + o_v[None, :]
    b_q = tl.load(q + q_off, mask=m_m[:, None], other=0.0)
    b_do = tl.load(do + do_off, mask=m_m[:, None], other=0.0)
    b_lse = tl.load(lse + (i_b * HQ + i_hq) * T + o_m, mask=m_m, other=0.0)
    b_delta = tl.load(delta + (i_b * HQ + i_hq) * T + o_m, mask=m_m, other=0.0)
    b_qmask = tl.load(q_block_mask + (i_b * HQ + i_hq) * T + o_m, mask=m_m, other=0).to(tl.uint64)
    tile_mask = tl.reduce(b_qmask, axis=0, combine_fn=_bitwise_or_u64)

    scale: tl.constexpr = 1.0 / math.sqrt(float(K))
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale_log2: tl.constexpr = scale * RCP_LN2

    b_dq = tl.zeros([BM, K], dtype=tl.float32)
    k_base = k + i_b * (T * HKV * K) + i_hkv * K
    v_base = v + i_b * (T * HKV * V) + i_hkv * V

    c_q = (i_m * BM) // CHUNK_SIZE
    q_tile_end = tl.minimum((i_m + 1) * BM, T)

    # Single unified hardware loop with coalesced row-major [BN, K] and [BN, V] loads
    for c_k in range(0, c_q + 1):
        active_c = ((tile_mask >> c_k.to(tl.uint64)) & 1) != 0
        if active_c:
            row_sel = ((b_qmask >> c_k.to(tl.uint64)) & 1) != 0
            n_block_start = c_k * CHUNK_SIZE
            n_block_end = tl.minimum(n_block_start + CHUNK_SIZE, q_tile_end)
            for n_start in range(n_block_start, n_block_end, BN):
                o_n = n_start + tl.arange(0, BN)
                m_n = o_n < T
                b_k = tl.load(k_base + o_n[:, None] * (HKV * K) + o_d[None, :], mask=m_n[:, None], other=0.0)
                b_v = tl.load(v_base + o_n[:, None] * (HKV * V) + o_v[None, :], mask=m_n[:, None], other=0.0)
                if IS_FP32:
                    b_s = tl.dot(b_q, tl.trans(b_k), input_precision="ieee") * scale_log2
                    b_dp = tl.dot(b_do, tl.trans(b_v), input_precision="ieee")
                else:
                    b_s = tl.dot(b_q, tl.trans(b_k)) * scale_log2
                    b_dp = tl.dot(b_do, tl.trans(b_v))
                m_valid = row_sel[:, None] & (o_m[:, None] >= o_n[None, :]) & m_n[None, :]
                b_p = tl.where(m_valid, tl.math.exp2(b_s - b_lse[:, None]), 0.0)
                b_ds = b_p * (b_dp - b_delta[:, None])
                if IS_FP32:
                    b_dq = tl.dot(b_ds, b_k, b_dq, input_precision="ieee")
                else:
                    b_dq = tl.dot(b_ds.to(b_k.dtype), b_k, b_dq)

    b_dq = b_dq * scale
    tl.store(dq + q_off, b_dq.to(dq.dtype.element_ty), mask=m_m[:, None])


@triton.jit(do_not_specialize=["T", "g_count"])
def _moba_bwd_dkv_rsi_kernel(
    q,
    k,
    v,
    do,
    lse,
    delta,
    q_block_mask,
    qtile_block_mask,
    dk,
    dv,
    T,
    g_count,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_bhkv = tl.program_id(1)
    i_b = i_bhkv // HKV
    i_hkv = i_bhkv % HKV

    NT_M = tl.cdiv(T, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    m_n = o_n < T
    o_d = tl.arange(0, K)
    o_v = tl.arange(0, V)

    k_off = i_b * (T * HKV * K) + o_n[:, None] * (HKV * K) + i_hkv * K + o_d[None, :]
    v_off = i_b * (T * HKV * V) + o_n[:, None] * (HKV * V) + i_hkv * V + o_v[None, :]
    b_k = tl.load(k + k_off, mask=m_n[:, None], other=0.0)
    b_v = tl.load(v + v_off, mask=m_n[:, None], other=0.0)

    b_dk = tl.zeros([BN, K], dtype=tl.float32)
    b_dv = tl.zeros([BN, V], dtype=tl.float32)

    scale: tl.constexpr = 1.0 / math.sqrt(float(K))
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale_log2: tl.constexpr = scale * RCP_LN2

    c_k = (i_n * BN) // CHUNK_SIZE
    c_k_u64 = c_k.to(tl.uint64)
    m_local_start = (i_n * BN) // BM

    for i_g in range(0, g_count):
        i_hq = i_hkv * G + i_g
        q_base = q + i_b * (T * HQ * K) + i_hq * K
        do_base = do + i_b * (T * HQ * V) + i_hq * V
        lse_base = lse + (i_b * HQ + i_hq) * T
        delta_base = delta + (i_b * HQ + i_hq) * T
        qmask_base = q_block_mask + (i_b * HQ + i_hq) * T
        tmask_base = qtile_block_mask + (i_b * HQ + i_hq) * NT_M

        # Single unified loop over i_m in [m_local_start, NT_M) (eliminates dual-loop live-range duplication!)
        for i_m in range(m_local_start, NT_M):
            tile_mask = tl.load(tmask_base + i_m).to(tl.uint64)
            active_m = ((tile_mask >> c_k_u64) & 1) != 0
            if active_m:
                o_m = i_m * BM + tl.arange(0, BM)
                m_m = o_m < T
                b_q = tl.load(q_base + o_m[:, None] * (HQ * K) + o_d[None, :], mask=m_m[:, None], other=0.0)
                b_do = tl.load(do_base + o_m[:, None] * (HQ * V) + o_v[None, :], mask=m_m[:, None], other=0.0)
                b_lse = tl.load(lse_base + o_m, mask=m_m, other=0.0)
                b_delta = tl.load(delta_base + o_m, mask=m_m, other=0.0)
                b_qmask = tl.load(qmask_base + o_m, mask=m_m, other=0).to(tl.uint64)
                col_sel = (((b_qmask >> c_k_u64) & 1) != 0) & m_m
                if IS_FP32:
                    b_s = tl.dot(b_k, tl.trans(b_q), input_precision="ieee") * scale_log2
                    b_dp = tl.dot(b_v, tl.trans(b_do), input_precision="ieee")
                else:
                    b_s = tl.dot(b_k, tl.trans(b_q)) * scale_log2
                    b_dp = tl.dot(b_v, tl.trans(b_do))
                m_valid = col_sel[None, :] & (o_n[:, None] <= o_m[None, :])
                b_p = tl.where(m_valid, tl.math.exp2(b_s - b_lse[None, :]), 0.0)
                b_ds = b_p * (b_dp - b_delta[None, :])
                if IS_FP32:
                    b_dv = tl.dot(b_p, b_do, b_dv, input_precision="ieee")
                    b_dk = tl.dot(b_ds, b_q, b_dk, input_precision="ieee")
                else:
                    b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
                    b_dk = tl.dot(b_ds.to(b_q.dtype), b_q, b_dk)

    b_dk = b_dk * scale
    tl.store(dk + k_off, b_dk.to(dk.dtype.element_ty), mask=m_n[:, None])
    tl.store(dv + v_off, b_dv.to(dv.dtype.element_ty), mask=m_n[:, None])


# ============================================================================
# 4. Single-Pass Centroid-Routed Block-Sparse Decoding Kernels (moba_decode_rsi)
# ============================================================================

@triton.jit
def _moba_decode_gate_bitmask_rsi_kernel(
    q,
    k_mean,
    decode_block_mask,
    T_Q: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    NC: tl.constexpr,
    NC_PAD: tl.constexpr,
    TOPK_HIST: tl.constexpr,
):
    i_tq = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_hkv = i_hq // G

    o_d = tl.arange(0, K)
    o_c = tl.arange(0, NC_PAD)
    b_q = tl.load(q + i_b * (T_Q * HQ * K) + i_tq * (HQ * K) + i_hq * K + o_d).to(tl.float32)
    b_kc = tl.load(
        k_mean + (i_b * HKV + i_hkv) * (NC * K) + o_c[:, None] * K + o_d[None, :],
        mask=o_c[:, None] < (NC - 1),
        other=0.0,
    ).to(tl.float32)
    b_scores = tl.sum(b_kc * b_q[None, :], axis=1)
    b_scores = tl.where(o_c < (NC - 1), b_scores - o_c.to(tl.float32) * 1e-7, float("-inf"))

    b_mask = tl.zeros([], dtype=tl.uint64)
    for _ in range(TOPK_HIST):
        max_v = tl.max(b_scores, axis=0)
        best_c = tl.argmax(b_scores, axis=0)
        if max_v > -1e30:
            b_mask = b_mask | (tl.full([], 1, dtype=tl.uint64) << best_c.to(tl.uint64))
        b_scores = tl.where(o_c == best_c, float("-inf"), b_scores)

    # Always include the last (local/active) block NC - 1
    last_bit = tl.full([], 1, dtype=tl.uint64) << tl.full([], NC - 1, dtype=tl.uint64)
    b_mask = b_mask | last_bit
    tl.store(decode_block_mask + (i_b * HQ + i_hq) * T_Q + i_tq, b_mask)


@triton.jit
def _moba_decode_readout_rsi_kernel(
    q,
    k_cache,
    v_cache,
    decode_block_mask,
    o,
    T_Q: tl.constexpr,
    T_KV: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NC: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BN: tl.constexpr,
):
    i_tq = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_hkv = i_hq // G

    o_d = tl.arange(0, K)
    o_v = tl.arange(0, V)
    b_q = tl.load(q + i_b * (T_Q * HQ * K) + i_tq * (HQ * K) + i_hq * K + o_d).to(tl.float32)
    b_mask = tl.load(decode_block_mask + (i_b * HQ + i_hq) * T_Q + i_tq).to(tl.uint64)

    scale: tl.constexpr = 1.0 / math.sqrt(float(K))
    RCP_LN2: tl.constexpr = 1.4426950408889634
    b_q_scaled = b_q * (scale * RCP_LN2)

    b_m = float("-inf")
    b_acc = 0.0
    b_o = tl.zeros([V], dtype=tl.float32)

    k_base = k_cache + i_b * (T_KV * HKV * K) + i_hkv * K
    v_base = v_cache + i_b * (T_KV * HKV * V) + i_hkv * V
    q_abs_pos = (T_KV - T_Q) + i_tq
    NSUB: tl.constexpr = CHUNK_SIZE // BN

    for c_k in range(0, NC):
        c_k_u64 = tl.full([], c_k, dtype=tl.uint64)
        if ((b_mask >> c_k_u64) & 1) != 0:
            for sub in range(0, NSUB):
                n_start = c_k * CHUNK_SIZE + sub * BN
                o_n = n_start + tl.arange(0, BN)
                m_n = o_n <= q_abs_pos
                b_k = tl.load(k_base + o_n[:, None] * (HKV * K) + o_d[None, :], mask=m_n[:, None], other=0.0).to(tl.float32)
                b_v = tl.load(v_base + o_n[:, None] * (HKV * V) + o_v[None, :], mask=m_n[:, None], other=0.0).to(tl.float32)
                b_s = tl.sum(b_k * b_q_scaled[None, :], axis=1)
                b_s = tl.where(m_n, b_s, float("-inf"))
                m_new = tl.maximum(b_m, tl.max(b_s, axis=0))
                mw = tl.where(m_new == float("-inf"), 0.0, m_new)
                r = tl.math.exp2(b_m - mw)
                p = tl.math.exp2(b_s - mw)
                b_acc = b_acc * r + tl.sum(p, axis=0)
                b_o = b_o * r + tl.sum(p[:, None] * b_v, axis=0)
                b_m = m_new

    b_o = b_o / b_acc
    o_ptr = o + i_b * (T_Q * HQ * V) + i_tq * (HQ * V) + i_hq * V + o_v
    tl.store(o_ptr, b_o.to(o.dtype.element_ty))


# ============================================================================
# 5. Superfused Periphery Kernels (Pre-RMSNorm, Fused QK-RMSNorm+RoPE, Sigmoid-Gated RMSNorm)
# ============================================================================

@triton.jit
def _moba_rmsnorm_fwd_rsi_kernel(
    x,
    weight,
    y,
    rstd,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    i_row = tl.program_id(0)
    o_d = tl.arange(0, D)
    b_x = tl.load(x + i_row * D + o_d).to(tl.float32)
    b_w = tl.load(weight + o_d).to(tl.float32)
    b_rstd = tl.rsqrt(tl.sum(b_x * b_x, axis=0) * (1.0 / float(D)) + EPS)
    tl.store(rstd + i_row, b_rstd)
    tl.store(y + i_row * D + o_d, (b_x * b_rstd * b_w).to(y.dtype.element_ty))


@triton.jit
def _moba_rmsnorm_bwd_rsi_kernel(
    x,
    weight,
    rstd,
    dy,
    dx,
    dw_part,
    N_ROWS,
    D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    i_p = tl.program_id(0)
    row_start = i_p * ROWS_PER_PROG
    o_d = tl.arange(0, D)
    b_w = tl.load(weight + o_d).to(tl.float32)
    b_dw = tl.zeros([D], dtype=tl.float32)
    inv_d = 1.0 / float(D)

    for r_off in range(ROWS_PER_PROG):
        row = row_start + r_off
        if row < N_ROWS:
            b_x = tl.load(x + row * D + o_d).to(tl.float32)
            b_dy = tl.load(dy + row * D + o_d).to(tl.float32)
            b_rstd = tl.load(rstd + row).to(tl.float32)
            x_hat = b_x * b_rstd
            wdy = b_w * b_dy
            b_dw += b_dy * x_hat
            c1 = tl.sum(x_hat * wdy, axis=0) * inv_d
            b_dx = (wdy - x_hat * c1) * b_rstd
            tl.store(dx + row * D + o_d, b_dx.to(dx.dtype.element_ty))

    tl.store(dw_part + i_p * D + o_d, b_dw)


@triton.jit
def _moba_qk_rmsnorm_rope_fwd_rsi_kernel(
    x,
    weight,
    cos_cache,
    sin_cache,
    y,
    rstd,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    HALF_K: tl.constexpr,
    EPS: tl.constexpr,
):
    i_row = tl.program_id(0)
    i_t = (i_row // H) % T
    o_h = tl.arange(0, HALF_K)

    base = x + i_row * K
    b_x0 = tl.load(base + o_h).to(tl.float32)
    b_x1 = tl.load(base + HALF_K + o_h).to(tl.float32)
    b_w0 = tl.load(weight + o_h).to(tl.float32)
    b_w1 = tl.load(weight + HALF_K + o_h).to(tl.float32)

    ss = (tl.sum(b_x0 * b_x0, axis=0) + tl.sum(b_x1 * b_x1, axis=0)) * (1.0 / float(K))
    b_rstd = tl.rsqrt(ss + EPS)
    tl.store(rstd + i_row, b_rstd)

    n0 = b_x0 * b_rstd * b_w0
    n1 = b_x1 * b_rstd * b_w1
    b_cos = tl.load(cos_cache + i_t * HALF_K + o_h).to(tl.float32)
    b_sin = tl.load(sin_cache + i_t * HALF_K + o_h).to(tl.float32)

    y0 = n0 * b_cos - n1 * b_sin
    y1 = n1 * b_cos + n0 * b_sin
    out_base = y + i_row * K
    tl.store(out_base + o_h, y0.to(y.dtype.element_ty))
    tl.store(out_base + HALF_K + o_h, y1.to(y.dtype.element_ty))


@triton.jit
def _moba_qk_rmsnorm_rope_bwd_rsi_kernel(
    x,
    weight,
    cos_cache,
    sin_cache,
    rstd,
    dy,
    dx,
    dw_part,
    N_ROWS,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    HALF_K: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    i_p = tl.program_id(0)
    row_start = i_p * ROWS_PER_PROG
    o_h = tl.arange(0, HALF_K)
    b_w0 = tl.load(weight + o_h).to(tl.float32)
    b_w1 = tl.load(weight + HALF_K + o_h).to(tl.float32)
    b_dw0 = tl.zeros([HALF_K], dtype=tl.float32)
    b_dw1 = tl.zeros([HALF_K], dtype=tl.float32)
    inv_k: tl.constexpr = 1.0 / float(K)

    for r_off in range(ROWS_PER_PROG):
        row = row_start + r_off
        if row < N_ROWS:
            i_t = (row // H) % T
            b_cos = tl.load(cos_cache + i_t * HALF_K + o_h).to(tl.float32)
            b_sin = tl.load(sin_cache + i_t * HALF_K + o_h).to(tl.float32)
            dy0 = tl.load(dy + row * K + o_h).to(tl.float32)
            dy1 = tl.load(dy + row * K + HALF_K + o_h).to(tl.float32)
            # Inverse rotation
            dn0 = dy0 * b_cos + dy1 * b_sin
            dn1 = dy1 * b_cos - dy0 * b_sin

            x0 = tl.load(x + row * K + o_h).to(tl.float32)
            x1 = tl.load(x + row * K + HALF_K + o_h).to(tl.float32)
            b_rstd = tl.load(rstd + row).to(tl.float32)
            xh0 = x0 * b_rstd
            xh1 = x1 * b_rstd
            b_dw0 += dn0 * xh0
            b_dw1 += dn1 * xh1
            wdn0 = b_w0 * dn0
            wdn1 = b_w1 * dn1
            c1 = (tl.sum(xh0 * wdn0, axis=0) + tl.sum(xh1 * wdn1, axis=0)) * inv_k
            dx0 = (wdn0 - xh0 * c1) * b_rstd
            dx1 = (wdn1 - xh1 * c1) * b_rstd
            tl.store(dx + row * K + o_h, dx0.to(dx.dtype.element_ty))
            tl.store(dx + row * K + HALF_K + o_h, dx1.to(dx.dtype.element_ty))

    tl.store(dw_part + i_p * K + o_h, b_dw0)
    tl.store(dw_part + i_p * K + HALF_K + o_h, b_dw1)


@triton.jit
def _moba_rmsnorm_sigmoid_gated_fwd_rsi_kernel(
    x,
    g,
    weight,
    y,
    rstd,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    i_row = tl.program_id(0)
    o_d = tl.arange(0, D)
    b_x = tl.load(x + i_row * D + o_d).to(tl.float32)
    b_g = tl.load(g + i_row * D + o_d).to(tl.float32)
    b_w = tl.load(weight + o_d).to(tl.float32)
    b_rstd = tl.rsqrt(tl.sum(b_x * b_x, axis=0) * (1.0 / float(D)) + EPS)
    tl.store(rstd + i_row, b_rstd)
    sig_g = tl.sigmoid(b_g)
    b_y = (b_x * b_rstd * b_w) * sig_g
    tl.store(y + i_row * D + o_d, b_y.to(y.dtype.element_ty))


@triton.jit
def _moba_rmsnorm_sigmoid_gated_bwd_rsi_kernel(
    x,
    g,
    weight,
    rstd,
    dy,
    dx,
    dg,
    dw_part,
    N_ROWS,
    D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    i_p = tl.program_id(0)
    row_start = i_p * ROWS_PER_PROG
    o_d = tl.arange(0, D)
    b_w = tl.load(weight + o_d).to(tl.float32)
    b_dw = tl.zeros([D], dtype=tl.float32)
    inv_d: tl.constexpr = 1.0 / float(D)

    for r_off in range(ROWS_PER_PROG):
        row = row_start + r_off
        if row < N_ROWS:
            b_x = tl.load(x + row * D + o_d).to(tl.float32)
            b_g = tl.load(g + row * D + o_d).to(tl.float32)
            b_dy = tl.load(dy + row * D + o_d).to(tl.float32)
            b_rstd = tl.load(rstd + row).to(tl.float32)

            sig_g = tl.sigmoid(b_g)
            x_hat = b_x * b_rstd
            normed = x_hat * b_w
            b_dg = b_dy * normed * sig_g * (1.0 - sig_g)
            dy_norm = b_dy * sig_g
            wdy = b_w * dy_norm
            b_dw += dy_norm * x_hat
            c1 = tl.sum(x_hat * wdy, axis=0) * inv_d
            b_dx = (wdy - x_hat * c1) * b_rstd
            tl.store(dx + row * D + o_d, b_dx.to(dx.dtype.element_ty))
            tl.store(dg + row * D + o_d, b_dg.to(dg.dtype.element_ty))

    tl.store(dw_part + i_p * D + o_d, b_dw)


# ============================================================================
# 6. Python Autograd Function, FP64 Oracle, Decode, & MoBABlockRSI
# ============================================================================

def compute_moba_bitmask_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    chunk_size: int = 128,
    topk: int = 4,
    bm: int = 32,
    mutate_mode: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, HQ, K = q.shape
    HKV = k.shape[2]
    G = HQ // HKV
    NC = triton.cdiv(T, chunk_size)
    NC_PAD = max(16, triton.next_power_of_2(NC))
    NT_M = triton.cdiv(T, bm)
    IS_FP32 = q.dtype == torch.float32

    k_mean = torch.empty((B, HKV, NC, K), device=q.device, dtype=torch.float32)
    BS_C = min(64, chunk_size)
    _moba_block_centroid_fwd_rsi_kernel[(NC, B * HKV)](
        k, k_mean, T, HKV=HKV, K=K, NC=NC, CHUNK_SIZE=chunk_size, BS=BS_C, num_warps=4
    )

    q_block_mask = torch.empty((B, HQ, T), device=q.device, dtype=torch.uint64)
    qtile_block_mask = torch.empty((B, HQ, NT_M), device=q.device, dtype=torch.uint64)
    topk_hist = max(0, min(topk - 1, NC - 1))
    _moba_gate_topk_bitmask_rsi_kernel[(NT_M, B * HQ)](
        q,
        k_mean,
        q_block_mask,
        qtile_block_mask,
        T,
        HQ=HQ,
        HKV=HKV,
        G=G,
        K=K,
        NC=NC,
        NC_PAD=NC_PAD,
        CHUNK_SIZE=chunk_size,
        BM=bm,
        TOPK_HIST=topk_hist,
        IS_FP32=IS_FP32,
        MUTATE_MODE=mutate_mode,
        num_warps=4,
    )
    return q_block_mask, qtile_block_mask, k_mean


class ParallelMoBARSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        precomputed_q_block_mask: Optional[torch.Tensor],
        precomputed_qtile_block_mask: Optional[torch.Tensor],
        chunk_size: int,
        topk: int,
        mutate_mode: int,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        B, T, HQ, K = q.shape
        HKV = k.shape[2]
        V = v.shape[3]
        G = HQ // HKV
        IS_FP32 = q.dtype == torch.float32
        BM = 32 if not IS_FP32 else 16
        BN = 32 if not IS_FP32 else 16
        NT_M = triton.cdiv(T, BM)

        if precomputed_q_block_mask is not None and precomputed_qtile_block_mask is not None:
            q_block_mask = precomputed_q_block_mask
            qtile_block_mask = precomputed_qtile_block_mask
        else:
            q_block_mask, qtile_block_mask, _ = compute_moba_bitmask_rsi(
                q, k, chunk_size=chunk_size, topk=topk, bm=BM, mutate_mode=mutate_mode
            )

        o = torch.empty((B, T, HQ, V), device=q.device, dtype=q.dtype)
        lse = torch.empty((B, HQ, T), device=q.device, dtype=torch.float32)

        _moba_fwd_rsi_kernel[(NT_M, B * HQ)](
            q,
            k,
            v,
            q_block_mask,
            qtile_block_mask,
            o,
            lse,
            T,
            HQ=HQ,
            HKV=HKV,
            G=G,
            K=K,
            V=V,
            CHUNK_SIZE=chunk_size,
            BM=BM,
            BN=BN,
            IS_FP32=IS_FP32,
            MUTATE_MODE=mutate_mode,
            num_warps=4,
            num_stages=2 if (K <= 64 and not IS_FP32) else 1,
        )

        ctx.save_for_backward(q, k, v, o, lse, q_block_mask, qtile_block_mask)
        ctx.chunk_size = chunk_size
        ctx.topk = topk
        ctx.mutate_mode = mutate_mode
        ctx.mark_non_differentiable(q_block_mask, qtile_block_mask)
        return o, q_block_mask, qtile_block_mask

    @staticmethod
    def backward(ctx, do: torch.Tensor, _dm1, _dm2):
        q, k, v, o, lse, q_block_mask, qtile_block_mask = ctx.saved_tensors
        do = do.contiguous()
        B, T, HQ, K = q.shape
        HKV = k.shape[2]
        V = v.shape[3]
        G = HQ // HKV
        IS_FP32 = q.dtype == torch.float32
        BM = 32 if not IS_FP32 else 16
        BN = 32 if not IS_FP32 else 16
        BM_DQ = 32 if (K <= 64 and not IS_FP32) else 16
        NT_M = triton.cdiv(T, BM)
        NT_M_DQ = triton.cdiv(T, BM_DQ)
        NT_N = triton.cdiv(T, BN)

        delta = torch.empty((B, HQ, T), device=q.device, dtype=torch.float32)
        _moba_bwd_prep_delta_rsi_kernel[(triton.cdiv(T, 64), B * HQ)](
            o, do, delta, T, HQ=HQ, V=V, BM=64, MUTATE_MODE=ctx.mutate_mode, num_warps=4
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        _moba_bwd_dq_rsi_kernel[(NT_M_DQ, B * HQ)](
            q,
            k,
            v,
            do,
            lse,
            delta,
            q_block_mask,
            qtile_block_mask,
            dq,
            T,
            HQ=HQ,
            HKV=HKV,
            G=G,
            K=K,
            V=V,
            CHUNK_SIZE=ctx.chunk_size,
            BM=BM_DQ,
            BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4,
            num_stages=2 if not IS_FP32 else 1,
        )

        _moba_bwd_dkv_rsi_kernel[(NT_N, B * HKV)](
            q,
            k,
            v,
            do,
            lse,
            delta,
            q_block_mask,
            qtile_block_mask,
            dk,
            dv,
            T,
            G,
            HQ=HQ,
            HKV=HKV,
            G=G,
            K=K,
            V=V,
            CHUNK_SIZE=ctx.chunk_size,
            BM=BM,
            BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4,
            num_stages=2 if (K <= 64 and not IS_FP32) else 1,
        )

        return dq, dk, dv, None, None, None, None, None


def parallel_moba_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
    topk: int = 4,
    precomputed_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_masks: bool = False,
    mutate_mode: int = 0,
):
    q_bm = precomputed_masks[0] if precomputed_masks is not None else None
    qt_bm = precomputed_masks[1] if precomputed_masks is not None else None
    o, q_bm_out, qt_bm_out = ParallelMoBARSIFunction.apply(
        q, k, v, q_bm, qt_bm, chunk_size, topk, mutate_mode
    )
    if return_masks:
        return o, (q_bm_out, qt_bm_out)
    return o


def moba_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
    topk: int = 4,
) -> torch.Tensor:
    """
    Exact Differentiable FP64 Reference Oracle for Mixture of Block Attention (MoBA, arXiv:2502.13189)
    supporting both MHA (HQ == HKV) and GQA (HQ % HKV == 0).
    """
    qd = q.to(torch.float64)
    kd = k.to(torch.float64)
    vd = v.to(torch.float64)
    B, T, HQ, K = qd.shape
    HKV = kd.shape[2]
    G = HQ // HKV
    NC = T // chunk_size

    with torch.no_grad():
        k_mean = kd.view(B, NC, chunk_size, HKV, K).mean(dim=2)  # [B, NC, HKV, K]
        if G > 1:
            k_mean_hq = k_mean.unsqueeze(3).expand(B, NC, HKV, G, K).reshape(B, NC, HQ, K)
        else:
            k_mean_hq = k_mean
        gate_scores = torch.einsum("bthk,bchk->bhtc", qd, k_mean_hq)
        t_idx = torch.arange(T, device=q.device)
        c_q = t_idx // chunk_size
        c_k = torch.arange(NC, device=q.device)
        prior_mask = c_k[None, :] < c_q[:, None]
        gate_scores = gate_scores - c_k[None, None, None, :].to(torch.float64) * 1e-7
        gate_scores = gate_scores.masked_fill(~prior_mask[None, None, :, :], float("-inf"))

        k_hist = max(0, min(topk - 1, NC - 1))
        block_mask = torch.zeros((B, HQ, T, NC), dtype=torch.bool, device=q.device)
        if k_hist > 0:
            vals, idx = torch.topk(gate_scores, k=k_hist, dim=-1)
            valid_sel = ~torch.isinf(vals)
            block_mask.scatter_(-1, idx, valid_sel)
        hist_tok_mask = block_mask.unsqueeze(-1).expand(B, HQ, T, NC, chunk_size).reshape(B, HQ, T, T)
        s_idx = torch.arange(T, device=q.device)
        local_causal = (t_idx[:, None] // chunk_size == s_idx[None, :] // chunk_size) & (
            s_idx[None, :] <= t_idx[:, None]
        )
        full_mask = hist_tok_mask | local_causal[None, None, :, :]

    if G > 1:
        kd_hq = kd.unsqueeze(3).expand(B, T, HKV, G, K).reshape(B, T, HQ, K)
        vd_hq = vd.unsqueeze(3).expand(B, T, HKV, G, vd.shape[-1]).reshape(B, T, HQ, vd.shape[-1])
    else:
        kd_hq = kd
        vd_hq = vd

    scale = K ** -0.5
    scores = torch.einsum("bthk,bshk->bhts", qd, kd_hq) * scale
    scores = scores.masked_fill(~full_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bshv->bthv", probs, vd_hq)
    return out


def moba_decode_rsi(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    chunk_size: int = 128,
    topk: int = 4,
    k_mean_cache: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    B, T_Q, HQ, K = q.shape
    _, T_KV, HKV, _ = k_cache.shape
    V = v_cache.shape[3]
    G = HQ // HKV
    NC = triton.cdiv(T_KV, chunk_size)
    NC_PAD = max(16, triton.next_power_of_2(NC))

    if k_mean_cache is None:
        k_mean_cache = torch.empty((B, HKV, NC, K), device=q.device, dtype=torch.float32)
        _moba_block_centroid_fwd_rsi_kernel[(NC, B * HKV)](
            k_cache, k_mean_cache, T_KV, HKV=HKV, K=K, NC=NC, CHUNK_SIZE=chunk_size, BS=min(64, chunk_size), num_warps=4
        )

    decode_block_mask = torch.empty((B, HQ, T_Q), device=q.device, dtype=torch.uint64)
    topk_hist = max(0, min(topk - 1, NC - 1))
    _moba_decode_gate_bitmask_rsi_kernel[(T_Q, B * HQ)](
        q,
        k_mean_cache,
        decode_block_mask,
        T_Q=T_Q,
        HQ=HQ,
        HKV=HKV,
        G=G,
        K=K,
        NC=NC,
        NC_PAD=NC_PAD,
        TOPK_HIST=topk_hist,
        num_warps=2,
    )

    o = torch.empty((B, T_Q, HQ, V), device=q.device, dtype=q.dtype)
    BN = min(64, chunk_size)
    _moba_decode_readout_rsi_kernel[(T_Q, B * HQ)](
        q,
        k_cache,
        v_cache,
        decode_block_mask,
        o,
        T_Q=T_Q,
        T_KV=T_KV,
        HQ=HQ,
        HKV=HKV,
        G=G,
        K=K,
        V=V,
        NC=NC,
        CHUNK_SIZE=chunk_size,
        BN=BN,
        num_warps=4,
    )
    return o


# ============================================================================
# 7. Superfused Periphery Autograd Wrappers & MoBABlockRSI (Stages A0..A4)
# ============================================================================

class FusedMoBARMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        x_c = x.contiguous()
        D = x_c.shape[-1]
        n_rows = x_c.numel() // D
        y = torch.empty_like(x_c)
        rstd = torch.empty((n_rows,), device=x_c.device, dtype=torch.float32)
        _moba_rmsnorm_fwd_rsi_kernel[(n_rows,)](x_c, weight, y, rstd, D=D, EPS=eps, num_warps=4)
        ctx.save_for_backward(x_c, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, weight, rstd = ctx.saved_tensors
        dy_c = dy.contiguous()
        D = x_c.shape[-1]
        n_rows = x_c.numel() // D
        dx = torch.empty_like(x_c)
        rows_per_prog = 16
        n_progs = triton.cdiv(n_rows, rows_per_prog)
        dw_part = torch.empty((n_progs, D), device=x_c.device, dtype=torch.float32)
        _moba_rmsnorm_bwd_rsi_kernel[(n_progs,)](
            x_c, weight, rstd, dy_c, dx, dw_part, n_rows, D=D, ROWS_PER_PROG=rows_per_prog, num_warps=4
        )
        dw = dw_part.sum(dim=0).to(weight.dtype)
        return dx, dw, None


class FusedMoBAQKRMSNormRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, eps: float = 1e-6):
        x_c = x.contiguous()
        B, T, H, K = x_c.shape
        HALF_K = K // 2
        n_rows = B * T * H
        y = torch.empty_like(x_c)
        rstd = torch.empty((n_rows,), device=x_c.device, dtype=torch.float32)
        _moba_qk_rmsnorm_rope_fwd_rsi_kernel[(n_rows,)](
            x_c, weight, cos_cache, sin_cache, y, rstd, T, H=H, K=K, HALF_K=HALF_K, EPS=eps, num_warps=2
        )
        ctx.save_for_backward(x_c, weight, cos_cache, sin_cache, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, weight, cos_cache, sin_cache, rstd = ctx.saved_tensors
        dy_c = dy.contiguous()
        B, T, H, K = x_c.shape
        HALF_K = K // 2
        n_rows = B * T * H
        dx = torch.empty_like(x_c)
        rows_per_prog = 16
        n_progs = triton.cdiv(n_rows, rows_per_prog)
        dw_part = torch.empty((n_progs, K), device=x_c.device, dtype=torch.float32)
        _moba_qk_rmsnorm_rope_bwd_rsi_kernel[(n_progs,)](
            x_c,
            weight,
            cos_cache,
            sin_cache,
            rstd,
            dy_c,
            dx,
            dw_part,
            n_rows,
            T,
            H=H,
            K=K,
            HALF_K=HALF_K,
            ROWS_PER_PROG=rows_per_prog,
            num_warps=2,
        )
        dw = dw_part.sum(dim=0).to(weight.dtype)
        return dx, dw, None, None, None


class FusedMoBARMSNormSigmoidGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        x_c = x.contiguous()
        g_c = g.contiguous()
        D = x_c.shape[-1]
        n_rows = x_c.numel() // D
        y = torch.empty_like(x_c)
        rstd = torch.empty((n_rows,), device=x_c.device, dtype=torch.float32)
        _moba_rmsnorm_sigmoid_gated_fwd_rsi_kernel[(n_rows,)](
            x_c, g_c, weight, y, rstd, D=D, EPS=eps, num_warps=2
        )
        ctx.save_for_backward(x_c, g_c, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, g_c, weight, rstd = ctx.saved_tensors
        dy_c = dy.contiguous()
        D = x_c.shape[-1]
        n_rows = x_c.numel() // D
        dx = torch.empty_like(x_c)
        dg = torch.empty_like(g_c)
        rows_per_prog = 16
        n_progs = triton.cdiv(n_rows, rows_per_prog)
        dw_part = torch.empty((n_progs, D), device=x_c.device, dtype=torch.float32)
        _moba_rmsnorm_sigmoid_gated_bwd_rsi_kernel[(n_progs,)](
            x_c, g_c, weight, rstd, dy_c, dx, dg, dw_part, n_rows, D=D, ROWS_PER_PROG=rows_per_prog, num_warps=2
        )
        dw = dw_part.sum(dim=0).to(weight.dtype)
        return dx, dg, dw, None


class MoBABlockRSI(nn.Module):
    """
    5-Stage Progressive Ablation Block for Mixture of Block Attention (MoBA, arXiv:2502.13189):
      - Stage A0: Upstream fla/ops/moba/parallel.py (parallel_moba) + unfused projections, QK-RMSNorm, RoPE, & gated norm
      - Stage A1: Single-Pass Block-Bitmask Flash-MoBA (parallel_moba_rsi, zero CPU syncs, zero moba_q/moba_kv copies)
      - Stage A2: + Fused QK-RMSNorm + RoPE (_moba_qk_rmsnorm_rope_fwd/bwd_rsi_kernel)
      - Stage A3: + Single-GEMM Packed Projection (qkvg_proj) + Fused Pre-RMSNorm & Sigmoid-Gated Output RMSNorm
      - Stage A4: + Cross-Layer MoBA Block-Bitmask Reuse (BlockShare across adjacent MoBA layers)
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_kv_heads: int = 16,
        head_dim: int = 64,
        chunk_size: int = 128,
        topk: int = 4,
        max_seq_len: int = 4096,
        stage: str = "A4",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.topk = topk
        self.stage = stage

        self.q_dim = n_heads * head_dim
        self.kv_dim = n_kv_heads * head_dim
        half_k = head_dim // 2

        self.pre_norm_weight = nn.Parameter(torch.ones(d_model))
        self.q_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.o_norm_weight = nn.Parameter(torch.ones(head_dim))

        self.q_proj = nn.Linear(d_model, self.q_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.q_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, d_model, bias=False)

        self.qkvg_proj = nn.Linear(d_model, 2 * self.q_dim + 2 * self.kv_dim, bias=False)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_k, dtype=torch.float32) / float(half_k)))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cache", freqs.cos().contiguous(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin().contiguous(), persistent=False)
        self.sync_packed_weights()

    @torch.no_grad()
    def sync_packed_weights(self):
        self.qkvg_proj.weight.copy_(
            torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.g_proj.weight],
                dim=0,
            )
        )

    def _eager_qk_norm_rope(self, x: torch.Tensor, w: torch.Tensor, T: int) -> torch.Tensor:
        xf = x.float()
        rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        xn = (xf * rstd) * w.float()
        half_k = self.head_dim // 2
        x0 = xn[..., :half_k]
        x1 = xn[..., half_k:]
        cos = self.cos_cache[:T].view(1, T, 1, half_k)
        sin = self.sin_cache[:T].view(1, T, 1, half_k)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat([y0, y1], dim=-1).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        shared_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_shared_masks: bool = False,
    ):
        B, T, D = x.shape
        if self.stage in ("A0", "A1", "A2"):
            xf = x.float()
            rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            h = (xf * rstd).to(x.dtype) * self.pre_norm_weight
            q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim)
            k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim)
            v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim)
            g = self.g_proj(h).view(B, T, self.n_heads, self.head_dim)
        else:
            h = FusedMoBARMSNormFunction.apply(x, self.pre_norm_weight, 1e-6)
            qkvg = self.qkvg_proj(h)
            s0 = self.q_dim
            s1 = s0 + self.kv_dim
            s2 = s1 + self.kv_dim
            q = qkvg[..., :s0].contiguous().view(B, T, self.n_heads, self.head_dim)
            k = qkvg[..., s0:s1].contiguous().view(B, T, self.n_kv_heads, self.head_dim)
            v = qkvg[..., s1:s2].contiguous().view(B, T, self.n_kv_heads, self.head_dim)
            g = qkvg[..., s2:].contiguous().view(B, T, self.n_heads, self.head_dim)

        if self.stage in ("A0", "A1"):
            q = self._eager_qk_norm_rope(q, self.q_norm_weight, T)
            k = self._eager_qk_norm_rope(k, self.k_norm_weight, T)
        else:
            q = FusedMoBAQKRMSNormRoPEFunction.apply(q, self.q_norm_weight, self.cos_cache, self.sin_cache, 1e-6)
            k = FusedMoBAQKRMSNormRoPEFunction.apply(k, self.k_norm_weight, self.cos_cache, self.sin_cache, 1e-6)

        if self.stage == "A0":
            from fla.ops.moba import parallel_moba
            cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=x.device)
            q_pack = q.reshape(1, B * T, self.n_heads, self.head_dim).contiguous()
            if self.n_kv_heads < self.n_heads:
                g_rep = self.n_heads // self.n_kv_heads
                k_exp = k.unsqueeze(3).expand(B, T, self.n_kv_heads, g_rep, self.head_dim).reshape(1, B * T, self.n_heads, self.head_dim).contiguous()
                v_exp = v.unsqueeze(3).expand(B, T, self.n_kv_heads, g_rep, self.head_dim).reshape(1, B * T, self.n_heads, self.head_dim).contiguous()
            else:
                k_exp = k.reshape(1, B * T, self.n_heads, self.head_dim).contiguous()
                v_exp = v.reshape(1, B * T, self.n_heads, self.head_dim).contiguous()
            o = parallel_moba(q_pack, k_exp, v_exp, cu_seqlens, max_seqlen=T, chunk_size=self.chunk_size, topk=self.topk).reshape(B, T, self.n_heads, self.head_dim)
            out_masks = None
        else:
            if shared_masks is not None and self.stage == "A4":
                o = parallel_moba_rsi(q, k, v, chunk_size=self.chunk_size, topk=self.topk, precomputed_masks=shared_masks)
                out_masks = shared_masks
            elif return_shared_masks and self.stage == "A4":
                o, out_masks = parallel_moba_rsi(q, k, v, chunk_size=self.chunk_size, topk=self.topk, return_masks=True)
            else:
                o = parallel_moba_rsi(q, k, v, chunk_size=self.chunk_size, topk=self.topk)
                out_masks = None

        if self.stage in ("A0", "A1", "A2"):
            of = o.float()
            rstd_o = torch.rsqrt(of.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            o_norm = ((of * rstd_o) * self.o_norm_weight.float() * torch.sigmoid(g.float())).to(x.dtype)
        else:
            o_norm = FusedMoBARMSNormSigmoidGatedFunction.apply(o, g, self.o_norm_weight, 1e-6)

        out = x + self.o_proj(o_norm.reshape(B, T, self.q_dim))
        if return_shared_masks:
            return out, out_masks
        return out


# ============================================================================
# 8. 8-GPU NVSwitch Context Parallelism (MoBA-CP = 8)
# ============================================================================

class _MoBAAllToAllSingleAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        x_c = x.contiguous()
        out = torch.empty_like(x_c)
        dist.all_to_all_single(out, x_c, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        g_c = grad_out.contiguous()
        grad_in = torch.empty_like(g_c)
        dist.all_to_all_single(grad_in, g_c, group=ctx.group)
        return grad_in, None


class _MoBAAllGatherSequenceAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        world_size = dist.get_world_size(group)
        x_c = x.contiguous()
        chunks = [torch.empty_like(x_c) for _ in range(world_size)]
        dist.all_gather(chunks, x_c, group=group)
        return torch.cat(chunks, dim=1)

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        group = ctx.group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        chunks = [c.contiguous() for c in grad_full.chunk(world_size, dim=1)]
        grad_local = torch.empty_like(chunks[rank])
        dist.reduce_scatter(grad_local, chunks, group=group)
        return grad_local, None


def moba_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    chunk_size: int = 256,
    topk: int = 4,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    B, T_local, HQ, K = q_local.shape
    HKV = k_local.shape[2]
    V = v_local.shape[3]
    HQ_loc = HQ // world_size
    HKV_loc = HKV // world_size
    T_total = T_local * world_size

    q_p = q_local.view(B, T_local, world_size, HQ_loc * K).permute(2, 0, 1, 3)
    k_p = k_local.view(B, T_local, world_size, HKV_loc * K).permute(2, 0, 1, 3)
    v_p = v_local.view(B, T_local, world_size, HKV_loc * V).permute(2, 0, 1, 3)
    packed_send = torch.cat([q_p, k_p, v_p], dim=-1).contiguous()
    packed_recv = _MoBAAllToAllSingleAutograd.apply(packed_send, group)

    c0 = HQ_loc * K
    c1 = c0 + HKV_loc * K
    recv_b = packed_recv.permute(1, 0, 2, 3).contiguous()
    q_u = recv_b[..., :c0].reshape(B, T_total, HQ_loc, K)
    k_u = recv_b[..., c0:c1].reshape(B, T_total, HKV_loc, K)
    v_u = recv_b[..., c1:].reshape(B, T_total, HKV_loc, V)

    o_u = parallel_moba_rsi(q_u, k_u, v_u, chunk_size=chunk_size, topk=topk)

    o_send = o_u.view(B, world_size, T_local, HQ_loc * V).permute(1, 0, 2, 3).contiguous()
    o_recv = _MoBAAllToAllSingleAutograd.apply(o_send, group)
    o_local = o_recv.permute(1, 2, 0, 3).contiguous().view(B, T_local, HQ, V)
    return o_local


def moba_cp8_ring_baseline(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    chunk_size: int = 256,
    topk: int = 4,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    from fla.ops.moba import parallel_moba
    rank = dist.get_rank(group)
    B, T_local, HQ, K = q_local.shape
    HKV = k_local.shape[2]
    V = v_local.shape[3]
    q_full = _MoBAAllGatherSequenceAutograd.apply(q_local, group)
    k_full = _MoBAAllGatherSequenceAutograd.apply(k_local, group)
    v_full = _MoBAAllGatherSequenceAutograd.apply(v_local, group)
    T_total = q_full.shape[1]
    if HKV < HQ:
        g_rep = HQ // HKV
        k_full = k_full.unsqueeze(3).expand(B, T_total, HKV, g_rep, K).reshape(B, T_total, HQ, K)
        v_full = v_full.unsqueeze(3).expand(B, T_total, HKV, g_rep, V).reshape(B, T_total, HQ, V)
    cu = torch.arange(0, (B + 1) * T_total, T_total, dtype=torch.int32, device=q_local.device)
    o_full = parallel_moba(
        q_full.reshape(1, B * T_total, HQ, K).contiguous(),
        k_full.reshape(1, B * T_total, HQ, K).contiguous(),
        v_full.reshape(1, B * T_total, HQ, V).contiguous(),
        cu,
        max_seqlen=T_total,
        chunk_size=chunk_size,
        topk=topk,
    ).reshape(B, T_total, HQ, V)
    return o_full[:, rank * T_local : (rank + 1) * T_local].contiguous()


# ============================================================================
# 9. SM80 Cubin Register, Spill, & Shared Memory Audit
# ============================================================================

MOBA_RSI_KERNEL_OBJECTS = [
    ("_moba_block_centroid_fwd_rsi_kernel", _moba_block_centroid_fwd_rsi_kernel),
    ("_moba_gate_topk_bitmask_rsi_kernel", _moba_gate_topk_bitmask_rsi_kernel),
    ("_moba_fwd_rsi_kernel", _moba_fwd_rsi_kernel),
    ("_moba_bwd_prep_delta_rsi_kernel", _moba_bwd_prep_delta_rsi_kernel),
    ("_moba_bwd_dq_rsi_kernel", _moba_bwd_dq_rsi_kernel),
    ("_moba_bwd_dkv_rsi_kernel", _moba_bwd_dkv_rsi_kernel),
    ("_moba_decode_gate_bitmask_rsi_kernel", _moba_decode_gate_bitmask_rsi_kernel),
    ("_moba_decode_readout_rsi_kernel", _moba_decode_readout_rsi_kernel),
    ("_moba_rmsnorm_fwd_rsi_kernel", _moba_rmsnorm_fwd_rsi_kernel),
    ("_moba_rmsnorm_bwd_rsi_kernel", _moba_rmsnorm_bwd_rsi_kernel),
    ("_moba_qk_rmsnorm_rope_fwd_rsi_kernel", _moba_qk_rmsnorm_rope_fwd_rsi_kernel),
    ("_moba_qk_rmsnorm_rope_bwd_rsi_kernel", _moba_qk_rmsnorm_rope_bwd_rsi_kernel),
    ("_moba_rmsnorm_sigmoid_gated_fwd_rsi_kernel", _moba_rmsnorm_sigmoid_gated_fwd_rsi_kernel),
    ("_moba_rmsnorm_sigmoid_gated_bwd_rsi_kernel", _moba_rmsnorm_sigmoid_gated_bwd_rsi_kernel),
]


def audit_moba_rsi_cubin_kernels() -> Dict:
    per_kernel = {}
    total_specs = 0
    spilling_specs = 0
    over168_specs = 0
    max_regs_global = 0
    max_spills_global = 0
    total_spills_global = 0
    max_smem_global = 0

    for name, kobj in MOBA_RSI_KERNEL_OBJECTS:
        caches = []
        if hasattr(kobj, "device_caches"):
            for dev_cache in kobj.device_caches.values():
                if isinstance(dev_cache, tuple) and len(dev_cache) > 0 and isinstance(dev_cache[0], dict):
                    caches.append(dev_cache[0])
                elif isinstance(dev_cache, dict):
                    caches.append(dev_cache)
        if hasattr(kobj, "cache") and isinstance(kobj.cache, dict):
            caches.append(kobj.cache)

        k_specs = 0
        k_spilling = 0
        k_over168 = 0
        k_max_regs = 0
        k_max_spills = 0
        k_total_spills = 0
        k_max_smem = 0

        seen = set()
        for cdict in caches:
            for key, compiled in cdict.items():
                if id(compiled) in seen:
                    continue
                seen.add(id(compiled))
                n_regs = int(getattr(compiled, "n_regs", 0) or 0)
                n_spills = int(getattr(compiled, "n_spills", 0) or 0)
                md = getattr(compiled, "metadata", None)
                smem = int(getattr(md, "shared", 0) or 0) if md is not None else 0
                k_specs += 1
                if n_spills > 0:
                    k_spilling += 1
                if n_regs > 168:
                    k_over168 += 1
                k_max_regs = max(k_max_regs, n_regs)
                k_max_spills = max(k_max_spills, n_spills)
                k_total_spills += n_spills
                k_max_smem = max(k_max_smem, smem)

        per_kernel[name] = {
            "specializations": k_specs,
            "spilling_specs": k_spilling,
            "over168_specs": k_over168,
            "max_regs": k_max_regs,
            "max_spills": k_max_spills,
            "total_spills": k_total_spills,
            "max_smem_bytes": k_max_smem,
        }
        total_specs += k_specs
        spilling_specs += k_spilling
        over168_specs += k_over168
        max_regs_global = max(max_regs_global, k_max_regs)
        max_spills_global = max(max_spills_global, k_max_spills)
        total_spills_global += k_total_spills
        max_smem_global = max(max_smem_global, k_max_smem)

    return {
        "kernel_count": len(MOBA_RSI_KERNEL_OBJECTS),
        "total_specializations": total_specs,
        "spilling_specializations": spilling_specs,
        "over168_specializations": over168_specs,
        "max_regs": max_regs_global,
        "max_spills": max_spills_global,
        "total_spills": total_spills_global,
        "max_smem_bytes": max_smem_global,
        "all_zero_spill": (spilling_specs == 0 and over168_specs == 0 and max_regs_global <= 168),
        "kernels": per_kernel,
    }
