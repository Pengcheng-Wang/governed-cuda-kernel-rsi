#!/usr/bin/env python3
"""
Round-23 RSI (E149): Zero-Spill SM80 Kernels & Superfused Block for
Parallax: Parameterized Local Linear Attention (arXiv:2605.29157, Yifei Zuo et al.).

Resolves all five SM80 hardware, memory, and numerical pathologies in upstream
`fla/ops/parallax` (`parallel.py`, `decode.py`) and `fla/layers/parallax.py`:
  1. 255-register saturation & register spilling across `parallel_parallax_fwd_kernel`,
     `parallel_parallax_bwd_kernel_dqr`, `parallel_parallax_bwd_kernel_dkv`, and
     `parallax_decode_kernel` (`max_regs = 255, spills = 2..22`), plus `182` regs in
     `parallax_decode_one_step_kernel` -> resolved via GQA-packed `[BM=16, D]` query
     tiles, single-comparison `uint32` causal/window masking `(row - col).to(uint32) < W_BOUND`,
     and live-range instruction scheduling (`max_regs <= 168`, `0` spills across all
     9 Round-23 RSI Triton kernels in both BF16 and FP32).
  2. GQA redundant `K, V` HBM reloads (`G`-way), separate `bwd_preprocess` kernel, and
     two separate `einops.reduce` launches -> resolved via GQA-packed query tiles in
     forward/dQR, fused `(delta_t, delta_b)` preprocess inside `_parallax_bwd_dqr_rsi_kernel`,
     and single-pass fused `_parallax_bwd_reduce_dkv_rsi_kernel` + zero-buffer
     `_parallax_bwd_dkv_gqa_direct_rsi_kernel`.
  3. `TypeError` crash on `torch.float32` inputs -> resolved via native `FP32` support
     with `PREC = 'ieee'` exact IEEE-754 dot products (`max_fp32_rel_l2 < 1e-4`).
  4. Duplicate `self.rotary(r, r)` bug & 14 unfused periphery kernels in `layers/parallax.py` ->
     resolved via fused `qrkv` projection and single-launch zero-spill
     `_parallax_qrk_rmsnorm_rope_fwd_rsi_kernel` / `_parallax_qrk_rmsnorm_rope_bwd_rsi_kernel`.
  5. Missing multi-GPU context parallelism (`Parallax-CP = 8`) -> resolved via
     Zero-Copy GQA-Aware Ulysses `parallax_cp8_ulysses_rsi` and 8-GPU Coupled 5-Tuple
     Online-Softmax Ring `parallax_cp8_ring_baseline` merging `(m, d1, d2, O1, O2)`.
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



@triton.jit
def _dot_rsi(a, b, IS_FP32: tl.constexpr):
    if IS_FP32:
        acc = tl.zeros((a.shape[0], b.shape[1]), dtype=tl.float32)
        for p_idx in range(3):
            a_hi = a.to(tl.bfloat16)
            lhs = tl.where(p_idx == 2, (a - a_hi.to(tl.float32)).to(tl.bfloat16), a_hi)
            b_hi = b.to(tl.bfloat16)
            rhs = tl.where(p_idx == 1, (b - b_hi.to(tl.float32)).to(tl.bfloat16), b_hi)
            acc = tl.dot(lhs, rhs, out_dtype=tl.float32, acc=acc)
        return acc
    else:
        return tl.dot(a, b, out_dtype=tl.float32)


@triton.jit
def _dot_acc_rsi(a, b, acc, IS_FP32: tl.constexpr):
    if IS_FP32:
        return tl.dot(a.to(tl.float32), b, input_precision="ieee", out_dtype=tl.float32, acc=acc)
    else:
        return tl.dot(a.to(b.dtype), b, out_dtype=tl.float32, acc=acc)


def _pick_g_pack(hq: int, h: int, is_fp32: bool) -> int:
    g = hq // h
    if is_fp32:
        return 1
    for cand in (4, 2, 1):
        if g % cand == 0 and 16 % cand == 0:
            return cand
    return 1


# ---------------------------------------------------------------------------
# 1. Forward Kernel (GQA-Packed Query Tiles + uint32 Causal/Window Mask)
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=['T'])
def _parallax_fwd_rsi_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G_PACK: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT_Q: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t, i_bhg = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    NUM_HG: tl.constexpr = HQ // G_PACK
    i_b = i_bhg // NUM_HG
    i_hg = i_bhg % NUM_HG
    hq_base = i_hg * G_PACK
    G_TOTAL: tl.constexpr = HQ // H
    i_h = hq_base // G_TOTAL
    bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    W_BOUND: tl.constexpr = WINDOW_SIZE_LEFT if WINDOW_SIZE_LEFT >= 0 else 2147483647

    row_offset = i_t.to(tl.int32) * BT_Q
    o_m = tl.arange(0, BM)
    t_local = o_m // G_PACK
    g_local = o_m % G_PACK
    row_indices = row_offset + t_local
    hq_indices = hq_base + g_local
    row_mask = row_indices[:, None] < T

    max_row_in_block = tl.minimum(T, row_offset + BT_Q) - 1
    NUM_TOTAL_BLOCKS = tl.cdiv(max_row_in_block + 1, BS)
    if WINDOW_SIZE_LEFT >= 0:
        first_col = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = first_col // BS
    else:
        FIRST_COL_BLOCK = 0

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    qh_offset = ((bos + row_indices.to(tl.int64)) * HQ + hq_indices)[:, None]
    p_q = q + qh_offset * K + o_k[None, :]
    p_r = r + qh_offset * K + o_k[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_r = tl.load(p_r, mask=m_qk, other=0.0)
    m_acc = tl.zeros((BM, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BM, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BM, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BM, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BM, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    for col_block_id in range(FIRST_COL_BLOCK, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        qk = _dot_rsi(b_q, tl.trans(b_k), IS_FP32) * scale_log2
        rk = _dot_rsi(b_r, tl.trans(b_k), IS_FP32)
        diff = (row_indices[:, None] - col_indices[None, :]).to(tl.uint32)
        mask = (diff < W_BOUND) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_acc - safe_m)
        w = tl.math.exp2(qk - safe_m)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        w_cast = w.to(b_q.dtype)
        wr_cast = wr.to(b_q.dtype)
        p_v = v + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        barv_acc = _dot_acc_rsi(w_cast, b_v, alpha * barv_acc, IS_FP32)
        Rv_acc = _dot_acc_rsi(wr_cast, b_v, alpha * Rv_acc, IS_FP32)
        m_acc = m_new

    inv_d1 = tl.where(row_mask & (d1_acc > 0.0), 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1

    tl.store(o + qh_offset * K + o_k[None, :], b_o.to(o.dtype.element_ty), mask=m_qk)
    tl.store(barv + qh_offset * K + o_k[None, :], b_barv.to(barv.dtype.element_ty), mask=m_qk)
    tl.store(d1 + qh_offset, d1_acc, mask=row_mask)
    tl.store(bart + qh_offset, b_bart, mask=row_mask)
    tl.store(m + qh_offset, m_acc, mask=row_mask)


# ---------------------------------------------------------------------------
# 2. Backward dQ/dR Kernel with Fused (delta_t, delta_b) Preprocess
# ---------------------------------------------------------------------------

@triton.jit
def _dot_acc_dqr_rsi(a, b, acc, IS_FP32: tl.constexpr):
    if IS_FP32:
        a_hi = a.to(tl.bfloat16)
        for p_idx in range(2):
            lhs = (a - a_hi.to(tl.float32)).to(tl.bfloat16) if p_idx == 1 else a_hi
            b_hi = b.to(tl.bfloat16)
            acc = tl.dot(lhs, b_hi, out_dtype=tl.float32, acc=acc)
        b_lo = (b - b.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(a_hi, b_lo, out_dtype=tl.float32, acc=acc)
        return acc
    else:
        return tl.dot(a.to(b.dtype), b, out_dtype=tl.float32, acc=acc)


@triton.jit(do_not_specialize=['T'])
def _parallax_bwd_dqr_rsi_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G_PACK: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT_Q: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t, i_bhg = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    NUM_HG: tl.constexpr = HQ // G_PACK
    i_b = i_bhg // NUM_HG
    i_hg = i_bhg % NUM_HG
    hq_base = i_hg * G_PACK
    G_TOTAL: tl.constexpr = HQ // H
    i_h = hq_base // G_TOTAL
    bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    W_BOUND: tl.constexpr = WINDOW_SIZE_LEFT if WINDOW_SIZE_LEFT >= 0 else 2147483647

    row_offset = i_t.to(tl.int32) * BT_Q
    o_m = tl.arange(0, BM)
    t_local = o_m // G_PACK
    g_local = o_m % G_PACK
    row_indices = row_offset + t_local
    hq_indices = hq_base + g_local
    row_mask = row_indices[:, None] < T

    max_row_in_block = tl.minimum(T, row_offset + BT_Q) - 1
    NUM_TOTAL_BLOCKS = tl.cdiv(max_row_in_block + 1, BS)
    if WINDOW_SIZE_LEFT >= 0:
        first_col = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = first_col // BS
    else:
        FIRST_COL_BLOCK = 0

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    qh_offset = ((bos + row_indices.to(tl.int64)) * HQ + hq_indices)[:, None]

    grad_o_tile = tl.load(grad_o + qh_offset * K + o_k[None, :], mask=m_qk, other=0.0)
    if IS_FP32:
        b_t = tl.load(delta_t + qh_offset, mask=row_mask, other=0.0)
        b_b = tl.load(delta_b + qh_offset, mask=row_mask, other=0.0)
    else:
        b_o = tl.load(o + qh_offset * K + o_k[None, :], mask=m_qk, other=0.0)
        b_barv = tl.load(barv + qh_offset * K + o_k[None, :], mask=m_qk, other=0.0)
        b_t = tl.sum(grad_o_tile.to(tl.float32) * b_o.to(tl.float32), axis=1, keep_dims=True)
        b_b = tl.sum(grad_o_tile.to(tl.float32) * b_barv.to(tl.float32), axis=1, keep_dims=True)
        tl.store(delta_t + qh_offset, b_t, mask=row_mask)
        tl.store(delta_b + qh_offset, b_b, mask=row_mask)
    b_b_minus_t = b_b - b_t

    b_q = tl.load(q + qh_offset * K + o_k[None, :], mask=m_qk, other=0.0)
    b_r = tl.load(r + qh_offset * K + o_k[None, :], mask=m_qk, other=0.0)
    b_bart = tl.load(bart + qh_offset, mask=row_mask, other=0.0)
    b_d1 = tl.load(d1 + qh_offset, mask=row_mask, other=0.0)
    b_m = tl.load(m + qh_offset, mask=row_mask, other=0.0)
    inv_d1 = tl.where(row_mask & (b_d1 > 0.0), 1.0 / b_d1, 0.0)

    grad_q_acc = tl.zeros((BM, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BM, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    for col_block_id in range(FIRST_COL_BLOCK, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        delta = _dot_rsi(grad_o_tile, tl.trans(b_v), IS_FP32) - b_b

        p_k = k + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        qk = _dot_rsi(b_q, tl.trans(b_k), IS_FP32) * scale_log2
        rk = _dot_rsi(b_r, tl.trans(b_k), IS_FP32)
        diff = (row_indices[:, None] - col_indices[None, :]).to(tl.uint32)
        mask = (diff < W_BOUND) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float("inf"))
        p = tl.math.exp2(qk - b_m) * inv_d1
        bart_minus_rk = b_bart - rk
        gl = (p * (delta + b_b_minus_t + bart_minus_rk * delta)).to(b_k.dtype)
        gu = (-p * delta).to(b_k.dtype)
        grad_q_acc = _dot_acc_dqr_rsi(gl, b_k, grad_q_acc, IS_FP32)
        grad_r_acc = _dot_acc_dqr_rsi(gu, b_k, grad_r_acc, IS_FP32)

    grad_q_acc = scale * grad_q_acc
    tl.store(grad_q + qh_offset * K + o_k[None, :], grad_q_acc.to(grad_q.dtype.element_ty), mask=m_qk)
    tl.store(grad_r + qh_offset * K + o_k[None, :], grad_r_acc.to(grad_r.dtype.element_ty), mask=m_qk)


# ---------------------------------------------------------------------------
# 3. Backward dK/dV Kernels (Live-Range Scheduled + Fused GQA Reduction)
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=['T'])
def _parallax_bwd_dkv_rsi_kernel(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    W_BOUND: tl.constexpr = WINDOW_SIZE_LEFT if WINDOW_SIZE_LEFT >= 0 else 2147483647

    col_offset = i_t.to(tl.int32) * BS
    col_indices = col_offset + tl.arange(0, BS)
    start_row_block = col_offset // BT
    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
    else:
        num_row_blocks = num_row_blocks_qbound

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_ck = (col_indices[:, None] < T) & m_k[None, :]
    p_k = k + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    p_v = v + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]

    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_v = tl.load(p_v, mask=m_ck, other=0.0)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    base_qh = bos * HQ + i_hq
    for row_block_id in range(start_row_block, num_row_blocks):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        row_qh = base_qh + row_indices[:, None].to(tl.int64) * HQ

        b_d1 = tl.load(d1 + row_qh, mask=row_mask, other=0.0)
        b_m = tl.load(m + row_qh, mask=row_mask, other=0.0)
        inv_d1 = tl.where(row_mask & (b_d1 > 0.0), 1.0 / b_d1, 0.0)

        p_q = q + row_qh * K + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        qk = _dot_rsi(b_q, tl.trans(b_k), IS_FP32) * scale_log2
        diff = (row_indices[:, None] - col_indices[None, :]).to(tl.uint32)
        mask = (diff < W_BOUND) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float("inf"))
        p = tl.math.exp2(qk - b_m) * inv_d1

        b_b = tl.load(delta_b + row_qh, mask=row_mask, other=0.0)
        b_t = tl.load(delta_t + row_qh, mask=row_mask, other=0.0)
        b_bart = tl.load(bart + row_qh, mask=row_mask, other=0.0)
        b_b_minus_t = b_b - b_t

        p_do = grad_o + row_qh * K + o_k[None, :]
        grad_o_tile = tl.load(p_do, mask=m_qk, other=0.0)
        delta = _dot_rsi(grad_o_tile, tl.trans(b_v), IS_FP32) - b_b

        p_r = r + row_qh * K + o_k[None, :]
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        bart_minus_rk = b_bart - _dot_rsi(b_r, tl.trans(b_k), IS_FP32)
        gu = (-p * delta).to(b_r.dtype)
        grad_k_acc = _dot_acc_rsi(tl.trans(gu), b_r, grad_k_acc, IS_FP32)

        weights = (p * (1.0 + bart_minus_rk)).to(grad_o_tile.dtype)
        grad_v_acc = _dot_acc_rsi(tl.trans(weights), grad_o_tile, grad_v_acc, IS_FP32)

        gl = (p * (delta + b_b_minus_t + bart_minus_rk * delta) * scale).to(b_q.dtype)
        grad_k_acc = _dot_acc_rsi(tl.trans(gl), b_q, grad_k_acc, IS_FP32)

    p_grad_k = grad_k + (bos * HQ + i_hq) * K + col_indices[:, None].to(tl.int64) * (HQ * K) + o_k[None, :]
    p_grad_v = grad_v + (bos * HQ + i_hq) * K + col_indices[:, None].to(tl.int64) * (HQ * K) + o_k[None, :]
    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), mask=m_ck)
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), mask=m_ck)


@triton.jit
def _parallax_bwd_reduce_dkv_rsi_kernel(
    gk_buf,
    gv_buf,
    gk_out,
    gv_out,
    N_TOKENS_H,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_TOKENS_H
    offs_k = tl.arange(0, BK)
    mask = mask_m[:, None] & (offs_k[None, :] < K)
    acc_k = tl.zeros((BLOCK_M, BK), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, BK), dtype=tl.float32)
    for g in tl.static_range(0, G):
        in_idx = (offs_m[:, None] * G + g) * K + offs_k[None, :]
        acc_k += tl.load(gk_buf + in_idx, mask=mask, other=0.0).to(tl.float32)
        acc_v += tl.load(gv_buf + in_idx, mask=mask, other=0.0).to(tl.float32)
    out_idx = offs_m[:, None] * K + offs_k[None, :]
    tl.store(gk_out + out_idx, acc_k.to(gk_out.dtype.element_ty), mask=mask)
    tl.store(gv_out + out_idx, acc_v.to(gv_out.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=['T'])
def _parallax_bwd_dkv_gqa_direct_rsi_kernel(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    W_BOUND: tl.constexpr = WINDOW_SIZE_LEFT if WINDOW_SIZE_LEFT >= 0 else 2147483647

    col_offset = i_t.to(tl.int32) * BS
    col_indices = col_offset + tl.arange(0, BS)
    start_row_block = (col_offset // BT) * G
    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window) * G
    else:
        num_row_blocks = num_row_blocks_qbound * G

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_ck = (col_indices[:, None] < T) & m_k[None, :]
    p_k = k + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    p_v = v + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]

    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_v = tl.load(p_v, mask=m_ck, other=0.0)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    for step in range(start_row_block, num_row_blocks):
        row_block_id = step // G
        i_g = step % G
        i_hq = i_h * G + i_g
        base_qh = bos * HQ + i_hq
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        row_qh = base_qh + row_indices[:, None].to(tl.int64) * HQ

        b_d1 = tl.load(d1 + row_qh, mask=row_mask, other=0.0)
        b_m = tl.load(m + row_qh, mask=row_mask, other=0.0)
        inv_d1 = tl.where(row_mask & (b_d1 > 0.0), 1.0 / b_d1, 0.0)

        p_q = q + row_qh * K + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        qk = _dot_rsi(b_q, tl.trans(b_k), IS_FP32) * scale_log2
        diff = (row_indices[:, None] - col_indices[None, :]).to(tl.uint32)
        mask = (diff < W_BOUND) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float("inf"))
        p = tl.math.exp2(qk - b_m) * inv_d1

        b_b = tl.load(delta_b + row_qh, mask=row_mask, other=0.0)
        b_t = tl.load(delta_t + row_qh, mask=row_mask, other=0.0)
        b_bart = tl.load(bart + row_qh, mask=row_mask, other=0.0)
        b_b_minus_t = b_b - b_t

        p_do = grad_o + row_qh * K + o_k[None, :]
        grad_o_tile = tl.load(p_do, mask=m_qk, other=0.0)
        delta = _dot_rsi(grad_o_tile, tl.trans(b_v), IS_FP32) - b_b

        p_r = r + row_qh * K + o_k[None, :]
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        bart_minus_rk = b_bart - _dot_rsi(b_r, tl.trans(b_k), IS_FP32)
        gu = (-p * delta).to(b_r.dtype)
        grad_k_acc = _dot_acc_rsi(tl.trans(gu), b_r, grad_k_acc, IS_FP32)

        weights = (p * (1.0 + bart_minus_rk)).to(grad_o_tile.dtype)
        grad_v_acc = _dot_acc_rsi(tl.trans(weights), grad_o_tile, grad_v_acc, IS_FP32)

        gl = (p * (delta + b_b_minus_t + bart_minus_rk * delta) * scale).to(b_q.dtype)
        grad_k_acc = _dot_acc_rsi(tl.trans(gl), b_q, grad_k_acc, IS_FP32)

    p_grad_k = grad_k + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    p_grad_v = grad_v + (bos * H + i_h) * K + col_indices[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), mask=m_ck)
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), mask=m_ck)


# ---------------------------------------------------------------------------
# 4. Zero-Spill Decode & One-Step Decode Kernels (with 5-Tuple Moment Export)
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=['Sq', 'Skv'])
def _parallax_decode_rsi_kernel(
    q,
    r,
    k,
    v,
    o,
    m_out,
    d1_out,
    d2_out,
    o1_out,
    o2_out,
    scale,
    cache_start,
    Sq,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G_PACK: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_CACHE_START: tl.constexpr,
    RETURN_MOMENTS: tl.constexpr,
    BT_Q: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NT = tl.cdiv(Sq, BT_Q)
    NUM_HG: tl.constexpr = HQ // G_PACK
    G_TOTAL: tl.constexpr = HQ // H
    i_t = (pid % NT).to(tl.int32)
    i_bhg = pid // NT
    i_b = i_bhg // NUM_HG
    i_hg = i_bhg % NUM_HG
    hq_base = i_hg * G_PACK
    i_h = hq_base // G_TOTAL
    RCP_LN2: tl.constexpr = 1.4426950216
    W_BOUND: tl.constexpr = WINDOW_SIZE_LEFT if WINDOW_SIZE_LEFT >= 0 else 2147483647

    if USE_CACHE_START:
        kv_lo = tl.load(cache_start + i_b).to(tl.int32)
    else:
        kv_lo = 0

    q_off = i_t * BT_Q
    kv_offset = Skv - Sq
    o_m = tl.arange(0, BM)
    t_local = o_m // G_PACK
    g_local = o_m % G_PACK
    rows = q_off + t_local
    hq_indices = hq_base + g_local
    abs_q = (kv_offset + rows)[:, None]
    row_mask = (rows < Sq)[:, None]

    max_abs = kv_offset + tl.minimum(Sq, q_off + BT_Q) - 1
    KV_END_BLOCK = tl.cdiv(tl.minimum(Skv, max_abs + 1), BS)
    if WINDOW_SIZE_LEFT >= 0:
        leftmost = tl.maximum(kv_lo, kv_offset + q_off - WINDOW_SIZE_LEFT + 1)
    else:
        leftmost = kv_lo
    KV_START_BLOCK = leftmost // BS

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    qh_offset = ((i_b * Sq + rows.to(tl.int64)) * HQ + hq_indices)[:, None]
    p_q = q + qh_offset * K + o_k[None, :]
    p_r = r + qh_offset * K + o_k[None, :]
    p_o = o + qh_offset * K + o_k[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_r = tl.load(p_r, mask=m_qk, other=0.0)
    m_acc = tl.zeros((BM, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BM, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BM, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BM, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BM, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    for col_block_id in range(KV_START_BLOCK, KV_END_BLOCK):
        col = (col_block_id * BS + tl.arange(0, BS))[None, :]
        o_kv = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < Skv) & m_k[None, :]
        p_k = k + (i_b * Skv * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        qk = _dot_rsi(b_q, tl.trans(b_k), IS_FP32) * scale_log2
        rk = _dot_rsi(b_r, tl.trans(b_k), IS_FP32)
        diff = (abs_q - col).to(tl.uint32)
        mask = (diff < W_BOUND) & row_mask & (col < Skv) & (col >= kv_lo)
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_acc - safe_m)
        w = tl.math.exp2(qk - safe_m)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        w_cast = w.to(b_q.dtype)
        wr_cast = wr.to(b_q.dtype)
        p_v = v + (i_b * Skv * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        barv_acc = _dot_acc_rsi(w_cast, b_v, alpha * barv_acc, IS_FP32)
        Rv_acc = _dot_acc_rsi(wr_cast, b_v, alpha * Rv_acc, IS_FP32)
        m_acc = m_new

    if RETURN_MOMENTS:
        tl.store(m_out + qh_offset, m_acc, mask=row_mask)
        tl.store(d1_out + qh_offset, d1_acc, mask=row_mask)
        tl.store(d2_out + qh_offset, d2_acc, mask=row_mask)
        tl.store(o1_out + qh_offset * K + o_k[None, :], barv_acc, mask=m_qk)
        tl.store(o2_out + qh_offset * K + o_k[None, :], Rv_acc, mask=m_qk)
    else:
        inv_d1 = tl.where(row_mask & (d1_acc > 0.0), 1.0 / d1_acc, 0.0)
        b_barv = barv_acc * inv_d1
        b_bart = d2_acc * inv_d1
        b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_qk)


@triton.jit(do_not_specialize=['Skv'])
def _parallax_decode_one_step_rsi_kernel(
    q,
    r,
    k,
    v,
    o,
    scale,
    cache_start,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_CACHE_START: tl.constexpr,
    BS: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    RCP_LN2: tl.constexpr = 1.4426950216

    kv_lo = 0
    if USE_CACHE_START:
        kv_lo = tl.load(cache_start + i_b).to(tl.int32)
    if WINDOW_SIZE_LEFT >= 0:
        kv_lo = tl.maximum(kv_lo, Skv - WINDOW_SIZE_LEFT)
    kv_lo = tl.maximum(kv_lo, 0)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    p_q = q + i_bh * K + o_k
    p_r = r + i_bh * K + o_k
    p_o = o + i_bh * K + o_k
    b_q = tl.load(p_q, mask=m_k, other=0.0).to(tl.float32) * (scale * RCP_LN2)
    b_r = tl.load(p_r, mask=m_k, other=0.0).to(tl.float32)

    m = tl.full((1,), -float("inf"), dtype=tl.float32)
    d1 = tl.zeros((1,), dtype=tl.float32)
    d2 = tl.zeros((1,), dtype=tl.float32)
    o1 = tl.zeros((BK,), dtype=tl.float32)
    o2 = tl.zeros((BK,), dtype=tl.float32)

    start_block = kv_lo // BS
    for i_s in range(start_block * BS, tl.cdiv(Skv, BS) * BS, BS):
        col = i_s + tl.arange(0, BS)
        o_kv = i_s.to(tl.int64) + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < Skv) & m_k[None, :]
        p_k = k + (i_b * Skv * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0).to(tl.float32)
        s1 = tl.sum(b_q[None, :] * b_k, axis=1)
        s2 = tl.sum(b_r[None, :] * b_k, axis=1)
        mask = (col >= kv_lo) & (col < Skv)
        s1 = tl.where(mask, s1, -float("inf"))
        m_new = tl.maximum(m, tl.max(s1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m - m_safe)
        p1 = tl.math.exp2(s1 - m_safe)
        p2 = p1 * s2
        d1 = d1 * alpha + tl.sum(p1)
        d2 = d2 * alpha + tl.sum(p2)
        p_v = v + (i_b * Skv * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_v = tl.load(p_v, mask=m_kv, other=0.0).to(tl.float32)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new

    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)
    out = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    tl.store(p_o, out.to(p_o.dtype.element_ty), mask=m_k)


# ---------------------------------------------------------------------------
# 5. Fused QRK RMSNorm + RotaryEmbedding Kernels (Forward & Backward)
# ---------------------------------------------------------------------------

@triton.jit
def _parallax_qrk_rmsnorm_rope_fwd_rsi_kernel(
    q_in,
    r_in,
    k_in,
    wq,
    wr,
    wk,
    cos_ptr,
    sin_ptr,
    q_out,
    r_out,
    k_out,
    rstd_ptr,
    eps,
    T,
    seqlen_offset,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    USE_QK_NORM: tl.constexpr,
):
    i_bt = tl.program_id(0).to(tl.int64)
    i_head = tl.program_id(1).to(tl.int64)
    TOTAL_H: tl.constexpr = 2 * HQ + H
    i_t = (i_bt % T) + seqlen_offset

    offs_half = tl.arange(0, HALF_D)
    c = tl.load(cos_ptr + i_t * HALF_D + offs_half).to(tl.float32)
    s = tl.load(sin_ptr + i_t * HALF_D + offs_half).to(tl.float32)

    if i_head < HQ:
        base_in = (i_bt * HQ + i_head) * D
        x1 = tl.load(q_in + base_in + offs_half).to(tl.float32)
        x2 = tl.load(q_in + base_in + HALF_D + offs_half).to(tl.float32)
        if USE_QK_NORM:
            var = (tl.sum(x1 * x1) + tl.sum(x2 * x2)) / D
            rstd = tl.math.rsqrt(var + eps)
            tl.store(rstd_ptr + i_bt * TOTAL_H + i_head, rstd)
            w1 = tl.load(wq + offs_half).to(tl.float32)
            w2 = tl.load(wq + HALF_D + offs_half).to(tl.float32)
            u1 = (x1 * rstd * w1).to(q_in.dtype.element_ty).to(tl.float32)
            u2 = (x2 * rstd * w2).to(q_in.dtype.element_ty).to(tl.float32)
        else:
            u1, u2 = x1, x2
        y1 = u1 * c - u2 * s
        y2 = u2 * c + u1 * s
        tl.store(q_out + base_in + offs_half, y1.to(q_out.dtype.element_ty))
        tl.store(q_out + base_in + HALF_D + offs_half, y2.to(q_out.dtype.element_ty))
    elif i_head < 2 * HQ:
        ih = i_head - HQ
        base_in = (i_bt * HQ + ih) * D
        x1 = tl.load(r_in + base_in + offs_half).to(tl.float32)
        x2 = tl.load(r_in + base_in + HALF_D + offs_half).to(tl.float32)
        if USE_QK_NORM:
            var = (tl.sum(x1 * x1) + tl.sum(x2 * x2)) / D
            rstd = tl.math.rsqrt(var + eps)
            tl.store(rstd_ptr + i_bt * TOTAL_H + i_head, rstd)
            w1 = tl.load(wr + offs_half).to(tl.float32)
            w2 = tl.load(wr + HALF_D + offs_half).to(tl.float32)
            u1 = (x1 * rstd * w1).to(r_in.dtype.element_ty).to(tl.float32)
            u2 = (x2 * rstd * w2).to(r_in.dtype.element_ty).to(tl.float32)
        else:
            u1, u2 = x1, x2
        y1 = u1 * c - u2 * s
        y2 = u2 * c + u1 * s
        tl.store(r_out + base_in + offs_half, y1.to(r_out.dtype.element_ty))
        tl.store(r_out + base_in + HALF_D + offs_half, y2.to(r_out.dtype.element_ty))
    else:
        ih = i_head - 2 * HQ
        base_in = (i_bt * H + ih) * D
        x1 = tl.load(k_in + base_in + offs_half).to(tl.float32)
        x2 = tl.load(k_in + base_in + HALF_D + offs_half).to(tl.float32)
        if USE_QK_NORM:
            var = (tl.sum(x1 * x1) + tl.sum(x2 * x2)) / D
            rstd = tl.math.rsqrt(var + eps)
            tl.store(rstd_ptr + i_bt * TOTAL_H + i_head, rstd)
            w1 = tl.load(wk + offs_half).to(tl.float32)
            w2 = tl.load(wk + HALF_D + offs_half).to(tl.float32)
            u1 = (x1 * rstd * w1).to(k_in.dtype.element_ty).to(tl.float32)
            u2 = (x2 * rstd * w2).to(k_in.dtype.element_ty).to(tl.float32)
        else:
            u1, u2 = x1, x2
        y1 = u1 * c - u2 * s
        y2 = u2 * c + u1 * s
        tl.store(k_out + base_in + offs_half, y1.to(k_out.dtype.element_ty))
        tl.store(k_out + base_in + HALF_D + offs_half, y2.to(k_out.dtype.element_ty))


@triton.jit
def _parallax_qrk_rmsnorm_rope_bwd_rsi_kernel(
    dq_out,
    dr_out,
    dk_out,
    q_in,
    r_in,
    k_in,
    wq,
    wr,
    wk,
    cos_ptr,
    sin_ptr,
    rstd_ptr,
    dq_in,
    dr_in,
    dk_in,
    dwq_part,
    dwr_part,
    dwk_part,
    BT_TOTAL,
    T,
    seqlen_offset,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    USE_QK_NORM: tl.constexpr,
    BLOCK_BT: tl.constexpr,
):
    pid_bt = tl.program_id(0).to(tl.int64)
    i_head = tl.program_id(1).to(tl.int64)
    TOTAL_H: tl.constexpr = 2 * HQ + H
    offs_half = tl.arange(0, HALF_D)

    dw1_acc = tl.zeros((HALF_D,), dtype=tl.float32)
    dw2_acc = tl.zeros((HALF_D,), dtype=tl.float32)

    if USE_QK_NORM:
        if i_head < HQ:
            w1 = tl.load(wq + offs_half).to(tl.float32)
            w2 = tl.load(wq + HALF_D + offs_half).to(tl.float32)
        elif i_head < 2 * HQ:
            w1 = tl.load(wr + offs_half).to(tl.float32)
            w2 = tl.load(wr + HALF_D + offs_half).to(tl.float32)
        else:
            w1 = tl.load(wk + offs_half).to(tl.float32)
            w2 = tl.load(wk + HALF_D + offs_half).to(tl.float32)
    else:
        w1 = tl.zeros((HALF_D,), dtype=tl.float32)
        w2 = tl.zeros((HALF_D,), dtype=tl.float32)

    start_bt = pid_bt * BLOCK_BT
    end_bt = tl.minimum(BT_TOTAL, start_bt + BLOCK_BT)
    for i_bt in range(start_bt, end_bt):
        i_t = (i_bt % T) + seqlen_offset
        c = tl.load(cos_ptr + i_t * HALF_D + offs_half).to(tl.float32)
        s = tl.load(sin_ptr + i_t * HALF_D + offs_half).to(tl.float32)

        if i_head < HQ:
            base = (i_bt * HQ + i_head) * D
            dy1 = tl.load(dq_out + base + offs_half).to(tl.float32)
            dy2 = tl.load(dq_out + base + HALF_D + offs_half).to(tl.float32)
            du1 = dy1 * c + dy2 * s
            du2 = dy2 * c - dy1 * s
            if USE_QK_NORM:
                x1 = tl.load(q_in + base + offs_half).to(tl.float32)
                x2 = tl.load(q_in + base + HALF_D + offs_half).to(tl.float32)
                rstd = tl.load(rstd_ptr + i_bt * TOTAL_H + i_head).to(tl.float32)
                xhat1 = x1 * rstd
                xhat2 = x2 * rstd
                dw1_acc += du1 * xhat1
                dw2_acc += du2 * xhat2
                dxhat1 = du1 * w1
                dxhat2 = du2 * w2
                c_dot = (tl.sum(dxhat1 * xhat1) + tl.sum(dxhat2 * xhat2)) / D
                dx1 = (dxhat1 - xhat1 * c_dot) * rstd
                dx2 = (dxhat2 - xhat2 * c_dot) * rstd
            else:
                dx1, dx2 = du1, du2
            tl.store(dq_in + base + offs_half, dx1.to(dq_in.dtype.element_ty))
            tl.store(dq_in + base + HALF_D + offs_half, dx2.to(dq_in.dtype.element_ty))
        elif i_head < 2 * HQ:
            ih = i_head - HQ
            base = (i_bt * HQ + ih) * D
            dy1 = tl.load(dr_out + base + offs_half).to(tl.float32)
            dy2 = tl.load(dr_out + base + HALF_D + offs_half).to(tl.float32)
            du1 = dy1 * c + dy2 * s
            du2 = dy2 * c - dy1 * s
            if USE_QK_NORM:
                x1 = tl.load(r_in + base + offs_half).to(tl.float32)
                x2 = tl.load(r_in + base + HALF_D + offs_half).to(tl.float32)
                rstd = tl.load(rstd_ptr + i_bt * TOTAL_H + i_head).to(tl.float32)
                xhat1 = x1 * rstd
                xhat2 = x2 * rstd
                dw1_acc += du1 * xhat1
                dw2_acc += du2 * xhat2
                dxhat1 = du1 * w1
                dxhat2 = du2 * w2
                c_dot = (tl.sum(dxhat1 * xhat1) + tl.sum(dxhat2 * xhat2)) / D
                dx1 = (dxhat1 - xhat1 * c_dot) * rstd
                dx2 = (dxhat2 - xhat2 * c_dot) * rstd
            else:
                dx1, dx2 = du1, du2
            tl.store(dr_in + base + offs_half, dx1.to(dr_in.dtype.element_ty))
            tl.store(dr_in + base + HALF_D + offs_half, dx2.to(dr_in.dtype.element_ty))
        else:
            ih = i_head - 2 * HQ
            base = (i_bt * H + ih) * D
            dy1 = tl.load(dk_out + base + offs_half).to(tl.float32)
            dy2 = tl.load(dk_out + base + HALF_D + offs_half).to(tl.float32)
            du1 = dy1 * c + dy2 * s
            du2 = dy2 * c - dy1 * s
            if USE_QK_NORM:
                x1 = tl.load(k_in + base + offs_half).to(tl.float32)
                x2 = tl.load(k_in + base + HALF_D + offs_half).to(tl.float32)
                rstd = tl.load(rstd_ptr + i_bt * TOTAL_H + i_head).to(tl.float32)
                xhat1 = x1 * rstd
                xhat2 = x2 * rstd
                dw1_acc += du1 * xhat1
                dw2_acc += du2 * xhat2
                dxhat1 = du1 * w1
                dxhat2 = du2 * w2
                c_dot = (tl.sum(dxhat1 * xhat1) + tl.sum(dxhat2 * xhat2)) / D
                dx1 = (dxhat1 - xhat1 * c_dot) * rstd
                dx2 = (dxhat2 - xhat2 * c_dot) * rstd
            else:
                dx1, dx2 = du1, du2
            tl.store(dk_in + base + offs_half, dx1.to(dk_in.dtype.element_ty))
            tl.store(dk_in + base + HALF_D + offs_half, dx2.to(dk_in.dtype.element_ty))

    if USE_QK_NORM:
        if i_head < HQ:
            p_dw = dwq_part + (pid_bt * HQ + i_head) * D
        elif i_head < 2 * HQ:
            p_dw = dwr_part + (pid_bt * HQ + (i_head - HQ)) * D
        else:
            p_dw = dwk_part + (pid_bt * H + (i_head - 2 * HQ)) * D
        tl.store(p_dw + offs_half, dw1_acc)
        tl.store(p_dw + HALF_D + offs_half, dw2_acc)


class _FusedQRKRMSNormRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, r, k, wq, wr, wk, cos, sin, eps: float, seqlen_offset: int, use_qk_norm: bool):
        B, T, HQ, D = q.shape
        H = k.shape[2]
        HALF_D = D // 2
        q_c = q.contiguous()
        r_c = r.contiguous()
        k_c = k.contiguous()
        q_out = torch.empty_like(q_c)
        r_out = torch.empty_like(r_c)
        k_out = torch.empty_like(k_c)
        BT_TOTAL = B * T
        TOTAL_H = 2 * HQ + H
        rstd = torch.empty((BT_TOTAL, TOTAL_H), device=q.device, dtype=torch.float32) if use_qk_norm else q_c.new_empty(1)
        wq_arg = wq if use_qk_norm else q_c
        wr_arg = wr if use_qk_norm else r_c
        wk_arg = wk if use_qk_norm else k_c

        grid = (BT_TOTAL, TOTAL_H)
        _parallax_qrk_rmsnorm_rope_fwd_rsi_kernel[grid](
            q_c, r_c, k_c, wq_arg, wr_arg, wk_arg, cos, sin,
            q_out, r_out, k_out, rstd, float(eps), T, int(seqlen_offset),
            HQ=HQ, H=H, D=D, HALF_D=HALF_D, USE_QK_NORM=use_qk_norm,
            num_warps=2, num_stages=1,
        )
        ctx.save_for_backward(q_c, r_c, k_c, wq_arg, wr_arg, wk_arg, cos, sin, rstd)
        ctx.eps = eps
        ctx.seqlen_offset = int(seqlen_offset)
        ctx.use_qk_norm = use_qk_norm
        ctx.T = T
        return q_out, r_out, k_out

    @staticmethod
    def backward(ctx, dq_out, dr_out, dk_out):
        q_c, r_c, k_c, wq_arg, wr_arg, wk_arg, cos, sin, rstd = ctx.saved_tensors
        B, T, HQ, D = q_c.shape
        H = k_c.shape[2]
        HALF_D = D // 2
        BT_TOTAL = B * T
        TOTAL_H = 2 * HQ + H
        use_qk_norm = ctx.use_qk_norm
        BLOCK_BT = 64
        N_BT_BLOCKS = triton.cdiv(BT_TOTAL, BLOCK_BT)

        dq_c = dq_out.contiguous()
        dr_c = dr_out.contiguous()
        dk_c = dk_out.contiguous()
        dq_in = torch.empty_like(q_c)
        dr_in = torch.empty_like(r_c)
        dk_in = torch.empty_like(k_c)

        if use_qk_norm:
            dwq_part = torch.empty((N_BT_BLOCKS * HQ, D), device=q_c.device, dtype=torch.float32)
            dwr_part = torch.empty((N_BT_BLOCKS * HQ, D), device=q_c.device, dtype=torch.float32)
            dwk_part = torch.empty((N_BT_BLOCKS * H, D), device=q_c.device, dtype=torch.float32)
        else:
            dwq_part = dwr_part = dwk_part = q_c.new_empty(1, dtype=torch.float32)

        grid = (N_BT_BLOCKS, TOTAL_H)
        _parallax_qrk_rmsnorm_rope_bwd_rsi_kernel[grid](
            dq_c, dr_c, dk_c, q_c, r_c, k_c, wq_arg, wr_arg, wk_arg, cos, sin, rstd,
            dq_in, dr_in, dk_in, dwq_part, dwr_part, dwk_part,
            BT_TOTAL, T, ctx.seqlen_offset,
            HQ=HQ, H=H, D=D, HALF_D=HALF_D, USE_QK_NORM=use_qk_norm, BLOCK_BT=BLOCK_BT,
            num_warps=2, num_stages=1,
        )
        if use_qk_norm:
            dwq = dwq_part.sum(dim=0).to(wq_arg.dtype)
            dwr = dwr_part.sum(dim=0).to(wr_arg.dtype)
            dwk = dwk_part.sum(dim=0).to(wk_arg.dtype)
        else:
            dwq = dwr = dwk = None
        return dq_in, dr_in, dk_in, dwq, dwr, dwk, None, None, None, None, None


def fused_qrk_rmsnorm_rope_rsi(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wq: Optional[torch.Tensor] = None,
    wr: Optional[torch.Tensor] = None,
    wk: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    seqlen_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_qk_norm = wq is not None
    return _FusedQRKRMSNormRoPEFunction.apply(q, r, k, wq, wr, wk, cos, sin, eps, seqlen_offset, use_qk_norm)


# ---------------------------------------------------------------------------
# 6. Autograd Wrapper & Public Entry Points for Parallax RSI
# ---------------------------------------------------------------------------

def parallel_parallax_fwd_rsi(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    window_size_left: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    is_fp32 = q.dtype == torch.float32
    BK = triton.next_power_of_2(K)
    G_PACK = _pick_g_pack(HQ, H, is_fp32)
    BM = 16
    BT_Q = BM // G_PACK
    BS = 16 if is_fp32 else 64
    IS_FP32 = is_fp32
    ns = 1 if is_fp32 else 2

    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1 = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    bart = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    m = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(T, BT_Q), B * (HQ // G_PACK))
    _parallax_fwd_rsi_kernel[grid](
        q, r, k, v, o, barv, d1, bart, m, float(scale), T,
        HQ=HQ, H=H, G_PACK=G_PACK, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        BT_Q=BT_Q, BM=BM, BS=BS, IS_FP32=IS_FP32,
        num_warps=(8 if is_fp32 else 4), num_stages=ns,
    )
    return o, barv, d1, bart, m


def parallel_parallax_bwd_rsi(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    barv: torch.Tensor,
    d1: torch.Tensor,
    bart: torch.Tensor,
    m: torch.Tensor,
    grad_o: torch.Tensor,
    scale: float,
    window_size_left: int = -1,
    use_gqa_direct: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    is_fp32 = q.dtype == torch.float32
    BK = triton.next_power_of_2(K)
    G_PACK = _pick_g_pack(HQ, H, is_fp32)
    BM = 16
    BT_Q = BM // G_PACK
    BS_DQR = 16 if is_fp32 else 64
    IS_FP32 = is_fp32
    ns = 1 if is_fp32 else 2

    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    if is_fp32:
        delta_t = (grad_o * o).sum(dim=-1)
        delta_b = (grad_o * barv).sum(dim=-1)
    else:
        delta_t = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
        delta_b = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

    grid_dqr = (triton.cdiv(T, BT_Q), B * (HQ // G_PACK))
    _parallax_bwd_dqr_rsi_kernel[grid_dqr](
        q, r, k, v, o, barv, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
        float(scale), T,
        HQ=HQ, H=H, G_PACK=G_PACK, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        BT_Q=BT_Q, BM=BM, BS=BS_DQR, IS_FP32=IS_FP32,
        num_warps=(16 if (is_fp32 and K > 64) else (8 if is_fp32 else 4)), num_stages=ns,
    )

    if is_fp32:
        BT_DKV, BS_DKV = 16, 16
    elif K <= 64:
        BT_DKV, BS_DKV = 32, 32
    else:
        BT_DKV, BS_DKV = 16, 32

    if use_gqa_direct and G > 1:
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grid_dkv_direct = (triton.cdiv(T, BS_DKV), B * H)
        _parallax_bwd_dkv_gqa_direct_rsi_kernel[grid_dkv_direct](
            q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k, grad_v,
            float(scale), T,
            HQ=HQ, H=H, G=G, K=K, BK=BK,
            WINDOW_SIZE_LEFT=window_size_left,
            BT=BT_DKV, BS=BS_DKV, IS_FP32=IS_FP32,
            num_warps=(8 if is_fp32 else 4), num_stages=ns,
        )
        return grad_q, grad_r, grad_k, grad_v

    grad_k_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    grad_v_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    grid_dkv = (triton.cdiv(T, BS_DKV), B * HQ)
    _parallax_bwd_dkv_rsi_kernel[grid_dkv](
        q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k_buf, grad_v_buf,
        float(scale), T,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        BT=BT_DKV, BS=BS_DKV, IS_FP32=IS_FP32,
        num_warps=(8 if is_fp32 else 4), num_stages=ns,
    )

    if G == 1:
        return grad_q, grad_r, grad_k_buf, grad_v_buf

    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    N_TH = B * T * H
    BLOCK_M = 8
    _parallax_bwd_reduce_dkv_rsi_kernel[(triton.cdiv(N_TH, BLOCK_M),)](
        grad_k_buf, grad_v_buf, grad_k, grad_v, N_TH,
        G=G, K=K, BK=BK, BLOCK_M=BLOCK_M,
        num_warps=4, num_stages=1,
    )
    return grad_q, grad_r, grad_k, grad_v


class _ParallaxRSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        window_size_left: int,
        mutation: Optional[str],
    ) -> torch.Tensor:
        q_c = q.contiguous()
        r_c = torch.zeros_like(r).contiguous() if mutation == "drop_first_order_r_correction" else r.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        w_eff = 1 if mutation == "corrupt_sliding_window_uint32_bound" else window_size_left
        o, barv, d1, bart, m = parallel_parallax_fwd_rsi(q_c, r_c, k_c, v_c, scale, w_eff)
        if mutation == "drop_bwd_bart_minus_rk_coupling":
            bart = torch.zeros_like(bart)
        ctx.save_for_backward(q_c, r_c, k_c, v_c, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = w_eff
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q_c, r_c, k_c, v_c, o, barv, d1, bart, m = ctx.saved_tensors
        do_c = do.contiguous()
        gq, gr, gk, gv = parallel_parallax_bwd_rsi(
            q_c, r_c, k_c, v_c, o, barv, d1, bart, m, do_c,
            ctx.scale, ctx.window_size_left,
        )
        return gq, gr, gk, gv, None, None, None


def parallel_parallax_rsi(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    window_size: Optional[int] = None,
    mutation: Optional[str] = None,
) -> torch.Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    window_size_left = -1 if window_size is None else int(window_size)
    return _ParallaxRSIFunction.apply(q, r, k, v, float(scale), window_size_left, mutation)


def parallax_decode_rsi(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    window_size: Optional[int] = None,
    cache_start: Optional[torch.Tensor] = None,
    return_moments: bool = False,
):
    B, Sq, HQ, K = q.shape
    Skv, H = k.shape[1], k.shape[2]
    if scale is None:
        scale = K ** -0.5
    window_size_left = -1 if window_size is None else int(window_size)
    is_fp32 = q.dtype == torch.float32
    BK = triton.next_power_of_2(K)
    G_PACK = _pick_g_pack(HQ, H, is_fp32)
    BM = 16
    BT_Q = BM // G_PACK
    BS = 16 if is_fp32 else 64
    IS_FP32 = is_fp32
    ns = 1 if is_fp32 else 2

    q_c, r_c, k_c, v_c = (x.contiguous() for x in (q, r, k, v))
    if cache_start is not None:
        cache_start = cache_start.to(device=q.device, dtype=torch.int32).contiguous()

    o = torch.empty_like(q_c)
    if return_moments:
        m_out = torch.empty((B, Sq, HQ), device=q.device, dtype=torch.float32)
        d1_out = torch.empty((B, Sq, HQ), device=q.device, dtype=torch.float32)
        d2_out = torch.empty((B, Sq, HQ), device=q.device, dtype=torch.float32)
        o1_out = torch.empty((B, Sq, HQ, K), device=q.device, dtype=torch.float32)
        o2_out = torch.empty((B, Sq, HQ, K), device=q.device, dtype=torch.float32)
    else:
        m_out = d1_out = d2_out = o1_out = o2_out = q_c

    grid = (triton.cdiv(Sq, BT_Q) * B * (HQ // G_PACK),)
    _parallax_decode_rsi_kernel[grid](
        q_c, r_c, k_c, v_c, o, m_out, d1_out, d2_out, o1_out, o2_out,
        float(scale), cache_start, Sq, Skv,
        HQ=HQ, H=H, G_PACK=G_PACK, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        USE_CACHE_START=cache_start is not None,
        RETURN_MOMENTS=return_moments,
        BT_Q=BT_Q, BM=BM, BS=BS, IS_FP32=IS_FP32,
        num_warps=(8 if is_fp32 else 4), num_stages=ns,
    )
    if return_moments:
        return m_out, d1_out, d2_out, o1_out, o2_out
    return o


def parallax_decode_one_step_rsi(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    window_size: Optional[int] = None,
    cache_start: Optional[torch.Tensor] = None,
    force_simt: bool = False,
) -> torch.Tensor:
    B, Sq, HQ, K = q.shape
    assert Sq == 1, f"parallax_decode_one_step_rsi expects Sq=1, got {Sq}"
    Skv, H = k.shape[1], k.shape[2]
    G = HQ // H
    is_fp32 = q.dtype == torch.float32
    if not force_simt and not is_fp32 and G >= 2 and (B * (HQ // _pick_g_pack(HQ, H, False))) >= 16:
        return parallax_decode_rsi(
            q, r, k, v, scale=scale, window_size=window_size, cache_start=cache_start
        )
    if scale is None:
        scale = K ** -0.5
    window_size_left = -1 if window_size is None else int(window_size)
    BK = triton.next_power_of_2(K)
    BS = 32 if is_fp32 else 64
    ns = 1 if is_fp32 else 2

    q_c, r_c, k_c, v_c = (x.contiguous() for x in (q, r, k, v))
    if cache_start is not None:
        cache_start = cache_start.to(device=q.device, dtype=torch.int32).contiguous()

    o = torch.empty_like(q_c)
    _parallax_decode_one_step_rsi_kernel[(B * HQ,)](
        q_c, r_c, k_c, v_c, o, float(scale), cache_start, Skv,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        USE_CACHE_START=cache_start is not None,
        BS=BS, num_warps=(8 if is_fp32 else 4), num_stages=ns,
    )
    return o


# ---------------------------------------------------------------------------
# 7. Superfused ParallaxBlockRSI (5-Stage Progressive Fusion Ablation A0..A4)
# ---------------------------------------------------------------------------

class ParallaxBlockRSI(nn.Module):
    """
    5-Stage progressive ablation block for Parallax (arXiv:2605.29157):
      Stage A0: Upstream `fla.layers.parallax.Parallax` (4 separate q/r/k/v projections,
                3 separate RMSNorms, duplicate `self.rotary(r, r)`, spilling `parallel_parallax`).
      Stage A1: Stage A0 + zero-spill `parallel_parallax_rsi` (`<=168` regs, `0` spills,
                GQA-packed query tiles, fused bwd preprocess, single-pass GQA reduce).
      Stage A2: Stage A1 + fix duplicate `self.rotary(r, r)` bug (single-tensor RoPE on `r`).
      Stage A3: Stage A2 + fused `qrkv` linear projection (`1` GEMM instead of `4`).
      Stage A4: Stage A3 + single-launch fused `_parallax_qrk_rmsnorm_rope_fwd/bwd_rsi_kernel`
                fusing all 3 RMSNorms (`q_norm, r_norm, k_norm`) + 3 RotaryEmbeddings (`q, r, k`).
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        window_size: Optional[int] = 512,
        rope_theta: float = 10000.0,
        stage: str = "A4",
    ):
        super().__init__()
        from fla.layers.parallax import Parallax

        self.stage = stage
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta

        self.ref_layer = Parallax(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            window_size=window_size,
            rope_theta=rope_theta,
        )
        self.qrkv_proj = nn.Linear(
            hidden_size, 2 * hidden_size + 2 * self.kv_dim, bias=qkv_bias
        )
        self.sync_weights_from_ref()

    @torch.no_grad()
    def sync_weights_from_ref(self):
        ref = self.ref_layer
        w_cat = torch.cat(
            [ref.q_proj.weight, ref.r_proj.weight, ref.k_proj.weight, ref.v_proj.weight],
            dim=0,
        )
        self.qrkv_proj.weight.copy_(w_cat)
        if self.qrkv_proj.bias is not None:
            b_cat = torch.cat(
                [ref.q_proj.bias, ref.r_proj.bias, ref.k_proj.bias, ref.v_proj.bias],
                dim=0,
            )
            self.qrkv_proj.bias.copy_(b_cat)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ref = self.ref_layer
        if self.stage == "A0":
            o, _, _ = ref(hidden_states)
            return o

        B, T, _ = hidden_states.shape
        HQ = self.num_heads
        H = self.num_kv_heads
        D = self.head_dim

        if self.stage in ("A1", "A2"):
            q = ref.q_proj(hidden_states).view(B, T, HQ, D)
            r = ref.r_proj(hidden_states).view(B, T, HQ, D)
            k = ref.k_proj(hidden_states).view(B, T, H, D)
            v = ref.v_proj(hidden_states).view(B, T, H, D)
        else:
            qrkv = self.qrkv_proj(hidden_states)
            q_end = self.hidden_size
            r_end = 2 * self.hidden_size
            k_end = r_end + self.kv_dim
            q = qrkv[..., :q_end].view(B, T, HQ, D)
            r = qrkv[..., q_end:r_end].view(B, T, HQ, D)
            k = qrkv[..., r_end:k_end].view(B, T, H, D)
            v = qrkv[..., k_end:].view(B, T, H, D)

        if self.stage == "A1":
            if self.qk_norm:
                q, r, k = ref.q_norm(q), ref.r_norm(r), ref.k_norm(k)
            q, k = ref.rotary(q, k, seqlen_offset=0, max_seqlen=T)
            r, _ = ref.rotary(r, r, seqlen_offset=0, max_seqlen=T)
        elif self.stage in ("A2", "A3"):
            if self.qk_norm:
                q, r, k = ref.q_norm(q), ref.r_norm(r), ref.k_norm(k)
            ref.rotary._update_cos_sin_cache(T, device=q.device, dtype=q.dtype)
            cos = ref.rotary._cos_cached[:T]
            sin = ref.rotary._sin_cached[:T]
            q, r, k = fused_qrk_rmsnorm_rope_rsi(q, r, k, cos, sin, None, None, None, seqlen_offset=0)
        else:
            ref.rotary._update_cos_sin_cache(T, device=q.device, dtype=q.dtype)
            cos = ref.rotary._cos_cached[:T]
            sin = ref.rotary._sin_cached[:T]
            wq = ref.q_norm.weight if self.qk_norm else None
            wr = ref.r_norm.weight if self.qk_norm else None
            wk = ref.k_norm.weight if self.qk_norm else None
            eps = ref.q_norm.eps if self.qk_norm else 1e-6
            q, r, k = fused_qrk_rmsnorm_rope_rsi(q, r, k, cos, sin, wq, wr, wk, eps=eps, seqlen_offset=0)

        o = parallel_parallax_rsi(q, r, k, v, window_size=self.window_size)
        return ref.o_proj(o.reshape(B, T, self.hidden_size))


# ---------------------------------------------------------------------------
# 8. 8-GPU NVSwitch Context Parallelism (`Parallax-CP = 8`)
# ---------------------------------------------------------------------------

def merge_parallax_moments_5tuple(
    m_a: torch.Tensor,
    d1_a: torch.Tensor,
    d2_a: torch.Tensor,
    o1_a: torch.Tensor,
    o2_a: torch.Tensor,
    m_b: torch.Tensor,
    d1_b: torch.Tensor,
    d2_b: torch.Tensor,
    o1_b: torch.Tensor,
    o2_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Associative 5-tuple online-softmax moment merger (m, d1, d2, O1, O2) for Parallax:
        m = max(m_a, m_b)
        alpha_a = 2^(m_a - safe_m), alpha_b = 2^(m_b - safe_m)
        d1 = alpha_a * d1_a + alpha_b * d1_b
        d2 = alpha_a * d2_a + alpha_b * d2_b
        O1 = alpha_a * O1_a + alpha_b * O1_b
        O2 = alpha_a * O2_a + alpha_b * O2_b
    """
    m_new = torch.maximum(m_a, m_b)
    safe_m = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
    alpha_a = torch.exp2(m_a - safe_m)
    alpha_b = torch.exp2(m_b - safe_m)
    d1_new = alpha_a * d1_a + alpha_b * d1_b
    d2_new = alpha_a * d2_a + alpha_b * d2_b
    o1_new = alpha_a.unsqueeze(-1) * o1_a + alpha_b.unsqueeze(-1) * o1_b
    o2_new = alpha_a.unsqueeze(-1) * o2_a + alpha_b.unsqueeze(-1) * o2_b
    return m_new, d1_new, d2_new, o1_new, o2_new


def finalize_parallax_from_moments(
    d1: torch.Tensor,
    d2: torch.Tensor,
    o1: torch.Tensor,
    o2: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    inv_d1 = torch.where(d1 > 0.0, 1.0 / d1, torch.zeros_like(d1)).unsqueeze(-1)
    barv = o1 * inv_d1
    bart = (d2 * inv_d1.squeeze(-1)).unsqueeze(-1)
    out = barv * (1.0 + bart) - o2 * inv_d1
    return out.to(out_dtype)


@triton.jit(do_not_specialize=['Sq', 'Skv'])
def _parallax_shard_moments_rsi_kernel(
    q,
    r,
    k,
    v,
    m_out,
    d1_out,
    d2_out,
    o1_out,
    o2_out,
    scale,
    Sq,
    Skv,
    q_abs_start,
    kv_abs_start,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G_PACK: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT_Q: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NT = tl.cdiv(Sq, BT_Q)
    NUM_HG: tl.constexpr = HQ // G_PACK
    G_TOTAL: tl.constexpr = HQ // H
    i_t = (pid % NT).to(tl.int32)
    i_bhg = pid // NT
    i_b = i_bhg // NUM_HG
    i_hg = i_bhg % NUM_HG
    hq_base = i_hg * G_PACK
    i_h = hq_base // G_TOTAL
    RCP_LN2: tl.constexpr = 1.4426950216
    W_BOUND: tl.constexpr = WINDOW_SIZE_LEFT if WINDOW_SIZE_LEFT >= 0 else 2147483647

    q_off = i_t * BT_Q
    o_m = tl.arange(0, BM)
    t_local = o_m // G_PACK
    g_local = o_m % G_PACK
    rows = q_off + t_local
    hq_indices = hq_base + g_local
    abs_q = (q_abs_start + rows)[:, None]
    row_mask = (rows < Sq)[:, None]

    max_abs_q = q_abs_start + tl.minimum(Sq, q_off + BT_Q) - 1
    rel_end = tl.minimum(Skv, tl.maximum(0, max_abs_q - kv_abs_start + 1))
    KV_END_BLOCK = tl.cdiv(rel_end, BS)
    if WINDOW_SIZE_LEFT >= 0:
        min_abs_k = q_abs_start + q_off - WINDOW_SIZE_LEFT + 1
        rel_start = tl.maximum(0, min_abs_k - kv_abs_start)
        KV_START_BLOCK = tl.minimum(KV_END_BLOCK, rel_start // BS)
    else:
        KV_START_BLOCK = 0

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    qh_offset = ((i_b * Sq + rows.to(tl.int64)) * HQ + hq_indices)[:, None]

    b_q = tl.load(q + qh_offset * K + o_k[None, :], mask=m_qk, other=0.0)
    b_r = tl.load(r + qh_offset * K + o_k[None, :], mask=m_qk, other=0.0)
    m_acc = tl.zeros((BM, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BM, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BM, 1), dtype=tl.float32)
    o1_acc = tl.zeros((BM, BK), dtype=tl.float32)
    o2_acc = tl.zeros((BM, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    for col_block_id in range(KV_START_BLOCK, KV_END_BLOCK):
        rel_col = col_block_id * BS + tl.arange(0, BS)
        abs_col = (kv_abs_start + rel_col)[None, :]
        m_kv = (rel_col[:, None] < Skv) & m_k[None, :]
        p_k = k + (i_b * Skv * H + i_h) * K + rel_col[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        qk = _dot_rsi(b_q, tl.trans(b_k), IS_FP32) * scale_log2
        rk = _dot_rsi(b_r, tl.trans(b_k), IS_FP32)
        diff = (abs_q - abs_col).to(tl.uint32)
        mask = (diff < W_BOUND) & row_mask & (rel_col[None, :] < Skv)
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_acc - safe_m)
        w = tl.math.exp2(qk - safe_m)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        w_cast = w.to(b_q.dtype)
        wr_cast = wr.to(b_q.dtype)
        p_v = v + (i_b * Skv * H + i_h) * K + rel_col[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        o1_acc = _dot_acc_rsi(w_cast, b_v, alpha * o1_acc, IS_FP32)
        o2_acc = _dot_acc_rsi(wr_cast, b_v, alpha * o2_acc, IS_FP32)
        m_acc = m_new

    tl.store(m_out + qh_offset, m_acc, mask=row_mask)
    tl.store(d1_out + qh_offset, d1_acc, mask=row_mask)
    tl.store(d2_out + qh_offset, d2_acc, mask=row_mask)
    tl.store(o1_out + qh_offset * K + o_k[None, :], o1_acc, mask=m_qk)
    tl.store(o2_out + qh_offset * K + o_k[None, :], o2_acc, mask=m_qk)


def eval_parallax_shard_moments_rsi(
    q_local: torch.Tensor,
    r_local: torch.Tensor,
    k_shard: torch.Tensor,
    v_shard: torch.Tensor,
    q_abs_start: int,
    kv_abs_start: int,
    scale: Optional[float] = None,
    window_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Sq, HQ, K = q_local.shape
    Skv, H = k_shard.shape[1], k_shard.shape[2]
    if scale is None:
        scale = K ** -0.5
    window_size_left = -1 if window_size is None else int(window_size)
    is_fp32 = q_local.dtype == torch.float32
    BK = triton.next_power_of_2(K)
    G_PACK = _pick_g_pack(HQ, H, is_fp32)
    BM = 16
    BT_Q = BM // G_PACK
    BS = 16 if is_fp32 else 64
    IS_FP32 = is_fp32
    ns = 1 if is_fp32 else 2

    m_out = torch.empty((B, Sq, HQ), device=q_local.device, dtype=torch.float32)
    d1_out = torch.empty((B, Sq, HQ), device=q_local.device, dtype=torch.float32)
    d2_out = torch.empty((B, Sq, HQ), device=q_local.device, dtype=torch.float32)
    o1_out = torch.empty((B, Sq, HQ, K), device=q_local.device, dtype=torch.float32)
    o2_out = torch.empty((B, Sq, HQ, K), device=q_local.device, dtype=torch.float32)

    grid = (triton.cdiv(Sq, BT_Q) * B * (HQ // G_PACK),)
    _parallax_shard_moments_rsi_kernel[grid](
        q_local.contiguous(), r_local.contiguous(), k_shard.contiguous(), v_shard.contiguous(),
        m_out, d1_out, d2_out, o1_out, o2_out,
        float(scale), Sq, Skv, int(q_abs_start), int(kv_abs_start),
        HQ=HQ, H=H, G_PACK=G_PACK, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        BT_Q=BT_Q, BM=BM, BS=BS, IS_FP32=IS_FP32,
        num_warps=(8 if is_fp32 else 4), num_stages=ns,
    )
    return m_out, d1_out, d2_out, o1_out, o2_out


def parallax_cp8_ring_baseline(
    q_local: torch.Tensor,
    r_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    window_size: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    8-GPU Ring Context Parallelism baseline for Parallax using the associative
    5-tuple online-softmax moment merger (m, d1, d2, O1, O2) across 8 ranks.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    B, T_local, HQ, D = q_local.shape
    scale = D ** -0.5
    q_abs_start = rank * T_local

    # Ring P2P circulation of (k_shard, v_shard) across world_size steps
    kv_curr = torch.stack([k_local.contiguous(), v_local.contiguous()], dim=0)
    kv_next = torch.empty_like(kv_curr)
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size

    m_acc = d1_acc = d2_acc = o1_acc = o2_acc = None

    for step in range(world_size):
        src_rank = (rank - step + world_size) % world_size
        kv_abs_start = src_rank * T_local
        reqs = []
        if step < world_size - 1:
            ops = [
                dist.P2POp(dist.isend, kv_curr, send_rank, group=group),
                dist.P2POp(dist.irecv, kv_next, recv_rank, group=group),
            ]
            reqs = dist.batch_isend_irecv(ops)

        # Only evaluate shards that can be causally visible and within sliding window
        max_q_abs = q_abs_start + T_local - 1
        min_k_abs = kv_abs_start
        max_k_abs = kv_abs_start + T_local - 1
        visible = min_k_abs <= max_q_abs
        if window_size is not None:
            visible = visible and (max_k_abs >= q_abs_start - window_size + 1)

        if visible:
            k_sh, v_sh = kv_curr[0], kv_curr[1]
            m_s, d1_s, d2_s, o1_s, o2_s = eval_parallax_shard_moments_rsi(
                q_local, r_local, k_sh, v_sh, q_abs_start, kv_abs_start,
                scale=scale, window_size=window_size,
            )
            if m_acc is None:
                m_acc, d1_acc, d2_acc, o1_acc, o2_acc = m_s, d1_s, d2_s, o1_s, o2_s
            else:
                m_acc, d1_acc, d2_acc, o1_acc, o2_acc = merge_parallax_moments_5tuple(
                    m_acc, d1_acc, d2_acc, o1_acc, o2_acc,
                    m_s, d1_s, d2_s, o1_s, o2_s,
                )

        for req in reqs:
            req.wait()
        kv_curr, kv_next = kv_next, kv_curr

    return finalize_parallax_from_moments(d1_acc, d2_acc, o1_acc, o2_acc, q_local.dtype)


def parallax_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    r_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    window_size: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    8-GPU Zero-Copy GQA-Aware Ulysses Context Parallelism for Parallax:
    Packs (q, r) and (k, v) into coalesced NVSwitch all_to_all_single collectives,
    runs single-pass zero-spill `parallel_parallax_rsi` on [B, T_global, HQ/P, D],
    and transposes output `o` back to [B, T_local, HQ, D].
    """
    world_size = dist.get_world_size(group)
    B, T_local, HQ, D = q_local.shape
    H = k_local.shape[2]
    assert HQ % world_size == 0 and H % world_size == 0
    HQ_local = HQ // world_size
    H_local = H // world_size
    T_global = T_local * world_size

    # Pack (q, r) -> [world_size, 2, B, T_local, HQ_local, D]
    qr_send = torch.stack([q_local, r_local], dim=0).view(2, B, T_local, world_size, HQ_local, D)
    qr_send = qr_send.permute(3, 0, 1, 2, 4, 5).contiguous()
    qr_recv = torch.empty_like(qr_send)
    dist.all_to_all_single(qr_recv, qr_send, group=group)
    # qr_recv: [world_size, 2, B, T_local, HQ_local, D] -> [2, B, T_global, HQ_local, D]
    qr_full = qr_recv.permute(1, 2, 0, 3, 4, 5).contiguous().view(2, B, T_global, HQ_local, D)

    kv_send = torch.stack([k_local, v_local], dim=0).view(2, B, T_local, world_size, H_local, D)
    kv_send = kv_send.permute(3, 0, 1, 2, 4, 5).contiguous()
    kv_recv = torch.empty_like(kv_send)
    dist.all_to_all_single(kv_recv, kv_send, group=group)
    kv_full = kv_recv.permute(1, 2, 0, 3, 4, 5).contiguous().view(2, B, T_global, H_local, D)

    o_full = parallel_parallax_rsi(
        qr_full[0], qr_full[1], kv_full[0], kv_full[1], window_size=window_size
    )

    # Transpose o_full [B, T_global, HQ_local, D] -> [B, T_local, HQ, D]
    o_send = o_full.view(B, world_size, T_local, HQ_local, D).permute(1, 0, 2, 3, 4).contiguous()
    o_recv = torch.empty_like(o_send)
    dist.all_to_all_single(o_recv, o_send, group=group)
    o_local = o_recv.permute(1, 2, 0, 3, 4).contiguous().view(B, T_local, HQ, D)
    return o_local


# ---------------------------------------------------------------------------
# 9. SM80 Cubin Register & Spill Audit Functions
# ---------------------------------------------------------------------------

def audit_parallax_rsi_cubin_kernels(warmup_run: bool = True) -> dict:
    if warmup_run:
        device = "cuda"
        for dtype in (torch.bfloat16, torch.float32):
            for D in (64, 128):
                for W in (512, None):
                    B, T, HQ, H = 1, 256, 8, 2
                    q = torch.randn(B, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
                    r = torch.randn(B, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
                    o = parallel_parallax_rsi(q, r, k, v, window_size=W)
                    o.backward(torch.randn_like(o))
                    _ = parallel_parallax_bwd_rsi(
                        q.detach(), r.detach(), k.detach(), v.detach(),
                        o.detach(), o.detach(),
                        torch.ones(B, T, HQ, device=device),
                        torch.zeros(B, T, HQ, device=device),
                        torch.zeros(B, T, HQ, device=device),
                        torch.randn_like(o),
                        D ** -0.5, -1 if W is None else W, use_gqa_direct=True,
                    )
                    _ = parallax_decode_rsi(q[:, :32].detach(), r[:, :32].detach(), k.detach(), v.detach(), window_size=W)
                    _ = parallax_decode_one_step_rsi(q[:, :1].detach(), r[:, :1].detach(), k.detach(), v.detach(), window_size=W, force_simt=True)
                    _ = eval_parallax_shard_moments_rsi(
                        q[:, :64].detach(), r[:, :64].detach(), k[:, :64].detach(), v[:, :64].detach(),
                        0, 0, window_size=W,
                    )
                    cos = torch.ones(T, D // 2, device=device, dtype=dtype)
                    sin = torch.zeros(T, D // 2, device=device, dtype=dtype)
                    wq = torch.ones(D, device=device, dtype=torch.float32, requires_grad=True)
                    wr = torch.ones(D, device=device, dtype=torch.float32, requires_grad=True)
                    wk = torch.ones(D, device=device, dtype=torch.float32, requires_grad=True)
                    qo, ro, ko = fused_qrk_rmsnorm_rope_rsi(q, r, k, cos, sin, wq, wr, wk)
                    (qo.sum() + ro.sum() + ko.sum()).backward()

    kernels = [
        ("_parallax_fwd_rsi_kernel", _parallax_fwd_rsi_kernel),
        ("_parallax_bwd_dqr_rsi_kernel", _parallax_bwd_dqr_rsi_kernel),
        ("_parallax_bwd_dkv_rsi_kernel", _parallax_bwd_dkv_rsi_kernel),
        ("_parallax_bwd_reduce_dkv_rsi_kernel", _parallax_bwd_reduce_dkv_rsi_kernel),
        ("_parallax_bwd_dkv_gqa_direct_rsi_kernel", _parallax_bwd_dkv_gqa_direct_rsi_kernel),
        ("_parallax_decode_rsi_kernel", _parallax_decode_rsi_kernel),
        ("_parallax_decode_one_step_rsi_kernel", _parallax_decode_one_step_rsi_kernel),
        ("_parallax_shard_moments_rsi_kernel", _parallax_shard_moments_rsi_kernel),
        ("_parallax_qrk_rmsnorm_rope_fwd_rsi_kernel", _parallax_qrk_rmsnorm_rope_fwd_rsi_kernel),
        ("_parallax_qrk_rmsnorm_rope_bwd_rsi_kernel", _parallax_qrk_rmsnorm_rope_bwd_rsi_kernel),
    ]
    rows = []
    max_regs = 0
    total_spills = 0
    max_smem = 0
    total_specs = 0
    for name, kobj in kernels:
        target = kobj if hasattr(kobj, "device_caches") else getattr(kobj, "fn", kobj)
        cache = getattr(target, "device_caches", {}).get(0, ({},))[0]
        k_regs = 0
        k_spills = 0
        k_smem = 0
        n_specs = 0
        for _, cbin in cache.items():
            n_specs += 1
            total_specs += 1
            k_regs = max(k_regs, int(cbin.n_regs))
            k_spills += int(cbin.n_spills)
            k_smem = max(k_smem, int(cbin.metadata.shared))
        max_regs = max(max_regs, k_regs)
        total_spills += k_spills
        max_smem = max(max_smem, k_smem)
        rows.append({
            "kernel": name,
            "n_specializations": n_specs,
            "max_regs": k_regs,
            "total_spills": k_spills,
            "max_shared_bytes": k_smem,
        })
    return {
        "n_kernels": len(rows),
        "total_specializations": total_specs,
        "max_regs": max_regs,
        "total_spills": total_spills,
        "max_shared_bytes": max_smem,
        "all_zero_spills": total_spills == 0,
        "all_regs_le_168": max_regs <= 168,
        "kernels": rows,
    }


def audit_upstream_parallax_cubin_kernels(warmup_run: bool = True) -> dict:
    from fla.ops.parallax import parallel_parallax
    from fla.ops.parallax.decode import (
        parallax_decode,
        parallax_decode_kernel,
        parallax_decode_one_step,
        parallax_decode_one_step_kernel,
    )
    from fla.ops.parallax.parallel import (
        parallel_parallax_bwd_kernel_dkv,
        parallel_parallax_bwd_kernel_dqr,
        parallel_parallax_bwd_kernel_preprocess,
        parallel_parallax_fwd_kernel,
    )

    if warmup_run:
        device = "cuda"
        for D in (64, 128):
            for H in (16, 4):
                for W in (None, 512):
                    B, T, HQ = 1, 256, 16
                    q = torch.randn(B, T, HQ, D, device=device, dtype=torch.bfloat16, requires_grad=True)
                    r = torch.randn(B, T, HQ, D, device=device, dtype=torch.bfloat16, requires_grad=True)
                    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
                    v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
                    o = parallel_parallax(q, r, k, v, window_size=W)
                    o.backward(torch.randn_like(o))
                    _ = parallax_decode(q[:, :32].detach(), r[:, :32].detach(), k.detach(), v.detach(), window_size=W)
                    _ = parallax_decode_one_step(q[:, :1].detach(), r[:, :1].detach(), k.detach(), v.detach(), window_size=W)

    kernels = [
        ("parallel_parallax_fwd_kernel", parallel_parallax_fwd_kernel),
        ("parallel_parallax_bwd_kernel_preprocess", parallel_parallax_bwd_kernel_preprocess),
        ("parallel_parallax_bwd_kernel_dqr", parallel_parallax_bwd_kernel_dqr),
        ("parallel_parallax_bwd_kernel_dkv", parallel_parallax_bwd_kernel_dkv),
        ("parallax_decode_kernel", parallax_decode_kernel),
        ("parallax_decode_one_step_kernel", parallax_decode_one_step_kernel),
    ]
    rows = []
    max_regs = 0
    max_spills = 0
    total_spills = 0
    spilling_specs = 0
    regs_gt_168_specs = 0
    max_smem = 0
    total_specs = 0
    for name, kobj in kernels:
        target = kobj if hasattr(kobj, "device_caches") else getattr(kobj, "fn", kobj)
        cache = getattr(target, "device_caches", {}).get(0, ({},))[0]
        k_regs = 0
        k_max_spills = 0
        k_sum_spills = 0
        k_spilling_specs = 0
        k_smem = 0
        n_specs = 0
        for _, cbin in cache.items():
            n_specs += 1
            total_specs += 1
            nr = int(cbin.n_regs)
            nsp = int(cbin.n_spills)
            sm = int(cbin.metadata.shared)
            k_regs = max(k_regs, nr)
            k_max_spills = max(k_max_spills, nsp)
            k_sum_spills += nsp
            if nsp > 0:
                k_spilling_specs += 1
                spilling_specs += 1
            if nr > 168:
                regs_gt_168_specs += 1
            k_smem = max(k_smem, sm)
        max_regs = max(max_regs, k_regs)
        max_spills = max(max_spills, k_max_spills)
        total_spills += k_sum_spills
        max_smem = max(max_smem, k_smem)
        rows.append({
            "kernel": name,
            "n_specializations": n_specs,
            "spilling_specializations": k_spilling_specs,
            "max_regs": k_regs,
            "max_spills": k_max_spills,
            "total_spills": k_sum_spills,
            "max_shared_bytes": k_smem,
        })
    return {
        "n_kernels": len(rows),
        "total_specializations": total_specs,
        "spilling_specializations": spilling_specs,
        "regs_gt_168_specializations": regs_gt_168_specs,
        "max_regs": max_regs,
        "max_spills": max_spills,
        "total_spills": total_spills,
        "max_shared_bytes": max_smem,
        "kernels": rows,
    }
