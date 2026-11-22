from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# 1. Flattened-(BM * HI) Register-Resident MQA Lightning Indexer Kernels
# ============================================================================

@triton.jit
def _dsa_indexer_mqa_fwd_rsi_kernel(
    q_idx,
    k_idx,
    w_idx,
    logits_out,
    T: tl.constexpr,
    HI: tl.constexpr,
    DI: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_b = tl.program_id(1)
    BMH: tl.constexpr = BM * HI
    o_mh = tl.arange(0, BMH)
    o_m_of_mh = i_m * BM + (o_mh // HI)
    mask_mh = o_m_of_mh < T
    o_d = tl.arange(0, DI)
    scale = 1.0 / tl.sqrt(float(DI))

    p_q = q_idx + i_b * (T * HI * DI) + (i_m * BMH + o_mh[:, None]) * DI + o_d[None, :]
    b_q = tl.load(p_q, mask=mask_mh[:, None], other=0.0)
    if not IS_FP32:
        b_q = b_q.to(tl.bfloat16)
    else:
        b_q = b_q.to(tl.float32)

    if HAS_W:
        p_w = w_idx + i_b * (T * HI) + i_m * BMH + o_mh
        b_w = tl.load(p_w, mask=mask_mh, other=0.0).to(tl.float32)
    else:
        b_w = tl.full([BMH], 1.0, dtype=tl.float32)

    o_m = i_m * BM + tl.arange(0, BM)
    mask_m = o_m < T
    o_n = tl.arange(0, BN)
    n_blocks = tl.cdiv(tl.minimum((i_m + 1) * BM, T), BN)
    tot_blocks = tl.cdiv(T, BN)

    for i_n in range(0, n_blocks):
        cols = i_n * BN + o_n
        mask_n = cols < T
        p_k = k_idx + i_b * (T * DI) + cols[None, :] * DI + o_d[:, None]
        b_k = tl.load(p_k, mask=mask_n[None, :], other=0.0)
        if not IS_FP32:
            b_k = b_k.to(tl.bfloat16)
            b_s = tl.dot(b_q, b_k) * scale
        else:
            b_k = b_k.to(tl.float32)
            b_s = tl.dot(b_q, b_k, input_precision="ieee") * scale
        b_s = tl.maximum(b_s, 0.0) * b_w[:, None]
        b_s3d = tl.reshape(b_s, [BM, HI, BN])
        b_acc = tl.sum(b_s3d, axis=1)
        causal = o_m[:, None] >= cols[None, :]
        b_acc = tl.where(causal & mask_m[:, None] & mask_n[None, :], b_acc, float("-inf"))
        p_out = logits_out + i_b * (T * T) + o_m[:, None] * T + cols[None, :]
        tl.store(p_out, b_acc, mask=mask_m[:, None] & mask_n[None, :])

    for i_n in range(n_blocks, tot_blocks):
        cols = i_n * BN + o_n
        mask_n = cols < T
        p_out = logits_out + i_b * (T * T) + o_m[:, None] * T + cols[None, :]
        tl.store(p_out, tl.full([BM, BN], float("-inf"), dtype=tl.float32), mask=mask_m[:, None] & mask_n[None, :])


# ============================================================================
# 2. 64-Bit Bitmask & Tile-Skip Builder Kernel
# ============================================================================

@triton.jit
def _dsa_build_bitmask_rsi_kernel(
    indices,
    bitmask_out,
    tile_active_out,
    T: tl.constexpr,
    TOPK: tl.constexpr,
    NB64: tl.constexpr,
    BS: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_b = tl.program_id(1)
    i_m64 = i_t // 64
    o_s = tl.arange(0, BS)
    num_sb = tl.cdiv(TOPK, BS)
    idx_base = indices + (i_b * T + i_t) * TOPK
    bm_base = bitmask_out + (i_b * T + i_t) * NB64
    ta_base = tile_active_out + (i_b * NB64 + i_m64) * NB64

    for i_sb in range(num_sb):
        r_offs = i_sb * BS + o_s
        mask_r = r_offs < TOPK
        b_idx = tl.load(idx_base + r_offs, mask=mask_r, other=-1)
        valid = mask_r & (b_idx >= 0) & (b_idx <= i_t)
        blk_id = tl.where(valid, b_idx // 64, 0)
        bit_pos = tl.where(valid, b_idx % 64, 0).to(tl.uint64)
        word = tl.where(valid, (tl.full([BS], 1, dtype=tl.uint64) << bit_pos), tl.zeros([BS], dtype=tl.uint64))
        tl.atomic_or(bm_base + blk_id, word, mask=valid)
        tl.store(ta_base + blk_id, 1, mask=valid)


# ============================================================================
# 3. GQA-Packed 64-Bit Bitmask Tensor-Core Sparse Attention Forward & Backward
# ============================================================================

@triton.jit
def _dsa_gqa_bitmask_fwd_rsi_kernel(
    q,
    k,
    v,
    bitmask,
    tile_active,
    o,
    lse_out,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NB64: tl.constexpr,
    BMT: tl.constexpr,
    BMG: tl.constexpr,
    BN: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_mg = tl.arange(0, BMG)
    o_t = i_m * BMT + (o_mg // G)
    o_g = o_mg % G
    mask_mg = (o_t < T) & ((o_mg // G) < BMT)
    i_m64 = (i_m * BMT) // 64
    o_n = tl.arange(0, BN)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    scale_log2 = (1.0 / tl.sqrt(float(K))) * 1.4426950408889634

    row_qh = (i_b * T + o_t) * HQ + (i_h * G + o_g)
    p_q = q + row_qh[:, None] * K + o_k[None, :]
    m_i = tl.full([BMG], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BMG], dtype=tl.float32)
    acc_o = tl.zeros([BMG, V], dtype=tl.float32)

    ta_base = tile_active + (i_b * NB64 + i_m64) * NB64
    bm_base = bitmask + (i_b * T + o_t) * NB64
    n_limit = tl.cdiv(tl.minimum((i_m + 1) * BMT, T), BN)

    for i_n in range(0, n_limit):
        i_n64 = (i_n * BN) // 64
        act = tl.load(ta_base + i_n64)
        if act != 0:
            sub_shift = (i_n * BN) % 64
            cols = i_n * BN + o_n
            mask_n = cols < T
            b_mword = tl.load(bm_base + i_n64, mask=mask_mg, other=0).to(tl.uint64)
            shift_u64 = (sub_shift + o_n).to(tl.uint64)
            sel_mask = ((b_mword[:, None] >> shift_u64[None, :]) & 1) != 0
            valid = sel_mask & (o_t[:, None] >= cols[None, :]) & mask_mg[:, None] & mask_n[None, :]

            b_q = tl.load(p_q, mask=mask_mg[:, None], other=0.0)
            p_k = k + ((i_b * T + cols[None, :]) * H + i_h) * K + o_k[:, None]
            b_k = tl.load(p_k, mask=mask_n[None, :], other=0.0)
            if not IS_FP32:
                b_q = (b_q.to(tl.float32) * scale_log2).to(tl.bfloat16)
                b_k = b_k.to(tl.bfloat16)
                b_s = tl.dot(b_q, b_k)
            else:
                b_q = b_q.to(tl.float32) * scale_log2
                b_k = b_k.to(tl.float32)
                b_s = tl.dot(b_q, b_k, input_precision="ieee")
            b_s = tl.where(valid, b_s, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(b_s, axis=1))
            m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            alpha = tl.math.exp2(m_i - m_ij_safe)
            p_ij = tl.math.exp2(b_s - m_ij_safe[:, None])
            l_i = l_i * alpha + tl.sum(p_ij, axis=1)
            acc_o = acc_o * alpha[:, None]

            p_v = v + ((i_b * T + cols[:, None]) * H + i_h) * V + o_v[None, :]
            b_v = tl.load(p_v, mask=mask_n[:, None], other=0.0)
            if not IS_FP32:
                b_v = b_v.to(tl.bfloat16)
                acc_o += tl.dot(p_ij.to(tl.bfloat16), b_v)
            else:
                b_v = b_v.to(tl.float32)
                acc_o += tl.dot(p_ij, b_v, input_precision="ieee")
            m_i = m_ij

    l_rcp = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
    b_o = acc_o * l_rcp[:, None]
    b_lse = m_i + tl.math.log2(tl.maximum(l_i, 1e-30))

    p_o = o + row_qh[:, None] * V + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_mg[:, None])
    tl.store(lse_out + row_qh, b_lse, mask=mask_mg)


@triton.jit
def _dsa_bwd_prep_delta_rsi_kernel(
    o,
    do,
    delta,
    T: tl.constexpr,
    HQ: tl.constexpr,
    V: tl.constexpr,
    BM: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_bhq = tl.program_id(1)
    i_b = i_bhq // HQ
    i_hq = i_bhq % HQ
    o_m = i_m * BM + tl.arange(0, BM)
    mask_m = o_m < T
    o_v = tl.arange(0, V)
    p_o = o + ((i_b * T + o_m[:, None]) * HQ + i_hq) * V + o_v[None, :]
    p_do = do + ((i_b * T + o_m[:, None]) * HQ + i_hq) * V + o_v[None, :]
    b_o = tl.load(p_o, mask=mask_m[:, None], other=0.0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_m[:, None], other=0.0).to(tl.float32)
    b_d = tl.sum(b_o * b_do, axis=1)
    tl.store(delta + (i_b * T + o_m) * HQ + i_hq, b_d, mask=mask_m)


@triton.jit
def _dsa_gqa_bitmask_bwd_dq_rsi_kernel(
    q,
    k,
    v,
    do,
    lse,
    delta,
    bitmask,
    tile_active,
    dq,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NB64: tl.constexpr,
    BMT: tl.constexpr,
    BMG: tl.constexpr,
    BN: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_mg = tl.arange(0, BMG)
    o_t = i_m * BMT + (o_mg // G)
    o_g = o_mg % G
    mask_mg = (o_t < T) & ((o_mg // G) < BMT)
    i_m64 = (i_m * BMT) // 64
    o_n = tl.arange(0, BN)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    scale = 1.0 / tl.sqrt(float(K))
    scale_log2 = scale * 1.4426950408889634

    row_qh = (i_b * T + o_t) * HQ + (i_h * G + o_g)
    p_q = q + row_qh[:, None] * K + o_k[None, :]
    p_do = do + row_qh[:, None] * V + o_v[None, :]
    b_lse = tl.load(lse + row_qh, mask=mask_mg, other=0.0)
    b_delta = tl.load(delta + row_qh, mask=mask_mg, other=0.0)
    acc_dq = tl.zeros([BMG, K], dtype=tl.float32)

    ta_base = tile_active + (i_b * NB64 + i_m64) * NB64
    bm_base = bitmask + (i_b * T + o_t) * NB64
    n_limit = tl.cdiv(tl.minimum((i_m + 1) * BMT, T), BN)

    for i_n in range(0, n_limit):
        i_n64 = (i_n * BN) // 64
        act = tl.load(ta_base + i_n64)
        if act != 0:
            sub_shift = (i_n * BN) % 64
            cols = i_n * BN + o_n
            mask_n = cols < T
            b_mword = tl.load(bm_base + i_n64, mask=mask_mg, other=0).to(tl.uint64)
            shift_u64 = (sub_shift + o_n).to(tl.uint64)
            sel_mask = ((b_mword[:, None] >> shift_u64[None, :]) & 1) != 0
            valid = sel_mask & (o_t[:, None] >= cols[None, :]) & mask_mg[:, None] & mask_n[None, :]

            b_q = tl.load(p_q, mask=mask_mg[:, None], other=0.0)
            p_k = k + ((i_b * T + cols[None, :]) * H + i_h) * K + o_k[:, None]
            b_kt = tl.load(p_k, mask=mask_n[None, :], other=0.0)
            if not IS_FP32:
                b_q = (b_q.to(tl.float32) * scale_log2).to(tl.bfloat16)
                b_kt = b_kt.to(tl.bfloat16)
                b_s = tl.dot(b_q, b_kt)
            else:
                b_q = b_q.to(tl.float32) * scale_log2
                b_kt = b_kt.to(tl.float32)
                b_s = tl.dot(b_q, b_kt, input_precision="ieee")
            p_ij = tl.where(valid, tl.math.exp2(b_s - b_lse[:, None]), 0.0)

            b_do = tl.load(p_do, mask=mask_mg[:, None], other=0.0)
            p_v = v + ((i_b * T + cols[None, :]) * H + i_h) * V + o_v[:, None]
            b_vt = tl.load(p_v, mask=mask_n[None, :], other=0.0)
            if not IS_FP32:
                b_do = b_do.to(tl.bfloat16)
                b_vt = b_vt.to(tl.bfloat16)
                b_dp = tl.dot(b_do, b_vt)
            else:
                b_do = b_do.to(tl.float32)
                b_vt = b_vt.to(tl.float32)
                b_dp = tl.dot(b_do, b_vt, input_precision="ieee")

            b_ds = p_ij * (b_dp - b_delta[:, None])
            if not IS_FP32:
                acc_dq += tl.dot(b_ds.to(tl.bfloat16), tl.trans(b_kt))
            else:
                acc_dq += tl.dot(b_ds, tl.trans(b_kt), input_precision="ieee")

    acc_dq = acc_dq * scale
    p_dq = dq + row_qh[:, None] * K + o_k[None, :]
    tl.store(p_dq, acc_dq.to(p_dq.dtype.element_ty), mask=mask_mg[:, None])


@triton.jit
def _dsa_gqa_bitmask_bwd_dkv_rsi_kernel(
    q,
    k,
    v,
    do,
    lse,
    delta,
    bitmask,
    tile_active,
    dk,
    dv,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NB64: tl.constexpr,
    BMT: tl.constexpr,
    BMG: tl.constexpr,
    BN: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    cols = i_n * BN + tl.arange(0, BN)
    mask_n = cols < T
    i_n64 = (i_n * BN) // 64
    sub_n = (i_n * BN) % 64
    shift_u64 = (sub_n + tl.arange(0, BN)).to(tl.uint64)
    o_mg = tl.arange(0, BMG)
    o_g = o_mg % G
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    scale = 1.0 / tl.sqrt(float(K))
    scale_log2 = scale * 1.4426950408889634

    p_k = k + ((i_b * T + cols[None, :]) * H + i_h) * K + o_k[:, None]
    p_v = v + ((i_b * T + cols[None, :]) * H + i_h) * V + o_v[:, None]

    acc_dk = tl.zeros([BN, K], dtype=tl.float32)
    acc_dv = tl.zeros([BN, V], dtype=tl.float32)
    m_start = (i_n * BN) // BMT
    m_limit = tl.cdiv(T, BMT)

    for i_m in range(m_start, m_limit):
        i_m64 = (i_m * BMT) // 64
        act = tl.load(tile_active + (i_b * NB64 + i_m64) * NB64 + i_n64)
        if act != 0:
            o_t = i_m * BMT + (o_mg // G)
            mask_mg = (o_t < T) & ((o_mg // G) < BMT)
            row_qh = (i_b * T + o_t) * HQ + (i_h * G + o_g)
            b_mword = tl.load(bitmask + (i_b * T + o_t) * NB64 + i_n64, mask=mask_mg, other=0).to(tl.uint64)
            sel_mask = ((b_mword[:, None] >> shift_u64[None, :]) & 1) != 0
            valid = sel_mask & (o_t[:, None] >= cols[None, :]) & mask_mg[:, None] & mask_n[None, :]

            p_q = q + row_qh[:, None] * K + o_k[None, :]
            b_q_raw = tl.load(p_q, mask=mask_mg[:, None], other=0.0)
            b_kt = tl.load(p_k, mask=mask_n[None, :], other=0.0)
            b_lse = tl.load(lse + row_qh, mask=mask_mg, other=0.0)
            b_delta = tl.load(delta + row_qh, mask=mask_mg, other=0.0)

            if not IS_FP32:
                b_q_sc = (b_q_raw.to(tl.float32) * scale_log2).to(tl.bfloat16)
                b_kt = b_kt.to(tl.bfloat16)
                b_s = tl.dot(b_q_sc, b_kt)
            else:
                b_q_sc = b_q_raw.to(tl.float32) * scale_log2
                b_kt = b_kt.to(tl.float32)
                b_s = tl.dot(b_q_sc, b_kt, input_precision="ieee")

            p_ij = tl.where(valid, tl.math.exp2(b_s - b_lse[:, None]), 0.0)
            p_do = do + row_qh[:, None] * V + o_v[None, :]
            b_do = tl.load(p_do, mask=mask_mg[:, None], other=0.0)
            b_vt = tl.load(p_v, mask=mask_n[None, :], other=0.0)
            if not IS_FP32:
                b_do = b_do.to(tl.bfloat16)
                b_vt = b_vt.to(tl.bfloat16)
                b_dp = tl.dot(b_do, b_vt)
            else:
                b_do = b_do.to(tl.float32)
                b_vt = b_vt.to(tl.float32)
                b_dp = tl.dot(b_do, b_vt, input_precision="ieee")

            b_ds = p_ij * (b_dp - b_delta[:, None])
            if not IS_FP32:
                acc_dv += tl.dot(tl.trans(p_ij.to(tl.bfloat16)), b_do)
                acc_dk += tl.dot(tl.trans(b_ds.to(tl.bfloat16)), b_q_raw.to(tl.bfloat16))
            else:
                acc_dv += tl.dot(tl.trans(p_ij), b_do, input_precision="ieee")
                acc_dk += tl.dot(tl.trans(b_ds), b_q_raw.to(tl.float32), input_precision="ieee")

    acc_dk = acc_dk * scale
    p_dk = dk + ((i_b * T + cols[:, None]) * H + i_h) * K + o_k[None, :]
    p_dv = dv + ((i_b * T + cols[:, None]) * H + i_h) * V + o_v[None, :]
    tl.store(p_dk, acc_dk.to(p_dk.dtype.element_ty), mask=mask_n[:, None])
    tl.store(p_dv, acc_dv.to(p_dv.dtype.element_ty), mask=mask_n[:, None])


# ============================================================================
# 4. Online Teacher-Student KL Distillation Kernels for Lightning Indexer
# ============================================================================

@triton.jit
def _dsa_kl_teacher_student_fwd_bwd_rsi_kernel(
    q,
    k,
    lse,
    indices,
    idx_vals,
    kl_per_token,
    d_logits_out,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_b = tl.program_id(1)

    o_s = tl.arange(0, BS)
    num_sb = tl.cdiv(TOPK, BS)
    idx_base = indices + (i_b * T + i_t) * TOPK
    val_base = idx_vals + (i_b * T + i_t) * TOPK

    # Pass 1: student online log-sum-exp over selected topk indexer scores
    m_st = float("-inf")
    l_st = 0.0
    for i_sb in range(num_sb):
        r_offs = i_sb * BS + o_s
        mask_r = r_offs < TOPK
        b_idx = tl.load(idx_base + r_offs, mask=mask_r, other=-1)
        valid = mask_r & (b_idx >= 0) & (b_idx <= i_t)
        b_v = tl.load(val_base + r_offs, mask=valid, other=float("-inf")).to(tl.float32)
        b_v_log2 = b_v * 1.4426950408889634
        m_new = tl.maximum(m_st, tl.max(b_v_log2, axis=0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        l_st = l_st * tl.math.exp2(m_st - m_safe) + tl.sum(tl.where(valid, tl.math.exp2(b_v_log2 - m_safe), 0.0), axis=0)
        m_st = m_new
    lse_st_log2 = m_st + tl.math.log2(tl.maximum(l_st, 1e-30))

    # Pass 2: compute head-averaged teacher prob p_bar, KL loss, and scatter dI = q_student - p_bar
    o_k = tl.arange(0, K)
    scale_log2 = (1.0 / tl.sqrt(float(K))) * 1.4426950408889634
    rcp_hq = 1.0 / float(HQ)
    kl_acc = 0.0
    dlog_row = d_logits_out + (i_b * T + i_t) * T

    for i_sb in range(num_sb):
        r_offs = i_sb * BS + o_s
        mask_r = r_offs < TOPK
        b_idx = tl.load(idx_base + r_offs, mask=mask_r, other=-1)
        valid = mask_r & (b_idx >= 0) & (b_idx <= i_t)
        safe_idx = tl.where(valid, b_idx, 0)

        b_v = tl.load(val_base + r_offs, mask=valid, other=float("-inf")).to(tl.float32)
        b_v_log2 = b_v * 1.4426950408889634
        q_st = tl.where(valid, tl.math.exp2(b_v_log2 - lse_st_log2), 0.0)

        p_bar = tl.zeros([BS], dtype=tl.float32)
        for i_hq in range(HQ):
            i_h = i_hq // G
            p_q = q + ((i_b * T + i_t) * HQ + i_hq) * K + o_k
            b_q = tl.load(p_q).to(tl.float32) * scale_log2
            p_k = k + ((i_b * T + safe_idx[:, None]) * H + i_h) * K + o_k[None, :]
            b_k = tl.load(p_k, mask=valid[:, None], other=0.0).to(tl.float32)
            b_s = tl.sum(b_k * b_q[None, :], axis=1)
            b_lse = tl.load(lse + (i_b * T + i_t) * HQ + i_hq)
            p_h = tl.where(valid, tl.math.exp2(b_s - b_lse), 0.0)
            p_bar += p_h * rcp_hq

        log_p = tl.math.log(tl.maximum(p_bar, 1e-30))
        log_q = tl.math.log(tl.maximum(q_st, 1e-30))
        kl_acc += tl.sum(tl.where(valid & (p_bar > 0.0), p_bar * (log_p - log_q), 0.0), axis=0)

        b_di = tl.where(valid, q_st - p_bar, 0.0)
        tl.store(dlog_row + safe_idx, b_di, mask=valid)

    tl.store(kl_per_token + i_b * T + i_t, kl_acc)


@triton.jit
def _dsa_indexer_kl_bwd_dq_dw_rsi_kernel(
    q_idx,
    k_idx,
    w_idx,
    d_logits,
    tile_active,
    dq_idx,
    dw_idx,
    grad_scale,
    T: tl.constexpr,
    HI: tl.constexpr,
    DI: tl.constexpr,
    NB64: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_b = tl.program_id(1)
    BMH: tl.constexpr = BM * HI
    o_mh = tl.arange(0, BMH)
    o_m_local = o_mh // HI
    o_m_of_mh = i_m * BM + o_m_local
    mask_mh = o_m_of_mh < T
    i_m64 = (i_m * BM) // 64
    o_d = tl.arange(0, DI)
    o_n = tl.arange(0, BN)
    scale = 1.0 / tl.sqrt(float(DI))

    p_q = q_idx + i_b * (T * HI * DI) + (i_m * BMH + o_mh[:, None]) * DI + o_d[None, :]
    if HAS_W:
        p_w = w_idx + i_b * (T * HI) + i_m * BMH + o_mh
        b_w = tl.load(p_w, mask=mask_mh, other=0.0).to(tl.float32)
    else:
        b_w = tl.full([BMH], 1.0, dtype=tl.float32)

    acc_dq = tl.zeros([BMH, DI], dtype=tl.float32)
    acc_dw = tl.zeros([BMH], dtype=tl.float32)
    n_blocks = tl.cdiv(tl.minimum((i_m + 1) * BM, T), BN)
    ta_base = tile_active + (i_b * NB64 + i_m64) * NB64
    g_sc = tl.load(grad_scale).to(tl.float32)

    for i_n in range(0, n_blocks):
        i_n64 = (i_n * BN) // 64
        act = tl.load(ta_base + i_n64)
        if act != 0:
            cols = i_n * BN + o_n
            mask_n = cols < T
            b_q = tl.load(p_q, mask=mask_mh[:, None], other=0.0)
            p_k = k_idx + i_b * (T * DI) + cols[None, :] * DI + o_d[:, None]
            b_kt = tl.load(p_k, mask=mask_n[None, :], other=0.0)
            if not IS_FP32:
                b_q = b_q.to(tl.bfloat16)
                b_kt = b_kt.to(tl.bfloat16)
                b_s = tl.dot(b_q, b_kt) * scale
            else:
                b_q = b_q.to(tl.float32)
                b_kt = b_kt.to(tl.float32)
                b_s = tl.dot(b_q, b_kt, input_precision="ieee") * scale

            p_di = d_logits + i_b * (T * T) + o_m_of_mh[:, None] * T + cols[None, :]
            b_di = tl.load(p_di, mask=mask_mh[:, None] & mask_n[None, :], other=0.0).to(tl.float32) * g_sc

            relu_s = tl.maximum(b_s, 0.0)
            acc_dw += tl.sum(b_di * relu_s, axis=1)
            b_dz = tl.where(b_s > 0.0, b_di * b_w[:, None] * scale, 0.0)
            if not IS_FP32:
                acc_dq += tl.dot(b_dz.to(tl.bfloat16), tl.trans(b_kt))
            else:
                acc_dq += tl.dot(b_dz, tl.trans(b_kt), input_precision="ieee")

    p_dq = dq_idx + i_b * (T * HI * DI) + (i_m * BMH + o_mh[:, None]) * DI + o_d[None, :]
    tl.store(p_dq, acc_dq.to(p_dq.dtype.element_ty), mask=mask_mh[:, None])
    if HAS_W:
        p_dw = dw_idx + i_b * (T * HI) + i_m * BMH + o_mh
        tl.store(p_dw, acc_dw.to(p_dw.dtype.element_ty), mask=mask_mh)


@triton.jit
def _dsa_indexer_kl_bwd_dk_rsi_kernel(
    q_idx,
    k_idx,
    w_idx,
    d_logits,
    tile_active,
    dk_idx,
    grad_scale,
    T: tl.constexpr,
    HI: tl.constexpr,
    DI: tl.constexpr,
    NB64: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_b = tl.program_id(1)
    BMH: tl.constexpr = BM * HI
    o_mh = tl.arange(0, BMH)
    o_d = tl.arange(0, DI)
    cols = i_n * BN + tl.arange(0, BN)
    mask_n = cols < T
    i_n64 = (i_n * BN) // 64
    scale = 1.0 / tl.sqrt(float(DI))

    p_k = k_idx + i_b * (T * DI) + cols[None, :] * DI + o_d[:, None]
    acc_dk = tl.zeros([BN, DI], dtype=tl.float32)
    m_start = (i_n * BN) // BM
    m_limit = tl.cdiv(T, BM)
    g_sc = tl.load(grad_scale).to(tl.float32)

    for i_m in range(m_start, m_limit):
        i_m64 = (i_m * BM) // 64
        act = tl.load(tile_active + (i_b * NB64 + i_m64) * NB64 + i_n64)
        if act != 0:
            o_m_of_mh = i_m * BM + (o_mh // HI)
            mask_mh = o_m_of_mh < T
            p_q = q_idx + i_b * (T * HI * DI) + (i_m * BMH + o_mh[:, None]) * DI + o_d[None, :]
            b_q = tl.load(p_q, mask=mask_mh[:, None], other=0.0)
            b_kt = tl.load(p_k, mask=mask_n[None, :], other=0.0)
            if not IS_FP32:
                b_q = b_q.to(tl.bfloat16)
                b_kt = b_kt.to(tl.bfloat16)
                b_s = tl.dot(b_q, b_kt) * scale
            else:
                b_q = b_q.to(tl.float32)
                b_kt = b_kt.to(tl.float32)
                b_s = tl.dot(b_q, b_kt, input_precision="ieee") * scale

            if HAS_W:
                p_w = w_idx + i_b * (T * HI) + i_m * BMH + o_mh
                b_w = tl.load(p_w, mask=mask_mh, other=0.0).to(tl.float32)
            else:
                b_w = tl.full([BMH], 1.0, dtype=tl.float32)

            p_di = d_logits + i_b * (T * T) + o_m_of_mh[:, None] * T + cols[None, :]
            b_di = tl.load(p_di, mask=mask_mh[:, None] & mask_n[None, :], other=0.0).to(tl.float32) * g_sc
            b_dz = tl.where(b_s > 0.0, b_di * b_w[:, None] * scale, 0.0)
            if not IS_FP32:
                acc_dk += tl.dot(tl.trans(b_dz.to(tl.bfloat16)), b_q)
            else:
                acc_dk += tl.dot(tl.trans(b_dz), b_q, input_precision="ieee")

    p_dk = dk_idx + i_b * (T * DI) + cols[:, None] * DI + o_d[None, :]
    tl.store(p_dk, acc_dk.to(p_dk.dtype.element_ty), mask=mask_n[:, None])


# ============================================================================
# 5. Single-Pass Multi-Warp Online Sparse Decoding Kernels
# ============================================================================

@triton.jit
def _dsa_decode_indexer_rsi_kernel(
    q_idx,
    k_idx_cache,
    w_idx,
    logits_out,
    T_Q: tl.constexpr,
    T_KV: tl.constexpr,
    HI: tl.constexpr,
    DI: tl.constexpr,
    BM_Q: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_bm = tl.program_id(1)
    i_b = i_bm // tl.cdiv(T_Q, BM_Q)
    i_m = i_bm % tl.cdiv(T_Q, BM_Q)

    BMH: tl.constexpr = BM_Q * HI
    o_mh = tl.arange(0, BMH)
    o_m_local = o_mh // HI
    o_m = i_m * BM_Q + o_m_local
    mask_mh = (o_m < T_Q) & (o_m_local < BM_Q)
    o_d = tl.arange(0, DI)
    cols = i_n * BN + tl.arange(0, BN)
    mask_n = cols < T_KV
    scale = 1.0 / tl.sqrt(float(DI))

    p_q = q_idx + i_b * (T_Q * HI * DI) + (i_m * BMH + o_mh[:, None]) * DI + o_d[None, :]
    b_q = tl.load(p_q, mask=mask_mh[:, None], other=0.0).to(tl.bfloat16)

    if HAS_W:
        p_w = w_idx + i_b * (T_Q * HI) + i_m * BMH + o_mh
        b_w = tl.load(p_w, mask=mask_mh, other=0.0).to(tl.float32) * scale
    else:
        b_w = tl.full([BMH], scale, dtype=tl.float32)

    p_k = k_idx_cache + i_b * (T_KV * DI) + cols[None, :] * DI + o_d[:, None]
    b_k = tl.load(p_k, mask=mask_n[None, :], other=0.0).to(tl.bfloat16)
    b_s = tl.dot(b_q, b_k)
    b_s = tl.maximum(b_s, 0.0) * b_w[:, None]
    b_s3d = tl.reshape(b_s, [BM_Q, HI, BN])
    b_acc = tl.sum(b_s3d, axis=1)

    o_q = i_m * BM_Q + tl.arange(0, BM_Q)
    mask_q = o_q < T_Q
    q_global_pos = (T_KV - T_Q) + o_q
    causal = q_global_pos[:, None] >= cols[None, :]
    b_acc = tl.where(causal & mask_q[:, None] & mask_n[None, :], b_acc, float("-inf"))
    p_out = logits_out + i_b * (T_Q * T_KV) + o_q[:, None] * T_KV + cols[None, :]
    tl.store(p_out, b_acc, mask=mask_q[:, None] & mask_n[None, :])


@triton.jit
def _dsa_decode_gqa_readout_rsi_kernel(
    q,
    k_cache,
    v_cache,
    indices,
    o,
    T_Q: tl.constexpr,
    T_KV: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_g = tl.arange(0, G_PAD)
    mask_g = o_g < G
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    scale_log2 = (1.0 / tl.sqrt(float(K))) * 1.4426950408889634
    q_global_pos = (T_KV - T_Q) + i_t

    p_q = q + ((i_b * T_Q + i_t) * HQ + i_h * G + o_g[:, None]) * K + o_k[None, :]
    b_q = (tl.load(p_q, mask=mask_g[:, None], other=0.0).to(tl.float32) * scale_log2).to(tl.bfloat16)

    m_i = tl.full([G_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([G_PAD], dtype=tl.float32)
    acc_o = tl.zeros([G_PAD, V], dtype=tl.float32)

    o_s = tl.arange(0, BS)
    num_sb = tl.cdiv(TOPK, BS)
    idx_base = indices + (i_b * T_Q + i_t) * TOPK
    kv_base_k = k_cache + (i_b * T_KV * H + i_h) * K
    kv_base_v = v_cache + (i_b * T_KV * H + i_h) * V

    for i_sb in range(num_sb):
        r_offs = i_sb * BS + o_s
        mask_r = r_offs < TOPK
        b_idx = tl.load(idx_base + r_offs, mask=mask_r, other=-1)
        valid = mask_r & (b_idx >= 0) & (b_idx <= q_global_pos)
        safe_idx = tl.where(valid, b_idx, 0)

        p_k = kv_base_k + safe_idx[None, :] * (H * K) + o_k[:, None]
        b_k = tl.load(p_k, mask=valid[None, :], other=0.0).to(tl.bfloat16)
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(valid[None, :] & mask_g[:, None], b_s, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(b_s, axis=1))
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        alpha = tl.math.exp2(m_i - m_ij_safe)
        p_ij = tl.math.exp2(b_s - m_ij_safe[:, None])
        l_i = l_i * alpha + tl.sum(p_ij, axis=1)
        acc_o = acc_o * alpha[:, None]

        p_v = kv_base_v + safe_idx[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_o += tl.dot(p_ij.to(tl.bfloat16), b_v)
        m_i = m_ij

    l_rcp = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
    b_o = acc_o * l_rcp[:, None]
    p_o = o + ((i_b * T_Q + i_t) * HQ + i_h * G + o_g[:, None]) * V + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_g[:, None])


# ============================================================================
# 6. Superfused Periphery Kernels (L2Norm, Softplus Gate, RMSNorm, Gated RMSNorm)
# ============================================================================

@triton.jit
def _dsa_l2norm_fwd_rsi_kernel(
    x,
    y,
    rstd,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr = 1.0,
):
    i_row = tl.program_id(0)
    o_d = tl.arange(0, D)
    b_x = tl.load(x + i_row * D + o_d).to(tl.float32)
    b_rstd = tl.rsqrt(tl.sum(b_x * b_x, axis=0) + EPS) * SCALE
    tl.store(rstd + i_row, b_rstd)
    tl.store(y + i_row * D + o_d, (b_x * b_rstd).to(y.dtype.element_ty))


@triton.jit
def _dsa_l2norm_bwd_rsi_kernel(
    y,
    rstd,
    dy,
    dx,
    D: tl.constexpr,
    INV_SCALE_SQ: tl.constexpr = 1.0,
):
    i_row = tl.program_id(0)
    o_d = tl.arange(0, D)
    b_y = tl.load(y + i_row * D + o_d).to(tl.float32)
    b_dy = tl.load(dy + i_row * D + o_d).to(tl.float32)
    b_rstd = tl.load(rstd + i_row).to(tl.float32)
    proj = tl.sum(b_y * b_dy, axis=0) * INV_SCALE_SQ
    b_dx = (b_dy - b_y * proj) * b_rstd
    tl.store(dx + i_row * D + o_d, b_dx.to(dx.dtype.element_ty))


@triton.jit
def _dsa_w_softplus_fwd_rsi_kernel(
    w_raw,
    w_out,
    N: tl.constexpr,
    BD: tl.constexpr,
):
    i_b = tl.program_id(0)
    offs = i_b * BD + tl.arange(0, BD)
    mask = offs < N
    b_x = tl.load(w_raw + offs, mask=mask, other=0.0).to(tl.float32)
    b_w = tl.where(b_x > 15.0, b_x, tl.math.log(1.0 + tl.exp(b_x)))
    tl.store(w_out + offs, b_w.to(w_out.dtype.element_ty), mask=mask)


@triton.jit
def _dsa_w_softplus_bwd_rsi_kernel(
    w_raw,
    dw_out,
    dw_raw,
    N: tl.constexpr,
    BD: tl.constexpr,
):
    i_b = tl.program_id(0)
    offs = i_b * BD + tl.arange(0, BD)
    mask = offs < N
    b_x = tl.load(w_raw + offs, mask=mask, other=0.0).to(tl.float32)
    b_dw = tl.load(dw_out + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(b_x)
    tl.store(dw_raw + offs, (b_dw * sig).to(dw_raw.dtype.element_ty), mask=mask)


@triton.jit
def _dsa_rmsnorm_fwd_rsi_kernel(
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
def _dsa_rmsnorm_bwd_rsi_kernel(
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
def _dsa_rmsnorm_gated_fwd_rsi_kernel(
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
    silu_g = b_g * tl.sigmoid(b_g)
    b_y = b_x * b_rstd * b_w * silu_g
    tl.store(y + i_row * D + o_d, b_y.to(y.dtype.element_ty))


@triton.jit
def _dsa_rmsnorm_gated_bwd_rsi_kernel(
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
    inv_d = 1.0 / float(D)

    for r_off in range(ROWS_PER_PROG):
        row = row_start + r_off
        if row < N_ROWS:
            b_x = tl.load(x + row * D + o_d).to(tl.float32)
            b_g = tl.load(g + row * D + o_d).to(tl.float32)
            b_dy = tl.load(dy + row * D + o_d).to(tl.float32)
            b_rstd = tl.load(rstd + row).to(tl.float32)
            sig_g = tl.sigmoid(b_g)
            silu_g = b_g * sig_g
            dsilu_g = sig_g * (1.0 + b_g * (1.0 - sig_g))
            x_hat = b_x * b_rstd
            norm_x = x_hat * b_w
            b_dg = b_dy * norm_x * dsilu_g
            dy_gated = b_dy * silu_g
            b_dw += dy_gated * x_hat
            wdy = b_w * dy_gated
            c1 = tl.sum(x_hat * wdy, axis=0) * inv_d
            b_dx = (wdy - x_hat * c1) * b_rstd
            tl.store(dx + row * D + o_d, b_dx.to(dx.dtype.element_ty))
            tl.store(dg + row * D + o_d, b_dg.to(dg.dtype.element_ty))

    tl.store(dw_part + i_p * D + o_d, b_dw)


# ============================================================================
# 7. Differentiable Autograd Function & Reference FP64 Oracle
# ============================================================================

def build_dsa_bitmask_rsi(
    indices: torch.Tensor,
    T: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, topk = indices.shape
    NB64 = triton.cdiv(T, 64)
    bitmask = torch.zeros((B, T, NB64), device=indices.device, dtype=torch.uint64)
    tile_active = torch.zeros((B, NB64, NB64), device=indices.device, dtype=torch.int32)
    BS = min(64, triton.next_power_of_2(max(16, topk)))
    _dsa_build_bitmask_rsi_kernel[(T, B)](
        indices,
        bitmask,
        tile_active,
        T=T,
        TOPK=topk,
        NB64=NB64,
        BS=BS,
        num_warps=2,
    )
    return bitmask, tile_active


def dsa_indexer_rsi(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    w_idx: Optional[torch.Tensor] = None,
    topk: int = 256,
    return_vals: bool = False,
    mutate_mode: int = 0,
):
    B, T, HI, DI = q_idx.shape
    IS_FP32 = q_idx.dtype == torch.float32
    logits = torch.empty((B, T, T), device=q_idx.device, dtype=torch.float32)
    if mutate_mode == 2:
        # M2 mutation: drop ReLU in Lightning Indexer
        scale = DI ** -0.5
        qf = q_idx.float()
        kf = k_idx.float()
        s = torch.einsum("bthd,bsd->bhts", qf, kf) * scale
        if w_idx is not None:
            logits = torch.einsum("bhts,bth->bts", s, w_idx.float())
        else:
            logits = s.sum(dim=1)
        i_t = torch.arange(T, device=q_idx.device)
        logits = logits.masked_fill(i_t[:, None] < i_t[None, :], float("-inf"))
    else:
        BM = max(1, (32 if IS_FP32 else 64) // HI)
        BN = 32 if IS_FP32 else 64
        _dsa_indexer_mqa_fwd_rsi_kernel[(triton.cdiv(T, BM), B)](
            q_idx,
            k_idx,
            w_idx if w_idx is not None else q_idx,
            logits,
            T=T,
            HI=HI,
            DI=DI,
            BM=BM,
            BN=BN,
            HAS_W=(w_idx is not None),
            IS_FP32=IS_FP32,
            num_warps=4,
            num_stages=1 if IS_FP32 else 2,
        )
    S = min(topk, T)
    vals, idx = torch.topk(logits, S, dim=-1)
    idx = idx.masked_fill(torch.isinf(vals), -1)
    if topk > S:
        idx = torch.cat([idx, idx.new_full((B, T, topk - S), -1)], dim=-1)
        vals = torch.cat([vals, vals.new_full((B, T, topk - S), float("-inf"))], dim=-1)
    if return_vals:
        return idx, vals
    return idx


class DSAChunkRSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_idx: Optional[torch.Tensor],
        k_idx: Optional[torch.Tensor],
        w_idx: Optional[torch.Tensor],
        precomputed_indices: Optional[torch.Tensor],
        precomputed_bitmask: Optional[torch.Tensor],
        precomputed_tile_active: Optional[torch.Tensor],
        topk: int,
        train_indexer: bool,
        mutate_mode: int,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        B, T, HQ, K = q.shape
        H = k.shape[2]
        V = v.shape[3]
        G = HQ // H
        IS_FP32 = q.dtype == torch.float32
        NB64 = triton.cdiv(T, 64)

        idx_vals = None
        if precomputed_indices is not None:
            indices = precomputed_indices
            if precomputed_bitmask is not None and precomputed_tile_active is not None:
                bitmask = precomputed_bitmask
                tile_active = precomputed_tile_active
            else:
                bitmask, tile_active = build_dsa_bitmask_rsi(indices, T)
        else:
            q_idx = q_idx.contiguous()
            k_idx = k_idx.contiguous()
            w_idx = w_idx.contiguous() if w_idx is not None else None
            indices, idx_vals = dsa_indexer_rsi(
                q_idx, k_idx, w_idx, topk=topk, return_vals=True, mutate_mode=mutate_mode
            )
            bitmask, tile_active = build_dsa_bitmask_rsi(indices, T)

        if mutate_mode == 1:
            # M1 mutation: drop sparse bitmask selection and attend over all causal tokens
            bitmask = torch.full_like(bitmask, 0xFFFFFFFFFFFFFFFF)
            tile_active = torch.ones_like(tile_active)

        BMG = 32 if (K <= 64 and not IS_FP32) else 16
        BMT = max(1, BMG // G)
        BMG_ACT = max(16, BMT * G)
        BN = 32 if (K <= 64 and not IS_FP32) else 16

        o = torch.empty((B, T, HQ, V), device=q.device, dtype=q.dtype)
        lse = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

        _dsa_gqa_bitmask_fwd_rsi_kernel[(triton.cdiv(T, BMT), B * H)](
            q,
            k,
            v,
            bitmask,
            tile_active,
            o,
            lse,
            T=T,
            HQ=HQ,
            H=H,
            G=G,
            K=K,
            V=V,
            NB64=NB64,
            BMT=BMT,
            BMG=BMG_ACT,
            BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4,
            num_stages=1,
        )

        d_logits = None
        if train_indexer and q_idx is not None and idx_vals is not None:
            kl_per_token = torch.empty((B, T), device=q.device, dtype=torch.float32)
            d_logits = torch.zeros((B, T, T), device=q.device, dtype=torch.float32)
            BS_KL = min(64, triton.next_power_of_2(max(16, topk)))
            _dsa_kl_teacher_student_fwd_bwd_rsi_kernel[(T, B)](
                q,
                k,
                lse,
                indices,
                idx_vals,
                kl_per_token,
                d_logits,
                T=T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                TOPK=topk,
                BS=BS_KL,
                IS_FP32=IS_FP32,
                num_warps=2,
            )
            if mutate_mode == 3:
                # M3 mutation: invert KL teacher-student gradient sign
                d_logits = -d_logits
            kl_loss = kl_per_token.mean()
        else:
            kl_loss = torch.zeros((), device=q.device, dtype=torch.float32)

        ctx.save_for_backward(
            q,
            k,
            v,
            o,
            lse,
            bitmask,
            tile_active,
            q_idx if (train_indexer and q_idx is not None) else torch.empty(0, device=q.device),
            k_idx if (train_indexer and k_idx is not None) else torch.empty(0, device=q.device),
            w_idx if (train_indexer and w_idx is not None) else torch.empty(0, device=q.device),
            d_logits if d_logits is not None else torch.empty(0, device=q.device),
        )
        ctx.train_indexer = train_indexer and (q_idx is not None) and (d_logits is not None)
        ctx.has_w = w_idx is not None
        ctx.topk = topk
        ctx.mark_non_differentiable(indices, bitmask, tile_active)
        return o, kl_loss, indices, bitmask, tile_active

    @staticmethod
    def backward(ctx, do: torch.Tensor, dkl: Optional[torch.Tensor], _d_idx, _d_bm, _d_ta):
        (
            q,
            k,
            v,
            o,
            lse,
            bitmask,
            tile_active,
            q_idx,
            k_idx,
            w_idx,
            d_logits,
        ) = ctx.saved_tensors
        do = do.contiguous()
        B, T, HQ, K = q.shape
        H = k.shape[2]
        V = v.shape[3]
        G = HQ // H
        IS_FP32 = q.dtype == torch.float32
        NB64 = triton.cdiv(T, 64)

        delta = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
        _dsa_bwd_prep_delta_rsi_kernel[(triton.cdiv(T, 64), B * HQ)](
            o, do, delta, T=T, HQ=HQ, V=V, BM=64, num_warps=4
        )

        BMG = 32 if (K <= 64 and not IS_FP32) else 16
        BMT = max(1, BMG // G)
        BMG_ACT = max(16, BMT * G)
        BN = 32 if (K <= 64 and not IS_FP32) else 16

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        _dsa_gqa_bitmask_bwd_dq_rsi_kernel[(triton.cdiv(T, BMT), B * H)](
            q,
            k,
            v,
            do,
            lse,
            delta,
            bitmask,
            tile_active,
            dq,
            T=T,
            HQ=HQ,
            H=H,
            G=G,
            K=K,
            V=V,
            NB64=NB64,
            BMT=BMT,
            BMG=BMG_ACT,
            BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4,
            num_stages=1,
        )

        _dsa_gqa_bitmask_bwd_dkv_rsi_kernel[(triton.cdiv(T, BN), B * H)](
            q,
            k,
            v,
            do,
            lse,
            delta,
            bitmask,
            tile_active,
            dk,
            dv,
            T=T,
            HQ=HQ,
            H=H,
            G=G,
            K=K,
            V=V,
            NB64=NB64,
            BMT=BMT,
            BMG=BMG_ACT,
            BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4,
            num_stages=1,
        )

        dq_idx = dk_idx = dw_idx = None
        if ctx.train_indexer:
            _, _, HI, DI = q_idx.shape
            IS_IDX_FP32 = q_idx.dtype == torch.float32
            dq_idx = torch.empty_like(q_idx)
            dk_idx = torch.empty_like(k_idx)
            dw_idx = torch.empty_like(w_idx) if ctx.has_w else torch.empty(0, device=q.device)
            dkl_val = dkl.float() if dkl is not None else torch.ones((), device=q.device, dtype=torch.float32)
            grad_scale = (dkl_val / float(B * T)).contiguous()

            BM_I = max(1, (16 if IS_IDX_FP32 else 32) // HI)
            BN_I = 16 if IS_IDX_FP32 else 32
            _dsa_indexer_kl_bwd_dq_dw_rsi_kernel[(triton.cdiv(T, BM_I), B)](
                q_idx,
                k_idx,
                w_idx if ctx.has_w else q_idx,
                d_logits,
                tile_active,
                dq_idx,
                dw_idx if ctx.has_w else q_idx,
                grad_scale,
                T=T,
                HI=HI,
                DI=DI,
                NB64=NB64,
                BM=BM_I,
                BN=BN_I,
                HAS_W=ctx.has_w,
                IS_FP32=IS_IDX_FP32,
                num_warps=4,
                num_stages=1,
            )
            _dsa_indexer_kl_bwd_dk_rsi_kernel[(triton.cdiv(T, BN_I), B)](
                q_idx,
                k_idx,
                w_idx if ctx.has_w else q_idx,
                d_logits,
                tile_active,
                dk_idx,
                grad_scale,
                T=T,
                HI=HI,
                DI=DI,
                NB64=NB64,
                BM=BM_I,
                BN=BN_I,
                HAS_W=ctx.has_w,
                IS_FP32=IS_IDX_FP32,
                num_warps=4,
                num_stages=1,
            )
            if not ctx.has_w:
                dw_idx = None

        return dq, dk, dv, dq_idx, dk_idx, dw_idx, None, None, None, None, None, None


def dsa_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: Optional[torch.Tensor] = None,
    k_idx: Optional[torch.Tensor] = None,
    w_idx: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    bitmask: Optional[torch.Tensor] = None,
    tile_active: Optional[torch.Tensor] = None,
    topk: int = 256,
    train_indexer: bool = True,
    return_indices: bool = False,
    return_kl_loss: bool = False,
    return_shared_state: bool = False,
    mutate_mode: int = 0,
):
    o, kl_loss, idx_out, bm_out, ta_out = DSAChunkRSIFunction.apply(
        q,
        k,
        v,
        q_idx,
        k_idx,
        w_idx,
        indices,
        bitmask,
        tile_active,
        topk,
        train_indexer,
        mutate_mode,
    )
    if return_shared_state:
        return o, kl_loss, idx_out, bm_out, ta_out
    if return_indices and return_kl_loss:
        return o, idx_out, kl_loss
    if return_indices:
        return o, idx_out
    if return_kl_loss:
        return o, kl_loss
    return o


def dsa_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    w_idx: Optional[torch.Tensor] = None,
    topk: int = 256,
    return_kl_loss: bool = True,
):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[3]
    G = HQ // H
    _, _, HI, DI = q_idx.shape

    qd = q.to(torch.float64)
    kd = k.to(torch.float64)
    vd = v.to(torch.float64)
    qid = q_idx.to(torch.float64)
    kid = k_idx.to(torch.float64)
    wid = w_idx.to(torch.float64) if w_idx is not None else None

    scale_idx = DI ** -0.5
    s_idx = torch.einsum("bthd,bsd->bhts", qid, kid) * scale_idx
    s_idx = F.relu(s_idx)
    if wid is not None:
        logits_idx = torch.einsum("bhts,bth->bts", s_idx, wid)
    else:
        logits_idx = s_idx.sum(dim=1)

    i_t = torch.arange(T, device=q.device)
    causal_mask = i_t[:, None] >= i_t[None, :]
    logits_masked = logits_idx.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
    S = min(topk, T)
    vals, idx = torch.topk(logits_masked.detach(), S, dim=-1)
    valid_sel = ~torch.isinf(vals)
    safe_idx = idx.masked_fill(~valid_sel, T)

    sel_mask = torch.zeros((B, T, T + 1), device=q.device, dtype=torch.bool)
    sel_mask.scatter_(2, safe_idx, valid_sel)
    sel_mask = sel_mask[:, :, :T] & causal_mask.unsqueeze(0)

    kd_rep = kd.repeat_interleave(G, dim=2)
    vd_rep = vd.repeat_interleave(G, dim=2)
    scale_attn = K ** -0.5
    s_attn = torch.einsum("bthd,bshd->bhts", qd, kd_rep) * scale_attn
    s_attn = s_attn.masked_fill(~sel_mask.unsqueeze(1), float("-inf"))
    p_attn = torch.softmax(s_attn, dim=-1)
    o = torch.einsum("bhts,bshd->bthd", p_attn, vd_rep)

    if not return_kl_loss:
        return o

    p_bar = p_attn.detach().mean(dim=1)  # [B, T, T]
    logits_sel = logits_idx.masked_fill(~sel_mask, float("-inf"))
    log_q_idx = F.log_softmax(logits_sel, dim=-1)
    log_p_bar = torch.log(torch.clamp(p_bar, min=1e-30))
    kl_mat = torch.where(sel_mask & (p_bar > 0.0), p_bar * (log_p_bar - log_q_idx), torch.zeros_like(p_bar))
    kl_loss = kl_mat.sum(dim=-1).mean()
    return o, kl_loss


# ============================================================================
# 8. Single-Pass Multi-Warp Sparse Decoding (dsa_decode_rsi)
# ============================================================================

def dsa_decode_rsi(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx_cache: torch.Tensor,
    w_idx: Optional[torch.Tensor] = None,
    topk: int = 256,
    return_indices: bool = False,
):
    B, T_Q, HQ, K = q.shape
    T_KV = k_cache.shape[1]
    H = k_cache.shape[2]
    V = v_cache.shape[3]
    G = HQ // H
    G_PAD = max(16, triton.next_power_of_2(G))
    _, _, HI, DI = q_idx.shape

    BM_Q = max(2, 16 // HI)
    BN = 64
    logits = torch.empty((B, T_Q, T_KV), device=q.device, dtype=torch.float32)
    grid_idx = (triton.cdiv(T_KV, BN), B * triton.cdiv(T_Q, BM_Q))
    _dsa_decode_indexer_rsi_kernel[grid_idx](
        q_idx,
        k_idx_cache,
        w_idx if w_idx is not None else q_idx,
        logits,
        T_Q=T_Q,
        T_KV=T_KV,
        HI=HI,
        DI=DI,
        BM_Q=BM_Q,
        BN=BN,
        HAS_W=(w_idx is not None),
        num_warps=4,
        num_stages=2,
    )
    S = min(topk, T_KV)
    vals, indices = torch.topk(logits, S, dim=-1)
    indices = indices.masked_fill(torch.isinf(vals), -1)
    if topk > S:
        indices = torch.cat([indices, indices.new_full((B, T_Q, topk - S), -1)], dim=-1)

    o = torch.empty((B, T_Q, HQ, V), device=q.device, dtype=q.dtype)
    BS = min(64, max(16, triton.next_power_of_2(min(32, topk))))
    _dsa_decode_gqa_readout_rsi_kernel[(T_Q, B * H)](
        q,
        k_cache,
        v_cache,
        indices,
        o,
        T_Q=T_Q,
        T_KV=T_KV,
        HQ=HQ,
        H=H,
        G=G,
        G_PAD=G_PAD,
        K=K,
        V=V,
        TOPK=topk,
        BS=BS,
        num_warps=4,
        num_stages=2,
    )
    if return_indices:
        return o, indices
    return o


# ============================================================================
# 9. Superfused Periphery Autograd Wrappers & DeepSeekSparseAttentionBlockRSI
# ============================================================================

class FusedDSAL2NormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-6, scale: float = 1.0):
        x_c = x.contiguous()
        D = x_c.shape[-1]
        n_rows = x_c.numel() // D
        y = torch.empty_like(x_c)
        rstd = torch.empty((n_rows,), device=x_c.device, dtype=torch.float32)
        _dsa_l2norm_fwd_rsi_kernel[(n_rows,)](x_c, y, rstd, D=D, EPS=eps, SCALE=scale, num_warps=2)
        ctx.save_for_backward(y, rstd)
        ctx.inv_scale_sq = 1.0 / (scale * scale)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        y, rstd = ctx.saved_tensors
        dy_c = dy.contiguous()
        D = y.shape[-1]
        n_rows = y.numel() // D
        dx = torch.empty_like(y)
        _dsa_l2norm_bwd_rsi_kernel[(n_rows,)](y, rstd, dy_c, dx, D=D, INV_SCALE_SQ=ctx.inv_scale_sq, num_warps=2)
        return dx, None, None


class FusedDSASoftplusGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_raw: torch.Tensor):
        w_c = w_raw.contiguous()
        N = w_c.numel()
        w_out = torch.empty_like(w_c)
        BD = 256
        _dsa_w_softplus_fwd_rsi_kernel[(triton.cdiv(N, BD),)](w_c, w_out, N=N, BD=BD, num_warps=2)
        ctx.save_for_backward(w_c)
        return w_out

    @staticmethod
    def backward(ctx, dw_out: torch.Tensor):
        (w_c,) = ctx.saved_tensors
        dw_c = dw_out.contiguous()
        N = w_c.numel()
        dw_raw = torch.empty_like(w_c)
        BD = 256
        _dsa_w_softplus_bwd_rsi_kernel[(triton.cdiv(N, BD),)](w_c, dw_c, dw_raw, N=N, BD=BD, num_warps=2)
        return dw_raw


class FusedDSARMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        x_c = x.contiguous()
        D = x_c.shape[-1]
        n_rows = x_c.numel() // D
        y = torch.empty_like(x_c)
        rstd = torch.empty((n_rows,), device=x_c.device, dtype=torch.float32)
        _dsa_rmsnorm_fwd_rsi_kernel[(n_rows,)](x_c, weight, y, rstd, D=D, EPS=eps, num_warps=4)
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
        _dsa_rmsnorm_bwd_rsi_kernel[(n_progs,)](
            x_c, weight, rstd, dy_c, dx, dw_part, n_rows, D=D, ROWS_PER_PROG=rows_per_prog, num_warps=4
        )
        dw = dw_part.sum(dim=0).to(weight.dtype)
        return dx, dw, None


class FusedDSARMSNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        x_c = x.contiguous()
        g_c = g.contiguous()
        D = x_c.shape[-1]
        n_rows = x_c.numel() // D
        y = torch.empty_like(x_c)
        rstd = torch.empty((n_rows,), device=x_c.device, dtype=torch.float32)
        _dsa_rmsnorm_gated_fwd_rsi_kernel[(n_rows,)](x_c, g_c, weight, y, rstd, D=D, EPS=eps, num_warps=4)
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
        _dsa_rmsnorm_gated_bwd_rsi_kernel[(n_progs,)](
            x_c, g_c, weight, rstd, dy_c, dx, dg, dw_part, n_rows, D=D, ROWS_PER_PROG=rows_per_prog, num_warps=4
        )
        dw = dw_part.sum(dim=0).to(weight.dtype)
        return dx, dg, dw, None


class DeepSeekSparseAttentionBlockRSI(nn.Module):
    """
    5-Stage Progressive Ablation Block for DeepSeek Sparse Attention (DSA) + GLM-5.2 IndexShare:
      - Stage A0: Upstream fla/ops/dsa/naive.py (naive_dsa_indexer + naive_dsa) + unfused projections & norms
      - Stage A1: Zero-spill 64-Bit Bitmask Flash-DSA + Flattened-(BM*HI) MQA Lightning Indexer + Online KL Distillation
      - Stage A2: + Fused Triton L2Norm on (q, k, q_idx, k_idx) + Fused Softplus on w_idx
      - Stage A3: + Single-GEMM Packed Projection (qkv_idx_gate_proj) + Fused Pre-RMSNorm & Gated RMSNorm
      - Stage A4: + GLM-5.2 IndexShare (reusing precomputed (indices, bitmask, tile_active) across shared layers)
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_kv_heads: int = 2,
        head_dim: int = 64,
        n_idx_heads: int = 8,
        idx_dim: int = 64,
        topk: int = 256,
        stage: str = "A4",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_idx_heads = n_idx_heads
        self.idx_dim = idx_dim
        self.topk = topk
        self.stage = stage

        self.q_dim = n_heads * head_dim
        self.kv_dim = n_kv_heads * head_dim
        self.qi_dim = n_idx_heads * idx_dim
        self.ki_dim = idx_dim
        self.wi_dim = n_idx_heads

        self.pre_norm_weight = nn.Parameter(torch.ones(d_model))
        self.post_norm_weight = nn.Parameter(torch.ones(self.q_dim))

        self.q_proj = nn.Linear(d_model, self.q_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.qi_proj = nn.Linear(d_model, self.qi_dim, bias=False)
        self.ki_proj = nn.Linear(d_model, self.ki_dim, bias=False)
        self.wi_proj = nn.Linear(d_model, self.wi_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.q_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, d_model, bias=False)

        tot_out = self.q_dim + 2 * self.kv_dim + self.qi_dim + self.ki_dim + self.wi_dim + self.q_dim
        self.packed_proj = nn.Linear(d_model, tot_out, bias=False)
        self.sync_packed_weights()

    @torch.no_grad()
    def sync_packed_weights(self):
        self.packed_proj.weight.copy_(
            torch.cat(
                [
                    self.q_proj.weight,
                    self.k_proj.weight,
                    self.v_proj.weight,
                    self.qi_proj.weight,
                    self.ki_proj.weight,
                    self.wi_proj.weight,
                    self.g_proj.weight,
                ],
                dim=0,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        shared_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        return_shared_state: bool = False,
    ):
        B, T, D = x.shape
        if self.stage in ("A0", "A1", "A2"):
            xf = x.float()
            rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            h = (xf * rstd).to(x.dtype) * self.pre_norm_weight
            q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim)
            k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim)
            v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim)
            q_idx = self.qi_proj(h).view(B, T, self.n_idx_heads, self.idx_dim)
            k_idx = self.ki_proj(h).view(B, T, self.idx_dim)
            w_raw = self.wi_proj(h).view(B, T, self.n_idx_heads)
            g_out = self.g_proj(h)
        else:
            h = FusedDSARMSNormFunction.apply(x, self.pre_norm_weight, 1e-6)
            packed = self.packed_proj(h)
            s0 = self.q_dim
            s1 = s0 + self.kv_dim
            s2 = s1 + self.kv_dim
            s3 = s2 + self.qi_dim
            s4 = s3 + self.ki_dim
            s5 = s4 + self.wi_dim
            q = packed[..., :s0].contiguous().view(B, T, self.n_heads, self.head_dim)
            k = packed[..., s0:s1].contiguous().view(B, T, self.n_kv_heads, self.head_dim)
            v = packed[..., s1:s2].contiguous().view(B, T, self.n_kv_heads, self.head_dim)
            q_idx = packed[..., s2:s3].contiguous().view(B, T, self.n_idx_heads, self.idx_dim)
            k_idx = packed[..., s3:s4].contiguous().view(B, T, self.idx_dim)
            w_raw = packed[..., s4:s5].contiguous().view(B, T, self.n_idx_heads)
            g_out = packed[..., s5:].contiguous()

        scale_qk = float(self.head_dim) ** 0.5
        if self.stage in ("A0", "A1"):
            q = (F.normalize(q.float(), p=2, dim=-1, eps=1e-6) * scale_qk).to(x.dtype)
            k = (F.normalize(k.float(), p=2, dim=-1, eps=1e-6) * scale_qk).to(x.dtype)
            q_idx = F.normalize(q_idx.float(), p=2, dim=-1, eps=1e-6).to(x.dtype)
            k_idx = F.normalize(k_idx.float(), p=2, dim=-1, eps=1e-6).to(x.dtype)
            w_idx = F.softplus(w_raw.float()).to(x.dtype)
        else:
            q = FusedDSAL2NormFunction.apply(q, 1e-6, scale_qk)
            k = FusedDSAL2NormFunction.apply(k, 1e-6, scale_qk)
            if shared_state is None:
                q_idx = FusedDSAL2NormFunction.apply(q_idx, 1e-6, 1.0)
                k_idx = FusedDSAL2NormFunction.apply(k_idx, 1e-6, 1.0)
                w_idx = FusedDSASoftplusGateFunction.apply(w_raw)
            else:
                w_idx = None

        if self.stage == "A0":
            from fla.ops.dsa import naive_dsa
            o = naive_dsa(q, k, v, q_idx, k_idx, w_idx, topk=self.topk)
            kl_loss = torch.zeros((), device=x.device, dtype=torch.float32)
            out_shared = None
        else:
            if shared_state is not None and self.stage == "A4":
                idx_s, bm_s, ta_s = shared_state
                o, kl_loss, _, _, _ = DSAChunkRSIFunction.apply(
                    q, k, v, None, None, None, idx_s, bm_s, ta_s, self.topk, False, 0
                )
                out_shared = shared_state
            elif return_shared_state and self.stage == "A4":
                o, kl_loss, idx_s, bm_s, ta_s = DSAChunkRSIFunction.apply(
                    q, k, v, q_idx, k_idx, w_idx, None, None, None, self.topk, True, 0
                )
                out_shared = (idx_s, bm_s, ta_s)
            else:
                o, kl_loss, _, _, _ = DSAChunkRSIFunction.apply(
                    q, k, v, q_idx, k_idx, w_idx, None, None, None, self.topk, True, 0
                )
                out_shared = None

        o_flat = o.reshape(B, T, self.q_dim)
        if self.stage in ("A0", "A1", "A2"):
            of = o_flat.float()
            rstd_o = torch.rsqrt(of.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            o_norm = (of * rstd_o).to(x.dtype) * self.post_norm_weight * F.silu(g_out)
        else:
            o_norm = FusedDSARMSNormGatedFunction.apply(o_flat, g_out, self.post_norm_weight, 1e-6)

        out = x + self.o_proj(o_norm)
        if return_shared_state:
            return out, kl_loss, out_shared
        return out, kl_loss


# ============================================================================
# 10. 8-GPU NVSwitch Context Parallelism (DSA-CP = 8)
# ============================================================================

class _DSAAllToAllSingleAutograd(torch.autograd.Function):
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


class _DSAAllGatherSequenceAutograd(torch.autograd.Function):
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
        chunks = list(grad_full.chunk(world_size, dim=1))
        chunks = [c.contiguous() for c in chunks]
        grad_local = torch.empty_like(chunks[rank])
        dist.reduce_scatter(grad_local, chunks, group=group)
        return grad_local, None


def dsa_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    q_idx_local: torch.Tensor,
    k_idx_local: torch.Tensor,
    w_idx_local: torch.Tensor,
    topk: int = 256,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Fused Single-AllToAll Ulysses Context Parallelism for DeepSeek Sparse Attention (DSA-CP = 8):
      - q_local: [B, T_local, HQ, K] (HQ divisible by P=8, e.g. HQ=32 -> HQ_local=4)
      - k_local, v_local: [B, T_local, H, K] (H divisible by P=8, e.g. H=8 -> H_local=1)
      - q_idx_local: [B, T_local, HI_local, DI] (shared per KV head group)
      - k_idx_local: [B, T_local, DI] (replicated across the P ranks via the single packed AllToAll)
      - w_idx_local: [B, T_local, HI_local]
    """
    world_size = dist.get_world_size(group)
    B, T_local, HQ, K = q_local.shape
    H = k_local.shape[2]
    V = v_local.shape[3]
    _, _, HI_loc, DI = q_idx_local.shape
    HQ_loc = HQ // world_size
    H_loc = H // world_size
    T_total = T_local * world_size

    # Reshape into [P, B, T_local, C_per_rank] and pack into 1 all_to_all_single
    q_p = q_local.view(B, T_local, world_size, HQ_loc * K).permute(2, 0, 1, 3)
    k_p = k_local.view(B, T_local, world_size, H_loc * K).permute(2, 0, 1, 3)
    v_p = v_local.view(B, T_local, world_size, H_loc * V).permute(2, 0, 1, 3)
    qi_p = q_idx_local.reshape(1, B, T_local, HI_loc * DI).expand(world_size, -1, -1, -1)
    ki_p = k_idx_local.reshape(1, B, T_local, DI).expand(world_size, -1, -1, -1)
    wi_p = w_idx_local.reshape(1, B, T_local, HI_loc).expand(world_size, -1, -1, -1)

    packed_send = torch.cat([q_p, k_p, v_p, qi_p, ki_p, wi_p], dim=-1).contiguous()
    packed_recv = _DSAAllToAllSingleAutograd.apply(packed_send, group)

    c0 = HQ_loc * K
    c1 = c0 + H_loc * K
    c2 = c1 + H_loc * V
    c3 = c2 + HI_loc * DI
    c4 = c3 + DI
    # permute(1, 0, 2, 3) -> [B, P, T_local, C] -> [B, T_total, ...]
    recv_b = packed_recv.permute(1, 0, 2, 3).contiguous()
    q_u = recv_b[..., :c0].reshape(B, T_total, HQ_loc, K)
    k_u = recv_b[..., c0:c1].reshape(B, T_total, H_loc, K)
    v_u = recv_b[..., c1:c2].reshape(B, T_total, H_loc, V)
    qi_u = recv_b[..., c2:c3].reshape(B, T_total, HI_loc, DI)
    ki_u = recv_b[..., c3:c4].reshape(B, T_total, DI)
    wi_u = recv_b[..., c4:].reshape(B, T_total, HI_loc)

    o_u, kl_loss, _, _, _ = DSAChunkRSIFunction.apply(
        q_u, k_u, v_u, qi_u, ki_u, wi_u, None, None, None, topk, True, 0
    )

    o_send = o_u.view(B, world_size, T_local, HQ_loc * V).permute(1, 0, 2, 3).contiguous()
    o_recv = _DSAAllToAllSingleAutograd.apply(o_send, group)
    o_local = o_recv.permute(1, 2, 0, 3).contiguous().view(B, T_local, HQ, V)
    return o_local, kl_loss


def dsa_cp8_ring_baseline(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    q_idx_local: torch.Tensor,
    k_idx_local: torch.Tensor,
    w_idx_local: torch.Tensor,
    topk: int = 256,
    group: Optional[dist.ProcessGroup] = None,
):
    from fla.ops.dsa import naive_dsa
    rank = dist.get_rank(group)
    T_local = q_local.shape[1]
    q_full = _DSAAllGatherSequenceAutograd.apply(q_local, group)
    k_full = _DSAAllGatherSequenceAutograd.apply(k_local, group)
    v_full = _DSAAllGatherSequenceAutograd.apply(v_local, group)
    qi_full = _DSAAllGatherSequenceAutograd.apply(q_idx_local, group)
    ki_full = _DSAAllGatherSequenceAutograd.apply(k_idx_local, group)
    wi_full = _DSAAllGatherSequenceAutograd.apply(w_idx_local, group)
    o_full = naive_dsa(q_full, k_full, v_full, qi_full, ki_full, wi_full, topk=topk)
    return o_full[:, rank * T_local : (rank + 1) * T_local].contiguous()


# ============================================================================
# 11. Cubin Register & Spill Auditor
# ============================================================================

ALL_DSA_RSI_KERNELS = [
    ("_dsa_indexer_mqa_fwd_rsi_kernel", _dsa_indexer_mqa_fwd_rsi_kernel),
    ("_dsa_build_bitmask_rsi_kernel", _dsa_build_bitmask_rsi_kernel),
    ("_dsa_gqa_bitmask_fwd_rsi_kernel", _dsa_gqa_bitmask_fwd_rsi_kernel),
    ("_dsa_bwd_prep_delta_rsi_kernel", _dsa_bwd_prep_delta_rsi_kernel),
    ("_dsa_gqa_bitmask_bwd_dq_rsi_kernel", _dsa_gqa_bitmask_bwd_dq_rsi_kernel),
    ("_dsa_gqa_bitmask_bwd_dkv_rsi_kernel", _dsa_gqa_bitmask_bwd_dkv_rsi_kernel),
    ("_dsa_kl_teacher_student_fwd_bwd_rsi_kernel", _dsa_kl_teacher_student_fwd_bwd_rsi_kernel),
    ("_dsa_indexer_kl_bwd_dq_dw_rsi_kernel", _dsa_indexer_kl_bwd_dq_dw_rsi_kernel),
    ("_dsa_indexer_kl_bwd_dk_rsi_kernel", _dsa_indexer_kl_bwd_dk_rsi_kernel),
    ("_dsa_decode_indexer_rsi_kernel", _dsa_decode_indexer_rsi_kernel),
    ("_dsa_decode_gqa_readout_rsi_kernel", _dsa_decode_gqa_readout_rsi_kernel),
    ("_dsa_l2norm_fwd_rsi_kernel", _dsa_l2norm_fwd_rsi_kernel),
    ("_dsa_l2norm_bwd_rsi_kernel", _dsa_l2norm_bwd_rsi_kernel),
    ("_dsa_w_softplus_fwd_rsi_kernel", _dsa_w_softplus_fwd_rsi_kernel),
    ("_dsa_w_softplus_bwd_rsi_kernel", _dsa_w_softplus_bwd_rsi_kernel),
    ("_dsa_rmsnorm_fwd_rsi_kernel", _dsa_rmsnorm_fwd_rsi_kernel),
    ("_dsa_rmsnorm_bwd_rsi_kernel", _dsa_rmsnorm_bwd_rsi_kernel),
    ("_dsa_rmsnorm_gated_fwd_rsi_kernel", _dsa_rmsnorm_gated_fwd_rsi_kernel),
    ("_dsa_rmsnorm_gated_bwd_rsi_kernel", _dsa_rmsnorm_gated_bwd_rsi_kernel),
]


def audit_dsa_rsi_cubin_kernels() -> dict:
    per_kernel = {}
    tot_specs = 0
    tot_spilling_specs = 0
    tot_over168_specs = 0
    tot_spills = 0
    max_regs = 0
    max_spills = 0
    max_smem = 0

    for name, fn in ALL_DSA_RSI_KERNELS:
        specs = 0
        spilling = 0
        over168 = 0
        k_spills = 0
        k_max_regs = 0
        k_max_spills = 0
        k_max_smem = 0
        for dev_cache in getattr(fn, "device_caches", {}).values():
            for compiled in dev_cache[0].values():
                if hasattr(compiled, "n_regs"):
                    specs += 1
                    r = int(compiled.n_regs)
                    s = int(compiled.n_spills)
                    sm = int(getattr(compiled.metadata, "shared", 0))
                    k_max_regs = max(k_max_regs, r)
                    k_max_spills = max(k_max_spills, s)
                    k_max_smem = max(k_max_smem, sm)
                    k_spills += s
                    if s > 0:
                        spilling += 1
                    if r > 168:
                        over168 += 1
        per_kernel[name] = {
            "specializations": specs,
            "spilling_specs": spilling,
            "over168_specs": over168,
            "max_regs": k_max_regs,
            "max_spills": k_max_spills,
            "total_spills": k_spills,
            "max_smem_bytes": k_max_smem,
        }
        tot_specs += specs
        tot_spilling_specs += spilling
        tot_over168_specs += over168
        tot_spills += k_spills
        max_regs = max(max_regs, k_max_regs)
        max_spills = max(max_spills, k_max_spills)
        max_smem = max(max_smem, k_max_smem)

    return {
        "kernel_count": len(ALL_DSA_RSI_KERNELS),
        "total_specializations": tot_specs,
        "spilling_specializations": tot_spilling_specs,
        "over168_specializations": tot_over168_specs,
        "max_regs": max_regs,
        "max_spills": max_spills,
        "total_spills": tot_spills,
        "max_smem_bytes": max_smem,
        "all_zero_spill": (tot_spilling_specs == 0 and tot_over168_specs == 0 and max_regs <= 168),
        "kernels": per_kernel,
    }
