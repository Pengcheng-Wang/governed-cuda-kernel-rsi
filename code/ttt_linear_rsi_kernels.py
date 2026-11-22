#!/usr/bin/env python3
"""
Round-27 RSI (E153, v144): Test-Time Training (TTT-Linear) — Zero-Spill SM80 Kernels,
Self-Adjoint LN^2 Backward, Fused Output LayerNorm-Residual, Single-Pass Online
Test-Time Adaptation Decoding, Superfused TTTLinearBlockRSI (Stages A0..A4), and
8-GPU NVSwitch Context Parallelism (TTT-CP = 8).

Paper:
  "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
  Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang,
  Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto,
  Carlos Guestrin (ICML 2024 / JMLR 2025, arXiv:2407.04620).
  Upstream reference: fla-org/flash-linear-attention/tree/main/fla/ops/ttt
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# 1. Zero-Spill Round-27 RSI Triton Kernels for TTT-Linear (max_regs <= 168)
# ============================================================================

@triton.jit(do_not_specialize=['T'])
def _ttt_fwd_h_rsi_kernel(
    k,
    v,
    v_new,
    x,
    r,
    eta,
    w,
    b,
    eps,
    h,
    hb,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    SAVE_FOR_BWD: tl.constexpr,
    CORRUPT_MUT1: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    NT = tl.cdiv(T, BT)
    bos = i_n * T
    boh = i_n * NT

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    # Transposed state b_ht = h^T [BV, BK]: SM80 MMA layout == DotOperand0 layout!
    if USE_INITIAL_STATE:
        p_h0 = h + (boh * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
        b_ht = tl.load(p_h0).to(tl.float32)
    else:
        b_ht = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE_B:
        b_hb = tl.load(hb + (boh * H + i_h) * V + o_v).to(tl.float32)
    else:
        b_hb = tl.zeros([BV], dtype=tl.float32)

    b_w = tl.load(w + i_h * V + o_v).to(tl.float32)
    b_b = tl.load(b + i_h * V + o_v).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_h = h + ((boh + i_t) * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
        p_hb = hb + ((boh + i_t) * H + i_h) * V + o_v
        tl.store(p_h, b_ht.to(p_h.dtype.element_ty))
        tl.store(p_hb, b_hb.to(p_hb.dtype.element_ty))

        p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
        b_kh = tl.trans(tl.dot(b_ht.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)) + b_hb[None, :]

        mean = tl.sum(b_kh, axis=1, keep_dims=True) * (1.0 / V)
        b_kh -= mean
        rstd = tl.rsqrt(tl.sum(b_kh * b_kh, axis=1, keep_dims=True) * (1.0 / V) + eps)
        b_x = b_kh * rstd

        p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None], other=0.0).to(tl.float32)
        b_y = (b_x * b_w[None, :] + b_b[None, :] - b_v + b_k.to(tl.float32)) * b_w[None, :]
        if CORRUPT_MUT1:
            b_v2 = rstd * b_y
        else:
            b_v2 = rstd * (V * b_y - tl.sum(b_y, axis=1, keep_dims=True) - b_x * tl.sum(b_y * b_x, axis=1, keep_dims=True)) * (1.0 / V)
        # Mask tail rows o_t >= T so padded rows never corrupt b_hb or v_new!
        b_v2 = tl.where(m_t[:, None], b_v2, 0.0)

        p_v_new = v_new + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), mask=m_t[:, None])
        if SAVE_FOR_BWD:
            p_x = x + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
            p_r = r + bos * H + i_h + o_t[:, None] * H
            tl.store(p_x, b_x.to(p_x.dtype.element_ty), mask=m_t[:, None])
            tl.store(p_r, rstd, mask=m_t[:, None])

        last_idx = tl.minimum(i_t * BT + BT - 1, T - 1)
        b_eta_last = tl.load(eta + bos * H + i_h + last_idx * H).to(tl.float32)
        b_hb -= tl.sum(b_eta_last * b_v2, axis=0)
        b_v2t_neg = tl.trans((-b_eta_last * b_v2).to(b_k.dtype))
        # In-place MMA accumulator update
        b_ht = tl.dot(b_v2t_neg, b_k, b_ht, allow_tf32=False)


@triton.jit(do_not_specialize=['T'])
def _ttt_fwd_o_norm_rsi_kernel(
    q,
    k,
    v_new,
    eta,
    w,
    b,
    h,
    hb,
    o,
    o_hat,
    rstd_o,
    scale,
    eps,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FUSE_OUTPUT_LN: tl.constexpr,
    SAVE_FOR_BWD: tl.constexpr,
    CORRUPT_MUT2: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_o = tl.load(hb + (i_tg * H + i_h) * V + o_v).to(tl.float32)[None, :] + tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_q = q + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None], other=0.0)
        b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
        b_A = tl.dot(b_q, tl.trans(b_k), b_A, allow_tf32=False)
        p_ht = h + (i_tg * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
        b_ht = tl.load(p_ht)
        b_q_scaled = (b_q * scale).to(b_q.dtype)
        b_o = tl.dot(b_q_scaled, tl.trans(b_ht.to(b_q.dtype)), b_o, allow_tf32=False)

    if not CORRUPT_MUT2:
        b_eta = tl.load(eta + bos * H + i_h + o_t[:, None] * H, mask=m_t[:, None], other=0.0).to(tl.float32)
        o_i = tl.arange(0, BT)
        m_A = o_i[:, None] >= o_i[None, :]
        # Factored dual-attention matrix: (b_A * scale + 1) * b_eta
        b_A_comb = tl.where(m_A, (b_A * scale + 1.0) * b_eta, 0.0).to(v_new.dtype.element_ty)

        p_v = v_new + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None], other=0.0)
        b_o = tl.dot(-b_A_comb, b_v, b_o, allow_tf32=False)

    if FUSE_OUTPUT_LN:
        mean_o = tl.sum(b_o, axis=1, keep_dims=True) * (1.0 / V)
        o_bar = b_o - mean_o
        rstd = tl.rsqrt(tl.sum(o_bar * o_bar, axis=1, keep_dims=True) * (1.0 / V) + eps)
        b_oh = o_bar * rstd
        b_w = tl.load(w + i_h * V + o_v).to(tl.float32)
        b_b = tl.load(b + i_h * V + o_v).to(tl.float32)
        b_o = b_o + b_oh * b_w[None, :] + b_b[None, :]
        if SAVE_FOR_BWD:
            p_oh = o_hat + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
            p_ro = rstd_o + bos * H + i_h + o_t[:, None] * H
            tl.store(p_oh, b_oh.to(p_oh.dtype.element_ty), mask=m_t[:, None])
            tl.store(p_ro, rstd, mask=m_t[:, None])

    p_o = o + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None])


@triton.jit(do_not_specialize=['T'])
def _ttt_bwd_dv_local_rsi_kernel(
    q,
    k,
    v,
    x,
    eta,
    w,
    b,
    do_in,
    o_hat,
    rstd_o,
    do_raw,
    dv_new,
    y_out,
    dw,
    db,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FUSE_OUTPUT_LN: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T

    b_w = tl.load(w + i_h * V + o_v).to(tl.float32)
    b_b = tl.load(b + i_h * V + o_v).to(tl.float32)

    p_x = x + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_kf = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_v[None, :]
    b_x = tl.load(p_x, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_kf = tl.load(p_kf, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_y = (b_x * b_w[None, :] + b_b[None, :] - b_v + b_kf) * b_w[None, :]
    p_y = y_out + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_t[:, None])

    p_do_in = do_in + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    b_do_in = tl.load(p_do_in, mask=m_t[:, None], other=0.0).to(tl.float32)

    if FUSE_OUTPUT_LN:
        p_oh = o_hat + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_ro = rstd_o + bos * H + i_h + o_t[:, None] * H
        b_oh = tl.load(p_oh, mask=m_t[:, None], other=0.0).to(tl.float32)
        b_ro = tl.load(p_ro, mask=m_t[:, None], other=0.0).to(tl.float32)

        b_dy_o = b_do_in * b_w[None, :]
        b_do = b_do_in + b_ro * (V * b_dy_o - tl.sum(b_dy_o, axis=1, keep_dims=True) - b_oh * tl.sum(b_dy_o * b_oh, axis=1, keep_dims=True)) * (1.0 / V)
        p_do_raw = do_raw + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        tl.store(p_do_raw, b_do.to(p_do_raw.dtype.element_ty), mask=m_t[:, None])

        b_dw_o = tl.sum(tl.where(m_t[:, None], b_do_in * b_oh, 0.0), axis=0)
        b_db_o = tl.sum(tl.where(m_t[:, None], b_do_in, 0.0), axis=0)
        tl.atomic_add(dw + i_h * V + o_v, b_dw_o)
        tl.atomic_add(db + i_h * V + o_v, b_db_o)
        b_do_cast = b_do.to(do_raw.dtype.element_ty)
    else:
        b_do_cast = b_do_in.to(do_in.dtype.element_ty)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_q = q + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None], other=0.0)
        b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
        b_A = tl.dot(b_k, tl.trans(b_q), b_A, allow_tf32=False)

    b_eta = tl.load(eta + bos * H + i_h + o_t[None, :] * H, mask=m_t[None, :], other=0.0).to(tl.float32)
    o_i = tl.arange(0, BT)
    m_A_t = o_i[:, None] <= o_i[None, :]
    b_A_comb = -tl.where(m_A_t, (b_A * scale + 1.0) * b_eta, 0.0).to(b_do_cast.dtype)

    b_dv = tl.dot(b_A_comb, b_do_cast, allow_tf32=False)
    p_dv = dv_new + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None])


@triton.jit(do_not_specialize=['T'])
def _ttt_bwd_norm_rsi_kernel(
    q,
    k,
    y,
    x,
    r,
    w,
    eta,
    do,
    dh,
    dhb,
    dv,
    dkh_out,
    dv_new,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT_B: tl.constexpr,
    STORE_INIT_GRAD_B: tl.constexpr,
    CORRUPT_MUT1: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    boh = i_n * (NT + 1)

    o_v = tl.arange(0, BV)
    if USE_FINAL_STATE_GRADIENT_B:
        b_dhb = tl.load(dhb + ((boh + NT) * H + i_h) * V + o_v).to(tl.float32)
    else:
        b_dhb = tl.zeros([BV], dtype=tl.float32)

    if not USE_FINAL_STATE_GRADIENT:
        for i_k in range(K // BK):
            o_k = i_k * BK + tl.arange(0, BK)
            p_dh_last = dh + ((boh + NT) * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
            tl.store(p_dh_last, tl.zeros([BV, BK], dtype=p_dh_last.dtype.element_ty))

    for i_t in range(NT - 1, -1, -1):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_dhb = dhb + ((boh + i_t + 1) * H + i_h) * V + o_v
        tl.store(p_dhb, b_dhb.to(p_dhb.dtype.element_ty))

        last_idx = tl.minimum(i_t * BT + BT - 1, T - 1)
        b_eta_last = tl.load(eta + bos * H + i_h + last_idx * H).to(tl.float32)
        p_dv_new = dv_new + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_dv_new = tl.load(p_dv_new, mask=m_t[:, None], other=0.0).to(tl.float32) - b_eta_last * b_dhb[None, :]

        for i_k in range(K // BK):
            o_k = i_k * BK + tl.arange(0, BK)
            p_dh_cur = dh + ((boh + i_t + 1) * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
            b_dht_k = tl.load(p_dh_cur)
            p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
            b_k_scaled_t = tl.trans((b_eta_last * b_k.to(tl.float32)).to(b_k.dtype))
            b_dv_new -= tl.trans(tl.dot(b_dht_k.to(b_k.dtype), b_k_scaled_t, allow_tf32=False))

        p_x = x + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_y = y + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_r = r + bos * H + i_h + o_t[:, None] * H
        b_x = tl.load(p_x, mask=m_t[:, None], other=0.0).to(tl.float32)
        b_y = tl.load(p_y, mask=m_t[:, None], other=0.0).to(tl.float32)
        b_rstd = tl.load(p_r, mask=m_t[:, None], other=0.0).to(tl.float32)
        b_w = tl.load(w + i_h * V + o_v).to(tl.float32)

        s_dv_x = tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)
        b_dy = b_rstd * (b_dv_new * V - tl.sum(b_dv_new, axis=1, keep_dims=True) - b_x * s_dv_x) * (1.0 / V)
        # Self-Adjoint LayerNorm Identity: <dv_new, v_new> == <dy, y>!
        b_drstd = tl.sum(b_dy * b_y, axis=1, keep_dims=True) / b_rstd
        if CORRUPT_MUT1:
            b_dx = b_w[None, :] * b_w[None, :] * b_dy
        else:
            b_dx = -b_rstd * (b_dv_new * tl.sum(b_x * b_y, axis=1, keep_dims=True) + b_y * s_dv_x) * (1.0 / V) + b_w[None, :] * b_w[None, :] * b_dy

        p_dv = dv + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        tl.store(p_dv, (-b_w[None, :] * b_dy).to(p_dv.dtype.element_ty), mask=m_t[:, None])

        b_dkh = b_rstd * (V * b_dx - tl.sum(b_dx, axis=1, keep_dims=True) - b_x * tl.sum(b_x * b_dx, axis=1, keep_dims=True) - b_rstd * b_drstd * b_x) * (1.0 / V)
        b_dkh = tl.where(m_t[:, None], b_dkh, 0.0)

        p_dkh = dkh_out + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        tl.store(p_dkh, b_dkh.to(p_dkh.dtype.element_ty), mask=m_t[:, None])

        p_do = do + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None], other=0.0)
        b_dhb += tl.sum(tl.where(m_t[:, None], b_do.to(tl.float32) + b_dkh, 0.0), axis=0)
        b_dkh_t = tl.trans(b_dkh.to(b_do.dtype))
        b_do_t = tl.trans(b_do)

        for i_k in range(K // BK):
            o_k = i_k * BK + tl.arange(0, BK)
            p_dh_cur = dh + ((boh + i_t + 1) * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
            p_dh_prev = dh + ((boh + i_t) * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
            b_dht_k = tl.load(p_dh_cur).to(tl.float32)
            p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
            p_q = q + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
            b_q = (tl.load(p_q, mask=m_t[:, None], other=0.0) * scale).to(b_do.dtype)
            b_dht_k = tl.dot(b_dkh_t, b_k, b_dht_k, allow_tf32=False)
            b_dht_k = tl.dot(b_do_t, b_q, b_dht_k, allow_tf32=False)
            tl.store(p_dh_prev, b_dht_k.to(p_dh_prev.dtype.element_ty))

    if STORE_INIT_GRAD_B:
        tl.store(dhb + (boh * H + i_h) * V + o_v, b_dhb.to(dhb.dtype.element_ty))


@triton.jit(do_not_specialize=['T'])
def _ttt_bwd_dwdb_rsi_kernel(
    k,
    v,
    x,
    dv,
    w,
    b,
    dw,
    db,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T

    p_dv = dv + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_x = x + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_v[None, :]
    b_dv = tl.load(p_dv, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_x = tl.load(p_x, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_k = tl.load(p_k, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_w = tl.load(w + i_h * V + o_v).to(tl.float32)
    b_b = tl.load(b + i_h * V + o_v).to(tl.float32)

    b_dy = -b_dv / b_w[None, :]
    b_dw = tl.sum(tl.where(m_t[:, None], (2.0 * b_w[None, :] * b_x + b_b[None, :] - b_v + b_k) * b_dy, 0.0), axis=0)
    b_db = tl.sum(tl.where(m_t[:, None], -b_dv, 0.0), axis=0)
    tl.atomic_add(dw + i_h * V + o_v, b_dw)
    tl.atomic_add(db + i_h * V + o_v, b_db)


@triton.jit(do_not_specialize=['T'])
def _ttt_bwd_dqke_rsi_kernel(
    q,
    k,
    v_new,
    e,
    h,
    do,
    dh,
    dhb,
    dv,
    dkh_in,
    dq,
    dk,
    de,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kt = tl.program_id(0).to(tl.int64)
    NK = K // BK
    i_k = i_kt % NK
    i_t = i_kt // NK
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    i_tg_h = i_b * NT + i_t
    i_tg_dh = i_b * (NT + 1) + i_t + 1
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)

    # Loop 1: h pass (coalesced [BV, BK] loads along K)
    for i_v in range(V // BV):
        o_v = i_v * BV + tl.arange(0, BV)
        p_do = do + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_dkh = dkh_in + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_ht = h + (i_tg_h * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
        b_do_v = tl.load(p_do, mask=m_t[:, None], other=0.0)
        b_dkh_v = tl.load(p_dkh, mask=m_t[:, None], other=0.0)
        b_ht = tl.load(p_ht).to(b_do_v.dtype)
        b_dq = tl.dot(b_do_v, b_ht, b_dq, allow_tf32=False)
        b_dk = tl.dot(b_dkh_v, b_ht, b_dk, allow_tf32=False)

    p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
    last_idx = tl.minimum(i_t * BT + BT - 1, T - 1)
    b_e_last = tl.load(e + bos * H + i_h + last_idx * H).to(tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_de_last_scalar = tl.zeros([1], dtype=tl.float32)

    # Loop 2: dh pass (coalesced [BV, BK] loads along K)
    for i_v in range(V // BV):
        o_v = i_v * BV + tl.arange(0, BV)
        p_v_new = v_new + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_do = do + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_dht = dh + (i_tg_dh * H + i_h) * (K * V) + o_v[:, None] * K + o_k[None, :]
        b_vn = tl.load(p_v_new, mask=m_t[:, None], other=0.0)
        b_do_v = tl.load(p_do, mask=m_t[:, None], other=0.0)
        b_dht = tl.load(p_dht)

        b_ds = tl.dot(b_do_v, tl.trans(b_vn), b_ds, allow_tf32=False)
        b_vn_neg_e = (-b_e_last * b_vn.to(tl.float32)).to(b_vn.dtype)
        b_dk = tl.dot(b_vn_neg_e, b_dht.to(b_vn.dtype), b_dk, allow_tf32=False)
        b_vk = tl.dot(tl.trans(b_vn), b_k, allow_tf32=False)
        b_de_last_scalar -= tl.sum(b_dht.to(tl.float32) * b_vk)
        if i_k == 0:
            p_dhb = dhb + (i_tg_dh * H + i_h) * V + o_v
            b_dhb_v = tl.load(p_dhb).to(tl.float32)
            b_de_last_scalar -= tl.sum(b_dhb_v * tl.sum(b_vn.to(tl.float32), axis=0))

    p_dv_full = dv + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_k[None, :]
    b_dk -= tl.load(p_dv_full, mask=m_t[:, None], other=0.0).to(tl.float32)

    o_i = tl.arange(0, BT)
    p_q = q + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    p_e = e + bos * H + i_h + o_t * H
    b_q = tl.load(p_q, mask=m_t[:, None], other=0.0)
    b_e = tl.load(p_e, mask=m_t, other=0.0).to(tl.float32)

    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds, 0.0)
    b_ds_cast = b_ds.to(b_k.dtype)
    b_ds_k = tl.dot(b_ds_cast, b_k, allow_tf32=False)
    b_dq -= b_ds_k * b_e[:, None]
    b_dq *= scale

    b_q_e = (b_q.to(tl.float32) * (b_e[:, None] * scale)).to(b_k.dtype)
    b_dk -= tl.dot(tl.trans(b_ds_cast), b_q_e, allow_tf32=False)

    i_last = (BT - 1) if (i_t * BT + BT) <= T else (T % BT - 1)
    mask_last = (o_i == i_last)
    b_de = tl.where(mask_last, b_de_last_scalar, 0.0)
    b_de -= tl.sum(scale * b_ds_k * b_q.to(tl.float32), axis=1)
    if i_k == 0:
        b_de -= tl.sum(b_ds, axis=1)

    p_dq = dq + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    p_dk = dk + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    p_de = de + bos * H + i_h + o_t * H
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_t[:, None])
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_t[:, None])
    if NK == 1:
        tl.store(p_de, b_de.to(p_de.dtype.element_ty), mask=m_t)
    else:
        tl.atomic_add(p_de, b_de.to(p_de.dtype.element_ty), mask=m_t)


@triton.jit(do_not_specialize=['T'])
def _ttt_decode_adapt_rsi_kernel(
    q,
    k,
    v,
    eta,
    w,
    b,
    h0,
    hb0,
    o,
    ht,
    hbt,
    scale,
    eps,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_b, i_h = i_nh // H, i_nh % H
    bos = i_b * T

    o_t = tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T

    b_hb = tl.load(hb0 + i_nh * V + o_v).to(tl.float32)
    b_kh = tl.zeros([BT, BV], dtype=tl.float32) + b_hb[None, :]

    # Loop 1: b_kh = k @ h0 + hb0 (coalesced [BK, BV] along V)
    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_h0 = h0 + i_nh * (K * V) + o_k[:, None] * V + o_v[None, :]
        b_h_k = tl.load(p_h0).to(q.dtype.element_ty)
        p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
        b_kh = tl.dot(b_k, b_h_k, b_kh, allow_tf32=False)

    mean = tl.sum(b_kh, axis=1, keep_dims=True) * (1.0 / V)
    b_kh -= mean
    rstd = tl.rsqrt(tl.sum(b_kh * b_kh, axis=1, keep_dims=True) * (1.0 / V) + eps)
    b_x = b_kh * rstd

    b_w = tl.load(w + i_h * V + o_v).to(tl.float32)
    b_b = tl.load(b + i_h * V + o_v).to(tl.float32)
    p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_kf = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_v[None, :]
    b_v = tl.load(p_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_kf = tl.load(p_kf, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_y = (b_x * b_w[None, :] + b_b[None, :] - b_v + b_kf) * b_w[None, :]
    b_v2 = rstd * (V * b_y - tl.sum(b_y, axis=1, keep_dims=True) - b_x * tl.sum(b_y * b_x, axis=1, keep_dims=True)) * (1.0 / V)
    # Mask tail rows o_t >= T to fix upstream T % BT != 0 bias corruption bug!
    b_v2 = tl.where(m_t[:, None], b_v2, 0.0)

    # Loop 2: Update fast weights (ht, hbt), then b_v2 dies after -b_A_comb @ b_v2!
    b_eta_last = tl.load(eta + bos * H + i_h + (T - 1) * H).to(tl.float32)
    b_hbt_new = b_hb - tl.sum(b_eta_last * b_v2, axis=0)
    tl.store(hbt + i_nh * V + o_v, b_hbt_new.to(hbt.dtype.element_ty))
    b_v2_neg = (-b_eta_last * b_v2).to(q.dtype.element_ty)

    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_h0 = h0 + i_nh * (K * V) + o_k[:, None] * V + o_v[None, :]
        b_h_k = tl.load(p_h0).to(tl.float32)
        p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
        b_h_k = tl.dot(tl.trans(b_k), b_v2_neg, b_h_k, allow_tf32=False)
        p_ht = ht + i_nh * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h_k.to(p_ht.dtype.element_ty))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_q = q + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None], other=0.0)
        b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
        b_A = tl.dot(b_q, tl.trans(b_k), b_A, allow_tf32=False)

    b_eta = tl.load(eta + bos * H + i_h + o_t[:, None] * H, mask=m_t[:, None], other=0.0).to(tl.float32)
    m_A = o_t[:, None] >= o_t[None, :]
    b_A_comb = tl.where(m_A, (b_A * scale + 1.0) * b_eta, 0.0).to(q.dtype.element_ty)
    b_o = tl.zeros([BT, BV], dtype=tl.float32) + b_hb[None, :]
    b_o = tl.dot(-b_A_comb, b_v2.to(q.dtype.element_ty), b_o, allow_tf32=False)

    # Loop 3: b_v2 and b_A are dead; accumulate (q * scale) @ h0 into b_o!
    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_h0 = h0 + i_nh * (K * V) + o_k[:, None] * V + o_v[None, :]
        b_h_k = tl.load(p_h0).to(q.dtype.element_ty)
        p_q = q + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None], other=0.0)
        b_o = tl.dot((b_q * scale).to(b_q.dtype), b_h_k, b_o, allow_tf32=False)

    mean_o = tl.sum(b_o, axis=1, keep_dims=True) * (1.0 / V)
    o_bar = b_o - mean_o
    rstd_o = tl.rsqrt(tl.sum(o_bar * o_bar, axis=1, keep_dims=True) * (1.0 / V) + eps)
    b_o_final = b_o + o_bar * rstd_o * b_w[None, :] + b_b[None, :]
    p_o = o + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    tl.store(p_o, b_o_final.to(p_o.dtype.element_ty), mask=m_t[:, None])


@triton.jit
def _ttt_rmsnorm_fwd_rsi_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    N_ROWS,
    D: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= N_ROWS:
        return
    cols = tl.arange(0, BD)
    m = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + EPS)
    tl.store(rstd_ptr + row, rstd)
    y = x * rstd * w
    tl.store(y_ptr + row * D + cols, y.to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _ttt_rmsnorm_bwd_rsi_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    dw_partial_ptr,
    N_ROWS,
    ROWS_PER_PROG: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * ROWS_PER_PROG
    cols = tl.arange(0, BD)
    m = cols < D
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BD], dtype=tl.float32)
    for idx in range(ROWS_PER_PROG):
        row = row_start + idx
        if row < N_ROWS:
            dy = tl.load(dy_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + row).to(tl.float32)
            x_hat = x * rstd
            dw_acc += dy * x_hat
            dyw = dy * w
            c1 = tl.sum(x_hat * dyw, axis=0) * (1.0 / D)
            dx = rstd * (dyw - x_hat * c1)
            tl.store(dx_ptr + row * D + cols, dx.to(dx_ptr.dtype.element_ty), mask=m)
    tl.store(dw_partial_ptr + pid * D + cols, dw_acc, mask=m)


# ============================================================================
# 2. Autograd Wrappers, FP64 Oracle, Mutation Gates, and Cubin Resource Audit
# ============================================================================

class _TTTRMSNormRSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.contiguous().view(-1, D)
        N_ROWS = x_2d.shape[0]
        BD = triton.next_power_of_2(D)
        y_2d = torch.empty_like(x_2d)
        rstd = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
        nw = 4 if BD <= 1024 else 8
        _ttt_rmsnorm_fwd_rsi_kernel[(N_ROWS,)](
            x_2d, weight, y_2d, rstd, N_ROWS, D=D, BD=BD, EPS=eps, num_warps=nw, num_stages=1
        )
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.D = D
        ctx.BD = BD
        return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, weight, rstd = ctx.saved_tensors
        D, BD = ctx.D, ctx.BD
        dy_2d = dy.contiguous().view(-1, D)
        N_ROWS = dy_2d.shape[0]
        dx_2d = torch.empty_like(x_2d)
        ROWS_PER_PROG = 64
        n_progs = triton.cdiv(N_ROWS, ROWS_PER_PROG)
        dw_partial = torch.empty((n_progs, D), device=x_2d.device, dtype=torch.float32)
        nw = 4 if BD <= 1024 else 8
        _ttt_rmsnorm_bwd_rsi_kernel[(n_progs,)](
            dy_2d, x_2d, weight, rstd, dx_2d, dw_partial,
            N_ROWS, ROWS_PER_PROG=ROWS_PER_PROG, D=D, BD=BD, num_warps=nw, num_stages=1
        )
        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx_2d.view(dy.shape), dw, None


def ttt_rmsnorm_rsi(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _TTTRMSNormRSIFunction.apply(x, weight, eps)


class ChunkTTTLinearRSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        eta: torch.Tensor,
        scale: float,
        eps: float,
        BT: int,
        initial_state: Optional[torch.Tensor],
        initial_state_bias: Optional[torch.Tensor],
        output_final_state: bool,
        fuse_output_ln: bool,
        mutation_mode: int,
    ):
        B, T, H, K = q.shape
        V = v.shape[-1]
        NT = triton.cdiv(T, BT)
        need_bwd = (
            q.requires_grad
            or k.requires_grad
            or v.requires_grad
            or w.requires_grad
            or b.requires_grad
            or eta.requires_grad
            or (initial_state is not None and initial_state.requires_grad)
            or (initial_state_bias is not None and initial_state_bias.requires_grad)
        )

        h = k.new_empty(B, NT, H, V, K)
        hb = k.new_empty(B, NT, H, 1, V)
        if initial_state is not None:
            h[:, 0].copy_(initial_state.transpose(-1, -2))
        if initial_state_bias is not None:
            hb[:, 0].copy_(initial_state_bias)

        v_new = torch.empty_like(v)
        x = torch.empty_like(v) if need_bwd else None
        r = v.new_empty(B, T, H, 1, dtype=torch.float32) if need_bwd else None

        nw_h = 4 if K <= 64 else 16
        corrupt_mut1 = mutation_mode == 1
        corrupt_mut2 = mutation_mode == 2
        effective_fuse_ln = False if mutation_mode == 3 else fuse_output_ln

        _ttt_fwd_h_rsi_kernel[(B * H,)](
            k,
            v,
            v_new,
            x if need_bwd else v_new,
            r if need_bwd else v_new,
            eta,
            w,
            b,
            eps,
            h,
            hb,
            T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=K,
            BV=V,
            USE_INITIAL_STATE=initial_state is not None,
            USE_INITIAL_STATE_B=initial_state_bias is not None,
            SAVE_FOR_BWD=need_bwd,
            CORRUPT_MUT1=corrupt_mut1,
            num_warps=nw_h,
            num_stages=1,
        )

        if output_final_state:
            t0 = (NT - 1) * BT
            h_last = h[:, NT - 1].transpose(-1, -2).to(torch.float32)
            hb_last = hb[:, NT - 1].to(torch.float32)
            k_last = k[:, t0:T].transpose(1, 2).to(torch.float32)
            vn_last = v_new[:, t0:T].transpose(1, 2).to(torch.float32)
            e_last = eta[:, T - 1].unsqueeze(-1).to(torch.float32)
            ht = h_last - torch.matmul((e_last * k_last).transpose(-1, -2), vn_last)
            hbt = hb_last - (e_last * vn_last).sum(dim=-2, keepdim=True)
        else:
            ht, hbt = None, None

        o = torch.empty_like(v)
        o_hat = torch.empty_like(v) if (effective_fuse_ln and need_bwd) else None
        rstd_o = v.new_empty(B, T, H, 1, dtype=torch.float32) if (effective_fuse_ln and need_bwd) else None
        BK_o = 16 if q.dtype == torch.float32 else 64
        nw_o = 4 if K <= 64 else 8
        _ttt_fwd_o_norm_rsi_kernel[(NT, B * H)](
            q,
            k,
            v_new,
            eta,
            w,
            b,
            h,
            hb,
            o,
            o_hat if o_hat is not None else o,
            rstd_o if rstd_o is not None else o,
            scale,
            eps,
            T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK_o,
            BV=V,
            FUSE_OUTPUT_LN=effective_fuse_ln,
            SAVE_FOR_BWD=need_bwd,
            CORRUPT_MUT2=corrupt_mut2,
            num_warps=nw_o,
            num_stages=1,
        )
        if need_bwd:
            ctx.save_for_backward(
                q, k, v, w, b, eta, h, v_new, x, r, o_hat, rstd_o, initial_state, initial_state_bias
            )
            ctx.scale = scale
            ctx.eps = eps
            ctx.BT = BT
            ctx.fuse_output_ln = effective_fuse_ln
            ctx.mutation_mode = mutation_mode
        return o, ht, hbt

    @staticmethod
    def backward(ctx, do_in, dht, dhbt):
        (
            q,
            k,
            v,
            w,
            b,
            eta,
            h,
            v_new,
            x,
            r,
            o_hat,
            rstd_o,
            initial_state,
            initial_state_bias,
        ) = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        BT = ctx.BT
        NT = triton.cdiv(T, BT)
        scale = ctx.scale
        fuse_output_ln = ctx.fuse_output_ln
        corrupt_mut1 = ctx.mutation_mode == 1

        do_raw = torch.empty_like(do_in) if fuse_output_ln else do_in
        dv_new = torch.empty_like(v)
        y = torch.empty_like(v)
        dw = torch.zeros(H, V, device=w.device, dtype=torch.float32)
        db = torch.zeros(H, V, device=b.device, dtype=torch.float32)

        BK_dv = 32 if q.dtype == torch.float32 else 64
        nw_dv = 4 if K <= 64 else 8
        _ttt_bwd_dv_local_rsi_kernel[(NT, B * H)](
            q,
            k,
            v,
            x,
            eta,
            w,
            b,
            do_in,
            o_hat if o_hat is not None else do_in,
            rstd_o if rstd_o is not None else do_in,
            do_raw,
            dv_new,
            y,
            dw,
            db,
            scale,
            T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK_dv,
            BV=V,
            FUSE_OUTPUT_LN=fuse_output_ln,
            num_warps=nw_dv,
            num_stages=1,
        )

        dh = q.new_empty(B, NT + 1, H, V, K, dtype=torch.float32)
        dhb = q.new_empty(B, NT + 1, H, 1, V, dtype=torch.float32)
        if dht is not None:
            dh[:, NT].copy_(dht.transpose(-1, -2))
        if dhbt is not None:
            dhb[:, NT].copy_(dhbt)

        dv = torch.empty_like(v)
        dkh_out = torch.empty_like(v)

        BK_norm = 16 if (K == 128 or q.dtype == torch.float32) else 32
        _ttt_bwd_norm_rsi_kernel[(B * H,)](
            q,
            k,
            y,
            x,
            r,
            w,
            eta,
            do_raw,
            dh,
            dhb,
            dv,
            dkh_out,
            dv_new,
            scale,
            T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK_norm,
            BV=V,
            USE_FINAL_STATE_GRADIENT=dht is not None,
            USE_FINAL_STATE_GRADIENT_B=dhbt is not None,
            STORE_INIT_GRAD_B=initial_state_bias is not None,
            CORRUPT_MUT1=corrupt_mut1,
            num_warps=8,
            num_stages=1,
        )

        _ttt_bwd_dwdb_rsi_kernel[(NT, B * H)](
            k,
            v,
            x,
            dv,
            w,
            b,
            dw,
            db,
            T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BV=V,
            num_warps=4,
            num_stages=1,
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        BK_dqke = 32 if q.dtype == torch.float32 else 64
        BV_dqke = 32
        NK_dqke = K // BK_dqke
        de = torch.zeros(B, T, H, 1, device=eta.device, dtype=torch.float32) if NK_dqke > 1 else torch.empty_like(eta)
        nw_dqke = 8 if q.dtype == torch.float32 else 4
        _ttt_bwd_dqke_rsi_kernel[(NK_dqke * NT, B * H)](
            q,
            k,
            v_new,
            eta,
            h,
            do_raw,
            dh,
            dhb,
            dv,
            dkh_out,
            dq,
            dk,
            de,
            scale,
            T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK_dqke,
            BV=BV_dqke,
            num_warps=nw_dqke,
            num_stages=1,
        )
        dh0 = dh[:, 0].transpose(-1, -2).contiguous() if initial_state is not None else None
        dhb0 = dhb[:, 0].contiguous() if initial_state_bias is not None else None
        return (
            dq,
            dk,
            dv,
            dw.to(w.dtype),
            db.to(b.dtype),
            de.to(eta.dtype),
            None,
            None,
            None,
            dh0,
            dhb0,
            None,
            None,
            None,
        )


def chunk_ttt_linear_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: Optional[float] = None,
    eps: float = 1e-6,
    chunk_size: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_bias: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    fuse_output_ln: bool = True,
    mutation_mode: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)
    o, ht, hbt = ChunkTTTLinearRSIFunction.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        w.contiguous(),
        b.contiguous(),
        eta.contiguous(),
        scale,
        eps,
        chunk_size,
        initial_state.contiguous() if initial_state is not None else None,
        initial_state_bias.contiguous() if initial_state_bias is not None else None,
        output_final_state,
        fuse_output_ln,
        mutation_mode,
    )
    if not fuse_output_ln and mutation_mode != 3:
        B, T, H, D = o.shape
        o_f = o.float()
        mean_o = o_f.mean(dim=-1, keepdim=True)
        var_o = o_f.var(dim=-1, unbiased=False, keepdim=True)
        o_ln = (o_f - mean_o) * torch.rsqrt(var_o + eps) * w.float().view(1, 1, H, D) + b.float().view(1, 1, H, D)
        o = (o_f + o_ln).to(q.dtype)
    return o, ht, hbt


def ttt_decode_adapt_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_bias: torch.Tensor,
    scale: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert T <= 16, f"ttt_decode_adapt_rsi supports T <= 16, got T={T}"
    if scale is None:
        scale = K ** -0.5
    o = torch.empty_like(v)
    ht = torch.empty_like(initial_state, dtype=torch.float32)
    hbt = torch.empty_like(initial_state_bias, dtype=torch.float32)
    _ttt_decode_adapt_rsi_kernel[(B * H,)](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        eta.contiguous(),
        w.contiguous(),
        b.contiguous(),
        initial_state.contiguous(),
        initial_state_bias.contiguous(),
        o,
        ht,
        hbt,
        scale,
        eps,
        T,
        H=H,
        K=K,
        V=V,
        BT=16,
        BK=32,
        BV=V,
        num_warps=8,
        num_stages=1,
    )
    return o, ht, hbt


def ttt_linear_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: Optional[float] = None,
    eps: float = 1e-6,
    BT: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_bias: Optional[torch.Tensor] = None,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    qd = q.detach().double().requires_grad_(True)
    kd = k.detach().double().requires_grad_(True)
    vd = v.detach().double().requires_grad_(True)
    wd = w.detach().double().requires_grad_(True)
    bd = b.detach().double().requires_grad_(True)
    ed = eta.detach().double().requires_grad_(True)
    h0d = initial_state.detach().double().requires_grad_(True) if initial_state is not None else None
    hb0d = initial_state_bias.detach().double().requires_grad_(True) if initial_state_bias is not None else None

    NT = triton.cdiv(T, BT)
    pad_len = (BT - (T % BT)) % BT
    qp, kp, vp, ep = qd.transpose(1, 2), kd.transpose(1, 2), vd.transpose(1, 2), ed.transpose(1, 2)
    if pad_len > 0:
        qp = F.pad(qp, (0, 0, 0, pad_len))
        kp = F.pad(kp, (0, 0, 0, pad_len))
        vp = F.pad(vp, (0, 0, 0, pad_len))
        ep = F.pad(ep, (0, 0, 0, pad_len))
        ep[:, :, -1, :] = ep[:, :, -(pad_len + 1), :]

    q_c = qp.reshape(B, H, NT, BT, K)
    k_c = kp.reshape(B, H, NT, BT, K)
    v_c = vp.reshape(B, H, NT, BT, V)
    e_c = ep.reshape(B, H, NT, BT, 1)
    w_c = wd.reshape(1, H, 1, V)
    b_c = bd.reshape(1, H, 1, V)

    h = h0d.clone() if h0d is not None else torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    hb = hb0d.clone() if hb0d is not None else torch.zeros(B, H, 1, V, dtype=torch.float64, device=q.device)
    outs = []
    mask = torch.tril(torch.ones(BT, BT, dtype=torch.float64, device=q.device))
    for i in range(NT):
        valid_len = min(BT, T - i * BT)
        row_mask = (torch.arange(BT, device=q.device) < valid_len).to(torch.float64).view(1, 1, BT, 1)
        qi, ki, vi, ei = q_c[:, :, i], k_c[:, :, i], v_c[:, :, i], e_c[:, :, i]
        kh = ki @ h + hb
        mean = kh.mean(-1, keepdim=True)
        var = kh.var(-1, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x = (kh - mean) * rstd
        y = (x * w_c + b_c - vi + ki) * w_c
        v_new = rstd * (V * y - y.sum(-1, keepdim=True) - x * (y * x).sum(-1, keepdim=True)) / V
        v_new = v_new * row_mask
        attn = (qi @ ki.transpose(-1, -2)) * mask
        oi = (qi * scale) @ h - ((ei * attn) * scale + ei * mask) @ v_new + hb
        mo = oi.mean(-1, keepdim=True)
        vo = oi.var(-1, unbiased=False, keepdim=True)
        oi_norm = (oi - mo) * torch.rsqrt(vo + eps) * w_c + b_c
        outs.append(oi + oi_norm)
        e_last = ei[:, :, valid_len - 1 : valid_len, :]
        h = h - (e_last * ki).transpose(-1, -2) @ v_new
        hb = hb - (e_last * v_new).sum(-2, keepdim=True)
    o = torch.stack(outs, dim=2).reshape(B, H, NT * BT, V)[:, :, :T, :].transpose(1, 2).contiguous()
    return o, h, hb, (qd, kd, vd, wd, bd, ed, h0d, hb0d)


def audit_ttt_rsi_cubin_resources() -> Dict[str, Dict[str, int]]:
    kernels = {
        "_ttt_fwd_h_rsi_kernel": _ttt_fwd_h_rsi_kernel,
        "_ttt_fwd_o_norm_rsi_kernel": _ttt_fwd_o_norm_rsi_kernel,
        "_ttt_bwd_dv_local_rsi_kernel": _ttt_bwd_dv_local_rsi_kernel,
        "_ttt_bwd_norm_rsi_kernel": _ttt_bwd_norm_rsi_kernel,
        "_ttt_bwd_dwdb_rsi_kernel": _ttt_bwd_dwdb_rsi_kernel,
        "_ttt_bwd_dqke_rsi_kernel": _ttt_bwd_dqke_rsi_kernel,
        "_ttt_decode_adapt_rsi_kernel": _ttt_decode_adapt_rsi_kernel,
        "_ttt_rmsnorm_fwd_rsi_kernel": _ttt_rmsnorm_fwd_rsi_kernel,
        "_ttt_rmsnorm_bwd_rsi_kernel": _ttt_rmsnorm_bwd_rsi_kernel,
    }
    report: Dict[str, Dict[str, int]] = {}
    for name, kern in kernels.items():
        cur = kern
        while hasattr(cur, "fn") and not isinstance(cur, triton.runtime.JITFunction):
            cur = cur.fn
        regs: List[int] = []
        spills: List[int] = []
        smems: List[int] = []
        for _dev, tup in getattr(cur, "device_caches", {}).items():
            for _sig, c in tup[0].items():
                if hasattr(c, "n_regs"):
                    regs.append(int(c.n_regs))
                    spills.append(int(c.n_spills))
                    smems.append(int(getattr(c.metadata, "shared", 0)))
        report[name] = {
            "total_specs": len(regs),
            "spilling_specs": sum(1 for s in spills if s > 0),
            "over168_specs": sum(1 for r in regs if r > 168),
            "max_regs": max(regs) if regs else 0,
            "max_spills": max(spills) if spills else 0,
            "total_spills": sum(spills),
            "max_smem": max(smems) if smems else 0,
        }
    return report


# ============================================================================
# 3. Track 3: End-to-End TTTLinearBlockBaseline (A0) & TTTLinearBlockRSI (A1..A4)
# ============================================================================

class TTTLinearBlockBaseline(nn.Module):
    """Stage A0: Upstream TTT-Linear block with eager RMSNorm, 4 separate projections, and upstream chunk_ttt_linear."""

    def __init__(self, d_model: int, num_heads: int, base_lr: float = 0.05, eps: float = 1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.base_lr = base_lr
        self.eps = eps

        self.in_norm_weight = nn.Parameter(torch.ones(d_model))
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.eta_proj = nn.Linear(d_model, num_heads, bias=True)
        self.ttt_norm_weight = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor, upstream_op) -> torch.Tensor:
        B, T, D = hidden_states.shape
        H, K = self.num_heads, self.head_dim
        x_f = hidden_states.float()
        rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = (x_f * rstd).to(hidden_states.dtype) * self.in_norm_weight

        q = self.q_proj(x_norm).view(B, T, H, K)
        k = self.k_proj(x_norm).view(B, T, H, K)
        v = self.v_proj(x_norm).view(B, T, H, K)
        eta = (torch.sigmoid(self.eta_proj(x_norm).float()) * (self.base_lr / K)).to(hidden_states.dtype).unsqueeze(-1)

        o, _, _ = upstream_op(q, k, v, self.ttt_norm_weight, self.ttt_norm_bias, eta, eps=self.eps)
        return hidden_states + self.o_proj(o.reshape(B, T, D))


class TTTLinearBlockRSI(nn.Module):
    """
    Stages A1..A4 progressive optimization ladder for TTT-Linear:
      stage=1 (A1): Zero-spill chunk_ttt_linear_rsi (without fused output LN, eager periphery)
      stage=2 (A2): + Fused in-kernel output LayerNorm-residual (eliminating 3 .clone() + group_norm)
      stage=3 (A3): + Fused single-GEMM QKVEta projection
      stage=4 (A4): + Persistent 2D Triton input RMSNorm (_ttt_rmsnorm_fwd/bwd_rsi_kernel)
    """

    def __init__(self, d_model: int, num_heads: int, base_lr: float = 0.05, eps: float = 1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.base_lr = base_lr
        self.eps = eps

        self.in_norm_weight = nn.Parameter(torch.ones(d_model))
        self.qkve_proj = nn.Linear(d_model, 3 * d_model + num_heads, bias=False)
        self.eta_bias = nn.Parameter(torch.zeros(num_heads))
        self.ttt_norm_weight = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    @classmethod
    def from_baseline(cls, base: TTTLinearBlockBaseline) -> "TTTLinearBlockRSI":
        mod = cls(base.d_model, base.num_heads, base.base_lr, base.eps).to(
            device=base.in_norm_weight.device, dtype=base.in_norm_weight.dtype
        )
        with torch.no_grad():
            mod.in_norm_weight.copy_(base.in_norm_weight)
            mod.qkve_proj.weight.copy_(
                torch.cat([base.q_proj.weight, base.k_proj.weight, base.v_proj.weight, base.eta_proj.weight], dim=0)
            )
            mod.eta_bias.copy_(base.eta_proj.bias)
            mod.ttt_norm_weight.copy_(base.ttt_norm_weight)
            mod.ttt_norm_bias.copy_(base.ttt_norm_bias)
            mod.o_proj.weight.copy_(base.o_proj.weight)
        return mod

    def forward(
        self,
        hidden_states: torch.Tensor,
        stage: int = 4,
        sep_projs: Optional[Tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]] = None,
    ) -> torch.Tensor:
        B, T, D = hidden_states.shape
        H, K = self.num_heads, self.head_dim

        if stage >= 4:
            x_norm = ttt_rmsnorm_rsi(hidden_states, self.in_norm_weight, self.eps)
        else:
            x_f = hidden_states.float()
            rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            x_norm = (x_f * rstd).to(hidden_states.dtype) * self.in_norm_weight

        if stage >= 3 or sep_projs is None:
            qkve = self.qkve_proj(x_norm)
            q_flat, k_flat, v_flat, e_raw = torch.split(qkve, [D, D, D, H], dim=-1)
            q = q_flat.view(B, T, H, K)
            k = k_flat.view(B, T, H, K)
            v = v_flat.view(B, T, H, K)
            eta = (torch.sigmoid((e_raw + self.eta_bias).float()) * (self.base_lr / K)).to(hidden_states.dtype).unsqueeze(-1)
        else:
            q_p, k_p, v_p, e_p = sep_projs
            q = q_p(x_norm).view(B, T, H, K)
            k = k_p(x_norm).view(B, T, H, K)
            v = v_p(x_norm).view(B, T, H, K)
            eta = (torch.sigmoid(e_p(x_norm).float()) * (self.base_lr / K)).to(hidden_states.dtype).unsqueeze(-1)

        fuse_ln = stage >= 2
        o, _, _ = chunk_ttt_linear_rsi(
            q, k, v, self.ttt_norm_weight, self.ttt_norm_bias, eta, eps=self.eps, fuse_output_ln=fuse_ln
        )
        return hidden_states + self.o_proj(o.reshape(B, T, D))


# ============================================================================
# 4. Track 4: 8-GPU NVSwitch Context Parallelism (TTT-CP = 8)
# ============================================================================

class _UlyssesAllToAllSeqToHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, world_size: int):
        ctx.group = group
        ctx.world_size = world_size
        # x: [N_TENSORS, B, T_local, H, K] -> [N_TENSORS, B, T_total, H_local, K]
        NT_PACK, B, T_local, H, K = x.shape
        H_local = H // world_size
        x_send = x.view(NT_PACK, B, T_local, world_size, H_local, K).permute(3, 0, 1, 2, 4, 5).contiguous()
        x_recv = torch.empty_like(x_send)
        dist.all_to_all_single(x_recv, x_send, group=group)
        return x_recv.permute(1, 2, 0, 3, 4, 5).contiguous().view(NT_PACK, B, T_local * world_size, H_local, K)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        world_size = ctx.world_size
        NT_PACK, B, T_total, H_local, K = grad_out.shape
        T_local = T_total // world_size
        g_send = grad_out.view(NT_PACK, B, world_size, T_local, H_local, K).permute(2, 0, 1, 3, 4, 5).contiguous()
        g_recv = torch.empty_like(g_send)
        dist.all_to_all_single(g_recv, g_send, group=ctx.group)
        return g_recv.permute(1, 2, 3, 0, 4, 5).contiguous().view(NT_PACK, B, T_local, H_local * world_size, K), None, None


class _UlyssesAllToAllHeadToSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, world_size: int):
        ctx.group = group
        ctx.world_size = world_size
        # x: [B, T_total, H_local, K] -> [B, T_local, H, K]
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


def ttt_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta_local: torch.Tensor,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, T_local, H, K = q_local.shape
    H_local = H // world_size
    eta_pad = F.pad(eta_local, (0, K - 1))
    packed_in = torch.stack([q_local, k_local, v_local, eta_pad], dim=0)
    packed_full = _UlyssesAllToAllSeqToHead.apply(packed_in, group, world_size)
    q_full = packed_full[0]
    k_full = packed_full[1]
    v_full = packed_full[2]
    eta_full = packed_full[3, :, :, :, :1].contiguous()
    w_local = w[rank * H_local : (rank + 1) * H_local].contiguous()
    b_local = b[rank * H_local : (rank + 1) * H_local].contiguous()
    o_full, _, _ = chunk_ttt_linear_rsi(q_full, k_full, v_full, w_local, b_local, eta_full, eps=eps)
    return _UlyssesAllToAllHeadToSeq.apply(o_full, group, world_size)


class _AllGatherSeqAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, rank: int, world_size: int):
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        chunks = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(chunks, x.contiguous(), group=group)
        return torch.cat(chunks, dim=1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        world_size = ctx.world_size
        T_local = grad_out.shape[1] // world_size
        g_chunks = list(grad_out.split(T_local, dim=1))
        g_local = torch.empty_like(g_chunks[0].contiguous())
        dist.reduce_scatter(g_local, [c.contiguous() for c in g_chunks], op=dist.ReduceOp.SUM, group=ctx.group)
        return g_local, None, None, None


def ttt_cp8_ring_baseline(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta_local: torch.Tensor,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int = 8,
    upstream_op=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    T_local = q_local.shape[1]
    q_full = _AllGatherSeqAutograd.apply(q_local, group, rank, world_size)
    k_full = _AllGatherSeqAutograd.apply(k_local, group, rank, world_size)
    v_full = _AllGatherSeqAutograd.apply(v_local, group, rank, world_size)
    eta_full = _AllGatherSeqAutograd.apply(eta_local, group, rank, world_size)
    op = upstream_op if upstream_op is not None else chunk_ttt_linear_rsi
    o_full, _, _ = op(q_full, k_full, v_full, w, b, eta_full, eps=eps)
    return o_full[:, rank * T_local : (rank + 1) * T_local].contiguous()
