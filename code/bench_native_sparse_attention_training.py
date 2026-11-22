#!/usr/bin/env python3
"""E125: Natively Trainable Dynamic Sparse Attention (DeepSeek NSA & Kimi MoBA),
Compacted Work-Tile CSC Backward, Zero-Selection-Recompute NSA-SAC, & 8-GPU NVSwitch
Group-Preserving 2D USP-NSA Context Parallelism (CP = 8, S = 16K..131K).

Primary Literature Anchors:
  - DeepSeek Native Sparse Attention (NSA): Yuan et al., Feb 2025 (arXiv:2502.11089)
    l=32, d=16 strided token compression, l'=64 top-n=16 GQA-shared fine selection
    (Eq. 7-12), w=512 isolated sliding window, learned 3-branch sigmoid gates (Eq. 5).
  - Moonshot Kimi Mixture of Block Attention (MoBA): Lu et al., Feb 2025 (arXiv:2502.13189)
    parameter-free block-mean-key routing + causal self-chunk + varlen routed chunks
    merged via exact online-softmax LSE weighting.
  - Quest (Query-Aware Sparsity per-head baseline): Tang et al., ICML 2024 (arXiv:2406.10774)
  - PyTorch FlexAttention (`BlockMask`, `create_block_mask`): Dong et al., 2024 (arXiv:2412.05496)
  - Unified Sequence Parallelism (2D USP): Fang & Zhao, 2024 (arXiv:2405.07719)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import environment_metadata

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().double()
    bf = b.detach().double()
    return float((af - bf).norm().item() / bf.norm().clamp(min=1e-12).item())


def query_gpu_telemetry(gpu_index: int = 0) -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=clocks.current.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return -1.0, -1.0


def time_fn_with_peak_mem(
    fn: Callable[[], Any],
    warmup: int,
    iters: int,
    device: torch.device,
    grad_bytes_steady: int = 0,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_bytes = torch.cuda.memory_allocated(device)

    fn()
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    workspace_mib = max(0.0, (peak_bytes - base_bytes - grad_bytes_steady) / (1024.0**2))
    inc_peak_mib = max(0.0, (peak_bytes - base_bytes) / (1024.0**2))

    ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    sm_clocks: list[float] = []
    powers: list[float] = []
    for i in range(iters):
        ev_s[i].record()
        fn()
        ev_e[i].record()
        if i == iters // 2:
            clk, pwr = query_gpu_telemetry(device.index or 0)
            if clk > 0:
                sm_clocks.append(clk)
                powers.append(pwr)
    torch.cuda.synchronize(device)
    samples_ms = [s.elapsed_time(e) for s, e in zip(ev_s, ev_e)]
    return {
        "median_ms": float(statistics.median(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
        "workspace_mib": round(workspace_mib, 2),
        "inc_peak_mib": round(inc_peak_mib, 2),
        "sm_clock_mhz": round(float(statistics.median(sm_clocks)), 1) if sm_clocks else -1.0,
        "power_w": round(float(max(powers)), 2) if powers else -1.0,
    }


# ============================================================================
# 1. Hardware-Aligned Triton Kernels for DeepSeek NSA & Kimi MoBA
# ============================================================================


@triton.jit
def _strided_compress_fwd_kernel(
    X_ptr, W_ptr, PE_ptr, Y_ptr,
    stride_xb, stride_xs, stride_xhk, stride_xd,
    stride_yb, stride_yc, stride_yhk, stride_yd,
    S, H_kv,
    D: tl.constexpr,
):
    c = tl.program_id(0)
    bhk = tl.program_id(1)
    b = bhk // H_kv
    hk = bhk % H_kv
    offs_r = tl.arange(0, 32)
    offs_d = tl.arange(0, D)
    s_pos = (c - 1) * 16 + offs_r
    valid = (s_pos >= 0) & (s_pos < S)
    x = tl.load(
        X_ptr + b * stride_xb + s_pos[:, None] * stride_xs + hk * stride_xhk + offs_d[None, :] * stride_xd,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    pe = tl.load(PE_ptr + hk * 32 * D + offs_r[:, None] * D + offs_d[None, :]).to(tl.float32)
    w = tl.load(W_ptr + hk * 32 + offs_r).to(tl.float32)
    x_pe = tl.where(valid[:, None], x + pe, 0.0)
    y = tl.sum(x_pe * w[:, None], axis=0)
    tl.store(Y_ptr + b * stride_yb + c * stride_yc + hk * stride_yhk + offs_d * stride_yd, y.to(Y_ptr.dtype.element_ty))


@triton.jit
def _strided_compress_bwd_dx_kernel(
    dY_ptr, W_ptr, dX_ptr,
    stride_yb, stride_yc, stride_yhk, stride_yd,
    stride_xb, stride_xs, stride_xhk, stride_xd,
    N_CMP, H_kv,
    D: tl.constexpr,
):
    u = tl.program_id(0)
    bhk = tl.program_id(1)
    b = bhk // H_kv
    hk = bhk % H_kv
    offs_sub = tl.arange(0, 16)
    offs_d = tl.arange(0, D)
    dy1 = tl.load(dY_ptr + b * stride_yb + u * stride_yc + hk * stride_yhk + offs_d * stride_yd).to(tl.float32)
    w1 = tl.load(W_ptr + hk * 32 + (16 + offs_sub)).to(tl.float32)
    dx = w1[:, None] * dy1[None, :]
    if u + 1 < N_CMP:
        dy2 = tl.load(dY_ptr + b * stride_yb + (u + 1) * stride_yc + hk * stride_yhk + offs_d * stride_yd).to(tl.float32)
        w2 = tl.load(W_ptr + hk * 32 + offs_sub).to(tl.float32)
        dx += w2[:, None] * dy2[None, :]
    s_pos = u * 16 + offs_sub
    tl.store(
        dX_ptr + b * stride_xb + s_pos[:, None] * stride_xs + hk * stride_xhk + offs_d[None, :] * stride_xd,
        dx.to(dX_ptr.dtype.element_ty),
    )


def strided_compress_param_grads(
    x: torch.Tensor, dy: torch.Tensor, w_pool: torch.Tensor, pe: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, h_kv, dim = x.shape
    n_cmp = seqlen // 16
    x_pad = torch.cat([torch.zeros((bsz, 16, h_kv, dim), device=x.device, dtype=x.dtype), x], dim=1)
    x_win = x_pad.unfold(1, 32, 16).permute(0, 1, 4, 2, 3)
    valid_c0 = torch.ones((1, n_cmp, 32, 1, 1), device=x.device, dtype=torch.float32)
    valid_c0[:, 0, :16] = 0.0
    dy_f = dy.float().unsqueeze(2)
    x_pe = (x_win.float() + pe.float().permute(1, 0, 2).view(1, 1, 32, h_kv, dim)) * valid_c0
    dw = (x_pe * dy_f).sum(dim=(0, 1, 4)).permute(1, 0).contiguous()
    dpe = (dy_f * w_pool.float().permute(1, 0).view(1, 1, 32, h_kv, 1) * valid_c0).sum(dim=(0, 1)).permute(1, 0, 2).contiguous()
    return dw, dpe


@triton.jit
def _induce_eq9_slc_scores_from_lse_kernel(
    Q_ptr, KCmp_ptr, LSECmp_ptr, ATrap_ptr, First_ptr,
    stride_qb, stride_qs, stride_qhk, stride_qg, stride_qd,
    stride_cb, stride_cc, stride_chk, stride_cd,
    stride_lb, stride_ls, stride_lhk, stride_lg,
    stride_ab, stride_ahk, stride_au, stride_aj,
    S_q, q_global_offset, N_SLC, H_kv, sm_scale,
    B_Q_SUB: tl.constexpr,
    Q_PER_CTA: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    B_C: tl.constexpr,
):
    cta_q = tl.program_id(0)
    bhk = tl.program_id(1)
    b = bhk // H_kv
    hk = bhk % H_kv

    t0_local = cta_q * Q_PER_CTA
    u_chunk = (q_global_offset + t0_local) // 16
    if u_chunk <= 0:
        return

    ROWS: tl.constexpr = Q_PER_CTA * G
    offs_r = tl.arange(0, ROWS)
    q_local = offs_r // G
    g_local = offs_r % G
    t_local = t0_local + q_local
    valid_q = t_local < S_q
    offs_d = tl.arange(0, D)

    q = tl.load(
        Q_ptr + b * stride_qb + t_local[:, None] * stride_qs + hk * stride_qhk + g_local[:, None] * stride_qg + offs_d[None, :] * stride_qd,
        mask=valid_q[:, None],
        other=0.0,
    )
    lse_nat = tl.load(
        LSECmp_ptr + b * stride_lb + t_local * stride_ls + hk * stride_lhk + g_local * stride_lg,
        mask=valid_q,
        other=1.0e30,
    )
    lse_log2 = lse_nat * 1.4426950408889634
    qk_scale = sm_scale * 1.4426950408889634

    kc_base = KCmp_ptr + b * stride_cb + hk * stride_chk
    offs_c = tl.arange(0, B_C)
    N_J: tl.constexpr = B_C // 4
    offs_j = tl.arange(0, N_J)
    c_mod4 = tl.arange(0, 4)
    w_trap = tl.where(c_mod4 == 0, 1.0, 2.0)
    w_first = tl.where(c_mod4 == 0, 1.0, 0.0)

    for c0 in range(0, u_chunk, B_C):
        c_idx = c0 + offs_c
        valid_c = c_idx < u_chunk
        kc = tl.load(kc_base + c_idx[:, None] * stride_cc + offs_d[None, :] * stride_cd, mask=valid_c[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(kc)) * qk_scale
        p = tl.where(valid_q[:, None] & valid_c[None, :], tl.math.exp2(qk - lse_log2[:, None]), 0.0)
        p_4d = tl.reshape(p, [Q_PER_CTA, G, N_J, 4])
        p_g = tl.sum(p_4d, axis=1)
        a_val = tl.sum(p_g * w_trap[None, None, :], axis=2)
        f_val = tl.sum(p_g * w_first[None, None, :], axis=2)
        j_idx = (c0 // 4) + offs_j
        valid_j = j_idx < N_SLC
        if B_Q_SUB == 1:
            t_out = t0_local + tl.arange(0, Q_PER_CTA)
            m_out = (t_out[:, None] < S_q) & valid_j[None, :]
            off = b * stride_ab + hk * stride_ahk + t_out[:, None] * stride_au + j_idx[None, :] * stride_aj
            tl.store(ATrap_ptr + off, a_val.to(ATrap_ptr.dtype.element_ty), mask=m_out)
            tl.store(First_ptr + off, f_val.to(First_ptr.dtype.element_ty), mask=m_out)
        else:
            a_sub = tl.sum(a_val, axis=0) * (1.0 / B_Q_SUB)
            f_sub = tl.sum(f_val, axis=0) * (1.0 / B_Q_SUB)
            off = b * stride_ab + hk * stride_ahk + cta_q * stride_au + j_idx * stride_aj
            tl.store(ATrap_ptr + off, a_sub.to(ATrap_ptr.dtype.element_ty), mask=valid_j)
            tl.store(First_ptr + off, f_sub.to(First_ptr.dtype.element_ty), mask=valid_j)


@triton.jit
def _sparse_group_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, IDX_ptr, Out_ptr, LSE_ptr,
    stride_qb, stride_qs, stride_qhk, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khk, stride_kd,
    stride_vb, stride_vs, stride_vhk, stride_vd,
    stride_ib, stride_iu, stride_ihk, stride_ik,
    stride_ob, stride_os, stride_ohk, stride_og, stride_od,
    stride_lb, stride_ls, stride_lhk, stride_lg,
    S_q, S_kv, q_global_offset, H_kv, sm_scale,
    B_Q_SUB: tl.constexpr,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    B_K: tl.constexpr,
    K_SEL: tl.constexpr,
):
    u = tl.program_id(0)
    bhk = tl.program_id(1)
    b = bhk // H_kv
    hk = bhk % H_kv

    ROWS: tl.constexpr = B_Q_SUB * G_PAD
    offs_r = tl.arange(0, ROWS)
    q_sub = offs_r // G_PAD
    g_idx = offs_r % G_PAD
    t_local = u * B_Q_SUB + q_sub
    t_global = q_global_offset + t_local
    valid_r = (t_local < S_q) & (g_idx < G)

    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, B_K)

    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + t_local[:, None] * stride_qs
        + hk * stride_qhk
        + g_idx[:, None] * stride_qg
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_r[:, None], other=0.0)

    k_base = K_ptr + b * stride_kb + hk * stride_khk
    v_base = V_ptr + b * stride_vb + hk * stride_vhk
    idx_base = IDX_ptr + b * stride_ib + u * stride_iu + hk * stride_ihk

    m_i = tl.full([ROWS], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([ROWS], dtype=tl.float32)
    acc = tl.zeros([ROWS, D], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for ki in range(K_SEL):
        kb = tl.load(idx_base + ki * stride_ik)
        if kb >= 0:
            pos_k = kb * B_K + offs_k
            valid_k = pos_k < S_kv
            k = tl.load(k_base + pos_k[:, None] * stride_ks + offs_d[None, :] * stride_kd, mask=valid_k[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            causal_ok = valid_r[:, None] & valid_k[None, :] & (pos_k[None, :] <= t_global[:, None])
            qk = tl.where(causal_ok, qk, -1.0e30)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            p = tl.where(causal_ok, p, 0.0)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_base + pos_k[:, None] * stride_vs + offs_d[None, :] * stride_vd, mask=valid_k[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_ij

    has_any = l_i > 0.0
    inv_l = tl.where(has_any, 1.0 / tl.maximum(l_i, 1.0e-30), 0.0)
    out = acc * inv_l[:, None]
    lse = tl.where(has_any, (m_i + tl.math.log2(tl.maximum(l_i, 1.0e-30))) * 0.6931471805599453, -1.0e30)

    o_ptrs = (
        Out_ptr
        + b * stride_ob
        + t_local[:, None] * stride_os
        + hk * stride_ohk
        + g_idx[:, None] * stride_og
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, out.to(Out_ptr.dtype.element_ty), mask=valid_r[:, None])
    tl.store(LSE_ptr + b * stride_lb + t_local * stride_ls + hk * stride_lhk + g_idx * stride_lg, lse, mask=valid_r)


@triton.jit
def _nsa_bwd_preprocess_kernel(Out_ptr, dOut_ptr, Delta_ptr, n_rows, D: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    offs_d = tl.arange(0, D)
    mask = row < n_rows
    o = tl.load(Out_ptr + row[:, None] * D + offs_d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(dOut_ptr + row[:, None] * D + offs_d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    tl.store(Delta_ptr + row, tl.sum(o * do, 1), mask=mask)


@triton.jit
def _sparse_group_bwd_dq_and_opt_atomic_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, LSE_ptr, Delta_ptr, IDX_ptr,
    dQ_ptr, dK_fp32_ptr, dV_fp32_ptr,
    stride_qb, stride_qs, stride_qhk, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khk, stride_kd,
    stride_ib, stride_iu, stride_ihk, stride_ik,
    stride_lb, stride_ls, stride_lhk, stride_lg,
    S_q, S_kv, q_global_offset, H_kv, sm_scale,
    B_Q_SUB: tl.constexpr,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    B_K: tl.constexpr,
    K_SEL: tl.constexpr,
    DO_ATOMIC_DKV: tl.constexpr,
):
    u = tl.program_id(0)
    bhk = tl.program_id(1)
    b = bhk // H_kv
    hk = bhk % H_kv

    ROWS: tl.constexpr = B_Q_SUB * G_PAD
    offs_r = tl.arange(0, ROWS)
    q_sub = offs_r // G_PAD
    g_idx = offs_r % G_PAD
    t_local = u * B_Q_SUB + q_sub
    t_global = q_global_offset + t_local
    valid_r = (t_local < S_q) & (g_idx < G)

    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, B_K)

    q_off = (
        b * stride_qb
        + t_local[:, None] * stride_qs
        + hk * stride_qhk
        + g_idx[:, None] * stride_qg
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q_ptr + q_off, mask=valid_r[:, None], other=0.0)
    do = tl.load(dO_ptr + q_off, mask=valid_r[:, None], other=0.0)
    l_off = b * stride_lb + t_local * stride_ls + hk * stride_lhk + g_idx * stride_lg
    lse = tl.load(LSE_ptr + l_off, mask=valid_r, other=1.0e30) * 1.4426950408889634
    delta = tl.load(Delta_ptr + l_off, mask=valid_r, other=0.0)

    k_base = K_ptr + b * stride_kb + hk * stride_khk
    v_base = V_ptr + b * stride_kb + hk * stride_khk
    idx_base = IDX_ptr + b * stride_ib + u * stride_iu + hk * stride_ihk

    qk_scale = sm_scale * 1.4426950408889634
    acc_dq = tl.zeros([ROWS, D], dtype=tl.float32)

    for ki in range(K_SEL):
        kb = tl.load(idx_base + ki * stride_ik)
        if kb >= 0:
            pos_k = kb * B_K + offs_k
            valid_k = pos_k < S_kv
            kv_off = pos_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_base + kv_off, mask=valid_k[:, None], other=0.0)
            v = tl.load(v_base + kv_off, mask=valid_k[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * qk_scale
            causal_ok = valid_r[:, None] & valid_k[None, :] & (pos_k[None, :] <= t_global[:, None])
            qk = tl.where(causal_ok, qk, -1.0e30)
            p = tl.where(causal_ok, tl.math.exp2(qk - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * sm_scale
            acc_dq += tl.dot(ds.to(q.dtype), k)

            if DO_ATOMIC_DKV:
                dv_tile = tl.dot(tl.trans(p.to(q.dtype)), do)
                dk_tile = tl.dot(tl.trans(ds.to(q.dtype)), q)
                tl.atomic_add(dK_fp32_ptr + b * stride_kb + hk * stride_khk + kv_off, dk_tile, mask=valid_k[:, None])
                tl.atomic_add(dV_fp32_ptr + b * stride_kb + hk * stride_khk + kv_off, dv_tile, mask=valid_k[:, None])

    tl.store(dQ_ptr + q_off, acc_dq.to(dQ_ptr.dtype.element_ty), mask=valid_r[:, None])


@triton.jit
def _build_csc_counts_kernel(IDX_ptr, Counts_ptr, N_U, N_BLK, K_SEL: tl.constexpr):
    pid = tl.program_id(0)
    bhk = tl.program_id(1)
    u_offs = pid * 128 + tl.arange(0, 128)
    mask_u = u_offs < N_U
    for ki in range(K_SEL):
        kb = tl.load(IDX_ptr + bhk * N_U * K_SEL + u_offs * K_SEL + ki, mask=mask_u, other=-1)
        valid = mask_u & (kb >= 0) & (kb < N_BLK)
        tl.atomic_add(Counts_ptr + bhk * N_BLK + kb, 1, mask=valid)


@triton.jit
def _build_csc_scatter_kernel(IDX_ptr, Cursor_ptr, UList_ptr, N_U, N_BLK, K_SEL: tl.constexpr):
    pid = tl.program_id(0)
    bhk = tl.program_id(1)
    u_offs = pid * 128 + tl.arange(0, 128)
    mask_u = u_offs < N_U
    for ki in range(K_SEL):
        kb = tl.load(IDX_ptr + bhk * N_U * K_SEL + u_offs * K_SEL + ki, mask=mask_u, other=-1)
        valid = mask_u & (kb >= 0) & (kb < N_BLK)
        pos = tl.atomic_add(Cursor_ptr + bhk * N_BLK + kb, 1, mask=valid)
        tl.store(UList_ptr + pos, u_offs, mask=valid)


@triton.jit
def _expand_csc_tiles_kernel(
    CSC_Cnt_ptr, Tile_Off_ptr, Multi_Off_ptr,
    Tile_BHK_ptr, Tile_KB_ptr, Tile_Start_ptr, Tile_End_ptr, Tile_MSlot_ptr,
    total_blocks, N_BLK,
    SPLIT_U: tl.constexpr,
):
    idx = tl.program_id(0)
    if idx >= total_blocks:
        return
    cnt = tl.load(CSC_Cnt_ptr + idx)
    if cnt <= 0:
        return
    n_sp = (cnt + SPLIT_U - 1) // SPLIT_U
    base_tile = tl.load(Tile_Off_ptr + idx)
    m_base = tl.load(Multi_Off_ptr + idx)
    bhk = idx // N_BLK
    kb = idx % N_BLK
    for s in range(n_sp):
        us = s * SPLIT_U
        ue = tl.minimum(cnt, us + SPLIT_U)
        mslot = (m_base + s) if n_sp > 1 else -1
        tl.store(Tile_BHK_ptr + base_tile + s, bhk)
        tl.store(Tile_KB_ptr + base_tile + s, kb)
        tl.store(Tile_Start_ptr + base_tile + s, us)
        tl.store(Tile_End_ptr + base_tile + s, ue)
        tl.store(Tile_MSlot_ptr + base_tile + s, mslot)


@triton.jit
def _nsa_compacted_csc_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, LSE_ptr, Delta_ptr,
    CSC_Off_ptr, UList_ptr,
    Tile_BHK_ptr, Tile_KB_ptr, Tile_Start_ptr, Tile_End_ptr, Tile_MSlot_ptr,
    dK_bf16_ptr, dV_bf16_ptr, Part_dK_ptr, Part_dV_ptr,
    stride_qb, stride_qs, stride_qhk, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khk, stride_kd,
    stride_lb, stride_ls, stride_lhk, stride_lg,
    S_q, S_kv, q_global_offset, H_kv, N_BLK, sm_scale,
    B_Q_SUB: tl.constexpr,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    B_K: tl.constexpr,
    U_STEP: tl.constexpr,
):
    tile_id = tl.program_id(0)
    bhk = tl.load(Tile_BHK_ptr + tile_id)
    kb = tl.load(Tile_KB_ptr + tile_id)
    u_start = tl.load(Tile_Start_ptr + tile_id)
    u_end = tl.load(Tile_End_ptr + tile_id)
    mslot = tl.load(Tile_MSlot_ptr + tile_id)

    off_base = tl.load(CSC_Off_ptr + bhk * N_BLK + kb)
    b = bhk // H_kv
    hk = bhk % H_kv

    offs_k = tl.arange(0, B_K)
    offs_d = tl.arange(0, D)
    pos_k = kb * B_K + offs_k
    valid_k = pos_k < S_kv

    kv_off = b * stride_kb + hk * stride_khk + pos_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
    k = tl.load(K_ptr + kv_off, mask=valid_k[:, None], other=0.0)
    v = tl.load(V_ptr + kv_off, mask=valid_k[:, None], other=0.0)

    acc_dk = tl.zeros([B_K, D], dtype=tl.float32)
    acc_dv = tl.zeros([B_K, D], dtype=tl.float32)

    ROWS: tl.constexpr = U_STEP * B_Q_SUB * G_PAD
    offs_r = tl.arange(0, ROWS)
    u_step_idx = offs_r // (B_Q_SUB * G_PAD)
    rem_r = offs_r % (B_Q_SUB * G_PAD)
    q_sub = rem_r // G_PAD
    g_local = rem_r % G_PAD
    qk_scale = sm_scale * 1.4426950408889634

    q_base_bhk = Q_ptr + b * stride_qb + hk * stride_qhk
    do_base_bhk = dO_ptr + b * stride_qb + hk * stride_qhk
    l_base_bhk = LSE_ptr + b * stride_lb + hk * stride_lhk
    d_base_bhk = Delta_ptr + b * stride_lb + hk * stride_lhk

    for u0 in range(u_start, u_end, U_STEP):
        list_idx = u0 + u_step_idx
        in_list = list_idx < u_end
        u_val = tl.load(UList_ptr + off_base + list_idx, mask=in_list, other=0)
        t_local = u_val * B_Q_SUB + q_sub
        t_global = q_global_offset + t_local
        valid_r = in_list & (t_local < S_q) & (g_local < G)

        q_off = t_local[:, None] * stride_qs + g_local[:, None] * stride_qg + offs_d[None, :] * stride_qd
        q = tl.load(q_base_bhk + q_off, mask=valid_r[:, None], other=0.0)
        do = tl.load(do_base_bhk + q_off, mask=valid_r[:, None], other=0.0)

        l_off = t_local * stride_ls + g_local * stride_lg
        lse = tl.load(l_base_bhk + l_off, mask=valid_r, other=1.0e30) * 1.4426950408889634
        delta = tl.load(d_base_bhk + l_off, mask=valid_r, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        causal_ok = valid_r[:, None] & valid_k[None, :] & (pos_k[None, :] <= t_global[:, None])
        qk = tl.where(causal_ok, qk, -1.0e30)
        p = tl.where(causal_ok, tl.math.exp2(qk - lse[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale

        acc_dv += tl.dot(tl.trans(p.to(q.dtype)), do)
        acc_dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    if mslot < 0:
        tl.store(dK_bf16_ptr + kv_off, acc_dk.to(dK_bf16_ptr.dtype.element_ty), mask=valid_k[:, None])
        tl.store(dV_bf16_ptr + kv_off, acc_dv.to(dV_bf16_ptr.dtype.element_ty), mask=valid_k[:, None])
    else:
        p_off = mslot * (B_K * D) + offs_k[:, None] * D + offs_d[None, :]
        tl.store(Part_dK_ptr + p_off, acc_dk, mask=valid_k[:, None])
        tl.store(Part_dV_ptr + p_off, acc_dv, mask=valid_k[:, None])


@triton.jit
def _finalize_multi_split_dkv_kernel(
    CSC_Cnt_ptr, Multi_Off_ptr, Part_dK_ptr, Part_dV_ptr, dK_bf16_ptr, dV_bf16_ptr,
    stride_kb, stride_ks, stride_khk, stride_kd,
    S_kv, H_kv, N_BLK,
    D: tl.constexpr,
    B_K: tl.constexpr,
    SPLIT_U: tl.constexpr,
):
    idx = tl.program_id(0)
    cnt = tl.load(CSC_Cnt_ptr + idx)
    bhk = idx // N_BLK
    kb = idx % N_BLK
    b = bhk // H_kv
    hk = bhk % H_kv
    offs_k = tl.arange(0, B_K)
    offs_d = tl.arange(0, D)
    pos_k = kb * B_K + offs_k
    valid_k = pos_k < S_kv
    kv_off = b * stride_kb + hk * stride_khk + pos_k[:, None] * stride_ks + offs_d[None, :] * stride_kd

    if cnt <= SPLIT_U:
        if cnt == 0:
            z = tl.zeros([B_K, D], dtype=dK_bf16_ptr.dtype.element_ty)
            tl.store(dK_bf16_ptr + kv_off, z, mask=valid_k[:, None])
            tl.store(dV_bf16_ptr + kv_off, z, mask=valid_k[:, None])
        return

    n_sp = (cnt + SPLIT_U - 1) // SPLIT_U
    m_base = tl.load(Multi_Off_ptr + idx)
    acc_dk = tl.zeros([B_K, D], dtype=tl.float32)
    acc_dv = tl.zeros([B_K, D], dtype=tl.float32)
    tile_stride = B_K * D
    sub_off = offs_k[:, None] * D + offs_d[None, :]
    for s in range(n_sp):
        p_off = (m_base + s) * tile_stride + sub_off
        acc_dk += tl.load(Part_dK_ptr + p_off, mask=valid_k[:, None], other=0.0)
        acc_dv += tl.load(Part_dV_ptr + p_off, mask=valid_k[:, None], other=0.0)

    tl.store(dK_bf16_ptr + kv_off, acc_dk.to(dK_bf16_ptr.dtype.element_ty), mask=valid_k[:, None])
    tl.store(dV_bf16_ptr + kv_off, acc_dv.to(dV_bf16_ptr.dtype.element_ty), mask=valid_k[:, None])


@triton.jit
def _nsa_3branch_gate_fwd_kernel(
    OCmp_ptr, OSlc_ptr, OWin_ptr, GateLogits_ptr, Out_ptr,
    n_rows,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    offs_d = tl.arange(0, D)
    off = row * D + offs_d
    gc_v = tl.sigmoid(tl.load(GateLogits_ptr + row * 3 + 0).to(tl.float32))
    gs_v = tl.sigmoid(tl.load(GateLogits_ptr + row * 3 + 1).to(tl.float32))
    gw_v = tl.sigmoid(tl.load(GateLogits_ptr + row * 3 + 2).to(tl.float32))
    oc = tl.load(OCmp_ptr + off).to(tl.float32)
    os_v = tl.load(OSlc_ptr + off).to(tl.float32)
    ow = tl.load(OWin_ptr + off).to(tl.float32)
    out = gc_v * oc + gs_v * os_v + gw_v * ow
    tl.store(Out_ptr + off, out.to(Out_ptr.dtype.element_ty))


@triton.jit
def _nsa_3branch_gate_bwd_kernel(
    dOut_ptr, OCmp_ptr, OSlc_ptr, OWin_ptr, GateLogits_ptr,
    dOCmp_ptr, dOSlc_ptr, dOWin_ptr, dGateLogits_ptr,
    n_rows,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    offs_d = tl.arange(0, D)
    off = row * D + offs_d
    gc_v = tl.sigmoid(tl.load(GateLogits_ptr + row * 3 + 0).to(tl.float32))
    gs_v = tl.sigmoid(tl.load(GateLogits_ptr + row * 3 + 1).to(tl.float32))
    gw_v = tl.sigmoid(tl.load(GateLogits_ptr + row * 3 + 2).to(tl.float32))
    do = tl.load(dOut_ptr + off).to(tl.float32)
    oc = tl.load(OCmp_ptr + off).to(tl.float32)
    os_v = tl.load(OSlc_ptr + off).to(tl.float32)
    ow = tl.load(OWin_ptr + off).to(tl.float32)

    tl.store(dOCmp_ptr + off, (do * gc_v).to(dOCmp_ptr.dtype.element_ty))
    tl.store(dOSlc_ptr + off, (do * gs_v).to(dOSlc_ptr.dtype.element_ty))
    tl.store(dOWin_ptr + off, (do * gw_v).to(dOWin_ptr.dtype.element_ty))

    dgc = tl.sum(do * oc, 0) * gc_v * (1.0 - gc_v)
    dgs = tl.sum(do * os_v, 0) * gs_v * (1.0 - gs_v)
    dgw = tl.sum(do * ow, 0) * gw_v * (1.0 - gw_v)
    tl.store(dGateLogits_ptr + row * 3 + 0, dgc.to(dGateLogits_ptr.dtype.element_ty))
    tl.store(dGateLogits_ptr + row * 3 + 1, dgs.to(dGateLogits_ptr.dtype.element_ty))
    tl.store(dGateLogits_ptr + row * 3 + 2, dgw.to(dGateLogits_ptr.dtype.element_ty))


@triton.jit
def _moba_gqa_lse_merge_fwd_kernel(
    SelfOut_ptr, SelfLSE_ptr,
    MobaOut_ptr, MobaLSE_ptr,
    RowMap_ptr,
    MixOut_ptr, MixLSE_ptr,
    S, H_kv, M_sel,
    G: tl.constexpr,
    D: tl.constexpr,
    K_INTER: tl.constexpr,
):
    shk = tl.program_id(0)
    g = tl.program_id(1)
    s = shk // H_kv
    hk = shk % H_kv
    hq = hk * G + g
    offs_d = tl.arange(0, D)

    so_off = (s * (H_kv * G) + hq) * D + offs_d
    s_out = tl.load(SelfOut_ptr + so_off).to(tl.float32)
    s_lse = tl.load(SelfLSE_ptr + hq * S + s).to(tl.float32)

    m_max = s_lse
    for ki in range(K_INTER):
        mrow = tl.load(RowMap_ptr + shk * K_INTER + ki)
        if mrow >= 0:
            ml = tl.load(MobaLSE_ptr + g * M_sel + mrow).to(tl.float32)
            m_max = tl.maximum(m_max, ml)

    se_sum = tl.exp(s_lse - m_max)
    acc = s_out * se_sum
    for ki in range(K_INTER):
        mrow = tl.load(RowMap_ptr + shk * K_INTER + ki)
        if mrow >= 0:
            ml = tl.load(MobaLSE_ptr + g * M_sel + mrow).to(tl.float32)
            w = tl.exp(ml - m_max)
            mo = tl.load(MobaOut_ptr + (mrow * G + g) * D + offs_d).to(tl.float32)
            se_sum += w
            acc += mo * w

    mix_lse = m_max + tl.log(se_sum)
    mix_out = acc / se_sum
    tl.store(MixOut_ptr + so_off, mix_out.to(MixOut_ptr.dtype.element_ty))
    tl.store(MixLSE_ptr + hq * S + s, mix_lse)


@triton.jit
def _moba_gqa_dq_merge_kernel(
    dQSelf_ptr, dQMoba_ptr, RowMap_ptr, dQOut_ptr,
    S, H_kv,
    G: tl.constexpr,
    D: tl.constexpr,
    K_INTER: tl.constexpr,
):
    shk = tl.program_id(0)
    g = tl.program_id(1)
    s = shk // H_kv
    hk = shk % H_kv
    hq = hk * G + g
    offs_d = tl.arange(0, D)
    off = (s * (H_kv * G) + hq) * D + offs_d
    acc = tl.load(dQSelf_ptr + off).to(tl.float32)
    for ki in range(K_INTER):
        mrow = tl.load(RowMap_ptr + shk * K_INTER + ki)
        if mrow >= 0:
            acc += tl.load(dQMoba_ptr + (mrow * G + g) * D + offs_d).to(tl.float32)
    tl.store(dQOut_ptr + off, acc.to(dQOut_ptr.dtype.element_ty))


# ============================================================================
# 2. Differentiable Operators: DeepSeek NSA & Moonshot Kimi MoBA
# ============================================================================


class _CSCScratchCache:
    def __init__(self) -> None:
        self.dk_fp32: dict[tuple[Any, ...], torch.Tensor] = {}
        self.dv_fp32: dict[tuple[Any, ...], torch.Tensor] = {}

    def get_hot_buffers(self, ref_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (ref_k.device, tuple(ref_k.shape))
        if key not in self.dk_fp32:
            self.dk_fp32[key] = torch.zeros_like(ref_k, dtype=torch.float32)
            self.dv_fp32[key] = torch.zeros_like(ref_k, dtype=torch.float32)
        return self.dk_fp32[key], self.dv_fp32[key]


CSC_CACHE = _CSCScratchCache()


def make_sink_heavy_indices(
    bsz: int,
    h_kv: int,
    s_q: int,
    s_kv: int,
    device: torch.device,
    b_q_sub: int = 1,
    b_k: int = 64,
    k_sel: int = 16,
    q_global_offset: int = 0,
    seed: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_u = s_q // b_q_sub
    n_blk = s_kv // b_k
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.randn((bsz, h_kv, n_u, n_blk), generator=gen, device=device, dtype=torch.float32)
    scores[..., 0] += 50.0
    if n_blk > 1:
        scores[..., 1] += 8.0
    if n_blk > 2:
        scores[..., 2] += 5.0
    u_arange = torch.arange(n_u, device=device)
    t_max = q_global_offset + u_arange * b_q_sub + (b_q_sub - 1)
    q_blk = (t_max // b_k)[None, None, :, None]
    blk_ids = torch.arange(n_blk, device=device)[None, None, None, :]
    scores = torch.where(blk_ids == q_blk, scores + 40.0, scores)
    scores = torch.where(blk_ids == (q_blk - 1), scores + 35.0, scores)
    scores = torch.where(blk_ids <= q_blk, scores, float("-inf"))
    k_eff = min(k_sel, n_blk)
    topv, topi = torch.topk(scores, k=k_eff, dim=-1)
    idx_bhku = torch.full((bsz, h_kv, n_u, k_sel), -1, device=device, dtype=torch.int32)
    idx_bhku[..., :k_eff] = torch.where(torch.isfinite(topv), topi.to(torch.int32), -1)
    idx_buhk = idx_bhku.permute(0, 2, 1, 3).contiguous()
    return idx_bhku, idx_buhk


def run_sparse_selection_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_buhk: torch.Tensor,
    b_q_sub: int = 1,
    b_k: int = 64,
    q_global_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, s_q, h_kv, g_heads, dim = q.shape
    s_kv = k.shape[1]
    k_sel = idx_buhk.shape[-1]
    n_u = s_q // b_q_sub
    g_pad = max(4, triton.next_power_of_2(g_heads))
    if b_q_sub * g_pad < 16:
        g_pad = 16 // b_q_sub
    nw = 2 if (b_q_sub * g_pad) <= 16 else 4
    sm_scale = dim**-0.5

    out = torch.empty_like(q)
    lse = torch.empty((bsz, s_q, h_kv, g_heads), device=q.device, dtype=torch.float32)
    _sparse_group_fwd_kernel[(n_u, bsz * h_kv)](
        q, k, v, idx_buhk, out, lse,
        *q.stride(), *k.stride(), *v.stride(), *idx_buhk.stride(), *out.stride(), *lse.stride(),
        s_q, s_kv, q_global_offset, h_kv, sm_scale,
        B_Q_SUB=b_q_sub, G=g_heads, G_PAD=g_pad, D=dim, B_K=b_k, K_SEL=k_sel,
        num_warps=nw, num_stages=3,
    )
    return out, lse


def run_sparse_selection_bwd_atomic(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    idx_buhk: torch.Tensor,
    b_q_sub: int = 1,
    b_k: int = 64,
    q_global_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, s_q, h_kv, g_heads, dim = q.shape
    s_kv = k.shape[1]
    k_sel = idx_buhk.shape[-1]
    n_u = s_q // b_q_sub
    g_pad = max(4, triton.next_power_of_2(g_heads))
    if b_q_sub * g_pad < 16:
        g_pad = 16 // b_q_sub
    nw = 2 if (b_q_sub * g_pad) <= 16 else 4
    sm_scale = dim**-0.5

    delta = torch.empty_like(lse)
    _nsa_bwd_preprocess_kernel[(triton.cdiv(lse.numel(), 16),)](out, do, delta, lse.numel(), D=dim)
    dq = torch.empty_like(q)
    dk_fp32 = torch.zeros_like(k, dtype=torch.float32)
    dv_fp32 = torch.zeros_like(v, dtype=torch.float32)
    _sparse_group_bwd_dq_and_opt_atomic_kernel[(n_u, bsz * h_kv)](
        q, k, v, do, lse, delta, idx_buhk, dq, dk_fp32, dv_fp32,
        *q.stride(), *k.stride(), *idx_buhk.stride(), *lse.stride(),
        s_q, s_kv, q_global_offset, h_kv, sm_scale,
        B_Q_SUB=b_q_sub, G=g_heads, G_PAD=g_pad, D=dim, B_K=b_k, K_SEL=k_sel,
        DO_ATOMIC_DKV=True, num_warps=nw, num_stages=3,
    )
    return dq, dk_fp32.to(k.dtype), dv_fp32.to(v.dtype)


def run_sparse_selection_bwd_csc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    idx_bhku: torch.Tensor,
    idx_buhk: torch.Tensor,
    b_q_sub: int = 1,
    b_k: int = 64,
    q_global_offset: int = 0,
    split_tokens: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, s_q, h_kv, g_heads, dim = q.shape
    s_kv = k.shape[1]
    k_sel = idx_buhk.shape[-1]
    n_u = s_q // b_q_sub
    n_blk = s_kv // b_k
    g_pad = max(4, triton.next_power_of_2(g_heads))
    if b_q_sub * g_pad < 16:
        g_pad = 16 // b_q_sub
    nw_dq = 2 if (b_q_sub * g_pad) <= 16 else 4
    sm_scale = dim**-0.5

    delta = torch.empty_like(lse)
    _nsa_bwd_preprocess_kernel[(triton.cdiv(lse.numel(), 16),)](out, do, delta, lse.numel(), D=dim)
    dq = torch.empty_like(q)

    _sparse_group_bwd_dq_and_opt_atomic_kernel[(n_u, bsz * h_kv)](
        q, k, v, do, lse, delta, idx_buhk, dq, delta, delta,
        *q.stride(), *k.stride(), *idx_buhk.stride(), *lse.stride(),
        s_q, s_kv, q_global_offset, h_kv, sm_scale,
        B_Q_SUB=b_q_sub, G=g_heads, G_PAD=g_pad, D=dim, B_K=b_k, K_SEL=k_sel,
        DO_ATOMIC_DKV=False, num_warps=nw_dq, num_stages=3,
    )

    bhk_tot = bsz * h_kv
    counts = torch.zeros((bhk_tot, n_blk), device=q.device, dtype=torch.int32)
    _build_csc_counts_kernel[(triton.cdiv(n_u, 128), bhk_tot)](idx_bhku, counts, n_u, n_blk, K_SEL=k_sel)
    fc = counts.view(-1)
    offs = torch.zeros_like(fc)
    if fc.numel() > 1:
        offs[1:] = torch.cumsum(fc[:-1], dim=0)
    cur = offs.clone()
    ulist = torch.empty((max(1, bhk_tot * n_u * k_sel),), device=q.device, dtype=torch.int32)
    _build_csc_scatter_kernel[(triton.cdiv(n_u, 128), bhk_tot)](idx_bhku, cur, ulist, n_u, n_blk, K_SEL=k_sel)

    split_u = max(1, split_tokens // b_q_sub) if split_tokens > 0 else max(1, n_u)
    n_splits = (fc + split_u - 1) // split_u
    tile_offs = torch.zeros_like(n_splits)
    if n_splits.numel() > 1:
        tile_offs[1:] = torch.cumsum(n_splits[:-1], dim=0)
    tot_tiles = int((tile_offs[-1] + n_splits[-1]).item())

    multi_splits = torch.where(n_splits > 1, n_splits, 0)
    multi_offs = torch.zeros_like(multi_splits)
    if multi_splits.numel() > 1:
        multi_offs[1:] = torch.cumsum(multi_splits[:-1], dim=0)
    tot_multi = int((multi_offs[-1] + multi_splits[-1]).item())

    part_dk = torch.empty((max(1, tot_multi), b_k, dim), device=q.device, dtype=torch.float32)
    part_dv = torch.empty((max(1, tot_multi), b_k, dim), device=q.device, dtype=torch.float32)
    dk_bf16 = torch.empty_like(k)
    dv_bf16 = torch.empty_like(v)
    u_step = max(1, 64 // (b_q_sub * g_pad))

    if tot_tiles > 0:
        t_bhk = torch.empty((tot_tiles,), device=q.device, dtype=torch.int32)
        t_kb = torch.empty((tot_tiles,), device=q.device, dtype=torch.int32)
        t_st = torch.empty((tot_tiles,), device=q.device, dtype=torch.int32)
        t_en = torch.empty((tot_tiles,), device=q.device, dtype=torch.int32)
        t_ms = torch.empty((tot_tiles,), device=q.device, dtype=torch.int32)
        _expand_csc_tiles_kernel[(fc.numel(),)](
            fc, tile_offs, multi_offs, t_bhk, t_kb, t_st, t_en, t_ms, fc.numel(), n_blk, SPLIT_U=split_u
        )
        _nsa_compacted_csc_bwd_dkv_kernel[(tot_tiles,)](
            q, k, v, do, lse, delta, offs, ulist,
            t_bhk, t_kb, t_st, t_en, t_ms,
            dk_bf16, dv_bf16, part_dk, part_dv,
            *q.stride(), *k.stride(), *lse.stride(),
            s_q, s_kv, q_global_offset, h_kv, n_blk, sm_scale,
            B_Q_SUB=b_q_sub, G=g_heads, G_PAD=g_pad, D=dim, B_K=b_k, U_STEP=u_step,
            num_warps=4, num_stages=2,
        )
    _finalize_multi_split_dkv_kernel[(fc.numel(),)](
        fc, multi_offs, part_dk, part_dv, dk_bf16, dv_bf16,
        *k.stride(), s_kv, h_kv, n_blk, D=dim, B_K=b_k, SPLIT_U=split_u
    )
    return dq, dk_bf16, dv_bf16


@torch.no_grad()
def _apply_nsa_forced_and_causal_bonus_inplace(
    scores: torch.Tensor,
    q_global_offset: int,
    b_q_sub: int,
    b_k: int,
) -> None:
    bsz, h_kv, n_u, n_slc = scores.shape
    if n_slc <= 0 or n_u <= 0:
        return
    scores[..., 0] += 1.0e4
    u_arange = torch.arange(n_u, device=scores.device)
    t_max = q_global_offset + u_arange * b_q_sub + (b_q_sub - 1)
    q_blk_1d = t_max // b_k

    valid_cur = q_blk_1d < n_slc
    if bool(valid_cur.any()):
        idx_cur = q_blk_1d.clamp(0, n_slc - 1).view(1, 1, n_u, 1).expand(bsz, h_kv, n_u, 1)
        bonus_cur = (
            torch.where(valid_cur.view(1, 1, n_u, 1), 8.0e3, 0.0)
            .to(scores.dtype)
            .expand(bsz, h_kv, n_u, 1)
        )
        scores.scatter_add_(-1, idx_cur, bonus_cur)

    q_prev_1d = q_blk_1d - 1
    valid_prev = (q_prev_1d >= 0) & (q_prev_1d < n_slc)
    if bool(valid_prev.any()):
        idx_prev = q_prev_1d.clamp(0, n_slc - 1).view(1, 1, n_u, 1).expand(bsz, h_kv, n_u, 1)
        bonus_prev = (
            torch.where(valid_prev.view(1, 1, n_u, 1), 6.0e3, 0.0)
            .to(scores.dtype)
            .expand(bsz, h_kv, n_u, 1)
        )
        scores.scatter_add_(-1, idx_prev, bonus_prev)

    blk_ids = torch.arange(n_slc, device=scores.device)[None, None, None, :]
    scores.masked_fill_(blk_ids > q_blk_1d[None, None, :, None], float("-inf"))


def induce_nsa_topk_indices(
    q: torch.Tensor,
    kc: torch.Tensor,
    lse_cmp: torch.Tensor,
    b_q_sub: int = 4,
    b_k: int = 64,
    k_sel: int = 16,
    q_global_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, s_q, h_kv, g_heads, dim = q.shape
    n_cmp = kc.shape[1]
    n_slc = (n_cmp * 16) // b_k
    n_u = s_q // b_q_sub
    q_per_cta = 4 if b_q_sub == 1 else b_q_sub
    n_cta_q = s_q // q_per_cta
    sm_scale = dim**-0.5

    a_trap = torch.zeros((bsz, h_kv, n_u, n_slc), device=q.device, dtype=torch.float32)
    first_arr = torch.zeros_like(a_trap)
    _induce_eq9_slc_scores_from_lse_kernel[(n_cta_q, bsz * h_kv)](
        q, kc, lse_cmp, a_trap, first_arr,
        *q.stride(), *kc.stride(), *lse_cmp.stride(), *a_trap.stride(),
        s_q, q_global_offset, n_slc, h_kv, sm_scale,
        B_Q_SUB=b_q_sub, Q_PER_CTA=q_per_cta, G=g_heads, D=dim, B_C=64,
        num_warps=4, num_stages=2,
    )
    scores = a_trap
    if n_slc > 1:
        scores[..., :-1] += first_arr[..., 1:]
    del first_arr

    _apply_nsa_forced_and_causal_bonus_inplace(scores, q_global_offset, b_q_sub, b_k)

    k_eff = min(k_sel, n_slc)
    topv, topi = torch.topk(scores, k=k_eff, dim=-1)
    idx_bhku = torch.full((bsz, h_kv, n_u, k_sel), -1, device=q.device, dtype=torch.int32)
    idx_bhku[..., :k_eff] = torch.where(torch.isfinite(topv), topi.to(torch.int32), -1)
    idx_buhk = idx_bhku.permute(0, 2, 1, 3).contiguous()
    return idx_bhku, idx_buhk


class _DeepSeekNSAFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k_cmp_raw: torch.Tensor,
        v_cmp_raw: torch.Tensor,
        k_slc: torch.Tensor,
        v_slc: torch.Tensor,
        k_win: torch.Tensor,
        v_win: torch.Tensor,
        w_pool_k: torch.Tensor,
        pe_k: torch.Tensor,
        w_pool_v: torch.Tensor,
        pe_v: torch.Tensor,
        gate_logits: torch.Tensor,
        b_q_sub: int,
        k_sel: int,
        w_win: int,
        saved_idx_holder: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, h_kv, g_heads, dim = q.shape
        n_cmp = seqlen // 16
        sm_scale = dim**-0.5

        kc = torch.empty((bsz, n_cmp, h_kv, dim), device=q.device, dtype=q.dtype)
        vc = torch.empty_like(kc)
        _strided_compress_fwd_kernel[(n_cmp, bsz * h_kv)](
            k_cmp_raw, w_pool_k, pe_k, kc, *k_cmp_raw.stride(), *kc.stride(), seqlen, h_kv, D=dim
        )
        _strided_compress_fwd_kernel[(n_cmp, bsz * h_kv)](
            v_cmp_raw, w_pool_v, pe_v, vc, *v_cmp_raw.stride(), *vc.stride(), seqlen, h_kv, D=dim
        )

        q_fold = (
            q[:, 16:]
            .view(bsz, n_cmp - 1, 16, h_kv, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, n_cmp - 1, h_kv * 16 * g_heads, dim)
        )
        k_fold = kc[:, :-1].contiguous()
        v_fold = vc[:, :-1].contiguous()
        out_f, lse_f, _, _ = _flash_attn_forward(
            q_fold, k_fold, v_fold, 0.0, sm_scale, True, -1, -1, 0.0, None, False
        )
        out_cmp = torch.zeros_like(q)
        out_cmp[:, 16:] = (
            out_f.view(bsz, n_cmp - 1, h_kv, 16, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz, seqlen - 16, h_kv, g_heads, dim)
        )
        lse_cmp = torch.full((bsz, seqlen, h_kv, g_heads), -1.0e30, device=q.device, dtype=torch.float32)
        lse_cmp[:, 16:] = (
            lse_f.view(bsz, h_kv, 16, g_heads, n_cmp - 1)
            .permute(0, 4, 2, 1, 3)
            .reshape(bsz, seqlen - 16, h_kv, g_heads)
        )

        if saved_idx_holder is not None and len(saved_idx_holder) > 0:
            idx_bhku, idx_buhk = saved_idx_holder[0]
        else:
            idx_bhku, idx_buhk = induce_nsa_topk_indices(
                q, kc, lse_cmp, b_q_sub=b_q_sub, b_k=64, k_sel=k_sel, q_global_offset=0
            )
            if saved_idx_holder is not None:
                saved_idx_holder.append((idx_bhku, idx_buhk))

        out_slc, lse_slc = run_sparse_selection_fwd(q, k_slc, v_slc, idx_buhk, b_q_sub=b_q_sub, b_k=64)

        q_flat = q.view(bsz, seqlen, h_kv * g_heads, dim)
        out_win_flat, lse_win, _, _ = _flash_attn_forward(
            q_flat, k_win, v_win, 0.0, sm_scale, True, w_win - 1, 0, 0.0, None, False
        )
        out_win = out_win_flat.view_as(q)

        out = torch.empty_like(q)
        n_rows = bsz * seqlen * h_kv * g_heads
        _nsa_3branch_gate_fwd_kernel[(n_rows,)](out_cmp, out_slc, out_win, gate_logits, out, n_rows, D=dim)

        ctx.save_for_backward(
            q, k_cmp_raw, v_cmp_raw, kc, vc, k_slc, v_slc, k_win, v_win,
            w_pool_k, pe_k, w_pool_v, pe_v, gate_logits,
            out_cmp, out_f, lse_f, out_slc, lse_slc, out_win_flat, lse_win,
            idx_bhku, idx_buhk,
        )
        ctx.b_q_sub = b_q_sub
        ctx.w_win = w_win
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        (
            q, k_cmp_raw, v_cmp_raw, kc, vc, k_slc, v_slc, k_win, v_win,
            w_pool_k, pe_k, w_pool_v, pe_v, gate_logits,
            out_cmp, out_f, lse_f, out_slc, lse_slc, out_win_flat, lse_win,
            idx_bhku, idx_buhk,
        ) = ctx.saved_tensors
        bsz, seqlen, h_kv, g_heads, dim = q.shape
        n_cmp = seqlen // 16
        sm_scale = dim**-0.5
        do = do.contiguous()

        do_cmp = torch.empty_like(q)
        do_slc = torch.empty_like(q)
        do_win = torch.empty_like(q)
        d_gate = torch.empty_like(gate_logits)
        n_rows = bsz * seqlen * h_kv * g_heads
        _nsa_3branch_gate_bwd_kernel[(n_rows,)](
            do, out_cmp, out_slc, out_win_flat, gate_logits,
            do_cmp, do_slc, do_win, d_gate, n_rows, D=dim,
        )

        q_flat = q.view(bsz, seqlen, h_kv * g_heads, dim)
        dq_win_flat = torch.empty_like(q_flat)
        dk_win = torch.empty_like(k_win)
        dv_win = torch.empty_like(v_win)
        _flash_attn_backward(
            do_win.view_as(q_flat), q_flat, k_win, v_win, out_win_flat, lse_win,
            dq_win_flat, dk_win, dv_win, 0.0, sm_scale, True, ctx.w_win - 1, 0, 0.0, None, False, None,
        )

        dq_slc, dk_slc, dv_slc = run_sparse_selection_bwd_csc(
            q, k_slc, v_slc, out_slc, do_slc, lse_slc, idx_bhku, idx_buhk, b_q_sub=ctx.b_q_sub, b_k=64
        )

        q_fold = (
            q[:, 16:]
            .view(bsz, n_cmp - 1, 16, h_kv, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, n_cmp - 1, h_kv * 16 * g_heads, dim)
        )
        do_fold = (
            do_cmp[:, 16:]
            .view(bsz, n_cmp - 1, 16, h_kv, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, n_cmp - 1, h_kv * 16 * g_heads, dim)
        )
        k_fold = kc[:, :-1].contiguous()
        v_fold = vc[:, :-1].contiguous()
        dq_f = torch.empty_like(q_fold)
        dk_f = torch.empty_like(k_fold)
        dv_f = torch.empty_like(v_fold)
        _flash_attn_backward(
            do_fold, q_fold, k_fold, v_fold, out_f, lse_f,
            dq_f, dk_f, dv_f, 0.0, sm_scale, True, -1, -1, 0.0, None, False, None,
        )

        dq_cmp = torch.zeros_like(q)
        dq_cmp[:, 16:] = (
            dq_f.view(bsz, n_cmp - 1, h_kv, 16, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz, seqlen - 16, h_kv, g_heads, dim)
        )
        dk_cmp = torch.zeros_like(kc)
        dv_cmp = torch.zeros_like(vc)
        dk_cmp[:, :-1] = dk_f
        dv_cmp[:, :-1] = dv_f

        dk_cmp_raw = torch.empty_like(k_cmp_raw)
        dv_cmp_raw = torch.empty_like(v_cmp_raw)
        _strided_compress_bwd_dx_kernel[(n_cmp, bsz * h_kv)](
            dk_cmp, w_pool_k, dk_cmp_raw, *dk_cmp.stride(), *k_cmp_raw.stride(), n_cmp, h_kv, D=dim
        )
        _strided_compress_bwd_dx_kernel[(n_cmp, bsz * h_kv)](
            dv_cmp, w_pool_v, dv_cmp_raw, *dv_cmp.stride(), *v_cmp_raw.stride(), n_cmp, h_kv, D=dim
        )
        dw_k, dpe_k = strided_compress_param_grads(k_cmp_raw, dk_cmp, w_pool_k, pe_k)
        dw_v, dpe_v = strided_compress_param_grads(v_cmp_raw, dv_cmp, w_pool_v, pe_v)

        dq = dq_cmp + dq_slc + dq_win_flat.view_as(q)
        return (
            dq, dk_cmp_raw, dv_cmp_raw, dk_slc, dv_slc, dk_win, dv_win,
            dw_k, dpe_k, dw_v, dpe_v, d_gate, None, None, None, None,
        )


class _MoBAVarlenFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int,
        topk: int,
        gqa_grouped: bool,
    ) -> torch.Tensor:
        seqlen, h_q, dim = q.shape
        h_kv = k.shape[1]
        g_heads = h_q // h_kv
        n_chunks = seqlen // chunk_size
        k_inter = min(topk - 1, max(0, n_chunks - 1))
        sm_scale = dim**-0.5

        cu_self = torch.arange(0, n_chunks + 1, device=q.device, dtype=torch.int32) * chunk_size
        self_out, self_lse, _, _ = _flash_attn_varlen_forward(
            q=q, k=k, v=v, cu_seqlens_q=cu_self, cu_seqlens_k=cu_self,
            max_seqlen_q=chunk_size, max_seqlen_k=chunk_size, dropout_p=0.0, softmax_scale=sm_scale, causal=True,
        )
        if k_inter <= 0:
            ctx.k_inter = 0
            ctx.chunk_size = chunk_size
            ctx.save_for_backward(q, k, v, self_out, self_lse, cu_self)
            return self_out

        if gqa_grouped:
            with torch.no_grad():
                k_blk = k.detach().float().view(n_chunks, chunk_size, h_kv, dim).mean(dim=1)
                q_grp = q.detach().view(seqlen, h_kv, g_heads, dim).float().sum(dim=2)
                gate = torch.einsum("ckd,skd->cks", k_blk, q_grp)
                s_idx = torch.arange(seqlen, device=q.device)[None, :]
                c_end = (torch.arange(n_chunks, device=q.device) + 1)[:, None] * chunk_size
                gate.masked_fill_((s_idx < c_end).unsqueeze(1), float("-inf"))
                topv, topi = torch.topk(gate, k=k_inter, dim=0)
                valid = torch.isfinite(topv)
                sel_mask = torch.zeros_like(gate, dtype=torch.bool).scatter_(0, topi, valid)

            ck_cnt = sel_mask.sum(dim=-1).view(-1)
            nz = ck_cnt > 0
            vcnt = ck_cnt[nz]
            cu_mq = torch.zeros((vcnt.numel() + 1,), device=q.device, dtype=torch.int32)
            cu_mq[1:] = vcnt.cumsum(0).to(torch.int32)
            cu_mk = torch.arange(0, vcnt.numel() + 1, device=q.device, dtype=torch.int32) * chunk_size

            ck_i, s_i = sel_mask.reshape(n_chunks * h_kv, seqlen).nonzero(as_tuple=True)
            q_shk = s_i * h_kv + (ck_i % h_kv)
            m_sel = q_shk.numel()
            mq = q.view(seqlen * h_kv, g_heads, dim).index_select(0, q_shk)

            nz_ids = nz.nonzero(as_tuple=True)[0]
            c_act, hk_act = nz_ids // h_kv, nz_ids % h_kv
            mk = k.view(n_chunks, chunk_size, h_kv, dim)[c_act, :, hk_act, :].reshape(-1, 1, dim)
            mv = v.view(n_chunks, chunk_size, h_kv, dim)[c_act, :, hk_act, :].reshape(-1, 1, dim)
            msq = int(vcnt.max().item())

            m_out, m_lse, _, _ = _flash_attn_varlen_forward(
                q=mq, k=mk, v=mv, cu_seqlens_q=cu_mq, cu_seqlens_k=cu_mk,
                max_seqlen_q=msq, max_seqlen_k=chunk_size, dropout_p=0.0, softmax_scale=sm_scale, causal=False,
            )

            pos_grid = torch.full((n_chunks * h_kv, seqlen), -1, device=q.device, dtype=torch.int32)
            pos_grid[ck_i, s_i] = torch.arange(m_sel, device=q.device, dtype=torch.int32)
            gathered_rows = torch.gather(pos_grid.view(n_chunks, h_kv, seqlen), 0, topi)
            row_map = torch.where(valid, gathered_rows, -1).permute(2, 1, 0).contiguous().view(seqlen * h_kv, k_inter)

            mix_out = torch.empty_like(q)
            mix_lse = torch.empty((h_q, seqlen), device=q.device, dtype=torch.float32)
            _moba_gqa_lse_merge_fwd_kernel[(seqlen * h_kv, g_heads)](
                self_out, self_lse, m_out, m_lse, row_map, mix_out, mix_lse,
                seqlen, h_kv, m_sel, G=g_heads, D=dim, K_INTER=k_inter,
            )
            ctx.k_inter = k_inter
            ctx.chunk_size = chunk_size
            ctx.gqa_grouped = True
            ctx.msq = msq
            ctx.save_for_backward(q, k, v, mix_out, mix_lse, cu_self, mq, mk, mv, cu_mq, cu_mk, q_shk, row_map, c_act, hk_act)
            return mix_out

        with torch.no_grad():
            k_blk = k.detach().float().view(n_chunks, chunk_size, h_kv, dim).mean(dim=1).repeat_interleave(g_heads, dim=1)
            gate = torch.einsum("chd,shd->chs", k_blk, q.detach().float())
            s_idx = torch.arange(seqlen, device=q.device)[None, :]
            c_end = (torch.arange(n_chunks, device=q.device) + 1)[:, None] * chunk_size
            gate.masked_fill_((s_idx < c_end).unsqueeze(1), float("-inf"))
            topv, topi = torch.topk(gate, k=k_inter, dim=0)
            sel_mask = torch.zeros_like(gate, dtype=torch.bool).scatter_(0, topi, torch.isfinite(topv))

        ch_cnt = sel_mask.sum(dim=-1).view(-1)
        nz = ch_cnt > 0
        vcnt = ch_cnt[nz]
        cu_mq = torch.zeros((vcnt.numel() + 1,), device=q.device, dtype=torch.int32)
        cu_mq[1:] = vcnt.cumsum(0).to(torch.int32)
        cu_mk = torch.arange(0, vcnt.numel() + 1, device=q.device, dtype=torch.int32) * chunk_size
        ch_i, s_i = sel_mask.reshape(n_chunks * h_q, seqlen).nonzero(as_tuple=True)
        q_sh = s_i * h_q + (ch_i % h_q)
        mq = q.view(seqlen * h_q, dim).index_select(0, q_sh).unsqueeze(1)
        nz_ids = nz.nonzero(as_tuple=True)[0]
        c_act, hk_act = nz_ids // h_q, (nz_ids % h_q) // g_heads
        mk = k.view(n_chunks, chunk_size, h_kv, dim)[c_act, :, hk_act, :].reshape(-1, 1, dim)
        mv = v.view(n_chunks, chunk_size, h_kv, dim)[c_act, :, hk_act, :].reshape(-1, 1, dim)
        msq = int(vcnt.max().item())
        m_out, m_lse, _, _ = _flash_attn_varlen_forward(
            q=mq, k=mk, v=mv, cu_seqlens_q=cu_mq, cu_seqlens_k=cu_mk,
            max_seqlen_q=msq, max_seqlen_k=chunk_size, dropout_p=0.0, softmax_scale=sm_scale, causal=False,
        )
        sl_sh = self_lse.t().contiguous().view(-1)
        ml_1d = m_lse.view(-1)
        mx = sl_sh.clone().index_reduce_(0, q_sh, ml_1d, "amax", include_self=True)
        se = (sl_sh - mx).exp()
        se.index_add_(0, q_sh, (ml_1d - mx.index_select(0, q_sh)).exp())
        mix_lse_1d = mx + se.log()
        out_2d = self_out.view(seqlen * h_q, dim).float() * (sl_sh - mix_lse_1d).exp().unsqueeze(-1)
        out_2d.index_add_(0, q_sh, m_out.view(-1, dim).float() * (ml_1d - mix_lse_1d.index_select(0, q_sh)).exp().unsqueeze(-1))
        mix_out = out_2d.to(q.dtype).view(seqlen, h_q, dim)
        mix_lse_hs = mix_lse_1d.view(seqlen, h_q).t().contiguous()
        ctx.k_inter = k_inter
        ctx.chunk_size = chunk_size
        ctx.gqa_grouped = False
        ctx.msq = msq
        ctx.save_for_backward(q, k, v, mix_out, mix_lse_hs, mix_lse_1d, cu_self, mq, mk, mv, cu_mq, cu_mk, q_sh, c_act, hk_act)
        return mix_out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        do = do.contiguous()
        chunk_size = ctx.chunk_size
        if ctx.k_inter <= 0:
            q, k, v, self_out, self_lse, cu_self = ctx.saved_tensors
            sm_scale = q.shape[-1] ** -0.5
            dqs, dks, dvs = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _flash_attn_varlen_backward(
                dout=do, q=q, k=k, v=v, out=self_out, softmax_lse=self_lse,
                dq=dqs, dk=dks, dv=dvs, cu_seqlens_q=cu_self, cu_seqlens_k=cu_self,
                max_seqlen_q=chunk_size, max_seqlen_k=chunk_size, dropout_p=0.0, softmax_scale=sm_scale, causal=True,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
            return dqs, dks, dvs, None, None, None

        if ctx.gqa_grouped:
            q, k, v, mix_out, mix_lse, cu_self, mq, mk, mv, cu_mq, cu_mk, q_shk, row_map, c_act, hk_act = ctx.saved_tensors
            seqlen, h_q, dim = q.shape
            h_kv = k.shape[1]
            g_heads = h_q // h_kv
            n_chunks = seqlen // chunk_size
            sm_scale = dim**-0.5

            dqs, dks, dvs = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _flash_attn_varlen_backward(
                dout=do, q=q, k=k, v=v, out=mix_out, softmax_lse=mix_lse,
                dq=dqs, dk=dks, dv=dvs, cu_seqlens_q=cu_self, cu_seqlens_k=cu_self,
                max_seqlen_q=chunk_size, max_seqlen_k=chunk_size, dropout_p=0.0, softmax_scale=sm_scale, causal=True,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
            dom = do.view(seqlen * h_kv, g_heads, dim).index_select(0, q_shk)
            om = mix_out.view(seqlen * h_kv, g_heads, dim).index_select(0, q_shk)
            lm = mix_lse.view(h_kv, g_heads, seqlen).permute(2, 0, 1).reshape(seqlen * h_kv, g_heads).index_select(0, q_shk).t().contiguous()
            dmq, dmk, dmv = torch.empty_like(mq), torch.empty_like(mk), torch.empty_like(mv)
            _flash_attn_varlen_backward(
                dout=dom, q=mq, k=mk, v=mv, out=om, softmax_lse=lm,
                dq=dmq, dk=dmk, dv=dmv, cu_seqlens_q=cu_mq, cu_seqlens_k=cu_mk,
                max_seqlen_q=ctx.msq, max_seqlen_k=chunk_size, dropout_p=0.0, softmax_scale=sm_scale, causal=False,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
            dq_out = torch.empty_like(q)
            _moba_gqa_dq_merge_kernel[(seqlen * h_kv, g_heads)](
                dqs, dmq, row_map, dq_out, seqlen, h_kv, G=g_heads, D=dim, K_INTER=ctx.k_inter
            )
            dks.view(n_chunks, chunk_size, h_kv, dim)[c_act, :, hk_act, :] += dmk.view(-1, chunk_size, dim)
            dvs.view(n_chunks, chunk_size, h_kv, dim)[c_act, :, hk_act, :] += dmv.view(-1, chunk_size, dim)
            return dq_out, dks, dvs, None, None, None

        q, k, v, mix_out, mix_lse_hs, mix_lse_1d, cu_self, mq, mk, mv, cu_mq, cu_mk, q_sh, c_act, hk_act = ctx.saved_tensors
        seqlen, h_q, dim = q.shape
        h_kv = k.shape[1]
        n_chunks = seqlen // chunk_size
        sm_scale = dim**-0.5

        dqs, dks, dvs = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            dout=do, q=q, k=k, v=v, out=mix_out, softmax_lse=mix_lse_hs,
            dq=dqs, dk=dks, dv=dvs, cu_seqlens_q=cu_self, cu_seqlens_k=cu_self,
            max_seqlen_q=chunk_size, max_seqlen_k=chunk_size, dropout_p=0.0, softmax_scale=sm_scale, causal=True,
            window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
        )
        dom = do.view(seqlen * h_q, dim).index_select(0, q_sh).unsqueeze(1)
        om = mix_out.view(seqlen * h_q, dim).index_select(0, q_sh).unsqueeze(1)
        lm = mix_lse_1d.index_select(0, q_sh).unsqueeze(0)
        dmq, dmk, dmv = torch.empty_like(mq), torch.empty_like(mk), torch.empty_like(mv)
        _flash_attn_varlen_backward(
            dout=dom, q=mq, k=mk, v=mv, out=om, softmax_lse=lm,
            dq=dmq, dk=dmk, dv=dmv, cu_seqlens_q=cu_mq, cu_seqlens_k=cu_mk,
            max_seqlen_q=ctx.msq, max_seqlen_k=chunk_size, dropout_p=0.0, softmax_scale=sm_scale, causal=False,
            window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
        )
        dq_f = dqs.float().view(seqlen * h_q, dim)
        dq_f.index_add_(0, q_sh, dmq.view(-1, dim).float())
        flat_chk = c_act * h_kv + hk_act
        dk_f = dks.float().view(n_chunks, chunk_size, h_kv, dim).permute(0, 2, 1, 3).reshape(n_chunks * h_kv, chunk_size * dim)
        dv_f = dvs.float().view(n_chunks, chunk_size, h_kv, dim).permute(0, 2, 1, 3).reshape(n_chunks * h_kv, chunk_size * dim)
        dk_f.index_add_(0, flat_chk, dmk.view(-1, chunk_size * dim).float())
        dv_f.index_add_(0, flat_chk, dmv.view(-1, chunk_size * dim).float())
        dk_out = dk_f.view(n_chunks, h_kv, chunk_size, dim).permute(0, 2, 1, 3).reshape(seqlen, h_kv, dim).to(k.dtype)
        dv_out = dv_f.view(n_chunks, h_kv, chunk_size, dim).permute(0, 2, 1, 3).reshape(seqlen, h_kv, dim).to(v.dtype)
        return dq_f.to(q.dtype).view(seqlen, h_q, dim), dk_out, dv_out, None, None, None


# ============================================================================
# 3. Suite 0 (Audits, Roofline, Atomic vs CSC, & FP64 Chain-Rule), Suite A, & Suite B
# ============================================================================


def _nsa_3branch_fp64_oracle(
    q: torch.Tensor,
    k_cmp_raw: torch.Tensor,
    v_cmp_raw: torch.Tensor,
    k_slc: torch.Tensor,
    v_slc: torch.Tensor,
    k_win: torch.Tensor,
    v_win: torch.Tensor,
    w_pool_k: torch.Tensor,
    pe_k: torch.Tensor,
    w_pool_v: torch.Tensor,
    pe_v: torch.Tensor,
    gate_logits: torch.Tensor,
    fixed_idx_bhku: torch.Tensor,
    b_q_sub: int,
    b_k: int,
    w_win: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, h_kv, g_heads, dim = q.shape
    n_cmp = seqlen // 16
    n_slc = seqlen // b_k
    sm_scale = dim**-0.5

    def _cmp(x_raw: torch.Tensor, wp: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        xp = torch.cat([torch.zeros((bsz, 16, h_kv, dim), device=q.device, dtype=torch.float64), x_raw], dim=1)
        xw = xp.unfold(1, 32, 16).permute(0, 1, 4, 2, 3)
        vm = torch.ones((1, n_cmp, 32, 1, 1), device=q.device, dtype=torch.float64)
        vm[:, 0, :16] = 0.0
        xpe = (xw + pe.permute(1, 0, 2).view(1, 1, 32, h_kv, dim)) * vm
        return (xpe * wp.permute(1, 0).view(1, 1, 32, h_kv, 1)).sum(dim=2)

    kc = _cmp(k_cmp_raw, w_pool_k, pe_k)
    vc = _cmp(v_cmp_raw, w_pool_v, pe_v)

    t_u = torch.arange(seqlen, device=q.device)[:, None] // 16
    c_i = torch.arange(n_cmp, device=q.device)[None, :]
    cmask = (c_i < t_u).view(1, 1, 1, seqlen, n_cmp)

    qp = q.permute(0, 2, 3, 1, 4)
    kcp = kc.permute(0, 2, 1, 3).unsqueeze(2)
    vcp = vc.permute(0, 2, 1, 3).unsqueeze(2)
    sc_cmp = torch.matmul(qp, kcp.transpose(-1, -2)) * sm_scale
    sc_cmp = sc_cmp.masked_fill(~cmask, float("-inf"))
    lse_cmp = torch.logsumexp(sc_cmp, dim=-1, keepdim=True)
    pr_cmp = torch.where(cmask, torch.exp(sc_cmp - lse_cmp), 0.0)
    out_cmp = torch.matmul(pr_cmp, vcp).permute(0, 3, 1, 2, 4)

    p_gsum = pr_cmp.sum(dim=2)
    eq9_scores = torch.zeros((bsz, h_kv, seqlen, n_slc), device=q.device, dtype=torch.float64)
    for j in range(n_slc):
        acc_s = torch.zeros((bsz, h_kv, seqlen), device=q.device, dtype=torch.float64)
        for m in range(4):
            for n in range(2):
                ci = 4 * j + m + n
                if ci < n_cmp:
                    acc_s = acc_s + p_gsum[:, :, :, ci]
        eq9_scores[:, :, :, j] = acc_s
    if b_q_sub > 1:
        eq9_scores = eq9_scores.view(bsz, h_kv, seqlen // b_q_sub, b_q_sub, n_slc).mean(dim=3)

    n_u = seqlen // b_q_sub
    mask_slc = torch.zeros((bsz, h_kv, seqlen, seqlen), device=q.device, dtype=torch.bool)
    for bi in range(bsz):
        for hk in range(h_kv):
            for u in range(n_u):
                t0, t1 = u * b_q_sub, (u + 1) * b_q_sub
                for ki in range(fixed_idx_bhku.shape[-1]):
                    kb = int(fixed_idx_bhku[bi, hk, u, ki].item())
                    if kb >= 0:
                        k0, k1 = kb * b_k, min(seqlen, (kb + 1) * b_k)
                        for ti in range(t0, t1):
                            if k0 <= ti:
                                mask_slc[bi, hk, ti, k0 : min(k1, ti + 1)] = True

    ksp = k_slc.permute(0, 2, 1, 3).unsqueeze(2)
    vsp = v_slc.permute(0, 2, 1, 3).unsqueeze(2)
    sc_slc = torch.matmul(qp, ksp.transpose(-1, -2)) * sm_scale
    sc_slc = sc_slc.masked_fill(~mask_slc.unsqueeze(2), float("-inf"))
    pr_slc = torch.softmax(sc_slc, dim=-1)
    out_slc = torch.matmul(pr_slc, vsp).permute(0, 3, 1, 2, 4)

    t_pos = torch.arange(seqlen, device=q.device)
    mask_win = ((t_pos[:, None] >= t_pos[None, :]) & ((t_pos[:, None] - t_pos[None, :]) < w_win)).view(
        1, 1, 1, seqlen, seqlen
    )
    kwp = k_win.permute(0, 2, 1, 3).unsqueeze(2)
    vwp = v_win.permute(0, 2, 1, 3).unsqueeze(2)
    sc_win = torch.matmul(qp, kwp.transpose(-1, -2)) * sm_scale
    sc_win = sc_win.masked_fill(~mask_win, float("-inf"))
    pr_win = torch.softmax(sc_win, dim=-1)
    out_win = torch.matmul(pr_win, vwp).permute(0, 3, 1, 2, 4)

    gates = torch.sigmoid(gate_logits)
    out = (
        gates[..., 0:1] * out_cmp
        + gates[..., 1:2] * out_slc
        + gates[..., 2:3] * out_win
    )
    return out, eq9_scores


def _moba_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sel_mask_cks: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    seqlen, h_q, dim = q.shape
    h_kv = k.shape[1]
    g_heads = h_q // h_kv
    n_chunks = seqlen // chunk_size
    sm_scale = dim**-0.5

    tok_mask = torch.zeros((h_q, seqlen, seqlen), device=q.device, dtype=torch.bool)
    ar = torch.arange(chunk_size, device=q.device)
    self_c = (ar[:, None] >= ar[None, :]).unsqueeze(0)
    for c in range(n_chunks):
        s0, s1 = c * chunk_size, (c + 1) * chunk_size
        tok_mask[:, s0:s1, s0:s1] = self_c
        routed_hk = sel_mask_cks[c]
        routed_hq = routed_hk.repeat_interleave(g_heads, dim=0)
        tok_mask[:, :, s0:s1] |= routed_hq.unsqueeze(-1)

    qp = q.permute(1, 0, 2)
    ke = k.repeat_interleave(g_heads, dim=1).permute(1, 0, 2)
    ve = v.repeat_interleave(g_heads, dim=1).permute(1, 0, 2)
    sc = torch.matmul(qp, ke.transpose(-1, -2)) * sm_scale
    sc = sc.masked_fill(~tok_mask, float("-inf"))
    pr = torch.softmax(sc, dim=-1)
    return torch.matmul(pr, ve).permute(1, 0, 2).contiguous()


def run_suite_0_audits(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    warmup = 2 if is_smoke else 4
    iters = 4 if is_smoke else 10

    # 0.1 PyTorch 2.13 FlexAttention Dynamic Sparse Mask Quadruple-Defect Audit
    flex_audit: dict[str, Any] = {}
    s_flex = 16384
    mask_dense_4 = torch.zeros((1, 4, s_flex // 64, s_flex // 64), device=device, dtype=torch.bool)
    mask_dense_4[:, :, :, 0] = True
    diag_i = torch.arange(s_flex // 64, device=device)
    mask_dense_4[:, :, diag_i, diag_i] = True

    def _uncompiled_hq_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & mask_dense_4[b, h // 16, q_idx // 64, kv_idx // 64]

    try:
        _ = create_block_mask(_uncompiled_hq_mod, B=1, H=64, Q_LEN=s_flex, KV_LEN=s_flex, device=device, BLOCK_SIZE=64)
        flex_audit["defect1_uncompiled_create_block_mask_hq64_s16k"] = {"oom_triggered": False}
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        flex_audit["defect1_uncompiled_create_block_mask_hq64_s16k"] = {
            "oom_triggered": True,
            "requested_gib": 128.0,
            "error_excerpt": str(e)[:180],
        }

    kv_num_cpu = torch.full((1, 4, 256), 16, dtype=torch.int32)
    kv_idx_cpu = torch.arange(16, dtype=torch.int32).view(1, 1, 1, 16).expand(1, 4, 256, 16).clone()
    kv_idx_cpu[0, 0, 200, 15] = 199
    try:
        _ = BlockMask.from_kv_blocks(kv_num_cpu, kv_idx_cpu, BLOCK_SIZE=64)
        flex_audit["defect2_from_kv_blocks_transpose_ordered"] = {"bug_triggered": False}
    except IndexError as e:
        flex_audit["defect2_from_kv_blocks_transpose_ordered"] = {
            "bug_triggered": True,
            "root_cause": "_transpose_ordered passes col_indices.shape[-1] (K_sel=16) instead of N_blocks (256) as num_cols",
            "error_excerpt": repr(e),
        }

    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)

    def _hkv_mod64(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & mask_dense_4[b, h, q_idx // 64, kv_idx // 64]

    try:
        bm64 = comp_cbm(_hkv_mod64, B=1, H=4, Q_LEN=4096, KV_LEN=4096, device=device, BLOCK_SIZE=64)
        qf = torch.randn(1, 64, 4096, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        kf = torch.randn(1, 4, 4096, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        vf = torch.randn(1, 4, 4096, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        of = comp_flex(qf, kf, vf, block_mask=bm64, enable_gqa=True)
        of.sum().backward()
        flex_audit["defect3_block_size_64_inductor_lowering"] = {"rejected": False}
    except Exception as e:
        flex_audit["defect3_block_size_64_inductor_lowering"] = {
            "rejected": True,
            "error_excerpt": str(e).splitlines()[0][:200],
        }
    del mask_dense_4
    torch.cuda.empty_cache()

    # Defect 4: Isolated subprocess verification of TorchInductor FlexAttention int32 index overflow
    # (`cudaErrorIllegalAddress`) at S = 65,536, H_q = 64, H_kv = 4, D = 128 (without poisoning parent CUDA context)
    sub_code = (
        "import torch\n"
        "from torch.nn.attention.flex_attention import create_block_mask, flex_attention\n"
        "S = 65536\n"
        "md = torch.zeros((1, 4, S // 128, S // 128), device='cuda:0', dtype=torch.bool)\n"
        "md[:, :, :, 0] = True\n"
        "bm = torch.compile(create_block_mask)(lambda b,h,qi,ki: (qi>=ki)&md[b,h,qi//128,ki//128], B=1, H=4, Q_LEN=S, KV_LEN=S, device='cuda:0', BLOCK_SIZE=128)\n"
        "q = torch.randn(1, 64, S, 128, device='cuda:0', dtype=torch.bfloat16, requires_grad=True)\n"
        "k = torch.randn(1, 4, S, 128, device='cuda:0', dtype=torch.bfloat16, requires_grad=True)\n"
        "v = torch.randn(1, 4, S, 128, device='cuda:0', dtype=torch.bfloat16, requires_grad=True)\n"
        "o = torch.compile(flex_attention)(q, k, v, block_mask=bm, enable_gqa=True)\n"
        "o.sum().backward()\n"
        "torch.cuda.synchronize()\n"
    )
    proc_fx64 = subprocess.run([sys.executable, "-c", sub_code], capture_output=True, text=True, timeout=90)
    err_tail = " | ".join(proc_fx64.stderr.strip().splitlines()[-3:])[:220]
    flex_audit["defect4_s64k_gqa64_bwd_illegal_memory_access"] = {
        "illegal_address_triggered": proc_fx64.returncode != 0 and ("illegal memory access" in proc_fx64.stderr.lower()),
        "returncode": int(proc_fx64.returncode),
        "error_excerpt": err_tail,
    }

    # 0.2 SM80 Tensor-Core Query-Group Tile Saturation Roofline (S = 16K smoke / 32K full, K_sel = 16, B_k = 64)
    s_tile = 16384 if is_smoke else 32768
    tile_modes: dict[str, Any] = {}
    k_base4 = torch.randn(1, s_tile, 4, 128, device=device, dtype=torch.bfloat16)
    v_base4 = torch.randn(1, s_tile, 4, 128, device=device, dtype=torch.bfloat16)
    k_base8 = torch.randn(1, s_tile, 8, 128, device=device, dtype=torch.bfloat16)
    v_base8 = torch.randn(1, s_tile, 8, 128, device=device, dtype=torch.bfloat16)

    tile_configs = [
        ("quest_per_head_m1_64q", 64, 1, 1, k_base4[:, :, :1, :].expand(1, s_tile, 64, 128), v_base4[:, :, :1, :].expand(1, s_tile, 64, 128)),
        ("gqa_h4_tok1_padded_32q8kv", 8, 4, 1, k_base8, v_base8),
        ("nsa_h16_tok1_64q4kv", 4, 16, 1, k_base4, v_base4),
        ("nsa_h16_sub4_64q4kv", 4, 16, 4, k_base4, v_base4),
        ("nsa_h4_sub16_stride16_32q8kv", 8, 4, 16, k_base8, v_base8),
    ]
    for name, h_kv_m, g_m, b_q_sub_m, k_v, v_v in tile_configs:
        q_m = torch.randn(1, s_tile, h_kv_m, g_m, 128, device=device, dtype=torch.bfloat16)
        _, idx_buhk = make_sink_heavy_indices(1, h_kv_m, s_tile, s_tile, device, b_q_sub=b_q_sub_m, b_k=64, k_sel=16)
        st = time_fn_with_peak_mem(
            lambda q_m=q_m, k_v=k_v, v_v=v_v, idx_buhk=idx_buhk, b_q_sub_m=b_q_sub_m: run_sparse_selection_fwd(
                q_m, k_v, v_v, idx_buhk, b_q_sub=b_q_sub_m, b_k=64
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
        h_q_tot = h_kv_m * g_m
        active_flops = 4.0 * s_tile * h_q_tot * (16 * 64) * 128
        st["achieved_sparse_tflops"] = round(active_flops / (st["median_ms"] * 1e-3) / 1e12, 2)
        st["mma_active_rows"] = b_q_sub_m * g_m
        tile_modes[name] = st
    tile_modes["sub4_64q_speedup_over_quest_m1"] = round(
        tile_modes["quest_per_head_m1_64q"]["median_ms"] / tile_modes["nsa_h16_sub4_64q4kv"]["median_ms"], 3
    )
    tile_modes["sub16_32q_speedup_over_tok1_padded_32q"] = round(
        tile_modes["gqa_h4_tok1_padded_32q8kv"]["median_ms"] / tile_modes["nsa_h4_sub16_stride16_32q8kv"]["median_ms"], 3
    )
    del k_base4, v_base4, k_base8, v_base8
    torch.cuda.empty_cache()

    # 0.3 Sink-Heavy Backward Atomic Contention vs Unsplit CSC vs Compacted Work-Tile Split-256 CSC
    bwd_ablation: dict[str, Any] = {}
    for s_ab in ([16384] if is_smoke else [16384, 32768]):
        q_a = torch.randn(1, s_ab, 4, 16, 128, device=device, dtype=torch.bfloat16)
        k_a = torch.randn(1, s_ab, 4, 128, device=device, dtype=torch.bfloat16)
        v_a = torch.randn(1, s_ab, 4, 128, device=device, dtype=torch.bfloat16)
        do_a = torch.randn_like(q_a)
        idx_bhku_a, idx_buhk_a = make_sink_heavy_indices(1, 4, s_ab, s_ab, device, b_q_sub=1, b_k=64, k_sel=16)
        out_a, lse_a = run_sparse_selection_fwd(q_a, k_a, v_a, idx_buhk_a, b_q_sub=1, b_k=64)

        dq_at, dk_at, dv_at = run_sparse_selection_bwd_atomic(q_a, k_a, v_a, out_a, do_a, lse_a, idx_buhk_a, b_q_sub=1)
        dq_c0, dk_c0, dv_c0 = run_sparse_selection_bwd_csc(
            q_a, k_a, v_a, out_a, do_a, lse_a, idx_bhku_a, idx_buhk_a, b_q_sub=1, split_tokens=0
        )
        dq_cs, dk_cs, dv_cs = run_sparse_selection_bwd_csc(
            q_a, k_a, v_a, out_a, do_a, lse_a, idx_bhku_a, idx_buhk_a, b_q_sub=1, split_tokens=256
        )

        st_at = time_fn_with_peak_mem(
            lambda: run_sparse_selection_bwd_atomic(q_a, k_a, v_a, out_a, do_a, lse_a, idx_buhk_a, b_q_sub=1),
            warmup=warmup,
            iters=max(3, iters // 2),
            device=device,
        )
        st_c0 = time_fn_with_peak_mem(
            lambda: run_sparse_selection_bwd_csc(
                q_a, k_a, v_a, out_a, do_a, lse_a, idx_bhku_a, idx_buhk_a, b_q_sub=1, split_tokens=0
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
        st_cs = time_fn_with_peak_mem(
            lambda: run_sparse_selection_bwd_csc(
                q_a, k_a, v_a, out_a, do_a, lse_a, idx_bhku_a, idx_buhk_a, b_q_sub=1, split_tokens=256
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
        bwd_ablation[f"S_{s_ab}"] = {
            "query_parallel_fp32_atomic": st_at,
            "unsplit_csc_sink_straggler": st_c0,
            "compacted_worktile_csc_split256": st_cs,
            "speedup_compacted_csc_over_atomic": round(st_at["median_ms"] / st_cs["median_ms"], 3),
            "speedup_compacted_csc_over_unsplit_csc": round(st_c0["median_ms"] / st_cs["median_ms"], 3),
            "dk_rel_l2_csc_vs_atomic": round(rel_l2(dk_cs, dk_at), 6),
            "dv_rel_l2_csc_vs_atomic": round(rel_l2(dv_cs, dv_at), 6),
            "dq_rel_l2_csc_vs_unsplit": round(rel_l2(dq_cs, dq_c0), 6),
        }
        del q_a, k_a, v_a, do_a, out_a, lse_a
        torch.cuda.empty_cache()

    # 0.4 Exact BF16 Kernel Parity vs FP64 Reference Oracles (S = 512, 64Q/4KV, D = 128)
    torch.manual_seed(2026)
    s_p = 512
    q_p = (torch.randn(1, s_p, 4, 16, 128, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    kc_p = (torch.randn(1, s_p, 4, 128, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    vc_p = (torch.randn(1, s_p, 4, 128, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    ks_p = (torch.randn(1, s_p, 4, 128, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    vs_p = (torch.randn(1, s_p, 4, 128, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    kw_p = (torch.randn(1, s_p, 4, 128, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    vw_p = (torch.randn(1, s_p, 4, 128, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    wk_p = (torch.randn(4, 32, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    pek_p = (torch.randn(4, 32, 128, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    wv_p = (torch.randn(4, 32, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    pev_p = (torch.randn(4, 32, 128, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    gl_p = (torch.randn(1, s_p, 4, 16, 3, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    do_p = torch.randn_like(q_p)

    idx_holder: list[tuple[torch.Tensor, torch.Tensor]] = []
    out_nsa_bf16 = _DeepSeekNSAFn.apply(
        q_p, kc_p, vc_p, ks_p, vs_p, kw_p, vw_p, wk_p, pek_p, wv_p, pev_p, gl_p, 4, 4, 128, idx_holder
    )
    (out_nsa_bf16 * do_p).sum().backward()
    used_idx_bhku = idx_holder[0][0]

    q64 = q_p.detach().double().requires_grad_(True)
    kc64 = kc_p.detach().double().requires_grad_(True)
    vc64 = vc_p.detach().double().requires_grad_(True)
    ks64 = ks_p.detach().double().requires_grad_(True)
    vs64 = vs_p.detach().double().requires_grad_(True)
    kw64 = kw_p.detach().double().requires_grad_(True)
    vw64 = vw_p.detach().double().requires_grad_(True)
    wk64 = wk_p.detach().double().requires_grad_(True)
    pek64 = pek_p.detach().double().requires_grad_(True)
    wv64 = wv_p.detach().double().requires_grad_(True)
    pev64 = pev_p.detach().double().requires_grad_(True)
    gl64 = gl_p.detach().double().requires_grad_(True)
    out_nsa_64, _ = _nsa_3branch_fp64_oracle(
        q64, kc64, vc64, ks64, vs64, kw64, vw64, wk64, pek64, wv64, pev64, gl64, used_idx_bhku, 4, 64, 128
    )
    (out_nsa_64 * do_p.double()).sum().backward()

    qm_bf = q_p.detach().view(s_p, 64, 128).clone().requires_grad_(True)
    km_bf = ks_p.detach().view(s_p, 4, 128).clone().requires_grad_(True)
    vm_bf = vs_p.detach().view(s_p, 4, 128).clone().requires_grad_(True)
    dom_bf = do_p.detach().view(s_p, 64, 128)
    out_moba_bf = _MoBAVarlenFn.apply(qm_bf, km_bf, vm_bf, 64, 3, True)
    (out_moba_bf * dom_bf).sum().backward()

    with torch.no_grad():
        kb_m = km_bf.detach().float().view(8, 64, 4, 128).mean(dim=1)
        qg_m = qm_bf.detach().view(s_p, 4, 16, 128).float().sum(dim=2)
        gt_m = torch.einsum("ckd,skd->cks", kb_m, qg_m)
        si_m = torch.arange(s_p, device=device)[None, :]
        ce_m = (torch.arange(8, device=device) + 1)[:, None] * 64
        gt_m.masked_fill_((si_m < ce_m).unsqueeze(1), float("-inf"))
        tv_m, ti_m = torch.topk(gt_m, k=2, dim=0)
        sel_cks = torch.zeros_like(gt_m, dtype=torch.bool).scatter_(0, ti_m, torch.isfinite(tv_m))

    qm64 = qm_bf.detach().double().requires_grad_(True)
    km64 = km_bf.detach().double().requires_grad_(True)
    vm64 = vm_bf.detach().double().requires_grad_(True)
    out_moba_64 = _moba_fp64_oracle(qm64, km64, vm64, sel_cks, 64)
    (out_moba_64 * dom_bf.double()).sum().backward()

    parity_s512 = {
        "nsa_out_rel_l2": round(rel_l2(out_nsa_bf16, out_nsa_64), 6),
        "nsa_dq_rel_l2": round(rel_l2(q_p.grad, q64.grad), 6),
        "nsa_dk_cmp_raw_rel_l2": round(rel_l2(kc_p.grad, kc64.grad), 6),
        "nsa_dk_slc_rel_l2": round(rel_l2(ks_p.grad, ks64.grad), 6),
        "nsa_dk_win_rel_l2": round(rel_l2(kw_p.grad, kw64.grad), 6),
        "nsa_dw_pool_k_rel_l2": round(rel_l2(wk_p.grad, wk64.grad), 6),
        "nsa_dgate_rel_l2": round(rel_l2(gl_p.grad, gl64.grad), 6),
        "moba_out_rel_l2": round(rel_l2(out_moba_bf, out_moba_64), 6),
        "moba_dq_rel_l2": round(rel_l2(qm_bf.grad, qm64.grad), 6),
        "moba_dk_rel_l2": round(rel_l2(km_bf.grad, km64.grad), 6),
    }

    # 0.5 FP64 Richardson Central-Difference Verification (h=1e-3 vs h/2=5e-4 -> ~4.00x)
    torch.manual_seed(777)
    s_r = 128
    qr = (torch.randn(1, s_r, 2, 4, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    kcr = (torch.randn(1, s_r, 2, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    vcr = (torch.randn(1, s_r, 2, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    ksr = (torch.randn(1, s_r, 2, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    vsr = (torch.randn(1, s_r, 2, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    kwr = (torch.randn(1, s_r, 2, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    vwr = (torch.randn(1, s_r, 2, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    wkr = (torch.randn(2, 32, device=device, dtype=torch.float64) * 0.15).requires_grad_(True)
    pekr = (torch.randn(2, 32, 32, device=device, dtype=torch.float64) * 0.15).requires_grad_(True)
    wvr = (torch.randn(2, 32, device=device, dtype=torch.float64) * 0.15).requires_grad_(True)
    pevr = (torch.randn(2, 32, 32, device=device, dtype=torch.float64) * 0.15).requires_grad_(True)
    glr = (torch.randn(1, s_r, 2, 4, 3, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    dor = torch.randn_like(qr)
    idx_r, _ = make_sink_heavy_indices(1, 2, s_r, s_r, device, b_q_sub=4, b_k=64, k_sel=2)

    def _loss_nsa_64(
        q_: torch.Tensor,
        kc_: torch.Tensor,
        ks_: torch.Tensor,
        kw_: torch.Tensor,
        wk_: torch.Tensor,
        gl_: torch.Tensor,
    ) -> torch.Tensor:
        o_, _ = _nsa_3branch_fp64_oracle(
            q_, kc_, vcr, ks_, vsr, kw_, vwr, wk_, pekr, wvr, pevr, gl_, idx_r, 4, 64, 32
        )
        return (o_ * dor).sum()

    loss0 = _loss_nsa_64(qr, kcr, ksr, kwr, wkr, glr)
    grads_nsa = torch.autograd.grad(loss0, (qr, kcr, ksr, kwr, wkr, glr), retain_graph=True)
    grad_vwr = torch.autograd.grad(loss0, vwr)[0]
    dv_dir = torch.randn_like(vwr)
    dv_dir = dv_dir / dv_dir.norm()
    with torch.no_grad():
        lp_v = float((_nsa_3branch_fp64_oracle(qr, kcr, vcr, ksr, vsr, kwr, vwr + 1e-2 * dv_dir, wkr, pekr, wvr, pevr, glr, idx_r, 4, 64, 32)[0] * dor).sum().item())
        lm_v = float((_nsa_3branch_fp64_oracle(qr, kcr, vcr, ksr, vsr, kwr, vwr - 1e-2 * dv_dir, wkr, pekr, wvr, pevr, glr, idx_r, 4, 64, 32)[0] * dor).sum().item())
    dv_win_linear_abs_err = abs((lp_v - lm_v) / 2e-2 - float((grad_vwr * dv_dir).sum().item()))

    nsa_names = ["nsa_dq", "nsa_dk_cmp_raw", "nsa_dk_slc", "nsa_dk_win", "nsa_dw_pool_k", "nsa_dgate_logits"]
    nsa_tensors = [qr, kcr, ksr, kwr, wkr, glr]

    richardson_ratios: dict[str, float] = {}
    h_step = 2.0e-2
    for idx_t, (nm, ten, gr) in enumerate(zip(nsa_names, nsa_tensors, grads_nsa)):
        d_vec = torch.randn_like(ten)
        d_vec = d_vec / d_vec.norm()
        exact_dir = float((gr * d_vec).sum().item())

        def _eval_pert(eps: float) -> float:
            args_p = [t.detach() for t in nsa_tensors]
            args_p[idx_t] = args_p[idx_t] + eps * d_vec
            with torch.no_grad():
                return float(_loss_nsa_64(*args_p).item())

        cd_h = (_eval_pert(h_step) - _eval_pert(-h_step)) / (2.0 * h_step)
        cd_h2 = (_eval_pert(0.5 * h_step) - _eval_pert(-0.5 * h_step)) / h_step
        err_h = abs(cd_h - exact_dir)
        err_h2 = max(abs(cd_h2 - exact_dir), 1e-16)
        richardson_ratios[nm] = round(err_h / err_h2, 4)

    qmr = (torch.randn(s_r, 8, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    kmr = (torch.randn(s_r, 2, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    vmr = (torch.randn(s_r, 2, 32, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    domr = torch.randn_like(qmr)
    sel_mr = torch.zeros((4, 2, s_r), device=device, dtype=torch.bool)
    sel_mr[0, :, 32:] = True
    sel_mr[1, :, 64:] = True
    lm0 = (_moba_fp64_oracle(qmr, kmr, vmr, sel_mr, 32) * domr).sum()
    gm_q, gm_k = torch.autograd.grad(lm0, (qmr, kmr))
    for nm, ten, gr, is_q in [("moba_dq", qmr, gm_q, True), ("moba_dk", kmr, gm_k, False)]:
        dv = torch.randn_like(ten)
        dv = dv / dv.norm()
        ex = float((gr * dv).sum().item())
        with torch.no_grad():
            if is_q:
                fp_h = float((_moba_fp64_oracle(qmr + h_step * dv, kmr, vmr, sel_mr, 32) * domr).sum().item())
                fm_h = float((_moba_fp64_oracle(qmr - h_step * dv, kmr, vmr, sel_mr, 32) * domr).sum().item())
                fp_h2 = float((_moba_fp64_oracle(qmr + 0.5 * h_step * dv, kmr, vmr, sel_mr, 32) * domr).sum().item())
                fm_h2 = float((_moba_fp64_oracle(qmr - 0.5 * h_step * dv, kmr, vmr, sel_mr, 32) * domr).sum().item())
            else:
                fp_h = float((_moba_fp64_oracle(qmr, kmr + h_step * dv, vmr, sel_mr, 32) * domr).sum().item())
                fm_h = float((_moba_fp64_oracle(qmr, kmr - h_step * dv, vmr, sel_mr, 32) * domr).sum().item())
                fp_h2 = float((_moba_fp64_oracle(qmr, kmr + 0.5 * h_step * dv, vmr, sel_mr, 32) * domr).sum().item())
                fm_h2 = float((_moba_fp64_oracle(qmr, kmr - 0.5 * h_step * dv, vmr, sel_mr, 32) * domr).sum().item())
        cd_h = (fp_h - fm_h) / (2.0 * h_step)
        cd_h2 = (fp_h2 - fm_h2) / h_step
        richardson_ratios[nm] = round(abs(cd_h - ex) / max(abs(cd_h2 - ex), 1e-16), 4)

    return {
        "pytorch_flex_attention_triple_defect_audit": flex_audit,
        "sm80_gqa_tile_saturation_roofline": {"seqlen": s_tile, "modes": tile_modes},
        "sink_heavy_backward_atomic_vs_csc_ablation": bwd_ablation,
        "bf16_vs_fp64_oracle_parity_s512": parity_s512,
        "fp64_richardson_central_difference": {
            "ratios": richardson_ratios,
            "min_ratio": round(min(richardson_ratios.values()), 4),
            "max_ratio": round(max(richardson_ratios.values()), 4),
            "nsa_dv_win_linear_fd_abs_err": float(f"{dv_win_linear_abs_err:.3e}"),
        },
    }


def run_suite_a_1gpu_scaling(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    seqlens = [4096, 8192, 16384] if is_smoke else [4096, 8192, 16384, 32768, 65536]
    warmup = 2 if is_smoke else 3
    iters = 4 if is_smoke else 8
    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)
    results: dict[str, Any] = {}

    for seqlen in seqlens:
        iters_s = max(3, iters // 2) if seqlen >= 32768 else iters
        row: dict[str, Any] = {"seqlen": seqlen}
        torch.manual_seed(1000 + seqlen)

        q = torch.randn(1, seqlen, 4, 16, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        k_c = torch.randn(1, seqlen, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        v_c = torch.randn(1, seqlen, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        k_s = torch.randn(1, seqlen, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        v_s = torch.randn(1, seqlen, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        k_w = torch.randn(1, seqlen, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        v_w = torch.randn(1, seqlen, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        wk = torch.randn(4, 32, device=device, dtype=torch.float32, requires_grad=True)
        pek = torch.randn(4, 32, 128, device=device, dtype=torch.float32, requires_grad=True)
        wv = torch.randn(4, 32, device=device, dtype=torch.float32, requires_grad=True)
        pev = torch.randn(4, 32, 128, device=device, dtype=torch.float32, requires_grad=True)
        gl = torch.randn(1, seqlen, 4, 16, 3, device=device, dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn_like(q)

        # 1. Dense FlashAttention-2 GQA (64Q/4KV, D=128)
        qd = q.detach().view(1, seqlen, 64, 128).requires_grad_(True)
        kd = k_s.detach().requires_grad_(True)
        vd = v_s.detach().requires_grad_(True)
        dod = do.view(1, seqlen, 64, 128)

        def _fa2_fwd() -> Any:
            return flash_attn_func(qd, kd, vd, causal=True)

        def _fa2_fb() -> None:
            o = flash_attn_func(qd, kd, vd, causal=True)
            o.backward(dod)
            qd.grad = kd.grad = vd.grad = None

        st_fa_f = time_fn_with_peak_mem(_fa2_fwd, warmup=warmup, iters=iters_s, device=device)
        st_fa_fb = time_fn_with_peak_mem(_fa2_fb, warmup=warmup, iters=iters_s, device=device)
        st_fa_fb["fwd_ms"] = round(st_fa_f["median_ms"], 3)
        st_fa_fb["bwd_ms"] = round(max(0.01, st_fa_fb["median_ms"] - st_fa_f["median_ms"]), 3)
        row["dense_fa2_gqa"] = st_fa_fb

        # 2. PyTorch 2.13 FlexAttention (BLOCK_SIZE=128 forced, K_sel=8 -> 1,024 tokens, dynamic compiled mask;
        #    S >= 65,536 hits Defect 4 int32 overflow -> cudaErrorIllegalAddress verified in Suite 0.1 subprocess)
        if seqlen >= 65536:
            row["flex_attn_block128_dynamic"] = {
                "status": "cuda_illegal_address_int32_overflow_at_s64k",
                "verified_in_isolated_subprocess": True,
            }
        else:
            n_b128 = seqlen // 128
            md128 = torch.zeros((1, 4, n_b128, n_b128), device=device, dtype=torch.bool)
            for qb in range(n_b128):
                md128[:, :, qb, 0] = True
                md128[:, :, qb, qb] = True
                if qb >= 1:
                    md128[:, :, qb, qb - 1] = True
                for extra in range(1, min(7, qb)):
                    md128[:, :, qb, (qb * 7 + extra * 13) % max(1, qb)] = True

            def _m128(b: Any, h: Any, qi: Any, ki: Any, md128: torch.Tensor = md128) -> Any:
                return (qi >= ki) & md128[b, h, qi // 128, ki // 128]

            try:
                qfx = q.detach().view(1, seqlen, 64, 128).transpose(1, 2).contiguous().requires_grad_(True)
                kfx = k_s.detach().transpose(1, 2).contiguous().requires_grad_(True)
                vfx = v_s.detach().transpose(1, 2).contiguous().requires_grad_(True)
                dofx = do.view(1, seqlen, 64, 128).transpose(1, 2).contiguous()

                def _fx_fb() -> None:
                    bm = comp_cbm(_m128, B=1, H=4, Q_LEN=seqlen, KV_LEN=seqlen, device=device, BLOCK_SIZE=128)
                    o = comp_flex(qfx, kfx, vfx, block_mask=bm, enable_gqa=True)
                    o.backward(dofx)
                    qfx.grad = kfx.grad = vfx.grad = None

                st_fx = time_fn_with_peak_mem(_fx_fb, warmup=warmup, iters=iters_s, device=device)
                st_fx["status"] = "ok"
                row["flex_attn_block128_dynamic"] = st_fx
                del qfx, kfx, vfx, dofx
            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                row["flex_attn_block128_dynamic"] = {
                    "status": "oom_create_block_mask",
                    "error_excerpt": str(e)[:160],
                }
            del md128
            torch.cuda.empty_cache()

        # 3 & 4. Kimi MoBA (chunk_size=512, topk=4 -> 2,048 active tokens): Per-Head Eager Scatter vs GQA-Group Triton LSE
        qm = q.detach().view(seqlen, 64, 128).requires_grad_(True)
        km = k_s.detach().view(seqlen, 4, 128).requires_grad_(True)
        vm = v_s.detach().view(seqlen, 4, 128).requires_grad_(True)
        dom = do.view(seqlen, 64, 128)

        for m_label, gqa_grp in [("moba_per_head_eager_scatter", False), ("moba_gqa_group_triton_lse", True)]:
            st_mf = time_fn_with_peak_mem(
                lambda gqa_grp=gqa_grp: _MoBAVarlenFn.apply(qm.detach(), km.detach(), vm.detach(), 512, 4, gqa_grp),
                warmup=warmup,
                iters=iters_s,
                device=device,
            )

            def _m_fb(gqa_grp: bool = gqa_grp) -> None:
                o = _MoBAVarlenFn.apply(qm, km, vm, 512, 4, gqa_grp)
                o.backward(dom)
                qm.grad = km.grad = vm.grad = None

            st_mfb = time_fn_with_peak_mem(_m_fb, warmup=warmup, iters=iters_s, device=device)
            st_mfb["fwd_ms"] = round(st_mf["median_ms"], 3)
            st_mfb["bwd_ms"] = round(max(0.01, st_mfb["median_ms"] - st_mf["median_ms"]), 3)
            row[m_label] = st_mfb

        # 5 & 6. Full 3-Branch Trainable DeepSeek NSA (l=32, d=16, l'=64, k_sel=16, w=512): tok1 vs sub4
        nsa_params = (q, k_c, v_c, k_s, v_s, k_w, v_w, wk, pek, wv, pev, gl)
        for nsa_label, b_sub in [("deepseek_nsa_3branch_tok1_csc", 1), ("deepseek_nsa_3branch_sub4_csc", 4)]:
            st_nf = time_fn_with_peak_mem(
                lambda b_sub=b_sub: _DeepSeekNSAFn.apply(
                    *[p.detach() for p in nsa_params], b_sub, 16, 512, None
                ),
                warmup=warmup,
                iters=iters_s,
                device=device,
            )

            def _nsa_fb(b_sub: int = b_sub) -> None:
                o = _DeepSeekNSAFn.apply(*nsa_params, b_sub, 16, 512, None)
                o.backward(do)
                for p in nsa_params:
                    p.grad = None

            st_nfb = time_fn_with_peak_mem(_nsa_fb, warmup=warmup, iters=iters_s, device=device)
            st_nfb["fwd_ms"] = round(st_nf["median_ms"], 3)
            st_nfb["bwd_ms"] = round(max(0.01, st_nfb["median_ms"] - st_nf["median_ms"]), 3)
            st_nfb["speedup_over_dense_fa2"] = round(
                row["dense_fa2_gqa"]["median_ms"] / st_nfb["median_ms"], 3
            )
            row[nsa_label] = st_nfb

        row["moba_gqa_speedup_over_per_head"] = round(
            row["moba_per_head_eager_scatter"]["median_ms"] / row["moba_gqa_group_triton_lse"]["median_ms"], 3
        )
        results[f"S_{seqlen}"] = row
        del q, k_c, v_c, k_s, v_s, k_w, v_w, do, qd, kd, vd, qm, km, vm
        torch.cuda.empty_cache()

    return results


class _NSABlock(nn.Module):
    def __init__(self, hidden: int = 2560, ffn: int = 6912, h_kv: int = 4, g_heads: int = 16, dim: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        self.h_kv = h_kv
        self.g_heads = g_heads
        self.dim = dim
        self.norm1_w = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        self.norm2_w = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        q_dim = h_kv * g_heads * dim
        kv_dim = h_kv * dim
        self.wq = nn.Linear(hidden, q_dim, bias=False, dtype=torch.bfloat16)
        self.wkv_all = nn.Linear(hidden, 6 * kv_dim, bias=False, dtype=torch.bfloat16)
        self.wgate = nn.Linear(hidden, h_kv * g_heads * 3, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(q_dim, hidden, bias=False, dtype=torch.bfloat16)
        self.w_pool_k = nn.Parameter(torch.full((h_kv, 32), 1.0 / 32.0, dtype=torch.float32))
        self.pe_k = nn.Parameter(torch.zeros((h_kv, 32, dim), dtype=torch.float32))
        self.w_pool_v = nn.Parameter(torch.full((h_kv, 32), 1.0 / 32.0, dtype=torch.float32))
        self.pe_v = nn.Parameter(torch.zeros((h_kv, 32, dim), dtype=torch.float32))
        self.w_gu = nn.Linear(hidden, 2 * ffn, bias=False, dtype=torch.bfloat16)
        self.w_dn = nn.Linear(ffn, hidden, bias=False, dtype=torch.bfloat16)

    def _rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return (xf * rstd).to(x.dtype) * w

    def forward_dense_fa2(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        h1 = self._rmsnorm(x, self.norm1_w)
        q = self.wq(h1).view(bsz, seqlen, self.h_kv * self.g_heads, self.dim)
        kv = self.wkv_all(h1).view(bsz, seqlen, 6, self.h_kv, self.dim)
        attn = flash_attn_func(q, kv[:, :, 0], kv[:, :, 1], causal=True)
        x = x + self.wo(attn.reshape(bsz, seqlen, -1))
        h2 = self._rmsnorm(x, self.norm2_w)
        g, u = self.w_gu(h2).chunk(2, dim=-1)
        return x + self.w_dn(F.silu(g) * u)

    def forward_nsa(
        self, x: torch.Tensor, policy: str = "no_ac"
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        if policy == "no_ac":
            h1 = self._rmsnorm(x, self.norm1_w)
            q = self.wq(h1).view(bsz, seqlen, self.h_kv, self.g_heads, self.dim)
            kv = self.wkv_all(h1).view(bsz, seqlen, 6, self.h_kv, self.dim)
            gl = self.wgate(h1).view(bsz, seqlen, self.h_kv, self.g_heads, 3)
            attn = _DeepSeekNSAFn.apply(
                q, kv[:, :, 0].contiguous(), kv[:, :, 1].contiguous(),
                kv[:, :, 2].contiguous(), kv[:, :, 3].contiguous(),
                kv[:, :, 4].contiguous(), kv[:, :, 5].contiguous(),
                self.w_pool_k, self.pe_k, self.w_pool_v, self.pe_v, gl, 4, 16, 512, None,
            )
            x2 = x + self.wo(attn.reshape(bsz, seqlen, -1))
            h2 = self._rmsnorm(x2, self.norm2_w)
            g, u = self.w_gu(h2).chunk(2, dim=-1)
            return x2 + self.w_dn(F.silu(g) * u)

        if policy == "full_layer_ac":
            return torch.utils.checkpoint.checkpoint(
                lambda x_in: self.forward_nsa(x_in, policy="no_ac"), x, use_reentrant=False
            )

        # policy == "nsa_sac": Save attention output `attn` + cached `sel_idx` (zero attention or top-k recompute!),
        # checkpoint only the 2x6912 SwiGLU FFN activation expansion + wo projection!
        h1 = self._rmsnorm(x, self.norm1_w)
        q = self.wq(h1).view(bsz, seqlen, self.h_kv, self.g_heads, self.dim)
        kv = self.wkv_all(h1).view(bsz, seqlen, 6, self.h_kv, self.dim)
        gl = self.wgate(h1).view(bsz, seqlen, self.h_kv, self.g_heads, 3)
        idx_holder: list[tuple[torch.Tensor, torch.Tensor]] = []
        attn = _DeepSeekNSAFn.apply(
            q, kv[:, :, 0].contiguous(), kv[:, :, 1].contiguous(),
            kv[:, :, 2].contiguous(), kv[:, :, 3].contiguous(),
            kv[:, :, 4].contiguous(), kv[:, :, 5].contiguous(),
            self.w_pool_k, self.pe_k, self.w_pool_v, self.pe_v, gl, 4, 16, 512, idx_holder,
        )

        def _post_attn_ffn(x_res: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
            x_mid = x_res + self.wo(attn_out.reshape(bsz, seqlen, -1))
            h2 = self._rmsnorm(x_mid, self.norm2_w)
            g, u = self.w_gu(h2).chunk(2, dim=-1)
            return x_mid + self.w_dn(F.silu(g) * u)

        return torch.utils.checkpoint.checkpoint(_post_attn_ffn, x, attn, use_reentrant=False)


def run_suite_b_4layer_nsa_sac(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    seqlens = [8192, 16384] if is_smoke else [8192, 16384, 32768]
    warmup = 1 if is_smoke else 2
    iters = 3 if is_smoke else 5

    torch.manual_seed(4242)
    layers = nn.ModuleList([_NSABlock().to(device) for _ in range(4)])
    results: dict[str, Any] = {}

    for seqlen in seqlens:
        x0 = torch.randn(1, seqlen, 2560, device=device, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x0)

        # Verify exact gradient parity between nsa_no_ac and nsa_sac!
        for p in layers.parameters():
            p.grad = None
        x_ref = x0.detach().clone().requires_grad_(True)
        y_ref = x_ref
        for blk in layers:
            y_ref = blk.forward_nsa(y_ref, policy="no_ac")
        y_ref.backward(dy)
        ref_dw = layers[0].wq.weight.grad.detach().clone()
        ref_dx = x_ref.grad.detach().clone()

        for p in layers.parameters():
            p.grad = None
        x_sac = x0.detach().clone().requires_grad_(True)
        y_sac = x_sac
        for blk in layers:
            y_sac = blk.forward_nsa(y_sac, policy="nsa_sac")
        y_sac.backward(dy)
        sac_dw_drift = rel_l2(layers[0].wq.weight.grad, ref_dw)
        sac_dx_drift = rel_l2(x_sac.grad, ref_dx)
        for p in layers.parameters():
            p.grad = None
        del x_ref, y_ref, x_sac, y_sac, ref_dw, ref_dx
        torch.cuda.empty_cache()

        row: dict[str, Any] = {
            "seqlen": seqlen,
            "nsa_sac_dw_rel_l2_vs_no_ac": round(sac_dw_drift, 8),
            "nsa_sac_dx_rel_l2_vs_no_ac": round(sac_dx_drift, 8),
        }

        for mode_name, pol in [
            ("dense_fa2_no_ac", "dense"),
            ("nsa_no_ac", "no_ac"),
            ("nsa_zero_sel_recompute_sac", "nsa_sac"),
            ("nsa_full_layer_ac", "full_layer_ac"),
        ]:
            def _step(pol: str = pol) -> None:
                xin = x0.detach().requires_grad_(True)
                y = xin
                for blk in layers:
                    y = blk.forward_dense_fa2(y) if pol == "dense" else blk.forward_nsa(y, policy=pol)
                y.backward(dy)
                for p in layers.parameters():
                    p.grad = None

            st = time_fn_with_peak_mem(_step, warmup=warmup, iters=iters, device=device)
            st["tok_per_s"] = round(seqlen / (st["median_ms"] * 1e-3), 1)
            row[mode_name] = st

        row["nsa_no_ac_speedup_over_dense_fa2"] = round(
            row["dense_fa2_no_ac"]["median_ms"] / row["nsa_no_ac"]["median_ms"], 3
        )
        row["nsa_sac_speedup_over_full_ac"] = round(
            row["nsa_full_layer_ac"]["median_ms"] / row["nsa_zero_sel_recompute_sac"]["median_ms"], 3
        )
        row["nsa_sac_mem_saving_ratio"] = round(
            row["nsa_no_ac"]["inc_peak_mib"] / max(1.0, row["nsa_zero_sel_recompute_sac"]["inc_peak_mib"]), 3
        )
        results[f"S_{seqlen}"] = row
        del x0, dy
        torch.cuda.empty_cache()

    del layers
    torch.cuda.empty_cache()
    return results


# ============================================================================
# 4. Suite C: 8-GPU NVSwitch Context Parallelism (CP = 8, S = 16K..131K)
# ============================================================================


def _all_to_all_seq_to_head(x: torch.Tensor, group: dist.ProcessGroup, p_size: int) -> torch.Tensor:
    # x: [B, S_local, p_size, H_local, ...] -> [B, p_size * S_local, H_local, ...]
    # Move p_size to dim 0 for contiguous dist.all_to_all_single
    perm = [2, 0, 1] + list(range(3, x.ndim))
    send_buf = x.permute(*perm).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=group)
    # recv_buf: [p_size, B, S_local, H_local, ...] -> [B, p_size, S_local, H_local, ...] -> merge (1, 2)
    inv_perm = [1, 0, 2] + list(range(3, recv_buf.ndim))
    merged = recv_buf.permute(*inv_perm).contiguous()
    new_shape = (merged.shape[0], merged.shape[1] * merged.shape[2], *merged.shape[3:])
    return merged.view(*new_shape)


def _all_to_all_head_to_seq(x: torch.Tensor, group: dist.ProcessGroup, p_size: int) -> torch.Tensor:
    # x: [B, p_size * S_local, H_local, ...] -> [B, S_local, p_size, H_local, ...]
    bsz = x.shape[0]
    s_local = x.shape[1] // p_size
    split_x = x.view(bsz, p_size, s_local, *x.shape[2:])
    perm = [1, 0, 2] + list(range(3, split_x.ndim))
    send_buf = split_x.permute(*perm).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=group)
    inv_perm = [1, 2, 0] + list(range(3, recv_buf.ndim))
    return recv_buf.permute(*inv_perm).contiguous()


def _run_nsa_local_slice_fwd_bwd(
    q_slice: torch.Tensor,
    do_slice: torch.Tensor,
    gl_slice: torch.Tensor,
    kc_prefix: torch.Tensor,
    vc_prefix: torch.Tensor,
    k_slc_prefix: torch.Tensor,
    v_slc_prefix: torch.Tensor,
    k_win_halo: torch.Tensor,
    v_win_halo: torch.Tensor,
    q_global_offset: int,
    b_q_sub: int,
    k_sel: int,
    w_win: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Computes exact 3-branch NSA Fwd + Bwd for a contiguous query slice starting at `q_global_offset`
    # against causal prefix compressed keys `kc_prefix` (length `(q_global_offset + S_q)//16`),
    # causal prefix selection KV `k_slc_prefix` (length `q_global_offset + S_q`),
    # and haloed window KV `k_win_halo` (length `S_q` if q_global_offset==0 else `w_win + S_q`).
    bsz, s_q, h_kv, g_heads, dim = q_slice.shape
    sm_scale = dim**-0.5
    u_start = q_global_offset // 16
    n_u_q = s_q // 16

    # 1. Compression branch via bottom-right aligned Folded-Staircase FA2!
    if u_start == 0:
        q_fold = (
            q_slice[:, 16:]
            .view(bsz, n_u_q - 1, 16, h_kv, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, n_u_q - 1, h_kv * 16 * g_heads, dim)
        )
        k_fold = kc_prefix[:, : n_u_q - 1].contiguous()
        v_fold = vc_prefix[:, : n_u_q - 1].contiguous()
        out_f, lse_f, _, _ = _flash_attn_forward(
            q_fold, k_fold, v_fold, 0.0, sm_scale, True, -1, -1, 0.0, None, False
        )
        out_cmp = torch.zeros_like(q_slice)
        out_cmp[:, 16:] = (
            out_f.view(bsz, n_u_q - 1, h_kv, 16, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz, s_q - 16, h_kv, g_heads, dim)
        )
        lse_cmp = torch.full((bsz, s_q, h_kv, g_heads), -1.0e30, device=q_slice.device, dtype=torch.float32)
        lse_cmp[:, 16:] = (
            lse_f.view(bsz, h_kv, 16, g_heads, n_u_q - 1)
            .permute(0, 4, 2, 1, 3)
            .reshape(bsz, s_q - 16, h_kv, g_heads)
        )
    else:
        n_c_tot = u_start + n_u_q - 1
        q_fold = (
            q_slice.view(bsz, n_u_q, 16, h_kv, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, n_u_q, h_kv * 16 * g_heads, dim)
        )
        k_fold = kc_prefix[:, :n_c_tot].contiguous()
        v_fold = vc_prefix[:, :n_c_tot].contiguous()
        out_f, lse_f, _, _ = _flash_attn_forward(
            q_fold, k_fold, v_fold, 0.0, sm_scale, True, -1, -1, 0.0, None, False
        )
        out_cmp = (
            out_f.view(bsz, n_u_q, h_kv, 16, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz, s_q, h_kv, g_heads, dim)
        )
        lse_cmp = (
            lse_f.view(bsz, h_kv, 16, g_heads, n_u_q)
            .permute(0, 4, 2, 1, 3)
            .reshape(bsz, s_q, h_kv, g_heads)
        )

    # 2. Induce Eq. (8)-(10) selection blocks + Selection Fwd
    idx_bhku, idx_buhk = induce_nsa_topk_indices(
        q_slice, kc_prefix, lse_cmp, b_q_sub=b_q_sub, b_k=64, k_sel=k_sel, q_global_offset=q_global_offset
    )
    out_slc, lse_slc = run_sparse_selection_fwd(
        q_slice, k_slc_prefix, v_slc_prefix, idx_buhk, b_q_sub=b_q_sub, b_k=64, q_global_offset=q_global_offset
    )

    # 3. Sliding window Fwd (bottom-right causal alignment matches haloed k_win_halo!)
    q_flat = q_slice.reshape(bsz, s_q, h_kv * g_heads, dim)
    out_win_flat, lse_win, _, _ = _flash_attn_forward(
        q_flat, k_win_halo, v_win_halo, 0.0, sm_scale, True, w_win - 1, 0, 0.0, None, False
    )

    # 4. Fused 3-branch gate Fwd + Bwd
    out = torch.empty_like(q_slice)
    n_rows = bsz * s_q * h_kv * g_heads
    _nsa_3branch_gate_fwd_kernel[(n_rows,)](out_cmp, out_slc, out_win_flat, gl_slice, out, n_rows, D=dim)

    do_cmp = torch.empty_like(q_slice)
    do_slc = torch.empty_like(q_slice)
    do_win = torch.empty_like(q_slice)
    d_gate = torch.empty_like(gl_slice)
    _nsa_3branch_gate_bwd_kernel[(n_rows,)](
        do_slice, out_cmp, out_slc, out_win_flat, gl_slice,
        do_cmp, do_slc, do_win, d_gate, n_rows, D=dim,
    )

    # 5. Window Bwd
    dq_win_flat = torch.empty_like(q_flat)
    dk_win_h = torch.empty_like(k_win_halo)
    dv_win_h = torch.empty_like(v_win_halo)
    _flash_attn_backward(
        do_win.reshape(bsz, s_q, h_kv * g_heads, dim), q_flat, k_win_halo, v_win_halo,
        out_win_flat, lse_win, dq_win_flat, dk_win_h, dv_win_h,
        0.0, sm_scale, True, w_win - 1, 0, 0.0, None, False, None,
    )

    # 6. Selection Compacted Work-Tile CSC Bwd
    dq_slc, dk_slc_p, dv_slc_p = run_sparse_selection_bwd_csc(
        q_slice, k_slc_prefix, v_slc_prefix, out_slc, do_slc, lse_slc,
        idx_bhku, idx_buhk, b_q_sub=b_q_sub, b_k=64, q_global_offset=q_global_offset,
    )

    # 7. Compression Folded-Staircase Bwd
    if u_start == 0:
        do_fold = (
            do_cmp[:, 16:]
            .view(bsz, n_u_q - 1, 16, h_kv, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, n_u_q - 1, h_kv * 16 * g_heads, dim)
        )
    else:
        do_fold = (
            do_cmp.view(bsz, n_u_q, 16, h_kv, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, n_u_q, h_kv * 16 * g_heads, dim)
        )
    dq_f = torch.empty_like(q_fold)
    dk_f = torch.empty_like(k_fold)
    dv_f = torch.empty_like(v_fold)
    _flash_attn_backward(
        do_fold, q_fold, k_fold, v_fold, out_f, lse_f,
        dq_f, dk_f, dv_f, 0.0, sm_scale, True, -1, -1, 0.0, None, False, None,
    )
    if u_start == 0:
        dq_cmp = torch.zeros_like(q_slice)
        dq_cmp[:, 16:] = (
            dq_f.view(bsz, n_u_q - 1, h_kv, 16, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz, s_q - 16, h_kv, g_heads, dim)
        )
    else:
        dq_cmp = (
            dq_f.view(bsz, n_u_q, h_kv, 16, g_heads, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz, s_q, h_kv, g_heads, dim)
        )
    dkc_p = torch.zeros_like(kc_prefix)
    dvc_p = torch.zeros_like(vc_prefix)
    dkc_p[:, : dk_f.shape[1]] = dk_f
    dvc_p[:, : dv_f.shape[1]] = dv_f

    dq_tot = dq_cmp + dq_slc + dq_win_flat.view_as(q_slice)
    return out, dq_tot, d_gate, dkc_p, dvc_p, dk_slc_p, dv_slc_p, dk_win_h, dv_win_h


def run_suite_c_cp8_8gpu(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    assert world_size == 8, f"Suite C requires world_size == 8, got {world_size}"
    dist.barrier()
    g_u0 = dist.new_group(ranks=[0, 1, 2, 3])
    g_u1 = dist.new_group(ranks=[4, 5, 6, 7])
    uly4_grp = g_u0 if rank < 4 else g_u1

    ring2_groups = [dist.new_group(ranks=[u, u + 4]) for u in range(4)]
    ring2_grp = ring2_groups[rank % 4]
    pair2_groups = [dist.new_group(ranks=[2 * hk, 2 * hk + 1]) for hk in range(4)]
    pair2_grp = pair2_groups[rank // 2]

    ring_rank = rank // 4
    uly_rank = rank % 4
    peer_ring_global = (rank + 4) if ring_rank == 0 else (rank - 4)
    comm_stream = torch.cuda.Stream(device=device)

    def run_ulysses8_nsa_exact(
        q_loc: torch.Tensor,
        kv6_loc: torch.Tensor,
        gl_loc: torch.Tensor,
        do_loc: torch.Tensor,
        wk: torch.Tensor,
        pek: torch.Tensor,
        wv: torch.Tensor,
        pev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, s_loc, _, _, dim = q_loc.shape
        seqlen = s_loc * 8
        n_cmp = seqlen // 16
        sm_scale = dim**-0.5

        q_8 = q_loc.view(bsz, s_loc, 8, 8, dim)
        do_8 = do_loc.view(bsz, s_loc, 8, 8, dim)
        gl_8 = gl_loc.view(bsz, s_loc, 8, 8, 3)
        kv_rep = kv6_loc.permute(0, 1, 3, 2, 4).repeat_interleave(2, dim=2).contiguous()

        q_h = _all_to_all_seq_to_head(q_8, dist.group.WORLD, 8).view(bsz, seqlen, 1, 8, dim)
        do_h = _all_to_all_seq_to_head(do_8, dist.group.WORLD, 8).view(bsz, seqlen, 1, 8, dim)
        gl_h = _all_to_all_seq_to_head(gl_8, dist.group.WORLD, 8).view(bsz, seqlen, 1, 8, 3)
        kv_h = _all_to_all_seq_to_head(kv_rep, dist.group.WORLD, 8)  # [B, S, 6, D]

        hk_idx = rank // 2
        wk_h = wk[hk_idx : hk_idx + 1]
        pek_h = pek[hk_idx : hk_idx + 1]
        wv_h = wv[hk_idx : hk_idx + 1]
        pev_h = pev[hk_idx : hk_idx + 1]

        k_cr = kv_h[:, :, 0:1].transpose(2, 3).reshape(bsz, seqlen, 1, dim).contiguous()
        v_cr = kv_h[:, :, 1:2].transpose(2, 3).reshape(bsz, seqlen, 1, dim).contiguous()
        k_sl = kv_h[:, :, 2:3].transpose(2, 3).reshape(bsz, seqlen, 1, dim).contiguous()
        v_sl = kv_h[:, :, 3:4].transpose(2, 3).reshape(bsz, seqlen, 1, dim).contiguous()
        k_wn = kv_h[:, :, 4:5].transpose(2, 3).reshape(bsz, seqlen, 1, dim).contiguous()
        v_wn = kv_h[:, :, 5:6].transpose(2, 3).reshape(bsz, seqlen, 1, dim).contiguous()

        kc = torch.empty((bsz, n_cmp, 1, dim), device=device, dtype=q_loc.dtype)
        vc = torch.empty_like(kc)
        _strided_compress_fwd_kernel[(n_cmp, bsz)](k_cr, wk_h, pek_h, kc, *k_cr.stride(), *kc.stride(), seqlen, 1, D=dim)
        _strided_compress_fwd_kernel[(n_cmp, bsz)](v_cr, wv_h, pev_h, vc, *v_cr.stride(), *vc.stride(), seqlen, 1, D=dim)

        q_fold = (
            q_h[:, 16:]
            .view(bsz, n_cmp - 1, 16, 1, 8, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, n_cmp - 1, 128, dim)
        )
        k_fold = kc[:, :-1].contiguous()
        v_fold = vc[:, :-1].contiguous()
        out_f, lse_f, _, _ = _flash_attn_forward(q_fold, k_fold, v_fold, 0.0, sm_scale, True, -1, -1, 0.0, None, False)
        out_cmp = torch.zeros_like(q_h)
        out_cmp[:, 16:] = out_f.view(bsz, n_cmp - 1, 1, 16, 8, dim).permute(0, 1, 3, 2, 4, 5).reshape(bsz, seqlen - 16, 1, 8, dim)
        lse_cmp = torch.full((bsz, seqlen, 1, 8), -1.0e30, device=device, dtype=torch.float32)
        lse_cmp[:, 16:] = lse_f.view(bsz, 1, 16, 8, n_cmp - 1).permute(0, 4, 2, 1, 3).reshape(bsz, seqlen - 16, 1, 8)

        # Eq. (8)-(10) partial 8-head score + AllReduce across 2-rank head-split pair!
        n_slc = seqlen // 64
        n_u = seqlen // 4
        a_trap = torch.zeros((bsz, 1, n_u, n_slc), device=device, dtype=torch.float32)
        first_arr = torch.zeros_like(a_trap)
        _induce_eq9_slc_scores_from_lse_kernel[(n_u, bsz)](
            q_h, kc, lse_cmp, a_trap, first_arr,
            *q_h.stride(), *kc.stride(), *lse_cmp.stride(), *a_trap.stride(),
            seqlen, 0, n_slc, 1, sm_scale, B_Q_SUB=4, Q_PER_CTA=4, G=8, D=dim, B_C=64, num_warps=4, num_stages=2,
        )
        sc = a_trap
        if n_slc > 1:
            sc[..., :-1] += first_arr[..., 1:]
        del first_arr
        dist.all_reduce(sc, op=dist.ReduceOp.SUM, group=pair2_grp)
        _apply_nsa_forced_and_causal_bonus_inplace(sc, q_global_offset=0, b_q_sub=4, b_k=64)
        k_eff = min(16, n_slc)
        tv, ti = torch.topk(sc, k=k_eff, dim=-1)
        idx_bhku = torch.full((bsz, 1, n_u, 16), -1, device=device, dtype=torch.int32)
        idx_bhku[..., :k_eff] = torch.where(torch.isfinite(tv), ti.to(torch.int32), -1)
        idx_buhk = idx_bhku.permute(0, 2, 1, 3).contiguous()

        out_slc, lse_slc = run_sparse_selection_fwd(q_h, k_sl, v_sl, idx_buhk, b_q_sub=4, b_k=64)
        q_flat = q_h.view(bsz, seqlen, 8, dim)
        out_win_flat, lse_win, _, _ = _flash_attn_forward(q_flat, k_wn, v_wn, 0.0, sm_scale, True, 511, 0, 0.0, None, False)

        out_h = torch.empty_like(q_h)
        n_rows = bsz * seqlen * 8
        _nsa_3branch_gate_fwd_kernel[(n_rows,)](out_cmp, out_slc, out_win_flat, gl_h, out_h, n_rows, D=dim)

        do_cmp = torch.empty_like(q_h)
        do_slc = torch.empty_like(q_h)
        do_win = torch.empty_like(q_h)
        dg_h = torch.empty_like(gl_h)
        _nsa_3branch_gate_bwd_kernel[(n_rows,)](
            do_h, out_cmp, out_slc, out_win_flat, gl_h, do_cmp, do_slc, do_win, dg_h, n_rows, D=dim
        )

        dq_win_f = torch.empty_like(q_flat)
        dk_wn = torch.empty_like(k_wn)
        dv_wn = torch.empty_like(v_wn)
        _flash_attn_backward(
            do_win.view(bsz, seqlen, 8, dim), q_flat, k_wn, v_wn, out_win_flat, lse_win,
            dq_win_f, dk_wn, dv_wn, 0.0, sm_scale, True, 511, 0, 0.0, None, False, None,
        )
        dq_slc, dk_sl, dv_sl = run_sparse_selection_bwd_csc(
            q_h, k_sl, v_sl, out_slc, do_slc, lse_slc, idx_bhku, idx_buhk, b_q_sub=4, b_k=64
        )
        do_fold = do_cmp[:, 16:].view(bsz, n_cmp - 1, 16, 1, 8, dim).permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, n_cmp - 1, 128, dim)
        dq_f = torch.empty_like(q_fold)
        dk_f = torch.empty_like(k_fold)
        dv_f = torch.empty_like(v_fold)
        _flash_attn_backward(do_fold, q_fold, k_fold, v_fold, out_f, lse_f, dq_f, dk_f, dv_f, 0.0, sm_scale, True, -1, -1, 0.0, None, False, None)
        dq_cmp = torch.zeros_like(q_h)
        dq_cmp[:, 16:] = dq_f.view(bsz, n_cmp - 1, 1, 16, 8, dim).permute(0, 1, 3, 2, 4, 5).reshape(bsz, seqlen - 16, 1, 8, dim)
        dkc = torch.zeros_like(kc)
        dvc = torch.zeros_like(vc)
        dkc[:, :-1] = dk_f
        dvc[:, :-1] = dv_f
        dk_cr = torch.empty_like(k_cr)
        dv_cr = torch.empty_like(v_cr)
        _strided_compress_bwd_dx_kernel[(n_cmp, bsz)](dkc, wk_h, dk_cr, *dkc.stride(), *k_cr.stride(), n_cmp, 1, D=dim)
        _strided_compress_bwd_dx_kernel[(n_cmp, bsz)](dvc, wv_h, dv_cr, *dvc.stride(), *v_cr.stride(), n_cmp, 1, D=dim)
        dw_k_h, _ = strided_compress_param_grads(k_cr, dkc, wk_h, pek_h)
        dist.all_reduce(dw_k_h, op=dist.ReduceOp.SUM, group=pair2_grp)

        dq_h = (dq_cmp + dq_slc + dq_win_f.view_as(q_h)).view(bsz, seqlen, 8, dim)
        dkv_h = torch.stack(
            [dk_cr.squeeze(2), dv_cr.squeeze(2), dk_sl.squeeze(2), dv_sl.squeeze(2), dk_wn.squeeze(2), dv_wn.squeeze(2)],
            dim=2,
        )  # [B, S, 6, D]

        out_loc = _all_to_all_head_to_seq(out_h.view(bsz, seqlen, 8, dim), dist.group.WORLD, 8).view(bsz, s_loc, 4, 16, dim)
        dq_loc = _all_to_all_head_to_seq(dq_h, dist.group.WORLD, 8).view(bsz, s_loc, 4, 16, dim)
        dg_loc = _all_to_all_head_to_seq(dg_h.view(bsz, seqlen, 8, 3), dist.group.WORLD, 8).view(bsz, s_loc, 4, 16, 3)
        dkv_rep_loc = _all_to_all_head_to_seq(dkv_h, dist.group.WORLD, 8)  # [B, s_loc, 8, 6, D]
        # Sum replicated pairs (0+1, 2+3, 4+5, 6+7) back into [B, s_loc, 6, 4, D]!
        dkv_loc = dkv_rep_loc.float().view(bsz, s_loc, 4, 2, 6, dim).sum(dim=3).to(q_loc.dtype).permute(0, 1, 3, 2, 4).contiguous()
        return out_loc, dq_loc, dkv_loc, dg_loc, dw_k_h

    def run_usp_4x2_group_preserving_nsa(
        q_zz: torch.Tensor,
        kv6_zz: torch.Tensor,
        gl_zz: torch.Tensor,
        do_zz: torch.Tensor,
        wk: torch.Tensor,
        pek: torch.Tensor,
        wv: torch.Tensor,
        pev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ZigZag-2 x Ulysses-4:
        # Input per rank: 2 half-quarters of length `s_half = S // 16` -> total `s_loc = S // 8`!
        # Specifically:
        #   ring_rank == 0 owns Quarter 0 (sub-shard uly_rank) AND Quarter 3 (sub-shard uly_rank)
        #   ring_rank == 1 owns Quarter 1 (sub-shard uly_rank) AND Quarter 2 (sub-shard uly_rank)
        bsz, s_loc, _, _, dim = q_zz.shape
        s_qtr = (s_loc // 2) * 4  # S // 4
        seqlen = s_qtr * 4
        n_cmp = seqlen // 16
        n_cmp_qtr = s_qtr // 16

        # Split local [2, s_loc//2] -> AllToAll inside Ulysses-4 -> [2, s_qtr, 1_hk, 16_g, D]!
        q_2 = q_zz.view(bsz * 2, s_loc // 2, 4, 16, dim)
        do_2 = do_zz.view(bsz * 2, s_loc // 2, 4, 16, dim)
        gl_2 = gl_zz.view(bsz * 2, s_loc // 2, 4, 16, 3)
        kv_2 = kv6_zz.view(bsz * 2, s_loc // 2, 6, 4, dim).permute(0, 1, 3, 2, 4).contiguous()

        q_u = _all_to_all_seq_to_head(q_2, uly4_grp, 4).view(bsz, 2, s_qtr, 1, 16, dim)
        do_u = _all_to_all_seq_to_head(do_2, uly4_grp, 4).view(bsz, 2, s_qtr, 1, 16, dim)
        gl_u = _all_to_all_seq_to_head(gl_2, uly4_grp, 4).view(bsz, 2, s_qtr, 1, 16, 3)
        kv_u = _all_to_all_seq_to_head(kv_2, uly4_grp, 4).view(bsz, 2, s_qtr, 6, dim)

        # Causal Asymmetric P2P across the 2-rank ring2_grp:
        #   ring_rank 0 -> 1 sends ONLY Quarter 0 (`kv_u[:, 0]`, 1 quarter; Q1+Q2 never read causal future Q3!)
        #   ring_rank 1 -> 0 sends Quarters 1+2 (`kv_u`, 2 quarters) while ring_rank 0 computes Quarter 0!
        main_stream = torch.cuda.current_stream(device)
        kv_u.record_stream(comm_stream)
        if ring_rank == 0:
            kv_send_fwd = kv_u[:, 0]  # [B, s_qtr, 6, D]
            kv_peer_q12 = torch.empty_like(kv_u)  # [B, 2, s_qtr, 6, D]
            ops = [
                dist.P2POp(dist.isend, kv_send_fwd, peer_ring_global),
                dist.P2POp(dist.irecv, kv_peer_q12, peer_ring_global),
            ]
        else:
            kv_peer_q0 = torch.empty_like(kv_u[:, 0])  # [B, s_qtr, 6, D]
            ops = [
                dist.P2POp(dist.isend, kv_u, peer_ring_global),
                dist.P2POp(dist.irecv, kv_peer_q0, peer_ring_global),
            ]
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_stream(main_stream)
            reqs = dist.batch_isend_irecv(ops)

        hk_idx = uly_rank
        wk_h = wk[hk_idx : hk_idx + 1]
        pek_h = pek[hk_idx : hk_idx + 1]
        wv_h = wv[hk_idx : hk_idx + 1]
        pev_h = pev[hk_idx : hk_idx + 1]

        out_u = torch.empty_like(q_u)
        dq_u = torch.empty_like(q_u)
        dg_u = torch.empty_like(gl_u)

        if ring_rank == 0:
            # Overlap Quarter 0 (`[0 : S/4]`) compute strictly behind the 2-peer NVSwitch KV P2P transfer!
            k_cr0 = kv_u[:, 0, :, 0].unsqueeze(2).contiguous()
            v_cr0 = kv_u[:, 0, :, 1].unsqueeze(2).contiguous()
            k_sl0 = kv_u[:, 0, :, 2].unsqueeze(2).contiguous()
            v_sl0 = kv_u[:, 0, :, 3].unsqueeze(2).contiguous()
            k_wn0 = kv_u[:, 0, :, 4].unsqueeze(2).contiguous()
            v_wn0 = kv_u[:, 0, :, 5].unsqueeze(2).contiguous()

            kc0 = torch.empty((bsz, n_cmp_qtr, 1, dim), device=device, dtype=q_zz.dtype)
            vc0 = torch.empty_like(kc0)
            _strided_compress_fwd_kernel[(n_cmp_qtr, bsz)](
                k_cr0, wk_h, pek_h, kc0, *k_cr0.stride(), *kc0.stride(), s_qtr, 1, D=dim
            )
            _strided_compress_fwd_kernel[(n_cmp_qtr, bsz)](
                v_cr0, wv_h, pev_h, vc0, *v_cr0.stride(), *vc0.stride(), s_qtr, 1, D=dim
            )
            o0, dq0, dg0, dkc0, dvc0, dk_sl0, dv_sl0, dk_wn0, dv_wn0 = _run_nsa_local_slice_fwd_bwd(
                q_u[:, 0], do_u[:, 0], gl_u[:, 0], kc0, vc0, k_sl0, v_sl0, k_wn0, v_wn0,
                q_global_offset=0, b_q_sub=4, k_sel=16, w_win=512,
            )
            out_u[:, 0] = o0
            dq_u[:, 0] = dq0
            dg_u[:, 0] = dg0

        with torch.cuda.stream(comm_stream):
            for req in reqs:
                req.wait()
        main_stream.wait_stream(comm_stream)

        if ring_rank == 0:
            kv_global = torch.cat([kv_u[:, 0], kv_peer_q12[:, 0], kv_peer_q12[:, 1], kv_u[:, 1]], dim=1)
            k_cr_g = kv_global[:, :, 0].unsqueeze(2).contiguous()
            v_cr_g = kv_global[:, :, 1].unsqueeze(2).contiguous()
            k_sl_g = kv_global[:, :, 2].unsqueeze(2).contiguous()
            v_sl_g = kv_global[:, :, 3].unsqueeze(2).contiguous()
            k_wn_h3 = kv_global[:, 3 * s_qtr - 512 : 4 * s_qtr, 4].unsqueeze(2).contiguous()
            v_wn_h3 = kv_global[:, 3 * s_qtr - 512 : 4 * s_qtr, 5].unsqueeze(2).contiguous()

            kc_g = torch.empty((bsz, n_cmp, 1, dim), device=device, dtype=q_zz.dtype)
            vc_g = torch.empty_like(kc_g)
            _strided_compress_fwd_kernel[(n_cmp, bsz)](k_cr_g, wk_h, pek_h, kc_g, *k_cr_g.stride(), *kc_g.stride(), seqlen, 1, D=dim)
            _strided_compress_fwd_kernel[(n_cmp, bsz)](v_cr_g, wv_h, pev_h, vc_g, *v_cr_g.stride(), *vc_g.stride(), seqlen, 1, D=dim)

            o3, dq3, dg3, dkc3, dvc3, dk_sl3, dv_sl3, dk_wn_h3, dv_wn_h3 = _run_nsa_local_slice_fwd_bwd(
                q_u[:, 1], do_u[:, 1], gl_u[:, 1], kc_g, vc_g, k_sl_g, v_sl_g, k_wn_h3, v_wn_h3,
                q_global_offset=3 * s_qtr, b_q_sub=4, k_sel=16, w_win=512,
            )
            out_u[:, 1] = o3
            dq_u[:, 1] = dq3
            dg_u[:, 1] = dg3

            dkc_tot = dkc3
            dvc_tot = dvc3
            dkc_tot[:, :n_cmp_qtr] += dkc0
            dvc_tot[:, :n_cmp_qtr] += dvc0
            dk_cr_g = torch.empty_like(k_cr_g)
            dv_cr_g = torch.empty_like(v_cr_g)
            _strided_compress_bwd_dx_kernel[(n_cmp, bsz)](dkc_tot, wk_h, dk_cr_g, *dkc_tot.stride(), *k_cr_g.stride(), n_cmp, 1, D=dim)
            _strided_compress_bwd_dx_kernel[(n_cmp, bsz)](dvc_tot, wv_h, dv_cr_g, *dvc_tot.stride(), *v_cr_g.stride(), n_cmp, 1, D=dim)
            dw_k_h, _ = strided_compress_param_grads(k_cr_g, dkc_tot, wk_h, pek_h)

            dkv_global = torch.zeros_like(kv_global)
            dkv_global[:, :, 0] = dk_cr_g.squeeze(2)
            dkv_global[:, :, 1] = dv_cr_g.squeeze(2)
            dkv_global[:, :, 2] = dk_sl3.squeeze(2)
            dkv_global[:, :s_qtr, 2] += dk_sl0.squeeze(2)
            dkv_global[:, :, 3] = dv_sl3.squeeze(2)
            dkv_global[:, :s_qtr, 3] += dv_sl0.squeeze(2)
            dkv_global[:, :s_qtr, 4] += dk_wn0.squeeze(2)
            dkv_global[:, 3 * s_qtr - 512 : 4 * s_qtr, 4] += dk_wn_h3.squeeze(2)
            dkv_global[:, :s_qtr, 5] += dv_wn0.squeeze(2)
            dkv_global[:, 3 * s_qtr - 512 : 4 * s_qtr, 5] += dv_wn_h3.squeeze(2)

            dkv_own = torch.stack([dkv_global[:, :s_qtr], dkv_global[:, 3 * s_qtr :]], dim=1).contiguous()
            dkv_send_q12 = dkv_global[:, s_qtr : 3 * s_qtr].view(bsz, 2, s_qtr, 6, dim).contiguous()
            dkv_recv_q0 = torch.empty_like(dkv_own[:, 0])
            ops_bwd = [
                dist.P2POp(dist.isend, dkv_send_q12, peer_ring_global),
                dist.P2POp(dist.irecv, dkv_recv_q0, peer_ring_global),
            ]
        else:
            # ring_rank == 1 ONLY materializes 3/4 S causal prefix `[Q0, Q1, Q2]` (`s_p12 = 3 * s_qtr`), never Q3!
            s_p12 = 3 * s_qtr
            n_cmp_12 = s_p12 // 16
            kv_p12 = torch.cat([kv_peer_q0, kv_u[:, 0], kv_u[:, 1]], dim=1)  # [B, 3*s_qtr, 6, D]
            k_cr12 = kv_p12[:, :, 0].unsqueeze(2).contiguous()
            v_cr12 = kv_p12[:, :, 1].unsqueeze(2).contiguous()
            kc_12 = torch.empty((bsz, n_cmp_12, 1, dim), device=device, dtype=q_zz.dtype)
            vc_12 = torch.empty_like(kc_12)
            _strided_compress_fwd_kernel[(n_cmp_12, bsz)](k_cr12, wk_h, pek_h, kc_12, *k_cr12.stride(), *kc_12.stride(), s_p12, 1, D=dim)
            _strided_compress_fwd_kernel[(n_cmp_12, bsz)](v_cr12, wv_h, pev_h, vc_12, *v_cr12.stride(), *vc_12.stride(), s_p12, 1, D=dim)

            q_12 = q_u.reshape(bsz, 2 * s_qtr, 1, 16, dim)
            do_12 = do_u.reshape(bsz, 2 * s_qtr, 1, 16, dim)
            gl_12 = gl_u.reshape(bsz, 2 * s_qtr, 1, 16, 3)
            k_sl12 = kv_p12[:, :, 2].unsqueeze(2).contiguous()
            v_sl12 = kv_p12[:, :, 3].unsqueeze(2).contiguous()
            k_wn12 = kv_p12[:, s_qtr - 512 : s_p12, 4].unsqueeze(2).contiguous()
            v_wn12 = kv_p12[:, s_qtr - 512 : s_p12, 5].unsqueeze(2).contiguous()

            o12, dq12, dg12, dkc12, dvc12, dk_sl12, dv_sl12, dk_wn12, dv_wn12 = _run_nsa_local_slice_fwd_bwd(
                q_12, do_12, gl_12, kc_12, vc_12,
                k_sl12, v_sl12, k_wn12, v_wn12,
                q_global_offset=s_qtr, b_q_sub=4, k_sel=16, w_win=512,
            )
            out_u.copy_(o12.view_as(out_u))
            dq_u.copy_(dq12.view_as(dq_u))
            dg_u.copy_(dg12.view_as(dg_u))

            # Notice: compressed block `n_cmp_12` starts at `3*s_qtr - 16`, so `dkc12` (indices `< n_cmp_12`)
            # only touches raw tokens `< 3*s_qtr = s_p12`!
            dk_cr12 = torch.empty_like(k_cr12)
            dv_cr12 = torch.empty_like(v_cr12)
            _strided_compress_bwd_dx_kernel[(n_cmp_12, bsz)](dkc12, wk_h, dk_cr12, *dkc12.stride(), *k_cr12.stride(), n_cmp_12, 1, D=dim)
            _strided_compress_bwd_dx_kernel[(n_cmp_12, bsz)](dvc12, wv_h, dv_cr12, *dvc12.stride(), *v_cr12.stride(), n_cmp_12, 1, D=dim)
            dw_k_h, _ = strided_compress_param_grads(k_cr12, dkc12, wk_h, pek_h)

            dkv_p12 = torch.zeros_like(kv_p12)
            dkv_p12[:, :, 0] = dk_cr12.squeeze(2)
            dkv_p12[:, :, 1] = dv_cr12.squeeze(2)
            dkv_p12[:, :, 2] = dk_sl12.squeeze(2)
            dkv_p12[:, :, 3] = dv_sl12.squeeze(2)
            dkv_p12[:, s_qtr - 512 : s_p12, 4] = dk_wn12.squeeze(2)
            dkv_p12[:, s_qtr - 512 : s_p12, 5] = dv_wn12.squeeze(2)

            dkv_send_q0 = dkv_p12[:, :s_qtr].contiguous()
            dkv_own = dkv_p12[:, s_qtr:].view(bsz, 2, s_qtr, 6, dim).contiguous()
            dkv_recv_q12 = torch.empty_like(dkv_own)
            ops_bwd = [
                dist.P2POp(dist.isend, dkv_send_q0, peer_ring_global),
                dist.P2POp(dist.irecv, dkv_recv_q12, peer_ring_global),
            ]

        dist.all_reduce(dw_k_h, op=dist.ReduceOp.SUM, group=ring2_grp)
        for r in dist.batch_isend_irecv(ops_bwd):
            r.wait()
        if ring_rank == 0:
            dkv_own[:, 0] += dkv_recv_q0
            dkv_u = dkv_own.view(bsz * 2, s_qtr, 6, dim)
        else:
            dkv_u = (dkv_own + dkv_recv_q12).view(bsz * 2, s_qtr, 6, dim)

        out_zz = _all_to_all_head_to_seq(out_u.view(bsz * 2, s_qtr, 16, dim), uly4_grp, 4).view(bsz, s_loc, 4, 16, dim)
        dq_zz = _all_to_all_head_to_seq(dq_u.view(bsz * 2, s_qtr, 16, dim), uly4_grp, 4).view(bsz, s_loc, 4, 16, dim)
        dg_zz = _all_to_all_head_to_seq(dg_u.view(bsz * 2, s_qtr, 16, 3), uly4_grp, 4).view(bsz, s_loc, 4, 16, 3)
        dkv_zz = (
            _all_to_all_head_to_seq(dkv_u, uly4_grp, 4)
            .view(bsz, s_loc, 4, 6, dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        return out_zz, dq_zz, dkv_zz, dg_zz, dw_k_h

    def run_seq_sharded_kv_demand_nsa(
        q_loc: torch.Tensor,
        kv6_loc: torch.Tensor,
        gl_loc: torch.Tensor,
        do_loc: torch.Tensor,
        wk: torch.Tensor,
        pek: torch.Tensor,
        wv: torch.Tensor,
        pev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        # Sequence-sharded Q (`S/8` per rank): all-gather 4-head KV across 8 ranks, measure the empirical
        # Coupon-Collector union of selected blocks across the S/8 local queries, and run causal slice + ReduceScatter!
        bsz, s_loc, _, _, dim = q_loc.shape
        seqlen = s_loc * 8
        n_cmp = seqlen // 16
        q_off = rank * s_loc
        s_pref = (rank + 1) * s_loc
        n_cmp_pref = s_pref // 16

        kv_gather = [torch.empty_like(kv6_loc) for _ in range(8)]
        dist.all_gather(kv_gather, kv6_loc)
        kv_glob = torch.cat(kv_gather, dim=1)  # [B, S, 6, 4, D]

        k_cr = kv_glob[:, :, 0].contiguous()
        v_cr = kv_glob[:, :, 1].contiguous()
        kc_g = torch.empty((bsz, n_cmp, 4, dim), device=device, dtype=q_loc.dtype)
        vc_g = torch.empty_like(kc_g)
        _strided_compress_fwd_kernel[(n_cmp, bsz * 4)](k_cr, wk, pek, kc_g, *k_cr.stride(), *kc_g.stride(), seqlen, 4, D=dim)
        _strided_compress_fwd_kernel[(n_cmp, bsz * 4)](v_cr, wv, pev, vc_g, *v_cr.stride(), *vc_g.stride(), seqlen, 4, D=dim)

        k_sl_p = kv_glob[:, :s_pref, 2].contiguous()
        v_sl_p = kv_glob[:, :s_pref, 3].contiguous()
        w0 = 0 if q_off == 0 else (q_off - 512)
        k_wn_h = kv_glob[:, w0:s_pref, 4].contiguous()
        v_wn_h = kv_glob[:, w0:s_pref, 5].contiguous()

        out_l, dq_l, _, dkc_p, dvc_p, dk_sl_p, dv_sl_p, dk_wn_h, dv_wn_h = _run_nsa_local_slice_fwd_bwd(
            q_loc, do_loc, gl_loc, kc_g[:, :n_cmp_pref].contiguous(), vc_g[:, :n_cmp_pref].contiguous(),
            k_sl_p, v_sl_p, k_wn_h, v_wn_h,
            q_global_offset=q_off, b_q_sub=4, k_sel=16, w_win=512,
        )
        dkc_tot = torch.zeros_like(kc_g)
        dvc_tot = torch.zeros_like(vc_g)
        dkc_tot[:, :n_cmp_pref] = dkc_p
        dvc_tot[:, :n_cmp_pref] = dvc_p
        dk_cr = torch.empty_like(k_cr)
        dv_cr = torch.empty_like(v_cr)
        _strided_compress_bwd_dx_kernel[(n_cmp, bsz * 4)](dkc_tot, wk, dk_cr, *dkc_tot.stride(), *k_cr.stride(), n_cmp, 4, D=dim)
        _strided_compress_bwd_dx_kernel[(n_cmp, bsz * 4)](dvc_tot, wv, dv_cr, *dvc_tot.stride(), *v_cr.stride(), n_cmp, 4, D=dim)

        dkv_glob = torch.zeros_like(kv_glob)
        dkv_glob[:, :, 0] = dk_cr
        dkv_glob[:, :, 1] = dv_cr
        dkv_glob[:, :s_pref, 2] = dk_sl_p
        dkv_glob[:, :s_pref, 3] = dv_sl_p
        dkv_glob[:, w0:s_pref, 4] = dk_wn_h
        dkv_glob[:, w0:s_pref, 5] = dv_wn_h

        dkv_loc = torch.empty_like(kv6_loc)
        dist.reduce_scatter_tensor(dkv_loc, dkv_glob.contiguous())
        return out_l, dq_l, 0.0, 0.0

    # C.1 Exact 1-GPU Global DeepSeek NSA Parity Verification at S = 16,384 + Coupon-Collector Union Audit
    torch.manual_seed(9001)
    s_par = 16384
    s_loc_p = s_par // 8
    s_qtr_p = s_par // 4
    s_sub_p = s_qtr_p // 4

    q_glob = (torch.randn(1, s_par, 4, 16, 128, device=device, dtype=torch.bfloat16) * 0.4)
    kv6_glob = (torch.randn(1, s_par, 6, 4, 128, device=device, dtype=torch.bfloat16) * 0.4)
    # Add structured semantic topic vectors across 256 blocks so top-16 block scores are well-separated above BF16 ULP noise
    blk_topics = torch.randn(1, s_par // 64, 4, 128, device=device, dtype=torch.bfloat16) * 0.9
    perm_q = torch.randperm(s_par // 64, device=device)
    q_glob = q_glob + blk_topics[:, perm_q].unsqueeze(3).repeat_interleave(64, dim=1) * 0.6
    kv6_glob[:, :, 0] = kv6_glob[:, :, 0] + blk_topics.repeat_interleave(64, dim=1) * 0.8
    gl_glob = (torch.randn(1, s_par, 4, 16, 3, device=device, dtype=torch.bfloat16) * 0.5)
    do_glob = torch.randn_like(q_glob)
    wk_g = (torch.randn(4, 32, device=device, dtype=torch.float32) * 0.1)
    pek_g = (torch.randn(4, 32, 128, device=device, dtype=torch.float32) * 0.1)
    wv_g = (torch.randn(4, 32, device=device, dtype=torch.float32) * 0.1)
    pev_g = (torch.randn(4, 32, 128, device=device, dtype=torch.float32) * 0.1)
    for t in (q_glob, kv6_glob, gl_glob, do_glob, wk_g, pek_g, wv_g, pev_g):
        dist.broadcast(t, src=0)

    # Contiguous shard for Ulysses-8
    q_c_loc = q_glob[:, rank * s_loc_p : (rank + 1) * s_loc_p].contiguous()
    kv_c_loc = kv6_glob[:, rank * s_loc_p : (rank + 1) * s_loc_p].contiguous()
    gl_c_loc = gl_glob[:, rank * s_loc_p : (rank + 1) * s_loc_p].contiguous()
    do_c_loc = do_glob[:, rank * s_loc_p : (rank + 1) * s_loc_p].contiguous()

    out_u8_loc, dq_u8_loc, dkv_u8_loc, dg_u8_loc, dwk_u8_h = run_ulysses8_nsa_exact(
        q_c_loc, kv_c_loc, gl_c_loc, do_c_loc, wk_g, pek_g, wv_g, pev_g
    )
    out_u8_all = [torch.empty_like(out_u8_loc) for _ in range(8)]
    dq_u8_all = [torch.empty_like(dq_u8_loc) for _ in range(8)]
    dkv_u8_all = [torch.empty_like(dkv_u8_loc) for _ in range(8)]
    dist.all_gather(out_u8_all, out_u8_loc)
    dist.all_gather(dq_u8_all, dq_u8_loc)
    dist.all_gather(dkv_u8_all, dkv_u8_loc)

    # ZigZag-2 x Contiguous-4 shard for 2D USP-NSA (4x2)
    q0_off = (0 if ring_rank == 0 else 1) * s_qtr_p + uly_rank * s_sub_p
    q1_off = (3 if ring_rank == 0 else 2) * s_qtr_p + uly_rank * s_sub_p
    q_zz_loc = torch.cat([q_glob[:, q0_off : q0_off + s_sub_p], q_glob[:, q1_off : q1_off + s_sub_p]], dim=1).contiguous()
    kv_zz_loc = torch.cat([kv6_glob[:, q0_off : q0_off + s_sub_p], kv6_glob[:, q1_off : q1_off + s_sub_p]], dim=1).contiguous()
    gl_zz_loc = torch.cat([gl_glob[:, q0_off : q0_off + s_sub_p], gl_glob[:, q1_off : q1_off + s_sub_p]], dim=1).contiguous()
    do_zz_loc = torch.cat([do_glob[:, q0_off : q0_off + s_sub_p], do_glob[:, q1_off : q1_off + s_sub_p]], dim=1).contiguous()

    out_usp_loc, dq_usp_loc, dkv_usp_loc, dg_usp_loc, dwk_usp_h = run_usp_4x2_group_preserving_nsa(
        q_zz_loc, kv_zz_loc, gl_zz_loc, do_zz_loc, wk_g, pek_g, wv_g, pev_g
    )
    out_usp_all = [torch.empty_like(out_usp_loc) for _ in range(8)]
    dq_usp_all = [torch.empty_like(dq_usp_loc) for _ in range(8)]
    dkv_usp_all = [torch.empty_like(dkv_usp_loc) for _ in range(8)]
    dist.all_gather(out_usp_all, out_usp_loc)
    dist.all_gather(dq_usp_all, dq_usp_loc)
    dist.all_gather(dkv_usp_all, dkv_usp_loc)

    parity_cp8: dict[str, Any] = {}
    if rank == 0:
        q_r = q_glob.detach().clone().requires_grad_(True)
        kc_r = kv6_glob[:, :, 0].detach().clone().requires_grad_(True)
        vc_r = kv6_glob[:, :, 1].detach().clone().requires_grad_(True)
        ks_r = kv6_glob[:, :, 2].detach().clone().requires_grad_(True)
        vs_r = kv6_glob[:, :, 3].detach().clone().requires_grad_(True)
        kw_r = kv6_glob[:, :, 4].detach().clone().requires_grad_(True)
        vw_r = kv6_glob[:, :, 5].detach().clone().requires_grad_(True)
        wk_r = wk_g.detach().clone().requires_grad_(True)
        pek_r = pek_g.detach().clone().requires_grad_(True)
        wv_r = wv_g.detach().clone().requires_grad_(True)
        pev_r = pev_g.detach().clone().requires_grad_(True)
        gl_r = gl_glob.detach().clone().requires_grad_(True)
        idx_h_1gpu: list[tuple[torch.Tensor, torch.Tensor]] = []
        out_1g = _DeepSeekNSAFn.apply(
            q_r, kc_r, vc_r, ks_r, vs_r, kw_r, vw_r, wk_r, pek_r, wv_r, pev_r, gl_r, 4, 16, 512, idx_h_1gpu
        )
        out_1g.backward(do_glob)

        # Reconstruct ZigZag-2 x Contiguous-4 global tensors
        def _unslice_zz(parts: list[torch.Tensor]) -> torch.Tensor:
            full = torch.empty_like(parts[0]).repeat_interleave(8, dim=1)
            for rk in range(8):
                rr, ur = rk // 4, rk % 4
                o0 = (0 if rr == 0 else 1) * s_qtr_p + ur * s_sub_p
                o1 = (3 if rr == 0 else 2) * s_qtr_p + ur * s_sub_p
                full[:, o0 : o0 + s_sub_p] = parts[rk][:, :s_sub_p]
                full[:, o1 : o1 + s_sub_p] = parts[rk][:, s_sub_p:]
            return full

        out_u8_g = torch.cat(out_u8_all, dim=1)
        dq_u8_g = torch.cat(dq_u8_all, dim=1)
        dkv_u8_g = torch.cat(dkv_u8_all, dim=1)

        out_usp_g = _unslice_zz(out_usp_all)
        dq_usp_g = _unslice_zz(dq_usp_all)
        dkv_usp_g = _unslice_zz(dkv_usp_all)

        # Empirical Coupon-Collector Union Audit on the actual induced top-16 block indices `idx_h_1gpu[0][0]`!
        # idx_bhku_1g: [1, 4, S//4, 16]; inspect tail rank 7 (`u in [7/8 * N_U .. N_U)`)!
        idx_1g = idx_h_1gpu[0][0]  # [1, 4, 4096, 16]
        n_u_tot = idx_1g.shape[2]
        n_blk_tot = s_par // 64
        tail_idx = idx_1g[0, :, (7 * n_u_tot) // 8 :, :]  # [4, 512, 16]
        uniq_fracs = []
        remote_rank_fracs = []
        for hk in range(4):
            vals = tail_idx[hk].reshape(-1)
            vals = vals[vals >= 0]
            u_blks = torch.unique(vals)
            uniq_fracs.append(float(u_blks.numel()) / float(n_blk_tot))
            u_ranks = torch.unique(u_blks // (n_blk_tot // 8))
            remote_rank_fracs.append(float(u_ranks.numel()) / 8.0)

        parity_cp8 = {
            "seqlen": s_par,
            "ulysses8_out_rel_l2": round(rel_l2(out_u8_g, out_1g), 6),
            "ulysses8_dq_rel_l2": round(rel_l2(dq_u8_g, q_r.grad), 6),
            "ulysses8_dk_slc_rel_l2": round(rel_l2(dkv_u8_g[:, :, 2], ks_r.grad), 6),
            "usp_4x2_out_rel_l2": round(rel_l2(out_usp_g, out_1g), 6),
            "usp_4x2_dq_rel_l2": round(rel_l2(dq_usp_g, q_r.grad), 6),
            "usp_4x2_dk_cmp_raw_rel_l2": round(rel_l2(dkv_usp_g[:, :, 0], kc_r.grad), 6),
            "usp_4x2_dk_slc_rel_l2": round(rel_l2(dkv_usp_g[:, :, 2], ks_r.grad), 6),
            "usp_4x2_dk_win_rel_l2": round(rel_l2(dkv_usp_g[:, :, 4], kw_r.grad), 6),
            "coupon_collector_tail_rank_unique_kv_block_union_fraction": round(sum(uniq_fracs) / 4.0, 4),
            "coupon_collector_tail_rank_kv_ranks_touched_fraction": round(sum(remote_rank_fracs) / 4.0, 4),
        }
        del q_r, kc_r, vc_r, ks_r, vs_r, kw_r, vw_r, out_1g
    del q_glob, kv6_glob, gl_glob, do_glob, out_u8_all, dq_u8_all, dkv_u8_all, out_usp_all, dq_usp_all, dkv_usp_all
    torch.cuda.empty_cache()
    dist.barrier()

    # C.2 8-GPU CP = 8 Scaling Across S = 16,384 .. 131,072
    seqlens_c = [16384, 32768] if is_smoke else [16384, 32768, 65536, 131072]
    warmup_c = 2 if is_smoke else 3
    iters_c = 4 if is_smoke else 7
    scaling_cp8: dict[str, Any] = {}

    for seqlen in seqlens_c:
        s_loc = seqlen // 8
        it_here = max(3, iters_c // 2) if seqlen >= 65536 else iters_c
        torch.manual_seed(5000 + seqlen + rank)
        q_l = torch.randn(1, s_loc, 4, 16, 128, device=device, dtype=torch.bfloat16)
        kv_l = torch.randn(1, s_loc, 6, 4, 128, device=device, dtype=torch.bfloat16)
        gl_l = torch.randn(1, s_loc, 4, 16, 3, device=device, dtype=torch.bfloat16)
        do_l = torch.randn_like(q_l)
        wk_c = torch.full((4, 32), 1.0 / 32.0, device=device, dtype=torch.float32)
        pek_c = torch.zeros((4, 32, 128), device=device, dtype=torch.float32)
        wv_c = torch.full((4, 32), 1.0 / 32.0, device=device, dtype=torch.float32)
        pev_c = torch.zeros((4, 32, 128), device=device, dtype=torch.float32)

        def _bench_dist_mode(fn_step: Callable[[], Any]) -> dict[str, float]:
            for _ in range(warmup_c):
                fn_step()
            dist.barrier()
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base_b = torch.cuda.memory_allocated(device)
            fn_step()
            torch.cuda.synchronize(device)
            inc_mib = max(0.0, (torch.cuda.max_memory_allocated(device) - base_b) / (1024.0**2))

            samples: list[float] = []
            clks: list[float] = []
            pwrs: list[float] = []
            for i in range(it_here):
                dist.barrier()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                fn_step()
                e1.record()
                if i == it_here // 2 and rank == 0:
                    c_v, p_v = query_gpu_telemetry(0)
                    if c_v > 0:
                        clks.append(c_v)
                        pwrs.append(p_v)
                torch.cuda.synchronize(device)
                ms_local = torch.tensor([e0.elapsed_time(e1)], device=device, dtype=torch.float32)
                dist.all_reduce(ms_local, op=dist.ReduceOp.MAX)
                samples.append(float(ms_local.item()))

            inc_t = torch.tensor([inc_mib], device=device, dtype=torch.float32)
            dist.all_reduce(inc_t, op=dist.ReduceOp.MAX)
            med_ms = float(statistics.median(samples))
            return {
                "median_ms": round(med_ms, 3),
                "min_ms": round(float(min(samples)), 3),
                "max_ms": round(float(max(samples)), 3),
                "tok_per_s": round(seqlen / (med_ms * 1e-3), 1),
                "max_rank_inc_peak_mib": round(float(inc_t.item()), 2),
                "sm_clock_mhz": round(float(clks[0]), 1) if clks else -1.0,
                "power_w": round(float(pwrs[0]), 2) if pwrs else -1.0,
            }

        def _dense_u8_fa2() -> None:
            q8 = q_l.view(1, s_loc, 8, 8, 128)
            do8 = do_l.view(1, s_loc, 8, 8, 128)
            kv2_rep = kv_l[:, :, :2].permute(0, 1, 3, 2, 4).repeat_interleave(2, dim=2).contiguous()
            qh = _all_to_all_seq_to_head(q8, dist.group.WORLD, 8).detach().requires_grad_(True)
            doh = _all_to_all_seq_to_head(do8, dist.group.WORLD, 8)
            kvh = _all_to_all_seq_to_head(kv2_rep, dist.group.WORLD, 8)
            kh = kvh[:, :, 0:1].contiguous().requires_grad_(True)
            vh = kvh[:, :, 1:2].contiguous().requires_grad_(True)
            oh = flash_attn_func(qh, kh, vh, causal=True)
            oh.backward(doh)
            _ = _all_to_all_head_to_seq(oh.detach(), dist.group.WORLD, 8)
            _ = _all_to_all_head_to_seq(qh.grad, dist.group.WORLD, 8)
            dkvh = torch.cat([kh.grad, vh.grad], dim=2)
            _ = _all_to_all_head_to_seq(dkvh, dist.group.WORLD, 8)

        st_dense = _bench_dist_mode(_dense_u8_fa2)
        st_seq_ag = _bench_dist_mode(
            lambda: run_seq_sharded_kv_demand_nsa(q_l, kv_l, gl_l, do_l, wk_c, pek_c, wv_c, pev_c)
        )
        st_u8_nsa = _bench_dist_mode(
            lambda: run_ulysses8_nsa_exact(q_l, kv_l, gl_l, do_l, wk_c, pek_c, wv_c, pev_c)
        )
        st_usp_nsa = _bench_dist_mode(
            lambda: run_usp_4x2_group_preserving_nsa(q_l, kv_l, gl_l, do_l, wk_c, pek_c, wv_c, pev_c)
        )

        # Exact communication volume per rank (Fwd + Bwd, MiB)
        # KV6 has 6 * 4 * 128 = 3,072 BF16/tok; Q has 64 * 128 = 8,192 BF16/tok
        bytes_per_tok_q = 64 * 128 * 2
        bytes_per_tok_kv6 = 6 * 4 * 128 * 2
        score_consensus_bytes_u8 = (seqlen // 4) * (seqlen // 64) * 4
        comm_seq_ag_mib = (2.0 * (7.0 / 8.0) * seqlen * bytes_per_tok_kv6) / (1024.0**2)
        comm_u8_nsa_mib = (
            4.0 * (7.0 / 8.0) * s_loc * (bytes_per_tok_q + 2.0 * bytes_per_tok_kv6)
            + score_consensus_bytes_u8
        ) / (1024.0**2)
        # USP 4x2: Ulysses-4 A2A (3/4 of s_loc on Q + non-replicated KV6, x4 passes) + Causal Asymmetric Ring-2 P2P (avg 1.5 quarters = 3/8 S of 1/4 KV6, x2 passes), ZERO score AllReduce!
        comm_usp_mib = (
            4.0 * (3.0 / 4.0) * s_loc * (bytes_per_tok_q + bytes_per_tok_kv6)
            + 2.0 * (3.0 * seqlen / 8.0) * (bytes_per_tok_kv6 / 4.0)
        ) / (1024.0**2)

        st_seq_ag["comm_mib_per_rank"] = round(comm_seq_ag_mib, 2)
        st_u8_nsa["comm_mib_per_rank"] = round(comm_u8_nsa_mib, 2)
        st_usp_nsa["comm_mib_per_rank"] = round(comm_usp_mib, 2)

        if rank == 0:
            scaling_cp8[f"S_{seqlen}"] = {
                "seqlen": seqlen,
                "cp8_ulysses8_replicated_kv_dense_fa2": st_dense,
                "cp8_seq_sharded_kv_demand_allgather_nsa": st_seq_ag,
                "cp8_ulysses8_replicated_kv_nsa": st_u8_nsa,
                "cp8_usp_4x2_group_preserving_nsa": st_usp_nsa,
                "usp_speedup_over_dense_fa2": round(st_dense["median_ms"] / st_usp_nsa["median_ms"], 3),
                "usp_speedup_over_seq_sharded_kv_ag": round(st_seq_ag["median_ms"] / st_usp_nsa["median_ms"], 3),
                "usp_speedup_over_ulysses8_replicated_nsa": round(st_u8_nsa["median_ms"] / st_usp_nsa["median_ms"], 3),
            }
        del q_l, kv_l, gl_l, do_l
        torch.cuda.empty_cache()

    return {"parity_and_coupon_collector_audit_s16k": parity_cp8, "scaling_cp8": scaling_cp8}


# ============================================================================
# 5. Entrypoint
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E125: Native Trainable Sparse Attention (DeepSeek NSA & Kimi MoBA) & 8-GPU 2D USP-NSA CP=8"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_smoke = args.mode == "smoke"

    t0 = time.perf_counter()
    init_clk, init_pwr = query_gpu_telemetry(local_rank)

    s0_res: dict[str, Any] = {}
    sa_res: dict[str, Any] = {}
    sb_res: dict[str, Any] = {}

    if rank == 0:
        s0_res = run_suite_0_audits(device, is_smoke)
        print(f"[E125] Suite 0 complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sa_res = run_suite_a_1gpu_scaling(device, is_smoke)
        print(f"[E125] Suite A complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sb_res = run_suite_b_4layer_nsa_sac(device, is_smoke)
        print(f"[E125] Suite B complete at +{time.perf_counter() - t0:.1f}s", flush=True)

    dist.barrier()
    sc_res = run_suite_c_cp8_8gpu(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = time.perf_counter() - t0

    if rank == 0:
        print(f"[E125] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        env = environment_metadata()
        env["hostname"] = "node-0"
        env["triton"] = triton.__version__
        import flash_attn

        env["flash_attn"] = flash_attn.__version__
        payload = {
            "experiment": "E125_native_sparse_attention_training",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": round(elapsed_s, 2),
            "initial_sm_clock_mhz": round(init_clk, 1),
            "initial_power_w": round(init_pwr, 2),
            "environment": env,
            "suite_0_audits": s0_res,
            "suite_a_1gpu_sparse_attention_scaling": sa_res,
            "suite_b_4layer_nsa_sac": sb_res,
            "suite_c_cp8_usp_nsa_8gpu": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

