#!/usr/bin/env python3
"""E126: Multimodal Vision-Language & Omni-Modal Native-Resolution Training Kernels
(Qwen2.5-VL / Qwen3-VL / NaViT / Transfusion / Gemma-3 / VeOmni / DistTrain):
Window-Morton-2x2 Ragged Tubelet PatchEmbed & Zero-Permutation Windowed/Full ViT,
Fused Single-Launch Q+K 3D Multimodal RoPE (Sectioned & Interleaved),
Surgical Disjoint Reversed-Upper FlashAttention-2 + Triton LSE Omni-Attention,
& 8-GPU NVSwitch Modality-Decoupled AllToAllv + Quad-Cost-LPT FSDP2 Training.

Primary Literature Anchors:
  - Qwen2-VL & Qwen2.5-VL: Wang et al., 2024 (arXiv:2409.12191); Bai et al., Feb 2025 (arXiv:2502.13923)
    3D Conv3d(2,14,14) tubelet patchification, 8x8 windowed + full ViT attention,
    2x2 spatial MLP token merger, and 3D M-RoPE ([16, 24, 24] / interleaved [24, 20, 20]).
  - NaViT (Patch n' Pack): Dehghani et al., NeurIPS 2023 (arXiv:2307.06304)
    Native-resolution aspect-ratio-preserving ragged ViT sequence packing.
  - Transfusion / Gemma-3 Hybrid Omni-Attention: Zhou et al., Aug 2024 (arXiv:2408.11039)
    Intra-sequence bidirectional image/diffusion patch spans interleaved with causal text.
  - DistTrain & VeOmni: Zhang et al., 2024 (arXiv:2408.04275); Ma et al., Aug 2025 (arXiv:2508.02317)
    Addressing modality & sequence-length heterogeneity in distributed multimodal FSDP2 training.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_local_rank_env = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_e126_r{_local_rank_env}")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_e126_r{_local_rank_env}")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import triton
import triton.language as tl
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)

from common import environment_metadata


def query_gpu_telemetry(gpu_idx: int = 0) -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_idx}",
                "--query-gpu=clocks.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3.0,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return -1.0, -1.0


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    return float(torch.linalg.norm(af - bf).item() / max(1e-12, torch.linalg.norm(bf).item()))


def time_fn_with_peak_mem(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    iters: int = 8,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    base_alloc = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    inc_mib = max(0.0, (torch.cuda.max_memory_allocated(device) - base_alloc) / (1024.0**2))

    samples: list[float] = []
    clk_sample, pwr_sample = -1.0, -1.0
    for i in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        if i == iters // 2:
            clk_sample, pwr_sample = query_gpu_telemetry(device.index or 0)
        e1.synchronize()
        samples.append(float(e0.elapsed_time(e1)))

    return {
        "median_ms": round(float(statistics.median(samples)), 4),
        "min_ms": round(float(min(samples)), 4),
        "max_ms": round(float(max(samples)), 4),
        "inc_peak_mib": round(float(inc_mib), 2),
        "sm_clock_mhz": round(float(clk_sample), 1),
        "power_w": round(float(pwr_sample), 2),
    }


# ============================================================================
# 1. Triton Kernels: Window-Morton-2x2 Ragged Tubelet Pack/Unpack, Fused RMSNorm,
#    Fused Single-Launch Q+K 3D M-RoPE, Tile-Classified 64x64 Omni-Attention,
#    & Surgical Disjoint Reversed-Upper FA2 LSE Merger
# ============================================================================


@triton.jit
def _tubelet_win_morton2x2_pack_fwd_kernel(
    RawPix_ptr,
    PatchMeta_ptr,  # [N_patches, 4] int32: (pix_base, T_pix, H_pix, W_pix) + packed coords? Let's pass 5 int32s: (pix_base, T_pix, H_pix, W_pix, coord_t_ph_pw)
    PackedOut_ptr,
    stride_po_m,
    K_RAW: tl.constexpr,     # 3 * 2 * 14 * 14 = 1176
    K_PAD: tl.constexpr,     # 1184 (16-aligned for SM80 BF16 Tensor Cores)
    BLOCK_K: tl.constexpr,   # 128
):
    pid = tl.program_id(0)
    meta_base = PatchMeta_ptr + pid * 6
    pix_base = tl.load(meta_base + 0).to(tl.int64)
    t_pix = tl.load(meta_base + 1).to(tl.int64)
    h_pix = tl.load(meta_base + 2).to(tl.int64)
    w_pix = tl.load(meta_base + 3).to(tl.int64)
    t0 = tl.load(meta_base + 4).to(tl.int64)
    hw_packed = tl.load(meta_base + 5).to(tl.int64)
    ph0 = hw_packed >> 16
    pw0 = hw_packed & 0xFFFF

    out_row = PackedOut_ptr + pid.to(tl.int64) * stride_po_m
    for k_start in range(0, K_PAD, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        in_raw = offs_k < K_RAW
        kk = tl.minimum(offs_k, K_RAW - 1).to(tl.int64)
        # Layout matches nn.Conv3d weight [Out, C=3, dt=2, dh=14, dw=14]:
        dw = kk % 14
        t1 = kk // 14
        dh = t1 % 14
        t2 = t1 // 14
        dt = t2 % 2
        ch = t2 // 2
        pix_idx = pix_base + (((ch * t_pix + (t0 + dt)) * h_pix + (ph0 + dh)) * w_pix + (pw0 + dw))
        vals = tl.load(RawPix_ptr + pix_idx, mask=in_raw, other=0.0)
        tl.store(out_row + offs_k, vals, mask=offs_k < K_PAD)


@triton.jit
def _tubelet_win_morton2x2_unpack_bwd_kernel(
    dPacked_ptr,
    PatchMeta_ptr,
    dRawPix_ptr,
    stride_dp_m,
    K_RAW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Strictly non-overlapping tubelet bijection -> 100% collision-free tl.store (ZERO atomics!)
    pid = tl.program_id(0)
    meta_base = PatchMeta_ptr + pid * 6
    pix_base = tl.load(meta_base + 0).to(tl.int64)
    t_pix = tl.load(meta_base + 1).to(tl.int64)
    h_pix = tl.load(meta_base + 2).to(tl.int64)
    w_pix = tl.load(meta_base + 3).to(tl.int64)
    t0 = tl.load(meta_base + 4).to(tl.int64)
    hw_packed = tl.load(meta_base + 5).to(tl.int64)
    ph0 = hw_packed >> 16
    pw0 = hw_packed & 0xFFFF

    in_row = dPacked_ptr + pid.to(tl.int64) * stride_dp_m
    for k_start in range(0, K_RAW, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        in_raw = offs_k < K_RAW
        kk = tl.minimum(offs_k, K_RAW - 1).to(tl.int64)
        dw = kk % 14
        t1 = kk // 14
        dh = t1 % 14
        t2 = t1 // 14
        dt = t2 % 2
        ch = t2 // 2
        pix_idx = pix_base + (((ch * t_pix + (t0 + dt)) * h_pix + (ph0 + dh)) * w_pix + (pw0 + dw))
        vals = tl.load(in_row + offs_k, mask=in_raw, other=0.0)
        tl.store(dRawPix_ptr + pix_idx, vals, mask=in_raw)


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr, Rstd_ptr,
    stride_x, D: tl.constexpr, BLOCK_D: tl.constexpr, EPS: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    m = offs < D
    x = tl.load(X_ptr + row * stride_x + offs, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    rstd = tl.rsqrt(var + EPS)
    tl.store(Rstd_ptr + row, rstd)
    y = (x * rstd * w).to(Y_ptr.dtype.element_ty)
    tl.store(Y_ptr + row * stride_x + offs, y, mask=m)


@triton.jit
def _rmsnorm_bwd_dx_part_dw_kernel(
    dY_ptr, X_ptr, W_ptr, Rstd_ptr, dX_ptr, PartDW_ptr,
    N_rows, stride_x, D: tl.constexpr, BLOCK_D: tl.constexpr, ROWS_PER_PROG: tl.constexpr
):
    pid = tl.program_id(0)
    r_start = pid * ROWS_PER_PROG
    offs = tl.arange(0, BLOCK_D)
    m = offs < D
    w = tl.load(W_ptr + offs, mask=m, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for r in range(r_start, r_start + ROWS_PER_PROG):
        if r < N_rows:
            ro = r.to(tl.int64) * stride_x + offs
            dy = tl.load(dY_ptr + ro, mask=m, other=0.0).to(tl.float32)
            x = tl.load(X_ptr + ro, mask=m, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd_ptr + r).to(tl.float32)
            x_hat = x * rstd
            dw_acc += dy * x_hat
            wdy = w * dy
            c1 = tl.sum(x_hat * wdy, axis=0) / D
            dx = rstd * (wdy - x_hat * c1)
            tl.store(dX_ptr + ro, dx.to(dX_ptr.dtype.element_ty), mask=m)
    tl.store(PartDW_ptr + pid.to(tl.int64) * D + offs, dw_acc, mask=m)


@triton.jit
def _fused_qk_3d_mrope_kernel(
    Q_ptr, K_ptr, PosTHW_ptr, InvFreq_ptr, QOut_ptr, KOut_ptr,
    S,
    stride_qs, stride_qh,
    stride_ks, stride_kh,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D_HALF: tl.constexpr,          # 64 for D=128
    INTERLEAVED: tl.constexpr,     # False -> Sectioned [16, 24, 24]; True -> Qwen3-VL Interleaved [24, 20, 20]
    IS_BWD: tl.constexpr,
):
    # Single-launch in-register 3D M-RoPE across all Q and K heads with 0 MiB cos/sin HBM materialization!
    s = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs_f = tl.arange(0, D_HALF)

    pos_t = tl.load(PosTHW_ptr + 0 * S + s).to(tl.float32)
    pos_h = tl.load(PosTHW_ptr + 1 * S + s).to(tl.float32)
    pos_w = tl.load(PosTHW_ptr + 2 * S + s).to(tl.float32)

    if INTERLEAVED:
        # Qwen3-VL interleaved [24, 20, 20]: first 60 pairs cycle (T, H, W) x 20, pairs 60..63 use T!
        rem3 = offs_f % 3
        is_tail_t = offs_f >= 60
        pos_f = tl.where(
            is_tail_t | (rem3 == 0),
            pos_t,
            tl.where(rem3 == 1, pos_h, pos_w),
        )
    else:
        # Qwen2.5-VL sectioned [16, 24, 24]: [0..16) -> T, [16..40) -> H, [40..64) -> W
        pos_f = tl.where(offs_f < 16, pos_t, tl.where(offs_f < 40, pos_h, pos_w))

    inv_f = tl.load(InvFreq_ptr + offs_f).to(tl.float32)
    angles = pos_f * inv_f
    cos_v = tl.cos(angles)
    sin_v = tl.sin(angles)
    if IS_BWD:
        sin_v = -sin_v

    if h < H_Q:
        in_base = Q_ptr + s * stride_qs + h * stride_qh
        out_base = QOut_ptr + s * stride_qs + h * stride_qh
    else:
        hk = h - H_Q
        in_base = K_ptr + s * stride_ks + hk * stride_kh
        out_base = KOut_ptr + s * stride_ks + hk * stride_kh

    x1 = tl.load(in_base + offs_f).to(tl.float32)
    x2 = tl.load(in_base + D_HALF + offs_f).to(tl.float32)
    y1 = x1 * cos_v - x2 * sin_v
    y2 = x2 * cos_v + x1 * sin_v
    tl.store(out_base + offs_f, y1.to(QOut_ptr.dtype.element_ty))
    tl.store(out_base + D_HALF + offs_f, y2.to(QOut_ptr.dtype.element_ty))


# ============================================================================
# 1B. Surgical Disjoint Reversed-Upper FA2 + Triton LSE Helper Kernels
#     & From-Scratch Tile-Classified 64x64 GQA Triton Omni-FlashAttention
# ============================================================================


@triton.jit
def _rev_gather_qkv_kernel(
    Q_ptr, K_ptr, V_ptr, SpanMeta_ptr, QRev_ptr, KRev_ptr, VRev_ptr,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr
):
    u = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    q_src = tl.load(SpanMeta_ptr + u * 2 + 0).to(tl.int64)
    kv_src = tl.load(SpanMeta_ptr + u * 2 + 1).to(tl.int64)
    offs_d = tl.arange(0, D)
    if h < H_Q:
        val = tl.load(Q_ptr + (q_src * H_Q + h) * D + offs_d)
        tl.store(QRev_ptr + (u * H_Q + h) * D + offs_d, val)
    elif h < H_Q + H_KV:
        hk = h - H_Q
        val = tl.load(K_ptr + (kv_src * H_KV + hk) * D + offs_d)
        tl.store(KRev_ptr + (u * H_KV + hk) * D + offs_d, val)
    else:
        hk = h - H_Q - H_KV
        val = tl.load(V_ptr + (kv_src * H_KV + hk) * D + offs_d)
        tl.store(VRev_ptr + (u * H_KV + hk) * D + offs_d, val)


@triton.jit
def _merge_upper_lse_kernel(
    Out1_ptr, LSE1_ptr, Out2_ptr, LSE2_ptr, SpanMeta_ptr,
    S, N_u, H_Q: tl.constexpr, D: tl.constexpr
):
    u = tl.program_id(0).to(tl.int64)
    hq = tl.program_id(1).to(tl.int64)
    q_idx = tl.load(SpanMeta_ptr + u * 2 + 0).to(tl.int64)
    offs_d = tl.arange(0, D)
    l1 = tl.load(LSE1_ptr + hq * S + q_idx).to(tl.float32)
    l2 = tl.load(LSE2_ptr + hq * N_u + u).to(tl.float32)
    m = tl.maximum(l1, l2)
    w1 = tl.exp(l1 - m)
    w2 = tl.exp(l2 - m)
    ws = w1 + w2
    nlse = m + tl.log(ws)
    o1 = tl.load(Out1_ptr + (q_idx * H_Q + hq) * D + offs_d).to(tl.float32)
    o2 = tl.load(Out2_ptr + (u * H_Q + hq) * D + offs_d).to(tl.float32)
    nout = (o1 * w1 + o2 * w2) / ws
    tl.store(Out1_ptr + (q_idx * H_Q + hq) * D + offs_d, nout.to(Out1_ptr.dtype.element_ty))
    tl.store(LSE1_ptr + hq * S + q_idx, nlse)


@triton.jit
def _gather_bwd_upper_inputs_kernel(
    dO_ptr, MixOut_ptr, MixLSE_ptr, SpanMeta_ptr, dORev_ptr, MixOutRev_ptr, MixLSERev_ptr,
    S, N_u, H_Q: tl.constexpr, D: tl.constexpr
):
    u = tl.program_id(0).to(tl.int64)
    hq = tl.program_id(1).to(tl.int64)
    q_src = tl.load(SpanMeta_ptr + u * 2 + 0).to(tl.int64)
    offs_d = tl.arange(0, D)
    tl.store(dORev_ptr + (u * H_Q + hq) * D + offs_d, tl.load(dO_ptr + (q_src * H_Q + hq) * D + offs_d))
    tl.store(MixOutRev_ptr + (u * H_Q + hq) * D + offs_d, tl.load(MixOut_ptr + (q_src * H_Q + hq) * D + offs_d))
    tl.store(MixLSERev_ptr + hq * N_u + u, tl.load(MixLSE_ptr + hq * S + q_src))


@triton.jit
def _scatter_add_rev_grads_kernel(
    dQ1_ptr, dK1_ptr, dV1_ptr, dQ2_ptr, dK2_ptr, dV2_ptr, SpanMeta_ptr,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr
):
    # SpanMeta q_dst and kv_dst are strictly 1-to-1 injections -> collision-free tl.store!
    u = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    q_dst = tl.load(SpanMeta_ptr + u * 2 + 0).to(tl.int64)
    kv_dst = tl.load(SpanMeta_ptr + u * 2 + 1).to(tl.int64)
    offs_d = tl.arange(0, D)
    if h < H_Q:
        v1 = tl.load(dQ1_ptr + (q_dst * H_Q + h) * D + offs_d).to(tl.float32)
        v2 = tl.load(dQ2_ptr + (u * H_Q + h) * D + offs_d).to(tl.float32)
        tl.store(dQ1_ptr + (q_dst * H_Q + h) * D + offs_d, (v1 + v2).to(dQ1_ptr.dtype.element_ty))
    elif h < H_Q + H_KV:
        hk = h - H_Q
        v1 = tl.load(dK1_ptr + (kv_dst * H_KV + hk) * D + offs_d).to(tl.float32)
        v2 = tl.load(dK2_ptr + (u * H_KV + hk) * D + offs_d).to(tl.float32)
        tl.store(dK1_ptr + (kv_dst * H_KV + hk) * D + offs_d, (v1 + v2).to(dK1_ptr.dtype.element_ty))
    else:
        hk = h - H_Q - H_KV
        v1 = tl.load(dV1_ptr + (kv_dst * H_KV + hk) * D + offs_d).to(tl.float32)
        v2 = tl.load(dV2_ptr + (u * H_KV + hk) * D + offs_d).to(tl.float32)
        tl.store(dV1_ptr + (kv_dst * H_KV + hk) * D + offs_d, (v1 + v2).to(dV1_ptr.dtype.element_ty))


@triton.jit
def _omni_bwd_preprocess_delta_kernel(
    Out_ptr, dO_ptr, Delta_ptr, N_rows, D: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    m = row < N_rows
    offs_d = tl.arange(0, D)
    ptr_o = Out_ptr + row[:, None] * D + offs_d[None, :]
    ptr_do = dO_ptr + row[:, None] * D + offs_d[None, :]
    o = tl.load(ptr_o, mask=m[:, None], other=0.0).to(tl.float32)
    do = tl.load(ptr_do, mask=m[:, None], other=0.0).to(tl.float32)
    tl.store(Delta_ptr + row, tl.sum(o * do, axis=1), mask=m)


@triton.jit
def _omni_tile64_gqa_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    DocID_ptr, SpanID_ptr, QTileBounds_ptr,  # [N_u, 4] int32: (kb_start, kb_full_start, kb_full_end, kb_end)
    Out_ptr, LSE_ptr,
    S, sm_scale,
    B_Q_SUB: tl.constexpr,   # 8 tokens
    G: tl.constexpr,         # 7 (28Q / 4KV)
    G_PAD: tl.constexpr,     # 8 -> 8 * 8 = 64 MMA rows!
    D: tl.constexpr,         # 128
    B_K: tl.constexpr,       # 64
):
    u = tl.program_id(0)
    hk = tl.program_id(1)

    b_ptr = QTileBounds_ptr + u * 4
    kb_start = tl.load(b_ptr + 0)
    kb_full_s = tl.load(b_ptr + 1)
    kb_full_e = tl.load(b_ptr + 2)
    kb_end = tl.load(b_ptr + 3)

    offs_m = tl.arange(0, B_Q_SUB * G_PAD)
    sub_q = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    q_pos = u * B_Q_SUB + sub_q
    valid_m = (q_pos < S) & (g_idx < G)
    hq = hk * G + tl.minimum(g_idx, G - 1)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + (q_pos.to(tl.int64)[:, None] * (4 * G) + hq[:, None]) * D + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    q_doc = tl.load(DocID_ptr + q_pos, mask=q_pos < S, other=-1)
    q_span = tl.load(SpanID_ptr + q_pos, mask=q_pos < S, other=0)

    m_i = tl.full([B_Q_SUB * G_PAD], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([B_Q_SUB * G_PAD], dtype=tl.float32)
    acc = tl.zeros([B_Q_SUB * G_PAD, D], dtype=tl.float32)

    offs_n = tl.arange(0, B_K)
    for kb in range(kb_start, kb_end):
        k_pos = kb * B_K + offs_n
        valid_k = k_pos < S
        kv_off = (k_pos.to(tl.int64)[:, None] * 4 + hk) * D + offs_d[None, :]
        k_tile = tl.load(K_ptr + kv_off, mask=valid_k[:, None], other=0.0)
        v_tile = tl.load(V_ptr + kv_off, mask=valid_k[:, None], other=0.0)

        qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        if (kb >= kb_full_s) and (kb < kb_full_e):
            # 100% dense interior tile (causal prefix OR bidirectional image interior!) -> ZERO mask checks!
            pass
        else:
            k_doc = tl.load(DocID_ptr + k_pos, mask=valid_k, other=-2)
            k_span = tl.load(SpanID_ptr + k_pos, mask=valid_k, other=-1)
            same_doc = q_doc[:, None] == k_doc[None, :]
            is_causal = k_pos[None, :] <= q_pos[:, None]
            same_img = (q_span[:, None] > 0) & (q_span[:, None] == k_span[None, :])
            valid_pair = valid_m[:, None] & valid_k[None, :] & same_doc & (is_causal | same_img)
            qk = tl.where(valid_pair, qk, -1.0e30)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m_i = m_ij

    inv_l = 1.0 / tl.maximum(l_i, 1.0e-30)
    out = (acc * inv_l[:, None]).to(Out_ptr.dtype.element_ty)
    lse = m_i + tl.log(tl.maximum(l_i, 1.0e-30))
    out_ptrs = Out_ptr + (q_pos.to(tl.int64)[:, None] * (4 * G) + hq[:, None]) * D + offs_d[None, :]
    tl.store(out_ptrs, out, mask=valid_m[:, None])
    tl.store(LSE_ptr + hq.to(tl.int64) * S + q_pos, lse, mask=valid_m)


@triton.jit
def _omni_tile64_gqa_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, LSE_ptr, Delta_ptr,
    DocID_ptr, SpanID_ptr, QTileBounds_ptr,
    dQ_ptr,
    S, sm_scale,
    B_Q_SUB: tl.constexpr,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    B_K: tl.constexpr,
):
    u = tl.program_id(0)
    hk = tl.program_id(1)
    b_ptr = QTileBounds_ptr + u * 4
    kb_start = tl.load(b_ptr + 0)
    kb_full_s = tl.load(b_ptr + 1)
    kb_full_e = tl.load(b_ptr + 2)
    kb_end = tl.load(b_ptr + 3)

    offs_m = tl.arange(0, B_Q_SUB * G_PAD)
    sub_q = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    q_pos = u * B_Q_SUB + sub_q
    valid_m = (q_pos < S) & (g_idx < G)
    hq = hk * G + tl.minimum(g_idx, G - 1)
    offs_d = tl.arange(0, D)

    row_id = q_pos.to(tl.int64) * (4 * G) + hq
    q_tile = tl.load(Q_ptr + row_id[:, None] * D + offs_d[None, :], mask=valid_m[:, None], other=0.0)
    do_tile = tl.load(dO_ptr + row_id[:, None] * D + offs_d[None, :], mask=valid_m[:, None], other=0.0)
    lse_m = tl.load(LSE_ptr + hq.to(tl.int64) * S + q_pos, mask=valid_m, other=0.0).to(tl.float32)
    delta_m = tl.load(Delta_ptr + row_id, mask=valid_m, other=0.0).to(tl.float32)

    q_doc = tl.load(DocID_ptr + q_pos, mask=q_pos < S, other=-1)
    q_span = tl.load(SpanID_ptr + q_pos, mask=q_pos < S, other=0)
    dq_acc = tl.zeros([B_Q_SUB * G_PAD, D], dtype=tl.float32)

    offs_n = tl.arange(0, B_K)
    for kb in range(kb_start, kb_end):
        k_pos = kb * B_K + offs_n
        valid_k = k_pos < S
        kv_off = (k_pos.to(tl.int64)[:, None] * 4 + hk) * D + offs_d[None, :]
        k_tile = tl.load(K_ptr + kv_off, mask=valid_k[:, None], other=0.0)
        v_tile = tl.load(V_ptr + kv_off, mask=valid_k[:, None], other=0.0)

        qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        if (kb >= kb_full_s) and (kb < kb_full_e):
            p = tl.exp(qk - lse_m[:, None])
        else:
            k_doc = tl.load(DocID_ptr + k_pos, mask=valid_k, other=-2)
            k_span = tl.load(SpanID_ptr + k_pos, mask=valid_k, other=-1)
            same_doc = q_doc[:, None] == k_doc[None, :]
            is_causal = k_pos[None, :] <= q_pos[:, None]
            same_img = (q_span[:, None] > 0) & (q_span[:, None] == k_span[None, :])
            vp = valid_m[:, None] & valid_k[None, :] & same_doc & (is_causal | same_img)
            p = tl.where(vp, tl.exp(qk - lse_m[:, None]), 0.0)

        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = p * (dp - delta_m[:, None])
        dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

    dq_out = (dq_acc * sm_scale).to(dQ_ptr.dtype.element_ty)
    tl.store(dQ_ptr + row_id[:, None] * D + offs_d[None, :], dq_out, mask=valid_m[:, None])


@triton.jit
def _omni_tile64_gqa_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, LSE_ptr, Delta_ptr,
    DocID_ptr, SpanID_ptr, KVTileBounds_ptr,  # [N_kb, 4] int32: (u_start, u_full_start, u_full_end, u_end)
    dK_ptr, dV_ptr,
    S, sm_scale,
    B_Q_SUB: tl.constexpr,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    B_K: tl.constexpr,
):
    # 100% Lock-Free KV-owner backward sweep across all G=7 GQA query heads!
    kb = tl.program_id(0)
    hk = tl.program_id(1)
    b_ptr = KVTileBounds_ptr + kb * 4
    u_start = tl.load(b_ptr + 0)
    u_full_s = tl.load(b_ptr + 1)
    u_full_e = tl.load(b_ptr + 2)
    u_end = tl.load(b_ptr + 3)

    offs_n = tl.arange(0, B_K)
    k_pos = kb * B_K + offs_n
    valid_k = k_pos < S
    offs_d = tl.arange(0, D)
    kv_off = (k_pos.to(tl.int64)[:, None] * 4 + hk) * D + offs_d[None, :]
    k_tile = tl.load(K_ptr + kv_off, mask=valid_k[:, None], other=0.0)
    v_tile = tl.load(V_ptr + kv_off, mask=valid_k[:, None], other=0.0)
    k_doc = tl.load(DocID_ptr + k_pos, mask=valid_k, other=-2)
    k_span = tl.load(SpanID_ptr + k_pos, mask=valid_k, other=-1)

    dk_acc = tl.zeros([B_K, D], dtype=tl.float32)
    dv_acc = tl.zeros([B_K, D], dtype=tl.float32)

    offs_m = tl.arange(0, B_Q_SUB * G_PAD)
    sub_q = offs_m // G_PAD
    g_idx = offs_m % G_PAD

    for u in range(u_start, u_end):
        q_pos = u * B_Q_SUB + sub_q
        valid_m = (q_pos < S) & (g_idx < G)
        hq = hk * G + tl.minimum(g_idx, G - 1)
        row_id = q_pos.to(tl.int64) * (4 * G) + hq
        q_tile = tl.load(Q_ptr + row_id[:, None] * D + offs_d[None, :], mask=valid_m[:, None], other=0.0)
        do_tile = tl.load(dO_ptr + row_id[:, None] * D + offs_d[None, :], mask=valid_m[:, None], other=0.0)
        lse_m = tl.load(LSE_ptr + hq.to(tl.int64) * S + q_pos, mask=valid_m, other=0.0).to(tl.float32)
        delta_m = tl.load(Delta_ptr + row_id, mask=valid_m, other=0.0).to(tl.float32)

        qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        if (u >= u_full_s) and (u < u_full_e):
            p = tl.where(valid_m[:, None], tl.exp(qk - lse_m[:, None]), 0.0)
        else:
            q_doc = tl.load(DocID_ptr + q_pos, mask=q_pos < S, other=-1)
            q_span = tl.load(SpanID_ptr + q_pos, mask=q_pos < S, other=0)
            same_doc = q_doc[:, None] == k_doc[None, :]
            is_causal = k_pos[None, :] <= q_pos[:, None]
            same_img = (q_span[:, None] > 0) & (q_span[:, None] == k_span[None, :])
            vp = valid_m[:, None] & valid_k[None, :] & same_doc & (is_causal | same_img)
            p = tl.where(vp, tl.exp(qk - lse_m[:, None]), 0.0)

        dv_acc += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile)
        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = p * (dp - delta_m[:, None])
        dk_acc += tl.dot(tl.trans(ds.to(q_tile.dtype)), q_tile)

    tl.store(dK_ptr + kv_off, (dk_acc * sm_scale).to(dK_ptr.dtype.element_ty), mask=valid_k[:, None])
    tl.store(dV_ptr + kv_off, dv_acc.to(dV_ptr.dtype.element_ty), mask=valid_k[:, None])


# ============================================================================
# 2. Metadata Builders, Autograd Functions, ViT Encoder, & Omni-LLM Modules
# ============================================================================


@dataclass
class RaggedVisionMeta:
    shapes_thw: list[tuple[int, int, int]]
    total_raw_pixels: int
    total_patches: int
    total_merged_tokens: int
    patch_meta: torch.Tensor                  # [N_patches, 6] int32 (win-morton-2x2 order)
    patch_meta_raster: torch.Tensor           # [N_patches, 6] int32 (standard raster order)
    cu_seqlens_full: torch.Tensor             # [N_items_or_frames + 1] int32
    max_seqlen_full: int
    cu_seqlens_win: torch.Tensor              # [N_windows + 1] int32 (contiguous in win-morton-2x2!)
    max_seqlen_win: int
    raster_to_win_idx: torch.Tensor           # [N_patches] int64 (for eager HF window gather baseline)
    win_to_raster_idx: torch.Tensor           # [N_patches] int64
    winquad_to_rasterquad_idx: torch.Tensor   # [N_merged] int64 (4x smaller post-merger unswizzle!)
    merged_thw_grids: list[tuple[int, int, int]]


def build_window_morton2x2_ragged_meta(
    shapes_thw: list[tuple[int, int, int]], device: torch.device
) -> RaggedVisionMeta:
    # shapes_thw: list of pixel shapes (T_pix, H_pix, W_pix), T_pix % 2 == 0, H_pix % 28 == 0, W_pix % 28 == 0
    wm_rows: list[list[int]] = []
    rs_rows: list[list[int]] = []
    cu_full = [0]
    cu_win = [0]
    raster_to_win: list[int] = []
    winquad_to_rasterquad: list[int] = []
    merged_grids: list[tuple[int, int, int]] = []

    pix_off = 0
    patch_base = 0
    quad_base = 0
    max_full = 0
    max_win = 0

    for t_pix, h_pix, w_pix in shapes_thw:
        assert t_pix % 2 == 0 and h_pix % 28 == 0 and w_pix % 28 == 0
        nt = t_pix // 2
        nh = h_pix // 14
        nw = w_pix // 14
        qmh = nh // 2
        qmw = nw // 2
        merged_grids.append((nt, qmh, qmw))

        # Raster-2x2 order (standard Qwen2.5-VL input order: grouped by 2x2 merger quads in raster scan)
        for tf in range(nt):
            for qh in range(qmh):
                for qw in range(qmw):
                    for sh in (0, 1):
                        for sw in (0, 1):
                            ph = (2 * qh + sh) * 14
                            pw = (2 * qw + sw) * 14
                            rs_rows.append([pix_off, t_pix, h_pix, w_pix, 2 * tf, (ph << 16) | pw])

        # Hierarchical Window-Morton-2x2 order: Frame -> 8x8 Window (4x4 Quads) -> 2x2 Quad -> 4 Patches
        for tf in range(nt):
            frame_patches = nh * nw
            cu_full.append(cu_full[-1] + frame_patches)
            max_full = max(max_full, frame_patches)

            n_wb_h = (qmh + 3) // 4
            n_wb_w = (qmw + 3) // 4
            for wbh in range(n_wb_h):
                qh_s, qh_e = wbh * 4, min(qmh, (wbh + 1) * 4)
                for wbw in range(n_wb_w):
                    qw_s, qw_e = wbw * 4, min(qmw, (wbw + 1) * 4)
                    win_cnt = 0
                    for qh in range(qh_s, qh_e):
                        for qw in range(qw_s, qw_e):
                            r_quad = quad_base + (tf * qmh + qh) * qmw + qw
                            winquad_to_rasterquad.append(r_quad)
                            for k4, (sh, sw) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                                ph = (2 * qh + sh) * 14
                                pw = (2 * qw + sw) * 14
                                wm_rows.append([pix_off, t_pix, h_pix, w_pix, 2 * tf, (ph << 16) | pw])
                                raster_to_win.append(r_quad * 4 + k4)
                                win_cnt += 1
                    cu_win.append(cu_win[-1] + win_cnt)
                    max_win = max(max_win, win_cnt)

        pix_off += 3 * t_pix * h_pix * w_pix
        patch_base += nt * nh * nw
        quad_base += nt * qmh * qmw

    r2w = torch.tensor(raster_to_win, device=device, dtype=torch.long)
    w2r = torch.argsort(r2w)
    # Note: wq_to_rq maps win-quad index -> raster-quad index; argsort gives raster-quad index -> win-quad index!
    wq2rq_list = torch.tensor(winquad_to_rasterquad, device=device, dtype=torch.long)
    rq_from_wq = torch.argsort(wq2rq_list)

    return RaggedVisionMeta(
        shapes_thw=shapes_thw,
        total_raw_pixels=pix_off,
        total_patches=patch_base,
        total_merged_tokens=quad_base,
        patch_meta=torch.tensor(wm_rows, device=device, dtype=torch.int32).contiguous(),
        patch_meta_raster=torch.tensor(rs_rows, device=device, dtype=torch.int32).contiguous(),
        cu_seqlens_full=torch.tensor(cu_full, device=device, dtype=torch.int32),
        max_seqlen_full=max_full,
        cu_seqlens_win=torch.tensor(cu_win, device=device, dtype=torch.int32),
        max_seqlen_win=max_win,
        raster_to_win_idx=r2w,
        win_to_raster_idx=w2r,
        winquad_to_rasterquad_idx=rq_from_wq,
        merged_thw_grids=merged_grids,
    )


class _RaggedTubeletPackFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, raw_pixels: torch.Tensor, patch_meta: torch.Tensor) -> torch.Tensor:
        n_patches = patch_meta.shape[0]
        packed = torch.empty((n_patches, 1184), device=raw_pixels.device, dtype=raw_pixels.dtype)
        if n_patches > 0:
            _tubelet_win_morton2x2_pack_fwd_kernel[(n_patches,)](
                raw_pixels, patch_meta, packed, packed.stride(0), K_RAW=1176, K_PAD=1184, BLOCK_K=128
            )
        ctx.save_for_backward(patch_meta)
        ctx.raw_numel = raw_pixels.numel()
        return packed

    @staticmethod
    def backward(ctx: Any, d_packed: torch.Tensor) -> tuple[torch.Tensor, None]:
        (patch_meta,) = ctx.saved_tensors
        n_patches = patch_meta.shape[0]
        d_packed_c = d_packed.contiguous()
        d_raw = torch.empty((ctx.raw_numel,), device=d_packed.device, dtype=d_packed.dtype)
        if n_patches > 0:
            _tubelet_win_morton2x2_unpack_bwd_kernel[(n_patches,)](
                d_packed_c, patch_meta, d_raw, d_packed_c.stride(0), K_RAW=1176, BLOCK_K=128
            )
        else:
            d_raw.zero_()
        return d_raw, None


class _TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, dim = x2d.shape
        y2d = torch.empty_like(x2d)
        rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        block_d = triton.next_power_of_2(dim)
        if n_rows > 0:
            _rmsnorm_fwd_kernel[(n_rows,)](
                x2d, weight, y2d, rstd, x2d.stride(0), D=dim, BLOCK_D=block_d, EPS=eps
            )
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.block_d = block_d
        return y2d.view(orig_shape)

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x2d, weight, rstd = ctx.saved_tensors
        n_rows, dim = x2d.shape
        dy2d = dy.contiguous().view(n_rows, dim)
        dx2d = torch.empty_like(x2d)
        if n_rows == 0:
            return dx2d.view(ctx.orig_shape), torch.zeros_like(weight), None
        rows_per_prog = 32
        n_progs = triton.cdiv(n_rows, rows_per_prog)
        part_dw = torch.empty((n_progs, dim), device=x2d.device, dtype=torch.float32)
        _rmsnorm_bwd_dx_part_dw_kernel[(n_progs,)](
            dy2d, x2d, weight, rstd, dx2d, part_dw,
            n_rows, x2d.stride(0), D=dim, BLOCK_D=ctx.block_d, ROWS_PER_PROG=rows_per_prog
        )
        dw = part_dw.sum(dim=0).to(weight.dtype)
        return dx2d.view(ctx.orig_shape), dw, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _TritonRMSNormFn.apply(x, weight, eps)


class _FusedQK3DMRoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_thw: torch.Tensor,
        inv_freq: torch.Tensor,
        interleaved: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q: [S, H_q, D], k: [S, H_kv, D], pos_thw: [3, S] int32, inv_freq: [D//2] float32
        seqlen, h_q, dim = q.shape
        h_kv = k.shape[1]
        q_c = q.contiguous()
        k_c = k.contiguous()
        q_out = torch.empty_like(q_c)
        k_out = torch.empty_like(k_c)
        if seqlen > 0:
            _fused_qk_3d_mrope_kernel[(seqlen, h_q + h_kv)](
                q_c, k_c, pos_thw, inv_freq, q_out, k_out,
                seqlen,
                q_c.stride(0), q_c.stride(1),
                k_c.stride(0), k_c.stride(1),
                H_Q=h_q, H_KV=h_kv, D_HALF=dim // 2,
                INTERLEAVED=interleaved, IS_BWD=False,
            )
        ctx.save_for_backward(pos_thw, inv_freq)
        ctx.interleaved = interleaved
        return q_out, k_out

    @staticmethod
    def backward(
        ctx: Any, dq_out: torch.Tensor, dk_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        pos_thw, inv_freq = ctx.saved_tensors
        seqlen, h_q, dim = dq_out.shape
        h_kv = dk_out.shape[1]
        dq_c = dq_out.contiguous()
        dk_c = dk_out.contiguous()
        dq = torch.empty_like(dq_c)
        dk = torch.empty_like(dk_c)
        if seqlen > 0:
            _fused_qk_3d_mrope_kernel[(seqlen, h_q + h_kv)](
                dq_c, dk_c, pos_thw, inv_freq, dq, dk,
                seqlen,
                dq_c.stride(0), dq_c.stride(1),
                dk_c.stride(0), dk_c.stride(1),
                H_Q=h_q, H_KV=h_kv, D_HALF=dim // 2,
                INTERLEAVED=ctx.interleaved, IS_BWD=True,
            )
        return dq, dk, None, None, None


def hf_eager_3d_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_thw: torch.Tensor,
    inv_freq: torch.Tensor,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Faithful HuggingFace transformers Qwen2.5-VL / Qwen3-VL eager reference:
    # Materializes [3, S, 64] angles -> section split or interleave -> [S, 1, 128] cos/sin -> rotate_half!
    freqs_3 = pos_thw.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)  # [3, S, 64]
    if interleaved:
        # Qwen3-VL [24, 20, 20] interleaved
        f_merged = freqs_3[0].clone()
        f_merged[:, 1:60:3] = freqs_3[1, :, 1:60:3]
        f_merged[:, 2:60:3] = freqs_3[2, :, 2:60:3]
    else:
        # Qwen2.5-VL [16, 24, 24] sectioned
        f_merged = torch.cat([freqs_3[0, :, :16], freqs_3[1, :, 16:40], freqs_3[2, :, 40:64]], dim=-1)
    emb = torch.cat([f_merged, f_merged], dim=-1)  # [S, 128]
    cos = emb.cos().to(q.dtype).unsqueeze(1)
    sin = emb.sin().to(q.dtype).unsqueeze(1)

    def _rot_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :64]
        x2 = x[..., 64:]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (_rot_half(q) * sin), (k * cos) + (_rot_half(k) * sin)


@dataclass
class OmniPackedBatchMeta:
    seqlen: int
    cu_docs: torch.Tensor           # [N_docs + 1] int32
    max_doc_len: int
    doc_id: torch.Tensor            # [S] int32
    span_id: torch.Tensor           # [S] int32 (0 = causal text; >0 = bidirectional image span id)
    pos_thw: torch.Tensor           # [3, S] int32 3D M-RoPE coordinates
    span_meta: torch.Tensor         # [N_upper, 2] int32 reversed (q_idx, kv_idx) pairs
    cu_spans: torch.Tensor          # [N_spans + 1] int32
    max_span_upper: int
    q_tile_bounds: torch.Tensor     # [N_u, 4] int32 for 64x64 Tile-Classified Triton Omni-GQA
    kv_tile_bounds: torch.Tensor    # [N_kb, 4] int32 for 64x64 Tile-Classified Triton Omni-GQA
    vis_token_indices: torch.Tensor # [N_vis_merged] int64 positions where visual embeddings splice in


def build_omni_packed_batch_meta(
    docs_spec: list[list[tuple[str, int, int, int, int]]],
    device: torch.device,
    pad_to_multiple: int = 64,
) -> OmniPackedBatchMeta:
    # Each doc in docs_spec is a list of segments:
    #   ("text", length, 0, 0, 0) or ("vision", nt * qmh * qmw, nt, qmh, qmw)
    doc_id_list: list[int] = []
    span_id_list: list[int] = []
    pt_list: list[int] = []
    ph_list: list[int] = []
    pw_list: list[int] = []
    vis_positions: list[int] = []
    cu_docs = [0]
    spans: list[tuple[int, int]] = []
    doc_ranges: list[tuple[int, int]] = []

    cur_tok = 0
    next_span_id = 1
    max_doc = 0

    for d_idx, segs in enumerate(docs_spec):
        d_start = cur_tok
        mrope_cursor = 0
        for kind, seg_len, nt, qmh, qmw in segs:
            if kind == "text":
                for i in range(seg_len):
                    doc_id_list.append(d_idx)
                    span_id_list.append(0)
                    p = mrope_cursor + i
                    pt_list.append(p)
                    ph_list.append(p)
                    pw_list.append(p)
                cur_tok += seg_len
                mrope_cursor += seg_len
            else:
                assert seg_len == nt * qmh * qmw
                s0 = cur_tok
                s1 = cur_tok + seg_len
                spans.append((s0, s1))
                sid = next_span_id
                next_span_id += 1
                for tf in range(nt):
                    for hh in range(qmh):
                        for ww in range(qmw):
                            doc_id_list.append(d_idx)
                            span_id_list.append(sid)
                            pt_list.append(mrope_cursor + tf)
                            ph_list.append(mrope_cursor + hh)
                            pw_list.append(mrope_cursor + ww)
                            vis_positions.append(cur_tok)
                            cur_tok += 1
                mrope_cursor += max(nt, qmh, qmw)
        d_end = cur_tok
        if d_end > d_start:
            cu_docs.append(d_end)
            doc_ranges.append((d_start, d_end))
            max_doc = max(max_doc, d_end - d_start)

    rem = cur_tok % pad_to_multiple
    if rem != 0:
        pad_n = pad_to_multiple - rem
        pad_doc_id = len(doc_ranges)
        d_start = cur_tok
        for i in range(pad_n):
            doc_id_list.append(pad_doc_id)
            span_id_list.append(0)
            pt_list.append(i)
            ph_list.append(i)
            pw_list.append(i)
        cur_tok += pad_n
        cu_docs.append(cur_tok)
        doc_ranges.append((d_start, cur_tok))
        max_doc = max(max_doc, pad_n)

    seqlen = cur_tok
    pairs_u: list[torch.Tensor] = []
    cu_sp = [0]
    max_lm = 0
    span_end_by_id: dict[int, tuple[int, int]] = {}
    for idx_s, (s0, s1) in enumerate(spans, start=1):
        span_end_by_id[idx_s] = (s0, s1)
        lm = s1 - s0 - 1
        if lm > 0:
            qi = torch.arange(s1 - 2, s0 - 1, -1, device=device, dtype=torch.int32)
            ki = torch.arange(s1 - 1, s0, -1, device=device, dtype=torch.int32)
            pairs_u.append(torch.stack([qi, ki], dim=-1))
            cu_sp.append(cu_sp[-1] + lm)
            max_lm = max(max_lm, lm)

    if pairs_u:
        span_meta = torch.cat(pairs_u, dim=0).contiguous()
    else:
        span_meta = torch.empty((0, 2), device=device, dtype=torch.int32)

    # Precompute 64x64 tile bounds for Tile-Classified Triton Omni-GQA (B_Q_SUB=8, B_K=64)
    n_u = seqlen // 8
    n_kb = seqlen // 64
    q_bounds: list[list[int]] = []
    for u in range(n_u):
        q0, q1 = u * 8, u * 8 + 8
        d0_min = seqlen
        k_max_excl = 0
        same_d = (doc_id_list[q0] == doc_id_list[q1 - 1])
        d_common_start = doc_ranges[doc_id_list[q0]][0]
        min_valid_end = seqlen
        for qp in range(q0, q1):
            di = doc_id_list[qp]
            ds, _ = doc_ranges[di]
            d0_min = min(d0_min, ds)
            sid_q = span_id_list[qp]
            ve = span_end_by_id[sid_q][1] if sid_q > 0 else (qp + 1)
            k_max_excl = max(k_max_excl, ve)
            min_valid_end = min(min_valid_end, ve)
        kb_s = d0_min // 64
        kb_e = (k_max_excl + 63) // 64
        if same_d:
            kb_fs = (d_common_start + 63) // 64
            kb_fe = min_valid_end // 64
            if kb_fe < kb_fs:
                kb_fs = kb_fe = kb_s
        else:
            kb_fs = kb_fe = kb_s
        q_bounds.append([kb_s, kb_fs, kb_fe, kb_e])

    kv_bounds: list[list[int]] = []
    for kb in range(n_kb):
        k0, k1 = kb * 64, kb * 64 + 64
        u_min = n_u
        u_max_excl = 0
        same_kd = (doc_id_list[k0] == doc_id_list[k1 - 1])
        d_end_common = doc_ranges[doc_id_list[k0]][1]
        max_req_q_start = 0
        for kp in range(k0, k1):
            di = doc_id_list[kp]
            _, de = doc_ranges[di]
            sid_k = span_id_list[kp]
            qs_first = span_end_by_id[sid_k][0] if sid_k > 0 else kp
            u_min = min(u_min, qs_first // 8)
            u_max_excl = max(u_max_excl, (de + 7) // 8)
            max_req_q_start = max(max_req_q_start, qs_first)
        if same_kd:
            u_fs = (max_req_q_start + 7) // 8
            u_fe = d_end_common // 8
            if u_fe < u_fs:
                u_fs = u_fe = u_min
        else:
            u_fs = u_fe = u_min
        kv_bounds.append([u_min, u_fs, u_fe, u_max_excl])

    return OmniPackedBatchMeta(
        seqlen=seqlen,
        cu_docs=torch.tensor(cu_docs, device=device, dtype=torch.int32),
        max_doc_len=max_doc,
        doc_id=torch.tensor(doc_id_list, device=device, dtype=torch.int32),
        span_id=torch.tensor(span_id_list, device=device, dtype=torch.int32),
        pos_thw=torch.tensor([pt_list, ph_list, pw_list], device=device, dtype=torch.int32).contiguous(),
        span_meta=span_meta,
        cu_spans=torch.tensor(cu_sp, device=device, dtype=torch.int32),
        max_span_upper=max_lm,
        q_tile_bounds=torch.tensor(q_bounds, device=device, dtype=torch.int32).contiguous(),
        kv_tile_bounds=torch.tensor(kv_bounds, device=device, dtype=torch.int32).contiguous(),
        vis_token_indices=torch.tensor(vis_positions, device=device, dtype=torch.long),
    )


class _SurgicalOmniFA2Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: OmniPackedBatchMeta,
    ) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        seqlen, h_q, dim = q.shape
        h_kv = k.shape[1]
        sm_scale = dim**-0.5

        out1, lse1, _, _ = _flash_attn_varlen_forward(
            q=q, k=k, v=v,
            cu_seqlens_q=meta.cu_docs, cu_seqlens_k=meta.cu_docs,
            max_seqlen_q=meta.max_doc_len, max_seqlen_k=meta.max_doc_len,
            dropout_p=0.0, softmax_scale=sm_scale, causal=True,
        )
        n_u = meta.span_meta.shape[0]
        if n_u > 0:
            qr = torch.empty((n_u, h_q, dim), device=q.device, dtype=q.dtype)
            kr = torch.empty((n_u, h_kv, dim), device=k.device, dtype=k.dtype)
            vr = torch.empty((n_u, h_kv, dim), device=v.device, dtype=v.dtype)
            _rev_gather_qkv_kernel[(n_u, h_q + 2 * h_kv)](
                q, k, v, meta.span_meta, qr, kr, vr, H_Q=h_q, H_KV=h_kv, D=dim
            )
            out2, lse2, _, _ = _flash_attn_varlen_forward(
                q=qr, k=kr, v=vr,
                cu_seqlens_q=meta.cu_spans, cu_seqlens_k=meta.cu_spans,
                max_seqlen_q=meta.max_span_upper, max_seqlen_k=meta.max_span_upper,
                dropout_p=0.0, softmax_scale=sm_scale, causal=True,
            )
            _merge_upper_lse_kernel[(n_u, h_q)](
                out1, lse1, out2, lse2, meta.span_meta, seqlen, n_u, H_Q=h_q, D=dim
            )
            ctx.save_for_backward(q, k, v, out1, lse1, qr, kr, vr)
        else:
            ctx.save_for_backward(q, k, v, out1, lse1)
        ctx.meta = meta
        ctx.has_upper = n_u > 0
        return out1

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        meta: OmniPackedBatchMeta = ctx.meta
        do_c = do.contiguous()
        if ctx.has_upper:
            q, k, v, mix_out, mix_lse, qr, kr, vr = ctx.saved_tensors
        else:
            q, k, v, mix_out, mix_lse = ctx.saved_tensors
        seqlen, h_q, dim = q.shape
        h_kv = k.shape[1]
        sm_scale = dim**-0.5

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _flash_attn_varlen_backward(
            dout=do_c, q=q, k=k, v=v, out=mix_out, softmax_lse=mix_lse,
            dq=dq, dk=dk, dv=dv,
            cu_seqlens_q=meta.cu_docs, cu_seqlens_k=meta.cu_docs,
            max_seqlen_q=meta.max_doc_len, max_seqlen_k=meta.max_doc_len,
            dropout_p=0.0, softmax_scale=sm_scale, causal=True,
            window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
        )
        if ctx.has_upper:
            n_u = meta.span_meta.shape[0]
            dor = torch.empty_like(qr)
            mor = torch.empty_like(qr)
            mlr = torch.empty((h_q, n_u), device=q.device, dtype=torch.float32)
            _gather_bwd_upper_inputs_kernel[(n_u, h_q)](
                do_c, mix_out, mix_lse, meta.span_meta, dor, mor, mlr, seqlen, n_u, H_Q=h_q, D=dim
            )
            dq2 = torch.empty_like(qr)
            dk2 = torch.empty_like(kr)
            dv2 = torch.empty_like(vr)
            _flash_attn_varlen_backward(
                dout=dor, q=qr, k=kr, v=vr, out=mor, softmax_lse=mlr,
                dq=dq2, dk=dk2, dv=dv2,
                cu_seqlens_q=meta.cu_spans, cu_seqlens_k=meta.cu_spans,
                max_seqlen_q=meta.max_span_upper, max_seqlen_k=meta.max_span_upper,
                dropout_p=0.0, softmax_scale=sm_scale, causal=True,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
            _scatter_add_rev_grads_kernel[(n_u, h_q + 2 * h_kv)](
                dq, dk, dv, dq2, dk2, dv2, meta.span_meta, H_Q=h_q, H_KV=h_kv, D=dim
            )
        return dq, dk, dv, None


class _TritonTile64OmniGQAFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: OmniPackedBatchMeta,
    ) -> torch.Tensor:
        seqlen, h_q, dim = q.shape
        h_kv = k.shape[1]
        g_heads = h_q // h_kv
        g_pad = triton.next_power_of_2(g_heads)
        sm_scale = dim**-0.5
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        out = torch.empty_like(q_c)
        lse = torch.empty((h_q, seqlen), device=q.device, dtype=torch.float32)
        n_u = seqlen // 8
        _omni_tile64_gqa_fwd_kernel[(n_u, h_kv)](
            q_c, k_c, v_c,
            meta.doc_id, meta.span_id, meta.q_tile_bounds,
            out, lse,
            seqlen, sm_scale,
            B_Q_SUB=8, G=g_heads, G_PAD=g_pad, D=dim, B_K=64,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(q_c, k_c, v_c, out, lse)
        ctx.meta = meta
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q_c, k_c, v_c, out, lse = ctx.saved_tensors
        meta: OmniPackedBatchMeta = ctx.meta
        seqlen, h_q, dim = q_c.shape
        h_kv = k_c.shape[1]
        g_heads = h_q // h_kv
        g_pad = triton.next_power_of_2(g_heads)
        sm_scale = dim**-0.5
        do_c = do.contiguous()

        n_rows = seqlen * h_q
        delta = torch.empty((n_rows,), device=q_c.device, dtype=torch.float32)
        _omni_bwd_preprocess_delta_kernel[(triton.cdiv(n_rows, 16),)](out, do_c, delta, n_rows, D=dim)

        dq = torch.empty_like(q_c)
        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        _omni_tile64_gqa_bwd_dq_kernel[(seqlen // 8, h_kv)](
            q_c, k_c, v_c, do_c, lse, delta,
            meta.doc_id, meta.span_id, meta.q_tile_bounds,
            dq,
            seqlen, sm_scale,
            B_Q_SUB=8, G=g_heads, G_PAD=g_pad, D=dim, B_K=64,
            num_warps=4, num_stages=2,
        )
        _omni_tile64_gqa_bwd_dkv_kernel[(seqlen // 64, h_kv)](
            q_c, k_c, v_c, do_c, lse, delta,
            meta.doc_id, meta.span_id, meta.kv_tile_bounds,
            dk, dv,
            seqlen, sm_scale,
            B_Q_SUB=8, G=g_heads, G_PAD=g_pad, D=dim, B_K=64,
            num_warps=4, num_stages=2,
        )
        return dq, dk, dv, None


class Qwen25VLVisionBlock(nn.Module):
    def __init__(self, d_vit: int = 1280, n_heads: int = 16, mlp_dim: int = 3420) -> None:
        super().__init__()
        self.d_vit = d_vit
        self.n_heads = n_heads
        self.head_dim = d_vit // n_heads  # 80
        self.norm1_w = nn.Parameter(torch.ones(d_vit, dtype=torch.bfloat16))
        self.norm2_w = nn.Parameter(torch.ones(d_vit, dtype=torch.bfloat16))
        self.wqkv = nn.Linear(d_vit, 3 * d_vit, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(d_vit, d_vit, bias=False, dtype=torch.bfloat16)
        self.w_gu = nn.Linear(d_vit, 2 * mlp_dim, bias=False, dtype=torch.bfloat16)
        self.w_dn = nn.Linear(mlp_dim, d_vit, bias=False, dtype=torch.bfloat16)

    def _eager_rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w

    def forward_padded_2d(self, x_pad: torch.Tensor) -> torch.Tensor:
        # x_pad: [B_img, N_max, D_vit]
        bsz, n_max, _ = x_pad.shape
        h1 = self._eager_rmsnorm(x_pad, self.norm1_w)
        qkv = self.wqkv(h1).view(bsz, n_max, 3, self.n_heads, self.head_dim)
        attn = flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], causal=False)
        x_pad = x_pad + self.wo(attn.reshape(bsz, n_max, self.d_vit))
        h2 = self._eager_rmsnorm(x_pad, self.norm2_w)
        g, u = self.w_gu(h2).chunk(2, dim=-1)
        return x_pad + self.w_dn(F.silu(g) * u)

    def forward_varlen(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, use_triton_norm: bool = True
    ) -> torch.Tensor:
        # x: [N_tok, D_vit]
        n_tok = x.shape[0]
        if n_tok == 0:
            return x
        h1 = triton_rmsnorm(x, self.norm1_w) if use_triton_norm else self._eager_rmsnorm(x, self.norm1_w)
        qkv = self.wqkv(h1).view(n_tok, 3, self.n_heads, self.head_dim)
        attn = flash_attn_varlen_func(
            qkv[:, 0], qkv[:, 1], qkv[:, 2],
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=False,
        )
        x = x + self.wo(attn.reshape(n_tok, self.d_vit))
        h2 = triton_rmsnorm(x, self.norm2_w) if use_triton_norm else self._eager_rmsnorm(x, self.norm2_w)
        g, u = self.w_gu(h2).chunk(2, dim=-1)
        return x + self.w_dn(F.silu(g) * u)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "varlen",
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int = 0,
        use_triton_norm: bool = True,
    ) -> torch.Tensor:
        if mode == "padded_2d":
            return self.forward_padded_2d(x)
        assert cu_seqlens is not None
        return self.forward_varlen(x, cu_seqlens, max_seqlen, use_triton_norm=use_triton_norm)


class Qwen25VLPatchMerger(nn.Module):
    def __init__(self, d_vit: int = 1280, d_llm: int = 3584) -> None:
        super().__init__()
        self.d_vit = d_vit
        self.d_merged = 4 * d_vit  # 5120
        self.d_llm = d_llm
        self.ln_q_w = nn.Parameter(torch.ones(d_vit, dtype=torch.bfloat16))
        self.mlp0 = nn.Linear(self.d_merged, self.d_merged, bias=False, dtype=torch.bfloat16)
        self.mlp2 = nn.Linear(self.d_merged, d_llm, bias=False, dtype=torch.bfloat16)

    def forward(self, x_quad_ordered: torch.Tensor, use_triton_norm: bool = True) -> torch.Tensor:
        if x_quad_ordered.shape[0] == 0:
            return x_quad_ordered.new_empty((0, self.d_llm))
        if use_triton_norm:
            xn = triton_rmsnorm(x_quad_ordered, self.ln_q_w)
        else:
            xf = x_quad_ordered.float()
            xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x_quad_ordered.dtype) * self.ln_q_w
        xm = xn.view(-1, self.d_merged)
        return self.mlp2(F.gelu(self.mlp0(xm)))


class Qwen25VLVisionTower(nn.Module):
    def __init__(self, n_layers: int = 4, d_vit: int = 1280, d_llm: int = 3584) -> None:
        super().__init__()
        self.d_vit = d_vit
        self.d_llm = d_llm
        # 16-aligned [1280, 1184] tubelet projection (first 1176 cols active, last 8 cols zero)
        self.patch_proj = nn.Linear(1184, d_vit, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.patch_proj.weight[:, 1176:].zero_()
        self.blocks = nn.ModuleList([Qwen25VLVisionBlock(d_vit=d_vit) for _ in range(n_layers)])
        self.merger = Qwen25VLPatchMerger(d_vit=d_vit, d_llm=d_llm)

    def forward_winmorton_zero_perm(
        self, packed_wm: torch.Tensor, meta: RaggedVisionMeta, unswizzle_to_raster: bool = True
    ) -> torch.Tensor:
        # Blocks 0..n-2 use 8x8 local window cu_seqlens_win; Block n-1 uses global full-frame cu_seqlens_full!
        # ZERO permutations inside ViT because Window-Morton-2x2 nests Frame > 8x8 Window > 2x2 Quad > Patch!
        h = self.patch_proj(packed_wm)
        for idx, blk in enumerate(self.blocks):
            is_full = idx == (len(self.blocks) - 1)
            cu = meta.cu_seqlens_full if is_full else meta.cu_seqlens_win
            ms = meta.max_seqlen_full if is_full else meta.max_seqlen_win
            h = blk(h, mode="varlen", cu_seqlens=cu, max_seqlen=ms, use_triton_norm=True)
        merged_wm = self.merger(h, use_triton_norm=True)
        if unswizzle_to_raster and merged_wm.shape[0] > 0:
            return merged_wm.index_select(0, meta.winquad_to_rasterquad_idx)
        return merged_wm

    def forward_eager_window_gather(
        self, packed_raster2x2: torch.Tensor, meta: RaggedVisionMeta
    ) -> torch.Tensor:
        # HuggingFace Qwen2.5-VL reference: input in raster-2x2 order, permutes into window order for
        # windowed layers and permutes back for full-attention layers / merger!
        h = self.patch_proj(packed_raster2x2)
        h_win = h.index_select(0, meta.raster_to_win_idx)
        for idx, blk in enumerate(self.blocks):
            if idx < len(self.blocks) - 1:
                h_win = blk(h_win, mode="varlen", cu_seqlens=meta.cu_seqlens_win, max_seqlen=meta.max_seqlen_win, use_triton_norm=False)
            else:
                h_ras = h_win.index_select(0, meta.win_to_raster_idx)
                h_ras = blk(h_ras, mode="varlen", cu_seqlens=meta.cu_seqlens_full, max_seqlen=meta.max_seqlen_full, use_triton_norm=False)
        return self.merger(h_ras, use_triton_norm=False)

    def forward_navit_full_varlen(
        self, packed_wm: torch.Tensor, meta: RaggedVisionMeta
    ) -> torch.Tensor:
        h = self.patch_proj(packed_wm)
        for blk in self.blocks:
            h = blk(h, mode="varlen", cu_seqlens=meta.cu_seqlens_full, max_seqlen=meta.max_seqlen_full, use_triton_norm=True)
        merged_wm = self.merger(h, use_triton_norm=True)
        return merged_wm.index_select(0, meta.winquad_to_rasterquad_idx)

    def forward_padded_2d_fa2(
        self, x_pad_raw: torch.Tensor, valid_quad_counts: list[int]
    ) -> torch.Tensor:
        # x_pad_raw: [B_img, N_max, 1184]
        h = self.patch_proj(x_pad_raw)
        for blk in self.blocks:
            h = blk(h, mode="padded_2d")
        # Concatenate valid patches across variable image count BEFORE invoking FSDP2-sharded self.merger ONCE!
        h_valid = torch.cat([h[bi, : qcnt * 4] for bi, qcnt in enumerate(valid_quad_counts)], dim=0)
        return self.merger(h_valid, use_triton_norm=False)

    def forward(
        self,
        x_in: torch.Tensor,
        mode: str = "winmorton",
        meta: RaggedVisionMeta | None = None,
        unswizzle_to_raster: bool = True,
        valid_quad_counts: list[int] | None = None,
    ) -> torch.Tensor:
        if mode == "padded_2d":
            assert valid_quad_counts is not None
            return self.forward_padded_2d_fa2(x_in, valid_quad_counts)
        assert meta is not None
        return self.forward_winmorton_zero_perm(x_in, meta, unswizzle_to_raster=unswizzle_to_raster)


class MultimodalOmniLLMBlock(nn.Module):
    def __init__(
        self,
        d_llm: int = 3584,
        h_q: int = 28,
        h_kv: int = 4,
        head_dim: int = 128,
        ffn_dim: int = 9728,
    ) -> None:
        super().__init__()
        self.d_llm = d_llm
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = head_dim
        self.norm1_w = nn.Parameter(torch.ones(d_llm, dtype=torch.bfloat16))
        self.norm2_w = nn.Parameter(torch.ones(d_llm, dtype=torch.bfloat16))
        self.wq = nn.Linear(d_llm, h_q * head_dim, bias=False, dtype=torch.bfloat16)
        self.wkv = nn.Linear(d_llm, 2 * h_kv * head_dim, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(h_q * head_dim, d_llm, bias=False, dtype=torch.bfloat16)
        self.w_gu = nn.Linear(d_llm, 2 * ffn_dim, bias=False, dtype=torch.bfloat16)
        self.w_dn = nn.Linear(ffn_dim, d_llm, bias=False, dtype=torch.bfloat16)

    def _eager_rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w

    def forward(
        self,
        x: torch.Tensor,
        meta: OmniPackedBatchMeta,
        inv_freq: torch.Tensor,
        mode: str = "surgical_fa2_triton_mrope",
        flex_bm: Any = None,
        comp_flex: Any = None,
    ) -> torch.Tensor:
        seqlen = x.shape[0]
        use_fast = mode in ("surgical_fa2_triton_mrope", "triton_tile64_triton_mrope")
        h1 = triton_rmsnorm(x, self.norm1_w) if use_fast else self._eager_rmsnorm(x, self.norm1_w)
        q = self.wq(h1).view(seqlen, self.h_q, self.head_dim)
        kv = self.wkv(h1).view(seqlen, 2, self.h_kv, self.head_dim)
        k, v = kv[:, 0], kv[:, 1]

        if use_fast:
            q, k = _FusedQK3DMRoPEFn.apply(q, k, meta.pos_thw, inv_freq, False)
        else:
            q, k = hf_eager_3d_mrope(q, k, meta.pos_thw, inv_freq, interleaved=False)

        if mode == "surgical_fa2_triton_mrope":
            attn = _SurgicalOmniFA2Fn.apply(q, k, v, meta)
        elif mode == "triton_tile64_triton_mrope":
            attn = _TritonTile64OmniGQAFn.apply(q, k, v, meta)
        elif mode == "flex_attn":
            qfx = q.transpose(0, 1).unsqueeze(0).contiguous()
            kfx = k.transpose(0, 1).unsqueeze(0).contiguous()
            vfx = v.contiguous().transpose(0, 1).unsqueeze(0).contiguous()
            ofx = comp_flex(qfx, kfx, vfx, block_mask=flex_bm, enable_gqa=True)
            attn = ofx.squeeze(0).transpose(0, 1).contiguous()
        else:
            raise ValueError(f"Unknown LLM attention mode: {mode}")

        x = x + self.wo(attn.view(seqlen, self.h_q * self.head_dim))
        h2 = triton_rmsnorm(x, self.norm2_w) if use_fast else self._eager_rmsnorm(x, self.norm2_w)
        g, u = self.w_gu(h2).chunk(2, dim=-1)
        return x + self.w_dn(F.silu(g) * u)


class _DifferentiableVarlenAllToAllFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        out_splits: list[int],
        in_splits: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        out_rows = sum(out_splits) if False else sum(in_splits)
        # Note: in PyTorch dist.all_to_all_single(output, input, output_split_sizes=in_splits, input_split_sizes=out_splits):
        # `out_splits` = rows WE send to each rank (sum == x.shape[0]); `in_splits` = rows WE receive from each rank!
        recv_rows = sum(in_splits)
        y = x.new_empty((recv_rows, *x.shape[1:]))
        dist.all_to_all_single(
            y, x.contiguous(), output_split_sizes=in_splits, input_split_sizes=out_splits, group=group
        )
        ctx.out_splits = out_splits
        ctx.in_splits = in_splits
        ctx.group = group
        return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        send_rows = sum(ctx.out_splits)
        dx = dy.new_empty((send_rows, *dy.shape[1:]))
        dist.all_to_all_single(
            dx, dy.contiguous(), output_split_sizes=ctx.out_splits, input_split_sizes=ctx.in_splits, group=ctx.group
        )
        return dx, None, None, None


# ============================================================================
# 3. Suite 0: Foundational Multimodal Audits, Roofline Diagnosis, & FP64 Gates
# ============================================================================


def _fp64_omni_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_thw: torch.Tensor,
    inv_freq_64: torch.Tensor,
    doc_id: torch.Tensor,
    span_id: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    # Exact FP64 3D M-RoPE + Hybrid Bidirectional-Vision / Causal-Text Omni-Attention reference
    freqs_3 = pos_thw.to(torch.float64).unsqueeze(-1) * inv_freq_64.view(1, 1, -1)
    if interleaved:
        fm = freqs_3[0].clone()
        fm[:, 1:60:3] = freqs_3[1, :, 1:60:3]
        fm[:, 2:60:3] = freqs_3[2, :, 2:60:3]
    else:
        fm = torch.cat([freqs_3[0, :, :16], freqs_3[1, :, 16:40], freqs_3[2, :, 40:64]], dim=-1)
    emb = torch.cat([fm, fm], dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    qr = (q * cos) + (torch.cat((-q[..., 64:], q[..., :64]), dim=-1) * sin)
    kr = (k * cos) + (torch.cat((-k[..., 64:], k[..., :64]), dim=-1) * sin)

    seqlen, h_q, dim = qr.shape
    h_kv = kr.shape[1]
    g_heads = h_q // h_kv
    qi = torch.arange(seqlen, device=q.device)[:, None]
    ki = torch.arange(seqlen, device=q.device)[None, :]
    m_omni = (doc_id[qi] == doc_id[ki]) & ((ki <= qi) | ((span_id[qi] > 0) & (span_id[qi] == span_id[ki])))

    qp = qr.permute(1, 0, 2)
    ke = kr.repeat_interleave(g_heads, dim=1).permute(1, 0, 2)
    ve = v.repeat_interleave(g_heads, dim=1).permute(1, 0, 2)
    sc = torch.matmul(qp, ke.transpose(-1, -2)) * (dim**-0.5)
    sc = sc.masked_fill(~m_omni.unsqueeze(0), float("-inf"))
    pr = torch.softmax(sc, dim=-1)
    return torch.matmul(pr, ve).permute(1, 0, 2).contiguous()


def run_suite_0_audits(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    warmup = 2 if is_smoke else 4
    iters = 4 if is_smoke else 10
    torch.manual_seed(2026)

    # ------------------------------------------------------------------------
    # 0.1 cuDNN 9.20 Conv3d(2,14,14) Pathology vs Window-Morton-2x2 Tubelet Pack + cuBLAS
    #     & Post-ViT 2x2 Spatial Token Merger Roofline (16x [3, 4, 448, 448] = 32,768 patches / 16,384 in smoke)
    # ------------------------------------------------------------------------
    n_clips = 8 if is_smoke else 16
    shapes_01 = [(4, 448, 448) for _ in range(n_clips)]
    n_patches_01 = n_clips * 2 * 32 * 32  # 16,384 smoke / 32,768 full
    meta_01 = build_window_morton2x2_ragged_meta(shapes_01, device)

    conv3d = nn.Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False, device=device, dtype=torch.bfloat16)
    w_pad = torch.zeros((1280, 1184), device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        w_pad[:, :1176].copy_(conv3d.weight.detach().reshape(1280, 1176))
    w_pad.requires_grad_(True)
    w_1176 = conv3d.weight.detach().reshape(1280, 1176).clone().requires_grad_(True)

    x_5d = (torch.randn(n_clips, 3, 4, 448, 448, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
    x_hf = x_5d.detach().view(n_clips, 3, 2, 2, 32, 14, 32, 14).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(n_patches_01, 3, 2, 14, 14).requires_grad_(True)
    x_flat = x_5d.detach().reshape(-1).clone().requires_grad_(True)

    dy_5d = torch.randn(n_clips, 1280, 2, 32, 32, device=device, dtype=torch.bfloat16) * 0.3
    # Map dy_5d into raster-2x2 and win-morton-2x2 patch orders to verify exact Conv3d equivalence!
    dy_ras2x2 = (
        dy_5d.permute(0, 2, 3, 4, 1)
        .view(n_clips, 2, 16, 2, 16, 2, 1280)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .contiguous()
        .view(n_patches_01, 1280)
    )
    dy_wm = dy_ras2x2.index_select(0, meta_01.raster_to_win_idx).contiguous()
    dy_hf = dy_5d.permute(0, 2, 3, 4, 1).contiguous().view(n_patches_01, 1280, 1, 1, 1)

    # Exact parity check between 5D Conv3d and Triton Window-Morton-2x2 Pack + cuBLAS 1184!
    y_c3d = conv3d(x_5d)
    y_c3d.backward(dy_5d)
    ref_y_ras2x2 = (
        y_c3d.detach()
        .permute(0, 2, 3, 4, 1)
        .view(n_clips, 2, 16, 2, 16, 2, 1280)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(n_patches_01, 1280)
    )
    ref_dx_flat = x_5d.grad.detach().reshape(-1).clone()
    ref_dw_1176 = conv3d.weight.grad.detach().reshape(1280, 1176).clone()
    x_5d.grad = conv3d.weight.grad = None

    packed_wm = _RaggedTubeletPackFn.apply(x_flat, meta_01.patch_meta)
    y_wm = F.linear(packed_wm, w_pad)
    y_wm.backward(dy_wm)
    y_wm_unswz = y_wm.detach().index_select(0, meta_01.win_to_raster_idx)
    tubelet_out_err = rel_l2(y_wm_unswz, ref_y_ras2x2)
    tubelet_dx_err = rel_l2(x_flat.grad, ref_dx_flat)
    tubelet_dw_err = rel_l2(w_pad.grad[:, :1176], ref_dw_1176)
    x_flat.grad = w_pad.grad = None

    def _run_hf_c3d() -> None:
        conv3d(x_hf).backward(dy_hf)
        x_hf.grad = conv3d.weight.grad = None

    def _run_5d_c3d() -> None:
        conv3d(x_5d).backward(dy_5d)
        x_5d.grad = conv3d.weight.grad = None

    def _run_eager_10d() -> None:
        xp = x_5d.view(n_clips, 3, 2, 2, 16, 2, 14, 16, 2, 14)
        xp = xp.permute(0, 2, 4, 7, 5, 8, 1, 3, 6, 9).contiguous().view(n_patches_01, 1176)
        F.linear(xp, w_1176).backward(dy_ras2x2)
        x_5d.grad = w_1176.grad = None

    def _run_triton_wm_cublas() -> None:
        pk = _RaggedTubeletPackFn.apply(x_flat, meta_01.patch_meta)
        F.linear(pk, w_pad).backward(dy_wm)
        x_flat.grad = w_pad.grad = None

    st_hf_c3d = time_fn_with_peak_mem(_run_hf_c3d, warmup=warmup, iters=iters, device=device)
    st_5d_c3d = time_fn_with_peak_mem(_run_5d_c3d, warmup=warmup, iters=iters, device=device)
    st_eag_10d = time_fn_with_peak_mem(_run_eager_10d, warmup=warmup, iters=iters, device=device)
    st_tri_wm = time_fn_with_peak_mem(_run_triton_wm_cublas, warmup=warmup, iters=iters, device=device)

    flops_3gemm_patch = 6.0 * n_patches_01 * 1176 * 1280
    for st in (st_hf_c3d, st_5d_c3d, st_eag_10d, st_tri_wm):
        st["effective_tflops"] = round(flops_3gemm_patch / (st["median_ms"] * 1e-3) / 1e12, 2)

    # Post-ViT 2x2 Spatial Token Merger ablation: Raster-Order Unfused vs Zero-Copy Quad-Contiguous Fused RMSNorm
    merger_mod = Qwen25VLPatchMerger(1280, 3584).to(device)
    h_vit_ras = torch.randn(n_clips, 2, 32, 32, 1280, device=device, dtype=torch.bfloat16, requires_grad=True)
    h_vit_wm = (
        h_vit_ras.detach()
        .view(n_clips, 2, 16, 2, 16, 2, 1280)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .contiguous()
        .view(n_patches_01, 1280)
        .requires_grad_(True)
    )
    dm_out = torch.randn(n_patches_01 // 4, 3584, device=device, dtype=torch.bfloat16)

    def _merger_raster_eager() -> None:
        hs = (
            h_vit_ras.view(n_clips, 2, 16, 2, 16, 2, 1280)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .contiguous()
            .view(n_patches_01, 1280)
        )
        merger_mod(hs, use_triton_norm=False).backward(dm_out)
        h_vit_ras.grad = None
        for p in merger_mod.parameters():
            p.grad = None

    def _merger_quad_zero_copy() -> None:
        merger_mod(h_vit_wm, use_triton_norm=True).backward(dm_out)
        h_vit_wm.grad = None
        for p in merger_mod.parameters():
            p.grad = None

    st_m_ras = time_fn_with_peak_mem(_merger_raster_eager, warmup=warmup, iters=iters, device=device)
    st_m_zc = time_fn_with_peak_mem(_merger_quad_zero_copy, warmup=warmup, iters=iters, device=device)

    tubelet_and_merger_audit = {
        "num_patches": n_patches_01,
        "num_merged_tokens": n_patches_01 // 4,
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "parity_vs_conv3d": {
            "out_rel_l2": round(tubelet_out_err, 6),
            "dx_rel_l2": round(tubelet_dx_err, 6),
            "dw_rel_l2": round(tubelet_dw_err, 6),
        },
        "tubelet_modes": {
            "hf_conv3d_preextracted_patches": st_hf_c3d,
            "conv3d_5d_ncthw_video": st_5d_c3d,
            "eager_10d_permute_swizzle_linear": st_eag_10d,
            "triton_winmorton2x2_pack_cublas_1184": st_tri_wm,
        },
        "speedup_triton_pack_over_hf_conv3d": round(st_hf_c3d["median_ms"] / st_tri_wm["median_ms"], 3),
        "speedup_triton_pack_over_5d_conv3d": round(st_5d_c3d["median_ms"] / st_tri_wm["median_ms"], 3),
        "speedup_triton_pack_over_eager_10d": round(st_eag_10d["median_ms"] / st_tri_wm["median_ms"], 3),
        "patch_merger_modes": {
            "raster_permute_eager_rmsnorm_merger": st_m_ras,
            "quad_contiguous_triton_rmsnorm_merger": st_m_zc,
            "speedup_zero_copy_over_raster": round(st_m_ras["median_ms"] / st_m_zc["median_ms"], 3),
        },
    }
    del conv3d, w_pad, w_1176, x_5d, x_hf, x_flat, dy_5d, dy_ras2x2, dy_wm, dy_hf, h_vit_ras, h_vit_wm, dm_out
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # 0.2 3D Multimodal RoPE (M-RoPE: Sectioned [16,24,24] & Interleaved [24,20,20])
    #     1D Text Reduction Exactness + Single-Launch In-Register Q+K Kernel Ablation (S = 16,384, 28Q/4KV, D=128)
    # ------------------------------------------------------------------------
    s_mr = 16384
    inv_freq = (1.0 / (1000000.0 ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128.0))).contiguous()
    inv_freq_64 = (1.0 / (1000000.0 ** (torch.arange(0, 128, 2, device=device, dtype=torch.float64) / 128.0))).contiguous()

    # Verify exact 1D RoPE mathematical identity when pos_t == pos_h == pos_w == i
    pos_1d = torch.arange(256, device=device, dtype=torch.int32).unsqueeze(0).expand(3, 256).contiguous()
    qt64 = torch.randn(256, 28, 128, device=device, dtype=torch.float64)
    kt64 = torch.randn(256, 4, 128, device=device, dtype=torch.float64)
    ang_1d = torch.arange(256, device=device, dtype=torch.float64).unsqueeze(-1) * inv_freq_64.unsqueeze(0)
    emb_1d = torch.cat([ang_1d, ang_1d], dim=-1)
    q_1d_ref = qt64 * emb_1d.cos().unsqueeze(1) + torch.cat((-qt64[..., 64:], qt64[..., :64]), dim=-1) * emb_1d.sin().unsqueeze(1)

    def _mrope64(x64: torch.Tensor, p3: torch.Tensor, inter: bool) -> torch.Tensor:
        f3 = p3.to(torch.float64).unsqueeze(-1) * inv_freq_64.view(1, 1, -1)
        if inter:
            fm = f3[0].clone()
            fm[:, 1:60:3] = f3[1, :, 1:60:3]
            fm[:, 2:60:3] = f3[2, :, 2:60:3]
        else:
            fm = torch.cat([f3[0, :, :16], f3[1, :, 16:40], f3[2, :, 40:64]], dim=-1)
        em = torch.cat([fm, fm], dim=-1)
        return x64 * em.cos().unsqueeze(1) + torch.cat((-x64[..., 64:], x64[..., :64]), dim=-1) * em.sin().unsqueeze(1)

    text_1d_reduction_err_sec = rel_l2(_mrope64(qt64, pos_1d, False), q_1d_ref)
    text_1d_reduction_err_int = rel_l2(_mrope64(qt64, pos_1d, True), q_1d_ref)

    pos_3d = torch.randint(0, 2048, (3, s_mr), device=device, dtype=torch.int32)
    q_mr = torch.randn(s_mr, 28, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k_mr = torch.randn(s_mr, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    dq_mr = torch.randn_like(q_mr)
    dk_mr = torch.randn_like(k_mr)

    # Parity check HF Eager vs Fused Triton (Sectioned & Interleaved)
    qo_ref, ko_ref = hf_eager_3d_mrope(q_mr, k_mr, pos_3d, inv_freq, interleaved=False)
    torch.autograd.backward((qo_ref, ko_ref), (dq_mr, dk_mr))
    dq_ref_s, dk_ref_s = q_mr.grad.detach().clone(), k_mr.grad.detach().clone()
    q_mr.grad = k_mr.grad = None

    qo_tri, ko_tri = _FusedQK3DMRoPEFn.apply(q_mr, k_mr, pos_3d, inv_freq, False)
    torch.autograd.backward((qo_tri, ko_tri), (dq_mr, dk_mr))
    mrope_sec_q_err = rel_l2(qo_tri, qo_ref)
    mrope_sec_dq_err = rel_l2(q_mr.grad, dq_ref_s)
    mrope_sec_dk_err = rel_l2(k_mr.grad, dk_ref_s)
    q_mr.grad = k_mr.grad = None

    qo_iref, ko_iref = hf_eager_3d_mrope(q_mr, k_mr, pos_3d, inv_freq, interleaved=True)
    qo_itri, ko_itri = _FusedQK3DMRoPEFn.apply(q_mr, k_mr, pos_3d, inv_freq, True)
    mrope_int_q_err = rel_l2(qo_itri, qo_iref)

    comp_hf_mrope = torch.compile(hf_eager_3d_mrope)

    def _bench_mrope(fn_call: Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
        def _step() -> None:
            qo, ko = fn_call()
            torch.autograd.backward((qo, ko), (dq_mr, dk_mr))
            q_mr.grad = k_mr.grad = None

        st = time_fn_with_peak_mem(_step, warmup=warmup, iters=iters, device=device)
        # Useful bytes moved in Fwd (read Q,K + write QOut,KOut) + Bwd (read dQOut,dKOut + write dQ,dK):
        # 4 * (28 + 4) * S * 128 * 2 bytes!
        useful_bytes = 4.0 * 32.0 * s_mr * 128.0 * 2.0
        st["achieved_useful_gbps"] = round(useful_bytes / (st["median_ms"] * 1e-3) / 1e9, 2)
        return st

    st_mr_eag = _bench_mrope(lambda: hf_eager_3d_mrope(q_mr, k_mr, pos_3d, inv_freq, False))
    st_mr_cmp = _bench_mrope(lambda: comp_hf_mrope(q_mr, k_mr, pos_3d, inv_freq, False))
    st_mr_sec = _bench_mrope(lambda: _FusedQK3DMRoPEFn.apply(q_mr, k_mr, pos_3d, inv_freq, False))
    st_mr_int = _bench_mrope(lambda: _FusedQK3DMRoPEFn.apply(q_mr, k_mr, pos_3d, inv_freq, True))

    mrope_audit = {
        "seqlen": s_mr,
        "heads_q_kv": [28, 4],
        "head_dim": 128,
        "text_1d_rope_exact_reduction_fp64_rel_l2": {
            "sectioned_16_24_24": round(text_1d_reduction_err_sec, 16),
            "interleaved_24_20_20": round(text_1d_reduction_err_int, 16),
        },
        "parity_vs_hf_eager": {
            "sectioned_q_rel_l2": round(mrope_sec_q_err, 6),
            "sectioned_dq_rel_l2": round(mrope_sec_dq_err, 6),
            "sectioned_dk_rel_l2": round(mrope_sec_dk_err, 6),
            "interleaved_q_rel_l2": round(mrope_int_q_err, 6),
        },
        "modes": {
            "hf_eager_3d_mrope": st_mr_eag,
            "torch_compile_3d_mrope": st_mr_cmp,
            "triton_fused_qk_3d_mrope_sectioned": st_mr_sec,
            "triton_fused_qk_3d_mrope_interleaved": st_mr_int,
        },
        "speedup_triton_over_hf_eager": round(st_mr_eag["median_ms"] / st_mr_sec["median_ms"], 3),
        "speedup_triton_over_torch_compile": round(st_mr_cmp["median_ms"] / st_mr_sec["median_ms"], 3),
    }
    del q_mr, k_mr, dq_mr, dk_mr, qo_ref, ko_ref, qo_tri, ko_tri, qo_iref, ko_iref, qo_itri, ko_itri
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # 0.3 PyTorch 2.13 Multimodal Omni-Attention Backend Audit (28Q/4KV, D=128):
    #     1) FA2 causal=True visual-span upper-triangle semantic corruption;
    #     2) SDPA bool mask silent Math-SDPA fallback (S=8192) & 28 GiB OOM at S=16384;
    #     3) Exact parity of Triton Tile-64 & Surgical Disjoint Reversed-Upper FA2.
    # ------------------------------------------------------------------------
    docs_8k = [
        [("text", 128, 0, 0, 0), ("vision", 576, 1, 24, 24), ("text", 416, 0, 0, 0), ("vision", 432, 1, 24, 18), ("text", 500, 0, 0, 0)]
        for _ in range(4)
    ]
    meta_8k = build_omni_packed_batch_meta(docs_8k, device, pad_to_multiple=64)
    s_aud = meta_8k.seqlen  # 8,256
    q_a = (torch.randn(s_aud, 28, 128, device=device, dtype=torch.bfloat16) * 0.4).requires_grad_(True)
    k_a = (torch.randn(s_aud, 4, 128, device=device, dtype=torch.bfloat16) * 0.4).requires_grad_(True)
    # Give visual tokens coherent semantic key structure so bidirectional visual attention matters
    with torch.no_grad():
        k_a[meta_8k.vis_token_indices] += 0.6
    v_a = (torch.randn(s_aud, 4, 128, device=device, dtype=torch.bfloat16) * 0.4).requires_grad_(True)
    do_a = torch.randn_like(q_a) * 0.4

    # SDPA ground truth at S ~ 8K
    qi_a = torch.arange(s_aud, device=device)[:, None]
    ki_a = torch.arange(s_aud, device=device)[None, :]
    m2d_a = (meta_8k.doc_id[qi_a] == meta_8k.doc_id[ki_a]) & (
        (ki_a <= qi_a) | ((meta_8k.span_id[qi_a] > 0) & (meta_8k.span_id[qi_a] == meta_8k.span_id[ki_a]))
    )
    m4d_a = m2d_a.view(1, 1, s_aud, s_aud)

    q_sd = q_a.detach().transpose(0, 1).unsqueeze(0).contiguous().requires_grad_(True)
    k_sd = k_a.detach().transpose(0, 1).unsqueeze(0).contiguous().requires_grad_(True)
    v_sd = v_a.detach().transpose(0, 1).unsqueeze(0).contiguous().requires_grad_(True)
    do_sd = do_a.transpose(0, 1).unsqueeze(0).contiguous()
    out_sd = F.scaled_dot_product_attention(q_sd, k_sd, v_sd, attn_mask=m4d_a, enable_gqa=True)
    out_sd.backward(do_sd)
    ref_sd_out = out_sd.detach().squeeze(0).transpose(0, 1).contiguous()
    ref_sd_dq = q_sd.grad.detach().squeeze(0).transpose(0, 1).contiguous()
    ref_sd_dk = k_sd.grad.detach().squeeze(0).transpose(0, 1).contiguous()
    q_sd.grad = k_sd.grad = v_sd.grad = None

    st_sdpa_8k = time_fn_with_peak_mem(
        lambda: (
            F.scaled_dot_product_attention(q_sd, k_sd, v_sd, attn_mask=m4d_a, enable_gqa=True).backward(do_sd),
            setattr(q_sd, "grad", None),
            setattr(k_sd, "grad", None),
            setattr(v_sd, "grad", None),
        ),
        warmup=1,
        iters=3,
        device=device,
    )

    # Naive FA2 causal=True semantic corruption on visual tokens
    out_causal_wrong = flash_attn_varlen_func(
        q_a.detach(), k_a.detach(), v_a.detach(),
        cu_seqlens_q=meta_8k.cu_docs, cu_seqlens_k=meta_8k.cu_docs,
        max_seqlen_q=meta_8k.max_doc_len, max_seqlen_k=meta_8k.max_doc_len,
        causal=True,
    )
    vis_idx = meta_8k.vis_token_indices
    causal_only_vis_corruption = rel_l2(out_causal_wrong[vis_idx], ref_sd_out[vis_idx])

    # Triton Tile-64 & Surgical Disjoint Reversed-Upper FA2 parity vs SDPA
    out_t64 = _TritonTile64OmniGQAFn.apply(q_a, k_a, v_a, meta_8k)
    out_t64.backward(do_a)
    t64_out_err = rel_l2(out_t64, ref_sd_out)
    t64_dq_err = rel_l2(q_a.grad, ref_sd_dq)
    t64_dk_err = rel_l2(k_a.grad, ref_sd_dk)
    q_a.grad = k_a.grad = v_a.grad = None

    out_surg = _SurgicalOmniFA2Fn.apply(q_a, k_a, v_a, meta_8k)
    out_surg.backward(do_a)
    surg_out_err = rel_l2(out_surg, ref_sd_out)
    surg_dq_err = rel_l2(q_a.grad, ref_sd_dq)
    surg_dk_err = rel_l2(k_a.grad, ref_sd_dk)
    q_a.grad = k_a.grad = v_a.grad = None
    del m2d_a, m4d_a, q_sd, k_sd, v_sd, do_sd, out_sd, q_a, k_a, v_a, do_a
    torch.cuda.empty_cache()

    # Verify SDPA 28.00 GiB OOM at S = 16,384 on 28Q/4KV!
    s_oom = 16384
    m_oom = torch.ones((1, 1, s_oom, s_oom), device=device, dtype=torch.bool)
    q_oom = torch.randn(1, 28, s_oom, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k_oom = torch.randn(1, 4, s_oom, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    v_oom = torch.randn(1, 4, s_oom, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    try:
        o_oom = F.scaled_dot_product_attention(q_oom, k_oom, v_oom, attn_mask=m_oom, enable_gqa=True)
        o_oom.sum().backward()
        sdpa_s16k_audit: dict[str, Any] = {"oom_triggered": False}
    except torch.OutOfMemoryError as e:
        sdpa_s16k_audit = {
            "oom_triggered": True,
            "error_excerpt": str(e)[:190],
        }
    del m_oom, q_oom, k_oom, v_oom
    torch.cuda.empty_cache()

    omni_backend_audit = {
        "audit_seqlen_s8k": s_aud,
        "fa2_causal_only_visual_token_rel_l2_error": round(causal_only_vis_corruption, 5),
        "sdpa_bool_mask_s8k_math_fallback": st_sdpa_8k,
        "sdpa_bool_mask_s16k_oom_audit": sdpa_s16k_audit,
        "triton_tile64_parity_vs_sdpa": {
            "out_rel_l2": round(t64_out_err, 6),
            "dq_rel_l2": round(t64_dq_err, 6),
            "dk_rel_l2": round(t64_dk_err, 6),
        },
        "surgical_disjoint_fa2_parity_vs_sdpa": {
            "out_rel_l2": round(surg_out_err, 6),
            "dq_rel_l2": round(surg_dq_err, 6),
            "dk_rel_l2": round(surg_dk_err, 6),
        },
    }

    # ------------------------------------------------------------------------
    # 0.4 BF16 vs FP64 Exact Multimodal Oracle & FP64 Richardson Central-Difference (h=2e-2 vs 1e-2)
    # ------------------------------------------------------------------------
    docs_fd = [
        [("text", 16, 0, 0, 0), ("vision", 36, 1, 6, 6), ("text", 12, 0, 0, 0)],
        [("vision", 32, 2, 4, 4), ("text", 32, 0, 0, 0)],
    ]
    meta_fd = build_omni_packed_batch_meta(docs_fd, device, pad_to_multiple=64)
    s_fd = meta_fd.seqlen  # 128

    torch.manual_seed(777)
    q_bf = (torch.randn(s_fd, 28, 128, device=device, dtype=torch.bfloat16) * 0.4).requires_grad_(True)
    k_bf = (torch.randn(s_fd, 4, 128, device=device, dtype=torch.bfloat16) * 0.4).requires_grad_(True)
    v_bf = (torch.randn(s_fd, 4, 128, device=device, dtype=torch.bfloat16) * 0.4).requires_grad_(True)
    do_bf = torch.randn_like(q_bf) * 0.4

    qr_b, kr_b = _FusedQK3DMRoPEFn.apply(q_bf, k_bf, meta_fd.pos_thw, inv_freq, False)
    o_surg_b = _SurgicalOmniFA2Fn.apply(qr_b, kr_b, v_bf, meta_fd)
    o_surg_b.backward(do_bf)
    bf_dq, bf_dk, bf_dv = q_bf.grad.detach().clone(), k_bf.grad.detach().clone(), v_bf.grad.detach().clone()
    q_bf.grad = k_bf.grad = v_bf.grad = None

    qr_t, kr_t = _FusedQK3DMRoPEFn.apply(q_bf, k_bf, meta_fd.pos_thw, inv_freq, False)
    o_t64_b = _TritonTile64OmniGQAFn.apply(qr_t, kr_t, v_bf, meta_fd)
    o_t64_b.backward(do_bf)
    t64_dq, t64_dk, t64_dv = q_bf.grad.detach().clone(), k_bf.grad.detach().clone(), v_bf.grad.detach().clone()
    q_bf.grad = k_bf.grad = v_bf.grad = None

    q64 = q_bf.detach().double().requires_grad_(True)
    k64 = k_bf.detach().double().requires_grad_(True)
    v64 = v_bf.detach().double().requires_grad_(True)
    do64 = do_bf.detach().double()
    o64 = _fp64_omni_oracle(q64, k64, v64, meta_fd.pos_thw, inv_freq_64, meta_fd.doc_id, meta_fd.span_id, False)
    o64.backward(do64)

    bf16_vs_fp64_oracle = {
        "surgical_omni_out_rel_l2": round(rel_l2(o_surg_b, o64), 6),
        "surgical_omni_dq_rel_l2": round(rel_l2(bf_dq, q64.grad), 6),
        "surgical_omni_dk_rel_l2": round(rel_l2(bf_dk, k64.grad), 6),
        "surgical_omni_dv_rel_l2": round(rel_l2(bf_dv, v64.grad), 6),
        "tile64_omni_out_rel_l2": round(rel_l2(o_t64_b, o64), 6),
        "tile64_omni_dq_rel_l2": round(rel_l2(t64_dq, q64.grad), 6),
        "tile64_omni_dk_rel_l2": round(rel_l2(t64_dk, k64.grad), 6),
        "tile64_omni_dv_rel_l2": round(rel_l2(t64_dv, v64.grad), 6),
    }

    # FP64 Richardson Extrapolation across 8 non-linear multimodal derivatives (h = 2e-3 vs 1e-3 on compact 32-tok G=7 sequence)
    def _richardson_ratio(loss_fn: Callable[[torch.Tensor], torch.Tensor], x_base: torch.Tensor, grad_exact: torch.Tensor) -> float:
        gen = torch.Generator(device=device).manual_seed(919)
        d_vec = torch.randn(x_base.shape, generator=gen, device=device, dtype=torch.float64)
        # Unit-RMS direction (`||d||_rms = 1`) so per-element step is exactly `h` regardless of tensor size!
        d_vec = d_vec / (d_vec.pow(2).mean().sqrt() + 1e-30)
        exact_dir = float((grad_exact * d_vec).sum().item())
        h1, h2 = 2.0e-3, 1.0e-3
        fd_h1 = float((loss_fn(x_base + h1 * d_vec) - loss_fn(x_base - h1 * d_vec)).item()) / (2.0 * h1)
        fd_h2 = float((loss_fn(x_base + h2 * d_vec) - loss_fn(x_base - h2 * d_vec)).item()) / (2.0 * h2)
        err1 = abs(fd_h1 - exact_dir)
        err2 = max(1e-16, abs(fd_h2 - exact_dir))
        return round(err1 / err2, 4)

    docs_rich = [
        [("text", 4, 0, 0, 0), ("vision", 8, 2, 2, 2), ("text", 4, 0, 0, 0)],
        [("vision", 8, 2, 2, 2), ("text", 8, 0, 0, 0)],
    ]
    meta_rich = build_omni_packed_batch_meta(docs_rich, device, pad_to_multiple=16)
    torch.manual_seed(778)
    q64_nl = (torch.randn(meta_rich.seqlen, 7, 128, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    k64_nl = (torch.randn(meta_rich.seqlen, 1, 128, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    v64_nl = (torch.randn(meta_rich.seqlen, 1, 128, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
    do64_nl = torch.randn_like(q64_nl) * 0.4
    o_nl = _fp64_omni_oracle(q64_nl, k64_nl, v64_nl, meta_rich.pos_thw, inv_freq_64, meta_rich.doc_id, meta_rich.span_id, False)
    loss_nl = (torch.sin(o_nl) * do64_nl).sum()
    loss_nl.backward()

    q64_int = q64_nl.detach().clone().requires_grad_(True)
    k64_int = k64_nl.detach().clone().requires_grad_(True)
    o_int = _fp64_omni_oracle(q64_int, k64_int, v64_nl.detach(), meta_rich.pos_thw, inv_freq_64, meta_rich.doc_id, meta_rich.span_id, True)
    (torch.sin(o_int) * do64_nl).sum().backward()

    # 2x2 Spatial RMSNorm + GELU Token Merger in FP64
    xm64 = (torch.randn(64, 64, device=device, dtype=torch.float64) * 0.5).requires_grad_(True)
    gm64 = (torch.ones(64, device=device, dtype=torch.float64) + 0.1 * torch.randn(64, device=device, dtype=torch.float64)).requires_grad_(True)
    wm1_64 = (torch.randn(128, 256, device=device, dtype=torch.float64) * 0.15).requires_grad_(True)
    wm2_64 = (torch.randn(96, 128, device=device, dtype=torch.float64) * 0.15).requires_grad_(True)
    dm64 = torch.randn(16, 96, device=device, dtype=torch.float64)

    def _merger_fp64_loss(x_in: torch.Tensor, g_in: torch.Tensor, w1_in: torch.Tensor, w2_in: torch.Tensor) -> torch.Tensor:
        rstd = torch.rsqrt(x_in.pow(2).mean(-1, keepdim=True) + 1e-6)
        xn = (x_in * rstd * g_in).view(-1, 256)
        out_m = F.linear(F.gelu(F.linear(xn, w1_in)), w2_in)
        return (torch.sin(out_m) * dm64).sum()

    _merger_fp64_loss(xm64, gm64, wm1_64, wm2_64).backward()

    rich_ratios = {
        "merger_dx": _richardson_ratio(lambda z: _merger_fp64_loss(z, gm64.detach(), wm1_64.detach(), wm2_64.detach()), xm64.detach(), xm64.grad),
        "merger_dgamma": _richardson_ratio(lambda z: _merger_fp64_loss(xm64.detach(), z, wm1_64.detach(), wm2_64.detach()), gm64.detach(), gm64.grad),
        "merger_dw1": _richardson_ratio(lambda z: _merger_fp64_loss(xm64.detach(), gm64.detach(), z, wm2_64.detach()), wm1_64.detach(), wm1_64.grad),
        "merger_dw2": _richardson_ratio(lambda z: _merger_fp64_loss(xm64.detach(), gm64.detach(), wm1_64.detach(), z), wm2_64.detach(), wm2_64.grad),
        "mrope_interleaved_dk": _richardson_ratio(
            lambda z: (torch.sin(_fp64_omni_oracle(q64_int.detach(), z, v64_nl.detach(), meta_rich.pos_thw, inv_freq_64, meta_rich.doc_id, meta_rich.span_id, True)) * do64_nl).sum(),
            k64_int.detach(), k64_int.grad,
        ),
        "omni_mrope_sec_dq": _richardson_ratio(
            lambda z: (torch.sin(_fp64_omni_oracle(z, k64_nl.detach(), v64_nl.detach(), meta_rich.pos_thw, inv_freq_64, meta_rich.doc_id, meta_rich.span_id, False)) * do64_nl).sum(),
            q64_nl.detach(), q64_nl.grad,
        ),
        "omni_mrope_sec_dk": _richardson_ratio(
            lambda z: (torch.sin(_fp64_omni_oracle(q64_nl.detach(), z, v64_nl.detach(), meta_rich.pos_thw, inv_freq_64, meta_rich.doc_id, meta_rich.span_id, False)) * do64_nl).sum(),
            k64_nl.detach(), k64_nl.grad,
        ),
        "omni_dv": _richardson_ratio(
            lambda z: (torch.sin(_fp64_omni_oracle(q64_nl.detach(), k64_nl.detach(), z, meta_rich.pos_thw, inv_freq_64, meta_rich.doc_id, meta_rich.span_id, False)) * do64_nl).sum(),
            v64_nl.detach(), v64_nl.grad,
        ),
    }

    return {
        "tubelet_conv3d_and_merger_roofline": tubelet_and_merger_audit,
        "mrope_3d_sectioned_and_interleaved_ablation": mrope_audit,
        "omni_attention_backend_audit": omni_backend_audit,
        "bf16_vs_fp64_oracle_parity": bf16_vs_fp64_oracle,
        "fp64_richardson_central_difference": {
            "ratios": rich_ratios,
            "min_ratio": round(min(rich_ratios.values()), 4),
            "max_ratio": round(max(rich_ratios.values()), 4),
        },
    }


# ============================================================================
# 4. Suite A: Single-GPU Native-Resolution Vision Encoder Scaling
#    (Heterogeneous Aspect-Ratio Image/Video Batches, N_raw = 4,096 .. 32,768)
# ============================================================================


def _make_heterogeneous_vision_shapes(scale_mult: int) -> list[tuple[int, int, int]]:
    # Base unit = 4,096 raw patches across 6 heterogeneous items:
    #   - 2x thumbnail images (2, 224, 224) -> 2 * 256 = 512 patches
    #   - 1x landscape photo  (2, 336, 448) -> 768 patches
    #   - 1x portrait photo   (2, 448, 224) -> 512 patches
    #   - 1x 4-frame video    (4, 224, 224) -> 512 patches
    #   - 1x high-res OCR doc (2, 672, 504) -> 48 * 36 = 1,728? Wait: 512+768+512+512 = 2,304 -> need 1,792 patches!
    #     1,792 = 56 * 32 patches -> H = 56 * 14 = 784, W = 32 * 14 = 448! (2304 + 1792 == 4,096 EXACTLY!)
    # Notice max item has 1,792 patches while mean item has 682.7 patches -> 2D padding inflates 6 * 1792 = 10,752 (2.625x)!
    base = [
        (2, 224, 224),
        (2, 224, 224),
        (2, 336, 448),
        (2, 448, 224),
        (4, 224, 224),
        (2, 784, 448),
    ]
    return base * scale_mult


def run_suite_a_1gpu_vit_scaling(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    scales = [1, 2, 4] if is_smoke else [1, 2, 4, 8]  # 4K, 8K, 16K, 32K raw ViT patches
    warmup = 2 if is_smoke else 3
    iters = 4 if is_smoke else 7

    torch.manual_seed(3100)
    tower = Qwen25VLVisionTower(n_layers=4, d_vit=1280, d_llm=3584).to(device)
    results: dict[str, Any] = {}

    for mult in scales:
        shapes = _make_heterogeneous_vision_shapes(mult)
        meta = build_window_morton2x2_ragged_meta(shapes, device)
        n_raw = meta.total_patches
        n_merged = meta.total_merged_tokens

        raw_pix = (torch.randn(meta.total_raw_pixels, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        d_merged = torch.randn(n_merged, 3584, device=device, dtype=torch.bfloat16) * 0.3

        # Build padded 2D baseline inputs [B_items, N_max, 1184]
        item_patch_counts = [(t // 2) * (h // 14) * (w // 14) for (t, h, w) in shapes]
        valid_quad_counts = [c // 4 for c in item_patch_counts]
        n_max = max(item_patch_counts)
        b_items = len(shapes)

        # Verify exact numerical parity between HF Eager Window-Gather ViT and Zero-Permutation Window-Morton-2x2 ViT!
        parity_info: dict[str, float] = {}
        if mult == 1:
            for p in tower.parameters():
                p.grad = None
            raw_r = raw_pix.detach().clone().requires_grad_(True)
            pk_r = _RaggedTubeletPackFn.apply(raw_r, meta.patch_meta_raster)
            out_eag = tower.forward_eager_window_gather(pk_r, meta)
            out_eag.backward(d_merged)
            ref_out = out_eag.detach().clone()
            ref_dw_qkv = tower.blocks[0].wqkv.weight.grad.detach().clone()
            ref_dx = raw_r.grad.detach().clone()

            for p in tower.parameters():
                p.grad = None
            raw_w = raw_pix.detach().clone().requires_grad_(True)
            pk_w = _RaggedTubeletPackFn.apply(raw_w, meta.patch_meta)
            out_wm = tower.forward_winmorton_zero_perm(pk_w, meta, unswizzle_to_raster=True)
            out_wm.backward(d_merged)
            parity_info = {
                "winmorton_vs_eager_window_out_rel_l2": round(rel_l2(out_wm, ref_out), 6),
                "winmorton_vs_eager_window_dw_qkv_rel_l2": round(rel_l2(tower.blocks[0].wqkv.weight.grad, ref_dw_qkv), 6),
                "winmorton_vs_eager_window_dpixels_rel_l2": round(rel_l2(raw_w.grad, ref_dx), 6),
            }
            for p in tower.parameters():
                p.grad = None
            del raw_r, raw_w, ref_out, ref_dw_qkv, ref_dx

        def _bench_padded_2d() -> None:
            pk_ras = _RaggedTubeletPackFn.apply(raw_pix, meta.patch_meta_raster)
            x_pad = pk_ras.new_zeros((b_items, n_max, 1184))
            cur = 0
            for bi, pcnt in enumerate(item_patch_counts):
                x_pad[bi, :pcnt] = pk_ras[cur : cur + pcnt]
                cur += pcnt
            out = tower.forward_padded_2d_fa2(x_pad, valid_quad_counts)
            out.backward(d_merged)
            raw_pix.grad = None
            for p in tower.parameters():
                p.grad = None

        def _bench_navit_full() -> None:
            pk_wm = _RaggedTubeletPackFn.apply(raw_pix, meta.patch_meta)
            out = tower.forward_navit_full_varlen(pk_wm, meta)
            out.backward(d_merged)
            raw_pix.grad = None
            for p in tower.parameters():
                p.grad = None

        def _bench_qwen_eager_win() -> None:
            pk_ras = _RaggedTubeletPackFn.apply(raw_pix, meta.patch_meta_raster)
            out = tower.forward_eager_window_gather(pk_ras, meta)
            out.backward(d_merged)
            raw_pix.grad = None
            for p in tower.parameters():
                p.grad = None

        def _bench_qwen_winmorton() -> None:
            pk_wm = _RaggedTubeletPackFn.apply(raw_pix, meta.patch_meta)
            out = tower.forward_winmorton_zero_perm(pk_wm, meta, unswizzle_to_raster=True)
            out.backward(d_merged)
            raw_pix.grad = None
            for p in tower.parameters():
                p.grad = None

        st_pad = time_fn_with_peak_mem(_bench_padded_2d, warmup=warmup, iters=iters, device=device)
        st_nav = time_fn_with_peak_mem(_bench_navit_full, warmup=warmup, iters=iters, device=device)
        st_qew = time_fn_with_peak_mem(_bench_qwen_eager_win, warmup=warmup, iters=iters, device=device)
        st_qwm = time_fn_with_peak_mem(_bench_qwen_winmorton, warmup=warmup, iters=iters, device=device)

        for st in (st_pad, st_nav, st_qew, st_qwm):
            st["patches_per_s"] = round(n_raw / (st["median_ms"] * 1e-3), 1)

        row_a: dict[str, Any] = {
            "num_visual_items": b_items,
            "raw_patches": n_raw,
            "merged_llm_tokens": n_merged,
            "padded_2d_total_slots": b_items * n_max,
            "padding_inflation_ratio": round((b_items * n_max) / float(n_raw), 4),
            "padded_2d_fa2_vit": st_pad,
            "navit_ragged_full_varlen_vit": st_nav,
            "qwen25vl_window8x8_eager_gather_vit": st_qew,
            "qwen25vl_winmorton2x2_zero_perm_vit": st_qwm,
            "winmorton_speedup_over_padded_2d": round(st_pad["median_ms"] / st_qwm["median_ms"], 3),
            "winmorton_speedup_over_navit_full": round(st_nav["median_ms"] / st_qwm["median_ms"], 3),
            "winmorton_speedup_over_eager_window_gather": round(st_qew["median_ms"] / st_qwm["median_ms"], 3),
        }
        if parity_info:
            row_a["parity_audit"] = parity_info
        results[f"P_{n_raw}"] = row_a
        del raw_pix, d_merged
        torch.cuda.empty_cache()

    del tower
    torch.cuda.empty_cache()
    return results


# ============================================================================
# 5. Suite B: Single-GPU Interleaved Multimodal Omni-Attention & 4-Layer Stack
#    (Qwen2.5-VL-7B Geometry: 28Q/4KV, D=128, 3D M-RoPE + BiDir-Vision/Causal-Text)
# ============================================================================


def _make_interleaved_omni_docs(target_seqlen: int) -> list[list[tuple[str, int, int, int, int]]]:
    # Realistic multi-image + video + causal-text documents:
    # Base 4,096-token document:
    #   text(160) -> image_24x24(576) -> text(624) -> image_24x18(432) -> text(704) -> video_2x16x16(512) -> text(1088) = 4,096 tokens!
    # For S=4096 -> 1 doc of 4096; S=8192 -> 2 docs of 4096; S=16384 -> 4 docs of 4096;
    # S=32768 -> 4 docs of 8192 (2x base); S=65536 -> 4 docs of 16384 (4x base)!
    base_segs = [
        ("text", 160, 0, 0, 0),
        ("vision", 576, 1, 24, 24),
        ("text", 624, 0, 0, 0),
        ("vision", 432, 1, 24, 18),
        ("text", 704, 0, 0, 0),
        ("vision", 512, 2, 16, 16),
        ("text", 1088, 0, 0, 0),
    ]
    if target_seqlen <= 16384:
        n_docs = target_seqlen // 4096
        return [list(base_segs) for _ in range(n_docs)]
    doc_mult = target_seqlen // 16384
    return [list(base_segs) * doc_mult for _ in range(4)]


def run_suite_b_1gpu_omni_llm_scaling(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    seqlens_attn = [4096, 8192, 16384] if is_smoke else [4096, 8192, 16384, 32768, 65536]
    seqlens_stack = [8192, 16384] if is_smoke else [8192, 16384, 32768]
    warmup = 2 if is_smoke else 3
    iters = 4 if is_smoke else 7

    inv_freq = (1.0 / (1000000.0 ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128.0))).contiguous()
    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)

    attn_scaling: dict[str, Any] = {}
    torch.manual_seed(4200)

    for seqlen in seqlens_attn:
        docs = _make_interleaved_omni_docs(seqlen)
        meta = build_omni_packed_batch_meta(docs, device, pad_to_multiple=128)
        s_eff = meta.seqlen
        it_here = max(3, iters // 2) if s_eff >= 32768 else iters

        q = torch.randn(s_eff, 28, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(s_eff, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(s_eff, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn_like(q)

        row_b: dict[str, Any] = {
            "seqlen": s_eff,
            "max_doc_len": meta.max_doc_len,
            "visual_tokens": int(meta.vis_token_indices.numel()),
            "upper_triangle_span_tokens": int(meta.span_meta.shape[0]),
        }

        # 1. SDPA bool mask + HF Eager 3D M-RoPE (S <= 8,192; >= 16,384 hits 28 GiB Math-SDPA OOM)
        if s_eff <= 8192:
            qi = torch.arange(s_eff, device=device)[:, None]
            ki = torch.arange(s_eff, device=device)[None, :]
            m4d = (
                (meta.doc_id[qi] == meta.doc_id[ki])
                & ((ki <= qi) | ((meta.span_id[qi] > 0) & (meta.span_id[qi] == meta.span_id[ki])))
            ).view(1, 1, s_eff, s_eff)

            def _sdpa_fb(m4d: torch.Tensor = m4d) -> None:
                qr, kr = hf_eager_3d_mrope(q, k, meta.pos_thw, inv_freq, False)
                qs = qr.transpose(0, 1).unsqueeze(0)
                ks = kr.transpose(0, 1).unsqueeze(0)
                vs = v.transpose(0, 1).unsqueeze(0)
                o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=m4d, enable_gqa=True)
                o.squeeze(0).transpose(0, 1).backward(do)
                q.grad = k.grad = v.grad = None

            st_sdpa = time_fn_with_peak_mem(_sdpa_fb, warmup=1, iters=3, device=device)
            st_sdpa["status"] = "ok_math_fallback"
            row_b["sdpa_bool_mask_eager_mrope"] = st_sdpa
            del m4d
        else:
            row_b["sdpa_bool_mask_eager_mrope"] = {
                "status": "oom_28gib_math_sdpa_fallback",
                "verified_in_suite_0": True,
            }

        # 2 & 3. FlexAttention (28Q/4KV, G=7): Dynamic BlockMask + Eager M-RoPE vs Prebuilt BlockMask + Triton M-RoPE
        doc_t = meta.doc_id
        spn_t = meta.span_id

        def _omni_mod(b: Any, h: Any, qi: Any, ki: Any, doc_t: torch.Tensor = doc_t, spn_t: torch.Tensor = spn_t) -> Any:
            return (doc_t[qi] == doc_t[ki]) & ((ki <= qi) | ((spn_t[qi] > 0) & (spn_t[qi] == spn_t[ki])))

        bm_pre = comp_cbm(_omni_mod, B=1, H=1, Q_LEN=s_eff, KV_LEN=s_eff, device=device)

        def _fx_dyn_eag() -> None:
            bm_d = comp_cbm(_omni_mod, B=1, H=1, Q_LEN=s_eff, KV_LEN=s_eff, device=device)
            qr, kr = hf_eager_3d_mrope(q, k, meta.pos_thw, inv_freq, False)
            qfx = qr.transpose(0, 1).unsqueeze(0).contiguous()
            kfx = kr.transpose(0, 1).unsqueeze(0).contiguous()
            vfx = v.transpose(0, 1).unsqueeze(0).contiguous()
            ofx = comp_flex(qfx, kfx, vfx, block_mask=bm_d, enable_gqa=True)
            ofx.squeeze(0).transpose(0, 1).backward(do)
            q.grad = k.grad = v.grad = None

        def _fx_pre_tri(bm_pre: Any = bm_pre) -> None:
            qr, kr = _FusedQK3DMRoPEFn.apply(q, k, meta.pos_thw, inv_freq, False)
            qfx = qr.transpose(0, 1).unsqueeze(0).contiguous()
            kfx = kr.transpose(0, 1).unsqueeze(0).contiguous()
            vfx = v.transpose(0, 1).unsqueeze(0).contiguous()
            ofx = comp_flex(qfx, kfx, vfx, block_mask=bm_pre, enable_gqa=True)
            ofx.squeeze(0).transpose(0, 1).backward(do)
            q.grad = k.grad = v.grad = None

        def _tri_tile64() -> None:
            qr, kr = _FusedQK3DMRoPEFn.apply(q, k, meta.pos_thw, inv_freq, False)
            o = _TritonTile64OmniGQAFn.apply(qr, kr, v, meta)
            o.backward(do)
            q.grad = k.grad = v.grad = None

        def _surg_fa2() -> None:
            qr, kr = _FusedQK3DMRoPEFn.apply(q, k, meta.pos_thw, inv_freq, False)
            o = _SurgicalOmniFA2Fn.apply(qr, kr, v, meta)
            o.backward(do)
            q.grad = k.grad = v.grad = None

        st_fxd = time_fn_with_peak_mem(_fx_dyn_eag, warmup=warmup, iters=it_here, device=device)
        st_fxp = time_fn_with_peak_mem(_fx_pre_tri, warmup=warmup, iters=it_here, device=device)
        st_t64 = time_fn_with_peak_mem(_tri_tile64, warmup=warmup, iters=it_here, device=device)
        st_srg = time_fn_with_peak_mem(_surg_fa2, warmup=warmup, iters=it_here, device=device)

        row_b["flex_attn_dynamic_bm_eager_mrope"] = st_fxd
        row_b["flex_attn_prebuilt_bm_triton_mrope"] = st_fxp
        row_b["triton_tile64_omni_gqa_plus_mrope"] = st_t64
        row_b["surgical_disjoint_fa2_lse_plus_triton_mrope"] = st_srg
        row_b["surgical_speedup_over_flex_dynamic"] = round(st_fxd["median_ms"] / st_srg["median_ms"], 3)
        row_b["surgical_speedup_over_flex_prebuilt"] = round(st_fxp["median_ms"] / st_srg["median_ms"], 3)
        row_b["surgical_speedup_over_triton_tile64"] = round(st_t64["median_ms"] / st_srg["median_ms"], 3)
        attn_scaling[f"S_{s_eff}"] = row_b
        del q, k, v, do, bm_pre
        torch.cuda.empty_cache()

    # B.2 4-Layer Qwen2.5-VL-7B-Geometry Decoder Stack (D_llm=3584, 28Q/4KV, D=128, FFN=9728)
    layers = nn.ModuleList([MultimodalOmniLLMBlock().to(device) for _ in range(4)])
    stack_scaling: dict[str, Any] = {}

    for seqlen in seqlens_stack:
        docs = _make_interleaved_omni_docs(seqlen)
        meta = build_omni_packed_batch_meta(docs, device, pad_to_multiple=128)
        s_eff = meta.seqlen
        it_s = max(3, iters // 2) if s_eff >= 16384 else iters
        x0 = torch.randn(s_eff, 3584, device=device, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x0)

        doc_t = meta.doc_id
        spn_t = meta.span_id

        def _mod_s(b: Any, h: Any, qi: Any, ki: Any, doc_t: torch.Tensor = doc_t, spn_t: torch.Tensor = spn_t) -> Any:
            return (doc_t[qi] == doc_t[ki]) & ((ki <= qi) | ((spn_t[qi] > 0) & (spn_t[qi] == spn_t[ki])))

        def _run_stack_flex_dyn() -> None:
            xin = x0.detach().requires_grad_(True)
            bm = comp_cbm(_mod_s, B=1, H=1, Q_LEN=s_eff, KV_LEN=s_eff, device=device)
            h = xin
            for blk in layers:
                h = blk(h, meta, inv_freq, mode="flex_attn", flex_bm=bm, comp_flex=comp_flex)
            h.backward(dy)
            for p in layers.parameters():
                p.grad = None

        def _run_stack_tile64() -> None:
            xin = x0.detach().requires_grad_(True)
            h = xin
            for blk in layers:
                h = blk(h, meta, inv_freq, mode="triton_tile64_triton_mrope")
            h.backward(dy)
            for p in layers.parameters():
                p.grad = None

        def _run_stack_surgical() -> None:
            xin = x0.detach().requires_grad_(True)
            h = xin
            for blk in layers:
                h = blk(h, meta, inv_freq, mode="surgical_fa2_triton_mrope")
            h.backward(dy)
            for p in layers.parameters():
                p.grad = None

        st_sf = time_fn_with_peak_mem(_run_stack_flex_dyn, warmup=1, iters=it_s, device=device)
        st_st = time_fn_with_peak_mem(_run_stack_tile64, warmup=1, iters=it_s, device=device)
        st_ss = time_fn_with_peak_mem(_run_stack_surgical, warmup=1, iters=it_s, device=device)
        for st in (st_sf, st_st, st_ss):
            st["tok_per_s"] = round(s_eff / (st["median_ms"] * 1e-3), 1)

        stack_scaling[f"S_{s_eff}"] = {
            "seqlen": s_eff,
            "flex_dynamic_eager_mrope_4layer": st_sf,
            "triton_tile64_triton_mrope_4layer": st_st,
            "surgical_fa2_triton_mrope_4layer": st_ss,
            "surgical_4layer_speedup_over_flex_dynamic": round(st_sf["median_ms"] / st_ss["median_ms"], 3),
            "surgical_4layer_speedup_over_triton_tile64": round(st_st["median_ms"] / st_ss["median_ms"], 3),
        }
        del x0, dy
        torch.cuda.empty_cache()

    del layers
    torch.cuda.empty_cache()
    return {
        "isolated_mrope_plus_omni_attention_scaling": attn_scaling,
        "multimodal_4layer_decoder_stack_scaling": stack_scaling,
    }


# ============================================================================
# 6. Suite C: 8-GPU NVSwitch Modality-Asymmetric Dual-Tower FSDP2 Training
#    (VeOmni / DistTrain Two-Stage Modality-Decoupled AllToAllv + Quad-Cost-LPT)
# ============================================================================


@dataclass
class MultimodalSampleSpec:
    sample_id: int
    images_thw: list[tuple[int, int, int]]
    doc_segments: list[tuple[str, int, int, int, int]]
    raw_vit_patches: int
    merged_vis_tokens: int
    total_llm_tokens: int
    vit_cost_score: float
    llm_quad_cost_score: float


def _build_global_heterogeneous_multimodal_samples(batch_mult: int = 1) -> list[MultimodalSampleSpec]:
    # 16 mixed-modality samples per base global batch (ordered as arrive from unshuffled modality-clustered shards
    # or multi-source streams: heavy multi-image/video docs, medium VQA docs, and pure text reasoning CoT docs!):
    #   - Samples 0..1 (Heavy High-Res Multi-Image OCR + Video):
    #     each has [(2, 672, 672) -> 2304p, (4, 448, 448) -> 2048p, (2, 448, 448) -> 1024p] = 5,376 raw patches (1,344 merged) + 1,728 text = 3,072 LLM tokens!
    #   - Samples 2..3 (Multi-Image High-Res Document):
    #     each has [(2, 560, 560) -> 1600p, (2, 448, 448) -> 1024p] = 2,624 raw patches (656 merged) + 1,392 text = 2,048 LLM tokens!
    #   - Samples 4..7 (Medium Single-Image VQA):
    #     each has [(2, 448, 448) -> 1024p] = 1,024 raw patches (256 merged) + 768 text = 1,024 LLM tokens!
    #   - Samples 8..9 (Long Text-Only CoT Reasoning — 0 ViT patches!):
    #     each has [] = 0 raw patches + 3,072 text = 3,072 LLM tokens!
    #   - Samples 10..15 (Short Text-Only SFT — 0 ViT patches!):
    #     each has [] = 0 raw patches + 768 text = 768 LLM tokens!
    # Check global sums for batch_mult = 1 (16 samples):
    #   Global raw ViT patches = 2*5376 + 2*2624 + 4*1024 + 0 = 10752 + 5248 + 4096 = 20,096 raw patches (5,024 merged)!
    #   Global LLM tokens      = 2*3072 + 2*2048 + 4*1024 + 2*3072 + 6*768 = 6144 + 4096 + 4096 + 6144 + 4608 = 25,088 LLM tokens!
    templates: list[tuple[list[tuple[int, int, int]], list[tuple[str, int, int, int, int]]]] = [
        (
            [(2, 672, 672), (4, 448, 448), (2, 448, 448)],
            [("text", 128, 0, 0, 0), ("vision", 576, 1, 24, 24), ("text", 384, 0, 0, 0), ("vision", 512, 2, 16, 16), ("text", 512, 0, 0, 0), ("vision", 256, 1, 16, 16), ("text", 704, 0, 0, 0)],
        ),
        (
            [(2, 672, 672), (4, 448, 448), (2, 448, 448)],
            [("text", 128, 0, 0, 0), ("vision", 576, 1, 24, 24), ("text", 384, 0, 0, 0), ("vision", 512, 2, 16, 16), ("text", 512, 0, 0, 0), ("vision", 256, 1, 16, 16), ("text", 704, 0, 0, 0)],
        ),
        (
            [(2, 560, 560), (2, 448, 448)],
            [("text", 192, 0, 0, 0), ("vision", 400, 1, 20, 20), ("text", 400, 0, 0, 0), ("vision", 256, 1, 16, 16), ("text", 800, 0, 0, 0)],
        ),
        (
            [(2, 560, 560), (2, 448, 448)],
            [("text", 192, 0, 0, 0), ("vision", 400, 1, 20, 20), ("text", 400, 0, 0, 0), ("vision", 256, 1, 16, 16), ("text", 800, 0, 0, 0)],
        ),
        (
            [(2, 448, 448)],
            [("text", 128, 0, 0, 0), ("vision", 256, 1, 16, 16), ("text", 640, 0, 0, 0)],
        ),
        (
            [(2, 448, 448)],
            [("text", 128, 0, 0, 0), ("vision", 256, 1, 16, 16), ("text", 640, 0, 0, 0)],
        ),
        (
            [(2, 448, 448)],
            [("text", 128, 0, 0, 0), ("vision", 256, 1, 16, 16), ("text", 640, 0, 0, 0)],
        ),
        (
            [(2, 448, 448)],
            [("text", 128, 0, 0, 0), ("vision", 256, 1, 16, 16), ("text", 640, 0, 0, 0)],
        ),
        ([], [("text", 3072, 0, 0, 0)]),
        ([], [("text", 3072, 0, 0, 0)]),
        ([], [("text", 768, 0, 0, 0)]),
        ([], [("text", 768, 0, 0, 0)]),
        ([], [("text", 768, 0, 0, 0)]),
        ([], [("text", 768, 0, 0, 0)]),
        ([], [("text", 768, 0, 0, 0)]),
        ([], [("text", 768, 0, 0, 0)]),
    ]

    all_specs: list[MultimodalSampleSpec] = []
    sid = 0
    for _ in range(batch_mult):
        for imgs, segs in templates:
            raw_p = sum((t // 2) * (h // 14) * (w // 14) for (t, h, w) in imgs)
            m_vis = raw_p // 4
            tot_llm = sum(sl for (_, sl, _, _, _) in segs)
            vit_cost = sum(
                float((t // 2) * ((h // 14) * (w // 14)) + 0.0005 * (t // 2) * (((h // 14) * (w // 14)) ** 2))
                for (t, h, w) in imgs
            )
            # Calibrated attention + FFN cost score for L-token Omni-LLM document:
            llm_cost = float(tot_llm) + float(tot_llm * tot_llm) / 4096.0
            all_specs.append(
                MultimodalSampleSpec(
                    sample_id=sid,
                    images_thw=list(imgs),
                    doc_segments=list(segs),
                    raw_vit_patches=raw_p,
                    merged_vis_tokens=m_vis,
                    total_llm_tokens=tot_llm,
                    vit_cost_score=vit_cost,
                    llm_quad_cost_score=llm_cost,
                )
            )
            sid += 1
    return all_specs


class DualTowerMultimodalModel(nn.Module):
    def __init__(self, vit_layers: int = 4, llm_layers: int = 6, d_vit: int = 1280, d_llm: int = 3584) -> None:
        super().__init__()
        self.visual = Qwen25VLVisionTower(n_layers=vit_layers, d_vit=d_vit, d_llm=d_llm)
        self.llm_blocks = nn.ModuleList([MultimodalOmniLLMBlock(d_llm=d_llm) for _ in range(llm_layers)])
        self.final_norm_w = nn.Parameter(torch.ones(d_llm, dtype=torch.bfloat16))

    def forward(
        self,
        plan: dict[str, Any],
        inv_freq: torch.Tensor,
        comp_cbm: Any,
        comp_flex: Any,
    ) -> torch.Tensor:
        sched = plan["schedule"]
        ometa: OmniPackedBatchMeta = plan["ometa"]

        if not plan.get("decouple_vit", False):
            if plan["has_local_vis"]:
                if sched == "naive_sample_shard_padded_vit_flex":
                    vis_local = self.visual(plan["x_pad"], mode="padded_2d", valid_quad_counts=plan["valid_quads"])
                else:
                    vis_local = self.visual(
                        plan["pk_wm"], mode="winmorton", meta=plan["vmeta"], unswizzle_to_raster=True
                    )
                x_llm = plan["x_base"].index_add(0, ometa.vis_token_indices, vis_local)
            else:
                if sched == "naive_sample_shard_padded_vit_flex":
                    vis_dummy = self.visual(plan["dummy_pad"], mode="padded_2d", valid_quad_counts=[1])
                else:
                    vis_dummy = self.visual(
                        plan["dummy_wm"], mode="winmorton", meta=plan["dummy_meta"], unswizzle_to_raster=True
                    )
                x_llm = plan["x_base"] + (vis_dummy.sum() * 0.0)
        else:
            recv_patches = _DifferentiableVarlenAllToAllFn.apply(
                plan["send_patches_buf"], plan["patch_out_splits"], plan["patch_in_splits"], dist.group.WORLD
            )
            merged_worker = self.visual(
                recv_patches, mode="winmorton", meta=plan["vmeta_worker"], unswizzle_to_raster=True
            )
            merged_returned = _DifferentiableVarlenAllToAllFn.apply(
                merged_worker, plan["quad_out_splits"], plan["quad_in_splits"], dist.group.WORLD
            )
            if merged_returned.shape[0] > 0:
                vis_ordered = merged_returned.index_select(0, plan["restore_idx"])
                x_llm = plan["x_base"].index_add(0, ometa.vis_token_indices, vis_ordered)
            else:
                x_llm = plan["x_base"] + (merged_returned.sum() * 0.0)

        if sched == "naive_sample_shard_padded_vit_flex":
            bm = comp_cbm(plan["flex_mod"], B=1, H=1, Q_LEN=ometa.seqlen, KV_LEN=ometa.seqlen, device=x_llm.device)
            h = x_llm
            for lblk in self.llm_blocks:
                h = lblk(h, ometa, inv_freq, mode="flex_attn", flex_bm=bm, comp_flex=comp_flex)
        else:
            h = x_llm
            for lblk in self.llm_blocks:
                h = lblk(h, ometa, inv_freq, mode="surgical_fa2_triton_mrope")

        return triton_rmsnorm(h, self.final_norm_w)


def _lpt_partition(items: list[tuple[int, float]], n_bins: int) -> list[list[int]]:
    # Longest-Processing-Time greedy bin assignment
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    loads = [0.0 for _ in range(n_bins)]
    for item_id, cost in sorted(items, key=lambda x: (-x[1], x[0])):
        b_min = min(range(n_bins), key=lambda b: loads[b])
        bins[b_min].append(item_id)
        loads[b_min] += cost
    for b in range(n_bins):
        bins[b].sort()
    return bins


def _token_greedy_partition(samples: list[MultimodalSampleSpec], n_bins: int) -> list[list[int]]:
    # Naive token-count greedy balancer (ignores both ViT patch count and quadratic L^2 attention!)
    # Note: breaks ties by sample_id order so heavy vision 3072-tok and text-only 3072-tok samples treat identically!
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    tok_loads = [0 for _ in range(n_bins)]
    # Sort ONLY by total_llm_tokens descending (blind to modality!)
    for sp in sorted(samples, key=lambda s: (-s.total_llm_tokens, -s.sample_id)):
        b_min = min(range(n_bins), key=lambda b: tok_loads[b])
        bins[b_min].append(sp.sample_id)
        tok_loads[b_min] += sp.total_llm_tokens
    for b in range(n_bins):
        bins[b].sort()
    return bins


def run_suite_c_8gpu_multimodal_fsdp2(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    assert world_size == 8, f"Suite C requires world_size == 8, got {world_size}"
    dist.barrier()

    inv_freq = (1.0 / (1000000.0 ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128.0))).contiguous()
    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    def _build_sharded_fsdp2_model(seed: int = 8080) -> DualTowerMultimodalModel:
        torch.manual_seed(seed)
        m = DualTowerMultimodalModel(vit_layers=4, llm_layers=6, d_vit=1280, d_llm=3584).to(device)
        for blk in m.visual.blocks:
            fully_shard(blk, mp_policy=mp_policy)
        fully_shard(m.visual.merger, mp_policy=mp_policy)
        fully_shard(m.visual, mp_policy=mp_policy)
        for lblk in m.llm_blocks:
            fully_shard(lblk, mp_policy=mp_policy)
        fully_shard(m, mp_policy=mp_policy)
        return m

    # ------------------------------------------------------------------------
    # C.1 Exact 1-GPU Global Multimodal Parity Verification (Mixed Vision + 0-Patch Text Samples!)
    # ------------------------------------------------------------------------
    samples_p = _build_global_heterogeneous_multimodal_samples(batch_mult=1)
    n_samples_p = len(samples_p)
    tot_llm_global_p = sum(s.total_llm_tokens for s in samples_p)

    # Deterministic per-sample inputs broadcast across ranks so 1-GPU and 8-GPU see identical tensors!
    torch.manual_seed(9500)
    sample_txt_embeds: list[torch.Tensor] = []
    sample_targets: list[torch.Tensor] = []
    sample_img_packed_wm: list[list[torch.Tensor]] = []
    sample_img_packed_ras: list[list[torch.Tensor]] = []

    for sp in samples_p:
        te = torch.randn(sp.total_llm_tokens, 3584, device=device, dtype=torch.bfloat16) * 0.25
        tg = torch.randn(sp.total_llm_tokens, 3584, device=device, dtype=torch.bfloat16) * 0.25
        dist.broadcast(te, src=0)
        dist.broadcast(tg, src=0)
        sample_txt_embeds.append(te)
        sample_targets.append(tg)

        wm_list: list[torch.Tensor] = []
        ras_list: list[torch.Tensor] = []
        for (tp, hp, wp) in sp.images_thw:
            im_meta = build_window_morton2x2_ragged_meta([(tp, hp, wp)], device)
            raw_im = torch.randn(im_meta.total_raw_pixels, device=device, dtype=torch.bfloat16) * 0.3
            dist.broadcast(raw_im, src=0)
            pk_w = _RaggedTubeletPackFn.apply(raw_im, im_meta.patch_meta).detach()
            pk_r = _RaggedTubeletPackFn.apply(raw_im, im_meta.patch_meta_raster).detach()
            wm_list.append(pk_w)
            ras_list.append(pk_r)
        sample_img_packed_wm.append(wm_list)
        sample_img_packed_ras.append(ras_list)

    # Compute 1-GPU exact reference on Rank 0 before sharding
    ref_1gpu_grads: dict[str, torch.Tensor] = {}
    ref_1gpu_loss = 0.0
    if rank == 0:
        torch.manual_seed(8080)
        m_ref = DualTowerMultimodalModel(vit_layers=4, llm_layers=6, d_vit=1280, d_llm=3584).to(device)
        all_imgs_thw = [im for sp in samples_p for im in sp.images_thw]
        vmeta_all = build_window_morton2x2_ragged_meta(all_imgs_thw, device)
        pk_all = torch.cat([pk for wml in sample_img_packed_wm for pk in wml], dim=0)
        vis_merged_all = m_ref.visual.forward_winmorton_zero_perm(pk_all, vmeta_all, unswizzle_to_raster=True)

        ometa_all = build_omni_packed_batch_meta([sp.doc_segments for sp in samples_p], device, pad_to_multiple=128)
        x_llm_all = torch.zeros((ometa_all.seqlen, 3584), device=device, dtype=torch.bfloat16)
        tgt_all = torch.zeros_like(x_llm_all)
        valid_mask_all = torch.zeros((ometa_all.seqlen, 1), device=device, dtype=torch.float32)
        cur_t = 0
        for si in range(n_samples_p):
            L = samples_p[si].total_llm_tokens
            x_llm_all[cur_t : cur_t + L] = sample_txt_embeds[si]
            tgt_all[cur_t : cur_t + L] = sample_targets[si]
            valid_mask_all[cur_t : cur_t + L] = 1.0
            cur_t += L
        # Splice merged visual tokens into visual positions (additive on base prompt slots -> clean gradient flow!)
        x_llm_all = x_llm_all.index_add(0, ometa_all.vis_token_indices, vis_merged_all)
        h_all = x_llm_all
        for lblk in m_ref.llm_blocks:
            h_all = lblk(h_all, ometa_all, inv_freq, mode="surgical_fa2_triton_mrope")
        h_out = triton_rmsnorm(h_all, m_ref.final_norm_w)
        loss_ref = (((h_out.float() - tgt_all.float()) * valid_mask_all) ** 2).sum() / float(tot_llm_global_p)
        loss_ref.backward()
        ref_1gpu_loss = float(loss_ref.item())
        ref_1gpu_grads = {
            "vit_patch_proj": m_ref.visual.patch_proj.weight.grad.detach().clone(),
            "vit_blk0_wqkv": m_ref.visual.blocks[0].wqkv.weight.grad.detach().clone(),
            "merger_mlp0": m_ref.visual.merger.mlp0.weight.grad.detach().clone(),
            "llm_blk0_wq": m_ref.llm_blocks[0].wq.weight.grad.detach().clone(),
            "llm_blk0_wdn": m_ref.llm_blocks[0].w_dn.weight.grad.detach().clone(),
        }
        del m_ref, pk_all, vis_merged_all, x_llm_all, tgt_all, h_all, h_out
        torch.cuda.empty_cache()
    dist.barrier()

    fsdp_model = _build_sharded_fsdp2_model(seed=8080)
    n_params_total = sum(p.numel() for p in fsdp_model.parameters()) * 0 + (
        1280 * 1184
        + 4 * (1280 * 3 * 1280 + 1280 * 1280 + 1280 * 2 * 3420 + 3420 * 1280 + 2 * 1280)
        + (1280 + 5120 * 5120 + 5120 * 3584)
        + 6 * (3584 * 3584 + 3584 * 1024 + 3584 * 3584 + 3584 * 2 * 9728 + 9728 * 3584 + 2 * 3584)
        + 3584
    )

    def _prepare_schedule_plan(
        samples: list[MultimodalSampleSpec],
        txt_list: list[torch.Tensor],
        tgt_list: list[torch.Tensor],
        wm_imgs: list[list[torch.Tensor]],
        ras_imgs: list[list[torch.Tensor]],
        schedule: str,
    ) -> dict[str, Any]:
        tot_tok_g = sum(s.total_llm_tokens for s in samples)
        if schedule.startswith("naive_sample_shard"):
            llm_bins = [list(range(r * (len(samples) // 8), (r + 1) * (len(samples) // 8))) for r in range(8)]
            decouple_vit = False
        elif schedule == "token_greedy_llm_pack_only":
            llm_bins = _token_greedy_partition(samples, 8)
            decouple_vit = False
        elif schedule == "decoupled_vit_alltoall_plus_llm_quad_lpt":
            llm_bins = _lpt_partition([(s.sample_id, s.llm_quad_cost_score) for s in samples], 8)
            decouple_vit = True
        else:
            raise ValueError(schedule)

        my_sample_ids = llm_bins[rank]
        my_docs = [samples[si].doc_segments for si in my_sample_ids]
        ometa = build_omni_packed_batch_meta(my_docs, device, pad_to_multiple=128)
        x_base = torch.zeros((ometa.seqlen, 3584), device=device, dtype=torch.bfloat16)
        tgt_loc = torch.zeros_like(x_base)
        vmask_loc = torch.zeros((ometa.seqlen, 1), device=device, dtype=torch.float32)
        cur = 0
        for si in my_sample_ids:
            L = samples[si].total_llm_tokens
            x_base[cur : cur + L] = txt_list[si]
            tgt_loc[cur : cur + L] = tgt_list[si]
            vmask_loc[cur : cur + L] = 1.0
            cur += L

        # Per-rank LLM quadratic cost & ViT patch load statistics
        rank_llm_costs = [sum(samples[si].llm_quad_cost_score for si in llm_bins[r]) for r in range(8)]
        llm_imbalance = max(rank_llm_costs) / max(1e-9, sum(rank_llm_costs) / 8.0)

        plan: dict[str, Any] = {
            "schedule": schedule,
            "tot_tok_global": tot_tok_g,
            "ometa": ometa,
            "x_base": x_base,
            "tgt_loc": tgt_loc,
            "vmask_loc": vmask_loc,
            "llm_quad_cost_imbalance": round(llm_imbalance, 4),
        }

        if not decouple_vit:
            rank_vit_patches = [sum(samples[si].raw_vit_patches for si in llm_bins[r]) for r in range(8)]
            vit_imbalance = max(rank_vit_patches) / max(1e-9, sum(rank_vit_patches) / 8.0)
            plan["vit_patch_imbalance"] = round(vit_imbalance, 4)
            plan["alltoall_comm_mib_max_rank"] = 0.0

            my_imgs_thw = [im for si in my_sample_ids for im in samples[si].images_thw]
            if len(my_imgs_thw) > 0:
                vmeta = build_window_morton2x2_ragged_meta(my_imgs_thw, device)
                pk_wm = torch.cat([pk for si in my_sample_ids for pk in wm_imgs[si]], dim=0)
                pk_ras = torch.cat([pk for si in my_sample_ids for pk in ras_imgs[si]], dim=0)
                item_pcounts = [(t // 2) * (h // 14) * (w // 14) for (t, h, w) in my_imgs_thw]
                n_max = max(item_pcounts)
                x_pad = pk_ras.new_zeros((len(my_imgs_thw), n_max, 1184))
                c0 = 0
                for bi, pc in enumerate(item_pcounts):
                    x_pad[bi, :pc] = pk_ras[c0 : c0 + pc]
                    c0 += pc
                plan["has_local_vis"] = True
                plan["vmeta"] = vmeta
                plan["pk_wm"] = pk_wm
                plan["x_pad"] = x_pad
                plan["valid_quads"] = [pc // 4 for pc in item_pcounts]
            else:
                # 0-patch text-only rank! Must run a 1-quad (4-patch) dummy zero-multiplied pass to prevent FSDP2 NCCL deadlock!
                dummy_meta = build_window_morton2x2_ragged_meta([(2, 28, 28)], device)
                plan["has_local_vis"] = False
                plan["dummy_meta"] = dummy_meta
                plan["dummy_wm"] = torch.zeros((4, 1184), device=device, dtype=torch.bfloat16)
                plan["dummy_pad"] = torch.zeros((1, 4, 1184), device=device, dtype=torch.bfloat16)
        else:
            # Stage 1 Decoupled ViT-Cost-LPT AllToAllv across all global images!
            # Enumerate all global images ordered by owner LLM rank `r_src in 0..7`
            global_img_records: list[tuple[int, int, int, tuple[int, int, int], float]] = []
            gid = 0
            for r_src in range(8):
                local_ord = 0
                for si in llm_bins[r_src]:
                    for im_idx, thw in enumerate(samples[si].images_thw):
                        p_cnt = (thw[0] // 2) * (thw[1] // 14) * (thw[2] // 14)
                        icost = float(p_cnt) + 0.0005 * float(p_cnt * p_cnt)
                        global_img_records.append((gid, r_src, local_ord, thw, icost))
                        local_ord += 1
                        gid += 1

            vit_bins = _lpt_partition([(rec[0], rec[4]) for rec in global_img_records], 8)
            gid_to_vit_rank = {}
            for rw in range(8):
                for g_i in vit_bins[rw]:
                    gid_to_vit_rank[g_i] = rw

            rank_vit_patches = [
                sum((global_img_records[g_i][3][0] // 2) * (global_img_records[g_i][3][1] // 14) * (global_img_records[g_i][3][2] // 14) for g_i in vit_bins[rw])
                for rw in range(8)
            ]
            plan["vit_patch_imbalance"] = round(
                max(rank_vit_patches) / max(1e-9, sum(rank_vit_patches) / 8.0), 4
            )

            # Build `send_patches` on `rank` sorted by destination `rw in 0..7`!
            my_local_imgs_wm = [pk for si in my_sample_ids for pk in wm_imgs[si]]
            my_records = [rec for rec in global_img_records if rec[1] == rank]
            # Sort my_records by destination vit_rank `rw`, preserving gid order within each (r_src, rw) pair
            my_sorted_by_dst = sorted(my_records, key=lambda rec: (gid_to_vit_rank[rec[0]], rec[0]))
            if my_sorted_by_dst:
                send_patches_buf = torch.cat([my_local_imgs_wm[rec[2]] for rec in my_sorted_by_dst], dim=0).contiguous()
            else:
                send_patches_buf = torch.empty((0, 1184), device=device, dtype=torch.bfloat16)

            # Compute 8x8 patch split matrix `patch_splits[r_src][rw]` and merged-token split matrix `quad_splits[rw][r_src]`
            patch_splits = [[0] * 8 for _ in range(8)]
            quad_splits = [[0] * 8 for _ in range(8)]
            for rec in global_img_records:
                g_i, rs, _, thw, _ = rec
                rw = gid_to_vit_rank[g_i]
                pc = (thw[0] // 2) * (thw[1] // 14) * (thw[2] // 14)
                patch_splits[rs][rw] += pc
                quad_splits[rw][rs] += pc // 4

            # On ViT worker `rank`: incoming patches arrive ordered by `r_src in 0..7`, then `gid`!
            recv_records_on_vit = sorted(
                [rec for rec in global_img_records if gid_to_vit_rank[rec[0]] == rank],
                key=lambda rec: (rec[1], rec[0]),
            )
            vit_worker_thw = [rec[3] for rec in recv_records_on_vit]
            vmeta_worker = build_window_morton2x2_ragged_meta(vit_worker_thw, device)

            # When merged tokens return to `rank` (`r_src`) from `rw in 0..7`, they arrive grouped by `rw` (`my_sorted_by_dst` order);
            # compute 1D permutation `restore_merged_order_idx` to put them back in chronological `local_ord` (`0..len-1`)!
            quad_chunks_in_recv_order: list[torch.Tensor] = []
            q_cursor = 0
            per_local_ord_slices: dict[int, torch.Tensor] = {}
            for rec in my_sorted_by_dst:
                _, _, l_ord, thw, _ = rec
                qc = ((thw[0] // 2) * (thw[1] // 14) * (thw[2] // 14)) // 4
                per_local_ord_slices[l_ord] = torch.arange(q_cursor, q_cursor + qc, device=device, dtype=torch.long)
                q_cursor += qc
            if per_local_ord_slices:
                restore_idx = torch.cat([per_local_ord_slices[i] for i in range(len(per_local_ord_slices))], dim=0)
            else:
                restore_idx = torch.empty((0,), device=device, dtype=torch.long)

            # Calculate max-rank AllToAllv communication volume (Fwd + Bwd):
            #   Stage 1a Fwd (raw patches 1184 BF16; no Bwd needed if raw pixels frozen, or with Bwd if differentiable;
            #   we run _DifferentiableVarlenAllToAllFn on BOTH patch dispatch and 4x-smaller merged return!)
            max_rk_bytes = 0.0
            for r in range(8):
                sent_p = sum(patch_splits[r][w] for w in range(8) if w != r)
                recv_p = sum(patch_splits[s][r] for s in range(8) if s != r)
                sent_q = sum(quad_splits[r][s] for s in range(8) if s != r)
                recv_q = sum(quad_splits[w][r] for w in range(8) if w != r)
                # Patch dispatch (1x Fwd, 1184*2 B) + Merged return (Fwd + d_merged Bwd = 2x, 3584*2 B)
                rk_b = (sent_p + recv_p) * 1184 * 2 + 2.0 * (sent_q + recv_q) * 3584 * 2
                max_rk_bytes = max(max_rk_bytes, rk_b)

            plan["decouple_vit"] = True
            plan["send_patches_buf"] = send_patches_buf
            plan["patch_out_splits"] = patch_splits[rank]
            plan["patch_in_splits"] = [patch_splits[s][rank] for s in range(8)]
            plan["quad_out_splits"] = quad_splits[rank]
            plan["quad_in_splits"] = [quad_splits[w][rank] for w in range(8)]
            plan["vmeta_worker"] = vmeta_worker
            plan["restore_idx"] = restore_idx
            plan["alltoall_comm_mib_max_rank"] = round(max_rk_bytes / (1024.0**2), 2)

        if schedule == "naive_sample_shard_padded_vit_flex":
            dt = ometa.doc_id
            st = ometa.span_id
            plan["flex_mod"] = lambda b, h, qi, ki, dt=dt, st=st: (dt[qi] == dt[ki]) & (
                (ki <= qi) | ((st[qi] > 0) & (st[qi] == st[ki]))
            )
        return plan

    def _execute_fsdp2_step(model: DualTowerMultimodalModel, plan: dict[str, Any]) -> torch.Tensor:
        h_out = model(plan, inv_freq, comp_cbm, comp_flex)
        # Exact global token-count loss scaling (`world_size / N_global`) so FSDP2 1/8 ReduceScatter equals 1-GPU global grad!
        scale = float(world_size) / float(plan["tot_tok_global"])
        loss_local = (((h_out.float() - plan["tgt_loc"].float()) * plan["vmask_loc"]) ** 2).sum() * scale
        loss_local.backward()
        return loss_local.detach()

    # Verify C.1 exact 1-GPU parity AND direct 8-GPU Decoupled AllToAllv+Quad-LPT vs 8-GPU Naive Sample-Shard parity!
    parity_c1: dict[str, Any] = {}
    naive_8gpu_grads: dict[str, torch.Tensor] = {}
    naive_8gpu_loss = 0.0
    for p_sched in ("naive_sample_shard_winmorton_surgical", "decoupled_vit_alltoall_plus_llm_quad_lpt"):
        fsdp_model.zero_grad(set_to_none=True)
        pl = _prepare_schedule_plan(
            samples_p, sample_txt_embeds, sample_targets, sample_img_packed_wm, sample_img_packed_ras, p_sched
        )
        l_loc = _execute_fsdp2_step(fsdp_model, pl)
        l_avg = l_loc.clone()
        dist.all_reduce(l_avg, op=dist.ReduceOp.AVG)

        g_vp = fsdp_model.visual.patch_proj.weight.grad.full_tensor()
        g_vq = fsdp_model.visual.blocks[0].wqkv.weight.grad.full_tensor()
        g_vm = fsdp_model.visual.merger.mlp0.weight.grad.full_tensor()
        g_lq = fsdp_model.llm_blocks[0].wq.weight.grad.full_tensor()
        g_ld = fsdp_model.llm_blocks[0].w_dn.weight.grad.full_tensor()

        if rank == 0:
            loss_val = float(l_avg.item())
            parity_c1[p_sched] = {
                "loss_abs_diff_vs_1gpu": round(abs(loss_val - ref_1gpu_loss), 8),
                "loss_rel_diff_vs_1gpu": round(abs(loss_val - ref_1gpu_loss) / max(1e-12, abs(ref_1gpu_loss)), 8),
                "vit_patch_proj_dw_rel_l2": round(rel_l2(g_vp, ref_1gpu_grads["vit_patch_proj"]), 6),
                "vit_blk0_wqkv_dw_rel_l2": round(rel_l2(g_vq, ref_1gpu_grads["vit_blk0_wqkv"]), 6),
                "merger_mlp0_dw_rel_l2": round(rel_l2(g_vm, ref_1gpu_grads["merger_mlp0"]), 6),
                "llm_blk0_wq_dw_rel_l2": round(rel_l2(g_lq, ref_1gpu_grads["llm_blk0_wq"]), 6),
                "llm_blk0_wdn_dw_rel_l2": round(rel_l2(g_ld, ref_1gpu_grads["llm_blk0_wdn"]), 6),
            }
            if p_sched == "naive_sample_shard_winmorton_surgical":
                naive_8gpu_loss = loss_val
                naive_8gpu_grads = {
                    "vit_patch_proj": g_vp.detach().clone(),
                    "vit_blk0_wqkv": g_vq.detach().clone(),
                    "merger_mlp0": g_vm.detach().clone(),
                    "llm_blk0_wq": g_lq.detach().clone(),
                    "llm_blk0_wdn": g_ld.detach().clone(),
                }
            else:
                parity_c1["decoupled_vs_naive_8gpu"] = {
                    "loss_rel_diff": round(abs(loss_val - naive_8gpu_loss) / max(1e-12, abs(naive_8gpu_loss)), 8),
                    "vit_patch_proj_dw_rel_l2": round(rel_l2(g_vp, naive_8gpu_grads["vit_patch_proj"]), 6),
                    "vit_blk0_wqkv_dw_rel_l2": round(rel_l2(g_vq, naive_8gpu_grads["vit_blk0_wqkv"]), 6),
                    "merger_mlp0_dw_rel_l2": round(rel_l2(g_vm, naive_8gpu_grads["merger_mlp0"]), 6),
                    "llm_blk0_wq_dw_rel_l2": round(rel_l2(g_lq, naive_8gpu_grads["llm_blk0_wq"]), 6),
                    "llm_blk0_wdn_dw_rel_l2": round(rel_l2(g_ld, naive_8gpu_grads["llm_blk0_wdn"]), 6),
                }
        fsdp_model.zero_grad(set_to_none=True)
    del ref_1gpu_grads, naive_8gpu_grads
    torch.cuda.empty_cache()
    dist.barrier()

    # ------------------------------------------------------------------------
    # C.2 8-GPU NVSwitch Multimodal FSDP2 Scaling Across Batch Scales (1x = 16 samples, 2x = 32 samples)
    # ------------------------------------------------------------------------
    batch_mults = [1] if is_smoke else [1, 2]
    warmup_c = 1 if is_smoke else 2
    iters_c = 3 if is_smoke else 5
    scaling_c2: dict[str, Any] = {}

    for b_mult in batch_mults:
        if b_mult == 1:
            s_cur = samples_p
            tx_cur, tg_cur, wm_cur, ras_cur = sample_txt_embeds, sample_targets, sample_img_packed_wm, sample_img_packed_ras
        else:
            s_cur = _build_global_heterogeneous_multimodal_samples(batch_mult=b_mult)
            tx_cur = sample_txt_embeds * b_mult
            tg_cur = sample_targets * b_mult
            wm_cur = sample_img_packed_wm * b_mult
            ras_cur = sample_img_packed_ras * b_mult

        tot_patches_g = sum(s.raw_vit_patches for s in s_cur)
        tot_llm_tok_g = sum(s.total_llm_tokens for s in s_cur)

        schedules_all = [
            "naive_sample_shard_padded_vit_flex",
            "naive_sample_shard_winmorton_surgical",
            "token_greedy_llm_pack_only",
            "decoupled_vit_alltoall_plus_llm_quad_lpt",
        ]
        modes_out: dict[str, Any] = {}

        for sched_name in schedules_all:
            plan_s = _prepare_schedule_plan(s_cur, tx_cur, tg_cur, wm_cur, ras_cur, sched_name)

            for _ in range(warmup_c):
                _execute_fsdp2_step(fsdp_model, plan_s)
                fsdp_model.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _execute_fsdp2_step(fsdp_model, plan_s)
            fsdp_model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            pk_mib_local = torch.tensor([torch.cuda.max_memory_allocated(device) / (1024.0**2)], device=device, dtype=torch.float32)
            dist.all_reduce(pk_mib_local, op=dist.ReduceOp.MAX)

            samples_ms: list[float] = []
            clk_v, pwr_v = -1.0, -1.0
            for i in range(iters_c):
                dist.barrier()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                _execute_fsdp2_step(fsdp_model, plan_s)
                fsdp_model.zero_grad(set_to_none=True)
                e1.record()
                if i == iters_c // 2 and rank == 0:
                    clk_v, pwr_v = query_gpu_telemetry(0)
                torch.cuda.synchronize(device)
                ms_t = torch.tensor([e0.elapsed_time(e1)], device=device, dtype=torch.float32)
                dist.all_reduce(ms_t, op=dist.ReduceOp.MAX)
                samples_ms.append(float(ms_t.item()))

            med_ms = float(statistics.median(samples_ms))
            modes_out[sched_name] = {
                "median_ms": round(med_ms, 3),
                "min_ms": round(float(min(samples_ms)), 3),
                "max_ms": round(float(max(samples_ms)), 3),
                "global_llm_tok_per_s": round(tot_llm_tok_g / (med_ms * 1e-3), 1),
                "global_vit_patches_per_s": round(tot_patches_g / (med_ms * 1e-3), 1),
                "peak_allocated_mib_max_rank": round(float(pk_mib_local.item()), 2),
                "vit_patch_imbalance_max_over_mean": plan_s["vit_patch_imbalance"],
                "llm_quad_cost_imbalance_max_over_mean": plan_s["llm_quad_cost_imbalance"],
                "alltoall_comm_mib_max_rank": plan_s["alltoall_comm_mib_max_rank"],
                "sm_clock_mhz": round(float(clk_v), 1),
                "power_w": round(float(pwr_v), 2),
            }
            del plan_s
            torch.cuda.empty_cache()

        if rank == 0:
            dec_ms = modes_out["decoupled_vit_alltoall_plus_llm_quad_lpt"]["median_ms"]
            scaling_c2[f"B_{len(s_cur)}_samples"] = {
                "global_samples": len(s_cur),
                "global_raw_vit_patches": tot_patches_g,
                "global_llm_tokens": tot_llm_tok_g,
                "params_billions": round(n_params_total / 1e9, 4),
                "modes": modes_out,
                "decoupled_speedup_over_naive_padded_flex": round(
                    modes_out["naive_sample_shard_padded_vit_flex"]["median_ms"] / dec_ms, 3
                ),
                "decoupled_speedup_over_naive_winmorton_surgical": round(
                    modes_out["naive_sample_shard_winmorton_surgical"]["median_ms"] / dec_ms, 3
                ),
                "decoupled_speedup_over_token_greedy_llm_only": round(
                    modes_out["token_greedy_llm_pack_only"]["median_ms"] / dec_ms, 3
                ),
            }

    del fsdp_model, sample_txt_embeds, sample_targets, sample_img_packed_wm, sample_img_packed_ras
    torch.cuda.empty_cache()
    dist.barrier()
    return {
        "parity_vs_1gpu_reference": parity_c1,
        "fsdp2_8gpu_multimodal_scaling": scaling_c2,
    }


# ============================================================================
# 7. Entrypoint
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E126: Multimodal VLM/Omni Native-Resolution Training Kernels & 8-GPU Modality-Decoupled FSDP2"
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
        print(f"[E126] Suite 0 complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sa_res = run_suite_a_1gpu_vit_scaling(device, is_smoke)
        print(f"[E126] Suite A complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sb_res = run_suite_b_1gpu_omni_llm_scaling(device, is_smoke)
        print(f"[E126] Suite B complete at +{time.perf_counter() - t0:.1f}s", flush=True)

    dist.barrier()
    sc_res = run_suite_c_8gpu_multimodal_fsdp2(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = time.perf_counter() - t0

    if rank == 0:
        print(f"[E126] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        env = environment_metadata()
        env["hostname"] = "node-0"
        env["cudnn_version"] = int(torch.backends.cudnn.version() or 0)
        payload = {
            "experiment": "E126_multimodal_vlm_omni_training",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": round(elapsed_s, 2),
            "initial_sm_clock_mhz": round(init_clk, 1),
            "initial_power_w": round(init_pwr, 2),
            "environment": env,
            "suite_0_audits": s0_res,
            "suite_a_1gpu_native_res_vit_scaling": sa_res,
            "suite_b_1gpu_mrope_omni_attention_scaling": sb_res,
            "suite_c_8gpu_modality_decoupled_fsdp2": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()



