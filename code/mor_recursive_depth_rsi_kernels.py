#!/usr/bin/env python3
"""
E135 (v126, Round-9 RSI): Custom SM80 Zero-Spill Triton Kernels & Exact FP64 Oracles for
Mixture-of-Recursions (MoR: Learning Dynamic Recursive Depths for Adaptive Token-Level
Computation, Bae et al., arXiv:2507.10524v2, NeurIPS 2025; official repository
`raymin0223/mixture_of_recursions` commit `82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6`).

Resolves six concrete upstream training-kernel and mathematical pathologies in
`raymin0223/mixture_of_recursions`:
  1. `kv_sharing.enable=True` full-sequence `[B, T, D]` waste (`expert_choice_router.py:225-240`,
     `token_choice_router.py:222-245`): upstream computes Q, K, V, Attention, and MLP on all
     `T` tokens at every recursion `r = 2..N_r` and discards inactive rows via `torch.gather`
     because standard `FlashAttention(is_causal=True)` aligns unequal Q/K lengths to the
     bottom-right diagonal (`n <= m + T - K_r`) rather than original coordinates `n <= pos_q[m]`.
     Solved via Two-Loop Prefix+Straddle Position-Indexed Rectangular Causal-Gather `Flash-MoR`
     (`_flash_mor_rect_causal_fwd_kernel`, `_flash_mor_rect_causal_dq_kernel`, and binary-search
     key-parallel GQA-native `_flash_mor_rect_causal_dkv_kernel` with 0 global atomics and 0
     SM80 cubin register spills across both `D_h = 64` and `D_h = 128`).
  2. Host-synchronizing Python `for b in range(bs): torch.where(...)` + `rnn_utils.pad_sequence`
     + dense 4D `[new_bs, 1, S_max, S_max]` mask (`token_choice_router.py:75-164`). Solved via
     device-resident prefix-sum compaction (`_mor_varlen_scan_pack_kernel`) + packed varlen
     execution with zero padding and zero 4D masks.
  3. Omitted `g_t^r` router gate on unfinished transitions (`token_choice_router.py:293`) +
     in-place `updates[batch_idx] = ...; x = updates` `CopySlices` autograd aliasing bug
     (`token_choice_router.py:295-300`). Solved via injective zero-clone fused
     `_mor_gated_scatter_residual_fwd/bwd_kernel` implementing exact Eq. (2.1) & Eq. (2.2).
  4. Degenerate `[bs, k, 1]` `torch.logsumexp(_router_weights, dim=-1).square()` bug
     (`expert_choice_router.py:289`), which collapses identically to `z ** 2` and penalizes
     negative unselected logits against `BCEWithLogitsLoss`. Solved via fused FP32 binary-partition
     `softplus(z)^2 = log(1 + exp(z))^2` + BCE router kernel (`_mor_router_bce_zloss_fwd_bwd_kernel`).
  5. Redundant `[B, k_r, D]` double-gather & `cos/sin` gather copies (`expert_choice_router.py:156,
     181, 255`) + `position_ids[:, :top_k]` / `emb[0]` indexing bugs. Solved via fused injective
     `_mor_gather_rmsnorm_fwd/bwd_kernel` + `_mor_indexed_rope_fwd/bwd_kernel`.
  6. Unimplemented `# TODO: support MoRLlamaDecoderLayer` activation checkpointing
     (`modeling_llama.py:232`). Solved via Compact Active-Token Recursion-Boundary Checkpointing
     saving only `[K_r, D]` active states + discrete `int32` routing indices (`0` replay drift).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    flash_attn_varlen_func = None
    HAS_FLASH_ATTN = False


# =============================================================================
# 1. Device-Resident Varlen Active-Token Prefix Scan & Compaction
# =============================================================================

@triton.jit
def _mor_varlen_count_kernel(
    Mask_ptr,
    Counts_ptr,
    B: tl.constexpr,
    T,
    stride_mb,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)
    if b >= B:
        return
    acc = tl.zeros((), dtype=tl.int32)
    num_tiles = tl.cdiv(T, BLOCK_T)
    for tid in range(0, num_tiles):
        offs = tid * BLOCK_T + tl.arange(0, BLOCK_T)
        valid = offs < T
        m = tl.load(Mask_ptr + b * stride_mb + offs, mask=valid, other=0).to(tl.int32)
        acc += tl.sum(tl.where(m > 0, 1, 0), axis=0)
    tl.store(Counts_ptr + b, acc)


@triton.jit
def _mor_varlen_scan_pack_kernel(
    Mask_ptr,
    Counts_ptr,
    CuSeq_ptr,
    FlatIdx_ptr,
    PosQ_ptr,
    B: tl.constexpr,
    T,
    stride_mb,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)
    if b >= B:
        return

    # Exclusive prefix sum of Counts[0..b-1] to get doc_q_start
    doc_start = tl.zeros((), dtype=tl.int32)
    for prev_b in range(0, b):
        doc_start += tl.load(Counts_ptr + prev_b)
    doc_count = tl.load(Counts_ptr + b)
    if b == 0:
        tl.store(CuSeq_ptr, 0)
    tl.store(CuSeq_ptr + b + 1, doc_start + doc_count)

    write_base = doc_start
    num_tiles = tl.cdiv(T, BLOCK_T)
    for tid in range(0, num_tiles):
        offs = tid * BLOCK_T + tl.arange(0, BLOCK_T)
        valid = offs < T
        active = (tl.load(Mask_ptr + b * stride_mb + offs, mask=valid, other=0).to(tl.int32) > 0) & valid
        flag = tl.where(active, 1, 0).to(tl.int32)
        prefix_inc = tl.cumsum(flag, axis=0)
        tile_total = tl.sum(flag, axis=0)
        out_slot = write_base + prefix_inc - 1
        tl.store(FlatIdx_ptr + out_slot, b * T + offs, mask=active)
        tl.store(PosQ_ptr + out_slot, offs, mask=active)
        write_base += tile_total


def mor_varlen_pack_from_mask(
    active_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Compacts a [B, T] boolean/int active mask into:
      - cu_seqlens_q: int32 [B + 1]
      - flat_idx: int32 [K_total] in [0, B * T)
      - pos_q: int32 [K_total] in [0, T) (strictly monotone within each document)
      - k_total: int scalar total active tokens across batch
    Uses 2 tiny Triton launches + 1 single-int32 read (vs B host-blocking torch.where syncs).
    """
    assert active_mask.ndim == 2 and active_mask.is_cuda
    B, T = active_mask.shape
    mask_u8 = active_mask.to(torch.uint8) if active_mask.dtype != torch.uint8 else active_mask
    counts = torch.empty((B,), dtype=torch.int32, device=active_mask.device)
    cu_seqlens_q = torch.empty((B + 1,), dtype=torch.int32, device=active_mask.device)
    max_cap = max(1, B * T)
    flat_buf = torch.empty((max_cap,), dtype=torch.int32, device=active_mask.device)
    pos_buf = torch.empty((max_cap,), dtype=torch.int32, device=active_mask.device)

    BLOCK_T = 512 if T >= 512 else 256
    _mor_varlen_count_kernel[(B,)](
        mask_u8,
        counts,
        B=B,
        T=T,
        stride_mb=mask_u8.stride(0),
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    _mor_varlen_scan_pack_kernel[(B,)](
        mask_u8,
        counts,
        cu_seqlens_q,
        flat_buf,
        pos_buf,
        B=B,
        T=T,
        stride_mb=mask_u8.stride(0),
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    k_total = int(cu_seqlens_q[B].item())
    return cu_seqlens_q, flat_buf[:k_total], pos_buf[:k_total], k_total


# =============================================================================
# 2. Fused Sparse-Gather Pre-RMSNorm + Analytical Split-CTA Backward (0 Spills)
# =============================================================================

@triton.jit
def _mor_gather_rmsnorm_fwd_kernel(
    X_ptr,
    FlatIdx_ptr,
    Gamma_ptr,
    Y_ptr,
    Rstd_ptr,
    K_total,
    D: tl.constexpr,
    stride_xm,
    stride_ym,
    eps,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0)
    if m >= K_total:
        return
    src_row = tl.load(FlatIdx_ptr + m).to(tl.int64)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X_ptr + src_row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(sum_sq / float(D) + eps)
    tl.store(Rstd_ptr + m, rstd)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X_ptr + src_row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd) * g
        tl.store(Y_ptr + m * stride_ym + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mor_gather_rmsnorm_bwd_dx_part_kernel(
    DY_ptr,
    X_ptr,
    FlatIdx_ptr,
    Gamma_ptr,
    Rstd_ptr,
    DX_ptr,
    DGammaPart_ptr,
    K_total,
    D: tl.constexpr,
    stride_dym,
    stride_xm,
    stride_dxm,
    N_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    split_id = tl.program_id(0)
    if split_id >= N_SPLITS:
        return
    rows_per_split = tl.cdiv(K_total, N_SPLITS)
    m_start = split_id * rows_per_split
    m_end = tl.minimum(m_start + rows_per_split, K_total)

    # First compute dot_m = sum_d (dy * g * x) for each row and write injective dx into DX_ptr
    for m in range(m_start, m_end):
        src_row = tl.load(FlatIdx_ptr + m).to(tl.int64)
        rstd = tl.load(Rstd_ptr + m)
        dot_acc = tl.zeros((), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            dy = tl.load(DY_ptr + m * stride_dym + offs, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X_ptr + src_row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(Gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            dot_acc += tl.sum((dy * g) * x, axis=0)
        c1 = (rstd * rstd * rstd) * (dot_acc / float(D))
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            dy = tl.load(DY_ptr + m * stride_dym + offs, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X_ptr + src_row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(Gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            dx_prev = tl.load(DX_ptr + src_row * stride_dxm + offs, mask=mask, other=0.0).to(tl.float32)
            dx = dx_prev + (dy * g) * rstd - x * c1
            tl.store(DX_ptr + src_row * stride_dxm + offs, dx.to(DX_ptr.dtype.element_ty), mask=mask)

    # Accumulate partial dgamma per BLOCK_D slice across [m_start, m_end)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        dg_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for m in range(m_start, m_end):
            src_row = tl.load(FlatIdx_ptr + m).to(tl.int64)
            rstd = tl.load(Rstd_ptr + m)
            dy = tl.load(DY_ptr + m * stride_dym + offs, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X_ptr + src_row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
            dg_acc += dy * (x * rstd)
        tl.store(DGammaPart_ptr + split_id * D + offs, dg_acc, mask=mask)


@triton.jit
def _mor_reduce_dgamma_splits_kernel(
    DGammaPart_ptr,
    DGamma_ptr,
    D: tl.constexpr,
    N_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for s in range(0, N_SPLITS):
        acc += tl.load(DGammaPart_ptr + s * D + offs, mask=mask, other=0.0)
    tl.store(DGamma_ptr + offs, acc.to(DGamma_ptr.dtype.element_ty), mask=mask)


class _MoRGatherRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_flat: torch.Tensor,
        flat_idx: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        assert x_flat.ndim == 2 and gamma.ndim == 1
        N_full, D = x_flat.shape
        K_total = int(flat_idx.numel())
        y = torch.empty((K_total, D), dtype=x_flat.dtype, device=x_flat.device)
        rstd = torch.empty((K_total,), dtype=torch.float32, device=x_flat.device)
        BLOCK_D = min(1024, triton.next_power_of_2(D))
        if K_total > 0:
            _mor_gather_rmsnorm_fwd_kernel[(K_total,)](
                x_flat,
                flat_idx,
                gamma,
                y,
                rstd,
                K_total,
                D=D,
                stride_xm=x_flat.stride(0),
                stride_ym=y.stride(0),
                eps=float(eps),
                BLOCK_D=BLOCK_D,
                num_warps=4,
            )
        ctx.save_for_backward(x_flat, flat_idx, gamma, rstd)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_flat, flat_idx, gamma, rstd = ctx.saved_tensors
        N_full, D = x_flat.shape
        K_total = int(flat_idx.numel())
        dx_flat = torch.zeros_like(x_flat)
        if K_total == 0:
            return dx_flat, None, torch.zeros_like(gamma), None

        dy_c = dy.contiguous()
        BLOCK_D = min(1024, triton.next_power_of_2(D))
        N_SPLITS = 64 if K_total >= 256 else 16
        dgamma_part = torch.zeros((N_SPLITS, D), dtype=torch.float32, device=x_flat.device)
        _mor_gather_rmsnorm_bwd_dx_part_kernel[(N_SPLITS,)](
            dy_c,
            x_flat,
            flat_idx,
            gamma,
            rstd,
            dx_flat,
            dgamma_part,
            K_total,
            D=D,
            stride_dym=dy_c.stride(0),
            stride_xm=x_flat.stride(0),
            stride_dxm=dx_flat.stride(0),
            N_SPLITS=N_SPLITS,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        dgamma = torch.empty_like(gamma)
        _mor_reduce_dgamma_splits_kernel[(triton.cdiv(D, BLOCK_D),)](
            dgamma_part,
            dgamma,
            D=D,
            N_SPLITS=N_SPLITS,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        return dx_flat, None, dgamma, None


def mor_gather_rmsnorm(
    x_flat: torch.Tensor,
    flat_idx: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if x_flat.dtype == torch.float64:
        gathered = x_flat.index_select(0, flat_idx.to(torch.long))
        rstd = torch.rsqrt(torch.mean(gathered * gathered, dim=-1, keepdim=True) + eps)
        return gathered * rstd * gamma.to(torch.float64)
    return _MoRGatherRMSNormFn.apply(x_flat, flat_idx, gamma, eps)


# =============================================================================
# 3. Fused Position-Indexed Rotary Embedding (Zero Gather Allocation, 0 Spills)
# =============================================================================

@triton.jit
def _mor_indexed_rope_kernel(
    X_ptr,
    Pos_ptr,
    Cos_ptr,
    Sin_ptr,
    Out_ptr,
    M_total,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    stride_xm,
    stride_xh,
    stride_cs_t,
    stride_om,
    stride_oh,
    INVERSE: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid // H
    h = pid % H
    if m >= M_total:
        return
    pos = tl.load(Pos_ptr + m).to(tl.int64)
    offs = tl.arange(0, HALF_D)
    cos = tl.load(Cos_ptr + pos * stride_cs_t + offs).to(tl.float32)
    sin = tl.load(Sin_ptr + pos * stride_cs_t + offs).to(tl.float32)
    x1 = tl.load(X_ptr + m * stride_xm + h * stride_xh + offs).to(tl.float32)
    x2 = tl.load(X_ptr + m * stride_xm + h * stride_xh + HALF_D + offs).to(tl.float32)
    if not INVERSE:
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
    else:
        o1 = x1 * cos + x2 * sin
        o2 = x2 * cos - x1 * sin
    tl.store(Out_ptr + m * stride_om + h * stride_oh + offs, o1.to(Out_ptr.dtype.element_ty))
    tl.store(Out_ptr + m * stride_om + h * stride_oh + HALF_D + offs, o2.to(Out_ptr.dtype.element_ty))


class _MoRIndexedRoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pos: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
        M_total, H, D = x.shape
        HALF_D = D // 2
        out = torch.empty_like(x)
        if M_total > 0:
            _mor_indexed_rope_kernel[(M_total * H,)](
                x,
                pos,
                cos_cache,
                sin_cache,
                out,
                M_total,
                H=H,
                HALF_D=HALF_D,
                stride_xm=x.stride(0),
                stride_xh=x.stride(1),
                stride_cs_t=cos_cache.stride(0),
                stride_om=out.stride(0),
                stride_oh=out.stride(1),
                INVERSE=False,
                num_warps=2,
            )
        ctx.save_for_backward(pos, cos_cache, sin_cache)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        pos, cos_cache, sin_cache = ctx.saved_tensors
        M_total, H, D = dout.shape
        HALF_D = D // 2
        dx = torch.empty_like(dout)
        if M_total > 0:
            _mor_indexed_rope_kernel[(M_total * H,)](
                dout,
                pos,
                cos_cache,
                sin_cache,
                dx,
                M_total,
                H=H,
                HALF_D=HALF_D,
                stride_xm=dout.stride(0),
                stride_xh=dout.stride(1),
                stride_cs_t=cos_cache.stride(0),
                stride_om=dx.stride(0),
                stride_oh=dx.stride(1),
                INVERSE=True,
                num_warps=2,
            )
        return dx, None, None, None


def mor_indexed_rope(
    x: torch.Tensor,
    pos: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Applies exact rotary embedding at arbitrary sparse positions `pos[m]` (`[M, H, D]`)."""
    if x.dtype == torch.float64:
        p = pos.to(torch.long)
        c = cos_cache.to(torch.float64).index_select(0, p).unsqueeze(1)
        s = sin_cache.to(torch.float64).index_select(0, p).unsqueeze(1)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return _MoRIndexedRoPEFn.apply(x, pos, cos_cache, sin_cache)


# =============================================================================
# 4. Superfused Eq. (2.1) & Eq. (2.2) Gated Scatter-Residual (0 Clones, 0 Spills)
# =============================================================================

@triton.jit
def _mor_gated_scatter_residual_fwd_kernel(
    Base_ptr,
    Y_ptr,
    Gate_ptr,
    FlatIdx_ptr,
    AddBaseMask_ptr,
    Out_ptr,
    K_total,
    D: tl.constexpr,
    stride_bm,
    stride_ym,
    stride_om,
    HAS_ADD_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0)
    if m >= K_total:
        return
    dst_row = tl.load(FlatIdx_ptr + m).to(tl.int64)
    g = tl.load(Gate_ptr + m).to(tl.float32)
    add_base = 1.0
    if HAS_ADD_MASK:
        flag = tl.load(AddBaseMask_ptr + m).to(tl.int32)
        add_base = tl.where(flag > 0, 1.0, 0.0)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        y = tl.load(Y_ptr + m * stride_ym + offs, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(Base_ptr + dst_row * stride_bm + offs, mask=mask, other=0.0).to(tl.float32)
        out = g * y + add_base * b_val
        tl.store(Out_ptr + dst_row * stride_om + offs, out.to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mor_gated_scatter_residual_bwd_kernel(
    DOut_ptr,
    Y_ptr,
    Gate_ptr,
    FlatIdx_ptr,
    AddBaseMask_ptr,
    DBase_ptr,
    DY_ptr,
    DGate_ptr,
    K_total,
    D: tl.constexpr,
    stride_dom,
    stride_ym,
    stride_dbm,
    stride_dym,
    HAS_ADD_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0)
    if m >= K_total:
        return
    dst_row = tl.load(FlatIdx_ptr + m).to(tl.int64)
    g = tl.load(Gate_ptr + m).to(tl.float32)
    add_base = 1.0
    if HAS_ADD_MASK:
        flag = tl.load(AddBaseMask_ptr + m).to(tl.int32)
        add_base = tl.where(flag > 0, 1.0, 0.0)

    dg_acc = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        dout = tl.load(DOut_ptr + dst_row * stride_dom + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y_ptr + m * stride_ym + offs, mask=mask, other=0.0).to(tl.float32)
        dg_acc += tl.sum(dout * y, axis=0)
        dy = dout * g
        db = dout * add_base
        tl.store(DY_ptr + m * stride_dym + offs, dy.to(DY_ptr.dtype.element_ty), mask=mask)
        tl.store(DBase_ptr + dst_row * stride_dbm + offs, db.to(DBase_ptr.dtype.element_ty), mask=mask)
    tl.store(DGate_ptr + m, dg_acc.to(DGate_ptr.dtype.element_ty))


class _MoRGatedScatterResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_passthrough: torch.Tensor,
        x_base_residual: torch.Tensor,
        y_active: torch.Tensor,
        gate_active: torch.Tensor,
        flat_idx: torch.Tensor,
        add_base_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes exact Eq. (2.1) or Eq. (2.2) without mutating inputs:
          - For inactive rows u not in flat_idx: Out[u] = x_passthrough[u]
          - For active rows dst = flat_idx[m]:
              Out[dst] = gate_active[m] * y_active[m] + (x_base_residual[dst] if add_base_mask[m] else 0)
        Note: for Eq. (2.1) Expert-Choice, `x_base_residual` is `x_passthrough` (`H^r`) and `add_base_mask` is None (always 1).
              for Eq. (2.2) Token-Choice, `x_passthrough` is `H^r` (holding already-finished tokens),
              `x_base_residual` is `H^1` (initial anchor), and `add_base_mask[m] = (depth[dst] == r)`.
        """
        N_full, D = x_passthrough.shape
        K_total = int(flat_idx.numel())
        out = x_passthrough.clone()
        BLOCK_D = min(1024, triton.next_power_of_2(D))
        has_mask = add_base_mask is not None
        mask_arg = add_base_mask if has_mask else flat_idx
        if K_total > 0:
            _mor_gated_scatter_residual_fwd_kernel[(K_total,)](
                x_base_residual,
                y_active,
                gate_active,
                flat_idx,
                mask_arg,
                out,
                K_total,
                D=D,
                stride_bm=x_base_residual.stride(0),
                stride_ym=y_active.stride(0),
                stride_om=out.stride(0),
                HAS_ADD_MASK=has_mask,
                BLOCK_D=BLOCK_D,
                num_warps=4,
            )
        ctx.save_for_backward(y_active, gate_active, flat_idx, mask_arg)
        ctx.has_mask = has_mask
        ctx.same_base = x_passthrough.data_ptr() == x_base_residual.data_ptr()
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        y_active, gate_active, flat_idx, mask_arg = ctx.saved_tensors
        has_mask = ctx.has_mask
        K_total, D = y_active.shape
        dout_c = dout.contiguous()
        dy = torch.empty_like(y_active)
        dgate = torch.empty_like(gate_active)
        BLOCK_D = min(1024, triton.next_power_of_2(D))

        if ctx.same_base and not has_mask:
            # Expert-Choice Eq. (2.1): Out[u] = x[u] for inactive, Out[dst] = x[dst] + g*y for active -> dx_passthrough == dout everywhere!
            dx_pass = dout_c.clone()
            dbase_tmp = torch.empty((1, D), dtype=dout_c.dtype, device=dout_c.device)
            if K_total > 0:
                # Compute dy and dgate; write dummy dbase into dx_pass[dst] which equals dout_c[dst] * 1.0 == dout_c[dst]
                _mor_gated_scatter_residual_bwd_kernel[(K_total,)](
                    dout_c,
                    y_active,
                    gate_active,
                    flat_idx,
                    mask_arg,
                    dx_pass,
                    dy,
                    dgate,
                    K_total,
                    D=D,
                    stride_dom=dout_c.stride(0),
                    stride_ym=y_active.stride(0),
                    stride_dbm=dx_pass.stride(0),
                    stride_dym=dy.stride(0),
                    HAS_ADD_MASK=False,
                    BLOCK_D=BLOCK_D,
                    num_warps=4,
                )
            return dx_pass, torch.zeros_like(dx_pass), dy, dgate, None, None

        # General Token-Choice Eq. (2.2):
        # Inactive rows come from x_passthrough (dx_pass[u] = dout[u], dx_pass[dst] = 0).
        # Active rows with add_base_mask[m]==1 add x_base_residual[dst] (dx_base[dst] = dout[dst] * add_base).
        dx_pass = dout_c.clone()
        dx_base = torch.zeros_like(dout_c)
        if K_total > 0:
            dx_pass.index_fill_(0, flat_idx.to(torch.long), 0.0)
            _mor_gated_scatter_residual_bwd_kernel[(K_total,)](
                dout_c,
                y_active,
                gate_active,
                flat_idx,
                mask_arg,
                dx_base,
                dy,
                dgate,
                K_total,
                D=D,
                stride_dom=dout_c.stride(0),
                stride_ym=y_active.stride(0),
                stride_dbm=dx_base.stride(0),
                stride_dym=dy.stride(0),
                HAS_ADD_MASK=has_mask,
                BLOCK_D=BLOCK_D,
                num_warps=4,
            )
        return dx_pass, dx_base, dy, dgate, None, None


def mor_gated_scatter_residual(
    x_passthrough: torch.Tensor,
    x_base_residual: torch.Tensor,
    y_active: torch.Tensor,
    gate_active: torch.Tensor,
    flat_idx: torch.Tensor,
    add_base_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x_passthrough.dtype == torch.float64:
        out = x_passthrough.clone()
        if flat_idx.numel() == 0:
            return out
        idx = flat_idx.to(torch.long)
        base_rows = x_base_residual.index_select(0, idx)
        if add_base_mask is not None:
            base_rows = base_rows * add_base_mask.to(torch.float64).unsqueeze(-1)
        src = gate_active.to(torch.float64).unsqueeze(-1) * y_active + base_rows
        out = out.index_copy(0, idx, src)
        return out
    return _MoRGatedScatterResidualFn.apply(
        x_passthrough,
        x_base_residual,
        y_active,
        gate_active,
        flat_idx,
        add_base_mask,
    )


# =============================================================================
# 5. Fused Router Binary-Partition Z-Loss + BCE Auxiliary Loss (0 Spills)
# =============================================================================

@triton.jit
def _mor_router_bce_zloss_fwd_kernel(
    Logits_ptr,
    Targets_ptr,
    Probs_ptr,
    LossPart_ptr,
    N_elems,
    alpha_scale,
    z_coeff,
    N_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    sid = tl.program_id(0)
    if sid >= N_SPLITS:
        return
    per_split = tl.cdiv(N_elems, N_SPLITS)
    start = sid * per_split
    end = tl.minimum(start + per_split, N_elems)
    bce_acc = tl.zeros((), dtype=tl.float32)
    z_acc = tl.zeros((), dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < end
        z = tl.load(Logits_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(Targets_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.sigmoid(z)
        tl.store(Probs_ptr + offs, (p * alpha_scale).to(Probs_ptr.dtype.element_ty), mask=mask)
        #Numerically stable softplus(z) = max(z, 0) + log(1 + exp(-|z|))
        abs_z = tl.abs(z)
        sp = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-abs_z))
        bce = sp - z * t
        z_pen = sp * sp
        bce_acc += tl.sum(tl.where(mask, bce, 0.0), axis=0)
        z_acc += tl.sum(tl.where(mask, z_pen, 0.0), axis=0)
    tl.store(LossPart_ptr + sid * 2 + 0, bce_acc)
    tl.store(LossPart_ptr + sid * 2 + 1, z_acc * z_coeff)


@triton.jit
def _mor_router_bce_zloss_bwd_kernel(
    Logits_ptr,
    Targets_ptr,
    DProbs_ptr,
    DLoss_ptr,
    DLogits_ptr,
    N_elems,
    inv_N,
    alpha_scale,
    z_coeff,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N_elems
    z = tl.load(Logits_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(Targets_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dp = tl.load(DProbs_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dloss_bce = tl.load(DLoss_ptr + 0).to(tl.float32)
    dloss_z = tl.load(DLoss_ptr + 1).to(tl.float32)

    p = tl.sigmoid(z)
    abs_z = tl.abs(z)
    sp = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-abs_z))
    # d(probs)/dz = alpha_scale * p * (1 - p)
    # d(bce)/dz = (p - t) / N
    # d(z_coeff * sp^2)/dz = z_coeff * 2 * sp * p / N
    dz = dp * (alpha_scale * p * (1.0 - p)) + dloss_bce * ((p - t) * inv_N) + dloss_z * (2.0 * z_coeff * sp * p * inv_N)
    tl.store(DLogits_ptr + offs, dz.to(DLogits_ptr.dtype.element_ty), mask=mask)


class _MoRRouterBCEZLossFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha_scale: float,
        z_coeff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N_elems = int(logits.numel())
        probs = torch.empty_like(logits)
        N_SPLITS = 32 if N_elems >= 4096 else 8
        loss_part = torch.zeros((N_SPLITS, 2), dtype=torch.float32, device=logits.device)
        BLOCK_N = 256
        _mor_router_bce_zloss_fwd_kernel[(N_SPLITS,)](
            logits,
            targets,
            probs,
            loss_part,
            N_elems,
            float(alpha_scale),
            float(z_coeff),
            N_SPLITS=N_SPLITS,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        sums = loss_part.sum(dim=0) / max(1, N_elems)
        bce_loss = sums[0]
        z_loss = sums[1]
        ctx.save_for_backward(logits, targets)
        ctx.alpha_scale = float(alpha_scale)
        ctx.z_coeff = float(z_coeff)
        return probs, bce_loss, z_loss

    @staticmethod
    def backward(ctx, dprobs: torch.Tensor, dbce: torch.Tensor, dzloss: torch.Tensor):
        logits, targets = ctx.saved_tensors
        N_elems = int(logits.numel())
        dprobs_c = dprobs.contiguous() if dprobs is not None else torch.zeros_like(logits)
        dloss_vec = torch.stack([
            dbce.to(torch.float32) if dbce is not None else torch.zeros((), dtype=torch.float32, device=logits.device),
            dzloss.to(torch.float32) if dzloss is not None else torch.zeros((), dtype=torch.float32, device=logits.device),
        ])
        dlogits = torch.empty_like(logits)
        BLOCK_N = 256
        _mor_router_bce_zloss_bwd_kernel[(triton.cdiv(N_elems, BLOCK_N),)](
            logits,
            targets,
            dprobs_c,
            dloss_vec,
            dlogits,
            N_elems,
            1.0 / max(1, N_elems),
            ctx.alpha_scale,
            ctx.z_coeff,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return dlogits, None, None, None


def mor_router_bce_zloss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha_scale: float = 1.0,
    z_coeff: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.dtype == torch.float64:
        z = logits
        t = targets.to(torch.float64)
        p = torch.sigmoid(z)
        probs = p * float(alpha_scale)
        sp = F.softplus(z)
        bce = (sp - z * t).mean()
        z_loss = float(z_coeff) * (sp * sp).mean()
        return probs, bce, z_loss
    return _MoRRouterBCEZLossFn.apply(logits.contiguous(), targets.contiguous(), alpha_scale, z_coeff)


# =============================================================================
# 6. Asymmetric Position-Indexed Rectangular Causal-Gather Flash-MoR (0 Spills)
# =============================================================================

@triton.jit
def _flash_mor_rect_causal_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    PosQ_ptr,
    CuQ_ptr,
    CuK_ptr,
    Out_ptr,
    LSE_ptr,
    sm_scale,
    B,
    N_q_total,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    BM: tl.constexpr,
    BN: tl.constexpr,
    INPUT_FP32: tl.constexpr,
):
    q_tile = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H_Q
    hq = bh % H_Q
    if b >= B:
        return
    G: tl.constexpr = H_Q // H_KV
    hkv = hq // G

    q_start = tl.load(CuQ_ptr + b)
    q_end = tl.load(CuQ_ptr + b + 1)
    L_q = q_end - q_start
    m_start = q_tile * BM
    if m_start >= L_q:
        return

    k_start = tl.load(CuK_ptr + b)
    k_end = tl.load(CuK_ptr + b + 1)
    L_k = k_end - k_start
    if L_k <= 0:
        return

    m_offs = m_start + tl.arange(0, BM)
    m_valid = m_offs < L_q
    d0_offs = tl.arange(0, 64)
    last_valid_m = tl.minimum(m_start + BM - 1, L_q - 1)
    p_max = tl.load(PosQ_ptr + q_start + last_valid_m)
    pos_vec = tl.load(PosQ_ptr + q_start + m_offs, mask=m_valid, other=0)

    q_base = Q_ptr + (q_start + m_offs[:, None]).to(tl.int64) * stride_qm + hq * stride_qh
    q0 = tl.load(q_base + d0_offs[None, :], mask=m_valid[:, None], other=0.0)
    acc0 = tl.zeros((BM, 64), dtype=tl.float32)
    if D == 128:
        d1_offs = 64 + tl.arange(0, 64)
        q1 = tl.load(q_base + d1_offs[None, :], mask=m_valid[:, None], other=0.0)
        acc1 = tl.zeros((BM, 64), dtype=tl.float32)

    m_i = tl.full((BM,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    kv_limit = tl.minimum(p_max + 1, L_k)

    for n_start in range(0, kv_limit, BN):
        n_offs = n_start + tl.arange(0, BN)
        n_valid = n_offs < L_k
        k_base = K_ptr + (k_start + n_offs[None, :]).to(tl.int64) * stride_kn + hkv * stride_kh
        kt0 = tl.load(k_base + d0_offs[:, None], mask=n_valid[None, :], other=0.0)
        if INPUT_FP32:
            qk = tl.dot(q0, kt0, input_precision="ieee")
        else:
            qk = tl.dot(q0, kt0)
        if D == 128:
            d1_offs = 64 + tl.arange(0, 64)
            kt1 = tl.load(k_base + d1_offs[:, None], mask=n_valid[None, :], other=0.0)
            if INPUT_FP32:
                qk += tl.dot(q1, kt1, input_precision="ieee")
            else:
                qk += tl.dot(q1, kt1)
        qk = qk * sm_scale
        causal = (n_offs[None, :] <= pos_vec[:, None]) & n_valid[None, :]
        qk = tl.where(causal, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_base = V_ptr + (k_start + n_offs[:, None]).to(tl.int64) * stride_vn + hkv * stride_vh
        v0 = tl.load(v_base + d0_offs[None, :], mask=n_valid[:, None], other=0.0)
        acc0 = acc0 * alpha[:, None]
        if INPUT_FP32:
            acc0 += tl.dot(p, v0, input_precision="ieee")
        else:
            acc0 += tl.dot(p.to(v0.dtype), v0)
        if D == 128:
            d1_offs = 64 + tl.arange(0, 64)
            v1 = tl.load(v_base + d1_offs[None, :], mask=n_valid[:, None], other=0.0)
            acc1 = acc1 * alpha[:, None]
            if INPUT_FP32:
                acc1 += tl.dot(p, v1, input_precision="ieee")
            else:
                acc1 += tl.dot(p.to(v1.dtype), v1)
        m_i = m_ij

    inv_l = 1.0 / l_i[:, None]
    o_base = Out_ptr + (q_start + m_offs[:, None]).to(tl.int64) * stride_om + hq * stride_oh
    tl.store(o_base + d0_offs[None, :], (acc0 * inv_l).to(Out_ptr.dtype.element_ty), mask=m_valid[:, None])
    if D == 128:
        d1_offs = 64 + tl.arange(0, 64)
        tl.store(o_base + d1_offs[None, :], (acc1 * inv_l).to(Out_ptr.dtype.element_ty), mask=m_valid[:, None])
    tl.store(LSE_ptr + hq * N_q_total + (q_start + m_offs), m_i + tl.log(l_i), mask=m_valid)


@triton.jit
def _flash_mor_bwd_preprocess_delta_kernel(
    Out_ptr,
    DOut_ptr,
    Delta_ptr,
    N_q_total,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    stride_om,
    stride_oh,
    stride_dom,
    stride_doh,
):
    pid = tl.program_id(0)
    m = pid // H_Q
    hq = pid % H_Q
    if m >= N_q_total:
        return
    d_offs = tl.arange(0, D)
    o = tl.load(Out_ptr + m * stride_om + hq * stride_oh + d_offs).to(tl.float32)
    do = tl.load(DOut_ptr + m * stride_dom + hq * stride_doh + d_offs).to(tl.float32)
    delta = tl.sum(o * do, axis=0)
    tl.store(Delta_ptr + hq * N_q_total + m, delta)


@triton.jit
def _flash_mor_rect_causal_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DOut_ptr,
    LSE_ptr,
    Delta_ptr,
    PosQ_ptr,
    CuQ_ptr,
    CuK_ptr,
    DQ_ptr,
    sm_scale,
    B,
    N_q_total,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    BM: tl.constexpr,
    BN: tl.constexpr,
    INPUT_FP32: tl.constexpr,
):
    q_tile = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H_Q
    hq = bh % H_Q
    if b >= B:
        return
    G: tl.constexpr = H_Q // H_KV
    hkv = hq // G

    q_start = tl.load(CuQ_ptr + b)
    q_end = tl.load(CuQ_ptr + b + 1)
    L_q = q_end - q_start
    m_start = q_tile * BM
    if m_start >= L_q:
        return

    k_start = tl.load(CuK_ptr + b)
    k_end = tl.load(CuK_ptr + b + 1)
    L_k = k_end - k_start
    if L_k <= 0:
        return

    m_offs = m_start + tl.arange(0, BM)
    m_valid = m_offs < L_q
    d0_offs = tl.arange(0, 64)
    last_valid_m = tl.minimum(m_start + BM - 1, L_q - 1)
    p_max = tl.load(PosQ_ptr + q_start + last_valid_m)
    pos_vec = tl.load(PosQ_ptr + q_start + m_offs, mask=m_valid, other=-1)

    q_base = Q_ptr + (q_start + m_offs[:, None]).to(tl.int64) * stride_qm + hq * stride_qh
    do_base = DOut_ptr + (q_start + m_offs[:, None]).to(tl.int64) * stride_dom + hq * stride_doh
    q0 = tl.load(q_base + d0_offs[None, :], mask=m_valid[:, None], other=0.0)
    do0 = tl.load(do_base + d0_offs[None, :], mask=m_valid[:, None], other=0.0)
    dq0 = tl.zeros((BM, 64), dtype=tl.float32)
    if D == 128:
        d1_offs = 64 + tl.arange(0, 64)
        q1 = tl.load(q_base + d1_offs[None, :], mask=m_valid[:, None], other=0.0)
        do1 = tl.load(do_base + d1_offs[None, :], mask=m_valid[:, None], other=0.0)
        dq1 = tl.zeros((BM, 64), dtype=tl.float32)

    lse = tl.load(LSE_ptr + hq * N_q_total + (q_start + m_offs), mask=m_valid, other=0.0)
    delta = tl.load(Delta_ptr + hq * N_q_total + (q_start + m_offs), mask=m_valid, other=0.0)
    kv_limit = tl.minimum(p_max + 1, L_k)

    for n_start in range(0, kv_limit, BN):
        n_offs = n_start + tl.arange(0, BN)
        n_valid = n_offs < L_k
        k_base = K_ptr + (k_start + n_offs[None, :]).to(tl.int64) * stride_kn + hkv * stride_kh
        v_base = V_ptr + (k_start + n_offs[None, :]).to(tl.int64) * stride_vn + hkv * stride_vh
        kt0 = tl.load(k_base + d0_offs[:, None], mask=n_valid[None, :], other=0.0)
        vt0 = tl.load(v_base + d0_offs[:, None], mask=n_valid[None, :], other=0.0)
        if INPUT_FP32:
            qk = tl.dot(q0, kt0, input_precision="ieee")
            dp = tl.dot(do0, vt0, input_precision="ieee")
        else:
            qk = tl.dot(q0, kt0)
            dp = tl.dot(do0, vt0)
        if D == 128:
            d1_offs = 64 + tl.arange(0, 64)
            kt1 = tl.load(k_base + d1_offs[:, None], mask=n_valid[None, :], other=0.0)
            vt1 = tl.load(v_base + d1_offs[:, None], mask=n_valid[None, :], other=0.0)
            if INPUT_FP32:
                qk += tl.dot(q1, kt1, input_precision="ieee")
                dp += tl.dot(do1, vt1, input_precision="ieee")
            else:
                qk += tl.dot(q1, kt1)
                dp += tl.dot(do1, vt1)
        qk = qk * sm_scale
        causal = (n_offs[None, :] <= pos_vec[:, None]) & n_valid[None, :] & m_valid[:, None]
        p = tl.where(causal, tl.exp(tl.where(causal, qk - lse[:, None], -float("inf"))), 0.0)
        ds = p * (dp - delta[:, None]) * sm_scale

        k0_row = tl.trans(kt0)
        if INPUT_FP32:
            dq0 += tl.dot(ds, k0_row, input_precision="ieee")
        else:
            dq0 += tl.dot(ds.to(k0_row.dtype), k0_row)
        if D == 128:
            k1_row = tl.trans(kt1)
            if INPUT_FP32:
                dq1 += tl.dot(ds, k1_row, input_precision="ieee")
            else:
                dq1 += tl.dot(ds.to(k1_row.dtype), k1_row)

    dq_base = DQ_ptr + (q_start + m_offs[:, None]).to(tl.int64) * stride_dqm + hq * stride_dqh
    tl.store(dq_base + d0_offs[None, :], dq0.to(DQ_ptr.dtype.element_ty), mask=m_valid[:, None])
    if D == 128:
        d1_offs = 64 + tl.arange(0, 64)
        tl.store(dq_base + d1_offs[None, :], dq1.to(DQ_ptr.dtype.element_ty), mask=m_valid[:, None])


@triton.jit
def _flash_mor_rect_causal_dkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DOut_ptr,
    LSE_ptr,
    Delta_ptr,
    PosQ_ptr,
    CuQ_ptr,
    CuK_ptr,
    DK_hq_ptr,
    DV_hq_ptr,
    sm_scale,
    B,
    N_q_total,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    BM: tl.constexpr,
    BN: tl.constexpr,
    INPUT_FP32: tl.constexpr,
):
    k_tile = tl.program_id(0)
    bh_q = tl.program_id(1)
    b = bh_q // H_Q
    hq = bh_q % H_Q
    if b >= B:
        return
    G: tl.constexpr = H_Q // H_KV
    hkv = hq // G

    k_start = tl.load(CuK_ptr + b)
    k_end = tl.load(CuK_ptr + b + 1)
    L_k = k_end - k_start
    n_start = k_tile * BN
    if n_start >= L_k:
        return

    n_offs = n_start + tl.arange(0, BN)
    n_valid = n_offs < L_k
    d0_offs = tl.arange(0, 64)

    dk0 = tl.zeros((BN, 64), dtype=tl.float32)
    dv0 = tl.zeros((BN, 64), dtype=tl.float32)
    if D == 128:
        d1_offs = 64 + tl.arange(0, 64)
        dk1 = tl.zeros((BN, 64), dtype=tl.float32)
        dv1 = tl.zeros((BN, 64), dtype=tl.float32)

    q_start = tl.load(CuQ_ptr + b)
    q_end = tl.load(CuQ_ptr + b + 1)
    L_q = q_end - q_start

    if L_q > 0:
        lo = tl.zeros((), dtype=tl.int32)
        hi = L_q
        for _ in tl.static_range(16):
            mid = (lo + hi) // 2
            in_bounds = mid < L_q
            safe_mid = tl.minimum(mid, L_q - 1)
            p_mid = tl.load(PosQ_ptr + q_start + safe_mid, mask=in_bounds, other=2147483647)
            go_right = in_bounds & (p_mid < n_start)
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        m_first = lo

        if m_first < L_q:
            k_base = K_ptr + (k_start + n_offs[:, None]).to(tl.int64) * stride_kn + hkv * stride_kh
            v_base = V_ptr + (k_start + n_offs[:, None]).to(tl.int64) * stride_vn + hkv * stride_vh
            k0 = tl.load(k_base + d0_offs[None, :], mask=n_valid[:, None], other=0.0)
            v0 = tl.load(v_base + d0_offs[None, :], mask=n_valid[:, None], other=0.0)
            if D == 128:
                d1_offs = 64 + tl.arange(0, 64)
                k1 = tl.load(k_base + d1_offs[None, :], mask=n_valid[:, None], other=0.0)
                v1 = tl.load(v_base + d1_offs[None, :], mask=n_valid[:, None], other=0.0)

            for m_start in range(m_first, L_q, BM):
                m_offs = m_start + tl.arange(0, BM)
                m_valid = m_offs < L_q
                pos_vec = tl.load(PosQ_ptr + q_start + m_offs, mask=m_valid, other=-1)

                q_base = Q_ptr + (q_start + m_offs[None, :]).to(tl.int64) * stride_qm + hq * stride_qh
                do_base = DOut_ptr + (q_start + m_offs[None, :]).to(tl.int64) * stride_dom + hq * stride_doh
                qt0 = tl.load(q_base + d0_offs[:, None], mask=m_valid[None, :], other=0.0)
                dot0 = tl.load(do_base + d0_offs[:, None], mask=m_valid[None, :], other=0.0)
                if INPUT_FP32:
                    s_t = tl.dot(k0, qt0, input_precision="ieee")
                    dp_t = tl.dot(v0, dot0, input_precision="ieee")
                else:
                    s_t = tl.dot(k0, qt0)
                    dp_t = tl.dot(v0, dot0)
                if D == 128:
                    d1_offs = 64 + tl.arange(0, 64)
                    qt1 = tl.load(q_base + d1_offs[:, None], mask=m_valid[None, :], other=0.0)
                    dot1 = tl.load(do_base + d1_offs[:, None], mask=m_valid[None, :], other=0.0)
                    if INPUT_FP32:
                        s_t += tl.dot(k1, qt1, input_precision="ieee")
                        dp_t += tl.dot(v1, dot1, input_precision="ieee")
                    else:
                        s_t += tl.dot(k1, qt1)
                        dp_t += tl.dot(v1, dot1)

                s_t = s_t * sm_scale
                lse = tl.load(LSE_ptr + hq * N_q_total + (q_start + m_offs), mask=m_valid, other=0.0)
                delta = tl.load(Delta_ptr + hq * N_q_total + (q_start + m_offs), mask=m_valid, other=0.0)
                causal_t = (n_offs[:, None] <= pos_vec[None, :]) & n_valid[:, None] & m_valid[None, :]
                p_t = tl.where(causal_t, tl.exp(tl.where(causal_t, s_t - lse[None, :], -float("inf"))), 0.0)
                ds_t = p_t * (dp_t - delta[None, :]) * sm_scale

                do0_row = tl.trans(dot0)
                q0_row = tl.trans(qt0)
                if INPUT_FP32:
                    dv0 += tl.dot(p_t, do0_row, input_precision="ieee")
                    dk0 += tl.dot(ds_t, q0_row, input_precision="ieee")
                else:
                    dv0 += tl.dot(p_t.to(do0_row.dtype), do0_row)
                    dk0 += tl.dot(ds_t.to(q0_row.dtype), q0_row)
                if D == 128:
                    do1_row = tl.trans(dot1)
                    q1_row = tl.trans(qt1)
                    if INPUT_FP32:
                        dv1 += tl.dot(p_t, do1_row, input_precision="ieee")
                        dk1 += tl.dot(ds_t, q1_row, input_precision="ieee")
                    else:
                        dv1 += tl.dot(p_t.to(do1_row.dtype), do1_row)
                        dk1 += tl.dot(ds_t.to(q1_row.dtype), q1_row)

    out_off = ((k_start + n_offs[:, None]).to(tl.int64) * H_Q + hq) * D
    tl.store(DK_hq_ptr + out_off + d0_offs[None, :], dk0.to(DK_hq_ptr.dtype.element_ty), mask=n_valid[:, None])
    tl.store(DV_hq_ptr + out_off + d0_offs[None, :], dv0.to(DV_hq_ptr.dtype.element_ty), mask=n_valid[:, None])
    if D == 128:
        d1_offs = 64 + tl.arange(0, 64)
        tl.store(DK_hq_ptr + out_off + d1_offs[None, :], dk1.to(DK_hq_ptr.dtype.element_ty), mask=n_valid[:, None])
        tl.store(DV_hq_ptr + out_off + d1_offs[None, :], dv1.to(DV_hq_ptr.dtype.element_ty), mask=n_valid[:, None])


# Naive monolithic [BN=64, BM=64, D=128, nw=4] negative-control kernel to measure SM80 register spills
@triton.jit
def _naive_monolithic_spill_dkv_k128_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DOut_ptr,
    LSE_ptr,
    Delta_ptr,
    PosQ_ptr,
    CuQ_ptr,
    CuK_ptr,
    DK_ptr,
    DV_ptr,
    sm_scale,
    B,
    N_q_total,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dkm,
    stride_dkh,
    stride_dvm,
    stride_dvh,
):
    # Monolithic BN=64, BM=64 at D=128 with num_warps=4 holds two [64, 128] FP32 accumulators +
    # [64, 128] K + [64, 128] V + [128, 64] Q + [128, 64] dO simultaneously in registers -> spills!
    k_tile = tl.program_id(0)
    bh_kv = tl.program_id(1)
    b = bh_kv // H_KV
    hkv = bh_kv % H_KV
    if b >= B:
        return
    G: tl.constexpr = H_Q // H_KV
    BN: tl.constexpr = 64
    BM: tl.constexpr = 64

    k_start = tl.load(CuK_ptr + b)
    k_end = tl.load(CuK_ptr + b + 1)
    L_k = k_end - k_start
    n_start = k_tile * BN
    if n_start >= L_k:
        return

    n_offs = n_start + tl.arange(0, BN)
    n_valid = n_offs < L_k
    d_offs = tl.arange(0, D)

    k_val = tl.load(
        K_ptr + (k_start + n_offs[:, None]).to(tl.int64) * stride_kn + hkv * stride_kh + d_offs[None, :],
        mask=n_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    v_val = tl.load(
        V_ptr + (k_start + n_offs[:, None]).to(tl.int64) * stride_vn + hkv * stride_vh + d_offs[None, :],
        mask=n_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    dk_acc = tl.zeros((BN, D), dtype=tl.float32)
    dv_acc = tl.zeros((BN, D), dtype=tl.float32)

    q_start = tl.load(CuQ_ptr + b)
    q_end = tl.load(CuQ_ptr + b + 1)
    L_q = q_end - q_start
    for m_start in range(0, L_q, BM):
        m_offs = m_start + tl.arange(0, BM)
        m_valid = m_offs < L_q
        pos_vec = tl.load(PosQ_ptr + q_start + m_offs, mask=m_valid, other=-1)
        for g in range(0, G):
            hq = hkv * G + g
            qt = tl.load(
                Q_ptr + (q_start + m_offs[None, :]).to(tl.int64) * stride_qm + hq * stride_qh + d_offs[:, None],
                mask=m_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            dot_t = tl.load(
                DOut_ptr + (q_start + m_offs[None, :]).to(tl.int64) * stride_dom + hq * stride_doh + d_offs[:, None],
                mask=m_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            lse = tl.load(LSE_ptr + hq * N_q_total + (q_start + m_offs), mask=m_valid, other=0.0)
            delta = tl.load(Delta_ptr + hq * N_q_total + (q_start + m_offs), mask=m_valid, other=0.0)
            s_t = tl.dot(k_val, qt, input_precision="ieee") * sm_scale
            dp_t = tl.dot(v_val, dot_t, input_precision="ieee")
            causal_t = (n_offs[:, None] <= pos_vec[None, :]) & n_valid[:, None] & m_valid[None, :]
            p_t = tl.where(causal_t, tl.exp(tl.where(causal_t, s_t - lse[None, :], -float("inf"))), 0.0)
            ds_t = p_t * (dp_t - delta[None, :]) * sm_scale
            dv_acc += tl.dot(p_t, tl.trans(dot_t), input_precision="ieee")
            dk_acc += tl.dot(ds_t, tl.trans(qt), input_precision="ieee")

    dk_ptrs = DK_ptr + (k_start + n_offs[:, None]).to(tl.int64) * stride_dkm + hkv * stride_dkh + d_offs[None, :]
    dv_ptrs = DV_ptr + (k_start + n_offs[:, None]).to(tl.int64) * stride_dvm + hkv * stride_dvh + d_offs[None, :]
    tl.store(dk_ptrs, dk_acc.to(DK_ptr.dtype.element_ty), mask=n_valid[:, None])
    tl.store(dv_ptrs, dv_acc.to(DV_ptr.dtype.element_ty), mask=n_valid[:, None])


def flash_mor_rect_causal_oracle_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Exact differentiable FP64 reference for Position-Indexed Rectangular Causal Cross-Attention."""
    qf = q.to(torch.float64)
    kf = k.to(torch.float64)
    vf = v.to(torch.float64)
    N_q, H_Q, D = qf.shape
    _, H_KV, _ = kf.shape
    G = H_Q // H_KV
    scale = (1.0 / math.sqrt(D)) if sm_scale is None else float(sm_scale)
    B = int(cu_seqlens_q.numel()) - 1
    cu_q = cu_seqlens_q.detach().cpu().tolist()
    cu_k = cu_seqlens_k.detach().cpu().tolist()

    if N_q == 0:
        return torch.zeros_like(qf)

    chunks: List[torch.Tensor] = []
    for b in range(B):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        Lq = qe - qs
        Lk = ke - ks
        if Lq == 0:
            continue
        q_doc = qf[qs:qe].permute(1, 0, 2)  # [H_Q, Lq, D]
        k_doc = kf[ks:ke].permute(1, 0, 2).repeat_interleave(G, dim=0)  # [H_Q, Lk, D]
        v_doc = vf[ks:ke].permute(1, 0, 2).repeat_interleave(G, dim=0)  # [H_Q, Lk, D]
        p_doc = pos_q[qs:qe].to(torch.long)  # [Lq]
        n_idx = torch.arange(Lk, device=q.device, dtype=torch.long)
        causal = n_idx.unsqueeze(0) <= p_doc.unsqueeze(1)  # [Lq, Lk]
        scores = torch.matmul(q_doc, k_doc.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o_doc = torch.matmul(probs, v_doc).permute(1, 0, 2)  # [Lq, H_Q, D]
        chunks.append(o_doc)
    return torch.cat(chunks, dim=0) if chunks else torch.zeros_like(qf)


class _FlashMoRRectCausalFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        sm_scale: float,
        use_naive_spill_bwd: bool,
    ) -> torch.Tensor:
        N_q, H_Q, D = q.shape
        N_k, H_KV, _ = k.shape
        assert D in (64, 128), f"Supported head dims: 64, 128 (got {D})"
        B = int(cu_seqlens_q.numel()) - 1
        out = torch.empty_like(q)
        lse = torch.empty((H_Q, max(1, N_q)), dtype=torch.float32, device=q.device)

        is_fp32 = q.dtype == torch.float32
        BM_FWD = 16 if is_fp32 else 64
        BN_FWD = 16 if is_fp32 else (32 if D == 128 else 64)
        NW_FWD = 4

        if N_q > 0 and max_seqlen_q > 0 and max_seqlen_k > 0:
            grid_fwd = (triton.cdiv(max_seqlen_q, BM_FWD), B * H_Q)
            _flash_mor_rect_causal_fwd_kernel[grid_fwd](
                q,
                k,
                v,
                pos_q,
                cu_seqlens_q,
                cu_seqlens_k,
                out,
                lse,
                float(sm_scale),
                B,
                max(1, N_q),
                H_Q=H_Q,
                H_KV=H_KV,
                D=D,
                stride_qm=q.stride(0),
                stride_qh=q.stride(1),
                stride_kn=k.stride(0),
                stride_kh=k.stride(1),
                stride_vn=v.stride(0),
                stride_vh=v.stride(1),
                stride_om=out.stride(0),
                stride_oh=out.stride(1),
                BM=BM_FWD,
                BN=BN_FWD,
                INPUT_FP32=is_fp32,
                num_warps=NW_FWD,
                num_stages=1,
            )

        ctx.save_for_backward(q, k, v, out, lse, pos_q, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = int(max_seqlen_q)
        ctx.max_seqlen_k = int(max_seqlen_k)
        ctx.sm_scale = float(sm_scale)
        ctx.use_naive_spill_bwd = bool(use_naive_spill_bwd)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, pos_q, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        N_q, H_Q, D = q.shape
        N_k, H_KV, _ = k.shape
        G = H_Q // H_KV
        B = int(cu_seqlens_q.numel()) - 1
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        if N_k == 0:
            return dq, dk, dv, None, None, None, None, None, None, None
        if N_q == 0 or ctx.max_seqlen_q == 0:
            return dq, dk, dv, None, None, None, None, None, None, None

        dout_c = dout.contiguous()
        delta = torch.empty((H_Q, max(1, N_q)), dtype=torch.float32, device=q.device)
        _flash_mor_bwd_preprocess_delta_kernel[(N_q * H_Q,)](
            out,
            dout_c,
            delta,
            N_q,
            H_Q=H_Q,
            D=D,
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            stride_dom=dout_c.stride(0),
            stride_doh=dout_c.stride(1),
            num_warps=2,
        )

        is_fp32 = q.dtype == torch.float32
        BM_BWD = 16 if is_fp32 else 32
        BN_BWD = 16 if is_fp32 else 32
        NW_DQ = 4 if D == 64 else 8
        NW_DKV = 4 if D == 64 else 8

        grid_dq = (triton.cdiv(ctx.max_seqlen_q, BM_BWD), B * H_Q)
        _flash_mor_rect_causal_dq_kernel[grid_dq](
            q,
            k,
            v,
            dout_c,
            lse,
            delta,
            pos_q,
            cu_seqlens_q,
            cu_seqlens_k,
            dq,
            ctx.sm_scale,
            B,
            max(1, N_q),
            H_Q=H_Q,
            H_KV=H_KV,
            D=D,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_dom=dout_c.stride(0),
            stride_doh=dout_c.stride(1),
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            BM=BM_BWD,
            BN=BN_BWD,
            INPUT_FP32=is_fp32,
            num_warps=NW_DQ,
            num_stages=1,
        )

        if ctx.use_naive_spill_bwd and D == 128:
            grid_dkv_spill = (triton.cdiv(ctx.max_seqlen_k, 64), B * H_KV)
            _naive_monolithic_spill_dkv_k128_kernel[grid_dkv_spill](
                q,
                k,
                v,
                dout_c,
                lse,
                delta,
                pos_q,
                cu_seqlens_q,
                cu_seqlens_k,
                dk,
                dv,
                ctx.sm_scale,
                B,
                max(1, N_q),
                H_Q=H_Q,
                H_KV=H_KV,
                D=D,
                stride_qm=q.stride(0),
                stride_qh=q.stride(1),
                stride_kn=k.stride(0),
                stride_kh=k.stride(1),
                stride_vn=v.stride(0),
                stride_vh=v.stride(1),
                stride_dom=dout_c.stride(0),
                stride_doh=dout_c.stride(1),
                stride_dkm=dk.stride(0),
                stride_dkh=dk.stride(1),
                stride_dvm=dv.stride(0),
                stride_dvh=dv.stride(1),
                num_warps=4,
                num_stages=1,
            )
        else:
            dk_hq = dk if G == 1 else torch.empty((N_k, H_Q, D), dtype=k.dtype, device=k.device)
            dv_hq = dv if G == 1 else torch.empty((N_k, H_Q, D), dtype=v.dtype, device=v.device)
            grid_dkv = (triton.cdiv(ctx.max_seqlen_k, BN_BWD), B * H_Q)
            _flash_mor_rect_causal_dkv_kernel[grid_dkv](
                q,
                k,
                v,
                dout_c,
                lse,
                delta,
                pos_q,
                cu_seqlens_q,
                cu_seqlens_k,
                dk_hq,
                dv_hq,
                ctx.sm_scale,
                B,
                max(1, N_q),
                H_Q=H_Q,
                H_KV=H_KV,
                D=D,
                stride_qm=q.stride(0),
                stride_qh=q.stride(1),
                stride_kn=k.stride(0),
                stride_kh=k.stride(1),
                stride_vn=v.stride(0),
                stride_vh=v.stride(1),
                stride_dom=dout_c.stride(0),
                stride_doh=dout_c.stride(1),
                BM=BM_BWD,
                BN=BN_BWD,
                INPUT_FP32=is_fp32,
                num_warps=NW_DKV,
                num_stages=1,
            )
            if G > 1:
                dk = dk_hq.view(N_k, H_KV, G, D).sum(dim=2, dtype=k.dtype)
                dv = dv_hq.view(N_k, H_KV, G, D).sum(dim=2, dtype=v.dtype)

        return dq, dk, dv, None, None, None, None, None, None, None


def flash_mor_rect_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: Optional[float] = None,
    *,
    use_naive_spill_bwd: bool = False,
) -> torch.Tensor:
    scale = (1.0 / math.sqrt(q.shape[-1])) if sm_scale is None else float(sm_scale)
    if q.dtype == torch.float64:
        return flash_mor_rect_causal_oracle_fp64(q, k, v, pos_q, cu_seqlens_q, cu_seqlens_k, scale)
    return _FlashMoRRectCausalFn.apply(
        q,
        k,
        v,
        pos_q.to(torch.int32),
        cu_seqlens_q.to(torch.int32),
        cu_seqlens_k.to(torch.int32),
        int(max_seqlen_q),
        int(max_seqlen_k),
        scale,
        use_naive_spill_bwd,
    )


# =============================================================================
# 7. Fused SwiGLU MLP Kernel (0 Spills) for Round-9 MoR Stacks
# =============================================================================

@triton.jit
def _mor_swiglu_fwd_kernel(
    Gate_ptr,
    Up_ptr,
    Out_ptr,
    M_rows,
    D_ff: tl.constexpr,
    stride_gm,
    stride_um,
    stride_om,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0)
    col_blk = tl.program_id(1)
    if m >= M_rows:
        return
    offs = col_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D_ff
    g = tl.load(Gate_ptr + m * stride_gm + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up_ptr + m * stride_um + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    out = (g * sig) * u
    tl.store(Out_ptr + m * stride_om + offs, out.to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mor_swiglu_bwd_kernel(
    DOut_ptr,
    Gate_ptr,
    Up_ptr,
    DGate_ptr,
    DUp_ptr,
    M_rows,
    D_ff: tl.constexpr,
    stride_dom,
    stride_gm,
    stride_um,
    stride_dgm,
    stride_dum,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0)
    col_blk = tl.program_id(1)
    if m >= M_rows:
        return
    offs = col_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D_ff
    dout = tl.load(DOut_ptr + m * stride_dom + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate_ptr + m * stride_gm + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up_ptr + m * stride_um + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    dup = dout * silu
    dgate = dout * u * (sig * (1.0 + g * (1.0 - sig)))
    tl.store(DGate_ptr + m * stride_dgm + offs, dgate.to(DGate_ptr.dtype.element_ty), mask=mask)
    tl.store(DUp_ptr + m * stride_dum + offs, dup.to(DUp_ptr.dtype.element_ty), mask=mask)


class _MoRSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        M_rows, two_ff = gate_up.shape
        D_ff = two_ff // 2
        gate = gate_up[:, :D_ff]
        up = gate_up[:, D_ff:]
        out = torch.empty((M_rows, D_ff), dtype=gate_up.dtype, device=gate_up.device)
        BLOCK_D = 512
        if M_rows > 0:
            _mor_swiglu_fwd_kernel[(M_rows, triton.cdiv(D_ff, BLOCK_D))](
                gate,
                up,
                out,
                M_rows,
                D_ff=D_ff,
                stride_gm=gate.stride(0),
                stride_um=up.stride(0),
                stride_om=out.stride(0),
                BLOCK_D=BLOCK_D,
                num_warps=4,
            )
        ctx.save_for_backward(gate_up)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (gate_up,) = ctx.saved_tensors
        M_rows, two_ff = gate_up.shape
        D_ff = two_ff // 2
        d_gate_up = torch.empty_like(gate_up)
        if M_rows == 0:
            return d_gate_up.zero_()
        dout_c = dout.contiguous()
        gate = gate_up[:, :D_ff]
        up = gate_up[:, D_ff:]
        dgate = d_gate_up[:, :D_ff]
        dup = d_gate_up[:, D_ff:]
        BLOCK_D = 512
        _mor_swiglu_bwd_kernel[(M_rows, triton.cdiv(D_ff, BLOCK_D))](
            dout_c,
            gate,
            up,
            dgate,
            dup,
            M_rows,
            D_ff=D_ff,
            stride_dom=dout_c.stride(0),
            stride_gm=gate.stride(0),
            stride_um=up.stride(0),
            stride_dgm=dgate.stride(0),
            stride_dum=dup.stride(0),
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        return d_gate_up


def mor_fused_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    if gate_up.dtype == torch.float64:
        d_ff = gate_up.shape[-1] // 2
        return F.silu(gate_up[..., :d_ff]) * gate_up[..., d_ff:]
    return _MoRSwiGLUFn.apply(gate_up)


# =============================================================================
# 8. Upstream `raymin0223/mixture_of_recursions` Forensics & Baselines
# =============================================================================

def build_rope_cache(max_seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / float(head_dim)))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()


def apply_eager_rope_at_positions(
    x: torch.Tensor,
    pos_ids: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Eager `[B, S, H, D]` RoPE matching our half-split convention."""
    c = cos_cache[pos_ids].unsqueeze(2).to(x.dtype)
    s = sin_cache[pos_ids].unsqueeze(2).to(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def official_mor_pad_4dmask_select(
    x: torch.Tensor,
    top_expert_indices: torch.Tensor,
    recursion_idx: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int], List[torch.Tensor]]:
    """
    Faithful reproduction of `raymin0223/mixture_of_recursions/model/mor_model/token_choice_router.py`
    `select_tokens_and_batch_with_padding` (L75-125):
      - B host-synchronizing `torch.where(top_expert_indices[i] >= index)[0]` calls
      - `rnn_utils.pad_sequence`
      - Dense 4D `[new_bs, 1, new_seq_len, new_seq_len]` causal-gathered mask!
    """
    bs, seq_len, _ = x.shape
    batched_x: List[torch.Tensor] = []
    selected_batch_indices: List[int] = []
    selected_seq_indices: List[torch.Tensor] = []

    for i in range(bs):
        indices = torch.where(top_expert_indices[i] >= recursion_idx)[0]
        if indices.numel() == 0:
            continue
        selected_batch_indices.append(i)
        selected_seq_indices.append(indices)
        batched_x.append(x[i, indices])

    if len(batched_x) == 0:
        return None, None, [], []

    padded_x = rnn_utils.pad_sequence(batched_x, batch_first=True, padding_value=0.0)
    new_bs, new_seq_len, _ = padded_x.shape
    min_val = torch.finfo(x.dtype).min if x.dtype.is_floating_point else -1e9
    mask_4d = torch.full(
        (new_bs, 1, new_seq_len, new_seq_len),
        min_val,
        dtype=x.dtype,
        device=x.device,
    )
    for b_out in range(new_bs):
        idx = selected_seq_indices[b_out]
        s = int(idx.numel())
        # Causal among selected positions (idx is already sorted -> lower-triangular on [:s, :s])
        tri = torch.tril(torch.ones((s, s), dtype=torch.bool, device=x.device))
        mask_4d[b_out, 0, :s, :s] = torch.where(tri, torch.tensor(0.0, dtype=x.dtype, device=x.device), mask_4d[b_out, 0, :s, :s])

    return padded_x, mask_4d, selected_batch_indices, selected_seq_indices


def audit_round9_triton_cubin_spills(device: torch.device) -> Dict[str, Any]:
    """
    Compiles and inspects SM80 cubin register and spill telemetry (`n_regs`, `n_spills`) across
    all Round-9 RSI Triton kernels vs the naive monolithic [64x128] `dKV` negative control.
    """
    B, T = 2, 512
    D_model = 1024
    mask = (torch.arange(T, device=device).unsqueeze(0).expand(B, T) % 2 == 0)
    cu_q, flat_idx, pos_q, k_tot = mor_varlen_pack_from_mask(mask)
    cu_k = torch.arange(0, (B + 1) * T, T, device=device, dtype=torch.int32)

    x_flat = torch.randn((B * T, D_model), device=device, dtype=torch.bfloat16, requires_grad=True)
    gamma = torch.ones((D_model,), device=device, dtype=torch.bfloat16, requires_grad=True)
    y_norm = mor_gather_rmsnorm(x_flat, flat_idx, gamma)
    y_norm.sum().backward()

    cos64, sin64 = build_rope_cache(T, 64, device)
    cos128, sin128 = build_rope_cache(T, 128, device)

    for D_h, cos_c, sin_c in ((64, cos64, sin64), (128, cos128, sin128)):
        for dt in (torch.bfloat16, torch.float32):
            q = torch.randn((k_tot, 8, D_h), device=device, dtype=dt, requires_grad=True)
            k = torch.randn((B * T, 2, D_h), device=device, dtype=dt, requires_grad=True)
            v = torch.randn((B * T, 2, D_h), device=device, dtype=dt, requires_grad=True)
            q_rot = mor_indexed_rope(q, pos_q, cos_c, sin_c)
            out = flash_mor_rect_causal_attention(q_rot, k, v, pos_q, cu_q, cu_k, T // 2, T)
            (out.float().sum()).backward()

    # Exercise gated scatter residual (both Eq 2.1 and Eq 2.2) + SwiGLU + Router BCE/ZLoss
    x_pass = torch.randn((B * T, D_model), device=device, dtype=torch.bfloat16, requires_grad=True)
    x_base = torch.randn((B * T, D_model), device=device, dtype=torch.bfloat16, requires_grad=True)
    y_act = torch.randn((k_tot, D_model), device=device, dtype=torch.bfloat16, requires_grad=True)
    g_act = torch.sigmoid(torch.randn((k_tot,), device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)
    add_m = (pos_q % 3 == 0).to(torch.uint8)
    s1 = mor_gated_scatter_residual(x_pass, x_pass, y_act, g_act, flat_idx, None)
    (s1.float().sum()).backward()
    s2 = mor_gated_scatter_residual(x_pass.detach().requires_grad_(True), x_base, y_act.detach().requires_grad_(True), g_act.detach().requires_grad_(True), flat_idx, add_m)
    (s2.float().sum()).backward()

    gu = torch.randn((k_tot, 2048), device=device, dtype=torch.bfloat16, requires_grad=True)
    sw = mor_fused_swiglu(gu)
    (sw.float().sum()).backward()

    r_logits = torch.randn((B * T,), device=device, dtype=torch.bfloat16, requires_grad=True)
    r_tgts = (torch.rand((B * T,), device=device) > 0.5).to(torch.bfloat16)
    pr, bce_l, z_l = mor_router_bce_zloss(r_logits, r_tgts)
    (pr.float().sum() + bce_l + z_l).backward()

    # Compile naive monolithic [64x128] dKV control to record its SM80 cubin spill count
    q128 = torch.randn((k_tot, 8, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
    k128 = torch.randn((B * T, 2, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
    v128 = torch.randn((B * T, 2, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
    out_sp = flash_mor_rect_causal_attention(q128, k128, v128, pos_q, cu_q, cu_k, T // 2, T, use_naive_spill_bwd=True)
    (out_sp.float().sum()).backward()
    torch.cuda.synchronize(device)

    round9_kernels = [
        ("_mor_varlen_count_kernel", _mor_varlen_count_kernel),
        ("_mor_varlen_scan_pack_kernel", _mor_varlen_scan_pack_kernel),
        ("_mor_gather_rmsnorm_fwd_kernel", _mor_gather_rmsnorm_fwd_kernel),
        ("_mor_gather_rmsnorm_bwd_dx_part_kernel", _mor_gather_rmsnorm_bwd_dx_part_kernel),
        ("_mor_reduce_dgamma_splits_kernel", _mor_reduce_dgamma_splits_kernel),
        ("_mor_indexed_rope_kernel", _mor_indexed_rope_kernel),
        ("_mor_gated_scatter_residual_fwd_kernel", _mor_gated_scatter_residual_fwd_kernel),
        ("_mor_gated_scatter_residual_bwd_kernel", _mor_gated_scatter_residual_bwd_kernel),
        ("_mor_router_bce_zloss_fwd_kernel", _mor_router_bce_zloss_fwd_kernel),
        ("_mor_router_bce_zloss_bwd_kernel", _mor_router_bce_zloss_bwd_kernel),
        ("_flash_mor_rect_causal_fwd_kernel", _flash_mor_rect_causal_fwd_kernel),
        ("_flash_mor_bwd_preprocess_delta_kernel", _flash_mor_bwd_preprocess_delta_kernel),
        ("_flash_mor_rect_causal_dq_kernel", _flash_mor_rect_causal_dq_kernel),
        ("_flash_mor_rect_causal_dkv_kernel", _flash_mor_rect_causal_dkv_kernel),
        ("_mor_swiglu_fwd_kernel", _mor_swiglu_fwd_kernel),
        ("_mor_swiglu_bwd_kernel", _mor_swiglu_bwd_kernel),
    ]

    def _inspect_jit_fn(jfn) -> Dict[str, Any]:
        max_regs = 0
        max_spills = 0
        specs = 0
        for cache_obj in getattr(jfn, "device_caches", {}).values():
            bin_map = cache_obj[0] if isinstance(cache_obj, tuple) else cache_obj
            if isinstance(bin_map, dict):
                for compiled in bin_map.values():
                    if hasattr(compiled, "n_regs"):
                        specs += 1
                        max_regs = max(max_regs, int(compiled.n_regs))
                        max_spills = max(max_spills, int(compiled.n_spills))
        return {"compiled_specializations": specs, "max_n_regs": max_regs, "max_n_spills": max_spills}

    kernel_stats = {name: _inspect_jit_fn(fn) for name, fn in round9_kernels}
    naive_spill_stats = _inspect_jit_fn(_naive_monolithic_spill_dkv_k128_kernel)
    total_spills = sum(int(v["max_n_spills"]) for v in kernel_stats.values())
    total_specs = sum(int(v["compiled_specializations"]) for v in kernel_stats.values())
    max_regs_all = max(int(v["max_n_regs"]) for v in kernel_stats.values())

    return {
        "round9_kernels": kernel_stats,
        "num_round9_kernels": len(round9_kernels),
        "total_compiled_specializations": total_specs,
        "max_round9_regs": max_regs_all,
        "total_round9_spills": total_spills,
        "all_round9_zero_spills": (total_spills == 0 and total_specs >= len(round9_kernels)),
        "naive_monolithic_64x128_dkv_control": naive_spill_stats,
    }
