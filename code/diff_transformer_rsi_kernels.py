#!/usr/bin/env python3
"""
E136 (v127, Round-10 RSI): Differential Transformer V1 & V2 Zero-Spill Triton Kernels
=====================================================================================

Paper & Official Repository Grounding:
  - Paper: "Differential Transformer" (Ye, Dong, Xia, Sun, Zhu, Huang & Wei,
    Microsoft Research, arXiv:2410.05258, ICLR 2025) + January 2026 "Differential
    Transformer V2" (DIFF V2 architectural evolution).
  - Official Repository: https://github.com/microsoft/unilm/tree/master/Diff-Transformer
    and https://github.com/microsoft/unilm/tree/master/Diff-Transformer/Diff-Transformer-V2

Upstream Pathologies Diagnosed & Resolved:
  1. `multihead_diffattn.py:80,86,101` (GQA `repeat_kv` Pre-Split Channel-Pairing Bug):
     Reshaping `k` to `(B, T, 2 * H_kv, d)` and calling `repeat_kv(k.transpose(1, 2), n_rep)`
     before `.view(B, H, 2, T, T)` repeats each KV head `n_rep` times consecutively. For any
     even `n_rep >= 2` (GQA), stream 0 (`2h`) and stream 1 (`2h+1`) receive identical KV heads
     `(k_{g,0}, k_{g,0})` or `(k_{g,1}, k_{g,1})` instead of `(k_{g,0}, k_{g,1})`, diverging
     by >45% relative L2 error from `multihead_flashdiff_1.py:79-84`.
  2. `multihead_flashdiff_2.py:90-98` (`4x` FlashAttention Launches + `2x` `torch.cat`):
     When `d_v = 2 * d_qk` is unsupported by standard FlashAttention, upstream splits
     `V -> (V1, V2)` and calls `flash_attn_func` 4 times (`attn11, attn12, attn21, attn22`),
     recomputing `softmax(Q1 K1^T)` and `softmax(Q2 K2^T)` twice in forward and backward.
  3. `multihead_flashdiff_1.py:88-96` & `rms_norm.py` (`2x` Redundant V Reads + 4 Unfused
     HBM Epilogue Passes):
     Calls `flash_attn_func` twice on shared `V` (`[B, T, H_kv, 2d]`), then executes unfused
     `attn1 - lambda_full * attn2`, unfused FP32-upcast `RMSNorm(2d)`, and unfused
     `*(1 - lambda_init)`.
  4. `Diff-Transformer-V2/multihead_flashdiffv2.py:9-11,69-73` (`[B, T, 2H, d]` HBM Materialization
     + Stride-2 `attn[:, :, 0::2] / 1::2` Views + Duplicate Shared-KV Reads):
     Paired query heads `(2h, 2h+1)` belong to the same GQA group (`h_kv = h // n_rep`) and
     share identical `(K_{hkv}, V_{hkv})`, yet upstream calls `flash_attn_func(q, k, v)` on
     `2H` heads (reading `K, V` twice and writing a `2x`-wider `[B, T, 2H, d]` tensor to HBM)
     before slicing stride-2 views into a separate `diff_func` kernel.
  5. `kernel/rotary.py:124-156,204,253` (Interleaved RoPE `rk_swap` Double Load + `BLOCK_M=4`
     + Unconditional `do = do.clone()` in Non-Interleaved Backward).

Round-10 RSI Kernel Inventory (All 0 Cubin Register Spills on SM80):
  1. `_fused_interleaved_rope_fwd_kernel` & `_fused_interleaved_rope_bwd_kernel`
  2. `_flash_diffv2_fwd_kernel` & `_flash_diffv2_fwd_d128_kernel`
  3. `_flash_diffv2_bwd_prep_kernel`
  4. `_flash_diffv2_bwd_dq_kernel` & `_flash_diffv2_bwd_dq_d128_kernel`
  5. `_flash_diffv2_bwd_dkv_kernel` & `_flash_diffv2_bwd_dkv_d128_kernel`
  6. `_flash_diffv1_fwd_slice_kernel`
  7. `_diffv1_subln_scale_fwd_kernel` & `_diffv1_subln_scale_bwd_kernel`
  8. `_flash_diffv1_bwd_dq_kernel` & `_flash_diffv1_bwd_dkv_kernel`
  9. Negative-control spill kernel `_flash_diffv2_bwd_monolithic_spill_kernel` (demonstrates
     SM80 cubin register spilling when holding `[BM=64, BN=64, D=128]` paired streams in a
     single monolithic CTA without 64+64 slice staging).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# =============================================================================
# 1. Single-Pass Coalesced Interleaved RoPE Kernels (Fixes kernel/rotary.py)
# =============================================================================

@triton.jit
def _fused_interleaved_rope_fwd_kernel(
    X_ptr,
    COS_ptr,
    SIN_ptr,
    OUT_ptr,
    stride_xb, stride_xt, stride_xh, stride_xd,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_ct, stride_cd,
    T,
    H,
    HALF_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d2 = tl.arange(0, HALF_D)
    mask_m = offs_m < T

    cos = tl.load(
        COS_ptr + offs_m[:, None] * stride_ct + offs_d2[None, :] * stride_cd,
        mask=mask_m[:, None],
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        SIN_ptr + offs_m[:, None] * stride_ct + offs_d2[None, :] * stride_cd,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    base_x = X_ptr + b * stride_xb + h * stride_xh + offs_m[:, None] * stride_xt
    x_even = tl.load(
        base_x + (offs_d2[None, :] * 2) * stride_xd,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    x_odd = tl.load(
        base_x + (offs_d2[None, :] * 2 + 1) * stride_xd,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    y_even = x_even * cos - x_odd * sin
    y_odd = x_odd * cos + x_even * sin

    base_o = OUT_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot
    tl.store(
        base_o + (offs_d2[None, :] * 2) * stride_od,
        y_even.to(OUT_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )
    tl.store(
        base_o + (offs_d2[None, :] * 2 + 1) * stride_od,
        y_odd.to(OUT_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


@triton.jit
def _fused_interleaved_rope_bwd_kernel(
    DOUT_ptr,
    COS_ptr,
    SIN_ptr,
    DX_ptr,
    stride_dob, stride_dot, stride_doh, stride_dod,
    stride_dxb, stride_dxt, stride_dxh, stride_dxd,
    stride_ct, stride_cd,
    T,
    H,
    HALF_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d2 = tl.arange(0, HALF_D)
    mask_m = offs_m < T

    cos = tl.load(
        COS_ptr + offs_m[:, None] * stride_ct + offs_d2[None, :] * stride_cd,
        mask=mask_m[:, None],
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        SIN_ptr + offs_m[:, None] * stride_ct + offs_d2[None, :] * stride_cd,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    base_do = DOUT_ptr + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dot
    dy_even = tl.load(
        base_do + (offs_d2[None, :] * 2) * stride_dod,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    dy_odd = tl.load(
        base_do + (offs_d2[None, :] * 2 + 1) * stride_dod,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    dx_even = dy_even * cos + dy_odd * sin
    dx_odd = dy_odd * cos - dy_even * sin

    base_dx = DX_ptr + b * stride_dxb + h * stride_dxh + offs_m[:, None] * stride_dxt
    tl.store(
        base_dx + (offs_d2[None, :] * 2) * stride_dxd,
        dx_even.to(DX_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )
    tl.store(
        base_dx + (offs_d2[None, :] * 2 + 1) * stride_dxd,
        dx_odd.to(DX_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


class _FusedInterleavedRoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, H, D = x.shape
        assert D % 2 == 0
        out = torch.empty_like(x)
        BM = 32
        grid = (triton.cdiv(T, BM), B * H)
        _fused_interleaved_rope_fwd_kernel[grid](
            x, cos, sin, out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cos.stride(0), cos.stride(1),
            T, H,
            HALF_D=D // 2,
            BLOCK_M=BM,
            num_warps=4,
        )
        ctx.save_for_backward(cos, sin)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        cos, sin = ctx.saved_tensors
        B, T, H, D = dout.shape
        dx = torch.empty_like(dout)
        BM = 32
        grid = (triton.cdiv(T, BM), B * H)
        _fused_interleaved_rope_bwd_kernel[grid](
            dout, cos, sin, dx,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            cos.stride(0), cos.stride(1),
            T, H,
            HALF_D=D // 2,
            BLOCK_M=BM,
            num_warps=4,
        )
        return dx, None, None


@triton.jit
def _fused_interleaved_rope_qk_fwd_kernel(
    Q_ptr, K_ptr, COS_ptr, SIN_ptr, QO_ptr, KO_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_ct, stride_cd,
    T, H_Q, H_KV,
    HALF_D: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    H_TOT = H_Q + H_KV
    b = pid_bh // H_TOT
    h_idx = pid_bh % H_TOT

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d2 = tl.arange(0, HALF_D)
    mask_m = offs_m < T

    cos = tl.load(COS_ptr + offs_m[:, None] * stride_ct + offs_d2[None, :] * stride_cd, mask=mask_m[:, None], other=1.0).to(tl.float32)
    sin = tl.load(SIN_ptr + offs_m[:, None] * stride_ct + offs_d2[None, :] * stride_cd, mask=mask_m[:, None], other=0.0).to(tl.float32)

    if h_idx < H_Q:
        base_in = Q_ptr + b * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qt
        base_out = QO_ptr + b * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qt
        s_d = stride_qd
    else:
        hk = h_idx - H_Q
        base_in = K_ptr + b * stride_kb + hk * stride_kh + offs_m[:, None] * stride_kt
        base_out = KO_ptr + b * stride_kb + hk * stride_kh + offs_m[:, None] * stride_kt
        s_d = stride_kd

    x_even = tl.load(base_in + (offs_d2[None, :] * 2) * s_d, mask=mask_m[:, None], other=0.0).to(tl.float32)
    x_odd = tl.load(base_in + (offs_d2[None, :] * 2 + 1) * s_d, mask=mask_m[:, None], other=0.0).to(tl.float32)
    y_even = x_even * cos - x_odd * sin
    y_odd = x_odd * cos + x_even * sin
    tl.store(base_out + (offs_d2[None, :] * 2) * s_d, y_even.to(QO_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(base_out + (offs_d2[None, :] * 2 + 1) * s_d, y_odd.to(QO_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _fused_interleaved_rope_qk_bwd_kernel(
    DQO_ptr, DKO_ptr, COS_ptr, SIN_ptr, DQ_ptr, DK_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_ct, stride_cd,
    T, H_Q, H_KV,
    HALF_D: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    H_TOT = H_Q + H_KV
    b = pid_bh // H_TOT
    h_idx = pid_bh % H_TOT

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d2 = tl.arange(0, HALF_D)
    mask_m = offs_m < T

    cos = tl.load(COS_ptr + offs_m[:, None] * stride_ct + offs_d2[None, :] * stride_cd, mask=mask_m[:, None], other=1.0).to(tl.float32)
    sin = tl.load(SIN_ptr + offs_m[:, None] * stride_ct + offs_d2[None, :] * stride_cd, mask=mask_m[:, None], other=0.0).to(tl.float32)

    if h_idx < H_Q:
        base_in = DQO_ptr + b * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qt
        base_out = DQ_ptr + b * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qt
        s_d = stride_qd
    else:
        hk = h_idx - H_Q
        base_in = DKO_ptr + b * stride_kb + hk * stride_kh + offs_m[:, None] * stride_kt
        base_out = DK_ptr + b * stride_kb + hk * stride_kh + offs_m[:, None] * stride_kt
        s_d = stride_kd

    dy_even = tl.load(base_in + (offs_d2[None, :] * 2) * s_d, mask=mask_m[:, None], other=0.0).to(tl.float32)
    dy_odd = tl.load(base_in + (offs_d2[None, :] * 2 + 1) * s_d, mask=mask_m[:, None], other=0.0).to(tl.float32)
    dx_even = dy_even * cos + dy_odd * sin
    dx_odd = dy_odd * cos - dy_even * sin
    tl.store(base_out + (offs_d2[None, :] * 2) * s_d, dx_even.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(base_out + (offs_d2[None, :] * 2 + 1) * s_d, dx_odd.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])


class _FusedInterleavedRoPEQKFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, T, H_Q, D = q.shape
        H_KV = k.shape[2]
        qo = torch.empty_like(q)
        ko = torch.empty_like(k)
        BM = 64
        _fused_interleaved_rope_qk_fwd_kernel[(triton.cdiv(T, BM), B * (H_Q + H_KV))](
            q, k, cos, sin, qo, ko,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            cos.stride(0), cos.stride(1),
            T, H_Q, H_KV, HALF_D=D // 2, BLOCK_M=BM, num_warps=4,
        )
        ctx.save_for_backward(cos, sin)
        ctx.shapes = (B, T, H_Q, H_KV, D)
        return qo, ko

    @staticmethod
    def backward(ctx, dqo: torch.Tensor, dko: torch.Tensor):
        cos, sin = ctx.saved_tensors
        B, T, H_Q, H_KV, D = ctx.shapes
        dqo = dqo.contiguous()
        dko = dko.contiguous()
        dq = torch.empty_like(dqo)
        dk = torch.empty_like(dko)
        BM = 64
        _fused_interleaved_rope_qk_bwd_kernel[(triton.cdiv(T, BM), B * (H_Q + H_KV))](
            dqo, dko, cos, sin, dq, dk,
            dqo.stride(0), dqo.stride(1), dqo.stride(2), dqo.stride(3),
            dko.stride(0), dko.stride(1), dko.stride(2), dko.stride(3),
            cos.stride(0), cos.stride(1),
            T, H_Q, H_KV, HALF_D=D // 2, BLOCK_M=BM, num_warps=4,
        )
        return dq, dk, None, None


def fused_interleaved_rope_qk_rsi(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return _FusedInterleavedRoPEQKFn.apply(q, k, cos, sin)


def fused_interleaved_rope_rsi(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _FusedInterleavedRoPEFn.apply(x, cos, sin)


# =============================================================================
# 2. Upstream Official Reference Implementations & Bug Reproducers
# =============================================================================

def upstream_interleaved_rope_eager(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact mathematical equivalent of microsoft/unilm/Diff-Transformer/kernel/rotary.py (interleaved=True)."""
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x1 = x[..., 0::2].to(work_dtype)
    x2 = x[..., 1::2].to(work_dtype)
    c = cos[None, :, None, :].to(work_dtype)
    s = sin[None, :, None, :].to(work_dtype)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.stack([o1, o2], dim=-1).flatten(-2).to(x.dtype)


@triton.jit
def _upstream_rotary_kernel_repro(
    OUT, X, COS, SIN,
    SEQLEN, HEADS,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    stride_cos_seqlen, stride_cos_rotary,
    ROTARY_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IS_INTERLEAVED: tl.constexpr,
):
    """Verbatim reproduction of microsoft/unilm/Diff-Transformer/kernel/rotary.py lines 124-156."""
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, ROTARY_DIM // 2)

    if IS_INTERLEAVED:
        rk = tl.arange(0, ROTARY_DIM)
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1
        COS = COS + (rm[:, None] * stride_cos_seqlen + (rk // 2)[None, :] * stride_cos_rotary)
        SIN = SIN + (rm[:, None] * stride_cos_seqlen + (rk // 2)[None, :] * stride_cos_rotary)
        cos = tl.load(COS, mask=rm[:, None] < SEQLEN, other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=rm[:, None] < SEQLEN, other=0.0).to(tl.float32)
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        x0 = tl.load(X0, mask=rm[:, None] < SEQLEN, other=0.0).to(tl.float32)
        x1 = tl.load(X1, mask=rm[:, None] < SEQLEN, other=0.0).to(tl.float32)
        out = x0 * cos + x1 * tl.where(rk[None, :] % 2 == 0, -sin, sin)
        OUT_ptrs = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT_ptrs, out.to(OUT.dtype.element_ty), mask=rm[:, None] < SEQLEN)


class _UpstreamRotaryEmbFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True, inplace: bool = False):
        B, T, H, D = x.shape
        out = x if inplace else torch.empty_like(x)
        BM = 4 if interleaved else 8
        grid = (triton.cdiv(T, BM), B, H)
        _upstream_rotary_kernel_repro[grid](
            out, x, cos, sin, T, H,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            cos.stride(0), cos.stride(1),
            ROTARY_DIM=D, BLOCK_M=BM, IS_INTERLEAVED=interleaved,
            num_warps=4,
        )
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        cos, sin = ctx.saved_tensors
        # Upstream kernel/rotary.py:253 clone hazard when not interleaved and not inplace
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        B, T, H, D = do.shape
        dx = do if ctx.inplace else torch.empty_like(do)
        BM = 4 if ctx.interleaved else 8
        grid = (triton.cdiv(T, BM), B, H)
        _upstream_rotary_kernel_repro[grid](
            dx, do, cos, -sin, T, H,
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            cos.stride(0), cos.stride(1),
            ROTARY_DIM=D, BLOCK_M=BM, IS_INTERLEAVED=ctx.interleaved,
            num_warps=4,
        )
        return dx, None, None, None, None


def upstream_apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    return _UpstreamRotaryEmbFn.apply(x, cos, sin, interleaved, False)


# =============================================================================
# 3. Co-Scheduled Paired-Query Flash-DiffAttn-V2 Kernels (0 Cubin Spills)
# =============================================================================

@triton.jit
def _flash_diffv2_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, LAM_ptr,
    O_ptr, O2_ptr, LSE1_ptr, LSE2_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_lb, stride_lt, stride_lh,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Co-scheduled paired-query Flash-DiffAttn-V2 forward for D=64.
    Loads shared (K_{hkv}, V_{hkv}) ONCE per paired heads (2h, 2h+1).
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    h_kv = h // N_REP

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T

    q1_ptrs = Q_ptr + b * stride_qb + (2 * h) * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q2_ptrs = Q_ptr + b * stride_qb + (2 * h + 1) * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q1 = tl.load(q1_ptrs, mask=mask_m[:, None], other=0.0)
    q2 = tl.load(q2_ptrs, mask=mask_m[:, None], other=0.0)

    lam_logit = tl.load(
        LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    lam = tl.sigmoid(lam_logit)

    m1 = tl.full([BM], -float("inf"), dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    acc1 = tl.zeros([BM, D], dtype=tl.float32)

    m2 = tl.full([BM], -float("inf"), dtype=tl.float32)
    l2 = tl.zeros([BM], dtype=tl.float32)
    acc2 = tl.zeros([BM, D], dtype=tl.float32)

    if IS_CAUSAL:
        max_q_global = q_offset + (pid_m + 1) * BM - 1
        max_local_k = max_q_global - k_offset + 1
        n_limit = tl.minimum(T, tl.maximum(0, max_local_k))
    else:
        n_limit = T

    k_base = K_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh

    for n_start in range(0, n_limit, BN):
        offs_n = n_start + tl.arange(0, BN)
        mask_n = offs_n < T

        k = tl.load(
            k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        s1 = tl.dot(q1, tl.trans(k), allow_tf32=ALLOW_TF32) * sm_scale
        s2 = tl.dot(q2, tl.trans(k), allow_tf32=ALLOW_TF32) * sm_scale

        if IS_CAUSAL:
            q_glob = q_offset + offs_m[:, None]
            k_glob = k_offset + offs_n[None, :]
            valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        s1 = tl.where(valid, s1, -float("inf"))
        s2 = tl.where(valid, s2, -float("inf"))

        m1_ij = tl.max(s1, axis=1)
        m1_new = tl.maximum(m1, m1_ij)
        m1_safe = tl.where(m1_new == -float("inf"), 0.0, m1_new)
        alpha1 = tl.exp(tl.where(m1 == -float("inf"), -float("inf"), m1 - m1_safe))
        p1 = tl.exp(tl.where(s1 == -float("inf"), -float("inf"), s1 - m1_safe[:, None]))
        l1 = l1 * alpha1 + tl.sum(p1, axis=1)

        m2_ij = tl.max(s2, axis=1)
        m2_new = tl.maximum(m2, m2_ij)
        m2_safe = tl.where(m2_new == -float("inf"), 0.0, m2_new)
        alpha2 = tl.exp(tl.where(m2 == -float("inf"), -float("inf"), m2 - m2_safe))
        p2 = tl.exp(tl.where(s2 == -float("inf"), -float("inf"), s2 - m2_safe[:, None]))
        l2 = l2 * alpha2 + tl.sum(p2, axis=1)

        v = tl.load(
            v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )
        acc1 = acc1 * alpha1[:, None] + tl.dot(p1.to(v.dtype), v, allow_tf32=ALLOW_TF32)
        acc2 = acc2 * alpha2[:, None] + tl.dot(p2.to(v.dtype), v, allow_tf32=ALLOW_TF32)
        m1 = m1_new
        m2 = m2_new

    inv_l1 = tl.where(l1 > 0.0, 1.0 / l1, 0.0)
    inv_l2 = tl.where(l2 > 0.0, 1.0 / l2, 0.0)
    o1 = acc1 * inv_l1[:, None]
    o2 = acc2 * inv_l2[:, None]
    o = o1 - lam[:, None] * o2

    lse1 = tl.where(l1 > 0.0, m1 + tl.log(l1), -float("inf"))
    lse2 = tl.where(l2 > 0.0, m2 + tl.log(l2), -float("inf"))

    tl.store(
        LSE1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t,
        lse1,
        mask=mask_m,
    )
    tl.store(
        LSE2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t,
        lse2,
        mask=mask_m,
    )
    o_ptrs = O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    o2_ptrs = O2_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(o2_ptrs, o2.to(O2_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _flash_diffv2_fwd_d128_kernel(
    Q_ptr, K_ptr, V_ptr, LAM_ptr,
    O_ptr, O2_ptr, LSE1_ptr, LSE2_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_lb, stride_lt, stride_lh,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Zero-spill Two-Slice (64+64) Co-Scheduled Paired-Query Flash-DiffAttn-V2 forward for D=128.
    pid_slice in {0, 1} processes the V[0:64] or V[64:128] output slice so each CTA only holds
    two [BM, 64] FP32 accumulators in registers (0 cubin spills).
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_slice = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H
    h_kv = h // N_REP

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d0 = tl.arange(0, 64)
    offs_d1 = 64 + tl.arange(0, 64)
    offs_out = pid_slice * 64 + tl.arange(0, 64)
    mask_m = offs_m < T

    q1_base = Q_ptr + b * stride_qb + (2 * h) * stride_qh + offs_m[:, None] * stride_qt
    q2_base = Q_ptr + b * stride_qb + (2 * h + 1) * stride_qh + offs_m[:, None] * stride_qt
    q1_0 = tl.load(q1_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q1_1 = tl.load(q1_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q2_0 = tl.load(q2_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q2_1 = tl.load(q2_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

    lam_logit = tl.load(
        LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    lam = tl.sigmoid(lam_logit)

    m1 = tl.full([BM], -float("inf"), dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    acc1 = tl.zeros([BM, 64], dtype=tl.float32)

    m2 = tl.full([BM], -float("inf"), dtype=tl.float32)
    l2 = tl.zeros([BM], dtype=tl.float32)
    acc2 = tl.zeros([BM, 64], dtype=tl.float32)

    if IS_CAUSAL:
        max_q_global = q_offset + (pid_m + 1) * BM - 1
        max_local_k = max_q_global - k_offset + 1
        n_limit = tl.minimum(T, tl.maximum(0, max_local_k))
    else:
        n_limit = T

    k_base = K_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh

    for n_start in range(0, n_limit, BN):
        offs_n = n_start + tl.arange(0, BN)
        mask_n = offs_n < T

        k0 = tl.load(
            k_base + offs_n[:, None] * stride_kt + offs_d0[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        k1 = tl.load(
            k_base + offs_n[:, None] * stride_kt + offs_d1[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        s1 = (tl.dot(q1_0, tl.trans(k0), allow_tf32=ALLOW_TF32) + tl.dot(q1_1, tl.trans(k1), allow_tf32=ALLOW_TF32)) * sm_scale
        s2 = (tl.dot(q2_0, tl.trans(k0), allow_tf32=ALLOW_TF32) + tl.dot(q2_1, tl.trans(k1), allow_tf32=ALLOW_TF32)) * sm_scale

        if IS_CAUSAL:
            q_glob = q_offset + offs_m[:, None]
            k_glob = k_offset + offs_n[None, :]
            valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        s1 = tl.where(valid, s1, -float("inf"))
        s2 = tl.where(valid, s2, -float("inf"))

        m1_ij = tl.max(s1, axis=1)
        m1_new = tl.maximum(m1, m1_ij)
        m1_safe = tl.where(m1_new == -float("inf"), 0.0, m1_new)
        alpha1 = tl.exp(tl.where(m1 == -float("inf"), -float("inf"), m1 - m1_safe))
        p1 = tl.exp(tl.where(s1 == -float("inf"), -float("inf"), s1 - m1_safe[:, None]))
        l1 = l1 * alpha1 + tl.sum(p1, axis=1)

        m2_ij = tl.max(s2, axis=1)
        m2_new = tl.maximum(m2, m2_ij)
        m2_safe = tl.where(m2_new == -float("inf"), 0.0, m2_new)
        alpha2 = tl.exp(tl.where(m2 == -float("inf"), -float("inf"), m2 - m2_safe))
        p2 = tl.exp(tl.where(s2 == -float("inf"), -float("inf"), s2 - m2_safe[:, None]))
        l2 = l2 * alpha2 + tl.sum(p2, axis=1)

        v_slice = tl.load(
            v_base + offs_n[:, None] * stride_vt + offs_out[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )
        acc1 = acc1 * alpha1[:, None] + tl.dot(p1.to(v_slice.dtype), v_slice, allow_tf32=ALLOW_TF32)
        acc2 = acc2 * alpha2[:, None] + tl.dot(p2.to(v_slice.dtype), v_slice, allow_tf32=ALLOW_TF32)
        m1 = m1_new
        m2 = m2_new

    inv_l1 = tl.where(l1 > 0.0, 1.0 / l1, 0.0)
    inv_l2 = tl.where(l2 > 0.0, 1.0 / l2, 0.0)
    o1 = acc1 * inv_l1[:, None]
    o2 = acc2 * inv_l2[:, None]
    o = o1 - lam[:, None] * o2

    if pid_slice == 0:
        lse1 = tl.where(l1 > 0.0, m1 + tl.log(l1), -float("inf"))
        lse2 = tl.where(l2 > 0.0, m2 + tl.log(l2), -float("inf"))
        tl.store(
            LSE1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t,
            lse1,
            mask=mask_m,
        )
        tl.store(
            LSE2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t,
            lse2,
            mask=mask_m,
        )

    o_ptrs = O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_out[None, :] * stride_od
    o2_ptrs = O2_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_out[None, :] * stride_od
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(o2_ptrs, o2.to(O2_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _flash_diffv2_bwd_prep_kernel(
    DO_ptr, O_ptr, O2_ptr, LAM_ptr,
    DELTA1_ptr, DELTA2_ptr, DLAM_ptr,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_lb, stride_lt, stride_lh,
    stride_db, stride_dh, stride_dt,
    T, H,
    BM: tl.constexpr,
    D: tl.constexpr,
    OMIT_DLAMBDA: tl.constexpr,
):
    """
    Computes exact softmax backward scalars Delta1, Delta2 and dlambda_logit in a single
    elementwise reduction pass over D:
      delta2 = <dO, O2>
      delta1 = <dO, O1> = <dO, O + lam * O2> = <dO, O> + lam * delta2
      Delta1 = delta1
      Delta2 = -lam * delta2
      dlambda_logit = -lam * (1 - lam) * delta2
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T

    base_o = b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    do = tl.load(DO_ptr + base_o, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o = tl.load(O_ptr + base_o, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o2 = tl.load(O2_ptr + base_o, mask=mask_m[:, None], other=0.0).to(tl.float32)

    lam_logit = tl.load(
        LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    lam = tl.sigmoid(lam_logit)

    dot_o = tl.sum(do * o, axis=1)
    dot_o2 = tl.sum(do * o2, axis=1)

    delta1 = dot_o + lam * dot_o2
    delta2 = -lam * dot_o2

    if OMIT_DLAMBDA:
        dlam = tl.zeros([BM], dtype=tl.float32)
    else:
        dlam = -lam * (1.0 - lam) * dot_o2

    tl.store(DELTA1_ptr + b * stride_db + h * stride_dh + offs_m * stride_dt, delta1, mask=mask_m)
    tl.store(DELTA2_ptr + b * stride_db + h * stride_dh + offs_m * stride_dt, delta2, mask=mask_m)
    tl.store(
        DLAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt,
        dlam.to(DLAM_ptr.dtype.element_ty),
        mask=mask_m,
    )


@triton.jit
def _flash_diffv2_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, LAM_ptr,
    LSE1_ptr, LSE2_ptr, DELTA1_ptr, DELTA2_ptr,
    DQ_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_lb, stride_lt, stride_lh,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Co-scheduled paired-query Flash-DiffAttn-V2 dQ kernel for D=64.
    Computes dQ_{2h} and dQ_{2h+1} simultaneously while loading shared K_{hkv}, V_{hkv}
    and computing dO @ V^T ONCE per tile.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    h_kv = h // N_REP

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T

    q1_ptrs = Q_ptr + b * stride_qb + (2 * h) * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q2_ptrs = Q_ptr + b * stride_qb + (2 * h + 1) * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q1 = tl.load(q1_ptrs, mask=mask_m[:, None], other=0.0)
    q2 = tl.load(q2_ptrs, mask=mask_m[:, None], other=0.0)

    do = tl.load(
        DO_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    )
    lam_logit = tl.load(
        LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    lam = tl.sigmoid(lam_logit)

    lse1 = tl.load(LSE1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
    lse2 = tl.load(LSE2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
    delta1 = tl.load(DELTA1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)
    delta2 = tl.load(DELTA2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)

    dq1 = tl.zeros([BM, D], dtype=tl.float32)
    dq2 = tl.zeros([BM, D], dtype=tl.float32)

    if IS_CAUSAL:
        max_q_global = q_offset + (pid_m + 1) * BM - 1
        max_local_k = max_q_global - k_offset + 1
        n_limit = tl.minimum(T, tl.maximum(0, max_local_k))
    else:
        n_limit = T

    k_base = K_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh

    for n_start in range(0, n_limit, BN):
        offs_n = n_start + tl.arange(0, BN)
        mask_n = offs_n < T

        k = tl.load(k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

        s1 = tl.dot(q1, tl.trans(k), allow_tf32=ALLOW_TF32) * sm_scale
        s2 = tl.dot(q2, tl.trans(k), allow_tf32=ALLOW_TF32) * sm_scale

        if IS_CAUSAL:
            q_glob = q_offset + offs_m[:, None]
            k_glob = k_offset + offs_n[None, :]
            valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        p1 = tl.where(valid & (lse1[:, None] != -float("inf")), tl.exp(s1 - lse1[:, None]), 0.0)
        p2 = tl.where(valid & (lse2[:, None] != -float("inf")), tl.exp(s2 - lse2[:, None]), 0.0)

        # Single shared dO @ V^T GEMM for both differential streams
        dov = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
        ds1 = p1 * (dov - delta1[:, None])
        ds2 = p2 * (-lam[:, None] * dov - delta2[:, None])

        dq1 += tl.dot(ds1.to(k.dtype), k, allow_tf32=ALLOW_TF32)
        dq2 += tl.dot(ds2.to(k.dtype), k, allow_tf32=ALLOW_TF32)

    dq1 = dq1 * sm_scale
    dq2 = dq2 * sm_scale

    dq1_ptrs = DQ_ptr + b * stride_qb + (2 * h) * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    dq2_ptrs = DQ_ptr + b * stride_qb + (2 * h + 1) * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    tl.store(dq1_ptrs, dq1.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq2_ptrs, dq2.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _flash_diffv2_bwd_dq_d128_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, LAM_ptr,
    LSE1_ptr, LSE2_ptr, DELTA1_ptr, DELTA2_ptr,
    DQ_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_lb, stride_lt, stride_lh,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Zero-spill Two-Slice (64+64) Co-Scheduled Paired-Query Flash-DiffAttn-V2 dQ kernel for D=128.
    pid_slice in {0, 1} accumulates dQ1 and dQ2 over the 64-wide slice (0 cubin spills).
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_slice = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H
    h_kv = h // N_REP

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d0 = tl.arange(0, 64)
    offs_d1 = 64 + tl.arange(0, 64)
    offs_out = pid_slice * 64 + tl.arange(0, 64)
    mask_m = offs_m < T

    q1_base = Q_ptr + b * stride_qb + (2 * h) * stride_qh + offs_m[:, None] * stride_qt
    q2_base = Q_ptr + b * stride_qb + (2 * h + 1) * stride_qh + offs_m[:, None] * stride_qt
    q1_0 = tl.load(q1_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q1_1 = tl.load(q1_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q2_0 = tl.load(q2_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q2_1 = tl.load(q2_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

    do_base = DO_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot
    do_0 = tl.load(do_base + offs_d0[None, :] * stride_od, mask=mask_m[:, None], other=0.0)
    do_1 = tl.load(do_base + offs_d1[None, :] * stride_od, mask=mask_m[:, None], other=0.0)

    lam_logit = tl.load(LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0).to(tl.float32)
    lam = tl.sigmoid(lam_logit)

    lse1 = tl.load(LSE1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
    lse2 = tl.load(LSE2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
    delta1 = tl.load(DELTA1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)
    delta2 = tl.load(DELTA2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)

    dq1 = tl.zeros([BM, 64], dtype=tl.float32)
    dq2 = tl.zeros([BM, 64], dtype=tl.float32)

    if IS_CAUSAL:
        max_q_global = q_offset + (pid_m + 1) * BM - 1
        max_local_k = max_q_global - k_offset + 1
        n_limit = tl.minimum(T, tl.maximum(0, max_local_k))
    else:
        n_limit = T

    k_base = K_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh

    for n_start in range(0, n_limit, BN):
        offs_n = n_start + tl.arange(0, BN)
        mask_n = offs_n < T

        k0 = tl.load(k_base + offs_n[:, None] * stride_kt + offs_d0[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
        k1 = tl.load(k_base + offs_n[:, None] * stride_kt + offs_d1[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
        s1 = (tl.dot(q1_0, tl.trans(k0), allow_tf32=ALLOW_TF32) + tl.dot(q1_1, tl.trans(k1), allow_tf32=ALLOW_TF32)) * sm_scale
        s2 = (tl.dot(q2_0, tl.trans(k0), allow_tf32=ALLOW_TF32) + tl.dot(q2_1, tl.trans(k1), allow_tf32=ALLOW_TF32)) * sm_scale

        if IS_CAUSAL:
            q_glob = q_offset + offs_m[:, None]
            k_glob = k_offset + offs_n[None, :]
            valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        p1 = tl.where(valid & (lse1[:, None] != -float("inf")), tl.exp(s1 - lse1[:, None]), 0.0)
        p2 = tl.where(valid & (lse2[:, None] != -float("inf")), tl.exp(s2 - lse2[:, None]), 0.0)

        v0 = tl.load(v_base + offs_n[:, None] * stride_vt + offs_d0[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
        v1 = tl.load(v_base + offs_n[:, None] * stride_vt + offs_d1[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
        dov = tl.dot(do_0, tl.trans(v0), allow_tf32=ALLOW_TF32) + tl.dot(do_1, tl.trans(v1), allow_tf32=ALLOW_TF32)

        ds1 = p1 * (dov - delta1[:, None])
        ds2 = p2 * (-lam[:, None] * dov - delta2[:, None])

        k_out = k0 if pid_slice == 0 else k1
        dq1 += tl.dot(ds1.to(k_out.dtype), k_out, allow_tf32=ALLOW_TF32)
        dq2 += tl.dot(ds2.to(k_out.dtype), k_out, allow_tf32=ALLOW_TF32)

    dq1 = dq1 * sm_scale
    dq2 = dq2 * sm_scale

    dq1_ptrs = DQ_ptr + b * stride_qb + (2 * h) * stride_qh + offs_m[:, None] * stride_qt + offs_out[None, :] * stride_qd
    dq2_ptrs = DQ_ptr + b * stride_qb + (2 * h + 1) * stride_qh + offs_m[:, None] * stride_qt + offs_out[None, :] * stride_qd
    tl.store(dq1_ptrs, dq1.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq2_ptrs, dq2.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _flash_diffv2_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, LAM_ptr,
    LSE1_ptr, LSE2_ptr, DELTA1_ptr, DELTA2_ptr,
    DK_ptr, DV_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_lb, stride_lt, stride_lh,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    GQA-Native Key-Parallel Flash-DiffAttn-V2 dKV kernel for D=64 (0 atomics, 0 spills).
    Accumulates dV via a single fused (P1 - lam * P2)^T @ dO GEMM and dO @ V^T GEMM per tile.
    """
    pid_n = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    b = pid_bhkv // H_KV
    h_kv = pid_bhkv % H_KV

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < T

    k = tl.load(
        K_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        V_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
    )

    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)

    if IS_CAUSAL:
        min_k_global = k_offset + pid_n * BN
        min_local_q = min_k_global - q_offset
        m_start_init = (tl.maximum(0, min_local_q) // BM) * BM
    else:
        m_start_init = 0

    for r in range(0, N_REP):
        h = h_kv * N_REP + r
        q1_base = Q_ptr + b * stride_qb + (2 * h) * stride_qh
        q2_base = Q_ptr + b * stride_qb + (2 * h + 1) * stride_qh
        do_base = DO_ptr + b * stride_ob + h * stride_oh
        lam_base = LAM_ptr + b * stride_lb + h * stride_lh
        lse1_base = LSE1_ptr + b * stride_lse_b + h * stride_lse_h
        lse2_base = LSE2_ptr + b * stride_lse_b + h * stride_lse_h
        d1_base = DELTA1_ptr + b * stride_lse_b + h * stride_lse_h
        d2_base = DELTA2_ptr + b * stride_lse_b + h * stride_lse_h

        for m_start in range(m_start_init, T, BM):
            offs_m = m_start + tl.arange(0, BM)
            mask_m = offs_m < T

            q1 = tl.load(q1_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            q2 = tl.load(q2_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, mask=mask_m[:, None], other=0.0)

            lam_logit = tl.load(lam_base + offs_m * stride_lt, mask=mask_m, other=0.0).to(tl.float32)
            lam = tl.sigmoid(lam_logit)
            lse1 = tl.load(lse1_base + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
            lse2 = tl.load(lse2_base + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
            delta1 = tl.load(d1_base + offs_m * stride_lse_t, mask=mask_m, other=0.0)
            delta2 = tl.load(d2_base + offs_m * stride_lse_t, mask=mask_m, other=0.0)

            s1 = tl.dot(q1, tl.trans(k), allow_tf32=ALLOW_TF32) * sm_scale
            s2 = tl.dot(q2, tl.trans(k), allow_tf32=ALLOW_TF32) * sm_scale

            if IS_CAUSAL:
                q_glob = q_offset + offs_m[:, None]
                k_glob = k_offset + offs_n[None, :]
                valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
            else:
                valid = mask_m[:, None] & mask_n[None, :]

            p1 = tl.where(valid & (lse1[:, None] != -float("inf")), tl.exp(s1 - lse1[:, None]), 0.0)
            p2 = tl.where(valid & (lse2[:, None] != -float("inf")), tl.exp(s2 - lse2[:, None]), 0.0)

            p_diff = p1 - lam[:, None] * p2
            dv += tl.dot(tl.trans(p_diff.to(do.dtype)), do, allow_tf32=ALLOW_TF32)

            dov = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
            ds1 = p1 * (dov - delta1[:, None])
            ds2 = p2 * (-lam[:, None] * dov - delta2[:, None])

            dk += tl.dot(tl.trans(ds1.to(q1.dtype)), q1, allow_tf32=ALLOW_TF32)
            dk += tl.dot(tl.trans(ds2.to(q2.dtype)), q2, allow_tf32=ALLOW_TF32)

    dk = dk * sm_scale
    tl.store(
        DK_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
        dk.to(DK_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )
    tl.store(
        DV_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
        dv.to(DV_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )


@triton.jit
def _flash_diffv2_bwd_dkv_d128_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, LAM_ptr,
    LSE1_ptr, LSE2_ptr, DELTA1_ptr, DELTA2_ptr,
    DK_ptr, DV_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_lb, stride_lt, stride_lh,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Zero-spill Two-Slice (64+64) GQA-Native Key-Parallel Flash-DiffAttn-V2 dKV kernel for D=128.
    pid_slice in {0, 1} accumulates dK and dV over the 64-wide slice (0 cubin spills).
    """
    pid_n = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    pid_slice = tl.program_id(2)
    b = pid_bhkv // H_KV
    h_kv = pid_bhkv % H_KV

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d0 = tl.arange(0, 64)
    offs_d1 = 64 + tl.arange(0, 64)
    offs_out = pid_slice * 64 + tl.arange(0, 64)
    mask_n = offs_n < T

    k_base = K_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vt
    k0 = tl.load(k_base + offs_d0[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
    k1 = tl.load(k_base + offs_d1[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
    v0 = tl.load(v_base + offs_d0[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
    v1 = tl.load(v_base + offs_d1[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BN, 64], dtype=tl.float32)
    dv = tl.zeros([BN, 64], dtype=tl.float32)

    if IS_CAUSAL:
        min_k_global = k_offset + pid_n * BN
        min_local_q = min_k_global - q_offset
        m_start_init = (tl.maximum(0, min_local_q) // BM) * BM
    else:
        m_start_init = 0

    num_m = tl.maximum(0, tl.cdiv(T - m_start_init, BM))
    total_steps = N_REP * num_m

    for step in range(0, total_steps):
        r = step // num_m
        m_idx = step % num_m
        h = h_kv * N_REP + r
        m_start = m_start_init + m_idx * BM
        offs_m = m_start + tl.arange(0, BM)
        mask_m = offs_m < T

        q1_base = Q_ptr + b * stride_qb + (2 * h) * stride_qh + offs_m[:, None] * stride_qt
        q2_base = Q_ptr + b * stride_qb + (2 * h + 1) * stride_qh + offs_m[:, None] * stride_qt
        do_base = DO_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot

        q1_0 = tl.load(q1_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        q1_1 = tl.load(q1_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        q2_0 = tl.load(q2_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        q2_1 = tl.load(q2_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

        s1 = (tl.dot(q1_0, tl.trans(k0), allow_tf32=ALLOW_TF32) + tl.dot(q1_1, tl.trans(k1), allow_tf32=ALLOW_TF32)) * sm_scale
        s2 = (tl.dot(q2_0, tl.trans(k0), allow_tf32=ALLOW_TF32) + tl.dot(q2_1, tl.trans(k1), allow_tf32=ALLOW_TF32)) * sm_scale

        if IS_CAUSAL:
            q_glob = q_offset + offs_m[:, None]
            k_glob = k_offset + offs_n[None, :]
            valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        lse1 = tl.load(LSE1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
        lse2 = tl.load(LSE2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
        p1 = tl.where(valid & (lse1[:, None] != -float("inf")), tl.exp(s1 - lse1[:, None]), 0.0)
        p2 = tl.where(valid & (lse2[:, None] != -float("inf")), tl.exp(s2 - lse2[:, None]), 0.0)

        do_0 = tl.load(do_base + offs_d0[None, :] * stride_od, mask=mask_m[:, None], other=0.0)
        do_1 = tl.load(do_base + offs_d1[None, :] * stride_od, mask=mask_m[:, None], other=0.0)
        dov = tl.dot(do_0, tl.trans(v0), allow_tf32=ALLOW_TF32) + tl.dot(do_1, tl.trans(v1), allow_tf32=ALLOW_TF32)

        lam_logit = tl.load(LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0).to(tl.float32)
        lam = tl.sigmoid(lam_logit)
        delta1 = tl.load(DELTA1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)
        delta2 = tl.load(DELTA2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)

        p_diff = p1 - lam[:, None] * p2
        do_out = do_0 if pid_slice == 0 else do_1
        dv += tl.dot(tl.trans(p_diff.to(do_out.dtype)), do_out, allow_tf32=ALLOW_TF32)

        ds1 = p1 * (dov - delta1[:, None])
        ds2 = p2 * (-lam[:, None] * dov - delta2[:, None])
        q1_out = q1_0 if pid_slice == 0 else q1_1
        q2_out = q2_0 if pid_slice == 0 else q2_1
        dk += tl.dot(tl.trans(ds1.to(q1_out.dtype)), q1_out, allow_tf32=ALLOW_TF32)
        dk += tl.dot(tl.trans(ds2.to(q2_out.dtype)), q2_out, allow_tf32=ALLOW_TF32)

    dk = dk * sm_scale
    tl.store(
        DK_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_out[None, :] * stride_kd,
        dk.to(DK_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )
    tl.store(
        DV_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vt + offs_out[None, :] * stride_vd,
        dv.to(DV_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )


# =============================================================================
# Negative-Control Monolithic D=128 Spill Kernel (Demonstrates SM80 Spill Trap)
# =============================================================================

@triton.jit
def _flash_diffv2_bwd_monolithic_spill_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, LAM_ptr,
    LSE1_ptr, LSE2_ptr, DELTA1_ptr, DELTA2_ptr,
    DK_ptr, DV_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_lb, stride_lt, stride_lh,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    """
    Negative-control monolithic [BM=64, BN=64, D=128] paired-stream dKV kernel without
    64+64 slice staging. Holding k[64,128], v[64,128], dk[64,128], dv[64,128], q1[64,128],
    q2[64,128], do[64,128] simultaneously in registers overflows SM80's 255-register ceiling.
    """
    pid_n = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    b = pid_bhkv // H_KV
    h_kv = pid_bhkv % H_KV

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < T

    k = tl.load(K_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
    v = tl.load(V_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)
    dk_aux = tl.zeros([BN, D], dtype=tl.float32)
    dv_aux = tl.zeros([BN, D], dtype=tl.float32)
    m_start_init = (pid_n * BN // BM) * BM

    for r in range(0, N_REP):
        h = h_kv * N_REP + r
        for m_start in range(m_start_init, T, BM):
            offs_m = m_start + tl.arange(0, BM)
            mask_m = offs_m < T
            q1 = tl.load(Q_ptr + b * stride_qb + (2 * h) * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            q2 = tl.load(Q_ptr + b * stride_qb + (2 * h + 1) * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            do = tl.load(DO_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, mask=mask_m[:, None], other=0.0)
            lam_logit = tl.load(LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0).to(tl.float32)
            lam = tl.sigmoid(lam_logit)
            lse1 = tl.load(LSE1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
            lse2 = tl.load(LSE2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
            delta1 = tl.load(DELTA1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)
            delta2 = tl.load(DELTA2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)

            s1 = tl.dot(q1, tl.trans(k), allow_tf32=False) * sm_scale
            s2 = tl.dot(q2, tl.trans(k), allow_tf32=False) * sm_scale
            valid = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            p1 = tl.where(valid & (lse1[:, None] != -float("inf")), tl.exp(s1 - lse1[:, None]), 0.0)
            p2 = tl.where(valid & (lse2[:, None] != -float("inf")), tl.exp(s2 - lse2[:, None]), 0.0)
            dov = tl.dot(do, tl.trans(v), allow_tf32=False)
            ds1 = p1 * (dov - delta1[:, None])
            ds2 = p2 * (-lam[:, None] * dov - delta2[:, None])
            dv += tl.dot(tl.trans(p1.to(do.dtype)), do, allow_tf32=False)
            dv_aux += tl.dot(tl.trans((lam[:, None] * p2).to(do.dtype)), do, allow_tf32=False)
            dk += tl.dot(tl.trans(ds1.to(q1.dtype)), q1, allow_tf32=False)
            dk_aux += tl.dot(tl.trans(ds2.to(q2.dtype)), q2, allow_tf32=False)

    dk = (dk + dk_aux) * sm_scale
    dv = dv - dv_aux
    tl.store(DK_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, dk.to(DK_ptr.dtype.element_ty), mask=mask_n[:, None])
    tl.store(DV_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd, dv.to(DV_ptr.dtype.element_ty), mask=mask_n[:, None])


# =============================================================================
# 4. Paired-Stream Flash-DiffAttn-V1 + Fused SubLN Epilogue Kernels (0 Spills)
# =============================================================================

@triton.jit
def _flash_diffv1_fwd_slice_kernel(
    Q1_ptr, Q2_ptr, K1_ptr, K2_ptr, V_ptr, LAM_FULL_ptr,
    U_ptr, O2_ptr, LSE1_ptr, LSE2_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ub, stride_ut, stride_uh, stride_ud,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    HAS_PER_HEAD_LAM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Paired-Stream Flash-DiffAttn-V1 forward kernel.
    Q1, Q2, K1, K2 have head_dim = D (64); shared V has head_dim = 2*D (128).
    pid_slice in {0, 1} processes V[..., 0:D] or V[..., D:2D] in a single online-softmax pass
    so Q1@K1^T and Q2@K2^T are not split into 4 separate flash_attn_func calls + torch.cat.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_slice = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H
    h_kv = h // N_REP

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_vd = pid_slice * D + tl.arange(0, D)
    mask_m = offs_m < T

    q1 = tl.load(Q1_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q2 = tl.load(Q2_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

    if HAS_PER_HEAD_LAM:
        lam_full = tl.load(LAM_FULL_ptr + h).to(tl.float32)
    else:
        lam_full = tl.load(LAM_FULL_ptr).to(tl.float32)

    m1 = tl.full([BM], -float("inf"), dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    acc1 = tl.zeros([BM, D], dtype=tl.float32)

    m2 = tl.full([BM], -float("inf"), dtype=tl.float32)
    l2 = tl.zeros([BM], dtype=tl.float32)
    acc2 = tl.zeros([BM, D], dtype=tl.float32)

    if IS_CAUSAL:
        max_q_global = q_offset + (pid_m + 1) * BM - 1
        max_local_k = max_q_global - k_offset + 1
        n_limit = tl.minimum(T, tl.maximum(0, max_local_k))
    else:
        n_limit = T

    k1_base = K1_ptr + b * stride_kb + h_kv * stride_kh
    k2_base = K2_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh

    for n_start in range(0, n_limit, BN):
        offs_n = n_start + tl.arange(0, BN)
        mask_n = offs_n < T

        k1 = tl.load(k1_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
        k2 = tl.load(k2_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)

        s1 = tl.dot(q1, tl.trans(k1), allow_tf32=ALLOW_TF32) * sm_scale
        s2 = tl.dot(q2, tl.trans(k2), allow_tf32=ALLOW_TF32) * sm_scale

        if IS_CAUSAL:
            q_glob = q_offset + offs_m[:, None]
            k_glob = k_offset + offs_n[None, :]
            valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        s1 = tl.where(valid, s1, -float("inf"))
        s2 = tl.where(valid, s2, -float("inf"))

        m1_ij = tl.max(s1, axis=1)
        m1_new = tl.maximum(m1, m1_ij)
        m1_safe = tl.where(m1_new == -float("inf"), 0.0, m1_new)
        alpha1 = tl.exp(tl.where(m1 == -float("inf"), -float("inf"), m1 - m1_safe))
        p1 = tl.exp(tl.where(s1 == -float("inf"), -float("inf"), s1 - m1_safe[:, None]))
        l1 = l1 * alpha1 + tl.sum(p1, axis=1)

        m2_ij = tl.max(s2, axis=1)
        m2_new = tl.maximum(m2, m2_ij)
        m2_safe = tl.where(m2_new == -float("inf"), 0.0, m2_new)
        alpha2 = tl.exp(tl.where(m2 == -float("inf"), -float("inf"), m2 - m2_safe))
        p2 = tl.exp(tl.where(s2 == -float("inf"), -float("inf"), s2 - m2_safe[:, None]))
        l2 = l2 * alpha2 + tl.sum(p2, axis=1)

        v_slice = tl.load(v_base + offs_n[:, None] * stride_vt + offs_vd[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
        acc1 = acc1 * alpha1[:, None] + tl.dot(p1.to(v_slice.dtype), v_slice, allow_tf32=ALLOW_TF32)
        acc2 = acc2 * alpha2[:, None] + tl.dot(p2.to(v_slice.dtype), v_slice, allow_tf32=ALLOW_TF32)
        m1 = m1_new
        m2 = m2_new

    inv_l1 = tl.where(l1 > 0.0, 1.0 / l1, 0.0)
    inv_l2 = tl.where(l2 > 0.0, 1.0 / l2, 0.0)
    o1 = acc1 * inv_l1[:, None]
    o2 = acc2 * inv_l2[:, None]
    u = o1 - lam_full * o2

    if pid_slice == 0:
        lse1 = tl.where(l1 > 0.0, m1 + tl.log(l1), -float("inf"))
        lse2 = tl.where(l2 > 0.0, m2 + tl.log(l2), -float("inf"))
        tl.store(LSE1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, lse1, mask=mask_m)
        tl.store(LSE2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, lse2, mask=mask_m)

    u_ptrs = U_ptr + b * stride_ub + h * stride_uh + offs_m[:, None] * stride_ut + offs_vd[None, :] * stride_ud
    o2_ptrs = O2_ptr + b * stride_ub + h * stride_uh + offs_m[:, None] * stride_ut + offs_vd[None, :] * stride_ud
    tl.store(u_ptrs, u.to(U_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(o2_ptrs, o2.to(O2_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _diffv1_subln_scale_fwd_kernel(
    U_ptr, GAMMA_ptr, Y_ptr, RSTD_ptr,
    stride_ub, stride_ut, stride_uh, stride_ud,
    stride_rb, stride_rh, stride_rt,
    one_minus_lambda_init, eps,
    T, H,
    BM: tl.constexpr,
    D2: tl.constexpr,
):
    """
    Fused per-head SubLN RMSNorm(2d) + (1 - lambda_init) post-scale forward kernel.
    Replaces 3 separate HBM passes in upstream multihead_flashdiff_1.py + rms_norm.py.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D2)
    mask_m = offs_m < T

    u_ptrs = U_ptr + b * stride_ub + h * stride_uh + offs_m[:, None] * stride_ut + offs_d[None, :] * stride_ud
    u = tl.load(u_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    gamma = tl.load(GAMMA_ptr + offs_d).to(tl.float32)

    var = tl.sum(u * u, axis=1) / D2
    rstd = tl.rsqrt(var + eps)
    y = (u * rstd[:, None]) * (gamma[None, :] * one_minus_lambda_init)

    tl.store(RSTD_ptr + b * stride_rb + h * stride_rh + offs_m * stride_rt, rstd, mask=mask_m)
    y_ptrs = Y_ptr + b * stride_ub + h * stride_uh + offs_m[:, None] * stride_ut + offs_d[None, :] * stride_ud
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _diffv1_subln_scale_bwd_kernel(
    DY_ptr, U_ptr, O2_ptr, GAMMA_ptr, RSTD_ptr, LAM_FULL_ptr,
    DU_ptr, DGAMMA_PART_ptr, DELTA1_ptr, DELTA2_ptr, DLAM_PART_ptr,
    stride_ub, stride_ut, stride_uh, stride_ud,
    stride_rb, stride_rh, stride_rt,
    stride_dp_m, stride_dp_bh, stride_dp_d,
    one_minus_lambda_init,
    T, H,
    HAS_PER_HEAD_LAM: tl.constexpr,
    OMIT_RADIAL_JACOBIAN: tl.constexpr,
    BM: tl.constexpr,
    D2: tl.constexpr,
):
    """
    Fused SubLN radial-Jacobian backward + Delta1, Delta2, dgamma_part, dlambda_full_part.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D2)
    mask_m = offs_m < T

    base_u = b * stride_ub + h * stride_uh + offs_m[:, None] * stride_ut + offs_d[None, :] * stride_ud
    dy = tl.load(DY_ptr + base_u, mask=mask_m[:, None], other=0.0).to(tl.float32)
    u = tl.load(U_ptr + base_u, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o2 = tl.load(O2_ptr + base_u, mask=mask_m[:, None], other=0.0).to(tl.float32)
    gamma = tl.load(GAMMA_ptr + offs_d).to(tl.float32)
    rstd = tl.load(RSTD_ptr + b * stride_rb + h * stride_rh + offs_m * stride_rt, mask=mask_m, other=0.0)

    if HAS_PER_HEAD_LAM:
        lam_full = tl.load(LAM_FULL_ptr + h).to(tl.float32)
    else:
        lam_full = tl.load(LAM_FULL_ptr).to(tl.float32)

    u_hat = u * rstd[:, None]
    du_hat = dy * (gamma[None, :] * one_minus_lambda_init)

    if OMIT_RADIAL_JACOBIAN:
        du = rstd[:, None] * du_hat
    else:
        mean_dot = tl.sum(du_hat * u_hat, axis=1) / D2
        du = rstd[:, None] * (du_hat - u_hat * mean_dot[:, None])

    dgamma_tile = tl.sum(dy * u_hat, axis=0) * one_minus_lambda_init
    tl.store(DGAMMA_PART_ptr + pid_m * stride_dp_m + pid_bh * stride_dp_bh + offs_d * stride_dp_d, dgamma_tile)

    dot_u = tl.sum(du * u, axis=1)
    dot_o2 = tl.sum(du * o2, axis=1)

    delta1 = dot_u + lam_full * dot_o2
    delta2 = -lam_full * dot_o2
    dlam_tile = -tl.sum(dot_o2, axis=0)

    tl.store(DU_ptr + base_u, du.to(DU_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(DELTA1_ptr + b * stride_rb + h * stride_rh + offs_m * stride_rt, delta1, mask=mask_m)
    tl.store(DELTA2_ptr + b * stride_rb + h * stride_rh + offs_m * stride_rt, delta2, mask=mask_m)
    tl.store(DLAM_PART_ptr + pid_m * tl.num_programs(1) + pid_bh, dlam_tile)


@triton.jit
def _flash_diffv1_bwd_dq_kernel(
    Q1_ptr, Q2_ptr, K1_ptr, K2_ptr, V_ptr, DU_ptr, LAM_FULL_ptr,
    LSE1_ptr, LSE2_ptr, DELTA1_ptr, DELTA2_ptr,
    DQ1_ptr, DQ2_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ub, stride_ut, stride_uh, stride_ud,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    HAS_PER_HEAD_LAM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Paired-Stream Flash-DiffAttn-V1 dQ1/dQ2 kernel with two-slice (0:D, D:2D) V dot product.
    0 cubin register spills on SM80.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    h_kv = h // N_REP

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_d1 = D + tl.arange(0, D)
    mask_m = offs_m < T

    q1 = tl.load(Q1_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q2 = tl.load(Q2_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

    du_base = DU_ptr + b * stride_ub + h * stride_uh + offs_m[:, None] * stride_ut
    du0 = tl.load(du_base + offs_d[None, :] * stride_ud, mask=mask_m[:, None], other=0.0)
    du1 = tl.load(du_base + offs_d1[None, :] * stride_ud, mask=mask_m[:, None], other=0.0)

    if HAS_PER_HEAD_LAM:
        lam_full = tl.load(LAM_FULL_ptr + h).to(tl.float32)
    else:
        lam_full = tl.load(LAM_FULL_ptr).to(tl.float32)

    lse1 = tl.load(LSE1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
    lse2 = tl.load(LSE2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
    delta1 = tl.load(DELTA1_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)
    delta2 = tl.load(DELTA2_ptr + b * stride_lse_b + h * stride_lse_h + offs_m * stride_lse_t, mask=mask_m, other=0.0)

    dq1 = tl.zeros([BM, D], dtype=tl.float32)
    dq2 = tl.zeros([BM, D], dtype=tl.float32)

    if IS_CAUSAL:
        max_q_global = q_offset + (pid_m + 1) * BM - 1
        max_local_k = max_q_global - k_offset + 1
        n_limit = tl.minimum(T, tl.maximum(0, max_local_k))
    else:
        n_limit = T

    k1_base = K1_ptr + b * stride_kb + h_kv * stride_kh
    k2_base = K2_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh

    for n_start in range(0, n_limit, BN):
        offs_n = n_start + tl.arange(0, BN)
        mask_n = offs_n < T

        k1 = tl.load(k1_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
        k2 = tl.load(k2_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)

        s1 = tl.dot(q1, tl.trans(k1), allow_tf32=ALLOW_TF32) * sm_scale
        s2 = tl.dot(q2, tl.trans(k2), allow_tf32=ALLOW_TF32) * sm_scale

        if IS_CAUSAL:
            q_glob = q_offset + offs_m[:, None]
            k_glob = k_offset + offs_n[None, :]
            valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        p1 = tl.where(valid & (lse1[:, None] != -float("inf")), tl.exp(s1 - lse1[:, None]), 0.0)
        p2 = tl.where(valid & (lse2[:, None] != -float("inf")), tl.exp(s2 - lse2[:, None]), 0.0)

        v0 = tl.load(v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
        v1 = tl.load(v_base + offs_n[:, None] * stride_vt + offs_d1[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
        duv = tl.dot(du0, tl.trans(v0), allow_tf32=ALLOW_TF32) + tl.dot(du1, tl.trans(v1), allow_tf32=ALLOW_TF32)

        ds1 = p1 * (duv - delta1[:, None])
        ds2 = p2 * (-lam_full * duv - delta2[:, None])

        dq1 += tl.dot(ds1.to(k1.dtype), k1, allow_tf32=ALLOW_TF32)
        dq2 += tl.dot(ds2.to(k2.dtype), k2, allow_tf32=ALLOW_TF32)

    dq1 = dq1 * sm_scale
    dq2 = dq2 * sm_scale

    tl.store(DQ1_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, dq1.to(DQ1_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(DQ2_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, dq2.to(DQ2_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _flash_diffv1_bwd_dkv_kernel(
    Q1_ptr, Q2_ptr, K1_ptr, K2_ptr, V_ptr, DU_ptr, LAM_FULL_ptr,
    LSE1_ptr, LSE2_ptr, DELTA1_ptr, DELTA2_ptr,
    DK1_ptr, DK2_ptr, DV_ptr,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ub, stride_ut, stride_uh, stride_ud,
    stride_lse_b, stride_lse_h, stride_lse_t,
    sm_scale,
    T, H, H_KV, N_REP,
    q_offset, k_offset,
    HAS_PER_HEAD_LAM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Zero-spill Two-Slice (pid_slice in {0, 1}) GQA-Native Key-Parallel Flash-DiffAttn-V1 dKV kernel.
    Slice 0 computes dK1 (width D) and dV[..., 0:D]; Slice 1 computes dK2 (width D) and dV[..., D:2D],
    so each CTA holds only TWO [BN, D] FP32 accumulators (dk_sel and dv_slice) -> 0 cubin spills!
    """
    pid_n = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    pid_slice = tl.program_id(2)
    b = pid_bhkv // H_KV
    h_kv = pid_bhkv % H_KV

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    offs_d1 = D + tl.arange(0, D)
    offs_vd = pid_slice * D + tl.arange(0, D)
    mask_n = offs_n < T

    k1 = tl.load(K1_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
    k2 = tl.load(K2_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh
    v0 = tl.load(v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
    v1 = tl.load(v_base + offs_n[:, None] * stride_vt + offs_d1[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

    dk_sel = tl.zeros([BN, D], dtype=tl.float32)
    dv_slice = tl.zeros([BN, D], dtype=tl.float32)

    if IS_CAUSAL:
        min_k_global = k_offset + pid_n * BN
        min_local_q = min_k_global - q_offset
        m_start_init = (tl.maximum(0, min_local_q) // BM) * BM
    else:
        m_start_init = 0

    for r in range(0, N_REP):
        h = h_kv * N_REP + r
        if HAS_PER_HEAD_LAM:
            lam_full = tl.load(LAM_FULL_ptr + h).to(tl.float32)
        else:
            lam_full = tl.load(LAM_FULL_ptr).to(tl.float32)

        q1_base = Q1_ptr + b * stride_qb + h * stride_qh
        q2_base = Q2_ptr + b * stride_qb + h * stride_qh
        du_base = DU_ptr + b * stride_ub + h * stride_uh
        lse1_base = LSE1_ptr + b * stride_lse_b + h * stride_lse_h
        lse2_base = LSE2_ptr + b * stride_lse_b + h * stride_lse_h
        d1_base = DELTA1_ptr + b * stride_lse_b + h * stride_lse_h
        d2_base = DELTA2_ptr + b * stride_lse_b + h * stride_lse_h

        for m_start in range(m_start_init, T, BM):
            offs_m = m_start + tl.arange(0, BM)
            mask_m = offs_m < T

            q1 = tl.load(q1_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            q2 = tl.load(q2_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

            s1 = tl.dot(q1, tl.trans(k1), allow_tf32=ALLOW_TF32) * sm_scale
            s2 = tl.dot(q2, tl.trans(k2), allow_tf32=ALLOW_TF32) * sm_scale

            if IS_CAUSAL:
                q_glob = q_offset + offs_m[:, None]
                k_glob = k_offset + offs_n[None, :]
                valid = (q_glob >= k_glob) & mask_m[:, None] & mask_n[None, :]
            else:
                valid = mask_m[:, None] & mask_n[None, :]

            lse1 = tl.load(lse1_base + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
            lse2 = tl.load(lse2_base + offs_m * stride_lse_t, mask=mask_m, other=-float("inf"))
            p1 = tl.where(valid & (lse1[:, None] != -float("inf")), tl.exp(s1 - lse1[:, None]), 0.0)
            p2 = tl.where(valid & (lse2[:, None] != -float("inf")), tl.exp(s2 - lse2[:, None]), 0.0)

            du0 = tl.load(du_base + offs_m[:, None] * stride_ut + offs_d[None, :] * stride_ud, mask=mask_m[:, None], other=0.0)
            du1 = tl.load(du_base + offs_m[:, None] * stride_ut + offs_d1[None, :] * stride_ud, mask=mask_m[:, None], other=0.0)
            duv = tl.dot(du0, tl.trans(v0), allow_tf32=ALLOW_TF32) + tl.dot(du1, tl.trans(v1), allow_tf32=ALLOW_TF32)

            delta1 = tl.load(d1_base + offs_m * stride_lse_t, mask=mask_m, other=0.0)
            delta2 = tl.load(d2_base + offs_m * stride_lse_t, mask=mask_m, other=0.0)

            p_diff = p1 - lam_full * p2
            du_out = du0 if pid_slice == 0 else du1
            dv_slice += tl.dot(tl.trans(p_diff.to(du_out.dtype)), du_out, allow_tf32=ALLOW_TF32)

            if pid_slice == 0:
                ds1 = p1 * (duv - delta1[:, None])
                dk_sel += tl.dot(tl.trans(ds1.to(q1.dtype)), q1, allow_tf32=ALLOW_TF32)
            else:
                ds2 = p2 * (-lam_full * duv - delta2[:, None])
                dk_sel += tl.dot(tl.trans(ds2.to(q2.dtype)), q2, allow_tf32=ALLOW_TF32)

    dk_sel = dk_sel * sm_scale
    if pid_slice == 0:
        tl.store(
            DK1_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            dk_sel.to(DK1_ptr.dtype.element_ty),
            mask=mask_n[:, None],
        )
    else:
        tl.store(
            DK2_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            dk_sel.to(DK2_ptr.dtype.element_ty),
            mask=mask_n[:, None],
        )
    tl.store(
        DV_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vt + offs_vd[None, :] * stride_vd,
        dv_slice.to(DV_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )


# =============================================================================
# 5. Autograd Functions & High-Level APIs for DiffAttn V2 and DiffAttn V1
# =============================================================================

class _FlashDiffAttnV2Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lam_logit: torch.Tensor,
        allow_tf32: bool = False,
        omit_dlambda: bool = False,
        use_monolithic_spill_bwd: bool = False,
    ) -> torch.Tensor:
        """
        q: [B, T, 2*H, D]
        k: [B, T, H_KV, D]
        v: [B, T, H_KV, D]
        lam_logit: [B, T, H]
        returns: O [B, T, H, D]
        """
        B, T, H2, D = q.shape
        H = H2 // 2
        H_KV = k.shape[2]
        assert H % H_KV == 0
        n_rep = H // H_KV
        sm_scale = 1.0 / math.sqrt(D)

        o = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
        o2 = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
        lse1 = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
        lse2 = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

        is_fp32 = q.dtype == torch.float32
        BM = 16 if is_fp32 else 32
        BN = 16 if is_fp32 else 32

        if D == 64:
            grid = (triton.cdiv(T, BM), B * H)
            _flash_diffv2_fwd_kernel[grid](
                q, k, v, lam_logit, o, o2, lse1, lse2,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                lse1.stride(0), lse1.stride(1), lse1.stride(2),
                sm_scale, T, H, H_KV, n_rep, 0, 0,
                IS_CAUSAL=True, BM=BM, BN=BN, D=64, ALLOW_TF32=allow_tf32,
                num_warps=4, num_stages=1 if is_fp32 else 2,
            )
        elif D == 128:
            grid = (triton.cdiv(T, BM), B * H, 2)
            _flash_diffv2_fwd_d128_kernel[grid](
                q, k, v, lam_logit, o, o2, lse1, lse2,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                lse1.stride(0), lse1.stride(1), lse1.stride(2),
                sm_scale, T, H, H_KV, n_rep, 0, 0,
                IS_CAUSAL=True, BM=BM, BN=BN, ALLOW_TF32=allow_tf32,
                num_warps=4, num_stages=1 if is_fp32 else 2,
            )
        else:
            raise ValueError(f"Unsupported head_dim D={D}; expected 64 or 128.")

        ctx.save_for_backward(q, k, v, lam_logit, o, o2, lse1, lse2)
        ctx.allow_tf32 = allow_tf32
        ctx.omit_dlambda = omit_dlambda
        ctx.use_monolithic_spill_bwd = use_monolithic_spill_bwd
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, lam_logit, o, o2, lse1, lse2 = ctx.saved_tensors
        B, T, H2, D = q.shape
        H = H2 // 2
        H_KV = k.shape[2]
        n_rep = H // H_KV
        sm_scale = 1.0 / math.sqrt(D)

        do = do.contiguous()
        delta1 = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
        delta2 = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
        dlam = torch.empty_like(lam_logit)

        BM_prep = 32
        grid_prep = (triton.cdiv(T, BM_prep), B * H)
        _flash_diffv2_bwd_prep_kernel[grid_prep](
            do, o, o2, lam_logit, delta1, delta2, dlam,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
            delta1.stride(0), delta1.stride(1), delta1.stride(2),
            T, H, BM=BM_prep, D=D, OMIT_DLAMBDA=ctx.omit_dlambda,
            num_warps=4,
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        is_fp32 = q.dtype == torch.float32
        BM = 16 if is_fp32 else 32
        BN = 16 if is_fp32 else 32

        if D == 64:
            grid_dq = (triton.cdiv(T, BM), B * H)
            _flash_diffv2_bwd_dq_kernel[grid_dq](
                q, k, v, do, lam_logit, lse1, lse2, delta1, delta2, dq,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
                lse1.stride(0), lse1.stride(1), lse1.stride(2),
                sm_scale, T, H, H_KV, n_rep, 0, 0,
                IS_CAUSAL=True, BM=BM, BN=BN, D=64, ALLOW_TF32=ctx.allow_tf32,
                num_warps=4, num_stages=1 if is_fp32 else 2,
            )
            grid_dkv = (triton.cdiv(T, BN), B * H_KV)
            _flash_diffv2_bwd_dkv_kernel[grid_dkv](
                q, k, v, do, lam_logit, lse1, lse2, delta1, delta2, dk, dv,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
                lse1.stride(0), lse1.stride(1), lse1.stride(2),
                sm_scale, T, H, H_KV, n_rep, 0, 0,
                IS_CAUSAL=True, BM=BM, BN=BN, D=64, ALLOW_TF32=ctx.allow_tf32,
                num_warps=4, num_stages=1 if is_fp32 else 2,
            )
        else:
            grid_dq = (triton.cdiv(T, BM), B * H, 2)
            _flash_diffv2_bwd_dq_d128_kernel[grid_dq](
                q, k, v, do, lam_logit, lse1, lse2, delta1, delta2, dq,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
                lse1.stride(0), lse1.stride(1), lse1.stride(2),
                sm_scale, T, H, H_KV, n_rep, 0, 0,
                IS_CAUSAL=True, BM=BM, BN=BN, ALLOW_TF32=ctx.allow_tf32,
                num_warps=4, num_stages=1 if is_fp32 else 2,
            )
            if ctx.use_monolithic_spill_bwd:
                grid_dkv = (triton.cdiv(T, 64), B * H_KV)
                _flash_diffv2_bwd_monolithic_spill_kernel[grid_dkv](
                    q, k, v, do, lam_logit, lse1, lse2, delta1, delta2, dk, dv,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
                    lse1.stride(0), lse1.stride(1), lse1.stride(2),
                    sm_scale, T, H, H_KV, n_rep,
                    BM=64, BN=64, D=128,
                    num_warps=4, num_stages=3,
                )
            else:
                grid_dkv = (triton.cdiv(T, BN), B * H_KV, 2)
                _flash_diffv2_bwd_dkv_d128_kernel[grid_dkv](
                    q, k, v, do, lam_logit, lse1, lse2, delta1, delta2, dk, dv,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
                    lse1.stride(0), lse1.stride(1), lse1.stride(2),
                    sm_scale, T, H, H_KV, n_rep, 0, 0,
                    IS_CAUSAL=True, BM=BM, BN=BN, ALLOW_TF32=ctx.allow_tf32,
                    num_warps=4, num_stages=1 if is_fp32 else 2,
                )
        return dq, dk, dv, dlam, None, None, None


def flash_diffattn_v2_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lam_logit: torch.Tensor,
    allow_tf32: bool = False,
    omit_dlambda: bool = False,
    use_monolithic_spill_bwd: bool = False,
) -> torch.Tensor:
    return _FlashDiffAttnV2Fn.apply(q, k, v, lam_logit, allow_tf32, omit_dlambda, use_monolithic_spill_bwd)


class _FlashDiffAttnV1FusedSubLNFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q1: torch.Tensor,
        q2: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        v: torch.Tensor,
        lambda_q1: torch.Tensor,
        lambda_k1: torch.Tensor,
        lambda_q2: torch.Tensor,
        lambda_k2: torch.Tensor,
        subln_weight: torch.Tensor,
        lambda_init: float,
        eps: float = 1e-5,
        allow_tf32: bool = False,
        omit_radial_jacobian: bool = False,
    ) -> torch.Tensor:
        """
        q1, q2: [B, T, H, D]
        k1, k2: [B, T, H_KV, D]
        v: [B, T, H_KV, 2*D]
        lambda_q1, lambda_k1, lambda_q2, lambda_k2: [D] or [H, D]
        subln_weight: [2*D]
        returns: Y [B, T, H, 2*D]
        """
        B, T, H, D = q1.shape
        H_KV = k1.shape[2]
        D2 = 2 * D
        assert v.shape[-1] == D2 and D == 64
        n_rep = H // H_KV
        sm_scale = 1.0 / math.sqrt(D)

        dot1 = torch.sum(lambda_q1.float() * lambda_k1.float(), dim=-1)
        dot2 = torch.sum(lambda_q2.float() * lambda_k2.float(), dim=-1)
        lam1 = torch.exp(dot1)
        lam2 = torch.exp(dot2)
        lam_full = (lam1 - lam2 + float(lambda_init)).contiguous()
        has_per_head_lam = lam_full.ndim == 1 and lam_full.numel() == H

        u = torch.empty((B, T, H, D2), device=q1.device, dtype=q1.dtype)
        o2 = torch.empty((B, T, H, D2), device=q1.device, dtype=q1.dtype)
        lse1 = torch.empty((B, H, T), device=q1.device, dtype=torch.float32)
        lse2 = torch.empty((B, H, T), device=q1.device, dtype=torch.float32)

        is_fp32 = q1.dtype == torch.float32
        BM = 16 if is_fp32 else 32
        BN = 16 if is_fp32 else 32

        grid_fwd = (triton.cdiv(T, BM), B * H, 2)
        _flash_diffv1_fwd_slice_kernel[grid_fwd](
            q1, q2, k1, k2, v, lam_full,
            u, o2, lse1, lse2,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            u.stride(0), u.stride(1), u.stride(2), u.stride(3),
            lse1.stride(0), lse1.stride(1), lse1.stride(2),
            sm_scale, T, H, H_KV, n_rep, 0, 0,
            HAS_PER_HEAD_LAM=has_per_head_lam,
            IS_CAUSAL=True, BM=BM, BN=BN, D=D, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )

        y = torch.empty_like(u)
        rstd = torch.empty((B, H, T), device=q1.device, dtype=torch.float32)
        BM_subln = 32
        grid_subln = (triton.cdiv(T, BM_subln), B * H)
        _diffv1_subln_scale_fwd_kernel[grid_subln](
            u, subln_weight, y, rstd,
            u.stride(0), u.stride(1), u.stride(2), u.stride(3),
            rstd.stride(0), rstd.stride(1), rstd.stride(2),
            float(1.0 - lambda_init), float(eps),
            T, H, BM=BM_subln, D2=D2,
            num_warps=4,
        )

        ctx.save_for_backward(
            q1, q2, k1, k2, v,
            lambda_q1, lambda_k1, lambda_q2, lambda_k2,
            subln_weight, lam_full, lam1, lam2,
            u, o2, lse1, lse2, rstd,
        )
        ctx.lambda_init = float(lambda_init)
        ctx.eps = float(eps)
        ctx.allow_tf32 = allow_tf32
        ctx.omit_radial_jacobian = omit_radial_jacobian
        ctx.has_per_head_lam = has_per_head_lam
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (
            q1, q2, k1, k2, v,
            lambda_q1, lambda_k1, lambda_q2, lambda_k2,
            subln_weight, lam_full, lam1, lam2,
            u, o2, lse1, lse2, rstd,
        ) = ctx.saved_tensors
        B, T, H, D = q1.shape
        H_KV = k1.shape[2]
        D2 = 2 * D
        n_rep = H // H_KV
        sm_scale = 1.0 / math.sqrt(D)

        dy = dy.contiguous()
        du = torch.empty_like(u)
        delta1 = torch.empty((B, H, T), device=q1.device, dtype=torch.float32)
        delta2 = torch.empty((B, H, T), device=q1.device, dtype=torch.float32)

        BM_subln = 32
        n_m_tiles = triton.cdiv(T, BM_subln)
        dgamma_part = torch.empty((n_m_tiles, B * H, D2), device=q1.device, dtype=torch.float32)
        dlam_part = torch.empty((n_m_tiles, B * H), device=q1.device, dtype=torch.float32)

        grid_subln = (n_m_tiles, B * H)
        _diffv1_subln_scale_bwd_kernel[grid_subln](
            dy, u, o2, subln_weight, rstd, lam_full,
            du, dgamma_part, delta1, delta2, dlam_part,
            u.stride(0), u.stride(1), u.stride(2), u.stride(3),
            rstd.stride(0), rstd.stride(1), rstd.stride(2),
            dgamma_part.stride(0), dgamma_part.stride(1), dgamma_part.stride(2),
            float(1.0 - ctx.lambda_init),
            T, H,
            HAS_PER_HEAD_LAM=ctx.has_per_head_lam,
            OMIT_RADIAL_JACOBIAN=ctx.omit_radial_jacobian,
            BM=BM_subln, D2=D2,
            num_warps=4,
        )

        dsubln_weight = dgamma_part.sum(dim=(0, 1)).to(subln_weight.dtype)
        dlam_per_bh = dlam_part.sum(dim=0).view(B, H)
        if ctx.has_per_head_lam:
            dlam_full = dlam_per_bh.sum(dim=0)  # [H]
            dlq1 = (dlam_full[:, None] * lam1[:, None] * lambda_k1.float()).to(lambda_q1.dtype)
            dlk1 = (dlam_full[:, None] * lam1[:, None] * lambda_q1.float()).to(lambda_k1.dtype)
            dlq2 = (-dlam_full[:, None] * lam2[:, None] * lambda_k2.float()).to(lambda_q2.dtype)
            dlk2 = (-dlam_full[:, None] * lam2[:, None] * lambda_q2.float()).to(lambda_k2.dtype)
        else:
            dlam_full = dlam_per_bh.sum()  # scalar
            dlq1 = (dlam_full * lam1 * lambda_k1.float()).to(lambda_q1.dtype)
            dlk1 = (dlam_full * lam1 * lambda_q1.float()).to(lambda_k1.dtype)
            dlq2 = (-dlam_full * lam2 * lambda_k2.float()).to(lambda_q2.dtype)
            dlk2 = (-dlam_full * lam2 * lambda_q2.float()).to(lambda_k2.dtype)

        dq1 = torch.empty_like(q1)
        dq2 = torch.empty_like(q2)
        dk1 = torch.empty_like(k1)
        dk2 = torch.empty_like(k2)
        dv = torch.empty_like(v)

        is_fp32 = q1.dtype == torch.float32
        BM = 16 if is_fp32 else 32
        BN = 16 if is_fp32 else 32

        grid_dq = (triton.cdiv(T, BM), B * H)
        _flash_diffv1_bwd_dq_kernel[grid_dq](
            q1, q2, k1, k2, v, du, lam_full, lse1, lse2, delta1, delta2, dq1, dq2,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            du.stride(0), du.stride(1), du.stride(2), du.stride(3),
            lse1.stride(0), lse1.stride(1), lse1.stride(2),
            sm_scale, T, H, H_KV, n_rep, 0, 0,
            HAS_PER_HEAD_LAM=ctx.has_per_head_lam,
            IS_CAUSAL=True, BM=BM, BN=BN, D=D, ALLOW_TF32=ctx.allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )

        grid_dkv = (triton.cdiv(T, BN), B * H_KV, 2)
        _flash_diffv1_bwd_dkv_kernel[grid_dkv](
            q1, q2, k1, k2, v, du, lam_full, lse1, lse2, delta1, delta2, dk1, dk2, dv,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            du.stride(0), du.stride(1), du.stride(2), du.stride(3),
            lse1.stride(0), lse1.stride(1), lse1.stride(2),
            sm_scale, T, H, H_KV, n_rep, 0, 0,
            HAS_PER_HEAD_LAM=ctx.has_per_head_lam,
            IS_CAUSAL=True, BM=BM, BN=BN, D=D, ALLOW_TF32=ctx.allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )

        return dq1, dq2, dk1, dk2, dv, dlq1, dlk1, dlq2, dlk2, dsubln_weight, None, None, None, None


def flash_diffattn_v1_fused_subln_rsi(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    v: torch.Tensor,
    lambda_q1: torch.Tensor,
    lambda_k1: torch.Tensor,
    lambda_q2: torch.Tensor,
    lambda_k2: torch.Tensor,
    subln_weight: torch.Tensor,
    lambda_init: float,
    eps: float = 1e-5,
    allow_tf32: bool = False,
    omit_radial_jacobian: bool = False,
) -> torch.Tensor:
    return _FlashDiffAttnV1FusedSubLNFn.apply(
        q1, q2, k1, k2, v,
        lambda_q1, lambda_k1, lambda_q2, lambda_k2,
        subln_weight, lambda_init, eps, allow_tf32, omit_radial_jacobian,
    )


# =============================================================================
# 6. Mathematical Oracles & Upstream Comparator Baselines (V2 and V1 Separate)
# =============================================================================

def eager_diffattn_v2_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lam_logit: torch.Tensor,
    wrong_gqa_chunk2_split: bool = False,
) -> torch.Tensor:
    """
    Exact mathematical oracle for Differential Transformer V2 (multihead_flashdiffv2.py).
    q: [B, T, 2*H, D]
    k: [B, T, H_KV, D]
    v: [B, T, H_KV, D]
    lam_logit: [B, T, H]
    """
    B, T, H2, D = q.shape
    H = H2 // 2
    H_KV = k.shape[2]
    n_rep2 = H2 // H_KV  # each KV head serves 2 * (H // H_KV) query heads
    work_dtype = torch.float64 if (q.dtype == torch.float64 or k.dtype == torch.float64 or v.dtype == torch.float64 or lam_logit.dtype == torch.float64) else torch.float32

    qf = q.to(work_dtype).transpose(1, 2)  # [B, 2H, T, D]
    kf = k.to(work_dtype).transpose(1, 2).repeat_interleave(n_rep2, dim=1)  # [B, 2H, T, D]
    vf = v.to(work_dtype).transpose(1, 2).repeat_interleave(n_rep2, dim=1)  # [B, 2H, T, D]

    scores = torch.matmul(qf, kf.transpose(-1, -2)) / math.sqrt(D)
    causal = torch.triu(torch.ones((T, T), device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    attn = torch.matmul(probs, vf).transpose(1, 2)  # [B, T, 2H, D]

    if wrong_gqa_chunk2_split:
        # Negative control: Diff-Transformer-V2 README documented GQA bug (chunk(2) across head dim)
        attn1, attn2 = attn[:, :, :H, :], attn[:, :, H:, :]
    else:
        attn1, attn2 = attn[:, :, 0::2, :], attn[:, :, 1::2, :]

    lam = torch.sigmoid(lam_logit.to(work_dtype)).unsqueeze(-1)
    out = attn1 - lam * attn2
    return out.to(q.dtype)


def eager_diffattn_v1_oracle(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    v: torch.Tensor,
    lambda_q1: torch.Tensor,
    lambda_k1: torch.Tensor,
    lambda_q2: torch.Tensor,
    lambda_k2: torch.Tensor,
    subln_weight: torch.Tensor,
    lambda_init: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Exact mathematical oracle for Differential Transformer V1 (multihead_flashdiff_1.py / 2.py).
    """
    B, T, H, D = q1.shape
    H_KV = k1.shape[2]
    n_rep = H // H_KV
    work_dtype = torch.float64 if (q1.dtype == torch.float64 or v.dtype == torch.float64 or lambda_q1.dtype == torch.float64) else torch.float32

    q1f = q1.to(work_dtype).transpose(1, 2)
    q2f = q2.to(work_dtype).transpose(1, 2)
    k1f = k1.to(work_dtype).transpose(1, 2).repeat_interleave(n_rep, dim=1)
    k2f = k2.to(work_dtype).transpose(1, 2).repeat_interleave(n_rep, dim=1)
    vf = v.to(work_dtype).transpose(1, 2).repeat_interleave(n_rep, dim=1)

    causal = torch.triu(torch.ones((T, T), device=q1.device, dtype=torch.bool), diagonal=1)
    s1 = torch.matmul(q1f, k1f.transpose(-1, -2)) / math.sqrt(D)
    s2 = torch.matmul(q2f, k2f.transpose(-1, -2)) / math.sqrt(D)
    s1 = s1.masked_fill(causal[None, None, :, :], float("-inf"))
    s2 = s2.masked_fill(causal[None, None, :, :], float("-inf"))
    p1 = torch.softmax(s1, dim=-1)
    p2 = torch.softmax(s2, dim=-1)

    o1 = torch.matmul(p1, vf).transpose(1, 2)  # [B, T, H, 2D]
    o2 = torch.matmul(p2, vf).transpose(1, 2)  # [B, T, H, 2D]

    lam1 = torch.exp(torch.sum(lambda_q1.to(work_dtype) * lambda_k1.to(work_dtype), dim=-1))
    lam2 = torch.exp(torch.sum(lambda_q2.to(work_dtype) * lambda_k2.to(work_dtype), dim=-1))
    lam_full = lam1 - lam2 + float(lambda_init)
    if lam_full.ndim == 1:
        u = o1 - lam_full[None, None, :, None] * o2
    else:
        u = o1 - lam_full * o2

    rstd = torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = u * rstd * subln_weight.to(work_dtype)[None, None, None, :] * (1.0 - float(lambda_init))
    return y.to(q1.dtype)


def upstream_multihead_diffattn_v1_buggy_gqa(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    v: torch.Tensor,
    lambda_q1: torch.Tensor,
    lambda_k1: torch.Tensor,
    lambda_q2: torch.Tensor,
    lambda_k2: torch.Tensor,
    subln_weight: torch.Tensor,
    lambda_init: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Verbatim reproduction of microsoft/unilm/Diff-Transformer/multihead_diffattn.py lines 79-104:
    `k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)` followed by
    `repeat_kv(k.transpose(1, 2), self.n_rep)` and `.view(bsz, self.num_heads, 2, tgt_len, src_len)`.
    Under GQA (n_rep >= 2 even), repeat_kv duplicates each of the 2*H_kv channels consecutively,
    causing stream 0 and stream 1 of head h to use the SAME repeated KV head instead of (k1, k2)!
    """
    B, T, H, D = q1.shape
    H_KV = k1.shape[2]
    n_rep = H // H_KV

    q = torch.stack([q1.float(), q2.float()], dim=3).view(B, T, 2 * H, D).transpose(1, 2)
    k = torch.stack([k1.float(), k2.float()], dim=3).view(B, T, 2 * H_KV, D).transpose(1, 2)
    # Bug in multihead_diffattn.py:86: repeat_kv on (B, 2*H_KV, T, D) before splitting streams!
    k_rep = k[:, :, None, :, :].expand(B, 2 * H_KV, n_rep, T, D).reshape(B, 2 * H, T, D)
    vf = v.float().transpose(1, 2).repeat_interleave(n_rep, dim=1)

    causal = torch.triu(torch.ones((T, T), device=q1.device, dtype=torch.bool), diagonal=1)
    attn_weights = torch.matmul(q, k_rep.transpose(-1, -2)) / math.sqrt(D)
    attn_weights = attn_weights.masked_fill(causal[None, None, :, :], float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)

    lam1 = torch.exp(torch.sum(lambda_q1.float() * lambda_k1.float(), dim=-1))
    lam2 = torch.exp(torch.sum(lambda_q2.float() * lambda_k2.float(), dim=-1))
    lam_full = lam1 - lam2 + float(lambda_init)

    attn_weights = attn_weights.view(B, H, 2, T, T)
    if lam_full.ndim == 1:
        diff_weights = attn_weights[:, :, 0] - lam_full[None, :, None, None] * attn_weights[:, :, 1]
    else:
        diff_weights = attn_weights[:, :, 0] - lam_full * attn_weights[:, :, 1]

    u = torch.matmul(diff_weights, vf).transpose(1, 2)
    rstd = torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = u * rstd * subln_weight.float()[None, None, None, :] * (1.0 - float(lambda_init))
    return y.to(q1.dtype)


@triton.jit
def _fused_diffv2_stride2_epilogue_fwd_kernel(
    ATTN_ptr, LAM_ptr, OUT_ptr,
    stride_ab, stride_at, stride_ah, stride_ad,
    stride_lb, stride_lt, stride_lh,
    stride_ob, stride_ot, stride_oh, stride_od,
    T, H,
    BM: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T
    base_a = ATTN_ptr + b * stride_ab + offs_m[:, None] * stride_at + offs_d[None, :] * stride_ad
    a1 = tl.load(base_a + (2 * h) * stride_ah, mask=mask_m[:, None], other=0.0).to(tl.float32)
    a2 = tl.load(base_a + (2 * h + 1) * stride_ah, mask=mask_m[:, None], other=0.0).to(tl.float32)
    lam = tl.sigmoid(tl.load(LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0).to(tl.float32))
    out = a1 - lam[:, None] * a2
    tl.store(OUT_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, out.to(OUT_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _fused_diffv2_stride2_epilogue_bwd_kernel(
    DO_ptr, ATTN_ptr, LAM_ptr, DATTN_ptr, DLAM_ptr,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_ab, stride_at, stride_ah, stride_ad,
    stride_lb, stride_lt, stride_lh,
    T, H,
    BM: tl.constexpr, D: tl.constexpr, OMIT_DLAMBDA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T
    do = tl.load(DO_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, mask=mask_m[:, None], other=0.0).to(tl.float32)
    a2 = tl.load(ATTN_ptr + b * stride_ab + (2 * h + 1) * stride_ah + offs_m[:, None] * stride_at + offs_d[None, :] * stride_ad, mask=mask_m[:, None], other=0.0).to(tl.float32)
    lam = tl.sigmoid(tl.load(LAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0).to(tl.float32))
    da2 = -lam[:, None] * do
    dlam = tl.zeros([BM], dtype=tl.float32) if OMIT_DLAMBDA else (-lam * (1.0 - lam) * tl.sum(do * a2, axis=1))
    base_da = DATTN_ptr + b * stride_ab + offs_m[:, None] * stride_at + offs_d[None, :] * stride_ad
    tl.store(base_da + (2 * h) * stride_ah, do.to(DATTN_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(base_da + (2 * h + 1) * stride_ah, da2.to(DATTN_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(DLAM_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, dlam.to(DLAM_ptr.dtype.element_ty), mask=mask_m)


class _FusedDiffV2Stride2EpilogueFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn: torch.Tensor, lam_logit: torch.Tensor, omit_dlambda: bool = False) -> torch.Tensor:
        B, T, H2, D = attn.shape
        H = H2 // 2
        out = torch.empty((B, T, H, D), device=attn.device, dtype=attn.dtype)
        _fused_diffv2_stride2_epilogue_fwd_kernel[(triton.cdiv(T, 32), B * H)](
            attn, lam_logit, out,
            attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
            lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            T, H, BM=32, D=D, num_warps=4,
        )
        ctx.save_for_backward(attn, lam_logit)
        ctx.omit_dlambda = omit_dlambda
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        attn, lam_logit = ctx.saved_tensors
        B, T, H2, D = attn.shape
        H = H2 // 2
        do = do.contiguous()
        dattn = torch.empty_like(attn)
        dlam = torch.empty_like(lam_logit)
        _fused_diffv2_stride2_epilogue_bwd_kernel[(triton.cdiv(T, 32), B * H)](
            do, attn, lam_logit, dattn, dlam,
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
            lam_logit.stride(0), lam_logit.stride(1), lam_logit.stride(2),
            T, H, BM=32, D=D, OMIT_DLAMBDA=ctx.omit_dlambda, num_warps=4,
        )
        return dattn, dlam, None


def fused_diffv2_stride2_epilogue_rsi(attn: torch.Tensor, lam_logit: torch.Tensor, omit_dlambda: bool = False) -> torch.Tensor:
    return _FusedDiffV2Stride2EpilogueFn.apply(attn, lam_logit, omit_dlambda)


@triton.jit
def _diffv1_cat_subln_fwd_kernel(
    ALO_ptr, AHI_ptr, LAM_FULL_ptr, GAMMA_ptr, Y_ptr, U_ptr, O2_ptr, RSTD_ptr,
    stride_ab, stride_at, stride_ah, stride_ad,
    stride_ub, stride_ut, stride_uh, stride_ud,
    stride_rb, stride_rh, stride_rt,
    one_minus_lambda_init, eps, B, T, H,
    BM: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d0 = tl.arange(0, D)
    offs_d1 = D + tl.arange(0, D)
    mask_m = offs_m < T

    lam_full = tl.load(LAM_FULL_ptr + h).to(tl.float32)
    base_a1 = b * stride_ab + h * stride_ah + offs_m[:, None] * stride_at + offs_d0[None, :] * stride_ad
    base_a2 = (B + b) * stride_ab + h * stride_ah + offs_m[:, None] * stride_at + offs_d0[None, :] * stride_ad
    o1_lo = tl.load(ALO_ptr + base_a1, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o2_lo = tl.load(ALO_ptr + base_a2, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o1_hi = tl.load(AHI_ptr + base_a1, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o2_hi = tl.load(AHI_ptr + base_a2, mask=mask_m[:, None], other=0.0).to(tl.float32)

    u_lo = o1_lo - lam_full * o2_lo
    u_hi = o1_hi - lam_full * o2_hi
    var = (tl.sum(u_lo * u_lo, axis=1) + tl.sum(u_hi * u_hi, axis=1)) / (2 * D)
    rstd = tl.rsqrt(var + eps)
    g0 = tl.load(GAMMA_ptr + offs_d0).to(tl.float32) * one_minus_lambda_init
    g1 = tl.load(GAMMA_ptr + offs_d1).to(tl.float32) * one_minus_lambda_init

    y_lo = (u_lo * rstd[:, None]) * g0[None, :]
    y_hi = (u_hi * rstd[:, None]) * g1[None, :]

    tl.store(RSTD_ptr + b * stride_rb + h * stride_rh + offs_m * stride_rt, rstd, mask=mask_m)
    base_u = b * stride_ub + h * stride_uh + offs_m[:, None] * stride_ut
    tl.store(U_ptr + base_u + offs_d0[None, :] * stride_ud, u_lo.to(U_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(U_ptr + base_u + offs_d1[None, :] * stride_ud, u_hi.to(U_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(O2_ptr + base_u + offs_d0[None, :] * stride_ud, o2_lo.to(O2_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(O2_ptr + base_u + offs_d1[None, :] * stride_ud, o2_hi.to(O2_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(Y_ptr + base_u + offs_d0[None, :] * stride_ud, y_lo.to(Y_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(Y_ptr + base_u + offs_d1[None, :] * stride_ud, y_hi.to(Y_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _diffv1_cat_subln_bwd_kernel(
    DY_ptr, U_ptr, O2_ptr, GAMMA_ptr, RSTD_ptr, LAM_FULL_ptr,
    DALO_ptr, DAHI_ptr, DGAMMA_PART_ptr, DLAM_PART_ptr,
    stride_ub, stride_ut, stride_uh, stride_ud,
    stride_ab, stride_at, stride_ah, stride_ad,
    stride_rb, stride_rh, stride_rt,
    stride_dp_m, stride_dp_bh, stride_dp_d,
    one_minus_lambda_init, B, T, H,
    OMIT_RADIAL_JACOBIAN: tl.constexpr, BM: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d0 = tl.arange(0, D)
    offs_d1 = D + tl.arange(0, D)
    mask_m = offs_m < T

    base_u = b * stride_ub + h * stride_uh + offs_m[:, None] * stride_ut
    dy0 = tl.load(DY_ptr + base_u + offs_d0[None, :] * stride_ud, mask=mask_m[:, None], other=0.0).to(tl.float32)
    dy1 = tl.load(DY_ptr + base_u + offs_d1[None, :] * stride_ud, mask=mask_m[:, None], other=0.0).to(tl.float32)
    u0 = tl.load(U_ptr + base_u + offs_d0[None, :] * stride_ud, mask=mask_m[:, None], other=0.0).to(tl.float32)
    u1 = tl.load(U_ptr + base_u + offs_d1[None, :] * stride_ud, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o2_0 = tl.load(O2_ptr + base_u + offs_d0[None, :] * stride_ud, mask=mask_m[:, None], other=0.0).to(tl.float32)
    o2_1 = tl.load(O2_ptr + base_u + offs_d1[None, :] * stride_ud, mask=mask_m[:, None], other=0.0).to(tl.float32)
    g0 = tl.load(GAMMA_ptr + offs_d0).to(tl.float32)
    g1 = tl.load(GAMMA_ptr + offs_d1).to(tl.float32)
    rstd = tl.load(RSTD_ptr + b * stride_rb + h * stride_rh + offs_m * stride_rt, mask=mask_m, other=0.0)
    lam_full = tl.load(LAM_FULL_ptr + h).to(tl.float32)

    uhat0 = u0 * rstd[:, None]
    uhat1 = u1 * rstd[:, None]
    duhat0 = dy0 * (g0[None, :] * one_minus_lambda_init)
    duhat1 = dy1 * (g1[None, :] * one_minus_lambda_init)

    if OMIT_RADIAL_JACOBIAN:
        du0 = rstd[:, None] * duhat0
        du1 = rstd[:, None] * duhat1
    else:
        mdot = (tl.sum(duhat0 * uhat0, axis=1) + tl.sum(duhat1 * uhat1, axis=1)) / (2 * D)
        du0 = rstd[:, None] * (duhat0 - uhat0 * mdot[:, None])
        du1 = rstd[:, None] * (duhat1 - uhat1 * mdot[:, None])

    dg0 = tl.sum(dy0 * uhat0, axis=0) * one_minus_lambda_init
    dg1 = tl.sum(dy1 * uhat1, axis=0) * one_minus_lambda_init
    tl.store(DGAMMA_PART_ptr + pid_m * stride_dp_m + pid_bh * stride_dp_bh + offs_d0 * stride_dp_d, dg0)
    tl.store(DGAMMA_PART_ptr + pid_m * stride_dp_m + pid_bh * stride_dp_bh + offs_d1 * stride_dp_d, dg1)

    dot_o2 = tl.sum(du0 * o2_0, axis=1) + tl.sum(du1 * o2_1, axis=1)
    tl.store(DLAM_PART_ptr + pid_m * tl.num_programs(1) + pid_bh, -tl.sum(dot_o2, axis=0))

    base_a1 = b * stride_ab + h * stride_ah + offs_m[:, None] * stride_at + offs_d0[None, :] * stride_ad
    base_a2 = (B + b) * stride_ab + h * stride_ah + offs_m[:, None] * stride_at + offs_d0[None, :] * stride_ad
    tl.store(DALO_ptr + base_a1, du0.to(DALO_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(DALO_ptr + base_a2, (-lam_full * du0).to(DALO_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(DAHI_ptr + base_a1, du1.to(DAHI_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(DAHI_ptr + base_a2, (-lam_full * du1).to(DAHI_ptr.dtype.element_ty), mask=mask_m[:, None])


class _FusedDiffV1CatSubLNFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_lo, a_hi, lq1, lk1, lq2, lk2, subln_w, lambda_init, eps=1e-5, omit_radial=False):
        B2, T, H, D = a_lo.shape
        B = B2 // 2
        D2 = 2 * D
        lam1 = torch.exp(torch.sum(lq1.float() * lk1.float(), dim=-1))
        lam2 = torch.exp(torch.sum(lq2.float() * lk2.float(), dim=-1))
        lam_full = (lam1 - lam2 + float(lambda_init)).contiguous()

        y = torch.empty((B, T, H, D2), device=a_lo.device, dtype=a_lo.dtype)
        u = torch.empty((B, T, H, D2), device=a_lo.device, dtype=a_lo.dtype)
        o2 = torch.empty((B, T, H, D2), device=a_lo.device, dtype=a_lo.dtype)
        rstd = torch.empty((B, H, T), device=a_lo.device, dtype=torch.float32)

        _diffv1_cat_subln_fwd_kernel[(triton.cdiv(T, 32), B * H)](
            a_lo, a_hi, lam_full, subln_w, y, u, o2, rstd,
            a_lo.stride(0), a_lo.stride(1), a_lo.stride(2), a_lo.stride(3),
            u.stride(0), u.stride(1), u.stride(2), u.stride(3),
            rstd.stride(0), rstd.stride(1), rstd.stride(2),
            float(1.0 - lambda_init), float(eps), B, T, H, BM=32, D=D, num_warps=4,
        )
        ctx.save_for_backward(lq1, lk1, lq2, lk2, subln_w, lam_full, lam1, lam2, u, o2, rstd)
        ctx.lambda_init = float(lambda_init)
        ctx.omit_radial = omit_radial
        ctx.a_shape = a_lo.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        lq1, lk1, lq2, lk2, subln_w, lam_full, lam1, lam2, u, o2, rstd = ctx.saved_tensors
        B2, T, H, D = ctx.a_shape
        B = B2 // 2
        D2 = 2 * D
        dy = dy.contiguous()
        da_lo = torch.empty((B2, T, H, D), device=dy.device, dtype=dy.dtype)
        da_hi = torch.empty((B2, T, H, D), device=dy.device, dtype=dy.dtype)
        n_m = triton.cdiv(T, 32)
        dg_part = torch.empty((n_m, B * H, D2), device=dy.device, dtype=torch.float32)
        dl_part = torch.empty((n_m, B * H), device=dy.device, dtype=torch.float32)

        _diffv1_cat_subln_bwd_kernel[(n_m, B * H)](
            dy, u, o2, subln_w, rstd, lam_full,
            da_lo, da_hi, dg_part, dl_part,
            u.stride(0), u.stride(1), u.stride(2), u.stride(3),
            da_lo.stride(0), da_lo.stride(1), da_lo.stride(2), da_lo.stride(3),
            rstd.stride(0), rstd.stride(1), rstd.stride(2),
            dg_part.stride(0), dg_part.stride(1), dg_part.stride(2),
            float(1.0 - ctx.lambda_init), B, T, H,
            OMIT_RADIAL_JACOBIAN=ctx.omit_radial, BM=32, D=D, num_warps=4,
        )
        dsw = dg_part.sum(dim=(0, 1)).to(subln_w.dtype)
        dlam_f = dl_part.sum(dim=0).view(B, H).sum(dim=0)
        dlq1 = (dlam_f[:, None] * lam1[:, None] * lk1.float()).to(lq1.dtype)
        dlk1 = (dlam_f[:, None] * lam1[:, None] * lq1.float()).to(lk1.dtype)
        dlq2 = (-dlam_f[:, None] * lam2[:, None] * lk2.float()).to(lq2.dtype)
        dlk2 = (-dlam_f[:, None] * lam2[:, None] * lq2.float()).to(lk2.dtype)
        return da_lo, da_hi, dlq1, dlk1, dlq2, dlk2, dsw, None, None, None


def fused_diffv1_cat_subln_rsi(
    a_lo: torch.Tensor,
    a_hi: torch.Tensor,
    lq1: torch.Tensor,
    lk1: torch.Tensor,
    lq2: torch.Tensor,
    lk2: torch.Tensor,
    subln_w: torch.Tensor,
    lambda_init: float,
    eps: float = 1e-5,
    omit_radial: bool = False,
) -> torch.Tensor:
    return _FusedDiffV1CatSubLNFn.apply(a_lo, a_hi, lq1, lk1, lq2, lk2, subln_w, lambda_init, eps, omit_radial)


# =============================================================================
# 7. Cubin Register-Spill Introspection Helper
# =============================================================================

def inspect_compiled_kernel_spills() -> Dict[str, Dict[str, int]]:
    """
    Introspects compiled Triton caches for all Round-10 RSI kernels and the negative-control
    monolithic D=128 spill kernel, returning {kernel_name: {n_regs, n_spills, specializations}}.
    """
    kernels = [
        ("_fused_interleaved_rope_fwd_kernel", _fused_interleaved_rope_fwd_kernel),
        ("_fused_interleaved_rope_bwd_kernel", _fused_interleaved_rope_bwd_kernel),
        ("_fused_interleaved_rope_qk_fwd_kernel", _fused_interleaved_rope_qk_fwd_kernel),
        ("_fused_interleaved_rope_qk_bwd_kernel", _fused_interleaved_rope_qk_bwd_kernel),
        ("_fused_diffv2_stride2_epilogue_fwd_kernel", _fused_diffv2_stride2_epilogue_fwd_kernel),
        ("_fused_diffv2_stride2_epilogue_bwd_kernel", _fused_diffv2_stride2_epilogue_bwd_kernel),
        ("_diffv1_cat_subln_fwd_kernel", _diffv1_cat_subln_fwd_kernel),
        ("_diffv1_cat_subln_bwd_kernel", _diffv1_cat_subln_bwd_kernel),
        ("_flash_diffv2_fwd_kernel", _flash_diffv2_fwd_kernel),
        ("_flash_diffv2_fwd_d128_kernel", _flash_diffv2_fwd_d128_kernel),
        ("_flash_diffv2_bwd_prep_kernel", _flash_diffv2_bwd_prep_kernel),
        ("_flash_diffv2_bwd_dq_kernel", _flash_diffv2_bwd_dq_kernel),
        ("_flash_diffv2_bwd_dq_d128_kernel", _flash_diffv2_bwd_dq_d128_kernel),
        ("_flash_diffv2_bwd_dkv_kernel", _flash_diffv2_bwd_dkv_kernel),
        ("_flash_diffv2_bwd_dkv_d128_kernel", _flash_diffv2_bwd_dkv_d128_kernel),
        ("_flash_diffv1_fwd_slice_kernel", _flash_diffv1_fwd_slice_kernel),
        ("_diffv1_subln_scale_fwd_kernel", _diffv1_subln_scale_fwd_kernel),
        ("_diffv1_subln_scale_bwd_kernel", _diffv1_subln_scale_bwd_kernel),
        ("_flash_diffv1_bwd_dq_kernel", _flash_diffv1_bwd_dq_kernel),
        ("_flash_diffv1_bwd_dkv_kernel", _flash_diffv1_bwd_dkv_kernel),
        ("_flash_diffv2_bwd_monolithic_spill_kernel", _flash_diffv2_bwd_monolithic_spill_kernel),
    ]
    report: Dict[str, Dict[str, int]] = {}
    for name, k_obj in kernels:
        max_regs = 0
        max_spills = 0
        specs = 0
        dev_caches = getattr(k_obj, "device_caches", {})
        if isinstance(dev_caches, dict) and dev_caches:
            for cache_obj in dev_caches.values():
                bin_map = cache_obj[0] if isinstance(cache_obj, tuple) else cache_obj
                if isinstance(bin_map, dict):
                    for compiled in bin_map.values():
                        if hasattr(compiled, "n_regs"):
                            max_regs = max(max_regs, int(compiled.n_regs or 0))
                            max_spills = max(max_spills, int(compiled.n_spills or 0))
                            specs += 1
        else:
            cache = getattr(k_obj, "cache", {})
            if isinstance(cache, dict):
                for dev_map in cache.values():
                    if isinstance(dev_map, dict):
                        for compiled in dev_map.values():
                            max_regs = max(max_regs, int(getattr(compiled, "n_regs", 0) or 0))
                            max_spills = max(max_spills, int(getattr(compiled, "n_spills", 0) or 0))
                            specs += 1
        report[name] = {
            "max_n_regs": max_regs,
            "max_n_spills": max_spills,
            "num_specializations": specs,
        }
    return report


# =============================================================================
# 8. Dual-LSE Ring DiffAttn-CP=8 Context Parallelism (V2 and V1 Contracts)
# =============================================================================

def _dual_lse_ring_update(
    o_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    o_step: torch.Tensor,
    lse_step: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Numerically exact online-softmax monoid combine across sequence shards for a single stream:
      o_acc:   [B, T_loc, H, D] (float32)
      lse_acc: [B, H, T_loc]    (float32, may be -inf on empty prefix)
      o_step:  [B, T_loc, H, D] (float32)
      lse_step:[B, H, T_loc]    (float32, may be -inf on causally masked rows)
    """
    valid_step = lse_step != float("-inf")
    valid_acc = lse_acc != float("-inf")

    lse_max = torch.maximum(lse_acc, lse_step)
    lse_max_safe = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)

    w_acc = torch.where(valid_acc, torch.exp(lse_acc - lse_max_safe), torch.zeros_like(lse_acc))
    w_step = torch.where(valid_step, torch.exp(lse_step - lse_max_safe), torch.zeros_like(lse_step))
    w_sum = w_acc + w_step

    lse_new = torch.where(w_sum > 0.0, lse_max_safe + torch.log(w_sum), torch.full_like(lse_max, float("-inf")))
    inv_w_sum = torch.where(w_sum > 0.0, 1.0 / w_sum, torch.zeros_like(w_sum))

    alpha = (w_acc * inv_w_sum).transpose(1, 2).unsqueeze(-1).contiguous()
    beta = (w_step * inv_w_sum).transpose(1, 2).unsqueeze(-1).contiguous()
    o_new = (alpha * o_acc + beta * o_step).contiguous()
    return o_new, lse_new


def ring_diffattn_v2_shard_fwd_triton(
    q_loc: torch.Tensor,
    k_shard: torch.Tensor,
    v_shard: torch.Tensor,
    lam_logit_loc: torch.Tensor,
    q_offset: int,
    k_offset: int,
    allow_tf32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs one causal shard of Co-Scheduled Paired-Query Flash-DiffAttn-V2 on (q_loc, k_shard, v_shard)
    and returns (o1_step, o2_step, lse1_step, lse2_step) in FP32.
    """
    B, T_loc, H2, D = q_loc.shape
    H = H2 // 2
    H_KV = k_shard.shape[2]
    n_rep = H // H_KV
    sm_scale = 1.0 / math.sqrt(D)

    o_step = torch.empty((B, T_loc, H, D), device=q_loc.device, dtype=torch.float32)
    o2_step = torch.empty((B, T_loc, H, D), device=q_loc.device, dtype=torch.float32)
    lse1_step = torch.empty((B, H, T_loc), device=q_loc.device, dtype=torch.float32)
    lse2_step = torch.empty((B, H, T_loc), device=q_loc.device, dtype=torch.float32)

    is_fp32 = q_loc.dtype == torch.float32
    BM = 16 if is_fp32 else 32
    BN = 16 if is_fp32 else 32

    if D == 64:
        grid = (triton.cdiv(T_loc, BM), B * H)
        _flash_diffv2_fwd_kernel[grid](
            q_loc, k_shard, v_shard, lam_logit_loc,
            o_step, o2_step, lse1_step, lse2_step,
            q_loc.stride(0), q_loc.stride(1), q_loc.stride(2), q_loc.stride(3),
            k_shard.stride(0), k_shard.stride(1), k_shard.stride(2), k_shard.stride(3),
            v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
            lam_logit_loc.stride(0), lam_logit_loc.stride(1), lam_logit_loc.stride(2),
            o_step.stride(0), o_step.stride(1), o_step.stride(2), o_step.stride(3),
            lse1_step.stride(0), lse1_step.stride(1), lse1_step.stride(2),
            sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
            IS_CAUSAL=True, BM=BM, BN=BN, D=64, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )
    else:
        grid = (triton.cdiv(T_loc, BM), B * H, 2)
        _flash_diffv2_fwd_d128_kernel[grid](
            q_loc, k_shard, v_shard, lam_logit_loc,
            o_step, o2_step, lse1_step, lse2_step,
            q_loc.stride(0), q_loc.stride(1), q_loc.stride(2), q_loc.stride(3),
            k_shard.stride(0), k_shard.stride(1), k_shard.stride(2), k_shard.stride(3),
            v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
            lam_logit_loc.stride(0), lam_logit_loc.stride(1), lam_logit_loc.stride(2),
            o_step.stride(0), o_step.stride(1), o_step.stride(2), o_step.stride(3),
            lse1_step.stride(0), lse1_step.stride(1), lse1_step.stride(2),
            sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
            IS_CAUSAL=True, BM=BM, BN=BN, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )

    lam = torch.sigmoid(lam_logit_loc.float()).unsqueeze(-1)
    o1_step = o_step + lam * o2_step
    return o1_step, o2_step, lse1_step, lse2_step


def ring_diffattn_v2_shard_bwd_triton(
    q_loc: torch.Tensor,
    k_shard: torch.Tensor,
    v_shard: torch.Tensor,
    do_loc: torch.Tensor,
    lam_logit_loc: torch.Tensor,
    lse1_glob: torch.Tensor,
    lse2_glob: torch.Tensor,
    delta1_glob: torch.Tensor,
    delta2_glob: torch.Tensor,
    q_offset: int,
    k_offset: int,
    allow_tf32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs one causal shard of Co-Scheduled Paired-Query Flash-DiffAttn-V2 backward using global
    (lse1_glob, lse2_glob, delta1_glob, delta2_glob), returning (dq_step, dk_step, dv_step).
    """
    B, T_loc, H2, D = q_loc.shape
    H = H2 // 2
    H_KV = k_shard.shape[2]
    n_rep = H // H_KV
    sm_scale = 1.0 / math.sqrt(D)

    dq_step = torch.empty_like(q_loc)
    dk_step = torch.empty_like(k_shard)
    dv_step = torch.empty_like(v_shard)

    is_fp32 = q_loc.dtype == torch.float32
    BM = 16 if is_fp32 else 32
    BN = 16 if is_fp32 else 32

    if D == 64:
        grid_dq = (triton.cdiv(T_loc, BM), B * H)
        _flash_diffv2_bwd_dq_kernel[grid_dq](
            q_loc, k_shard, v_shard, do_loc, lam_logit_loc,
            lse1_glob, lse2_glob, delta1_glob, delta2_glob, dq_step,
            q_loc.stride(0), q_loc.stride(1), q_loc.stride(2), q_loc.stride(3),
            k_shard.stride(0), k_shard.stride(1), k_shard.stride(2), k_shard.stride(3),
            v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
            do_loc.stride(0), do_loc.stride(1), do_loc.stride(2), do_loc.stride(3),
            lam_logit_loc.stride(0), lam_logit_loc.stride(1), lam_logit_loc.stride(2),
            lse1_glob.stride(0), lse1_glob.stride(1), lse1_glob.stride(2),
            sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
            IS_CAUSAL=True, BM=BM, BN=BN, D=64, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )
        grid_dkv = (triton.cdiv(T_loc, BN), B * H_KV)
        _flash_diffv2_bwd_dkv_kernel[grid_dkv](
            q_loc, k_shard, v_shard, do_loc, lam_logit_loc,
            lse1_glob, lse2_glob, delta1_glob, delta2_glob, dk_step, dv_step,
            q_loc.stride(0), q_loc.stride(1), q_loc.stride(2), q_loc.stride(3),
            k_shard.stride(0), k_shard.stride(1), k_shard.stride(2), k_shard.stride(3),
            v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
            do_loc.stride(0), do_loc.stride(1), do_loc.stride(2), do_loc.stride(3),
            lam_logit_loc.stride(0), lam_logit_loc.stride(1), lam_logit_loc.stride(2),
            lse1_glob.stride(0), lse1_glob.stride(1), lse1_glob.stride(2),
            sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
            IS_CAUSAL=True, BM=BM, BN=BN, D=64, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )
    else:
        grid_dq = (triton.cdiv(T_loc, BM), B * H, 2)
        _flash_diffv2_bwd_dq_d128_kernel[grid_dq](
            q_loc, k_shard, v_shard, do_loc, lam_logit_loc,
            lse1_glob, lse2_glob, delta1_glob, delta2_glob, dq_step,
            q_loc.stride(0), q_loc.stride(1), q_loc.stride(2), q_loc.stride(3),
            k_shard.stride(0), k_shard.stride(1), k_shard.stride(2), k_shard.stride(3),
            v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
            do_loc.stride(0), do_loc.stride(1), do_loc.stride(2), do_loc.stride(3),
            lam_logit_loc.stride(0), lam_logit_loc.stride(1), lam_logit_loc.stride(2),
            lse1_glob.stride(0), lse1_glob.stride(1), lse1_glob.stride(2),
            sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
            IS_CAUSAL=True, BM=BM, BN=BN, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )
        grid_dkv = (triton.cdiv(T_loc, BN), B * H_KV, 2)
        _flash_diffv2_bwd_dkv_d128_kernel[grid_dkv](
            q_loc, k_shard, v_shard, do_loc, lam_logit_loc,
            lse1_glob, lse2_glob, delta1_glob, delta2_glob, dk_step, dv_step,
            q_loc.stride(0), q_loc.stride(1), q_loc.stride(2), q_loc.stride(3),
            k_shard.stride(0), k_shard.stride(1), k_shard.stride(2), k_shard.stride(3),
            v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
            do_loc.stride(0), do_loc.stride(1), do_loc.stride(2), do_loc.stride(3),
            lam_logit_loc.stride(0), lam_logit_loc.stride(1), lam_logit_loc.stride(2),
            lse1_glob.stride(0), lse1_glob.stride(1), lse1_glob.stride(2),
            sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
            IS_CAUSAL=True, BM=BM, BN=BN, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1 if is_fp32 else 2,
        )
    return dq_step, dk_step, dv_step


def ring_diffattn_v1_shard_fwd_triton(
    q1_loc: torch.Tensor,
    q2_loc: torch.Tensor,
    k1_shard: torch.Tensor,
    k2_shard: torch.Tensor,
    v_shard: torch.Tensor,
    lam_full: torch.Tensor,
    q_offset: int,
    k_offset: int,
    allow_tf32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs one causal shard of Paired-Stream Flash-DiffAttn-V1 on (q1_loc, q2_loc, k1_shard, k2_shard, v_shard)
    and returns (o1_step, o2_step, lse1_step, lse2_step) in FP32.
    """
    B, T_loc, H, D = q1_loc.shape
    H_KV = k1_shard.shape[2]
    D2 = 2 * D
    n_rep = H // H_KV
    sm_scale = 1.0 / math.sqrt(D)
    has_per_head_lam = lam_full.ndim == 1 and lam_full.numel() == H

    u_step = torch.empty((B, T_loc, H, D2), device=q1_loc.device, dtype=torch.float32)
    o2_step = torch.empty((B, T_loc, H, D2), device=q1_loc.device, dtype=torch.float32)
    lse1_step = torch.empty((B, H, T_loc), device=q1_loc.device, dtype=torch.float32)
    lse2_step = torch.empty((B, H, T_loc), device=q1_loc.device, dtype=torch.float32)

    is_fp32 = q1_loc.dtype == torch.float32
    BM = 16 if is_fp32 else 32
    BN = 16 if is_fp32 else 32

    grid_fwd = (triton.cdiv(T_loc, BM), B * H, 2)
    _flash_diffv1_fwd_slice_kernel[grid_fwd](
        q1_loc, q2_loc, k1_shard, k2_shard, v_shard, lam_full,
        u_step, o2_step, lse1_step, lse2_step,
        q1_loc.stride(0), q1_loc.stride(1), q1_loc.stride(2), q1_loc.stride(3),
        k1_shard.stride(0), k1_shard.stride(1), k1_shard.stride(2), k1_shard.stride(3),
        v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
        u_step.stride(0), u_step.stride(1), u_step.stride(2), u_step.stride(3),
        lse1_step.stride(0), lse1_step.stride(1), lse1_step.stride(2),
        sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
        HAS_PER_HEAD_LAM=has_per_head_lam,
        IS_CAUSAL=True, BM=BM, BN=BN, D=D, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=1 if is_fp32 else 2,
    )

    if has_per_head_lam:
        o1_step = u_step + lam_full.float()[None, None, :, None] * o2_step
    else:
        o1_step = u_step + lam_full.float() * o2_step
    return o1_step, o2_step, lse1_step, lse2_step


def ring_diffattn_v1_shard_bwd_triton(
    q1_loc: torch.Tensor,
    q2_loc: torch.Tensor,
    k1_shard: torch.Tensor,
    k2_shard: torch.Tensor,
    v_shard: torch.Tensor,
    du_loc: torch.Tensor,
    lam_full: torch.Tensor,
    lse1_glob: torch.Tensor,
    lse2_glob: torch.Tensor,
    delta1_glob: torch.Tensor,
    delta2_glob: torch.Tensor,
    q_offset: int,
    k_offset: int,
    allow_tf32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs one causal shard of Paired-Stream Flash-DiffAttn-V1 backward using global
    (lse1_glob, lse2_glob, delta1_glob, delta2_glob), returning (dq1_step, dq2_step, dk1_step, dk2_step, dv_step).
    """
    B, T_loc, H, D = q1_loc.shape
    H_KV = k1_shard.shape[2]
    n_rep = H // H_KV
    sm_scale = 1.0 / math.sqrt(D)
    has_per_head_lam = lam_full.ndim == 1 and lam_full.numel() == H

    dq1_step = torch.empty_like(q1_loc)
    dq2_step = torch.empty_like(q2_loc)
    dk1_step = torch.empty_like(k1_shard)
    dk2_step = torch.empty_like(k2_shard)
    dv_step = torch.empty_like(v_shard)

    is_fp32 = q1_loc.dtype == torch.float32
    BM = 16 if is_fp32 else 32
    BN = 16 if is_fp32 else 32

    grid_dq = (triton.cdiv(T_loc, BM), B * H)
    _flash_diffv1_bwd_dq_kernel[grid_dq](
        q1_loc, q2_loc, k1_shard, k2_shard, v_shard, du_loc, lam_full,
        lse1_glob, lse2_glob, delta1_glob, delta2_glob, dq1_step, dq2_step,
        q1_loc.stride(0), q1_loc.stride(1), q1_loc.stride(2), q1_loc.stride(3),
        k1_shard.stride(0), k1_shard.stride(1), k1_shard.stride(2), k1_shard.stride(3),
        v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
        du_loc.stride(0), du_loc.stride(1), du_loc.stride(2), du_loc.stride(3),
        lse1_glob.stride(0), lse1_glob.stride(1), lse1_glob.stride(2),
        sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
        HAS_PER_HEAD_LAM=has_per_head_lam,
        IS_CAUSAL=True, BM=BM, BN=BN, D=D, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=1 if is_fp32 else 2,
    )

    grid_dkv = (triton.cdiv(T_loc, BN), B * H_KV, 2)
    _flash_diffv1_bwd_dkv_kernel[grid_dkv](
        q1_loc, q2_loc, k1_shard, k2_shard, v_shard, du_loc, lam_full,
        lse1_glob, lse2_glob, delta1_glob, delta2_glob, dk1_step, dk2_step, dv_step,
        q1_loc.stride(0), q1_loc.stride(1), q1_loc.stride(2), q1_loc.stride(3),
        k1_shard.stride(0), k1_shard.stride(1), k1_shard.stride(2), k1_shard.stride(3),
        v_shard.stride(0), v_shard.stride(1), v_shard.stride(2), v_shard.stride(3),
        du_loc.stride(0), du_loc.stride(1), du_loc.stride(2), du_loc.stride(3),
        lse1_glob.stride(0), lse1_glob.stride(1), lse1_glob.stride(2),
        sm_scale, T_loc, H, H_KV, n_rep, int(q_offset), int(k_offset),
        HAS_PER_HEAD_LAM=has_per_head_lam,
        IS_CAUSAL=True, BM=BM, BN=BN, D=D, ALLOW_TF32=allow_tf32,
        num_warps=4, num_stages=1 if is_fp32 else 2,
    )
    return dq1_step, dq2_step, dk1_step, dk2_step, dv_step


