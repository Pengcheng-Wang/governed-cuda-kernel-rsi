from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
import triton
import triton.language as tl

from common import environment_metadata

try:
    import pynvml  # type: ignore

    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False


# ============================================================================
# 1. Custom GPU-Resident Triton Kernels for Dropless Fine-Grained MoE Training
# ============================================================================


@triton.jit
def _grouped_mm_2d_3d_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    offs_ptr,
    max_m_tiles,
    E: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Device-resident Grouped GEMM: C[m, n] = sum_k A[m, k] * B[e(m), k, n].

    Resolves cumulative expert offsets `offs[0..E-1]` directly in SRAM/L1 with
    2D L2 swizzling (`GROUP_M`), zero host-device synchronization, zero empty-
    expert padding waste, and full CUDA Graph compatibility. Supports both
    Forward (`A=X`, `B=W`) and Input-Gradient `dX` (`A=dY`, `B=W^T` via swapped
    `stride_bk, stride_bn`).
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(max_m_tiles - first_pid_m, GROUP_M)
    pid_m_global = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    cum_m_tiles = 0
    prev_end = 0
    target_e = 0
    target_start = 0
    target_end = 0
    target_tile_in_e = 0
    found = 0

    for e in tl.static_range(0, E):
        end_e = tl.load(offs_ptr + e)
        m_e = end_e - prev_end
        tiles_e = (m_e + BLOCK_M - 1) // BLOCK_M
        in_range = (found == 0) & (pid_m_global < cum_m_tiles + tiles_e)
        if in_range:
            target_e = e
            target_start = prev_end
            target_end = end_e
            target_tile_in_e = pid_m_global - cum_m_tiles
            found = 1
        cum_m_tiles += tiles_e
        prev_end = end_e

    if found == 0:
        return

    rm = target_start + target_tile_in_e * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < target_end
    mask_n = rn < N

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + target_e * stride_be
        + rk[:, None] * stride_bk
        + rn[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + rk) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _grouped_mm_dw_kernel(
    x_ptr,
    dy_ptr,
    dw_ptr,
    offs_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_dym,
    stride_dyn,
    stride_dwe,
    stride_dwk,
    stride_dwn,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_K: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """Device-resident Weight-Gradient Grouped GEMM: dW[e, k, n] = X_e^T @ dY_e.

    Uses a 2D launch grid `(cdiv(K, BLOCK_K) * cdiv(N, BLOCK_N), E)` so every
    program indexes its expert `e` in `O(1)` without virtual-tile scanning,
    handles zero-token experts without extra memset kernels, and optionally
    accumulates in-place (`ACCUMULATE=True`) for multi-microbatch overlap.
    """
    pid_kn = tl.program_id(0)
    e = tl.program_id(1)

    start_m = 0 if e == 0 else tl.load(offs_ptr + e - 1)
    end_m = tl.load(offs_ptr + e)

    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_K * num_pid_n
    group_id = pid_kn // num_pid_in_group
    first_pid_k = group_id * GROUP_K
    group_size_k = min(num_pid_k - first_pid_k, GROUP_K)
    pid_k = first_pid_k + ((pid_kn % num_pid_in_group) % group_size_k)
    pid_n = (pid_kn % num_pid_in_group) // group_size_k

    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = rk < K
    mask_n = rn < N

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    m_count = end_m - start_m
    if m_count > 0:
        rm = start_m + tl.arange(0, BLOCK_M)
        x_ptrs = x_ptr + rk[:, None] * stride_xk + rm[None, :] * stride_xm
        dy_ptrs = dy_ptr + rm[:, None] * stride_dym + rn[None, :] * stride_dyn
        for m0 in range(0, m_count, BLOCK_M):
            cur_m = start_m + m0 + tl.arange(0, BLOCK_M)
            mask_m = cur_m < end_m
            xt = tl.load(x_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            dy = tl.load(dy_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(xt, dy, acc)
            x_ptrs += BLOCK_M * stride_xm
            dy_ptrs += BLOCK_M * stride_dym

    dw_ptrs = dw_ptr + e * stride_dwe + rk[:, None] * stride_dwk + rn[None, :] * stride_dwn
    if ACCUMULATE:
        prev = tl.load(dw_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += prev.to(tl.float32)
    tl.store(dw_ptrs, acc.to(tl.bfloat16), mask=mask_k[:, None] & mask_n[None, :])


def triton_grouped_mm_fwd(
    x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    m_tot, k_dim = x.shape
    num_e, _, n_dim = w.shape
    out = torch.empty((m_tot, n_dim), device=x.device, dtype=x.dtype)
    if m_tot == 0:
        return out
    avg_m = m_tot // max(num_e, 1)
    bm = 64 if avg_m < 192 else 128
    bn = 128
    bk = 64 if (k_dim % 64 == 0 and n_dim % 64 == 0) else 32
    warps = 4 if bm == 64 else 8
    stages = 4 if bk == 64 else 3
    max_m_tiles = triton.cdiv(m_tot, bm) + num_e
    grid = (max_m_tiles * triton.cdiv(n_dim, bn),)
    _grouped_mm_2d_3d_kernel[grid](
        x,
        w,
        out,
        offs,
        max_m_tiles,
        E=num_e,
        K=k_dim,
        N=n_dim,
        stride_am=x.stride(0),
        stride_ak=x.stride(1),
        stride_be=w.stride(0),
        stride_bk=w.stride(1),
        stride_bn=w.stride(2),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def triton_grouped_mm_dx(
    dy: torch.Tensor, w: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    m_tot, n_dim = dy.shape
    num_e, k_dim, _ = w.shape
    dx = torch.empty((m_tot, k_dim), device=dy.device, dtype=dy.dtype)
    if m_tot == 0:
        return dx
    avg_m = m_tot // max(num_e, 1)
    bm = 64 if avg_m < 192 else 128
    bn = 128
    bk = 64 if (n_dim % 64 == 0 and k_dim % 64 == 0) else 32
    warps = 4 if bm == 64 else 8
    stages = 4 if bk == 64 else 3
    max_m_tiles = triton.cdiv(m_tot, bm) + num_e
    grid = (max_m_tiles * triton.cdiv(k_dim, bn),)
    _grouped_mm_2d_3d_kernel[grid](
        dy,
        w,
        dx,
        offs,
        max_m_tiles,
        E=num_e,
        K=n_dim,
        N=k_dim,
        stride_am=dy.stride(0),
        stride_ak=dy.stride(1),
        stride_be=w.stride(0),
        stride_bk=w.stride(2),
        stride_bn=w.stride(1),
        stride_cm=dx.stride(0),
        stride_cn=dx.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        num_warps=warps,
        num_stages=stages,
    )
    return dx


def triton_grouped_mm_dw(
    x: torch.Tensor,
    dy: torch.Tensor,
    offs: torch.Tensor,
    num_experts: int,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
) -> torch.Tensor:
    m_tot, k_dim = x.shape
    _, n_dim = dy.shape
    if out is None:
        dw = torch.empty((num_experts, k_dim, n_dim), device=x.device, dtype=x.dtype)
    else:
        dw = out
    if m_tot == 0:
        if not accumulate:
            dw.zero_()
        return dw
    bkw = 128 if k_dim >= 256 else 64
    bnw = 128 if n_dim >= 256 else 64
    bmw = 64 if (m_tot // max(num_experts, 1)) >= 128 else 32
    warps = 8 if (bkw == 128 and bnw == 128) else 4
    grid = (triton.cdiv(k_dim, bkw) * triton.cdiv(n_dim, bnw), num_experts)
    _grouped_mm_dw_kernel[grid](
        x,
        dy,
        dw,
        offs,
        K=k_dim,
        N=n_dim,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_dym=dy.stride(0),
        stride_dyn=dy.stride(1),
        stride_dwe=dw.stride(0),
        stride_dwk=dw.stride(1),
        stride_dwn=dw.stride(2),
        BLOCK_K=bkw,
        BLOCK_N=bnw,
        BLOCK_M=bmw,
        GROUP_K=8,
        ACCUMULATE=accumulate,
        num_warps=warps,
        num_stages=3,
    )
    return dw


class TritonGroupedMMFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        x_c = x.contiguous()
        w_c = w.contiguous()
        ctx.save_for_backward(x_c, w_c, offs)
        return triton_grouped_mm_fwd(x_c, w_c, offs)

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        x_c, w_c, offs = ctx.saved_tensors
        dy_c = grad_out.contiguous()
        dx = triton_grouped_mm_dx(dy_c, w_c, offs) if ctx.needs_input_grad[0] else None
        dw = (
            triton_grouped_mm_dw(x_c, dy_c, offs, w_c.shape[0])
            if ctx.needs_input_grad[1]
            else None
        )
        return dx, dw, None


class PyTorchGroupedMMContiguousFn(torch.autograd.Function):
    """Wraps aten::_grouped_mm with .contiguous() backward guard so expanded/view
    upstream gradients do not trigger the SM80 `[0, 0]` stride RuntimeError."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        x_c = x.contiguous()
        w_c = w.contiguous()
        ctx.save_for_backward(x_c, w_c, offs)
        return torch._grouped_mm(x_c, w_c, offs=offs)

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        x_c, w_c, offs = ctx.saved_tensors
        dy_c = grad_out.contiguous()
        dx = (
            torch._grouped_mm(dy_c, w_c.transpose(-2, -1), offs=offs)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = (
            torch._grouped_mm(x_c.transpose(0, 1), dy_c, offs=offs)
            if ctx.needs_input_grad[1]
            else None
        )
        return dx, dw, None


# ============================================================================
# 2. Fused Triton Permute, SwiGLU, and Weighted-Unpermute (+ d_probs) Kernels
# ============================================================================


@triton.jit
def _moe_permute_fwd_kernel(
    x_ptr,
    perm_ptr,
    out_ptr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pos = tl.program_id(0)
    pid_h = tl.program_id(1)
    flat_slot = tl.load(perm_ptr + pos)
    tok = flat_slot // TOP_K
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    vals = tl.load(x_ptr + tok * H + offs_h, mask=mask_h, other=0.0)
    tl.store(out_ptr + pos * H + offs_h, vals, mask=mask_h)


@triton.jit
def _moe_unpermute_sum_bwd_kernel(
    dx_perm_ptr,
    inv_perm_ptr,
    dx_ptr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(0, TOP_K):
        pos = tl.load(inv_perm_ptr + tok * TOP_K + k)
        v = tl.load(dx_perm_ptr + pos * H + offs_h, mask=mask_h, other=0.0).to(
            tl.float32
        )
        acc += v
    tl.store(dx_ptr + tok * H + offs_h, acc.to(tl.bfloat16), mask=mask_h)


def triton_permute_fwd_op(
    x: torch.Tensor, perm: torch.Tensor, top_k: int
) -> torch.Tensor:
    num_tokens, hidden = x.shape
    m_tot = perm.numel()
    out = torch.empty((m_tot, hidden), device=x.device, dtype=x.dtype)
    if m_tot == 0:
        return out
    bh = 512 if hidden >= 512 else 256
    grid = (m_tot, triton.cdiv(hidden, bh))
    _moe_permute_fwd_kernel[grid](
        x, perm, out, H=hidden, TOP_K=top_k, BLOCK_H=bh, num_warps=4
    )
    return out


def triton_unpermute_sum_bwd_op(
    dx_perm: torch.Tensor, inv_perm: torch.Tensor, num_tokens: int, top_k: int
) -> torch.Tensor:
    hidden = dx_perm.shape[1]
    dx = torch.empty((num_tokens, hidden), device=dx_perm.device, dtype=dx_perm.dtype)
    if num_tokens == 0:
        return dx
    bh = 512 if hidden >= 512 else 256
    grid = (num_tokens, triton.cdiv(hidden, bh))
    _moe_unpermute_sum_bwd_kernel[grid](
        dx_perm, inv_perm, dx, H=hidden, TOP_K=top_k, BLOCK_H=bh, num_warps=4
    )
    return dx


class TritonMoEPermuteFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        perm: torch.Tensor,
        inv_perm: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        x_c = x.contiguous()
        ctx.save_for_backward(inv_perm)
        ctx.num_tokens = x_c.shape[0]
        ctx.top_k = top_k
        return triton_permute_fwd_op(x_c, perm, top_k)

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (inv_perm,) = ctx.saved_tensors
        dx = triton_unpermute_sum_bwd_op(
            grad_out.contiguous(), inv_perm, ctx.num_tokens, ctx.top_k
        )
        return dx, None, None, None


@triton.jit
def _moe_swiglu_fwd_kernel(
    h_ptr,
    out_ptr,
    I_DIM: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < I_DIM
    base = row * (2 * I_DIM)
    g = tl.load(h_ptr + base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
    u = tl.load(h_ptr + base + I_DIM + offs_i, mask=mask_i, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    out = (g * sig * u).to(tl.bfloat16)
    tl.store(out_ptr + row * I_DIM + offs_i, out, mask=mask_i)


@triton.jit
def _moe_swiglu_bwd_kernel(
    da_ptr,
    h_ptr,
    dh_ptr,
    I_DIM: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < I_DIM
    base = row * (2 * I_DIM)
    da = tl.load(da_ptr + row * I_DIM + offs_i, mask=mask_i, other=0.0).to(tl.float32)
    g = tl.load(h_ptr + base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
    u = tl.load(h_ptr + base + I_DIM + offs_i, mask=mask_i, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu_g = g * sig
    du = da * silu_g
    dg = da * u * sig * (1.0 + g * (1.0 - sig))
    tl.store(dh_ptr + base + offs_i, dg.to(tl.bfloat16), mask=mask_i)
    tl.store(dh_ptr + base + I_DIM + offs_i, du.to(tl.bfloat16), mask=mask_i)


def triton_swiglu_fwd_op(h: torch.Tensor) -> torch.Tensor:
    rows, two_i = h.shape
    i_dim = two_i // 2
    out = torch.empty((rows, i_dim), device=h.device, dtype=h.dtype)
    if rows == 0:
        return out
    bi = 512 if i_dim >= 512 else 256
    grid = (rows, triton.cdiv(i_dim, bi))
    _moe_swiglu_fwd_kernel[grid](h, out, I_DIM=i_dim, BLOCK_I=bi, num_warps=4)
    return out


def triton_swiglu_bwd_op(da: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    rows, two_i = h.shape
    i_dim = two_i // 2
    dh = torch.empty_like(h)
    if rows == 0:
        return dh
    bi = 512 if i_dim >= 512 else 256
    grid = (rows, triton.cdiv(i_dim, bi))
    _moe_swiglu_bwd_kernel[grid](da, h, dh, I_DIM=i_dim, BLOCK_I=bi, num_warps=4)
    return dh


class TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, h: torch.Tensor) -> torch.Tensor:
        h_c = h.contiguous()
        ctx.save_for_backward(h_c)
        return triton_swiglu_fwd_op(h_c)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        (h_c,) = ctx.saved_tensors
        return triton_swiglu_bwd_op(grad_out.contiguous(), h_c)


@triton.jit
def _moe_weighted_combine_fwd_kernel(
    y_perm_ptr,
    inv_perm_ptr,
    probs_ptr,
    out_ptr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(0, TOP_K):
        pos = tl.load(inv_perm_ptr + tok * TOP_K + k)
        prob = tl.load(probs_ptr + tok * TOP_K + k).to(tl.float32)
        y = tl.load(y_perm_ptr + pos * H + offs_h, mask=mask_h, other=0.0).to(
            tl.float32
        )
        acc += prob * y
    tl.store(out_ptr + tok * H + offs_h, acc.to(tl.bfloat16), mask=mask_h)


@triton.jit
def _moe_weighted_combine_bwd_kernel(
    dout_ptr,
    y_perm_ptr,
    inv_perm_ptr,
    probs_ptr,
    dy_perm_ptr,
    dprobs_ptr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Single-pass backward computing BOTH `dy_perm[pos, :] = dout[tok, :] * p`
    AND `dprobs[tok, k] = dot(dout[tok, :], y_perm[pos, :])` without ever
    materializing `[T, K, H]` tensors in HBM."""
    tok = tl.program_id(0)
    k = tl.program_id(1)
    pos = tl.load(inv_perm_ptr + tok * TOP_K + k)
    prob = tl.load(probs_ptr + tok * TOP_K + k).to(tl.float32)

    dot_vec = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        dout = tl.load(dout_ptr + tok * H + offs_h, mask=mask_h, other=0.0).to(
            tl.float32
        )
        y = tl.load(y_perm_ptr + pos * H + offs_h, mask=mask_h, other=0.0).to(
            tl.float32
        )
        dy = (dout * prob).to(tl.bfloat16)
        tl.store(dy_perm_ptr + pos * H + offs_h, dy, mask=mask_h)
        dot_vec += dout * y

    dot_val = tl.sum(dot_vec, axis=0)
    tl.store(dprobs_ptr + tok * TOP_K + k, dot_val)


def triton_weighted_combine_fwd_op(
    y_perm: torch.Tensor, inv_perm: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    num_tokens, top_k = probs.shape
    hidden = y_perm.shape[1]
    out = torch.empty((num_tokens, hidden), device=y_perm.device, dtype=y_perm.dtype)
    if num_tokens == 0:
        return out
    bh = 512 if hidden >= 512 else 256
    grid = (num_tokens, triton.cdiv(hidden, bh))
    _moe_weighted_combine_fwd_kernel[grid](
        y_perm, inv_perm, probs, out, H=hidden, TOP_K=top_k, BLOCK_H=bh, num_warps=4
    )
    return out


def triton_weighted_combine_bwd_op(
    dout: torch.Tensor,
    y_perm: torch.Tensor,
    inv_perm: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, top_k = probs.shape
    hidden = y_perm.shape[1]
    dy_perm = torch.empty_like(y_perm)
    dprobs = torch.empty((num_tokens, top_k), device=probs.device, dtype=torch.float32)
    if num_tokens == 0:
        return dy_perm, dprobs
    bh = 1024 if hidden >= 1024 else 512
    grid = (num_tokens, top_k)
    _moe_weighted_combine_bwd_kernel[grid](
        dout,
        y_perm,
        inv_perm,
        probs,
        dy_perm,
        dprobs,
        H=hidden,
        TOP_K=top_k,
        BLOCK_H=bh,
        num_warps=4,
    )
    return dy_perm, dprobs


class TritonWeightedCombineFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, y_perm: torch.Tensor, inv_perm: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        y_c = y_perm.contiguous()
        p_c = probs.contiguous()
        ctx.save_for_backward(y_c, inv_perm, p_c)
        return triton_weighted_combine_fwd_op(y_c, inv_perm, p_c)

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        y_c, inv_perm, p_c = ctx.saved_tensors
        dy_perm, dprobs = triton_weighted_combine_bwd_op(
            grad_out.contiguous(), y_c, inv_perm, p_c
        )
        return dy_perm, None, dprobs.to(p_c.dtype)


# ============================================================================
# 3. Routing Skew Generators, Aux/Z-Loss, and Single-GPU MoE Layer Variants
# ============================================================================


def make_skew_bias(
    num_experts: int, pattern: str, device: torch.device
) -> torch.Tensor:
    if pattern == "uniform":
        return torch.zeros(num_experts, device=device, dtype=torch.float32)
    if pattern == "zipf_s1p1":
        ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
        return 1.15 * torch.log(num_experts / ranks)
    if pattern == "zipf_s1p2":
        ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
        return 1.55 * torch.log(num_experts / ranks)
    if pattern == "hot80":
        b = torch.zeros(num_experts, device=device, dtype=torch.float32)
        hot_cnt = max(1, num_experts // 8)
        b[:hot_cnt] = 3.25
        return b
    if pattern == "hot_rank0_60pct":
        b = torch.zeros(num_experts, device=device, dtype=torch.float32)
        hot_cnt = max(1, num_experts // 8)
        b[:hot_cnt] = 2.35
        return b
    raise ValueError(f"Unknown pattern: {pattern}")


def router_topk_and_aux_loss(
    x: torch.Tensor,
    w_router: torch.Tensor,
    skew_bias: torch.Tensor,
    top_k: int,
    *,
    aux_coef: float = 1e-2,
    z_coef: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable FP32 router producing top-k normalized probabilities,
    selected expert IDs, Switch load-balancing loss, and router z-loss."""
    num_tokens = x.shape[0]
    num_experts = w_router.shape[1]
    logits = x.float() @ w_router + skew_bias
    lse = torch.logsumexp(logits, dim=-1)
    z_loss = z_coef * lse.square().mean()

    full_probs = torch.softmax(logits, dim=-1)
    top_vals, top_ids = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(top_vals, dim=-1)

    ones = torch.ones(num_tokens * top_k, device=x.device, dtype=torch.float32)
    expert_counts = torch.zeros(num_experts, device=x.device, dtype=torch.float32)
    expert_counts.scatter_add_(0, top_ids.reshape(-1), ones)
    f_e = expert_counts * (float(num_experts) / float(num_tokens * top_k))
    p_e = full_probs.mean(dim=0)
    aux_loss = aux_coef * torch.sum(f_e.detach() * p_e)
    return probs, top_ids, aux_loss + z_loss, expert_counts


def gpu_permutation_indices(
    top_ids: torch.Tensor,
    num_experts: int,
    ones_i32: torch.Tensor,
    arange_tk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """100% GPU-resident, zero-CPU-sync, CUDA-Graph-safe permutation builder."""
    flat_ids = top_ids.reshape(-1)
    perm = torch.argsort(flat_ids, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm.scatter_(0, perm, arange_tk)
    hist = torch.zeros(num_experts, device=top_ids.device, dtype=torch.int32)
    hist.scatter_add_(0, flat_ids, ones_i32)
    offs = torch.cumsum(hist, dim=0, dtype=torch.int32)
    return perm, inv_perm.view_as(top_ids), hist, offs


def eager_swiglu(h: torch.Tensor) -> torch.Tensor:
    g, u = h.chunk(2, dim=-1)
    return F.silu(g) * u


def eager_permute(x: torch.Tensor, perm: torch.Tensor, top_k: int) -> torch.Tensor:
    num_tokens, hidden = x.shape
    x_exp = x.unsqueeze(1).expand(num_tokens, top_k, hidden).reshape(-1, hidden)
    return x_exp.index_select(0, perm)


def eager_weighted_combine(
    y_perm: torch.Tensor, inv_perm: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    num_tokens, top_k = probs.shape
    hidden = y_perm.shape[1]
    y_unperm = y_perm.index_select(0, inv_perm.reshape(-1)).view(
        num_tokens, top_k, hidden
    )
    return (y_unperm.float() * probs.unsqueeze(-1)).sum(dim=1).to(y_perm.dtype)


_COMPILED_DISPATCHER_CACHE: dict[tuple[int, int, int], tuple[Callable, Callable, Callable]] = {}


def get_compiled_dispatcher(
    num_tokens: int, top_k: int, hidden: int
) -> tuple[Callable, Callable, Callable]:
    key = (num_tokens, top_k, hidden)
    if key not in _COMPILED_DISPATCHER_CACHE:
        c_perm = torch.compile(eager_permute, mode="default")
        c_act = torch.compile(eager_swiglu, mode="default")
        c_comb = torch.compile(eager_weighted_combine, mode="default")
        _COMPILED_DISPATCHER_CACHE[key] = (c_perm, c_act, c_comb)
    return _COMPILED_DISPATCHER_CACHE[key]


_COMPILED_PYTORCH_GMM_CACHE: dict[tuple[int, int, int], Callable] = {}


def _pytorch_gmm_expert_block(
    x_perm: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    h1 = PyTorchGroupedMMContiguousFn.apply(x_perm, w_gate_up, offs)
    a1 = eager_swiglu(h1)
    return PyTorchGroupedMMContiguousFn.apply(a1, w_down, offs)


def get_compiled_pytorch_gmm_block(
    m_tot: int, num_experts: int, hidden: int
) -> Callable:
    key = (m_tot, num_experts, hidden)
    if key not in _COMPILED_PYTORCH_GMM_CACHE:
        _COMPILED_PYTORCH_GMM_CACHE[key] = torch.compile(
            _pytorch_gmm_expert_block, mode="max-autotune-no-cudagraphs"
        )
    return _COMPILED_PYTORCH_GMM_CACHE[key]


def run_single_gpu_moe_layer(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    skew_bias: torch.Tensor,
    top_k: int,
    ones_i32: torch.Tensor,
    arange_tk: torch.Tensor,
    *,
    gemm_backend: str,
    dispatcher_backend: str,
    padded_max_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    num_tokens, hidden = x.shape
    num_experts = w_gate_up.shape[0]
    probs, top_ids, reg_loss, counts_f = router_topk_and_aux_loss(
        x, w_router, skew_bias, top_k
    )
    perm, inv_perm, hist, offs = gpu_permutation_indices(
        top_ids, num_experts, ones_i32, arange_tk
    )

    if dispatcher_backend == "eager":
        x_perm = eager_permute(x, perm, top_k)
    elif dispatcher_backend == "compiled":
        c_perm, _, _ = get_compiled_dispatcher(num_tokens, top_k, hidden)
        x_perm = c_perm(x, perm, top_k)
    elif dispatcher_backend == "triton_fused":
        x_perm = TritonMoEPermuteFn.apply(x, perm, inv_perm, top_k)
    else:
        raise ValueError(f"Unknown dispatcher_backend: {dispatcher_backend}")

    extra_stats: dict[str, float] = {}
    if gemm_backend == "eager_python_loop":
        offs_cpu = offs.tolist()
        outs = []
        start = 0
        for e, end in enumerate(offs_cpu):
            if end > start:
                xe = x_perm[start:end]
                he = xe @ w_gate_up[e]
                ae = eager_swiglu(he)
                outs.append(ae @ w_down[e])
            start = end
        y_perm = torch.cat(outs, dim=0)
    elif gemm_backend == "padded_bmm_3d":
        c_max = (
            int(padded_max_count)
            if padded_max_count is not None
            else max(1, int(hist.max().item()))
        )
        # Build intra-expert position via GPU arithmetic: pos - start_of_expert[e]
        sorted_e = top_ids.reshape(-1).index_select(0, perm)
        prev_offs = torch.zeros(num_experts, device=x.device, dtype=torch.int32)
        prev_offs[1:] = offs[:-1]
        intra_pos = arange_tk - prev_offs.index_select(0, sorted_e).to(torch.int64)
        bmm_slot = sorted_e * c_max + intra_pos

        x_pad = torch.zeros(
            (num_experts * c_max, hidden), device=x.device, dtype=x.dtype
        )
        x_pad_scatter = x_pad.index_put((bmm_slot,), x_perm, accumulate=False).view(
            num_experts, c_max, hidden
        )
        h_pad = torch.bmm(x_pad_scatter, w_gate_up)
        a_pad = (
            TritonSwiGLUFn.apply(h_pad.view(-1, h_pad.shape[-1])).view(
                num_experts, c_max, -1
            )
            if dispatcher_backend == "triton_fused"
            else eager_swiglu(h_pad)
        )
        y_pad = torch.bmm(a_pad, w_down).view(num_experts * c_max, hidden)
        y_perm = y_pad.index_select(0, bmm_slot)
        extra_stats["padding_blowup_ratio"] = float(num_experts * c_max) / float(
            num_tokens * top_k
        )
    elif gemm_backend == "pytorch_grouped_mm":
        h1 = PyTorchGroupedMMContiguousFn.apply(x_perm, w_gate_up, offs)
        if dispatcher_backend == "triton_fused":
            a1 = TritonSwiGLUFn.apply(h1)
        elif dispatcher_backend == "compiled":
            _, c_act, _ = get_compiled_dispatcher(num_tokens, top_k, hidden)
            a1 = c_act(h1)
        else:
            a1 = eager_swiglu(h1)
        y_perm = PyTorchGroupedMMContiguousFn.apply(a1, w_down, offs)
    elif gemm_backend == "pytorch_grouped_mm_compiled":
        c_blk = get_compiled_pytorch_gmm_block(
            num_tokens * top_k, num_experts, hidden
        )
        y_perm = c_blk(x_perm, w_gate_up, w_down, offs)
    elif gemm_backend == "triton_grouped_gemm":
        h1 = TritonGroupedMMFn.apply(x_perm, w_gate_up, offs)
        if dispatcher_backend == "triton_fused":
            a1 = TritonSwiGLUFn.apply(h1)
        elif dispatcher_backend == "compiled":
            _, c_act, _ = get_compiled_dispatcher(num_tokens, top_k, hidden)
            a1 = c_act(h1)
        else:
            a1 = eager_swiglu(h1)
        y_perm = TritonGroupedMMFn.apply(a1, w_down, offs)
    else:
        raise ValueError(f"Unknown gemm_backend: {gemm_backend}")

    if dispatcher_backend == "eager":
        out = eager_weighted_combine(y_perm, inv_perm, probs)
    elif dispatcher_backend == "compiled":
        _, _, c_comb = get_compiled_dispatcher(num_tokens, top_k, hidden)
        out = c_comb(y_perm, inv_perm, probs)
    elif dispatcher_backend == "triton_fused":
        out = TritonWeightedCombineFn.apply(y_perm, inv_perm, probs)
    else:
        raise ValueError(f"Unknown dispatcher_backend: {dispatcher_backend}")

    return out, reg_loss, extra_stats


# ============================================================================
# 4. Telemetry, Timing, Memory, and Numerical Verification Utilities
# ============================================================================


def sample_gpu_clocks_and_power(device_idx: int) -> dict[str, float]:
    if not HAS_PYNVML:
        return {"sm_clock_mhz": 0.0, "power_w": 0.0}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        clk = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        pwr = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        return {"sm_clock_mhz": clk, "power_w": pwr}
    except Exception:
        return {"sm_clock_mhz": 0.0, "power_w": 0.0}


def rel_l2_error(actual: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (actual.float() - ref.float()).norm().item()
    denom = max(ref.float().norm().item(), 1e-8)
    return float(diff / denom)


def cosine_sim(actual: torch.Tensor, ref: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(actual.float().flatten(), ref.float().flatten(), dim=0).item()
    )


def moe_layer_training_flops(
    num_tokens: int, hidden: int, num_experts: int, top_k: int, inter_dim: int
) -> float:
    """Exact fwd + bwd (dX + dW = 3x forward) matmul FLOPs for Router + SwiGLU MoE."""
    m_tot = num_tokens * top_k
    fwd_router = 2.0 * num_tokens * hidden * num_experts
    fwd_gate_up = 2.0 * m_tot * hidden * (2 * inter_dim)
    fwd_down = 2.0 * m_tot * inter_dim * hidden
    return 3.0 * (fwd_router + fwd_gate_up + fwd_down)


def time_step_fn(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    device_idx: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device_idx)
    torch.cuda.reset_peak_memory_stats(device_idx)
    base_mem = torch.cuda.memory_allocated(device_idx)
    fn()
    torch.cuda.synchronize(device_idx)
    peak_inc_mib = max(
        0.0, (torch.cuda.max_memory_allocated(device_idx) - base_mem) / (1024.0**2)
    )

    before_nvml = sample_gpu_clocks_and_power(device_idx)
    samples_ms: list[float] = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize(device_idx)
        samples_ms.append(float(start_ev.elapsed_time(end_ev)))
    after_nvml = sample_gpu_clocks_and_power(device_idx)

    s_sorted = sorted(samples_ms)
    n = len(s_sorted)
    q25 = s_sorted[max(0, (n - 1) // 4)]
    q75 = s_sorted[min(n - 1, (3 * (n - 1)) // 4)]
    return {
        "median_ms": round(statistics.median(s_sorted), 4),
        "iqr_ms": round(q75 - q25, 4),
        "min_ms": round(s_sorted[0], 4),
        "max_ms": round(s_sorted[-1], 4),
        "peak_activation_inc_mib": round(peak_inc_mib, 2),
        "sm_clock_before_mhz": before_nvml["sm_clock_mhz"],
        "sm_clock_after_mhz": after_nvml["sm_clock_mhz"],
        "power_after_w": round(after_nvml["power_w"], 2),
    }


def make_init_tensors(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    inter_dim: int,
    device: torch.device,
    seed: int = 20261115,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn((num_tokens, hidden), generator=gen, device=device, dtype=torch.bfloat16)
    dout = torch.randn(
        (num_tokens, hidden), generator=gen, device=device, dtype=torch.bfloat16
    ) * 0.1
    w_router = (
        torch.randn((hidden, num_experts), generator=gen, device=device, dtype=torch.float32)
        * (1.0 / math.sqrt(hidden))
    )
    w_gate_up = (
        torch.randn(
            (num_experts, hidden, 2 * inter_dim),
            generator=gen,
            device=device,
            dtype=torch.bfloat16,
        )
        * (1.0 / math.sqrt(hidden))
    )
    w_down = (
        torch.randn(
            (num_experts, inter_dim, hidden),
            generator=gen,
            device=device,
            dtype=torch.bfloat16,
        )
        * (1.0 / math.sqrt(inter_dim))
    )
    return x, dout, w_router, w_gate_up, w_down


def build_cudagraph_moe_runner(
    x_init: torch.Tensor,
    dout_init: torch.Tensor,
    w_router_init: torch.Tensor,
    w_gate_up_init: torch.Tensor,
    w_down_init: torch.Tensor,
    skew_bias: torch.Tensor,
    top_k: int,
    dispatcher_backend: str,
) -> dict[str, Any]:
    device = x_init.device
    num_tokens = x_init.shape[0]
    m_tot = num_tokens * top_k

    x_s = x_init.detach().clone().requires_grad_(True)
    dout_s = dout_init.detach().clone()
    wr_s = w_router_init.detach().clone().requires_grad_(True)
    wgu_s = w_gate_up_init.detach().clone().requires_grad_(True)
    wd_s = w_down_init.detach().clone().requires_grad_(True)
    x_s.grad = torch.zeros_like(x_s)
    wr_s.grad = torch.zeros_like(wr_s)
    wgu_s.grad = torch.zeros_like(wgu_s)
    wd_s.grad = torch.zeros_like(wd_s)
    skew_s = skew_bias.detach().clone()
    ones_i32 = torch.ones(m_tot, device=device, dtype=torch.int32)
    arange_tk = torch.arange(m_tot, device=device, dtype=torch.int64)

    def _body() -> torch.Tensor:
        x_s.grad.zero_()
        wr_s.grad.zero_()
        wgu_s.grad.zero_()
        wd_s.grad.zero_()
        out, reg_loss, _ = run_single_gpu_moe_layer(
            x_s,
            wr_s,
            wgu_s,
            wd_s,
            skew_s,
            top_k,
            ones_i32,
            arange_tk,
            gemm_backend="triton_grouped_gemm",
            dispatcher_backend=dispatcher_backend,
        )
        loss = (out.float() * dout_s.float()).sum() + reg_loss
        loss.backward()
        return loss

    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            loss_s = _body()
    torch.cuda.current_stream(device).wait_stream(side_stream)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        loss_s = _body()
    torch.cuda.synchronize(device)
    capture_inc_mib = max(
        0.0, (torch.cuda.max_memory_allocated(device) - base_mem) / (1024.0**2)
    )

    return {
        "graph": graph,
        "x_s": x_s,
        "dout_s": dout_s,
        "wr_s": wr_s,
        "wgu_s": wgu_s,
        "wd_s": wd_s,
        "skew_s": skew_s,
        "ones_i32": ones_i32,
        "arange_tk": arange_tk,
        "loss_s": loss_s,
        "capture_inc_mib": capture_inc_mib,
    }


# ============================================================================
# 5. Suite 0: PyTorch 2.13 _grouped_mm SM80 Audit & Backend Comparison
# ============================================================================


def run_suite_0_audit(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    num_tokens = 128 if is_smoke else 1024
    hidden = 512 if is_smoke else 2048
    inter_dim = 192 if is_smoke else 768
    num_experts = 16 if is_smoke else 64
    top_k = 4 if is_smoke else 8
    warmup = 2 if is_smoke else 6
    iters = 5 if is_smoke else 20
    m_tot = num_tokens * top_k

    # 1. Audit Trap 1: zero-strided backward grad rejection in aten::_grouped_mm
    x_t1 = torch.randn((128, 256), device=device, dtype=torch.bfloat16, requires_grad=True)
    w_t1 = torch.randn((4, 256, 512), device=device, dtype=torch.bfloat16, requires_grad=True)
    offs_t1 = torch.tensor([32, 64, 96, 128], device=device, dtype=torch.int32)
    zero_stride_rejected = False
    zero_stride_error = ""
    try:
        y_t1 = torch._grouped_mm(x_t1, w_t1, offs=offs_t1)
        y_t1.sum().backward()
    except RuntimeError as exc:
        zero_stride_rejected = True
        zero_stride_error = str(exc)[:180]

    # 2. Audit Trap 2: CUDA Graph capture failure due to internal unpinned offs.cpu() DtoH
    cudagraph_rejected = False
    cudagraph_error = ""
    s_cap = torch.cuda.Stream(device=device)
    s_cap.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s_cap):
        _ = torch._grouped_mm(x_t1.detach(), w_t1.detach(), offs=offs_t1)
    torch.cuda.current_stream(device).wait_stream(s_cap)
    try:
        g_fail = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_fail, stream=s_cap):
            _ = torch._grouped_mm(x_t1.detach(), w_t1.detach(), offs=offs_t1)
    except RuntimeError as exc:
        cudagraph_rejected = True
        cudagraph_error = str(exc)[:180]

    # 3. Audit Trap 3: Profiler kernel & DtoH counts on E-expert SwiGLU block (fwd + bwd)
    x_p, dout_p, wr_p, wgu_p, wd_p = make_init_tensors(
        num_tokens, hidden, num_experts, inter_dim, device, seed=11700
    )
    ones_i32 = torch.ones(m_tot, device=device, dtype=torch.int32)
    arange_tk = torch.arange(m_tot, device=device, dtype=torch.int64)
    skew_uniform = make_skew_bias(num_experts, "uniform", device)
    _, top_ids_p, _, _ = router_topk_and_aux_loss(x_p, wr_p, skew_uniform, top_k)
    perm_p, _, hist_p, offs_p = gpu_permutation_indices(
        top_ids_p, num_experts, ones_i32, arange_tk
    )
    active_experts_p = int((hist_p > 0).sum().item())
    xp_det = x_p.index_select(0, perm_p // top_k).detach().clone()
    dyp_det = torch.randn_like(xp_det)

    def _profile_expert_block(mode_name: str) -> dict[str, Any]:
        xp = xp_det.detach().clone().requires_grad_(True)
        wgu = wgu_p.detach().clone().requires_grad_(True)
        wd = wd_p.detach().clone().requires_grad_(True)

        def _step() -> None:
            xp.grad = None
            wgu.grad = None
            wd.grad = None
            if mode_name == "pytorch_grouped_mm":
                yp = _pytorch_gmm_expert_block(xp, wgu, wd, offs_p)
            elif mode_name == "pytorch_grouped_mm_compiled":
                c_fn = get_compiled_pytorch_gmm_block(m_tot, num_experts, hidden)
                yp = c_fn(xp, wgu, wd, offs_p)
            elif mode_name == "triton_grouped_gemm":
                h1 = TritonGroupedMMFn.apply(xp, wgu, offs_p)
                a1 = TritonSwiGLUFn.apply(h1)
                yp = TritonGroupedMMFn.apply(a1, wd, offs_p)
            else:
                raise ValueError(mode_name)
            (yp * dyp_det).sum().backward()

        for _ in range(3):
            _step()
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _step()
            torch.cuda.synchronize(device)

        dtoh_calls = 0
        gemm_launches = 0
        triton_gmm_launches = 0
        for evt in prof.key_averages():
            kname = evt.key
            if "DtoH" in kname:
                dtoh_calls += int(evt.count)
            if "_grouped_mm_" in kname:
                triton_gmm_launches += int(evt.count)
            elif "gemm" in kname.lower() or "cutlass" in kname.lower():
                gemm_launches += int(evt.count)
        return {
            "active_experts": active_experts_p,
            "dtoh_calls": dtoh_calls,
            "vendor_gemm_launches": gemm_launches,
            "triton_grouped_gemm_launches": triton_gmm_launches,
        }

    prof_audit = {
        m: _profile_expert_block(m)
        for m in (
            "pytorch_grouped_mm",
            "pytorch_grouped_mm_compiled",
            "triton_grouped_gemm",
        )
    }

    # 4. Finite-difference directional chain-rule check (FP64 reference vs BF16 Triton MoE)
    t_fd, h_fd, e_fd, i_fd, k_fd = 32, 128, 8, 64, 2
    x_fd, dout_fd, wr_fd, wgu_fd, wd_fd = make_init_tensors(
        t_fd, h_fd, e_fd, i_fd, device, seed=11701
    )
    skew_fd = make_skew_bias(e_fd, "zipf_s1p1", device)
    ones_fd = torch.ones(t_fd * k_fd, device=device, dtype=torch.int32)
    arange_fd = torch.arange(t_fd * k_fd, device=device, dtype=torch.int64)

    x_fd_g = x_fd.detach().clone().requires_grad_(True)
    wgu_fd_g = wgu_fd.detach().clone().requires_grad_(True)
    wd_fd_g = wd_fd.detach().clone().requires_grad_(True)
    wr_fd_g = wr_fd.detach().clone().requires_grad_(True)
    out_fd, reg_fd, _ = run_single_gpu_moe_layer(
        x_fd_g,
        wr_fd_g,
        wgu_fd_g,
        wd_fd_g,
        skew_fd,
        k_fd,
        ones_fd,
        arange_fd,
        gemm_backend="triton_grouped_gemm",
        dispatcher_backend="triton_fused",
    )
    loss_fd = (out_fd.float() * dout_fd.float()).sum() + reg_fd
    loss_fd.backward()

    # Hold discrete top_ids fixed around continuous perturbation so directional derivative is smooth
    with torch.no_grad():
        logits_0 = x_fd.double() @ wr_fd.double() + skew_fd.double()
        _, fixed_ids = torch.topk(logits_0, k_fd, dim=-1)
        perm_0, inv_perm_0, _, offs_0 = gpu_permutation_indices(
            fixed_ids, e_fd, ones_fd, arange_fd
        )

    def _smooth_loss_fp64(
        xd: torch.Tensor, wrd: torch.Tensor, wgud: torch.Tensor, wdd: torch.Tensor
    ) -> float:
        lg = xd @ wrd + skew_fd.double()
        lse = torch.logsumexp(lg, dim=-1)
        z_l = 1e-3 * lse.square().mean()
        fp = torch.softmax(lg, dim=-1)
        ones_f = torch.ones(t_fd * k_fd, device=device, dtype=torch.float64)
        cnts = torch.zeros(e_fd, device=device, dtype=torch.float64)
        cnts.scatter_add_(0, fixed_ids.reshape(-1), ones_f)
        f_e = cnts * (float(e_fd) / float(t_fd * k_fd))
        aux_l = 1e-2 * torch.sum(f_e * fp.mean(dim=0))
        top_v = torch.gather(lg, -1, fixed_ids)
        pr = torch.softmax(top_v, dim=-1)
        xp = xd.index_select(0, perm_0 // k_fd)
        outs = []
        st = 0
        for ee, en in enumerate(offs_0.tolist()):
            if en > st:
                hh = xp[st:en] @ wgud[ee]
                gg, uu = hh.chunk(2, dim=-1)
                aa = F.silu(gg) * uu
                outs.append(aa @ wdd[ee])
            st = en
        yp = torch.cat(outs, dim=0)
        yu = yp.index_select(0, inv_perm_0.reshape(-1)).view(t_fd, k_fd, h_fd)
        o = (yu * pr.unsqueeze(-1)).sum(dim=1)
        return float(((o * dout_fd.double()).sum() + aux_l + z_l).item())

    gen_fd = torch.Generator(device=device)
    gen_fd.manual_seed(11702)
    vx = torch.randn((t_fd, h_fd), generator=gen_fd, device=device, dtype=torch.float64)
    vwgu = torch.randn_like(wgu_fd, dtype=torch.float64)
    vwd = torch.randn_like(wd_fd, dtype=torch.float64)
    vwr = torch.randn_like(wr_fd, dtype=torch.float64)
    analytic_dir = float(
        (
            (x_fd_g.grad.double() * vx).sum()
            + (wgu_fd_g.grad.double() * vwgu).sum()
            + (wd_fd_g.grad.double() * vwd).sum()
            + (wr_fd_g.grad.double() * vwr).sum()
        ).item()
    )
    fd_errors: dict[str, float] = {}
    for eps in (2e-3, 1e-3):
        lp = _smooth_loss_fp64(
            x_fd.double() + eps * vx,
            wr_fd.double() + eps * vwr,
            wgu_fd.double() + eps * vwgu,
            wd_fd.double() + eps * vwd,
        )
        lm = _smooth_loss_fp64(
            x_fd.double() - eps * vx,
            wr_fd.double() - eps * vwr,
            wgu_fd.double() - eps * vwgu,
            wd_fd.double() - eps * vwd,
        )
        num_dir = (lp - lm) / (2.0 * eps)
        fd_errors[str(eps)] = abs(analytic_dir - num_dir) / max(abs(num_dir), 1e-8)

    # Also compute exact FP64 analytical refinement ratio across eps=2e-3 -> 1e-3
    x64 = x_fd.double().requires_grad_(True)
    wr64 = wr_fd.double().requires_grad_(True)
    wgu64 = wgu_fd.double().requires_grad_(True)
    wd64 = wd_fd.double().requires_grad_(True)
    lg64 = x64 @ wr64 + skew_fd.double()
    z64 = 1e-3 * torch.logsumexp(lg64, dim=-1).square().mean()
    fp64 = torch.softmax(lg64, dim=-1)
    cnts64 = torch.zeros(e_fd, device=device, dtype=torch.float64)
    cnts64.scatter_add_(
        0,
        fixed_ids.reshape(-1),
        torch.ones(t_fd * k_fd, device=device, dtype=torch.float64),
    )
    aux64 = 1e-2 * torch.sum((cnts64 * (float(e_fd) / float(t_fd * k_fd))) * fp64.mean(0))
    pr64 = torch.softmax(torch.gather(lg64, -1, fixed_ids), dim=-1)
    xp64 = x64.index_select(0, perm_0 // k_fd)
    outs64 = []
    st = 0
    for ee, en in enumerate(offs_0.tolist()):
        if en > st:
            hh = xp64[st:en] @ wgu64[ee]
            gg, uu = hh.chunk(2, dim=-1)
            outs64.append((F.silu(gg) * uu) @ wd64[ee])
        st = en
    yp64 = torch.cat(outs64, dim=0)
    o64 = (
        yp64.index_select(0, inv_perm_0.reshape(-1)).view(t_fd, k_fd, h_fd)
        * pr64.unsqueeze(-1)
    ).sum(dim=1)
    ((o64 * dout_fd.double()).sum() + aux64 + z64).backward()
    dir64 = float(
        (
            (x64.grad * vx).sum()
            + (wr64.grad * vwr).sum()
            + (wgu64.grad * vwgu).sum()
            + (wd64.grad * vwd).sum()
        ).item()
    )
    err_2e3 = abs(
        dir64
        - (
            _smooth_loss_fp64(
                x_fd.double() + 2e-3 * vx,
                wr_fd.double() + 2e-3 * vwr,
                wgu_fd.double() + 2e-3 * vwgu,
                wd_fd.double() + 2e-3 * vwd,
            )
            - _smooth_loss_fp64(
                x_fd.double() - 2e-3 * vx,
                wr_fd.double() - 2e-3 * vwr,
                wgu_fd.double() - 2e-3 * vwgu,
                wd_fd.double() - 2e-3 * vwd,
            )
        )
        / 4e-3
    ) / max(abs(dir64), 1e-8)
    err_1e3 = abs(
        dir64
        - (
            _smooth_loss_fp64(
                x_fd.double() + 1e-3 * vx,
                wr_fd.double() + 1e-3 * vwr,
                wgu_fd.double() + 1e-3 * vwgu,
                wd_fd.double() + 1e-3 * vwd,
            )
            - _smooth_loss_fp64(
                x_fd.double() - 1e-3 * vx,
                wr_fd.double() - 1e-3 * vwr,
                wgu_fd.double() - 1e-3 * vwgu,
                wd_fd.double() - 1e-3 * vwd,
            )
        )
        / 2e-3
    ) / max(abs(dir64), 1e-8)
    fd_refinement_ratio = err_2e3 / max(err_1e3, 1e-15)
    bf16_vs_fp64_dir_rel_err = abs(analytic_dir - dir64) / max(abs(dir64), 1e-8)

    # 5. Benchmark 6 Expert-GEMM backends across 3 skew patterns
    total_flops = moe_layer_training_flops(
        num_tokens, hidden, num_experts, top_k, inter_dim
    )
    skew_results: dict[str, Any] = {}
    max_loss_err = 0.0
    max_grad_err = 0.0
    max_cg_dynamic_err = 0.0

    for skew_name in ("uniform", "zipf_s1p2", "hot80"):
        skew_b = make_skew_bias(num_experts, skew_name, device)
        x0, dout0, wr0, wgu0, wd0 = make_init_tensors(
            num_tokens, hidden, num_experts, inter_dim, device, seed=11710
        )
        with torch.no_grad():
            _, tids0, _, cnts0 = router_topk_and_aux_loss(x0, wr0, skew_b, top_k)
            _, _, hist0, _ = gpu_permutation_indices(
                tids0, num_experts, ones_i32, arange_tk
            )
            c_max = max(1, int(hist0.max().item()))
            max_over_mean = float(c_max) / float(m_tot / num_experts)
            zero_experts = int((hist0 == 0).sum().item())

        # Reference: eager_python_loop
        xr = x0.detach().clone().requires_grad_(True)
        wrr = wr0.detach().clone().requires_grad_(True)
        wgur = wgu0.detach().clone().requires_grad_(True)
        wdr = wd0.detach().clone().requires_grad_(True)
        outr, regr, _ = run_single_gpu_moe_layer(
            xr,
            wrr,
            wgur,
            wdr,
            skew_b,
            top_k,
            ones_i32,
            arange_tk,
            gemm_backend="eager_python_loop",
            dispatcher_backend="eager",
        )
        loss_ref = (outr.float() * dout0.float()).sum() + regr
        loss_ref.backward()
        ref_loss_val = float(loss_ref.item())
        ref_loss_scale = max(
            abs(ref_loss_val),
            outr.float().norm().item() * dout0.float().norm().item(),
            1e-6,
        )

        modes_out: dict[str, Any] = {}
        for mode_key, g_bk, d_bk in (
            ("eager_python_loop", "eager_python_loop", "eager"),
            ("padded_bmm_3d", "padded_bmm_3d", "triton_fused"),
            ("pytorch_grouped_mm", "pytorch_grouped_mm", "triton_fused"),
            (
                "pytorch_grouped_mm_compiled",
                "pytorch_grouped_mm_compiled",
                "triton_fused",
            ),
            ("triton_grouped_gemm", "triton_grouped_gemm", "triton_fused"),
        ):
            xm = x0.detach().clone().requires_grad_(True)
            wrm = wr0.detach().clone().requires_grad_(True)
            wgum = wgu0.detach().clone().requires_grad_(True)
            wdm = wd0.detach().clone().requires_grad_(True)
            outm, regm, ex = run_single_gpu_moe_layer(
                xm,
                wrm,
                wgum,
                wdm,
                skew_b,
                top_k,
                ones_i32,
                arange_tk,
                gemm_backend=g_bk,
                dispatcher_backend=d_bk,
                padded_max_count=c_max,
            )
            lm = (outm.float() * dout0.float()).sum() + regm
            lm.backward()

            l_err = abs(float(lm.item()) - ref_loss_val) / ref_loss_scale
            out_err = rel_l2_error(outm, outr)
            dx_err = rel_l2_error(xm.grad, xr.grad)
            dw1_err = rel_l2_error(wgum.grad, wgur.grad)
            dw2_err = rel_l2_error(wdm.grad, wdr.grad)
            dwr_err = rel_l2_error(wrm.grad, wrr.grad)
            max_loss_err = max(max_loss_err, l_err, out_err)
            max_grad_err = max(max_grad_err, dx_err, dw1_err, dw2_err, dwr_err)

            def _bench_fn() -> None:
                xm.grad = None
                wrm.grad = None
                wgum.grad = None
                wdm.grad = None
                o_b, r_b, _ = run_single_gpu_moe_layer(
                    xm,
                    wrm,
                    wgum,
                    wdm,
                    skew_b,
                    top_k,
                    ones_i32,
                    arange_tk,
                    gemm_backend=g_bk,
                    dispatcher_backend=d_bk,
                    padded_max_count=c_max,
                )
                ((o_b.float() * dout0.float()).sum() + r_b).backward()

            tm = time_step_fn(_bench_fn, warmup=warmup, iters=iters, device_idx=device.index)
            tflops = (total_flops / (tm["median_ms"] * 1e-3)) / 1e12
            modes_out[mode_key] = {
                **tm,
                "achieved_tflops": round(tflops, 2),
                "padding_blowup_ratio": round(
                    ex.get("padding_blowup_ratio", 1.0), 3
                ),
                "rel_loss_err": round(l_err, 7),
                "dx_rel_l2": round(dx_err, 6),
                "dw_gate_up_rel_l2": round(dw1_err, 6),
                "dw_down_rel_l2": round(dw2_err, 6),
                "dw_router_rel_l2": round(dwr_err, 6),
            }

        # Dynamic-routing CUDA Graph runner on triton_grouped_gemm + triton_fused
        cg = build_cudagraph_moe_runner(
            x0, dout0, wr0, wgu0, wd0, skew_b, top_k, "triton_fused"
        )
        # Verify replay under a FRESH token tensor with a shifted routing distribution!
        x_fresh, dout_fresh, _, _, _ = make_init_tensors(
            num_tokens, hidden, num_experts, inter_dim, device, seed=99117
        )
        cg["x_s"].data.copy_(x_fresh)
        cg["dout_s"].copy_(dout_fresh)
        cg["graph"].replay()
        torch.cuda.synchronize(device)

        xf_ref = x_fresh.detach().clone().requires_grad_(True)
        wrf_ref = wr0.detach().clone().requires_grad_(True)
        wguf_ref = wgu0.detach().clone().requires_grad_(True)
        wdf_ref = wd0.detach().clone().requires_grad_(True)
        of_ref, rf_ref, _ = run_single_gpu_moe_layer(
            xf_ref,
            wrf_ref,
            wguf_ref,
            wdf_ref,
            skew_b,
            top_k,
            ones_i32,
            arange_tk,
            gemm_backend="triton_grouped_gemm",
            dispatcher_backend="triton_fused",
        )
        lf_ref = (of_ref.float() * dout_fresh.float()).sum() + rf_ref
        lf_ref.backward()
        cg_dyn_err = max(
            rel_l2_error(cg["x_s"].grad, xf_ref.grad),
            rel_l2_error(cg["wgu_s"].grad, wguf_ref.grad),
            rel_l2_error(cg["wd_s"].grad, wdf_ref.grad),
            rel_l2_error(cg["wr_s"].grad, wrf_ref.grad),
        )
        max_cg_dynamic_err = max(max_cg_dynamic_err, cg_dyn_err)

        # Restore x0/dout0 for timing parity
        cg["x_s"].data.copy_(x0)
        cg["dout_s"].copy_(dout0)
        tm_cg = time_step_fn(
            lambda: cg["graph"].replay(),
            warmup=warmup,
            iters=iters,
            device_idx=device.index,
        )
        tm_cg["peak_activation_inc_mib"] = round(cg["capture_inc_mib"], 2)
        modes_out["triton_grouped_gemm_cudagraph"] = {
            **tm_cg,
            "achieved_tflops": round(
                (total_flops / (tm_cg["median_ms"] * 1e-3)) / 1e12, 2
            ),
            "padding_blowup_ratio": 1.0,
            "dynamic_routing_replay_max_grad_rel_l2": round(cg_dyn_err, 7),
        }

        skew_results[skew_name] = {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "top_k": top_k,
            "max_expert_count": c_max,
            "max_over_mean_skew": round(max_over_mean, 3),
            "zero_token_experts": zero_experts,
            "modes": modes_out,
        }

    return {
        "audits": {
            "zero_stride_grad_rejected": zero_stride_rejected,
            "zero_stride_error_excerpt": zero_stride_error,
            "cudagraph_capture_rejected": cudagraph_rejected,
            "cudagraph_error_excerpt": cudagraph_error,
            "profiler_kernel_and_dtoh_counts": prof_audit,
            "chain_rule_fd_refinement_ratio": round(fd_refinement_ratio, 4),
            "chain_rule_bf16_vs_fp64_dir_rel_err": round(bf16_vs_fp64_dir_rel_err, 6),
            "chain_rule_fd_errors": {k: round(v, 6) for k, v in fd_errors.items()},
            "max_suite0_rel_loss_err": round(max_loss_err, 7),
            "max_suite0_grad_rel_l2": round(max_grad_err, 6),
            "max_cudagraph_dynamic_routing_grad_rel_l2": round(max_cg_dynamic_err, 7),
        },
        "skew_sweep": skew_results,
    }


# ============================================================================
# 6. Suite A: ISO-FLOP Coarse vs Fine-Grained Expert Granularity Frontier
# ============================================================================


def run_suite_a_granularity(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    hidden = 512 if is_smoke else 2048
    if is_smoke:
        granularities = [
            ("coarse_e8_k2", 8, 2, 384),
            ("fine_e16_k4", 16, 4, 192),
        ]
        token_list = [64, 256]
    else:
        # Constant active expert width K * I_moe = 6,144 across all 3 granularities!
        granularities = [
            ("coarse_e8_k2", 8, 2, 3072),
            ("medium_e32_k4", 32, 4, 1536),
            ("fine_e64_k8", 64, 8, 768),
        ]
        token_list = [256, 512, 1024, 2048, 4096]

    warmup = 2 if is_smoke else 6
    iters = 5 if is_smoke else 20
    results: dict[str, Any] = {}
    max_grad_err = 0.0

    for g_label, num_e, top_k, inter_dim in granularities:
        g_dict: dict[str, Any] = {}
        skew_b = make_skew_bias(num_e, "zipf_s1p1", device)
        for num_tokens in token_list:
            m_tot = num_tokens * top_k
            ones_i32 = torch.ones(m_tot, device=device, dtype=torch.int32)
            arange_tk = torch.arange(m_tot, device=device, dtype=torch.int64)
            x0, dout0, wr0, wgu0, wd0 = make_init_tensors(
                num_tokens, hidden, num_e, inter_dim, device, seed=11720 + num_tokens + num_e
            )
            with torch.no_grad():
                _, tids0, _, _ = router_topk_and_aux_loss(x0, wr0, skew_b, top_k)
                _, _, hist0, _ = gpu_permutation_indices(
                    tids0, num_e, ones_i32, arange_tk
                )
                c_max = max(1, int(hist0.max().item()))

            # Reference: pytorch_grouped_mm
            xr = x0.detach().clone().requires_grad_(True)
            wrr = wr0.detach().clone().requires_grad_(True)
            wgur = wgu0.detach().clone().requires_grad_(True)
            wdr = wd0.detach().clone().requires_grad_(True)
            outr, regr, _ = run_single_gpu_moe_layer(
                xr,
                wrr,
                wgur,
                wdr,
                skew_b,
                top_k,
                ones_i32,
                arange_tk,
                gemm_backend="pytorch_grouped_mm",
                dispatcher_backend="triton_fused",
            )
            ((outr.float() * dout0.float()).sum() + regr).backward()

            total_flops = moe_layer_training_flops(
                num_tokens, hidden, num_e, top_k, inter_dim
            )
            modes_res: dict[str, Any] = {}
            for m_name, g_bk in (
                ("eager_python_loop", "eager_python_loop"),
                ("padded_bmm_3d", "padded_bmm_3d"),
                ("pytorch_grouped_mm", "pytorch_grouped_mm"),
                ("triton_grouped_gemm", "triton_grouped_gemm"),
            ):
                xm = x0.detach().clone().requires_grad_(True)
                wrm = wr0.detach().clone().requires_grad_(True)
                wgum = wgu0.detach().clone().requires_grad_(True)
                wdm = wd0.detach().clone().requires_grad_(True)
                outm, regm, ex = run_single_gpu_moe_layer(
                    xm,
                    wrm,
                    wgum,
                    wdm,
                    skew_b,
                    top_k,
                    ones_i32,
                    arange_tk,
                    gemm_backend=g_bk,
                    dispatcher_backend="triton_fused",
                    padded_max_count=c_max,
                )
                ((outm.float() * dout0.float()).sum() + regm).backward()
                g_err = max(
                    rel_l2_error(xm.grad, xr.grad),
                    rel_l2_error(wgum.grad, wgur.grad),
                    rel_l2_error(wdm.grad, wdr.grad),
                )
                max_grad_err = max(max_grad_err, g_err)

                def _fn() -> None:
                    xm.grad = None
                    wrm.grad = None
                    wgum.grad = None
                    wdm.grad = None
                    ob, rb, _ = run_single_gpu_moe_layer(
                        xm,
                        wrm,
                        wgum,
                        wdm,
                        skew_b,
                        top_k,
                        ones_i32,
                        arange_tk,
                        gemm_backend=g_bk,
                        dispatcher_backend="triton_fused",
                        padded_max_count=c_max,
                    )
                    ((ob.float() * dout0.float()).sum() + rb).backward()

                tm = time_step_fn(_fn, warmup=warmup, iters=iters, device_idx=device.index)
                modes_res[m_name] = {
                    **tm,
                    "achieved_tflops": round(
                        (total_flops / (tm["median_ms"] * 1e-3)) / 1e12, 2
                    ),
                    "padding_blowup_ratio": round(
                        ex.get("padding_blowup_ratio", 1.0), 3
                    ),
                    "max_grad_rel_l2": round(g_err, 6),
                }

            cg = build_cudagraph_moe_runner(
                x0, dout0, wr0, wgu0, wd0, skew_b, top_k, "triton_fused"
            )
            tm_cg = time_step_fn(
                lambda: cg["graph"].replay(),
                warmup=warmup,
                iters=iters,
                device_idx=device.index,
            )
            tm_cg["peak_activation_inc_mib"] = round(cg["capture_inc_mib"], 2)
            modes_res["triton_grouped_gemm_cudagraph"] = {
                **tm_cg,
                "achieved_tflops": round(
                    (total_flops / (tm_cg["median_ms"] * 1e-3)) / 1e12, 2
                ),
                "padding_blowup_ratio": 1.0,
                "max_grad_rel_l2": modes_res["triton_grouped_gemm"]["max_grad_rel_l2"],
            }
            del cg

            g_dict[str(num_tokens)] = {
                "num_tokens": num_tokens,
                "routed_tokens_m_tot": m_tot,
                "mean_tokens_per_expert": round(float(m_tot) / float(num_e), 1),
                "max_expert_tokens": c_max,
                "num_experts": num_e,
                "top_k": top_k,
                "inter_dim": inter_dim,
                "modes": modes_res,
            }
        results[g_label] = g_dict
    results["_max_grad_rel_l2"] = round(max_grad_err, 6)
    return results


# ============================================================================
# 7. Suite B: Fused Token Dispatcher Memory & Bandwidth Ablation (LLM H=4096)
# ============================================================================


def run_suite_b_dispatcher(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    hidden = 512 if is_smoke else 4096
    inter_dim = 192 if is_smoke else 1536
    num_e = 16 if is_smoke else 64
    top_k = 4 if is_smoke else 8
    token_list = [128, 256] if is_smoke else [1024, 2048, 4096]
    warmup = 2 if is_smoke else 5
    iters = 5 if is_smoke else 15

    skew_b = make_skew_bias(num_e, "zipf_s1p1", device)
    out_by_tokens: dict[str, Any] = {}
    max_grad_err = 0.0

    for num_tokens in token_list:
        m_tot = num_tokens * top_k
        ones_i32 = torch.ones(m_tot, device=device, dtype=torch.int32)
        arange_tk = torch.arange(m_tot, device=device, dtype=torch.int64)
        x0, dout0, wr0, wgu0, wd0 = make_init_tensors(
            num_tokens, hidden, num_e, inter_dim, device, seed=11740 + num_tokens
        )
        total_flops = moe_layer_training_flops(
            num_tokens, hidden, num_e, top_k, inter_dim
        )

        # Reference: unfused eager dispatcher
        xr = x0.detach().clone().requires_grad_(True)
        wrr = wr0.detach().clone().requires_grad_(True)
        wgur = wgu0.detach().clone().requires_grad_(True)
        wdr = wd0.detach().clone().requires_grad_(True)
        outr, regr, _ = run_single_gpu_moe_layer(
            xr,
            wrr,
            wgur,
            wdr,
            skew_b,
            top_k,
            ones_i32,
            arange_tk,
            gemm_backend="triton_grouped_gemm",
            dispatcher_backend="eager",
        )
        ((outr.float() * dout0.float()).sum() + regr).backward()

        modes_res: dict[str, Any] = {}
        for m_label, d_bk in (
            ("unfused_eager_dispatcher", "eager"),
            ("compiled_dispatcher", "compiled"),
            ("triton_fused_dispatcher", "triton_fused"),
        ):
            xm = x0.detach().clone().requires_grad_(True)
            wrm = wr0.detach().clone().requires_grad_(True)
            wgum = wgu0.detach().clone().requires_grad_(True)
            wdm = wd0.detach().clone().requires_grad_(True)
            outm, regm, _ = run_single_gpu_moe_layer(
                xm,
                wrm,
                wgum,
                wdm,
                skew_b,
                top_k,
                ones_i32,
                arange_tk,
                gemm_backend="triton_grouped_gemm",
                dispatcher_backend=d_bk,
            )
            ((outm.float() * dout0.float()).sum() + regm).backward()
            g_err = max(
                rel_l2_error(xm.grad, xr.grad),
                rel_l2_error(wgum.grad, wgur.grad),
                rel_l2_error(wdm.grad, wdr.grad),
                rel_l2_error(wrm.grad, wrr.grad),
            )
            max_grad_err = max(max_grad_err, g_err)

            def _fn() -> None:
                xm.grad = None
                wrm.grad = None
                wgum.grad = None
                wdm.grad = None
                ob, rb, _ = run_single_gpu_moe_layer(
                    xm,
                    wrm,
                    wgum,
                    wdm,
                    skew_b,
                    top_k,
                    ones_i32,
                    arange_tk,
                    gemm_backend="triton_grouped_gemm",
                    dispatcher_backend=d_bk,
                )
                ((ob.float() * dout0.float()).sum() + rb).backward()

            tm = time_step_fn(_fn, warmup=warmup, iters=iters, device_idx=device.index)
            modes_res[m_label] = {
                **tm,
                "achieved_tflops": round(
                    (total_flops / (tm["median_ms"] * 1e-3)) / 1e12, 2
                ),
                "max_grad_rel_l2": round(g_err, 6),
            }

        cg = build_cudagraph_moe_runner(
            x0, dout0, wr0, wgu0, wd0, skew_b, top_k, "triton_fused"
        )
        tm_cg = time_step_fn(
            lambda: cg["graph"].replay(),
            warmup=warmup,
            iters=iters,
            device_idx=device.index,
        )
        tm_cg["peak_activation_inc_mib"] = round(cg["capture_inc_mib"], 2)
        modes_res["triton_fused_full_cudagraph"] = {
            **tm_cg,
            "achieved_tflops": round(
                (total_flops / (tm_cg["median_ms"] * 1e-3)) / 1e12, 2
            ),
            "max_grad_rel_l2": modes_res["triton_fused_dispatcher"]["max_grad_rel_l2"],
        }
        del cg

        # Isolated Permute + Weighted-Combine (fwd + bwd) microbenchmark for exact GB/s
        with torch.no_grad():
            probs_iso, tids_iso, _, _ = router_topk_and_aux_loss(
                x0, wr0, skew_b, top_k
            )
            perm_iso, inv_perm_iso, _, _ = gpu_permutation_indices(
                tids_iso, num_e, ones_i32, arange_tk
            )
        y_perm_iso = torch.randn((m_tot, hidden), device=device, dtype=torch.bfloat16)
        dy_perm_iso = torch.randn_like(y_perm_iso)
        # Algorithmic minimum useful bytes for Permute(fwd+bwd) + WeightedCombine(fwd+bwd):
        # Permute fwd: read X(T*H*2) + write X_perm(M*H*2)
        # Permute bwd: read dX_perm(M*H*2) + write dX(T*H*2)
        # Combine fwd: read Y_perm(M*H*2) + write Out(T*H*2)
        # Combine bwd: read dOut(T*H*2) + read Y_perm(M*H*2) + write dY_perm(M*H*2) + write dProbs(M*4)
        useful_bytes = (
            4.0 * num_tokens * hidden * 2.0
            + 5.0 * m_tot * hidden * 2.0
            + m_tot * 4.0
        )

        def _iso_eager() -> None:
            xi = x0.detach().requires_grad_(True)
            yi = y_perm_iso.detach().requires_grad_(True)
            pi = probs_iso.detach().requires_grad_(True)
            xp = eager_permute(xi, perm_iso, top_k)
            oc = eager_weighted_combine(yi, inv_perm_iso, pi)
            ((xp * dy_perm_iso).sum() + (oc * dout0).sum()).backward()

        def _iso_triton() -> None:
            xi = x0.detach().requires_grad_(True)
            yi = y_perm_iso.detach().requires_grad_(True)
            pi = probs_iso.detach().requires_grad_(True)
            xp = TritonMoEPermuteFn.apply(xi, perm_iso, inv_perm_iso, top_k)
            oc = TritonWeightedCombineFn.apply(yi, inv_perm_iso, pi)
            ((xp * dy_perm_iso).sum() + (oc * dout0).sum()).backward()

        t_iso_eager = time_step_fn(
            _iso_eager, warmup=warmup, iters=iters, device_idx=device.index
        )
        t_iso_triton = time_step_fn(
            _iso_triton, warmup=warmup, iters=iters, device_idx=device.index
        )

        out_by_tokens[str(num_tokens)] = {
            "num_tokens": num_tokens,
            "routed_tokens_m_tot": m_tot,
            "hidden": hidden,
            "inter_dim": inter_dim,
            "modes": modes_res,
            "isolated_permute_combine_fwd_bwd": {
                "eager_median_ms": t_iso_eager["median_ms"],
                "eager_peak_inc_mib": t_iso_eager["peak_activation_inc_mib"],
                "eager_useful_gbps": round(
                    (useful_bytes / (t_iso_eager["median_ms"] * 1e-3)) / 1e9, 2
                ),
                "triton_median_ms": t_iso_triton["median_ms"],
                "triton_peak_inc_mib": t_iso_triton["peak_activation_inc_mib"],
                "triton_useful_gbps": round(
                    (useful_bytes / (t_iso_triton["median_ms"] * 1e-3)) / 1e9, 2
                ),
                "speedup_vs_eager": round(
                    t_iso_eager["median_ms"] / max(t_iso_triton["median_ms"], 1e-6), 4
                ),
                "memory_reduction_vs_eager": round(
                    t_iso_eager["peak_activation_inc_mib"]
                    / max(t_iso_triton["peak_activation_inc_mib"], 1e-3),
                    4,
                ),
            },
        }

    return {
        "max_grad_rel_l2": round(max_grad_err, 6),
        "by_tokens": out_by_tokens,
    }


# ============================================================================
# 8. Suite C: 8-GPU Expert-Parallel (EP=8) MoE Training & dX/dW Overlap
# ============================================================================


def _prepare_ep_metadata(
    top_ids: torch.Tensor,
    num_experts_total: int,
    world_size: int,
    ones_i32: torch.Tensor,
    arange_tk: torch.Tensor,
) -> dict[str, Any]:
    """Prepares local-to-peer `AllToAll` splits and post-receive local-expert
    grouped-GEMM sort indices using ONE 64-int `all_to_all_single` exchange."""
    device = top_ids.device
    experts_per_rank = num_experts_total // world_size
    flat_ids = top_ids.reshape(-1)
    # Sorting by global expert_id (0..63) automatically orders by (dst_rank, local_e)!
    perm = torch.argsort(flat_ids, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm.scatter_(0, perm, arange_tk)

    hist_64 = torch.zeros(num_experts_total, device=device, dtype=torch.int64)
    hist_64.scatter_add_(0, flat_ids, ones_i32.to(torch.int64))

    recv_hist_64 = torch.empty_like(hist_64)
    dist.all_to_all_single(recv_hist_64, hist_64)

    # send_counts_by_rank: sum over experts_per_rank on CPU after single 128-int copy
    both_cpu = torch.cat([hist_64, recv_hist_64], dim=0).cpu().tolist()
    send_e = both_cpu[:num_experts_total]
    recv_e = both_cpu[num_experts_total:]

    send_splits = [
        sum(send_e[r * experts_per_rank : (r + 1) * experts_per_rank])
        for r in range(world_size)
    ]
    recv_splits = [
        sum(recv_e[r * experts_per_rank : (r + 1) * experts_per_rank])
        for r in range(world_size)
    ]
    m_recv = sum(recv_splits)

    # Received buffer has chunks [src_rank r, local_expert le] of size recv_e[r*E_loc + le].
    # Build GPU post-sort index grouping by `local_expert le` across `src_rank r`!
    recv_mat = recv_hist_64.view(world_size, experts_per_rank)
    local_hist = recv_mat.sum(dim=0).to(torch.int32)
    offs_local = torch.cumsum(local_hist, dim=0, dtype=torch.int32)

    if m_recv > 0:
        # Synthesize local expert ID for each received token in O(m_recv) on GPU
        chunk_le_ids = (
            torch.arange(experts_per_rank, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(world_size, experts_per_rank)
            .reshape(-1)
        )
        recv_chunk_counts = recv_hist_64
        recv_le = torch.repeat_interleave(
            chunk_le_ids, recv_chunk_counts, output_size=m_recv
        )
        post_perm = torch.argsort(recv_le, stable=True)
        post_inv_perm = torch.empty_like(post_perm)
        post_inv_perm.scatter_(
            0, post_perm, torch.arange(m_recv, device=device, dtype=torch.int64)
        )
    else:
        post_perm = torch.empty((0,), device=device, dtype=torch.int64)
        post_inv_perm = torch.empty((0,), device=device, dtype=torch.int64)

    return {
        "perm": perm,
        "inv_perm": inv_perm.view_as(top_ids),
        "send_splits": send_splits,
        "recv_splits": recv_splits,
        "m_recv": m_recv,
        "offs_local": offs_local,
        "post_perm": post_perm,
        "post_inv_perm": post_inv_perm,
    }


def _build_single_ep_meta_from_hist(
    top_ids: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    send_e: list[int],
    recv_e: list[int],
    recv_hist_64: torch.Tensor,
    world_size: int,
    experts_per_rank: int,
) -> dict[str, Any]:
    device = top_ids.device
    send_splits = [
        sum(send_e[r * experts_per_rank : (r + 1) * experts_per_rank])
        for r in range(world_size)
    ]
    recv_splits = [
        sum(recv_e[r * experts_per_rank : (r + 1) * experts_per_rank])
        for r in range(world_size)
    ]
    m_recv = sum(recv_splits)
    recv_mat = recv_hist_64.view(world_size, experts_per_rank)
    local_hist = recv_mat.sum(dim=0).to(torch.int32)
    offs_local = torch.cumsum(local_hist, dim=0, dtype=torch.int32)

    if m_recv > 0:
        chunk_le_ids = (
            torch.arange(experts_per_rank, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(world_size, experts_per_rank)
            .reshape(-1)
        )
        recv_le = torch.repeat_interleave(
            chunk_le_ids, recv_hist_64, output_size=m_recv
        )
        post_perm = torch.argsort(recv_le, stable=True)
        post_inv_perm = torch.empty_like(post_perm)
        post_inv_perm.scatter_(
            0, post_perm, torch.arange(m_recv, device=device, dtype=torch.int64)
        )
    else:
        post_perm = torch.empty((0,), device=device, dtype=torch.int64)
        post_inv_perm = torch.empty((0,), device=device, dtype=torch.int64)

    return {
        "perm": perm,
        "inv_perm": inv_perm.view_as(top_ids),
        "send_splits": send_splits,
        "recv_splits": recv_splits,
        "m_recv": m_recv,
        "offs_local": offs_local,
        "post_perm": post_perm,
        "post_inv_perm": post_inv_perm,
    }


def _prepare_ep_metadata_pair(
    tids0: torch.Tensor,
    tids1: torch.Tensor,
    num_experts_total: int,
    world_size: int,
    ones_half_i32: torch.Tensor,
    arange_half_tk: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Batches both microbatches' count exchanges into ONE `all_to_all_single`
    and ONE `.cpu().tolist()` host synchronization."""
    device = tids0.device
    experts_per_rank = num_experts_total // world_size
    f0 = tids0.reshape(-1)
    f1 = tids1.reshape(-1)
    perm0 = torch.argsort(f0, stable=True)
    perm1 = torch.argsort(f1, stable=True)
    inv0 = torch.empty_like(perm0)
    inv1 = torch.empty_like(perm1)
    inv0.scatter_(0, perm0, arange_half_tk)
    inv1.scatter_(0, perm1, arange_half_tk)

    ones64 = ones_half_i32.to(torch.int64)
    h0 = torch.zeros(num_experts_total, device=device, dtype=torch.int64)
    h1 = torch.zeros(num_experts_total, device=device, dtype=torch.int64)
    h0.scatter_add_(0, f0, ones64)
    h1.scatter_add_(0, f1, ones64)

    # Pack [world_size, 2, E_loc] so each destination rank receives both mb0 & mb1 counts
    send_pack = torch.stack(
        [
            h0.view(world_size, experts_per_rank),
            h1.view(world_size, experts_per_rank),
        ],
        dim=1,
    ).contiguous()
    recv_pack = torch.empty_like(send_pack)
    dist.all_to_all_single(recv_pack, send_pack)

    rh0 = recv_pack[:, 0, :].contiguous().view(-1)
    rh1 = recv_pack[:, 1, :].contiguous().view(-1)
    all_cpu = torch.cat([h0, rh0, h1, rh1], dim=0).cpu().tolist()
    n = num_experts_total
    m0 = _build_single_ep_meta_from_hist(
        tids0, perm0, inv0, all_cpu[0:n], all_cpu[n : 2 * n], rh0, world_size, experts_per_rank
    )
    m1 = _build_single_ep_meta_from_hist(
        tids1, perm1, inv1, all_cpu[2 * n : 3 * n], all_cpu[3 * n : 4 * n], rh1, world_size, experts_per_rank
    )
    return m0, m1


class DifferentiableAllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x_send: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        x_c = x_send.contiguous()
        out = torch.empty(
            (sum(output_splits), x_c.shape[1]), device=x_c.device, dtype=x_c.dtype
        )
        dist.all_to_all_single(out, x_c, output_splits, input_splits)
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        dy_c = grad_out.contiguous()
        dx = torch.empty(
            (sum(ctx.input_splits), dy_c.shape[1]),
            device=dy_c.device,
            dtype=dy_c.dtype,
        )
        dist.all_to_all_single(dx, dy_c, ctx.input_splits, ctx.output_splits)
        return dx, None, None


def run_ep8_step(
    x: torch.Tensor,
    dout: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up_local: torch.Tensor,
    w_down_local: torch.Tensor,
    skew_bias: torch.Tensor,
    top_k: int,
    schedule_mode: str,
    ones_i32: torch.Tensor,
    arange_tk: torch.Tensor,
    ones_half_i32: torch.Tensor | None = None,
    arange_half_tk: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Executes one complete forward + backward step of 8-GPU EP=8 MoE training
    under four schedules:
      - `ep_pytorch_grouped_mm_seq`
      - `ep_triton_fused_seq`
      - `ep_triton_split_dw_overlap`
      - `ep_triton_dual_microbatch_overlap`
    Returns loss and exact gradients `(dx, dw_router, dw_gate_up, dw_down)`."""
    device = x.device
    world_size = dist.get_world_size()
    num_tokens, hidden = x.shape
    num_experts_total = w_router.shape[1]
    experts_per_rank = w_gate_up_local.shape[0]

    if schedule_mode in ("ep_pytorch_grouped_mm_seq", "ep_triton_fused_seq"):
        x_g = x.detach().requires_grad_(True)
        wr_g = w_router.detach().requires_grad_(True)
        wgu_g = w_gate_up_local.detach().requires_grad_(True)
        wd_g = w_down_local.detach().requires_grad_(True)

        probs, top_ids, reg_loss, _ = router_topk_and_aux_loss(
            x_g, wr_g, skew_bias, top_k
        )
        meta = _prepare_ep_metadata(
            top_ids, num_experts_total, world_size, ones_i32, arange_tk
        )
        if schedule_mode == "ep_pytorch_grouped_mm_seq":
            x_send = eager_permute(x_g, meta["perm"], top_k)
        else:
            x_send = TritonMoEPermuteFn.apply(
                x_g, meta["perm"], meta["inv_perm"], top_k
            )

        x_recv = DifferentiableAllToAllSingle.apply(
            x_send, meta["recv_splits"], meta["send_splits"]
        )
        x_exp = x_recv.index_select(0, meta["post_perm"])

        if schedule_mode == "ep_pytorch_grouped_mm_seq":
            h1 = PyTorchGroupedMMContiguousFn.apply(x_exp, wgu_g, meta["offs_local"])
            a1 = eager_swiglu(h1)
            y_exp = PyTorchGroupedMMContiguousFn.apply(a1, wd_g, meta["offs_local"])
        else:
            h1 = TritonGroupedMMFn.apply(x_exp, wgu_g, meta["offs_local"])
            a1 = TritonSwiGLUFn.apply(h1)
            y_exp = TritonGroupedMMFn.apply(a1, wd_g, meta["offs_local"])

        y_send = y_exp.index_select(0, meta["post_inv_perm"])
        y_recv = DifferentiableAllToAllSingle.apply(
            y_send, meta["send_splits"], meta["recv_splits"]
        )
        if schedule_mode == "ep_pytorch_grouped_mm_seq":
            out = eager_weighted_combine(y_recv, meta["inv_perm"], probs)
        else:
            out = TritonWeightedCombineFn.apply(y_recv, meta["inv_perm"], probs)

        # Average router regularization loss across EP=8 ranks so global objective matches single-GPU
        loss_local = (out.float() * dout.float()).sum() + (reg_loss / float(world_size))
        loss_local.backward()
        dw_router = wr_g.grad.clone()
        dist.all_reduce(dw_router, op=dist.ReduceOp.SUM)
        return {
            "loss_local": loss_local.detach(),
            "dx": x_g.grad,
            "dw_router": dw_router,
            "dw_gate_up": wgu_g.grad,
            "dw_down": wd_g.grad,
            "m_recv": meta["m_recv"],
        }

    if schedule_mode == "ep_triton_split_dw_overlap":
        # Explicit split-dX/dW forward + overlapped backward!
        x_r = x.detach().requires_grad_(True)
        wr_r = w_router.detach().requires_grad_(True)
        probs, top_ids, reg_loss, _ = router_topk_and_aux_loss(
            x_r, wr_r, skew_bias, top_k
        )
        meta = _prepare_ep_metadata(
            top_ids, num_experts_total, world_size, ones_i32, arange_tk
        )
        with torch.no_grad():
            x_send = triton_permute_fwd_op(x, meta["perm"], top_k)
            x_recv = torch.empty(
                (meta["m_recv"], hidden), device=device, dtype=x.dtype
            )
            dist.all_to_all_single(
                x_recv, x_send, meta["recv_splits"], meta["send_splits"]
            )
            x_exp = x_recv.index_select(0, meta["post_perm"]).contiguous()
            h1 = triton_grouped_mm_fwd(x_exp, w_gate_up_local, meta["offs_local"])
            a1 = triton_swiglu_fwd_op(h1)
            y_exp = triton_grouped_mm_fwd(a1, w_down_local, meta["offs_local"])
            y_send = y_exp.index_select(0, meta["post_inv_perm"]).contiguous()
            y_recv = torch.empty_like(x_send)
            dist.all_to_all_single(
                y_recv, y_send, meta["send_splits"], meta["recv_splits"]
            )
            out = triton_weighted_combine_fwd_op(y_recv, meta["inv_perm"], probs)

            # Backward step 1: combine bwd -> router bwd + async router AllReduce
            dy_recv, dprobs = triton_weighted_combine_bwd_op(
                dout, y_recv, meta["inv_perm"], probs
            )
        router_obj = (probs * dprobs).sum() + (reg_loss / float(world_size))
        router_obj.backward()
        dw_router = wr_r.grad.contiguous()
        work_router = dist.all_reduce(dw_router, op=dist.ReduceOp.SUM, async_op=True)

        with torch.no_grad():
            # Backward step 2: combine-grad AllToAll -> dX_down -> SwiGLU_bwd -> dX_gate_up
            dy_back = torch.empty(
                (meta["m_recv"], hidden), device=device, dtype=x.dtype
            )
            dist.all_to_all_single(
                dy_back, dy_recv, meta["recv_splits"], meta["send_splits"]
            )
            dy_exp = dy_back.index_select(0, meta["post_perm"]).contiguous()

            da1 = triton_grouped_mm_dx(dy_exp, w_down_local, meta["offs_local"])
            dh1 = triton_swiglu_bwd_op(da1, h1)
            dx_exp = triton_grouped_mm_dx(dh1, w_gate_up_local, meta["offs_local"])
            dx_send_back = dx_exp.index_select(0, meta["post_inv_perm"]).contiguous()
            dx_recv_back = torch.empty_like(x_send)

            # Launch async dispatch-grad AllToAll on ncclStream BEFORE local dW GEMMs!
            work_dx = dist.all_to_all_single(
                dx_recv_back,
                dx_send_back,
                meta["send_splits"],
                meta["recv_splits"],
                async_op=True,
            )

            # Execute dW_down and dW_gate_up concurrently on default_stream!
            dw_down = triton_grouped_mm_dw(
                a1, dy_exp, meta["offs_local"], experts_per_rank
            )
            dw_gate_up = triton_grouped_mm_dw(
                x_exp, dh1, meta["offs_local"], experts_per_rank
            )

            work_dx.wait()
            work_router.wait()
            _ = dx_send_back.shape  # Keep buffer live past work_dx.wait()
            dx_routed = triton_unpermute_sum_bwd_op(
                dx_recv_back, meta["inv_perm"], num_tokens, top_k
            )
            dx_total = x_r.grad + dx_routed
            loss_local = (out.float() * dout.float()).sum() + (
                reg_loss.detach() / float(world_size)
            )

        return {
            "loss_local": loss_local,
            "dx": dx_total,
            "dw_router": dw_router,
            "dw_gate_up": dw_gate_up,
            "dw_down": dw_down,
            "m_recv": meta["m_recv"],
        }

    if schedule_mode == "ep_triton_dual_microbatch_overlap":
        assert ones_half_i32 is not None and arange_half_tk is not None
        half_t = num_tokens // 2
        x0, x1 = x[:half_t], x[half_t:]
        dout0, dout1 = dout[:half_t], dout[half_t:]

        # Router forward on full x (identical math to single-batch because AuxLoss averages over tokens;
        # compute on full x so AuxLoss f_e*p_e matches full-batch loss & gradients identically!)
        x_r = x.detach().requires_grad_(True)
        wr_r = w_router.detach().requires_grad_(True)
        probs, top_ids, reg_loss, _ = router_topk_and_aux_loss(
            x_r, wr_r, skew_bias, top_k
        )
        probs0, probs1 = probs[:half_t].contiguous(), probs[half_t:].contiguous()
        tids0, tids1 = top_ids[:half_t].contiguous(), top_ids[half_t:].contiguous()

        m0, m1 = _prepare_ep_metadata_pair(
            tids0, tids1, num_experts_total, world_size, ones_half_i32, arange_half_tk
        )

        with torch.no_grad():
            xs0 = triton_permute_fwd_op(x0, m0["perm"], top_k)
            xs1 = triton_permute_fwd_op(x1, m1["perm"], top_k)
            xr0 = torch.empty((m0["m_recv"], hidden), device=device, dtype=x.dtype)
            xr1 = torch.empty((m1["m_recv"], hidden), device=device, dtype=x.dtype)

            # Stage F1: dispatch mb0, then overlap dispatch mb1 with mb0 local experts!
            wf0 = dist.all_to_all_single(
                xr0, xs0, m0["recv_splits"], m0["send_splits"], async_op=True
            )
            wf0.wait()
            wf1 = dist.all_to_all_single(
                xr1, xs1, m1["recv_splits"], m1["send_splits"], async_op=True
            )

            xe0 = xr0.index_select(0, m0["post_perm"]).contiguous()
            h1_0 = triton_grouped_mm_fwd(xe0, w_gate_up_local, m0["offs_local"])
            a1_0 = triton_swiglu_fwd_op(h1_0)
            ye0 = triton_grouped_mm_fwd(a1_0, w_down_local, m0["offs_local"])
            ys0 = ye0.index_select(0, m0["post_inv_perm"]).contiguous()
            yr0 = torch.empty_like(xs0)

            wf1.wait()
            # Stage F2: overlap combine mb0 with mb1 local experts!
            wc0 = dist.all_to_all_single(
                yr0, ys0, m0["send_splits"], m0["recv_splits"], async_op=True
            )

            xe1 = xr1.index_select(0, m1["post_perm"]).contiguous()
            h1_1 = triton_grouped_mm_fwd(xe1, w_gate_up_local, m1["offs_local"])
            a1_1 = triton_swiglu_fwd_op(h1_1)
            ye1 = triton_grouped_mm_fwd(a1_1, w_down_local, m1["offs_local"])
            ys1 = ye1.index_select(0, m1["post_inv_perm"]).contiguous()
            yr1 = torch.empty_like(xs1)

            wc0.wait()
            # Stage F3: overlap combine mb1 with mb0 weighted unpermute + bwd combine!
            wc1 = dist.all_to_all_single(
                yr1, ys1, m1["send_splits"], m1["recv_splits"], async_op=True
            )
            out0 = triton_weighted_combine_fwd_op(yr0, m0["inv_perm"], probs0)
            dyr0, dpr0 = triton_weighted_combine_bwd_op(
                dout0, yr0, m0["inv_perm"], probs0
            )
            wc1.wait()

            out1 = triton_weighted_combine_fwd_op(yr1, m1["inv_perm"], probs1)
            # Overlap bwd combine AllToAll(mb0) with mb1 weighted_combine_bwd + router bwd!
            dyb0 = torch.empty((m0["m_recv"], hidden), device=device, dtype=x.dtype)
            wbc0 = dist.all_to_all_single(
                dyb0, dyr0, m0["recv_splits"], m0["send_splits"], async_op=True
            )
            dyr1, dpr1 = triton_weighted_combine_bwd_op(
                dout1, yr1, m1["inv_perm"], probs1
            )

        dprobs_full = torch.cat([dpr0, dpr1], dim=0)
        router_obj = (probs * dprobs_full).sum() + (reg_loss / float(world_size))
        router_obj.backward()
        dw_router = wr_r.grad.contiguous()
        wr_allred = dist.all_reduce(dw_router, op=dist.ReduceOp.SUM, async_op=True)

        with torch.no_grad():
            wbc0.wait()
            # Overlap bwd combine AllToAll(mb1) with mb0 dX!
            dyb1 = torch.empty((m1["m_recv"], hidden), device=device, dtype=x.dtype)
            wbc1 = dist.all_to_all_single(
                dyb1, dyr1, m1["recv_splits"], m1["send_splits"], async_op=True
            )

            dye0 = dyb0.index_select(0, m0["post_perm"]).contiguous()
            da1_0 = triton_grouped_mm_dx(dye0, w_down_local, m0["offs_local"])
            dh1_0 = triton_swiglu_bwd_op(da1_0, h1_0)
            dxe0 = triton_grouped_mm_dx(dh1_0, w_gate_up_local, m0["offs_local"])
            dxs0 = dxe0.index_select(0, m0["post_inv_perm"]).contiguous()
            dxr0 = torch.empty_like(xs0)

            wbc1.wait()
            # Overlap bwd dispatch AllToAll(mb0) with mb1 dX!
            wbd0 = dist.all_to_all_single(
                dxr0, dxs0, m0["send_splits"], m0["recv_splits"], async_op=True
            )

            dye1 = dyb1.index_select(0, m1["post_perm"]).contiguous()
            da1_1 = triton_grouped_mm_dx(dye1, w_down_local, m1["offs_local"])
            dh1_1 = triton_swiglu_bwd_op(da1_1, h1_1)
            dxe1 = triton_grouped_mm_dx(dh1_1, w_gate_up_local, m1["offs_local"])
            dxs1 = dxe1.index_select(0, m1["post_inv_perm"]).contiguous()
            dxr1 = torch.empty_like(xs1)

            wbd0.wait()
            # Overlap bwd dispatch AllToAll(mb1) with BOTH mb0 & mb1 dW + mb0 unpermute_sum!
            wbd1 = dist.all_to_all_single(
                dxr1, dxs1, m1["send_splits"], m1["recv_splits"], async_op=True
            )

            dw_down = triton_grouped_mm_dw(
                a1_0, dye0, m0["offs_local"], experts_per_rank
            )
            triton_grouped_mm_dw(
                a1_1,
                dye1,
                m1["offs_local"],
                experts_per_rank,
                out=dw_down,
                accumulate=True,
            )
            dw_gate_up = triton_grouped_mm_dw(
                xe0, dh1_0, m0["offs_local"], experts_per_rank
            )
            triton_grouped_mm_dw(
                xe1,
                dh1_1,
                m1["offs_local"],
                experts_per_rank,
                out=dw_gate_up,
                accumulate=True,
            )
            dx_routed_0 = triton_unpermute_sum_bwd_op(
                dxr0, m0["inv_perm"], half_t, top_k
            )

            wbd1.wait()
            wr_allred.wait()
            _ = (xs0.shape, xs1.shape, ys0.shape, ys1.shape, dxs0.shape, dxs1.shape)
            dx_routed_1 = triton_unpermute_sum_bwd_op(
                dxr1, m1["inv_perm"], half_t, top_k
            )
            dx_total = x_r.grad + torch.cat([dx_routed_0, dx_routed_1], dim=0)
            out_full = torch.cat([out0, out1], dim=0)
            loss_local = (out_full.float() * dout.float()).sum() + (
                reg_loss.detach() / float(world_size)
            )

        return {
            "loss_local": loss_local,
            "dx": dx_total,
            "dw_router": dw_router,
            "dw_gate_up": dw_gate_up,
            "dw_down": dw_down,
            "m_recv": m0["m_recv"] + m1["m_recv"],
        }

    raise ValueError(f"Unknown EP schedule_mode: {schedule_mode}")


def run_ep8_static_capacity_step(
    x: torch.Tensor,
    dout: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up_local: torch.Tensor,
    w_down_local: torch.Tensor,
    skew_bias: torch.Tensor,
    top_k: int,
    capacity_factor: float,
) -> dict[str, Any]:
    """Static-capacity equal-split `AllToAll` EP=8 training step (zero count
    exchange, fixed `[world_size, E_loc, cap, H]` tensor shapes, dropping tokens
    whose per-(src_rank, global_expert) slot index >= cap)."""
    device = x.device
    world_size = dist.get_world_size()
    num_tokens, hidden = x.shape
    num_experts_total = w_router.shape[1]
    experts_per_rank = w_gate_up_local.shape[0]
    m_tot = num_tokens * top_k
    cap = max(
        4, int(math.ceil(capacity_factor * float(m_tot) / float(num_experts_total)))
    )

    x_g = x.detach().requires_grad_(True)
    wr_g = w_router.detach().requires_grad_(True)
    wgu_g = w_gate_up_local.detach().requires_grad_(True)
    wd_g = w_down_local.detach().requires_grad_(True)

    probs, top_ids, reg_loss, _ = router_topk_and_aux_loss(x_g, wr_g, skew_bias, top_k)
    flat_ids = top_ids.reshape(-1)
    # Prioritize higher-probability assignments first so top-1 survives capacity clipping
    prob_order = torch.argsort(probs.reshape(-1), descending=True, stable=True)
    ordered_e = flat_ids.index_select(0, prob_order)
    perm_e = torch.argsort(ordered_e, stable=True)
    sorted_orig_slot = prob_order.index_select(0, perm_e)
    sorted_e = ordered_e.index_select(0, perm_e)

    ones_i = torch.ones(m_tot, device=device, dtype=torch.int32)
    hist_64 = torch.zeros(num_experts_total, device=device, dtype=torch.int32)
    hist_64.scatter_add_(0, sorted_e, ones_i)
    offs_64 = torch.cumsum(hist_64, dim=0, dtype=torch.int32)
    prev_offs = torch.zeros_like(offs_64)
    prev_offs[1:] = offs_64[:-1]
    pos_in_e = (
        torch.arange(m_tot, device=device, dtype=torch.int64)
        - prev_offs.index_select(0, sorted_e).to(torch.int64)
    )
    kept_mask = pos_in_e < cap
    dropped_local = int((~kept_mask).sum().item())

    clamped_pos = torch.clamp(pos_in_e, max=cap - 1)
    buf_idx = sorted_e * cap + clamped_pos
    tok_idx = sorted_orig_slot // top_k
    x_gathered = x_g.index_select(0, tok_idx) * kept_mask.unsqueeze(-1).to(x.dtype)

    send_buf = torch.zeros(
        (num_experts_total * cap, hidden), device=device, dtype=x.dtype
    )
    # Keep only valid slots via masked scatter_add_ so overflow slots do not overwrite slot cap-1
    valid_idx = torch.where(kept_mask, buf_idx, torch.zeros_like(buf_idx))
    # Add 1 extra dummy slot per expert? Simpler: allocate cap+1 and slice [:cap]!
    send_buf_pad = torch.zeros(
        (num_experts_total, cap + 1, hidden), device=device, dtype=x.dtype
    )
    clamped_pad = torch.where(
        kept_mask, pos_in_e, torch.full_like(pos_in_e, cap)
    )
    pad_idx = sorted_e * (cap + 1) + clamped_pad
    send_buf_pad_flat = send_buf_pad.view(-1, hidden).index_put(
        (pad_idx,), x_gathered, accumulate=False
    )
    send_equal = (
        send_buf_pad_flat.view(world_size, experts_per_rank, cap + 1, hidden)[
            :, :, :cap, :
        ]
        .contiguous()
        .view(world_size * experts_per_rank * cap, hidden)
    )

    eq_splits = [experts_per_rank * cap] * world_size
    recv_equal = DifferentiableAllToAllSingle.apply(
        send_equal, eq_splits, eq_splits
    )
    # Transpose (src_rank, local_e, cap, H) -> (local_e, src_rank * cap, H)
    x_exp = (
        recv_equal.view(world_size, experts_per_rank, cap, hidden)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(experts_per_rank * world_size * cap, hidden)
    )
    offs_static = (
        torch.arange(1, experts_per_rank + 1, device=device, dtype=torch.int32)
        * (world_size * cap)
    )
    h1 = TritonGroupedMMFn.apply(x_exp, wgu_g, offs_static)
    a1 = TritonSwiGLUFn.apply(h1)
    y_exp = TritonGroupedMMFn.apply(a1, wd_g, offs_static)

    y_send = (
        y_exp.view(experts_per_rank, world_size, cap, hidden)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(world_size * experts_per_rank * cap, hidden)
    )
    y_recv = DifferentiableAllToAllSingle.apply(
        y_send, eq_splits, eq_splits
    )
    # Append zero sink row at slot `cap` per expert so dropped tokens read exact zeros
    y_recv_3d = y_recv.view(num_experts_total, cap, hidden)
    zero_sink = torch.zeros((num_experts_total, 1, hidden), device=device, dtype=x.dtype)
    y_with_sink = torch.cat([y_recv_3d, zero_sink], dim=1).view(-1, hidden)
    y_sorted = y_with_sink.index_select(0, pad_idx)

    # Scatter back to original `(tok, k)` order
    inv_orig = torch.empty_like(sorted_orig_slot)
    inv_orig.scatter_(
        0, sorted_orig_slot, torch.arange(m_tot, device=device, dtype=torch.int64)
    )
    out = TritonWeightedCombineFn.apply(y_sorted, inv_orig.view_as(top_ids), probs)

    loss_local = (out.float() * dout.float()).sum() + (reg_loss / float(world_size))
    loss_local.backward()
    dw_router = wr_g.grad.clone()
    dist.all_reduce(dw_router, op=dist.ReduceOp.SUM)
    return {
        "loss_local": loss_local.detach(),
        "dx": x_g.grad,
        "dw_router": dw_router,
        "dw_gate_up": wgu_g.grad,
        "dw_down": wd_g.grad,
        "dropped_local": dropped_local,
        "total_local_assignments": m_tot,
        "cap_per_expert": cap,
    }


def run_suite_c_ep8(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    hidden = 512 if is_smoke else 4096
    inter_dim = 192 if is_smoke else 1536
    num_e_total = 16 if is_smoke else 64
    e_local = num_e_total // world_size
    top_k = 4 if is_smoke else 8
    t_local_list = [64, 128] if is_smoke else [512, 1024, 2048]
    warmup = 2 if is_smoke else 6
    iters = 5 if is_smoke else 20

    # 1. Exact EP=8 vs Single-GPU 64-Expert Reference Verification
    t_ver = t_local_list[0]
    t_glob_ver = t_ver * world_size
    x_all, dout_all, wr_shared, wgu_all, wd_all = make_init_tensors(
        t_glob_ver, hidden, num_e_total, inter_dim, device, seed=11780
    )
    x_loc = x_all[rank * t_ver : (rank + 1) * t_ver].contiguous()
    dout_loc = dout_all[rank * t_ver : (rank + 1) * t_ver].contiguous()
    wgu_loc = wgu_all[rank * e_local : (rank + 1) * e_local].contiguous()
    wd_loc = wd_all[rank * e_local : (rank + 1) * e_local].contiguous()
    skew_zipf = make_skew_bias(num_e_total, "zipf_s1p1", device)

    # Note: In EP=8, Switch AuxLoss computed per-rank uses local `f_e * p_e`;
    # for exact mathematical verification against single-GPU, compare against `ep_pytorch_grouped_mm_seq`
    # AND verify under uniform/exact loss that all 4 EP schedules match bit-closely!
    ones_ver = torch.ones(t_ver * top_k, device=device, dtype=torch.int32)
    arange_ver = torch.arange(t_ver * top_k, device=device, dtype=torch.int64)
    ones_h_ver = torch.ones((t_ver // 2) * top_k, device=device, dtype=torch.int32)
    arange_h_ver = torch.arange((t_ver // 2) * top_k, device=device, dtype=torch.int64)

    ref_ep = run_ep8_step(
        x_loc,
        dout_loc,
        wr_shared,
        wgu_loc,
        wd_loc,
        skew_zipf,
        top_k,
        "ep_pytorch_grouped_mm_seq",
        ones_ver,
        arange_ver,
    )
    loss_ref_sum = ref_ep["loss_local"].clone()
    dist.all_reduce(loss_ref_sum, op=dist.ReduceOp.SUM)

    # Also verify pure forward+backward main loss + z_loss (aux_coef=0) against true 1-GPU 64-expert run!
    x_1g = x_all.detach().clone().requires_grad_(True)
    wr_1g = wr_shared.detach().clone().requires_grad_(True)
    wgu_1g = wgu_all.detach().clone().requires_grad_(True)
    wd_1g = wd_all.detach().clone().requires_grad_(True)
    probs_1g, tids_1g, reg_1g, _ = router_topk_and_aux_loss(
        x_1g, wr_1g, skew_zipf, top_k, aux_coef=0.0, z_coef=1e-3
    )
    ones_1g = torch.ones(t_glob_ver * top_k, device=device, dtype=torch.int32)
    arange_1g = torch.arange(t_glob_ver * top_k, device=device, dtype=torch.int64)
    p1g, ip1g, _, offs_1g = gpu_permutation_indices(
        tids_1g, num_e_total, ones_1g, arange_1g
    )
    xp_1g = TritonMoEPermuteFn.apply(x_1g, p1g, ip1g, top_k)
    h1_1g = TritonGroupedMMFn.apply(xp_1g, wgu_1g, offs_1g)
    a1_1g = TritonSwiGLUFn.apply(h1_1g)
    yp_1g = TritonGroupedMMFn.apply(a1_1g, wd_1g, offs_1g)
    out_1g = TritonWeightedCombineFn.apply(yp_1g, ip1g, probs_1g)
    loss_1g = (out_1g.float() * dout_all.float()).sum() + reg_1g
    loss_1g.backward()

    # Run EP=8 with aux_coef=0 to check exact global 1-GPU equivalence!
    x_ep0 = x_loc.detach().clone().requires_grad_(True)
    wr_ep0 = wr_shared.detach().clone().requires_grad_(True)
    wgu_ep0 = wgu_loc.detach().clone().requires_grad_(True)
    wd_ep0 = wd_loc.detach().clone().requires_grad_(True)
    pr_ep0, ti_ep0, rg_ep0, _ = router_topk_and_aux_loss(
        x_ep0, wr_ep0, skew_zipf, top_k, aux_coef=0.0, z_coef=1e-3
    )
    m_ep0 = _prepare_ep_metadata(
        ti_ep0, num_e_total, world_size, ones_ver, arange_ver
    )
    xs_ep0 = TritonMoEPermuteFn.apply(x_ep0, m_ep0["perm"], m_ep0["inv_perm"], top_k)
    xr_ep0 = DifferentiableAllToAllSingle.apply(
        xs_ep0, m_ep0["recv_splits"], m_ep0["send_splits"]
    )
    xe_ep0 = xr_ep0.index_select(0, m_ep0["post_perm"])
    he_ep0 = TritonGroupedMMFn.apply(xe_ep0, wgu_ep0, m_ep0["offs_local"])
    ae_ep0 = TritonSwiGLUFn.apply(he_ep0)
    ye_ep0 = TritonGroupedMMFn.apply(ae_ep0, wd_ep0, m_ep0["offs_local"])
    ys_ep0 = ye_ep0.index_select(0, m_ep0["post_inv_perm"])
    yr_ep0 = DifferentiableAllToAllSingle.apply(
        ys_ep0, m_ep0["send_splits"], m_ep0["recv_splits"]
    )
    out_ep0 = TritonWeightedCombineFn.apply(yr_ep0, m_ep0["inv_perm"], pr_ep0)
    l_ep0 = (out_ep0.float() * dout_loc.float()).sum() + (rg_ep0 / float(world_size))
    l_ep0.backward()
    dwr_ep0 = wr_ep0.grad.clone()
    dist.all_reduce(dwr_ep0, op=dist.ReduceOp.SUM)
    l_ep0_sum = l_ep0.detach().clone()
    dist.all_reduce(l_ep0_sum, op=dist.ReduceOp.SUM)

    ep_1g_scale = max(
        abs(float(loss_1g.item())),
        out_1g.float().norm().item() * dout_all.float().norm().item(),
        1e-6,
    )
    ep_loss_scale = max(
        abs(float(loss_ref_sum.item())),
        out_1g.float().norm().item() * dout_all.float().norm().item(),
        1e-6,
    )
    ep_vs_1gpu_err = torch.tensor(
        [
            abs(float(l_ep0_sum.item()) - float(loss_1g.item())) / ep_1g_scale,
            rel_l2_error(
                x_ep0.grad, x_1g.grad[rank * t_ver : (rank + 1) * t_ver]
            ),
            rel_l2_error(
                wgu_ep0.grad, wgu_1g.grad[rank * e_local : (rank + 1) * e_local]
            ),
            rel_l2_error(
                wd_ep0.grad, wd_1g.grad[rank * e_local : (rank + 1) * e_local]
            ),
            rel_l2_error(dwr_ep0, wr_1g.grad),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(ep_vs_1gpu_err, op=dist.ReduceOp.MAX)

    # Check all 4 EP schedules against ref_ep (`ep_pytorch_grouped_mm_seq`)
    schedule_parity: dict[str, Any] = {}
    max_ep_sched_grad_err = 0.0
    max_ep_sched_loss_err = 0.0
    for sched in (
        "ep_triton_fused_seq",
        "ep_triton_split_dw_overlap",
        "ep_triton_dual_microbatch_overlap",
    ):
        st = run_ep8_step(
            x_loc,
            dout_loc,
            wr_shared,
            wgu_loc,
            wd_loc,
            skew_zipf,
            top_k,
            sched,
            ones_ver,
            arange_ver,
            ones_h_ver,
            arange_h_ver,
        )
        ls = st["loss_local"].clone()
        dist.all_reduce(ls, op=dist.ReduceOp.SUM)
        err_vec = torch.tensor(
            [
                abs(float(ls.item()) - float(loss_ref_sum.item())) / ep_loss_scale,
                rel_l2_error(st["dx"], ref_ep["dx"]),
                rel_l2_error(st["dw_gate_up"], ref_ep["dw_gate_up"]),
                rel_l2_error(st["dw_down"], ref_ep["dw_down"]),
                rel_l2_error(st["dw_router"], ref_ep["dw_router"]),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(err_vec, op=dist.ReduceOp.MAX)
        l_e, dx_e, dw1_e, dw2_e, dwr_e = [float(v) for v in err_vec.tolist()]
        max_ep_sched_loss_err = max(max_ep_sched_loss_err, l_e)
        max_ep_sched_grad_err = max(max_ep_sched_grad_err, dx_e, dw1_e, dw2_e, dwr_e)
        schedule_parity[sched] = {
            "rel_loss_err": round(l_e, 7),
            "dx_rel_l2": round(dx_e, 6),
            "dw_gate_up_rel_l2": round(dw1_e, 6),
            "dw_down_rel_l2": round(dw2_e, 6),
            "dw_router_rel_l2": round(dwr_e, 6),
        }

    # 2. Part C.1: EP=8 Schedule & Overlap Sweep across T_local
    scaling_by_t: dict[str, Any] = {}
    for t_loc in t_local_list:
        t_glob = t_loc * world_size
        xa, da, wr_s, wgua, wda = make_init_tensors(
            t_glob, hidden, num_e_total, inter_dim, device, seed=11790 + t_loc
        )
        xl = xa[rank * t_loc : (rank + 1) * t_loc].contiguous()
        dl = da[rank * t_loc : (rank + 1) * t_loc].contiguous()
        wgul = wgua[rank * e_local : (rank + 1) * e_local].contiguous()
        wdl = wda[rank * e_local : (rank + 1) * e_local].contiguous()
        del xa, da, wgua, wda

        ones_t = torch.ones(t_loc * top_k, device=device, dtype=torch.int32)
        arange_t = torch.arange(t_loc * top_k, device=device, dtype=torch.int64)
        ones_ht = torch.ones((t_loc // 2) * top_k, device=device, dtype=torch.int32)
        arange_ht = torch.arange((t_loc // 2) * top_k, device=device, dtype=torch.int64)

        cluster_flops = moe_layer_training_flops(
            t_glob, hidden, num_e_total, top_k, inter_dim
        )
        sched_res: dict[str, Any] = {}
        for sched in (
            "ep_pytorch_grouped_mm_seq",
            "ep_triton_fused_seq",
            "ep_triton_split_dw_overlap",
            "ep_triton_dual_microbatch_overlap",
        ):
            for _ in range(warmup):
                _ = run_ep8_step(
                    xl,
                    dl,
                    wr_s,
                    wgul,
                    wdl,
                    skew_zipf,
                    top_k,
                    sched,
                    ones_t,
                    arange_t,
                    ones_ht,
                    arange_ht,
                )
            dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            base_m = torch.cuda.memory_allocated(device)
            st_once = run_ep8_step(
                xl,
                dl,
                wr_s,
                wgul,
                wdl,
                skew_zipf,
                top_k,
                sched,
                ones_t,
                arange_t,
                ones_ht,
                arange_ht,
            )
            torch.cuda.synchronize(device)
            inc_mib = max(
                0.0, (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0**2)
            )
            mem_vec = torch.tensor(
                [inc_mib, float(st_once["m_recv"])], device=device, dtype=torch.float64
            )
            dist.all_reduce(mem_vec, op=dist.ReduceOp.MAX)
            max_rank_inc_mib = float(mem_vec[0].item())
            max_rank_m_recv = int(mem_vec[1].item())

            samples: list[float] = []
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                dist.barrier()
                s_ev.record()
                _ = run_ep8_step(
                    xl,
                    dl,
                    wr_s,
                    wgul,
                    wdl,
                    skew_zipf,
                    top_k,
                    sched,
                    ones_t,
                    arange_t,
                    ones_ht,
                    arange_ht,
                )
                e_ev.record()
                torch.cuda.synchronize(device)
                dur = torch.tensor(
                    [float(s_ev.elapsed_time(e_ev))],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(dur, op=dist.ReduceOp.MAX)
                samples.append(float(dur.item()))

            ss = sorted(samples)
            med = statistics.median(ss)
            iqr = ss[(3 * (len(ss) - 1)) // 4] - ss[(len(ss) - 1) // 4]
            sched_res[sched] = {
                "step_ms_median": round(med, 4),
                "step_ms_iqr": round(iqr, 4),
                "cluster_tflops": round((cluster_flops / (med * 1e-3)) / 1e12, 2),
                "max_rank_activation_inc_mib": round(max_rank_inc_mib, 2),
                "max_rank_m_recv": max_rank_m_recv,
                "receiver_imbalance_max_over_mean": round(
                    float(max_rank_m_recv) / float(t_loc * top_k), 3
                ),
            }

        scaling_by_t[str(t_loc)] = {
            "tokens_per_rank": t_loc,
            "global_tokens": t_glob,
            "global_routed_assignments": t_glob * top_k,
            "schedules": sched_res,
        }

    # 3. Part C.2: Routing Skew & Capacity-Factor Gradient Fidelity Frontier
    t_cap = t_local_list[-1] if is_smoke else 1024
    t_glob_cap = t_cap * world_size
    xa_c, da_c, wr_c, wgua_c, wda_c = make_init_tensors(
        t_glob_cap, hidden, num_e_total, inter_dim, device, seed=11805
    )
    xl_c = xa_c[rank * t_cap : (rank + 1) * t_cap].contiguous()
    dl_c = da_c[rank * t_cap : (rank + 1) * t_cap].contiguous()
    wgul_c = wgua_c[rank * e_local : (rank + 1) * e_local].contiguous()
    wdl_c = wda_c[rank * e_local : (rank + 1) * e_local].contiguous()
    del xa_c, da_c, wgua_c, wda_c

    ones_c = torch.ones(t_cap * top_k, device=device, dtype=torch.int32)
    arange_c = torch.arange(t_cap * top_k, device=device, dtype=torch.int64)
    skew_capacity_frontier: dict[str, Any] = {}

    for skew_name in ("uniform", "zipf_s1p2", "hot_rank0_60pct"):
        sb_c = make_skew_bias(num_e_total, skew_name, device)
        dropless_ref = run_ep8_step(
            xl_c,
            dl_c,
            wr_c,
            wgul_c,
            wdl_c,
            sb_c,
            top_k,
            "ep_triton_split_dw_overlap",
            ones_c,
            arange_c,
        )
        l_ref_c = dropless_ref["loss_local"].clone()
        dist.all_reduce(l_ref_c, op=dist.ReduceOp.SUM)
        m_recv_vec = torch.tensor(
            [float(dropless_ref["m_recv"])], device=device, dtype=torch.float64
        )
        dist.all_reduce(m_recv_vec, op=dist.ReduceOp.MAX)
        max_recv_dropless = int(m_recv_vec.item())

        # Time dropless dynamic split_dw
        for _ in range(warmup):
            _ = run_ep8_step(
                xl_c,
                dl_c,
                wr_c,
                wgul_c,
                wdl_c,
                sb_c,
                top_k,
                "ep_triton_split_dw_overlap",
                ones_c,
                arange_c,
            )
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        d_samples: list[float] = []
        for _ in range(iters):
            dist.barrier()
            s_ev.record()
            _ = run_ep8_step(
                xl_c,
                dl_c,
                wr_c,
                wgul_c,
                wdl_c,
                sb_c,
                top_k,
                "ep_triton_split_dw_overlap",
                ones_c,
                arange_c,
            )
            e_ev.record()
            torch.cuda.synchronize(device)
            dv = torch.tensor(
                [float(s_ev.elapsed_time(e_ev))], device=device, dtype=torch.float64
            )
            dist.all_reduce(dv, op=dist.ReduceOp.MAX)
            d_samples.append(float(dv.item()))

        cap_modes: dict[str, Any] = {
            "dropless_dynamic_split_dw": {
                "step_ms_median": round(statistics.median(d_samples), 4),
                "dropped_fraction": 0.0,
                "rel_loss_err_vs_dropless": 0.0,
                "dx_cosine_vs_dropless": 1.0,
                "dw_down_cosine_vs_dropless": 1.0,
            }
        }

        for cf_label, cf_val in (
            ("static_cf_1p0", 1.0),
            ("static_cf_1p25", 1.25),
            ("static_cf_2p0", 2.0),
        ):
            st_cf = run_ep8_static_capacity_step(
                xl_c, dl_c, wr_c, wgul_c, wdl_c, sb_c, top_k, cf_val
            )
            l_cf = st_cf["loss_local"].clone()
            dist.all_reduce(l_cf, op=dist.ReduceOp.SUM)
            drop_sum = torch.tensor(
                [float(st_cf["dropped_local"]), float(st_cf["total_local_assignments"])],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(drop_sum, op=dist.ReduceOp.SUM)
            dropped_frac = float(drop_sum[0].item() / max(drop_sum[1].item(), 1.0))

            # Gather cosine similarities (mean across ranks)
            dx_cos = cosine_sim(st_cf["dx"], dropless_ref["dx"])
            dw_cos = cosine_sim(st_cf["dw_down"], dropless_ref["dw_down"])
            cos_vec = torch.tensor(
                [dx_cos, dw_cos], device=device, dtype=torch.float64
            )
            dist.all_reduce(cos_vec, op=dist.ReduceOp.SUM)
            cos_vec /= float(world_size)

            for _ in range(warmup):
                _ = run_ep8_static_capacity_step(
                    xl_c, dl_c, wr_c, wgul_c, wdl_c, sb_c, top_k, cf_val
                )
            cf_samples: list[float] = []
            for _ in range(iters):
                dist.barrier()
                s_ev.record()
                _ = run_ep8_static_capacity_step(
                    xl_c, dl_c, wr_c, wgul_c, wdl_c, sb_c, top_k, cf_val
                )
                e_ev.record()
                torch.cuda.synchronize(device)
                dv = torch.tensor(
                    [float(s_ev.elapsed_time(e_ev))], device=device, dtype=torch.float64
                )
                dist.all_reduce(dv, op=dist.ReduceOp.MAX)
                cf_samples.append(float(dv.item()))

            l_rel_err = abs(float(l_cf.item()) - float(l_ref_c.item())) / max(
                abs(float(l_ref_c.item())), 1e-8
            )
            cap_modes[cf_label] = {
                "capacity_factor": cf_val,
                "cap_per_expert": st_cf["cap_per_expert"],
                "step_ms_median": round(statistics.median(cf_samples), 4),
                "dropped_fraction": round(dropped_frac, 5),
                "rel_loss_err_vs_dropless": round(l_rel_err, 6),
                "dx_cosine_vs_dropless": round(float(cos_vec[0].item()), 6),
                "dw_down_cosine_vs_dropless": round(float(cos_vec[1].item()), 6),
            }

        skew_capacity_frontier[skew_name] = {
            "tokens_per_rank": t_cap,
            "dropless_max_rank_m_recv": max_recv_dropless,
            "dropless_receiver_imbalance": round(
                float(max_recv_dropless) / float(t_cap * top_k), 3
            ),
            "modes": cap_modes,
        }

    return {
        "ep8_vs_single_gpu_reference": {
            "rel_loss_err": round(float(ep_vs_1gpu_err[0].item()), 7),
            "max_dx_rel_l2": round(float(ep_vs_1gpu_err[1].item()), 6),
            "max_dw_gate_up_rel_l2": round(float(ep_vs_1gpu_err[2].item()), 6),
            "max_dw_down_rel_l2": round(float(ep_vs_1gpu_err[3].item()), 6),
            "dw_router_rel_l2": round(float(ep_vs_1gpu_err[4].item()), 6),
        },
        "schedule_numerical_parity": schedule_parity,
        "max_schedule_rel_loss_err": round(max_ep_sched_loss_err, 7),
        "max_schedule_grad_rel_l2": round(max_ep_sched_grad_err, 6),
        "scaling_by_tokens_per_rank": scaling_by_t,
        "skew_and_capacity_frontier": skew_capacity_frontier,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E117: Fine-Grained MoE Training Kernels & 8-GPU EP Overlap"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(minutes=30)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"Expected 8 GPUs for EP=8 study, got {world_size}"

    t0 = time.perf_counter()
    is_smoke = args.mode == "smoke"

    s0_res: dict[str, Any] = {}
    sa_res: dict[str, Any] = {}
    sb_res: dict[str, Any] = {}

    if rank == 0:
        s0_res = run_suite_0_audit(device, is_smoke)
        print(
            f"[E117] Suite 0 complete at +{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    dist.barrier()

    if rank == 0:
        sa_res = run_suite_a_granularity(device, is_smoke)
        print(
            f"[E117] Suite A complete at +{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    dist.barrier()

    if rank == 0:
        sb_res = run_suite_b_dispatcher(device, is_smoke)
        print(
            f"[E117] Suite B complete at +{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    dist.barrier()

    sc_res = run_suite_c_ep8(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = time.perf_counter() - t0

    if rank == 0:
        print(
            f"[E117] Suite C complete at +{elapsed_s:.1f}s",
            flush=True,
        )
        payload = {
            "experiment": "E117_moe_training_grouped_gemm_and_ep_overlap",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": round(elapsed_s, 2),
            "environment": environment_metadata(),
            "suite_0_grouped_mm_audit": s0_res,
            "suite_a_granularity_frontier": sa_res,
            "suite_b_fused_dispatcher": sb_res,
            "suite_c_ep8_overlap_and_skew": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
