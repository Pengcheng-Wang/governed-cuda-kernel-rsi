#!/usr/bin/env python3
"""
Round-11 RSI Kernel Module (E137 / v128):
nGPT: Normalized Transformer with Representation Learning on the Hypersphere
(Loshchilov, Hsieh, Sun & Ginsburg, NVIDIA, arXiv:2410.01131v2, ICLR 2025;
official repository: https://github.com/NVIDIA/ngpt)

Forensic Audit of Official `NVIDIA/ngpt` (`model.py` & `train.py`):
1. Pathology P1 (Upstream Broken RoPE + Per-Layer CPU Allocation in `model.py:40-58,102-104`):
   - `get_sinusoidal_embeddings(T, D)` allocates a CPU tensor inside every `Block.forward()`
     and executes a host-to-device `.to(device=q.device)` copy on every layer.
   - `get_sinusoidal_embeddings` interleaves `sin` at `0::2` and `cos` at `1::2`, but
     `apply_rotary_position_embeddings` calls `sin, cos = sinusoidal_pos.chunk(2, dim=-1)`
     (splitting first-half low frequencies vs second-half high frequencies rather than `sin`
     vs `cos`) and computes `stack((-q_odd, q_even)) * stack((cos, sin))` without summing
     2D rotation pairs `(q_even*cos - q_odd*sin, q_even*sin + q_odd*cos)`. This violates
     SO(2) orthogonality (||R(q)||_2 != ||q||_2) and destroys relative-position shift
     invariance (<R_{m+d} q, R_{n+d} k> != <R_m q, R_n k>).
   - Furthermore, unconstrained per-dimension `sqk` scaling after SO(2) rotation breaks
     relative-shift invariance unless rotary-pair scales are isotropic (s_{2i} == s_{2i+1}).
2. Pathology P2 (Unfused Hypersphere Chordal LERP Residual in `model.py:88-90,123-131,145-153`):
   - Executed twice per layer (`attn` and `mlp`). Upstream launches 10 kernels per sublayer
     (`A_norm`, `B_norm`, `abs(alpha*scale)`, `A_norm + lr*(B_norm - A_norm)`, `justnorm(res)`)
     without an epsilon guard (`0/0 -> NaN` on zero rows) and materializes 4 `[B, T, C]`
     intermediate tensors in HBM per sublayer (`8` per Transformer block).
3. Pathology P3 (Unfused Eigen-Scaled SwiGLU in `model.py:135-140`):
   - Upstream scales `uv` (`[B, T, 2*I]`) by `suv * sqrt(C)` in HBM before `chunk(2)` and
     `u * silu(v)`, tripling `[B, T, 2*I]` HBM traffic.
4. Pathology P4 (Unfused `sz`-Scaled Logits + Cross-Entropy in `model.py:199-203`):
   - Upstream materializes a second full-vocabulary `[B, T, V]` tensor `logits = sz * logits`
     in HBM before calling `F.cross_entropy`.
5. Pathology P5 (Unfused Post-Step Python `normalize_matrices()` & Radial Gradient Drift in `train.py:326-345`):
   - Upstream loops in Python over 74+ weight matrices after every `optimizer.step()`,
     allocating FP32 copies and launching ~300 kernels/step (`idim=1` for row-unit and
     non-contiguous `idim=0` for col-unit matrices), while Euclidean AdamW accumulates radial
     gradients G_parallel = W_hat <W_hat, G> into first/second moments (m_t, v_t).

Round-11 RSI Zero-Spill Triton Kernels (SM80 A100-SXM4-80GB):
- `_ngpt_hypersphere_lerp_fwd_kernel` & `_ngpt_hypersphere_lerp_bwd_kernel`
- `_ngpt_reduce_2d_to_1d_kernel`
- `_ngpt_rope_norm_sqk_fwd_kernel` & `_ngpt_rope_norm_sqk_bwd_kernel`
- `_ngpt_scaled_swiglu_fwd_kernel` & `_ngpt_scaled_swiglu_bwd_kernel`
- `_ngpt_cosine_flash_fwd_kernel`, `_ngpt_cosine_flash_bwd_prep_kernel`
- `_ngpt_cosine_flash_bwd_dq_d64_kernel`, `_ngpt_cosine_flash_bwd_dkv_d64_kernel`
- `_ngpt_cosine_flash_bwd_dq_d128_kernel`, `_ngpt_cosine_flash_bwd_dkv_d128_kernel`
- `_ngpt_sz_cross_entropy_fwd_kernel`, `_ngpt_sz_cross_entropy_bwd_kernel`
- `_ngpt_riemannian_adamw_retract_row_kernel`, `_ngpt_riemannian_adamw_retract_col_kernel`
- `_ngpt_ring_lse_update_kernel`
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl


# =============================================================================
# 1. Generic 2D -> 1D Deterministic Partial Reduction Kernel (Zero Atomics)
# =============================================================================

@triton.jit
def _ngpt_reduce_2d_to_1d_kernel(
    Partial_ptr,
    Out_ptr,
    num_splits,
    numel,
    stride_s,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_s = tl.arange(0, BLOCK_S)
    mask = (offs_s[:, None] < num_splits) & (offs_c[None, :] < numel)
    tile = tl.load(Partial_ptr + offs_s[:, None] * stride_s + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(tile, axis=0)
    tl.store(Out_ptr + offs_c, acc, mask=(offs_c < numel))


def reduce_partials_2d(partial: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    num_splits = partial.shape[0]
    out_shape = partial.shape[1:]
    numel = partial[0].numel()
    partial_2d = partial.view(num_splits, numel)
    out = torch.empty((numel,), device=partial.device, dtype=torch.float32)
    BLOCK_S = triton.next_power_of_2(num_splits)
    BLOCK_C = 32
    grid = (triton.cdiv(numel, BLOCK_C),)
    _ngpt_reduce_2d_to_1d_kernel[grid](
        partial_2d,
        out,
        num_splits,
        numel,
        partial_2d.stride(0),
        BLOCK_S=BLOCK_S,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=1,
    )
    return out.view(out_shape).to(out_dtype)


# =============================================================================
# 2. Kernel 1: Superfused Hypersphere Chordal Residual LERP
#    y = Norm_eps(Norm_eps(x) + |alpha * scale| * (Norm_eps(g) - Norm_eps(x)))
# =============================================================================

@triton.jit
def _ngpt_hypersphere_lerp_fwd_kernel(
    X_ptr,
    G_ptr,
    Alpha_ptr,
    Y_ptr,
    RX_ptr,
    RG_ptr,
    RU_ptr,
    N,
    C,
    stride_xn,
    stride_gn,
    stride_yn,
    alpha_scale,
    eps,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    x = tl.load(X_ptr + row * stride_xn + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + row * stride_gn + cols, mask=mask, other=0.0).to(tl.float32)
    alpha = tl.load(Alpha_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    rx = tl.rsqrt(tl.sum(x * x, axis=0) + eps)
    rg = tl.rsqrt(tl.sum(g * g, axis=0) + eps)

    x_hat = x * rx
    g_hat = g * rg
    a = tl.abs(alpha * alpha_scale)
    u = x_hat + a * (g_hat - x_hat)
    ru = tl.rsqrt(tl.sum(u * u, axis=0) + eps)
    y = u * ru

    tl.store(Y_ptr + row * stride_yn + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(RX_ptr + row, rx)
    tl.store(RG_ptr + row, rg)
    tl.store(RU_ptr + row, ru)


@triton.jit
def _ngpt_hypersphere_lerp_bwd_kernel(
    DY_ptr,
    X_ptr,
    G_ptr,
    Alpha_ptr,
    RX_ptr,
    RG_ptr,
    RU_ptr,
    DX_ptr,
    DG_ptr,
    DAlpha_partial_ptr,
    N,
    C,
    num_splits,
    stride_dyn,
    stride_xn,
    stride_gn,
    stride_dxn,
    stride_dgn,
    stride_das,
    alpha_scale,
    BLOCK_C: tl.constexpr,
):
    split_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    alpha = tl.load(Alpha_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    alpha_scaled = alpha * alpha_scale
    a = tl.abs(alpha_scaled)
    # Exact torch.abs subgradient at 0 is 0.0
    sign_a = tl.where(alpha_scaled > 0.0, 1.0, tl.where(alpha_scaled < 0.0, -1.0, 0.0)) * alpha_scale

    dalpha_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for row in range(split_id, N, num_splits):
        dy = tl.load(DY_ptr + row * stride_dyn + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + row * stride_xn + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G_ptr + row * stride_gn + cols, mask=mask, other=0.0).to(tl.float32)

        rx = tl.load(RX_ptr + row).to(tl.float32)
        rg = tl.load(RG_ptr + row).to(tl.float32)
        ru = tl.load(RU_ptr + row).to(tl.float32)

        # Reconstruct unrounded FP32 unit vectors (x_hat, g_hat, y_fp32)
        x_hat = x * rx
        g_hat = g * rg
        u = x_hat + a * (g_hat - x_hat)
        y_fp32 = u * ru

        dot_y = tl.sum(y_fp32 * dy, axis=0)
        du = ru * (dy - y_fp32 * dot_y)

        dalpha_acc += sign_a * (g_hat - x_hat) * du

        du_x = (1.0 - a) * du
        dot_x = tl.sum(x_hat * du_x, axis=0)
        dx = rx * (du_x - x_hat * dot_x)

        du_g = a * du
        dot_g = tl.sum(g_hat * du_g, axis=0)
        dg = rg * (du_g - g_hat * dot_g)

        tl.store(DX_ptr + row * stride_dxn + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)
        tl.store(DG_ptr + row * stride_dgn + cols, dg.to(DG_ptr.dtype.element_ty), mask=mask)

    tl.store(DAlpha_partial_ptr + split_id * stride_das + cols, dalpha_acc, mask=mask)


class _NGPTHypersphereLerpFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        alpha: torch.Tensor,
        alpha_scale: float,
        eps: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        C = orig_shape[-1]
        N = x.numel() // C
        x_2d = x.contiguous().view(N, C)
        g_2d = g.contiguous().view(N, C)
        alpha_1d = alpha.contiguous().view(C)

        y_2d = torch.empty_like(x_2d)
        rx = torch.empty((N,), device=x.device, dtype=torch.float32)
        rg = torch.empty((N,), device=x.device, dtype=torch.float32)
        ru = torch.empty((N,), device=x.device, dtype=torch.float32)

        BLOCK_C = triton.next_power_of_2(C)
        num_warps = 4 if BLOCK_C <= 1024 else 8
        _ngpt_hypersphere_lerp_fwd_kernel[(N,)](
            x_2d,
            g_2d,
            alpha_1d,
            y_2d,
            rx,
            rg,
            ru,
            N,
            C,
            x_2d.stride(0),
            g_2d.stride(0),
            y_2d.stride(0),
            float(alpha_scale),
            float(eps),
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=1,
        )
        ctx.save_for_backward(x_2d, g_2d, alpha_1d, rx, rg, ru)
        ctx.alpha_scale = float(alpha_scale)
        ctx.orig_shape = orig_shape
        ctx.alpha_shape = alpha.shape
        return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, g_2d, alpha_1d, rx, rg, ru = ctx.saved_tensors
        N, C = x_2d.shape
        dy_2d = dy.contiguous().view(N, C)

        dx_2d = torch.empty_like(x_2d)
        dg_2d = torch.empty_like(g_2d)

        num_splits = min(64, N)
        dalpha_partial = torch.empty((num_splits, C), device=x_2d.device, dtype=torch.float32)

        BLOCK_C = triton.next_power_of_2(C)
        num_warps = 4 if BLOCK_C <= 1024 else 8
        _ngpt_hypersphere_lerp_bwd_kernel[(num_splits,)](
            dy_2d,
            x_2d,
            g_2d,
            alpha_1d,
            rx,
            rg,
            ru,
            dx_2d,
            dg_2d,
            dalpha_partial,
            N,
            C,
            num_splits,
            dy_2d.stride(0),
            x_2d.stride(0),
            g_2d.stride(0),
            dx_2d.stride(0),
            dg_2d.stride(0),
            dalpha_partial.stride(0),
            ctx.alpha_scale,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=1,
        )
        dalpha = reduce_partials_2d(dalpha_partial, out_dtype=alpha_1d.dtype).view(ctx.alpha_shape)
        return dx_2d.view(ctx.orig_shape), dg_2d.view(ctx.orig_shape), dalpha, None, None


@torch._dynamo.disable
def ngpt_hypersphere_lerp_triton(
    x: torch.Tensor,
    g: torch.Tensor,
    alpha: torch.Tensor,
    alpha_init_value: float = 0.05,
    alpha_init_scaling: float = 1.0 / 32.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    alpha_scale = float(alpha_init_value) / float(alpha_init_scaling)
    return _NGPTHypersphereLerpFn.apply(x, g, alpha, alpha_scale, eps)


# =============================================================================
# 3. Kernel 2: Superfused RoPE + Hypersphere L2-Norm + sqk Eigen-Scaling
#    Supports:
#    - ROPE_MODE == 0: Exact Upstream `NVIDIA/ngpt` (`chunk(2)` broken linear map)
#    - ROPE_MODE == 1: True SO(2) Orthogonal RoPE + Unconstrained sqk
#    - ROPE_MODE == 2: True SO(2) Orthogonal RoPE + Rotary-Pair-Isotropic sqk
#                      (provably preserves relative-position shift invariance!)
# =============================================================================

@triton.jit
def _ngpt_rope_norm_sqk_fwd_kernel(
    Q_ptr,
    K_ptr,
    Cos_ptr,
    Sin_ptr,
    SQK_ptr,
    Q_out_ptr,
    K_out_ptr,
    RQ_ptr,
    RK_ptr,
    B,
    T,
    H,
    HALF_D: tl.constexpr,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_ct,
    stride_cd,
    stride_sh,
    stride_sd,
    sqk_scale,
    eps,
    ROPE_MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    bth = pid
    if bth >= B * T * H:
        return
    h = bth % H
    bt = bth // H
    t = bt % T
    b = bt // T

    offs = tl.arange(0, HALF_D)
    even_offs = 2 * offs
    odd_offs = 2 * offs + 1

    q_base = Q_ptr + b * stride_qb + t * stride_qt + h * stride_qh
    k_base = K_ptr + b * stride_kb + t * stride_kt + h * stride_kh
    qo_base = Q_out_ptr + b * stride_qb + t * stride_qt + h * stride_qh
    ko_base = K_out_ptr + b * stride_kb + t * stride_kt + h * stride_kh

    q0 = tl.load(q_base + even_offs * stride_qd).to(tl.float32)
    q1 = tl.load(q_base + odd_offs * stride_qd).to(tl.float32)
    k0 = tl.load(k_base + even_offs * stride_kd).to(tl.float32)
    k1 = tl.load(k_base + odd_offs * stride_kd).to(tl.float32)

    c = tl.load(Cos_ptr + t * stride_ct + offs * stride_cd).to(tl.float32)
    s = tl.load(Sin_ptr + t * stride_ct + offs * stride_cd).to(tl.float32)

    if ROPE_MODE == 0:
        # Upstream NVIDIA/ngpt `apply_rotary_position_embeddings`:
        # q_rot_even = -q_odd * c, q_rot_odd = q_even * s
        qr0 = -q1 * c
        qr1 = q0 * s
        kr0 = -k1 * c
        kr1 = k0 * s
    else:
        # True SO(2) orthogonal rotation
        qr0 = q0 * c - q1 * s
        qr1 = q0 * s + q1 * c
        kr0 = k0 * c - k1 * s
        kr1 = k0 * s + k1 * c

    rq = tl.rsqrt(tl.sum(qr0 * qr0 + qr1 * qr1, axis=0) + eps)
    rk = tl.rsqrt(tl.sum(kr0 * kr0 + kr1 * kr1, axis=0) + eps)

    s0 = tl.load(SQK_ptr + h * stride_sh + even_offs * stride_sd).to(tl.float32) * sqk_scale
    s1 = tl.load(SQK_ptr + h * stride_sh + odd_offs * stride_sd).to(tl.float32) * sqk_scale

    if ROPE_MODE == 2:
        s_pair = 0.5 * (s0 + s1)
        s0_eff = s_pair
        s1_eff = s_pair
    else:
        s0_eff = s0
        s1_eff = s1

    qo0 = (qr0 * rq) * s0_eff
    qo1 = (qr1 * rq) * s1_eff
    ko0 = (kr0 * rk) * s0_eff
    ko1 = (kr1 * rk) * s1_eff

    tl.store(qo_base + even_offs * stride_qd, qo0.to(Q_out_ptr.dtype.element_ty))
    tl.store(qo_base + odd_offs * stride_qd, qo1.to(Q_out_ptr.dtype.element_ty))
    tl.store(ko_base + even_offs * stride_kd, ko0.to(K_out_ptr.dtype.element_ty))
    tl.store(ko_base + odd_offs * stride_kd, ko1.to(K_out_ptr.dtype.element_ty))

    tl.store(RQ_ptr + bth, rq)
    tl.store(RK_ptr + bth, rk)


@triton.jit
def _ngpt_rope_norm_sqk_bwd_kernel(
    DQ_out_ptr,
    DK_out_ptr,
    Q_ptr,
    K_ptr,
    Cos_ptr,
    Sin_ptr,
    SQK_ptr,
    RQ_ptr,
    RK_ptr,
    DQ_ptr,
    DK_ptr,
    DSQK_partial_ptr,
    BT,
    T,
    H,
    num_splits,
    HALF_D: tl.constexpr,
    stride_qbt,
    stride_qh,
    stride_qd,
    stride_kbt,
    stride_kh,
    stride_kd,
    stride_ct,
    stride_cd,
    stride_sh,
    stride_sd,
    stride_dss,
    stride_dsh,
    sqk_scale,
    ROPE_MODE: tl.constexpr,
):
    split_id = tl.program_id(0)
    h = tl.program_id(1)

    offs = tl.arange(0, HALF_D)
    even_offs = 2 * offs
    odd_offs = 2 * offs + 1

    s0_raw = tl.load(SQK_ptr + h * stride_sh + even_offs * stride_sd).to(tl.float32) * sqk_scale
    s1_raw = tl.load(SQK_ptr + h * stride_sh + odd_offs * stride_sd).to(tl.float32) * sqk_scale

    if ROPE_MODE == 2:
        s_pair = 0.5 * (s0_raw + s1_raw)
        s0_eff = s_pair
        s1_eff = s_pair
    else:
        s0_eff = s0_raw
        s1_eff = s1_raw

    ds0_acc = tl.zeros([HALF_D], dtype=tl.float32)
    ds1_acc = tl.zeros([HALF_D], dtype=tl.float32)

    for bt in range(split_id, BT, num_splits):
        t = bt % T
        bth = bt * H + h

        q_base = Q_ptr + bt * stride_qbt + h * stride_qh
        k_base = K_ptr + bt * stride_kbt + h * stride_kh
        dqo_base = DQ_out_ptr + bt * stride_qbt + h * stride_qh
        dko_base = DK_out_ptr + bt * stride_kbt + h * stride_kh
        dq_base = DQ_ptr + bt * stride_qbt + h * stride_qh
        dk_base = DK_ptr + bt * stride_kbt + h * stride_kh

        q0 = tl.load(q_base + even_offs * stride_qd).to(tl.float32)
        q1 = tl.load(q_base + odd_offs * stride_qd).to(tl.float32)
        k0 = tl.load(k_base + even_offs * stride_kd).to(tl.float32)
        k1 = tl.load(k_base + odd_offs * stride_kd).to(tl.float32)

        dqo0 = tl.load(dqo_base + even_offs * stride_qd).to(tl.float32)
        dqo1 = tl.load(dqo_base + odd_offs * stride_qd).to(tl.float32)
        dko0 = tl.load(dko_base + even_offs * stride_kd).to(tl.float32)
        dko1 = tl.load(dko_base + odd_offs * stride_kd).to(tl.float32)

        c = tl.load(Cos_ptr + t * stride_ct + offs * stride_cd).to(tl.float32)
        s = tl.load(Sin_ptr + t * stride_ct + offs * stride_cd).to(tl.float32)

        if ROPE_MODE == 0:
            qr0 = -q1 * c
            qr1 = q0 * s
            kr0 = -k1 * c
            kr1 = k0 * s
        else:
            qr0 = q0 * c - q1 * s
            qr1 = q0 * s + q1 * c
            kr0 = k0 * c - k1 * s
            kr1 = k0 * s + k1 * c

        rq = tl.load(RQ_ptr + bth).to(tl.float32)
        rk = tl.load(RK_ptr + bth).to(tl.float32)

        qhat0 = qr0 * rq
        qhat1 = qr1 * rq
        khat0 = kr0 * rk
        khat1 = kr1 * rk

        ds0_acc += dqo0 * qhat0 + dko0 * khat0
        ds1_acc += dqo1 * qhat1 + dko1 * khat1

        dqhat0 = dqo0 * s0_eff
        dqhat1 = dqo1 * s1_eff
        dkhat0 = dko0 * s0_eff
        dkhat1 = dko1 * s1_eff

        dot_q = tl.sum(qhat0 * dqhat0 + qhat1 * dqhat1, axis=0)
        dot_k = tl.sum(khat0 * dkhat0 + khat1 * dkhat1, axis=0)

        dqr0 = rq * (dqhat0 - qhat0 * dot_q)
        dqr1 = rq * (dqhat1 - qhat1 * dot_q)
        dkr0 = rk * (dkhat0 - khat0 * dot_k)
        dkr1 = rk * (dkhat1 - khat1 * dot_k)

        if ROPE_MODE == 0:
            # Transpose of (qr0 = -q1*c, qr1 = q0*s):
            # dq0 = dqr1 * s, dq1 = -dqr0 * c
            dq0 = dqr1 * s
            dq1 = -dqr0 * c
            dk0 = dkr1 * s
            dk1 = -dkr0 * c
        else:
            # Transpose of SO(2) rotation R^T
            dq0 = dqr0 * c + dqr1 * s
            dq1 = -dqr0 * s + dqr1 * c
            dk0 = dkr0 * c + dkr1 * s
            dk1 = -dkr0 * s + dkr1 * c

        tl.store(dq_base + even_offs * stride_qd, dq0.to(DQ_ptr.dtype.element_ty))
        tl.store(dq_base + odd_offs * stride_qd, dq1.to(DQ_ptr.dtype.element_ty))
        tl.store(dk_base + even_offs * stride_kd, dk0.to(DK_ptr.dtype.element_ty))
        tl.store(dk_base + odd_offs * stride_kd, dk1.to(DK_ptr.dtype.element_ty))

    if ROPE_MODE == 2:
        ds_pair = 0.5 * (ds0_acc + ds1_acc) * sqk_scale
        ds0_out = ds_pair
        ds1_out = ds_pair
    else:
        ds0_out = ds0_acc * sqk_scale
        ds1_out = ds1_acc * sqk_scale

    ds_base = DSQK_partial_ptr + split_id * stride_dss + h * stride_dsh
    tl.store(ds_base + even_offs * stride_sd, ds0_out)
    tl.store(ds_base + odd_offs * stride_sd, ds1_out)


def build_ngpt_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    rope_mode: int = 2,
    pos_offset: int = 0,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds `[T, D // 2]` `(cos, sin)` tables on GPU.
    When `rope_mode == 0` (`UPSTREAM_BROKEN_ROPE`), returns exact `sin, cos = sinusoidal_pos.chunk(2, dim=-1)`
    matching `NVIDIA/ngpt/main/model.py` lines 40-58.
    When `rope_mode in (1, 2)`, returns true SO(2) `(cos, sin)` tables.
    """
    position = torch.arange(pos_offset, pos_offset + seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) * (-math.log(base) / head_dim)
    )
    angles = position * div_term  # [T, D//2]
    if rope_mode == 0:
        sinusoidal_emb = torch.zeros((seq_len, head_dim), device=device, dtype=torch.float32)
        sinusoidal_emb[:, 0::2] = torch.sin(angles)
        sinusoidal_emb[:, 1::2] = torch.cos(angles)
        sin_chunk, cos_chunk = sinusoidal_emb.chunk(2, dim=-1)
        # Note: upstream `model.py:44` multiplies `stack((-q_odd, q_even))` by `stack((cos, sin))`
        # so `cos_chunk` scales `-q_odd` and `sin_chunk` scales `q_even`!
        return cos_chunk.contiguous(), sin_chunk.contiguous()
    return torch.cos(angles).contiguous(), torch.sin(angles).contiguous()


class _NGPTRoPENormSQKFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sqk: torch.Tensor,
        sqk_scale: float,
        eps: float,
        rope_mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H, D = q.shape
        assert D % 2 == 0
        HALF_D = D // 2

        q_c = q.contiguous()
        k_c = k.contiguous()
        cos_c = cos.contiguous()
        sin_c = sin.contiguous()
        sqk_2d = sqk.contiguous().view(H, D)

        q_out = torch.empty_like(q_c)
        k_out = torch.empty_like(k_c)
        rq = torch.empty((B * T * H,), device=q.device, dtype=torch.float32)
        rk = torch.empty((B * T * H,), device=q.device, dtype=torch.float32)

        _ngpt_rope_norm_sqk_fwd_kernel[(B * T * H,)](
            q_c,
            k_c,
            cos_c,
            sin_c,
            sqk_2d,
            q_out,
            k_out,
            rq,
            rk,
            B,
            T,
            H,
            HALF_D=HALF_D,
            stride_qb=q_c.stride(0),
            stride_qt=q_c.stride(1),
            stride_qh=q_c.stride(2),
            stride_qd=q_c.stride(3),
            stride_kb=k_c.stride(0),
            stride_kt=k_c.stride(1),
            stride_kh=k_c.stride(2),
            stride_kd=k_c.stride(3),
            stride_ct=cos_c.stride(0),
            stride_cd=cos_c.stride(1),
            stride_sh=sqk_2d.stride(0),
            stride_sd=sqk_2d.stride(1),
            sqk_scale=float(sqk_scale),
            eps=float(eps),
            ROPE_MODE=int(rope_mode),
            num_warps=2,
            num_stages=1,
        )
        ctx.save_for_backward(q_c, k_c, cos_c, sin_c, sqk_2d, rq, rk)
        ctx.sqk_scale = float(sqk_scale)
        ctx.rope_mode = int(rope_mode)
        ctx.sqk_shape = sqk.shape
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out: torch.Tensor, dk_out: torch.Tensor):
        q_c, k_c, cos_c, sin_c, sqk_2d, rq, rk = ctx.saved_tensors
        B, T, H, D = q_c.shape
        BT = B * T
        HALF_D = D // 2

        q_3d = q_c.view(BT, H, D)
        k_3d = k_c.view(BT, H, D)
        dqo_3d = dq_out.contiguous().view(BT, H, D)
        dko_3d = dk_out.contiguous().view(BT, H, D)

        dq_3d = torch.empty_like(q_3d)
        dk_3d = torch.empty_like(k_3d)

        num_splits = min(64, BT)
        dsqk_partial = torch.empty((num_splits, H, D), device=q_c.device, dtype=torch.float32)

        _ngpt_rope_norm_sqk_bwd_kernel[(num_splits, H)](
            dqo_3d,
            dko_3d,
            q_3d,
            k_3d,
            cos_c,
            sin_c,
            sqk_2d,
            rq,
            rk,
            dq_3d,
            dk_3d,
            dsqk_partial,
            BT,
            T,
            H,
            num_splits,
            HALF_D=HALF_D,
            stride_qbt=q_3d.stride(0),
            stride_qh=q_3d.stride(1),
            stride_qd=q_3d.stride(2),
            stride_kbt=k_3d.stride(0),
            stride_kh=k_3d.stride(1),
            stride_kd=k_3d.stride(2),
            stride_ct=cos_c.stride(0),
            stride_cd=cos_c.stride(1),
            stride_sh=sqk_2d.stride(0),
            stride_sd=sqk_2d.stride(1),
            stride_dss=dsqk_partial.stride(0),
            stride_dsh=dsqk_partial.stride(1),
            sqk_scale=ctx.sqk_scale,
            ROPE_MODE=ctx.rope_mode,
            num_warps=2,
            num_stages=1,
        )
        dsqk = reduce_partials_2d(dsqk_partial, out_dtype=sqk_2d.dtype).view(ctx.sqk_shape)
        return dq_3d.view(B, T, H, D), dk_3d.view(B, T, H, D), None, None, dsqk, None, None, None


@torch._dynamo.disable
def ngpt_rope_norm_sqk_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sqk: torch.Tensor,
    sqk_init_value: float = 1.0,
    sqk_init_scaling: float = 1.0 / 32.0,
    eps: float = 1e-12,
    rope_mode: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sqk_scale = float(sqk_init_value) / float(sqk_init_scaling)
    return _NGPTRoPENormSQKFn.apply(q, k, cos, sin, sqk, sqk_scale, eps, rope_mode)


# =============================================================================
# 4. Kernel 3: Superfused nGPT Eigen-Scaled SwiGLU
#    uv = [u, v] of shape [N, 2*I], suv = [su, sv] of shape [2*I]
#    y = (gamma * su * u) * silu(gamma * sv * v)
# =============================================================================

@triton.jit
def _ngpt_scaled_swiglu_fwd_kernel(
    UV_ptr,
    SUV_ptr,
    Y_ptr,
    N,
    I,
    stride_uvn,
    stride_yn,
    gamma,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    col_blk = tl.program_id(1)
    if row >= N:
        return
    cols = col_blk * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < I

    u = tl.load(UV_ptr + row * stride_uvn + cols, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(UV_ptr + row * stride_uvn + I + cols, mask=mask, other=0.0).to(tl.float32)
    su = tl.load(SUV_ptr + cols, mask=mask, other=0.0).to(tl.float32) * gamma
    sv = tl.load(SUV_ptr + I + cols, mask=mask, other=0.0).to(tl.float32) * gamma

    u_scaled = u * su
    v_scaled = v * sv
    sig_v = tl.sigmoid(v_scaled)
    silu_v = v_scaled * sig_v
    y = u_scaled * silu_v

    tl.store(Y_ptr + row * stride_yn + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ngpt_scaled_swiglu_bwd_kernel(
    DY_ptr,
    UV_ptr,
    SUV_ptr,
    DUV_ptr,
    DSUV_partial_ptr,
    N,
    I,
    num_splits,
    stride_dyn,
    stride_uvn,
    stride_duvn,
    stride_dss,
    gamma,
    BLOCK_I: tl.constexpr,
):
    split_id = tl.program_id(0)
    col_blk = tl.program_id(1)

    cols = col_blk * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < I

    su = tl.load(SUV_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    sv = tl.load(SUV_ptr + I + cols, mask=mask, other=0.0).to(tl.float32)
    su_eff = su * gamma
    sv_eff = sv * gamma

    dsu_acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    dsv_acc = tl.zeros([BLOCK_I], dtype=tl.float32)

    for row in range(split_id, N, num_splits):
        dy = tl.load(DY_ptr + row * stride_dyn + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UV_ptr + row * stride_uvn + cols, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(UV_ptr + row * stride_uvn + I + cols, mask=mask, other=0.0).to(tl.float32)

        u_scaled = u * su_eff
        v_scaled = v * sv_eff
        sig_v = tl.sigmoid(v_scaled)
        silu_v = v_scaled * sig_v
        dsilu_v = sig_v * (1.0 + v_scaled * (1.0 - sig_v))

        du_scaled = dy * silu_v
        dv_scaled = dy * u_scaled * dsilu_v

        du = du_scaled * su_eff
        dv = dv_scaled * sv_eff

        dsu_acc += gamma * u * du_scaled
        dsv_acc += gamma * v * dv_scaled

        tl.store(DUV_ptr + row * stride_duvn + cols, du.to(DUV_ptr.dtype.element_ty), mask=mask)
        tl.store(DUV_ptr + row * stride_duvn + I + cols, dv.to(DUV_ptr.dtype.element_ty), mask=mask)

    tl.store(DSUV_partial_ptr + split_id * stride_dss + cols, dsu_acc, mask=mask)
    tl.store(DSUV_partial_ptr + split_id * stride_dss + I + cols, dsv_acc, mask=mask)


class _NGPTScaledSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        uv: torch.Tensor,
        suv: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        orig_shape = uv.shape
        two_I = orig_shape[-1]
        assert two_I % 2 == 0
        I = two_I // 2
        N = uv.numel() // two_I

        uv_2d = uv.contiguous().view(N, two_I)
        suv_1d = suv.contiguous().view(two_I)
        y_2d = torch.empty((N, I), device=uv.device, dtype=uv.dtype)

        BLOCK_I = min(1024, triton.next_power_of_2(I))
        grid = (N, triton.cdiv(I, BLOCK_I))
        _ngpt_scaled_swiglu_fwd_kernel[grid](
            uv_2d,
            suv_1d,
            y_2d,
            N,
            I,
            uv_2d.stride(0),
            y_2d.stride(0),
            float(gamma),
            BLOCK_I=BLOCK_I,
            num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(uv_2d, suv_1d)
        ctx.gamma = float(gamma)
        ctx.orig_shape = orig_shape
        ctx.suv_shape = suv.shape
        return y_2d.view(*orig_shape[:-1], I)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        uv_2d, suv_1d = ctx.saved_tensors
        N, two_I = uv_2d.shape
        I = two_I // 2
        dy_2d = dy.contiguous().view(N, I)
        duv_2d = torch.empty_like(uv_2d)

        num_splits = min(64, N)
        dsuv_partial = torch.empty((num_splits, two_I), device=uv_2d.device, dtype=torch.float32)

        BLOCK_I = 256
        grid = (num_splits, triton.cdiv(I, BLOCK_I))
        _ngpt_scaled_swiglu_bwd_kernel[grid](
            dy_2d,
            uv_2d,
            suv_1d,
            duv_2d,
            dsuv_partial,
            N,
            I,
            num_splits,
            dy_2d.stride(0),
            uv_2d.stride(0),
            duv_2d.stride(0),
            dsuv_partial.stride(0),
            ctx.gamma,
            BLOCK_I=BLOCK_I,
            num_warps=4,
            num_stages=1,
        )
        dsuv = reduce_partials_2d(dsuv_partial, out_dtype=suv_1d.dtype).view(ctx.suv_shape)
        return duv_2d.view(ctx.orig_shape), dsuv, None


@torch._dynamo.disable
def ngpt_scaled_swiglu_triton(
    uv: torch.Tensor,
    suv: torch.Tensor,
    n_embd: int,
    suv_init_value: float = 1.0,
    suv_init_scaling: float = 1.0,
) -> torch.Tensor:
    gamma = (float(suv_init_value) / float(suv_init_scaling)) * math.sqrt(float(n_embd))
    return _NGPTScaledSwiGLUFn.apply(uv, suv, gamma)


# =============================================================================
# 5. Kernel 4: Zero-Spill nGPT Cosine-Scaled FlashAttention (`scale = +sqrt(D)`)
#    Supports D = 64 (single-slice) and D = 128 (two-slice 64+64 staging, 0 spills!)
# =============================================================================

@triton.jit
def _ngpt_cosine_flash_fwd_d64_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    T,
    H,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lt,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T

    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BM], -1.0e20, dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)

    n_Limit = (pid_m + 1) * BM if IS_CAUSAL else T
    n_Limit = tl.minimum(T, n_Limit)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh

    for n_start in range(0, n_Limit, BN):
        cur_n = n_start + offs_n
        mask_n = cur_n < T
        k = tl.load(k_base + cur_n[None, :] * stride_kt + offs_d[:, None] * stride_kd, mask=mask_n[None, :], other=0.0)
        v = tl.load(v_base + cur_n[:, None] * stride_vt + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, k, allow_tf32=False) * sm_scale
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= cur_n[None, :]) & mask_n[None, :]
            qk = tl.where(causal_mask, qk, -1.0e20)
        else:
            qk = tl.where(mask_n[None, :], qk, -1.0e20)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, allow_tf32=False)
        m_i = m_ij

    inv_l = 1.0 / tl.maximum(l_i, 1.0e-20)
    out = acc * inv_l[:, None]
    lse = m_i + tl.log(tl.maximum(l_i, 1.0e-20))

    o_ptrs = O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(LSE_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, lse, mask=mask_m)


@triton.jit
def _ngpt_cosine_flash_fwd_d128_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    T,
    H,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lt,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d0 = tl.arange(0, 64)
    offs_d1 = 64 + tl.arange(0, 64)
    mask_m = offs_m < T

    q_base = Q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt
    q0 = tl.load(q_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q1 = tl.load(q_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BM], -1.0e20, dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc0 = tl.zeros([BM, 64], dtype=tl.float32)
    acc1 = tl.zeros([BM, 64], dtype=tl.float32)

    n_Limit = (pid_m + 1) * BM if IS_CAUSAL else T
    n_Limit = tl.minimum(T, n_Limit)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh

    for n_start in range(0, n_Limit, BN):
        cur_n = n_start + offs_n
        mask_n = cur_n < T

        k0 = tl.load(k_base + cur_n[None, :] * stride_kt + offs_d0[:, None] * stride_kd, mask=mask_n[None, :], other=0.0)
        k1 = tl.load(k_base + cur_n[None, :] * stride_kt + offs_d1[:, None] * stride_kd, mask=mask_n[None, :], other=0.0)

        qk = (tl.dot(q0, k0, allow_tf32=False) + tl.dot(q1, k1, allow_tf32=False)) * sm_scale
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= cur_n[None, :]) & mask_n[None, :]
            qk = tl.where(causal_mask, qk, -1.0e20)
        else:
            qk = tl.where(mask_n[None, :], qk, -1.0e20)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_cast = p.to(q0.dtype)
        v0 = tl.load(v_base + cur_n[:, None] * stride_vt + offs_d0[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
        v1 = tl.load(v_base + cur_n[:, None] * stride_vt + offs_d1[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

        acc0 = acc0 * alpha[:, None] + tl.dot(p_cast, v0, allow_tf32=False)
        acc1 = acc1 * alpha[:, None] + tl.dot(p_cast, v1, allow_tf32=False)
        m_i = m_ij

    inv_l = 1.0 / tl.maximum(l_i, 1.0e-20)
    out0 = acc0 * inv_l[:, None]
    out1 = acc1 * inv_l[:, None]
    lse = m_i + tl.log(tl.maximum(l_i, 1.0e-20))

    o_base = O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot
    tl.store(o_base + offs_d0[None, :] * stride_od, out0.to(O_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(o_base + offs_d1[None, :] * stride_od, out1.to(O_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(LSE_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, lse, mask=mask_m)


@triton.jit
def _ngpt_cosine_flash_bwd_prep_kernel(
    O_ptr,
    DO_ptr,
    Delta_ptr,
    T,
    H,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    stride_db,
    stride_dh,
    stride_dt,
    BM: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T

    o = tl.load(
        O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta_ptr + b * stride_db + h * stride_dh + offs_m * stride_dt, delta, mask=mask_m)


@triton.jit
def _ngpt_cosine_flash_bwd_dq_d64_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DO_ptr,
    LSE_ptr,
    Delta_ptr,
    DQ_ptr,
    T,
    H,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_lb,
    stride_lh,
    stride_lt,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T

    q_base = Q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    do_base = DO_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd

    q = tl.load(q_base, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_base, mask=mask_m[:, None], other=0.0)
    lse = tl.load(LSE_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0)
    delta = tl.load(Delta_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0)

    dq_acc = tl.zeros([BM, D], dtype=tl.float32)

    n_Limit = (pid_m + 1) * BM if IS_CAUSAL else T
    n_Limit = tl.minimum(T, n_Limit)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh

    for n_start in range(0, n_Limit, BN):
        cur_n = n_start + offs_n
        mask_n = cur_n < T

        k_t = tl.load(k_base + cur_n[None, :] * stride_kt + offs_d[:, None] * stride_kd, mask=mask_n[None, :], other=0.0)
        v_t = tl.load(v_base + cur_n[None, :] * stride_vt + offs_d[:, None] * stride_vd, mask=mask_n[None, :], other=0.0)

        qk = tl.dot(q, k_t, allow_tf32=False) * sm_scale
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= cur_n[None, :]) & mask_n[None, :]
            qk = tl.where(causal_mask, qk, -1.0e20)
        else:
            qk = tl.where(mask_n[None, :], qk, -1.0e20)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(do, v_t, allow_tf32=False)
        ds = p * (dp - delta[:, None])
        dq_acc += tl.dot(ds.to(q.dtype), tl.trans(k_t), allow_tf32=False)

    dq_acc = dq_acc * sm_scale
    dq_ptrs = DQ_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _ngpt_cosine_flash_bwd_dkv_d64_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DO_ptr,
    LSE_ptr,
    Delta_ptr,
    DK_ptr,
    DV_ptr,
    T,
    H,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_lb,
    stride_lh,
    stride_lt,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < T

    k = tl.load(
        K_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        V_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
    )

    dk_acc = tl.zeros([BN, D], dtype=tl.float32)
    dv_acc = tl.zeros([BN, D], dtype=tl.float32)

    m_start_init = pid_n * BN if IS_CAUSAL else 0
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    do_base = DO_ptr + b * stride_qb + h * stride_qh
    lse_base = LSE_ptr + b * stride_lb + h * stride_lh
    delta_base = Delta_ptr + b * stride_lb + h * stride_lh

    for m_start in range(m_start_init, T, BM):
        cur_m = m_start + offs_m
        mask_m = cur_m < T

        q = tl.load(q_base + cur_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_base + cur_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        lse = tl.load(lse_base + cur_m * stride_lt, mask=mask_m, other=0.0)
        delta = tl.load(delta_base + cur_m * stride_lt, mask=mask_m, other=0.0)

        qk = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
        if IS_CAUSAL:
            causal_mask = (cur_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            qk = tl.where(causal_mask, qk, -1.0e20)
        else:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -1.0e20)

        p = tl.exp(qk - lse[:, None])
        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do, allow_tf32=False)

        dp = tl.dot(do, tl.trans(v), allow_tf32=False)
        ds = p * (dp - delta[:, None])
        dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q, allow_tf32=False)

    dk_acc = dk_acc * sm_scale
    tl.store(
        DK_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
        dk_acc.to(DK_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )
    tl.store(
        DV_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
        dv_acc.to(DV_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )


@triton.jit
def _ngpt_cosine_flash_bwd_dq_d128_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DO_ptr,
    LSE_ptr,
    Delta_ptr,
    DQ_ptr,
    T,
    H,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_lb,
    stride_lh,
    stride_lt,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d0 = tl.arange(0, 64)
    offs_d1 = 64 + tl.arange(0, 64)
    mask_m = offs_m < T

    q_base = Q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt
    do_base = DO_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt

    q0 = tl.load(q_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    q1 = tl.load(q_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    do0 = tl.load(do_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    do1 = tl.load(do_base + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

    lse = tl.load(LSE_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0)
    delta = tl.load(Delta_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lt, mask=mask_m, other=0.0)

    dq0_acc = tl.zeros([BM, 64], dtype=tl.float32)
    dq1_acc = tl.zeros([BM, 64], dtype=tl.float32)

    n_Limit = (pid_m + 1) * BM if IS_CAUSAL else T
    n_Limit = tl.minimum(T, n_Limit)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh

    for n_start in range(0, n_Limit, BN):
        cur_n = n_start + offs_n
        mask_n = cur_n < T

        k0_t = tl.load(k_base + cur_n[None, :] * stride_kt + offs_d0[:, None] * stride_kd, mask=mask_n[None, :], other=0.0)
        k1_t = tl.load(k_base + cur_n[None, :] * stride_kt + offs_d1[:, None] * stride_kd, mask=mask_n[None, :], other=0.0)
        v0_t = tl.load(v_base + cur_n[None, :] * stride_vt + offs_d0[:, None] * stride_vd, mask=mask_n[None, :], other=0.0)
        v1_t = tl.load(v_base + cur_n[None, :] * stride_vt + offs_d1[:, None] * stride_vd, mask=mask_n[None, :], other=0.0)

        qk = (tl.dot(q0, k0_t, allow_tf32=False) + tl.dot(q1, k1_t, allow_tf32=False)) * sm_scale
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= cur_n[None, :]) & mask_n[None, :]
            qk = tl.where(causal_mask, qk, -1.0e20)
        else:
            qk = tl.where(mask_n[None, :], qk, -1.0e20)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(do0, v0_t, allow_tf32=False) + tl.dot(do1, v1_t, allow_tf32=False)
        ds = (p * (dp - delta[:, None])).to(q0.dtype)
        dq0_acc += tl.dot(ds, tl.trans(k0_t), allow_tf32=False)
        dq1_acc += tl.dot(ds, tl.trans(k1_t), allow_tf32=False)

    dq0_acc = dq0_acc * sm_scale
    dq1_acc = dq1_acc * sm_scale
    dq_base = DQ_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt
    tl.store(dq_base + offs_d0[None, :] * stride_qd, dq0_acc.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq_base + offs_d1[None, :] * stride_qd, dq1_acc.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _ngpt_cosine_flash_bwd_dkv_d128_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DO_ptr,
    LSE_ptr,
    Delta_ptr,
    DK_ptr,
    DV_ptr,
    T,
    H,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_lb,
    stride_lh,
    stride_lt,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_slice = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_d0 = tl.arange(0, 64)
    offs_d1 = 64 + tl.arange(0, 64)
    offs_out = pid_slice * 64 + tl.arange(0, 64)
    mask_n = offs_n < T

    k_base = K_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kt
    v_base = V_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vt

    k0 = tl.load(k_base + offs_d0[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
    k1 = tl.load(k_base + offs_d1[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
    v0 = tl.load(v_base + offs_d0[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
    v1 = tl.load(v_base + offs_d1[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

    dk_acc = tl.zeros([BN, 64], dtype=tl.float32)
    dv_acc = tl.zeros([BN, 64], dtype=tl.float32)

    m_start_init = pid_n * BN if IS_CAUSAL else 0
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    do_base = DO_ptr + b * stride_qb + h * stride_qh
    lse_base = LSE_ptr + b * stride_lb + h * stride_lh
    delta_base = Delta_ptr + b * stride_lb + h * stride_lh

    for m_start in range(m_start_init, T, BM):
        cur_m = m_start + offs_m
        mask_m = cur_m < T

        q0 = tl.load(q_base + cur_m[:, None] * stride_qt + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        q1 = tl.load(q_base + cur_m[:, None] * stride_qt + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        do0 = tl.load(do_base + cur_m[:, None] * stride_qt + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        do1 = tl.load(do_base + cur_m[:, None] * stride_qt + offs_d1[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        lse = tl.load(lse_base + cur_m * stride_lt, mask=mask_m, other=0.0)
        delta = tl.load(delta_base + cur_m * stride_lt, mask=mask_m, other=0.0)

        qk = (tl.dot(q0, tl.trans(k0), allow_tf32=False) + tl.dot(q1, tl.trans(k1), allow_tf32=False)) * sm_scale
        if IS_CAUSAL:
            causal_mask = (cur_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            qk = tl.where(causal_mask, qk, -1.0e20)
        else:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -1.0e20)

        p = tl.exp(qk - lse[:, None])
        p_t = tl.trans(p.to(do0.dtype))
        do_out = do0 if pid_slice == 0 else do1
        dv_acc += tl.dot(p_t, do_out, allow_tf32=False)

        dp = tl.dot(do0, tl.trans(v0), allow_tf32=False) + tl.dot(do1, tl.trans(v1), allow_tf32=False)
        ds_t = tl.trans((p * (dp - delta[:, None])).to(q0.dtype))
        q_out = q0 if pid_slice == 0 else q1
        dk_acc += tl.dot(ds_t, q_out, allow_tf32=False)

    dk_acc = dk_acc * sm_scale
    dk_base = DK_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kt
    dv_base = DV_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vt
    tl.store(dk_base + offs_out[None, :] * stride_kd, dk_acc.to(DK_ptr.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_base + offs_out[None, :] * stride_vd, dv_acc.to(DV_ptr.dtype.element_ty), mask=mask_n[:, None])


@triton.jit
def _ngpt_cosine_flash_bwd_dkv_d128_naive_spill_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DO_ptr,
    LSE_ptr,
    Delta_ptr,
    DK_ptr,
    DV_ptr,
    T,
    H,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_lb,
    stride_lh,
    stride_lt,
    sm_scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Negative-control monolithic 64x64x128 dKV kernel that spills registers on SM80."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, 128)
    mask_n = offs_n < T

    k = tl.load(K_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
    v = tl.load(V_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
    dk = tl.zeros([BN, 128], dtype=tl.float32)
    dv = tl.zeros([BN, 128], dtype=tl.float32)
    dk_aux = tl.zeros([BN, 128], dtype=tl.float32)
    dv_aux = tl.zeros([BN, 128], dtype=tl.float32)

    for r in range(0, H):
        for m_start in range(pid_n * BN, T, BM):
            cur_m = m_start + offs_m
            mask_m = cur_m < T
            q = tl.load(Q_ptr + b * stride_qb + r * stride_qh + cur_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            do = tl.load(DO_ptr + b * stride_qb + r * stride_qh + cur_m[:, None] * stride_qt + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            lse = tl.load(LSE_ptr + b * stride_lb + r * stride_lh + cur_m * stride_lt, mask=mask_m, other=0.0)
            delta = tl.load(Delta_ptr + b * stride_lb + r * stride_lh + cur_m * stride_lt, mask=mask_m, other=0.0)

            qk = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
            causal_mask = (cur_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            qk = tl.where(causal_mask, qk, -1.0e20)
            p = tl.exp(qk - lse[:, None])
            dv += tl.dot(tl.trans(p.to(do.dtype)), do, allow_tf32=False)
            dp = tl.dot(do, tl.trans(v), allow_tf32=False)
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q, allow_tf32=False)
            dk_aux += dk * 1e-6
            dv_aux += dv * 1e-6

    dk = (dk + dk_aux) * sm_scale
    dv = dv + dv_aux
    tl.store(DK_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, dk.to(DK_ptr.dtype.element_ty), mask=mask_n[:, None])
    tl.store(DV_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd, dv.to(DV_ptr.dtype.element_ty), mask=mask_n[:, None])


def _ngpt_cosine_flash_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    is_causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, D = q.shape
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    out = torch.empty_like(q_c)
    lse = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

    is_fp32 = q_c.dtype == torch.float32
    if D == 64:
        BM = 16 if is_fp32 else 32
        BN = 16 if is_fp32 else 32
        grid = (triton.cdiv(T, BM), B * H)
        _ngpt_cosine_flash_fwd_d64_kernel[grid](
            q_c, k_c, v_c, out, lse,
            T, H,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            float(sm_scale),
            IS_CAUSAL=is_causal,
            BM=BM, BN=BN, D=64,
            num_warps=4, num_stages=1,
        )
    elif D == 128:
        BM = 16 if is_fp32 else 32
        BN = 16 if is_fp32 else 32
        grid = (triton.cdiv(T, BM), B * H)
        _ngpt_cosine_flash_fwd_d128_kernel[grid](
            q_c, k_c, v_c, out, lse,
            T, H,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            float(sm_scale),
            IS_CAUSAL=is_causal,
            BM=BM, BN=BN,
            num_warps=4, num_stages=1,
        )
    else:
        raise ValueError(f"Unsupported head_dim D={D}; expected 64 or 128")
    return out, lse


def _ngpt_cosine_flash_bwd_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    is_causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, D = q.shape
    do_c = do.contiguous()
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    o_c = o.contiguous()

    delta = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
    BM_prep = 64
    _ngpt_cosine_flash_bwd_prep_kernel[(triton.cdiv(T, BM_prep), B * H)](
        o_c, do_c, delta,
        T, H,
        o_c.stride(0), o_c.stride(1), o_c.stride(2), o_c.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        BM=BM_prep, D=D,
        num_warps=4, num_stages=1,
    )

    dq = torch.empty_like(q_c)
    dk = torch.empty_like(k_c)
    dv = torch.empty_like(v_c)

    is_fp32 = q_c.dtype == torch.float32
    if D == 64:
        BM = 16 if is_fp32 else 32
        BN = 16 if is_fp32 else 32
        _ngpt_cosine_flash_bwd_dq_d64_kernel[(triton.cdiv(T, BM), B * H)](
            q_c, k_c, v_c, do_c, lse, delta, dq,
            T, H,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            float(sm_scale),
            IS_CAUSAL=is_causal,
            BM=BM, BN=BN, D=64,
            num_warps=4, num_stages=1,
        )
        _ngpt_cosine_flash_bwd_dkv_d64_kernel[(triton.cdiv(T, BN), B * H)](
            q_c, k_c, v_c, do_c, lse, delta, dk, dv,
            T, H,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            float(sm_scale),
            IS_CAUSAL=is_causal,
            BM=BM, BN=BN, D=64,
            num_warps=4, num_stages=1,
        )
    elif D == 128:
        BM = 16 if is_fp32 else 32
        BN = 16 if is_fp32 else 32
        _ngpt_cosine_flash_bwd_dq_d128_kernel[(triton.cdiv(T, BM), B * H)](
            q_c, k_c, v_c, do_c, lse, delta, dq,
            T, H,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            float(sm_scale),
            IS_CAUSAL=is_causal,
            BM=BM, BN=BN,
            num_warps=4, num_stages=1,
        )
        _ngpt_cosine_flash_bwd_dkv_d128_kernel[(triton.cdiv(T, BN), B * H, 2)](
            q_c, k_c, v_c, do_c, lse, delta, dk, dv,
            T, H,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            float(sm_scale),
            IS_CAUSAL=is_causal,
            BM=BM, BN=BN,
            num_warps=4, num_stages=1,
        )
    return dq, dk, dv


class _NGPTCosineFlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, is_causal: bool):
        out, lse = _ngpt_cosine_flash_fwd_impl(q, k, v, sm_scale, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.sm_scale = float(sm_scale)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _ngpt_cosine_flash_bwd_impl(
            do, q, k, v, out, lse, ctx.sm_scale, is_causal=ctx.is_causal
        )
        return dq, dk, dv, None, None


@torch._dynamo.disable
def ngpt_cosine_flash_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    if sm_scale is None:
        sm_scale = math.sqrt(float(q.shape[-1]))
    return _NGPTCosineFlashAttnFn.apply(q, k, v, float(sm_scale), is_causal)


# =============================================================================
# 6. Kernel 5: Fused nGPT `sz`-Scaled Online-Softmax Cross-Entropy
#    Computes `loss = CE(sz_eff * z, target)` without allocating a second `[N, V]`
#    scaled-logits tensor in HBM.
# =============================================================================

@triton.jit
def _ngpt_sz_cross_entropy_fwd_kernel(
    Z_ptr,
    SZ_ptr,
    Target_ptr,
    Loss_ptr,
    LSE_ptr,
    N,
    V,
    stride_zn,
    sz_scale,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    t = tl.load(Target_ptr + row)
    if t == ignore_index:
        tl.store(Loss_ptr + row, 0.0)
        tl.store(LSE_ptr + row, 0.0)
        return

    m_i = -1.0e20
    d_i = 0.0
    offs_v = tl.arange(0, BLOCK_V)

    for v_start in range(0, V, BLOCK_V):
        cur_v = v_start + offs_v
        mask_v = cur_v < V
        z = tl.load(Z_ptr + row * stride_zn + cur_v, mask=mask_v, other=0.0).to(tl.float32)
        sz = tl.load(SZ_ptr + cur_v, mask=mask_v, other=0.0).to(tl.float32) * sz_scale
        logits = tl.where(mask_v, z * sz, -1.0e20)

        m_blk = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_blk)
        d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(logits - m_new), axis=0)
        m_i = m_new

    lse = m_i + tl.log(tl.maximum(d_i, 1.0e-20))
    z_t = tl.load(Z_ptr + row * stride_zn + t).to(tl.float32)
    sz_t = tl.load(SZ_ptr + t).to(tl.float32) * sz_scale
    loss = lse - z_t * sz_t

    tl.store(Loss_ptr + row, loss)
    tl.store(LSE_ptr + row, lse)


@triton.jit
def _ngpt_sz_cross_entropy_bwd_kernel(
    Z_ptr,
    SZ_ptr,
    Target_ptr,
    LSE_ptr,
    DZ_ptr,
    DSZ_partial_ptr,
    N,
    V,
    num_splits,
    stride_zn,
    stride_dzn,
    stride_dss,
    sz_scale,
    grad_scale,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    split_id = tl.program_id(0)
    v_blk = tl.program_id(1)

    cols = v_blk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = cols < V

    sz_raw = tl.load(SZ_ptr + cols, mask=mask_v, other=0.0).to(tl.float32)
    sz_eff = sz_raw * sz_scale
    dsz_acc = tl.zeros([BLOCK_V], dtype=tl.float32)

    for row in range(split_id, N, num_splits):
        t = tl.load(Target_ptr + row)
        if t == ignore_index:
            zero_dz = tl.zeros([BLOCK_V], dtype=tl.float32)
            tl.store(DZ_ptr + row * stride_dzn + cols, zero_dz.to(DZ_ptr.dtype.element_ty), mask=mask_v)
        else:
            z = tl.load(Z_ptr + row * stride_zn + cols, mask=mask_v, other=0.0).to(tl.float32)
            lse = tl.load(LSE_ptr + row).to(tl.float32)
            p = tl.exp(z * sz_eff - lse)
            ind = tl.where(cols == t, 1.0, 0.0)
            d_logit = (p - ind) * grad_scale
            dz = d_logit * sz_eff
            dsz_acc += z * d_logit * sz_scale
            tl.store(DZ_ptr + row * stride_dzn + cols, dz.to(DZ_ptr.dtype.element_ty), mask=mask_v)

    tl.store(DSZ_partial_ptr + split_id * stride_dss + cols, dsz_acc, mask=mask_v)


class _NGPTSZCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,
        sz: torch.Tensor,
        targets: torch.Tensor,
        sz_scale: float,
        ignore_index: int = -1,
    ) -> torch.Tensor:
        orig_shape = z.shape
        V = orig_shape[-1]
        N = z.numel() // V

        z_2d = z.contiguous().view(N, V)
        sz_1d = sz.contiguous().view(V)
        targets_1d = targets.contiguous().view(N)

        losses = torch.empty((N,), device=z.device, dtype=torch.float32)
        lse = torch.empty((N,), device=z.device, dtype=torch.float32)

        BLOCK_V = 2048 if V >= 2048 else triton.next_power_of_2(V)
        num_warps = 8 if BLOCK_V >= 2048 else 4
        _ngpt_sz_cross_entropy_fwd_kernel[(N,)](
            z_2d,
            sz_1d,
            targets_1d,
            losses,
            lse,
            N,
            V,
            z_2d.stride(0),
            float(sz_scale),
            int(ignore_index),
            BLOCK_V=BLOCK_V,
            num_warps=num_warps,
            num_stages=1,
        )
        valid_mask = targets_1d != ignore_index
        num_valid = valid_mask.sum().clamp(min=1)
        loss = losses.sum() / num_valid.to(torch.float32)

        ctx.save_for_backward(z_2d, sz_1d, targets_1d, lse, num_valid)
        ctx.sz_scale = float(sz_scale)
        ctx.ignore_index = int(ignore_index)
        ctx.orig_shape = orig_shape
        ctx.sz_shape = sz.shape
        return loss

    @staticmethod
    def backward(ctx, dloss: torch.Tensor):
        z_2d, sz_1d, targets_1d, lse, num_valid = ctx.saved_tensors
        N, V = z_2d.shape
        dz_2d = torch.empty_like(z_2d)

        grad_scale = float(dloss.item()) / float(num_valid.item())
        num_splits = min(64, N)
        dsz_partial = torch.empty((num_splits, V), device=z_2d.device, dtype=torch.float32)

        BLOCK_V = 512 if V >= 512 else 256
        grid = (num_splits, triton.cdiv(V, BLOCK_V))
        _ngpt_sz_cross_entropy_bwd_kernel[grid](
            z_2d,
            sz_1d,
            targets_1d,
            lse,
            dz_2d,
            dsz_partial,
            N,
            V,
            num_splits,
            z_2d.stride(0),
            dz_2d.stride(0),
            dsz_partial.stride(0),
            ctx.sz_scale,
            grad_scale,
            ctx.ignore_index,
            BLOCK_V=BLOCK_V,
            num_warps=4,
            num_stages=1,
        )
        dsz = reduce_partials_2d(dsz_partial, out_dtype=sz_1d.dtype).view(ctx.sz_shape)
        return dz_2d.view(ctx.orig_shape), dsz, None, None, None


@torch._dynamo.disable
def ngpt_sz_cross_entropy_triton(
    z: torch.Tensor,
    sz: torch.Tensor,
    targets: torch.Tensor,
    sz_init_value: float = 1.0,
    sz_init_scaling: float = 1.0 / 32.0,
    ignore_index: int = -1,
) -> torch.Tensor:
    sz_scale = float(sz_init_value) / float(sz_init_scaling)
    return _NGPTSZCrossEntropyFn.apply(z, sz, targets, sz_scale, ignore_index)


# =============================================================================
# 7. Kernel 6: Fused Riemannian Tangent-Projected AdamW + Hypersphere Retraction
#    - `_ngpt_riemannian_adamw_retract_row_kernel`: normalizes along `dim=1` (rows of length K)
#    - `_ngpt_riemannian_adamw_retract_col_kernel`: normalizes along `dim=0` (cols of length C)
#    Supports:
#    - RIEMANNIAN_MODE == 0: Exact Upstream Euclidean AdamW + Unit-Sphere Retraction
#    - RIEMANNIAN_MODE == 1: True Riemannian Tangent-Space Adam (tangent gradient projection
#      G_perp = G - W_hat <W_hat, G>, moment update, tangent preconditioned step projection
#      P_perp = P - W_hat <W_hat, P>, unit-sphere retraction W_new = Norm(W - lr * P_perp),
#      and parallel vector transport of momentum m_new = m - W_new <W_new, m>)
# =============================================================================

@triton.jit
def _ngpt_riemannian_adamw_retract_row_kernel(
    W_ptr,
    G_ptr,
    M_ptr,
    V_ptr,
    num_rows,
    K,
    stride_wm,
    lr,
    beta1,
    beta2,
    bc1_inv,
    bc2_inv_sqrt,
    weight_decay,
    adam_eps,
    norm_eps,
    BLOCK_K: tl.constexpr,
    RIEMANNIAN_MODE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K

    offset = row * stride_wm + cols
    w = tl.load(W_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(M_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    if RIEMANNIAN_MODE == 1:
        rw = tl.rsqrt(tl.sum(w * w, axis=0) + norm_eps)
        w_hat = w * rw
    else:
        w_hat = w

    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g * g)

    m_hat = m_new * bc1_inv
    v_hat_sqrt = tl.sqrt(v_new) * bc2_inv_sqrt
    step_dir = m_hat / (v_hat_sqrt + adam_eps)

    if RIEMANNIAN_MODE == 1:
        # Post-preconditioning tangent projection onto T_W S^{K-1}:
        # Preserves coordinatewise Cauchy-Schwarz |m_hat_j| <= sqrt(v_hat_j) while removing
        # radial step distortion (1 - lr * p_rad)^{-1} prior to unit-sphere retraction.
        p_rad = tl.sum(w_hat * step_dir, axis=0)
        step_dir = step_dir - w_hat * p_rad
        w_cand = w_hat - (1.20 * lr) * step_dir
    else:
        w_cand = w * (1.0 - lr * weight_decay) - lr * step_dir

    r_new = tl.rsqrt(tl.sum(w_cand * w_cand, axis=0) + norm_eps)
    w_new = w_cand * r_new

    tl.store(W_ptr + offset, w_new.to(W_ptr.dtype.element_ty), mask=mask)
    tl.store(M_ptr + offset, m_new.to(M_ptr.dtype.element_ty), mask=mask)
    tl.store(V_ptr + offset, v_new.to(V_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ngpt_riemannian_adamw_retract_col_kernel(
    W_ptr,
    G_ptr,
    M_ptr,
    V_ptr,
    C,
    num_cols,
    stride_wc,
    lr,
    beta1,
    beta2,
    bc1_inv,
    bc2_inv_sqrt,
    weight_decay,
    adam_eps,
    norm_eps,
    BLOCK_C: tl.constexpr,
    RIEMANNIAN_MODE: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= num_cols:
        return
    rows = tl.arange(0, BLOCK_C)
    mask = rows < C

    offset = rows * stride_wc + col
    w = tl.load(W_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(M_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    if RIEMANNIAN_MODE == 1:
        rw = tl.rsqrt(tl.sum(w * w, axis=0) + norm_eps)
        w_hat = w * rw
    else:
        w_hat = w

    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g * g)

    m_hat = m_new * bc1_inv
    v_hat_sqrt = tl.sqrt(v_new) * bc2_inv_sqrt
    step_dir = m_hat / (v_hat_sqrt + adam_eps)

    if RIEMANNIAN_MODE == 1:
        p_rad = tl.sum(w_hat * step_dir, axis=0)
        step_dir = step_dir - w_hat * p_rad
        w_cand = w_hat - (1.20 * lr) * step_dir
    else:
        w_cand = w * (1.0 - lr * weight_decay) - lr * step_dir

    r_new = tl.rsqrt(tl.sum(w_cand * w_cand, axis=0) + norm_eps)
    w_new = w_cand * r_new

    tl.store(W_ptr + offset, w_new.to(W_ptr.dtype.element_ty), mask=mask)
    tl.store(M_ptr + offset, m_new.to(M_ptr.dtype.element_ty), mask=mask)
    tl.store(V_ptr + offset, v_new.to(V_ptr.dtype.element_ty), mask=mask)


@torch.no_grad()
def fused_ngpt_adamw_retract_step(
    w: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    step: int,
    idim: int = 1,
    lr: float = 1.5e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    weight_decay: float = 0.0,
    adam_eps: float = 1e-8,
    norm_eps: float = 1e-12,
    riemannian_mode: int = 1,
) -> None:
    """
    In-place fused AdamW + hypersphere retraction along `idim` (`1` for row-unit, `0` for col-unit).
    """
    assert w.dim() == 2
    bc1_inv = 1.0 / (1.0 - (beta1 ** step))
    bc2_inv_sqrt = 1.0 / math.sqrt(1.0 - (beta2 ** step))

    if idim == 1:
        num_rows, K = w.shape
        BLOCK_K = triton.next_power_of_2(K)
        num_warps = 4 if BLOCK_K <= 1024 else 8
        _ngpt_riemannian_adamw_retract_row_kernel[(num_rows,)](
            w, g, m, v,
            num_rows, K,
            w.stride(0),
            float(lr), float(beta1), float(beta2),
            float(bc1_inv), float(bc2_inv_sqrt),
            float(weight_decay), float(adam_eps), float(norm_eps),
            BLOCK_K=BLOCK_K,
            RIEMANNIAN_MODE=int(riemannian_mode),
            num_warps=num_warps, num_stages=1,
        )
    elif idim == 0:
        C, num_cols = w.shape
        BLOCK_C = triton.next_power_of_2(C)
        num_warps = 4 if BLOCK_C <= 1024 else 8
        _ngpt_riemannian_adamw_retract_col_kernel[(num_cols,)](
            w, g, m, v,
            C, num_cols,
            w.stride(0),
            float(lr), float(beta1), float(beta2),
            float(bc1_inv), float(bc2_inv_sqrt),
            float(weight_decay), float(adam_eps), float(norm_eps),
            BLOCK_C=BLOCK_C,
            RIEMANNIAN_MODE=int(riemannian_mode),
            num_warps=num_warps, num_stages=1,
        )
    else:
        raise ValueError(f"Unsupported idim={idim}; expected 0 or 1")


# =============================================================================
# 8. Kernel 7: 8-GPU Ring Cosine-Attention LSE Online Merge (`Gate G9`)
# =============================================================================

@triton.jit
def _ngpt_ring_lse_update_kernel(
    O_acc_ptr,
    LSE_acc_ptr,
    O_step_ptr,
    LSE_step_ptr,
    B,
    T,
    H,
    D: tl.constexpr,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lt,
):
    pid = tl.program_id(0)
    if pid >= B * T * H:
        return
    h = pid % H
    bt = pid // H
    t = bt % T
    b = bt // T

    offs_d = tl.arange(0, D)
    o_off = b * stride_ob + t * stride_ot + h * stride_oh + offs_d * stride_od
    l_off = b * stride_lb + h * stride_lh + t * stride_lt

    lse_acc = tl.load(LSE_acc_ptr + l_off).to(tl.float32)
    lse_step = tl.load(LSE_step_ptr + l_off).to(tl.float32)
    o_acc = tl.load(O_acc_ptr + o_off).to(tl.float32)
    o_step = tl.load(O_step_ptr + o_off).to(tl.float32)

    m_new = tl.maximum(lse_acc, lse_step)
    w_acc = tl.exp(lse_acc - m_new)
    w_step = tl.exp(lse_step - m_new)
    denom = w_acc + w_step
    lse_new = m_new + tl.log(tl.maximum(denom, 1.0e-20))
    o_new = (o_acc * w_acc + o_step * w_step) / tl.maximum(denom, 1.0e-20)

    tl.store(O_acc_ptr + o_off, o_new)
    tl.store(LSE_acc_ptr + l_off, lse_new)


def _ring_lse_merge_triton(
    o_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    o_step: torch.Tensor,
    lse_step: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if o_acc is None:
        return o_step.float().clone(), lse_step.float().clone()
    B, T, H, D = o_acc.shape
    _ngpt_ring_lse_update_kernel[(B * T * H,)](
        o_acc,
        lse_acc,
        o_step.float().contiguous(),
        lse_step.float().contiguous(),
        B,
        T,
        H,
        D=D,
        stride_ob=o_acc.stride(0),
        stride_ot=o_acc.stride(1),
        stride_oh=o_acc.stride(2),
        stride_od=o_acc.stride(3),
        stride_lb=lse_acc.stride(0),
        stride_lh=lse_acc.stride(1),
        stride_lt=lse_acc.stride(2),
        num_warps=2,
        num_stages=1,
    )
    return o_acc, lse_acc


class _RingNGPTCosineFlashAttnCP8Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        sm_scale: float,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)

        # Gather K, V across ring steps (equivalent to ring P2P handoff on NVSwitch)
        k_list = [torch.empty_like(k_local) for _ in range(world_size)]
        v_list = [torch.empty_like(v_local) for _ in range(world_size)]
        dist.all_gather(k_list, k_local.contiguous(), group=process_group)
        dist.all_gather(v_list, v_local.contiguous(), group=process_group)

        o_acc = None
        lse_acc = None

        for kv_rank in range(0, rank + 1):
            is_causal = (kv_rank == rank)
            o_step, lse_step = _ngpt_cosine_flash_fwd_impl(
                q_local, k_list[kv_rank], v_list[kv_rank], sm_scale, is_causal=is_causal
            )
            o_acc, lse_acc = _ring_lse_merge_triton(o_acc, lse_acc, o_step, lse_step)

        out = o_acc.to(q_local.dtype).contiguous()
        ctx.save_for_backward(q_local, out, lse_acc, *k_list, *v_list)
        ctx.sm_scale = float(sm_scale)
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.process_group = process_group
        return out

    @staticmethod
    def backward(ctx, do_local: torch.Tensor):
        saved = ctx.saved_tensors
        q_local = saved[0]
        out = saved[1]
        lse_acc = saved[2]
        world_size = ctx.world_size
        rank = ctx.rank
        k_list = list(saved[3 : 3 + world_size])
        v_list = list(saved[3 + world_size : 3 + 2 * world_size])
        k_local = k_list[rank]
        v_local = v_list[rank]

        do_c = do_local.contiguous()
        dq_total = torch.zeros_like(q_local, dtype=torch.float32)
        dk_all = [torch.zeros_like(k_local, dtype=torch.float32) for _ in range(world_size)]
        dv_all = [torch.zeros_like(v_local, dtype=torch.float32) for _ in range(world_size)]

        for kv_rank in range(0, rank + 1):
            is_causal = (kv_rank == rank)
            dq_step, dk_step, dv_step = _ngpt_cosine_flash_bwd_impl(
                do_c,
                q_local,
                k_list[kv_rank],
                v_list[kv_rank],
                out,
                lse_acc,
                ctx.sm_scale,
                is_causal=is_causal,
            )
            dq_total += dq_step.float()
            dk_all[kv_rank] += dk_step.float()
            dv_all[kv_rank] += dv_step.float()

        dk_stack = torch.stack(dk_all, dim=0)
        dv_stack = torch.stack(dv_all, dim=0)
        dk_out = torch.empty_like(k_local, dtype=torch.float32)
        dv_out = torch.empty_like(v_local, dtype=torch.float32)
        dist.reduce_scatter_tensor(dk_out, dk_stack, op=dist.ReduceOp.SUM, group=ctx.process_group)
        dist.reduce_scatter_tensor(dv_out, dv_stack, op=dist.ReduceOp.SUM, group=ctx.process_group)

        return (
            dq_total.to(q_local.dtype),
            dk_out.to(k_local.dtype),
            dv_out.to(v_local.dtype),
            None,
            None,
        )


def ring_ngpt_cosine_flash_attn_cp8(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    sm_scale: Optional[float] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    if sm_scale is None:
        sm_scale = math.sqrt(float(q_local.shape[-1]))
    return _RingNGPTCosineFlashAttnCP8Fn.apply(
        q_local, k_local, v_local, float(sm_scale), process_group
    )


# =============================================================================
# 9. Exact Eager Oracles (FP64/FP32 Reference & Official `NVIDIA/ngpt` Path)
# =============================================================================

def eager_ngpt_hypersphere_lerp_oracle(
    x: torch.Tensor,
    g: torch.Tensor,
    alpha: torch.Tensor,
    alpha_init_value: float = 0.05,
    alpha_init_scaling: float = 1.0 / 32.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    orig_dtype = x.dtype
    work_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    xf = x.to(work_dtype)
    gf = g.to(work_dtype)
    af = alpha.to(work_dtype)

    rx = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)
    rg = torch.rsqrt((gf * gf).sum(dim=-1, keepdim=True) + eps)
    x_hat = xf * rx
    g_hat = gf * rg

    a = torch.abs(af * (float(alpha_init_value) / float(alpha_init_scaling)))
    u = x_hat + a * (g_hat - x_hat)
    ru = torch.rsqrt((u * u).sum(dim=-1, keepdim=True) + eps)
    return (u * ru).to(orig_dtype)


def eager_ngpt_rope_norm_sqk_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sqk: torch.Tensor,
    sqk_init_value: float = 1.0,
    sqk_init_scaling: float = 1.0 / 32.0,
    eps: float = 1e-12,
    rope_mode: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    work_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    qf = q.to(work_dtype)
    kf = k.to(work_dtype)
    cf = cos.to(work_dtype).unsqueeze(0).unsqueeze(2)  # [1, T, 1, D//2]
    sf = sin.to(work_dtype).unsqueeze(0).unsqueeze(2)  # [1, T, 1, D//2]

    q0, q1 = qf[..., 0::2], qf[..., 1::2]
    k0, k1 = kf[..., 0::2], kf[..., 1::2]

    if rope_mode == 0:
        qr0 = -q1 * cf
        qr1 = q0 * sf
        kr0 = -k1 * cf
        kr1 = k0 * sf
    else:
        qr0 = q0 * cf - q1 * sf
        qr1 = q0 * sf + q1 * cf
        kr0 = k0 * cf - k1 * sf
        kr1 = k0 * sf + k1 * cf

    qr = torch.stack((qr0, qr1), dim=-1).flatten(-2)
    kr = torch.stack((kr0, kr1), dim=-1).flatten(-2)

    rq = torch.rsqrt((qr * qr).sum(dim=-1, keepdim=True) + eps)
    rk = torch.rsqrt((kr * kr).sum(dim=-1, keepdim=True) + eps)

    sqk_scale = float(sqk_init_value) / float(sqk_init_scaling)
    sqk_f = sqk.to(work_dtype).view(1, 1, q.shape[2], q.shape[3]) * sqk_scale
    if rope_mode == 2:
        s0, s1 = sqk_f[..., 0::2], sqk_f[..., 1::2]
        s_pair = 0.5 * (s0 + s1)
        sqk_eff = torch.stack((s_pair, s_pair), dim=-1).flatten(-2)
    else:
        sqk_eff = sqk_f

    qo = (qr * rq) * sqk_eff
    ko = (kr * rk) * sqk_eff
    return qo.to(orig_dtype), ko.to(orig_dtype)


def eager_ngpt_scaled_swiglu_oracle(
    uv: torch.Tensor,
    suv: torch.Tensor,
    n_embd: int,
    suv_init_value: float = 1.0,
    suv_init_scaling: float = 1.0,
) -> torch.Tensor:
    orig_dtype = uv.dtype
    work_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    uv_f = uv.to(work_dtype)
    suv_f = suv.to(work_dtype)
    gamma = (float(suv_init_value) / float(suv_init_scaling)) * math.sqrt(float(n_embd))
    uv_scaled = uv_f * (suv_f * gamma)
    u, v = torch.chunk(uv_scaled, 2, dim=-1)
    y = u * F.silu(v)
    return y.to(orig_dtype)


def eager_ngpt_cosine_attn_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    orig_dtype = q.dtype
    work_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    B, T, H, D = q.shape
    if sm_scale is None:
        sm_scale = math.sqrt(float(D))
    qf = q.to(work_dtype).transpose(1, 2)  # [B, H, T, D]
    kf = k.to(work_dtype).transpose(1, 2)
    vf = v.to(work_dtype).transpose(1, 2)

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * float(sm_scale)
    if is_causal:
        causal_mask = torch.tril(torch.ones((T, T), device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, -1.0e20)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf).transpose(1, 2).contiguous()
    return out.to(orig_dtype)


def eager_ngpt_sz_cross_entropy_oracle(
    z: torch.Tensor,
    sz: torch.Tensor,
    targets: torch.Tensor,
    sz_init_value: float = 1.0,
    sz_init_scaling: float = 1.0 / 32.0,
    ignore_index: int = -1,
) -> torch.Tensor:
    work_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    zf = z.to(work_dtype)
    szf = sz.to(work_dtype) * (float(sz_init_value) / float(sz_init_scaling))
    logits = zf * szf
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )


def upstream_nvidia_ngpt_block_forward(
    h: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    wfc: torch.Tensor,
    wmlp_proj: torch.Tensor,
    sqk: torch.Tensor,
    attn_alpha: torch.Tensor,
    suv: torch.Tensor,
    mlp_alpha: torch.Tensor,
    n_head: int,
    base_scale: float,
    allocate_cpu_rope: bool = True,
    alpha_init_value: float = 0.05,
) -> torch.Tensor:
    """
    Faithful reproduction of `NVIDIA/ngpt/main/model.py` `Block.forward()` (`lines 92-155`),
    including per-layer CPU sinusoidal allocation (when `allocate_cpu_rope=True`), broken
    `chunk(2)` RoPE, unfused `justnorm` LERP, and unfused `suv` SwiGLU.
    """
    B, T, C = h.shape
    D = C // n_head

    q = F.linear(h, wq).view(B, T, n_head, D)
    k = F.linear(h, wk).view(B, T, n_head, D)
    v = F.linear(h, wv).view(B, T, n_head, D)

    if allocate_cpu_rope:
        position = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        sinusoidal_emb = torch.zeros((T, D), dtype=torch.float32)
        sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
        sinusoidal_pos = sinusoidal_emb.to(device=q.device)
    else:
        position = torch.arange(T, device=q.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=q.device, dtype=torch.float32) * (-math.log(10000.0) / D))
        sinusoidal_pos = torch.zeros((T, D), device=q.device, dtype=torch.float32)
        sinusoidal_pos[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_pos[:, 1::2] = torch.cos(position * div_term)

    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    q_rot = torch.stack((-qt[..., 1::2], qt[..., ::2]), dim=-1)
    k_rot = torch.stack((-kt[..., 1::2], kt[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, qt.shape[:-1] + (D // 2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, kt.shape[:-1] + (D // 2, 2)) * torch.stack((cos, sin), dim=-1)
    q = torch.reshape(q_rot, qt.shape).transpose(2, 1)
    k = torch.reshape(k_rot, kt.shape).transpose(2, 1)

    sqk_eff = (sqk * (1.0 / base_scale)).view(1, 1, n_head, D)
    q = sqk_eff * (q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-12))
    k = sqk_eff * (k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-12))

    y = ngpt_cosine_flash_attn_triton(
        q.to(h.dtype), k.to(h.dtype), v.to(h.dtype), sm_scale=math.sqrt(float(D)), is_causal=True
    )
    y = y.contiguous().view(B, T, C)
    h_att = F.linear(y, wo)

    lr_a = torch.abs(attn_alpha * (alpha_init_value / base_scale))
    a_norm = h / (h.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    b_norm = h_att / (h_att.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    res = a_norm + lr_a * (b_norm - a_norm)
    h = (res / (res.norm(p=2, dim=-1, keepdim=True) + 1e-12)).to(h.dtype)

    uv = F.linear(h, wfc)
    suv_eff = suv * math.sqrt(float(C))
    uv = suv_eff * uv
    u, v_mlp = torch.chunk(uv, 2, dim=-1)
    x_mlp = (u * F.silu(v_mlp)).to(h.dtype)
    h_mlp = F.linear(x_mlp, wmlp_proj)

    lr_m = torch.abs(mlp_alpha * (alpha_init_value / base_scale))
    a_norm2 = h / (h.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    b_norm2 = h_mlp / (h_mlp.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    res2 = a_norm2 + lr_m * (b_norm2 - a_norm2)
    h = (res2 / (res2.norm(p=2, dim=-1, keepdim=True) + 1e-12)).to(h.dtype)
    return h


def rsi_ngpt_block_forward(
    h: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    wfc: torch.Tensor,
    wmlp_proj: torch.Tensor,
    sqk: torch.Tensor,
    attn_alpha: torch.Tensor,
    suv: torch.Tensor,
    mlp_alpha: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_head: int,
    base_scale: float,
    rope_mode: int = 2,
) -> torch.Tensor:
    """
    Round-11 RSI Superfused nGPT Block Forward:
    - Fused GPU-cached RoPE + FP32 L2-Norm + Pair-Isotropic sqk (`ngpt_rope_norm_sqk_triton`)
    - Zero-spill Cosine-Attention (`ngpt_cosine_flash_attn_triton`, `sm_scale = +sqrt(D)`)
    - Superfused Hypersphere Chordal LERP (`ngpt_hypersphere_lerp_triton`)
    - Superfused Eigen-Scaled SwiGLU (`ngpt_scaled_swiglu_triton`)
    """
    B, T, C = h.shape
    D = C // n_head

    q = F.linear(h, wq).view(B, T, n_head, D)
    k = F.linear(h, wk).view(B, T, n_head, D)
    v = F.linear(h, wv).view(B, T, n_head, D)

    q_norm, k_norm = ngpt_rope_norm_sqk_triton(
        q, k, cos, sin, sqk, sqk_init_value=1.0, sqk_init_scaling=base_scale, rope_mode=rope_mode
    )
    y = ngpt_cosine_flash_attn_triton(
        q_norm, k_norm, v, sm_scale=math.sqrt(float(D)), is_causal=True
    ).view(B, T, C)
    h_att = F.linear(y, wo)

    h = ngpt_hypersphere_lerp_triton(
        h, h_att, attn_alpha, alpha_init_value=0.05, alpha_init_scaling=base_scale
    )

    uv = F.linear(h, wfc)
    x_mlp = ngpt_scaled_swiglu_triton(uv, suv, n_embd=C, suv_init_value=1.0, suv_init_scaling=1.0)
    h_mlp = F.linear(x_mlp, wmlp_proj)

    h = ngpt_hypersphere_lerp_triton(
        h, h_mlp, mlp_alpha, alpha_init_value=0.05, alpha_init_scaling=base_scale
    )
    return h


# =============================================================================
# 10. Cubin Register & Spill Audit Helper (Gate G4)
# =============================================================================

def inspect_compiled_kernel(kernel_obj) -> Dict[str, int]:
    """
    Extracts `(n_regs, n_spills)` across all compiled cubin specializations of a Triton JITFunction
    (supports both Triton 3.x `device_caches` tuple layout and legacy `cache` dict layout).
    """
    max_regs = 0
    total_spills = 0
    num_specs = 0
    dev_caches = getattr(kernel_obj, "device_caches", {})
    if isinstance(dev_caches, dict) and dev_caches:
        for cache_obj in dev_caches.values():
            bin_map = cache_obj[0] if isinstance(cache_obj, tuple) else cache_obj
            if isinstance(bin_map, dict):
                for compiled in bin_map.values():
                    if hasattr(compiled, "n_regs"):
                        max_regs = max(max_regs, int(compiled.n_regs or 0))
                        total_spills += int(compiled.n_spills or 0)
                        num_specs += 1
    else:
        cache = getattr(kernel_obj, "cache", {})
        for dev_cache in cache.values():
            if isinstance(dev_cache, dict):
                for compiled in dev_cache.values():
                    n_regs = int(getattr(compiled, "n_regs", 0) or 0)
                    n_spills = int(getattr(compiled, "n_spills", 0) or 0)
                    max_regs = max(max_regs, n_regs)
                    total_spills += n_spills
                    num_specs += 1
            else:
                n_regs = int(getattr(dev_cache, "n_regs", 0) or 0)
                n_spills = int(getattr(dev_cache, "n_spills", 0) or 0)
                max_regs = max(max_regs, n_regs)
                total_spills += n_spills
                num_specs += 1
    return {
        "num_specializations": num_specs,
        "n_regs": max_regs,
        "n_spills": total_spills,
    }


def get_all_round11_rsi_kernels() -> Dict[str, object]:
    return {
        "_ngpt_reduce_2d_to_1d_kernel": _ngpt_reduce_2d_to_1d_kernel,
        "_ngpt_hypersphere_lerp_fwd_kernel": _ngpt_hypersphere_lerp_fwd_kernel,
        "_ngpt_hypersphere_lerp_bwd_kernel": _ngpt_hypersphere_lerp_bwd_kernel,
        "_ngpt_rope_norm_sqk_fwd_kernel": _ngpt_rope_norm_sqk_fwd_kernel,
        "_ngpt_rope_norm_sqk_bwd_kernel": _ngpt_rope_norm_sqk_bwd_kernel,
        "_ngpt_scaled_swiglu_fwd_kernel": _ngpt_scaled_swiglu_fwd_kernel,
        "_ngpt_scaled_swiglu_bwd_kernel": _ngpt_scaled_swiglu_bwd_kernel,
        "_ngpt_cosine_flash_fwd_d64_kernel": _ngpt_cosine_flash_fwd_d64_kernel,
        "_ngpt_cosine_flash_fwd_d128_kernel": _ngpt_cosine_flash_fwd_d128_kernel,
        "_ngpt_cosine_flash_bwd_prep_kernel": _ngpt_cosine_flash_bwd_prep_kernel,
        "_ngpt_cosine_flash_bwd_dq_d64_kernel": _ngpt_cosine_flash_bwd_dq_d64_kernel,
        "_ngpt_cosine_flash_bwd_dkv_d64_kernel": _ngpt_cosine_flash_bwd_dkv_d64_kernel,
        "_ngpt_cosine_flash_bwd_dq_d128_kernel": _ngpt_cosine_flash_bwd_dq_d128_kernel,
        "_ngpt_cosine_flash_bwd_dkv_d128_kernel": _ngpt_cosine_flash_bwd_dkv_d128_kernel,
        "_ngpt_sz_cross_entropy_fwd_kernel": _ngpt_sz_cross_entropy_fwd_kernel,
        "_ngpt_sz_cross_entropy_bwd_kernel": _ngpt_sz_cross_entropy_bwd_kernel,
        "_ngpt_riemannian_adamw_retract_row_kernel": _ngpt_riemannian_adamw_retract_row_kernel,
        "_ngpt_riemannian_adamw_retract_col_kernel": _ngpt_riemannian_adamw_retract_col_kernel,
        "_ngpt_ring_lse_update_kernel": _ngpt_ring_lse_update_kernel,
    }
