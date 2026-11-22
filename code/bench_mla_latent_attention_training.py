from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.checkpoint import checkpoint
import triton
import triton.language as tl

from flash_attn import flash_attn_func
import flashinfer
from common import environment_metadata

try:
    import pynvml  # type: ignore

    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False

RCP_LN2 = 1.4426950408889634


# ============================================================================
# 1. Custom Fused Triton Kernel: Latent RMSNorm(512) + Decoupled RoPE(64)
# ============================================================================


@triton.jit
def _fused_latent_rmsnorm_rope_fwd_kernel(
    Raw_ptr,
    W_ptr,
    Cos_ptr,
    Sin_ptr,
    Cnorm_ptr,
    Krope_ptr,
    Rstd_ptr,
    eps,
    T: tl.int32,
):
    row = tl.program_id(0)
    if row >= T:
        return
    d512 = tl.arange(0, 512)
    d32 = tl.arange(0, 32)

    raw_base = Raw_ptr + row.to(tl.int64) * 576
    c_raw = tl.load(raw_base + d512).to(tl.float32)
    w = tl.load(W_ptr + d512).to(tl.float32)

    var = tl.sum(c_raw * c_raw, axis=0) * (1.0 / 512.0)
    rstd = tl.math.rsqrt(var + eps)
    c_out = (c_raw * rstd) * w

    tl.store(Cnorm_ptr + row.to(tl.int64) * 512 + d512, c_out.to(tl.bfloat16))
    tl.store(Rstd_ptr + row, rstd)

    k1 = tl.load(raw_base + 512 + d32).to(tl.float32)
    k2 = tl.load(raw_base + 544 + d32).to(tl.float32)
    cos = tl.load(Cos_ptr + row.to(tl.int64) * 32 + d32).to(tl.float32)
    sin = tl.load(Sin_ptr + row.to(tl.int64) * 32 + d32).to(tl.float32)

    kr1 = k1 * cos - k2 * sin
    kr2 = k1 * sin + k2 * cos
    kr_base = Krope_ptr + row.to(tl.int64) * 64
    tl.store(kr_base + d32, kr1.to(tl.bfloat16))
    tl.store(kr_base + 32 + d32, kr2.to(tl.bfloat16))


@triton.jit
def _fused_latent_rmsnorm_rope_bwd_kernel(
    dC_ptr,
    dKr_ptr,
    Raw_ptr,
    W_ptr,
    Rstd_ptr,
    Cos_ptr,
    Sin_ptr,
    dRaw_ptr,
    dW_part_ptr,
    T: tl.int32,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    r_start = pid * ROWS_PER_PROG
    d512 = tl.arange(0, 512)
    d32 = tl.arange(0, 32)
    w = tl.load(W_ptr + d512).to(tl.float32)
    dw_acc = tl.zeros([512], dtype=tl.float32)

    for step in range(0, ROWS_PER_PROG):
        row = r_start + step
        if row < T:
            raw_base = Raw_ptr + row.to(tl.int64) * 576
            c_raw = tl.load(raw_base + d512).to(tl.float32)
            dc = tl.load(dC_ptr + row.to(tl.int64) * 512 + d512).to(tl.float32)
            rstd = tl.load(Rstd_ptr + row).to(tl.float32)

            x_hat = c_raw * rstd
            dw_acc += dc * x_hat
            dy_w = dc * w
            mean_dot = tl.sum(dy_w * x_hat, axis=0) * (1.0 / 512.0)
            dx_c = rstd * (dy_w - x_hat * mean_dot)

            dkr_base = dKr_ptr + row.to(tl.int64) * 64
            dkr1 = tl.load(dkr_base + d32).to(tl.float32)
            dkr2 = tl.load(dkr_base + 32 + d32).to(tl.float32)
            cos = tl.load(Cos_ptr + row.to(tl.int64) * 32 + d32).to(tl.float32)
            sin = tl.load(Sin_ptr + row.to(tl.int64) * 32 + d32).to(tl.float32)

            dk1 = dkr1 * cos + dkr2 * sin
            dk2 = -dkr1 * sin + dkr2 * cos

            draw_base = dRaw_ptr + row.to(tl.int64) * 576
            tl.store(draw_base + d512, dx_c.to(tl.bfloat16))
            tl.store(draw_base + 512 + d32, dk1.to(tl.bfloat16))
            tl.store(draw_base + 544 + d32, dk2.to(tl.bfloat16))

    tl.store(dW_part_ptr + pid.to(tl.int64) * 512 + d512, dw_acc)


class _FusedLatentRMSNormRoPEAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        raw_ckv_kr: torch.Tensor,
        w_norm: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = raw_ckv_kr.shape[:-1]
        raw_2d = raw_ckv_kr.contiguous().view(-1, 576)
        T = raw_2d.shape[0]
        cos_2d = cos.contiguous().view(T, 32)
        sin_2d = sin.contiguous().view(T, 32)
        w_c = w_norm.contiguous()

        c_norm = torch.empty((T, 512), device=raw_2d.device, dtype=torch.bfloat16)
        k_rope = torch.empty((T, 64), device=raw_2d.device, dtype=torch.bfloat16)
        rstd = torch.empty((T,), device=raw_2d.device, dtype=torch.float32)

        _fused_latent_rmsnorm_rope_fwd_kernel[(T,)](
            raw_2d, w_c, cos_2d, sin_2d, c_norm, k_rope, rstd, eps, T, num_warps=4
        )
        ctx.save_for_backward(raw_2d, w_c, rstd, cos_2d, sin_2d)
        ctx.orig_shape = orig_shape
        return c_norm.view(*orig_shape, 512), k_rope.view(*orig_shape, 64)

    @staticmethod
    def backward(
        ctx: Any, dc_norm: torch.Tensor, dk_rope: torch.Tensor
    ) -> tuple[Any, ...]:
        raw_2d, w_c, rstd, cos_2d, sin_2d = ctx.saved_tensors
        T = raw_2d.shape[0]
        dc_2d = dc_norm.contiguous().view(T, 512)
        dkr_2d = dk_rope.contiguous().view(T, 64)

        rows_per_prog = 16
        n_progs = triton.cdiv(T, rows_per_prog)
        d_raw = torch.empty_like(raw_2d)
        dw_part = torch.empty((n_progs, 512), device=raw_2d.device, dtype=torch.float32)

        _fused_latent_rmsnorm_rope_bwd_kernel[(n_progs,)](
            dc_2d,
            dkr_2d,
            raw_2d,
            w_c,
            rstd,
            cos_2d,
            sin_2d,
            d_raw,
            dw_part,
            T,
            ROWS_PER_PROG=rows_per_prog,
            num_warps=4,
        )
        dw = dw_part.sum(dim=0).to(w_c.dtype)
        return d_raw.view(*ctx.orig_shape, 576), dw, None, None, None


def fused_latent_rmsnorm_rope(
    raw_ckv_kr: torch.Tensor,
    w_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _FusedLatentRMSNormRoPEAutograd.apply(raw_ckv_kr, w_norm, cos, sin, eps)


def build_rope_cache(
    positions: torch.Tensor, dim: int = 64, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """`positions`: `[B, S]` -> returns `(cos, sin)` each `[B, S, dim // 2]` in `FP32`."""
    half = dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=positions.device, dtype=torch.float32) / half)
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq[None, None, :]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope_bhsd(
    x_bhsd: torch.Tensor, cos_bs: torch.Tensor, sin_bs: torch.Tensor
) -> torch.Tensor:
    """Applies 1D RoPE to `[B, H, S, 64]` using `cos_bs, sin_bs` `[B, S, 32]`."""
    c = cos_bs.unsqueeze(1)
    s = sin_bs.unsqueeze(1)
    xf = x_bhsd.float()
    x1, x2 = xf[..., :32], xf[..., 32:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x_bhsd.dtype)


# ============================================================================
# 2. Custom Split-Dimension Triton Flash-MLA (128 NoPE + 64 Shared-RoPE -> 128 V)
# ============================================================================


@triton.jit
def _mla_split_fwd_inner(
    acc,
    l_i,
    m_i,
    q_n,
    q_r,
    Kn_base,
    Kr_base,
    V_base,
    start_m,
    qk_scale,
    S_kv,
    BM: tl.constexpr,
    BN: tl.constexpr,
    MASK: tl.constexpr,
):
    lo = start_m * BM if MASK else 0
    hi = (start_m + 1) * BM if MASK else start_m * BM
    hi = tl.minimum(hi, S_kv)
    offs_m = start_m * BM + tl.arange(0, BM)
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    for start_n in range(lo, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        kn = tl.load(Kn_base + offs_n[:, None] * 128 + d128[None, :])
        kr = tl.load(Kr_base + offs_n[:, None] * 64 + d64[None, :])
        qk = (tl.dot(q_n, tl.trans(kn)) + tl.dot(q_r, tl.trans(kr))) * qk_scale
        if MASK:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -1.0e20)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V_base + offs_n[:, None] * 128 + d128[None, :])
        acc += tl.dot(p.to(tl.bfloat16), v)
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _mla_split_fwd_kernel(
    Qn_ptr,
    Qr_ptr,
    Kn_ptr,
    Kr_ptr,
    V_ptr,
    Out_ptr,
    Lse2_ptr,
    qk_scale,
    n_h: tl.int32,
    S_q: tl.int32,
    S_kv: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh_id = tl.program_id(1)
    b_id = bh_id // n_h

    offs_m = start_m * BM + tl.arange(0, BM)
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    qn_base = Qn_ptr + bh_id.to(tl.int64) * S_q * 128
    qr_base = Qr_ptr + bh_id.to(tl.int64) * S_q * 64
    kn_base = Kn_ptr + bh_id.to(tl.int64) * S_kv * 128
    kr_base = Kr_ptr + b_id.to(tl.int64) * S_kv * 64
    v_base = V_ptr + bh_id.to(tl.int64) * S_kv * 128

    q_n = tl.load(qn_base + offs_m[:, None] * 128 + d128[None, :])
    q_r = tl.load(qr_base + offs_m[:, None] * 64 + d64[None, :])

    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, 128], dtype=tl.float32)

    if CAUSAL:
        acc, l_i, m_i = _mla_split_fwd_inner(
            acc,
            l_i,
            m_i,
            q_n,
            q_r,
            kn_base,
            kr_base,
            v_base,
            start_m,
            qk_scale,
            S_kv,
            BM=BM,
            BN=BN,
            MASK=False,
        )
        acc, l_i, m_i = _mla_split_fwd_inner(
            acc,
            l_i,
            m_i,
            q_n,
            q_r,
            kn_base,
            kr_base,
            v_base,
            start_m,
            qk_scale,
            S_kv,
            BM=BM,
            BN=BN,
            MASK=True,
        )
    else:
        for start_n in range(0, S_kv, BN):
            offs_n = start_n + tl.arange(0, BN)
            kn = tl.load(kn_base + offs_n[:, None] * 128 + d128[None, :])
            kr = tl.load(kr_base + offs_n[:, None] * 64 + d64[None, :])
            qk = (tl.dot(q_n, tl.trans(kn)) + tl.dot(q_r, tl.trans(kr))) * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16),
                tl.load(v_base + offs_n[:, None] * 128 + d128[None, :]),
            )
            m_i = m_ij

    acc = acc / l_i[:, None]
    lse2 = m_i + tl.math.log2(l_i)
    tl.store(
        Out_ptr + bh_id.to(tl.int64) * S_q * 128 + offs_m[:, None] * 128 + d128[None, :],
        acc.to(tl.bfloat16),
    )
    tl.store(Lse2_ptr + bh_id.to(tl.int64) * S_q + offs_m, lse2)


@triton.jit
def _mla_bwd_prep_kernel(
    Out_ptr, dO_ptr, Delta_ptr, N_rows: tl.int32, D_V: tl.constexpr
):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)
    mask = row < N_rows
    if D_V == 128:
        d_offs = tl.arange(0, 128)
        o = tl.load(
            Out_ptr + row[:, None].to(tl.int64) * 128 + d_offs[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            dO_ptr + row[:, None].to(tl.int64) * 128 + d_offs[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(o * do, 1)
    else:
        d_offs = tl.arange(0, 128)
        acc = tl.zeros([64], dtype=tl.float32)
        for s_off in tl.static_range(0, 512, 128):
            cols = s_off + d_offs
            o = tl.load(
                Out_ptr + row[:, None].to(tl.int64) * 512 + cols[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            do = tl.load(
                dO_ptr + row[:, None].to(tl.int64) * 512 + cols[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(o * do, 1)
    tl.store(Delta_ptr + row, acc, mask=mask)


@triton.jit
def _mla_split_bwd_dkdv_kernel(
    Qn_ptr,
    Qr_ptr,
    Kn_ptr,
    Kr_ptr,
    V_ptr,
    dO_ptr,
    Lse2_ptr,
    Delta_ptr,
    dKn_ptr,
    dKr_part_ptr,
    dV_ptr,
    sm_scale,
    qk_scale,
    n_h: tl.int32,
    S_q: tl.int32,
    S_kv: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CAUSAL: tl.constexpr,
    ACCUM_DKDV: tl.constexpr,
):
    start_n = tl.program_id(0) * BN
    bh_id = tl.program_id(1)
    b_id = bh_id // n_h

    offs_n = start_n + tl.arange(0, BN)
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    kn = tl.load(
        Kn_ptr + bh_id.to(tl.int64) * S_kv * 128 + offs_n[:, None] * 128 + d128[None, :]
    )
    kr = tl.load(
        Kr_ptr + b_id.to(tl.int64) * S_kv * 64 + offs_n[:, None] * 64 + d64[None, :]
    )
    v = tl.load(
        V_ptr + bh_id.to(tl.int64) * S_kv * 128 + offs_n[:, None] * 128 + d128[None, :]
    )

    dkn = tl.zeros([BN, 128], dtype=tl.float32)
    dkr = tl.zeros([BN, 64], dtype=tl.float32)
    dv = tl.zeros([BN, 128], dtype=tl.float32)

    qn_base = Qn_ptr + bh_id.to(tl.int64) * S_q * 128
    qr_base = Qr_ptr + bh_id.to(tl.int64) * S_q * 64
    do_base = dO_ptr + bh_id.to(tl.int64) * S_q * 128
    lse_base = Lse2_ptr + bh_id.to(tl.int64) * S_q
    del_base = Delta_ptr + bh_id.to(tl.int64) * S_q

    if CAUSAL:
        for start_m in range(start_n, tl.minimum(start_n + BN, S_q), BM):
            offs_m = start_m + tl.arange(0, BM)
            qn = tl.load(qn_base + offs_m[:, None] * 128 + d128[None, :])
            qr = tl.load(qr_base + offs_m[:, None] * 64 + d64[None, :])
            lse2 = tl.load(lse_base + offs_m)
            delta = tl.load(del_base + offs_m)
            do = tl.load(do_base + offs_m[:, None] * 128 + d128[None, :])
            qkT = (tl.dot(kn, tl.trans(qn)) + tl.dot(kr, tl.trans(qr))) * qk_scale
            pT = tl.math.exp2(qkT - lse2[None, :])
            pT = tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)
            dv += tl.dot(pT.to(tl.bfloat16), do)
            dpT = tl.dot(v, tl.trans(do))
            dsT = (pT * (dpT - delta[None, :])).to(tl.bfloat16)
            dkn += tl.dot(dsT, qn)
            dkr += tl.dot(dsT, qr)
        m_unmasked = start_n + BN
    else:
        m_unmasked = 0

    for start_m in range(m_unmasked, S_q, BM):
        offs_m = start_m + tl.arange(0, BM)
        qn = tl.load(qn_base + offs_m[:, None] * 128 + d128[None, :])
        qr = tl.load(qr_base + offs_m[:, None] * 64 + d64[None, :])
        lse2 = tl.load(lse_base + offs_m)
        delta = tl.load(del_base + offs_m)
        do = tl.load(do_base + offs_m[:, None] * 128 + d128[None, :])
        qkT = (tl.dot(kn, tl.trans(qn)) + tl.dot(kr, tl.trans(qr))) * qk_scale
        pT = tl.math.exp2(qkT - lse2[None, :])
        dv += tl.dot(pT.to(tl.bfloat16), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = (pT * (dpT - delta[None, :])).to(tl.bfloat16)
        dkn += tl.dot(dsT, qn)
        dkr += tl.dot(dsT, qr)

    dkn_off = dKn_ptr + bh_id.to(tl.int64) * S_kv * 128 + offs_n[:, None] * 128 + d128[None, :]
    dkr_off = dKr_part_ptr + bh_id.to(tl.int64) * S_kv * 64 + offs_n[:, None] * 64 + d64[None, :]
    dv_off = dV_ptr + bh_id.to(tl.int64) * S_kv * 128 + offs_n[:, None] * 128 + d128[None, :]

    if ACCUM_DKDV:
        dkn_prev = tl.load(dkn_off).to(tl.float32)
        dkr_prev = tl.load(dkr_off).to(tl.float32)
        dv_prev = tl.load(dv_off).to(tl.float32)
        tl.store(dkn_off, (dkn_prev + dkn * sm_scale).to(tl.bfloat16))
        tl.store(dkr_off, dkr_prev + dkr * sm_scale)
        tl.store(dv_off, (dv_prev + dv).to(tl.bfloat16))
    else:
        tl.store(dkn_off, (dkn * sm_scale).to(tl.bfloat16))
        tl.store(dkr_off, dkr * sm_scale)
        tl.store(dv_off, dv.to(tl.bfloat16))


@triton.jit
def _mla_split_bwd_dq_kernel(
    Qn_ptr,
    Qr_ptr,
    Kn_ptr,
    Kr_ptr,
    V_ptr,
    dO_ptr,
    Lse2_ptr,
    Delta_ptr,
    dQn_ptr,
    dQr_ptr,
    sm_scale,
    qk_scale,
    n_h: tl.int32,
    S_q: tl.int32,
    S_kv: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CAUSAL: tl.constexpr,
    ACCUM_DQ: tl.constexpr,
):
    start_m = tl.program_id(0) * BM
    bh_id = tl.program_id(1)
    b_id = bh_id // n_h

    offs_m = start_m + tl.arange(0, BM)
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    qn = tl.load(
        Qn_ptr + bh_id.to(tl.int64) * S_q * 128 + offs_m[:, None] * 128 + d128[None, :]
    )
    qr = tl.load(
        Qr_ptr + bh_id.to(tl.int64) * S_q * 64 + offs_m[:, None] * 64 + d64[None, :]
    )
    do = tl.load(
        dO_ptr + bh_id.to(tl.int64) * S_q * 128 + offs_m[:, None] * 128 + d128[None, :]
    )
    lse2 = tl.load(Lse2_ptr + bh_id.to(tl.int64) * S_q + offs_m)
    delta = tl.load(Delta_ptr + bh_id.to(tl.int64) * S_q + offs_m)

    dqn = tl.zeros([BM, 128], dtype=tl.float32)
    dqr = tl.zeros([BM, 64], dtype=tl.float32)

    kn_base = Kn_ptr + bh_id.to(tl.int64) * S_kv * 128
    kr_base = Kr_ptr + b_id.to(tl.int64) * S_kv * 64
    v_base = V_ptr + bh_id.to(tl.int64) * S_kv * 128

    n_unmasked = start_m if CAUSAL else S_kv
    for start_n in range(0, n_unmasked, BN):
        offs_n = start_n + tl.arange(0, BN)
        kn = tl.load(kn_base + offs_n[:, None] * 128 + d128[None, :])
        kr = tl.load(kr_base + offs_n[:, None] * 64 + d64[None, :])
        v = tl.load(v_base + offs_n[:, None] * 128 + d128[None, :])
        qk = (tl.dot(qn, tl.trans(kn)) + tl.dot(qr, tl.trans(kr))) * qk_scale
        p = tl.math.exp2(qk - lse2[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(tl.bfloat16)
        dqn += tl.dot(ds, kn)
        dqr += tl.dot(ds, kr)

    if CAUSAL:
        for start_n in range(start_m, tl.minimum(start_m + BM, S_kv), BN):
            offs_n = start_n + tl.arange(0, BN)
            kn = tl.load(kn_base + offs_n[:, None] * 128 + d128[None, :])
            kr = tl.load(kr_base + offs_n[:, None] * 64 + d64[None, :])
            v = tl.load(v_base + offs_n[:, None] * 128 + d128[None, :])
            qk = (tl.dot(qn, tl.trans(kn)) + tl.dot(qr, tl.trans(kr))) * qk_scale
            p = tl.math.exp2(qk - lse2[:, None])
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delta[:, None])).to(tl.bfloat16)
            dqn += tl.dot(ds, kn)
            dqr += tl.dot(ds, kr)

    dqn_off = dQn_ptr + bh_id.to(tl.int64) * S_q * 128 + offs_m[:, None] * 128 + d128[None, :]
    dqr_off = dQr_ptr + bh_id.to(tl.int64) * S_q * 64 + offs_m[:, None] * 64 + d64[None, :]
    if ACCUM_DQ:
        dqn_prev = tl.load(dqn_off).to(tl.float32)
        dqr_prev = tl.load(dqr_off).to(tl.float32)
        tl.store(dqn_off, (dqn_prev + dqn * sm_scale).to(tl.bfloat16))
        tl.store(dqr_off, (dqr_prev + dqr * sm_scale).to(tl.bfloat16))
    else:
        tl.store(dqn_off, (dqn * sm_scale).to(tl.bfloat16))
        tl.store(dqr_off, (dqr * sm_scale).to(tl.bfloat16))


@triton.jit
def _mla_split_bwd_onepass_atomic_control_kernel(
    Qn_ptr,
    Qr_ptr,
    Kn_ptr,
    Kr_ptr,
    V_ptr,
    dO_ptr,
    Lse2_ptr,
    Delta_ptr,
    dQn_acc_ptr,
    dQr_acc_ptr,
    dKn_ptr,
    dKr_part_ptr,
    dV_ptr,
    sm_scale,
    qk_scale,
    n_h: tl.int32,
    S_q: tl.int32,
    S_kv: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Suite 0 negative-control kernel demonstrating L2 atomic-add serialization
    when `S // BN` CTAs atomically accumulate `dQ` in global HBM."""
    start_n = tl.program_id(0) * BN
    bh_id = tl.program_id(1)
    b_id = bh_id // n_h

    offs_n = start_n + tl.arange(0, BN)
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    kn = tl.load(
        Kn_ptr + bh_id.to(tl.int64) * S_kv * 128 + offs_n[:, None] * 128 + d128[None, :]
    )
    kr = tl.load(
        Kr_ptr + b_id.to(tl.int64) * S_kv * 64 + offs_n[:, None] * 64 + d64[None, :]
    )
    v = tl.load(
        V_ptr + bh_id.to(tl.int64) * S_kv * 128 + offs_n[:, None] * 128 + d128[None, :]
    )

    dkn = tl.zeros([BN, 128], dtype=tl.float32)
    dkr = tl.zeros([BN, 64], dtype=tl.float32)
    dv = tl.zeros([BN, 128], dtype=tl.float32)

    qn_base = Qn_ptr + bh_id.to(tl.int64) * S_q * 128
    qr_base = Qr_ptr + bh_id.to(tl.int64) * S_q * 64
    do_base = dO_ptr + bh_id.to(tl.int64) * S_q * 128
    dqn_base = dQn_acc_ptr + bh_id.to(tl.int64) * S_q * 128
    dqr_base = dQr_acc_ptr + bh_id.to(tl.int64) * S_q * 64
    lse_base = Lse2_ptr + bh_id.to(tl.int64) * S_q
    del_base = Delta_ptr + bh_id.to(tl.int64) * S_q

    for start_m in range(start_n, S_q, BM):
        offs_m = start_m + tl.arange(0, BM)
        qn = tl.load(qn_base + offs_m[:, None] * 128 + d128[None, :])
        qr = tl.load(qr_base + offs_m[:, None] * 64 + d64[None, :])
        lse2 = tl.load(lse_base + offs_m)
        delta = tl.load(del_base + offs_m)
        do = tl.load(do_base + offs_m[:, None] * 128 + d128[None, :])

        qkT = (tl.dot(kn, tl.trans(qn)) + tl.dot(kr, tl.trans(qr))) * qk_scale
        pT = tl.math.exp2(qkT - lse2[None, :])
        if start_m == start_n:
            pT = tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)
        dv += tl.dot(pT.to(tl.bfloat16), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = (pT * (dpT - delta[None, :])).to(tl.bfloat16)
        dkn += tl.dot(dsT, qn)
        dkr += tl.dot(dsT, qr)
        tl.atomic_add(
            dqn_base + offs_m[:, None] * 128 + d128[None, :],
            tl.dot(tl.trans(dsT), kn) * sm_scale,
        )
        tl.atomic_add(
            dqr_base + offs_m[:, None] * 64 + d64[None, :],
            tl.dot(tl.trans(dsT), kr) * sm_scale,
        )

    tl.store(
        dKn_ptr + bh_id.to(tl.int64) * S_kv * 128 + offs_n[:, None] * 128 + d128[None, :],
        (dkn * sm_scale).to(tl.bfloat16),
    )
    tl.store(
        dKr_part_ptr + bh_id.to(tl.int64) * S_kv * 64 + offs_n[:, None] * 64 + d64[None, :],
        dkr * sm_scale,
    )
    tl.store(
        dV_ptr + bh_id.to(tl.int64) * S_kv * 128 + offs_n[:, None] * 128 + d128[None, :],
        dv.to(tl.bfloat16),
    )


def _mla_split_fwd_raw(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S_q, _ = q_nope.shape
    S_kv = k_nope.shape[2]
    qn = q_nope.contiguous()
    qr = q_rope.contiguous()
    kn = k_nope.contiguous()
    kr = k_rope.contiguous()
    vc = v.contiguous()

    out = torch.empty_like(vc[:, :, :S_q, :])
    lse2 = torch.empty((B, H, S_q), device=qn.device, dtype=torch.float32)
    bm = 128 if (S_q >= 128 and S_q % 128 == 0) else 64
    nw = 8 if bm == 128 else 4
    _mla_split_fwd_kernel[(S_q // bm, B * H)](
        qn,
        qr,
        kn,
        kr,
        vc,
        out,
        lse2,
        sm_scale * RCP_LN2,
        H,
        S_q,
        S_kv,
        BM=bm,
        BN=64,
        CAUSAL=causal,
        num_warps=nw,
        num_stages=2,
    )
    return out, lse2


def _mla_split_bwd_raw(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    do: torch.Tensor,
    lse2: torch.Tensor,
    sm_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S_q, _ = q_nope.shape
    S_kv = k_nope.shape[2]
    qn = q_nope.contiguous()
    qr = q_rope.contiguous()
    kn = k_nope.contiguous()
    kr = k_rope.contiguous()
    vc = v.contiguous()
    oc = out.contiguous()
    doc = do.contiguous()

    delta = torch.empty((B, H, S_q), device=qn.device, dtype=torch.float32)
    _mla_bwd_prep_kernel[(triton.cdiv(B * H * S_q, 64),)](
        oc, doc, delta, B * H * S_q, D_V=128, num_warps=4
    )

    dqn = torch.empty_like(qn)
    dqr = torch.empty_like(qr)
    dkn = torch.empty_like(kn)
    dkr_part = torch.empty((B, H, S_kv, 64), device=qn.device, dtype=torch.float32)
    dv = torch.empty_like(vc)
    qk_scale = sm_scale * RCP_LN2

    _mla_split_bwd_dkdv_kernel[(S_kv // 64, B * H)](
        qn,
        qr,
        kn,
        kr,
        vc,
        doc,
        lse2,
        delta,
        dkn,
        dkr_part,
        dv,
        sm_scale,
        qk_scale,
        H,
        S_q,
        S_kv,
        BM=64,
        BN=64,
        CAUSAL=causal,
        ACCUM_DKDV=False,
        num_warps=4,
        num_stages=2,
    )
    _mla_split_bwd_dq_kernel[(S_q // 64, B * H)](
        qn,
        qr,
        kn,
        kr,
        vc,
        doc,
        lse2,
        delta,
        dqn,
        dqr,
        sm_scale,
        qk_scale,
        H,
        S_q,
        S_kv,
        BM=64,
        BN=64,
        CAUSAL=causal,
        ACCUM_DQ=False,
        num_warps=4,
        num_stages=2,
    )
    dkr = dkr_part.sum(dim=1, keepdim=True).to(kr.dtype)
    return dqn, dqr, dkn, dkr, dv


class _TritonSplitMLAAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        out, lse2 = _mla_split_fwd_raw(
            q_nope, q_rope, k_nope, k_rope, v, sm_scale, causal=True
        )
        ctx.save_for_backward(q_nope, q_rope, k_nope, k_rope, v, out, lse2)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        q_nope, q_rope, k_nope, k_rope, v, out, lse2 = ctx.saved_tensors
        dqn, dqr, dkn, dkr, dv = _mla_split_bwd_raw(
            q_nope, q_rope, k_nope, k_rope, v, out, do, lse2, ctx.sm_scale, causal=True
        )
        return dqn, dqr, dkn, dkr, dv, None


def triton_split_mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    return _TritonSplitMLAAutograd.apply(q_nope, q_rope, k_nope, k_rope, v, sm_scale)


# ============================================================================
# 3. Custom SM80 SRAM-Safe Weight-Absorbed MQA-Latent Flash-MLA (512 + 64 -> 512)
# ============================================================================


@triton.jit
def _mla_absorbed_fwd_kernel(
    Qa_ptr,
    Qr_ptr,
    Ckv_ptr,
    Kr_ptr,
    Out_ptr,
    Lse2_ptr,
    qk_scale,
    n_h: tl.int32,
    S_q: tl.int32,
    S_kv: tl.int32,
    CAUSAL_OFFSET: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh_id = tl.program_id(1)
    b_id = bh_id // n_h

    offs_m = start_m * BM + tl.arange(0, BM)
    mask_m = offs_m < S_q
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    qa_base = Qa_ptr + bh_id.to(tl.int64) * S_q * 512
    qr_base = Qr_ptr + bh_id.to(tl.int64) * S_q * 64
    ckv_base = Ckv_ptr + b_id.to(tl.int64) * S_kv * 512
    kr_base = Kr_ptr + b_id.to(tl.int64) * S_kv * 64

    qa0 = tl.load(qa_base + offs_m[:, None] * 512 + (0 + d128)[None, :], mask=mask_m[:, None], other=0.0)
    qa1 = tl.load(qa_base + offs_m[:, None] * 512 + (128 + d128)[None, :], mask=mask_m[:, None], other=0.0)
    qa2 = tl.load(qa_base + offs_m[:, None] * 512 + (256 + d128)[None, :], mask=mask_m[:, None], other=0.0)
    qa3 = tl.load(qa_base + offs_m[:, None] * 512 + (384 + d128)[None, :], mask=mask_m[:, None], other=0.0)
    qr = tl.load(qr_base + offs_m[:, None] * 64 + d64[None, :], mask=mask_m[:, None], other=0.0)

    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc0 = tl.zeros([BM, 128], dtype=tl.float32)
    acc1 = tl.zeros([BM, 128], dtype=tl.float32)
    acc2 = tl.zeros([BM, 128], dtype=tl.float32)
    acc3 = tl.zeros([BM, 128], dtype=tl.float32)

    hi = tl.minimum(CAUSAL_OFFSET + (start_m + 1) * BM, S_kv) if CAUSAL else S_kv
    for start_n in range(0, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        c0 = tl.load(ckv_base + offs_n[:, None] * 512 + (0 + d128)[None, :])
        c1 = tl.load(ckv_base + offs_n[:, None] * 512 + (128 + d128)[None, :])
        c2 = tl.load(ckv_base + offs_n[:, None] * 512 + (256 + d128)[None, :])
        c3 = tl.load(ckv_base + offs_n[:, None] * 512 + (384 + d128)[None, :])
        kr = tl.load(kr_base + offs_n[:, None] * 64 + d64[None, :])

        qk = (
            tl.dot(qa0, tl.trans(c0))
            + tl.dot(qa1, tl.trans(c1))
            + tl.dot(qa2, tl.trans(c2))
            + tl.dot(qa3, tl.trans(c3))
            + tl.dot(qr, tl.trans(kr))
        ) * qk_scale
        if CAUSAL:
            if start_n >= CAUSAL_OFFSET + start_m * BM:
                qk = tl.where(
                    (CAUSAL_OFFSET + offs_m[:, None]) >= offs_n[None, :],
                    qk,
                    -1.0e20,
                )

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        pb = p.to(tl.bfloat16)
        acc0 = acc0 * alpha[:, None] + tl.dot(pb, c0)
        acc1 = acc1 * alpha[:, None] + tl.dot(pb, c1)
        acc2 = acc2 * alpha[:, None] + tl.dot(pb, c2)
        acc3 = acc3 * alpha[:, None] + tl.dot(pb, c3)
        m_i = m_ij

    inv_l = 1.0 / tl.maximum(l_i[:, None], 1e-20)
    out_base = Out_ptr + bh_id.to(tl.int64) * S_q * 512
    tl.store(out_base + offs_m[:, None] * 512 + (0 + d128)[None, :], (acc0 * inv_l).to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(out_base + offs_m[:, None] * 512 + (128 + d128)[None, :], (acc1 * inv_l).to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(out_base + offs_m[:, None] * 512 + (256 + d128)[None, :], (acc2 * inv_l).to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(out_base + offs_m[:, None] * 512 + (384 + d128)[None, :], (acc3 * inv_l).to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(Lse2_ptr + bh_id.to(tl.int64) * S_q + offs_m, m_i + tl.math.log2(tl.maximum(l_i, 1e-20)), mask=mask_m)


@triton.jit
def _mla_absorbed_bwd_dckv_kernel(
    Qa_ptr,
    Qr_ptr,
    Ckv_ptr,
    Kr_ptr,
    dO_ptr,
    Lse2_ptr,
    Delta_ptr,
    dCkv_part_ptr,
    dKr_part_ptr,
    sm_scale,
    qk_scale,
    n_h: tl.int32,
    S: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Merges `P^T dO + sm_scale * dS^T Q_abs` into one `[BN, 512]` register
    accumulator (`72` regs/thread at `num_warps=8`) because `K_nope == V == C_kv`!"""
    start_n = tl.program_id(0) * BN
    bh_id = tl.program_id(1)
    b_id = bh_id // n_h

    offs_n = start_n + tl.arange(0, BN)
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    ckv_base = Ckv_ptr + b_id.to(tl.int64) * S * 512
    kr_base = Kr_ptr + b_id.to(tl.int64) * S * 64
    c0 = tl.load(ckv_base + offs_n[:, None] * 512 + (0 + d128)[None, :])
    c1 = tl.load(ckv_base + offs_n[:, None] * 512 + (128 + d128)[None, :])
    c2 = tl.load(ckv_base + offs_n[:, None] * 512 + (256 + d128)[None, :])
    c3 = tl.load(ckv_base + offs_n[:, None] * 512 + (384 + d128)[None, :])
    kr = tl.load(kr_base + offs_n[:, None] * 64 + d64[None, :])

    dc0 = tl.zeros([BN, 128], dtype=tl.float32)
    dc1 = tl.zeros([BN, 128], dtype=tl.float32)
    dc2 = tl.zeros([BN, 128], dtype=tl.float32)
    dc3 = tl.zeros([BN, 128], dtype=tl.float32)
    dkr = tl.zeros([BN, 64], dtype=tl.float32)

    qa_base = Qa_ptr + bh_id.to(tl.int64) * S * 512
    qr_base = Qr_ptr + bh_id.to(tl.int64) * S * 64
    do_base = dO_ptr + bh_id.to(tl.int64) * S * 512
    lse_base = Lse2_ptr + bh_id.to(tl.int64) * S
    del_base = Delta_ptr + bh_id.to(tl.int64) * S

    for start_m in range(start_n, S, BM):
        offs_m = start_m + tl.arange(0, BM)
        qa0 = tl.load(qa_base + offs_m[:, None] * 512 + (0 + d128)[None, :])
        qa1 = tl.load(qa_base + offs_m[:, None] * 512 + (128 + d128)[None, :])
        qa2 = tl.load(qa_base + offs_m[:, None] * 512 + (256 + d128)[None, :])
        qa3 = tl.load(qa_base + offs_m[:, None] * 512 + (384 + d128)[None, :])
        qr = tl.load(qr_base + offs_m[:, None] * 64 + d64[None, :])
        lse2 = tl.load(lse_base + offs_m)
        delta = tl.load(del_base + offs_m)

        do0 = tl.load(do_base + offs_m[:, None] * 512 + (0 + d128)[None, :])
        do1 = tl.load(do_base + offs_m[:, None] * 512 + (128 + d128)[None, :])
        do2 = tl.load(do_base + offs_m[:, None] * 512 + (256 + d128)[None, :])
        do3 = tl.load(do_base + offs_m[:, None] * 512 + (384 + d128)[None, :])

        qkT = (
            tl.dot(c0, tl.trans(qa0))
            + tl.dot(c1, tl.trans(qa1))
            + tl.dot(c2, tl.trans(qa2))
            + tl.dot(c3, tl.trans(qa3))
            + tl.dot(kr, tl.trans(qr))
        ) * qk_scale
        pT = tl.math.exp2(qkT - lse2[None, :])
        if start_m == start_n:
            pT = tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)

        dpT = (
            tl.dot(c0, tl.trans(do0))
            + tl.dot(c1, tl.trans(do1))
            + tl.dot(c2, tl.trans(do2))
            + tl.dot(c3, tl.trans(do3))
        )
        pTb = pT.to(tl.bfloat16)
        dsTb = (pT * (dpT - delta[None, :]) * sm_scale).to(tl.bfloat16)

        dc0 += tl.dot(pTb, do0) + tl.dot(dsTb, qa0)
        dc1 += tl.dot(pTb, do1) + tl.dot(dsTb, qa1)
        dc2 += tl.dot(pTb, do2) + tl.dot(dsTb, qa2)
        dc3 += tl.dot(pTb, do3) + tl.dot(dsTb, qa3)
        dkr += tl.dot(dsTb, qr)

    dcp_base = dCkv_part_ptr + bh_id.to(tl.int64) * S * 512
    tl.store(dcp_base + offs_n[:, None] * 512 + (0 + d128)[None, :], dc0)
    tl.store(dcp_base + offs_n[:, None] * 512 + (128 + d128)[None, :], dc1)
    tl.store(dcp_base + offs_n[:, None] * 512 + (256 + d128)[None, :], dc2)
    tl.store(dcp_base + offs_n[:, None] * 512 + (384 + d128)[None, :], dc3)
    tl.store(
        dKr_part_ptr + bh_id.to(tl.int64) * S * 64 + offs_n[:, None] * 64 + d64[None, :],
        dkr,
    )


@triton.jit
def _mla_absorbed_bwd_dq_kernel(
    Qa_ptr,
    Qr_ptr,
    Ckv_ptr,
    Kr_ptr,
    dO_ptr,
    Lse2_ptr,
    Delta_ptr,
    dQa_ptr,
    dQr_ptr,
    sm_scale,
    qk_scale,
    n_h: tl.int32,
    S: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    start_m = tl.program_id(0) * BM
    bh_id = tl.program_id(1)
    b_id = bh_id // n_h

    offs_m = start_m + tl.arange(0, BM)
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    qa_base = Qa_ptr + bh_id.to(tl.int64) * S * 512
    qr_base = Qr_ptr + bh_id.to(tl.int64) * S * 64
    do_base = dO_ptr + bh_id.to(tl.int64) * S * 512
    ckv_base = Ckv_ptr + b_id.to(tl.int64) * S * 512
    kr_base = Kr_ptr + b_id.to(tl.int64) * S * 64

    qa0 = tl.load(qa_base + offs_m[:, None] * 512 + (0 + d128)[None, :])
    qa1 = tl.load(qa_base + offs_m[:, None] * 512 + (128 + d128)[None, :])
    qa2 = tl.load(qa_base + offs_m[:, None] * 512 + (256 + d128)[None, :])
    qa3 = tl.load(qa_base + offs_m[:, None] * 512 + (384 + d128)[None, :])
    qr = tl.load(qr_base + offs_m[:, None] * 64 + d64[None, :])

    do0 = tl.load(do_base + offs_m[:, None] * 512 + (0 + d128)[None, :])
    do1 = tl.load(do_base + offs_m[:, None] * 512 + (128 + d128)[None, :])
    do2 = tl.load(do_base + offs_m[:, None] * 512 + (256 + d128)[None, :])
    do3 = tl.load(do_base + offs_m[:, None] * 512 + (384 + d128)[None, :])

    lse2 = tl.load(Lse2_ptr + bh_id.to(tl.int64) * S + offs_m)
    delta = tl.load(Delta_ptr + bh_id.to(tl.int64) * S + offs_m)

    dqa0 = tl.zeros([BM, 128], dtype=tl.float32)
    dqa1 = tl.zeros([BM, 128], dtype=tl.float32)
    dqa2 = tl.zeros([BM, 128], dtype=tl.float32)
    dqa3 = tl.zeros([BM, 128], dtype=tl.float32)
    dqr = tl.zeros([BM, 64], dtype=tl.float32)

    for start_n in range(0, start_m + BM, BN):
        offs_n = start_n + tl.arange(0, BN)
        c0 = tl.load(ckv_base + offs_n[:, None] * 512 + (0 + d128)[None, :])
        c1 = tl.load(ckv_base + offs_n[:, None] * 512 + (128 + d128)[None, :])
        c2 = tl.load(ckv_base + offs_n[:, None] * 512 + (256 + d128)[None, :])
        c3 = tl.load(ckv_base + offs_n[:, None] * 512 + (384 + d128)[None, :])
        kr = tl.load(kr_base + offs_n[:, None] * 64 + d64[None, :])

        qk = (
            tl.dot(qa0, tl.trans(c0))
            + tl.dot(qa1, tl.trans(c1))
            + tl.dot(qa2, tl.trans(c2))
            + tl.dot(qa3, tl.trans(c3))
            + tl.dot(qr, tl.trans(kr))
        ) * qk_scale
        p = tl.math.exp2(qk - lse2[:, None])
        if start_n >= start_m:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        dp = (
            tl.dot(do0, tl.trans(c0))
            + tl.dot(do1, tl.trans(c1))
            + tl.dot(do2, tl.trans(c2))
            + tl.dot(do3, tl.trans(c3))
        )
        ds = (p * (dp - delta[:, None]) * sm_scale).to(tl.bfloat16)
        dqa0 += tl.dot(ds, c0)
        dqa1 += tl.dot(ds, c1)
        dqa2 += tl.dot(ds, c2)
        dqa3 += tl.dot(ds, c3)
        dqr += tl.dot(ds, kr)

    dqa_base = dQa_ptr + bh_id.to(tl.int64) * S * 512
    tl.store(dqa_base + offs_m[:, None] * 512 + (0 + d128)[None, :], dqa0.to(tl.bfloat16))
    tl.store(dqa_base + offs_m[:, None] * 512 + (128 + d128)[None, :], dqa1.to(tl.bfloat16))
    tl.store(dqa_base + offs_m[:, None] * 512 + (256 + d128)[None, :], dqa2.to(tl.bfloat16))
    tl.store(dqa_base + offs_m[:, None] * 512 + (384 + d128)[None, :], dqa3.to(tl.bfloat16))
    tl.store(
        dQr_ptr + bh_id.to(tl.int64) * S * 64 + offs_m[:, None] * 64 + d64[None, :],
        dqr.to(tl.bfloat16),
    )


class _TritonAbsorbedMLAAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q_abs: torch.Tensor,
        q_rope: torch.Tensor,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        B, H, S, _ = q_abs.shape
        qa = q_abs.contiguous()
        qr = q_rope.contiguous()
        ckv = c_kv.contiguous()
        kr = k_rope.contiguous()

        out = torch.empty_like(qa)
        lse2 = torch.empty((B, H, S), device=qa.device, dtype=torch.float32)
        _mla_absorbed_fwd_kernel[(S // 32, B * H)](
            qa,
            qr,
            ckv,
            kr,
            out,
            lse2,
            sm_scale * RCP_LN2,
            H,
            S,
            S,
            0,
            BM=32,
            BN=32,
            CAUSAL=True,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(qa, qr, ckv, kr, out, lse2)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        qa, qr, ckv, kr, out, lse2 = ctx.saved_tensors
        B, H, S, _ = qa.shape
        doc = do.contiguous()
        delta = torch.empty((B, H, S), device=qa.device, dtype=torch.float32)
        _mla_bwd_prep_kernel[(triton.cdiv(B * H * S, 64),)](
            out, doc, delta, B * H * S, D_V=512, num_warps=4
        )
        dckv_part = torch.empty((B, H, S, 512), device=qa.device, dtype=torch.float32)
        dkr_part = torch.empty((B, H, S, 64), device=qa.device, dtype=torch.float32)
        dqa = torch.empty_like(qa)
        dqr = torch.empty_like(qr)
        qk_scale = ctx.sm_scale * RCP_LN2

        _mla_absorbed_bwd_dckv_kernel[(S // 32, B * H)](
            qa,
            qr,
            ckv,
            kr,
            doc,
            lse2,
            delta,
            dckv_part,
            dkr_part,
            ctx.sm_scale,
            qk_scale,
            H,
            S,
            BM=32,
            BN=32,
            num_warps=8,
            num_stages=2,
        )
        _mla_absorbed_bwd_dq_kernel[(S // 32, B * H)](
            qa,
            qr,
            ckv,
            kr,
            doc,
            lse2,
            delta,
            dqa,
            dqr,
            ctx.sm_scale,
            qk_scale,
            H,
            S,
            BM=32,
            BN=32,
            num_warps=8,
            num_stages=2,
        )
        return (
            dqa,
            dqr,
            dckv_part.sum(dim=1).to(ckv.dtype),
            dkr_part.sum(dim=1).to(kr.dtype),
            None,
        )


def triton_absorbed_mla(
    q_abs: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    return _TritonAbsorbedMLAAutograd.apply(q_abs, q_rope, c_kv, k_rope, sm_scale)


# Query-Head-Grouped Split-KV Decode for `S_q = 1` (`[B, NH, 512]` + `[B, NH, 64]`)
@triton.jit
def _mla_absorbed_splitkv_decode_stage1(
    Qa_ptr,
    Qr_ptr,
    Ckv_ptr,
    Kr_ptr,
    PartO_ptr,
    PartLse2_ptr,
    qk_scale,
    L_kv: tl.int32,
    kv_per_split: tl.int32,
    NH: tl.constexpr,
    BN: tl.constexpr,
):
    split_id = tl.program_id(0)
    b_id = tl.program_id(1)
    kv_start = split_id * kv_per_split
    kv_end = tl.minimum(kv_start + kv_per_split, L_kv)

    h_offs = tl.arange(0, NH)
    d128 = tl.arange(0, 128)
    d64 = tl.arange(0, 64)

    qa_base = Qa_ptr + b_id.to(tl.int64) * NH * 512
    qr_base = Qr_ptr + b_id.to(tl.int64) * NH * 64
    ckv_base = Ckv_ptr + b_id.to(tl.int64) * L_kv * 512
    kr_base = Kr_ptr + b_id.to(tl.int64) * L_kv * 64

    qa0 = tl.load(qa_base + h_offs[:, None] * 512 + (0 + d128)[None, :])
    qa1 = tl.load(qa_base + h_offs[:, None] * 512 + (128 + d128)[None, :])
    qa2 = tl.load(qa_base + h_offs[:, None] * 512 + (256 + d128)[None, :])
    qa3 = tl.load(qa_base + h_offs[:, None] * 512 + (384 + d128)[None, :])
    qr = tl.load(qr_base + h_offs[:, None] * 64 + d64[None, :])

    m_i = tl.zeros([NH], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([NH], dtype=tl.float32)
    acc0 = tl.zeros([NH, 128], dtype=tl.float32)
    acc1 = tl.zeros([NH, 128], dtype=tl.float32)
    acc2 = tl.zeros([NH, 128], dtype=tl.float32)
    acc3 = tl.zeros([NH, 128], dtype=tl.float32)

    for start_n in range(kv_start, kv_end, BN):
        offs_n = start_n + tl.arange(0, BN)
        mask_n = offs_n < kv_end
        c0 = tl.load(ckv_base + offs_n[:, None] * 512 + (0 + d128)[None, :], mask=mask_n[:, None], other=0.0)
        c1 = tl.load(ckv_base + offs_n[:, None] * 512 + (128 + d128)[None, :], mask=mask_n[:, None], other=0.0)
        c2 = tl.load(ckv_base + offs_n[:, None] * 512 + (256 + d128)[None, :], mask=mask_n[:, None], other=0.0)
        c3 = tl.load(ckv_base + offs_n[:, None] * 512 + (384 + d128)[None, :], mask=mask_n[:, None], other=0.0)
        kr = tl.load(kr_base + offs_n[:, None] * 64 + d64[None, :], mask=mask_n[:, None], other=0.0)

        qk = (
            tl.dot(qa0, tl.trans(c0))
            + tl.dot(qa1, tl.trans(c1))
            + tl.dot(qa2, tl.trans(c2))
            + tl.dot(qa3, tl.trans(c3))
            + tl.dot(qr, tl.trans(kr))
        ) * qk_scale
        qk = tl.where(mask_n[None, :], qk, -1.0e20)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        pb = p.to(tl.bfloat16)
        acc0 = acc0 * alpha[:, None] + tl.dot(pb, c0)
        acc1 = acc1 * alpha[:, None] + tl.dot(pb, c1)
        acc2 = acc2 * alpha[:, None] + tl.dot(pb, c2)
        acc3 = acc3 * alpha[:, None] + tl.dot(pb, c3)
        m_i = m_ij

    inv_l = 1.0 / tl.maximum(l_i[:, None], 1e-20)
    num_splits = tl.num_programs(0)
    po_base = PartO_ptr + (b_id.to(tl.int64) * num_splits + split_id) * NH * 512
    tl.store(po_base + h_offs[:, None] * 512 + (0 + d128)[None, :], acc0 * inv_l)
    tl.store(po_base + h_offs[:, None] * 512 + (128 + d128)[None, :], acc1 * inv_l)
    tl.store(po_base + h_offs[:, None] * 512 + (256 + d128)[None, :], acc2 * inv_l)
    tl.store(po_base + h_offs[:, None] * 512 + (384 + d128)[None, :], acc3 * inv_l)
    tl.store(
        PartLse2_ptr + (b_id.to(tl.int64) * num_splits + split_id) * NH + h_offs,
        m_i + tl.math.log2(tl.maximum(l_i, 1e-20)),
    )


@triton.jit
def _mla_absorbed_splitkv_decode_stage2(
    PartO_ptr,
    PartLse2_ptr,
    Out_ptr,
    NH: tl.int32,
    NUM_SPLITS: tl.constexpr,
):
    bh_id = tl.program_id(0)
    b_id = bh_id // NH
    h_id = bh_id % NH
    s_offs = tl.arange(0, NUM_SPLITS)
    d128 = tl.arange(0, 128)

    lse_vec = tl.load(PartLse2_ptr + (b_id.to(tl.int64) * NUM_SPLITS + s_offs) * NH + h_id)
    m_max = tl.max(lse_vec, axis=0)
    w_vec = tl.math.exp2(lse_vec - m_max)
    w_norm = w_vec / tl.sum(w_vec, axis=0)

    out_base = Out_ptr + bh_id.to(tl.int64) * 512
    for sub in tl.static_range(0, 512, 128):
        cols = sub + d128
        po = tl.load(
            PartO_ptr
            + ((b_id.to(tl.int64) * NUM_SPLITS + s_offs[:, None]) * NH + h_id) * 512
            + cols[None, :]
        )
        merged = tl.sum(po * w_norm[:, None], axis=0)
        tl.store(out_base + cols, merged.to(tl.bfloat16))


# ============================================================================
# 4. Fused Online-Softmax Base-2 LSE Ring Merger for Context Parallelism
# ============================================================================


@triton.jit
def _ring_lse2_merge_kernel(
    O_acc_ptr,
    Lse_acc_ptr,
    O_step_ptr,
    Lse_step_ptr,
    N_rows: tl.int32,
    D: tl.constexpr,
    INIT_STEP: tl.constexpr,
):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)
    mask = row < N_rows
    d_offs = tl.arange(0, D)

    os_val = tl.load(
        O_step_ptr + row[:, None].to(tl.int64) * D + d_offs[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    ls_val = tl.load(Lse_step_ptr + row, mask=mask, other=-1.0e20)

    if INIT_STEP:
        tl.store(
            O_acc_ptr + row[:, None].to(tl.int64) * D + d_offs[None, :],
            os_val,
            mask=mask[:, None],
        )
        tl.store(Lse_acc_ptr + row, ls_val, mask=mask)
    else:
        oa_val = tl.load(
            O_acc_ptr + row[:, None].to(tl.int64) * D + d_offs[None, :],
            mask=mask[:, None],
            other=0.0,
        )
        la_val = tl.load(Lse_acc_ptr + row, mask=mask, other=-1.0e20)
        m_new = tl.maximum(la_val, ls_val)
        w_a = tl.math.exp2(la_val - m_new)
        w_s = tl.math.exp2(ls_val - m_new)
        denom = w_a + w_s
        o_new = (oa_val * w_a[:, None] + os_val * w_s[:, None]) / denom[:, None]
        l_new = m_new + tl.math.log2(denom)
        tl.store(
            O_acc_ptr + row[:, None].to(tl.int64) * D + d_offs[None, :],
            o_new,
            mask=mask[:, None],
        )
        tl.store(Lse_acc_ptr + row, l_new, mask=mask)


# ============================================================================
# 5. DeepSeek-V3-Architecture MLA Module & Telemetry Helpers
# ============================================================================


class DeepSeekMLALayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        d_c: int = 512,
        d_q_c: int = 1536,
        d_nope: int = 128,
        d_rope: int = 64,
        d_v: int = 128,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.d_c = d_c
        self.d_q_c = d_q_c
        self.d_nope = d_nope
        self.d_rope = d_rope
        self.d_v = d_v
        self.sm_scale = 1.0 / math.sqrt(float(d_nope + d_rope))

        self.in_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.w_dq = nn.Linear(hidden, d_q_c, bias=False)
        self.q_norm = nn.RMSNorm(d_q_c, eps=1e-5)
        self.w_uq_qr = nn.Linear(d_q_c, n_heads * (d_nope + d_rope), bias=False)

        self.w_dkv_kr = nn.Linear(hidden, d_c + d_rope, bias=False)
        self.kv_norm = nn.RMSNorm(d_c, eps=1e-5)
        self.w_uk_uv = nn.Linear(d_c, n_heads * (d_nope + d_v), bias=False)

        self.w_o = nn.Linear(n_heads * d_v, hidden, bias=False)

    def forward_eager_projections(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        H = self.n_heads
        h = self.in_norm(x)
        cq = self.q_norm(self.w_dq(h))
        q_all = self.w_uq_qr(cq).view(B, S, H, self.d_nope + self.d_rope).transpose(1, 2)
        q_nope = q_all[..., : self.d_nope].contiguous()
        q_rope = apply_rope_bhsd(q_all[..., self.d_nope :], cos, sin).contiguous()

        raw_kv = self.w_dkv_kr(h)
        c_kv = self.kv_norm(raw_kv[..., : self.d_c])
        k_rope = apply_rope_bhsd(
            raw_kv[..., self.d_c :].unsqueeze(1), cos, sin
        ).contiguous()
        kv_up = self.w_uk_uv(c_kv).view(B, S, H, self.d_nope + self.d_v).transpose(1, 2)
        k_nope = kv_up[..., : self.d_nope].contiguous()
        v = kv_up[..., self.d_nope :].contiguous()
        return q_nope, q_rope, k_nope, k_rope, v

    def forward_mode(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mode: str,
        compiled_flex: Any = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H = self.n_heads
        if mode == "mla_triton_split_fused":
            h = self.in_norm(x)
            cq = self.q_norm(self.w_dq(h))
            q_all = self.w_uq_qr(cq).view(B, S, H, 192).transpose(1, 2)
            q_nope = q_all[..., :128].contiguous()
            q_rope = apply_rope_bhsd(q_all[..., 128:], cos, sin).contiguous()

            raw_kv = self.w_dkv_kr(h)
            c_kv, k_rope_bs = fused_latent_rmsnorm_rope(
                raw_kv, self.kv_norm.weight, cos, sin, 1e-5
            )
            k_rope = k_rope_bs.unsqueeze(1)
            kv_up = self.w_uk_uv(c_kv).view(B, S, H, 256).transpose(1, 2)
            k_nope = kv_up[..., :128].contiguous()
            v = kv_up[..., 128:].contiguous()
            o = triton_split_mla(q_nope, q_rope, k_nope, k_rope, v, self.sm_scale)
            return x + self.w_o(o.transpose(1, 2).contiguous().view(B, S, H * 128))

        if mode == "mla_triton_weight_absorbed":
            h = self.in_norm(x)
            cq = self.q_norm(self.w_dq(h))
            q_all = self.w_uq_qr(cq).view(B, S, H, 192).transpose(1, 2)
            q_nope = q_all[..., :128]
            q_rope = apply_rope_bhsd(q_all[..., 128:], cos, sin).contiguous()

            raw_kv = self.w_dkv_kr(h)
            c_kv, k_rope = fused_latent_rmsnorm_rope(
                raw_kv, self.kv_norm.weight, cos, sin, 1e-5
            )
            # Weight absorption: W_uk_uv.weight is [H * 256, 512] -> split into W_uk [H, 128, 512] and W_uv [H, 128, 512]
            w_ukv = self.w_uk_uv.weight.view(H, 256, 512)
            w_uk = w_ukv[:, :128, :]  # [H, 128, 512]
            w_uv = w_ukv[:, 128:, :]  # [H, 128, 512]
            q_abs = torch.einsum("bhsd,hdc->bhsc", q_nope, w_uk).contiguous()
            o_lat = triton_absorbed_mla(q_abs, q_rope, c_kv, k_rope, self.sm_scale)
            o_v = torch.einsum("bhsc,hvc->bhsv", o_lat, w_uv)
            return x + self.w_o(o_v.transpose(1, 2).contiguous().view(B, S, H * 128))

        q_nope, q_rope, k_nope, k_rope, v = self.forward_eager_projections(x, cos, sin)
        if mode == "mla_fa2_vpadded_eager":
            qc = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2).contiguous()
            kc = (
                torch.cat([k_nope, k_rope.expand(-1, H, -1, -1)], dim=-1)
                .transpose(1, 2)
                .contiguous()
            )
            vp = F.pad(v.transpose(1, 2).contiguous(), (0, 64))
            op = flash_attn_func(qc, kc, vp, causal=True, softmax_scale=self.sm_scale)
            o = op[..., :128].contiguous().view(B, S, H * 128)
            return x + self.w_o(o)

        qc_bh = torch.cat([q_nope, q_rope], dim=-1)
        kc_bh = torch.cat([k_nope, k_rope.expand(-1, H, -1, -1)], dim=-1)
        if mode == "mla_sdpa_math_eager":
            with sdpa_kernel(SDPBackend.MATH):
                o_bh = F.scaled_dot_product_attention(
                    qc_bh, kc_bh, v, is_causal=True, scale=self.sm_scale
                )
        elif mode == "mla_sdpa_cutlass_efficient":
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                o_bh = F.scaled_dot_product_attention(
                    qc_bh, kc_bh, v, is_causal=True, scale=self.sm_scale
                )
        elif mode == "mla_flex_attention_compiled":
            assert compiled_flex is not None
            o_bh = compiled_flex(qc_bh, kc_bh, v, scale=self.sm_scale)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return x + self.w_o(o_bh.transpose(1, 2).contiguous().view(B, S, H * 128))


def sample_gpu_clocks(device_idx: int) -> dict[str, float]:
    if not HAS_PYNVML:
        return {"sm_clock_mhz": 0.0, "power_w": 0.0}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        return {
            "sm_clock_mhz": float(
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            ),
            "power_w": float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0,
        }
    except Exception:
        return {"sm_clock_mhz": 0.0, "power_w": 0.0}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-8)).item()
    )


def time_fn_with_peak_mem(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    base_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    inc_mib = max(
        0.0, (torch.cuda.max_memory_allocated(device) - base_mem) / (1024.0**2)
    )

    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        s_ev.record()
        fn()
        e_ev.record()
        torch.cuda.synchronize(device)
        samples.append(float(s_ev.elapsed_time(e_ev)))
    nv = sample_gpu_clocks(device.index)
    ss = sorted(samples)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    return {
        "median_ms": round(statistics.median(ss), 4),
        "iqr_ms": round(q75 - q25, 4),
        "peak_inc_mib": round(inc_mib, 2),
        "sm_clock_after_mhz": nv["sm_clock_mhz"],
        "power_after_w": round(nv["power_w"], 2),
    }


# ============================================================================
# 6. Suite 0: Ecosystem Audit, FP64 Richardson Chain-Rule, & SM80 Bwd Ablation
# ============================================================================


def run_suite_0_audits(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    # 1. Ecosystem Compatibility Audit on Decompressed (192->128) & Weight-Absorbed (576->512 MQA)
    qd = torch.randn((1, 128, 8, 192), device=device, dtype=torch.bfloat16, requires_grad=True)
    kd = torch.randn((1, 128, 8, 192), device=device, dtype=torch.bfloat16, requires_grad=True)
    vd = torch.randn((1, 128, 8, 128), device=device, dtype=torch.bfloat16, requires_grad=True)

    def _probe(fn_call: Callable[[], None]) -> dict[str, Any]:
        try:
            fn_call()
            torch.cuda.synchronize(device)
            return {"supported": True, "error_type": None, "error_excerpt": None}
        except Exception as exc:
            return {
                "supported": False,
                "error_type": type(exc).__name__,
                "error_excerpt": str(exc).splitlines()[0][:180],
            }

    cflex_probe = torch.compile(flex_attention)
    qdt, kdt, vdt = (
        qd.detach().transpose(1, 2).requires_grad_(True),
        kd.detach().transpose(1, 2).requires_grad_(True),
        vd.detach().transpose(1, 2).requires_grad_(True),
    )

    decomp_audit = {
        "flash_attn_func_direct_192_128": _probe(
            lambda: flash_attn_func(qd, kd, vd, causal=True).sum().backward()
        ),
        "flash_attn_func_vpadded_192_192": _probe(
            lambda: flash_attn_func(
                qd.detach().requires_grad_(True),
                kd.detach().requires_grad_(True),
                F.pad(vd.detach(), (0, 64)).requires_grad_(True),
                causal=True,
            )[..., :128]
            .sum()
            .backward()
        ),
        "sdpa_flash_192_128": _probe(
            lambda: (
                sdpa_kernel(SDPBackend.FLASH_ATTENTION).__enter__(),
            )
        ),
    }
    # Clean context-managed probes for SDPA
    for b_key, b_enum in (
        ("sdpa_flash_192_128", SDPBackend.FLASH_ATTENTION),
        ("sdpa_efficient_192_128", SDPBackend.EFFICIENT_ATTENTION),
        ("sdpa_cudnn_192_128", SDPBackend.CUDNN_ATTENTION),
        ("sdpa_math_192_128", SDPBackend.MATH),
    ):
        def _run_sdpa_d(be: Any = b_enum) -> None:
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(
                    qdt.detach().requires_grad_(True),
                    kdt.detach().requires_grad_(True),
                    vdt.detach().requires_grad_(True),
                    is_causal=True,
                )
                o.sum().backward()

        decomp_audit[b_key] = _probe(_run_sdpa_d)

    decomp_audit["flex_attention_192_128"] = _probe(
        lambda: cflex_probe(
            qdt.detach().requires_grad_(True),
            kdt.detach().requires_grad_(True),
            vdt.detach().requires_grad_(True),
        )
        .sum()
        .backward()
    )

    # Weight-absorbed (576 -> 512, 8Q / 1KV MQA)
    qa_p = torch.randn((1, 8, 128, 576), device=device, dtype=torch.bfloat16)
    ka_p = torch.randn((1, 1, 128, 576), device=device, dtype=torch.bfloat16)
    va_p = torch.randn((1, 1, 128, 512), device=device, dtype=torch.bfloat16)

    absorbed_audit: dict[str, Any] = {}
    absorbed_audit["flash_attn_func_576_576"] = _probe(
        lambda: flash_attn_func(
            qa_p.transpose(1, 2).contiguous().requires_grad_(True),
            ka_p.transpose(1, 2).contiguous().requires_grad_(True),
            F.pad(va_p.transpose(1, 2).contiguous(), (0, 64)).requires_grad_(True),
            causal=True,
        )
        .sum()
        .backward()
    )
    for b_key, b_enum in (
        ("sdpa_flash_576_512", SDPBackend.FLASH_ATTENTION),
        ("sdpa_efficient_576_512", SDPBackend.EFFICIENT_ATTENTION),
        ("sdpa_cudnn_576_512", SDPBackend.CUDNN_ATTENTION),
        ("sdpa_math_576_512", SDPBackend.MATH),
    ):
        def _run_sdpa_a(be: Any = b_enum) -> None:
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(
                    qa_p.clone().requires_grad_(True),
                    ka_p.clone().requires_grad_(True),
                    va_p.clone().requires_grad_(True),
                    is_causal=True,
                    enable_gqa=True,
                )
                o.sum().backward()

        absorbed_audit[b_key] = _probe(_run_sdpa_a)

    absorbed_audit["flex_attention_576_512_sm80"] = _probe(
        lambda: cflex_probe(
            qa_p.clone().requires_grad_(True),
            ka_p.clone().requires_grad_(True),
            va_p.clone().requires_grad_(True),
            enable_gqa=True,
        )
        .sum()
        .backward()
    )

    ws_fi = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)

    def _run_fi_fa2() -> None:
        wm = flashinfer.BatchMLAPagedAttentionWrapper(ws_fi, backend="fa2")
        qo_ind = torch.arange(0, 3, dtype=torch.int32, device=device)
        kv_ind = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
        kv_idx = torch.arange(0, 16, dtype=torch.int32, device=device)
        kv_l = torch.full((2,), 128, dtype=torch.int32, device=device)
        wm.plan(
            qo_ind,
            kv_ind,
            kv_idx,
            kv_l,
            8,
            512,
            64,
            16,
            False,
            1.0 / math.sqrt(192.0),
            torch.bfloat16,
            torch.bfloat16,
        )
        _ = wm.run(
            torch.randn((2, 8, 512), device=device, dtype=torch.bfloat16),
            torch.randn((2, 8, 64), device=device, dtype=torch.bfloat16),
            torch.randn((16, 16, 512), device=device, dtype=torch.bfloat16),
            torch.randn((16, 16, 64), device=device, dtype=torch.bfloat16),
        )

    absorbed_audit["flashinfer_mla_paged_fa2"] = _probe(_run_fi_fa2)

    # Probe `backend="fa3"` in an isolated child process so `cudaErrorNoKernelImageForDevice`
    # inside FlashInfer's Hopper WGMMA kernel launch never taints the parent NCCL/CUDA context!
    import subprocess
    import sys

    fa3_code = (
        "import math, torch, flashinfer\n"
        "ws = torch.empty(16*1024*1024, dtype=torch.uint8, device='cuda')\n"
        "wm = flashinfer.BatchMLAPagedAttentionWrapper(ws, backend='fa3')\n"
        "wm.plan(torch.arange(0,3,dtype=torch.int32,device='cuda'),"
        "torch.tensor([0,8,16],dtype=torch.int32,device='cuda'),"
        "torch.arange(0,16,dtype=torch.int32,device='cuda'),"
        "torch.full((2,),128,dtype=torch.int32,device='cuda'),"
        "8,512,64,16,False,1.0/math.sqrt(192.0),torch.bfloat16,torch.bfloat16)\n"
        "wm.run(torch.randn((2,8,512),device='cuda',dtype=torch.bfloat16),"
        "torch.randn((2,8,64),device='cuda',dtype=torch.bfloat16),"
        "torch.randn((16,16,512),device='cuda',dtype=torch.bfloat16),"
        "torch.randn((16,16,64),device='cuda',dtype=torch.bfloat16))\n"
        "torch.cuda.synchronize()\n"
    )
    proc_fa3 = subprocess.run(
        [sys.executable, "-c", fa3_code], capture_output=True, text=True, timeout=60
    )
    if proc_fa3.returncode == 0:
        absorbed_audit["flashinfer_mla_paged_fa3"] = {
            "supported": True,
            "error_type": None,
            "error_excerpt": None,
        }
    else:
        err_tail = " ".join((proc_fa3.stderr or proc_fa3.stdout).strip().splitlines()[-3:])
        absorbed_audit["flashinfer_mla_paged_fa3"] = {
            "supported": False,
            "error_type": "SubprocessCUDAError",
            "error_excerpt": err_tail[:220],
        }

    # 2. FP64 Analytical Chain-Rule & Richardson Central-Difference (h vs h/2 -> ~4.00x)
    gen64 = torch.Generator(device=device)
    gen64.manual_seed(12005)
    Bf, Hf, Sf = 1, 4, 64
    sm_f = 1.0 / math.sqrt(192.0)

    qn64 = torch.randn((Bf, Hf, Sf, 128), generator=gen64, device=device, dtype=torch.float64)
    qr64 = torch.randn((Bf, Hf, Sf, 64), generator=gen64, device=device, dtype=torch.float64)
    kn64 = torch.randn((Bf, Hf, Sf, 128), generator=gen64, device=device, dtype=torch.float64)
    kr64 = torch.randn((Bf, 1, Sf, 64), generator=gen64, device=device, dtype=torch.float64)
    v64 = torch.randn((Bf, Hf, Sf, 128), generator=gen64, device=device, dtype=torch.float64)
    do64 = torch.randn((Bf, Hf, Sf, 128), generator=gen64, device=device, dtype=torch.float64)

    def _loss_split64(
        q_n: torch.Tensor,
        q_r: torch.Tensor,
        k_n: torch.Tensor,
        k_r: torch.Tensor,
        vv: torch.Tensor,
    ) -> torch.Tensor:
        scores = (
            torch.matmul(q_n, k_n.transpose(-1, -2))
            + torch.matmul(q_r, k_r.expand(-1, Hf, -1, -1).transpose(-1, -2))
        ) * sm_f
        ca = torch.tril(torch.ones((Sf, Sf), device=device, dtype=torch.bool))
        scores = torch.where(ca, scores, -1e30)
        p = torch.softmax(scores, dim=-1)
        out = torch.matmul(p, vv)
        return (torch.sin(out) * do64).sum()

    # Exact analytical backward matching our Split-Dim kernel equations in FP64
    scores_a = (
        torch.matmul(qn64, kn64.transpose(-1, -2))
        + torch.matmul(qr64, kr64.expand(-1, Hf, -1, -1).transpose(-1, -2))
    ) * sm_f
    ca_a = torch.tril(torch.ones((Sf, Sf), device=device, dtype=torch.bool))
    p_a = torch.softmax(torch.where(ca_a, scores_a, -1e30), dim=-1)
    o_a = torch.matmul(p_a, v64)
    g_out = torch.cos(o_a) * do64
    delta_a = (o_a * g_out).sum(dim=-1, keepdim=True)
    dv_a = torch.matmul(p_a.transpose(-1, -2), g_out)
    ds_a = p_a * (torch.matmul(g_out, v64.transpose(-1, -2)) - delta_a) * sm_f
    dqn_a = torch.matmul(ds_a, kn64)
    dqr_a = torch.matmul(ds_a, kr64.expand(-1, Hf, -1, -1))
    dkn_a = torch.matmul(ds_a.transpose(-1, -2), qn64)
    dkr_a = torch.matmul(ds_a.transpose(-1, -2), qr64).sum(dim=1, keepdim=True)

    split_fd_ratios: dict[str, float] = {}
    split_base = [qn64, qr64, kn64, kr64, v64]
    split_grads = [dqn_a, dqr_a, dkn_a, dkr_a, dv_a]
    for idx_i, (nm, ga) in enumerate(
        zip(["q_nope", "q_rope", "k_nope", "k_rope_shared", "v"], split_grads)
    ):
        d_vec = torch.randn_like(split_base[idx_i], generator=gen64)
        d_vec = d_vec / d_vec.pow(2).mean().sqrt().clamp(min=1e-18)
        ana_dir = float((ga * d_vec).sum().item())
        errs = []
        for h_st in (5e-3, 2.5e-3):
            ip = [x.clone() for x in split_base]
            im = [x.clone() for x in split_base]
            ip[idx_i] = ip[idx_i] + h_st * d_vec
            im[idx_i] = im[idx_i] - h_st * d_vec
            fd_v = float((_loss_split64(*ip) - _loss_split64(*im)).item()) / (2.0 * h_st)
            errs.append(abs(fd_v - ana_dir))
        split_fd_ratios[nm] = round(errs[0] / max(errs[1], 1e-18), 4)

    # Also verify Weight-Absorbed merged `dC_kv = sum_h (P^T dO + sm_scale * dS^T Q_abs)` in FP64!
    qa64 = torch.randn((Bf, Hf, Sf, 512), generator=gen64, device=device, dtype=torch.float64)
    ckv64 = torch.randn((Bf, Sf, 512), generator=gen64, device=device, dtype=torch.float64)
    kr_s64 = kr64.squeeze(1)
    doa64 = torch.randn((Bf, Hf, Sf, 512), generator=gen64, device=device, dtype=torch.float64)

    def _loss_abs64(
        qa_in: torch.Tensor,
        qr_in: torch.Tensor,
        ckv_in: torch.Tensor,
        kr_in: torch.Tensor,
    ) -> torch.Tensor:
        sc = (
            torch.matmul(qa_in, ckv_in.unsqueeze(1).transpose(-1, -2))
            + torch.matmul(qr_in, kr_in.unsqueeze(1).transpose(-1, -2))
        ) * sm_f
        sc = torch.where(ca_a, sc, -1e30)
        pa = torch.softmax(sc, dim=-1)
        oa = torch.matmul(pa, ckv_in.unsqueeze(1))
        return (torch.sin(oa) * doa64).sum()

    sc_b = (
        torch.matmul(qa64, ckv64.unsqueeze(1).transpose(-1, -2))
        + torch.matmul(qr64, kr_s64.unsqueeze(1).transpose(-1, -2))
    ) * sm_f
    pa_b = torch.softmax(torch.where(ca_a, sc_b, -1e30), dim=-1)
    oa_b = torch.matmul(pa_b, ckv64.unsqueeze(1))
    go_b = torch.cos(oa_b) * doa64
    del_b = (oa_b * go_b).sum(dim=-1, keepdim=True)
    ds_b = pa_b * (torch.matmul(go_b, ckv64.unsqueeze(1).transpose(-1, -2)) - del_b) * sm_f
    dqa_b = torch.matmul(ds_b, ckv64.unsqueeze(1))
    dqr_b = torch.matmul(ds_b, kr_s64.unsqueeze(1))
    dckv_b = (
        torch.matmul(pa_b.transpose(-1, -2), go_b)
        + torch.matmul(ds_b.transpose(-1, -2), qa64)
    ).sum(dim=1)
    dkr_b = torch.matmul(ds_b.transpose(-1, -2), qr64).sum(dim=1)

    abs_fd_ratios: dict[str, float] = {}
    abs_base = [qa64, qr64, ckv64, kr_s64]
    abs_grads = [dqa_b, dqr_b, dckv_b, dkr_b]
    for idx_i, (nm, ga) in enumerate(
        zip(["q_abs", "q_rope", "c_kv_merged_kv", "k_rope_shared"], abs_grads)
    ):
        d_vec = torch.randn_like(abs_base[idx_i], generator=gen64)
        d_vec = d_vec / d_vec.pow(2).mean().sqrt().clamp(min=1e-18)
        ana_dir = float((ga * d_vec).sum().item())
        errs = []
        for h_st in (5e-3, 2.5e-3):
            ip = [x.clone() for x in abs_base]
            im = [x.clone() for x in abs_base]
            ip[idx_i] = ip[idx_i] + h_st * d_vec
            im[idx_i] = im[idx_i] - h_st * d_vec
            fd_v = float((_loss_abs64(*ip) - _loss_abs64(*im)).item()) / (2.0 * h_st)
            errs.append(abs(fd_v - ana_dir))
        abs_fd_ratios[nm] = round(errs[0] / max(errs[1], 1e-18), 4)

    # 3. BF16 Triton Parity vs Exact FP32 Reference (B=2, H=8, S=256)
    gen_p = torch.Generator(device=device)
    gen_p.manual_seed(12010)
    Bp, Hp, Sp = 2, 8, 256
    qnp = torch.randn((Bp, Hp, Sp, 128), generator=gen_p, device=device, dtype=torch.bfloat16)
    qrp = torch.randn((Bp, Hp, Sp, 64), generator=gen_p, device=device, dtype=torch.bfloat16)
    knp = torch.randn((Bp, Hp, Sp, 128), generator=gen_p, device=device, dtype=torch.bfloat16)
    krp = torch.randn((Bp, 1, Sp, 64), generator=gen_p, device=device, dtype=torch.bfloat16)
    vp = torch.randn((Bp, Hp, Sp, 128), generator=gen_p, device=device, dtype=torch.bfloat16)
    dop = torch.randn_like(vp)

    qnr, qrr, knr, krr, vr = [
        x.float().requires_grad_(True) for x in (qnp, qrp, knp, krp, vp)
    ]
    with sdpa_kernel(SDPBackend.MATH):
        or_s = F.scaled_dot_product_attention(
            torch.cat([qnr, qrr], dim=-1),
            torch.cat([knr, krr.expand(-1, Hp, -1, -1)], dim=-1),
            vr,
            is_causal=True,
            scale=sm_f,
        )
    (or_s * dop.float()).sum().backward()

    qnt, qrt, knt, krt, vt = [
        x.clone().requires_grad_(True) for x in (qnp, qrp, knp, krp, vp)
    ]
    ot_s = triton_split_mla(qnt, qrt, knt, krt, vt, sm_f)
    torch.autograd.backward(ot_s, dop)

    split_parity = {
        "out_rel_l2": round(rel_l2(ot_s, or_s), 6),
        "dq_nope_rel_l2": round(rel_l2(qnt.grad, qnr.grad), 6),
        "dq_rope_rel_l2": round(rel_l2(qrt.grad, qrr.grad), 6),
        "dk_nope_rel_l2": round(rel_l2(knt.grad, knr.grad), 6),
        "dk_rope_rel_l2": round(rel_l2(krt.grad, krr.grad), 6),
        "dv_rel_l2": round(rel_l2(vt.grad, vr.grad), 6),
    }

    # Weight-Absorbed BF16 Parity
    qap = torch.randn((Bp, Hp, Sp, 512), generator=gen_p, device=device, dtype=torch.bfloat16)
    ckvp = torch.randn((Bp, Sp, 512), generator=gen_p, device=device, dtype=torch.bfloat16)
    krsp = krp.squeeze(1).contiguous()
    doap = torch.randn_like(qap)

    qar, qrar, ckvr, krsr = [
        x.float().requires_grad_(True) for x in (qap, qrp, ckvp, krsp)
    ]
    with sdpa_kernel(SDPBackend.MATH):
        or_a = F.scaled_dot_product_attention(
            torch.cat([qar, qrar], dim=-1),
            torch.cat([ckvr, krsr], dim=-1).unsqueeze(1).expand(-1, Hp, -1, -1),
            ckvr.unsqueeze(1).expand(-1, Hp, -1, -1),
            is_causal=True,
            scale=sm_f,
        )
    (or_a * doap.float()).sum().backward()

    qat, qrat, ckvt, krst = [
        x.clone().requires_grad_(True) for x in (qap, qrp, ckvp, krsp)
    ]
    ot_a = triton_absorbed_mla(qat, qrat, ckvt, krst, sm_f)
    torch.autograd.backward(ot_a, doap)

    absorbed_parity = {
        "out_rel_l2": round(rel_l2(ot_a, or_a), 6),
        "dq_abs_rel_l2": round(rel_l2(qat.grad, qar.grad), 6),
        "dq_rope_rel_l2": round(rel_l2(qrat.grad, qrar.grad), 6),
        "dc_kv_rel_l2": round(rel_l2(ckvt.grad, ckvr.grad), 6),
        "dk_rope_rel_l2": round(rel_l2(krst.grad, krsr.grad), 6),
    }

    # Fused Latent RMSNorm + RoPE Parity
    raw_in = torch.randn((Bp, Sp, 576), generator=gen_p, device=device, dtype=torch.bfloat16)
    w_nm = torch.ones((512,), device=device, dtype=torch.bfloat16)
    pos_p = torch.arange(Sp, device=device).unsqueeze(0).expand(Bp, Sp)
    cos_p, sin_p = build_rope_cache(pos_p)
    dc_g = torch.randn((Bp, Sp, 512), generator=gen_p, device=device, dtype=torch.bfloat16)
    dkr_g = torch.randn((Bp, Sp, 64), generator=gen_p, device=device, dtype=torch.bfloat16)

    raw_r = raw_in.float().requires_grad_(True)
    wn_r = w_nm.float().requires_grad_(True)
    cr_part = raw_r[..., :512]
    cn_ref = cr_part * torch.rsqrt(cr_part.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * wn_r
    kr_ref = apply_rope_bhsd(raw_r[..., 512:].unsqueeze(1), cos_p, sin_p).squeeze(1)
    ((cn_ref * dc_g.float()).sum() + (kr_ref * dkr_g.float()).sum()).backward()

    raw_t = raw_in.clone().requires_grad_(True)
    wn_t = w_nm.clone().requires_grad_(True)
    cn_tri, kr_tri = fused_latent_rmsnorm_rope(raw_t, wn_t, cos_p, sin_p, 1e-5)
    torch.autograd.backward([cn_tri, kr_tri], [dc_g, dkr_g])

    norm_rope_parity = {
        "c_norm_rel_l2": round(rel_l2(cn_tri, cn_ref), 6),
        "k_rope_rel_l2": round(rel_l2(kr_tri, kr_ref), 6),
        "d_raw_rel_l2": round(rel_l2(raw_t.grad, raw_r.grad), 6),
        "dw_norm_rel_l2": round(rel_l2(wn_t.grad, wn_r.grad), 6),
    }

    # 4. SM80 Split-MLA Kernel Physics Ablation (`B=1, H=32, S=2048` in smoke, `S=4096` in full)
    Sa = 2048 if is_smoke else 4096
    Ha = 32
    warmup_a = 2 if is_smoke else 4
    iters_a = 4 if is_smoke else 10

    qna = torch.randn((1, Ha, Sa, 128), device=device, dtype=torch.bfloat16)
    qra = torch.randn((1, Ha, Sa, 64), device=device, dtype=torch.bfloat16)
    kna = torch.randn((1, Ha, Sa, 128), device=device, dtype=torch.bfloat16)
    kra = torch.randn((1, 1, Sa, 64), device=device, dtype=torch.bfloat16)
    va = torch.randn((1, Ha, Sa, 128), device=device, dtype=torch.bfloat16)
    doa = torch.randn_like(va)

    # (a) 1-Pass Global Atomic dQ negative control (only 2 iters since L2 atomic serialization is ~150-200ms!)
    oa_tmp, lsea_tmp = _mla_split_fwd_raw(qna, qra, kna, kra, va, sm_f, causal=True)
    dela_tmp = torch.empty_like(lsea_tmp)
    _mla_bwd_prep_kernel[(triton.cdiv(Ha * Sa, 64),)](
        oa_tmp, doa, dela_tmp, Ha * Sa, D_V=128, num_warps=4
    )
    dqn_acc = torch.zeros((1, Ha, Sa, 128), device=device, dtype=torch.float32)
    dqr_acc = torch.zeros((1, Ha, Sa, 64), device=device, dtype=torch.float32)
    dkn_tmp = torch.empty_like(kna)
    dkr_pt = torch.empty((1, Ha, Sa, 64), device=device, dtype=torch.float32)
    dv_tmp = torch.empty_like(va)

    def _run_atomic_bwd() -> None:
        dqn_acc.zero_()
        dqr_acc.zero_()
        _mla_split_bwd_onepass_atomic_control_kernel[(Sa // 64, Ha)](
            qna, qra, kna, kra, va, doa, lsea_tmp, dela_tmp,
            dqn_acc, dqr_acc, dkn_tmp, dkr_pt, dv_tmp,
            sm_f, sm_f * RCP_LN2, Ha, Sa, Sa, BM=64, BN=64, num_warps=4, num_stages=2,
        )

    def _run_stages1_bwd() -> None:
        dqn_s = torch.empty_like(qna)
        dqr_s = torch.empty_like(qra)
        _mla_split_bwd_dkdv_kernel[(Sa // 64, Ha)](
            qna, qra, kna, kra, va, doa, lsea_tmp, dela_tmp, dkn_tmp, dkr_pt, dv_tmp,
            sm_f, sm_f * RCP_LN2, Ha, Sa, Sa, BM=64, BN=64, CAUSAL=True, ACCUM_DKDV=False,
            num_warps=4, num_stages=1,
        )
        _mla_split_bwd_dq_kernel[(Sa // 64, Ha)](
            qna, qra, kna, kra, va, doa, lsea_tmp, dela_tmp, dqn_s, dqr_s,
            sm_f, sm_f * RCP_LN2, Ha, Sa, Sa, BM=64, BN=64, CAUSAL=True, ACCUM_DQ=False,
            num_warps=4, num_stages=1,
        )

    def _run_stages2_bwd() -> None:
        _ = _mla_split_bwd_raw(qna, qra, kna, kra, va, oa_tmp, doa, lsea_tmp, sm_f, causal=True)

    t_atom = time_fn_with_peak_mem(_run_atomic_bwd, warmup=1, iters=3, device=device)
    t_stg1 = time_fn_with_peak_mem(_run_stages1_bwd, warmup=1, iters=3, device=device)
    t_stg2 = time_fn_with_peak_mem(_run_stages2_bwd, warmup=warmup_a, iters=iters_a, device=device)

    # Full Fwd+Bwd attention-only autograd memory & latency: Triton Split-MLA vs FA2 V-padded vs Cutlass SDPA
    qna_g, qra_g, kna_g, kra_g, va_g = [
        x.clone().requires_grad_(True) for x in (qna, qra, kna, kra, va)
    ]

    def _run_tri_autograd() -> None:
        for tx in (qna_g, qra_g, kna_g, kra_g, va_g):
            tx.grad = None
        o = triton_split_mla(qna_g, qra_g, kna_g, kra_g, va_g, sm_f)
        torch.autograd.backward(o, doa)

    def _run_fa2_vpad_autograd() -> None:
        for tx in (qna_g, qra_g, kna_g, kra_g, va_g):
            tx.grad = None
        qc = torch.cat([qna_g, qra_g], dim=-1).transpose(1, 2).contiguous()
        kc = torch.cat([kna_g, kra_g.expand(-1, Ha, -1, -1)], dim=-1).transpose(1, 2).contiguous()
        vp_a = F.pad(va_g.transpose(1, 2).contiguous(), (0, 64))
        op = flash_attn_func(qc, kc, vp_a, causal=True, softmax_scale=sm_f)
        torch.autograd.backward(op[..., :128].transpose(1, 2), doa)

    def _run_sdpa_eff_autograd() -> None:
        for tx in (qna_g, qra_g, kna_g, kra_g, va_g):
            tx.grad = None
        qc = torch.cat([qna_g, qra_g], dim=-1)
        kc = torch.cat([kna_g, kra_g.expand(-1, Ha, -1, -1)], dim=-1)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o = F.scaled_dot_product_attention(qc, kc, va_g, is_causal=True, scale=sm_f)
        torch.autograd.backward(o, doa)

    t_tri_ag = time_fn_with_peak_mem(_run_tri_autograd, warmup=warmup_a, iters=iters_a, device=device)
    t_fa2_ag = time_fn_with_peak_mem(_run_fa2_vpad_autograd, warmup=warmup_a, iters=iters_a, device=device)
    t_eff_ag = time_fn_with_peak_mem(_run_sdpa_eff_autograd, warmup=warmup_a, iters=iters_a, device=device)

    del qna, qra, kna, kra, va, doa, oa_tmp, lsea_tmp, dela_tmp, dqn_acc, dqr_acc, dkn_tmp, dkr_pt, dv_tmp
    del qna_g, qra_g, kna_g, kra_g, va_g
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "ecosystem_compatibility_audit": {
            "decompressed_192_to_128": decomp_audit,
            "weight_absorbed_576_to_512_mqa": absorbed_audit,
        },
        "fp64_chain_rule_refinement": {
            "split_mla_ratios_h_vs_half_h": split_fd_ratios,
            "absorbed_mla_ratios_h_vs_half_h": abs_fd_ratios,
            "min_refinement_ratio": round(
                min(min(split_fd_ratios.values()), min(abs_fd_ratios.values())), 4
            ),
        },
        "bf16_triton_parity": {
            "split_mla_128_64_to_128": split_parity,
            "absorbed_mla_512_64_to_512": absorbed_parity,
            "fused_latent_rmsnorm_rope": norm_rope_parity,
        },
        "sm80_split_mla_kernel_ablation": {
            "seq_len": Sa,
            "heads": Ha,
            "bwd_only_1pass_global_atomic_dq": t_atom,
            "bwd_only_2pass_sync_stages1": t_stg1,
            "bwd_only_2pass_cp_async_stages2": t_stg2,
            "speedup_2pass_vs_global_atomic_dq": round(
                t_atom["median_ms"] / max(t_stg2["median_ms"], 1e-6), 4
            ),
            "speedup_stages2_vs_stages1": round(
                t_stg1["median_ms"] / max(t_stg2["median_ms"], 1e-6), 4
            ),
            "autograd_fwd_bwd_triton_split_mla": t_tri_ag,
            "autograd_fwd_bwd_fa2_vpadded_192": t_fa2_ag,
            "autograd_fwd_bwd_sdpa_cutlass_efficient": t_eff_ag,
            "memory_reduction_vs_fa2_vpadded": round(
                t_fa2_ag["peak_inc_mib"] / max(t_tri_ag["peak_inc_mib"], 1e-6), 4
            ),
            "speedup_vs_sdpa_cutlass_efficient": round(
                t_eff_ag["median_ms"] / max(t_tri_ag["median_ms"], 1e-6), 4
            ),
        },
    }


# ============================================================================
# 7. Suite A: 1-GPU End-to-End DeepSeek-V3 MLA Layer Scaling (Fwd + Bwd)
# ============================================================================


def run_suite_a_mla_layer_scaling(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    seq_lens = [2048, 8192] if is_smoke else [2048, 4096, 8192, 16384, 32768]
    H_mod, n_h = (2048, 16) if is_smoke else (4096, 32)
    warmup = 2 if is_smoke else 4
    iters = 4 if is_smoke else 10

    layer = DeepSeekMLALayer(H_mod, n_h, d_c=512, d_q_c=1536, d_nope=128, d_rope=64, d_v=128).to(
        device=device, dtype=torch.bfloat16
    )
    cflex = torch.compile(flex_attention)
    results: dict[str, Any] = {}

    for S in seq_lens:
        x_in = torch.randn((1, S, H_mod), device=device, dtype=torch.bfloat16)
        dy = torch.randn_like(x_in)
        pos = torch.arange(S, device=device).unsqueeze(0)
        cos, sin = build_rope_cache(pos)
        modes_out: dict[str, Any] = {}

        mode_list = [
            "mla_sdpa_math_eager",
            "mla_sdpa_cutlass_efficient",
            "mla_flex_attention_compiled",
            "mla_fa2_vpadded_eager",
            "mla_triton_split_fused",
            "mla_triton_weight_absorbed",
        ]
        for mk in mode_list:
            if mk == "mla_sdpa_math_eager" and S > 4096:
                continue
            if mk == "mla_triton_weight_absorbed" and S > 8192:
                continue
            if mk == "mla_flex_attention_compiled" and S > 16384:
                continue
            if is_smoke and mk == "mla_flex_attention_compiled" and S > 2048:
                continue

            xr = x_in.detach().clone().requires_grad_(True)

            def _step_m(m_name: str = mk) -> None:
                for p in layer.parameters():
                    p.grad = None
                xr.grad = None
                y = layer.forward_mode(xr, cos, sin, m_name, compiled_flex=cflex)
                torch.autograd.backward(y, dy)

            t0 = time.perf_counter()
            _step_m()
            torch.cuda.synchronize(device)
            first_s = time.perf_counter() - t0

            wu_use = 1 if (mk == "mla_triton_weight_absorbed" and S >= 8192) else warmup
            it_use = min(3, iters) if (mk == "mla_triton_weight_absorbed" and S >= 8192) else iters
            tm = time_fn_with_peak_mem(_step_m, warmup=wu_use, iters=it_use, device=device)
            if mk == "mla_flex_attention_compiled":
                tm["first_call_compile_s"] = round(first_s, 2)
            modes_out[mk] = tm
            del xr
            gc.collect()
            torch.cuda.empty_cache()

        results[str(S)] = {
            "hidden": H_mod,
            "heads": n_h,
            "seq_len": S,
            "modes": modes_out,
        }

    del layer
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================================
# 8. Suite B: Prefill-to-Decode Weight-Absorption Crossover & Latent-SAC Stack
# ============================================================================


class MLATransformerBlock(nn.Module):
    def __init__(self, hidden: int, ffn_dim: int, n_heads: int) -> None:
        super().__init__()
        self.mla = DeepSeekMLALayer(hidden, n_heads, d_c=512, d_q_c=1536)
        self.ffn_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.w_gate = nn.Linear(hidden, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden, bias=False)

    def forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ffn_norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class _LatentZeroAttnRecompFn(torch.autograd.Function):
    """Saves ONLY compressed latents `(c_Q [1536], c_KV [512], k_R [64])` + `(O, LSE2)`
    for attention — discarding decompressed `(Q_nope, Q_rope, K_nope, V)` (`14,336`
    BF16/tok at `n_h = 32`!) and recomputing only `O(S)` linear `W_UQ_QR` and `W_UK_UV`
    in backward with ZERO `O(S^2)` forward attention recomputation!"""

    @staticmethod
    def forward(
        ctx: Any,
        c_q: torch.Tensor,
        c_kv: torch.Tensor,
        k_rope_bs: torch.Tensor,
        w_uq_qr: torch.Tensor,
        w_uk_uv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        n_heads: int,
        sm_scale: float,
    ) -> torch.Tensor:
        B, S, _ = c_q.shape
        q_all = F.linear(c_q, w_uq_qr).view(B, S, n_heads, 192).transpose(1, 2)
        q_nope = q_all[..., :128].contiguous()
        q_rope = apply_rope_bhsd(q_all[..., 128:], cos, sin).contiguous()

        kv_up = F.linear(c_kv, w_uk_uv).view(B, S, n_heads, 256).transpose(1, 2)
        k_nope = kv_up[..., :128].contiguous()
        v = kv_up[..., 128:].contiguous()
        k_rope = k_rope_bs.unsqueeze(1)

        out, lse2 = _mla_split_fwd_raw(q_nope, q_rope, k_nope, k_rope, v, sm_scale, causal=True)
        # Discard q_nope, q_rope, k_nope, v! Save ONLY compact c_q, c_kv, k_rope_bs, out, lse2!
        ctx.save_for_backward(c_q, c_kv, k_rope_bs, w_uq_qr, w_uk_uv, cos, sin, out, lse2)
        ctx.n_heads = n_heads
        ctx.sm_scale = sm_scale
        return out.transpose(1, 2).contiguous().view(B, S, n_heads * 128)

    @staticmethod
    def backward(ctx: Any, do_flat: torch.Tensor) -> tuple[Any, ...]:
        c_q, c_kv, k_rope_bs, w_uq_qr, w_uk_uv, cos, sin, out, lse2 = ctx.saved_tensors
        H = ctx.n_heads
        B, S, _ = c_q.shape
        # Cheap O(S) linear up-projection recomputation on Tensor Cores (NO O(S^2) attention fwd!)
        q_all = F.linear(c_q, w_uq_qr).view(B, S, H, 192).transpose(1, 2)
        q_nope = q_all[..., :128].contiguous()
        q_rope = apply_rope_bhsd(q_all[..., 128:], cos, sin).contiguous()

        kv_up = F.linear(c_kv, w_uk_uv).view(B, S, H, 256).transpose(1, 2)
        k_nope = kv_up[..., :128].contiguous()
        v = kv_up[..., 128:].contiguous()
        k_rope = k_rope_bs.unsqueeze(1)

        do_bh = do_flat.view(B, S, H, 128).transpose(1, 2).contiguous()
        dqn, dqr, dkn, dkr, dv = _mla_split_bwd_raw(
            q_nope, q_rope, k_nope, k_rope, v, out, do_bh, lse2, ctx.sm_scale, causal=True
        )
        # Inverse RoPE on dqr: rotate by (cos, -sin)
        dqr_unrot = apply_rope_bhsd(dqr, cos, -sin)
        dq_all = torch.cat([dqn, dqr_unrot], dim=-1).transpose(1, 2).contiguous().view(B * S, H * 192)
        dkv_all = torch.cat([dkn, dv], dim=-1).transpose(1, 2).contiguous().view(B * S, H * 256)

        dc_q = torch.matmul(dq_all, w_uq_qr).view(B, S, -1)
        dw_uq = torch.matmul(dq_all.t(), c_q.view(B * S, -1))
        dc_kv = torch.matmul(dkv_all, w_uk_uv).view(B, S, -1)
        dw_ukv = torch.matmul(dkv_all.t(), c_kv.view(B * S, -1))
        return dc_q, dc_kv, dkr.squeeze(1), dw_uq, dw_ukv, None, None, None, None


def run_suite_b_crossover_and_sac(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    # Part B.1: Query-Length Crossover `S_q` at `L_kv = 16,384` (`B = 8, H = 32` in full)
    # Compares Decompressed KV-Cache (`8,256` BF16/tok = `2,064 MiB` at `B=8, L=16K`) vs
    # Weight-Absorbed Compressed-Latent KV-Cache (`576` BF16/tok = `144 MiB`, `14.33x` smaller!)
    B_x = 4 if is_smoke else 8
    H_x = 32
    L_kv = 8192 if is_smoke else 16384
    sq_list = [1, 64, 1024] if is_smoke else [1, 16, 64, 256, 1024, 4096]
    warmup_b1 = 2 if is_smoke else 4
    iters_b1 = 5 if is_smoke else 12
    sm_s = 1.0 / math.sqrt(192.0)

    c_kv_cache = torch.randn((B_x, L_kv, 512), device=device, dtype=torch.bfloat16)
    k_r_cache = torch.randn((B_x, L_kv, 64), device=device, dtype=torch.bfloat16)
    w_uk = torch.randn((H_x, 128, 512), device=device, dtype=torch.bfloat16) * 0.02
    w_uv_t = (torch.randn((H_x, 512, 128), device=device, dtype=torch.bfloat16) * 0.02).contiguous()

    # Pre-materialized decompressed KV cache (`[B, L_kv, H, 192]` K + `[B, L_kv, H, 192]` V-padded for FA2)
    k_dec_cache = torch.randn((B_x, L_kv, H_x, 192), device=device, dtype=torch.bfloat16)
    v_dec_cache = torch.randn((B_x, L_kv, H_x, 192), device=device, dtype=torch.bfloat16)

    latent_cache_mib = round((B_x * L_kv * 576 * 2) / (1024.0**2), 2)
    decomp_logical_mib = round((B_x * L_kv * (H_x * 256 + 64) * 2) / (1024.0**2), 2)

    ws_fi = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    ps_fi = 16
    npages = (B_x * L_kv) // ps_fi
    ckv_paged = c_kv_cache.view(npages, ps_fi, 512)
    kr_paged = k_r_cache.view(npages, ps_fi, 64)

    crossover_out: dict[str, Any] = {}
    for Sq in sq_list:
        qn_q = torch.randn((B_x, H_x, Sq, 128), device=device, dtype=torch.bfloat16)
        qr_q = torch.randn((B_x, H_x, Sq, 64), device=device, dtype=torch.bfloat16)
        qc_fa = torch.cat([qn_q, qr_q], dim=-1).transpose(1, 2).contiguous()
        qn_h_bs_d = qn_q.permute(1, 0, 2, 3).contiguous().view(H_x, B_x * Sq, 128)

        # 1. Decompressed KV-cache + FA2 V-padded
        def _run_decomp_kv() -> None:
            _ = flash_attn_func(
                qc_fa, k_dec_cache, v_dec_cache, causal=(Sq > 1), softmax_scale=sm_s
            )[..., :128]

        t_dec_kv = time_fn_with_peak_mem(_run_decomp_kv, warmup=warmup_b1, iters=iters_b1, device=device)

        # 2. Weight-Absorbed Triton MLA (`q_abs = bmm(q_nope, W_uk)`, attend `[c_KV | k_R]`, `o = bmm(o_lat, W_uv^T)`)
        num_sp = 64 if L_kv >= 16384 else 32
        kv_ps = (L_kv + num_sp - 1) // num_sp
        part_o = torch.empty((B_x, num_sp, H_x, 512), device=device, dtype=torch.float32)
        part_lse = torch.empty((B_x, num_sp, H_x), device=device, dtype=torch.float32)
        out_s1 = torch.empty((B_x, H_x, 512), device=device, dtype=torch.bfloat16)

        def _run_abs_triton() -> None:
            qa_h_bs = torch.bmm(qn_h_bs_d, w_uk)  # [H, B*Sq, 512]
            if Sq == 1:
                qa_1 = qa_h_bs.transpose(0, 1).contiguous()  # [B, H, 512]
                qr_1 = qr_q.squeeze(2).contiguous()
                _mla_absorbed_splitkv_decode_stage1[(num_sp, B_x)](
                    qa_1, qr_1, c_kv_cache, k_r_cache, part_o, part_lse,
                    sm_s * RCP_LN2, L_kv, kv_ps, NH=H_x, BN=64, num_warps=4, num_stages=2,
                )
                _mla_absorbed_splitkv_decode_stage2[(B_x * H_x,)](
                    part_o, part_lse, out_s1, H_x, NUM_SPLITS=num_sp, num_warps=4
                )
                _ = torch.bmm(out_s1.transpose(0, 1), w_uv_t)
            else:
                qa_cur = qa_h_bs.view(H_x, B_x, Sq, 512).permute(1, 0, 2, 3).contiguous()
                out_lat = torch.empty((B_x, H_x, Sq, 512), device=device, dtype=torch.bfloat16)
                lse_lat = torch.empty((B_x, H_x, Sq), device=device, dtype=torch.float32)
                _mla_absorbed_fwd_kernel[(triton.cdiv(Sq, 32), B_x * H_x)](
                    qa_cur, qr_q, c_kv_cache, k_r_cache, out_lat, lse_lat,
                    sm_s * RCP_LN2, H_x, Sq, L_kv, L_kv - Sq,
                    BM=32, BN=32, CAUSAL=True, num_warps=4, num_stages=2,
                )
                _ = torch.bmm(out_lat.permute(1, 0, 2, 3).contiguous().view(H_x, B_x * Sq, 512), w_uv_t)

        t_abs_tri = time_fn_with_peak_mem(_run_abs_triton, warmup=warmup_b1, iters=iters_b1, device=device)

        entry_sq: dict[str, Any] = {
            "batch": B_x,
            "heads": H_x,
            "kv_context_len": L_kv,
            "query_len": Sq,
            "latent_kv_cache_mib": latent_cache_mib,
            "decompressed_kv_cache_mib": decomp_logical_mib,
            "kv_cache_compression_ratio": round(decomp_logical_mib / max(latent_cache_mib, 1e-6), 2),
            "decompressed_kv_fa2": t_dec_kv,
            "weight_absorbed_triton_mla": t_abs_tri,
            "absorbed_triton_speedup_vs_decompressed": round(
                t_dec_kv["median_ms"] / max(t_abs_tri["median_ms"], 1e-6), 4
            ),
        }

        # At Sq == 1, also benchmark production FlashInfer FA2 Paged MLA + bmm W_uk/W_uv absorption!
        if Sq == 1:
            wm = flashinfer.BatchMLAPagedAttentionWrapper(ws_fi, backend="fa2")
            qo_ind = torch.arange(0, B_x + 1, dtype=torch.int32, device=device)
            kv_ind = torch.arange(0, B_x + 1, dtype=torch.int32, device=device) * (L_kv // ps_fi)
            kv_idx = torch.arange(0, npages, dtype=torch.int32, device=device)
            kv_len = torch.full((B_x,), L_kv, dtype=torch.int32, device=device)
            wm.plan(
                qo_ind, kv_ind, kv_idx, kv_len, H_x, 512, 64, ps_fi, False,
                sm_s, torch.bfloat16, torch.bfloat16,
            )

            def _run_abs_fi() -> None:
                qa_1 = torch.bmm(qn_h_bs_d, w_uk).transpose(0, 1).contiguous()
                qr_1 = qr_q.squeeze(2).contiguous()
                o_lat = wm.run(qa_1, qr_1, ckv_paged, kr_paged)
                _ = torch.bmm(o_lat.transpose(0, 1), w_uv_t)

            t_abs_fi = time_fn_with_peak_mem(_run_abs_fi, warmup=warmup_b1, iters=iters_b1, device=device)
            entry_sq["weight_absorbed_flashinfer_fa2_paged"] = t_abs_fi
            entry_sq["absorbed_flashinfer_speedup_vs_decompressed"] = round(
                t_dec_kv["median_ms"] / max(t_abs_fi["median_ms"], 1e-6), 4
            )

        crossover_out[str(Sq)] = entry_sq

    del c_kv_cache, k_r_cache, k_dec_cache, v_dec_cache, ws_fi
    gc.collect()
    torch.cuda.empty_cache()

    # Part B.2: 4-Layer DeepSeek-V3-Width (`H=4096, n_h=32, FFN=11008`) `Latent-SAC` Stack
    sac_seqs = [4096] if is_smoke else [4096, 16384]
    H_s, I_s, nh_s = (2048, 5632, 16) if is_smoke else (4096, 11008, 32)
    n_layers = 4
    warmup_b2 = 2 if is_smoke else 3
    iters_b2 = 4 if is_smoke else 8

    blocks = nn.ModuleList([MLATransformerBlock(H_s, I_s, nh_s) for _ in range(n_layers)]).to(
        device=device, dtype=torch.bfloat16
    )

    # First verify gradient parity of `compressed_latent_zero_attn_recomp_sac` vs `no_ac` at S = 1024
    x_v = torch.randn((1, 1024, H_s), device=device, dtype=torch.bfloat16)
    dy_v = torch.randn_like(x_v)
    cos_v, sin_v = build_rope_cache(torch.arange(1024, device=device).unsqueeze(0))

    def _fwd_policy(x_cur: torch.Tensor, c_p: torch.Tensor, s_p: torch.Tensor, pol: str) -> torch.Tensor:
        for blk in blocks:
            if pol == "no_ac":
                x_cur = blk.mla.forward_mode(x_cur, c_p, s_p, "mla_triton_split_fused")
                x_cur = blk.forward_ffn(x_cur)
            elif pol == "compressed_latent_sac":
                # Save ONLY latents (c_q, c_kv, k_r) + (O, LSE2) in attention (0 O(S^2) attn recompute!) + checkpoint FFN
                m = blk.mla
                h = m.in_norm(x_cur)
                cq = m.q_norm(m.w_dq(h))
                ckv, kr_bs = fused_latent_rmsnorm_rope(m.w_dkv_kr(h), m.kv_norm.weight, c_p, s_p, 1e-5)
                o_flat = _LatentZeroAttnRecompFn.apply(
                    cq, ckv, kr_bs, m.w_uq_qr.weight, m.w_uk_uv.weight, c_p, s_p, m.n_heads, m.sm_scale
                )
                x_cur = x_cur + m.w_o(o_flat)
                x_cur = checkpoint(blk.forward_ffn, x_cur, use_reentrant=False)
            elif pol == "full_layer_ac":
                def _full_blk(xin: torch.Tensor, b_mod: MLATransformerBlock = blk) -> torch.Tensor:
                    t = b_mod.mla.forward_mode(xin, c_p, s_p, "mla_triton_split_fused")
                    return b_mod.forward_ffn(t)

                x_cur = checkpoint(_full_blk, x_cur, use_reentrant=False)
            else:
                raise ValueError(pol)
        return x_cur

    xv1 = x_v.clone().requires_grad_(True)
    for p in blocks.parameters():
        p.grad = None
    y_no = _fwd_policy(xv1, cos_v, sin_v, "no_ac")
    torch.autograd.backward(y_no, dy_v)
    dx_no = xv1.grad.clone()
    dw_no = blocks[0].mla.w_uk_uv.weight.grad.clone()

    xv2 = x_v.clone().requires_grad_(True)
    for p in blocks.parameters():
        p.grad = None
    y_ls = _fwd_policy(xv2, cos_v, sin_v, "compressed_latent_sac")
    torch.autograd.backward(y_ls, dy_v)
    sac_parity = {
        "out_rel_l2": round(rel_l2(y_ls, y_no), 6),
        "dx_rel_l2": round(rel_l2(xv2.grad, dx_no), 6),
        "dw_ukv_rel_l2": round(rel_l2(blocks[0].mla.w_uk_uv.weight.grad, dw_no), 6),
    }

    sac_scaling: dict[str, Any] = {}
    for S_sc in sac_seqs:
        xs = torch.randn((1, S_sc, H_s), device=device, dtype=torch.bfloat16)
        dys = torch.randn_like(xs)
        cos_s, sin_s = build_rope_cache(torch.arange(S_sc, device=device).unsqueeze(0))
        pol_res: dict[str, Any] = {}
        for pol in ("no_ac", "compressed_latent_sac", "full_layer_ac"):
            x_req = xs.detach().clone().requires_grad_(True)

            def _run_pol(p_name: str = pol) -> None:
                for p in blocks.parameters():
                    p.grad = None
                x_req.grad = None
                yo = _fwd_policy(x_req, cos_s, sin_s, p_name)
                torch.autograd.backward(yo, dys)

            pol_res[pol] = time_fn_with_peak_mem(
                _run_pol, warmup=warmup_b2, iters=iters_b2, device=device
            )
            del x_req
            gc.collect()
            torch.cuda.empty_cache()
        sac_scaling[str(S_sc)] = pol_res

    del blocks
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "prefill_to_decode_absorption_crossover": crossover_out,
        "latent_sac_parity_vs_no_ac": sac_parity,
        "four_layer_latent_sac_scaling": sac_scaling,
    }


# ============================================================================
# 9. Suite C: 8-GPU Zig-Zag Ring Context Parallelism (Latent-Ring vs Decomp/GQA)
# ============================================================================


def _zigzag_local_positions(S_global: int, rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    """Returns `[1, 2 * S_chunk]` global token indices `[c0_indices, c1_indices]` for Zig-Zag Ring."""
    n_chunks = 2 * world_size
    s_ch = S_global // n_chunks
    c0 = rank
    c1 = (2 * world_size - 1) - rank
    p0 = torch.arange(c0 * s_ch, (c0 + 1) * s_ch, device=device, dtype=torch.long)
    p1 = torch.arange(c1 * s_ch, (c1 + 1) * s_ch, device=device, dtype=torch.long)
    return torch.cat([p0, p1], dim=0).unsqueeze(0)


def _zigzag_pack_from_full(full_t: torch.Tensor, rank: int, world_size: int, seq_dim: int) -> torch.Tensor:
    S_global = full_t.shape[seq_dim]
    s_ch = S_global // (2 * world_size)
    c0 = rank
    c1 = (2 * world_size - 1) - rank
    idx0 = slice(c0 * s_ch, (c0 + 1) * s_ch)
    idx1 = slice(c1 * s_ch, (c1 + 1) * s_ch)
    sl0 = [slice(None)] * full_t.ndim
    sl1 = [slice(None)] * full_t.ndim
    sl0[seq_dim] = idx0
    sl1[seq_dim] = idx1
    return torch.cat([full_t[tuple(sl0)], full_t[tuple(sl1)]], dim=seq_dim).contiguous()


class _CP8CompressedLatentZigZagFn(torch.autograd.Function):
    """8-GPU Zig-Zag Ring MLA exchanging ONLY `[c_KV (512) | k_R (64)]` (`576` BF16/tok —
    `14.33x` smaller than decompressed `(K_nope, V, K_rope)` `8,256` BF16/tok!),
    using chunk-first `[2, B, H, Sc, D]` zero-copy contiguous views and active-chunk-only
    `W_UK_UV` up-projection overlapped with P2P ring transfers!"""

    @staticmethod
    def forward(
        ctx: Any,
        q_nope: torch.Tensor,   # [B, H, 2*Sc, 128]
        q_rope: torch.Tensor,   # [B, H, 2*Sc, 64]
        c_kv: torch.Tensor,     # [B, 2*Sc, 512]
        k_rope: torch.Tensor,   # [B, 2*Sc, 64]
        w_uk_uv: torch.Tensor,  # [H * 256, 512]
        sm_scale: float,
    ) -> torch.Tensor:
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, _ = q_nope.shape
        Sc = S_loc // 2
        dst = (rank + 1) % w_sz
        src = (rank - 1) % w_sz

        # Chunk-first `[2, B, H, Sc, D]` so `qn_p[a]` is 100% contiguous!
        qn_p = q_nope.view(B, H, 2, Sc, 128).permute(2, 0, 1, 3, 4).contiguous()
        qr_p = q_rope.view(B, H, 2, Sc, 64).permute(2, 0, 1, 3, 4).contiguous()
        lat_local = torch.cat([c_kv, k_rope], dim=-1).contiguous()  # [B, 2*Sc, 576]

        # Pre-allocated ping-pong ring buffers (prevents PyTorch caching-allocator stream aliasing with ncclStream!)
        lat_bufs = [lat_local.clone(), torch.empty_like(lat_local)]
        buf_idx = 0

        o_acc = torch.zeros((2, B, H, Sc, 128), device=q_nope.device, dtype=torch.float32)
        lse_acc = torch.full((2, B, H, Sc), -1.0e20, device=q_nope.device, dtype=torch.float32)
        inited = [False, False]
        n_rows_ch = B * H * Sc
        c_r = [rank, 2 * w_sz - 1 - rank]

        for step in range(w_sz):
            cur_lat = lat_bufs[buf_idx]
            nxt_lat = lat_bufs[1 - buf_idx]
            reqs = []
            if step < w_sz - 1:
                reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, cur_lat, dst),
                        dist.P2POp(dist.irecv, nxt_lat, src),
                    ]
                )

            src_rank = (rank - step) % w_sz
            c_j = [src_rank, 2 * w_sz - 1 - src_rank]
            lat_2 = cur_lat.view(B, 2, Sc, 576)

            for b in (0, 1):
                # Skip `W_UK_UV` up-projection completely if KV chunk `c_j[b]` is strictly in the future of both local query chunks!
                if c_r[0] < c_j[b] and c_r[1] < c_j[b]:
                    continue
                ckv_b = lat_2[:, b, :, :512].contiguous()
                kr_b = lat_2[:, b, :, 512:].contiguous().unsqueeze(1)  # [B, 1, Sc, 64]
                kv_up_b = F.linear(ckv_b, w_uk_uv).view(B, Sc, H, 256).transpose(1, 2)
                kn_b = kv_up_b[..., :128].contiguous()
                v_b = kv_up_b[..., 128:].contiguous()

                for a in (0, 1):
                    if c_r[a] < c_j[b]:
                        continue
                    is_diag = c_r[a] == c_j[b]
                    o_sub, lse_sub = _mla_split_fwd_raw(
                        qn_p[a], qr_p[a], kn_b, kr_b, v_b, sm_scale, causal=is_diag
                    )
                    _ring_lse2_merge_kernel[(triton.cdiv(n_rows_ch, 64),)](
                        o_acc[a],
                        lse_acc[a],
                        o_sub,
                        lse_sub,
                        n_rows_ch,
                        D=128,
                        INIT_STEP=(not inited[a]),
                        num_warps=4,
                    )
                    inited[a] = True

            if reqs:
                for rq in reqs:
                    rq.wait()
                buf_idx = 1 - buf_idx

        out_final = o_acc.permute(1, 2, 0, 3, 4).contiguous().to(torch.bfloat16).view(B, H, S_loc, 128)
        lse_final = lse_acc.permute(1, 2, 0, 3).contiguous().view(B, H, S_loc)
        ctx.save_for_backward(q_nope, q_rope, lat_local, w_uk_uv, out_final, lse_final)
        ctx.sm_scale = sm_scale
        return out_final

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        q_nope, q_rope, lat_local, w_uk_uv, out_final, lse_final = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        qk_scale = sm_scale * RCP_LN2
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, _ = q_nope.shape
        Sc = S_loc // 2
        dst = (rank + 1) % w_sz
        src = (rank - 1) % w_sz

        doc = do.contiguous()
        delta_full = torch.empty((B, H, S_loc), device=q_nope.device, dtype=torch.float32)
        _mla_bwd_prep_kernel[(triton.cdiv(B * H * S_loc, 64),)](
            out_final, doc, delta_full, B * H * S_loc, D_V=128, num_warps=4
        )

        # Chunk-first `[2, B, H, Sc, D]` so every `[a]` slice is 100% contiguous!
        qn_p = q_nope.view(B, H, 2, Sc, 128).permute(2, 0, 1, 3, 4).contiguous()
        qr_p = q_rope.view(B, H, 2, Sc, 64).permute(2, 0, 1, 3, 4).contiguous()
        do_p = doc.view(B, H, 2, Sc, 128).permute(2, 0, 1, 3, 4).contiguous()
        lse_p = lse_final.view(B, H, 2, Sc).permute(2, 0, 1, 3).contiguous()
        del_p = delta_full.view(B, H, 2, Sc).permute(2, 0, 1, 3).contiguous()

        dqn_p = torch.zeros_like(qn_p)
        dqr_p = torch.zeros_like(qr_p)
        dw_ukv_acc = torch.zeros_like(w_uk_uv, dtype=torch.float32)

        # Pre-allocated ping-pong ring buffers (eliminates caching-allocator / ncclStream aliasing!)
        lat_bufs = [lat_local.clone(), torch.empty_like(lat_local)]
        dlat_bufs = [torch.zeros_like(lat_local), torch.empty_like(lat_local)]
        buf_idx = 0
        c_r = [rank, 2 * w_sz - 1 - rank]

        for step in range(w_sz):
            cur_lat = lat_bufs[buf_idx]
            nxt_lat = lat_bufs[1 - buf_idx]
            cur_dlat = dlat_bufs[buf_idx]
            nxt_dlat = dlat_bufs[1 - buf_idx]

            src_rank = (rank - step) % w_sz
            c_j = [src_rank, 2 * w_sz - 1 - src_rank]

            lat_reqs = []
            if step < w_sz - 1:
                lat_reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, cur_lat, dst),
                        dist.P2POp(dist.irecv, nxt_lat, src),
                    ]
                )

            lat_2 = cur_lat.view(B, 2, Sc, 576)
            dlat_2 = cur_dlat.view(B, 2, Sc, 576).float()

            for b in (0, 1):
                if c_r[0] < c_j[b] and c_r[1] < c_j[b]:
                    continue
                ckv_b = lat_2[:, b, :, :512].contiguous()
                kr_b = lat_2[:, b, :, 512:].contiguous().unsqueeze(1)
                kv_up_b = F.linear(ckv_b, w_uk_uv).view(B, Sc, H, 256).transpose(1, 2)
                kn_b = kv_up_b[..., :128].contiguous()
                v_b = kv_up_b[..., 128:].contiguous()

                dkn_b = torch.zeros_like(kn_b)
                dv_b = torch.zeros_like(v_b)
                dkr_part_b = torch.zeros((B, H, Sc, 64), device=q_nope.device, dtype=torch.float32)
                dk_written = False

                for a in (0, 1):
                    if c_r[a] < c_j[b]:
                        continue
                    is_diag = c_r[a] == c_j[b]
                    _mla_split_bwd_dkdv_kernel[(Sc // 64, B * H)](
                        qn_p[a],
                        qr_p[a],
                        kn_b,
                        kr_b,
                        v_b,
                        do_p[a],
                        lse_p[a],
                        del_p[a],
                        dkn_b,
                        dkr_part_b,
                        dv_b,
                        sm_scale,
                        qk_scale,
                        H,
                        Sc,
                        Sc,
                        BM=64,
                        BN=64,
                        CAUSAL=is_diag,
                        ACCUM_DKDV=dk_written,
                        num_warps=4,
                        num_stages=2,
                    )
                    dk_written = True

                    _mla_split_bwd_dq_kernel[(Sc // 64, B * H)](
                        qn_p[a],
                        qr_p[a],
                        kn_b,
                        kr_b,
                        v_b,
                        do_p[a],
                        lse_p[a],
                        del_p[a],
                        dqn_p[a],
                        dqr_p[a],
                        sm_scale,
                        qk_scale,
                        H,
                        Sc,
                        Sc,
                        BM=64,
                        BN=64,
                        CAUSAL=is_diag,
                        ACCUM_DQ=True,
                        num_warps=4,
                        num_stages=2,
                    )

                dkv_flat = (
                    torch.cat([dkn_b, dv_b], dim=-1)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B * Sc, H * 256)
                )
                dc_kv_b = torch.matmul(dkv_flat, w_uk_uv).view(B, Sc, 512).float()
                dw_ukv_acc.add_(torch.matmul(dkv_flat.t(), ckv_b.view(B * Sc, 512)).float())
                dk_r_b = dkr_part_b.sum(dim=1)  # [B, Sc, 64] FP32
                dlat_2[:, b, :, :512].add_(dc_kv_b)
                dlat_2[:, b, :, 512:].add_(dk_r_b)

            cur_dlat.copy_(dlat_2.to(torch.bfloat16).view(B, 2 * Sc, 576))

            if lat_reqs:
                for rq in lat_reqs:
                    rq.wait()

            dlat_reqs = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, cur_dlat, dst),
                    dist.P2POp(dist.irecv, nxt_dlat, src),
                ]
            )
            for rq in dlat_reqs:
                rq.wait()
            buf_idx = 1 - buf_idx

        final_dlat = dlat_bufs[buf_idx]
        dist.all_reduce(dw_ukv_acc, op=dist.ReduceOp.SUM)
        return (
            dqn_p.permute(1, 2, 0, 3, 4).contiguous().view(B, H, S_loc, 128),
            dqr_p.permute(1, 2, 0, 3, 4).contiguous().view(B, H, S_loc, 64),
            final_dlat[..., :512].contiguous(),
            final_dlat[..., 512:].contiguous(),
            dw_ukv_acc.to(w_uk_uv.dtype),
            None,
        )


class _CP8DecompressedMLAZigZagFn(torch.autograd.Function):
    """Naive Decompressed MLA Zig-Zag Ring exchanging `(K_nope [H*128], V [H*128], K_rope [64])`
    (`8,256` BF16/tok at `H=32` — `14.33x` larger ring payload than Compressed-Latent Ring!)."""

    @staticmethod
    def forward(
        ctx: Any,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope_bs: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, _ = q_nope.shape
        Sc = S_loc // 2
        dst = (rank + 1) % w_sz
        src = (rank - 1) % w_sz

        qn_p = q_nope.view(B, H, 2, Sc, 128).permute(2, 0, 1, 3, 4).contiguous()
        qr_p = q_rope.view(B, H, 2, Sc, 64).permute(2, 0, 1, 3, 4).contiguous()

        kv_pack_loc = torch.cat(
            [
                k_nope.transpose(1, 2).contiguous().view(B, S_loc, H * 128),
                v.transpose(1, 2).contiguous().view(B, S_loc, H * 128),
                k_rope_bs.contiguous(),
            ],
            dim=-1,
        ).contiguous()

        kv_bufs = [kv_pack_loc.clone(), torch.empty_like(kv_pack_loc)]
        buf_idx = 0
        o_acc = torch.zeros((2, B, H, Sc, 128), device=q_nope.device, dtype=torch.float32)
        lse_acc = torch.full((2, B, H, Sc), -1.0e20, device=q_nope.device, dtype=torch.float32)
        inited = [False, False]
        n_rows_ch = B * H * Sc
        c_r = [rank, 2 * w_sz - 1 - rank]

        for step in range(w_sz):
            cur_kv = kv_bufs[buf_idx]
            nxt_kv = kv_bufs[1 - buf_idx]
            reqs = []
            if step < w_sz - 1:
                reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, cur_kv, dst),
                        dist.P2POp(dist.irecv, nxt_kv, src),
                    ]
                )

            src_rank = (rank - step) % w_sz
            c_j = [src_rank, 2 * w_sz - 1 - src_rank]

            kn_s = cur_kv[..., : H * 128].view(B, 2, Sc, H, 128).permute(1, 0, 3, 2, 4).contiguous()
            v_s = cur_kv[..., H * 128 : H * 256].view(B, 2, Sc, H, 128).permute(1, 0, 3, 2, 4).contiguous()
            kr_s = cur_kv[..., H * 256 :].view(B, 2, Sc, 64).permute(1, 0, 2, 3).unsqueeze(2).contiguous()

            for a in (0, 1):
                for b in (0, 1):
                    if c_r[a] < c_j[b]:
                        continue
                    is_diag = c_r[a] == c_j[b]
                    o_sub, lse_sub = _mla_split_fwd_raw(
                        qn_p[a], qr_p[a], kn_s[b], kr_s[b], v_s[b], sm_scale, causal=is_diag
                    )
                    _ring_lse2_merge_kernel[(triton.cdiv(n_rows_ch, 64),)](
                        o_acc[a],
                        lse_acc[a],
                        o_sub,
                        lse_sub,
                        n_rows_ch,
                        D=128,
                        INIT_STEP=(not inited[a]),
                        num_warps=4,
                    )
                    inited[a] = True

            if reqs:
                for rq in reqs:
                    rq.wait()
                buf_idx = 1 - buf_idx

        out_final = o_acc.permute(1, 2, 0, 3, 4).contiguous().to(torch.bfloat16).view(B, H, S_loc, 128)
        lse_final = lse_acc.permute(1, 2, 0, 3).contiguous().view(B, H, S_loc)
        ctx.save_for_backward(q_nope, q_rope, kv_pack_loc, out_final, lse_final)
        ctx.sm_scale = sm_scale
        return out_final

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        q_nope, q_rope, kv_pack_loc, out_final, lse_final = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        qk_scale = sm_scale * RCP_LN2
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, _ = q_nope.shape
        Sc = S_loc // 2
        dst = (rank + 1) % w_sz
        src = (rank - 1) % w_sz

        doc = do.contiguous()
        delta_full = torch.empty((B, H, S_loc), device=q_nope.device, dtype=torch.float32)
        _mla_bwd_prep_kernel[(triton.cdiv(B * H * S_loc, 64),)](
            out_final, doc, delta_full, B * H * S_loc, D_V=128, num_warps=4
        )

        qn_p = q_nope.view(B, H, 2, Sc, 128).permute(2, 0, 1, 3, 4).contiguous()
        qr_p = q_rope.view(B, H, 2, Sc, 64).permute(2, 0, 1, 3, 4).contiguous()
        do_p = doc.view(B, H, 2, Sc, 128).permute(2, 0, 1, 3, 4).contiguous()
        lse_p = lse_final.view(B, H, 2, Sc).permute(2, 0, 1, 3).contiguous()
        del_p = delta_full.view(B, H, 2, Sc).permute(2, 0, 1, 3).contiguous()

        dqn_p = torch.zeros_like(qn_p)
        dqr_p = torch.zeros_like(qr_p)

        kv_bufs = [kv_pack_loc.clone(), torch.empty_like(kv_pack_loc)]
        dkv_bufs = [torch.zeros_like(kv_pack_loc), torch.empty_like(kv_pack_loc)]
        buf_idx = 0
        c_r = [rank, 2 * w_sz - 1 - rank]

        for step in range(w_sz):
            cur_kv = kv_bufs[buf_idx]
            nxt_kv = kv_bufs[1 - buf_idx]
            cur_dkv = dkv_bufs[buf_idx]
            nxt_dkv = dkv_bufs[1 - buf_idx]

            src_rank = (rank - step) % w_sz
            c_j = [src_rank, 2 * w_sz - 1 - src_rank]

            kv_reqs = []
            if step < w_sz - 1:
                kv_reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, cur_kv, dst),
                        dist.P2POp(dist.irecv, nxt_kv, src),
                    ]
                )

            kn_s = cur_kv[..., : H * 128].view(B, 2, Sc, H, 128).permute(1, 0, 3, 2, 4).contiguous()
            v_s = cur_kv[..., H * 128 : H * 256].view(B, 2, Sc, H, 128).permute(1, 0, 3, 2, 4).contiguous()
            kr_s = cur_kv[..., H * 256 :].view(B, 2, Sc, 64).permute(1, 0, 2, 3).unsqueeze(2).contiguous()

            dkn_p = torch.zeros_like(kn_s)
            dv_p = torch.zeros_like(v_s)
            dkr_part = torch.zeros((2, B, H, Sc, 64), device=q_nope.device, dtype=torch.float32)
            dk_written = [False, False]

            for a in (0, 1):
                for b in (0, 1):
                    if c_r[a] < c_j[b]:
                        continue
                    is_diag = c_r[a] == c_j[b]
                    _mla_split_bwd_dkdv_kernel[(Sc // 64, B * H)](
                        qn_p[a],
                        qr_p[a],
                        kn_s[b],
                        kr_s[b],
                        v_s[b],
                        do_p[a],
                        lse_p[a],
                        del_p[a],
                        dkn_p[b],
                        dkr_part[b],
                        dv_p[b],
                        sm_scale,
                        qk_scale,
                        H,
                        Sc,
                        Sc,
                        BM=64,
                        BN=64,
                        CAUSAL=is_diag,
                        ACCUM_DKDV=dk_written[b],
                        num_warps=4,
                        num_stages=2,
                    )
                    dk_written[b] = True

                    _mla_split_bwd_dq_kernel[(Sc // 64, B * H)](
                        qn_p[a],
                        qr_p[a],
                        kn_s[b],
                        kr_s[b],
                        v_s[b],
                        do_p[a],
                        lse_p[a],
                        del_p[a],
                        dqn_p[a],
                        dqr_p[a],
                        sm_scale,
                        qk_scale,
                        H,
                        Sc,
                        Sc,
                        BM=64,
                        BN=64,
                        CAUSAL=is_diag,
                        ACCUM_DQ=True,
                        num_warps=4,
                        num_stages=2,
                    )

            dkv_loc = torch.cat(
                [
                    dkn_p.permute(1, 0, 3, 2, 4).contiguous().view(B, S_loc, H * 128),
                    dv_p.permute(1, 0, 3, 2, 4).contiguous().view(B, S_loc, H * 128),
                    dkr_part.sum(dim=2).permute(1, 0, 2, 3).contiguous().view(B, S_loc, 64).bfloat16(),
                ],
                dim=-1,
            )
            cur_dkv.copy_((cur_dkv.float() + dkv_loc.float()).bfloat16())

            if kv_reqs:
                for rq in kv_reqs:
                    rq.wait()

            dkv_reqs = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, cur_dkv, dst),
                    dist.P2POp(dist.irecv, nxt_dkv, src),
                ]
            )
            for rq in dkv_reqs:
                rq.wait()
            buf_idx = 1 - buf_idx

        final_dkv = dkv_bufs[buf_idx]
        dkn_out = final_dkv[..., : H * 128].view(B, S_loc, H, 128).transpose(1, 2).contiguous()
        dv_out = final_dkv[..., H * 128 : H * 256].view(B, S_loc, H, 128).transpose(1, 2).contiguous()
        dkr_out = final_dkv[..., H * 256 :].contiguous()
        return (
            dqn_p.permute(1, 2, 0, 3, 4).contiguous().view(B, H, S_loc, 128),
            dqr_p.permute(1, 2, 0, 3, 4).contiguous().view(B, H, S_loc, 64),
            dkn_out,
            dkr_out,
            dv_out,
            None,
        )


class _DiffAllGatherSeq(torch.autograd.Function):
    """AllGather `[B, S_loc, 64]` shared `K_rope` across `CP` -> `[B, 1, W*S_loc, 64]`,
    with `ReduceScatter` backward (`32x` smaller than head-replicated `AllToAll`!)."""

    @staticmethod
    def forward(ctx: Any, kr_bs: torch.Tensor) -> torch.Tensor:
        w_sz = dist.get_world_size()
        B, S_loc, D = kr_bs.shape
        kc = kr_bs.contiguous()
        gathered = torch.empty((w_sz, B, S_loc, D), device=kc.device, dtype=kc.dtype)
        ag_fn = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
        ag_fn(gathered, kc)
        return gathered.permute(1, 0, 2, 3).contiguous().view(B, 1, w_sz * S_loc, D)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        w_sz = dist.get_world_size()
        B = grad_out.shape[0]
        S_loc = grad_out.shape[2] // w_sz
        D = grad_out.shape[3]
        gin_buf = grad_out.squeeze(1).view(B, w_sz, S_loc, D).permute(1, 0, 2, 3).contiguous()
        g_loc = torch.empty((B, S_loc, D), device=grad_out.device, dtype=grad_out.dtype)
        rs_fn = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)
        rs_fn(g_loc, gin_buf, op=dist.ReduceOp.SUM)
        return g_loc


def cp_ulysses_triton_split_mla(
    qn_bhsd: torch.Tensor,
    qr_bhsd: torch.Tensor,
    kn_bhsd: torch.Tensor,
    kr_bs: torch.Tensor,
    v_bhsd: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Head-sharded Split-Dim Ulysses MLA: `AllToAll` on `(Q_nope[128], Q_rope[64],
    K_nope[128], V[128])` + `AllGather` on shared `1`-head `K_rope[64]` (`0` V-padding,
    `0` `32x` `K_rope` expansion) + Split-Dim Triton `Flash-MLA` (`128+64 -> 128`)!"""
    w_sz = dist.get_world_size()
    B, H, S_loc, _ = qn_bhsd.shape
    hl = H // w_sz

    def _s2h_bhsd(t: torch.Tensor, d_dim: int) -> torch.Tensor:
        # [B, w_sz, hl, S_loc, d_dim] -> [w_sz, B, hl, S_loc, d_dim] -> AllToAll -> [B, hl, w_sz*S_loc, d_dim]
        send = t.view(B, w_sz, hl, S_loc, d_dim).permute(1, 0, 2, 3, 4).contiguous()
        recv = _DiffAllToAll.apply(send)
        return recv.permute(1, 2, 0, 3, 4).contiguous().view(B, hl, w_sz * S_loc, d_dim)

    def _h2s_bhsd(t: torch.Tensor, d_dim: int) -> torch.Tensor:
        send = t.view(B, hl, w_sz, S_loc, d_dim).permute(2, 0, 1, 3, 4).contiguous()
        recv = _DiffAllToAll.apply(send)
        return recv.permute(1, 0, 2, 3, 4).contiguous().view(B, H, S_loc, d_dim)

    qnf = _s2h_bhsd(qn_bhsd, 128)
    qrf = _s2h_bhsd(qr_bhsd, 64)
    knf = _s2h_bhsd(kn_bhsd, 128)
    vf = _s2h_bhsd(v_bhsd, 128)
    krf = _DiffAllGatherSeq.apply(kr_bs)
    of = triton_split_mla(qnf, qrf, knf, krf, vf, sm_scale)
    return _h2s_bhsd(of, 128)


class _DiffAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        xc = x.contiguous()
        out = torch.empty_like(xc)
        dist.all_to_all_single(out, xc)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        gc_t = grad_out.contiguous()
        grad_in = torch.empty_like(gc_t)
        dist.all_to_all_single(grad_in, gc_t)
        return grad_in


def cp_ulysses_fa2(q_bshd: torch.Tensor, k_bshd: torch.Tensor, v_bshd: torch.Tensor, sm_scale: float) -> torch.Tensor:
    w_sz = dist.get_world_size()
    B, S_loc, Hq, D = q_bshd.shape
    Hkv = k_bshd.shape[2]

    def _s2h(t: torch.Tensor, h_tot: int, d_dim: int) -> torch.Tensor:
        hl = h_tot // w_sz
        send = t.view(B, S_loc, w_sz, hl, d_dim).permute(2, 0, 1, 3, 4).contiguous()
        recv = _DiffAllToAll.apply(send)
        return recv.permute(1, 0, 2, 3, 4).contiguous().view(B, w_sz * S_loc, hl, d_dim)

    def _h2s(t: torch.Tensor, h_tot: int, d_dim: int) -> torch.Tensor:
        hl = h_tot // w_sz
        send = t.view(B, w_sz, S_loc, hl, d_dim).permute(1, 0, 2, 3, 4).contiguous()
        recv = _DiffAllToAll.apply(send)
        return recv.permute(1, 2, 0, 3, 4).contiguous().view(B, S_loc, h_tot, d_dim)

    qf = _s2h(q_bshd, Hq, D)
    kf = _s2h(k_bshd, Hkv, D)
    vf = _s2h(v_bshd, Hkv, v_bshd.shape[-1])
    of = flash_attn_func(qf, kf, vf, causal=True, softmax_scale=sm_scale)
    return _h2s(of, Hq, of.shape[-1])


def run_suite_c_cp8_latent_ring(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    # 1. Exact 8-GPU vs 1-GPU End-to-End MLA Layer Parity at S_global = 4,096
    torch.manual_seed(12070)
    H_v, nh_v, S_gv = 2048, 16, 4096
    layer_1 = DeepSeekMLALayer(H_v, nh_v, d_c=512, d_q_c=1536).to(
        device=device, dtype=torch.bfloat16
    )
    gen_v = torch.Generator(device=device)
    gen_v.manual_seed(12075)
    x_full = torch.randn((1, S_gv, H_v), generator=gen_v, device=device, dtype=torch.bfloat16)
    dy_full = torch.randn((1, S_gv, H_v), generator=gen_v, device=device, dtype=torch.bfloat16)

    # 1-GPU exact reference
    x1 = x_full.clone().requires_grad_(True)
    for p in layer_1.parameters():
        p.grad = None
    cos_1, sin_1 = build_rope_cache(torch.arange(S_gv, device=device).unsqueeze(0))
    y1 = layer_1.forward_mode(x1, cos_1, sin_1, "mla_triton_split_fused")
    torch.autograd.backward(y1, dy_full)
    dx1_ref = x1.grad.clone()
    dw_dkv_ref = layer_1.w_dkv_kr.weight.grad.clone()
    dw_ukv_ref = layer_1.w_uk_uv.weight.grad.clone()
    dw_o_ref = layer_1.w_o.weight.grad.clone()

    x_zz_ref = _zigzag_pack_from_full(x_full, rank, world_size, seq_dim=1)
    dy_zz = _zigzag_pack_from_full(dy_full, rank, world_size, seq_dim=1)
    y1_zz_ref = _zigzag_pack_from_full(y1.detach(), rank, world_size, seq_dim=1)
    dx1_zz_ref = _zigzag_pack_from_full(dx1_ref, rank, world_size, seq_dim=1)
    pos_zz = _zigzag_local_positions(S_gv, rank, world_size, device)
    cos_zz, sin_zz = build_rope_cache(pos_zz)

    def _max_rank_err(loc_t: torch.Tensor, ref_t: torch.Tensor) -> float:
        v_e = rel_l2(loc_t, ref_t)
        tv = torch.tensor([v_e], device=device, dtype=torch.float64)
        dist.all_reduce(tv, op=dist.ReduceOp.MAX)
        return round(float(tv.item()), 6)

    # Verify (a) CP=8 Compressed-Latent Zig-Zag Ring
    for p in layer_1.parameters():
        p.grad = None
    x_lat = x_zz_ref.clone().requires_grad_(True)
    h_l = layer_1.in_norm(x_lat)
    cq_l = layer_1.q_norm(layer_1.w_dq(h_l))
    qa_l = layer_1.w_uq_qr(cq_l).view(1, x_lat.shape[1], nh_v, 192).transpose(1, 2)
    qn_l = qa_l[..., :128].contiguous()
    qr_l = apply_rope_bhsd(qa_l[..., 128:], cos_zz, sin_zz).contiguous()
    ckv_l, kr_l = fused_latent_rmsnorm_rope(
        layer_1.w_dkv_kr(h_l), layer_1.kv_norm.weight, cos_zz, sin_zz, 1e-5
    )
    o_lat_cp = _CP8CompressedLatentZigZagFn.apply(
        qn_l, qr_l, ckv_l, kr_l, layer_1.w_uk_uv.weight, layer_1.sm_scale
    )
    y_lat_cp = x_lat + layer_1.w_o(
        o_lat_cp.transpose(1, 2).contiguous().view(1, x_lat.shape[1], nh_v * 128)
    )
    torch.autograd.backward(y_lat_cp, dy_zz)
    for name_p, p in layer_1.named_parameters():
        if p.grad is not None and name_p != "w_uk_uv.weight":
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    # note: in RMSNorm, w_dkv_kr / w_o are linear projections summed over tokens, so sum across ranks == full!
    lat_ring_parity = {
        "max_out_rel_l2": _max_rank_err(y_lat_cp, y1_zz_ref),
        "max_dx_rel_l2": _max_rank_err(x_lat.grad, dx1_zz_ref),
        "max_dw_dkv_rel_l2": _max_rank_err(layer_1.w_dkv_kr.weight.grad, dw_dkv_ref),
        "max_dw_ukv_rel_l2": _max_rank_err(layer_1.w_uk_uv.weight.grad, dw_ukv_ref),
        "max_dw_o_rel_l2": _max_rank_err(layer_1.w_o.weight.grad, dw_o_ref),
    }

    # Verify (b) CP=8 Decompressed Zig-Zag Ring
    for p in layer_1.parameters():
        p.grad = None
    x_dec = x_zz_ref.clone().requires_grad_(True)
    h_d = layer_1.in_norm(x_dec)
    cq_d = layer_1.q_norm(layer_1.w_dq(h_d))
    qa_d = layer_1.w_uq_qr(cq_d).view(1, x_dec.shape[1], nh_v, 192).transpose(1, 2)
    qn_d = qa_d[..., :128].contiguous()
    qr_d = apply_rope_bhsd(qa_d[..., 128:], cos_zz, sin_zz).contiguous()
    ckv_d, kr_d = fused_latent_rmsnorm_rope(
        layer_1.w_dkv_kr(h_d), layer_1.kv_norm.weight, cos_zz, sin_zz, 1e-5
    )
    kv_up_d = layer_1.w_uk_uv(ckv_d).view(1, x_dec.shape[1], nh_v, 256).transpose(1, 2)
    o_dec_cp = _CP8DecompressedMLAZigZagFn.apply(
        qn_d, qr_d, kv_up_d[..., :128].contiguous(), kr_d, kv_up_d[..., 128:].contiguous(), layer_1.sm_scale
    )
    y_dec_cp = x_dec + layer_1.w_o(
        o_dec_cp.transpose(1, 2).contiguous().view(1, x_dec.shape[1], nh_v * 128)
    )
    torch.autograd.backward(y_dec_cp, dy_zz)
    for p in layer_1.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    dec_ring_parity = {
        "max_out_rel_l2": _max_rank_err(y_dec_cp, y1_zz_ref),
        "max_dx_rel_l2": _max_rank_err(x_dec.grad, dx1_zz_ref),
        "max_dw_ukv_rel_l2": _max_rank_err(layer_1.w_uk_uv.weight.grad, dw_ukv_ref),
    }

    # Verify (c) CP=8 Split-Dim Ulysses + Shared-K_rope AllGather (`cp_ulysses_triton_split_mla`)
    s_loc_v = S_gv // world_size
    sl_u = slice(rank * s_loc_v, (rank + 1) * s_loc_v)
    for p in layer_1.parameters():
        p.grad = None
    x_ul = x_full[:, sl_u].contiguous().detach().requires_grad_(True)
    dy_ul = dy_full[:, sl_u].contiguous()
    cos_ul, sin_ul = build_rope_cache((torch.arange(s_loc_v, device=device) + rank * s_loc_v).unsqueeze(0))
    h_u = layer_1.in_norm(x_ul)
    cq_u = layer_1.q_norm(layer_1.w_dq(h_u))
    qa_u = layer_1.w_uq_qr(cq_u).view(1, s_loc_v, nh_v, 192).transpose(1, 2)
    qn_u = qa_u[..., :128].contiguous()
    qr_u = apply_rope_bhsd(qa_u[..., 128:], cos_ul, sin_ul).contiguous()
    ckv_u, kr_u = fused_latent_rmsnorm_rope(
        layer_1.w_dkv_kr(h_u), layer_1.kv_norm.weight, cos_ul, sin_ul, 1e-5
    )
    kv_up_u = layer_1.w_uk_uv(ckv_u).view(1, s_loc_v, nh_v, 256).transpose(1, 2)
    o_ul_cp = cp_ulysses_triton_split_mla(
        qn_u, qr_u, kv_up_u[..., :128].contiguous(), kr_u, kv_up_u[..., 128:].contiguous(), layer_1.sm_scale
    )
    y_ul_cp = x_ul + layer_1.w_o(o_ul_cp.transpose(1, 2).contiguous().view(1, s_loc_v, nh_v * 128))
    torch.autograd.backward(y_ul_cp, dy_ul)
    for p in layer_1.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    uly_split_parity = {
        "max_out_rel_l2": _max_rank_err(y_ul_cp, y1.detach()[:, sl_u]),
        "max_dx_rel_l2": _max_rank_err(x_ul.grad, dx1_ref[:, sl_u]),
        "max_dw_ukv_rel_l2": _max_rank_err(layer_1.w_uk_uv.weight.grad, dw_ukv_ref),
    }

    del layer_1, x_full, dy_full, x1, y1
    gc.collect()
    torch.cuda.empty_cache()

    # 2. CP=8 Long-Context Scaling (S_global = 16,384 .. 131,072, H_model = 4,096, n_h = 32)
    cp_seqs = [16384, 65536] if is_smoke else [16384, 32768, 65536, 131072]
    warmup_c = 2 if is_smoke else 3
    iters_c = 4 if is_smoke else 8
    H_c, nh_c = 4096, 32
    mla_c = DeepSeekMLALayer(H_c, nh_c, d_c=512, d_q_c=1536).to(
        device=device, dtype=torch.bfloat16
    )

    def _time_dist_fn(fn_c: Callable[[], None], it_ov: int | None = None) -> dict[str, float]:
        it_run = it_ov if it_ov is not None else iters_c
        for _ in range(warmup_c):
            fn_c()
        torch.cuda.synchronize(device)
        dist.barrier()
        base_m = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        fn_c()
        torch.cuda.synchronize(device)
        inc_m = max(0.0, (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0**2))
        m_t = torch.tensor([inc_m], device=device, dtype=torch.float64)
        dist.all_reduce(m_t, op=dist.ReduceOp.MAX)

        samples_ms: list[float] = []
        s_e = torch.cuda.Event(enable_timing=True)
        e_e = torch.cuda.Event(enable_timing=True)
        for _ in range(it_run):
            dist.barrier()
            s_e.record()
            fn_c()
            e_e.record()
            torch.cuda.synchronize(device)
            dur = torch.tensor([float(s_e.elapsed_time(e_e))], device=device, dtype=torch.float64)
            dist.all_reduce(dur, op=dist.ReduceOp.MAX)
            samples_ms.append(float(dur.item()))
        ss = sorted(samples_ms)
        q25 = ss[max(0, (len(ss) - 1) // 4)]
        q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
        return {
            "step_ms_median": round(statistics.median(ss), 4),
            "step_ms_iqr": round(q75 - q25, 4),
            "per_rank_peak_inc_mib": round(float(m_t.item()), 2),
        }

    cp_scaling: dict[str, Any] = {}
    for S_glob in cp_seqs:
        S_loc = S_glob // world_size
        it_use = 5 if S_glob >= 131072 and not is_smoke else None
        pos_c = _zigzag_local_positions(S_glob, rank, world_size, device)
        cos_c, sin_c = build_rope_cache(pos_c)

        xl = torch.randn((1, S_loc, H_c), device=device, dtype=torch.bfloat16)
        dyl = torch.randn_like(xl)

        # Ring P2P communication bytes/rank across Fwd (7 sends) + Bwd (7 lat sends + 8 dlat sends = 22 sends)
        lat_comm_mib = round((22 * S_loc * 576 * 2) / (1024.0**2), 2)
        dec_comm_mib = round((22 * S_loc * (nh_c * 256 + 64) * 2) / (1024.0**2), 2)
        # DeepSpeed-Ulysses MLA (32 heads of 192 Q + 32 heads of 192 K + 32 heads of 192 V-pad + 32 heads of 192 O) * 2 (fwd+bwd) * (7/8)
        uly_mla_comm_mib = round(
            (2 * S_loc * (4 * nh_c * 192) * 2 * ((world_size - 1) / world_size)) / (1024.0**2),
            2,
        )

        modes_sc: dict[str, Any] = {}

        # Mode 1: CP=8 Compressed-Latent Zig-Zag Ring MLA (Ours!)
        x_m1 = xl.clone().requires_grad_(True)

        def _f_lat_ring() -> None:
            for p in mla_c.parameters():
                p.grad = None
            x_m1.grad = None
            h = mla_c.in_norm(x_m1)
            cq = mla_c.q_norm(mla_c.w_dq(h))
            qa = mla_c.w_uq_qr(cq).view(1, S_loc, nh_c, 192).transpose(1, 2)
            qn = qa[..., :128].contiguous()
            qr = apply_rope_bhsd(qa[..., 128:], cos_c, sin_c).contiguous()
            ckv, kr = fused_latent_rmsnorm_rope(
                mla_c.w_dkv_kr(h), mla_c.kv_norm.weight, cos_c, sin_c, 1e-5
            )
            o_cp = _CP8CompressedLatentZigZagFn.apply(
                qn, qr, ckv, kr, mla_c.w_uk_uv.weight, mla_c.sm_scale
            )
            y = x_m1 + mla_c.w_o(o_cp.transpose(1, 2).contiguous().view(1, S_loc, nh_c * 128))
            torch.autograd.backward(y, dyl)

        t1 = _time_dist_fn(_f_lat_ring, it_ov=it_use)
        modes_sc["cp8_mla_compressed_latent_zigzag_ring"] = {
            **t1,
            "cluster_ktok_per_s": round((S_glob / (t1["step_ms_median"] * 1e-3)) / 1e3, 2),
            "comm_mib_per_rank": lat_comm_mib,
        }
        del x_m1

        # Mode 2: CP=8 Decompressed-KV Zig-Zag Ring MLA (Naive 8,256 BF16/tok ring!)
        x_m2 = xl.clone().requires_grad_(True)

        def _f_dec_ring() -> None:
            for p in mla_c.parameters():
                p.grad = None
            x_m2.grad = None
            h = mla_c.in_norm(x_m2)
            cq = mla_c.q_norm(mla_c.w_dq(h))
            qa = mla_c.w_uq_qr(cq).view(1, S_loc, nh_c, 192).transpose(1, 2)
            qn = qa[..., :128].contiguous()
            qr = apply_rope_bhsd(qa[..., 128:], cos_c, sin_c).contiguous()
            ckv, kr = fused_latent_rmsnorm_rope(
                mla_c.w_dkv_kr(h), mla_c.kv_norm.weight, cos_c, sin_c, 1e-5
            )
            kv_up = mla_c.w_uk_uv(ckv).view(1, S_loc, nh_c, 256).transpose(1, 2)
            o_cp = _CP8DecompressedMLAZigZagFn.apply(
                qn, qr, kv_up[..., :128].contiguous(), kr, kv_up[..., 128:].contiguous(), mla_c.sm_scale
            )
            y = x_m2 + mla_c.w_o(o_cp.transpose(1, 2).contiguous().view(1, S_loc, nh_c * 128))
            torch.autograd.backward(y, dyl)

        t2 = _time_dist_fn(_f_dec_ring, it_ov=it_use)
        modes_sc["cp8_mla_decompressed_zigzag_ring"] = {
            **t2,
            "cluster_ktok_per_s": round((S_glob / (t2["step_ms_median"] * 1e-3)) / 1e3, 2),
            "comm_mib_per_rank": dec_comm_mib,
        }
        del x_m2

        # Mode 3: CP=8 DeepSpeed-Ulysses MLA (AllToAll on 32-head decompressed 192->192 V-padded FA2)
        pos_u = (torch.arange(S_loc, device=device) + rank * S_loc).unsqueeze(0)
        cos_u, sin_u = build_rope_cache(pos_u)
        x_m3 = xl.clone().requires_grad_(True)

        def _f_uly_mla() -> None:
            for p in mla_c.parameters():
                p.grad = None
            x_m3.grad = None
            qn, qr, kn, kr, v = mla_c.forward_eager_projections(x_m3, cos_u, sin_u)
            qc = torch.cat([qn, qr], dim=-1).transpose(1, 2).contiguous()
            kc = torch.cat([kn, kr.expand(-1, nh_c, -1, -1)], dim=-1).transpose(1, 2).contiguous()
            vp = F.pad(v.transpose(1, 2).contiguous(), (0, 64))
            op = cp_ulysses_fa2(qc, kc, vp, mla_c.sm_scale)
            y = x_m3 + mla_c.w_o(op[..., :128].contiguous().view(1, S_loc, nh_c * 128))
            torch.autograd.backward(y, dyl)

        t3 = _time_dist_fn(_f_uly_mla, it_ov=it_use)
        modes_sc["cp8_mla_ulysses_fa2_vpadded"] = {
            **t3,
            "cluster_ktok_per_s": round((S_glob / (t3["step_ms_median"] * 1e-3)) / 1e3, 2),
            "comm_mib_per_rank": uly_mla_comm_mib,
        }
        del x_m3

        # Mode 4: CP=8 Split-Dim Ulysses + Shared-K_rope AllGather (Ours: 0 V-pad, 0 K_rope 32x expand!)
        uly_split_comm_mib = round(
            (
                (2 * S_loc * (nh_c * (192 + 128 + 128 + 128)) * 2 * ((world_size - 1) / world_size))
                + (2 * S_loc * 64 * 2 * (world_size - 1))
            )
            / (1024.0**2),
            2,
        )
        x_m4 = xl.clone().requires_grad_(True)

        def _f_uly_split() -> None:
            for p in mla_c.parameters():
                p.grad = None
            x_m4.grad = None
            h = mla_c.in_norm(x_m4)
            cq = mla_c.q_norm(mla_c.w_dq(h))
            qa = mla_c.w_uq_qr(cq).view(1, S_loc, nh_c, 192).transpose(1, 2)
            qn = qa[..., :128].contiguous()
            qr = apply_rope_bhsd(qa[..., 128:], cos_u, sin_u).contiguous()
            ckv, kr = fused_latent_rmsnorm_rope(
                mla_c.w_dkv_kr(h), mla_c.kv_norm.weight, cos_u, sin_u, 1e-5
            )
            kv_up = mla_c.w_uk_uv(ckv).view(1, S_loc, nh_c, 256).transpose(1, 2)
            o_cp = cp_ulysses_triton_split_mla(
                qn, qr, kv_up[..., :128].contiguous(), kr, kv_up[..., 128:].contiguous(), mla_c.sm_scale
            )
            y = x_m4 + mla_c.w_o(o_cp.transpose(1, 2).contiguous().view(1, S_loc, nh_c * 128))
            torch.autograd.backward(y, dyl)

        t4 = _time_dist_fn(_f_uly_split, it_ov=it_use)
        modes_sc["cp8_mla_ulysses_triton_split"] = {
            **t4,
            "cluster_ktok_per_s": round((S_glob / (t4["step_ms_median"] * 1e-3)) / 1e3, 2),
            "comm_mib_per_rank": uly_split_comm_mib,
        }
        del x_m4, xl, dyl
        gc.collect()
        torch.cuda.empty_cache()

        cp_scaling[str(S_glob)] = {
            "global_seq_len": S_glob,
            "local_seq_len": S_loc,
            "heads": nh_c,
            "hidden": H_c,
            "modes": modes_sc,
        }

    del mla_c
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "cp8_vs_1gpu_parity": {
            "compressed_latent_ring": lat_ring_parity,
            "decompressed_kv_ring": dec_ring_parity,
            "ulysses_triton_split": uly_split_parity,
        },
        "cp8_long_context_scaling": cp_scaling,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E120: Multi-Head Latent Attention (MLA) Training Kernels, Weight-Absorption Crossover, & CP=8 Latent-Ring"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(minutes=30), device_id=device
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"Expected 8 GPUs, got {world_size}"

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    is_smoke = args.mode == "smoke"

    t_start = time.perf_counter()
    s0_res: dict[str, Any] = {}
    sa_res: dict[str, Any] = {}
    sb_res: dict[str, Any] = {}

    if rank == 0:
        s0_res = run_suite_0_audits(device, is_smoke)
        print(f"[E120] Suite 0 complete at +{time.perf_counter() - t_start:.1f}s", flush=True)
        sa_res = run_suite_a_mla_layer_scaling(device, is_smoke)
        print(f"[E120] Suite A complete at +{time.perf_counter() - t_start:.1f}s", flush=True)
        sb_res = run_suite_b_crossover_and_sac(device, is_smoke)
        print(f"[E120] Suite B complete at +{time.perf_counter() - t_start:.1f}s", flush=True)

    dist.barrier()
    sc_res = run_suite_c_cp8_latent_ring(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = round(time.perf_counter() - t_start, 2)

    if rank == 0:
        print(f"[E120] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        env = environment_metadata()
        env["triton"] = getattr(triton, "__version__", "unknown")
        import flash_attn as fa_pkg

        env["flash_attn"] = getattr(fa_pkg, "__version__", "unknown")
        env["flashinfer"] = getattr(flashinfer, "__version__", "unknown")
        payload = {
            "experiment": "E120_mla_latent_attention_training",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": elapsed_s,
            "environment": env,
            "suite_0_audits": s0_res,
            "suite_a_mla_layer_scaling": sa_res,
            "suite_b_crossover_and_latent_sac": sb_res,
            "suite_c_cp8_latent_ring": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
