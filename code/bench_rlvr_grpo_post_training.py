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
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import triton
import triton.language as tl
from flash_attn import flash_attn_func, flash_attn_varlen_func

try:
    import pynvml

    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False

from common import environment_metadata

RCP_LN2 = 1.4426950408889634


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float()
    bf = b.detach().float()
    return float((af - bf).norm().item() / max(bf.norm().item(), 1e-12))


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


def time_fn_with_peak_mem(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    reset_grads_fn: Callable[[], None] | None = None,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if reset_grads_fn is not None:
        reset_grads_fn()
    torch.cuda.synchronize(device)
    base_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    peak_inc_mib = max(
        0.0, (torch.cuda.max_memory_allocated(device) - base_mem) / (1024.0**2)
    )
    after_step_mib = max(
        0.0, (torch.cuda.memory_allocated(device) - base_mem) / (1024.0**2)
    )
    transient_ws_mib = max(0.0, peak_inc_mib - after_step_mib)

    samples: list[float] = []
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        s_ev.record()
        fn()
        e_ev.record()
        torch.cuda.synchronize(device)
        samples.append(float(s_ev.elapsed_time(e_ev)))
    tel = sample_gpu_clocks(device.index or 0)
    ss = sorted(samples)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    return {
        "median_ms": round(float(statistics.median(ss)), 4),
        "iqr_ms": round(float(q75 - q25), 4),
        "peak_inc_mib": round(float(peak_inc_mib), 2),
        "transient_workspace_mib": round(float(transient_ws_mib), 2),
        "sm_clock_after_mhz": round(tel["sm_clock_mhz"], 1),
        "power_after_w": round(tel["power_w"], 2),
    }


# ============================================================================
# 1. Online (m, d, e) 3-Statistic Monoid Triton Kernels: GRPO/DAPO & DPO
# ============================================================================


@triton.jit
def _naive_uncentered_entropy_kernel(
    Z_ptr, stride_z_row: tl.int64, OutEnt_ptr, V: tl.int32, BLOCK_V: tl.constexpr
):
    """Negative-control kernel: computes H(p) = (m + ln d) - (sum z_v exp(z_v - m)) / d
    using uncentered z_v, exposing catastrophic FP32 mantissa cancellation at nonzero logit shifts!"""
    row = tl.program_id(0)
    row_ptr = Z_ptr + row.to(tl.int64) * stride_z_row
    m_acc = -1.0e30
    d_acc = 0.0
    s_raw_acc = 0.0
    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < V
        z = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m_tile = tl.max(tl.where(mask, z, -1.0e30), axis=0)
        exp_sh = tl.where(mask, tl.exp(z - m_tile), 0.0)
        d_tile = tl.sum(exp_sh, axis=0)
        s_tile = tl.sum(z * exp_sh, axis=0)

        m_new = tl.maximum(m_acc, m_tile)
        a1 = tl.exp(m_acc - m_new)
        a2 = tl.exp(m_tile - m_new)
        d_acc = a1 * d_acc + a2 * d_tile
        s_raw_acc = a1 * s_raw_acc + a2 * s_tile
        m_acc = m_new

    lse = m_acc + tl.log(d_acc)
    ent_naive = lse - (s_raw_acc / d_acc)
    tl.store(OutEnt_ptr + row, ent_naive)


@triton.jit
def _centered_monoid_entropy_kernel(
    Z_ptr, stride_z_row: tl.int64, OutEnt_ptr, V: tl.int32, BLOCK_V: tl.constexpr
):
    """IEEE-754-safe shift-invariant centered online (m, d, e) monoid kernel:
    e_new = a1 * (e_acc + d_acc * (m_acc - m_new)) + a2 * (e_tile + d_tile * (m_tile - m_new))
    H(p) = ln(d_acc) - e_acc / d_acc >= 0 identically for any baseline shift mu!"""
    row = tl.program_id(0)
    row_ptr = Z_ptr + row.to(tl.int64) * stride_z_row
    m_acc = -1.0e30
    d_acc = 0.0
    e_acc = 0.0
    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < V
        z = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m_tile = tl.max(tl.where(mask, z, -1.0e30), axis=0)
        z_sh = tl.where(mask, z - m_tile, 0.0)
        exp_sh = tl.where(mask, tl.exp(z_sh), 0.0)
        d_tile = tl.sum(exp_sh, axis=0)
        e_tile = tl.sum(z_sh * exp_sh, axis=0)

        m_new = tl.maximum(m_acc, m_tile)
        a1 = tl.exp(m_acc - m_new)
        a2 = tl.exp(m_tile - m_new)
        e_acc = a1 * (e_acc + d_acc * (m_acc - m_new)) + a2 * (
            e_tile + d_tile * (m_tile - m_new)
        )
        d_acc = a1 * d_acc + a2 * d_tile
        m_acc = m_new

    ent = tl.log(d_acc) - (e_acc / d_acc)
    tl.store(OutEnt_ptr + row, ent)


@triton.jit
def _grpo_dapo_fwd_bwd_inplace_kernel(
    U_ptr,
    stride_u_row: tl.int64,
    Act_ptr,
    LpOld_ptr,
    LpRef_ptr,
    Adv_ptr,
    Weight_ptr,
    LossOut_ptr,
    inv_temp,
    eps_low,
    eps_high,
    beta_kl,
    alpha_ent,
    V: tl.int32,
    BLOCK_V: tl.constexpr,
):
    """Single-launch fused GRPO/DAPO (Clip-Higher + Schulman k3 KL + Shannon Entropy + 1/tau)
    forward + backward in-place logit-gradient kernel over [C, V]."""
    row = tl.program_id(0)
    w_t = tl.load(Weight_ptr + row).to(tl.float32)
    row_ptr = U_ptr + row.to(tl.int64) * stride_u_row

    if w_t == 0.0:
        for v0 in range(0, V, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            tl.store(
                row_ptr + offs, tl.zeros([BLOCK_V], dtype=tl.bfloat16), mask=offs < V
            )
        for c_i in tl.static_range(0, 4):
            tl.store(LossOut_ptr + row * 4 + c_i, 0.0)
        return

    act = tl.load(Act_ptr + row)
    lp_old = tl.load(LpOld_ptr + row).to(tl.float32)
    lp_ref = tl.load(LpRef_ptr + row).to(tl.float32)
    adv = tl.load(Adv_ptr + row).to(tl.float32)

    m_acc = -1.0e30
    d_acc = 0.0
    e_acc = 0.0
    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < V
        z_raw = (
            tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_temp
        )
        m_tile = tl.max(tl.where(mask, z_raw, -1.0e30), axis=0)
        z_sh = tl.where(mask, z_raw - m_tile, 0.0)
        exp_sh = tl.where(mask, tl.exp(z_sh), 0.0)
        d_tile = tl.sum(exp_sh, axis=0)
        e_tile = tl.sum(z_sh * exp_sh, axis=0)

        m_new = tl.maximum(m_acc, m_tile)
        a1 = tl.exp(m_acc - m_new)
        a2 = tl.exp(m_tile - m_new)
        e_acc = a1 * (e_acc + d_acc * (m_acc - m_new)) + a2 * (
            e_tile + d_tile * (m_tile - m_new)
        )
        d_acc = a1 * d_acc + a2 * d_tile
        m_acc = m_new

    lse = m_acc + tl.log(d_acc)
    ent = tl.log(d_acc) - (e_acc / d_acc)

    z_act = tl.load(row_ptr + act).to(tl.float32) * inv_temp
    lp_theta = z_act - lse
    ratio = tl.exp(lp_theta - lp_old)
    ratio_clip = tl.minimum(tl.maximum(ratio, 1.0 - eps_low), 1.0 + eps_high)
    surr1 = ratio * adv
    surr2 = ratio_clip * adv
    is_unclipped = surr1 <= surr2
    pg_t = -tl.minimum(surr1, surr2)

    diff_ref = lp_ref - lp_theta
    exp_diff_ref = tl.exp(diff_ref)
    kl_t = exp_diff_ref - diff_ref - 1.0

    tot_t = w_t * (pg_t + beta_kl * kl_t - alpha_ent * ent)
    tl.store(LossOut_ptr + row * 4 + 0, tot_t)
    tl.store(LossOut_ptr + row * 4 + 1, w_t * pg_t)
    tl.store(LossOut_ptr + row * 4 + 2, w_t * kl_t)
    tl.store(LossOut_ptr + row * 4 + 3, w_t * ent)

    pg_grad_lp = tl.where(is_unclipped, -ratio * adv, 0.0)
    kl_grad_lp = beta_kl * (1.0 - exp_diff_ref)
    g_lp = w_t * (pg_grad_lp + kl_grad_lp)
    ent_coeff = w_t * alpha_ent

    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < V
        z_raw = (
            tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_temp
        )
        p = tl.where(mask, tl.exp(z_raw - lse), 0.0)
        ind = tl.where(offs == act, 1.0, 0.0)
        dz = g_lp * (ind - p) + ent_coeff * p * (z_raw - lse + ent)
        du = (dz * inv_temp).to(tl.bfloat16)
        tl.store(row_ptr + offs, du, mask=mask)


@triton.jit
def _dpo_fwd_lse_gather_kernel(
    U_ptr,
    stride_u_row: tl.int64,
    Act_ptr,
    LseOut_ptr,
    LpOut_ptr,
    inv_temp,
    V: tl.int32,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = U_ptr + row.to(tl.int64) * stride_u_row
    act = tl.load(Act_ptr + row)

    m_acc = -1.0e30
    d_acc = 0.0
    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < V
        z = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_temp
        m_tile = tl.max(tl.where(mask, z, -1.0e30), axis=0)
        d_tile = tl.sum(tl.where(mask, tl.exp(z - m_tile), 0.0), axis=0)
        m_new = tl.maximum(m_acc, m_tile)
        d_acc = d_acc * tl.exp(m_acc - m_new) + d_tile * tl.exp(m_tile - m_new)
        m_acc = m_new

    lse = m_acc + tl.log(d_acc)
    z_act = tl.load(row_ptr + act).to(tl.float32) * inv_temp
    tl.store(LseOut_ptr + row, lse)
    tl.store(LpOut_ptr + row, z_act - lse)


@triton.jit
def _dpo_bwd_saved_lse_inplace_kernel(
    U_ptr,
    stride_u_row: tl.int64,
    Act_ptr,
    SavedLse_ptr,
    GradTok_ptr,
    inv_temp,
    V: tl.int32,
    BLOCK_V: tl.constexpr,
):
    """Pass-2 DPO backward kernel: uses saved 4-byte `lse[row]` to stream [C, V] ONCE
    with ZERO softmax reduction in backward!"""
    row = tl.program_id(0)
    row_ptr = U_ptr + row.to(tl.int64) * stride_u_row
    g_tok = tl.load(GradTok_ptr + row).to(tl.float32)
    if g_tok == 0.0:
        for v0 in range(0, V, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            tl.store(
                row_ptr + offs, tl.zeros([BLOCK_V], dtype=tl.bfloat16), mask=offs < V
            )
        return

    act = tl.load(Act_ptr + row)
    lse = tl.load(SavedLse_ptr + row).to(tl.float32)
    scale = g_tok * inv_temp

    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < V
        z = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_temp
        p = tl.where(mask, tl.exp(z - lse), 0.0)
        ind = tl.where(offs == act, 1.0, 0.0)
        du = (scale * (ind - p)).to(tl.bfloat16)
        tl.store(row_ptr + offs, du, mask=mask)


class _FusedLinearGRPOFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,          # [N, H] BF16
        weight: torch.Tensor,     # [V, H] BF16
        actions: torch.Tensor,    # [N] int64
        lp_old: torch.Tensor,     # [N] fp32
        lp_ref: torch.Tensor,     # [N] fp32
        advantages: torch.Tensor, # [N] fp32
        tok_weights: torch.Tensor,# [N] fp32
        tau: float,
        eps_low: float,
        eps_high: float,
        beta_kl: float,
        alpha_ent: float,
        chunk_size: int,
        fp32_dw_accum: bool,
    ) -> torch.Tensor:
        x_c2d = x.contiguous()
        N, H = x_c2d.shape
        V = weight.shape[0]
        inv_temp = 1.0 / tau
        block_v = 8192 if V >= 65536 else 4096
        num_warps = 16 if block_v == 8192 else 8

        dx = torch.empty_like(x_c2d)
        dw_acc = torch.zeros(
            (V, H),
            device=weight.device,
            dtype=torch.float32 if fp32_dw_accum else weight.dtype,
        )
        loss_terms = torch.empty((N, 4), device=x.device, dtype=torch.float32)
        tile_rows = min(chunk_size, N)
        logits_buf = torch.empty((tile_rows, V), device=x.device, dtype=x.dtype)
        w_t = weight.t()

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            rows = end - start
            xc = x_c2d[start:end]
            uc = logits_buf[:rows]
            torch.mm(xc, w_t, out=uc)
            _grpo_dapo_fwd_bwd_inplace_kernel[(rows,)](
                uc,
                uc.stride(0),
                actions[start:end],
                lp_old[start:end],
                lp_ref[start:end],
                advantages[start:end],
                tok_weights[start:end],
                loss_terms[start:end],
                inv_temp,
                eps_low,
                eps_high,
                beta_kl,
                alpha_ent,
                V,
                BLOCK_V=block_v,
                num_warps=num_warps,
            )
            torch.mm(uc, weight, out=dx[start:end])
            if fp32_dw_accum:
                torch.addmm(dw_acc, uc.t(), xc, out_dtype=torch.float32, out=dw_acc)
            else:
                dw_acc.addmm_(uc.t(), xc)

        del logits_buf
        dw_out = dw_acc.to(weight.dtype) if fp32_dw_accum else dw_acc
        del dw_acc
        ctx.save_for_backward(dx, dw_out)
        return loss_terms[:, 0].sum()

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        dx, dw = ctx.saved_tensors
        if float(grad_out.item()) != 1.0:
            dx = dx * grad_out.to(dx.dtype)
            dw = dw * grad_out.to(dw.dtype)
        return (dx, dw) + (None,) * 12


class _FusedSavedLSELinearDPOFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,          # [N, H] BF16
        weight: torch.Tensor,     # [V, H] BF16
        actions: torch.Tensor,    # [N] int64
        pair_id: torch.Tensor,    # [N] int64 in 0..B_pair-1
        sign: torch.Tensor,       # [N] fp32 (+1 chosen, -1 rejected, 0 masked)
        ref_margin: torch.Tensor, # [B_pair] fp32
        beta_dpo: float,
        tau: float,
        chunk_size: int,
    ) -> torch.Tensor:
        x_c2d = x.contiguous()
        N, H = x_c2d.shape
        V = weight.shape[0]
        B_pair = ref_margin.shape[0]
        inv_temp = 1.0 / tau
        block_v = 8192 if V >= 65536 else 4096
        num_warps = 16 if block_v == 8192 else 8

        saved_lse = torch.empty((N,), device=x.device, dtype=torch.float32)
        lp_theta = torch.empty((N,), device=x.device, dtype=torch.float32)
        tile_rows = min(chunk_size, N)
        logits_buf = torch.empty((tile_rows, V), device=x.device, dtype=x.dtype)
        w_t = weight.t()

        # Pass 1: Chunked forward LSE + action log-prob extraction (0 [N, V] retention!)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            rows = end - start
            uc = logits_buf[:rows]
            torch.mm(x_c2d[start:end], w_t, out=uc)
            _dpo_fwd_lse_gather_kernel[(rows,)](
                uc,
                uc.stride(0),
                actions[start:end],
                saved_lse[start:end],
                lp_theta[start:end],
                inv_temp,
                V,
                BLOCK_V=block_v,
                num_warps=num_warps,
            )

        pol_margin = torch.zeros((B_pair,), device=x.device, dtype=torch.float32)
        pol_margin.scatter_add_(0, pair_id, sign * lp_theta)
        dpo_logits = beta_dpo * (pol_margin - ref_margin)
        loss = -F.logsigmoid(dpo_logits).mean()

        # Exact sequence-coupled scalar upstream gradient -> per-token scalar
        g_pair = (-beta_dpo / float(B_pair)) * torch.sigmoid(-dpo_logits)
        g_tok = sign * g_pair.index_select(0, pair_id)

        # Pass 2: Saved-LSE streaming backward (no softmax reduction!)
        dx = torch.empty_like(x_c2d)
        dw_out = torch.zeros((V, H), device=weight.device, dtype=weight.dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            rows = end - start
            xc = x_c2d[start:end]
            uc = logits_buf[:rows]
            torch.mm(xc, w_t, out=uc)
            _dpo_bwd_saved_lse_inplace_kernel[(rows,)](
                uc,
                uc.stride(0),
                actions[start:end],
                saved_lse[start:end],
                g_tok[start:end],
                inv_temp,
                V,
                BLOCK_V=block_v,
                num_warps=num_warps,
            )
            torch.mm(uc, weight, out=dx[start:end])
            dw_out.addmm_(uc.t(), xc)

        del logits_buf
        ctx.save_for_backward(dx, dw_out)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        dx, dw = ctx.saved_tensors
        if float(grad_out.item()) != 1.0:
            dx = dx * grad_out.to(dx.dtype)
            dw = dw * grad_out.to(dw.dtype)
        return (dx, dw) + (None,) * 7


def eager_materialized_grpo_loss(
    x: torch.Tensor,
    weight: torch.Tensor,
    actions: torch.Tensor,
    lp_old: torch.Tensor,
    lp_ref: torch.Tensor,
    advantages: torch.Tensor,
    tok_weights: torch.Tensor,
    tau: float,
    eps_low: float,
    eps_high: float,
    beta_kl: float,
    alpha_ent: float,
) -> torch.Tensor:
    logits = F.linear(x, weight).float() / tau
    lpr = F.log_softmax(logits, dim=-1)
    pr = lpr.exp()
    ent = -(pr * lpr).sum(dim=-1)
    lp_t = lpr.gather(1, actions.unsqueeze(1)).squeeze(1)
    rat = (lp_t - lp_old).exp()
    pg = -torch.minimum(
        rat * advantages, torch.clamp(rat, 1.0 - eps_low, 1.0 + eps_high) * advantages
    )
    kl = (lp_ref - lp_t).exp() - (lp_ref - lp_t) - 1.0
    return (tok_weights * (pg + beta_kl * kl - alpha_ent * ent)).sum()


def chunked_pytorch_grpo_loss(
    x: torch.Tensor,
    weight: torch.Tensor,
    actions: torch.Tensor,
    lp_old: torch.Tensor,
    lp_ref: torch.Tensor,
    advantages: torch.Tensor,
    tok_weights: torch.Tensor,
    tau: float,
    eps_low: float,
    eps_high: float,
    beta_kl: float,
    alpha_ent: float,
    chunk_size: int = 1024,
) -> torch.Tensor:
    N = x.shape[0]
    losses: list[torch.Tensor] = []
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        losses.append(
            eager_materialized_grpo_loss(
                x[s:e],
                weight,
                actions[s:e],
                lp_old[s:e],
                lp_ref[s:e],
                advantages[s:e],
                tok_weights[s:e],
                tau,
                eps_low,
                eps_high,
                beta_kl,
                alpha_ent,
            )
        )
    return torch.stack(losses).sum()


def eager_materialized_dpo_loss(
    x: torch.Tensor,
    weight: torch.Tensor,
    actions: torch.Tensor,
    pair_id: torch.Tensor,
    sign: torch.Tensor,
    ref_margin: torch.Tensor,
    beta_dpo: float,
    tau: float,
) -> torch.Tensor:
    logits = F.linear(x, weight).float() / tau
    lpr = F.log_softmax(logits, dim=-1)
    lp_t = lpr.gather(1, actions.unsqueeze(1)).squeeze(1)
    pol_margin = torch.zeros_like(ref_margin, dtype=torch.float32)
    pol_margin.scatter_add_(0, pair_id, sign * lp_t)
    return -F.logsigmoid(beta_dpo * (pol_margin - ref_margin)).mean()


# ============================================================================
# 2. Two-Slice Ragged Group-Prefix-Tree Triton GQA FlashAttention (Fwd + Bwd)
# ============================================================================


@triton.jit
def _prefix_tree_gqa_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    Lse2_ptr,
    TileMeta_ptr,
    qk_scale2,
    T_total: tl.int32,
    GROUP_SIZE: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tile_id = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = qh // GROUP_SIZE

    meta_base = TileMeta_ptr + tile_id * 6
    t_start = tl.load(meta_base + 0)
    seg_end = tl.load(meta_base + 1)
    p_start = tl.load(meta_base + 2)
    p_end = tl.load(meta_base + 3)
    seg_start = tl.load(meta_base + 4)

    offs_m = t_start + tl.arange(0, BM)
    d_offs = tl.arange(0, D)
    m_valid = offs_m < seg_end

    q_base = Q_ptr + qh.to(tl.int64) * T_total * D
    k_base = K_ptr + kvh.to(tl.int64) * T_total * D
    v_base = V_ptr + kvh.to(tl.int64) * T_total * D

    q = tl.load(
        q_base + offs_m[:, None] * D + d_offs[None, :], mask=m_valid[:, None], other=0.0
    )
    m_i = tl.full([BM], -1.0e20, dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)

    if seg_start > p_start:
        p_aligned = p_start + ((p_end - p_start) // BN) * BN
        for kv_s in range(p_start, p_aligned, BN):
            offs_n = kv_s + tl.arange(0, BN)
            k = tl.load(k_base + offs_n[:, None] * D + d_offs[None, :])
            v = tl.load(v_base + offs_n[:, None] * D + d_offs[None, :])
            qk = tl.dot(q, tl.trans(k)) * qk_scale2
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_ij
        if p_aligned < p_end:
            offs_n = p_aligned + tl.arange(0, BN)
            n_valid = offs_n < p_end
            k = tl.load(
                k_base + offs_n[:, None] * D + d_offs[None, :],
                mask=n_valid[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + offs_n[:, None] * D + d_offs[None, :],
                mask=n_valid[:, None],
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * qk_scale2
            qk = tl.where(n_valid[None, :], qk, -1.0e20)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_ij

    for kv_s in range(seg_start, t_start, BN):
        offs_n = kv_s + tl.arange(0, BN)
        k = tl.load(k_base + offs_n[:, None] * D + d_offs[None, :])
        v = tl.load(v_base + offs_n[:, None] * D + d_offs[None, :])
        qk = tl.dot(q, tl.trans(k)) * qk_scale2
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_ij

    diag_end = tl.minimum(t_start + BM, seg_end)
    for kv_s in range(t_start, diag_end, BN):
        offs_n = kv_s + tl.arange(0, BN)
        n_valid = offs_n < seg_end
        k = tl.load(
            k_base + offs_n[:, None] * D + d_offs[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * D + d_offs[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * qk_scale2
        qk = tl.where(
            (offs_m[:, None] >= offs_n[None, :]) & n_valid[None, :], qk, -1.0e20
        )
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_ij

    out = (acc / l_i[:, None]).to(tl.bfloat16)
    lse2 = m_i + tl.math.log2(l_i)
    tl.store(
        Out_ptr + qh.to(tl.int64) * T_total * D + offs_m[:, None] * D + d_offs[None, :],
        out,
        mask=m_valid[:, None],
    )
    tl.store(
        Lse2_ptr + qh.to(tl.int64) * T_total + offs_m,
        lse2,
        mask=m_valid,
    )


@triton.jit
def _bwd_prep_delta_kernel(
    Out_ptr, dO_ptr, Delta_ptr, N_rows: tl.int32, D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * 64 + tl.arange(0, 64)
    d = tl.arange(0, D)
    m = rows < N_rows
    o = tl.load(
        Out_ptr + rows[:, None].to(tl.int64) * D + d[None, :], mask=m[:, None], other=0.0
    ).to(tl.float32)
    do = tl.load(
        dO_ptr + rows[:, None].to(tl.int64) * D + d[None, :], mask=m[:, None], other=0.0
    ).to(tl.float32)
    tl.store(Delta_ptr + rows, tl.sum(o * do, axis=1), mask=m)


@triton.jit
def _prefix_tree_gqa_bwd_dkdv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    Lse2_ptr,
    Delta_ptr,
    dK_ptr,
    dV_ptr,
    TileMeta_ptr,
    sm_scale,
    qk_scale2,
    T_total: tl.int32,
    GROUP_SIZE: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Lock-free GQA-fused dK/dV kernel: for a prompt KV tile (`seg_start == p_start`),
    sweeps straight across `[t_start, doc_end)` covering prompt causal queries AND all G
    ragged response branches as ONE flat contiguous slice with zero atomics!"""
    tile_id = tl.program_id(0)
    kvh = tl.program_id(1)

    meta_base = TileMeta_ptr + tile_id * 6
    t_start = tl.load(meta_base + 0)
    seg_end = tl.load(meta_base + 1)
    p_start = tl.load(meta_base + 2)
    seg_start = tl.load(meta_base + 4)
    doc_end = tl.load(meta_base + 5)

    offs_n = t_start + tl.arange(0, BN)
    d_offs = tl.arange(0, D)
    n_valid = offs_n < seg_end

    k = tl.load(
        K_ptr + kvh.to(tl.int64) * T_total * D + offs_n[:, None] * D + d_offs[None, :],
        mask=n_valid[:, None],
        other=0.0,
    )
    v = tl.load(
        V_ptr + kvh.to(tl.int64) * T_total * D + offs_n[:, None] * D + d_offs[None, :],
        mask=n_valid[:, None],
        other=0.0,
    )

    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)
    q_limit = tl.where(seg_start == p_start, doc_end, seg_end)

    for g in tl.static_range(0, GROUP_SIZE):
        qh = kvh * GROUP_SIZE + g
        q_base = Q_ptr + qh.to(tl.int64) * T_total * D
        do_base = dO_ptr + qh.to(tl.int64) * T_total * D
        lse_base = Lse2_ptr + qh.to(tl.int64) * T_total
        del_base = Delta_ptr + qh.to(tl.int64) * T_total

        offs_m0 = t_start + tl.arange(0, BM)
        m0_valid = offs_m0 < q_limit
        q0 = tl.load(
            q_base + offs_m0[:, None] * D + d_offs[None, :],
            mask=m0_valid[:, None],
            other=0.0,
        )
        do0 = tl.load(
            do_base + offs_m0[:, None] * D + d_offs[None, :],
            mask=m0_valid[:, None],
            other=0.0,
        )
        lse0 = tl.load(lse_base + offs_m0, mask=m0_valid, other=1.0e20)
        del0 = tl.load(del_base + offs_m0, mask=m0_valid, other=0.0)
        qkT0 = tl.dot(k, tl.trans(q0)) * qk_scale2
        pT0 = tl.math.exp2(qkT0 - lse0[None, :])
        pT0 = tl.where(
            (offs_m0[None, :] >= offs_n[:, None]) & m0_valid[None, :] & n_valid[:, None],
            pT0,
            0.0,
        )
        dv += tl.dot(pT0.to(tl.bfloat16), do0)
        dpT0 = tl.dot(v, tl.trans(do0))
        dsT0 = (pT0 * (dpT0 - del0[None, :])).to(tl.bfloat16)
        dk += tl.dot(dsT0, q0)

        q_rem_start = t_start + BM
        if q_rem_start < q_limit:
            q_aligned = q_rem_start + ((q_limit - q_rem_start) // BM) * BM
            for q_s in range(q_rem_start, q_aligned, BM):
                offs_m = q_s + tl.arange(0, BM)
                q = tl.load(q_base + offs_m[:, None] * D + d_offs[None, :])
                do = tl.load(do_base + offs_m[:, None] * D + d_offs[None, :])
                lse = tl.load(lse_base + offs_m)
                delt = tl.load(del_base + offs_m)
                qkT = tl.dot(k, tl.trans(q)) * qk_scale2
                pT = tl.math.exp2(qkT - lse[None, :])
                pT = tl.where(n_valid[:, None], pT, 0.0)
                dv += tl.dot(pT.to(tl.bfloat16), do)
                dpT = tl.dot(v, tl.trans(do))
                dsT = (pT * (dpT - delt[None, :])).to(tl.bfloat16)
                dk += tl.dot(dsT, q)
            if q_aligned < q_limit:
                offs_m = q_aligned + tl.arange(0, BM)
                m_valid = offs_m < q_limit
                q = tl.load(
                    q_base + offs_m[:, None] * D + d_offs[None, :],
                    mask=m_valid[:, None],
                    other=0.0,
                )
                do = tl.load(
                    do_base + offs_m[:, None] * D + d_offs[None, :],
                    mask=m_valid[:, None],
                    other=0.0,
                )
                lse = tl.load(lse_base + offs_m, mask=m_valid, other=1.0e20)
                delt = tl.load(del_base + offs_m, mask=m_valid, other=0.0)
                qkT = tl.dot(k, tl.trans(q)) * qk_scale2
                pT = tl.math.exp2(qkT - lse[None, :])
                pT = tl.where(m_valid[None, :] & n_valid[:, None], pT, 0.0)
                dv += tl.dot(pT.to(tl.bfloat16), do)
                dpT = tl.dot(v, tl.trans(do))
                dsT = (pT * (dpT - delt[None, :])).to(tl.bfloat16)
                dk += tl.dot(dsT, q)

    tl.store(
        dK_ptr + kvh.to(tl.int64) * T_total * D + offs_n[:, None] * D + d_offs[None, :],
        (dk * sm_scale).to(tl.bfloat16),
        mask=n_valid[:, None],
    )
    tl.store(
        dV_ptr + kvh.to(tl.int64) * T_total * D + offs_n[:, None] * D + d_offs[None, :],
        dv.to(tl.bfloat16),
        mask=n_valid[:, None],
    )


@triton.jit
def _prefix_tree_gqa_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    Lse2_ptr,
    Delta_ptr,
    dQ_ptr,
    TileMeta_ptr,
    sm_scale,
    qk_scale2,
    T_total: tl.int32,
    GROUP_SIZE: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tile_id = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = qh // GROUP_SIZE

    meta_base = TileMeta_ptr + tile_id * 6
    t_start = tl.load(meta_base + 0)
    seg_end = tl.load(meta_base + 1)
    p_start = tl.load(meta_base + 2)
    p_end = tl.load(meta_base + 3)
    seg_start = tl.load(meta_base + 4)

    offs_m = t_start + tl.arange(0, BM)
    d_offs = tl.arange(0, D)
    m_valid = offs_m < seg_end

    q = tl.load(
        Q_ptr + qh.to(tl.int64) * T_total * D + offs_m[:, None] * D + d_offs[None, :],
        mask=m_valid[:, None],
        other=0.0,
    )
    do = tl.load(
        dO_ptr + qh.to(tl.int64) * T_total * D + offs_m[:, None] * D + d_offs[None, :],
        mask=m_valid[:, None],
        other=0.0,
    )
    lse2 = tl.load(
        Lse2_ptr + qh.to(tl.int64) * T_total + offs_m, mask=m_valid, other=1.0e20
    )
    delt = tl.load(
        Delta_ptr + qh.to(tl.int64) * T_total + offs_m, mask=m_valid, other=0.0
    )

    k_base = K_ptr + kvh.to(tl.int64) * T_total * D
    v_base = V_ptr + kvh.to(tl.int64) * T_total * D
    dq = tl.zeros([BM, D], dtype=tl.float32)

    if seg_start > p_start:
        p_aligned = p_start + ((p_end - p_start) // BN) * BN
        for kv_s in range(p_start, p_aligned, BN):
            offs_n = kv_s + tl.arange(0, BN)
            k = tl.load(k_base + offs_n[:, None] * D + d_offs[None, :])
            v = tl.load(v_base + offs_n[:, None] * D + d_offs[None, :])
            qk = tl.dot(q, tl.trans(k)) * qk_scale2
            p = tl.math.exp2(qk - lse2[:, None])
            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delt[:, None])).to(tl.bfloat16)
            dq += tl.dot(ds, k)
        if p_aligned < p_end:
            offs_n = p_aligned + tl.arange(0, BN)
            n_valid = offs_n < p_end
            k = tl.load(
                k_base + offs_n[:, None] * D + d_offs[None, :],
                mask=n_valid[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + offs_n[:, None] * D + d_offs[None, :],
                mask=n_valid[:, None],
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * qk_scale2
            p = tl.where(n_valid[None, :], tl.math.exp2(qk - lse2[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delt[:, None])).to(tl.bfloat16)
            dq += tl.dot(ds, k)

    for kv_s in range(seg_start, t_start, BN):
        offs_n = kv_s + tl.arange(0, BN)
        k = tl.load(k_base + offs_n[:, None] * D + d_offs[None, :])
        v = tl.load(v_base + offs_n[:, None] * D + d_offs[None, :])
        qk = tl.dot(q, tl.trans(k)) * qk_scale2
        p = tl.math.exp2(qk - lse2[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delt[:, None])).to(tl.bfloat16)
        dq += tl.dot(ds, k)

    diag_end = tl.minimum(t_start + BM, seg_end)
    for kv_s in range(t_start, diag_end, BN):
        offs_n = kv_s + tl.arange(0, BN)
        n_valid = offs_n < seg_end
        k = tl.load(
            k_base + offs_n[:, None] * D + d_offs[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * D + d_offs[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * qk_scale2
        p = tl.math.exp2(qk - lse2[:, None])
        p = tl.where((offs_m[:, None] >= offs_n[None, :]) & n_valid[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delt[:, None])).to(tl.bfloat16)
        dq += tl.dot(ds, k)

    tl.store(
        dQ_ptr + qh.to(tl.int64) * T_total * D + offs_m[:, None] * D + d_offs[None, :],
        (dq * sm_scale).to(tl.bfloat16),
        mask=m_valid[:, None],
    )


class _PrefixTreeGQAFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q_htd: torch.Tensor,
        k_htd: torch.Tensor,
        v_htd: torch.Tensor,
        tile_meta: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        qc = q_htd.contiguous()
        kc = k_htd.contiguous()
        vc = v_htd.contiguous()
        Hq, T_tot, D = qc.shape
        Hkv = kc.shape[0]
        grp = Hq // Hkv
        n_tiles = tile_meta.shape[0]

        out = torch.empty_like(qc)
        lse2 = torch.empty((Hq, T_tot), device=qc.device, dtype=torch.float32)
        _prefix_tree_gqa_fwd_kernel[(n_tiles, Hq)](
            qc,
            kc,
            vc,
            out,
            lse2,
            tile_meta,
            sm_scale * RCP_LN2,
            T_tot,
            GROUP_SIZE=grp,
            D=D,
            BM=64,
            BN=64,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(qc, kc, vc, out, lse2, tile_meta)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx: Any, do_htd: torch.Tensor) -> tuple[Any, ...]:
        qc, kc, vc, out, lse2, tile_meta = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        Hq, T_tot, D = qc.shape
        Hkv = kc.shape[0]
        grp = Hq // Hkv
        n_tiles = tile_meta.shape[0]
        doc = do_htd.contiguous()

        delta = torch.empty((Hq, T_tot), device=qc.device, dtype=torch.float32)
        _bwd_prep_delta_kernel[(triton.cdiv(Hq * T_tot, 64),)](
            out, doc, delta, Hq * T_tot, D=D, num_warps=4
        )

        dq = torch.empty_like(qc)
        dk = torch.empty_like(kc)
        dv = torch.empty_like(vc)

        _prefix_tree_gqa_bwd_dkdv_kernel[(n_tiles, Hkv)](
            qc,
            kc,
            vc,
            doc,
            lse2,
            delta,
            dk,
            dv,
            tile_meta,
            sm_scale,
            sm_scale * RCP_LN2,
            T_tot,
            GROUP_SIZE=grp,
            D=D,
            BM=64,
            BN=64,
            num_warps=4,
            num_stages=2,
        )
        _prefix_tree_gqa_bwd_dq_kernel[(n_tiles, Hq)](
            qc,
            kc,
            vc,
            doc,
            lse2,
            delta,
            dq,
            tile_meta,
            sm_scale,
            sm_scale * RCP_LN2,
            T_tot,
            GROUP_SIZE=grp,
            D=D,
            BM=64,
            BN=64,
            num_warps=4,
            num_stages=2,
        )
        return dq, dk, dv, None, None


# ============================================================================
# 3. RoPE, Tree Metadata Builder, & RLVR Transformer Model
# ============================================================================


def build_rope_cache_1d(
    pos_ids: torch.Tensor, head_dim: int, base: float = 500000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, half, device=pos_ids.device, dtype=torch.float32)
            / float(half)
        )
    )
    angles = pos_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def apply_rope_thd(
    x_thd: torch.Tensor, cos_td: torch.Tensor, sin_td: torch.Tensor
) -> torch.Tensor:
    # x_thd: [T, H, D], cos_td/sin_td: [T, D//2]
    half = x_thd.shape[-1] // 2
    x1 = x_thd[..., :half].float()
    x2 = x_thd[..., half:].float()
    c = cos_td.unsqueeze(1)
    s = sin_td.unsqueeze(1)
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    return torch.cat([y1, y2], dim=-1).to(x_thd.dtype)


def build_prefix_tree_pack_metadata(
    trees: list[tuple[int, list[int]]], device: torch.device
) -> dict[str, Any]:
    """Builds exact GPU metadata for packed prefix trees `[(L_p, [L_{g0}, L_{g1}, ...]), ...]`."""
    meta_rows: list[list[int]] = []
    pos_list: list[int] = []
    resp_indices: list[int] = []
    doc_ids: list[int] = []
    branch_ids: list[int] = []
    cur = 0
    for d_idx, (lp, r_lens) in enumerate(trees):
        p_s = cur
        p_e = cur + lp
        doc_e = p_e + sum(r_lens)
        for ts in range(p_s, p_e, 64):
            meta_rows.append([ts, p_e, p_s, p_e, p_s, doc_e])
        pos_list.extend(range(lp))
        doc_ids.extend([d_idx] * lp)
        branch_ids.extend([0] * lp)
        cur = p_e
        for g_idx, rl in enumerate(r_lens):
            r_s = cur
            r_e = cur + rl
            for ts in range(r_s, r_e, 64):
                meta_rows.append([ts, r_e, p_s, p_e, r_s, doc_e])
            pos_list.extend(range(lp, lp + rl))
            resp_indices.extend(range(r_s, r_e))
            doc_ids.extend([d_idx] * rl)
            branch_ids.extend([g_idx + 1] * rl)
            cur = r_e

    T_tot = cur
    T_pad128 = ((T_tot + 127) // 128) * 128
    pad_extra = T_pad128 - T_tot
    doc_pad = doc_ids + [-1] * pad_extra
    branch_pad = branch_ids + [-1] * pad_extra

    return {
        "T_total": T_tot,
        "T_pad128": T_pad128,
        "T_resp": len(resp_indices),
        "tile_meta": torch.tensor(meta_rows, dtype=torch.int32, device=device).contiguous(),
        "pos_ids": torch.tensor(pos_list, dtype=torch.long, device=device),
        "resp_idx": torch.tensor(resp_indices, dtype=torch.long, device=device),
        "doc_id_pad": torch.tensor(doc_pad, dtype=torch.int32, device=device),
        "branch_id_pad": torch.tensor(branch_pad, dtype=torch.int32, device=device),
    }


class RLVRGQABlock(nn.Module):
    def __init__(
        self, hidden: int, ffn_dim: int, n_q: int, n_kv: int, head_dim: int = 128
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.sm_scale = 1.0 / math.sqrt(float(head_dim))

        self.in_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.w_q = nn.Linear(hidden, n_q * head_dim, bias=False)
        self.w_k = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.w_v = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.w_o = nn.Linear(n_q * head_dim, hidden, bias=False)

        self.ffn_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.w_gate = nn.Linear(hidden, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden, bias=False)

    def forward_padded_fa2(
        self, x_bsh: torch.Tensor, cos_sd: torch.Tensor, sin_sd: torch.Tensor
    ) -> torch.Tensor:
        B, S, _ = x_bsh.shape
        h = self.in_norm(x_bsh)
        q = self.w_q(h).view(B * S, self.n_q, self.head_dim)
        k = self.w_k(h).view(B * S, self.n_kv, self.head_dim)
        v = self.w_v(h).view(B, S, self.n_kv, self.head_dim)
        c_rep = cos_sd.repeat(B, 1)
        s_rep = sin_sd.repeat(B, 1)
        q = apply_rope_thd(q, c_rep, s_rep).view(B, S, self.n_q, self.head_dim)
        k = apply_rope_thd(k, c_rep, s_rep).view(B, S, self.n_kv, self.head_dim)
        o = flash_attn_func(q, k, v, causal=True, softmax_scale=self.sm_scale)
        x_bsh = x_bsh + self.w_o(o.reshape(B, S, self.n_q * self.head_dim))
        hf = self.ffn_norm(x_bsh)
        return x_bsh + self.w_down(F.silu(self.w_gate(hf)) * self.w_up(hf))

    def forward_varlen_fa2(
        self,
        x_th: torch.Tensor,
        cos_td: torch.Tensor,
        sin_td: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        T = x_th.shape[0]
        h = self.in_norm(x_th)
        q = apply_rope_thd(
            self.w_q(h).view(T, self.n_q, self.head_dim), cos_td, sin_td
        )
        k = apply_rope_thd(
            self.w_k(h).view(T, self.n_kv, self.head_dim), cos_td, sin_td
        )
        v = self.w_v(h).view(T, self.n_kv, self.head_dim)
        o = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
            softmax_scale=self.sm_scale,
        )
        x_th = x_th + self.w_o(o.reshape(T, self.n_q * self.head_dim))
        hf = self.ffn_norm(x_th)
        return x_th + self.w_down(F.silu(self.w_gate(hf)) * self.w_up(hf))

    def forward_prefix_tree_triton(
        self,
        x_th: torch.Tensor,
        cos_td: torch.Tensor,
        sin_td: torch.Tensor,
        tile_meta: torch.Tensor,
    ) -> torch.Tensor:
        T = x_th.shape[0]
        h = self.in_norm(x_th)
        q = (
            apply_rope_thd(
                self.w_q(h).view(T, self.n_q, self.head_dim), cos_td, sin_td
            )
            .transpose(0, 1)
            .contiguous()
        )
        k = (
            apply_rope_thd(
                self.w_k(h).view(T, self.n_kv, self.head_dim), cos_td, sin_td
            )
            .transpose(0, 1)
            .contiguous()
        )
        v = self.w_v(h).view(T, self.n_kv, self.head_dim).transpose(0, 1).contiguous()
        o_htd = _PrefixTreeGQAFn.apply(q, k, v, tile_meta, self.sm_scale)
        o_th = o_htd.transpose(0, 1).contiguous().view(T, self.n_q * self.head_dim)
        x_th = x_th + self.w_o(o_th)
        hf = self.ffn_norm(x_th)
        return x_th + self.w_down(F.silu(self.w_gate(hf)) * self.w_up(hf))

    def forward_prefix_tree_flex(
        self,
        x_th: torch.Tensor,
        cos_td: torch.Tensor,
        sin_td: torch.Tensor,
        T_pad128: int,
        block_mask: Any,
        compiled_flex: Any,
    ) -> torch.Tensor:
        T = x_th.shape[0]
        pad = T_pad128 - T
        h = self.in_norm(x_th)
        q = apply_rope_thd(
            self.w_q(h).view(T, self.n_q, self.head_dim), cos_td, sin_td
        ).permute(1, 0, 2)
        k = apply_rope_thd(
            self.w_k(h).view(T, self.n_kv, self.head_dim), cos_td, sin_td
        ).permute(1, 0, 2)
        v = self.w_v(h).view(T, self.n_kv, self.head_dim).permute(1, 0, 2)
        if pad > 0:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        o_pad = compiled_flex(
            q.unsqueeze(0).contiguous(),
            k.unsqueeze(0).contiguous(),
            v.unsqueeze(0).contiguous(),
            block_mask=block_mask,
            scale=self.sm_scale,
            enable_gqa=True,
        )
        o_th = (
            o_pad.squeeze(0)[:, :T, :]
            .transpose(0, 1)
            .contiguous()
            .view(T, self.n_q * self.head_dim)
        )
        x_th = x_th + self.w_o(o_th)
        hf = self.ffn_norm(x_th)
        return x_th + self.w_down(F.silu(self.w_gate(hf)) * self.w_up(hf))

    def forward(
        self,
        x: torch.Tensor,
        mode: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int = 0,
        tile_meta: torch.Tensor | None = None,
        T_pad128: int = 0,
        block_mask: Any = None,
        compiled_flex: Any = None,
    ) -> torch.Tensor:
        if mode == "padded_fa2":
            return self.forward_padded_fa2(x, cos, sin)
        if mode == "varlen_fa2":
            assert cu_seqlens is not None
            return self.forward_varlen_fa2(x, cos, sin, cu_seqlens, max_seqlen)
        if mode == "prefix_tree_triton":
            assert tile_meta is not None
            return self.forward_prefix_tree_triton(x, cos, sin, tile_meta)
        if mode == "prefix_tree_flex":
            return self.forward_prefix_tree_flex(
                x, cos, sin, T_pad128, block_mask, compiled_flex
            )
        raise ValueError(f"Unknown RLVRGQABlock mode: {mode}")


class FusedGRPOLMHead(nn.Module):
    def __init__(self, hidden: int, vocab_size: int) -> None:
        super().__init__()
        self.final_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden)))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_2d: torch.Tensor,
        resp_idx: torch.Tensor | None,
        actions: torch.Tensor,
        lp_old: torch.Tensor,
        lp_ref: torch.Tensor,
        advantages: torch.Tensor,
        tok_weights: torch.Tensor,
        *,
        tau: float = 0.7,
        eps_low: float = 0.20,
        eps_high: float = 0.28,
        beta_kl: float = 0.04,
        alpha_ent: float = 0.01,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        if resp_idx is not None:
            x_sel = x_2d.index_select(0, resp_idx)
        else:
            x_sel = x_2d
        h_norm = self.final_norm(x_sel)
        return _FusedLinearGRPOFn.apply(
            h_norm,
            self.weight,
            actions,
            lp_old,
            lp_ref,
            advantages,
            tok_weights,
            tau,
            eps_low,
            eps_high,
            beta_kl,
            alpha_ent,
            chunk_size,
            True,
        )


class RLVRModel(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden: int,
        ffn_dim: int,
        n_q: int,
        n_kv: int,
        vocab_size: int,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.blocks = nn.ModuleList(
            [RLVRGQABlock(hidden, ffn_dim, n_q, n_kv, head_dim) for _ in range(n_layers)]
        )
        self.lm_head = FusedGRPOLMHead(hidden, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        mode: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        resp_idx: torch.Tensor | None,
        actions: torch.Tensor,
        lp_old: torch.Tensor,
        lp_ref: torch.Tensor,
        advantages: torch.Tensor,
        tok_weights: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int = 0,
        tile_meta: torch.Tensor | None = None,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        cur_x = x
        for blk in self.blocks:
            cur_x = blk(
                cur_x,
                mode,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                tile_meta=tile_meta,
            )
        x_2d = cur_x.reshape(-1, self.hidden)
        return self.lm_head(
            x_2d,
            resp_idx,
            actions,
            lp_old,
            lp_ref,
            advantages,
            tok_weights,
            chunk_size=chunk_size,
        )


# ============================================================================
# 4. Suite 0: Entropy Cancellation Proof, FP64 Chain-Rule, & BF16 Parity
# ============================================================================


def run_suite_0_audits(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    # 0.1 FP32 Catastrophic Cancellation Audit on Vocabulary Entropy (V = 128,256)
    gen_e = torch.Generator(device=device)
    gen_e.manual_seed(12101)
    R_e, V_e = 64, 128256
    # Confident reasoning CoT logit distribution (std=1.0 + +24.0 spike -> true H ~ 0.015 nats!)
    base_z32 = torch.randn((R_e, V_e), generator=gen_e, device=device, dtype=torch.float32)
    top_tok = torch.randint(0, V_e, (R_e, 1), generator=gen_e, device=device)
    base_z32.scatter_add_(
        1, top_tok, torch.full((R_e, 1), 24.0, device=device, dtype=torch.float32)
    )

    entropy_shift_audit: dict[str, Any] = {}
    for shift_mu in (0.0, 4096.0, 65536.0):
        z32 = (base_z32 + shift_mu).contiguous()
        # Exact FP64 entropy of the exact FP32 tensor `z32` passed into both kernels
        lpr64 = F.log_softmax(z32.double(), dim=-1)
        ent_exact64 = -(lpr64.exp() * lpr64).sum(dim=-1)

        out_naive = torch.empty((R_e,), device=device, dtype=torch.float32)
        out_monoid = torch.empty((R_e,), device=device, dtype=torch.float32)
        _naive_uncentered_entropy_kernel[(R_e,)](
            z32, z32.stride(0), out_naive, V_e, BLOCK_V=8192, num_warps=16
        )
        _centered_monoid_entropy_kernel[(R_e,)](
            z32, z32.stride(0), out_monoid, V_e, BLOCK_V=8192, num_warps=16
        )
        naive_rel = float(
            (out_naive.double() - ent_exact64).norm().item() / ent_exact64.norm().item()
        )
        monoid_rel = float(
            (out_monoid.double() - ent_exact64).norm().item() / ent_exact64.norm().item()
        )
        entropy_shift_audit[f"shift_{int(shift_mu)}"] = {
            "mean_true_entropy_nats": round(float(ent_exact64.mean().item()), 6),
            "naive_uncentered_fp32_rel_l2": round(naive_rel, 8),
            "naive_uncentered_negative_count": int((out_naive < 0.0).sum().item()),
            "naive_uncentered_min_val": round(float(out_naive.min().item()), 6),
            "centered_monoid_fp32_rel_l2": round(monoid_rel, 8),
            "centered_monoid_negative_count": int((out_monoid < 0.0).sum().item()),
        }

    # 0.2 FP64 Analytical Chain-Rule & Richardson Central-Difference (h vs h/2 -> ~4.00x)
    gen64 = torch.Generator(device=device)
    gen64.manual_seed(12105)

    # (a) Fused Linear-GRPO/DAPO analytical chain-rule in FP64 (with both unclipped & clipped tokens!)
    Nt_f, H_f, V_f = 32, 64, 256
    tau_f, el_f, eh_f, bkl_f, aent_f = 0.7, 0.20, 0.28, 0.04, 0.015
    h64 = torch.randn((Nt_f, H_f), generator=gen64, device=device, dtype=torch.float64) * 0.25
    w64 = torch.randn((V_f, H_f), generator=gen64, device=device, dtype=torch.float64) * 0.25
    act_f = torch.randint(0, V_f, (Nt_f,), generator=gen64, device=device)
    wt_f = (torch.arange(Nt_f, device=device) >= 6).to(torch.float64) / 26.0

    with torch.no_grad():
        lp_init = F.log_softmax((h64 @ w64.t()) / tau_f, dim=-1).gather(
            1, act_f.unsqueeze(1)
        ).squeeze(1)
        # First half: unclipped (ratio ~ 1.02); second half: actively clipped (ratio ~ 1.55 with A > 0!)
        delta_lp = torch.full((Nt_f,), -0.02, device=device, dtype=torch.float64)
        delta_lp[Nt_f // 2 :] = -0.44
        lp_old_f = lp_init + delta_lp
        lp_ref_f = lp_init + 0.03 * torch.randn((Nt_f,), generator=gen64, device=device, dtype=torch.float64)
        adv_f = torch.linspace(-1.2, 1.4, Nt_f, device=device, dtype=torch.float64)
        adv_f[Nt_f // 2 :] = torch.abs(adv_f[Nt_f // 2 :]) + 0.5  # strictly > 0 so r ~ 1.55 > 1.28 clips!

    def _grpo_loss64(x_in: torch.Tensor, w_in: torch.Tensor) -> torch.Tensor:
        z = (x_in @ w_in.t()) / tau_f
        lpr = F.log_softmax(z, dim=-1)
        p = lpr.exp()
        ent = -(p * lpr).sum(dim=-1)
        lpt = lpr.gather(1, act_f.unsqueeze(1)).squeeze(1)
        r = (lpt - lp_old_f).exp()
        rc = torch.clamp(r, 1.0 - el_f, 1.0 + eh_f)
        pg = -torch.minimum(r * adv_f, rc * adv_f)
        kl = (lp_ref_f - lpt).exp() - (lp_ref_f - lpt) - 1.0
        return (wt_f * (pg + bkl_f * kl - aent_f * ent)).sum()

    # Analytical gradient using exact kernel equations in FP64
    z_a = (h64 @ w64.t()) / tau_f
    m_a = z_a.max(dim=-1, keepdim=True).values
    d_a = (z_a - m_a).exp().sum(dim=-1, keepdim=True)
    e_a = ((z_a - m_a) * (z_a - m_a).exp()).sum(dim=-1, keepdim=True)
    lse_a = (m_a + d_a.log()).squeeze(-1)
    ent_a = (d_a.log() - e_a / d_a).squeeze(-1)
    p_a = (z_a - lse_a.unsqueeze(-1)).exp()
    lpt_a = z_a.gather(1, act_f.unsqueeze(1)).squeeze(1) - lse_a
    r_a = (lpt_a - lp_old_f).exp()
    rc_a = torch.clamp(r_a, 1.0 - el_f, 1.0 + eh_f)
    unclip_a = (r_a * adv_f) <= (rc_a * adv_f)
    g_lp_a = wt_f * (
        torch.where(unclip_a, -r_a * adv_f, 0.0)
        + bkl_f * (1.0 - (lp_ref_f - lpt_a).exp())
    )
    ind_a = F.one_hot(act_f, num_classes=V_f).to(torch.float64)
    du_a = (
        g_lp_a.unsqueeze(-1) * (ind_a - p_a)
        + (wt_f * aent_f).unsqueeze(-1) * p_a * (z_a - lse_a.unsqueeze(-1) + ent_a.unsqueeze(-1))
    ) / tau_f
    dh_grpo_a = du_a @ w64
    dw_grpo_a = du_a.t() @ h64

    fd_ratios: dict[str, float] = {}
    for idx_i, (nm, base_t, ga) in enumerate(
        [("grpo_dh", h64, dh_grpo_a), ("grpo_dw", w64, dw_grpo_a)]
    ):
        dv = torch.randn_like(base_t, generator=gen64)
        dv = dv / dv.pow(2).mean().sqrt().clamp(min=1e-18)
        ana = float((ga * dv).sum().item())
        errs = []
        for h_st in (5e-3, 2.5e-3):
            ins_p = [h64.clone(), w64.clone()]
            ins_m = [h64.clone(), w64.clone()]
            ins_p[idx_i] = ins_p[idx_i] + h_st * dv
            ins_m[idx_i] = ins_m[idx_i] - h_st * dv
            fd_v = float((_grpo_loss64(*ins_p) - _grpo_loss64(*ins_m)).item()) / (2.0 * h_st)
            errs.append(abs(fd_v - ana))
        fd_ratios[nm] = round(errs[0] / max(errs[1], 1e-18), 4)

    # (b) Fused Saved-LSE Linear-DPO analytical chain-rule in FP64
    pair_f = torch.repeat_interleave(torch.arange(4, device=device), 8)
    sign_f = torch.tensor(([1.0] * 4 + [-1.0] * 4) * 4, device=device, dtype=torch.float64)
    ref_m_f = torch.randn((4,), generator=gen64, device=device, dtype=torch.float64) * 0.5
    bdpo_f = 0.1

    def _dpo_loss64(x_in: torch.Tensor, w_in: torch.Tensor) -> torch.Tensor:
        z = (x_in @ w_in.t()) / tau_f
        lpr = F.log_softmax(z, dim=-1)
        lpt = lpr.gather(1, act_f.unsqueeze(1)).squeeze(1)
        pm = torch.zeros((4,), device=device, dtype=torch.float64)
        pm.scatter_add_(0, pair_f, sign_f * lpt)
        return -F.logsigmoid(bdpo_f * (pm - ref_m_f)).mean()

    pm_a = torch.zeros((4,), device=device, dtype=torch.float64)
    pm_a.scatter_add_(0, pair_f, sign_f * lpt_a)
    gp_a = (-bdpo_f / 4.0) * torch.sigmoid(-bdpo_f * (pm_a - ref_m_f))
    gt_a = sign_f * gp_a.index_select(0, pair_f)
    du_dpo_a = (gt_a.unsqueeze(-1) * (ind_a - p_a)) / tau_f
    dh_dpo_a = du_dpo_a @ w64
    dw_dpo_a = du_dpo_a.t() @ h64

    for idx_i, (nm, base_t, ga) in enumerate(
        [("dpo_dh", h64, dh_dpo_a), ("dpo_dw", w64, dw_dpo_a)]
    ):
        dv = torch.randn_like(base_t, generator=gen64)
        dv = dv / dv.pow(2).mean().sqrt().clamp(min=1e-18)
        ana = float((ga * dv).sum().item())
        errs = []
        for h_st in (5e-3, 2.5e-3):
            ins_p = [h64.clone(), w64.clone()]
            ins_m = [h64.clone(), w64.clone()]
            ins_p[idx_i] = ins_p[idx_i] + h_st * dv
            ins_m[idx_i] = ins_m[idx_i] - h_st * dv
            fd_v = float((_dpo_loss64(*ins_p) - _dpo_loss64(*ins_m)).item()) / (2.0 * h_st)
            errs.append(abs(fd_v - ana))
        fd_ratios[nm] = round(errs[0] / max(errs[1], 1e-18), 4)

    # (c) Prefix-Tree GQA Attention analytical chain-rule in FP64 (4Q / 2KV, D=32, 2 prompts x 3 branches)
    trees_fd = [(18, [11, 15, 9]), (14, [13, 8, 12])]
    T_tfd = sum(lp + sum(rl) for lp, rl in trees_fd)
    Hq_f, Hkv_f, D_tf = 4, 2, 32
    sm_tf = 1.0 / math.sqrt(float(D_tf))
    qt64 = torch.randn((Hq_f, T_tfd, D_tf), generator=gen64, device=device, dtype=torch.float64)
    kt64 = torch.randn((Hkv_f, T_tfd, D_tf), generator=gen64, device=device, dtype=torch.float64)
    vt64 = torch.randn((Hkv_f, T_tfd, D_tf), generator=gen64, device=device, dtype=torch.float64)
    dot64 = torch.randn_like(qt64, generator=gen64)

    # Build exact tree visibility boolean mask [T_tfd, T_tfd]
    vis_mask = torch.zeros((T_tfd, T_tfd), device=device, dtype=torch.bool)
    cur_t = 0
    for lp, rls in trees_fd:
        ps, pe = cur_t, cur_t + lp
        for i in range(ps, pe):
            vis_mask[i, ps : i + 1] = True
        cur_t = pe
        for rl in rls:
            rs, re = cur_t, cur_t + rl
            vis_mask[rs:re, ps:pe] = True
            for i in range(rs, re):
                vis_mask[i, rs : i + 1] = True
            cur_t = re

    def _tree_loss64(qi: torch.Tensor, ki: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        ke = ki.repeat_interleave(2, dim=0)
        ve = vi.repeat_interleave(2, dim=0)
        sc = torch.matmul(qi, ke.transpose(-1, -2)) * sm_tf
        sc = sc.masked_fill(~vis_mask.unsqueeze(0), -1.0e30)
        p = torch.softmax(sc, dim=-1)
        o = torch.matmul(p, ve)
        return (torch.sin(o) * dot64).sum()

    # Exact analytical tree GQA backward matching our Triton two-slice equations
    ke_a = kt64.repeat_interleave(2, dim=0)
    ve_a = vt64.repeat_interleave(2, dim=0)
    sc_t = torch.matmul(qt64, ke_a.transpose(-1, -2)) * sm_tf
    pt_a = torch.softmax(sc_t.masked_fill(~vis_mask.unsqueeze(0), -1.0e30), dim=-1)
    ot_a = torch.matmul(pt_a, ve_a)
    go_t = torch.cos(ot_a) * dot64
    del_t = (ot_a * go_t).sum(dim=-1, keepdim=True)
    ds_t = pt_a * (torch.matmul(go_t, ve_a.transpose(-1, -2)) - del_t) * sm_tf
    dq_t64 = torch.matmul(ds_t, ke_a)
    dk_t64 = torch.matmul(ds_t.transpose(-1, -2), qt64).view(Hkv_f, 2, T_tfd, D_tf).sum(dim=1)
    dv_t64 = torch.matmul(pt_a.transpose(-1, -2), go_t).view(Hkv_f, 2, T_tfd, D_tf).sum(dim=1)

    for idx_i, (nm, base_t, ga) in enumerate(
        [("tree_gqa_dq", qt64, dq_t64), ("tree_gqa_dk_shared", kt64, dk_t64), ("tree_gqa_dv_shared", vt64, dv_t64)]
    ):
        dv = torch.randn_like(base_t, generator=gen64)
        dv = dv / dv.pow(2).mean().sqrt().clamp(min=1e-18)
        ana = float((ga * dv).sum().item())
        errs = []
        for h_st in (5e-3, 2.5e-3):
            ins_p = [qt64.clone(), kt64.clone(), vt64.clone()]
            ins_m = [qt64.clone(), kt64.clone(), vt64.clone()]
            ins_p[idx_i] = ins_p[idx_i] + h_st * dv
            ins_m[idx_i] = ins_m[idx_i] - h_st * dv
            fd_v = float((_tree_loss64(*ins_p) - _tree_loss64(*ins_m)).item()) / (2.0 * h_st)
            errs.append(abs(fd_v - ana))
        fd_ratios[nm] = round(errs[0] / max(errs[1], 1e-18), 4)

    # 0.3 BF16 Triton Parity vs Exact FP32 Autograd (GRPO, DPO, & 4-Layer Prefix-Tree Stack)
    gen_p = torch.Generator(device=device)
    gen_p.manual_seed(12115)
    Np, Hp, Vp = 1536, 512, 128256
    xp = torch.randn((Np, Hp), generator=gen_p, device=device, dtype=torch.bfloat16) * 0.2
    wp = torch.randn((Vp, Hp), generator=gen_p, device=device, dtype=torch.bfloat16) * 0.04
    actp = torch.randint(0, Vp, (Np,), generator=gen_p, device=device)
    with torch.no_grad():
        lp_base_p = (
            F.log_softmax((xp.float() @ wp.float().t()) / 0.7, dim=-1)
            .gather(1, actp.unsqueeze(1))
            .squeeze(1)
        )
        # 75% on-policy unclipped (|delta| <= 0.05) + 25% actively clipped (delta = -0.45, A > 0), away from 0.80/1.28 jump!
        lag_p = (torch.rand((Np,), generator=gen_p, device=device) - 0.5) * 0.10
        lag_p[3 * (Np // 4) :] = -0.45
        lpop = lp_base_p + lag_p
        lprp = lp_base_p + 0.04 * torch.randn((Np,), generator=gen_p, device=device, dtype=torch.float32)
        advp = torch.randn((Np,), generator=gen_p, device=device, dtype=torch.float32)
        advp[3 * (Np // 4) :] = torch.abs(advp[3 * (Np // 4) :]) + 0.5
    wtp = (torch.rand((Np,), generator=gen_p, device=device) > 0.2).float() / 1200.0

    xr_g = xp.float().requires_grad_(True)
    wr_g = wp.float().requires_grad_(True)
    loss_grpo_ref = eager_materialized_grpo_loss(
        xr_g, wr_g, actp, lpop, lprp, advp, wtp, 0.7, 0.20, 0.28, 0.04, 0.01
    )
    loss_grpo_ref.backward()

    xt_g = xp.clone().requires_grad_(True)
    wt_g = wp.clone().requires_grad_(True)
    loss_grpo_tri = _FusedLinearGRPOFn.apply(
        xt_g, wt_g, actp, lpop, lprp, advp, wtp, 0.7, 0.20, 0.28, 0.04, 0.01, 128, True
    )
    loss_grpo_tri.backward()

    # Also check BF16 dW accumulation drift across 12 chunks (1536 / 128) as negative control!
    xt_gb = xp.clone().requires_grad_(True)
    wt_gb = wp.clone().requires_grad_(True)
    loss_grpo_bf16dw = _FusedLinearGRPOFn.apply(
        xt_gb, wt_gb, actp, lpop, lprp, advp, wtp, 0.7, 0.20, 0.28, 0.04, 0.01, 128, False
    )
    loss_grpo_bf16dw.backward()

    grpo_parity = {
        "loss_rel_err": round(
            abs(float(loss_grpo_tri.item() - loss_grpo_ref.item()))
            / max(abs(float(loss_grpo_ref.item())), 1e-12),
            6,
        ),
        "dx_rel_l2": round(rel_l2(xt_g.grad, xr_g.grad), 6),
        "dw_fp32_accum_rel_l2": round(rel_l2(wt_g.grad, wr_g.grad), 6),
        "dw_bf16_accum_rel_l2": round(rel_l2(wt_gb.grad, wr_g.grad), 6),
    }

    # DPO BF16 Parity
    Nd = 1024
    pair_p = torch.repeat_interleave(torch.arange(4, device=device), Nd // 4)
    sign_p = torch.tensor(([1.0] * 128 + [-1.0] * 128) * 4, device=device, dtype=torch.float32)
    ref_mp = torch.randn((4,), generator=gen_p, device=device, dtype=torch.float32)

    xr_d = xp[:Nd].float().requires_grad_(True)
    wr_d = wp.float().requires_grad_(True)
    loss_dpo_ref = eager_materialized_dpo_loss(
        xr_d, wr_d, actp[:Nd], pair_p, sign_p, ref_mp, 0.1, 0.7
    )
    loss_dpo_ref.backward()

    xt_d = xp[:Nd].clone().requires_grad_(True)
    wt_d = wp.clone().requires_grad_(True)
    loss_dpo_tri = _FusedSavedLSELinearDPOFn.apply(
        xt_d, wt_d, actp[:Nd], pair_p, sign_p, ref_mp, 0.1, 0.7, 256
    )
    loss_dpo_tri.backward()

    dpo_parity = {
        "loss_rel_err": round(
            abs(float(loss_dpo_tri.item() - loss_dpo_ref.item()))
            / max(abs(float(loss_dpo_ref.item())), 1e-12),
            6,
        ),
        "dx_rel_l2": round(rel_l2(xt_d.grad, xr_d.grad), 6),
        "dw_rel_l2": round(rel_l2(wt_d.grad, wr_d.grad), 6),
    }
    del xp, wp, xr_g, wr_g, xt_g, wt_g, xt_gb, wt_gb, xr_d, wr_d, xt_d, wt_d

    # 4-Layer End-to-End Prefix-Tree Triton GQA vs Duplicated-Prompt Varlen FA2 Parity
    torch.manual_seed(12125)
    m_check = RLVRModel(4, 1024, 2816, 16, 4, 32000, head_dim=64).to(
        device=device, dtype=torch.bfloat16
    )
    trees_chk = [(290, [115, 240, 395, 180]), (410, [95, 330, 260, 150])]
    meta_chk = build_prefix_tree_pack_metadata(trees_chk, device)
    T_resp_chk = meta_chk["T_resp"]

    p_embs = [
        torch.randn((lp, 1024), device=device, dtype=torch.bfloat16)
        for lp, _ in trees_chk
    ]
    r_embs = [
        [torch.randn((rl, 1024), device=device, dtype=torch.bfloat16) for rl in rls]
        for _, rls in trees_chk
    ]
    act_chk = torch.randint(0, 32000, (T_resp_chk,), device=device)
    with torch.no_grad():
        x_init_tr = torch.cat(
            [t for b_i in range(len(trees_chk)) for t in ([p_embs[b_i]] + r_embs[b_i])],
            dim=0,
        )
        c_i, s_i = build_rope_cache_1d(meta_chk["pos_ids"], 64)
        for blk in m_check.blocks:
            x_init_tr = blk.forward_prefix_tree_triton(
                x_init_tr, c_i, s_i, meta_chk["tile_meta"]
            )
        h_resp_i = m_check.lm_head.final_norm(x_init_tr.index_select(0, meta_chk["resp_idx"]))
        lp_init_chk = (
            F.log_softmax(F.linear(h_resp_i, m_check.lm_head.weight).float() / 0.7, dim=-1)
            .gather(1, act_chk.unsqueeze(1))
            .squeeze(1)
        )
        lag_c = (torch.rand((T_resp_chk,), device=device) - 0.5) * 0.10
        lag_c[3 * (T_resp_chk // 4) :] = -0.45
        lpo_chk = lp_init_chk + lag_c
        lpr_chk = lp_init_chk + 0.03 * torch.randn((T_resp_chk,), device=device, dtype=torch.float32)
        adv_chk = torch.randn((T_resp_chk,), device=device, dtype=torch.float32)
        adv_chk[3 * (T_resp_chk // 4) :] = torch.abs(adv_chk[3 * (T_resp_chk // 4) :]) + 0.5
    wt_chk = torch.full((T_resp_chk,), 1.0 / float(T_resp_chk), device=device, dtype=torch.float32)

    # (i) Reference: Duplicated-Prompt Varlen FA2 (`veRL` style)
    p_dup = [p.clone().requires_grad_(True) for p in p_embs]
    r_dup = [[r.clone().requires_grad_(True) for r in rls] for rls in r_embs]
    for param in m_check.parameters():
        param.grad = None
    dup_chunks: list[torch.Tensor] = []
    dup_pos: list[int] = []
    dup_cu = [0]
    dup_resp_idx: list[int] = []
    cur_d = 0
    for b_i, (lp, rls) in enumerate(trees_chk):
        for g_i, rl in enumerate(rls):
            dup_chunks.append(p_dup[b_i])
            dup_chunks.append(r_dup[b_i][g_i])
            dup_pos.extend(range(lp + rl))
            dup_resp_idx.extend(range(cur_d + lp, cur_d + lp + rl))
            cur_d += lp + rl
            dup_cu.append(cur_d)
    x_dup = torch.cat(dup_chunks, dim=0)
    cu_t = torch.tensor(dup_cu, dtype=torch.int32, device=device)
    max_s_d = max(lp + max(rls) for lp, rls in trees_chk)
    cos_d, sin_d = build_rope_cache_1d(torch.tensor(dup_pos, dtype=torch.long, device=device), 64)
    for blk in m_check.blocks:
        x_dup = blk.forward_varlen_fa2(x_dup, cos_d, sin_d, cu_t, max_s_d)
    loss_dup = m_check.lm_head(
        x_dup,
        torch.tensor(dup_resp_idx, dtype=torch.long, device=device),
        act_chk,
        lpo_chk,
        lpr_chk,
        adv_chk,
        wt_chk,
        chunk_size=512,
    )
    loss_dup.backward()
    dw_q_ref = m_check.blocks[0].w_q.weight.grad.clone()
    dw_gate_ref = m_check.blocks[0].w_gate.weight.grad.clone()
    dw_lm_ref = m_check.lm_head.weight.grad.clone()

    # (ii) Ours: Prefix-Deduplicated Tree Triton GQA + Active-Response FusedLinearGRPO
    p_tr = [p.clone().requires_grad_(True) for p in p_embs]
    r_tr = [[r.clone().requires_grad_(True) for r in rls] for rls in r_embs]
    for param in m_check.parameters():
        param.grad = None
    tr_chunks: list[torch.Tensor] = []
    for b_i in range(len(trees_chk)):
        tr_chunks.append(p_tr[b_i])
        tr_chunks.extend(r_tr[b_i])
    x_tr = torch.cat(tr_chunks, dim=0)
    cos_t, sin_t = build_rope_cache_1d(meta_chk["pos_ids"], 64)
    for blk in m_check.blocks:
        x_tr = blk.forward_prefix_tree_triton(x_tr, cos_t, sin_t, meta_chk["tile_meta"])
    loss_tr = m_check.lm_head(
        x_tr,
        meta_chk["resp_idx"],
        act_chk,
        lpo_chk,
        lpr_chk,
        adv_chk,
        wt_chk,
        chunk_size=512,
    )
    loss_tr.backward()

    tree_stack_parity = {
        "loss_rel_err": round(
            abs(float(loss_tr.item() - loss_dup.item()))
            / max(abs(float(loss_dup.item())), 1e-12),
            6,
        ),
        "dx_shared_prompt0_rel_l2": round(rel_l2(p_tr[0].grad, p_dup[0].grad), 6),
        "dx_resp0_0_rel_l2": round(rel_l2(r_tr[0][0].grad, r_dup[0][0].grad), 6),
        "dw_layer0_wq_rel_l2": round(rel_l2(m_check.blocks[0].w_q.weight.grad, dw_q_ref), 6),
        "dw_layer0_wgate_rel_l2": round(
            rel_l2(m_check.blocks[0].w_gate.weight.grad, dw_gate_ref), 6
        ),
        "dw_lm_head_rel_l2": round(rel_l2(m_check.lm_head.weight.grad, dw_lm_ref), 6),
    }

    del m_check
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fp32_entropy_shift_invariance_audit": entropy_shift_audit,
        "fp64_chain_rule_refinement": {
            "ratios_h_vs_half_h": fd_ratios,
            "min_refinement_ratio": round(min(fd_ratios.values()), 4),
        },
        "bf16_triton_parity": {
            "fused_linear_grpo_dapo": grpo_parity,
            "fused_saved_lse_linear_dpo": dpo_parity,
            "four_layer_prefix_tree_vs_duplicated_varlen": tree_stack_parity,
        },
    }


# ============================================================================
# 5. Suite A: 1-GPU Large-Vocabulary (V = 128,256) GRPO/DAPO & DPO Head Scaling
# ============================================================================


def run_suite_a_terminal_post_training_kernels(
    device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    H_a = 4096
    V_a = 128256
    grpo_tokens = [2048, 8192] if is_smoke else [2048, 4096, 8192, 16384]
    dpo_tokens = [2048] if is_smoke else [2048, 8192]
    warmup_a = 2 if is_smoke else 3
    iters_a = 4 if is_smoke else 8
    chunk_sz = 1024

    weight_a = (torch.randn((V_a, H_a), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    compiled_grpo = torch.compile(eager_materialized_grpo_loss, mode="default")
    compiled_dpo = torch.compile(eager_materialized_dpo_loss, mode="default")

    grpo_scaling: dict[str, Any] = {}
    for N in grpo_tokens:
        x_base = torch.randn((N, H_a), device=device, dtype=torch.bfloat16)
        acts = torch.randint(0, V_a, (N,), device=device)
        lpo = torch.randn((N,), device=device, dtype=torch.float32) - 11.5
        lpr = lpo + 0.04 * torch.randn((N,), device=device, dtype=torch.float32)
        adv = torch.randn((N,), device=device, dtype=torch.float32)
        wts = torch.full((N,), 1.0 / float(N), device=device, dtype=torch.float32)

        # 3-GEMM FLOPs: Fwd (2 N H V) + dX (2 N V H) + dW (2 V N H) = 6 N H V
        gemm3_flops = 6.0 * float(N) * float(H_a) * float(V_a)
        modes_n: dict[str, Any] = {}

        mode_defs: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = []
        if N <= 8192:
            mode_defs.append(
                (
                    "eager_fp32_grpo_materialized",
                    lambda xin: eager_materialized_grpo_loss(
                        xin, weight_a, acts, lpo, lpr, adv, wts, 0.7, 0.20, 0.28, 0.04, 0.01
                    ),
                )
            )
            mode_defs.append(
                (
                    "torch_compile_grpo_materialized",
                    lambda xin: compiled_grpo(
                        xin, weight_a, acts, lpo, lpr, adv, wts, 0.7, 0.20, 0.28, 0.04, 0.01
                    ),
                )
            )
        mode_defs.append(
            (
                "chunked_pytorch_grpo_c1024",
                lambda xin: chunked_pytorch_grpo_loss(
                    xin, weight_a, acts, lpo, lpr, adv, wts, 0.7, 0.20, 0.28, 0.04, 0.01, chunk_sz
                ),
            )
        )
        mode_defs.append(
            (
                "triton_fused_linear_grpo_bf16dw",
                lambda xin: _FusedLinearGRPOFn.apply(
                    xin,
                    weight_a,
                    acts,
                    lpo,
                    lpr,
                    adv,
                    wts,
                    0.7,
                    0.20,
                    0.28,
                    0.04,
                    0.01,
                    chunk_sz,
                    False,
                ),
            )
        )
        mode_defs.append(
            (
                "triton_fused_linear_grpo_dapo",
                lambda xin: _FusedLinearGRPOFn.apply(
                    xin,
                    weight_a,
                    acts,
                    lpo,
                    lpr,
                    adv,
                    wts,
                    0.7,
                    0.20,
                    0.28,
                    0.04,
                    0.01,
                    chunk_sz,
                    True,
                ),
            )
        )

        for mk, fn_m in mode_defs:
            xr = x_base.detach().clone().requires_grad_(True)

            def _reset_g() -> None:
                weight_a.grad = None
                xr.grad = None

            def _run_m(f_call: Any = fn_m) -> None:
                weight_a.grad = None
                xr.grad = None
                loss = f_call(xr)
                loss.backward()

            tm = time_fn_with_peak_mem(
                _run_m, warmup=warmup_a, iters=iters_a, device=device, reset_grads_fn=_reset_g
            )
            tm["effective_3gemm_tflops"] = round(
                (gemm3_flops / (tm["median_ms"] * 1e-3)) / 1e12, 2
            )
            modes_n[mk] = tm
            del xr
            weight_a.grad = None
            gc.collect()
            torch.cuda.empty_cache()

        grpo_scaling[str(N)] = {
            "num_tokens": N,
            "hidden": H_a,
            "vocab_size": V_a,
            "chunk_size": chunk_sz,
            "modes": modes_n,
        }
        del x_base, acts, lpo, lpr, adv, wts
        gc.collect()
        torch.cuda.empty_cache()

    dpo_scaling: dict[str, Any] = {}
    for N in dpo_tokens:
        B_pair = 8
        tok_per_pair = N // B_pair
        half_p = tok_per_pair // 2
        x_base = torch.randn((N, H_a), device=device, dtype=torch.bfloat16)
        acts = torch.randint(0, V_a, (N,), device=device)
        pair_id = torch.repeat_interleave(torch.arange(B_pair, device=device), tok_per_pair)
        sign = torch.tensor(([1.0] * half_p + [-1.0] * half_p) * B_pair, device=device, dtype=torch.float32)
        ref_m = torch.randn((B_pair,), device=device, dtype=torch.float32)

        modes_d: dict[str, Any] = {}
        for mk, fn_d in (
            (
                "eager_fp32_dpo_materialized",
                lambda xin: eager_materialized_dpo_loss(
                    xin, weight_a, acts, pair_id, sign, ref_m, 0.1, 0.7
                ),
            ),
            (
                "torch_compile_dpo_materialized",
                lambda xin: compiled_dpo(
                    xin, weight_a, acts, pair_id, sign, ref_m, 0.1, 0.7
                ),
            ),
            (
                "triton_saved_lse_linear_dpo",
                lambda xin: _FusedSavedLSELinearDPOFn.apply(
                    xin, weight_a, acts, pair_id, sign, ref_m, 0.1, 0.7, chunk_sz
                ),
            ),
        ):
            xr = x_base.detach().clone().requires_grad_(True)

            def _reset_d() -> None:
                weight_a.grad = None
                xr.grad = None

            def _run_d(f_call: Any = fn_d) -> None:
                weight_a.grad = None
                xr.grad = None
                loss = f_call(xr)
                loss.backward()

            modes_d[mk] = time_fn_with_peak_mem(
                _run_d, warmup=warmup_a, iters=iters_a, device=device, reset_grads_fn=_reset_d
            )
            del xr
            weight_a.grad = None
            gc.collect()
            torch.cuda.empty_cache()

        dpo_scaling[str(N)] = {
            "num_tokens": N,
            "num_pairs": B_pair,
            "hidden": H_a,
            "vocab_size": V_a,
            "modes": modes_d,
        }
        del x_base, acts, pair_id, sign, ref_m
        gc.collect()
        torch.cuda.empty_cache()

    del weight_a
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "grpo_dapo_vocab_scaling": grpo_scaling,
        "dpo_vocab_scaling": dpo_scaling,
    }


# ============================================================================
# 6. Suite B: 4-Layer Group-Prefix-Tree GRPO Stack (G = 4, 8, 16)
# ============================================================================


def _make_skewed_cot_lengths(G: int, prompt_idx: int) -> list[int]:
    """Deterministic skewed CoT rollout lengths (non-multiples of 64) with mean ~395, max ~1,437."""
    base8 = [138, 195, 258, 326, 412, 524, 782, 1437]
    shift = 11 * prompt_idx
    if G == 4:
        return [base8[0] + shift, base8[2] + shift, base8[5] + shift, base8[7] - shift]
    if G == 8:
        return [x + (shift if i % 2 == 0 else -shift) for i, x in enumerate(base8)]
    if G == 16:
        out16 = []
        for i, x in enumerate(base8):
            out16.append(x + shift + 7 * (i % 3))
            out16.append(max(96, x - shift - 5 * (i % 4)))
        return out16
    raise ValueError(G)


def run_suite_b_prefix_tree_grpo_stack(
    device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    H_b, I_b, nq_b, nkv_b = (2048, 5632, 16, 4) if is_smoke else (4096, 14336, 32, 8)
    V_b = 64000 if is_smoke else 128256
    L_prompt = 1024 if is_smoke else 1536
    g_list = [4, 8] if is_smoke else [4, 8, 16]
    B_p = 2
    n_layers = 4
    warmup_b = 2 if is_smoke else 3
    iters_b = 4 if is_smoke else 8

    model_b = RLVRModel(n_layers, H_b, I_b, nq_b, nkv_b, V_b, head_dim=128).to(
        device=device, dtype=torch.bfloat16
    )
    compiled_flex = torch.compile(flex_attention, mode="default")

    group_results: dict[str, Any] = {}
    for G in g_list:
        trees = [(L_prompt, _make_skewed_cot_lengths(G, b)) for b in range(B_p)]
        meta = build_prefix_tree_pack_metadata(trees, device)
        T_tree = meta["T_total"]
        T_resp = meta["T_resp"]
        T_pad128 = meta["T_pad128"]

        # Count tokens processed under each batching paradigm
        L_max_pad = max(lp + max(rls) for lp, rls in trees)
        T_padded_dense = B_p * G * L_max_pad
        T_dup_ragged = sum(G * lp + sum(rls) for lp, rls in trees)

        # Shared synthetic embeddings & rollout targets
        acts = torch.randint(0, V_b, (T_resp,), device=device)
        lpo = torch.randn((T_resp,), device=device, dtype=torch.float32) - 11.4
        lpr = lpo + 0.04 * torch.randn((T_resp,), device=device, dtype=torch.float32)
        adv = torch.randn((T_resp,), device=device, dtype=torch.float32)
        wts = torch.full((T_resp,), 1.0 / float(T_resp), device=device, dtype=torch.float32)

        modes_g: dict[str, Any] = {}

        # Mode 1: Padded Dense `[B_p * G, L_max, H]` (`TRL` default: pads & runs LM head over padded batch)
        x_pad_in = torch.randn(
            (B_p * G, L_max_pad, H_b), device=device, dtype=torch.bfloat16, requires_grad=True
        )
        cos_pad, sin_pad = build_rope_cache_1d(torch.arange(L_max_pad, device=device), 128)
        acts_pad = torch.randint(0, V_b, (T_padded_dense,), device=device)
        lpo_pad = torch.zeros((T_padded_dense,), device=device, dtype=torch.float32) - 11.4
        lpr_pad = lpo_pad.clone()
        adv_pad = torch.randn((T_padded_dense,), device=device, dtype=torch.float32)
        wts_pad = torch.zeros((T_padded_dense,), device=device, dtype=torch.float32)
        seq_cursor = 0
        for lp, rls in trees:
            for rl in rls:
                base_off = seq_cursor * L_max_pad
                wts_pad[base_off + lp : base_off + lp + rl] = 1.0 / float(T_resp)
                seq_cursor += 1

        def _run_padded() -> None:
            for p in model_b.parameters():
                p.grad = None
            x_pad_in.grad = None
            cur_x = x_pad_in
            for blk in model_b.blocks:
                cur_x = blk.forward_padded_fa2(cur_x, cos_pad, sin_pad)
            loss = model_b.lm_head(
                cur_x.view(T_padded_dense, H_b),
                None,
                acts_pad,
                lpo_pad,
                lpr_pad,
                adv_pad,
                wts_pad,
                chunk_size=1024,
            )
            loss.backward()

        modes_g["grpo_padded_dense_fa2"] = {
            **time_fn_with_peak_mem(_run_padded, warmup=warmup_b, iters=iters_b, device=device),
            "backbone_tokens": T_padded_dense,
        }
        del x_pad_in, acts_pad, lpo_pad, lpr_pad, adv_pad, wts_pad
        gc.collect()
        torch.cuda.empty_cache()

        # Mode 2: Ragged Duplicated-Prompt `flash_attn_varlen_func` (`veRL` / `OpenRLHF` `remove_padding`)
        cu_list = [0]
        dup_pos_l: list[int] = []
        dup_resp_l: list[int] = []
        cur_c = 0
        for lp, rls in trees:
            for rl in rls:
                dup_pos_l.extend(range(lp + rl))
                dup_resp_l.extend(range(cur_c + lp, cur_c + lp + rl))
                cur_c += lp + rl
                cu_list.append(cur_c)
        cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=device)
        cos_dup, sin_dup = build_rope_cache_1d(
            torch.tensor(dup_pos_l, dtype=torch.long, device=device), 128
        )
        dup_resp_idx = torch.tensor(dup_resp_l, dtype=torch.long, device=device)
        x_dup_in = torch.randn(
            (T_dup_ragged, H_b), device=device, dtype=torch.bfloat16, requires_grad=True
        )

        def _run_varlen_dup() -> None:
            for p in model_b.parameters():
                p.grad = None
            x_dup_in.grad = None
            cur_x = x_dup_in
            for blk in model_b.blocks:
                cur_x = blk.forward_varlen_fa2(cur_x, cos_dup, sin_dup, cu_seqlens, L_max_pad)
            loss = model_b.lm_head(
                cur_x, dup_resp_idx, acts, lpo, lpr, adv, wts, chunk_size=1024
            )
            loss.backward()

        modes_g["grpo_ragged_duplicated_prompt_fa2"] = {
            **time_fn_with_peak_mem(_run_varlen_dup, warmup=warmup_b, iters=iters_b, device=device),
            "backbone_tokens": T_dup_ragged,
        }
        del x_dup_in
        gc.collect()
        torch.cuda.empty_cache()

        # Shared Prefix-Tree input tensor for Modes 3 & 4
        cos_tr, sin_tr = build_rope_cache_1d(meta["pos_ids"], 128)

        # Mode 3: Prefix-Tree `FlexAttention` `BlockMask` (`enable_gqa=True`)
        doc_buf = meta["doc_id_pad"]
        br_buf = meta["branch_id_pad"]

        def _tree_mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
            dq = doc_buf[q_idx]
            dk = doc_buf[kv_idx]
            bq = br_buf[q_idx]
            bk = br_buf[kv_idx]
            return (dq >= 0) & (dq == dk) & ((bk == 0) | (bq == bk)) & (q_idx >= kv_idx)

        t_bm0 = time.perf_counter()
        bm_tree = create_block_mask(
            _tree_mask_mod, B=None, H=None, Q_LEN=T_pad128, KV_LEN=T_pad128, device=device
        )
        x_flex_in = torch.randn(
            (T_tree, H_b), device=device, dtype=torch.bfloat16, requires_grad=True
        )

        def _run_tree_flex() -> None:
            for p in model_b.parameters():
                p.grad = None
            x_flex_in.grad = None
            cur_x = x_flex_in
            for blk in model_b.blocks:
                cur_x = blk.forward_prefix_tree_flex(
                    cur_x, cos_tr, sin_tr, T_pad128, bm_tree, compiled_flex
                )
            loss = model_b.lm_head(
                cur_x, meta["resp_idx"], acts, lpo, lpr, adv, wts, chunk_size=1024
            )
            loss.backward()

        _run_tree_flex()
        torch.cuda.synchronize(device)
        flex_first_s = round(time.perf_counter() - t_bm0, 2)
        modes_g["grpo_prefix_tree_flex_blockmask"] = {
            **time_fn_with_peak_mem(_run_tree_flex, warmup=warmup_b, iters=iters_b, device=device),
            "backbone_tokens": T_tree,
            "first_call_compile_s": flex_first_s,
        }
        del x_flex_in, bm_tree
        gc.collect()
        torch.cuda.empty_cache()

        # Mode 4: Ours — Prefix-Tree Two-Slice Ragged Triton GQA + Active-Response FusedLinearGRPO
        x_tri_in = torch.randn(
            (T_tree, H_b), device=device, dtype=torch.bfloat16, requires_grad=True
        )

        def _run_tree_triton() -> None:
            for p in model_b.parameters():
                p.grad = None
            x_tri_in.grad = None
            cur_x = x_tri_in
            for blk in model_b.blocks:
                cur_x = blk.forward_prefix_tree_triton(
                    cur_x, cos_tr, sin_tr, meta["tile_meta"]
                )
            loss = model_b.lm_head(
                cur_x, meta["resp_idx"], acts, lpo, lpr, adv, wts, chunk_size=1024
            )
            loss.backward()

        modes_g["grpo_prefix_tree_triton_gqa"] = {
            **time_fn_with_peak_mem(_run_tree_triton, warmup=warmup_b, iters=iters_b, device=device),
            "backbone_tokens": T_tree,
        }
        del x_tri_in
        gc.collect()
        torch.cuda.empty_cache()

        group_results[str(G)] = {
            "group_size": G,
            "num_prompts": B_p,
            "prompt_len": L_prompt,
            "response_tokens": T_resp,
            "padded_tokens": T_padded_dense,
            "duplicated_ragged_tokens": T_dup_ragged,
            "prefix_tree_tokens": T_tree,
            "token_compression_vs_duplicated": round(T_dup_ragged / float(T_tree), 4),
            "modes": modes_g,
        }

    del model_b
    gc.collect()
    torch.cuda.empty_cache()
    return {"group_size_scaling": group_results}


# ============================================================================
# 7. Suite C: 8-GPU FSDP2 Long-CoT RLVR Straggler Balancing & Global Norm Parity
# ============================================================================


def _wrap_rlvr_model_fsdp2(model: RLVRModel) -> RLVRModel:
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    for blk in model.blocks:
        fully_shard(blk, mp_policy=mp)
    fully_shard(model.lm_head, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model


def _tree_quadratic_cost(lp: int, r_lens: list[int], hidden: int) -> float:
    """Analytical Transformer + Attention FLOP proxy: linear backbone FLOPs + quadratic tree attention pairs."""
    t_tok = lp + sum(r_lens)
    attn_pairs = (lp * lp) // 2 + sum(lp * rl + (rl * rl) // 2 for rl in r_lens)
    # Linear FLOPs per token ~ 24 * H^2 (fwd+bwd ~ 72 * H^2); Attention FLOPs per pair ~ 12 * H
    return float(t_tok) * (6.0 * float(hidden)) + float(attn_pairs)


def _lpt_partition_items(
    items: list[tuple[float, Any]], world_size: int
) -> list[list[Any]]:
    """Longest-Processing-Time (LPT) min-heap bin packing across `world_size` ranks."""
    bins: list[list[Any]] = [[] for _ in range(world_size)]
    loads = [0.0 for _ in range(world_size)]
    for cost, payload in sorted(items, key=lambda x: x[0], reverse=True):
        # Ensure every rank gets at least 1 item first before stacking
        empty_bins = [r for r in range(world_size) if len(bins[r]) == 0]
        target = empty_bins[0] if empty_bins else min(range(world_size), key=lambda r: loads[r])
        bins[target].append(payload)
        loads[target] += cost
    return bins


def run_suite_c_fsdp2_rlvr(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    # ------------------------------------------------------------------------
    # C.1 Exact 8-GPU FSDP2 vs 1-GPU Global DAPO Parameter-Gradient Parity
    #     AND Proof of Naive Local-Token-Mean (`1 / N_resp_r`) Gradient Bias!
    # ------------------------------------------------------------------------
    torch.manual_seed(12170)
    H_p, I_p, nq_p, nkv_p, V_p = 1024, 2816, 8, 2, 32000
    m_ref = RLVRModel(2, H_p, I_p, nq_p, nkv_p, V_p, head_dim=128).to(
        device=device, dtype=torch.bfloat16
    )
    sd_cpu = {k: v.detach().cpu().clone() for k, v in m_ref.state_dict().items()}

    # 8 prompts x G=4 rollouts with severe prompt-correlated CoT skew:
    # Prompts 0, 1 are hard long-CoT (~650-920 tok), Prompts 2..7 are concise (~80-170 tok)!
    parity_specs = [
        (256, [680, 790, 850, 910]),
        (256, [640, 720, 810, 880]),
        (256, [90, 115, 130, 155]),
        (256, [85, 105, 125, 145]),
        (256, [95, 110, 140, 165]),
        (256, [80, 100, 120, 150]),
        (256, [88, 108, 128, 158]),
        (256, [92, 112, 132, 162]),
    ]
    gen_c = torch.Generator(device=device)
    gen_c.manual_seed(12175)

    prompt_data: list[dict[str, Any]] = []
    N_resp_global = sum(sum(rls) for _, rls in parity_specs)
    for lp, rls in parity_specs:
        xp = torch.randn((lp, H_p), generator=gen_c, device=device, dtype=torch.bfloat16)
        r_items = []
        for rl in rls:
            xr = torch.randn((rl, H_p), generator=gen_c, device=device, dtype=torch.bfloat16)
            ac = torch.randint(0, V_p, (rl,), generator=gen_c, device=device)
            av = torch.randn((rl,), generator=gen_c, device=device, dtype=torch.float32)
            r_items.append({"rl": rl, "xr": xr, "act": ac, "adv": av})
        prompt_data.append({"lp": lp, "xp": xp, "rollouts": r_items})

    # Compute on-policy + away-from-kink `lpo` / `lpr` from initial model forward!
    ref_trees = [(d["lp"], [r["rl"] for r in d["rollouts"]]) for d in prompt_data]
    meta_1g = build_prefix_tree_pack_metadata(ref_trees, device)
    cos_1g, sin_1g = build_rope_cache_1d(meta_1g["pos_ids"], 128)
    act_1g = torch.cat([r["act"] for d in prompt_data for r in d["rollouts"]], dim=0)
    with torch.no_grad():
        x_init_1g = torch.cat(
            [t for d in prompt_data for t in ([d["xp"]] + [r["xr"] for r in d["rollouts"]])],
            dim=0,
        )
        for blk in m_ref.blocks:
            x_init_1g = blk.forward_prefix_tree_triton(
                x_init_1g, cos_1g, sin_1g, meta_1g["tile_meta"]
            )
        h_r_1g = m_ref.lm_head.final_norm(x_init_1g.index_select(0, meta_1g["resp_idx"]))
        lp_init_1g = (
            F.log_softmax(F.linear(h_r_1g, m_ref.lm_head.weight).float() / 0.7, dim=-1)
            .gather(1, act_1g.unsqueeze(1))
            .squeeze(1)
        )
        cur_off = 0
        for d in prompt_data:
            for g_i, r in enumerate(d["rollouts"]):
                rl = r["rl"]
                lp_sl = lp_init_1g[cur_off : cur_off + rl]
                if g_i == 3:
                    r["lpo"] = lp_sl - 0.45
                    r["adv"] = torch.abs(r["adv"]) + 0.5
                else:
                    r["lpo"] = lp_sl + (torch.rand((rl,), generator=gen_c, device=device) - 0.5) * 0.10
                r["lpr"] = lp_sl + 0.03 * torch.randn((rl,), generator=gen_c, device=device, dtype=torch.float32)
                cur_off += rl

    # 1-GPU exact global reference (computed identically on every rank to compare against FSDP2 full_tensor!)
    for p in m_ref.parameters():
        p.grad = None
    x_1g = torch.cat(
        [t for d in prompt_data for t in ([d["xp"]] + [r["xr"] for r in d["rollouts"]])],
        dim=0,
    ).detach().requires_grad_(True)
    lpo_1g = torch.cat([r["lpo"] for d in prompt_data for r in d["rollouts"]], dim=0)
    lpr_1g = torch.cat([r["lpr"] for d in prompt_data for r in d["rollouts"]], dim=0)
    adv_1g = torch.cat([r["adv"] for d in prompt_data for r in d["rollouts"]], dim=0)
    wt_1g = torch.full(
        (N_resp_global,), 1.0 / float(N_resp_global), device=device, dtype=torch.float32
    )
    cur_1g = x_1g
    for blk in m_ref.blocks:
        cur_1g = blk.forward_prefix_tree_triton(cur_1g, cos_1g, sin_1g, meta_1g["tile_meta"])
    loss_1g = m_ref.lm_head(
        cur_1g, meta_1g["resp_idx"], act_1g, lpo_1g, lpr_1g, adv_1g, wt_1g, chunk_size=512
    )
    loss_1g.backward()
    ref_wq_grad = m_ref.blocks[0].w_q.weight.grad.detach().clone()
    ref_wdown_grad = m_ref.blocks[1].w_down.weight.grad.detach().clone()
    ref_lm_grad = m_ref.lm_head.weight.grad.detach().clone()
    del m_ref, x_1g, cur_1g

    def _eval_fsdp2_parity(mode_kind: str) -> dict[str, float]:
        m_f = RLVRModel(2, H_p, I_p, nq_p, nkv_p, V_p, head_dim=128).to(
            device=device, dtype=torch.bfloat16
        )
        m_f.load_state_dict({k: v.to(device) for k, v in sd_cpu.items()})
        m_f = _wrap_rlvr_model_fsdp2(m_f)

        if mode_kind in ("naive_local_token_mean", "prompt_sharded_varlen_global_norm"):
            # Prompt-sharded: rank r owns prompt_data[r]
            d_loc = [prompt_data[rank]]
            cu_l = [0]
            pos_l: list[int] = []
            resp_l: list[int] = []
            x_parts: list[torch.Tensor] = []
            cur_i = 0
            for d in d_loc:
                lp = d["lp"]
                for r in d["rollouts"]:
                    rl = r["rl"]
                    x_parts.append(d["xp"])
                    x_parts.append(r["xr"])
                    pos_l.extend(range(lp + rl))
                    resp_l.extend(range(cur_i + lp, cur_i + lp + rl))
                    cur_i += lp + rl
                    cu_l.append(cur_i)
            x_loc = torch.cat(x_parts, dim=0).detach().requires_grad_(True)
            act_loc = torch.cat([r["act"] for d in d_loc for r in d["rollouts"]], dim=0)
            lpo_loc = torch.cat([r["lpo"] for d in d_loc for r in d["rollouts"]], dim=0)
            lpr_loc = torch.cat([r["lpr"] for d in d_loc for r in d["rollouts"]], dim=0)
            adv_loc = torch.cat([r["adv"] for d in d_loc for r in d["rollouts"]], dim=0)
            n_loc = act_loc.numel()
            # Crucial difference: FSDP2 averages gradients by 1 / world_size!
            # Naive local-mean uses 1 / n_loc -> effective global weight 1 / (8 * n_loc) (severely biased!)
            # Exact global DAPO norm uses world_size / N_resp_global -> effective global weight 1 / N_resp_global!
            w_val = (
                (1.0 / float(n_loc))
                if mode_kind == "naive_local_token_mean"
                else (float(world_size) / float(N_resp_global))
            )
            wt_loc = torch.full((n_loc,), w_val, device=device, dtype=torch.float32)
            cos_l, sin_l = build_rope_cache_1d(
                torch.tensor(pos_l, dtype=torch.long, device=device), 128
            )
            cu_t = torch.tensor(cu_l, dtype=torch.int32, device=device)
            max_sl = max(d["lp"] + max(r["rl"] for r in d["rollouts"]) for d in d_loc)
            loss_loc = m_f(
                x_loc,
                "varlen_fa2",
                cos_l,
                sin_l,
                torch.tensor(resp_l, dtype=torch.long, device=device),
                act_loc,
                lpo_loc,
                lpr_loc,
                adv_loc,
                wt_loc,
                cu_seqlens=cu_t,
                max_seqlen=max_sl,
                chunk_size=512,
            )
        elif mode_kind == "cost_lpt_prefix_tree_global_norm":
            # Split hard prompts (sum(rl) > 1600) into 2 sub-trees of G=2 and pack via quadratic-cost LPT!
            sub_trees_all: list[tuple[float, dict[str, Any]]] = []
            for d in prompt_data:
                lp = d["lp"]
                rls_all = d["rollouts"]
                if sum(r["rl"] for r in rls_all) > 1600:
                    groups = [rls_all[:2], rls_all[2:]]
                else:
                    groups = [rls_all]
                for grp in groups:
                    c_est = _tree_quadratic_cost(lp, [r["rl"] for r in grp], H_p)
                    sub_trees_all.append((c_est, {"lp": lp, "xp": d["xp"], "rollouts": grp}))
            bins = _lpt_partition_items(sub_trees_all, world_size)
            my_trees = bins[rank]
            t_shapes = [(d["lp"], [r["rl"] for r in d["rollouts"]]) for d in my_trees]
            meta_l = build_prefix_tree_pack_metadata(t_shapes, device)
            x_loc = torch.cat(
                [t for d in my_trees for t in ([d["xp"]] + [r["xr"] for r in d["rollouts"]])],
                dim=0,
            ).detach().requires_grad_(True)
            act_loc = torch.cat([r["act"] for d in my_trees for r in d["rollouts"]], dim=0)
            lpo_loc = torch.cat([r["lpo"] for d in my_trees for r in d["rollouts"]], dim=0)
            lpr_loc = torch.cat([r["lpr"] for d in my_trees for r in d["rollouts"]], dim=0)
            adv_loc = torch.cat([r["adv"] for d in my_trees for r in d["rollouts"]], dim=0)
            wt_loc = torch.full(
                (act_loc.numel(),),
                float(world_size) / float(N_resp_global),
                device=device,
                dtype=torch.float32,
            )
            cos_l, sin_l = build_rope_cache_1d(meta_l["pos_ids"], 128)
            loss_loc = m_f(
                x_loc,
                "prefix_tree_triton",
                cos_l,
                sin_l,
                meta_l["resp_idx"],
                act_loc,
                lpo_loc,
                lpr_loc,
                adv_loc,
                wt_loc,
                tile_meta=meta_l["tile_meta"],
                chunk_size=512,
            )
        else:
            raise ValueError(mode_kind)

        loss_loc.backward()
        wq_full = m_f.blocks[0].w_q.weight.grad.full_tensor()
        wd_full = m_f.blocks[1].w_down.weight.grad.full_tensor()
        wlm_full = m_f.lm_head.weight.grad.full_tensor()
        l_avg = loss_loc.detach().double() / float(world_size)
        dist.all_reduce(l_avg, op=dist.ReduceOp.SUM)

        res = {
            "global_loss_rel_err": round(
                abs(float(l_avg.item()) - float(loss_1g.item()))
                / max(abs(float(loss_1g.item())), 1e-12),
                6,
            ),
            "dw_layer0_wq_rel_l2": round(rel_l2(wq_full, ref_wq_grad), 6),
            "dw_layer1_wdown_rel_l2": round(rel_l2(wd_full, ref_wdown_grad), 6),
            "dw_lm_head_rel_l2": round(rel_l2(wlm_full, ref_lm_grad), 6),
        }
        grads_keep = (
            float(l_avg.item()),
            wq_full.detach().clone(),
            wd_full.detach().clone(),
            wlm_full.detach().clone(),
        )
        del m_f, wq_full, wd_full, wlm_full
        gc.collect()
        torch.cuda.empty_cache()
        return res, grads_keep

    naive_res, _ = _eval_fsdp2_parity("naive_local_token_mean")
    varlen_res, varlen_g = _eval_fsdp2_parity("prompt_sharded_varlen_global_norm")
    lpt_res, lpt_g = _eval_fsdp2_parity("cost_lpt_prefix_tree_global_norm")

    parity_c1 = {
        "naive_local_token_mean_bias_control": naive_res,
        "prompt_sharded_varlen_global_norm": varlen_res,
        "cost_lpt_prefix_tree_global_norm": lpt_res,
        "cost_lpt_vs_prompt_varlen_fsdp2_parity": {
            "global_loss_rel_err": round(
                abs(lpt_g[0] - varlen_g[0]) / max(abs(varlen_g[0]), 1e-12), 6
            ),
            "dw_layer0_wq_rel_l2": round(rel_l2(lpt_g[1], varlen_g[1]), 6),
            "dw_layer1_wdown_rel_l2": round(rel_l2(lpt_g[2], varlen_g[2]), 6),
            "dw_lm_head_rel_l2": round(rel_l2(lpt_g[3], varlen_g[3]), 6),
        },
    }
    del varlen_g, lpt_g

    # ------------------------------------------------------------------------
    # C.2 8-GPU FSDP2 Long-CoT RLVR Step Scaling (16 Prompts x G=8 = 128 Rollouts)
    # ------------------------------------------------------------------------
    H_s, I_s, nq_s, nkv_s, V_s = (
        (2048, 5632, 16, 4, 64000) if is_smoke else (4096, 14336, 32, 8, 128256)
    )
    Lp_s = 640 if is_smoke else 1280
    scale_len = 0.5 if is_smoke else 1.0
    warmup_c = 2 if is_smoke else 3
    iters_c = 4 if is_smoke else 8

    # 16 prompts x G=8 rollouts: prompts 0..2 have bimodal hard math CoTs (3 ultra-long ~3600..4300 + 5 medium),
    # prompts 3..7 medium (~620..1050), prompts 8..15 concise (~150..300)
    raw_specs_16: list[tuple[int, list[int]]] = []
    for p_id in range(16):
        if p_id < 3:
            base_r = [420, 480, 540, 610, 680, 3620, 3940, 4280]
        elif p_id < 8:
            base_r = [620, 680, 740, 800, 860, 920, 980, 1050]
        else:
            base_r = [150, 170, 190, 210, 230, 250, 275, 300]
        rls_scaled = [max(96, int((x + 9 * (p_id % 4)) * scale_len)) for x in base_r]
        raw_specs_16.append((Lp_s, rls_scaled))

    total_resp_tok = sum(sum(rls) for _, rls in raw_specs_16)

    torch.manual_seed(12185)
    m_scale = RLVRModel(4, H_s, I_s, nq_s, nkv_s, V_s, head_dim=128).to(
        device=device, dtype=torch.bfloat16
    )
    n_params_b = round(sum(p.numel() for p in m_scale.parameters()) / 1e9, 3)
    m_scale = _wrap_rlvr_model_fsdp2(m_scale)

    def _bench_fsdp2_mode(
        step_fn: Callable[[], None], my_tok: int, my_cost: float
    ) -> dict[str, Any]:
        for _ in range(warmup_c):
            step_fn()
        torch.cuda.synchronize(device)
        dist.barrier()

        base_m = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_fn()
        torch.cuda.synchronize(device)
        inc_m = max(0.0, (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0**2))
        max_m = torch.tensor([inc_m], device=device, dtype=torch.float64)
        dist.all_reduce(max_m, op=dist.ReduceOp.MAX)

        tok_vec = [torch.zeros(1, device=device, dtype=torch.float64) for _ in range(world_size)]
        cost_vec = [torch.zeros(1, device=device, dtype=torch.float64) for _ in range(world_size)]
        dist.all_gather(tok_vec, torch.tensor([float(my_tok)], device=device, dtype=torch.float64))
        dist.all_gather(cost_vec, torch.tensor([float(my_cost)], device=device, dtype=torch.float64))
        toks_all = [float(t.item()) for t in tok_vec]
        costs_all = [float(c.item()) for c in cost_vec]

        samples_ms: list[float] = []
        s_e = torch.cuda.Event(enable_timing=True)
        e_e = torch.cuda.Event(enable_timing=True)
        for _ in range(iters_c):
            dist.barrier()
            s_e.record()
            step_fn()
            e_e.record()
            torch.cuda.synchronize(device)
            dur = torch.tensor([float(s_e.elapsed_time(e_e))], device=device, dtype=torch.float64)
            dist.all_reduce(dur, op=dist.ReduceOp.MAX)
            samples_ms.append(float(dur.item()))
        ss = sorted(samples_ms)
        q25 = ss[max(0, (len(ss) - 1) // 4)]
        q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
        med_ms = float(statistics.median(ss))
        return {
            "step_ms_median": round(med_ms, 4),
            "step_ms_iqr": round(float(q75 - q25), 4),
            "per_rank_peak_inc_mib": round(float(max_m.item()), 2),
            "cluster_backbone_tokens": int(round(sum(toks_all))),
            "rank_token_max_over_min": round(max(toks_all) / max(min(toks_all), 1.0), 3),
            "rank_cost_max_over_min": round(max(costs_all) / max(min(costs_all), 1.0), 3),
            "response_ktok_per_s": round((total_resp_tok / (med_ms * 1e-3)) / 1e3, 2),
        }

    fsdp2_modes: dict[str, Any] = {}

    # 1. `fsdp2_prompt_sharded_padded_fa2`: rank r gets prompts `[2*r, 2*r+1]` (`16` rollouts), padded to local `L_max`
    my_p2 = raw_specs_16[2 * rank : 2 * rank + 2]
    lmax_loc = max(lp + max(rls) for lp, rls in my_p2)
    B_loc = 16
    T_pad_loc = B_loc * lmax_loc
    cost_pad_loc = float(B_loc) * _tree_quadratic_cost(0, [lmax_loc], H_s)
    x_m1 = torch.randn((B_loc, lmax_loc, H_s), device=device, dtype=torch.bfloat16, requires_grad=True)
    cos_m1, sin_m1 = build_rope_cache_1d(torch.arange(lmax_loc, device=device), 128)
    act_m1 = torch.randint(0, V_s, (T_pad_loc,), device=device)
    lpo_m1 = torch.zeros((T_pad_loc,), device=device, dtype=torch.float32) - 11.4
    lpr_m1 = lpo_m1.clone()
    adv_m1 = torch.randn((T_pad_loc,), device=device, dtype=torch.float32)
    wt_m1 = torch.zeros((T_pad_loc,), device=device, dtype=torch.float32)
    sc_i = 0
    for lp, rls in my_p2:
        for rl in rls:
            wt_m1[sc_i * lmax_loc + lp : sc_i * lmax_loc + lp + rl] = (
                float(world_size) / float(total_resp_tok)
            )
            sc_i += 1

    def _f_m1() -> None:
        for p in m_scale.parameters():
            p.grad = None
        x_m1.grad = None
        loss = m_scale(
            x_m1, "padded_fa2", cos_m1, sin_m1, None, act_m1, lpo_m1, lpr_m1, adv_m1, wt_m1, chunk_size=1024
        )
        loss.backward()

    fsdp2_modes["fsdp2_prompt_sharded_padded_fa2"] = _bench_fsdp2_mode(
        _f_m1, T_pad_loc, cost_pad_loc
    )
    del x_m1, act_m1, lpo_m1, lpr_m1, adv_m1, wt_m1
    gc.collect()
    torch.cuda.empty_cache()

    # 2. `fsdp2_prompt_sharded_ragged_fa2`: rank r gets prompts `[2*r, 2*r+1]` with varlen FA2 (un-rebalanced straggler!)
    cu_m2 = [0]
    pos_m2: list[int] = []
    resp_m2: list[int] = []
    cur2 = 0
    cost_m2 = 0.0
    for lp, rls in my_p2:
        for rl in rls:
            pos_m2.extend(range(lp + rl))
            resp_m2.extend(range(cur2 + lp, cur2 + lp + rl))
            cur2 += lp + rl
            cu_m2.append(cur2)
            cost_m2 += _tree_quadratic_cost(0, [lp + rl], H_s)
    T_m2 = cur2
    n_resp_m2 = len(resp_m2)
    x_m2 = torch.randn((T_m2, H_s), device=device, dtype=torch.bfloat16, requires_grad=True)
    cos_m2, sin_m2 = build_rope_cache_1d(torch.tensor(pos_m2, dtype=torch.long, device=device), 128)
    cu_t2 = torch.tensor(cu_m2, dtype=torch.int32, device=device)
    resp_t2 = torch.tensor(resp_m2, dtype=torch.long, device=device)
    act_m2 = torch.randint(0, V_s, (n_resp_m2,), device=device)
    lpo_m2 = torch.randn((n_resp_m2,), device=device, dtype=torch.float32) - 11.4
    lpr_m2 = lpo_m2.clone()
    adv_m2 = torch.randn((n_resp_m2,), device=device, dtype=torch.float32)
    wt_m2 = torch.full(
        (n_resp_m2,), float(world_size) / float(total_resp_tok), device=device, dtype=torch.float32
    )

    def _f_m2() -> None:
        for p in m_scale.parameters():
            p.grad = None
        x_m2.grad = None
        loss = m_scale(
            x_m2,
            "varlen_fa2",
            cos_m2,
            sin_m2,
            resp_t2,
            act_m2,
            lpo_m2,
            lpr_m2,
            adv_m2,
            wt_m2,
            cu_seqlens=cu_t2,
            max_seqlen=lmax_loc,
            chunk_size=1024,
        )
        loss.backward()

    fsdp2_modes["fsdp2_prompt_sharded_ragged_fa2"] = _bench_fsdp2_mode(_f_m2, T_m2, cost_m2)
    del x_m2, act_m2, lpo_m2, lpr_m2, adv_m2, wt_m2
    gc.collect()
    torch.cuda.empty_cache()

    # 3. `fsdp2_token_greedy_ragged_fa2`: balance individual `(lp, rl)` trajectories across 8 ranks by token count `lp + rl`
    traj_items: list[tuple[float, tuple[int, int]]] = []
    for lp, rls in raw_specs_16:
        for rl in rls:
            traj_items.append((float(lp + rl), (lp, rl)))
    traj_bins = _lpt_partition_items(traj_items, world_size)
    my_trajs = traj_bins[rank]
    cu_m3 = [0]
    pos_m3: list[int] = []
    resp_m3: list[int] = []
    cur3 = 0
    cost_m3 = 0.0
    max_s3 = 0
    for lp, rl in my_trajs:
        pos_m3.extend(range(lp + rl))
        resp_m3.extend(range(cur3 + lp, cur3 + lp + rl))
        cur3 += lp + rl
        cu_m3.append(cur3)
        max_s3 = max(max_s3, lp + rl)
        cost_m3 += _tree_quadratic_cost(0, [lp + rl], H_s)
    T_m3 = cur3
    n_resp_m3 = len(resp_m3)
    x_m3 = torch.randn((T_m3, H_s), device=device, dtype=torch.bfloat16, requires_grad=True)
    cos_m3, sin_m3 = build_rope_cache_1d(torch.tensor(pos_m3, dtype=torch.long, device=device), 128)
    cu_t3 = torch.tensor(cu_m3, dtype=torch.int32, device=device)
    resp_t3 = torch.tensor(resp_m3, dtype=torch.long, device=device)
    act_m3 = torch.randint(0, V_s, (n_resp_m3,), device=device)
    lpo_m3 = torch.randn((n_resp_m3,), device=device, dtype=torch.float32) - 11.4
    lpr_m3 = lpo_m3.clone()
    adv_m3 = torch.randn((n_resp_m3,), device=device, dtype=torch.float32)
    wt_m3 = torch.full(
        (n_resp_m3,), float(world_size) / float(total_resp_tok), device=device, dtype=torch.float32
    )

    def _f_m3() -> None:
        for p in m_scale.parameters():
            p.grad = None
        x_m3.grad = None
        loss = m_scale(
            x_m3,
            "varlen_fa2",
            cos_m3,
            sin_m3,
            resp_t3,
            act_m3,
            lpo_m3,
            lpr_m3,
            adv_m3,
            wt_m3,
            cu_seqlens=cu_t3,
            max_seqlen=max_s3,
            chunk_size=1024,
        )
        loss.backward()

    fsdp2_modes["fsdp2_token_greedy_ragged_fa2"] = _bench_fsdp2_mode(_f_m3, T_m3, cost_m3)
    del x_m3, act_m3, lpo_m3, lpr_m3, adv_m3, wt_m3
    gc.collect()
    torch.cuda.empty_cache()

    # 4. `fsdp2_cost_lpt_prefix_tree_gqa` (Ours): split oversized hard trees (G=8 -> 2x4) + Quadratic-Cost LPT + Prefix-Tree Triton GQA
    tree_items_4: list[tuple[float, tuple[int, list[int]]]] = []
    thresh_split = int(12000 * scale_len)
    for lp, rls in raw_specs_16:
        if sum(rls) > thresh_split:
            # Interleave short/long within the 2 sub-trees of G=4 for balanced sub-trees!
            sub0 = rls[0::2]
            sub1 = rls[1::2]
            for sg in (sub0, sub1):
                tree_items_4.append((_tree_quadratic_cost(lp, sg, H_s), (lp, sg)))
        else:
            tree_items_4.append((_tree_quadratic_cost(lp, rls, H_s), (lp, rls)))
    tree_bins_4 = _lpt_partition_items(tree_items_4, world_size)
    my_trees_4 = tree_bins_4[rank]
    meta_m4 = build_prefix_tree_pack_metadata(my_trees_4, device)
    T_m4 = meta_m4["T_total"]
    n_resp_m4 = meta_m4["T_resp"]
    cost_m4 = sum(_tree_quadratic_cost(lp, rls, H_s) for lp, rls in my_trees_4)

    x_m4 = torch.randn((T_m4, H_s), device=device, dtype=torch.bfloat16, requires_grad=True)
    cos_m4, sin_m4 = build_rope_cache_1d(meta_m4["pos_ids"], 128)
    act_m4 = torch.randint(0, V_s, (n_resp_m4,), device=device)
    lpo_m4 = torch.randn((n_resp_m4,), device=device, dtype=torch.float32) - 11.4
    lpr_m4 = lpo_m4.clone()
    adv_m4 = torch.randn((n_resp_m4,), device=device, dtype=torch.float32)
    wt_m4 = torch.full(
        (n_resp_m4,), float(world_size) / float(total_resp_tok), device=device, dtype=torch.float32
    )

    def _f_m4() -> None:
        for p in m_scale.parameters():
            p.grad = None
        x_m4.grad = None
        loss = m_scale(
            x_m4,
            "prefix_tree_triton",
            cos_m4,
            sin_m4,
            meta_m4["resp_idx"],
            act_m4,
            lpo_m4,
            lpr_m4,
            adv_m4,
            wt_m4,
            tile_meta=meta_m4["tile_meta"],
            chunk_size=1024,
        )
        loss.backward()

    fsdp2_modes["fsdp2_cost_lpt_prefix_tree_gqa"] = _bench_fsdp2_mode(_f_m4, T_m4, cost_m4)
    del x_m4, act_m4, lpo_m4, lpr_m4, adv_m4, wt_m4, m_scale
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fsdp2_vs_1gpu_dapo_gradient_parity": parity_c1,
        "fsdp2_long_cot_rlvr_scaling": {
            "model_params_billions": n_params_b,
            "global_prompts": 16,
            "group_size": 8,
            "prompt_len": Lp_s,
            "global_response_tokens": total_resp_tok,
            "modes": fsdp2_modes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E121: RLVR / GRPO / DAPO / DPO Fused Terminal Kernels, Ragged Group-Prefix-Tree FlashAttention, & 8-GPU FSDP2 Long-CoT Training"
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
        print(f"[E121] Suite 0 complete at +{time.perf_counter() - t_start:.1f}s", flush=True)
        sa_res = run_suite_a_terminal_post_training_kernels(device, is_smoke)
        print(f"[E121] Suite A complete at +{time.perf_counter() - t_start:.1f}s", flush=True)
        sb_res = run_suite_b_prefix_tree_grpo_stack(device, is_smoke)
        print(f"[E121] Suite B complete at +{time.perf_counter() - t_start:.1f}s", flush=True)

    dist.barrier()
    sc_res = run_suite_c_fsdp2_rlvr(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = round(time.perf_counter() - t_start, 2)

    if rank == 0:
        print(f"[E121] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        env = environment_metadata()
        env["triton"] = getattr(triton, "__version__", "unknown")
        import flash_attn as fa_pkg

        env["flash_attn"] = getattr(fa_pkg, "__version__", "unknown")
        payload = {
            "experiment": "E121_rlvr_grpo_post_training",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": elapsed_s,
            "environment": env,
            "suite_0_audits": s0_res,
            "suite_a_terminal_post_training_kernels": sa_res,
            "suite_b_prefix_tree_grpo_stack": sb_res,
            "suite_c_fsdp2_rlvr": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
