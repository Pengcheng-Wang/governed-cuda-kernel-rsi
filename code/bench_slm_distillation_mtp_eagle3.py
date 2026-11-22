#!/usr/bin/env python3
"""E122: SLM Knowledge Distillation Coupled-Monoid Kernels (Forward KL / Reverse KL / JSD),
DeepSeek-V3 Multi-Token Prediction (MTP) Fused Dual-RMSNorm & Shared-Head Kernels,
EAGLE-3 Training-Time Test (TTT) GQA-Packed Two-Region Attention, &
8-GPU Vocabulary-Parallel Coupled-Monoid Teacher-Student Distillation.

Architecture & Primary Literature Anchor:
  - MiniLLM (Reverse KL Distillation): Gu et al., ICLR 2024 (arXiv:2306.08543)
  - GKD (Generalized JSD Distillation): Agarwal et al., ICLR 2024 (arXiv:2306.13649)
  - Better & Faster LLMs via Multi-Token Prediction: Gloeckle et al., ICML 2024 (arXiv:2404.19737)
  - DeepSeek-V3 Sequential MTP: DeepSeek-AI, 2024 (arXiv:2412.19437)
  - EAGLE-3 Training-Time Test (TTT): Li et al., 2025 (arXiv:2503.01840)
  - Liger-Kernel Fused Linear JSD / KL: Hsu et al., 2024 (arXiv:2410.10989)
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

warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from flash_attn import flash_attn_func
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import environment_metadata

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# 0. Utilities & NVML Telemetry
# ============================================================================


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
        return 0.0, 0.0


def time_fn_with_peak_mem(
    fn: Callable[[], None],
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
    base_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    peak_inc_bytes = max(0, torch.cuda.max_memory_allocated(device) - base_mem)
    peak_inc_mib = peak_inc_bytes / (1024.0**2)
    transient_ws_mib = max(0.0, (peak_inc_bytes - grad_bytes_steady) / (1024.0**2))

    samples: list[float] = []
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        s_ev.record()
        fn()
        e_ev.record()
        torch.cuda.synchronize(device)
        samples.append(float(s_ev.elapsed_time(e_ev)))

    sm_clk, pwr_w = query_gpu_telemetry(device.index or 0)
    ss = sorted(samples)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    return {
        "median_ms": round(float(statistics.median(ss)), 4),
        "iqr_ms": round(float(q75 - q25), 4),
        "peak_inc_mib": round(float(peak_inc_mib), 2),
        "transient_workspace_mib": round(float(transient_ws_mib), 2),
        "sm_clock_after_mhz": round(sm_clk, 1),
        "power_after_w": round(pwr_w, 2),
    }


# ============================================================================
# 1. Coupled 6-Statistic Online Softmax-Divergence Monoid Kernels (V = 128,256)
#    Forward KL (MODE=0), Reverse KL / MiniLLM (MODE=1), Generalized JSD (MODE=2)
# ============================================================================


@triton.jit
def _naive_uncentered_rkl_kernel(
    ZS_ptr,
    stride_zs,
    ZT_ptr,
    stride_zt,
    Out_ptr,
    V,
    inv_tau,
    BLOCK_V: tl.constexpr,
):
    """Negative-control uncentered online Reverse-KL:
    RKL = (sum (z_s - z_t) * exp(z_s - m_s)) / d_s - (lse_s - lse_t).
    Suffers catastrophic FP32 mantissa cancellation under independent Teacher/Student shifts!
    """
    row = tl.program_id(0)
    zs_row = ZS_ptr + row * stride_zs
    zt_row = ZT_ptr + row * stride_zt
    ms = -1.0e30
    ds = 0.0
    mt = -1.0e30
    dt = 0.0
    udiff = 0.0
    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        zs = tl.load(zs_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        zt = tl.load(zt_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau

        ms_t = tl.max(zs, axis=0)
        es_v = tl.where(mask, tl.exp(zs - ms_t), 0.0)
        ds_t = tl.sum(es_v, axis=0)
        ud_t = tl.sum(tl.where(mask, zs - zt, 0.0) * es_v, axis=0)

        ms_n = tl.maximum(ms, ms_t)
        as1 = tl.exp(ms - ms_n)
        as2 = tl.exp(ms_t - ms_n)
        ds = ds * as1 + ds_t * as2
        udiff = udiff * as1 + ud_t * as2
        ms = ms_n

        mt_t = tl.max(zt, axis=0)
        dt_t = tl.sum(tl.where(mask, tl.exp(zt - mt_t), 0.0), axis=0)
        mt_n = tl.maximum(mt, mt_t)
        dt = dt * tl.exp(mt - mt_n) + dt_t * tl.exp(mt_t - mt_n)
        mt = mt_n

    lse_s = ms + tl.log(ds)
    lse_t = mt + tl.log(dt)
    rkl = udiff / ds - (lse_s - lse_t)
    tl.store(Out_ptr + row, rkl)


@triton.jit
def _coupled_kd_fwd_bwd_inplace_kernel(
    ZS_ptr,
    stride_zs,
    ZT_ptr,
    stride_zt,
    W_tok_ptr,
    Loss_ptr,
    V,
    inv_tau,
    tau_sq,
    beta_jsd,
    MODE: tl.constexpr,  # 0: Forward KL, 1: Reverse KL (MiniLLM), 2: Generalized JSD_beta (GKD)
    BLOCK_V: tl.constexpr,
):
    """Fused in-place Teacher-Student Knowledge Distillation Forward + Student-Logit-Grad kernel.
    Maintains the shift-invariant 6-statistic coupled online monoid (m_S, d_S, e_S, m_T, d_T, c_ST):
      e_S  = sum (z_S - m_S) exp(z_S - m_S) <= 0
      c_ST = sum (z_T - m_T) exp(z_S - m_S)
      c_ST_new = exp(m_S_old - m_S_new) * (c_ST_old + d_S_old * (m_T_old - m_T_new))
      RKL(P_S || P_T) = (e_S - c_ST) / d_S - ln(d_S) + ln(d_T)
    """
    row = tl.program_id(0)
    zs_row = ZS_ptr + row * stride_zs
    zt_row = ZT_ptr + row * stride_zt
    w_tok = tl.load(W_tok_ptr + row).to(tl.float32)

    ms = -1.0e30
    ds = 0.0
    es = 0.0
    mt = -1.0e30
    dt = 0.0
    cst = 0.0

    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        zs = tl.load(zs_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        zt = tl.load(zt_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau

        ms_t = tl.max(zs, axis=0)
        mt_t = tl.max(zt, axis=0)
        z_sh_s = tl.where(mask, zs - ms_t, 0.0)
        z_sh_t = tl.where(mask, zt - mt_t, 0.0)
        exp_s = tl.where(mask, tl.exp(z_sh_s), 0.0)
        exp_t = tl.where(mask, tl.exp(z_sh_t), 0.0)
        ds_t = tl.sum(exp_s, axis=0)
        dt_t = tl.sum(exp_t, axis=0)

        ms_n = tl.maximum(ms, ms_t)
        mt_n = tl.maximum(mt, mt_t)
        as1 = tl.exp(ms - ms_n)
        as2 = tl.exp(ms_t - ms_n)
        at1 = tl.exp(mt - mt_n)
        at2 = tl.exp(mt_t - mt_n)

        if MODE == 1:
            es_t = tl.sum(z_sh_s * exp_s, axis=0)
            cst_t = tl.sum(z_sh_t * exp_s, axis=0)
            es = as1 * (es + ds * (ms - ms_n)) + as2 * (es_t + ds_t * (ms_t - ms_n))
            cst = as1 * (cst + ds * (mt - mt_n)) + as2 * (cst_t + ds_t * (mt_t - mt_n))

        ds = ds * as1 + ds_t * as2
        dt = dt * at1 + dt_t * at2
        ms = ms_n
        mt = mt_n

    lse_s = ms + tl.log(ds)
    lse_t = mt + tl.log(dt)

    loss_acc = 0.0
    kl_s_m = 0.0
    log_b = tl.log(beta_jsd)
    log_1mb = tl.log(1.0 - beta_jsd)

    if MODE == 2:
        # JSD_beta is non-linear in lse_T - lse_S, requiring one scalar reduction pass for KL(P_S || M)
        for off in range(0, V, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            mask = cols < V
            zs = tl.load(zs_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
            zt = tl.load(zt_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
            ls = zs - lse_s
            lt = zt - lse_t
            ps = tl.where(mask, tl.exp(ls), 0.0)
            pt = tl.where(mask, tl.exp(lt), 0.0)
            a = log_b + lt
            b = log_1mb + ls
            mx = tl.maximum(a, b)
            lm = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))
            kl_t_m_v = tl.where(mask, pt * (lt - lm), 0.0)
            kl_s_m_v = tl.where(mask, ps * (ls - lm), 0.0)
            loss_acc += tl.sum(beta_jsd * kl_t_m_v + (1.0 - beta_jsd) * kl_s_m_v, axis=0)
            kl_s_m += tl.sum(kl_s_m_v, axis=0)

    rkl_monoid = 0.0
    if MODE == 1:
        rkl_monoid = (es - cst) / ds - tl.log(ds) + tl.log(dt)
        loss_acc = rkl_monoid

    scale_out = w_tok * tau_sq * inv_tau
    ln_ds_minus_dt = tl.log(ds) - tl.log(dt)
    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        zs = tl.load(zs_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        zt = tl.load(zt_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        ls = zs - lse_s
        lt = zt - lse_t
        ps = tl.where(mask, tl.exp(ls), 0.0)
        pt = tl.where(mask, tl.exp(lt), 0.0)

        if MODE == 0:
            loss_acc += tl.sum(tl.where(mask, pt * (lt - ls), 0.0), axis=0)
            du = scale_out * (ps - pt)
        elif MODE == 1:
            diff_st = tl.where(mask, (zs - ms) - (zt - mt) - ln_ds_minus_dt, 0.0)
            du = scale_out * ps * (diff_st - rkl_monoid)
        else:
            a = log_b + lt
            b = log_1mb + ls
            mx = tl.maximum(a, b)
            lm = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))
            du = scale_out * (1.0 - beta_jsd) * ps * ((ls - lm) - kl_s_m)

        tl.store(zs_row + cols, du.to(ZS_ptr.dtype.element_ty), mask=mask)

    tl.store(Loss_ptr + row, w_tok * tau_sq * loss_acc)


class _FusedCoupledLinearKDFn(torch.autograd.Function):
    """Token-tiled Teacher (no_grad) + Student (grad) Linear + Coupled-Monoid KD (FKL / RKL / JSD)."""

    @staticmethod
    def forward(
        ctx: Any,
        h_s: torch.Tensor,
        w_s: torch.Tensor,
        h_t: torch.Tensor,
        w_t: torch.Tensor,
        w_tok: torch.Tensor,
        tau: float,
        beta_jsd: float,
        mode_id: int,
        chunk_size: int,
        use_fp32_dw_accum: bool,
    ) -> torch.Tensor:
        N, H_s = h_s.shape
        V = w_s.shape[0]
        device = h_s.device
        inv_tau = 1.0 / tau
        tau_sq = tau * tau

        dh_s = torch.empty_like(h_s)
        if use_fp32_dw_accum:
            dw_s_acc = torch.zeros((V, H_s), device=device, dtype=torch.float32)
        else:
            dw_s_out = torch.zeros_like(w_s)
        loss_vec = torch.empty((N,), device=device, dtype=torch.float32)

        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            hs_c = h_s[start:end]
            ht_c = h_t[start:end]
            wt_c = w_tok[start:end]
            # Student & Teacher chunk logits: [chunk, V] BF16 (250.5 MiB each at chunk=1024, V=128256!)
            zs_c = F.linear(hs_c, w_s)
            zt_c = F.linear(ht_c, w_t)
            _coupled_kd_fwd_bwd_inplace_kernel[(end - start,)](
                zs_c,
                zs_c.stride(0),
                zt_c,
                zt_c.stride(0),
                wt_c,
                loss_vec[start:end],
                V,
                inv_tau,
                tau_sq,
                beta_jsd,
                MODE=mode_id,
                BLOCK_V=8192,
                num_warps=16,
            )
            del zt_c
            # zs_c now holds exact student logit gradient du_s in-place!
            torch.matmul(zs_c, w_s, out=dh_s[start:end])
            if use_fp32_dw_accum:
                torch.addmm(
                    dw_s_acc, zs_c.t(), hs_c, out_dtype=torch.float32, out=dw_s_acc
                )
            else:
                dw_s_out.addmm_(zs_c.t(), hs_c, beta=1.0, alpha=1.0)
            del zs_c

        dw_final = dw_s_acc.to(w_s.dtype) if use_fp32_dw_accum else dw_s_out
        ctx.save_for_backward(dh_s, dw_final)
        return loss_vec.sum()

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        dh_s, dw_s = ctx.saved_tensors
        g = grad_out.to(dh_s.dtype)
        return dh_s * g, dw_s * g, None, None, None, None, None, None, None, None


# ============================================================================
# 2. Vocab-Parallel (TP=8, V_shard = 16,032) Coupled-Monoid Kernels
#    Exchanges ONLY [N, 6] FP32 scalars (24 B/tok) across all 8 GPUs!
# ============================================================================


@triton.jit
def _vocab_shard_pass1_monoid_kernel(
    ZS_ptr,
    stride_zs,
    ZT_ptr,
    stride_zt,
    Stats_ptr,  # [N, 6] float32: (ms, ds, es, mt, dt, cst)
    stride_st,
    V_shard,
    inv_tau,
    NEED_RKL_STATS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    zs_row = ZS_ptr + row * stride_zs
    zt_row = ZT_ptr + row * stride_zt

    ms = -1.0e30
    ds = 0.0
    es = 0.0
    mt = -1.0e30
    dt = 0.0
    cst = 0.0

    for off in range(0, V_shard, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V_shard
        zs = tl.load(zs_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        zt = tl.load(zt_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau

        ms_t = tl.max(zs, axis=0)
        mt_t = tl.max(zt, axis=0)
        z_sh_s = tl.where(mask, zs - ms_t, 0.0)
        z_sh_t = tl.where(mask, zt - mt_t, 0.0)
        exp_s = tl.where(mask, tl.exp(z_sh_s), 0.0)
        exp_t = tl.where(mask, tl.exp(z_sh_t), 0.0)
        ds_t = tl.sum(exp_s, axis=0)
        dt_t = tl.sum(exp_t, axis=0)

        ms_n = tl.maximum(ms, ms_t)
        mt_n = tl.maximum(mt, mt_t)
        as1 = tl.exp(ms - ms_n)
        as2 = tl.exp(ms_t - ms_n)
        at1 = tl.exp(mt - mt_n)
        at2 = tl.exp(mt_t - mt_n)

        if NEED_RKL_STATS:
            es_t = tl.sum(z_sh_s * exp_s, axis=0)
            cst_t = tl.sum(z_sh_t * exp_s, axis=0)
            es = as1 * (es + ds * (ms - ms_n)) + as2 * (es_t + ds_t * (ms_t - ms_n))
            cst = as1 * (cst + ds * (mt - mt_n)) + as2 * (cst_t + ds_t * (mt_t - mt_n))

        ds = ds * as1 + ds_t * as2
        dt = dt * at1 + dt_t * at2
        ms = ms_n
        mt = mt_n

    st_base = Stats_ptr + row * stride_st
    tl.store(st_base + 0, ms)
    tl.store(st_base + 1, ds)
    tl.store(st_base + 2, es)
    tl.store(st_base + 3, mt)
    tl.store(st_base + 4, dt)
    tl.store(st_base + 5, cst)


@triton.jit
def _vocab_shard_jsd_partial_kernel(
    ZS_ptr,
    stride_zs,
    ZT_ptr,
    stride_zt,
    LSE_S_ptr,
    LSE_T_ptr,
    Out2_ptr,  # [N, 2] float32: (loss_shard, kl_s_m_shard)
    stride_o2,
    V_shard,
    inv_tau,
    beta_jsd,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    zs_row = ZS_ptr + row * stride_zs
    zt_row = ZT_ptr + row * stride_zt
    lse_s = tl.load(LSE_S_ptr + row)
    lse_t = tl.load(LSE_T_ptr + row)
    log_b = tl.log(beta_jsd)
    log_1mb = tl.log(1.0 - beta_jsd)

    loss_s = 0.0
    kl_sm = 0.0
    for off in range(0, V_shard, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V_shard
        zs = tl.load(zs_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        zt = tl.load(zt_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        ls = zs - lse_s
        lt = zt - lse_t
        ps = tl.where(mask, tl.exp(ls), 0.0)
        pt = tl.where(mask, tl.exp(lt), 0.0)
        a = log_b + lt
        b = log_1mb + ls
        mx = tl.maximum(a, b)
        lm = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))
        ktm = tl.where(mask, pt * (lt - lm), 0.0)
        ksm = tl.where(mask, ps * (ls - lm), 0.0)
        loss_s += tl.sum(beta_jsd * ktm + (1.0 - beta_jsd) * ksm, axis=0)
        kl_sm += tl.sum(ksm, axis=0)

    tl.store(Out2_ptr + row * stride_o2 + 0, loss_s)
    tl.store(Out2_ptr + row * stride_o2 + 1, kl_sm)


@triton.jit
def _vocab_shard_pass2_inplace_kernel(
    ZS_ptr,
    stride_zs,
    ZT_ptr,
    stride_zt,
    MS_ptr,
    DS_ptr,
    MT_ptr,
    DT_ptr,
    ScalarAux_ptr,  # RKL global (MODE=1) or kl_s_m global (MODE=2)
    W_tok_ptr,
    LossShard_ptr,
    V_shard,
    inv_tau,
    tau_sq,
    beta_jsd,
    MODE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    zs_row = ZS_ptr + row * stride_zs
    zt_row = ZT_ptr + row * stride_zt
    ms = tl.load(MS_ptr + row)
    ds = tl.load(DS_ptr + row)
    mt = tl.load(MT_ptr + row)
    dt = tl.load(DT_ptr + row)
    aux = tl.load(ScalarAux_ptr + row)
    w_tok = tl.load(W_tok_ptr + row).to(tl.float32)

    lse_s = ms + tl.log(ds)
    lse_t = mt + tl.log(dt)
    ln_ds_minus_dt = tl.log(ds) - tl.log(dt)
    scale_out = w_tok * tau_sq * inv_tau
    log_b = tl.log(beta_jsd)
    log_1mb = tl.log(1.0 - beta_jsd)

    fkl_part = 0.0
    for off in range(0, V_shard, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V_shard
        zs = tl.load(zs_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        zt = tl.load(zt_row + cols, mask=mask, other=-1.0e30).to(tl.float32) * inv_tau
        ls = zs - lse_s
        lt = zt - lse_t
        ps = tl.where(mask, tl.exp(ls), 0.0)
        pt = tl.where(mask, tl.exp(lt), 0.0)

        if MODE == 0:
            fkl_part += tl.sum(tl.where(mask, pt * (lt - ls), 0.0), axis=0)
            du = scale_out * (ps - pt)
        elif MODE == 1:
            diff_st = tl.where(mask, (zs - ms) - (zt - mt) - ln_ds_minus_dt, 0.0)
            du = scale_out * ps * (diff_st - aux)
        else:
            a = log_b + lt
            b = log_1mb + ls
            mx = tl.maximum(a, b)
            lm = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))
            du = scale_out * (1.0 - beta_jsd) * ps * ((ls - lm) - aux)

        tl.store(zs_row + cols, du.to(ZS_ptr.dtype.element_ty), mask=mask)

    if MODE == 0:
        tl.store(LossShard_ptr + row, w_tok * tau_sq * fkl_part)


def reduce_coupled_monoid_across_ranks(
    local_stats: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Associative merge of [N, 6] local monoids across `world_size` vocabulary shards."""
    N = local_stats.shape[0]
    gathered = torch.empty(
        (world_size, N, 6), device=local_stats.device, dtype=torch.float32
    )
    dist.all_gather_into_tensor(gathered, local_stats)

    ms = gathered[0, :, 0].clone()
    ds = gathered[0, :, 1].clone()
    es = gathered[0, :, 2].clone()
    mt = gathered[0, :, 3].clone()
    dt = gathered[0, :, 4].clone()
    cst = gathered[0, :, 5].clone()

    for r in range(1, world_size):
        ms_r = gathered[r, :, 0]
        ds_r = gathered[r, :, 1]
        es_r = gathered[r, :, 2]
        mt_r = gathered[r, :, 3]
        dt_r = gathered[r, :, 4]
        cst_r = gathered[r, :, 5]

        ms_n = torch.maximum(ms, ms_r)
        mt_n = torch.maximum(mt, mt_r)
        as1 = torch.exp(ms - ms_n)
        as2 = torch.exp(ms_r - ms_n)
        at1 = torch.exp(mt - mt_n)
        at2 = torch.exp(mt_r - mt_n)

        es = as1 * (es + ds * (ms - ms_n)) + as2 * (es_r + ds_r * (ms_r - ms_n))
        cst = as1 * (cst + ds * (mt - mt_n)) + as2 * (cst_r + ds_r * (mt_r - mt_n))
        ds = ds * as1 + ds_r * as2
        dt = dt * at1 + dt_r * at2
        ms = ms_n
        mt = mt_n

    rkl_global = (es - cst) / ds - torch.log(ds) + torch.log(dt)
    return ms, ds, mt, dt, rkl_global


class _VocabParallelCoupledKDFn(torch.autograd.Function):
    """8-GPU Vocab-Parallel Student Head + Coupled-Monoid KD:
    Input: h_s_local [N_loc, H_s], w_s_shard [V_shard, H_s], z_t_shard [N_global, V_shard].
    AllGathers h_s -> [N_global, H_s], tiles over chunks, exchanges ONLY 24 B/tok monoid stats,
    accumulates dW_s_shard locally with ZERO weight collectives, and ReduceScatters dh_s_local!
    """

    @staticmethod
    def forward(
        ctx: Any,
        h_s_local: torch.Tensor,
        w_s_shard: torch.Tensor,
        z_t_shard: torch.Tensor,
        w_tok_global: torch.Tensor,
        tau: float,
        beta_jsd: float,
        mode_id: int,
        chunk_size: int,
        world_size: int,
    ) -> torch.Tensor:
        N_loc, H_s = h_s_local.shape
        N_glob = N_loc * world_size
        V_sh = w_s_shard.shape[0]
        device = h_s_local.device
        inv_tau = 1.0 / tau
        tau_sq = tau * tau

        # AllGather compact student hidden states: [N_global, H_s] (62.6x smaller than [N_global, V] logits!)
        h_s_glob = torch.empty((N_glob, H_s), device=device, dtype=h_s_local.dtype)
        dist.all_gather_into_tensor(h_s_glob, h_s_local.contiguous())

        # Accumulate partial vocabulary-shard dH_s in FP32 BEFORE ReduceScatter!
        # Truncating 1/8-vocab partial dot products to BF16 prior to ReduceScatter injects 8x
        # zero-sum cancellation noise (~3.7% dW drift); FP32 ReduceScatter restores ~0.2% parity!
        dh_s_glob32 = torch.empty((N_glob, H_s), device=device, dtype=torch.float32)
        dw_sh_acc = torch.zeros((V_sh, H_s), device=device, dtype=torch.float32)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)

        for start in range(0, N_glob, chunk_size):
            end = min(N_glob, start + chunk_size)
            cN = end - start
            hs_c = h_s_glob[start:end]
            zt_c = z_t_shard[start:end]
            wt_c = w_tok_global[start:end]

            zs_c = F.linear(hs_c, w_s_shard)
            st_loc = torch.empty((cN, 6), device=device, dtype=torch.float32)
            _vocab_shard_pass1_monoid_kernel[(cN,)](
                zs_c,
                zs_c.stride(0),
                zt_c,
                zt_c.stride(0),
                st_loc,
                st_loc.stride(0),
                V_sh,
                inv_tau,
                NEED_RKL_STATS=(mode_id == 1),
                BLOCK_V=4096,
                num_warps=8,
            )
            ms_g, ds_g, mt_g, dt_g, rkl_g = reduce_coupled_monoid_across_ranks(
                st_loc, world_size
            )

            loss_shard = torch.zeros((cN,), device=device, dtype=torch.float32)
            if mode_id == 2:
                lse_s = ms_g + torch.log(ds_g)
                lse_t = mt_g + torch.log(dt_g)
                out2 = torch.empty((cN, 2), device=device, dtype=torch.float32)
                _vocab_shard_jsd_partial_kernel[(cN,)](
                    zs_c,
                    zs_c.stride(0),
                    zt_c,
                    zt_c.stride(0),
                    lse_s,
                    lse_t,
                    out2,
                    out2.stride(0),
                    V_sh,
                    inv_tau,
                    beta_jsd,
                    BLOCK_V=4096,
                    num_warps=8,
                )
                dist.all_reduce(out2, op=dist.ReduceOp.SUM)
                aux_vec = out2[:, 1].contiguous()
                total_loss = total_loss + (wt_c * tau_sq * out2[:, 0]).sum()
            else:
                aux_vec = rkl_g.contiguous()
                if mode_id == 1:
                    total_loss = total_loss + (wt_c * tau_sq * rkl_g).sum()

            _vocab_shard_pass2_inplace_kernel[(cN,)](
                zs_c,
                zs_c.stride(0),
                zt_c,
                zt_c.stride(0),
                ms_g,
                ds_g,
                mt_g,
                dt_g,
                aux_vec,
                wt_c,
                loss_shard,
                V_sh,
                inv_tau,
                tau_sq,
                beta_jsd,
                MODE=mode_id,
                BLOCK_V=4096,
                num_warps=8,
            )
            if mode_id == 0:
                dist.all_reduce(loss_shard, op=dist.ReduceOp.SUM)
                total_loss = total_loss + loss_shard.sum()

            torch.mm(
                zs_c, w_s_shard, out_dtype=torch.float32, out=dh_s_glob32[start:end]
            )
            torch.addmm(
                dw_sh_acc, zs_c.t(), hs_c, out_dtype=torch.float32, out=dw_sh_acc
            )
            del zs_c

        # ReduceScatter dh_s_glob32 [N_glob, H_s] -> dh_s_local32 [N_loc, H_s] in exact FP32!
        dh_s_local32 = torch.empty((N_loc, H_s), device=device, dtype=torch.float32)
        dist.reduce_scatter_tensor(dh_s_local32, dh_s_glob32, op=dist.ReduceOp.SUM)
        dh_s_local = (dh_s_local32 * float(world_size)).to(h_s_local.dtype)
        ctx.save_for_backward(dh_s_local, dw_sh_acc.to(w_s_shard.dtype))
        return total_loss

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        dh_s_scaled, dw_sh = ctx.saved_tensors
        g = grad_out.to(dh_s_scaled.dtype)
        return dh_s_scaled * g, dw_sh * g, None, None, None, None, None, None, None


# ============================================================================
# 3. DeepSeek-V3 Multi-Token Prediction (MTP):
#    Fused Dual-RMSNorm Concat Kernel & Shared-W_unembed Batched Linear CE
# ============================================================================


@triton.jit
def _fused_dual_rmsnorm_concat_fwd_kernel(
    H_ptr,
    stride_h,
    E_ptr,
    stride_e,
    WH_ptr,
    WE_ptr,
    Out2H_ptr,
    stride_out,
    RstdH_ptr,
    RstdE_ptr,
    H_DIM: tl.constexpr,
    eps,
):
    """Computes [RMSNorm(h_row, w_h) ; RMSNorm(e_row, w_e)] directly into contiguous [N, 2*H]
    in ONE kernel launch with zero intermediate HBM buffers or torch.cat copy!
    """
    row = tl.program_id(0)
    cols = tl.arange(0, H_DIM)
    h = tl.load(H_ptr + row * stride_h + cols).to(tl.float32)
    e = tl.load(E_ptr + row * stride_e + cols).to(tl.float32)
    wh = tl.load(WH_ptr + cols).to(tl.float32)
    we = tl.load(WE_ptr + cols).to(tl.float32)

    rstd_h = tl.math.rsqrt(tl.sum(h * h, axis=0) / H_DIM + eps)
    rstd_e = tl.math.rsqrt(tl.sum(e * e, axis=0) / H_DIM + eps)
    tl.store(RstdH_ptr + row, rstd_h)
    tl.store(RstdE_ptr + row, rstd_e)

    yh = (h * rstd_h * wh).to(Out2H_ptr.dtype.element_ty)
    ye = (e * rstd_e * we).to(Out2H_ptr.dtype.element_ty)
    out_base = Out2H_ptr + row * stride_out
    tl.store(out_base + cols, yh)
    tl.store(out_base + H_DIM + cols, ye)


@triton.jit
def _fused_dual_rmsnorm_concat_bwd_kernel(
    dOut2H_ptr,
    stride_dout,
    H_ptr,
    stride_h,
    E_ptr,
    stride_e,
    WH_ptr,
    WE_ptr,
    RstdH_ptr,
    RstdE_ptr,
    dH_ptr,
    dE_ptr,
    dWH_part_ptr,
    dWE_part_ptr,
    stride_part,
    N_rows,
    H_DIM: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    cols = tl.arange(0, H_DIM)
    wh = tl.load(WH_ptr + cols).to(tl.float32)
    we = tl.load(WE_ptr + cols).to(tl.float32)

    dwh_acc = tl.zeros([H_DIM], dtype=tl.float32)
    dwe_acc = tl.zeros([H_DIM], dtype=tl.float32)

    for idx in range(0, ROWS_PER_PROG):
        row = row_start + idx
        if row < N_rows:
            dout_base = dOut2H_ptr + row * stride_dout
            dyh = tl.load(dout_base + cols).to(tl.float32)
            dye = tl.load(dout_base + H_DIM + cols).to(tl.float32)
            h = tl.load(H_ptr + row * stride_h + cols).to(tl.float32)
            e = tl.load(E_ptr + row * stride_e + cols).to(tl.float32)
            rh = tl.load(RstdH_ptr + row)
            re_v = tl.load(RstdE_ptr + row)

            xh = h * rh
            xe = e * re_v
            dwh_acc += dyh * xh
            dwe_acc += dye * xe

            dxh = dyh * wh
            dxe = dye * we
            dh = rh * (dxh - xh * (tl.sum(dxh * xh, axis=0) / H_DIM))
            de = re_v * (dxe - xe * (tl.sum(dxe * xe, axis=0) / H_DIM))
            tl.store(dH_ptr + row * stride_h + cols, dh.to(dH_ptr.dtype.element_ty))
            tl.store(dE_ptr + row * stride_e + cols, de.to(dE_ptr.dtype.element_ty))

    tl.store(dWH_part_ptr + pid * stride_part + cols, dwh_acc)
    tl.store(dWE_part_ptr + pid * stride_part + cols, dwe_acc)


class _FusedDualRMSNormConcatFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        h: torch.Tensor,
        e: torch.Tensor,
        w_h: torch.Tensor,
        w_e: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        N, H = h.shape
        out_2h = torch.empty((N, 2 * H), device=h.device, dtype=h.dtype)
        rstd_h = torch.empty((N,), device=h.device, dtype=torch.float32)
        rstd_e = torch.empty((N,), device=h.device, dtype=torch.float32)
        _fused_dual_rmsnorm_concat_fwd_kernel[(N,)](
            h,
            h.stride(0),
            e,
            e.stride(0),
            w_h,
            w_e,
            out_2h,
            out_2h.stride(0),
            rstd_h,
            rstd_e,
            H_DIM=H,
            eps=eps,
            num_warps=8,
        )
        ctx.save_for_backward(h, e, w_h, w_e, rstd_h, rstd_e)
        return out_2h

    @staticmethod
    def backward(ctx: Any, dout_2h: torch.Tensor) -> tuple[Any, ...]:
        h, e, w_h, w_e, rstd_h, rstd_e = ctx.saved_tensors
        N, H = h.shape
        dout_c = dout_2h.contiguous()
        dh = torch.empty_like(h)
        de = torch.empty_like(e)
        rows_per = 32
        n_progs = triton.cdiv(N, rows_per)
        dwh_p = torch.empty((n_progs, H), device=h.device, dtype=torch.float32)
        dwe_p = torch.empty((n_progs, H), device=h.device, dtype=torch.float32)
        _fused_dual_rmsnorm_concat_bwd_kernel[(n_progs,)](
            dout_c,
            dout_c.stride(0),
            h,
            h.stride(0),
            e,
            e.stride(0),
            w_h,
            w_e,
            rstd_h,
            rstd_e,
            dh,
            de,
            dwh_p,
            dwe_p,
            dwh_p.stride(0),
            N,
            H_DIM=H,
            ROWS_PER_PROG=rows_per,
            num_warps=8,
        )
        return dh, de, dwh_p.sum(dim=0).to(w_h.dtype), dwe_p.sum(dim=0).to(w_e.dtype), None


@triton.jit
def _weighted_ce_fwd_bwd_inplace_kernel(
    Z_ptr,
    stride_z,
    Target_ptr,
    W_tok_ptr,
    Loss_ptr,
    V,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    z_row = Z_ptr + row * stride_z
    tgt = tl.load(Target_ptr + row)
    w_tok = tl.load(W_tok_ptr + row).to(tl.float32)

    m_acc = -1.0e30
    d_acc = 0.0
    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        z = tl.load(z_row + cols, mask=mask, other=-1.0e30).to(tl.float32)
        mt = tl.max(z, axis=0)
        dt = tl.sum(tl.where(mask, tl.exp(z - mt), 0.0), axis=0)
        mn = tl.maximum(m_acc, mt)
        d_acc = d_acc * tl.exp(m_acc - mn) + dt * tl.exp(mt - mn)
        m_acc = mn

    lse = m_acc + tl.log(d_acc)
    z_t = tl.load(z_row + tgt).to(tl.float32)
    tl.store(Loss_ptr + row, w_tok * (lse - z_t))

    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        z = tl.load(z_row + cols, mask=mask, other=-1.0e30).to(tl.float32)
        p = tl.where(mask, tl.exp(z - lse), 0.0)
        ind = tl.where(cols == tgt, 1.0, 0.0)
        du = w_tok * (p - ind)
        tl.store(z_row + cols, du.to(Z_ptr.dtype.element_ty), mask=mask)


class _SharedHeadLinearCEFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        h_all: torch.Tensor,
        w_unembed: torch.Tensor,
        targets_all: torch.Tensor,
        weights_all: torch.Tensor,
        chunk_size: int,
        use_fp32_dw_accum: bool,
    ) -> torch.Tensor:
        N_tot, H = h_all.shape
        V = w_unembed.shape[0]
        dh = torch.empty_like(h_all)
        if use_fp32_dw_accum:
            dw_acc = torch.zeros((V, H), device=h_all.device, dtype=torch.float32)
        else:
            dw_out = torch.zeros_like(w_unembed)
        loss_v = torch.empty((N_tot,), device=h_all.device, dtype=torch.float32)

        for start in range(0, N_tot, chunk_size):
            end = min(N_tot, start + chunk_size)
            hc = h_all[start:end]
            zc = F.linear(hc, w_unembed)
            _weighted_ce_fwd_bwd_inplace_kernel[(end - start,)](
                zc,
                zc.stride(0),
                targets_all[start:end],
                weights_all[start:end],
                loss_v[start:end],
                V,
                BLOCK_V=8192,
                num_warps=16,
            )
            torch.matmul(zc, w_unembed, out=dh[start:end])
            if use_fp32_dw_accum:
                torch.addmm(dw_acc, zc.t(), hc, out_dtype=torch.float32, out=dw_acc)
            else:
                dw_out.addmm_(zc.t(), hc, beta=1.0, alpha=1.0)
            del zc

        ctx.save_for_backward(dh, dw_acc.to(w_unembed.dtype) if use_fp32_dw_accum else dw_out)
        return loss_v.sum()

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        dh, dw = ctx.saved_tensors
        g = grad_out.to(dh.dtype)
        return dh * g, dw * g, None, None, None, None


# ============================================================================
# 4. EAGLE-3 Training-Time Test (TTT) GQA-Packed Two-Region Triton Attention
#    (4 anchors * 4 TTT steps * 4 GQA heads = 64 queries per CTA, 0 atomics!)
# ============================================================================


def eagle3_ttt_ref_fp32(
    q_ttt: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    k_ttt: torch.Tensor,
    v_ttt: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    B, S, K, Hq, D = q_ttt.shape
    Hkv = k_ctx.shape[2]
    gqa = Hq // Hkv
    qf = q_ttt.float()
    kc = k_ctx.float().repeat_interleave(gqa, dim=2)
    vc = v_ctx.float().repeat_interleave(gqa, dim=2)
    kt = k_ttt.float().repeat_interleave(gqa, dim=3)
    vt = v_ttt.float().repeat_interleave(gqa, dim=3)

    q_perm = qf.permute(0, 3, 1, 2, 4).reshape(B, Hq, S * K, D)
    kc_perm = kc.permute(0, 2, 1, 3)
    vc_perm = vc.permute(0, 2, 1, 3)
    s_ctx = torch.matmul(q_perm, kc_perm.transpose(-1, -2)).view(B, Hq, S, K, S) * sm_scale
    t_idx = torch.arange(S, device=q_ttt.device)
    s_ctx = s_ctx.masked_fill(
        ~(t_idx[:, None] >= t_idx[None, :])[None, None, :, None, :], -1.0e30
    )

    q_ch = qf.permute(0, 1, 3, 2, 4)
    k_ch = kt.permute(0, 1, 3, 2, 4)
    v_ch = vt.permute(0, 1, 3, 2, 4)
    s_ch = torch.matmul(q_ch, k_ch.transpose(-1, -2)) * sm_scale
    k_idx = torch.arange(K, device=q_ttt.device)
    s_ch = s_ch.masked_fill(
        ~(k_idx[:, None] >= k_idx[None, :])[None, None, None, :, :], -1.0e30
    ).permute(0, 2, 1, 3, 4)

    p_all = torch.softmax(torch.cat([s_ctx, s_ch], dim=-1), dim=-1)
    out_ctx = torch.matmul(p_all[..., :S].reshape(B, Hq, S * K, S), vc_perm).view(
        B, Hq, S, K, D
    )
    out_ch = torch.matmul(p_all[..., S:], v_ch.permute(0, 2, 1, 3, 4))
    return (out_ctx + out_ch).permute(0, 2, 3, 1, 4)


@triton.jit
def _eagle3_ttt_unpacked_bwd_dq_kernel(
    Q_ptr,
    KC_ptr,
    VC_ptr,
    KT_ptr,
    VT_ptr,
    dOut_ptr,
    LSE2_ptr,
    Delta_ptr,
    dQ_ptr,
    dKT_ptr,
    dVT_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_kcb,
    stride_kcs,
    stride_kch,
    stride_ktb,
    stride_ktm,
    stride_kth,
    stride_lb,
    stride_lh,
    S,
    sm_scale,
    H_KV: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Ablation control: unpacked GQA loop reloads KC/VC `GQA_GROUP=4` separate times per KV head!"""
    pid_m = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    b = pid_bhkv // H_KV
    hkv = pid_bhkv % H_KV

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_t = offs_m // 4
    q_k = offs_m % 4
    mask_m = q_t < S
    offs_d = tl.arange(0, HEAD_DIM)
    qk_scale2 = sm_scale * 1.4426950408889634

    max_t = tl.minimum(S, (pid_m * BLOCK_M + BLOCK_M - 1) // 4 + 1)
    full_end = ((pid_m * BLOCK_M) // 4 // BLOCK_N) * BLOCK_N
    kc_base = KC_ptr + b * stride_kcb + hkv * stride_kch
    vc_base = VC_ptr + b * stride_kcb + hkv * stride_kch
    kt_base = KT_ptr + b * stride_ktb + hkv * stride_kth
    vt_base = VT_ptr + b * stride_ktb + hkv * stride_kth

    offs_t16 = (pid_m * (BLOCK_M // 4)) + tl.arange(0, BLOCK_M // 4)
    mask_t16 = offs_t16 < S
    dkt0 = tl.zeros([BLOCK_M // 4, HEAD_DIM], dtype=tl.float32)
    dkt1 = tl.zeros([BLOCK_M // 4, HEAD_DIM], dtype=tl.float32)
    dkt2 = tl.zeros([BLOCK_M // 4, HEAD_DIM], dtype=tl.float32)
    dkt3 = tl.zeros([BLOCK_M // 4, HEAD_DIM], dtype=tl.float32)
    dvt0 = tl.zeros([BLOCK_M // 4, HEAD_DIM], dtype=tl.float32)
    dvt1 = tl.zeros([BLOCK_M // 4, HEAD_DIM], dtype=tl.float32)
    dvt2 = tl.zeros([BLOCK_M // 4, HEAD_DIM], dtype=tl.float32)
    dvt3 = tl.zeros([BLOCK_M // 4, HEAD_DIM], dtype=tl.float32)

    for g in tl.static_range(0, GQA_GROUP):
        hq = hkv * GQA_GROUP + g
        q = tl.load(
            Q_ptr + b * stride_qb + offs_m[:, None] * stride_qm + hq * stride_qh + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            dOut_ptr + b * stride_qb + offs_m[:, None] * stride_qm + hq * stride_qh + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        lse2 = tl.load(LSE2_ptr + b * stride_lb + hq * stride_lh + offs_m, mask=mask_m, other=1.0e30)
        delta = tl.load(Delta_ptr + b * stride_lb + hq * stride_lh + offs_m, mask=mask_m, other=0.0)
        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for start_n in range(0, full_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(kc_base + offs_n[:, None] * stride_kcs + offs_d[None, :])
            v = tl.load(vc_base + offs_n[:, None] * stride_kcs + offs_d[None, :])
            qk = tl.dot(q, tl.trans(k)) * qk_scale2
            p = tl.math.exp2(qk - lse2[:, None])
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * sm_scale
            dq += tl.dot(ds.to(q.dtype), k)

        for start_n in range(full_end, max_t, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < max_t
            k = tl.load(kc_base + offs_n[:, None] * stride_kcs + offs_d[None, :], mask=mask_n[:, None], other=0.0)
            v = tl.load(vc_base + offs_n[:, None] * stride_kcs + offs_d[None, :], mask=mask_n[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * qk_scale2
            valid = (offs_n[None, :] <= q_t[:, None]) & mask_n[None, :] & mask_m[:, None]
            p = tl.where(valid, tl.math.exp2(qk - lse2[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * sm_scale
            dq += tl.dot(ds.to(q.dtype), k)

        qf = q.to(tl.float32)
        dof = do.to(tl.float32)
        for j in tl.static_range(0, 4):
            idx_j = q_t * 4 + j
            kj = tl.load(kt_base + idx_j[:, None] * stride_ktm + offs_d[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
            vj = tl.load(vt_base + idx_j[:, None] * stride_ktm + offs_d[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
            active = (j <= q_k) & mask_m
            sj = tl.sum(qf * kj, axis=1) * qk_scale2
            pj = tl.where(active, tl.math.exp2(sj - lse2), 0.0)
            dpj = tl.sum(dof * vj, axis=1)
            dsj = pj * (dpj - delta) * sm_scale
            dq += dsj[:, None] * kj
            dv_c = tl.sum(tl.reshape(pj[:, None] * dof, [BLOCK_M // 4, 4, HEAD_DIM]), axis=1)
            dk_c = tl.sum(tl.reshape(dsj[:, None] * qf, [BLOCK_M // 4, 4, HEAD_DIM]), axis=1)
            if j == 0:
                dvt0 += dv_c
                dkt0 += dk_c
            elif j == 1:
                dvt1 += dv_c
                dkt1 += dk_c
            elif j == 2:
                dvt2 += dv_c
                dkt2 += dk_c
            else:
                dvt3 += dv_c
                dkt3 += dk_c

        tl.store(
            dQ_ptr + b * stride_qb + offs_m[:, None] * stride_qm + hq * stride_qh + offs_d[None, :],
            dq.to(dQ_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )

    dkt_b = dKT_ptr + b * stride_ktb + hkv * stride_kth
    dvt_b = dVT_ptr + b * stride_ktb + hkv * stride_kth
    tl.store(dkt_b + (offs_t16 * 4 + 0)[:, None] * stride_ktm + offs_d[None, :], dkt0.to(dKT_ptr.dtype.element_ty), mask=mask_t16[:, None])
    tl.store(dkt_b + (offs_t16 * 4 + 1)[:, None] * stride_ktm + offs_d[None, :], dkt1.to(dKT_ptr.dtype.element_ty), mask=mask_t16[:, None])
    tl.store(dkt_b + (offs_t16 * 4 + 2)[:, None] * stride_ktm + offs_d[None, :], dkt2.to(dKT_ptr.dtype.element_ty), mask=mask_t16[:, None])
    tl.store(dkt_b + (offs_t16 * 4 + 3)[:, None] * stride_ktm + offs_d[None, :], dkt3.to(dKT_ptr.dtype.element_ty), mask=mask_t16[:, None])
    tl.store(dvt_b + (offs_t16 * 4 + 0)[:, None] * stride_ktm + offs_d[None, :], dvt0.to(dVT_ptr.dtype.element_ty), mask=mask_t16[:, None])
    tl.store(dvt_b + (offs_t16 * 4 + 1)[:, None] * stride_ktm + offs_d[None, :], dvt1.to(dVT_ptr.dtype.element_ty), mask=mask_t16[:, None])
    tl.store(dvt_b + (offs_t16 * 4 + 2)[:, None] * stride_ktm + offs_d[None, :], dvt2.to(dVT_ptr.dtype.element_ty), mask=mask_t16[:, None])
    tl.store(dvt_b + (offs_t16 * 4 + 3)[:, None] * stride_ktm + offs_d[None, :], dvt3.to(dVT_ptr.dtype.element_ty), mask=mask_t16[:, None])


@triton.jit
def _eagle3_ttt_gqa_packed_fwd_kernel(
    Q_ptr,
    KC_ptr,
    VC_ptr,
    KT_ptr,
    VT_ptr,
    Out_ptr,
    LSE2_ptr,
    stride_qb,
    stride_qhkv,
    stride_qm,
    stride_kcb,
    stride_kcs,
    stride_kchkv,
    stride_ktb,
    stride_kts,
    stride_ktk,
    stride_kthkv,
    stride_lb,
    stride_lhkv,
    S,
    sm_scale,
    H_KV: tl.constexpr,
    K_STEPS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    b = pid_bhkv // H_KV
    hkv = pid_bhkv % H_KV

    KG: tl.constexpr = K_STEPS * GQA_GROUP
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_t = offs_m // KG
    q_k = (offs_m % KG) // GQA_GROUP
    mask_m = q_t < S
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = Q_ptr + b * stride_qb + hkv * stride_qhkv
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    qk_scale2 = sm_scale * 1.4426950408889634
    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    max_t = tl.minimum(S, (pid_m * BLOCK_M + BLOCK_M - 1) // KG + 1)
    full_end = ((pid_m * BLOCK_M) // KG // BLOCK_N) * BLOCK_N

    kc_base = KC_ptr + b * stride_kcb + hkv * stride_kchkv
    vc_base = VC_ptr + b * stride_kcb + hkv * stride_kchkv

    for start_n in range(0, full_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(kc_base + offs_n[None, :] * stride_kcs + offs_d[:, None])
        v = tl.load(vc_base + offs_n[:, None] * stride_kcs + offs_d[None, :])
        qk = tl.dot(q, k) * qk_scale2
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_ij

    for start_n in range(full_end, max_t, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < max_t
        k = tl.load(
            kc_base + offs_n[None, :] * stride_kcs + offs_d[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )
        v = tl.load(
            vc_base + offs_n[:, None] * stride_kcs + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        qk = tl.dot(q, k) * qk_scale2
        valid = (offs_n[None, :] <= q_t[:, None]) & mask_n[None, :]
        qk = tl.where(valid, qk, -1.0e30)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.where(valid, tl.math.exp2(qk - m_ij[:, None]), 0.0)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_ij

    kt_base = KT_ptr + b * stride_ktb + hkv * stride_kthkv
    vt_base = VT_ptr + b * stride_ktb + hkv * stride_kthkv
    qf = q.to(tl.float32)
    for j in tl.static_range(0, 4):
        kj = tl.load(
            kt_base + q_t[:, None] * stride_kts + j * stride_ktk + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        vj = tl.load(
            vt_base + q_t[:, None] * stride_kts + j * stride_ktk + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        active = (j <= q_k) & mask_m
        s_j = tl.sum(qf * kj, axis=1) * qk_scale2
        s_j = tl.where(active, s_j, -1.0e30)
        m_ij = tl.maximum(m_i, s_j)
        pj = tl.where(active, tl.math.exp2(s_j - m_ij), 0.0)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + pj
        acc = acc * alpha[:, None] + pj[:, None] * vj
        m_i = m_ij

    out = acc / l_i[:, None]
    lse2 = m_i + tl.math.log2(l_i)
    tl.store(
        Out_ptr + b * stride_qb + hkv * stride_qhkv + offs_m[:, None] * stride_qm + offs_d[None, :],
        out.to(Out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )
    tl.store(LSE2_ptr + b * stride_lb + hkv * stride_lhkv + offs_m, lse2, mask=mask_m)


@triton.jit
def _eagle3_ttt_gqa_packed_bwd_prep_kernel(
    Out_ptr,
    dOut_ptr,
    Delta_ptr,
    stride_ob,
    stride_ohkv,
    stride_om,
    stride_db,
    stride_dhkv,
    M_tot,
    H_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    b = pid_bhkv // H_KV
    hkv = pid_bhkv % H_KV
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M_tot
    offs_d = tl.arange(0, HEAD_DIM)
    o = tl.load(
        Out_ptr + b * stride_ob + hkv * stride_ohkv + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        dOut_ptr + b * stride_ob + hkv * stride_ohkv + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(Delta_ptr + b * stride_db + hkv * stride_dhkv + offs_m, tl.sum(o * do, axis=1), mask=mask_m)


@triton.jit
def _eagle3_ttt_gqa_packed_bwd_dkv_ctx_kernel(
    Q_ptr,
    KC_ptr,
    VC_ptr,
    dOut_ptr,
    LSE2_ptr,
    Delta_ptr,
    dKC_ptr,
    dVC_ptr,
    stride_qb,
    stride_qhkv,
    stride_qm,
    stride_kcb,
    stride_kcs,
    stride_kchkv,
    stride_lb,
    stride_lhkv,
    S,
    sm_scale,
    H_KV: tl.constexpr,
    K_STEPS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    b = pid_bhkv // H_KV
    hkv = pid_bhkv % H_KV
    KG: tl.constexpr = K_STEPS * GQA_GROUP

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, HEAD_DIM)

    k = tl.load(
        KC_ptr + b * stride_kcb + offs_n[:, None] * stride_kcs + hkv * stride_kchkv + offs_d[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        VC_ptr + b * stride_kcb + offs_n[:, None] * stride_kcs + hkv * stride_kchkv + offs_d[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qk_scale2 = sm_scale * 1.4426950408889634

    M_tot = S * KG
    start_m_all = pid_n * BLOCK_N * KG
    boundary_end = tl.minimum(M_tot, (pid_n * BLOCK_N + BLOCK_N) * KG)

    q_base = Q_ptr + b * stride_qb + hkv * stride_qhkv
    do_base = dOut_ptr + b * stride_qb + hkv * stride_qhkv
    l_base = LSE2_ptr + b * stride_lb + hkv * stride_lhkv
    d_base = Delta_ptr + b * stride_lb + hkv * stride_lhkv

    for start_m in range(start_m_all, boundary_end, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M_tot
        q_t = offs_m // KG
        q = tl.load(q_base + offs_m[None, :] * stride_qm + offs_d[:, None], mask=mask_m[None, :], other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None], other=0.0)
        lse2 = tl.load(l_base + offs_m, mask=mask_m, other=1.0e30)
        delta = tl.load(d_base + offs_m, mask=mask_m, other=0.0)

        qk_t = tl.dot(k, q) * qk_scale2
        valid = (offs_n[:, None] <= q_t[None, :]) & mask_n[:, None] & mask_m[None, :]
        p_t = tl.where(valid, tl.math.exp2(qk_t - lse2[None, :]), 0.0)
        dv += tl.dot(p_t.to(k.dtype), do)
        dp_t = tl.dot(v, tl.trans(do))
        ds_t = p_t * (dp_t - delta[None, :]) * sm_scale
        dk += tl.dot(ds_t.to(k.dtype), tl.trans(q))

    for start_m in range(boundary_end, M_tot, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M_tot
        q = tl.load(q_base + offs_m[None, :] * stride_qm + offs_d[:, None], mask=mask_m[None, :], other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None], other=0.0)
        lse2 = tl.load(l_base + offs_m, mask=mask_m, other=1.0e30)
        delta = tl.load(d_base + offs_m, mask=mask_m, other=0.0)

        qk_t = tl.dot(k, q) * qk_scale2
        valid = mask_n[:, None] & mask_m[None, :]
        p_t = tl.where(valid, tl.math.exp2(qk_t - lse2[None, :]), 0.0)
        dv += tl.dot(p_t.to(k.dtype), do)
        dp_t = tl.dot(v, tl.trans(do))
        ds_t = p_t * (dp_t - delta[None, :]) * sm_scale
        dk += tl.dot(ds_t.to(k.dtype), tl.trans(q))

    tl.store(
        dKC_ptr + b * stride_kcb + offs_n[:, None] * stride_kcs + hkv * stride_kchkv + offs_d[None, :],
        dk.to(dKC_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )
    tl.store(
        dVC_ptr + b * stride_kcb + offs_n[:, None] * stride_kcs + hkv * stride_kchkv + offs_d[None, :],
        dv.to(dVC_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )


@triton.jit
def _eagle3_ttt_gqa_packed_bwd_dq_dkv_ttt_kernel(
    Q_ptr,
    KC_ptr,
    VC_ptr,
    KT_ptr,
    VT_ptr,
    dOut_ptr,
    LSE2_ptr,
    Delta_ptr,
    dQ_ptr,
    dKT_ptr,
    dVT_ptr,
    stride_qb,
    stride_qhkv,
    stride_qm,
    stride_kcb,
    stride_kcs,
    stride_kchkv,
    stride_ktb,
    stride_kts,
    stride_ktk,
    stride_kthkv,
    stride_lb,
    stride_lhkv,
    S,
    sm_scale,
    H_KV: tl.constexpr,
    K_STEPS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    b = pid_bhkv // H_KV
    hkv = pid_bhkv % H_KV
    KG: tl.constexpr = K_STEPS * GQA_GROUP
    ANCHORS: tl.constexpr = BLOCK_M // KG

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_t = offs_m // KG
    q_k = (offs_m % KG) // GQA_GROUP
    mask_m = q_t < S
    offs_d = tl.arange(0, HEAD_DIM)
    qk_scale2 = sm_scale * 1.4426950408889634

    q_base = Q_ptr + b * stride_qb + hkv * stride_qhkv
    do_base = dOut_ptr + b * stride_qb + hkv * stride_qhkv
    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None], other=0.0)
    do = tl.load(do_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None], other=0.0)
    lse2 = tl.load(LSE2_ptr + b * stride_lb + hkv * stride_lhkv + offs_m, mask=mask_m, other=1.0e30)
    delta = tl.load(Delta_ptr + b * stride_lb + hkv * stride_lhkv + offs_m, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    max_t = tl.minimum(S, (pid_m * BLOCK_M + BLOCK_M - 1) // KG + 1)
    full_end = ((pid_m * BLOCK_M) // KG // BLOCK_N) * BLOCK_N
    kc_base = KC_ptr + b * stride_kcb + hkv * stride_kchkv
    vc_base = VC_ptr + b * stride_kcb + hkv * stride_kchkv

    for start_n in range(0, full_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(kc_base + offs_n[:, None] * stride_kcs + offs_d[None, :])
        v = tl.load(vc_base + offs_n[:, None] * stride_kcs + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * qk_scale2
        p = tl.math.exp2(qk - lse2[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        dq += tl.dot(ds.to(q.dtype), k)

    for start_n in range(full_end, max_t, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < max_t
        k = tl.load(kc_base + offs_n[:, None] * stride_kcs + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        v = tl.load(vc_base + offs_n[:, None] * stride_kcs + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale2
        valid = (offs_n[None, :] <= q_t[:, None]) & mask_n[None, :] & mask_m[:, None]
        p = tl.where(valid, tl.math.exp2(qk - lse2[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        dq += tl.dot(ds.to(q.dtype), k)

    kt_base = KT_ptr + b * stride_ktb + hkv * stride_kthkv
    vt_base = VT_ptr + b * stride_ktb + hkv * stride_kthkv
    dkt_base = dKT_ptr + b * stride_ktb + hkv * stride_kthkv
    dvt_base = dVT_ptr + b * stride_ktb + hkv * stride_kthkv
    offs_a = pid_m * ANCHORS + tl.arange(0, ANCHORS)
    mask_a = offs_a < S
    qf = q.to(tl.float32)
    dof = do.to(tl.float32)

    for j in tl.static_range(0, 4):
        kj = tl.load(
            kt_base + q_t[:, None] * stride_kts + j * stride_ktk + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        vj = tl.load(
            vt_base + q_t[:, None] * stride_kts + j * stride_ktk + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        active = (j <= q_k) & mask_m
        sj = tl.sum(qf * kj, axis=1) * qk_scale2
        pj = tl.where(active, tl.math.exp2(sj - lse2), 0.0)
        dpj = tl.sum(dof * vj, axis=1)
        dsj = pj * (dpj - delta) * sm_scale
        dq += dsj[:, None] * kj

        dv_j = tl.sum(tl.reshape(pj[:, None] * dof, [ANCHORS, KG, HEAD_DIM]), axis=1)
        dk_j = tl.sum(tl.reshape(dsj[:, None] * qf, [ANCHORS, KG, HEAD_DIM]), axis=1)
        tl.store(
            dkt_base + offs_a[:, None] * stride_kts + j * stride_ktk + offs_d[None, :],
            dk_j.to(dKT_ptr.dtype.element_ty),
            mask=mask_a[:, None],
        )
        tl.store(
            dvt_base + offs_a[:, None] * stride_kts + j * stride_ktk + offs_d[None, :],
            dv_j.to(dVT_ptr.dtype.element_ty),
            mask=mask_a[:, None],
        )

    tl.store(
        dQ_ptr + b * stride_qb + hkv * stride_qhkv + offs_m[:, None] * stride_qm + offs_d[None, :],
        dq.to(dQ_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


class _Eagle3TTTGQAPackedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q_ttt: torch.Tensor,  # [B, S, 4, Hq, D]
        k_ctx: torch.Tensor,  # [B, S, Hkv, D]
        v_ctx: torch.Tensor,  # [B, S, Hkv, D]
        k_ttt: torch.Tensor,  # [B, S, 4, Hkv, D]
        v_ttt: torch.Tensor,  # [B, S, 4, Hkv, D]
    ) -> torch.Tensor:
        B, S, K, Hq, D = q_ttt.shape
        Hkv = k_ctx.shape[2]
        GQA = Hq // Hkv
        sm_scale = 1.0 / math.sqrt(D)
        M_tot = S * K * GQA

        q_pack = (
            q_ttt.view(B, S, K, Hkv, GQA, D)
            .permute(0, 3, 1, 2, 4, 5)
            .reshape(B, Hkv, M_tot, D)
            .contiguous()
        )
        kc_c = k_ctx.contiguous()
        vc_c = v_ctx.contiguous()
        kt_c = k_ttt.contiguous()
        vt_c = v_ttt.contiguous()

        out_pack = torch.empty_like(q_pack)
        lse2 = torch.empty((B, Hkv, M_tot), device=q_ttt.device, dtype=torch.float32)

        _eagle3_ttt_gqa_packed_fwd_kernel[(triton.cdiv(M_tot, 64), B * Hkv)](
            q_pack,
            kc_c,
            vc_c,
            kt_c,
            vt_c,
            out_pack,
            lse2,
            q_pack.stride(0),
            q_pack.stride(1),
            q_pack.stride(2),
            kc_c.stride(0),
            kc_c.stride(1),
            kc_c.stride(2),
            kt_c.stride(0),
            kt_c.stride(1),
            kt_c.stride(2),
            kt_c.stride(3),
            lse2.stride(0),
            lse2.stride(1),
            S,
            sm_scale,
            H_KV=Hkv,
            K_STEPS=K,
            GQA_GROUP=GQA,
            HEAD_DIM=D,
            BLOCK_M=64,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(q_pack, kc_c, vc_c, kt_c, vt_c, out_pack, lse2)
        ctx.shapes = (B, S, K, Hq, Hkv, GQA, D, sm_scale)
        return (
            out_pack.view(B, Hkv, S, K, GQA, D)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(B, S, K, Hq, D)
        )

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[Any, ...]:
        q_pack, kc_c, vc_c, kt_c, vt_c, out_pack, lse2 = ctx.saved_tensors
        B, S, K, Hq, Hkv, GQA, D, sm_scale = ctx.shapes
        M_tot = S * K * GQA

        do_pack = (
            dout.contiguous()
            .view(B, S, K, Hkv, GQA, D)
            .permute(0, 3, 1, 2, 4, 5)
            .reshape(B, Hkv, M_tot, D)
            .contiguous()
        )
        delta = torch.empty_like(lse2)
        dq_pack = torch.empty_like(q_pack)
        dkc = torch.empty_like(kc_c)
        dvc = torch.empty_like(vc_c)
        dkt = torch.empty_like(kt_c)
        dvt = torch.empty_like(vt_c)

        _eagle3_ttt_gqa_packed_bwd_prep_kernel[(triton.cdiv(M_tot, 64), B * Hkv)](
            out_pack,
            do_pack,
            delta,
            out_pack.stride(0),
            out_pack.stride(1),
            out_pack.stride(2),
            delta.stride(0),
            delta.stride(1),
            M_tot,
            H_KV=Hkv,
            HEAD_DIM=D,
            BLOCK_M=64,
            num_warps=4,
        )
        _eagle3_ttt_gqa_packed_bwd_dkv_ctx_kernel[(triton.cdiv(S, 64), B * Hkv)](
            q_pack,
            kc_c,
            vc_c,
            do_pack,
            lse2,
            delta,
            dkc,
            dvc,
            q_pack.stride(0),
            q_pack.stride(1),
            q_pack.stride(2),
            kc_c.stride(0),
            kc_c.stride(1),
            kc_c.stride(2),
            lse2.stride(0),
            lse2.stride(1),
            S,
            sm_scale,
            H_KV=Hkv,
            K_STEPS=K,
            GQA_GROUP=GQA,
            HEAD_DIM=D,
            BLOCK_N=64,
            BLOCK_M=64,
            num_warps=4,
            num_stages=2,
        )
        _eagle3_ttt_gqa_packed_bwd_dq_dkv_ttt_kernel[(triton.cdiv(M_tot, 64), B * Hkv)](
            q_pack,
            kc_c,
            vc_c,
            kt_c,
            vt_c,
            do_pack,
            lse2,
            delta,
            dq_pack,
            dkt,
            dvt,
            q_pack.stride(0),
            q_pack.stride(1),
            q_pack.stride(2),
            kc_c.stride(0),
            kc_c.stride(1),
            kc_c.stride(2),
            kt_c.stride(0),
            kt_c.stride(1),
            kt_c.stride(2),
            kt_c.stride(3),
            lse2.stride(0),
            lse2.stride(1),
            S,
            sm_scale,
            H_KV=Hkv,
            K_STEPS=K,
            GQA_GROUP=GQA,
            HEAD_DIM=D,
            BLOCK_M=64,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )
        dq = (
            dq_pack.view(B, Hkv, S, K, GQA, D)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(B, S, K, Hq, D)
        )
        return dq, dkc, dvc, dkt, dvt


# ============================================================================
# 5. Suite 0: Shift-Cancellation, Top-K Bias, FP64 Chain-Rule, & GQA Ablation
# ============================================================================


def run_suite_0_audits(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    V_full = 128256
    # 0.1 Cross-Distribution FP32 Shift-Cancellation Audit (V = 128,256)
    gen0 = torch.Generator(device=device).manual_seed(12201)
    R_s = 64
    base_zt = torch.randn((R_s, V_full), generator=gen0, device=device, dtype=torch.float32)
    top_tok = torch.randint(0, V_full, (R_s, 1), generator=gen0, device=device)
    base_zt.scatter_add_(
        1, top_tok, torch.full((R_s, 1), 19.5, device=device, dtype=torch.float32)
    )
    base_zs = base_zt.clone()
    base_zs.scatter_add_(
        1, top_tok, torch.full((R_s, 1), -0.35, device=device, dtype=torch.float32)
    )
    base_zs = base_zs + 0.05 * torch.randn(
        (R_s, V_full), generator=gen0, device=device, dtype=torch.float32
    )

    shift_audit: dict[str, Any] = {}
    wt_ones = torch.ones((R_s,), device=device, dtype=torch.float32)
    for sh in (0.0, 4096.0, 32768.0):
        zs = (base_zs + sh).contiguous()
        zt = (base_zt - sh).contiguous()
        lps64 = F.log_softmax(zs.double(), dim=-1)
        lpt64 = F.log_softmax(zt.double(), dim=-1)
        rkl64 = (lps64.exp() * (lps64 - lpt64)).sum(dim=-1)

        out_naive = torch.empty((R_s,), device=device, dtype=torch.float32)
        _naive_uncentered_rkl_kernel[(R_s,)](
            zs, zs.stride(0), zt, zt.stride(0), out_naive, V_full, 1.0, BLOCK_V=8192, num_warps=16
        )
        zs_cp = zs.clone()
        out_monoid = torch.empty((R_s,), device=device, dtype=torch.float32)
        _coupled_kd_fwd_bwd_inplace_kernel[(R_s,)](
            zs_cp,
            zs_cp.stride(0),
            zt,
            zt.stride(0),
            wt_ones,
            out_monoid,
            V_full,
            1.0,
            1.0,
            0.5,
            MODE=1,
            BLOCK_V=8192,
            num_warps=16,
        )
        shift_audit[f"opposing_shift_{int(sh)}"] = {
            "mean_true_rkl_nats": round(float(rkl64.mean().item()), 6),
            "naive_uncentered_fp32_rel_l2": round(rel_l2(out_naive, rkl64), 8),
            "naive_uncentered_negative_count": int((out_naive < 0.0).sum().item()),
            "naive_uncentered_min_val": round(float(out_naive.min().item()), 6),
            "coupled_monoid_fp32_rel_l2": round(rel_l2(out_monoid, rkl64), 8),
            "coupled_monoid_negative_count": int((out_monoid < 0.0).sum().item()),
        }

    # 0.2 Top-K (K = 32, 64, 128) Truncated Logit-Distillation Student Gradient Bias vs Coupled Monoid
    N_tk, H_tk = 256, 512
    tau_tk = 2.0
    hs_tk = torch.randn((N_tk, H_tk), generator=gen0, device=device, dtype=torch.bfloat16)
    ws_tk = torch.randn((V_full, H_tk), generator=gen0, device=device, dtype=torch.bfloat16) * 0.04
    ht_tk = torch.randn((N_tk, H_tk), generator=gen0, device=device, dtype=torch.bfloat16)
    wt_tk = torch.randn((V_full, H_tk), generator=gen0, device=device, dtype=torch.bfloat16) * 0.06
    w_unif = torch.full((N_tk,), 1.0 / float(N_tk), device=device, dtype=torch.float32)

    # Exact FP32 Reverse-KL reference
    hs_ref = hs_tk.float().detach().requires_grad_(True)
    ws_ref = ws_tk.float().detach().requires_grad_(True)
    with torch.no_grad():
        zt_full = F.linear(ht_tk.float(), wt_tk.float())
    zs_full = F.linear(hs_ref, ws_ref)
    lps_f = F.log_softmax(zs_full / tau_tk, dim=-1)
    lpt_f = F.log_softmax(zt_full / tau_tk, dim=-1)
    loss_rkl_ref = (w_unif * (tau_tk**2) * (lps_f.exp() * (lps_f - lpt_f)).sum(dim=-1)).sum()
    loss_rkl_ref.backward()

    topk_bias_audit: dict[str, Any] = {}
    for K_val in (32, 64, 128):
        hs_k = hs_tk.float().detach().requires_grad_(True)
        ws_k = ws_tk.float().detach().requires_grad_(True)
        with torch.no_grad():
            zt_top_v, zt_top_i = torch.topk(zt_full, K_val, dim=-1)
        zs_all_k = F.linear(hs_k, ws_k)
        zs_top_v = zs_all_k.gather(1, zt_top_i)
        lps_k = F.log_softmax(zs_top_v / tau_tk, dim=-1)
        lpt_k = F.log_softmax(zt_top_v / tau_tk, dim=-1)
        lk = (w_unif * (tau_tk**2) * (lps_k.exp() * (lps_k - lpt_k)).sum(dim=-1)).sum()
        lk.backward()
        topk_bias_audit[f"top_{K_val}"] = {
            "dhs_rel_l2_bias": round(rel_l2(hs_k.grad, hs_ref.grad), 6),
            "dws_rel_l2_bias": round(rel_l2(ws_k.grad, ws_ref.grad), 6),
        }

    # Exact Triton Coupled Monoid BF16 on the exact same inputs
    hs_tri = hs_tk.detach().clone().requires_grad_(True)
    ws_tri = ws_tk.detach().clone().requires_grad_(True)
    loss_tri_rkl = _FusedCoupledLinearKDFn.apply(
        hs_tri, ws_tri, ht_tk, wt_tk, w_unif, tau_tk, 0.5, 1, 256, True
    )
    loss_tri_rkl.backward()
    topk_bias_audit["triton_coupled_monoid_full_vocab_bf16"] = {
        "loss_rel_err": round(
            abs(float(loss_tri_rkl.item()) - float(loss_rkl_ref.item()))
            / abs(float(loss_rkl_ref.item())),
            6,
        ),
        "dhs_rel_l2_err": round(rel_l2(hs_tri.grad, hs_ref.grad), 6),
        "dws_rel_l2_err": round(rel_l2(ws_tri.grad, ws_ref.grad), 6),
    }

    # Also check BF16 parity on FKL and JSD_0.5
    bf16_kd_parity: dict[str, Any] = {"rkl": topk_bias_audit["triton_coupled_monoid_full_vocab_bf16"]}
    for mode_id, mkey in ((0, "fkl"), (2, "jsd_0p5")):
        hs_r2 = hs_tk.float().detach().requires_grad_(True)
        ws_r2 = ws_tk.float().detach().requires_grad_(True)
        zs_r2 = F.linear(hs_r2, ws_r2)
        lps2 = F.log_softmax(zs_r2 / tau_tk, dim=-1)
        ps2, pt2 = lps2.exp(), lpt_f.exp()
        if mode_id == 0:
            lv2 = (pt2 * (lpt_f - lps2)).sum(dim=-1)
        else:
            lm2 = torch.log(0.5 * pt2 + 0.5 * ps2)
            lv2 = 0.5 * (pt2 * (lpt_f - lm2) + ps2 * (lps2 - lm2)).sum(dim=-1)
        lr2 = (w_unif * (tau_tk**2) * lv2).sum()
        lr2.backward()

        hs_t2 = hs_tk.detach().clone().requires_grad_(True)
        ws_t2 = ws_tk.detach().clone().requires_grad_(True)
        lt2 = _FusedCoupledLinearKDFn.apply(
            hs_t2, ws_t2, ht_tk, wt_tk, w_unif, tau_tk, 0.5, mode_id, 256, True
        )
        lt2.backward()
        bf16_kd_parity[mkey] = {
            "loss_rel_err": round(
                abs(float(lt2.item()) - float(lr2.item())) / abs(float(lr2.item())), 6
            ),
            "dhs_rel_l2_err": round(rel_l2(hs_t2.grad, hs_r2.grad), 6),
            "dws_rel_l2_err": round(rel_l2(ws_t2.grad, ws_r2.grad), 6),
        }

    # 0.3 FP64 Analytical Chain-Rule & Richardson Central-Difference (h vs h/2 -> ~4.00x)
    gen64 = torch.Generator(device=device).manual_seed(12209)
    fd_ratios: dict[str, float] = {}

    N64, H64, V64 = 24, 48, 192
    tau64, b64 = 1.7, 0.35
    hs64 = torch.randn((N64, H64), generator=gen64, device=device, dtype=torch.float64) * 0.3
    ws64 = torch.randn((V64, H64), generator=gen64, device=device, dtype=torch.float64) * 0.3
    zt64 = torch.randn((N64, V64), generator=gen64, device=device, dtype=torch.float64)
    wt64 = torch.linspace(0.5, 1.5, N64, device=device, dtype=torch.float64) / float(N64)

    for mkey, mode_id in (("fkl", 0), ("rkl", 1), ("jsd", 2)):
        def _loss_kd64(x_in: torch.Tensor, w_in: torch.Tensor, mid: int = mode_id) -> torch.Tensor:
            zs = (x_in @ w_in.t()) / tau64
            zt_s = zt64 / tau64
            lps = F.log_softmax(zs, dim=-1)
            lpt = F.log_softmax(zt_s, dim=-1)
            ps, pt = lps.exp(), lpt.exp()
            if mid == 0:
                lv = (pt * (lpt - lps)).sum(dim=-1)
            elif mid == 1:
                lv = (ps * (lps - lpt)).sum(dim=-1)
            else:
                lm = torch.log(b64 * pt + (1.0 - b64) * ps)
                lv = b64 * (pt * (lpt - lm)).sum(dim=-1) + (1.0 - b64) * (ps * (lps - lm)).sum(dim=-1)
            return (wt64 * (tau64**2) * lv).sum()

        # Analytical Coupled-Monoid gradient equations in FP64
        zs_a = (hs64 @ ws64.t()) / tau64
        zt_a = zt64 / tau64
        ms_a = zs_a.max(dim=-1, keepdim=True).values
        mt_a = zt_a.max(dim=-1, keepdim=True).values
        ds_a = (zs_a - ms_a).exp().sum(dim=-1, keepdim=True)
        dt_a = (zt_a - mt_a).exp().sum(dim=-1, keepdim=True)
        lse_s_a = ms_a + ds_a.log()
        lse_t_a = mt_a + dt_a.log()
        ls_a = zs_a - lse_s_a
        lt_a = zt_a - lse_t_a
        ps_a, pt_a = ls_a.exp(), lt_a.exp()
        sc_a = (wt64 * tau64).unsqueeze(-1)
        if mode_id == 0:
            du_a = sc_a * (ps_a - pt_a)
        elif mode_id == 1:
            es_a = ((zs_a - ms_a) * (zs_a - ms_a).exp()).sum(dim=-1, keepdim=True)
            cst_a = ((zt_a - mt_a) * (zs_a - ms_a).exp()).sum(dim=-1, keepdim=True)
            rkl_a = (es_a - cst_a) / ds_a - ds_a.log() + dt_a.log()
            du_a = sc_a * ps_a * ((ls_a - lt_a) - rkl_a)
        else:
            lm_a = torch.logaddexp(math.log(b64) + lt_a, math.log(1.0 - b64) + ls_a)
            ksm_a = (ps_a * (ls_a - lm_a)).sum(dim=-1, keepdim=True)
            du_a = sc_a * (1.0 - b64) * ps_a * ((ls_a - lm_a) - ksm_a)
        dh_a = du_a @ ws64
        dw_a = du_a.t() @ hs64

        for idx_i, (nm, bt, ga) in enumerate(
            [(f"{mkey}_dh_s", hs64, dh_a), (f"{mkey}_dw_s", ws64, dw_a)]
        ):
            dv = torch.randn_like(bt, generator=gen64)
            dv = dv / dv.pow(2).mean().sqrt()
            ana = float((ga * dv).sum().item())
            errs = []
            for h_st in (4e-3, 2e-3):
                ip = [hs64.clone(), ws64.clone()]
                im = [hs64.clone(), ws64.clone()]
                ip[idx_i] = ip[idx_i] + h_st * dv
                im[idx_i] = im[idx_i] - h_st * dv
                fd_v = float((_loss_kd64(*ip) - _loss_kd64(*im)).item()) / (2.0 * h_st)
                errs.append(abs(fd_v - ana))
            fd_ratios[nm] = round(errs[0] / max(errs[1], 1e-18), 4)

    # EAGLE-3 TTT FP64 Richardson check
    qt64 = torch.randn((1, 8, 4, 4, 16), generator=gen64, device=device, dtype=torch.float64) * 0.3
    kc64 = torch.randn((1, 8, 2, 16), generator=gen64, device=device, dtype=torch.float64) * 0.3
    vc64 = torch.randn((1, 8, 2, 16), generator=gen64, device=device, dtype=torch.float64) * 0.3
    kt64 = torch.randn((1, 8, 4, 2, 16), generator=gen64, device=device, dtype=torch.float64) * 0.3
    vt64 = torch.randn((1, 8, 4, 2, 16), generator=gen64, device=device, dtype=torch.float64) * 0.3
    do64 = torch.randn_like(qt64, generator=gen64) * 0.3

    def _ttt_loss64(q_i: torch.Tensor, kc_i: torch.Tensor, kt_i: torch.Tensor) -> torch.Tensor:
        kc_rep = kc_i.repeat_interleave(2, dim=2)
        vc_rep = vc64.repeat_interleave(2, dim=2)
        kt_rep = kt_i.repeat_interleave(2, dim=3)
        vt_rep = vt64.repeat_interleave(2, dim=3)
        qp = q_i.permute(0, 3, 1, 2, 4).reshape(1, 4, 32, 16)
        sc = torch.matmul(qp, kc_rep.permute(0, 2, 3, 1)).view(1, 4, 8, 4, 8) * 0.25
        ti = torch.arange(8, device=device)
        sc = sc.masked_fill(~(ti[:, None] >= ti[None, :])[None, None, :, None, :], -1.0e30)
        sch = torch.matmul(
            q_i.permute(0, 1, 3, 2, 4), kt_rep.permute(0, 1, 3, 4, 2)
        ) * 0.25
        ki = torch.arange(4, device=device)
        sch = sch.masked_fill(~(ki[:, None] >= ki[None, :])[None, None, None, :, :]).permute(0, 2, 1, 3, 4) if False else sch.masked_fill(~(ki[:, None] >= ki[None, :])[None, None, None, :, :], -1.0e30).permute(0, 2, 1, 3, 4)
        pa = torch.softmax(torch.cat([sc, sch], dim=-1), dim=-1)
        oc = torch.matmul(pa[..., :8].reshape(1, 4, 32, 8), vc_rep.permute(0, 2, 1, 3)).view(1, 4, 8, 4, 16)
        och = torch.matmul(pa[..., 8:], vt_rep.permute(0, 3, 1, 2, 4))
        out = (oc + och).permute(0, 2, 3, 1, 4)
        return (out * do64).sum()

    qt_r = qt64.detach().clone().requires_grad_(True)
    kc_r = kc64.detach().clone().requires_grad_(True)
    kt_r = kt64.detach().clone().requires_grad_(True)
    _ttt_loss64(qt_r, kc_r, kt_r).backward()
    for idx_i, (nm, bt, ga) in enumerate(
        [("ttt_dq", qt64, qt_r.grad), ("ttt_dk_ctx", kc64, kc_r.grad), ("ttt_dk_ttt", kt64, kt_r.grad)]
    ):
        dv = torch.randn_like(bt, generator=gen64)
        dv = dv / dv.pow(2).mean().sqrt()
        ana = float((ga * dv).sum().item())
        errs = []
        for h_st in (4e-3, 2e-3):
            args_p = [qt64.clone(), kc64.clone(), kt64.clone()]
            args_m = [qt64.clone(), kc64.clone(), kt64.clone()]
            args_p[idx_i] = args_p[idx_i] + h_st * dv
            args_m[idx_i] = args_m[idx_i] - h_st * dv
            fd_v = float((_ttt_loss64(*args_p) - _ttt_loss64(*args_m)).item()) / (2.0 * h_st)
            errs.append(abs(fd_v - ana))
        fd_ratios[nm] = round(errs[0] / max(errs[1], 1e-18), 4)

    # 0.4 GQA-Packed vs Unpacked EAGLE-3 TTT Kernel Ablation & BF16 Parity (B=2, S=2048, K=4, 16Q/4KV, D=128)
    B_ab, S_ab, K_ab, Hq_ab, Hkv_ab, D_ab = 2, (1024 if is_smoke else 2048), 4, 16, 4, 128
    sm_ab = 1.0 / math.sqrt(D_ab)
    q_ab = torch.randn((B_ab, S_ab, K_ab, Hq_ab, D_ab), device=device, dtype=torch.bfloat16)
    kc_ab = torch.randn((B_ab, S_ab, Hkv_ab, D_ab), device=device, dtype=torch.bfloat16)
    vc_ab = torch.randn((B_ab, S_ab, Hkv_ab, D_ab), device=device, dtype=torch.bfloat16)
    kt_ab = torch.randn((B_ab, S_ab, K_ab, Hkv_ab, D_ab), device=device, dtype=torch.bfloat16)
    vt_ab = torch.randn((B_ab, S_ab, K_ab, Hkv_ab, D_ab), device=device, dtype=torch.bfloat16)
    do_ab = torch.randn_like(q_ab)

    # Parity on B=2, S=256
    qp_s = q_ab[:, :256].detach().clone().requires_grad_(True)
    kcp_s = kc_ab[:, :256].detach().clone().requires_grad_(True)
    vcp_s = vc_ab[:, :256].detach().clone().requires_grad_(True)
    ktp_s = kt_ab[:, :256].detach().clone().requires_grad_(True)
    vtp_s = vt_ab[:, :256].detach().clone().requires_grad_(True)
    dop_s = do_ab[:, :256].detach().clone()
    out_ref_s = eagle3_ttt_ref_fp32(qp_s, kcp_s, vcp_s, ktp_s, vtp_s, sm_ab)
    (out_ref_s * dop_s.float()).sum().backward()

    qt_s = q_ab[:, :256].detach().clone().requires_grad_(True)
    kct_s = kc_ab[:, :256].detach().clone().requires_grad_(True)
    vct_s = vc_ab[:, :256].detach().clone().requires_grad_(True)
    ktt_s = kt_ab[:, :256].detach().clone().requires_grad_(True)
    vtt_s = vt_ab[:, :256].detach().clone().requires_grad_(True)
    out_tri_s = _Eagle3TTTGQAPackedFn.apply(qt_s, kct_s, vct_s, ktt_s, vtt_s)
    out_tri_s.backward(dop_s)

    ttt_parity = {
        "out_rel_l2": round(rel_l2(out_tri_s, out_ref_s), 6),
        "dq_rel_l2": round(rel_l2(qt_s.grad, qp_s.grad), 6),
        "dk_ctx_rel_l2": round(rel_l2(kct_s.grad, kcp_s.grad), 6),
        "dv_ctx_rel_l2": round(rel_l2(vct_s.grad, vcp_s.grad), 6),
        "dk_ttt_rel_l2": round(rel_l2(ktt_s.grad, ktp_s.grad), 6),
        "dv_ttt_rel_l2": round(rel_l2(vtt_s.grad, vtp_s.grad), 6),
    }

    # Ablate unpacked dQ vs GQA-packed dQ on (B_ab, S_ab)
    q_fl_u = q_ab.view(B_ab, S_ab * K_ab, Hq_ab, D_ab).contiguous()
    kt_fl_u = kt_ab.view(B_ab, S_ab * K_ab, Hkv_ab, D_ab).contiguous()
    vt_fl_u = vt_ab.view(B_ab, S_ab * K_ab, Hkv_ab, D_ab).contiguous()
    do_fl_u = do_ab.view(B_ab, S_ab * K_ab, Hq_ab, D_ab).contiguous()
    lse_u = torch.zeros((B_ab, Hq_ab, S_ab * K_ab), device=device, dtype=torch.float32)
    del_u = torch.zeros_like(lse_u)
    dq_u = torch.empty_like(q_fl_u)
    dkt_u = torch.empty_like(kt_fl_u)
    dvt_u = torch.empty_like(vt_fl_u)

    def _run_unpacked_dq() -> None:
        _eagle3_ttt_unpacked_bwd_dq_kernel[(triton.cdiv(S_ab * K_ab, 64), B_ab * Hkv_ab)](
            q_fl_u,
            kc_ab,
            vc_ab,
            kt_fl_u,
            vt_fl_u,
            do_fl_u,
            lse_u,
            del_u,
            dq_u,
            dkt_u,
            dvt_u,
            q_fl_u.stride(0),
            q_fl_u.stride(1),
            q_fl_u.stride(2),
            kc_ab.stride(0),
            kc_ab.stride(1),
            kc_ab.stride(2),
            kt_fl_u.stride(0),
            kt_fl_u.stride(1),
            kt_fl_u.stride(2),
            lse_u.stride(0),
            lse_u.stride(1),
            S_ab,
            sm_ab,
            H_KV=Hkv_ab,
            GQA_GROUP=4,
            HEAD_DIM=D_ab,
            BLOCK_M=64,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )

    M_p = S_ab * K_ab * 4
    q_pk = torch.randn((B_ab, Hkv_ab, M_p, D_ab), device=device, dtype=torch.bfloat16)
    do_pk = torch.randn_like(q_pk)
    lse_pk = torch.zeros((B_ab, Hkv_ab, M_p), device=device, dtype=torch.float32)
    del_pk = torch.zeros_like(lse_pk)
    dq_pk = torch.empty_like(q_pk)
    dkt_pk = torch.empty_like(kt_ab)
    dvt_pk = torch.empty_like(vt_ab)

    def _run_packed_dq() -> None:
        _eagle3_ttt_gqa_packed_bwd_dq_dkv_ttt_kernel[(triton.cdiv(M_p, 64), B_ab * Hkv_ab)](
            q_pk,
            kc_ab,
            vc_ab,
            kt_ab,
            vt_ab,
            do_pk,
            lse_pk,
            del_pk,
            dq_pk,
            dkt_pk,
            dvt_pk,
            q_pk.stride(0),
            q_pk.stride(1),
            q_pk.stride(2),
            kc_ab.stride(0),
            kc_ab.stride(1),
            kc_ab.stride(2),
            kt_ab.stride(0),
            kt_ab.stride(1),
            kt_ab.stride(2),
            kt_ab.stride(3),
            lse_pk.stride(0),
            lse_pk.stride(1),
            S_ab,
            sm_ab,
            H_KV=Hkv_ab,
            K_STEPS=4,
            GQA_GROUP=4,
            HEAD_DIM=D_ab,
            BLOCK_M=64,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )

    t_unp = time_fn_with_peak_mem(_run_unpacked_dq, warmup=3, iters=10, device=device)
    t_pk = time_fn_with_peak_mem(_run_packed_dq, warmup=3, iters=10, device=device)

    # Dual-RMSNorm BF16 parity
    h_rm = torch.randn((512, 2048), device=device, dtype=torch.bfloat16, requires_grad=True)
    e_rm = torch.randn((512, 2048), device=device, dtype=torch.bfloat16, requires_grad=True)
    wh_rm = torch.ones((2048,), device=device, dtype=torch.bfloat16, requires_grad=True)
    we_rm = torch.ones((2048,), device=device, dtype=torch.bfloat16, requires_grad=True)
    do_2h = torch.randn((512, 4096), device=device, dtype=torch.bfloat16)

    def _ref_dual_rmsnorm(hi: torch.Tensor, ei: torch.Tensor, whi: torch.Tensor, wei: torch.Tensor) -> torch.Tensor:
        hf, ef = hi.float(), ei.float()
        yh = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * whi.float()
        ye = ef * torch.rsqrt(ef.pow(2).mean(-1, keepdim=True) + 1e-6) * wei.float()
        return torch.cat([yh, ye], dim=-1)

    hr_c = h_rm.detach().clone().requires_grad_(True)
    er_c = e_rm.detach().clone().requires_grad_(True)
    whr_c = wh_rm.detach().clone().requires_grad_(True)
    wer_c = we_rm.detach().clone().requires_grad_(True)
    out_rm_ref = _ref_dual_rmsnorm(hr_c, er_c, whr_c, wer_c)
    (out_rm_ref * do_2h.float()).sum().backward()

    out_rm_tri = _FusedDualRMSNormConcatFn.apply(h_rm, e_rm, wh_rm, we_rm, 1e-6)
    out_rm_tri.backward(do_2h)
    dual_rmsnorm_parity = {
        "out_rel_l2": round(rel_l2(out_rm_tri, out_rm_ref), 6),
        "dh_rel_l2": round(rel_l2(h_rm.grad, hr_c.grad), 6),
        "de_rel_l2": round(rel_l2(e_rm.grad, er_c.grad), 6),
        "dwh_rel_l2": round(rel_l2(wh_rm.grad, whr_c.grad), 6),
    }

    gc.collect()
    torch.cuda.empty_cache()
    return {
        "fp32_cross_distribution_shift_audit": shift_audit,
        "topk_truncation_gradient_bias_audit": topk_bias_audit,
        "bf16_kd_parity": bf16_kd_parity,
        "fp64_chain_rule_refinement": {
            "ratios_h_vs_half_h": fd_ratios,
            "min_refinement_ratio": min(fd_ratios.values()),
        },
        "eagle3_ttt_parity_and_gqa_packing_ablation": {
            "seq_len": S_ab,
            "bf16_parity": ttt_parity,
            "bwd_dq_unpacked_4x_kv_reload_ms": t_unp["median_ms"],
            "bwd_dq_gqa_packed_single_kv_load_ms": t_pk["median_ms"],
            "gqa_packing_bwd_dq_speedup": round(
                t_unp["median_ms"] / max(t_pk["median_ms"], 1e-6), 4
            ),
        },
        "dual_rmsnorm_bf16_parity": dual_rmsnorm_parity,
    }


# ============================================================================
# 6. Suite A: Single-GPU SLM Terminal Distillation & DeepSeek-V3 Shared-Head MTP
# ============================================================================


def eager_materialized_rkl_loss(
    h_s: torch.Tensor,
    w_s: torch.Tensor,
    h_t: torch.Tensor,
    w_t: torch.Tensor,
    w_tok: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    with torch.no_grad():
        zt = F.linear(h_t, w_t).float() / tau
        lpt = F.log_softmax(zt, dim=-1)
    zs = F.linear(h_s, w_s).float() / tau
    lps = F.log_softmax(zs, dim=-1)
    rkl = (lps.exp() * (lps - lpt)).sum(dim=-1)
    return (w_tok * (tau * tau) * rkl).sum()


def chunked_pytorch_rkl_loss(
    h_s: torch.Tensor,
    w_s: torch.Tensor,
    h_t: torch.Tensor,
    w_t: torch.Tensor,
    w_tok: torch.Tensor,
    tau: float,
    chunk_size: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for st in range(0, h_s.shape[0], chunk_size):
        en = min(h_s.shape[0], st + chunk_size)
        losses.append(
            eager_materialized_rkl_loss(
                h_s[st:en], w_s, h_t[st:en], w_t, w_tok[st:en], tau
            )
        )
    return torch.stack(losses).sum()


def run_suite_a_slm_distillation_and_mtp(
    device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    H_t, H_s, V_a = 4096, 1536, 128256
    tau_a = 2.0
    chunk_a = 1024
    token_counts = (2048, 8192) if is_smoke else (2048, 4096, 8192, 16384)
    warmup_a = 2 if is_smoke else 3
    iters_a = 4 if is_smoke else 8

    w_t = torch.randn((V_a, H_t), device=device, dtype=torch.bfloat16) * 0.02
    w_s = nn.Parameter(torch.randn((V_a, H_s), device=device, dtype=torch.bfloat16) * 0.02)
    compiled_rkl = torch.compile(eager_materialized_rkl_loss)

    kd_scaling: dict[str, Any] = {}
    for N in token_counts:
        ht_in = torch.randn((N, H_t), device=device, dtype=torch.bfloat16)
        hs_base = torch.randn((N, H_s), device=device, dtype=torch.bfloat16)
        wt_in = torch.full((N,), 1.0 / float(N), device=device, dtype=torch.float32)

        # Total GEMM FLOPs = Teacher Fwd (2 N H_t V) + Student Fwd/dX/dW (6 N H_s V)
        gemm_flops = 2.0 * float(N) * float(V_a) * float(H_t + 3 * H_s)
        modes_n: dict[str, Any] = {}

        mode_defs: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = []
        if N <= 8192:
            mode_defs.append(
                (
                    "eager_fp32_rkl_materialized",
                    lambda xin: eager_materialized_rkl_loss(xin, w_s, ht_in, w_t, wt_in, tau_a),
                )
            )
            mode_defs.append(
                (
                    "torch_compile_rkl_materialized",
                    lambda xin: compiled_rkl(xin, w_s, ht_in, w_t, wt_in, tau_a),
                )
            )
        mode_defs.extend(
            [
                (
                    "chunked_pytorch_rkl_c1024",
                    lambda xin: chunked_pytorch_rkl_loss(
                        xin, w_s, ht_in, w_t, wt_in, tau_a, chunk_a
                    ),
                ),
                (
                    "triton_coupled_monoid_fkl_bf16dw",
                    lambda xin: _FusedCoupledLinearKDFn.apply(
                        xin, w_s, ht_in, w_t, wt_in, tau_a, 0.5, 0, chunk_a, False
                    ),
                ),
                (
                    "triton_coupled_monoid_rkl_bf16dw",
                    lambda xin: _FusedCoupledLinearKDFn.apply(
                        xin, w_s, ht_in, w_t, wt_in, tau_a, 0.5, 1, chunk_a, False
                    ),
                ),
                (
                    "triton_coupled_monoid_rkl_fp32dw",
                    lambda xin: _FusedCoupledLinearKDFn.apply(
                        xin, w_s, ht_in, w_t, wt_in, tau_a, 0.5, 1, chunk_a, True
                    ),
                ),
                (
                    "triton_coupled_monoid_jsd_bf16dw",
                    lambda xin: _FusedCoupledLinearKDFn.apply(
                        xin, w_s, ht_in, w_t, wt_in, tau_a, 0.5, 2, chunk_a, False
                    ),
                ),
            ]
        )

        for mk, fn_m in mode_defs:
            hs_run = hs_base.detach().clone().requires_grad_(True)
            grad_steady = hs_run.numel() * 2 + w_s.numel() * 2

            def _step_m(f_call: Any = fn_m) -> None:
                w_s.grad = None
                hs_run.grad = None
                loss = f_call(hs_run)
                loss.backward()

            tm = time_fn_with_peak_mem(
                _step_m, warmup=warmup_a, iters=iters_a, device=device, grad_bytes_steady=grad_steady
            )
            tm["effective_gemm_tflops"] = round(
                (gemm_flops / (tm["median_ms"] * 1e-3)) / 1e12, 2
            )
            modes_n[mk] = tm
            del hs_run
            w_s.grad = None
            gc.collect()
            torch.cuda.empty_cache()

        kd_scaling[str(N)] = {
            "num_tokens": N,
            "teacher_hidden": H_t,
            "student_hidden": H_s,
            "vocab_size": V_a,
            "modes": modes_n,
        }
        del ht_in, hs_base, wt_in
        gc.collect()
        torch.cuda.empty_cache()

    del w_t, w_s
    gc.collect()
    torch.cuda.empty_cache()

    # A.2 DeepSeek-V3 Sequential Multi-Token Prediction (1 Main + D=3 MTP Heads = 4 Shared Heads!)
    N_mtp = 2048 if is_smoke else 4096
    H_mtp, V_mtp, D_mtp = 2048, 128256, 3
    lambda_mtp = 0.1  # DeepSeek-V3 auxiliary MTP weight (w_k = 0.1 / 3 per MTP depth!)

    gen_m = torch.Generator(device=device).manual_seed(12233)
    w_unembed = nn.Parameter(
        torch.randn((V_mtp, H_mtp), generator=gen_m, device=device, dtype=torch.bfloat16) * 0.02
    )
    proj_weights = nn.ParameterList(
        [
            nn.Parameter(
                torch.randn((H_mtp, 2 * H_mtp), generator=gen_m, device=device, dtype=torch.bfloat16)
                * 0.02
            )
            for _ in range(D_mtp)
        ]
    )
    norm_wh = nn.ParameterList(
        [nn.Parameter(torch.ones((H_mtp,), device=device, dtype=torch.bfloat16)) for _ in range(D_mtp)]
    )
    norm_we = nn.ParameterList(
        [nn.Parameter(torch.ones((H_mtp,), device=device, dtype=torch.bfloat16)) for _ in range(D_mtp)]
    )

    h_main_base = torch.randn((N_mtp, H_mtp), generator=gen_m, device=device, dtype=torch.bfloat16)
    emb_depths = [
        torch.randn((N_mtp, H_mtp), generator=gen_m, device=device, dtype=torch.bfloat16)
        for _ in range(D_mtp)
    ]
    tgts_depths = [
        torch.randint(0, V_mtp, (N_mtp,), generator=gen_m, device=device)
        for _ in range(1 + D_mtp)
    ]

    # Precision Audit: Sub-ULP Auxiliary MTP dW_unembed retention when 3 depth heads
    # (weight = lambda_mtp / (3 * N), ~60x smaller than main head!) accumulate onto 1.0 * dW_main!
    h_main_t = torch.randn((512, H_mtp), generator=gen_m, device=device, dtype=torch.bfloat16)
    t_main_t = torch.randint(0, V_mtp, (512,), generator=gen_m, device=device)
    h_aux_test = [
        torch.randn((512, H_mtp), generator=gen_m, device=device, dtype=torch.bfloat16)
        for _ in range(D_mtp)
    ]
    t_aux_test = [
        torch.randint(0, V_mtp, (512,), generator=gen_m, device=device)
        for _ in range(D_mtp)
    ]
    w_main_vec = torch.full((512,), 1.0 / 512.0, device=device, dtype=torch.float32)
    w_aux_val = 0.05 / float(D_mtp * 512)
    w_aux_vec = torch.full((512,), w_aux_val, device=device, dtype=torch.float32)

    # Exact FP32 reference for auxiliary component alone & main-only BF16 baseline
    w_u_main = w_unembed.detach().clone().requires_grad_(True)
    _SharedHeadLinearCEFn.apply(h_main_t, w_u_main, t_main_t, w_main_vec, 512, True).backward()
    dw_main_bf16 = w_u_main.grad.detach().float()

    w_u_ref32 = w_unembed.detach().float().requires_grad_(True)
    l_tot_ref = (
        w_main_vec
        * F.cross_entropy(F.linear(h_main_t.float(), w_u_ref32), t_main_t, reduction="none")
    ).sum() + sum(
        (
            w_aux_vec
            * F.cross_entropy(F.linear(h_aux_test[d].float(), w_u_ref32), t_aux_test[d], reduction="none")
        ).sum()
        for d in range(D_mtp)
    )
    l_tot_ref.backward()
    w_u_aux_only32 = w_unembed.detach().float().requires_grad_(True)
    sum(
        (
            w_aux_vec
            * F.cross_entropy(
                F.linear(h_aux_test[d].float(), w_u_aux_only32), t_aux_test[d], reduction="none"
            )
        ).sum()
        for d in range(D_mtp)
    ).backward()

    # Sequential 4x BF16 AccumulateGrad (1 main + 3 small MTP heads added sequentially in BF16!)
    w_u_seq = w_unembed.detach().clone().requires_grad_(True)
    l_seq = _SharedHeadLinearCEFn.apply(
        h_main_t, w_u_seq, t_main_t, w_main_vec, 512, False
    ) + sum(
        _SharedHeadLinearCEFn.apply(
            h_aux_test[d], w_u_seq, t_aux_test[d], w_aux_vec, 512, False
        )
        for d in range(D_mtp)
    )
    l_seq.backward()

    # Batched 1x Unified FP32 Accumulator across [main ; aux_0 ; aux_1 ; aux_2]
    w_u_bat = w_unembed.detach().clone().requires_grad_(True)
    l_bat = _SharedHeadLinearCEFn.apply(
        torch.cat([h_main_t, *h_aux_test], dim=0),
        w_u_bat,
        torch.cat([t_main_t, *t_aux_test], dim=0),
        torch.cat([w_main_vec, w_aux_vec.repeat(D_mtp)], dim=0),
        512,
        True,
    )
    l_bat.backward()
    mtp_aux_dw_precision_audit = {
        "sequential_bf16_accum_aux_signal_rel_l2": round(
            rel_l2(w_u_seq.grad.float() - dw_main_bf16, w_u_aux_only32.grad), 6
        ),
        "batched_shared_fp32_accum_aux_signal_rel_l2": round(
            rel_l2(w_u_bat.grad.float() - dw_main_bf16, w_u_aux_only32.grad), 6
        ),
        "sequential_bf16_accum_total_dw_rel_l2": round(rel_l2(w_u_seq.grad, w_u_ref32.grad), 6),
        "batched_shared_fp32_accum_total_dw_rel_l2": round(rel_l2(w_u_bat.grad, w_u_ref32.grad), 6),
    }
    del w_u_main, w_u_ref32, w_u_aux_only32, w_u_seq, w_u_bat

    all_mtp_params = [w_unembed, *proj_weights, *norm_wh, *norm_we]
    grad_bytes_mtp = sum(p.numel() * 2 for p in all_mtp_params) + h_main_base.numel() * 2
    wts_all_4 = torch.cat(
        [
            torch.full((N_mtp,), 1.0 / float(N_mtp), device=device, dtype=torch.float32),
            *[
                torch.full(
                    (N_mtp,),
                    lambda_mtp / float(D_mtp * N_mtp),
                    device=device,
                    dtype=torch.float32,
                )
                for _ in range(D_mtp)
            ],
        ],
        dim=0,
    )
    tgts_all_4 = torch.cat(tgts_depths, dim=0)

    mtp_modes: dict[str, Any] = {}

    # Mode 1: Eager 4-head materialized (RMSNorm + torch.cat + Linear + FP32 CE)
    hm1 = h_main_base.detach().clone().requires_grad_(True)

    def _run_mtp_eager() -> None:
        for p in all_mtp_params:
            p.grad = None
        hm1.grad = None
        heads = [hm1]
        cur = hm1
        for d in range(D_mtp):
            nh = (cur.float() * torch.rsqrt(cur.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(
                torch.bfloat16
            ) * norm_wh[d]
            ne = (
                emb_depths[d].float()
                * torch.rsqrt(emb_depths[d].float().pow(2).mean(-1, keepdim=True) + 1e-6)
            ).to(torch.bfloat16) * norm_we[d]
            cat_2h = torch.cat([nh, ne], dim=-1)
            cur = F.silu(F.linear(cat_2h, proj_weights[d]))
            heads.append(cur)
        loss = F.cross_entropy(
            F.linear(heads[0], w_unembed).float(), tgts_depths[0], reduction="mean"
        )
        for d in range(D_mtp):
            loss = loss + (lambda_mtp / float(D_mtp)) * F.cross_entropy(
                F.linear(heads[1 + d], w_unembed).float(), tgts_depths[1 + d], reduction="mean"
            )
        loss.backward()

    mtp_modes["mtp_eager_4head_materialized"] = time_fn_with_peak_mem(
        _run_mtp_eager, warmup=warmup_a, iters=iters_a, device=device, grad_bytes_steady=grad_bytes_mtp
    )
    del hm1
    for p in all_mtp_params:
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()

    # Mode 2: Sequential 4x single-head Triton CE + unfused RMSNorm/cat
    hm2 = h_main_base.detach().clone().requires_grad_(True)

    def _run_mtp_seq_triton() -> None:
        for p in all_mtp_params:
            p.grad = None
        hm2.grad = None
        heads = [hm2]
        cur = hm2
        for d in range(D_mtp):
            nh = (cur.float() * torch.rsqrt(cur.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(
                torch.bfloat16
            ) * norm_wh[d]
            ne = (
                emb_depths[d].float()
                * torch.rsqrt(emb_depths[d].float().pow(2).mean(-1, keepdim=True) + 1e-6)
            ).to(torch.bfloat16) * norm_we[d]
            cur = F.silu(F.linear(torch.cat([nh, ne], dim=-1), proj_weights[d]))
            heads.append(cur)
        loss = sum(
            _SharedHeadLinearCEFn.apply(
                heads[k],
                w_unembed,
                tgts_depths[k],
                wts_all_4[k * N_mtp : (k + 1) * N_mtp],
                chunk_a,
                False,
            )
            for k in range(1 + D_mtp)
        )
        loss.backward()

    mtp_modes["mtp_sequential_4x_triton_bf16dw"] = time_fn_with_peak_mem(
        _run_mtp_seq_triton,
        warmup=warmup_a,
        iters=iters_a,
        device=device,
        grad_bytes_steady=grad_bytes_mtp,
    )
    del hm2
    for p in all_mtp_params:
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()

    # Mode 3 (Ours): Fused Dual-RMSNorm-Concat + Batched [4*N, H] Shared-W_unembed Single-FP32-dW Triton MTP!
    hm3 = h_main_base.detach().clone().requires_grad_(True)

    def _run_mtp_fused_batched() -> None:
        for p in all_mtp_params:
            p.grad = None
        hm3.grad = None
        heads = [hm3]
        cur = hm3
        for d in range(D_mtp):
            cat_2h = _FusedDualRMSNormConcatFn.apply(
                cur, emb_depths[d], norm_wh[d], norm_we[d], 1e-6
            )
            cur = F.silu(F.linear(cat_2h, proj_weights[d]))
            heads.append(cur)
        h_packed = torch.cat(heads, dim=0)
        loss = _SharedHeadLinearCEFn.apply(
            h_packed, w_unembed, tgts_all_4, wts_all_4, chunk_a, True
        )
        loss.backward()

    mtp_modes["mtp_fused_dual_rmsnorm_batched_shared_fp32dw"] = time_fn_with_peak_mem(
        _run_mtp_fused_batched,
        warmup=warmup_a,
        iters=iters_a,
        device=device,
        grad_bytes_steady=grad_bytes_mtp,
    )
    del hm3, w_unembed, proj_weights, norm_wh, norm_we
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "slm_distillation_vocab_scaling": kd_scaling,
        "deepseek_v3_mtp_shared_head": {
            "tokens_per_head": N_mtp,
            "num_heads_total": 1 + D_mtp,
            "hidden": H_mtp,
            "vocab_size": V_mtp,
            "aux_dw_precision_audit": mtp_aux_dw_precision_audit,
            "modes": mtp_modes,
        },
    }


# ============================================================================
# 7. Suite B: End-to-End EAGLE-3 Multi-Step TTT Draft Layer Scaling (K = 4)
# ============================================================================


class Eagle3TTTDraftLayer(nn.Module):
    def __init__(self, hidden: int, inter: int, n_q: int, n_kv: int, head_dim: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.fc_fuse = nn.Linear(2 * hidden, hidden, bias=False)
        self.ctx_kv = nn.Linear(hidden, 2 * n_kv * head_dim, bias=False)
        self.ttt_q = nn.Linear(hidden, n_q * head_dim, bias=False)
        self.ttt_kv = nn.Linear(hidden, 2 * n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q * head_dim, hidden, bias=False)
        self.gate_up = nn.Linear(hidden, 2 * inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.gate_up(x)
        g, u = gu.chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)

    def forward_sdpa(
        self, h_ctx: torch.Tensor, h_ttt_2h: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        B, S, _ = h_ctx.shape
        K = h_ttt_2h.shape[2]
        kv_c = self.ctx_kv(h_ctx).view(B, S, 2, self.n_kv, self.head_dim)
        kc, vc = kv_c[:, :, 0].permute(0, 2, 1, 3), kv_c[:, :, 1].permute(0, 2, 1, 3)

        x_t = self.fc_fuse(h_ttt_2h)  # [B, S, K, H]
        q_t = (
            self.ttt_q(x_t)
            .view(B, S, K, self.n_q, self.head_dim)
            .permute(0, 3, 2, 1, 4)
            .reshape(B, self.n_q, K * S, self.head_dim)
        )
        kv_t = self.ttt_kv(x_t).view(B, S, K, 2, self.n_kv, self.head_dim)
        kt = kv_t[:, :, :, 0].permute(0, 3, 2, 1, 4).reshape(B, self.n_kv, K * S, self.head_dim)
        vt = kv_t[:, :, :, 1].permute(0, 3, 2, 1, 4).reshape(B, self.n_kv, K * S, self.head_dim)

        k_all = torch.cat([kc, kt], dim=2)
        v_all = torch.cat([vc, vt], dim=2)
        o = F.scaled_dot_product_attention(
            q_t, k_all, v_all, attn_mask=attn_mask, enable_gqa=True
        )
        o_unp = o.view(B, self.n_q, K, S, self.head_dim).permute(0, 3, 2, 1, 4).reshape(B, S, K, -1)
        return self._mlp(x_t + self.o_proj(o_unp))

    def forward_flex(
        self, h_ctx: torch.Tensor, h_ttt_2h: torch.Tensor, bm: Any, comp_flex: Any
    ) -> torch.Tensor:
        B, S, _ = h_ctx.shape
        K = h_ttt_2h.shape[2]
        kv_c = self.ctx_kv(h_ctx).view(B, S, 2, self.n_kv, self.head_dim)
        kc, vc = kv_c[:, :, 0].permute(0, 2, 1, 3), kv_c[:, :, 1].permute(0, 2, 1, 3)

        x_t = self.fc_fuse(h_ttt_2h)
        q_t = (
            self.ttt_q(x_t)
            .view(B, S, K, self.n_q, self.head_dim)
            .permute(0, 3, 2, 1, 4)
            .reshape(B, self.n_q, K * S, self.head_dim)
        )
        kv_t = self.ttt_kv(x_t).view(B, S, K, 2, self.n_kv, self.head_dim)
        kt = kv_t[:, :, :, 0].permute(0, 3, 2, 1, 4).reshape(B, self.n_kv, K * S, self.head_dim)
        vt = kv_t[:, :, :, 1].permute(0, 3, 2, 1, 4).reshape(B, self.n_kv, K * S, self.head_dim)

        k_all = torch.cat([kc, kt], dim=2)
        v_all = torch.cat([vc, vt], dim=2)
        o = comp_flex(q_t, k_all, v_all, block_mask=bm, enable_gqa=True)
        o_unp = o.view(B, self.n_q, K, S, self.head_dim).permute(0, 3, 2, 1, 4).reshape(B, S, K, -1)
        return self._mlp(x_t + self.o_proj(o_unp))

    def forward_triton_two_region(
        self, h_ctx: torch.Tensor, h_ttt_2h: torch.Tensor
    ) -> torch.Tensor:
        B, S, _ = h_ctx.shape
        K = h_ttt_2h.shape[2]
        kv_c = self.ctx_kv(h_ctx).view(B, S, 2, self.n_kv, self.head_dim)
        kc, vc = kv_c[:, :, 0], kv_c[:, :, 1]

        x_t = self.fc_fuse(h_ttt_2h)
        q_t = self.ttt_q(x_t).view(B, S, K, self.n_q, self.head_dim)
        kv_t = self.ttt_kv(x_t).view(B, S, K, 2, self.n_kv, self.head_dim)
        kt, vt = kv_t[:, :, :, 0], kv_t[:, :, :, 1]

        o = _Eagle3TTTGQAPackedFn.apply(q_t, kc, vc, kt, vt).reshape(B, S, K, -1)
        return self._mlp(x_t + self.o_proj(o))


def run_suite_b_eagle3_ttt_layer(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    B_b, K_b = 2, 4
    H_b, I_b, nq_b, nkv_b = 2048, 5632, 16, 4
    seq_lens = (512, 2048) if is_smoke else (512, 1024, 2048, 4096)
    warmup_b = 2 if is_smoke else 3
    iters_b = 4 if is_smoke else 8

    layer = Eagle3TTTDraftLayer(H_b, I_b, nq_b, nkv_b, head_dim=128).to(
        device=device, dtype=torch.bfloat16
    )
    comp_flex = torch.compile(flex_attention)
    p_bytes = sum(p.numel() * 2 for p in layer.parameters())

    scaling_b: dict[str, Any] = {}
    for S in seq_lens:
        hc_base = torch.randn((B_b, S, H_b), device=device, dtype=torch.bfloat16)
        ht_base = torch.randn((B_b, S, K_b, 2 * H_b), device=device, dtype=torch.bfloat16)
        do_b = torch.randn((B_b, S, K_b, H_b), device=device, dtype=torch.bfloat16)
        steady_g = p_bytes + hc_base.numel() * 2 + ht_base.numel() * 2

        modes_s: dict[str, Any] = {}

        # Mode 1: SDPA with explicit [1, 1, 4*S, 5*S] staircase boolean mask (S <= 2048; OOM/glacial at 4096)
        if S <= 2048:
            q_idx = torch.arange(K_b * S, device=device)[:, None]
            kv_idx = torch.arange((1 + K_b) * S, device=device)[None, :]
            q_st, q_po = q_idx // S, q_idx % S
            is_c = kv_idx < S
            kv_st, kv_po = (kv_idx - S) // S, (kv_idx - S) % S
            mask_bool = ((is_c & (kv_idx <= q_po)) | ((~is_c) & (kv_st <= q_st) & (kv_po == q_po)))[
                None, None, :, :
            ]
            hc1 = hc_base.detach().clone().requires_grad_(True)
            ht1 = ht_base.detach().clone().requires_grad_(True)

            def _run_sdpa() -> None:
                for p in layer.parameters():
                    p.grad = None
                hc1.grad = ht1.grad = None
                out = layer.forward_sdpa(hc1, ht1, mask_bool)
                out.backward(do_b)

            modes_s["eagle3_sdpa_staircase_mask"] = time_fn_with_peak_mem(
                _run_sdpa, warmup=warmup_b, iters=iters_b, device=device, grad_bytes_steady=steady_g
            )
            del hc1, ht1, mask_bool
            gc.collect()
            torch.cuda.empty_cache()

        # Mode 2: FlexAttention BlockMask on [4*S, 5*S]
        S_val = S

        def _mask_mod(b: Any, h: Any, qi: Any, kvi: Any) -> Any:
            qs = qi // S_val
            qp = qi % S_val
            ic = kvi < S_val
            ks = (kvi - S_val) // S_val
            kp = (kvi - S_val) % S_val
            return (ic & (kvi <= qp)) | ((~ic) & (ks <= qs) & (kp == qp))

        bm = create_block_mask(
            _mask_mod, B=None, H=None, Q_LEN=K_b * S, KV_LEN=(1 + K_b) * S, device=device
        )
        hc2 = hc_base.detach().clone().requires_grad_(True)
        ht2 = ht_base.detach().clone().requires_grad_(True)

        def _run_flex() -> None:
            for p in layer.parameters():
                p.grad = None
            hc2.grad = ht2.grad = None
            out = layer.forward_flex(hc2, ht2, bm, comp_flex)
            out.backward(do_b)

        modes_s["eagle3_flex_blockmask_5s"] = time_fn_with_peak_mem(
            _run_flex, warmup=warmup_b, iters=iters_b, device=device, grad_bytes_steady=steady_g
        )
        del hc2, ht2, bm
        gc.collect()
        torch.cuda.empty_cache()

        # Mode 3 (Ours): GQA-Packed Two-Region Triton EAGLE-3 TTT
        hc3 = hc_base.detach().clone().requires_grad_(True)
        ht3 = ht_base.detach().clone().requires_grad_(True)

        def _run_tri_ttt() -> None:
            for p in layer.parameters():
                p.grad = None
            hc3.grad = ht3.grad = None
            out = layer.forward_triton_two_region(hc3, ht3)
            out.backward(do_b)

        modes_s["eagle3_triton_gqa_packed_two_region"] = time_fn_with_peak_mem(
            _run_tri_ttt, warmup=warmup_b, iters=iters_b, device=device, grad_bytes_steady=steady_g
        )
        del hc3, ht3, hc_base, ht_base, do_b
        for p in layer.parameters():
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()

        scaling_b[str(S)] = {
            "batch": B_b,
            "context_len": S,
            "ttt_steps": K_b,
            "total_draft_queries": B_b * S * K_b,
            "modes": modes_s,
        }

    del layer
    gc.collect()
    torch.cuda.empty_cache()
    return {"sequence_scaling": scaling_b}


# ============================================================================
# 8. Suite C: 8-GPU NVSwitch Vocabulary-Parallel (TP=8) Coupled-Monoid
#    Teacher-Student SLM Distillation
# ============================================================================


class _RMSNormModule(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class CausalGQABlock(nn.Module):
    def __init__(self, hidden: int, inter: int, n_q: int, n_kv: int, head_dim: int = 128) -> None:
        super().__init__()
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.norm1 = _RMSNormModule(hidden)
        self.w_q = nn.Linear(hidden, n_q * head_dim, bias=False)
        self.w_kv = nn.Linear(hidden, 2 * n_kv * head_dim, bias=False)
        self.w_o = nn.Linear(n_q * head_dim, hidden, bias=False)
        self.norm2 = _RMSNormModule(hidden)
        self.w_gate_up = nn.Linear(hidden, 2 * inter, bias=False)
        self.w_down = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        h = self.norm1(x)
        q = self.w_q(h).view(B, S, self.n_q, self.head_dim)
        kv = self.w_kv(h).view(B, S, 2, self.n_kv, self.head_dim)
        o = flash_attn_func(q, kv[:, :, 0], kv[:, :, 1], causal=True).reshape(B, S, H)
        x = x + self.w_o(o)
        gu = self.w_gate_up(self.norm2(x))
        g, u = gu.chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u)


class TransformerBackbone(nn.Module):
    def __init__(self, n_layers: int, hidden: int, inter: int, n_q: int, n_kv: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [CausalGQABlock(hidden, inter, n_q, n_kv, 128) for _ in range(n_layers)]
        )
        self.final_norm = _RMSNormModule(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)


class FullCausalLM(nn.Module):
    def __init__(
        self, n_layers: int, hidden: int, inter: int, n_q: int, n_kv: int, vocab_size: int
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(n_layers, hidden, inter, n_q, n_kv)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TP8TeacherInferenceModel(nn.Module):
    """Megatron-style permanently TP=8 sharded Teacher (no_grad, 0 FSDP AllGathers!).
    Ends in a Vocab-Parallel ColumnParallel LM Head producing z_t_shard [N_global, V_shard]!
    """

    def __init__(
        self,
        full_teacher: FullCausalLM,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.layers = nn.ModuleList()

        with torch.no_grad():
            for blk in full_teacher.backbone.blocks:
                H = blk.norm1.weight.numel()
                nq_loc = blk.n_q // world_size
                nkv_loc = max(1, blk.n_kv // world_size)
                d = blk.head_dim
                inter_full = blk.w_down.weight.shape[1]
                inter_loc = inter_full // world_size

                q_sl = slice(rank * nq_loc * d, (rank + 1) * nq_loc * d)
                kv_view = blk.w_kv.weight.view(2, blk.n_kv, d, H)
                kv_loc = kv_view[:, rank * nkv_loc : (rank + 1) * nkv_loc].reshape(
                    2 * nkv_loc * d, H
                )
                o_sl = blk.w_o.weight[:, q_sl].contiguous()

                gu_view = blk.w_gate_up.weight.view(2, inter_full, H)
                gu_loc = gu_view[:, rank * inter_loc : (rank + 1) * inter_loc].reshape(
                    2 * inter_loc, H
                )
                dn_sl = blk.w_down.weight[:, rank * inter_loc : (rank + 1) * inter_loc].contiguous()

                layer_d = nn.ModuleDict(
                    {
                        "n1": _RMSNormModule(H).to(device=device, dtype=torch.bfloat16),
                        "n2": _RMSNormModule(H).to(device=device, dtype=torch.bfloat16),
                    }
                )
                layer_d["n1"].weight.copy_(blk.norm1.weight)
                layer_d["n2"].weight.copy_(blk.norm2.weight)
                layer_d.register_buffer("wq", blk.w_q.weight[q_sl].to(device=device, dtype=torch.bfloat16).contiguous())
                layer_d.register_buffer("wkv", kv_loc.to(device=device, dtype=torch.bfloat16).contiguous())
                layer_d.register_buffer("wo", o_sl.to(device=device, dtype=torch.bfloat16))
                layer_d.register_buffer("wgu", gu_loc.to(device=device, dtype=torch.bfloat16).contiguous())
                layer_d.register_buffer("wdn", dn_sl.to(device=device, dtype=torch.bfloat16))
                layer_d.nq_loc = nq_loc
                layer_d.nkv_loc = nkv_loc
                layer_d.head_dim = d
                self.layers.append(layer_d)

            H_top = full_teacher.backbone.final_norm.weight.numel()
            self.final_norm = _RMSNormModule(H_top).to(device=device, dtype=torch.bfloat16)
            self.final_norm.weight.copy_(full_teacher.backbone.final_norm.weight)

            V_full = full_teacher.lm_head.weight.shape[0]
            V_sh = V_full // world_size
            self.register_buffer(
                "w_head_shard",
                full_teacher.lm_head.weight[rank * V_sh : (rank + 1) * V_sh]
                .to(device=device, dtype=torch.bfloat16)
                .contiguous(),
            )

    @torch.no_grad()
    def forward_vocab_shard(self, x_global: torch.Tensor) -> torch.Tensor:
        B, S, H = x_global.shape
        x = x_global
        for lyr in self.layers:
            h1 = lyr["n1"](x)
            q = F.linear(h1, lyr.wq).view(B, S, lyr.nq_loc, lyr.head_dim)
            kv = F.linear(h1, lyr.wkv).view(B, S, 2, lyr.nkv_loc, lyr.head_dim)
            o_loc = flash_attn_func(q, kv[:, :, 0], kv[:, :, 1], causal=True).reshape(B, S, -1)
            attn_out = F.linear(o_loc, lyr.wo)
            dist.all_reduce(attn_out, op=dist.ReduceOp.SUM)
            x = x + attn_out

            h2 = lyr["n2"](x)
            gu = F.linear(h2, lyr.wgu)
            g, u = gu.chunk(2, dim=-1)
            mlp_out = F.linear(F.silu(g) * u, lyr.wdn)
            dist.all_reduce(mlp_out, op=dist.ReduceOp.SUM)
            x = x + mlp_out

        h_f = self.final_norm(x).view(B * S, H)
        return F.linear(h_f, self.w_head_shard)


def _wrap_fsdp2_full_lm(model: FullCausalLM) -> FullCausalLM:
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in model.backbone.blocks:
        fully_shard(blk, mp_policy=mp)
    fully_shard(model.backbone, mp_policy=mp)
    fully_shard(model.lm_head, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model


def _wrap_fsdp2_backbone(backbone: TransformerBackbone) -> TransformerBackbone:
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in backbone.blocks:
        fully_shard(blk, mp_policy=mp)
    fully_shard(backbone, mp_policy=mp)
    return backbone


def run_suite_c_vocab_parallel_distillation(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    # ------------------------------------------------------------------------
    # C.1 Exact 8-GPU Vocab-Parallel (V_shard = 16,032) vs 1-GPU Global KD
    #     Gradient Parity across Forward KL, Reverse KL, and JSD_0.5!
    # ------------------------------------------------------------------------
    V_c = 128256
    V_sh = V_c // world_size  # 16,032 exact!
    tau_c = 2.0

    torch.manual_seed(12260)
    t_cpu = FullCausalLM(2, 1024, 2816, 8, 8, V_c).to(dtype=torch.bfloat16)
    s_cpu = FullCausalLM(2, 512, 1536, 4, 2, V_c).to(dtype=torch.bfloat16)

    gen_p = torch.Generator(device="cpu").manual_seed(12265)
    B_glob_p, S_p = 8, 128
    N_glob_p = B_glob_p * S_p  # 1,024 global tokens
    x_t_glob = torch.randn((B_glob_p, S_p, 1024), generator=gen_p, dtype=torch.bfloat16).to(device)
    x_s_glob = torch.randn((B_glob_p, S_p, 512), generator=gen_p, dtype=torch.bfloat16).to(device)
    w_tok_p = torch.full((N_glob_p,), 1.0 / float(N_glob_p), device=device, dtype=torch.float32)

    # 1-GPU exact reference for FKL (0), RKL (1), JSD_0.5 (2)
    t_1g = FullCausalLM(2, 1024, 2816, 8, 8, V_c).to(device=device, dtype=torch.bfloat16)
    t_1g.load_state_dict(t_cpu.state_dict())
    with torch.no_grad():
        ht_1g = t_1g(x_t_glob).view(N_glob_p, 1024)

    ref_1g: dict[int, tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for mid in (0, 1, 2):
        s_1g = FullCausalLM(2, 512, 1536, 4, 2, V_c).to(device=device, dtype=torch.bfloat16)
        s_1g.load_state_dict(s_cpu.state_dict())
        hs_1g = s_1g(x_s_glob).view(N_glob_p, 512)
        loss_1g = _FusedCoupledLinearKDFn.apply(
            hs_1g, s_1g.lm_head.weight, ht_1g, t_1g.lm_head.weight, w_tok_p, tau_c, 0.5, mid, 512, True
        )
        loss_1g.backward()
        ref_1g[mid] = (
            float(loss_1g.item()),
            s_1g.backbone.blocks[0].w_q.weight.grad.detach().clone(),
            s_1g.backbone.blocks[1].w_down.weight.grad.detach().clone(),
            s_1g.lm_head.weight.grad.detach().clone(),
        )
        del s_1g

    # Top-128 truncated RKL bias control on 1-GPU
    s_tk = FullCausalLM(2, 512, 1536, 4, 2, V_c).to(device=device, dtype=torch.bfloat16)
    s_tk.load_state_dict(s_cpu.state_dict())
    with torch.no_grad():
        zt_all_p = F.linear(ht_1g, t_1g.lm_head.weight).float()
        zt_tv, zt_ti = torch.topk(zt_all_p, 128, dim=-1)
    hs_tk_p = s_tk(x_s_glob).view(N_glob_p, 512)
    zs_tv = F.linear(hs_tk_p, s_tk.lm_head.weight).float().gather(1, zt_ti)
    lps_tk = F.log_softmax(zs_tv / tau_c, dim=-1)
    lpt_tk = F.log_softmax(zt_tv / tau_c, dim=-1)
    loss_tk = (w_tok_p * (tau_c**2) * (lps_tk.exp() * (lps_tk - lpt_tk)).sum(dim=-1)).sum()
    loss_tk.backward()
    top128_rkl_bias = {
        "dw_layer0_wq_rel_l2": round(
            rel_l2(s_tk.backbone.blocks[0].w_q.weight.grad, ref_1g[1][1]), 6
        ),
        "dw_lm_head_rel_l2": round(rel_l2(s_tk.lm_head.weight.grad, ref_1g[1][3]), 6),
    }
    del s_tk, t_1g, ht_1g, zt_all_p

    # TP=8 Teacher + FSDP2 Student Backbone + Vocab-Parallel Coupled-Monoid Head across FKL, RKL, JSD!
    tp8_teacher_p = TP8TeacherInferenceModel(t_cpu, rank, world_size, device)
    with torch.no_grad():
        zt_sh_p = tp8_teacher_p.forward_vocab_shard(x_t_glob)

    x_s_loc = x_s_glob[rank : rank + 1].contiguous()
    vp_parity: dict[str, Any] = {"top128_truncated_rkl_bias_control": top128_rkl_bias}
    for mid, mname in ((0, "vocab_parallel_tp8_fkl"), (1, "vocab_parallel_tp8_rkl"), (2, "vocab_parallel_tp8_jsd_0p5")):
        sb_f = TransformerBackbone(2, 512, 1536, 4, 2).to(device=device, dtype=torch.bfloat16)
        sb_f.load_state_dict(s_cpu.backbone.state_dict())
        sb_f = _wrap_fsdp2_backbone(sb_f)
        ws_sh = nn.Parameter(
            s_cpu.lm_head.weight[rank * V_sh : (rank + 1) * V_sh]
            .to(device=device, dtype=torch.bfloat16)
            .detach()
            .clone()
        )

        hs_loc = sb_f(x_s_loc).view(S_p, 512)
        loss_vp = _VocabParallelCoupledKDFn.apply(
            hs_loc, ws_sh, zt_sh_p, w_tok_p, tau_c, 0.5, mid, 512, world_size
        )
        loss_vp.backward()

        wq_full = sb_f.blocks[0].w_q.weight.grad.full_tensor()
        wd_full = sb_f.blocks[1].w_down.weight.grad.full_tensor()
        wlm_full = torch.empty((V_c, 512), device=device, dtype=torch.bfloat16)
        dist.all_gather_into_tensor(wlm_full, ws_sh.grad.contiguous())

        l_ref, wq_r, wd_r, wlm_r = ref_1g[mid]
        vp_parity[mname] = {
            "loss_rel_err": round(abs(float(loss_vp.item()) - l_ref) / abs(l_ref), 6),
            "dw_layer0_wq_rel_l2": round(rel_l2(wq_full, wq_r), 6),
            "dw_layer1_wdown_rel_l2": round(rel_l2(wd_full, wd_r), 6),
            "dw_lm_head_rel_l2": round(rel_l2(wlm_full, wlm_r), 6),
        }
        del sb_f, ws_sh, wq_full, wd_full, wlm_full

    del tp8_teacher_p, t_cpu, s_cpu, ref_1g
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # C.2 8-GPU Online Teacher (4L 8B-width H=4096, I=14336) -> Student SLM
    #     (4L 1.5B-width H=2048, I=5632, V=128,256) Step Scaling!
    # ------------------------------------------------------------------------
    Ht_s, It_s, nqt_s, nkvt_s = (2048, 5632, 16, 8) if is_smoke else (4096, 14336, 32, 8)
    Hs_s, Is_s, nqs_s, nkvs_s = (1024, 2816, 8, 2) if is_smoke else (2048, 5632, 16, 4)
    B_glob_s = 8
    B_loc_s = B_glob_s // world_size  # 1 sequence per GPU (8 global sequences)
    S_s = 1024 if is_smoke else 2048
    N_loc_s = B_loc_s * S_s
    N_glob_s = B_glob_s * S_s
    warmup_c = 2 if is_smoke else 3
    iters_c = 4 if is_smoke else 8

    torch.manual_seed(12280)
    teacher_cpu = FullCausalLM(4, Ht_s, It_s, nqt_s, nkvt_s, V_c).to(dtype=torch.bfloat16)
    student_cpu = FullCausalLM(4, Hs_s, Is_s, nqs_s, nkvs_s, V_c).to(dtype=torch.bfloat16)
    teacher_params_b = round(sum(p.numel() for p in teacher_cpu.parameters()) / 1e9, 3)
    student_params_b = round(sum(p.numel() for p in student_cpu.parameters()) / 1e9, 3)

    xt_loc = torch.randn((B_loc_s, S_s, Ht_s), device=device, dtype=torch.bfloat16)
    xs_loc = torch.randn((B_loc_s, S_s, Hs_s), device=device, dtype=torch.bfloat16)
    wt_loc_dp = torch.full((N_loc_s,), 1.0 / float(N_loc_s), device=device, dtype=torch.float32)
    wt_glob_vp = torch.full((N_glob_s,), 1.0 / float(N_glob_s), device=device, dtype=torch.float32)

    def _bench_dist_kd_mode(step_fn: Callable[[], None]) -> dict[str, Any]:
        for _ in range(warmup_c):
            step_fn()
        torch.cuda.synchronize(device)
        dist.barrier()

        gc.collect()
        torch.cuda.empty_cache()
        base_m = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_fn()
        torch.cuda.synchronize(device)
        inc_m = max(0.0, (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0**2))
        max_m = torch.tensor([inc_m], device=device, dtype=torch.float64)
        dist.all_reduce(max_m, op=dist.ReduceOp.MAX)

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
            "global_tokens_per_s": round(float(N_glob_s) / (med_ms * 1e-3), 1),
        }

    c2_modes: dict[str, Any] = {}

    # Modes 1..3 share FSDP2 Teacher + FSDP2 Student setup
    t_fsdp = FullCausalLM(4, Ht_s, It_s, nqt_s, nkvt_s, V_c).to(device=device, dtype=torch.bfloat16)
    t_fsdp.load_state_dict(teacher_cpu.state_dict())
    t_fsdp.requires_grad_(False)
    t_fsdp = _wrap_fsdp2_full_lm(t_fsdp)

    s_fsdp = FullCausalLM(4, Hs_s, Is_s, nqs_s, nkvs_s, V_c).to(device=device, dtype=torch.bfloat16)
    s_fsdp.load_state_dict(student_cpu.state_dict())
    s_fsdp = _wrap_fsdp2_full_lm(s_fsdp)

    # Mode 1: FSDP2 Colocated Materialized FP32 Reverse-KL
    def _f_m1() -> None:
        for p in s_fsdp.parameters():
            p.grad = None
        with torch.no_grad():
            ht = t_fsdp(xt_loc).view(N_loc_s, Ht_s)
            zt = t_fsdp.lm_head(ht).float() / tau_c
            lpt = F.log_softmax(zt, dim=-1)
        hs = s_fsdp(xs_loc).view(N_loc_s, Hs_s)
        zs = s_fsdp.lm_head(hs).float() / tau_c
        lps = F.log_softmax(zs, dim=-1)
        loss = (wt_loc_dp * (tau_c**2) * (lps.exp() * (lps - lpt)).sum(dim=-1)).sum()
        loss.backward()

    c2_modes["fsdp2_colocated_materialized_rkl"] = _bench_dist_kd_mode(_f_m1)

    # Mode 2: FSDP2 Colocated Top-128 Truncated Reverse-KL
    def _f_m2() -> None:
        for p in s_fsdp.parameters():
            p.grad = None
        with torch.no_grad():
            ht = t_fsdp(xt_loc).view(N_loc_s, Ht_s)
            zt = t_fsdp.lm_head(ht)
            zt_v, zt_i = torch.topk(zt, 128, dim=-1)
            lpt = F.log_softmax(zt_v.float() / tau_c, dim=-1)
        hs = s_fsdp(xs_loc).view(N_loc_s, Hs_s)
        zs_v = s_fsdp.lm_head(hs).gather(1, zt_i).float() / tau_c
        lps = F.log_softmax(zs_v, dim=-1)
        loss = (wt_loc_dp * (tau_c**2) * (lps.exp() * (lps - lpt)).sum(dim=-1)).sum()
        loss.backward()

    c2_modes["fsdp2_colocated_topk128_rkl"] = _bench_dist_kd_mode(_f_m2)

    # Mode 3: FSDP2 Colocated + Local Token-Tiled Coupled-Monoid Triton Kernel
    # Note: to trigger FSDP2 pre-forward unshard hooks on lm_head, we wrap the coupled KD call inside lm_head!
    def _f_m3() -> None:
        for p in s_fsdp.parameters():
            p.grad = None
        with torch.no_grad():
            ht = t_fsdp(xt_loc).view(N_loc_s, Ht_s)
        hs = s_fsdp(xs_loc).view(N_loc_s, Hs_s)
        # Unshard both lm_heads cleanly via FSDP2 handle or thin Module forward!
        # Let's unshard t_fsdp.lm_head and s_fsdp.lm_head via their FSDPModule unshard() API!
        t_fsdp.lm_head.unshard()
        s_fsdp.lm_head.unshard()
        try:
            loss = _FusedCoupledLinearKDFn.apply(
                hs,
                s_fsdp.lm_head.weight,
                ht,
                t_fsdp.lm_head.weight,
                wt_loc_dp,
                tau_c,
                0.5,
                1,
                1024,
                True,
            )
        finally:
            t_fsdp.lm_head.reshard()
        loss.backward()

    # Wait: let's make sure FSDP2 backward post-backward ReduceScatter hook fires on s_fsdp.lm_head!
    # Using a dedicated FSDP2-wrapped `CoupledKDHeadModule` guarantees BOTH forward unshard AND backward ReduceScatter hooks fire!
    del t_fsdp, s_fsdp
    gc.collect()
    torch.cuda.empty_cache()

    class _CoupledHeadWrapper(nn.Module):
        def __init__(self, w_init: torch.Tensor) -> None:
            super().__init__()
            self.weight = nn.Parameter(w_init.clone())

        def forward(
            self,
            hs: torch.Tensor,
            ht: torch.Tensor,
            wt_full: torch.Tensor,
            wt_vec: torch.Tensor,
        ) -> torch.Tensor:
            return _FusedCoupledLinearKDFn.apply(
                hs, self.weight, ht, wt_full, wt_vec, tau_c, 0.5, 1, 1024, True
            )

    class _TeacherHeadNoGrad(nn.Module):
        def __init__(self, w_init: torch.Tensor) -> None:
            super().__init__()
            self.weight = nn.Parameter(w_init.clone(), requires_grad=False)

        def forward(self, hs: torch.Tensor, ht: torch.Tensor, s_head: nn.Module, wt_vec: torch.Tensor) -> torch.Tensor:
            return s_head(hs, ht, self.weight, wt_vec)

    t_bb3 = TransformerBackbone(4, Ht_s, It_s, nqt_s, nkvt_s).to(device=device, dtype=torch.bfloat16)
    t_bb3.load_state_dict(teacher_cpu.backbone.state_dict())
    t_bb3.requires_grad_(False)
    t_bb3 = _wrap_fsdp2_backbone(t_bb3)
    t_hd3 = _TeacherHeadNoGrad(teacher_cpu.lm_head.weight.to(device=device, dtype=torch.bfloat16))
    fully_shard(t_hd3, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    s_bb3 = TransformerBackbone(4, Hs_s, Is_s, nqs_s, nkvs_s).to(device=device, dtype=torch.bfloat16)
    s_bb3.load_state_dict(student_cpu.backbone.state_dict())
    s_bb3 = _wrap_fsdp2_backbone(s_bb3)
    s_hd3 = _CoupledHeadWrapper(student_cpu.lm_head.weight.to(device=device, dtype=torch.bfloat16))
    fully_shard(s_hd3, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    def _f_m3_clean() -> None:
        for p in s_bb3.parameters():
            p.grad = None
        for p in s_hd3.parameters():
            p.grad = None
        with torch.no_grad():
            ht = t_bb3(xt_loc).view(N_loc_s, Ht_s)
        hs = s_bb3(xs_loc).view(N_loc_s, Hs_s)
        loss = t_hd3(hs, ht, s_hd3, wt_loc_dp)
        loss.backward()

    c2_modes["fsdp2_colocated_triton_coupled_rkl"] = _bench_dist_kd_mode(_f_m3_clean)
    del t_hd3, s_hd3
    gc.collect()
    torch.cuda.empty_cache()

    # Mode 4 (Ours — Speed Pareto Winner): Hybrid FSDP2 Teacher & Student Backbones
    #   + Permanently Vocab-Parallel (V_shard=16,032) Teacher & Student LM-Heads
    #   + 24 B/tok Coupled-Monoid Reduction (eliminates 2.50 GiB FSDP2 LM-Head AllGather+ReduceScatter!)
    wt_sh_hyb = (
        teacher_cpu.lm_head.weight[rank * V_sh : (rank + 1) * V_sh]
        .to(device=device, dtype=torch.bfloat16)
        .contiguous()
    )
    ws_sh_hyb = nn.Parameter(
        student_cpu.lm_head.weight[rank * V_sh : (rank + 1) * V_sh]
        .to(device=device, dtype=torch.bfloat16)
        .detach()
        .clone()
    )
    ht_glob_buf = torch.empty((N_glob_s, Ht_s), device=device, dtype=torch.bfloat16)

    def _f_m4_hybrid() -> None:
        for p in s_bb3.parameters():
            p.grad = None
        ws_sh_hyb.grad = None
        with torch.no_grad():
            ht_loc = t_bb3(xt_loc).view(N_loc_s, Ht_s)
            dist.all_gather_into_tensor(ht_glob_buf, ht_loc.contiguous())
            zt_shard = F.linear(ht_glob_buf, wt_sh_hyb)
        hs_loc = s_bb3(xs_loc).view(N_loc_s, Hs_s)
        loss = _VocabParallelCoupledKDFn.apply(
            hs_loc, ws_sh_hyb, zt_shard, wt_glob_vp, tau_c, 0.5, 1, 4096, world_size
        )
        loss.backward()

    c2_modes["hybrid_fsdp2_backbone_vocab_parallel_coupled_rkl"] = _bench_dist_kd_mode(
        _f_m4_hybrid
    )
    del t_bb3, s_bb3, wt_sh_hyb, ws_sh_hyb, ht_glob_buf
    gc.collect()
    torch.cuda.empty_cache()

    # Mode 5 (Ours — Memory Pareto Winner): TP=8 Resident Teacher (0 Teacher FSDP AllGather!)
    #   + FSDP2 Student Backbone + Vocab-Parallel (V_shard=16,032) Coupled-Monoid Head!
    tp8_teacher = TP8TeacherInferenceModel(teacher_cpu, rank, world_size, device)
    s_bb4 = TransformerBackbone(4, Hs_s, Is_s, nqs_s, nkvs_s).to(device=device, dtype=torch.bfloat16)
    s_bb4.load_state_dict(student_cpu.backbone.state_dict())
    s_bb4 = _wrap_fsdp2_backbone(s_bb4)
    ws_sh4 = nn.Parameter(
        student_cpu.lm_head.weight[rank * V_sh : (rank + 1) * V_sh]
        .to(device=device, dtype=torch.bfloat16)
        .detach()
        .clone()
    )
    xt_glob_buf = torch.empty((B_glob_s, S_s, Ht_s), device=device, dtype=torch.bfloat16)

    def _f_m4() -> None:
        for p in s_bb4.parameters():
            p.grad = None
        ws_sh4.grad = None
        with torch.no_grad():
            dist.all_gather_into_tensor(xt_glob_buf, xt_loc)
            zt_shard = tp8_teacher.forward_vocab_shard(xt_glob_buf)
        hs_loc = s_bb4(xs_loc).view(N_loc_s, Hs_s)
        loss = _VocabParallelCoupledKDFn.apply(
            hs_loc, ws_sh4, zt_shard, wt_glob_vp, tau_c, 0.5, 1, 4096, world_size
        )
        loss.backward()

    c2_modes["tp8_teacher_vocab_parallel_coupled_rkl"] = _bench_dist_kd_mode(_f_m4)
    del tp8_teacher, s_bb4, ws_sh4, xt_glob_buf, teacher_cpu, student_cpu
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "vocab_parallel_vs_1gpu_gradient_parity": vp_parity,
        "online_8gpu_teacher_student_scaling": {
            "teacher_params_billions": teacher_params_b,
            "student_params_billions": student_params_b,
            "global_batch": B_glob_s,
            "seq_len": S_s,
            "global_tokens": N_glob_s,
            "vocab_size": V_c,
            "vocab_shard_per_gpu": V_sh,
            "coupled_monoid_comm_mib": round((N_glob_s * 24.0) / (1024.0**2), 3),
            "materialized_logit_comm_mib": round((N_glob_s * V_c * 2.0) / (1024.0**2), 1),
            "modes": c2_modes,
        },
    }


# ============================================================================
# 9. Entrypoint
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E122: SLM Distillation Coupled-Monoid, MTP/EAGLE-3 TTT, & 8-GPU Vocab-Parallel KD"
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
        print(f"[E122] Suite 0 complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sa_res = run_suite_a_slm_distillation_and_mtp(device, is_smoke)
        print(f"[E122] Suite A complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sb_res = run_suite_b_eagle3_ttt_layer(device, is_smoke)
        print(f"[E122] Suite B complete at +{time.perf_counter() - t0:.1f}s", flush=True)

    dist.barrier()
    sc_res = run_suite_c_vocab_parallel_distillation(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = round(time.perf_counter() - t0, 2)

    if rank == 0:
        print(f"[E122] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        payload = {
            "experiment": "E122_slm_distillation_mtp_eagle3",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": elapsed_s,
            "initial_sm_clock_mhz": round(init_clk, 1),
            "initial_power_w": round(init_pwr, 2),
            "environment": environment_metadata(),
            "suite_0_audits": s0_res,
            "suite_a_slm_distillation_and_mtp": sa_res,
            "suite_b_eagle3_ttt_layer": sb_res,
            "suite_c_vocab_parallel_distillation": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
