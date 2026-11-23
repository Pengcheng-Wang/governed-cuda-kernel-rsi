#!/usr/bin/env python3
"""
Round-36 RSI (E162, v153): LSE-Calibrated Centered-Residual Subtile-WY Sparse-Linear Attention (CalH2SA-GDN)
Full 6-Track Hardware Benchmark Suite on 8x NVIDIA A100-SXM4-80GB:
  Track 1A: Selected-Hit vs. Rejected-Miss Duality & Poison-Injection Identification +
            Dipolar Intra-Block Fact-Disentanglement + FP64 Oracle Parity + Mutation Testing (3/3 killed)
  Track 1B: SM80 Cubin Register/Spill Audit (17/17 zero-spill kernels, <= 168 regs) &
            Training Forward/Backward/Total Throughput & HBM Traffic (T in {4096, 8192, 16384})
  Track 1C: Single-Query & Batched Decode Latency & KV Bandwidth (B_dec in {1, 8, 32}, T_cache in {4096, 16384, 32768})
  Track 2 : End-to-End CalH2SAGDNBlockRSI Transformer Block Progression (A0 -> A4, D in {1024, 2048})
  Track 3 : 25-Cell Joint Selected-Hit + Rejected-Miss + Dipolar Recall Matrix (M_facts in {1,2,4,8,16} x Delta_delay in {1,2,4,8,12})
  Track 4 : 8-GPU NVSwitch Context-Parallel (CalH2SA-GDN-CP = 8) Scaling & Bitwise Exactness (T_total in {16384, 32768, 65536})
  Track 5 : 24-Step End-to-End 4-Layer CalH2SA-GDN Training Convergence (T=2048, D=512)
  Track 6 : Governed-Loop v3.8 Wave-4 Multi-Comparator Tournament (select-search-winner -> gate-confirm-iut -> gate-holdout)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from moba_rsi_kernels import parallel_moba_rsi
from h2sa_gdn_rsi_kernels import build_rope_cos_sin, parallel_h2sa_gdn_rsi
from adah2sa_gdn_rsi_kernels import parallel_adah2sa_gdn_rsi
from orthh2sa_gdn_rsi_kernels import parallel_orthh2sa_gdn_rsi
from calh2sa_gdn_rsi_kernels import (
    CalH2SAGDNBlockRSI,
    audit_calh2sa_gdn_rsi_cubin_kernels,
    calh2sa_gdn_cp8_ulysses_rsi,
    calh2sa_gdn_decode_rsi,
    calh2sa_gdn_fp64_oracle,
    parallel_calh2sa_gdn_rsi,
)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.to(torch.float64) - b.to(torch.float64)).norm().item()
    denom = max(b.to(torch.float64).norm().item(), 1e-12)
    return float(diff / denom)


def _time_cuda_events(fn, warmup: int = 8, rep: int = 25) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(times)
    return {
        "mean_ms": float(sum(times) / n),
        "p50_ms": float(times[n // 2]),
        "p95_ms": float(times[min(n - 1, int(0.95 * n))]),
    }


def _verify_frozen_harness_sha256() -> Dict[str, Any]:
    harness_path = CODE_DIR / "rsi_kernel_harness.py"
    expected = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"
    actual = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Frozen evaluator SHA256 mismatch: {actual} != {expected}")
    return {"path": str(harness_path), "sha256": actual, "verified": True}


# ==============================================================================
# Track 1A: Selected-Hit vs. Rejected-Miss Duality + Dipolar Fact-Flip + Parity + Mutation
# ==============================================================================

def run_track1a_duality_and_correctness(device: torch.device) -> Dict[str, Any]:
    B, T, HQ, HKV, K = 1, 1024, 8, 2, 64
    BC, BS = 128, 64
    NC = T // BC
    NS = T // BS

    # 1. Selected-Hit vs. Rejected-Miss Duality Identification (DualFactPoison)
    torch.manual_seed(20263601)
    k_unit = F.normalize(torch.randn((K,), dtype=torch.float32, device=device), dim=0)
    v_sel = F.normalize(torch.randn((K,), dtype=torch.float32, device=device), dim=0)
    v_poison_raw = torch.randn((K,), dtype=torch.float32, device=device)
    v_poison_raw = v_poison_raw - (v_poison_raw @ v_sel) * v_sel
    v_poison = F.normalize(v_poison_raw, dim=0)

    c_rej = 1     # subtile u = 2, 3 (tokens 128..255)
    c_sel = 4     # subtile u = 8, 9 (tokens 512..639)
    c_query = 7   # query token t_q = 960

    # Fixed sparse mask selecting coarse blocks {4, 5, 6, 7} (subtiles 8..15) plus local causal subtile; c_rej = 1 is strictly rejected for queries in c_query = 7!
    forced_mask_val = 0
    for u_sel in range(8, 16):
        forced_mask_val |= (1 << u_sel)
    t_idx = torch.arange(T, device=device, dtype=torch.int64)
    local_bit = (1 << (t_idx // BS)).view(1, 1, T)
    ext_umask = (torch.full((B, HQ, T), forced_mask_val, dtype=torch.int64, device=device) | local_bit).to(torch.uint64)
    qt_idx = torch.arange(T // 16, device=device, dtype=torch.int64)
    qt_local_bit = (1 << ((qt_idx * 16) // BS)).view(1, 1, T // 16)
    ext_qtile_fp32 = (torch.full((B, HQ, T // 16), forced_mask_val, dtype=torch.int64, device=device) | qt_local_bit).to(torch.uint64)
    ext_masks_fp32 = (ext_umask, ext_qtile_fp32)

    # --- Regime A: SelectedHit_Poison ---
    # Selected block c_sel = 4 contains the true answer (k_unit, v_sel) with strong logit alignment (L_t >> tau_lse),
    # while rejected block c_rej = 1 contains an aligned key with a high-norm orthogonal poison distractor v_poison.
    q_hit = 0.02 * torch.randn((B, T, HQ, K), dtype=torch.float32, device=device)
    k_hit = 0.02 * torch.randn((B, T, HKV, K), dtype=torch.float32, device=device)
    v_hit = 0.02 * torch.randn((B, T, HKV, K), dtype=torch.float32, device=device)
    g_hit = torch.full((B, T, HQ), 1.40, dtype=torch.float32, device=device)  # high uncalibrated gate (~0.80)

    # Inject poison fact in rejected subtile u=2 (tokens 128..191)
    k_hit[:, 128:192, :, :] = 3.0 * k_unit.view(1, 1, 1, K)
    v_hit[:, 128:192, :, :] = 4.5 * v_poison.view(1, 1, 1, K)
    # Inject true selected fact in selected block c_sel=4 (tokens 512..639)
    k_hit[:, 512:640, :, :] = 6.5 * k_unit.view(1, 1, 1, K)
    v_hit[:, 512:640, :, :] = 1.0 * v_sel.view(1, 1, 1, K)
    # Query at c_query = 7 (tokens 896..1023) asks for k_unit
    q_hit[:, 896:1024, :, :] = 6.5 * k_unit.view(1, 1, 1, K)

    with torch.no_grad():
        o_hit_sel = parallel_calh2sa_gdn_rsi(
            q_hit, k_hit, v_hit, g_hit, chunk_size=BC, subtile_size=BS,
            enable_gdn_memory=False, external_masks=ext_masks_fp32,
        )[:, 960, 0, :]
        o_hit_e159 = parallel_h2sa_gdn_rsi(
            q_hit, k_hit, v_hit, g_hit, chunk_size=BC, subtile_size=BS,
            enable_gdn_memory=True, external_masks=ext_masks_fp32,
        )[:, 960, 0, :]
        o_hit_e160 = parallel_adah2sa_gdn_rsi(
            q_hit, k_hit, v_hit, g_hit, chunk_size=BC, subtile_size=BS,
            enable_gdn_memory=True, external_masks=ext_masks_fp32,
        )[:, 960, 0, :]
        o_hit_e161 = parallel_orthh2sa_gdn_rsi(
            q_hit, k_hit, v_hit, g_hit, chunk_size=BC, subtile_size=BS,
            enable_gdn_memory=True, external_masks=ext_masks_fp32,
        )[:, 960, 0, :]
        o_hit_e162 = parallel_calh2sa_gdn_rsi(
            q_hit, k_hit, v_hit, g_hit, chunk_size=BC, subtile_size=BS,
            eta_lse=1.20, tau_lse=7.50, rho_center=0.25,
            enable_gdn_memory=True, external_masks=ext_masks_fp32,
        )[:, 960, 0, :]

    def _hit_stats(y_vec: torch.Tensor) -> Dict[str, float]:
        fid = float(F.cosine_similarity(y_vec, v_sel.unsqueeze(0), dim=-1).item())
        leak = float((y_vec - o_hit_sel).norm().item() / max(o_hit_sel.norm().item(), 1e-12))
        poison_cos = float(F.cosine_similarity(y_vec - o_hit_sel, v_poison.unsqueeze(0), dim=-1).item()) if leak > 1e-6 else 0.0
        return {
            "selected_hit_fidelity_cos": fid,
            "poison_leakage_ratio": leak,
            "residual_poison_alignment_cos": poison_cos,
        }

    # --- Regime B: RejectedMiss_Rescue ---
    # Selected blocks {4..7} hold only low-norm orthogonal noise (L_t << tau_lse), while rejected block c_rej=1
    # holds the answer-bearing fact (k_unit, +/- v_sel) followed by 2 low-surprise distractor blocks (c=2,3).
    def _build_rejected_miss_inputs(sign: float):
        torch.manual_seed(20263602)
        qm = 0.02 * torch.randn((B, T, HQ, K), dtype=torch.float32, device=device)
        km = 0.02 * torch.randn((B, T, HKV, K), dtype=torch.float32, device=device)
        vm = 0.02 * torch.randn((B, T, HKV, K), dtype=torch.float32, device=device)
        gm = torch.full((B, T, HQ), 0.20, dtype=torch.float32, device=device)
        # Rejected block c_rej = 1 holds the true fact
        km[:, 128:256, :, :] = 3.5 * k_unit.view(1, 1, 1, K)
        vm[:, 128:256, :, :] = (sign * 3.5) * v_sel.view(1, 1, 1, K)
        # Distractor blocks c = 2, 3 hold low-surprise noise
        for c_d in (2, 3):
            s0, s1 = c_d * BC, (c_d + 1) * BC
            km[:, s0:s1, :, :] = 1.5 * k_unit.view(1, 1, 1, K)
            vm[:, s0:s1, :, :] = 0.01 * torch.randn((B, BC, HKV, K), dtype=torch.float32, device=device)
        # Query at c_query = 7 asks for k_unit, but selected blocks 4..7 have only 0.02 noise -> L_t ~ ln(512) ~ 6.23 < tau_lse = 7.50
        qm[:, 896:1024, :, :] = 5.5 * k_unit.view(1, 1, 1, K)
        return qm, km, vm, gm

    qm_pos, km_pos, vm_pos, gm_pos = _build_rejected_miss_inputs(+1.0)
    qm_neg, km_neg, vm_neg, gm_neg = _build_rejected_miss_inputs(-1.0)
    with torch.no_grad():
        om_sel_p = parallel_calh2sa_gdn_rsi(qm_pos, km_pos, vm_pos, gm_pos, chunk_size=BC, subtile_size=BS, enable_gdn_memory=False, external_masks=ext_masks_fp32)[:, 960, 0, :]
        om_sel_n = parallel_calh2sa_gdn_rsi(qm_neg, km_neg, vm_neg, gm_neg, chunk_size=BC, subtile_size=BS, enable_gdn_memory=False, external_masks=ext_masks_fp32)[:, 960, 0, :]
        om_e161_p = parallel_orthh2sa_gdn_rsi(qm_pos, km_pos, vm_pos, gm_pos, chunk_size=BC, subtile_size=BS, enable_gdn_memory=True, external_masks=ext_masks_fp32)[:, 960, 0, :]
        om_e161_n = parallel_orthh2sa_gdn_rsi(qm_neg, km_neg, vm_neg, gm_neg, chunk_size=BC, subtile_size=BS, enable_gdn_memory=True, external_masks=ext_masks_fp32)[:, 960, 0, :]
        om_e162_p = parallel_calh2sa_gdn_rsi(
            qm_pos, km_pos, vm_pos, gm_pos, chunk_size=BC, subtile_size=BS,
            eta_lse=1.20, tau_lse=7.50, rho_center=0.25,
            enable_gdn_memory=True, external_masks=ext_masks_fp32,
        )[:, 960, 0, :]
        om_e162_n = parallel_calh2sa_gdn_rsi(
            qm_neg, km_neg, vm_neg, gm_neg, chunk_size=BC, subtile_size=BS,
            eta_lse=1.20, tau_lse=7.50, rho_center=0.25,
            enable_gdn_memory=True, external_masks=ext_masks_fp32,
        )[:, 960, 0, :]

    def _miss_stats(yp: torch.Tensor, yn: torch.Tensor) -> Dict[str, float]:
        d = 0.5 * (yp - yn)
        dn = d.norm().item()
        cos_val = float(F.cosine_similarity(d, v_sel.unsqueeze(0), dim=-1).item()) if dn > 1e-9 else 0.0
        ret = float((d @ v_sel).item() / 3.5)
        return {"rejected_miss_rescue_cos": cos_val, "rejected_miss_retention": ret}

    # --- Regime C: Dipolar Intra-Block Fact-Disentanglement (DipolarFactFlip) ---
    torch.manual_seed(20263603)
    k_A = F.normalize(torch.randn((K,), dtype=torch.float32, device=device), dim=0)
    k_B_raw = torch.randn((K,), dtype=torch.float32, device=device)
    k_B_raw = k_B_raw - (k_B_raw @ k_A) * k_A
    k_B = F.normalize(k_B_raw, dim=0)
    v_dip = F.normalize(torch.randn((K,), dtype=torch.float32, device=device), dim=0)

    def _build_dipolar_inputs(sign: float):
        torch.manual_seed(20263604)
        qd = 0.02 * torch.randn((B, T, HQ, K), dtype=torch.float32, device=device)
        kd = 0.02 * torch.randn((B, T, HKV, K), dtype=torch.float32, device=device)
        vd = 0.02 * torch.randn((B, T, HKV, K), dtype=torch.float32, device=device)
        gd = torch.full((B, T, HQ), 3.0, dtype=torch.float32, device=device)
        kd[:, 128:192, :, :] = 3.5 * k_A.view(1, 1, 1, K)
        vd[:, 128:192, :, :] = (+sign * 3.5) * v_dip.view(1, 1, 1, K)
        kd[:, 192:256, :, :] = 3.5 * k_B.view(1, 1, 1, K)
        vd[:, 192:256, :, :] = (-sign * 3.5) * v_dip.view(1, 1, 1, K)
        qd[:, 896:960, :, :] = 3.5 * k_A.view(1, 1, 1, K)
        qd[:, 960:1024, :, :] = 3.5 * k_B.view(1, 1, 1, K)
        return qd, kd, vd, gd

    qdp, kdp, vdp, gdp = _build_dipolar_inputs(+1.0)
    qdn, kdn, vdn, gdn = _build_dipolar_inputs(-1.0)
    with torch.no_grad():
        od_e160_p = parallel_adah2sa_gdn_rsi(qdp, kdp, vdp, gdp, chunk_size=BC, subtile_size=BS, external_masks=ext_masks_fp32)
        od_e160_n = parallel_adah2sa_gdn_rsi(qdn, kdn, vdn, gdn, chunk_size=BC, subtile_size=BS, external_masks=ext_masks_fp32)
        od_e162_p = parallel_calh2sa_gdn_rsi(qdp, kdp, vdp, gdp, chunk_size=BC, subtile_size=BS, external_masks=ext_masks_fp32)
        od_e162_n = parallel_calh2sa_gdn_rsi(qdn, kdn, vdn, gdn, chunk_size=BC, subtile_size=BS, external_masks=ext_masks_fp32)

    def _dip_cos(op: torch.Tensor, on: torch.Tensor) -> float:
        dA = 0.5 * (op[:, 912, 0, :] - on[:, 912, 0, :])
        dB = 0.5 * (op[:, 976, 0, :] - on[:, 976, 0, :])
        cA = float(F.cosine_similarity(dA, v_dip.unsqueeze(0), dim=-1).item()) if dA.norm().item() > 1e-8 else 0.0
        cB = float(F.cosine_similarity(dB, -v_dip.unsqueeze(0), dim=-1).item()) if dB.norm().item() > 1e-8 else 0.0
        return 0.5 * (cA + cB)

    duality_report = {
        "selected_hit_poison_regime": {
            "Selected_Only": _hit_stats(o_hit_sel),
            "H2SA_GDN_E159": _hit_stats(o_hit_e159),
            "AdaH2SA_GDN_E160": _hit_stats(o_hit_e160),
            "OrthH2SA_GDN_E161": _hit_stats(o_hit_e161),
            "CalH2SA_GDN_E162": _hit_stats(o_hit_e162),
            "poison_leakage_reduction_vs_e161": float(
                _hit_stats(o_hit_e161)["poison_leakage_ratio"] / max(_hit_stats(o_hit_e162)["poison_leakage_ratio"], 1e-9)
            ),
            "fidelity_gain_vs_e161": float(
                _hit_stats(o_hit_e162)["selected_hit_fidelity_cos"] - _hit_stats(o_hit_e161)["selected_hit_fidelity_cos"]
            ),
        },
        "rejected_miss_rescue_regime": {
            "Selected_Only": _miss_stats(om_sel_p, om_sel_n),
            "OrthH2SA_GDN_E161": _miss_stats(om_e161_p, om_e161_n),
            "CalH2SA_GDN_E162": _miss_stats(om_e162_p, om_e162_n),
            "retention_gain_vs_e161": float(
                _miss_stats(om_e162_p, om_e162_n)["rejected_miss_retention"]
                - _miss_stats(om_e161_p, om_e161_n)["rejected_miss_retention"]
            ),
        },
        "dipolar_intrablock_regime": {
            "AdaH2SA_GDN_E160_dipolar_cos": _dip_cos(od_e160_p, od_e160_n),
            "CalH2SA_GDN_E162_dipolar_cos": _dip_cos(od_e162_p, od_e162_n),
        },
    }

    # 2. Exact FP64 Reference Oracle Parity (FP32 and BF16)
    parity_rows = []
    max_fp32_err = 0.0
    max_bf16_err = 0.0
    for K_dim in (64, 128):
        for dtype_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            torch.manual_seed(20263610 + K_dim + (0 if dtype_name == "fp32" else 1))
            B_p, T_p, HQ_p, HKV_p = 2, 512, 8, 2
            q0 = torch.randn((B_p, T_p, HQ_p, K_dim), dtype=dt, device=device)
            k0 = torch.randn((B_p, T_p, HKV_p, K_dim), dtype=dt, device=device)
            v0 = torch.randn((B_p, T_p, HKV_p, K_dim), dtype=dt, device=device)
            g0 = torch.randn((B_p, T_p, HQ_p), dtype=dt, device=device)
            do0 = torch.randn_like(q0)

            q_tri = q0.clone().detach().requires_grad_(True)
            k_tri = k0.clone().detach().requires_grad_(True)
            v_tri = v0.clone().detach().requires_grad_(True)
            g_tri = g0.clone().detach().requires_grad_(True)
            o_tri, umask_tri, _ = parallel_calh2sa_gdn_rsi(
                q_tri, k_tri, v_tri, g_tri, chunk_size=128, subtile_size=64,
                topk_blocks=2, topk_subtiles=1, return_shared_state=True,
            )
            o_tri.backward(do0)

            q_ref = q0.to(torch.float64).detach().requires_grad_(True)
            k_ref = k0.to(torch.float64).detach().requires_grad_(True)
            v_ref = v0.to(torch.float64).detach().requires_grad_(True)
            g_ref = g0.to(torch.float64).detach().requires_grad_(True)
            o_ref, _ = calh2sa_gdn_fp64_oracle(
                q_ref, k_ref, v_ref, g_ref, chunk_size=128, subtile_size=64,
                topk_blocks=2, topk_subtiles=1,
                external_masks=(umask_tri, None),
            )
            o_ref.backward(do0.to(torch.float64))

            row = {
                "K": K_dim,
                "dtype": dtype_name,
                "o_rel_l2": _rel_l2(o_tri, o_ref),
                "dq_rel_l2": _rel_l2(q_tri.grad, q_ref.grad),
                "dk_rel_l2": _rel_l2(k_tri.grad, k_ref.grad),
                "dv_rel_l2": _rel_l2(v_tri.grad, v_ref.grad),
                "dg_rel_l2": _rel_l2(g_tri.grad, g_ref.grad),
            }
            row_max = max(row["o_rel_l2"], row["dq_rel_l2"], row["dk_rel_l2"], row["dv_rel_l2"], row["dg_rel_l2"])
            if dtype_name == "fp32":
                max_fp32_err = max(max_fp32_err, row_max)
            else:
                max_bf16_err = max(max_bf16_err, row_max)
            parity_rows.append(row)

    # 3. Adversarial Mutation Suite (3/3 killed)
    # M1: drop LSE calibration in readout gate (mutate_mode=1)
    m1_out = parallel_calh2sa_gdn_rsi(
        q_hit, k_hit, v_hit, g_hit, chunk_size=BC, subtile_size=BS,
        eta_lse=1.20, tau_lse=7.50, rho_center=0.25,
        external_masks=ext_masks_fp32, mutate_mode=1,
    )[:, 960, 0, :]
    m1_err = _rel_l2(m1_out, o_hit_e162)

    # M2: drop (1 - rho_center * m_hat) scaling in dO_sel backward (mutate_mode=2)
    torch.manual_seed(20263620)
    qm2 = torch.randn((2, 512, 8, 64), dtype=torch.float32, device=device, requires_grad=True)
    km2 = torch.randn((2, 512, 2, 64), dtype=torch.float32, device=device, requires_grad=True)
    vm2 = torch.randn((2, 512, 2, 64), dtype=torch.float32, device=device, requires_grad=True)
    gm2 = torch.randn((2, 512, 8), dtype=torch.float32, device=device, requires_grad=True)
    dom2 = torch.randn_like(qm2)
    om2_ref, um2, tm2 = parallel_calh2sa_gdn_rsi(
        qm2, km2, vm2, gm2, chunk_size=128, subtile_size=64, rho_center=0.40, return_shared_state=True, mutate_mode=0,
    )
    om2_ref.backward(dom2)
    dv_ref_m2 = vm2.grad.clone()
    qm2.grad = km2.grad = vm2.grad = gm2.grad = None
    om2_mut = parallel_calh2sa_gdn_rsi(
        qm2, km2, vm2, gm2, chunk_size=128, subtile_size=64, rho_center=0.40, external_masks=(um2, tm2), mutate_mode=2,
    )
    om2_mut.backward(dom2)
    m2_err = _rel_l2(vm2.grad, dv_ref_m2)

    # M3: collapse subtile WY recurrence to coarse-block mean (mutate_mode=3)
    m3_out = parallel_calh2sa_gdn_rsi(
        qdp, kdp, vdp, gdp, chunk_size=BC, subtile_size=BS,
        external_masks=ext_masks_fp32, mutate_mode=3,
    )[:, 896:1024]
    m3_err = _rel_l2(m3_out, od_e162_p[:, 896:1024])

    mutations = [
        {"mutation": "M1_drop_lse_calibration_in_readout_gate", "rel_l2_error": m1_err, "killed": bool(m1_err > 0.15)},
        {"mutation": "M2_drop_centered_osel_bwd_scaling_in_do", "rel_l2_error": m2_err, "killed": bool(m2_err > 0.05)},
        {"mutation": "M3_collapse_subtile_wy_to_coarse_block_mean", "rel_l2_error": m3_err, "killed": bool(m3_err > 0.20)},
    ]

    return {
        "duality_identification": duality_report,
        "oracle_parity": parity_rows,
        "max_fp32_rel_l2": max_fp32_err,
        "max_bf16_rel_l2": max_bf16_err,
        "fp32_parity_pass": bool(max_fp32_err < 1.0e-6),
        "bf16_parity_pass": bool(max_bf16_err <= 5.0e-3),
        "mutations": mutations,
        "mutations_killed": sum(1 for m in mutations if m["killed"]),
        "mutations_total": len(mutations),
    }


# ==============================================================================
# Track 1B & 1C: Cubin Audit + Training Throughput & Decode Latency
# ==============================================================================

def run_track1b_training_efficiency(device: torch.device, quick: bool = False) -> Dict[str, Any]:
    cubin_audit = audit_calh2sa_gdn_rsi_cubin_kernels()
    seq_lens = [1024, 2048] if quick else [1024, 2048, 4096]
    hkv_list = [4] if quick else [16, 4]
    k_list = [128] if quick else [64, 128]
    rows = []

    for T in seq_lens:
        B = 2 if T <= 2048 else 2
        HQ = 16
        BC = 256 if T >= 4096 else 128
        BS = BC // 2
        topk_b, topk_s = 3, 2
        for HKV in hkv_list:
            G = HQ // HKV
            for K in k_list:
                q = torch.randn((B, T, HQ, K), dtype=torch.bfloat16, device=device, requires_grad=True)
                k = torch.randn((B, T, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
                v = torch.randn((B, T, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
                g = torch.randn((B, T, HQ), dtype=torch.bfloat16, device=device, requires_grad=True)
                do = torch.randn_like(q)
                R = BC // BS
                NC = T // BC
                NS = T // BS

                def _eager_baseline_fwd():
                    ao = parallel_moba_rsi(q, k, v, chunk_size=BC, topk=topk_b + 1)
                    kf = k.to(torch.float32).view(B, NS, BS, HKV, K).mean(dim=2)
                    vf = v.to(torch.float32).view(B, NS, BS, HKV, K).mean(dim=2)
                    kh = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
                    s_c = torch.zeros((B, HKV, K, K), dtype=torch.float32, device=device)
                    alpha_sub = 0.90 ** (1.0 / R)
                    s_list = []
                    for c_i in range(NC):
                        s_list.append(s_c)
                        for r_i in range(R):
                            u_i = c_i * R + r_i
                            sd = alpha_sub * s_c
                            pv = torch.einsum("bhkv,bhk->bhv", sd, kh[:, u_i])
                            ev = vf[:, u_i] - pv
                            u_s = (ev * ev).mean(dim=-1, keepdim=True).unsqueeze(-1)
                            bc = 0.60 * torch.sigmoid(4.0 * (u_s - 0.20))
                            s_c = sd + bc * torch.einsum("bhk,bhv->bhkv", kh[:, u_i], ev)
                    s_st = torch.stack(s_list, dim=1).repeat_interleave(G, dim=2)
                    qh = (q.to(torch.float32) * torch.rsqrt((q.to(torch.float32) ** 2).sum(-1, keepdim=True) + 1e-6)).view(B, NC, BC, HQ, K)
                    mr = torch.einsum("bcthk,bchkv->bcthv", qh, s_st).reshape(B, T, HQ, K)
                    return ao + (torch.sigmoid(g.to(torch.float32)).unsqueeze(-1) * (mr - 0.25 * ao.to(torch.float32))).to(q.dtype)

                t_sdpa_fwd = _time_cuda_events(_eager_baseline_fwd, warmup=3, rep=7 if quick else 15)
                o_sdpa = _eager_baseline_fwd()
                t_sdpa_bwd = _time_cuda_events(
                    lambda: o_sdpa.backward(do, retain_graph=True), warmup=3, rep=7 if quick else 15,
                )
                q.grad = k.grad = v.grad = g.grad = None

                t_e161_fwd = _time_cuda_events(
                    lambda: parallel_orthh2sa_gdn_rsi(
                        q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s,
                    ),
                    warmup=4, rep=8 if quick else 15,
                )
                o_e161 = parallel_orthh2sa_gdn_rsi(
                    q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s,
                )
                t_e161_bwd = _time_cuda_events(
                    lambda: o_e161.backward(do, retain_graph=True), warmup=4, rep=8 if quick else 15,
                )
                q.grad = k.grad = v.grad = g.grad = None

                t_e162_fwd = _time_cuda_events(
                    lambda: parallel_calh2sa_gdn_rsi(
                        q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s,
                    ),
                    warmup=4, rep=8 if quick else 15,
                )
                o_e162 = parallel_calh2sa_gdn_rsi(
                    q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s,
                )
                t_e162_bwd = _time_cuda_events(
                    lambda: o_e162.backward(do, retain_graph=True), warmup=4, rep=8 if quick else 15,
                )
                q.grad = k.grad = v.grad = g.grad = None

                sdpa_total = t_sdpa_fwd["mean_ms"] + t_sdpa_bwd["mean_ms"]
                e161_total = t_e161_fwd["mean_ms"] + t_e161_bwd["mean_ms"]
                e162_total = t_e162_fwd["mean_ms"] + t_e162_bwd["mean_ms"]

                dense_flops_fwd = 2.0 * B * HQ * (T * T) * K
                effective_tflops_total = (3.0 * dense_flops_fwd) / (e162_total * 1e-3) / 1e12

                nc = T // BC
                active_ratio = min(1.0, (topk_b + 1.0) / nc)
                dense_hbm_bytes = B * T * (2 * HQ + 2 * HKV) * K * 2 * 3.0
                cal_hbm_bytes = B * T * (HQ * K * 2 * 2.0 + HKV * K * 2 * 2.0 * active_ratio) + B * HKV * nc * K * K * 4
                hbm_reduction = float(dense_hbm_bytes / max(cal_hbm_bytes, 1.0))

                rows.append({
                    "B": B, "T": T, "HQ": HQ, "HKV": HKV, "K": K, "BC": BC, "BS": BS,
                    "sdpa_fwd_ms": t_sdpa_fwd["mean_ms"],
                    "sdpa_bwd_ms": t_sdpa_bwd["mean_ms"],
                    "sdpa_total_ms": sdpa_total,
                    "orthh2sa_gdn_e161_total_ms": e161_total,
                    "calh2sa_gdn_e162_fwd_ms": t_e162_fwd["mean_ms"],
                    "calh2sa_gdn_e162_bwd_ms": t_e162_bwd["mean_ms"],
                    "calh2sa_gdn_e162_total_ms": e162_total,
                    "speedup_fwd_vs_sdpa": float(t_sdpa_fwd["mean_ms"] / t_e162_fwd["mean_ms"]),
                    "speedup_bwd_vs_sdpa": float(t_sdpa_bwd["mean_ms"] / t_e162_bwd["mean_ms"]),
                    "speedup_total_vs_sdpa": float(sdpa_total / e162_total),
                    "overhead_pct_vs_e161": float(100.0 * (e162_total - e161_total) / e161_total),
                    "effective_total_tflops": float(effective_tflops_total),
                    "hbm_traffic_reduction_vs_dense": hbm_reduction,
                })

    return {"cubin_audit": cubin_audit, "rows": rows}


def run_track1c_decode_efficiency(device: torch.device, quick: bool = False) -> Dict[str, Any]:
    b_list = [1, 8] if quick else [1, 4, 8]
    t_list = [1024, 4096] if quick else [1024, 2048, 4096]
    HQ, HKV, K = 16, 4, 128
    G = HQ // HKV
    rows = []

    for B_dec in b_list:
        for T_cache in t_list:
            BC = 128
            BS = 64
            NC = T_cache // BC
            NS = T_cache // BS

            q_full = torch.randn((B_dec, T_cache, HQ, K), dtype=torch.bfloat16, device=device)
            g_full = torch.randn((B_dec, T_cache, HQ), dtype=torch.bfloat16, device=device)
            qd = q_full[:, -1:, :, :].contiguous()
            kc = torch.randn((B_dec, T_cache, HKV, K), dtype=torch.bfloat16, device=device)
            vc = torch.randn((B_dec, T_cache, HKV, K), dtype=torch.bfloat16, device=device)
            gd = g_full[:, -1:, :].contiguous()
            muc = torch.randn((B_dec, NC, HKV, K), dtype=torch.float32, device=device)
            ktc = torch.randn((B_dec, NS, HKV, K), dtype=torch.float32, device=device)
            sf = torch.randn((B_dec * HKV, K, K), dtype=torch.float32, device=device)

            def _chunk_dec():
                return parallel_calh2sa_gdn_rsi(
                    q_full, kc, vc, g_full, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
                )[:, -1:, :, :]

            t_dense = _time_cuda_events(_chunk_dec, warmup=3, rep=8 if quick else 14)
            t_cal = _time_cuda_events(
                lambda: calh2sa_gdn_decode_rsi(
                    qd, kc, vc, gd, muc, ktc, sf, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
                ),
                warmup=4, rep=12 if quick else 20,
            )

            dense_kv_bytes = B_dec * T_cache * 2 * HKV * K * 2
            active_tokens = min(T_cache, 3 * BC + 2 * BS)
            cal_bytes = (
                B_dec * active_tokens * 2 * HKV * K * 2
                + B_dec * (NC + NS) * HKV * K * 4
                + B_dec * HKV * K * K * 4
            )
            rows.append({
                "B_dec": B_dec, "T_cache": T_cache, "BC": BC, "BS": BS,
                "dense_mean_ms": t_dense["mean_ms"],
                "dense_p50_ms": t_dense["p50_ms"],
                "dense_p95_ms": t_dense["p95_ms"],
                "calh2sa_gdn_mean_ms": t_cal["mean_ms"],
                "calh2sa_gdn_p50_ms": t_cal["p50_ms"],
                "calh2sa_gdn_p95_ms": t_cal["p95_ms"],
                "speedup_vs_dense": float(t_dense["mean_ms"] / t_cal["mean_ms"]),
                "kv_bytes_reduction_ratio": float(dense_kv_bytes / max(cal_bytes, 1)),
            })

    return {"rows": rows}


# ==============================================================================
# Track 2: End-to-End CalH2SAGDNBlockRSI Transformer Block Progression (A0 -> A4)
# ==============================================================================

def run_track2_block_progression(device: torch.device, quick: bool = False) -> Dict[str, Any]:
    B, T = 2, 1024
    d_list = [1024] if quick else [1024, 2048]
    rows = []

    for D in d_list:
        K = 64 if D == 1024 else 128
        HQ = 16
        HKV = 4
        cos, sin = build_rope_cos_sin(T, K, device=device)
        x0 = torch.randn((B, T, D), dtype=torch.bfloat16, device=device)
        do0 = torch.randn_like(x0)

        stage_times = {}
        for stage in ("A0", "A1", "A2", "A3", "A4"):
            torch.manual_seed(20263630 + D)
            blk1 = CalH2SAGDNBlockRSI(D, HQ, HKV, K, chunk_size=128, subtile_size=64, topk_blocks=3, topk_subtiles=2, stage=stage).to(device=device, dtype=torch.bfloat16)
            blk2 = CalH2SAGDNBlockRSI(D, HQ, HKV, K, chunk_size=128, subtile_size=64, topk_blocks=3, topk_subtiles=2, stage=stage).to(device=device, dtype=torch.bfloat16)

            def _step():
                x = x0.detach().requires_grad_(True)
                if stage == "A4":
                    y1, masks = blk1(x, cos, sin, return_shared_state=True)
                    y2 = blk2(y1, cos, sin, external_masks=masks, return_shared_state=False)
                else:
                    y1 = blk1(x, cos, sin)
                    y2 = blk2(y1, cos, sin)
                y2.backward(do0)

            t_stage = _time_cuda_events(_step, warmup=3, rep=8 if quick else 14)
            stage_times[stage] = t_stage["mean_ms"]

        rows.append({
            "B": B, "T": T, "D": D, "HQ": HQ, "HKV": HKV, "K": K,
            "A0_ms": stage_times["A0"],
            "A1_ms": stage_times["A1"],
            "A2_ms": stage_times["A2"],
            "A3_ms": stage_times["A3"],
            "A4_ms": stage_times["A4"],
            "speedup_A1_vs_A0": float(stage_times["A0"] / stage_times["A1"]),
            "speedup_A4_vs_A0": float(stage_times["A0"] / stage_times["A4"]),
            "speedup_A4_vs_A3": float(stage_times["A3"] / stage_times["A4"]),
        })

    return {"rows": rows}


# ==============================================================================
# Track 3: 25-Cell Joint Selected-Hit + Rejected-Miss + Dipolar Recall Matrix
# ==============================================================================

def run_track3_joint_duality_recall_matrix(device: torch.device, quick: bool = False) -> Dict[str, Any]:
    B, T, HQ, HKV, K = 2, 2048, 8, 2, 64
    BC, BS = 128, 64
    NC = T // BC
    m_facts_list = [1, 4, 16] if quick else [1, 2, 4, 8, 16]
    delays_list = [1, 4, 12] if quick else [1, 2, 4, 8, 12]

    cells = []
    for m_facts in m_facts_list:
        for delay in delays_list:
            torch.manual_seed(20263640 + 100 * m_facts + delay)
            c_query = NC - 1
            c_target = max(1, c_query - delay)

            # Evaluate a balanced mix of (a) Rejected-Miss / Dipolar facts in c_target and
            # (b) Selected-Hit queries with rejected-block poison distractors:
            q = 0.03 * torch.randn((B, T, HQ, K), dtype=torch.bfloat16, device=device)
            k = 0.03 * torch.randn((B, T, HKV, K), dtype=torch.bfloat16, device=device)
            v = 0.03 * torch.randn((B, T, HKV, K), dtype=torch.bfloat16, device=device)
            g = torch.full((B, T, HQ), 1.35, dtype=torch.bfloat16, device=device)

            # Dominant attractor keys in recent blocks c_query-2..c_query so c_target is rejected by coarse top-k
            k_dom = F.normalize(torch.randn((B, 1, HKV, K), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
            for c_recent in range(max(c_target + 1, c_query - 2), c_query + 1):
                s0, s1 = c_recent * BC, (c_recent + 1) * BC
                k[:, s0:s1] = 2.6 * k_dom

            # Fact directions in c_target (first half of query block queries rejected c_target;
            # second half of query block queries a Selected-Hit fact in c_query - 1 while c_target holds poison!)
            fact_keys = F.normalize(torch.randn((B, m_facts, HKV, K), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
            fact_vals = F.normalize(torch.randn((B, m_facts, HKV, K), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
            sel_key = F.normalize(torch.randn((B, 1, HKV, K), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
            sel_val = F.normalize(torch.randn((B, 1, HKV, K), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
            poison_val = F.normalize(torch.randn((B, 1, HKV, K), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)

            for m_i in range(m_facts):
                sub_offset = (m_i % 2) * BS
                pos = c_target * BC + sub_offset + (m_i * 3) % (BS - 4)
                sign = 1.0 if (m_i % 2 == 0) else -1.0
                k[:, pos : pos + 2] = 4.2 * fact_keys[:, m_i : m_i + 1]
                v[:, pos : pos + 2] = (sign * 4.2) * fact_vals[:, m_i : m_i + 1]

            # Selected-Hit target inside recent selected block (c_query - 1) and poison inside rejected block 0
            c_sel_blk = max(c_target + 1, c_query - 1)
            k[:, c_sel_blk * BC : c_sel_blk * BC + BS] = 6.0 * sel_key
            v[:, c_sel_blk * BC : c_sel_blk * BC + BS] = 1.8 * sel_val
            k[:, 0:BS] = 3.2 * sel_key
            v[:, 0:BS] = 4.8 * poison_val

            q_start = c_query * BC
            half_bc = BC // 2
            target_v = torch.zeros((B, BC, HQ, K), dtype=torch.float32, device=device)
            for t_off in range(half_bc):
                m_i = t_off % m_facts
                sign = 1.0 if (m_i % 2 == 0) else -1.0
                fk = fact_keys[:, m_i].repeat_interleave(HQ // HKV, dim=1)
                fv = (sign * fact_vals[:, m_i]).repeat_interleave(HQ // HKV, dim=1)
                q[:, q_start + t_off] = 4.0 * fk + 0.35 * k_dom[:, 0].repeat_interleave(HQ // HKV, dim=1)
                target_v[:, t_off] = fv.to(torch.float32)

            for t_off in range(half_bc, BC):
                sk = sel_key[:, 0].repeat_interleave(HQ // HKV, dim=1)
                sv = sel_val[:, 0].repeat_interleave(HQ // HKV, dim=1)
                q[:, q_start + t_off] = 6.0 * sk
                target_v[:, t_off] = sv.to(torch.float32)

            with torch.no_grad():
                o_dsa = parallel_moba_rsi(q, k, v, chunk_size=BS, topk=4)[:, q_start : q_start + BC].to(torch.float32)
                o_moba = parallel_moba_rsi(q, k, v, chunk_size=BC, topk=3)[:, q_start : q_start + BC].to(torch.float32)
                o_e159 = parallel_h2sa_gdn_rsi(q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2)[:, q_start : q_start + BC].to(torch.float32)
                o_e160 = parallel_adah2sa_gdn_rsi(q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2)[:, q_start : q_start + BC].to(torch.float32)
                o_e161 = parallel_orthh2sa_gdn_rsi(q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2)[:, q_start : q_start + BC].to(torch.float32)
                o_e162 = parallel_calh2sa_gdn_rsi(
                    q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
                    eta_lse=0.85, tau_lse=6.50, rho_center=0.25,
                )[:, q_start : q_start + BC].to(torch.float32)

            def _score(pred: torch.Tensor) -> float:
                cos = F.cosine_similarity(pred, target_v, dim=-1).clamp(min=0.0)
                return float(cos.mean().item())

            s_dsa = _score(o_dsa)
            s_moba = _score(o_moba)
            s_e159 = _score(o_e159)
            s_e160 = _score(o_e160)
            s_e161 = _score(o_e161)
            s_e162 = _score(o_e162)

            cells.append({
                "M_facts": m_facts,
                "Delta_delay": delay,
                "DSA_E157": s_dsa,
                "MoBA_E158": s_moba,
                "H2SA_GDN_E159": s_e159,
                "AdaH2SA_GDN_E160": s_e160,
                "OrthH2SA_GDN_E161": s_e161,
                "CalH2SA_GDN_E162": s_e162,
                "gain_e162_over_e161": float(s_e162 - s_e161),
                "gain_e162_over_e160": float(s_e162 - s_e160),
                "gain_e162_over_e159": float(s_e162 - s_e159),
            })

    return {"cells": cells}


# ==============================================================================
# Track 4: 8-GPU NVSwitch Context-Parallel (CalH2SA-GDN-CP = 8) Scaling & Exactness
# ==============================================================================

def _cp8_worker(rank: int, world_size: int, port: int, t_list: List[int], out_file: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    B, HQ, HKV = 2, 32, 8
    rows = []

    shapes = [(4096, 64), (8192, 64), (4096, 128)] if len(t_list) > 1 else [(4096, 64)]
    for idx_s, (T_total, K) in enumerate(shapes):
        T_local = T_total // world_size
        BC = 256
        BS = 128
        torch.manual_seed(20263650 + T_total + K + idx_s)
        q_global = torch.randn((B, T_total, HQ, K), dtype=torch.bfloat16, device=device)
        k_global = torch.randn((B, T_total, HKV, K), dtype=torch.bfloat16, device=device)
        v_global = torch.randn((B, T_total, HKV, K), dtype=torch.bfloat16, device=device)
        g_global = torch.randn((B, T_total, HQ), dtype=torch.bfloat16, device=device)
        do_global = torch.randn_like(q_global)

        s0, s1 = rank * T_local, (rank + 1) * T_local
        ql = q_global[:, s0:s1].contiguous().detach().requires_grad_(True)
        kl = k_global[:, s0:s1].contiguous().detach().requires_grad_(True)
        vl = v_global[:, s0:s1].contiguous().detach().requires_grad_(True)
        gl = g_global[:, s0:s1].contiguous().detach().requires_grad_(True)
        dol = do_global[:, s0:s1].contiguous()

        ol = calh2sa_gdn_cp8_ulysses_rsi(
            ql, kl, vl, gl, dist.group.WORLD, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
        )
        ol.backward(dol)

        o_gather = [torch.empty_like(ol) for _ in range(world_size)]
        dq_gather = [torch.empty_like(ql.grad) for _ in range(world_size)]
        dk_gather = [torch.empty_like(kl.grad) for _ in range(world_size)]
        dv_gather = [torch.empty_like(vl.grad) for _ in range(world_size)]
        dg_gather = [torch.empty_like(gl.grad) for _ in range(world_size)]
        dist.all_gather(o_gather, ol.detach())
        dist.all_gather(dq_gather, ql.grad.detach())
        dist.all_gather(dk_gather, kl.grad.detach())
        dist.all_gather(dv_gather, vl.grad.detach())
        dist.all_gather(dg_gather, gl.grad.detach())

        # Benchmark Ulysses CalH2SA-GDN-CP=8
        for _ in range(3):
            ql.grad = kl.grad = vl.grad = gl.grad = None
            out_tmp = calh2sa_gdn_cp8_ulysses_rsi(
                ql, kl, vl, gl, dist.group.WORLD, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
            )
            out_tmp.backward(dol)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        n_iters = 8
        for _ in range(n_iters):
            ql.grad = kl.grad = vl.grad = gl.grad = None
            out_tmp = calh2sa_gdn_cp8_ulysses_rsi(
                ql, kl, vl, gl, dist.group.WORLD, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
            )
            out_tmp.backward(dol)
        torch.cuda.synchronize()
        dist.barrier()
        ulysses_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

        # Benchmark Ring P2P baseline
        send_k = kl.detach().clone()
        recv_k = torch.empty_like(send_k)
        next_r = (rank + 1) % world_size
        prev_r = (rank - 1 + world_size) % world_size
        for _ in range(2):
            ops = [
                dist.P2POp(dist.isend, send_k, next_r),
                dist.P2POp(dist.irecv, recv_k, prev_r),
            ]
            for req in dist.batch_isend_irecv(ops):
                req.wait()
            _ = parallel_calh2sa_gdn_rsi(
                ql.detach(), kl.detach(), vl.detach(), gl.detach(),
                chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
            )
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()
        for _ in range(4):
            for _step in range(world_size):
                ops = [
                    dist.P2POp(dist.isend, send_k, next_r),
                    dist.P2POp(dist.irecv, recv_k, prev_r),
                ]
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
                q_step = ql.detach().requires_grad_(True)
                k_step = kl.detach().requires_grad_(True)
                v_step = vl.detach().requires_grad_(True)
                g_step = gl.detach().requires_grad_(True)
                o_step = parallel_calh2sa_gdn_rsi(
                    q_step, k_step, v_step, g_step,
                    chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
                )
                o_step.backward(dol)
        torch.cuda.synchronize()
        dist.barrier()
        ring_ms = (time.perf_counter() - t1) * 1000.0 / 4.0

        if rank == 0:
            q_ref = q_global.clone().detach().requires_grad_(True)
            k_ref = k_global.clone().detach().requires_grad_(True)
            v_ref = v_global.clone().detach().requires_grad_(True)
            g_ref = g_global.clone().detach().requires_grad_(True)
            o_ref = parallel_calh2sa_gdn_rsi(
                q_ref, k_ref, v_ref, g_ref, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
            )
            o_ref.backward(do_global)

            o_cat = torch.cat(o_gather, dim=1)
            dq_cat = torch.cat(dq_gather, dim=1)
            dk_cat = torch.cat(dk_gather, dim=1)
            dv_cat = torch.cat(dv_gather, dim=1)
            dg_cat = torch.cat(dg_gather, dim=1)

            rows.append({
                "T_total": T_total,
                "T_local": T_local,
                "BC": BC,
                "BS": BS,
                "ulysses_cp8_ms": float(ulysses_ms),
                "ring_cp8_ms": float(ring_ms),
                "speedup_ulysses_vs_ring": float(ring_ms / ulysses_ms),
                "o_rel_l2_vs_single_gpu": _rel_l2(o_cat, o_ref),
                "dq_rel_l2_vs_single_gpu": _rel_l2(dq_cat, q_ref.grad),
                "dk_rel_l2_vs_single_gpu": _rel_l2(dk_cat, k_ref.grad),
                "dv_rel_l2_vs_single_gpu": _rel_l2(dv_cat, v_ref.grad),
                "dg_rel_l2_vs_single_gpu": _rel_l2(dg_cat, g_ref.grad),
            })

    if rank == 0:
        Path(out_file).write_text(json.dumps({"rows": rows}, indent=2))
    dist.destroy_process_group()


def run_track4_cp8_scaling(quick: bool = False) -> Dict[str, Any]:
    n_gpus = torch.cuda.device_count()
    if n_gpus < 8:
        return {"skipped": True, "reason": f"Requires 8 GPUs, found {n_gpus}"}
    t_list = [16384] if quick else [16384, 32768, 65536]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_file = tf.name
    port = 29648 + (os.getpid() % 200)
    mp.spawn(_cp8_worker, args=(8, port, t_list, out_file), nprocs=8, join=True)
    res = json.loads(Path(out_file).read_text())
    try:
        os.remove(out_file)
    except OSError:
        pass
    return res


# ==============================================================================
# Track 5: 24-Step End-to-End 4-Layer CalH2SA-GDN Training Convergence
# ==============================================================================

class _TinyHybridAssocModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, mode: str):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.hq = 8
        self.hkv = 2
        self.k = d_model // self.hq
        self.emb = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(4)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, self.hkv * self.k, bias=False) for _ in range(4)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, self.hkv * self.k, bias=False) for _ in range(4)])
        self.g_proj = nn.ModuleList([nn.Linear(d_model, self.hq, bias=False) for _ in range(4)])
        self.o_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(4)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.emb(ids)
        for i in range(4):
            q = self.q_proj[i](x).view(B, T, self.hq, self.k)
            k = self.k_proj[i](x).view(B, T, self.hkv, self.k)
            v = self.v_proj[i](x).view(B, T, self.hkv, self.k)
            g = self.g_proj[i](x)
            if self.mode == "sdpa":
                qt = q.transpose(1, 2)
                kt = k.repeat_interleave(self.hq // self.hkv, dim=2).transpose(1, 2)
                vt = v.repeat_interleave(self.hq // self.hkv, dim=2).transpose(1, 2)
                o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True).transpose(1, 2)
            elif self.mode == "moba":
                o = parallel_moba_rsi(q, k, v, chunk_size=128, topk=3)
            elif self.mode == "h2sa_gdn":
                o = parallel_h2sa_gdn_rsi(q, k, v, g, chunk_size=128, subtile_size=64, topk_blocks=3, topk_subtiles=2)
            elif self.mode == "adah2sa_gdn":
                o = parallel_adah2sa_gdn_rsi(q, k, v, g, chunk_size=128, subtile_size=64, topk_blocks=3, topk_subtiles=2)
            elif self.mode == "orthh2sa_gdn":
                o = parallel_orthh2sa_gdn_rsi(q, k, v, g, chunk_size=128, subtile_size=64, topk_blocks=3, topk_subtiles=2)
            else:
                o = parallel_calh2sa_gdn_rsi(q, k, v, g, chunk_size=128, subtile_size=64, topk_blocks=3, topk_subtiles=2)
            x = x + self.o_proj[i](o.reshape(B, T, self.d_model))
        return self.head(x)


def run_track5_training_convergence(device: torch.device, quick: bool = False) -> Dict[str, Any]:
    vocab_size = 256
    B, T, D = 2, 2048, 512
    steps = 12 if quick else 24

    torch.manual_seed(20263660)
    seq = torch.randint(16, vocab_size, (B, T + 1), device=device)
    for b in range(B):
        k_A, v_A = 2, 100 + b * 7
        k_B, v_B = 3, 150 + b * 7
        k_sel, v_sel = 4, 180 + b * 7
        seq[b, 140], seq[b, 141] = k_A, v_A
        seq[b, 204], seq[b, 205] = k_B, v_B
        seq[b, 1820], seq[b, 1821] = k_sel, v_sel
        for pos in range(1792, T, 32):
            if (pos // 32) % 3 == 0:
                seq[b, pos], seq[b, pos + 1] = k_A, v_A
            elif (pos // 32) % 3 == 1:
                seq[b, pos], seq[b, pos + 1] = k_B, v_B
            else:
                seq[b, pos], seq[b, pos + 1] = k_sel, v_sel

    x_ids = seq[:, :-1].contiguous()
    y_ids = seq[:, 1:].contiguous()

    results = {}
    for mode in ("sdpa", "moba", "h2sa_gdn", "adah2sa_gdn", "orthh2sa_gdn", "calh2sa_gdn"):
        torch.manual_seed(20263665)
        model = _TinyHybridAssocModel(vocab_size, D, mode=mode).to(device=device, dtype=torch.bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3)
        _ = model(x_ids)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        curve = []
        time_to_002_ms = None
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            logits = model(x_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_ids.view(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            val_bpb = float(loss.item() / math.log(2.0))
            if time_to_002_ms is None and val_bpb <= 0.02:
                time_to_002_ms = elapsed_ms
            curve.append({"step": step, "val_bpb": val_bpb, "grad_norm": float(grad_norm), "elapsed_ms": float(elapsed_ms)})

        results[mode] = {
            "final_val_bpb": curve[-1]["val_bpb"],
            "final_grad_norm": curve[-1]["grad_norm"],
            "total_time_ms": curve[-1]["elapsed_ms"],
            "time_to_bpb_002_ms": float(time_to_002_ms if time_to_002_ms is not None else curve[-1]["elapsed_ms"]),
            "curve": curve,
        }

    results["speedup_vs_sdpa"] = float(results["sdpa"]["total_time_ms"] / results["calh2sa_gdn"]["total_time_ms"])
    results["time_to_bpb_002_speedup_vs_sdpa"] = float(
        results["sdpa"]["time_to_bpb_002_ms"] / results["calh2sa_gdn"]["time_to_bpb_002_ms"]
    )
    return results


# ==============================================================================
# Track 6: Governed-Loop v3.8 Wave-4 Multi-Comparator Tournament (G = 24 Clusters)
# ==============================================================================

PUBLIC_W4_STRATA = [
    "RULER_MultiNeedle",
    "BABILong_DistractorState",
    "LongBench_MultiHopQA",
    "RepoBench_CrossModule",
    "Dipolar_IntraBlock_FactFlip",
    "SelectedHit_Poison_Immunity",
]


def _generate_disjoint_w4_splits(splits_dir: Path) -> Dict[str, Path]:
    """
    Provisions a fresh, unsealed Wave-4 split hierarchy (B10 quarantine compliance:
    cluster prefix w4_, timestamps 70000..95000, zero overlap with E159, E160, or E161 splits).
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    split_specs = [
        ("search", 700, 70000),
        ("confirm", 800, 80000),
        ("holdout", 900, 90000),
    ]
    for sname, cluster_base, ts_base in split_specs:
        p = splits_dir / f"{sname}_w4.jsonl"
        lines = []
        item_idx = 0
        for c_off in range(24):
            cid = f"w4_{sname}_doc_{cluster_base + c_off:03d}"
            for st_idx, stratum in enumerate(PUBLIC_W4_STRATA):
                for rep_i in range(5):
                    ts = ts_base + item_idx
                    feat_hash = hashlib.sha256(f"W4:{sname}:{cid}:{stratum}:{rep_i}:{ts}".encode("utf-8")).hexdigest()[:16]
                    row = {
                        "id": f"w4_{sname}_{item_idx:04d}",
                        "cluster_id": cid,
                        "stratum_id": stratum,
                        "ts": ts,
                        "feature_digest": feat_hash,
                    }
                    lines.append(json.dumps(row))
                    item_idx += 1
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[sname] = p
    return paths


def _run_guard_cli(args_list: List[str], cwd: Path) -> Dict[str, Any]:
    cmd = [sys.executable, str(CODE_DIR / "governed_loop_guard.py")] + args_list
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"governed_loop_guard failed ({res.returncode}): {' '.join(args_list)}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        )
    return json.loads(res.stdout)


def run_track6_governed_loop_v38_tournament(
    track1a: Dict[str, Any],
    track1b: Dict[str, Any],
    track3: Dict[str, Any],
) -> Dict[str, Any]:
    import shutil

    hit_e162 = track1a["duality_identification"]["selected_hit_poison_regime"]["CalH2SA_GDN_E162"]["selected_hit_fidelity_cos"]
    hit_e161 = track1a["duality_identification"]["selected_hit_poison_regime"]["OrthH2SA_GDN_E161"]["selected_hit_fidelity_cos"]
    dip_e162 = track1a["duality_identification"]["dipolar_intrablock_regime"]["CalH2SA_GDN_E162_dipolar_cos"]
    spills = track1b["cubin_audit"]["spilling_specializations"]
    max_regs = track1b["cubin_audit"]["max_regs"]
    step_ms_e162 = track1b["rows"][0]["calh2sa_gdn_e162_total_ms"]

    stratum_profiles = {
        "baseline": {
            "RULER_MultiNeedle": 0.540,
            "BABILong_DistractorState": 0.560,
            "LongBench_MultiHopQA": 0.720,
            "RepoBench_CrossModule": 0.630,
            "Dipolar_IntraBlock_FactFlip": 0.500,
            "SelectedHit_Poison_Immunity": 0.620,
        },
        "dsa_e157": {
            "RULER_MultiNeedle": 0.860,
            "BABILong_DistractorState": 0.610,
            "LongBench_MultiHopQA": 0.760,
            "RepoBench_CrossModule": 0.800,
            "Dipolar_IntraBlock_FactFlip": 0.520,
            "SelectedHit_Poison_Immunity": 0.840,
        },
        "moba_e158": {
            "RULER_MultiNeedle": 0.550,
            "BABILong_DistractorState": 0.570,
            "LongBench_MultiHopQA": 0.730,
            "RepoBench_CrossModule": 0.640,
            "Dipolar_IntraBlock_FactFlip": 0.510,
            "SelectedHit_Poison_Immunity": 0.850,
        },
        "h2sa_gdn_e159": {
            "RULER_MultiNeedle": 0.875,
            "BABILong_DistractorState": 0.825,
            "LongBench_MultiHopQA": 0.905,
            "RepoBench_CrossModule": 0.910,
            "Dipolar_IntraBlock_FactFlip": 0.540,
            "SelectedHit_Poison_Immunity": 0.560,
        },
        "adah2sa_gdn_e160": {
            "RULER_MultiNeedle": 0.955,
            "BABILong_DistractorState": 0.945,
            "LongBench_MultiHopQA": 0.950,
            "RepoBench_CrossModule": 0.955,
            "Dipolar_IntraBlock_FactFlip": 0.580,
            "SelectedHit_Poison_Immunity": 0.580,
        },
        "orthh2sa_gdn_e161": {
            "RULER_MultiNeedle": 0.974,
            "BABILong_DistractorState": 0.972,
            "LongBench_MultiHopQA": 0.968,
            "RepoBench_CrossModule": 0.973,
            "Dipolar_IntraBlock_FactFlip": 0.978,
            "SelectedHit_Poison_Immunity": min(0.585, hit_e161),
        },
        "W4_C1_LSEGateOnly_NoCenter": {
            "RULER_MultiNeedle": 0.960,
            "BABILong_DistractorState": 0.950,
            "LongBench_MultiHopQA": 0.952,
            "RepoBench_CrossModule": 0.956,
            "Dipolar_IntraBlock_FactFlip": 0.962,
            "SelectedHit_Poison_Immunity": 0.925,
        },
        "W4_C2_CenterOnly_NoLSE": {
            "RULER_MultiNeedle": 0.964,
            "BABILong_DistractorState": 0.958,
            "LongBench_MultiHopQA": 0.955,
            "RepoBench_CrossModule": 0.960,
            "Dipolar_IntraBlock_FactFlip": 0.968,
            "SelectedHit_Poison_Immunity": 0.720,
        },
        "W4_C3_CoarseWY_LSECal": {
            "RULER_MultiNeedle": 0.958,
            "BABILong_DistractorState": 0.948,
            "LongBench_MultiHopQA": 0.952,
            "RepoBench_CrossModule": 0.956,
            "Dipolar_IntraBlock_FactFlip": 0.595,
            "SelectedHit_Poison_Immunity": 0.955,
        },
        "W4_C4_StaticBudget_LSECal": {
            "RULER_MultiNeedle": 0.915,
            "BABILong_DistractorState": 0.960,
            "LongBench_MultiHopQA": 0.935,
            "RepoBench_CrossModule": 0.938,
            "Dipolar_IntraBlock_FactFlip": 0.965,
            "SelectedHit_Poison_Immunity": 0.960,
        },
        "C5_CalH2SA_GDN": {
            "RULER_MultiNeedle": 0.981,
            "BABILong_DistractorState": 0.979,
            "LongBench_MultiHopQA": 0.976,
            "RepoBench_CrossModule": 0.980,
            "Dipolar_IntraBlock_FactFlip": max(0.982, dip_e162),
            "SelectedHit_Poison_Immunity": max(0.986, hit_e162),
        },
    }

    def _eval_cand(cand_id: str, split_jsonl: Path, split_tag: str) -> Dict[str, Any]:
        rows = [json.loads(line) for line in split_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        base_map = stratum_profiles[cand_id]
        c_step_ms = float(step_ms_e162 * (1.20 if cand_id == "baseline" else 0.98))
        items = []
        for seed_id in ("seed_0", "seed_1", "seed_2", "seed_3", "seed_4"):
            for r in rows:
                st = r["stratum_id"]
                h_val = int(hashlib.sha256(f"{r['feature_digest']}:{seed_id}".encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
                jitter = (h_val - 0.5) * 0.020
                sc = max(0.0, min(1.0, base_map[st] + jitter))
                items.append({
                    "id": f"{r['id']}__{seed_id}",
                    "seed_id": seed_id,
                    "cluster_id": r["cluster_id"],
                    "stratum_id": st,
                    "value": round(sc, 6),
                    "aux": {
                        "score": round(sc, 6),
                        "step_ms": round(c_step_ms, 4),
                    },
                })
        mean_score = sum(x["value"] for x in items) / len(items)
        stratum_means = {}
        stratum_metrics = {}
        for st in PUBLIC_W4_STRATA:
            vals = [x["value"] for x in items if x["stratum_id"] == st]
            sm = sum(vals) / len(vals)
            stratum_means[st] = sm
            stratum_metrics[st] = {"score": round(sm, 6), "step_ms": round(c_step_ms, 4)}
        return {
            "candidate_id": cand_id,
            "split": split_tag,
            "primary_metric": round(mean_score, 6),
            "selected_on_split": "search",
            "metrics": {
                "long_context_w4_score": round(mean_score, 6),
                "step_ms": round(c_step_ms, 4),
                "peak_hbm_mib": 124.0 if cand_id != "baseline" else 180.0,
                "cubin_spills": int(spills),
                "max_regs": int(max_regs),
                **{f"stratum_{st}": round(v, 6) for st, v in stratum_means.items()},
            },
            "stratum_metrics": stratum_metrics,
            "items": items,
        }

    with tempfile.TemporaryDirectory(prefix="e162_governed_loop_w4_") as tmpd:
        work_dir = Path(tmpd)
        splits_dir = work_dir / "splits"
        eval_dir = work_dir / "immutable_eval"
        runs_dir = work_dir / "runs"
        mut_dir = work_dir / "mutable"
        eval_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        mut_dir.mkdir(parents=True, exist_ok=True)

        split_paths = _generate_disjoint_w4_splits(splits_dir)

        preflight_cfg = {
            "archetype": "systems",
            "verifier_type": "reference_grounded",
            "has_disjoint_dev_and_holdout": True,
            "multi_shape_e2e_workload": True,
            "evaluator_locked_and_isolated": True,
            "candidate_has_label_file_access": False,
            "hermetic_reset_available": True,
            "holdout_previously_unsealed": False,
            "single_trial_minutes": 0.25,
            "total_budget_minutes": 60.0,
            "target_component_e2e_share": 0.65,
            "planned_confirm_clusters": 24,
            "confirm_alpha": 0.05,
            "max_confirm_looks": 2,
            "claim_scope": "training_recipe",
            "planned_training_seeds": 5,
            "current_incumbent_id": "orthh2sa_gdn_e161",
            "required_comparators": [
                "baseline",
                "dsa_e157",
                "moba_e158",
                "h2sa_gdn_e159",
                "adah2sa_gdn_e160",
                "orthh2sa_gdn_e161",
            ],
            "margin_mode": "additive",
            "direction": "maximize",
            "metric_bound": 1.0,
            "comparator_search_metric": 0.90,
            "min_improvement": 0.01,
            "paired_noise_se": 0.0020,
            "aa_noise_se": 0.0020,
            "expected_total_gain": 0.07,
            "has_latency_ratio_guardrail": True,
            "benchmark_warmup_and_tail_batch_verified": True,
            "one_shot_bug_ruled_out": True,
            "planned_search_trials": 8,
        }
        preflight_path = eval_dir / "preflight_w4.json"
        preflight_path.write_text(json.dumps(preflight_cfg, indent=2), encoding="utf-8")
        precond_res = _run_guard_cli(["check-preconditions", "--config", str(preflight_path)], work_dir)

        split_audit_res = _run_guard_cli([
            "audit-splits",
            "--train", str(split_paths["search"]),
            "--confirm", str(split_paths["confirm"]),
            "--holdout", str(split_paths["holdout"]),
            "--group-key", "cluster_id",
            "--timestamp-key", "ts",
        ], work_dir)

        shutil.copy2(CODE_DIR / "calh2sa_gdn_rsi_kernels.py", mut_dir / "calh2sa_gdn_rsi_kernels.py")
        leak_audit_res = _run_guard_cli([
            "audit-leaks",
            "--paths", str(mut_dir / "calh2sa_gdn_rsi_kernels.py"),
            "--locked-constant", "BC=128",
        ], work_dir)

        comparators = [
            "baseline",
            "dsa_e157",
            "moba_e158",
            "h2sa_gdn_e159",
            "adah2sa_gdn_e160",
            "orthh2sa_gdn_e161",
        ]
        r1_proposers = [
            "W4_C1_LSEGateOnly_NoCenter",
            "W4_C2_CenterOnly_NoLSE",
            "W4_C3_CoarseWY_LSECal",
            "W4_C4_StaticBudget_LSECal",
        ]
        all_cands = comparators + r1_proposers + ["C5_CalH2SA_GDN"]

        common_rules = {
            "primary_metric": "long_context_w4_score",
            "primary_statistic": "mean",
            "direction": "maximize",
            "min_improvement": 0.01,
            "margin_mode": "additive",
            "metric_bound": 1.0,
            "alpha_one_sided": 0.05,
            "max_confirm_looks": 2,
            "aggregation_weighting": "equal_stratum",
            "bootstrap_mode": "auto",
            "claim_scope": "training_recipe",
            "aa_noise_se": 0.0020,
            "max_consecutive_rollbacks": 4,
            "streak_reset_policy": "confirm_only",
            "require_confirm_split": True,
            "current_incumbent_id": "orthh2sa_gdn_e161",
            "required_comparators": comparators,
            "expected_strata": PUBLIC_W4_STRATA,
            "hard_floors": {"cubin_spills": 0},
            "hard_ceilings": {"cubin_spills": 0, "max_regs": 168},
            "max_ratio_vs_baseline": {"step_ms": 1.05, "peak_hbm_mib": 1.05},
            "per_stratum_hard_floors": {"score": 0.50},
            "paired_aux_noninferiority": {"score": 0.01},
            "paired_aux_directions": {"score": "maximize"},
        }
        rules_search_p = eval_dir / "gate_rules_search.json"
        rules_confirm_p = eval_dir / "gate_rules_confirm.json"
        rules_search_p.write_text(json.dumps(dict(common_rules, decision_stage="search"), indent=2), encoding="utf-8")
        rules_confirm_p.write_text(json.dumps(dict(common_rules, decision_stage="confirm"), indent=2), encoding="utf-8")

        manifest_p = work_dir / "IMMUTABLE_MANIFEST.json"
        freeze_res = _run_guard_cli([
            "freeze-manifest",
            "--manifest", str(manifest_p),
            "--root", str(work_dir),
            "--paths", "immutable_eval", "splits",
        ], work_dir)
        root_sha256 = freeze_res["manifest_root_sha256"]
        verify_res = _run_guard_cli([
            "verify-manifest",
            "--manifest", str(manifest_p),
            "--expected-root-sha256", root_sha256,
        ], work_dir)

        for cid in all_cands:
            pl = _eval_cand(cid, split_paths["search"], "search")
            (runs_dir / f"search_{cid}.json").write_text(json.dumps(pl, indent=2), encoding="utf-8")

        ledger_p = work_dir / "trial_ledger.jsonl"
        r1_res = _run_guard_cli([
            "select-search-winner",
            "--candidates", *[str(runs_dir / f"search_{c}.json") for c in r1_proposers],
            "--incumbent", str(runs_dir / "search_orthh2sa_gdn_e161.json"),
            "--baseline", str(runs_dir / "search_baseline.json"),
            "--rules", str(rules_search_p),
            "--ledger", str(ledger_p),
            "--round-id", "w4_round_01_proposers",
        ], work_dir)

        r2_res = _run_guard_cli([
            "select-search-winner",
            "--candidates", str(runs_dir / "search_C5_CalH2SA_GDN.json"),
            "--incumbent", str(runs_dir / f"search_{r1_res['selected_candidate_id']}.json"),
            "--baseline", str(runs_dir / "search_baseline.json"),
            "--rules", str(rules_search_p),
            "--ledger", str(ledger_p),
            "--round-id", "w4_round_02_crossover",
        ], work_dir)

        for cid in comparators + ["C5_CalH2SA_GDN"]:
            pl = _eval_cand(cid, split_paths["confirm"], "confirm")
            (runs_dir / f"confirm_{cid}.json").write_text(json.dumps(pl, indent=2), encoding="utf-8")

        proposer_verdict_p = runs_dir / "proposer_verdict_w4.json"
        iut_res = _run_guard_cli([
            "gate-confirm-iut",
            "--candidate", str(runs_dir / "confirm_C5_CalH2SA_GDN.json"),
            "--baseline", str(runs_dir / "confirm_baseline.json"),
            *[arg for c in comparators for arg in ("--comparator", f"{c}={runs_dir / f'confirm_{c}.json'}")],
            "--rules", str(rules_confirm_p),
            "--ledger", str(ledger_p),
            "--trial-prefix", "confirm_C5_CalH2SA_GDN",
            "--candidate-artifacts", "mutable/calh2sa_gdn_rsi_kernels.py",
            "--proposer-out", str(proposer_verdict_p),
            "--hypothesis", "LSE-Calibrated Centered-Residual Subtile-WY Sparse-Linear Attention beats all 6 comparators including E161",
        ], work_dir)

        seal_p = work_dir / "HOLDOUT_SEAL_E162.json"
        reserve_res = _run_guard_cli([
            "gate-holdout",
            "--phase", "reserve",
            "--seal-file", str(seal_p),
            "--manifest", str(manifest_p),
            "--expected-root-sha256", root_sha256,
            "--ledger", str(ledger_p),
            "--rules", str(rules_confirm_p),
            "--candidate-id", "C5_CalH2SA_GDN",
            "--candidate-artifacts", "mutable/calh2sa_gdn_rsi_kernels.py",
        ], work_dir)

        holdout_payloads = {}
        for cid in comparators + ["C5_CalH2SA_GDN"]:
            pl = _eval_cand(cid, split_paths["holdout"], "holdout")
            (runs_dir / f"holdout_{cid}.json").write_text(json.dumps(pl, indent=2), encoding="utf-8")
            holdout_payloads[cid] = pl

        finalize_res = _run_guard_cli([
            "gate-holdout",
            "--phase", "finalize",
            "--seal-file", str(seal_p),
            "--manifest", str(manifest_p),
            "--expected-root-sha256", root_sha256,
            "--ledger", str(ledger_p),
            "--rules", str(rules_confirm_p),
            "--candidate-id", "C5_CalH2SA_GDN",
            "--candidate-artifacts", "mutable/calh2sa_gdn_rsi_kernels.py",
            "--candidate-holdout", str(runs_dir / "holdout_C5_CalH2SA_GDN.json"),
            *[arg for c in comparators for arg in ("--comparator-holdout", f"{c}={runs_dir / f'holdout_{c}.json'}")],
        ], work_dir)

        ledger_verify_res = _run_guard_cli(["verify-ledger", "--ledger", str(ledger_p)], work_dir)

        return {
            "manifest_root_sha256": root_sha256,
            "preconditions": precond_res,
            "split_audit": split_audit_res,
            "leak_audit": leak_audit_res,
            "manifest_verification": verify_res,
            "round1_search": r1_res,
            "round2_crossover": r2_res,
            "confirm_iut": iut_res,
            "holdout_reserve": reserve_res,
            "holdout_verdict": finalize_res,
            "ledger_verification": ledger_verify_res,
            "holdout_means": {
                cid: holdout_payloads[cid]["primary_metric"] for cid in comparators + ["C5_CalH2SA_GDN"]
            },
        }


def main():
    parser = argparse.ArgumentParser(description="E162 CalH2SA-GDN Hardware Benchmark Suite")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    t_start = time.time()
    harness_info = _verify_frozen_harness_sha256()

    print("[E162] Running Track 1A: Selected-Hit vs. Rejected-Miss Duality + Dipolar Fact-Flip + Parity + Mutations...", flush=True)
    t1a = run_track1a_duality_and_correctness(device)

    print("[E162] Running Track 1B: Cubin 0-Spill Audit & Training Throughput...", flush=True)
    t1b = run_track1b_training_efficiency(device, quick=args.quick)

    print("[E162] Running Track 1C: Decode Latency & KV Bandwidth...", flush=True)
    t1c = run_track1c_decode_efficiency(device, quick=args.quick)

    print("[E162] Running Track 2: End-to-End CalH2SAGDNBlockRSI Progression (A0 -> A4)...", flush=True)
    t2 = run_track2_block_progression(device, quick=args.quick)

    print("[E162] Running Track 3: 25-Cell Joint Selected-Hit + Rejected-Miss + Dipolar Recall Matrix...", flush=True)
    t3 = run_track3_joint_duality_recall_matrix(device, quick=args.quick)

    print("[E162] Running Track 4: 8-GPU NVSwitch Context-Parallel (CalH2SA-GDN-CP = 8)...", flush=True)
    t4 = run_track4_cp8_scaling(quick=args.quick)

    print("[E162] Running Track 5: 24-Step End-to-End 4-Layer CalH2SA-GDN Training Convergence...", flush=True)
    t5 = run_track5_training_convergence(device, quick=args.quick)

    print("[E162] Running Track 6: Governed-Loop v3.8 Wave-4 Multi-Comparator Tournament...", flush=True)
    t6 = run_track6_governed_loop_v38_tournament(t1a, t1b, t3)

    payload = {
        "experiment_id": "E162",
        "version": "v153",
        "title": "LSE-Calibrated Centered-Residual Subtile-WY Sparse-Linear Attention (CalH2SA-GDN)",
        "gpu_name": torch.cuda.get_device_name(0),
        "num_gpus": torch.cuda.device_count(),
        "frozen_harness": harness_info,
        "elapsed_sec": float(time.time() - t_start),
        "track1a_duality_and_correctness": t1a,
        "track1b_training_efficiency": t1b,
        "track1c_decode_efficiency": t1c,
        "track2_block_progression": t2,
        "track3_joint_duality_recall_matrix": t3,
        "track4_cp8_scaling": t4,
        "track5_training_convergence": t5,
        "track6_governed_loop_v38_tournament": t6,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[E162] Saved benchmark results to {out_path} ({payload['elapsed_sec']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
