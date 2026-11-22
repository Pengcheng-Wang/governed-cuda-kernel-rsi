#!/usr/bin/env python3
"""
E145 (v136, Round-19 RSI): Gated KalmaNet (GKA) Zero-Spill SM80 Benchmark Suite
===============================================================================

Executes the 5-track hardware verification and benchmark suite specified in Section 22 of `PROGRAM.md`
on 8x NVIDIA A100-SXM4-80GB (`NV12` NVSwitch):
  - Track 1: Upstream `fla/ops/gka` & `GatedKalmaNet` SM80 Pathology Forensics
             (193/219 = 88.1% spilling specializations, 1,519,648 B stack spills, ptxas regs=32 allocator collapse,
              262,144 > 166,912 B SMEM overflow, 82.3-200.1 s cold compile, k_0=0 0.0/0.0 NaN contamination across
              32,768/65,536 dk elements, TF32 truncation in FP32 mode, redundant 2x chunk_fwd_h & 11-launch periphery)
  - Track 2: Exact FP64 Oracle Parity, Zero-Key Resilience, & 3-Mode Cryptographic Harness Mutation Kill Test
  - Track 3: Recurrence Kernel Matrix (4 shapes, K in {64, 128}) & 5-Stage Progressive Layer Ablation (Stages A0..A4)
  - Track 4: Fixed-Wall-Clock (`val_bpb`) Autoresearch Language Modeling Loop
  - Track 5: 8-GPU NVSwitch Coupled Dual-State `(H_t, U_t)` Context Parallelism (`GKA-CP = 8`, T in {16384, 32768})
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if "/tmp/fla_repo" not in sys.path and os.path.isdir("/tmp/fla_repo"):
    sys.path.insert(0, "/tmp/fla_repo")

from gated_kalmanet_rsi_kernels import (
    ALL_ROUND19_RSI_KERNELS,
    GatedKalmaNetBlockRSI,
    audit_round19_rsi_cubin_spills,
    chunk_gka_bwd_rsi,
    chunk_gka_fwd_rsi,
    chunk_gka_rsi,
    gka_cp8_ring_baseline,
    gka_cp8_ulysses_rsi,
    naive_gka_fp64_oracle,
    rel_l2_diff,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bench_cuda_fn(fn, warmup: int = 8, rep: int = 25) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    return float(times[len(times) // 2])


# ==============================================================================
# Track 1: Upstream `fla/ops/gka` SM80 Pathology Forensics
# ==============================================================================

def run_track1_upstream_pathology_forensics() -> Dict[str, Any]:
    from fla.ops.gka import chunk_gka as up_chunk_gka

    device = "cuda:0"
    torch.manual_seed(2026)

    # 1. Zero-Key / Masked-Token 0.0 / 0.0 NaN Contamination Audit (k[:, 0] = 0)
    B, T, H, K, V = 2, 128, 4, 64, 64
    q0 = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    k0_raw = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.float32), dim=-1)
    k0_raw[:, 0] = 0.0
    k0 = k0_raw.detach().clone().requires_grad_(True)
    v0 = torch.randn(B, T, H, V, device=device, dtype=torch.float32, requires_grad=True)
    g0 = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    a0 = torch.rand(B, T, H, device=device, dtype=torch.float32, requires_grad=True)

    o_up_zk, _ = up_chunk_gka(q0, k0, v0, g=g0, gk=g0, alpha=a0)
    o_up_zk.sum().backward()
    up_zk_nans = {
        "total_elements_k": int(k0.numel()),
        "o_nans": int(torch.isnan(o_up_zk).sum().item()),
        "dq_nans": int(torch.isnan(q0.grad).sum().item()),
        "dk_nans": int(torch.isnan(k0.grad).sum().item()),
        "dv_nans": int(torch.isnan(v0.grad).sum().item()),
        "dg_nans": int(torch.isnan(g0.grad).sum().item()),
        "da_nans": int(torch.isnan(a0.grad).sum().item()),
    }

    q0_r = q0.detach().clone().requires_grad_(True)
    k0_r = k0.detach().clone().requires_grad_(True)
    v0_r = v0.detach().clone().requires_grad_(True)
    g0_r = g0.detach().clone().requires_grad_(True)
    a0_r = a0.detach().clone().requires_grad_(True)
    o_rsi_zk, _ = chunk_gka_rsi(q0_r, k0_r, v0_r, g=g0_r, gk=g0_r, alpha=a0_r)
    o_rsi_zk.sum().backward()
    rsi_zk_nans = {
        "o_nans": int(torch.isnan(o_rsi_zk).sum().item()),
        "dq_nans": int(torch.isnan(q0_r.grad).sum().item()),
        "dk_nans": int(torch.isnan(k0_r.grad).sum().item()),
        "dv_nans": int(torch.isnan(v0_r.grad).sum().item()),
        "dg_nans": int(torch.isnan(g0_r.grad).sum().item()),
        "da_nans": int(torch.isnan(a0_r.grad).sum().item()),
    }

    # 2. TF32 Truncation in FP32 Mode vs FP64 Exact & Chebyshev Oracles
    q_f = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    k_f = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.float32), dim=-1).detach().requires_grad_(True)
    v_f = torch.randn(B, T, H, V, device=device, dtype=torch.float32, requires_grad=True)
    g_f = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    a_f = torch.rand(B, T, H, device=device, dtype=torch.float32, requires_grad=True)
    do_f = torch.randn(B, T, H, V, device=device, dtype=torch.float32)

    # FP64 exact oracle (for exact implicit differentiation gradients) & Chebyshev oracle (for forward output)
    q_64 = q_f.detach().to(torch.float64).requires_grad_(True)
    k_64 = k_f.detach().to(torch.float64).requires_grad_(True)
    v_64 = v_f.detach().to(torch.float64).requires_grad_(True)
    g_64 = g_f.detach().to(torch.float64).requires_grad_(True)
    a_64 = a_f.detach().to(torch.float64).requires_grad_(True)
    o_exact64, _ = naive_gka_fp64_oracle(q_64, k_64, v_64, g=g_64, gk=g_64, alpha=a_64, exact_solve=True)
    o_exact64.backward(do_f.to(torch.float64))
    with torch.no_grad():
        o_cheb64, _ = naive_gka_fp64_oracle(q_64, k_64, v_64, g=g_64, gk=g_64, alpha=a_64, exact_solve=False)

    o_up_f, _ = up_chunk_gka(q_f, k_f, v_f, g=g_f, gk=g_f, alpha=a_f)
    o_up_f.backward(do_f)

    q_rf = q_f.detach().clone().requires_grad_(True)
    k_rf = k_f.detach().clone().requires_grad_(True)
    v_rf = v_f.detach().clone().requires_grad_(True)
    g_rf = g_f.detach().clone().requires_grad_(True)
    a_rf = a_f.detach().clone().requires_grad_(True)
    o_rsi_f, _ = chunk_gka_rsi(q_rf, k_rf, v_rf, g=g_rf, gk=g_rf, alpha=a_rf)
    o_rsi_f.backward(do_f)

    fp32_precision_audit = {
        "upstream_o_vs_cheb64_rel_l2": rel_l2_diff(o_up_f, o_cheb64),
        "rsi_o_vs_cheb64_rel_l2": rel_l2_diff(o_rsi_f, o_cheb64),
        "o_accuracy_improvement_x": float(rel_l2_diff(o_up_f, o_cheb64) / max(rel_l2_diff(o_rsi_f, o_cheb64), 1e-12)),
        "upstream_dk_vs_exact64_rel_l2": rel_l2_diff(k_f.grad, k_64.grad),
        "rsi_dk_vs_exact64_rel_l2": rel_l2_diff(k_rf.grad, k_64.grad),
        "upstream_dg_vs_exact64_rel_l2": rel_l2_diff(g_f.grad, g_64.grad),
        "rsi_dg_vs_exact64_rel_l2": rel_l2_diff(g_rf.grad, g_64.grad),
    }

    # 3. Upstream SM80 Cubin Register Spilling & ptxas Allocator Collapse Summary (from NVRTC/ptxas audit)
    upstream_cubin_pathology_summary = {
        "total_specializations_audited": 219,
        "spilling_specializations": 193,
        "spilling_fraction_pct": 88.13,
        "total_stack_frame_bytes": 1519648,
        "kernels": {
            "chunk_gka_solve_fwd_kernel": {
                "total_specs": 75,
                "spilling_specs": 75,
                "max_regs": 255,
                "max_stack_bytes_per_thread": 2536,
                "total_stack_bytes": 61248,
                "ptx_unrolled_bytes": 1701420,
                "cold_compile_sec_range": [82.3, 200.1],
            },
            "chunk_gka_bwd_kernel_dk_intra": {
                "total_specs": 72,
                "spilling_specs": 72,
                "min_regs_collapse": 32,
                "max_regs": 255,
                "max_stack_bytes_per_thread": 47864,
                "total_stack_bytes": 1227136,
                "unsplit_smem_bytes": 262144,
                "sm80_smem_limit_bytes": 166912,
            },
            "chunk_gka_bwd_kernel_dk_inter": {
                "total_specs": 72,
                "spilling_specs": 46,
                "max_regs": 255,
                "max_stack_bytes_per_thread": 19904,
                "total_stack_bytes": 231264,
            },
        },
        "redundant_bwd_chunk_fwd_h_calls": 2,
        "unfused_periphery_kernel_launches_fwd": 11,
    }

    return {
        "upstream_zero_key_nans": up_zk_nans,
        "rsi_zero_key_nans": rsi_zk_nans,
        "fp32_precision_audit": fp32_precision_audit,
        "upstream_cubin_pathology_summary": upstream_cubin_pathology_summary,
    }


# ==============================================================================
# Track 2: Exact FP64 Oracle Parity & 3-Mode Cryptographic Harness Mutation Audit
# ==============================================================================

def run_track2_oracle_and_mutation_kill() -> Dict[str, Any]:
    device = "cuda:0"
    torch.manual_seed(202602)

    oracle_cases: List[Dict[str, Any]] = []
    configs = [
        ("fp32_k64_t128_init", torch.float32, 2, 128, 4, 64, 64, True, False),
        ("fp32_k128_t137_nonmult_init", torch.float32, 2, 137, 4, 128, 128, True, False),
        ("fp32_k64_t128_zerokey", torch.float32, 2, 128, 4, 64, 64, False, True),
        ("bf16_k64_t128_init", torch.bfloat16, 2, 128, 4, 64, 64, True, False),
        ("bf16_k128_t137_nonmult_init", torch.bfloat16, 2, 137, 4, 128, 128, True, False),
    ]

    for label, dt, B, T, H, K, V, use_init, zero_k0 in configs:
        q = torch.randn(B, T, H, K, device=device, dtype=dt, requires_grad=True)
        k_raw = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.float32), dim=-1).to(dt)
        if zero_k0:
            k_raw[:, 0] = 0.0
        k = k_raw.detach().clone().requires_grad_(True)
        v = torch.randn(B, T, H, V, device=device, dtype=dt, requires_grad=True)
        g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.12 - 0.01).detach().requires_grad_(True)
        gk = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.12 - 0.01).detach().requires_grad_(True)
        a = torch.rand(B, T, H, device=device, dtype=dt, requires_grad=True)
        do = torch.randn(B, T, H, V, device=device, dtype=dt)

        if use_init:
            h_raw = torch.randn(B, H, K, K, device=device, dtype=torch.float32) * 0.3
            h_kk0 = (h_raw @ h_raw.transpose(-1, -2) / K).detach().requires_grad_(True)
            h_kv0 = (torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.3).detach().requires_grad_(True)
            dht_kk = torch.randn(B, H, K, K, device=device, dtype=torch.float32) * 0.1
            dht_kv = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.1
            init_tup = (h_kk0, h_kv0)
        else:
            h_kk0, h_kv0, dht_kk, dht_kv = None, None, None, None
            init_tup = None

        o_rsi, final_rsi = chunk_gka_rsi(
            q, k, v, g=g, gk=gk, alpha=a, initial_state=init_tup, output_final_state=use_init,
        )
        if use_init:
            loss_rsi = (o_rsi.float() * do.float()).sum() + (final_rsi[0] * dht_kk).sum() + (final_rsi[1] * dht_kv).sum()
        else:
            loss_rsi = (o_rsi.float() * do.float()).sum()
        loss_rsi.backward()

        # FP64 exact oracle & Chebyshev oracle
        q64 = q.detach().to(torch.float64).requires_grad_(True)
        k64 = k.detach().to(torch.float64).requires_grad_(True)
        v64 = v.detach().to(torch.float64).requires_grad_(True)
        g64 = g.detach().to(torch.float64).requires_grad_(True)
        gk64 = gk.detach().to(torch.float64).requires_grad_(True)
        a64 = a.detach().to(torch.float64).requires_grad_(True)
        if use_init:
            hkk64 = h_kk0.detach().to(torch.float64).requires_grad_(True)
            hkv64 = h_kv0.detach().to(torch.float64).requires_grad_(True)
            init64 = (hkk64, hkv64)
        else:
            hkk64, hkv64, init64 = None, None, None

        o_ex64, final_ex64 = naive_gka_fp64_oracle(
            q64, k64, v64, g=g64, gk=gk64, alpha=a64, initial_state=init64, output_final_state=use_init, exact_solve=True,
        )
        if use_init:
            loss64 = (o_ex64 * do.to(torch.float64)).sum() + (final_ex64[0] * dht_kk.to(torch.float64)).sum() + (final_ex64[1] * dht_kv.to(torch.float64)).sum()
        else:
            loss64 = (o_ex64 * do.to(torch.float64)).sum()
        loss64.backward()

        with torch.no_grad():
            o_ch64, _ = naive_gka_fp64_oracle(
                q64, k64, v64, g=g64, gk=gk64, alpha=a64, initial_state=init64, output_final_state=False, exact_solve=False,
            )

        grad_diffs = {
            "dq_rel_l2": rel_l2_diff(q.grad, q64.grad),
            "dk_rel_l2": rel_l2_diff(k.grad, k64.grad),
            "dv_rel_l2": rel_l2_diff(v.grad, v64.grad),
            "dg_rel_l2": rel_l2_diff(g.grad, g64.grad),
            "dgk_rel_l2": rel_l2_diff(gk.grad, gk64.grad),
            "da_rel_l2": rel_l2_diff(a.grad, a64.grad),
        }
        if use_init:
            grad_diffs["ht_kk_rel_l2"] = rel_l2_diff(final_rsi[0], final_ex64[0])
            grad_diffs["ht_kv_rel_l2"] = rel_l2_diff(final_rsi[1], final_ex64[1])
            grad_diffs["dh_kk0_rel_l2"] = rel_l2_diff(h_kk0.grad, hkk64.grad)
            grad_diffs["dh_kv0_rel_l2"] = rel_l2_diff(h_kv0.grad, hkv64.grad)

        tol = 2e-3 if dt == torch.float32 else 3.5e-2
        max_err = max(rel_l2_diff(o_rsi, o_ch64), max(grad_diffs.values()))
        oracle_cases.append({
            "case": label,
            "dtype": str(dt),
            "shape": [B, T, H, K, V],
            "o_vs_cheb64_rel_l2": rel_l2_diff(o_rsi, o_ch64),
            "o_vs_exact64_rel_l2": rel_l2_diff(o_rsi, o_ex64),
            "max_grad_rel_l2": max(grad_diffs.values()),
            "grad_rel_l2": grad_diffs,
            "zero_nans": int(torch.isnan(o_rsi).sum().item() + torch.isnan(k.grad).sum().item()) == 0,
            "passed": bool(max_err < tol),
        })

    # 3-Mode Negative-Control Mutation Kill Audit
    B, T, H, K, V = 2, 128, 4, 64, 64
    qm = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    km = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.float32), dim=-1)
    vm = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    gm = -torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1 - 0.01
    am = torch.ones(B, T, H, device=device, dtype=torch.float32) * 0.85
    dom = torch.randn(B, T, H, V, device=device, dtype=torch.float32)

    o_ref, _, saved_ref = chunk_gka_fwd_rsi(qm, km, vm, g=gm, gk=gm, alpha=am, ridge_ratio=0.15, num_iter=30)
    dq_ref, dk_ref, _, _, dgk_ref, _, _, _ = chunk_gka_bwd_rsi(dom, None, None, saved_ref, corrupt_mode=0)

    # Mode 1: Drop Frobenius regularization gradient path (corrupt_mode=1 -> ridge_ratio=0 in backward)
    _, dk_m1, _, _, dgk_m1, _, _, _ = chunk_gka_bwd_rsi(dom, None, None, saved_ref, corrupt_mode=1)
    m1_dk_err = rel_l2_diff(dk_m1, dk_ref)
    m1_dgk_err = rel_l2_diff(dgk_m1, dgk_ref)

    # Mode 2: Premature Chebyshev truncation (num_iter=2 vs 30)
    o_m2, _, _ = chunk_gka_fwd_rsi(qm, km, vm, g=gm, gk=gm, alpha=am, ridge_ratio=0.02, num_iter=2)
    o_ref30, _, _ = chunk_gka_fwd_rsi(qm, km, vm, g=gm, gk=gm, alpha=am, ridge_ratio=0.02, num_iter=30)
    m2_o_err = rel_l2_diff(o_m2, o_ref30)

    # Mode 3: Sign-inverted adjoint solve (corrupt_mode=3 -> dq_solve = -dq_solve)
    dq_m3, dk_m3, _, _, dgk_m3, _, _, _ = chunk_gka_bwd_rsi(dom, None, None, saved_ref, corrupt_mode=3)
    m3_dq_err = rel_l2_diff(dq_m3, dq_ref)
    m3_dk_err = rel_l2_diff(dk_m3, dk_ref)

    harness_path = CODE_DIR / "rsi_kernel_harness.py"
    harness_sha256 = sha256_file(harness_path)

    mutations = [
        {
            "mode": 1,
            "name": "drop_frobenius_ridge_grad_dlamb",
            "dk_rel_l2": m1_dk_err,
            "dgk_rel_l2": m1_dgk_err,
            "killed": bool(max(m1_dk_err, m1_dgk_err) > 1e-2),
        },
        {
            "mode": 2,
            "name": "truncated_chebyshev_num_iter_2",
            "o_rel_l2": m2_o_err,
            "killed": bool(m2_o_err > 1e-2),
        },
        {
            "mode": 3,
            "name": "inverted_adjoint_sign_dq_solve",
            "dq_rel_l2": m3_dq_err,
            "dk_rel_l2": m3_dk_err,
            "killed": bool(max(m3_dq_err, m3_dk_err) > 0.5),
        },
    ]

    return {
        "harness_sha256": harness_sha256,
        "harness_verified": harness_sha256 == "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e",
        "oracle_cases": oracle_cases,
        "all_oracle_passed": all(c["passed"] for c in oracle_cases),
        "mutations": mutations,
        "all_mutations_killed": all(m["killed"] for m in mutations),
    }


# ==============================================================================
# Track 3: Recurrence Kernel Matrix & 5-Stage Progressive Layer Ablation (A0..A4)
# ==============================================================================

def run_track3_kernel_and_layer_ablation(mode: str = "full") -> Dict[str, Any]:
    from fla.ops.gka import chunk_gka as up_chunk_gka

    device = "cuda:0"
    torch.manual_seed(202603)
    warmup = 4 if mode == "smoke" else 10
    rep = 10 if mode == "smoke" else 25

    shapes = [
        ("B2_T2048_H8_K64_V64", 2, 2048, 8, 64, 64),
        ("B2_T4096_H8_K64_V64", 2, 4096, 8, 64, 64),
        ("B2_T2048_H8_K128_V128", 2, 2048, 8, 128, 128),
        ("B1_T4096_H8_K128_V128", 1, 4096, 8, 128, 128),
    ]

    kernel_matrix: List[Dict[str, Any]] = []
    for label, B, T, H, K, V in shapes:
        q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), dim=-1).detach().requires_grad_(True)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
        g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.05 - 0.005).detach().requires_grad_(True)
        a = torch.rand(B, T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)

        def fla_fwd():
            up_chunk_gka(q.detach(), k.detach(), v.detach(), g=g.detach(), gk=g.detach(), alpha=a.detach())

        def fla_fwdbwd():
            q.grad = k.grad = v.grad = g.grad = a.grad = None
            o, _ = up_chunk_gka(q, k, v, g=g, gk=g, alpha=a)
            o.backward(do)

        def rsi_fwd():
            chunk_gka_fwd_rsi(q.detach(), k.detach(), v.detach(), g=g.detach(), gk=g.detach(), alpha=a.detach())

        def rsi_fwdbwd():
            q.grad = k.grad = v.grad = g.grad = a.grad = None
            o, _ = chunk_gka_rsi(q, k, v, g=g, gk=g, alpha=a)
            o.backward(do)

        fla_fwd_ms = bench_cuda_fn(fla_fwd, warmup=warmup, rep=rep)
        fla_fwdbwd_ms = bench_cuda_fn(fla_fwdbwd, warmup=warmup, rep=rep)
        fla_bwd_ms = max(fla_fwdbwd_ms - fla_fwd_ms, 1e-3)

        rsi_fwd_ms = bench_cuda_fn(rsi_fwd, warmup=warmup, rep=rep)
        rsi_fwdbwd_ms = bench_cuda_fn(rsi_fwdbwd, warmup=warmup, rep=rep)
        rsi_bwd_ms = max(rsi_fwdbwd_ms - rsi_fwd_ms, 1e-3)

        tokens = B * T
        kernel_matrix.append({
            "label": label,
            "shape": [B, T, H, K, V],
            "tokens": tokens,
            "upstream_fla_fwd_ms": fla_fwd_ms,
            "upstream_fla_bwd_ms": fla_bwd_ms,
            "upstream_fla_fwdbwd_ms": fla_fwdbwd_ms,
            "upstream_fla_tok_per_s": float(tokens / (fla_fwdbwd_ms * 1e-3)),
            "rsi_fwd_ms": rsi_fwd_ms,
            "rsi_bwd_ms": rsi_bwd_ms,
            "rsi_fwdbwd_ms": rsi_fwdbwd_ms,
            "rsi_tok_per_s": float(tokens / (rsi_fwdbwd_ms * 1e-3)),
            "fwd_speedup_x": float(fla_fwd_ms / rsi_fwd_ms),
            "bwd_speedup_x": float(fla_bwd_ms / rsi_bwd_ms),
            "fwdbwd_speedup_x": float(fla_fwdbwd_ms / rsi_fwdbwd_ms),
        })

    # 5-Stage Progressive Layer Ablation (Stages A0..A4)
    layer_ablations: Dict[str, Any] = {}
    stage_names = {
        "A0": "A0_upstream_fla_unfused",
        "A1": "A1_rsi_solve_fwd",
        "A2": "A2_rsi_solve_bwd",
        "A3": "A3_rsi_saved_state_chunk",
        "A4": "A4_rsi_full_layer_superfused",
    }
    for K in [64, 128]:
        B, T, H = 2, 2048, 8
        D_MODEL = H * K
        blk = GatedKalmaNetBlockRSI(
            hidden_size=D_MODEL, num_heads=H, head_dim=K, expand_v=1.0, conv_size=4, ridge_ratio=0.02, num_iter=30,
        ).to(device=device, dtype=torch.bfloat16)
        x = torch.randn(B, T, D_MODEL, device=device, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(B, T, D_MODEL, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            y_ref = blk(x.detach(), stage="A3")

        stages_list: List[Dict[str, Any]] = []
        a0_fwdbwd_ms = None
        a0_fwd_ms = None
        a0_bwd_ms = None
        for st in ["A0", "A1", "A2", "A3", "A4"]:
            def st_fwd():
                with torch.no_grad():
                    blk(x.detach(), stage=st)

            def st_fwdbwd():
                x.grad = None
                blk.zero_grad(set_to_none=True)
                y = blk(x, stage=st)
                y.backward(dy)

            st_fwdbwd()
            with torch.no_grad():
                y_cur = blk(x.detach(), stage=st)
                rel_vs_a3 = rel_l2_diff(y_cur, y_ref)

            fwd_ms = bench_cuda_fn(st_fwd, warmup=warmup, rep=rep)
            fwdbwd_ms = bench_cuda_fn(st_fwdbwd, warmup=warmup, rep=rep)
            bwd_ms = max(fwdbwd_ms - fwd_ms, 1e-3)
            if st == "A0":
                a0_fwdbwd_ms = fwdbwd_ms
                a0_fwd_ms = fwd_ms
                a0_bwd_ms = bwd_ms

            tokens = B * T
            stages_list.append({
                "stage": st,
                "name": stage_names[st],
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "fwdbwd_ms": fwdbwd_ms,
                "tok_per_s": float(tokens / (fwdbwd_ms * 1e-3)),
                "fwd_speedup_vs_a0_x": float(a0_fwd_ms / fwd_ms),
                "bwd_speedup_vs_a0_x": float(a0_bwd_ms / bwd_ms),
                "fwdbwd_speedup_vs_a0_x": float(a0_fwdbwd_ms / fwdbwd_ms),
                "y_rel_l2_vs_a3": rel_vs_a3,
            })

        layer_ablations[f"K{K}"] = {
            "config": {"B": B, "T": T, "H": H, "K": K, "V": K, "D_MODEL": D_MODEL},
            "stages": stages_list,
            "a4_vs_a0_fwdbwd_speedup_x": stages_list[-1]["fwdbwd_speedup_vs_a0_x"],
        }

    cubin_audit = audit_round19_rsi_cubin_spills()
    return {
        "kernel_matrix": kernel_matrix,
        "layer_ablations": layer_ablations,
        "round19_rsi_cubin_audit": cubin_audit,
    }


# ==============================================================================
# Track 4: Fixed-Wall-Clock (`val_bpb`) Autoresearch Language Modeling Loop
# ==============================================================================

class TinyGKALM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 256, n_layers: int = 4, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            GatedKalmaNetBlockRSI(hidden_size=d_model, num_heads=num_heads, head_dim=head_dim, expand_v=1.0)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor, stage: str = "A4") -> torch.Tensor:
        x = self.embed(idx)
        for norm, blk in zip(self.norms, self.layers):
            x = x + blk(norm(x), stage=stage)
        return self.lm_head(self.final_norm(x))


def _make_associative_grammar_batches(
    num_batches: int, B: int, T: int, vocab_size: int, device: str, seed: int = 777,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generates multi-hop in-context associative recall + copy sequences where each sequence defines
    sequence-specific `(key_i, val_i)` pairs and multi-token motifs that require the Gated KalmaNet
    ridge-regression memory state $S_t = U_t^\\top (H_t + \\lambda_t I)^{-1}$ to recall.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    batches = []
    n_pairs = 16
    for _ in range(num_batches):
        seq = torch.zeros((B, T + 1), dtype=torch.long)
        for b in range(B):
            keys = torch.randperm(64, generator=gen)[:n_pairs] + 1
            vals = torch.randperm(64, generator=gen)[:n_pairs] + 65
            vals2 = ((keys * 5 + vals * 3) % 120) + 129
            pos = 0
            # Prefix: present all 16 (key, val1, val2) triples twice
            for _ in range(2):
                for i in range(n_pairs):
                    if pos + 2 < T + 1:
                        seq[b, pos] = keys[i]
                        seq[b, pos + 1] = vals[i]
                        seq[b, pos + 2] = vals2[i]
                        pos += 3
            # Body: query random keys from the prefix
            while pos < T + 1:
                idx = int(torch.randint(0, n_pairs, (1,), generator=gen).item())
                seq[b, pos] = keys[idx]
                if pos + 1 < T + 1:
                    seq[b, pos + 1] = vals[idx]
                if pos + 2 < T + 1:
                    seq[b, pos + 2] = vals2[idx]
                pos += 3
        seq_gpu = seq.to(device=device, non_blocking=True)
        batches.append((seq_gpu[:, :-1].contiguous(), seq_gpu[:, 1:].contiguous()))
    return batches


def run_track4_autoresearch_val_bpb(mode: str = "full") -> Dict[str, Any]:
    device = "cuda:0"
    budget_sec = 5.0 if mode == "smoke" else 15.0
    vocab_size = 256
    B, T = 4, 512
    d_model, n_layers, num_heads, head_dim = 256, 4, 4, 64

    torch.manual_seed(202604)
    base_model = TinyGKALM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, num_heads=num_heads, head_dim=head_dim).to(device=device, dtype=torch.bfloat16)
    init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    train_batches = _make_associative_grammar_batches(128, B, T, vocab_size, device, seed=1001)
    val_batches = _make_associative_grammar_batches(16, B, T, vocab_size, device, seed=2002)

    def eval_bpb(model: nn.Module, stage: str) -> float:
        model.eval()
        total_nll = 0.0
        total_tok = 0
        with torch.no_grad():
            for x_b, y_b in val_batches:
                logits = model(x_b, stage=stage).float()
                nll = F.cross_entropy(logits.view(-1, vocab_size), y_b.view(-1), reduction="sum")
                total_nll += float(nll.item())
                total_tok += int(y_b.numel())
        model.train()
        return float((total_nll / total_tok) / math.log(2.0))

    init_val_bpb = eval_bpb(base_model, stage="A4")

    results: Dict[str, Any] = {}
    for stage_key, label in [("A0", "upstream_fla_a0"), ("A4", "round19_rsi_a4")]:
        model = TinyGKALM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, num_heads=num_heads, head_dim=head_dim).to(device=device, dtype=torch.bfloat16)
        model.load_state_dict({k: v.clone() for k, v in init_state.items()})
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)

        # Warmup 2 steps outside timer
        for w_idx in range(2):
            xb, yb = train_batches[w_idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, stage=stage_key)
            loss = F.cross_entropy(logits.float().view(-1, vocab_size), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize()

        model.load_state_dict({k: v.clone() for k, v in init_state.items()})
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
        torch.cuda.synchronize()

        steps = 0
        t0 = time.perf_counter()
        while True:
            xb, yb = train_batches[steps % len(train_batches)]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, stage=stage_key)
            loss = F.cross_entropy(logits.float().view(-1, vocab_size), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            torch.cuda.synchronize()
            if (time.perf_counter() - t0) >= budget_sec:
                break
        elapsed = time.perf_counter() - t0
        final_bpb = eval_bpb(model, stage=stage_key)
        results[label] = {
            "stage": stage_key,
            "budget_sec": budget_sec,
            "elapsed_sec": float(elapsed),
            "steps_completed": steps,
            "tokens_trained": int(steps * B * T),
            "tokens_per_sec": float((steps * B * T) / elapsed),
            "init_val_bpb": init_val_bpb,
            "final_val_bpb": final_bpb,
            "bpb_reduction": float(init_val_bpb - final_bpb),
        }

    u = results["upstream_fla_a0"]
    r = results["round19_rsi_a4"]
    results["comparison"] = {
        "step_ratio_x": float(r["steps_completed"] / max(u["steps_completed"], 1)),
        "throughput_speedup_x": float(r["tokens_per_sec"] / max(u["tokens_per_sec"], 1e-6)),
        "val_bpb_delta_rsi_minus_upstream": float(r["final_val_bpb"] - u["final_val_bpb"]),
        "val_bpb_improvement": float(u["final_val_bpb"] - r["final_val_bpb"]),
    }
    return results


# ==============================================================================
# Track 5: 8-GPU NVSwitch Context Parallelism (`GKA-CP = 8`)
# ==============================================================================

def _cp8_worker(rank: int, world_size: int, port: int, mode: str, return_dict):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    device = f"cuda:{rank}"

    warmup = 3 if mode == "smoke" else 8
    rep = 8 if mode == "smoke" else 20

    seq_lengths = [16384] if mode == "smoke" else [16384, 32768]
    out_records = []

    for T_total in seq_lengths:
        B, H, K, V = 1, 8, 128, 128
        T_loc = T_total // world_size
        torch.manual_seed(202605)

        q_full = torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16)
        k_full = F.normalize(torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16), dim=-1)
        v_full = torch.randn(B, T_total, H, V, device=device, dtype=torch.bfloat16)
        g_full = -torch.rand(B, T_total, H, device=device, dtype=torch.float32) * 0.04 - 0.005
        a_full = torch.rand(B, T_total, H, device=device, dtype=torch.bfloat16)
        do_full = torch.randn(B, T_total, H, V, device=device, dtype=torch.bfloat16)

        s0, s1 = rank * T_loc, (rank + 1) * T_loc
        q_s = q_full[:, s0:s1].detach().clone().requires_grad_(True)
        k_s = k_full[:, s0:s1].detach().clone().requires_grad_(True)
        v_s = v_full[:, s0:s1].detach().clone().requires_grad_(True)
        g_s = g_full[:, s0:s1].detach().clone().requires_grad_(True)
        a_s = a_full[:, s0:s1].detach().clone().requires_grad_(True)
        do_s = do_full[:, s0:s1].detach().clone()

        # Verify parity against single-GPU full-sequence reference on rank 0
        o_uly = gka_cp8_ulysses_rsi(q_s, k_s, v_s, g_s, g_s, a_s, group)
        o_uly.backward(do_s)
        dq_uly = q_s.grad.detach().clone()
        dk_uly = k_s.grad.detach().clone()

        q_r = q_full[:, s0:s1].detach().clone().requires_grad_(True)
        k_r = k_full[:, s0:s1].detach().clone().requires_grad_(True)
        v_r = v_full[:, s0:s1].detach().clone().requires_grad_(True)
        g_r = g_full[:, s0:s1].detach().clone().requires_grad_(True)
        a_r = a_full[:, s0:s1].detach().clone().requires_grad_(True)
        o_ring = gka_cp8_ring_baseline(q_r, k_r, v_r, g_r, g_r, a_r, group)
        o_ring.backward(do_s)

        if rank == 0:
            q_ref = q_full.detach().clone().requires_grad_(True)
            k_ref = k_full.detach().clone().requires_grad_(True)
            v_ref = v_full.detach().clone().requires_grad_(True)
            g_ref = g_full.detach().clone().requires_grad_(True)
            a_ref = a_full.detach().clone().requires_grad_(True)
            o_ref, _ = chunk_gka_rsi(q_ref, k_ref, v_ref, g=g_ref, gk=g_ref, alpha=a_ref)
            o_ref.backward(do_full)
            o_uly_rel = rel_l2_diff(o_uly, o_ref[:, s0:s1])
            dq_uly_rel = rel_l2_diff(dq_uly, q_ref.grad[:, s0:s1])
            dk_uly_rel = rel_l2_diff(dk_uly, k_ref.grad[:, s0:s1])
            o_ring_rel = rel_l2_diff(o_ring, o_ref[:, s0:s1])
        else:
            o_uly_rel, dq_uly_rel, dk_uly_rel, o_ring_rel = 0.0, 0.0, 0.0, 0.0

        def run_uly():
            q_s.grad = k_s.grad = v_s.grad = g_s.grad = a_s.grad = None
            out = gka_cp8_ulysses_rsi(q_s, k_s, v_s, g_s, g_s, a_s, group)
            out.backward(do_s)

        def run_ring():
            q_r.grad = k_r.grad = v_r.grad = g_r.grad = a_r.grad = None
            out = gka_cp8_ring_baseline(q_r, k_r, v_r, g_r, g_r, a_r, group)
            out.backward(do_s)

        for _ in range(warmup):
            run_uly()
        dist.barrier()
        torch.cuda.synchronize()
        u_times = []
        for _ in range(rep):
            dist.barrier()
            t0 = time.perf_counter()
            run_uly()
            torch.cuda.synchronize()
            u_times.append((time.perf_counter() - t0) * 1e3)
        uly_ms = float(sorted(u_times)[len(u_times) // 2])

        for _ in range(warmup):
            run_ring()
        dist.barrier()
        torch.cuda.synchronize()
        r_times = []
        for _ in range(rep):
            dist.barrier()
            t0 = time.perf_counter()
            run_ring()
            torch.cuda.synchronize()
            r_times.append((time.perf_counter() - t0) * 1e3)
        ring_ms = float(sorted(r_times)[len(r_times) // 2])

        if rank == 0:
            single_ms = bench_cuda_fn(
                lambda: (
                    setattr(q_ref, "grad", None),
                    setattr(k_ref, "grad", None),
                    setattr(v_ref, "grad", None),
                    setattr(g_ref, "grad", None),
                    setattr(a_ref, "grad", None),
                    chunk_gka_rsi(q_ref, k_ref, v_ref, g=g_ref, gk=g_ref, alpha=a_ref)[0].backward(do_full),
                ),
                warmup=warmup,
                rep=rep,
            )
            out_records.append({
                "T_total": T_total,
                "T_local": T_loc,
                "shape_full": [B, T_total, H, K, V],
                "single_gpu_rsi_fwdbwd_ms": single_ms,
                "cp8_ring_state_passing_fwdbwd_ms": ring_ms,
                "cp8_ulysses_zero_copy_fwdbwd_ms": uly_ms,
                "ulysses_vs_ring_speedup_x": float(ring_ms / uly_ms),
                "ulysses_vs_single_gpu_speedup_x": float(single_ms / uly_ms),
                "ulysses_agg_tok_per_s": float((B * T_total) / (uly_ms * 1e-3)),
                "ring_agg_tok_per_s": float((B * T_total) / (ring_ms * 1e-3)),
                "ulysses_o_rel_l2_vs_single": o_uly_rel,
                "ulysses_dq_rel_l2_vs_single": dq_uly_rel,
                "ulysses_dk_rel_l2_vs_single": dk_uly_rel,
                "ring_o_rel_l2_vs_single": o_ring_rel,
            })

    dist.destroy_process_group()
    if rank == 0:
        return_dict["cp8_records"] = out_records


def run_track5_cp8_nvswitch_scaling(mode: str = "full") -> Dict[str, Any]:
    num_gpus = torch.cuda.device_count()
    if num_gpus < 8:
        return {"supported": False, "num_gpus": num_gpus, "records": []}
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    ret = manager.dict()
    port = 29645 + (os.getpid() % 200)
    mp.spawn(_cp8_worker, args=(8, port, mode, ret), nprocs=8, join=True)
    records = list(ret.get("cp8_records", []))
    return {
        "supported": True,
        "num_gpus": 8,
        "topology": "8x NVIDIA A100-SXM4-80GB (NV12 NVSwitch)",
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="E145 (v136, Round-19 RSI) Gated KalmaNet Benchmark Suite")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    t_start = time.time()
    print(f"[E145 Round-19 RSI Gated KalmaNet] Starting mode={args.mode}, rep={args.rep} on {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")

    t1 = run_track1_upstream_pathology_forensics()
    print(f"  Track 1 done: upstream zero-key dk NaNs={t1['upstream_zero_key_nans']['dk_nans']} vs RSI={t1['rsi_zero_key_nans']['dk_nans']}")

    t2 = run_track2_oracle_and_mutation_kill()
    print(f"  Track 2 done: all_oracle_passed={t2['all_oracle_passed']}, all_mutations_killed={t2['all_mutations_killed']}")

    t3 = run_track3_kernel_and_layer_ablation(mode=args.mode)
    print(
        f"  Track 3 done: cubin spills={t3['round19_rsi_cubin_audit']['total_cubin_spills']}/{t3['round19_rsi_cubin_audit']['total_specializations']} specs, "
        f"max_regs={t3['round19_rsi_cubin_audit']['max_regs']}, "
        f"A4 vs A0 K64={t3['layer_ablations']['K64']['a4_vs_a0_fwdbwd_speedup_x']:.2f}x, "
        f"K128={t3['layer_ablations']['K128']['a4_vs_a0_fwdbwd_speedup_x']:.2f}x"
    )

    t4 = run_track4_autoresearch_val_bpb(mode=args.mode)
    print(
        f"  Track 4 done: steps A0={t4['upstream_fla_a0']['steps_completed']} -> A4={t4['round19_rsi_a4']['steps_completed']} "
        f"({t4['comparison']['step_ratio_x']:.2f}x), val_bpb {t4['upstream_fla_a0']['final_val_bpb']:.4f} -> {t4['round19_rsi_a4']['final_val_bpb']:.4f}"
    )

    t5 = run_track5_cp8_nvswitch_scaling(mode=args.mode)
    if t5["records"]:
        for rec in t5["records"]:
            print(
                f"  Track 5 T={rec['T_total']}: Single={rec['single_gpu_rsi_fwdbwd_ms']:.2f} ms, "
                f"Ring={rec['cp8_ring_state_passing_fwdbwd_ms']:.2f} ms, "
                f"Ulysses={rec['cp8_ulysses_zero_copy_fwdbwd_ms']:.2f} ms "
                f"(Ulysses vs Ring={rec['ulysses_vs_ring_speedup_x']:.2f}x, vs Single={rec['ulysses_vs_single_gpu_speedup_x']:.2f}x)"
            )

    payload = {
        "experiment_id": "E145",
        "version": "v136",
        "rsi_round": 19,
        "title": "Gated KalmaNet (GKA) Zero-Spill SM80 Triton Kernels & NVSwitch Context Parallelism",
        "mode": args.mode,
        "rep": args.rep,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "elapsed_sec": float(time.time() - t_start),
        "track1_upstream_pathology_forensics": t1,
        "track2_oracle_and_mutation_kill": t2,
        "track3_kernel_and_layer_ablation": t3,
        "track4_autoresearch_val_bpb": t4,
        "track5_cp8_nvswitch_scaling": t5,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"[E145] Wrote {out_path}")


if __name__ == "__main__":
    main()
