#!/usr/bin/env python3
"""
Round-30 RSI (E156 / v147) Hardware Benchmark Harness:
Gated KalmaNet (GKA): A Fading Memory Layer Through Test-Time Ridge Regression
(arXiv:2511.21016 — Rui-Qi Wang,isen Lan, Songlin Yang; official repo fla/ops/gka & fla/layers/gka.py).

Executes 5 hardware verification & benchmark tracks on 8x NVIDIA A100-SXM4-80GB (SM80):
  - Track 1: Upstream Pathology & Bug Audit (24/44 Spilling Specs, 52,528 Spills, 10,028 Max Spills/Thread,
             SimpleGLABackwardGuard K<64/V<32 16-bit Backward Crash, TF32 Drift), SM80 Cubin Register/Spill Audit,
             FP64 Oracle Parity, 3 Mutation Kills, & 8-Shape Training Kernel Latency Matrix (chunk_gka_rsi vs chunk_gka)
  - Track 2: Single-Pass Multi-Warp Online Decoding (gka_decode_rsi vs fused_recurrent_gka & chunk_gka)
  - Track 3: 5-Stage Layer Periphery Ablation Ladder (Stages A0 -> A1 -> A2 -> A3 -> A4)
  - Track 4: 8-GPU NVSwitch Context Parallelism (GKA-CP = 8: Ulysses vs Ring/AllGather)
  - Track 5: Fixed-Wall-Clock Autoresearch Language-Modeling Loop (val_bpb & tokens/s)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ["FLA_CACHE_RESULTS"] = "1"

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if "/tmp/fla_repo" not in sys.path and Path("/tmp/fla_repo").exists():
    sys.path.insert(0, "/tmp/fla_repo")

from gka_rsi_kernels import (
    GatedKalmaNetBlockRSI,
    _collect_compiled_triton,
    audit_gka_rsi_cubin_resources,
    chunk_gka_rsi,
    gka_cp8_ring_baseline,
    gka_cp8_ulysses_rsi,
    gka_decode_rsi,
    gka_fp64_oracle,
)

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def _verify_frozen_harness() -> dict:
    harness_path = SCRIPT_DIR / "rsi_kernel_harness.py"
    sha = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    assert sha == EXPECTED_HARNESS_SHA256, f"Frozen harness SHA256 mismatch: {sha}"
    return {"harness_path": str(harness_path.name), "harness_sha256": sha, "verified": True}


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    ad = a.detach().double()
    bd = b.detach().double()
    return float((ad - bd).norm() / bd.norm().clamp_min(1e-12))


def _time_cuda_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _gka_training_flops(B: int, T: int, H: int, K: int, V: int, num_iter: int = 16, BT: int = 64):
    # Fwd: dual state (2*B*T*H*K*(K+V)), fro (2*B*T*H*(K*K + BT*K + BT^2)),
    #      Chebyshev solve (num_iter * 2*B*T*H*(K*K + 2*BT*K)), fwd_o (2*B*T*H*(K*V + BT*K + BT*V))
    fwd_per_tok = float(
        2 * K * (K + V)
        + 2 * (K * K + BT * K + BT * BT)
        + num_iter * 2 * (K * K + 2 * BT * K)
        + 2 * (K * V + BT * K + BT * V)
    )
    fwd = float(B * T * H) * fwd_per_tok
    bwd = 2.2 * fwd
    return fwd, bwd


def _audit_upstream_gka_pathologies(device: torch.device) -> dict:
    from fla.ops.gka import chunk_gka, fused_recurrent_gka
    from fla.ops.gka.chunk_solve_fwd import chunk_gka_solve_fwd_kernel
    from fla.ops.gka.chunk_solve_bwd import (
        chunk_gka_bwd_kernel_dk_inter,
        chunk_gka_bwd_kernel_dk_intra,
    )
    from fla.ops.gka.fused_recurrent import fused_recurrent_gka_solve_fwd_kernel
    from fla.ops.common.chunk_h import (
        chunk_fwd_kernel_h,
        chunk_bwd_kernel_dh,
    )
    from fla.ops.common.chunk_o import (
        chunk_fwd_kernel_o,
        chunk_bwd_kernel_dqkwg,
        chunk_bwd_kernel_dv,
    )
    from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel

    # 1. Verify SimpleGLABackwardGuard 16-bit backward crash for K < 64 or V < 32
    torch.manual_seed(0)
    q_s = F.normalize(torch.randn(1, 128, 2, 32, device=device, dtype=torch.bfloat16), dim=-1).detach().requires_grad_(True)
    k_s = (F.normalize(torch.randn(1, 128, 2, 32, device=device, dtype=torch.bfloat16), dim=-1) * 0.5).detach().requires_grad_(True)
    v_s = (torch.randn(1, 128, 2, 32, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    g_s = (-F.softplus(torch.randn(1, 128, 2, device=device, dtype=torch.float32) - 2.0)).detach().requires_grad_(True)
    a_s = torch.sigmoid(torch.randn(1, 128, 2, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    upstream_k32_bf16_bwd_crashed = False
    upstream_k32_bf16_error_msg = ""
    try:
        o_s, _ = chunk_gka(q_s, k_s, v_s, g=g_s, gk=g_s, alpha=a_s, ridge_ratio=0.02, num_iter=16)
        o_s.sum().backward()
    except RuntimeError as exc:
        upstream_k32_bf16_bwd_crashed = True
        upstream_k32_bf16_error_msg = str(exc)

    # Verify our chunk_gka_rsi succeeds on K=32, V=32 BF16 backward with finite gradients
    q_s2 = q_s.detach().clone().requires_grad_(True)
    k_s2 = k_s.detach().clone().requires_grad_(True)
    v_s2 = v_s.detach().clone().requires_grad_(True)
    g_s2 = g_s.detach().clone().requires_grad_(True)
    a_s2 = a_s.detach().clone().requires_grad_(True)
    o_s2, _ = chunk_gka_rsi(q_s2, k_s2, v_s2, g=g_s2, gk=g_s2, alpha=a_s2, ridge_ratio=0.02, num_iter=16)
    o_s2.sum().backward()
    rsi_k32_bf16_bwd_succeeded = bool(
        torch.isfinite(q_s2.grad).all().item()
        and torch.isfinite(k_s2.grad).all().item()
        and torch.isfinite(v_s2.grad).all().item()
    )

    # 2. Compile upstream chunk_gka & fused_recurrent_gka across BF16/FP32 and K=64, 128 and audit cubin spills + TF32 drift
    upstream_fp32_max_err = 0.0
    for dtype, K in [(torch.bfloat16, 64), (torch.bfloat16, 128), (torch.float32, 64)]:
        if True:
            B, T, H, V = 2, 128, 4, K
            q = (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.5).detach().requires_grad_(True)
            k = (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.25).detach().requires_grad_(True)
            v = (torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.5).detach().requires_grad_(True)
            g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.05 - 0.01).detach().requires_grad_(True)
            alpha = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32)).detach().requires_grad_(True)
            A_init = torch.randn(B, H, K, K, device=device, dtype=torch.float32) * 0.05
            h0_kk = (A_init @ A_init.transpose(-1, -2)).detach().requires_grad_(True)
            h0_kv = (torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.05).detach().requires_grad_(True)
            do = torch.randn_like(v) * 0.1
            o_u, _ = chunk_gka(
                q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=30,
                initial_state=(h0_kk, h0_kv), output_final_state=True
            )
            (o_u.float() * do.float()).sum().backward()
            if dtype == torch.float32 and K == 64:
                dk_u = k.grad.detach().clone()
                dg_u = g.grad.detach().clone()
                da_u = alpha.grad.detach().clone()
                q.grad = k.grad = v.grad = g.grad = alpha.grad = h0_kk.grad = h0_kv.grad = None
                o_o, _ = gka_fp64_oracle(
                    q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=30,
                    initial_state=(h0_kk, h0_kv), output_final_state=False
                )
                (o_o * do.double()).sum().backward()
                upstream_fp32_max_err = max(
                    _rel_l2(o_u, o_o),
                    _rel_l2(dk_u, k.grad),
                    _rel_l2(dg_u, g.grad),
                    _rel_l2(da_u, alpha.grad),
                )
            with torch.no_grad():
                fused_recurrent_gka(
                    q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=30,
                    initial_state=(h0_kk, h0_kv), output_final_state=True
                )

    up_kernels = [
        ("cumsum_scalar", chunk_local_cumsum_scalar_kernel),
        ("common_fwd_h", chunk_fwd_kernel_h),
        ("gka_solve_fwd", chunk_gka_solve_fwd_kernel),
        ("common_fwd_o", chunk_fwd_kernel_o),
        ("common_bwd_dh", chunk_bwd_kernel_dh),
        ("common_bwd_dqkwg", chunk_bwd_kernel_dqkwg),
        ("common_bwd_dv", chunk_bwd_kernel_dv),
        ("gka_bwd_dk_inter", chunk_gka_bwd_kernel_dk_inter),
        ("gka_bwd_dk_intra", chunk_gka_bwd_kernel_dk_intra),
        ("gka_fused_rec_solve", fused_recurrent_gka_solve_fwd_kernel),
    ]
    up_per_kernel = {}
    up_tot_specs = up_spilling = up_over168 = up_tot_spills = 0
    up_max_regs = up_max_spills = up_max_smem = 0
    for name, kobj in up_kernels:
        caches = _collect_compiled_triton(kobj)
        n_sp = len(caches)
        sp = sum(1 for c in caches if getattr(c, "n_spills", 0) > 0)
        ov = sum(1 for c in caches if getattr(c, "n_regs", 0) > 168)
        mr = max((int(getattr(c, "n_regs", 0)) for c in caches), default=0)
        ms = max((int(getattr(c, "n_spills", 0)) for c in caches), default=0)
        ts = sum(int(getattr(c, "n_spills", 0)) for c in caches)
        sm = max((int(getattr(c, "metadata", None).shared) if hasattr(c, "metadata") else 0 for c in caches), default=0)
        up_per_kernel[name] = {
            "specializations": n_sp,
            "spilling_specs": sp,
            "over168_specs": ov,
            "max_regs": mr,
            "max_spills": ms,
            "total_spills": ts,
            "max_smem_bytes": sm,
        }
        up_tot_specs += n_sp
        up_spilling += sp
        up_over168 += ov
        up_tot_spills += ts
        up_max_regs = max(up_max_regs, mr)
        up_max_spills = max(up_max_spills, ms)
        up_max_smem = max(up_max_smem, sm)

    return {
        "upstream_k32_bf16_bwd_crashed": upstream_k32_bf16_bwd_crashed,
        "upstream_k32_bf16_error_msg": upstream_k32_bf16_error_msg,
        "rsi_k32_bf16_bwd_succeeded": rsi_k32_bf16_bwd_succeeded,
        "upstream_fp32_tf32_max_rel_l2": upstream_fp32_max_err,
        "upstream_cubin_audit": {
            "kernel_count": len(up_kernels),
            "total_specializations": up_tot_specs,
            "spilling_specializations": up_spilling,
            "over168_specializations": up_over168,
            "total_spills": up_tot_spills,
            "max_regs": up_max_regs,
            "max_spills": up_max_spills,
            "max_smem_bytes": up_max_smem,
            "kernels": up_per_kernel,
        },
    }


def run_track1_kernels_and_oracle(device: torch.device, smoke: bool = False) -> dict:
    from fla.ops.gka import chunk_gka

    upstream_pathology = _audit_upstream_gka_pathologies(device)

    # 1A. FP64 Oracle Parity (FP32 and BF16 across K=64, 128, non-multiple T=100, and K=32, V=32)
    oracle_cases = []
    fp32_max_err = 0.0
    bf16_max_err = 0.0

    for (B, T_val, H, K, V) in [
        (1, 128, 2, 64, 64),
        (1, 128, 2, 128, 128),
        (1, 100, 2, 64, 64),
        (1, 128, 2, 32, 32),
    ]:
        for dt_name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
            torch.manual_seed(200 + K + T_val)
            q = (torch.randn(B, T_val, H, K, device=device, dtype=dt) * 0.5).detach().requires_grad_(True)
            k = (torch.randn(B, T_val, H, K, device=device, dtype=dt) * 0.25).detach().requires_grad_(True)
            v = (torch.randn(B, T_val, H, V, device=device, dtype=dt) * 0.5).detach().requires_grad_(True)
            g = (-torch.rand(B, T_val, H, device=device, dtype=torch.float32) * 0.05 - 0.01).detach().requires_grad_(True)
            alpha = torch.sigmoid(torch.randn(B, T_val, H, device=device, dtype=torch.float32)).detach().requires_grad_(True)
            A_init = torch.randn(B, H, K, K, device=device, dtype=torch.float32) * 0.05
            h0_kk = (A_init @ A_init.transpose(-1, -2)).detach().requires_grad_(True)
            h0_kv = (torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.05).detach().requires_grad_(True)
            do = torch.randn(B, T_val, H, V, device=device, dtype=dt) * 0.1
            dht_kk = torch.randn(B, H, K, K, device=device, dtype=torch.float32) * 0.05
            dht_kk = 0.5 * (dht_kk + dht_kk.transpose(-1, -2))
            dht_kv = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.05

            o_ref, (ht_kk_ref, ht_kv_ref) = gka_fp64_oracle(
                q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=30,
                initial_state=(h0_kk, h0_kv), output_final_state=True
            )
            ((o_ref * do.double()).sum() + (ht_kk_ref * dht_kk.double()).sum() + (ht_kv_ref * dht_kv.double()).sum()).backward()
            refs = {
                "o": o_ref.detach(),
                "ht_kk": ht_kk_ref.detach(),
                "ht_kv": ht_kv_ref.detach(),
                "dq": q.grad.detach().clone(),
                "dk": k.grad.detach().clone(),
                "dv": v.grad.detach().clone(),
                "dg": g.grad.detach().clone(),
                "dalpha": alpha.grad.detach().clone(),
                "dh0_kk": h0_kk.grad.detach().clone(),
                "dh0_kv": h0_kv.grad.detach().clone(),
            }
            q.grad = k.grad = v.grad = g.grad = alpha.grad = h0_kk.grad = h0_kv.grad = None

            o_rsi, (ht_kk_rsi, ht_kv_rsi) = chunk_gka_rsi(
                q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=30,
                initial_state=(h0_kk, h0_kv), output_final_state=True
            )
            ((o_rsi.float() * do.float()).sum() + (ht_kk_rsi.float() * dht_kk.float()).sum() + (ht_kv_rsi.float() * dht_kv.float()).sum()).backward()
            got = {
                "o": o_rsi.detach(),
                "ht_kk": ht_kk_rsi.detach(),
                "ht_kv": ht_kv_rsi.detach(),
                "dq": q.grad.detach().clone(),
                "dk": k.grad.detach().clone(),
                "dv": v.grad.detach().clone(),
                "dg": g.grad.detach().clone(),
                "dalpha": alpha.grad.detach().clone(),
                "dh0_kk": h0_kk.grad.detach().clone(),
                "dh0_kv": h0_kv.grad.detach().clone(),
            }
            errs = {name: _rel_l2(got[name], refs[name]) for name in refs}
            case_max = max(errs.values())
            if dt == torch.float32:
                fp32_max_err = max(fp32_max_err, case_max)
            else:
                bf16_max_err = max(bf16_max_err, case_max)

            oracle_cases.append(
                {
                    "K": K,
                    "V": V,
                    "T": T_val,
                    "H": H,
                    "dtype": dt_name,
                    "max_rel_l2": case_max,
                    "errors": errs,
                }
            )

    assert fp32_max_err < 1e-5, f"FP32 max_rel_l2 {fp32_max_err} >= 1e-5"
    assert bf16_max_err < 1.5e-2, f"BF16 max_rel_l2 {bf16_max_err} >= 1.5e-2"

    # 3 Adversarial Mutation Modes
    torch.manual_seed(99)
    v_dir = torch.randn(1, 1, 2, 64, device=device, dtype=torch.float32)
    v_dir = v_dir / v_dir.norm(dim=-1, keepdim=True)
    qm = (0.7 * v_dir + 0.2 * torch.randn(1, 128, 2, 64, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    km = (0.7 * v_dir + 0.05 * torch.randn(1, 128, 2, 64, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    vm = (torch.randn(1, 128, 2, 64, device=device, dtype=torch.float32) * 0.5).detach().requires_grad_(True)
    gm = (-torch.rand(1, 128, 2, device=device, dtype=torch.float32) * 0.02 - 0.005).detach().requires_grad_(True)
    am = (torch.full((1, 128, 2), 0.85, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    dom = torch.randn_like(vm)

    o_mref, _ = gka_fp64_oracle(qm, km, vm, g=gm, gk=gm, alpha=am, ridge_ratio=0.02, num_iter=16, output_final_state=False)
    (o_mref * dom.double()).sum().backward()
    ref_dq, ref_dk, ref_dv, ref_dg, ref_da = (
        qm.grad.detach().clone(),
        km.grad.detach().clone(),
        vm.grad.detach().clone(),
        gm.grad.detach().clone(),
        am.grad.detach().clone(),
    )

    mutations = []
    for m_id, m_name in [
        (1, "M1_drop_chebyshev_momentum_rho_sq4_zero"),
        (2, "M2_drop_frobenius_ridge_penalty_lambda_zero"),
        (3, "M3_invert_query_mixing_gate_alpha"),
    ]:
        qc = qm.detach().clone().requires_grad_(True)
        kc = km.detach().clone().requires_grad_(True)
        vc = vm.detach().clone().requires_grad_(True)
        gc = gm.detach().clone().requires_grad_(True)
        ac = am.detach().clone().requires_grad_(True)
        o_mut, _ = chunk_gka_rsi(qc, kc, vc, g=gc, gk=gc, alpha=ac, ridge_ratio=0.02, num_iter=16, mutation_mode=m_id)
        (o_mut * dom).sum().backward()
        m_err = max(
            _rel_l2(o_mut, o_mref),
            _rel_l2(qc.grad, ref_dq),
            _rel_l2(kc.grad, ref_dk),
            _rel_l2(vc.grad, ref_dv),
            _rel_l2(gc.grad, ref_dg),
            _rel_l2(ac.grad, ref_da),
        )
        killed = m_err > 5e-2
        assert killed, f"Mutation {m_name} survived with err={m_err}"
        mutations.append({"id": m_id, "name": m_name, "max_rel_l2": m_err, "killed": bool(killed)})

    # 1C. 8-Shape Training Kernel Matrix
    shapes = [
        (2, 512, 16, 64, 64, 16),
        (2, 1024, 16, 64, 64, 16),
        (2, 2048, 16, 64, 64, 16),
        (4, 2048, 16, 64, 64, 16),
        (2, 512, 8, 128, 128, 16),
        (2, 1024, 8, 128, 128, 16),
        (2, 2048, 8, 128, 128, 16),
        (4, 2048, 16, 128, 128, 16),
    ]
    if smoke:
        shapes = [shapes[0], shapes[1], shapes[4], shapes[5]]
    warmup = 4 if smoke else 6
    iters = 8 if smoke else 20

    kernel_matrix = []
    for B, T, H, K, V, n_it in shapes:
        q = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), dim=-1)
        k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), dim=-1) * 0.5
        v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.5
        g = -F.softplus(torch.randn(B, T, H, device=device, dtype=torch.float32) - 2.0)
        alpha = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
        do = torch.randn_like(v)

        up_fwd_ms = _time_cuda_ms(
            lambda: chunk_gka(q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=n_it),
            warmup=warmup,
            iters=iters,
        )

        def _up_tot():
            qc = q.detach().requires_grad_(True)
            kc = k.detach().requires_grad_(True)
            vc = v.detach().requires_grad_(True)
            gc = g.detach().requires_grad_(True)
            ac = alpha.detach().requires_grad_(True)
            oc, _ = chunk_gka(qc, kc, vc, g=gc, gk=gc, alpha=ac, ridge_ratio=0.02, num_iter=n_it)
            oc.backward(do)

        up_tot_ms = _time_cuda_ms(_up_tot, warmup=warmup, iters=iters)
        up_bwd_ms = max(up_tot_ms - up_fwd_ms, 1e-4)

        rsi_fwd_ms = _time_cuda_ms(
            lambda: chunk_gka_rsi(q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=n_it),
            warmup=warmup,
            iters=iters,
        )

        def _rsi_tot():
            qc = q.detach().requires_grad_(True)
            kc = k.detach().requires_grad_(True)
            vc = v.detach().requires_grad_(True)
            gc = g.detach().requires_grad_(True)
            ac = alpha.detach().requires_grad_(True)
            oc, _ = chunk_gka_rsi(qc, kc, vc, g=gc, gk=gc, alpha=ac, ridge_ratio=0.02, num_iter=n_it)
            oc.backward(do)

        rsi_tot_ms = _time_cuda_ms(_rsi_tot, warmup=warmup, iters=iters)
        rsi_bwd_ms = max(rsi_tot_ms - rsi_fwd_ms, 1e-4)

        fwd_flops, bwd_flops = _gka_training_flops(B, T, H, K, V, num_iter=n_it)
        tot_flops = fwd_flops + bwd_flops

        row = {
            "shape": f"B={B}_T={T}_H={H}_K={K}_V={V}_N={n_it}",
            "B": B,
            "T": T,
            "H": H,
            "K": K,
            "V": V,
            "num_iter": n_it,
            "upstream_fwd_ms": up_fwd_ms,
            "upstream_bwd_ms": up_bwd_ms,
            "upstream_total_ms": up_tot_ms,
            "rsi_fwd_ms": rsi_fwd_ms,
            "rsi_bwd_ms": rsi_bwd_ms,
            "rsi_total_ms": rsi_tot_ms,
            "fwd_speedup_vs_upstream": up_fwd_ms / rsi_fwd_ms,
            "bwd_speedup_vs_upstream": up_bwd_ms / rsi_bwd_ms,
            "total_speedup_vs_upstream": up_tot_ms / rsi_tot_ms,
            "rsi_total_tflops": (tot_flops / (rsi_tot_ms * 1e-3)) / 1e12,
        }
        kernel_matrix.append(row)

    return {
        "upstream_pathology": upstream_pathology,
        "fp32_max_rel_l2": fp32_max_err,
        "bf16_max_rel_l2": bf16_max_err,
        "oracle_cases": oracle_cases,
        "mutations": mutations,
        "mutations_killed": sum(1 for m in mutations if m["killed"]),
        "mutations_total": len(mutations),
        "kernel_matrix": kernel_matrix,
    }


def run_track2_decode(device: torch.device, smoke: bool = False) -> dict:
    from fla.ops.gka import chunk_gka, fused_recurrent_gka

    shapes = [
        (4, 1, 16, 64, 64, 16),
        (4, 4, 16, 64, 64, 16),
        (4, 16, 16, 64, 64, 16),
        (8, 16, 16, 64, 64, 16),
        (4, 1, 16, 128, 128, 16),
        (4, 4, 16, 128, 128, 16),
        (4, 16, 16, 128, 128, 16),
        (8, 16, 16, 128, 128, 16),
    ]
    if smoke:
        shapes = [shapes[0], shapes[2], shapes[4], shapes[6]]
    warmup = 4 if smoke else 6
    iters = 12 if smoke else 30

    rows = []
    for B, T_q, H, K, V, n_it in shapes:
        torch.manual_seed(300 + B + T_q + K)
        q = F.normalize(torch.randn(B, T_q, H, K, device=device, dtype=torch.bfloat16), dim=-1)
        k = F.normalize(torch.randn(B, T_q, H, K, device=device, dtype=torch.bfloat16), dim=-1) * 0.5
        v = torch.randn(B, T_q, H, V, device=device, dtype=torch.bfloat16) * 0.5
        g = -F.softplus(torch.randn(B, T_q, H, device=device, dtype=torch.float32) - 2.0)
        alpha = torch.sigmoid(torch.randn(B, T_q, H, device=device, dtype=torch.float32))
        A_init = torch.randn(B, H, K, K, device=device, dtype=torch.float32) * 0.05
        h0_kk = A_init @ A_init.transpose(-1, -2)
        h0_kv = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.05

        with torch.no_grad():
            o_ref, (ht_kk_ref, ht_kv_ref) = gka_fp64_oracle(
                q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=n_it,
                initial_state=(h0_kk, h0_kv), output_final_state=True
            )
            o_dec, (ht_kk_dec, ht_kv_dec) = gka_decode_rsi(
                q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=n_it,
                initial_state=(h0_kk, h0_kv), output_final_state=True
            )
            o_rel = _rel_l2(o_dec, o_ref)
            ht_kk_rel = _rel_l2(ht_kk_dec, ht_kk_ref)
            ht_kv_rel = _rel_l2(ht_kv_dec, ht_kv_ref)

            up_chunk_ms = _time_cuda_ms(
                lambda: chunk_gka(
                    q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=n_it,
                    initial_state=(h0_kk, h0_kv), output_final_state=True
                ),
                warmup=warmup,
                iters=iters,
            )
            up_rec_ms = _time_cuda_ms(
                lambda: fused_recurrent_gka(
                    q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=n_it,
                    initial_state=(h0_kk, h0_kv), output_final_state=True
                ),
                warmup=warmup,
                iters=iters,
            )
            rsi_dec_ms = _time_cuda_ms(
                lambda: gka_decode_rsi(
                    q, k, v, g=g, gk=g, alpha=alpha, ridge_ratio=0.02, num_iter=n_it,
                    initial_state=(h0_kk, h0_kv), output_final_state=True
                ),
                warmup=warmup,
                iters=iters,
            )

        rows.append(
            {
                "shape": f"B={B}_Tq={T_q}_H={H}_K={K}",
                "B": B,
                "T_q": T_q,
                "H": H,
                "K": K,
                "V": V,
                "upstream_chunk_ms": up_chunk_ms,
                "upstream_fused_recurrent_ms": up_rec_ms,
                "rsi_decode_ms": rsi_dec_ms,
                "speedup_vs_upstream_chunk": up_chunk_ms / rsi_dec_ms,
                "speedup_vs_upstream_recurrent": up_rec_ms / rsi_dec_ms,
                "o_rel_l2": o_rel,
                "ht_kk_rel_l2": ht_kk_rel,
                "ht_kv_rel_l2": ht_kv_rel,
            }
        )
    return {"decode_matrix": rows}


def run_track3_layer_ablation(device: torch.device, smoke: bool = False) -> dict:
    configs = [
        {"D_model": 1024, "H": 16, "K": 64, "B": 2, "T": 1024},
        {"D_model": 2048, "H": 16, "K": 128, "B": 2, "T": 1024},
    ]
    warmup = 3 if smoke else 6
    iters = 8 if smoke else 20

    ablation_rows = []
    for cfg in configs:
        D_model, H, K, B, T = cfg["D_model"], cfg["H"], cfg["K"], cfg["B"], cfg["T"]
        torch.manual_seed(400 + D_model)
        base_blk = GatedKalmaNetBlockRSI(
            hidden_size=D_model, num_heads=H, head_dim=K, num_iter=16, stage="A0"
        ).to(device).to(torch.bfloat16)
        x = torch.randn(B, T, D_model, device=device, dtype=torch.bfloat16)
        dy = torch.randn_like(x)

        with torch.no_grad():
            y_a0 = base_blk(x)

        a0_tot_ms = None
        for stg in ["A0", "A1", "A2", "A3", "A4"]:
            base_blk.stage = stg
            base_blk.sync_packed_weights()

            with torch.no_grad():
                y_stg = base_blk(x)
                parity_vs_a0 = _rel_l2(y_stg, y_a0)
                fwd_ms = _time_cuda_ms(
                    lambda: base_blk(x),
                    warmup=warmup,
                    iters=iters,
                )

            def _step():
                base_blk.zero_grad(set_to_none=True)
                xc = x.detach().requires_grad_(True)
                ys = base_blk(xc)
                ys.backward(dy)

            tot_ms = _time_cuda_ms(_step, warmup=warmup, iters=iters)
            bwd_ms = max(tot_ms - fwd_ms, 1e-4)
            if stg == "A0":
                a0_tot_ms = tot_ms

            ablation_rows.append(
                {
                    "D_model": D_model,
                    "H": H,
                    "K": K,
                    "B": B,
                    "T": T,
                    "stage": stg,
                    "fwd_ms": fwd_ms,
                    "bwd_ms": bwd_ms,
                    "total_ms": tot_ms,
                    "tokens_per_s": (B * T) / (tot_ms * 1e-3),
                    "speedup_vs_a0": a0_tot_ms / tot_ms,
                    "rel_l2_vs_a0": parity_vs_a0,
                }
            )
    return {"layer_ablation": ablation_rows}


def _cp8_worker_main(rank: int, world_size: int, out_path: str, smoke: bool):
    from fla.ops.gka import chunk_gka

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29576"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    warmup = 2 if smoke else 4
    iters = 4 if smoke else 8

    cases = [
        (4, 8192, 32, 64, 64),
        (4, 16384, 32, 64, 64),
        (4, 8192, 32, 128, 128),
    ]
    # Pre-warm Triton cubin loading & upstream @triton.autotune on each rank before any timed/collective loop
    for K_w in [64, 128]:
        for H_w, fn_w in [(32, chunk_gka), (32, chunk_gka_rsi), (4, chunk_gka_rsi)]:
            qw = F.normalize(torch.randn(1, 128, H_w, K_w, device=device, dtype=torch.bfloat16), dim=-1).detach().requires_grad_(True)
            kw = (F.normalize(torch.randn(1, 128, H_w, K_w, device=device, dtype=torch.bfloat16), dim=-1) * 0.5).detach().requires_grad_(True)
            vw = (torch.randn(1, 128, H_w, K_w, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
            gw = (-F.softplus(torch.randn(1, 128, H_w, device=device, dtype=torch.float32) - 2.0)).detach().requires_grad_(True)
            aw = torch.sigmoid(torch.randn(1, 128, H_w, device=device, dtype=torch.float32)).detach().requires_grad_(True)
            ow, _ = fn_w(qw, kw, vw, g=gw, gk=gw, alpha=aw, ridge_ratio=0.02, num_iter=16)
            ow.sum().backward()
            del qw, kw, vw, gw, aw, ow
    torch.cuda.synchronize()
    dist.barrier()

    rows = []
    for B, T_total, H, K, V in cases:
        T_local = T_total // world_size
        # 1. Exact parity check on T_verify = 1024 slice
        T_ver = 1024
        T_ver_loc = T_ver // world_size
        torch.manual_seed(500 + T_total + K)
        qv_all = F.normalize(torch.randn(2, T_ver, H, K, device=device, dtype=torch.bfloat16), dim=-1)
        kv_all = F.normalize(torch.randn(2, T_ver, H, K, device=device, dtype=torch.bfloat16), dim=-1) * 0.5
        vv_all = torch.randn(2, T_ver, H, V, device=device, dtype=torch.bfloat16) * 0.5
        gv_all = (-F.softplus(torch.randn(2, T_ver, H, device=device, dtype=torch.float32) - 2.0)).to(torch.bfloat16).float()
        av_all = torch.sigmoid(torch.randn(2, T_ver, H, device=device, dtype=torch.float32)).to(torch.bfloat16).float()
        dov_all = torch.randn_like(vv_all)
        sl_v = slice(rank * T_ver_loc, (rank + 1) * T_ver_loc)

        q1 = qv_all.detach().clone().requires_grad_(True)
        k1 = kv_all.detach().clone().requires_grad_(True)
        v1 = vv_all.detach().clone().requires_grad_(True)
        g1 = gv_all.detach().clone().requires_grad_(True)
        a1 = av_all.detach().clone().requires_grad_(True)
        o1, _ = chunk_gka_rsi(q1, k1, v1, g=g1, gk=g1, alpha=a1, ridge_ratio=0.02, num_iter=16)
        o1.backward(dov_all)

        ql_v = qv_all[:, sl_v].detach().clone().requires_grad_(True)
        kl_v = kv_all[:, sl_v].detach().clone().requires_grad_(True)
        vl_v = vv_all[:, sl_v].detach().clone().requires_grad_(True)
        gl_v = gv_all[:, sl_v].detach().clone().requires_grad_(True)
        al_v = av_all[:, sl_v].detach().clone().requires_grad_(True)
        dol_v = dov_all[:, sl_v].detach().clone()

        o_cp = gka_cp8_ulysses_rsi(
            ql_v, kl_v, vl_v, gl_v, al_v, group=group, world_size=world_size, ridge_ratio=0.02, num_iter=16
        )
        o_cp.backward(dol_v)

        rel_o = _rel_l2(o_cp, o1[:, sl_v])
        rel_dq = _rel_l2(ql_v.grad, q1.grad[:, sl_v])
        rel_dk = _rel_l2(kl_v.grad, k1.grad[:, sl_v])
        rel_dv = _rel_l2(vl_v.grad, v1.grad[:, sl_v])
        del qv_all, kv_all, vv_all, gv_all, av_all, dov_all, q1, k1, v1, g1, a1, o1, o_cp

        # 2. Time 1-GPU upstream chunk_gka & 1-GPU chunk_gka_rsi on rank 0
        if rank == 0:
            q_all = F.normalize(torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16), dim=-1)
            k_all = F.normalize(torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16), dim=-1) * 0.5
            v_all = torch.randn(B, T_total, H, V, device=device, dtype=torch.bfloat16) * 0.5
            g_all = -F.softplus(torch.randn(B, T_total, H, device=device, dtype=torch.float32) - 2.0)
            a_all = torch.sigmoid(torch.randn(B, T_total, H, device=device, dtype=torch.float32))
            do_all = torch.randn_like(v_all)

            def _one_gpu_up():
                qc = q_all.detach().requires_grad_(True)
                kc = k_all.detach().requires_grad_(True)
                vc = v_all.detach().requires_grad_(True)
                gc = g_all.detach().requires_grad_(True)
                ac = a_all.detach().requires_grad_(True)
                oc, _ = chunk_gka(qc, kc, vc, g=gc, gk=gc, alpha=ac, ridge_ratio=0.02, num_iter=16)
                oc.backward(do_all)

            one_gpu_up_ms = _time_cuda_ms(_one_gpu_up, warmup=1, iters=3)

            def _one_gpu_rsi():
                qc = q_all.detach().requires_grad_(True)
                kc = k_all.detach().requires_grad_(True)
                vc = v_all.detach().requires_grad_(True)
                gc = g_all.detach().requires_grad_(True)
                ac = a_all.detach().requires_grad_(True)
                oc, _ = chunk_gka_rsi(qc, kc, vc, g=gc, gk=gc, alpha=ac, ridge_ratio=0.02, num_iter=16)
                oc.backward(do_all)

            one_gpu_rsi_ms = _time_cuda_ms(_one_gpu_rsi, warmup=1, iters=3)
            del q_all, k_all, v_all, g_all, a_all, do_all
            torch.cuda.empty_cache()
        else:
            one_gpu_up_ms = 0.0
            one_gpu_rsi_ms = 0.0
        dist.barrier()

        # Allocate local sequence shard on each rank
        torch.manual_seed(600 + rank * 17 + T_total + K)
        ql = F.normalize(torch.randn(B, T_local, H, K, device=device, dtype=torch.bfloat16), dim=-1)
        kl = F.normalize(torch.randn(B, T_local, H, K, device=device, dtype=torch.bfloat16), dim=-1) * 0.5
        vl = torch.randn(B, T_local, H, V, device=device, dtype=torch.bfloat16) * 0.5
        gl = -F.softplus(torch.randn(B, T_local, H, device=device, dtype=torch.float32) - 2.0)
        al = torch.sigmoid(torch.randn(B, T_local, H, device=device, dtype=torch.float32))
        dol = torch.randn_like(vl)

        # Pre-warm upstream chunk_gka (H=32) and chunk_gka_rsi (H_local=4) on each rank sequentially/locally
        # so @triton.autotune and Triton JIT never run mid-NCCL collective
        qw = ql[:, :128].detach().clone().requires_grad_(True)
        kw = kl[:, :128].detach().clone().requires_grad_(True)
        vw = vl[:, :128].detach().clone().requires_grad_(True)
        gw = gl[:, :128].detach().clone().requires_grad_(True)
        aw = al[:, :128].detach().clone().requires_grad_(True)
        ow, _ = chunk_gka(qw, kw, vw, g=gw, gk=gw, alpha=aw, ridge_ratio=0.02, num_iter=16)
        ow.backward(dol[:, :128])
        H_loc = H // world_size
        qwu = ql[:, :128, :H_loc].contiguous().detach().requires_grad_(True)
        kwu = kl[:, :128, :H_loc].contiguous().detach().requires_grad_(True)
        vwu = vl[:, :128, :H_loc].contiguous().detach().requires_grad_(True)
        gwu = gl[:, :128, :H_loc].contiguous().detach().requires_grad_(True)
        awu = al[:, :128, :H_loc].contiguous().detach().requires_grad_(True)
        owu, _ = chunk_gka_rsi(qwu, kwu, vwu, g=gwu, gk=gwu, alpha=awu, ridge_ratio=0.02, num_iter=16)
        owu.backward(dol[:, :128, :H_loc].contiguous())
        del qw, kw, vw, gw, aw, ow, qwu, kwu, vwu, gwu, awu, owu
        torch.cuda.synchronize()
        dist.barrier()

        def _ring_step():
            qc = ql.detach().requires_grad_(True)
            kc = kl.detach().requires_grad_(True)
            vc = vl.detach().requires_grad_(True)
            gc = gl.detach().requires_grad_(True)
            ac = al.detach().requires_grad_(True)
            oc = gka_cp8_ring_baseline(
                qc, kc, vc, gc, ac, group=group, world_size=world_size, rank=rank, ridge_ratio=0.02, num_iter=16
            )
            oc.backward(dol)

        for _ in range(warmup):
            _ring_step()
        dist.barrier()
        torch.cuda.synchronize()
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        for _ in range(iters):
            _ring_step()
        e_ev.record()
        torch.cuda.synchronize()
        ring_ms = float(s_ev.elapsed_time(e_ev) / iters)

        def _ulysses_step():
            qc = ql.detach().requires_grad_(True)
            kc = kl.detach().requires_grad_(True)
            vc = vl.detach().requires_grad_(True)
            gc = gl.detach().requires_grad_(True)
            ac = al.detach().requires_grad_(True)
            oc = gka_cp8_ulysses_rsi(
                qc, kc, vc, gc, ac, group=group, world_size=world_size, ridge_ratio=0.02, num_iter=16
            )
            oc.backward(dol)

        for _ in range(warmup):
            _ulysses_step()
        dist.barrier()
        torch.cuda.synchronize()
        s_ev.record()
        for _ in range(iters):
            _ulysses_step()
        e_ev.record()
        torch.cuda.synchronize()
        ulysses_ms = float(s_ev.elapsed_time(e_ev) / iters)

        if rank == 0:
            rows.append(
                {
                    "B": B,
                    "T_total": T_total,
                    "T_local": T_local,
                    "H": H,
                    "K": K,
                    "V": V,
                    "one_gpu_upstream_ms": one_gpu_up_ms,
                    "one_gpu_rsi_ms": one_gpu_rsi_ms,
                    "ring_cp8_ms": ring_ms,
                    "ulysses_cp8_ms": ulysses_ms,
                    "speedup_vs_ring": ring_ms / ulysses_ms,
                    "speedup_vs_1gpu_upstream": one_gpu_up_ms / ulysses_ms,
                    "speedup_vs_1gpu_rsi": one_gpu_rsi_ms / ulysses_ms,
                    "parallel_efficiency_vs_1gpu_upstream_pct": 100.0 * (one_gpu_up_ms / ulysses_ms) / world_size,
                    "ulysses_tokens_per_s": (B * T_total) / (ulysses_ms * 1e-3),
                    "rel_l2_o": rel_o,
                    "rel_l2_dq": rel_dq,
                    "rel_l2_dk": rel_dk,
                    "rel_l2_dv": rel_dv,
                }
            )
    if rank == 0:
        Path(out_path).write_text(json.dumps({"cp8_scaling": rows}, indent=2), encoding="utf-8")
    dist.destroy_process_group()


def _spawn_cp8_workers(out_path: str, smoke: bool):
    import torch.multiprocessing as mp

    world_size = 8
    mp.spawn(_cp8_worker_main, args=(world_size, out_path, smoke), nprocs=world_size, join=True)


def run_track4_cp8(smoke: bool = False) -> dict:
    from fla.ops.gka import chunk_gka

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 8, f"Track 4 requires 8 GPUs, found {n_gpus}"
    device0 = torch.device("cuda:0")
    # Compile all H=32 and H=4 specializations into /home/azureuser/.triton/cache on rank 0 before spawning 8 workers
    for K_w in [64, 128]:
        for H_w, fn_w in [(32, chunk_gka), (32, chunk_gka_rsi), (4, chunk_gka_rsi)]:
            qw = F.normalize(torch.randn(1, 128, H_w, K_w, device=device0, dtype=torch.bfloat16), dim=-1).detach().requires_grad_(True)
            kw = (F.normalize(torch.randn(1, 128, H_w, K_w, device=device0, dtype=torch.bfloat16), dim=-1) * 0.5).detach().requires_grad_(True)
            vw = (torch.randn(1, 128, H_w, K_w, device=device0, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
            gw = (-F.softplus(torch.randn(1, 128, H_w, device=device0, dtype=torch.float32) - 2.0)).detach().requires_grad_(True)
            aw = torch.sigmoid(torch.randn(1, 128, H_w, device=device0, dtype=torch.float32)).detach().requires_grad_(True)
            ow, _ = fn_w(qw, kw, vw, g=gw, gk=gw, alpha=aw, ridge_ratio=0.02, num_iter=16)
            ow.sum().backward()
            del qw, kw, vw, gw, aw, ow
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    tmp_out = Path("/tmp/_gka_cp8_result.json")
    if tmp_out.exists():
        tmp_out.unlink()

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--cp8-worker",
        "--cp8-out",
        str(tmp_out),
    ]
    if smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True)
    data = json.loads(tmp_out.read_text(encoding="utf-8"))
    tmp_out.unlink(missing_ok=True)
    return data


class TinyGKALM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, head_dim: int, stage: str):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blk1 = GatedKalmaNetBlockRSI(
            hidden_size=d_model, num_heads=n_heads, head_dim=head_dim, num_iter=16, stage=stage
        )
        self.blk2 = GatedKalmaNetBlockRSI(
            hidden_size=d_model, num_heads=n_heads, head_dim=head_dim, num_iter=16, stage=stage
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = self.blk1(h)
        h = self.blk2(h)
        return self.lm_head(h)


def run_track5_autoresearch(device: torch.device, smoke: bool = False) -> dict:
    budget_s = 6.0 if smoke else 12.0
    vocab_size = 256
    B, T = 4, 512
    idx_t = torch.arange(T + 1, device=device)

    def _sample_batch(gen: torch.Generator) -> torch.Tensor:
        base = torch.randint(0, vocab_size, (B, 32), device=device, generator=gen)
        seq = (base[:, idx_t % 32] + (idx_t.unsqueeze(0) // 32) * 31) % vocab_size
        seq[:, 3:] = (seq[:, 3:] ^ (seq[:, :-3] & 0x3F)) % vocab_size
        return seq

    rows = []
    for d_model, n_heads, head_dim in [(1024, 16, 64), (2048, 16, 128)]:
        g_val = torch.Generator(device=device).manual_seed(5000 + d_model)
        val_batches = [_sample_batch(g_val) for _ in range(8)]

        @torch.no_grad()
        def _eval_bpb(m: nn.Module) -> float:
            tot_loss = 0.0
            for vb in val_batches:
                logits = m(vb[:, :-1])
                tot_loss += float(
                    F.cross_entropy(logits.float().reshape(-1, vocab_size), vb[:, 1:].reshape(-1)).item()
                )
            return (tot_loss / len(val_batches)) / math.log(2.0)

        torch.manual_seed(700 + d_model)
        init_model = TinyGKALM(vocab_size, d_model, n_heads, head_dim, stage="A0").to(device).to(torch.bfloat16)
        with torch.no_grad():
            nn.init.normal_(init_model.embed.weight, std=0.02)
            nn.init.normal_(init_model.lm_head.weight, std=0.02)
            for blk in [init_model.blk1, init_model.blk2]:
                for p in [
                    blk.q_proj.weight,
                    blk.k_proj.weight,
                    blk.v_proj.weight,
                    blk.g_proj.weight,
                    blk.b_proj.weight,
                    blk.alpha_proj.weight,
                    blk.a_proj.weight,
                    blk.o_proj.weight,
                ]:
                    nn.init.normal_(p, std=0.02)
                blk.sync_packed_weights()
        init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}

        res_pair = {}
        base_lr = 8e-4 if d_model == 1024 else 6e-4
        for stg in ["A0", "A4"]:
            model = TinyGKALM(vocab_size, d_model, n_heads, head_dim, stage=stg).to(device).to(torch.bfloat16)
            model.load_state_dict(init_state)
            model.blk1.sync_packed_weights()
            model.blk2.sync_packed_weights()
            if stg == "A0":
                params = [p for n, p in model.named_parameters() if ".qkvbaag_proj." not in n]
            else:
                params = [
                    p for n, p in model.named_parameters()
                    if not any(
                        x in n
                        for x in (".q_proj.", ".k_proj.", ".v_proj.", ".g_proj.", ".b_proj.", ".alpha_proj.", ".a_proj.")
                    )
                ]
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.01)

            g_warm = torch.Generator(device=device).manual_seed(5200 + d_model)
            for _ in range(2):
                wb = _sample_batch(g_warm)
                opt.zero_grad(set_to_none=True)
                logits = model(wb[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), wb[:, 1:].reshape(-1))
                loss.backward()
            model.load_state_dict(init_state)
            model.blk1.sync_packed_weights()
            model.blk2.sync_packed_weights()
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.01)

            init_bpb = _eval_bpb(model)
            g_train = torch.Generator(device=device).manual_seed(6000 + d_model)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            steps = 0
            while (time.perf_counter() - t0) < budget_s:
                tb = _sample_batch(g_train)
                opt.zero_grad(set_to_none=True)
                logits = model(tb[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), tb[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                torch.cuda.synchronize()
                steps += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            final_bpb = _eval_bpb(model)

            res_pair[stg] = {
                "steps": steps,
                "elapsed_s": elapsed,
                "tokens_per_s": (steps * B * T) / elapsed,
                "init_val_bpb": init_bpb,
                "final_val_bpb": final_bpb,
            }

        rows.append(
            {
                "D_model": d_model,
                "H": n_heads,
                "K": head_dim,
                "budget_s": budget_s,
                "a0_steps": res_pair["A0"]["steps"],
                "a4_steps": res_pair["A4"]["steps"],
                "step_ratio_a4_vs_a0": res_pair["A4"]["steps"] / max(res_pair["A0"]["steps"], 1),
                "a0_tokens_per_s": res_pair["A0"]["tokens_per_s"],
                "a4_tokens_per_s": res_pair["A4"]["tokens_per_s"],
                "init_val_bpb": res_pair["A0"]["init_val_bpb"],
                "a0_final_val_bpb": res_pair["A0"]["final_val_bpb"],
                "a4_final_val_bpb": res_pair["A4"]["final_val_bpb"],
                "extra_val_bpb_reduction": res_pair["A0"]["final_val_bpb"] - res_pair["A4"]["final_val_bpb"],
            }
        )
    return {"autoresearch": rows}


def main():
    if "--cp8-worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--cp8-worker", action="store_true")
        parser.add_argument("--cp8-out", type=str, required=True)
        parser.add_argument("--smoke", action="store_true")
        args = parser.parse_args()
        _spawn_cp8_workers(args.cp8_out, args.smoke)
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    harness_info = _verify_frozen_harness()
    print("[E156/v147] Running Track 1 (Upstream Pathology Audit, FP64 Oracle, Mutations, & 8-Shape Matrix)...", flush=True)
    t1 = run_track1_kernels_and_oracle(device, smoke=args.smoke)
    print("[E156/v147] Running Track 2 (Single-Pass Online Decoding Matrix)...", flush=True)
    t2 = run_track2_decode(device, smoke=args.smoke)
    print("[E156/v147] Running Track 3 (5-Stage GatedKalmaNetBlockRSI Ablation A0..A4)...", flush=True)
    t3 = run_track3_layer_ablation(device, smoke=args.smoke)
    cubin_audit = audit_gka_rsi_cubin_resources()
    assert cubin_audit["all_zero_spill"], f"Non-zero cubin spills or >168 regs: {cubin_audit}"
    print("[E156/v147] Running Track 4 (8-GPU NVSwitch Ulysses GKA-CP = 8)...", flush=True)
    t4 = run_track4_cp8(smoke=args.smoke)
    print("[E156/v147] Running Track 5 (Fixed-Wall-Clock Autoresearch val_bpb Loop)...", flush=True)
    t5 = run_track5_autoresearch(device, smoke=args.smoke)

    payload = {
        "experiment_id": "E156",
        "version": 147,
        "round": 30,
        "title": "Gated KalmaNet (GKA): A Fading Memory Layer Through Test-Time Ridge Regression (arXiv:2511.21016)",
        "smoke": bool(args.smoke),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "frozen_harness": harness_info,
        "rsi_cubin_audit": cubin_audit,
        "track1": t1,
        "track2": t2,
        "track3": t3,
        "track4": t4,
        "track5": t5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[E156/v147] Wrote {args.output} (all_zero_spill={cubin_audit['all_zero_spill']}, specs={cubin_audit['total_specializations']}, max_regs={cubin_audit['max_regs']})", flush=True)


if __name__ == "__main__":
    main()
