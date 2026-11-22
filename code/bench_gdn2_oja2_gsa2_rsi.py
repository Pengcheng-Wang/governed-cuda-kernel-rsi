#!/usr/bin/env python3
"""
E146 (Round-20 RSI): Benchmark & Verification Harness for Gated DeltaNet-2 (GDN-2),
Gated Oja Rule 2 (Oja-2), and Gated Slot Attention 2 (GSA-2) on 8x NVIDIA A100-SXM4-80GB.

Strict RSI Contract:
  - Evaluates via frozen code/rsi_kernel_harness.py (SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e).
  - Verifies FP64 recurrent oracle parity (GDN-2, Oja-2, and cascaded GSA-2) for forward + backward.
  - Verifies 3/3 adversarial mutation kills via verify_mutation_harness.
  - Audits SM80 cubin register & spill telemetry for upstream FLA vs 35/35 RSI kernels (0 spills, <= 168 regs).
  - Benchmarks GDN-2 and Oja-2 (M in {64, 128, 256}) Fwd / Bwd / Fwd+Bwd latency.
  - Benchmarks 5-stage progressive layer ablation (A0 -> A1 -> A2 -> A3 -> A4) for GDN2 and GSA2.
  - Benchmarks 8-GPU NVSwitch Context Parallelism (CP=8) Ulysses vs Dual-State Ring across T in {16384, 32768, 65536}.
  - Runs fixed-wall-clock val_bpb autoresearch loop (Upstream A0 vs RSI A4).
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
if "/tmp/fla_repo" not in sys.path and os.path.isdir("/tmp/fla_repo"):
    sys.path.insert(0, "/tmp/fla_repo")

import rsi_kernel_harness as harness
from gdn2_oja2_gsa2_rsi_kernels import (
    GDN2BlockRSI,
    GSA2BlockRSI,
    audit_rsi_cubin_kernels,
     chunk_gdn2_rsi,
    chunk_gated_oja_rule2_rsi,
    clear_rsi_kernel_caches,
    gdn2_recurrent_fp64_oracle,
    gsa2_cp8_ring_baseline,
    gsa2_cp8_ulysses_rsi,
    oja2_recurrent_fp64_oracle,
    _extract_compiled_kernels,
)

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def verify_frozen_harness() -> Dict[str, Any]:
    digest = harness.eval_harness_sha256()
    assert digest == EXPECTED_HARNESS_SHA256, f"Frozen harness hash mismatch: {digest} != {EXPECTED_HARNESS_SHA256}"
    return {
        "harness_path": "code/rsi_kernel_harness.py",
        "sha256": digest,
        "expected_sha256": EXPECTED_HARNESS_SHA256,
        "verified": True,
    }


def _bench_fn(fn, warmup: int, rep: int) -> Dict[str, Any]:
    groups = max(3, rep // 3)
    iters = max(2, rep // groups)
    return harness.benchmark_rotating_cuda([fn], warmup=warmup, groups=groups, iters_per_group=iters)


def _make_gdn2_inputs(B: int, T: int, H: int, K: int, V: int, dtype: torch.dtype, seed: int = 1460):
    g_gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype, generator=g_gen) * 0.5
    k = F.normalize(torch.randn(B, T, H, K, device="cuda", dtype=dtype, generator=g_gen), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype, generator=g_gen) * 0.5
    g = -F.softplus(torch.randn(B, T, H, K, device="cuda", dtype=torch.float32, generator=g_gen) - 1.5)
    b = torch.sigmoid(torch.randn(B, T, H, K, device="cuda", dtype=dtype, generator=g_gen) - 1.0)
    w = torch.sigmoid(torch.randn(B, T, H, V, device="cuda", dtype=dtype, generator=g_gen))
    h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32, generator=g_gen) * 0.2
    do = torch.randn(B, T, H, V, device="cuda", dtype=dtype, generator=g_gen) * 0.5
    dht = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32, generator=g_gen) * 0.2
    return q, k, v, g, b, w, h0, do, dht


def _make_oja2_inputs(B: int, T: int, H: int, K: int, M: int, dtype: torch.dtype, seed: int = 2460):
    g_gen = torch.Generator(device="cuda").manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, device="cuda", dtype=dtype, generator=g_gen), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, device="cuda", dtype=dtype, generator=g_gen), p=2, dim=-1)
    v = F.normalize(torch.randn(B, T, H, M, device="cuda", dtype=torch.float32, generator=g_gen), p=2, dim=-1)
    if dtype != torch.float32:
        v = v.to(dtype)
    gv = -5.0 * torch.sigmoid(torch.randn(B, T, H, M, device="cuda", dtype=torch.float32, generator=g_gen) - 1.5)
    b = torch.sigmoid(torch.randn(B, T, H, M, device="cuda", dtype=dtype, generator=g_gen) - 1.0)
    c = torch.sigmoid(torch.randn(B, T, H, K, device="cuda", dtype=dtype, generator=g_gen))
    h0 = torch.randn(B, H, K, M, device="cuda", dtype=torch.float32, generator=g_gen) * 0.2
    do = torch.randn(B, T, H, M, device="cuda", dtype=dtype, generator=g_gen) * 0.5
    dht = torch.randn(B, H, K, M, device="cuda", dtype=torch.float32, generator=g_gen) * 0.2
    return q, k, v, gv, b, c, h0, do, dht


def run_fp64_oracle_and_mutation_checks() -> Dict[str, Any]:
    # 1. GDN-2 FP32 & BF16 vs FP64 Oracle (Forward + Backward + h0/ht)
    B, T, H, K, V = 2, 128, 4, 64, 64
    q_ref, k_ref, v_ref, g_ref, b_ref, w_ref, h0_ref, do_ref, dht_ref = _make_gdn2_inputs(
        B, T, H, K, V, torch.float32, seed=1461
    )

    def _run_gdn2_oracle(dtype_run: torch.dtype, run_rsi: bool):
        q_ = q_ref.to(dtype_run).detach().requires_grad_(True)
        k_ = k_ref.to(dtype_run).detach().requires_grad_(True)
        v_ = v_ref.to(dtype_run).detach().requires_grad_(True)
        g_ = g_ref.detach().requires_grad_(True)
        b_ = b_ref.to(dtype_run).detach().requires_grad_(True)
        w_ = w_ref.to(dtype_run).detach().requires_grad_(True)
        h0_ = h0_ref.detach().requires_grad_(True)
        if run_rsi:
            o, ht = chunk_gdn2_rsi(
                q_, k_, v_, g_, b_, w_, initial_state=h0_, output_final_state=True, use_qk_l2norm_in_kernel=True
            )
        else:
            o, ht = gdn2_recurrent_fp64_oracle(
                q_, k_, v_, g_, b_, w_, initial_state=h0_, use_qk_l2norm=True
            )
        loss = (o.float() * do_ref.float()).sum() + (ht.float() * dht_ref.float()).sum()
        loss.backward()
        return {
            "o": o.detach().float(),
            "ht": ht.detach().float(),
            "dq": q_.grad.detach().float(),
            "dk": k_.grad.detach().float(),
            "dv": v_.grad.detach().float(),
            "dg": g_.grad.detach().float(),
            "db": b_.grad.detach().float(),
            "dw": w_.grad.detach().float(),
            "dh0": h0_.grad.detach().float(),
        }

    gdn2_fp64 = _run_gdn2_oracle(torch.float32, run_rsi=False)
    gdn2_fp32 = _run_gdn2_oracle(torch.float32, run_rsi=True)
    gdn2_bf16 = _run_gdn2_oracle(torch.bfloat16, run_rsi=True)

    gdn2_fp32_diffs = {
        k: float((gdn2_fp32[k] - gdn2_fp64[k]).abs().max().item()) for k in gdn2_fp64
    }
    gdn2_bf16_diffs = {
        k: float((gdn2_bf16[k] - gdn2_fp64[k]).abs().max().item()) for k in gdn2_fp64
    }
    assert max(gdn2_fp32_diffs.values()) < 5e-4, f"GDN2 FP32 diff too large: {gdn2_fp32_diffs}"
    assert max(gdn2_bf16_diffs.values()) < 5e-2, f"GDN2 BF16 diff too large: {gdn2_bf16_diffs}"

    # 2. Oja-2 FP32 & BF16 vs FP64 Oracle across M in {64, 128, 256}
    oja2_parity = {}
    for M in (64, 128, 256):
        q_o, k_o, v_o, gv_o, b_o, c_o, h0_o, do_o, dht_o = _make_oja2_inputs(
            2, 128, 4, 64, M, torch.float32, seed=2460 + M
        )

        def _run_oja2(dtype_run: torch.dtype, run_rsi: bool):
            q_ = q_o.to(dtype_run).detach().requires_grad_(True)
            k_ = k_o.to(dtype_run).detach().requires_grad_(True)
            v_ = v_o.to(dtype_run if dtype_run != torch.bfloat16 else torch.float32).detach().requires_grad_(True)
            gv_ = gv_o.detach().requires_grad_(True)
            b_ = b_o.to(dtype_run).detach().requires_grad_(True)
            c_ = c_o.to(dtype_run).detach().requires_grad_(True)
            h0_ = h0_o.detach().requires_grad_(True)
            if run_rsi:
                o, ht = chunk_gated_oja_rule2_rsi(
                    q_, k_, v_, gv_, b_, c_, initial_state=h0_, output_final_state=True,
                    use_q_l2norm=True, use_k_l2norm=True,
                )
            else:
                o, ht = oja2_recurrent_fp64_oracle(
                    q_, k_, v_, gv_, b_, c_, initial_state=h0_,
                    use_q_l2norm=True, use_k_l2norm=True,
                )
            loss = (o.float() * do_o.float()).sum() + (ht.float() * dht_o.float()).sum()
            loss.backward()
            return {
                "o": o.detach().float(),
                "ht": ht.detach().float(),
                "dq": q_.grad.detach().float(),
                "dk": k_.grad.detach().float(),
                "dv": v_.grad.detach().float(),
                "dgv": gv_.grad.detach().float(),
                "db": b_.grad.detach().float(),
                "dc": c_.grad.detach().float(),
                "dh0": h0_.grad.detach().float(),
            }

        oja_fp64 = _run_oja2(torch.float32, run_rsi=False)
        oja_fp32 = _run_oja2(torch.float32, run_rsi=True)
        oja_bf16 = _run_oja2(torch.bfloat16, run_rsi=True)

        fp32_d = {k: float((oja_fp32[k] - oja_fp64[k]).abs().max().item()) for k in oja_fp64}
        bf16_d = {k: float((oja_bf16[k] - oja_fp64[k]).abs().max().item()) for k in oja_fp64}
        assert max(fp32_d.values()) < 5e-4, f"Oja2 M={M} FP32 diff too large: {fp32_d}"
        assert max(bf16_d.values()) < 5e-2, f"Oja2 M={M} BF16 diff too large: {bf16_d}"
        oja2_parity[f"M_{M}"] = {"fp32_vs_fp64": fp32_d, "bf16_vs_fp64": bf16_d}

    # 3. Cascaded GSA-2 Layer Parity (A0 Upstream vs A4 RSI)
    from fla.layers.gdn2 import GatedDeltaNet2
    from fla.layers.gsa2 import GatedSlotAttention2

    torch.manual_seed(3460)
    gdn2_base = GatedDeltaNet2(hidden_size=256, num_heads=4, head_dim=64).cuda().bfloat16()
    gdn2_a0 = GDN2BlockRSI(copy.deepcopy(gdn2_base), stage="A0")
    gdn2_a4 = GDN2BlockRSI(copy.deepcopy(gdn2_base), stage="A4")
    x_in = torch.randn(2, 256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_in2 = x_in.detach().clone().requires_grad_(True)
    dy_in = torch.randn_like(x_in)
    y_a0 = gdn2_a0(x_in)
    (y_a0 * dy_in).sum().backward()
    y_a4 = gdn2_a4(x_in2)
    (y_a4 * dy_in).sum().backward()
    gdn2_layer_fwd_diff = float((y_a0.float() - y_a4.float()).abs().max().item())
    gdn2_layer_bwd_diff = float((x_in.grad.float() - x_in2.grad.float()).abs().max().item())
    assert gdn2_layer_fwd_diff < 2e-2 and gdn2_layer_bwd_diff < 5e-2

    gsa2_base = GatedSlotAttention2(hidden_size=256, num_heads=4, head_dim=64, num_slots=128).cuda().bfloat16()
    gsa2_a0 = GSA2BlockRSI(copy.deepcopy(gsa2_base), stage="A0")
    gsa2_a4 = GSA2BlockRSI(copy.deepcopy(gsa2_base), stage="A4")
    x_g1 = torch.randn(2, 256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_g2 = x_g1.detach().clone().requires_grad_(True)
    dy_g = torch.randn_like(x_g1)
    yg_a0 = gsa2_a0(x_g1)
    (yg_a0 * dy_g).sum().backward()
    yg_a4 = gsa2_a4(x_g2)
    (yg_a4 * dy_g).sum().backward()
    gsa2_layer_fwd_diff = float((yg_a0.float() - yg_a4.float()).abs().max().item())
    gsa2_layer_bwd_diff = float((x_g1.grad.float() - x_g2.grad.float()).abs().max().item())
    assert gsa2_layer_fwd_diff < 3e-2 and gsa2_layer_bwd_diff < 6e-2

    # 4. 3/3 Adversarial Mutation Harness Verification via harness.compare_tensor_to_fp64
    o_ref_fp64 = gdn2_fp64["o"]
    cand_cmp = harness.compare_tensor_to_fp64(
        gdn2_bf16["o"], o_ref_fp64, rel_tol=1.5e-2, name="candidate_unmutated"
    )
    assert cand_cmp["ok"], f"Unmutated candidate failed compare_tensor_to_fp64: {cand_cmp}"

    mut_Args = (q_ref.bfloat16(), k_ref.bfloat16(), v_ref.bfloat16(), g_ref, b_ref.bfloat16(), w_ref.bfloat16())
    m1_o, _ = chunk_gdn2_rsi(mut_Args[0], mut_Args[1], mut_Args[2], torch.zeros_like(g_ref), mut_Args[4], mut_Args[5], initial_state=h0_ref, use_qk_l2norm_in_kernel=True)
    m2_o, _ = chunk_gdn2_rsi(mut_Args[0], mut_Args[1], mut_Args[2], mut_Args[3], torch.zeros_like(mut_Args[4]), mut_Args[5], initial_state=h0_ref, use_qk_l2norm_in_kernel=True)
    m3_o, _ = chunk_gdn2_rsi(mut_Args[0], mut_Args[1], mut_Args[2], mut_Args[3], mut_Args[4], torch.ones_like(mut_Args[5]), initial_state=h0_ref, use_qk_l2norm_in_kernel=True)

    m1_cmp = harness.compare_tensor_to_fp64(m1_o, o_ref_fp64, rel_tol=1.5e-2, name="MUT1_DROP_DECAY_GATE_G")
    m2_cmp = harness.compare_tensor_to_fp64(m2_o, o_ref_fp64, rel_tol=1.5e-2, name="MUT2_DROP_ERASURE_GATE_B")
    m3_cmp = harness.compare_tensor_to_fp64(m3_o, o_ref_fp64, rel_tol=1.5e-2, name="MUT3_DROP_WRITE_GATE_W")

    mut_res = {
        "candidate_unmutated": cand_cmp,
        "mutations": [m1_cmp, m2_cmp, m3_cmp],
        "killed_count": int(sum(1 for m in (m1_cmp, m2_cmp, m3_cmp) if not m["ok"])),
        "total_mutations": 3,
        "all_mutations_killed": bool((not m1_cmp["ok"]) and (not m2_cmp["ok"]) and (not m3_cmp["ok"])),
    }
    assert mut_res["all_mutations_killed"], f"Mutation harness failed: {mut_res}"

    return {
        "gdn2_fp32_vs_fp64_max_abs_diff": gdn2_fp32_diffs,
        "gdn2_bf16_vs_fp64_max_abs_diff": gdn2_bf16_diffs,
        "oja2_vs_fp64": oja2_parity,
        "gdn2_layer_a0_vs_a4": {
            "fwd_max_abs_diff": gdn2_layer_fwd_diff,
            "bwd_dx_max_abs_diff": gdn2_layer_bwd_diff,
        },
        "gsa2_layer_a0_vs_a4": {
            "fwd_max_abs_diff": gsa2_layer_fwd_diff,
            "bwd_dx_max_abs_diff": gsa2_layer_bwd_diff,
        },
        "mutation_harness": mut_res,
    }


def audit_upstream_fla_cubin() -> Dict[str, Any]:
    report = {}
    total_spilling_specs = 0
    max_regs_all = 0
    max_spills_all = 0
    seen_objs = set()
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not (
            mod_name.startswith("fla.ops.gdn2") or mod_name.startswith("fla.ops.gated_oja_rule")
        ):
            continue
        short_pkg = "gdn2" if "gdn2" in mod_name else ("oja2" if "gated_oja_rule2" in mod_name else "oja")
        for attr_name, obj in vars(mod).items():
            if id(obj) in seen_objs:
                continue
            compiled = _extract_compiled_kernels(obj)
            if not compiled:
                continue
            seen_objs.add(id(obj))
            key = f"{short_pkg}.{attr_name}"
            spilling = sum(1 for c in compiled if getattr(c, "n_spills", 0) > 0)
            m_regs = max((getattr(c, "n_regs", 0) for c in compiled), default=0)
            m_spills = max((getattr(c, "n_spills", 0) for c in compiled), default=0)
            m_smem = max((getattr(c.metadata, "shared", 0) for c in compiled if hasattr(c, "metadata")), default=0)
            report[key] = {
                "specializations": len(compiled),
                "spilling_specializations": int(spilling),
                "max_regs": int(m_regs),
                "max_spills": int(m_spills),
                "max_shared_bytes": int(m_smem),
            }
            total_spilling_specs += int(spilling)
            max_regs_all = max(max_regs_all, int(m_regs))
            max_spills_all = max(max_spills_all, int(m_spills))
    return {
        "total_spilling_specializations": int(total_spilling_specs),
        "max_regs": int(max_regs_all),
        "max_spills": int(max_spills_all),
        "per_kernel": report,
    }


def benchmark_gdn2_and_oja2_kernels(warmup: int, rep: int, shapes_mode: str) -> Dict[str, Any]:
    from fla.ops.gdn2 import chunk_gdn2
    from fla.ops.gated_oja_rule2 import chunk_gated_oja_rule2

    if shapes_mode == "smoke":
        gdn2_shapes = [(1, 4096, 16, 128, 128), (2, 4096, 16, 128, 128)]
        oja2_shapes = [(2, 4096, 16, 128, 64), (2, 4096, 16, 128, 128), (2, 4096, 16, 128, 256)]
    else:
        gdn2_shapes = [
            (1, 4096, 16, 128, 128),
            (2, 4096, 16, 128, 128),
            (1, 8192, 16, 128, 128),
        ]
        oja2_shapes = [
            (2, 4096, 16, 128, 64),
            (2, 4096, 16, 128, 128),
            (2, 4096, 16, 128, 256),
            (1, 8192, 16, 128, 128),
            (1, 8192, 16, 128, 256),
        ]

    gdn2_results = []
    for (B, T, H, K, V) in gdn2_shapes:
        q, k, v, g, b, w, _, do, _ = _make_gdn2_inputs(B, T, H, K, V, torch.bfloat16, seed=4000 + B * 100 + T)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        g.requires_grad_(True)
        b.requires_grad_(True)
        w.requires_grad_(True)

        # Warm and benchmark Upstream (safe_gate=False)
        o_up0, _ = chunk_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, safe_gate=False)
        o_up0.backward(do, retain_graph=True)

        def _up0_fwd():
            chunk_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, safe_gate=False)

        def _up0_bwd():
            o_up0.backward(do, retain_graph=True)

        def _up0_fwdbwd():
            q.grad = k.grad = v.grad = g.grad = b.grad = w.grad = None
            out, _ = chunk_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, safe_gate=False)
            out.backward(do)

        # Warm and benchmark Upstream (safe_gate=True)
        o_up1, _ = chunk_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, safe_gate=True)
        o_up1.backward(do, retain_graph=True)

        def _up1_fwd():
            chunk_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, safe_gate=True)

        def _up1_bwd():
            o_up1.backward(do, retain_graph=True)

        def _up1_fwdbwd():
            q.grad = k.grad = v.grad = g.grad = b.grad = w.grad = None
            out, _ = chunk_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, safe_gate=True)
            out.backward(do)

        # Warm and benchmark RSI
        o_rsi, _ = chunk_gdn2_rsi(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True)
        o_rsi.backward(do, retain_graph=True)

        def _rsi_fwd():
            chunk_gdn2_rsi(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True)

        def _rsi_bwd():
            o_rsi.backward(do, retain_graph=True)

        def _rsi_fwdbwd():
            q.grad = k.grad = v.grad = g.grad = b.grad = w.grad = None
            out, _ = chunk_gdn2_rsi(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True)
            out.backward(do)

        t_up0_fwd = _bench_fn(_up0_fwd, warmup=warmup, rep=rep)["median_ms"]
        t_up0_bwd = _bench_fn(_up0_bwd, warmup=warmup, rep=rep)["median_ms"]
        t_up0_tot = _bench_fn(_up0_fwdbwd, warmup=warmup, rep=rep)["median_ms"]

        t_up1_fwd = _bench_fn(_up1_fwd, warmup=warmup, rep=rep)["median_ms"]
        t_up1_bwd = _bench_fn(_up1_bwd, warmup=warmup, rep=rep)["median_ms"]
        t_up1_tot = _bench_fn(_up1_fwdbwd, warmup=warmup, rep=rep)["median_ms"]

        t_rsi_fwd = _bench_fn(_rsi_fwd, warmup=warmup, rep=rep)["median_ms"]
        t_rsi_bwd = _bench_fn(_rsi_bwd, warmup=warmup, rep=rep)["median_ms"]
        t_rsi_tot = _bench_fn(_rsi_fwdbwd, warmup=warmup, rep=rep)["median_ms"]

        gdn2_results.append({
            "shape": f"B={B},T={T},H={H},K={K},V={V}",
            "B": B, "T": T, "H": H, "K": K, "V": V,
            "upstream_safe_gate_false": {
                "fwd_ms": round(t_up0_fwd, 4),
                "bwd_ms": round(t_up0_bwd, 4),
                "fwdbwd_ms": round(t_up0_tot, 4),
            },
            "upstream_safe_gate_true": {
                "fwd_ms": round(t_up1_fwd, 4),
                "bwd_ms": round(t_up1_bwd, 4),
                "fwdbwd_ms": round(t_up1_tot, 4),
            },
            "rsi_zero_spill": {
                "fwd_ms": round(t_rsi_fwd, 4),
                "bwd_ms": round(t_rsi_bwd, 4),
                "fwdbwd_ms": round(t_rsi_tot, 4),
            },
            "speedup_vs_safe_gate_false": {
                "fwd_x": round(t_up0_fwd / t_rsi_fwd, 3),
                "bwd_x": round(t_up0_bwd / t_rsi_bwd, 3),
                "fwdbwd_x": round(t_up0_tot / t_rsi_tot, 3),
            },
            "speedup_vs_safe_gate_true": {
                "fwd_x": round(t_up1_fwd / t_rsi_fwd, 3),
                "bwd_x": round(t_up1_bwd / t_rsi_bwd, 3),
                "fwdbwd_x": round(t_up1_tot / t_rsi_tot, 3),
            },
        })

    oja2_results = []
    for (B, T, H, K, M) in oja2_shapes:
        q, k, v, gv, b, c, _, do, _ = _make_oja2_inputs(B, T, H, K, M, torch.bfloat16, seed=5000 + B * 100 + M)
        v_fp32 = v.float().detach().requires_grad_(True)
        q.requires_grad_(True)
        k.requires_grad_(True)
        gv.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)

        o_up, _ = chunk_gated_oja_rule2(q, k, v_fp32, gv, b, c, use_q_l2norm=True, use_k_l2norm=True)
        o_up.backward(do, retain_graph=True)

        def _oja_up_fwd():
            chunk_gated_oja_rule2(q, k, v_fp32, gv, b, c, use_q_l2norm=True, use_k_l2norm=True)

        def _oja_up_bwd():
            o_up.backward(do, retain_graph=True)

        def _oja_up_fwdbwd():
            q.grad = k.grad = v_fp32.grad = gv.grad = b.grad = c.grad = None
            out, _ = chunk_gated_oja_rule2(q, k, v_fp32, gv, b, c, use_q_l2norm=True, use_k_l2norm=True)
            out.backward(do)

        o_rsi, _ = chunk_gated_oja_rule2_rsi(q, k, v_fp32, gv, b, c, use_q_l2norm=True, use_k_l2norm=True)
        o_rsi.backward(do, retain_graph=True)

        def _oja_rsi_fwd():
            chunk_gated_oja_rule2_rsi(q, k, v_fp32, gv, b, c, use_q_l2norm=True, use_k_l2norm=True)

        def _oja_rsi_bwd():
            o_rsi.backward(do, retain_graph=True)

        def _oja_rsi_fwdbwd():
            q.grad = k.grad = v_fp32.grad = gv.grad = b.grad = c.grad = None
            out, _ = chunk_gated_oja_rule2_rsi(q, k, v_fp32, gv, b, c, use_q_l2norm=True, use_k_l2norm=True)
            out.backward(do)

        t_up_fwd = _bench_fn(_oja_up_fwd, warmup=warmup, rep=rep)["median_ms"]
        t_up_bwd = _bench_fn(_oja_up_bwd, warmup=warmup, rep=rep)["median_ms"]
        t_up_tot = _bench_fn(_oja_up_fwdbwd, warmup=warmup, rep=rep)["median_ms"]

        t_rsi_fwd = _bench_fn(_oja_rsi_fwd, warmup=warmup, rep=rep)["median_ms"]
        t_rsi_bwd = _bench_fn(_oja_rsi_bwd, warmup=warmup, rep=rep)["median_ms"]
        t_rsi_tot = _bench_fn(_oja_rsi_fwdbwd, warmup=warmup, rep=rep)["median_ms"]

        oja2_results.append({
            "shape": f"B={B},T={T},H={H},K={K},M={M}",
            "B": B, "T": T, "H": H, "K": K, "M": M,
            "upstream": {
                "fwd_ms": round(t_up_fwd, 4),
                "bwd_ms": round(t_up_bwd, 4),
                "fwdbwd_ms": round(t_up_tot, 4),
            },
            "rsi_zero_spill": {
                "fwd_ms": round(t_rsi_fwd, 4),
                "bwd_ms": round(t_rsi_bwd, 4),
                "fwdbwd_ms": round(t_rsi_tot, 4),
            },
            "speedup": {
                "fwd_x": round(t_up_fwd / t_rsi_fwd, 3),
                "bwd_x": round(t_up_bwd / t_rsi_bwd, 3),
                "fwdbwd_x": round(t_up_tot / t_rsi_tot, 3),
            },
        })

    return {"gdn2_kernels": gdn2_results, "oja2_kernels": oja2_results}


def benchmark_5stage_layer_ablation(warmup: int, rep: int, shapes_mode: str) -> Dict[str, Any]:
    from fla.layers.gdn2 import GatedDeltaNet2
    from fla.layers.gsa2 import GatedSlotAttention2

    B, T, D, H, K = (2, 2048, 2048, 16, 128) if shapes_mode == "smoke" else (2, 4096, 2048, 16, 128)
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)

    # 1. GDN2 5-Stage Ablation
    torch.manual_seed(6100)
    gdn2_base = GatedDeltaNet2(hidden_size=D, num_heads=H, head_dim=K).cuda().bfloat16()
    gdn2_stages = {}
    for stage in ("A0", "A1", "A2", "A3", "A4"):
        mod = GDN2BlockRSI(gdn2_base, stage=stage)
        out = mod(x)
        out.backward(dy)

        def _step_gdn2():
            x.grad = None
            for p in mod.parameters():
                p.grad = None
            y = mod(x)
            y.backward(dy)

        res = _bench_fn(_step_gdn2, warmup=warmup, rep=rep)
        gdn2_stages[stage] = round(res["median_ms"], 4)

    gdn2_speedup = round(gdn2_stages["A0"] / gdn2_stages["A4"], 3)

    # 2. GSA2 5-Stage Ablation across M in {64, 128, 256}
    gsa2_by_slots = {}
    slot_list = (128, 256) if shapes_mode == "smoke" else (64, 128, 256)
    for M in slot_list:
        torch.manual_seed(6200 + M)
        gsa2_base = GatedSlotAttention2(hidden_size=D, num_heads=H, head_dim=K, num_slots=M).cuda().bfloat16()
        stage_ms = {}
        for stage in ("A0", "A1", "A2", "A3", "A4"):
            mod = GSA2BlockRSI(gsa2_base, stage=stage)
            out = mod(x)
            out.backward(dy)

            def _step_gsa2():
                x.grad = None
                for p in mod.parameters():
                    p.grad = None
                y = mod(x)
                y.backward(dy)

            res = _bench_fn(_step_gsa2, warmup=warmup, rep=rep)
            stage_ms[stage] = round(res["median_ms"], 4)
        stage_ms["speedup_A4_vs_A0_x"] = round(stage_ms["A0"] / stage_ms["A4"], 3)
        stage_ms["speedup_A4_vs_A1_x"] = round(stage_ms["A1"] / stage_ms["A4"], 3)
        gsa2_by_slots[f"M_{M}"] = stage_ms

    return {
        "shape": f"B={B},T={T},D={D},H={H},K={K}",
        "gdn2_stages_ms": gdn2_stages,
        "gdn2_speedup_A4_vs_A0_x": gdn2_speedup,
        "gsa2_by_slots": gsa2_by_slots,
    }


def _cp8_worker(rank: int, world_size: int, port: int, mode: str, out_file: str):
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_e146_r{rank}"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    B, H, K, V = 1, 16, 128, 128
    # 1. Single-GPU equivalence verification at T_total=2048, M=128
    T_verify = 2048
    T_loc_v = T_verify // world_size
    M_v = 128
    q_full, k_full, v_full, gv_full, b1_full, c1_full, _, do_full, _ = _make_oja2_inputs(
        B, T_verify, H, K, M_v, torch.bfloat16, seed=7100
    )
    _, _, v2_full, g2_full, b2_full, w2_full, _, do2_full, _ = _make_gdn2_inputs(
        B, T_verify, H, M_v, V, torch.bfloat16, seed=7200
    )
    w_slot_full = v_full.float()

    # Single-GPU reference on rank 0
    if rank == 0:
        q_s = q_full.detach().clone().requires_grad_(True)
        k_s = k_full.detach().clone().requires_grad_(True)
        v_s = v2_full.detach().clone().requires_grad_(True)
        ws_s = w_slot_full.detach().clone().requires_grad_(True)
        gv_s = gv_full.detach().clone().requires_grad_(True)
        b1_s = b1_full.detach().clone().requires_grad_(True)
        c1_s = c1_full.detach().clone().requires_grad_(True)
        g2_s = g2_full.detach().clone().requires_grad_(True)
        b2_s = b2_full.detach().clone().requires_grad_(True)
        w2_s = w2_full.detach().clone().requires_grad_(True)

        o1_s, _ = chunk_gated_oja_rule2_rsi(
            q=q_s, k=k_s, v=ws_s, gv=gv_s, b=b1_s, c=c1_s, use_q_l2norm=True, use_k_l2norm=True
        )
        q2_s = F.silu(o1_s)
        k2_s = ws_s.to(q2_s.dtype)
        o2_s, _ = chunk_gdn2_rsi(
            q=q2_s, k=k2_s, v=v_s, g=g2_s, b=b2_s, w=w2_s, use_qk_l2norm_in_kernel=True
        )
        (o2_s * do2_full).sum().backward()

    sl = slice(rank * T_loc_v, (rank + 1) * T_loc_v)
    do_loc = do2_full[:, sl].contiguous()

    def _run_dist_verify(fn_dist):
        q_l = q_full[:, sl].detach().clone().requires_grad_(True)
        k_l = k_full[:, sl].detach().clone().requires_grad_(True)
        v_l = v2_full[:, sl].detach().clone().requires_grad_(True)
        ws_l = w_slot_full[:, sl].detach().clone().requires_grad_(True)
        gv_l = gv_full[:, sl].detach().clone().requires_grad_(True)
        b1_l = b1_full[:, sl].detach().clone().requires_grad_(True)
        c1_l = c1_full[:, sl].detach().clone().requires_grad_(True)
        g2_l = g2_full[:, sl].detach().clone().requires_grad_(True)
        b2_l = b2_full[:, sl].detach().clone().requires_grad_(True)
        w2_l = w2_full[:, sl].detach().clone().requires_grad_(True)

        o_l = fn_dist(q_l, k_l, v_l, ws_l, gv_l, b1_l, c1_l, g2_l, b2_l, w2_l, group)
        (o_l * do_loc).sum().backward()

        o_gat = [torch.empty_like(o_l) for _ in range(world_size)]
        dq_gat = [torch.empty_like(q_l.grad) for _ in range(world_size)]
        dv_gat = [torch.empty_like(v_l.grad) for _ in range(world_size)]
        dist.all_gather(o_gat, o_l.detach(), group=group)
        dist.all_gather(dq_gat, q_l.grad.detach(), group=group)
        dist.all_gather(dv_gat, v_l.grad.detach(), group=group)
        return torch.cat(o_gat, dim=1), torch.cat(dq_gat, dim=1), torch.cat(dv_gat, dim=1)

    o_uly, dq_uly, dv_uly = _run_dist_verify(gsa2_cp8_ulysses_rsi)
    o_rng, dq_rng, dv_rng = _run_dist_verify(gsa2_cp8_ring_baseline)

    parity_summary = {}
    if rank == 0:
        parity_summary = {
            "ulysses_vs_single_gpu_fwd_max_diff": float((o_uly.float() - o2_s.detach().float()).abs().max().item()),
            "ulysses_vs_single_gpu_dq_max_diff": float((dq_uly.float() - q_s.grad.detach().float()).abs().max().item()),
            "ulysses_vs_single_gpu_dv_max_diff": float((dv_uly.float() - v_s.grad.detach().float()).abs().max().item()),
            "ring_vs_single_gpu_fwd_max_diff": float((o_rng.float() - o2_s.detach().float()).abs().max().item()),
            "ring_vs_single_gpu_dq_max_diff": float((dq_rng.float() - q_s.grad.detach().float()).abs().max().item()),
        }
        assert parity_summary["ulysses_vs_single_gpu_fwd_max_diff"] < 1e-4
        assert parity_summary["ulysses_vs_single_gpu_dq_max_diff"] < 1e-4
        assert parity_summary["ring_vs_single_gpu_fwd_max_diff"] < 2e-2

    # 2. Benchmark CP=8 Ulysses vs Dual-State Ring across sequence lengths and slot counts
    configs = (
        [(16384, 128), (32768, 128), (32768, 256)]
        if mode == "smoke"
        else [(16384, 128), (32768, 64), (32768, 128), (32768, 256), (65536, 128)]
    )
    warmup = 3 if mode == "smoke" else 6
    rep = 8 if mode == "smoke" else 15

    bench_rows = []
    for (T_total, M) in configs:
        T_local = T_total // world_size
        q_l, k_l, v_s_l, gv_l, b1_l, c1_l, _, _, _ = _make_oja2_inputs(
            B, T_local, H, K, M, torch.bfloat16, seed=8000 + rank * 17 + M
        )
        _, _, v_l, g2_l, b2_l, w2_l, _, do_l, _ = _make_gdn2_inputs(
            B, T_local, H, M, V, torch.bfloat16, seed=8500 + rank * 17 + M
        )
        ws_l = v_s_l.float().detach().requires_grad_(True)
        for t in (q_l, k_l, v_l, gv_l, b1_l, c1_l, g2_l, b2_l, w2_l):
            t.requires_grad_(True)

        def _measure(fn_cp):
            for _ in range(warmup):
                for t in (q_l, k_l, v_l, ws_l, gv_l, b1_l, c1_l, g2_l, b2_l, w2_l):
                    t.grad = None
                out = fn_cp(q_l, k_l, v_l, ws_l, gv_l, b1_l, c1_l, g2_l, b2_l, w2_l, group)
                out.backward(do_l)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            times = []
            for _ in range(rep):
                for t in (q_l, k_l, v_l, ws_l, gv_l, b1_l, c1_l, g2_l, b2_l, w2_l):
                    t.grad = None
                t0 = time.perf_counter()
                out = fn_cp(q_l, k_l, v_l, ws_l, gv_l, b1_l, c1_l, g2_l, b2_l, w2_l, group)
                out.backward(do_l)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            return times[len(times) // 2]

        uly_ms = _measure(gsa2_cp8_ulysses_rsi)
        ring_ms = _measure(gsa2_cp8_ring_baseline)
        if rank == 0:
            bench_rows.append({
                "T_total": T_total,
                "T_local": T_local,
                "M_slots": M,
                "ring_cp8_fwdbwd_ms": round(ring_ms, 4),
                "ulysses_cp8_fwdbwd_ms": round(uly_ms, 4),
                "speedup_ulysses_vs_ring_x": round(ring_ms / uly_ms, 3),
            })

    if rank == 0:
        Path(out_file).write_text(
            json.dumps({"parity": parity_summary, "benchmarks": bench_rows}, indent=2),
            encoding="utf-8",
        )
    dist.destroy_process_group()


def run_cp8_benchmarks(mode: str) -> Dict[str, Any]:
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 8, f"Expected 8 GPUs for CP=8 benchmark, found {n_gpus}"
    out_file = f"/tmp/e146_cp8_{os.getpid()}.json"
    port = 29640 + (os.getpid() % 200)
    mp.spawn(_cp8_worker, args=(8, port, mode, out_file), nprocs=8, join=True)
    res = json.loads(Path(out_file).read_text(encoding="utf-8"))
    try:
        os.remove(out_file)
    except OSError:
        pass
    return res


class _GSA2AutoresearchLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 256, n_layers: int = 4, n_heads: int = 4, head_dim: int = 64, num_slots: int = 128, stage: str = "A4"):
        super().__init__()
        from fla.layers.gsa2 import GatedSlotAttention2
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.layers = nn.ModuleList([
            GSA2BlockRSI(
                GatedSlotAttention2(
                    hidden_size=d_model,
                    num_heads=n_heads,
                    head_dim=head_dim,
                    num_slots=num_slots,
                ),
                stage=stage,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for norm, layer in zip(self.norms, self.layers):
            h = h + layer(norm(h))
        return self.lm_head(self.final_norm(h))


def run_val_bpb_autoresearch_loop(budget_sec: float) -> Dict[str, Any]:
    B_tr, T_tr = 4, 512

    def _sample_batch(gen: torch.Generator) -> torch.Tensor:
        seq = torch.randint(0, 256, (B_tr, T_tr + 1), device="cuda", generator=gen)
        # Multi-scale associative recall + delayed copy rules over long horizons
        seq[:, 16:128] = (seq[:, :112] * 5 + 11) % 256
        seq[:, 128:320] = (seq[:, 64:256] * 3 + 19) % 256
        seq[:, 320:] = (seq[:, 128 : 128 + (T_tr + 1 - 320)] ^ 0x2A) % 256
        return seq

    g_val = torch.Generator(device="cuda").manual_seed(2026146)
    val_seq = torch.stack([_sample_batch(g_val) for _ in range(16)], dim=0)

    torch.manual_seed(9146)
    init_model = _GSA2AutoresearchLM(stage="A0").cuda().bfloat16()
    init_state = copy.deepcopy(init_model.state_dict())

    @torch.no_grad()
    def _eval_bpb(model: nn.Module) -> float:
        model.eval()
        total_nats = 0.0
        total_tokens = 0
        for i in range(val_seq.shape[0]):
            xb = val_seq[i, :, :-1]
            yb = val_seq[i, :, 1:]
            logits = model(xb)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), yb.reshape(-1), reduction="sum")
            total_nats += float(loss.item())
            total_tokens += yb.numel()
        model.train()
        return (total_nats / total_tokens) / math.log(2.0)

    init_bpb = _eval_bpb(init_model)
    arms = {}
    for stage in ("A0", "A4"):
        model = _GSA2AutoresearchLM(stage=stage).cuda().bfloat16()
        model.load_state_dict(copy.deepcopy(init_state))
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)

        # Warmup 2 steps so JIT compilation does not distort steady-state wall-clock training
        g_warm = torch.Generator(device="cuda").manual_seed(2026999)
        for _ in range(2):
            batch_w = _sample_batch(g_warm)
            xb = batch_w[:, :-1]
            yb = batch_w[:, 1:]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            opt.step()
        torch.cuda.synchronize()

        # Reload exact initial weights and identical training stream generator for fair fixed-wall-clock comparison
        model.load_state_dict(copy.deepcopy(init_state))
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
        g_train = torch.Generator(device="cuda").manual_seed(20261460)
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        steps = 0
        while (time.perf_counter() - t_start) < budget_sec:
            batch = _sample_batch(g_train)
            xb = batch[:, :-1]
            yb = batch[:, 1:]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            if steps % 4 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        final_bpb = _eval_bpb(model)
        arms[stage] = {
            "wall_clock_sec": round(elapsed, 3),
            "optimizer_steps": int(steps),
            "steps_per_sec": round(steps / elapsed, 3),
            "tokens_per_sec": round((steps * B_tr * T_tr) / elapsed, 1),
            "initial_val_bpb": round(init_bpb, 4),
            "final_val_bpb": round(final_bpb, 4),
            "val_bpb_improvement": round(init_bpb - final_bpb, 4),
        }

    return {
        "budget_sec": budget_sec,
        "A0_upstream": arms["A0"],
        "A4_rsi": arms["A4"],
        "step_rate_speedup_x": round(arms["A4"]["steps_per_sec"] / max(arms["A0"]["steps_per_sec"], 1e-6), 3),
        "extra_val_bpb_gain": round(arms["A0"]["final_val_bpb"] - arms["A4"]["final_val_bpb"], 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    t0_global = time.perf_counter()
    harness_info = verify_frozen_harness()
    oracle_and_mutations = run_fp64_oracle_and_mutation_checks()
    # Clear FP32 ALLOW_TF32=False (input_precision="ieee") oracle-test specializations before BF16 benchmark & cubin audit
    clear_rsi_kernel_caches()

    warmup = 4 if args.mode == "smoke" else 10
    rep = 10 if args.mode == "smoke" else 25
    kernel_benchmarks = benchmark_gdn2_and_oja2_kernels(warmup=warmup, rep=rep, shapes_mode=args.mode)
    layer_ablation = benchmark_5stage_layer_ablation(warmup=warmup, rep=rep, shapes_mode=args.mode)

    upstream_cubin = audit_upstream_fla_cubin()
    rsi_cubin = audit_rsi_cubin_kernels()
    assert rsi_cubin["zero_spill_verified"], f"RSI cubin audit failed zero-spill gate: {rsi_cubin}"
    assert rsi_cubin["compiled_kernels"] == rsi_cubin["total_kernels"], (
        f"Expected all {rsi_cubin['total_kernels']} RSI kernels compiled, got {rsi_cubin['compiled_kernels']}"
    )

    cp8_results = run_cp8_benchmarks(mode=args.mode)
    budget_sec = 8.0 if args.mode == "smoke" else 30.0
    autoresearch = run_val_bpb_autoresearch_loop(budget_sec=budget_sec)

    payload = {
        "experiment_id": "E146",
        "round": 20,
        "mode": args.mode,
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "torch_version": torch.__version__,
        },
        "frozen_harness": harness_info,
        "oracle_and_mutations": oracle_and_mutations,
        "cubin_audit": {
            "upstream_fla": upstream_cubin,
            "rsi_zero_spill": rsi_cubin,
        },
        "kernel_benchmarks": kernel_benchmarks,
        "layer_5stage_ablation": layer_ablation,
        "cp8_benchmarks": cp8_results,
        "autoresearch_val_bpb": autoresearch,
        "elapsed_sec": round(time.perf_counter() - t0_global, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[E146 {args.mode}] Complete in {payload['elapsed_sec']}s -> {out_path}")
    print(
        f"  Cubin Audit: {rsi_cubin['compiled_kernels']}/{rsi_cubin['total_kernels']} kernels, "
        f"{rsi_cubin['total_specializations']} specs, {rsi_cubin['spilling_specializations']} spills, "
        f"max_regs={rsi_cubin['max_regs']} (Upstream FLA: {upstream_cubin['total_spilling_specializations']} spilling specs, "
        f"max_regs={upstream_cubin['max_regs']}, max_spills={upstream_cubin['max_spills']})"
    )


if __name__ == "__main__":
    main()
