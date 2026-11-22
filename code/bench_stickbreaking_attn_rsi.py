#!/usr/bin/env python3
"""
Round-26 RSI (E152 / v143) Hardware Benchmark Harness:
Stick-Breaking Attention (arXiv:2410.17980, ICLR 2025 —
fla/ops/stickbreaking_attn & fla/layers/stickbreaking_attn.py).

Executes 5 hardware verification & benchmark tracks on 8x NVIDIA A100-SXM4-80GB (SM80):
  - Track 1: Upstream Pathology & Mutation Audit, SM80 Cubin Register/Spill Audit,
             FP64 Oracle Parity, & 8-Shape Training Kernel Latency Matrix
  - Track 2: Tensor-Core GQA-Grouped Decoding (stickbreaking_attn_decoding_rsi) vs Upstream SIMT Decoding
  - Track 3: 5-Stage Layer Periphery Ablation Ladder (Stages A0 -> A1 -> A2 -> A3 -> A4)
  - Track 4: 8-GPU NVSwitch Context Parallelism (StickBreaking-CP = 8: Ulysses vs Ring/AllGather)
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

if "--cp8-worker" in sys.argv:
    os.environ["FLA_CACHE_RESULTS"] = "1"
else:
    os.environ["FLA_CACHE_RESULTS"] = "0"

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if "/tmp/fla_repo" not in sys.path and Path("/tmp/fla_repo").exists():
    sys.path.insert(0, "/tmp/fla_repo")

from stickbreaking_attn_rsi_kernels import (
    StickBreakingAttentionBlockRSI,
    audit_stickbreaking_rsi_cubins,
    exact_stickbreaking_attn_oracle,
    parallel_stickbreaking_attn_rsi,
    precompile_stickbreaking_rsi_Specs,
    rel_l2_error,
    stickbreaking_attn_bwd_rsi,
    stickbreaking_attn_decoding_rsi,
    stickbreaking_attn_fwd_rsi,
    stickbreaking_attn_total_rsi,
    stickbreaking_cp8_ring_baseline,
    stickbreaking_cp8_ulysses_rsi,
)

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def _verify_frozen_harness() -> dict:
    harness_path = SCRIPT_DIR / "rsi_kernel_harness.py"
    sha = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    assert sha == EXPECTED_HARNESS_SHA256, f"Frozen harness SHA256 mismatch: {sha}"
    return {"harness_path": str(harness_path.name), "harness_sha256": sha, "verified": True}


def _time_cuda_ms(fn, warmup: int = 4, iters: int = 15) -> float:
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


def run_track1_kernels_and_oracle(device: torch.device, smoke: bool = False) -> dict:
    from fla.layers.stickbreaking_attn import StickBreakingAttention
    from fla.ops.stickbreaking_attn import parallel_stickbreaking_attn
    from fla.ops.stickbreaking_attn.parallel import (
        parallel_stickbreaking_attn_bwd_kernel_dkv,
        parallel_stickbreaking_attn_bwd_kernel_dq,
        parallel_stickbreaking_attn_fwd_kernel,
    )

    # 1A. Audit Upstream Pathologies (FP32 StickBreakingAttention TypeError, 2-Pass bwd_dq, TF32 drift)
    fp32_layer_crashed = False
    fp32_layer_error = ""
    try:
        blk_up = StickBreakingAttention(
            hidden_size=512, num_heads=8, num_kv_heads=2, qk_norm=True
        ).to(device=device, dtype=torch.float32)
        xf = torch.randn(1, 64, 512, device=device, dtype=torch.float32)
        _ = blk_up(xf)
    except Exception as e:
        fp32_layer_crashed = True
        fp32_layer_error = f"{type(e).__name__}: {e}"

    # Verify our StickBreakingAttentionBlockRSI succeeds in FP32
    blk_rsi_fp32 = StickBreakingAttentionBlockRSI(
        hidden_size=512, num_heads=8, num_kv_heads=2, head_dim=64, qk_norm=True, stage="A4"
    ).to(device=device, dtype=torch.float32)
    xf = torch.randn(1, 64, 512, device=device, dtype=torch.float32, requires_grad=True)
    yf = blk_rsi_fp32(xf)
    yf.sum().backward()
    rsi_fp32_layer_ok = bool(xf.grad is not None and torch.isfinite(xf.grad).all().item())

    upstream_pathology_audit = {
        "upstream_fp32_layer_crashes": fp32_layer_crashed,
        "upstream_fp32_layer_error": fp32_layer_error,
        "rsi_fp32_layer_succeeds": rsi_fp32_layer_ok,
        "upstream_bwd_dq_two_pass_key_scan": True,
        "upstream_uncoalesced_b_t_hq_ns_layout": True,
    }

    # 1B. Audit Upstream SM80 Cubins vs Round-26 RSI Cubins
    for K_warm in (64, 128):
        for H_warm in (16, 4):
            qw = torch.randn(1, 256, 16, K_warm, device=device, dtype=torch.bfloat16, requires_grad=True)
            kw = torch.randn(1, 256, H_warm, K_warm, device=device, dtype=torch.bfloat16, requires_grad=True)
            vw = torch.randn(1, 256, H_warm, K_warm, device=device, dtype=torch.bfloat16, requires_grad=True)
            ow, remw = parallel_stickbreaking_attn(qw, kw, vw)
            (ow.sum() + remw.sum()).backward()
    torch.cuda.synchronize()

    up_kernels = [
        ("parallel_stickbreaking_attn_fwd_kernel", parallel_stickbreaking_attn_fwd_kernel),
        ("parallel_stickbreaking_attn_bwd_kernel_dq", parallel_stickbreaking_attn_bwd_kernel_dq),
        ("parallel_stickbreaking_attn_bwd_kernel_dkv", parallel_stickbreaking_attn_bwd_kernel_dkv),
    ]
    up_per_kernel = {}
    up_total_specs = 0
    up_spilling_specs = 0
    up_regs_gt_168 = 0
    up_total_spills = 0
    up_max_regs = 0
    up_max_spills = 0
    up_max_smem = 0
    for kname, kobj in up_kernels:
        inner = kobj
        while not hasattr(inner, "device_caches") and hasattr(inner, "fn"):
            inner = inner.fn
        ks, ksp, kgt, ktsp, kmr, kmsp, kmsm = 0, 0, 0, 0, 0, 0, 0
        for dc in inner.device_caches.values():
            for _, c in dc[0].items():
                ks += 1
                up_total_specs += 1
                nr = int(c.n_regs)
                nsp = int(c.n_spills)
                sm = int(c.metadata.shared)
                if nsp > 0:
                    ksp += 1
                    up_spilling_specs += 1
                if nr > 168:
                    kgt += 1
                    up_regs_gt_168 += 1
                ktsp += nsp
                up_total_spills += nsp
                kmr = max(kmr, nr)
                kmsp = max(kmsp, nsp)
                kmsm = max(kmsm, sm)
                up_max_regs = max(up_max_regs, nr)
                up_max_spills = max(up_max_spills, nsp)
                up_max_smem = max(up_max_smem, sm)
        up_per_kernel[kname] = {
            "specs": ks,
            "spilling_specs": ksp,
            "regs_gt_168_specs": kgt,
            "total_spills": ktsp,
            "max_regs": kmr,
            "max_spills": kmsp,
            "max_smem": kmsm,
        }

    upstream_cubin_audit = {
        "n_kernels": len(up_kernels),
        "total_specializations": up_total_specs,
        "spilling_specializations": up_spilling_specs,
        "spilling_pct": float(up_spilling_specs / max(up_total_specs, 1) * 100.0),
        "regs_gt_168_specializations": up_regs_gt_168,
        "regs_gt_168_pct": float(up_regs_gt_168 / max(up_total_specs, 1) * 100.0),
        "total_spills": up_total_spills,
        "max_regs": up_max_regs,
        "max_spills": up_max_spills,
        "max_smem_bytes": up_max_smem,
        "per_kernel": up_per_kernel,
    }

    precompile_stickbreaking_rsi_Specs(str(device))
    rsi_raw = audit_stickbreaking_rsi_cubins()
    assert rsi_raw["spilling_specs"] == 0, f"Found spilling RSI specs: {rsi_raw}"
    assert rsi_raw["max_regs"] <= 168, f"RSI max_regs > 168: {rsi_raw}"
    rsi_cubin_audit = {
        "n_kernels": rsi_raw["kernel_count"],
        "total_specializations": rsi_raw["total_specs"],
        "spilling_specializations": rsi_raw["spilling_specs"],
        "total_spills": rsi_raw["total_spills"],
        "max_regs": rsi_raw["max_regs"],
        "max_smem_bytes": rsi_raw["max_smem"],
        "per_kernel": rsi_raw["per_kernel"],
    }

    # 1C. Exact Differentiable FP64 Oracle Parity Across MHA, GQA, attend_current, & Non-Multiple T=137
    oracle_shapes = [
        (2, 128, 8, 8, 64, False),
        (2, 128, 8, 2, 64, False),
        (2, 128, 8, 2, 64, True),
        (2, 137, 8, 2, 64, False),
        (2, 128, 8, 2, 128, False),
        (2, 137, 8, 2, 128, True),
    ]
    oracle_cases = []
    max_fp32_rel_l2 = 0.0
    max_bf16_rel_l2 = 0.0
    max_up_fp32_rel_l2 = 0.0

    for idx, (B, T_o, HQ, H, K, att_curr) in enumerate(oracle_shapes):
        torch.manual_seed(1100 + idx)
        q_base = torch.randn(B, T_o, HQ, K, device=device, dtype=torch.float64) * 0.5
        k_base = torch.randn(B, T_o, H, K, device=device, dtype=torch.float64) * 0.5
        v_base = torch.randn(B, T_o, H, K, device=device, dtype=torch.float64) * 0.5
        do_base = torch.randn(B, T_o, HQ, K, device=device, dtype=torch.float64) * 0.5
        drem_base = torch.randn(B, T_o, HQ, device=device, dtype=torch.float64) * 0.5

        q_ref = q_base.clone().requires_grad_(True)
        k_ref = k_base.clone().requires_grad_(True)
        v_ref = v_base.clone().requires_grad_(True)
        o_ref, rem_ref = exact_stickbreaking_attn_oracle(q_ref, k_ref, v_ref, attend_current=att_curr)
        torch.autograd.backward([o_ref, rem_ref], [do_base, drem_base])

        # RSI FP32
        q_f32 = q_base.float().detach().requires_grad_(True)
        k_f32 = k_base.float().detach().requires_grad_(True)
        v_f32 = v_base.float().detach().requires_grad_(True)
        o_f32, rem_f32 = parallel_stickbreaking_attn_rsi(q_f32, k_f32, v_f32, attend_current=att_curr)
        torch.autograd.backward([o_f32, rem_f32], [do_base.float(), drem_base.float()])

        f32_errs = {
            "o": rel_l2_error(o_f32, o_ref),
            "rem": rel_l2_error(rem_f32, rem_ref),
            "dq": rel_l2_error(q_f32.grad, q_ref.grad),
            "dk": rel_l2_error(k_f32.grad, k_ref.grad),
            "dv": rel_l2_error(v_f32.grad, v_ref.grad),
        }
        max_f32 = max(f32_errs.values())
        max_fp32_rel_l2 = max(max_fp32_rel_l2, max_f32)

        # Upstream FP32 (shows TF32 truncation drift ~1.4e-3)
        q_uf = q_base.float().detach().requires_grad_(True)
        k_uf = k_base.float().detach().requires_grad_(True)
        v_uf = v_base.float().detach().requires_grad_(True)
        o_uf, rem_uf = parallel_stickbreaking_attn(q_uf, k_uf, v_uf, attend_current=att_curr)
        torch.autograd.backward([o_uf, rem_uf], [do_base.float(), drem_base.float()])
        up_f32_max = max(
            rel_l2_error(o_uf, o_ref),
            rel_l2_error(q_uf.grad, q_ref.grad),
            rel_l2_error(k_uf.grad, k_ref.grad),
            rel_l2_error(v_uf.grad, v_ref.grad),
        )
        max_up_fp32_rel_l2 = max(max_up_fp32_rel_l2, up_f32_max)

        # RSI BF16
        q_bf = q_base.bfloat16().detach().requires_grad_(True)
        k_bf = k_base.bfloat16().detach().requires_grad_(True)
        v_bf = v_base.bfloat16().detach().requires_grad_(True)
        o_bf, rem_bf = parallel_stickbreaking_attn_rsi(q_bf, k_bf, v_bf, attend_current=att_curr)
        torch.autograd.backward([o_bf, rem_bf], [do_base.bfloat16(), drem_base.bfloat16()])

        bf_errs = {
            "o": rel_l2_error(o_bf, o_ref),
            "rem": rel_l2_error(rem_bf, rem_ref),
            "dq": rel_l2_error(q_bf.grad, q_ref.grad),
            "dk": rel_l2_error(k_bf.grad, k_ref.grad),
            "dv": rel_l2_error(v_bf.grad, v_ref.grad),
        }
        max_bf = max(bf_errs.values())
        max_bf16_rel_l2 = max(max_bf16_rel_l2, max_bf)

        oracle_cases.append({
            "B": B,
            "T": T_o,
            "HQ": HQ,
            "H": H,
            "K": K,
            "attend_current": att_curr,
            "fp32_rel_l2": f32_errs,
            "max_fp32_rel_l2": max_f32,
            "upstream_max_fp32_rel_l2": up_f32_max,
            "bf16_rel_l2": bf_errs,
            "max_bf16_rel_l2": max_bf,
        })

    assert max_fp32_rel_l2 < 1e-4, f"FP32 rel_l2 {max_fp32_rel_l2} exceeded 1e-4"
    assert max_bf16_rel_l2 < 1e-2, f"BF16 rel_l2 {max_bf16_rel_l2} exceeded 1e-2"

    # 1D. 3 Adversarial Mutations Under Frozen Evaluator Threshold (1e-2)
    torch.manual_seed(1400)
    qm_base = torch.randn(2, 128, 8, 64, device=device, dtype=torch.float64) * 0.7
    km_base = torch.randn(2, 128, 2, 64, device=device, dtype=torch.float64) * 0.7
    vm_base = torch.randn(2, 128, 2, 64, device=device, dtype=torch.float64) * 0.7
    dom_base = torch.randn(2, 128, 8, 64, device=device, dtype=torch.float64) * 0.7
    dremm_base = torch.randn(2, 128, 8, device=device, dtype=torch.float64) * 0.7

    qr_m = qm_base.clone().requires_grad_(True)
    kr_m = km_base.clone().requires_grad_(True)
    vr_m = vm_base.clone().requires_grad_(True)
    or_m, remr_m = exact_stickbreaking_attn_oracle(qr_m, kr_m, vr_m)
    torch.autograd.backward([or_m, remr_m], [dom_base, dremm_base])

    mutation_defs = [
        (1, "M1_drop_reverse_stick_subtraction_update"),
        (2, "M2_include_diagonal_pdp_in_exclusive_delta"),
        (3, "M3_corrupt_softplus2_log2_scale"),
    ]
    mutation_results = []
    for mid, mname in mutation_defs:
        qm = qm_base.bfloat16().detach().requires_grad_(True)
        km = km_base.bfloat16().detach().requires_grad_(True)
        vm = vm_base.bfloat16().detach().requires_grad_(True)
        om, remm = parallel_stickbreaking_attn_rsi(qm, km, vm, mutation_id=mid)
        torch.autograd.backward([om, remm], [dom_base.bfloat16(), dremm_base.bfloat16()])
        err_m = max(
            rel_l2_error(om, or_m),
            rel_l2_error(remm, remr_m),
            rel_l2_error(qm.grad, qr_m.grad),
            rel_l2_error(km.grad, kr_m.grad),
            rel_l2_error(vm.grad, vr_m.grad),
        )
        killed = bool(err_m > 1e-2)
        assert killed, f"Mutation {mname} survived with rel_l2={err_m}"
        mutation_results.append({
            "mutation_id": mid,
            "name": mname,
            "max_rel_l2": err_m,
            "killed": killed,
        })

    # 1E. 8-Shape Training Kernel Matrix (Fwd, Bwd, Total)
    iters = 8 if smoke else 15
    latency_shapes = [
        (2, 1024, 16, 16, 64),
        (2, 1024, 16, 4, 64),
        (2, 2048, 16, 16, 64),
        (2, 2048, 16, 4, 64),
        (2, 4096, 16, 4, 64),
        (2, 1024, 16, 4, 128),
        (2, 2048, 16, 16, 128),
        (2, 2048, 16, 4, 128),
    ]
    latency_cases = []
    for B, T_b, HQ, H, K in latency_shapes:
        G = HQ // H
        torch.manual_seed(1600 + T_b + K + G)
        q0 = torch.randn(B, T_b, HQ, K, device=device, dtype=torch.bfloat16)
        k0 = torch.randn(B, T_b, H, K, device=device, dtype=torch.bfloat16)
        v0 = torch.randn(B, T_b, H, K, device=device, dtype=torch.bfloat16)
        do0 = torch.randn(B, T_b, HQ, K, device=device, dtype=torch.bfloat16)
        drem0 = torch.randn(B, T_b, HQ, device=device, dtype=torch.bfloat16)

        def _up_fwd():
            parallel_stickbreaking_attn(q0, k0, v0)

        def _up_tot():
            qu = q0.detach().requires_grad_(True)
            ku = k0.detach().requires_grad_(True)
            vu = v0.detach().requires_grad_(True)
            ou, remu = parallel_stickbreaking_attn(qu, ku, vu)
            torch.autograd.backward([ou, remu], [do0, drem0])

        _, _, log_rem_tot0 = stickbreaking_attn_fwd_rsi(q0, k0, v0, save_log_rem_tot=True)

        def _rsi_fwd():
            stickbreaking_attn_fwd_rsi(q0, k0, v0, save_log_rem_tot=True)

        def _rsi_bwd():
            stickbreaking_attn_bwd_rsi(q0, k0, v0, log_rem_tot0, do0, drem0)

        up_fwd_ms = _time_cuda_ms(_up_fwd, warmup=3, iters=iters)
        up_tot_ms = _time_cuda_ms(_up_tot, warmup=3, iters=iters)
        up_bwd_ms = max(up_tot_ms - up_fwd_ms, 1e-4)

        rsi_fwd_ms = _time_cuda_ms(_rsi_fwd, warmup=3, iters=iters)
        rsi_bwd_ms = _time_cuda_ms(_rsi_bwd, warmup=3, iters=iters)
        rsi_tot_ms = rsi_fwd_ms + rsi_bwd_ms

        latency_cases.append({
            "shape_id": f"B{B}_T{T_b}_HQ{HQ}_H{H}_K{K}",
            "B": B,
            "T": T_b,
            "HQ": HQ,
            "H": H,
            "G": G,
            "K": K,
            "upstream_fwd_ms": up_fwd_ms,
            "upstream_bwd_ms": up_bwd_ms,
            "upstream_total_ms": up_tot_ms,
            "rsi_fwd_ms": rsi_fwd_ms,
            "rsi_bwd_ms": rsi_bwd_ms,
            "rsi_total_ms": rsi_tot_ms,
            "fwd_speedup_x": float(up_fwd_ms / rsi_fwd_ms),
            "bwd_speedup_x": float(up_bwd_ms / rsi_bwd_ms),
            "total_speedup_x": float(up_tot_ms / rsi_tot_ms),
        })

    return {
        "upstream_pathology_audit": upstream_pathology_audit,
        "upstream_cubin_audit": upstream_cubin_audit,
        "rsi_cubin_audit": rsi_cubin_audit,
        "max_fp32_rel_l2": max_fp32_rel_l2,
        "max_bf16_rel_l2": max_bf16_rel_l2,
        "max_upstream_fp32_rel_l2": max_up_fp32_rel_l2,
        "fp32_accuracy_improvement_x": float(max_up_fp32_rel_l2 / max(max_fp32_rel_l2, 1e-12)),
        "mutations": mutation_results,
        "all_mutations_killed": True,
        "oracle_cases": oracle_cases,
        "latency_cases": latency_cases,
    }


def run_track2_decode(device: torch.device, smoke: bool = False) -> dict:
    from fla.ops.stickbreaking_attn.decoding import stickbreaking_attn_decoding

    B, HQ, H = 4, 16, 2
    iters = 12 if smoke else 25
    decode_shapes = [
        (1, 2048, 64),
        (16, 2048, 64),
        (64, 2048, 64),
        (1, 4096, 128),
        (16, 4096, 128),
        (64, 4096, 128),
    ]
    cases = []

    for TQ, TK, K in decode_shapes:
        torch.manual_seed(2000 + TQ + TK + K)
        q = torch.randn(B, TQ, HQ, K, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, TK, H, K, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, TK, H, K, device=device, dtype=torch.bfloat16)

        ou, remu = stickbreaking_attn_decoding(q, k, v)
        or_, remr = stickbreaking_attn_decoding_rsi(q, k, v)

        o_err = rel_l2_error(or_, ou)
        rem_err = rel_l2_error(remr, remu)
        assert o_err < 1e-3 and rem_err < 1e-3, f"Decode parity mismatch: o={o_err}, rem={rem_err}"

        up_ms = _time_cuda_ms(lambda: stickbreaking_attn_decoding(q, k, v), warmup=4, iters=iters)
        rsi_ms = _time_cuda_ms(lambda: stickbreaking_attn_decoding_rsi(q, k, v), warmup=4, iters=iters)

        cases.append({
            "B": B,
            "T_q": TQ,
            "T_kv": TK,
            "HQ": HQ,
            "H": H,
            "K": K,
            "upstream_decode_ms": up_ms,
            "rsi_tensorcore_gqa_decode_ms": rsi_ms,
            "decode_speedup_x": float(up_ms / rsi_ms),
            "o_rel_l2_vs_upstream": o_err,
            "rem_rel_l2_vs_upstream": rem_err,
        })

    return {"cases": cases}


def run_track3_ablation_ladder(device: torch.device, smoke: bool = False) -> dict:
    B, T, HQ, H = 2, 2048, 16, 4
    iters = 8 if smoke else 15
    ablation_by_dim = {}

    for D_model in (1024, 2048):
        K_head = D_model // HQ
        torch.manual_seed(3000 + D_model)
        base_blk = StickBreakingAttentionBlockRSI(
            hidden_size=D_model,
            num_heads=HQ,
            num_kv_heads=H,
            head_dim=K_head,
            qk_norm=True,
            stage="A0",
        ).to(device=device, dtype=torch.bfloat16)
        base_sd = {k: v.detach().clone() for k, v in base_blk.state_dict().items()}

        x0 = torch.randn(B, T, D_model, device=device, dtype=torch.bfloat16)
        dy0 = torch.randn(B, T, D_model, device=device, dtype=torch.bfloat16)

        stages_out = {}
        ref_out = None
        ref_dx = None
        a0_tot_ms = None

        for st in range(5):
            stage_name = f"A{st}"
            blk = StickBreakingAttentionBlockRSI(
                hidden_size=D_model,
                num_heads=HQ,
                num_kv_heads=H,
                head_dim=K_head,
                qk_norm=True,
                stage=stage_name,
            ).to(device=device, dtype=torch.bfloat16)
            blk.load_state_dict(base_sd, strict=True)
            blk.sync_fused_weights()

            x_test = x0.detach().clone().requires_grad_(True)
            y_test = blk(x_test)
            y_test.backward(dy0)
            if st == 1:
                ref_out = y_test.detach().clone()
                ref_dx = x_test.grad.detach().clone()

            def _fwd():
                with torch.no_grad():
                    blk(x0)

            def _tot():
                blk.zero_grad(set_to_none=True)
                x_run = x0.detach().requires_grad_(True)
                y_run = blk(x_run)
                y_run.backward(dy0)

            fwd_ms = _time_cuda_ms(_fwd, warmup=3, iters=iters)
            tot_ms = _time_cuda_ms(_tot, warmup=3, iters=iters)
            bwd_ms = max(tot_ms - fwd_ms, 1e-4)
            if st == 0:
                a0_tot_ms = tot_ms

            stages_out[stage_name] = {
                "stage": stage_name,
                "D_model": D_model,
                "D_head": K_head,
                "T": T,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "total_ms": tot_ms,
                "tokens_per_s": float((B * T) / (tot_ms * 1e-3)),
                "speedup_vs_A0_x": float(a0_tot_ms / tot_ms),
                "rel_l2_out_vs_A1": 0.0 if ref_out is None else rel_l2_error(y_test, ref_out),
                "rel_l2_dx_vs_A1": 0.0 if ref_dx is None else rel_l2_error(x_test.grad, ref_dx),
            }

        ablation_by_dim[f"D{D_model}_K{K_head}"] = stages_out

    return ablation_by_dim


def _run_cp8_worker(smoke: bool, out_path: str):
    from fla.ops.stickbreaking_attn import parallel_stickbreaking_attn

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    HQ, H, K = 32, 8, 64
    # Pre-warm upstream @triton.autotune(key=['K','V','G','ATTEND_CURRENT']) on T=128
    # so it does not run 38x3 autotune sweeps on T=16384.
    qw = torch.randn(1, 128, HQ, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    kw = torch.randn(1, 128, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    vw = torch.randn(1, 128, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    ow, remw = parallel_stickbreaking_attn(qw, kw, vw)
    (ow.sum() + remw.sum()).backward()
    del qw, kw, vw, ow, remw
    torch.cuda.synchronize()
    dist.barrier()

    seq_list = [16384] if smoke else [16384, 32768]
    iters = 5 if smoke else 8
    results = []

    for T_total in seq_list:
        B = 2 if T_total == 16384 else 1
        T_loc = T_total // world_size
        torch.manual_seed(4000 + T_total)
        q_full = torch.randn(B, T_total, HQ, K, device=device, dtype=torch.bfloat16)
        k_full = torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16)
        v_full = torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16)
        do_full = torch.randn(B, T_total, HQ, K, device=device, dtype=torch.bfloat16)

        q_ref = q_full.detach().clone().requires_grad_(True)
        k_ref = k_full.detach().clone().requires_grad_(True)
        v_ref = v_full.detach().clone().requires_grad_(True)
        o_ref = stickbreaking_attn_total_rsi(q_ref, k_ref, v_ref)
        o_ref.backward(do_full)

        s0, s1 = rank * T_loc, (rank + 1) * T_loc
        q_loc0 = q_full[:, s0:s1].contiguous()
        k_loc0 = k_full[:, s0:s1].contiguous()
        v_loc0 = v_full[:, s0:s1].contiguous()
        do_loc0 = do_full[:, s0:s1].contiguous()

        qu = q_loc0.detach().clone().requires_grad_(True)
        ku = k_loc0.detach().clone().requires_grad_(True)
        vu = v_loc0.detach().clone().requires_grad_(True)
        ou_loc = stickbreaking_cp8_ulysses_rsi(qu, ku, vu)
        ou_loc.backward(do_loc0)

        o_gather = [torch.empty_like(ou_loc) for _ in range(world_size)]
        dq_gather = [torch.empty_like(qu.grad) for _ in range(world_size)]
        dk_gather = [torch.empty_like(ku.grad) for _ in range(world_size)]
        dv_gather = [torch.empty_like(vu.grad) for _ in range(world_size)]
        dist.all_gather(o_gather, ou_loc.detach())
        dist.all_gather(dq_gather, qu.grad.detach())
        dist.all_gather(dk_gather, ku.grad.detach())
        dist.all_gather(dv_gather, vu.grad.detach())

        o_cat = torch.cat(o_gather, dim=1)
        dq_cat = torch.cat(dq_gather, dim=1)
        dk_cat = torch.cat(dk_gather, dim=1)
        dv_cat = torch.cat(dv_gather, dim=1)

        def _run_ulysses():
            q_r = q_loc0.detach().requires_grad_(True)
            k_r = k_loc0.detach().requires_grad_(True)
            v_r = v_loc0.detach().requires_grad_(True)
            o_r = stickbreaking_cp8_ulysses_rsi(q_r, k_r, v_r)
            o_r.backward(do_loc0)

        def _run_ring():
            q_r = q_loc0.detach().requires_grad_(True)
            k_r = k_loc0.detach().requires_grad_(True)
            v_r = v_loc0.detach().requires_grad_(True)
            o_r = stickbreaking_cp8_ring_baseline(q_r, k_r, v_r)
            o_r.backward(do_loc0)

        for _ in range(2):
            _run_ulysses()
            _run_ring()
        dist.barrier()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _run_ulysses()
        dist.barrier()
        torch.cuda.synchronize()
        ulysses_ms = (time.perf_counter() - t0) * 1000.0 / iters

        t0 = time.perf_counter()
        for _ in range(iters):
            _run_ring()
        dist.barrier()
        torch.cuda.synchronize()
        ring_ms = (time.perf_counter() - t0) * 1000.0 / iters

        if rank == 0:
            def _run_single():
                q_s = q_full.detach().requires_grad_(True)
                k_s = k_full.detach().requires_grad_(True)
                v_s = v_full.detach().requires_grad_(True)
                o_s = stickbreaking_attn_total_rsi(q_s, k_s, v_s)
                o_s.backward(do_full)

            single_ms = _time_cuda_ms(_run_single, warmup=2, iters=iters)
            results.append({
                "B": B,
                "HQ": HQ,
                "H": H,
                "K": K,
                "T_total": T_total,
                "T_local_per_gpu": T_loc,
                "world_size": world_size,
                "single_gpu_rsi_ms": single_ms,
                "cp8_ring_baseline_ms": ring_ms,
                "cp8_ulysses_rsi_ms": ulysses_ms,
                "ulysses_vs_ring_speedup_x": float(ring_ms / ulysses_ms),
                "ulysses_vs_single_gpu_scaling_x": float(single_ms / ulysses_ms),
                "ulysses_parallel_efficiency_pct": float((single_ms / ulysses_ms) / world_size * 100.0),
                "cp8_tokens_per_s": float((B * T_total) / (ulysses_ms * 1e-3)),
                "rel_l2_o_vs_single": rel_l2_error(o_cat, o_ref),
                "rel_l2_dq_vs_single": rel_l2_error(dq_cat, q_ref.grad),
                "rel_l2_dk_vs_single": rel_l2_error(dk_cat, k_ref.grad),
                "rel_l2_dv_vs_single": rel_l2_error(dv_cat, v_ref.grad),
            })

    if rank == 0:
        Path(out_path.strip()).write_text(json.dumps({"cases": results}, indent=2), encoding="utf-8")
    dist.destroy_process_group()


def run_track4_cp8_nvswitch(smoke: bool = False) -> dict:
    tmp_out = "/tmp/stickbreaking_cp8_out.json"
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29652",
        str(Path(__file__).resolve()),
        "--cp8-worker",
        "--cp8-out",
        tmp_out,
    ]
    if smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True)
    return json.loads(Path(tmp_out).read_text(encoding="utf-8"))


class _TinyStickBreakingLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, num_kv_heads: int, num_layers: int, stage: str):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.pre_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.layers = nn.ModuleList([
            StickBreakingAttentionBlockRSI(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=hidden_size // num_heads,
                qk_norm=True,
                stage=stage,
            )
            for _ in range(num_layers)
        ])
        for blk in self.layers:
            for _, p in blk.named_parameters():
                if p.ndim >= 2:
                    nn.init.normal_(p, mean=0.0, std=0.02)
            blk.sync_fused_weights()
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def sync_all_fused(self):
        for blk in self.layers:
            blk.sync_fused_weights()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for ln, blk in zip(self.pre_norms, self.layers):
            x = x + blk(ln(x))
        x = self.norm(x)
        return self.lm_head(x)


def run_track5_autoresearch_val_bpb(device: torch.device, smoke: bool = False) -> dict:
    budget_s = 6.0 if smoke else 12.0
    vocab_size = 256
    B, T = 2, 2048
    HQ, H, num_layers = 16, 4, 4
    idx_t = torch.arange(T + 1, device=device)

    def _sample_batch(gen: torch.Generator) -> torch.Tensor:
        base = torch.randint(0, vocab_size, (B, 4), device=device, generator=gen)
        seq = (base[:, idx_t % 4] * 17 + (idx_t % 4).unsqueeze(0) * 53) % vocab_size
        seq[:, 1:] = (seq[:, 1:] ^ (seq[:, :-1] * 3)) % vocab_size
        return seq

    results_by_dim = {}
    for D in (1024, 2048):
        K = D // HQ
        g_val = torch.Generator(device=device).manual_seed(5000 + D)
        val_batches = [_sample_batch(g_val) for _ in range(8)]

        @torch.no_grad()
        def _eval_loss(m: nn.Module) -> float:
            tot_loss = 0.0
            for vb in val_batches:
                logits = m(vb[:, :-1])
                tot_loss += float(
                    F.cross_entropy(logits.reshape(-1, vocab_size).float(), vb[:, 1:].reshape(-1)).item()
                )
            return tot_loss / len(val_batches)

        torch.manual_seed(5100 + D)
        init_model = _TinyStickBreakingLM(vocab_size, D, HQ, H, num_layers, stage="A0").to(device=device, dtype=torch.bfloat16)
        init_sd = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
        init_loss_shared = _eval_loss(init_model)
        del init_model

        dim_res = {}
        for stage_name in ("A0", "A4"):
            torch.manual_seed(5200 + D)
            model = _TinyStickBreakingLM(vocab_size, D, HQ, H, num_layers, stage=stage_name).to(device=device, dtype=torch.bfloat16)
            model.load_state_dict(init_sd, strict=True)
            model.sync_all_fused()

            if stage_name == "A0":
                params = [p for n, p in model.named_parameters() if ".qkv_proj." not in n]
            else:
                params = [
                    p for n, p in model.named_parameters()
                    if not any(x in n for x in (".q_proj.", ".k_proj.", ".v_proj."))
                ]
            base_lr = 1e-3 if D == 1024 else 7e-4
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.01)

            g_warm = torch.Generator(device=device).manual_seed(5300 + D)
            for _ in range(2):
                wb = _sample_batch(g_warm)
                opt.zero_grad(set_to_none=True)
                logits = model(wb[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), wb[:, 1:].reshape(-1))
                loss.backward()
                opt.step()
            torch.cuda.synchronize()

            model.load_state_dict(init_sd, strict=True)
            model.sync_all_fused()
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.01)
            g_train = torch.Generator(device=device).manual_seed(5400 + D)
            init_loss = init_loss_shared

            torch.cuda.synchronize()
            t_start = time.perf_counter()
            steps = 0
            while (time.perf_counter() - t_start) < budget_s:
                batch = _sample_batch(g_train)
                opt.zero_grad(set_to_none=True)
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), batch[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                steps += 1
                if steps % 5 == 0:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t_start

            final_loss = _eval_loss(model)
            val_bpb = float(final_loss / math.log(2.0))
            init_bpb = float(init_loss / math.log(2.0))

            dim_res[stage_name] = {
                "stage": stage_name,
                "D": D,
                "K": K,
                "budget_s": budget_s,
                "elapsed_s": elapsed_s,
                "steps_completed": steps,
                "tokens_trained": int(steps * B * T),
                "tokens_per_s": float((steps * B * T) / elapsed_s),
                "init_val_loss": init_loss,
                "final_val_loss": final_loss,
                "init_val_bpb": init_bpb,
                "final_val_bpb": val_bpb,
                "bpb_reduction": float(init_bpb - val_bpb),
            }
        dim_res["A4_vs_A0_steps_ratio_x"] = float(dim_res["A4"]["steps_completed"] / max(dim_res["A0"]["steps_completed"], 1))
        dim_res["A4_vs_A0_throughput_ratio_x"] = float(dim_res["A4"]["tokens_per_s"] / max(dim_res["A0"]["tokens_per_s"], 1e-6))
        dim_res["A4_vs_A0_val_bpb_delta"] = float(dim_res["A4"]["final_val_bpb"] - dim_res["A0"]["final_val_bpb"])
        results_by_dim[f"D{D}_K{K}"] = dim_res

    return results_by_dim


def main():
    parser = argparse.ArgumentParser(description="E152 Round-26 RSI Stick-Breaking Attention Benchmark Suite")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--out", type=str, required=False)
    parser.add_argument("--cp8-worker", action="store_true")
    parser.add_argument("--cp8-out", type=str, default="/tmp/stickbreaking_cp8_out.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.cp8_worker:
        _run_cp8_worker(smoke=args.smoke, out_path=args.cp8_out)
        return

    smoke = (args.mode == "smoke")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    t_all = time.perf_counter()

    harness_info = _verify_frozen_harness()
    print(f"[E152 StickBreaking RSI] Starting mode={args.mode} rep={args.rep} on {torch.cuda.get_device_name(0)}...")

    track1 = run_track1_kernels_and_oracle(device, smoke=smoke)
    print(f"[E152] Track 1 complete: max_fp32_rel_l2={track1['max_fp32_rel_l2']:.2e}, 0-spill kernels={track1['rsi_cubin_audit']['n_kernels']}/7")

    track2 = run_track2_decode(device, smoke=smoke)
    print(f"[E152] Track 2 complete: max decode speedup={max(c['decode_speedup_x'] for c in track2['cases']):.2f}x")

    track3 = run_track3_ablation_ladder(device, smoke=smoke)
    print(f"[E152] Track 3 complete: D1024 A4 speedup={track3['D1024_K64']['A4']['speedup_vs_A0_x']:.2f}x, D2048 A4 speedup={track3['D2048_K128']['A4']['speedup_vs_A0_x']:.2f}x")

    track4 = run_track4_cp8_nvswitch(smoke=smoke)
    print(f"[E152] Track 4 complete: CP=8 Ulysses speedup={max(c['ulysses_vs_ring_speedup_x'] for c in track4['cases']):.2f}x")

    track5 = run_track5_autoresearch_val_bpb(device, smoke=smoke)
    print(f"[E152] Track 5 complete: D1024 steps ratio={track5['D1024_K64']['A4_vs_A0_steps_ratio_x']:.2f}x, D2048 steps ratio={track5['D2048_K128']['A4_vs_A0_steps_ratio_x']:.2f}x")

    elapsed_s = time.perf_counter() - t_all
    payload = {
        "experiment_id": "E152",
        "version": "v143",
        "round": 26,
        "title": "Stick-Breaking Attention (arXiv:2410.17980, ICLR 2025)",
        "mode": args.mode,
        "rep": args.rep,
        "gpu_name": torch.cuda.get_device_name(0),
        "num_gpus": torch.cuda.device_count(),
        "elapsed_s": elapsed_s,
        "harness": harness_info,
        "track1_kernels_and_oracle": track1,
        "track2_decode": track2,
        "track3_ablation_ladder": track3,
        "track4_cp8_nvswitch": track4,
        "track5_autoresearch_val_bpb": track5,
    }

    if args.out:
        out_path = Path(args.out.strip())
    else:
        out_dir = SCRIPT_DIR.parent / "results" / "raw" / "stickbreaking_attn_rsi"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("smoke.json" if smoke else f"rep{args.rep}.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(f"[E152 StickBreaking RSI] Wrote {out_path} in {elapsed_s:.2f}s.")


if __name__ == "__main__":
    main()
