#!/usr/bin/env python3
"""
Round-27 RSI (E153 / v144) Hardware Benchmark Harness:
Test-Time Training (TTT-Linear): Learning to (Learn at Test Time): RNNs with Expressive Hidden States
(arXiv:2407.04620, ICML 2024 / JMLR 2025 — fla/ops/ttt/chunk.py & fused_chunk.py).

Executes 5 hardware verification & benchmark tracks on 8x NVIDIA A100-SXM4-80GB (SM80):
  - Track 1: Upstream Pathology & Mutation Audit, SM80 Cubin Register/Spill Audit,
             FP64 Oracle Parity, & 8-Shape Training Kernel Latency Matrix (vs chunk_ttt_linear & fused_chunk_ttt_linear)
  - Track 2: Single-Pass Online Test-Time Adaptation Decoding (_ttt_decode_adapt_rsi_kernel) vs Upstream
  - Track 3: 5-Stage Layer Periphery Ablation Ladder (Stages A0 -> A1 -> A2 -> A3 -> A4)
  - Track 4: 8-GPU NVSwitch Context Parallelism (TTT-CP = 8: Ulysses vs Ring/AllGather)
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

if "--cp8-worker" in sys.argv:
    os.environ["FLA_CACHE_RESULTS"] = "1"
else:
    os.environ["FLA_CACHE_RESULTS"] = "0"

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

from ttt_linear_rsi_kernels import (
    TTTLinearBlockBaseline,
    TTTLinearBlockRSI,
    audit_ttt_rsi_cubin_resources,
    chunk_ttt_linear_rsi,
    ttt_cp8_ring_baseline,
    ttt_cp8_ulysses_rsi,
    ttt_decode_adapt_rsi,
    ttt_linear_fp64_oracle,
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


def _ttt_training_flops(B: int, T: int, H: int, K: int, V: int, BT: int = 16) -> float:
    # Fwd: kh = k@h (2*B*T*H*K*V), h update (2*B*T*H*K*V), q@k^T (2*B*T*H*BT*K),
    #      q@h (2*B*T*H*K*V), A@v_new (2*B*T*H*BT*V) => ~6*B*T*H*K*V + 2*B*T*H*BT*(K+V)
    # Bwd: ~2.5x Fwd (dv_local, bwd_norm, bwd_dqke)
    fwd = float(B * T * H * (6 * K * V + 2 * BT * (K + V)))
    bwd = 2.5 * fwd
    return fwd, bwd


def _audit_upstream_ttt_cubins(device: torch.device) -> dict:
    import fla.ops.ttt.chunk as tc
    import fla.ops.ttt.fused_chunk as tfc
    from fla.ops.ttt import chunk_ttt_linear, fused_chunk_ttt_linear

    # Trigger compilation across K=64 and K=128 in BF16 and FP32
    fp32_k128_smem_crashed = False
    fp32_k128_smem_error = ""
    for K in [64, 128]:
        for dt in [torch.bfloat16, torch.float32]:
            q = torch.randn(2, 128, 4, K, device=device, dtype=dt, requires_grad=True)
            k = torch.randn(2, 128, 4, K, device=device, dtype=dt, requires_grad=True)
            v = torch.randn(2, 128, 4, K, device=device, dtype=dt, requires_grad=True)
            w = torch.ones(4, K, device=device, dtype=dt, requires_grad=True)
            b = torch.zeros(4, K, device=device, dtype=dt, requires_grad=True)
            eta = (torch.rand(2, 128, 4, 1, device=device, dtype=dt) * 0.05).requires_grad_(True)
            try:
                o, _, _ = chunk_ttt_linear(q, k, v, w, b, eta)
                o.backward(torch.randn_like(o))
            except Exception as e:
                if K == 128 and dt == torch.float32:
                    fp32_k128_smem_crashed = True
                    fp32_k128_smem_error = f"{type(e).__name__}: {e}"
            if dt == torch.bfloat16:
                q2 = q.detach().clone().requires_grad_(True)
                k2 = k.detach().clone().requires_grad_(True)
                v2 = v.detach().clone().requires_grad_(True)
                w2 = w.detach().clone().requires_grad_(True)
                b2 = b.detach().clone().requires_grad_(True)
                e2 = eta.detach().clone().requires_grad_(True)
                o2, _, _ = fused_chunk_ttt_linear(q2, k2, v2, w2, b2, e2)
                o2.backward(torch.randn_like(o2))

    report: Dict[str, dict] = {}
    tot_specs = 0
    tot_spilling = 0
    tot_over168 = 0
    tot_spills = 0
    max_regs = 0
    max_spills = 0
    max_smem = 0

    for mod_prefix, mod in [("chunk", tc), ("fused_chunk", tfc)]:
        for name in sorted(dir(mod)):
            if "kernel" not in name:
                continue
            obj = getattr(mod, name)
            cur = obj
            while hasattr(cur, "fn") and not isinstance(cur, triton.runtime.JITFunction):
                cur = cur.fn
            dc = getattr(cur, "device_caches", None)
            if not dc:
                continue
            regs: List[int] = []
            spills: List[int] = []
            smems: List[int] = []
            for _dev, tup in dc.items():
                for _sig, c in tup[0].items():
                    if hasattr(c, "n_regs"):
                        regs.append(int(c.n_regs))
                        spills.append(int(c.n_spills))
                        smems.append(int(getattr(c.metadata, "shared", 0)))
            if not regs:
                continue
            key = f"{mod_prefix}.{name}"
            sp_cnt = sum(1 for s in spills if s > 0)
            ov_cnt = sum(1 for r in regs if r > 168)
            report[key] = {
                "total_specs": len(regs),
                "spilling_specs": sp_cnt,
                "over168_specs": ov_cnt,
                "max_regs": max(regs),
                "max_spills": max(spills),
                "total_spills": sum(spills),
                "max_smem": max(smems),
            }
            tot_specs += len(regs)
            tot_spilling += sp_cnt
            tot_over168 += ov_cnt
            tot_spills += sum(spills)
            max_regs = max(max_regs, max(regs))
            max_spills = max(max_spills, max(spills))
            max_smem = max(max_smem, max(smems))

    return {
        "kernels": report,
        "summary": {
            "total_specs": tot_specs,
            "spilling_specs": tot_spilling,
            "spilling_pct": round(100.0 * tot_spilling / max(tot_specs, 1), 2),
            "over168_specs": tot_over168,
            "over168_pct": round(100.0 * tot_over168 / max(tot_specs, 1), 2),
            "max_regs": max_regs,
            "max_spills": max_spills,
            "total_spills": tot_spills,
            "max_smem": max_smem,
            "fp32_k128_smem_crashed": fp32_k128_smem_crashed,
            "fp32_k128_smem_error": fp32_k128_smem_error,
        },
    }


def run_track1_kernels_and_oracle(device: torch.device, smoke: bool = False) -> dict:
    from fla.ops.ttt import chunk_ttt_linear, fused_chunk_ttt_linear

    # 1A. FP64 Oracle Parity (FP32 and BF16 across K=64, 128, init=False/True, non-multiple T=37)
    oracle_cases = []
    fp32_max_err = 0.0
    bf16_max_err = 0.0
    upstream_fp32_max_err = 0.0
    upstream_tail_hbt_err = 0.0

    for K in [64, 128]:
        for T_val, use_init in [(128, False), (128, True), (37, True)]:
            for dt_name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
                torch.manual_seed(100 + K + T_val + int(use_init))
                B, H = 2, 4
                q = torch.randn(B, T_val, H, K, device=device, dtype=dt, requires_grad=True)
                k = torch.randn(B, T_val, H, K, device=device, dtype=dt, requires_grad=True)
                v = torch.randn(B, T_val, H, K, device=device, dtype=dt, requires_grad=True)
                w = (torch.ones(H, K, device=device, dtype=dt) + 0.1 * torch.randn(H, K, device=device, dtype=dt)).requires_grad_(True)
                b = (0.1 * torch.randn(H, K, device=device, dtype=dt)).requires_grad_(True)
                eta = (torch.rand(B, T_val, H, 1, device=device, dtype=dt) * 0.05).requires_grad_(True)
                h0 = (torch.randn(B, H, K, K, device=device, dtype=dt) * 0.1).requires_grad_(True) if use_init else None
                hb0 = (torch.randn(B, H, 1, K, device=device, dtype=dt) * 0.1).requires_grad_(True) if use_init else None
                do = torch.randn_like(q)
                dht = torch.randn(B, H, K, K, device=device, dtype=torch.float32) * 0.1 if use_init else None
                dhbt = torch.randn(B, H, 1, K, device=device, dtype=torch.float32) * 0.1 if use_init else None

                o_ref, ht_ref, hbt_ref, inp_d = ttt_linear_fp64_oracle(
                    q, k, v, w, b, eta, initial_state=h0, initial_state_bias=hb0
                )
                if use_init:
                    torch.autograd.backward([o_ref, ht_ref, hbt_ref], [do.double(), dht.double(), dhbt.double()])
                else:
                    o_ref.backward(do.double())
                dq_r, dk_r, dv_r, dw_r, db_r, de_r = [x.grad for x in inp_d[:6]]

                o_rsi, ht_rsi, hbt_rsi = chunk_ttt_linear_rsi(
                    q, k, v, w, b, eta, initial_state=h0, initial_state_bias=hb0, output_final_state=use_init
                )
                if use_init:
                    torch.autograd.backward([o_rsi, ht_rsi, hbt_rsi], [do, dht, dhbt])
                else:
                    o_rsi.backward(do)

                errs = {
                    "o": _rel_l2(o_rsi, o_ref),
                    "dq": _rel_l2(q.grad, dq_r),
                    "dk": _rel_l2(k.grad, dk_r),
                    "dv": _rel_l2(v.grad, dv_r),
                    "dw": _rel_l2(w.grad, dw_r),
                    "db": _rel_l2(b.grad, db_r),
                    "de": _rel_l2(eta.grad, de_r),
                }
                if use_init:
                    errs["ht"] = _rel_l2(ht_rsi, ht_ref)
                    errs["hbt"] = _rel_l2(hbt_rsi, hbt_ref)
                    errs["dh0"] = _rel_l2(h0.grad, inp_d[6].grad)
                    errs["dhb0"] = _rel_l2(hb0.grad, inp_d[7].grad)

                case_max = max(errs.values())
                if dt == torch.float32:
                    fp32_max_err = max(fp32_max_err, case_max)
                    # Measure upstream FP32 TF32 backward truncation error and T % 16 != 0 tail bug
                    qu = q.detach().clone().requires_grad_(True)
                    ku = k.detach().clone().requires_grad_(True)
                    vu = v.detach().clone().requires_grad_(True)
                    wu = w.detach().clone().requires_grad_(True)
                    bu = b.detach().clone().requires_grad_(True)
                    eu = eta.detach().clone().requires_grad_(True)
                    h0u = h0.detach().clone().requires_grad_(True) if use_init else None
                    hb0u = hb0.detach().clone().requires_grad_(True) if use_init else None
                    ou, htu, hbtu = chunk_ttt_linear(
                        qu, ku, vu, wu, bu, eu, initial_state=h0u, initial_state_bias=hb0u, output_final_state=use_init
                    )
                    if use_init and T_val == 37:
                        upstream_tail_hbt_err = max(upstream_tail_hbt_err, _rel_l2(hbtu, hbt_ref))
                    try:
                        if use_init:
                            torch.autograd.backward([ou, htu, hbtu], [do, dht, dhbt])
                        else:
                            ou.backward(do)
                        up_bwd_max = max(
                            _rel_l2(qu.grad, dq_r),
                            _rel_l2(ku.grad, dk_r),
                            _rel_l2(vu.grad, dv_r),
                        )
                        if T_val == 128:
                            upstream_fp32_max_err = max(upstream_fp32_max_err, up_bwd_max)
                        errs["upstream_bwd_max_rel_l2"] = up_bwd_max
                    except Exception as e:
                        errs["upstream_bwd_crashed"] = f"{type(e).__name__}: {e}"
                else:
                    bf16_max_err = max(bf16_max_err, case_max)

                oracle_cases.append(
                    {
                        "K": K,
                        "T": T_val,
                        "use_init": use_init,
                        "dtype": dt_name,
                        "max_rel_l2": case_max,
                        "errors": errs,
                    }
                )

    assert fp32_max_err < 1e-5, f"FP32 max_rel_l2 {fp32_max_err} >= 1e-5"
    assert bf16_max_err < 1e-2, f"BF16 max_rel_l2 {bf16_max_err} >= 1e-2"

    # 3 Adversarial Mutation Modes
    torch.manual_seed(99)
    qm = torch.randn(2, 128, 4, 64, device=device, dtype=torch.float32, requires_grad=True)
    km = torch.randn(2, 128, 4, 64, device=device, dtype=torch.float32, requires_grad=True)
    vm = torch.randn(2, 128, 4, 64, device=device, dtype=torch.float32, requires_grad=True)
    wm = (torch.ones(4, 64, device=device, dtype=torch.float32) + 0.2 * torch.randn(4, 64, device=device, dtype=torch.float32)).requires_grad_(True)
    bm = (0.2 * torch.randn(4, 64, device=device, dtype=torch.float32)).requires_grad_(True)
    em = (torch.rand(2, 128, 4, 1, device=device, dtype=torch.float32) * 0.05).requires_grad_(True)
    h0m = (torch.randn(2, 4, 64, 64, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
    hb0m = (torch.randn(2, 4, 1, 64, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
    dom = torch.randn_like(qm)
    o_mref, _, _, inp_m = ttt_linear_fp64_oracle(qm, km, vm, wm, bm, em, initial_state=h0m, initial_state_bias=hb0m)
    o_mref.backward(dom.double())

    mutations = []
    for m_id, m_name in [
        (1, "M1_drop_self_adjoint_ln2_second_deriv"),
        (2, "M2_drop_dual_intra_chunk_attn_update"),
        (3, "M3_drop_fused_output_ln_residual"),
    ]:
        q_c = qm.detach().clone().requires_grad_(True)
        k_c = km.detach().clone().requires_grad_(True)
        v_c = vm.detach().clone().requires_grad_(True)
        w_c = wm.detach().clone().requires_grad_(True)
        b_c = bm.detach().clone().requires_grad_(True)
        e_c = em.detach().clone().requires_grad_(True)
        h0_c = h0m.detach().clone().requires_grad_(True)
        hb0_c = hb0m.detach().clone().requires_grad_(True)
        o_mut, _, _ = chunk_ttt_linear_rsi(
            q_c, k_c, v_c, w_c, b_c, e_c, initial_state=h0_c, initial_state_bias=hb0_c, mutation_mode=m_id
        )
        o_mut.backward(dom)
        m_err = max(
            _rel_l2(o_mut, o_mref),
            _rel_l2(q_c.grad, inp_m[0].grad),
            _rel_l2(k_c.grad, inp_m[1].grad),
            _rel_l2(v_c.grad, inp_m[2].grad),
            _rel_l2(w_c.grad, inp_m[3].grad),
            _rel_l2(b_c.grad, inp_m[4].grad),
        )
        killed = m_err > 5e-2
        assert killed, f"Mutation {m_name} survived with err={m_err}"
        mutations.append({"id": m_id, "name": m_name, "max_rel_l2": m_err, "killed": bool(killed)})

    # 1B. SM80 Cubin Resource Audit
    upstream_cubin = _audit_upstream_ttt_cubins(device)
    rsi_cubin_prelim = audit_ttt_rsi_cubin_resources()

    # 1C. 8-Shape Training Kernel Matrix
    shapes = [
        (2, 512, 16, 64, 64),
        (2, 1024, 16, 64, 64),
        (2, 2048, 16, 64, 64),
        (4, 2048, 16, 64, 64),
        (2, 512, 16, 128, 128),
        (2, 1024, 16, 128, 128),
        (2, 2048, 16, 128, 128),
        (4, 1024, 16, 128, 128),
    ]
    if smoke:
        shapes = [shapes[1], shapes[2], shapes[5], shapes[6]]
    warmup = 3 if smoke else 5
    iters = 8 if smoke else 20

    kernel_matrix = []
    for B, T, H, K, V in shapes:
        q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
        w = torch.ones(H, V, device=device, dtype=torch.bfloat16)
        b = torch.zeros(H, V, device=device, dtype=torch.bfloat16)
        eta = torch.rand(B, T, H, 1, device=device, dtype=torch.bfloat16) * 0.05
        do = torch.randn_like(v)

        up_fwd_ms = _time_cuda_ms(lambda: chunk_ttt_linear(q, k, v, w, b, eta), warmup=warmup, iters=iters)

        def _up_tot():
            qc = q.detach().requires_grad_(True)
            kc = k.detach().requires_grad_(True)
            vc = v.detach().requires_grad_(True)
            wc = w.detach().requires_grad_(True)
            bc = b.detach().requires_grad_(True)
            ec = eta.detach().requires_grad_(True)
            oc, _, _ = chunk_ttt_linear(qc, kc, vc, wc, bc, ec)
            oc.backward(do)

        up_tot_ms = _time_cuda_ms(_up_tot, warmup=warmup, iters=iters)
        up_bwd_ms = max(up_tot_ms - up_fwd_ms, 1e-4)

        fc_fwd_ms = _time_cuda_ms(lambda: fused_chunk_ttt_linear(q, k, v, w, b, eta), warmup=warmup, iters=iters)

        def _fc_tot():
            qc = q.detach().requires_grad_(True)
            kc = k.detach().requires_grad_(True)
            vc = v.detach().requires_grad_(True)
            wc = w.detach().requires_grad_(True)
            bc = b.detach().requires_grad_(True)
            ec = eta.detach().requires_grad_(True)
            oc, _, _ = fused_chunk_ttt_linear(qc, kc, vc, wc, bc, ec)
            oc.backward(do)

        fc_tot_ms = _time_cuda_ms(_fc_tot, warmup=warmup, iters=max(4, iters // 2))
        fc_bwd_ms = max(fc_tot_ms - fc_fwd_ms, 1e-4)

        rsi_fwd_ms = _time_cuda_ms(lambda: chunk_ttt_linear_rsi(q, k, v, w, b, eta), warmup=warmup, iters=iters)

        def _rsi_tot():
            qc = q.detach().requires_grad_(True)
            kc = k.detach().requires_grad_(True)
            vc = v.detach().requires_grad_(True)
            wc = w.detach().requires_grad_(True)
            bc = b.detach().requires_grad_(True)
            ec = eta.detach().requires_grad_(True)
            oc, _, _ = chunk_ttt_linear_rsi(qc, kc, vc, wc, bc, ec)
            oc.backward(do)

        rsi_tot_ms = _time_cuda_ms(_rsi_tot, warmup=warmup, iters=iters)
        rsi_bwd_ms = max(rsi_tot_ms - rsi_fwd_ms, 1e-4)

        fwd_flops, bwd_flops = _ttt_training_flops(B, T, H, K, V)
        tot_flops = fwd_flops + bwd_flops

        kernel_matrix.append(
            {
                "shape": f"B={B},T={T},H={H},K={K},V={V}",
                "B": B,
                "T": T,
                "H": H,
                "K": K,
                "V": V,
                "upstream_chunk_fwd_ms": up_fwd_ms,
                "upstream_chunk_bwd_ms": up_bwd_ms,
                "upstream_chunk_total_ms": up_tot_ms,
                "upstream_fused_fwd_ms": fc_fwd_ms,
                "upstream_fused_bwd_ms": fc_bwd_ms,
                "upstream_fused_total_ms": fc_tot_ms,
                "rsi_fwd_ms": rsi_fwd_ms,
                "rsi_bwd_ms": rsi_bwd_ms,
                "rsi_total_ms": rsi_tot_ms,
                "fwd_speedup_vs_chunk": up_fwd_ms / rsi_fwd_ms,
                "bwd_speedup_vs_chunk": up_bwd_ms / rsi_bwd_ms,
                "total_speedup_vs_chunk": up_tot_ms / rsi_tot_ms,
                "total_speedup_vs_fused": fc_tot_ms / rsi_tot_ms,
                "upstream_chunk_tflops": (tot_flops / (up_tot_ms * 1e-3)) / 1e12,
                "rsi_tflops": (tot_flops / (rsi_tot_ms * 1e-3)) / 1e12,
            }
        )

    rsi_cubin = audit_ttt_rsi_cubin_resources()
    for k_name, stats in rsi_cubin.items():
        assert stats["spilling_specs"] == 0, f"RSI kernel {k_name} spilled: {stats}"
        assert stats["over168_specs"] == 0, f"RSI kernel {k_name} exceeded 168 regs: {stats}"

    return {
        "oracle": {
            "fp32_max_rel_l2": fp32_max_err,
            "bf16_max_rel_l2": bf16_max_err,
            "upstream_fp32_bwd_tf32_rel_l2": upstream_fp32_max_err,
            "fp32_accuracy_improvement_x": upstream_fp32_max_err / max(fp32_max_err, 1e-12),
            "upstream_tail_padding_hbt_bug_rel_l2": upstream_tail_hbt_err,
            "cases": oracle_cases,
        },
        "mutations": {
            "total": len(mutations),
            "killed": sum(1 for m in mutations if m["killed"]),
            "kill_rate_pct": 100.0,
            "details": mutations,
        },
        "cubin_audit": {
            "upstream": upstream_cubin,
            "rsi": rsi_cubin,
        },
        "kernel_matrix": kernel_matrix,
    }


def run_track2_decode_adaptation(device: torch.device, smoke: bool = False) -> dict:
    from fla.ops.ttt import chunk_ttt_linear, fused_chunk_ttt_linear

    decode_shapes = [
        (4, 1, 16, 64),
        (4, 4, 16, 64),
        (4, 16, 16, 64),
        (8, 16, 16, 64),
        (4, 1, 16, 128),
        (4, 4, 16, 128),
        (4, 16, 16, 128),
        (8, 16, 16, 128),
    ]
    if smoke:
        decode_shapes = [decode_shapes[0], decode_shapes[2], decode_shapes[4], decode_shapes[6]]
    warmup = 4 if smoke else 6
    iters = 15 if smoke else 35

    rows = []
    for B, T_q, H, K in decode_shapes:
        torch.manual_seed(200 + B + T_q + K)
        q = torch.randn(B, T_q, H, K, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, T_q, H, K, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, T_q, H, K, device=device, dtype=torch.bfloat16)
        w = torch.ones(H, K, device=device, dtype=torch.bfloat16)
        b = torch.zeros(H, K, device=device, dtype=torch.bfloat16)
        eta = torch.rand(B, T_q, H, 1, device=device, dtype=torch.bfloat16) * 0.05
        h0 = torch.randn(B, H, K, K, device=device, dtype=torch.float32) * 0.1
        hb0 = torch.randn(B, H, 1, K, device=device, dtype=torch.float32) * 0.1

        o_ref, ht_ref, hbt_ref, _ = ttt_linear_fp64_oracle(
            q, k, v, w, b, eta, initial_state=h0, initial_state_bias=hb0
        )
        o_rsi, ht_rsi, hbt_rsi = ttt_decode_adapt_rsi(q, k, v, w, b, eta, h0, hb0)
        o_up, ht_up, hbt_up = chunk_ttt_linear(
            q, k, v, w, b, eta, initial_state=h0, initial_state_bias=hb0, output_final_state=True
        )

        up_chunk_ms = _time_cuda_ms(
            lambda: chunk_ttt_linear(q, k, v, w, b, eta, initial_state=h0, initial_state_bias=hb0, output_final_state=True),
            warmup=warmup,
            iters=iters,
        )
        up_fused_ms = _time_cuda_ms(
            lambda: fused_chunk_ttt_linear(
                q, k, v, w, b, eta, initial_state=h0, initial_state_bias=hb0, output_final_state=True
            ),
            warmup=warmup,
            iters=iters,
        )
        rsi_ms = _time_cuda_ms(
            lambda: ttt_decode_adapt_rsi(q, k, v, w, b, eta, h0, hb0),
            warmup=warmup,
            iters=iters,
        )

        rows.append(
            {
                "shape": f"B={B},Tq={T_q},H={H},K={K}",
                "B": B,
                "T_q": T_q,
                "H": H,
                "K": K,
                "upstream_chunk_ms": up_chunk_ms,
                "upstream_fused_ms": up_fused_ms,
                "rsi_decode_ms": rsi_ms,
                "speedup_vs_chunk": up_chunk_ms / rsi_ms,
                "speedup_vs_fused": up_fused_ms / rsi_ms,
                "rsi_o_rel_l2_vs_fp64": _rel_l2(o_rsi, o_ref),
                "rsi_ht_rel_l2_vs_fp64": _rel_l2(ht_rsi, ht_ref),
                "rsi_hbt_rel_l2_vs_fp64": _rel_l2(hbt_rsi, hbt_ref),
                "upstream_hbt_rel_l2_vs_fp64": _rel_l2(hbt_up, hbt_ref),
            }
        )
    return {"decode_matrix": rows}


def run_track3_layer_ablation(device: torch.device, smoke: bool = False) -> dict:
    from fla.ops.ttt import chunk_ttt_linear

    configs = [
        ("D=1024,H=16,K=64", 2, 2048, 1024, 16),
        ("D=2048,H=16,K=128", 2, 2048, 2048, 16),
    ]
    warmup = 3 if smoke else 5
    iters = 8 if smoke else 18

    results = []
    for label, B, T, D, H in configs:
        torch.manual_seed(300 + D)
        base_blk = TTTLinearBlockBaseline(D, H).to(device=device, dtype=torch.bfloat16)
        rsi_blk = TTTLinearBlockRSI.from_baseline(base_blk)
        sep_projs = (base_blk.q_proj, base_blk.k_proj, base_blk.v_proj, base_blk.eta_proj)

        x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
        dy = torch.randn_like(x)

        # Verify numerical parity between A0 and A4
        x0 = x.detach().clone().requires_grad_(True)
        y0 = base_blk(x0, chunk_ttt_linear)
        y0.backward(dy)
        x4 = x.detach().clone().requires_grad_(True)
        y4 = rsi_blk(x4, stage=4)
        y4.backward(dy)
        fwd_rel = _rel_l2(y4, y0)
        bwd_rel = _rel_l2(x4.grad, x0.grad)
        assert fwd_rel < 2e-2 and bwd_rel < 2e-2, f"Block parity failed: fwd={fwd_rel}, bwd={bwd_rel}"

        def _bench_stage(st: int):
            if st == 0:
                f_ms = _time_cuda_ms(lambda: base_blk(x, chunk_ttt_linear), warmup=warmup, iters=iters)

                def _tot():
                    base_blk.zero_grad(set_to_none=True)
                    xi = x.detach().requires_grad_(True)
                    yi = base_blk(xi, chunk_ttt_linear)
                    yi.backward(dy)

                t_ms = _time_cuda_ms(_tot, warmup=warmup, iters=iters)
            else:
                f_ms = _time_cuda_ms(
                    lambda: rsi_blk(x, stage=st, sep_projs=sep_projs), warmup=warmup, iters=iters
                )

                def _tot():
                    rsi_blk.zero_grad(set_to_none=True)
                    if st < 3:
                        base_blk.zero_grad(set_to_none=True)
                    xi = x.detach().requires_grad_(True)
                    yi = rsi_blk(xi, stage=st, sep_projs=sep_projs)
                    yi.backward(dy)

                t_ms = _time_cuda_ms(_tot, warmup=warmup, iters=iters)
            return f_ms, max(t_ms - f_ms, 1e-4), t_ms

        stage_names = [
            (0, "A0_upstream_ttt_block_baseline"),
            (1, "A1_zero_spill_ttt_kernel_no_bwd_replay"),
            (2, "A2_plus_fused_in_kernel_output_ln_residual"),
            (3, "A3_plus_fused_qkve_single_gemm"),
            (4, "A4_plus_persistent_2d_triton_rmsnorm"),
        ]
        stages_out = []
        a0_tot = None
        for st_id, st_name in stage_names:
            f_ms, b_ms, t_ms = _bench_stage(st_id)
            if st_id == 0:
                a0_tot = t_ms
            tok_s = (B * T) / (t_ms * 1e-3)
            stages_out.append(
                {
                    "stage_id": f"A{st_id}",
                    "stage_name": st_name,
                    "fwd_ms": f_ms,
                    "bwd_ms": b_ms,
                    "total_ms": t_ms,
                    "speedup_vs_a0": a0_tot / t_ms,
                    "tokens_per_sec": tok_s,
                }
            )
        results.append(
            {
                "config": label,
                "B": B,
                "T": T,
                "D": D,
                "H": H,
                "K": D // H,
                "a4_vs_a0_fwd_rel_l2": fwd_rel,
                "a4_vs_a0_bwd_rel_l2": bwd_rel,
                "stages": stages_out,
            }
        )
    return {"configs": results}


def _cp8_worker_main(rank: int, world_size: int, out_path: str, smoke: bool):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29573"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    from fla.ops.ttt import chunk_ttt_linear

    cp_shapes = [
        (2, 16384, 16, 64),
        (2, 32768, 16, 64),
        (2, 16384, 16, 128),
    ]
    if smoke:
        cp_shapes = [cp_shapes[0], cp_shapes[1]]
    warmup = 3 if smoke else 5
    iters = 6 if smoke else 14

    rows = []
    for B, T_total, H, K in cp_shapes:
        T_local = T_total // world_size
        torch.manual_seed(400 + T_total + K)
        q_global = torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16)
        k_global = torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16)
        v_global = torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16)
        w = torch.ones(H, K, device=device, dtype=torch.bfloat16)
        b = torch.zeros(H, K, device=device, dtype=torch.bfloat16)
        eta_global = torch.rand(B, T_total, H, 1, device=device, dtype=torch.bfloat16) * 0.05
        do_global = torch.randn_like(v_global)

        s0, s1 = rank * T_local, (rank + 1) * T_local
        q_loc = q_global[:, s0:s1].contiguous().requires_grad_(True)
        k_loc = k_global[:, s0:s1].contiguous().requires_grad_(True)
        v_loc = v_global[:, s0:s1].contiguous().requires_grad_(True)
        w_u = w.detach().clone().requires_grad_(True)
        b_u = b.detach().clone().requires_grad_(True)
        eta_loc = eta_global[:, s0:s1].contiguous().requires_grad_(True)
        do_loc = do_global[:, s0:s1].contiguous()

        o_cp = ttt_cp8_ulysses_rsi(q_loc, k_loc, v_loc, w_u, b_u, eta_loc, group, rank, world_size)
        o_cp.backward(do_loc)

        # Verify exact single-GPU bitwise parity on rank 0
        if rank == 0:
            q_s = q_global.detach().clone().requires_grad_(True)
            k_s = k_global.detach().clone().requires_grad_(True)
            v_s = v_global.detach().clone().requires_grad_(True)
            w_s = w.detach().clone().requires_grad_(True)
            b_s = b.detach().clone().requires_grad_(True)
            e_s = eta_global.detach().clone().requires_grad_(True)
            o_s, _, _ = chunk_ttt_linear_rsi(q_s, k_s, v_s, w_s, b_s, e_s)
            o_s.backward(do_global)
            err_o = _rel_l2(o_cp, o_s[:, s0:s1])
            err_dq = _rel_l2(q_loc.grad, q_s.grad[:, s0:s1])
            err_dk = _rel_l2(k_loc.grad, k_s.grad[:, s0:s1])
            err_dv = _rel_l2(v_loc.grad, v_s.grad[:, s0:s1])

            def _single_gpu_step():
                qs = q_global.detach().requires_grad_(True)
                ks = k_global.detach().requires_grad_(True)
                vs = v_global.detach().requires_grad_(True)
                ws = w.detach().requires_grad_(True)
                bs = b.detach().requires_grad_(True)
                es = eta_global.detach().requires_grad_(True)
                os_, _, _ = chunk_ttt_linear_rsi(qs, ks, vs, ws, bs, es)
                os_.backward(do_global)

            single_rsi_ms = _time_cuda_ms(_single_gpu_step, warmup=warmup, iters=iters)

            def _single_up_step():
                qs = q_global.detach().requires_grad_(True)
                ks = k_global.detach().requires_grad_(True)
                vs = v_global.detach().requires_grad_(True)
                ws = w.detach().requires_grad_(True)
                bs = b.detach().requires_grad_(True)
                es = eta_global.detach().requires_grad_(True)
                os_, _, _ = chunk_ttt_linear(qs, ks, vs, ws, bs, es)
                os_.backward(do_global)

            single_up_ms = _time_cuda_ms(_single_up_step, warmup=warmup, iters=iters)
        else:
            err_o = err_dq = err_dk = err_dv = 0.0
            single_rsi_ms = single_up_ms = 0.0

        dist.barrier()

        def _ring_step():
            ql = q_loc.detach().requires_grad_(True)
            kl = k_loc.detach().requires_grad_(True)
            vl = v_loc.detach().requires_grad_(True)
            wl = w.detach().requires_grad_(True)
            bl = b.detach().requires_grad_(True)
            el = eta_loc.detach().requires_grad_(True)
            ol = ttt_cp8_ring_baseline(
                ql, kl, vl, wl, bl, el, group, rank, world_size, upstream_op=chunk_ttt_linear
            )
            ol.backward(do_loc)

        ring_ms = _time_cuda_ms(_ring_step, warmup=warmup, iters=iters)
        dist.barrier()

        def _ulysses_step():
            ql = q_loc.detach().requires_grad_(True)
            kl = k_loc.detach().requires_grad_(True)
            vl = v_loc.detach().requires_grad_(True)
            wl = w.detach().requires_grad_(True)
            bl = b.detach().requires_grad_(True)
            el = eta_loc.detach().requires_grad_(True)
            ol = ttt_cp8_ulysses_rsi(ql, kl, vl, wl, bl, el, group, rank, world_size)
            ol.backward(do_loc)

        ulysses_ms = _time_cuda_ms(_ulysses_step, warmup=warmup, iters=iters)
        dist.barrier()

        if rank == 0:
            rows.append(
                {
                    "shape": f"B={B},T={T_total},H={H},K={K}",
                    "B": B,
                    "T_total": T_total,
                    "T_local": T_local,
                    "H": H,
                    "K": K,
                    "single_gpu_upstream_ms": single_up_ms,
                    "single_gpu_rsi_ms": single_rsi_ms,
                    "cp8_ring_baseline_ms": ring_ms,
                    "cp8_ulysses_rsi_ms": ulysses_ms,
                    "speedup_vs_ring_cp8": ring_ms / ulysses_ms,
                    "speedup_vs_single_upstream": single_up_ms / ulysses_ms,
                    "scaling_vs_single_rsi": single_rsi_ms / ulysses_ms,
                    "parallel_efficiency_pct": 100.0 * (single_rsi_ms / ulysses_ms) / world_size,
                    "rel_l2_o": err_o,
                    "rel_l2_dq": err_dq,
                    "rel_l2_dk": err_dk,
                    "rel_l2_dv": err_dv,
                }
            )

    if rank == 0:
        Path(out_path).write_text(json.dumps({"cp8_rows": rows}, indent=2), encoding="utf-8")
    dist.destroy_process_group()


def run_track4_cp8(smoke: bool = False) -> dict:
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 8, f"Track 4 requires 8 GPUs, found {n_gpus}"
    tmp_out = Path("/tmp/_ttt_cp8_result.json")
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


def _spawn_cp8_workers(out_path: str, smoke: bool):
    import torch.multiprocessing as mp

    world_size = 8
    mp.spawn(_cp8_worker_main, args=(world_size, out_path, smoke), nprocs=world_size, join=True)


class _TinyTTTLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, use_rsi: bool):
        super().__init__()
        self.use_rsi = use_rsi
        self.embed = nn.Embedding(vocab_size, d_model)
        if use_rsi:
            self.layers = nn.ModuleList([TTTLinearBlockRSI(d_model, num_heads) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([TTTLinearBlockBaseline(d_model, num_heads) for _ in range(num_layers)])
        self.final_norm_weight = nn.Parameter(torch.ones(d_model))
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor, upstream_op=None) -> torch.Tensor:
        x = self.embed(idx)
        for layer in self.layers:
            if self.use_rsi:
                x = layer(x, stage=4)
            else:
                x = layer(x, upstream_op)
        xf = x.float()
        x_norm = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(x.dtype) * self.final_norm_weight
        return self.head(x_norm)


def run_track5_autoresearch(device: torch.device, smoke: bool = False) -> dict:
    from fla.ops.ttt import chunk_ttt_linear

    budget_s = 6.0 if smoke else 12.0
    vocab_size = 256
    B, T = 4, 512
    n_layers = 2
    idx_t = torch.arange(T + 1, device=device)

    def _sample_batch(gen: torch.Generator) -> torch.Tensor:
        base = torch.randint(0, vocab_size, (B, 4), device=device, generator=gen)
        seq = (base[:, idx_t % 4] * 17 + (idx_t % 4).unsqueeze(0) * 53) % vocab_size
        seq[:, 1:] = (seq[:, 1:] ^ (seq[:, :-1] * 3)) % vocab_size
        return seq

    configs = [
        ("D=1024,H=16,K=64", 1024, 16),
        ("D=2048,H=16,K=128", 2048, 16),
    ]

    rows = []
    for label, D, H in configs:
        g_val = torch.Generator(device=device).manual_seed(5000 + D)
        val_batches = [_sample_batch(g_val) for _ in range(8)]

        @torch.no_grad()
        def _eval_bpb(m: nn.Module, is_r: bool) -> float:
            tot_loss = 0.0
            for vb in val_batches:
                logits = m(vb[:, :-1], None if is_r else chunk_ttt_linear)
                tot_loss += float(
                    F.cross_entropy(logits.float().reshape(-1, vocab_size), vb[:, 1:].reshape(-1)).item()
                )
            return (tot_loss / len(val_batches)) / math.log(2.0)

        torch.manual_seed(5100 + D)
        init_base = _TinyTTTLM(vocab_size, D, H, n_layers, use_rsi=False).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            nn.init.normal_(init_base.embed.weight, std=0.02)
            for lb in init_base.layers:
                for p in [lb.q_proj.weight, lb.k_proj.weight, lb.v_proj.weight, lb.eta_proj.weight, lb.o_proj.weight]:
                    nn.init.normal_(p, std=0.02)
        init_rsi = _TinyTTTLM(vocab_size, D, H, n_layers, use_rsi=True).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            init_rsi.embed.weight.copy_(init_base.embed.weight)
            init_rsi.final_norm_weight.copy_(init_base.final_norm_weight)
            for lb, lr in zip(init_base.layers, init_rsi.layers):
                copied = TTTLinearBlockRSI.from_baseline(lb)
                lr.load_state_dict(copied.state_dict())

        # Warmup compilation before wall-clock timer
        g_warm = torch.Generator(device=device).manual_seed(5200 + D)
        wb = _sample_batch(g_warm)
        for m_warm, is_r in [(init_base, False), (init_rsi, True)]:
            logits_w = m_warm(wb[:, :-1], None if is_r else chunk_ttt_linear)
            loss_w = F.cross_entropy(logits_w.float().reshape(-1, vocab_size), wb[:, 1:].reshape(-1))
            loss_w.backward()
            m_warm.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        base_lr = 1e-3 if D == 1024 else 7e-4

        def _run_loop(model: nn.Module, is_rsi: bool):
            opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
            g_train = torch.Generator(device=device).manual_seed(5400 + D)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            steps = 0
            while True:
                batch = _sample_batch(g_train)
                opt.zero_grad(set_to_none=True)
                logits = model(batch[:, :-1], None if is_rsi else chunk_ttt_linear)
                loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps += 1
                if steps % 5 == 0:
                    torch.cuda.synchronize()
                    if (time.perf_counter() - t0) >= budget_s:
                        break
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            val_bpb = _eval_bpb(model, is_rsi)
            return steps, elapsed, val_bpb

        base_steps, base_el, base_bpb = _run_loop(init_base, False)
        rsi_steps, rsi_el, rsi_bpb = _run_loop(init_rsi, True)
        rows.append(
            {
                "config": label,
                "D": D,
                "H": H,
                "K": D // H,
                "budget_s": budget_s,
                "baseline_steps": base_steps,
                "rsi_steps": rsi_steps,
                "step_ratio": rsi_steps / max(base_steps, 1),
                "baseline_tokens_per_sec": (base_steps * B * T) / base_el,
                "rsi_tokens_per_sec": (rsi_steps * B * T) / rsi_el,
                "baseline_val_bpb": base_bpb,
                "rsi_val_bpb": rsi_bpb,
                "val_bpb_improvement": base_bpb - rsi_bpb,
            }
        )
    return {"autoresearch_rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--cp8-worker", action="store_true")
    parser.add_argument("--cp8-out", type=str, default="")
    args = parser.parse_args()

    if args.cp8_worker:
        _spawn_cp8_workers(args.cp8_out, args.smoke)
        return

    harness_info = _verify_frozen_harness()
    device = torch.device("cuda:0")
    t_start = time.time()

    print("[E153/v144] Running Track 1 (FP64 Oracle, Mutations, Cubin Audit, 8-Shape Training Matrix)...")
    t1 = run_track1_kernels_and_oracle(device, smoke=args.smoke)
    print(
        f"  Track 1 done: FP32 max_rel_l2={t1['oracle']['fp32_max_rel_l2']:.3e}, "
        f"BF16 max_rel_l2={t1['oracle']['bf16_max_rel_l2']:.3e}, "
        f"upstream spills={t1['cubin_audit']['upstream']['summary']['total_spills']} vs RSI spills=0"
    )

    print("[E153/v144] Running Track 2 (Single-Pass Online Test-Time Adaptation Decoding)...")
    t2 = run_track2_decode_adaptation(device, smoke=args.smoke)
    print(f"  Track 2 done: max speedup vs chunk={max(r['speedup_vs_chunk'] for r in t2['decode_matrix']):.2f}x")

    print("[E153/v144] Running Track 3 (TTTLinearBlockRSI Stages A0 -> A4)...")
    t3 = run_track3_layer_ablation(device, smoke=args.smoke)
    t1["cubin_audit"]["rsi"] = audit_ttt_rsi_cubin_resources()
    for c in t3["configs"]:
        print(f"  Track 3 {c['config']}: A4 speedup={c['stages'][-1]['speedup_vs_a0']:.2f}x")

    print("[E153/v144] Running Track 4 (8-GPU NVSwitch TTT-CP = 8)...")
    t4 = run_track4_cp8(smoke=args.smoke)
    for r in t4["cp8_rows"]:
        print(f"  Track 4 {r['shape']}: Ulysses={r['cp8_ulysses_rsi_ms']:.2f}ms vs Ring={r['cp8_ring_baseline_ms']:.2f}ms ({r['speedup_vs_ring_cp8']:.2f}x)")

    print("[E153/v144] Running Track 5 (Fixed-Wall-Clock val_bpb Autoresearch)...")
    t5 = run_track5_autoresearch(device, smoke=args.smoke)
    for r in t5["autoresearch_rows"]:
        print(
            f"  Track 5 {r['config']}: steps {r['baseline_steps']} -> {r['rsi_steps']} ({r['step_ratio']:.2f}x), "
            f"val_bpb {r['baseline_val_bpb']:.4f} -> {r['rsi_val_bpb']:.4f} (delta={-r['val_bpb_improvement']:+.4f})"
        )

    payload = {
        "experiment_id": "E153",
        "round": "Round-27 RSI",
        "version": "v144",
        "paper": {
            "title": "Learning to (Learn at Test Time): RNNs with Expressive Hidden States",
            "authors": "Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin",
            "venue": "ICML 2024 / JMLR 2025",
            "arxiv": "2407.04620",
            "upstream_module": "fla/ops/ttt (chunk.py, fused_chunk.py, naive.py)",
        },
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
        },
        "frozen_harness": harness_info,
        "smoke": bool(args.smoke),
        "elapsed_s": round(time.time() - t_start, 2),
        "track1": t1,
        "track2": t2,
        "track3": t3,
        "track4": t4,
        "track5": t5,
    }

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[E153/v144] Wrote {out_p}")


if __name__ == "__main__":
    main()
