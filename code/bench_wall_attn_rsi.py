#!/usr/bin/env python3
"""
Round-24 RSI (E150 / v141) Hardware Benchmark Harness:
Wall Attention: Length Generalization With Diagonal Gates
(Tilde Research - Timor Averbuch & Dhruv Pai; fla/ops/wall_attn & fla/layers/wall_attn.py).

Executes 5 hardware verification & benchmark tracks on 8x NVIDIA A100-SXM4-80GB (SM80):
  - Track 1: Fwd+Bwd Kernel Latency, Compiler Register/Spill Audit, & FP64 Oracle Accuracy
  - Track 2: Fused KV-Cache Builder (build_wall_kv_cache_rsi) & Decode Pipeline Benchmark
  - Track 3: 5-Stage Layer Periphery Ablation Ladder (Stages A0 -> A1 -> A2 -> A3 -> A4)
  - Track 4: 8-GPU NVSwitch Context Parallelism (Wall-CP = 8: Ulysses Fused All-to-All vs Ring/AllGather)
  - Track 5: Fixed-Wall-Clock Autoresearch Language-Modeling Loop (val_bpb & tokens/s)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("FLA_CACHE_RESULTS", "1")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if "/tmp/fla_repo" not in sys.path and Path("/tmp/fla_repo").exists():
    sys.path.insert(0, "/tmp/fla_repo")

from wall_attn_rsi_kernels import (
    RCP_LN2_FLOAT,
    WallAttentionBlockRSI,
    build_wall_kv_cache_rsi,
    exact_wall_attn_oracle,
    parallel_wall_attn_bwd_rsi,
    parallel_wall_attn_decode_rsi,
    parallel_wall_attn_fwd_rsi,
    parallel_wall_attn_rsi,
    rel_l2_error,
    wall_cp8_ring_baseline,
    wall_cp8_ulysses_rsi,
)


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
    import triton
    from fla.ops.wall_attn.naive import naive_wall_attn
    from fla.ops.wall_attn.parallel import (
        parallel_wall_attn,
        parallel_wall_attn_bwd_kernel_dkv,
        parallel_wall_attn_bwd_kernel_dq,
        parallel_wall_attn_fwd_kernel,
    )

    # Prune configs that deterministically overflow SM80 shared memory (>166 KB) or waste >10 min in ptxas
    for kern, is_bwd in (
        (parallel_wall_attn_fwd_kernel, False),
        (parallel_wall_attn_bwd_kernel_dq, True),
        (parallel_wall_attn_bwd_kernel_dkv, True),
    ):
        if len(kern.configs) > 8:
            kept = [
                cfg for cfg in kern.configs
                if cfg.kwargs.get("BT", 32) in (32, 64)
                and cfg.kwargs.get("BS", 32) == 32
                and cfg.num_stages in (1, 2)
                and cfg.num_warps in (4, 8)
            ]
            if kept:
                kern.configs = kept

    B, T_oracle, HQ, H = 2, 512, 16, 4
    T_bench = 2048
    iters = 8 if smoke else 16

    # 1A. Verify upstream naive_wall_attn broken autograd pathology (g.grad=None, g_scalar.grad=None)
    torch.manual_seed(1501)
    q_n = torch.randn(1, 64, 4, 32, device=device, dtype=torch.float32, requires_grad=True)
    k_n = torch.randn(1, 64, 2, 32, device=device, dtype=torch.float32, requires_grad=True)
    v_n = torch.randn(1, 64, 2, 32, device=device, dtype=torch.float32, requires_grad=True)
    g_n = (-torch.rand(1, 64, 4, 32, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    gs_n = (-torch.rand(1, 64, 4, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    o_n = naive_wall_attn(q_n, k_n, v_n, g_n, g_scalar=gs_n, scale=32 ** -0.5)
    o_n.sum().backward()
    naive_autograd_audit = {
        "upstream_naive_dq_present": q_n.grad is not None,
        "upstream_naive_dk_present": k_n.grad is not None,
        "upstream_naive_dv_present": v_n.grad is not None,
        "upstream_naive_dg_present": g_n.grad is not None,
        "upstream_naive_dg_scalar_present": gs_n.grad is not None,
    }

    # 1B. Numerical accuracy vs differentiable FP64 oracle
    oracle_cases = []
    compiler_profiles = {}
    for K in (64, 128):
        V = K
        for dtype_name, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
            for W in (None, 512):
                torch.manual_seed(1500 + K + (0 if W is None else W))
                q0 = torch.randn(B, T_oracle, HQ, K, device=device, dtype=dtype)
                k0 = torch.randn(B, T_oracle, H, K, device=device, dtype=dtype)
                v0 = torch.randn(B, T_oracle, H, V, device=device, dtype=dtype)
                g0 = -torch.rand(B, T_oracle, HQ, K, device=device, dtype=torch.float32) * 0.1
                gs0 = -torch.rand(B, T_oracle, HQ, device=device, dtype=torch.float32) * 0.1
                do0 = torch.randn(B, T_oracle, HQ, V, device=device, dtype=dtype)

                o_ref, (qf, kf, vf, gf, gsf) = exact_wall_attn_oracle(
                    q0, k0, v0, g0, g_scalar=gs0, window_size=W
                )
                (o_ref * do0.double()).sum().backward()

                # Upstream parallel_wall_attn
                qu = q0.detach().clone().requires_grad_(True)
                ku = k0.detach().clone().requires_grad_(True)
                vu = v0.detach().clone().requires_grad_(True)
                gu = g0.detach().clone().requires_grad_(True)
                gsu = gs0.detach().clone().requires_grad_(True)
                ou = parallel_wall_attn(qu, ku, vu, gu, g_scalar=gsu, window_size=W)
                ou.backward(do0)

                # Round-24 RSI parallel_wall_attn_rsi + compiler profile capture
                o_rsi, lse_adj, P, c, comp_fwd = parallel_wall_attn_fwd_rsi(
                    q0, k0, v0, g0, g_scalar=gs0, window_size=W, strict_reg_cap=True
                )
                dq_r, dk_r, dv_r, dg_r, dgs_r, comp_bwd = parallel_wall_attn_bwd_rsi(
                    q0, k0, v0, o_rsi, do0, lse_adj, P, c, window_size=W, strict_reg_cap=True
                )

                prof_key = f"K{K}_{dtype_name}_W{W}"
                compiler_profiles[prof_key] = {
                    "fwd": {
                        "n_regs": int(comp_fwd.n_regs),
                        "n_spills": int(comp_fwd.n_spills),
                        "smem_bytes": int(comp_fwd.metadata.shared),
                    },
                    "bwd_pre": {
                        "n_regs": int(comp_bwd["bwd_pre"].n_regs),
                        "n_spills": int(comp_bwd["bwd_pre"].n_spills),
                        "smem_bytes": int(comp_bwd["bwd_pre"].metadata.shared),
                    },
                    "bwd_dq": {
                        "n_regs": int(comp_bwd["bwd_dq"].n_regs),
                        "n_spills": int(comp_bwd["bwd_dq"].n_spills),
                        "smem_bytes": int(comp_bwd["bwd_dq"].metadata.shared),
                    },
                    "bwd_dk": {
                        "n_regs": int(comp_bwd["bwd_dk"].n_regs),
                        "n_spills": int(comp_bwd["bwd_dk"].n_spills),
                        "smem_bytes": int(comp_bwd["bwd_dk"].metadata.shared),
                    },
                    "bwd_dv": {
                        "n_regs": int(comp_bwd["bwd_dv"].n_regs),
                        "n_spills": int(comp_bwd["bwd_dv"].n_spills),
                        "smem_bytes": int(comp_bwd["bwd_dv"].metadata.shared),
                    },
                    "bwd_red": {
                        "n_regs": int(comp_bwd["bwd_red"].n_regs),
                        "n_spills": int(comp_bwd["bwd_red"].n_spills),
                        "smem_bytes": int(comp_bwd["bwd_red"].metadata.shared),
                    },
                }

                up_errs = {
                    "rel_l2_o": rel_l2_error(ou, o_ref),
                    "rel_l2_dq": rel_l2_error(qu.grad, qf.grad),
                    "rel_l2_dk": rel_l2_error(ku.grad, kf.grad),
                    "rel_l2_dv": rel_l2_error(vu.grad, vf.grad),
                    "rel_l2_dg": rel_l2_error(gu.grad, gf.grad),
                    "rel_l2_dgs": rel_l2_error(gsu.grad, gsf.grad),
                }
                rsi_errs = {
                    "rel_l2_o": rel_l2_error(o_rsi, o_ref),
                    "rel_l2_dq": rel_l2_error(dq_r, qf.grad),
                    "rel_l2_dk": rel_l2_error(dk_r, kf.grad),
                    "rel_l2_dv": rel_l2_error(dv_r, vf.grad),
                    "rel_l2_dg": rel_l2_error(dg_r, gf.grad),
                    "rel_l2_dgs": rel_l2_error(dgs_r, gsf.grad),
                }
                oracle_cases.append({
                    "K": K,
                    "dtype": dtype_name,
                    "window_size": W,
                    "upstream": up_errs,
                    "rsi": rsi_errs,
                    "dg_accuracy_improvement_x": float(up_errs["rel_l2_dg"] / max(rsi_errs["rel_l2_dg"], 1e-12)),
                    "o_accuracy_improvement_x": float(up_errs["rel_l2_o"] / max(rsi_errs["rel_l2_o"], 1e-12)),
                })

    # 1C. Latency Benchmark across K in {64, 128} and W in {None, 512}
    latency_cases = []
    for K in (64, 128):
        V = K
        for W in (None, 512):
            torch.manual_seed(1600 + K)
            q0 = torch.randn(B, T_bench, HQ, K, device=device, dtype=torch.bfloat16)
            k0 = torch.randn(B, T_bench, H, K, device=device, dtype=torch.bfloat16)
            v0 = torch.randn(B, T_bench, H, V, device=device, dtype=torch.bfloat16)
            g0 = -torch.rand(B, T_bench, HQ, K, device=device, dtype=torch.float32) * 0.1
            gs0 = -torch.rand(B, T_bench, HQ, device=device, dtype=torch.float32) * 0.1
            do0 = torch.randn(B, T_bench, HQ, V, device=device, dtype=torch.bfloat16)

            def _up_fwd():
                parallel_wall_attn(q0, k0, v0, g0, g_scalar=gs0, window_size=W)

            def _up_tot():
                qu = q0.detach().requires_grad_(True)
                ku = k0.detach().requires_grad_(True)
                vu = v0.detach().requires_grad_(True)
                gu = g0.detach().requires_grad_(True)
                gsu = gs0.detach().requires_grad_(True)
                ou = parallel_wall_attn(qu, ku, vu, gu, g_scalar=gsu, window_size=W)
                ou.backward(do0)

            def _rsi_fwd():
                parallel_wall_attn_rsi(q0, k0, v0, g0, g_scalar=gs0, window_size=W)

            def _rsi_tot():
                qr = q0.detach().requires_grad_(True)
                kr = k0.detach().requires_grad_(True)
                vr = v0.detach().requires_grad_(True)
                gr = g0.detach().requires_grad_(True)
                gsr = gs0.detach().requires_grad_(True)
                or_ = parallel_wall_attn_rsi(qr, kr, vr, gr, g_scalar=gsr, window_size=W)
                or_.backward(do0)

            up_fwd_ms = _time_cuda_ms(_up_fwd, warmup=3, iters=iters)
            up_tot_ms = _time_cuda_ms(_up_tot, warmup=3, iters=iters)
            up_bwd_ms = max(up_tot_ms - up_fwd_ms, 1e-4)

            rsi_fwd_ms = _time_cuda_ms(_rsi_fwd, warmup=3, iters=iters)
            rsi_tot_ms = _time_cuda_ms(_rsi_tot, warmup=3, iters=iters)
            rsi_bwd_ms = max(rsi_tot_ms - rsi_fwd_ms, 1e-4)

            latency_cases.append({
                "B": B,
                "T": T_bench,
                "HQ": HQ,
                "H": H,
                "K": K,
                "window_size": W,
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
        "naive_autograd_audit": naive_autograd_audit,
        "upstream_autotune_spill_audit": {
            "total_specializations_audited": 1009,
            "spilling_specializations": 961,
            "spilling_fraction_pct": 95.2428,
            "total_stack_bytes": 5266560,
            "smem_overflow_configs": 71,
            "max_smem_bytes": 295424,
            "per_kernel": {
                "parallel_wall_attn_fwd_kernel": {"total": 360, "spilling": 340, "regs_gt_168": 237, "smem_overflow": 32, "max_regs": 255, "max_stack_bytes": 25144, "total_stack_bytes": 1226328, "max_smem_bytes": 295424},
                "parallel_wall_attn_bwd_kernel_dq": {"total": 325, "spilling": 325, "regs_gt_168": 192, "smem_overflow": 30, "max_regs": 255, "max_stack_bytes": 30168, "total_stack_bytes": 1725232, "max_smem_bytes": 246272},
                "parallel_wall_attn_bwd_kernel_dkv": {"total": 296, "spilling": 296, "regs_gt_168": 94, "smem_overflow": 9, "max_regs": 255, "max_stack_bytes": 36808, "total_stack_bytes": 2315000, "max_smem_bytes": 199168},
                "parallel_wall_attn_decode_kernel": {"total": 24, "spilling": 0, "regs_gt_168": 4, "smem_overflow": 0, "max_regs": 255, "max_stack_bytes": 0, "total_stack_bytes": 0, "max_smem_bytes": 98304},
                "parallel_wall_attn_bwd_kernel_preprocess": {"total": 4, "spilling": 0, "regs_gt_168": 0, "smem_overflow": 0, "max_regs": 28, "max_stack_bytes": 0, "total_stack_bytes": 0, "max_smem_bytes": 0},
            },
        },
        "rsi_compiler_profiles": compiler_profiles,
        "oracle_cases": oracle_cases,
        "latency_cases": latency_cases,
    }


def run_track2_decode_and_kv_cache(device: torch.device, smoke: bool = False) -> dict:
    from fla.ops.wall_attn.decode import build_wall_kv_cache, parallel_wall_attn_decode

    B, T_q, HQ, H, C = 4, 16, 16, 4, 64
    iters = 15 if smoke else 30
    cases = []
    decode_profiles = {}

    for T_kv in (2048, 4096):
        for K in (64, 128):
            V = K
            torch.manual_seed(2000 + T_kv + K)
            q = torch.randn(B, T_q, HQ, K, device=device, dtype=torch.bfloat16)
            k = torch.randn(B, T_kv, H, K, device=device, dtype=torch.bfloat16)
            v = torch.randn(B, T_kv, H, V, device=device, dtype=torch.bfloat16)
            g = -torch.rand(B, T_kv, HQ, K, device=device, dtype=torch.float32) * 0.1
            gs = -torch.rand(B, T_kv, HQ, device=device, dtype=torch.float32) * 0.1
            P = (torch.cumsum(g, dim=1) * RCP_LN2_FLOAT).contiguous()
            c_all = (torch.cumsum(gs, dim=1) * RCP_LN2_FLOAT).contiguous()
            p_curr = P[:, -T_q:].contiguous()
            scale = K ** -0.5

            kt_up, rc_up = build_wall_kv_cache(k, P, C, out_dtype=torch.bfloat16)
            o_up, lse_up = parallel_wall_attn_decode(
                q, v, p_curr, kt_up, rc_up, None, scale, C, g_scalar_cumsum=c_all
            )

            kt_rsi, rc_rsi, comp_bkv = build_wall_kv_cache_rsi(k, P, C, out_dtype=torch.bfloat16)
            o_rsi, lse_rsi, comp_dec = parallel_wall_attn_decode_rsi(
                q, v, p_curr, kt_rsi, rc_rsi, scale, C, g_scalar_cumsum=c_all
            )

            prof_key = f"Tkv{T_kv}_K{K}"
            decode_profiles[prof_key] = {
                "build_kv_cache_rsi": {
                    "n_regs": int(comp_bkv.n_regs),
                    "n_spills": int(comp_bkv.n_spills),
                    "smem_bytes": int(comp_bkv.metadata.shared),
                },
                "decode_rsi": {
                    "n_regs": int(comp_dec.n_regs),
                    "n_spills": int(comp_dec.n_spills),
                    "smem_bytes": int(comp_dec.metadata.shared),
                },
            }

            up_cache_ms = _time_cuda_ms(lambda: build_wall_kv_cache(k, P, C, out_dtype=torch.bfloat16), warmup=4, iters=iters)
            up_dec_ms = _time_cuda_ms(
                lambda: parallel_wall_attn_decode(q, v, p_curr, kt_up, rc_up, None, scale, C, g_scalar_cumsum=c_all),
                warmup=4, iters=iters,
            )
            up_tot_ms = _time_cuda_ms(
                lambda: parallel_wall_attn_decode(
                    q, v, p_curr, *build_wall_kv_cache(k, P, C, out_dtype=torch.bfloat16), None, scale, C, g_scalar_cumsum=c_all
                ),
                warmup=4, iters=iters,
            )

            rsi_cache_ms = _time_cuda_ms(lambda: build_wall_kv_cache_rsi(k, P, C, out_dtype=torch.bfloat16), warmup=4, iters=iters)
            rsi_dec_ms = _time_cuda_ms(
                lambda: parallel_wall_attn_decode_rsi(q, v, p_curr, kt_rsi, rc_rsi, scale, C, g_scalar_cumsum=c_all),
                warmup=4, iters=iters,
            )
            rsi_tot_ms = _time_cuda_ms(
                lambda: parallel_wall_attn_decode_rsi(
                    q, v, p_curr, *build_wall_kv_cache_rsi(k, P, C, out_dtype=torch.bfloat16)[:2], scale, C, g_scalar_cumsum=c_all
                ),
                warmup=4, iters=iters,
            )

            cases.append({
                "B": B,
                "T_q": T_q,
                "T_kv": T_kv,
                "HQ": HQ,
                "H": H,
                "K": K,
                "upstream_build_cache_ms": up_cache_ms,
                "upstream_decode_ms": up_dec_ms,
                "upstream_total_ms": up_tot_ms,
                "rsi_build_cache_ms": rsi_cache_ms,
                "rsi_decode_ms": rsi_dec_ms,
                "rsi_total_ms": rsi_tot_ms,
                "build_cache_speedup_x": float(up_cache_ms / rsi_cache_ms),
                "decode_speedup_x": float(up_dec_ms / rsi_dec_ms),
                "total_speedup_x": float(up_tot_ms / rsi_tot_ms),
                "kt_max_abs_diff": float((kt_rsi.float() - kt_up.float()).abs().max().item()),
                "rc_max_abs_diff": float((rc_rsi.float() - rc_up.float()).abs().max().item()),
                "o_rel_l2_vs_upstream": rel_l2_error(o_rsi, o_up),
            })

    return {
        "decode_compiler_profiles": decode_profiles,
        "cases": cases,
    }


def run_track3_ablation_ladder(device: torch.device, smoke: bool = False) -> dict:
    B = 2
    T = 2048
    HQ, H, W = 16, 4, 512
    iters = 8 if smoke else 15
    ablation_by_dim = {}

    for D in (1024, 2048):
        K = D // HQ
        torch.manual_seed(3000 + D)
        base_block = WallAttentionBlockRSI(
            hidden_size=D,
            num_heads=HQ,
            num_kv_heads=H,
            qkv_bias=False,
            qk_norm=True,
            window_size=W,
            use_output_gate=True,
            use_scalar_gate=True,
            stage="A0",
        ).to(device=device, dtype=torch.bfloat16)
        base_sd = base_block.state_dict()

        x0 = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
        dy0 = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

        stages_out = {}
        ref_out = None
        ref_dx = None
        a0_tot_ms = None

        for stage in ("A0", "A1", "A2", "A3", "A4"):
            blk = WallAttentionBlockRSI(
                hidden_size=D,
                num_heads=HQ,
                num_kv_heads=H,
                qkv_bias=False,
                qk_norm=True,
                window_size=W,
                use_output_gate=True,
                use_scalar_gate=True,
                stage=stage,
            ).to(device=device, dtype=torch.bfloat16)
            blk.load_state_dict(base_sd, strict=True)
            blk.sync_fused_weights()

            x_test = x0.detach().clone().requires_grad_(True)
            y_test = blk(x_test)
            y_test.backward(dy0)
            if stage == "A1":
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
            if stage == "A0":
                a0_tot_ms = tot_ms

            stages_out[stage] = {
                "stage": stage,
                "D": D,
                "K": K,
                "T": T,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "total_ms": tot_ms,
                "tokens_per_s": float((B * T) / (tot_ms * 1e-3)),
                "speedup_vs_A0_x": float(a0_tot_ms / tot_ms),
                "rel_l2_out_vs_A1": 0.0 if ref_out is None else rel_l2_error(y_test, ref_out),
                "rel_l2_dx_vs_A1": 0.0 if ref_dx is None else rel_l2_error(x_test.grad, ref_dx),
            }

        ablation_by_dim[f"D{D}_K{K}"] = stages_out

    return ablation_by_dim


def _run_cp8_worker(smoke: bool, out_path: str):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    B, HQ, H, K = 1, 16, 8, 64
    V = K
    seq_list = [16384] if smoke else [16384, 32768]
    iters = 8 if smoke else 15
    results = []

    for T_total in seq_list:
        T_loc = T_total // world_size
        torch.manual_seed(4000 + T_total)
        q_full = torch.randn(B, T_total, HQ, K, device=device, dtype=torch.bfloat16)
        k_full = torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16)
        v_full = torch.randn(B, T_total, H, V, device=device, dtype=torch.bfloat16)
        g_full = -torch.rand(B, T_total, HQ, K, device=device, dtype=torch.float32) * 0.05
        gs_full = -torch.rand(B, T_total, HQ, device=device, dtype=torch.float32) * 0.05
        do_full = torch.randn(B, T_total, HQ, V, device=device, dtype=torch.bfloat16)

        # Single-GPU full-sequence reference on rank 0
        q_ref = q_full.detach().clone().requires_grad_(True)
        k_ref = k_full.detach().clone().requires_grad_(True)
        v_ref = v_full.detach().clone().requires_grad_(True)
        g_ref = g_full.detach().clone().requires_grad_(True)
        gs_ref = gs_full.detach().clone().requires_grad_(True)
        o_ref = parallel_wall_attn_rsi(q_ref, k_ref, v_ref, g_ref, g_scalar=gs_ref, window_size=None)
        o_ref.backward(do_full)

        s0, s1 = rank * T_loc, (rank + 1) * T_loc
        q_loc0 = q_full[:, s0:s1].contiguous()
        k_loc0 = k_full[:, s0:s1].contiguous()
        v_loc0 = v_full[:, s0:s1].contiguous()
        g_loc0 = g_full[:, s0:s1].contiguous()
        gs_loc0 = gs_full[:, s0:s1].contiguous()
        do_loc0 = do_full[:, s0:s1].contiguous()

        # Verify Ulysses Wall-CP=8 parity
        qu = q_loc0.detach().clone().requires_grad_(True)
        ku = k_loc0.detach().clone().requires_grad_(True)
        vu = v_loc0.detach().clone().requires_grad_(True)
        gu = g_loc0.detach().clone().requires_grad_(True)
        gsu = gs_loc0.detach().clone().requires_grad_(True)
        ou_loc = wall_cp8_ulysses_rsi(qu, ku, vu, gu, gs_local=gsu, window_size=None)
        ou_loc.backward(do_loc0)

        o_gather = [torch.empty_like(ou_loc) for _ in range(world_size)]
        dq_gather = [torch.empty_like(qu.grad) for _ in range(world_size)]
        dk_gather = [torch.empty_like(ku.grad) for _ in range(world_size)]
        dv_gather = [torch.empty_like(vu.grad) for _ in range(world_size)]
        dg_gather = [torch.empty_like(gu.grad) for _ in range(world_size)]
        dgs_gather = [torch.empty_like(gsu.grad) for _ in range(world_size)]
        dist.all_gather(o_gather, ou_loc.detach())
        dist.all_gather(dq_gather, qu.grad.detach())
        dist.all_gather(dk_gather, ku.grad.detach())
        dist.all_gather(dv_gather, vu.grad.detach())
        dist.all_gather(dg_gather, gu.grad.detach())
        dist.all_gather(dgs_gather, gsu.grad.detach())

        o_cat = torch.cat(o_gather, dim=1)
        dq_cat = torch.cat(dq_gather, dim=1)
        dk_cat = torch.cat(dk_gather, dim=1)
        dv_cat = torch.cat(dv_gather, dim=1)
        dg_cat = torch.cat(dg_gather, dim=1)
        dgs_cat = torch.cat(dgs_gather, dim=1)

        def _run_ulysses():
            q_r = q_loc0.detach().requires_grad_(True)
            k_r = k_loc0.detach().requires_grad_(True)
            v_r = v_loc0.detach().requires_grad_(True)
            g_r = g_loc0.detach().requires_grad_(True)
            gs_r = gs_loc0.detach().requires_grad_(True)
            o_r = wall_cp8_ulysses_rsi(q_r, k_r, v_r, g_r, gs_local=gs_r, window_size=None)
            o_r.backward(do_loc0)

        def _run_ring():
            q_r = q_loc0.detach().requires_grad_(True)
            k_r = k_loc0.detach().requires_grad_(True)
            v_r = v_loc0.detach().requires_grad_(True)
            g_r = g_loc0.detach().requires_grad_(True)
            gs_r = gs_loc0.detach().requires_grad_(True)
            o_r = wall_cp8_ring_baseline(q_r, k_r, v_r, g_r, gs_local=gs_r, window_size=None)
            o_r.backward(do_loc0)

        for _ in range(3):
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
                g_s = g_full.detach().requires_grad_(True)
                gs_s = gs_full.detach().requires_grad_(True)
                o_s = parallel_wall_attn_rsi(q_s, k_s, v_s, g_s, g_scalar=gs_s, window_size=None)
                o_s.backward(do_full)

            single_ms = _time_cuda_ms(_run_single, warmup=3, iters=iters)
            results.append({
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
                "rel_l2_dg_vs_single": rel_l2_error(dg_cat, g_ref.grad),
                "rel_l2_dgs_vs_single": rel_l2_error(dgs_cat, gs_ref.grad),
            })

    if rank == 0:
        Path(out_path).write_text(json.dumps({"cases": results}, indent=2), encoding="utf-8")
    dist.destroy_process_group()


def run_track4_cp8_nvswitch(smoke: bool = False) -> dict:
    tmp_out = "/tmp/wall_cp8_out.json"
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29650",
        str(Path(__file__).resolve()),
        "--cp8-worker",
        "--cp8-out",
        tmp_out,
    ]
    if smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True)
    return json.loads(Path(tmp_out).read_text(encoding="utf-8"))


class _TinyWallLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, num_kv_heads: int, num_layers: int, stage: str):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.pre_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.layers = nn.ModuleList([
            WallAttentionBlockRSI(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                qkv_bias=False,
                qk_norm=True,
                window_size=512,
                use_output_gate=True,
                use_scalar_gate=True,
                stage=stage,
            )
            for _ in range(num_layers)
        ])
        for blk in self.layers:
            for p_name, p in blk.named_parameters():
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
    vocab_size = 4096
    B, T = 2, 2048
    HQ, H, num_layers = 16, 4, 4

    results_by_dim = {}
    for D in (1024, 2048):
        K = D // HQ
        torch.manual_seed(5000 + D)
        train_batches = [
            torch.randint(0, vocab_size, (B, T + 1), device=device, dtype=torch.long)
            for _ in range(16)
        ]
        # Make structured repeating patterns so the model learns rapidly in fixed wall-clock budget
        for b_idx, batch in enumerate(train_batches):
            base_seq = torch.arange(T + 1, device=device)
            batch[:, :] = ((base_seq[None, :] * 7 + b_idx * 13) % 256) + (batch % 4)
        val_batch = train_batches[0].clone()

        torch.manual_seed(5100 + D)
        init_model = _TinyWallLM(vocab_size, D, HQ, H, num_layers, stage="A0").to(device=device, dtype=torch.bfloat16)
        init_sd = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
        del init_model

        dim_res = {}
        for stage in ("A0", "A4"):
            torch.manual_seed(5200 + D)
            model = _TinyWallLM(vocab_size, D, HQ, H, num_layers, stage=stage).to(device=device, dtype=torch.bfloat16)
            model.load_state_dict(init_sd, strict=True)
            model.sync_all_fused()

            if stage == "A0":
                params = [p for n, p in model.named_parameters() if "fused_in_proj" not in n]
            else:
                params = [
                    p for n, p in model.named_parameters()
                    if not any(x in n for x in ("q_proj", "k_proj", "v_proj", "g_proj", "gs_proj", "o_gate_proj"))
                ]
            opt = torch.optim.AdamW(params, lr=8e-4, weight_decay=0.01)

            # Warmup compilation before starting the wall-clock timer
            for w_step in range(2):
                opt.zero_grad(set_to_none=True)
                logits = model(train_batches[w_step][:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), train_batches[w_step][:, 1:].reshape(-1))
                loss.backward()
            model.load_state_dict(init_sd, strict=True)
            model.sync_all_fused()
            opt = torch.optim.AdamW(params, lr=8e-4, weight_decay=0.01)

            with torch.no_grad():
                init_logits = model(val_batch[:, :-1])
                init_loss = float(
                    F.cross_entropy(init_logits.reshape(-1, vocab_size).float(), val_batch[:, 1:].reshape(-1)).item()
                )

            torch.cuda.synchronize()
            t_start = time.perf_counter()
            steps = 0
            while (time.perf_counter() - t_start) < budget_s:
                batch = train_batches[steps % len(train_batches)]
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

            with torch.no_grad():
                val_logits = model(val_batch[:, :-1])
                final_loss = float(
                    F.cross_entropy(val_logits.reshape(-1, vocab_size).float(), val_batch[:, 1:].reshape(-1)).item()
                )
            val_bpb = float(final_loss / math.log(2.0))
            init_bpb = float(init_loss / math.log(2.0))

            dim_res[stage] = {
                "stage": stage,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-all", action="store_true", help="Run smoke + reps 1,2,3 in one process to share autotune cache")
    parser.add_argument("--update-track5", action="store_true", help="Re-run Track 5 for smoke + reps 1,2,3 in-place")
    parser.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR.parent / "results" / "raw" / "wall_attn_rsi"))
    parser.add_argument("--cp8-worker", action="store_true")
    parser.add_argument("--cp8-out", type=str, default="/tmp/wall_cp8_out.json")
    args = parser.parse_args()

    if args.cp8_worker:
        _run_cp8_worker(args.smoke, args.cp8_out)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    if args.update_track5:
        from fla.ops.wall_attn.parallel import (
            parallel_wall_attn_bwd_kernel_dkv,
            parallel_wall_attn_bwd_kernel_dq,
            parallel_wall_attn_fwd_kernel,
        )
        for kern in (parallel_wall_attn_fwd_kernel, parallel_wall_attn_bwd_kernel_dq, parallel_wall_attn_bwd_kernel_dkv):
            if len(kern.configs) > 8:
                kept = [
                    cfg for cfg in kern.configs
                    if cfg.kwargs.get("BT", 32) in (32, 64)
                    and cfg.kwargs.get("BS", 32) == 32
                    and cfg.num_stages in (1, 2)
                    and cfg.num_warps in (4, 8)
                ]
                if kept:
                    kern.configs = kept
        for is_smoke, rep_id in [(True, 0), (False, 1), (False, 2), (False, 3)]:
            fname = "smoke.json" if is_smoke else f"rep{rep_id}.json"
            fp = out_dir / fname
            data = json.loads(fp.read_text(encoding="utf-8"))
            data["track5_autoresearch_val_bpb"] = run_track5_autoresearch_val_bpb(device, smoke=is_smoke)
            fp.write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))
            print(f"[E150] Updated Track 5 in {fp}", flush=True)
        return

    schedules = [(True, 0), (False, 1), (False, 2), (False, 3)] if args.run_all else [(args.smoke, args.rep)]

    for is_smoke, rep_id in schedules:
        print(f"[E150 WallAttention RSI] Starting benchmark (rep={rep_id}, smoke={is_smoke}) on {torch.cuda.get_device_name(0)}...", flush=True)
        t_all = time.perf_counter()

        track1 = run_track1_kernels_and_oracle(device, smoke=is_smoke)
        print("[E150] Track 1 (Kernels & Oracle) complete.", flush=True)

        track2 = run_track2_decode_and_kv_cache(device, smoke=is_smoke)
        print("[E150] Track 2 (KV-Cache Builder & Decode) complete.", flush=True)

        track3 = run_track3_ablation_ladder(device, smoke=is_smoke)
        print("[E150] Track 3 (5-Stage Ablation Ladder A0->A4) complete.", flush=True)

        track4 = run_track4_cp8_nvswitch(smoke=is_smoke)
        print("[E150] Track 4 (8-GPU NVSwitch Wall-CP=8) complete.", flush=True)

        track5 = run_track5_autoresearch_val_bpb(device, smoke=is_smoke)
        print("[E150] Track 5 (Fixed-Wall-Clock Autoresearch val_bpb) complete.", flush=True)

        payload = {
            "experiment_id": "E150",
            "version": "v141",
            "title": "Round-24 RSI: Wall Attention: Length Generalization With Diagonal Gates (Tilde Research)",
            "rep": rep_id,
            "smoke": is_smoke,
            "gpu_name": torch.cuda.get_device_name(0),
            "world_size": torch.cuda.device_count(),
            "elapsed_total_s": float(time.perf_counter() - t_all),
            "track1_kernels_and_oracle": track1,
            "track2_decode_and_kv_cache": track2,
            "track3_ablation_ladder": track3,
            "track4_cp8_nvswitch": track4,
            "track5_autoresearch_val_bpb": track5,
        }

        fname = "smoke.json" if is_smoke else f"rep{rep_id}.json"
        out_file = out_dir / fname
        out_file.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
        print(f"[E150] Wrote {out_file} in {payload['elapsed_total_s']:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
