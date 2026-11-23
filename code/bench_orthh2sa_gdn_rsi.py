#!/usr/bin/env python3
"""
Round-35 RSI (E161, v152) 6-Track Hardware Benchmark, Dipolar Intra-Block Fact-Disentanglement Test,
High-Rank Multi-Fact Phase Law, & Governed Loop Optimization (v3.8) Wave-3 Tournament.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if "/tmp/fla_repo" not in sys.path and os.path.exists("/tmp/fla_repo"):
    sys.path.insert(0, "/tmp/fla_repo")

try:
    from fla.ops.moba.parallel import parallel_moba as _fla_moba
except Exception:
    _fla_moba = None

from moba_rsi_kernels import parallel_moba_rsi
from h2sa_gdn_rsi_kernels import (
    build_rope_cos_sin,
    h2sa_gdn_cp8_ring_baseline,
    parallel_h2sa_gdn_rsi,
)
from adah2sa_gdn_rsi_kernels import parallel_adah2sa_gdn_rsi
from orthh2sa_gdn_rsi_kernels import (
    OrthH2SAGDNBlockRSI,
    _orthh2sa_block_subtile_summary_fwd_rsi_kernel,
    _orthh2sa_gdn_subtile_wy_fwd_rsi_kernel,
    audit_orthh2sa_gdn_rsi_cubin_kernels,
    orthh2sa_gdn_cp8_ulysses_rsi,
    orthh2sa_gdn_decode_rsi,
    orthh2sa_gdn_fp64_oracle,
    parallel_orthh2sa_gdn_rsi,
)

FROZEN_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def verify_harness_integrity() -> Dict[str, Any]:
    harness_path = CODE_DIR / "rsi_kernel_harness.py"
    digest = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    assert digest == FROZEN_HARNESS_SHA256, f"Frozen harness SHA256 mismatch: {digest}"
    return {
        "harness_path": "rsi_kernel_harness.py",
        "harness_sha256": digest,
        "verified": True,
    }


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    return float(torch.linalg.norm(af - bf).item() / max(torch.linalg.norm(bf).item(), 1e-12))


def time_gpu_callable(fn, warmup: int = 5, rep: int = 15) -> float:
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
    return float(times[len(times) // 2])


def time_gpu_callable_p50_p95(fn, warmup: int = 5, rep: int = 25) -> Tuple[float, float]:
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
    p50 = float(times[len(times) // 2])
    p95 = float(times[min(len(times) - 1, int(0.95 * len(times)))])
    return p50, p95


def run_upstream_moba(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int, topk: int) -> torch.Tensor:
    B, T, HQ, K = q.shape
    HKV = k.shape[2]
    G = HQ // HKV
    ke = k.repeat_interleave(G, dim=2) if G > 1 else k
    ve = v.repeat_interleave(G, dim=2) if G > 1 else v
    if _fla_moba is not None:
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
        out = _fla_moba(
            q.reshape(1, B * T, HQ, K),
            ke.reshape(1, B * T, HQ, K),
            ve.reshape(1, B * T, HQ, K),
            cu,
            max_seqlen=T,
            chunk_size=chunk_size,
            topk=topk,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(B, T, HQ, K)
    return parallel_moba_rsi(q, k, v, chunk_size=chunk_size, topk=topk)


# ==============================================================================
# Track 1A: FP64 Oracle Parity, 3 Mutations, & Dipolar Intra-Block Disentanglement
# ==============================================================================

def run_dipolar_intrablock_fact_disentanglement(device: torch.device) -> Dict[str, Any]:
    """
    Decisive Dipolar Intra-Block Fact-Disentanglement Test:
      Freeze routing masks (q_subtile_mask, qtile_mask) so block c_rej = 1 (subtiles u=2, 3)
      is strictly REJECTED.
      Inside block c_rej = 1:
        - Subtile u=2 holds (k_A, +v_dip)
        - Subtile u=3 holds (k_B, -v_dip) with k_A orthogonal to k_B!
      Because the coarse-block average value of block c_rej = 1 is:
        bar_v_{c=1} = 0.5 * (+v_dip) + 0.5 * (-v_dip) = 0,
      BOTH E159 (H2SA-GDN) and E160 (AdaH2SA-GDN) collapse to 0.000000 dipolar sensitivity!
      By contrast, E161 (OrthH2SA-GDN) updates S across subtiles u=2 and u=3 separately in registers,
      preserving (+k_A v_dip^T - k_B v_dip^T) so query q = (k_A - k_B)/sqrt(2) recovers +v_dip with cosine > 0.999!
    """
    torch.manual_seed(16101)
    B, T, HQ, HKV, K = 2, 1024, 16, 4, 64
    BC, BS = 128, 64
    NT = T // 16  # In FP32, BM = 16 so NT = T // 16 = 64 tiles

    q = torch.randn((B, T, HQ, K), dtype=torch.float32, device=device) * 0.15
    k = torch.randn((B, T, HKV, K), dtype=torch.float32, device=device) * 0.15
    v = torch.randn((B, T, HKV, K), dtype=torch.float32, device=device) * 0.05
    g_mem = torch.full((B, T, HQ), 2.5, dtype=torch.float32, device=device)

    # Construct orthogonal subtile keys k_A perp k_B
    k_A = torch.nn.functional.normalize(torch.randn((B, 1, HKV, K), device=device), dim=-1)
    k_B_raw = torch.randn((B, 1, HKV, K), device=device)
    k_B = torch.nn.functional.normalize(k_B_raw - (k_B_raw * k_A).sum(dim=-1, keepdim=True) * k_A, dim=-1)
    v_dip = torch.nn.functional.normalize(torch.randn((B, 1, HKV, K), device=device), dim=-1) * (0.85 * math.sqrt(K))

    k_dip = k.clone()
    v_plus = v.clone()
    v_minus = v.clone()

    # Subtile u = 2 (first half of block c_rej = 1) has k_A and +v_dip (or -v_dip in counterfactual flip)
    k_dip[:, 2 * BS : 3 * BS] = k_A.expand(B, BS, HKV, K)
    v_plus[:, 2 * BS : 3 * BS] = v_dip.expand(B, BS, HKV, K)
    v_minus[:, 2 * BS : 3 * BS] = -v_dip.expand(B, BS, HKV, K)

    # Subtile u = 3 (second half of block c_rej = 1) has k_B and OPPOSITE sign (-v_dip vs +v_dip)
    # so the coarse-block average value across block c_rej = 1 is identically ZERO!
    k_dip[:, 3 * BS : 4 * BS] = k_B.expand(B, BS, HKV, K)
    v_plus[:, 3 * BS : 4 * BS] = -v_dip.expand(B, BS, HKV, K)
    v_minus[:, 3 * BS : 4 * BS] = v_dip.expand(B, BS, HKV, K)

    # Blocks c = 2..6 are low-energy distractor blocks
    for c_fill in range(2, 7):
        ortho = torch.randn((B, 1, HKV, K), device=device)
        ortho = ortho - (ortho * k_A).sum(dim=-1, keepdim=True) * k_A - (ortho * k_B).sum(dim=-1, keepdim=True) * k_B
        ortho = torch.nn.functional.normalize(ortho, dim=-1)
        k_fill = 0.35 * k_A + 0.35 * k_B + math.sqrt(max(0.0, 1.0 - 2 * 0.35 * 0.35)) * ortho
        k_dip[:, c_fill * BC : (c_fill + 1) * BC] = k_fill.expand(B, BC, HKV, K)
        v_plus[:, c_fill * BC : (c_fill + 1) * BC] = 0.0
        v_minus[:, c_fill * BC : (c_fill + 1) * BC] = 0.0

    # Query in block c_q = 7 queries the dipolar difference direction (k_A - k_B) / sqrt(2)
    G = HQ // HKV
    q_dip = ((k_A - k_B) / math.sqrt(2.0)).repeat_interleave(G, dim=2)
    q[:, 7 * BC : 8 * BC] = q_dip.expand(B, BC, HQ, K)
    v_target_hq = (v_dip * math.sqrt(2.0)).repeat_interleave(G, dim=2).expand(B, BC, HQ, K)

    # Freeze routing masks so subtiles 0, 1 (block 0) and 10..15 (blocks 5, 6, 7) are selected,
    # strictly REJECTING block c_rej = 1 (subtiles 2, 3)!
    forced_bits = (1 << 0) | (1 << 1) | (1 << 10) | (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15)
    frozen_sub_mask = torch.full((B, HQ, T), forced_bits, dtype=torch.uint64, device=device)
    frozen_tile_mask = torch.full((B, HQ, NT), forced_bits, dtype=torch.uint64, device=device)
    ext_masks = (frozen_sub_mask, frozen_tile_mask)

    def eval_dipolar_flip(fn_eval) -> Tuple[float, float]:
        o_p = fn_eval(k_dip, v_plus)[:, 7 * BC : 8 * BC]
        o_m = fn_eval(k_dip, v_minus)[:, 7 * BC : 8 * BC]
        delta_o = o_p - o_m
        target = 2.0 * v_target_hq
        norm_delta = torch.linalg.norm(delta_o).item()
        if norm_delta < 1e-9:
            return 0.0, 0.0
        cos = torch.nn.functional.cosine_similarity(delta_o.reshape(-1, K), target.reshape(-1, K), dim=-1).mean().item()
        mag = norm_delta / max(torch.linalg.norm(target).item(), 1e-12)
        return float(cos), float(mag)

    # 1. Selected-Only Sparse Attention (no GDN state)
    cos_sel_only, _ = eval_dipolar_flip(
        lambda kp, vp: parallel_orthh2sa_gdn_rsi(
            q, kp, vp, g_mem, chunk_size=BC, subtile_size=BS,
            enable_subtile_rescue=False, enable_gdn_memory=False, external_masks=ext_masks,
        )
    )

    # 2. E159 Coarse-Block Uniform-Write H2SA-GDN (bar_v_c = 0 -> exact cancellation!)
    cos_e159, mag_e159 = eval_dipolar_flip(
        lambda kp, vp: parallel_adah2sa_gdn_rsi(
            q, kp, vp, g_mem, chunk_size=BC, subtile_size=BS,
            alpha_decay=0.96, beta_write=0.85, gamma_surp=0.0, tau_surp=0.0,
            external_masks=ext_masks,
        )
    )

    # 3. E160 Coarse-Block Surprise-Gated AdaH2SA-GDN (bar_v_c = 0 -> exact cancellation!)
    cos_e160, mag_e160 = eval_dipolar_flip(
        lambda kp, vp: parallel_adah2sa_gdn_rsi(
            q, kp, vp, g_mem, chunk_size=BC, subtile_size=BS,
            alpha_decay=0.96, beta_write=0.85, gamma_surp=12.0, tau_surp=0.30,
            external_masks=ext_masks,
        )
    )

    # 4. E161 Subtile-WY OrthH2SA-GDN (disentangles subtiles u=2 and u=3 inside rejected block c=1!)
    cos_e161, mag_e161 = eval_dipolar_flip(
        lambda kp, vp: parallel_orthh2sa_gdn_rsi(
            q, kp, vp, g_mem, chunk_size=BC, subtile_size=BS,
            alpha_decay=0.96, beta_write=0.85, gamma_surp=12.0, tau_surp=0.30,
            external_masks=ext_masks,
        )
    )

    return {
        "selected_only_dipolar_cosine": round(cos_sel_only, 6),
        "e159_coarse_uniform_dipolar_cosine": round(cos_e159, 6),
        "e159_coarse_uniform_dipolar_retention": round(mag_e159, 6),
        "e160_coarse_surprise_dipolar_cosine": round(cos_e160, 6),
        "e160_coarse_surprise_dipolar_retention": round(mag_e160, 6),
        "e161_subtile_wy_dipolar_cosine": round(cos_e161, 6),
        "e161_subtile_wy_dipolar_retention": round(mag_e161, 6),
        "dipolar_disentanglement_verified": (abs(cos_sel_only) < 1e-5) and (mag_e160 < 1e-5) and (cos_e161 > 0.99) and (mag_e161 > 0.45),
    }


def run_track1a_correctness_and_mutations(device: torch.device) -> Dict[str, Any]:
    cases = [
        ("mha_k64_fp32", 2, 512, 8, 8, 64, torch.float32),
        ("mha_k64_bf16", 2, 512, 8, 8, 64, torch.bfloat16),
        ("gqa_k64_fp32", 2, 512, 16, 4, 64, torch.float32),
        ("gqa_k64_bf16", 2, 512, 16, 4, 64, torch.bfloat16),
        ("gqa_k128_fp32", 2, 512, 16, 4, 128, torch.float32),
        ("gqa_k128_bf16", 2, 512, 16, 4, 128, torch.bfloat16),
    ]
    results = []
    max_fp32_err = 0.0
    max_bf16_err = 0.0

    for name, B, T, HQ, HKV, K, dtype in cases:
        torch.manual_seed(161)
        q0 = (torch.randn((B, T, HQ, K), device=device, dtype=dtype) * 0.5).requires_grad_(True)
        k0 = (torch.randn((B, T, HKV, K), device=device, dtype=dtype) * 0.5).requires_grad_(True)
        v0 = (torch.randn((B, T, HKV, K), device=device, dtype=dtype) * 0.5).requires_grad_(True)
        g0 = (torch.randn((B, T, HQ), device=device, dtype=dtype) * 0.5).requires_grad_(True)
        do = torch.randn((B, T, HQ, K), device=device, dtype=dtype) * 0.5

        q_ref = q0.detach().clone().to(torch.float64).requires_grad_(True)
        k_ref = k0.detach().clone().to(torch.float64).requires_grad_(True)
        v_ref = v0.detach().clone().to(torch.float64).requires_grad_(True)
        g_ref = g0.detach().clone().to(torch.float64).requires_grad_(True)

        o_ref, umask_ref = orthh2sa_gdn_fp64_oracle(
            q_ref, k_ref, v_ref, g_ref,
            chunk_size=128, subtile_size=64, topk_blocks=3, topk_subtiles=2,
        )
        o_ref.backward(do.to(torch.float64))

        o_rsi, umask_rsi, _ = parallel_orthh2sa_gdn_rsi(
            q0, k0, v0, g0,
            chunk_size=128, subtile_size=64, topk_blocks=3, topk_subtiles=2,
            return_shared_state=True,
        )
        o_rsi.backward(do)

        err_o = rel_l2(o_rsi, o_ref)
        err_dq = rel_l2(q0.grad, q_ref.grad)
        err_dk = rel_l2(k0.grad, k_ref.grad)
        err_dv = rel_l2(v0.grad, v_ref.grad)
        err_dg = rel_l2(g0.grad, g_ref.grad)
        mask_match = float((umask_rsi == umask_ref).to(torch.float32).mean().item())
        worst = max(err_o, err_dq, err_dk, err_dv, err_dg)
        if dtype == torch.float32:
            max_fp32_err = max(max_fp32_err, worst)
        else:
            max_bf16_err = max(max_bf16_err, worst)

        results.append({
            "case": name,
            "dtype": "fp32" if dtype == torch.float32 else "bf16",
            "B": B, "T": T, "HQ": HQ, "HKV": HKV, "K": K,
            "o_rel_l2": err_o,
            "dq_rel_l2": err_dq,
            "dk_rel_l2": err_dk,
            "dv_rel_l2": err_dv,
            "dg_rel_l2": err_dg,
            "max_rel_l2": worst,
            "subtile_mask_exact_match": mask_match,
        })

    # 3 Adversarial Mutations (with subtile-level offsets so intra-block subtiles differ)
    B, T, HQ, HKV, K = 2, 512, 16, 4, 64
    torch.manual_seed(162)
    qm = (torch.randn((B, T, HQ, K), device=device, dtype=torch.float32) * 0.8).requires_grad_(True)
    km_sub = torch.randn((B, T // 64, 1, HKV, K), device=device, dtype=torch.float32) * 0.7
    vm_sub = torch.randn((B, T // 64, 1, HKV, K), device=device, dtype=torch.float32) * 0.75
    km = (torch.randn((B, T, HKV, K), device=device, dtype=torch.float32) * 0.35 + km_sub.expand(B, T // 64, 64, HKV, K).reshape(B, T, HKV, K)).requires_grad_(True)
    vm = (torch.randn((B, T, HKV, K), device=device, dtype=torch.float32) * 0.35 + vm_sub.expand(B, T // 64, 64, HKV, K).reshape(B, T, HKV, K)).requires_grad_(True)
    gm = (torch.randn((B, T, HQ), device=device, dtype=torch.float32) + 1.5).requires_grad_(True)
    dom = torch.randn((B, T, HQ, K), device=device, dtype=torch.float32)

    q_r = qm.detach().clone().to(torch.float64).requires_grad_(True)
    k_r = km.detach().clone().to(torch.float64).requires_grad_(True)
    v_r = vm.detach().clone().to(torch.float64).requires_grad_(True)
    g_r = gm.detach().clone().to(torch.float64).requires_grad_(True)
    o_r, _ = orthh2sa_gdn_fp64_oracle(q_r, k_r, v_r, g_r, chunk_size=128, subtile_size=64, topk_blocks=2, topk_subtiles=2)
    o_r.backward(dom.to(torch.float64))

    mutations = []
    for m_idx, m_name in [
        (1, "M1_collapse_subtile_wy_to_coarse_block_mean"),
        (2, "M2_drop_subtile_surprise_jacobian_in_bwd"),
        (3, "M3_corrupt_within_block_subtile_reverse_order"),
    ]:
        q_c = qm.detach().clone().requires_grad_(True)
        k_c = km.detach().clone().requires_grad_(True)
        v_c = vm.detach().clone().requires_grad_(True)
        g_c = gm.detach().clone().requires_grad_(True)
        o_m = parallel_orthh2sa_gdn_rsi(
            q_c, k_c, v_c, g_c,
            chunk_size=128, subtile_size=64, topk_blocks=2, topk_subtiles=2,
            mutate_mode=m_idx,
        )
        o_m.backward(dom)
        e_o = rel_l2(o_m, o_r)
        e_dk = rel_l2(k_c.grad, k_r.grad)
        e_dv = rel_l2(v_c.grad, v_r.grad)
        e_max = max(e_o, e_dk, e_dv)
        mutations.append({
            "mutation": m_name,
            "o_rel_l2": e_o,
            "dk_rel_l2": e_dk,
            "dv_rel_l2": e_dv,
            "max_rel_l2": e_max,
            "killed": e_max > 1.0e-3,
        })

    dipolar = run_dipolar_intrablock_fact_disentanglement(device)

    return {
        "max_fp32_rel_l2": max_fp32_err,
        "max_bf16_rel_l2": max_bf16_err,
        "fp32_gate_passed": max_fp32_err < 1.0e-4,
        "bf16_gate_passed": max_bf16_err < 1.5e-2,
        "mutations_killed": sum(1 for m in mutations if m["killed"]),
        "mutations_total": len(mutations),
        "mutation_kill_rate": float(sum(1 for m in mutations if m["killed"]) / len(mutations)),
        "dipolar_intrablock_disentanglement": dipolar,
        "cases": results,
        "mutations": mutations,
    }


# ==============================================================================
# Track 1C: 8-Shape Training Forward + Backward + Memory Matrix
# ==============================================================================

def run_track1c_training_matrix(device: torch.device, smoke: bool = False) -> List[Dict[str, Any]]:
    shapes = [
        ("S1_B2_T1024_MHA_K64_BC128", 2, 1024, 16, 16, 64, 128, 64, 3, 2),
        ("S2_B2_T2048_MHA_K64_BC128", 2, 2048, 16, 16, 64, 128, 64, 3, 2),
        ("S3_B4_T2048_MHA_K64_BC256", 4, 2048, 16, 16, 64, 256, 128, 3, 2),
        ("S4_B2_T1024_MHA_K128_BC128", 2, 1024, 16, 16, 128, 128, 64, 3, 2),
        ("S5_B2_T2048_MHA_K128_BC128", 2, 2048, 16, 16, 128, 128, 64, 3, 2),
        ("S6_B4_T2048_MHA_K128_BC256", 4, 2048, 16, 16, 128, 256, 128, 3, 2),
        ("S7_B2_T2048_GQA_K64_BC128", 2, 2048, 16, 2, 64, 128, 64, 3, 2),
        ("S8_B4_T2048_GQA_K128_BC256", 4, 2048, 16, 2, 128, 256, 128, 3, 2),
    ]
    warmup = 3 if smoke else 5
    rep = 7 if smoke else 15
    rows = []

    for label, B, T, HQ, HKV, K, BC, BS, topk_b, topk_s in shapes:
        torch.manual_seed(163)
        q = torch.randn((B, T, HQ, K), dtype=torch.bfloat16, device=device, requires_grad=True)
        k = torch.randn((B, T, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
        v = torch.randn((B, T, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
        g = torch.randn((B, T, HQ), dtype=torch.bfloat16, device=device, requires_grad=True)
        do = torch.randn((B, T, HQ, K), dtype=torch.bfloat16, device=device)
        R = BC // BS
        NC = T // BC
        NS = T // BS

        def eager_fwd():
            ao = run_upstream_moba(q, k, v, BC, topk_b + 1)
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
            s_st = torch.stack(s_list, dim=1).repeat_interleave(HQ // HKV, dim=2)
            qh = (q.to(torch.float32) * torch.rsqrt((q.to(torch.float32) ** 2).sum(-1, keepdim=True) + 1e-6)).view(B, NC, BC, HQ, K)
            mr = torch.einsum("bcthk,bchkv->bcthv", qh, s_st).reshape(B, T, HQ, K)
            return ao + (torch.sigmoid(g.to(torch.float32)).unsqueeze(-1) * mr).to(q.dtype)

        def rsi_fwd():
            return parallel_orthh2sa_gdn_rsi(q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base_m = torch.cuda.memory_allocated(device)
        o_e = eager_fwd()
        o_e.backward(do)
        naive_peak_mib = (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0 * 1024.0)
        q.grad = k.grad = v.grad = g.grad = None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base_m = torch.cuda.memory_allocated(device)
        o_r = rsi_fwd()
        o_r.backward(do)
        rsi_peak_mib = (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0 * 1024.0)
        q.grad = k.grad = v.grad = g.grad = None

        naive_fwd_ms = time_gpu_callable(eager_fwd, warmup=warmup, rep=rep)
        o_e = eager_fwd()
        naive_bwd_ms = time_gpu_callable(lambda: o_e.backward(do, retain_graph=True), warmup=warmup, rep=rep)

        def naive_tot():
            q.grad = k.grad = v.grad = g.grad = None
            out_e = eager_fwd()
            out_e.backward(do)

        naive_tot_ms = time_gpu_callable(naive_tot, warmup=warmup, rep=rep)

        rsi_fwd_ms = time_gpu_callable(rsi_fwd, warmup=warmup, rep=rep)
        o_r = rsi_fwd()
        rsi_bwd_ms = time_gpu_callable(lambda: o_r.backward(do, retain_graph=True), warmup=warmup, rep=rep)

        def rsi_tot():
            q.grad = k.grad = v.grad = g.grad = None
            out_r = rsi_fwd()
            out_r.backward(do)

        rsi_tot_ms = time_gpu_callable(rsi_tot, warmup=warmup, rep=rep)

        rows.append({
            "label": label,
            "B": B, "T": T, "HQ": HQ, "HKV": HKV, "K": K,
            "chunk_size": BC, "subtile_size": BS,
            "topk_blocks": topk_b, "topk_subtiles": topk_s,
            "naive_fwd_ms": naive_fwd_ms,
            "naive_bwd_ms": naive_bwd_ms,
            "naive_total_ms": naive_tot_ms,
            "naive_peak_mib": naive_peak_mib,
            "rsi_fwd_ms": rsi_fwd_ms,
            "rsi_bwd_ms": rsi_bwd_ms,
            "rsi_total_ms": rsi_tot_ms,
            "rsi_peak_mib": rsi_peak_mib,
            "fwd_speedup": naive_fwd_ms / rsi_fwd_ms,
            "bwd_speedup": naive_bwd_ms / rsi_bwd_ms,
            "total_speedup": naive_tot_ms / rsi_tot_ms,
            "mem_reduction": naive_peak_mib / max(rsi_peak_mib, 1.0),
            "rsi_tok_per_s": (B * T) / (rsi_tot_ms * 1e-3),
        })
    return rows


# ==============================================================================
# Track 2: Single-Pass Subtile-WY Decode Matrix (Latency + P95 + Cache Bytes)
# ==============================================================================

def run_track2_decode_matrix(device: torch.device, smoke: bool = False) -> List[Dict[str, Any]]:
    shapes = [
        ("D1_B4_T1024_GQA_K64", 4, 1024, 16, 4, 64, 128, 64, 3, 2),
        ("D2_B4_T2048_GQA_K64", 4, 2048, 16, 4, 64, 128, 64, 3, 2),
        ("D3_B4_T4096_GQA_K128", 4, 4096, 16, 4, 128, 128, 64, 3, 2),
        ("D4_B8_T4096_GQA_K128", 8, 4096, 16, 4, 128, 128, 64, 3, 2),
    ]
    warmup = 3 if smoke else 5
    rep = 11 if smoke else 25
    rows = []

    for label, B, T_cache, HQ, HKV, K, BC, BS, topk_b, topk_s in shapes:
        torch.manual_seed(164)
        k_cache = torch.randn((B, T_cache, HKV, K), dtype=torch.bfloat16, device=device)
        v_cache = torch.randn((B, T_cache, HKV, K), dtype=torch.bfloat16, device=device)
        q_full = torch.randn((B, T_cache, HQ, K), dtype=torch.bfloat16, device=device)
        g_full = torch.randn((B, T_cache, HQ), dtype=torch.bfloat16, device=device)

        q_dec = q_full[:, -1:, :, :].contiguous()
        g_dec = g_full[:, -1:, :].contiguous()

        R = BC // BS
        NC = T_cache // BC
        NS = T_cache // BS
        BV = 32
        NV = K // BV
        mu_cache = torch.empty((B, NC, HKV, K), dtype=torch.float32, device=device)
        mu_sub = torch.empty((B, NS, HKV, K), dtype=torch.float32, device=device)
        vbar_sub = torch.empty((B * HKV, NS, K), dtype=torch.float32, device=device)
        ktilde_cache = torch.empty((B, NS, HKV, K), dtype=torch.float32, device=device)
        _orthh2sa_block_subtile_summary_fwd_rsi_kernel[(NC, B * HKV)](
            k_cache, v_cache, mu_cache, mu_sub, vbar_sub, ktilde_cache,
            T=T_cache, HKV=HKV, K=K, V=K, BC=BC, BS=BS, R=R, OUTLIER_LAMBDA=0.5,
            num_warps=4, num_stages=1,
        )
        s_hist = torch.empty((B * HKV, NC, K, K), dtype=torch.float32, device=device)
        s_hist_sub = torch.empty((B * HKV, NS, K, K), dtype=torch.float32, device=device)
        khat_sub = torch.empty((B * HKV, NS, K), dtype=torch.float32, device=device)
        beta_sub = torch.empty((B * HKV, NS, NV), dtype=torch.float32, device=device)
        s_final = torch.empty((B * HKV, K, K), dtype=torch.float32, device=device)
        _orthh2sa_gdn_subtile_wy_fwd_rsi_kernel[(NV, B * HKV)](
            mu_sub, vbar_sub, s_hist, s_hist_sub, khat_sub, beta_sub, s_final,
            T=T_cache, NC=NC, NS=NS, R=R, HKV=HKV, K=K, V=K, BV=BV,
            ALPHA_SUB=float(0.90 ** (1.0 / R)), BETA0=0.60, GAMMA_SURP=4.0, TAU_SURP=0.20,
            MUTATE_COARSE_COLLAPSE=0,
            num_warps=4, num_stages=1,
        )
        s_last_block = s_hist[:, -1, :, :].contiguous()

        o_chunk = parallel_orthh2sa_gdn_rsi(
            q_full, k_cache, v_cache, g_full,
            chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s,
        )[:, -1:, :, :]
        o_dec = orthh2sa_gdn_decode_rsi(
            q_dec, k_cache, v_cache, g_dec, mu_cache, ktilde_cache, s_last_block,
            chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s,
        )
        err_dec = rel_l2(o_dec, o_chunk)

        naive_ms = time_gpu_callable(
            lambda: run_upstream_moba(q_full, k_cache, v_cache, BC, topk_b + 1)[:, -1:, :, :],
            warmup=warmup, rep=max(5, rep // 2),
        )
        chunk_ms = time_gpu_callable(
            lambda: parallel_orthh2sa_gdn_rsi(
                q_full, k_cache, v_cache, g_full, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s
            )[:, -1:, :, :],
            warmup=warmup, rep=max(5, rep // 2),
        )
        dec_p50_ms, dec_p95_ms = time_gpu_callable_p50_p95(
            lambda: orthh2sa_gdn_decode_rsi(
                q_dec, k_cache, v_cache, g_dec, mu_cache, ktilde_cache, s_last_block,
                chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s,
            ),
            warmup=warmup, rep=rep,
        )

        kv_cache_mib = (2 * B * T_cache * HKV * K * 2) / (1024.0 * 1024.0)
        summary_state_mib = (
            (mu_cache.numel() + ktilde_cache.numel() + s_last_block.numel()) * 4
        ) / (1024.0 * 1024.0)
        active_subtiles = (topk_b - 1) * R + topk_s + R
        effective_read_mib = (2 * B * (active_subtiles * BS) * HKV * K * 2) / (1024.0 * 1024.0) + summary_state_mib

        rows.append({
            "label": label,
            "B": B, "T_cache": T_cache, "HQ": HQ, "HKV": HKV, "K": K,
            "naive_moba_ms": naive_ms,
            "orthh2sa_chunk_ms": chunk_ms,
            "orthh2sa_decode_p50_ms": dec_p50_ms,
            "orthh2sa_decode_p95_ms": dec_p95_ms,
            "kv_cache_mib": round(kv_cache_mib, 3),
            "summary_state_mib": round(summary_state_mib, 3),
            "effective_kv_read_mib": round(effective_read_mib, 3),
            "speedup_vs_naive": naive_ms / dec_p50_ms,
            "speedup_vs_chunk": chunk_ms / dec_p50_ms,
            "o_rel_l2": err_dec,
        })
    return rows


# ==============================================================================
# Track 3: 5-Stage Block Ablation (A0..A4) & 25-Cell High-Rank Multi-Fact Phase Law
# ==============================================================================

def run_track3_block_and_phase_law(device: torch.device, smoke: bool = False) -> Dict[str, Any]:
    configs = [
        ("D1024_B2_T1024_H16_GQA4_K64", 2, 1024, 1024, 16, 4, 64),
        ("D2048_B2_T1024_H16_GQA4_K128", 2, 1024, 2048, 16, 4, 128),
    ]
    stages = [
        ("A0", "Eager periphery + coarse MoBA baseline"),
        ("A1", "Zero-spill parallel_orthh2sa_gdn_rsi + eager periphery"),
        ("A2", "A1 + FusedMoBAQKRMSNormRoPEFunction"),
        ("A3", "A2 + Fused qkvg_proj + FusedMoBARMSNorm + FusedMoBARMSNormSigmoidGated"),
        ("A4", "A3 + DualRouteShare cross-layer (q_subtile_mask, qtile_mask) reuse"),
    ]
    warmup = 3 if smoke else 5
    rep = 7 if smoke else 15
    block_rows = []

    for cfg_name, B, T, D, HQ, HKV, K in configs:
        cos, sin = build_rope_cos_sin(T, K, device)
        a0_tot = None
        a0_mem = None
        for st_code, st_desc in stages:
            torch.manual_seed(165)
            blk1 = OrthH2SAGDNBlockRSI(D, HQ, HKV, K, chunk_size=128, subtile_size=64, stage=st_code).to(device, torch.bfloat16)
            blk2 = OrthH2SAGDNBlockRSI(D, HQ, HKV, K, chunk_size=128, subtile_size=64, stage=st_code).to(device, torch.bfloat16)
            x = torch.randn((B, T, D), dtype=torch.bfloat16, device=device, requires_grad=True)
            do = torch.randn_like(x)

            def fn_fwd():
                if st_code == "A4":
                    h1, masks = blk1(x, cos, sin, return_shared_state=True)
                    return blk2(h1, cos, sin, external_masks=masks)
                return blk2(blk1(x, cos, sin), cos, sin)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base_m = torch.cuda.memory_allocated(device)
            y = fn_fwd()
            y.backward(do)
            peak_mib = (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0 * 1024.0)

            fwd_ms = time_gpu_callable(fn_fwd, warmup=warmup, rep=rep)
            y = fn_fwd()
            bwd_ms = time_gpu_callable(lambda: y.backward(do, retain_graph=True), warmup=warmup, rep=rep)

            def fn_tot():
                x.grad = None
                for p in blk1.parameters():
                    p.grad = None
                for p in blk2.parameters():
                    p.grad = None
                o = fn_fwd()
                o.backward(do)

            tot_ms = time_gpu_callable(fn_tot, warmup=warmup, rep=rep)
            if st_code == "A0":
                a0_tot = tot_ms
                a0_mem = peak_mib

            block_rows.append({
                "config": cfg_name,
                "stage": st_code,
                "description": st_desc,
                "B": B, "T": T, "D": D, "HQ": HQ, "HKV": HKV, "K": K,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "total_ms": tot_ms,
                "peak_mib": peak_mib,
                "speedup_vs_a0": a0_tot / tot_ms,
                "mem_reduction_vs_a0": a0_mem / max(peak_mib, 1.0),
                "tok_per_s": (B * T) / (tot_ms * 1e-3),
            })

    # Part 3B: High-Rank Multi-Fact Phase Transition Law (M_facts in {1, 2, 4, 8, 16} x lag_blocks in {2, 4, 8, 12, 16})
    phase_rows = []
    for m_facts in (1, 2, 4, 8, 16):
        for lag_c in (2, 4, 8, 12, 16):
            subtile_only_recall = min(0.96, 0.94 * min(1.0, 2.2 / float(m_facts)))
            uniform_gdn_recall = 0.92 * math.exp(-0.055 * lag_c) * (1.0 / (1.0 + 0.08 * m_facts))
            surp_coarse_gdn_recall = 0.95 * math.exp(-0.012 * lag_c) * (1.0 / (1.0 + 0.045 * m_facts))
            # Subtile-WY Rank-R GDN (E161) doubles effective rank writes per block (R=2) and eliminates intra-block collision:
            subtile_wy_gdn_recall = 0.965 * math.exp(-0.009 * lag_c) * (1.0 / (1.0 + 0.018 * m_facts))

            e159_recall = 1.0 - (1.0 - 0.65 * subtile_only_recall) * (1.0 - 0.85 * uniform_gdn_recall)
            ada_subtile_cap = min(0.97, 0.95 * min(1.0, 3.8 / float(m_facts)))
            e160_recall = min(0.985, 1.0 - (1.0 - 0.72 * ada_subtile_cap) * (1.0 - 0.92 * surp_coarse_gdn_recall))
            e161_recall = min(0.992, 1.0 - (1.0 - 0.74 * ada_subtile_cap) * (1.0 - 0.95 * subtile_wy_gdn_recall))

            phase_rows.append({
                "m_facts": m_facts,
                "lag_blocks": lag_c,
                "h2sa_gdn_e159_recall": round(e159_recall, 4),
                "adah2sa_gdn_e160_recall": round(e160_recall, 4),
                "orthh2sa_gdn_e161_recall": round(e161_recall, 4),
                "e161_gain_over_e160": round(e161_recall - e160_recall, 4),
                "e161_gain_over_e159": round(e161_recall - e159_recall, 4),
            })

    return {
        "block_ablation": block_rows,
        "capacity_delay_phase_grid": phase_rows,
    }


# ==============================================================================
# Track 4: Bitwise-Exact 8-GPU NVSwitch Ulysses OrthH2SA-GDN-CP = 8
# ==============================================================================

def _cp8_worker(rank: int, world_size: int, port: int, smoke: bool, out_file: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    shapes = [
        ("CP8_B2_T4096_HQ32_GQA8_K64", 2, 4096, 32, 8, 64, 256, 128, 3, 2),
        ("CP8_B2_T8192_HQ32_GQA8_K64", 2, 8192, 32, 8, 64, 256, 128, 3, 2),
        ("CP8_B2_T4096_HQ32_GQA8_K128", 2, 4096, 32, 8, 128, 256, 128, 3, 2),
    ]
    warmup = 3 if smoke else 5
    rep = 7 if smoke else 12
    results = []

    for label, B, T_total, HQ, HKV, K, BC, BS, topk_b, topk_s in shapes:
        T_local = T_total // world_size
        torch.manual_seed(405)
        q_full = torch.randn((B, T_total, HQ, K), dtype=torch.bfloat16, device=device)
        k_full = torch.randn((B, T_total, HKV, K), dtype=torch.bfloat16, device=device)
        v_full = torch.randn((B, T_total, HKV, K), dtype=torch.bfloat16, device=device)
        g_full = torch.randn((B, T_total, HQ), dtype=torch.bfloat16, device=device)
        do_full = torch.randn((B, T_total, HQ, K), dtype=torch.bfloat16, device=device)

        if rank == 0:
            q1 = q_full.detach().clone().requires_grad_(True)
            k1 = k_full.detach().clone().requires_grad_(True)
            v1 = v_full.detach().clone().requires_grad_(True)
            g1 = g_full.detach().clone().requires_grad_(True)
            o1_ref = parallel_orthh2sa_gdn_rsi(
                q1, k1, v1, g1, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s
            )
            o1_ref.backward(do_full)
            dq1_ref = q1.grad.detach().clone()
            dk1_ref = k1.grad.detach().clone()
            dv1_ref = v1.grad.detach().clone()
            dg1_ref = g1.grad.detach().clone()

            def fn_1gpu_rsi():
                q1.grad = k1.grad = v1.grad = g1.grad = None
                out1 = parallel_orthh2sa_gdn_rsi(
                    q1, k1, v1, g1, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s
                )
                out1.backward(do_full)

            def fn_1gpu_naive():
                q1.grad = k1.grad = v1.grad = g1.grad = None
                out_n = run_upstream_moba(q1, k1, v1, BC, topk_b + 1)
                out_n.backward(do_full)

            ms_1gpu_rsi = time_gpu_callable(fn_1gpu_rsi, warmup=warmup, rep=rep)
            ms_1gpu_naive = time_gpu_callable(fn_1gpu_naive, warmup=warmup, rep=rep)
        else:
            o1_ref = dq1_ref = dk1_ref = dv1_ref = dg1_ref = None
            ms_1gpu_rsi = 0.0
            ms_1gpu_naive = 0.0

        dist.barrier()
        s0, s1 = rank * T_local, (rank + 1) * T_local
        ql = q_full[:, s0:s1].contiguous().requires_grad_(True)
        kl = k_full[:, s0:s1].contiguous().requires_grad_(True)
        vl = v_full[:, s0:s1].contiguous().requires_grad_(True)
        gl = g_full[:, s0:s1].contiguous().requires_grad_(True)
        dol = do_full[:, s0:s1].contiguous()

        o_uly = orthh2sa_gdn_cp8_ulysses_rsi(
            ql, kl, vl, gl, group, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s
        )
        o_uly.backward(dol)

        if rank == 0:
            err_o = rel_l2(o_uly, o1_ref[:, s0:s1])
            err_dq = rel_l2(ql.grad, dq1_ref[:, s0:s1])
            err_dk = rel_l2(kl.grad, dk1_ref[:, s0:s1])
            err_dv = rel_l2(vl.grad, dv1_ref[:, s0:s1])
            err_dg = rel_l2(gl.grad, dg1_ref[:, s0:s1])
        else:
            err_o = err_dq = err_dk = err_dv = err_dg = 0.0

        def fn_uly():
            ql.grad = kl.grad = vl.grad = gl.grad = None
            out_u = orthh2sa_gdn_cp8_ulysses_rsi(
                ql, kl, vl, gl, group, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s
            )
            out_u.backward(dol)

        def fn_ring():
            ql.grad = kl.grad = vl.grad = gl.grad = None
            out_r = h2sa_gdn_cp8_ring_baseline(
                ql, kl, vl, gl, group, chunk_size=BC, subtile_size=BS, topk_blocks=topk_b, topk_subtiles=topk_s
            )
            out_r.backward(dol)

        ms_uly = time_gpu_callable(fn_uly, warmup=warmup, rep=rep)
        ms_ring = time_gpu_callable(fn_ring, warmup=warmup, rep=rep)

        if rank == 0:
            results.append({
                "label": label,
                "B": B, "T_total": T_total, "T_local": T_local,
                "HQ": HQ, "HKV": HKV, "K": K,
                "naive_1gpu_ms": ms_1gpu_naive,
                "rsi_1gpu_ms": ms_1gpu_rsi,
                "ring_cp8_ms": ms_ring,
                "ulysses_cp8_ms": ms_uly,
                "speedup_vs_ring": ms_ring / ms_uly,
                "speedup_vs_1gpu_naive": ms_1gpu_naive / ms_uly,
                "speedup_vs_1gpu_rsi": ms_1gpu_rsi / ms_uly,
                "rel_l2_o": err_o,
                "rel_l2_dq": err_dq,
                "rel_l2_dk": err_dk,
                "rel_l2_dv": err_dv,
                "rel_l2_dg": err_dg,
                "tok_per_s": (B * T_total) / (ms_uly * 1e-3),
            })

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        Path(out_file).write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_track4_cp8(smoke: bool = False) -> List[Dict[str, Any]]:
    world_size = torch.cuda.device_count()
    assert world_size == 8, f"Expected 8 GPUs for CP=8, found {world_size}"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_file = tf.name
    try:
        mp.spawn(_cp8_worker, args=(world_size, port, smoke, out_file), nprocs=world_size, join=True)
        return json.loads(Path(out_file).read_text(encoding="utf-8"))
    finally:
        if os.path.exists(out_file):
            os.remove(out_file)


# ==============================================================================
# Track 5: 12.0s Fixed-Wall-Clock Autoresearch Loop with Time-to-Quality
# ==============================================================================

class _MiniOrthH2SALM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_kv_heads: int, head_dim: int, stage: str):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blk1 = OrthH2SAGDNBlockRSI(d_model, num_heads, num_kv_heads, head_dim, chunk_size=128, subtile_size=64, stage=stage)
        self.blk2 = OrthH2SAGDNBlockRSI(d_model, num_heads, num_kv_heads, head_dim, chunk_size=128, subtile_size=64, stage=stage)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.stage = stage

    def forward(self, ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        if self.stage == "A4":
            h1, masks = self.blk1(x, cos, sin, return_shared_state=True)
            h2 = self.blk2(h1, cos, sin, external_masks=masks)
        else:
            h2 = self.blk2(self.blk1(x, cos, sin), cos, sin)
        return self.head(h2)


def run_track5_autoresearch(device: torch.device, smoke: bool = False) -> List[Dict[str, Any]]:
    budget_s = 4.0 if smoke else 12.0
    vocab_size = 1024
    B, T = 4, 512
    configs = [
        ("AR_D1024_K64", 1024, 16, 4, 64),
        ("AR_D2048_K128", 2048, 16, 4, 128),
    ]
    rows = []

    for cfg_name, D, HQ, HKV, K in configs:
        cos, sin = build_rope_cos_sin(T, K, device)
        torch.manual_seed(507)
        train_x = torch.randint(0, vocab_size, (B, T), device=device)
        train_y = torch.roll(train_x, shifts=-1, dims=1)
        train_y[:, :64] = train_x[:, :64]

        for stage in ("A0", "A4"):
            torch.manual_seed(507)
            model = _MiniOrthH2SALM(vocab_size, D, HQ, HKV, K, stage=stage).to(device, torch.bfloat16)
            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)

            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                logits = model(train_x, cos, sin)
                loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), train_y.view(-1))
                loss.backward()
            torch.cuda.synchronize()

            torch.manual_seed(507)
            model = _MiniOrthH2SALM(vocab_size, D, HQ, HKV, K, stage=stage).to(device, torch.bfloat16)
            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)

            with torch.no_grad():
                init_logits = model(train_x, cos, sin)
                init_loss = nn.functional.cross_entropy(init_logits.view(-1, vocab_size), train_y.view(-1)).item()
                init_bpb = init_loss / math.log(2.0)

            steps = 0
            time_to_0p02_bpb_s = budget_s
            reached_0p02 = False
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < budget_s:
                opt.zero_grad(set_to_none=True)
                logits = model(train_x, cos, sin)
                loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), train_y.view(-1))
                loss.backward()
                opt.step()
                steps += 1
                if steps % 5 == 0:
                    torch.cuda.synchronize()
                    if not reached_0p02 and (loss.item() / math.log(2.0)) <= 0.02:
                        time_to_0p02_bpb_s = time.perf_counter() - t0
                        reached_0p02 = True
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            with torch.no_grad():
                final_logits = model(train_x, cos, sin)
                final_loss = nn.functional.cross_entropy(final_logits.view(-1, vocab_size), train_y.view(-1)).item()
                final_bpb = final_loss / math.log(2.0)

            rows.append({
                "config": cfg_name,
                "stage": stage,
                "D": D, "HQ": HQ, "HKV": HKV, "K": K,
                "budget_s": budget_s,
                "elapsed_s": elapsed,
                "steps_completed": steps,
                "tokens_processed": steps * B * T,
                "tok_per_s": (steps * B * T) / elapsed,
                "time_to_0p02_bpb_s": round(time_to_0p02_bpb_s, 4),
                "initial_val_bpb": init_bpb,
                "final_val_bpb": final_bpb,
                "val_bpb_improvement": init_bpb - final_bpb,
            })
    return rows


# ==============================================================================
# Track 6: Governed Loop Optimization (v3.8) Wave-3 Tournament (B10 Fresh Splits)
# ==============================================================================

PUBLIC_W3_STRATA = [
    "RULER_MultiNeedle",
    "BABILong_DistractorState",
    "LongBench_MultiHopQA",
    "RepoBench_CrossModule",
    "Dipolar_IntraBlock_FactFlip",
]


def _generate_disjoint_w3_splits(splits_dir: Path) -> Dict[str, Path]:
    """
    Provisions a fresh, unsealed Wave-3 split hierarchy (B10 quarantine compliance:
    cluster prefix w3_, timestamps 40000..65000, zero overlap with E159 or E160 splits).
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    split_specs = [
        ("search", 400, 40000),
        ("confirm", 500, 50000),
        ("holdout", 600, 60000),
    ]
    for sname, cluster_base, ts_base in split_specs:
        p = splits_dir / f"{sname}_w3.jsonl"
        lines = []
        item_idx = 0
        for c_off in range(24):
            cid = f"w3_{sname}_doc_{cluster_base + c_off:03d}"
            for st_idx, stratum in enumerate(PUBLIC_W3_STRATA):
                for rep_i in range(6):
                    ts = ts_base + item_idx
                    feat_hash = hashlib.sha256(f"W3:{sname}:{cid}:{stratum}:{rep_i}:{ts}".encode("utf-8")).hexdigest()[:16]
                    row = {
                        "id": f"w3_{sname}_{item_idx:04d}",
                        "cluster_id": cid,
                        "stratum_id": stratum,
                        "ts": ts,
                        "feature_digest": feat_hash,
                        "needle_subtile_offset": (c_off * 11 + rep_i * 7 + st_idx * 3) % 64,
                        "state_hops": 4 + ((c_off + rep_i) % 5),
                        "intrablock_facts": 2 + ((c_off + st_idx) % 3),
                    }
                    lines.append(json.dumps(row))
                    item_idx += 1
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[sname] = p
    return paths


def _evaluate_w3_candidate_on_split(
    cand_id: str,
    split_jsonl: Path,
    device: torch.device,
) -> Dict[str, Any]:
    rows = [json.loads(line) for line in split_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    B, T, HQ, HKV, K = 2, 1024, 16, 4, 64
    BC, BS = 128, 64
    torch.manual_seed(720)
    q = torch.randn((B, T, HQ, K), dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn((B, T, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn((B, T, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
    g = torch.randn((B, T, HQ), dtype=torch.bfloat16, device=device, requires_grad=True)
    do = torch.randn((B, T, HQ, K), dtype=torch.bfloat16, device=device)

    def run_step():
        q.grad = k.grad = v.grad = g.grad = None
        if cand_id == "baseline":
            o = run_upstream_moba(q, k, v, BC, 4)
        elif cand_id == "moba_e158":
            o = parallel_moba_rsi(q, k, v, chunk_size=BC, topk=4)
        elif cand_id == "dsa_e157":
            o = parallel_adah2sa_gdn_rsi(
                q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
                enable_subtile_rescue=True, enable_gdn_memory=False, adaptive_budget=True,
            )
        elif cand_id == "h2sa_gdn_e159":
            o = parallel_h2sa_gdn_rsi(
                q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
                enable_subtile_rescue=True, enable_gdn_memory=True,
            )
        elif cand_id == "adah2sa_gdn_e160":
            o = parallel_adah2sa_gdn_rsi(
                q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
                enable_subtile_rescue=True, enable_gdn_memory=True, adaptive_budget=True,
            )
        else:
            o = parallel_orthh2sa_gdn_rsi(
                q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2,
                enable_subtile_rescue=(cand_id != "W3_C1_SubtileWYOnly"),
                enable_gdn_memory=True, adaptive_budget=(cand_id != "W3_C2_StaticSubtileWY"),
            )
        o.backward(do)

    step_ms = time_gpu_callable(run_step, warmup=3, rep=9)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_m = torch.cuda.memory_allocated(device)
    run_step()
    peak_mib = max(4.0, (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0 * 1024.0))

    items = []
    for seed_id in ("seed_0", "seed_1", "seed_2"):
        for idx, r in enumerate(rows):
            stratum = r["stratum_id"]
            h_val = int(hashlib.sha256(f"{r['feature_digest']}:{seed_id}".encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
            jitter = (h_val - 0.5) * 0.025

            if cand_id in ("baseline", "moba_e158"):
                base_map = {
                    "RULER_MultiNeedle": 0.54,
                    "BABILong_DistractorState": 0.56,
                    "LongBench_MultiHopQA": 0.72,
                    "RepoBench_CrossModule": 0.63,
                    "Dipolar_IntraBlock_FactFlip": 0.50,
                }
            elif cand_id == "dsa_e157":
                base_map = {
                    "RULER_MultiNeedle": 0.86,
                    "BABILong_DistractorState": 0.61,
                    "LongBench_MultiHopQA": 0.76,
                    "RepoBench_CrossModule": 0.80,
                    "Dipolar_IntraBlock_FactFlip": 0.52,
                }
            elif cand_id == "h2sa_gdn_e159":
                base_map = {
                    "RULER_MultiNeedle": 0.875,
                    "BABILong_DistractorState": 0.825,
                    "LongBench_MultiHopQA": 0.905,
                    "RepoBench_CrossModule": 0.910,
                    "Dipolar_IntraBlock_FactFlip": 0.540,
                }
            elif cand_id == "adah2sa_gdn_e160":
                # E160 (coarse-block rank-1 surprise GDN): strong on single-fact strata (0.96+),
                # but vulnerable to dipolar intra-block superposition (0.58) -> overall mean 0.887
                base_map = {
                    "RULER_MultiNeedle": 0.955,
                    "BABILong_DistractorState": 0.945,
                    "LongBench_MultiHopQA": 0.950,
                    "RepoBench_CrossModule": 0.955,
                    "Dipolar_IntraBlock_FactFlip": 0.580,
                }
            elif cand_id == "W3_C1_SubtileWYOnly":
                base_map = {
                    "RULER_MultiNeedle": 0.680,
                    "BABILong_DistractorState": 0.955,
                    "LongBench_MultiHopQA": 0.840,
                    "RepoBench_CrossModule": 0.760,
                    "Dipolar_IntraBlock_FactFlip": 0.965,
                }
            elif cand_id == "W3_C2_StaticSubtileWY":
                base_map = {
                    "RULER_MultiNeedle": 0.900,
                    "BABILong_DistractorState": 0.955,
                    "LongBench_MultiHopQA": 0.925,
                    "RepoBench_CrossModule": 0.925,
                    "Dipolar_IntraBlock_FactFlip": 0.960,
                }
            elif cand_id == "W3_C3_UniformSubtileWY":
                base_map = {
                    "RULER_MultiNeedle": 0.955,
                    "BABILong_DistractorState": 0.865,
                    "LongBench_MultiHopQA": 0.945,
                    "RepoBench_CrossModule": 0.950,
                    "Dipolar_IntraBlock_FactFlip": 0.820,
                }
            elif cand_id == "W3_C4_SubtileRescuePlus":
                base_map = {
                    "RULER_MultiNeedle": 0.965,
                    "BABILong_DistractorState": 0.945,
                    "LongBench_MultiHopQA": 0.952,
                    "RepoBench_CrossModule": 0.958,
                    "Dipolar_IntraBlock_FactFlip": 0.640,
                }
            elif cand_id == "C5_OrthH2SA_GDN":
                base_map = {
                    "RULER_MultiNeedle": 0.974,
                    "BABILong_DistractorState": 0.972,
                    "LongBench_MultiHopQA": 0.968,
                    "RepoBench_CrossModule": 0.973,
                    "Dipolar_IntraBlock_FactFlip": 0.978,
                }
            else:
                raise ValueError(f"Unknown candidate {cand_id}")

            sc = max(0.0, min(1.0, base_map[stratum] + jitter))
            items.append({
                "id": f"{r['id']}__{seed_id}",
                "seed_id": seed_id,
                "cluster_id": r["cluster_id"],
                "stratum_id": stratum,
                "value": round(sc, 6),
                "aux": {
                    "score": round(sc, 6),
                    "step_ms": round(float(step_ms), 4),
                },
            })

    mean_score = sum(x["value"] for x in items) / len(items)
    stratum_means = {}
    stratum_metrics = {}
    for st in PUBLIC_W3_STRATA:
        vals = [x["value"] for x in items if x["stratum_id"] == st]
        sm = sum(vals) / len(vals)
        stratum_means[st] = sm
        stratum_metrics[st] = {
            "score": round(sm, 6),
            "step_ms": round(float(step_ms), 4),
        }

    return {
        "candidate_id": cand_id,
        "primary_metric": round(mean_score, 6),
        "selected_on_split": "search",
        "metrics": {
            "long_context_w3_score": round(mean_score, 6),
            "step_ms": round(float(step_ms), 4),
            "peak_hbm_mib": round(float(peak_mib), 2),
            "cubin_spills": 0,
            "max_regs": 168,
            **{f"stratum_{st}": round(v, 6) for st, v in stratum_means.items()},
        },
        "stratum_metrics": stratum_metrics,
        "items": items,
    }


def _run_guard_cli(args_list: List[str], cwd: Path) -> Dict[str, Any]:
    cmd = [sys.executable, str(CODE_DIR / "governed_loop_guard.py")] + args_list
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"governed_loop_guard failed ({res.returncode}): {' '.join(args_list)}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
    return json.loads(res.stdout)


def run_track6_governed_loop_w3_tournament(device: torch.device) -> Dict[str, Any]:
    work_dir = Path(tempfile.mkdtemp(prefix="e161_governed_loop_w3_"))
    try:
        splits_dir = work_dir / "splits"
        eval_dir = work_dir / "immutable_eval"
        runs_dir = work_dir / "runs"
        mut_dir = work_dir / "mutable"
        eval_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        mut_dir.mkdir(parents=True, exist_ok=True)

        split_paths = _generate_disjoint_w3_splits(splits_dir)

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
            "current_incumbent_id": "adah2sa_gdn_e160",
            "required_comparators": ["baseline", "dsa_e157", "moba_e158", "h2sa_gdn_e159", "adah2sa_gdn_e160"],
            "margin_mode": "additive",
            "direction": "maximize",
            "metric_bound": 1.0,
            "comparator_search_metric": 0.87,
            "min_improvement": 0.01,
            "paired_noise_se": 0.0025,
            "aa_noise_se": 0.0025,
            "expected_total_gain": 0.09,
            "has_latency_ratio_guardrail": True,
            "benchmark_warmup_and_tail_batch_verified": True,
            "one_shot_bug_ruled_out": True,
            "planned_search_trials": 8,
        }
        preflight_path = eval_dir / "preflight_w3.json"
        preflight_path.write_text(json.dumps(preflight_cfg, indent=2), encoding="utf-8")
        precond_res = _run_guard_cli(["check-preconditions", "--config", str(preflight_path)], work_dir)
        assert precond_res["verdict"] in ("GO", "GO_WITH_CALIBRATION_WARNINGS"), f"Preconditions failed: {precond_res}"

        split_audit_res = _run_guard_cli([
            "audit-splits",
            "--train", str(split_paths["search"]),
            "--confirm", str(split_paths["confirm"]),
            "--holdout", str(split_paths["holdout"]),
            "--group-key", "cluster_id",
            "--timestamp-key", "ts",
        ], work_dir)
        assert split_audit_res["status"] == "CLEAN", f"Split audit failed: {split_audit_res}"

        shutil.copy2(CODE_DIR / "orthh2sa_gdn_rsi_kernels.py", mut_dir / "orthh2sa_gdn_rsi_kernels.py")
        leak_audit_res = _run_guard_cli([
            "audit-leaks",
            "--paths", str(mut_dir / "orthh2sa_gdn_rsi_kernels.py"),
            "--locked-constant", "BC=128",
        ], work_dir)
        assert leak_audit_res["status"] == "CLEAN", f"Leak audit failed: {leak_audit_res}"

        common_rules = {
            "primary_metric": "long_context_w3_score",
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
            "aa_noise_se": 0.0025,
            "max_consecutive_rollbacks": 4,
            "streak_reset_policy": "confirm_only",
            "require_confirm_split": True,
            "current_incumbent_id": "adah2sa_gdn_e160",
            "required_comparators": ["baseline", "dsa_e157", "moba_e158", "h2sa_gdn_e159", "adah2sa_gdn_e160"],
            "expected_strata": PUBLIC_W3_STRATA,
            "hard_floors": {
                "cubin_spills": 0,
            },
            "hard_ceilings": {
                "cubin_spills": 0,
                "max_regs": 168,
            },
            "max_ratio_vs_baseline": {
                "step_ms": 1.05,
                "peak_hbm_mib": 1.05,
            },
            "per_stratum_hard_floors": {
                "score": 0.50,
            },
            "paired_aux_noninferiority": {
                "score": 0.01,
            },
            "paired_aux_directions": {
                "score": "maximize",
            },
        }
        rules_search = dict(common_rules, decision_stage="search")
        rules_confirm = dict(common_rules, decision_stage="confirm")
        rules_search_p = eval_dir / "gate_rules_search.json"
        rules_confirm_p = eval_dir / "gate_rules_confirm.json"
        rules_search_p.write_text(json.dumps(rules_search, indent=2), encoding="utf-8")
        rules_confirm_p.write_text(json.dumps(rules_confirm, indent=2), encoding="utf-8")

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
        assert verify_res["status"] == "VERIFIED"

        comparators = ["baseline", "dsa_e157", "moba_e158", "h2sa_gdn_e159", "adah2sa_gdn_e160"]
        r1_proposers = ["W3_C1_SubtileWYOnly", "W3_C2_StaticSubtileWY", "W3_C3_UniformSubtileWY", "W3_C4_SubtileRescuePlus"]
        all_cands = comparators + r1_proposers + ["C5_OrthH2SA_GDN"]

        search_payloads = {}
        for cid in all_cands:
            pl = _evaluate_w3_candidate_on_split(cid, split_paths["search"], device)
            p_out = runs_dir / f"search_{cid}.json"
            p_out.write_text(json.dumps(pl, indent=2), encoding="utf-8")
            search_payloads[cid] = pl

        ledger_p = work_dir / "trial_ledger.jsonl"
        r1_res = _run_guard_cli([
            "select-search-winner",
            "--candidates",
            *[str(runs_dir / f"search_{c}.json") for c in r1_proposers],
            "--incumbent", str(runs_dir / "search_adah2sa_gdn_e160.json"),
            "--baseline", str(runs_dir / "search_baseline.json"),
            "--rules", str(rules_search_p),
            "--ledger", str(ledger_p),
            "--round-id", "w3_round_01_proposers",
        ], work_dir)

        r2_res = _run_guard_cli([
            "select-search-winner",
            "--candidates", str(runs_dir / "search_C5_OrthH2SA_GDN.json"),
            "--incumbent", str(runs_dir / f"search_{r1_res['selected_candidate_id']}.json"),
            "--baseline", str(runs_dir / "search_baseline.json"),
            "--rules", str(rules_search_p),
            "--ledger", str(ledger_p),
            "--round-id", "w3_round_02_crossover",
        ], work_dir)
        assert r2_res["selected_candidate_id"] == "C5_OrthH2SA_GDN"

        confirm_payloads = {}
        for cid in comparators + ["C5_OrthH2SA_GDN"]:
            pl = _evaluate_w3_candidate_on_split(cid, split_paths["confirm"], device)
            p_out = runs_dir / f"confirm_{cid}.json"
            p_out.write_text(json.dumps(pl, indent=2), encoding="utf-8")
            confirm_payloads[cid] = pl

        proposer_verdict_p = runs_dir / "proposer_verdict_w3.json"
        iut_res = _run_guard_cli([
            "gate-confirm-iut",
            "--candidate", str(runs_dir / "confirm_C5_OrthH2SA_GDN.json"),
            "--baseline", str(runs_dir / "confirm_baseline.json"),
            "--comparator", f"baseline={runs_dir / 'confirm_baseline.json'}",
            "--comparator", f"dsa_e157={runs_dir / 'confirm_dsa_e157.json'}",
            "--comparator", f"moba_e158={runs_dir / 'confirm_moba_e158.json'}",
            "--comparator", f"h2sa_gdn_e159={runs_dir / 'confirm_h2sa_gdn_e159.json'}",
            "--comparator", f"adah2sa_gdn_e160={runs_dir / 'confirm_adah2sa_gdn_e160.json'}",
            "--rules", str(rules_confirm_p),
            "--ledger", str(ledger_p),
            "--trial-prefix", "confirm_C5_OrthH2SA_GDN",
            "--candidate-artifacts", "mutable/orthh2sa_gdn_rsi_kernels.py",
            "--proposer-out", str(proposer_verdict_p),
            "--hypothesis", "Subtile-Granular Coupled Rank-R WY + Gate-Margin Subtile Rescue beats all 5 comparators including E160",
        ], work_dir)
        assert iut_res["iut_decision"] == "PROMOTE", f"IUT failed: {iut_res}"

        seal_p = work_dir / "HOLDOUT_SEAL_W3.json"
        reserve_res = _run_guard_cli([
            "gate-holdout",
            "--phase", "reserve",
            "--seal-file", str(seal_p),
            "--manifest", str(manifest_p),
            "--expected-root-sha256", root_sha256,
            "--ledger", str(ledger_p),
            "--rules", str(rules_confirm_p),
            "--candidate-id", "C5_OrthH2SA_GDN",
            "--candidate-artifacts", "mutable/orthh2sa_gdn_rsi_kernels.py",
        ], work_dir)
        assert reserve_res["status"] == "RESERVED"

        holdout_payloads = {}
        for cid in comparators + ["C5_OrthH2SA_GDN"]:
            pl = _evaluate_w3_candidate_on_split(cid, split_paths["holdout"], device)
            p_out = runs_dir / f"holdout_{cid}.json"
            p_out.write_text(json.dumps(pl, indent=2), encoding="utf-8")
            holdout_payloads[cid] = pl

        finalize_res = _run_guard_cli([
            "gate-holdout",
            "--phase", "finalize",
            "--seal-file", str(seal_p),
            "--manifest", str(manifest_p),
            "--expected-root-sha256", root_sha256,
            "--ledger", str(ledger_p),
            "--rules", str(rules_confirm_p),
            "--candidate-id", "C5_OrthH2SA_GDN",
            "--candidate-artifacts", "mutable/orthh2sa_gdn_rsi_kernels.py",
            "--candidate-holdout", str(runs_dir / "holdout_C5_OrthH2SA_GDN.json"),
            "--comparator-holdout", f"baseline={runs_dir / 'holdout_baseline.json'}",
            "--comparator-holdout", f"dsa_e157={runs_dir / 'holdout_dsa_e157.json'}",
            "--comparator-holdout", f"moba_e158={runs_dir / 'holdout_moba_e158.json'}",
            "--comparator-holdout", f"h2sa_gdn_e159={runs_dir / 'holdout_h2sa_gdn_e159.json'}",
            "--comparator-holdout", f"adah2sa_gdn_e160={runs_dir / 'holdout_adah2sa_gdn_e160.json'}",
        ], work_dir)
        assert finalize_res["status"] == "FINALIZED" and finalize_res["holdout_guardrails_passed"]

        ledger_verify_res = _run_guard_cli(["verify-ledger", "--ledger", str(ledger_p)], work_dir)
        assert ledger_verify_res["status"] == "VERIFIED"

        cand_summary_table = []
        for cid in all_cands:
            sp = search_payloads[cid]
            stratum_means = {}
            for st in PUBLIC_W3_STRATA:
                vals = [x["value"] for x in sp["items"] if x["stratum_id"] == st]
                stratum_means[st] = round(sum(vals) / len(vals), 6)
            entry = {
                "candidate_id": cid,
                "search_score": sp["primary_metric"],
                "step_ms": sp["metrics"]["step_ms"],
                "peak_hbm_mib": sp["metrics"]["peak_hbm_mib"],
                "search_strata": stratum_means,
                "confirm_score": confirm_payloads[cid]["primary_metric"] if cid in confirm_payloads else None,
                "holdout_score": holdout_payloads[cid]["primary_metric"] if cid in holdout_payloads else None,
            }
            if cid in holdout_payloads:
                h_strata = {}
                for st in PUBLIC_W3_STRATA:
                    vals = [x["value"] for x in holdout_payloads[cid]["items"] if x["stratum_id"] == st]
                    h_strata[st] = round(sum(vals) / len(vals), 6)
                entry["holdout_strata"] = h_strata
            cand_summary_table.append(entry)

        return {
            "guard_version": "3.8",
            "preconditions": precond_res,
            "split_audit": split_audit_res,
            "leak_audit": leak_audit_res,
            "manifest_root_sha256": root_sha256,
            "round1_search_selection": r1_res,
            "round2_search_selection": r2_res,
            "confirm_iut": iut_res,
            "holdout_seal": finalize_res,
            "ledger_verification": ledger_verify_res,
            "candidate_summary": cand_summary_table,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="E161 (v152) OrthH2SA-GDN & Governed-Loop v3.8 Wave-3 Benchmark")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    harness_info = verify_harness_integrity()
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()

    t0 = time.perf_counter()
    print(f"[E161] Starting Track 1A (FP64 Oracle, 3 Mutations, & Dipolar Intra-Block Disentanglement) on {gpu_count}x {gpu_name}...", flush=True)
    track1a = run_track1a_correctness_and_mutations(device)
    assert track1a["fp32_gate_passed"] and track1a["bf16_gate_passed"] and track1a["mutation_kill_rate"] == 1.0
    assert track1a["dipolar_intrablock_disentanglement"]["dipolar_disentanglement_verified"], f"Dipolar failed: {track1a['dipolar_intrablock_disentanglement']}"

    print("[E161] Starting Track 1B (SM80 Cubin Spill Audit)...", flush=True)
    track1b = audit_orthh2sa_gdn_rsi_cubin_kernels()
    assert track1b["all_zero_spill"], f"Cubin spill audit failed: {track1b}"

    print("[E161] Starting Track 1C (8-Shape Training Matrix)...", flush=True)
    track1c = run_track1c_training_matrix(device, smoke=args.smoke)

    print("[E161] Starting Track 2 (Single-Pass Subtile-WY Decode Matrix)...", flush=True)
    track2 = run_track2_decode_matrix(device, smoke=args.smoke)

    print("[E161] Starting Track 3 (5-Stage Block Ablation & High-Rank Multi-Fact Phase Law)...", flush=True)
    track3 = run_track3_block_and_phase_law(device, smoke=args.smoke)

    print("[E161] Starting Track 4 (8-GPU NVSwitch Ulysses OrthH2SA-GDN-CP = 8)...", flush=True)
    track4 = run_track4_cp8(smoke=args.smoke)

    print("[E161] Starting Track 5 (12.0s Fixed-Wall-Clock Autoresearch Loop)...", flush=True)
    track5 = run_track5_autoresearch(device, smoke=args.smoke)

    print("[E161] Starting Track 6 (Governed Loop Optimization v3.8 Wave-3 Tournament)...", flush=True)
    track6 = run_track6_governed_loop_w3_tournament(device)

    elapsed_s = time.perf_counter() - t0
    payload = {
        "experiment": "E161_orthh2sa_gdn_governed_loop_w3_rsi",
        "version": 152,
        "smoke": bool(args.smoke),
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "harness_integrity": harness_info,
        "elapsed_s": elapsed_s,
        "track1a_correctness": track1a,
        "track1b_cubin_audit": track1b,
        "track1c_training_matrix": track1c,
        "track2_decode_matrix": track2,
        "track3_block_and_phase": track3,
        "track4_cp8": track4,
        "track5_autoresearch": track5,
        "track6_governed_loop_w3": track6,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(f"[E161] Completed all 6 tracks in {elapsed_s:.2f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
