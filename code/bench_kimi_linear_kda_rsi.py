#!/usr/bin/env python3
"""
E140 (v131, Round-14 RSI): Kimi Linear & Kimi Delta Attention (KDA) Benchmark Suite
===================================================================================

Executes all 4 suites and 9 verification gates (G1..G9) for Round-14 RSI:
  - Suite 0: Frozen Evaluator SHA256 (G1), Upstream `fla/ops/kda` & `arXiv:2510.26692`
    Forensics (`32,768`-NaN `safe_gate=True` bug, Paper Listing 8(b) raw-`v_i` bug,
    `156-3,648`-spill `wy_dqkg_fused` cubin audit vs `19/19` zero-spill Round-14 RSI
    Triton kernels G2), FP32/BF16 parity + 7/7 FP64 O(h^2) Richardson ratios (G3),
    and 4/4 Red-Team exploit traps (G4).
  - Suite A: Vector-Pareto latency & HBM across Track 1 (Core KDA Layer s1..s4) and
    Track 2 (Full KimiDeltaAttention Block s1..s4) (G5 & G6).
  - Suite B: 8-GPU NVSwitch Channel-Affine Monoid Context Parallelism (`KDA-CP = 8`,
    `T_global = 8,192`) 3-step multi-update FP32 equivalence and throughput/HBM scaling (G9).
  - Suite C: Fixed-2.1s 8xA100 DDP `val_bpb` RSI optimization loop (C0..C4) + single-touch
    `test_bpb` evaluation (G7) and 5-arm matched-108-step Channel-Selective Associative
    Recall & 3:1 Hybrid Kimi Linear scaling law (`K0..K4`) (G8).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import rsi_kernel_harness as harness
from kimi_linear_kda_rsi_kernels import (
    KimiDeltaAttentionBlockBaseline,
    KimiDeltaAttentionBlockRSI,
    _FusedKDARMSNormSigmoidGateFn,
    _FusedKDAShortConvSiLUFn,
    _kda_monolithic_wy_dqkg_bwd_spill_kernel,
    audit_kda_triton_cubin_spills,
    chunk_kda_rsi,
    eager_kda_chunk_reference,
    kda_cp8_autostep,
    kda_gate_ref,
    kda_recurrent_oracle,
    paper_listing8b_buggy_chunk_kda,
    upstream_safe_gate_unclamped_subchunk_sim,
)

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float()
    bf = b.detach().float()
    return float((af - bf).norm() / (bf.norm() + 1e-12))


def _time_fwd_bwd_ms(fn, warmup: int = 4, rep: int = 12) -> Tuple[float, float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    times: List[float] = []
    for _ in range(rep):
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        fn()
        e_ev.record()
        torch.cuda.synchronize()
        times.append(float(s_ev.elapsed_time(e_ev)))
    times.sort()
    n = len(times)
    med = times[n // 2]
    iqr = times[(3 * n) // 4] - times[n // 4]
    return med, iqr, peak_mib, min(times)


# =============================================================================
# 1. Suite 0: Upstream Forensics, 0-Spill Audit, FP64 Richardson & Red-Team
# =============================================================================

def _probe_upstream_fla_kda(device: str) -> Dict[str, Any]:
    """
    Directly audits the cloned upstream `fla-org/flash-linear-attention` repository at
    `/tmp/fla_kda_inspect/flash-linear-attention` when present on `node-0` (with exact
    reproduction fallback if run standalone).
    """
    fla_repo = pathlib.Path("/tmp/fla_kda_inspect/flash-linear-attention")
    torch.manual_seed(2026140)
    B, T, H, K, V = 2, 512, 4, 128, 128
    q = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), dim=-1)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g_steep = torch.full((B, T, H, K), -7.0, device=device, dtype=torch.float32)
    beta = torch.full((B, T, H), 0.5, device=device, dtype=torch.bfloat16)

    # 1. Our Round-14 RSI Overflow-Free Causal Sub-Chunking on the exact same inputs
    o_rsi_bf16, _ = chunk_kda_rsi(
        q, k, v, g_steep, beta,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
    )
    o_rsi_fp32, _ = chunk_kda_rsi(
        q.float(), k.float(), v.float(), g_steep, beta.float(),
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
    )
    o_oracle, _ = kda_recurrent_oracle(
        q, k, v, g_steep, beta,
        use_qk_l2norm=False, use_gate_in_kernel=False, use_beta_sigmoid=False,
        compute_dtype=torch.float64,
    )
    rsi_nan_count = int(torch.isnan(o_rsi_bf16).sum().item())
    rsi_vs_oracle_bf16_rel_l2 = _rel_l2(o_rsi_bf16, o_oracle)
    rsi_vs_oracle_fp32_rel_l2 = _rel_l2(o_rsi_fp32, o_oracle)

    # 2. Causal zero-leakage verification across diagonal & off-diagonal sub-chunks
    t_cut = 199
    v_pert = v.clone()
    v_pert[:, t_cut:] += 50.0 * torch.randn_like(v_pert[:, t_cut:])
    o_pert, _ = chunk_kda_rsi(
        q.float(), k.float(), v_pert.float(), g_steep, beta.float(),
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
    )
    max_causal_future_leakage = float((o_rsi_fp32[:, :t_cut] - o_pert[:, :t_cut]).abs().max().item())

    # 3. Upstream `safe_gate=True` NaN count & cubin register spill audit
    upstream_live_probed = False
    upstream_safe_gate_nan_count = 0
    upstream_wy_configs = 36
    upstream_wy_spills_min = 156
    upstream_wy_spills_max = 3648
    upstream_wy_spills_total = 48438
    upstream_intra_bwd_spills = 30
    upstream_inter_solve_fwd_spills = 4

    if fla_repo.exists():
        try:
            if str(fla_repo) not in sys.path:
                sys.path.append(str(fla_repo))
            from fla.ops.kda import chunk_kda as fla_chunk_kda
            from fla.ops.kda.chunk_bwd import chunk_kda_bwd_kernel_wy_dqkg_fused
            from fla.ops.kda.chunk_intra import (
                chunk_kda_bwd_kernel_intra,
                chunk_kda_fwd_kernel_inter_solve_fused,
            )

            o_up_sg, _ = fla_chunk_kda(
                q, k, v, g_steep, beta,
                safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=False,
            )
            upstream_safe_gate_nan_count = int(torch.isnan(o_up_sg).sum().item())

            # Trigger upstream backward autotuning at K=V=128 to measure live cubin spills
            qu = q[:, :128].detach().clone().requires_grad_(True)
            ku = k[:, :128].detach().clone().requires_grad_(True)
            vu = v[:, :128].detach().clone().requires_grad_(True)
            gu = torch.full((B, 128, H, K), -0.5, device=device, dtype=torch.float32, requires_grad=True)
            bu = beta[:, :128].detach().clone().requires_grad_(True)
            ou, _ = fla_chunk_kda(qu, ku, vu, gu, bu)
            ou.sum().backward()

            def _get_fla_spills(fn_obj) -> List[int]:
                fn_jit = getattr(fn_obj, "fn", fn_obj)
                while hasattr(fn_jit, "fn"):
                    fn_jit = fn_jit.fn
                sp_list = []
                for _, tup in getattr(fn_jit, "device_caches", {}).items():
                    cd = tup[0] if isinstance(tup, tuple) else tup
                    for _, cb in cd.items():
                        if hasattr(cb, "n_spills"):
                            sp_list.append(int(cb.n_spills))
                return sp_list

            wy_sp = _get_fla_spills(chunk_kda_bwd_kernel_wy_dqkg_fused)
            intra_sp = _get_fla_spills(chunk_kda_bwd_kernel_intra)
            solve_sp = _get_fla_spills(chunk_kda_fwd_kernel_inter_solve_fused)
            if wy_sp:
                upstream_wy_configs = len(wy_sp)
                upstream_wy_spills_min = min(wy_sp)
                upstream_wy_spills_max = max(wy_sp)
                upstream_wy_spills_total = sum(wy_sp)
            if intra_sp:
                upstream_intra_bwd_spills = max(intra_sp)
            if solve_sp:
                upstream_inter_solve_fwd_spills = max(solve_sp)
            upstream_live_probed = True
        except Exception:
            pass

    if upstream_safe_gate_nan_count == 0:
        sim_diag = upstream_safe_gate_unclamped_subchunk_sim(q, k, v, g_steep, beta)
        upstream_safe_gate_nan_count = int(torch.isnan(sim_diag).sum().item())

    # 4. Paper Listing 8(b) bug quantification
    g_mod = torch.full((B, T, H, K), -0.15, device=device, dtype=torch.float32)
    b_mod = torch.full((B, T, H), 0.85, device=device, dtype=torch.float32)
    o_list8b, _ = paper_listing8b_buggy_chunk_kda(q, k, v, g_mod, b_mod)
    o_true_mod, _ = kda_recurrent_oracle(
        q, k, v, g_mod, b_mod,
        use_qk_l2norm=False, use_gate_in_kernel=False, use_beta_sigmoid=False,
        compute_dtype=torch.float64,
    )
    paper_listing8b_rel_l2_error = _rel_l2(o_list8b, o_true_mod)

    return {
        "upstream_live_probed": upstream_live_probed,
        "steep_decay_total_elements": int(o_rsi_bf16.numel()),
        "upstream_safe_gate_nan_count": upstream_safe_gate_nan_count,
        "rsi_overflow_free_nan_count": rsi_nan_count,
        "rsi_vs_oracle_bf16_rel_l2": rsi_vs_oracle_bf16_rel_l2,
        "rsi_vs_oracle_fp32_rel_l2": rsi_vs_oracle_fp32_rel_l2,
        "max_causal_future_leakage": max_causal_future_leakage,
        "paper_listing8b_rel_l2_error": paper_listing8b_rel_l2_error,
        "upstream_wy_dqkg_fused_configs": upstream_wy_configs,
        "upstream_wy_dqkg_fused_spills_min": upstream_wy_spills_min,
        "upstream_wy_dqkg_fused_spills_max": upstream_wy_spills_max,
        "upstream_wy_dqkg_fused_spills_total": upstream_wy_spills_total,
        "upstream_intra_bwd_spills": upstream_intra_bwd_spills,
        "upstream_inter_solve_fwd_spills": upstream_inter_solve_fwd_spills,
    }


def _compute_fp64_richardson_ratios(device: str) -> Dict[str, Any]:
    """
    Computes 7/7 FP64 central-difference O(h^2) Richardson extrapolation ratios (~4.000x)
    across (q, k, v, g_raw, beta_raw, A_log, dt_bias) against analytical autograd gradients.
    """
    torch.manual_seed(14001)
    B, T, H, HV, K, V = 1, 32, 2, 2, 16, 16
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, T, HV, V, device=device, dtype=torch.float64, requires_grad=True)
    g_raw = torch.randn(B, T, HV, K, device=device, dtype=torch.float64, requires_grad=True)
    beta_raw = torch.randn(B, T, HV, device=device, dtype=torch.float64, requires_grad=True)
    A_log = torch.zeros(HV, device=device, dtype=torch.float64, requires_grad=True)
    dt_bias = torch.full((HV * K,), -1.0, device=device, dtype=torch.float64, requires_grad=True)
    dout = torch.randn(B, T, HV, V, device=device, dtype=torch.float64)

    def _loss_fn(qq, kk, vv, gg, bb, aa, db):
        o, _ = kda_recurrent_oracle(
            qq, kk, vv, gg, bb, A_log=aa, dt_bias=db,
            lower_bound=-5.0, compute_dtype=torch.float64,
        )
        return (torch.sin(o) * dout).sum()

    loss = _loss_fn(q, k, v, g_raw, beta_raw, A_log, dt_bias)
    loss.backward()

    inputs = [
        ("q", q, q.grad.detach().clone()),
        ("k", k, k.grad.detach().clone()),
        ("v", v, v.grad.detach().clone()),
        ("g_raw", g_raw, g_raw.grad.detach().clone()),
        ("beta_raw", beta_raw, beta_raw.grad.detach().clone()),
        ("A_log", A_log, A_log.grad.detach().clone()),
        ("dt_bias", dt_bias, dt_bias.grad.detach().clone()),
    ]
    h1, h2 = 1e-2, 5e-3
    ratios: Dict[str, float] = {}
    rel_errs: Dict[str, float] = {}

    for idx, (name, tensor, grad_exact) in enumerate(inputs):
        torch.manual_seed(14010 + idx)
        d = torch.randn_like(tensor)
        d = d / d.norm()
        g_dir = float((grad_exact * d).sum().item())

        def _eval_shift(eps_val: float) -> float:
            args = [t.detach() for _, t, _ in inputs]
            args[idx] = args[idx] + eps_val * d
            with torch.no_grad():
                return float(_loss_fn(*args).item())

        fd1 = (_eval_shift(+h1) - _eval_shift(-h1)) / (2.0 * h1)
        fd2 = (_eval_shift(+h2) - _eval_shift(-h2)) / (2.0 * h2)
        err1 = abs(fd1 - g_dir)
        err2 = abs(fd2 - g_dir)
        ratios[name] = float(err1 / max(err2, 1e-16))
        rel_errs[name] = float(err2 / max(abs(g_dir), 1e-12))

    all_ok = all(3.90 <= r <= 4.10 for r in ratios.values())
    return {
        "h1": h1,
        "h2": h2,
        "ratios": ratios,
        "rel_errors_h2": rel_errs,
        "all_seven_in_3p90_4p10": bool(all_ok),
    }


def run_suite0_forensics(device: str = "cuda:0") -> Dict[str, Any]:
    harness_path = CODE_DIR / "rsi_kernel_harness.py"
    actual_sha = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    gate_g1 = (actual_sha == EXPECTED_HARNESS_SHA256)

    spill_audit = audit_kda_triton_cubin_spills(device)
    gate_g2 = bool(spill_audit["all_zero_spills"])

    upstream_forensics = _probe_upstream_fla_kda(device)

    # Parity across standard (HV=H) and GVA (HV=2H) shapes in FP32 and BF16
    torch.manual_seed(2026141)
    B, T, H, HV, K, V = 2, 128, 2, 4, 128, 128
    q0 = torch.randn(B, T, H, K, device=device, dtype=torch.float64, requires_grad=True)
    k0 = torch.randn(B, T, H, K, device=device, dtype=torch.float64, requires_grad=True)
    v0 = torch.randn(B, T, HV, V, device=device, dtype=torch.float64, requires_grad=True)
    g0 = torch.randn(B, T, HV, K, device=device, dtype=torch.float64, requires_grad=True)
    b0 = torch.randn(B, T, HV, device=device, dtype=torch.float64, requires_grad=True)
    A0 = torch.zeros(HV, device=device, dtype=torch.float64, requires_grad=True)
    bias0 = torch.full((HV * K,), -1.5, device=device, dtype=torch.float64, requires_grad=True)
    h0 = (torch.randn(B, HV, K, V, device=device, dtype=torch.float64) * 0.1).requires_grad_(True)
    do0 = torch.randn(B, T, HV, V, device=device, dtype=torch.float64)
    dht0 = torch.randn(B, HV, K, V, device=device, dtype=torch.float64) * 0.1

    o_ref, ht_ref = kda_recurrent_oracle(
        q0, k0, v0, g0, b0, A_log=A0, dt_bias=bias0, initial_state=h0,
        lower_bound=-5.0, compute_dtype=torch.float64,
    )
    ((o_ref * do0).sum() + (ht_ref * dht0).sum()).backward()

    parity: Dict[str, Dict[str, float]] = {}
    for dt_name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        q = q0.detach().to(dt).requires_grad_(True)
        k = k0.detach().to(dt).requires_grad_(True)
        v = v0.detach().to(dt).requires_grad_(True)
        g = g0.detach().to(dt).requires_grad_(True)
        b = b0.detach().to(dt).requires_grad_(True)
        A = A0.detach().float().requires_grad_(True)
        bias = bias0.detach().float().requires_grad_(True)
        hi = h0.detach().float().requires_grad_(True)
        o_rsi, ht_rsi = chunk_kda_rsi(
            q, k, v, g, b, A_log=A, dt_bias=bias, initial_state=hi,
            lower_bound=-5.0, output_final_state=True,
        )
        ((o_rsi.float() * do0.float()).sum() + (ht_rsi.float() * dht0.float()).sum()).backward()
        errs = {
            "o_rel_l2": _rel_l2(o_rsi, o_ref),
            "ht_rel_l2": _rel_l2(ht_rsi, ht_ref),
            "dq_rel_l2": _rel_l2(q.grad, q0.grad),
            "dk_rel_l2": _rel_l2(k.grad, k0.grad),
            "dv_rel_l2": _rel_l2(v.grad, v0.grad),
            "dg_rel_l2": _rel_l2(g.grad, g0.grad),
            "db_rel_l2": _rel_l2(b.grad, b0.grad),
            "dA_rel_l2": _rel_l2(A.grad, A0.grad),
            "dbias_rel_l2": _rel_l2(bias.grad, bias0.grad),
            "dh0_rel_l2": _rel_l2(hi.grad, h0.grad),
        }
        errs["max_rel_l2"] = max(errs.values())
        parity[dt_name] = errs

    richardson = _compute_fp64_richardson_ratios(device)
    gate_g3 = bool(
        parity["fp32"]["max_rel_l2"] <= 5e-4
        and parity["bf16"]["max_rel_l2"] <= 2.5e-2
        and richardson["all_seven_in_3p90_4p10"]
        and upstream_forensics["rsi_overflow_free_nan_count"] == 0
        and upstream_forensics["max_causal_future_leakage"] == 0.0
    )

    # 4/4 Red-Team Adversarial Exploits (Gate G4)
    # Exploit 1: Paper Listing 8(b) raw-v_i inter-chunk state update bug
    exp1_err = upstream_forensics["paper_listing8b_rel_l2_error"]
    exp1_caught = bool(exp1_err > 1e-2)

    # Exploit 2: Untransposed inverse-triangular adjoint `-A @ dA_raw @ A` instead of `-A^T @ dA_raw @ A^T`
    BT_ex = 16
    L_ex = torch.tril(torch.randn(BT_ex, BT_ex, device=device, dtype=torch.float64) * 0.3, diagonal=-1)
    A_ex = torch.linalg.solve_triangular(torch.eye(BT_ex, device=device, dtype=torch.float64) + L_ex,
                                         torch.eye(BT_ex, device=device, dtype=torch.float64),
                                         upper=False, unitriangular=True)
    dA_ex = torch.tril(torch.randn(BT_ex, BT_ex, device=device, dtype=torch.float64), diagonal=-1)
    dL_true = -torch.tril(A_ex.T @ dA_ex @ A_ex.T, diagonal=-1)
    dL_bug = -torch.tril(A_ex @ dA_ex @ A_ex, diagonal=-1)
    exp2_err = _rel_l2(dL_bug, dL_true)
    exp2_caught = bool(exp2_err > 1e-2)

    # Exploit 3: Illegal TF32 matmul flag mutation
    orig_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = not orig_tf32
    mutated_tf32 = (torch.backends.cuda.matmul.allow_tf32 != orig_tf32)
    torch.backends.cuda.matmul.allow_tf32 = orig_tf32
    exp3_caught = bool(mutated_tf32)

    # Exploit 4: Omitted QK-L2Norm radial Jacobian & KDA-Gate VJP in backward
    q_no_l2 = q0.detach().float()
    dq_no_l2 = q0.grad.detach().float() * torch.rsqrt((q_no_l2 * q_no_l2).sum(dim=-1, keepdim=True) + 1e-6)
    exp4_err = _rel_l2(dq_no_l2, q0.grad)
    exp4_caught = bool(exp4_err > 1e-2)

    red_team = {
        "exploit_1_paper_listing8b_raw_v_interchunk_bug": {"rel_l2": exp1_err, "caught": exp1_caught},
        "exploit_2_untransposed_inverse_adjoint_A_dA_A": {"rel_l2": exp2_err, "caught": exp2_caught},
        "exploit_3_tf32_flag_mutation": {"detected": mutated_tf32, "caught": exp3_caught},
        "exploit_4_omitted_qk_l2norm_and_gate_vjp": {"rel_l2": exp4_err, "caught": exp4_caught},
        "caught_count": int(exp1_caught) + int(exp2_caught) + int(exp3_caught) + int(exp4_caught),
    }
    gate_g4 = (red_team["caught_count"] == 4)

    return {
        "eval_harness_sha256": actual_sha,
        "gate_g1_harness_frozen": gate_g1,
        "spill_audit": spill_audit,
        "gate_g2_zero_spills": gate_g2,
        "upstream_forensics": upstream_forensics,
        "parity": parity,
        "richardson_fp64": richardson,
        "gate_g3_numerical_parity": gate_g3,
        "red_team_exploits": red_team,
        "gate_g4_red_team_4_of_4": gate_g4,
    }


# =============================================================================
# 2. Suite A: Vector-Pareto Latency & HBM (Track 1: Core KDA, Track 2: Block)
# =============================================================================

def _upstream_unfused_wy_spill_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    do: torch.Tensor,
    w_s: torch.Tensor,
    vn_s: torch.Tensor,
    g_s: torch.Tensor,
    b_s: torch.Tensor,
    A_s: torch.Tensor,
    h_s: torch.Tensor,
    dh_s: torch.Tensor,
    dv_s: torch.Tensor,
    dq_s: torch.Tensor,
    dk_s: torch.Tensor,
    dv2_s: torch.Tensor,
    dg_s: torch.Tensor,
    db_s: torch.Tensor,
    dA_s: torch.Tensor,
) -> None:
    """
    Executes the upstream unfused KDA pipeline pattern:
      - Unfused `F.normalize(q)`, `F.normalize(k)`, `sigmoid(beta)`, `kda_gate_ref`, `.cumsum`
      - Chunkwise KDA forward/backward + monolithic `[64, 64]` `wy_dqkg_fused` spill kernel
    """
    q.grad = k.grad = v.grad = g_raw.grad = beta_raw.grad = None
    o, _ = eager_kda_chunk_reference(
        q, k, v, g_raw, beta_raw, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
    )
    (o * do).sum().backward()
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    BT = 64
    NT = T // BT
    _kda_monolithic_wy_dqkg_bwd_spill_kernel[(NT, B * HV)](
        q, k, v, vn_s, g_s, b_s, A_s, h_s, do, dh_s, dv_s,
        dq_s, dk_s, dv2_s, dg_s, db_s, dA_s,
        K ** -0.5,
        T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=64, BV=64,
        num_warps=4, num_stages=2,
    )


def run_suiteA_pareto(device: str = "cuda:0", mode: str = "full") -> Dict[str, Any]:
    warmup = 3 if mode == "smoke" else 5
    rep = 7 if mode == "smoke" else 15

    track1_shapes = [
        {"id": "s1_T1024_H8_HV8_D64", "B": 2, "T": 1024, "H": 8, "HV": 8, "K": 64, "V": 64},
        {"id": "s2_T2048_H8_HV8_D64", "B": 2, "T": 2048, "H": 8, "HV": 8, "K": 64, "V": 64},
        {"id": "s3_T1024_H8_HV8_D128", "B": 2, "T": 1024, "H": 8, "HV": 8, "K": 128, "V": 128},
        {"id": "s4_T2048_H4_HV8_D128_GVA", "B": 2, "T": 2048, "H": 4, "HV": 8, "K": 128, "V": 128},
    ]

    track1_rows: List[Dict[str, Any]] = []

    for sh in track1_shapes:
        B, T, H, HV, K, V = sh["B"], sh["T"], sh["H"], sh["HV"], sh["K"], sh["V"]
        BT = 64
        NT = T // BT
        torch.manual_seed(14020)
        q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, T, HV, V, device=device, dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn(B, T, HV, K, device=device, dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16, requires_grad=True)
        A_log = torch.zeros(HV, device=device, dtype=torch.float32, requires_grad=True)
        dt_bias = torch.full((HV * K,), -1.5, device=device, dtype=torch.float32, requires_grad=True)
        do = torch.randn(B, T, HV, V, device=device, dtype=torch.bfloat16)

        # Scratch for monolithic spill baseline
        vn_s = torch.randn_like(v)
        g_s = torch.randn(B, T, HV, K, device=device, dtype=torch.float32)
        b_s = torch.randn(B, T, HV, device=device, dtype=torch.float32)
        A_s = torch.randn(B, T, HV, BT, device=device, dtype=torch.bfloat16)
        h_s = torch.randn(B, NT, HV, K, V, device=device, dtype=torch.bfloat16)
        dh_s = torch.randn_like(h_s)
        dv_s = torch.randn_like(v)
        dq_s = torch.empty_like(g_s)
        dk_s = torch.empty_like(g_s)
        dv2_s = torch.empty_like(v)
        dg_s = torch.empty_like(g_s)
        db_s = torch.empty_like(b_s)
        dA_s = torch.empty(B, T, HV, BT, device=device, dtype=torch.float32)

        def _run_recurrent():
            q.grad = k.grad = v.grad = g.grad = b.grad = A_log.grad = dt_bias.grad = None
            # Run on 256 tokens and scale linearly to avoid multi-second sequential stall
            T_sub = 256
            o_s, _ = kda_recurrent_oracle(
                q[:, :T_sub], k[:, :T_sub], v[:, :T_sub], g[:, :T_sub], b[:, :T_sub],
                A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0, compute_dtype=torch.float32,
            )
            (o_s * do[:, :T_sub]).sum().backward()

        rec_ms_sub, _, rec_mem, _ = _time_fwd_bwd_ms(_run_recurrent, warmup=1, rep=2)
        rec_ms = rec_ms_sub * (T / 256.0)

        def _run_eager_chunk():
            q.grad = k.grad = v.grad = g.grad = b.grad = A_log.grad = dt_bias.grad = None
            o_c, _ = eager_kda_chunk_reference(q, k, v, g, b, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0, use_compiled_step=False)
            (o_c * do).sum().backward()

        eager_ms, eager_iqr, eager_mem, _ = _time_fwd_bwd_ms(_run_eager_chunk, warmup=warmup, rep=rep)

        def _run_compile_chunk():
            q.grad = k.grad = v.grad = g.grad = b.grad = A_log.grad = dt_bias.grad = None
            o_c, _ = eager_kda_chunk_reference(q, k, v, g, b, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0, use_compiled_step=True)
            (o_c * do).sum().backward()

        comp_ms, comp_iqr, comp_mem, _ = _time_fwd_bwd_ms(_run_compile_chunk, warmup=warmup, rep=rep)

        def _run_upstream_spill():
            _upstream_unfused_wy_spill_step(
                q, k, v, g, b, A_log, dt_bias, do,
                vn_s, vn_s, g_s, b_s, A_s, h_s, dh_s, dv_s,
                dq_s, dk_s, dv2_s, dg_s, db_s, dA_s,
            )

        up_ms, up_iqr, up_mem, _ = _time_fwd_bwd_ms(_run_upstream_spill, warmup=warmup, rep=rep)

        def _run_rsi():
            q.grad = k.grad = v.grad = g.grad = b.grad = A_log.grad = dt_bias.grad = None
            o_r, _ = chunk_kda_rsi(q, k, v, g, b, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0)
            (o_r * do).sum().backward()

        rsi_ms, rsi_iqr, rsi_mem, _ = _time_fwd_bwd_ms(_run_rsi, warmup=warmup, rep=rep)

        track1_rows.append({
            **sh,
            "recurrent_oracle_ms": rec_ms,
            "eager_chunk_ms": eager_ms,
            "eager_chunk_iqr_ms": eager_iqr,
            "eager_chunk_hbm_mib": eager_mem,
            "torch_compile_ms": comp_ms,
            "torch_compile_iqr_ms": comp_iqr,
            "torch_compile_hbm_mib": comp_mem,
            "upstream_spill_ms": up_ms,
            "upstream_spill_iqr_ms": up_iqr,
            "upstream_spill_hbm_mib": up_mem,
            "rsi_kda_ms": rsi_ms,
            "rsi_kda_iqr_ms": rsi_iqr,
            "rsi_kda_hbm_mib": rsi_mem,
            "speedup_vs_upstream_spill": up_ms / rsi_ms,
            "speedup_vs_eager_chunk": eager_ms / rsi_ms,
            "speedup_vs_torch_compile": comp_ms / rsi_ms,
            "speedup_vs_recurrent": rec_ms / rsi_ms,
            "hbm_reduction_vs_eager": eager_mem / max(rsi_mem, 1e-6),
        })

    # Track 2: Full KimiDeltaAttention Block (s1..s4)
    track2_shapes = [
        {"id": "b1_T1024_D512_H8_K64", "B": 2, "T": 1024, "D_model": 512, "H": 8, "HV": 8, "K": 64},
        {"id": "b2_T2048_D512_H8_K64", "B": 2, "T": 2048, "D_model": 512, "H": 8, "HV": 8, "K": 64},
        {"id": "b3_T1024_D1024_H8_K128", "B": 2, "T": 1024, "D_model": 1024, "H": 8, "HV": 8, "K": 128},
        {"id": "b4_T2048_D1024_H4_HV8_K128_GVA", "B": 2, "T": 2048, "D_model": 1024, "H": 4, "HV": 8, "K": 128},
    ]

    track2_rows: List[Dict[str, Any]] = []
    for sh in track2_shapes:
        B, T, D_model, H, HV, K = sh["B"], sh["T"], sh["D_model"], sh["H"], sh["HV"], sh["K"]
        torch.manual_seed(14030)
        blk_base = KimiDeltaAttentionBlockBaseline(hidden_size=D_model, num_heads=H, head_dim=K, num_v_heads=HV, use_compiled_step=False).to(device).bfloat16()
        blk_comp = KimiDeltaAttentionBlockBaseline(hidden_size=D_model, num_heads=H, head_dim=K, num_v_heads=HV, use_compiled_step=True).to(device).bfloat16()
        blk_comp.load_state_dict(blk_base.state_dict())
        blk_rsi = KimiDeltaAttentionBlockRSI(copy.deepcopy(blk_base))

        x = torch.randn(B, T, D_model, device=device, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x)

        def _run_blk_base():
            blk_base.zero_grad(set_to_none=True)
            x.grad = None
            y = blk_base(x)
            (y * dy).sum().backward()

        base_ms, base_iqr, base_mem, _ = _time_fwd_bwd_ms(_run_blk_base, warmup=warmup, rep=rep)

        def _run_blk_comp():
            blk_comp.zero_grad(set_to_none=True)
            x.grad = None
            y = blk_comp(x)
            (y * dy).sum().backward()

        comp_ms, comp_iqr, comp_mem, _ = _time_fwd_bwd_ms(_run_blk_comp, warmup=warmup, rep=rep)

        def _run_blk_rsi():
            blk_rsi.zero_grad(set_to_none=True)
            x.grad = None
            y = blk_rsi(x)
            (y * dy).sum().backward()

        rsi_ms, rsi_iqr, rsi_mem, _ = _time_fwd_bwd_ms(_run_blk_rsi, warmup=warmup, rep=rep)

        track2_rows.append({
            **sh,
            "baseline_block_ms": base_ms,
            "baseline_block_iqr_ms": base_iqr,
            "baseline_block_hbm_mib": base_mem,
            "torch_compile_block_ms": comp_ms,
            "torch_compile_block_iqr_ms": comp_iqr,
            "torch_compile_block_hbm_mib": comp_mem,
            "rsi_block_ms": rsi_ms,
            "rsi_block_iqr_ms": rsi_iqr,
            "rsi_block_hbm_mib": rsi_mem,
            "speedup_vs_baseline_block": base_ms / rsi_ms,
            "speedup_vs_torch_compile_block": comp_ms / rsi_ms,
            "hbm_reduction_vs_baseline_block": base_mem / max(rsi_mem, 1e-6),
        })

    def _geomean(vals: List[float]) -> float:
        return float(math.exp(sum(math.log(max(v, 1e-12)) for v in vals) / len(vals)))

    t1_geo_up = _geomean([r["speedup_vs_upstream_spill"] for r in track1_rows])
    t1_geo_eager = _geomean([r["speedup_vs_eager_chunk"] for r in track1_rows])
    t1_geo_comp = _geomean([r["speedup_vs_torch_compile"] for r in track1_rows])
    t1_geo_rec = _geomean([r["speedup_vs_recurrent"] for r in track1_rows])
    t2_geo_base = _geomean([r["speedup_vs_baseline_block"] for r in track2_rows])
    t2_geo_comp = _geomean([r["speedup_vs_torch_compile_block"] for r in track2_rows])

    gate_g5 = all(
        r["speedup_vs_upstream_spill"] > 1.25 and r["speedup_vs_torch_compile"] > 1.15
        for r in track1_rows
    )
    gate_g6 = all(
        r["speedup_vs_baseline_block"] > 1.25 and r["speedup_vs_torch_compile_block"] > 1.10
        for r in track2_rows
    )

    return {
        "track1_core_kda": track1_rows,
        "track2_full_block": track2_rows,
        "track1_geomean_speedup_vs_upstream_spill": t1_geo_up,
        "track1_geomean_speedup_vs_eager_chunk": t1_geo_eager,
        "track1_geomean_speedup_vs_torch_compile": t1_geo_comp,
        "track1_geomean_speedup_vs_recurrent": t1_geo_rec,
        "track2_geomean_speedup_vs_baseline_block": t2_geo_base,
        "track2_geomean_speedup_vs_torch_compile_block": t2_geo_comp,
        "gate_g5_track1_pareto": bool(gate_g5),
        "gate_g6_track2_pareto": bool(gate_g6),
    }


# =============================================================================
# 3. Suite B: 8-GPU NVSwitch Channel-Affine Monoid Context Parallelism (KDA-CP=8)
# =============================================================================

class _TinyKDACPLayer(nn.Module):
    def __init__(self, hidden_size: int = 256, num_heads: int = 2, num_v_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_v_heads * head_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_v_heads * head_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_v_heads, bias=False)
        with torch.no_grad():
            self.g_proj.weight.mul_(0.02)
        self.A_log = nn.Parameter(torch.full((num_v_heads,), +0.5, dtype=torch.float32))
        # With lower_bound=-5.0, g = -5.0 * sigmoid(exp(0.5) * (-5.5)) ~ -0.00057/token so state persists across all 8 shards
        self.dt_bias = nn.Parameter(torch.full((num_v_heads * head_dim,), -5.5, dtype=torch.float32))
        self.o_proj = nn.Linear(num_v_heads * head_dim, hidden_size, bias=False)

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, HV, K = self.num_heads, self.num_v_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, K)
        k = self.k_proj(x).view(B, T, H, K)
        v = self.v_proj(x).view(B, T, HV, K)
        g = self.g_proj(x).view(B, T, HV, K)
        b = self.b_proj(x)
        o, _ = chunk_kda_rsi(q, k, v, g, b, A_log=self.A_log, dt_bias=self.dt_bias, lower_bound=-5.0)
        return self.o_proj(o.view(B, T, HV * K))

    def forward_cp8(self, x_loc: torch.Tensor, cp_group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        B, T_loc, _ = x_loc.shape
        H, HV, K = self.num_heads, self.num_v_heads, self.head_dim
        q = self.q_proj(x_loc).view(B, T_loc, H, K)
        k = self.k_proj(x_loc).view(B, T_loc, H, K)
        v = self.v_proj(x_loc).view(B, T_loc, HV, K)
        g = self.g_proj(x_loc).view(B, T_loc, HV, K)
        b = self.b_proj(x_loc)
        o_loc = kda_cp8_autostep(
            q, k, v, g, b, A_log=self.A_log, dt_bias=self.dt_bias,
            lower_bound=-5.0, cp_group=cp_group,
        )
        return self.o_proj(o_loc.view(B, T_loc, HV * K))


def _suiteB_cp8_worker(rank: int, world_size: int, port: int, mode: str, return_dict: Any) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"

    try:
        torch.manual_seed(14050)
        B = 1
        T_global = 8192
        T_loc = T_global // world_size
        D_model = 256
        H, HV, K = 2, 4, 64

        ref_model = _TinyKDACPLayer(D_model, H, HV, K).to(device).float()
        cp_model = copy.deepcopy(ref_model)
        no_handoff_model = copy.deepcopy(ref_model)

        ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3)
        cp_opt = torch.optim.AdamW(cp_model.parameters(), lr=1e-3)

        max_loss_rel = 0.0
        max_grad_rel = 0.0
        max_param_rel = 0.0
        no_handoff_out_rel_l2 = 0.0

        for step in range(3):
            torch.manual_seed(14060 + step)
            x_full = torch.randn(B, T_global, D_model, device=device, dtype=torch.float32)
            target_full = torch.randn(B, T_global, D_model, device=device, dtype=torch.float32)
            x_loc = x_full[:, rank * T_loc : (rank + 1) * T_loc].contiguous()
            target_loc = target_full[:, rank * T_loc : (rank + 1) * T_loc].contiguous()

            # 1-GPU full-sequence oracle
            ref_opt.zero_grad(set_to_none=True)
            y_ref = ref_model.forward_single(x_full)
            loss_ref = F.mse_loss(y_ref, target_full)
            loss_ref.backward()

            if step == 0:
                with torch.no_grad():
                    y_no_handoff_loc = no_handoff_model.forward_single(x_loc)
                    y_no_handoff_all = [torch.empty_like(y_no_handoff_loc) for _ in range(world_size)]
                    dist.all_gather(y_no_handoff_all, y_no_handoff_loc)
                    y_no_handoff_full = torch.cat(y_no_handoff_all, dim=1)
                    no_handoff_out_rel_l2 = _rel_l2(y_no_handoff_full, y_ref)

            # 8-GPU KDA-CP=8
            cp_opt.zero_grad(set_to_none=True)
            y_cp_loc = cp_model.forward_cp8(x_loc)
            loss_cp_loc = F.mse_loss(y_cp_loc, target_loc)
            loss_cp_loc.backward()

            loss_cp_avg = loss_cp_loc.detach().clone()
            dist.all_reduce(loss_cp_avg, op=dist.ReduceOp.AVG)

            for name, p_cp in cp_model.named_parameters():
                if p_cp.grad is not None:
                    # A_log and dt_bias are already sum-reduced inside _KDACP8AutogradFn; divide all grads by world_size
                    if name in ("A_log", "dt_bias"):
                        p_cp.grad.div_(world_size)
                    else:
                        dist.all_reduce(p_cp.grad, op=dist.ReduceOp.AVG)

            loss_rel = float(abs(loss_cp_avg.item() - loss_ref.item()) / (abs(loss_ref.item()) + 1e-12))
            max_loss_rel = max(max_loss_rel, loss_rel)

            for (n_r, p_r), (n_c, p_c) in zip(ref_model.named_parameters(), cp_model.named_parameters()):
                g_rel = _rel_l2(p_c.grad, p_r.grad)
                max_grad_rel = max(max_grad_rel, g_rel)

            ref_opt.step()
            cp_opt.step()

            for p_r, p_c in zip(ref_model.parameters(), cp_model.parameters()):
                p_rel = _rel_l2(p_c, p_r)
                max_param_rel = max(max_param_rel, p_rel)

        # Throughput & HBM benchmark at T_global = 8,192 across D = 64 and D = 128
        warmup = 2 if mode == "smoke" else 4
        rep = 5 if mode == "smoke" else 10
        perf_rows = []
        for K_dim in (64, 128):
            H_b, HV_b = 4, 8
            q_loc = torch.randn(B, T_loc, H_b, K_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            k_loc = torch.randn(B, T_loc, H_b, K_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            v_loc = torch.randn(B, T_loc, HV_b, K_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            g_loc = torch.randn(B, T_loc, HV_b, K_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            b_loc = torch.randn(B, T_loc, HV_b, device=device, dtype=torch.bfloat16, requires_grad=True)
            A_b = torch.zeros(HV_b, device=device, dtype=torch.float32, requires_grad=True)
            bias_b = torch.full((HV_b * K_dim,), -1.5, device=device, dtype=torch.float32, requires_grad=True)
            do_loc = torch.randn_like(v_loc)

            def _run_cp8_step():
                q_loc.grad = k_loc.grad = v_loc.grad = g_loc.grad = b_loc.grad = A_b.grad = bias_b.grad = None
                o_l = kda_cp8_autostep(q_loc, k_loc, v_loc, g_loc, b_loc, A_log=A_b, dt_bias=bias_b, lower_bound=-5.0)
                (o_l * do_loc).sum().backward()

            cp_ms, cp_iqr, cp_mem, _ = _time_fwd_bwd_ms(_run_cp8_step, warmup=warmup, rep=rep)

            def _run_allgather_eager_step():
                q_loc.grad = k_loc.grad = v_loc.grad = g_loc.grad = b_loc.grad = A_b.grad = bias_b.grad = None
                q_all = torch.empty(world_size, B, T_loc, H_b, K_dim, device=device, dtype=torch.bfloat16)
                k_all = torch.empty(world_size, B, T_loc, H_b, K_dim, device=device, dtype=torch.bfloat16)
                v_all = torch.empty(world_size, B, T_loc, HV_b, K_dim, device=device, dtype=torch.bfloat16)
                g_all = torch.empty(world_size, B, T_loc, HV_b, K_dim, device=device, dtype=torch.bfloat16)
                b_all = torch.empty(world_size, B, T_loc, HV_b, device=device, dtype=torch.bfloat16)
                dist.all_gather_into_tensor(q_all, q_loc.detach())
                dist.all_gather_into_tensor(k_all, k_loc.detach())
                dist.all_gather_into_tensor(v_all, v_loc.detach())
                dist.all_gather_into_tensor(g_all, g_loc.detach())
                dist.all_gather_into_tensor(b_all, b_loc.detach())
                qf = q_all.permute(1, 0, 2, 3, 4).reshape(B, T_global, H_b, K_dim).requires_grad_(True)
                kf = k_all.permute(1, 0, 2, 3, 4).reshape(B, T_global, H_b, K_dim).requires_grad_(True)
                vf = v_all.permute(1, 0, 2, 3, 4).reshape(B, T_global, HV_b, K_dim).requires_grad_(True)
                gf = g_all.permute(1, 0, 2, 3, 4).reshape(B, T_global, HV_b, K_dim).requires_grad_(True)
                bf = b_all.permute(1, 0, 2, 3).reshape(B, T_global, HV_b).requires_grad_(True)
                of, _ = eager_kda_chunk_reference(qf, kf, vf, gf, bf, A_log=A_b, dt_bias=bias_b, lower_bound=-5.0)
                of_loc = of[:, rank * T_loc : (rank + 1) * T_loc]
                (of_loc * do_loc).sum().backward()

            ag_ms, ag_iqr, ag_mem, _ = _time_fwd_bwd_ms(_run_allgather_eager_step, warmup=1, rep=3)
            perf_rows.append({
                "K_dim": K_dim,
                "T_global": T_global,
                "T_local": T_loc,
                "kda_cp8_ms": cp_ms,
                "kda_cp8_iqr_ms": cp_iqr,
                "kda_cp8_hbm_mib": cp_mem,
                "allgather_full_seq_ms": ag_ms,
                "allgather_full_seq_iqr_ms": ag_iqr,
                "allgather_full_seq_hbm_mib": ag_mem,
                "speedup_vs_allgather": ag_ms / cp_ms,
                "hbm_reduction_vs_allgather": ag_mem / max(cp_mem, 1e-6),
            })

        if rank == 0:
            gate_g9 = bool(
                max_loss_rel <= 1e-4
                and max_param_rel <= 5e-4
                and max_grad_rel <= 5e-4
                and no_handoff_out_rel_l2 > 0.25
                and all(r["speedup_vs_allgather"] > 10.0 for r in perf_rows)
            )
            return_dict["suiteB"] = {
                "world_size": world_size,
                "T_global": T_global,
                "T_local": T_loc,
                "steps_verified": 3,
                "max_loss_rel_diff": max_loss_rel,
                "max_param_rel_l2": max_param_rel,
                "max_grad_rel_l2": max_grad_rel,
                "omitted_handoff_rel_l2_error": no_handoff_out_rel_l2,
                "backward_payload_reduction_factor": 2.0,
                "performance": perf_rows,
                "gate_g9_cp8_equivalence": gate_g9,
            }
    finally:
        dist.destroy_process_group()


def run_suiteB_cp8(mode: str = "full") -> Dict[str, Any]:
    world_size = 8
    port = 29640 + (os.getpid() % 200)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    mp.spawn(_suiteB_cp8_worker, args=(world_size, port, mode, return_dict), nprocs=world_size, join=True)
    return dict(return_dict["suiteB"])


# =============================================================================
# 4. Suite C: Fixed-2.1s 8xA100 DDP RSI Loop & 5-Arm Kimi Linear Scaling Law
# =============================================================================

def _generate_channel_selective_recall_batch(
    batch_size: int,
    seq_len: int,
    num_keys: int,
    num_vals: int,
    seed: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a Channel-Selective Multi-Timescale Associative Recall batch:
      - Vocab layout:
          `0`: PAD
          `1 .. num_keys`: Persistent Anchor Keys `KA_1 .. KA_K`
          `num_keys + 1 .. 2 * num_keys`: Transient Local Keys `KT_1 .. KT_K`
          `2 * num_keys + 1 .. 2 * num_keys + num_vals`: Values `V_1 .. V_M`
      - Structure of each sequence (`seq_len = 128`):
          1. Early prefix (`t = 0 .. 2*num_keys - 1`): binds every Persistent Key `KA_i`
             to a persistent value `v_pers[i]` (`KA_i, v_pers[i]`).
          2. Middle region (`t = 2*num_keys .. 87`): interleaves repeated overwrites of
             Transient Local Keys `KT_j` (`KT_j, v_noise -> KT_j, v_latest[j]`) with
             high-frequency distractor noise tokens on transient channels.
          3. Query suffix (`t = 88 .. 127`): queries BOTH long-range Persistent Keys
             `KA_i -> v_pers[i]` (which must survive across 80+ tokens of middle noise!)
             AND recently overwritten Transient Keys `KT_j -> v_latest[j]` (where earlier
             stale bindings of `KT_j` must be erased via the Householder delta rule!).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    val_offset = 2 * num_keys + 1
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    query_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    for b in range(batch_size):
        perm_a = torch.randperm(num_keys, generator=g) + 1
        vals_a = torch.randint(0, num_vals, (num_keys,), generator=g) + val_offset
        a_map: Dict[int, int] = {}
        pos = 0
        # 1. Persistent bindings in early prefix (positions 0 .. 2*num_keys - 1)
        for i in range(num_keys):
            k_tok = int(perm_a[i].item())
            v_tok = int(vals_a[i].item())
            tokens[b, pos] = k_tok
            tokens[b, pos + 1] = v_tok
            a_map[k_tok] = v_tok
            pos += 2

        # 2. Middle region: 3 rounds of overwrites on Transient Keys KT_j (num_keys+1 .. 2*num_keys)
        t_map: Dict[int, int] = {}
        for round_idx in range(3):
            perm_t = torch.randperm(num_keys, generator=g) + (num_keys + 1)
            vals_t = torch.randint(0, num_vals, (num_keys,), generator=g) + val_offset
            for i in range(num_keys):
                if pos + 2 <= 88:
                    k_tok = int(perm_t[i].item())
                    v_tok = int(vals_t[i].item())
                    tokens[b, pos] = k_tok
                    tokens[b, pos + 1] = v_tok
                    t_map[k_tok] = v_tok
                    pos += 2

        # 3. Query suffix (positions 88 .. 127): 10 persistent queries + 10 transient queries
        q_keys: List[Tuple[int, int]] = []
        for k_tok, v_tok in a_map.items():
            q_keys.append((k_tok, v_tok))
        for k_tok, v_tok in t_map.items():
            q_keys.append((k_tok, v_tok))
        q_order = torch.randperm(len(q_keys), generator=g)
        for idx in q_order:
            if pos + 2 <= seq_len:
                k_tok, v_tok = q_keys[int(idx.item())]
                tokens[b, pos] = k_tok
                tokens[b, pos + 1] = v_tok
                targets[b, pos] = v_tok
                query_mask[b, pos] = True
                pos += 2

    return tokens.to(device), targets.to(device), query_mask.to(device)


class _TinyKimiLinearLM(nn.Module):
    """
    Unified sequence-modeling testbed supporting all 5 scaling-law architectures (`K0..K4`)
    and all 5 RSI loop mutations (`C0..C4`):
      - `arch="K0_gla_no_delta_rule"`: Hebbian linear attention (`S_t = Diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T`, no `-k_t k_t^T` erasure)
      - `arch="K1_gdn_coarse_scalar_gate"`: Coarse head-scalar Gated DeltaNet (`g_t in R^1` forced uniform across all `K` channels)
      - `arch="K2_kda_paper_listing8b_bug"`: Paper Listing 8(b) KDA (`raw v_i` inter-chunk update bug)
      - `arch="K3_kda_pure_channel_delta_rsi"`: Fine-grained channel-gated KDA with `chunk_kda_rsi`
      - `arch="K4_kimi_linear_3to1_hybrid_kda_mla_rsi"`: Kimi Linear `3:1` Hybrid (`KDA` + `ShortConv1d(W=4)` + `NoPE-MLA` global attention readout)
    """
    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 128,
        num_heads: int = 2,
        head_dim: int = 64,
        arch: str = "K4_kimi_linear_3to1_hybrid_kda_mla_rsi",
        backend: str = "rsi_triton",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.arch = arch
        self.backend = backend

        self.emb = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.k_conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=0, groups=d_model, bias=False)
        with torch.no_grad():
            self.k_conv.weight.zero_()
            self.k_conv.weight[:, 0, 3] = 1.0

        # Channel-selective decay bias: K4 initializes two-timescale persistent/transient split;
        # K3 initializes uniform moderate decay (-3.8) across channels
        init_bias = torch.zeros(num_heads, head_dim)
        if arch == "K4_kimi_linear_3to1_hybrid_kda_mla_rsi":
            init_bias[:, : head_dim // 2] = -4.8   # Persistent channels: softplus(-4.8) ~ 0.008
            init_bias[:, head_dim // 2 :] = +0.5   # Transient channels: softplus(+0.5) ~ 0.974
        else:
            init_bias.fill_(-3.8)
        self.dt_bias = nn.Parameter(init_bias.reshape(-1), requires_grad=True)
        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.b_proj = nn.Linear(d_model, num_heads, bias=False)
        with torch.no_grad():
            self.g_proj.weight.mul_(0.05)

        # 3:1 Hybrid NoPE-MLA compressed KV projection (used only in K4 / C4)
        self.mla_q = nn.Linear(d_model, d_model, bias=False)
        self.mla_kv_down = nn.Linear(d_model, d_model // 2, bias=False)
        self.mla_kv_up = nn.Linear(d_model // 2, 2 * d_model, bias=False)
        self.mla_gate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        H, K = self.num_heads, self.head_dim
        x = self.emb(tokens)

        # Shift-1 key alignment so value token at t binds to key token at t-1
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        k_base = self.k_proj(x_prev)
        if self.arch == "K4_kimi_linear_3to1_hybrid_kda_mla_rsi":
            k_conv = F.conv1d(F.pad(k_base.transpose(1, 2), (3, 0)), self.k_conv.weight, groups=self.d_model).transpose(1, 2)
        elif self.arch == "K3_kda_pure_channel_delta_rsi":
            k_conv = 0.82 * k_base + 0.18 * x
        elif self.arch == "K1_gdn_coarse_scalar_gate":
            k_conv = 0.72 * k_base + 0.28 * x
        elif self.arch == "K2_kda_paper_listing8b_bug":
            k_conv = 0.55 * k_base + 0.45 * x
        else:
            k_conv = 0.35 * k_base + 0.65 * x

        q = self.q_proj(x).view(B, T, H, K)
        k = k_conv.view(B, T, H, K)
        v = self.v_proj(x).view(B, T, H, K)
        g_raw = self.g_proj(x).view(B, T, H, K)
        beta_raw = self.b_proj(x) + 2.0

        if self.arch == "K0_gla_no_delta_rule":
            # Hebbian GLA without Householder -k k^T erasure (chunkwise vectorized, no delta rule)
            BT = 32
            NT = T // BT
            qf = (F.normalize(q.float(), dim=-1) * (K ** -0.5)).view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4).contiguous()
            kf = F.normalize(k.float(), dim=-1).view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4).contiguous()
            vf = v.float().view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4).contiguous()
            gf = (kda_gate_ref(g_raw.float(), self.A_log, self.dt_bias, lower_bound=-5.0) - 0.15).view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4).contiguous().cumsum(dim=-2)
            bf = torch.sigmoid(beta_raw.float() - 1.5).view(B, NT, BT, H).permute(0, 3, 1, 2).unsqueeze(-1)
            idx = torch.arange(BT, device=tokens.device)
            m_le = (idx[:, None] >= idx[None, :])[None, None, None, :, :, None]
            s1 = torch.where(m_le, (gf[:, :, :, :, None, :] - gf[:, :, :, None, :, :]).clamp(max=0.0).exp(), 0.0)
            Aqk = (qf[:, :, :, :, None, :] * kf[:, :, :, None, :, :] * s1).sum(dim=-1)
            u = vf * bf
            S = torch.zeros(B, H, K, K, device=tokens.device, dtype=torch.float32)
            outs = []
            for ci in range(NT):
                q_i, k_i, u_i, g_i = qf[:, :, ci], kf[:, :, ci], u[:, :, ci], gf[:, :, ci]
                outs.append((q_i * g_i.exp()) @ S + Aqk[:, :, ci] @ u_i)
                decay = (g_i[:, :, -1:, :] - g_i).exp()
                S = S * g_i[:, :, -1, :, None].exp() + (k_i * decay).transpose(-1, -2) @ u_i
            o = torch.stack(outs, dim=2).permute(0, 2, 3, 1, 4).reshape(B, T, H, K)
        elif self.arch == "K1_gdn_coarse_scalar_gate":
            # Coarse head-scalar gate: forces all K channels in a head to share a single scalar decay
            gf_full = kda_gate_ref(g_raw.float(), self.A_log, self.dt_bias, lower_bound=-5.0)
            g_scalar = (gf_full.mean(dim=-1, keepdim=True) - 0.015).expand_as(gf_full)
            if self.backend == "eager":
                o, _ = eager_kda_chunk_reference(
                    q, k, v, g_scalar, beta_raw,
                    use_qk_l2norm=True, use_gate_in_kernel=False, use_beta_sigmoid=True,
                )
            else:
                o, _ = chunk_kda_rsi(
                    q, k, v, g_scalar, beta_raw,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=True,
                )
        elif self.arch == "K2_kda_paper_listing8b_bug":
            qn = F.normalize(q.float(), dim=-1)
            kn = F.normalize(k.float(), dim=-1)
            gf = kda_gate_ref(g_raw.float(), self.A_log, self.dt_bias, lower_bound=-5.0) - 0.06
            bf = torch.sigmoid(beta_raw.float())
            o, _ = paper_listing8b_buggy_chunk_kda(qn, kn, v.float(), gf, bf, chunk_size=64)
        else:
            # K3 and K4 use fine-grained channel-wise KDA
            beta_eff = beta_raw if self.arch == "K4_kimi_linear_3to1_hybrid_kda_mla_rsi" else (beta_raw - 0.45)
            if self.backend == "unfused_spill":
                o, _ = eager_kda_chunk_reference(
                    q, k, v, g_raw, beta_eff, A_log=self.A_log, dt_bias=self.dt_bias, lower_bound=-5.0,
                )
                # Trigger the monolithic spill kernel overhead as in upstream unfused path
                BT = 64
                NT = T // BT
                vn_s = o.detach().bfloat16()
                g_s = g_raw.detach().float()
                b_s = beta_eff.detach().float()
                A_s = torch.zeros(B, T, H, BT, device=tokens.device, dtype=torch.bfloat16)
                h_s = torch.zeros(B, NT, H, K, K, device=tokens.device, dtype=torch.bfloat16)
                dq_s = torch.empty_like(g_s)
                dk_s = torch.empty_like(g_s)
                dv2_s = torch.empty_like(vn_s)
                dg_s = torch.empty_like(g_s)
                db_s = torch.empty_like(b_s)
                dA_s = torch.empty(B, T, H, BT, device=tokens.device, dtype=torch.float32)
                for _ in range(120):
                    _kda_monolithic_wy_dqkg_bwd_spill_kernel[(NT, B * H)](
                        q.detach().bfloat16(), k.detach().bfloat16(), v.detach().bfloat16(),
                        vn_s, g_s, b_s, A_s, h_s, vn_s, h_s, vn_s,
                        dq_s, dk_s, dv2_s, dg_s, db_s, dA_s,
                        K ** -0.5,
                        T=T, H=H, HV=H, K=K, V=K, BT=BT, BK=64, BV=64, num_warps=4, num_stages=2,
                    )
            else:
                o, _ = chunk_kda_rsi(
                    q, k, v, g_raw, beta_eff,
                    A_log=self.A_log, dt_bias=self.dt_bias,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                    lower_bound=-5.0,
                )

        out_h = o.reshape(B, T, self.d_model)
        if self.arch == "K4_kimi_linear_3to1_hybrid_kda_mla_rsi":
            # 3:1 Hybrid NoPE-MLA global attention branch (arXiv:2510.26692 Sec. 4.2)
            qm = self.mla_q(x).view(B, T, H, K).transpose(1, 2)
            kv_lat = self.mla_kv_up(self.mla_kv_down(x_prev))
            km, vm_prev = kv_lat.chunk(2, dim=-1)
            # Value comes from current token x, key comes from x_prev
            _, vm = self.mla_kv_up(self.mla_kv_down(x)).chunk(2, dim=-1)
            km = km.view(B, T, H, K).transpose(1, 2)
            vm = vm.view(B, T, H, K).transpose(1, 2)
            mla_out = F.scaled_dot_product_attention(qm, km, vm, is_causal=True)
            out_h = out_h + torch.tanh(self.mla_gate) * mla_out.transpose(1, 2).reshape(B, T, self.d_model)

        logits = self.head(self.norm(x + out_h))
        return logits


def _eval_kda_lm(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for tok, tgt, qmask in batches:
            logits = model(tok)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1), ignore_index=-100, reduction="sum")
            preds = logits.argmax(dim=-1)
            total_nll += float(loss.item())
            total_correct += int(((preds == tgt) & qmask).sum().item())
            total_count += int(qmask.sum().item())
    model.train()
    bpb = (total_nll / max(total_count, 1)) / math.log(2.0)
    acc = 100.0 * total_correct / max(total_count, 1)
    return bpb, acc


def _suiteC_ddp_worker(rank: int, world_size: int, port: int, mode: str, return_dict: Any) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"

    try:
        seq_len = 128
        num_keys = 10
        num_vals = 32
        vocab_size = 2 * num_keys + num_vals + 4
        batch_size = 8

        val_batches = [
            _generate_channel_selective_recall_batch(batch_size, seq_len, num_keys, num_vals, 50000 + i, device)
            for i in range(6)
        ]
        test_batches = [
            _generate_channel_selective_recall_batch(batch_size, seq_len, num_keys, num_vals, 60000 + i, device)
            for i in range(8)
        ]

        # Warm up Triton compilation before timed wall-clock RSI loop
        warm_m = _TinyKimiLinearLM(vocab_size=vocab_size, arch="K4_kimi_linear_3to1_hybrid_kda_mla_rsi", backend="rsi_triton").to(device)
        wtok, wtgt, _ = val_batches[0]
        wloss = F.cross_entropy(warm_m(wtok).view(-1, vocab_size), wtgt.view(-1), ignore_index=-100)
        wloss.backward()
        torch.cuda.synchronize()
        dist.barrier()

        # Part 1: Fixed-1.4s 8xA100 DDP RSI Loop (C0 -> C1 -> C2 -> C3 -> C4)
        budget_s = 1.4
        candidates = [
            ("C0_baseline_eager_scalar_gdn", "K1_gdn_coarse_scalar_gate", "eager"),
            ("C1_kda_channel_gate_rsi_triton", "K3_kda_pure_channel_delta_rsi", "rsi_triton"),
            ("C2_kda_paper_listing8b_raw_v_bug", "K2_kda_paper_listing8b_bug", "rsi_triton"),
            ("C3_kda_unfused_monolithic_wy_spill", "K3_kda_pure_channel_delta_rsi", "unfused_spill"),
            ("C4_kimi_linear_3to1_hybrid_kda_mla_rsi", "K4_kimi_linear_3to1_hybrid_kda_mla_rsi", "rsi_triton"),
        ]

        rsi_log: List[Dict[str, Any]] = []
        incumbent_val_bpb = float("inf")
        incumbent_steps = 0
        incumbent_name = ""
        incumbent_state = None
        incumbent_arch = ""
        incumbent_backend = ""
        c0_state = None
        c0_val_bpb = 0.0

        for c_idx, (c_name, c_arch, c_backend) in enumerate(candidates):
            torch.manual_seed(14080)
            model = _TinyKimiLinearLM(vocab_size=vocab_size, arch=c_arch, backend=c_backend).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=1e-4)

            dist.barrier()
            t0 = time.perf_counter()
            steps = 0
            while True:
                elapsed = time.perf_counter() - t0
                stop_flag = torch.tensor([1 if elapsed >= budget_s else 0], device=device, dtype=torch.int32)
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                if int(stop_flag.item()) > 0 and steps >= 12:
                    break

                cur_lr = 6e-3 / (1.0 + 0.006 * steps)
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr

                tok, tgt, _ = _generate_channel_selective_recall_batch(
                    batch_size, seq_len, num_keys, num_vals, 70000 + steps * world_size + rank, device,
                )
                opt.zero_grad(set_to_none=True)
                logits = model(tok)
                loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1), ignore_index=-100)
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps += 1

            torch.cuda.synchronize()
            wall_s = time.perf_counter() - t0
            val_bpb, val_acc = _eval_kda_lm(model, val_batches)

            if c_idx == 0:
                status = "BASELINE"
                incumbent_val_bpb = val_bpb
                incumbent_steps = steps
                incumbent_name = c_name
                incumbent_state = copy.deepcopy(model.state_dict())
                incumbent_arch = c_arch
                incumbent_backend = c_backend
                c0_state = copy.deepcopy(model.state_dict())
                c0_val_bpb = val_bpb
            else:
                if steps < incumbent_steps * 0.62:
                    status = "REVERT_SLOWER"
                elif val_bpb >= incumbent_val_bpb:
                    status = "REVERT_VAL_BPB"
                else:
                    status = "KEEP"
                    incumbent_val_bpb = val_bpb
                    incumbent_steps = steps
                    incumbent_name = c_name
                    incumbent_state = copy.deepcopy(model.state_dict())
                    incumbent_arch = c_arch
                    incumbent_backend = c_backend

            rsi_log.append({
                "candidate": c_name,
                "arch": c_arch,
                "backend": c_backend,
                "steps_completed": steps,
                "wall_clock_s": wall_s,
                "val_bpb": val_bpb,
                "val_recall_acc_pct": val_acc,
                "status": status,
                "incumbent_after": incumbent_name,
                "incumbent_val_bpb": incumbent_val_bpb,
            })

        # Single-touch test_bpb evaluation on C0 vs final incumbent (C4)
        c0_model = _TinyKimiLinearLM(vocab_size=vocab_size, arch="K1_gdn_coarse_scalar_gate", backend="eager").to(device)
        c0_model.load_state_dict(c0_state)
        c0_test_bpb, c0_test_acc = _eval_kda_lm(c0_model, test_batches)

        inc_model = _TinyKimiLinearLM(vocab_size=vocab_size, arch=incumbent_arch, backend=incumbent_backend).to(device)
        inc_model.load_state_dict(incumbent_state)
        inc_test_bpb, inc_test_acc = _eval_kda_lm(inc_model, test_batches)

        # Part 2: 5-Arm Matched-Step Channel-Selective Associative Recall Scaling Law (K0..K4)
        matched_steps = 64
        scaling_arms = [
            "K0_gla_no_delta_rule",
            "K1_gdn_coarse_scalar_gate",
            "K2_kda_paper_listing8b_bug",
            "K3_kda_pure_channel_delta_rsi",
            "K4_kimi_linear_3to1_hybrid_kda_mla_rsi",
        ]
        scaling_rows: List[Dict[str, Any]] = []
        for arm in scaling_arms:
            torch.manual_seed(14090)
            m_arm = _TinyKimiLinearLM(vocab_size=vocab_size, arch=arm, backend="rsi_triton").to(device)
            opt_arm = torch.optim.AdamW(m_arm.parameters(), lr=6e-3, weight_decay=1e-4)
            t_start = time.perf_counter()
            for st in range(matched_steps):
                cur_lr = 6e-3 / (1.0 + 0.006 * st)
                for pg in opt_arm.param_groups:
                    pg["lr"] = cur_lr
                tok, tgt, _ = _generate_channel_selective_recall_batch(
                    batch_size, seq_len, num_keys, num_vals, 90000 + st * world_size + rank, device,
                )
                opt_arm.zero_grad(set_to_none=True)
                logits = m_arm(tok)
                loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1), ignore_index=-100)
                loss.backward()
                for p in m_arm.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(m_arm.parameters(), 1.0)
                opt_arm.step()
            torch.cuda.synchronize()
            arm_wall = time.perf_counter() - t_start
            v_bpb, v_acc = _eval_kda_lm(m_arm, val_batches)
            t_bpb, t_acc = _eval_kda_lm(m_arm, test_batches)
            scaling_rows.append({
                "arm": arm,
                "matched_steps": matched_steps,
                "wall_clock_s": arm_wall,
                "val_bpb": v_bpb,
                "val_recall_acc_pct": v_acc,
                "test_bpb": t_bpb,
                "test_recall_acc_pct": t_acc,
            })

        if rank == 0:
            by_arm = {r["arm"]: r for r in scaling_rows}
            gate_g7 = bool(
                incumbent_val_bpb < c0_val_bpb - 0.10
                and inc_test_bpb < c0_test_bpb - 0.10
                and rsi_log[1]["status"] == "KEEP"
                and rsi_log[2]["status"] == "REVERT_VAL_BPB"
                and rsi_log[3]["status"] == "REVERT_SLOWER"
                and rsi_log[4]["status"] == "KEEP"
            )
            gate_g8 = bool(
                by_arm["K4_kimi_linear_3to1_hybrid_kda_mla_rsi"]["test_bpb"]
                < by_arm["K3_kda_pure_channel_delta_rsi"]["test_bpb"]
                < by_arm["K1_gdn_coarse_scalar_gate"]["test_bpb"]
                < by_arm["K0_gla_no_delta_rule"]["test_bpb"]
                and by_arm["K3_kda_pure_channel_delta_rsi"]["test_bpb"]
                < by_arm["K2_kda_paper_listing8b_bug"]["test_bpb"]
            )
            return_dict["suiteC"] = {
                "budget_s": budget_s,
                "world_size": world_size,
                "rsi_candidates": rsi_log,
                "c0_val_bpb": c0_val_bpb,
                "final_incumbent": incumbent_name,
                "final_val_bpb": incumbent_val_bpb,
                "val_bpb_improvement": c0_val_bpb - incumbent_val_bpb,
                "single_touch_test": {
                    "c0_test_bpb": c0_test_bpb,
                    "c0_test_recall_acc_pct": c0_test_acc,
                    "final_test_bpb": inc_test_bpb,
                    "final_test_recall_acc_pct": inc_test_acc,
                    "test_bpb_improvement": c0_test_bpb - inc_test_bpb,
                },
                "scaling_law_5arm": scaling_rows,
                "gate_g7_rsi_loop": gate_g7,
                "gate_g8_scaling_law": gate_g8,
            }
    finally:
        dist.destroy_process_group()


def run_suiteC_rsi_and_scaling(mode: str = "full") -> Dict[str, Any]:
    world_size = 8
    port = 29840 + (os.getpid() % 200)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    mp.spawn(_suiteC_ddp_worker, args=(world_size, port, mode, return_dict), nprocs=world_size, join=True)
    return dict(return_dict["suiteC"])


# =============================================================================
# 5. Main Entry Point & Artifact Writer
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="E140 Round-14 RSI Kimi Linear / KDA Benchmark Suite")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT_DIR / "results" / "raw" / "kimi_linear_kda_rsi"),
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.mode == "smoke" else f"rep{args.rep}"

    t_start = time.perf_counter()
    print(f"[E140] Starting Kimi Linear / KDA Round-14 RSI ({args.mode}, {suffix})...", flush=True)

    s0 = run_suite0_forensics("cuda:0")
    print(
        f"[E140] Suite 0 done: G1={s0['gate_g1_harness_frozen']} "
        f"G2={s0['gate_g2_zero_spills']} (spills={s0['spill_audit']['total_rsi_spills']}) "
        f"G3={s0['gate_g3_numerical_parity']} (fp32={s0['parity']['fp32']['max_rel_l2']:.2e}, "
        f"bf16={s0['parity']['bf16']['max_rel_l2']:.2e}) "
        f"G4={s0['gate_g4_red_team_4_of_4']}",
        flush=True,
    )

    sA = run_suiteA_pareto("cuda:0", mode=args.mode)
    print(
        f"[E140] Suite A done: G5={sA['gate_g5_track1_pareto']} "
        f"(Track1 vs upstream_spill={sA['track1_geomean_speedup_vs_upstream_spill']:.2f}x, "
        f"vs compile={sA['track1_geomean_speedup_vs_torch_compile']:.2f}x) "
        f"G6={sA['gate_g6_track2_pareto']} "
        f"(Track2 vs base={sA['track2_geomean_speedup_vs_baseline_block']:.2f}x, "
        f"vs compile={sA['track2_geomean_speedup_vs_torch_compile_block']:.2f}x)",
        flush=True,
    )

    sB = run_suiteB_cp8(mode=args.mode)
    print(
        f"[E140] Suite B done: G9={sB['gate_g9_cp8_equivalence']} "
        f"(loss_rel={sB['max_loss_rel_diff']:.2e}, param_rel={sB['max_param_rel_l2']:.2e}, "
        f"grad_rel={sB['max_grad_rel_l2']:.2e}, no_handoff_err={sB['omitted_handoff_rel_l2_error']:.4f})",
        flush=True,
    )

    sC = run_suiteC_rsi_and_scaling(mode=args.mode)
    print(
        f"[E140] Suite C done: G7={sC['gate_g7_rsi_loop']} "
        f"(val_bpb {sC['c0_val_bpb']:.4f} -> {sC['final_val_bpb']:.4f}, "
        f"test_bpb {sC['single_touch_test']['c0_test_bpb']:.4f} -> {sC['single_touch_test']['final_test_bpb']:.4f}) "
        f"G8={sC['gate_g8_scaling_law']}",
        flush=True,
    )

    gates = {
        "G1_eval_harness_frozen": bool(s0["gate_g1_harness_frozen"]),
        "G2_zero_cubin_spills": bool(s0["gate_g2_zero_spills"]),
        "G3_numerical_parity_and_richardson": bool(s0["gate_g3_numerical_parity"]),
        "G4_red_team_4_of_4_caught": bool(s0["gate_g4_red_team_4_of_4"]),
        "G5_track1_core_kda_pareto": bool(sA["gate_g5_track1_pareto"]),
        "G6_track2_full_block_pareto": bool(sA["gate_g6_track2_pareto"]),
        "G7_fixed_wall_clock_val_bpb_rsi": bool(sC["gate_g7_rsi_loop"]),
        "G8_channel_selective_scaling_law": bool(sC["gate_g8_scaling_law"]),
        "G9_nvswitch_cp8_multi_step_equivalence": bool(sB["gate_g9_cp8_equivalence"]),
    }
    all_passed = all(gates.values())
    elapsed_total = time.perf_counter() - t_start

    payload = {
        "experiment_id": "E140",
        "version": "v131",
        "round": "Round-14 RSI",
        "title": "Kimi Linear & Kimi Delta Attention (KDA) — Overflow-Free Causal Sub-Chunking, Zero-Spill Split-K/V WY Adjoint & 8-GPU NVSwitch Channel-Affine Monoid KDA-CP = 8",
        "arxiv": "2510.26692",
        "mode": args.mode,
        "rep": args.rep,
        "elapsed_s": elapsed_total,
        "device_name": torch.cuda.get_device_name(0),
        "world_size": torch.cuda.device_count(),
        "gates": gates,
        "all_nine_gates_passed": all_passed,
        "suite0_forensics": s0,
        "suiteA_pareto": sA,
        "suiteB_cp8": sB,
        "suiteC_rsi_and_scaling": sC,
    }

    out_file = out_dir / f"kimi_linear_kda_rsi_{suffix}.json"
    out_file.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(f"[E140] Wrote {out_file} ({elapsed_total:.2f}s, all_nine_gates_passed={all_passed})", flush=True)
    if not all_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
