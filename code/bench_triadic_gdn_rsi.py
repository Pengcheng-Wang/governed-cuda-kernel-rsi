#!/usr/bin/env python3
"""
E143 (v134, Round-17 RSI): Triadic Gated DeltaNet & Triadic Kimi Delta Attention Benchmark Suite
=================================================================================================

Executes the 5-track hardware verification and benchmark suite specified in Section 20 of `PROGRAM.md`
on 8x NVIDIA A100-SXM4-80GB (`NV12` NVSwitch):
  - Track 1: Upstream `fla/ops/triadic_gdn` & `triadic_kda` Pathology Forensics
             (Kronecker spectral explosion, 1,698 SM80 cubin register spills, TF32 mantissa truncation)
  - Track 2: Exact FP64 Oracle Parity, Stress Regimes, & Cryptographic Harness Negative-Control Audit
  - Track 3: Kernel & Block Ablation Ladder (Stages A0..A4, Triadic-GDN fwd+bwd, Triadic-KDA fwd)
  - Track 4: Fixed-Wall-Clock (`val_bpb`) Autoresearch Language Modeling Loop
  - Track 5: 8-GPU NVSwitch Parallel-WY Triadic State Context Parallelism (`Triadic-CP = 8`)
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

from triadic_gdn_rsi_kernels import (
    ALL_ROUND17_RSI_KERNELS,
    TriadicGDNBlockRSI,
    _run_triadic_gdn_bwd_impl,
    _run_triadic_gdn_fwd_impl,
    audit_round17_rsi_cubin_spills,
    chunk_triadic_gdn_rsi,
    chunk_triadic_kda_fwd_rsi,
    dual_sphere_l2norm_ref,
    triadic_gdn_chunk_eager_ref,
    triadic_gdn_cp8_step,
    triadic_gdn_recurrent_fp64_oracle,
    triadic_kda_recurrent_fp64_oracle,
)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    return float(torch.linalg.norm(af - bf) / (torch.linalg.norm(bf) + 1e-12))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bench_cuda_fn(fn, warmup: int = 5, rep: int = 15) -> float:
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


# =================================================================_____________
# Track 1: Upstream `fla` Pathology Forensics (Kronecker Explosion, Spills, TF32)
# =================================================================_____________

def run_track1_upstream_pathology_forensics() -> Dict[str, Any]:
    from fla.ops.triadic_gdn import chunk_bwd as up_bwd
    from fla.ops.triadic_gdn import chunk_fwd as up_fwd
    from fla.ops.triadic_gdn.chunk import chunk_triadic_gdn as up_chunk_triadic_gdn

    device = "cuda:0"
    torch.manual_seed(2026)

    # 1A. Unnormalized Kronecker Spectral Explosion (use_qk_l2norm absent in upstream chunk.py)
    B, T, H, K, V, E = 2, 128, 4, 64, 64, 4
    q_raw = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k_raw = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v_raw = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    k2_raw = torch.randn(B, T, H, E, device=device, dtype=torch.float32)
    q2_raw = torch.randn(B, T, H, E, device=device, dtype=torch.float32)
    g_raw = -torch.rand(B, T, H, E, device=device, dtype=torch.float32) * 0.05 - 0.005
    beta_raw = torch.full((B, T, H), 1.5, device=device, dtype=torch.float32)

    kappa_raw_norm_sq = float(((k2_raw ** 2).sum(dim=-1) * (k_raw ** 2).sum(dim=-1)).mean().item())
    qn, kn, q2n, k2n = dual_sphere_l2norm_ref(q_raw, k_raw, q2_raw, k2_raw)
    kappa_norm_sq = float(((k2n ** 2).sum(dim=-1) * (kn ** 2).sum(dim=-1)).mean().item())

    # Inspect unnormalized A and Ai = (I + A)^{-1}
    g_cum_up = up_fwd.chunk_local_cumsum(g_raw, chunk_size=64)
    A_unnorm, _ = up_fwd.chunk_triadic_gdn_fwd_intra(
        q=q_raw, k=k_raw, k2=k2_raw, q2=q2_raw, g=g_cum_up, beta=beta_raw, scale=K ** -0.5, chunk_size=64
    )
    Ai_unnorm = up_fwd.solve_tril(A=A_unnorm, cu_seqlens=None, output_dtype=torch.float32)

    q_u = q_raw.clone().requires_grad_(True)
    k_u = k_raw.clone().requires_grad_(True)
    v_u = v_raw.clone().requires_grad_(True)
    k2_u = k2_raw.clone().requires_grad_(True)
    q2_u = q2_raw.clone().requires_grad_(True)
    g_u = g_raw.clone().requires_grad_(True)
    beta_u = beta_raw.clone().requires_grad_(True)
    o_u, ht_u = up_chunk_triadic_gdn(q_u, k_u, v_u, k2_u, q2_u, g_u, beta_u, output_final_state=True)
    o_u.sum().backward()
    up_unnorm_o_nan = int(torch.isnan(o_u).sum().item() + torch.isinf(o_u).sum().item())
    up_unnorm_grad_nan = int(sum(
        torch.isnan(t.grad).sum().item() + torch.isinf(t.grad).sum().item()
        for t in (q_u, k_u, v_u, k2_u, q2_u, g_u, beta_u)
    ))

    # Compare with Round-17 RSI Dual-Sphere L2Norm on the exact same raw inputs
    q_r = q_raw.clone().requires_grad_(True)
    k_r = k_raw.clone().requires_grad_(True)
    v_r = v_raw.clone().requires_grad_(True)
    k2_r = k2_raw.clone().requires_grad_(True)
    q2_r = q2_raw.clone().requires_grad_(True)
    g_r = g_raw.clone().requires_grad_(True)
    beta_r = beta_raw.clone().requires_grad_(True)
    (
        o_rsi, ht_rsi, _, _, _, _, _, _, _, _, _, _, _, Ai_rsi, _, _, _
    ) = _run_triadic_gdn_fwd_impl(
        q_r, k_r, v_r, k2_r, q2_r, g_r, beta_r, K ** -0.5, None, use_qk_l2norm=True
    )
    o_r_ag, _ = chunk_triadic_gdn_rsi(
        q_r, k_r, v_r, k2_r, q2_r, g_r, beta_r, output_final_state=True, use_qk_l2norm=True
    )
    o_r_ag.sum().backward()
    rsi_o_nan = int(torch.isnan(o_r_ag).sum().item() + torch.isinf(o_r_ag).sum().item())
    rsi_grad_nan = int(sum(
        torch.isnan(t.grad).sum().item() + torch.isinf(t.grad).sum().item()
        for t in (q_r, k_r, v_r, k2_r, q2_r, g_r, beta_r)
    ))

    # 1B. Upstream TF32 mantissa truncation in FP32 mode vs Round-17 RSI IEEE FP32
    torch.manual_seed(42)
    B2, T2, H2, K2, V2, E2 = 2, 128, 4, 64, 64, 4
    q_f = torch.randn(B2, T2, H2, K2, device=device, dtype=torch.float32)
    k_f = torch.randn(B2, T2, H2, K2, device=device, dtype=torch.float32)
    v_f = torch.randn(B2, T2, H2, V2, device=device, dtype=torch.float32)
    k2_f = torch.randn(B2, T2, H2, E2, device=device, dtype=torch.float32)
    q2_f = torch.randn(B2, T2, H2, E2, device=device, dtype=torch.float32)
    g_f = -torch.rand(B2, T2, H2, E2, device=device, dtype=torch.float32) * 0.25 - 0.01
    beta_f = torch.sigmoid(torch.randn(B2, T2, H2, device=device, dtype=torch.float32)) * 1.5
    do_f = torch.randn_like(v_f)
    qn_f, kn_f, q2n_f, k2n_f = dual_sphere_l2norm_ref(q_f, k_f, q2_f, k2_f)

    qr64 = qn_f.to(torch.float64).requires_grad_(True)
    kr64 = kn_f.to(torch.float64).requires_grad_(True)
    vr64 = v_f.to(torch.float64).requires_grad_(True)
    k2r64 = k2n_f.to(torch.float64).requires_grad_(True)
    q2r64 = q2n_f.to(torch.float64).requires_grad_(True)
    gr64 = g_f.to(torch.float64).requires_grad_(True)
    br64 = beta_f.to(torch.float64).requires_grad_(True)
    o_ref64, ht_ref64 = triadic_gdn_recurrent_fp64_oracle(
        qr64, kr64, vr64, k2r64, q2r64, gr64, br64, output_final_state=True, use_qk_l2norm=False
    )
    (o_ref64 * do_f.to(torch.float64)).sum().backward()

    qu32 = qn_f.clone().requires_grad_(True)
    ku32 = kn_f.clone().requires_grad_(True)
    vu32 = v_f.clone().requires_grad_(True)
    k2u32 = k2n_f.clone().requires_grad_(True)
    q2u32 = q2n_f.clone().requires_grad_(True)
    gu32 = g_f.clone().requires_grad_(True)
    bu32 = beta_f.clone().requires_grad_(True)
    o_up32, ht_up32 = up_chunk_triadic_gdn(qu32, ku32, vu32, k2u32, q2u32, gu32, bu32, output_final_state=True)
    (o_up32 * do_f).sum().backward()

    qrsi32 = qn_f.clone().requires_grad_(True)
    krsi32 = kn_f.clone().requires_grad_(True)
    vrsi32 = v_f.clone().requires_grad_(True)
    k2rsi32 = k2n_f.clone().requires_grad_(True)
    q2rsi32 = q2n_f.clone().requires_grad_(True)
    grsi32 = g_f.clone().requires_grad_(True)
    brsi32 = beta_f.clone().requires_grad_(True)
    o_rsi32, ht_rsi32 = chunk_triadic_gdn_rsi(
        qrsi32, krsi32, vrsi32, k2rsi32, q2rsi32, grsi32, brsi32,
        output_final_state=True, use_qk_l2norm=False,
    )
    (o_rsi32 * do_f).sum().backward()

    # 1C. Warm all upstream specialization shapes so upstream cubin cache has all 42 specializations
    upstream_k128_fp32_smem_crash = None
    for (K_s, V_s, E_s) in [(64, 64, 4), (128, 128, 4), (128, 128, 8)]:
        for dt in [torch.bfloat16, torch.float32]:
            qs = F.normalize(torch.randn(1, 128, 2, K_s, device=device, dtype=dt), dim=-1).requires_grad_(True)
            ks = F.normalize(torch.randn(1, 128, 2, K_s, device=device, dtype=dt), dim=-1).requires_grad_(True)
            vs = torch.randn(1, 128, 2, V_s, device=device, dtype=dt, requires_grad=True)
            k2s = F.normalize(torch.randn(1, 128, 2, E_s, device=device, dtype=dt), dim=-1).requires_grad_(True)
            q2s = F.normalize(torch.randn(1, 128, 2, E_s, device=device, dtype=dt), dim=-1).requires_grad_(True)
            gs = (-torch.rand(1, 128, 2, E_s, device=device, dtype=dt) * 0.2).requires_grad_(True)
            bs = (torch.rand(1, 128, 2, device=device, dtype=dt) * 0.5).requires_grad_(True)
            try:
                os_up, _ = up_chunk_triadic_gdn(qs, ks, vs, k2s, q2s, gs, bs, output_final_state=True)
                os_up.sum().backward()
            except Exception as exc:
                if K_s == 128 and dt == torch.float32 and upstream_k128_fp32_smem_crash is None:
                    upstream_k128_fp32_smem_crash = f"{type(exc).__name__}: {exc}"
                # Also compile the remaining upstream backward kernels (dhu & intra) for this shape/dtype
                try:
                    g_c_s = up_fwd.chunk_local_cumsum(gs.detach(), chunk_size=64)
                    A_s, Aqk_s = up_fwd.chunk_triadic_gdn_fwd_intra(
                        q=qs.detach(), k=ks.detach(), k2=k2s.detach(), q2=q2s.detach(),
                        g=g_c_s, beta=bs.detach(), scale=K_s ** -0.5, chunk_size=64,
                    )
                    Ai_s = up_fwd.solve_tril(A=A_s, cu_seqlens=None, output_dtype=dt)
                    h_s, u_s, _ = up_fwd.chunk_triadic_gdn_fwd_h(
                        k=ks.detach(), v=vs.detach(), k2=k2s.detach(), g=g_c_s, beta=bs.detach(),
                        A=Ai_s, h0=None, output_final_state=False, cu_seqlens=None, chunk_size=64,
                    )
                    do_s = torch.randn_like(vs.detach())
                    dh_s, _, dv_s = up_bwd.chunk_triadic_gdn_bwd_dhu(
                        q=qs.detach(), k=ks.detach(), k2=k2s.detach(), q2=q2s.detach(), g=g_c_s,
                        beta=bs.detach(), A=Ai_s, A_qk=Aqk_s, h0=None, do=do_s, dht=None,
                        scale=K_s ** -0.5, cu_seqlens=None, chunk_size=64,
                    )
                    dq_i = torch.zeros_like(qs.detach(), dtype=torch.float32)
                    dk_i = torch.zeros_like(ks.detach(), dtype=torch.float32)
                    ks_i = torch.zeros((2, 1, 128, 2, E_s), device=device, dtype=torch.float32)
                    up_bwd.chunk_triadic_gdn_bwd_intra(
                        q=qs.detach(), k=ks.detach(), v=vs.detach(), k2=k2s.detach(), q2=q2s.detach(),
                        g=g_c_s, beta=bs.detach(), u=u_s, do=do_s, dv=dv_s,
                        dq=dq_i, dk=dk_i, ks_red=ks_i, qs_red=ks_i, kd_red=ks_i, ss_red=torch.zeros((2, 2, 2, E_s), device=device),
                        scale=K_s ** -0.5, cu_seqlens=None, chunk_size=64,
                    )
                except Exception:
                    pass

            os_rsi, _ = chunk_triadic_gdn_rsi(qs, ks, vs, k2s, q2s, gs.float(), bs.float(), output_final_state=True)
            os_rsi.sum().backward()
            gks = -torch.rand(1, 128, 2, K_s, device=device, dtype=torch.float32) * 0.15
            g2s = -torch.rand(1, 128, 2, E_s, device=device, dtype=torch.float32) * 0.15
            chunk_triadic_kda_fwd_rsi(qs, ks, vs, k2s, q2s, gks, g2s, bs.float(), output_final_state=True)

    blk = TriadicGDNBlockRSI(d_model=256, num_heads=4, head_dim=64, num_slices=4).to(device).bfloat16()
    xb = torch.randn(2, 128, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    blk(xb, stage="A4").sum().backward()

    upstream_kernels = [
        ("triadic_intra_r", up_fwd.triadic_intra_r),
        ("chunk_triadic_gdn_fwd_kernel_intra", up_fwd.chunk_triadic_gdn_fwd_kernel_intra),
        ("chunk_triadic_gdn_fwd_kernel_h", up_fwd.chunk_triadic_gdn_fwd_kernel_h),
        ("chunk_triadic_gdn_bwd_kernel_dhu", up_bwd.chunk_triadic_gdn_bwd_kernel_dhu),
        ("chunk_triadic_gdn_bwd_kernel_inter", up_bwd.chunk_triadic_gdn_bwd_kernel_inter),
        ("chunk_triadic_gdn_bwd_kernel_intra", up_bwd.chunk_triadic_gdn_bwd_kernel_intra),
    ]
    up_audit = {}
    up_total_spills = 0
    up_spilling_specs = 0
    up_total_specs = 0
    up_max_regs = 0
    for kname, kobj in upstream_kernels:
        inner = kobj.fn if hasattr(kobj, "fn") else kobj
        spills = []
        regs = []
        if hasattr(inner, "device_caches"):
            for dev_entry in inner.device_caches.values():
                cdict = dev_entry[0] if isinstance(dev_entry, tuple) else dev_entry
                for ck in cdict.values():
                    if hasattr(ck, "n_spills"):
                        spills.append(int(ck.n_spills))
                        regs.append(int(ck.n_regs))
        up_total_spills += sum(spills)
        up_spilling_specs += sum(1 for s in spills if s > 0)
        up_total_specs += len(spills)
        up_max_regs = max(up_max_regs, max(regs) if regs else 0)
        up_audit[kname] = {
            "specializations": len(spills),
            "spilling_specializations": sum(1 for s in spills if s > 0),
            "total_spills": sum(spills),
            "max_spills": max(spills) if spills else 0,
            "max_regs": max(regs) if regs else 0,
        }

    rsi_audit = audit_round17_rsi_cubin_spills()

    A_finite = A_unnorm[torch.isfinite(A_unnorm)]
    Ai_finite = Ai_unnorm[torch.isfinite(Ai_unnorm)]
    A_unnorm_max = float(A_finite.abs().max().item()) if A_finite.numel() > 0 else float("inf")
    Ai_unnorm_max = float(Ai_finite.abs().max().item()) if Ai_finite.numel() > 0 else float("inf")

    return {
        "kronecker_spectral_explosion": {
            "raw_kronecker_norm_sq_mean": kappa_raw_norm_sq,
            "dual_sphere_kronecker_norm_sq_mean": kappa_norm_sq,
            "upstream_unnorm_A_max_abs": A_unnorm_max,
            "upstream_unnorm_Ai_max_abs": Ai_unnorm_max,
            "upstream_unnorm_A_nonfinite_count": int((~torch.isfinite(A_unnorm)).sum().item()),
            "upstream_unnorm_Ai_nonfinite_count": int((~torch.isfinite(Ai_unnorm)).sum().item()),
            "upstream_unnorm_o_nan_count": up_unnorm_o_nan,
            "upstream_unnorm_grad_nan_count": up_unnorm_grad_nan,
            "rsi_dual_sphere_Ai_max_abs": float(Ai_rsi.abs().max().item()),
            "rsi_dual_sphere_o_nan_count": rsi_o_nan,
            "rsi_dual_sphere_grad_nan_count": rsi_grad_nan,
        },
        "fp32_tf32_vs_ieee_parity": {
            "upstream_tf32_o_rel_l2": rel_l2(o_up32, o_ref64),
            "upstream_tf32_ht_rel_l2": rel_l2(ht_up32, ht_ref64),
            "upstream_tf32_dq_rel_l2": rel_l2(qu32.grad, qr64.grad),
            "upstream_tf32_dk_rel_l2": rel_l2(ku32.grad, kr64.grad),
            "rsi_ieee_o_rel_l2": rel_l2(o_rsi32, o_ref64),
            "rsi_ieee_ht_rel_l2": rel_l2(ht_rsi32, ht_ref64),
            "rsi_ieee_dq_rel_l2": rel_l2(qrsi32.grad, qr64.grad),
            "rsi_ieee_dk_rel_l2": rel_l2(krsi32.grad, kr64.grad),
            "o_precision_improvement_factor": float(rel_l2(o_up32, o_ref64) / max(rel_l2(o_rsi32, o_ref64), 1e-12)),
            "dq_precision_improvement_factor": float(rel_l2(qu32.grad, qr64.grad) / max(rel_l2(qrsi32.grad, qr64.grad), 1e-12)),
        },
        "sm80_cubin_spill_comparison": {
            "upstream_total_kernels": len(upstream_kernels),
            "upstream_spilling_kernels": sum(1 for v in up_audit.values() if v["total_spills"] > 0),
            "upstream_total_specializations": up_total_specs,
            "upstream_spilling_specializations": up_spilling_specs,
            "upstream_total_cubin_spills": up_total_spills,
            "upstream_max_regs": up_max_regs,
            "upstream_k128_v128_fp32_smem_crash": upstream_k128_fp32_smem_crash,
            "upstream_per_kernel": up_audit,
            "rsi_total_kernels": rsi_audit["num_kernels"],
            "rsi_compiled_kernels": rsi_audit["compiled_kernels"],
            "rsi_total_specializations": rsi_audit["total_specializations"],
            "rsi_total_cubin_spills": rsi_audit["total_cubin_spills"],
            "rsi_max_regs": rsi_audit["max_regs_across_all_kernels"],
            "rsi_per_kernel": rsi_audit["per_kernel"],
        },
    }


# =================================================================_____________
# Track 2: Exact FP64 Oracle Parity, Stress Regimes, & Cryptographic Harness Audit
# =================================================================_____________

def run_track2_oracle_parity_and_stress() -> Dict[str, Any]:
    device = "cuda:0"
    configs = [
        {"name": "B2_T137_H4_K64_V64_E4_nonmultiple", "B": 2, "T": 137, "H": 4, "K": 64, "V": 64, "E": 4, "regime": "normal"},
        {"name": "B2_T256_H4_K64_V64_E4_standard", "B": 2, "T": 256, "H": 4, "K": 64, "V": 64, "E": 4, "regime": "normal"},
        {"name": "B1_T256_H4_K128_V128_E4_wide", "B": 1, "T": 256, "H": 4, "K": 128, "V": 128, "E": 4, "regime": "normal"},
        {"name": "B2_T128_H4_K64_V64_E4_stiff_beta", "B": 2, "T": 128, "H": 4, "K": 64, "V": 64, "E": 4, "regime": "stiff_beta"},
        {"name": "B2_T128_H4_K64_V64_E4_zero_gate", "B": 2, "T": 128, "H": 4, "K": 64, "V": 64, "E": 4, "regime": "zero_gate"},
    ]

    cases = []
    max_fp32_rel_l2 = 0.0
    max_bf16_rel_l2 = 0.0

    for idx, cfg in enumerate(configs):
        torch.manual_seed(100 + idx)
        B, T, H, K, V, E = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["E"]
        q = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
        k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
        k2 = torch.randn(B, T, H, E, device=device, dtype=torch.float32)
        q2 = torch.randn(B, T, H, E, device=device, dtype=torch.float32)
        if cfg["regime"] == "zero_gate":
            g = torch.zeros(B, T, H, E, device=device, dtype=torch.float32)
        else:
            g = -torch.rand(B, T, H, E, device=device, dtype=torch.float32) * 0.25 - 0.01
        if cfg["regime"] == "stiff_beta":
            beta = torch.empty(B, T, H, device=device, dtype=torch.float32).uniform_(1.5, 1.99)
        else:
            beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32)) * 1.5
        h0 = torch.randn(B, H, E, K, V, device=device, dtype=torch.float32) * 0.1
        do = torch.randn_like(v)
        dht = torch.randn_like(h0) * 0.1

        # FP64 Oracle
        qr = q.to(torch.float64).requires_grad_(True)
        kr = k.to(torch.float64).requires_grad_(True)
        vr = v.to(torch.float64).requires_grad_(True)
        k2r = k2.to(torch.float64).requires_grad_(True)
        q2r = q2.to(torch.float64).requires_grad_(True)
        gr = g.to(torch.float64).requires_grad_(True)
        br = beta.to(torch.float64).requires_grad_(True)
        h0r = h0.to(torch.float64).requires_grad_(True)
        o_ref, ht_ref = triadic_gdn_recurrent_fp64_oracle(
            qr, kr, vr, k2r, q2r, gr, br, initial_state=h0r, output_final_state=True, use_qk_l2norm=True
        )
        ((o_ref * do.to(torch.float64)).sum() + (ht_ref * dht.to(torch.float64)).sum()).backward()

        # FP32 Round-17 RSI
        q32 = q.clone().requires_grad_(True)
        k32 = k.clone().requires_grad_(True)
        v32 = v.clone().requires_grad_(True)
        k232 = k2.clone().requires_grad_(True)
        q232 = q2.clone().requires_grad_(True)
        g32 = g.clone().requires_grad_(True)
        b32 = beta.clone().requires_grad_(True)
        h032 = h0.clone().requires_grad_(True)
        o32, ht32 = chunk_triadic_gdn_rsi(
            q32, k32, v32, k232, q232, g32, b32, initial_state=h032, output_final_state=True, use_qk_l2norm=True
        )
        ((o32 * do).sum() + (ht32 * dht).sum()).backward()

        fp32_errs = {
            "o": rel_l2(o32, o_ref),
            "ht": rel_l2(ht32, ht_ref),
            "dq": rel_l2(q32.grad, qr.grad),
            "dk": rel_l2(k32.grad, kr.grad),
            "dv": rel_l2(v32.grad, vr.grad),
            "dk2": rel_l2(k232.grad, k2r.grad),
            "dq2": rel_l2(q232.grad, q2r.grad),
            "dg": rel_l2(g32.grad, gr.grad),
            "dbeta": rel_l2(b32.grad, br.grad),
            "dh0": rel_l2(h032.grad, h0r.grad),
        }
        max_fp32_rel_l2 = max(max_fp32_rel_l2, max(fp32_errs.values()))

        # BF16 Round-17 RSI
        q16 = q.to(torch.bfloat16).requires_grad_(True)
        k16 = k.to(torch.bfloat16).requires_grad_(True)
        v16 = v.to(torch.bfloat16).requires_grad_(True)
        k216 = k2.to(torch.bfloat16).requires_grad_(True)
        q216 = q2.to(torch.bfloat16).requires_grad_(True)
        g16 = g.clone().requires_grad_(True)
        b16 = beta.clone().requires_grad_(True)
        h016 = h0.clone().requires_grad_(True)
        o16, ht16 = chunk_triadic_gdn_rsi(
            q16, k16, v16, k216, q216, g16, b16, initial_state=h016, output_final_state=True, use_qk_l2norm=True
        )
        ((o16.float() * do).sum() + (ht16 * dht).sum()).backward()

        bf16_errs = {
            "o": rel_l2(o16, o_ref),
            "ht": rel_l2(ht16, ht_ref),
            "dq": rel_l2(q16.grad, qr.grad),
            "dk": rel_l2(k16.grad, kr.grad),
            "dv": rel_l2(v16.grad, vr.grad),
            "dk2": rel_l2(k216.grad, k2r.grad),
            "dq2": rel_l2(q216.grad, q2r.grad),
            "dg": rel_l2(g16.grad, gr.grad),
            "dbeta": rel_l2(b16.grad, br.grad),
            "dh0": rel_l2(h016.grad, h0r.grad),
        }
        max_bf16_rel_l2 = max(max_bf16_rel_l2, max(bf16_errs.values()))

        # Triadic-KDA Kronecker-sum parity
        gk = -torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 0.15 - 0.005
        g2 = -torch.rand(B, T, H, E, device=device, dtype=torch.float32) * 0.15 - 0.005
        ok_ref, htk_ref = triadic_kda_recurrent_fp64_oracle(
            q, k, v, k2, q2, gk, g2, beta, initial_state=h0, output_final_state=True, use_qk_l2norm=True
        )
        ok_32, htk_32 = chunk_triadic_kda_fwd_rsi(
            q, k, v, k2, q2, gk, g2, beta, initial_state=h0, output_final_state=True, use_qk_l2norm=True
        )
        ok_16, htk_16 = chunk_triadic_kda_fwd_rsi(
            q.bfloat16(), k.bfloat16(), v.bfloat16(), k2.bfloat16(), q2.bfloat16(),
            gk, g2, beta, initial_state=h0, output_final_state=True, use_qk_l2norm=True
        )

        cases.append({
            "config": cfg["name"],
            "fp32_rel_l2": fp32_errs,
            "bf16_rel_l2": bf16_errs,
            "triadic_kda_fp32_o_rel_l2": rel_l2(ok_32, ok_ref),
            "triadic_kda_fp32_ht_rel_l2": rel_l2(htk_32, htk_ref),
            "triadic_kda_bf16_o_rel_l2": rel_l2(ok_16, ok_ref),
            "triadic_kda_bf16_ht_rel_l2": rel_l2(htk_16, htk_ref),
        })

    # Mandatory Cryptographic Harness Evaluation + 3 Negative-Control Ablations
    harness_path = CODE_DIR / "rsi_kernel_harness.py"
    harness_sha256 = sha256_file(harness_path)

    torch.manual_seed(77)
    Bh, Th, Hh, Kh, Vh, Eh = 2, 128, 4, 64, 64, 4
    qh = torch.randn(Bh, Th, Hh, Kh, device=device, dtype=torch.float32)
    kh = torch.randn(Bh, Th, Hh, Kh, device=device, dtype=torch.float32)
    vh = torch.randn(Bh, Th, Hh, Vh, device=device, dtype=torch.float32)
    k2h = torch.randn(Bh, Th, Hh, Eh, device=device, dtype=torch.float32)
    q2h = torch.randn(Bh, Th, Hh, Eh, device=device, dtype=torch.float32)
    gh = -torch.rand(Bh, Th, Hh, Eh, device=device, dtype=torch.float32) * 0.25 - 0.01
    bh = torch.sigmoid(torch.randn(Bh, Th, Hh, device=device, dtype=torch.float32)) * 1.5
    doh = torch.randn_like(vh)

    def ref_fn():
        q_c = qh.to(torch.float64).requires_grad_(True)
        k_c = kh.to(torch.float64).requires_grad_(True)
        v_c = vh.to(torch.float64).requires_grad_(True)
        k2_c = k2h.to(torch.float64).requires_grad_(True)
        q2_c = q2h.to(torch.float64).requires_grad_(True)
        g_c = gh.to(torch.float64).requires_grad_(True)
        b_c = bh.to(torch.float64).requires_grad_(True)
        o_c, _ = triadic_gdn_recurrent_fp64_oracle(
            q_c, k_c, v_c, k2_c, q2_c, g_c, b_c, output_final_state=False, use_qk_l2norm=True
        )
        (o_c * doh.to(torch.float64)).sum().backward()
        return torch.cat([
            o_c.flatten().float(),
            k2_c.grad.flatten().float(),
            g_c.grad.flatten().float(),
            k_c.grad.flatten().float(),
        ])

    def make_cand_fn(omit_r: bool = False, wrong_dA: bool = False, omit_col_dk2: bool = False):
        def _fn():
            (
                o_out, _, q_hat, k_hat, q2_hat, k2_hat,
                q_rstd, k_rstd, q2_rstd, k2_rstd,
                g_cum, R, Rq, Ai, Aqk, h_saved, u_out,
            ) = _run_triadic_gdn_fwd_impl(
                qh, kh, vh, k2h, q2h, gh, bh, Kh ** -0.5, None,
                use_qk_l2norm=True, omit_hadamard_r_bug=omit_r,
            )
            dq, dk, dv, dk2, dq2, dg, dbeta, _ = _run_triadic_gdn_bwd_impl(
                q_hat, k_hat, vh, k2_hat, q2_hat, g_cum, bh, R, Rq, Ai, Aqk,
                h_saved, u_out, doh, None, q_rstd, k_rstd, q2_rstd, k2_rstd,
                scale=Kh ** -0.5, use_qk_l2norm=True, need_dh0=False,
                wrong_dA_sign_bug=wrong_dA, omit_col_dk2_bug=omit_col_dk2,
            )
            return torch.cat([
                o_out.flatten().float(),
                dk2.flatten().float(),
                dg.flatten().float(),
                dk.flatten().float(),
            ])
        return _fn

    ref_out = ref_fn()
    cand_valid_out = make_cand_fn()()
    cand_bug1_out = make_cand_fn(omit_r=True)()
    cand_bug2_out = make_cand_fn(wrong_dA=True)()
    cand_bug3_out = make_cand_fn(omit_col_dk2=True)()

    negative_controls = {
        "valid_round17_rsi": {
            "rel_l2_error": rel_l2(cand_valid_out, ref_out),
            "accepted": rel_l2(cand_valid_out, ref_out) < 1e-4,
        },
        "ablation1_omit_hadamard_r_factor": {
            "rel_l2_error": rel_l2(cand_bug1_out, ref_out),
            "rejected": rel_l2(cand_bug1_out, ref_out) > 0.1,
        },
        "ablation2_wrong_dA_sign": {
            "rel_l2_error": rel_l2(cand_bug2_out, ref_out),
            "rejected": rel_l2(cand_bug2_out, ref_out) > 0.05,
        },
        "ablation3_omit_column_dk2_dg": {
            "rel_l2_error": rel_l2(cand_bug3_out, ref_out),
            "rejected": rel_l2(cand_bug3_out, ref_out) > 0.1,
        },
    }

    return {
        "evaluator_sha256": harness_sha256,
        "max_fp32_rel_l2": max_fp32_rel_l2,
        "max_bf16_rel_l2": max_bf16_rel_l2,
        "cases": cases,
        "negative_controls": negative_controls,
    }


# =================================================================_____________
# Track 3: Kernel & Block Ablation Ladder (Triadic-GDN, Triadic-KDA, Block A0..A4)
# =================================================================_____________

def run_track3_ablation_ladder(mode: str = "full") -> Dict[str, Any]:
    from fla.ops.triadic_gdn.chunk import chunk_triadic_gdn as up_chunk_triadic_gdn
    from fla.ops.triadic_kda.naive import naive_chunk_triadic_kda

    device = "cuda:0"
    warmup = 2 if mode == "smoke" else 5
    rep = 5 if mode == "smoke" else 15

    shapes = [
        {"name": "B2_T2048_H8_K64_V64_E4", "B": 2, "T": 2048, "H": 8, "K": 64, "V": 64, "E": 4},
        {"name": "B2_T4096_H8_K64_V64_E4", "B": 2, "T": 4096, "H": 8, "K": 64, "V": 64, "E": 4},
        {"name": "B2_T4096_H8_K64_V128_E4", "B": 2, "T": 4096, "H": 8, "K": 64, "V": 128, "E": 4},
        {"name": "B2_T8192_H8_K64_V128_E4", "B": 2, "T": 8192, "H": 8, "K": 64, "V": 128, "E": 4},
        {"name": "B1_T4096_H8_K128_V128_E4", "B": 1, "T": 4096, "H": 8, "K": 128, "V": 128, "E": 4},
    ]
    if mode == "smoke":
        shapes = shapes[:3]

    kernel_sweep = []
    for s in shapes:
        B, T, H, K, V, E = s["B"], s["T"], s["H"], s["K"], s["V"], s["E"]
        tokens = B * T
        q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
        k2 = torch.randn(B, T, H, E, device=device, dtype=torch.bfloat16)
        q2 = torch.randn(B, T, H, E, device=device, dtype=torch.bfloat16)
        g = -torch.rand(B, T, H, E, device=device, dtype=torch.float32) * 0.25 - 0.01
        beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32)) * 1.5
        do = torch.randn_like(v)

        # 1. PyTorch Eager Chunkwise Reference (fwd+bwd)
        def run_eager_fb():
            qe = q.detach().requires_grad_(True)
            ke = k.detach().requires_grad_(True)
            ve = v.detach().requires_grad_(True)
            k2e = k2.detach().requires_grad_(True)
            q2e = q2.detach().requires_grad_(True)
            ge = g.detach().requires_grad_(True)
            be = beta.detach().requires_grad_(True)
            oe, _ = triadic_gdn_chunk_eager_ref(qe, ke, ve, k2e, q2e, ge, be, use_qk_l2norm=True)
            (oe * do).sum().backward()

        eager_fb_ms = bench_cuda_fn(run_eager_fb, warmup=warmup, rep=rep)

        # 2. Upstream `fla` `chunk_triadic_gdn` (with external PyTorch L2Norm since upstream omits it)
        def run_upstream_fwd():
            qn, kn, q2n, k2n = dual_sphere_l2norm_ref(q.float(), k.float(), q2.float(), k2.float())
            up_chunk_triadic_gdn(
                qn.to(torch.bfloat16), kn.to(torch.bfloat16), v,
                k2n.to(torch.bfloat16), q2n.to(torch.bfloat16),
                g.to(torch.bfloat16), beta.to(torch.bfloat16),
            )

        def run_upstream_fb():
            qu = q.detach().requires_grad_(True)
            ku = k.detach().requires_grad_(True)
            vu = v.detach().requires_grad_(True)
            k2u = k2.detach().requires_grad_(True)
            q2u = q2.detach().requires_grad_(True)
            gu = g.to(torch.bfloat16).detach().requires_grad_(True)
            bu = beta.to(torch.bfloat16).detach().requires_grad_(True)
            qn, kn, q2n, k2n = dual_sphere_l2norm_ref(qu.float(), ku.float(), q2u.float(), k2u.float())
            ou, _ = up_chunk_triadic_gdn(
                qn.to(torch.bfloat16), kn.to(torch.bfloat16), vu,
                k2n.to(torch.bfloat16), q2n.to(torch.bfloat16), gu, bu,
            )
            (ou * do).sum().backward()

        up_fwd_ms = bench_cuda_fn(run_upstream_fwd, warmup=warmup, rep=rep)
        up_fb_ms = bench_cuda_fn(run_upstream_fb, warmup=warmup, rep=rep)

        # 3. Round-17 RSI `chunk_triadic_gdn_rsi`
        def run_rsi_fwd():
            chunk_triadic_gdn_rsi(q, k, v, k2, q2, g, beta, use_qk_l2norm=True)

        def run_rsi_fb():
            qr = q.detach().requires_grad_(True)
            kr = k.detach().requires_grad_(True)
            vr = v.detach().requires_grad_(True)
            k2r = k2.detach().requires_grad_(True)
            q2r = q2.detach().requires_grad_(True)
            gr = g.detach().requires_grad_(True)
            br = beta.detach().requires_grad_(True)
            or_out, _ = chunk_triadic_gdn_rsi(qr, kr, vr, k2r, q2r, gr, br, use_qk_l2norm=True)
            (or_out * do).sum().backward()

        rsi_fwd_ms = bench_cuda_fn(run_rsi_fwd, warmup=warmup, rep=rep)
        rsi_fb_ms = bench_cuda_fn(run_rsi_fb, warmup=warmup, rep=rep)

        kernel_sweep.append({
            "shape": s["name"],
            "B": B, "T": T, "H": H, "K": K, "V": V, "E": E,
            "tokens": tokens,
            "eager_fb_ms": eager_fb_ms,
            "eager_fb_ktok_s": (tokens / (eager_fb_ms * 1e-3)) / 1e3,
            "upstream_fwd_ms": up_fwd_ms,
            "upstream_fb_ms": up_fb_ms,
            "upstream_fb_ktok_s": (tokens / (up_fb_ms * 1e-3)) / 1e3,
            "rsi_fwd_ms": rsi_fwd_ms,
            "rsi_fb_ms": rsi_fb_ms,
            "rsi_fb_ktok_s": (tokens / (rsi_fb_ms * 1e-3)) / 1e3,
            "rsi_vs_upstream_fwd_speedup": up_fwd_ms / rsi_fwd_ms,
            "rsi_vs_upstream_fb_speedup": up_fb_ms / rsi_fb_ms,
            "rsi_vs_eager_fb_speedup": eager_fb_ms / rsi_fb_ms,
        })

    # Triadic-KDA forward benchmark: upstream `naive_chunk_triadic_kda` vs `chunk_triadic_kda_fwd_rsi`
    Bk, Tk, Hk, Kk, Vk, Ek = 2, 2048, 8, 64, 64, 4
    qk = torch.randn(Bk, Tk, Hk, Kk, device=device, dtype=torch.bfloat16)
    kk = torch.randn(Bk, Tk, Hk, Kk, device=device, dtype=torch.bfloat16)
    vk = torch.randn(Bk, Tk, Hk, Vk, device=device, dtype=torch.bfloat16)
    k2k = torch.randn(Bk, Tk, Hk, Ek, device=device, dtype=torch.bfloat16)
    q2k = torch.randn(Bk, Tk, Hk, Ek, device=device, dtype=torch.bfloat16)
    gkk = -torch.rand(Bk, Tk, Hk, Kk, device=device, dtype=torch.float32) * 0.15 - 0.005
    g2k = -torch.rand(Bk, Tk, Hk, Ek, device=device, dtype=torch.float32) * 0.15 - 0.005
    betak = torch.sigmoid(torch.randn(Bk, Tk, Hk, device=device, dtype=torch.float32)) * 1.5

    def run_up_kda():
        qn, kn, q2n, k2n = dual_sphere_l2norm_ref(qk.float(), kk.float(), q2k.float(), k2k.float())
        naive_chunk_triadic_kda(qn, kn, vk.float(), k2n, q2n, gkk, g2k, betak)

    def run_rsi_kda():
        chunk_triadic_kda_fwd_rsi(qk, kk, vk, k2k, q2k, gkk, g2k, betak, use_qk_l2norm=True)

    up_kda_ms = bench_cuda_fn(run_up_kda, warmup=1, rep=3 if mode == "smoke" else 5)
    rsi_kda_ms = bench_cuda_fn(run_rsi_kda, warmup=warmup, rep=rep)

    # Full TriadicGDNBlockRSI Stage A0..A4 Ablation Ladder
    Bb, Tb, Db, Hb, Kb, Eb = 2, 2048, 512, 8, 64, 4
    block = TriadicGDNBlockRSI(d_model=Db, num_heads=Hb, head_dim=Kb, num_slices=Eb).to(device).bfloat16()
    xb = torch.randn(Bb, Tb, Db, device=device, dtype=torch.bfloat16)
    dyb = torch.randn_like(xb)

    stage_results = {}
    for stg in ["A0", "A2", "A3", "A4"]:
        def _run_stage():
            x_in = xb.detach().requires_grad_(True)
            block.zero_grad(set_to_none=True)
            y_out = block(x_in, stage=stg)
            (y_out * dyb).sum().backward()
        ms = bench_cuda_fn(_run_stage, warmup=warmup, rep=rep)
        stage_results[stg] = {
            "fb_ms": ms,
            "ktok_s": ((Bb * Tb) / (ms * 1e-3)) / 1e3,
        }
    for stg in stage_results:
        stage_results[stg]["speedup_vs_A0"] = stage_results["A0"]["fb_ms"] / stage_results[stg]["fb_ms"]

    return {
        "kernel_sweep": kernel_sweep,
        "triadic_kda_fwd_benchmark": {
            "shape": f"B{Bk}_T{Tk}_H{Hk}_K{Kk}_V{Vk}_E{Ek}",
            "upstream_naive_chunk_ms": up_kda_ms,
            "upstream_naive_chunk_ktok_s": ((Bk * Tk) / (up_kda_ms * 1e-3)) / 1e3,
            "rsi_triton_fwd_ms": rsi_kda_ms,
            "rsi_triton_fwd_ktok_s": ((Bk * Tk) / (rsi_kda_ms * 1e-3)) / 1e3,
            "rsi_vs_upstream_speedup": up_kda_ms / rsi_kda_ms,
        },
        "block_stage_ladder": stage_results,
    }


# =================================================================_____________
# Track 4: Fixed-Wall-Clock (`val_bpb`) Autoresearch Language Modeling Loop
# =================================================================_____________

class TinyTriadicLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 256, num_layers: int = 2, num_heads: int = 4, head_dim: int = 64, num_slices: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.RMSNorm(d_model),
                "attn": TriadicGDNBlockRSI(d_model=d_model, num_heads=num_heads, head_dim=head_dim, num_slices=num_slices),
                "norm2": nn.RMSNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, 2 * d_model, bias=False),
                    nn.SiLU(),
                    nn.Linear(2 * d_model, d_model, bias=False),
                ),
            })
            for _ in range(num_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, stage: str = "A4") -> torch.Tensor:
        x = self.emb(idx)
        for layer in self.layers:
            x = x + layer["attn"](layer["norm1"](x), stage=stage)
            x = x + layer["mlp"](layer["norm2"](x))
        x = self.norm_f(x)
        return self.lm_head(x)


def build_synthetic_triadic_corpus(num_seqs: int, seq_len: int, vocab_size: int = 256, seed: int = 1337, device: str = "cuda:0") -> torch.Tensor:
    """
    Multi-entity associative state-tracking corpus:
    Pairs of (entity_group e in 0..3, key k in 4..35, value v in 36..163) are written and later queried.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    data = torch.zeros((num_seqs, seq_len + 1), dtype=torch.long)
    for b in range(num_seqs):
        table = {}
        pos = 0
        while pos + 4 <= seq_len + 1:
            e_tok = int(torch.randint(0, 4, (1,), generator=g).item())
            k_tok = int(torch.randint(4, 28, (1,), generator=g).item())
            if (e_tok, k_tok) in table and torch.rand((1,), generator=g).item() < 0.55:
                v_tok = table[(e_tok, k_tok)]
                q_marker = 240 + e_tok
                data[b, pos] = q_marker
                data[b, pos + 1] = k_tok
                data[b, pos + 2] = v_tok
                data[b, pos + 3] = (e_tok + k_tok + v_tok) % vocab_size
            else:
                v_tok = int(torch.randint(32, 160, (1,), generator=g).item())
                table[(e_tok, k_tok)] = v_tok
                data[b, pos] = e_tok
                data[b, pos + 1] = k_tok
                data[b, pos + 2] = v_tok
                data[b, pos + 3] = (e_tok + k_tok + v_tok) % vocab_size
            pos += 4
    return data.to(device)


def run_track4_autoresearch(mode: str = "full") -> Dict[str, Any]:
    device = "cuda:0"
    budget_s = 6.0 if mode == "smoke" else 25.0
    B_batch = 8
    T_seq = 512
    vocab_size = 256

    train_data = build_synthetic_triadic_corpus(1024, T_seq, vocab_size=vocab_size, seed=2026, device=device)
    val_data = build_synthetic_triadic_corpus(64, T_seq, vocab_size=vocab_size, seed=9090, device=device)

    torch.manual_seed(4242)
    base_model = TinyTriadicLM(vocab_size=vocab_size, d_model=256, num_layers=2, num_heads=4, head_dim=64, num_slices=4).to(device).bfloat16()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    arms = [
        ("arm1_eager_chunk_A0", "A0"),
        ("arm2_unfused_block_A2", "A2"),
        ("arm3_round17_rsi_superfused_A4", "A4"),
    ]
    results = {}

    for arm_name, stg in arms:
        torch.manual_seed(4242)
        model = TinyTriadicLM(vocab_size=vocab_size, d_model=256, num_layers=2, num_heads=4, head_dim=64, num_slices=4).to(device).bfloat16()
        model.load_state_dict(init_state)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)

        # Warmup compile
        x_w = train_data[:B_batch, :-1]
        y_w = train_data[:B_batch, 1:]
        logits_w = model(x_w, stage=stg)
        loss_w = F.cross_entropy(logits_w.float().reshape(-1, vocab_size), y_w.reshape(-1))
        loss_w.backward()
        opt.zero_grad(set_to_none=True)
        model.load_state_dict(init_state)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        step = 0
        while (time.perf_counter() - t0) < budget_s:
            idx_start = (step * B_batch) % (train_data.shape[0] - B_batch)
            xb = train_data[idx_start:idx_start + B_batch, :-1]
            yb = train_data[idx_start:idx_start + B_batch, 1:]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, stage=stg)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 5 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            val_losses = []
            for vi in range(0, val_data.shape[0], B_batch):
                xv = val_data[vi:vi + B_batch, :-1]
                yv = val_data[vi:vi + B_batch, 1:]
                lv = model(xv, stage=stg)
                val_losses.append(F.cross_entropy(lv.float().reshape(-1, vocab_size), yv.reshape(-1)).item())
            val_loss = float(sum(val_losses) / len(val_losses))
            val_bpb = float(val_loss / math.log(2.0))

        results[arm_name] = {
            "stage": stg,
            "wallclock_s": elapsed,
            "completed_steps": step,
            "tokens_processed": step * B_batch * T_seq,
            "tokens_per_sec": (step * B_batch * T_seq) / elapsed,
            "final_val_loss": val_loss,
            "final_val_bpb": val_bpb,
        }

    results["rsi_vs_eager_step_ratio"] = (
        results["arm3_round17_rsi_superfused_A4"]["completed_steps"]
        / max(results["arm1_eager_chunk_A0"]["completed_steps"], 1)
    )
    results["rsi_vs_eager_val_bpb_delta"] = (
        results["arm1_eager_chunk_A0"]["final_val_bpb"]
        - results["arm3_round17_rsi_superfused_A4"]["final_val_bpb"]
    )
    return results


# =================================================================_____________
# Track 5: 8-GPU NVSwitch Parallel-WY Triadic State Context Parallelism (`Triadic-CP = 8`)
# =================================================================_____________

def _cp8_worker(rank: int, world_size: int, port: int, mode: str, return_dict):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "WARN"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"

    # 5A. Exactness verification (with vs without coupled [B, H, E, K, V] state handoff)
    torch.manual_seed(999)
    B, T_global, H, K, V, E = 1, 1024, 8, 64, 64, 4
    T_loc = T_global // world_size
    q_all = torch.randn(B, T_global, H, K, device=device, dtype=torch.float32)
    k_all = torch.randn(B, T_global, H, K, device=device, dtype=torch.float32)
    v_all = torch.randn(B, T_global, H, V, device=device, dtype=torch.float32)
    k2_all = torch.randn(B, T_global, H, E, device=device, dtype=torch.float32)
    q2_all = torch.randn(B, T_global, H, E, device=device, dtype=torch.float32)
    g_all = -torch.rand(B, T_global, H, E, device=device, dtype=torch.float32) * 0.15 - 0.005
    beta_all = torch.sigmoid(torch.randn(B, T_global, H, device=device, dtype=torch.float32)) * 1.5
    do_all = torch.randn_like(v_all)

    s0, s1 = rank * T_loc, (rank + 1) * T_loc
    q_loc = q_all[:, s0:s1].contiguous().requires_grad_(True)
    k_loc = k_all[:, s0:s1].contiguous().requires_grad_(True)
    v_loc = v_all[:, s0:s1].contiguous().requires_grad_(True)
    k2_loc = k2_all[:, s0:s1].contiguous().requires_grad_(True)
    q2_loc = q2_all[:, s0:s1].contiguous().requires_grad_(True)
    g_loc = g_all[:, s0:s1].contiguous().requires_grad_(True)
    beta_loc = beta_all[:, s0:s1].contiguous().requires_grad_(True)
    do_loc = do_all[:, s0:s1].contiguous()

    o_loc = triadic_gdn_cp8_step(
        q_loc, k_loc, v_loc, k2_loc, q2_loc, g_loc, beta_loc, omit_handoff=False
    )
    (o_loc * do_loc).sum().backward()

    o_gather = [torch.empty_like(o_loc) for _ in range(world_size)]
    dq_gather = [torch.empty_like(q_loc.grad) for _ in range(world_size)]
    dk2_gather = [torch.empty_like(k2_loc.grad) for _ in range(world_size)]
    dist.all_gather(o_gather, o_loc.detach())
    dist.all_gather(dq_gather, q_loc.grad.detach())
    dist.all_gather(dk2_gather, k2_loc.grad.detach())

    # Ablation: omit_handoff=True
    q_no = q_all[:, s0:s1].contiguous().requires_grad_(True)
    k_no = k_all[:, s0:s1].contiguous().requires_grad_(True)
    v_no = v_all[:, s0:s1].contiguous().requires_grad_(True)
    k2_no = k2_all[:, s0:s1].contiguous().requires_grad_(True)
    q2_no = q2_all[:, s0:s1].contiguous().requires_grad_(True)
    g_no = g_all[:, s0:s1].contiguous().requires_grad_(True)
    beta_no = beta_all[:, s0:s1].contiguous().requires_grad_(True)
    o_no = triadic_gdn_cp8_step(
        q_no, k_no, v_no, k2_no, q2_no, g_no, beta_no, omit_handoff=True
    )
    (o_no * do_loc).sum().backward()
    o_no_gather = [torch.empty_like(o_no) for _ in range(world_size)]
    dist.all_gather(o_no_gather, o_no.detach())

    parity_info = {}
    if rank == 0:
        q_s = q_all.clone().requires_grad_(True)
        k_s = k_all.clone().requires_grad_(True)
        v_s = v_all.clone().requires_grad_(True)
        k2_s = k2_all.clone().requires_grad_(True)
        q2_s = q2_all.clone().requires_grad_(True)
        g_s = g_all.clone().requires_grad_(True)
        b_s = beta_all.clone().requires_grad_(True)
        o_single, _ = chunk_triadic_gdn_rsi(q_s, k_s, v_s, k2_s, q2_s, g_s, b_s, use_qk_l2norm=True)
        (o_single * do_all).sum().backward()

        o_cp8 = torch.cat(o_gather, dim=1)
        dq_cp8 = torch.cat(dq_gather, dim=1)
        dk2_cp8 = torch.cat(dk2_gather, dim=1)
        o_no_cp8 = torch.cat(o_no_gather, dim=1)

        parity_info = {
            "cp8_with_handoff_o_rel_l2": rel_l2(o_cp8, o_single),
            "cp8_with_handoff_dq_rel_l2": rel_l2(dq_cp8, q_s.grad),
            "cp8_with_handoff_dk2_rel_l2": rel_l2(dk2_cp8, k2_s.grad),
            "cp8_omit_handoff_o_rel_l2": rel_l2(o_no_cp8, o_single),
        }

    # 5B. Scaling sweep across T_global in {16384, 32768, 65536}
    t_globals = [16384, 32768] if mode == "smoke" else [16384, 32768, 65536]
    warmup = 2 if mode == "smoke" else 4
    rep = 4 if mode == "smoke" else 10
    scaling = []

    for Tg in t_globals:
        Tl = Tg // world_size
        ql = torch.randn(1, Tl, 8, 64, device=device, dtype=torch.bfloat16)
        kl = torch.randn(1, Tl, 8, 64, device=device, dtype=torch.bfloat16)
        vl = torch.randn(1, Tl, 8, 64, device=device, dtype=torch.bfloat16)
        k2l = torch.randn(1, Tl, 8, 4, device=device, dtype=torch.bfloat16)
        q2l = torch.randn(1, Tl, 8, 4, device=device, dtype=torch.bfloat16)
        gl = -torch.rand(1, Tl, 8, 4, device=device, dtype=torch.float32) * 0.15 - 0.005
        bl = torch.sigmoid(torch.randn(1, Tl, 8, device=device, dtype=torch.float32)) * 1.5
        dol = torch.randn_like(vl)

        def run_cp8_fb():
            q_i = ql.detach().requires_grad_(True)
            k_i = kl.detach().requires_grad_(True)
            v_i = vl.detach().requires_grad_(True)
            k2_i = k2l.detach().requires_grad_(True)
            q2_i = q2l.detach().requires_grad_(True)
            g_i = gl.detach().requires_grad_(True)
            b_i = bl.detach().requires_grad_(True)
            o_i = triadic_gdn_cp8_step(q_i, k_i, v_i, k2_i, q2_i, g_i, b_i, omit_handoff=False)
            (o_i * dol).sum().backward()

        # Sequential ring baseline (rank r waits for rank r-1 before running its entire local fwd, and reverse for bwd)
        def run_ring_seq_fb():
            h_in = torch.zeros((1, 8, 4, 64, 64), device=device, dtype=torch.float32)
            if rank > 0:
                dist.recv(h_in, src=rank - 1)
            q_i = ql.detach().requires_grad_(True)
            k_i = kl.detach().requires_grad_(True)
            v_i = vl.detach().requires_grad_(True)
            k2_i = k2l.detach().requires_grad_(True)
            q2_i = q2l.detach().requires_grad_(True)
            g_i = gl.detach().requires_grad_(True)
            b_i = bl.detach().requires_grad_(True)
            o_i, h_out = chunk_triadic_gdn_rsi(
                q_i, k_i, v_i, k2_i, q2_i, g_i, b_i, initial_state=h_in, output_final_state=True, use_qk_l2norm=True
            )
            if rank < world_size - 1:
                dist.send(h_out.detach(), dst=rank + 1)
            dh_in = torch.zeros_like(h_in)
            if rank < world_size - 1:
                dist.recv(dh_in, src=rank + 1)
            (o_i * dol).sum().backward()
            if rank > 0:
                dist.send(dh_in, dst=rank - 1)

        for _ in range(warmup):
            run_cp8_fb()
            dist.barrier()
        torch.cuda.synchronize()
        cp8_times = []
        for _ in range(rep):
            dist.barrier()
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            run_cp8_fb()
            e_ev.record()
            torch.cuda.synchronize()
            cp8_times.append(s_ev.elapsed_time(e_ev))
        cp8_times.sort()
        cp8_ms = float(cp8_times[len(cp8_times) // 2])

        for _ in range(warmup):
            run_ring_seq_fb()
            dist.barrier()
        torch.cuda.synchronize()
        ring_times = []
        for _ in range(rep):
            dist.barrier()
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            run_ring_seq_fb()
            e_ev.record()
            torch.cuda.synchronize()
            ring_times.append(s_ev.elapsed_time(e_ev))
        ring_times.sort()
        ring_ms = float(ring_times[len(ring_times) // 2])

        if rank == 0:
            # Measure single-GPU full-sequence latency
            qs1 = torch.randn(1, Tg, 8, 64, device=device, dtype=torch.bfloat16)
            ks1 = torch.randn(1, Tg, 8, 64, device=device, dtype=torch.bfloat16)
            vs1 = torch.randn(1, Tg, 8, 64, device=device, dtype=torch.bfloat16)
            k2s1 = torch.randn(1, Tg, 8, 4, device=device, dtype=torch.bfloat16)
            q2s1 = torch.randn(1, Tg, 8, 4, device=device, dtype=torch.bfloat16)
            gs1 = -torch.rand(1, Tg, 8, 4, device=device, dtype=torch.float32) * 0.15 - 0.005
            bs1 = torch.sigmoid(torch.randn(1, Tg, 8, device=device, dtype=torch.float32)) * 1.5
            dos1 = torch.randn_like(vs1)

            def run_single1():
                q_i = qs1.detach().requires_grad_(True)
                k_i = ks1.detach().requires_grad_(True)
                v_i = vs1.detach().requires_grad_(True)
                k2_i = k2s1.detach().requires_grad_(True)
                q2_i = q2s1.detach().requires_grad_(True)
                g_i = gs1.detach().requires_grad_(True)
                b_i = bs1.detach().requires_grad_(True)
                o_i, _ = chunk_triadic_gdn_rsi(q_i, k_i, v_i, k2_i, q2_i, g_i, b_i, use_qk_l2norm=True)
                (o_i * dos1).sum().backward()

            single_ms = bench_cuda_fn(run_single1, warmup=warmup, rep=rep)
            scaling.append({
                "T_global": Tg,
                "T_local_per_gpu": Tl,
                "single_gpu_rsi_fb_ms": single_ms,
                "ring_seq_8gpu_fb_ms": ring_ms,
                "triadic_cp8_parallel_wy_fb_ms": cp8_ms,
                "cp8_ktok_s": (Tg / (cp8_ms * 1e-3)) / 1e3,
                "cp8_vs_single_gpu_speedup": single_ms / cp8_ms,
                "cp8_vs_ring_seq_speedup": ring_ms / cp8_ms,
            })
        dist.barrier()

    if rank == 0:
        return_dict["parity"] = parity_info
        return_dict["scaling"] = scaling
    dist.destroy_process_group()


def run_track5_cp8_scaling(mode: str = "full") -> Dict[str, Any]:
    world_size = torch.cuda.device_count()
    assert world_size == 8, f"Expected 8 A100 GPUs for Triadic-CP=8, found {world_size}"
    port = 29643 + (os.getpid() % 500)
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(_cp8_worker, args=(world_size, port, mode, return_dict), nprocs=world_size, join=True)
    return {
        "world_size": world_size,
        "parity": dict(return_dict["parity"]),
        "scaling": list(return_dict["scaling"]),
    }


def main():
    parser = argparse.ArgumentParser(description="E143 (v134, Round-17 RSI): Triadic-GDN & Triadic-KDA Benchmark Suite")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    t_start = time.time()
    print(f"[E143] Starting Round-17 RSI Triadic-GDN & Triadic-KDA benchmark (--mode {args.mode}, --rep {args.rep})...")

    track1 = run_track1_upstream_pathology_forensics()
    print(
        f"[E143][Track 1] Upstream spills={track1['sm80_cubin_spill_comparison']['upstream_total_cubin_spills']} "
        f"vs RSI spills={track1['sm80_cubin_spill_comparison']['rsi_total_cubin_spills']} "
        f"(max_regs={track1['sm80_cubin_spill_comparison']['rsi_max_regs']})"
    )

    track2 = run_track2_oracle_parity_and_stress()
    print(
        f"[E143][Track 2] Max FP32 rel_l2={track2['max_fp32_rel_l2']:.3e}, "
        f"Max BF16 rel_l2={track2['max_bf16_rel_l2']:.3e}"
    )

    track3 = run_track3_ablation_ladder(mode=args.mode)
    print(
        f"[E143][Track 3] Kernel sweep count={len(track3['kernel_sweep'])}, "
        f"Triadic-KDA speedup={track3['triadic_kda_fwd_benchmark']['rsi_vs_upstream_speedup']:.2f}x, "
        f"Block Stage A4 vs A0={track3['block_stage_ladder']['A4']['speedup_vs_A0']:.2f}x"
    )

    track4 = run_track4_autoresearch(mode=args.mode)
    print(
        f"[E143][Track 4] Autoresearch A0 steps={track4['arm1_eager_chunk_A0']['completed_steps']} "
        f"(bpb={track4['arm1_eager_chunk_A0']['final_val_bpb']:.4f}) -> "
        f"A4 steps={track4['arm3_round17_rsi_superfused_A4']['completed_steps']} "
        f"(bpb={track4['arm3_round17_rsi_superfused_A4']['final_val_bpb']:.4f})"
    )

    track5 = run_track5_cp8_scaling(mode=args.mode)
    print(
        f"[E143][Track 5] CP=8 handoff o rel_l2={track5['parity']['cp8_with_handoff_o_rel_l2']:.3e} "
        f"(omit_handoff={track5['parity']['cp8_omit_handoff_o_rel_l2']:.3e})"
    )

    payload = {
        "experiment_id": "E143",
        "version": "v134",
        "rsi_round": 17,
        "title": "Triadic Gated DeltaNet & Triadic Kimi Delta Attention (arXiv:2609.36529)",
        "mode": args.mode,
        "rep": args.rep,
        "device_name": torch.cuda.get_device_name(0),
        "world_size": torch.cuda.device_count(),
        "elapsed_sec": time.time() - t_start,
        "track1_upstream_pathology_forensics": track1,
        "track2_oracle_parity_and_stress": track2,
        "track3_ablation_ladder": track3,
        "track4_fixed_wallclock_autoresearch": track4,
        "track5_cp8_multi_gpu_scaling": track5,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[E143] Wrote {out_path} ({out_path.stat().st_size} bytes) in {payload['elapsed_sec']:.1f}s.")


if __name__ == "__main__":
    main()
