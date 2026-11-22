#!/usr/bin/env python3
"""E129: Round-3 Recursive Self-Improvement (`RSI`) Study of DeepSeek `mHC`
(`Manifold-Constrained Hyper-Connections`, `arXiv:2512.24880`) Multi-Stream Residual Kernels,
On-Chip Division-Free Log-Sinkhorn-20 Adjoint, Selective `L_r^*` Stream Checkpointing (`mHC-SAC`),
8-GPU NVSwitch `1.14B` `mHC-FSDP2`, & Fixed-Wall-Clock `val_bpb` Architecture-Kernel Co-Design Loop.

Governed by `PROGRAM.md` and byte-identical frozen `code/rsi_kernel_harness.py`
(`eval_harness_sha256 == 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
Co-designed and adversarially audited via `/copilot-app-frontier-cowork` (`GPT 6 Astra, Max`).
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import gc
import hashlib
import json
import math
import os
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_local_rank_env = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_e129_r{_local_rank_env}")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_e129_r{_local_rank_env}")

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

for _dyn_attr, _dyn_val in (
    ("recompile_limit", 64),
    ("cache_size_limit", 64),
    ("accumulated_recompile_limit", 256),
    ("accumulated_cache_size_limit", 256),
):
    if hasattr(torch._dynamo.config, _dyn_attr):
        setattr(torch._dynamo.config, _dyn_attr, _dyn_val)

from common import environment_metadata
from mhc_rsi_kernels import (
    DeepSeek2KernelStreamSandwichFn,
    FusedMHCCoeffFn,
    FusedPostResMergeFn,
    MHCStreamSegmentBank,
    RSISuperfusedStreamSandwichFn,
    SACFusedMHCCoeffFn,
    SACFusedPostResMergeFn,
    SACSuperfusedPreRMSNormFn,
    SuperfusedPreRMSNormFn,
    _fused_post_res_fwd_kernel,
    _mhc_epilogue_log_sinkhorn20_bwd_ckpt5_kernel,
    _mhc_epilogue_log_sinkhorn20_fwd_kernel,
    _rsi_unified_sandwich_bwd_split_kernel,
    _single_scan_dx_norm_proj_bwd_kernel,
    _single_scan_proj_rms_fwd_kernel,
    _superfused_pre_rmsnorm_fwd_kernel,
    eager_mhc_coeff_eq7_eq9,
    eager_mhc_stream_sandwich,
    fp64_mhc_coeff_forward,
    fp64_mhc_coeff_oracle,
    fp64_mhc_stream_forward,
    fp64_mhc_stream_oracle,
    fused_mhc_coeff,
    inspect_mhc_cubin_spills,
    log_sinkhorn_knopp_pytorch,
    naive_colmax_exp_sinkhorn_knopp,
)
from rsi_kernel_harness import (
    HARNESS_VERSION,
    PrecisionStateSnapshot,
    RSIVectorRatchet,
    benchmark_rotating_cuda,
    compare_tensor_to_fp64,
    eval_harness_sha256,
    hash_callable_sources,
    query_sm_telemetry,
    richardson_unit_rms_audit,
    write_rsi_ledger_tsv,
)
from rsi_round2_kernels import FusedSwiGLUFn


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    return float(torch.linalg.norm(af - bf).item() / max(1e-12, torch.linalg.norm(bf).item()))


# ============================================================================
# 0. Chunked 3-GEMM Fused Linear Cross-Entropy (`V = 128,256` / `4,096`)
# ============================================================================


class ChunkedLinearCrossEntropyFn(torch.autograd.Function):
    """Memory-minimal token-chunked 3-GEMM Linear Cross-Entropy returning exact sum of token NLLs (nats)."""

    @staticmethod
    def forward(ctx, h: torch.Tensor, w_lm: torch.Tensor, targets: torch.Tensor, chunk_size: int = 2048):
        h_c = h.contiguous()
        N, D = h_c.shape
        total_nll = torch.zeros((), device=h.device, dtype=torch.float32)
        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)
            logits = F.linear(h_c[s:e], w_lm).float()
            total_nll = total_nll + F.cross_entropy(logits, targets[s:e], reduction="sum")
        ctx.save_for_backward(h_c, w_lm, targets)
        ctx.chunk_size = chunk_size
        return total_nll

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h_c, w_lm, targets = ctx.saved_tensors
        N, D = h_c.shape
        cs = ctx.chunk_size
        dh = torch.empty_like(h_c)
        dw = torch.zeros_like(w_lm, dtype=torch.float32)
        g_scale = float(grad_out.float().item())
        for s in range(0, N, cs):
            e = min(N, s + cs)
            hc = h_c[s:e]
            logits = F.linear(hc, w_lm).float()
            probs = torch.softmax(logits, dim=-1)
            probs.scatter_add_(
                1,
                targets[s:e].unsqueeze(1),
                torch.full((e - s, 1), -1.0, device=h_c.device, dtype=torch.float32),
            )
            probs.mul_(g_scale)
            p_bf16 = probs.to(h_c.dtype)
            dh[s:e] = p_bf16 @ w_lm
            dw.add_(torch.mm(p_bf16.t(), hc, out_dtype=torch.float32))
        return dh, dw.to(w_lm.dtype), None, None


# ============================================================================
# 1. Suite 0: Adversarial Red-Team Torture + GPT-6-Astra Extreme-Logit Audit
#    + Per-Tensor FP64 Parity + 0-Spill Cubin Check + Unit-RMS O(h^2) Richardson
# ============================================================================


def run_suite_0_mhc_audits(device: torch.device, is_smoke: bool) -> tuple[dict[str, Any], RSIVectorRatchet]:
    redteam = RSIVectorRatchet(
        track="mhc_adversarial_redteam_audit",
        shape_keys=["N256_n4", "N512_n4"],
        hbm_cap_mib=16000.0,
    )

    gen = torch.Generator(device=device).manual_seed(20261290)
    C_rt = 512
    n_rt = 4
    M_rt = n_rt * n_rt + 2 * n_rt

    def _make_rt_inputs(N_tok: int, seed_off: int, extreme_row: bool = False):
        g = torch.Generator(device=device).manual_seed(20261290 + seed_off)
        X = (torch.randn((N_tok, n_rt, C_rt), device=device, dtype=torch.bfloat16, generator=g) * 0.6).contiguous()
        phi = (torch.randn((n_rt * C_rt, M_rt), device=device, dtype=torch.bfloat16, generator=g) / math.sqrt(n_rt * C_rt)).contiguous()
        if not extreme_row:
            # Smooth positive shift across all channels gives large `<X, V @ A^T>` (`rt3` drop_dr -> ~3% dX error = REJECT_CORRECTNESS)
            # without creating artificial 16x BF16 outliers in `phi`.
            X = (X + 0.9).contiguous()
            phi = (phi + 0.015).contiguous()
        gamma = (torch.ones((n_rt * C_rt,), device=device, dtype=torch.bfloat16) + 0.1 * torch.randn((n_rt * C_rt,), device=device, dtype=torch.bfloat16, generator=g)).contiguous()
        # Keep pre/post alpha unsaturated (0.12) so distinct inputs X_a != X_b yield distinct h_pre (`rt2` static alias caught!),
        # while residual alpha = 0.85 strongly exercises Sinkhorn-20 + Norm-After-Matmul `dr` VJP (`rt3`, `rt4` caught!).
        alpha = torch.tensor([0.12, 0.12, 0.85], device=device, dtype=torch.float32)
        b = (torch.randn((M_rt,), device=device, dtype=torch.float32, generator=g) * 0.4).contiguous()
        if extreme_row:
            alpha = torch.tensor([0.1, 0.1, 0.5], device=device, dtype=torch.float32)
            br = torch.zeros((n_rt, n_rt), device=device, dtype=torch.float32)
            br[0, :] = 115.0
            br += 1.5 * torch.eye(n_rt, device=device, dtype=torch.float32)
            b[2 * n_rt :] = br.reshape(-1)
        dhp = torch.randn((N_tok, n_rt), device=device, dtype=torch.float32, generator=g) * 0.3 + 0.4
        dho = torch.randn((N_tok, n_rt), device=device, dtype=torch.float32, generator=g) * 0.3 + 0.4
        dhr = torch.randn((N_tok, n_rt, n_rt), device=device, dtype=torch.float32, generator=g) * 0.3
        return X, phi, gamma, alpha, b, dhp, dho, dhr

    X_a, phi_a, gam_a, alp_a, b_a, dhp_a, dho_a, dhr_a = _make_rt_inputs(192, 1)
    X_b, _, _, _, _, _, _, _ = _make_rt_inputs(192, 2)
    # Skewed off-diagonal logits where finite-20 Sinkhorn contraction factor rho ~ 0.92 exposes row-shift bias (`rt4`)!
    b_skew = b_a.clone()
    b_skew[2 * n_rt :] = torch.tensor(
        [3.2, 3.0, -3.0, -3.2, 3.0, 3.2, -3.1, -3.0, -3.1, -3.0, 3.1, 3.0, -3.2, -3.1, 3.0, 3.2],
        device=device, dtype=torch.float32,
    )
    ora_rt = fp64_mhc_coeff_oracle(X_a, phi_a, gam_a, alp_a, b_skew, dhp_a, dho_a, dhr_a)

    static_hp = torch.empty((192, n_rt), device=device, dtype=torch.float32)

    def _eval_rt_candidate(tid: int, cid: str, mode: str, note: str):
        snap_before = PrecisionStateSnapshot.capture()
        Xi = X_a.clone().requires_grad_(True)
        phii = phi_a.clone().requires_grad_(True)
        gami = gam_a.clone().requires_grad_(True)
        alpi = alp_a.clone().requires_grad_(True)
        bi = b_skew.clone().requires_grad_(True)

        mutated_flag = False
        try:
            if mode == "honest":
                hp, ho, hr = fused_mhc_coeff(Xi, phii, gami, alpi, bi, single_scan=True)
            elif mode == "tf32_mutator":
                torch.backends.cuda.matmul.allow_tf32 = not snap_before.allow_tf32_matmul
                hp, ho, hr = fused_mhc_coeff(Xi, phii, gami, alpi, bi, single_scan=True)
            elif mode == "static_alias":
                hp_tmp, ho, hr = fused_mhc_coeff(Xi, phii, gami, alpi, bi, single_scan=True)
                static_hp.copy_(hp_tmp.detach())
                hp = static_hp + 0.0 * hp_tmp.sum()
            elif mode == "drop_dr_term":
                # Subtle GPT 6 Astra Max Q2 bug: omits -x * sum_j(q*u)*inv_r^3/K denominator gradient in Norm-After-Matmul!
                hp, ho, hr = fused_mhc_coeff(Xi, phii, gami, alpi, bi, single_scan=True, drop_dr_exploit=True)
            else:
                # Subtle GPT 6 Astra Max Q1 bug: subtracts row-max instead of col-max before T_c on finite 20-step Sinkhorn!
                N_t = Xi.shape[0]
                K_t = n_rt * C_rt
                xf = Xi.reshape(N_t, K_t).float()
                rstd = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-6)
                u = ((xf * rstd * gami.float()).to(Xi.dtype) @ phii).float()
                zp = alpi[0] * u[:, :n_rt] + bi[:n_rt]
                zo = alpi[1] * u[:, n_rt : 2 * n_rt] + bi[n_rt : 2 * n_rt]
                zr = (alpi[2] * u[:, 2 * n_rt :] + bi[2 * n_rt :]).view(N_t, n_rt, n_rt)
                hp = torch.sigmoid(zp)
                ho = 2.0 * torch.sigmoid(zo)
                # Wrong axis shift: row-max (-1) before column norm (-2) alters finite-20 trajectory!
                m = torch.exp(zr - zr.amax(dim=-1, keepdim=True).detach())
                for _ in range(20):
                    m = m / m.sum(dim=-2, keepdim=True)
                    m = m / m.sum(dim=-1, keepdim=True)
                hr = m

            snap_after = PrecisionStateSnapshot.capture()
            if snap_after != snap_before:
                mutated_flag = True
        finally:
            torch.backends.cuda.matmul.allow_tf32 = snap_before.allow_tf32_matmul
            torch.backends.cudnn.allow_tf32 = snap_before.allow_tf32_cudnn
            torch.set_float32_matmul_precision(snap_before.float32_matmul_precision)

        ((hp * dhp_a).sum() + (ho * dho_a).sum() + (hr * dhr_a).sum()).backward()

        # Check static buffer aliasing across two back-to-back calls
        if mode == "static_alias":
            hp1 = static_hp
            snap_1 = hp1.clone()
            hp_tmp2, _, _ = fused_mhc_coeff(X_b, phi_a, gam_a, alp_a, b_skew, single_scan=True)
            static_hp.copy_(hp_tmp2.detach())
            non_alias = bool(torch.equal(hp1, snap_1))
        else:
            hp1, _, _ = fused_mhc_coeff(X_a, phi_a, gam_a, alp_a, b_skew, single_scan=True)
            snap_1 = hp1.clone()
            _ = fused_mhc_coeff(X_b, phi_a, gam_a, alp_a, b_skew, single_scan=True)
            non_alias = bool(torch.equal(hp1, snap_1))

        in_unmut = bool(torch.equal(Xi.detach(), X_a))
        p_cks = [
            compare_tensor_to_fp64(hp, ora_rt["h_pre"], rel_tol=1e-2, name="h_pre"),
            compare_tensor_to_fp64(ho, ora_rt["h_post"], rel_tol=1e-2, name="h_post"),
            compare_tensor_to_fp64(hr, ora_rt["h_res"], rel_tol=1e-2, name="h_res"),
            compare_tensor_to_fp64(Xi.grad, ora_rt["dX"], rel_tol=1.5e-2, name="dX"),
            compare_tensor_to_fp64(phii.grad, ora_rt["d_phi"], rel_tol=1.5e-2, name="d_phi"),
            compare_tensor_to_fp64(gami.grad, ora_rt["d_gamma"], rel_tol=1.5e-2, name="d_gamma"),
            compare_tensor_to_fp64(alpi.grad, ora_rt["d_alpha"], rel_tol=1.5e-2, name="d_alpha"),
            compare_tensor_to_fp64(bi.grad, ora_rt["d_b"], rel_tol=1.5e-2, name="d_b"),
        ]
        fake_perf = {
            "N256_n4": {"median_ms": 0.25 if tid == 0 else 0.22, "peak_alloc_mib": 32.0},
            "N512_n4": {"median_ms": 0.42 if tid == 0 else 0.37, "peak_alloc_mib": 48.0},
        }
        return redteam.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=hash_callable_sources(cid, mode),
            parity_checks=p_cks,
            richardson_checks=[],
            anti_cheat_checks={
                "precision_flags_preserved": not mutated_flag,
                "input_unmutated": in_unmut,
                "output_non_aliasing": non_alias,
            },
            per_shape_metrics=fake_perf,
            hypothesis_note=note,
        )

    _eval_rt_candidate(0, "rt0_honest_mhc_fused_baseline", "honest", "Honest Round-3 fused mHC baseline")
    _eval_rt_candidate(1, "rt1_tf32_flag_mutator", "tf32_mutator", "Exploit mutating allow_tf32 flag")
    _eval_rt_candidate(2, "rt2_static_output_buffer_aliaser", "static_alias", "Exploit reusing static output buffer")
    _eval_rt_candidate(3, "rt3_norm_after_matmul_drops_dr_term", "drop_dr_term", "Bug omitting -x*sum(q*u)*inv_r^3/K denominator VJP")
    _eval_rt_candidate(4, "rt4_rowmax_before_colnorm_finite20_bias", "rowmax_bias", "Bug subtracting row-max before T_c on skewed finite-20 Sinkhorn")

    # ------------------------------------------------------------------------
    # 0.2 GPT 6 Astra Max Q1 Extreme-Logit (+120 Row-Offset) Sinkhorn Audit
    # ------------------------------------------------------------------------
    X_ex, phi_ex, gam_ex, alp_ex, b_ex, dhp_ex, dho_ex, dhr_ex = _make_rt_inputs(512, 99, extreme_row=True)
    ora_ex = fp64_mhc_coeff_oracle(X_ex, phi_ex, gam_ex, alp_ex, b_ex, dhp_ex, dho_ex, dhr_ex)

    # Unshifted exp(z) & Col-Max exp(z - col_max) vs On-Chip Log-Domain Sinkhorn-20
    _, _, hr_colmax_broken = eager_mhc_coeff_eq7_eq9(
        X_ex, phi_ex, gam_ex, alp_ex, b_ex, use_broken_colmax_exp=True
    )
    colmax_nan_count = int(torch.isnan(hr_colmax_broken).sum().item())

    X_ex_t = X_ex.clone().requires_grad_(True)
    phi_ex_t = phi_ex.clone().requires_grad_(True)
    gam_ex_t = gam_ex.clone().requires_grad_(True)
    alp_ex_t = alp_ex.clone().requires_grad_(True)
    b_ex_t = b_ex.clone().requires_grad_(True)
    hp_log, ho_log, hr_log = fused_mhc_coeff(X_ex_t, phi_ex_t, gam_ex_t, alp_ex_t, b_ex_t, single_scan=True)
    ((hp_log * dhp_ex).sum() + (ho_log * dho_ex).sum() + (hr_log * dhr_ex).sum()).backward()

    row_sum_err = float((hr_log.sum(dim=-1) - 1.0).abs().max().item())
    col_sum_err = float((hr_log.sum(dim=-2) - 1.0).abs().max().item())
    extreme_logit_audit = {
        "row_offset_added": 115.0,
        "total_matrix_elements": int(hr_log.numel()),
        "colmax_exp_sinkhorn_nan_elements": colmax_nan_count,
        "onchip_log_sinkhorn_nan_elements": int(torch.isnan(hr_log).sum().item() + torch.isnan(X_ex_t.grad).sum().item()),
        "onchip_log_sinkhorn_hr_fp64_rel_l2": round(rel_l2(hr_log, ora_ex["h_res"]), 8),
        "onchip_log_sinkhorn_dX_fp64_rel_l2": round(rel_l2(X_ex_t.grad, ora_ex["dX"]), 8),
        "onchip_log_sinkhorn_db_fp64_rel_l2": round(rel_l2(b_ex_t.grad, ora_ex["d_b"]), 8),
        "birkhoff_max_row_sum_abs_err": round(row_sum_err, 8),
        "birkhoff_max_col_sum_abs_err": round(col_sum_err, 8),
    }

    # ------------------------------------------------------------------------
    # 0.3 Per-Tensor FP64 Parity Across `n in {2, 4, 8}`, Ragged `N = 517`,
    #     Non-Contiguous Sliced Views, & 0-Spill Cubin Inspection
    # ------------------------------------------------------------------------
    fp64_cases: dict[str, Any] = {}
    for n_s in (2, 4, 8):
        N_rag = 517
        C_p = 768
        M_p = n_s * n_s + 2 * n_s
        gp = torch.Generator(device=device).manual_seed(8800 + n_s)
        X_pad = torch.randn((N_rag + 11, n_s + 2, C_p), device=device, dtype=torch.bfloat16, generator=gp) * 0.7
        X_nc = X_pad[:N_rag, :n_s, :]  # non-contiguous view!
        phi_p = (torch.randn((n_s * C_p, M_p), device=device, dtype=torch.bfloat16, generator=gp) * 0.3).contiguous()
        gam_p = (torch.ones((n_s * C_p,), device=device, dtype=torch.bfloat16) + 0.1 * torch.randn((n_s * C_p,), device=device, dtype=torch.bfloat16, generator=gp)).contiguous()
        alp_p = torch.tensor([0.05, 0.08, 0.12], device=device, dtype=torch.float32)
        b_p = (torch.randn((M_p,), device=device, dtype=torch.float32, generator=gp) * 0.4).contiguous()
        dhp_p = torch.randn((N_rag, n_s), device=device, dtype=torch.float32, generator=gp) * 0.4
        dho_p = torch.randn((N_rag, n_s), device=device, dtype=torch.float32, generator=gp) * 0.4
        dhr_p = torch.randn((N_rag, n_s, n_s), device=device, dtype=torch.float32, generator=gp) * 0.4

        ora_c = fp64_mhc_coeff_oracle(X_nc.contiguous(), phi_p, gam_p, alp_p, b_p, dhp_p, dho_p, dhr_p)
        Xi = X_nc.detach().requires_grad_(True)
        phii = phi_p.clone().requires_grad_(True)
        gami = gam_p.clone().requires_grad_(True)
        alpi = alp_p.clone().requires_grad_(True)
        bi = b_p.clone().requires_grad_(True)
        hp_c, ho_c, hr_c = fused_mhc_coeff(Xi, phii, gami, alpi, bi, single_scan=True)
        ((hp_c * dhp_p).sum() + (ho_c * dho_p).sum() + (hr_c * dhr_p).sum()).backward()

        # Stream sandwich parity (`RSISuperfusedStreamSandwichFn`) on the same non-contiguous view
        y_p = (torch.randn((N_rag, C_p), device=device, dtype=torch.bfloat16, generator=gp) * 0.6).contiguous()
        w_sub_p = (torch.ones((C_p,), device=device, dtype=torch.bfloat16) + 0.15 * torch.randn((C_p,), device=device, dtype=torch.bfloat16, generator=gp)).contiguous()
        dhn_p = torch.randn_like(y_p) * 0.4
        dxn_p = torch.randn((N_rag, n_s, C_p), device=device, dtype=torch.bfloat16, generator=gp) * 0.4

        ora_s = fp64_mhc_stream_oracle(
            X_nc.contiguous(), y_p, hp_c.detach(), ho_c.detach(), hr_c.detach(), w_sub_p, dhn_p, dxn_p
        )
        Xs = X_nc.detach().requires_grad_(True)
        ys = y_p.clone().requires_grad_(True)
        hps = hp_c.detach().clone().requires_grad_(True)
        hos = ho_c.detach().clone().requires_grad_(True)
        hrs = hr_c.detach().clone().requires_grad_(True)
        ws = w_sub_p.clone().requires_grad_(True)
        hn_s, xn_s = RSISuperfusedStreamSandwichFn.apply(Xs, ys, hps, hos, hrs, ws, 1e-6)
        torch.autograd.backward([hn_s, xn_s], [dhn_p, dxn_p])

        coeff_rels = {
            "h_pre": rel_l2(hp_c, ora_c["h_pre"]),
            "h_post": rel_l2(ho_c, ora_c["h_post"]),
            "h_res": rel_l2(hr_c, ora_c["h_res"]),
            "dX": rel_l2(Xi.grad, ora_c["dX"]),
            "d_phi": rel_l2(phii.grad, ora_c["d_phi"]),
            "d_gamma": rel_l2(gami.grad, ora_c["d_gamma"]),
            "d_alpha": rel_l2(alpi.grad, ora_c["d_alpha"]),
            "d_b": rel_l2(bi.grad, ora_c["d_b"]),
        }
        stream_rels = {
            "h_norm": rel_l2(hn_s, ora_s["h_norm"]),
            "X_next": rel_l2(xn_s, ora_s["X_next"]),
            "dX": rel_l2(Xs.grad, ora_s["dX"]),
            "dy": rel_l2(ys.grad, ora_s["dy"]),
            "dh_pre": rel_l2(hps.grad, ora_s["dh_pre"]),
            "dh_post": rel_l2(hos.grad, ora_s["dh_post"]),
            "dh_res": rel_l2(hrs.grad, ora_s["dh_res"]),
            "dw_sub": rel_l2(ws.grad, ora_s["dw_sub"]),
        }
        fp64_cases[f"ragged_noncontig_n{n_s}"] = {
            "coeff_max_rel_l2": round(max(coeff_rels.values()), 6),
            "stream_max_rel_l2": round(max(stream_rels.values()), 6),
            "coeff_per_tensor_rel_l2": {k: round(v, 6) for k, v in coeff_rels.items()},
            "stream_per_tensor_rel_l2": {k: round(v, 6) for k, v in stream_rels.items()},
        }

    cubin_report = inspect_mhc_cubin_spills()

    # ------------------------------------------------------------------------
    # 0.4 Pure FP64 Unit-RMS O(h^2) Richardson Extrapolation (~4.0000x, 8 Derivatives)
    # ------------------------------------------------------------------------
    gr = torch.Generator(device=device).manual_seed(20261295)
    N_r, n_r, C_r = 32, 4, 128
    M_r = n_r * n_r + 2 * n_r
    X64 = (torch.randn((N_r, n_r, C_r), device=device, dtype=torch.float64, generator=gr) * 0.7).requires_grad_(True)
    phi64 = (torch.randn((n_r * C_r, M_r), device=device, dtype=torch.float64, generator=gr) * 0.4).requires_grad_(True)
    gam64 = (torch.ones((n_r * C_r,), device=device, dtype=torch.float64) + 0.2 * torch.randn((n_r * C_r,), device=device, dtype=torch.float64, generator=gr)).requires_grad_(True)
    alp64 = torch.tensor([0.6, 0.5, 0.9], device=device, dtype=torch.float64, requires_grad=True)
    b64 = (torch.randn((M_r,), device=device, dtype=torch.float64, generator=gr) * 0.3).requires_grad_(True)
    dhp64 = torch.randn((N_r, n_r), device=device, dtype=torch.float64, generator=gr)
    dho64 = torch.randn((N_r, n_r), device=device, dtype=torch.float64, generator=gr)
    dhr64 = torch.randn((N_r, n_r, n_r), device=device, dtype=torch.float64, generator=gr)

    hp64, ho64, hr64 = fp64_mhc_coeff_forward(X64, phi64, gam64, alp64, b64)
    ((hp64 * dhp64).sum() + (ho64 * dho64).sum() + (hr64 * dhr64).sum()).backward()

    # Standalone 20-step Sinkhorn-Knopp non-linear VJP Richardson check
    zr64 = (torch.randn((N_r, n_r, n_r), device=device, dtype=torch.float64, generator=gr) * 0.9).requires_grad_(True)
    (log_sinkhorn_knopp_pytorch(zr64, t_max=20) * dhr64).sum().backward()

    # Non-linear stream readout + RMSNorm VJP Richardson checks (`dX`, `dH_pre`, `dw_sub` under quadratic loss so `dw_sub` has curvature!)
    Xs64 = (torch.randn((N_r, n_r, C_r), device=device, dtype=torch.float64, generator=gr) * 0.8).requires_grad_(True)
    ys64 = (torch.randn((N_r, C_r), device=device, dtype=torch.float64, generator=gr) * 0.6).requires_grad_(True)
    hps64 = torch.sigmoid(torch.randn((N_r, n_r), device=device, dtype=torch.float64, generator=gr)).detach().requires_grad_(True)
    hos64 = (2.0 * torch.sigmoid(torch.randn((N_r, n_r), device=device, dtype=torch.float64, generator=gr))).detach().requires_grad_(True)
    hrs64 = log_sinkhorn_knopp_pytorch(torch.randn((N_r, n_r, n_r), device=device, dtype=torch.float64, generator=gr)).detach().requires_grad_(True)
    ws64 = (torch.ones((C_r,), device=device, dtype=torch.float64) + 0.2 * torch.randn((C_r,), device=device, dtype=torch.float64, generator=gr)).requires_grad_(True)

    def _stream_nonlinear_loss(X_in, y_in, hp_in, ho_in, hr_in, w_in):
        hn_v, xn_v = fp64_mhc_stream_forward(X_in, y_in, hp_in, ho_in, hr_in, w_in)
        return torch.sin(hn_v).sum() + 0.25 * (xn_v * xn_v).sum()

    _stream_nonlinear_loss(Xs64, ys64, hps64, hos64, hrs64, ws64).backward()

    richardson_checks = [
        richardson_unit_rms_audit(
            lambda z: (
                fp64_mhc_coeff_forward(z, phi64.detach(), gam64.detach(), alp64.detach(), b64.detach())[0] * dhp64
            ).sum()
            + (
                fp64_mhc_coeff_forward(z, phi64.detach(), gam64.detach(), alp64.detach(), b64.detach())[1] * dho64
            ).sum()
            + (
                fp64_mhc_coeff_forward(z, phi64.detach(), gam64.detach(), alp64.detach(), b64.detach())[2] * dhr64
            ).sum(),
            X64.detach(), X64.grad, seed=301, name="mhc_coeff_dX",
        ),
        richardson_unit_rms_audit(
            lambda z: (
                fp64_mhc_coeff_forward(X64.detach(), z, gam64.detach(), alp64.detach(), b64.detach())[2] * dhr64
            ).sum(),
            phi64.detach(),
            torch.autograd.grad(
                (fp64_mhc_coeff_forward(X64.detach(), phi64, gam64.detach(), alp64.detach(), b64.detach())[2] * dhr64).sum(),
                phi64,
            )[0],
            seed=302, name="mhc_coeff_dphi_through_sinkhorn20",
        ),
        richardson_unit_rms_audit(
            lambda z: (
                fp64_mhc_coeff_forward(X64.detach(), phi64.detach(), z, alp64.detach(), b64.detach())[2] * dhr64
            ).sum(),
            gam64.detach(),
            torch.autograd.grad(
                (fp64_mhc_coeff_forward(X64.detach(), phi64.detach(), gam64, alp64.detach(), b64.detach())[2] * dhr64).sum(),
                gam64,
            )[0],
            seed=303, name="mhc_coeff_dgamma_through_sinkhorn20",
        ),
        richardson_unit_rms_audit(
            lambda z: (
                fp64_mhc_coeff_forward(X64.detach(), phi64.detach(), gam64.detach(), z, b64.detach())[0] * dhp64
            ).sum()
            + (
                fp64_mhc_coeff_forward(X64.detach(), phi64.detach(), gam64.detach(), z, b64.detach())[1] * dho64
            ).sum()
            + (
                fp64_mhc_coeff_forward(X64.detach(), phi64.detach(), gam64.detach(), z, b64.detach())[2] * dhr64
            ).sum(),
            alp64.detach(), alp64.grad, seed=304, name="mhc_coeff_dalpha",
        ),
        richardson_unit_rms_audit(
            lambda z: (log_sinkhorn_knopp_pytorch(z, t_max=20) * dhr64).sum(),
            zr64.detach(), zr64.grad, seed=305, name="sinkhorn20_dz_res",
        ),
        richardson_unit_rms_audit(
            lambda z: _stream_nonlinear_loss(z, ys64.detach(), hps64.detach(), hos64.detach(), hrs64.detach(), ws64.detach()),
            Xs64.detach(), Xs64.grad, seed=306, name="mhc_stream_sandwich_dX",
        ),
        richardson_unit_rms_audit(
            lambda z: _stream_nonlinear_loss(Xs64.detach(), ys64.detach(), z, hos64.detach(), hrs64.detach(), ws64.detach()),
            hps64.detach(), hps64.grad, seed=307, name="mhc_stream_sandwich_dH_pre",
        ),
        richardson_unit_rms_audit(
            lambda z: _stream_nonlinear_loss(Xs64.detach(), ys64.detach(), hps64.detach(), hos64.detach(), hrs64.detach(), z),
            ws64.detach(), ws64.grad, seed=308, name="mhc_stream_sandwich_dw_sub",
        ),
    ]

    return (
        {
            "redteam_gate_audit": redteam.to_summary_dict(),
            "extreme_logit_sinkhorn_audit": extreme_logit_audit,
            "per_tensor_fp64_parity": fp64_cases,
            "cubin_spill_inspection": cubin_report,
            "richardson_oh2_checks": richardson_checks,
        },
        redteam,
    )


# ============================================================================
# 2. Suite A: Round-3 Multi-Trial RSI Vector-Pareto Ratchets (Track 1 & Track 2)
# ============================================================================


def run_suite_a_mhc_ratchets(
    device: torch.device, is_smoke: bool
) -> tuple[dict[str, Any], list[RSIVectorRatchet]]:
    l2_flush = torch.zeros((28 * 1024 * 1024,), device=device, dtype=torch.float32)  # 112 MiB > 2.8x A100 L2!
    C_dim = 3584

    # ------------------------------------------------------------------------
    # Track 1: `mhc_coefficient_and_sinkhorn_manifold` (`C = 3,584`, `n in {2, 4, 8}`)
    # ------------------------------------------------------------------------
    t1_shapes = (
        [("N2048_n2", 2048, 2), ("N2048_n4", 2048, 4), ("N2048_n8", 2048, 8)]
        if is_smoke
        else [
            ("N4096_n2", 4096, 2),
            ("N4096_n4", 4096, 4),
            ("N4096_n8", 4096, 8),
            ("N16384_n2", 16384, 2),
            ("N16384_n4", 16384, 4),
            ("N16384_n8", 16384, 8),
        ]
    )
    ratchet_t1 = RSIVectorRatchet(
        track="mhc_coefficient_and_sinkhorn_manifold",
        shape_keys=[s[0] for s in t1_shapes],
        hbm_cap_mib=48000.0,
    )

    comp_coeff = torch.compile(eager_mhc_coeff_eq7_eq9, dynamic=False)

    t1_candidates = [
        (
            0,
            "trial0_eager_eq7_eq9_rmsnorm_sinkhorn20",
            hash_callable_sources(eager_mhc_coeff_eq7_eq9, log_sinkhorn_knopp_pytorch),
            "eager",
            "Literal Eq. 7-9: materializes [N, nC] RMSNorm state + 80+ unrolled PyTorch Sinkhorn-20 kernel launches",
        ),
        (
            1,
            "trial1_naive_colmax_exp_sinkhorn20_row_underflow",
            hash_callable_sources(naive_colmax_exp_sinkhorn_knopp),
            "broken_colmax",
            "Col-max shifted exp(Z - col_max) Sinkhorn-20: overflows/underflows whole rows to 0/0=NaN at +115 row offset!",
        ),
        (
            2,
            "trial2_torch_compile_inductor_eq7_eq9",
            hash_callable_sources(eager_mhc_coeff_eq7_eq9, "torch.compile"),
            "compiled",
            "TorchInductor compiled Eq. 7-9: fuses pointwise norms but still materializes [N, nC] & unrolled Sinkhorn graph",
        ),
        (
            3,
            "trial3_cublas_norm_after_matmul_plus_onchip_log_sinkhorn20",
            hash_callable_sources(
                FusedMHCCoeffFn, _mhc_epilogue_log_sinkhorn20_fwd_kernel, _mhc_epilogue_log_sinkhorn20_bwd_ckpt5_kernel
            ),
            "cublas_norm_after",
            "DeepSeek Eq. 10-19 Norm-After-Matmul (cuBLAS + row-RMS) + 0-spill On-Chip 5-Segment Log-Sinkhorn-20 VJP",
        ),
        (
            4,
            "trial4_single_scan_tensorcore_proj_rms_plus_onchip_log_sinkhorn20",
            hash_callable_sources(
                _single_scan_proj_rms_fwd_kernel,
                _single_scan_dx_norm_proj_bwd_kernel,
                _mhc_epilogue_log_sinkhorn20_fwd_kernel,
                _mhc_epilogue_log_sinkhorn20_bwd_ckpt5_kernel,
            ),
            "single_scan",
            "Single-Scan Tensor-Core [N,nC]@A + Row-RMS (1 HBM read of X in fwd & 1 in bwd!) + On-Chip Log-Sinkhorn-20",
        ),
    ]

    # Gate G3/G5 torture inputs for Track 1 (includes both ragged moderate AND +115 extreme row-offset logits!)
    g_t1 = torch.Generator(device=device).manual_seed(41001)
    N_g1, n_g1 = 259, 4
    M_g1 = n_g1 * n_g1 + 2 * n_g1
    X_g1 = (torch.randn((N_g1, n_g1, C_dim), device=device, dtype=torch.bfloat16, generator=g_t1) * 0.7).contiguous()
    X_g1_b = (torch.randn((N_g1, n_g1, C_dim), device=device, dtype=torch.bfloat16, generator=g_t1) * 0.7).contiguous()
    phi_g1 = (torch.randn((n_g1 * C_dim, M_g1), device=device, dtype=torch.bfloat16, generator=g_t1) * 0.25).contiguous()
    gam_g1 = (torch.ones((n_g1 * C_dim,), device=device, dtype=torch.bfloat16) + 0.1 * torch.randn((n_g1 * C_dim,), device=device, dtype=torch.bfloat16, generator=g_t1)).contiguous()
    alp_g1 = torch.tensor([0.05, 0.05, 0.10], device=device, dtype=torch.float32)
    b_g1_mod = (torch.randn((M_g1,), device=device, dtype=torch.float32, generator=g_t1) * 0.3).contiguous()
    b_g1_ext = b_g1_mod.clone()
    b_g1_ext[2 * n_g1 : 3 * n_g1] += 115.0  # Extreme row-0 offset (`GPT 6 Astra, Max` Q1 torture!)
    dhp_g1 = torch.randn((N_g1, n_g1), device=device, dtype=torch.float32, generator=g_t1) * 0.3
    dho_g1 = torch.randn((N_g1, n_g1), device=device, dtype=torch.float32, generator=g_t1) * 0.3
    dhr_g1 = torch.randn((N_g1, n_g1, n_g1), device=device, dtype=torch.float32, generator=g_t1) * 0.3

    ora_g1_mod = fp64_mhc_coeff_oracle(X_g1, phi_g1, gam_g1, alp_g1, b_g1_mod, dhp_g1, dho_g1, dhr_g1)
    ora_g1_ext = fp64_mhc_coeff_oracle(X_g1, phi_g1, gam_g1, alp_g1, b_g1_ext, dhp_g1, dho_g1, dhr_g1)

    def _run_t1(mtag: str, X, phi, gam, alp, b):
        if mtag == "eager":
            return eager_mhc_coeff_eq7_eq9(X, phi, gam, alp, b, use_broken_colmax_exp=False)
        if mtag == "broken_colmax":
            return eager_mhc_coeff_eq7_eq9(X, phi, gam, alp, b, use_broken_colmax_exp=True)
        if mtag == "compiled":
            return comp_coeff(X, phi, gam, alp, b)
        if mtag == "cublas_norm_after":
            return fused_mhc_coeff(X, phi, gam, alp, b, single_scan=False)
        return fused_mhc_coeff(X, phi, gam, alp, b, single_scan=True)

    t1_detail: dict[str, Any] = {}
    for tid, cid, csha, mtag, cnote in t1_candidates:
        snap0 = PrecisionStateSnapshot.capture()
        p_cks: list[dict[str, Any]] = []
        in_unmut = True
        for b_tag, b_vec, ora_cur in (("mod", b_g1_mod, ora_g1_mod), ("ext115", b_g1_ext, ora_g1_ext)):
            Xi = X_g1.clone().requires_grad_(True)
            phii = phi_g1.clone().requires_grad_(True)
            gami = gam_g1.clone().requires_grad_(True)
            alpi = alp_g1.clone().requires_grad_(True)
            bi = b_vec.clone().requires_grad_(True)
            hp, ho, hr = _run_t1(mtag, Xi, phii, gami, alpi, bi)
            ((hp * dhp_g1).sum() + (ho * dho_g1).sum() + (hr * dhr_g1).sum()).backward()
            in_unmut = in_unmut and bool(torch.equal(Xi.detach(), X_g1))
            for tname, ct, rt in (
                ("h_pre", hp, ora_cur["h_pre"]),
                ("h_post", ho, ora_cur["h_post"]),
                ("h_res", hr, ora_cur["h_res"]),
                ("dX", Xi.grad, ora_cur["dX"]),
                ("d_phi", phii.grad, ora_cur["d_phi"]),
                ("d_gamma", gami.grad, ora_cur["d_gamma"]),
                ("d_alpha", alpi.grad, ora_cur["d_alpha"]),
                ("d_b", bi.grad, ora_cur["d_b"]),
            ):
                p_cks.append(compare_tensor_to_fp64(ct, rt, rel_tol=1.5e-2, name=f"{b_tag}_{tname}"))

        hp_1, _, _ = _run_t1(mtag, X_g1, phi_g1, gam_g1, alp_g1, b_g1_mod)
        snap_hp1 = hp_1.detach().clone()
        _ = _run_t1(mtag, X_g1_b, phi_g1, gam_g1, alp_g1, b_g1_mod)
        non_alias = bool(torch.equal(hp_1.detach(), snap_hp1))
        flags_ok = PrecisionStateSnapshot.capture() == snap0

        shape_perf: dict[str, dict[str, Any]] = {}
        for skey, N_s, n_s in t1_shapes:
            K_s = n_s * C_dim
            M_s = n_s * n_s + 2 * n_s
            bytes_per_slot = N_s * K_s * 2
            n_slots = max(2, math.ceil((112 * 1024 * 1024) / bytes_per_slot))
            slots = []
            for sl in range(n_slots):
                gs = torch.Generator(device=device).manual_seed(50000 + 97 * tid + 13 * sl + n_s)
                Xs = torch.randn((N_s, n_s, C_dim), device=device, dtype=torch.bfloat16, generator=gs, requires_grad=True)
                phis = torch.randn((K_s, M_s), device=device, dtype=torch.bfloat16, generator=gs, requires_grad=True)
                gams = torch.ones((K_s,), device=device, dtype=torch.bfloat16, requires_grad=True)
                alps = torch.tensor([0.01, 0.01, 0.01], device=device, dtype=torch.float32, requires_grad=True)
                bs = torch.zeros((M_s,), device=device, dtype=torch.float32, requires_grad=True)
                dhps = torch.randn((N_s, n_s), device=device, dtype=torch.float32, generator=gs)
                dhos = torch.randn((N_s, n_s), device=device, dtype=torch.float32, generator=gs)
                dhrs = torch.randn((N_s, n_s, n_s), device=device, dtype=torch.float32, generator=gs)
                slots.append((Xs, phis, gams, alps, bs, dhps, dhos, dhrs))

            def _mk_t1_step(pk, mt=mtag):
                Xs, phis, gams, alps, bs, dhps, dhos, dhrs = pk

                def _step():
                    hp_o, ho_o, hr_o = _run_t1(mt, Xs, phis, gams, alps, bs)
                    torch.autograd.backward([hp_o, ho_o, hr_o], [dhps, dhos, dhrs])
                    Xs.grad = phis.grad = gams.grad = alps.grad = bs.grad = None

                return _step

            res = benchmark_rotating_cuda(
                [_mk_t1_step(pk) for pk in slots],
                warmup=2 if is_smoke else 4,
                groups=3 if is_smoke else 7,
                iters_per_group=3 if is_smoke else 6,
                l2_flush_buffer=l2_flush,
            )
            # Minimal HBM traffic for Fwd + Bwd: read X in fwd (2*N*K) + read X & write dX in bwd (4*N*K) = 6*N*K bytes
            min_bytes = 6.0 * float(N_s) * float(K_s) * 2.0
            res["algorithmic_minimal_gbps"] = round((min_bytes / (res["median_ms"] * 1e-3)) / 1e9, 2)
            shape_perf[skey] = res
            del slots
            torch.cuda.empty_cache()

        rec = ratchet_t1.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=csha,
            parity_checks=p_cks,
            richardson_checks=[],
            anti_cheat_checks={
                "input_unmutated": in_unmut,
                "output_non_aliasing": non_alias,
                "precision_flags_preserved": flags_ok,
            },
            per_shape_metrics=shape_perf,
            hypothesis_note=cnote,
        )
        t1_detail[cid] = {
            "decision": rec.decision,
            "geomean_ms": rec.geomean_ms,
            "speedup_vs_baseline": rec.speedup_vs_baseline,
            "peak_alloc_mib": rec.peak_alloc_mib,
            "shapes": shape_perf,
        }

    # ------------------------------------------------------------------------
    # Track 2: `mhc_stream_readout_pre_norm_and_post_res_merge` (`C = 3,584`)
    # ------------------------------------------------------------------------
    t2_shapes = (
        [("N4096_n2", 4096, 2), ("N4096_n4", 4096, 4), ("N4096_n8", 4096, 8)]
        if is_smoke
        else [
            ("N8192_n2", 8192, 2),
            ("N8192_n4", 8192, 4),
            ("N8192_n8", 8192, 8),
            ("N16384_n4", 16384, 4),
            ("N32768_n4", 32768, 4),
        ]
    )
    ratchet_t2 = RSIVectorRatchet(
        track="mhc_stream_readout_pre_norm_and_post_res_merge",
        shape_keys=[s[0] for s in t2_shapes],
        hbm_cap_mib=48000.0,
    )
    comp_sandwich = torch.compile(eager_mhc_stream_sandwich, dynamic=False)

    t2_candidates = [
        (
            0,
            "trial0_eager_table2_unfused_stream_sandwich",
            hash_callable_sources(eager_mhc_stream_sandwich),
            "eager",
            "Table-2 Eager PyTorch: (5n+1)C reads + (3n+1)C writes + [N,n,n,C] einsum bwd intermediates",
        ),
        (
            1,
            "trial1_torch_compile_inductor_stream_sandwich",
            hash_callable_sources(eager_mhc_stream_sandwich, "torch.compile"),
            "compiled",
            "TorchInductor compiled stream sandwich: fuses outer/add ops but retains multi-kernel reductions",
        ),
        (
            2,
            "trial2_deepseek_2kernel_fpre_separate_rmsnorm_fpostres",
            hash_callable_sources(DeepSeek2KernelStreamSandwichFn),
            "deepseek_2k",
            "DeepSeek Sec. 4.3.1 split: F_pre writes [N,C] h_in -> separate SublayerRMSNorm -> fused F_{post,res}",
        ),
        (
            3,
            "trial3_rsi_superfused_pre_rmsnorm_and_unified_sandwich_bwd",
            hash_callable_sources(
                RSISuperfusedStreamSandwichFn,
                _superfused_pre_rmsnorm_fwd_kernel,
                _fused_post_res_fwd_kernel,
                _rsi_unified_sandwich_bwd_split_kernel,
            ),
            "rsi_superfused",
            "Superfused Pre+RMSNorm (0 HBM bytes for h_in) + Unified 1-Read/1-Write Sandwich Bwd (Split-16 lock-free)",
        ),
    ]

    g_t2 = torch.Generator(device=device).manual_seed(42002)
    N_g2, n_g2 = 263, 4
    X_g2 = (torch.randn((N_g2, n_g2, C_dim), device=device, dtype=torch.bfloat16, generator=g_t2) * 0.7).contiguous()
    X_g2_b = (torch.randn((N_g2, n_g2, C_dim), device=device, dtype=torch.bfloat16, generator=g_t2) * 0.7).contiguous()
    y_g2 = (torch.randn((N_g2, C_dim), device=device, dtype=torch.bfloat16, generator=g_t2) * 0.6).contiguous()
    hp_g2 = torch.sigmoid(torch.randn((N_g2, n_g2), device=device, dtype=torch.float32, generator=g_t2)).contiguous()
    ho_g2 = (2.0 * torch.sigmoid(torch.randn((N_g2, n_g2), device=device, dtype=torch.float32, generator=g_t2))).contiguous()
    hr_g2 = log_sinkhorn_knopp_pytorch(torch.randn((N_g2, n_g2, n_g2), device=device, dtype=torch.float32, generator=g_t2)).contiguous()
    w_g2 = (torch.ones((C_dim,), device=device, dtype=torch.bfloat16) + 0.1 * torch.randn((C_dim,), device=device, dtype=torch.bfloat16, generator=g_t2)).contiguous()
    dhn_g2 = torch.randn_like(y_g2) * 0.4
    dxn_g2 = torch.randn_like(X_g2) * 0.4

    ora_g2 = fp64_mhc_stream_oracle(X_g2, y_g2, hp_g2, ho_g2, hr_g2, w_g2, dhn_g2, dxn_g2)

    def _run_t2(mtag: str, X, y, hp, ho, hr, w):
        if mtag == "eager":
            return eager_mhc_stream_sandwich(X, y, hp, ho, hr, w)
        if mtag == "compiled":
            return comp_sandwich(X, y, hp, ho, hr, w)
        if mtag == "deepseek_2k":
            return DeepSeek2KernelStreamSandwichFn.apply(X, y, hp, ho, hr, w, 1e-6)
        return RSISuperfusedStreamSandwichFn.apply(X, y, hp, ho, hr, w, 1e-6)

    t2_detail: dict[str, Any] = {}
    for tid, cid, csha, mtag, cnote in t2_candidates:
        snap0 = PrecisionStateSnapshot.capture()
        Xi = X_g2.clone().requires_grad_(True)
        yi = y_g2.clone().requires_grad_(True)
        hpi = hp_g2.clone().requires_grad_(True)
        hoi = ho_g2.clone().requires_grad_(True)
        hri = hr_g2.clone().requires_grad_(True)
        wi = w_g2.clone().requires_grad_(True)

        hn_o, xn_o = _run_t2(mtag, Xi, yi, hpi, hoi, hri, wi)
        torch.autograd.backward([hn_o, xn_o], [dhn_g2, dxn_g2])
        in_unmut = bool(torch.equal(Xi.detach(), X_g2) and torch.equal(yi.detach(), y_g2))
        p_cks = [
            compare_tensor_to_fp64(hn_o, ora_g2["h_norm"], rel_tol=1.2e-2, name="h_norm"),
            compare_tensor_to_fp64(xn_o, ora_g2["X_next"], rel_tol=1.2e-2, name="X_next"),
            compare_tensor_to_fp64(Xi.grad, ora_g2["dX"], rel_tol=1.2e-2, name="dX"),
            compare_tensor_to_fp64(yi.grad, ora_g2["dy"], rel_tol=1.2e-2, name="dy"),
            compare_tensor_to_fp64(hpi.grad, ora_g2["dh_pre"], rel_tol=1.2e-2, name="dh_pre"),
            compare_tensor_to_fp64(hoi.grad, ora_g2["dh_post"], rel_tol=1.2e-2, name="dh_post"),
            compare_tensor_to_fp64(hri.grad, ora_g2["dh_res"], rel_tol=1.2e-2, name="dh_res"),
            compare_tensor_to_fp64(wi.grad, ora_g2["dw_sub"], rel_tol=1.2e-2, name="dw_sub"),
        ]

        hn_1, _ = _run_t2(mtag, X_g2, y_g2, hp_g2, ho_g2, hr_g2, w_g2)
        snap_hn1 = hn_1.detach().clone()
        _ = _run_t2(mtag, X_g2_b, y_g2, hp_g2, ho_g2, hr_g2, w_g2)
        non_alias = bool(torch.equal(hn_1.detach(), snap_hn1))
        flags_ok = PrecisionStateSnapshot.capture() == snap0

        shape_perf = {}
        for skey, N_s, n_s in t2_shapes:
            bytes_slot = N_s * (2 * n_s + 2) * C_dim * 2
            n_slots = max(2, math.ceil((112 * 1024 * 1024) / bytes_slot))
            slots = []
            for sl in range(n_slots):
                gs = torch.Generator(device=device).manual_seed(60000 + 83 * tid + 17 * sl + n_s)
                Xs = torch.randn((N_s, n_s, C_dim), device=device, dtype=torch.bfloat16, generator=gs, requires_grad=True)
                ys = torch.randn((N_s, C_dim), device=device, dtype=torch.bfloat16, generator=gs, requires_grad=True)
                hps = torch.sigmoid(torch.randn((N_s, n_s), device=device, dtype=torch.float32, generator=gs)).detach().requires_grad_(True)
                hos = (2.0 * torch.sigmoid(torch.randn((N_s, n_s), device=device, dtype=torch.float32, generator=gs))).detach().requires_grad_(True)
                hrs = log_sinkhorn_knopp_pytorch(torch.randn((N_s, n_s, n_s), device=device, dtype=torch.float32, generator=gs)).detach().requires_grad_(True)
                ws = torch.ones((C_dim,), device=device, dtype=torch.bfloat16, requires_grad=True)
                dhns = torch.randn_like(ys)
                dxns = torch.randn_like(Xs)
                slots.append((Xs, ys, hps, hos, hrs, ws, dhns, dxns))

            def _mk_t2_step(pk, mt=mtag):
                Xs, ys, hps, hos, hrs, ws, dhns, dxns = pk

                def _step():
                    hno, xno = _run_t2(mt, Xs, ys, hps, hos, hrs, ws)
                    torch.autograd.backward([hno, xno], [dhns, dxns])
                    Xs.grad = ys.grad = hps.grad = hos.grad = hrs.grad = ws.grad = None

                return _step

            res = benchmark_rotating_cuda(
                [_mk_t2_step(pk) for pk in slots],
                warmup=2 if is_smoke else 4,
                groups=3 if is_smoke else 7,
                iters_per_group=3 if is_smoke else 6,
                l2_flush_buffer=l2_flush,
            )
            # Minimal sandwich HBM bytes (Fwd: read X twice or cached + y, write h_norm + X_next = (2n+2)C;
            # Unified Bwd: read X + y + dh_norm + dX_next, write dX + dy = (3n+3)C) -> total (5n+5)*N*C*2B
            min_b = float(5 * n_s + 5) * float(N_s) * float(C_dim) * 2.0
            phys_b = min_b + (4.0 * float(N_s) * float(C_dim) * 2.0 + 2.0 * float(N_s) * float(n_s) * float(C_dim) * 2.0 if mtag == "deepseek_2k" else 0.0)
            res["algorithmic_minimal_gbps"] = round((min_b / (res["median_ms"] * 1e-3)) / 1e9, 2)
            res["physical_realized_gbps"] = round((phys_b / (res["median_ms"] * 1e-3)) / 1e9, 2)
            shape_perf[skey] = res
            del slots
            torch.cuda.empty_cache()

        rec = ratchet_t2.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=csha,
            parity_checks=p_cks,
            richardson_checks=[],
            anti_cheat_checks={
                "input_unmutated": in_unmut,
                "output_non_aliasing": non_alias,
                "precision_flags_preserved": flags_ok,
            },
            per_shape_metrics=shape_perf,
            hypothesis_note=cnote,
        )
        t2_detail[cid] = {
            "decision": rec.decision,
            "geomean_ms": rec.geomean_ms,
            "speedup_vs_baseline": rec.speedup_vs_baseline,
            "peak_alloc_mib": rec.peak_alloc_mib,
            "shapes": shape_perf,
        }

    return (
        {"track1_mhc_coeff_sinkhorn": t1_detail, "track2_mhc_stream_sandwich": t2_detail},
        [ratchet_t1, ratchet_t2],
    )


# ============================================================================
# 3. Unified GQA Causal Transformer Supporting:
#    - `1-Stream` Standard Pre-Norm Residual (`topology="1stream"`)
#    - Unconstrained `HC` (`topology="hc_unconstrained"`, Eq. 5)
#    - DeepSeek `mHC` (`topology="mhc"`, Eq. 7-20: Eager, TorchCompile, RSI-Fused, & `L_r` `mHC-SAC`)
# ============================================================================


def build_rope_cos_sin(seq_len: int, head_dim: int, device: torch.device, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / float(head_dim)))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()


def apply_rope_bshd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, S, H, D], cos/sin: [S, D/2]
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2].float()
    x2 = x[..., d2:].float()
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def _hc_unconstrained_coeff(
    X: torch.Tensor, phi: torch.Tensor, gamma: torch.Tensor, alpha: torch.Tensor, b: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ByteDance Seed `HC` Eq. 5 unconstrained formulation (`H = alpha * tanh(x_norm @ phi) + b`, NO Birkhoff projection!)."""
    N, n_s, C = X.shape
    xf = X.reshape(N, n_s * C).float()
    rstd = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    u = ((xf * rstd * gamma.float()).to(X.dtype) @ phi).float()
    hp = alpha[0].float() * torch.tanh(u[:, :n_s]) + b[:n_s].float()
    ho = alpha[1].float() * torch.tanh(u[:, n_s : 2 * n_s]) + b[n_s : 2 * n_s].float()
    hr = (alpha[2].float() * torch.tanh(u[:, 2 * n_s :]) + b[2 * n_s :].float()).view(N, n_s, n_s)
    return hp, ho, hr


_compiled_coeff_step = torch.compile(eager_mhc_coeff_eq7_eq9, dynamic=False)


class MHCSublayerAdapter(nn.Module):
    """Per-sublayer (`Attn` or `MLP`) dynamic/static `mHC` / `HC` parameters (`BF16` matrices + `FP32` gating scalars)."""

    def __init__(
        self,
        d_model: int,
        n_streams: int,
        *,
        topology: str,
        sublayer_idx: int = 0,
        symmetric_saddle_init: bool = False,
        hc_alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_streams = n_streams
        self.topology = topology
        K = n_streams * d_model
        M = n_streams * n_streams + 2 * n_streams

        self.phi = nn.Parameter(torch.empty((K, M), dtype=torch.bfloat16))
        self.gamma = nn.Parameter(torch.ones((K,), dtype=torch.bfloat16))
        self.w_sub = nn.Parameter(torch.ones((d_model,), dtype=torch.bfloat16))
        # Kept in exact FP32 per DeepSeek Eq. 12-13 (excluded from BF16 FSDPParamGroup via ignored_params)
        self.alpha = nn.Parameter(torch.full((3,), hc_alpha_init, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((M,), dtype=torch.float32))

        with torch.no_grad():
            if symmetric_saddle_init:
                # Naive symmetric initialization (`GPT 6 Astra, Max` Q4 saddle-point trap!):
                # phi = 0, b_res = 0 (uniform 1/n), b_post = 0 -> identical streams stay identical forever (`dZ_res == 0`)!
                self.phi.zero_()
                self.b[:n_streams].fill_(-math.log(max(1, n_streams - 1)))
                self.b[n_streams : 2 * n_streams].zero_()
                self.b[2 * n_streams :].zero_()
            elif topology == "hc_unconstrained":
                # Unconstrained HC Eq. 5: b_pre = 1/n, b_post = e_{l mod n}, b_res = I_n (no manifold projection!)
                nn.init.normal_(self.phi, mean=0.0, std=1.0 / math.sqrt(K))
                self.b[:n_streams].fill_(1.0 / float(n_streams))
                self.b[n_streams : 2 * n_streams].zero_()
                self.b[n_streams + (sublayer_idx % n_streams)] = 1.0
                eye = torch.eye(n_streams, dtype=torch.float32).reshape(-1)
                self.b[2 * n_streams :].copy_(eye)
            else:
                # Broken-symmetry Asymmetric Birkhoff mHC init (Zhu et al. HC Sec. 3.1 + DeepSeek mHC Eq. 8 + GPT 6 Astra Q4):
                # sublayer `l` initially writes preferentially into stream `l mod n` while reading across streams with
                # unit-sum mean `1/n` and near-identity Birkhoff `b_res = 2.5 * I_n`, breaking stream symmetry at step 0!
                nn.init.normal_(self.phi, mean=0.0, std=1.0 / math.sqrt(K))
                self.b[:n_streams].fill_(-math.log(max(1, n_streams - 1)))
                self.b[:n_streams][sublayer_idx % n_streams] += 0.35
                self.b[n_streams : 2 * n_streams].fill_(-1.6)
                self.b[n_streams + (sublayer_idx % n_streams)] = 1.2
                self.b[2 * n_streams :].copy_(2.5 * torch.eye(n_streams, dtype=torch.float32).reshape(-1))


class MHCTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        h_q: int,
        h_kv: int,
        ffn_dim: int,
        n_streams: int = 4,
        *,
        topology: str = "mhc",
        block_idx: int = 0,
        symmetric_saddle_init: bool = False,
        hc_alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = d_model // h_q
        self.ffn_dim = ffn_dim
        self.n_streams = n_streams
        self.topology = topology

        qkv_out = (h_q + 2 * h_kv) * self.head_dim
        self.w_qkv = nn.Linear(d_model, qkv_out, bias=False, dtype=torch.bfloat16)
        self.w_o = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_dim, bias=False, dtype=torch.bfloat16)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False, dtype=torch.bfloat16)

        with torch.no_grad():
            for lin in (self.w_qkv, self.w_o, self.w_gate_up, self.w_down):
                nn.init.normal_(lin.weight, mean=0.0, std=0.02)

        if topology == "1stream":
            self.norm1 = nn.Parameter(torch.ones((d_model,), dtype=torch.bfloat16))
            self.norm2 = nn.Parameter(torch.ones((d_model,), dtype=torch.bfloat16))
        else:
            self.mhc_attn = MHCSublayerAdapter(
                d_model, n_streams, topology=topology, sublayer_idx=2 * block_idx,
                symmetric_saddle_init=symmetric_saddle_init, hc_alpha_init=hc_alpha_init,
            )
            self.mhc_mlp = MHCSublayerAdapter(
                d_model, n_streams, topology=topology, sublayer_idx=2 * block_idx + 1,
                symmetric_saddle_init=symmetric_saddle_init, hc_alpha_init=hc_alpha_init,
            )

    def _attn_core(self, h_norm: torch.Tensor, B: int, S: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        qkv = self.w_qkv(h_norm)
        q_dim = self.h_q * self.head_dim
        kv_dim = self.h_kv * self.head_dim
        q = qkv[:, :q_dim].view(B, S, self.h_q, self.head_dim)
        k = qkv[:, q_dim : q_dim + kv_dim].view(B, S, self.h_kv, self.head_dim)
        v = qkv[:, q_dim + kv_dim :].view(B, S, self.h_kv, self.head_dim)
        q = apply_rope_bshd(q, cos, sin)
        k = apply_rope_bshd(k, cos, sin)
        out = flash_attn_func(q, k, v, causal=True).reshape(B * S, self.d_model)
        return self.w_o(out)

    def _mlp_core(self, h_norm: torch.Tensor, use_fused_swiglu: bool) -> torch.Tensor:
        gu = self.w_gate_up(h_norm)
        if use_fused_swiglu:
            act = FusedSwiGLUFn.apply(gu)
        else:
            g, u = gu.chunk(2, dim=-1)
            act = (F.silu(g.float()) * u.float()).to(gu.dtype)
        return self.w_down(act)

    def _run_stream_sublayer(
        self,
        X: torch.Tensor,
        adapter: MHCSublayerAdapter,
        core_fn,
        *,
        kernel_mode: str,
        bank: MHCStreamSegmentBank | None,
        sublayer_idx: int,
        hr_collector: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.topology == "hc_unconstrained":
            hp, ho, hr = _hc_unconstrained_coeff(X, adapter.phi, adapter.gamma, adapter.alpha, adapter.b)
            h_in = torch.einsum("bi,bic->bc", hp, X.float()).to(X.dtype)
            rstd = torch.rsqrt(torch.mean(h_in.float() ** 2, dim=-1, keepdim=True) + 1e-6)
            h_norm = (h_in.float() * rstd * adapter.w_sub.float()).to(X.dtype)
            y = core_fn(h_norm)
            X_next = torch.einsum("bij,bjc->bic", hr, X.float()).to(X.dtype) + (ho.unsqueeze(-1) * y.float().unsqueeze(1)).to(X.dtype)
            if hr_collector is not None:
                hr_collector.append(hr.detach())
            return X_next

        if kernel_mode == "eager":
            hp, ho, hr = eager_mhc_coeff_eq7_eq9(X, adapter.phi, adapter.gamma, adapter.alpha, adapter.b)
            h_in = torch.einsum("bi,bic->bc", hp, X.float()).to(X.dtype)
            rstd = torch.rsqrt(torch.mean(h_in.float() ** 2, dim=-1, keepdim=True) + 1e-6)
            h_norm = (h_in.float() * rstd * adapter.w_sub.float()).to(X.dtype)
            y = core_fn(h_norm)
            X_next = torch.einsum("bij,bjc->bic", hr, X.float()).to(X.dtype) + (ho.unsqueeze(-1) * y.float().unsqueeze(1)).to(X.dtype)
        elif kernel_mode == "compiled":
            hp, ho, hr = _compiled_coeff_step(X, adapter.phi, adapter.gamma, adapter.alpha, adapter.b)
            h_in = torch.einsum("bi,bic->bc", hp, X.float()).to(X.dtype)
            rstd = torch.rsqrt(torch.mean(h_in.float() ** 2, dim=-1, keepdim=True) + 1e-6)
            h_norm = (h_in.float() * rstd * adapter.w_sub.float()).to(X.dtype)
            y = core_fn(h_norm)
            X_next = FusedPostResMergeFn.apply(X, y, ho, hr)
        elif kernel_mode == "rsi_sac" and bank is not None:
            seg_id = sublayer_idx // bank.L_r
            offset = sublayer_idx % bank.L_r
            hp, ho, hr = SACFusedMHCCoeffFn.apply(
                X, adapter.phi, adapter.gamma, adapter.alpha, adapter.b, bank, seg_id, offset, 1e-6
            )
            h_norm = SACSuperfusedPreRMSNormFn.apply(X, hp, adapter.w_sub, bank, seg_id, offset, 1e-6)
            y = core_fn(h_norm)
            X_next = SACFusedPostResMergeFn.apply(X, y, ho, hr, bank, sublayer_idx)
        else:
            hp, ho, hr = fused_mhc_coeff(X, adapter.phi, adapter.gamma, adapter.alpha, adapter.b, single_scan=True)
            h_norm = SuperfusedPreRMSNormFn.apply(X, hp, adapter.w_sub, 1e-6)
            y = core_fn(h_norm)
            X_next = FusedPostResMergeFn.apply(X, y, ho, hr)

        if hr_collector is not None:
            hr_collector.append(hr.detach())
        return X_next

    def forward(
        self,
        state: torch.Tensor,
        B: int,
        S: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        block_idx: int,
        kernel_mode: str = "rsi_fused",
        bank: MHCStreamSegmentBank | None = None,
        hr_collector: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        use_f_swiglu = kernel_mode in ("rsi_fused", "rsi_sac", "compiled")
        if self.topology == "1stream":
            xf = state.float()
            h1 = (xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-6) * self.norm1.float()).to(state.dtype)
            state = state + self._attn_core(h1, B, S, cos, sin)
            xf2 = state.float()
            h2 = (xf2 * torch.rsqrt(torch.mean(xf2 * xf2, dim=-1, keepdim=True) + 1e-6) * self.norm2.float()).to(state.dtype)
            state = state + self._mlp_core(h2, use_fused_swiglu=True)
            return state

        X = self._run_stream_sublayer(
            state, self.mhc_attn, lambda hn: self._attn_core(hn, B, S, cos, sin),
            kernel_mode=kernel_mode, bank=bank, sublayer_idx=2 * block_idx, hr_collector=hr_collector,
        )
        X = self._run_stream_sublayer(
            X, self.mhc_mlp, lambda hn: self._mlp_core(hn, use_f_swiglu),
            kernel_mode=kernel_mode, bank=bank, sublayer_idx=2 * block_idx + 1, hr_collector=hr_collector,
        )
        return X


class MHCTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        h_q: int,
        h_kv: int,
        ffn_dim: int,
        n_streams: int = 4,
        *,
        topology: str = "mhc",
        symmetric_saddle_init: bool = False,
        hc_alpha_init: float = 0.01,
        enable_dynamic_readout: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_streams = n_streams
        self.topology = topology
        self.enable_dynamic_readout = enable_dynamic_readout and (topology != "1stream")

        self.tok_emb = nn.Embedding(vocab_size, d_model, dtype=torch.bfloat16)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        if topology != "1stream":
            # Zero-sum centered per-stream embedding tag (`mean_stream(stream_emb) == 0`, breaks equal-stream saddle!)
            self.stream_emb = nn.Parameter(torch.zeros((n_streams, d_model), dtype=torch.float32))
            if not symmetric_saddle_init:
                with torch.no_grad():
                    raw = torch.randn((n_streams, d_model), dtype=torch.float32) * 0.02
                    raw = raw - raw.mean(dim=0, keepdim=True)
                    self.stream_emb.copy_(raw)
        else:
            self.register_parameter("stream_emb", None)

        self.blocks = nn.ModuleList(
            [
                MHCTransformerBlock(
                    d_model, h_q, h_kv, ffn_dim, n_streams=n_streams,
                    topology=topology, block_idx=blk_i,
                    symmetric_saddle_init=symmetric_saddle_init, hc_alpha_init=hc_alpha_init,
                )
                for blk_i in range(n_layers)
            ]
        )
        if self.enable_dynamic_readout:
            self.final_mhc = MHCSublayerAdapter(
                d_model, n_streams, topology=topology, sublayer_idx=99,
                symmetric_saddle_init=symmetric_saddle_init, hc_alpha_init=hc_alpha_init,
            )
            self.stream_in_scale = nn.Parameter(torch.ones((n_streams, d_model), dtype=torch.float32))
            if (not symmetric_saddle_init) and topology == "mhc" and n_streams == 4:
                with torch.no_grad():
                    had = torch.tensor(
                        [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]],
                        dtype=torch.float32,
                    )
                    cols = torch.arange(d_model) % 4
                    self.stream_in_scale.copy_(1.0 + 0.50 * had[:, cols])
                    for bi_blk, blk in enumerate(self.blocks):
                        for si, ad in enumerate((blk.mhc_attn, blk.mhc_mlp)):
                            idx = (2 * bi_blk + si) % 4
                            ad.b[:4].fill_(-1.4)
                            ad.b[idx] = 0.5
                            ad.b[4:8].fill_(-2.2)
                            ad.b[4 + idx] = -0.1
                            ad.b[8:].copy_(2.2 * torch.eye(4, dtype=torch.float32).reshape(-1))
        else:
            self.final_mhc = None
            self.register_parameter("stream_in_scale", None)

        self.final_norm = nn.Parameter(torch.ones((d_model,), dtype=torch.bfloat16))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        kernel_mode: str = "rsi_fused",
        sac_L_r: int = 1,
        use_chunked_ce: bool = True,
        global_tokens: int = 1,
        world_size_scale: float = 1.0,
        collect_diagnostics: bool = False,
        backbone_only_loss: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, S = input_ids.shape
        N = B * S
        emb = self.tok_emb(input_ids.reshape(-1))  # [N, C]

        bank = MHCStreamSegmentBank(sac_L_r) if (self.topology == "mhc" and kernel_mode == "rsi_sac") else None
        hr_list: list[torch.Tensor] | None = [] if collect_diagnostics else None

        if self.topology == "1stream":
            state = emb
        elif self.enable_dynamic_readout and self.stream_in_scale is not None:
            assert self.stream_emb is not None
            state = (
                emb.float().unsqueeze(1) * self.stream_in_scale.unsqueeze(0) + self.stream_emb.unsqueeze(0)
            ).to(emb.dtype).contiguous()
        else:
            assert self.stream_emb is not None
            state = (emb.float().unsqueeze(1) + self.stream_emb.unsqueeze(0)).to(emb.dtype).contiguous()

        for idx, blk in enumerate(self.blocks):
            state = blk(
                state, B, S, cos, sin,
                block_idx=idx, kernel_mode=kernel_mode, bank=bank, hr_collector=hr_list,
            )

        diag: dict[str, float] = {}
        h_final: torch.Tensor | None = None
        if self.topology == "1stream":
            h_out = state
            if collect_diagnostics:
                diag = {
                    "amax_fwd_gain": 1.0,
                    "amax_bwd_gain": 1.0,
                    "stream_pairwise_cos": 1.0,
                }
        else:
            if collect_diagnostics:
                with torch.no_grad():
                    # Pairwise cosine similarity across the n streams (1.0000 = collapsed equal-stream saddle!)
                    sf = F.normalize(state.detach().float(), dim=-1)  # [N, n, C]
                    gram = torch.einsum("bic,bjc->bij", sf, sf).mean(dim=0)  # [n, n]
                    off_mask = ~torch.eye(self.n_streams, dtype=torch.bool, device=state.device)
                    pair_cos = float(gram[off_mask].mean().item())
                    # DeepSeek Sec. 3.1 / 5.4 Composite Amax Gain Magnitude: prod_{l=2L-1..0} H_l^res
                    assert hr_list is not None and len(hr_list) > 0
                    comp = hr_list[0][:256].float()
                    for m_next in hr_list[1:]:
                        comp = torch.bmm(m_next[:256].float(), comp)
                    fwd_gain = float(comp.sum(dim=-1).abs().max(dim=-1).values.mean().item())
                    bwd_gain = float(comp.sum(dim=-2).abs().max(dim=-1).values.mean().item())
                    diag = {
                        "amax_fwd_gain": round(fwd_gain, 5),
                        "amax_bwd_gain": round(bwd_gain, 5),
                        "stream_pairwise_cos": round(pair_cos, 6),
                    }
            if self.enable_dynamic_readout and self.final_mhc is not None:
                if self.topology == "hc_unconstrained":
                    hp_f, _, _ = _hc_unconstrained_coeff(
                        state, self.final_mhc.phi, self.final_mhc.gamma, self.final_mhc.alpha, self.final_mhc.b
                    )
                    hin = torch.einsum("bi,bic->bc", hp_f, state.float())
                    h_final = (
                        hin * torch.rsqrt(torch.mean(hin * hin, dim=-1, keepdim=True) + 1e-6) * self.final_norm.float()
                    ).to(emb.dtype)
                elif kernel_mode == "eager":
                    hp_f, _, _ = eager_mhc_coeff_eq7_eq9(
                        state, self.final_mhc.phi, self.final_mhc.gamma, self.final_mhc.alpha, self.final_mhc.b
                    )
                    hin = torch.einsum("bi,bic->bc", hp_f, state.float())
                    h_final = (
                        hin * torch.rsqrt(torch.mean(hin * hin, dim=-1, keepdim=True) + 1e-6) * self.final_norm.float()
                    ).to(emb.dtype)
                elif kernel_mode == "compiled":
                    hp_f, _, _ = _compiled_coeff_step(
                        state, self.final_mhc.phi, self.final_mhc.gamma, self.final_mhc.alpha, self.final_mhc.b
                    )
                    h_final = SuperfusedPreRMSNormFn.apply(state, hp_f, self.final_norm, 1e-6)
                else:
                    hp_f, _, _ = fused_mhc_coeff(
                        state, self.final_mhc.phi, self.final_mhc.gamma, self.final_mhc.alpha, self.final_mhc.b, single_scan=True
                    )
                    h_final = SuperfusedPreRMSNormFn.apply(state, hp_f, self.final_norm, 1e-6)
            else:
                h_out = state.float().mean(dim=1).to(emb.dtype)

        if h_final is None:
            hf = h_out.float()
            h_final = (hf * torch.rsqrt(torch.mean(hf * hf, dim=-1, keepdim=True) + 1e-6) * self.final_norm.float()).to(emb.dtype)

        if backbone_only_loss:
            return 0.5 * (h_final.float() ** 2).mean(), diag

        tgt_flat = targets.reshape(-1)

        if use_chunked_ce:
            nll_sum = ChunkedLinearCrossEntropyFn.apply(h_final, self.lm_head.weight, tgt_flat, 2048)
        else:
            logits = self.lm_head(h_final).float()
            nll_sum = F.cross_entropy(logits, tgt_flat, reduction="sum")

        loss = nll_sum * (float(world_size_scale) / float(global_tokens))
        return loss, diag


def _shard_mhc_lm_fsdp2(model: MHCTransformerLM, mp_policy: MixedPrecisionPolicy, enable_2block_prefetch: bool) -> list[nn.Parameter]:
    # Deterministic parameter-name ordering across all 8 ranks (NEVER sort by Python object `id(p)`!)
    ordered_fp32 = [p for _, p in sorted(model.named_parameters()) if p.dtype == torch.float32]
    fp32_ignored = set(ordered_fp32)
    fully_shard(model.tok_emb, mp_policy=mp_policy)
    for blk in model.blocks:
        blk_ignored = {p for p in blk.parameters() if p.dtype == torch.float32}
        fully_shard(blk, mp_policy=mp_policy, ignored_params=blk_ignored)
    fully_shard(model, mp_policy=mp_policy, ignored_params=fp32_ignored)

    if enable_2block_prefetch and len(model.blocks) >= 3:
        for i, blk in enumerate(model.blocks):
            f_next = list(model.blocks[i + 1 : min(len(model.blocks), i + 3)])
            if f_next:
                blk.set_modules_to_forward_prefetch(f_next)
            b_prev = list(reversed(model.blocks[max(0, i - 2) : i]))
            if b_prev:
                blk.set_modules_to_backward_prefetch(b_prev)
    return ordered_fp32


def _allreduce_ignored_fp32_grads(ignored_params: list[nn.Parameter], world_size: int) -> None:
    grads = [p.grad for p in ignored_params if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    flat.div_(float(world_size))
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    off = 0
    for g in grads:
        numel = g.numel()
        g.copy_(flat[off : off + numel].view_as(g))
        off += numel


# ============================================================================
# 4. Suite B: DeepSeek Eq. 20 Selective `L_r^*` Checkpointing Curve (1-GPU) +
#    8-GPU NVSwitch `1.14B` `mHC-Transformer` `FSDP2` Parity & Scaling
# ============================================================================


def run_suite_b_sac_and_8gpu_fsdp2(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    assert world_size == 8
    dist.barrier()

    # ------------------------------------------------------------------------
    # B.1 1-GPU Empirical Verification of DeepSeek Eq. 20 Optimal `L_r^*`
    #     (`L_blocks = 12 => L_sub = 24` mHC sublayers, `n = 4 => sqrt(4*24/6) = 4.0`)
    # ------------------------------------------------------------------------
    sac_curve_summary: dict[str, Any] = {}
    if rank == 0:
        B_sac = 4 if is_smoke else 16
        S_sac = 512  # N = 8,192 tokens (`X_l [8192, 4, 2048]` = 128 MiB per sublayer!)
        C_sac = 2048
        L_blk_sac = 12
        L_sub_sac = 2 * L_blk_sac  # 24 sublayers!
        torch.manual_seed(20261291)
        m_sac = MHCTransformerLM(
            vocab_size=32768, d_model=C_sac, n_layers=L_blk_sac, h_q=16, h_kv=4, ffn_dim=5632, n_streams=4, topology="mhc"
        ).to(device)
        cos_s, sin_s = build_rope_cos_sin(S_sac, 128, device)
        g_sac = torch.Generator(device=device).manual_seed(9100)
        ids_sac = torch.randint(0, 32768, (B_sac, S_sac), device=device, generator=g_sac)
        tgt_sac = torch.randint(0, 32768, (B_sac, S_sac), device=device, generator=g_sac)

        # Reference gradients at L_r = 1 on isolated 24-sublayer mHC backbone (no V=32768 lm_head CE spike masking Eq. 20!)
        l_ref, _ = m_sac(
            ids_sac, tgt_sac, cos_s, sin_s, kernel_mode="rsi_fused", sac_L_r=1,
            use_chunked_ce=True, global_tokens=B_sac * S_sac, world_size_scale=1.0, backbone_only_loss=True,
        )
        l_ref.backward()
        ref_loss_val = float(l_ref.item())
        ref_g_qkv0 = m_sac.blocks[0].w_qkv.weight.grad.detach().clone()
        ref_g_phi0 = m_sac.blocks[0].mhc_attn.phi.grad.detach().clone()
        ref_g_b11 = m_sac.blocks[-1].mhc_mlp.b.grad.detach().clone()
        m_sac.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        lr_candidates = [1, 2, 4, 8, 24] if is_smoke else [1, 2, 3, 4, 6, 8, 12, 24]
        lr_points: dict[str, Any] = {}
        for lr_val in lr_candidates:
            kmode = "rsi_fused" if lr_val == 1 else "rsi_sac"
            # Verify gradient parity against L_r = 1
            lv, _ = m_sac(
                ids_sac, tgt_sac, cos_s, sin_s, kernel_mode=kmode, sac_L_r=lr_val,
                use_chunked_ce=True, global_tokens=B_sac * S_sac, world_size_scale=1.0, backbone_only_loss=True,
            )
            lv.backward()
            max_g_rel = max(
                rel_l2(m_sac.blocks[0].w_qkv.weight.grad, ref_g_qkv0),
                rel_l2(m_sac.blocks[0].mhc_attn.phi.grad, ref_g_phi0),
                rel_l2(m_sac.blocks[-1].mhc_mlp.b.grad, ref_g_b11),
            )
            loss_rd = abs(float(lv.item()) - ref_loss_val) / max(1e-12, abs(ref_loss_val))
            m_sac.zero_grad(set_to_none=True)

            for _ in range(2):
                lv2, _ = m_sac(
                    ids_sac, tgt_sac, cos_s, sin_s, kernel_mode=kmode, sac_L_r=lr_val,
                    use_chunked_ce=True, global_tokens=B_sac * S_sac, world_size_scale=1.0, backbone_only_loss=True,
                )
                lv2.backward()
                m_sac.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            base_model_mib = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
            torch.cuda.reset_peak_memory_stats(device)
            lv3, _ = m_sac(
                ids_sac, tgt_sac, cos_s, sin_s, kernel_mode=kmode, sac_L_r=lr_val,
                use_chunked_ce=True, global_tokens=B_sac * S_sac, world_size_scale=1.0, backbone_only_loss=True,
            )
            torch.cuda.synchronize(device)
            saved_fwd_mib = (torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)) - base_model_mib
            torch.cuda.reset_peak_memory_stats(device)
            lv3.backward()
            torch.cuda.synchronize(device)
            bwd_peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            bwd_act_peak_mib = bwd_peak_mib - base_model_mib
            m_sac.zero_grad(set_to_none=True)

            iters = 3 if is_smoke else 6
            s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s_ev.record()
            for _ in range(iters):
                lvt, _ = m_sac(
                    ids_sac, tgt_sac, cos_s, sin_s, kernel_mode=kmode, sac_L_r=lr_val,
                    use_chunked_ce=True, global_tokens=B_sac * S_sac, world_size_scale=1.0, backbone_only_loss=True,
                )
                lvt.backward()
                m_sac.zero_grad(set_to_none=True)
            e_ev.record()
            e_ev.synchronize()
            step_ms = s_ev.elapsed_time(e_ev) / float(iters)

            # DeepSeek Eq. 20 theoretical stream-checkpoint elements per token: nC * ceil(L_sub / L_r) + (n + 2)C * L_r
            eq20_c_units = 4 * math.ceil(L_sub_sac / lr_val) + 6 * lr_val
            lr_points[f"Lr_{lr_val}"] = {
                "L_r": lr_val,
                "step_ms": round(float(step_ms), 3),
                "saved_fwd_act_mib": round(float(saved_fwd_mib), 2),
                "bwd_peak_act_mib": round(float(bwd_act_peak_mib), 2),
                "bwd_total_peak_alloc_mib": round(float(bwd_peak_mib), 2),
                "eq20_theoretical_c_units": eq20_c_units,
                "loss_rel_diff_vs_Lr1": round(loss_rd, 9),
                "max_grad_rel_l2_vs_Lr1": round(max_g_rel, 7),
            }

        best_lr_entry = min(lr_points.values(), key=lambda d: d["bwd_peak_act_mib"])
        sac_curve_summary = {
            "n_streams": 4,
            "n_blocks": L_blk_sac,
            "n_sublayers": L_sub_sac,
            "tokens": B_sac * S_sac,
            "eq20_closed_form_Lr_star": round(math.sqrt(4.0 * L_sub_sac / 6.0), 4),
            "empirical_min_hbm_L_r": best_lr_entry["L_r"],
            "hbm_saved_vs_Lr1_mib": round(lr_points["Lr_1"]["bwd_peak_act_mib"] - best_lr_entry["bwd_peak_act_mib"], 2),
            "points": lr_points,
        }
        del m_sac, ref_g_qkv0, ref_g_phi0, ref_g_b11
        gc.collect()
        torch.cuda.empty_cache()

    dist.barrier()

    # ------------------------------------------------------------------------
    # B.2 Gate G9: 1-GPU Oracle vs 8-GPU NVSwitch `mHC-FSDP2` (`L_r^*=4` + 2-Block Prefetch)
    # ------------------------------------------------------------------------
    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
    V_big = 128256
    D_big = 2048
    L_big = 12  # 12 blocks = 24 mHC sublayers -> 1.071B parameters in B.3!
    I_big = 5632
    L_par = 4   # 4 blocks = 8 mHC sublayers (exercises 2 full L_r=4 SAC segments + 2-block prefetch!)
    B_par = 2
    S_par = 256
    glob_par_tok = world_size * B_par * S_par
    cos_p, sin_p = build_rope_cos_sin(S_par, 128, device)

    torch.manual_seed(20261292)
    cpu_mhc_ref = MHCTransformerLM(
        vocab_size=V_big, d_model=D_big, n_layers=L_par, h_q=16, h_kv=4, ffn_dim=I_big, n_streams=4, topology="mhc"
    )
    param_count_mhc = sum(p.numel() for p in cpu_mhc_ref.parameters())

    per_rank_ids = []
    per_rank_tgt = []
    for r in range(world_size):
        gr = torch.Generator(device="cpu").manual_seed(7700 + r)
        per_rank_ids.append(torch.randint(0, V_big, (B_par, S_par), generator=gr))
        per_rank_tgt.append(torch.randint(0, V_big, (B_par, S_par), generator=gr))

    ref_1g: dict[str, Any] = {}
    if rank == 0:
        m1 = copy.deepcopy(cpu_mhc_ref).to(device)
        all_ids = torch.cat(per_rank_ids, dim=0).to(device)
        all_tgt = torch.cat(per_rank_tgt, dim=0).to(device)
        l1, _ = m1(
            all_ids, all_tgt, cos_p, sin_p, kernel_mode="rsi_fused", sac_L_r=1,
            use_chunked_ce=True, global_tokens=glob_par_tok, world_size_scale=1.0,
        )
        l1.backward()
        ref_1g = {
            "loss": float(l1.item()),
            "tok_emb": m1.tok_emb.weight.grad.detach().cpu(),
            "stream_emb": m1.stream_emb.grad.detach().cpu(),
            "blk0_wqkv": m1.blocks[0].w_qkv.weight.grad.detach().cpu(),
            "blk0_phi": m1.blocks[0].mhc_attn.phi.grad.detach().cpu(),
            "blk0_gamma": m1.blocks[0].mhc_attn.gamma.grad.detach().cpu(),
            "blk0_alpha": m1.blocks[0].mhc_attn.alpha.grad.detach().cpu(),
            "blk11_b": m1.blocks[-1].mhc_mlp.b.grad.detach().cpu(),
            "lm_head": m1.lm_head.weight.grad.detach().cpu(),
        }
        del m1, all_ids, all_tgt
        torch.cuda.empty_cache()

    dist.barrier()
    m8 = copy.deepcopy(cpu_mhc_ref).to(device)
    del cpu_mhc_ref
    ignored_8 = _shard_mhc_lm_fsdp2(m8, mp_policy, enable_2block_prefetch=True)
    ids_r = per_rank_ids[rank].to(device)
    tgt_r = per_rank_tgt[rank].to(device)

    l8, _ = m8(
        ids_r, tgt_r, cos_p, sin_p, kernel_mode="rsi_sac", sac_L_r=4,
        use_chunked_ce=True, global_tokens=glob_par_tok, world_size_scale=float(world_size),
    )
    l8.backward()
    # Notice: forward scaled loss by `world_size`, so FSDP2 AVG reduce-scatter yields exact global average on sharded BF16 params;
    # for unsharded ignored FP32 params (`alpha, b, stream_emb`), divide by `world_size` once and SUM across ranks!
    _allreduce_ignored_fp32_grads(ignored_8, world_size)

    l8_avg = l8.detach().to(torch.float64) / float(world_size)
    dist.all_reduce(l8_avg, op=dist.ReduceOp.SUM)

    g_emb = m8.tok_emb.weight.grad.full_tensor().detach().cpu()
    g_qkv0 = m8.blocks[0].w_qkv.weight.grad.full_tensor().detach().cpu()
    g_phi0 = m8.blocks[0].mhc_attn.phi.grad.full_tensor().detach().cpu()
    g_gam0 = m8.blocks[0].mhc_attn.gamma.grad.full_tensor().detach().cpu()
    g_lm = m8.lm_head.weight.grad.full_tensor().detach().cpu()
    g_semb = m8.stream_emb.grad.detach().cpu()
    g_alp0 = m8.blocks[0].mhc_attn.alpha.grad.detach().cpu()
    g_b11 = m8.blocks[-1].mhc_mlp.b.grad.detach().cpu()

    fsdp2_parity: dict[str, Any] = {}
    if rank == 0:
        lv8 = float(l8_avg.item())
        lv1 = ref_1g["loss"]
        l_rd = abs(lv8 - lv1) / max(1e-12, abs(lv1))
        dw_map = {
            "tok_emb": round(rel_l2(g_emb, ref_1g["tok_emb"]), 6),
            "stream_emb_fp32": round(rel_l2(g_semb, ref_1g["stream_emb"]), 6),
            "blk0_wqkv": round(rel_l2(g_qkv0, ref_1g["blk0_wqkv"]), 6),
            "blk0_mhc_phi": round(rel_l2(g_phi0, ref_1g["blk0_phi"]), 6),
            "blk0_mhc_gamma": round(rel_l2(g_gam0, ref_1g["blk0_gamma"]), 6),
            "blk0_mhc_alpha_fp32": round(rel_l2(g_alp0, ref_1g["blk0_alpha"]), 6),
            "blk11_mhc_b_fp32": round(rel_l2(g_b11, ref_1g["blk11_b"]), 6),
            "lm_head": round(rel_l2(g_lm, ref_1g["lm_head"]), 6),
        }
        fsdp2_parity = {
            "param_count": param_count_mhc,
            "loss_1gpu_eager_oracle": round(lv1, 8),
            "loss_8gpu_fsdp2_rsi_sac4": round(lv8, 8),
            "loss_rel_diff": round(l_rd, 8),
            "dw_rel_l2": dw_map,
            "max_dw_rel_l2": round(max(dw_map.values()), 6),
            "parity_ok": (l_rd <= 1e-4) and (max(dw_map.values()) <= 1.5e-2),
        }
    del m8, g_emb, g_qkv0, g_phi0, g_gam0, g_lm, g_semb, g_alp0, g_b11, ref_1g
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    # ------------------------------------------------------------------------
    # B.3 8-GPU NVSwitch `1.07B` `mHC-FSDP2` (`n = 4`) 5-Way Throughput & HBM Scaling
    #     (`S = 2,048` long-context pretraining regime: `B_rank = 4 => 65,536` global tokens/step)
    # ------------------------------------------------------------------------
    B_bench = 1 if is_smoke else 4
    S_bench = 2048
    glob_tok_bench = world_size * B_bench * S_bench
    cos_b, sin_b = build_rope_cos_sin(S_bench, 128, device)
    gb = torch.Generator(device=device).manual_seed(8100 + rank)
    ids_b = torch.randint(0, V_big, (B_bench, S_bench), device=device, generator=gb)
    tgt_b = torch.randint(0, V_big, (B_bench, S_bench), device=device, generator=gb)

    configs = [
        ("ref_1stream_fsdp2_prefetch", "1stream", "rsi_fused", 1, True, True),
        ("rung0_mhc_n4_eager_table2_full_ce", "mhc", "eager", 1, False, False),
        ("rung1_mhc_n4_torch_compile_chunked_ce", "mhc", "compiled", 1, True, False),
        ("rung2_mhc_n4_round3_rsi_fused_Lr1", "mhc", "rsi_fused", 1, True, False),
        ("rung3_mhc_n4_round3_rsi_fused_sac_Lr4_prefetch", "mhc", "rsi_sac", 4, True, True),
    ]

    scaling_out: dict[str, Any] = {}
    for cname, topo, kmode, lr_v, chunk_ce, pf in configs:
        torch.manual_seed(20261293)
        mc = MHCTransformerLM(
            vocab_size=V_big, d_model=D_big, n_layers=L_big, h_q=16, h_kv=4, ffn_dim=I_big, n_streams=4, topology=topo
        ).to(device)
        p_cnt = sum(p.numel() for p in mc.parameters())
        ign_c = _shard_mhc_lm_fsdp2(mc, mp_policy, enable_2block_prefetch=pf)

        def _step_c():
            lv, _ = mc(
                ids_b, tgt_b, cos_b, sin_b, kernel_mode=kmode, sac_L_r=lr_v,
                use_chunked_ce=chunk_ce, global_tokens=glob_tok_bench, world_size_scale=float(world_size),
            )
            lv.backward()
            if ign_c:
                _allreduce_ignored_fp32_grads(ign_c, world_size)
            mc.zero_grad(set_to_none=True)

        w_it = 2 if is_smoke else 3
        g_cnt = 3 if is_smoke else 6
        it_g = 2 if is_smoke else 3
        for _ in range(w_it):
            _step_c()
        torch.cuda.synchronize(device)
        dist.barrier()

        torch.cuda.reset_peak_memory_stats(device)
        _step_c()
        torch.cuda.synchronize(device)
        pk_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        pk_t = torch.tensor([pk_mib], device=device, dtype=torch.float64)
        dist.all_reduce(pk_t, op=dist.ReduceOp.MAX)

        samples: list[float] = []
        for _ in range(g_cnt):
            dist.barrier()
            s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s_ev.record()
            for _ in range(it_g):
                _step_c()
            e_ev.record()
            e_ev.synchronize()
            tl_v = torch.tensor([s_ev.elapsed_time(e_ev) / float(it_g)], device=device, dtype=torch.float64)
            dist.all_reduce(tl_v, op=dist.ReduceOp.MAX)
            samples.append(float(tl_v.item()))

        ss = sorted(samples)
        med = float(statistics.median(ss))
        iqr = float(ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)] - ss[max(0, (len(ss) - 1) // 4)])
        scaling_out[cname] = {
            "param_count": p_cnt,
            "median_step_ms": round(med, 3),
            "iqr_step_ms": round(iqr, 3),
            "global_tok_per_s": round(glob_tok_bench / (med * 1e-3), 1),
            "max_rank_peak_alloc_mib": round(float(pk_t.item()), 2),
        }
        del mc, ign_c
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    t_1s = scaling_out["ref_1stream_fsdp2_prefetch"]["median_step_ms"]
    t_r0 = scaling_out["rung0_mhc_n4_eager_table2_full_ce"]["median_step_ms"]
    t_r3 = scaling_out["rung3_mhc_n4_round3_rsi_fused_sac_Lr4_prefetch"]["median_step_ms"]
    m_r0 = scaling_out["rung0_mhc_n4_eager_table2_full_ce"]["max_rank_peak_alloc_mib"]
    m_r2 = scaling_out["rung2_mhc_n4_round3_rsi_fused_Lr1"]["max_rank_peak_alloc_mib"]
    m_r3 = scaling_out["rung3_mhc_n4_round3_rsi_fused_sac_Lr4_prefetch"]["max_rank_peak_alloc_mib"]

    return {
        "sac_eq20_1gpu_curve": sac_curve_summary,
        "fsdp2_1gpu_parity_gate_g9": fsdp2_parity,
        "fsdp2_8gpu_1_14b_scaling": {
            "b_rank": B_bench,
            "seq_len": S_bench,
            "global_tokens_per_step": glob_tok_bench,
            "configs": scaling_out,
            "speedup_rung3_vs_eager_rung0": round(t_r0 / t_r3, 4),
            "mhc_n4_step_overhead_pct_vs_1stream": round(((t_r3 / t_1s) - 1.0) * 100.0, 2),
            "eager_mhc_n4_step_overhead_pct_vs_1stream": round(((t_r0 / t_1s) - 1.0) * 100.0, 2),
            "memory_saved_rung3_vs_rung0_mib_per_rank": round(m_r0 - m_r3, 2),
            "memory_saved_sac_Lr4_vs_Lr1_mib_per_rank": round(m_r2 - m_r3, 2),
        },
    }


# ============================================================================
# 5. Suite C: Fixed-12.0s-Wall-Clock `8xA100` `val_bpb` Architecture-Kernel
#    Co-Design `autoresearch` Loop (`1-Stream` vs `HC` vs `Eager mHC` vs `RSI mHC`)
# ============================================================================


def _generate_4register_state_tracking_corpus(
    n_seqs: int, seq_len: int, vocab_size: int, seed: int, device: torch.device, grammar_seed: int = 20261299
) -> torch.Tensor:
    """Deterministic 4-Thread Multiplexed State-Tracking & Phrase Corpus (`1` token = `2` UTF-8 bytes).

    Models 4 concurrent latent Markov/register threads interleaved across a single token stream with periodic
    register-pair permutations `SWAP(R_a, R_b)`. All splits (`train`, `val`, `test`) share `grammar_seed`
    while drawing strictly disjoint sequence realizations (`seed`).
    """
    gl = torch.Generator(device="cpu").manual_seed(grammar_seed)
    n_st = 24
    phr = torch.randint(32, vocab_size, (4, n_st, n_st, 3), generator=gl)
    nxt_tbl = torch.randint(0, n_st, (4, n_st, 8), generator=gl)
    g = torch.Generator(device="cpu").manual_seed(seed)
    s_prev = torch.randint(0, n_st, (n_seqs, 4), generator=g)
    s_cur = torch.randint(0, n_st, (n_seqs, 4), generator=g)
    bi = torch.arange(n_seqs)
    out = []
    swap_pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    for f in range(seq_len // 4):
        th = torch.randint(0, 4, (n_seqs,), generator=g)
        inp = torch.randint(0, 8, (n_seqs,), generator=g)
        if f % 4 == 3:
            sw = inp % 6
            for sid, (pa, pb) in enumerate(swap_pairs):
                m = sw == sid
                if bool(m.any()):
                    t1 = s_cur[m, pa].clone()
                    s_cur[m, pa] = s_cur[m, pb]
                    s_cur[m, pb] = t1
                    t2 = s_prev[m, pa].clone()
                    s_prev[m, pa] = s_prev[m, pb]
                    s_prev[m, pb] = t2
        ns = nxt_tbl[th, s_cur[bi, th], inp]
        p0 = phr[th, s_prev[bi, th], ns, 0]
        p1 = phr[th, s_prev[bi, th], ns, 1]
        p2 = phr[th, s_prev[bi, th], ns, 2]
        s_prev[bi, th] = s_cur[bi, th]
        s_cur[bi, th] = ns
        out.append(torch.stack([th * 8 + inp, p0, p1, p2], dim=1))
    return torch.cat(out, dim=1).contiguous().to(device, non_blocking=True)


@torch.no_grad()
def evaluate_autoregressive_bpb(
    model: MHCTransformerLM,
    eval_seqs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    kernel_mode: str,
    sac_L_r: int,
    bytes_per_token: float = 2.0,
) -> tuple[float, dict[str, float]]:
    """Immutable exact global corpus `bpb = sum_nll_nats / (ln(2) * total_utf8_bytes)` + propagation stability telemetry."""
    was_train = model.training
    model.eval()
    inp = eval_seqs[:, :-1].contiguous()
    tgt = eval_seqs[:, 1:].contiguous()
    S_ev = inp.shape[1]
    loss_sum_rank, diag = model(
        inp, tgt, cos[:S_ev], sin[:S_ev],
        kernel_mode=kernel_mode, sac_L_r=sac_L_r,
        use_chunked_ce=True, global_tokens=1, world_size_scale=1.0,
        collect_diagnostics=True,
    )
    n_scored = float(tgt.numel())
    agg = torch.tensor(
        [
            float(loss_sum_rank.item()),
            n_scored * bytes_per_token,
            diag.get("amax_fwd_gain", 1.0),
            diag.get("amax_bwd_gain", 1.0),
            diag.get("stream_pairwise_cos", 1.0),
        ],
        device=eval_seqs.device,
        dtype=torch.float64,
    )
    dist.all_reduce(agg, op=dist.ReduceOp.SUM)
    if was_train:
        model.train()
    ws = float(dist.get_world_size())
    bpb = float(agg[0].item() / (math.log(2.0) * agg[1].item()))
    diag_avg = {
        "amax_fwd_gain": round(float(agg[2].item() / ws), 5),
        "amax_bwd_gain": round(float(agg[3].item() / ws), 5),
        "stream_pairwise_cos": round(float(agg[4].item() / ws), 6),
    }
    return bpb, diag_avg


class _FP32MasterFSDP2AdamW:
    """Exact 1-AllReduce `_foreach` FP32-master-weight AdamW (`eps = 1e-12` per DeepSeek mHC Table 5 scale invariance)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-12,
        weight_decay: float = 0.1,
    ) -> None:
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        self.is_dtensor: list[bool] = [isinstance(p, DTensor) for p in self.params]
        # Apply weight decay strictly to 2D BF16 projection matrices (never decay 1D RMSNorm/alpha/b or stream_emb!)
        self.use_wd: list[bool] = [(p.ndim >= 2 and p.dtype == torch.bfloat16) for p in self.params]
        self.local_params: list[torch.Tensor] = [
            p._local_tensor if isinstance(p, DTensor) else p.data for p in self.params
        ]
        self.masters: list[torch.Tensor] = [lp.detach().float().clone() for lp in self.local_params]
        self.exp_avg: list[torch.Tensor] = [torch.zeros_like(m) for m in self.masters]
        self.exp_avg_sq: list[torch.Tensor] = [torch.zeros_like(m) for m in self.masters]
        self.step_num = 0

    @torch.no_grad()
    def step_with_global_clip(self, lr: float, max_norm: float, device: torch.device) -> float:
        self.step_num += 1
        sharded_g: list[torch.Tensor] = []
        replicated_g: list[torch.Tensor] = []
        all_g: list[torch.Tensor] = []
        for p, is_dt, lp in zip(self.params, self.is_dtensor, self.local_params):
            if p.grad is None:
                gl = torch.zeros_like(lp, dtype=torch.float32)
            else:
                gl = (p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad).detach().float()
            all_g.append(gl)
            if is_dt:
                sharded_g.append(gl)
            else:
                replicated_g.append(gl)

        sq_sharded = sum(n * n for n in torch._foreach_norm(sharded_g)) if sharded_g else torch.zeros((), device=device)
        dist.all_reduce(sq_sharded, op=dist.ReduceOp.SUM)
        sq_rep = sum(n * n for n in torch._foreach_norm(replicated_g)) if replicated_g else torch.zeros((), device=device)
        grad_norm = float(torch.sqrt(sq_sharded + sq_rep).item())
        if not math.isfinite(grad_norm):
            return grad_norm

        clip_coef = min(1.0, max_norm / max(grad_norm, 1e-12))
        if clip_coef < 1.0:
            torch._foreach_mul_(all_g, clip_coef)

        b1, b2 = self.betas
        bc1 = 1.0 - (b1 ** self.step_num)
        bc2 = 1.0 - (b2 ** self.step_num)
        torch._foreach_mul_(self.exp_avg, b1)
        torch._foreach_add_(self.exp_avg, all_g, alpha=1.0 - b1)
        torch._foreach_mul_(self.exp_avg_sq, b2)
        torch._foreach_addcmul_(self.exp_avg_sq, all_g, all_g, value=1.0 - b2)

        if self.weight_decay > 0.0:
            wd_masters = [m for m, u in zip(self.masters, self.use_wd) if u]
            if wd_masters:
                torch._foreach_mul_(wd_masters, 1.0 - lr * self.weight_decay)

        denom = [torch.sqrt(v / bc2).add_(self.eps) for v in self.exp_avg_sq]
        torch._foreach_addcdiv_(self.masters, self.exp_avg, denom, value=-(lr / bc1))

        for lp, m in zip(self.local_params, self.masters):
            lp.copy_(m.to(lp.dtype))
        return grad_norm


def run_suite_c_mhc_valbpb_autoresearch(
    rank: int, world_size: int, device: torch.device, is_smoke: bool, out_dir: Path
) -> dict[str, Any]:
    assert world_size == 8
    dist.barrier()
    budget_s = 4.0 if is_smoke else 12.0

    # Deep 8-block (`16` residual sublayers!) Multi-Stream SLM (`C = 128, 4Q/2KV, D_head = 32, V = 512, seq_len = 513 -> 512 scored`)
    # Width-matched fairness (`GPT 6 Astra, Max` Q4):
    # Reference `1-Stream` uses `ffn_dim = 560` (`2,246,784` params) > `n = 4 mHC` `ffn_dim = 448` (`2,121,931` params),
    # so `n = 4 mHC` has `124,853` STRICTLY FEWER parameters than the `1-Stream` reference control!
    V_slm = 512
    C_slm = 128
    L_slm = 8  # 16 residual sublayers!
    HQ_slm = 4
    HKV_slm = 2
    I_1stream = 560
    I_mhc = 448
    seq_full = 513  # 512 input -> 512 next-token targets!
    S_tok = 512
    B_rank = 16 if is_smoke else 32  # 8 * 32 * 512 = 131,072 global scored tokens/step!
    glob_tok = world_size * B_rank * S_tok

    n_train_pool = 1024 if is_smoke else 4096
    n_eval_rank = 8 if is_smoke else 32
    train_pool = _generate_4register_state_tracking_corpus(
        n_train_pool, 516, V_slm, seed=31000 + rank, device=device
    )[:, :seq_full].contiguous()
    val_seqs = _generate_4register_state_tracking_corpus(
        n_eval_rank, 516, V_slm, seed=41000 + rank, device=device
    )[:, :seq_full].contiguous()
    test_seqs = _generate_4register_state_tracking_corpus(
        n_eval_rank, 516, V_slm, seed=51000 + rank, device=device
    )[:, :seq_full].contiguous()

    cos_c, sin_c = build_rope_cos_sin(S_tok, 32, device)
    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)

    # Canonical shared backbone initialization across all mHC generations (`Gen 0`, `Gen 2`, `Gen 4` start from 100% identical weights!)
    torch.manual_seed(20261297)
    init_1stream_cpu = MHCTransformerLM(
        V_slm, C_slm, L_slm, HQ_slm, HKV_slm, I_1stream, n_streams=1, topology="1stream"
    )
    torch.manual_seed(20261297)
    init_mhc_asym_cpu = MHCTransformerLM(
        V_slm, C_slm, L_slm, HQ_slm, HKV_slm, I_mhc, n_streams=4, topology="mhc",
        symmetric_saddle_init=False, hc_alpha_init=0.25, enable_dynamic_readout=True,
    )
    torch.manual_seed(20261297)
    init_mhc_sym_cpu = MHCTransformerLM(
        V_slm, C_slm, L_slm, HQ_slm, HKV_slm, I_mhc, n_streams=4, topology="mhc",
        symmetric_saddle_init=True, hc_alpha_init=0.25, enable_dynamic_readout=True,
    )
    torch.manual_seed(20261297)
    init_hc_uncon_cpu = MHCTransformerLM(
        V_slm, C_slm, L_slm, HQ_slm, HKV_slm, I_mhc, n_streams=4, topology="hc_unconstrained",
        hc_alpha_init=0.25, enable_dynamic_readout=True,
    )

    params_1stream = sum(p.numel() for p in init_1stream_cpu.parameters())
    params_mhc = sum(p.numel() for p in init_mhc_asym_cpu.parameters())

    # Untrained initial val_bpb probe
    m_probe = copy.deepcopy(init_mhc_asym_cpu).to(device)
    _shard_mhc_lm_fsdp2(m_probe, mp_policy, enable_2block_prefetch=True)
    init_val_bpb, _ = evaluate_autoregressive_bpb(
        m_probe, val_seqs, cos_c, sin_c, kernel_mode="rsi_fused", sac_L_r=1
    )
    del m_probe
    torch.cuda.empty_cache()
    dist.barrier()

    generations = [
        {
            "gen_id": 0,
            "candidate_id": "gen0_eager_deepseek_mhc_n4_table2_baseline",
            "template": "mhc_asym",
            "kernel_mode": "eager",
            "sac_L_r": 1,
            "prefetch": False,
            "chunked_ce": False,
            "peak_lr": 3.0e-3,
            "max_grad_norm": 1.0,
            "note": "Eager DeepSeek mHC n=4 (arXiv:2512.24880 Eq. 7-9 Table-2 baseline): stable Amax Gain=1.0, I/O-bottlenecked",
        },
        {
            "gen_id": 1,
            "candidate_id": "gen1_unconstrained_hc_n4_manifold_drop_shortcut",
            "template": "hc_uncon",
            "kernel_mode": "eager",
            "sac_L_r": 1,
            "prefetch": False,
            "chunked_ce": True,
            "peak_lr": 6.0e-3,
            "max_grad_norm": 50.0,
            "note": "Manifold-drop shortcut (Unconstrained HC n=4, skip Sinkhorn-20): composite Amax Gain explodes >500x -> REVERT_VAL_BPB",
        },
        {
            "gen_id": 2,
            "candidate_id": "gen2_torch_compile_inductor_mhc_n4_chunked_ce",
            "template": "mhc_asym",
            "kernel_mode": "compiled",
            "sac_L_r": 1,
            "prefetch": True,
            "chunked_ce": True,
            "peak_lr": 3.0e-3,
            "max_grad_norm": 1.0,
            "note": "torch.compile (inductor) mHC n=4 + Chunked Linear-CE + 2-block prefetch -> KEEP",
        },
        {
            "gen_id": 3,
            "candidate_id": "gen3_rsi_fused_mhc_n4_symmetric_equal_stream_saddle",
            "template": "mhc_sym",
            "kernel_mode": "rsi_fused",
            "sac_L_r": 1,
            "prefetch": True,
            "chunked_ce": True,
            "peak_lr": 3.0e-3,
            "max_grad_norm": 1.0,
            "note": "Round-3 RSI Fused mHC n=4 with naive symmetric zero-init (stream_emb=0, phi=0): dZ_res==0 equal-stream saddle -> REVERT_VAL_BPB",
        },
        {
            "gen_id": 4,
            "candidate_id": "gen4_round3_rsi_fused_mhc_n4_asymmetric_birkhoff_codesign",
            "template": "mhc_asym",
            "kernel_mode": "rsi_fused",
            "sac_L_r": 1,
            "prefetch": True,
            "chunked_ce": True,
            "peak_lr": 3.0e-3,
            "max_grad_norm": 1.0,
            "note": "Full Round-3 RSI Fused mHC n=4 + On-Chip Log-Sinkhorn-20 + Superfused Pre+RMSNorm + Dynamic H_final^pre Readout + Broken-Symmetry Birkhoff Init -> KEEP",
        },
    ]

    def _build_gen_model(tmpl: str, pf: bool) -> tuple[MHCTransformerLM, list[nn.Parameter]]:
        src = (
            init_1stream_cpu if tmpl == "1stream"
            else (init_hc_uncon_cpu if tmpl == "hc_uncon" else (init_mhc_sym_cpu if tmpl == "mhc_sym" else init_mhc_asym_cpu))
        )
        m = copy.deepcopy(src).to(device)
        ign = _shard_mhc_lm_fsdp2(m, mp_policy, enable_2block_prefetch=pf)
        return m, ign

    def _get_batch(step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s0 = (step_idx * B_rank) % (n_train_pool - B_rank)
        seq = train_pool[s0 : s0 + B_rank]
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    incumbent_val_bpb: float | None = None
    incumbent_gen_id = 0
    gen_records: list[dict[str, Any]] = []
    gen4_completed_steps = 0

    for gcfg in generations:
        gid = gcfg["gen_id"]

        # Zero-Advantage Pre-Timer Compilation Warmup (`PROGRAM.md` Sec. 5 Rule 2):
        # Warm Triton/cuBLAS/FSDP2 kernels outside timer, then discard all state and instantiate fresh canonical weights!
        m_warm, ign_warm = _build_gen_model(gcfg["template"], gcfg["prefetch"])
        opt_warm = _FP32MasterFSDP2AdamW(m_warm)
        inp_w, tgt_w = _get_batch(0)
        for _ in range(2):
            lw, _ = m_warm(
                inp_w, tgt_w, cos_c, sin_c,
                kernel_mode=gcfg["kernel_mode"], sac_L_r=gcfg["sac_L_r"],
                use_chunked_ce=gcfg["chunked_ce"], global_tokens=glob_tok, world_size_scale=float(world_size),
            )
            lw.backward()
            if ign_warm:
                _allreduce_ignored_fp32_grads(ign_warm, world_size)
            opt_warm.step_with_global_clip(1e-4, gcfg["max_grad_norm"], device)
            m_warm.zero_grad(set_to_none=True)
        del m_warm, ign_warm, opt_warm
        gc.collect()
        torch.cuda.empty_cache()

        # Fresh canonical weights + fresh FP32 master AdamW + cursor 0 before synchronized 12.0s timer!
        m_run, ign_run = _build_gen_model(gcfg["template"], gcfg["prefetch"])
        opt_run = _FP32MasterFSDP2AdamW(m_run)
        torch.cuda.synchronize(device)
        dist.barrier()

        t_start = time.perf_counter()
        step_idx = 0
        elapsed_sync = 0.0
        peak_grad_norm = 0.0
        diverged = False
        ctrl = torch.zeros((2,), device=device, dtype=torch.float64)

        while True:
            u_prog = min(1.0, elapsed_sync / budget_s)
            if u_prog < 0.08:
                lr_cur = gcfg["peak_lr"] * (u_prog / 0.08 + 0.05)
            else:
                decay_u = (u_prog - 0.08) / 0.92
                lr_cur = gcfg["peak_lr"] * (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * decay_u)))

            inp_b, tgt_b = _get_batch(step_idx)
            lv, _ = m_run(
                inp_b, tgt_b, cos_c, sin_c,
                kernel_mode=gcfg["kernel_mode"], sac_L_r=gcfg["sac_L_r"],
                use_chunked_ce=gcfg["chunked_ce"], global_tokens=glob_tok, world_size_scale=float(world_size),
            )
            lv.backward()
            if ign_run:
                _allreduce_ignored_fp32_grads(ign_run, world_size)
            gnorm = opt_run.step_with_global_clip(lr_cur, gcfg["max_grad_norm"], device)
            m_run.zero_grad(set_to_none=True)
            step_idx += 1
            if math.isfinite(gnorm):
                peak_grad_norm = max(peak_grad_norm, gnorm)
            else:
                diverged = True

            torch.cuda.synchronize(device)
            if rank == 0:
                el = time.perf_counter() - t_start
                stop_f = 1.0 if (el >= budget_s or diverged) else 0.0
                ctrl[0] = stop_f
                ctrl[1] = el
            dist.broadcast(ctrl, src=0)
            elapsed_sync = float(ctrl[1].item())
            if ctrl[0].item() >= 0.5:
                break

        val_bpb, val_diag = evaluate_autoregressive_bpb(
            m_run, val_seqs, cos_c, sin_c, kernel_mode=gcfg["kernel_mode"], sac_L_r=gcfg["sac_L_r"]
        )
        test_bpb, _ = evaluate_autoregressive_bpb(
            m_run, test_seqs, cos_c, sin_c, kernel_mode=gcfg["kernel_mode"], sac_L_r=gcfg["sac_L_r"]
        )
        if not math.isfinite(val_bpb):
            val_bpb = 99.0
            test_bpb = 99.0

        if gid == 0:
            decision = "BASELINE"
            incumbent_val_bpb = val_bpb
            incumbent_gen_id = 0
        else:
            assert incumbent_val_bpb is not None
            if val_bpb <= incumbent_val_bpb - 0.01:
                decision = "KEEP"
                incumbent_val_bpb = val_bpb
                incumbent_gen_id = gid
            else:
                decision = "REVERT_VAL_BPB"

        if gid == 4:
            gen4_completed_steps = step_idx

        p_count = params_1stream if gcfg["template"] == "1stream" else params_mhc
        gen_records.append(
            {
                "gen_id": gid,
                "candidate_id": gcfg["candidate_id"],
                "param_count": p_count,
                "budget_s": budget_s,
                "elapsed_s": round(elapsed_sync, 4),
                "completed_steps": step_idx,
                "steps_per_sec": round(step_idx / max(elapsed_sync, 1e-9), 3),
                "global_tok_per_s": round((step_idx * glob_tok) / max(elapsed_sync, 1e-9), 1),
                "val_bpb": round(val_bpb, 5),
                "untouched_test_bpb": round(test_bpb, 5),
                "amax_fwd_gain": val_diag["amax_fwd_gain"],
                "amax_bwd_gain": val_diag["amax_bwd_gain"],
                "stream_pairwise_cos": val_diag["stream_pairwise_cos"],
                "peak_grad_norm": round(peak_grad_norm, 4),
                "decision": decision,
                "incumbent_val_bpb_after": round(float(incumbent_val_bpb), 5),
                "hypothesis_note": gcfg["note"],
            }
        )
        del m_run, ign_run, opt_run
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    # Supplementary Width-Matched `1-Stream` Control (`GPT 6 Astra, Max` Q4, zero baseline handicapping!):
    # Train width-matched `1-Stream` (`2,246,784` params > `2,121,931` mHC params) at `1-Stream`'s own oracle-best `lr = 1.0e-3`
    # for the EXACT same integer step count `gen4_completed_steps` (plus record its aggressive `lr = 3.0e-3` instability control),
    # verifying both per-token architectural capacity AND mHC's 3x learning-rate stability headroom!
    m_ctrl, _ = _build_gen_model("1stream", True)
    opt_ctrl = _FP32MasterFSDP2AdamW(m_ctrl, weight_decay=0.05)
    for st in range(gen4_completed_steps):
        u_p = float(st) / max(1.0, float(gen4_completed_steps))
        lr_c = 1.0e-3 * ((u_p / 0.08 + 0.05) if u_p < 0.08 else (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * (u_p - 0.08) / 0.92))))
        ib, tb = _get_batch(st)
        lc, _ = m_ctrl(
            ib, tb, cos_c, sin_c, kernel_mode="rsi_fused", sac_L_r=1,
            use_chunked_ce=True, global_tokens=glob_tok, world_size_scale=float(world_size),
        )
        lc.backward()
        opt_ctrl.step_with_global_clip(lr_c, 1.0, device)
        m_ctrl.zero_grad(set_to_none=True)
    ctrl_1s_val_bpb, _ = evaluate_autoregressive_bpb(m_ctrl, val_seqs, cos_c, sin_c, kernel_mode="rsi_fused", sac_L_r=1)
    ctrl_1s_test_bpb, _ = evaluate_autoregressive_bpb(m_ctrl, test_seqs, cos_c, sin_c, kernel_mode="rsi_fused", sac_L_r=1)
    del m_ctrl, opt_ctrl
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    ledger_sha = ""
    if rank == 0:
        ledger_path = out_dir / "valbpb_autoresearch_ledger.tsv"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "gen_id", "candidate_id", "param_count", "budget_s", "elapsed_s", "completed_steps",
            "steps_per_sec", "global_tok_per_s", "val_bpb", "untouched_test_bpb",
            "amax_fwd_gain", "amax_bwd_gain", "stream_pairwise_cos", "peak_grad_norm",
            "decision", "incumbent_val_bpb_after", "hypothesis_note",
        ]
        with ledger_path.open("w", encoding="utf-8", newline="\n") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for row in gen_records:
                w.writerow(row)
        ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    final_rec = next(r for r in gen_records if r["gen_id"] == incumbent_gen_id)
    gen0_rec = gen_records[0]
    gen2_rec = gen_records[2]
    gen4_rec = gen_records[4]

    return {
        "params_1stream_width_matched": params_1stream,
        "params_mhc_n4": params_mhc,
        "mhc_has_fewer_params_than_1stream_baseline": params_mhc <= params_1stream,
        "vocab_size": V_slm,
        "d_model": C_slm,
        "n_blocks": L_slm,
        "n_residual_sublayers": 2 * L_slm,
        "seq_len": S_tok,
        "global_tokens_per_step": glob_tok,
        "budget_seconds_per_generation": budget_s,
        "untrained_init_val_bpb": round(init_val_bpb, 5),
        "baseline_gen0_eager_mhc_val_bpb": gen0_rec["val_bpb"],
        "compiled_mhc_gen2_val_bpb": gen2_rec["val_bpb"],
        "final_incumbent_gen_id": incumbent_gen_id,
        "final_incumbent_candidate": final_rec["candidate_id"],
        "final_incumbent_val_bpb": final_rec["val_bpb"],
        "final_incumbent_untouched_test_bpb": final_rec["untouched_test_bpb"],
        "val_bpb_gain_vs_eager_mhc_gen0": round(gen0_rec["val_bpb"] - final_rec["val_bpb"], 5),
        "val_bpb_gain_vs_compiled_mhc_gen2": round(gen2_rec["val_bpb"] - final_rec["val_bpb"], 5),
        "test_bpb_gain_vs_eager_mhc_gen0": round(gen0_rec["untouched_test_bpb"] - final_rec["untouched_test_bpb"], 5),
        "step_multiplier_gen4_vs_eager_mhc_gen0": round(gen4_rec["completed_steps"] / max(1, gen0_rec["completed_steps"]), 4),
        "matched_step_1stream_oracle_lr_control": {
            "matched_steps": gen4_completed_steps,
            "control_1stream_oracle_lr": 1.0e-3,
            "control_1stream_param_count": params_1stream,
            "control_1stream_val_bpb": round(ctrl_1s_val_bpb, 5),
            "control_1stream_untouched_test_bpb": round(ctrl_1s_test_bpb, 5),
            "gen4_mhc_param_count": params_mhc,
            "gen4_mhc_val_bpb": gen4_rec["val_bpb"],
            "gen4_mhc_untouched_test_bpb": gen4_rec["untouched_test_bpb"],
            "per_token_val_bpb_gain_mhc_over_1stream": round(ctrl_1s_val_bpb - gen4_rec["val_bpb"], 5),
        },
        "valbpb_ledger_sha256": ledger_sha,
        "generations": gen_records,
    }


# ============================================================================
# 6. Main Entrypoint (`torchrun --nproc_per_node=8`)
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_smoke = args.mode == "smoke"

    t_wall0 = time.perf_counter()
    harness_sha = eval_harness_sha256()

    suite0_res: dict[str, Any] = {}
    suitea_res: dict[str, Any] = {}
    rsi_ledger_sha = ""
    if rank == 0:
        suite0_res, rt_ratchet = run_suite_0_mhc_audits(device, is_smoke)
        suitea_res, ratchets_a = run_suite_a_mhc_ratchets(device, is_smoke)
        rsi_ledger_sha = write_rsi_ledger_tsv(
            [rt_ratchet, *ratchets_a], args.out_dir / "rsi_ledger.tsv"
        )
        gc.collect()
        torch.cuda.empty_cache()

    dist.barrier()
    suiteb_res = run_suite_b_sac_and_8gpu_fsdp2(rank, world_size, device, is_smoke)
    dist.barrier()
    suitec_res = run_suite_c_mhc_valbpb_autoresearch(rank, world_size, device, is_smoke, args.out_dir)
    dist.barrier()

    total_wall_s = time.perf_counter() - t_wall0
    clk_end, pwr_end = query_sm_telemetry(0)

    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_json = args.out_dir / f"mhc_manifold_hyper_connections_rsi_{args.mode}.json"
        payload = {
            "experiment": "E129_deepseek_mhc_manifold_hyper_connections_round3_rsi",
            "mode": args.mode,
            "harness_version": HARNESS_VERSION,
            "eval_harness_sha256": harness_sha,
            "rsi_ledger_tsv_sha256": rsi_ledger_sha,
            "total_wall_seconds": round(total_wall_s, 2),
            "end_gpu0_sm_clock_mhz": clk_end,
            "end_gpu0_power_w": pwr_end,
            "environment": environment_metadata(),
            "suite_0_adversarial_and_oracle_audits": suite0_res,
            "suite_a_round3_mhc_vector_ratchets": suitea_res,
            "suite_b_sac_eq20_and_8gpu_1_14b_fsdp2": suiteb_res,
            "suite_c_fixed_wallclock_valbpb_autoresearch": suitec_res,
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "OK",
                    "mode": args.mode,
                    "wall_s": round(total_wall_s, 2),
                    "eval_harness_sha256": harness_sha,
                    "out_json": str(out_json),
                }
            ),
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
