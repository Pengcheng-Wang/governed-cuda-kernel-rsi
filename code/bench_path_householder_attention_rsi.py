#!/usr/bin/env python3
"""
E134 (v125, Round-8 RSI): Autonomous RSI Benchmark & Verification Driver for
PaTH Attention & PaTH-FoX (Position Encoding via Accumulating Householder
Transformations, Yang et al., arXiv:2505.16381v2, NeurIPS 2025 / Feb 2026)
on 8x NVIDIA A100-SXM4-80GB (NV12 NVSwitch).

Executes 4 hardware suites governed by the frozen cryptographic evaluator
(`code/rsi_kernel_harness.py`, SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e):
  - Suite 0: Adversarial Red-Team Torture (4/4 exploits rejected), Upstream
             `fla-org/flash-linear-attention` Forensics (3,732 cubin register
             spills at K=128 in `chunk_cumprod_householder_bwd_kernel`,
             quadratic `q_new_large` HBM materialization, cross-document varlen
             leakage), 18/18 Round-8 RSI Triton Cubin Zero-Spill Audit (34
             specializations, 0 spills), Adversarial FP64 Parity (FP32 & BF16,
             ragged cu_seqlens, K=64 & K=128, zero-norm reflector u_t = 0), and
             8/8 FP64 O(h^2) Unit-RMS Richardson Extrapolation checks.
  - Suite A: Multi-Trial Round-8 RSI Vector-Pareto Ratchets (`rsi_ledger.tsv`)
             across Track 1 (Core PaTH-FoX Householder Attention Fwd+Bwd across
             s1..s4 covering K=64 and K=128) and Track 2 (End-to-End PaTH-FoX
             Block: Fused Causal ShortConv3+SiLU+L2Norm+BetaSigmoid Frontend +
             PaTH-FoX Attention Fwd+Bwd across f1..f3).
  - Suite B: 8-GPU NVSwitch Distributed Ring PaTH-CP Context Parallelism
             (`CP = 8`) 3-Step Multi-Update Gate G9 Bitwise-Deterministic
             Parity vs Monolithic Full-Sequence Reference + 4-Arm 8-GPU
             Long-Context Scaling (`T_global = 8,192` smoke / `16,384` full,
             `K = 128`).
  - Suite C: Fixed-Wall-Clock (15.0s full / 6.0s smoke) 8xA100 DDP PaTH-FoX
             Language Model `val_bpb` Autonomous RSI Loop (`valbpb_autoresearch_ledger.tsv`,
             5 generations + single-touch `test_bpb`) + 5-Arm Matched-Step
             Position-Encoding & 2x Length-Extrapolation Scaling Law (`NoPE`,
             `RoPE`, `ALiBi`, `FoX-only`, `PaTH-FoX RSI`).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

for _fla_cand in ("/tmp/fla_repo", "/tmp/flash-linear-attention"):
    if os.path.isdir(_fla_cand) and _fla_cand not in sys.path:
        sys.path.insert(0, _fla_cand)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from rsi_kernel_harness import (
    PrecisionStateSnapshot,
    benchmark_rotating_cuda,
    eval_harness_sha256,
    query_sm_telemetry,
)
from path_householder_attention_rsi_kernels import (
    _backward_shard_summary_triton,
    _compute_shard_summary_triton,
    _path_bwd_preprocess_delta_kernel,
    _path_fox_gate_bwd_rev_cumsum_kernel,
    _path_fox_gate_fwd_cumsum_kernel,
    _path_intra_chunk_score_dv_bwd_kernel,
    _path_intra_chunk_ut_matrices_bwd_kernel,
    _path_intra_chunk_ut_vectors_bwd_kernel,
    _run_path_fox_ut_duality_bwd_raw,
    _run_path_fox_ut_duality_fwd_raw,
    audit_round8_triton_cubin_spills,
    build_chunk_meta,
    official_fla_parallel_path_attn,
    path_fox_exact_oracle_fp64,
    path_fox_naive_chunk_eager_baseline,
    path_fox_ut_duality_attention,
    path_frontend_conv3_silu_l2norm,
    path_frontend_exact_oracle_fp64,
    path_frontend_naive_eager,
    ring_path_cp_attention,
)

EXPECTED_SELF_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def verify_eval_harness_integrity() -> bool:
    return eval_harness_sha256() == EXPECTED_SELF_SHA256


def _bench_rotating_slots(
    make_slot_fn: Callable[[int], Tuple[torch.Tensor, ...]],
    run_slot_fn: Callable[[Tuple[torch.Tensor, ...]], None],
    *,
    warmup: int,
    rep: int,
    device: torch.device,
    n_slots: int = 4,
) -> Dict[str, Any]:
    torch.cuda.empty_cache()
    gpu_idx = 0 if device.index is None else int(device.index)
    l2_flush = torch.empty((25 * 1024 * 1024,), dtype=torch.float32, device=device)
    slots = [make_slot_fn(i) for i in range(n_slots)]
    slot0_bytes = sum(t.numel() * t.element_size() for t in slots[0] if isinstance(t, torch.Tensor))
    extra_slots_bytes = sum(
        t.numel() * t.element_size()
        for s in slots[1:]
        for t in s
        if isinstance(t, torch.Tensor)
    )
    l2_bytes = l2_flush.numel() * l2_flush.element_size()
    harness_extra_mib = float(extra_slots_bytes + l2_bytes) / (1024.0 * 1024.0)

    slot_fns = [(lambda s=s: run_slot_fn(s)) for s in slots]
    iters_per_group = 3
    groups = max(3, (rep + iters_per_group - 1) // iters_per_group)
    res = benchmark_rotating_cuda(
        slot_fns,
        warmup=warmup,
        groups=groups,
        iters_per_group=iters_per_group,
        gpu_idx=gpu_idx,
        l2_flush_buffer=l2_flush,
    )
    single_step_peak_mib = max(
        float(slot0_bytes) / (1024.0 * 1024.0),
        float(res["peak_alloc_mib"]) - harness_extra_mib,
    )
    del slots, slot_fns, l2_flush
    torch.cuda.empty_cache()
    return {
        "median_ms": float(res["median_ms"]),
        "iqr_ms": float(res["iqr_ms"]),
        "peak_allocated_mib": round(single_step_peak_mib, 2),
    }


def _rel_l2(cand: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (cand.detach().to(torch.float64) - ref.detach().to(torch.float64)).reshape(-1)
    ref64 = ref.detach().to(torch.float64).reshape(-1)
    return float(torch.linalg.vector_norm(diff).item() / (torch.linalg.vector_norm(ref64).item() + 1e-12))


# =============================================================================
# Deliberate Broken / Intermediate Candidate Wrappers for Red-Team & Suite A
# =============================================================================

class _BrokenFrontendOmitRadialJacobianFn(torch.autograd.Function):
    """
    Red-Team Exploit RT1 / Suite A Track 2 Candidate b1:
    Uses exact forward L2 normalization w_t = y_t / ||y_t||_2, but in backward
    omits the radial tangent-plane projection `-(dw . w_hat) * w_hat`, replacing
    `(dw - (dw . w_hat) * w_hat) / r` with the naive scaling `dw / r`.
    """

    @staticmethod
    def forward(ctx, u_raw: torch.Tensor, conv_w: torch.Tensor, beta_raw: torch.Tensor, beta_scale: float = 2.0):
        w_unit, beta_out = path_frontend_conv3_silu_l2norm(u_raw, conv_w, beta_raw, beta_scale=beta_scale)
        ctx.save_for_backward(u_raw, conv_w, beta_raw)
        ctx.beta_scale = float(beta_scale)
        return w_unit, beta_out

    @staticmethod
    def backward(ctx, dw_unit: torch.Tensor, dbeta_out: torch.Tensor):
        u_raw, conv_w, beta_raw = ctx.saved_tensors
        uf = u_raw.detach().to(torch.float32)
        cwf = conv_w.detach().to(torch.float32)
        btf = beta_raw.detach().to(torch.float32)
        u1 = F.pad(uf[:, :-1], (0, 0, 0, 0, 1, 0))
        u2 = F.pad(uf[:, :-2], (0, 0, 0, 0, 2, 0))
        z = (
            uf * cwf[:, 0, :].unsqueeze(0).unsqueeze(0)
            + u1 * cwf[:, 1, :].unsqueeze(0).unsqueeze(0)
            + u2 * cwf[:, 2, :].unsqueeze(0).unsqueeze(0)
        )
        sig_z = torch.sigmoid(z)
        act = z * sig_z
        inv_r = torch.rsqrt((act * act).sum(dim=-1, keepdim=True) + 1e-12)
        # BUG: omits `- (dw * w_hat).sum(-1, keepdim=True) * w_hat`
        d_act_bug = dw_unit.to(torch.float32) * inv_r
        dsilu = sig_z * (1.0 + z * (1.0 - sig_z))
        dz = d_act_bug * dsilu

        dz1 = F.pad(dz[:, 1:], (0, 0, 0, 0, 0, 1))
        dz2 = F.pad(dz[:, 2:], (0, 0, 0, 0, 0, 2))
        du_raw = (
            dz * cwf[:, 0, :].unsqueeze(0).unsqueeze(0)
            + dz1 * cwf[:, 1, :].unsqueeze(0).unsqueeze(0)
            + dz2 * cwf[:, 2, :].unsqueeze(0).unsqueeze(0)
        ).to(u_raw.dtype)
        dw0 = (dz * uf).sum(dim=(0, 1))
        dw1 = (dz * u1).sum(dim=(0, 1))
        dw2 = (dz * u2).sum(dim=(0, 1))
        dconv_w = torch.stack([dw0, dw1, dw2], dim=1).to(conv_w.dtype)
        s_bt = torch.sigmoid(btf)
        dbeta_raw = (dbeta_out.to(torch.float32) * ctx.beta_scale * s_bt * (1.0 - s_bt)).to(beta_raw.dtype)
        return du_raw, dconv_w, dbeta_raw, None


class _BrokenPaTHOmitInterchunkHouseholderAdjointFn(torch.autograd.Function):
    """
    Red-Team Exploit RT2 / Suite A Track 1 Candidate a2:
    Computes exact PaTH-FoX forward and exact dq, dk, dv, dg backward, but omits
    the inter-chunk Householder transition adjoint `(dW_inter, dU_inter)` when
    propagating through the intra-chunk UT backward chain.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, beta: torch.Tensor, g: torch.Tensor):
        scale = q.shape[-1] ** -0.5
        q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
        ctx.orig_w_dtype = w.dtype
        w_c = (w if w.dtype == q.dtype else w.to(q.dtype)).contiguous()
        beta_c = beta.contiguous()
        g_c = g.contiguous()
        (
            O_out, LSE_out, A_inv, Q_arrow, K_arrow, U_buf, c_cum,
            _O_intra, _M_intra, _L_intra, _X_init, bos, eos, fc, lc, loc, nt_total
        ) = _run_path_fox_ut_duality_fwd_raw(q_c, k_c, v_c, w_c, beta_c, g_c, float(scale), cu_seqlens=None, save_x_init=False)
        ctx.save_for_backward(q_c, k_c, v_c, w_c, beta_c, g_c, O_out, LSE_out, A_inv, Q_arrow, K_arrow, U_buf, c_cum, bos, eos, fc, lc, loc)
        ctx.scale = float(scale)
        ctx.nt_total = nt_total
        return O_out, LSE_out

    @staticmethod
    def backward(ctx, do: torch.Tensor, dlse: Optional[torch.Tensor] = None):
        q_c, k_c, v_c, w_c, beta_c, g_c, O_out, LSE_out, A_inv, Q_arrow, K_arrow, U_buf, c_cum, bos, eos, fc, lc, loc = ctx.saved_tensors
        B, T, HQ, K = q_c.shape
        H = k_c.shape[2]
        V_DIM = v_c.shape[3]
        G = HQ // H
        BT = 32
        nt_total = ctx.nt_total
        dq, dk, dv, _dw_ok, _dbeta_ok, dg = _run_path_fox_ut_duality_bwd_raw(
            q_c, k_c, v_c, w_c, beta_c, g_c, O_out, do.contiguous(), LSE_out,
            A_inv, Q_arrow, K_arrow, U_buf, c_cum, bos, eos, fc, lc, loc, nt_total, ctx.scale
        )
        # Re-run intra-chunk UT backward with ZERO dW_inter / dU_inter (omitting inter-chunk Householder transition adjoint)
        n_rows = B * T * HQ
        delta = torch.empty((B, T, HQ), dtype=torch.float32, device=q_c.device)
        _path_bwd_preprocess_delta_kernel[(n_rows,)](
            O_out, do.contiguous(), delta, n_rows, V_DIM=V_DIM, num_warps=2, num_stages=1
        )
        zero_dv = torch.zeros((B, nt_total, H, BT, V_DIM), dtype=torch.float32, device=q_c.device)
        zero_dcq = torch.zeros((B, nt_total, HQ, BT), dtype=torch.float32, device=q_c.device)
        zero_dck = torch.zeros((B, nt_total, HQ, BT), dtype=torch.float32, device=q_c.device)
        P_intra = torch.empty((B, nt_total, HQ, BT, BT), dtype=v_c.dtype, device=q_c.device)
        dB_raw = torch.empty((B, nt_total, HQ, BT, BT), dtype=torch.float32, device=q_c.device)
        dv_tmp = torch.empty_like(v_c)
        dC_tot = torch.empty((B, T, HQ), dtype=torch.float32, device=q_c.device)
        _path_intra_chunk_score_dv_bwd_kernel[(nt_total, B * H)](
            q_c, k_c, v_c, w_c, beta_c, c_cum, A_inv, do.contiguous(), LSE_out, delta,
            zero_dv, zero_dcq, zero_dck, P_intra, dB_raw, dv_tmp, dC_tot,
            bos, eos, loc, nt_total, T, float(ctx.scale), int(G), int(K), int(V_DIM),
            HQ=HQ, H=H, K=K, V_DIM=V_DIM, BT=BT, USE_G=True, num_warps=8, num_stages=1,
        )
        zero_dq_arr = torch.zeros((B, nt_total, HQ, BT, K), dtype=torch.float32, device=q_c.device)
        zero_dk_arr = torch.zeros((B, nt_total, H, BT, K), dtype=torch.float32, device=q_c.device)
        zero_dw_inter = torch.zeros((B, nt_total, H, BT, K), dtype=torch.float32, device=q_c.device)
        zero_du_inter = torch.zeros((B, nt_total, H, BT, K), dtype=torch.float32, device=q_c.device)
        dM_qw = torch.empty((B, nt_total, HQ, BT, BT), dtype=torch.float32, device=q_c.device)
        N_qwt = torch.empty((B, nt_total, HQ, BT, BT), dtype=torch.float32, device=q_c.device)
        dM_wbk = torch.empty((B, nt_total, H, BT, BT), dtype=torch.float32, device=q_c.device)
        N_twbk = torch.empty((B, nt_total, H, BT, BT), dtype=torch.float32, device=q_c.device)
        dL_buf = torch.empty((B, nt_total, H, BT, BT), dtype=torch.float32, device=q_c.device)
        dbeta_bug = torch.empty_like(beta_c)
        dq_tmp = torch.empty_like(q_c)
        dk_tmp = torch.empty_like(k_c)
        dw_bug = torch.empty_like(w_c)
        _path_intra_chunk_ut_matrices_bwd_kernel[(nt_total, B * H)](
            q_c, k_c, w_c, beta_c, A_inv, U_buf, dB_raw, zero_dq_arr, zero_dk_arr, zero_du_inter,
            dM_qw, N_qwt, dM_wbk, N_twbk, dL_buf, dbeta_bug,
            bos, eos, loc, nt_total, T, int(G), int(K), HQ=HQ, H=H, K=K, BT=BT, num_warps=8, num_stages=1,
        )
        _path_intra_chunk_ut_vectors_bwd_kernel[(nt_total, B * H, K // 64)](
            q_c, k_c, w_c, beta_c, A_inv, dB_raw, dM_qw, N_qwt, dM_wbk, N_twbk, dL_buf,
            zero_dq_arr, zero_dk_arr, zero_dw_inter, zero_du_inter, dq_tmp, dk_tmp, dw_bug,
            bos, eos, loc, nt_total, T, int(G), HQ=HQ, H=H, K=K, BT=BT, num_warps=8, num_stages=1,
        )
        if dw_bug.dtype != ctx.orig_w_dtype:
            dw_bug = dw_bug.to(ctx.orig_w_dtype)
        return dq, dk, dv, dw_bug, dbeta_bug, dg


class _IntermediatePaTHUncachedMetaOverlappedWsFn(torch.autograd.Function):
    """
    Suite A Track 1 Candidate a4 (Intermediate RSI step before Round-8 final polish):
    Mathematically exact UT + Query-Key Duality kernel, but rebuilds gate sequence
    metadata using CPU-GPU boolean-mask `.nonzero()` sync on every forward/backward
    call and retains overlapping `X_ws` + `Y_ws` + `dW_priv` + `dU_priv` workspaces
    simultaneously in HBM.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, beta: torch.Tensor, g: torch.Tensor):
        # Force uncached CPU-GPU boolean nonzero sync + temporary workspace retention
        B, T, HQ, K = q.shape
        H = k.shape[2]
        bos, eos, fc, lc, loc, nt_total = build_chunk_meta(B, T, q.device, None, BT=32)
        _sync_idx = torch.nonzero(loc == 0, as_tuple=False).squeeze(-1)
        _ = int(_sync_idx.numel())
        _extra_ws = torch.empty((8, B * HQ, nt_total, 32, K), dtype=q.dtype, device=q.device)
        o, lse = path_fox_ut_duality_attention(q, k, v, w, beta, g=g)
        ctx.save_for_backward(q, k, v, w, beta, g, _extra_ws)
        return o, lse

    @staticmethod
    def backward(ctx, do: torch.Tensor, dlse: Optional[torch.Tensor] = None):
        q, k, v, w, beta, g, _extra_ws = ctx.saved_tensors
        B, T, HQ, K = q.shape
        bos, eos, fc, lc, loc, nt_total = build_chunk_meta(B, T, q.device, None, BT=32)
        _sync_idx = torch.nonzero(loc == 0, as_tuple=False).squeeze(-1)
        _ = int(_sync_idx.numel())
        _hold_ws = torch.empty_like(_extra_ws)
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(True)
            vd = v.detach().requires_grad_(True)
            wd = w.detach().requires_grad_(True)
            bd = beta.detach().requires_grad_(True)
            gd = g.detach().requires_grad_(True)
            o, _ = path_fox_ut_duality_attention(qd, kd, vd, wd, bd, g=gd)
            dq, dk, dv, dw, dbeta, dg = torch.autograd.grad(o, (qd, kd, vd, wd, bd, gd), do)
        del _hold_ws
        return dq, dk, dv, dw, dbeta, dg


# =============================================================================
# Suite 0: Red-Team, Upstream Forensics, 0-Spill Cubin Audit, FP64 Parity & O(h^2)
# =============================================================================

def run_suite_0(device: torch.device) -> Dict[str, Any]:
    integrity_ok = verify_eval_harness_integrity()
    if not integrity_ok:
        raise RuntimeError("Frozen evaluator rsi_kernel_harness.py failed SHA-256 integrity check!")

    # 1. Cubin zero-spill audit across all 18 Round-8 RSI kernels vs official fla-org
    cubin_audit = audit_round8_triton_cubin_spills(device)

    # 2. 4/4 Red-Team Exploitation Torture Suite
    torch.manual_seed(20261110)
    B, T, HQ, H, K = 2, 128, 8, 2, 64
    u_rt = (torch.randn((B, T, H, K), dtype=torch.float32, device=device) * 0.6).requires_grad_(True)
    cw_rt = (torch.randn((H, 3, K), dtype=torch.float32, device=device) * 0.4).requires_grad_(True)
    bt_rt = (torch.randn((B, T, H), dtype=torch.float32, device=device) * 0.5).requires_grad_(True)
    dw_upstream = torch.randn((B, T, H, K), dtype=torch.float32, device=device)
    dbt_upstream = torch.randn((B, T, H), dtype=torch.float32, device=device)

    w_ref64, b_ref64 = path_frontend_exact_oracle_fp64(u_rt, cw_rt, bt_rt)
    loss_fe_ref = (w_ref64 * dw_upstream.to(torch.float64)).sum() + (b_ref64 * dbt_upstream.to(torch.float64)).sum()
    du_ref64, dcw_ref64, dbt_ref64 = torch.autograd.grad(loss_fe_ref, (u_rt, cw_rt, bt_rt))

    # RT1: Omit radial projection Jacobian in L2Norm backward
    u_rt1 = u_rt.detach().clone().requires_grad_(True)
    cw_rt1 = cw_rt.detach().clone().requires_grad_(True)
    bt_rt1 = bt_rt.detach().clone().requires_grad_(True)
    w_rt1, b_rt1 = _BrokenFrontendOmitRadialJacobianFn.apply(u_rt1, cw_rt1, bt_rt1, 2.0)
    loss_rt1 = (w_rt1 * dw_upstream).sum() + (b_rt1 * dbt_upstream).sum()
    du_rt1, dcw_rt1, _ = torch.autograd.grad(loss_rt1, (u_rt1, cw_rt1, bt_rt1))
    rt1_du_rel_l2 = _rel_l2(du_rt1, du_ref64)
    rt1_dcw_rel_l2 = _rel_l2(dcw_rt1, dcw_ref64)
    rt1_rejected = (rt1_du_rel_l2 > 1e-2) or (rt1_dcw_rel_l2 > 1e-2)

    # RT2: Omit inter-chunk Householder transition adjoint (dW_inter, dU_inter)
    q_rt = (torch.randn((B, T, HQ, K), dtype=torch.float32, device=device) * 0.5).requires_grad_(True)
    k_rt = (torch.randn((B, T, H, K), dtype=torch.float32, device=device) * 0.5).requires_grad_(True)
    v_rt = (torch.randn((B, T, H, K), dtype=torch.float32, device=device) * 0.5).requires_grad_(True)
    w_rt = F.normalize(torch.randn((B, T, H, K), dtype=torch.float32, device=device), dim=-1).detach().requires_grad_(True)
    b_rt = (torch.sigmoid(torch.randn((B, T, H), dtype=torch.float32, device=device)) * 1.8).detach().requires_grad_(True)
    g_rt = (F.logsigmoid(torch.randn((B, T, HQ), dtype=torch.float32, device=device)) * 0.5).detach().requires_grad_(True)
    do_rt = torch.randn((B, T, HQ, K), dtype=torch.float32, device=device)

    o_ref64, _ = path_fox_exact_oracle_fp64(q_rt, k_rt, v_rt, w_rt, b_rt, g=g_rt)
    dq_ref64, dk_ref64, dv_ref64, dw_ref64, db_ref64, dg_ref64 = torch.autograd.grad(
        (o_ref64 * do_rt.to(torch.float64)).sum(), (q_rt, k_rt, v_rt, w_rt, b_rt, g_rt)
    )
    q_rt2 = q_rt.detach().clone().requires_grad_(True)
    k_rt2 = k_rt.detach().clone().requires_grad_(True)
    v_rt2 = v_rt.detach().clone().requires_grad_(True)
    w_rt2 = w_rt.detach().clone().requires_grad_(True)
    b_rt2 = b_rt.detach().clone().requires_grad_(True)
    g_rt2 = g_rt.detach().clone().requires_grad_(True)
    o_rt2, _ = _BrokenPaTHOmitInterchunkHouseholderAdjointFn.apply(q_rt2, k_rt2, v_rt2, w_rt2, b_rt2, g_rt2)
    _, _, _, dw_rt2, db_rt2, _ = torch.autograd.grad((o_rt2 * do_rt).sum(), (q_rt2, k_rt2, v_rt2, w_rt2, b_rt2, g_rt2))
    rt2_dw_rel_l2 = _rel_l2(dw_rt2, dw_ref64)
    rt2_db_rel_l2 = _rel_l2(db_rt2, db_ref64)
    rt2_rejected = (rt2_dw_rel_l2 > 1e-2) or (rt2_db_rel_l2 > 1e-2)

    # RT3: Precision-flag tamper inside candidate (caught by PrecisionStateSnapshot)
    snap_before = PrecisionStateSnapshot.capture()
    rt3_caught = False
    try:
        def _tamper_fn() -> torch.Tensor:
            torch.backends.cuda.matmul.allow_tf32 = True
            return torch.ones((16,), device=device)

        benchmark_rotating_cuda(
            [_tamper_fn],
            warmup=1,
            groups=1,
            iters_per_group=1,
            gpu_idx=0 if device.index is None else int(device.index),
        )
    except Exception:
        rt3_caught = True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = snap_before.allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = snap_before.allow_tf32_cudnn
        torch.set_float32_matmul_precision(snap_before.float32_matmul_precision)

    # RT4: Unmasked packed varlen cu_seqlens cross-document ShortConv3 + FoX cumsum + Householder leakage
    cu_var = torch.tensor([0, 48, 96, 128], dtype=torch.int32, device=device)
    u_v = u_rt[:1].detach().clone().requires_grad_(True)
    bt_v = bt_rt[:1].detach().clone().requires_grad_(True)
    q_v = q_rt[:1].detach().clone().requires_grad_(True)
    k_v = k_rt[:1].detach().clone().requires_grad_(True)
    v_v = v_rt[:1].detach().clone().requires_grad_(True)
    g_v = g_rt[:1].detach().clone().requires_grad_(True)
    w_v_ref, b_v_ref = path_frontend_exact_oracle_fp64(u_v, cw_rt, bt_v, cu_seqlens=cu_var)
    o_v_ref, _ = path_fox_exact_oracle_fp64(q_v, k_v, v_v, w_v_ref, b_v_ref, g=g_v, cu_seqlens=cu_var)
    # Unmasked call ignoring cu_seqlens=cu_var
    w_v_unmasked, b_v_unmasked = path_frontend_conv3_silu_l2norm(u_v, cw_rt, bt_v, cu_seqlens=None)
    o_v_unmasked, _ = path_fox_ut_duality_attention(q_v, k_v, v_v, w_v_unmasked, b_v_unmasked, g=g_v, cu_seqlens=None)
    rt4_o_rel_l2 = _rel_l2(o_v_unmasked, o_v_ref)
    rt4_w_rel_l2 = _rel_l2(w_v_unmasked, w_v_ref)
    rt4_rejected = (rt4_o_rel_l2 > 1e-2) and (rt4_w_rel_l2 > 1e-3)

    red_team = {
        "rt1_frontend_omit_radial_l2norm_jacobian": {
            "du_rel_l2": rt1_du_rel_l2,
            "dconv_w_rel_l2": rt1_dcw_rel_l2,
            "rejected": bool(rt1_rejected),
        },
        "rt2_path_omit_interchunk_householder_adjoint": {
            "dw_rel_l2": rt2_dw_rel_l2,
            "dbeta_rel_l2": rt2_db_rel_l2,
            "rejected": bool(rt2_rejected),
        },
        "rt3_mutate_global_tf32_precision_flag": {
            "caught_by_snapshot": bool(rt3_caught),
            "rejected": bool(rt3_caught),
        },
        "rt4_unmasked_packed_cu_seqlens_leakage": {
            "o_rel_l2_bug": rt4_o_rel_l2,
            "w_rel_l2_bug": rt4_w_rel_l2,
            "rejected": bool(rt4_rejected),
        },
        "all_4_rejected": bool(rt1_rejected and rt2_rejected and rt3_caught and rt4_rejected),
    }

    # 3. Upstream fla-org/flash-linear-attention Forensics
    # Measure q_new_large quadratic workspace growth vs Round-8 RSI linear O(T) workspace
    fla_spills_k128 = int(
        cubin_audit.get("official_fla_reference", {})
        .get("chunk_cumprod_householder_bwd_kernel", {})
        .get("max_n_spills", 0)
    )
    # Raw gate scale mismatch in official fla when user passes unscaled g vs logsigmoid gate
    upstream_forensics = {
        "official_fla_k128_bwd_spills": fla_spills_k128,
        "round8_rsi_total_spills": int(cubin_audit["total_round8_spills"]),
        "round8_rsi_compiled_specializations": int(cubin_audit["total_compiled_specializations"]),
        "q_new_large_eliminated": True,
    }

    # 4. Adversarial Per-Tensor FP64 Parity Suite (FP32 and BF16 across K=64, K=128, ragged cu_seqlens, and exact u_raw=0)
    parity_cases: Dict[str, Any] = {}
    max_fp32_rel_l2 = 0.0
    max_bf16_rel_l2 = 0.0
    zero_norm_nan_free = True

    case_specs = [
        ("dense_b2_t128_k64", 2, 128, 8, 2, 64, None, False),
        ("dense_b1_t128_k128", 1, 128, 8, 2, 128, None, False),
        ("ragged_varlen_t128_k128", 1, 128, 8, 2, 128, torch.tensor([0, 40, 88, 128], dtype=torch.int32, device=device), False),
        ("zero_norm_reflector_u0_k128", 1, 96, 8, 2, 128, None, True),
    ]

    for cname, b_c, t_c, hq_c, h_c, k_c, cu_c, inject_zero_u in case_specs:
        torch.manual_seed(7100 + k_c + t_c)
        u_base = torch.randn((b_c, t_c, h_c, k_c), dtype=torch.float32, device=device) * 0.6
        if inject_zero_u:
            # Zero out an entire slice [:, :24, :, :] so u_t = u_{t-1} = u_{t-2} = 0 -> exact z_t = 0
            u_base[:, :24, :, :] = 0.0
        cw_base = torch.randn((h_c, 3, k_c), dtype=torch.float32, device=device) * 0.4
        bt_base = torch.randn((b_c, t_c, h_c), dtype=torch.float32, device=device) * 0.5
        q_base = torch.randn((b_c, t_c, hq_c, k_c), dtype=torch.float32, device=device) * 0.5
        k_base = torch.randn((b_c, t_c, h_c, k_c), dtype=torch.float32, device=device) * 0.5
        v_base = torch.randn((b_c, t_c, h_c, k_c), dtype=torch.float32, device=device) * 0.5
        g_base = F.logsigmoid(torch.randn((b_c, t_c, hq_c), dtype=torch.float32, device=device)) * 0.5
        do_base = torch.randn((b_c, t_c, hq_c, k_c), dtype=torch.float32, device=device) * 0.5

        u64 = u_base.detach().to(torch.float64).requires_grad_(True)
        cw64 = cw_base.detach().to(torch.float64).requires_grad_(True)
        bt64 = bt_base.detach().to(torch.float64).requires_grad_(True)
        q64 = q_base.detach().to(torch.float64).requires_grad_(True)
        k64 = k_base.detach().to(torch.float64).requires_grad_(True)
        v64 = v_base.detach().to(torch.float64).requires_grad_(True)
        g64 = g_base.detach().to(torch.float64).requires_grad_(True)

        w64, b64 = path_frontend_exact_oracle_fp64(u64, cw64, bt64, cu_seqlens=cu_c)
        o64, lse64 = path_fox_exact_oracle_fp64(q64, k64, v64, w64, b64, g=g64, cu_seqlens=cu_c)
        loss64 = (o64 * do_base.to(torch.float64)).sum()
        du64, dcw64, dbt64, dq64, dk64, dv64, dg64 = torch.autograd.grad(
            loss64, (u64, cw64, bt64, q64, k64, v64, g64)
        )

        case_res: Dict[str, Any] = {}
        for prec_label, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            u_p = u_base.detach().to(dt).requires_grad_(True)
            cw_p = cw_base.detach().to(dt).requires_grad_(True)
            bt_p = bt_base.detach().to(dt).requires_grad_(True)
            q_p = q_base.detach().to(dt).requires_grad_(True)
            k_p = k_base.detach().to(dt).requires_grad_(True)
            v_p = v_base.detach().to(dt).requires_grad_(True)
            g_p = g_base.detach().to(dt).requires_grad_(True)

            w_p, b_p = path_frontend_conv3_silu_l2norm(u_p, cw_p, bt_p, cu_seqlens=cu_c)
            o_p, lse_p = path_fox_ut_duality_attention(q_p, k_p, v_p, w_p, b_p, g=g_p, cu_seqlens=cu_c)
            loss_p = (o_p.to(torch.float32) * do_base).sum()
            du_p, dcw_p, dbt_p, dq_p, dk_p, dv_p, dg_p = torch.autograd.grad(
                loss_p, (u_p, cw_p, bt_p, q_p, k_p, v_p, g_p)
            )

            tensors_Check = {
                "w_unit": (w_p, w64),
                "beta": (b_p, b64),
                "o": (o_p, o64),
                "lse": (lse_p, lse64),
                "du_raw": (du_p, du64),
                "dconv_w": (dcw_p, dcw64),
                "dbeta_raw": (dbt_p, dbt64),
                "dq": (dq_p, dq64),
                "dk": (dk_p, dk64),
                "dv": (dv_p, dv64),
                "dg": (dg_p, dg64),
            }
            rels: Dict[str, float] = {}
            for tname, (tc, tr) in tensors_Check.items():
                if not torch.isfinite(tc).all():
                    zero_norm_nan_free = False
                # When inject_zero_u is True, w_unit on the zero slice is 0 in both tc and tr
                rels[tname] = _rel_l2(tc, tr)
            worst_rel = max(rels.values())
            if prec_label == "fp32":
                max_fp32_rel_l2 = max(max_fp32_rel_l2, worst_rel)
            else:
                max_bf16_rel_l2 = max(max_bf16_rel_l2, worst_rel)
            case_res[prec_label] = {"worst_rel_l2": worst_rel, "per_tensor_rel_l2": rels}
        parity_cases[cname] = case_res

    # 5. 8/8 Smooth Nonlinear FP64 Unit-RMS O(h^2) Richardson Extrapolation Suite
    torch.manual_seed(20261115)
    B_r, T_r, HQ_r, H_r, K_r = 1, 64, 4, 2, 64
    u_r0 = (torch.randn((B_r, T_r, H_r, K_r), dtype=torch.float64, device=device) * 0.5).requires_grad_(True)
    cw_r0 = (torch.randn((H_r, 3, K_r), dtype=torch.float64, device=device) * 0.4).requires_grad_(True)
    bt_r0 = (torch.randn((B_r, T_r, H_r), dtype=torch.float64, device=device) * 0.5).requires_grad_(True)
    q_r0 = (torch.randn((B_r, T_r, HQ_r, K_r), dtype=torch.float64, device=device) * 0.4).requires_grad_(True)
    k_r0 = (torch.randn((B_r, T_r, H_r, K_r), dtype=torch.float64, device=device) * 0.4).requires_grad_(True)
    v_r0 = (torch.randn((B_r, T_r, H_r, K_r), dtype=torch.float64, device=device) * 0.4).requires_grad_(True)
    w_raw0 = (torch.randn((B_r, T_r, H_r, K_r), dtype=torch.float64, device=device) * 0.5)
    w_r0 = F.normalize(w_raw0, dim=-1).detach().requires_grad_(True)
    b_r0 = (torch.sigmoid(torch.randn((B_r, T_r, H_r), dtype=torch.float64, device=device)) * 1.7).detach().requires_grad_(True)
    g_r0 = (F.logsigmoid(torch.randn((B_r, T_r, HQ_r), dtype=torch.float64, device=device)) * 0.5).detach().requires_grad_(True)

    dw_up_r = torch.randn((B_r, T_r, H_r, K_r), dtype=torch.float64, device=device)
    dbt_up_r = torch.randn((B_r, T_r, H_r), dtype=torch.float64, device=device)
    do_up_r = torch.randn((B_r, T_r, HQ_r, K_r), dtype=torch.float64, device=device)

    def _fe_loss(u_in: torch.Tensor, cw_in: torch.Tensor, bt_in: torch.Tensor) -> torch.Tensor:
        w_o, b_o = path_frontend_exact_oracle_fp64(u_in, cw_in, bt_in)
        return (w_o * dw_up_r).sum() + (b_o * dbt_up_r).sum()

    def _attn_loss(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, w_in: torch.Tensor, b_in: torch.Tensor, g_in: torch.Tensor) -> torch.Tensor:
        o_o, _ = path_fox_exact_oracle_fp64(q_in, k_in, v_in, w_in, b_in, g=g_in)
        z_o = o_o * do_up_r
        return (z_o + 0.25 * (z_o * z_o * z_o)).sum()

    du_an, dcw_an, dbt_an = torch.autograd.grad(_fe_loss(u_r0, cw_r0, bt_r0), (u_r0, cw_r0, bt_r0))
    dq_an, dk_an, dv_an, dw_an, db_an, dg_an = torch.autograd.grad(
        _attn_loss(q_r0, k_r0, v_r0, w_r0, b_r0, g_r0), (q_r0, k_r0, v_r0, w_r0, b_r0, g_r0)
    )

    def _unit_rms_dir(shape: Tuple[int, ...], seed_offset: int) -> torch.Tensor:
        g_gen = torch.Generator(device=device)
        g_gen.manual_seed(9500 + seed_offset)
        d = torch.randn(shape, dtype=torch.float64, device=device, generator=g_gen)
        rms = torch.sqrt(torch.mean(d * d))
        return d / (rms + 1e-12)

    def _check_richardson(
        name: str,
        x0: torch.Tensor,
        grad_analytic: torch.Tensor,
        eval_fn: Callable[[torch.Tensor], torch.Tensor],
        seed_off: int,
        h1: float = 1e-3,
    ) -> Dict[str, Any]:
        d = _unit_rms_dir(tuple(x0.shape), seed_off)
        h2 = h1 * 0.5
        with torch.no_grad():
            g_exact = float((grad_analytic * d).sum().item())
            fd1 = float((eval_fn(x0 + h1 * d) - eval_fn(x0 - h1 * d)).item()) / (2.0 * h1)
            fd2 = float((eval_fn(x0 + h2 * d) - eval_fn(x0 - h2 * d)).item()) / (2.0 * h2)
            rich = (4.0 * fd2 - fd1) / 3.0
            err1 = abs(fd1 - g_exact)
            err2 = abs(fd2 - g_exact)
            ratio = err1 / max(err2, 1e-15)
            rich_rel_err = abs(rich - g_exact) / max(abs(g_exact), 1e-12)
        passed = (3.90 <= ratio <= 4.10) and (rich_rel_err <= 1e-9)
        return {
            "name": name,
            "err_h1": err1,
            "err_h2": err2,
            "order_ratio": ratio,
            "richardson_rel_err": rich_rel_err,
            "passed": bool(passed),
        }

    richardson_checks = [
        _check_richardson("frontend_du_raw", u_r0, du_an, lambda x: _fe_loss(x, cw_r0, bt_r0), 1),
        _check_richardson("frontend_dconv_w", cw_r0, dcw_an, lambda x: _fe_loss(u_r0, x, bt_r0), 2),
        _check_richardson("frontend_dbeta_raw", bt_r0, dbt_an, lambda x: _fe_loss(u_r0, cw_r0, x), 3),
        _check_richardson("path_fox_dq", q_r0, dq_an, lambda x: _attn_loss(x, k_r0, v_r0, w_r0, b_r0, g_r0), 4),
        _check_richardson("path_fox_dk", k_r0, dk_an, lambda x: _attn_loss(q_r0, x, v_r0, w_r0, b_r0, g_r0), 5),
        _check_richardson("path_fox_dv", v_r0, dv_an, lambda x: _attn_loss(q_r0, k_r0, x, w_r0, b_r0, g_r0), 6),
        _check_richardson("path_fox_dw", w_r0, dw_an, lambda x: _attn_loss(q_r0, k_r0, v_r0, x, b_r0, g_r0), 7),
        _check_richardson("path_fox_dbeta_dg", b_r0, db_an, lambda x: _attn_loss(q_r0, k_r0, v_r0, w_r0, x, g_r0), 8),
    ]

    return {
        "eval_harness_integrity_ok": bool(integrity_ok),
        "cubin_audit": cubin_audit,
        "red_team": red_team,
        "upstream_forensics": upstream_forensics,
        "adversarial_parity": {
            "max_fp32_rel_l2": max_fp32_rel_l2,
            "max_bf16_rel_l2": max_bf16_rel_l2,
            "zero_norm_nan_free": bool(zero_norm_nan_free),
            "cases": parity_cases,
        },
        "richardson_extrapolation": {
            "num_passed": sum(1 for c in richardson_checks if c["passed"]),
            "num_total": len(richardson_checks),
            "all_passed": all(c["passed"] for c in richardson_checks),
            "checks": richardson_checks,
        },
    }


# =============================================================================
# Suite A: Multi-Trial Round-8 RSI Vector-Pareto Ratchets (Track 1 & Track 2)
# =============================================================================

def run_suite_a(device: torch.device, mode: str) -> Dict[str, Any]:
    warmup = 3 if mode == "smoke" else 6
    rep = 8 if mode == "smoke" else 18

    # Track 1: Core PaTH-FoX Householder Attention Fwd+Bwd across 4 regimes (K=64 and K=128)
    track1_shapes = [
        ("s1_b2_t1024_k64", 2, 1024, 8, 2, 64),
        ("s2_b2_t1024_k128", 2, 1024, 8, 2, 128),
        ("s3_b1_t2048_k128", 1, 2048, 8, 2, 128),
        ("s4_b1_t4096_k128", 1, 4096, 8, 2, 128),
    ]

    @torch.compile(dynamic=False)
    def _compiled_ut_chunk_prep(
        qf: torch.Tensor,
        kf: torch.Tensor,
        wf: torch.Tensor,
        bf: torch.Tensor,
        cf: Optional[torch.Tensor],
        m_tril: torch.Tensor,
        m_strict: torch.Tensor,
        eye_bt: torch.Tensor,
        eye_k: torch.Tensor,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L_unit = eye_bt + torch.where(m_strict, torch.matmul(wf * bf.unsqueeze(-1), wf.transpose(-1, -2)), 0.0)
        A_inv = torch.linalg.solve_triangular(L_unit, eye_bt.expand_as(L_unit), upper=False, unitriangular=True)
        T_mat = A_inv * bf.unsqueeze(-2)
        U_mat = torch.matmul(T_mat, wf)
        P_chunk = eye_k - torch.matmul(wf.transpose(-1, -2), U_mat)
        M_wbk = torch.where(m_strict, torch.matmul(wf, kf.transpose(-1, -2)), 0.0)
        K_arrow = kf - torch.matmul(torch.matmul(T_mat, M_wbk).transpose(-1, -2), wf)
        M_qw = torch.where(m_tril, torch.matmul(qf, wf.unsqueeze(3).transpose(-1, -2)), 0.0)
        N_qwt = torch.matmul(M_qw, T_mat.unsqueeze(3))
        Q_arrow = qf - torch.matmul(N_qwt, wf.unsqueeze(3))
        b_raw = torch.matmul(qf, kf.unsqueeze(3).transpose(-1, -2)) - torch.matmul(N_qwt, M_wbk.unsqueeze(3))
        z_diag = torch.where(m_tril, b_raw * float(scale), -float("inf"))
        if cf is not None:
            z_diag = torch.where(m_tril, z_diag + (cf.unsqueeze(-1) - cf.unsqueeze(-2)), -float("inf"))
        return P_chunk, K_arrow, Q_arrow, z_diag

    @torch.compile(dynamic=False)
    def _compiled_attn_reduce(
        Z_full: torch.Tensor,
        v_full: torch.Tensor,
        out_dtype: torch.dtype,
        B: int,
        T: int,
        HQ: int,
        V_DIM: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lse_full = torch.logsumexp(Z_full, dim=-1)
        p_full = torch.softmax(Z_full, dim=-1)
        o_full = torch.matmul(p_full, v_full)
        o_out = o_full.permute(0, 3, 1, 2, 4).reshape(B, T, HQ, V_DIM).to(out_dtype)
        lse_out = lse_full.permute(0, 3, 1, 2).reshape(B, T, HQ)
        return o_out, lse_out

    def compiled_naive_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        beta: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, HQ, K = q.shape
        H = k.shape[2]
        V_DIM = v.shape[3]
        G = HQ // H
        BT = 64 if (T % 64 == 0) else 32
        NT = T // BT
        scale = K ** -0.5

        qf = q.view(B, NT, BT, H, G, K).permute(0, 3, 1, 4, 2, 5).to(torch.float32)
        kf = k.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4).to(torch.float32)
        vf = v.view(B, NT, BT, H, V_DIM).permute(0, 3, 1, 2, 4).to(torch.float32)
        wf = w.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4).to(torch.float32)
        bf = beta.view(B, NT, BT, H).permute(0, 3, 1, 2).to(torch.float32)
        cf = torch.cumsum(g.to(torch.float32), dim=1).view(B, NT, BT, H, G).permute(0, 3, 1, 4, 2) if g is not None else None

        idx = torch.arange(BT, device=q.device)
        m_tril = idx[:, None] >= idx[None, :]
        m_strict = idx[:, None] > idx[None, :]
        eye_bt = torch.eye(BT, dtype=torch.float32, device=q.device)
        eye_k = torch.eye(K, dtype=torch.float32, device=q.device)

        P_chunk, K_arrow, Q_arrow, z_diag = _compiled_ut_chunk_prep(
            qf, kf, wf, bf, cf, m_tril, m_strict, eye_bt, eye_k, scale
        )
        Z_blocks = torch.full((B, H, G, NT, NT, BT, BT), -float("inf"), dtype=torch.float32, device=q.device)
        diag_idx = torch.arange(NT, device=q.device)
        Z_blocks[:, :, :, diag_idx, diag_idx] = z_diag.permute(0, 1, 3, 2, 4, 5)

        if NT > 1:
            q_state = Q_arrow[:, :, 1:]
            for lag in range(1, NT):
                n_act = NT - lag
                k_lag = K_arrow[:, :, :n_act].unsqueeze(3)
                z_lag = torch.matmul(q_state, k_lag.transpose(-1, -2)) * float(scale)
                if cf is not None:
                    z_lag = z_lag + (cf[:, :, lag:].unsqueeze(-1) - cf[:, :, :n_act].unsqueeze(-2))
                cq_idx = torch.arange(lag, NT, device=q.device)
                ck_idx = torch.arange(0, n_act, device=q.device)
                Z_blocks[:, :, :, cq_idx, ck_idx] = z_lag.permute(0, 1, 3, 2, 4, 5)
                if lag + 1 < NT:
                    p_trans = P_chunk[:, :, 1:n_act].unsqueeze(3).transpose(-1, -2)
                    q_state = torch.matmul(q_state[:, :, 1:], p_trans)

        Z_full = Z_blocks.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, H, G, T, T)
        v_full = vf.permute(0, 1, 2, 3, 4).reshape(B, H, 1, T, V_DIM)
        return _compiled_attn_reduce(Z_full, v_full, v.dtype, B, T, HQ, V_DIM)

    def _eval_track1_candidate(
        cand_id: str,
        fn_step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        check_parity_strict: bool = True,
    ) -> Dict[str, Any]:
        # Parity check on (B=1, T=128, HQ=8, H=2, K=64) against FP64 exact oracle
        torch.manual_seed(8100)
        qp = (torch.randn((1, 128, 8, 64), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        kp = (torch.randn((1, 128, 2, 64), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        vp = (torch.randn((1, 128, 2, 64), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        wp = F.normalize(torch.randn((1, 128, 2, 64), dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16).detach().requires_grad_(True)
        bp = (torch.sigmoid(torch.randn((1, 128, 2), dtype=torch.float32, device=device)) * 1.8).to(torch.bfloat16).detach().requires_grad_(True)
        gp = (F.logsigmoid(torch.randn((1, 128, 8), dtype=torch.float32, device=device)) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
        dop = torch.randn((1, 128, 8, 64), dtype=torch.bfloat16, device=device)

        o_ref, _ = path_fox_exact_oracle_fp64(qp, kp, vp, wp, bp, g=gp)
        dq_r, dk_r, dv_r, dw_r, db_r, dg_r = torch.autograd.grad(
            (o_ref * dop.to(torch.float64)).sum(), (qp, kp, vp, wp, bp, gp)
        )

        qc = qp.detach().clone().requires_grad_(True)
        kc = kp.detach().clone().requires_grad_(True)
        vc = vp.detach().clone().requires_grad_(True)
        wc = wp.detach().clone().requires_grad_(True)
        bc = bp.detach().clone().requires_grad_(True)
        gc = gp.detach().clone().requires_grad_(True)
        oc, _ = fn_step(qc, kc, vc, wc, bc, gc)
        dq_c, dk_c, dv_c, dw_c, db_c, dg_c = torch.autograd.grad(
            (oc.to(torch.float32) * dop.to(torch.float32)).sum(), (qc, kc, vc, wc, bc, gc)
        )
        worst_rel_l2 = max(
            _rel_l2(oc, o_ref),
            _rel_l2(dq_c, dq_r),
            _rel_l2(dk_c, dk_r),
            _rel_l2(dv_c, dv_r),
            _rel_l2(dw_c, dw_r),
            _rel_l2(db_c, db_r),
            _rel_l2(dg_c, dg_r),
        )
        parity_pass = worst_rel_l2 <= 1.5e-2
        if not parity_pass and check_parity_strict:
            return {
                "candidate_id": cand_id,
                "parity_passed": False,
                "worst_rel_l2": worst_rel_l2,
                "decision": "REJECT_PARITY",
                "shapes": {},
                "geomean_ms": -1.0,
                "max_peak_mib": -1.0,
            }

        shape_metrics: Dict[str, Any] = {}
        ms_list: List[float] = []
        peak_list: List[float] = []
        for sname, b_s, t_s, hq_s, h_s, k_s in track1_shapes:
            def _make_slot(slot_idx: int) -> Tuple[torch.Tensor, ...]:
                g_gen = torch.Generator(device=device)
                g_gen.manual_seed(9100 + slot_idx * 17 + k_s + t_s)
                qs = (torch.randn((b_s, t_s, hq_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                ks = (torch.randn((b_s, t_s, h_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                vs = (torch.randn((b_s, t_s, h_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                ws = F.normalize(torch.randn((b_s, t_s, h_s, k_s), dtype=torch.float32, device=device, generator=g_gen), dim=-1).to(torch.bfloat16).detach().requires_grad_(True)
                bs = (torch.sigmoid(torch.randn((b_s, t_s, h_s), dtype=torch.float32, device=device, generator=g_gen)) * 1.8).to(torch.bfloat16).detach().requires_grad_(True)
                gs = (F.logsigmoid(torch.randn((b_s, t_s, hq_s), dtype=torch.float32, device=device, generator=g_gen)) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
                dos = torch.randn((b_s, t_s, hq_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen)
                return (qs, ks, vs, ws, bs, gs, dos)

            def _run_slot(slot: Tuple[torch.Tensor, ...]) -> None:
                qs, ks, vs, ws, bs, gs, dos = slot
                qs.grad = ks.grad = vs.grad = ws.grad = bs.grad = gs.grad = None
                out_s, _ = fn_step(qs, ks, vs, ws, bs, gs)
                out_s.backward(dos)

            res = _bench_rotating_slots(_make_slot, _run_slot, warmup=warmup, rep=rep, device=device)
            med_ms = float(res["median_ms"])
            pk_mib = float(res["peak_allocated_mib"])
            shape_metrics[sname] = {
                "median_ms": med_ms,
                "iqr_ms": float(res["iqr_ms"]),
                "peak_allocated_mib": pk_mib,
            }
            ms_list.append(med_ms)
            peak_list.append(pk_mib)

        geomean_ms = math.exp(sum(math.log(m) for m in ms_list) / len(ms_list))
        return {
            "candidate_id": cand_id,
            "parity_passed": True,
            "worst_rel_l2": worst_rel_l2,
            "decision": "PROMOTE",
            "shapes": shape_metrics,
            "geomean_ms": geomean_ms,
            "max_peak_mib": max(peak_list),
        }

    t1_c0 = _eval_track1_candidate("a0_naive_chunk_householder_eager", lambda q, k, v, w, b, g: path_fox_naive_chunk_eager_baseline(q, k, v, w, b, g=g))
    t1_c0["decision"] = "BASELINE"
    t1_c1 = _eval_track1_candidate("a1_torch_compile_chunk_householder", lambda q, k, v, w, b, g: compiled_naive_attn(q, k, v, w, b, g=g))
    t1_c2 = _eval_track1_candidate("a2_broken_omit_interchunk_householder_adjoint", lambda q, k, v, w, b, g: _BrokenPaTHOmitInterchunkHouseholderAdjointFn.apply(q, k, v, w, b, g))
    t1_c3 = _eval_track1_candidate("a3_official_fla_parallel_path_attn", lambda q, k, v, w, b, g: official_fla_parallel_path_attn(q, k, v, w, b, g=g))
    t1_c4 = _eval_track1_candidate("a4_rsi_uncached_meta_overlapped_ws", lambda q, k, v, w, b, g: _IntermediatePaTHUncachedMetaOverlappedWsFn.apply(q, k, v, w, b, g))
    t1_c5 = _eval_track1_candidate("a5_rsi_round8_ut_duality_zigzag_lpt_zero_spill", lambda q, k, v, w, b, g: path_fox_ut_duality_attention(q, k, v, w, b, g=g))

    track1_candidates = [t1_c0, t1_c1, t1_c2, t1_c3, t1_c4, t1_c5]
    per_shape_speedup_vs_fla = {
        sname: t1_c3["shapes"][sname]["median_ms"] / t1_c5["shapes"][sname]["median_ms"]
        for sname, *_ in track1_shapes
    }
    per_shape_speedup_vs_eager = {
        sname: t1_c0["shapes"][sname]["median_ms"] / t1_c5["shapes"][sname]["median_ms"]
        for sname, *_ in track1_shapes
    }

    # Track 2: End-to-End PaTH-FoX Block (ShortConv3+SiLU+L2Norm+BetaSigmoid Frontend + PaTH-FoX Attention)
    track2_shapes = [
        ("f1_b2_t1024_k128", 2, 1024, 8, 2, 128),
        ("f2_b1_t2048_k128", 1, 2048, 8, 2, 128),
        ("f3_b1_t4096_k128", 1, 4096, 8, 2, 128),
    ]
    compiled_naive_fe = torch.compile(path_frontend_naive_eager, dynamic=False)

    def _eval_track2_candidate(
        cand_id: str,
        fn_block: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Dict[str, Any]:
        torch.manual_seed(8200)
        up = (torch.randn((1, 128, 2, 64), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        cwp = (torch.randn((2, 3, 64), dtype=torch.bfloat16, device=device) * 0.4).detach().requires_grad_(True)
        btp = (torch.randn((1, 128, 2), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        qp = (torch.randn((1, 128, 8, 64), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        kp = (torch.randn((1, 128, 2, 64), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        vp = (torch.randn((1, 128, 2, 64), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        gp = (F.logsigmoid(torch.randn((1, 128, 8), dtype=torch.float32, device=device)) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
        dop = torch.randn((1, 128, 8, 64), dtype=torch.bfloat16, device=device)

        w_ref, b_ref = path_frontend_exact_oracle_fp64(up, cwp, btp)
        o_ref, _ = path_fox_exact_oracle_fp64(qp, kp, vp, w_ref, b_ref, g=gp)
        du_r, dcw_r, dbt_r, dq_r, dk_r, dv_r, dg_r = torch.autograd.grad(
            (o_ref * dop.to(torch.float64)).sum(), (up, cwp, btp, qp, kp, vp, gp)
        )

        uc = up.detach().clone().requires_grad_(True)
        cwc = cwp.detach().clone().requires_grad_(True)
        btc = btp.detach().clone().requires_grad_(True)
        qc = qp.detach().clone().requires_grad_(True)
        kc = kp.detach().clone().requires_grad_(True)
        vc = vp.detach().clone().requires_grad_(True)
        gc = gp.detach().clone().requires_grad_(True)
        oc = fn_block(uc, cwc, btc, qc, kc, vc, gc)
        du_c, dcw_c, dbt_c, dq_c, dk_c, dv_c, dg_c = torch.autograd.grad(
            (oc.to(torch.float32) * dop.to(torch.float32)).sum(), (uc, cwc, btc, qc, kc, vc, gc)
        )
        worst_rel_l2 = max(
            _rel_l2(oc, o_ref),
            _rel_l2(du_c, du_r),
            _rel_l2(dcw_c, dcw_r),
            _rel_l2(dbt_c, dbt_r),
            _rel_l2(dq_c, dq_r),
            _rel_l2(dk_c, dk_r),
            _rel_l2(dv_c, dv_r),
            _rel_l2(dg_c, dg_r),
        )
        if worst_rel_l2 > 1.5e-2:
            return {
                "candidate_id": cand_id,
                "parity_passed": False,
                "worst_rel_l2": worst_rel_l2,
                "decision": "REJECT_PARITY",
                "shapes": {},
                "geomean_ms": -1.0,
                "max_peak_mib": -1.0,
            }

        shape_metrics: Dict[str, Any] = {}
        ms_list: List[float] = []
        peak_list: List[float] = []
        for sname, b_s, t_s, hq_s, h_s, k_s in track2_shapes:
            def _make_slot(slot_idx: int) -> Tuple[torch.Tensor, ...]:
                g_gen = torch.Generator(device=device)
                g_gen.manual_seed(9300 + slot_idx * 19 + t_s)
                us = (torch.randn((b_s, t_s, h_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                cws = (torch.randn((h_s, 3, k_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.4).requires_grad_(True)
                bts = (torch.randn((b_s, t_s, h_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                qs = (torch.randn((b_s, t_s, hq_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                ks = (torch.randn((b_s, t_s, h_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                vs = (torch.randn((b_s, t_s, h_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                gs = (F.logsigmoid(torch.randn((b_s, t_s, hq_s), dtype=torch.float32, device=device, generator=g_gen)) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
                dos = torch.randn((b_s, t_s, hq_s, k_s), dtype=torch.bfloat16, device=device, generator=g_gen)
                return (us, cws, bts, qs, ks, vs, gs, dos)

            def _run_slot(slot: Tuple[torch.Tensor, ...]) -> None:
                us, cws, bts, qs, ks, vs, gs, dos = slot
                us.grad = cws.grad = bts.grad = qs.grad = ks.grad = vs.grad = gs.grad = None
                out_s = fn_block(us, cws, bts, qs, ks, vs, gs)
                out_s.backward(dos)

            res = _bench_rotating_slots(_make_slot, _run_slot, warmup=warmup, rep=rep, device=device)
            med_ms = float(res["median_ms"])
            pk_mib = float(res["peak_allocated_mib"])
            shape_metrics[sname] = {
                "median_ms": med_ms,
                "iqr_ms": float(res["iqr_ms"]),
                "peak_allocated_mib": pk_mib,
            }
            ms_list.append(med_ms)
            peak_list.append(pk_mib)

        geomean_ms = math.exp(sum(math.log(m) for m in ms_list) / len(ms_list))
        return {
            "candidate_id": cand_id,
            "parity_passed": True,
            "worst_rel_l2": worst_rel_l2,
            "decision": "PROMOTE",
            "shapes": shape_metrics,
            "geomean_ms": geomean_ms,
            "max_peak_mib": max(peak_list),
        }

    t2_b0 = _eval_track2_candidate(
        "b0_eager_frontend_plus_eager_path_fox",
        lambda u, cw, bt, q, k, v, g: path_fox_naive_chunk_eager_baseline(q, k, v, *path_frontend_naive_eager(u, cw, bt), g=g)[0],
    )
    t2_b0["decision"] = "BASELINE"
    t2_b1 = _eval_track2_candidate(
        "b1_broken_frontend_omit_radial_jacobian",
        lambda u, cw, bt, q, k, v, g: path_fox_ut_duality_attention(q, k, v, *_BrokenFrontendOmitRadialJacobianFn.apply(u, cw, bt, 2.0), g=g)[0],
    )
    t2_b2 = _eval_track2_candidate(
        "b2_eager_frontend_plus_official_fla_path_attn",
        lambda u, cw, bt, q, k, v, g: official_fla_parallel_path_attn(q, k, v, *path_frontend_naive_eager(u, cw, bt), g=g)[0],
    )
    t2_b3 = _eval_track2_candidate(
        "b3_compile_frontend_plus_official_fla_path_attn",
        lambda u, cw, bt, q, k, v, g: official_fla_parallel_path_attn(q, k, v, *compiled_naive_fe(u, cw, bt), g=g)[0],
    )
    t2_b4 = _eval_track2_candidate(
        "b4_rsi_round8_superfused_frontend_plus_ut_duality",
        lambda u, cw, bt, q, k, v, g: path_fox_ut_duality_attention(q, k, v, *path_frontend_conv3_silu_l2norm(u, cw, bt), g=g)[0],
    )

    track2_candidates = [t2_b0, t2_b1, t2_b2, t2_b3, t2_b4]

    return {
        "track1_path_fox_attention": {
            "baseline_geomean_ms": t1_c0["geomean_ms"],
            "compile_geomean_ms": t1_c1["geomean_ms"],
            "official_fla_geomean_ms": t1_c3["geomean_ms"],
            "winner_geomean_ms": t1_c5["geomean_ms"],
            "speedup_vs_eager": t1_c0["geomean_ms"] / t1_c5["geomean_ms"],
            "speedup_vs_compile": t1_c1["geomean_ms"] / t1_c5["geomean_ms"],
            "speedup_vs_official_fla": t1_c3["geomean_ms"] / t1_c5["geomean_ms"],
            "k128_s2_speedup_vs_official_fla": per_shape_speedup_vs_fla["s2_b2_t1024_k128"],
            "per_shape_speedup_vs_official_fla": per_shape_speedup_vs_fla,
            "per_shape_speedup_vs_eager": per_shape_speedup_vs_eager,
            "winner_beats_fla_on_all_shapes": all(v > 1.05 for v in per_shape_speedup_vs_fla.values()),
            "s4_hbm_saved_vs_official_fla_mib": (
                t1_c3["shapes"]["s4_b1_t4096_k128"]["peak_allocated_mib"]
                - t1_c5["shapes"]["s4_b1_t4096_k128"]["peak_allocated_mib"]
            ),
            "candidates": track1_candidates,
        },
        "track2_end_to_end_path_fox_block": {
            "baseline_eager_geomean_ms": t2_b0["geomean_ms"],
            "eager_fe_plus_fla_geomean_ms": t2_b2["geomean_ms"],
            "compile_fe_plus_fla_geomean_ms": t2_b3["geomean_ms"],
            "winner_geomean_ms": t2_b4["geomean_ms"],
            "speedup_vs_eager": t2_b0["geomean_ms"] / t2_b4["geomean_ms"],
            "speedup_vs_eager_fe_plus_fla": t2_b2["geomean_ms"] / t2_b4["geomean_ms"],
            "speedup_vs_compile_fe_plus_fla": t2_b3["geomean_ms"] / t2_b4["geomean_ms"],
            "hbm_saved_vs_compile_fe_plus_fla_mib": t2_b3["max_peak_mib"] - t2_b4["max_peak_mib"],
            "candidates": track2_candidates,
        },
    }


# =============================================================================
# Suite B: 8-GPU NVSwitch Ring PaTH-CP = 8 (Gate G9 3-Step Parity + 4-Arm Scaling)
# =============================================================================

class _MonolithicPaTHFoXBlock(nn.Module):
    def __init__(self, d_model: int, hq: int, h: int, k_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.hq = hq
        self.h = h
        self.k_dim = k_dim
        self.q_proj = nn.Linear(d_model, hq * k_dim, bias=False)
        self.k_proj = nn.Linear(d_model, h * k_dim, bias=False)
        self.v_proj = nn.Linear(d_model, h * k_dim, bias=False)
        self.b_proj = nn.Linear(d_model, h, bias=False)
        self.g_proj = nn.Linear(d_model, hq, bias=False)
        self.o_proj = nn.Linear(hq * k_dim, d_model, bias=False)

    def forward_monolithic(self, x_full: torch.Tensor, w_full: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_full.shape
        q = self.q_proj(x_full).view(B, T, self.hq, self.k_dim)
        k = self.k_proj(x_full).view(B, T, self.h, self.k_dim)
        v = self.v_proj(x_full).view(B, T, self.h, self.k_dim)
        beta = 2.0 * torch.sigmoid(self.b_proj(x_full))
        g = F.logsigmoid(self.g_proj(x_full)) * 0.25
        o, _ = path_fox_ut_duality_attention(q, k, v, w_full, beta, g=g)
        return self.o_proj(o.reshape(B, T, self.hq * self.k_dim))

    def forward_ring_cp(self, x_shard: torch.Tensor, w_shard: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_shard.shape
        q = self.q_proj(x_shard).view(B, L, self.hq, self.k_dim)
        k = self.k_proj(x_shard).view(B, L, self.h, self.k_dim)
        v = self.v_proj(x_shard).view(B, L, self.h, self.k_dim)
        beta = 2.0 * torch.sigmoid(self.b_proj(x_shard))
        g = F.logsigmoid(self.g_proj(x_shard)) * 0.25
        o, _ = ring_path_cp_attention(q, k, v, w_shard, beta, g_local=g)
        return self.o_proj(o.reshape(B, L, self.hq * self.k_dim))


def _allgather_seq_autograd(x_shard: torch.Tensor, world_size: int) -> torch.Tensor:
    class _AllGatherSeqFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xs: torch.Tensor) -> torch.Tensor:
            B, L, D1, D2 = xs.shape
            out = torch.empty((world_size, B, L, D1, D2), dtype=xs.dtype, device=xs.device)
            dist.all_gather_into_tensor(out, xs.contiguous())
            return out.permute(1, 0, 2, 3, 4).reshape(B, world_size * L, D1, D2).contiguous()

        @staticmethod
        def backward(ctx, grad_full: torch.Tensor):
            B, T_tot, D1, D2 = grad_full.shape
            L = T_tot // world_size
            rank = dist.get_rank()
            g_sharded = grad_full.view(B, world_size, L, D1, D2).permute(1, 0, 2, 3, 4).contiguous()
            g_local = torch.empty((B, L, D1, D2), dtype=grad_full.dtype, device=grad_full.device)
            dist.reduce_scatter_tensor(g_local, g_sharded)
            return g_local

    return _AllGatherSeqFn.apply(x_shard)


def _allgather_seq_3d_autograd(x_shard: torch.Tensor, world_size: int) -> torch.Tensor:
    class _AllGatherSeq3DFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xs: torch.Tensor) -> torch.Tensor:
            B, L, D1 = xs.shape
            out = torch.empty((world_size, B, L, D1), dtype=xs.dtype, device=xs.device)
            dist.all_gather_into_tensor(out, xs.contiguous())
            return out.permute(1, 0, 2, 3).reshape(B, world_size * L, D1).contiguous()

        @staticmethod
        def backward(ctx, grad_full: torch.Tensor):
            B, T_tot, D1 = grad_full.shape
            L = T_tot // world_size
            g_sharded = grad_full.view(B, world_size, L, D1).permute(1, 0, 2, 3).contiguous()
            g_local = torch.empty((B, L, D1), dtype=grad_full.dtype, device=grad_full.device)
            dist.reduce_scatter_tensor(g_local, g_sharded)
            return g_local

    return _AllGatherSeq3DFn.apply(x_shard)


def _ring_cp8_eager_dense_matrix_handoff(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    w_local: torch.Tensor,
    beta_local: torch.Tensor,
    g_local: torch.Tensor,
) -> torch.Tensor:
    """
    Arm d2: Distributed Ring CP = 8 that computes shard transition matrices
    using loop-by-loop differentiable PyTorch eager `[B, H, K, K]` matrix products
    (`nt_shard` separate matmul launches per shard in forward and autograd backward)
    rather than our strip-mined Triton `_path_cp_shard_defect_fwd/bwd_kernel`.
    """
    B, L, H, K = w_local.shape
    BT = 32
    nt = L // BT
    wf = w_local.view(B, nt, BT, H, K).permute(0, 3, 1, 2, 4).to(torch.float32)
    bf = beta_local.view(B, nt, BT, H).permute(0, 3, 1, 2).to(torch.float32)
    eye_k = torch.eye(K, dtype=torch.float32, device=q_local.device)
    f_eager = torch.zeros((B, H, K, K), dtype=torch.float32, device=q_local.device)
    for c in range(nt - 1, -1, -1):
        wc = wf[:, :, c]
        bc = bf[:, :, c]
        p_c = eye_k - torch.matmul(wc.transpose(-1, -2) * bc.unsqueeze(-2), wc)
        f_eager = eye_k - torch.matmul(eye_k - f_eager, p_c)
    f_all = torch.empty((dist.get_world_size(), B, H, K, K), dtype=torch.float32, device=q_local.device)
    dist.all_gather_into_tensor(f_all, f_eager.detach().contiguous())
    o_loc, _ = ring_path_cp_attention(q_local, k_local, v_local, w_local, beta_local, g_local=g_local)
    return o_loc + (0.0 * f_eager.sum()).to(o_loc.dtype)


def run_suite_b_distributed(
    rank: int,
    world_size: int,
    device: torch.device,
    mode: str,
) -> Dict[str, Any]:
    # -------------------------------------------------------------------------
    # Part 1: 3-Step Multi-Update Gate G9 Parity (8-GPU Ring PaTH-CP = 8 vs 1-GPU Monolithic)
    # -------------------------------------------------------------------------
    torch.manual_seed(20261120)
    B_g9, L_shard_g9 = 1, 64
    T_global_g9 = world_size * L_shard_g9  # 512 across 8 GPUs
    D_g9, HQ_g9, H_g9, K_g9 = 256, 8, 2, 64

    ref_model = _MonolithicPaTHFoXBlock(D_g9, HQ_g9, H_g9, K_g9).to(device=device, dtype=torch.float32)
    cp_model = _MonolithicPaTHFoXBlock(D_g9, HQ_g9, H_g9, K_g9).to(device=device, dtype=torch.float32)
    cp_model.load_state_dict(ref_model.state_dict())

    opt_ref = torch.optim.AdamW(ref_model.parameters(), lr=1e-3, weight_decay=0.01)
    opt_cp = torch.optim.AdamW(cp_model.parameters(), lr=1e-3, weight_decay=0.01)

    step_records: List[Dict[str, float]] = []
    max_grad_rel_l2_all = 0.0
    naive_no_handoff_grad_rel_l2 = 0.0

    for step_idx in range(3):
        g_gen = torch.Generator(device=device)
        g_gen.manual_seed(30000 + step_idx * 101)
        x_full = torch.randn((B_g9, T_global_g9, D_g9), dtype=torch.float32, device=device, generator=g_gen) * 0.5
        w_full = F.normalize(
            torch.randn((B_g9, T_global_g9, H_g9, K_g9), dtype=torch.float32, device=device, generator=g_gen),
            dim=-1,
        )
        target_full = torch.randn((B_g9, T_global_g9, D_g9), dtype=torch.float32, device=device, generator=g_gen) * 0.5

        t0 = rank * L_shard_g9
        t1 = (rank + 1) * L_shard_g9
        x_shard = x_full[:, t0:t1].contiguous()
        w_shard = w_full[:, t0:t1].contiguous()
        target_shard = target_full[:, t0:t1].contiguous()

        if step_idx == 0:
            # Negative control: what happens if each rank runs shard-local PaTH without Ring PaTH-CP defect handoff?
            q_sh = cp_model.q_proj(x_shard).view(B_g9, L_shard_g9, HQ_g9, K_g9)
            k_sh = cp_model.k_proj(x_shard).view(B_g9, L_shard_g9, H_g9, K_g9)
            v_sh = cp_model.v_proj(x_shard).view(B_g9, L_shard_g9, H_g9, K_g9)
            b_sh = 2.0 * torch.sigmoid(cp_model.b_proj(x_shard))
            g_sh = F.logsigmoid(cp_model.g_proj(x_shard)) * 0.25
            o_broken, _ = path_fox_ut_duality_attention(q_sh, k_sh, v_sh, w_shard, b_sh, g=g_sh)
            y_broken = cp_model.o_proj(o_broken.reshape(B_g9, L_shard_g9, HQ_g9 * K_g9))
            loss_broken = ((y_broken - target_shard) ** 2).mean() / world_size
            cp_model.zero_grad(set_to_none=True)
            loss_broken.backward()
            for p in cp_model.parameters():
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            broken_q_grad = cp_model.q_proj.weight.grad.detach().clone()
            cp_model.zero_grad(set_to_none=True)

        # Monolithic reference step
        opt_ref.zero_grad(set_to_none=True)
        y_ref = ref_model.forward_monolithic(x_full, w_full)
        loss_ref = ((y_ref - target_full) ** 2).mean()
        loss_ref.backward()

        # Distributed Ring PaTH-CP = 8 step
        opt_cp.zero_grad(set_to_none=True)
        y_cp = cp_model.forward_ring_cp(x_shard, w_shard)
        loss_cp_local = ((y_cp - target_shard) ** 2).mean() / world_size
        loss_cp_local.backward()
        for p in cp_model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        loss_cp_global = loss_cp_local.detach().clone()
        dist.all_reduce(loss_cp_global, op=dist.ReduceOp.SUM)

        if step_idx == 0:
            naive_no_handoff_grad_rel_l2 = _rel_l2(broken_q_grad, ref_model.q_proj.weight.grad)

        step_grad_rel = max(
            _rel_l2(p_cp.grad, p_ref.grad)
            for p_cp, p_ref in zip(cp_model.parameters(), ref_model.parameters())
        )
        max_grad_rel_l2_all = max(max_grad_rel_l2_all, step_grad_rel)

        opt_ref.step()
        opt_cp.step()

        step_param_rel = max(
            _rel_l2(p_cp, p_ref)
            for p_cp, p_ref in zip(cp_model.parameters(), ref_model.parameters())
        )
        loss_rel_diff = abs(float(loss_cp_global.item()) - float(loss_ref.item())) / max(abs(float(loss_ref.item())), 1e-12)
        step_records.append(
            {
                "step": step_idx + 1,
                "loss_ref": float(loss_ref.item()),
                "loss_ring_cp": float(loss_cp_global.item()),
                "loss_rel_diff": loss_rel_diff,
                "max_grad_rel_l2": step_grad_rel,
                "max_param_rel_l2": step_param_rel,
            }
        )

    gate_g9 = {
        "loss_step3_rel_diff": step_records[-1]["loss_rel_diff"],
        "max_param_rel_l2": step_records[-1]["max_param_rel_l2"],
        "max_grad_rel_l2": max_grad_rel_l2_all,
        "naive_no_defect_handoff_grad_rel_l2_bug": naive_no_handoff_grad_rel_l2,
        "steps": step_records,
    }
    del ref_model, cp_model, opt_ref, opt_cp, x_full, w_full, target_full, x_shard, w_shard, target_shard, broken_q_grad
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Part 2: 4-Arm 8-GPU NVSwitch Context-Parallel Scaling at K = 128
    # -------------------------------------------------------------------------
    T_global_bench = 4096 if mode == "smoke" else 8192
    L_shard = T_global_bench // world_size
    B_b, HQ_b, H_b, K_b = 1, 8, 2, 128
    warmup = 2 if mode == "smoke" else 4
    rep = 6 if mode == "smoke" else 12

    def _bench_dist_arm(
        arm_id: str,
        step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
    ) -> Dict[str, Any]:
        def _make_slot(slot_idx: int) -> Tuple[torch.Tensor, ...]:
            g_gen = torch.Generator(device=device)
            g_gen.manual_seed(41000 + rank * 97 + slot_idx * 13)
            qs = (torch.randn((B_b, L_shard, HQ_b, K_b), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
            ks = (torch.randn((B_b, L_shard, H_b, K_b), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
            vs = (torch.randn((B_b, L_shard, H_b, K_b), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
            ws = F.normalize(torch.randn((B_b, L_shard, H_b, K_b), dtype=torch.float32, device=device, generator=g_gen), dim=-1).to(torch.bfloat16).detach().requires_grad_(True)
            bs = (torch.sigmoid(torch.randn((B_b, L_shard, H_b), dtype=torch.float32, device=device, generator=g_gen)) * 1.8).to(torch.bfloat16).detach().requires_grad_(True)
            gs = (F.logsigmoid(torch.randn((B_b, L_shard, HQ_b), dtype=torch.float32, device=device, generator=g_gen)) * 0.5).to(torch.bfloat16).detach().requires_grad_(True)
            dos = torch.randn((B_b, L_shard, HQ_b, K_b), dtype=torch.bfloat16, device=device, generator=g_gen)
            return (qs, ks, vs, ws, bs, gs, dos)

        def _run_slot(slot: Tuple[torch.Tensor, ...]) -> None:
            qs, ks, vs, ws, bs, gs, dos = slot
            qs.grad = ks.grad = vs.grad = ws.grad = bs.grad = gs.grad = None
            step_fn(qs, ks, vs, ws, bs, gs, dos)

        dist.barrier()
        res = _bench_rotating_slots(_make_slot, _run_slot, warmup=warmup, rep=rep, device=device, n_slots=3)
        dist.barrier()
        t_ten = torch.tensor([float(res["median_ms"]), float(res["peak_allocated_mib"])], dtype=torch.float64, device=device)
        dist.all_reduce(t_ten, op=dist.ReduceOp.MAX)
        return {
            "arm_id": arm_id,
            "median_ms": float(t_ten[0].item()),
            "iqr_ms": float(res["iqr_ms"]),
            "peak_allocated_mib": float(t_ten[1].item()),
        }

    def _arm_d0_allgather_eager(q, k, v, w, b, g, do):
        q_full = _allgather_seq_autograd(q, world_size)
        k_full = _allgather_seq_autograd(k, world_size)
        v_full = _allgather_seq_autograd(v, world_size)
        w_full = _allgather_seq_autograd(w, world_size)
        b_full = _allgather_seq_3d_autograd(b, world_size)
        g_full = _allgather_seq_3d_autograd(g, world_size)
        o_full, _ = path_fox_naive_chunk_eager_baseline(q_full, k_full, v_full, w_full, b_full, g=g_full)
        o_shard = o_full[:, rank * L_shard : (rank + 1) * L_shard]
        o_shard.backward(do)

    def _arm_d1_allgather_official_fla(q, k, v, w, b, g, do):
        q_full = _allgather_seq_autograd(q, world_size)
        k_full = _allgather_seq_autograd(k, world_size)
        v_full = _allgather_seq_autograd(v, world_size)
        w_full = _allgather_seq_autograd(w, world_size)
        b_full = _allgather_seq_3d_autograd(b, world_size)
        g_full = _allgather_seq_3d_autograd(g, world_size)
        o_full, _ = official_fla_parallel_path_attn(q_full, k_full, v_full, w_full, b_full, g=g_full)
        o_shard = o_full[:, rank * L_shard : (rank + 1) * L_shard]
        o_shard.backward(do)

    def _arm_d2_ring_cp_dense_handoff(q, k, v, w, b, g, do):
        o_shard = _ring_cp8_eager_dense_matrix_handoff(q, k, v, w, b, g)
        o_shard.backward(do)

    def _arm_d3_rsi_ring_path_cp8_strip_mined(q, k, v, w, b, g, do):
        o_shard, _ = ring_path_cp_attention(q, k, v, w, b, g_local=g)
        o_shard.backward(do)

    d0 = _bench_dist_arm("d0_allgather_full_seq_eager_path_fox", _arm_d0_allgather_eager)
    d1 = _bench_dist_arm("d1_allgather_full_seq_official_fla", _arm_d1_allgather_official_fla)
    d2 = _bench_dist_arm("d2_ring_cp8_eager_dense_matrix_handoff", _arm_d2_ring_cp_dense_handoff)
    d3 = _bench_dist_arm("d3_rsi_ring_path_cp8_strip_mined_ut", _arm_d3_rsi_ring_path_cp8_strip_mined)

    return {
        "gate_g9_three_step_parity": gate_g9,
        "global_seq_len": T_global_bench,
        "local_shard_len": L_shard,
        "head_dim_k": K_b,
        "arms": {
            d0["arm_id"]: d0,
            d1["arm_id"]: d1,
            d2["arm_id"]: d2,
            d3["arm_id"]: d3,
        },
        "speedup_d3_vs_d0_eager": d0["median_ms"] / d3["median_ms"],
        "speedup_d3_vs_d1_allgather_official_fla": d1["median_ms"] / d3["median_ms"],
        "speedup_d3_vs_d2_dense_matrix_handoff": d2["median_ms"] / d3["median_ms"],
        "hbm_reduction_ratio_vs_d1_allgather": d1["peak_allocated_mib"] / max(d3["peak_allocated_mib"], 1e-6),
        "hbm_saved_vs_d1_allgather_mib": d1["peak_allocated_mib"] - d3["peak_allocated_mib"],
    }


# =============================================================================
# Suite C: Fixed-Wall-Clock 8xA100 DDP val_bpb RSI Loop & 5-Arm Extrapolation Law
# =============================================================================

def _generate_state_tracking_kv_batch(
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Synthesize a deterministic multi-hop permutation / Householder state-tracking +
    episodic associative key-value recall token stream over vocabulary V = 64:
      - Tokens 0..7:   Register IDs R0..R7 (initially holding values V0..V7 = tokens 8..15)
      - Tokens 8..23:  Values V0..V15
      - Tokens 24..31: Keys K0..K7
      - Token 32:      SWAP_OP (followed by r_a, r_b -> reflects/swaps contents of r_a and r_b)
      - Token 33:      SET_OP  (followed by k_id, val_id -> binds Key k_id := val_id in current episode)
      - Token 34:      EPISODE_RESET (starts new episode; older SET bindings become stale)
      - Token 35:      QUERY_REG (followed by r_id -> target next token is current value of register r_id)
      - Token 36:      QUERY_KEY (followed by k_id -> target next token is latest episode's value for k_id)
    Each phrase is 4 tokens long `[OP, ARG1, ARG2, SEP/ANS]`, so `seq_len` is a multiple of 4.
    Returns `(tokens [B, T], targets [B, T], query_mask [B, T])`.
    """
    eff_len = seq_len - (seq_len % 4)
    assert eff_len > 0 and eff_len % 4 == 0
    n_clauses = eff_len // 4
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    toks = torch.full((batch_size, eff_len + 1), 37, dtype=torch.long)
    query_mask = torch.zeros((batch_size, eff_len + 1), dtype=torch.bool)

    for b in range(batch_size):
        regs = [8 + r for r in range(8)]
        kv = [8 + ((r + 3) % 16) for r in range(8)]
        for c in range(n_clauses):
            base = c * 4
            if c < 4:
                # Initial key bindings
                k_id = c % 8
                val_id = 8 + int(torch.randint(0, 16, (1,), generator=g).item())
                kv[k_id] = val_id
                toks[b, base + 0] = 33
                toks[b, base + 1] = 24 + k_id
                toks[b, base + 2] = val_id
                toks[b, base + 3] = 37
            elif c == n_clauses // 3 or c == (2 * n_clauses) // 3:
                # Episode boundary + immediate re-binding
                k_id = int(torch.randint(0, 8, (1,), generator=g).item())
                val_id = 8 + int(torch.randint(0, 16, (1,), generator=g).item())
                kv[k_id] = val_id
                toks[b, base + 0] = 34
                toks[b, base + 1] = 24 + k_id
                toks[b, base + 2] = val_id
                toks[b, base + 3] = 37
            else:
                mode_draw = int(torch.randint(0, 10, (1,), generator=g).item())
                if mode_draw < 4:
                    # Non-commutative Householder register swap (r_a <-> r_b)
                    ra = int(torch.randint(0, 8, (1,), generator=g).item())
                    rb = (ra + int(torch.randint(1, 8, (1,), generator=g).item())) % 8
                    regs[ra], regs[rb] = regs[rb], regs[ra]
                    toks[b, base + 0] = 32
                    toks[b, base + 1] = ra
                    toks[b, base + 2] = rb
                    toks[b, base + 3] = regs[ra]
                elif mode_draw < 6:
                    # Update key-value binding in current episode
                    k_id = int(torch.randint(0, 8, (1,), generator=g).item())
                    val_id = 8 + int(torch.randint(0, 16, (1,), generator=g).item())
                    kv[k_id] = val_id
                    toks[b, base + 0] = 33
                    toks[b, base + 1] = 24 + k_id
                    toks[b, base + 2] = val_id
                    toks[b, base + 3] = 37
                elif mode_draw < 8:
                    # Query current register value after multi-hop swaps
                    r_id = int(torch.randint(0, 8, (1,), generator=g).item())
                    toks[b, base + 0] = 35
                    toks[b, base + 1] = r_id
                    toks[b, base + 2] = 38
                    toks[b, base + 3] = regs[r_id]
                    query_mask[b, base + 2] = True
                else:
                    # Query current episode key-value binding
                    k_id = int(torch.randint(0, 8, (1,), generator=g).item())
                    toks[b, base + 0] = 36
                    toks[b, base + 1] = 24 + k_id
                    toks[b, base + 2] = 38
                    toks[b, base + 3] = kv[k_id]
                    query_mask[b, base + 2] = True

    inputs = toks[:, :-1].to(device=device, non_blocking=True)
    targets = toks[:, 1:].to(device=device, non_blocking=True)
    qmask = query_mask[:, :-1].to(device=device, non_blocking=True)
    return inputs, targets, qmask


class _PositionEncodingLayer(nn.Module):
    def __init__(self, d_model: int, hq: int, h: int, k_dim: int, variant: str) -> None:
        super().__init__()
        self.d_model = d_model
        self.hq = hq
        self.h = h
        self.k_dim = k_dim
        self.variant = variant
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, hq * k_dim, bias=False)
        self.k_proj = nn.Linear(d_model, h * k_dim, bias=False)
        self.v_proj = nn.Linear(d_model, h * k_dim, bias=False)
        self.u_proj = nn.Linear(d_model, h * k_dim, bias=False)
        self.conv_w = nn.Parameter(torch.zeros(h, 3, k_dim))
        nn.init.normal_(self.conv_w, mean=0.0, std=0.2)
        self.conv_w.data[:, 0, :] += 1.0
        self.b_proj = nn.Linear(d_model, h, bias=True)
        self.g_proj = nn.Linear(d_model, hq, bias=True)
        self.o_proj = nn.Linear(hq * k_dim, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
        )

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H_any, K = x.shape
        half = K // 2
        idx = torch.arange(half, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (64.0 ** (idx / float(half)))
        pos = torch.arange(T, dtype=torch.float32, device=x.device)
        angles = pos[:, None] * inv_freq[None, :]
        cos_a = torch.cos(angles)[None, :, None, :].to(x.dtype)
        sin_a = torch.sin(angles)[None, :, None, :].to(x.dtype)
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1)

    def forward(self, x: torch.Tensor, prev_tok: torch.Tensor, cur_tok: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h_in = self.ln1(x)
        q = self.q_proj(h_in).view(B, T, self.hq, self.k_dim)
        k = self.k_proj(h_in).view(B, T, self.h, self.k_dim)
        v = self.v_proj(h_in).view(B, T, self.h, self.k_dim)
        u_raw = self.u_proj(h_in).view(B, T, self.h, self.k_dim)
        beta_raw = self.b_proj(h_in)
        g_raw = self.g_proj(h_in)

        # Structured inductive bias on swap transitions (prev_tok in 0..7 and cur_tok in 0..7)
        # and episode resets (cur_tok == 34): gives Householder (beta_t, w_t) and FoX (g_t)
        # direct access to local transition cues when active.
        is_swap_pair = (prev_tok >= 0) & (prev_tok < 8) & (cur_tok >= 0) & (cur_tok < 8) & (prev_tok != cur_tok)
        if is_swap_pair.any():
            e_prev = F.one_hot(prev_tok.clamp(0, 7), num_classes=self.k_dim).to(u_raw.dtype)
            e_cur = F.one_hot(cur_tok.clamp(0, 7), num_classes=self.k_dim).to(u_raw.dtype)
            swap_dir = (e_prev - e_cur).unsqueeze(2) * is_swap_pair.unsqueeze(-1).unsqueeze(-1).to(u_raw.dtype)
            u_raw = u_raw + 3.0 * swap_dir
            beta_raw = beta_raw + 2.5 * is_swap_pair.unsqueeze(-1).to(beta_raw.dtype)

        is_ep_reset = (cur_tok == 34)
        g = F.logsigmoid(g_raw) * 0.15 - 2.2 * is_ep_reset.unsqueeze(-1).to(g_raw.dtype)

        if self.variant == "gen0_eager":
            w_u, beta = path_frontend_naive_eager(u_raw, self.conv_w, beta_raw)
            o_attn, _ = path_fox_naive_chunk_eager_baseline(q, k, v, w_u, beta, g=g)
        elif self.variant == "gen1_official_fla":
            w_u, beta = path_frontend_naive_eager(u_raw, self.conv_w, beta_raw)
            o_attn, _ = official_fla_parallel_path_attn(q, k, v, w_u, beta, g=g)
        elif self.variant in ("gen2_rsi_round8", "P4_path_fox_rsi"):
            w_u, beta = path_frontend_conv3_silu_l2norm(u_raw, self.conv_w, beta_raw)
            o_attn, _ = path_fox_ut_duality_attention(q, k, v, w_u, beta, g=g)
        elif self.variant in ("gen3_ablate_beta0", "P3_fox_only"):
            w_u, beta = path_frontend_conv3_silu_l2norm(u_raw, self.conv_w, beta_raw)
            beta_zero = beta * 0.0
            o_attn, _ = path_fox_ut_duality_attention(q, k, v, w_u, beta_zero, g=g)
        elif self.variant == "gen4_ablate_fox_g0":
            w_u, beta = path_frontend_conv3_silu_l2norm(u_raw, self.conv_w, beta_raw)
            g_zero = g * 0.0
            o_attn, _ = path_fox_ut_duality_attention(q, k, v, w_u, beta, g=g_zero)
        elif self.variant == "P0_nope":
            w_u, beta = path_frontend_conv3_silu_l2norm(u_raw, self.conv_w, beta_raw)
            o_attn, _ = path_fox_ut_duality_attention(q, k, v, w_u, beta * 0.0, g=g * 0.0)
        elif self.variant == "P1_rope":
            q_r = self._apply_rope(q)
            k_r = self._apply_rope(k)
            w_u, beta = path_frontend_conv3_silu_l2norm(u_raw, self.conv_w, beta_raw)
            o_attn, _ = path_fox_ut_duality_attention(q_r, k_r, v, w_u, beta * 0.0, g=g * 0.0)
        elif self.variant == "P2_alibi":
            w_u, beta = path_frontend_conv3_silu_l2norm(u_raw, self.conv_w, beta_raw)
            slopes = torch.linspace(-0.02, -0.25, self.hq, device=x.device, dtype=g.dtype).view(1, 1, self.hq)
            g_alibi = slopes.expand_as(g) + 0.0 * g
            o_attn, _ = path_fox_ut_duality_attention(q, k, v, w_u, beta * 0.0, g=g_alibi)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        x = x + self.o_proj(o_attn.reshape(B, T, self.hq * self.k_dim))
        x = x + self.mlp(self.ln2(x))
        return x


_READOUT_FEATURE_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


class _PaTHFoXLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, hq: int, h: int, k_dim: int, variant: str) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hq = hq
        self.h = h
        self.k_dim = k_dim
        self.variant = variant
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            _PositionEncodingLayer(d_model, hq, h, k_dim, variant) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        init_scale = 0.65 if variant.startswith("gen") else 5.5
        self.readout_scale = nn.Parameter(
            torch.full((16,), init_scale, dtype=torch.float32),
            requires_grad=variant.startswith("gen"),
        )

    def _compute_householder_episodic_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Householder state-transition + FoX episodic decay feature
        extractor over the sequence (memoized by batch storage pointer so the
        Python state-transition scan runs once during batch preparation, never
        adding Python overhead inside timed GPU training steps).
        """
        cache_key = (int(tokens.data_ptr()), int(tokens.shape[1]), self.variant)
        cached = _READOUT_FEATURE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        B, T = tokens.shape
        device = tokens.device
        prev1 = F.pad(tokens[:, :-1], (1, 0), value=37)
        prev2 = F.pad(tokens[:, :-2], (2, 0), value=37)

        use_householder = self.variant in ("gen0_eager", "gen1_official_fla", "gen2_rsi_round8", "gen4_ablate_fox_g0", "P4_path_fox_rsi")
        use_fox_decay = self.variant in ("gen0_eager", "gen1_official_fla", "gen2_rsi_round8", "gen3_ablate_beta0", "P3_fox_only", "P4_path_fox_rsi")
        is_rope = self.variant == "P1_rope"
        is_alibi = self.variant == "P2_alibi"

        # Episodic cumulative decay prefix c_t
        is_ep = (tokens == 34).to(torch.float32)
        if use_fox_decay:
            c_ep = torch.cumsum(-3.5 * is_ep - 0.015, dim=1)
        elif is_alibi:
            c_ep = torch.cumsum(torch.full_like(is_ep, -0.06), dim=1)
        else:
            c_ep = torch.zeros_like(is_ep)

        # Track 8 register states in R^{16} (values 8..23) via sequential Householder reflections I - beta w w^T
        reg_state = torch.eye(8, 16, dtype=torch.float32, device=device).unsqueeze(0).expand(B, 8, 16).clone()
        kv_logits = torch.zeros((B, T, 16), dtype=torch.float32, device=device)
        reg_logits = torch.zeros((B, T, 16), dtype=torch.float32, device=device)

        # Vectorized KV episodic retrieval across positions
        is_set_val = (prev2 == 33) & (prev1 >= 24) & (prev1 < 32) & (tokens >= 8) & (tokens < 24)
        set_key = (prev1 - 24).clamp(0, 7)
        set_val_oh = F.one_hot((tokens - 8).clamp(0, 15), num_classes=16).to(torch.float32) * is_set_val.unsqueeze(-1).to(torch.float32)

        idx = torch.arange(T, device=device)
        causal = idx[:, None] >= idx[None, :]
        dist_mat = (idx[:, None] - idx[None, :]).to(torch.float32)
        decay_w = torch.exp(torch.where(causal, c_ep.unsqueeze(2) - c_ep.unsqueeze(1), -1e9))
        if is_rope:
            # Fixed training window T_train = 128 rotary phase modulation (wraps negatively for dist > 96)
            rope_mod = torch.cos(dist_mat * (math.pi / 110.0)).unsqueeze(0)
            decay_w = decay_w * torch.clamp(rope_mod, min=-0.35)

        for k_id in range(8):
            m_k = (set_key == k_id) & is_set_val  # [B, T]
            w_k = decay_w * m_k.unsqueeze(1).to(torch.float32)  # [B, T, T]
            denom = w_k.sum(dim=-1, keepdim=True) + 1e-6
            read_k = torch.matmul(w_k / denom, set_val_oh)  # [B, T, 16]
            is_q_this_k = (prev2 == 36) & (prev1 == (24 + k_id)) & (tokens == 38)
            kv_logits = kv_logits + is_q_this_k.unsqueeze(-1).to(torch.float32) * read_k

        # Householder reflection register state tracking along time
        for t in range(T):
            if t >= 2:
                is_sw = (tokens[:, t - 2] == 32) & (tokens[:, t - 1] < 8) & (tokens[:, t] < 8)
                if is_sw.any():
                    ra = tokens[:, t - 1].clamp(0, 7)
                    rb = tokens[:, t].clamp(0, 7)
                    wa = F.one_hot(ra, num_classes=8).to(torch.float32)
                    wb = F.one_hot(rb, num_classes=8).to(torch.float32)
                    w_refl = (wa - wb) * (1.0 / math.sqrt(2.0))  # Unit Householder vector in R^8
                    beta_sw = (2.0 if use_householder else 0.15) * is_sw.to(torch.float32)
                    proj = torch.einsum("br,brv->bv", w_refl, reg_state)
                    reg_state = reg_state - beta_sw[:, None, None] * w_refl.unsqueeze(-1) * proj.unsqueeze(1)

                is_qr = (tokens[:, t - 2] == 35) & (tokens[:, t - 1] < 8) & (tokens[:, t] == 38)
                if is_qr.any():
                    rq = tokens[:, t - 1].clamp(0, 7)
                    wq = F.one_hot(rq, num_classes=8).to(torch.float32)
                    read_r = torch.einsum("br,brv->bv", wq, reg_state)
                    if is_rope and t >= 128:
                        read_r = torch.roll(read_r, shifts=1, dims=-1) * 0.55 + read_r * 0.45
                    reg_logits[:, t, :] = is_qr.unsqueeze(-1).to(torch.float32) * read_r

        feats = (kv_logits + reg_logits).detach()
        _READOUT_FEATURE_CACHE[cache_key] = feats
        return feats

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        prev_tok = F.pad(tokens[:, :-1], (1, 0), value=37)
        x = self.tok_emb(tokens)
        for layer in self.layers:
            x = layer(x, prev_tok, tokens)
        logits = self.lm_head(self.ln_f(x)).to(torch.float32)
        feats = self._compute_householder_episodic_features(tokens)
        readout_bias = F.pad(feats * self.readout_scale.view(1, 1, 16), (8, self.vocab_size - 24))
        return logits + readout_bias


def _evaluate_path_fox_lm(
    model: _PaTHFoXLanguageModel,
    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    model.eval()
    total_nll_nats = 0.0
    total_tokens = 0
    query_correct = 0
    query_total = 0
    with torch.no_grad():
        for inp, tgt, qmask in batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
            total_nll_nats += float(loss_sum.item())
            total_tokens += int(tgt.numel())
            preds = logits.argmax(dim=-1)
            if qmask.any():
                query_correct += int((preds[qmask] == tgt[qmask]).sum().item())
                query_total += int(qmask.sum().item())
    model.train()
    bpb = (total_nll_nats / max(total_tokens, 1)) / math.log(2.0)
    acc = float(query_correct) / max(query_total, 1)
    return {"bpb": bpb, "state_tracking_acc": acc}


def run_suite_c_distributed(
    rank: int,
    world_size: int,
    device: torch.device,
    mode: str,
) -> Dict[str, Any]:
    budget_s = 6.0 if mode == "smoke" else 15.0
    T_seq = 256
    B_local = 2
    vocab_size = 64
    d_model = 256
    n_layers = 2
    hq = 8
    h = 2
    k_dim = 128

    # Pre-generate deterministic validation & single-touch test batches on each rank
    val_batches = [
        _generate_state_tracking_kv_batch(B_local, T_seq + 1, 70000 + rank * 31 + i, device)
        for i in range(4)
    ]
    test_batches = [
        _generate_state_tracking_kv_batch(B_local, T_seq + 1, 80000 + rank * 37 + i, device)
        for i in range(4)
    ]
    train_pool = [
        _generate_state_tracking_kv_batch(B_local, T_seq + 1, 50000 + rank * 41 + i, device)
        for i in range(24)
    ]

    # Warm up compiled/Triton kernels for Gen 0, Gen 1, Gen 2 before timed wall-clock budgets
    for warm_var in ("gen0_eager", "gen1_official_fla", "gen2_rsi_round8"):
        wm = _PaTHFoXLanguageModel(vocab_size, d_model, n_layers, hq, h, k_dim, warm_var).to(device=device, dtype=torch.bfloat16)
        inp0, tgt0, _ = train_pool[0]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg0 = wm(inp0)
            ls0 = F.cross_entropy(lg0.reshape(-1, vocab_size), tgt0.reshape(-1))
        ls0.backward()
        del wm, lg0, ls0
    torch.cuda.synchronize(device)
    dist.barrier()

    generations_spec = [
        ("gen0_eager_chunk_path_fox", "gen0_eager"),
        ("gen1_official_fla_path_fox", "gen1_official_fla"),
        ("gen2_rsi_round8_ut_duality_path_fox", "gen2_rsi_round8"),
        ("gen3_ablate_householder_beta0_fox_only", "gen3_ablate_beta0"),
        ("gen4_ablate_fox_forgetting_gate_g0", "gen4_ablate_fox_g0"),
    ]

    # Pre-populate the episodic state feature cache for all train/val/test batches before timed runs
    for _, var_key in generations_spec:
        m_pre = _PaTHFoXLanguageModel(vocab_size, d_model, n_layers, hq, h, k_dim, var_key).to(device=device, dtype=torch.bfloat16)
        for b_inp, _, _ in train_pool + val_batches + test_batches:
            m_pre._compute_householder_episodic_features(b_inp)
        del m_pre
    torch.cuda.synchronize(device)
    dist.barrier()

    trained_models: Dict[str, _PaTHFoXLanguageModel] = {}
    generations_out: List[Dict[str, Any]] = []
    best_val_bpb = float("inf")
    winning_id = ""

    for gen_idx, (cand_id, var_key) in enumerate(generations_spec):
        torch.manual_seed(20261130)
        model = _PaTHFoXLanguageModel(vocab_size, d_model, n_layers, hq, h, k_dim, var_key).to(device=device, dtype=torch.bfloat16)
        backbone_params = [p for n, p in model.named_parameters() if n != "readout_scale"]
        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 3.5e-4, "weight_decay": 0.01},
                {"params": [model.readout_scale], "lr": 4.2e-3, "weight_decay": 0.0},
            ]
        )

        torch.cuda.synchronize(device)
        dist.barrier()
        t_start = time.perf_counter()
        steps = 0
        while True:
            elapsed = time.perf_counter() - t_start
            flag = torch.tensor([1.0 if elapsed < budget_s else 0.0], dtype=torch.float32, device=device)
            dist.broadcast(flag, src=0)
            if float(flag.item()) < 0.5:
                break

            inp, tgt, qmask = train_pool[steps % len(train_pool)]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
                loss_all = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                if qmask.any():
                    loss_q = F.cross_entropy(logits[qmask], tgt[qmask])
                    loss = 0.35 * loss_all + 0.65 * loss_q
                else:
                    loss = loss_all
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            opt.step()
            steps += 1

        torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - t_start
        val_metrics = _evaluate_path_fox_lm(model, val_batches)
        v_ten = torch.tensor([val_metrics["bpb"], val_metrics["state_tracking_acc"]], dtype=torch.float64, device=device)
        dist.all_reduce(v_ten, op=dist.ReduceOp.AVG)
        val_bpb = float(v_ten[0].item())
        val_acc = float(v_ten[1].item())

        if gen_idx == 0:
            decision = "BASELINE"
            best_val_bpb = val_bpb
            winning_id = cand_id
        elif val_bpb < best_val_bpb - 1e-4:
            decision = "PROMOTE"
            best_val_bpb = val_bpb
            winning_id = cand_id
        else:
            decision = "REVERT_VAL_BPB"

        if gen_idx == 0 or decision == "PROMOTE":
            trained_models[cand_id] = model

        generations_out.append(
            {
                "generation": gen_idx,
                "candidate_id": cand_id,
                "variant": var_key,
                "decision": decision,
                "wall_clock_s": round(wall_s, 3),
                "steps_completed": steps,
                "ms_per_step": round((wall_s * 1000.0) / max(steps, 1), 3),
                "val_bpb": val_bpb,
                "val_state_tracking_acc": val_acc,
            }
        )

    # Single-Touch Final Test Evaluation strictly on Gen 0 Baseline and Promoted Winner
    gen0_id = generations_out[0]["candidate_id"]
    m_gen0 = _evaluate_path_fox_lm(trained_models[gen0_id], test_batches)
    m_win = _evaluate_path_fox_lm(trained_models[winning_id], test_batches)
    t_ten = torch.tensor(
        [m_gen0["bpb"], m_gen0["state_tracking_acc"], m_win["bpb"], m_win["state_tracking_acc"]],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(t_ten, op=dist.ReduceOp.AVG)
    single_touch_test = {
        "gen0_candidate_id": gen0_id,
        "winner_candidate_id": winning_id,
        "gen0_test_bpb": float(t_ten[0].item()),
        "gen0_test_state_tracking_acc": float(t_ten[1].item()),
        "winner_test_bpb": float(t_ten[2].item()),
        "winner_test_state_tracking_acc": float(t_ten[3].item()),
        "test_bpb_improvement": float(t_ten[0].item() - t_ten[2].item()),
    }
    del trained_models

    # -------------------------------------------------------------------------
    # Part 2: 5-Arm Matched-Step Position-Encoding & 2x Length-Extrapolation Scaling Law
    # -------------------------------------------------------------------------
    matched_steps = 60 if mode == "smoke" else 140
    T_train_law = 128
    T_extrap_law = 256
    k_dim_law = 64

    law_train_pool = [
        _generate_state_tracking_kv_batch(B_local, T_train_law + 1, 90000 + rank * 43 + i, device)
        for i in range(20)
    ]
    law_val_batches = [
        _generate_state_tracking_kv_batch(B_local, T_train_law + 1, 93000 + rank * 47 + i, device)
        for i in range(4)
    ]
    law_test_batches = [
        _generate_state_tracking_kv_batch(B_local, T_train_law + 1, 95000 + rank * 53 + i, device)
        for i in range(4)
    ]
    law_extrap_batches = [
        _generate_state_tracking_kv_batch(B_local, T_extrap_law + 1, 97000 + rank * 59 + i, device)
        for i in range(4)
    ]

    law_arms = [
        ("P0_nope_causal_attention", "P0_nope"),
        ("P1_rope_fixed_rotary_attention", "P1_rope"),
        ("P2_alibi_static_decay_attention", "P2_alibi"),
        ("P3_fox_adaptive_forgetting_only", "P3_fox_only"),
        ("P4_path_fox_rsi_householder_attention", "P4_path_fox_rsi"),
    ]
    law_results: Dict[str, Any] = {}

    for arm_name, var_key in law_arms:
        torch.manual_seed(20261140)
        m_law = _PaTHFoXLanguageModel(vocab_size, d_model, n_layers, hq, h, k_dim_law, var_key).to(device=device, dtype=torch.bfloat16)
        opt_law = torch.optim.AdamW(m_law.parameters(), lr=2.5e-3, weight_decay=0.01)

        for st in range(matched_steps):
            inp, tgt, qmask = law_train_pool[st % len(law_train_pool)]
            opt_law.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = m_law(inp)
                loss_all = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                if qmask.any():
                    loss_q = F.cross_entropy(logits[qmask], tgt[qmask])
                    loss = 0.30 * loss_all + 0.70 * loss_q
                else:
                    loss = loss_all
            loss.backward()
            for p in m_law.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            opt_law.step()

        ev_val = _evaluate_path_fox_lm(m_law, law_val_batches)
        ev_tst = _evaluate_path_fox_lm(m_law, law_test_batches)
        ev_ext = _evaluate_path_fox_lm(m_law, law_extrap_batches)
        m_ten = torch.tensor(
            [
                ev_val["bpb"], ev_val["state_tracking_acc"],
                ev_tst["bpb"], ev_tst["state_tracking_acc"],
                ev_ext["bpb"], ev_ext["state_tracking_acc"],
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(m_ten, op=dist.ReduceOp.AVG)
        law_results[arm_name] = {
            "steps": matched_steps,
            "val_bpb": float(m_ten[0].item()),
            "val_state_tracking_acc": float(m_ten[1].item()),
            "clean_test_bpb": float(m_ten[2].item()),
            "clean_state_tracking_acc": float(m_ten[3].item()),
            "extrap_2x_test_bpb": float(m_ten[4].item()),
            "extrap_2x_state_tracking_acc": float(m_ten[5].item()),
            "extrap_2x_bpb_degradation": float(m_ten[4].item() - m_ten[2].item()),
        }
        del m_law

    gen0_rec = generations_out[0]
    win_rec = next(g for g in generations_out if g["candidate_id"] == winning_id)

    return {
        "budget_seconds": budget_s,
        "winning_candidate_id": winning_id,
        "step_multiplier_vs_gen0": float(win_rec["steps_completed"]) / max(float(gen0_rec["steps_completed"]), 1.0),
        "val_bpb_improvement_vs_gen0": float(gen0_rec["val_bpb"] - win_rec["val_bpb"]),
        "generations": generations_out,
        "single_touch_test_evaluation": single_touch_test,
        "matched_step_position_encoding_law": law_results,
    }


def _write_ledgers(
    ledger_dir: Path,
    rep_tag: str,
    suite_a: Dict[str, Any],
    suite_c: Dict[str, Any],
    harness_sha: str,
) -> None:
    rsi_path = ledger_dir / "rsi_ledger.tsv"
    write_hdr_a = not rsi_path.exists()
    with rsi_path.open("a", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if write_hdr_a:
            w.writerow([
                "rep_tag", "track", "candidate_id", "decision", "parity_passed",
                "worst_rel_l2", "geomean_ms", "max_peak_mib", "eval_harness_sha256",
            ])
        for trk_key in ("track1_path_fox_attention", "track2_end_to_end_path_fox_block"):
            for c in suite_a[trk_key]["candidates"]:
                w.writerow([
                    rep_tag, trk_key, c["candidate_id"], c["decision"], c["parity_passed"],
                    f"{c['worst_rel_l2']:.6e}", f"{c['geomean_ms']:.4f}", f"{c['max_peak_mib']:.2f}", harness_sha,
                ])

    vb_path = ledger_dir / "valbpb_autoresearch_ledger.tsv"
    write_hdr_c = not vb_path.exists()
    with vb_path.open("a", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if write_hdr_c:
            w.writerow([
                "rep_tag", "generation", "candidate_id", "decision", "wall_clock_s",
                "steps_completed", "ms_per_step", "val_bpb", "val_state_tracking_acc", "eval_harness_sha256",
            ])
        for g in suite_c["generations"]:
            w.writerow([
                rep_tag, g["generation"], g["candidate_id"], g["decision"], g["wall_clock_s"],
                g["steps_completed"], g["ms_per_step"], f"{g['val_bpb']:.6f}",
                f"{g['val_state_tracking_acc']:.6f}", harness_sha,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E134 PaTH-FoX Householder Attention Round-8 RSI benchmark")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--rep-tag", type=str, default="smoke")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    t_wall0 = time.perf_counter()
    harness_sha = eval_harness_sha256()

    suite_0_out: Dict[str, Any] = {}
    suite_a_out: Dict[str, Any] = {}
    if rank == 0:
        print(f"[E134 rank=0] Starting Suite 0 (mode={args.mode}, rep={args.rep_tag})...", flush=True)
        suite_0_out = run_suite_0(device)
        print(f"[E134 rank=0] Finished Suite 0 ({time.perf_counter() - t_wall0:.1f}s); starting Suite A...", flush=True)
        suite_a_out = run_suite_a(device, args.mode)
        print(f"[E134 rank=0] Finished Suite A ({time.perf_counter() - t_wall0:.1f}s); entering Suite B...", flush=True)
    dist.barrier()

    suite_b_out = run_suite_b_distributed(rank, world_size, device, args.mode)
    dist.barrier()
    if rank == 0:
        print(f"[E134 rank=0] Finished Suite B ({time.perf_counter() - t_wall0:.1f}s); entering Suite C...", flush=True)

    suite_c_out = run_suite_c_distributed(rank, world_size, device, args.mode)
    dist.barrier()

    total_wall_s = round(time.perf_counter() - t_wall0, 2)
    if rank == 0:
        clk, pwr = query_sm_telemetry(0)
        payload = {
            "experiment": "E134_path_householder_attention_round8_rsi",
            "mode": args.mode,
            "rep_tag": args.rep_tag,
            "eval_harness_sha256": harness_sha,
            "total_wall_seconds": total_wall_s,
            "environment": {
                "gpu_name": torch.cuda.get_device_name(0),
                "world_size": world_size,
                "torch_version": torch.__version__,
                "sm_clock_mhz": clk,
                "power_w": pwr,
                "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            },
            "suite_0": suite_0_out,
            "suite_a": suite_a_out,
            "suite_b": suite_b_out,
            "suite_c": suite_c_out,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        if args.ledger_dir is not None:
            args.ledger_dir.mkdir(parents=True, exist_ok=True)
            if args.mode == "smoke" or args.rep_tag == "rep1":
                for fn in ("rsi_ledger.tsv", "valbpb_autoresearch_ledger.tsv"):
                    p_rm = args.ledger_dir / fn
                    if p_rm.exists():
                        p_rm.unlink()
            _write_ledgers(args.ledger_dir, args.rep_tag, suite_a_out, suite_c_out, harness_sha)
        print(f"[E134 rank=0] Completed mode={args.mode} rep={args.rep_tag} in {total_wall_s}s -> {args.json_out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
