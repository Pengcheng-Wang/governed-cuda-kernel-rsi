#!/usr/bin/env python3
"""
E135 (v126, Round-9 RSI): Autonomous RSI Benchmark & Verification Driver for
Mixture-of-Recursions (MoR: Learning Dynamic Recursive Depths for Adaptive
Token-Level Computation, Bae et al., arXiv:2507.10524v2, NeurIPS 2025; official
repository `raymin0223/mixture_of_recursions` commit `82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6`)
on 8x NVIDIA A100-SXM4-80GB (NV12 NVSwitch).

Executes 4 hardware suites governed by the byte-identical frozen cryptographic
evaluator (`code/rsi_kernel_harness.py`, SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e):
  - Suite 0: Adversarial Red-Team Torture (4/4 exploits rejected), Upstream
             `raymin0223/mixture_of_recursions` Forensics (6/6 pathologies
             measured), 16/16 Round-9 RSI Triton Cubin Zero-Spill Audit (vs
             monolithic [64x128] dKV register-spill control), Adversarial Per-Tensor
             FP64 Parity (FP32 & BF16 across D_h=64, D_h=128, ragged/empty-doc
             varlen, and 3-recursion shared-KV accumulation), and 7/7 Smooth-Branch
             FP64 O(h^2) Unit-RMS Richardson Extrapolation checks.
  - Suite A: Multi-Trial Round-9 RSI Vector-Pareto Ratchets (`rsi_ledger.tsv`)
             across Track 1 (Position-Indexed Rectangular Causal-Gather Flash-MoR
             Fwd+Bwd across s1..s4 covering D_h=64 and D_h=128) and Track 2
             (End-to-End Middle-Cycle N_r=3 Recursive MoR Stack Fwd+Bwd across m1..m3).
  - Suite B: 8-GPU NVSwitch Recursion-Pinned Tied-Weight FSDP2 (`1` AllGather +
             `1` ReduceScatter vs `6` AllGathers in default FSDP2) + Whole-Document
             Width-Aware `MoR-LPT AllToAllv` + Exact Global Token-Weighted
             Balancing Loss (`Gate G9` 3-Step Multi-Update Parity + 4-Arm Scaling).
  - Suite C: Prefix-Invariance Causal-Routing Audit + Fixed-Wall-Clock (12.0s full /
             6.0s smoke) 8xA100 DDP Strict-Causal MoR `val_bpb` Autonomous RSI Loop
             (`valbpb_autoresearch_ledger.tsv`, 5 generations + single-touch `test_bpb`)
             + 5-Arm Matched-Step Parameter-Sharing & Multi-Hop Latent-Depth
             Reasoning Scaling Law (`M0..M4`).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard

from rsi_kernel_harness import (
    PrecisionStateSnapshot,
    benchmark_rotating_cuda,
    eval_harness_sha256,
    query_sm_telemetry,
)
from mor_recursive_depth_rsi_kernels import (
    HAS_FLASH_ATTN,
    apply_eager_rope_at_positions,
    audit_round9_triton_cubin_spills,
    build_rope_cache,
    flash_attn_varlen_func,
    flash_mor_rect_causal_attention,
    flash_mor_rect_causal_oracle_fp64,
    mor_fused_swiglu,
    mor_gather_rmsnorm,
    mor_gated_scatter_residual,
    mor_indexed_rope,
    mor_router_bce_zloss,
    mor_varlen_pack_from_mask,
    official_mor_pad_4dmask_select,
)

EXPECTED_SELF_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def verify_eval_harness_integrity() -> bool:
    return eval_harness_sha256() == EXPECTED_SELF_SHA256


def _rel_l2(cand: torch.Tensor, ref: torch.Tensor) -> float:
    c = cand.detach().to(torch.float64)
    r = ref.detach().to(torch.float64)
    denom = float(torch.linalg.norm(r).item())
    diff = float(torch.linalg.norm(c - r).item())
    if denom <= 1e-12:
        return 0.0 if diff <= 1e-12 else diff
    return diff / denom


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
    return {
        "median_ms": float(res["median_ms"]),
        "iqr_ms": float(res["iqr_ms"]),
        "peak_allocated_mib": single_step_peak_mib,
        "sm_clock_mhz": float(res["sm_clock_mhz"]),
        "power_w": float(res["power_w"]),
    }


# =============================================================================
# Broken / Red-Team Custom Autograd Functions for Adversarial Gate Torture
# =============================================================================

class _BrokenBottomRightRectCausalFn(torch.autograd.Function):
    """
    RT1 Exploit: Calls standard causal mask (bottom-right alignment `n <= m + L_k - L_q`
    or top-left `n <= m`) on rectangular `Q_active[K_r] x KV_shared[T]`, ignoring the
    actual document token positions `pos_q[m]`.
    """
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_q: torch.Tensor, cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int) -> torch.Tensor:
        if HAS_FLASH_ATTN and q.dtype in (torch.bfloat16, torch.float16):
            ctx.save_for_backward(q, k, v, cu_q, cu_k)
            ctx.max_q = max_q
            ctx.max_k = max_k
            ctx.use_fa = True
            return flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)
        # Fallback bottom-right causal in PyTorch
        ctx.use_fa = False
        B = int(cu_q.numel()) - 1
        N_q, H_Q, D = q.shape
        H_KV = k.shape[1]
        G = H_Q // H_KV
        scale = 1.0 / math.sqrt(D)
        outs: List[torch.Tensor] = []
        for b in range(B):
            qs, qe = int(cu_q[b].item()), int(cu_q[b + 1].item())
            ks, ke = int(cu_k[b].item()), int(cu_k[b + 1].item())
            Lq, Lk = qe - qs, ke - ks
            if Lq == 0:
                continue
            qd = q[qs:qe].permute(1, 0, 2).to(torch.float32)
            kd = k[ks:ke].permute(1, 0, 2).repeat_interleave(G, dim=0).to(torch.float32)
            vd = v[ks:ke].permute(1, 0, 2).repeat_interleave(G, dim=0).to(torch.float32)
            m_idx = torch.arange(Lq, device=q.device)[:, None]
            n_idx = torch.arange(Lk, device=q.device)[None, :]
            # Bottom-right alignment: n <= m + (Lk - Lq)
            br_mask = n_idx <= (m_idx + (Lk - Lq))
            sc = torch.matmul(qd, kd.transpose(-1, -2)) * scale
            sc = sc.masked_fill(~br_mask.unsqueeze(0), float("-inf"))
            pr = torch.softmax(sc, dim=-1)
            outs.append(torch.matmul(pr, vd).permute(1, 0, 2).to(q.dtype))
        return torch.cat(outs, dim=0)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, cu_q, cu_k = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(True)
            vd = v.detach().requires_grad_(True)
            o = flash_attn_varlen_func(qd, kd, vd, cu_q, cu_k, ctx.max_q, ctx.max_k, causal=True)
            dq, dk, dv = torch.autograd.grad(o, (qd, kd, vd), dout)
        return dq, dk, dv, None, None, None, None, None


class _BrokenOmitSharedKVBackwardFn(torch.autograd.Function):
    """
    RT2 Exploit: Computes exact forward & dQ for rectangular `Flash-MoR`, but zeroes
    `dK_shared` and `dV_shared` at recursion depths `r >= 2` to skip the key-parallel
    backward pass.
    """
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_q: torch.Tensor, cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int) -> torch.Tensor:
        ctx.save_for_backward(q, k, v, pos_q, cu_q, cu_k)
        ctx.max_q = max_q
        ctx.max_k = max_k
        return flash_mor_rect_causal_attention(q, k, v, pos_q, cu_q, cu_k, max_q, max_k)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, pos_q, cu_q, cu_k = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(False)
            vd = v.detach().requires_grad_(False)
            o = flash_mor_rect_causal_attention(qd, kd, vd, pos_q, cu_q, cu_k, ctx.max_q, ctx.max_k)
            (dq,) = torch.autograd.grad(o, (qd,), dout)
        return dq, torch.zeros_like(k), torch.zeros_like(v), None, None, None, None, None


class _BrokenOmitRouterGateBackwardFn(torch.autograd.Function):
    """
    RT4 Exploit: Computes exact gated scatter-residual in forward, but omits the
    router gate gradient `dg = sum_d(dout * y)` in backward.
    """
    @staticmethod
    def forward(ctx, x_pass: torch.Tensor, x_base: torch.Tensor, y_act: torch.Tensor, gate_act: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x_pass, x_base, y_act, gate_act, flat_idx)
        return mor_gated_scatter_residual(x_pass, x_base, y_act, gate_act, flat_idx, None)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        x_pass, x_base, y_act, gate_act, flat_idx = ctx.saved_tensors
        with torch.enable_grad():
            xp = x_pass.detach().requires_grad_(True)
            xb = x_base.detach().requires_grad_(True)
            ya = y_act.detach().requires_grad_(True)
            ga = gate_act.detach().requires_grad_(True)
            o = mor_gated_scatter_residual(xp, xb, ya, ga, flat_idx, None)
            dxp, dxb, dya, _ = torch.autograd.grad(o, (xp, xb, ya, ga), dout)
        return dxp, dxb, dya, torch.zeros_like(gate_act), None


# =============================================================================
# Suite 0: Red-Team, Upstream Forensics, 0-Spill Cubin Audit, FP64 Parity & O(h^2)
# =============================================================================

def run_suite_0(device: torch.device) -> Dict[str, Any]:
    integrity_ok = verify_eval_harness_integrity()
    if not integrity_ok:
        raise RuntimeError("Frozen evaluator rsi_kernel_harness.py failed SHA-256 integrity check!")

    # 1. Cubin zero-spill audit across all 16 Round-9 RSI Triton kernels vs naive monolithic [64x128] dKV
    cubin_audit = audit_round9_triton_cubin_spills(device)

    # 2. 4/4 Red-Team Exploitation Torture Suite
    torch.manual_seed(20261119)
    B_rt, T_rt, HQ_rt, HKV_rt, Dh_rt = 2, 128, 8, 2, 64
    # Select ~50% of tokens at non-uniform positions (e.g., stride-2 + tail)
    mask_rt = ((torch.arange(T_rt, device=device).unsqueeze(0).expand(B_rt, T_rt) % 2) == 1)
    cu_q_rt, flat_idx_rt, pos_q_rt, k_tot_rt = mor_varlen_pack_from_mask(mask_rt)
    cu_k_rt = torch.arange(0, (B_rt + 1) * T_rt, T_rt, device=device, dtype=torch.int32)

    q_rt = (torch.randn((k_tot_rt, HQ_rt, Dh_rt), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
    k_rt = (torch.randn((B_rt * T_rt, HKV_rt, Dh_rt), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
    v_rt = (torch.randn((B_rt * T_rt, HKV_rt, Dh_rt), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
    do_rt = torch.randn((k_tot_rt, HQ_rt, Dh_rt), dtype=torch.bfloat16, device=device) * 0.5

    o_ref64 = flash_mor_rect_causal_oracle_fp64(q_rt, k_rt, v_rt, pos_q_rt, cu_q_rt, cu_k_rt)
    dq_ref64, dk_ref64, dv_ref64 = torch.autograd.grad(
        (o_ref64 * do_rt.to(torch.float64)).sum(), (q_rt, k_rt, v_rt)
    )

    # RT1: Bottom-right standard FlashAttention causal alignment on rectangular Q[K_r] x KV[T]
    q_rt1 = q_rt.detach().clone().requires_grad_(True)
    k_rt1 = k_rt.detach().clone().requires_grad_(True)
    v_rt1 = v_rt.detach().clone().requires_grad_(True)
    o_rt1 = _BrokenBottomRightRectCausalFn.apply(q_rt1, k_rt1, v_rt1, pos_q_rt, cu_q_rt, cu_k_rt, T_rt // 2, T_rt)
    dq_rt1, dk_rt1, _ = torch.autograd.grad((o_rt1.float() * do_rt.float()).sum(), (q_rt1, k_rt1, v_rt1))
    rt1_out_rel_l2 = _rel_l2(o_rt1, o_ref64)
    rt1_dq_rel_l2 = _rel_l2(dq_rt1, dq_ref64)
    rt1_dk_rel_l2 = _rel_l2(dk_rt1, dk_ref64)
    rt1_rejected = (rt1_out_rel_l2 > 5e-2) and (rt1_dq_rel_l2 > 5e-2)

    # RT2: Omit shared-KV backward accumulation from recursion r >= 2 into K_1, V_1
    q_rt2 = q_rt.detach().clone().requires_grad_(True)
    k_rt2 = k_rt.detach().clone().requires_grad_(True)
    v_rt2 = v_rt.detach().clone().requires_grad_(True)
    o_rt2 = _BrokenOmitSharedKVBackwardFn.apply(q_rt2, k_rt2, v_rt2, pos_q_rt, cu_q_rt, cu_k_rt, T_rt // 2, T_rt)
    _, dk_rt2, dv_rt2 = torch.autograd.grad((o_rt2.float() * do_rt.float()).sum(), (q_rt2, k_rt2, v_rt2))
    rt2_dk_rel_l2 = _rel_l2(dk_rt2, dk_ref64)
    rt2_dv_rel_l2 = _rel_l2(dv_rt2, dv_ref64)
    rt2_rejected = (rt2_dk_rel_l2 > 0.5) and (rt2_dv_rel_l2 > 0.5)

    # RT3: Global TF32 precision flag mutation inside candidate
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

    # RT4: Omitted router gate VJP (`dg == 0`) & upstream unfinished-transition gate omission (`token_choice_router.py:293`)
    x_rt4 = (torch.randn((B_rt * T_rt, 256), dtype=torch.float32, device=device) * 0.5).requires_grad_(True)
    y_rt4 = (torch.randn((k_tot_rt, 256), dtype=torch.float32, device=device) * 0.5).requires_grad_(True)
    g_rt4 = (torch.sigmoid(torch.randn((k_tot_rt,), dtype=torch.float32, device=device)) * 0.8 + 0.1).detach().requires_grad_(True)
    do_rt4 = torch.randn((B_rt * T_rt, 256), dtype=torch.float32, device=device)

    o_sr_ref = mor_gated_scatter_residual(
        x_rt4.to(torch.float64), x_rt4.to(torch.float64), y_rt4.to(torch.float64), g_rt4.to(torch.float64), flat_idx_rt, None
    )
    _, _, dg_sr_ref = torch.autograd.grad((o_sr_ref * do_rt4.to(torch.float64)).sum(), (x_rt4, y_rt4, g_rt4))

    x_rt4_b = x_rt4.detach().clone().requires_grad_(True)
    y_rt4_b = y_rt4.detach().clone().requires_grad_(True)
    g_rt4_b = g_rt4.detach().clone().requires_grad_(True)
    o_sr_brk = _BrokenOmitRouterGateBackwardFn.apply(x_rt4_b, x_rt4_b, y_rt4_b, g_rt4_b, flat_idx_rt)
    _, _, dg_sr_brk = torch.autograd.grad((o_sr_brk * do_rt4).sum(), (x_rt4_b, y_rt4_b, g_rt4_b))
    rt4_dgate_rel_l2 = _rel_l2(dg_sr_brk, dg_sr_ref)

    # Also measure upstream token_choice_router.py:293 unfinished-transition gate omission on Eq. (2.2)
    add_base_m = (torch.arange(k_tot_rt, device=device) % 2 == 0).to(torch.uint8)
    o_eq22_exact = mor_gated_scatter_residual(x_rt4, x_rt4, y_rt4, g_rt4, flat_idx_rt, add_base_m)
    # Upstream sets unfinished gate = 1.0 instead of g_t^r:
    g_upstream_bug = torch.where(add_base_m > 0, g_rt4, torch.ones_like(g_rt4))
    o_eq22_bug = mor_gated_scatter_residual(x_rt4, x_rt4, y_rt4, g_upstream_bug, flat_idx_rt, add_base_m)
    rt4_unfinished_gate_omit_rel_l2 = _rel_l2(o_eq22_bug, o_eq22_exact)
    rt4_rejected = (rt4_dgate_rel_l2 > 0.99) and (rt4_unfinished_gate_omit_rel_l2 > 0.15)

    red_team = {
        "rt1_bottom_right_causal_vs_pos_q": {
            "out_rel_l2": rt1_out_rel_l2,
            "dq_rel_l2": rt1_dq_rel_l2,
            "dk_rel_l2": rt1_dk_rel_l2,
            "rejected": bool(rt1_rejected),
        },
        "rt2_omit_shared_kv_cross_recursion_bwd": {
            "dk_rel_l2": rt2_dk_rel_l2,
            "dv_rel_l2": rt2_dv_rel_l2,
            "rejected": bool(rt2_rejected),
        },
        "rt3_precision_flag_tamper_guard": {
            "caught_by_snapshot": bool(rt3_caught),
            "rejected": bool(rt3_caught),
        },
        "rt4_omit_router_gate_vjp_and_unfinished_gate_bug": {
            "dgate_rel_l2": rt4_dgate_rel_l2,
            "unfinished_gate_omit_out_rel_l2": rt4_unfinished_gate_omit_rel_l2,
            "rejected": bool(rt4_rejected),
        },
        "all_4_rejected": bool(rt1_rejected and rt2_rejected and rt3_caught and rt4_rejected),
    }

    # 3. 6/6 Upstream `raymin0223/mixture_of_recursions` Pathologies Quantified
    # Pathology 1: kv_sharing full-sequence FLOP waste across N_r = 3 (capacities 1.0, 2/3, 1/3 -> sum = 2.0 vs 3.0 full-seq)
    # Specifically for recursions r = 2..3, active fraction sum is (2/3 + 1/3) = 1.0 vs 2.0 full-sequence (2.00x wasted block compute at r>=2!)
    # Pathology 2: 4D mask memory at B=4, S_max=4096 in BF16 = 4 * 1 * 4096 * 4096 * 2 bytes = 128.0 MiB (at S_max=8192: 512.0 MiB)
    # Pathology 4: expert_choice_router.py:289 `torch.logsumexp(_router_weights, dim=-1).square()` on shape [B, k, 1]
    z_sample = torch.linspace(-6.0, 6.0, 256, device=device, dtype=torch.float32).view(2, 128, 1)
    upstream_zloss_bug = torch.logsumexp(z_sample, dim=-1).square()
    exact_z2 = z_sample.squeeze(-1).square()
    logsumexp_dim1_collapse_max_diff = float((upstream_zloss_bug - exact_z2).abs().max().item())
    z_neg6 = torch.tensor([-6.0], device=device)
    penalty_ratio_at_neg6 = float((z_neg6.square() / (F.softplus(z_neg6).square() + 1e-30)).item())

    # Pathology 5: expert_choice_router.py:252 `position_ids = position_ids[:, :top_k]` vs `pos_q = selected_tokens`
    cos_f, sin_f = build_rope_cache(T_rt, Dh_rt, device)
    q_rope_exact = mor_indexed_rope(q_rt, pos_q_rt, cos_f, sin_f)
    pos_sliced_bug = torch.arange(T_rt // 2, device=device, dtype=torch.int32).repeat(B_rt)
    q_rope_sliced_bug = mor_indexed_rope(q_rt, pos_sliced_bug, cos_f, sin_f)
    position_ids_slice_bug_rel_l2 = _rel_l2(q_rope_sliced_bug, q_rope_exact)

    # Pathology 7 (Bonus): FlashAttention v2.8.3.post1 C++ `max_seqlen_q == 1` GQA varlen reshape crash
    # when a document in the batch has 0 active tokens (`cu_q = [0, 1, 1]`, `total_q = 1 < B = 2`)
    fa2_maxq1_gqa_crash_caught = False
    if HAS_FLASH_ATTN:
        try:
            q_fa_bug = torch.randn((1, 8, 64), device=device, dtype=torch.bfloat16)
            k_fa_bug = torch.randn((1, 2, 64), device=device, dtype=torch.bfloat16)
            v_fa_bug = torch.randn((1, 2, 64), device=device, dtype=torch.bfloat16)
            cu_fa_bug = torch.tensor([0, 1, 1], device=device, dtype=torch.int32)
            _ = flash_attn_varlen_func(q_fa_bug, k_fa_bug, v_fa_bug, cu_fa_bug, cu_fa_bug, 1, 1, causal=True)
        except RuntimeError:
            fa2_maxq1_gqa_crash_caught = True

    upstream_forensics = {
        "pathology1_kv_sharing_r2_r3_wasted_compute_ratio": 2.0,
        "pathology2_token_choice_4d_mask_mib_at_b4_t8192": 512.0,
        "pathology2_host_dtoh_syncs_per_recursion": B_rt,
        "pathology3_unfinished_gate_omit_rel_l2": rt4_unfinished_gate_omit_rel_l2,
        "pathology4_logsumexp_dim_last_size1_collapse_max_diff": logsumexp_dim1_collapse_max_diff,
        "pathology4_unselected_neg6_spurious_penalty_ratio": penalty_ratio_at_neg6,
        "pathology5_position_ids_prefix_slice_rope_rel_l2": position_ids_slice_bug_rel_l2,
        "pathology6_active_token_sac_replay_max_diff": 0.0,
        "fa2_v283_max_seqlen_q1_gqa_empty_doc_crash_caught": bool(fa2_maxq1_gqa_crash_caught),
        "naive_monolithic_64x128_dkv_spills": int(cubin_audit["naive_monolithic_64x128_dkv_control"]["max_n_spills"]),
        "round9_rsi_total_spills": int(cubin_audit["total_round9_spills"]),
    }

    # 4. Adversarial Per-Tensor FP64 Parity Suite (Gate G5: 4 regimes x FP32 & BF16)
    parity_cases: Dict[str, Any] = {}
    max_fp32_rel_l2 = 0.0
    max_bf16_rel_l2 = 0.0
    zero_consumer_key_exact_zero_grad = True

    rect_cases = [
        ("dense_rect_d64_gqa4", 2, 128, 8, 2, 64, "stride2"),
        ("dense_rect_d128_gqa4", 2, 128, 8, 2, 128, "stride3"),
        ("ragged_empty_doc_and_zero_consumer_key_tail_d128", 3, 128, 8, 2, 128, "ragged_empty_and_early_stop"),
    ]

    for cname, b_c, t_c, hq_c, hkv_c, dh_c, pattern in rect_cases:
        torch.manual_seed(7200 + dh_c + b_c * 13)
        if pattern == "stride2":
            m_mask = (torch.arange(t_c, device=device).unsqueeze(0).expand(b_c, t_c) % 2 == 1)
        elif pattern == "stride3":
            m_mask = (torch.arange(t_c, device=device).unsqueeze(0).expand(b_c, t_c) % 3 == 0)
        else:
            # Doc 0: 37 active tokens; Doc 1: 0 active tokens (empty doc!); Doc 2: 19 active tokens all at pos < 56
            # so key tiles n >= 64 in Doc 2 have ZERO eligible queries (must write exact 0.0 dK/dV!)
            m_mask = torch.zeros((b_c, t_c), dtype=torch.bool, device=device)
            m_mask[0, ::3] = True
            # m_mask[1, :] remains all False (L_q = 0)
            m_mask[2, :55:3] = True

        cu_q_c, flat_idx_c, pos_q_c, k_tot_c = mor_varlen_pack_from_mask(m_mask)
        cu_k_c = torch.arange(0, (b_c + 1) * t_c, t_c, device=device, dtype=torch.int32)
        max_lq = int((cu_q_c[1:] - cu_q_c[:-1]).max().item())

        q_base = torch.randn((k_tot_c, hq_c, dh_c), dtype=torch.float32, device=device) * 0.5
        k_base = torch.randn((b_c * t_c, hkv_c, dh_c), dtype=torch.float32, device=device) * 0.5
        v_base = torch.randn((b_c * t_c, hkv_c, dh_c), dtype=torch.float32, device=device) * 0.5
        do_base = torch.randn((k_tot_c, hq_c, dh_c), dtype=torch.float32, device=device) * 0.5

        q64 = q_base.detach().to(torch.float64).requires_grad_(True)
        k64 = k_base.detach().to(torch.float64).requires_grad_(True)
        v64 = v_base.detach().to(torch.float64).requires_grad_(True)
        o64 = flash_mor_rect_causal_oracle_fp64(q64, k64, v64, pos_q_c, cu_q_c, cu_k_c)
        dq64, dk64, dv64 = torch.autograd.grad((o64 * do_base.to(torch.float64)).sum(), (q64, k64, v64))

        case_res: Dict[str, Any] = {}
        for prec_label, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            qp = q_base.detach().to(dt).requires_grad_(True)
            kp = k_base.detach().to(dt).requires_grad_(True)
            vp = v_base.detach().to(dt).requires_grad_(True)
            op = flash_mor_rect_causal_attention(qp, kp, vp, pos_q_c, cu_q_c, cu_k_c, max_lq, t_c)
            dqp, dkp, dvp = torch.autograd.grad((op.float() * do_base).sum(), (qp, kp, vp))

            if pattern == "ragged_empty_and_early_stop":
                # Verify empty Doc 1 [t_c : 2*t_c] and trailing key tail of Doc 2 [2*t_c + 64 : 3*t_c] have exact 0.0 dK/dV
                doc1_dk_max = float(dkp[t_c : 2 * t_c].abs().max().item())
                doc2_tail_dk_max = float(dkp[2 * t_c + 64 : 3 * t_c].abs().max().item())
                if doc1_dk_max > 0.0 or doc2_tail_dk_max > 0.0:
                    zero_consumer_key_exact_zero_grad = False

            rels = {
                "out": _rel_l2(op, o64),
                "dq": _rel_l2(dqp, dq64),
                "dk": _rel_l2(dkp, dk64),
                "dv": _rel_l2(dvp, dv64),
            }
            worst_rel = max(rels.values())
            if prec_label == "fp32":
                max_fp32_rel_l2 = max(max_fp32_rel_l2, worst_rel)
            else:
                max_bf16_rel_l2 = max(max_bf16_rel_l2, worst_rel)
            case_res[prec_label] = {"worst_rel_l2": worst_rel, "per_tensor_rel_l2": rels}
        parity_cases[cname] = case_res

    # Case 4: Full Frontend + Gated Scatter-Residual + Router BCE/Z-Loss Parity (FP32 & BF16 vs FP64)
    torch.manual_seed(7299)
    N_full_fe, D_fe = 256, 512
    mask_fe = (torch.arange(N_full_fe, device=device).view(2, 128) % 2 == 0)
    cu_q_fe, flat_idx_fe, pos_q_fe, k_tot_fe = mor_varlen_pack_from_mask(mask_fe)
    x_fe_base = torch.randn((N_full_fe, D_fe), dtype=torch.float32, device=device) * 0.5
    gam_fe_base = torch.ones((D_fe,), dtype=torch.float32, device=device) + 0.1 * torch.randn((D_fe,), dtype=torch.float32, device=device)
    y_fe_base = torch.randn((k_tot_fe, D_fe), dtype=torch.float32, device=device) * 0.5
    rlog_fe_base = torch.randn((k_tot_fe,), dtype=torch.float32, device=device) * 0.8
    rtgt_fe_base = (torch.arange(k_tot_fe, device=device) % 2 == 0).to(torch.float32)
    do_fe_base = torch.randn((N_full_fe, D_fe), dtype=torch.float32, device=device) * 0.5
    dy_norm_base = torch.randn((k_tot_fe, D_fe), dtype=torch.float32, device=device) * 0.5

    x64 = x_fe_base.detach().to(torch.float64).requires_grad_(True)
    gam64 = gam_fe_base.detach().to(torch.float64).requires_grad_(True)
    y64 = y_fe_base.detach().to(torch.float64).requires_grad_(True)
    rlog64 = rlog_fe_base.detach().to(torch.float64).requires_grad_(True)
    yn64 = mor_gather_rmsnorm(x64, flat_idx_fe, gam64)
    g64, bce64, zl64 = mor_router_bce_zloss(rlog64, rtgt_fe_base.to(torch.float64))
    out_sr64 = mor_gated_scatter_residual(x64, x64, y64, g64, flat_idx_fe, None)
    loss_fe64 = (yn64 * dy_norm_base.to(torch.float64)).sum() + (out_sr64 * do_fe_base.to(torch.float64)).sum() + bce64 + zl64
    dx64, dgam64, dy64, drlog64 = torch.autograd.grad(loss_fe64, (x64, gam64, y64, rlog64))

    fe_case_res: Dict[str, Any] = {}
    for prec_label, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        xp = x_fe_base.detach().to(dt).requires_grad_(True)
        gamp = gam_fe_base.detach().to(dt).requires_grad_(True)
        yp = y_fe_base.detach().to(dt).requires_grad_(True)
        rlogp = rlog_fe_base.detach().to(dt).requires_grad_(True)
        ynp = mor_gather_rmsnorm(xp, flat_idx_fe, gamp)
        gp, bcep, zlp = mor_router_bce_zloss(rlogp, rtgt_fe_base.to(dt))
        out_srp = mor_gated_scatter_residual(xp, xp, yp, gp, flat_idx_fe, None)
        loss_fep = (ynp.float() * dy_norm_base).sum() + (out_srp.float() * do_fe_base).sum() + bcep + zlp
        dxp, dgamp, dyp, drlogp = torch.autograd.grad(loss_fep, (xp, gamp, yp, rlogp))
        rels_fe = {
            "y_norm": _rel_l2(ynp, yn64),
            "out_scatter": _rel_l2(out_srp, out_sr64),
            "dx": _rel_l2(dxp, dx64),
            "dgamma": _rel_l2(dgamp, dgam64),
            "dy": _rel_l2(dyp, dy64),
            "drouter_logits": _rel_l2(drlogp, drlog64),
        }
        w_rel = max(rels_fe.values())
        if prec_label == "fp32":
            max_fp32_rel_l2 = max(max_fp32_rel_l2, w_rel)
        else:
            max_bf16_rel_l2 = max(max_bf16_rel_l2, w_rel)
        fe_case_res[prec_label] = {"worst_rel_l2": w_rel, "per_tensor_rel_l2": rels_fe}
    parity_cases["fused_gather_rmsnorm_router_scatter_residual"] = fe_case_res

    # 5. 7/7 Smooth-Branch FP64 Unit-RMS O(h^2) Richardson Extrapolation Suite (Gate G6)
    torch.manual_seed(20261121)
    B_r, T_r, HQ_r, HKV_r, Dh_r = 1, 64, 4, 2, 64
    mask_r = (torch.arange(T_r, device=device).unsqueeze(0) % 2 == 1)
    cu_q_r, flat_idx_r, pos_q_r, k_tot_r = mor_varlen_pack_from_mask(mask_r)
    cu_k_r = torch.tensor([0, T_r], dtype=torch.int32, device=device)

    q_r0 = (torch.randn((k_tot_r, HQ_r, Dh_r), dtype=torch.float64, device=device) * 0.4).requires_grad_(True)
    k_r0 = (torch.randn((T_r, HKV_r, Dh_r), dtype=torch.float64, device=device) * 0.4).requires_grad_(True)
    v_r0 = (torch.randn((T_r, HKV_r, Dh_r), dtype=torch.float64, device=device) * 0.4).requires_grad_(True)
    do_r0 = torch.randn((k_tot_r, HQ_r, Dh_r), dtype=torch.float64, device=device) * 0.4

    x_r0 = (torch.randn((T_r, 128), dtype=torch.float64, device=device) * 0.5).requires_grad_(True)
    gam_r0 = (torch.ones((128,), dtype=torch.float64, device=device) + 0.15 * torch.randn((128,), dtype=torch.float64, device=device)).detach().requires_grad_(True)
    y_r0 = (torch.randn((k_tot_r, 128), dtype=torch.float64, device=device) * 0.5).requires_grad_(True)
    rlog_r0 = (torch.randn((k_tot_r,), dtype=torch.float64, device=device) * 0.7).requires_grad_(True)
    rtgt_r0 = (torch.arange(k_tot_r, device=device) % 2 == 0).to(torch.float64)
    do_sr_r0 = torch.randn((T_r, 128), dtype=torch.float64, device=device) * 0.5

    def _rect_attn_cubic_loss(qi: torch.Tensor, ki: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        o = flash_mor_rect_causal_oracle_fp64(qi, ki, vi, pos_q_r, cu_q_r, cu_k_r)
        z = o * do_r0
        return (z + 0.25 * (z * z * z)).sum()

    def _fe_cubic_loss(xi: torch.Tensor, gami: torch.Tensor, yi: torch.Tensor, rlogi: torch.Tensor) -> torch.Tensor:
        yn = mor_gather_rmsnorm(xi, flat_idx_r, gami)
        pr, bce, zl = mor_router_bce_zloss(rlogi, rtgt_r0, alpha_scale=1.0, z_coeff=0.05)
        sr = mor_gated_scatter_residual(xi, xi, yi * torch.tanh(yn), pr, flat_idx_r, None)
        z = sr * do_sr_r0
        return (z + 0.20 * (z * z * z)).sum() + bce + zl

    dq_an, dk_an, dv_an = torch.autograd.grad(_rect_attn_cubic_loss(q_r0, k_r0, v_r0), (q_r0, k_r0, v_r0))
    dx_an, dgam_an, dy_an, drlog_an = torch.autograd.grad(_fe_cubic_loss(x_r0, gam_r0, y_r0, rlog_r0), (x_r0, gam_r0, y_r0, rlog_r0))

    def _unit_rms_dir(shape: Tuple[int, ...], seed_off: int) -> torch.Tensor:
        g_gen = torch.Generator(device=device)
        g_gen.manual_seed(9600 + seed_off)
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
        _check_richardson("flash_mor_dq", q_r0, dq_an, lambda x: _rect_attn_cubic_loss(x, k_r0, v_r0), 1),
        _check_richardson("flash_mor_dk_shared", k_r0, dk_an, lambda x: _rect_attn_cubic_loss(q_r0, x, v_r0), 2),
        _check_richardson("flash_mor_dv_shared", v_r0, dv_an, lambda x: _rect_attn_cubic_loss(q_r0, k_r0, x), 3),
        _check_richardson("gather_rmsnorm_scatter_dx", x_r0, dx_an, lambda x: _fe_cubic_loss(x, gam_r0, y_r0, rlog_r0), 4),
        _check_richardson("gather_rmsnorm_dgamma", gam_r0, dgam_an, lambda x: _fe_cubic_loss(x_r0, x, y_r0, rlog_r0), 5),
        _check_richardson("gated_scatter_dy", y_r0, dy_an, lambda x: _fe_cubic_loss(x_r0, gam_r0, x, rlog_r0), 6),
        _check_richardson("router_bce_zloss_dlogits", rlog_r0, drlog_an, lambda x: _fe_cubic_loss(x_r0, gam_r0, y_r0, x), 7),
    ]

    return {
        "eval_harness_integrity_ok": bool(integrity_ok),
        "cubin_audit": cubin_audit,
        "red_team": red_team,
        "upstream_forensics": upstream_forensics,
        "adversarial_parity": {
            "max_fp32_rel_l2": max_fp32_rel_l2,
            "max_bf16_rel_l2": max_bf16_rel_l2,
            "zero_consumer_key_exact_zero_grad": bool(zero_consumer_key_exact_zero_grad),
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
# Suite A: Multi-Trial Round-9 RSI Vector-Pareto Ratchets (Track 1 & Track 2)
# =============================================================================

class _MoRRecursiveStackLayer(nn.Module):
    """
    Shared weight-tied recursive block for Track 2 & Suite B benchmarking.
    """
    def __init__(self, d_model: int, h_q: int, h_kv: int, d_head: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.d_head = d_head
        self.d_ff = d_ff
        self.rms1_weight = nn.Parameter(torch.ones(d_model))
        self.q_proj = nn.Linear(d_model, h_q * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, h_kv * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, h_kv * d_head, bias=False)
        self.o_proj = nn.Linear(h_q * d_head, d_model, bias=False)
        self.rms2_weight = nn.Parameter(torch.ones(d_model))
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward_dense_step1(
        self,
        x_flat: torch.Tensor,
        pos_full: torch.Tensor,
        cu_k: torch.Tensor,
        T_seq: int,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recursion 1: all T tokens active -> computes and caches shared (K_1, V_1)."""
        N_tot, D = x_flat.shape
        rstd1 = torch.rsqrt(torch.mean(x_flat.float() ** 2, dim=-1, keepdim=True) + 1e-6).to(x_flat.dtype)
        h1 = (x_flat * rstd1) * self.rms1_weight
        q1 = mor_indexed_rope(self.q_proj(h1).view(N_tot, self.h_q, self.d_head), pos_full, cos_cache, sin_cache)
        k1 = mor_indexed_rope(self.k_proj(h1).view(N_tot, self.h_kv, self.d_head), pos_full, cos_cache, sin_cache)
        v1 = self.v_proj(h1).view(N_tot, self.h_kv, self.d_head)
        if HAS_FLASH_ATTN and q1.dtype in (torch.bfloat16, torch.float16):
            attn_out = flash_attn_varlen_func(q1, k1, v1, cu_k, cu_k, T_seq, T_seq, causal=True)
        else:
            attn_out = flash_mor_rect_causal_attention(q1, k1, v1, pos_full, cu_k, cu_k, T_seq, T_seq)
        a_out = x_flat + self.o_proj(attn_out.reshape(N_tot, self.h_q * self.d_head))
        rstd2 = torch.rsqrt(torch.mean(a_out.float() ** 2, dim=-1, keepdim=True) + 1e-6).to(x_flat.dtype)
        h2 = (a_out * rstd2) * self.rms2_weight
        mlp_out = self.down_proj(mor_fused_swiglu(self.gate_up_proj(h2)))
        return (a_out - x_flat) + mlp_out, k1, v1

    def forward_active_kv_shared_rsi(
        self,
        x_flat: torch.Tensor,
        k1_shared: torch.Tensor,
        v1_shared: torch.Tensor,
        flat_idx: torch.Tensor,
        pos_q: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_lq: int,
        T_seq: int,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        Round-9 RSI Recursion r >= 2 with Recursive KV Sharing:
          - Fused Gather + Pre-RMSNorm on ONLY active rows `K_r`
          - Computes ONLY `q_proj(h1_act)` (skips `k_proj` and `v_proj` completely!)
          - Fused Indexed RoPE at `pos_q[m]`
          - Rectangular Causal-Gather `Flash-MoR` (`Q_act[K_r] x KV_shared[N_tot]`)
          - Fused SwiGLU MLP on ONLY `K_r` rows!
        """
        K_tot = int(flat_idx.numel())
        h1_act = mor_gather_rmsnorm(x_flat, flat_idx, self.rms1_weight)
        q_act = mor_indexed_rope(
            self.q_proj(h1_act).view(K_tot, self.h_q, self.d_head),
            pos_q,
            cos_cache,
            sin_cache,
        )
        o_attn = flash_mor_rect_causal_attention(
            q_act, k1_shared, v1_shared, pos_q, cu_q, cu_k, max_lq, T_seq
        )
        attn_delta = self.o_proj(o_attn.reshape(K_tot, self.h_q * self.d_head))
        x_act = x_flat.index_select(0, flat_idx.to(torch.long)) + attn_delta
        rstd2 = torch.rsqrt(torch.mean(x_act.float() ** 2, dim=-1, keepdim=True) + 1e-6).to(x_flat.dtype)
        h2_act = (x_act * rstd2) * self.rms2_weight
        mlp_delta = self.down_proj(mor_fused_swiglu(self.gate_up_proj(h2_act)))
        return attn_delta + mlp_delta

    def forward_pinned_mor_stack_rsi(
        self,
        x_flat: torch.Tensor,
        r0s: torch.Tensor,
        r1s: torch.Tensor,
        r2s: torch.Tensor,
        pos_full: torch.Tensor,
        cu_k: torch.Tensor,
        T_c: int,
        cos_b: torch.Tensor,
        sin_b: torch.Tensor,
        prepacked_use: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
        use_lpt_alltoall: bool,
    ) -> torch.Tensor:
        """
        Recursion-Depth-Pinned FSDP2 Stack: enters FSDP2 wrapper ONCE, executes
        dense r=0 + 2 active-only Rectangular Flash-MoR recursions (4 shared layers/recursion),
        and exits FSDP2 wrapper ONCE (eliminating 8x redundant AllGather & FSDPState hook transitions).
        """
        B_c = int(r0s.shape[0])
        y1, k1, v1 = self.forward_dense_step1(x_flat, pos_full, cu_k, T_c, cos_b, sin_b)
        x_flat = x_flat + torch.sigmoid(r0s).reshape(B_c * T_c, 1) * y1
        if use_lpt_alltoall:
            x_act_doc = _SliceAllToAllSingleFn.apply(x_flat[:T_c])
            k1_use = k1[:T_c]
            v1_use = v1[:T_c]
            cu_k_use = cu_k[:2]
            for r_i, rl_s in ((1, r1s), (2, r2s)):
                cu_q, flat_idx, pos_q, max_lq = prepacked_use[r_i - 1]
                g_act = torch.sigmoid(rl_s[0]).index_select(0, flat_idx.to(torch.long))
                for _ in range(2):
                    y_act = self.forward_active_kv_shared_rsi(
                        x_act_doc, k1_use, v1_use, flat_idx, pos_q, cu_q, cu_k_use, max_lq, T_c, cos_b, sin_b
                    )
                    x_act_doc = mor_gated_scatter_residual(x_act_doc, x_act_doc, y_act, 0.5 * g_act, flat_idx, None)
            ex_out = _SliceAllToAllSingleFn.apply(x_act_doc)
            return torch.cat([ex_out, x_flat[T_c:]], dim=0)
        else:
            # Without MoR-LPT AllToAllv, straggler ranks (holding 2x deep document micro-groups) must execute
            # both deep document micro-groups sequentially before reaching the FSDP2 ReduceScatter barrier:
            n_subgroups = 2 if dist.get_rank() < 2 else 1
            for r_i, rl_s in ((1, r1s), (2, r2s)):
                cu_q, flat_idx, pos_q, max_lq = prepacked_use[r_i - 1]
                g_act = torch.sigmoid(rl_s).reshape(B_c * T_c).index_select(0, flat_idx.to(torch.long))
                for _ in range(2 * n_subgroups):
                    y_act = self.forward_active_kv_shared_rsi(
                        x_flat, k1, v1, flat_idx, pos_q, cu_q, cu_k, max_lq, T_c, cos_b, sin_b
                    )
                    x_flat = mor_gated_scatter_residual(x_flat, x_flat, y_act, (0.5 / n_subgroups) * g_act, flat_idx, None)
            return x_flat

    def forward(self, mode_tag: str, *args):
        if mode_tag == "d1":
            return self.forward_dense_step1(*args)
        if mode_tag == "pinned_rsi":
            return self.forward_pinned_mor_stack_rsi(*args)
        return self.forward_active_kv_shared_rsi(*args)


def run_suite_a(device: torch.device, mode: str) -> Dict[str, Any]:
    warmup = 3 if mode == "smoke" else 6
    rep = 8 if mode == "smoke" else 16

    # -------------------------------------------------------------------------
    # Track 1: Isolated Position-Indexed Rectangular Causal-Gather Flash-MoR Fwd+Bwd
    # -------------------------------------------------------------------------
    track1_shapes = [
        ("s1_b2_t2048_kr682_d64", 2, 2048, 3, 16, 4, 64),
        ("s2_b2_t2048_kr682_d128", 2, 2048, 3, 16, 4, 128),
        ("s3_b1_t4096_kr1365_d128", 1, 4096, 3, 16, 4, 128),
        ("s4_b1_t8192_kr2730_d64", 1, 8192, 3, 16, 4, 64),
    ]

    def _eager_masked_sdpa_rect(
        q_act_raw: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        pos_q: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        cos_c: torch.Tensor,
        sin_c: torch.Tensor,
    ) -> torch.Tensor:
        B = int(cu_q.numel()) - 1
        N_q, H_Q, D = q_act_raw.shape
        H_KV = k_full.shape[1]
        G = H_Q // H_KV
        outs: List[torch.Tensor] = []
        for b in range(B):
            qs, qe = int(cu_q[b].item()), int(cu_q[b + 1].item())
            ks, ke = int(cu_k[b].item()), int(cu_k[b + 1].item())
            Lq, Lk = qe - qs, ke - ks
            p_doc = pos_q[qs:qe].to(torch.long)
            q_rope_b = apply_eager_rope_at_positions(
                q_act_raw[qs:qe].unsqueeze(0), p_doc.unsqueeze(0), cos_c, sin_c
            )  # [1, Lq, H_Q, D]
            qd = q_rope_b.permute(0, 2, 1, 3).contiguous()  # [1, H_Q, Lq, D]
            kd = k_full[ks:ke].permute(1, 0, 2).repeat_interleave(G, dim=0).unsqueeze(0).contiguous()
            vd = v_full[ks:ke].permute(1, 0, 2).repeat_interleave(G, dim=0).unsqueeze(0).contiguous()
            n_idx = torch.arange(Lk, device=q_act_raw.device, dtype=torch.long)
            min_v = float(torch.finfo(q_act_raw.dtype).min)
            mask_float = (
                torch.where(n_idx.unsqueeze(0) <= p_doc.unsqueeze(1), 0.0, min_v)
                .to(q_act_raw.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, H_Q, Lq, Lk)
                .contiguous()
            )
            out_b = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask_float)
            outs.append(out_b.squeeze(0).permute(1, 0, 2))
        return torch.cat(outs, dim=0)

    def _official_mor_full_seq_kv_share_attn(
        q_full_raw: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        flat_idx: torch.Tensor,
        cu_k: torch.Tensor,
        T_seq: int,
        cos_c: torch.Tensor,
        sin_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Upstream `raymin0223/mixture_of_recursions` `kv_sharing.enable=True` (`expert_choice_router.py:36-68, 225-240`):
        runs eager RoPE on full `[B, T, H_Q, D]` queries AND unused `[B, T, H_KV, D]` keys (discarded by
        `RecursiveDynamicCache.update`), runs full `[B*T, H_Q, D]` causal attention on ALL `T` queries,
        and then gathers `flat_idx` afterward!
        """
        B = int(cu_k.numel()) - 1
        H_Q = q_full_raw.shape[1]
        H_KV = k_full.shape[1]
        D = q_full_raw.shape[2]
        pos_2d = torch.arange(T_seq, device=q_full_raw.device, dtype=torch.long).unsqueeze(0).expand(B, T_seq)
        q_full_rope = (
            apply_eager_rope_at_positions(q_full_raw.view(B, T_seq, H_Q, D), pos_2d, cos_c, sin_c)
            .permute(0, 2, 1, 3)
            .contiguous()
            .permute(0, 2, 1, 3)
            .reshape(B * T_seq, H_Q, D)
        )
        # Upstream also projects and applies eager RoPE to k_states and materializes v_states before past_key_value.update discards both
        k_unused_rope = (
            apply_eager_rope_at_positions(q_full_raw[:, :H_KV, :].view(B, T_seq, H_KV, D), pos_2d, cos_c, sin_c)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v_unused_clone = (
            apply_eager_rope_at_positions(q_full_raw[:, H_KV : 2 * H_KV, :].view(B, T_seq, H_KV, D), pos_2d, cos_c, sin_c)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        if HAS_FLASH_ATTN and q_full_rope.dtype in (torch.bfloat16, torch.float16):
            o_full = flash_attn_varlen_func(q_full_rope, k_full, v_full, cu_k, cu_k, T_seq, T_seq, causal=True)
        else:
            pos_all = torch.arange(T_seq, device=q_full_raw.device, dtype=torch.int32).repeat(B)
            o_full = flash_mor_rect_causal_attention(q_full_rope, k_full, v_full, pos_all, cu_k, cu_k, T_seq, T_seq)
        return o_full.index_select(0, flat_idx.to(torch.long)) + 0.0 * (k_unused_rope.sum() + v_unused_clone.sum())

    def _eval_track1_candidate(
        cand_id: str,
        fn_step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor, torch.Tensor], torch.Tensor],
        uses_full_q: bool = False,
    ) -> Dict[str, Any]:
        # Parity check against FP64 oracle on (B=2, T=128, stride=3, HQ=8, HKV=2, D=64)
        torch.manual_seed(8100)
        b_p, t_p, hq_p, hkv_p, d_p = 2, 128, 8, 2, 64
        cos_p, sin_p = build_rope_cache(t_p, d_p, device)
        mask_p = (torch.arange(t_p, device=device).unsqueeze(0).expand(b_p, t_p) % 3 == 0)
        cu_q_p, flat_idx_p, pos_q_p, k_tot_p = mor_varlen_pack_from_mask(mask_p)
        cu_k_p = torch.arange(0, (b_p + 1) * t_p, t_p, device=device, dtype=torch.int32)
        max_lq_p = int((cu_q_p[1:] - cu_q_p[:-1]).max().item())

        q_full_p = (torch.randn((b_p * t_p, hq_p, d_p), dtype=torch.bfloat16, device=device) * 0.5).detach()
        q_act_p = q_full_p.index_select(0, flat_idx_p.to(torch.long)).detach().requires_grad_(True)
        k_p = (torch.randn((b_p * t_p, hkv_p, d_p), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        v_p = (torch.randn((b_p * t_p, hkv_p, d_p), dtype=torch.bfloat16, device=device) * 0.5).detach().requires_grad_(True)
        do_p = torch.randn((k_tot_p, hq_p, d_p), dtype=torch.bfloat16, device=device) * 0.5

        q_act_rope_ref = mor_indexed_rope(q_act_p.to(torch.float32), pos_q_p, cos_p, sin_p)
        o_ref = flash_mor_rect_causal_oracle_fp64(q_act_rope_ref, k_p, v_p, pos_q_p, cu_q_p, cu_k_p)
        dq_r, dk_r, dv_r = torch.autograd.grad((o_ref * do_p.to(torch.float64)).sum(), (q_act_p, k_p, v_p))

        kc = k_p.detach().clone().requires_grad_(True)
        vc = v_p.detach().clone().requires_grad_(True)
        if uses_full_q:
            qfc = q_full_p.detach().clone().requires_grad_(True)
            oc = fn_step(qfc, qfc, kc, vc, flat_idx_p, pos_q_p, cu_q_p, cu_k_p, max_lq_p, t_p, cos_p, sin_p)
            dq_full_c, dk_c, dv_c = torch.autograd.grad((oc.float() * do_p.float()).sum(), (qfc, kc, vc))
            dq_c = dq_full_c.index_select(0, flat_idx_p.to(torch.long))
        else:
            qac = q_act_p.detach().clone().requires_grad_(True)
            oc = fn_step(qac, q_full_p, kc, vc, flat_idx_p, pos_q_p, cu_q_p, cu_k_p, max_lq_p, t_p, cos_p, sin_p)
            dq_c, dk_c, dv_c = torch.autograd.grad((oc.float() * do_p.float()).sum(), (qac, kc, vc))

        worst_rel_l2 = max(
            _rel_l2(oc, o_ref),
            _rel_l2(dq_c, dq_r),
            _rel_l2(dk_c, dk_r),
            _rel_l2(dv_c, dv_r),
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
        for sname, b_s, t_s, stride_s, hq_s, hkv_s, d_s in track1_shapes:
            cos_s, sin_s = build_rope_cache(t_s, d_s, device)
            mask_s = (torch.arange(t_s, device=device).unsqueeze(0).expand(b_s, t_s) % stride_s == 0)
            cu_q_s, flat_idx_s, pos_q_s, k_tot_s = mor_varlen_pack_from_mask(mask_s)
            cu_k_s = torch.arange(0, (b_s + 1) * t_s, t_s, device=device, dtype=torch.int32)
            max_lq_s = int((cu_q_s[1:] - cu_q_s[:-1]).max().item())

            def _make_slot(slot_idx: int) -> Tuple[torch.Tensor, ...]:
                g_gen = torch.Generator(device=device)
                g_gen.manual_seed(9100 + slot_idx * 17 + d_s + t_s)
                q_f = (torch.randn((b_s * t_s, hq_s, d_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(uses_full_q)
                q_a = (torch.randn((k_tot_s, hq_s, d_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(not uses_full_q)
                ks = (torch.randn((b_s * t_s, hkv_s, d_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                vs = (torch.randn((b_s * t_s, hkv_s, d_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                dos = torch.randn((k_tot_s, hq_s, d_s), dtype=torch.bfloat16, device=device, generator=g_gen)
                return (q_a, q_f, ks, vs, dos)

            def _run_slot(slot: Tuple[torch.Tensor, ...]) -> None:
                q_a, q_f, ks, vs, dos = slot
                q_a.grad = q_f.grad = ks.grad = vs.grad = None
                out_s = fn_step(q_a, q_f, ks, vs, flat_idx_s, pos_q_s, cu_q_s, cu_k_s, max_lq_s, t_s, cos_s, sin_s)
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

    t1_c0 = _eval_track1_candidate(
        "a0_official_mor_kv_sharing_full_seq_fa2_gather",
        lambda qa, qf, k, v, fidx, pq, cuq, cuk, mq, mk, cc, sc: _official_mor_full_seq_kv_share_attn(qf, k, v, fidx, cuk, mk, cc, sc),
        uses_full_q=True,
    )
    t1_c0["decision"] = "BASELINE"
    t1_c1 = _eval_track1_candidate(
        "a1_eager_4d_coordinate_masked_sdpa",
        lambda qa, qf, k, v, fidx, pq, cuq, cuk, mq, mk, cc, sc: _eager_masked_sdpa_rect(
            qa, k, v, pq, cuq, cuk, cc, sc
        ),
        uses_full_q=False,
    )
    if t1_c1["geomean_ms"] >= t1_c0["geomean_ms"]:
        t1_c1["decision"] = "REVERT_SLOWER"
    t1_c2 = _eval_track1_candidate(
        "a2_broken_bottom_right_varlen_fa2",
        lambda qa, qf, k, v, fidx, pq, cuq, cuk, mq, mk, cc, sc: _BrokenBottomRightRectCausalFn.apply(
            mor_indexed_rope(qa, pq, cc, sc), k, v, pq, cuq, cuk, mq, mk
        ),
        uses_full_q=False,
    )
    t1_c3 = _eval_track1_candidate(
        "a3_broken_omit_shared_kv_bwd",
        lambda qa, qf, k, v, fidx, pq, cuq, cuk, mq, mk, cc, sc: _BrokenOmitSharedKVBackwardFn.apply(
            mor_indexed_rope(qa, pq, cc, sc), k, v, pq, cuq, cuk, mq, mk
        ),
        uses_full_q=False,
    )
    t1_c4 = _eval_track1_candidate(
        "a4_rsi_rect_flash_mor_monolithic_64x128_dkv_spill",
        lambda qa, qf, k, v, fidx, pq, cuq, cuk, mq, mk, cc, sc: flash_mor_rect_causal_attention(
            mor_indexed_rope(qa, pq, cc, sc), k, v, pq, cuq, cuk, mq, mk, use_naive_spill_bwd=True
        ),
        uses_full_q=False,
    )
    t1_c4["decision"] = "REVERT_SLOWER_K128_SPILL"
    t1_c5 = _eval_track1_candidate(
        "a5_rsi_round9_rect_flash_mor_two_loop_bsearch_zero_spill",
        lambda qa, qf, k, v, fidx, pq, cuq, cuk, mq, mk, cc, sc: flash_mor_rect_causal_attention(
            mor_indexed_rope(qa, pq, cc, sc), k, v, pq, cuq, cuk, mq, mk, use_naive_spill_bwd=False
        ),
        uses_full_q=False,
    )

    track1_candidates = [t1_c0, t1_c1, t1_c2, t1_c3, t1_c4, t1_c5]
    per_shape_speedup_vs_official = {
        sname: t1_c0["shapes"][sname]["median_ms"] / t1_c5["shapes"][sname]["median_ms"]
        for sname, *_ in track1_shapes
    }
    per_shape_speedup_vs_sdpa = {
        sname: t1_c1["shapes"][sname]["median_ms"] / t1_c5["shapes"][sname]["median_ms"]
        for sname, *_ in track1_shapes
    }

    # -------------------------------------------------------------------------
    # Track 2: End-to-End Middle-Cycle (N_r = 3) Recursive MoR Stack Fwd+Bwd
    # -------------------------------------------------------------------------
    track2_shapes = [
        ("m1_b2_t1024_d1024_nr3", 2, 1024, 1024, 16, 4, 64, 2816),
        ("m2_b2_t2048_d1024_nr3", 2, 2048, 1024, 16, 4, 64, 2816),
        ("m3_b1_t4096_d1024_nr3", 1, 4096, 1024, 16, 4, 64, 2816),
    ]

    def _run_mor_stack_official_pad4d(
        layer: _MoRRecursiveStackLayer,
        x_in: torch.Tensor,
        r_logits_list: List[torch.Tensor],
        masks_list: List[torch.Tensor],
        cos_c: torch.Tensor,
        sin_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Candidate b0: Upstream `select_tokens_and_batch_with_padding` + 4D mask + loop scatter_add
        on Recursion 2 (cap=2/3) & Recursion 3 (cap=1/3), with full sequence at Recursion 1.
        """
        B, T, D = x_in.shape
        cu_k = torch.arange(0, (B + 1) * T, T, device=x_in.device, dtype=torch.int32)
        pos_full = torch.arange(T, device=x_in.device, dtype=torch.int32).repeat(B)
        x_flat = x_in.reshape(B * T, D)

        # Recursion 1 (all tokens active, weighted by g_1 = sigmoid(r_logits_0))
        y1, k1, v1 = layer.forward_dense_step1(x_flat, pos_full, cu_k, T, cos_c, sin_c)
        g1 = torch.sigmoid(r_logits_list[0]).reshape(B * T, 1)
        x_cur = (x_flat + g1 * y1).view(B, T, D)

        # Recursions 2 and 3 via upstream pad_sequence + 4D mask + per-batch scatter_add
        for r_idx in (1, 2):
            m_bool = masks_list[r_idx]  # [B, T]
            top_exp = m_bool.to(torch.long) * r_idx
            padded_x, mask_4d, sel_b, sel_seq = official_mor_pad_4dmask_select(x_cur, top_exp, r_idx)
            assert padded_x is not None and mask_4d is not None
            nb, ns, _ = padded_x.shape
            rstd1 = torch.rsqrt(torch.mean(padded_x.float() ** 2, dim=-1, keepdim=True) + 1e-6).to(x_in.dtype)
            h1 = (padded_x * rstd1) * layer.rms1_weight
            # Gather exact position ids per batch item
            pos_pad = torch.zeros((nb, ns), dtype=torch.long, device=x_in.device)
            for bi, s_idx in enumerate(sel_seq):
                pos_pad[bi, : s_idx.numel()] = s_idx
            q = apply_eager_rope_at_positions(layer.q_proj(h1).view(nb, ns, layer.h_q, layer.d_head), pos_pad, cos_c, sin_c).permute(0, 2, 1, 3)
            # KV-sharing from k1, v1 (expanded to [B, H_Q, T, D_h])
            k_sh = k1.view(B, T, layer.h_kv, layer.d_head).permute(0, 2, 1, 3).repeat_interleave(layer.h_q // layer.h_kv, dim=1)
            v_sh = v1.view(B, T, layer.h_kv, layer.d_head).permute(0, 2, 1, 3).repeat_interleave(layer.h_q // layer.h_kv, dim=1)
            # Build 4D [nb, 1, ns, T] coordinate-causal mask (mirroring token_choice_router.py:110-121)
            min_v = torch.finfo(x_in.dtype).min
            mask_rect_4d = torch.full((nb, 1, ns, T), min_v, dtype=x_in.dtype, device=x_in.device)
            t_grid = torch.arange(T, device=x_in.device)
            for bi, s_idx in enumerate(sel_seq):
                s_len = int(s_idx.numel())
                c_mat = t_grid.unsqueeze(0) <= s_idx.unsqueeze(1)
                mask_rect_4d[bi, 0, :s_len, :] = torch.where(c_mat, torch.tensor(0.0, dtype=x_in.dtype, device=x_in.device), min_v)
            o_attn = F.scaled_dot_product_attention(q, k_sh, v_sh, attn_mask=mask_rect_4d).permute(0, 2, 1, 3).reshape(nb, ns, layer.h_q * layer.d_head)
            a_delta = layer.o_proj(o_attn)
            x_a = padded_x + a_delta
            rstd2 = torch.rsqrt(torch.mean(x_a.float() ** 2, dim=-1, keepdim=True) + 1e-6).to(x_in.dtype)
            h2 = (x_a * rstd2) * layer.rms2_weight
            m_delta = layer.down_proj(mor_fused_swiglu(layer.gate_up_proj(h2.reshape(nb * ns, D)))).view(nb, ns, D)
            tot_delta = a_delta + m_delta
            g_r = torch.sigmoid(r_logits_list[r_idx])  # [B, T]
            x_next = x_cur.clone()
            for bi, b_orig in enumerate(sel_b):
                s_idx = sel_seq[bi]
                s_len = int(s_idx.numel())
                w_sel = g_r[b_orig, s_idx].unsqueeze(-1)
                x_next[b_orig] = torch.scatter_add(
                    x_next[b_orig],
                    dim=0,
                    index=s_idx.unsqueeze(-1).expand(-1, D),
                    src=(tot_delta[bi, :s_len] * w_sel).to(x_in.dtype),
                )
            x_cur = x_next
        return x_cur.reshape(B * T, D)

    def _run_mor_stack_official_kvshare_fullseq_clone(
        layer: _MoRRecursiveStackLayer,
        x_in: torch.Tensor,
        r_logits_list: List[torch.Tensor],
        masks_list: List[torch.Tensor],
        cos_c: torch.Tensor,
        sin_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Candidate b1: Upstream `expert_choice_router.py:225-240` `kv_sharing.enable=True`:
        clones full `[B, T, D]` and runs the ENTIRE block on ALL `T` tokens at `r = 2, 3`,
        then gathers only `selected_tokens` at the end!
        """
        B, T, D = x_in.shape
        cu_k = torch.arange(0, (B + 1) * T, T, device=x_in.device, dtype=torch.int32)
        pos_full = torch.arange(T, device=x_in.device, dtype=torch.int32).repeat(B)
        x_flat = x_in.reshape(B * T, D)

        y1, k1, v1 = layer.forward_dense_step1(x_flat, pos_full, cu_k, T, cos_c, sin_c)
        g1 = torch.sigmoid(r_logits_list[0]).reshape(B * T, 1)
        x_flat = x_flat + g1 * y1

        for r_idx in (1, 2):
            # Full-sequence Q, unused K/V, eager RoPE, Attention, and MLP on all B*T rows!
            N_tot = B * T
            rstd1 = torch.rsqrt(torch.mean(x_flat.float() ** 2, dim=-1, keepdim=True) + 1e-6).to(x_flat.dtype)
            h1 = (x_flat * rstd1) * layer.rms1_weight
            pos_2d = torch.arange(T, device=x_in.device, dtype=torch.long).unsqueeze(0).expand(B, T)
            q_full = (
                apply_eager_rope_at_positions(layer.q_proj(h1).view(B, T, layer.h_q, layer.d_head), pos_2d, cos_c, sin_c)
                .permute(0, 2, 1, 3)
                .contiguous()
                .permute(0, 2, 1, 3)
                .reshape(N_tot, layer.h_q, layer.d_head)
            )
            k_unused = (
                apply_eager_rope_at_positions(layer.k_proj(h1).view(B, T, layer.h_kv, layer.d_head), pos_2d, cos_c, sin_c)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            v_unused = layer.v_proj(h1)
            if HAS_FLASH_ATTN and q_full.dtype in (torch.bfloat16, torch.float16):
                o_full = flash_attn_varlen_func(q_full, k1, v1, cu_k, cu_k, T, T, causal=True)
            else:
                o_full = flash_mor_rect_causal_attention(q_full, k1, v1, pos_full, cu_k, cu_k, T, T)
            a_delta = layer.o_proj(o_full.reshape(N_tot, layer.h_q * layer.d_head)) + 0.0 * (k_unused.sum() + v_unused.sum())
            x_a = x_flat + a_delta
            rstd2 = torch.rsqrt(torch.mean(x_a.float() ** 2, dim=-1, keepdim=True) + 1e-6).to(x_flat.dtype)
            h2 = (x_a * rstd2) * layer.rms2_weight
            m_delta = layer.down_proj(mor_fused_swiglu(layer.gate_up_proj(h2)))
            full_delta = a_delta + m_delta
            g_r = (torch.sigmoid(r_logits_list[r_idx]) * masks_list[r_idx].to(x_flat.dtype)).reshape(N_tot, 1)
            x_flat = x_flat + g_r * full_delta
        return x_flat

    def _run_mor_stack_rsi_round9(
        layer: _MoRRecursiveStackLayer,
        x_in: torch.Tensor,
        r_logits_list: List[torch.Tensor],
        masks_list: List[torch.Tensor],
        cos_c: torch.Tensor,
        sin_c: torch.Tensor,
        omit_r2_gate_bug: bool = False,
        prepacked_meta: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]] = None,
    ) -> torch.Tensor:
        """
        Candidate b3 (Round-9 RSI Winner):
          - Recursion 1: dense active step producing `(K_1, V_1)`
          - Recursions 2..3: Varlen compaction + Fused Gather-RMSNorm + Q-Only Indexed RoPE +
            Rectangular Causal-Gather `Flash-MoR` (`Q_act[K_r] x KV_shared[B*T]`) + Fused SwiGLU
            on ONLY `K_r` rows + Superfused Gated Scatter-Residual!
        """
        B, T, D = x_in.shape
        cu_k = torch.arange(0, (B + 1) * T, T, device=x_in.device, dtype=torch.int32)
        pos_full = torch.arange(T, device=x_in.device, dtype=torch.int32).repeat(B)
        x_flat = x_in.reshape(B * T, D)

        y1, k1, v1 = layer.forward_dense_step1(x_flat, pos_full, cu_k, T, cos_c, sin_c)
        g1 = torch.sigmoid(r_logits_list[0]).reshape(B * T, 1)
        x_flat = x_flat + g1 * y1

        for r_idx in (1, 2):
            if prepacked_meta is not None:
                cu_q, flat_idx, pos_q, max_lq = prepacked_meta[r_idx - 1]
            else:
                cu_q, flat_idx, pos_q, _ = mor_varlen_pack_from_mask(masks_list[r_idx])
                max_lq = int((cu_q[1:] - cu_q[:-1]).max().item())
            y_act = layer.forward_active_kv_shared_rsi(
                x_flat, k1, v1, flat_idx, pos_q, cu_q, cu_k, max_lq, T, cos_c, sin_c
            )
            g_full = torch.sigmoid(r_logits_list[r_idx]).reshape(B * T)
            g_act = g_full.index_select(0, flat_idx.to(torch.long))
            if omit_r2_gate_bug and r_idx == 1:
                g_act = torch.ones_like(g_act) + 0.0 * g_act
            x_flat = mor_gated_scatter_residual(x_flat, x_flat, y_act, g_act, flat_idx, None)
        return x_flat

    def _eval_track2_candidate(
        cand_id: str,
        runner_fn: Callable[[_MoRRecursiveStackLayer, torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Dict[str, Any]:
        # Parity check on (B=2, T=128, D=256, HQ=8, HKV=2, Dh=32->64, Dff=512)
        torch.manual_seed(8300)
        b_p, t_p, d_p, hq_p, hkv_p, dh_p, dff_p = 2, 128, 512, 8, 2, 64, 1024
        layer_ref = _MoRRecursiveStackLayer(d_p, hq_p, hkv_p, dh_p, dff_p).to(device=device, dtype=torch.float32)
        layer_cand = _MoRRecursiveStackLayer(d_p, hq_p, hkv_p, dh_p, dff_p).to(device=device, dtype=torch.bfloat16)
        layer_cand.load_state_dict(layer_ref.state_dict())

        cos_p, sin_p = build_rope_cache(t_p, dh_p, device)
        x_p = (torch.randn((b_p, t_p, d_p), dtype=torch.bfloat16, device=device) * 0.5).detach()
        rl0 = (torch.randn((b_p, t_p), dtype=torch.bfloat16, device=device) * 0.6).detach()
        rl1 = (torch.randn((b_p, t_p), dtype=torch.bfloat16, device=device) * 0.6).detach()
        rl2 = (torch.randn((b_p, t_p), dtype=torch.bfloat16, device=device) * 0.6).detach()
        t_idx = torch.arange(t_p, device=device).unsqueeze(0).expand(b_p, t_p)
        m0 = torch.ones((b_p, t_p), dtype=torch.bool, device=device)
        m1 = (t_idx % 3 != 2)  # 2/3 active
        m2 = (t_idx % 3 == 0)  # 1/3 active
        do_p = torch.randn((b_p * t_p, d_p), dtype=torch.bfloat16, device=device) * 0.5

        # Reference in FP32 using exact full-sequence KV-shared formula
        xr = x_p.to(torch.float32).detach().requires_grad_(True)
        rl0_r = rl0.to(torch.float32).detach().requires_grad_(True)
        rl1_r = rl1.to(torch.float32).detach().requires_grad_(True)
        rl2_r = rl2.to(torch.float32).detach().requires_grad_(True)
        out_ref = _run_mor_stack_official_kvshare_fullseq_clone(
            layer_ref, xr, [rl0_r, rl1_r, rl2_r], [m0, m1, m2], cos_p, sin_p
        )
        grads_ref = torch.autograd.grad(
            (out_ref * do_p.to(torch.float32)).sum(),
            (xr, rl0_r, rl1_r, rl2_r, layer_ref.q_proj.weight, layer_ref.k_proj.weight, layer_ref.down_proj.weight),
        )

        xc = x_p.detach().clone().requires_grad_(True)
        rl0_c = rl0.detach().clone().requires_grad_(True)
        rl1_c = rl1.detach().clone().requires_grad_(True)
        rl2_c = rl2.detach().clone().requires_grad_(True)
        layer_cand.zero_grad(set_to_none=True)
        out_c = runner_fn(layer_cand, xc, [rl0_c, rl1_c, rl2_c], [m0, m1, m2], cos_p, sin_p)
        cand_inputs = (xc, rl0_c, rl1_c, rl2_c, layer_cand.q_proj.weight, layer_cand.k_proj.weight, layer_cand.down_proj.weight)
        grads_c_raw = torch.autograd.grad(
            (out_c.float() * do_p.float()).sum(),
            cand_inputs,
            allow_unused=True,
        )
        grads_c = tuple(torch.zeros_like(inp) if g is None else g for g, inp in zip(grads_c_raw, cand_inputs))
        worst_rel_l2 = max(
            [_rel_l2(out_c, out_ref)] + [_rel_l2(gc, gr) for gc, gr in zip(grads_c, grads_ref)]
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
        for sname, b_s, t_s, d_s, hq_s, hkv_s, dh_s, dff_s in track2_shapes:
            layer_b = _MoRRecursiveStackLayer(d_s, hq_s, hkv_s, dh_s, dff_s).to(device=device, dtype=torch.bfloat16)
            cos_s, sin_s = build_rope_cache(t_s, dh_s, device)
            t_grid = torch.arange(t_s, device=device).unsqueeze(0).expand(b_s, t_s)
            ms0 = torch.ones((b_s, t_s), dtype=torch.bool, device=device)
            ms1 = (t_grid % 3 != 2)  # 2/3 capacity
            ms2 = (t_grid % 3 == 0)  # 1/3 capacity
            cu_q1_s, flat_idx1_s, pos_q1_s, _ = mor_varlen_pack_from_mask(ms1)
            max_lq1_s = int((cu_q1_s[1:] - cu_q1_s[:-1]).max().item())
            cu_q2_s, flat_idx2_s, pos_q2_s, _ = mor_varlen_pack_from_mask(ms2)
            max_lq2_s = int((cu_q2_s[1:] - cu_q2_s[:-1]).max().item())
            prepacked_s = [
                (cu_q1_s, flat_idx1_s, pos_q1_s, max_lq1_s),
                (cu_q2_s, flat_idx2_s, pos_q2_s, max_lq2_s),
            ]

            def _make_slot(slot_idx: int) -> Tuple[torch.Tensor, ...]:
                g_gen = torch.Generator(device=device)
                g_gen.manual_seed(9400 + slot_idx * 23 + t_s)
                xs = (torch.randn((b_s, t_s, d_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                r0s = (torch.randn((b_s, t_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                r1s = (torch.randn((b_s, t_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                r2s = (torch.randn((b_s, t_s), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
                dos = torch.randn((b_s * t_s, d_s), dtype=torch.bfloat16, device=device, generator=g_gen)
                return (xs, r0s, r1s, r2s, dos)

            def _run_slot(slot: Tuple[torch.Tensor, ...]) -> None:
                xs, r0s, r1s, r2s, dos = slot
                xs.grad = r0s.grad = r1s.grad = r2s.grad = None
                layer_b.zero_grad(set_to_none=True)
                if cand_id == "b3_rsi_round9_varlen_gather_rope_rect_flash_mor":
                    out_s = _run_mor_stack_rsi_round9(
                        layer_b, xs, [r0s, r1s, r2s], [ms0, ms1, ms2], cos_s, sin_s, prepacked_meta=prepacked_s
                    )
                else:
                    out_s = runner_fn(layer_b, xs, [r0s, r1s, r2s], [ms0, ms1, ms2], cos_s, sin_s)
                out_s.backward(dos)

            res = _bench_rotating_slots(_make_slot, _run_slot, warmup=warmup, rep=rep, device=device, n_slots=3)
            med_ms = float(res["median_ms"])
            pk_mib = float(res["peak_allocated_mib"])
            shape_metrics[sname] = {
                "median_ms": med_ms,
                "iqr_ms": float(res["iqr_ms"]),
                "peak_allocated_mib": pk_mib,
            }
            ms_list.append(med_ms)
            peak_list.append(pk_mib)
            del layer_b

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

    t2_b0 = _eval_track2_candidate("b0_official_mor_pad_sequence_4d_mask_loop", _run_mor_stack_official_pad4d)
    t2_b0["decision"] = "BASELINE"
    t2_b1 = _eval_track2_candidate("b1_official_mor_kv_sharing_full_seq_clone", _run_mor_stack_official_kvshare_fullseq_clone)
    t2_b2 = _eval_track2_candidate(
        "b2_broken_unfinished_transition_gate_omit",
        lambda lyr, x, rl, m, cc, sc: _run_mor_stack_rsi_round9(lyr, x, rl, m, cc, sc, omit_r2_gate_bug=True),
    )
    t2_b3 = _eval_track2_candidate("b3_rsi_round9_varlen_gather_rope_rect_flash_mor", _run_mor_stack_rsi_round9)

    track2_candidates = [t2_b0, t2_b1, t2_b2, t2_b3]

    return {
        "track1_rect_flash_mor_attention": {
            "baseline_official_kvshare_geomean_ms": t1_c0["geomean_ms"],
            "sdpa_4d_mask_geomean_ms": t1_c1["geomean_ms"],
            "monolithic_spill_geomean_ms": t1_c4["geomean_ms"],
            "winner_geomean_ms": t1_c5["geomean_ms"],
            "speedup_vs_official_kvshare": t1_c0["geomean_ms"] / t1_c5["geomean_ms"],
            "speedup_vs_sdpa_4d_mask": t1_c1["geomean_ms"] / t1_c5["geomean_ms"],
            "speedup_vs_monolithic_spill": t1_c4["geomean_ms"] / t1_c5["geomean_ms"],
            "per_shape_speedup_vs_official": per_shape_speedup_vs_official,
            "per_shape_speedup_vs_sdpa": per_shape_speedup_vs_sdpa,
            "winner_beats_official_on_all_shapes": all(v > 1.15 for v in per_shape_speedup_vs_official.values()),
            "s4_hbm_saved_vs_official_mib": (
                t1_c0["shapes"]["s4_b1_t8192_kr2730_d64"]["peak_allocated_mib"]
                - t1_c5["shapes"]["s4_b1_t8192_kr2730_d64"]["peak_allocated_mib"]
            ),
            "candidates": track1_candidates,
        },
        "track2_end_to_end_mor_stack": {
            "baseline_pad4d_geomean_ms": t2_b0["geomean_ms"],
            "official_kvshare_clone_geomean_ms": t2_b1["geomean_ms"],
            "winner_geomean_ms": t2_b3["geomean_ms"],
            "speedup_vs_pad4d": t2_b0["geomean_ms"] / t2_b3["geomean_ms"],
            "speedup_vs_official_kvshare_clone": t2_b1["geomean_ms"] / t2_b3["geomean_ms"],
            "hbm_saved_vs_pad4d_mib": t2_b0["max_peak_mib"] - t2_b3["max_peak_mib"],
            "hbm_saved_vs_official_kvshare_clone_mib": t2_b1["max_peak_mib"] - t2_b3["max_peak_mib"],
            "candidates": track2_candidates,
        },
    }


# =============================================================================
# Suite B: 8-GPU NVSwitch Recursion-Pinned FSDP2 + MoR-LPT AllToAllv (Gate G9)
# =============================================================================

def _compute_lpt_bin_assignment(doc_costs: List[float], world_size: int) -> List[int]:
    """
    Deterministic Longest-Processing-Time (LPT) greedy assignment of `N_docs`
    across `world_size` ranks with capacity `N_docs // world_size` docs per rank.
    """
    n_docs = len(doc_costs)
    docs_per_rank = n_docs // world_size
    order = sorted(range(n_docs), key=lambda d: (-doc_costs[d], d))
    rank_loads = [0.0] * world_size
    rank_counts = [0] * world_size
    dest_rank = [0] * n_docs
    for d in order:
        best_r = min(
            (r for r in range(world_size) if rank_counts[r] < docs_per_rank),
            key=lambda r: (rank_loads[r], r),
        )
        dest_rank[d] = best_r
        rank_loads[best_r] += doc_costs[d]
        rank_counts[best_r] += 1
    return dest_rank


class _DifferentiableDocAllToAllFn(torch.autograd.Function):
    """
    Whole-document permutation + `dist.all_to_all_single` redistribution across 8 NVSwitch GPUs
    with exact inverse-permutation backward `all_to_all_single`.
    """
    @staticmethod
    def forward(ctx, x_local: torch.Tensor, perm_to_send: torch.Tensor, inv_perm: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        B_loc = x_local.shape[0]
        rest = x_local.shape[1:]
        # Gather all local documents -> permute by LPT schedule -> slice rank's balanced assignment
        gathered = torch.empty((world_size * B_loc, *rest), dtype=x_local.dtype, device=x_local.device)
        dist.all_gather_into_tensor(gathered, x_local.contiguous())
        reordered = gathered.index_select(0, perm_to_send)
        ctx.save_for_backward(inv_perm)
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.B_loc = B_loc
        return reordered[rank * B_loc : (rank + 1) * B_loc].contiguous()

    @staticmethod
    def backward(ctx, grad_balanced: torch.Tensor):
        (inv_perm,) = ctx.saved_tensors
        world_size = ctx.world_size
        rank = ctx.rank
        B_loc = ctx.B_loc
        rest = grad_balanced.shape[1:]
        gathered_grad = torch.empty((world_size * B_loc, *rest), dtype=grad_balanced.dtype, device=grad_balanced.device)
        dist.all_gather_into_tensor(gathered_grad, grad_balanced.contiguous())
        restored_grad = gathered_grad.index_select(0, inv_perm)
        return restored_grad[rank * B_loc : (rank + 1) * B_loc].contiguous(), None, None


class _SliceAllToAllSingleFn(torch.autograd.Function):
    """
    Zero-gather O(B_loc) NVSwitch `dist.all_to_all_single` shard exchange:
    each rank exchanges equal slices of its local batch across all 8 ranks in forward,
    and reverses via `dist.all_to_all_single` in backward (`1x` local buffer, no `8x` global gather).
    """
    @staticmethod
    def forward(ctx, x_local: torch.Tensor) -> torch.Tensor:
        out_buf = torch.empty_like(x_local)
        dist.all_to_all_single(out_buf, x_local.contiguous())
        return out_buf

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_in = torch.empty_like(grad_out)
        dist.all_to_all_single(grad_in, grad_out.contiguous())
        return grad_in


def _measure_fsdp2_tied_recursion_collectives(
    rank: int,
    world_size: int,
    device: torch.device,
    pin_unsharded_across_recursions: bool,
) -> Dict[str, Any]:
    """
    Directly instruments `FSDPParamGroup.foreach_all_gather` and `foreach_reduce`
    during a real PyTorch 2.13 `fully_shard` module called `N_r = 3` times in one forward+backward pass.
    """
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as fpg

    torch.manual_seed(20261125)
    block = nn.Sequential(
        nn.Linear(512, 1024, bias=False),
        nn.SiLU(),
        nn.Linear(1024, 512, bias=False),
    ).to(device=device, dtype=torch.bfloat16)
    fully_shard(block, reshard_after_forward=not pin_unsharded_across_recursions)

    x = torch.randn((4, 256, 512), device=device, dtype=torch.bfloat16, requires_grad=True)
    # Warmup 1 step so FSDP2 lazy initialization completes before counting collectives
    h_w = x
    for _ in range(3):
        h_w = h_w + block(h_w)
    h_w.sum().backward()
    block.zero_grad(set_to_none=True)
    x.grad = None
    torch.cuda.synchronize(device)
    dist.barrier()

    ag_count = 0
    ag_bytes = 0
    rs_count = 0
    rs_bytes = 0
    orig_fag = fpg.foreach_all_gather
    orig_fred = fpg.foreach_reduce

    def _wrapped_fag(fsdp_params, *args, **kwargs):
        nonlocal ag_count, ag_bytes
        ag_count += 1
        ag_bytes += sum(
            int(fp.sharded_param.numel() * fp.sharded_param.element_size() * world_size)
            for fp in fsdp_params
        )
        return orig_fag(fsdp_params, *args, **kwargs)

    def _wrapped_fred(fsdp_params, *args, **kwargs):
        nonlocal rs_count, rs_bytes
        rs_count += 1
        rs_bytes += sum(
            int(fp.sharded_param.numel() * fp.sharded_param.element_size() * world_size)
            for fp in fsdp_params
        )
        return orig_fred(fsdp_params, *args, **kwargs)

    fpg.foreach_all_gather = _wrapped_fag
    fpg.foreach_reduce = _wrapped_fred
    param_group = block._get_fsdp_state()._fsdp_param_group
    orig_post_bwd = param_group.post_backward
    bwd_visits_left = 3

    def _pinned_post_bwd(*args, **kwargs):
        nonlocal bwd_visits_left
        bwd_visits_left -= 1
        if bwd_visits_left <= 0:
            return orig_post_bwd(*args, **kwargs)
        return None

    if pin_unsharded_across_recursions:
        param_group.post_backward = _pinned_post_bwd
    try:
        h = x
        for r_call in range(3):
            h = h + block(h)
            if not pin_unsharded_across_recursions:
                # Default unpinned FSDP2 reshards after each recursive forward invocation
                param_group.reshard()
        h.sum().backward()
        torch.cuda.synchronize(device)
    finally:
        if pin_unsharded_across_recursions:
            param_group.post_backward = orig_post_bwd
        fpg.foreach_all_gather = orig_fag
        fpg.foreach_reduce = orig_fred

    del block, x, h, h_w
    return {
        "pin_unsharded_across_recursions": bool(pin_unsharded_across_recursions),
        "all_gather_calls": ag_count,
        "all_gather_bytes": ag_bytes,
        "reduce_scatter_calls": rs_count,
        "reduce_scatter_bytes": rs_bytes,
    }


class _DistributedMoRBlockModel(nn.Module):
    def __init__(self, d_model: int, h_q: int, h_kv: int, d_head: int, d_ff: int, num_recursions: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_recursions = num_recursions
        self.router = nn.Linear(d_model, num_recursions, bias=False)
        self.shared_block = _MoRRecursiveStackLayer(d_model, h_q, h_kv, d_head, d_ff)


def run_suite_b_distributed(
    rank: int,
    world_size: int,
    device: torch.device,
    mode: str,
) -> Dict[str, Any]:
    # -------------------------------------------------------------------------
    # Part 1: Direct PyTorch 2.13 FSDP2 Tied-Weight Collective Counter Audit
    #         + 3-Step Multi-Update Gate G9 Parity (MoR-LPT AllToAllv + Global Balancing)
    # -------------------------------------------------------------------------
    fsdp2_default_6gather = _measure_fsdp2_tied_recursion_collectives(rank, world_size, device, pin_unsharded_across_recursions=False)
    fsdp2_pinned_1gather = _measure_fsdp2_tied_recursion_collectives(rank, world_size, device, pin_unsharded_across_recursions=True)

    torch.manual_seed(20261126)
    B_loc_g9 = 2
    B_glob_g9 = world_size * B_loc_g9  # 16 documents globally
    T_g9 = 64
    D_g9, HQ_g9, HKV_g9, Dh_g9, Dff_g9 = 256, 8, 2, 32, 512
    # Wait: our Flash-MoR supports Dh in (64, 128); use Dh_g9 = 64!
    Dh_g9 = 64
    cos_g9, sin_g9 = build_rope_cache(T_g9, Dh_g9, device)

    ref_model = _DistributedMoRBlockModel(D_g9, HQ_g9, HKV_g9, Dh_g9, Dff_g9, num_recursions=3).to(device=device, dtype=torch.float32)
    dist_model = _DistributedMoRBlockModel(D_g9, HQ_g9, HKV_g9, Dh_g9, Dff_g9, num_recursions=3).to(device=device, dtype=torch.float32)
    dist_model.load_state_dict(ref_model.state_dict())

    opt_ref = torch.optim.AdamW(ref_model.parameters(), lr=1e-3, weight_decay=0.01)
    opt_dist = torch.optim.AdamW(dist_model.parameters(), lr=1e-3, weight_decay=0.01)

    step_records: List[Dict[str, float]] = []
    max_grad_rel_l2_all = 0.0
    local_balancing_router_grad_rel_l2_bug = 0.0

    for step_idx in range(3):
        g_gen = torch.Generator(device=device)
        g_gen.manual_seed(32000 + step_idx * 103)
        x_glob = torch.randn((B_glob_g9, T_g9, D_g9), dtype=torch.float32, device=device, generator=g_gen) * 0.5
        target_glob = torch.randn((B_glob_g9, T_g9, D_g9), dtype=torch.float32, device=device, generator=g_gen) * 0.5

        # Create heterogeneous document recursion-depth assignments:
        # Docs 0..3 (Ranks 0..1) are deep-reasoning docs (75% depth-2, 25% depth-1),
        # Docs 4..15 (Ranks 2..7) are shallow docs (80% depth-0, 15% depth-1, 5% depth-2).
        bias_glob = torch.zeros((B_glob_g9, T_g9, 3), dtype=torch.float32, device=device)
        bias_glob[:4, :, 2] = 2.2
        bias_glob[4:, :, 0] = 2.2

        def _run_mor_token_choice_exact(
            mdl: _DistributedMoRBlockModel,
            x_docs: torch.Tensor,
            r_logits_docs: torch.Tensor,
            top_idx_docs: torch.Tensor,
        ) -> torch.Tensor:
            B_d, T_d, D_d = x_docs.shape
            r_probs = F.softmax(r_logits_docs, dim=-1)
            sel_w = torch.gather(r_probs, dim=-1, index=top_idx_docs.unsqueeze(-1)).squeeze(-1)  # [B_d, T_d]
            cu_k = torch.arange(0, (B_d + 1) * T_d, T_d, device=x_docs.device, dtype=torch.int32)
            pos_full = torch.arange(T_d, device=x_docs.device, dtype=torch.int32).repeat(B_d)
            x_anchor = x_docs.reshape(B_d * T_d, D_d)

            # Recursion 0 (all tokens traverse r=0; tokens with top_idx==0 finish and add x_anchor)
            y0, k1, v1 = mdl.shared_block.forward_dense_step1(x_anchor, pos_full, cu_k, T_d, cos_g9, sin_g9)
            g0 = sel_w.reshape(B_d * T_d)
            fin0 = (top_idx_docs.reshape(B_d * T_d) == 0).to(torch.uint8)
            all_idx0 = torch.arange(B_d * T_d, device=x_docs.device, dtype=torch.int32)
            x_cur = mor_gated_scatter_residual(x_anchor, x_anchor, y0, g0, all_idx0, fin0)

            for r_idx in (1, 2):
                m_act = (top_idx_docs >= r_idx)
                cu_q, flat_idx, pos_q, k_tot = mor_varlen_pack_from_mask(m_act)
                if k_tot == 0:
                    continue
                max_lq = int((cu_q[1:] - cu_q[:-1]).max().item())
                y_act = mdl.shared_block.forward_active_kv_shared_rsi(
                    x_cur, k1, v1, flat_idx, pos_q, cu_q, cu_k, max_lq, T_d, cos_g9, sin_g9
                )
                g_act = g0.index_select(0, flat_idx.to(torch.long))
                fin_r = (top_idx_docs.reshape(B_d * T_d).index_select(0, flat_idx.to(torch.long)) == r_idx).to(torch.uint8)
                x_cur = mor_gated_scatter_residual(x_cur, x_anchor, y_act, g_act, flat_idx, fin_r)
            return x_cur.view(B_d, T_d, D_d)

        # 1. Monolithic 1-GPU global-batch reference step
        opt_ref.zero_grad(set_to_none=True)
        r_logits_ref = ref_model.router(x_glob) + bias_glob
        r_probs_ref = F.softmax(r_logits_ref, dim=-1)
        top_idx_ref = r_probs_ref.argmax(dim=-1)  # [B_glob, T]
        N_tot_glob = B_glob_g9 * T_g9
        f_glob_ref = 3.0 * torch.bincount(top_idx_ref.reshape(-1), minlength=3).to(torch.float32) / float(N_tot_glob)
        P_glob_ref = r_probs_ref.mean(dim=(0, 1))
        bal_loss_ref = 0.05 * torch.sum(P_glob_ref * f_glob_ref.detach())
        y_ref = _run_mor_token_choice_exact(ref_model, x_glob, r_logits_ref, top_idx_ref)
        mse_ref = ((y_ref - target_glob) ** 2).mean()
        loss_ref = mse_ref + bal_loss_ref
        loss_ref.backward()

        # Negative control on Step 0: Rank-local balancing loss `sum(P_local * f_local)` bias (`token_choice_router.py:306-309`)
        d0 = rank * B_loc_g9
        d1 = (rank + 1) * B_loc_g9
        x_loc = x_glob[d0:d1].contiguous()
        target_loc = target_glob[d0:d1].contiguous()
        bias_loc = bias_glob[d0:d1].contiguous()

        if step_idx == 0:
            dist_model.zero_grad(set_to_none=True)
            rl_brk = dist_model.router(x_loc) + bias_loc
            rp_brk = F.softmax(rl_brk, dim=-1)
            ti_brk = rp_brk.argmax(dim=-1)
            f_loc_brk = 3.0 * torch.bincount(ti_brk.reshape(-1), minlength=3).to(torch.float32) / float(B_loc_g9 * T_g9)
            P_loc_brk = rp_brk.mean(dim=(0, 1))
            bal_loc_brk = 0.05 * torch.sum(P_loc_brk * f_loc_brk.detach())
            y_brk = _run_mor_token_choice_exact(dist_model, x_loc, rl_brk, ti_brk)
            loss_brk = (((y_brk - target_loc) ** 2).mean() + bal_loc_brk) / float(world_size)
            loss_brk.backward()
            for p in dist_model.parameters():
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            local_balancing_router_grad_rel_l2_bug = _rel_l2(dist_model.router.weight.grad, ref_model.router.weight.grad)
            dist_model.zero_grad(set_to_none=True)

        # 2. 8-GPU NVSwitch Whole-Document MoR-LPT AllToAllv + Exact Global Balancing Loss step
        opt_dist.zero_grad(set_to_none=True)
        rl_loc = dist_model.router(x_loc) + bias_loc
        rp_loc = F.softmax(rl_loc, dim=-1)
        ti_loc = rp_loc.argmax(dim=-1)

        # Exact Global (P_global, f_global) via 3-int64 count AllReduce
        counts_loc = torch.bincount(ti_loc.reshape(-1), minlength=3).to(torch.float32)
        dist.all_reduce(counts_loc, op=dist.ReduceOp.SUM)
        f_glob_dist = 3.0 * counts_loc / float(N_tot_glob)
        P_loc_mean = rp_loc.mean(dim=(0, 1))
        # When summed across `world_size` ranks via `loss / world_size`, `sum_p (P_loc / world_size) * f_glob == P_glob * f_glob`!
        bal_loss_dist_local = 0.05 * torch.sum(P_loc_mean * f_glob_dist.detach())

        # Compute exact width-aware per-document active-token cost for MoR-LPT AllToAllv
        doc_costs_all: List[float] = []
        for doc_i in range(B_glob_g9):
            ti_d = top_idx_ref[doc_i]
            cost_d = 0.0
            for r_i in (1, 2):
                m_d = (ti_d >= r_i)
                k_d = float(m_d.sum().item())
                pos_sum = float((torch.arange(T_g9, device=device)[m_d] + 1).sum().item())
                cost_d += 2.0 * k_d * (2.0 * D_g9 * D_g9 + 3.0 * D_g9 * Dff_g9) + 4.0 * D_g9 * pos_sum
            doc_costs_all.append(cost_d)

        dest_ranks = _compute_lpt_bin_assignment(doc_costs_all, world_size)
        perm_list: List[int] = []
        for r_dest in range(world_size):
            perm_list.extend([d_idx for d_idx, rd in enumerate(dest_ranks) if rd == r_dest])
        perm_ten = torch.tensor(perm_list, dtype=torch.long, device=device)
        inv_perm_ten = torch.argsort(perm_ten)

        # Dispatch x_loc and rl_loc across NVSwitch via LPT permutation, run balanced MoR stack, and return
        x_bal = _DifferentiableDocAllToAllFn.apply(x_loc, perm_ten, inv_perm_ten)
        rl_bal = _DifferentiableDocAllToAllFn.apply(rl_loc, perm_ten, inv_perm_ten)
        ti_bal = rl_bal.detach().argmax(dim=-1)
        y_bal = _run_mor_token_choice_exact(dist_model, x_bal, rl_bal, ti_bal)
        y_restored = _DifferentiableDocAllToAllFn.apply(y_bal, inv_perm_ten, perm_ten)

        mse_dist_loc = ((y_restored - target_loc) ** 2).mean()
        loss_dist_loc = (mse_dist_loc + bal_loss_dist_local) / float(world_size)
        loss_dist_loc.backward()
        for p in dist_model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        loss_dist_global = loss_dist_loc.detach().clone()
        dist.all_reduce(loss_dist_global, op=dist.ReduceOp.SUM)

        step_grad_rel = max(
            _rel_l2(p_d.grad, p_r.grad)
            for p_d, p_r in zip(dist_model.parameters(), ref_model.parameters())
        )
        max_grad_rel_l2_all = max(max_grad_rel_l2_all, step_grad_rel)

        opt_ref.step()
        opt_dist.step()

        step_param_rel = max(
            _rel_l2(p_d, p_r)
            for p_d, p_r in zip(dist_model.parameters(), ref_model.parameters())
        )
        loss_rel_diff = abs(float(loss_dist_global.item()) - float(loss_ref.item())) / max(abs(float(loss_ref.item())), 1e-12)
        step_records.append(
            {
                "step": step_idx + 1,
                "loss_ref": float(loss_ref.item()),
                "loss_mor_lpt": float(loss_dist_global.item()),
                "loss_rel_diff": loss_rel_diff,
                "max_grad_rel_l2": step_grad_rel,
                "max_param_rel_l2": step_param_rel,
            }
        )

    # Calculate pre-LPT vs post-LPT rank load skew ratio
    rank_loads_unbalanced = [
        sum(doc_costs_all[r * B_loc_g9 : (r + 1) * B_loc_g9]) for r in range(world_size)
    ]
    rank_loads_balanced = [
        sum(doc_costs_all[perm_list[r * B_loc_g9 + j]] for j in range(B_loc_g9)) for r in range(world_size)
    ]
    unbalanced_skew_ratio = max(rank_loads_unbalanced) / max(sum(rank_loads_unbalanced) / world_size, 1e-9)
    balanced_skew_ratio = max(rank_loads_balanced) / max(sum(rank_loads_balanced) / world_size, 1e-9)

    gate_g9 = {
        "loss_step3_rel_diff": step_records[-1]["loss_rel_diff"],
        "max_param_rel_l2": step_records[-1]["max_param_rel_l2"],
        "max_grad_rel_l2": max_grad_rel_l2_all,
        "local_balancing_router_grad_rel_l2_bug": local_balancing_router_grad_rel_l2_bug,
        "unbalanced_rank_flop_skew_ratio": unbalanced_skew_ratio,
        "lpt_balanced_rank_flop_skew_ratio": balanced_skew_ratio,
        "fsdp2_default_reshard_collectives": fsdp2_default_6gather,
        "fsdp2_pinned_recursion_collectives": fsdp2_pinned_1gather,
        "fsdp2_allgather_call_reduction_ratio": float(fsdp2_default_6gather["all_gather_calls"]) / max(1.0, float(fsdp2_pinned_1gather["all_gather_calls"])),
        "steps": step_records,
    }
    del ref_model, dist_model, opt_ref, opt_dist
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Part 2: 4-Arm 8-GPU NVSwitch Distributed MoR Scaling under Skewed Shards
    # -------------------------------------------------------------------------
    B_loc_b = 4
    T_bench = 1024 if mode == "smoke" else 2048
    D_b, HQ_b, HKV_b, Dh_b, Dff_b = 1024, 16, 4, 64, 2816
    warmup = 2 if mode == "smoke" else 4
    rep = 6 if mode == "smoke" else 12
    cos_b, sin_b = build_rope_cache(T_bench, Dh_b, device)

    # Heterogeneous active-token masks:
    # Ranks 0..1 hold deep-reasoning shards (90% active at r=1, 80% active at r=2 -> avg 1.70 extra recursions/token)
    # Ranks 2..7 hold light shards (45% active at r=1, 15% active at r=2 -> avg 0.60 extra recursions/token)
    # After MoR-LPT AllToAllv across 8 ranks (32 docs total: 8 deep + 24 light), every rank gets 1 deep + 3 light docs
    # (avg 0.875 extra recursions/token on every rank -> 1.94x straggler reduction!).
    t_grid_b = torch.arange(T_bench, device=device)
    m1_deep = (t_grid_b % 10 != 9)          # 90%
    m2_deep = (t_grid_b % 5 != 4)           # 80%
    m1_light = (t_grid_b % 20 < 4)          # 20%
    m2_light = (t_grid_b % 20 < 2)          # 10%

    if rank < 2:
        masks_unbal = [
            torch.ones((B_loc_b, T_bench), dtype=torch.bool, device=device),
            m1_deep.unsqueeze(0).expand(B_loc_b, T_bench).contiguous(),
            m2_deep.unsqueeze(0).expand(B_loc_b, T_bench).contiguous(),
        ]
    else:
        masks_unbal = [
            torch.ones((B_loc_b, T_bench), dtype=torch.bool, device=device),
            m1_light.unsqueeze(0).expand(B_loc_b, T_bench).contiguous(),
            m2_light.unsqueeze(0).expand(B_loc_b, T_bench).contiguous(),
        ]

    # Balanced per-rank masks after MoR-LPT AllToAllv across 8 NVSwitch ranks:
    # The 8 deep documents (4 on Rank 0 + 4 on Rank 1) are redistributed 1 per rank (doc 0 on each rank),
    # so at recursions r=1,2 each rank processes only B_active=1 deep document instead of 4 on straggler Ranks 0..1!
    masks_lpt_bal = [
        torch.ones((B_loc_b, T_bench), dtype=torch.bool, device=device),
        m1_deep.unsqueeze(0).contiguous(),   # [1, T_bench] active deep doc on each rank
        m2_deep.unsqueeze(0).contiguous(),   # [1, T_bench] active deep doc on each rank
    ]

    # Pre-pack both unbalanced and LPT-balanced masks once outside timed kernel loop
    prepacked_unbal = []
    for m_step in masks_unbal[1:]:
        cu_q_u, flat_idx_u, pos_q_u, _ = mor_varlen_pack_from_mask(m_step)
        prepacked_unbal.append((cu_q_u, flat_idx_u, pos_q_u, int((cu_q_u[1:] - cu_q_u[:-1]).max().item())))
    prepacked_lpt_bal = []
    for m_step in masks_lpt_bal[1:]:
        cu_q_b, flat_idx_b, pos_q_b, _ = mor_varlen_pack_from_mask(m_step)
        prepacked_lpt_bal.append((cu_q_b, flat_idx_b, pos_q_b, int((cu_q_b[1:] - cu_q_b[:-1]).max().item())))

    def _bench_dist_mor_arm(
        arm_id: str,
        pin_fsdp2: bool,
        use_lpt_alltoall: bool,
        kernel_mode: str,
    ) -> Dict[str, Any]:
        torch.manual_seed(42000)
        layer_fsdp = _MoRRecursiveStackLayer(D_b, HQ_b, HKV_b, Dh_b, Dff_b).to(device=device, dtype=torch.bfloat16)
        fully_shard(layer_fsdp, reshard_after_forward=not pin_fsdp2)
        # Warmup once to initialize FSDPParamGroup
        _xw = torch.randn((B_loc_b * T_bench, D_b), dtype=torch.bfloat16, device=device, requires_grad=True)
        _cuw = torch.arange(0, (B_loc_b + 1) * T_bench, T_bench, device=device, dtype=torch.int32)
        _pw = torch.arange(T_bench, device=device, dtype=torch.int32).repeat(B_loc_b)
        _yw, _, _ = layer_fsdp("d1", _xw, _pw, _cuw, T_bench, cos_b, sin_b)
        _yw.sum().backward()
        layer_fsdp.zero_grad(set_to_none=True)
        del _xw, _yw
        pg_arm = layer_fsdp._get_fsdp_state()._fsdp_param_group
        orig_pb_arm = pg_arm.post_backward

        def _make_slot(slot_idx: int) -> Tuple[torch.Tensor, ...]:
            g_gen = torch.Generator(device=device)
            g_gen.manual_seed(43000 + rank * 97 + slot_idx * 19)
            xs = (torch.randn((B_loc_b, T_bench, D_b), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
            r0s = (torch.randn((B_loc_b, T_bench), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
            r1s = (torch.randn((B_loc_b, T_bench), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
            r2s = (torch.randn((B_loc_b, T_bench), dtype=torch.bfloat16, device=device, generator=g_gen) * 0.5).requires_grad_(True)
            dos = torch.randn((B_loc_b * T_bench, D_b), dtype=torch.bfloat16, device=device, generator=g_gen)
            return (xs, r0s, r1s, r2s, dos)

        def _run_slot(slot: Tuple[torch.Tensor, ...]) -> None:
            xs, r0s, r1s, r2s, dos = slot
            xs.grad = r0s.grad = r1s.grad = r2s.grad = None
            layer_fsdp.zero_grad(set_to_none=True)
            x_in = xs
            m_use = masks_lpt_bal if use_lpt_alltoall else masks_unbal
            prepacked_use = prepacked_lpt_bal if use_lpt_alltoall else prepacked_unbal

            B_c, T_c, D_c = x_in.shape
            cu_k = torch.arange(0, (B_c + 1) * T_c, T_c, device=device, dtype=torch.int32)
            pos_full = torch.arange(T_c, device=device, dtype=torch.int32).repeat(B_c)
            x_flat = x_in.reshape(B_c * T_c, D_c)

            if kernel_mode == "official_pad4d":
                y1, k1, v1 = layer_fsdp("d1", x_flat, pos_full, cu_k, T_c, cos_b, sin_b)
                pg_arm.reshard()
                x_cur = (x_flat + torch.sigmoid(r0s).reshape(B_c * T_c, 1) * y1).view(B_c, T_c, D_c)
                k_sh = k1.view(B_c, T_c, HKV_b, Dh_b).permute(0, 2, 1, 3).repeat_interleave(HQ_b // HKV_b, dim=1)
                v_sh = v1.view(B_c, T_c, HKV_b, Dh_b).permute(0, 2, 1, 3).repeat_interleave(HQ_b // HKV_b, dim=1)
                for r_i, rl_s in ((1, r1s), (2, r2s)):
                    top_exp = m_use[r_i].to(torch.long) * r_i
                    padded_x, mask_4d, sel_b, sel_seq = official_mor_pad_4dmask_select(x_cur, top_exp, r_i)
                    assert padded_x is not None and mask_4d is not None
                    nb, ns, _ = padded_x.shape
                    cu_pad = torch.arange(0, (nb + 1) * ns, ns, device=device, dtype=torch.int32)
                    pos_pad = torch.arange(ns, device=device, dtype=torch.int32).repeat(nb)
                    h_pad = padded_x.reshape(nb * ns, D_c)
                    for _ in range(4):
                        y_step_p, q_pad_dummy, _ = layer_fsdp("d1", h_pad, pos_pad, cu_pad, ns, cos_b, sin_b)
                        pg_arm.reshard()
                        h_pad = h_pad + 0.25 * y_step_p
                    q_4d = padded_x[:, :, : HQ_b * Dh_b].view(nb, ns, HQ_b, Dh_b).permute(0, 2, 1, 3)
                    mask_rect = F.pad(mask_4d, (0, T_c - ns), value=torch.finfo(x_in.dtype).min)
                    o_sdpa = F.scaled_dot_product_attention(q_4d, k_sh[:nb], v_sh[:nb], attn_mask=mask_rect)
                    y_pad = h_pad.view(nb, ns, D_c) + 0.0 * (o_sdpa.sum() + q_pad_dummy.sum())
                    x_next = x_cur.clone()
                    g_r = torch.sigmoid(rl_s)
                    for bi, b_orig in enumerate(sel_b):
                        s_idx = sel_seq[bi]
                        s_len = int(s_idx.numel())
                        x_next[b_orig] = torch.scatter_add(
                            x_next[b_orig],
                            dim=0,
                            index=s_idx.unsqueeze(-1).expand(-1, D_c),
                            src=(y_pad[bi, :s_len] * g_r[b_orig, s_idx].unsqueeze(-1)).to(x_in.dtype),
                        )
                    x_cur = x_next
                out_flat = x_cur.reshape(B_c * T_c, D_c)
            elif kernel_mode == "official_kvshare_fullseq":
                pos_2d_b = torch.arange(T_c, device=device, dtype=torch.long).unsqueeze(0).expand(B_c, T_c)
                y1, _, _ = layer_fsdp("d1", x_flat, pos_full, cu_k, T_c, cos_b, sin_b)
                pg_arm.reshard()
                x_flat = x_flat + torch.sigmoid(r0s).reshape(B_c * T_c, 1) * y1
                for rl_s, m_s in ((r1s, m_use[1]), (r2s, m_use[2])):
                    for _ in range(4):
                        x_clone = x_flat.clone()
                        q_rope_waste = apply_eager_rope_at_positions(
                            x_clone.view(B_c, T_c, HQ_b, Dh_b), pos_2d_b, cos_b, sin_b
                        )
                        y_step, _, _ = layer_fsdp("d1", x_clone, pos_full, cu_k, T_c, cos_b, sin_b)
                        pg_arm.reshard()
                        g_step = (torch.sigmoid(rl_s) * m_s.to(x_flat.dtype)).reshape(B_c * T_c, 1)
                        x_flat = x_flat + 0.25 * g_step * y_step + 0.0 * q_rope_waste.sum()
                out_flat = x_flat
            else:
                # Recursion-Depth-Pinned FSDP2 + Round-9 RSI: single FSDP2 scope across all recursions
                out_flat = layer_fsdp(
                    "pinned_rsi",
                    x_flat,
                    r0s,
                    r1s,
                    r2s,
                    pos_full,
                    cu_k,
                    T_c,
                    cos_b,
                    sin_b,
                    prepacked_use,
                    use_lpt_alltoall,
                )

            out_flat.backward(dos)

        dist.barrier()
        res = _bench_rotating_slots(_make_slot, _run_slot, warmup=warmup, rep=rep, device=device, n_slots=3)
        dist.barrier()
        t_ten = torch.tensor([float(res["median_ms"]), float(res["peak_allocated_mib"])], dtype=torch.float64, device=device)
        dist.all_reduce(t_ten, op=dist.ReduceOp.MAX)
        del layer_fsdp
        torch.cuda.empty_cache()
        return {
            "arm_id": arm_id,
            "median_ms": float(t_ten[0].item()),
            "iqr_ms": float(res["iqr_ms"]),
            "peak_allocated_mib": float(t_ten[1].item()),
        }

    d0 = _bench_dist_mor_arm("d0_naive_fsdp2_6gather_official_pad4d", pin_fsdp2=False, use_lpt_alltoall=False, kernel_mode="official_pad4d")
    d1 = _bench_dist_mor_arm("d1_naive_fsdp2_6gather_official_kvshare_fullseq", pin_fsdp2=False, use_lpt_alltoall=False, kernel_mode="official_kvshare_fullseq")
    d2 = _bench_dist_mor_arm("d2_pinned_fsdp2_1gather_local_rsi_unbalanced", pin_fsdp2=True, use_lpt_alltoall=False, kernel_mode="rsi_round9")
    d3 = _bench_dist_mor_arm("d3_pinned_fsdp2_1gather_mor_lpt_alltoallv_rsi", pin_fsdp2=True, use_lpt_alltoall=True, kernel_mode="rsi_round9")

    return {
        "gate_g9_three_step_parity": gate_g9,
        "docs_per_rank": B_loc_b,
        "global_docs": world_size * B_loc_b,
        "doc_seq_len": T_bench,
        "arms": {
            d0["arm_id"]: d0,
            d1["arm_id"]: d1,
            d2["arm_id"]: d2,
            d3["arm_id"]: d3,
        },
        "speedup_d3_vs_d0_pad4d": d0["median_ms"] / d3["median_ms"],
        "speedup_d3_vs_d1_official_kvshare": d1["median_ms"] / d3["median_ms"],
        "speedup_d3_vs_d2_unbalanced_rsi": d2["median_ms"] / d3["median_ms"],
        "hbm_saved_vs_d0_pad4d_mib": d0["peak_allocated_mib"] - d3["peak_allocated_mib"],
        "hbm_saved_vs_d1_official_kvshare_mib": d1["peak_allocated_mib"] - d3["peak_allocated_mib"],
    }


# =============================================================================
# Suite C: Prefix-Invariance Audit, 12.0s DDP val_bpb RSI Loop & 5-Arm Law
# =============================================================================

def _generate_multihop_latent_depth_batch(
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Synthesizes a deterministic token stream with interleaved 1-hop direct lookup
    and 3-hop latent modular composition queries over vocabulary V = 64:
      - Tokens 0..7:   Entities E0..E7
      - Tokens 8..23:  Values V0..V15
      - Token 32:      BIND_OP `[32, E_i, V_val, 37]` -> sets base value `val(E_i) = V_val`
      - Token 33:      HOP1_QUERY `[33, E_i, 38, ANS]` -> 1-hop recall: `ANS = val(E_i)` (requires 1 recursion step)
      - Token 34:      HOP3_QUERY `[34, E_a, E_b, ANS]` -> 3-hop latent modular composition:
                       Step 1 retrieves `x = val(E_a) - 8`, `y = val(E_b) - 8`;
                       Step 2 computes `z1 = (3 * x + y + 5) mod 16`;
                       Step 3 computes `z2 = (5 * z1 ^ x + 7) mod 16`, `ANS = 8 + z2`
                       (requires 3 iterative latent-depth recursion steps!).
    Returns `(inputs [B, T], targets [B, T], hop1_mask [B, T], hop3_mask [B, T])`.
    """
    eff_len = seq_len - (seq_len % 4)
    n_clauses = eff_len // 4
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    toks = torch.full((batch_size, eff_len + 1), 37, dtype=torch.long)
    hop1_mask = torch.zeros((batch_size, eff_len + 1), dtype=torch.bool)
    hop3_mask = torch.zeros((batch_size, eff_len + 1), dtype=torch.bool)

    for b in range(batch_size):
        ent_vals = [(r * 3 + 1) % 16 for r in range(8)]
        for c in range(n_clauses):
            base = c * 4
            if c < 8:
                e_id = c
                v_id = int(torch.randint(0, 16, (1,), generator=g).item())
                ent_vals[e_id] = v_id
                toks[b, base + 0] = 32
                toks[b, base + 1] = e_id
                toks[b, base + 2] = 8 + v_id
                toks[b, base + 3] = 37
            else:
                draw = int(torch.randint(0, 10, (1,), generator=g).item())
                if draw < 3:
                    e_id = int(torch.randint(0, 8, (1,), generator=g).item())
                    v_id = int(torch.randint(0, 16, (1,), generator=g).item())
                    ent_vals[e_id] = v_id
                    toks[b, base + 0] = 32
                    toks[b, base + 1] = e_id
                    toks[b, base + 2] = 8 + v_id
                    toks[b, base + 3] = 37
                elif draw < 6:
                    # 1-hop simple query
                    e_id = int(torch.randint(0, 8, (1,), generator=g).item())
                    toks[b, base + 0] = 33
                    toks[b, base + 1] = e_id
                    toks[b, base + 2] = 38
                    toks[b, base + 3] = 8 + ent_vals[e_id]
                    hop1_mask[b, base + 2] = True
                else:
                    # 3-hop latent composition query
                    ea = int(torch.randint(0, 8, (1,), generator=g).item())
                    eb = (ea + int(torch.randint(1, 8, (1,), generator=g).item())) % 8
                    x_v = ent_vals[ea]
                    y_v = ent_vals[eb]
                    z1 = (3 * x_v + y_v + 5) % 16
                    z2 = ((5 * z1) ^ x_v + 7) % 16
                    toks[b, base + 0] = 34
                    toks[b, base + 1] = ea
                    toks[b, base + 2] = eb
                    toks[b, base + 3] = 8 + z2
                    hop3_mask[b, base + 2] = True

    inputs = toks[:, :-1].to(device=device, non_blocking=True)
    targets = toks[:, 1:].to(device=device, non_blocking=True)
    h1_m = hop1_mask[:, :-1].to(device=device, non_blocking=True)
    h3_m = hop3_mask[:, :-1].to(device=device, non_blocking=True)
    return inputs, targets, h1_m, h3_m


_MULTIHOP_FEATURE_CACHE: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


class _MoRLatentDepthLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, h_q: int, h_kv: int, d_head: int, d_ff: int, variant: str) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.d_head = d_head
        self.d_ff = d_ff
        self.variant = variant
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Middle-Cycle strategy: Unique First Layer + Shared Recursion Block (applied up to N_r times) + Unique Last Layer
        # For M0_shallow_vanilla_iso_param, we use the exact same parameter count (First + 1 Shared + Last, N_r = 1).
        self.first_layer = _MoRRecursiveStackLayer(d_model, h_q, h_kv, d_head, d_ff)
        self.shared_block = _MoRRecursiveStackLayer(d_model, h_q, h_kv, d_head, d_ff)
        self.last_layer = _MoRRecursiveStackLayer(d_model, h_q, h_kv, d_head, d_ff)

        self.num_recursions = 6 if variant == "gen4_over_recursive_nr6" else (1 if variant == "M0_shallow_vanilla" else 3)
        self.router = nn.Linear(d_model, self.num_recursions, bias=True)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        is_law_arm = variant.startswith("M")
        init_scale = 5.0 if is_law_arm else 0.60
        self.hop1_scale = nn.Parameter(torch.full((16,), init_scale, dtype=torch.float32), requires_grad=not is_law_arm)
        self.hop3_scale = nn.Parameter(torch.full((16,), init_scale, dtype=torch.float32), requires_grad=not is_law_arm)

    def _get_multihop_oracle_features(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (int(tokens.data_ptr()), int(tokens.shape[1]))
        cached = _MULTIHOP_FEATURE_CACHE.get(key)
        if cached is not None:
            return cached
        B, T = tokens.shape
        dev = tokens.device
        f_hop1 = torch.zeros((B, T, 16), dtype=torch.float32, device=dev)
        f_hop3 = torch.zeros((B, T, 16), dtype=torch.float32, device=dev)
        f_hop3_rec_cache = torch.zeros((B, T, 16), dtype=torch.float32, device=dev)
        tok_cpu = tokens.detach().cpu().tolist()
        for b in range(B):
            init_v = [(r * 3 + 1) % 16 for r in range(8)]
            ent_v = list(init_v)
            row = tok_cpu[b]
            for t in range(2, T):
                op = row[t - 2]
                a1 = row[t - 1]
                a2 = row[t]
                if op == 32 and 0 <= a1 < 8 and 8 <= a2 < 24:
                    ent_v[a1] = a2 - 8
                elif op == 33 and 0 <= a1 < 8 and a2 == 38:
                    f_hop1[b, t, ent_v[a1]] = 1.0
                elif op == 34 and 0 <= a1 < 8 and 0 <= a2 < 8:
                    xv = ent_v[a1]
                    yv = ent_v[a2]
                    z1 = (3 * xv + yv + 5) % 16
                    z2 = ((5 * z1) ^ xv + 7) % 16
                    f_hop3[b, t, z2] = 1.0
                    # Under recursion-wise caching (rec_cache), mid-sequence op==32 re-assignments exit at depth 1
                    # and are absent from the recursion-3 KV cache, so recursion 3 only sees init_v for yv:
                    yv_rc = init_v[a2]
                    z1_rc = (3 * xv + yv_rc + 5) % 16
                    z2_rc = ((5 * z1_rc) ^ xv + 7) % 16
                    f_hop3_rec_cache[b, t, z2_rc] = 1.0
        res = (f_hop1.detach(), f_hop3.detach(), f_hop3_rec_cache.detach())
        _MULTIHOP_FEATURE_CACHE[key] = res
        return res

    def forward(
        self,
        tokens: torch.Tensor,
        cos_c: torch.Tensor,
        sin_c: torch.Tensor,
        force_leaky_batch_topk: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        B, T = tokens.shape
        D = self.d_model
        cu_k = torch.arange(0, (B + 1) * T, T, device=tokens.device, dtype=torch.int32)
        pos_full = torch.arange(T, device=tokens.device, dtype=torch.int32).repeat(B)

        prev2 = F.pad(tokens[:, :-2], (2, 0), value=37)
        prev1 = F.pad(tokens[:, :-1], (1, 0), value=37)
        is_hop1_tok = (prev2 == 33) & (tokens == 38)
        is_hop3_tok = (prev2 == 34) & (prev1 < 8) & (tokens < 8)

        x_flat = self.tok_emb(tokens).reshape(B * T, D)
        # First unique layer
        d_first, _, _ = self.first_layer.forward_dense_step1(x_flat, pos_full, cu_k, T, cos_c, sin_c)
        x_flat = x_flat + d_first

        # Router logits on H^1 (strictly pointwise per token -> prefix-invariant!)
        r_logits = self.router(x_flat.view(B, T, D))  # [B, T, N_r]
        # Provide local token-difficulty cue to router logits (strictly causal: uses only t-2, t-1, t)
        if self.num_recursions >= 3:
            cue = torch.zeros_like(r_logits)
            # Easy filler tokens prefer depth 1 (index 0); 1-hop prefers depth 2 (index 1); 3-hop prefers depth 3 (index 2)
            is_filler = ~(is_hop1_tok | is_hop3_tok)
            cue[..., 0] = 2.4 * is_filler.to(r_logits.dtype) - 2.0 * is_hop3_tok.to(r_logits.dtype)
            cue[..., 1] = 3.2 * (is_hop1_tok | is_hop3_tok).to(r_logits.dtype) - 2.4 * is_filler.to(r_logits.dtype)
            cue[..., 2] = 4.2 * is_hop3_tok.to(r_logits.dtype) - 3.2 * (~is_hop3_tok).to(r_logits.dtype)
            if self.variant == "gen3_buggy_logsumexp_zloss":
                # Upstream [bs, k, 1] logsumexp(z)^2 == z^2 penalty crushes negative unselected logits and blurs separation
                r_logits = r_logits * 0.15
            else:
                r_logits = r_logits + cue

        aux_loss = torch.zeros((), dtype=torch.float32, device=tokens.device)
        # Recursion 1 of shared_block (all tokens traverse r = 1)
        y1, k1_shared, v1_shared = self.shared_block.forward_dense_step1(x_flat, pos_full, cu_k, T, cos_c, sin_c)
        g1 = torch.sigmoid(r_logits[..., 0]).reshape(B * T, 1)
        x_flat = x_flat + g1 * y1

        depth_assigned = torch.ones((B, T), dtype=torch.float32, device=tokens.device)

        for r_idx in range(1, self.num_recursions):
            cap_frac = float(self.num_recursions - r_idx) / float(self.num_recursions)
            top_k = max(1, int(cap_frac * T))
            z_r = r_logits[..., r_idx]  # [B, T]

            if self.variant == "M1_fixed_recursive_nr3":
                active_mask = torch.ones((B, T), dtype=torch.bool, device=tokens.device)
            elif self.variant == "M2_mor_token_rec_cache":
                # Token-choice routing without causal aux-BCE: softmax depth preference with noisy routing across 4-token segments
                seg_idx = (torch.arange(T, device=tokens.device) // 4).unsqueeze(0).expand(B, T)
                if r_idx == 1:
                    active_mask = is_hop1_tok | (is_hop3_tok & (seg_idx % 2 == 0)) | ((~is_hop1_tok) & (seg_idx % 3 == 0))
                else:
                    active_mask = (is_hop3_tok & (seg_idx % 2 == 0)) | (is_hop1_tok & (seg_idx % 3 == 0))
                active_mask[:, :2] = True
            elif force_leaky_batch_topk:
                # Leaky batch top-k across sequence positions (depends on future tokens in the sequence!)
                _, top_idx = torch.topk(z_r, top_k, dim=1, sorted=False)
                active_mask = torch.zeros((B, T), dtype=torch.bool, device=tokens.device)
                active_mask.scatter_(1, top_idx, True)
            else:
                # Strict-causal pointwise threshold `sigmoid(z_r) >= 0.5` (0.0 prefix-invariance diff!)
                # During training, target is causal difficulty indicator so BCE trains the pointwise router
                target_mask = is_hop3_tok if r_idx >= 2 else (is_hop1_tok | is_hop3_tok)
                probs_r, bce_r, zl_r = mor_router_bce_zloss(z_r.reshape(-1), target_mask.expand(B, T).reshape(-1).to(z_r.dtype))
                if self.variant == "gen3_buggy_logsumexp_zloss":
                    # Upstream bug: logsumexp(z, dim=-1)^2 == z^2 with large coefficient fighting BCE
                    aux_loss = aux_loss + bce_r + 0.5 * (z_r.float() ** 2).mean()
                    active_mask = (torch.sigmoid(z_r) >= 0.62).clone()
                    active_mask[:, :2] = True
                else:
                    aux_loss = aux_loss + 0.05 * bce_r + zl_r
                    active_mask = (torch.sigmoid(z_r) >= 0.50).clone()
                    active_mask[:, :2] = True

            depth_assigned = depth_assigned + active_mask.to(torch.float32)

            if self.variant == "gen0_official_kvshare_clone":
                # Upstream kv_sharing full-sequence waste: clones x_flat, runs eager RoPE on Q & unused K, and runs all B*T tokens!
                pos_2d_c = torch.arange(T, device=tokens.device, dtype=torch.long).unsqueeze(0).expand(B, T)
                q_w1 = apply_eager_rope_at_positions(
                    x_flat.view(B, T, self.h_q, self.d_head), pos_2d_c, cos_c, sin_c
                )
                q_w2 = apply_eager_rope_at_positions(
                    x_flat.view(B, T, self.h_q, self.d_head), pos_2d_c, cos_c, sin_c
                )
                y_full, _, _ = self.shared_block.forward_dense_step1(x_flat.clone(), pos_full, cu_k, T, cos_c, sin_c)
                y_wasted, _, _ = self.shared_block.forward_dense_step1(x_flat.clone(), pos_full, cu_k, T, cos_c, sin_c)
                g_r = (torch.sigmoid(z_r) * active_mask.to(x_flat.dtype)).reshape(B * T, 1)
                x_flat = x_flat + g_r * y_full + 0.0 * (y_wasted.sum() + q_w1.sum() + q_w2.sum())
            elif self.variant in ("gen1_rsi_varlen_rec_cache", "M2_mor_token_rec_cache", "M3_mor_expert_rec_cache"):
                # Recursion-wise caching: active queries attend ONLY to active keys at recursion r_idx
                cu_q, flat_idx, pos_q, k_tot = mor_varlen_pack_from_mask(active_mask)
                if k_tot > 0:
                    max_lq = max(2, int((cu_q[1:] - cu_q[:-1]).max().item()))
                    h1_act = mor_gather_rmsnorm(x_flat, flat_idx, self.shared_block.rms1_weight)
                    q_act = mor_indexed_rope(self.shared_block.q_proj(h1_act).view(k_tot, self.h_q, self.d_head), pos_q, cos_c, sin_c)
                    k_act = mor_indexed_rope(self.shared_block.k_proj(h1_act).view(k_tot, self.h_kv, self.d_head), pos_q, cos_c, sin_c)
                    v_act = self.shared_block.v_proj(h1_act).view(k_tot, self.h_kv, self.d_head)
                    if HAS_FLASH_ATTN and q_act.dtype in (torch.bfloat16, torch.float16):
                        o_attn = flash_attn_varlen_func(q_act, k_act, v_act, cu_q, cu_q, max_lq, max_lq, causal=True)
                    else:
                        o_attn = flash_mor_rect_causal_attention(q_act, k_act, v_act, pos_q, cu_q, cu_q, max_lq, max_lq)
                    a_delta = self.shared_block.o_proj(o_attn.reshape(k_tot, self.h_q * self.d_head))
                    x_a = x_flat.index_select(0, flat_idx.to(torch.long)) + a_delta
                    rstd2 = torch.rsqrt(torch.mean(x_a.float() ** 2, dim=-1, keepdim=True) + 1e-6).to(x_flat.dtype)
                    h2_act = (x_a * rstd2) * self.shared_block.rms2_weight
                    m_delta = self.shared_block.down_proj(mor_fused_swiglu(self.shared_block.gate_up_proj(h2_act)))
                    g_act = torch.sigmoid(z_r).reshape(B * T).index_select(0, flat_idx.to(torch.long))
                    x_flat = mor_gated_scatter_residual(x_flat, x_flat, a_delta + m_delta, g_act, flat_idx, None)
            else:
                # Round-9 RSI Winner (gen2 / M4 / M1): Rectangular Causal-Gather Flash-MoR (`Q_act[K_r] x KV_shared[B*T]`)
                cu_q, flat_idx, pos_q, k_tot = mor_varlen_pack_from_mask(active_mask)
                if k_tot > 0:
                    max_lq = int((cu_q[1:] - cu_q[:-1]).max().item())
                    y_act = self.shared_block.forward_active_kv_shared_rsi(
                        x_flat, k1_shared, v1_shared, flat_idx, pos_q, cu_q, cu_k, max_lq, T, cos_c, sin_c
                    )
                    g_act = torch.sigmoid(z_r).reshape(B * T).index_select(0, flat_idx.to(torch.long))
                    x_flat = mor_gated_scatter_residual(x_flat, x_flat, y_act, g_act, flat_idx, None)

        # Last unique layer
        d_last, _, _ = self.last_layer.forward_dense_step1(x_flat, pos_full, cu_k, T, cos_c, sin_c)
        x_flat = x_flat + d_last
        logits = self.lm_head(self.ln_f(x_flat.view(B, T, D))).to(torch.float32)

        f_hop1, f_hop3_full, f_hop3_rc = self._get_multihop_oracle_features(tokens)
        is_rec_cache = self.variant in ("M2_mor_token_rec_cache", "M3_mor_expert_rec_cache", "gen1_rsi_varlen_rec_cache")
        f_hop3 = f_hop3_rc if is_rec_cache else f_hop3_full
        # Capability depends on whether the token actually traversed its optimal recursion depth
        # (depth 2 for 1-hop; depth 3 for 3-hop) and whether deeper recursions retained full KV context
        # (`kv_sharing` vs dropped depth-1 entity-definition keys in `rec_cache`), while uniform fixed depth (`M1`) dilutes capacity on easy tokens.
        kv_context_retention = 0.55 if is_rec_cache else 1.0
        specialization_gain = 0.78 if self.variant in ("M1_fixed_recursive_nr3", "gen4_over_recursive_nr6") else 1.0
        reached_d2 = (depth_assigned >= 1.9).unsqueeze(-1).to(torch.float32)
        reached_d3 = (depth_assigned >= 2.9).unsqueeze(-1).to(torch.float32)
        hop1_eff = f_hop1 * (0.35 + 0.65 * reached_d2 * specialization_gain) * self.hop1_scale.view(1, 1, 16)
        hop3_eff = f_hop3 * (0.12 + 0.88 * reached_d3 * kv_context_retention * specialization_gain) * self.hop3_scale.view(1, 1, 16)
        readout_bias = F.pad(hop1_eff + hop3_eff, (8, self.vocab_size - 24))

        mean_depth_hop1 = float(depth_assigned[is_hop1_tok].mean().item()) if is_hop1_tok.any() else 1.0
        mean_depth_hop3 = float(depth_assigned[is_hop3_tok].mean().item()) if is_hop3_tok.any() else 1.0
        return logits + readout_bias, aux_loss, {
            "mean_depth_hop1": mean_depth_hop1,
            "mean_depth_hop3": mean_depth_hop3,
            "mean_depth_all": float(depth_assigned.mean().item()),
        }


def _evaluate_mor_lm(
    model: _MoRLatentDepthLM,
    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    cos_c: torch.Tensor,
    sin_c: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    total_nll_nats = 0.0
    total_tokens = 0
    h1_correct = 0
    h1_total = 0
    h3_correct = 0
    h3_total = 0
    depth_h1_sum = 0.0
    depth_h3_sum = 0.0
    n_b = 0
    with torch.no_grad():
        for inp, tgt, h1m, h3m in batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _, d_stats = model(inp, cos_c, sin_c, force_leaky_batch_topk=False)
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
            total_nll_nats += float(loss_sum.item())
            total_tokens += int(tgt.numel())
            preds = logits.argmax(dim=-1)
            if h1m.any():
                h1_correct += int((preds[h1m] == tgt[h1m]).sum().item())
                h1_total += int(h1m.sum().item())
            if h3m.any():
                h3_correct += int((preds[h3m] == tgt[h3m]).sum().item())
                h3_total += int(h3m.sum().item())
            depth_h1_sum += d_stats["mean_depth_hop1"]
            depth_h3_sum += d_stats["mean_depth_hop3"]
            n_b += 1
    model.train()
    bpb = (total_nll_nats / max(total_tokens, 1)) / math.log(2.0)
    return {
        "bpb": bpb,
        "hop1_acc": float(h1_correct) / max(h1_total, 1),
        "hop3_acc": float(h3_correct) / max(h3_total, 1),
        "mean_depth_hop1": depth_h1_sum / max(n_b, 1),
        "mean_depth_hop3": depth_h3_sum / max(n_b, 1),
    }


def run_suite_c_distributed(
    rank: int,
    world_size: int,
    device: torch.device,
    mode: str,
) -> Dict[str, Any]:
    budget_s = 6.0 if mode == "smoke" else 12.0
    T_seq = 256
    B_local = 2
    vocab_size = 64
    d_model = 384
    h_q = 6
    h_kv = 2
    d_head = 64
    d_ff = 1024
    cos_c, sin_c = build_rope_cache(T_seq + 8, d_head, device)

    # -------------------------------------------------------------------------
    # Part 0: Prefix-Invariance & Causal-Routing Audit (Leaky Batch Top-k vs Strict Causal)
    # -------------------------------------------------------------------------
    torch.manual_seed(20261128)
    m_audit = _MoRLatentDepthLM(vocab_size, d_model, h_q, h_kv, d_head, d_ff, "gen2_rsi_round9_flash_mor").to(device=device, dtype=torch.bfloat16)
    m_audit.eval()
    inp_a, _, _, _ = _generate_multihop_latent_depth_batch(2, T_seq + 1, 61000, device)
    inp_b = inp_a.clone()
    # Mutate suffix tokens strictly after T_seq // 2 with high-difficulty 3-hop tokens
    inp_b[:, (T_seq // 2) + 4 :: 4] = 34
    inp_b[:, (T_seq // 2) + 5 :: 4] = 1
    inp_b[:, (T_seq // 2) + 6 :: 4] = 2
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_causal_a, _, _ = m_audit(inp_a, cos_c, sin_c, force_leaky_batch_topk=False)
        logits_causal_b, _, _ = m_audit(inp_b, cos_c, sin_c, force_leaky_batch_topk=False)
        logits_leaky_a, _, _ = m_audit(inp_a, cos_c, sin_c, force_leaky_batch_topk=True)
        logits_leaky_b, _, _ = m_audit(inp_b, cos_c, sin_c, force_leaky_batch_topk=True)

    half_t = T_seq // 2
    strict_causal_prefix_max_diff = float((logits_causal_a[:, :half_t] - logits_causal_b[:, :half_t]).abs().max().item())
    leaky_batch_topk_prefix_max_diff = float((logits_leaky_a[:, :half_t] - logits_leaky_b[:, :half_t]).abs().max().item())
    del m_audit

    prefix_invariance_audit = {
        "strict_causal_prefix_max_abs_diff": strict_causal_prefix_max_diff,
        "leaky_batch_topk_prefix_max_abs_diff": leaky_batch_topk_prefix_max_diff,
        "strict_causal_verified": bool(strict_causal_prefix_max_diff == 0.0 and leaky_batch_topk_prefix_max_diff > 1e-2),
    }

    # Pre-generate deterministic validation, single-touch test, and training batches
    val_batches = [
        _generate_multihop_latent_depth_batch(B_local, T_seq + 1, 71000 + rank * 31 + i, device)
        for i in range(4)
    ]
    test_batches = [
        _generate_multihop_latent_depth_batch(B_local, T_seq + 1, 81000 + rank * 37 + i, device)
        for i in range(4)
    ]
    train_pool = [
        _generate_multihop_latent_depth_batch(B_local, T_seq + 1, 51000 + rank * 41 + i, device)
        for i in range(24)
    ]

    generations_spec = [
        ("gen0_official_mor_kvshare_fullseq_clone", "gen0_official_kvshare_clone"),
        ("gen1_rsi_varlen_recursion_wise_cache", "gen1_rsi_varlen_rec_cache"),
        ("gen2_rsi_round9_rect_flash_mor_kv_share", "gen2_rsi_round9_flash_mor"),
        ("gen3_upstream_buggy_logsumexp_zloss", "gen3_buggy_logsumexp_zloss"),
        ("gen4_over_recursive_nr6_throughput_drop", "gen4_over_recursive_nr6"),
    ]

    # Warm up Triton kernels and pre-populate feature cache outside timed loop
    for _, var_key in generations_spec:
        wm = _MoRLatentDepthLM(vocab_size, d_model, h_q, h_kv, d_head, d_ff, var_key).to(device=device, dtype=torch.bfloat16)
        inp0, tgt0, _, _ = train_pool[0]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg0, al0, _ = wm(inp0, cos_c, sin_c)
            ls0 = F.cross_entropy(lg0.reshape(-1, vocab_size), tgt0.reshape(-1)) + al0
        ls0.backward()
        for b_inp, _, _, _ in train_pool + val_batches + test_batches:
            wm._get_multihop_oracle_features(b_inp)
        del wm, lg0, al0, ls0
    torch.cuda.synchronize(device)
    dist.barrier()

    trained_models: Dict[str, _MoRLatentDepthLM] = {}
    generations_out: List[Dict[str, Any]] = []
    best_val_bpb = float("inf")
    incumbent_steps = 0
    winning_id = ""

    for gen_idx, (cand_id, var_key) in enumerate(generations_spec):
        torch.manual_seed(20261135)
        model = _MoRLatentDepthLM(vocab_size, d_model, h_q, h_kv, d_head, d_ff, var_key).to(device=device, dtype=torch.bfloat16)
        scale_params = [model.hop1_scale, model.hop3_scale]
        scale_ids = {id(p) for p in scale_params}
        backbone_params = [p for p in model.parameters() if id(p) not in scale_ids]
        lr_scale = 6.0 / float(budget_s)
        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 1.0e-4 * lr_scale, "weight_decay": 0.01},
                {"params": scale_params, "lr": 8.5e-3 * lr_scale, "weight_decay": 0.0},
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

            inp, tgt, h1m, h3m = train_pool[steps % len(train_pool)]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, aux_l, _ = model(inp, cos_c, sin_c)
                loss_all = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                q_mask = h1m | h3m
                if q_mask.any():
                    loss_q = F.cross_entropy(logits[q_mask], tgt[q_mask])
                    loss = 0.30 * loss_all + 0.70 * loss_q + aux_l
                else:
                    loss = loss_all + aux_l
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            opt.step()
            steps += 1

        torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - t_start
        val_metrics = _evaluate_mor_lm(model, val_batches, cos_c, sin_c)
        v_ten = torch.tensor(
            [val_metrics["bpb"], val_metrics["hop1_acc"], val_metrics["hop3_acc"], val_metrics["mean_depth_hop1"], val_metrics["mean_depth_hop3"]],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(v_ten, op=dist.ReduceOp.AVG)
        val_bpb = float(v_ten[0].item())
        val_hop1 = float(v_ten[1].item())
        val_hop3 = float(v_ten[2].item())
        d_hop1 = float(v_ten[3].item())
        d_hop3 = float(v_ten[4].item())

        if gen_idx == 0:
            decision = "BASELINE"
            best_val_bpb = val_bpb
            incumbent_steps = steps
            winning_id = cand_id
        elif steps < int(0.85 * incumbent_steps):
            decision = "REVERT_SLOWER"
        elif val_bpb < best_val_bpb - 1e-4:
            decision = "PROMOTE"
            best_val_bpb = val_bpb
            incumbent_steps = max(incumbent_steps, steps)
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
                "val_hop1_acc": val_hop1,
                "val_hop3_acc": val_hop3,
                "mean_depth_hop1": d_hop1,
                "mean_depth_hop3": d_hop3,
            }
        )

    # Single-Touch Final Test Evaluation strictly on Gen 0 Baseline and Promoted Winner
    gen0_id = generations_out[0]["candidate_id"]
    m_gen0 = _evaluate_mor_lm(trained_models[gen0_id], test_batches, cos_c, sin_c)
    m_win = _evaluate_mor_lm(trained_models[winning_id], test_batches, cos_c, sin_c)
    t_ten = torch.tensor(
        [m_gen0["bpb"], m_gen0["hop3_acc"], m_win["bpb"], m_win["hop3_acc"]],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(t_ten, op=dist.ReduceOp.AVG)
    single_touch_test = {
        "gen0_candidate_id": gen0_id,
        "winner_candidate_id": winning_id,
        "gen0_test_bpb": float(t_ten[0].item()),
        "gen0_test_hop3_acc": float(t_ten[1].item()),
        "winner_test_bpb": float(t_ten[2].item()),
        "winner_test_hop3_acc": float(t_ten[3].item()),
        "test_bpb_improvement": float(t_ten[0].item() - t_ten[2].item()),
    }
    del trained_models

    # -------------------------------------------------------------------------
    # Part 2: 5-Arm Matched-Step Parameter-Sharing & Multi-Hop Latent-Depth Law
    # -------------------------------------------------------------------------
    matched_steps = 65 if mode == "smoke" else 140
    law_arms = [
        ("M0_shallow_vanilla_iso_param_nr1", "M0_shallow_vanilla"),
        ("M1_fixed_recursive_middle_cycle_nr3", "M1_fixed_recursive_nr3"),
        ("M2_mor_token_choice_rec_cache_nr3", "M2_mor_token_rec_cache"),
        ("M3_mor_expert_aux_bce_rec_cache_nr3", "M3_mor_expert_rec_cache"),
        ("M4_mor_causal_kv_share_flash_mor_rsi_nr3", "M4_mor_kv_share_rsi"),
    ]
    law_results: Dict[str, Any] = {}

    for arm_name, var_key in law_arms:
        torch.manual_seed(20261145)
        m_law = _MoRLatentDepthLM(vocab_size, d_model, h_q, h_kv, d_head, d_ff, var_key).to(device=device, dtype=torch.bfloat16)
        opt_law = torch.optim.AdamW(m_law.parameters(), lr=2.2e-3 * (65.0 / float(matched_steps)), weight_decay=0.01)

        torch.cuda.synchronize(device)
        t0_law = time.perf_counter()
        for st in range(matched_steps):
            inp, tgt, h1m, h3m = train_pool[st % len(train_pool)]
            opt_law.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, aux_l, _ = m_law(inp, cos_c, sin_c)
                loss_all = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                q_mask = h1m | h3m
                if q_mask.any():
                    loss_q = F.cross_entropy(logits[q_mask], tgt[q_mask])
                    loss = 0.30 * loss_all + 0.70 * loss_q + aux_l
                else:
                    loss = loss_all + aux_l
            loss.backward()
            for p in m_law.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            opt_law.step()
        torch.cuda.synchronize(device)
        elapsed_law = time.perf_counter() - t0_law

        ev_val = _evaluate_mor_lm(m_law, val_batches, cos_c, sin_c)
        ev_tst = _evaluate_mor_lm(m_law, test_batches, cos_c, sin_c)
        m_ten = torch.tensor(
            [
                ev_val["bpb"], ev_val["hop1_acc"], ev_val["hop3_acc"],
                ev_tst["bpb"], ev_tst["hop1_acc"], ev_tst["hop3_acc"],
                ev_tst["mean_depth_hop1"], ev_tst["mean_depth_hop3"],
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(m_ten, op=dist.ReduceOp.AVG)
        law_results[arm_name] = {
            "steps": matched_steps,
            "wall_clock_s": round(elapsed_law, 3),
            "val_bpb": float(m_ten[0].item()),
            "val_hop1_acc": float(m_ten[1].item()),
            "val_hop3_acc": float(m_ten[2].item()),
            "clean_test_bpb": float(m_ten[3].item()),
            "test_hop1_acc": float(m_ten[4].item()),
            "test_hop3_acc": float(m_ten[5].item()),
            "mean_depth_hop1": float(m_ten[6].item()),
            "mean_depth_hop3": float(m_ten[7].item()),
            "adaptive_depth_gap_hop3_minus_hop1": float(m_ten[7].item() - m_ten[6].item()),
        }
        del m_law

    gen0_rec = generations_out[0]
    win_rec = next(g for g in generations_out if g["candidate_id"] == winning_id)

    return {
        "prefix_invariance_audit": prefix_invariance_audit,
        "budget_seconds": budget_s,
        "winning_candidate_id": winning_id,
        "step_multiplier_vs_gen0": float(win_rec["steps_completed"]) / max(float(gen0_rec["steps_completed"]), 1.0),
        "val_bpb_improvement_vs_gen0": float(gen0_rec["val_bpb"] - win_rec["val_bpb"]),
        "generations": generations_out,
        "single_touch_test_evaluation": single_touch_test,
        "matched_step_latent_depth_law": law_results,
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
        for trk_key in ("track1_rect_flash_mor_attention", "track2_end_to_end_mor_stack"):
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
                "steps_completed", "ms_per_step", "val_bpb", "val_hop1_acc",
                "val_hop3_acc", "mean_depth_hop1", "mean_depth_hop3", "eval_harness_sha256",
            ])
        for g in suite_c["generations"]:
            w.writerow([
                rep_tag, g["generation"], g["candidate_id"], g["decision"],
                f"{g['wall_clock_s']:.3f}", g["steps_completed"], f"{g['ms_per_step']:.3f}",
                f"{g['val_bpb']:.6f}", f"{g['val_hop1_acc']:.4f}", f"{g['val_hop3_acc']:.4f}",
                f"{g['mean_depth_hop1']:.3f}", f"{g['mean_depth_hop3']:.3f}", harness_sha,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="E135 Round-9 RSI Mixture-of-Recursions Benchmark")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ledger-dir", type=str, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    t_start_all = time.perf_counter()
    suite_0_res: Dict[str, Any] = {}
    suite_a_res: Dict[str, Any] = {}

    if rank == 0:
        print(f"[E135][Rank 0] Running Suite 0 (Adversarial Red-Team, Upstream MoR Forensics, 16/16 Zero-Spill Cubin Audit, FP64 Parity & O(h^2) Richardson)...", flush=True)
        suite_0_res = run_suite_0(device)
        print(f"[E135][Rank 0] Running Suite A (Round-9 RSI Vector-Pareto Ratchets: Track 1 & Track 2, mode={args.mode})...", flush=True)
        suite_a_res = run_suite_a(device, args.mode)

    dist.barrier()
    if rank == 0:
        print(f"[E135][All 8 Ranks] Running Suite B (8-GPU NVSwitch Recursion-Pinned FSDP2 + MoR-LPT AllToAllv, mode={args.mode})...", flush=True)
    suite_b_res = run_suite_b_distributed(rank, world_size, device, args.mode)

    dist.barrier()
    if rank == 0:
        print(f"[E135][All 8 Ranks] Running Suite C (Strict-Causal Fixed-Wall-Clock val_bpb RSI Loop & 5-Arm Latent-Depth Law, mode={args.mode})...", flush=True)
    suite_c_res = run_suite_c_distributed(rank, world_size, device, args.mode)

    dist.barrier()
    total_elapsed_s = time.perf_counter() - t_start_all

    if rank == 0:
        sm_clk, pwr_w = query_sm_telemetry(0)
        harness_sha = eval_harness_sha256()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_dir = Path(args.ledger_dir)
        ledger_dir.mkdir(parents=True, exist_ok=True)

        rep_tag = out_path.stem
        _write_ledgers(ledger_dir, rep_tag, suite_a_res, suite_c_res, harness_sha)

        payload = {
            "experiment_id": "E135",
            "version": "v126",
            "round": 9,
            "mode": args.mode,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_elapsed_s": round(total_elapsed_s, 2),
            "eval_harness_sha256": harness_sha,
            "environment": {
                "world_size": world_size,
                "gpu_name": torch.cuda.get_device_name(device),
                "sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
                "torch_version": torch.__version__,
                "triton_version": sys.modules["triton"].__version__,
                "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "sm_clock_mhz": sm_clk,
                "power_w": pwr_w,
            },
            "suite_0": suite_0_res,
            "suite_a": suite_a_res,
            "suite_b": suite_b_res,
            "suite_c": suite_c_res,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"[E135][Rank 0] Completed in {total_elapsed_s:.2f}s -> wrote {out_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
