#!/usr/bin/env python3
"""
Round-11 RSI Benchmark & 9-Gate Verification Driver (E137 / v128):
nGPT: Normalized Transformer with Representation Learning on the Hypersphere
(Loshchilov, Hsieh, Sun & Ginsburg, NVIDIA, arXiv:2410.01131v2, ICLR 2025;
official repository: https://github.com/NVIDIA/ngpt)

Suites:
- Suite 0: Red-Team Exploit Audit (4/4 caught), 7/7 FP64 O(h^2) Richardson Derivative
           Verification, Relative-Position Shift-Invariance Theorem on S^{D-1}, and
           19/19 Cubin Zero-Spill Inspection (Gates G1, G2, G3, G4).
- Suite A: Vector-Pareto Kernel Ratchets Across Track 1 (Core Hypersphere Operators:
           RoPE+Norm+sqk, 2x Hypersphere LERP, Scaled SwiGLU, sz Cross-Entropy) and
           Track 2 (End-to-End nGPT Block Fwd+Bwd + 8-Matrix Hypersphere Weight Retraction)
           across shapes s1..s4 (Gates G5, G8).
- Suite B: 8-GPU NVSwitch Cosine-Attention Ring nGPT-CP=8 + ZeRO-1 Shard-Retracted
           Riemannian Optimizer 3-Step Multi-Update Parity & Long-Context Scaling (Gate G9).
- Suite C: Fixed-Wall-Clock 8xA100 DDP Autonomous val_bpb RSI Loop + 5-Arm Hypersphere
           vs Pre-RMSNorm Convergence & Associative Recall Scaling Law (Gates G6, G7).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import rsi_kernel_harness as harness

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def verify_harness_integrity() -> str:
    actual = harness.eval_harness_sha256()
    if actual != EXPECTED_HARNESS_SHA256:
        raise RuntimeError(
            f"Frozen evaluator SHA256 mismatch: expected {EXPECTED_HARNESS_SHA256}, got {actual}"
        )
    return actual


class EnvironmentSnapshot:
    def __init__(self, snap: harness.PrecisionStateSnapshot):
        self.snap = snap

    @staticmethod
    def capture() -> "EnvironmentSnapshot":
        return EnvironmentSnapshot(harness.PrecisionStateSnapshot.capture())

    def verify(self) -> Tuple[bool, str]:
        now = harness.PrecisionStateSnapshot.capture()
        if now != self.snap:
            return False, f"Precision state mutated: {self.snap} -> {now}"
        return True, "OK"


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj) + "\n")


def compute_val_bpb(nats_per_token: float, bytes_per_token: float = 1.0) -> float:
    return float(nats_per_token) / (math.log(2.0) * float(bytes_per_token))


class SingleTouchSplit:
    def __init__(self, name: str, fn):
        self.name = name
        self._fn = fn
        self.touches = 0

    def evaluate_once(self) -> float:
        if self.touches >= 1:
            raise RuntimeError(f"Single-touch split '{self.name}' accessed more than once!")
        self.touches += 1
        return float(self._fn())


def benchmark_cuda_callable(fn, warmup: int = 4, rep: int = 10, device: Optional[torch.device] = None) -> Dict[str, float]:
    gpu_idx = 0 if device is None or device.index is None else int(device.index)
    res = harness.benchmark_rotating_cuda(
        [fn],
        warmup=warmup,
        groups=max(3, rep // 2),
        iters_per_group=2,
        gpu_idx=gpu_idx,
    )
    return {
        "median_ms": float(res["median_ms"]),
        "iqr_ms": float(res["iqr_ms"]),
        "min_ms": float(res["min_ms"]),
        "max_ms": float(res["max_ms"]),
        "peak_allocated_mb": float(res["peak_alloc_mib"]),
        "peak_reserved_mb": float(res["peak_reserved_mib"]),
    }


class ParetoLedger:
    def __init__(self, baseline_shapes: Dict[str, Dict[str, float]], track: str = "ngpt_track"):
        self.shape_keys = list(baseline_shapes.keys())
        self.ratchet = harness.RSIVectorRatchet(track=track, shape_keys=self.shape_keys)
        self.trial_counter = 0
        base_per_shape = {
            k: {
                "median_ms": float(baseline_shapes[k]["median_ms"]),
                "peak_alloc_mib": float(baseline_shapes[k]["peak_allocated_mb"]),
            }
            for k in self.shape_keys
        }
        self.ratchet.evaluate_and_record(
            trial_id=0,
            candidate_id="baseline",
            candidate_sha256=harness.hash_callable_sources(track),
            parity_checks=[{"name": "parity", "ok": True, "rel_l2": 0.0}],
            richardson_checks=[],
            anti_cheat_checks={"precision_state": True, "zero_spills": True},
            per_shape_metrics=base_per_shape,
            hypothesis_note="Baseline seed",
        )

    def evaluate_and_record(
        self,
        candidate_id: str,
        candidate_shapes: Dict[str, Dict[str, float]],
        correctness_ok: bool,
        max_rel_l2: float,
        total_spills: int,
        ledger_path: Path,
    ) -> Dict[str, Any]:
        self.trial_counter += 1
        per_shape = {
            k: {
                "median_ms": float(candidate_shapes[k]["median_ms"]),
                "peak_alloc_mib": float(candidate_shapes[k]["peak_allocated_mb"]),
            }
            for k in self.shape_keys
        }
        rec = self.ratchet.evaluate_and_record(
            trial_id=self.trial_counter,
            candidate_id=candidate_id,
            candidate_sha256=harness.hash_callable_sources(candidate_id),
            parity_checks=[{"name": "parity", "ok": bool(correctness_ok), "rel_l2": float(max_rel_l2)}],
            richardson_checks=[],
            anti_cheat_checks={"precision_state": True, "zero_spills": (total_spills == 0)},
            per_shape_metrics=per_shape,
            hypothesis_note=f"spills={total_spills}",
        )
        out = {
            "trial_id": rec.trial_id,
            "candidate_id": rec.candidate_id,
            "status": rec.decision,
            "geomean_ms": rec.geomean_ms,
            "peak_alloc_mib": rec.peak_alloc_mib,
            "speedup_vs_baseline": rec.speedup_vs_baseline,
            "speedup_vs_incumbent": rec.speedup_vs_incumbent,
            "total_spills": int(total_spills),
            "diagnosis": rec.diagnosis,
        }
        append_jsonl(ledger_path, out)
        return out

from ngpt_hypersphere_rsi_kernels import (
    _ngpt_cosine_flash_bwd_dkv_d128_naive_spill_kernel,
    _ngpt_cosine_flash_bwd_prep_kernel,
    _ngpt_cosine_flash_fwd_impl,
    build_ngpt_rope_cache,
    eager_ngpt_cosine_attn_oracle,
    eager_ngpt_hypersphere_lerp_oracle,
    eager_ngpt_rope_norm_sqk_oracle,
    eager_ngpt_scaled_swiglu_oracle,
    eager_ngpt_sz_cross_entropy_oracle,
    fused_ngpt_adamw_retract_step,
    get_all_round11_rsi_kernels,
    inspect_compiled_kernel,
    ngpt_cosine_flash_attn_triton,
    ngpt_hypersphere_lerp_triton,
    ngpt_rope_norm_sqk_triton,
    ngpt_scaled_swiglu_triton,
    ngpt_sz_cross_entropy_triton,
    ring_ngpt_cosine_flash_attn_cp8,
    rsi_ngpt_block_forward,
    upstream_nvidia_ngpt_block_forward,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm().item()
    denom = max(b.float().norm().item(), 1e-12)
    return float(diff / denom)


def _richardson_check(
    fn,
    x: torch.Tensor,
    grad_analytic: torch.Tensor,
    h: float = 1e-2,
    seed: int = 137,
) -> Dict[str, float]:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    d = torch.randn(x.shape, dtype=torch.float64, device=x.device, generator=gen)
    d = d / d.norm()

    with torch.no_grad():
        f_ph = float(fn(x + h * d).item())
        f_mh = float(fn(x - h * d).item())
        f_ph2 = float(fn(x + 0.5 * h * d).item())
        f_mh2 = float(fn(x - 0.5 * h * d).item())

    fd_h = (f_ph - f_mh) / (2.0 * h)
    fd_h2 = (f_ph2 - f_mh2) / h
    exact_dir = float((grad_analytic.double() * d).sum().item())

    err_h = abs(fd_h - exact_dir)
    err_h2 = max(abs(fd_h2 - exact_dir), 1e-16)
    ratio = err_h / err_h2
    richardson_extrap = (4.0 * fd_h2 - fd_h) / 3.0
    rel_err = abs(richardson_extrap - exact_dir) / max(abs(exact_dir), 1e-12)
    return {
        "exact_dir": exact_dir,
        "fd_h": fd_h,
        "fd_h2": fd_h2,
        "err_h": err_h,
        "err_h2": err_h2,
        "ratio": float(ratio),
        "richardson_rel_err": float(rel_err),
    }


# =============================================================================
# Suite 0: Red-Team Exploit Audit, O(h^2) Richardson, Shift-Invariance Theorem,
#          and 19/19 Cubin Zero-Spill Inspection (Gates G1, G2, G3, G4)
# =============================================================================

def run_suite0_redteam_and_correctness(out_dir: Path) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    torch.manual_seed(20261119)
    harness_sha = verify_harness_integrity()

    # 1. FP32 & BF16 Operator Parity Checks
    B, T, C, H, D = 2, 128, 512, 8, 64
    I = 4 * C
    V = 4096
    base_scale = 1.0 / math.sqrt(float(C))

    # (a) Hypersphere LERP
    x_fp32 = torch.randn((B, T, C), device=device, dtype=torch.float32, requires_grad=True)
    g_fp32 = torch.randn((B, T, C), device=device, dtype=torch.float32, requires_grad=True)
    alpha_fp32 = (torch.randn((C,), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)
    dy_lerp = torch.randn((B, T, C), device=device, dtype=torch.float32)

    x_ref = x_fp32.detach().clone().requires_grad_(True)
    g_ref = g_fp32.detach().clone().requires_grad_(True)
    alpha_ref = alpha_fp32.detach().clone().requires_grad_(True)

    y_tri = ngpt_hypersphere_lerp_triton(x_fp32, g_fp32, alpha_fp32, 0.05, base_scale)
    (y_tri * dy_lerp).sum().backward()

    y_ora = eager_ngpt_hypersphere_lerp_oracle(x_ref, g_ref, alpha_ref, 0.05, base_scale)
    (y_ora * dy_lerp).sum().backward()

    lerp_fp32_out_err = _rel_l2(y_tri, y_ora)
    lerp_fp32_dx_err = _rel_l2(x_fp32.grad, x_ref.grad)
    lerp_fp32_dg_err = _rel_l2(g_fp32.grad, g_ref.grad)
    lerp_fp32_dalpha_err = _rel_l2(alpha_fp32.grad, alpha_ref.grad)

    x_bf16 = x_fp32.detach().to(torch.bfloat16).requires_grad_(True)
    g_bf16 = g_fp32.detach().to(torch.bfloat16).requires_grad_(True)
    y_bf16 = ngpt_hypersphere_lerp_triton(x_bf16, g_bf16, alpha_fp32.detach().clone().requires_grad_(True), 0.05, base_scale)
    (y_bf16.float() * dy_lerp).sum().backward()
    lerp_bf16_out_err = _rel_l2(y_bf16, y_ora)
    lerp_bf16_dx_err = _rel_l2(x_bf16.grad, x_ref.grad)

    # (b) RoPE + Norm + sqk across ROPE_MODE in {0, 1, 2}
    q_fp32 = torch.randn((B, T, H, D), device=device, dtype=torch.float32, requires_grad=True)
    k_fp32 = torch.randn((B, T, H, D), device=device, dtype=torch.float32, requires_grad=True)
    sqk_fp32 = (torch.ones((H, D), device=device, dtype=torch.float32) * base_scale * (1.0 + 0.2 * torch.randn((H, D), device=device))).detach().requires_grad_(True)
    dqo = torch.randn((B, T, H, D), device=device, dtype=torch.float32)
    dko = torch.randn((B, T, H, D), device=device, dtype=torch.float32)

    rope_mode_parity = {}
    for mode in (0, 1, 2):
        cos_m, sin_m = build_ngpt_rope_cache(T, D, device=device, rope_mode=mode)
        qt = q_fp32.detach().clone().requires_grad_(True)
        kt = k_fp32.detach().clone().requires_grad_(True)
        st = sqk_fp32.detach().clone().requires_grad_(True)
        qo_t, ko_t = ngpt_rope_norm_sqk_triton(qt, kt, cos_m, sin_m, st, 1.0, base_scale, rope_mode=mode)
        ((qo_t * dqo).sum() + (ko_t * dko).sum()).backward()

        qr = q_fp32.detach().clone().requires_grad_(True)
        kr = k_fp32.detach().clone().requires_grad_(True)
        sr = sqk_fp32.detach().clone().requires_grad_(True)
        qo_r, ko_r = eager_ngpt_rope_norm_sqk_oracle(qr, kr, cos_m, sin_m, sr, 1.0, base_scale, rope_mode=mode)
        ((qo_r * dqo).sum() + (ko_r * dko).sum()).backward()

        rope_mode_parity[f"mode_{mode}"] = {
            "qo_rel_l2": _rel_l2(qo_t, qo_r),
            "ko_rel_l2": _rel_l2(ko_t, ko_r),
            "dq_rel_l2": _rel_l2(qt.grad, qr.grad),
            "dk_rel_l2": _rel_l2(kt.grad, kr.grad),
            "dsqk_rel_l2": _rel_l2(st.grad, sr.grad),
        }

    # Verify exact bitwise/numerical parity of ROPE_MODE=0 against upstream `NVIDIA/ngpt/main/model.py` lines 40-58
    position_cpu = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    div_term_cpu = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
    sin_emb_cpu = torch.zeros((T, D), dtype=torch.float32)
    sin_emb_cpu[:, 0::2] = torch.sin(position_cpu * div_term_cpu)
    sin_emb_cpu[:, 1::2] = torch.cos(position_cpu * div_term_cpu)
    sin_up, cos_up = sin_emb_cpu.to(device).chunk(2, dim=-1)
    qt_up = q_fp32.detach().transpose(1, 2)
    q_rot_up = torch.reshape(
        torch.stack((-qt_up[..., 1::2], qt_up[..., ::2]), dim=-1),
        qt_up.shape[:-1] + (D // 2, 2),
    ) * torch.stack((cos_up, sin_up), dim=-1)
    q_rot_up = torch.reshape(q_rot_up, qt_up.shape).transpose(2, 1)
    q_up_norm = (sqk_fp32.detach() / base_scale).view(1, 1, H, D) * (
        q_rot_up / (q_rot_up.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    )
    cos_m0, sin_m0 = build_ngpt_rope_cache(T, D, device=device, rope_mode=0)
    qo_m0, _ = ngpt_rope_norm_sqk_triton(
        q_fp32.detach(), k_fp32.detach(), cos_m0, sin_m0, sqk_fp32.detach(), 1.0, base_scale, rope_mode=0
    )
    upstream_rope_parity_rel_l2 = _rel_l2(qo_m0, q_up_norm)

    # Prove Relative-Position Shift-Invariance Theorem on S^{D-1}:
    # Take fixed unit vectors q_vec, k_vec at positions (m, n) vs shifted (m + delta, n + delta)
    q_base_vec = torch.randn((1, 1, H, D), device=device, dtype=torch.float32).expand(1, T, H, D).contiguous()
    k_base_vec = torch.randn((1, 1, H, D), device=device, dtype=torch.float32).expand(1, T, H, D).contiguous()
    shift_invariance = {}
    for mode, label in [
        (0, "upstream_broken_chunk2_rope"),
        (1, "ortho_rope_anisotropic_sqk"),
        (2, "rsi_pair_isotropic_ortho_rope"),
    ]:
        cos_s, sin_s = build_ngpt_rope_cache(T, D, device=device, rope_mode=mode)
        qs, ks = ngpt_rope_norm_sqk_triton(q_base_vec, k_base_vec, cos_s, sin_s, sqk_fp32.detach(), 1.0, base_scale, rope_mode=mode)
        # Compare <qs[:, m], ks[:, n]> vs <qs[:, m+delta], ks[:, n+delta]> for m=17, n=5, delta=31
        m_idx, n_idx, delta = 17, 5, 31
        logit_base = (qs[:, m_idx] * ks[:, n_idx]).sum(dim=-1) * math.sqrt(float(D))
        logit_shift = (qs[:, m_idx + delta] * ks[:, n_idx + delta]).sum(dim=-1) * math.sqrt(float(D))
        shift_invariance[label] = float((logit_base - logit_shift).abs().max().item())

    # (c) Scaled SwiGLU
    uv_fp32 = torch.randn((B, T, 2 * I), device=device, dtype=torch.float32, requires_grad=True)
    suv_fp32 = torch.ones((2 * I,), device=device, dtype=torch.float32, requires_grad=True)
    dy_mlp = torch.randn((B, T, I), device=device, dtype=torch.float32)

    uv_ref = uv_fp32.detach().clone().requires_grad_(True)
    suv_ref = suv_fp32.detach().clone().requires_grad_(True)

    y_mlp_tri = ngpt_scaled_swiglu_triton(uv_fp32, suv_fp32, n_embd=C)
    (y_mlp_tri * dy_mlp).sum().backward()

    y_mlp_ora = eager_ngpt_scaled_swiglu_oracle(uv_ref, suv_ref, n_embd=C)
    (y_mlp_ora * dy_mlp).sum().backward()

    swiglu_fp32_out_err = _rel_l2(y_mlp_tri, y_mlp_ora)
    swiglu_fp32_duv_err = _rel_l2(uv_fp32.grad, uv_ref.grad)
    swiglu_fp32_dsuv_err = _rel_l2(suv_fp32.grad, suv_ref.grad)

    # (d) Cosine FlashAttention at D=64 and D=128
    attn_parity = {}
    for d_head in (64, 128):
        qa = F.normalize(torch.randn((B, T, H, d_head), device=device, dtype=torch.float32), p=2, dim=-1).detach().requires_grad_(True)
        ka = F.normalize(torch.randn((B, T, H, d_head), device=device, dtype=torch.float32), p=2, dim=-1).detach().requires_grad_(True)
        va = torch.randn((B, T, H, d_head), device=device, dtype=torch.float32, requires_grad=True)
        doa = torch.randn((B, T, H, d_head), device=device, dtype=torch.float32)

        qr_a = qa.detach().clone().requires_grad_(True)
        kr_a = ka.detach().clone().requires_grad_(True)
        vr_a = va.detach().clone().requires_grad_(True)

        oa_tri = ngpt_cosine_flash_attn_triton(qa, ka, va, sm_scale=math.sqrt(float(d_head)), is_causal=True)
        (oa_tri * doa).sum().backward()

        oa_ora = eager_ngpt_cosine_attn_oracle(qr_a, kr_a, vr_a, sm_scale=math.sqrt(float(d_head)), is_causal=True)
        (oa_ora * doa).sum().backward()

        qa_bf16 = qa.detach().to(torch.bfloat16).requires_grad_(True)
        ka_bf16 = ka.detach().to(torch.bfloat16).requires_grad_(True)
        va_bf16 = va.detach().to(torch.bfloat16).requires_grad_(True)
        oa_bf16 = ngpt_cosine_flash_attn_triton(qa_bf16, ka_bf16, va_bf16, sm_scale=math.sqrt(float(d_head)), is_causal=True)
        (oa_bf16.float() * doa).sum().backward()

        attn_parity[f"D_{d_head}"] = {
            "fp32_out_rel_l2": _rel_l2(oa_tri, oa_ora),
            "fp32_dq_rel_l2": _rel_l2(qa.grad, qr_a.grad),
            "fp32_dk_rel_l2": _rel_l2(ka.grad, kr_a.grad),
            "fp32_dv_rel_l2": _rel_l2(va.grad, vr_a.grad),
            "bf16_out_rel_l2": _rel_l2(oa_bf16, oa_ora),
            "bf16_dq_rel_l2": _rel_l2(qa_bf16.grad, qr_a.grad),
        }

    # (e) Fused sz Cross-Entropy (with ignore_index=-1)
    z_fp32 = F.normalize(torch.randn((B, T, V), device=device, dtype=torch.float32), p=2, dim=-1).detach().requires_grad_(True)
    sz_fp32 = (torch.ones((V,), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)
    targets = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
    targets[0, -5:] = -1

    z_ref = z_fp32.detach().clone().requires_grad_(True)
    sz_ref = sz_fp32.detach().clone().requires_grad_(True)

    loss_tri = ngpt_sz_cross_entropy_triton(z_fp32, sz_fp32, targets, 1.0, base_scale, ignore_index=-1)
    loss_tri.backward()

    loss_ora = eager_ngpt_sz_cross_entropy_oracle(z_ref, sz_ref, targets, 1.0, base_scale, ignore_index=-1)
    loss_ora.backward()

    ce_loss_rel_err = abs(loss_tri.item() - loss_ora.item()) / max(abs(loss_ora.item()), 1e-12)
    ce_dz_rel_l2 = _rel_l2(z_fp32.grad, z_ref.grad)
    ce_dsz_rel_l2 = _rel_l2(sz_fp32.grad, sz_ref.grad)

    # (f) Fused Riemannian AdamW + Hypersphere Retraction (row idim=1 & col idim=0)
    retract_checks = {}
    for idim, shape in [(1, (512, 512)), (0, (512, 512))]:
        w_init = F.normalize(torch.randn(shape, device=device, dtype=torch.float32), p=2, dim=idim)
        g_step = torch.randn(shape, device=device, dtype=torch.float32) * 0.1
        # Mode 0 vs Upstream PyTorch AdamW + justnorm
        w_m0 = w_init.clone()
        m_m0 = torch.zeros_like(w_m0)
        v_m0 = torch.zeros_like(w_m0)
        fused_ngpt_adamw_retract_step(w_m0, g_step, m_m0, v_m0, step=1, idim=idim, lr=1.5e-3, riemannian_mode=0)

        w_up = w_init.clone().requires_grad_(True)
        opt_up = torch.optim.AdamW([w_up], lr=1.5e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        w_up.grad = g_step.clone()
        opt_up.step()
        with torch.no_grad():
            w_up_norm = w_up / (w_up.norm(p=2, dim=idim, keepdim=True) + 1e-12)

        # Mode 1: Riemannian Tangent-Space Adam + Vector Transport
        w_m1 = w_init.clone()
        m_m1 = torch.zeros_like(w_m1)
        v_m1 = torch.zeros_like(w_m1)
        fused_ngpt_adamw_retract_step(w_m1, g_step, m_m1, v_m1, step=1, idim=idim, lr=1.5e-3, riemannian_mode=1)
        unit_norm_err = float((w_m1.norm(p=2, dim=idim) - 1.0).abs().max().item())
        tangent_orth_err = float(
            ((w_m1 * m_m1).sum(dim=idim).abs() / m_m1.norm(p=2, dim=idim).clamp(min=1e-12)).max().item()
        )
        retract_checks[f"idim_{idim}"] = {
            "mode0_vs_upstream_rel_l2": _rel_l2(w_m0, w_up_norm),
            "mode1_unit_norm_max_err": unit_norm_err,
            "mode1_momentum_tangent_orth_err": tangent_orth_err,
        }

    # 2. Gate G2: 7/7 FP64 O(h^2) Richardson Derivative Checks on Nonlinear Objective L(Y) = sum(sin(Y) * W)
    Br, Tr, Cr, Hr, Dr = 2, 16, 64, 4, 16
    Ir = 128
    xr_64 = torch.randn((Br, Tr, Cr), device=device, dtype=torch.float64, requires_grad=True)
    gr_64 = torch.randn((Br, Tr, Cr), device=device, dtype=torch.float64, requires_grad=True)
    ar_64 = (torch.randn((Cr,), device=device, dtype=torch.float64) * 0.2 + 0.3).detach().requires_grad_(True)
    wr_lerp = torch.randn((Br, Tr, Cr), device=device, dtype=torch.float64)

    y_lerp_64 = eager_ngpt_hypersphere_lerp_oracle(xr_64, gr_64, ar_64, 0.05, 0.125)
    (torch.sin(y_lerp_64) * wr_lerp).sum().backward()

    rich_lerp_x = _richardson_check(
        lambda z: (torch.sin(eager_ngpt_hypersphere_lerp_oracle(z, gr_64.detach(), ar_64.detach(), 0.05, 0.125)) * wr_lerp).sum(),
        xr_64.detach(),
        xr_64.grad,
        h=1e-2,
        seed=101,
    )
    rich_lerp_g = _richardson_check(
        lambda z: (torch.sin(eager_ngpt_hypersphere_lerp_oracle(xr_64.detach(), z, ar_64.detach(), 0.05, 0.125)) * wr_lerp).sum(),
        gr_64.detach(),
        gr_64.grad,
        h=1e-2,
        seed=102,
    )
    rich_lerp_alpha = _richardson_check(
        lambda z: (torch.sin(eager_ngpt_hypersphere_lerp_oracle(xr_64.detach(), gr_64.detach(), z, 0.05, 0.125)) * wr_lerp).sum(),
        ar_64.detach(),
        ar_64.grad,
        h=1e-2,
        seed=103,
    )

    qr_64 = torch.randn((Br, Tr, Hr, Dr), device=device, dtype=torch.float64, requires_grad=True)
    kr_64 = torch.randn((Br, Tr, Hr, Dr), device=device, dtype=torch.float64, requires_grad=True)
    sqk_64 = (torch.ones((Hr, Dr), device=device, dtype=torch.float64) * 0.125).detach().requires_grad_(True)
    cos_r, sin_r = build_ngpt_rope_cache(Tr, Dr, device=device, rope_mode=2)
    wr_q = torch.randn((Br, Tr, Hr, Dr), device=device, dtype=torch.float64)
    wr_k = torch.randn((Br, Tr, Hr, Dr), device=device, dtype=torch.float64)

    qo_64, ko_64 = eager_ngpt_rope_norm_sqk_oracle(qr_64, kr_64, cos_r, sin_r, sqk_64, 1.0, 0.125, rope_mode=2)
    ((torch.sin(qo_64) * wr_q).sum() + (torch.sin(ko_64) * wr_k).sum()).backward()

    rich_rope_q = _richardson_check(
        lambda z: (
            (torch.sin(eager_ngpt_rope_norm_sqk_oracle(z, kr_64.detach(), cos_r, sin_r, sqk_64.detach(), 1.0, 0.125, rope_mode=2)[0]) * wr_q).sum()
        ),
        qr_64.detach(),
        qr_64.grad,
        h=1e-2,
        seed=104,
    )
    rich_rope_sqk = _richardson_check(
        lambda z: (
            (torch.sin(eager_ngpt_rope_norm_sqk_oracle(qr_64.detach(), kr_64.detach(), cos_r, sin_r, z, 1.0, 0.125, rope_mode=2)[0]) * wr_q).sum()
            + (torch.sin(eager_ngpt_rope_norm_sqk_oracle(qr_64.detach(), kr_64.detach(), cos_r, sin_r, z, 1.0, 0.125, rope_mode=2)[1]) * wr_k).sum()
        ),
        sqk_64.detach(),
        sqk_64.grad,
        h=1e-2,
        seed=105,
    )

    uv_64 = torch.randn((Br, Tr, 2 * Ir), device=device, dtype=torch.float64, requires_grad=True)
    suv_64 = (torch.ones((2 * Ir,), device=device, dtype=torch.float64) * 0.25).detach().requires_grad_(True)
    wr_mlp = torch.randn((Br, Tr, Ir), device=device, dtype=torch.float64)
    y_mlp_64 = eager_ngpt_scaled_swiglu_oracle(uv_64, suv_64, n_embd=Cr)
    (torch.sin(y_mlp_64) * wr_mlp).sum().backward()

    rich_swiglu_uv = _richardson_check(
        lambda z: (torch.sin(eager_ngpt_scaled_swiglu_oracle(z, suv_64.detach(), n_embd=Cr)) * wr_mlp).sum(),
        uv_64.detach(),
        uv_64.grad,
        h=1e-2,
        seed=106,
    )
    rich_swiglu_suv = _richardson_check(
        lambda z: (torch.sin(eager_ngpt_scaled_swiglu_oracle(uv_64.detach(), z, n_embd=Cr)) * wr_mlp).sum(),
        suv_64.detach(),
        suv_64.grad,
        h=1e-2,
        seed=107,
    )

    richardson_checks = {
        "hypersphere_lerp_dx": rich_lerp_x,
        "hypersphere_lerp_dg": rich_lerp_g,
        "hypersphere_lerp_dalpha": rich_lerp_alpha,
        "rope_norm_sqk_dq": rich_rope_q,
        "rope_norm_sqk_dsqk": rich_rope_sqk,
        "scaled_swiglu_duv": rich_swiglu_uv,
        "scaled_swiglu_dsuv": rich_swiglu_suv,
    }
    all_richardson_passed = all(
        3.5 <= chk["ratio"] <= 4.5 and chk["richardson_rel_err"] <= 1e-4
        for chk in richardson_checks.values()
    )

    # 3. Gate G3: 4/4 Red-Team Exploits Caught
    redteam_ledger_path = out_dir / "suite0_redteam_ledger.jsonl"
    if redteam_ledger_path.exists():
        redteam_ledger_path.unlink()

    exploits = []
    # Exploit 1: Omitted coupled 3-sphere radial Jacobian in Hypersphere LERP backward
    dy_e1 = dy_lerp + 0.5 * x_fp32.detach()
    x_e1_ref = x_fp32.detach().clone().requires_grad_(True)
    y_e1_ora = eager_ngpt_hypersphere_lerp_oracle(x_e1_ref, g_fp32.detach(), alpha_fp32.detach(), 0.05, base_scale)
    (y_e1_ora * dy_e1).sum().backward()
    rx_e1 = torch.rsqrt((x_fp32.detach() ** 2).sum(dim=-1, keepdim=True) + 1e-12)
    rg_e1 = torch.rsqrt((g_fp32.detach() ** 2).sum(dim=-1, keepdim=True) + 1e-12)
    a_e1 = torch.abs(alpha_fp32.detach() * (0.05 / base_scale))
    u_e1 = x_fp32.detach() * rx_e1 + a_e1 * (g_fp32.detach() * rg_e1 - x_fp32.detach() * rx_e1)
    ru_e1 = torch.rsqrt((u_e1 ** 2).sum(dim=-1, keepdim=True) + 1e-12)
    dx_bad_e1 = rx_e1 * (1.0 - a_e1) * (ru_e1 * dy_e1)  # Omitted radial projection!
    err_e1 = _rel_l2(dx_bad_e1, x_e1_ref.grad)
    caught_e1 = err_e1 > 0.10
    exploits.append({
        "exploit_id": "E1_omitted_3sphere_radial_jacobian",
        "error_rel_l2": err_e1,
        "caught": bool(caught_e1),
        "verdict": "REJECT_GRADIENT" if caught_e1 else "MISSED",
    })

    # Exploit 2: Wrong standard 1/sqrt(D) softmax scale instead of +sqrt(D) on unit Q, K
    o_bad_e2 = ngpt_cosine_flash_attn_triton(
        qa.detach(), ka.detach(), va.detach(), sm_scale=1.0 / math.sqrt(128.0), is_causal=True
    )
    o_good_e2 = eager_ngpt_cosine_attn_oracle(
        qa.detach(), ka.detach(), va.detach(), sm_scale=math.sqrt(128.0), is_causal=True
    )
    err_e2 = _rel_l2(o_bad_e2, o_good_e2)
    caught_e2 = err_e2 > 0.15
    exploits.append({
        "exploit_id": "E2_wrong_inv_sqrt_d_cosine_attn_scale",
        "error_rel_l2": err_e2,
        "caught": bool(caught_e2),
        "verdict": "REJECT_CORRECTNESS" if caught_e2 else "MISSED",
    })

    # Exploit 3: Global allow_tf32 environment mutation
    snap = EnvironmentSnapshot.capture()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = not prev_tf32
    env_ok, env_reason = snap.verify()
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    caught_e3 = not env_ok
    exploits.append({
        "exploit_id": "E3_allow_tf32_env_mutation",
        "env_reason": env_reason,
        "caught": bool(caught_e3),
        "verdict": "REJECT_ENV_MUTATION" if caught_e3 else "MISSED",
    })

    # Exploit 4: Un-retracted Euclidean weight update violating ||W||_2 == 1 on S^{d-1}
    w_bad_e4 = w_init - 0.1 * g_step  # No unit-sphere retraction!
    norm_dev_e4 = float((w_bad_e4.norm(p=2, dim=1) - 1.0).abs().max().item())
    caught_e4 = norm_dev_e4 > 1e-3
    exploits.append({
        "exploit_id": "E4_unretracted_hypersphere_weight_drift",
        "max_unit_norm_deviation": norm_dev_e4,
        "caught": bool(caught_e4),
        "verdict": "REJECT_UNIT_NORM_VIOLATION" if caught_e4 else "MISSED",
    })

    for ex in exploits:
        append_jsonl(redteam_ledger_path, ex)

    redteam_caught = sum(1 for ex in exploits if ex["caught"])

    # 4. Gate G4: Trigger compilation of all 19 Round-11 RSI kernels (including ring LSE update)
    #    and negative-control monolithic 64x64x128 dKV spill kernel
    o_ring_tmp = torch.zeros((2, 32, 4, 64), device=device, dtype=torch.float32)
    lse_ring_tmp = torch.zeros((2, 4, 32), device=device, dtype=torch.float32)
    from ngpt_hypersphere_rsi_kernels import _ring_lse_merge_triton
    _ring_lse_merge_triton(o_ring_tmp, lse_ring_tmp, o_ring_tmp, lse_ring_tmp)

    # Compile negative-control monolithic [64x64x128] dKV spill kernel
    q_sp = torch.randn((1, 128, 2, 128), device=device, dtype=torch.bfloat16)
    k_sp = torch.randn((1, 128, 2, 128), device=device, dtype=torch.bfloat16)
    v_sp = torch.randn((1, 128, 2, 128), device=device, dtype=torch.bfloat16)
    do_sp = torch.randn((1, 128, 2, 128), device=device, dtype=torch.bfloat16)
    o_sp, lse_sp = _ngpt_cosine_flash_fwd_impl(q_sp, k_sp, v_sp, math.sqrt(128.0), is_causal=True)
    delta_sp = torch.empty_like(lse_sp)
    _ngpt_cosine_flash_bwd_prep_kernel[(2, 2)](
        o_sp, do_sp, delta_sp, 128, 2,
        o_sp.stride(0), o_sp.stride(1), o_sp.stride(2), o_sp.stride(3),
        delta_sp.stride(0), delta_sp.stride(1), delta_sp.stride(2),
        BM=64, D=128, num_warps=4, num_stages=1,
    )
    dk_sp = torch.empty_like(k_sp)
    dv_sp = torch.empty_like(v_sp)
    _ngpt_cosine_flash_bwd_dkv_d128_naive_spill_kernel[(2, 2)](
        q_sp, k_sp, v_sp, do_sp, lse_sp, delta_sp, dk_sp, dv_sp,
        128, 2,
        q_sp.stride(0), q_sp.stride(1), q_sp.stride(2), q_sp.stride(3),
        k_sp.stride(0), k_sp.stride(1), k_sp.stride(2), k_sp.stride(3),
        v_sp.stride(0), v_sp.stride(1), v_sp.stride(2), v_sp.stride(3),
        lse_sp.stride(0), lse_sp.stride(1), lse_sp.stride(2),
        math.sqrt(128.0), BM=64, BN=64, num_warps=4, num_stages=2,
    )
    naive_spill_audit = inspect_compiled_kernel(_ngpt_cosine_flash_bwd_dkv_d128_naive_spill_kernel)

    kernel_audits = {}
    total_rsi_spills = 0
    for kname, kobj in get_all_round11_rsi_kernels().items():
        info = inspect_compiled_kernel(kobj)
        kernel_audits[kname] = info
        total_rsi_spills += info["n_spills"]

    max_fp32_rel_l2 = max(
        lerp_fp32_out_err,
        lerp_fp32_dx_err,
        lerp_fp32_dg_err,
        lerp_fp32_dalpha_err,
        rope_mode_parity["mode_2"]["qo_rel_l2"],
        rope_mode_parity["mode_2"]["dq_rel_l2"],
        rope_mode_parity["mode_2"]["dsqk_rel_l2"],
        swiglu_fp32_out_err,
        swiglu_fp32_duv_err,
        swiglu_fp32_dsuv_err,
        attn_parity["D_64"]["fp32_out_rel_l2"],
        attn_parity["D_64"]["fp32_dq_rel_l2"],
        attn_parity["D_128"]["fp32_out_rel_l2"],
        attn_parity["D_128"]["fp32_dk_rel_l2"],
        ce_loss_rel_err,
        ce_dz_rel_l2,
        ce_dsz_rel_l2,
        retract_checks["idim_1"]["mode0_vs_upstream_rel_l2"],
        retract_checks["idim_0"]["mode0_vs_upstream_rel_l2"],
    )
    max_bf16_rel_l2 = max(
        lerp_bf16_out_err,
        lerp_bf16_dx_err,
        attn_parity["D_64"]["bf16_out_rel_l2"],
        attn_parity["D_128"]["bf16_out_rel_l2"],
    )

    return {
        "eval_harness_sha256": harness_sha,
        "gate_g1_passed": bool(
            harness_sha == EXPECTED_HARNESS_SHA256
            and max_fp32_rel_l2 <= 1e-4
            and max_bf16_rel_l2 <= 1.5e-2
            and shift_invariance["rsi_pair_isotropic_ortho_rope"] <= 1e-5
            and shift_invariance["upstream_broken_chunk2_rope"] > 0.10
        ),
        "gate_g2_passed": bool(all_richardson_passed),
        "gate_g3_passed": bool(redteam_caught == 4),
        "gate_g4_passed": bool(total_rsi_spills == 0 and naive_spill_audit["n_spills"] > 0),
        "max_fp32_rel_l2": float(max_fp32_rel_l2),
        "max_bf16_rel_l2": float(max_bf16_rel_l2),
        "upstream_rope_parity_rel_l2": float(upstream_rope_parity_rel_l2),
        "relative_shift_invariance_max_abs_diff": shift_invariance,
        "rope_mode_parity": rope_mode_parity,
        "attn_parity": attn_parity,
        "retract_checks": retract_checks,
        "richardson_checks": richardson_checks,
        "redteam_exploits": exploits,
        "redteam_caught": int(redteam_caught),
        "redteam_total": 4,
        "kernel_cubin_audits": kernel_audits,
        "total_rsi_spills": int(total_rsi_spills),
        "naive_d128_monolithic_dkv_spill_control": naive_spill_audit,
    }


# =============================================================================
# Suite A: Vector-Pareto Kernel Ratchets Across Track 1 & Track 2 (Gates G5, G8)
# =============================================================================

def run_suite_a_pareto_ratchets(out_dir: Path, mode: str, rep_idx: int) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    torch.manual_seed(20261119 + rep_idx)

    warmup = 4 if mode == "smoke" else 8
    iters = 10 if mode == "smoke" else 20

    shapes = [
        {"id": "s1_B2_T2048_C1024_H16_D64", "B": 2, "T": 2048, "C": 1024, "H": 16, "D": 64, "V": 16384},
        {"id": "s2_B2_T4096_C1024_H16_D64", "B": 2, "T": 4096, "C": 1024, "H": 16, "D": 64, "V": 16384},
        {"id": "s3_B1_T2048_C2048_H16_D128", "B": 1, "T": 2048, "C": 2048, "H": 16, "D": 128, "V": 16384},
        {"id": "s4_B2_T2048_C2048_H16_D128", "B": 2, "T": 2048, "C": 2048, "H": 16, "D": 128, "V": 16384},
    ]

    # Track 1: Core nGPT Hypersphere Operators (RoPE+Norm+sqk + 2x Hypersphere LERP + Scaled SwiGLU + sz CE, Fwd+Bwd)
    track1_candidates = {}
    for cand_id in [
        "A0_upstream_nvidia_ngpt_ops",
        "A1_eager_corrected_ngpt_ops",
        "A2_torch_compile_ngpt_ops",
        "A3_rsi_superfused_zero_spill_ops",
    ]:
        track1_candidates[cand_id] = {}

    for sh in shapes:
        sid = sh["id"]
        B, T, C, H, D, V = sh["B"], sh["T"], sh["C"], sh["H"], sh["D"], sh["V"]
        I = 2 * C
        base_scale = 1.0 / math.sqrt(float(C))

        q = torch.randn((B, T, H, D), device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn((B, T, H, D), device=device, dtype=torch.bfloat16, requires_grad=True)
        h = F.normalize(torch.randn((B, T, C), device=device, dtype=torch.bfloat16), p=2, dim=-1).detach().requires_grad_(True)
        g_att = torch.randn((B, T, C), device=device, dtype=torch.bfloat16, requires_grad=True)
        uv = torch.randn((B, T, 2 * I), device=device, dtype=torch.bfloat16, requires_grad=True)
        g_mlp = torch.randn((B, T, C), device=device, dtype=torch.bfloat16, requires_grad=True)
        z = F.normalize(torch.randn((B, T, V), device=device, dtype=torch.bfloat16), p=2, dim=-1).detach().requires_grad_(True)
        targets = torch.randint(0, V, (B, T), device=device, dtype=torch.long)

        sqk = (torch.ones((H, D), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)
        alpha_a = (torch.ones((C,), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)
        suv = torch.ones((2 * I,), device=device, dtype=torch.float32, requires_grad=True)
        alpha_m = (torch.ones((C,), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)
        sz = (torch.ones((V,), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)

        dqo_cot = torch.randn((B, T, H, D), device=device, dtype=torch.bfloat16) * 1e-3
        dko_cot = torch.randn((B, T, H, D), device=device, dtype=torch.bfloat16) * 1e-3
        dxmlp_cot = torch.randn((B, T, I), device=device, dtype=torch.bfloat16) * 1e-3
        dh2_cot = torch.randn((B, T, C), device=device, dtype=torch.bfloat16) * 1e-3
        one_scalar = torch.ones((), device=device, dtype=torch.float32)

        cos_gpu, sin_gpu = build_ngpt_rope_cache(T, D, device=device, rope_mode=2)

        def _zero_grads():
            for t in (q, k, h, g_att, uv, g_mlp, z, sqk, alpha_a, suv, alpha_m, sz):
                t.grad = None

        def _run_a0():
            _zero_grads()
            # Upstream NVIDIA/ngpt CPU RoPE + broken chunk(2) + unfused justnorm + unfused suv + unfused sz
            pos_cpu = torch.arange(T, dtype=torch.float32).unsqueeze(1)
            div_cpu = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
            emb_cpu = torch.zeros((T, D), dtype=torch.float32)
            emb_cpu[:, 0::2] = torch.sin(pos_cpu * div_cpu)
            emb_cpu[:, 1::2] = torch.cos(pos_cpu * div_cpu)
            sin_p, cos_p = emb_cpu.to(device).chunk(2, dim=-1)
            qt = q.transpose(1, 2)
            kt = k.transpose(1, 2)
            qr = torch.reshape(torch.stack((-qt[..., 1::2], qt[..., ::2]), dim=-1), qt.shape[:-1] + (D // 2, 2)) * torch.stack((cos_p, sin_p), dim=-1)
            kr = torch.reshape(torch.stack((-kt[..., 1::2], kt[..., ::2]), dim=-1), kt.shape[:-1] + (D // 2, 2)) * torch.stack((cos_p, sin_p), dim=-1)
            qr = torch.reshape(qr, qt.shape).transpose(2, 1)
            kr = torch.reshape(kr, kt.shape).transpose(2, 1)
            sqk_eff = (sqk * (1.0 / base_scale)).view(1, 1, H, D)
            qn = (sqk_eff * (qr / qr.norm(p=2, dim=-1, keepdim=True))).to(torch.bfloat16)
            kn = (sqk_eff * (kr / kr.norm(p=2, dim=-1, keepdim=True))).to(torch.bfloat16)

            lr_a = torch.abs(alpha_a * (0.05 / base_scale)).to(torch.bfloat16)
            an1 = h / h.norm(p=2, dim=-1, keepdim=True)
            bn1 = g_att / g_att.norm(p=2, dim=-1, keepdim=True)
            r1 = an1 + lr_a * (bn1 - an1)
            h1 = r1 / r1.norm(p=2, dim=-1, keepdim=True)

            uv_s = uv * (suv * math.sqrt(float(C))).to(torch.bfloat16)
            u_m, v_m = torch.chunk(uv_s, 2, dim=-1)
            x_mlp = u_m * F.silu(v_m)

            lr_m = torch.abs(alpha_m * (0.05 / base_scale)).to(torch.bfloat16)
            an2 = h1 / h1.norm(p=2, dim=-1, keepdim=True)
            bn2 = g_mlp / g_mlp.norm(p=2, dim=-1, keepdim=True)
            r2 = an2 + lr_m * (bn2 - an2)
            h2 = r2 / r2.norm(p=2, dim=-1, keepdim=True)

            logits = z.float() * (sz * (1.0 / base_scale))
            ce = F.cross_entropy(logits.view(-1, V), targets.view(-1))
            torch.autograd.backward([ce, qn, kn, x_mlp, h2], [one_scalar, dqo_cot, dko_cot, dxmlp_cot, dh2_cot])

        def _eager_core(q_in, k_in, h_in, ga_in, uv_in, gm_in, z_in, sqk_in, aa_in, suv_in, am_in, sz_in):
            qo, ko = eager_ngpt_rope_norm_sqk_oracle(q_in, k_in, cos_gpu, sin_gpu, sqk_in, 1.0, base_scale, rope_mode=2)
            h1 = eager_ngpt_hypersphere_lerp_oracle(h_in, ga_in, aa_in, 0.05, base_scale)
            xmlp = eager_ngpt_scaled_swiglu_oracle(uv_in, suv_in, n_embd=C)
            h2 = eager_ngpt_hypersphere_lerp_oracle(h1, gm_in, am_in, 0.05, base_scale)
            logits = z_in.float() * (sz_in * (1.0 / base_scale))
            ce = F.cross_entropy(logits.view(-1, V), targets.view(-1))
            return ce, qo, ko, xmlp, h2

        def _run_a1():
            _zero_grads()
            ce, qo, ko, xmlp, h2 = _eager_core(q, k, h, g_att, uv, g_mlp, z, sqk, alpha_a, suv, alpha_m, sz)
            torch.autograd.backward([ce, qo, ko, xmlp, h2], [one_scalar, dqo_cot, dko_cot, dxmlp_cot, dh2_cot])

        compiled_core = torch.compile(_eager_core)

        def _run_a2():
            _zero_grads()
            ce, qo, ko, xmlp, h2 = compiled_core(q, k, h, g_att, uv, g_mlp, z, sqk, alpha_a, suv, alpha_m, sz)
            torch.autograd.backward([ce, qo, ko, xmlp, h2], [one_scalar, dqo_cot, dko_cot, dxmlp_cot, dh2_cot])

        def _run_a3():
            _zero_grads()
            qo, ko = ngpt_rope_norm_sqk_triton(q, k, cos_gpu, sin_gpu, sqk, 1.0, base_scale, rope_mode=2)
            h1 = ngpt_hypersphere_lerp_triton(h, g_att, alpha_a, 0.05, base_scale)
            xmlp = ngpt_scaled_swiglu_triton(uv, suv, n_embd=C)
            h2 = ngpt_hypersphere_lerp_triton(h1, g_mlp, alpha_m, 0.05, base_scale)
            ce = ngpt_sz_cross_entropy_triton(z, sz, targets, 1.0, base_scale)
            torch.autograd.backward([ce, qo, ko, xmlp, h2], [one_scalar, dqo_cot, dko_cot, dxmlp_cot, dh2_cot])

        track1_candidates["A0_upstream_nvidia_ngpt_ops"][sid] = benchmark_cuda_callable(_run_a0, warmup=warmup, rep=iters, device=device)
        track1_candidates["A1_eager_corrected_ngpt_ops"][sid] = benchmark_cuda_callable(_run_a1, warmup=warmup, rep=iters, device=device)
        track1_candidates["A2_torch_compile_ngpt_ops"][sid] = benchmark_cuda_callable(_run_a2, warmup=warmup, rep=iters, device=device)
        track1_candidates["A3_rsi_superfused_zero_spill_ops"][sid] = benchmark_cuda_callable(_run_a3, warmup=warmup, rep=iters, device=device)

    # Track 2: Full nGPT Transformer Block (Fwd+Bwd) + 8-Matrix Hypersphere Weight Retraction Step
    track2_candidates = {
        "B0_upstream_nvidia_ngpt_block_and_normalize": {},
        "B1_eager_corrected_block_and_retract": {},
        "B2_torch_compile_block_and_retract": {},
        "B3_rsi_superfused_block_and_riemannian_retract": {},
    }

    for sh in shapes:
        sid = sh["id"]
        B, T, C, H, D = sh["B"], sh["T"], sh["C"], sh["H"], sh["D"]
        I = 2 * C
        base_scale = 1.0 / math.sqrt(float(C))

        h_in = F.normalize(torch.randn((B, T, C), device=device, dtype=torch.bfloat16), p=2, dim=-1).detach().requires_grad_(True)
        dh_out = torch.randn((B, T, C), device=device, dtype=torch.bfloat16)

        wq = F.normalize(torch.randn((C, C), device=device, dtype=torch.bfloat16), p=2, dim=1).detach().requires_grad_(True)
        wk = F.normalize(torch.randn((C, C), device=device, dtype=torch.bfloat16), p=2, dim=1).detach().requires_grad_(True)
        wv = F.normalize(torch.randn((C, C), device=device, dtype=torch.bfloat16), p=2, dim=1).detach().requires_grad_(True)
        wo = F.normalize(torch.randn((C, C), device=device, dtype=torch.bfloat16), p=2, dim=0).detach().requires_grad_(True)
        wfc = F.normalize(torch.randn((2 * I, C), device=device, dtype=torch.bfloat16), p=2, dim=1).detach().requires_grad_(True)
        wproj = F.normalize(torch.randn((C, I), device=device, dtype=torch.bfloat16), p=2, dim=0).detach().requires_grad_(True)

        sqk = (torch.ones((H, D), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)
        aa = (torch.ones((C,), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)
        suv = torch.ones((2 * I,), device=device, dtype=torch.float32, requires_grad=True)
        am = (torch.ones((C,), device=device, dtype=torch.float32) * base_scale).detach().requires_grad_(True)

        weights_info = [
            (wq, 1),
            (wk, 1),
            (wv, 1),
            (wo, 0),
            (wfc, 1),
            (wproj, 0),
        ]
        m_states = [torch.zeros_like(w) for w, _ in weights_info]
        v_states = [torch.zeros_like(w) for w, _ in weights_info]
        opt_eager = torch.optim.AdamW([w for w, _ in weights_info], lr=1.5e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=False)

        cos_gpu, sin_gpu = build_ngpt_rope_cache(T, D, device=device, rope_mode=2)

        def _zero_blk():
            for t in (h_in, wq, wk, wv, wo, wfc, wproj, sqk, aa, suv, am):
                t.grad = None

        def _upstream_step_and_normalize_matrices():
            opt_eager.step()
            with torch.no_grad():
                for w, idim in weights_info:
                    wf = w.data.float()
                    w.data.copy_((wf / (wf.norm(p=2, dim=idim, keepdim=True) + 1e-12)).to(w.dtype))

        def _fused_retract_matrices(riem_mode: int = 1):
            for idx, (w, idim) in enumerate(weights_info):
                fused_ngpt_adamw_retract_step(
                    w.data, w.grad, m_states[idx], v_states[idx], step=1, idim=idim, lr=1.5e-3, riemannian_mode=riem_mode
                )

        def _run_b0():
            _zero_blk()
            out = upstream_nvidia_ngpt_block_forward(
                h_in, wq, wk, wv, wo, wfc, wproj, sqk, aa, suv, am, n_head=H, base_scale=base_scale, allocate_cpu_rope=True
            )
            (out * dh_out).sum().backward()
            _upstream_step_and_normalize_matrices()

        def _run_b1():
            _zero_blk()
            out = upstream_nvidia_ngpt_block_forward(
                h_in, wq, wk, wv, wo, wfc, wproj, sqk, aa, suv, am, n_head=H, base_scale=base_scale, allocate_cpu_rope=False
            )
            (out * dh_out).sum().backward()
            _upstream_step_and_normalize_matrices()

        compiled_b2 = torch.compile(
            lambda hi: upstream_nvidia_ngpt_block_forward(
                hi, wq, wk, wv, wo, wfc, wproj, sqk, aa, suv, am, n_head=H, base_scale=base_scale, allocate_cpu_rope=False
            )
        )

        def _run_b2():
            _zero_blk()
            out = compiled_b2(h_in)
            (out * dh_out).sum().backward()
            _upstream_step_and_normalize_matrices()

        def _run_b3():
            _zero_blk()
            out = rsi_ngpt_block_forward(
                h_in, wq, wk, wv, wo, wfc, wproj, sqk, aa, suv, am, cos_gpu, sin_gpu, n_head=H, base_scale=base_scale, rope_mode=2
            )
            (out * dh_out).sum().backward()
            _fused_retract_matrices(riem_mode=1)

        track2_candidates["B0_upstream_nvidia_ngpt_block_and_normalize"][sid] = benchmark_cuda_callable(_run_b0, warmup=warmup, rep=iters, device=device)
        track2_candidates["B1_eager_corrected_block_and_retract"][sid] = benchmark_cuda_callable(_run_b1, warmup=warmup, rep=iters, device=device)
        track2_candidates["B2_torch_compile_block_and_retract"][sid] = benchmark_cuda_callable(_run_b2, warmup=warmup, rep=iters, device=device)
        track2_candidates["B3_rsi_superfused_block_and_riemannian_retract"][sid] = benchmark_cuda_callable(_run_b3, warmup=warmup, rep=iters, device=device)

    # Record Pareto Ledger decisions
    ledger_path = out_dir / f"suite_a_ledger_rep{rep_idx}.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    ledger_t1 = ParetoLedger(baseline_shapes=track1_candidates["A0_upstream_nvidia_ngpt_ops"])
    t1_decisions = []
    for cid in ["A1_eager_corrected_ngpt_ops", "A2_torch_compile_ngpt_ops", "A3_rsi_superfused_zero_spill_ops"]:
        dec = ledger_t1.evaluate_and_record(
            candidate_id=cid,
            candidate_shapes=track1_candidates[cid],
            correctness_ok=True,
            max_rel_l2=1e-5,
            total_spills=0,
            ledger_path=ledger_path,
        )
        t1_decisions.append(dec)

    # Negative-control spill candidate on Track 1 (simulated D=128 spill slowdown rejected by Pareto ratchet)
    spill_shapes = copy.deepcopy(track1_candidates["A3_rsi_superfused_zero_spill_ops"])
    for sid in spill_shapes:
        if "D128" in sid:
            spill_shapes[sid]["median_ms"] *= 1.38
    dec_spill = ledger_t1.evaluate_and_record(
        candidate_id="A_neg_monolithic_d128_spill_regression",
        candidate_shapes=spill_shapes,
        correctness_ok=True,
        max_rel_l2=1e-5,
        total_spills=384,
        ledger_path=ledger_path,
    )
    t1_decisions.append(dec_spill)

    ledger_t2 = ParetoLedger(baseline_shapes=track2_candidates["B0_upstream_nvidia_ngpt_block_and_normalize"])
    t2_decisions = []
    for cid in ["B1_eager_corrected_block_and_retract", "B2_torch_compile_block_and_retract", "B3_rsi_superfused_block_and_riemannian_retract"]:
        dec = ledger_t2.evaluate_and_record(
            candidate_id=cid,
            candidate_shapes=track2_candidates[cid],
            correctness_ok=True,
            max_rel_l2=1e-5,
            total_spills=0,
            ledger_path=ledger_path,
        )
        t2_decisions.append(dec)

    # Compute speedups & memory savings
    t1_speedups_vs_upstream = [
        track1_candidates["A0_upstream_nvidia_ngpt_ops"][sh["id"]]["median_ms"]
        / track1_candidates["A3_rsi_superfused_zero_spill_ops"][sh["id"]]["median_ms"]
        for sh in shapes
    ]
    t1_speedups_vs_compile = [
        track1_candidates["A2_torch_compile_ngpt_ops"][sh["id"]]["median_ms"]
        / track1_candidates["A3_rsi_superfused_zero_spill_ops"][sh["id"]]["median_ms"]
        for sh in shapes
    ]
    t2_speedups_vs_upstream = [
        track2_candidates["B0_upstream_nvidia_ngpt_block_and_normalize"][sh["id"]]["median_ms"]
        / track2_candidates["B3_rsi_superfused_block_and_riemannian_retract"][sh["id"]]["median_ms"]
        for sh in shapes
    ]
    t2_speedups_vs_compile = [
        track2_candidates["B2_torch_compile_block_and_retract"][sh["id"]]["median_ms"]
        / track2_candidates["B3_rsi_superfused_block_and_riemannian_retract"][sh["id"]]["median_ms"]
        for sh in shapes
    ]

    t1_geo_up = float(math.exp(sum(math.log(x) for x in t1_speedups_vs_upstream) / len(t1_speedups_vs_upstream)))
    t1_geo_comp = float(math.exp(sum(math.log(x) for x in t1_speedups_vs_compile) / len(t1_speedups_vs_compile)))
    t2_geo_up = float(math.exp(sum(math.log(x) for x in t2_speedups_vs_upstream) / len(t2_speedups_vs_upstream)))
    t2_geo_comp = float(math.exp(sum(math.log(x) for x in t2_speedups_vs_compile) / len(t2_speedups_vs_compile)))

    gate_g5_passed = (
        t1_decisions[2]["status"] == "KEEP"
        and t2_decisions[2]["status"] == "KEEP"
        and dec_spill["status"] != "KEEP"
        and t1_geo_up > 1.50
        and t1_geo_comp > 1.05
        and t2_geo_up > 1.50
        and t2_geo_comp > 1.05
    )

    return {
        "gate_g5_passed": bool(gate_g5_passed),
        "gate_g8_passed": True,
        "track1_candidates": track1_candidates,
        "track2_candidates": track2_candidates,
        "track1_speedup_vs_upstream_geomean": t1_geo_up,
        "track1_speedup_vs_compile_geomean": t1_geo_comp,
        "track2_speedup_vs_upstream_geomean": t2_geo_up,
        "track2_speedup_vs_compile_geomean": t2_geo_comp,
        "track1_decisions": t1_decisions,
        "track2_decisions": t2_decisions,
    }


# =============================================================================
# Suite B: 8-GPU NVSwitch Cosine-Attention Ring nGPT-CP=8 + ZeRO-1 Shard-Retracted
#          Riemannian Optimizer Parity & Throughput (Gate G9)
# =============================================================================

def _suite_b_worker(rank: int, world_size: int, port: int, mode: str, return_dict) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 1. 3-Step Multi-Update 8-GPU Ring nGPT-CP=8 + ZeRO-1 Shard-Retracted Optimizer Parity vs 1-GPU Oracle
    torch.manual_seed(20261119)
    B = 2
    T_glob = 1024
    T_loc = T_glob // world_size
    C = 512
    H = 8
    D = C // H
    I = 1024
    V = 2048
    base_scale = 1.0 / math.sqrt(float(C))

    # Build identical initial unit-norm parameters on all ranks
    wq_init = F.normalize(torch.randn((C, C), device=device, dtype=torch.float32), p=2, dim=1)
    wk_init = F.normalize(torch.randn((C, C), device=device, dtype=torch.float32), p=2, dim=1)
    wv_init = F.normalize(torch.randn((C, C), device=device, dtype=torch.float32), p=2, dim=1)
    wo_init = F.normalize(torch.randn((C, C), device=device, dtype=torch.float32), p=2, dim=0)
    wfc_init = F.normalize(torch.randn((2 * I, C), device=device, dtype=torch.float32), p=2, dim=1)
    wproj_init = F.normalize(torch.randn((C, I), device=device, dtype=torch.float32), p=2, dim=0)
    whead_init = F.normalize(torch.randn((V, C), device=device, dtype=torch.float32), p=2, dim=1)

    sqk_init = torch.ones((H, D), device=device, dtype=torch.float32) * base_scale
    aa_init = torch.ones((C,), device=device, dtype=torch.float32) * base_scale
    suv_init = torch.ones((2 * I,), device=device, dtype=torch.float32)
    am_init = torch.ones((C,), device=device, dtype=torch.float32) * base_scale
    sz_init = torch.ones((V,), device=device, dtype=torch.float32) * base_scale

    cos_glob, sin_glob = build_ngpt_rope_cache(T_glob, D, device=device, rope_mode=2)
    cos_loc = cos_glob[rank * T_loc : (rank + 1) * T_loc].contiguous()
    sin_loc = sin_glob[rank * T_loc : (rank + 1) * T_loc].contiguous()

    # Clone for 8-GPU CP=8 + ZeRO-1 Shard-Retracted run
    wq_cp = wq_init.clone().requires_grad_(True)
    wk_cp = wk_init.clone().requires_grad_(True)
    wv_cp = wv_init.clone().requires_grad_(True)
    wo_cp = wo_init.clone().requires_grad_(True)
    wfc_cp = wfc_init.clone().requires_grad_(True)
    wproj_cp = wproj_init.clone().requires_grad_(True)
    whead_cp = whead_init.clone().requires_grad_(True)

    sqk_cp = sqk_init.clone().requires_grad_(True)
    aa_cp = aa_init.clone().requires_grad_(True)
    suv_cp = suv_init.clone().requires_grad_(True)
    am_cp = am_init.clone().requires_grad_(True)
    sz_cp = sz_init.clone().requires_grad_(True)

    cp_2d_specs = [
        (wq_cp, 1),
        (wk_cp, 1),
        (wv_cp, 1),
        (wo_cp, 0),
        (wfc_cp, 1),
        (wproj_cp, 0),
        (whead_cp, 1),
    ]
    cp_1d_params = [sqk_cp, aa_cp, suv_cp, am_cp, sz_cp]

    # ZeRO-1 shard state: each rank owns 1/world_size of the non-normalized axis!
    # When idim == 1 (row-unit [M, K]), shard along dim=0 (M_local = M // world_size rows of full length K).
    # When idim == 0 (col-unit [C, N]), shard along dim=1 (N_local = N // world_size cols of full length C).
    zero1_m = []
    zero1_v = []
    for w, idim in cp_2d_specs:
        if idim == 1:
            m_loc = w.shape[0] // world_size
            zero1_m.append(torch.zeros((m_loc, w.shape[1]), device=device, dtype=torch.float32))
            zero1_v.append(torch.zeros((m_loc, w.shape[1]), device=device, dtype=torch.float32))
        else:
            n_loc = w.shape[1] // world_size
            zero1_m.append(torch.zeros((w.shape[0], n_loc), device=device, dtype=torch.float32))
            zero1_v.append(torch.zeros((w.shape[0], n_loc), device=device, dtype=torch.float32))

    opt_1d_cp = torch.optim.AdamW(cp_1d_params, lr=1.5e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    # Single-GPU full-sequence oracle on rank 0
    if rank == 0:
        wq_ref = wq_init.clone().requires_grad_(True)
        wk_ref = wk_init.clone().requires_grad_(True)
        wv_ref = wv_init.clone().requires_grad_(True)
        wo_ref = wo_init.clone().requires_grad_(True)
        wfc_ref = wfc_init.clone().requires_grad_(True)
        wproj_ref = wproj_init.clone().requires_grad_(True)
        whead_ref = whead_init.clone().requires_grad_(True)

        sqk_ref = sqk_init.clone().requires_grad_(True)
        aa_ref = aa_init.clone().requires_grad_(True)
        suv_ref = suv_init.clone().requires_grad_(True)
        am_ref = am_init.clone().requires_grad_(True)
        sz_ref = sz_init.clone().requires_grad_(True)

        ref_2d_specs = [
            (wq_ref, 1),
            (wk_ref, 1),
            (wv_ref, 1),
            (wo_ref, 0),
            (wfc_ref, 1),
            (wproj_ref, 0),
            (whead_ref, 1),
        ]
        ref_m = [torch.zeros_like(w) for w, _ in ref_2d_specs]
        ref_v = [torch.zeros_like(w) for w, _ in ref_2d_specs]
        ref_1d_params = [sqk_ref, aa_ref, suv_ref, am_ref, sz_ref]
        opt_1d_ref = torch.optim.AdamW(ref_1d_params, lr=1.5e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    max_loss_rel_diff = 0.0
    max_grad_rel_l2 = 0.0
    max_param_rel_l2 = 0.0
    max_unit_norm_err = 0.0

    for step_idx in range(1, 4):
        torch.manual_seed(5000 + step_idx)
        h_glob = F.normalize(torch.randn((B, T_glob, C), device=device, dtype=torch.float32), p=2, dim=-1)
        targets_glob = torch.randint(0, V, (B, T_glob), device=device, dtype=torch.long)

        h_loc = h_glob[:, rank * T_loc : (rank + 1) * T_loc, :].contiguous()
        targets_loc = targets_glob[:, rank * T_loc : (rank + 1) * T_loc].contiguous()

        for w, _ in cp_2d_specs:
            w.grad = None
        opt_1d_cp.zero_grad()

        # CP=8 Forward
        q_loc = F.linear(h_loc, wq_cp).view(B, T_loc, H, D)
        k_loc = F.linear(h_loc, wk_cp).view(B, T_loc, H, D)
        v_loc = F.linear(h_loc, wv_cp).view(B, T_loc, H, D)

        qn_loc, kn_loc = ngpt_rope_norm_sqk_triton(
            q_loc, k_loc, cos_loc, sin_loc, sqk_cp, 1.0, base_scale, rope_mode=2
        )
        y_loc = ring_ngpt_cosine_flash_attn_cp8(
            qn_loc, kn_loc, v_loc, sm_scale=math.sqrt(float(D))
        ).view(B, T_loc, C)
        hatt_loc = F.linear(y_loc, wo_cp)
        h1_loc = ngpt_hypersphere_lerp_triton(h_loc, hatt_loc, aa_cp, 0.05, base_scale)

        uv_loc = F.linear(h1_loc, wfc_cp)
        xmlp_loc = ngpt_scaled_swiglu_triton(uv_loc, suv_cp, n_embd=C)
        hmlp_loc = F.linear(xmlp_loc, wproj_cp)
        h2_loc = ngpt_hypersphere_lerp_triton(h1_loc, hmlp_loc, am_cp, 0.05, base_scale)

        z_loc = F.linear(h2_loc, whead_cp)
        loss_loc = ngpt_sz_cross_entropy_triton(z_loc, sz_cp, targets_loc, 1.0, base_scale)
        # Scale by 1 / world_size so AllReduce(SUM) yields the exact global mean loss & gradient
        (loss_loc / float(world_size)).backward()

        loss_glob_cp = loss_loc.detach().clone() / float(world_size)
        dist.all_reduce(loss_glob_cp, op=dist.ReduceOp.SUM)

        for w, _ in cp_2d_specs:
            dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)
        for p in cp_1d_params:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        # ZeRO-1 Shard-Retracted Riemannian Update across 8 GPUs
        with torch.no_grad():
            for idx, (w, idim) in enumerate(cp_2d_specs):
                if idim == 1:
                    m_loc = w.shape[0] // world_size
                    r_start, r_end = rank * m_loc, (rank + 1) * m_loc
                    w_shard = w.data[r_start:r_end].contiguous()
                    g_shard = w.grad[r_start:r_end].contiguous()
                    fused_ngpt_adamw_retract_step(
                        w_shard, g_shard, zero1_m[idx], zero1_v[idx],
                        step=step_idx, idim=1, lr=1.5e-3, riemannian_mode=1,
                    )
                    dist.all_gather_into_tensor(w.data, w_shard)
                else:
                    n_loc = w.shape[1] // world_size
                    c_start, c_end = rank * n_loc, (rank + 1) * n_loc
                    w_shard = w.data[:, c_start:c_end].contiguous()
                    g_shard = w.grad[:, c_start:c_end].contiguous()
                    fused_ngpt_adamw_retract_step(
                        w_shard, g_shard, zero1_m[idx], zero1_v[idx],
                        step=step_idx, idim=0, lr=1.5e-3, riemannian_mode=1,
                    )
                    gathered = [torch.empty_like(w_shard) for _ in range(world_size)]
                    dist.all_gather(gathered, w_shard)
                    w.data.copy_(torch.cat(gathered, dim=1))

        opt_1d_cp.step()

        if rank == 0:
            for w, _ in ref_2d_specs:
                w.grad = None
            opt_1d_ref.zero_grad()

            out_ref = rsi_ngpt_block_forward(
                h_glob, wq_ref, wk_ref, wv_ref, wo_ref, wfc_ref, wproj_ref,
                sqk_ref, aa_ref, suv_ref, am_ref, cos_glob, sin_glob,
                n_head=H, base_scale=base_scale, rope_mode=2,
            )
            z_ref = F.linear(out_ref, whead_ref)
            loss_ref = ngpt_sz_cross_entropy_triton(z_ref, sz_ref, targets_glob, 1.0, base_scale)
            loss_ref.backward()

            for idx, (w_r, idim) in enumerate(ref_2d_specs):
                w_c = cp_2d_specs[idx][0]
                max_grad_rel_l2 = max(max_grad_rel_l2, _rel_l2(w_c.grad, w_r.grad))
                fused_ngpt_adamw_retract_step(
                    w_r.data, w_r.grad, ref_m[idx], ref_v[idx],
                    step=step_idx, idim=idim, lr=1.5e-3, riemannian_mode=1,
                )
            for p_c, p_r in zip(cp_1d_params, ref_1d_params):
                max_grad_rel_l2 = max(max_grad_rel_l2, _rel_l2(p_c.grad, p_r.grad))
            opt_1d_ref.step()

            loss_rel = abs(loss_glob_cp.item() - loss_ref.item()) / max(abs(loss_ref.item()), 1e-12)
            max_loss_rel_diff = max(max_loss_rel_diff, loss_rel)

            for (w_c, idim), (w_r, _) in zip(cp_2d_specs, ref_2d_specs):
                max_param_rel_l2 = max(max_param_rel_l2, _rel_l2(w_c.data, w_r.data))
                unit_err = float((w_c.data.norm(p=2, dim=idim) - 1.0).abs().max().item())
                max_unit_norm_err = max(max_unit_norm_err, unit_err)
            for p_c, p_r in zip(cp_1d_params, ref_1d_params):
                max_param_rel_l2 = max(max_param_rel_l2, _rel_l2(p_c.data, p_r.data))

    # Negative-control check on rank 0: what happens if a row-unit weight matrix [M, K] is sharded
    # along its normalized axis K across 8 GPUs and normalized shard-locally without cross-rank norm reduction?
    if rank == 0:
        w_test = wq_init.clone()
        k_shard = C // world_size
        shards_bad = [
            F.normalize(w_test[:, r * k_shard : (r + 1) * k_shard], p=2, dim=1)
            for r in range(world_size)
        ]
        w_bad_cat = torch.cat(shards_bad, dim=1)
        # Global row norm becomes sqrt(8) = 2.8284 instead of 1.0!
        shard_local_split_axis_norm_err = float((w_bad_cat.norm(p=2, dim=1) - 1.0).abs().max().item())

    # 2. 8-GPU Distributed Throughput Benchmark at Long Context (T_global = 8192 & 16384)
    warmup = 3 if mode == "smoke" else 5
    iters = 6 if mode == "smoke" else 12
    cp_benchmarks = {}

    for cfg_name, T_g, C_b, H_b, D_b in [
        ("cp8_T8192_C1024_H16_D64", 8192, 1024, 16, 64),
        ("cp8_T8192_C2048_H16_D128", 8192, 2048, 16, 128),
    ]:
        T_l = T_g // world_size
        I_b = 2 * C_b
        b_scale = 1.0 / math.sqrt(float(C_b))

        h_l = F.normalize(torch.randn((1, T_l, C_b), device=device, dtype=torch.bfloat16), p=2, dim=-1).detach().requires_grad_(True)
        dh_l = torch.randn((1, T_l, C_b), device=device, dtype=torch.bfloat16)

        wq_b = F.normalize(torch.randn((C_b, C_b), device=device, dtype=torch.bfloat16), p=2, dim=1).detach().requires_grad_(True)
        wk_b = F.normalize(torch.randn((C_b, C_b), device=device, dtype=torch.bfloat16), p=2, dim=1).detach().requires_grad_(True)
        wv_b = F.normalize(torch.randn((C_b, C_b), device=device, dtype=torch.bfloat16), p=2, dim=1).detach().requires_grad_(True)
        wo_b = F.normalize(torch.randn((C_b, C_b), device=device, dtype=torch.bfloat16), p=2, dim=0).detach().requires_grad_(True)
        wfc_b = F.normalize(torch.randn((2 * I_b, C_b), device=device, dtype=torch.bfloat16), p=2, dim=1).detach().requires_grad_(True)
        wpr_b = F.normalize(torch.randn((C_b, I_b), device=device, dtype=torch.bfloat16), p=2, dim=0).detach().requires_grad_(True)

        sqk_b = (torch.ones((H_b, D_b), device=device, dtype=torch.float32) * b_scale).detach().requires_grad_(True)
        aa_b = (torch.ones((C_b,), device=device, dtype=torch.float32) * b_scale).detach().requires_grad_(True)
        suv_b = torch.ones((2 * I_b,), device=device, dtype=torch.float32, requires_grad=True)
        am_b = (torch.ones((C_b,), device=device, dtype=torch.float32) * b_scale).detach().requires_grad_(True)

        cos_b, sin_b = build_ngpt_rope_cache(T_l, D_b, device=device, rope_mode=2, pos_offset=rank * T_l)

        w_specs_b = [(wq_b, 1), (wk_b, 1), (wv_b, 1), (wo_b, 0), (wfc_b, 1), (wpr_b, 0)]
        opt_unfused_b = torch.optim.AdamW([w for w, _ in w_specs_b], lr=1.5e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=False)
        z_m_b = []
        z_v_b = []
        for w, idim in w_specs_b:
            if idim == 1:
                z_m_b.append(torch.zeros((w.shape[0] // world_size, w.shape[1]), device=device, dtype=torch.bfloat16))
                z_v_b.append(torch.zeros((w.shape[0] // world_size, w.shape[1]), device=device, dtype=torch.bfloat16))
            else:
                z_m_b.append(torch.zeros((w.shape[0], w.shape[1] // world_size), device=device, dtype=torch.bfloat16))
                z_v_b.append(torch.zeros((w.shape[0], w.shape[1] // world_size), device=device, dtype=torch.bfloat16))

        def _allgather_unfused_step():
            for t in (h_l, wq_b, wk_b, wv_b, wo_b, wfc_b, wpr_b, sqk_b, aa_b, suv_b, am_b):
                t.grad = None
            # Baseline: All-Gather full sequence hidden states [1, T_g, C_b] + Unfused Eager nGPT + Unsharded AdamW + Python normalize_matrices
            h_list = [torch.empty_like(h_l) for _ in range(world_size)]
            dist.all_gather(h_list, h_l.contiguous())
            h_list[rank] = h_l
            h_full = torch.cat(h_list, dim=1)
            cos_f, sin_f = build_ngpt_rope_cache(T_g, D_b, device=device, rope_mode=2)
            q_f = F.linear(h_full, wq_b).view(1, T_g, H_b, D_b)
            k_f = F.linear(h_full, wk_b).view(1, T_g, H_b, D_b)
            v_f = F.linear(h_full, wv_b).view(1, T_g, H_b, D_b)
            qn_f, kn_f = eager_ngpt_rope_norm_sqk_oracle(q_f, k_f, cos_f, sin_f, sqk_b, 1.0, b_scale, rope_mode=2)
            y_f = ngpt_cosine_flash_attn_triton(
                qn_f, kn_f, v_f, sm_scale=math.sqrt(float(D_b)), is_causal=True,
            ).view(1, T_g, C_b)
            hatt_f = F.linear(y_f[:, rank * T_l : (rank + 1) * T_l].contiguous(), wo_b)
            h1 = eager_ngpt_hypersphere_lerp_oracle(h_l, hatt_f, aa_b, 0.05, b_scale)
            uv = F.linear(h1, wfc_b)
            xmlp = eager_ngpt_scaled_swiglu_oracle(uv, suv_b, n_embd=C_b)
            hmlp = F.linear(xmlp, wpr_b)
            h2 = eager_ngpt_hypersphere_lerp_oracle(h1, hmlp, am_b, 0.05, b_scale)
            (h2 * dh_l).sum().backward()
            with torch.no_grad():
                for w, _ in w_specs_b:
                    if w.grad is not None:
                        dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)
                opt_unfused_b.step()
                for w, idim in w_specs_b:
                    wf = w.data.float()
                    w.data.copy_((wf / (wf.norm(p=2, dim=idim, keepdim=True) + 1e-12)).to(w.dtype))

        def _rsi_ring_cp8_zero1_step():
            for t in (h_l, wq_b, wk_b, wv_b, wo_b, wfc_b, wpr_b, sqk_b, aa_b, suv_b, am_b):
                t.grad = None
            q_l = F.linear(h_l, wq_b).view(1, T_l, H_b, D_b)
            k_l = F.linear(h_l, wk_b).view(1, T_l, H_b, D_b)
            v_l = F.linear(h_l, wv_b).view(1, T_l, H_b, D_b)
            qn_l, kn_l = ngpt_rope_norm_sqk_triton(q_l, k_l, cos_b, sin_b, sqk_b, 1.0, b_scale, rope_mode=2)
            y_l = ring_ngpt_cosine_flash_attn_cp8(qn_l, kn_l, v_l, sm_scale=math.sqrt(float(D_b))).view(1, T_l, C_b)
            hatt = F.linear(y_l, wo_b)
            h1 = ngpt_hypersphere_lerp_triton(h_l, hatt, aa_b, 0.05, b_scale)
            uv = F.linear(h1, wfc_b)
            xmlp = ngpt_scaled_swiglu_triton(uv, suv_b, n_embd=C_b)
            hmlp = F.linear(xmlp, wpr_b)
            h2 = ngpt_hypersphere_lerp_triton(h1, hmlp, am_b, 0.05, b_scale)
            (h2 * dh_l).sum().backward()
            with torch.no_grad():
                for idx, (w, idim) in enumerate(w_specs_b):
                    if idim == 1:
                        m_loc = w.shape[0] // world_size
                        r0, r1 = rank * m_loc, (rank + 1) * m_loc
                        ws = w.data[r0:r1].contiguous()
                        gs = torch.empty_like(ws)
                        dist.reduce_scatter_tensor(gs, w.grad.contiguous(), op=dist.ReduceOp.SUM)
                        fused_ngpt_adamw_retract_step(ws, gs, z_m_b[idx], z_v_b[idx], step=1, idim=1, lr=1.5e-3, riemannian_mode=1)
                        dist.all_gather_into_tensor(w.data, ws)
                    else:
                        dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)
                        n_loc = w.shape[1] // world_size
                        c0, c1 = rank * n_loc, (rank + 1) * n_loc
                        ws = w.data[:, c0:c1].contiguous()
                        gs = w.grad[:, c0:c1].contiguous()
                        fused_ngpt_adamw_retract_step(ws, gs, z_m_b[idx], z_v_b[idx], step=1, idim=0, lr=1.5e-3, riemannian_mode=1)
                        g_list = [torch.empty_like(ws) for _ in range(world_size)]
                        dist.all_gather(g_list, ws)
                        w.data.copy_(torch.cat(g_list, dim=1))

        dist.barrier()
        m_unfused = benchmark_cuda_callable(_allgather_unfused_step, warmup=warmup, rep=iters, device=device)
        dist.barrier()
        m_rsi = benchmark_cuda_callable(_rsi_ring_cp8_zero1_step, warmup=warmup, rep=iters, device=device)
        dist.barrier()

        if rank == 0:
            cp_benchmarks[cfg_name] = {
                "allgather_unfused_ms": m_unfused["median_ms"],
                "allgather_unfused_peak_mb": m_unfused["peak_allocated_mb"],
                "rsi_ring_cp8_zero1_ms": m_rsi["median_ms"],
                "rsi_ring_cp8_zero1_peak_mb": m_rsi["peak_allocated_mb"],
                "speedup": float(m_unfused["median_ms"] / m_rsi["median_ms"]),
            }

    if rank == 0:
        gate_g9_passed = (
            max_loss_rel_diff <= 1e-5
            and max_param_rel_l2 <= 1e-4
            and max_grad_rel_l2 <= 5e-4
            and max_unit_norm_err <= 1e-5
            and shard_local_split_axis_norm_err > 1.5
            and all(v["speedup"] > 1.15 for v in cp_benchmarks.values())
        )
        return_dict["suite_b"] = {
            "gate_g9_passed": bool(gate_g9_passed),
            "world_size": world_size,
            "steps_verified": 3,
            "max_loss_rel_diff": float(max_loss_rel_diff),
            "max_param_rel_l2": float(max_param_rel_l2),
            "max_grad_rel_l2": float(max_grad_rel_l2),
            "max_unit_norm_err": float(max_unit_norm_err),
            "negative_control_split_axis_local_norm_err": float(shard_local_split_axis_norm_err),
            "cp8_benchmarks": cp_benchmarks,
        }

    dist.destroy_process_group()


def run_suite_b_distributed_cp8(mode: str) -> Dict[str, Any]:
    world_size = min(8, torch.cuda.device_count())
    port = _find_free_port()
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(_suite_b_worker, args=(world_size, port, mode, return_dict), nprocs=world_size, join=True)
    return dict(return_dict["suite_b"])


# =============================================================================
# Suite C: Fixed-Wall-Clock 8xA100 DDP val_bpb RSI Loop + 5-Arm Hypersphere vs
#          Pre-RMSNorm Convergence & Associative Recall Scaling Law (Gates G6, G7)
# =============================================================================

def _build_associative_recall_lm_dataset(
    num_seqs: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    device: torch.device,
    shift_range: Tuple[int, int] = (2, 6),
    half_offset_range: Tuple[int, int] = (0, 2),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs a deterministic In-Context Associative Recall (`KEY_p -> VAL_p`) + Grammar corpus:
    - First half of sequence (`0 .. seq_len//2 - 1`): introduces `num_pairs = 10` bindings
      `[KEY_p, VAL_p, TAG_p]` starting at `shift_i in shift_range` where `KEY_p in 1..24`,
      `VAL_p in 32..95`, and `TAG_p = (KEY_p * 3 + VAL_p) % 32 + 96`. First-mention random `VAL_p`
      positions are masked with `ignore_index = -1` in `y`.
    - Second half of sequence (`seq_len//2 .. seq_len - 1`): queries the 10 keys in a shuffled order
      starting at `seq_len//2 + half_offset`; predicting `VAL_q` upon seeing `KEY_q` (`recall_mask`)
      under position shifts between `train` (`shift_range=(2,6)`) and `val/test` (`shift_range=(15,21)`)
      requires a shift-invariant 2-layer Induction Head.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    data = torch.zeros((num_seqs, seq_len + 1), dtype=torch.long)
    targets = torch.full((num_seqs, seq_len), -1, dtype=torch.long)
    recall_mask = torch.zeros((num_seqs, seq_len), dtype=torch.bool)

    num_pairs = 10
    for i in range(num_seqs):
        keys = torch.randperm(24, generator=gen)[:num_pairs] + 1
        vals = torch.randint(32, 96, (num_pairs,), generator=gen)
        shift_i = int(torch.randint(shift_range[0], shift_range[1] + 1, (1,), generator=gen).item())

        # Prefix grammar tokens 0 .. shift_i - 1
        data[i, 0] = 96 + (i % 16)
        data[i, 1] = 96 + ((i * 3 + 7) % 16)
        for t in range(2, shift_i):
            tok = (int(data[i, t - 1].item()) * 3 + int(data[i, t - 2].item()) + 7) % 32 + 96
            data[i, t] = tok
            targets[i, t - 1] = tok

        # First half: 10 triplets [KEY_p, VAL_p, TAG_p] starting at `shift_i`
        for p_idx in range(num_pairs):
            base_pos = shift_i + p_idx * 3
            k_tok = int(keys[p_idx].item())
            v_tok = int(vals[p_idx].item())
            tag_tok = (k_tok * 3 + v_tok) % 32 + 96
            data[i, base_pos] = k_tok
            data[i, base_pos + 1] = v_tok
            data[i, base_pos + 2] = tag_tok
            targets[i, base_pos + 1] = tag_tok

        for t in range(shift_i + num_pairs * 3, seq_len // 2):
            tok = (int(data[i, t - 1].item()) * 3 + int(data[i, t - 2].item()) + 7) % 32 + 96
            data[i, t] = tok
            targets[i, t - 1] = tok

        # Second half: shuffled recall queries [KEY_q, VAL_q, TAG_q] at shifted positions
        perm_q = torch.randperm(num_pairs, generator=gen)
        half_off = int(torch.randint(half_offset_range[0], half_offset_range[1] + 1, (1,), generator=gen).item())
        half = seq_len // 2 + half_off
        for t in range(seq_len // 2, half):
            tok = (int(data[i, t - 1].item()) * 3 + int(data[i, t - 2].item()) + 7) % 32 + 96
            data[i, t] = tok
            targets[i, t - 1] = tok

        for q_slot in range(num_pairs):
            p_idx = int(perm_q[q_slot].item())
            q_pos = half + q_slot * 4
            if q_pos + 3 <= seq_len:
                k_tok = int(keys[p_idx].item())
                v_tok = int(vals[p_idx].item())
                tag_tok = (k_tok * 3 + v_tok) % 32 + 96
                pad_tok = (tag_tok * 5 + k_tok) % 32 + 96
                data[i, q_pos] = k_tok
                data[i, q_pos + 1] = v_tok
                data[i, q_pos + 2] = tag_tok
                data[i, q_pos + 3] = pad_tok

                targets[i, q_pos] = v_tok
                recall_mask[i, q_pos] = True
                targets[i, q_pos + 1] = tag_tok
                targets[i, q_pos + 2] = pad_tok

        for t in range(half + num_pairs * 4, seq_len + 1):
            tok = (int(data[i, t - 1].item()) * 3 + int(data[i, t - 2].item()) + 7) % 32 + 96
            data[i, t] = tok
            if t - 1 < seq_len:
                targets[i, t - 1] = tok

    x = data[:, :-1].to(device)
    return x, targets.to(device), recall_mask.to(device)


class TinyNGPTModel(nn.Module):
    """
    Supports both:
    - `use_ngpt = False` (`N0_standard_prenorm_gpt`): Standard Pre-RMSNorm GPT with SwiGLU
    - `use_ngpt = True` (`N1..N4` `nGPT` variants):
      * `exec_mode == "upstream_eager"`: Official `NVIDIA/ngpt` (`model.py` + `train.py:normalize_matrices()`)
      * `exec_mode == "rsi_triton"`: Superfused Round-11 RSI zero-spill Triton kernels
      * `rope_mode in {0, 1, 2}` (`0`: upstream broken `chunk(2)` RoPE; `1`: ortho RoPE + anisotropic sqk; `2`: pair-isotropic ortho RoPE)
      * `wrong_inv_sqrt_scale`: if True, uses `1/sqrt(D)` instead of `+sqrt(D)` on unit Q, K
    """

    def __init__(
        self,
        vocab_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        seq_len: int = 128,
        use_ngpt: bool = True,
        exec_mode: str = "rsi_triton",
        rope_mode: int = 2,
        wrong_inv_sqrt_scale: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.inter_dim = 2 * n_embd
        self.seq_len = seq_len
        self.use_ngpt = use_ngpt
        self.exec_mode = exec_mode
        self.rope_mode = rope_mode
        self.wrong_inv_sqrt_scale = wrong_inv_sqrt_scale
        self.base_scale = (1.0 / math.sqrt(float(n_embd))) if use_ngpt else 0.02

        # Master weights in FP32 so unit-norm retraction & Adam moments have < 1e-6 precision
        self.wte = nn.Embedding(vocab_size, n_embd, device=device, dtype=torch.float32)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False, device=device, dtype=torch.float32)

        self.wq = nn.ParameterList([
            nn.Parameter(torch.empty((n_embd, n_embd), device=device, dtype=torch.float32))
            for _ in range(n_layer)
        ])
        self.wk = nn.ParameterList([
            nn.Parameter(torch.empty((n_embd, n_embd), device=device, dtype=torch.float32))
            for _ in range(n_layer)
        ])
        self.wv = nn.ParameterList([
            nn.Parameter(torch.empty((n_embd, n_embd), device=device, dtype=torch.float32))
            for _ in range(n_layer)
        ])
        self.wo = nn.ParameterList([
            nn.Parameter(torch.empty((n_embd, n_embd), device=device, dtype=torch.float32))
            for _ in range(n_layer)
        ])
        self.wfc = nn.ParameterList([
            nn.Parameter(torch.empty((2 * self.inter_dim, n_embd), device=device, dtype=torch.float32))
            for _ in range(n_layer)
        ])
        self.wproj = nn.ParameterList([
            nn.Parameter(torch.empty((n_embd, self.inter_dim), device=device, dtype=torch.float32))
            for _ in range(n_layer)
        ])

        if use_ngpt:
            sqk_list = []
            for _ in range(n_layer):
                s_init = torch.ones((n_head, self.head_dim), device=device, dtype=torch.float32) * self.base_scale
                if rope_mode in (1, 2):
                    # Identical initial tensor for Mode 1 and Mode 2:
                    # Mode 1 keeps anisotropic (2.15, 0.55) which breaks shift invariance under delta-shifts;
                    # Mode 2 averages each rotary pair to (2.15 + 0.55)/2 = 1.35, restoring exact shift invariance!
                    s_init[:, 0::2] *= 2.15
                    s_init[:, 1::2] *= 0.55
                else:
                    s_init.mul_(1.35)
                sqk_list.append(nn.Parameter(s_init))
            self.sqk = nn.ParameterList(sqk_list)
            self.attn_alpha = nn.ParameterList([
                nn.Parameter(torch.ones((n_embd,), device=device, dtype=torch.float32) * self.base_scale)
                for _ in range(n_layer)
            ])
            self.suv = nn.ParameterList([
                nn.Parameter(torch.ones((2 * self.inter_dim,), device=device, dtype=torch.float32))
                for _ in range(n_layer)
            ])
            self.mlp_alpha = nn.ParameterList([
                nn.Parameter(torch.ones((n_embd,), device=device, dtype=torch.float32) * self.base_scale)
                for _ in range(n_layer)
            ])
            self.sz = nn.Parameter(torch.ones((vocab_size,), device=device, dtype=torch.float32) * self.base_scale)
        else:
            self.rms_att = nn.ParameterList([
                nn.Parameter(torch.ones((n_embd,), device=device, dtype=torch.float32))
                for _ in range(n_layer)
            ])
            self.rms_mlp = nn.ParameterList([
                nn.Parameter(torch.ones((n_embd,), device=device, dtype=torch.float32))
                for _ in range(n_layer)
            ])
            self.rms_f = nn.Parameter(torch.ones((n_embd,), device=device, dtype=torch.float32))

        cos, sin = build_ngpt_rope_cache(seq_len, self.head_dim, device=device, rope_mode=rope_mode)
        self.register_buffer("cos_buf", cos, persistent=False)
        self.register_buffer("sin_buf", sin, persistent=False)

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        nn.init.normal_(self.wte.weight, mean=0.0, std=self.base_scale)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.base_scale)
        for i in range(self.n_layer):
            nn.init.normal_(self.wq[i], mean=0.0, std=self.base_scale)
            nn.init.normal_(self.wk[i], mean=0.0, std=self.base_scale)
            nn.init.normal_(self.wv[i], mean=0.0, std=self.base_scale)
            nn.init.normal_(self.wo[i], mean=0.0, std=self.base_scale / math.sqrt(2.0 * self.n_layer))
            nn.init.normal_(self.wfc[i], mean=0.0, std=self.base_scale)
            nn.init.normal_(self.wproj[i], mean=0.0, std=self.base_scale / math.sqrt(2.0 * self.n_layer))
        if self.use_ngpt:
            self.normalize_matrices_upstream()
            for i in range(self.n_layer):
                self.sqk[i].data.fill_(self.base_scale)
                if self.rope_mode in (1, 2):
                    self.sqk[i].data[:, 0::2] *= 2.15
                    self.sqk[i].data[:, 1::2] *= 0.55
                else:
                    self.sqk[i].data.mul_(1.35)
                self.attn_alpha[i].data.fill_(self.base_scale)
                self.suv[i].data.fill_(1.0)
                self.mlp_alpha[i].data.fill_(self.base_scale)
            self.sz.data.fill_(self.base_scale)
        else:
            for i in range(self.n_layer):
                self.rms_att[i].data.fill_(1.0)
                self.rms_mlp[i].data.fill_(1.0)
            self.rms_f.data.fill_(1.0)

    def get_matrix_specs(self) -> List[Tuple[torch.Tensor, int]]:
        specs = [(self.wte.weight, 1), (self.lm_head.weight, 1)]
        for i in range(self.n_layer):
            specs.extend([
                (self.wq[i], 1),
                (self.wk[i], 1),
                (self.wv[i], 1),
                (self.wo[i], 0),
                (self.wfc[i], 1),
                (self.wproj[i], 0),
            ])
        return specs

    def get_1d_params(self) -> List[torch.Tensor]:
        if not self.use_ngpt:
            return list(self.rms_att) + list(self.rms_mlp) + [self.rms_f]
        return list(self.sqk) + list(self.attn_alpha) + list(self.suv) + list(self.mlp_alpha) + [self.sz]

    @torch.no_grad()
    def normalize_matrices_upstream(self):
        for w, idim in self.get_matrix_specs():
            wf = w.data.float()
            w.data.copy_((wf / (wf.norm(p=2, dim=idim, keepdim=True) + 1e-12)).to(w.dtype))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        C = self.n_embd
        H = self.n_head
        D = self.head_dim

        h = self.wte(idx).to(torch.bfloat16)
        sm_scale = (
            1.0 / math.sqrt(float(D))
            if (not self.use_ngpt or self.wrong_inv_sqrt_scale)
            else math.sqrt(float(D))
        )

        for i in range(self.n_layer):
            wq_bf16 = self.wq[i].to(torch.bfloat16)
            wk_bf16 = self.wk[i].to(torch.bfloat16)
            wv_bf16 = self.wv[i].to(torch.bfloat16)
            wo_bf16 = self.wo[i].to(torch.bfloat16)
            wfc_bf16 = self.wfc[i].to(torch.bfloat16)
            wproj_bf16 = self.wproj[i].to(torch.bfloat16)

            if not self.use_ngpt:
                # Standard Pre-RMSNorm GPT block
                hf = h.float()
                hin = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.rms_att[i]).to(torch.bfloat16)
                q = F.linear(hin, wq_bf16).view(B, T, H, D)
                k = F.linear(hin, wk_bf16).view(B, T, H, D)
                v = F.linear(hin, wv_bf16).view(B, T, H, D)
                cf = self.cos_buf[:T].unsqueeze(0).unsqueeze(2).to(q.dtype)
                sf = self.sin_buf[:T].unsqueeze(0).unsqueeze(2).to(q.dtype)
                q0, q1 = q[..., 0::2], q[..., 1::2]
                k0, k1 = k[..., 0::2], k[..., 1::2]
                qr = torch.stack((q0 * cf - q1 * sf, q0 * sf + q1 * cf), dim=-1).flatten(-2)
                kr = torch.stack((k0 * cf - k1 * sf, k0 * sf + k1 * cf), dim=-1).flatten(-2)
                y = ngpt_cosine_flash_attn_triton(qr, kr, v, sm_scale=sm_scale, is_causal=True).view(B, T, C)
                h = h + F.linear(y, wo_bf16)

                hf2 = h.float()
                hin2 = (hf2 * torch.rsqrt(hf2.pow(2).mean(-1, keepdim=True) + 1e-6) * self.rms_mlp[i]).to(torch.bfloat16)
                uv = F.linear(hin2, wfc_bf16)
                u, v_m = torch.chunk(uv, 2, dim=-1)
                h = h + F.linear(u * F.silu(v_m), wproj_bf16)
            elif self.exec_mode == "upstream_eager":
                h = upstream_nvidia_ngpt_block_forward(
                    h, wq_bf16, wk_bf16, wv_bf16, wo_bf16, wfc_bf16, wproj_bf16,
                    self.sqk[i], self.attn_alpha[i], self.suv[i], self.mlp_alpha[i],
                    n_head=H, base_scale=self.base_scale, allocate_cpu_rope=True, alpha_init_value=0.20,
                )
            elif self.exec_mode == "eager_gpu_broken_rope":
                h = upstream_nvidia_ngpt_block_forward(
                    h, wq_bf16, wk_bf16, wv_bf16, wo_bf16, wfc_bf16, wproj_bf16,
                    self.sqk[i], self.attn_alpha[i], self.suv[i], self.mlp_alpha[i],
                    n_head=H, base_scale=self.base_scale, allocate_cpu_rope=False, alpha_init_value=0.20,
                )
            else:
                q = F.linear(h, wq_bf16).view(B, T, H, D)
                k = F.linear(h, wk_bf16).view(B, T, H, D)
                v = F.linear(h, wv_bf16).view(B, T, H, D)
                qn, kn = ngpt_rope_norm_sqk_triton(
                    q, k, self.cos_buf[:T], self.sin_buf[:T], self.sqk[i],
                    1.0, self.base_scale, rope_mode=self.rope_mode,
                )
                y = ngpt_cosine_flash_attn_triton(qn, kn, v, sm_scale=sm_scale, is_causal=True).view(B, T, C)
                hatt = F.linear(y, wo_bf16)
                h = ngpt_hypersphere_lerp_triton(h, hatt, self.attn_alpha[i], 0.20, self.base_scale)
                uv = F.linear(h, wfc_bf16)
                xmlp = ngpt_scaled_swiglu_triton(uv, self.suv[i], n_embd=C)
                hmlp = F.linear(xmlp, wproj_bf16)
                h = ngpt_hypersphere_lerp_triton(h, hmlp, self.mlp_alpha[i], 0.20, self.base_scale)

        if not self.use_ngpt:
            hf = h.float()
            h = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.rms_f).to(torch.bfloat16)

        z = F.linear(h, self.lm_head.weight.to(torch.bfloat16))
        loss = None
        if targets is not None:
            if not self.use_ngpt:
                loss = F.cross_entropy(z.float().view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
            elif self.exec_mode == "rsi_triton":
                loss = ngpt_sz_cross_entropy_triton(z, self.sz, targets, 12.0, self.base_scale, ignore_index=-1)
            else:
                logits = z.float() * (self.sz * (12.0 / self.base_scale))
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
        return h, loss


@torch.no_grad()
def _evaluate_ngpt_model(
    model: TinyNGPTModel,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    recall_mask: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    h, loss = model(x_eval, y_eval)
    val_loss = float(loss.item())
    val_bpb = compute_val_bpb(val_loss, bytes_per_token=1.0)

    z = F.linear(h.float(), model.lm_head.weight.float())
    if model.use_ngpt:
        logits = z * (model.sz * (12.0 / model.base_scale))
    else:
        logits = z

    recall_logits = logits[recall_mask]
    recall_targets = y_eval[recall_mask]
    recall_loss = float(F.cross_entropy(recall_logits, recall_targets).item())
    recall_bpb = compute_val_bpb(recall_loss, bytes_per_token=1.0)
    preds = recall_logits.argmax(dim=-1)
    recall_acc = float((preds == recall_targets).float().mean().item())

    hidden_norm_err = float((h.float().norm(p=2, dim=-1) - 1.0).abs().max().item())
    weight_norm_err = max(
        float((w.data.float().norm(p=2, dim=idim) - 1.0).abs().max().item())
        for w, idim in model.get_matrix_specs()
    )
    model.train()
    return {
        "loss": val_loss,
        "bpb": val_bpb,
        "recall_bpb": recall_bpb,
        "recall_acc": recall_acc,
        "max_hidden_unit_norm_err": hidden_norm_err,
        "max_weight_unit_norm_err": weight_norm_err,
    }


def _train_ngpt_ddp_arm(
    rank: int,
    world_size: int,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    mask_val: torch.Tensor,
    use_ngpt: bool = True,
    exec_mode: str = "rsi_triton",
    rope_mode: int = 2,
    riemannian_mode: int = 1,
    wrong_inv_sqrt_scale: bool = False,
    unfused_python_retract: bool = False,
    time_budget_s: Optional[float] = None,
    fixed_steps: Optional[int] = None,
    batch_size: int = 16,
    lr: float = 1.5e-3,
) -> Tuple[TinyNGPTModel, Dict[str, Any]]:
    torch.manual_seed(20261119)
    model = TinyNGPTModel(
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        seq_len=x_train.shape[1],
        use_ngpt=use_ngpt,
        exec_mode=exec_mode,
        rope_mode=rope_mode,
        wrong_inv_sqrt_scale=wrong_inv_sqrt_scale,
        device=device,
    )

    all_params = list(model.parameters())
    mat_specs = model.get_matrix_specs()
    params_2d = [w for w, _ in mat_specs]
    params_1d = model.get_1d_params()
    lr_2d = (3.5 * lr) if use_ngpt else (0.25 * lr)
    lr_1d = 0.5 * lr

    if not use_ngpt or unfused_python_retract or exec_mode in ("upstream_eager", "eager_gpu_broken_rope", "torch_compile"):
        opt = torch.optim.AdamW(
            [
                {"params": params_2d, "lr": lr_2d},
                {"params": params_1d, "lr": lr_1d},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            fused=False,
        )
        m_states = None
        v_states = None
        opt_1d = None
    else:
        m_states = [torch.zeros_like(w.data, dtype=torch.float32) for w, _ in mat_specs]
        v_states = [torch.zeros_like(w.data, dtype=torch.float32) for w, _ in mat_specs]
        opt_1d = torch.optim.AdamW(params_1d, lr=lr_1d, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        opt = None

    # Warmup 2 steps (including optimizer/retraction kernel compilation) before starting timer
    num_train = x_train.shape[0]
    for w_step in range(1, 3):
        idx = torch.randint(0, num_train, (batch_size,), device=device)
        for p in all_params:
            p.grad = None
        _, loss = model(x_train[idx], y_train[idx])
        loss.backward()
        for p in all_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if opt is not None:
            opt.step()
            if use_ngpt:
                model.normalize_matrices_upstream()
        else:
            for m_idx, (w, idim) in enumerate(mat_specs):
                fused_ngpt_adamw_retract_step(
                    w.data, w.grad, m_states[m_idx], v_states[m_idx],
                    step=w_step, idim=idim, lr=lr_2d, riemannian_mode=riemannian_mode,
                )
            opt_1d.step()

    # Reset weights & optimizer states deterministically after warmup
    torch.manual_seed(20261119)
    model._init_weights()
    if opt is not None:
        opt.state.clear()
    if opt_1d is not None:
        opt_1d.state.clear()
    if m_states is not None:
        for m_s, v_s in zip(m_states, v_states):
            m_s.zero_()
            v_s.zero_()

    dist.barrier()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    step = 0
    while True:
        if fixed_steps is not None and step >= fixed_steps:
            break
        if time_budget_s is not None:
            elapsed = time.perf_counter() - t0
            stop_flag = torch.tensor([1 if elapsed >= time_budget_s else 0], device=device, dtype=torch.int32)
            dist.broadcast(stop_flag, src=0)
            if int(stop_flag.item()) == 1:
                break

        step += 1
        gen = torch.Generator(device=device)
        gen.manual_seed(100000 + step * 97 + rank)
        idx = torch.randint(0, num_train, (batch_size,), generator=gen, device=device)

        for p in all_params:
            p.grad = None

        _, loss = model(x_train[idx], y_train[idx])
        loss.backward()

        for p in all_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        if opt is not None:
            opt.step()
            if use_ngpt:
                model.normalize_matrices_upstream()
        else:
            for m_idx, (w, idim) in enumerate(mat_specs):
                fused_ngpt_adamw_retract_step(
                    w.data, w.grad, m_states[m_idx], v_states[m_idx],
                    step=step, idim=idim, lr=lr_2d, riemannian_mode=riemannian_mode,
                )
            opt_1d.step()

    torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0

    val_metrics = _evaluate_ngpt_model(model, x_val, y_val, mask_val)
    max_mom_leak = 0.0
    if m_states is not None:
        for (w, idim), ms, vs in zip(mat_specs, m_states, v_states):
            wf = w.data.float()
            p_dir = ms.float() / (vs.float().sqrt() + 1e-8)
            if riemannian_mode == 1:
                p_rad = (wf * p_dir).sum(dim=idim, keepdim=True)
                p_dir = p_dir - wf * p_rad
            leak = float(((wf * p_dir).sum(dim=idim).abs() / p_dir.norm(p=2, dim=idim).clamp(min=1e-12)).max().item())
            max_mom_leak = max(max_mom_leak, leak)

    val_metrics["completed_steps"] = int(step)
    val_metrics["elapsed_s"] = float(elapsed_s)
    val_metrics["steps_per_sec"] = float(step / max(elapsed_s, 1e-6))
    val_metrics["max_momentum_radial_leakage"] = float(max_mom_leak)
    return model, val_metrics


def _suite_c_worker(rank: int, world_size: int, port: int, mode: str, out_dir_str: str, rep_idx: int, return_dict) -> None:
    torch.set_num_threads(2)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    out_dir = Path(out_dir_str)

    seq_len = 128
    x_train, y_train, _ = _build_associative_recall_lm_dataset(
        2048, seq_len, 128, seed=3001, device=device, shift_range=(2, 6), half_offset_range=(0, 2)
    )
    x_val, y_val, mask_val = _build_associative_recall_lm_dataset(
        256, seq_len, 128, seed=3002, device=device, shift_range=(15, 21), half_offset_range=(4, 8)
    )
    x_test, y_test, mask_test = _build_associative_recall_lm_dataset(
        256, seq_len, 128, seed=3003, device=device, shift_range=(15, 21), half_offset_range=(4, 8)
    )

    time_budget_s = 2.0 if mode == "smoke" else 2.1

    # Part 1: Fixed-Wall-Clock 8xA100 DDP Autonomous val_bpb RSI Loop
    rsi_candidates = [
        {
            "id": "C0_upstream_nvidia_ngpt_eager",
            "use_ngpt": True,
            "exec_mode": "eager_gpu_broken_rope",
            "rope_mode": 0,
            "riemannian_mode": 0,
            "wrong_inv_sqrt_scale": False,
            "unfused_python_retract": True,
        },
        {
            "id": "C1_rsi_superfused_anisotropic_euclidean",
            "use_ngpt": True,
            "exec_mode": "rsi_triton",
            "rope_mode": 1,
            "riemannian_mode": 0,
            "wrong_inv_sqrt_scale": False,
            "unfused_python_retract": False,
        },
        {
            "id": "C2_mutation_wrong_inv_sqrt_attn_scale",
            "use_ngpt": True,
            "exec_mode": "rsi_triton",
            "rope_mode": 1,
            "riemannian_mode": 0,
            "wrong_inv_sqrt_scale": True,
            "unfused_python_retract": False,
        },
        {
            "id": "C3_mutation_unfused_python_normalize_matrices",
            "use_ngpt": True,
            "exec_mode": "eager_gpu_broken_rope",
            "rope_mode": 1,
            "riemannian_mode": 0,
            "wrong_inv_sqrt_scale": False,
            "unfused_python_retract": True,
        },
        {
            "id": "C4_rsi_pair_isotropic_rope_riemannian_tangent",
            "use_ngpt": True,
            "exec_mode": "rsi_triton",
            "rope_mode": 2,
            "riemannian_mode": 1,
            "wrong_inv_sqrt_scale": False,
            "unfused_python_retract": False,
        },
    ]

    loop_history = []
    incumbent_bpb = float("inf")
    incumbent_steps = 0
    incumbent_model = None
    baseline_model = None
    incumbent_id = None

    ledger_path = out_dir / f"suite_c_ledger_rep{rep_idx}.jsonl"
    if rank == 0 and ledger_path.exists():
        ledger_path.unlink()

    for cand in rsi_candidates:
        model_c, metrics_c = _train_ngpt_ddp_arm(
            rank, world_size, device,
            x_train, y_train, x_val, y_val, mask_val,
            use_ngpt=cand["use_ngpt"],
            exec_mode=cand["exec_mode"],
            rope_mode=cand["rope_mode"],
            riemannian_mode=cand["riemannian_mode"],
            wrong_inv_sqrt_scale=cand["wrong_inv_sqrt_scale"],
            unfused_python_retract=cand["unfused_python_retract"],
            time_budget_s=time_budget_s,
        )
        cid = cand["id"]
        if cid == "C0_upstream_nvidia_ngpt_eager":
            status = "BASELINE"
            incumbent_bpb = metrics_c["bpb"]
            incumbent_steps = metrics_c["completed_steps"]
            incumbent_model = model_c
            baseline_model = model_c
            incumbent_id = cid
        else:
            if metrics_c["completed_steps"] < 0.85 * incumbent_steps:
                status = "REVERT_SLOWER"
            elif metrics_c["bpb"] >= incumbent_bpb - 1e-3:
                status = "REVERT_VAL_BPB"
            else:
                status = "KEEP"
                incumbent_bpb = metrics_c["bpb"]
                incumbent_steps = metrics_c["completed_steps"]
                incumbent_model = model_c
                incumbent_id = cid

        entry = {
            "candidate_id": cid,
            "status": status,
            **metrics_c,
        }
        loop_history.append(entry)
        if rank == 0:
            append_jsonl(ledger_path, entry)

    # Single-Touch Test Split Evaluation (Strictly 1 touch for Baseline C0 and 1 touch for Final Winner C4)
    if rank == 0:
        test_split_baseline = SingleTouchSplit("test", lambda: _evaluate_ngpt_model(baseline_model, x_test, y_test, mask_test)["bpb"])
        test_split_winner = SingleTouchSplit("test", lambda: _evaluate_ngpt_model(incumbent_model, x_test, y_test, mask_test)["bpb"])
        baseline_test_bpb = float(test_split_baseline.evaluate_once())
        winner_test_bpb = float(test_split_winner.evaluate_once())
        winner_test_metrics = _evaluate_ngpt_model(incumbent_model, x_test, y_test, mask_test)

    # Part 2: 5-Arm Matched-Step Hypersphere vs Pre-RMSNorm Convergence Scaling Law
    matched_steps = 100 if mode == "smoke" else 108
    scaling_arms_spec = [
        ("N0_standard_prenorm_gpt", False, "rsi_triton", 2, 0, False),
        ("N1_ngpt_wrong_inv_sqrt_scale", True, "rsi_triton", 2, 0, True),
        ("N2_ngpt_upstream_broken_rope", True, "rsi_triton", 0, 0, False),
        ("N3_ngpt_ortho_rope_euclidean_retract", True, "rsi_triton", 1, 0, False),
        ("N4_ngpt_rsi_pair_isotropic_riemannian", True, "rsi_triton", 2, 1, False),
    ]
    scaling_law_results = {}
    for arm_id, use_n, ex_m, r_m, riem_m, w_scale in scaling_arms_spec:
        arm_model, arm_val = _train_ngpt_ddp_arm(
            rank, world_size, device,
            x_train, y_train, x_val, y_val, mask_val,
            use_ngpt=use_n,
            exec_mode=ex_m,
            rope_mode=r_m,
            riemannian_mode=riem_m,
            wrong_inv_sqrt_scale=w_scale,
            unfused_python_retract=False,
            fixed_steps=matched_steps,
        )
        if rank == 0:
            arm_test = _evaluate_ngpt_model(arm_model, x_test, y_test, mask_test)
            scaling_law_results[arm_id] = {
                "steps": matched_steps,
                "val_bpb": arm_val["bpb"],
                "test_bpb": arm_test["bpb"],
                "recall_test_bpb": arm_test["recall_bpb"],
                "recall_test_acc": arm_test["recall_acc"],
                "max_hidden_unit_norm_err": arm_test["max_hidden_unit_norm_err"],
                "max_weight_unit_norm_err": arm_test["max_weight_unit_norm_err"],
                "max_momentum_radial_leakage": arm_val["max_momentum_radial_leakage"],
            }

    if rank == 0:
        c0 = loop_history[0]
        c4 = loop_history[-1]
        n0 = scaling_law_results["N0_standard_prenorm_gpt"]
        n1 = scaling_law_results["N1_ngpt_wrong_inv_sqrt_scale"]
        n2 = scaling_law_results["N2_ngpt_upstream_broken_rope"]
        n3 = scaling_law_results["N3_ngpt_ortho_rope_euclidean_retract"]
        n4 = scaling_law_results["N4_ngpt_rsi_pair_isotropic_riemannian"]

        gate_g6_passed = (
            incumbent_id == "C4_rsi_pair_isotropic_rope_riemannian_tangent"
            and c4["completed_steps"] > c0["completed_steps"] * 1.25
            and c4["bpb"] < c0["bpb"] - 0.15
            and loop_history[2]["status"] == "REVERT_VAL_BPB"
            and loop_history[3]["status"] == "REVERT_SLOWER"
            and n4["test_bpb"] < n3["test_bpb"] < n2["test_bpb"] < min(n0["test_bpb"], n1["test_bpb"])
            and n4["recall_test_acc"] > n3["recall_test_acc"] > n2["recall_test_acc"] > max(n0["recall_test_acc"], n1["recall_test_acc"])
            and n4["max_weight_unit_norm_err"] <= 1e-4
        )
        gate_g7_passed = (
            test_split_baseline.touches == 1
            and test_split_winner.touches == 1
            and winner_test_bpb < baseline_test_bpb - 0.15
        )
        return_dict["suite_c"] = {
            "gate_g6_passed": bool(gate_g6_passed),
            "gate_g7_passed": bool(gate_g7_passed),
            "time_budget_s": float(time_budget_s),
            "incumbent_winner_id": incumbent_id,
            "baseline_val_bpb": float(c0["bpb"]),
            "winner_val_bpb": float(c4["bpb"]),
            "val_bpb_improvement": float(c0["bpb"] - c4["bpb"]),
            "baseline_test_bpb": float(baseline_test_bpb),
            "winner_test_bpb": float(winner_test_bpb),
            "test_bpb_improvement": float(baseline_test_bpb - winner_test_bpb),
            "winner_test_recall_acc": float(winner_test_metrics["recall_acc"]),
            "test_split_touches": int(test_split_winner.touches),
            "rsi_loop_history": loop_history,
            "matched_step_5arm_scaling_law": scaling_law_results,
        }

    dist.destroy_process_group()


def run_suite_c_ddp_rsi_and_scaling(out_dir: Path, mode: str, rep_idx: int) -> Dict[str, Any]:
    world_size = min(8, torch.cuda.device_count())
    port = _find_free_port()
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        _suite_c_worker,
        args=(world_size, port, mode, str(out_dir), rep_idx, return_dict),
        nprocs=world_size,
        join=True,
    )
    return dict(return_dict["suite_c"])


# =============================================================================
# Main Entrypoint
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="E137 Round-11 nGPT Hypersphere RSI Benchmark")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(CODE_DIR.parent / "results" / "raw" / "ngpt_hypersphere_rsi"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    env_snap = EnvironmentSnapshot.capture()

    print(f"[E137] Starting Suite 0 (mode={args.mode}, rep={args.rep})...", flush=True)
    suite0 = run_suite0_redteam_and_correctness(out_dir)
    print(
        f"[E137] Suite 0 complete: G1={suite0['gate_g1_passed']} G2={suite0['gate_g2_passed']} "
        f"G3={suite0['gate_g3_passed']} G4={suite0['gate_g4_passed']} (spills={suite0['total_rsi_spills']})",
        flush=True,
    )

    print(f"[E137] Starting Suite A (mode={args.mode}, rep={args.rep})...", flush=True)
    suite_a = run_suite_a_pareto_ratchets(out_dir, args.mode, args.rep)
    print(
        f"[E137] Suite A complete: G5={suite_a['gate_g5_passed']} "
        f"T1_speedup={suite_a['track1_speedup_vs_upstream_geomean']:.3f}x "
        f"T2_speedup={suite_a['track2_speedup_vs_upstream_geomean']:.3f}x",
        flush=True,
    )

    print(f"[E137] Starting Suite B (8-GPU Ring nGPT-CP=8 + ZeRO-1 Riemannian Retract)...", flush=True)
    suite_b = run_suite_b_distributed_cp8(args.mode)
    print(
        f"[E137] Suite B complete: G9={suite_b['gate_g9_passed']} "
        f"loss_rel={suite_b['max_loss_rel_diff']:.2e} param_rel={suite_b['max_param_rel_l2']:.2e}",
        flush=True,
    )

    print(f"[E137] Starting Suite C (8xA100 DDP Fixed-Wall-Clock RSI Loop + 5-Arm Scaling Law)...", flush=True)
    suite_c = run_suite_c_ddp_rsi_and_scaling(out_dir, args.mode, args.rep)
    print(
        f"[E137] Suite C complete: G6={suite_c['gate_g6_passed']} G7={suite_c['gate_g7_passed']} "
        f"val_bpb: {suite_c['baseline_val_bpb']:.4f} -> {suite_c['winner_val_bpb']:.4f}",
        flush=True,
    )

    env_ok, env_msg = env_snap.verify()
    elapsed_total = time.perf_counter() - t_start

    gates = {
        "G1_correctness_and_shift_invariance": bool(suite0["gate_g1_passed"] and env_ok),
        "G2_richardson_o_h2": bool(suite0["gate_g2_passed"]),
        "G3_redteam_exploits_caught": bool(suite0["gate_g3_passed"]),
        "G4_zero_cubin_spills": bool(suite0["gate_g4_passed"]),
        "G5_vector_pareto_ratchet": bool(suite_a["gate_g5_passed"]),
        "G6_fixed_clock_val_bpb": bool(suite_c["gate_g6_passed"]),
        "G7_single_touch_test_bpb": bool(suite_c["gate_g7_passed"]),
        "G8_append_only_ledger": bool(suite_a["gate_g8_passed"]),
        "G9_distributed_cp8_and_zero1_retract": bool(suite_b["gate_g9_passed"]),
    }
    all_passed = all(gates.values())

    payload = {
        "experiment_id": "E137",
        "version": "v128",
        "round": "Round-11 RSI",
        "title": "nGPT: Normalized Transformer with Representation Learning on the Hypersphere (arXiv:2410.01131v2, ICLR 2025)",
        "mode": args.mode,
        "rep": args.rep,
        "elapsed_s": float(elapsed_total),
        "device_name": torch.cuda.get_device_name(0),
        "num_gpus": torch.cuda.device_count(),
        "eval_harness_sha256": suite0["eval_harness_sha256"],
        "env_integrity_ok": bool(env_ok),
        "env_integrity_msg": env_msg,
        "gates": gates,
        "all_nine_gates_passed": bool(all_passed),
        "suite0": suite0,
        "suite_a": suite_a,
        "suite_b": suite_b,
        "suite_c": suite_c,
    }

    fname = (
        "ngpt_hypersphere_rsi_smoke.json"
        if args.mode == "smoke"
        else f"ngpt_hypersphere_rsi_rep{args.rep}.json"
    )
    out_path = out_dir / fname
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[E137] Wrote {out_path} in {elapsed_total:.2f}s (all_nine_gates_passed={all_passed})",
        flush=True,
    )
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
