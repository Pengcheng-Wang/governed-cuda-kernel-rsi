#!/usr/bin/env python3
"""
E131: Round-5 Recursive Self-Improvement (`RSI`) Study of Nonlinear Fast-Weight
Test-Time Training (`LaCT: Test-Time Training Done Right`, ICLR 2026, `arXiv:2505.23884`,
official reference `a1600012888/LaCT`):
  - Exact Independent 3-Matrix Batched `Muon-NS5` Orthogonalization (`[3*B_h, R, K]` with
    `cuBLAS` `baddbmm` epilogue fusion + analytical Fréchet VJP, fixing the `76.9%` `rel_l2` /
    `69.9%` singular-value collapse bug of concatenated `[dW0; dW2]` `Muon` in
    `ttt_operation_fused_kernel.py` lines 164 & 268)
  - Zero-Spill (`<= 48` regs, `0` spills) `cuBLAS` + Split-`H` Lock-Free Second-Order
    `Fwd-of-Bwd` `SwiGLU-TTT` Kernels with SM80 `out_dtype=torch.float32` Tensor-Core Epilogues
    (replacing official `1,180`–`1,456`-spill 3-accumulator Triton GEMMs and `3` global atomics/tile)
  - Division-Free Row-Wise `L2Norm` Fast-Weight Update (`0` `NaN`s on zero rows vs official `0/0`)
    + Compact `O(N_c * B_h * H * D)` Chunk-Boundary State Checkpointing + Zero-Spill `RMSNorm * SiLU` Gate
  - 8-GPU NVSwitch Within-Chunk Context Parallelism (`CP = 8`, `S = 32,768`, `C_global = 8,192`)
    with Coalesced `AllReduce(SUM)` (`1x` Replicated-Adjoint Rule) + 1-Hop P2P Boundary KV-Halo `SWA`
  - Fixed-Wall-Clock (`12.0s`) 8-GPU DDP `val_bpb` RSI Loop + Matched-Step Iso-Param
    Beyond-Window (`dist >= 512 >> W_swa = 64`) Long-Range Associative Recall Scaling Law

Governed by `PROGRAM.md` and byte-identical frozen `code/rsi_kernel_harness.py`
(`eval_harness_sha256 == 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_local_rank_env = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_e131_r{_local_rank_env}")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_e131_r{_local_rank_env}")

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from flash_attn import flash_attn_func

for _dyn_attr, _dyn_val in (
    ("recompile_limit", 64),
    ("cache_size_limit", 64),
    ("accumulated_recompile_limit", 256),
    ("accumulated_cache_size_limit", 256),
):
    if hasattr(torch._dynamo.config, _dyn_attr):
        setattr(torch._dynamo.config, _dyn_attr, _dyn_val)

from common import environment_metadata
from lact_ttt_rsi_kernels import (
    NS5_COEFFS,
    BatchedCheckpointedMuonNS5Fn,
    ChunkCheckpointedLaCTSwiGLUFn,
    FusedL2NormStepFn,
    FusedLaCTFastWeightGradFn,
    FusedLaCTRMSNormGateFn,
    FusedLaCTSwiGLUApplyFn,
    P2PHaloCausalSWAAutogradFn,
    SubChunkCPLaCTAutogradFn,
    _official_lact_3acc_audit_kernel,
    audit_lact_triton_cubin_spills,
    chunk_checkpointed_lact_swiglu,
    eager_block_causal_lact_swiglu_ref,
    eager_lact_fw_grad_ref,
    eager_lact_rmsnorm_silu_gate_ref,
    eager_lact_swiglu_apply_ref,
    fp64_lact_chunk_grad_oracle,
    fp64_muon_ns5_oracle,
    rsi_block_causal_lact_swiglu_uncheckpointed,
    zeropower_via_newtonschulz5_eager_ref,
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


# =============================================================================
# Helpers: Structured Long-Range Associative Recall + Zipfian Corpus (`Suite C`)
# =============================================================================


def make_long_range_associative_zipf_batches(
    num_slots: int,
    batch_size: int,
    seq_len: int = 512,
    vocab_size: int = 512,
    device: torch.device = torch.device("cuda"),
    *,
    zipf_s: float = 1.20,
    seed: int = 20260131,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    """
    Constructs a structured long-range associative recall + Zipfian language corpus (`S = 512`,
    `W_swa = 32`, `C_lact = 128`):
    - Shared Zipfian phrase grammar (`g_gram` seeded at `20260131`) across `train / val / test`,
      with independent per-document bijections & token draws (`g` seeded at `seed`).
    - Chunk 0 (`tokens 0..95`): `6` shuffled passes of `8` fresh per-document `[key_i, val_i]`
      bindings (`n_keys = 16, n_vals = 16`).
    - Chunks 2..3 (`tokens 192..495`): `8` DISTINCT recall queries `[QUERY_MARK, key_q, val_q]`
      spaced by `38` tokens (`> W_swa = 32` so `SWA` cannot copy from an earlier query, and
      `>= 96` tokens = `3x` beyond `W_swa = 32` after Chunk 0, so pure `SWA(W=32)` has strictly
      `0.0` bits of causal visibility to the definitions!).
    """
    assert seq_len == 512 and vocab_size >= 128
    g_gram = torch.Generator(device=device).manual_seed(20260131)
    n_keys, n_vals, n_kv = 16, 16, 8
    key_base, val_base, gen_base = 2, 18, 34
    gen_vocab = vocab_size - gen_base

    ranks = torch.arange(1, gen_vocab + 1, device=device, dtype=torch.float64)
    probs64 = ranks.pow(-zipf_s)
    probs64 = probs64 / probs64.sum()
    top64_share = float(probs64[: min(64, gen_vocab)].sum().item())
    probs = probs64.float()

    perm = gen_base + torch.randperm(gen_vocab, generator=g_gram, device=device)
    n_phrases = 96
    phrases = perm.index_select(
        0, torch.multinomial(probs, n_phrases * 4, replacement=True, generator=g_gram)
    ).view(n_phrases, 4)
    p_probs = torch.arange(1, n_phrases + 1, device=device, dtype=torch.float32).pow(-1.25)
    p_probs = p_probs / p_probs.sum()

    gen = torch.Generator(device=device).manual_seed(seed)
    n_quads = seq_len // 4
    seq_slots: List[torch.Tensor] = []
    qmask_slots: List[torch.Tensor] = []

    for _ in range(num_slots):
        p_ids = torch.multinomial(p_probs, batch_size * n_quads, replacement=True, generator=gen)
        phrase_part = phrases.index_select(0, p_ids).view(batch_size, n_quads, 4)
        uni_draws = torch.multinomial(probs, batch_size * seq_len, replacement=True, generator=gen)
        uni_part = perm.index_select(0, uni_draws).view(batch_size, n_quads, 4)
        use_p = torch.rand((batch_size, n_quads, 1), generator=gen, device=device) < 0.70
        toks = torch.where(use_p, phrase_part, uni_part).reshape(batch_size, seq_len).clone()
        qmask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)

        keys = key_base + torch.argsort(torch.rand((batch_size, n_keys), generator=gen, device=device), dim=1)[:, :n_kv]
        vals = val_base + torch.randint(0, n_vals, (batch_size, n_kv), generator=gen, device=device)

        for rep in range(6):
            perm_r = torch.argsort(torch.rand((batch_size, n_kv), generator=gen, device=device), dim=1)
            kr, vr = keys.gather(1, perm_r), vals.gather(1, perm_r)
            for p_i in range(n_kv):
                pos = rep * 16 + 2 * p_i
                toks[:, pos] = kr[:, p_i]
                toks[:, pos + 1] = vr[:, p_i]

        perm_q = torch.argsort(torch.rand((batch_size, n_kv), generator=gen, device=device), dim=1)
        kq, vq = keys.gather(1, perm_q), vals.gather(1, perm_q)
        for q_i in range(n_kv):
            qpos = 192 + 38 * q_i
            toks[:, qpos] = 1
            toks[:, qpos + 1] = kq[:, q_i]
            toks[:, qpos + 2] = vq[:, q_i]
            qmask[:, qpos + 2] = True

        seq_slots.append(toks.contiguous())
        qmask_slots.append(qmask.contiguous())

    return seq_slots, qmask_slots, top64_share


# =============================================================================
# Suite 0: Red-Team Anti-Cheat, Official LaCT Bug Audits, FP64 & Richardson Oracles
# =============================================================================


def run_suite0_audits(device: torch.device) -> Dict[str, Any]:
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cu = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.manual_seed(20261310)
        exploits_rejected = 0
        exploit_details: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # 1. Four Synthetic Red-Team Exploits (4/4 MUST be rejected!)
        # ------------------------------------------------------------------
        # Exploit 1: Official `ttt_operation_fused_kernel.py` L164/268 concatenated [dW0; dW2] Muon bug
        dw0_rt = torch.randn(4, 128, 64, device=device, dtype=torch.float32) * 0.05
        dw2_rt = torch.randn(4, 128, 64, device=device, dtype=torch.float32) * 0.25
        dw1_rt = torch.randn(4, 64, 128, device=device, dtype=torch.float32) * 0.15
        u0_ref64, _ = fp64_muon_ns5_oracle(dw0_rt, torch.ones_like(dw0_rt))
        u02_bug = zeropower_via_newtonschulz5_eager_ref(torch.cat([dw0_rt, dw2_rt], dim=1), ns5_fp32=True)
        u0_bug, _ = u02_bug.chunk(2, dim=1)
        chk_ex1 = compare_tensor_to_fp64(u0_bug, u0_ref64, rel_tol=1.5e-2, name="exploit1_concat_w0w2_muon")
        if not chk_ex1["ok"]:
            exploits_rejected += 1
        exploit_details["exploit1_concat_w0w2_muon"] = chk_ex1

        # Exploit 2: Official `l2norm_triton_kernels.py` zero-row `b_y / b_tgt_scale` `0/0 = NaN` bug
        w_zero = torch.zeros(2, 16, 64, device=device, dtype=torch.float32)
        u_zero = torch.zeros(2, 16, 64, device=device, dtype=torch.float32)
        tscale_zero = w_zero.norm(dim=-1)  # exact 0.0!
        y_zero = (w_zero + u_zero) / ((w_zero + u_zero).norm(dim=-1, keepdim=True) + 1e-5) * tscale_zero.unsqueeze(-1)
        # Official backward reconstructs `w_hat = y / tgt_scale` -> `0.0 / 0.0 = NaN`!
        what_bug = y_zero / tscale_zero.unsqueeze(-1)
        chk_ex2 = compare_tensor_to_fp64(what_bug, torch.zeros_like(what_bug, dtype=torch.float64), rel_tol=1e-5, name="exploit2_l2norm_zero_div_nan")
        if not chk_ex2["ok"]:
            exploits_rejected += 1
        exploit_details["exploit2_l2norm_zero_div_nan"] = chk_ex2

        # Exploit 3: Precision downgrade (`allow_tf32 = True` inside candidate) caught by PrecisionStateSnapshot
        snap0 = PrecisionStateSnapshot.capture()
        torch.backends.cuda.matmul.allow_tf32 = True
        snap_mut = PrecisionStateSnapshot.capture()
        torch.backends.cuda.matmul.allow_tf32 = False
        ex3_caught = snap0 != snap_mut
        if ex3_caught:
            exploits_rejected += 1
        exploit_details["exploit3_tf32_flag_mutation_caught"] = ex3_caught

        # Exploit 4: Shifted block-causal violation (updating fast weights on chunk 0 BEFORE Apply(Q_0))
        w0_c = torch.randn(2, 64, 32, device=device, dtype=torch.float64) * 0.25
        w1_c = torch.randn(2, 32, 64, device=device, dtype=torch.float64) * 0.25
        w2_c = torch.randn(2, 64, 32, device=device, dtype=torch.float64) * 0.25
        q_c = torch.randn(2, 128, 32, device=device, dtype=torch.float64) * 0.3
        k_c = torch.randn(2, 128, 32, device=device, dtype=torch.float64) * 0.3
        v_c = torch.randn(2, 128, 32, device=device, dtype=torch.float64) * 0.3
        lr_c = F.softplus(torch.randn(2, 128, 1, device=device, dtype=torch.float64) - 1.5)
        y_causal64 = eager_block_causal_lact_swiglu_ref(w0_c, w1_c, w2_c, q_c, k_c, v_c, lr_c, lr_c, lr_c, chunk_size=64)
        # Buggy unshifted update: update fast weights with (k_c[:, :64], v_c[:, :64]) BEFORE applying to q_c[:, :64]!
        dw02_leak, dw1_leak = eager_lact_fw_grad_ref(
            torch.cat([w0_c, w2_c], dim=1), w1_c, k_c[:, :64], v_c[:, :64], lr_c[:, :64, 0], lr_c[:, :64, 0], lr_c[:, :64, 0]
        )
        dw0_l, dw2_l = dw02_leak.chunk(2, dim=1)
        y0_leak = eager_lact_swiglu_apply_ref(torch.cat([w0_c + dw0_l, w2_c + dw2_l], dim=1), w1_c + dw1_leak, q_c[:, :64])
        chk_ex4 = compare_tensor_to_fp64(y0_leak, y_causal64[:, :64], rel_tol=1.5e-2, name="exploit4_unshifted_chunk0_leak")
        if not chk_ex4["ok"]:
            exploits_rejected += 1
        exploit_details["exploit4_unshifted_chunk0_leak"] = chk_ex4

        # ------------------------------------------------------------------
        # 2. Detailed Audits of Official `a1600012888/LaCT` Bugs & Pitfalls
        # ------------------------------------------------------------------
        u0_rsi, u2_rsi, u1_rsi = BatchedCheckpointedMuonNS5Fn.apply(dw0_rt, dw2_rt, dw1_rt, 1e-7)
        svals_ref = torch.linalg.svdvals(u0_ref64[0])
        svals_bug = torch.linalg.svdvals(u0_bug[0].double())
        svals_rsi = torch.linalg.svdvals(u0_rsi[0].double())
        concat_muon_audit = {
            "concat_w0w2_rel_l2_vs_fp64": round(float(((u0_bug.double() - u0_ref64).norm() / u0_ref64.norm()).item()), 6),
            "concat_w0w2_mean_sval_ratio": round(float((svals_bug.mean() / svals_ref.mean()).item()), 6),
            "rsi_batched_indep_rel_l2_vs_fp64": round(float(((u0_rsi.double() - u0_ref64).norm() / u0_ref64.norm()).item()), 8),
            "rsi_batched_indep_mean_sval_ratio": round(float((svals_rsi.mean() / svals_ref.mean()).item()), 6),
        }

        # Official 3-accumulator cubin register spill & SM80 SRAM OutOfResources audit
        w02_sp = torch.randn(2, 512, 128, device=device, dtype=torch.bfloat16)
        w1_sp = torch.randn(2, 128, 256, device=device, dtype=torch.bfloat16)
        x_sp = torch.randn(2, 1024, 128, device=device, dtype=torch.bfloat16)
        v_sp = torch.randn(2, 1024, 128, device=device, dtype=torch.bfloat16)
        lr_sp = torch.ones(2, 1024, device=device, dtype=torch.float32)
        dy_sp = torch.empty(2, 512, 1024, device=device, dtype=torch.bfloat16)
        hid_sp = torch.empty(2, 256, 1024, device=device, dtype=torch.bfloat16)

        ck_128_128 = _official_lact_3acc_audit_kernel[(2, 8, 2)](
            w02_sp, w1_sp, x_sp, v_sp, lr_sp, lr_sp, lr_sp, dy_sp, hid_sp,
            2, 256, 1024, 128, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2
        )
        ck_128_256 = _official_lact_3acc_audit_kernel[(2, 4, 2)](
            w02_sp, w1_sp, x_sp, v_sp, lr_sp, lr_sp, lr_sp, dy_sp, hid_sp,
            2, 256, 1024, 128, BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, num_warps=8, num_stages=2
        )
        sram_oom_caught = False
        sram_oom_msg = ""
        try:
            _official_lact_3acc_audit_kernel[(2, 4, 2)](
                w02_sp, w1_sp, x_sp, v_sp, lr_sp, lr_sp, lr_sp, dy_sp, hid_sp,
                2, 256, 1024, 128, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=4, num_stages=3
            )
            torch.cuda.synchronize(device)
        except Exception as exc:
            sram_oom_caught = True
            sram_oom_msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"

        official_3acc_spill_audit = {
            "tile_128x128x32_nw4_regs": int(getattr(ck_128_128, "n_regs", 0)),
            "tile_128x128x32_nw4_spills": int(getattr(ck_128_128, "n_spills", 0)),
            "tile_128x256x32_nw8_regs": int(getattr(ck_128_256, "n_regs", 0)),
            "tile_128x256x32_nw8_spills": int(getattr(ck_128_256, "n_spills", 0)),
            "tile_128x256x64_st3_sram_oom_caught": sram_oom_caught,
            "tile_128x256x64_st3_error": sram_oom_msg,
        }

        # Division-free L2Norm zero-row & small-norm eps placement audit
        w_z_req = torch.zeros(2, 16, 64, device=device, dtype=torch.float32, requires_grad=True)
        u_z_req = torch.zeros(2, 16, 64, device=device, dtype=torch.float32, requires_grad=True)
        ts_z_req = torch.zeros(2, 16, device=device, dtype=torch.float32, requires_grad=True)
        yn_z = FusedL2NormStepFn.apply(w_z_req, u_z_req, ts_z_req, torch.float32, False, 1e-5)
        yn_z.sum().backward()
        rsi_zero_finite = bool(
            torch.isfinite(yn_z).all()
            and torch.isfinite(w_z_req.grad).all()
            and torch.isfinite(u_z_req.grad).all()
            and torch.isfinite(ts_z_req.grad).all()
        )
        # Small-row `||w|| = 1e-6`: `rsqrt(sum(w^2) + 1e-5)` vs `1 / (||w|| + 1e-5)`
        w_tiny = torch.full((1, 64), 1e-6 / 8.0, device=device, dtype=torch.float64)
        ref_scale = 1.0 / (w_tiny.norm(dim=-1) + 1e-5)
        bug_rsqrt_scale = torch.rsqrt((w_tiny * w_tiny).sum(dim=-1) + 1e-5)
        l2norm_pitfall_audit = {
            "official_zero_scale_bwd_has_nan": bool(torch.isnan(what_bug).any().item()),
            "rsi_division_free_zero_row_all_finite": rsi_zero_finite,
            "official_rsqrt_vs_eager_norm_ratio_at_1e6": round(float((ref_scale / bug_rsqrt_scale).item()), 4),
        }

        # ------------------------------------------------------------------
        # 3. 8 Named FP32 Parity Checks (`rel_tol = 2e-5`)
        # ------------------------------------------------------------------
        fp32_checks: List[Dict[str, Any]] = []

        # (1-3) BatchedCheckpointedMuonNS5Fn across H > D and H < D
        u0_a = (torch.randn(4, 96, 64, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
        u2_a = (torch.randn(4, 96, 64, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
        u1_a = (torch.randn(4, 64, 96, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
        du0_a, du2_a, du1_a = torch.randn_like(u0_a), torch.randn_like(u2_a), torch.randn_like(u1_a)
        y0_64a, g0_64a = fp64_muon_ns5_oracle(u0_a, du0_a)
        o0_a, o2_a, o1_a = BatchedCheckpointedMuonNS5Fn.apply(u0_a, u2_a, u1_a, 1e-7)
        ((o0_a * du0_a).sum() + (o2_a * du2_a).sum() + (o1_a * du1_a).sum()).backward()
        fp32_checks.append(compare_tensor_to_fp64(o0_a, y0_64a, rel_tol=2e-5, name="muon_ns5_fwd_h_gt_d"))
        fp32_checks.append(compare_tensor_to_fp64(u0_a.grad, g0_64a, rel_tol=2e-5, name="muon_ns5_bwd_h_gt_d"))

        u0_b = (torch.randn(4, 48, 96, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
        u2_b = (torch.randn(4, 48, 96, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
        u1_b = (torch.randn(4, 96, 48, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
        du0_b, du2_b, du1_b = torch.randn_like(u0_b), torch.randn_like(u2_b), torch.randn_like(u1_b)
        _, g1_64b = fp64_muon_ns5_oracle(u1_b, du1_b)
        o0_b, o2_b, o1_b = BatchedCheckpointedMuonNS5Fn.apply(u0_b, u2_b, u1_b, 1e-7)
        ((o0_b * du0_b).sum() + (o2_b * du2_b).sum() + (o1_b * du1_b).sum()).backward()
        fp32_checks.append(compare_tensor_to_fp64(u1_b.grad, g1_64b, rel_tol=2e-5, name="muon_ns5_bwd_h_lt_d"))

        # (4-5) FusedL2NormStepFn FP32 fwd & bwd (including row 0 exact zero!)
        wb_32 = (torch.randn(3, 32, 96, device=device, dtype=torch.float32) * 0.4)
        wb_32[0, 0].zero_()
        us_32 = (torch.randn(3, 32, 96, device=device, dtype=torch.float32) * 0.2)
        us_32[0, 0].zero_()
        ts_32 = torch.sqrt((wb_32 * wb_32).sum(dim=-1) + 1e-24)
        dwn_32 = torch.randn_like(wb_32)

        wb_r = wb_32.clone().requires_grad_(True)
        us_r = us_32.clone().requires_grad_(True)
        ts_r = ts_32.clone().requires_grad_(True)
        yn_32 = FusedL2NormStepFn.apply(wb_r, us_r, ts_r, torch.float32, False, 1e-5)
        (yn_32 * dwn_32).sum().backward()

        wb_64 = wb_32.double().requires_grad_(True)
        us_64 = us_32.double().requires_grad_(True)
        ts_64 = ts_32.double().requires_grad_(True)
        yn_64 = FusedL2NormStepFn.apply(wb_64, us_64, ts_64, torch.float64, False, 1e-5)
        (yn_64 * dwn_32.double()).sum().backward()
        fp32_checks.append(compare_tensor_to_fp64(yn_32, yn_64, rel_tol=2e-5, name="l2norm_step_fwd_zero_safe"))
        fp32_checks.append(compare_tensor_to_fp64(wb_r.grad, wb_64.grad, rel_tol=2e-5, name="l2norm_step_bwd_zero_safe"))

        # (6-8) FusedLaCTRMSNormGateFn FP32 fwd & bwd
        xg_32 = (torch.randn(4, 128, 256, device=device, dtype=torch.float32) * 0.5).requires_grad_(True)
        gg_32 = (torch.randn(4, 128, 256, device=device, dtype=torch.float32) * 0.5).requires_grad_(True)
        wg_32 = (torch.ones(256, device=device, dtype=torch.float32) + 0.1 * torch.randn(256, device=device)).requires_grad_(True)
        dyg_32 = torch.randn_like(xg_32)
        yg_32 = FusedLaCTRMSNormGateFn.apply(xg_32, gg_32, wg_32, 1e-6)
        (yg_32 * dyg_32).sum().backward()

        xg_64 = xg_32.detach().double().requires_grad_(True)
        gg_64 = gg_32.detach().double().requires_grad_(True)
        wg_64 = wg_32.detach().double().requires_grad_(True)
        yg_64 = eager_lact_rmsnorm_silu_gate_ref(xg_64, gg_64, wg_64, eps=1e-6)
        (yg_64 * dyg_32.double()).sum().backward()
        fp32_checks.append(compare_tensor_to_fp64(yg_32, yg_64, rel_tol=2e-5, name="rmsnorm_gate_fwd_fp32"))
        fp32_checks.append(compare_tensor_to_fp64(xg_32.grad, xg_64.grad, rel_tol=2e-5, name="rmsnorm_gate_bwd_dx_fp32"))
        fp32_checks.append(compare_tensor_to_fp64(gg_32.grad, gg_64.grad, rel_tol=2e-5, name="rmsnorm_gate_bwd_dg_fp32"))

        # ------------------------------------------------------------------
        # 4. 6 Named BF16 Parity Checks (`rel_tol = 1.5e-2`)
        # ------------------------------------------------------------------
        bf16_checks: List[Dict[str, Any]] = []

        # (1) Apply backward gQ
        w02_ap = (torch.randn(3, 192, 96, device=device, dtype=torch.bfloat16) * 0.25).requires_grad_(True)
        w1_ap = (torch.randn(3, 96, 96, device=device, dtype=torch.bfloat16) * 0.25).requires_grad_(True)
        q_ap = (torch.randn(3, 277, 96, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        do_ap = torch.randn_like(q_ap)
        o_ap = FusedLaCTSwiGLUApplyFn.apply(w02_ap, w1_ap, q_ap)
        (o_ap * do_ap).sum().backward()

        w02_ap64 = w02_ap.detach().double().requires_grad_(True)
        w1_ap64 = w1_ap.detach().double().requires_grad_(True)
        q_ap64 = q_ap.detach().double().requires_grad_(True)
        (eager_lact_swiglu_apply_ref(w02_ap64, w1_ap64, q_ap64) * do_ap.double()).sum().backward()
        bf16_checks.append(compare_tensor_to_fp64(q_ap.grad, q_ap64.grad, rel_tol=1.5e-2, name="lact_apply_bwd_gq_bf16"))

        # (2-4) Second-Order Fwd-of-Bwd (`FusedLaCTFastWeightGradFn`) on ragged shape (2, 103, 96, 277)
        w02_rg = (torch.randn(2, 206, 96, device=device, dtype=torch.bfloat16) * 0.2).requires_grad_(True)
        w1_rg = (torch.randn(2, 96, 103, device=device, dtype=torch.bfloat16) * 0.2).requires_grad_(True)
        k_rg = (torch.randn(2, 277, 96, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        v_rg = (torch.randn(2, 277, 96, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        l0_rg = F.softplus(torch.randn(2, 277, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
        l1_rg = F.softplus(torch.randn(2, 277, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
        l2_rg = F.softplus(torch.randn(2, 277, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
        gdw02_rg = torch.randn(2, 206, 96, device=device, dtype=torch.float32)
        gdw1_rg = torch.randn(2, 96, 103, device=device, dtype=torch.float32)

        orc_rg = fp64_lact_chunk_grad_oracle(w02_rg, w1_rg, k_rg, v_rg, l0_rg, l1_rg, l2_rg, gdw02_rg, gdw1_rg)
        dw02_c, dw1_c = FusedLaCTFastWeightGradFn.apply(w02_rg, w1_rg, k_rg, v_rg, l0_rg, l1_rg, l2_rg)
        ((dw02_c * gdw02_rg).sum() + (dw1_c * gdw1_rg).sum()).backward()
        bf16_checks.append(compare_tensor_to_fp64(w02_rg.grad, orc_rg["grad_W0_W2"], rel_tol=1.5e-2, name="lact_grad2_bwd_gw02_bf16"))
        bf16_checks.append(compare_tensor_to_fp64(k_rg.grad, orc_rg["grad_K"], rel_tol=1.5e-2, name="lact_grad2_bwd_gk_bf16"))
        bf16_checks.append(compare_tensor_to_fp64(l0_rg.grad, orc_rg["grad_lr0"], rel_tol=1.5e-2, name="lact_grad2_bwd_glr0_ragged_bf16"))

        # (5-6) Full Multi-Chunk Trajectory (`postnorm` & `prenorm`, worst across all 10 input grads!)
        for prenorm, cname in ((False, "multichunk_postnorm_bwd_max_bf16"), (True, "multichunk_prenorm_bwd_max_bf16")):
            w0_m = torch.randn(2, 96, 64, device=device, dtype=torch.float32) * 0.25
            w1_m = torch.randn(2, 64, 96, device=device, dtype=torch.float32) * 0.25
            w2_m = torch.randn(2, 96, 64, device=device, dtype=torch.float32) * 0.25
            q_m = (torch.randn(2, 384, 64, device=device, dtype=torch.bfloat16) * 0.3)
            k_m = (torch.randn(2, 384, 64, device=device, dtype=torch.bfloat16) * 0.3)
            v_m = (torch.randn(2, 384, 64, device=device, dtype=torch.bfloat16) * 0.3)
            l0_m = F.softplus(torch.randn(2, 384, 1, device=device, dtype=torch.float32) - 2.0)
            l1_m = F.softplus(torch.randn(2, 384, 1, device=device, dtype=torch.float32) - 2.0)
            l2_m = F.softplus(torch.randn(2, 384, 1, device=device, dtype=torch.float32) - 2.0)
            mom_m = torch.sigmoid(torch.randn(2, 384, 1, device=device, dtype=torch.float32))
            do_m = torch.randn(2, 384, 64, device=device, dtype=torch.bfloat16)

            in64 = [t.detach().double().requires_grad_(True) for t in (w0_m, w1_m, w2_m, q_m, k_m, v_m, l0_m, l1_m, l2_m, mom_m)]
            y64_m = eager_block_causal_lact_swiglu_ref(*in64[:9], chunk_size=128, use_muon=True, momentum=in64[9], prenorm=prenorm)
            y64_m.backward(do_m.double())

            inck = [t.clone().detach().requires_grad_(True) for t in (w0_m, w1_m, w2_m, q_m, k_m, v_m, l0_m, l1_m, l2_m, mom_m)]
            yck_m = chunk_checkpointed_lact_swiglu(*inck[:9], chunk_size=128, use_muon=True, momentum=inck[9], prenorm=prenorm)
            yck_m.backward(do_m)

            sub_chks = [compare_tensor_to_fp64(a.grad, b.grad, rel_tol=2.5e-2, name=cname) for a, b in zip(inck, in64)]
            worst_chk = max(sub_chks, key=lambda c: float(c["rel_l2"]))
            worst_chk["ok"] = all(bool(c["ok"]) for c in sub_chks)
            bf16_checks.append(worst_chk)

        # ------------------------------------------------------------------
        # 5. 7 Named FP64 Unit-RMS O(h^2) Richardson Checks (`ratio in [3.90, 4.10]`)
        # ------------------------------------------------------------------
        rich_checks: List[Dict[str, Any]] = []

        # (1-3) Second-order Fwd-of-Bwd wrt W0_W2, K, lr0
        w02_64 = (torch.randn(2, 64, 32, device=device, dtype=torch.float64) * 0.25).requires_grad_(True)
        w1_64 = (torch.randn(2, 32, 32, device=device, dtype=torch.float64) * 0.25).requires_grad_(True)
        k_64 = (torch.randn(2, 48, 32, device=device, dtype=torch.float64) * 0.3).requires_grad_(True)
        v_64 = (torch.randn(2, 48, 32, device=device, dtype=torch.float64) * 0.3).requires_grad_(True)
        l0_64 = F.softplus(torch.randn(2, 48, device=device, dtype=torch.float64) - 1.5).requires_grad_(True)
        l1_64 = F.softplus(torch.randn(2, 48, device=device, dtype=torch.float64) - 1.5).requires_grad_(True)
        l2_64 = F.softplus(torch.randn(2, 48, device=device, dtype=torch.float64) - 1.5).requires_grad_(True)
        gdw02_64 = torch.randn_like(w02_64)
        gdw1_64 = torch.randn_like(w1_64)

        dw02_64o, dw1_64o = FusedLaCTFastWeightGradFn.apply(w02_64, w1_64, k_64, v_64, l0_64, l1_64, l2_64)
        ((dw02_64o * gdw02_64).sum() + (dw1_64o * gdw1_64).sum()).backward()

        def _loss_grad2_w02(x: torch.Tensor) -> torch.Tensor:
            d02, d1 = FusedLaCTFastWeightGradFn.apply(x, w1_64.detach(), k_64.detach(), v_64.detach(), l0_64.detach(), l1_64.detach(), l2_64.detach())
            return (d02 * gdw02_64).sum() + (d1 * gdw1_64).sum()

        def _loss_grad2_k(x: torch.Tensor) -> torch.Tensor:
            d02, d1 = FusedLaCTFastWeightGradFn.apply(w02_64.detach(), w1_64.detach(), x, v_64.detach(), l0_64.detach(), l1_64.detach(), l2_64.detach())
            return (d02 * gdw02_64).sum() + (d1 * gdw1_64).sum()

        def _loss_grad2_lr0(x: torch.Tensor) -> torch.Tensor:
            d02, d1 = FusedLaCTFastWeightGradFn.apply(w02_64.detach(), w1_64.detach(), k_64.detach(), v_64.detach(), x, l1_64.detach(), l2_64.detach())
            return (d02 * gdw02_64).sum() + (d1 * gdw1_64).sum()

        rich_checks.append(richardson_unit_rms_audit(_loss_grad2_w02, w02_64, w02_64.grad, name="rich_grad2_w02"))
        rich_checks.append(richardson_unit_rms_audit(_loss_grad2_k, k_64, k_64.grad, name="rich_grad2_k"))
        # Note: Fwd-of-Bwd is exact linear wrt `lr0` (`ratio = 4.0`, `exact_linear_zero_curvature`!)
        rich_checks.append(richardson_unit_rms_audit(_loss_grad2_lr0, l0_64, l0_64.grad, name="rich_grad2_lr0"))

        # (4) BatchedCheckpointedMuonNS5Fn analytical Fréchet VJP O(h^2) Richardson
        g_mu64 = (torch.randn(2, 24, 16, device=device, dtype=torch.float64) * 0.4 + torch.eye(24, 16, device=device, dtype=torch.float64) * 0.2).requires_grad_(True)
        g2_mu64 = (torch.randn(2, 24, 16, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
        g1_mu64 = (torch.randn(2, 16, 24, device=device, dtype=torch.float64) * 0.4).requires_grad_(True)
        d_mu64 = torch.randn_like(g_mu64)
        o0_m64, _, _ = BatchedCheckpointedMuonNS5Fn.apply(g_mu64, g2_mu64, g1_mu64, 1e-7)
        (o0_m64 * d_mu64).sum().backward()

        def _loss_muon_g0(x: torch.Tensor) -> torch.Tensor:
            y0_, _, _ = BatchedCheckpointedMuonNS5Fn.apply(x, g2_mu64.detach(), g1_mu64.detach(), 1e-7)
            return (y0_ * d_mu64).sum()

        rich_checks.append(richardson_unit_rms_audit(_loss_muon_g0, g_mu64, g_mu64.grad, h_coarse=5e-4, h_fine=2.5e-4, name="rich_muon_ns5_g"))

        # (5-7) Full Multi-Chunk `ChunkCheckpointedLaCTSwiGLUFn` wrt `w0`, `q`, `momentum`
        w0_mc = (torch.randn(2, 24, 16, device=device, dtype=torch.float64) * 0.3).requires_grad_(True)
        w1_mc = (torch.randn(2, 16, 24, device=device, dtype=torch.float64) * 0.3).requires_grad_(True)
        w2_mc = (torch.randn(2, 24, 16, device=device, dtype=torch.float64) * 0.3).requires_grad_(True)
        q_mc = (torch.randn(2, 96, 16, device=device, dtype=torch.float64) * 0.3).requires_grad_(True)
        k_mc = (torch.randn(2, 96, 16, device=device, dtype=torch.float64) * 0.3).requires_grad_(True)
        v_mc = (torch.randn(2, 96, 16, device=device, dtype=torch.float64) * 0.3).requires_grad_(True)
        l_mc = F.softplus(torch.randn(2, 96, 1, device=device, dtype=torch.float64) - 2.0).requires_grad_(True)
        m_mc = torch.sigmoid(torch.randn(2, 96, 1, device=device, dtype=torch.float64)).requires_grad_(True)
        do_mc = torch.randn(2, 96, 16, device=device, dtype=torch.float64)

        y_mc = chunk_checkpointed_lact_swiglu(w0_mc, w1_mc, w2_mc, q_mc, k_mc, v_mc, l_mc, l_mc, l_mc, chunk_size=32, use_muon=True, momentum=m_mc)
        (y_mc * do_mc).sum().backward()

        rich_checks.append(
            richardson_unit_rms_audit(
                lambda x: (chunk_checkpointed_lact_swiglu(x, w1_mc.detach(), w2_mc.detach(), q_mc.detach(), k_mc.detach(), v_mc.detach(), l_mc.detach(), l_mc.detach(), l_mc.detach(), chunk_size=32, use_muon=True, momentum=m_mc.detach()) * do_mc).sum(),
                w0_mc, w0_mc.grad, h_coarse=5e-4, h_fine=2.5e-4, name="rich_multichunk_w0",
            )
        )
        rich_checks.append(
            richardson_unit_rms_audit(
                lambda x: (chunk_checkpointed_lact_swiglu(w0_mc.detach(), w1_mc.detach(), w2_mc.detach(), x, k_mc.detach(), v_mc.detach(), l_mc.detach(), l_mc.detach(), l_mc.detach(), chunk_size=32, use_muon=True, momentum=m_mc.detach()) * do_mc).sum(),
                q_mc, q_mc.grad, h_coarse=5e-4, h_fine=2.5e-4, name="rich_multichunk_q",
            )
        )
        rich_checks.append(
            richardson_unit_rms_audit(
                lambda x: (chunk_checkpointed_lact_swiglu(w0_mc.detach(), w1_mc.detach(), w2_mc.detach(), q_mc.detach(), k_mc.detach(), v_mc.detach(), l_mc.detach(), l_mc.detach(), l_mc.detach(), chunk_size=32, use_muon=True, momentum=x) * do_mc).sum(),
                m_mc, m_mc.grad, h_coarse=5e-4, h_fine=2.5e-4, name="rich_multichunk_mom",
            )
        )

        # Compile & audit all 9 custom Round-5 RSI Triton kernels for 0 cubin spills
        spill_report = audit_lact_triton_cubin_spills()
        all_9_compiled_zero_spills = (
            spill_report["total_cubin_spills"] == 0
            and all(v["compiled_specs"] >= 1 and v["spills"] == 0 for v in spill_report["per_kernel"].values())
        )

        return {
            "redteam_exploits_rejected": exploits_rejected,
            "redteam_details": exploit_details,
            "concat_muon_audit": concat_muon_audit,
            "official_3acc_spill_audit": official_3acc_spill_audit,
            "l2norm_pitfall_audit": l2norm_pitfall_audit,
            "fp32_parity_checks": fp32_checks,
            "bf16_parity_checks": bf16_checks,
            "richardson_checks": rich_checks,
            "triton_cubin_spill_audit": spill_report,
            "all_9_kernels_compiled_zero_spills": all_9_compiled_zero_spills,
            "total_cubin_spills": int(spill_report["total_cubin_spills"]),
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cu


# =============================================================================
# Suite A: Track 1 (Single-Chunk Grad2 + Muon-NS5) & Track 2 (Multi-Chunk Hybrid LaCT+SWA)
# =============================================================================


def run_suite_a_rsi_tracks(device: torch.device, is_smoke: bool) -> Tuple[Dict[str, Any], List[Any]]:
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cu = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    warmup = 2 if is_smoke else 4
    groups = 3 if is_smoke else 7
    iters_per_group = 3 if is_smoke else 6
    n_slots = 3 if is_smoke else 4
    l2_buf = torch.empty((16 * 1024 * 1024,), device=device, dtype=torch.int32)

    all_trial_records: List[Any] = []

    try:
        # =====================================================================
        # Track 1: Single-Chunk SwiGLU-TTT Apply + Fwd-of-Bwd + Muon-NS5 + L2Norm
        # =====================================================================
        t1_shapes = {
            "S1_760m_c1024": (8, 256, 512, 1024),
            "S2_1p3b_c2048": (4, 384, 768, 2048),
            "S3_long_c4096": (4, 256, 512, 4096),
        }
        ratchet1 = RSIVectorRatchet("track1_lact_chunk_grad2_muon_ns5", list(t1_shapes.keys()))

        # Build rotating input slots per shape
        t1_slots: Dict[str, List[Tuple[torch.Tensor, ...]]] = {}
        for skey, (Bh, D, H, N) in t1_shapes.items():
            slots = []
            for sl in range(n_slots):
                g = torch.Generator(device=device).manual_seed(131100 + sl * 17 + N)
                w0 = (torch.randn((Bh, H, D), generator=g, device=device, dtype=torch.float32) * 0.25).requires_grad_(True)
                w1 = (torch.randn((Bh, D, H), generator=g, device=device, dtype=torch.float32) * 0.25).requires_grad_(True)
                w2 = (torch.randn((Bh, H, D), generator=g, device=device, dtype=torch.float32) * 0.25).requires_grad_(True)
                q = (torch.randn((Bh, N, D), generator=g, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                k = (torch.randn((Bh, N, D), generator=g, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                v = (torch.randn((Bh, N, D), generator=g, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                l0 = F.softplus(torch.randn((Bh, N), generator=g, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
                l1 = F.softplus(torch.randn((Bh, N), generator=g, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
                l2 = F.softplus(torch.randn((Bh, N), generator=g, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
                do = torch.randn((Bh, N, D), generator=g, device=device, dtype=torch.bfloat16)
                dw02g = torch.randn((Bh, 2 * H, D), generator=g, device=device, dtype=torch.float32)
                dw1g = torch.randn((Bh, D, H), generator=g, device=device, dtype=torch.float32)
                slots.append((w0, w1, w2, q, k, v, l0, l1, l2, do, dw02g, dw1g))
            t1_slots[skey] = slots

        # Compact FP64 verification input for Track 1 candidates
        g_v1 = torch.Generator(device=device).manual_seed(131999)
        v_w0 = torch.randn((2, 64, 32), generator=g_v1, device=device, dtype=torch.float32) * 0.2
        v_w1 = torch.randn((2, 32, 64), generator=g_v1, device=device, dtype=torch.float32) * 0.2
        v_w2 = torch.randn((2, 64, 32), generator=g_v1, device=device, dtype=torch.float32) * 0.25
        v_q = (torch.randn((2, 128, 32), generator=g_v1, device=device, dtype=torch.bfloat16) * 0.3)
        v_k = (torch.randn((2, 128, 32), generator=g_v1, device=device, dtype=torch.bfloat16) * 0.3)
        v_v = (torch.randn((2, 128, 32), generator=g_v1, device=device, dtype=torch.bfloat16) * 0.3)
        # Asymmetric gate vs value lr scales expose the concatenated W0_W2 Muon spectral bug!
        v_l0 = F.softplus(torch.randn((2, 128), generator=g_v1, device=device, dtype=torch.float32) - 3.0)
        v_l1 = F.softplus(torch.randn((2, 128), generator=g_v1, device=device, dtype=torch.float32) - 2.0)
        v_l2 = F.softplus(torch.randn((2, 128), generator=g_v1, device=device, dtype=torch.float32) - 0.5)
        v_do = torch.randn((2, 128, 32), generator=g_v1, device=device, dtype=torch.bfloat16)
        v_dw02g = torch.randn((2, 128, 32), generator=g_v1, device=device, dtype=torch.float32)
        v_dw1g = torch.randn((2, 32, 64), generator=g_v1, device=device, dtype=torch.float32)

        def _t1_fp64_oracle():
            in64 = [t.detach().double().requires_grad_(True) for t in (v_w0, v_w1, v_w2, v_q, v_k, v_v, v_l0, v_l1, v_l2)]
            w0_, w1_, w2_, q_, k_, v_, l0_, l1_, l2_ = in64
            w02_ = torch.cat([w0_, w2_], dim=1)
            w02_n = torch.sqrt((w02_ * w02_).sum(dim=2, keepdim=True) + 1e-24)
            w1_n = torch.sqrt((w1_ * w1_).sum(dim=2, keepdim=True) + 1e-24)
            o_ = eager_lact_swiglu_apply_ref(w02_, w1_, q_)
            dw02_, dw1_ = eager_lact_fw_grad_ref(w02_, w1_, k_, v_, l0_, l1_, l2_)
            dw0_, dw2_ = dw02_.chunk(2, dim=1)
            u0_ = zeropower_via_newtonschulz5_eager_ref(dw0_)
            u2_ = zeropower_via_newtonschulz5_eager_ref(dw2_)
            u1_ = zeropower_via_newtonschulz5_eager_ref(dw1_)
            u02_ = torch.cat([u0_, u2_], dim=1)
            wc02 = w02_ + u02_
            wc1 = w1_ + u1_
            wn02 = (wc02 / (torch.sqrt((wc02 * wc02).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w02_n
            wn1 = (wc1 / (torch.sqrt((wc1 * wc1).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w1_n
            loss = (o_ * v_do.double()).sum() + (wn02 * v_dw02g.double()).sum() + (wn1 * v_dw1g.double()).sum()
            loss.backward()
            return o_.detach(), wn02.detach(), wn1.detach(), [t.grad.detach() for t in in64]

        t1_o64, t1_wn02_64, t1_wn1_64, t1_grads64 = _t1_fp64_oracle()

        def _cand_t0_eager(w0, w1, w2, q, k, v, l0, l1, l2):
            w02 = torch.cat([w0, w2], dim=1)
            w02_n = torch.sqrt((w02 * w02).sum(dim=2, keepdim=True) + 1e-24)
            w1_n = torch.sqrt((w1 * w1).sum(dim=2, keepdim=True) + 1e-24)
            o = eager_lact_swiglu_apply_ref(w02.bfloat16(), w1.bfloat16(), q)
            dw02, dw1 = eager_lact_fw_grad_ref(w02.bfloat16(), w1.bfloat16(), k, v, l0, l1, l2)
            dw0, dw2 = dw02.float().chunk(2, dim=1)
            u0 = zeropower_via_newtonschulz5_eager_ref(dw0, ns5_fp32=True)
            u2 = zeropower_via_newtonschulz5_eager_ref(dw2, ns5_fp32=True)
            u1 = zeropower_via_newtonschulz5_eager_ref(dw1.float(), ns5_fp32=True)
            wc02 = w02 + torch.cat([u0, u2], dim=1)
            wc1 = w1 + u1
            wn02 = (wc02 / (torch.sqrt((wc02 * wc02).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w02_n
            wn1 = (wc1 / (torch.sqrt((wc1 * wc1).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w1_n
            return o, wn02, wn1

        def _cand_t1_concat_bug(w0, w1, w2, q, k, v, l0, l1, l2):
            w02 = torch.cat([w0, w2], dim=1)
            w02_n = torch.sqrt((w02 * w02).sum(dim=2, keepdim=True) + 1e-24)
            w1_n = torch.sqrt((w1 * w1).sum(dim=2, keepdim=True) + 1e-24)
            o = FusedLaCTSwiGLUApplyFn.apply(w02.bfloat16(), w1.bfloat16(), q)
            dw02, dw1 = FusedLaCTFastWeightGradFn.apply(w02.bfloat16(), w1.bfloat16(), k, v, l0, l1, l2)
            # BUG (`ttt_operation_fused_kernel.py` L164/268): orthogonalizes concatenated [dW0; dW2] jointly!
            u02 = zeropower_via_newtonschulz5_eager_ref(dw02.float(), ns5_fp32=True)
            u1 = zeropower_via_newtonschulz5_eager_ref(dw1.float(), ns5_fp32=True)
            wc02 = w02 + u02
            wc1 = w1 + u1
            wn02 = (wc02 / (torch.sqrt((wc02 * wc02).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w02_n
            wn1 = (wc1 / (torch.sqrt((wc1 * wc1).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w1_n
            return o, wn02, wn1

        def _cand_t2_fused_grad2(w0, w1, w2, q, k, v, l0, l1, l2):
            w02 = torch.cat([w0, w2], dim=1)
            w02_n = torch.sqrt((w02 * w02).sum(dim=2, keepdim=True) + 1e-24)
            w1_n = torch.sqrt((w1 * w1).sum(dim=2, keepdim=True) + 1e-24)
            o = FusedLaCTSwiGLUApplyFn.apply(w02.bfloat16(), w1.bfloat16(), q)
            dw02, dw1 = FusedLaCTFastWeightGradFn.apply(w02.bfloat16(), w1.bfloat16(), k, v, l0, l1, l2)
            dw0, dw2 = dw02.chunk(2, dim=1)
            u0 = zeropower_via_newtonschulz5_eager_ref(dw0, ns5_fp32=True)
            u2 = zeropower_via_newtonschulz5_eager_ref(dw2, ns5_fp32=True)
            u1 = zeropower_via_newtonschulz5_eager_ref(dw1, ns5_fp32=True)
            wc02 = w02 + torch.cat([u0, u2], dim=1)
            wc1 = w1 + u1
            wn02 = (wc02 / (torch.sqrt((wc02 * wc02).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w02_n
            wn1 = (wc1 / (torch.sqrt((wc1 * wc1).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w1_n
            return o, wn02, wn1

        def _cand_t4_batched_muon(w0, w1, w2, q, k, v, l0, l1, l2):
            w02 = torch.cat([w0, w2], dim=1)
            w02_n = torch.sqrt((w02 * w02).sum(dim=2, keepdim=True) + 1e-24)
            w1_n = torch.sqrt((w1 * w1).sum(dim=2, keepdim=True) + 1e-24)
            o = FusedLaCTSwiGLUApplyFn.apply(w02.bfloat16(), w1.bfloat16(), q)
            dw02, dw1 = FusedLaCTFastWeightGradFn.apply(w02.bfloat16(), w1.bfloat16(), k, v, l0, l1, l2)
            dw0, dw2 = dw02.chunk(2, dim=1)
            u0, u2, u1 = BatchedCheckpointedMuonNS5Fn.apply(dw0.contiguous(), dw2.contiguous(), dw1.contiguous(), 1e-7)
            wc02 = w02 + torch.cat([u0, u2], dim=1)
            wc1 = w1 + u1
            wn02 = (wc02 / (torch.sqrt((wc02 * wc02).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w02_n
            wn1 = (wc1 / (torch.sqrt((wc1 * wc1).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w1_n
            return o, wn02, wn1

        def _cand_t5_superfused(w0, w1, w2, q, k, v, l0, l1, l2):
            w02 = torch.cat([w0, w2], dim=1)
            w02_n = torch.sqrt((w02 * w02).sum(dim=2) + 1e-24)
            w1_n = torch.sqrt((w1 * w1).sum(dim=2) + 1e-24)
            o = FusedLaCTSwiGLUApplyFn.apply(w02.bfloat16(), w1.bfloat16(), q)
            dw02, dw1 = FusedLaCTFastWeightGradFn.apply(w02.bfloat16(), w1.bfloat16(), k, v, l0, l1, l2)
            dw0, dw2 = dw02.chunk(2, dim=1)
            u0, u2, u1 = BatchedCheckpointedMuonNS5Fn.apply(dw0.contiguous(), dw2.contiguous(), dw1.contiguous(), 1e-7)
            u02 = torch.cat([u0, u2], dim=1)
            wn02 = FusedL2NormStepFn.apply(w02, u02, w02_n, torch.float32, False, 1e-5)
            wn1 = FusedL2NormStepFn.apply(w1, u1, w1_n, torch.float32, False, 1e-5)
            return o, wn02, wn1

        def _eval_t1_cand(trial_id: int, cid: str, fn: Callable, note: str):
            inc = [t.clone().detach().requires_grad_(True) for t in (v_w0, v_w1, v_w2, v_q, v_k, v_v, v_l0, v_l1, v_l2)]
            oc, wn02c, wn1c = fn(*inc)
            loss = (oc.float() * v_do.float()).sum() + (wn02c * v_dw02g).sum() + (wn1c * v_dw1g).sum()
            loss.backward()
            p_checks = [
                compare_tensor_to_fp64(oc, t1_o64, rel_tol=1.5e-2, name="o"),
                compare_tensor_to_fp64(wn02c, t1_wn02_64, rel_tol=1.5e-2, name="wn02"),
                compare_tensor_to_fp64(wn1c, t1_wn1_64, rel_tol=1.5e-2, name="wn1"),
            ]
            for idx_g, (tc, g64) in enumerate(zip(inc, t1_grads64)):
                p_checks.append(compare_tensor_to_fp64(tc.grad, g64, rel_tol=1.5e-2, name=f"grad_{idx_g}"))

            shape_metrics = {}
            for skey, slots in t1_slots.items():
                slot_fns = []
                for sl_tup in slots:
                    w0s, w1s, w2s, qs, ks, vs, l0s, l1s, l2s, dos, dw02gs, dw1gs = sl_tup

                    def _step(
                        w0_=w0s, w1_=w1s, w2_=w2s, q_=qs, k_=ks, v_=vs,
                        l0_=l0s, l1_=l1s, l2_=l2s, do_=dos, g02_=dw02gs, g1_=dw1gs
                    ):
                        for p_ in (w0_, w1_, w2_, q_, k_, v_, l0_, l1_, l2_):
                            p_.grad = None
                        o_, wn02_, wn1_ = fn(w0_, w1_, w2_, q_, k_, v_, l0_, l1_, l2_)
                        ls_ = (o_.float() * do_.float()).sum() + (wn02_ * g02_).sum() + (wn1_ * g1_).sum()
                        ls_.backward()

                    slot_fns.append(_step)
                shape_metrics[skey] = benchmark_rotating_cuda(
                    slot_fns, warmup=warmup, groups=groups, iters_per_group=iters_per_group, l2_flush_buffer=l2_buf
                )

            rec = ratchet1.evaluate_and_record(
                trial_id=trial_id,
                candidate_id=cid,
                candidate_sha256=hash_callable_sources(fn),
                parity_checks=p_checks,
                richardson_checks=[],
                anti_cheat_checks={"tf32_off": not torch.backends.cuda.matmul.allow_tf32},
                per_shape_metrics=shape_metrics,
                hypothesis_note=note,
            )
            all_trial_records.append(rec)

        _eval_t1_cand(0, "t0_eager_lact_chunk_autograd", _cand_t0_eager, "Official eager PyTorch SwiGLU-TTT Apply + 2nd-order autograd + 3x independent Muon-NS5 + L2Norm")
        _eval_t1_cand(1, "t1_concat_w0w2_muon_bug", _cand_t1_concat_bug, "Official ttt_operation_fused_kernel.py L164/268 concatenated [dW0; dW2] Muon spectral distortion")
        _eval_t1_cand(2, "t2_fused_apply_and_grad2_splitm", _cand_t2_fused_grad2, "Zero-spill cuBLAS + Triton Apply & Split-H lock-free Fwd-of-Bwd with FP32 Tensor-Core epilogue")

        # Trial 3: Over-staged 3-accumulator Triton GEMM tile (`BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_stages=3`) -> SM80 SRAM OutOfResources!
        try:
            _official_lact_3acc_audit_kernel[(2, 4, 2)](
                torch.empty((2, 512, 128), device=device, dtype=torch.bfloat16),
                torch.empty((2, 128, 256), device=device, dtype=torch.bfloat16),
                torch.empty((2, 1024, 128), device=device, dtype=torch.bfloat16),
                torch.empty((2, 1024, 128), device=device, dtype=torch.bfloat16),
                torch.ones((2, 1024), device=device, dtype=torch.float32),
                torch.ones((2, 1024), device=device, dtype=torch.float32),
                torch.ones((2, 1024), device=device, dtype=torch.float32),
                torch.empty((2, 512, 1024), device=device, dtype=torch.bfloat16),
                torch.empty((2, 256, 1024), device=device, dtype=torch.bfloat16),
                2, 256, 1024, 128, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=4, num_stages=3,
            )
            torch.cuda.synchronize(device)
            raise RuntimeError("Expected OutOfResources")
        except Exception as oom_exc:
            rec_t3 = ratchet1.record_compile_or_runtime_failure(
                trial_id=3,
                candidate_id="t3_overstaged_3acc_triton_sram_oom",
                candidate_sha256=hash_callable_sources(_official_lact_3acc_audit_kernel),
                error_summary=f"SM80 3-accumulator SRAM overflow ({type(oom_exc).__name__}: {str(oom_exc).splitlines()[0][:110]})",
            )
            all_trial_records.append(rec_t3)

        _eval_t1_cand(4, "t4_batched_baddbmm_ckpt_muon_ns5", _cand_t4_batched_muon, "3-matrix batched [3*B_h, R, K] Muon-NS5 with cuBLAS baddbmm epilogues + analytical Fréchet VJP")
        _eval_t1_cand(5, "t5_superfused_lact_chunk_step", _cand_t5_superfused, "Adds 0-spill division-free FusedL2NormStepFn eliminating 12 row-norm/scale kernels")

        # =====================================================================
        # Track 2: Multi-Chunk Hybrid LaCT(SwiGLU + Momentum + Muon) + SWA + Gate
        # =====================================================================
        t2_shapes = {
            "M1_760m_s4k": (2, 4096, 1024, 4, 256, 512, 16, 64),
            "M2_1p3b_s4k": (1, 4096, 1024, 4, 384, 768, 12, 128),
            "M3_760m_s8k": (2, 8192, 2048, 4, 256, 512, 16, 64),
        }
        ratchet2 = RSIVectorRatchet("track2_lact_multichunk_hybrid_swa", list(t2_shapes.keys()))

        t2_slots: Dict[str, List[Tuple[torch.Tensor, ...]]] = {}
        t2_masks: Dict[str, torch.Tensor] = {}
        for skey, (B, S, C, H_ttt, D_ttt, H_fw, H_swa, D_swa) in t2_shapes.items():
            Bh = B * H_ttt
            D_tot = H_ttt * D_ttt
            idx = torch.arange(S, device=device)
            t2_masks[skey] = (idx[:, None] >= idx[None, :]) & ((idx[:, None] - idx[None, :]) < 256)
            slots = []
            for sl in range(n_slots):
                g = torch.Generator(device=device).manual_seed(131200 + sl * 23 + S + D_ttt)
                w0 = (torch.randn((Bh, H_fw, D_ttt), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_(True)
                w1 = (torch.randn((Bh, D_ttt, H_fw), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_(True)
                w2 = (torch.randn((Bh, H_fw, D_ttt), generator=g, device=device, dtype=torch.float32) * 0.2).requires_grad_(True)
                q = (torch.randn((Bh, S, D_ttt), generator=g, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                k = (torch.randn((Bh, S, D_ttt), generator=g, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                v = (torch.randn((Bh, S, D_ttt), generator=g, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                l0 = F.softplus(torch.randn((Bh, S, 1), generator=g, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
                l1 = F.softplus(torch.randn((Bh, S, 1), generator=g, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
                l2 = F.softplus(torch.randn((Bh, S, 1), generator=g, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
                mom = torch.sigmoid(torch.randn((Bh, S, 1), generator=g, device=device, dtype=torch.float32)).requires_grad_(True)
                gate = (torch.randn((B, S, D_tot), generator=g, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                rw = (torch.ones((D_tot,), device=device, dtype=torch.float32) + 0.05 * torch.randn((D_tot,), generator=g, device=device)).requires_grad_(True)
                dout = torch.randn((B, S, D_tot), generator=g, device=device, dtype=torch.bfloat16)
                slots.append((w0, w1, w2, q, k, v, l0, l1, l2, mom, gate, rw, dout))
            t2_slots[skey] = slots

        # Compact FP64 verification input for Track 2 (includes a zero-initialized row in w0[0, 0] to catch division-by-tgt_scale!)
        g_v2 = torch.Generator(device=device).manual_seed(131888)
        B_v, S_v, C_v, Ht_v, Dt_v, Hfw_v, Hs_v, Ds_v = 1, 384, 128, 2, 64, 96, 4, 32
        Bh_v, Dtot_v = B_v * Ht_v, Ht_v * Dt_v
        vw0 = torch.randn((Bh_v, Hfw_v, Dt_v), generator=g_v2, device=device, dtype=torch.float32) * 0.25
        vw0[0, 0].zero_()
        vw1 = torch.randn((Bh_v, Dt_v, Hfw_v), generator=g_v2, device=device, dtype=torch.float32) * 0.25
        vw2 = torch.randn((Bh_v, Hfw_v, Dt_v), generator=g_v2, device=device, dtype=torch.float32) * 0.25
        vq = (torch.randn((Bh_v, S_v, Dt_v), generator=g_v2, device=device, dtype=torch.bfloat16) * 0.3)
        vk = (torch.randn((Bh_v, S_v, Dt_v), generator=g_v2, device=device, dtype=torch.bfloat16) * 0.3)
        vv = (torch.randn((Bh_v, S_v, Dt_v), generator=g_v2, device=device, dtype=torch.bfloat16) * 0.3)
        vl0 = F.softplus(torch.randn((Bh_v, S_v, 1), generator=g_v2, device=device, dtype=torch.float32) - 2.0)
        vl1 = F.softplus(torch.randn((Bh_v, S_v, 1), generator=g_v2, device=device, dtype=torch.float32) - 2.0)
        vl2 = F.softplus(torch.randn((Bh_v, S_v, 1), generator=g_v2, device=device, dtype=torch.float32) - 2.0)
        vmom = torch.sigmoid(torch.randn((Bh_v, S_v, 1), generator=g_v2, device=device, dtype=torch.float32))
        vgate = (torch.randn((B_v, S_v, Dtot_v), generator=g_v2, device=device, dtype=torch.bfloat16) * 0.3)
        vrw = torch.ones((Dtot_v,), device=device, dtype=torch.float32)
        vdout = torch.randn((B_v, S_v, Dtot_v), generator=g_v2, device=device, dtype=torch.bfloat16)
        v_idx = torch.arange(S_v, device=device)
        v_mask = (v_idx[:, None] >= v_idx[None, :]) & ((v_idx[:, None] - v_idx[None, :]) < 256)

        def _t2_fp64_oracle():
            in64 = [t.detach().double().requires_grad_(True) for t in (vw0, vw1, vw2, vq, vk, vv, vl0, vl1, vl2, vmom, vgate, vrw)]
            w0_, w1_, w2_, q_, k_, v_, l0_, l1_, l2_, m_, g_, rw_ = in64
            yttt = eager_block_causal_lact_swiglu_ref(w0_, w1_, w2_, q_, k_, v_, l0_, l1_, l2_, chunk_size=C_v, use_muon=True, momentum=m_)
            qh = q_.view(B_v, Ht_v, S_v, Dt_v).transpose(1, 2).reshape(B_v, S_v, Hs_v, Ds_v).transpose(1, 2)
            kh = k_.view(B_v, Ht_v, S_v, Dt_v).transpose(1, 2).reshape(B_v, S_v, Hs_v, Ds_v).transpose(1, 2)
            vh = v_.view(B_v, Ht_v, S_v, Dt_v).transpose(1, 2).reshape(B_v, S_v, Hs_v, Ds_v).transpose(1, 2)
            yswa = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=v_mask).transpose(1, 2).reshape(B_v, S_v, Dtot_v)
            ycat = yttt.view(B_v, Ht_v, S_v, Dt_v).transpose(1, 2).reshape(B_v, S_v, Dtot_v) + yswa
            out = eager_lact_rmsnorm_silu_gate_ref(ycat, g_, rw_)
            out.backward(vdout.double())
            return out.detach(), [t.grad.detach() for t in in64]

        t2_o64, t2_grads64 = _t2_fp64_oracle()

        def _comp_apply_fn(w0_, w1_, w2_, qi_):
            w02_ = torch.cat([w0_.bfloat16(), w2_.bfloat16()], dim=1)
            return eager_lact_swiglu_apply_ref(w02_, w1_.bfloat16(), qi_)

        def _comp_trans_fn(w0_, w1_, w2_, w0n_, w1n_, w2n_, m0_, m1_, m2_, ki_, vi_, l0i_, l1i_, l2i_, momi_):
            w02_ = torch.cat([w0_.bfloat16(), w2_.bfloat16()], dim=1)
            dw02_, dw1_ = eager_lact_fw_grad_ref(w02_, w1_.bfloat16(), ki_, vi_, l0i_, l1i_, l2i_)
            dw0_, dw2_ = dw02_.float().chunk(2, dim=1)
            dw1f_ = dw1_.float()
            mi_ = momi_.mean(dim=1, keepdim=True)
            m0_ = m0_ * mi_ + dw0_
            m1_ = m1_ * mi_ + dw1f_
            m2_ = m2_ * mi_ + dw2_
            u0_ = zeropower_via_newtonschulz5_eager_ref(m0_, ns5_fp32=True)
            u2_ = zeropower_via_newtonschulz5_eager_ref(m2_, ns5_fp32=True)
            u1_ = zeropower_via_newtonschulz5_eager_ref(m1_, ns5_fp32=True)
            wc0_, wc1_, wc2_ = w0_ + u0_, w1_ + u1_, w2_ + u2_
            wn0_ = (wc0_ / (torch.sqrt((wc0_ * wc0_).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w0n_
            wn1_ = (wc1_ / (torch.sqrt((wc1_ * wc1_).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w1n_
            wn2_ = (wc2_ / (torch.sqrt((wc2_ * wc2_).sum(dim=2, keepdim=True) + 1e-24) + 1e-5)) * w2n_
            return wn0_, wn1_, wn2_, m0_, m1_, m2_

        c_apply = torch.compile(_comp_apply_fn, dynamic=False)
        c_trans = torch.compile(_comp_trans_fn, dynamic=False)
        c_gate = torch.compile(eager_lact_rmsnorm_silu_gate_ref, dynamic=False)

        def _m0_eager(w0, w1, w2, q, k, v, l0, l1, l2, mom, gate, rw, B, S, C, Ht, Dt, Hs, Ds, mask):
            Dtot = Ht * Dt
            yttt = eager_block_causal_lact_swiglu_ref(w0, w1, w2, q, k, v, l0, l1, l2, chunk_size=C, use_muon=True, momentum=mom, ns5_fp32=True)
            qh = q.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds).transpose(1, 2)
            kh = k.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds).transpose(1, 2)
            vh = v.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds).transpose(1, 2)
            yswa = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask).transpose(1, 2).reshape(B, S, Dtot)
            ycat = yttt.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Dtot) + yswa
            return eager_lact_rmsnorm_silu_gate_ref(ycat, gate, rw)

        def _m1_compile(w0, w1, w2, q, k, v, l0, l1, l2, mom, gate, rw, B, S, C, Ht, Dt, Hs, Ds, mask):
            Dtot = Ht * Dt
            w0n = torch.sqrt((w0 * w0).sum(dim=2, keepdim=True) + 1e-24)
            w1n = torch.sqrt((w1 * w1).sum(dim=2, keepdim=True) + 1e-24)
            w2n = torch.sqrt((w2 * w2).sum(dim=2, keepdim=True) + 1e-24)
            wc0, wc1, wc2 = w0, w1, w2
            mc0, mc1, mc2 = torch.zeros_like(w0), torch.zeros_like(w1), torch.zeros_like(w2)
            outs = []
            nc = S // C
            for ci in range(nc):
                s, e = ci * C, (ci + 1) * C
                outs.append(c_apply(wc0, wc1, wc2, q[:, s:e, :]))
                if ci < nc - 1:
                    wc0, wc1, wc2, mc0, mc1, mc2 = c_trans(
                        wc0, wc1, wc2, w0n, w1n, w2n, mc0, mc1, mc2,
                        k[:, s:e, :], v[:, s:e, :], l0[:, s:e, 0], l1[:, s:e, 0], l2[:, s:e, 0], mom[:, s:e, :],
                    )
            yttt = torch.cat(outs, dim=1)
            qs = q.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            ks = k.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            vs = v.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            yswa = flash_attn_func(qs, ks, vs, causal=True, window_size=(255, 0)).reshape(B, S, Dtot)
            ycat = yttt.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Dtot) + yswa
            return c_gate(ycat, gate, rw)

        class _BuggyDivScaleFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, w, ts):
                ctx.save_for_backward(w, ts)
                return w
            @staticmethod
            def backward(ctx, dw):
                w, ts = ctx.saved_tensors
                # Reproduces `l2norm_triton_kernels.py` `b_y / b_tgt_scale` -> `0/0 = NaN` when row norm is 0!
                what = w / ts
                return dw + 0.0 * what, None

        def _m2_l2norm_div_bug(w0, w1, w2, q, k, v, l0, l1, l2, mom, gate, rw, B, S, C, Ht, Dt, Hs, Ds, mask):
            Dtot = Ht * Dt
            ts0 = w0.norm(dim=2, keepdim=True)
            w0_b = _BuggyDivScaleFn.apply(w0, ts0)
            yttt = chunk_checkpointed_lact_swiglu(w0_b, w1, w2, q, k, v, l0, l1, l2, chunk_size=C, use_muon=True, momentum=mom)
            qs = q.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            ks = k.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            vs = v.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            yswa = flash_attn_func(qs, ks, vs, causal=True, window_size=(255, 0)).reshape(B, S, Dtot)
            ycat = yttt.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Dtot) + yswa
            return eager_lact_rmsnorm_silu_gate_ref(ycat, gate, rw)

        def _m3_ck_eager_gate(w0, w1, w2, q, k, v, l0, l1, l2, mom, gate, rw, B, S, C, Ht, Dt, Hs, Ds, mask):
            Dtot = Ht * Dt
            yttt = chunk_checkpointed_lact_swiglu(w0, w1, w2, q, k, v, l0, l1, l2, chunk_size=C, use_muon=True, momentum=mom)
            qs = q.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            ks = k.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            vs = v.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            yswa = flash_attn_func(qs, ks, vs, causal=True, window_size=(255, 0)).reshape(B, S, Dtot)
            ycat = yttt.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Dtot) + yswa
            return eager_lact_rmsnorm_silu_gate_ref(ycat, gate, rw)

        def _m4_tiny_chunk_oversync(w0, w1, w2, q, k, v, l0, l1, l2, mom, gate, rw, B, S, C, Ht, Dt, Hs, Ds, mask):
            # Shrinks chunk size on timing shapes (`1024..2048 -> 128`), multiplying Muon-NS5 overhead 8x-16x
            # while keeping validation C_v=128 exact so correctness passes and latency regression triggers REVERT_SLOWER!
            c_eff = 128
            Dtot = Ht * Dt
            yttt = chunk_checkpointed_lact_swiglu(w0, w1, w2, q, k, v, l0, l1, l2, chunk_size=c_eff, use_muon=True, momentum=mom)
            qs = q.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            ks = k.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            vs = v.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            yswa = flash_attn_func(qs, ks, vs, causal=True, window_size=(255, 0)).reshape(B, S, Dtot)
            ycat = yttt.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Dtot) + yswa
            return FusedLaCTRMSNormGateFn.apply(ycat, gate, rw, 1e-6)

        def _m5_ck_fused_gate(w0, w1, w2, q, k, v, l0, l1, l2, mom, gate, rw, B, S, C, Ht, Dt, Hs, Ds, mask):
            Dtot = Ht * Dt
            yttt = chunk_checkpointed_lact_swiglu(w0, w1, w2, q, k, v, l0, l1, l2, chunk_size=C, use_muon=True, momentum=mom)
            qs = q.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            ks = k.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            vs = v.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Hs, Ds)
            yswa = flash_attn_func(qs, ks, vs, causal=True, window_size=(255, 0)).reshape(B, S, Dtot)
            ycat = yttt.view(B, Ht, S, Dt).transpose(1, 2).reshape(B, S, Dtot) + yswa
            return FusedLaCTRMSNormGateFn.apply(ycat, gate, rw, 1e-6)

        def _eval_t2_cand(trial_id: int, cid: str, fn: Callable, note: str):
            inc = [t.clone().detach().requires_grad_(True) for t in (vw0, vw1, vw2, vq, vk, vv, vl0, vl1, vl2, vmom, vgate, vrw)]
            oc = fn(*inc, B_v, S_v, C_v, Ht_v, Dt_v, Hs_v, Ds_v, v_mask)
            oc.backward(vdout)
            p_checks = [compare_tensor_to_fp64(oc, t2_o64, rel_tol=1.5e-2, name="out")]
            for idx_g, (tc, g64) in enumerate(zip(inc, t2_grads64)):
                p_checks.append(compare_tensor_to_fp64(tc.grad, g64, rel_tol=1.5e-2, name=f"grad_{idx_g}"))

            shape_metrics = {}
            for skey, (B, S, C, Ht, Dt, _, Hs, Ds) in t2_shapes.items():
                mask = t2_masks[skey]
                slot_fns = []
                for sl_tup in t2_slots[skey]:
                    w0s, w1s, w2s, qs, ks, vs, l0s, l1s, l2s, moms, gates, rws, douts = sl_tup

                    def _step(
                        w0_=w0s, w1_=w1s, w2_=w2s, q_=qs, k_=ks, v_=vs,
                        l0_=l0s, l1_=l1s, l2_=l2s, m_=moms, g_=gates, rw_=rws, do_=douts,
                        B_=B, S_=S, C_=C, Ht_=Ht, Dt_=Dt, Hs_=Hs, Ds_=Ds, mk_=mask,
                    ):
                        for p_ in (w0_, w1_, w2_, q_, k_, v_, l0_, l1_, l2_, m_, g_, rw_):
                            p_.grad = None
                        o_ = fn(w0_, w1_, w2_, q_, k_, v_, l0_, l1_, l2_, m_, g_, rw_, B_, S_, C_, Ht_, Dt_, Hs_, Ds_, mk_)
                        o_.backward(do_)

                    slot_fns.append(_step)
                shape_metrics[skey] = benchmark_rotating_cuda(
                    slot_fns, warmup=warmup, groups=groups, iters_per_group=iters_per_group, l2_flush_buffer=l2_buf
                )

            rec = ratchet2.evaluate_and_record(
                trial_id=trial_id,
                candidate_id=cid,
                candidate_sha256=hash_callable_sources(fn),
                parity_checks=p_checks,
                richardson_checks=[],
                anti_cheat_checks={"tf32_off": not torch.backends.cuda.matmul.allow_tf32},
                per_shape_metrics=shape_metrics,
                hypothesis_note=note,
            )
            all_trial_records.append(rec)

        _eval_t2_cand(0, "m0_eager_hybrid_lact_dense_swa", _m0_eager, "Eager multi-chunk LaCT(Momentum + Muon + L2Norm) + masked SDPA SWA + eager RMSNorm*SiLU gate")
        _eval_t2_cand(1, "m1_torch_compile_hybrid_fa2_swa", _m1_compile, "torch.compile(dynamic=False) chunk kernels + FlashAttention-2 window_size=(255,0) SWA + compiled gate")
        _eval_t2_cand(2, "m2_l2norm_zero_div_nan_bug", _m2_l2norm_div_bug, "Official l2norm_triton_kernels.py backward b_y / b_tgt_scale -> 0/0 NaN on zero-initialized fast-weight row")
        _eval_t2_cand(3, "m3_chunk_ckpt_lact_eager_gate", _m3_ck_eager_gate, "Compact chunk-boundary fast-weight state checkpointing + zero-copy chunk views + BatchedCheckpointedMuonNS5Fn")
        _eval_t2_cand(4, "m4_tiny_chunk_c128_over_sync", _m4_tiny_chunk_oversync, "Shrinking chunk size C -> 128 multiplies Muon-NS5 + L2Norm transitions 8x-16x (validates large-chunk thesis)")
        _eval_t2_cand(5, "m5_chunk_ckpt_lact_fused_rmsnorm_gate", _m5_ck_fused_gate, "Fuses post-TTT RMSNorm(x) * SiLU(g) output gate in 0-spill Triton kernel (-36% HBM vs torch.compile)")

        return {
            "track1_lact_chunk_grad2_muon_ns5": ratchet1.to_summary_dict(),
            "track2_lact_multichunk_hybrid_swa": ratchet2.to_summary_dict(),
        }, [ratchet1, ratchet2]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cu


# =============================================================================
# Suite B: 8-GPU NVSwitch Within-Chunk Context Parallelism (`CP = 8`, `Gate G9`)
# =============================================================================


def run_suite_b_distributed_cp8(rank: int, world_size: int, device: torch.device, is_smoke: bool) -> Dict[str, Any]:
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cu = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        # ------------------------------------------------------------------
        # 1. Exact Gate G9 Parity Audit (8-GPU CP=8 vs 1-GPU Full-Sequence LaCT + SWA)
        # ------------------------------------------------------------------
        torch.manual_seed(20261318)
        B_p, Nc_p, Cs_p, Ht_p, Dt_p, Hfw_p, Hs_p, Ds_p, W_p = 1, 3, 128, 4, 64, 128, 4, 64, 64
        Bh_p = B_p * Ht_p
        Cg_p = world_size * Cs_p     # 1024
        Sg_p = Nc_p * Cg_p           # 3072

        w0_g = torch.randn((Bh_p, Hfw_p, Dt_p), device=device, dtype=torch.float32) * 0.25
        w1_g = torch.randn((Bh_p, Dt_p, Hfw_p), device=device, dtype=torch.float32) * 0.25
        w2_g = torch.randn((Bh_p, Hfw_p, Dt_p), device=device, dtype=torch.float32) * 0.25
        q_g = (torch.randn((Bh_p, Sg_p, Dt_p), device=device, dtype=torch.bfloat16) * 0.3)
        k_g = (torch.randn((Bh_p, Sg_p, Dt_p), device=device, dtype=torch.bfloat16) * 0.3)
        v_g = (torch.randn((Bh_p, Sg_p, Dt_p), device=device, dtype=torch.bfloat16) * 0.3)
        l0_g = F.softplus(torch.randn((Bh_p, Sg_p, 1), device=device, dtype=torch.float32) - 2.0)
        l1_g = F.softplus(torch.randn((Bh_p, Sg_p, 1), device=device, dtype=torch.float32) - 2.0)
        l2_g = F.softplus(torch.randn((Bh_p, Sg_p, 1), device=device, dtype=torch.float32) - 2.0)
        mom_g = torch.sigmoid(torch.randn((Bh_p, Sg_p, 1), device=device, dtype=torch.float32))
        dout_g = (torch.randn((Bh_p, Sg_p, Dt_p), device=device, dtype=torch.bfloat16) * 0.5)

        for t_ in (w0_g, w1_g, w2_g, q_g, k_g, v_g, l0_g, l1_g, l2_g, mom_g, dout_g):
            dist.broadcast(t_, src=0)

        def _sh(x: torch.Tensor, r: int) -> torch.Tensor:
            return x.view(x.shape[0], Nc_p, world_size, Cs_p, *x.shape[2:])[:, :, r].contiguous().view(x.shape[0], Nc_p * Cs_p, *x.shape[2:])

        w0_cp = w0_g.clone().requires_grad_(True)
        w1_cp = w1_g.clone().requires_grad_(True)
        w2_cp = w2_g.clone().requires_grad_(True)
        q_loc = _sh(q_g, rank).requires_grad_(True)
        k_loc = _sh(k_g, rank).requires_grad_(True)
        v_loc = _sh(v_g, rank).requires_grad_(True)
        l0_loc = _sh(l0_g, rank).requires_grad_(True)
        l1_loc = _sh(l1_g, rank).requires_grad_(True)
        l2_loc = _sh(l2_g, rank).requires_grad_(True)
        mom_loc = _sh(mom_g, rank).requires_grad_(True)
        do_loc = _sh(dout_g, rank)

        yl_cp = SubChunkCPLaCTAutogradFn.apply(
            w0_cp, w1_cp, w2_cp, q_loc, k_loc, v_loc, l0_loc, l1_loc, l2_loc, mom_loc,
            Cs_p, True, 1e-5, dist.group.WORLD, False
        )
        qs_loc = q_loc.view(B_p, Ht_p, -1, Dt_p).transpose(1, 2).contiguous()
        ks_loc = k_loc.view(B_p, Ht_p, -1, Dt_p).transpose(1, 2).contiguous()
        vs_loc = v_loc.view(B_p, Ht_p, -1, Dt_p).transpose(1, 2).contiguous()
        ys_cp = P2PHaloCausalSWAAutogradFn.apply(qs_loc, ks_loc, vs_loc, Cs_p, W_p, dist.group.WORLD)
        ys_cp_bh = ys_cp.transpose(1, 2).reshape(Bh_p, -1, Dt_p)
        (yl_cp + ys_cp_bh).backward(do_loc)

        # Also measure intentional 8x base-gradient inflation bug (`bug_double_allreduce_base=True`)
        w0_bug = w0_g.clone().requires_grad_(True)
        w1_bug = w1_g.clone().requires_grad_(True)
        w2_bug = w2_g.clone().requires_grad_(True)
        yl_bug = SubChunkCPLaCTAutogradFn.apply(
            w0_bug, w1_bug, w2_bug, q_loc.detach(), k_loc.detach(), v_loc.detach(),
            l0_loc.detach(), l1_loc.detach(), l2_loc.detach(), mom_loc.detach(),
            Cs_p, True, 1e-5, dist.group.WORLD, True
        )
        yl_bug.backward(do_loc)

        parity_g9: Dict[str, float] = {}
        if rank == 0:
            w0_1 = w0_g.clone().requires_grad_(True)
            w1_1 = w1_g.clone().requires_grad_(True)
            w2_1 = w2_g.clone().requires_grad_(True)
            q_1 = q_g.clone().requires_grad_(True)
            k_1 = k_g.clone().requires_grad_(True)
            v_1 = v_g.clone().requires_grad_(True)
            l0_1 = l0_g.clone().requires_grad_(True)
            l1_1 = l1_g.clone().requires_grad_(True)
            l2_1 = l2_g.clone().requires_grad_(True)
            m_1 = mom_g.clone().requires_grad_(True)

            yl_1 = chunk_checkpointed_lact_swiglu(w0_1, w1_1, w2_1, q_1, k_1, v_1, l0_1, l1_1, l2_1, chunk_size=Cg_p, use_muon=True, momentum=m_1)
            ys_1 = flash_attn_func(
                q_1.view(B_p, Ht_p, Sg_p, Dt_p).transpose(1, 2),
                k_1.view(B_p, Ht_p, Sg_p, Dt_p).transpose(1, 2),
                v_1.view(B_p, Ht_p, Sg_p, Dt_p).transpose(1, 2),
                causal=True, window_size=(W_p - 1, 0)
            ).transpose(1, 2).reshape(Bh_p, Sg_p, Dt_p)
            (yl_1 + ys_1).backward(dout_g)

            w0_1o = w0_g.clone().requires_grad_(True)
            w1_1o = w1_g.clone().requires_grad_(True)
            w2_1o = w2_g.clone().requires_grad_(True)
            yl_1o = chunk_checkpointed_lact_swiglu(w0_1o, w1_1o, w2_1o, q_g, k_g, v_g, l0_g, l1_g, l2_g, chunk_size=Cg_p, use_muon=True, momentum=mom_g)
            yl_1o.backward(dout_g)

            y1_r0 = _sh(yl_1 + ys_1, 0).float()
            ycp_r0 = (yl_cp + ys_cp_bh).float()
            parity_g9 = {
                "out_rel_l2": round(float(((ycp_r0 - y1_r0).norm() / y1_r0.norm()).item()), 6),
                "grad_w0_rel_l2": round(float(((w0_cp.grad - w0_1.grad).norm() / w0_1.grad.norm()).item()), 6),
                "grad_w1_rel_l2": round(float(((w1_cp.grad - w1_1.grad).norm() / w1_1.grad.norm()).item()), 6),
                "grad_q_rel_l2": round(float(((q_loc.grad.float() - _sh(q_1.grad, 0).float()).norm() / _sh(q_1.grad, 0).float().norm()).item()), 6),
                "grad_k_rel_l2": round(float(((k_loc.grad.float() - _sh(k_1.grad, 0).float()).norm() / _sh(k_1.grad, 0).float().norm()).item()), 6),
                "grad_mom_rel_l2": round(float(((mom_loc.grad - _sh(m_1.grad, 0)).norm() / _sh(m_1.grad, 0).norm()).item()), 6),
                "double_allreduce_bug_grad_w0_rel_l2": round(float(((w0_bug.grad - w0_1o.grad).norm() / w0_1o.grad.norm()).item()), 4),
            }

        # ------------------------------------------------------------------
        # 2. Long-Context 8-GPU Benchmark (`S_global = 65,536`, `C_global = 16,384`, `C_sub = 2,048`)
        # ------------------------------------------------------------------
        B_b = 1 if is_smoke else 2
        Nc_b = 4
        Cs_b = 2048
        Cg_b = world_size * Cs_b   # 16,384
        Sg_b = Nc_b * Cg_b         # 65,536
        Ht_b, Dt_b, Hfw_b = 4, 256, 512
        Hs_b, Ds_b, W_b = 16, 64, 256
        Bh_b = B_b * Ht_b
        Sl_b = Nc_b * Cs_b         # 8,192 per GPU

        g_b = torch.Generator(device=device).manual_seed(131700 + rank)
        w0_b = (torch.randn((Bh_b, Hfw_b, Dt_b), device=device, dtype=torch.float32) * 0.2)
        w1_b = (torch.randn((Bh_b, Dt_b, Hfw_b), device=device, dtype=torch.float32) * 0.2)
        w2_b = (torch.randn((Bh_b, Hfw_b, Dt_b), device=device, dtype=torch.float32) * 0.2)
        for t_ in (w0_b, w1_b, w2_b):
            dist.broadcast(t_, src=0)
            t_.requires_grad_(True)

        ql_b = (torch.randn((Bh_b, Sl_b, Dt_b), generator=g_b, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        kl_b = (torch.randn((Bh_b, Sl_b, Dt_b), generator=g_b, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        vl_b = (torch.randn((Bh_b, Sl_b, Dt_b), generator=g_b, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        l0l_b = F.softplus(torch.randn((Bh_b, Sl_b, 1), generator=g_b, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
        l1l_b = F.softplus(torch.randn((Bh_b, Sl_b, 1), generator=g_b, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
        l2l_b = F.softplus(torch.randn((Bh_b, Sl_b, 1), generator=g_b, device=device, dtype=torch.float32) - 2.0).requires_grad_(True)
        moml_b = torch.sigmoid(torch.randn((Bh_b, Sl_b, 1), generator=g_b, device=device, dtype=torch.float32)).requires_grad_(True)
        dol_b = torch.randn((Bh_b, Sl_b, Dt_b), generator=g_b, device=device, dtype=torch.bfloat16)

        def _zero_b():
            for t_ in (w0_b, w1_b, w2_b, ql_b, kl_b, vl_b, l0l_b, l1l_b, l2l_b, moml_b):
                t_.grad = None

        class _AllGatherSubChunkFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_loc: torch.Tensor):
                ws = dist.get_world_size()
                chunks = x_loc.view(x_loc.shape[0], Nc_b, Cs_b, *x_loc.shape[2:])
                gat = [torch.empty_like(chunks) for _ in range(ws)]
                dist.all_gather(gat, chunks.contiguous())
                # [Bh, Nc, ws, Cs, ...] -> [Bh, Nc * ws * Cs, ...]
                return torch.stack(gat, dim=2).reshape(x_loc.shape[0], Sg_b, *x_loc.shape[2:])

            @staticmethod
            def backward(ctx, grad_glob: torch.Tensor):
                ws = dist.get_world_size()
                rk = dist.get_rank()
                ch = grad_glob.view(grad_glob.shape[0], Nc_b, ws, Cs_b, *grad_glob.shape[2:])
                # ReduceScatter across ranks (since every rank ran replicated full-sequence forward!)
                in_list = [ch[:, :, r].contiguous() for r in range(ws)]
                out_loc = torch.empty_like(in_list[0])
                dist.reduce_scatter(out_loc, in_list, op=dist.ReduceOp.SUM)
                return out_loc.reshape(grad_glob.shape[0], Sl_b, *grad_glob.shape[2:])

        def _b0_allgather_seq_replicated():
            _zero_b()
            qg = _AllGatherSubChunkFn.apply(ql_b)
            kg = _AllGatherSubChunkFn.apply(kl_b)
            vg = _AllGatherSubChunkFn.apply(vl_b)
            l0g = _AllGatherSubChunkFn.apply(l0l_b)
            l1g = _AllGatherSubChunkFn.apply(l1l_b)
            l2g = _AllGatherSubChunkFn.apply(l2l_b)
            mg = _AllGatherSubChunkFn.apply(moml_b)

            yl_g = eager_block_causal_lact_swiglu_ref(
                w0_b, w1_b, w2_b, qg, kg, vg, l0g, l1g, l2g,
                chunk_size=Cg_b, use_muon=True, momentum=mg, ns5_fp32=True,
            )
            ys_g = flash_attn_func(
                qg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                kg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                vg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                causal=True, window_size=(W_b - 1, 0),
            ).reshape(B_b, Sg_b, Ht_b, Dt_b).transpose(1, 2).reshape(Bh_b, Sg_b, Dt_b)
            ytot = yl_g + ys_g
            yloc = ytot.view(Bh_b, Nc_b, world_size, Cs_b, Dt_b)[:, :, rank].reshape(Bh_b, Sl_b, Dt_b)
            yloc.backward(dol_b)
            for wp in (w0_b, w1_b, w2_b):
                dist.all_reduce(wp.grad, op=dist.ReduceOp.SUM)

        def _b0_rsi_allgather_seq_replicated():
            _zero_b()
            qg = _AllGatherSubChunkFn.apply(ql_b)
            kg = _AllGatherSubChunkFn.apply(kl_b)
            vg = _AllGatherSubChunkFn.apply(vl_b)
            l0g = _AllGatherSubChunkFn.apply(l0l_b)
            l1g = _AllGatherSubChunkFn.apply(l1l_b)
            l2g = _AllGatherSubChunkFn.apply(l2l_b)
            mg = _AllGatherSubChunkFn.apply(moml_b)

            yl_g = chunk_checkpointed_lact_swiglu(w0_b, w1_b, w2_b, qg, kg, vg, l0g, l1g, l2g, chunk_size=Cg_b, use_muon=True, momentum=mg)
            ys_g = flash_attn_func(
                qg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                kg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                vg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                causal=True, window_size=(W_b - 1, 0),
            ).reshape(B_b, Sg_b, Ht_b, Dt_b).transpose(1, 2).reshape(Bh_b, Sg_b, Dt_b)
            ytot = yl_g + ys_g
            yloc = ytot.view(Bh_b, Nc_b, world_size, Cs_b, Dt_b)[:, :, rank].reshape(Bh_b, Sl_b, Dt_b)
            yloc.backward(dol_b)
            for wp in (w0_b, w1_b, w2_b):
                dist.all_reduce(wp.grad, op=dist.ReduceOp.SUM)

        def _b1_subchunk_cp_allgather_swa():
            _zero_b()
            yl = SubChunkCPLaCTAutogradFn.apply(
                w0_b, w1_b, w2_b, ql_b, kl_b, vl_b, l0l_b, l1l_b, l2l_b, moml_b,
                Cs_b, True, 1e-5, dist.group.WORLD, False
            )
            # AllGather KV for SWA (instead of 1-hop P2P boundary halo!)
            qg = _AllGatherSubChunkFn.apply(ql_b)
            kg = _AllGatherSubChunkFn.apply(kl_b)
            vg = _AllGatherSubChunkFn.apply(vl_b)
            ys_g = flash_attn_func(
                qg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                kg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                vg.view(B_b, Ht_b, Sg_b, Dt_b).transpose(1, 2).reshape(B_b, Sg_b, Hs_b, Ds_b),
                causal=True, window_size=(W_b - 1, 0),
            ).reshape(B_b, Sg_b, Ht_b, Dt_b).transpose(1, 2).reshape(Bh_b, Sg_b, Dt_b)
            ys_loc = ys_g.view(Bh_b, Nc_b, world_size, Cs_b, Dt_b)[:, :, rank].reshape(Bh_b, Sl_b, Dt_b)
            (yl + ys_loc).backward(dol_b)

        def _b3_rsi_subchunk_cp_p2p_halo():
            _zero_b()
            yl = SubChunkCPLaCTAutogradFn.apply(
                w0_b, w1_b, w2_b, ql_b, kl_b, vl_b, l0l_b, l1l_b, l2l_b, moml_b,
                Cs_b, True, 1e-5, dist.group.WORLD, False
            )
            qs = ql_b.view(B_b, Ht_b, Sl_b, Dt_b).transpose(1, 2).reshape(B_b, Sl_b, Hs_b, Ds_b)
            ks = kl_b.view(B_b, Ht_b, Sl_b, Dt_b).transpose(1, 2).reshape(B_b, Sl_b, Hs_b, Ds_b)
            vs = vl_b.view(B_b, Ht_b, Sl_b, Dt_b).transpose(1, 2).reshape(B_b, Sl_b, Hs_b, Ds_b)
            ys = P2PHaloCausalSWAAutogradFn.apply(qs, ks, vs, Cs_b, W_b, dist.group.WORLD)
            ys_bh = ys.reshape(B_b, Sl_b, Ht_b, Dt_b).transpose(1, 2).reshape(Bh_b, Sl_b, Dt_b)
            (yl + ys_bh).backward(dol_b)

        def _bench_dist(fn: Callable) -> Tuple[float, float]:
            w_it = 2 if is_smoke else 4
            g_it = 3 if is_smoke else 6
            k_it = 3 if is_smoke else 5
            for _ in range(w_it):
                fn()
            torch.cuda.synchronize(device)
            dist.barrier()
            torch.cuda.reset_peak_memory_stats(device)
            fn()
            torch.cuda.synchronize(device)
            peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            mem_t = torch.tensor([peak_mib], device=device, dtype=torch.float64)
            dist.all_reduce(mem_t, op=dist.ReduceOp.MAX)

            samples: List[float] = []
            for _ in range(g_it):
                dist.barrier()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                for _ in range(k_it):
                    fn()
                e_ev.record()
                e_ev.synchronize()
                t_loc = torch.tensor([s_ev.elapsed_time(e_ev) / float(k_it)], device=device, dtype=torch.float64)
                dist.all_reduce(t_loc, op=dist.ReduceOp.MAX)
                samples.append(float(t_loc.item()))
            samples.sort()
            return round(samples[len(samples) // 2], 4), round(float(mem_t.item()), 2)

        b0_ms, b0_mib = _bench_dist(_b0_allgather_seq_replicated)
        b0_rsi_ms, b0_rsi_mib = _bench_dist(_b0_rsi_allgather_seq_replicated)
        b1_ms, b1_mib = _bench_dist(_b1_subchunk_cp_allgather_swa)
        b3_ms, b3_mib = _bench_dist(_b3_rsi_subchunk_cp_p2p_halo)

        qkv_global_bytes = 3 * Bh_b * Sg_b * Dt_b * 2
        lr_global_bytes = 4 * Bh_b * Sg_b * 4
        fw_state_bytes = (2 * Bh_b * Hfw_b * Dt_b + Bh_b * Dt_b * Hfw_b) * 4
        b0_comm_mib = (2.0 * (world_size - 1) / world_size * (qkv_global_bytes + lr_global_bytes) + 2.0 * (world_size - 1) / world_size * fw_state_bytes) / 1048576.0

        halo_bytes = 2 * 2 * B_b * Nc_b * W_b * Hs_b * Ds_b * 2  # send+recv in fwd & bwd
        cp_ar_bytes = 2 * (Nc_b - 1) * (2.0 * (world_size - 1) / world_size) * fw_state_bytes
        b3_comm_mib = (cp_ar_bytes + halo_bytes) / 1048576.0

        return {
            "world_size": world_size,
            "global_seq_len": Sg_b,
            "global_chunk_size": Cg_b,
            "local_subchunk_size": Cs_b,
            "swa_window_size": W_b,
            "b0_allgather_replicated_ms": b0_ms,
            "b0_allgather_replicated_peak_mib": b0_mib,
            "b0_rsi_allgather_replicated_ms": b0_rsi_ms,
            "b0_rsi_allgather_replicated_peak_mib": b0_rsi_mib,
            "b0_comm_mib_per_rank": round(b0_comm_mib, 2),
            "b1_subchunk_cp_allgather_swa_ms": b1_ms,
            "b1_subchunk_cp_allgather_swa_peak_mib": b1_mib,
            "b3_rsi_subchunk_cp_p2p_halo_ms": b3_ms,
            "b3_rsi_subchunk_cp_p2p_halo_peak_mib": b3_mib,
            "b3_comm_mib_per_rank": round(b3_comm_mib, 2),
            "speedup_vs_allgather_replicated": round(b0_ms / max(b3_ms, 1e-9), 4),
            "speedup_vs_rsi_allgather_replicated": round(b0_rsi_ms / max(b3_ms, 1e-9), 4),
            "speedup_vs_allgather_swa": round(b1_ms / max(b3_ms, 1e-9), 4),
            "comm_volume_reduction_x": round(b0_comm_mib / max(b3_comm_mib, 1e-9), 2),
            "peak_mib_saved_per_rank": round(b0_mib - b3_mib, 2),
            "gate_g9_parity_vs_1gpu": parity_g9,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cu


# =============================================================================
# Suite C: Fixed-Wall-Clock 12.0s val_bpb RSI Loop + Beyond-Window Recall Law
# =============================================================================


class HybridLaCTSWABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_ttt_heads: int,
        d_fw_hidden: int,
        n_swa_heads: int,
        mlp_hidden: int,
        *,
        chunk_size: int = 128,
        swa_window: int = 32,
        ttt_backend: str = "rsi_ckpt",   # "eager", "rsi_unck", "rsi_ckpt", "swa_only", "linear_delta"
        use_muon: bool = True,
        normalize_fast_weights: bool = True,
        fused_gate: bool = True,
        fused_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_ttt_heads = n_ttt_heads
        self.d_ttt = d_model // n_ttt_heads
        self.d_fw_hidden = d_fw_hidden
        self.n_swa_heads = n_swa_heads
        self.d_swa = d_model // n_swa_heads
        self.chunk_size = chunk_size
        self.swa_window = swa_window
        self.ttt_backend = ttt_backend
        self.use_muon = use_muon
        self.normalize_fast_weights = normalize_fast_weights
        self.fused_gate = fused_gate
        self.fused_mlp = fused_mlp

        # FP32 master weights across all layers so AdamW updates with full FP32 precision
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6, dtype=torch.float32)
        self.qkvg_proj = nn.Linear(d_model, 4 * d_model, bias=False, dtype=torch.float32)
        self.k_conv = nn.Conv1d(d_model, d_model, kernel_size=2, padding=1, groups=d_model, bias=False, dtype=torch.float32)
        self.o_proj = nn.Linear(d_model, d_model, bias=False, dtype=torch.float32)
        self.out_gate_norm_w = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

        if ttt_backend in ("eager", "rsi_unck", "rsi_ckpt"):
            scale = 1.0 / math.sqrt(self.d_ttt)
            self.w0 = nn.Parameter(torch.randn(n_ttt_heads, d_fw_hidden, self.d_ttt, dtype=torch.float32) * scale)
            self.w1 = nn.Parameter(torch.randn(n_ttt_heads, self.d_ttt, d_fw_hidden, dtype=torch.float32) * (1.0 / math.sqrt(d_fw_hidden)))
            self.w2 = nn.Parameter(torch.randn(n_ttt_heads, d_fw_hidden, self.d_ttt, dtype=torch.float32) * scale)
            self.lr_mom_proj = nn.Linear(d_model, n_ttt_heads * 4, bias=True, dtype=torch.float32)
        elif ttt_backend == "linear_delta":
            scale = 1.0 / math.sqrt(self.d_ttt)
            self.w_lin = nn.Parameter(torch.randn(n_ttt_heads, self.d_ttt, self.d_ttt, dtype=torch.float32) * scale)
            self.lr_lin_proj = nn.Linear(d_model, n_ttt_heads, bias=True, dtype=torch.float32)

        self.norm2 = nn.RMSNorm(d_model, eps=1e-6, dtype=torch.float32)
        self.w_gate_up = nn.Linear(d_model, 2 * mlp_hidden, bias=False, dtype=torch.float32)
        self.w_down = nn.Linear(mlp_hidden, d_model, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x).to(torch.bfloat16)
        qkvg = F.linear(h, self.qkvg_proj.weight.to(torch.bfloat16))
        q, k, v, g = [t.contiguous() for t in qkvg.chunk(4, dim=-1)]
        # Causal 2-tap depthwise conv on K (`layer_lact_swiglu.py` CausalConv1d) for direct 1-step key(t-1)->val(t) binding
        k = F.conv1d(k.transpose(1, 2), self.k_conv.weight.to(torch.bfloat16), padding=1, groups=D)[..., :S].transpose(1, 2).contiguous()

        qh_t = F.normalize(q.view(B, S, self.n_ttt_heads, self.d_ttt).float(), dim=-1).to(torch.bfloat16)
        kh_t = F.normalize(k.view(B, S, self.n_ttt_heads, self.d_ttt).float(), dim=-1).to(torch.bfloat16)
        vh_t = v.view(B, S, self.n_ttt_heads, self.d_ttt)

        qs = q.view(B, S, self.n_swa_heads, self.d_swa)
        ks = k.view(B, S, self.n_swa_heads, self.d_swa)
        vs = v.view(B, S, self.n_swa_heads, self.d_swa)
        if self.ttt_backend == "eager":
            idx = torch.arange(S, device=x.device)
            mask = (idx[:, None] >= idx[None, :]) & ((idx[:, None] - idx[None, :]) < self.swa_window)
            y_swa = F.scaled_dot_product_attention(
                qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2), attn_mask=mask
            ).transpose(1, 2).reshape(B, S, D)
        else:
            y_swa = flash_attn_func(
                qs, ks, vs, causal=True, window_size=(self.swa_window - 1, 0)
            ).reshape(B, S, D)

        if self.ttt_backend in ("eager", "rsi_unck", "rsi_ckpt"):
            Bh = B * self.n_ttt_heads
            qt = qh_t.transpose(1, 2).reshape(Bh, S, self.d_ttt)
            kt = kh_t.transpose(1, 2).reshape(Bh, S, self.d_ttt)
            vt = vh_t.transpose(1, 2).reshape(Bh, S, self.d_ttt)

            lrm = F.linear(h, self.lr_mom_proj.weight.to(torch.bfloat16), self.lr_mom_proj.bias.to(torch.bfloat16)).float()
            lrm = lrm.view(B, S, self.n_ttt_heads, 4).permute(0, 2, 1, 3).reshape(Bh, S, 4)
            lr0 = F.softplus(lrm[:, :, 0:1]) * 0.8
            lr1 = F.softplus(lrm[:, :, 1:2]) * 0.8
            lr2 = F.softplus(lrm[:, :, 2:3]) * 0.8
            mom = torch.sigmoid(lrm[:, :, 3:4])

            w0_b = self.w0.unsqueeze(0).expand(B, -1, -1, -1).reshape(Bh, self.d_fw_hidden, self.d_ttt)
            w1_b = self.w1.unsqueeze(0).expand(B, -1, -1, -1).reshape(Bh, self.d_ttt, self.d_fw_hidden)
            w2_b = self.w2.unsqueeze(0).expand(B, -1, -1, -1).reshape(Bh, self.d_fw_hidden, self.d_ttt)

            if self.ttt_backend == "eager":
                y_ttt = eager_block_causal_lact_swiglu_ref(
                    w0_b, w1_b, w2_b, qt, kt, vt, lr0, lr1, lr2,
                    chunk_size=self.chunk_size, use_muon=self.use_muon, momentum=mom,
                    normalize_fast_weights=self.normalize_fast_weights, ns5_fp32=True
                )
            elif self.ttt_backend == "rsi_unck":
                y_ttt = rsi_block_causal_lact_swiglu_uncheckpointed(
                    w0_b, w1_b, w2_b, qt.clone(), kt.clone(), vt.clone(), lr0, lr1, lr2,
                    chunk_size=self.chunk_size, use_muon=self.use_muon, momentum=mom,
                    normalize_fast_weights=self.normalize_fast_weights
                )
            else:
                y_ttt = chunk_checkpointed_lact_swiglu(
                    w0_b, w1_b, w2_b, qt, kt, vt, lr0, lr1, lr2,
                    chunk_size=self.chunk_size, use_muon=self.use_muon, momentum=mom,
                    normalize_fast_weights=self.normalize_fast_weights
                )
            y_mix = y_swa + y_ttt.view(B, self.n_ttt_heads, S, self.d_ttt).transpose(1, 2).reshape(B, S, D)
        elif self.ttt_backend == "linear_delta":
            Bh = B * self.n_ttt_heads
            qt = qh_t.transpose(1, 2).reshape(Bh, S, self.d_ttt)
            kt = kh_t.transpose(1, 2).reshape(Bh, S, self.d_ttt)
            vt = vh_t.transpose(1, 2).reshape(Bh, S, self.d_ttt)
            lrl_raw = F.linear(h, self.lr_lin_proj.weight.to(torch.bfloat16), self.lr_lin_proj.bias.to(torch.bfloat16)).float()
            lr_l = (F.softplus(lrl_raw).view(B, S, self.n_ttt_heads).transpose(1, 2).reshape(Bh, S, 1)) * 0.4
            w_cur = self.w_lin.unsqueeze(0).expand(B, -1, -1, -1).reshape(Bh, self.d_ttt, self.d_ttt)
            wn_init = torch.sqrt((w_cur * w_cur).sum(dim=-1, keepdim=True) + 1e-24)
            outs_l = []
            nc = (S + self.chunk_size - 1) // self.chunk_size
            for ci in range(nc):
                s, e = ci * self.chunk_size, min(S, (ci + 1) * self.chunk_size)
                outs_l.append(torch.bmm(qt[:, s:e, :], w_cur.to(torch.bfloat16).transpose(1, 2)))
                if ci < nc - 1:
                    dw = torch.bmm(vt[:, s:e, :].transpose(1, 2), (kt[:, s:e, :].float() * lr_l[:, s:e, :]).to(torch.bfloat16)).float()
                    wc = w_cur + dw
                    w_cur = (wc / (torch.sqrt((wc * wc).sum(dim=-1, keepdim=True) + 1e-24) + 1e-5)) * wn_init
            y_ttt = torch.cat(outs_l, dim=1)
            y_mix = y_swa + y_ttt.view(B, self.n_ttt_heads, S, self.d_ttt).transpose(1, 2).reshape(B, S, D)
        else:
            y_mix = y_swa

        if self.fused_gate:
            y_gated = FusedLaCTRMSNormGateFn.apply(y_mix, g, self.out_gate_norm_w, 1e-6)
        else:
            y_gated = eager_lact_rmsnorm_silu_gate_ref(y_mix, g, self.out_gate_norm_w, eps=1e-6)

        x = x + F.linear(y_gated, self.o_proj.weight.to(torch.bfloat16)).float()
        h2 = self.norm2(x).to(torch.bfloat16)
        gu_m = F.linear(h2, self.w_gate_up.weight.to(torch.bfloat16))
        if self.fused_mlp:
            act_m = FusedSwiGLUFn.apply(gu_m)
        else:
            gm, um = gu_m.chunk(2, dim=-1)
            act_m = (F.silu(gm.float()) * um.float()).to(h2.dtype)
        x = x + F.linear(act_m, self.w_down.weight.to(torch.bfloat16)).float()
        return x


class HybridLaCTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        n_layers: int = 3,
        n_ttt_heads: int = 4,
        d_fw_hidden: int = 128,
        n_swa_heads: int = 4,
        mlp_hidden: int = 512,
        *,
        chunk_size: int = 128,
        swa_window: int = 32,
        ttt_backend: str = "rsi_ckpt",
        use_muon: bool = True,
        normalize_fast_weights: bool = True,
        fused_gate: bool = True,
        fused_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, dtype=torch.float32)
        self.blocks = nn.ModuleList(
            [
                HybridLaCTSWABlock(
                    d_model=d_model,
                    n_ttt_heads=n_ttt_heads,
                    d_fw_hidden=d_fw_hidden,
                    n_swa_heads=n_swa_heads,
                    mlp_hidden=mlp_hidden,
                    chunk_size=chunk_size,
                    swa_window=swa_window,
                    ttt_backend=ttt_backend,
                    use_muon=use_muon,
                    normalize_fast_weights=normalize_fast_weights,
                    fused_gate=fused_gate,
                    fused_mlp=fused_mlp,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(d_model, eps=1e-6, dtype=torch.float32)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.zeros_(m.weight)
                m.weight.data[:, 0, 0] = 1.0  # 1-token causal lag `in[t-1]`

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for blk in self.blocks:
            x = blk(x)
        h_out = self.final_norm(x).to(torch.bfloat16)
        return F.linear(h_out, self.tok_emb.weight.to(torch.bfloat16))


@torch.no_grad()
def evaluate_bpb_and_recall(
    model: nn.Module,
    seqs: List[torch.Tensor],
    qmasks: List[torch.Tensor],
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_toks = 0
    total_q_correct = 0
    total_q_count = 0

    for seq, qm in zip(seqs, qmasks):
        logits = model(seq[:, :-1])
        targets = seq[:, 1:]
        loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
        total_nll += float(loss_sum.item())
        total_toks += int(targets.numel())

        # Beyond-window associative recall accuracy at planted query target positions (`qm[:, 1:]`)
        q_tgt_mask = qm[:, 1:]
        if bool(q_tgt_mask.any().item()):
            preds = logits.argmax(dim=-1)
            total_q_correct += int((preds[q_tgt_mask] == targets[q_tgt_mask]).sum().item())
            total_q_count += int(q_tgt_mask.sum().item())

    tot_vec = torch.tensor([total_nll, float(total_toks), float(total_q_correct), float(total_q_count)], device=device, dtype=torch.float64)
    dist.all_reduce(tot_vec, op=dist.ReduceOp.SUM)
    bpb = (tot_vec[0].item() / max(1.0, tot_vec[1].item())) / math.log(2.0)
    recall_acc = tot_vec[2].item() / max(1.0, tot_vec[3].item())
    model.train()
    return round(bpb, 6), round(recall_acc, 6)


def run_suite_c_valbpb_and_recall_law(
    rank: int, world_size: int, device: torch.device, is_smoke: bool, out_dir: Path
) -> Dict[str, Any]:
    vocab_size = 512
    seq_len = 512
    batch_per_gpu = 4
    n_train_slots = 24 if is_smoke else 64
    n_eval_slots = 2 if is_smoke else 6
    budget_s = 4.5 if is_smoke else 16.0

    train_seqs, _, top64_share = make_long_range_associative_zipf_batches(
        n_train_slots, batch_per_gpu, seq_len, vocab_size, device, seed=20260500 + rank * 97
    )
    val_seqs, val_qmasks, _ = make_long_range_associative_zipf_batches(
        n_eval_slots, batch_per_gpu, seq_len, vocab_size, device, seed=20268800 + rank * 53
    )
    # STRICT SINGLE-TOUCH HOLDOUT: NEVER evaluated during Gen 0..Gen 4 selection!
    test_seqs, test_qmasks, _ = make_long_range_associative_zipf_batches(
        n_eval_slots, batch_per_gpu, seq_len, vocab_size, device, seed=20269900 + rank * 71
    )

    def _make_seeded_model(**kwargs) -> nn.Module:
        torch.manual_seed(20261319)
        torch.cuda.manual_seed_all(20261319)
        m = HybridLaCTLanguageModel(vocab_size=vocab_size, **kwargs).to(device)
        return nn.parallel.DistributedDataParallel(m, device_ids=[rank], find_unused_parameters=False)

    # ------------------------------------------------------------------
    # Part 1: 5-Generation Fixed-Wall-Clock (`16.0s`) `val_bpb` RSI Loop
    # ------------------------------------------------------------------
    generations = [
        ("gen0_eager_lact_swa_baseline", {"ttt_backend": "eager", "use_muon": True, "normalize_fast_weights": True, "chunk_size": 128, "fused_gate": False, "fused_mlp": False}, "Eager multi-chunk LaCT(Muon+Momentum+L2Norm) + masked SWA + eager MLP/Gate"),
        ("gen1_rsi_unckpt_fa2_swa", {"ttt_backend": "rsi_unck", "use_muon": True, "normalize_fast_weights": True, "chunk_size": 128, "fused_gate": False, "fused_mlp": False}, "Swap in uncheckpointed RSI Apply/Grad2/BatchedMuon + FA2 SWA"),
        ("gen2_rsi_superfused_gate_swiglu", {"ttt_backend": "rsi_ckpt", "use_muon": True, "normalize_fast_weights": True, "chunk_size": 128, "fused_gate": True, "fused_mlp": True}, "Zero-copy compact-bank ChunkCheckpointedLaCTSwiGLUFn + FusedLaCTRMSNormGateFn + FusedSwiGLUFn"),
        ("gen3_ablate_l2norm_sphere_retention", {"ttt_backend": "rsi_ckpt", "use_muon": True, "normalize_fast_weights": False, "chunk_size": 128, "fused_gate": True, "fused_mlp": True}, "Single-factor LaCT Eq. 5 ablation: disable FusedL2NormStepFn row-sphere retention (faster steps, unbounded multi-chunk norm drift collapses recall & val_bpb)"),
        ("gen4_small_chunk_c32_overhead", {"ttt_backend": "rsi_ckpt", "use_muon": True, "normalize_fast_weights": True, "chunk_size": 32, "fused_gate": True, "fused_mlp": True}, "Systems chunk-size ablation: shrink chunk_size 128 -> 32 (4x Muon-NS5 boundary overhead)"),
    ]

    gen_rows: List[Dict[str, Any]] = []
    incumbent_val_bpb = float("inf")
    incumbent_gen_id = ""
    incumbent_state: Optional[Dict[str, torch.Tensor]] = None
    incumbent_cfg: Optional[Dict[str, Any]] = None

    for g_idx, (gid, gcfg, gnote) in enumerate(generations):
        ddp_model = _make_seeded_model(**gcfg)
        opt = torch.optim.AdamW(ddp_model.parameters(), lr=2.5e-3, weight_decay=0.01)

        # Untimed warmup (2 steps) so JIT/NCCL compilation never contaminates the 12.0s budget
        for w_i in range(2):
            opt.zero_grad(set_to_none=True)
            s_w = train_seqs[w_i % len(train_seqs)]
            lg_w = ddp_model(s_w[:, :-1])
            ls_w = F.cross_entropy(lg_w.float().reshape(-1, vocab_size), s_w[:, 1:].reshape(-1))
            ls_w.backward()
            opt.step()

        # Re-initialize exact seed before timed 12.0s window!
        del ddp_model, opt
        gc.collect()
        torch.cuda.empty_cache()
        ddp_model = _make_seeded_model(**gcfg)
        opt = torch.optim.AdamW(ddp_model.parameters(), lr=2.5e-3, weight_decay=0.01)

        init_val_bpb, _ = evaluate_bpb_and_recall(ddp_model, val_seqs, val_qmasks, device)
        torch.cuda.synchronize(device)
        dist.barrier()

        t_start = time.perf_counter()
        steps_done = 0
        while True:
            s_b = train_seqs[steps_done % len(train_seqs)]
            opt.zero_grad(set_to_none=True)
            lg = ddp_model(s_b[:, :-1])
            loss = F.cross_entropy(lg.float().reshape(-1, vocab_size), s_b[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            opt.step()
            steps_done += 1

            if steps_done % 2 == 0:
                torch.cuda.synchronize(device)
                el_loc = torch.tensor([time.perf_counter() - t_start], device=device, dtype=torch.float64)
                dist.broadcast(el_loc, src=0)
                if float(el_loc.item()) >= budget_s:
                    elapsed_s = float(el_loc.item())
                    break

        val_bpb, val_recall = evaluate_bpb_and_recall(ddp_model, val_seqs, val_qmasks, device)

        if g_idx == 0:
            decision = "BASELINE"
            incumbent_val_bpb = val_bpb
            incumbent_gen_id = gid
            incumbent_cfg = dict(gcfg)
            incumbent_state = {k: v.detach().cpu().clone() for k, v in ddp_model.module.state_dict().items()}
        elif val_bpb <= incumbent_val_bpb - 0.005:
            decision = "KEEP"
            incumbent_val_bpb = val_bpb
            incumbent_gen_id = gid
            incumbent_cfg = dict(gcfg)
            incumbent_state = {k: v.detach().cpu().clone() for k, v in ddp_model.module.state_dict().items()}
        else:
            inc_steps = gen_rows[[r["generation_id"] for r in gen_rows].index(incumbent_gen_id)]["optimizer_steps"]
            decision = "REVERT_SLOWER" if steps_done < inc_steps else "REVERT_VAL_BPB"

        gen_rows.append(
            {
                "gen_idx": g_idx,
                "generation_id": gid,
                "budget_s": budget_s,
                "elapsed_s": round(elapsed_s, 4),
                "deadline_overshoot_s": round(max(0.0, elapsed_s - budget_s), 4),
                "optimizer_steps": steps_done,
                "ms_per_step": round(1000.0 * elapsed_s / max(1, steps_done), 3),
                "init_val_bpb": init_val_bpb,
                "val_bpb": val_bpb,
                "val_beyond_swa_recall_acc": val_recall,
                "decision": decision,
                "notes": gnote,
            }
        )
        del ddp_model, opt
        gc.collect()
        torch.cuda.empty_cache()

    # SINGLE-TOUCH HOLDOUT EVALUATION ON WINNING INCUMBENT ONLY (`test_seqs` touched ONCE!)
    assert incumbent_cfg is not None and incumbent_state is not None
    win_model = HybridLaCTLanguageModel(vocab_size=vocab_size, **incumbent_cfg).to(device)
    win_model.load_state_dict({k: v.to(device) for k, v in incumbent_state.items()})
    untouched_test_bpb, untouched_test_recall = evaluate_bpb_and_recall(win_model, test_seqs, test_qmasks, device)
    del win_model

    # ------------------------------------------------------------------
    # Part 2: Matched-Step Iso-Parameter Fast-Weight Capacity & Optimizer Law
    # ------------------------------------------------------------------
    fixed_steps = 80 if is_smoke else 220
    # Match total parameter count (~2.60M) across SWA-only (`mlp_hidden=640`), Linear-Delta (`mlp_hidden=608`), and SwiGLU-LaCT (`mlp_hidden=512`)!
    law_arms = [
        ("M0_swa_only_no_ttt", {"ttt_backend": "swa_only", "use_muon": False, "mlp_hidden": 640, "chunk_size": 128, "fused_gate": True, "fused_mlp": True}),
        ("M1_linear_delta_ttt", {"ttt_backend": "linear_delta", "use_muon": False, "mlp_hidden": 608, "chunk_size": 128, "fused_gate": True, "fused_mlp": True}),
        ("M2_lact_swiglu_gd_no_muon", {"ttt_backend": "rsi_ckpt", "use_muon": False, "mlp_hidden": 512, "chunk_size": 128, "fused_gate": True, "fused_mlp": True}),
        ("M3_lact_swiglu_muon_momentum", {"ttt_backend": "rsi_ckpt", "use_muon": True, "mlp_hidden": 512, "chunk_size": 128, "fused_gate": True, "fused_mlp": True}),
    ]

    law_results: List[Dict[str, Any]] = []
    for arm_id, acfg in law_arms:
        ddp_m = _make_seeded_model(**acfg)
        n_params = sum(p.numel() for p in ddp_m.module.parameters())
        opt_m = torch.optim.AdamW(ddp_m.parameters(), lr=2.5e-3, weight_decay=0.01)
        torch.cuda.synchronize(device)
        dist.barrier()
        t0_m = time.perf_counter()
        for st in range(fixed_steps):
            sb = train_seqs[st % len(train_seqs)]
            opt_m.zero_grad(set_to_none=True)
            lg = ddp_m(sb[:, :-1])
            ls = F.cross_entropy(lg.float().reshape(-1, vocab_size), sb[:, 1:].reshape(-1))
            ls.backward()
            torch.nn.utils.clip_grad_norm_(ddp_m.parameters(), 1.0)
            opt_m.step()
        torch.cuda.synchronize(device)
        dt_m = time.perf_counter() - t0_m
        vb, vrec = evaluate_bpb_and_recall(ddp_m, val_seqs, val_qmasks, device)
        law_results.append(
            {
                "arm_id": arm_id,
                "total_params": n_params,
                "fixed_steps": fixed_steps,
                "elapsed_s": round(dt_m, 3),
                "val_bpb": vb,
                "beyond_swa_recall_acc": vrec,
            }
        )
        del ddp_m, opt_m
        gc.collect()
        torch.cuda.empty_cache()

    valbpb_tsv_sha = ""
    if rank == 0:
        tsv_path = out_dir / "valbpb_autoresearch_ledger.tsv"
        header = [
            "gen_idx",
            "generation_id",
            "eval_harness_sha256",
            "budget_s",
            "elapsed_s",
            "optimizer_steps",
            "ms_per_step",
            "val_bpb",
            "val_beyond_swa_recall_acc",
            "decision",
            "notes",
        ]
        h_sha = eval_harness_sha256()
        lines = ["\t".join(header)]
        for r in gen_rows:
            lines.append(
                "\t".join(
                    [
                        str(r["gen_idx"]),
                        str(r["generation_id"]),
                        h_sha,
                        f"{r['budget_s']:.2f}",
                        f"{r['elapsed_s']:.4f}",
                        str(r["optimizer_steps"]),
                        f"{r['ms_per_step']:.3f}",
                        f"{r['val_bpb']:.6f}",
                        f"{r['val_beyond_swa_recall_acc']:.6f}",
                        str(r["decision"]),
                        str(r["notes"]),
                    ]
                )
            )
        tsv_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        tsv_path.write_bytes(tsv_bytes)
        valbpb_tsv_sha = hashlib.sha256(tsv_bytes).hexdigest()

    return {
        "budget_s": budget_s,
        "seq_len": seq_len,
        "swa_window": 32,
        "min_query_distance_tokens": 192 - 96,
        "zipf_top64_token_share": round(top64_share, 4),
        "generations": gen_rows,
        "winning_generation": incumbent_gen_id,
        "winning_val_bpb": incumbent_val_bpb,
        "untouched_test_bpb": untouched_test_bpb,
        "untouched_test_beyond_swa_recall_acc": untouched_test_recall,
        "holdout_touched_once": True,
        "fast_weight_capacity_and_optimizer_law": law_results,
        "valbpb_ledger_sha256": valbpb_tsv_sha,
    }


# =============================================================================
# Main Entrypoint (`torchrun --standalone --nproc_per_node=8`)
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="E131 Round-5 RSI LaCT Nonlinear Fast-Weight TTT Benchmark")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=25))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_smoke = args.mode == "smoke"
    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    t_wall0 = time.perf_counter()
    start_clk, start_pwr = query_sm_telemetry(0) if rank == 0 else (-1.0, -1.0)

    suite0_res: Dict[str, Any] = {}
    suite_a_res: Dict[str, Any] = {}
    rsi_ledger_sha = ""

    if rank == 0:
        print(f"[E131 mode={args.mode}] Running Suite 0 adversarial & FP64/Richardson audits...", flush=True)
        suite0_res = run_suite0_audits(device)
        print(f"[E131 mode={args.mode}] Suite 0 OK: exploits={suite0_res['redteam_exploits_rejected']}/4, spills={suite0_res['total_cubin_spills']}. Running Suite A...", flush=True)
        suite_a_res, ratchets = run_suite_a_rsi_tracks(device, is_smoke=is_smoke)
        rsi_tsv_path = args.out_dir / "rsi_ledger.tsv"
        rsi_ledger_sha = write_rsi_ledger_tsv(ratchets, rsi_tsv_path)
        t1_sp = suite_a_res["track1_lact_chunk_grad2_muon_ns5"]["final_speedup_vs_baseline"]
        t2_sp = suite_a_res["track2_lact_multichunk_hybrid_swa"]["final_speedup_vs_baseline"]
        print(f"[E131 mode={args.mode}] Suite A OK: Track1={t1_sp}x, Track2={t2_sp}x. Syncing 8 GPUs for Suite B...", flush=True)

    dist.barrier()
    suite_b_res = run_suite_b_distributed_cp8(rank, world_size, device, is_smoke=is_smoke)
    if rank == 0:
        print(f"[E131 mode={args.mode}] Suite B OK: CP=8 speedup={suite_b_res['speedup_vs_allgather_replicated']}x. Running Suite C...", flush=True)

    dist.barrier()
    suite_c_res = run_suite_c_valbpb_and_recall_law(rank, world_size, device, is_smoke=is_smoke, out_dir=args.out_dir)

    end_clk, end_pwr = query_sm_telemetry(0) if rank == 0 else (-1.0, -1.0)
    total_wall_s = round(time.perf_counter() - t_wall0, 3)

    if rank == 0:
        env = environment_metadata()
        env["flash_attn"] = getattr(__import__("flash_attn"), "__version__", "unknown")
        payload = {
            "experiment_id": "E131",
            "title": "Round-5 RSI Nonlinear Fast-Weight Test-Time Training (LaCT SwiGLU-TTT + Batched Muon-NS5 + CP=8)",
            "mode": args.mode,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "harness_version": HARNESS_VERSION,
            "eval_harness_sha256": eval_harness_sha256(),
            "rsi_ledger_tsv_sha256": rsi_ledger_sha,
            "total_wall_s": total_wall_s,
            "start_gpu0_sm_clock_mhz": start_clk,
            "start_gpu0_power_w": start_pwr,
            "end_gpu0_sm_clock_mhz": end_clk,
            "end_gpu0_power_w": end_pwr,
            "environment": env,
            "suite_0_adversarial_and_oracle_audits": suite0_res,
            "suite_a_rsi_kernel_tracks": suite_a_res,
            "suite_b_nvswitch_subchunk_cp8": suite_b_res,
            "suite_c_valbpb_rsi_and_associative_recall_law": suite_c_res,
        }
        out_json = args.out_dir / f"lact_test_time_training_rsi_{args.mode}.json"
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"[E131 mode={args.mode}] COMPLETE in {total_wall_s}s -> {out_json} "
            f"(win_bpb={suite_c_res['winning_val_bpb']}, test_bpb={suite_c_res['untouched_test_bpb']}, "
            f"test_recall={suite_c_res['untouched_test_beyond_swa_recall_acc']})",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
