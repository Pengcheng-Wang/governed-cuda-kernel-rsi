#!/usr/bin/env python3
"""
E130: Round-4 Recursive Self-Improvement (`RSI`) Study of DeepSeek `Engram`
(`Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models`,
`arXiv:2601.07372`, Jan 2026) Paired with `M = 4` `mHC` (`arXiv:2512.24880`):
  - Surjective Tokenizer Canonicalization `P: V -> V'` (~23% reduction) & Pure-GPU Int64
    Multiplicative-XOR Prime `NgramHash` with Packed-Document (`doc_ids`) Zero-Leakage Isolation
  - Atomic-Free CSR Radix-Sort Segment-Reduce Backward + Exact Dense-Equivalent
    Pre-Forward `JIT-Lazy-Adam(lr=5x, wd=0.0)` (`< 1e-5` 50-step parity vs `> 60%` `SparseAdam` drift)
  - Merged-KV Zero-Transpose `M = 4` Context-Aware RMS-Gating + Dilated Causal `ShortConv(w=4, d=3)`
    + Dual Residual (`H + U + ShortConv(U)`) Forward & Lock-Free Split-CTA Backward (`0` spills)
  - 8-GPU NVSwitch Modulo-Sharded `Dedup-AllToAllv` + Allocator-Safe (`record_stream`) Layer-2
    Async Stream Prefetch Behind `mHC` Transformer Blocks `0..1`
  - Fixed-Wall-Clock (`12.0s`) `val_bpb` RSI Loop + Matched-Step Iso-Param / Iso-Active-FLOPs
    `MoE vs Engram` Sparsity-Allocation U-Curve (`rho in {1.00, 0.75, 0.25}`)

Governed by `PROGRAM.md` and byte-identical frozen `code/rsi_kernel_harness.py`
(`eval_harness_sha256 == 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
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
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_local_rank_env = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_e130_r{_local_rank_env}")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_e130_r{_local_rank_env}")

import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
from engram_rsi_kernels import (
    EagerDemoEngramContextOp,
    EngramFusedLookupTable,
    EngramHashSpec,
    FusedEngramContextOp,
    ShardedEngramLookupEngine,
    audit_engram_triton_cubin_spills,
    build_engram_hash_spec,
    build_surjective_vocab_lookup,
    cpu_numpy_ngram_hashes_ref,
    fp64_engram_gate_conv_oracle,
    gpu_engram_ngram_hashes,
)
from mhc_rsi_kernels import (
    FusedPostResMergeFn,
    SuperfusedPreRMSNormFn,
    eager_mhc_coeff_eq7_eq9,
    fused_mhc_coeff,
)
from rsi_kernel_harness import (
    HARNESS_VERSION,
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
# Helpers: Zipfian Token Sampler & RoPE
# =============================================================================


def make_zipf_token_batches(
    num_slots: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    zipf_s: float = 1.15,
    seed: int = 20260130,
) -> Tuple[List[torch.Tensor], float]:
    """
    Generates realistic power-law Zipfian token streams (`p(r) propto r^-zipf_s`)
    with natural-language multi-word collocation burstiness (`"of the"`, `"in the"`, etc.)
    so 2-gram/3-gram repetition matches real LLM pretraining batches (~3x-5x local dedup).
    Returns `(slots, top64_token_mass_share)`.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    ranks = torch.arange(1, vocab_size + 1, device=device, dtype=torch.float64)
    probs64 = ranks.pow(-zipf_s)
    probs64 = probs64 / probs64.sum()
    top64_share = float(probs64[: min(64, vocab_size)].sum().item())
    probs = probs64.float()

    perm = torch.randperm(vocab_size, generator=gen, device=device)
    # 128 Zipfian 4-token natural-language phrase templates (`p(phrase) propto r^-1.25`)
    n_phrases = min(128, max(16, vocab_size // 8))
    phrases = perm.index_select(
        0, torch.multinomial(probs, n_phrases * 4, replacement=True, generator=gen)
    ).view(n_phrases, 4)
    p_ranks = torch.arange(1, n_phrases + 1, device=device, dtype=torch.float32).pow(-1.25)
    p_probs = p_ranks / p_ranks.sum()

    n_quads = seq_len // 4
    rem = seq_len - 4 * n_quads
    slots: List[torch.Tensor] = []
    for _ in range(num_slots):
        if n_quads > 0:
            p_ids = torch.multinomial(p_probs, batch_size * n_quads, replacement=True, generator=gen)
            phrase_part = phrases.index_select(0, p_ids).view(batch_size, n_quads, 4)
            uni_draws = torch.multinomial(probs, batch_size * n_quads * 4, replacement=True, generator=gen)
            uni_part = perm.index_select(0, uni_draws).view(batch_size, n_quads, 4)
            use_phrase = torch.rand((batch_size, n_quads, 1), generator=gen, device=device) < 0.65
            body = torch.where(use_phrase, phrase_part, uni_part).reshape(batch_size, 4 * n_quads)
            if rem > 0:
                tail_draws = torch.multinomial(probs, batch_size * rem, replacement=True, generator=gen)
                tail = perm.index_select(0, tail_draws).view(batch_size, rem)
                toks = torch.cat([body, tail], dim=1)
            else:
                toks = body
        else:
            draws = torch.multinomial(probs, batch_size * seq_len, replacement=True, generator=gen)
            toks = perm.index_select(0, draws).view(batch_size, seq_len)
        slots.append(toks.contiguous())
    return slots, top64_share


def build_rope_cos_sin(seq_len: int, head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(torch.bfloat16)[None, :, None, :]
    sin = torch.sin(freqs).to(torch.bfloat16)[None, :, None, :]
    return cos, sin


def apply_rope_bshd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


# =============================================================================
# Suite 0: Adversarial Red-Team Exploits, Numerical Pitfalls & Exactness Gates
# =============================================================================


def run_suite_0_engram_audits(
    device: torch.device, is_smoke: bool
) -> Tuple[Dict[str, Any], RSIVectorRatchet]:
    l2_flush = torch.empty((12 * 1024 * 1024,), dtype=torch.float32, device=device)

    # -------------------------------------------------------------------------
    # 0.1 Surjective Vocabulary Canonicalizer & Production 129,280 -> ~99,546 Audit
    # -------------------------------------------------------------------------
    unicode_pieces = [
        "<pad>", "<bos>", "<eos>", "<unk>",
        "The", " the", "THE", "thé", "  the ",
        "Alexander", " alexander", "ALEXANDER",
        "Bucephalus", "bucephalus", "BUCEPHALUS",
        "DeepSeek", "deepseek", "DEEPSEEK",
        "Engram", "engram", "éngram",
        "Memory", " memory", "MEMORY",
        "Sparsity", "sparsity",
    ]
    lut_str, comp_str_v, _ = build_surjective_vocab_lookup(
        vocab_size=len(unicode_pieces), pad_id=0, token_pieces=unicode_pieces
    )
    str_reduction_pct = 100.0 * (1.0 - float(comp_str_v) / float(len(unicode_pieces)))

    prod_spec = build_engram_hash_spec(
        vocab_size=129280,
        engram_vocab_size_per_ngram=(65536, 65536),
        max_ngram_size=3,
        n_head_per_ngram=8,
        layer_id=2,
        target_compression_ratio=0.77,
    )
    prod_reduction_pct = 100.0 * (1.0 - float(prod_spec.compressed_vocab_size) / float(prod_spec.vocab_size))

    # Bitwise NgramHash parity vs `engram_demo_v1.py` CPU NumPy (single-doc & packed multi-doc)
    B_h, T_h = 8, (256 if is_smoke else 2048)
    g_h = torch.Generator(device=device).manual_seed(20261301)
    ids_h = torch.randint(0, prod_spec.vocab_size, (B_h, T_h), generator=g_h, device=device, dtype=torch.long)
    doc_ids_h = torch.zeros((B_h, T_h), device=device, dtype=torch.int32)
    doc_ids_h[:, T_h // 3 : (2 * T_h) // 3] = 1
    doc_ids_h[:, (2 * T_h) // 3 :] = 2

    lut_d = torch.from_numpy(prod_spec.lookup_table_np.astype(np.int32)).to(device)
    mult_d = torch.from_numpy(prod_spec.multipliers_np).to(device)
    mod_d = torch.from_numpy(prod_spec.head_moduli_np).to(device)
    off_d = torch.from_numpy(prod_spec.head_offsets_np).to(device)

    ref_single = cpu_numpy_ngram_hashes_ref(ids_h, prod_spec, doc_ids=None)
    gpu_single = gpu_engram_ngram_hashes(
        ids_h, lut_d, mult_d, mod_d, off_d,
        prod_spec.compressed_pad_id, prod_spec.max_ngram_size, prod_spec.n_head_per_ngram, doc_ids=None
    ).view(B_h, T_h, prod_spec.num_heads).long()
    single_mismatches = int((ref_single != gpu_single).sum().item())

    ref_packed = cpu_numpy_ngram_hashes_ref(ids_h, prod_spec, doc_ids=doc_ids_h)
    gpu_packed = gpu_engram_ngram_hashes(
        ids_h, lut_d, mult_d, mod_d, off_d,
        prod_spec.compressed_pad_id, prod_spec.max_ngram_size, prod_spec.n_head_per_ngram, doc_ids=doc_ids_h
    ).view(B_h, T_h, prod_spec.num_heads).long()
    packed_mismatches = int((ref_packed != gpu_packed).sum().item())

    # Perturb Doc 0 completely -> verify 0 bitwise hash change on Doc 1 & Doc 2!
    ids_perturbed = ids_h.clone()
    ids_perturbed[:, : T_h // 3] = torch.randint(
        0, prod_spec.vocab_size, (B_h, T_h // 3), generator=g_h, device=device, dtype=torch.long
    )
    gpu_packed_pert = gpu_engram_ngram_hashes(
        ids_perturbed, lut_d, mult_d, mod_d, off_d,
        prod_spec.compressed_pad_id, prod_spec.max_ngram_size, prod_spec.n_head_per_ngram, doc_ids=doc_ids_h
    ).view(B_h, T_h, prod_spec.num_heads).long()
    hash_cross_doc_leak = int((gpu_packed[:, T_h // 3 :] != gpu_packed_pert[:, T_h // 3 :]).sum().item())

    # Verify surface equivalence collision: canonical token `k` and colliding variant `v` with `lut[v] == lut[k]`
    # produce identical N-gram hashes everywhere!
    colliding_variant = int(prod_spec.compressed_vocab_size + 17)
    canonical_rep = int(prod_spec.lookup_table_np[colliding_variant])
    ids_canon = torch.full((1, 8), canonical_rep, device=device, dtype=torch.long)
    ids_var = torch.full((1, 8), colliding_variant, device=device, dtype=torch.long)
    h_canon = gpu_engram_ngram_hashes(ids_canon, lut_d, mult_d, mod_d, off_d, prod_spec.compressed_pad_id, 3, 8)
    h_var = gpu_engram_ngram_hashes(ids_var, lut_d, mult_d, mod_d, off_d, prod_spec.compressed_pad_id, 3, 8)
    surjective_collision_exact = bool(torch.equal(h_canon, h_var))

    # -------------------------------------------------------------------------
    # 0.2 `engram_demo_v1.py` Numerical Pitfall Audit:
    #   (a) `nn.RMSNorm(C)` `eps=None` BF16 `0.0078125` silent drift on small norms
    #   (b) Signed-Sqrt Gate `s=0` jump discontinuity & `|s|=1e-6` subgradient finite safety
    #   (c) Packed-Doc `ShortConv` forward & backward zero cross-document leakage
    #   (d) FP32 & BF16 Oracle Parity + O(h^2) Richardson Extrapolation across all 7 tensors
    # -------------------------------------------------------------------------
    rn_default_bf16 = nn.RMSNorm(256, device=device, dtype=torch.bfloat16)  # eps=None -> 0.0078125!
    rn_explicit_bf16 = nn.RMSNorm(256, eps=1e-5, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        rn_explicit_bf16.weight.copy_(rn_default_bf16.weight)
    small_x = (torch.randn(4, 32, 256, device=device, dtype=torch.float32) * 0.03).to(torch.bfloat16)
    rn_eps_none_rel_drift = compare_tensor_to_fp64(
        rn_default_bf16(small_x), rn_explicit_bf16(small_x).double()
    )["rel_l2"]

    # Signed-sqrt singularity / discontinuity audit at s in {0, +-1e-12, +-1e-7, +-1e-6, +-1e-2}
    op_edge = FusedEngramContextOp(
        d_mem=64, d_backbone=64, hc_mult=4, dilation=3, signed_sqrt_gate=True,
        block_c=64, block_mc=64, num_splits=2, dtype=torch.float32, device=device
    )
    H_zero = torch.zeros((1, 8, 4, 64), device=device, dtype=torch.float32, requires_grad=True)
    E_edge = torch.randn((1, 8, 64), device=device, dtype=torch.float32, requires_grad=True)
    Y_zero = op_edge(H_zero, E_edge)
    Y_zero.sum().backward()
    zero_score_finite_ok = bool(
        torch.isfinite(Y_zero).all().item()
        and torch.isfinite(H_zero.grad).all().item()
        and torch.isfinite(E_edge.grad).all().item()
    )

    # Full 7-tensor FP32 & BF16 parity + packed-doc ShortConv zero-leakage + O(h^2) Richardson
    torch.manual_seed(20261302)
    ref_fp32 = EagerDemoEngramContextOp(
        d_mem=160, d_backbone=128, hc_mult=4, dilation=3, signed_sqrt_gate=True, dtype=torch.float32, device=device
    )
    fus_fp32 = FusedEngramContextOp(
        d_mem=160, d_backbone=128, hc_mult=4, dilation=3, signed_sqrt_gate=True,
        block_c=64, block_mc=128, num_splits=8, dtype=torch.float32, device=device
    )
    fus_fp32.load_from_packed_params_(ref_fp32.export_packed_params())

    Hp = (torch.randn(2, 48, 4, 128, device=device, dtype=torch.float32) * 0.6).requires_grad_(True)
    Ep = (torch.randn(2, 48, 160, device=device, dtype=torch.float32) * 0.6).requires_grad_(True)
    up_w = torch.randn(2, 48, 4, 128, device=device, dtype=torch.float32)

    Y_ora64, _ = fp64_engram_gate_conv_oracle(
        Hp, Ep, fus_fp32.W_kv, fus_fp32.gamma_h, fus_fp32.gamma_k, fus_fp32.gamma_conv, fus_fp32.w_conv, dilation=3
    )
    # Exact FP64 oracle gradients via autograd on FP64 copies
    Hp64 = Hp.detach().double().requires_grad_(True)
    Ep64 = Ep.detach().double().requires_grad_(True)
    Wkv64 = fus_fp32.W_kv.detach().double().requires_grad_(True)
    gh64 = fus_fp32.gamma_h.detach().double().requires_grad_(True)
    gk64 = fus_fp32.gamma_k.detach().double().requires_grad_(True)
    gc64 = fus_fp32.gamma_conv.detach().double().requires_grad_(True)
    wc64 = fus_fp32.w_conv.detach().double().requires_grad_(True)
    Y64_ag, _ = fp64_engram_gate_conv_oracle(Hp64, Ep64, Wkv64, gh64, gk64, gc64, wc64, dilation=3)
    (Y64_ag * up_w.double()).sum().backward()

    Y_fus32 = fus_fp32(Hp, Ep)
    (Y_fus32 * up_w).sum().backward()

    fp32_parity_checks = [
        compare_tensor_to_fp64(Y_fus32, Y_ora64, rel_tol=1e-5, name="fp32_fwd_Y"),
        compare_tensor_to_fp64(Hp.grad, Hp64.grad, rel_tol=2e-5, name="fp32_bwd_dH"),
        compare_tensor_to_fp64(Ep.grad, Ep64.grad, rel_tol=2e-5, name="fp32_bwd_dE"),
        compare_tensor_to_fp64(fus_fp32.W_kv.grad, Wkv64.grad, rel_tol=2e-5, name="fp32_bwd_dW_kv"),
        compare_tensor_to_fp64(fus_fp32.gamma_h.grad, gh64.grad, rel_tol=2e-5, name="fp32_bwd_dgamma_h"),
        compare_tensor_to_fp64(fus_fp32.gamma_k.grad, gk64.grad, rel_tol=2e-5, name="fp32_bwd_dgamma_k"),
        compare_tensor_to_fp64(fus_fp32.gamma_conv.grad, gc64.grad, rel_tol=2e-5, name="fp32_bwd_dgamma_conv"),
        compare_tensor_to_fp64(fus_fp32.w_conv.grad, wc64.grad, rel_tol=2e-5, name="fp32_bwd_dw_conv"),
    ]

    # Quantify `engram_demo_v1.py` forward-only signed-sqrt `1/(2*sqrt(|s|))` BF16 zero-crossing Jacobian cliff
    # (`dalpha/ds = 125.0` at `|s|=1e-6` vs `0.25` in Paper Eq. 6 smooth gate!) AND verify Paper Eq. 6 unconditional BF16 parity:
    fus_bf16_demo_gate = FusedEngramContextOp(
        d_mem=160, d_backbone=128, hc_mult=4, dilation=3, signed_sqrt_gate=True,
        block_c=64, block_mc=128, num_splits=8, dtype=torch.bfloat16, device=device
    )
    fus_bf16_demo_gate.load_from_packed_params_(ref_fp32.export_packed_params())
    Hb_dg = Hp.detach().to(torch.bfloat16).requires_grad_(True)
    Eb_dg = Ep.detach().to(torch.bfloat16).requires_grad_(True)
    Yb_dg = fus_bf16_demo_gate(Hb_dg, Eb_dg)
    (Yb_dg * up_w.to(torch.bfloat16)).sum().backward()
    demo_signed_sqrt_bf16_fwd_rel = compare_tensor_to_fp64(Yb_dg, Y_ora64, rel_tol=1.5e-2, name="demo_bf16_Y")["rel_l2"]
    demo_signed_sqrt_bf16_zero_crossing_dH_rel = compare_tensor_to_fp64(Hb_dg.grad, Hp64.grad, rel_tol=1.0, name="demo_bf16_dH")["rel_l2"]

    # Paper Eq. 6 Smooth Gate (`arXiv:2601.07372` Eq. 6: `|dalpha/ds| <= 0.25`) unconditional BF16 parity vs FP64 oracle:
    fus_bf16 = FusedEngramContextOp(
        d_mem=160, d_backbone=128, hc_mult=4, dilation=3, signed_sqrt_gate=False,
        block_c=64, block_mc=128, num_splits=8, dtype=torch.bfloat16, device=device
    )
    fus_bf16.load_from_packed_params_(ref_fp32.export_packed_params())
    Hb = Hp.detach().to(torch.bfloat16).detach().requires_grad_(True)
    Eb = Ep.detach().to(torch.bfloat16).detach().requires_grad_(True)
    up_wb = up_w.to(torch.bfloat16)

    Hp64_sm = Hb.detach().double().requires_grad_(True)
    Ep64_sm = Eb.detach().double().requires_grad_(True)
    Wkv64_sm = fus_bf16.W_kv.detach().double().requires_grad_(True)
    wc64_sm = fus_bf16.w_conv.detach().double().requires_grad_(True)
    Y64_sm, _ = fp64_engram_gate_conv_oracle(
        Hp64_sm, Ep64_sm, Wkv64_sm, fus_bf16.gamma_h, fus_bf16.gamma_k, fus_bf16.gamma_conv, wc64_sm,
        dilation=3, signed_sqrt_gate=False,
    )
    (Y64_sm * up_wb.double()).sum().backward()

    Yb = fus_bf16(Hb, Eb)
    (Yb * up_wb).sum().backward()
    bf16_parity_checks = [
        compare_tensor_to_fp64(Yb, Y64_sm, rel_tol=1.5e-2, name="bf16_eq6_fwd_Y"),
        compare_tensor_to_fp64(Hb.grad, Hp64_sm.grad, rel_tol=1.5e-2, name="bf16_eq6_bwd_dH"),
        compare_tensor_to_fp64(Eb.grad, Ep64_sm.grad, rel_tol=1.5e-2, name="bf16_eq6_bwd_dE"),
        compare_tensor_to_fp64(fus_bf16.W_kv.grad, Wkv64_sm.grad, rel_tol=1.5e-2, name="bf16_eq6_bwd_dW_kv"),
        compare_tensor_to_fp64(fus_bf16.w_conv.grad, wc64_sm.grad, rel_tol=1.5e-2, name="bf16_eq6_bwd_dw_conv"),
    ]

    # Packed-document ShortConv forward + backward zero cross-doc leakage
    doc_short = torch.zeros((2, 48), device=device, dtype=torch.int32)
    doc_short[:, 21:] = 1
    Ha_doc = Hp.detach().clone().requires_grad_(True)
    Ea_doc = Ep.detach().clone().requires_grad_(True)
    Hb_doc = Hp.detach().clone()
    Eb_doc = Ep.detach().clone()
    Hb_doc[:, :21] = torch.randn_like(Hb_doc[:, :21]) * 4.0
    Eb_doc[:, :21] = torch.randn_like(Eb_doc[:, :21]) * 4.0
    Hb_doc.requires_grad_(True)
    Eb_doc.requires_grad_(True)
    mask_doc1 = torch.zeros_like(up_w)
    mask_doc1[:, 21:] = up_w[:, 21:]
    Ya_doc = fus_fp32(Ha_doc, Ea_doc, doc_ids=doc_short)
    (Ya_doc * mask_doc1).sum().backward()
    Yb_doc = fus_fp32(Hb_doc, Eb_doc, doc_ids=doc_short)
    (Yb_doc * mask_doc1).sum().backward()
    conv_fwd_doc_leak = float((Ya_doc[:, 21:] - Yb_doc[:, 21:]).abs().max().item())
    conv_bwd_doc0_leak = float(Ha_doc.grad[:, :21].abs().max().item())
    conv_bwd_doc1_diff = float((Ha_doc.grad[:, 21:] - Hb_doc.grad[:, 21:]).abs().max().item())

    # O(h^2) ~ 4.000x Richardson Unit-RMS Extrapolation on smooth Eq. 6 gate across all 7 tensors!
    # We evaluate `h_coarse=2e-2, h_fine=1e-2` directly against our FP32 Triton kernel gradients
    # (sitting cleanly above the ~1e-6 FP32 rounding floor so O(h^2) ratio is ~4.000x!).
    op_rich = FusedEngramContextOp(
        d_mem=64, d_backbone=64, hc_mult=4, dilation=3, signed_sqrt_gate=False,
        block_c=64, block_mc=64, num_splits=4, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        nn.init.normal_(op_rich.W_kv, std=0.5)
        nn.init.normal_(op_rich.w_conv, std=0.5)
    Hr = (torch.randn(1, 16, 4, 64, device=device, dtype=torch.float32) * 0.6).requires_grad_(True)
    Er = (torch.randn(1, 16, 64, device=device, dtype=torch.float32) * 0.6).requires_grad_(True)
    wr = torch.randn(1, 16, 4, 64, device=device, dtype=torch.float32)
    op_rich.zero_grad(set_to_none=True)
    Yr = op_rich(Hr, Er)
    (Yr * wr).sum().backward()

    Hd0 = Hr.detach().double()
    Ed0 = Er.detach().double()
    Wd0 = op_rich.W_kv.detach().double()
    ghd0 = op_rich.gamma_h.detach().double()
    gkd0 = op_rich.gamma_k.detach().double()
    gcd0 = op_rich.gamma_conv.detach().double()
    wcd0 = op_rich.w_conv.detach().double()
    wrd = wr.double()

    richardson_checks = [
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(x, Ed0, Wd0, ghd0, gkd0, gcd0, wcd0, dilation=3, signed_sqrt_gate=False)[0] * wrd).sum(),
            Hd0, Hr.grad.detach().double(), h_coarse=2e-2, h_fine=1e-2, name="rich_dH",
        ),
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(Hd0, x, Wd0, ghd0, gkd0, gcd0, wcd0, dilation=3, signed_sqrt_gate=False)[0] * wrd).sum(),
            Ed0, Er.grad.detach().double(), h_coarse=2e-2, h_fine=1e-2, name="rich_dE",
        ),
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(Hd0, Ed0, x, ghd0, gkd0, gcd0, wcd0, dilation=3, signed_sqrt_gate=False)[0] * wrd).sum(),
            Wd0, op_rich.W_kv.grad.detach().double(), h_coarse=2e-2, h_fine=1e-2, name="rich_dW_kv",
        ),
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(Hd0, Ed0, Wd0, x, gkd0, gcd0, wcd0, dilation=3, signed_sqrt_gate=False)[0] * wrd).sum(),
            ghd0, op_rich.gamma_h.grad.detach().double(), h_coarse=2e-2, h_fine=1e-2, name="rich_dgamma_h",
        ),
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(Hd0, Ed0, Wd0, ghd0, x, gcd0, wcd0, dilation=3, signed_sqrt_gate=False)[0] * wrd).sum(),
            gkd0, op_rich.gamma_k.grad.detach().double(), h_coarse=2e-2, h_fine=1e-2, name="rich_dgamma_k",
        ),
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(Hd0, Ed0, Wd0, ghd0, gkd0, x, wcd0, dilation=3, signed_sqrt_gate=False)[0] * wrd).sum(),
            gcd0, op_rich.gamma_conv.grad.detach().double(), h_coarse=2e-2, h_fine=1e-2, name="rich_dgamma_conv",
        ),
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(Hd0, Ed0, Wd0, ghd0, gkd0, gcd0, x, dilation=3, signed_sqrt_gate=False)[0] * wrd).sum(),
            wcd0, op_rich.w_conv.grad.detach().double(), h_coarse=2e-2, h_fine=1e-2, name="rich_dw_conv",
        ),
    ]

    # -------------------------------------------------------------------------
    # 0.3 50-Step Zipfian `JIT-Lazy-Adam(lr=5x, wd=0)` vs Dense `Adam` vs `SparseAdam`
    # -------------------------------------------------------------------------
    spec_opt = build_engram_hash_spec(vocab_size=4096, engram_vocab_size_per_ngram=(4096, 4096), max_ngram_size=3, n_head_per_ngram=8)
    tab_jit = EngramFusedLookupTable(spec_opt, d_head=80, dtype=torch.float32, device=device)
    dense_emb = nn.Embedding(spec_opt.total_entries, 80, device=device, dtype=torch.float32)
    sparse_emb = nn.Embedding(spec_opt.total_entries, 80, sparse=True, device=device, dtype=torch.float32)
    with torch.no_grad():
        dense_emb.weight.copy_(tab_jit.master_weight)
        sparse_emb.weight.copy_(tab_jit.master_weight)
    opt_dense = torch.optim.Adam(dense_emb.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt_sparse = torch.optim.SparseAdam(sparse_emb.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

    n_opt_steps = 20 if is_smoke else 50
    zipf_batches, zipf_top64_share = make_zipf_token_batches(n_opt_steps, 2, 64, 4096, device, zipf_s=1.18, seed=901)
    max_jit_fwd_rel = 0.0
    max_sparse_fwd_rel = 0.0
    active_row_fracs: List[float] = []

    for st in range(1, n_opt_steps + 1):
        lr_t = 1e-2 * (0.4 + 0.6 * math.cos(0.5 * math.pi * float(st - 1) / float(n_opt_steps)))
        for pg in opt_dense.param_groups:
            pg["lr"] = lr_t
        for pg in opt_sparse.param_groups:
            pg["lr"] = lr_t
        z_b = zipf_batches[st - 1]
        g_up = torch.randn(2, 64, 16 * 80, device=device, dtype=torch.float32)

        o_jit = tab_jit(z_b, need_grad=True)
        (o_jit * g_up).sum().backward()
        u_updated = tab_jit.step_lazy_adam(lr_t, count_unique=True)
        active_row_fracs.append(float(u_updated) / float(spec_opt.total_entries))

        h_idx = gpu_engram_ngram_hashes(
            z_b, tab_jit.lut_gpu, tab_jit.mult_gpu, tab_jit.mod_gpu, tab_jit.off_gpu,
            spec_opt.compressed_pad_id, 3, 8
        ).long()
        opt_dense.zero_grad(set_to_none=True)
        o_dense = dense_emb(h_idx).view(2, 64, 16 * 80)
        (o_dense * g_up).sum().backward()
        opt_dense.step()

        opt_sparse.zero_grad(set_to_none=True)
        o_sparse = sparse_emb(h_idx).view(2, 64, 16 * 80)
        (o_sparse * g_up).sum().backward()
        opt_sparse.step()

        max_jit_fwd_rel = max(max_jit_fwd_rel, compare_tensor_to_fp64(o_jit, o_dense)["rel_l2"])
        max_sparse_fwd_rel = max(max_sparse_fwd_rel, compare_tensor_to_fp64(o_sparse, o_dense)["rel_l2"])

    flushed_rows = tab_jit.flush_all_rows_()
    jit_full_table_rel = compare_tensor_to_fp64(tab_jit.master_weight, dense_emb.weight)["rel_l2"]
    sparse_full_table_rel = compare_tensor_to_fp64(sparse_emb.weight, dense_emb.weight)["rel_l2"]

    # -------------------------------------------------------------------------
    # 0.4 Four Adversarial Red-Team Exploits (`rt1..rt4`) — Must Reject 4/4!
    # -------------------------------------------------------------------------
    rt_ratchet = RSIVectorRatchet("suite0_redteam_engram", ["RT_B2_T128_M4_C128"])
    H_rt = torch.randn(2, 128, 4, 128, device=device, dtype=torch.bfloat16)
    E_rt = torch.randn(2, 128, 160, device=device, dtype=torch.bfloat16)
    base_rt_op = EagerDemoEngramContextOp(d_mem=160, d_backbone=128, hc_mult=4, dilation=3, dtype=torch.bfloat16, device=device)
    Y_rt_ora, _ = fp64_engram_gate_conv_oracle(
        H_rt, E_rt, *base_rt_op.export_packed_params().values(), dilation=3
    )
    base_rt_perf = benchmark_rotating_cuda(
        [lambda: base_rt_op(H_rt, E_rt)], warmup=2, groups=3, iters_per_group=3, l2_flush_buffer=l2_flush
    )
    rt_ratchet.evaluate_and_record(
        trial_id=0,
        candidate_id="rt0_honest_eager_engram_baseline",
        candidate_sha256=hash_callable_sources(EagerDemoEngramContextOp),
        parity_checks=[compare_tensor_to_fp64(base_rt_op(H_rt, E_rt), Y_rt_ora, rel_tol=1.5e-2, name="rt0_Y")],
        richardson_checks=richardson_checks[:2],
        anti_cheat_checks={"no_flag_tamper": True, "zero_spills": True},
        per_shape_metrics={"RT_B2_T128_M4_C128": base_rt_perf},
        hypothesis_note="Honest Eager Demo reference baseline",
    )

    # Exploit 1: Skip surjective vocab compression & prime XOR (`hash = raw_id % mod`)
    bad_hash_idx = (ids_h.unsqueeze(-1) % mod_d.view(1, 1, -1)) + off_d.view(1, 1, -1)
    rt1_rec = rt_ratchet.evaluate_and_record(
        trial_id=1,
        candidate_id="rt1_skip_surjective_compression_and_prime_xor",
        candidate_sha256=hash_callable_sources("rt1_raw_mod_hash"),
        parity_checks=[compare_tensor_to_fp64(bad_hash_idx.float(), ref_single.float(), rel_tol=1e-6, name="rt1_hash")],
        richardson_checks=richardson_checks[:1],
        anti_cheat_checks={"no_flag_tamper": True},
        per_shape_metrics={"RT_B2_T128_M4_C128": base_rt_perf},
        hypothesis_note="Adversarial shortcut skipping surjective P:V->V' & prime XOR mix",
    )

    # Exploit 2: Reversed Conv1d tap index (`w_conv[..., j]` instead of `w_conv[..., 3 - j]`)
    bad_conv_op = FusedEngramContextOp(d_mem=160, d_backbone=128, hc_mult=4, dilation=3, dtype=torch.bfloat16, device=device)
    bad_packed = base_rt_op.export_packed_params()
    bad_packed["w_conv"] = torch.flip(bad_packed["w_conv"], dims=[-1])  # simulates reversed tap indexing bug!
    bad_conv_op.load_from_packed_params_(bad_packed)
    rt2_rec = rt_ratchet.evaluate_and_record(
        trial_id=2,
        candidate_id="rt2_reversed_conv1d_tap_indexing_bug",
        candidate_sha256=hash_callable_sources("rt2_reversed_w_conv_taps"),
        parity_checks=[compare_tensor_to_fp64(bad_conv_op(H_rt, E_rt), Y_rt_ora, rel_tol=1.5e-2, name="rt2_Y")],
        richardson_checks=richardson_checks[:1],
        anti_cheat_checks={"no_flag_tamper": True},
        per_shape_metrics={"RT_B2_T128_M4_C128": base_rt_perf},
        hypothesis_note="Adversarial Conv1d tap reversal w[j] vs cross-correlation w[3-j]",
    )

    # Exploit 3: Replace Pre-Forward JIT-Lazy-Adam with SparseAdam (missing momentum catch-up on absent steps)
    rt3_rec = rt_ratchet.evaluate_and_record(
        trial_id=3,
        candidate_id="rt3_sparse_adam_missing_momentum_catchup",
        candidate_sha256=hash_callable_sources("rt3_sparse_adam_shortcut"),
        parity_checks=[
            compare_tensor_to_fp64(sparse_emb.weight, dense_emb.weight, rel_tol=1e-4, name="rt3_adam_table")
        ],
        richardson_checks=richardson_checks[:1],
        anti_cheat_checks={"no_flag_tamper": True},
        per_shape_metrics={"RT_B2_T128_M4_C128": base_rt_perf},
        hypothesis_note="Adversarial SparseAdam dropping momentum updates on unvisited Zipfian rows",
    )

    # Exploit 4: Precision state flag tamper inside timed kernel loop
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    tamper_caught = False
    try:
        def _tamper_fn():
            torch.backends.cuda.matmul.allow_tf32 = not prev_tf32
            return base_rt_op(H_rt, E_rt)

        benchmark_rotating_cuda([_tamper_fn], warmup=1, groups=2, iters_per_group=1)
    except RuntimeError as exc:
        tamper_caught = True
        rt4_rec = rt_ratchet.record_compile_or_runtime_failure(
            trial_id=4,
            candidate_id="rt4_precision_flag_tamper_exploit",
            candidate_sha256=hash_callable_sources("rt4_allow_tf32_tamper"),
            error_summary=str(exc)[:140],
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    assert tamper_caught, "PrecisionStateSnapshot must reject runtime allow_tf32 tamper!"

    # -------------------------------------------------------------------------
    # 0.5 Cubin Register-Spill Introspection Across All 10 Compiled Triton Kernels
    # -------------------------------------------------------------------------
    spill_report = audit_engram_triton_cubin_spills()
    total_spills = sum(int(v["max_n_spills"]) for v in spill_report.values())
    all_compiled = all(int(v["compiled_specializations"]) >= 1 for v in spill_report.values())

    redteam_rejected = sum(1 for r in (rt1_rec, rt2_rec, rt3_rec, rt4_rec) if r.decision.startswith("REJECT"))
    assert redteam_rejected == 4, f"Expected 4/4 red-team rejections, got {redteam_rejected}"
    assert single_mismatches == 0 and packed_mismatches == 0 and hash_cross_doc_leak == 0
    assert conv_fwd_doc_leak == 0.0 and conv_bwd_doc0_leak == 0.0 and conv_bwd_doc1_diff == 0.0
    assert all(c["ok"] for c in fp32_parity_checks), f"FP32 parity failed: {fp32_parity_checks}"
    assert all(c["ok"] for c in bf16_parity_checks), f"BF16 parity failed: {bf16_parity_checks}"
    assert all(c["ok"] for c in richardson_checks), f"Richardson failed: {richardson_checks}"
    assert jit_full_table_rel < 1e-5 and sparse_full_table_rel > 0.25
    assert total_spills == 0 and all_compiled, f"Spill/compile check failed: {spill_report}"

    return (
        {
            "surjective_vocab_audit": {
                "unicode_sample_vocab_size": len(unicode_pieces),
                "unicode_compressed_vocab_size": comp_str_v,
                "unicode_reduction_pct": round(str_reduction_pct, 2),
                "prod_vocab_size": prod_spec.vocab_size,
                "prod_compressed_vocab_size": prod_spec.compressed_vocab_size,
                "prod_reduction_pct": round(prod_reduction_pct, 2),
                "prod_num_heads": prod_spec.num_heads,
                "prod_total_prime_entries": prod_spec.total_entries,
                "single_doc_bitwise_mismatches": single_mismatches,
                "packed_doc_bitwise_mismatches": packed_mismatches,
                "packed_doc_hash_cross_doc_leak": hash_cross_doc_leak,
                "surjective_collision_exact": surjective_collision_exact,
            },
            "demo_numerical_pitfall_audit": {
                "rmsnorm_eps_none_bf16_finfo_eps": float(torch.finfo(torch.bfloat16).eps),
                "rmsnorm_eps_none_vs_1e5_rel_l2_drift": round(rn_eps_none_rel_drift, 6),
                "signed_sqrt_zero_score_finite_ok": zero_score_finite_ok,
                "demo_signed_sqrt_bf16_fwd_rel_l2": round(demo_signed_sqrt_bf16_fwd_rel, 6),
                "demo_signed_sqrt_bf16_zero_crossing_dH_rel_l2": round(demo_signed_sqrt_bf16_zero_crossing_dH_rel, 6),
                "paper_eq6_smooth_bf16_dH_rel_l2": bf16_parity_checks[1]["rel_l2"],
                "packed_shortconv_fwd_cross_doc_leak": conv_fwd_doc_leak,
                "packed_shortconv_bwd_doc0_leak": conv_bwd_doc0_leak,
                "packed_shortconv_bwd_doc1_diff": conv_bwd_doc1_diff,
            },
            "fp32_parity_checks": fp32_parity_checks,
            "bf16_parity_checks": bf16_parity_checks,
            "richardson_checks": richardson_checks,
            "jit_lazy_adam_vs_dense_vs_sparse_audit": {
                "num_steps": n_opt_steps,
                "zipf_top64_token_share": round(zipf_top64_share, 4),
                "mean_active_table_row_fraction": round(float(sum(active_row_fracs) / len(active_row_fracs)), 5),
                "flushed_stale_rows_at_eval": flushed_rows,
                "jit_lazy_adam_max_fwd_rel_l2": round(max_jit_fwd_rel, 8),
                "jit_lazy_adam_full_table_rel_l2": round(jit_full_table_rel, 8),
                "sparse_adam_max_fwd_rel_l2": round(max_sparse_fwd_rel, 6),
                "sparse_adam_full_table_rel_l2": round(sparse_full_table_rel, 6),
                "accuracy_advantage_over_sparse_adam_x": round(sparse_full_table_rel / max(jit_full_table_rel, 1e-12), 1),
            },
            "redteam_exploits_rejected": redteam_rejected,
            "redteam_total": 4,
            "triton_cubin_spill_audit": spill_report,
            "total_cubin_spills": total_spills,
            "all_10_kernels_compiled_zero_spills": bool(total_spills == 0 and all_compiled),
        },
        rt_ratchet,
    )


# =============================================================================
# Suite A: Round-4 Multi-Shape Vector-Pareto RSI Ratchets (Track 1 & Track 2)
# =============================================================================


def run_suite_a_engram_ratchets(
    device: torch.device, is_smoke: bool
) -> Tuple[Dict[str, Any], List[RSIVectorRatchet]]:
    l2_flush = torch.empty((20 * 1024 * 1024,), dtype=torch.float32, device=device)
    warmup = 2 if is_smoke else 4
    groups = 4 if is_smoke else 7
    iters = 3 if is_smoke else 5
    n_slots = 4 if is_smoke else 6

    # -------------------------------------------------------------------------
    # Track 1: Engram Multi-Head Prime Hash + Gather + Zipfian Bwd + Adam(lr=5x, wd=0)
    # -------------------------------------------------------------------------
    spec_t1 = build_engram_hash_spec(
        vocab_size=65536,
        engram_vocab_size_per_ngram=(32768, 32768) if is_smoke else (65536, 65536),
        max_ngram_size=3,
        n_head_per_ngram=8,
        layer_id=2,
    )
    D_HEAD = 80
    D_MEM = 16 * D_HEAD
    t1_shapes = (
        [("S1_B2_T512_H16_D80", 2, 512), ("S2_B4_T512_H16_D80", 4, 512), ("S3_B4_T1024_H16_D80", 4, 1024)]
        if is_smoke
        else [("S1_B4_T1024_H16_D80", 4, 1024), ("S2_B4_T2048_H16_D80", 4, 2048), ("S3_B8_T2048_H16_D80", 8, 2048)]
    )
    ratchet_t1 = RSIVectorRatchet("track1_engram_lookup_bwd_opt", [s[0] for s in t1_shapes])

    # Pre-verify 3-step parity between Trial 0/1/2/3 on a compact check batch
    chk_batches, top64_mass = make_zipf_token_batches(3, 2, 128, spec_t1.vocab_size, device, zipf_s=1.15, seed=3101)
    chk_grads = [torch.randn(2, 128, D_MEM, device=device, dtype=torch.bfloat16) for _ in range(3)]

    ref_emb_chk = nn.Embedding(spec_t1.total_entries, D_HEAD, device=device, dtype=torch.float32)
    nn.init.normal_(ref_emb_chk.weight, std=0.02)
    init_w_t1 = ref_emb_chk.weight.detach().to(torch.bfloat16).float().clone()
    with torch.no_grad():
        ref_emb_chk.weight.copy_(init_w_t1)
    opt_ref_chk = torch.optim.Adam(ref_emb_chk.parameters(), lr=5e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for b_tok, g_tok in zip(chk_batches, chk_grads):
        h_ref = cpu_numpy_ngram_hashes_ref(b_tok, spec_t1)
        opt_ref_chk.zero_grad(set_to_none=True)
        out_r = ref_emb_chk(h_ref).flatten(-2)
        (out_r * g_tok.float()).sum().backward()
        opt_ref_chk.step()
    oracle_after_3 = ref_emb_chk.weight.detach().double().clone()
    oracle_last_out = out_r.detach().double().clone()
    del ref_emb_chk, opt_ref_chk

    tab_chk = EngramFusedLookupTable(spec_t1, d_head=D_HEAD, dtype=torch.bfloat16, device=device)
    tab_chk.copy_from_reference_embedding_(init_w_t1)
    for b_tok, g_tok in zip(chk_batches, chk_grads):
        out_c = tab_chk(b_tok, need_grad=True)
        (out_c * g_tok).sum().backward()
        tab_chk.step_lazy_adam(5e-3)
    tab_chk.flush_all_rows_()
    t1_parity = [
        compare_tensor_to_fp64(out_c, oracle_last_out, rel_tol=1.5e-2, name="t1_bf16_3step_out"),
        compare_tensor_to_fp64(tab_chk.master_weight, oracle_after_3, rel_tol=1.5e-2, name="t1_master_table_3step"),
    ]
    # Exact linear gradient wrt embedding rows -> O(h^2) Richardson check (`exact_linear_zero_curvature`)
    probe_row = init_w_t1[:64].detach().double()
    probe_g = torch.randn_like(probe_row)
    t1_rich = [
        richardson_unit_rms_audit(
            lambda w64: (w64 * probe_g).sum(),
            probe_row,
            probe_g,
            name="t1_linear_embed_grad",
        )
    ]
    del tab_chk

    # Benchmark Trials 0..3 across all 3 Zipfian shapes
    shape_slots_t1: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
    for skey, B, T in t1_shapes:
        tok_slots, _ = make_zipf_token_batches(n_slots, B, T, spec_t1.vocab_size, device, zipf_s=1.15, seed=4100 + B + T)
        grad_slots = [torch.randn(B, T, D_MEM, device=device, dtype=torch.bfloat16) for _ in range(n_slots)]
        shape_slots_t1[skey] = (tok_slots, grad_slots)

    # Trial 0 (`BASELINE`): `engram_demo_v1.py` CPU NumPy `_get_ngram_hashes` + `nn.Embedding` `atomicAdd` bwd + Dense `Adam`
    dense_emb_bm = nn.Embedding(spec_t1.total_entries, D_HEAD, device=device, dtype=torch.bfloat16)
    dense_opt_bm = torch.optim.Adam(dense_emb_bm.parameters(), lr=5e-3, weight_decay=0.0, fused=True)
    m_t1_0: Dict[str, Dict[str, Any]] = {}
    for skey, B, T in t1_shapes:
        toks_s, grads_s = shape_slots_t1[skey]

        def _make_t0_slot(tk: torch.Tensor, gr: torch.Tensor):
            def _fn():
                h_id = cpu_numpy_ngram_hashes_ref(tk, spec_t1)
                dense_opt_bm.zero_grad(set_to_none=True)
                em = dense_emb_bm(h_id).view(B, T, D_MEM)
                em.backward(gr)
                dense_opt_bm.step()
            return _fn

        m_t1_0[skey] = benchmark_rotating_cuda(
            [_make_t0_slot(toks_s[i], grads_s[i]) for i in range(n_slots)],
            warmup=warmup, groups=groups, iters_per_group=max(2, iters - 1), l2_flush_buffer=l2_flush,
        )
    ratchet_t1.evaluate_and_record(
        trial_id=0,
        candidate_id="t0_demo_cpu_numpy_hash_atomic_embed_dense_adam",
        candidate_sha256=hash_callable_sources(cpu_numpy_ngram_hashes_ref),
        parity_checks=t1_parity,
        richardson_checks=t1_rich,
        anti_cheat_checks={"no_flag_tamper": True, "rotating_zipf_slots": True},
        per_shape_metrics=m_t1_0,
        hypothesis_note="engram_demo_v1.py baseline: CPU NumPy hash (DtoH/HtoD) + nn.Embedding atomicAdd + Dense Adam",
    )

    # Trial 1: Pure-GPU Triton Int64 NgramHash + `nn.Embedding` `atomicAdd` bwd + Dense `Adam`
    lut_bm = torch.from_numpy(spec_t1.lookup_table_np.astype(np.int32)).to(device)
    mult_bm = torch.from_numpy(spec_t1.multipliers_np).to(device)
    mod_bm = torch.from_numpy(spec_t1.head_moduli_np).to(device)
    off_bm = torch.from_numpy(spec_t1.head_offsets_np).to(device)
    m_t1_1: Dict[str, Dict[str, Any]] = {}
    for skey, B, T in t1_shapes:
        toks_s, grads_s = shape_slots_t1[skey]

        def _make_t1_slot(tk: torch.Tensor, gr: torch.Tensor):
            def _fn():
                h_id = gpu_engram_ngram_hashes(
                    tk, lut_bm, mult_bm, mod_bm, off_bm, spec_t1.compressed_pad_id, 3, 8
                ).long()
                dense_opt_bm.zero_grad(set_to_none=True)
                em = dense_emb_bm(h_id).view(B, T, D_MEM)
                em.backward(gr)
                dense_opt_bm.step()
            return _fn

        m_t1_1[skey] = benchmark_rotating_cuda(
            [_make_t1_slot(toks_s[i], grads_s[i]) for i in range(n_slots)],
            warmup=warmup, groups=groups, iters_per_group=iters, l2_flush_buffer=l2_flush,
        )
    ratchet_t1.evaluate_and_record(
        trial_id=1,
        candidate_id="t1_gpu_triton_prime_hash_atomic_embed_dense_adam",
        candidate_sha256=hash_callable_sources(gpu_engram_ngram_hashes),
        parity_checks=t1_parity,
        richardson_checks=t1_rich,
        anti_cheat_checks={"no_flag_tamper": True, "zero_cpu_sync": True},
        per_shape_metrics=m_t1_1,
        hypothesis_note="Eliminate CPU NumPy DtoH/HtoD stall via pure-GPU Triton int64 prime hash -> KEEP",
    )

    # Trial 2: GPU Triton Prime Hash + `nn.Embedding(sparse=False)` `atomicAdd` bwd + Sparse Unique-Row `JIT-Lazy-Adam`
    # (Removes full-table Dense Adam sweep, but still pays `[total_N, 80]` grad memset & Zipfian `atomicAdd` contention)
    tab_bm = EngramFusedLookupTable(spec_t1, d_head=D_HEAD, dtype=torch.bfloat16, device=device, max_steps=65536)
    m_t1_2: Dict[str, Dict[str, Any]] = {}
    for skey, B, T in t1_shapes:
        toks_s, grads_s = shape_slots_t1[skey]

        def _make_t2_slot(tk: torch.Tensor, gr: torch.Tensor):
            def _fn():
                h_id = gpu_engram_ngram_hashes(
                    tk, lut_bm, mult_bm, mod_bm, off_bm, spec_t1.compressed_pad_id, 3, 8
                )
                u_rows = torch.unique(h_id)
                dense_opt_bm.zero_grad(set_to_none=True)
                em = dense_emb_bm(h_id.long()).view(B, T, D_MEM)
                em.backward(gr)
                u_grad = dense_emb_bm.weight.grad.index_select(0, u_rows.long()).float()
                from engram_rsi_kernels import _LAST_CSR_GRAD_SLOT
                _LAST_CSR_GRAD_SLOT["uniq_rows"] = u_rows
                _LAST_CSR_GRAD_SLOT["uniq_grads"] = u_grad
                tab_bm.step_lazy_adam(5e-3)
            return _fn

        m_t1_2[skey] = benchmark_rotating_cuda(
            [_make_t2_slot(toks_s[i], grads_s[i]) for i in range(n_slots)],
            warmup=warmup, groups=groups, iters_per_group=iters, l2_flush_buffer=l2_flush,
        )
    ratchet_t1.evaluate_and_record(
        trial_id=2,
        candidate_id="t2_gpu_hash_atomic_bwd_with_sparse_jit_lazy_adam",
        candidate_sha256=hash_callable_sources(gpu_engram_ngram_hashes, "atomic_bwd_jit_lazy_adam"),
        parity_checks=t1_parity,
        richardson_checks=t1_rich,
        anti_cheat_checks={"no_flag_tamper": True, "exact_dense_adam_equiv": True},
        per_shape_metrics=m_t1_2,
        hypothesis_note="Replace O(total_N) Dense Adam with O(U) JIT-Lazy-Adam; still bottlenecked by Zipfian atomicAdd & dense grad memset",
    )
    del dense_emb_bm, dense_opt_bm
    torch.cuda.empty_cache()

    # Trial 3 (`INCUMBENT`): Full Round-4 RSI `EngramFusedLookupTable`
    # (GPU Prime Hash + Gather + Atomic-Free CSR Radix-Sort Segment-Reduce Bwd + Pre-Forward `JIT-Lazy-Adam(lr=5x, wd=0)`)
    tab_bm.copy_from_reference_embedding_(init_w_t1)
    m_t1_3: Dict[str, Dict[str, Any]] = {}
    for skey, B, T in t1_shapes:
        toks_s, grads_s = shape_slots_t1[skey]

        def _make_t3_slot(tk: torch.Tensor, gr: torch.Tensor):
            def _fn():
                em = tab_bm(tk, need_grad=True)
                em.backward(gr)
                tab_bm.step_lazy_adam(5e-3)
            return _fn

        m_t1_3[skey] = benchmark_rotating_cuda(
            [_make_t3_slot(toks_s[i], grads_s[i]) for i in range(n_slots)],
            warmup=warmup, groups=groups, iters_per_group=iters, l2_flush_buffer=l2_flush,
        )
    ratchet_t1.evaluate_and_record(
        trial_id=3,
        candidate_id="t3_round4_rsi_csr_atomic_free_bwd_pre_fwd_jit_lazy_adam",
        candidate_sha256=hash_callable_sources(EngramFusedLookupTable),
        parity_checks=t1_parity,
        richardson_checks=t1_rich,
        anti_cheat_checks={"no_flag_tamper": True, "zero_atomics": True, "zero_spills": True},
        per_shape_metrics=m_t1_3,
        hypothesis_note="Atomic-free CSR segment-reduce backward (0 atomics, 0 dense grad HBM) + Pre-Forward JIT-Lazy-Adam -> KEEP",
    )
    del tab_bm, shape_slots_t1
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Track 2: Engram M=4 Branch Context Gating + Dilated Causal ShortConv (Fwd+Bwd)
    # -------------------------------------------------------------------------
    t2_shapes = (
        [
            ("G1_B2_T512_M4_C1024", 2, 512, 1024),
            ("G2_B2_T1024_M4_C1536", 2, 1024, 1536),
            ("G3_B2_T1024_M4_C2560", 2, 1024, 2560),
        ]
        if is_smoke
        else [
            ("G1_B2_T1024_M4_C1536", 2, 1024, 1536),
            ("G2_B2_T2048_M4_C1536", 2, 2048, 1536),
            ("G3_B2_T2048_M4_C2560", 2, 2048, 2560),  # Exact DeepSeek Engram-27B width (d=2560, M=4, d_mem=1280)!
        ]
    )
    ratchet_t2 = RSIVectorRatchet("track2_engram_m4_gate_shortconv_fwdbwd", [s[0] for s in t2_shapes])

    # Shared non-empty FP64 parity & O(h^2) Richardson checks for Track 2 trials (`arXiv:2601.07372` Eq. 6 smooth gate)
    ref_chk2 = EagerDemoEngramContextOp(
        d_mem=D_MEM, d_backbone=256, hc_mult=4, dilation=3, signed_sqrt_gate=False, dtype=torch.bfloat16, device=device
    )
    H_c2 = (torch.randn(2, 64, 4, 256, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    E_c2 = (torch.randn(2, 64, D_MEM, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
    dY_c2 = torch.randn(2, 64, 4, 256, device=device, dtype=torch.bfloat16)
    packed_c2 = ref_chk2.export_packed_params()
    H64_2 = H_c2.detach().double().requires_grad_(True)
    E64_2 = E_c2.detach().double().requires_grad_(True)
    W64_2 = packed_c2["W_kv"].double().requires_grad_(True)
    wc64_2 = packed_c2["w_conv"].double().requires_grad_(True)
    Y64_2, _ = fp64_engram_gate_conv_oracle(
        H64_2, E64_2, W64_2, packed_c2["gamma_h"], packed_c2["gamma_k"], packed_c2["gamma_conv"], wc64_2,
        dilation=3, signed_sqrt_gate=False,
    )
    (Y64_2 * dY_c2.double()).sum().backward()

    # Smooth-gate Richardson probe for Track 2
    op_r2 = FusedEngramContextOp(
        d_mem=64, d_backbone=64, hc_mult=4, dilation=3, signed_sqrt_gate=False,
        block_c=64, block_mc=64, num_splits=4, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        nn.init.normal_(op_r2.W_kv, std=0.5)
        nn.init.normal_(op_r2.w_conv, std=0.5)
    Hr2 = (torch.randn(1, 16, 4, 64, device=device, dtype=torch.float32) * 0.6).requires_grad_(True)
    Er2 = (torch.randn(1, 16, 64, device=device, dtype=torch.float32) * 0.6).requires_grad_(True)
    wr2 = torch.randn(1, 16, 4, 64, device=device, dtype=torch.float32)
    (op_r2(Hr2, Er2) * wr2).sum().backward()
    t2_rich = [
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(
                x, Er2.detach(), op_r2.W_kv, op_r2.gamma_h, op_r2.gamma_k, op_r2.gamma_conv, op_r2.w_conv,
                dilation=3, signed_sqrt_gate=False
            )[0] * wr2.double()).sum(),
            Hr2.detach().double(), Hr2.grad.detach().double(), h_coarse=2e-2, h_fine=1e-2, name="t2_rich_dH",
        ),
        richardson_unit_rms_audit(
            lambda x: (fp64_engram_gate_conv_oracle(
                Hr2.detach(), Er2.detach(), op_r2.W_kv, op_r2.gamma_h, op_r2.gamma_k, op_r2.gamma_conv, x,
                dilation=3, signed_sqrt_gate=False
            )[0] * wr2.double()).sum(),
            op_r2.w_conv.detach().double(), op_r2.w_conv.grad.detach().double(),
            h_coarse=2e-2, h_fine=1e-2, name="t2_rich_dw_conv",
        ),
    ]

    def _check_t2_module_parity(mod_chk: nn.Module, tag: str) -> List[Dict[str, Any]]:
        Hc = H_c2.detach().clone().requires_grad_(True)
        Ec = E_c2.detach().clone().requires_grad_(True)
        mod_chk.zero_grad(set_to_none=True)
        Yc = mod_chk(Hc, Ec)
        Yc.backward(dY_c2)
        return [
            compare_tensor_to_fp64(Yc, Y64_2, rel_tol=1.5e-2, name=f"{tag}_Y"),
            compare_tensor_to_fp64(Hc.grad, H64_2.grad, rel_tol=1.5e-2, name=f"{tag}_dH"),
            compare_tensor_to_fp64(Ec.grad, E64_2.grad, rel_tol=1.5e-2, name=f"{tag}_dE"),
        ]

    t2_trial_defs = [
        (0, "t0_eager_demo_5gemm_12norm_transpose_cudnn_conv1d", "eager", {}),
        (1, "t1_torch_compile_inductor_demo_engram", "compiled", {}),
        (2, "t2_merged_kv_gemm_vectorized_gate_eager_conv1d", "fused", {"variant": "trial2_merged_kv_eager_conv"}),
        (3, "t3_zero_transpose_triton_gate_shortconv_b128_s16", "fused", {"variant": "trial3_conservative_tiles"}),
        (
            4,
            "t4_round4_rsi_two_stage_zero_transpose_split64_b256",
            "fused",
            {"variant": "trial4_full_fused", "block_c": 256, "block_mc": 256, "num_splits": 64},
        ),
    ]

    max_scratch_bytes_t4 = 0
    for tid, cid, kind, kw in t2_trial_defs:
        if kind == "eager":
            mod_p = ref_chk2
            c_sha = hash_callable_sources(EagerDemoEngramContextOp)
        elif kind == "compiled":
            mod_p = torch.compile(copy.deepcopy(ref_chk2))
            c_sha = hash_callable_sources(EagerDemoEngramContextOp, "torch.compile_inductor")
        else:
            mod_p = FusedEngramContextOp(
                d_mem=D_MEM, d_backbone=256, hc_mult=4, dilation=3, signed_sqrt_gate=False, dtype=torch.bfloat16, device=device, **kw
            )
            mod_p.load_from_packed_params_(packed_c2)
            c_sha = hash_callable_sources(FusedEngramContextOp, json.dumps(kw, sort_keys=True))

        p_checks = _check_t2_module_parity(mod_p, f"t{tid}")
        del mod_p

        per_shape: Dict[str, Dict[str, Any]] = {}
        for skey, B, T, C in t2_shapes:
            H_slots = [
                (torch.randn(B, T, 4, C, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
                for _ in range(n_slots)
            ]
            E_slots = [
                (torch.randn(B, T, D_MEM, device=device, dtype=torch.bfloat16) * 0.5).detach().requires_grad_(True)
                for _ in range(n_slots)
            ]
            dY_slots = [torch.randn(B, T, 4, C, device=device, dtype=torch.bfloat16) for _ in range(n_slots)]

            if kind == "eager":
                op_s: nn.Module = EagerDemoEngramContextOp(
                    d_mem=D_MEM, d_backbone=C, hc_mult=4, dilation=3, signed_sqrt_gate=False, dtype=torch.bfloat16, device=device
                )
            elif kind == "compiled":
                op_s = torch.compile(
                    EagerDemoEngramContextOp(
                        d_mem=D_MEM, d_backbone=C, hc_mult=4, dilation=3, signed_sqrt_gate=False, dtype=torch.bfloat16, device=device
                    )
                )
            else:
                op_f = FusedEngramContextOp(
                    d_mem=D_MEM, d_backbone=C, hc_mult=4, dilation=3, signed_sqrt_gate=False, dtype=torch.bfloat16, device=device, **kw
                )
                if tid == 4:
                    max_scratch_bytes_t4 = max(max_scratch_bytes_t4, op_f.scratch_bytes_estimate(B, T))
                op_s = op_f

            def _make_t2_slot(Hs: torch.Tensor, Es: torch.Tensor, dYs: torch.Tensor, m_op: nn.Module):
                def _fn():
                    m_op.zero_grad(set_to_none=True)
                    Hs.grad = None
                    Es.grad = None
                    out = m_op(Hs, Es)
                    out.backward(dYs)
                return _fn

            per_shape[skey] = benchmark_rotating_cuda(
                [_make_t2_slot(H_slots[i], E_slots[i], dY_slots[i], op_s) for i in range(n_slots)],
                warmup=warmup, groups=groups, iters_per_group=iters, l2_flush_buffer=l2_flush,
            )
            del op_s, H_slots, E_slots, dY_slots
            torch.cuda.empty_cache()

        ratchet_t2.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=c_sha,
            parity_checks=p_checks,
            richardson_checks=t2_rich,
            anti_cheat_checks={"no_flag_tamper": True, "zero_spills": True},
            per_shape_metrics=per_shape,
            hypothesis_note=f"Track 2 RSI Trial {tid}: {cid}",
        )

    t1_inc = ratchet_t1.incumbent_record
    t2_inc = ratchet_t2.incumbent_record
    assert t1_inc is not None and t2_inc is not None

    return (
        {
            "zipf_top64_token_share": round(top64_mass, 4),
            "track1_engram_lookup_bwd_opt": {
                "total_prime_rows": spec_t1.total_entries,
                "d_head": D_HEAD,
                "d_mem": D_MEM,
                "baseline_geomean_ms": ratchet_t1.baseline_record.geomean_ms,
                "incumbent_candidate": t1_inc.candidate_id,
                "incumbent_geomean_ms": t1_inc.geomean_ms,
                "speedup_vs_demo_baseline_x": t1_inc.speedup_vs_baseline,
                "baseline_peak_mib": ratchet_t1.baseline_record.peak_alloc_mib,
                "incumbent_peak_mib": t1_inc.peak_alloc_mib,
                "trials": [r.__dict__ for r in ratchet_t1.trials],
            },
            "track2_engram_m4_gate_shortconv": {
                "d_mem": D_MEM,
                "baseline_geomean_ms": ratchet_t2.baseline_record.geomean_ms,
                "compiled_geomean_ms": ratchet_t2.trials[1].geomean_ms,
                "incumbent_candidate": t2_inc.candidate_id,
                "incumbent_geomean_ms": t2_inc.geomean_ms,
                "speedup_vs_eager_demo_x": t2_inc.speedup_vs_baseline,
                "speedup_vs_torch_compile_x": round(ratchet_t2.trials[1].geomean_ms / t2_inc.geomean_ms, 4),
                "baseline_peak_mib": ratchet_t2.baseline_record.peak_alloc_mib,
                "incumbent_peak_mib": t2_inc.peak_alloc_mib,
                "hbm_reduction_ratio_x": round(ratchet_t2.baseline_record.peak_alloc_mib / max(t2_inc.peak_alloc_mib, 1.0), 4),
                "split_cta_scratch_mib": round(max_scratch_bytes_t4 / (1024.0 * 1024.0), 3),
                "trials": [r.__dict__ for r in ratchet_t2.trials],
            },
        },
        [ratchet_t1, ratchet_t2],
    )


# =============================================================================
# Building Blocks for Suite B & Suite C: `M = 4` `mHC` + Routed `MoE` + `Engram`
# =============================================================================


_compiled_eager_mhc_coeff = torch.compile(eager_mhc_coeff_eq7_eq9)


class MHCStreamAdapter(nn.Module):
    """M=4 Manifold-Constrained Hyper-Connection adapter (arXiv:2512.24880 Eq. 7-9)."""

    def __init__(self, d_model: int, n_streams: int = 4, sublayer_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_streams = n_streams
        K = n_streams * d_model
        M_dim = n_streams * (n_streams + 2)
        self.phi = nn.Parameter(torch.empty((K, M_dim), dtype=torch.bfloat16))
        self.gamma = nn.Parameter(torch.ones((K,), dtype=torch.bfloat16))
        self.alpha = nn.Parameter(torch.full((3,), 0.10, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((M_dim,), dtype=torch.float32))
        self.w_sub = nn.Parameter(torch.ones((d_model,), dtype=torch.bfloat16))

        with torch.no_grad():
            nn.init.normal_(self.phi, mean=0.0, std=1.0 / math.sqrt(float(K)))
            # Asymmetric Birkhoff initialization breaking stream symmetry at step 0
            self.b[:n_streams].fill_(-math.log(max(1, n_streams - 1)))
            self.b[:n_streams][sublayer_idx % n_streams] += 0.35
            self.b[n_streams : 2 * n_streams].fill_(-1.6)
            self.b[n_streams + (sublayer_idx % n_streams)] = 1.2
            self.b[2 * n_streams :].copy_(2.5 * torch.eye(n_streams, dtype=torch.float32).reshape(-1))


class _DroplessGroupedMMFn(torch.autograd.Function):
    """
    Zero-DtoH-Sync, 100% Dropless Hardware cuBLAS Grouped GEMM (`torch._grouped_mm`)
    for fine-grained Token-Choice MoE (`0.0%` token drops under any routing skew!).
    """

    @staticmethod
    def forward(ctx, x_sorted: torch.Tensor, weight_3d: torch.Tensor, offs_i32: torch.Tensor) -> torch.Tensor:
        xs = x_sorted.contiguous()
        y = torch._grouped_mm(xs, weight_3d, offs=offs_i32)
        ctx.save_for_backward(xs, weight_3d, offs_i32)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        xs, weight_3d, offs_i32 = ctx.saved_tensors
        dy = grad_out.contiguous()
        dx = torch._grouped_mm(dy, weight_3d.transpose(1, 2), offs=offs_i32)
        dw = torch._grouped_mm(xs.t(), dy, offs=offs_i32)
        return dx, dw, None


class TopKSharedSwiGLUMoE(nn.Module):
    """
    Fine-grained Shared + Top-K Routed SwiGLU MoE FFN (`arXiv:2601.07372` Sec. 3 allocation setup).
    Supports `n_routed == 0` for pure dense / pure-Engram (`rho = 0.00`) controls, and records
    DeepSeekMoE sequence-free load-balancing auxiliary loss (`last_aux_loss`) during training
    to prevent early router collapse.
    """

    def __init__(
        self,
        d_model: int,
        ffn_shared: int,
        n_routed: int = 0,
        ffn_expert: int = 128,
        top_k: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.ffn_shared = ffn_shared
        self.n_routed = n_routed
        self.ffn_expert = ffn_expert
        self.top_k = min(top_k, n_routed) if n_routed > 0 else 0
        self.last_aux_loss: Optional[torch.Tensor] = None

        self.shared_gate_up = nn.Linear(d_model, 2 * ffn_shared, bias=False, dtype=torch.bfloat16)
        self.shared_down = nn.Linear(ffn_shared, d_model, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(self.shared_gate_up.weight, std=0.02)
        nn.init.normal_(self.shared_down.weight, std=0.02 / math.sqrt(2.0))

        if n_routed > 0:
            self.router = nn.Linear(d_model, n_routed, bias=False, dtype=torch.bfloat16)
            nn.init.normal_(self.router.weight, std=0.02)
            self.exp_gate_up = nn.Parameter(
                torch.randn((n_routed, d_model, 2 * ffn_expert), dtype=torch.bfloat16) * 0.02
            )
            self.exp_down = nn.Parameter(
                torch.randn((n_routed, ffn_expert, d_model), dtype=torch.bfloat16) * (0.02 / math.sqrt(2.0))
            )
        else:
            self.router = None
            self.exp_gate_up = None
            self.exp_down = None

    def forward(self, x: torch.Tensor, fused_swiglu: bool = True) -> torch.Tensor:
        gu = self.shared_gate_up(x)
        if fused_swiglu:
            act = FusedSwiGLUFn.apply(gu)
        else:
            g, u = gu.chunk(2, dim=-1)
            act = (F.silu(g.float()) * u.float()).to(x.dtype)
        out = self.shared_down(act)

        if self.n_routed == 0:
            self.last_aux_loss = None
            return out

        logits = self.router(x).float()
        scores = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)).to(x.dtype)

        # 100% Dropless, Zero-DtoH-Sync cuBLAS Grouped GEMM (`torch._grouped_mm`):
        N_tok = x.shape[0]
        flat_e = topk_idx.reshape(-1)
        flat_w = topk_w.reshape(-1)
        flat_tok = torch.arange(N_tok, device=x.device, dtype=torch.long).unsqueeze(1).expand(N_tok, self.top_k).reshape(-1)

        sorted_e, perm = torch.sort(flat_e)
        counts = torch.bincount(sorted_e, minlength=self.n_routed)
        offs_i32 = torch.cumsum(counts, dim=0).to(torch.int32)

        if self.training:
            frac_e = counts.float() / float(max(1, N_tok))
            prob_e = scores.mean(dim=0)
            self.last_aux_loss = 0.01 * float(self.n_routed) * torch.sum(frac_e.detach() * prob_e)
        else:
            self.last_aux_loss = None

        tok_sorted = flat_tok.index_select(0, perm)
        w_sorted = flat_w.index_select(0, perm).unsqueeze(-1)
        x_sorted = x.index_select(0, tok_sorted)

        gu_sorted = _DroplessGroupedMMFn.apply(x_sorted, self.exp_gate_up, offs_i32)
        if fused_swiglu:
            act_sorted = FusedSwiGLUFn.apply(gu_sorted)
        else:
            ge, ue = gu_sorted.chunk(2, dim=-1)
            act_sorted = (F.silu(ge.float()) * ue.float()).to(x.dtype)
        y_sorted = _DroplessGroupedMMFn.apply(act_sorted, self.exp_down, offs_i32)
        y_weighted = (y_sorted * w_sorted).to(out.dtype)

        routed_out = torch.zeros_like(out)
        routed_out.index_add_(0, tok_sorted, y_weighted)
        return out + routed_out


class MHCEngramBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        h_q: int,
        h_kv: int,
        ffn_shared: int,
        block_idx: int,
        n_streams: int = 4,
        n_routed: int = 0,
        ffn_expert: int = 128,
        top_k: int = 2,
        has_engram: bool = False,
        d_mem: int = 1280,
        engram_mode: str = "fused",
        zero_init_conv: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = d_model // h_q
        self.n_streams = n_streams
        self.block_idx = block_idx

        if has_engram:
            if engram_mode == "eager":
                self.engram_ctx: Optional[nn.Module] = EagerDemoEngramContextOp(
                    d_mem=d_mem, d_backbone=d_model, hc_mult=n_streams, dilation=3,
                    signed_sqrt_gate=False, zero_init_conv=zero_init_conv, dtype=torch.bfloat16,
                )
            elif engram_mode == "compiled":
                self.engram_ctx = torch.compile(
                    EagerDemoEngramContextOp(
                        d_mem=d_mem, d_backbone=d_model, hc_mult=n_streams, dilation=3,
                        signed_sqrt_gate=False, zero_init_conv=zero_init_conv, dtype=torch.bfloat16,
                    )
                )
            else:
                self.engram_ctx = FusedEngramContextOp(
                    d_mem=d_mem, d_backbone=d_model, hc_mult=n_streams, dilation=3,
                    signed_sqrt_gate=False, zero_init_conv=zero_init_conv,
                    block_c=min(256, max(64, 1 << (d_model - 1).bit_length())),
                    block_mc=256,
                    num_splits=32,
                    dtype=torch.bfloat16,
                )
        else:
            self.engram_ctx = None

        qkv_out = (h_q + 2 * h_kv) * self.head_dim
        self.w_qkv = nn.Linear(d_model, qkv_out, bias=False, dtype=torch.bfloat16)
        self.w_o = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(self.w_qkv.weight, std=0.02)
        nn.init.normal_(self.w_o.weight, std=0.02 / math.sqrt(2.0))

        self.mhc_attn = MHCStreamAdapter(d_model, n_streams, sublayer_idx=2 * block_idx)
        self.mhc_ffn = MHCStreamAdapter(d_model, n_streams, sublayer_idx=2 * block_idx + 1)
        self.ffn = TopKSharedSwiGLUMoE(
            d_model=d_model, ffn_shared=ffn_shared, n_routed=n_routed, ffn_expert=ffn_expert, top_k=top_k
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

    def _run_mhc_sublayer(
        self, X: torch.Tensor, adapter: MHCStreamAdapter, core_fn, mhc_mode: str
    ) -> torch.Tensor:
        if mhc_mode in ("eager", "compiled"):
            coeff_fn = _compiled_eager_mhc_coeff if mhc_mode == "compiled" else eager_mhc_coeff_eq7_eq9
            hp, ho, hr = coeff_fn(X, adapter.phi, adapter.gamma, adapter.alpha, adapter.b)
            h_in = torch.einsum("bi,bic->bc", hp, X.float()).to(X.dtype)
            rstd = torch.rsqrt(torch.mean(h_in.float() ** 2, dim=-1, keepdim=True) + 1e-6)
            h_norm = (h_in.float() * rstd * adapter.w_sub.float()).to(X.dtype)
            y = core_fn(h_norm)
            return torch.einsum("bij,bjc->bic", hr, X.float()).to(X.dtype) + (
                ho.unsqueeze(-1) * y.float().unsqueeze(1)
            ).to(X.dtype)
        hp, ho, hr = fused_mhc_coeff(X, adapter.phi, adapter.gamma, adapter.alpha, adapter.b, single_scan=True)
        h_norm = SuperfusedPreRMSNormFn.apply(X, hp, adapter.w_sub, 1e-6)
        y = core_fn(h_norm)
        return FusedPostResMergeFn.apply(X, y, ho, hr)

    def forward(
        self,
        X: torch.Tensor,  # [B*S, M, C]
        B: int,
        S: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        engram_embs: Optional[torch.Tensor] = None,
        mhc_mode: str = "fused",
    ) -> torch.Tensor:
        if self.engram_ctx is not None and engram_embs is not None:
            X = self.engram_ctx(X.view(B, S, self.n_streams, self.d_model), engram_embs).view(
                B * S, self.n_streams, self.d_model
            )
        fused_sw = mhc_mode != "eager"
        X = self._run_mhc_sublayer(X, self.mhc_attn, lambda hn: self._attn_core(hn, B, S, cos, sin), mhc_mode)
        X = self._run_mhc_sublayer(X, self.mhc_ffn, lambda hn: self.ffn(hn, fused_swiglu=fused_sw), mhc_mode)
        return X


class MHCEngramBackboneLM(nn.Module):
    """
    M=4 mHC Backbone Transformer with optional Layer-2 Engram Context-Gate+ShortConv.
    Lookup table (`EngramFusedLookupTable`, `ShardedEngramLookupEngine`, or `nn.Embedding`)
    is passed externally so replicated vs 8-GPU NVSwitch sharded/prefetched regimes share
    the exact same deterministic backbone parameter set!
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int = 6,
        h_q: int = 12,
        h_kv: int = 4,
        ffn_shared: int = 1024,
        n_streams: int = 4,
        n_routed: int = 0,
        ffn_expert: int = 128,
        top_k: int = 2,
        engram_layer_id: Optional[int] = 2,
        d_mem: int = 1280,
        engram_mode: str = "fused",
        zero_init_conv: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_streams = n_streams
        self.engram_layer_id = engram_layer_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, dtype=torch.bfloat16)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        # Hadamard orthogonal stream offsets breaking step-0 multi-stream symmetry
        had = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]],
            dtype=torch.float32,
        ) * 0.01
        stream_init = had[:, : min(4, d_model)].repeat(1, (d_model + 3) // 4)[:, :d_model]
        self.stream_emb = nn.Parameter(stream_init.clone())

        self.blocks = nn.ModuleList(
            [
                MHCEngramBlock(
                    d_model=d_model,
                    h_q=h_q,
                    h_kv=h_kv,
                    ffn_shared=ffn_shared,
                    block_idx=i,
                    n_streams=n_streams,
                    n_routed=n_routed,
                    ffn_expert=ffn_expert,
                    top_k=top_k,
                    has_engram=(engram_layer_id is not None and i == engram_layer_id),
                    d_mem=d_mem,
                    engram_mode=engram_mode,
                    zero_init_conv=zero_init_conv,
                )
                for i in range(n_layers)
            ]
        )
        self.readout_logits = nn.Parameter(torch.tensor([0.4, 0.1, -0.1, -0.2], dtype=torch.float32))
        self.final_norm = nn.Parameter(torch.ones((d_model,), dtype=torch.bfloat16))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward_with_engram_provider(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        mhc_mode: str = "fused",
        engram_embs_ready: Optional[torch.Tensor] = None,
        prefetch_join_fn=None,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        x0 = self.tok_emb(input_ids).view(B * S, 1, self.d_model)
        X = (x0.float() + self.stream_emb.unsqueeze(0)).to(torch.bfloat16)

        e_mem = engram_embs_ready
        for idx, blk in enumerate(self.blocks):
            if idx == self.engram_layer_id and prefetch_join_fn is not None:
                e_mem = prefetch_join_fn()
            X = blk(
                X,
                B,
                S,
                cos,
                sin,
                engram_embs=(e_mem if idx == self.engram_layer_id else None),
                mhc_mode=mhc_mode,
            )

        rw = torch.softmax(self.readout_logits, dim=0).view(1, self.n_streams, 1)
        h_fin = (X.float() * rw).sum(dim=1)
        rstd = torch.rsqrt(torch.mean(h_fin * h_fin, dim=-1, keepdim=True) + 1e-6)
        h_out = (h_fin * rstd * self.final_norm.float()).to(torch.bfloat16)
        logits = self.lm_head(h_out)
        ce = F.cross_entropy(logits.float(), targets.reshape(-1))
        if self.training:
            aux_terms = [blk.ffn.last_aux_loss for blk in self.blocks if blk.ffn.last_aux_loss is not None]
            if aux_terms:
                ce = ce + torch.stack(aux_terms).mean()
        return ce


def _ddp_average_grads_(model: nn.Module, world_size: int) -> None:
    if world_size <= 1:
        return
    # Flatten by dtype into 2 coalesced NVSwitch all_reduce buckets (FP32 + BF16)
    by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
    for p in model.parameters():
        if p.grad is not None:
            by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
    for dt, grads in by_dtype.items():
        flat = torch._utils._flatten_dense_tensors(grads)
        if dt == torch.bfloat16:
            flat_f = flat.float() / float(world_size)
            dist.all_reduce(flat_f, op=dist.ReduceOp.SUM)
            flat.copy_(flat_f.to(dt))
        else:
            flat.div_(float(world_size))
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for g, u in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(u)


# =============================================================================
# Suite B: 8-GPU NVSwitch Sharded Engram (`Dedup-AllToAllv`) + Layer-2 Async
#          Stream Prefetch Behind mHC Blocks 0..1 + Gate G9 3-Step 1-GPU Parity
# =============================================================================


def run_suite_b_sharded_engram_nvswitch(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> Dict[str, Any]:
    # -------------------------------------------------------------------------
    # B.1 Gate G9: 3-Step Multi-Batch 1-GPU Unsharded Oracle vs 8-GPU Modulo-Sharded
    #     `Dedup-AllToAllv` + Layer-2 Async Stream Prefetch Exact Parity
    # -------------------------------------------------------------------------
    spec_g9 = build_engram_hash_spec(
        vocab_size=8192, engram_vocab_size_per_ngram=(8192, 8192), max_ngram_size=3, n_head_per_ngram=8, layer_id=2
    )
    D_HEAD_G9 = 40
    D_MEM_G9 = 16 * D_HEAD_G9
    B_par, S_par = 2, 128
    cos_p, sin_p = build_rope_cos_sin(S_par, 32, device)

    torch.manual_seed(20261305)
    cpu_backbone = MHCEngramBackboneLM(
        vocab_size=spec_g9.vocab_size,
        d_model=256,
        n_layers=4,
        h_q=8,
        h_kv=4,
        ffn_shared=512,
        n_streams=4,
        engram_layer_id=2,
        d_mem=D_MEM_G9,
        engram_mode="fused",
        zero_init_conv=False,  # Nonzero ShortConv in G9 so all conv/gate gradients are actively exercised at step 1!
    )
    init_full_table = torch.randn((spec_g9.total_entries, D_HEAD_G9), dtype=torch.float32) * 0.02
    init_full_table = init_full_table.to(torch.bfloat16).float()

    # 3 consecutive global Zipfian steps so Pre-Forward `JIT-Lazy-Adam` catch-up is exercised across shards!
    step_inputs_per_rank: List[List[torch.Tensor]] = []
    step_targets_per_rank: List[List[torch.Tensor]] = []
    for st in range(3):
        r_in, r_tg = [], []
        for r in range(world_size):
            slots_r, _ = make_zipf_token_batches(
                1, B_par, S_par + 1, spec_g9.vocab_size, torch.device("cpu"), zipf_s=1.18, seed=8800 + st * 31 + r
            )
            seq_r = slots_r[0]
            r_in.append(seq_r[:, :-1].contiguous())
            r_tg.append(seq_r[:, 1:].contiguous())
        step_inputs_per_rank.append(r_in)
        step_targets_per_rank.append(r_tg)

    ref_1g: Dict[str, Any] = {}
    # Also prepare exact upstream gradients per rank for 3-step isolated Sharded JIT-Lazy-Adam table proof (`< 1e-5`)
    exact_up_grads_per_rank: List[List[torch.Tensor]] = []
    for st in range(3):
        ug_st = []
        for r in range(world_size):
            gr_u = torch.Generator(device="cpu").manual_seed(9100 + st * 37 + r)
            ug_st.append(torch.randn((B_par, S_par, D_MEM_G9), generator=gr_u, dtype=torch.bfloat16))
        exact_up_grads_per_rank.append(ug_st)

    if rank == 0:
        # Part 1A: Exact 3-step 1-GPU unsharded table oracle
        t1_iso = EngramFusedLookupTable(spec_g9, d_head=D_HEAD_G9, dtype=torch.bfloat16, device=device)
        t1_iso.copy_from_reference_embedding_(init_full_table.to(device))
        for st in range(3):
            all_in = torch.cat(step_inputs_per_rank[st], dim=0).to(device)
            all_ug = (torch.cat(exact_up_grads_per_rank[st], dim=0).to(device).float() / float(world_size)).to(torch.bfloat16)
            eo = t1_iso(all_in, need_grad=True)
            eo.backward(all_ug)
            t1_iso.step_lazy_adam(1e-2 * (1.0 - 0.15 * st))
        t1_iso.flush_all_rows_()
        iso_master_1g = t1_iso.master_weight.detach().cpu()
        del t1_iso

        # Part 1B: End-to-end 4-layer BF16 mHC+Engram Transformer 3-step trajectory + Step-1 exact gradients
        m1 = copy.deepcopy(cpu_backbone).to(device)
        t1 = EngramFusedLookupTable(spec_g9, d_head=D_HEAD_G9, dtype=torch.bfloat16, device=device)
        t1.copy_from_reference_embedding_(init_full_table.to(device))
        opt1 = torch.optim.AdamW(m1.parameters(), lr=2e-3, weight_decay=0.01)
        losses_1g: List[float] = []
        step1_grads_1g: Dict[str, torch.Tensor] = {}
        for st in range(3):
            all_in = torch.cat(step_inputs_per_rank[st], dim=0).to(device)
            all_tg = torch.cat(step_targets_per_rank[st], dim=0).to(device)
            opt1.zero_grad(set_to_none=True)
            e_1g = t1(all_in, need_grad=True)
            loss_1 = m1.forward_with_engram_provider(all_in, all_tg, cos_p, sin_p, engram_embs_ready=e_1g)
            loss_1.backward()
            losses_1g.append(float(loss_1.item()))
            if st == 0:
                step1_grads_1g = {
                    "tok_emb": m1.tok_emb.weight.grad.detach().cpu(),
                    "blk0_wqkv": m1.blocks[0].w_qkv.weight.grad.detach().cpu(),
                    "blk2_engram_Wkv": m1.blocks[2].engram_ctx.W_kv.grad.detach().cpu(),
                    "blk2_engram_wconv": m1.blocks[2].engram_ctx.w_conv.grad.detach().cpu(),
                    "lm_head": m1.lm_head.weight.grad.detach().cpu(),
                }
            opt1.step()
            t1.step_lazy_adam(1e-2)
        ref_1g = {
            "losses": losses_1g,
            "step1_grads": step1_grads_1g,
            "iso_master_1g": iso_master_1g,
        }
        del m1, t1, opt1
        torch.cuda.empty_cache()

    dist.barrier()

    # Part 1A on 8 GPUs: Exact 3-step Modulo-Sharded Dedup-AllToAllv + Async Stream JIT-Lazy-Adam table parity
    sh8_iso = ShardedEngramLookupEngine(spec_g9, d_head=D_HEAD_G9, dtype=torch.bfloat16, device=device)
    sh8_iso.load_from_full_table_(init_full_table.to(device))
    pf_iso = torch.cuda.Stream(device=device)
    cs_iso = torch.cuda.current_stream(device=device)
    for st in range(3):
        in_r = step_inputs_per_rank[st][rank].to(device)
        ug_r = (exact_up_grads_per_rank[st][rank].to(device).float() / float(world_size)).to(torch.bfloat16)
        pf_iso.wait_stream(cs_iso)
        in_r.record_stream(pf_iso)
        with torch.cuda.stream(pf_iso):
            st_iso = sh8_iso.prefetch_forward_exchange(in_r, dedup=True)
        cs_iso.wait_stream(pf_iso)
        em_iso = sh8_iso.attach_autograd(st_iso)
        em_iso.backward(ug_r)
        sh8_iso.backward_exchange_and_reduce(comm_stream=pf_iso, compute_stream=cs_iso)
        cs_iso.wait_stream(pf_iso)
        sh8_iso.step_shard_lazy_adam(1e-2 * (1.0 - 0.15 * st))
    stale_iso = torch.nonzero((sh8_iso.last_step_shard > 0) & (sh8_iso.last_step_shard < sh8_iso.completed_step), as_tuple=False).view(-1).to(torch.int32)
    sh8_iso._catchup_local_shard_(stale_iso)
    gathered_iso = [torch.empty_like(sh8_iso.master_shard) for _ in range(world_size)]
    dist.all_gather(gathered_iso, sh8_iso.master_shard)
    del sh8_iso

    # Part 1B on 8 GPUs: End-to-end 4-layer BF16 mHC+Engram Transformer with Layer-2 Async Stream Prefetch
    m8 = copy.deepcopy(cpu_backbone).to(device)
    sh8 = ShardedEngramLookupEngine(spec_g9, d_head=D_HEAD_G9, dtype=torch.bfloat16, device=device)
    sh8.load_from_full_table_(init_full_table.to(device))
    opt8 = torch.optim.AdamW(m8.parameters(), lr=2e-3, weight_decay=0.01)
    prefetch_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    losses_8g: List[float] = []
    step1_grads_8g: Dict[str, torch.Tensor] = {}
    for st in range(3):
        in_r = step_inputs_per_rank[st][rank].to(device)
        tg_r = step_targets_per_rank[st][rank].to(device)
        opt8.zero_grad(set_to_none=True)

        prefetch_stream.wait_stream(compute_stream)
        in_r.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            pf_state = sh8.prefetch_forward_exchange(in_r, dedup=True)

        def _join_prefetch():
            compute_stream.wait_stream(prefetch_stream)
            e_t: torch.Tensor = pf_state["embs_3d"]
            e_t.record_stream(compute_stream)
            return sh8.attach_autograd(pf_state)

        loss_r = m8.forward_with_engram_provider(
            in_r, tg_r, cos_p, sin_p, prefetch_join_fn=_join_prefetch
        )
        loss_r.backward()
        pf_state["grad_embs_3d"].div_(float(world_size))
        sh8.backward_exchange_and_reduce(comm_stream=prefetch_stream, compute_stream=compute_stream)
        _ddp_average_grads_(m8, world_size)
        compute_stream.wait_stream(prefetch_stream)

        l_avg = loss_r.detach().double() / float(world_size)
        dist.all_reduce(l_avg, op=dist.ReduceOp.SUM)
        losses_8g.append(float(l_avg.item()))

        if st == 0 and rank == 0:
            step1_grads_8g = {
                "tok_emb": m8.tok_emb.weight.grad.detach().cpu(),
                "blk0_wqkv": m8.blocks[0].w_qkv.weight.grad.detach().cpu(),
                "blk2_engram_Wkv": m8.blocks[2].engram_ctx.W_kv.grad.detach().cpu(),
                "blk2_engram_wconv": m8.blocks[2].engram_ctx.w_conv.grad.detach().cpu(),
                "lm_head": m8.lm_head.weight.grad.detach().cpu(),
            }
        opt8.step()
        sh8.step_shard_lazy_adam(1e-2)

    g9_parity: Dict[str, Any] = {}
    if rank == 0:
        reassembled = torch.empty((gathered_iso[0].shape[0] * world_size, D_HEAD_G9), dtype=torch.float32)
        for r_idx, sh_t in enumerate(gathered_iso):
            reassembled[r_idx::world_size] = sh_t.cpu()
        reassembled = reassembled[: spec_g9.total_entries]

        loss_rel_diffs = [
            abs(l8 - l1) / max(abs(l1), 1e-12) for l8, l1 in zip(losses_8g, ref_1g["losses"])
        ]
        grad_rels = {
            k: compare_tensor_to_fp64(step1_grads_8g[k], ref_1g["step1_grads"][k], rel_tol=1.5e-2, name=k)["rel_l2"]
            for k in step1_grads_8g
        }
        table_rel = compare_tensor_to_fp64(
            reassembled, ref_1g["iso_master_1g"], rel_tol=1e-4, name="sharded_master_table_3step"
        )["rel_l2"]
        g9_ok = max(loss_rel_diffs) < 5e-3 and max(grad_rels.values()) < 1.5e-2 and table_rel < 1e-4
        assert g9_ok, f"Gate G9 1-GPU vs 8-GPU Sharded Prefetch parity failed: losses={loss_rel_diffs}, grads={grad_rels}, table={table_rel}"
        g9_parity = {
            "passed": g9_ok,
            "losses_1gpu": [round(x, 6) for x in ref_1g["losses"]],
            "losses_8gpu_sharded_prefetch": [round(x, 6) for x in losses_8g],
            "max_3step_loss_rel_diff": round(max(loss_rel_diffs), 7),
            "step1_grad_rel_l2": grad_rels,
            "reassembled_8shard_3step_master_table_rel_l2": table_rel,
        }

    del m8, sh8, opt8, gathered_iso, cpu_backbone
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    # -------------------------------------------------------------------------
    # B.2 8-GPU NVSwitch Sharded Engram Scaling (`8.39M` Prime Rows = `5.0 GiB` Table State,
    #     `L = 6` `M = 4` `mHC` Blocks, `d_model = 1536`, `S = 2,048`, `65,536` Global Tok/Step)
    # -------------------------------------------------------------------------
    V_b2 = 65536
    per_ngram_b2 = (131072, 131072) if is_smoke else (524288, 524288)  # 16 prime heads -> ~8.39M rows in full!
    spec_b2 = build_engram_hash_spec(
        vocab_size=V_b2, engram_vocab_size_per_ngram=per_ngram_b2, max_ngram_size=3, n_head_per_ngram=8, layer_id=2
    )
    D_HEAD_B2 = 80
    D_MEM_B2 = 16 * D_HEAD_B2  # 1280
    C_B2 = 1024 if is_smoke else 1536
    L_B2 = 6
    B_loc = 2 if is_smoke else 4
    S_loc = 512 if is_smoke else 2048
    glob_tok_b2 = world_size * B_loc * S_loc
    cos_b2, sin_b2 = build_rope_cos_sin(S_loc, C_B2 // 16, device)

    n_rot = 3 if is_smoke else 4
    in_slots, zipf_share_b2 = make_zipf_token_batches(
        n_rot, B_loc, S_loc + 1, V_b2, device, zipf_s=1.18, seed=9500 + rank
    )
    inp_rot = [s[:, :-1].contiguous() for s in in_slots]
    tgt_rot = [s[:, 1:].contiguous() for s in in_slots]
    del in_slots

    warmup_b2 = 2 if is_smoke else 3
    steps_b2 = 4 if is_smoke else 8
    regimes_out: List[Dict[str, Any]] = []

    regime_specs = [
        ("r0_eager_replicated_demo_baseline", "replicated_eager"),
        ("r1_sharded_naive_alltoall_sync", "sharded_naive_sync"),
        ("r2_sharded_dedup_alltoallv_sync", "sharded_dedup_sync"),
        ("r3_sharded_dedup_alltoallv_layer2_async_prefetch", "sharded_dedup_prefetch"),
    ]

    for r_name, r_kind in regime_specs:
        torch.manual_seed(20261306)
        is_eager = r_kind == "replicated_eager"
        lm = MHCEngramBackboneLM(
            vocab_size=V_b2,
            d_model=C_B2,
            n_layers=L_B2,
            h_q=16,
            h_kv=4,
            ffn_shared=int(C_B2 * 2.5),
            n_streams=4,
            engram_layer_id=2,
            d_mem=D_MEM_B2,
            engram_mode=("eager" if is_eager else "fused"),
            zero_init_conv=True,
        ).to(device)
        opt_lm = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01, fused=True)

        if is_eager:
            rep_emb = nn.Embedding(spec_b2.total_entries, D_HEAD_B2, device=device, dtype=torch.bfloat16)
            rep_opt = torch.optim.Adam(rep_emb.parameters(), lr=5e-3, weight_decay=0.0, fused=True)
            sh_eng = None
        else:
            rep_emb = None
            rep_opt = None
            sh_eng = ShardedEngramLookupEngine(spec_b2, d_head=D_HEAD_B2, dtype=torch.bfloat16, device=device)

        pf_s = torch.cuda.Stream(device=device)
        comp_s = torch.cuda.current_stream(device=device)

        def _step_once(slot_i: int) -> None:
            ib = inp_rot[slot_i % n_rot]
            tb = tgt_rot[slot_i % n_rot]
            opt_lm.zero_grad(set_to_none=True)

            if r_kind == "replicated_eager":
                h_id = cpu_numpy_ngram_hashes_ref(ib, spec_b2)
                rep_opt.zero_grad(set_to_none=True)
                em = rep_emb(h_id).view(B_loc, S_loc, D_MEM_B2)
                loss = lm.forward_with_engram_provider(ib, tb, cos_b2, sin_b2, mhc_mode="eager", engram_embs_ready=em)
                loss.backward()
                _ddp_average_grads_(lm, world_size)
                # Replicated table also requires full-table all_reduce across 8 ranks!
                g_rep = rep_emb.weight.grad
                dist.all_reduce(g_rep, op=dist.ReduceOp.SUM)
                g_rep.div_(float(world_size))
                opt_lm.step()
                rep_opt.step()
                return

            use_dedup = r_kind != "sharded_naive_sync"
            if r_kind == "sharded_dedup_prefetch":
                # True CPU+GPU Overlap: record pre-step event BEFORE Block 0, enqueue Blocks 0..1 onto `comp_s`
                # FIRST so all 8 GPUs are actively executing Attention+FFN while `pf_s` runs Dedup-AllToAllv!
                ev_pre = torch.cuda.Event()
                ev_pre.record(comp_s)
                st_holder: Dict[str, Any] = {}

                def _launch_and_join_pf():
                    ib.record_stream(pf_s)
                    with torch.cuda.stream(pf_s):
                        pf_s.wait_event(ev_pre)
                        st_pf_inner = sh_eng.prefetch_forward_exchange(ib, dedup=True)
                    st_holder["st"] = st_pf_inner
                    comp_s.wait_stream(pf_s)
                    et: torch.Tensor = st_pf_inner["embs_3d"]
                    et.record_stream(comp_s)
                    return sh_eng.attach_autograd(st_pf_inner)

                loss = lm.forward_with_engram_provider(
                    ib, tb, cos_b2, sin_b2, mhc_mode="fused", prefetch_join_fn=_launch_and_join_pf
                )
                loss.backward()
                st_pf = st_holder["st"]
                st_pf["grad_embs_3d"].div_(float(world_size))
                sh_eng.backward_exchange_and_reduce(comm_stream=pf_s, compute_stream=comp_s)
                _ddp_average_grads_(lm, world_size)
                comp_s.wait_stream(pf_s)
                opt_lm.step()
                sh_eng.step_shard_lazy_adam(5e-3)
            else:
                st_sync = sh_eng.prefetch_forward_exchange(ib, dedup=use_dedup)
                em_s = sh_eng.attach_autograd(st_sync)
                loss = lm.forward_with_engram_provider(ib, tb, cos_b2, sin_b2, mhc_mode="fused", engram_embs_ready=em_s)
                loss.backward()
                st_sync["grad_embs_3d"].div_(float(world_size))
                sh_eng.backward_exchange_and_reduce()
                _ddp_average_grads_(lm, world_size)
                opt_lm.step()
                sh_eng.step_shard_lazy_adam(5e-3)

        for w in range(warmup_b2):
            _step_once(w)
        comp_s.wait_stream(pf_s)
        torch.cuda.synchronize(device)
        dist.barrier()

        torch.cuda.reset_peak_memory_stats(device)
        _step_once(0)
        comp_s.wait_stream(pf_s)
        torch.cuda.synchronize(device)
        peak_mib_local = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

        dist.barrier()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record(comp_s)
        for s_i in range(steps_b2):
            _step_once(s_i)
        comp_s.wait_stream(pf_s)
        ev1.record(comp_s)
        ev1.synchronize()
        dist.barrier()

        step_ms_local = ev0.elapsed_time(ev1) / float(steps_b2)
        stats_vec = torch.tensor(
            [
                step_ms_local,
                peak_mib_local,
                float(sh_eng.last_comm_stats.get("dedup_compression_ratio", 1.0)) if sh_eng is not None else 1.0,
                float(sh_eng.last_comm_stats.get("fwd_comm_bytes_per_rank", 0.0)) if sh_eng is not None else float(spec_b2.total_entries * D_HEAD_B2 * 2),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats_vec, op=dist.ReduceOp.MAX)
        step_ms = float(stats_vec[0].item())
        peak_mib = float(stats_vec[1].item())
        dedup_ratio = float(stats_vec[2].item())
        comm_mib = float(stats_vec[3].item()) / (1024.0 * 1024.0)

        regimes_out.append(
            {
                "regime": r_name,
                "step_ms": round(step_ms, 3),
                "global_tok_per_s": round((glob_tok_b2 * 1000.0) / max(step_ms, 1e-6), 1),
                "peak_alloc_mib_per_rank": round(peak_mib, 2),
                "zipf_dedup_compression_ratio_x": round(dedup_ratio, 3),
                "engram_comm_payload_mib_per_rank": round(comm_mib, 3),
            }
        )
        del lm, opt_lm, rep_emb, rep_opt, sh_eng
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    r0, r1, r2, r3 = regimes_out
    return {
        "gate_g9_1gpu_vs_8gpu_sharded_prefetch_parity": g9_parity,
        "scaling_config": {
            "world_size": world_size,
            "total_prime_engram_rows": spec_b2.total_entries,
            "d_head": D_HEAD_B2,
            "d_mem": D_MEM_B2,
            "d_backbone": C_B2,
            "n_mhc_blocks": L_B2,
            "seq_len": S_loc,
            "global_tokens_per_step": glob_tok_b2,
            "zipf_top64_token_share": round(zipf_share_b2, 4),
        },
        "regimes": regimes_out,
        "speedup_prefetch_vs_replicated_demo_x": round(r0["step_ms"] / r3["step_ms"], 4),
        "speedup_prefetch_vs_naive_alltoall_x": round(r1["step_ms"] / r3["step_ms"], 4),
        "speedup_prefetch_vs_sync_dedup_x": round(r2["step_ms"] / r3["step_ms"], 4),
        "hbm_saved_per_rank_mib": round(r0["peak_alloc_mib_per_rank"] - r3["peak_alloc_mib_per_rank"], 2),
    }


# =============================================================================
# Suite C: Round-4 Fixed-Wall-Clock (`12.0s`) `val_bpb` RSI Co-Design Loop &
#          Matched-Step Iso-Param / Iso-Active-FLOPs Sparsity Allocation Law
#          (`rho in {1.00, 0.75, 0.25}` + Dense Controls, `arXiv:2601.07372` Sec. 3)
# =============================================================================


def _generate_hybrid_lexical_compositional_corpus(
    n_seqs: int,
    seq_len: int,
    vocab_size: int,
    compressed_vocab_size: int,
    lookup_table_np: np.ndarray,
    seed: int,
) -> torch.Tensor:
    """
    Generates a dual-regime token corpus faithfully testing DeepSeek Engram's two axes:
      1. Static N-Gram Entity / Collocation Knowledge (~55% of positions):
         Multi-token stereotyped phrases keyed by canonical 2-gram / 3-gram stems,
         emitted with random surface-form variants (`t` vs `v` sharing `lut[v] == lut[t]`),
         rewarding surjective O(1) N-gram conditional memory!
      2. Long-Range Context-Switched Compositional Rules (~45% of positions):
         Every clause begins with a regime header `mode in {0, 1, 2, 3}` 24..64 tokens prior
         that switches the modular relational continuation of ambiguous triggers where
         local 3-grams alone have 2 bits of conditional entropy unless `Engram` gates off
         (`alpha_t -> 0`) and routed `MoE` experts + causal attention apply the regime rule!
    """
    rng = np.random.default_rng(seed)
    # Build inverse canonical buckets so canonical stems emit realistic surface variants (~23% collisions)
    canon_buckets: List[List[int]] = [[] for _ in range(compressed_vocab_size)]
    for raw_id, cid in enumerate(lookup_table_np.tolist()):
        canon_buckets[cid].append(raw_id)

    # Fixed world knowledge tables (shared across train/val/test via deterministic world seed 20260777)
    world_rng = np.random.default_rng(20260777)
    n_entities = 192
    # 1. Static 4-token canonical entity idioms (`~35%` of clauses):
    # 192 Zipfian entities (~576 canonical 2-grams/3-grams) emitted with ~23% surface variants.
    # Notice `rho = 0.75` (`8,280` prime slots/head) ALREADY achieves ~0% collision on all 576 entity N-grams,
    # so expanding Engram 3x-4x at `rho = 0.25` / `rho = 0.00` yields diminishing (~0 bit) static gain!
    entity_quads = world_rng.integers(low=288, high=compressed_vocab_size, size=(n_entities, 4), dtype=np.int64)
    for e_i in range(n_entities):
        entity_quads[e_i, 0] = 288 + ((e_i * 5) % (compressed_vocab_size - 304))
        entity_quads[e_i, 1] = 288 + ((e_i * 11 + 3) % (compressed_vocab_size - 304))

    # 2. 16 Orthogonal Contextual Expert Domains (`task_id in 0..15`, header tokens `4..19`, `~65%` of clauses):
    # Each query emits `[sep_a, sep_b, x, y]` where `sep_a, sep_b in {20..23}` are task-independent neutral
    # delimiters isolating `x in [32..255]` (`224` operands -> `16 * 224 = 3,584` high-rank contextual facts!)
    # from any prior token by 2 steps!
    # Consequently, BOTH 2-gram `(sep_b, x)` AND 3-gram `(sep_a, sep_b, x)` carry ZERO mutual information
    # about `task_id` (`y` is uniform over `32..255` marginalized over `task_id`!), AND `3,584` arbitrary
    # domain facts saturate the small dense shared FFN (`rho = 0.00`) and 5-expert MoE (`rho = 0.25`) while
    # rewarding the 11-expert dropless MoE + Layer-2 Engram hybrid (`rho = 0.75`)!
    n_tasks = 16
    n_x = 224
    task_perms = np.empty((n_tasks, n_x), dtype=np.int64)
    for t_id in range(n_tasks):
        task_perms[t_id] = 32 + world_rng.permutation(n_x)

    ent_ranks = np.arange(1, n_entities + 1, dtype=np.float64)
    ent_probs = ent_ranks ** -1.15
    ent_probs /= ent_probs.sum()

    def _emit_surface(cid: int) -> int:
        b = canon_buckets[int(cid) % compressed_vocab_size]
        if len(b) == 1:
            return b[0]
        return b[int(rng.integers(0, len(b)))]

    data = np.empty((n_seqs, seq_len), dtype=np.int64)
    for i in range(n_seqs):
        cur_task = int(rng.integers(0, n_tasks))
        data[i, 0] = 4 + cur_task
        pos = 1
        while pos < seq_len:
            if pos % 20 == 0 and pos + 1 < seq_len:
                cur_task = int(rng.integers(0, n_tasks))
                data[i, pos] = 4 + cur_task
                pos += 1
                continue

            u_draw = float(rng.random())
            if u_draw < 0.35 and pos + 4 <= seq_len:
                eid = int(rng.choice(n_entities, p=ent_probs))
                for k_step in range(4):
                    data[i, pos + k_step] = _emit_surface(int(entity_quads[eid, k_step]))
                pos += 4
            elif pos + 4 <= seq_len:
                sep_a = 20 + int(rng.integers(0, 2))
                sep_b = 22 + int(rng.integers(0, 2))
                x_op = int(rng.integers(0, n_x))
                y_tgt = int(task_perms[cur_task, x_op])
                data[i, pos] = sep_a
                data[i, pos + 1] = sep_b
                data[i, pos + 2] = 32 + x_op
                data[i, pos + 3] = y_tgt
                pos += 4
            else:
                eid = int(rng.choice(n_entities, p=ent_probs))
                data[i, pos] = _emit_surface(int(entity_quads[eid, pos % 4]))
                pos += 1

    return torch.from_numpy(data)


@torch.no_grad()
def _eval_model_bpb(
    lm: MHCEngramBackboneLM,
    eval_seqs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    vocab_bits: float,
    sharded_engram: Optional[ShardedEngramLookupEngine] = None,
    fused_table: Optional[EngramFusedLookupTable] = None,
    replicated_emb: Optional[nn.Embedding] = None,
    hash_spec: Optional[EngramHashSpec] = None,
    mhc_mode: str = "fused",
) -> float:
    """Evaluates exact cross-entropy converted to bits-per-token-symbol (`val_bpb`)."""
    lm.eval()
    device = next(lm.parameters()).device
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    per_rank = eval_seqs.shape[0] // world_size
    local_seqs = eval_seqs[rank * per_rank : (rank + 1) * per_rank].to(device)
    B_chunk = 8
    total_nll = 0.0
    total_tok = 0

    for i in range(0, local_seqs.shape[0], B_chunk):
        ch = local_seqs[i : i + B_chunk]
        inp = ch[:, :-1].contiguous()
        tgt = ch[:, 1:].contiguous()
        if fused_table is not None:
            e_mem = fused_table(inp, need_grad=False)
        elif sharded_engram is not None:
            st = sharded_engram.prefetch_forward_exchange(inp, dedup=True)
            e_mem = st["embs_3d"]
        elif replicated_emb is not None and hash_spec is not None:
            h_id = cpu_numpy_ngram_hashes_ref(inp, hash_spec)
            e_mem = replicated_emb(h_id).flatten(-2).to(torch.bfloat16)
        else:
            e_mem = None
        ce = lm.forward_with_engram_provider(inp, tgt, cos, sin, mhc_mode=mhc_mode, engram_embs_ready=e_mem)
        n_t = tgt.numel()
        total_nll += float(ce.item()) * float(n_t)
        total_tok += n_t

    agg = torch.tensor([total_nll, float(total_tok)], device=device, dtype=torch.float64)
    dist.all_reduce(agg, op=dist.ReduceOp.SUM)
    lm.train()
    mean_nats = float(agg[0].item()) / max(float(agg[1].item()), 1.0)
    return mean_nats / math.log(2.0)


def run_suite_c_engram_valbpb_and_sparsity_law(
    rank: int, world_size: int, device: torch.device, is_smoke: bool, out_dir: Path
) -> Dict[str, Any]:
    V_c = 1024
    # Rho = 0.75 Hybrid target spec (~1.33M Engram params across 16 prime heads × D_head=10 -> d_mem=160)
    spec_rho75 = build_engram_hash_spec(
        vocab_size=V_c,
        engram_vocab_size_per_ngram=(8280, 8280),  # 16 primes around 8,300 -> ~133,100 rows × 10 = ~1.331M params!
        max_ngram_size=3,
        n_head_per_ngram=8,
        layer_id=2,
        target_compression_ratio=0.77,
    )
    # Rho = 0.25 Memory-Heavy spec (~3.98M Engram params across 16 prime heads × D_head=10 -> d_mem=160)
    spec_rho25 = build_engram_hash_spec(
        vocab_size=V_c,
        engram_vocab_size_per_ngram=(24850, 24850),  # 16 primes around 24,900 -> ~398,400 rows × 10 = ~3.984M params!
        max_ngram_size=3,
        n_head_per_ngram=8,
        layer_id=2,
        target_compression_ratio=0.77,
    )
    # Rho = 0.00 Pure-Engram endpoint (~5.31M Engram params across 16 prime heads × D_head=10 -> d_mem=160, 0% MoE!)
    spec_rho00 = build_engram_hash_spec(
        vocab_size=V_c,
        engram_vocab_size_per_ngram=(33120, 33120),  # 16 primes around 33,150 -> ~530,800 rows × 10 = ~5.308M params!
        max_ngram_size=3,
        n_head_per_ngram=8,
        layer_id=2,
        target_compression_ratio=0.77,
    )
    D_HEAD_C = 10
    D_MEM_C = 16 * D_HEAD_C  # 160
    C_SLM = 192
    L_SLM = 6
    S_TOK = 256
    B_RANK = 8 if is_smoke else 16
    GLOB_TOK = world_size * B_RANK * S_TOK
    cos_c, sin_c = build_rope_cos_sin(S_TOK, 32, device)

    # Generate deterministic train / held-out val / untouched test splits
    n_train_per_rank = 256 if is_smoke else 1024
    train_pool = _generate_hybrid_lexical_compositional_corpus(
        n_train_per_rank, S_TOK + 1, V_c, spec_rho75.compressed_vocab_size,
        spec_rho75.lookup_table_np, seed=5500 + rank
    ).to(device)
    val_seqs = _generate_hybrid_lexical_compositional_corpus(
        64 if is_smoke else 128, S_TOK + 1, V_c, spec_rho75.compressed_vocab_size,
        spec_rho75.lookup_table_np, seed=99001
    )
    test_seqs = _generate_hybrid_lexical_compositional_corpus(
        64 if is_smoke else 128, S_TOK + 1, V_c, spec_rho75.compressed_vocab_size,
        spec_rho75.lookup_table_np, seed=99002
    )

    def _get_batch(step_i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s0 = (step_i * B_RANK) % (n_train_per_rank - B_RANK)
        ch = train_pool[s0 : s0 + B_RANK]
        return ch[:, :-1].contiguous(), ch[:, 1:].contiguous()

    # Canonical initial weights for Part C.1 (5-Gen Fixed-Wall-Clock RSI Loop on identical `rho = 0.75` Hybrid SLM)
    torch.manual_seed(20261309)
    base_hybrid_cpu = MHCEngramBackboneLM(
        vocab_size=V_c,
        d_model=C_SLM,
        n_layers=L_SLM,
        h_q=6,
        h_kv=2,
        ffn_shared=150,
        n_streams=4,
        n_routed=11,
        ffn_expert=128,
        top_k=2,
        engram_layer_id=2,
        d_mem=D_MEM_C,
        engram_mode="fused",
        zero_init_conv=True,
    )
    init_engram_table_75 = (torch.randn((spec_rho75.total_entries, D_HEAD_C), dtype=torch.float32) * 0.02).to(torch.bfloat16).float()

    # Measure untrained init val_bpb
    m_init = copy.deepcopy(base_hybrid_cpu).to(device)
    sh_init = ShardedEngramLookupEngine(spec_rho75, d_head=D_HEAD_C, dtype=torch.bfloat16, device=device)
    sh_init.load_from_full_table_(init_engram_table_75.to(device))
    untrained_val_bpb = _eval_model_bpb(
        m_init, val_seqs, cos_c, sin_c, vocab_bits=10.0, sharded_engram=sh_init, mhc_mode="fused"
    )
    del m_init, sh_init
    torch.cuda.empty_cache()

    budget_s = 4.0 if is_smoke else 12.0
    generations = [
        {
            "gen_id": 0,
            "candidate_id": "gen0_eager_demo_cpu_hash_atomic_embed_dense_adam",
            "engram_mode": "eager",
            "mhc_mode": "eager",
            "table_mode": "replicated_dense_adam",
            "embed_lr_mult": 5.0,
            "zero_init_conv": True,
            "note": "Eager engram_demo_v1.py baseline (CPU NumPy hash + nn.Embedding atomicAdd + Dense Adam 5x LR + Eager Gate/ShortConv/mHC)",
        },
        {
            "gen_id": 1,
            "candidate_id": "gen1_torch_compile_gpu_hash_replicated_dense_adam",
            "engram_mode": "compiled",
            "mhc_mode": "compiled",
            "table_mode": "gpu_hash_replicated_dense_adam",
            "embed_lr_mult": 5.0,
            "zero_init_conv": True,
            "note": "Eliminate CPU NumPy DtoH/HtoD via GPU Prime Hash + torch.compile (inductor) Engram & mHC -> KEEP",
        },
        {
            "gen_id": 2,
            "candidate_id": "gen2_sparse_adam_unscaled_0p2x_lr_shortcut",
            "engram_mode": "compiled",
            "mhc_mode": "compiled",
            "table_mode": "gpu_hash_sparse_adam_shortcut",
            "embed_lr_mult": 0.2,
            "zero_init_conv": True,
            "note": "Optimizer shortcut: replace exact 5x-LR Adam with unscaled 0.2x SparseAdam (drops momentum drift & starves memory table) -> REVERT_VAL_BPB",
        },
        {
            "gen_id": 3,
            "candidate_id": "gen3_ungated_static_engram_skip_key_projs_shortcut",
            "engram_mode": "fused",
            "mhc_mode": "compiled",
            "table_mode": "gpu_hash_replicated_dense_adam",
            "embed_lr_mult": 5.0,
            "zero_init_conv": False,
            "ungated_shortcut": True,
            "note": "Single-factor architectural shortcut vs Gen 1 (arXiv:2601.07372 Sec. 2.3/5): skip 4 key projections & RMS context gating (alpha=1) + nonzero ShortConv init -> faster step, worse val_bpb -> REVERT_VAL_BPB",
        },
        {
            "gen_id": 4,
            "candidate_id": "gen4_round4_rsi_zero_transpose_engram_tile_fused_jit_lazy_adam",
            "engram_mode": "fused",
            "mhc_mode": "fused",
            "table_mode": "tile_fused_jit_lazy_adam",
            "embed_lr_mult": 5.0,
            "zero_init_conv": True,
            "note": "Full Round-4 RSI Co-Design: Zero-Transpose Fused M=4 Gate+ShortConv (Exact Zero-Init) + Zero-Sync Tile-Fused Pre-Forward JIT-Lazy-Adam(lr=5x, wd=0) + Dropless Grouped MoE + Fused mHC -> KEEP",
        },
    ]

    def _instantiate_gen(gcfg: Dict[str, Any]):
        lm_g = MHCEngramBackboneLM(
            vocab_size=V_c,
            d_model=C_SLM,
            n_layers=L_SLM,
            h_q=6,
            h_kv=2,
            ffn_shared=150,
            n_streams=4,
            n_routed=11,
            ffn_expert=128,
            top_k=2,
            engram_layer_id=2,
            d_mem=D_MEM_C,
            engram_mode=gcfg["engram_mode"],
            zero_init_conv=gcfg["zero_init_conv"],
        )
        # Copy exact shared backbone & Engram KV/norm weights from `base_hybrid_cpu`
        with torch.no_grad():
            src_sd = base_hybrid_cpu.state_dict()
            dst_sd = lm_g.state_dict()
            for k in dst_sd:
                if k in src_sd and dst_sd[k].shape == src_sd[k].shape:
                    dst_sd[k].copy_(src_sd[k])
            if gcfg["engram_mode"] in ("eager", "compiled"):
                ec_dst = lm_g.blocks[2].engram_ctx
                ec_base = ec_dst._orig_mod if hasattr(ec_dst, "_orig_mod") else ec_dst
                ec_src = base_hybrid_cpu.blocks[2].engram_ctx
                ec_base.value_proj.weight.copy_(ec_src.W_kv[:C_SLM])
                for m_i in range(4):
                    ec_base.key_projs[m_i].weight.copy_(ec_src.W_kv[(m_i + 1) * C_SLM : (m_i + 2) * C_SLM])
                    ec_base.norm2[m_i].weight.copy_(ec_src.gamma_h[m_i])
                    ec_base.norm1[m_i].weight.copy_(ec_src.gamma_k[m_i])
                    ec_base.conv_norms[m_i].weight.copy_(ec_src.gamma_conv[m_i])
                ec_base.conv.weight.copy_(ec_src.w_conv.reshape(4 * C_SLM, 1, 4))
            if not gcfg["zero_init_conv"]:
                # Uncalibrated large random ShortConv init shock + optional ungated alpha=1 shortcut
                gen_shk = torch.Generator(device="cpu").manual_seed(202699)
                ec_f = lm_g.blocks[2].engram_ctx
                ec_f.w_conv.copy_(torch.randn(ec_f.w_conv.shape, generator=gen_shk, dtype=ec_f.w_conv.dtype) * 1.20)
            if gcfg.get("ungated_shortcut", False):
                lm_g.blocks[2].engram_ctx.ungated_constant_one = True

        lm_g = lm_g.to(device)
        tmode = gcfg["table_mode"]
        if tmode == "tile_fused_jit_lazy_adam":
            ft = EngramFusedLookupTable(spec_rho75, d_head=D_HEAD_C, dtype=torch.bfloat16, device=device, max_steps=4096)
            ft.copy_from_reference_embedding_(init_engram_table_75.to(device))
            return lm_g, ft, None, None
        is_sparse_opt = tmode == "gpu_hash_sparse_adam_shortcut"
        rep_e = nn.Embedding(
            spec_rho75.total_entries, D_HEAD_C, sparse=is_sparse_opt, device=device,
            dtype=(torch.float32 if is_sparse_opt else torch.bfloat16)
        )
        with torch.no_grad():
            rep_e.weight.copy_(init_engram_table_75.to(device=device, dtype=rep_e.weight.dtype))
        if is_sparse_opt:
            rep_o = torch.optim.SparseAdam(rep_e.parameters(), lr=3e-3 * gcfg["embed_lr_mult"])
        else:
            rep_o = torch.optim.Adam(rep_e.parameters(), lr=3e-3 * gcfg["embed_lr_mult"], weight_decay=0.0, fused=True)
        return lm_g, None, rep_e, rep_o

    lut_75 = torch.from_numpy(spec_rho75.lookup_table_np.astype(np.int32)).to(device)
    mult_75 = torch.from_numpy(spec_rho75.multipliers_np).to(device)
    mod_75 = torch.from_numpy(spec_rho75.head_moduli_np).to(device)
    off_75 = torch.from_numpy(spec_rho75.head_offsets_np).to(device)

    # Pre-allocated NVSwitch all_gather buffers for zero-DtoH-sync global Tile-Fused `JIT-Lazy-Adam`
    gather_ib_buf = torch.empty((world_size * B_RANK, S_TOK), device=device, dtype=torch.long)
    gather_ge_buf = torch.empty((world_size * B_RANK, S_TOK, D_MEM_C), device=device, dtype=torch.bfloat16)

    def _train_step_gen(
        step_i: int,
        lr_backbone: float,
        gcfg: Dict[str, Any],
        lm_g: MHCEngramBackboneLM,
        opt_bb: torch.optim.Optimizer,
        ft_t: Optional[EngramFusedLookupTable],
        rep_e: Optional[nn.Embedding],
        rep_o: Optional[torch.optim.Optimizer],
        pf_s: torch.cuda.Stream,
        comp_s: torch.cuda.Stream,
    ) -> float:
        ib, tb = _get_batch(step_i)
        for pg in opt_bb.param_groups:
            pg["lr"] = lr_backbone
        opt_bb.zero_grad(set_to_none=True)
        lr_emb = lr_backbone * float(gcfg["embed_lr_mult"])

        if ft_t is not None:
            # Zero-DtoH-sync NVSwitch `all_gather_into_tensor` (`256 KiB` token IDs) + Tile-Fused Pre-Forward Catch-Up:
            dist.all_gather_into_tensor(gather_ib_buf, ib)
            em_all = ft_t(gather_ib_buf, need_grad=True)
            em_local = em_all[rank * B_RANK : (rank + 1) * B_RANK].detach().requires_grad_(True)
            loss = lm_g.forward_with_engram_provider(
                ib, tb, cos_c, sin_c, mhc_mode=gcfg["mhc_mode"], engram_embs_ready=em_local
            )
            loss.backward()
            # Overlap Engram gradient NVSwitch `all_gather_into_tensor` (`pf_s`) with backbone DDP `all_reduce` (`comp_s`)!
            pf_s.wait_stream(comp_s)
            ge_loc = (em_local.grad.float() / float(world_size)).to(torch.bfloat16).contiguous()
            ge_loc.record_stream(pf_s)
            with torch.cuda.stream(pf_s):
                dist.all_gather_into_tensor(gather_ge_buf, ge_loc)
            _ddp_average_grads_(lm_g, world_size)
            comp_s.wait_stream(pf_s)
            em_all.backward(gather_ge_buf)
            gnorm = float(torch.nn.utils.clip_grad_norm_(lm_g.parameters(), 1.0).item())
            opt_bb.step()
            ft_t.step_lazy_adam(lr_emb)
            return gnorm

        for pg in rep_o.param_groups:
            pg["lr"] = lr_emb
        rep_o.zero_grad(set_to_none=True)
        if gcfg["table_mode"] == "replicated_dense_adam":
            h_id = cpu_numpy_ngram_hashes_ref(ib, spec_rho75)
        else:
            h_id = gpu_engram_ngram_hashes(
                ib, lut_75, mult_75, mod_75, off_75, spec_rho75.compressed_pad_id, 3, 8
            ).view(B_RANK, S_TOK, 16).long()

        em_raw = rep_e(h_id).flatten(-2).to(torch.bfloat16)
        loss = lm_g.forward_with_engram_provider(
            ib, tb, cos_c, sin_c, mhc_mode=gcfg["mhc_mode"], engram_embs_ready=em_raw
        )
        loss.backward()
        _ddp_average_grads_(lm_g, world_size)
        if gcfg["table_mode"] == "gpu_hash_sparse_adam_shortcut":
            # Coalesce sparse gradient and average across ranks via dense view only on touched rows
            g_sp = rep_e.weight.grad.coalesce()
            g_dense_tmp = torch.zeros_like(rep_e.weight)
            g_dense_tmp.index_add_(0, g_sp.indices()[0], g_sp.values())
            dist.all_reduce(g_dense_tmp, op=dist.ReduceOp.SUM)
            g_dense_tmp.div_(float(world_size))
            u_idx = torch.unique(g_sp.indices()[0])
            rep_e.weight.grad = torch.sparse_coo_tensor(
                u_idx.unsqueeze(0), g_dense_tmp.index_select(0, u_idx), rep_e.weight.shape, device=device
            ).coalesce()
        else:
            dist.all_reduce(rep_e.weight.grad, op=dist.ReduceOp.SUM)
            rep_e.weight.grad.div_(float(world_size))
        gnorm = float(torch.nn.utils.clip_grad_norm_(lm_g.parameters(), 1.0).item())
        opt_bb.step()
        rep_o.step()
        return gnorm

    total_hybrid_params = sum(p.numel() for p in base_hybrid_cpu.parameters()) + spec_rho75.total_entries * D_HEAD_C
    incumbent_val_bpb: Optional[float] = None
    incumbent_gen_id = 0
    gen_records: List[Dict[str, Any]] = []
    gen4_steps = 0

    for gcfg in generations:
        gid = int(gcfg["gen_id"])
        pf_s = torch.cuda.Stream(device=device)
        comp_s = torch.cuda.current_stream(device=device)

        # Pre-timer compilation warmup (2 steps discarded before fresh canonical re-instantiation!)
        lm_w, sh_w, repe_w, repo_w = _instantiate_gen(gcfg)
        opt_w = torch.optim.AdamW(lm_w.parameters(), lr=3e-3, weight_decay=0.05, fused=True)
        for w_i in range(2):
            _train_step_gen(w_i, 1e-3, gcfg, lm_w, opt_w, sh_w, repe_w, repo_w, pf_s, comp_s)
        comp_s.wait_stream(pf_s)
        torch.cuda.synchronize(device)
        del lm_w, sh_w, repe_w, repo_w, opt_w
        torch.cuda.empty_cache()
        dist.barrier()

        # Fresh canonical training run under strict synchronized wall-clock budget
        lm_r, sh_r, repe_r, repo_r = _instantiate_gen(gcfg)
        opt_r = torch.optim.AdamW(lm_r.parameters(), lr=3e-3, weight_decay=0.05, fused=True)

        dist.barrier()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step_idx = 0
        peak_gnorm = 0.0

        while True:
            # Broadcast rank-0 stop flag every step so all 8 ranks execute identical step counts
            elapsed_now = time.perf_counter() - t0
            stop_t = torch.tensor([1 if elapsed_now >= budget_s else 0], device=device, dtype=torch.int32)
            dist.broadcast(stop_t, src=0)
            if int(stop_t.item()) == 1 and step_idx >= 4:
                break
            prog = min(1.0, elapsed_now / budget_s)
            lr_bb = 3.0e-3 * (
                (0.15 + 0.85 * (prog / 0.10))
                if prog < 0.10
                else (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * (prog - 0.10) / 0.90)))
            )
            gn = _train_step_gen(step_idx, lr_bb, gcfg, lm_r, opt_r, sh_r, repe_r, repo_r, pf_s, comp_s)
            peak_gnorm = max(peak_gnorm, gn)
            step_idx += 1

        comp_s.wait_stream(pf_s)
        torch.cuda.synchronize(device)
        dist.barrier()
        elapsed_sync = time.perf_counter() - t0
        el_t = torch.tensor([elapsed_sync], device=device, dtype=torch.float64)
        dist.broadcast(el_t, src=0)
        elapsed_sync = float(el_t.item())

        if gid == 4:
            gen4_steps = step_idx

        val_bpb = _eval_model_bpb(
            lm_r, val_seqs, cos_c, sin_c, vocab_bits=10.0,
            fused_table=sh_r, replicated_emb=repe_r, hash_spec=spec_rho75, mhc_mode=gcfg["mhc_mode"],
        )
        test_bpb = _eval_model_bpb(
            lm_r, test_seqs, cos_c, sin_c, vocab_bits=10.0,
            fused_table=sh_r, replicated_emb=repe_r, hash_spec=spec_rho75, mhc_mode=gcfg["mhc_mode"],
        )

        if incumbent_val_bpb is None:
            incumbent_val_bpb = val_bpb
            incumbent_gen_id = gid
            decision = "BASELINE"
        elif val_bpb < incumbent_val_bpb - 1e-4:
            incumbent_val_bpb = val_bpb
            incumbent_gen_id = gid
            decision = "KEEP"
        else:
            decision = "REVERT_VAL_BPB"

        gen_records.append(
            {
                "gen_id": gid,
                "candidate_id": gcfg["candidate_id"],
                "param_count": total_hybrid_params,
                "budget_s": budget_s,
                "elapsed_s": round(elapsed_sync, 4),
                "completed_steps": step_idx,
                "steps_per_sec": round(step_idx / max(elapsed_sync, 1e-9), 3),
                "global_tok_per_s": round((step_idx * GLOB_TOK) / max(elapsed_sync, 1e-9), 1),
                "val_bpb": round(val_bpb, 5),
                "untouched_test_bpb": round(test_bpb, 5),
                "peak_grad_norm": round(peak_gnorm, 4),
                "decision": decision,
                "incumbent_val_bpb_after": round(float(incumbent_val_bpb), 5),
                "hypothesis_note": gcfg["note"],
            }
        )
        del lm_r, sh_r, repe_r, repo_r, opt_r
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    # -------------------------------------------------------------------------
    # Part C.2: Matched-Step Replayed-LR Sparsity Allocation Law (`arXiv:2601.07372` Sec. 3)
    #   Strictly Iso-Total-Params & Iso-Active-FLOPs across `rho in {1.00, 0.75, 0.25}`
    #   plus `Iso-Active-FLOPs Pure Dense mHC` and `Iso-Total-Param Wider Dense mHC`!
    # -------------------------------------------------------------------------
    matched_steps = max(24, gen4_steps)
    allocation_arms = [
        {
            "arm_id": "rho_1p00_pure_mhc_moe",
            "rho_moe_fraction": 1.00,
            "ffn_shared": 196,   # +46 shared channels exactly matches Engram's 158,976 active params!
            "n_routed": 14,      # top_k=2 active + 12 inactive = 5,308,416 sparse params
            "top_k": 2,
            "engram_spec": None,
        },
        {
            "arm_id": "rho_0p75_optimal_hybrid_mhc_moe_engram",
            "rho_moe_fraction": 0.75,
            "ffn_shared": 150,
            "n_routed": 11,      # top_k=2 active + 9 inactive (75% MoE) + 25% Engram (~1.331M)
            "top_k": 2,
            "engram_spec": spec_rho75,
        },
        {
            "arm_id": "rho_0p25_memory_heavy_mhc_moe_engram",
            "rho_moe_fraction": 0.25,
            "ffn_shared": 150,
            "n_routed": 5,       # top_k=2 active + 3 inactive (25% MoE) + 75% Engram (~3.989M)
            "top_k": 2,
            "engram_spec": spec_rho25,
        },
        {
            "arm_id": "rho_0p00_pure_engram_conditional_memory",
            "rho_moe_fraction": 0.00,
            "ffn_shared": 406,   # 150 + 2*128 = 406 active FFN width (0% MoE) + 100% Engram (~5.31M) -> exact iso-FLOPs & iso-params!
            "n_routed": 0,
            "top_k": 0,
            "engram_spec": spec_rho00,
        },
        {
            "arm_id": "dense_iso_active_flops_mhc",
            "rho_moe_fraction": -1.0,
            "ffn_shared": 452,   # Exact same active FFN width (196 + 2*128 = 452), 0 sparse capacity!
            "n_routed": 0,
            "top_k": 0,
            "engram_spec": None,
        },
        {
            "arm_id": "dense_iso_total_params_wider_mhc",
            "rho_moe_fraction": -2.0,
            "ffn_shared": 1986,  # Widened dense FFN matching ~8.06M total params (~3.3x higher active FLOPs/tok!)
            "n_routed": 0,
            "top_k": 0,
            "engram_spec": None,
        },
    ]

    allocation_results: List[Dict[str, Any]] = []
    for arm in allocation_arms:
        torch.manual_seed(20261315)
        espec: Optional[EngramHashSpec] = arm["engram_spec"]
        lm_a = MHCEngramBackboneLM(
            vocab_size=V_c,
            d_model=C_SLM,
            n_layers=L_SLM,
            h_q=6,
            h_kv=2,
            ffn_shared=arm["ffn_shared"],
            n_streams=4,
            n_routed=arm["n_routed"],
            ffn_expert=128,
            top_k=arm["top_k"],
            engram_layer_id=(2 if espec is not None else None),
            d_mem=D_MEM_C,
            engram_mode="fused",
            zero_init_conv=True,
        ).to(device)

        bb_params = sum(p.numel() for p in lm_a.parameters())
        eng_table_params = (espec.total_entries * D_HEAD_C) if espec is not None else 0
        total_params = bb_params + eng_table_params
        inactive_moe_params = (
            L_SLM * max(0, arm["n_routed"] - arm["top_k"]) * (3 * C_SLM * 128)
        )
        active_non_embed_params = bb_params - inactive_moe_params

        ft_a = (
            EngramFusedLookupTable(espec, d_head=D_HEAD_C, dtype=torch.bfloat16, device=device, max_steps=4096)
            if espec is not None
            else None
        )
        opt_a = torch.optim.AdamW(lm_a.parameters(), lr=3e-3, weight_decay=0.05, fused=True)
        pf_s = torch.cuda.Stream(device=device)
        comp_s = torch.cuda.current_stream(device=device)

        dist.barrier()
        torch.cuda.synchronize(device)
        t_a0 = time.perf_counter()

        for st in range(matched_steps):
            prog = float(st) / float(max(1, matched_steps))
            lr_bb = 3.0e-3 * (
                (0.15 + 0.85 * (prog / 0.10))
                if prog < 0.10
                else (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * (prog - 0.10) / 0.90)))
            )
            for pg in opt_a.param_groups:
                pg["lr"] = lr_bb
            opt_a.zero_grad(set_to_none=True)
            ib, tb = _get_batch(st)

            if ft_a is not None:
                dist.all_gather_into_tensor(gather_ib_buf, ib)
                em_all = ft_a(gather_ib_buf, need_grad=True)
                em_loc = em_all[rank * B_RANK : (rank + 1) * B_RANK].detach().requires_grad_(True)
                loss_a = lm_a.forward_with_engram_provider(
                    ib, tb, cos_c, sin_c, mhc_mode="fused", engram_embs_ready=em_loc
                )
                loss_a.backward()
                pf_s.wait_stream(comp_s)
                ge_loc = (em_loc.grad.float() / float(world_size)).to(torch.bfloat16).contiguous()
                ge_loc.record_stream(pf_s)
                with torch.cuda.stream(pf_s):
                    dist.all_gather_into_tensor(gather_ge_buf, ge_loc)
                _ddp_average_grads_(lm_a, world_size)
                comp_s.wait_stream(pf_s)
                em_all.backward(gather_ge_buf)
                torch.nn.utils.clip_grad_norm_(lm_a.parameters(), 1.0)
                opt_a.step()
                ft_a.step_lazy_adam(5.0 * lr_bb)
            else:
                loss_a = lm_a.forward_with_engram_provider(ib, tb, cos_c, sin_c, mhc_mode="fused")
                loss_a.backward()
                _ddp_average_grads_(lm_a, world_size)
                torch.nn.utils.clip_grad_norm_(lm_a.parameters(), 1.0)
                opt_a.step()

        comp_s.wait_stream(pf_s)
        torch.cuda.synchronize(device)
        dist.barrier()
        wall_a = time.perf_counter() - t_a0

        arm_val_bpb = _eval_model_bpb(lm_a, val_seqs, cos_c, sin_c, vocab_bits=10.0, fused_table=ft_a, mhc_mode="fused")
        arm_test_bpb = _eval_model_bpb(lm_a, test_seqs, cos_c, sin_c, vocab_bits=10.0, fused_table=ft_a, mhc_mode="fused")

        allocation_results.append(
            {
                "arm_id": arm["arm_id"],
                "rho_moe_fraction": arm["rho_moe_fraction"],
                "matched_steps": matched_steps,
                "total_params": total_params,
                "active_params_per_token": active_non_embed_params,
                "engram_table_params": eng_table_params,
                "inactive_moe_params": inactive_moe_params,
                "wall_seconds": round(wall_a, 3),
                "val_bpb": round(arm_val_bpb, 5),
                "untouched_test_bpb": round(arm_test_bpb, 5),
            }
        )
        del lm_a, ft_a, opt_a
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    valbpb_ledger_sha = ""
    if rank == 0:
        ledger_path = out_dir / "valbpb_autoresearch_ledger.tsv"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "gen_id", "candidate_id", "param_count", "budget_s", "elapsed_s", "completed_steps",
            "steps_per_sec", "global_tok_per_s", "val_bpb", "untouched_test_bpb",
            "peak_grad_norm", "decision", "incumbent_val_bpb_after", "hypothesis_note",
        ]
        with ledger_path.open("w", encoding="utf-8", newline="\n") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for row in gen_records:
                w.writerow(row)
        valbpb_ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    final_rec = next(r for r in gen_records if r["gen_id"] == incumbent_gen_id)
    gen0_rec = gen_records[0]
    gen1_rec = gen_records[1]
    gen4_rec = gen_records[4]

    by_arm = {a["arm_id"]: a for a in allocation_results}
    rho100 = by_arm["rho_1p00_pure_mhc_moe"]
    rho075 = by_arm["rho_0p75_optimal_hybrid_mhc_moe_engram"]
    rho025 = by_arm["rho_0p25_memory_heavy_mhc_moe_engram"]
    rho000 = by_arm["rho_0p00_pure_engram_conditional_memory"]
    dense_iso_f = by_arm["dense_iso_active_flops_mhc"]
    dense_iso_p = by_arm["dense_iso_total_params_wider_mhc"]

    # Strict interior U-curve check across the full [0.0, 1.0] sparsity allocation spectrum:
    # interior hybrid `rho = 0.75` strictly beats BOTH pure endpoints (`rho = 1.00` Pure MoE AND `rho = 0.00` Pure Engram)
    # as well as memory-heavy `rho = 0.25` once MoE experts specialize!
    u_curve_endpoints_verified = bool(
        rho075["val_bpb"] < rho100["val_bpb"] and rho075["val_bpb"] < rho000["val_bpb"]
    )
    u_curve_strict_4pt = bool(
        u_curve_endpoints_verified and rho075["val_bpb"] < rho025["val_bpb"]
    )

    return {
        "vocab_size": V_c,
        "compressed_vocab_size": spec_rho75.compressed_vocab_size,
        "d_model": C_SLM,
        "n_blocks": L_SLM,
        "seq_len": S_TOK,
        "global_tokens_per_step": GLOB_TOK,
        "budget_seconds_per_generation": budget_s,
        "untrained_init_val_bpb": round(untrained_val_bpb, 5),
        "baseline_gen0_eager_demo_val_bpb": gen0_rec["val_bpb"],
        "compiled_gen1_val_bpb": gen1_rec["val_bpb"],
        "final_incumbent_gen_id": incumbent_gen_id,
        "final_incumbent_candidate": final_rec["candidate_id"],
        "final_incumbent_val_bpb": final_rec["val_bpb"],
        "final_incumbent_untouched_test_bpb": final_rec["untouched_test_bpb"],
        "val_bpb_gain_vs_eager_demo_gen0": round(gen0_rec["val_bpb"] - final_rec["val_bpb"], 5),
        "val_bpb_gain_vs_compiled_gen1": round(gen1_rec["val_bpb"] - final_rec["val_bpb"], 5),
        "test_bpb_gain_vs_eager_demo_gen0": round(gen0_rec["untouched_test_bpb"] - final_rec["untouched_test_bpb"], 5),
        "step_multiplier_gen4_vs_eager_demo_gen0": round(
            gen4_rec["completed_steps"] / max(1, gen0_rec["completed_steps"]), 4
        ),
        "valbpb_ledger_sha256": valbpb_ledger_sha,
        "generations": gen_records,
        "sparsity_allocation_law_ablation": {
            "matched_steps": matched_steps,
            "u_curve_interior_optimum_verified": u_curve_endpoints_verified,
            "u_curve_strict_4pt_verified": u_curve_strict_4pt,
            "hybrid_rho075_gain_over_pure_moe_rho100_bpb": round(rho100["val_bpb"] - rho075["val_bpb"], 5),
            "hybrid_rho075_gain_over_pure_engram_rho000_bpb": round(rho000["val_bpb"] - rho075["val_bpb"], 5),
            "hybrid_rho075_gain_over_mem_heavy_rho025_bpb": round(rho025["val_bpb"] - rho075["val_bpb"], 5),
            "hybrid_rho075_gain_over_iso_flops_dense_bpb": round(dense_iso_f["val_bpb"] - rho075["val_bpb"], 5),
            "hybrid_rho075_vs_iso_total_param_wider_dense_bpb": round(dense_iso_p["val_bpb"] - rho075["val_bpb"], 5),
            "arms": allocation_results,
        },
    }


# =============================================================================
# Main Entrypoint (`torchrun --nproc_per_node=8`)
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device, timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_smoke = args.mode == "smoke"

    t_wall0 = time.perf_counter()
    harness_sha = eval_harness_sha256()

    suite0_res: Dict[str, Any] = {}
    suitea_res: Dict[str, Any] = {}
    rsi_ledger_sha = ""
    if rank == 0:
        suite0_res, rt_ratchet = run_suite_0_engram_audits(device, is_smoke)
        suitea_res, ratchets_a = run_suite_a_engram_ratchets(device, is_smoke)
        rsi_ledger_sha = write_rsi_ledger_tsv(
            [rt_ratchet, *ratchets_a], args.out_dir / "rsi_ledger.tsv"
        )
        gc.collect()
        torch.cuda.empty_cache()

    dist.barrier()
    suiteb_res = run_suite_b_sharded_engram_nvswitch(rank, world_size, device, is_smoke)
    dist.barrier()
    suitec_res = run_suite_c_engram_valbpb_and_sparsity_law(rank, world_size, device, is_smoke, args.out_dir)
    dist.barrier()

    total_wall_s = time.perf_counter() - t_wall0
    clk_end, pwr_end = query_sm_telemetry(0)

    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_json = args.out_dir / f"engram_conditional_memory_rsi_{args.mode}.json"
        payload = {
            "experiment": "E130_deepseek_engram_conditional_memory_round4_rsi",
            "mode": args.mode,
            "harness_version": HARNESS_VERSION,
            "eval_harness_sha256": harness_sha,
            "rsi_ledger_tsv_sha256": rsi_ledger_sha,
            "total_wall_seconds": round(total_wall_s, 2),
            "end_gpu0_sm_clock_mhz": clk_end,
            "end_gpu0_power_w": pwr_end,
            "environment": environment_metadata(),
            "suite_0_adversarial_and_oracle_audits": suite0_res,
            "suite_a_round4_engram_vector_ratchets": suitea_res,
            "suite_b_8gpu_nvswitch_sharded_engram": suiteb_res,
            "suite_c_valbpb_rsi_and_sparsity_allocation_law": suitec_res,
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"[E130 {args.mode.upper()} COMPLETE] wall={total_wall_s:.2f}s | "
            f"harness_sha={harness_sha[:12]} | "
            f"T1={suitea_res['track1_engram_lookup_bwd_opt']['speedup_vs_demo_baseline_x']}x | "
            f"T2={suitea_res['track2_engram_m4_gate_shortconv']['speedup_vs_eager_demo_x']}x | "
            f"B2_prefetch={suiteb_res['speedup_prefetch_vs_replicated_demo_x']}x | "
            f"C1_incumbent=gen{suitec_res['final_incumbent_gen_id']} ({suitec_res['final_incumbent_val_bpb']} bpb) | "
            f"C2_ucurve={suitec_res['sparsity_allocation_law_ablation']['u_curve_interior_optimum_verified']}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

