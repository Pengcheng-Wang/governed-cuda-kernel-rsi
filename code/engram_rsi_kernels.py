#!/usr/bin/env python3
"""
E130 (Round-4 RSI): DeepSeek Engram Conditional Memory Kernels & Sharded Runtime
================================================================================
Source-verified reproduction and Recursive Self-Improvement (RSI) kernel stack for:
  - DeepSeek-AI & Peking University, "Conditional Memory via Scalable Lookup:
    A New Axis of Sparsity for Large Language Models" (Engram, arXiv:2601.07372, Jan 2026)
  - Official reference: github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
  - Paired directly with M=4 Manifold-Constrained Hyper-Connections (mHC, arXiv:2512.24880)
    from code/mhc_rsi_kernels.py (E129).

Four coupled GPU kernel & distributed systems bottlenecks solved:
  1. Track 1A (Compressed Multiplicative-XOR Prime N-Gram Hashing & Gather):
     Eliminates engram_demo_v1.py's synchronous CPU NumPy `_get_ngram_hashes` +
     `DtoH/HtoD` stall via a single GPU Triton int64 hash+gather kernel with
     surjective vocabulary canonicalization P: V -> V' (~23% reduction) and
     document-boundary (`doc_ids`) isolation for packed variable-length sequences.
  2. Track 1B (Atomic-Free CSR Segment-Reduce Backward + Pre-Forward JIT-Lazy-Adam):
     Eliminates nn.Embedding's `[total_N, D_head]` dense gradient allocation and
     severe Zipfian hot-row `atomicAdd` contention via GPU radix-sort + CSR segment
     reduction (`0` atomics), paired with an exact dense-equivalent `Adam(lr=5x, wd=0)`
     Just-In-Time Pre-Forward Momentum Catch-Up kernel (`< 1e-6` multi-step parity vs
     full-table dense Adam, unlike `SparseAdam` which drops momentum drift on absent steps).
  3. Track 2 (Merged-KV Zero-Transpose Multi-Branch Context Gating + Dilated ShortConv):
     Collapses `engram_demo_v1.py`'s 5 separate KV GEMMs, 8 gate RMSNorms, 4 ShortConv
     RMSNorms, 2 `[B, T, M*C] <-> [B, M*C, T]` layout transposes, cuDNN grouped Conv1d,
     SiLU, and dual residuals (`H + U + ShortConv(U)`) into 1 merged KV GEMM + 2 token-major
     Triton forward kernels and 2 lock-free (`0` atomics, `0` transposes, `0` spills)
     Split-CTA Triton backward kernels supporting both `demo_signed_sqrt` and `paper_eq6_smooth`.
  4. Distributed 8-GPU Sharded Engram (`Dedup-AllToAllv` + Layer-2 Overlapped Prefetch):
     Modulo-balanced (`owner = z % W`) 8-GPU NVSwitch table sharding with local Zipfian
     unique-key deduplication before `all_to_all_single` (`3.5x-6x` comm reduction in
     both fwd and bwd) and allocator-safe (`record_stream`) async prefetch behind
     mHC Transformer Layers 0..1.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# =============================================================================
# 1. Surjective Vocabulary Canonicalizer & Deterministic Prime Hash Metadata
# =============================================================================


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def find_next_prime(start: int, seen_primes: set) -> int:
    """Exact prime search matching DeepSeek engram_demo_v1.py."""
    cand = int(start) + 1
    while True:
        if _is_prime(cand) and cand not in seen_primes:
            return cand
        cand += 1


def canonicalize_token_piece(piece: str) -> str:
    """
    Surjective normalization recipe P: V -> V' from arXiv:2601.07372 Sec. 2.2
    & engram_demo_v1.py: NFKC -> NFD -> StripAccents (Mn) -> Lowercase -> Whitespace collapse.
    """
    s = unicodedata.normalize("NFKC", piece)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = " ".join(s.strip().split())
    return s


def build_surjective_vocab_lookup(
    vocab_size: int,
    target_ratio: float = 0.77,
    pad_id: int = 2,
    seed: int = 202601,
    token_pieces: Optional[List[str]] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Constructs surjective lookup_table: [vocab_size] -> [0 .. compressed_vocab_size - 1].
    If `token_pieces` is supplied, applies exact Unicode NFKC/NFD/accent-strip/lowercase
    surjective canonicalization; otherwise builds a deterministic ~23%-colliding
    synthetic vocabulary of case/whitespace/diacritic variants achieving `target_ratio`.
    Returns `(lookup_table_np_int64, compressed_vocab_size, compressed_pad_id)`.
    """
    if token_pieces is not None:
        assert len(token_pieces) == vocab_size
        canon_to_id: Dict[str, int] = {}
        lut = np.zeros((vocab_size,), dtype=np.int64)
        for tid, raw in enumerate(token_pieces):
            key = canonicalize_token_piece(raw) if tid >= 4 else f"<special_{tid}>"
            cid = canon_to_id.get(key)
            if cid is None:
                cid = len(canon_to_id)
                canon_to_id[key] = cid
            lut[tid] = cid
        comp_pad = int(lut[pad_id])
        return lut, len(canon_to_id), comp_pad

    # Deterministic synthetic byte/word vocabulary with realistic ~23% equivalence collisions
    # (e.g., "The", " the", "THE", "thé" -> "the") while keeping special tokens 0..3 distinct.
    comp_v = max(8, int(round(vocab_size * target_ratio)))
    lut = np.empty((vocab_size,), dtype=np.int64)
    # First `comp_v` tokens define canonical representatives 0..comp_v-1
    lut[:comp_v] = np.arange(comp_v, dtype=np.int64)
    if vocab_size > comp_v:
        rng = np.random.default_rng(seed)
        # Collide surface variants onto non-special canonical word stems [4 .. comp_v - 1]
        lut[comp_v:] = rng.integers(low=4, high=comp_v, size=(vocab_size - comp_v,), dtype=np.int64)
    comp_pad = int(lut[pad_id])
    return lut, comp_v, comp_pad


@dataclass
class EngramHashSpec:
    vocab_size: int
    compressed_vocab_size: int
    pad_id: int
    compressed_pad_id: int
    max_ngram_size: int
    n_head_per_ngram: int
    layer_id: int
    lookup_table_np: np.ndarray          # [vocab_size] int64
    multipliers_np: np.ndarray           # [max_ngram_size] int64 odd multipliers
    head_moduli_np: np.ndarray           # [num_heads] int64 distinct primes
    head_offsets_np: np.ndarray          # [num_heads] int64 prefix offsets
    total_entries: int

    @property
    def num_heads(self) -> int:
        return (self.max_ngram_size - 1) * self.n_head_per_ngram


def build_engram_hash_spec(
    vocab_size: int = 16384,
    engram_vocab_size_per_ngram: Tuple[int, ...] = (32768, 32768),
    max_ngram_size: int = 3,
    n_head_per_ngram: int = 8,
    layer_id: int = 2,
    all_layer_ids: Tuple[int, ...] = (2,),
    pad_id: int = 2,
    seed: int = 0,
    target_compression_ratio: float = 0.77,
    token_pieces: Optional[List[str]] = None,
) -> EngramHashSpec:
    """
    Constructs exact prime-modulo multiplicative-XOR hash metadata matching
    `engram_demo_v1.py` `NgramHashMapping` byte-for-byte.
    """
    assert len(engram_vocab_size_per_ngram) == max_ngram_size - 1
    lut_np, comp_v, comp_pad = build_surjective_vocab_lookup(
        vocab_size=vocab_size,
        target_ratio=target_compression_ratio,
        pad_id=pad_id,
        token_pieces=token_pieces,
    )

    max_long = int(np.iinfo(np.int64).max)
    m_max = int(max_long // comp_v)
    half_bound = max(1, m_max // 2)
    prime_1 = 10007

    layer_multipliers: Dict[int, np.ndarray] = {}
    for lid in all_layer_ids:
        base_seed = int(seed + prime_1 * int(lid))
        g = np.random.default_rng(base_seed)
        r = g.integers(low=0, high=half_bound, size=(max_ngram_size,), dtype=np.int64)
        layer_multipliers[lid] = r * 2 + 1

    seen_primes: set = set()
    layer_head_primes: Dict[int, List[int]] = {}
    for lid in all_layer_ids:
        heads_this_layer: List[int] = []
        for ng_idx in range(max_ngram_size - 1):
            cur = int(engram_vocab_size_per_ngram[ng_idx]) - 1
            for _ in range(n_head_per_ngram):
                p = find_next_prime(cur, seen_primes)
                seen_primes.add(p)
                heads_this_layer.append(p)
                cur = p
        layer_head_primes[lid] = heads_this_layer

    moduli = np.asarray(layer_head_primes[layer_id], dtype=np.int64)
    offsets = np.zeros_like(moduli)
    if len(moduli) > 1:
        offsets[1:] = np.cumsum(moduli[:-1])
    total_entries = int(moduli.sum())

    return EngramHashSpec(
        vocab_size=vocab_size,
        compressed_vocab_size=comp_v,
        pad_id=pad_id,
        compressed_pad_id=comp_pad,
        max_ngram_size=max_ngram_size,
        n_head_per_ngram=n_head_per_ngram,
        layer_id=layer_id,
        lookup_table_np=lut_np,
        multipliers_np=layer_multipliers[layer_id],
        head_moduli_np=moduli,
        head_offsets_np=offsets,
        total_entries=total_entries,
    )


def cpu_numpy_ngram_hashes_ref(
    input_ids: torch.Tensor,
    spec: EngramHashSpec,
    doc_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Exact reference of DeepSeek `engram_demo_v1.py` CPU NumPy `_get_ngram_hashes` +
    `MultiHeadEmbedding` offset shift (including the real DtoH -> NumPy -> HtoD transfer!).
    Returns global table indices `[B, T, H_heads]` (`torch.long` on `input_ids.device`).
    """
    dev = input_ids.device
    arr = input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    comp = spec.lookup_table_np[arr]
    B, T = comp.shape
    mults = spec.multipliers_np

    if doc_ids is not None:
        docs = doc_ids.detach().cpu().numpy()
    else:
        docs = None

    shifts = []
    for k in range(spec.max_ngram_size):
        if k == 0:
            shifts.append(comp)
        else:
            sh = np.pad(
                comp, ((0, 0), (k, 0)), mode="constant", constant_values=spec.compressed_pad_id
            )[:, :T]
            if docs is not None:
                doc_sh = np.pad(docs, ((0, 0), (k, 0)), mode="constant", constant_values=-1)[:, :T]
                sh = np.where(doc_sh == docs, sh, spec.compressed_pad_id)
            shifts.append(sh)

    all_hashes = []
    h_cursor = 0
    for n in range(2, spec.max_ngram_size + 1):
        mix = shifts[0] * mults[0]
        for k in range(1, n):
            mix = np.bitwise_xor(mix, shifts[k] * mults[k])
        for _ in range(spec.n_head_per_ngram):
            mod = int(spec.head_moduli_np[h_cursor])
            off = int(spec.head_offsets_np[h_cursor])
            h = (mix % mod) + off
            all_hashes.append(h.astype(np.int64, copy=False))
            h_cursor += 1

    stacked = np.stack(all_hashes, axis=2)
    return torch.from_numpy(stacked).to(device=dev, dtype=torch.long)


# =============================================================================
# 2. Track 1 Triton Kernels: GPU Int64 N-Gram Hash, Gather, CSR Segment-Reduce,
#    and Exact Dense-Equivalent Pre-Forward JIT-Lazy-Adam(lr=5x, wd=0)
# =============================================================================


@triton.jit
def _engram_ngram_hash_kernel(
    input_ids_ptr,        # [B, T] int64 or int32
    doc_ids_ptr,          # [B, T] int32 or nullptr
    lut_ptr,              # [V] int32 -> compressed token id
    mult_ptr,             # [MAX_NGRAM] int64
    mod_ptr,              # [NUM_HEADS] int64
    off_ptr,              # [NUM_HEADS] int64
    out_idx_ptr,          # [B * T * NUM_HEADS] int32 global table indices
    B: tl.constexpr,
    T,
    HAS_DOC_IDS: tl.constexpr,
    COMP_PAD_ID: tl.constexpr,
    MAX_NGRAM: tl.constexpr,
    N_HEAD_PER_NGRAM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Pure-GPU int64 multiplicative-XOR prime N-gram hash with surjective vocab lookup
    and packed-document `doc_ids` isolation. Zero CPU synchronization.
    Matches `engram_demo_v1.py` `_get_ngram_hashes` bit-for-bit on int64.
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    row_base = pid_b * T

    raw0 = tl.load(input_ids_ptr + row_base + t_offs, mask=t_mask, other=0)
    tok0 = tl.load(lut_ptr + raw0, mask=t_mask, other=COMP_PAD_ID).to(tl.int64)

    if HAS_DOC_IDS:
        doc0 = tl.load(doc_ids_ptr + row_base + t_offs, mask=t_mask, other=-1)
    else:
        doc0 = tl.zeros([BLOCK_T], dtype=tl.int32)

    m0 = tl.load(mult_ptr + 0).to(tl.int64)
    mix = tok0 * m0

    # Lag k = 1 -> 2-gram heads [0 .. N_HEAD_PER_NGRAM - 1]
    t1 = t_offs - 1
    valid1 = t_mask & (t1 >= 0)
    raw1 = tl.load(input_ids_ptr + row_base + t1, mask=valid1, other=0)
    tok1 = tl.load(lut_ptr + raw1, mask=valid1, other=COMP_PAD_ID).to(tl.int64)
    if HAS_DOC_IDS:
        doc1 = tl.load(doc_ids_ptr + row_base + t1, mask=valid1, other=-2)
        same1 = valid1 & (doc1 == doc0)
        tok1 = tl.where(same1, tok1, tl.full([BLOCK_T], COMP_PAD_ID, dtype=tl.int64))
    else:
        tok1 = tl.where(valid1, tok1, tl.full([BLOCK_T], COMP_PAD_ID, dtype=tl.int64))

    m1 = tl.load(mult_ptr + 1).to(tl.int64)
    mix2 = mix ^ (tok1 * m1)

    out_base = (row_base + t_offs) * NUM_HEADS
    for j in tl.static_range(0, N_HEAD_PER_NGRAM):
        mod_j = tl.load(mod_ptr + j).to(tl.int64)
        off_j = tl.load(off_ptr + j).to(tl.int64)
        rem = mix2 % mod_j
        rem = tl.where(rem < 0, rem + mod_j, rem)
        gid = (rem + off_j).to(tl.int32)
        tl.store(out_idx_ptr + out_base + j, gid, mask=t_mask)

    if MAX_NGRAM >= 3:
        t2 = t_offs - 2
        valid2 = t_mask & (t2 >= 0)
        raw2 = tl.load(input_ids_ptr + row_base + t2, mask=valid2, other=0)
        tok2 = tl.load(lut_ptr + raw2, mask=valid2, other=COMP_PAD_ID).to(tl.int64)
        if HAS_DOC_IDS:
            doc2 = tl.load(doc_ids_ptr + row_base + t2, mask=valid2, other=-2)
            same2 = valid2 & (doc2 == doc0)
            tok2 = tl.where(same2, tok2, tl.full([BLOCK_T], COMP_PAD_ID, dtype=tl.int64))
        else:
            tok2 = tl.where(valid2, tok2, tl.full([BLOCK_T], COMP_PAD_ID, dtype=tl.int64))

        m2 = tl.load(mult_ptr + 2).to(tl.int64)
        mix3 = mix2 ^ (tok2 * m2)
        for j in tl.static_range(0, N_HEAD_PER_NGRAM):
            h_idx = N_HEAD_PER_NGRAM + j
            mod_j = tl.load(mod_ptr + h_idx).to(tl.int64)
            off_j = tl.load(off_ptr + h_idx).to(tl.int64)
            rem = mix3 % mod_j
            rem = tl.where(rem < 0, rem + mod_j, rem)
            gid = (rem + off_j).to(tl.int32)
            tl.store(out_idx_ptr + out_base + h_idx, gid, mask=t_mask)


@triton.jit
def _engram_gather_fwd_kernel(
    embed_ptr,            # [total_N, D_HEAD] bf16/fp32
    idx_ptr,              # [N_Q] int32
    out_ptr,              # [N_Q, D_HEAD] bf16/fp32 (contiguous [B, T, NUM_HEADS * D_HEAD])
    N_Q,
    D_HEAD: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    q_offs = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < N_Q
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_HEAD

    row_ids = tl.load(idx_ptr + q_offs, mask=q_mask, other=0).to(tl.int64)
    mask_2d = q_mask[:, None] & d_mask[None, :]
    vals = tl.load(embed_ptr + row_ids[:, None] * D_HEAD + d_offs[None, :], mask=mask_2d, other=0.0)
    tl.store(out_ptr + q_offs[:, None] * D_HEAD + d_offs[None, :], vals, mask=mask_2d)


@triton.jit
def _engram_csr_segment_reduce_bwd_kernel(
    grad_out_ptr,         # [N_Q, D_HEAD] bf16/fp32
    perm_ptr,             # [N_Q] int64/int32 sorted permutation
    seg_offsets_ptr,      # [U + 1] int64/int32 CSR offsets per unique key
    uniq_grad_ptr,        # [U, D_HEAD] fp32 output (one unique row per program -> ZERO atomics!)
    U,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Atomic-free unique-row gradient reduction for Zipfian N-gram tables.
    Each program exclusively owns one unique row `u in [0, U)` and sums its
    sorted occurrences `perm[start:end]` in FP32 registers.
    """
    u = tl.program_id(0)
    if u >= U:
        return
    start = tl.load(seg_offsets_ptr + u)
    end = tl.load(seg_offsets_ptr + u + 1)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_HEAD
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    p = start
    while p < end:
        q_idx = tl.load(perm_ptr + p).to(tl.int64)
        g = tl.load(grad_out_ptr + q_idx * D_HEAD + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        acc += g
        p += 1

    tl.store(uniq_grad_ptr + u.to(tl.int64) * D_HEAD + d_offs, acc, mask=d_mask)


@triton.jit
def _jit_lazy_adam_catchup_kernel(
    param_ptr,            # [N_ROWS, D_HEAD] bf16/fp32 (master or fp32/bf16 param)
    master_ptr,           # [N_ROWS, D_HEAD] fp32 master weight
    exp_avg_ptr,          # [N_ROWS, D_HEAD] fp32
    exp_avg_sq_ptr,       # [N_ROWS, D_HEAD] fp32
    last_step_ptr,        # [N_ROWS] int32
    uniq_rows_ptr,        # [U] int32 active local row indices
    lr_hist_ptr,          # [MAX_STEPS] fp32 (1-indexed: lr_hist[k] = lr at step k)
    U,
    target_step,          # catch up through `target_step` (e.g. `step - 1` before forward at `step`)
    beta1,
    beta2,
    eps,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Pre-Forward Just-In-Time (JIT) Momentum Catch-Up for exact `Adam(wd=0.0)`:
    Advances only requested rows `z` having `0 < last_step[z] < target_step` through
    missed zero-gradient steps `k = last_step[z] + 1 .. target_step` entirely in registers.
    Rows with `last_step[z] == 0` (never touched yet -> m=0, v=0) exit in 1 branch!
    """
    u = tl.program_id(0)
    if u >= U:
        return
    row = tl.load(uniq_rows_ptr + u).to(tl.int64)
    prev_t = tl.load(last_step_ptr + row)
    if prev_t <= 0 or prev_t >= target_step:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_HEAD
    base = row * D_HEAD + d_offs

    w = tl.load(master_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    m = tl.load(exp_avg_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    v = tl.load(exp_avg_sq_ptr + base, mask=d_mask, other=0.0).to(tl.float32)

    k = prev_t + 1
    while k <= target_step:
        m = beta1 * m
        v = beta2 * v
        lr_k = tl.load(lr_hist_ptr + k).to(tl.float32)
        kf = k.to(tl.float32)
        bc1 = 1.0 - tl.extra.cuda.libdevice.pow(beta1, kf)
        bc2 = 1.0 - tl.extra.cuda.libdevice.pow(beta2, kf)
        denom = tl.sqrt(v / bc2) + eps
        w = w - lr_k * (m / bc1) / denom
        k += 1

    tl.store(master_ptr + base, w, mask=d_mask)
    tl.store(param_ptr + base, w.to(param_ptr.dtype.element_ty), mask=d_mask)
    tl.store(exp_avg_ptr + base, m, mask=d_mask)
    tl.store(exp_avg_sq_ptr + base, v, mask=d_mask)
    tl.store(last_step_ptr + row, target_step)


@triton.jit
def _jit_lazy_adam_step_kernel(
    param_ptr,            # [N_ROWS, D_HEAD] bf16/fp32
    master_ptr,           # [N_ROWS, D_HEAD] fp32
    exp_avg_ptr,          # [N_ROWS, D_HEAD] fp32
    exp_avg_sq_ptr,       # [N_ROWS, D_HEAD] fp32
    last_step_ptr,        # [N_ROWS] int32
    uniq_rows_ptr,        # [U] int32
    uniq_grad_ptr,        # [U, D_HEAD] fp32
    U,
    cur_step,             # int32 >= 1
    lr_cur,
    beta1,
    beta2,
    bc1,
    bc2,
    eps,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Applies exact Adam(wd=0.0) update at `cur_step` to the `U` unique active rows
    (which are already caught up to `cur_step - 1` by pre-forward catch-up, or
    have `last_step == 0` with `m=0, v=0`).
    """
    u = tl.program_id(0)
    if u >= U:
        return
    row = tl.load(uniq_rows_ptr + u).to(tl.int64)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_HEAD
    base = row * D_HEAD + d_offs

    g = tl.load(uniq_grad_ptr + u.to(tl.int64) * D_HEAD + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    w = tl.load(master_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    m = tl.load(exp_avg_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    v = tl.load(exp_avg_sq_ptr + base, mask=d_mask, other=0.0).to(tl.float32)

    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)

    denom = tl.sqrt(v / bc2) + eps
    w = w - lr_cur * (m / bc1) / denom

    tl.store(master_ptr + base, w, mask=d_mask)
    tl.store(param_ptr + base, w.to(param_ptr.dtype.element_ty), mask=d_mask)
    tl.store(exp_avg_ptr + base, m, mask=d_mask)
    tl.store(exp_avg_sq_ptr + base, v, mask=d_mask)
    tl.store(last_step_ptr + row, cur_step)


@triton.jit
def _jit_lazy_adam_sorted_catchup_kernel(
    param_ptr,            # [N_ROWS, D_HEAD] bf16/fp32
    master_ptr,           # [N_ROWS, D_HEAD] fp32
    exp_avg_ptr,          # [N_ROWS, D_HEAD] fp32
    exp_avg_sq_ptr,       # [N_ROWS, D_HEAD] fp32
    last_step_ptr,        # [N_ROWS] int32
    sorted_keys_ptr,      # [N_Q] int32 sorted non-decreasingly
    lr_hist_ptr,          # [MAX_STEPS] fp32
    N_Q,
    target_step,
    beta1,
    beta2,
    eps,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Zero-DtoH-Sync Pre-Forward JIT Momentum Catch-Up directly over `sorted_keys [N_Q]`:
    Program `q` only executes if `q == 0` or `sorted_keys[q] != sorted_keys[q - 1]`
    (guaranteeing exclusive single-owner execution per unique active row with ZERO
    `unique_consecutive` host synchronization!).
    """
    q = tl.program_id(0)
    if q >= N_Q:
        return
    row32 = tl.load(sorted_keys_ptr + q)
    if q > 0:
        prev_row32 = tl.load(sorted_keys_ptr + q - 1)
        if row32 == prev_row32:
            return

    row = row32.to(tl.int64)
    prev_t = tl.load(last_step_ptr + row)
    if prev_t <= 0 or prev_t >= target_step:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_HEAD
    base = row * D_HEAD + d_offs

    w = tl.load(master_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    m = tl.load(exp_avg_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    v = tl.load(exp_avg_sq_ptr + base, mask=d_mask, other=0.0).to(tl.float32)

    k = prev_t + 1
    while k <= target_step:
        m = beta1 * m
        v = beta2 * v
        lr_k = tl.load(lr_hist_ptr + k).to(tl.float32)
        kf = k.to(tl.float32)
        bc1 = 1.0 - tl.extra.cuda.libdevice.pow(beta1, kf)
        bc2 = 1.0 - tl.extra.cuda.libdevice.pow(beta2, kf)
        denom = tl.sqrt(v / bc2) + eps
        w = w - lr_k * (m / bc1) / denom
        k += 1

    tl.store(master_ptr + base, w, mask=d_mask)
    tl.store(param_ptr + base, w.to(param_ptr.dtype.element_ty), mask=d_mask)
    tl.store(exp_avg_ptr + base, m, mask=d_mask)
    tl.store(exp_avg_sq_ptr + base, v, mask=d_mask)
    tl.store(last_step_ptr + row, target_step)


@triton.jit
def _sorted_tile_segmented_bwd_and_adam_kernel(
    grad_out_ptr,         # [N_Q, D_HEAD] bf16/fp32
    sorted_keys_ptr,      # [N_Q] int32
    perm_ptr,             # [N_Q] int64
    param_ptr,            # [N_ROWS, D_HEAD] bf16/fp32
    master_ptr,           # [N_ROWS, D_HEAD] fp32
    exp_avg_ptr,          # [N_ROWS, D_HEAD] fp32
    exp_avg_sq_ptr,       # [N_ROWS, D_HEAD] fp32
    last_step_ptr,        # [N_ROWS] int32
    b_keys_ptr,           # [2 * NUM_TILES] int32 boundary carry keys (-1 if unused)
    b_grads_ptr,          # [2 * NUM_TILES, D_HEAD] fp32 boundary carry partial sums
    N_Q,
    cur_step,
    lr_cur,
    beta1,
    beta2,
    bc1,
    bc2,
    eps,
    TILE_Q: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Two-Stage Tile-Fused Atomic-Free Zipfian Backward + Adam Step (Stage 1):
    Each CTA owns tile `[t0, t1)` (`TILE_Q = 64` sorted queries):
      - Complete interior runs (`L >= t0` is global run-head AND `R < t1`) fuse
        gradient reduction + exact Adam step directly in registers with ZERO
        intermediate HBM writes and ZERO atomics!
      - Cross-tile runs (such as extreme Zipfian hot rows spanning dozens of tiles)
        are reduced in parallel across all tiles (`64x` parallelism!) into at most
        2 boundary carry slots (`2 * tile_id + {0, 1}`) finalized by Stage 2!
    """
    tile_id = tl.program_id(0)
    t0 = tile_id * TILE_Q
    if t0 >= N_Q:
        return
    t1 = tl.minimum(N_Q, t0 + TILE_Q)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_HEAD
    param_ty = param_ptr.dtype.element_ty

    slot0 = 2 * tile_id
    slot1 = slot0 + 1
    tl.store(b_keys_ptr + slot0, -1)
    tl.store(b_keys_ptr + slot1, -1)

    i = t0
    # 1. Check if `t0` continues a run from `t0 - 1` (Prefix cross-tile fragment -> slot0)
    if t0 > 0:
        k_prev = tl.load(sorted_keys_ptr + t0 - 1)
        k_first = tl.load(sorted_keys_ptr + t0)
        if k_first == k_prev:
            acc_pre = tl.zeros([BLOCK_D], dtype=tl.float32)
            keep_pre = 1
            while (i < t1) & (keep_pre != 0):
                ki = tl.load(sorted_keys_ptr + i)
                if ki == k_first:
                    qi = tl.load(perm_ptr + i).to(tl.int64)
                    acc_pre += tl.load(grad_out_ptr + qi * D_HEAD + d_offs, mask=d_mask, other=0.0).to(tl.float32)
                    i += 1
                else:
                    keep_pre = 0
            tl.store(b_keys_ptr + slot0, k_first)
            tl.store(b_grads_ptr + slot0.to(tl.int64) * D_HEAD + d_offs, acc_pre, mask=d_mask)

    # Check key right after `t1` to detect Suffix cross-tile run
    has_next_tile = t1 < N_Q
    k_after = tl.load(sorted_keys_ptr + t1) if has_next_tile else -2

    # 2. Sweep remaining runs starting at global run-heads `i in [t0, t1)`
    while i < t1:
        k_cur = tl.load(sorted_keys_ptr + i)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        keep_run = 1
        while (i < t1) & (keep_run != 0):
            ki = tl.load(sorted_keys_ptr + i)
            if ki == k_cur:
                qi = tl.load(perm_ptr + i).to(tl.int64)
                acc += tl.load(grad_out_ptr + qi * D_HEAD + d_offs, mask=d_mask, other=0.0).to(tl.float32)
                i += 1
            else:
                keep_run = 0

        if has_next_tile and (i == t1) and (k_cur == k_after):
            # Suffix cross-tile fragment -> slot1
            tl.store(b_keys_ptr + slot1, k_cur)
            tl.store(b_grads_ptr + slot1.to(tl.int64) * D_HEAD + d_offs, acc, mask=d_mask)
        else:
            # 100% complete interior run -> apply exact Adam(wd=0.0) step directly in registers!
            row = k_cur.to(tl.int64)
            base = row * D_HEAD + d_offs
            w = tl.load(master_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
            m = tl.load(exp_avg_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
            v = tl.load(exp_avg_sq_ptr + base, mask=d_mask, other=0.0).to(tl.float32)

            m = beta1 * m + (1.0 - beta1) * acc
            v = beta2 * v + (1.0 - beta2) * (acc * acc)
            denom = tl.sqrt(v / bc2) + eps
            w = w - lr_cur * (m / bc1) / denom

            tl.store(master_ptr + base, w, mask=d_mask)
            tl.store(param_ptr + base, w.to(param_ty), mask=d_mask)
            tl.store(exp_avg_ptr + base, m, mask=d_mask)
            tl.store(exp_avg_sq_ptr + base, v, mask=d_mask)
            tl.store(last_step_ptr + row, cur_step)


@triton.jit
def _boundary_carry_finalize_adam_kernel(
    b_keys_ptr,           # [NUM_SLOTS] int32 (`NUM_SLOTS = 2 * NUM_TILES`)
    b_grads_ptr,          # [NUM_SLOTS, D_HEAD] fp32
    param_ptr,            # [N_ROWS, D_HEAD] bf16/fp32
    master_ptr,           # [N_ROWS, D_HEAD] fp32
    exp_avg_ptr,          # [N_ROWS, D_HEAD] fp32
    exp_avg_sq_ptr,       # [N_ROWS, D_HEAD] fp32
    last_step_ptr,        # [N_ROWS] int32
    NUM_SLOTS,
    cur_step,
    lr_cur,
    beta1,
    beta2,
    bc1,
    bc2,
    eps,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Stage 2 Finalizer for Cross-Tile Runs (`<= 2 * NUM_TILES` tiny slots):
    Because any cross-tile continuation slot `s = 2*t + 0` MUST match either `s - 1`
    (suffix of tile `t - 1`) or `s - 2` (full-tile prefix of tile `t - 1`), checking
    `s - 1` and `s - 2` identifies the unique first fragment in O(1)!
    """
    s = tl.program_id(0)
    if s >= NUM_SLOTS:
        return
    k_cur = tl.load(b_keys_ptr + s)
    if k_cur < 0:
        return

    if s >= 1:
        if tl.load(b_keys_ptr + s - 1) == k_cur:
            return
    if s >= 2:
        if tl.load(b_keys_ptr + s - 2) == k_cur:
            return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_HEAD
    acc = tl.load(b_grads_ptr + s.to(tl.int64) * D_HEAD + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Advance across subsequent tiles `t_next = (s // 2) + 1, ...` where `slot = 2 * t_next + 0` holds `k_cur`
    next_tile_slot = ((s // 2) + 1) * 2
    keep_carry = 1
    while (next_tile_slot < NUM_SLOTS) & (keep_carry != 0):
        kn = tl.load(b_keys_ptr + next_tile_slot)
        if kn == k_cur:
            acc += tl.load(b_grads_ptr + next_tile_slot.to(tl.int64) * D_HEAD + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            next_tile_slot += 2
        else:
            keep_carry = 0

    row = k_cur.to(tl.int64)
    base = row * D_HEAD + d_offs
    w = tl.load(master_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    m = tl.load(exp_avg_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    v = tl.load(exp_avg_sq_ptr + base, mask=d_mask, other=0.0).to(tl.float32)

    m = beta1 * m + (1.0 - beta1) * acc
    v = beta2 * v + (1.0 - beta2) * (acc * acc)
    denom = tl.sqrt(v / bc2) + eps
    w = w - lr_cur * (m / bc1) / denom

    tl.store(master_ptr + base, w, mask=d_mask)
    tl.store(param_ptr + base, w.to(param_ptr.dtype.element_ty), mask=d_mask)
    tl.store(exp_avg_ptr + base, m, mask=d_mask)
    tl.store(exp_avg_sq_ptr + base, v, mask=d_mask)
    tl.store(last_step_ptr + row, cur_step)


def gpu_engram_ngram_hashes(
    input_ids: torch.Tensor,
    lut_gpu: torch.Tensor,
    mult_gpu: torch.Tensor,
    mod_gpu: torch.Tensor,
    off_gpu: torch.Tensor,
    compressed_pad_id: int,
    max_ngram_size: int,
    n_head_per_ngram: int,
    doc_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes `[B * T * H_heads]` (`int32`) global table indices strictly on GPU.
    """
    assert input_ids.is_cuda and input_ids.ndim == 2
    B, T = input_ids.shape
    num_heads = (max_ngram_size - 1) * n_head_per_ngram
    out_idx = torch.empty((B * T * num_heads,), device=input_ids.device, dtype=torch.int32)
    block_t = 256
    grid = (B, triton.cdiv(T, block_t))
    has_docs = doc_ids is not None
    doc_ptr = doc_ids if has_docs else input_ids
    _engram_ngram_hash_kernel[grid](
        input_ids,
        doc_ptr,
        lut_gpu,
        mult_gpu,
        mod_gpu,
        off_gpu,
        out_idx,
        B=B,
        T=T,
        HAS_DOC_IDS=has_docs,
        COMP_PAD_ID=int(compressed_pad_id),
        MAX_NGRAM=max_ngram_size,
        N_HEAD_PER_NGRAM=n_head_per_ngram,
        NUM_HEADS=num_heads,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return out_idx


def deduplicate_csr_indices(flat_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast GPU radix-sort + CSR deduplication:
    Returns `(uniq_rows [U] int32, perm [N_Q] int64, seg_offsets [U+1] int64)`.
    """
    sorted_keys, perm = torch.sort(flat_indices)
    uniq_rows, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    U = uniq_rows.numel()
    seg_offsets = torch.empty((U + 1,), device=flat_indices.device, dtype=torch.int64)
    seg_offsets[0] = 0
    if U > 0:
        torch.cumsum(counts, dim=0, out=seg_offsets[1:])
    return uniq_rows, perm, seg_offsets


class _EngramTileFusedAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dummy_leaf: torch.Tensor,    # 0-D scalar leaf (ZERO copy of `[total_N, D_HEAD]`!)
        embed_weight: torch.Tensor,
        flat_indices: torch.Tensor,
        sorted_keys: torch.Tensor,
        perm: torch.Tensor,
        batch_size: int,
        seq_len: int,
        num_heads: int,
    ) -> torch.Tensor:
        N_Q = flat_indices.numel()
        D_HEAD = embed_weight.shape[1]
        block_d = triton.next_power_of_2(D_HEAD)
        block_q = 64
        out = torch.empty((N_Q, D_HEAD), device=embed_weight.device, dtype=embed_weight.dtype)
        _engram_gather_fwd_kernel[(triton.cdiv(N_Q, block_q),)](
            embed_weight,
            flat_indices,
            out,
            N_Q=N_Q,
            D_HEAD=D_HEAD,
            BLOCK_Q=block_q,
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(sorted_keys, perm)
        ctx.d_head = D_HEAD
        return out.view(batch_size, seq_len, num_heads * D_HEAD)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sorted_keys, perm = ctx.saved_tensors
        D_HEAD = ctx.d_head
        # Defer segmented reduction directly into `step_lazy_adam` so backward reduction
        # and Adam update fuse into 1 register pass for 98%+ of unique active rows!
        _LAST_CSR_GRAD_SLOT["sorted_keys"] = sorted_keys
        _LAST_CSR_GRAD_SLOT["perm"] = perm
        _LAST_CSR_GRAD_SLOT["grad_flat"] = grad_output.contiguous().view(-1, D_HEAD)
        return grad_output.new_zeros(()), None, None, None, None, None, None, None


_LAST_CSR_GRAD_SLOT: Dict[str, Optional[torch.Tensor]] = {
    "uniq_rows": None,
    "uniq_grads": None,
    "sorted_keys": None,
    "perm": None,
    "grad_flat": None,
}


class EngramFusedLookupTable(nn.Module):
    """
    GPU-native Engram Multi-Head Lookup Table with Atomic-Free CSR Backward and
    Exact Dense-Equivalent Pre-Forward `JIT-Lazy-Adam(lr=5x, wd=0.0)`.
    """

    def __init__(
        self,
        spec: EngramHashSpec,
        d_head: int = 80,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        max_steps: int = 16384,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__()
        dev = device or torch.device("cuda")
        self.spec = spec
        self.total_entries = spec.total_entries
        self.d_head = d_head
        self.num_heads = spec.num_heads
        self.d_mem = self.num_heads * d_head
        self.betas = betas
        self.eps = eps

        # Embedding weight (stored in `dtype` for 2-byte gather, backed by FP32 master for exact Adam)
        init_w = torch.randn((self.total_entries, d_head), device=dev, dtype=torch.float32) * 0.02
        self.weight = nn.Parameter(init_w.to(dtype), requires_grad=False)
        self.register_buffer("master_weight", self.weight.detach().float().clone())
        self.register_buffer("exp_avg", torch.zeros((self.total_entries, d_head), device=dev, dtype=torch.float32))
        self.register_buffer("exp_avg_sq", torch.zeros((self.total_entries, d_head), device=dev, dtype=torch.float32))
        self.register_buffer("last_step", torch.zeros((self.total_entries,), device=dev, dtype=torch.int32))
        self.register_buffer("lr_history", torch.zeros((max_steps + 2,), device=dev, dtype=torch.float32))

        self.register_buffer("lut_gpu", torch.from_numpy(spec.lookup_table_np.astype(np.int32)).to(dev))
        self.register_buffer("mult_gpu", torch.from_numpy(spec.multipliers_np).to(dev))
        self.register_buffer("mod_gpu", torch.from_numpy(spec.head_moduli_np).to(dev))
        self.register_buffer("off_gpu", torch.from_numpy(spec.head_offsets_np).to(dev))

        self.completed_step: int = 0
        self._pending_uniq_rows: Optional[torch.Tensor] = None
        self._pending_uniq_grads: Optional[torch.Tensor] = None

        self._dummy_leaf = torch.zeros((), device=dev, dtype=dtype, requires_grad=True)

    def copy_from_reference_embedding_(self, ref_weight: torch.Tensor) -> None:
        with torch.no_grad():
            self.weight.copy_(ref_weight.to(self.weight.dtype))
            self.master_weight.copy_(self.weight.float())
            self.exp_avg.zero_()
            self.exp_avg_sq.zero_()
            self.last_step.zero_()
            self.lr_history.zero_()
            self.completed_step = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_ids: Optional[torch.Tensor] = None,
        need_grad: bool = True,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        flat_idx = gpu_engram_ngram_hashes(
            input_ids=input_ids,
            lut_gpu=self.lut_gpu,
            mult_gpu=self.mult_gpu,
            mod_gpu=self.mod_gpu,
            off_gpu=self.off_gpu,
            compressed_pad_id=self.spec.compressed_pad_id,
            max_ngram_size=self.spec.max_ngram_size,
            n_head_per_ngram=self.spec.n_head_per_ngram,
            doc_ids=doc_ids,
        )
        N_Q = flat_idx.numel()

        if not need_grad and self.completed_step == 0:
            block_d = triton.next_power_of_2(self.d_head)
            out = torch.empty((N_Q, self.d_head), device=input_ids.device, dtype=self.weight.dtype)
            _engram_gather_fwd_kernel[(triton.cdiv(N_Q, 64),)](
                self.weight,
                flat_idx,
                out,
                N_Q=N_Q,
                D_HEAD=self.d_head,
                BLOCK_Q=64,
                BLOCK_D=block_d,
                num_warps=4,
            )
            return out.view(B, T, self.d_mem)

        # Pure-GPU CUB radix sort (`0` unique_consecutive host sync!)
        sorted_keys, perm = torch.sort(flat_idx)
        if self.completed_step > 0 and N_Q > 0:
            block_d = triton.next_power_of_2(self.d_head)
            _jit_lazy_adam_sorted_catchup_kernel[(N_Q,)](
                self.weight,
                self.master_weight,
                self.exp_avg,
                self.exp_avg_sq,
                self.last_step,
                sorted_keys,
                self.lr_history,
                N_Q=N_Q,
                target_step=self.completed_step,
                beta1=self.betas[0],
                beta2=self.betas[1],
                eps=self.eps,
                D_HEAD=self.d_head,
                BLOCK_D=block_d,
                num_warps=2 if block_d <= 128 else 4,
            )

        if not need_grad:
            block_d = triton.next_power_of_2(self.d_head)
            out = torch.empty((N_Q, self.d_head), device=input_ids.device, dtype=self.weight.dtype)
            _engram_gather_fwd_kernel[(triton.cdiv(N_Q, 64),)](
                self.weight,
                flat_idx,
                out,
                N_Q=N_Q,
                D_HEAD=self.d_head,
                BLOCK_Q=64,
                BLOCK_D=block_d,
                num_warps=4,
            )
            return out.view(B, T, self.d_mem)

        self._dummy_leaf.grad = None
        return _EngramTileFusedAutogradFn.apply(
            self._dummy_leaf,
            self.weight,
            flat_idx,
            sorted_keys,
            perm,
            B,
            T,
            self.num_heads,
        )

    def step_lazy_adam(self, lr: float, count_unique: bool = False) -> int:
        """
        Executes Two-Stage Tile-Fused Atomic-Free Backward Reduction + Exact Adam(wd=0.0)
        with ZERO `unique_consecutive` host sync (or falls back to explicit `uniq_rows`
        if populated by Trial 2 ablation).
        """
        next_step = self.completed_step + 1
        self.lr_history[next_step] = float(lr)
        block_d = triton.next_power_of_2(self.d_head)
        bc1 = float(1.0 - (self.betas[0] ** next_step))
        bc2 = float(1.0 - (self.betas[1] ** next_step))

        sorted_keys = _LAST_CSR_GRAD_SLOT["sorted_keys"]
        if sorted_keys is not None:
            perm = _LAST_CSR_GRAD_SLOT["perm"]
            grad_flat = _LAST_CSR_GRAD_SLOT["grad_flat"]
            _LAST_CSR_GRAD_SLOT["sorted_keys"] = None
            _LAST_CSR_GRAD_SLOT["perm"] = None
            _LAST_CSR_GRAD_SLOT["grad_flat"] = None

            N_Q = sorted_keys.numel()
            if N_Q > 0:
                tile_q = 64
                num_tiles = triton.cdiv(N_Q, tile_q)
                num_slots = 2 * num_tiles
                b_keys = torch.empty((num_slots,), device=sorted_keys.device, dtype=torch.int32)
                b_grads = torch.empty((num_slots, self.d_head), device=sorted_keys.device, dtype=torch.float32)

                bwd_warps = max(1, block_d // 32)
                _sorted_tile_segmented_bwd_and_adam_kernel[(num_tiles,)](
                    grad_flat,
                    sorted_keys,
                    perm,
                    self.weight,
                    self.master_weight,
                    self.exp_avg,
                    self.exp_avg_sq,
                    self.last_step,
                    b_keys,
                    b_grads,
                    N_Q=N_Q,
                    cur_step=next_step,
                    lr_cur=float(lr),
                    beta1=self.betas[0],
                    beta2=self.betas[1],
                    bc1=bc1,
                    bc2=bc2,
                    eps=self.eps,
                    TILE_Q=tile_q,
                    D_HEAD=self.d_head,
                    BLOCK_D=block_d,
                    num_warps=bwd_warps,
                )
                _boundary_carry_finalize_adam_kernel[(num_slots,)](
                    b_keys,
                    b_grads,
                    self.weight,
                    self.master_weight,
                    self.exp_avg,
                    self.exp_avg_sq,
                    self.last_step,
                    NUM_SLOTS=num_slots,
                    cur_step=next_step,
                    lr_cur=float(lr),
                    beta1=self.betas[0],
                    beta2=self.betas[1],
                    bc1=bc1,
                    bc2=bc2,
                    eps=self.eps,
                    D_HEAD=self.d_head,
                    BLOCK_D=block_d,
                    num_warps=bwd_warps,
                )
            self.completed_step = next_step
            if count_unique and N_Q > 0:
                return int(1 + (sorted_keys[1:] != sorted_keys[:-1]).sum().item())
            return N_Q

        uniq_rows = _LAST_CSR_GRAD_SLOT["uniq_rows"]
        uniq_grads = _LAST_CSR_GRAD_SLOT["uniq_grads"]
        assert uniq_rows is not None and uniq_grads is not None, "Call backward() before step_lazy_adam()"
        _LAST_CSR_GRAD_SLOT["uniq_rows"] = None
        _LAST_CSR_GRAD_SLOT["uniq_grads"] = None

        U = uniq_rows.numel()
        if U > 0:
            _jit_lazy_adam_step_kernel[(U,)](
                self.weight,
                self.master_weight,
                self.exp_avg,
                self.exp_avg_sq,
                self.last_step,
                uniq_rows,
                uniq_grads,
                U=U,
                cur_step=next_step,
                lr_cur=float(lr),
                beta1=self.betas[0],
                beta2=self.betas[1],
                bc1=bc1,
                bc2=bc2,
                eps=self.eps,
                D_HEAD=self.d_head,
                BLOCK_D=block_d,
                num_warps=2 if block_d <= 128 else 4,
            )
        self.completed_step = next_step
        return U

    def flush_all_rows_(self) -> int:
        """
        Flushes all previously touched rows (`0 < last_step < completed_step`) up to
        `completed_step` for full-table inspection or checkpoint export.
        """
        if self.completed_step <= 0:
            return 0
        stale_mask = (self.last_step > 0) & (self.last_step < self.completed_step)
        stale_rows = torch.nonzero(stale_mask, as_tuple=False).view(-1).to(torch.int32)
        U = stale_rows.numel()
        if U > 0:
            block_d = triton.next_power_of_2(self.d_head)
            _jit_lazy_adam_catchup_kernel[(U,)](
                self.weight,
                self.master_weight,
                self.exp_avg,
                self.exp_avg_sq,
                self.last_step,
                stale_rows,
                self.lr_history,
                U=U,
                target_step=self.completed_step,
                beta1=self.betas[0],
                beta2=self.betas[1],
                eps=self.eps,
                D_HEAD=self.d_head,
                BLOCK_D=block_d,
                num_warps=2 if block_d <= 128 else 4,
            )
        return U


# =============================================================================
# 3. Track 2 Reference, FP64 Analytical Oracle, and Zero-Transpose Triton Kernels
#    (Multi-Branch Context Gating + Dilated Causal ShortConv + Dual Residual)
# =============================================================================


def fp64_engram_gate_conv_oracle(
    H: torch.Tensor,             # [B, T, M, C]
    E: torch.Tensor,             # [B, T, D_mem]
    W_kv: torch.Tensor,          # [(M+1)*C, D_mem]: rows [0:C]=W_V, rows [C:(M+1)*C]=W_K[0..M-1]
    gamma_h: torch.Tensor,       # [M, C]
    gamma_k: torch.Tensor,       # [M, C]
    gamma_conv: torch.Tensor,    # [M, C]
    w_conv: torch.Tensor,        # [M, C, 4] (PyTorch Conv1d tap layout: tap lag j uses w_conv[..., 3-j])
    dilation: int = 3,
    gate_eps: float = 1e-5,
    conv_eps: float = 1e-5,
    signed_sqrt_gate: bool = True,
    doc_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact FP64 ground-truth oracle for DeepSeek Engram Eq. 4-8 & `engram_demo_v1.py`:
      KV = E @ W_kv.T -> V [B,T,C], K [B,T,M,C]
      s_m = <RMSNorm_k(K_m), RMSNorm_h(H_m)> / sqrt(C)
      alpha_m = sigmoid(sign(s_m) * sqrt(max(|s_m|, 1e-6))) if signed_sqrt_gate else sigmoid(s_m)
      U_m = alpha_m * V
      A_m = RMSNorm_conv(U_m)
      Z_{b,t,m,c} = sum_{j=0..3} w_conv[m,c,3-j] * A_{b, t - j*dilation, m, c} * 1[same_doc]
      H_out = H + U + SiLU(Z)
    Returns `(H_out_fp64, alpha_fp64)`.
    """
    B, T, M, C = H.shape
    Hd = H.double()
    Ed = E.double()
    Wd = W_kv.double()
    gh = gamma_h.double()
    gk = gamma_k.double()
    gc = gamma_conv.double()
    wc = w_conv.double()

    KV = torch.matmul(Ed, Wd.t()).view(B, T, M + 1, C)
    V = KV[:, :, 0, :]       # [B, T, C]
    K = KV[:, :, 1:, :]      # [B, T, M, C]

    rh = torch.rsqrt(Hd.pow(2).mean(dim=-1, keepdim=True) + gate_eps)
    rk = torch.rsqrt(K.pow(2).mean(dim=-1, keepdim=True) + gate_eps)
    nh = Hd * rh * gh.view(1, 1, M, C)
    nk = K * rk * gk.view(1, 1, M, C)

    s = (nh * nk).sum(dim=-1) / math.sqrt(float(C))   # [B, T, M]
    if signed_sqrt_gate:
        g = s.abs().clamp_min(1e-6).sqrt() * s.sign()
        alpha = torch.sigmoid(g)
    else:
        alpha = torch.sigmoid(s)

    U = alpha.unsqueeze(-1) * V.unsqueeze(2)          # [B, T, M, C]
    ru = torch.rsqrt(U.pow(2).mean(dim=-1, keepdim=True) + conv_eps)
    A = U * ru * gc.view(1, 1, M, C)                  # [B, T, M, C]

    Z = torch.zeros_like(A)
    for j in range(4):
        lag = j * dilation
        w_tap = wc[:, :, 3 - j].view(1, 1, M, C)
        if lag == 0:
            Z = Z + A * w_tap
        elif lag < T:
            shifted = A[:, : T - lag, :, :] * w_tap
            if doc_ids is not None:
                same = (doc_ids[:, lag:] == doc_ids[:, : T - lag]).view(B, T - lag, 1, 1)
                shifted = torch.where(same, shifted, torch.zeros_like(shifted))
            Z[:, lag:, :, :] = Z[:, lag:, :, :] + shifted

    H_out = Hd + U + F.silu(Z)
    return H_out, alpha


class EagerDemoEngramContextOp(nn.Module):
    """
    Exact `engram_demo_v1.py` Track 2 reference (`Trial 0` baseline):
    5 separate Linear projections, 8 branch gate RMSNorms, 4 ShortConv RMSNorms,
    `torch.cat` + `[B, M*C, T]` transpose + cuDNN grouped `Conv1d` + `SiLU` +
    transpose back + `U + ShortConv(U)` + `H` residual.
    """

    def __init__(
        self,
        d_mem: int,
        d_backbone: int,
        hc_mult: int = 4,
        dilation: int = 3,
        gate_eps: float = 1e-5,
        conv_eps: float = 1e-5,
        signed_sqrt_gate: bool = True,
        zero_init_conv: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        dev = device or torch.device("cuda")
        self.d_mem = d_mem
        self.d_backbone = d_backbone
        self.hc_mult = hc_mult
        self.dilation = dilation
        self.signed_sqrt_gate = signed_sqrt_gate

        self.value_proj = nn.Linear(d_mem, d_backbone, bias=False, device=dev, dtype=dtype)
        self.key_projs = nn.ModuleList(
            [nn.Linear(d_mem, d_backbone, bias=False, device=dev, dtype=dtype) for _ in range(hc_mult)]
        )
        self.norm1 = nn.ModuleList(
            [nn.RMSNorm(d_backbone, eps=gate_eps, device=dev, dtype=dtype) for _ in range(hc_mult)]
        )
        self.norm2 = nn.ModuleList(
            [nn.RMSNorm(d_backbone, eps=gate_eps, device=dev, dtype=dtype) for _ in range(hc_mult)]
        )
        total_ch = hc_mult * d_backbone
        self.conv_norms = nn.ModuleList(
            [nn.RMSNorm(d_backbone, eps=conv_eps, device=dev, dtype=dtype) for _ in range(hc_mult)]
        )
        self.conv = nn.Conv1d(
            in_channels=total_ch,
            out_channels=total_ch,
            kernel_size=4,
            groups=total_ch,
            bias=False,
            padding=3 * dilation,
            dilation=dilation,
            device=dev,
            dtype=dtype,
        )
        if zero_init_conv:
            nn.init.zeros_(self.conv.weight)

    def export_packed_params(self) -> Dict[str, torch.Tensor]:
        w_kv = torch.cat([self.value_proj.weight.detach()] + [kp.weight.detach() for kp in self.key_projs], dim=0)
        gamma_k = torch.stack([n.weight.detach() for n in self.norm1], dim=0)
        gamma_h = torch.stack([n.weight.detach() for n in self.norm2], dim=0)
        gamma_conv = torch.stack([n.weight.detach() for n in self.conv_norms], dim=0)
        w_conv = self.conv.weight.detach().view(self.hc_mult, self.d_backbone, 4)
        return {
            "W_kv": w_kv.clone(),
            "gamma_h": gamma_h.clone(),
            "gamma_k": gamma_k.clone(),
            "gamma_conv": gamma_conv.clone(),
            "w_conv": w_conv.clone(),
        }

    def forward(self, H: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        B, T, M, C = H.shape
        inv_sqrt_d = 1.0 / math.sqrt(float(C))
        gates = []
        for m in range(M):
            key = self.key_projs[m](E)
            nk = self.norm1[m](key)
            nh = self.norm2[m](H[:, :, m, :])
            s = (nk.float() * nh.float()).sum(dim=-1) * inv_sqrt_d
            if self.signed_sqrt_gate:
                g = s.abs().clamp_min(1e-6).sqrt() * s.sign()
                a = g.sigmoid()
            else:
                a = s.sigmoid()
            gates.append(a.to(H.dtype).unsqueeze(-1))
        gates_t = torch.stack(gates, dim=2)                         # [B, T, M, 1]
        U = gates_t * self.value_proj(E).unsqueeze(2)               # [B, T, M, C]

        normed_chunks = [self.conv_norms[m](U[:, :, m, :]) for m in range(M)]
        x_norm = torch.cat(normed_chunks, dim=-1)                   # [B, T, M*C]
        x_bct = x_norm.transpose(1, 2)                              # [B, M*C, T]
        y_bct = F.silu(self.conv(x_bct)[..., :T])
        y = y_bct.transpose(1, 2).view(B, T, M, C).contiguous()
        return H + (U + y)


# -----------------------------------------------------------------------------
# Track 2 Triton Forward & Lock-Free Split-CTA Backward Kernels (`M = 4`)
# -----------------------------------------------------------------------------


@triton.jit
def _engram_gate_prenorm_fwd_kernel(
    H_ptr,                # [N_TOK, 4, C] bf16/fp32
    KV_ptr,               # [N_TOK, 5, C] bf16/fp32 (slice 0=V, slices 1..4=K_0..K_3)
    gamma_h_ptr,          # [4, C] bf16/fp32
    gamma_k_ptr,          # [4, C] bf16/fp32
    gamma_conv_ptr,       # [4, C] bf16/fp32
    A_ptr,                # [N_TOK, 4, C] bf16/fp32 pre-normed ShortConv input
    alpha_ptr,            # [N_TOK, 4] fp32 gate values
    s_gate_ptr,           # [N_TOK, 4] fp32 pre-gate dot scores
    rh_ptr,               # [N_TOK, 4] fp32 1/rms(H_m)
    rk_ptr,               # [N_TOK, 4] fp32 1/rms(K_m)
    ru_ptr,               # [N_TOK, 4] fp32 1/rms(U_m)
    N_TOK,
    C: tl.constexpr,
    INV_C: tl.constexpr,
    INV_SQRT_C: tl.constexpr,
    GATE_EPS: tl.constexpr,
    CONV_EPS: tl.constexpr,
    SIGNED_SQRT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fused M=4 Context-Gate + ShortConv Pre-RMSNorm Forward Kernel (`1` CTA per token).
    Collapses 12 RMSNorm launches + 4 dot products + signed-sqrt sigmoid + broadcast
    multiply into a two-sweep register/L1 pass without ever writing `U [B,T,4,C]` to HBM!
    """
    tok = tl.program_id(0).to(tl.int64)
    if tok >= N_TOK:
        return

    h_tok_base = tok * (4 * C)
    kv_tok_base = tok * (5 * C)

    sum_v2 = 0.0
    sum_h2_0 = 0.0
    sum_h2_1 = 0.0
    sum_h2_2 = 0.0
    sum_h2_3 = 0.0
    sum_k2_0 = 0.0
    sum_k2_1 = 0.0
    sum_k2_2 = 0.0
    sum_k2_3 = 0.0
    dot_0 = 0.0
    dot_1 = 0.0
    dot_2 = 0.0
    dot_3 = 0.0

    # Pass 1: single coalesced sweep across C accumulates V^2, H_m^2, K_m^2, AND raw <H_m*gh_m, K_m*gk_m>!
    # Notice: <RMSNorm_h(H_m), RMSNorm_k(K_m)> = rh_m * rk_m * sum_c (H_{m,c} * gh_{m,c} * K_{m,c} * gk_{m,c})!
    # Factoring `rh_m * rk_m` outside the sum eliminates an entire HBM pass!
    for c0 in range(0, C, BLOCK_C):
        c_offs = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        v = tl.load(KV_ptr + kv_tok_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        k0 = tl.load(KV_ptr + kv_tok_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        k1 = tl.load(KV_ptr + kv_tok_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        k2 = tl.load(KV_ptr + kv_tok_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        k3 = tl.load(KV_ptr + kv_tok_base + 4 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        h0 = tl.load(H_ptr + h_tok_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        h1 = tl.load(H_ptr + h_tok_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        h2 = tl.load(H_ptr + h_tok_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        h3 = tl.load(H_ptr + h_tok_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        gh0 = tl.load(gamma_h_ptr + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gh1 = tl.load(gamma_h_ptr + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gh2 = tl.load(gamma_h_ptr + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gh3 = tl.load(gamma_h_ptr + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        gk0 = tl.load(gamma_k_ptr + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gk1 = tl.load(gamma_k_ptr + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gk2 = tl.load(gamma_k_ptr + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gk3 = tl.load(gamma_k_ptr + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        sum_v2 += tl.sum(v * v, axis=0)
        sum_h2_0 += tl.sum(h0 * h0, axis=0)
        sum_h2_1 += tl.sum(h1 * h1, axis=0)
        sum_h2_2 += tl.sum(h2 * h2, axis=0)
        sum_h2_3 += tl.sum(h3 * h3, axis=0)

        sum_k2_0 += tl.sum(k0 * k0, axis=0)
        sum_k2_1 += tl.sum(k1 * k1, axis=0)
        sum_k2_2 += tl.sum(k2 * k2, axis=0)
        sum_k2_3 += tl.sum(k3 * k3, axis=0)

        dot_0 += tl.sum((h0 * gh0) * (k0 * gk0), axis=0)
        dot_1 += tl.sum((h1 * gh1) * (k1 * gk1), axis=0)
        dot_2 += tl.sum((h2 * gh2) * (k2 * gk2), axis=0)
        dot_3 += tl.sum((h3 * gh3) * (k3 * gk3), axis=0)

    mean_v2 = sum_v2 * INV_C
    rh0 = tl.rsqrt(sum_h2_0 * INV_C + GATE_EPS)
    rh1 = tl.rsqrt(sum_h2_1 * INV_C + GATE_EPS)
    rh2 = tl.rsqrt(sum_h2_2 * INV_C + GATE_EPS)
    rh3 = tl.rsqrt(sum_h2_3 * INV_C + GATE_EPS)

    rk0 = tl.rsqrt(sum_k2_0 * INV_C + GATE_EPS)
    rk1 = tl.rsqrt(sum_k2_1 * INV_C + GATE_EPS)
    rk2 = tl.rsqrt(sum_k2_2 * INV_C + GATE_EPS)
    rk3 = tl.rsqrt(sum_k2_3 * INV_C + GATE_EPS)

    s0 = dot_0 * rh0 * rk0 * INV_SQRT_C
    s1 = dot_1 * rh1 * rk1 * INV_SQRT_C
    s2 = dot_2 * rh2 * rk2 * INV_SQRT_C
    s3 = dot_3 * rh3 * rk3 * INV_SQRT_C

    if SIGNED_SQRT:
        sgn0 = tl.where(s0 > 0.0, 1.0, tl.where(s0 < 0.0, -1.0, 0.0))
        sgn1 = tl.where(s1 > 0.0, 1.0, tl.where(s1 < 0.0, -1.0, 0.0))
        sgn2 = tl.where(s2 > 0.0, 1.0, tl.where(s2 < 0.0, -1.0, 0.0))
        sgn3 = tl.where(s3 > 0.0, 1.0, tl.where(s3 < 0.0, -1.0, 0.0))
        g0 = tl.sqrt(tl.maximum(tl.abs(s0), 1e-6)) * sgn0
        g1 = tl.sqrt(tl.maximum(tl.abs(s1), 1e-6)) * sgn1
        g2 = tl.sqrt(tl.maximum(tl.abs(s2), 1e-6)) * sgn2
        g3 = tl.sqrt(tl.maximum(tl.abs(s3), 1e-6)) * sgn3
        a0 = tl.sigmoid(g0)
        a1 = tl.sigmoid(g1)
        a2 = tl.sigmoid(g2)
        a3 = tl.sigmoid(g3)
    else:
        a0 = tl.sigmoid(s0)
        a1 = tl.sigmoid(s1)
        a2 = tl.sigmoid(s2)
        a3 = tl.sigmoid(s3)

    # Exact identity: mean_c((alpha_m * V_c)^2) == alpha_m^2 * mean_v2!
    ru0 = tl.rsqrt(a0 * a0 * mean_v2 + CONV_EPS)
    ru1 = tl.rsqrt(a1 * a1 * mean_v2 + CONV_EPS)
    ru2 = tl.rsqrt(a2 * a2 * mean_v2 + CONV_EPS)
    ru3 = tl.rsqrt(a3 * a3 * mean_v2 + CONV_EPS)

    stat_base = tok * 4
    tl.store(alpha_ptr + stat_base + 0, a0)
    tl.store(alpha_ptr + stat_base + 1, a1)
    tl.store(alpha_ptr + stat_base + 2, a2)
    tl.store(alpha_ptr + stat_base + 3, a3)

    tl.store(s_gate_ptr + stat_base + 0, s0)
    tl.store(s_gate_ptr + stat_base + 1, s1)
    tl.store(s_gate_ptr + stat_base + 2, s2)
    tl.store(s_gate_ptr + stat_base + 3, s3)

    tl.store(rh_ptr + stat_base + 0, rh0)
    tl.store(rh_ptr + stat_base + 1, rh1)
    tl.store(rh_ptr + stat_base + 2, rh2)
    tl.store(rh_ptr + stat_base + 3, rh3)

    tl.store(rk_ptr + stat_base + 0, rk0)
    tl.store(rk_ptr + stat_base + 1, rk1)
    tl.store(rk_ptr + stat_base + 2, rk2)
    tl.store(rk_ptr + stat_base + 3, rk3)

    tl.store(ru_ptr + stat_base + 0, ru0)
    tl.store(ru_ptr + stat_base + 1, ru1)
    tl.store(ru_ptr + stat_base + 2, ru2)
    tl.store(ru_ptr + stat_base + 3, ru3)

    scale0 = a0 * ru0
    scale1 = a1 * ru1
    scale2 = a2 * ru2
    scale3 = a3 * ru3

    # Pass 2: Read only V[b,t,:] (already warm in L1/L2) + gamma_conv and emit A[b,t,0..3,:]!
    out_ty = A_ptr.dtype.element_ty
    for c0 in range(0, C, BLOCK_C):
        c_offs = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        v = tl.load(KV_ptr + kv_tok_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gc0 = tl.load(gamma_conv_ptr + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gc1 = tl.load(gamma_conv_ptr + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gc2 = tl.load(gamma_conv_ptr + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gc3 = tl.load(gamma_conv_ptr + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        tl.store(A_ptr + h_tok_base + 0 * C + c_offs, (v * scale0 * gc0).to(out_ty), mask=c_mask)
        tl.store(A_ptr + h_tok_base + 1 * C + c_offs, (v * scale1 * gc1).to(out_ty), mask=c_mask)
        tl.store(A_ptr + h_tok_base + 2 * C + c_offs, (v * scale2 * gc2).to(out_ty), mask=c_mask)
        tl.store(A_ptr + h_tok_base + 3 * C + c_offs, (v * scale3 * gc3).to(out_ty), mask=c_mask)


@triton.jit
def _engram_dilated_conv_silu_resid_fwd_kernel(
    H_ptr,                # [B, T, 4, C] bf16/fp32
    KV_ptr,               # [B, T, 5, C] bf16/fp32 (slice 0 = V)
    A_ptr,                # [B, T, 4, C] bf16/fp32
    alpha_ptr,            # [B, T, 4] fp32
    w_conv_ptr,           # [4*C, 4] bf16/fp32 (contiguous [M, C, 4])
    doc_ids_ptr,          # [B, T] int32 or dummy
    H_out_ptr,            # [B, T, 4, C] bf16/fp32
    T,
    C: tl.constexpr,
    MC: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    BLOCK_MC: tl.constexpr,
):
    """
    Token-Major Zero-Transpose Fused Dilated Causal ShortConv (w=4) + SiLU + Dual Residual:
      H_out[b,t,m,c] = H[b,t,m,c] + alpha[b,t,m]*V[b,t,c] + SiLU(sum_{j=0..3} w[m,c,3-j] * A[b,t-j*d,m,c])
    Completely eliminates `[B, T, M*C] <-> [B, M*C, T]` transposes and materializing `U`!
    """
    pid_bt = tl.program_id(0)
    pid_mc = tl.program_id(1)

    b = pid_bt // T
    t = pid_bt % T
    bt_idx = pid_bt.to(tl.int64)

    mc_offs = pid_mc * BLOCK_MC + tl.arange(0, BLOCK_MC)
    mc_mask = mc_offs < MC
    m_idx = mc_offs // C
    c_idx = mc_offs % C

    # Tap weights: PyTorch Conv1d cross-correlation tap lag j*DILATION uses w_conv[..., 3 - j]
    w_base = mc_offs * 4
    w_lag0 = tl.load(w_conv_ptr + w_base + 3, mask=mc_mask, other=0.0).to(tl.float32)
    w_lag1 = tl.load(w_conv_ptr + w_base + 2, mask=mc_mask, other=0.0).to(tl.float32)
    w_lag2 = tl.load(w_conv_ptr + w_base + 1, mask=mc_mask, other=0.0).to(tl.float32)
    w_lag3 = tl.load(w_conv_ptr + w_base + 0, mask=mc_mask, other=0.0).to(tl.float32)

    if HAS_DOC_IDS:
        doc_cur = tl.load(doc_ids_ptr + bt_idx)
    else:
        doc_cur = 0

    a0 = tl.load(A_ptr + bt_idx * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
    z = w_lag0 * a0

    t1 = t - DILATION
    if t1 >= 0:
        bt1 = (b * T + t1).to(tl.int64)
        ok1 = True
        if HAS_DOC_IDS:
            ok1 = tl.load(doc_ids_ptr + bt1) == doc_cur
        if ok1:
            a1 = tl.load(A_ptr + bt1 * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
            z += w_lag1 * a1

    t2 = t - 2 * DILATION
    if t2 >= 0:
        bt2 = (b * T + t2).to(tl.int64)
        ok2 = True
        if HAS_DOC_IDS:
            ok2 = tl.load(doc_ids_ptr + bt2) == doc_cur
        if ok2:
            a2 = tl.load(A_ptr + bt2 * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
            z += w_lag2 * a2

    t3 = t - 3 * DILATION
    if t3 >= 0:
        bt3 = (b * T + t3).to(tl.int64)
        ok3 = True
        if HAS_DOC_IDS:
            ok3 = tl.load(doc_ids_ptr + bt3) == doc_cur
        if ok3:
            a3 = tl.load(A_ptr + bt3 * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
            z += w_lag3 * a3

    silu_z = z * tl.sigmoid(z)

    h_val = tl.load(H_ptr + bt_idx * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
    v_val = tl.load(KV_ptr + bt_idx * (5 * C) + c_idx, mask=mc_mask, other=0.0).to(tl.float32)
    alpha_val = tl.load(alpha_ptr + bt_idx * 4 + m_idx, mask=mc_mask, other=0.0).to(tl.float32)

    out_val = h_val + alpha_val * v_val + silu_z
    tl.store(H_out_ptr + bt_idx * MC + mc_offs, out_val.to(H_out_ptr.dtype.element_ty), mask=mc_mask)


@triton.jit
def _engram_conv_silu_q_dw_bwd_kernel(
    dOut_ptr,             # [N_TOK, MC] bf16/fp32
    A_ptr,                # [N_TOK, MC] bf16/fp32
    w_conv_ptr,           # [MC, 4] bf16/fp32
    doc_ids_ptr,          # [N_TOK] int32 or dummy
    Q_ptr,                # [N_TOK, MC] bf16/fp32 output: dOut * dSiLU(Z)
    dw_partial_ptr,       # [NUM_SPLITS, MC, 4] fp32 disjoint split partials (ZERO atomics!)
    N_TOK,
    T,
    MC: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    TOKENS_PER_SPLIT,
    BLOCK_MC: tl.constexpr,
):
    """
    Backward Kernel 1 (Lock-Free Disjoint Split-CTA):
    Each CTA `(split_id, pid_mc)` sweeps a contiguous token range `[tok_start, tok_end)`,
    recomputes `Z` on the fly from `A` (saving `[B,T,4,C]` activation HBM), writes
    `Q = dOut * dSiLU(Z)`, and writes disjoint `dw_partial[split_id, mc, 0..3]` in FP32
    with ZERO global atomics!
    """
    split_id = tl.program_id(0)
    pid_mc = tl.program_id(1)

    mc_offs = pid_mc * BLOCK_MC + tl.arange(0, BLOCK_MC)
    mc_mask = mc_offs < MC

    w_base = mc_offs * 4
    w_lag0 = tl.load(w_conv_ptr + w_base + 3, mask=mc_mask, other=0.0).to(tl.float32)
    w_lag1 = tl.load(w_conv_ptr + w_base + 2, mask=mc_mask, other=0.0).to(tl.float32)
    w_lag2 = tl.load(w_conv_ptr + w_base + 1, mask=mc_mask, other=0.0).to(tl.float32)
    w_lag3 = tl.load(w_conv_ptr + w_base + 0, mask=mc_mask, other=0.0).to(tl.float32)

    dw0 = tl.zeros([BLOCK_MC], dtype=tl.float32)  # w[..., 0] <-> lag 3
    dw1 = tl.zeros([BLOCK_MC], dtype=tl.float32)  # w[..., 1] <-> lag 2
    dw2 = tl.zeros([BLOCK_MC], dtype=tl.float32)  # w[..., 2] <-> lag 1
    dw3 = tl.zeros([BLOCK_MC], dtype=tl.float32)  # w[..., 3] <-> lag 0

    tok_start = split_id * TOKENS_PER_SPLIT
    tok_end = tl.minimum(N_TOK, tok_start + TOKENS_PER_SPLIT)
    q_ty = Q_ptr.dtype.element_ty

    tok = tok_start
    while tok < tok_end:
        b = tok // T
        t = tok % T
        bt = tok.to(tl.int64)

        if HAS_DOC_IDS:
            doc_cur = tl.load(doc_ids_ptr + bt)
        else:
            doc_cur = 0

        a0 = tl.load(A_ptr + bt * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
        z = w_lag0 * a0

        a1 = tl.zeros([BLOCK_MC], dtype=tl.float32)
        t1 = t - DILATION
        if t1 >= 0:
            bt1 = (b * T + t1).to(tl.int64)
            ok1 = True
            if HAS_DOC_IDS:
                ok1 = tl.load(doc_ids_ptr + bt1) == doc_cur
            if ok1:
                a1 = tl.load(A_ptr + bt1 * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
                z += w_lag1 * a1

        a2 = tl.zeros([BLOCK_MC], dtype=tl.float32)
        t2 = t - 2 * DILATION
        if t2 >= 0:
            bt2 = (b * T + t2).to(tl.int64)
            ok2 = True
            if HAS_DOC_IDS:
                ok2 = tl.load(doc_ids_ptr + bt2) == doc_cur
            if ok2:
                a2 = tl.load(A_ptr + bt2 * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
                z += w_lag2 * a2

        a3 = tl.zeros([BLOCK_MC], dtype=tl.float32)
        t3 = t - 3 * DILATION
        if t3 >= 0:
            bt3 = (b * T + t3).to(tl.int64)
            ok3 = True
            if HAS_DOC_IDS:
                ok3 = tl.load(doc_ids_ptr + bt3) == doc_cur
            if ok3:
                a3 = tl.load(A_ptr + bt3 * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
                z += w_lag3 * a3

        sig = tl.sigmoid(z)
        dsilu = sig * (1.0 + z * (1.0 - sig))
        dout = tl.load(dOut_ptr + bt * MC + mc_offs, mask=mc_mask, other=0.0).to(tl.float32)
        q = dout * dsilu
        tl.store(Q_ptr + bt * MC + mc_offs, q.to(q_ty), mask=mc_mask)

        dw3 += q * a0
        dw2 += q * a1
        dw1 += q * a2
        dw0 += q * a3
        tok += 1

    part_base = (split_id.to(tl.int64) * MC + mc_offs) * 4
    tl.store(dw_partial_ptr + part_base + 0, dw0, mask=mc_mask)
    tl.store(dw_partial_ptr + part_base + 1, dw1, mask=mc_mask)
    tl.store(dw_partial_ptr + part_base + 2, dw2, mask=mc_mask)
    tl.store(dw_partial_ptr + part_base + 3, dw3, mask=mc_mask)


@triton.jit
def _engram_gate_prenorm_bwd_kernel(
    dOut_ptr,             # [N_TOK, 4, C] bf16/fp32
    Q_ptr,                # [N_TOK, 4, C] bf16/fp32
    H_ptr,                # [N_TOK, 4, C] bf16/fp32
    KV_ptr,               # [N_TOK, 5, C] bf16/fp32
    gamma_h_ptr,          # [4, C] bf16/fp32
    gamma_k_ptr,          # [4, C] bf16/fp32
    gamma_conv_ptr,       # [4, C] bf16/fp32
    w_conv_ptr,           # [4, C, 4] bf16/fp32
    alpha_ptr,            # [N_TOK, 4] fp32
    s_gate_ptr,           # [N_TOK, 4] fp32
    rh_ptr,               # [N_TOK, 4] fp32
    rk_ptr,               # [N_TOK, 4] fp32
    ru_ptr,               # [N_TOK, 4] fp32
    doc_ids_ptr,          # [N_TOK] int32 or dummy
    dH_ptr,               # [N_TOK, 4, C] bf16/fp32
    dKV_ptr,              # [N_TOK, 5, C] bf16/fp32
    dgamma_partial_ptr,   # [NUM_SPLITS, 12, C] fp32: [0:4]=dgh, [4:8]=dgk, [8:12]=dgc (ZERO atomics!)
    N_TOK,
    T,
    C: tl.constexpr,
    INV_C: tl.constexpr,
    INV_SQRT_C: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    SIGNED_SQRT: tl.constexpr,
    TOKENS_PER_SPLIT,
    BLOCK_C: tl.constexpr,
):
    """
    Backward Kernel 2 (Lock-Free Split-CTA across token ranges):
    Each CTA `split_id` sweeps tokens `[tok_start, tok_end)` and for each token:
      Pass 1 across C: gathers future `Q[t + 3j]` taps (`dA`), accumulates `dot_u = mean(dA * U)`
                       and `d_alpha` simultaneously!
      Pass 2 across C: forms exact `dU = dOut + dU_conv`, emits `dV`, `dK_0..3`, `dH_0..3`, and
                       accumulates `dgamma_h`, `dgamma_k`, `dgamma_conv` in FP32 registers before
                       flushing once to `dgamma_partial[split_id]` (ZERO global atomics!).
    Note: To keep register pressure low (`0` cubin spills), each CTA handles one `(split_id, c_block)`?
    Wait: RMSNorm backward across `C` needs the full-row scalars (`dot_u[0..3]`, `d_alpha[0..3]`,
    `dot_h[0..3]`, `dot_k[0..3]`). So we split into two ultra-clean kernels:
      2A (`1` CTA per token): computes token-wise scalars `dot_u[4]`, `ds_gate[4]`, `dot_h[4]`, `dot_k[4]`
                              AND emits `dH[tok]` + `dKV[tok]` directly (`0` cross-token writes!);
      2B (`(NUM_SPLITS, cdiv(C, BLOCK_C))`): accumulates `dgamma_h`, `dgamma_k`, `dgamma_conv`
                              in registers across `TOKENS_PER_SPLIT` with `0` atomics and `0` spills!
    """
    pass


@triton.jit
def _engram_token_grad_bwd_kernel(
    dOut_ptr,             # [N_TOK, 4, C]
    Q_ptr,                # [N_TOK, 4, C]
    H_ptr,                # [N_TOK, 4, C]
    KV_ptr,               # [N_TOK, 5, C]
    gamma_h_ptr,          # [4, C]
    gamma_k_ptr,          # [4, C]
    gamma_conv_ptr,       # [4, C]
    w_conv_ptr,           # [4, C, 4]
    alpha_ptr,            # [N_TOK, 4] fp32
    s_gate_ptr,           # [N_TOK, 4] fp32
    rh_ptr,               # [N_TOK, 4] fp32
    rk_ptr,               # [N_TOK, 4] fp32
    ru_ptr,               # [N_TOK, 4] fp32
    doc_ids_ptr,          # [N_TOK] int32 or dummy
    dot_u_out_ptr,        # [N_TOK, 4] fp32 (saved for dgamma_conv kernel 2B)
    ds_out_ptr,           # [N_TOK, 4] fp32 (saved for dgamma_h / dgamma_k kernel 2B)
    dH_ptr,               # [N_TOK, 4, C]
    dKV_ptr,              # [N_TOK, 5, C]
    N_TOK,
    T,
    C: tl.constexpr,
    MC: tl.constexpr,
    INV_C: tl.constexpr,
    INV_SQRT_C: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    SIGNED_SQRT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Backward Kernel 2A (`1` CTA per token, `0` spills, `0` atomics):
    Computes exact `dH[tok, 0..3, :]` and `dKV[tok, 0..4, :]` in 2 sweeps across C!
    Algebraic check on why only 2 sweeps suffice:
      Sweep 1:
        - Gather future causal-conv transpose taps `dA_raw_{m,c} = sum_{j=0..3} w_{m,c,3-j} * Q[t+j*d, m, c]`
        - `dA_{m,c} = dA_raw_{m,c} * gamma_conv_{m,c}`
        - Since `U_{m,c} = alpha_m * V_c`:
            `dot_u_m = (1/C) sum_c dA_{m,c} * U_{m,c} = alpha_m * (1/C) sum_c dA_{m,c} * V_c`
          AND look at `d_alpha_m = sum_c dU_{m,c} * V_c` where `dU_{m,c} = dOut_{m,c} + ru_m * (dA_{m,c} - U_{m,c} * ru_m^2 * dot_u_m)`:
            `sum_c dU_{m,c} * V_c = sum_c dOut_{m,c} * V_c + ru_m * (sum_c dA_{m,c} * V_c - alpha_m * ru_m^2 * dot_u_m * sum_c V_c^2)`!
          WAIT A SECOND! That means `d_alpha_m` AND `ds_m` are ALREADY KNOWN AT THE END OF SWEEP 1!
          AND what about `RMSNorm_h(H_m)` and `RMSNorm_k(K_m)` backward?!
            For `s_m = (1/sqrt(C)) * rh_m * rk_m * sum_c (H_{m,c} * gh_{m,c} * K_{m,c} * gk_{m,c})`:
            Notice `sum_c (ds_m * d(s_m)/d(nh_{m,c}) * nh_{m,c}) == ds_m * s_m` IDENTICALLY!!
            Let's verify that!!
            In `s_m = (1/sqrt(C)) * <nh_m, nk_m>` where `nh_m = H_m * rh_m * gh_m`:
              `dL/d(H_m * rh_m) = (ds_m / sqrt(C)) * nk_m * gh_m`
              Its dot product with `(H_m * rh_m)` divided by `C` is:
              `(1/C) * sum_c [(ds_m / sqrt(C)) * nk_{m,c} * gh_{m,c} * H_{m,c} * rh_m] = (ds_m / C) * s_m`!!
            IT IS AN EXACT CLOSED-FORM SCALAR IDENTITY (`dot_h_m = ds_m * s_m * INV_C`, `dot_k_m = ds_m * s_m * INV_C`) — NO EXTRA REDUCTION SWEEP NEEDED!!
    """
    bt = tl.program_id(0).to(tl.int64)
    if bt >= N_TOK:
        return

    b = bt // T
    t = bt % T
    if HAS_DOC_IDS:
        doc_cur = tl.load(doc_ids_ptr + bt)
    else:
        doc_cur = 0

    t1 = t + DILATION
    ok1 = t1 < T
    bt1 = (b * T + t1).to(tl.int64)
    if HAS_DOC_IDS and ok1:
        ok1 = tl.load(doc_ids_ptr + bt1) == doc_cur

    t2 = t + 2 * DILATION
    ok2 = t2 < T
    bt2 = (b * T + t2).to(tl.int64)
    if HAS_DOC_IDS and ok2:
        ok2 = tl.load(doc_ids_ptr + bt2) == doc_cur

    t3 = t + 3 * DILATION
    ok3 = t3 < T
    bt3 = (b * T + t3).to(tl.int64)
    if HAS_DOC_IDS and ok3:
        ok3 = tl.load(doc_ids_ptr + bt3) == doc_cur

    stat_base = bt * 4
    a0 = tl.load(alpha_ptr + stat_base + 0)
    a1 = tl.load(alpha_ptr + stat_base + 1)
    a2 = tl.load(alpha_ptr + stat_base + 2)
    a3 = tl.load(alpha_ptr + stat_base + 3)

    ru0 = tl.load(ru_ptr + stat_base + 0)
    ru1 = tl.load(ru_ptr + stat_base + 1)
    ru2 = tl.load(ru_ptr + stat_base + 2)
    ru3 = tl.load(ru_ptr + stat_base + 3)

    sum_v2 = 0.0
    sum_da_v0 = 0.0
    sum_da_v1 = 0.0
    sum_da_v2 = 0.0
    sum_da_v3 = 0.0
    sum_do_v0 = 0.0
    sum_do_v1 = 0.0
    sum_do_v2 = 0.0
    sum_do_v3 = 0.0

    kv_tok_base = bt * (5 * C)
    h_tok_base = bt * MC

    # Sweep 1: reduce sum(V^2), sum(dA_m * V), sum(dOut_m * V)
    for c0 in range(0, C, BLOCK_C):
        c_offs = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        v = tl.load(KV_ptr + kv_tok_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        sum_v2 += tl.sum(v * v, axis=0)

        for m in tl.static_range(0, 4):
            mc_offs = m * C + c_offs
            w_base = mc_offs * 4
            wl0 = tl.load(w_conv_ptr + w_base + 3, mask=c_mask, other=0.0).to(tl.float32)
            wl1 = tl.load(w_conv_ptr + w_base + 2, mask=c_mask, other=0.0).to(tl.float32)
            wl2 = tl.load(w_conv_ptr + w_base + 1, mask=c_mask, other=0.0).to(tl.float32)
            wl3 = tl.load(w_conv_ptr + w_base + 0, mask=c_mask, other=0.0).to(tl.float32)

            da_raw = wl0 * tl.load(Q_ptr + bt * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok1:
                da_raw += wl1 * tl.load(Q_ptr + bt1 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok2:
                da_raw += wl2 * tl.load(Q_ptr + bt2 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok3:
                da_raw += wl3 * tl.load(Q_ptr + bt3 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)

            gc = tl.load(gamma_conv_ptr + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            da = da_raw * gc
            do = tl.load(dOut_ptr + h_tok_base + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            dav = tl.sum(da * v, axis=0)
            dov = tl.sum(do * v, axis=0)
            if m == 0:
                sum_da_v0 += dav
                sum_do_v0 += dov
            elif m == 1:
                sum_da_v1 += dav
                sum_do_v1 += dov
            elif m == 2:
                sum_da_v2 += dav
                sum_do_v2 += dov
            else:
                sum_da_v3 += dav
                sum_do_v3 += dov

    dot_u0 = a0 * sum_da_v0 * INV_C
    dot_u1 = a1 * sum_da_v1 * INV_C
    dot_u2 = a2 * sum_da_v2 * INV_C
    dot_u3 = a3 * sum_da_v3 * INV_C

    dalpha0 = sum_do_v0 + ru0 * (sum_da_v0 - a0 * ru0 * ru0 * dot_u0 * sum_v2)
    dalpha1 = sum_do_v1 + ru1 * (sum_da_v1 - a1 * ru1 * ru1 * dot_u1 * sum_v2)
    dalpha2 = sum_do_v2 + ru2 * (sum_da_v2 - a2 * ru2 * ru2 * dot_u2 * sum_v2)
    dalpha3 = sum_do_v3 + ru3 * (sum_da_v3 - a3 * ru3 * ru3 * dot_u3 * sum_v2)

    s0 = tl.load(s_gate_ptr + stat_base + 0)
    s1 = tl.load(s_gate_ptr + stat_base + 1)
    s2 = tl.load(s_gate_ptr + stat_base + 2)
    s3 = tl.load(s_gate_ptr + stat_base + 3)

    if SIGNED_SQRT:
        # Exact PyTorch autograd subgradient of `sigmoid(clamp_min(|s|, 1e-6).sqrt() * sign(s))`:
        # 0.0 for |s| < 1e-6, and `alpha * (1 - alpha) / (2 * sqrt(|s|))` for |s| >= 1e-6!
        abs0 = tl.abs(s0)
        abs1 = tl.abs(s1)
        abs2 = tl.abs(s2)
        abs3 = tl.abs(s3)
        ds0 = tl.where(abs0 >= 1e-6, dalpha0 * a0 * (1.0 - a0) * 0.5 / tl.sqrt(tl.maximum(abs0, 1e-12)), 0.0)
        ds1 = tl.where(abs1 >= 1e-6, dalpha1 * a1 * (1.0 - a1) * 0.5 / tl.sqrt(tl.maximum(abs1, 1e-12)), 0.0)
        ds2 = tl.where(abs2 >= 1e-6, dalpha2 * a2 * (1.0 - a2) * 0.5 / tl.sqrt(tl.maximum(abs2, 1e-12)), 0.0)
        ds3 = tl.where(abs3 >= 1e-6, dalpha3 * a3 * (1.0 - a3) * 0.5 / tl.sqrt(tl.maximum(abs3, 1e-12)), 0.0)
    else:
        ds0 = dalpha0 * a0 * (1.0 - a0)
        ds1 = dalpha1 * a1 * (1.0 - a1)
        ds2 = dalpha2 * a2 * (1.0 - a2)
        ds3 = dalpha3 * a3 * (1.0 - a3)

    tl.store(dot_u_out_ptr + stat_base + 0, dot_u0)
    tl.store(dot_u_out_ptr + stat_base + 1, dot_u1)
    tl.store(dot_u_out_ptr + stat_base + 2, dot_u2)
    tl.store(dot_u_out_ptr + stat_base + 3, dot_u3)

    tl.store(ds_out_ptr + stat_base + 0, ds0)
    tl.store(ds_out_ptr + stat_base + 1, ds1)
    tl.store(ds_out_ptr + stat_base + 2, ds2)
    tl.store(ds_out_ptr + stat_base + 3, ds3)

    rh0 = tl.load(rh_ptr + stat_base + 0)
    rh1 = tl.load(rh_ptr + stat_base + 1)
    rh2 = tl.load(rh_ptr + stat_base + 2)
    rh3 = tl.load(rh_ptr + stat_base + 3)

    rk0 = tl.load(rk_ptr + stat_base + 0)
    rk1 = tl.load(rk_ptr + stat_base + 1)
    rk2 = tl.load(rk_ptr + stat_base + 2)
    rk3 = tl.load(rk_ptr + stat_base + 3)

    # Exact closed-form RMSNorm dot products: dot_h_m = dot_k_m = ds_m * s_m * INV_C!
    dh_ty = dH_ptr.dtype.element_ty
    dkv_ty = dKV_ptr.dtype.element_ty

    # Sweep 2: emit exact dV, dK_0..3, dH_0..3
    for c0 in range(0, C, BLOCK_C):
        c_offs = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        v = tl.load(KV_ptr + kv_tok_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        dv_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

        for m in tl.static_range(0, 4):
            mc_offs = m * C + c_offs
            w_base = mc_offs * 4
            wl0 = tl.load(w_conv_ptr + w_base + 3, mask=c_mask, other=0.0).to(tl.float32)
            wl1 = tl.load(w_conv_ptr + w_base + 2, mask=c_mask, other=0.0).to(tl.float32)
            wl2 = tl.load(w_conv_ptr + w_base + 1, mask=c_mask, other=0.0).to(tl.float32)
            wl3 = tl.load(w_conv_ptr + w_base + 0, mask=c_mask, other=0.0).to(tl.float32)

            da_raw = wl0 * tl.load(Q_ptr + bt * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok1:
                da_raw += wl1 * tl.load(Q_ptr + bt1 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok2:
                da_raw += wl2 * tl.load(Q_ptr + bt2 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok3:
                da_raw += wl3 * tl.load(Q_ptr + bt3 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)

            gc = tl.load(gamma_conv_ptr + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            da = da_raw * gc
            do = tl.load(dOut_ptr + h_tok_base + mc_offs, mask=c_mask, other=0.0).to(tl.float32)

            if m == 0:
                am, rum, dot_um, dsm, sm, rhm, rkm = a0, ru0, dot_u0, ds0, s0, rh0, rk0
            elif m == 1:
                am, rum, dot_um, dsm, sm, rhm, rkm = a1, ru1, dot_u1, ds1, s1, rh1, rk1
            elif m == 2:
                am, rum, dot_um, dsm, sm, rhm, rkm = a2, ru2, dot_u2, ds2, s2, rh2, rk2
            else:
                am, rum, dot_um, dsm, sm, rhm, rkm = a3, ru3, dot_u3, ds3, s3, rh3, rk3

            u_mc = am * v
            du_mc = do + rum * (da - u_mc * rum * rum * dot_um)
            dv_acc += am * du_mc

            hm = tl.load(H_ptr + h_tok_base + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            km = tl.load(KV_ptr + kv_tok_base + (m + 1) * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
            ghm = tl.load(gamma_h_ptr + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            gkm = tl.load(gamma_k_ptr + mc_offs, mask=c_mask, other=0.0).to(tl.float32)

            # Exact RMSNorm backward via `dot_m = dsm * sm * INV_C` (note `sm` already includes `rhm * rkm`!):
            # dH_gate = rhm * (dsm * INV_SQRT_C * (km * rkm * gkm) * ghm - hm * rhm * (dsm * sm * INV_C))
            scale_s = dsm * INV_SQRT_C
            dot_norm = dsm * sm * INV_C
            dh_gate = rhm * (scale_s * (km * rkm * gkm) * ghm - hm * rhm * dot_norm)
            dk_gate = rkm * (scale_s * (hm * rhm * ghm) * gkm - km * rkm * dot_norm)

            tl.store(dH_ptr + h_tok_base + mc_offs, (do + dh_gate).to(dh_ty), mask=c_mask)
            tl.store(dKV_ptr + kv_tok_base + (m + 1) * C + c_offs, dk_gate.to(dkv_ty), mask=c_mask)

        tl.store(dKV_ptr + kv_tok_base + c_offs, dv_acc.to(dkv_ty), mask=c_mask)


@triton.jit
def _engram_dgamma_split_bwd_kernel(
    Q_ptr,                # [N_TOK, 4, C]
    H_ptr,                # [N_TOK, 4, C]
    KV_ptr,               # [N_TOK, 5, C]
    gamma_h_ptr,          # [4, C]
    gamma_k_ptr,          # [4, C]
    w_conv_ptr,           # [4, C, 4]
    alpha_ptr,            # [N_TOK, 4] fp32
    ds_ptr,               # [N_TOK, 4] fp32
    rh_ptr,               # [N_TOK, 4] fp32
    rk_ptr,               # [N_TOK, 4] fp32
    ru_ptr,               # [N_TOK, 4] fp32
    doc_ids_ptr,          # [N_TOK] int32 or dummy
    dgamma_partial_ptr,   # [NUM_SPLITS, 12, C] fp32: [0:4]=dgh, [4:8]=dgk, [8:12]=dgc
    N_TOK,
    T,
    C: tl.constexpr,
    MC: tl.constexpr,
    INV_SQRT_C: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    TOKENS_PER_SPLIT,
    BLOCK_C: tl.constexpr,
):
    """
    Backward Kernel 2B (Disjoint Split-CTA Lock-Free Norm-Weight Gradient Reduction):
    Each CTA `(split_id, pid_c)` owns channel slice `c_offs` across `[tok_start, tok_end)`
    and accumulates `dgamma_h[0..3]`, `dgamma_k[0..3]`, and `dgamma_conv[0..3]` in FP32
    registers before writing once to `dgamma_partial[split_id, 0..11, c_offs]` (`0` atomics!).
    """
    split_id = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    gh0 = tl.load(gamma_h_ptr + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gh1 = tl.load(gamma_h_ptr + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gh2 = tl.load(gamma_h_ptr + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gh3 = tl.load(gamma_h_ptr + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    gk0 = tl.load(gamma_k_ptr + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gk1 = tl.load(gamma_k_ptr + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gk2 = tl.load(gamma_k_ptr + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gk3 = tl.load(gamma_k_ptr + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    dgh0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgh1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgh2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgh3 = tl.zeros([BLOCK_C], dtype=tl.float32)

    dgk0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgk1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgk2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgk3 = tl.zeros([BLOCK_C], dtype=tl.float32)

    dgc0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgc1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgc2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    dgc3 = tl.zeros([BLOCK_C], dtype=tl.float32)

    tok_start = split_id * TOKENS_PER_SPLIT
    tok_end = tl.minimum(N_TOK, tok_start + TOKENS_PER_SPLIT)

    tok = tok_start
    while tok < tok_end:
        b = tok // T
        t = tok % T
        bt = tok.to(tl.int64)
        if HAS_DOC_IDS:
            doc_cur = tl.load(doc_ids_ptr + bt)
        else:
            doc_cur = 0

        t1 = t + DILATION
        ok1 = t1 < T
        bt1 = (b * T + t1).to(tl.int64)
        if HAS_DOC_IDS and ok1:
            ok1 = tl.load(doc_ids_ptr + bt1) == doc_cur

        t2 = t + 2 * DILATION
        ok2 = t2 < T
        bt2 = (b * T + t2).to(tl.int64)
        if HAS_DOC_IDS and ok2:
            ok2 = tl.load(doc_ids_ptr + bt2) == doc_cur

        t3 = t + 3 * DILATION
        ok3 = t3 < T
        bt3 = (b * T + t3).to(tl.int64)
        if HAS_DOC_IDS and ok3:
            ok3 = tl.load(doc_ids_ptr + bt3) == doc_cur

        kv_tok_base = bt * (5 * C)
        h_tok_base = bt * MC
        stat_base = bt * 4

        v = tl.load(KV_ptr + kv_tok_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        for m in tl.static_range(0, 4):
            mc_offs = m * C + c_offs
            w_base = mc_offs * 4
            wl0 = tl.load(w_conv_ptr + w_base + 3, mask=c_mask, other=0.0).to(tl.float32)
            wl1 = tl.load(w_conv_ptr + w_base + 2, mask=c_mask, other=0.0).to(tl.float32)
            wl2 = tl.load(w_conv_ptr + w_base + 1, mask=c_mask, other=0.0).to(tl.float32)
            wl3 = tl.load(w_conv_ptr + w_base + 0, mask=c_mask, other=0.0).to(tl.float32)

            da_raw = wl0 * tl.load(Q_ptr + bt * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok1:
                da_raw += wl1 * tl.load(Q_ptr + bt1 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok2:
                da_raw += wl2 * tl.load(Q_ptr + bt2 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            if ok3:
                da_raw += wl3 * tl.load(Q_ptr + bt3 * MC + mc_offs, mask=c_mask, other=0.0).to(tl.float32)

            am = tl.load(alpha_ptr + stat_base + m)
            dsm = tl.load(ds_ptr + stat_base + m)
            rhm = tl.load(rh_ptr + stat_base + m)
            rkm = tl.load(rk_ptr + stat_base + m)
            rum = tl.load(ru_ptr + stat_base + m)

            hm = tl.load(H_ptr + h_tok_base + mc_offs, mask=c_mask, other=0.0).to(tl.float32)
            km = tl.load(KV_ptr + kv_tok_base + (m + 1) * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

            base_hk = (dsm * INV_SQRT_C) * (hm * rhm) * (km * rkm)
            dgc_step = da_raw * (am * v * rum)

            if m == 0:
                dgh0 += base_hk * gk0
                dgk0 += base_hk * gh0
                dgc0 += dgc_step
            elif m == 1:
                dgh1 += base_hk * gk1
                dgk1 += base_hk * gh1
                dgc1 += dgc_step
            elif m == 2:
                dgh2 += base_hk * gk2
                dgk2 += base_hk * gh2
                dgc2 += dgc_step
            else:
                dgh3 += base_hk * gk3
                dgk3 += base_hk * gh3
                dgc3 += dgc_step

        tok += 1

    split_base = split_id.to(tl.int64) * (12 * C)
    tl.store(dgamma_partial_ptr + split_base + 0 * C + c_offs, dgh0, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 1 * C + c_offs, dgh1, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 2 * C + c_offs, dgh2, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 3 * C + c_offs, dgh3, mask=c_mask)

    tl.store(dgamma_partial_ptr + split_base + 4 * C + c_offs, dgk0, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 5 * C + c_offs, dgk1, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 6 * C + c_offs, dgk2, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 7 * C + c_offs, dgk3, mask=c_mask)

    tl.store(dgamma_partial_ptr + split_base + 8 * C + c_offs, dgc0, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 9 * C + c_offs, dgc1, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 10 * C + c_offs, dgc2, mask=c_mask)
    tl.store(dgamma_partial_ptr + split_base + 11 * C + c_offs, dgc3, mask=c_mask)


class _FusedEngramContextAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        H: torch.Tensor,             # [B, T, 4, C]
        E: torch.Tensor,             # [B, T, D_mem]
        W_kv: torch.Tensor,          # [5*C, D_mem]
        gamma_h: torch.Tensor,       # [4, C]
        gamma_k: torch.Tensor,       # [4, C]
        gamma_conv: torch.Tensor,    # [4, C]
        w_conv: torch.Tensor,        # [4, C, 4]
        doc_ids: Optional[torch.Tensor],
        dilation: int,
        gate_eps: float,
        conv_eps: float,
        signed_sqrt_gate: bool,
        block_c: int,
        block_mc: int,
        num_splits: int,
    ) -> torch.Tensor:
        assert H.is_cuda and H.ndim == 4 and H.shape[2] == 4
        Hc = H.contiguous()
        Ec = E.contiguous()
        B, T, M, C = Hc.shape
        N_TOK = B * T
        MC = M * C

        # 1. Single merged cuBLAS GEMM replacing 5 separate linear projections
        KV = F.linear(Ec, W_kv).view(B, T, 5, C).contiguous()

        A = torch.empty_like(Hc)
        alpha = torch.empty((N_TOK, 4), device=Hc.device, dtype=torch.float32)
        s_gate = torch.empty((N_TOK, 4), device=Hc.device, dtype=torch.float32)
        rh = torch.empty((N_TOK, 4), device=Hc.device, dtype=torch.float32)
        rk = torch.empty((N_TOK, 4), device=Hc.device, dtype=torch.float32)
        ru = torch.empty((N_TOK, 4), device=Hc.device, dtype=torch.float32)

        inv_c = 1.0 / float(C)
        inv_sqrt_c = 1.0 / math.sqrt(float(C))

        # 2. Fused Multi-Branch Gate + ShortConv PreNorm
        _engram_gate_prenorm_fwd_kernel[(N_TOK,)](
            Hc,
            KV,
            gamma_h,
            gamma_k,
            gamma_conv,
            A,
            alpha,
            s_gate,
            rh,
            rk,
            ru,
            N_TOK=N_TOK,
            C=C,
            INV_C=inv_c,
            INV_SQRT_C=inv_sqrt_c,
            GATE_EPS=gate_eps,
            CONV_EPS=conv_eps,
            SIGNED_SQRT=signed_sqrt_gate,
            BLOCK_C=block_c,
            num_warps=4 if block_c <= 256 else 8,
        )

        # 3. Token-Major Zero-Transpose Fused Dilated ShortConv + SiLU + Dual Residual
        H_out = torch.empty_like(Hc)
        has_docs = doc_ids is not None
        doc_buf = doc_ids.contiguous() if has_docs else torch.empty((1,), device=Hc.device, dtype=torch.int32)
        _engram_dilated_conv_silu_resid_fwd_kernel[(N_TOK, triton.cdiv(MC, block_mc))](
            Hc,
            KV,
            A,
            alpha,
            w_conv,
            doc_buf,
            H_out,
            T=T,
            C=C,
            MC=MC,
            DILATION=dilation,
            HAS_DOC_IDS=has_docs,
            BLOCK_MC=block_mc,
            num_warps=4,
        )

        ctx.save_for_backward(
            Hc, Ec, KV, A, W_kv, gamma_h, gamma_k, gamma_conv, w_conv, alpha, s_gate, rh, rk, ru, doc_buf
        )
        ctx.has_docs = has_docs
        ctx.dilation = dilation
        ctx.signed_sqrt_gate = signed_sqrt_gate
        ctx.block_c = block_c
        ctx.block_mc = block_mc
        ctx.num_splits = num_splits
        return H_out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            Hc,
            Ec,
            KV,
            A,
            W_kv,
            gamma_h,
            gamma_k,
            gamma_conv,
            w_conv,
            alpha,
            s_gate,
            rh,
            rk,
            ru,
            doc_buf,
        ) = ctx.saved_tensors
        B, T, M, C = Hc.shape
        N_TOK = B * T
        MC = M * C
        dOut = grad_out.contiguous()

        num_splits = min(ctx.num_splits, max(1, N_TOK))
        tokens_per_split = triton.cdiv(N_TOK, num_splits)
        actual_splits = triton.cdiv(N_TOK, tokens_per_split)

        Q = torch.empty_like(Hc)
        dw_partial = torch.empty((actual_splits, MC, 4), device=Hc.device, dtype=torch.float32)

        # 1. Lock-free Split-CTA Conv+SiLU Q & dw_conv partial reduction (`0` atomics)
        _engram_conv_silu_q_dw_bwd_kernel[(actual_splits, triton.cdiv(MC, ctx.block_mc))](
            dOut,
            A,
            w_conv,
            doc_buf,
            Q,
            dw_partial,
            N_TOK=N_TOK,
            T=T,
            MC=MC,
            DILATION=ctx.dilation,
            HAS_DOC_IDS=ctx.has_docs,
            TOKENS_PER_SPLIT=tokens_per_split,
            BLOCK_MC=ctx.block_mc,
            num_warps=4,
        )
        dw_conv = dw_partial.sum(dim=0).view(M, C, 4).to(w_conv.dtype)

        # 2A. Token-parallel exact dH and dKV backward (`0` atomics)
        dH = torch.empty_like(Hc)
        dKV = torch.empty_like(KV)
        dot_u = torch.empty((N_TOK, 4), device=Hc.device, dtype=torch.float32)
        ds_gate = torch.empty((N_TOK, 4), device=Hc.device, dtype=torch.float32)

        inv_c = 1.0 / float(C)
        inv_sqrt_c = 1.0 / math.sqrt(float(C))

        _engram_token_grad_bwd_kernel[(N_TOK,)](
            dOut,
            Q,
            Hc,
            KV,
            gamma_h,
            gamma_k,
            gamma_conv,
            w_conv,
            alpha,
            s_gate,
            rh,
            rk,
            ru,
            doc_buf,
            dot_u,
            ds_gate,
            dH,
            dKV,
            N_TOK=N_TOK,
            T=T,
            C=C,
            MC=MC,
            INV_C=inv_c,
            INV_SQRT_C=inv_sqrt_c,
            DILATION=ctx.dilation,
            HAS_DOC_IDS=ctx.has_docs,
            SIGNED_SQRT=ctx.signed_sqrt_gate,
            BLOCK_C=ctx.block_c,
            num_warps=4 if ctx.block_c <= 256 else 8,
        )

        # 2B. Lock-free Split-CTA dgamma_h / dgamma_k / dgamma_conv reduction (`0` atomics)
        dgamma_partial = torch.empty((actual_splits, 12, C), device=Hc.device, dtype=torch.float32)
        _engram_dgamma_split_bwd_kernel[(actual_splits, triton.cdiv(C, ctx.block_c))](
            Q,
            Hc,
            KV,
            gamma_h,
            gamma_k,
            w_conv,
            alpha,
            ds_gate,
            rh,
            rk,
            ru,
            doc_buf,
            dgamma_partial,
            N_TOK=N_TOK,
            T=T,
            C=C,
            MC=MC,
            INV_SQRT_C=inv_sqrt_c,
            DILATION=ctx.dilation,
            HAS_DOC_IDS=ctx.has_docs,
            TOKENS_PER_SPLIT=tokens_per_split,
            BLOCK_C=ctx.block_c,
            num_warps=4,
        )
        dg_all = dgamma_partial.sum(dim=0)
        dgamma_h = dg_all[0:4].to(gamma_h.dtype)
        dgamma_k = dg_all[4:8].to(gamma_k.dtype)
        dgamma_conv = dg_all[8:12].to(gamma_conv.dtype)

        # 3. Merged KV cuBLAS backward GEMMs (`dE` and `dW_kv`)
        dKV_2d = dKV.view(N_TOK, 5 * C)
        E_2d = Ec.view(N_TOK, -1)
        dE = torch.matmul(dKV_2d, W_kv).view_as(Ec)
        dW_kv = torch.matmul(dKV_2d.t(), E_2d)

        return (
            dH,
            dE,
            dW_kv,
            dgamma_h,
            dgamma_k,
            dgamma_conv,
            dw_conv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedEngramContextOp(nn.Module):
    """
    Production Round-4 RSI `CANDIDATE` (`Trial 4`) and intermediate RSI ablation stages (`Trial 2/3`)
    for DeepSeek Engram M=4 Context-Aware Gating + Dilated Causal ShortConv + Dual Residual.
    """

    def __init__(
        self,
        d_mem: int,
        d_backbone: int,
        hc_mult: int = 4,
        dilation: int = 3,
        gate_eps: float = 1e-5,
        conv_eps: float = 1e-5,
        signed_sqrt_gate: bool = True,
        zero_init_conv: bool = False,
        ungated_constant_one: bool = False,
        block_c: int = 256,
        block_mc: int = 256,
        num_splits: int = 64,
        variant: str = "trial4_full_fused",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert hc_mult == 4, "DeepSeek mHC + Engram uses hc_mult = 4"
        dev = device or torch.device("cuda")
        self.d_mem = d_mem
        self.d_backbone = d_backbone
        self.hc_mult = hc_mult
        self.dilation = dilation
        self.gate_eps = gate_eps
        self.conv_eps = conv_eps
        self.signed_sqrt_gate = signed_sqrt_gate
        self.ungated_constant_one = ungated_constant_one
        self.block_c = block_c
        self.block_mc = block_mc
        self.num_splits = num_splits
        self.variant = variant

        self.W_kv = nn.Parameter(
            torch.empty(((hc_mult + 1) * d_backbone, d_mem), device=dev, dtype=dtype)
        )
        nn.init.normal_(self.W_kv, std=0.02)
        self.gamma_h = nn.Parameter(torch.ones((hc_mult, d_backbone), device=dev, dtype=dtype))
        self.gamma_k = nn.Parameter(torch.ones((hc_mult, d_backbone), device=dev, dtype=dtype))
        self.gamma_conv = nn.Parameter(torch.ones((hc_mult, d_backbone), device=dev, dtype=dtype))
        self.w_conv = nn.Parameter(torch.empty((hc_mult, d_backbone, 4), device=dev, dtype=dtype))
        if zero_init_conv:
            nn.init.zeros_(self.w_conv)
        else:
            nn.init.normal_(self.w_conv, std=0.02)

    def load_from_packed_params_(self, packed: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.W_kv.copy_(packed["W_kv"].to(self.W_kv.dtype))
            self.gamma_h.copy_(packed["gamma_h"].to(self.gamma_h.dtype))
            self.gamma_k.copy_(packed["gamma_k"].to(self.gamma_k.dtype))
            self.gamma_conv.copy_(packed["gamma_conv"].to(self.gamma_conv.dtype))
            self.w_conv.copy_(packed["w_conv"].to(self.w_conv.dtype))

    def scratch_bytes_estimate(self, batch_size: int, seq_len: int) -> int:
        n_tok = batch_size * seq_len
        mc = self.hc_mult * self.d_backbone
        splits = min(self.num_splits, max(1, n_tok))
        # dw_partial [splits, MC, 4] fp32 + dgamma_partial [splits, 12, C] fp32
        return splits * (mc * 4 + 12 * self.d_backbone) * 4

    def forward(
        self,
        H: torch.Tensor,
        E: torch.Tensor,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.ungated_constant_one:
            # Architectural shortcut ablation (`arXiv:2601.07372` Sec. 2.3 & Sec. 5):
            # Omit context-aware query-key RMS gating (`alpha_t == 1`), projecting only `V = E @ W_V.T`
            # and blindly injecting static N-gram memory regardless of conflicting context!
            B, T, M, C = H.shape
            V = F.linear(E, self.W_kv[:C])
            U = V.unsqueeze(2).expand(B, T, M, C)
            ru = torch.rsqrt(U.float().pow(2).mean(dim=-1, keepdim=True) + self.conv_eps)
            A = (U.float() * ru * self.gamma_conv.float().view(1, 1, M, C)).to(H.dtype)
            x_bct = A.reshape(B, T, M * C).transpose(1, 2)
            y_bct = F.silu(
                F.conv1d(
                    x_bct,
                    self.w_conv.reshape(M * C, 1, 4),
                    padding=3 * self.dilation,
                    dilation=self.dilation,
                    groups=M * C,
                )[..., :T]
            )
            return H + U + y_bct.transpose(1, 2).view(B, T, M, C)

        if self.variant == "trial2_merged_kv_eager_conv":
            # RSI Trial 2: Merged 5->1 KV GEMM, vectorized M=4 gate, eager transpose+Conv1d
            B, T, M, C = H.shape
            KV = F.linear(E, self.W_kv).view(B, T, 5, C)
            V = KV[:, :, 0, :]
            K = KV[:, :, 1:, :]
            rh = torch.rsqrt(H.float().pow(2).mean(dim=-1, keepdim=True) + self.gate_eps)
            rk = torch.rsqrt(K.float().pow(2).mean(dim=-1, keepdim=True) + self.gate_eps)
            nh = H.float() * rh * self.gamma_h.float().view(1, 1, M, C)
            nk = K.float() * rk * self.gamma_k.float().view(1, 1, M, C)
            s = (nh * nk).sum(dim=-1) / math.sqrt(float(C))
            if self.signed_sqrt_gate:
                alpha = (s.abs().clamp_min(1e-6).sqrt() * s.sign()).sigmoid()
            else:
                alpha = s.sigmoid()
            U = (alpha.unsqueeze(-1) * V.float().unsqueeze(2)).to(H.dtype)
            ru = torch.rsqrt(U.float().pow(2).mean(dim=-1, keepdim=True) + self.conv_eps)
            A = (U.float() * ru * self.gamma_conv.float().view(1, 1, M, C)).to(H.dtype)
            x_bct = A.reshape(B, T, M * C).transpose(1, 2)
            y_bct = F.silu(
                F.conv1d(
                    x_bct,
                    self.w_conv.reshape(M * C, 1, 4),
                    padding=3 * self.dilation,
                    dilation=self.dilation,
                    groups=M * C,
                )[..., :T]
            )
            return H + U + y_bct.transpose(1, 2).view(B, T, M, C)

        if self.variant == "trial3_conservative_tiles":
            return _FusedEngramContextAutogradFn.apply(
                H,
                E,
                self.W_kv,
                self.gamma_h,
                self.gamma_k,
                self.gamma_conv,
                self.w_conv,
                doc_ids,
                self.dilation,
                self.gate_eps,
                self.conv_eps,
                self.signed_sqrt_gate,
                128,
                128,
                16,
            )

        return _FusedEngramContextAutogradFn.apply(
            H,
            E,
            self.W_kv,
            self.gamma_h,
            self.gamma_k,
            self.gamma_conv,
            self.w_conv,
            doc_ids,
            self.dilation,
            self.gate_eps,
            self.conv_eps,
            self.signed_sqrt_gate,
            self.block_c,
            self.block_mc,
            self.num_splits,
        )


# =============================================================================
# 4. Distributed 8-GPU NVSwitch Sharded Engram Engine (`Suite B`)
#    Modulo-Balanced (`owner = z % W`) + Zipfian `Dedup-AllToAllv` +
#    Allocator-Safe (`record_stream`) Layer-2 Async Stream Prefetch
# =============================================================================


class ShardedEngramLookupEngine(nn.Module):
    """
    8-GPU NVSwitch Sharded Multi-Head Engram Lookup Table (`owner = global_z % world_size`,
    `local_z = global_z // world_size`), supporting three distributed regimes:
      1. `naive_alltoall`: Exchanges every `[B_local * T * H_heads]` token-head query without
         local deduplication (baseline multi-GPU All-to-All).
      2. `dedup_alltoallv`: Locally deduplicates Zipfian N-gram keys via GPU radix sort before
         `all_to_all_single` in both forward and backward (`3.5x-6x` NVSwitch traffic cut!) +
         shard-local Pre-Forward `JIT-Lazy-Adam` catch-up + atomic-free shard CSR update.
      3. `dedup_alltoallv_prefetch`: Runs GPU N-gram hash + local deduplication + `Dedup-AllToAllv`
         + Pre-Forward `JIT-Lazy-Adam` catch-up on a dedicated `prefetch_stream` overlapped
         behind mHC Transformer Layers 0..1 (and overlaps backward `Dedup-AllToAllv` + shard
         CSR segment-reduce behind Layers 1..0), with strict `wait_stream` + `record_stream`
         allocator safety!
    """

    def __init__(
        self,
        spec: EngramHashSpec,
        d_head: int = 80,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        max_steps: int = 4096,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.pg = process_group
        self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        dev = device or torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

        self.spec = spec
        self.d_head = d_head
        self.num_heads = spec.num_heads
        self.d_mem = self.num_heads * d_head
        self.betas = betas
        self.eps = eps

        # Modulo sharding: global row `z` belongs to `owner = z % W` at `local_z = z // W`
        self.shard_rows = (spec.total_entries + self.world_size - 1) // self.world_size
        init_w = torch.randn((self.shard_rows, d_head), device=dev, dtype=torch.float32) * 0.02
        self.weight_shard = nn.Parameter(init_w.to(dtype), requires_grad=False)
        self.register_buffer("master_shard", self.weight_shard.detach().float().clone())
        self.register_buffer("exp_avg_shard", torch.zeros((self.shard_rows, d_head), device=dev, dtype=torch.float32))
        self.register_buffer("exp_avg_sq_shard", torch.zeros((self.shard_rows, d_head), device=dev, dtype=torch.float32))
        self.register_buffer("last_step_shard", torch.zeros((self.shard_rows,), device=dev, dtype=torch.int32))
        self.register_buffer("lr_history", torch.zeros((max_steps + 2,), device=dev, dtype=torch.float32))

        self.register_buffer("lut_gpu", torch.from_numpy(spec.lookup_table_np.astype(np.int32)).to(dev))
        self.register_buffer("mult_gpu", torch.from_numpy(spec.multipliers_np).to(dev))
        self.register_buffer("mod_gpu", torch.from_numpy(spec.head_moduli_np).to(dev))
        self.register_buffer("off_gpu", torch.from_numpy(spec.head_offsets_np).to(dev))

        self.completed_step: int = 0
        self.last_comm_stats: Dict[str, float] = {}
        self._saved_exchange: Optional[Dict[str, object]] = None
        self._pending_shard_grad: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def load_from_full_table_(self, full_weight: torch.Tensor) -> None:
        """Deterministic modulo-shard slice from a full `[total_entries, d_head]` table (`Gate G9` parity)."""
        with torch.no_grad():
            self.weight_shard.zero_()
            owned = full_weight[self.rank :: self.world_size]
            self.weight_shard[: owned.shape[0]].copy_(owned.to(self.weight_shard.dtype))
            self.master_shard.copy_(self.weight_shard.float())
            self.exp_avg_shard.zero_()
            self.exp_avg_sq_shard.zero_()
            self.last_step_shard.zero_()
            self.lr_history.zero_()
            self.completed_step = 0

    def _catchup_local_shard_(self, local_rows: torch.Tensor) -> None:
        if self.completed_step <= 0 or local_rows.numel() == 0:
            return
        uniq_local = torch.unique(local_rows)
        U = uniq_local.numel()
        if U > 0:
            block_d = triton.next_power_of_2(self.d_head)
            _jit_lazy_adam_catchup_kernel[(U,)](
                self.weight_shard,
                self.master_shard,
                self.exp_avg_shard,
                self.exp_avg_sq_shard,
                self.last_step_shard,
                uniq_local,
                self.lr_history,
                U=U,
                target_step=self.completed_step,
                beta1=self.betas[0],
                beta2=self.betas[1],
                eps=self.eps,
                D_HEAD=self.d_head,
                BLOCK_D=block_d,
                num_warps=2 if block_d <= 128 else 4,
            )

    def prefetch_forward_exchange(
        self,
        input_ids: torch.Tensor,
        dedup: bool = True,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """
        Executes GPU N-gram hash + (optional) local Zipfian deduplication + `AllToAllv`
        key exchange + shard-local Pre-Forward `JIT-Lazy-Adam` catch-up + `AllToAllv`
        embedding return on the current CUDA stream (`compute_stream` or `prefetch_stream`).
        """
        dev = input_ids.device
        W = self.world_size
        B, T = input_ids.shape
        flat_z = gpu_engram_ngram_hashes(
            input_ids=input_ids,
            lut_gpu=self.lut_gpu,
            mult_gpu=self.mult_gpu,
            mod_gpu=self.mod_gpu,
            off_gpu=self.off_gpu,
            compressed_pad_id=self.spec.compressed_pad_id,
            max_ngram_size=self.spec.max_ngram_size,
            n_head_per_ngram=self.spec.n_head_per_ngram,
            doc_ids=doc_ids,
        )
        N_Q = flat_z.numel()

        if dedup:
            # Local radix-sort deduplication cuts NVSwitch payload 3.5x-6x on Zipfian text!
            uniq_z, inv = torch.unique(flat_z, sorted=True, return_inverse=True)
            keys_to_route = uniq_z
        else:
            uniq_z = None
            inv = None
            keys_to_route = flat_z

        owners = torch.remainder(keys_to_route, W)
        route_perm = torch.argsort(owners)
        sorted_owners = owners.index_select(0, route_perm)
        send_keys = keys_to_route.index_select(0, route_perm).contiguous()

        send_counts_t = torch.bincount(sorted_owners.to(torch.int64), minlength=W)
        recv_counts_t = torch.empty_like(send_counts_t)
        dist.all_to_all_single(recv_counts_t, send_counts_t, group=self.pg)

        # Single 64-byte host transfer of split sizes required by PyTorch all_to_all_single
        counts_cpu = torch.stack([send_counts_t, recv_counts_t], dim=0).cpu().tolist()
        send_splits: List[int] = counts_cpu[0]
        recv_splits: List[int] = counts_cpu[1]
        total_recv = int(sum(recv_splits))

        recv_keys = torch.empty((total_recv,), device=dev, dtype=torch.int32)
        dist.all_to_all_single(recv_keys, send_keys, recv_splits, send_splits, group=self.pg)

        local_rows = torch.div(recv_keys, W, rounding_mode="floor")
        self._catchup_local_shard_(local_rows)

        # Gather owned shard embeddings
        recv_embs = self.weight_shard.index_select(0, local_rows.to(torch.int64)).contiguous()
        returned_embs = torch.empty((send_keys.numel(), self.d_head), device=dev, dtype=self.weight_shard.dtype)
        dist.all_to_all_single(returned_embs, recv_embs, send_splits, recv_splits, group=self.pg)

        # Unpermute route_perm -> aligns with `keys_to_route`
        unrouted = torch.empty_like(returned_embs)
        unrouted.index_copy_(0, route_perm, returned_embs)

        if dedup:
            embs_3d = unrouted.index_select(0, inv).view(B, T, self.d_mem).contiguous()
        else:
            embs_3d = unrouted.view(B, T, self.d_mem).contiguous()

        elem_bytes = self.weight_shard.element_size()
        fwd_comm_bytes = float(send_keys.numel() * (4 + self.d_head * elem_bytes) + total_recv * (4 + self.d_head * elem_bytes))
        naive_comm_bytes = float(2 * N_Q * (4 + self.d_head * elem_bytes))
        self.last_comm_stats = {
            "n_queries_per_rank": float(N_Q),
            "routed_keys_per_rank": float(send_keys.numel()),
            "dedup_compression_ratio": float(N_Q) / max(float(send_keys.numel()), 1.0),
            "fwd_comm_bytes_per_rank": fwd_comm_bytes,
            "naive_comm_bytes_per_rank": naive_comm_bytes,
        }

        return {
            "embs_3d": embs_3d,
            "dedup": dedup,
            "B": B,
            "T": T,
            "N_Q": N_Q,
            "flat_z": flat_z,
            "uniq_z": uniq_z,
            "inv": inv,
            "route_perm": route_perm,
            "send_splits": send_splits,
            "recv_splits": recv_splits,
            "local_rows": local_rows,
        }

    def attach_autograd(self, state: Dict[str, object]) -> torch.Tensor:
        """
        Wraps prefetched `embs_3d` in an autograd node so Layer-2 Engram backward
        produces `grad_embs_3d` and stores it for sync or overlapped backward exchange.
        """
        self._saved_exchange = state
        embs = state["embs_3d"]
        assert isinstance(embs, torch.Tensor)
        out = embs.detach().requires_grad_(True)

        def _capture_grad_hook(grad: torch.Tensor) -> None:
            self._saved_exchange["grad_embs_3d"] = grad.contiguous()

        out.register_hook(_capture_grad_hook)
        return out

    def backward_exchange_and_reduce(
        self,
        comm_stream: Optional[torch.cuda.Stream] = None,
        compute_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Exchanges `grad_embs` back to shard owners (deduplicated first when `dedup=True`)
        and performs atomic-free CSR segment reduction on `local_rows`.
        """
        assert self._saved_exchange is not None and "grad_embs_3d" in self._saved_exchange
        st = self._saved_exchange
        grad_flat = st["grad_embs_3d"].view(-1, self.d_head)
        dedup: bool = bool(st["dedup"])
        route_perm: torch.Tensor = st["route_perm"]
        send_splits: List[int] = st["send_splits"]
        recv_splits: List[int] = st["recv_splits"]
        local_rows: torch.Tensor = st["local_rows"]

        active_stream = comm_stream or torch.cuda.current_stream(grad_flat.device)
        if comm_stream is not None and compute_stream is not None:
            comm_stream.wait_stream(compute_stream)
            grad_flat.record_stream(comm_stream)
            route_perm.record_stream(comm_stream)
            local_rows.record_stream(comm_stream)

        with torch.cuda.stream(active_stream):
            if dedup:
                inv: torch.Tensor = st["inv"]
                uniq_z: torch.Tensor = st["uniq_z"]
                if comm_stream is not None:
                    inv.record_stream(comm_stream)
                    uniq_z.record_stream(comm_stream)
                # Atomic-free local CSR reduction onto `uniq_z` BEFORE NVSwitch exchange!
                u_idx, perm, seg_off = deduplicate_csr_indices(inv.to(torch.int32))
                U_loc = u_idx.numel()
                local_uniq_grads = torch.empty(( uniq_z.numel(), self.d_head), device=grad_flat.device, dtype=torch.float32)
                if uniq_z.numel() != U_loc:
                    local_uniq_grads.zero_()
                if U_loc > 0:
                    block_d = triton.next_power_of_2(self.d_head)
                    reduced = torch.empty((U_loc, self.d_head), device=grad_flat.device, dtype=torch.float32)
                    _engram_csr_segment_reduce_bwd_kernel[(U_loc,)](
                        grad_flat,
                        perm,
                        seg_off,
                        reduced,
                        U=U_loc,
                        D_HEAD=self.d_head,
                        BLOCK_D=block_d,
                        num_warps=2 if block_d <= 128 else 4,
                    )
                    if U_loc == uniq_z.numel():
                        local_uniq_grads = reduced
                    else:
                        local_uniq_grads.index_copy_(0, u_idx.to(torch.int64), reduced)
                grads_to_send = local_uniq_grads.index_select(0, route_perm).contiguous()
                comm_grad_dtype = torch.float32
            else:
                grads_to_send = grad_flat.index_select(0, route_perm).to(self.weight_shard.dtype).contiguous()
                comm_grad_dtype = self.weight_shard.dtype

            total_recv = int(sum(recv_splits))
            recv_grads = torch.empty((total_recv, self.d_head), device=grad_flat.device, dtype=comm_grad_dtype)
            dist.all_to_all_single(recv_grads, grads_to_send, recv_splits, send_splits, group=self.pg)

            # Atomic-free CSR segment reduction across any cross-rank collisions on `local_rows`!
            if total_recv > 0:
                uniq_shard_rows, shard_perm, shard_seg = deduplicate_csr_indices(local_rows)
                U_sh = uniq_shard_rows.numel()
                shard_grads = torch.empty((U_sh, self.d_head), device=grad_flat.device, dtype=torch.float32)
                block_d = triton.next_power_of_2(self.d_head)
                _engram_csr_segment_reduce_bwd_kernel[(U_sh,)](
                    recv_grads,
                    shard_perm,
                    shard_seg,
                    shard_grads,
                    U=U_sh,
                    D_HEAD=self.d_head,
                    BLOCK_D=block_d,
                    num_warps=2 if block_d <= 128 else 4,
                )
            else:
                uniq_shard_rows = torch.empty((0,), device=grad_flat.device, dtype=torch.int32)
                shard_grads = torch.empty((0, self.d_head), device=grad_flat.device, dtype=torch.float32)

            self._pending_shard_grad = (uniq_shard_rows, shard_grads)

        self._saved_exchange = None

    def step_shard_lazy_adam(self, lr: float) -> int:
        assert self._pending_shard_grad is not None, "Call backward_exchange_and_reduce() before step"
        uniq_rows, uniq_grads = self._pending_shard_grad
        self._pending_shard_grad = None

        next_step = self.completed_step + 1
        self.lr_history[next_step] = float(lr)
        U = uniq_rows.numel()
        if U > 0:
            block_d = triton.next_power_of_2(self.d_head)
            bc1 = float(1.0 - (self.betas[0] ** next_step))
            bc2 = float(1.0 - (self.betas[1] ** next_step))
            _jit_lazy_adam_step_kernel[(U,)](
                self.weight_shard,
                self.master_shard,
                self.exp_avg_shard,
                self.exp_avg_sq_shard,
                self.last_step_shard,
                uniq_rows,
                uniq_grads,
                U=U,
                cur_step=next_step,
                lr_cur=float(lr),
                beta1=self.betas[0],
                beta2=self.betas[1],
                bc1=bc1,
                bc2=bc2,
                eps=self.eps,
                D_HEAD=self.d_head,
                BLOCK_D=block_d,
                num_warps=2 if block_d <= 128 else 4,
            )
        self.completed_step = next_step
        return U


# =============================================================================
# 5. Cubin Register-Spill Introspection Helper (`Gate G6`: Zero Spills)
# =============================================================================


def audit_engram_triton_cubin_spills() -> Dict[str, Dict[str, int]]:
    """
    Introspects every compiled Triton kernel in `engram_rsi_kernels` after warmup
    and extracts `(n_regs, n_spills)` from Triton's compiled kernel cache.
    """
    kernels = {
        "engram_ngram_hash": _engram_ngram_hash_kernel,
        "engram_gather_fwd": _engram_gather_fwd_kernel,
        "jit_lazy_adam_catchup": _jit_lazy_adam_catchup_kernel,
        "jit_lazy_adam_sorted_catchup": _jit_lazy_adam_sorted_catchup_kernel,
        "sorted_tile_segmented_bwd_and_adam": _sorted_tile_segmented_bwd_and_adam_kernel,
        "boundary_carry_finalize_adam": _boundary_carry_finalize_adam_kernel,
        "engram_gate_prenorm_fwd": _engram_gate_prenorm_fwd_kernel,
        "engram_dilated_conv_silu_resid_fwd": _engram_dilated_conv_silu_resid_fwd_kernel,
        "engram_conv_silu_q_dw_bwd": _engram_conv_silu_q_dw_bwd_kernel,
        "engram_token_grad_bwd": _engram_token_grad_bwd_kernel,
        "engram_dgamma_split_bwd": _engram_dgamma_split_bwd_kernel,
    }
    report: Dict[str, Dict[str, int]] = {}
    for name, kfn in kernels.items():
        max_regs = 0
        max_spills = 0
        n_compiled = 0
        dev_caches = getattr(kfn, "device_caches", None)
        if isinstance(dev_caches, dict) and dev_caches:
            for dev_entry in dev_caches.values():
                cdict = dev_entry[0] if isinstance(dev_entry, tuple) and len(dev_entry) > 0 else dev_entry
                if isinstance(cdict, dict):
                    for compiled in cdict.values():
                        if hasattr(compiled, "n_regs"):
                            n_compiled += 1
                            max_regs = max(max_regs, int(getattr(compiled, "n_regs", 0) or 0))
                            max_spills = max(max_spills, int(getattr(compiled, "n_spills", 0) or 0))
        else:
            cache = getattr(kfn, "cache", {})
            if isinstance(cache, dict):
                for dev_dict in cache.values():
                    entries = dev_dict.values() if isinstance(dev_dict, dict) else [dev_dict]
                    for compiled in entries:
                        if hasattr(compiled, "n_regs"):
                            n_compiled += 1
                            max_regs = max(max_regs, int(getattr(compiled, "n_regs", 0) or 0))
                            max_spills = max(max_spills, int(getattr(compiled, "n_spills", 0) or 0))
        report[name] = {
            "compiled_specializations": n_compiled,
            "max_n_regs": max_regs,
            "max_n_spills": max_spills,
        }
    return report

