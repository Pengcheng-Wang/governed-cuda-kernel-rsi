#!/usr/bin/env python3
"""E127: Diffusion-LLM, Block-Diffusion (`BD3-LMs`), & `DiT` `adaLN-Zero` Training Kernels
Driven by the Immutable Recursive Self-Improvement (`RSI`) Verification & Vector-Pareto Ratchet (`PROGRAM.md`).

Primary Literature Anchors:
  - LLaDA & LLaDA 1.5: Nie et al., Feb 2025 (arXiv:2502.09992); Zhu et al., May 2025 (arXiv:2505.19223)
    Masked diffusion language modeling continuous-time 1/t NELBO objective and variance behavior.
  - MDLM: Sahoo et al., NeurIPS 2024 (arXiv:2406.07524)
    Continuous-time Rao-Blackwellized masked diffusion NELBO and low-discrepancy stratified t-sampling.
  - BD3-LMs (Block Discrete Denoising Diffusion LMs): Arriola et al., ICLR 2025 Oral (arXiv:2503.09573)
    Dual-stream [x_t || x_0] 2L x 2L block-diffusion mask M_full = [[M_BD, M_OBC], [0, M_BC]]
    (Eq. 9, Appendix B.6, Figure 4) and clipped noise schedules reducing gradient variance.
  - DiT (Scalable Diffusion Models with Transformers): Peebles & Xie, ICCV 2023 (arXiv:2212.09748)
    Adaptive LayerNorm-Zero (`adaLN-Zero`) shift/scale/gate conditioning modulation.
  - RSI & Autonomous Kernel Optimization Infrastructure:
    NaiveAI Naive-N0.5-Flash / NaiveRT (2026), Karpathy autoresearch (`program.md`), AutoKernel (arXiv:2603.21331).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_local_rank_env = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_e127_r{_local_rank_env}")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_e127_r{_local_rank_env}")

import torch
import torch._dynamo
import torch.distributed as dist

for _dyn_attr, _dyn_val in (
    ("recompile_limit", 64),
    ("cache_size_limit", 64),
    ("accumulated_recompile_limit", 256),
    ("accumulated_cache_size_limit", 256),
):
    if hasattr(torch._dynamo.config, _dyn_attr):
        setattr(torch._dynamo.config, _dyn_attr, _dyn_val)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import triton
import triton.language as tl
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)

from common import environment_metadata
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


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    return float(torch.linalg.norm(af - bf).item() / max(1e-12, torch.linalg.norm(bf).item()))


# ============================================================================
# 1. Track 1 Kernels: Block-Diffusion `BD3-LM` Dual-Stream `[x_t || x_0]` Attention
#    - Trial 0a: Verbatim `BD3-LM` Fig. 4 `(x0_flag_q == x0_flag_kv)` -> PyTorch 2.13 SymPy crash
#    - Trial 0b: Bitwise-repaired `FlexAttention` `2L x 2L` (`BASELINE`)
#    - Trial 1 : 6-Call Varlen Shift-Isomorphic Surgical FA2 (`REJECT_SHAPE_REGRESSION` at W=4, 16)
#    - Trial 2 : Monolithic GQA-Folded Triton Microblock + 2-Call FA2 (`REJECT_SHAPE_REGRESSION` at W=64)
#    - Trial 3 : Adaptive Zero-Spill Split-Stream GQA Microblock (W<=32) + Surgical FA2 (W>=64) (`KEEP`)
#    - Trial 4 : Exact Last-Layer Clean-Branch Pruning (`KEEP_TERMINAL_STAGE`)
# ============================================================================


@triton.jit
def _rev_gather_qkv_masked_kernel(
    Q_ptr, K_ptr, V_ptr, SpanMeta_ptr,
    QRev_ptr, KRev_ptr, VRev_ptr,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
):
    u = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    q_src = tl.load(SpanMeta_ptr + u * 2 + 0).to(tl.int64)
    kv_src = tl.load(SpanMeta_ptr + u * 2 + 1).to(tl.int64)
    offs_d = tl.arange(0, D)
    if h < H_Q:
        valid_q = q_src >= 0
        q_safe = tl.maximum(q_src, 0)
        val_q = tl.load(Q_ptr + (q_safe * H_Q + h) * D + offs_d, mask=valid_q, other=0.0)
        tl.store(QRev_ptr + (u * H_Q + h) * D + offs_d, val_q)
    elif h < H_Q + H_KV:
        hk = h - H_Q
        tl.store(KRev_ptr + (u * H_KV + hk) * D + offs_d, tl.load(K_ptr + (kv_src * H_KV + hk) * D + offs_d))
    else:
        hk = h - H_Q - H_KV
        tl.store(VRev_ptr + (u * H_KV + hk) * D + offs_d, tl.load(V_ptr + (kv_src * H_KV + hk) * D + offs_d))


@triton.jit
def _merge_lse_indexed_masked_kernel(
    Out1_ptr, LSE1_ptr, Out2_ptr, LSE2_ptr, QIdx_ptr,
    S1, N2, STRIDE_QIDX: tl.constexpr, H_Q: tl.constexpr, D: tl.constexpr,
):
    u = tl.program_id(0).to(tl.int64)
    hq = tl.program_id(1).to(tl.int64)
    q_idx = tl.load(QIdx_ptr + u * STRIDE_QIDX).to(tl.int64)
    if q_idx < 0:
        return
    offs_d = tl.arange(0, D)
    l1 = tl.load(LSE1_ptr + hq * S1 + q_idx).to(tl.float32)
    l2 = tl.load(LSE2_ptr + hq * N2 + u).to(tl.float32)
    m = tl.maximum(l1, l2)
    w1 = tl.exp(l1 - m)
    w2 = tl.exp(l2 - m)
    ws = tl.maximum(w1 + w2, 1e-30)
    nlse = m + tl.log(ws)
    o1 = tl.load(Out1_ptr + (q_idx * H_Q + hq) * D + offs_d).to(tl.float32)
    o2 = tl.load(Out2_ptr + (u * H_Q + hq) * D + offs_d).to(tl.float32)
    nout = (o1 * w1 + o2 * w2) / ws
    tl.store(Out1_ptr + (q_idx * H_Q + hq) * D + offs_d, nout.to(Out1_ptr.dtype.element_ty))
    tl.store(LSE1_ptr + hq * S1 + q_idx, nlse)


@triton.jit
def _gather_bwd_indexed_masked_kernel(
    dO_ptr, MixOut_ptr, MixLSE_ptr, QIdx_ptr,
    dOSub_ptr, MixOutSub_ptr, MixLSESub_ptr,
    S1, N2, STRIDE_QIDX: tl.constexpr, H_Q: tl.constexpr, D: tl.constexpr,
):
    u = tl.program_id(0).to(tl.int64)
    hq = tl.program_id(1).to(tl.int64)
    q_src = tl.load(QIdx_ptr + u * STRIDE_QIDX).to(tl.int64)
    offs_d = tl.arange(0, D)
    valid_q = q_src >= 0
    q_safe = tl.maximum(q_src, 0)
    do_v = tl.load(dO_ptr + (q_safe * H_Q + hq) * D + offs_d, mask=valid_q, other=0.0)
    mo_v = tl.load(MixOut_ptr + (q_safe * H_Q + hq) * D + offs_d, mask=valid_q, other=0.0)
    ml_v = tl.load(MixLSE_ptr + hq * S1 + q_safe, mask=valid_q, other=0.0)
    tl.store(dOSub_ptr + (u * H_Q + hq) * D + offs_d, do_v)
    tl.store(MixOutSub_ptr + (u * H_Q + hq) * D + offs_d, mo_v)
    tl.store(MixLSESub_ptr + hq * N2 + u, ml_v)


@triton.jit
def _scatter_add_rev_grads_masked_kernel(
    dQ1_ptr, dK1_ptr, dV1_ptr,
    dQ2_ptr, dK2_ptr, dV2_ptr,
    SpanMeta_ptr,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
):
    u = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    q_dst = tl.load(SpanMeta_ptr + u * 2 + 0).to(tl.int64)
    kv_dst = tl.load(SpanMeta_ptr + u * 2 + 1).to(tl.int64)
    offs_d = tl.arange(0, D)
    if h < H_Q:
        if q_dst >= 0:
            v1 = tl.load(dQ1_ptr + (q_dst * H_Q + h) * D + offs_d).to(tl.float32)
            v2 = tl.load(dQ2_ptr + (u * H_Q + h) * D + offs_d).to(tl.float32)
            tl.store(dQ1_ptr + (q_dst * H_Q + h) * D + offs_d, (v1 + v2).to(dQ1_ptr.dtype.element_ty))
    elif h < H_Q + H_KV:
        hk = h - H_Q
        v1 = tl.load(dK1_ptr + (kv_dst * H_KV + hk) * D + offs_d).to(tl.float32)
        v2 = tl.load(dK2_ptr + (u * H_KV + hk) * D + offs_d).to(tl.float32)
        tl.store(dK1_ptr + (kv_dst * H_KV + hk) * D + offs_d, (v1 + v2).to(dK1_ptr.dtype.element_ty))
    else:
        hk = h - H_Q - H_KV
        v1 = tl.load(dV1_ptr + (kv_dst * H_KV + hk) * D + offs_d).to(tl.float32)
        v2 = tl.load(dV2_ptr + (u * H_KV + hk) * D + offs_d).to(tl.float32)
        tl.store(dV1_ptr + (kv_dst * H_KV + hk) * D + offs_d, (v1 + v2).to(dV1_ptr.dtype.element_ty))


@triton.jit
def _bd3_gqa_microblock_fwd_kernel(
    Qt_ptr, Kt_ptr, Vt_ptr,
    Q0_ptr, K0_ptr, V0_ptr,
    O_obc_ptr, LSE_obc_ptr,
    O_bc_ptr, LSE_bc_ptr,
    Ot_out_ptr, LSEt_out_ptr,
    BlkMeta_ptr,
    N_TOTAL, N_OBC, sm_scale,
    W: tl.constexpr, BLOCK_N: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    COMPUTE_CLEAN: tl.constexpr,
):
    b_idx = tl.program_id(0).to(tl.int64)
    q_tile_idx = tl.program_id(1)
    h_kv = tl.program_id(2).to(tl.int64)

    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0).to(tl.int64)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    prev_start = tl.load(BlkMeta_ptr + b_idx * 6 + 2).to(tl.int64)
    obc_off = tl.load(BlkMeta_ptr + b_idx * 6 + 4).to(tl.int64)

    q_base = q_tile_idx * Q_STEP
    if q_base >= blk_len:
        return

    BLOCK_M: tl.constexpr = Q_STEP * G_PAD
    offs_m = tl.arange(0, BLOCK_M)
    q_sub = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    q_pos = q_base + q_sub
    valid_m = (q_pos < blk_len) & (g_idx < G)
    hq_vec = h_kv * G + g_idx
    q_tok = blk_start + q_pos

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    qt_ptrs = Qt_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :]
    qt = tl.load(qt_ptrs, mask=valid_m[:, None], other=0.0)

    valid_kt = offs_n < blk_len
    kt_tok = blk_start + offs_n
    kt_ptrs = Kt_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :]
    vt_ptrs = Vt_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :]
    kt = tl.load(kt_ptrs, mask=valid_kt[:, None], other=0.0)
    vt = tl.load(vt_ptrs, mask=valid_kt[:, None], other=0.0)

    st = tl.dot(qt, tl.trans(kt)) * sm_scale
    st = tl.where(valid_m[:, None] & valid_kt[None, :], st, -float("inf"))
    m_acc = tl.max(st, axis=1)
    m_safe = tl.where(m_acc == -float("inf"), 0.0, m_acc)
    p_t = tl.exp(st - m_safe[:, None])
    l_acc = tl.sum(p_t, axis=1)
    o_acc = tl.dot(p_t.to(vt.dtype), vt)

    if prev_start >= 0:
        valid_kp = offs_n < W
        kp_tok = prev_start + offs_n
        kp_ptrs = K0_ptr + (kp_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :]
        vp_ptrs = V0_ptr + (kp_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :]
        kp = tl.load(kp_ptrs, mask=valid_kp[:, None], other=0.0)
        vp = tl.load(vp_ptrs, mask=valid_kp[:, None], other=0.0)

        sp = tl.dot(qt, tl.trans(kp)) * sm_scale
        upper_mask = valid_m[:, None] & valid_kp[None, :] & (offs_n[None, :] > q_pos[:, None])
        has_upper = (q_pos < (W - 1)) & valid_m
        sp = tl.where(upper_mask, sp, -float("inf"))
        m_p = tl.max(sp, axis=1)
        m_new = tl.where(has_upper, tl.maximum(m_acc, m_p), m_acc)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_safe - m_new_safe)
        p_p = tl.where(upper_mask, tl.exp(sp - m_new_safe[:, None]), 0.0)
        l_acc = l_acc * alpha + tl.sum(p_p, axis=1)
        o_acc = o_acc * alpha[:, None] + tl.dot(p_p.to(vp.dtype), vp)
        m_safe = m_new_safe

        obc_u = obc_off + q_pos
        lse_c = tl.load(LSE_obc_ptr + hq_vec * N_OBC + obc_u, mask=valid_m, other=-float("inf")).to(tl.float32)
        o_c = tl.load(
            O_obc_ptr + (obc_u[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :],
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.float32)
        lse_micro = m_safe + tl.log(tl.maximum(l_acc, 1e-30))
        m_tot = tl.maximum(lse_micro, lse_c)
        w_micro = tl.exp(m_safe - m_tot)
        w_c = tl.exp(lse_c - m_tot)
        denom = l_acc * w_micro + w_c
        o_final = (o_acc * w_micro[:, None] + o_c * w_c[:, None]) / tl.maximum(denom[:, None], 1e-30)
        lse_final = m_tot + tl.log(tl.maximum(denom, 1e-30))
    else:
        o_final = o_acc / tl.maximum(l_acc[:, None], 1e-30)
        lse_final = m_safe + tl.log(tl.maximum(l_acc, 1e-30))

    tl.store(
        Ot_out_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :],
        o_final.to(Ot_out_ptr.dtype.element_ty),
        mask=valid_m[:, None],
    )
    tl.store(LSEt_out_ptr + hq_vec * N_TOTAL + q_tok, lse_final, mask=valid_m)

    if COMPUTE_CLEAN:
        has_ubc = valid_m & (q_pos < (blk_len - 1))
        q0_ptrs = Q0_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :]
        q0 = tl.load(q0_ptrs, mask=has_ubc[:, None], other=0.0)

        k0_ptrs = K0_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :]
        v0_ptrs = V0_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :]
        k0 = tl.load(k0_ptrs, mask=valid_kt[:, None], other=0.0)
        v0 = tl.load(v0_ptrs, mask=valid_kt[:, None], other=0.0)

        s0 = tl.dot(q0, tl.trans(k0)) * sm_scale
        ubc_mask = has_ubc[:, None] & valid_kt[None, :] & (offs_n[None, :] > q_pos[:, None])
        s0 = tl.where(ubc_mask, s0, -float("inf"))
        m0 = tl.max(s0, axis=1)
        m0_safe = tl.where(m0 == -float("inf"), 0.0, m0)
        p0 = tl.where(ubc_mask, tl.exp(s0 - m0_safe[:, None]), 0.0)
        l0_u = tl.sum(p0, axis=1)
        o0_u = tl.dot(p0.to(v0.dtype), v0)

        lse_bc = tl.load(LSE_bc_ptr + hq_vec * N_TOTAL + q_tok, mask=has_ubc, other=-float("inf")).to(tl.float32)
        o_bc = tl.load(
            O_bc_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :],
            mask=has_ubc[:, None],
            other=0.0,
        ).to(tl.float32)
        lse0_micro = m0_safe + tl.log(tl.maximum(l0_u, 1e-30))
        m0_tot = tl.maximum(lse_bc, lse0_micro)
        w0_micro = tl.exp(m0_safe - m0_tot)
        w0_bc = tl.exp(lse_bc - m0_tot)
        denom0 = l0_u * w0_micro + w0_bc
        o0_final = (o0_u * w0_micro[:, None] + o_bc * w0_bc[:, None]) / tl.maximum(denom0[:, None], 1e-30)
        lse0_final = m0_tot + tl.log(tl.maximum(denom0, 1e-30))

        tl.store(
            O_bc_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :],
            o0_final.to(O_bc_ptr.dtype.element_ty),
            mask=has_ubc[:, None],
        )
        tl.store(LSE_bc_ptr + hq_vec * N_TOTAL + q_tok, lse0_final, mask=has_ubc)


@triton.jit
def _bd3_gqa_monolithic_bwd_kernel(
    Qt_ptr, Kt_ptr, Vt_ptr, Ot_ptr, LSEt_ptr, dOt_ptr,
    Q0_ptr, K0_ptr, V0_ptr, O0_ptr, LSE0_ptr, dO0_ptr,
    dQt_ptr, dKt_ptr, dVt_ptr,
    dQ0_ptr, dK0_ptr, dV0_ptr,
    BlkMeta_ptr,
    N_TOTAL, sm_scale,
    W: tl.constexpr, BLOCK_N: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    COMPUTE_CLEAN: tl.constexpr,
):
    """Trial 2 monolithic backward kernel (holds dkt, dvt, dk0, dv0 simultaneously -> spills at W=64)."""
    b_idx = tl.program_id(0).to(tl.int64)
    h_kv = tl.program_id(1).to(tl.int64)

    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0).to(tl.int64)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    prev_start = tl.load(BlkMeta_ptr + b_idx * 6 + 2).to(tl.int64)
    next_start = tl.load(BlkMeta_ptr + b_idx * 6 + 3).to(tl.int64)
    next_len = tl.load(BlkMeta_ptr + b_idx * 6 + 5)

    BLOCK_M: tl.constexpr = Q_STEP * G_PAD
    offs_m = tl.arange(0, BLOCK_M)
    q_sub = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    hq_vec = h_kv * G + g_idx

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    valid_kt = offs_n < blk_len
    kt_tok = blk_start + offs_n
    kt = tl.load(Kt_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
    vt = tl.load(Vt_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kt[:, None], other=0.0)

    dkt_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dvt_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    has_prev = prev_start >= 0
    valid_kp = offs_n < W
    kp_tok = tl.maximum(prev_start, 0) + offs_n
    kp = tl.load(K0_ptr + (kp_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=(has_prev & valid_kp)[:, None], other=0.0)
    vp = tl.load(V0_ptr + (kp_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=(has_prev & valid_kp)[:, None], other=0.0)

    for q_base in range(0, W, Q_STEP):
        q_pos = q_base + q_sub
        valid_m = (q_pos < blk_len) & (g_idx < G)
        q_tok = blk_start + q_pos
        ptrs_m = (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :]

        qt = tl.load(Qt_ptr + ptrs_m, mask=valid_m[:, None], other=0.0)
        ot = tl.load(Ot_ptr + ptrs_m, mask=valid_m[:, None], other=0.0).to(tl.float32)
        dot = tl.load(dOt_ptr + ptrs_m, mask=valid_m[:, None], other=0.0)
        lset = tl.load(LSEt_ptr + hq_vec * N_TOTAL + q_tok, mask=valid_m, other=0.0).to(tl.float32)
        delta_t = tl.sum(dot.to(tl.float32) * ot, axis=1)

        st = tl.dot(qt, tl.trans(kt)) * sm_scale
        pt = tl.where(valid_m[:, None] & valid_kt[None, :], tl.exp(st - lset[:, None]), 0.0)
        dvt_acc += tl.dot(tl.trans(pt.to(dot.dtype)), dot)
        dp_t = tl.dot(dot, tl.trans(vt))
        ds_t = pt * (dp_t - delta_t[:, None]) * sm_scale
        dqt_tile = tl.dot(ds_t.to(kt.dtype), kt)
        dkt_acc += tl.dot(tl.trans(ds_t.to(qt.dtype)), qt)

        if has_prev:
            sp = tl.dot(qt, tl.trans(kp)) * sm_scale
            up_mask = valid_m[:, None] & valid_kp[None, :] & (offs_n[None, :] > q_pos[:, None])
            pp = tl.where(up_mask, tl.exp(sp - lset[:, None]), 0.0)
            dp_p = tl.dot(dot, tl.trans(vp))
            ds_p = pp * (dp_p - delta_t[:, None]) * sm_scale
            dqt_tile += tl.dot(ds_p.to(kp.dtype), kp)

        tl.store(dQt_ptr + ptrs_m, dqt_tile.to(dQt_ptr.dtype.element_ty), mask=valid_m[:, None])

    tl.store(dKt_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], dkt_acc.to(dKt_ptr.dtype.element_ty), mask=valid_kt[:, None])
    tl.store(dVt_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], dvt_acc.to(dVt_ptr.dtype.element_ty), mask=valid_kt[:, None])

    k0 = tl.load(K0_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
    v0 = tl.load(V0_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
    dk0_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv0_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    if COMPUTE_CLEAN:
        for q_base in range(0, W, Q_STEP):
            q_pos = q_base + q_sub
            has_ubc = (q_pos < (blk_len - 1)) & (g_idx < G)
            q_tok = blk_start + q_pos
            ptrs_m = (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :]

            q0 = tl.load(Q0_ptr + ptrs_m, mask=has_ubc[:, None], other=0.0)
            o0 = tl.load(O0_ptr + ptrs_m, mask=has_ubc[:, None], other=0.0).to(tl.float32)
            do0 = tl.load(dO0_ptr + ptrs_m, mask=has_ubc[:, None], other=0.0)
            lse0 = tl.load(LSE0_ptr + hq_vec * N_TOTAL + q_tok, mask=has_ubc, other=0.0).to(tl.float32)
            delta0 = tl.sum(do0.to(tl.float32) * o0, axis=1)

            s0 = tl.dot(q0, tl.trans(k0)) * sm_scale
            ubc_mask = has_ubc[:, None] & valid_kt[None, :] & (offs_n[None, :] > q_pos[:, None])
            p0 = tl.where(ubc_mask, tl.exp(s0 - lse0[:, None]), 0.0)
            dv0_acc += tl.dot(tl.trans(p0.to(do0.dtype)), do0)
            dp0 = tl.dot(do0, tl.trans(v0))
            ds0 = p0 * (dp0 - delta0[:, None]) * sm_scale
            dq0_add = tl.dot(ds0.to(k0.dtype), k0)
            dk0_acc += tl.dot(tl.trans(ds0.to(q0.dtype)), q0)

            dq0_prev = tl.load(dQ0_ptr + ptrs_m, mask=has_ubc[:, None], other=0.0).to(tl.float32)
            tl.store(dQ0_ptr + ptrs_m, (dq0_prev + dq0_add).to(dQ0_ptr.dtype.element_ty), mask=has_ubc[:, None])

    if next_start >= 0:
        for q_base in range(0, W, Q_STEP):
            q_pos = q_base + q_sub
            valid_qn = (q_pos < next_len) & (q_pos < (W - 1)) & (g_idx < G)
            qn_tok = next_start + q_pos
            ptrs_mn = (qn_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :]

            qtn = tl.load(Qt_ptr + ptrs_mn, mask=valid_qn[:, None], other=0.0)
            otn = tl.load(Ot_ptr + ptrs_mn, mask=valid_qn[:, None], other=0.0).to(tl.float32)
            dotn = tl.load(dOt_ptr + ptrs_mn, mask=valid_qn[:, None], other=0.0)
            lsetn = tl.load(LSEt_ptr + hq_vec * N_TOTAL + qn_tok, mask=valid_qn, other=0.0).to(tl.float32)
            deltatn = tl.sum(dotn.to(tl.float32) * otn, axis=1)

            sn = tl.dot(qtn, tl.trans(k0)) * sm_scale
            uobc_mask = valid_qn[:, None] & valid_kt[None, :] & (offs_n[None, :] > q_pos[:, None])
            pn = tl.where(uobc_mask, tl.exp(sn - lsetn[:, None]), 0.0)
            dv0_acc += tl.dot(tl.trans(pn.to(dotn.dtype)), dotn)
            dpn = tl.dot(dotn, tl.trans(v0))
            dsn = pn * (dpn - deltatn[:, None]) * sm_scale
            dk0_acc += tl.dot(tl.trans(dsn.to(qtn.dtype)), qtn)

    dk0_prev = tl.load(dK0_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kt[:, None], other=0.0).to(tl.float32)
    dv0_prev = tl.load(dV0_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kt[:, None], other=0.0).to(tl.float32)
    tl.store(dK0_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], (dk0_prev + dk0_acc).to(dK0_ptr.dtype.element_ty), mask=valid_kt[:, None])
    tl.store(dV0_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], (dv0_prev + dv0_acc).to(dV0_ptr.dtype.element_ty), mask=valid_kt[:, None])


@triton.jit
def _bd3_gqa_noisy_bwd_kernel(
    Qt_ptr, Kt_ptr, Vt_ptr, K0_ptr, V0_ptr,
    Ot_ptr, LSEt_ptr, dOt_ptr,
    dQt_ptr, dKt_ptr, dVt_ptr,
    BlkMeta_ptr,
    N_TOTAL, sm_scale,
    W: tl.constexpr, BLOCK_N: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
):
    """Trial 3 Split-Stream Pass A: owns only (dKt, dVt) in registers with loop_unroll_factor=1 (zero spills)."""
    b_idx = tl.program_id(0)
    h_kv = tl.program_id(1)

    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    prev_start = tl.load(BlkMeta_ptr + b_idx * 6 + 2)

    BLOCK_M: tl.constexpr = Q_STEP * G_PAD
    offs_m = tl.arange(0, BLOCK_M)
    q_sub = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    hq_vec = h_kv * G + g_idx

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    valid_kt = offs_n < blk_len
    kt_row = ((blk_start + offs_n) * H_KV + h_kv).to(tl.int64) * D
    kt = tl.load(Kt_ptr + kt_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
    vt = tl.load(Vt_ptr + kt_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0)

    dkt_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dvt_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    has_prev = prev_start >= 0
    valid_kp = offs_n < W
    kp_row = ((tl.maximum(prev_start, 0) + offs_n) * H_KV + h_kv).to(tl.int64) * D
    kp = tl.load(K0_ptr + kp_row[:, None] + offs_d[None, :], mask=(has_prev & valid_kp)[:, None], other=0.0)
    vp = tl.load(V0_ptr + kp_row[:, None] + offs_d[None, :], mask=(has_prev & valid_kp)[:, None], other=0.0)

    for q_base in tl.range(0, W, Q_STEP, loop_unroll_factor=1):
        q_pos = q_base + q_sub
        valid_m = (q_pos < blk_len) & (g_idx < G)
        q_tok = blk_start + q_pos
        q_row = (q_tok * H_Q + hq_vec).to(tl.int64) * D

        qt = tl.load(Qt_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0)
        ot = tl.load(Ot_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
        dot = tl.load(dOt_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0)
        lset = tl.load(LSEt_ptr + hq_vec.to(tl.int64) * N_TOTAL + q_tok.to(tl.int64), mask=valid_m, other=0.0).to(tl.float32)
        delta_t = tl.sum(dot.to(tl.float32) * ot, axis=1)

        st = tl.dot(qt, tl.trans(kt)) * sm_scale
        pt = tl.where(valid_m[:, None] & valid_kt[None, :], tl.exp(st - lset[:, None]), 0.0)
        dvt_acc += tl.dot(tl.trans(pt.to(dot.dtype)), dot)
        dp_t = tl.dot(dot, tl.trans(vt))
        ds_t = pt * (dp_t - delta_t[:, None]) * sm_scale
        dqt_tile = tl.dot(ds_t.to(kt.dtype), kt)
        dkt_acc += tl.dot(tl.trans(ds_t.to(qt.dtype)), qt)

        if has_prev:
            sp = tl.dot(qt, tl.trans(kp)) * sm_scale
            up_mask = valid_m[:, None] & valid_kp[None, :] & (offs_n[None, :] > q_pos[:, None])
            pp = tl.where(up_mask, tl.exp(sp - lset[:, None]), 0.0)
            dp_p = tl.dot(dot, tl.trans(vp))
            ds_p = pp * (dp_p - delta_t[:, None]) * sm_scale
            dqt_tile += tl.dot(ds_p.to(kp.dtype), kp)

        tl.store(dQt_ptr + q_row[:, None] + offs_d[None, :], dqt_tile.to(dQt_ptr.dtype.element_ty), mask=valid_m[:, None])

    tl.store(dKt_ptr + kt_row[:, None] + offs_d[None, :], dkt_acc.to(dKt_ptr.dtype.element_ty), mask=valid_kt[:, None])
    tl.store(dVt_ptr + kt_row[:, None] + offs_d[None, :], dvt_acc.to(dVt_ptr.dtype.element_ty), mask=valid_kt[:, None])


@triton.jit
def _bd3_gqa_clean_bwd_kernel(
    Qt_ptr, Ot_ptr, LSEt_ptr, dOt_ptr,
    Q0_ptr, K0_ptr, V0_ptr, O0_ptr, LSE0_ptr, dO0_ptr,
    dQ0_ptr, dK0_ptr, dV0_ptr,
    BlkMeta_ptr,
    N_TOTAL, sm_scale,
    W: tl.constexpr, BLOCK_N: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    COMPUTE_CLEAN: tl.constexpr,
):
    """Trial 3 Split-Stream Pass B: owns only (dK0, dV0) in registers with loop_unroll_factor=1 (zero spills)."""
    b_idx = tl.program_id(0)
    h_kv = tl.program_id(1)

    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    next_start = tl.load(BlkMeta_ptr + b_idx * 6 + 3)
    next_len = tl.load(BlkMeta_ptr + b_idx * 6 + 5)

    if (not COMPUTE_CLEAN) and (next_start < 0):
        return

    BLOCK_M: tl.constexpr = Q_STEP * G_PAD
    offs_m = tl.arange(0, BLOCK_M)
    q_sub = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    hq_vec = h_kv * G + g_idx

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    valid_kt = offs_n < blk_len
    k0_row = ((blk_start + offs_n) * H_KV + h_kv).to(tl.int64) * D
    k0 = tl.load(K0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
    v0 = tl.load(V0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0)

    dk0_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv0_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    if COMPUTE_CLEAN:
        for q_base in tl.range(0, W, Q_STEP, loop_unroll_factor=1):
            q_pos = q_base + q_sub
            has_ubc = (q_pos < (blk_len - 1)) & (g_idx < G)
            q_tok = blk_start + q_pos
            q_row = (q_tok * H_Q + hq_vec).to(tl.int64) * D

            q0 = tl.load(Q0_ptr + q_row[:, None] + offs_d[None, :], mask=has_ubc[:, None], other=0.0)
            o0 = tl.load(O0_ptr + q_row[:, None] + offs_d[None, :], mask=has_ubc[:, None], other=0.0).to(tl.float32)
            do0 = tl.load(dO0_ptr + q_row[:, None] + offs_d[None, :], mask=has_ubc[:, None], other=0.0)
            lse0 = tl.load(LSE0_ptr + hq_vec.to(tl.int64) * N_TOTAL + q_tok.to(tl.int64), mask=has_ubc, other=0.0).to(tl.float32)
            delta0 = tl.sum(do0.to(tl.float32) * o0, axis=1)

            s0 = tl.dot(q0, tl.trans(k0)) * sm_scale
            ubc_mask = has_ubc[:, None] & valid_kt[None, :] & (offs_n[None, :] > q_pos[:, None])
            p0 = tl.where(ubc_mask, tl.exp(s0 - lse0[:, None]), 0.0)
            dv0_acc += tl.dot(tl.trans(p0.to(do0.dtype)), do0)
            dp0 = tl.dot(do0, tl.trans(v0))
            ds0 = p0 * (dp0 - delta0[:, None]) * sm_scale
            dq0_add = tl.dot(ds0.to(k0.dtype), k0)
            dk0_acc += tl.dot(tl.trans(ds0.to(q0.dtype)), q0)

            dq0_prev = tl.load(dQ0_ptr + q_row[:, None] + offs_d[None, :], mask=has_ubc[:, None], other=0.0).to(tl.float32)
            tl.store(dQ0_ptr + q_row[:, None] + offs_d[None, :], (dq0_prev + dq0_add).to(dQ0_ptr.dtype.element_ty), mask=has_ubc[:, None])

    if next_start >= 0:
        for q_base in tl.range(0, W, Q_STEP, loop_unroll_factor=1):
            q_pos = q_base + q_sub
            valid_qn = (q_pos < next_len) & (q_pos < (W - 1)) & (g_idx < G)
            qn_tok = next_start + q_pos
            qn_row = (qn_tok * H_Q + hq_vec).to(tl.int64) * D

            qtn = tl.load(Qt_ptr + qn_row[:, None] + offs_d[None, :], mask=valid_qn[:, None], other=0.0)
            otn = tl.load(Ot_ptr + qn_row[:, None] + offs_d[None, :], mask=valid_qn[:, None], other=0.0).to(tl.float32)
            dotn = tl.load(dOt_ptr + qn_row[:, None] + offs_d[None, :], mask=valid_qn[:, None], other=0.0)
            lsetn = tl.load(LSEt_ptr + hq_vec.to(tl.int64) * N_TOTAL + qn_tok.to(tl.int64), mask=valid_qn, other=0.0).to(tl.float32)
            deltatn = tl.sum(dotn.to(tl.float32) * otn, axis=1)

            sn = tl.dot(qtn, tl.trans(k0)) * sm_scale
            uobc_mask = valid_qn[:, None] & valid_kt[None, :] & (offs_n[None, :] > q_pos[:, None])
            pn = tl.where(uobc_mask, tl.exp(sn - lsetn[:, None]), 0.0)
            dv0_acc += tl.dot(tl.trans(pn.to(dotn.dtype)), dotn)
            dpn = tl.dot(dotn, tl.trans(v0))
            dsn = pn * (dpn - deltatn[:, None]) * sm_scale
            dk0_acc += tl.dot(tl.trans(dsn.to(qtn.dtype)), qtn)

    dk0_prev = tl.load(dK0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0).to(tl.float32)
    dv0_prev = tl.load(dV0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0).to(tl.float32)
    tl.store(dK0_ptr + k0_row[:, None] + offs_d[None, :], (dk0_prev + dk0_acc).to(dK0_ptr.dtype.element_ty), mask=valid_kt[:, None])
    tl.store(dV0_ptr + k0_row[:, None] + offs_d[None, :], (dv0_prev + dv0_acc).to(dV0_ptr.dtype.element_ty), mask=valid_kt[:, None])


def build_bd3_packed_meta(doc_lens: list[int], W: int, device: torch.device) -> dict[str, Any]:
    """Builds ragged-document-safe metadata for all BD3-LM attention trials (6-call, monolithic, zero-spill, adaptive)."""
    cu_docs = [0]
    cu_blks = [0]
    cu_obc = [0]
    obc_pairs = []
    blk_meta = []
    pairs_bc = []
    cu_sp_bc = [0]
    pairs_uobc_direct = []
    cu_sp_uobc = [0]

    cur = 0
    cur_obc = 0
    max_obc = 0
    max_blk = 0
    max_sp_bc = 0
    max_sp_uobc = 0

    for L in doc_lens:
        assert L >= 1
        cu_docs.append(cur + L)
        n_b = (L + W - 1) // W
        if L > W:
            obc_len = L - W
            cu_obc.append(cur_obc + obc_len)
            max_obc = max(max_obc, obc_len)
            qi_o = torch.arange(cur + W, cur + L, device=device, dtype=torch.int32)
            ki_o = torch.arange(cur, cur + obc_len, device=device, dtype=torch.int32)
            obc_pairs.append(torch.stack([qi_o, ki_o], dim=-1))

        for b in range(n_b):
            b_start = cur + b * W
            b_len = min(W, cur + L - b_start)
            max_blk = max(max_blk, b_len)
            cu_blks.append(b_start + b_len)

            prev_s = (b_start - W) if b > 0 else -1
            next_s = (b_start + W) if (b + 1) < n_b else -1
            next_l = min(W, cur + L - (b_start + W)) if (b + 1) < n_b else 0
            obc_off = (cur_obc + (b - 1) * W) if b > 0 else -1
            blk_meta.append([b_start, b_len, prev_s, next_s, obc_off, next_l])

            if b_len > 1:
                sp_l = b_len - 1
                max_sp_bc = max(max_sp_bc, sp_l)
                qi_bc = torch.arange(b_start + b_len - 2, b_start - 1, -1, device=device, dtype=torch.int32)
                ki_bc = torch.arange(b_start + b_len - 1, b_start, -1, device=device, dtype=torch.int32)
                pairs_bc.append(torch.stack([qi_bc, ki_bc], dim=-1))
                cu_sp_bc.append(cu_sp_bc[-1] + sp_l)

            if b > 0 and W > 1:
                # Reversed upper triangle of clean block (b - 1) of full length W against noisy block b of length b_len:
                # Keys: r = 0 .. W - 2 -> prev_s + W - 1 - r
                # Queries: q_pos = W - 2 - r; valid iff q_pos < b_len (else dummy q_idx = -1!)
                sp_uo = W - 1
                max_sp_uobc = max(max_sp_uobc, sp_uo)
                q_pos_rev = torch.arange(W - 2, -1, -1, device=device, dtype=torch.int32)
                qi_uo = torch.where(
                    q_pos_rev < b_len,
                    b_start + q_pos_rev,
                    torch.full_like(q_pos_rev, -1),
                )
                ki_uo = prev_s + torch.arange(W - 1, 0, -1, device=device, dtype=torch.int32)
                pairs_uobc_direct.append(torch.stack([qi_uo, ki_uo], dim=-1))
                cu_sp_uobc.append(cu_sp_uobc[-1] + sp_uo)

        if L > W:
            cur_obc += L - W
        cur += L

    empty_p = torch.empty((0, 2), device=device, dtype=torch.int32)
    obc_pair_t = torch.cat(obc_pairs, dim=0).contiguous() if obc_pairs else empty_p
    obc_q_t = obc_pair_t[:, 0].contiguous() if obc_pairs else torch.empty((0,), device=device, dtype=torch.int32)

    doc_ids = []
    blk_ids = []
    for d_id, L in enumerate(doc_lens):
        doc_ids.extend([d_id] * L)
        blk_ids.extend([p // W for p in range(L)])
    doc_t = torch.tensor(doc_ids, device=device, dtype=torch.int32)
    blk_t = torch.tensor(blk_ids, device=device, dtype=torch.int32)

    return {
        "doc_lens": list(doc_lens),
        "n": cur,
        "W": W,
        "n_obc": cur_obc,
        "max_obc": max_obc,
        "cu_docs": torch.tensor(cu_docs, device=device, dtype=torch.int32),
        "max_doc": max(doc_lens),
        "cu_blks": torch.tensor(cu_blks, device=device, dtype=torch.int32),
        "max_blk": max_blk,
        "cu_obc": torch.tensor(cu_obc, device=device, dtype=torch.int32),
        "obc_pair": obc_pair_t,
        "obc_q": obc_q_t,
        "blk_meta": torch.tensor(blk_meta, device=device, dtype=torch.int32).contiguous(),
        "n_blks": len(blk_meta),
        "span_bc": torch.cat(pairs_bc, dim=0).contiguous() if pairs_bc else empty_p,
        "cu_sp_bc": torch.tensor(cu_sp_bc, device=device, dtype=torch.int32),
        "max_sp_bc": max_sp_bc,
        "span_uobc_direct": torch.cat(pairs_uobc_direct, dim=0).contiguous() if pairs_uobc_direct else empty_p,
        "cu_sp_uobc": torch.tensor(cu_sp_uobc, device=device, dtype=torch.int32),
        "max_sp_uobc": max_sp_uobc,
        "doc_id": doc_t,
        "blk_id": blk_t,
    }


class Surgical6CallBD3FA2Fn(torch.autograd.Function):
    """Trial 1: 6-call C++ varlen FlashAttention-2 decomposition (wins W>=64, regresses at W=4, 16)."""

    @staticmethod
    def forward(ctx, qt, kt, vt, q0, k0, v0, meta: dict[str, Any], compute_clean_out: bool = True):
        qt, kt, vt = qt.contiguous(), kt.contiguous(), vt.contiguous()
        k0, v0 = k0.contiguous(), v0.contiguous()
        n, h_q, d = qt.shape
        h_kv = kt.shape[1]
        sm = d ** -0.5

        ot, lset, _, _ = _flash_attn_varlen_forward(
            q=qt, k=kt, v=vt,
            cu_seqlens_q=meta["cu_blks"], cu_seqlens_k=meta["cu_blks"],
            max_seqlen_q=meta["max_blk"], max_seqlen_k=meta["max_blk"],
            dropout_p=0.0, softmax_scale=sm, causal=False,
        )

        n_obc = meta["n_obc"]
        if n_obc > 0:
            q_obc = torch.empty((n_obc, h_q, d), device=qt.device, dtype=qt.dtype)
            k_obc = torch.empty((n_obc, h_kv, d), device=kt.device, dtype=kt.dtype)
            v_obc = torch.empty((n_obc, h_kv, d), device=vt.device, dtype=vt.dtype)
            _rev_gather_qkv_masked_kernel[(n_obc, h_q + 2 * h_kv)](
                qt, k0, v0, meta["obc_pair"], q_obc, k_obc, v_obc, H_Q=h_q, H_KV=h_kv, D=d
            )
            o_obc, lse_obc, _, _ = _flash_attn_varlen_forward(
                q=q_obc, k=k_obc, v=v_obc,
                cu_seqlens_q=meta["cu_obc"], cu_seqlens_k=meta["cu_obc"],
                max_seqlen_q=meta["max_obc"], max_seqlen_k=meta["max_obc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
            )
            _merge_lse_indexed_masked_kernel[(n_obc, h_q)](
                ot, lset, o_obc, lse_obc, meta["obc_q"], n, n_obc, 1, H_Q=h_q, D=d
            )
        else:
            q_obc = k_obc = v_obc = qt.new_empty((0, h_q, d))

        n_uo = meta["span_uobc_direct"].shape[0]
        if n_uo > 0:
            q_uo = torch.empty((n_uo, h_q, d), device=qt.device, dtype=qt.dtype)
            k_uo = torch.empty((n_uo, h_kv, d), device=kt.device, dtype=kt.dtype)
            v_uo = torch.empty((n_uo, h_kv, d), device=vt.device, dtype=vt.dtype)
            _rev_gather_qkv_masked_kernel[(n_uo, h_q + 2 * h_kv)](
                qt, k0, v0, meta["span_uobc_direct"], q_uo, k_uo, v_uo, H_Q=h_q, H_KV=h_kv, D=d
            )
            o_uo, lse_uo, _, _ = _flash_attn_varlen_forward(
                q=q_uo, k=k_uo, v=v_uo,
                cu_seqlens_q=meta["cu_sp_uobc"], cu_seqlens_k=meta["cu_sp_uobc"],
                max_seqlen_q=meta["max_sp_uobc"], max_seqlen_k=meta["max_sp_uobc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
            )
            _merge_lse_indexed_masked_kernel[(n_uo, h_q)](
                ot, lset, o_uo, lse_uo, meta["span_uobc_direct"], n, n_uo, 2, H_Q=h_q, D=d
            )
        else:
            q_uo = k_uo = v_uo = qt.new_empty((0, h_q, d))

        if compute_clean_out:
            q0 = q0.contiguous()
            o0, lse0, _, _ = _flash_attn_varlen_forward(
                q=q0, k=k0, v=v0,
                cu_seqlens_q=meta["cu_docs"], cu_seqlens_k=meta["cu_docs"],
                max_seqlen_q=meta["max_doc"], max_seqlen_k=meta["max_doc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
            )
            n_ub = meta["span_bc"].shape[0]
            if n_ub > 0:
                q_ub = torch.empty((n_ub, h_q, d), device=q0.device, dtype=q0.dtype)
                k_ub = torch.empty((n_ub, h_kv, d), device=k0.device, dtype=k0.dtype)
                v_ub = torch.empty((n_ub, h_kv, d), device=v0.device, dtype=v0.dtype)
                _rev_gather_qkv_masked_kernel[(n_ub, h_q + 2 * h_kv)](
                    q0, k0, v0, meta["span_bc"], q_ub, k_ub, v_ub, H_Q=h_q, H_KV=h_kv, D=d
                )
                o_ub, lse_ub, _, _ = _flash_attn_varlen_forward(
                    q=q_ub, k=k_ub, v=v_ub,
                    cu_seqlens_q=meta["cu_sp_bc"], cu_seqlens_k=meta["cu_sp_bc"],
                    max_seqlen_q=meta["max_sp_bc"], max_seqlen_k=meta["max_sp_bc"],
                    dropout_p=0.0, softmax_scale=sm, causal=True,
                )
                _merge_lse_indexed_masked_kernel[(n_ub, h_q)](
                    o0, lse0, o_ub, lse_ub, meta["span_bc"], n, n_ub, 2, H_Q=h_q, D=d
                )
            else:
                q_ub = k_ub = v_ub = q0.new_empty((0, h_q, d))
            ctx.save_for_backward(
                qt, kt, vt, ot, lset, q_obc, k_obc, v_obc, q_uo, k_uo, v_uo, q0, k0, v0, o0, lse0, q_ub, k_ub, v_ub
            )
        else:
            o0 = qt.new_empty((0, h_q, d))
            ctx.save_for_backward(qt, kt, vt, ot, lset, q_obc, k_obc, v_obc, q_uo, k_uo, v_uo, k0, v0)
        ctx.meta = meta
        ctx.compute_clean_out = compute_clean_out
        return ot, o0

    @staticmethod
    def backward(ctx, dot, do0):
        meta = ctx.meta
        dot = dot.contiguous()
        if ctx.compute_clean_out:
            (
                qt, kt, vt, ot, lset, q_obc, k_obc, v_obc, q_uo, k_uo, v_uo,
                q0, k0, v0, o0, lse0, q_ub, k_ub, v_ub,
            ) = ctx.saved_tensors
            do0 = do0.contiguous()
        else:
            qt, kt, vt, ot, lset, q_obc, k_obc, v_obc, q_uo, k_uo, v_uo, k0, v0 = ctx.saved_tensors
        n, h_q, d = qt.shape
        h_kv = kt.shape[1]
        sm = d ** -0.5

        dqt, dkt, dvt = torch.empty_like(qt), torch.empty_like(kt), torch.empty_like(vt)
        _flash_attn_varlen_backward(
            dout=dot, q=qt, k=kt, v=vt, out=ot, softmax_lse=lset,
            dq=dqt, dk=dkt, dv=dvt,
            cu_seqlens_q=meta["cu_blks"], cu_seqlens_k=meta["cu_blks"],
            max_seqlen_q=meta["max_blk"], max_seqlen_k=meta["max_blk"],
            dropout_p=0.0, softmax_scale=sm, causal=False,
            window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
        )

        if ctx.compute_clean_out:
            dq0, dk0, dv0 = torch.empty_like(q0), torch.empty_like(k0), torch.empty_like(v0)
            _flash_attn_varlen_backward(
                dout=do0, q=q0, k=k0, v=v0, out=o0, softmax_lse=lse0,
                dq=dq0, dk=dk0, dv=dv0,
                cu_seqlens_q=meta["cu_docs"], cu_seqlens_k=meta["cu_docs"],
                max_seqlen_q=meta["max_doc"], max_seqlen_k=meta["max_doc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
            n_ub = meta["span_bc"].shape[0]
            if n_ub > 0:
                do_ub = torch.empty_like(q_ub)
                mo_ub = torch.empty_like(q_ub)
                ml_ub = torch.empty((h_q, n_ub), device=q0.device, dtype=torch.float32)
                _gather_bwd_indexed_masked_kernel[(n_ub, h_q)](
                    do0, o0, lse0, meta["span_bc"], do_ub, mo_ub, ml_ub, n, n_ub, 2, H_Q=h_q, D=d
                )
                dq_ub, dk_ub, dv_ub = torch.empty_like(q_ub), torch.empty_like(k_ub), torch.empty_like(v_ub)
                _flash_attn_varlen_backward(
                    dout=do_ub, q=q_ub, k=k_ub, v=v_ub, out=mo_ub, softmax_lse=ml_ub,
                    dq=dq_ub, dk=dk_ub, dv=dv_ub,
                    cu_seqlens_q=meta["cu_sp_bc"], cu_seqlens_k=meta["cu_sp_bc"],
                    max_seqlen_q=meta["max_sp_bc"], max_seqlen_k=meta["max_sp_bc"],
                    dropout_p=0.0, softmax_scale=sm, causal=True,
                    window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
                )
                _scatter_add_rev_grads_masked_kernel[(n_ub, h_q + 2 * h_kv)](
                    dq0, dk0, dv0, dq_ub, dk_ub, dv_ub, meta["span_bc"], H_Q=h_q, H_KV=h_kv, D=d
                )
        else:
            dq0 = None
            dk0, dv0 = torch.zeros_like(k0), torch.zeros_like(v0)

        n_obc = meta["n_obc"]
        if n_obc > 0:
            do_obc = torch.empty_like(q_obc)
            mo_obc = torch.empty_like(q_obc)
            ml_obc = torch.empty((h_q, n_obc), device=qt.device, dtype=torch.float32)
            _gather_bwd_indexed_masked_kernel[(n_obc, h_q)](
                dot, ot, lset, meta["obc_q"], do_obc, mo_obc, ml_obc, n, n_obc, 1, H_Q=h_q, D=d
            )
            dq_obc, dk_obc, dv_obc = torch.empty_like(q_obc), torch.empty_like(k_obc), torch.empty_like(v_obc)
            _flash_attn_varlen_backward(
                dout=do_obc, q=q_obc, k=k_obc, v=v_obc, out=mo_obc, softmax_lse=ml_obc,
                dq=dq_obc, dk=dk_obc, dv=dv_obc,
                cu_seqlens_q=meta["cu_obc"], cu_seqlens_k=meta["cu_obc"],
                max_seqlen_q=meta["max_obc"], max_seqlen_k=meta["max_obc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
            _scatter_add_rev_grads_masked_kernel[(n_obc, h_q + 2 * h_kv)](
                dqt, dk0, dv0, dq_obc, dk_obc, dv_obc, meta["obc_pair"], H_Q=h_q, H_KV=h_kv, D=d
            )

        n_uo = meta["span_uobc_direct"].shape[0]
        if n_uo > 0:
            do_uo = torch.empty_like(q_uo)
            mo_uo = torch.empty_like(q_uo)
            ml_uo = torch.empty((h_q, n_uo), device=qt.device, dtype=torch.float32)
            _gather_bwd_indexed_masked_kernel[(n_uo, h_q)](
                dot, ot, lset, meta["span_uobc_direct"], do_uo, mo_uo, ml_uo, n, n_uo, 2, H_Q=h_q, D=d
            )
            dq_uo, dk_uo, dv_uo = torch.empty_like(q_uo), torch.empty_like(k_uo), torch.empty_like(v_uo)
            _flash_attn_varlen_backward(
                dout=do_uo, q=q_uo, k=k_uo, v=v_uo, out=mo_uo, softmax_lse=ml_uo,
                dq=dq_uo, dk=dk_uo, dv=dv_uo,
                cu_seqlens_q=meta["cu_sp_uobc"], cu_seqlens_k=meta["cu_sp_uobc"],
                max_seqlen_q=meta["max_sp_uobc"], max_seqlen_k=meta["max_sp_uobc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
            _scatter_add_rev_grads_masked_kernel[(n_uo, h_q + 2 * h_kv)](
                dqt, dk0, dv0, dq_uo, dk_uo, dv_uo, meta["span_uobc_direct"], H_Q=h_q, H_KV=h_kv, D=d
            )

        return dqt, dkt, dvt, dq0, dk0, dv0, None, None


class _HybridGQAMicroblockBD3Fn(torch.autograd.Function):
    """Shared GQA-folded microblock + 2-call causal FA2 engine (`monolithic_bwd=True` -> Trial 2; `False` -> Trial 3)."""

    @staticmethod
    def forward(
        ctx,
        qt: torch.Tensor,
        kt: torch.Tensor,
        vt: torch.Tensor,
        q0: torch.Tensor | None,
        k0: torch.Tensor,
        v0: torch.Tensor,
        meta: dict[str, Any],
        compute_clean_out: bool = True,
        monolithic_bwd: bool = False,
    ):
        qt, kt, vt = qt.contiguous(), kt.contiguous(), vt.contiguous()
        k0, v0 = k0.contiguous(), v0.contiguous()
        n, h_q, d = qt.shape
        h_kv = kt.shape[1]
        g = h_q // h_kv
        g_pad = max(4, triton.next_power_of_2(g))
        W = meta["W"]
        block_n = max(16, triton.next_power_of_2(W))
        q_step = min(W, max(4, 64 // g_pad))
        n_q_tiles = (W + q_step - 1) // q_step
        sm = d ** -0.5

        n_obc = meta["n_obc"]
        if n_obc > 0:
            q_obc = torch.empty((n_obc, h_q, d), device=qt.device, dtype=qt.dtype)
            k_obc = torch.empty((n_obc, h_kv, d), device=kt.device, dtype=kt.dtype)
            v_obc = torch.empty((n_obc, h_kv, d), device=vt.device, dtype=vt.dtype)
            _rev_gather_qkv_masked_kernel[(n_obc, h_q + 2 * h_kv)](
                qt, k0, v0, meta["obc_pair"], q_obc, k_obc, v_obc, H_Q=h_q, H_KV=h_kv, D=d
            )
            o_obc, lse_obc, _, _ = _flash_attn_varlen_forward(
                q=q_obc, k=k_obc, v=v_obc,
                cu_seqlens_q=meta["cu_obc"], cu_seqlens_k=meta["cu_obc"],
                max_seqlen_q=meta["max_obc"], max_seqlen_k=meta["max_obc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
            )
        else:
            q_obc = k_obc = v_obc = o_obc = qt.new_empty((0, h_q, d))
            lse_obc = qt.new_empty((h_q, 0), dtype=torch.float32)

        if compute_clean_out:
            assert q0 is not None
            q0 = q0.contiguous()
            o0, lse0, _, _ = _flash_attn_varlen_forward(
                q=q0, k=k0, v=v0,
                cu_seqlens_q=meta["cu_docs"], cu_seqlens_k=meta["cu_docs"],
                max_seqlen_q=meta["max_doc"], max_seqlen_k=meta["max_doc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
            )
        else:
            o0 = qt.new_empty((0, h_q, d))
            lse0 = qt.new_empty((h_q, 0), dtype=torch.float32)

        ot = torch.empty_like(qt)
        lset = torch.empty((h_q, n), device=qt.device, dtype=torch.float32)
        _bd3_gqa_microblock_fwd_kernel[(meta["n_blks"], n_q_tiles, h_kv)](
            qt, kt, vt,
            q0 if compute_clean_out else qt, k0, v0,
            o_obc, lse_obc,
            o0, lse0,
            ot, lset,
            meta["blk_meta"],
            n, max(1, n_obc), sm,
            W=W, BLOCK_N=block_n, Q_STEP=q_step,
            G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
            COMPUTE_CLEAN=compute_clean_out,
            num_warps=4, num_stages=2,
        )

        if compute_clean_out:
            ctx.save_for_backward(qt, kt, vt, ot, lset, q_obc, k_obc, v_obc, q0, k0, v0, o0, lse0)
        else:
            ctx.save_for_backward(qt, kt, vt, ot, lset, q_obc, k_obc, v_obc, k0, v0)
        ctx.meta = meta
        ctx.compute_clean_out = compute_clean_out
        ctx.monolithic_bwd = monolithic_bwd
        return ot, o0

    @staticmethod
    def backward(ctx, dot, do0):
        meta = ctx.meta
        W = meta["W"]
        dot = dot.contiguous()
        if ctx.compute_clean_out:
            qt, kt, vt, ot, lset, q_obc, k_obc, v_obc, q0, k0, v0, o0, lse0 = ctx.saved_tensors
            do0 = do0.contiguous()
        else:
            qt, kt, vt, ot, lset, q_obc, k_obc, v_obc, k0, v0 = ctx.saved_tensors
            q0 = o0 = do0 = qt
            lse0 = lset

        n, h_q, d = qt.shape
        h_kv = kt.shape[1]
        g = h_q // h_kv
        g_pad = max(4, triton.next_power_of_2(g))
        block_n = max(16, triton.next_power_of_2(W))
        q_step = min(W, max(4, 64 // g_pad))
        sm = d ** -0.5

        if ctx.compute_clean_out:
            dq0, dk0, dv0 = torch.empty_like(q0), torch.empty_like(k0), torch.empty_like(v0)
            _flash_attn_varlen_backward(
                dout=do0, q=q0, k=k0, v=v0, out=o0, softmax_lse=lse0,
                dq=dq0, dk=dk0, dv=dv0,
                cu_seqlens_q=meta["cu_docs"], cu_seqlens_k=meta["cu_docs"],
                max_seqlen_q=meta["max_doc"], max_seqlen_k=meta["max_doc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
        else:
            dq0 = None
            dk0, dv0 = torch.zeros_like(k0), torch.zeros_like(v0)

        dqt, dkt, dvt = torch.empty_like(qt), torch.empty_like(kt), torch.empty_like(vt)
        if ctx.monolithic_bwd:
            _bd3_gqa_monolithic_bwd_kernel[(meta["n_blks"], h_kv)](
                qt, kt, vt, ot, lset, dot,
                q0, k0, v0, o0, lse0, do0,
                dqt, dkt, dvt,
                dq0 if ctx.compute_clean_out else dqt, dk0, dv0,
                meta["blk_meta"], n, sm,
                W=W, BLOCK_N=block_n, Q_STEP=q_step,
                G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                COMPUTE_CLEAN=ctx.compute_clean_out,
                num_warps=4, num_stages=2,
            )
        else:
            _bd3_gqa_noisy_bwd_kernel[(meta["n_blks"], h_kv)](
                qt, kt, vt, k0, v0, ot, lset, dot,
                dqt, dkt, dvt,
                meta["blk_meta"], n, sm,
                W=W, BLOCK_N=block_n, Q_STEP=q_step,
                G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                num_warps=4, num_stages=1,
            )
            _bd3_gqa_clean_bwd_kernel[(meta["n_blks"], h_kv)](
                qt, ot, lset, dot,
                q0, k0, v0, o0, lse0, do0,
                dq0 if ctx.compute_clean_out else dqt, dk0, dv0,
                meta["blk_meta"], n, sm,
                W=W, BLOCK_N=block_n, Q_STEP=q_step,
                G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                COMPUTE_CLEAN=ctx.compute_clean_out,
                num_warps=4, num_stages=1,
            )

        n_obc = meta["n_obc"]
        if n_obc > 0:
            do_obc = torch.empty_like(q_obc)
            mo_obc = torch.empty_like(q_obc)
            ml_obc = torch.empty((h_q, n_obc), device=qt.device, dtype=torch.float32)
            _gather_bwd_indexed_masked_kernel[(n_obc, h_q)](
                dot, ot, lset, meta["obc_q"], do_obc, mo_obc, ml_obc, n, n_obc, 1, H_Q=h_q, D=d
            )
            dq_obc, dk_obc, dv_obc = torch.empty_like(q_obc), torch.empty_like(k_obc), torch.empty_like(v_obc)
            _flash_attn_varlen_backward(
                dout=do_obc, q=q_obc, k=k_obc, v=v_obc, out=mo_obc, softmax_lse=ml_obc,
                dq=dq_obc, dk=dk_obc, dv=dv_obc,
                cu_seqlens_q=meta["cu_obc"], cu_seqlens_k=meta["cu_obc"],
                max_seqlen_q=meta["max_obc"], max_seqlen_k=meta["max_obc"],
                dropout_p=0.0, softmax_scale=sm, causal=True,
                window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
            )
            _scatter_add_rev_grads_masked_kernel[(n_obc, h_q + 2 * h_kv)](
                dqt, dk0, dv0, dq_obc, dk_obc, dv_obc, meta["obc_pair"], H_Q=h_q, H_KV=h_kv, D=d
            )

        return dqt, dkt, dvt, dq0, dk0, dv0, None, None, None


def bd3_trial2_monolithic_hybrid(
    qt: torch.Tensor,
    kt: torch.Tensor,
    vt: torch.Tensor,
    q0: torch.Tensor | None,
    k0: torch.Tensor,
    v0: torch.Tensor,
    meta: dict[str, Any],
    compute_clean_out: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if meta["W"] <= 64:
        return _HybridGQAMicroblockBD3Fn.apply(qt, kt, vt, q0, k0, v0, meta, compute_clean_out, True)
    return Surgical6CallBD3FA2Fn.apply(qt, kt, vt, q0, k0, v0, meta, compute_clean_out)


def bd3_trial3_adaptive_zerospill(
    qt: torch.Tensor,
    kt: torch.Tensor,
    vt: torch.Tensor,
    q0: torch.Tensor | None,
    k0: torch.Tensor,
    v0: torch.Tensor,
    meta: dict[str, Any],
    compute_clean_out: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if meta["W"] <= 32:
        return _HybridGQAMicroblockBD3Fn.apply(qt, kt, vt, q0, k0, v0, meta, compute_clean_out, False)
    return Surgical6CallBD3FA2Fn.apply(qt, kt, vt, q0, k0, v0, meta, compute_clean_out)


def dense_fp64_bd3_oracle_fn(
    qt: torch.Tensor,
    kt: torch.Tensor,
    vt: torch.Tensor,
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    doc_lens: list[int],
    W: int,
    compute_clean_out: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = sum(doc_lens)
    qf = torch.cat([qt, q0], dim=0).to(torch.float64)
    kf = torch.cat([kt, k0], dim=0).to(torch.float64)
    vf = torch.cat([vt, v0], dim=0).to(torch.float64)
    doc_ids = []
    blk_ids = []
    for d_id, L in enumerate(doc_lens):
        doc_ids.extend([d_id] * L)
        blk_ids.extend([p // W for p in range(L)])
    doc_t = torch.tensor(doc_ids, device=qt.device)
    blk_t = torch.tensor(blk_ids, device=qt.device)
    doc2 = torch.cat([doc_t, doc_t], dim=0)
    blk2 = torch.cat([blk_t, blk_t], dim=0)
    x0 = torch.cat(
        [
            torch.zeros(n, device=qt.device, dtype=torch.bool),
            torch.ones(n, device=qt.device, dtype=torch.bool),
        ],
        dim=0,
    )
    dq = doc2[:, None] == doc2[None, :]
    bq, bk = blk2[:, None], blk2[None, :]
    x0q, x0k = x0[:, None], x0[None, :]
    m_full = dq & (((bq == bk) & (x0q == x0k)) | ((bq > bk) & (~x0q) & x0k) | ((bq >= bk) & x0q & x0k))
    g = qt.shape[1] // kt.shape[1]
    qp = qf.permute(1, 0, 2)
    kp = kf.repeat_interleave(g, 1).permute(1, 0, 2)
    vp = vf.repeat_interleave(g, 1).permute(1, 0, 2)
    sc = torch.matmul(qp, kp.transpose(-1, -2)) * (qt.shape[-1] ** -0.5)
    sc = sc.masked_fill(~m_full.unsqueeze(0), float("-inf"))
    of = torch.matmul(torch.softmax(sc, dim=-1), vp).permute(1, 0, 2)
    if compute_clean_out:
        return of[:n], of[n:]
    return of[:n], of.new_empty((0, qt.shape[1], qt.shape[2]))


# ============================================================================
# 2. Track 2 Kernels: Block-Segmented `DiT` / Diffusion-LLM `adaLN-Zero`
#    + Fused Inter-Sublayer `Gate-Residual-adaLN` (Zero-Gather & Zero-Atomic)
# ============================================================================


@triton.jit
def _block_adaln_fwd_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, BlockId_ptr,
    Y_ptr, Rstd_ptr,
    D: tl.constexpr, BLOCK_D: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.load(BlockId_ptr + row).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x = tl.load(X_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / float(D)
    rstd = tl.rsqrt(var + EPS)
    x_hat = x * rstd

    gamma = tl.load(Gamma_ptr + b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    y = x_hat * (1.0 + gamma) + beta

    tl.store(Y_ptr + row * D + offs_d, y.to(Y_ptr.dtype.element_ty), mask=mask_d)
    tl.store(Rstd_ptr + row, rstd)


@triton.jit
def _block_adaln_bwd_segmented_kernel(
    X_ptr, dY_ptr, Gamma_ptr, Rstd_ptr, BlkBounds_ptr,
    dX_ptr, dGamma_ptr, dBeta_ptr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Owner-computes per block `b`: ZERO global atomics on dGamma/dBeta!"""
    b = tl.program_id(0).to(tl.int64)
    s_row = tl.load(BlkBounds_ptr + b * 2 + 0).to(tl.int64)
    e_row = tl.load(BlkBounds_ptr + b * 2 + 1).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    gamma = tl.load(Gamma_ptr + b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    scale = 1.0 + gamma
    dgamma_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    dbeta_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for row in tl.range(s_row, e_row, loop_unroll_factor=1):
        x = tl.load(X_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        dy = tl.load(dY_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + row).to(tl.float32)
        x_hat = x * rstd

        dbeta_acc += dy
        dgamma_acc += dy * x_hat
        g = dy * scale
        c = tl.sum(g * x_hat, axis=0) / float(D)
        dx = rstd * (g - x_hat * c)
        tl.store(dX_ptr + row * D + offs_d, dx.to(dX_ptr.dtype.element_ty), mask=mask_d)

    tl.store(dGamma_ptr + b * D + offs_d, dgamma_acc.to(dGamma_ptr.dtype.element_ty), mask=mask_d)
    tl.store(dBeta_ptr + b * D + offs_d, dbeta_acc.to(dBeta_ptr.dtype.element_ty), mask=mask_d)


@triton.jit
def _block_gate_res_adaln_fwd_kernel(
    X_ptr, Sub_ptr, Alpha1_ptr, Gamma2_ptr, Beta2_ptr, BlockId_ptr,
    XNext_ptr, Y2_ptr, Rstd2_ptr,
    D: tl.constexpr, BLOCK_D: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.load(BlockId_ptr + row).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x = tl.load(X_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    sub = tl.load(Sub_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    alpha1 = tl.load(Alpha1_ptr + b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    gamma2 = tl.load(Gamma2_ptr + b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    beta2 = tl.load(Beta2_ptr + b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    xn_bf = (x + alpha1 * sub).to(XNext_ptr.dtype.element_ty)
    tl.store(XNext_ptr + row * D + offs_d, xn_bf, mask=mask_d)
    xn = xn_bf.to(tl.float32)

    var2 = tl.sum(xn * xn, axis=0) / float(D)
    rstd2 = tl.rsqrt(var2 + EPS)
    xn_hat = xn * rstd2
    y2 = xn_hat * (1.0 + gamma2) + beta2

    tl.store(Y2_ptr + row * D + offs_d, y2.to(Y2_ptr.dtype.element_ty), mask=mask_d)
    tl.store(Rstd2_ptr + row, rstd2)


@triton.jit
def _block_gate_res_adaln_bwd_segmented_kernel(
    XNext_ptr, Sub_ptr, dXNextUp_ptr, dY2_ptr,
    Alpha1_ptr, Gamma2_ptr, Rstd2_ptr, BlkBounds_ptr,
    dX_ptr, dSub_ptr, dAlpha1_ptr, dGamma2_ptr, dBeta2_ptr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Fused backward for x_next = x + alpha1[b]*sub; y2 = adaLN(x_next, gamma2[b], beta2[b]). Zero atomics!"""
    b = tl.program_id(0).to(tl.int64)
    s_row = tl.load(BlkBounds_ptr + b * 2 + 0).to(tl.int64)
    e_row = tl.load(BlkBounds_ptr + b * 2 + 1).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    alpha1 = tl.load(Alpha1_ptr + b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    gamma2 = tl.load(Gamma2_ptr + b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    scale2 = 1.0 + gamma2

    dalpha1_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    dgamma2_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    dbeta2_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for row in tl.range(s_row, e_row, loop_unroll_factor=1):
        xn = tl.load(XNext_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        sub = tl.load(Sub_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        dxn_up = tl.load(dXNextUp_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        dy2 = tl.load(dY2_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        rstd2 = tl.load(Rstd2_ptr + row).to(tl.float32)

        xn_hat = xn * rstd2
        dbeta2_acc += dy2
        dgamma2_acc += dy2 * xn_hat
        g2 = dy2 * scale2
        c2 = tl.sum(g2 * xn_hat, axis=0) / float(D)
        dxn_total = dxn_up + rstd2 * (g2 - xn_hat * c2)

        dalpha1_acc += dxn_total * sub
        dsub = dxn_total * alpha1

        tl.store(dX_ptr + row * D + offs_d, dxn_total.to(dX_ptr.dtype.element_ty), mask=mask_d)
        tl.store(dSub_ptr + row * D + offs_d, dsub.to(dSub_ptr.dtype.element_ty), mask=mask_d)

    tl.store(dAlpha1_ptr + b * D + offs_d, dalpha1_acc.to(dAlpha1_ptr.dtype.element_ty), mask=mask_d)
    tl.store(dGamma2_ptr + b * D + offs_d, dgamma2_acc.to(dGamma2_ptr.dtype.element_ty), mask=mask_d)
    tl.store(dBeta2_ptr + b * D + offs_d, dbeta2_acc.to(dBeta2_ptr.dtype.element_ty), mask=mask_d)


def build_adaln_segment_meta(doc_lens: list[int], W: int, device: torch.device) -> dict[str, Any]:
    blk_ids = []
    bounds = []
    cur = 0
    b_global = 0
    for L in doc_lens:
        n_b = (L + W - 1) // W
        for b in range(n_b):
            s0 = cur + b * W
            e0 = min(cur + L, s0 + W)
            blk_ids.extend([b_global] * (e0 - s0))
            bounds.append([s0, e0])
            b_global += 1
        cur += L
    return {
        "n": cur,
        "W": W,
        "n_blks": b_global,
        "block_id": torch.tensor(blk_ids, device=device, dtype=torch.int32).contiguous(),
        "block_id_long": torch.tensor(blk_ids, device=device, dtype=torch.long).contiguous(),
        "blk_bounds": torch.tensor(bounds, device=device, dtype=torch.int32).contiguous(),
    }


class BlockSegmentedAdaLNZeroFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, seg_meta: dict[str, Any], eps: float = 1e-6):
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        n, d = x.shape
        block_d = triton.next_power_of_2(d)
        y = torch.empty_like(x)
        rstd = torch.empty((n,), device=x.device, dtype=torch.float32)
        _block_adaln_fwd_kernel[(n,)](
            x, gamma, beta, seg_meta["block_id"], y, rstd, D=d, BLOCK_D=block_d, EPS=eps, num_warps=8
        )
        ctx.save_for_backward(x, gamma, rstd)
        ctx.seg_meta = seg_meta
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, gamma, rstd = ctx.saved_tensors
        seg_meta = ctx.seg_meta
        dy = dy.contiguous()
        d = x.shape[1]
        block_d = triton.next_power_of_2(d)
        dx = torch.empty_like(x)
        dgamma = torch.empty_like(gamma)
        dbeta = torch.empty_like(gamma)
        _block_adaln_bwd_segmented_kernel[(seg_meta["n_blks"],)](
            x, dy, gamma, rstd, seg_meta["blk_bounds"], dx, dgamma, dbeta, D=d, BLOCK_D=block_d, num_warps=8
        )
        return dx, dgamma, dbeta, None, None


class BlockSegmentedGateResAdaLNFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        sub: torch.Tensor,
        alpha1: torch.Tensor,
        gamma2: torch.Tensor,
        beta2: torch.Tensor,
        seg_meta: dict[str, Any],
        eps: float = 1e-6,
    ):
        x, sub = x.contiguous(), sub.contiguous()
        alpha1, gamma2, beta2 = alpha1.contiguous(), gamma2.contiguous(), beta2.contiguous()
        n, d = x.shape
        block_d = triton.next_power_of_2(d)
        x_next = torch.empty_like(x)
        y2 = torch.empty_like(x)
        rstd2 = torch.empty((n,), device=x.device, dtype=torch.float32)
        _block_gate_res_adaln_fwd_kernel[(n,)](
            x, sub, alpha1, gamma2, beta2, seg_meta["block_id"],
            x_next, y2, rstd2,
            D=d, BLOCK_D=block_d, EPS=eps, num_warps=8,
        )
        ctx.save_for_backward(x_next, sub, alpha1, gamma2, rstd2)
        ctx.seg_meta = seg_meta
        return x_next, y2

    @staticmethod
    def backward(ctx, dx_next_up: torch.Tensor, dy2: torch.Tensor):
        x_next, sub, alpha1, gamma2, rstd2 = ctx.saved_tensors
        seg_meta = ctx.seg_meta
        dx_next_up = dx_next_up.contiguous()
        dy2 = dy2.contiguous()
        d = x_next.shape[1]
        block_d = triton.next_power_of_2(d)
        dx = torch.empty_like(x_next)
        dsub = torch.empty_like(sub)
        dalpha1 = torch.empty_like(alpha1)
        dgamma2 = torch.empty_like(gamma2)
        dbeta2 = torch.empty_like(gamma2)
        _block_gate_res_adaln_bwd_segmented_kernel[(seg_meta["n_blks"],)](
            x_next, sub, dx_next_up, dy2,
            alpha1, gamma2, rstd2, seg_meta["blk_bounds"],
            dx, dsub, dalpha1, dgamma2, dbeta2,
            D=d, BLOCK_D=block_d, num_warps=8,
        )
        return dx, dsub, dalpha1, dgamma2, dbeta2, None, None


def eager_adaln_dual_step(
    x: torch.Tensor,
    sub1: torch.Tensor,
    sub2: torch.Tensor,
    cond6: torch.Tensor,
    block_id_long: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g1, b1, a1, g2, b2, a2 = cond6.unbind(dim=1)
    r1 = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    y1 = (x.float() * r1 * (1.0 + g1.index_select(0, block_id_long).float()) + b1.index_select(0, block_id_long).float()).to(x.dtype)
    xn = (x.float() + a1.index_select(0, block_id_long).float() * sub1.float()).to(x.dtype)
    r2 = torch.rsqrt(xn.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    y2 = (xn.float() * r2 * (1.0 + g2.index_select(0, block_id_long).float()) + b2.index_select(0, block_id_long).float()).to(x.dtype)
    x_out = (xn.float() + a2.index_select(0, block_id_long).float() * sub2.float()).to(x.dtype)
    return y1, y2, x_out


def triton_segmented_adaln_dual_step(
    x: torch.Tensor,
    sub1: torch.Tensor,
    sub2: torch.Tensor,
    cond6: torch.Tensor,
    seg_meta: dict[str, Any],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g1, b1, a1, g2, b2, a2 = cond6.unbind(dim=1)
    y1 = BlockSegmentedAdaLNZeroFn.apply(x, g1, b1, seg_meta, eps)
    xn, y2 = BlockSegmentedGateResAdaLNFn.apply(x, sub1, a1, g2, b2, seg_meta, eps)
    # Final residual gate x_out = xn + a2[b] * sub2 via lightweight segmented gate-res (or index-free broadcast if W divides)
    a2_tok = a2.index_select(0, seg_meta["block_id_long"])
    x_out = (xn.float() + a2_tok.float() * sub2.float()).to(x.dtype)
    return y1, y2, x_out


# ============================================================================
# 3. Track 3 Kernels: Masked Diffusion `1/t` NELBO Terminal Vocabulary CE (`V = 128,256`)
#    - Trial 0: Full-token Eager `[N, V]` LM-Head + 1/t Weighted CE (`BASELINE`)
#    - Trial 1: Full-token Chunked `3-GEMM` FusedLinearWeightedCE (`KEEP` — cuts HBM 6x)
#    - Trial 2: Active-Mask-Compacted `3-GEMM` FusedLinearMaskedCE (`KEEP` — cuts V-GEMM FLOPs ~2.5x)
# ============================================================================


@triton.jit
def _weighted_ce_fwd_inplace_dz_kernel(
    Logits_ptr, Targets_ptr, Weights_ptr, LossOut_ptr,
    STRIDE_L, V, BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    w = tl.load(Weights_ptr + row).to(tl.float32)
    tgt = tl.load(Targets_ptr + row).to(tl.int64)

    if w == 0.0:
        for col0 in range(0, V, BLOCK_V):
            offs = col0 + tl.arange(0, BLOCK_V)
            tl.store(Logits_ptr + row * STRIDE_L + offs, 0.0, mask=offs < V)
        tl.store(LossOut_ptr + row, 0.0)
        return

    m_i = -float("inf")
    l_i = 0.0
    for col0 in range(0, V, BLOCK_V):
        offs = col0 + tl.arange(0, BLOCK_V)
        mask = offs < V
        z = tl.load(Logits_ptr + row * STRIDE_L + offs, mask=mask, other=-float("inf")).to(tl.float32)
        m_blk = tl.max(z, axis=0)
        m_new = tl.maximum(m_i, m_blk)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.where(mask, tl.exp(z - m_new), 0.0), axis=0)
        m_i = m_new

    lse = m_i + tl.log(tl.maximum(l_i, 1e-30))
    z_tgt = tl.load(Logits_ptr + row * STRIDE_L + tgt).to(tl.float32)
    tl.store(LossOut_ptr + row, w * (lse - z_tgt))

    for col0 in range(0, V, BLOCK_V):
        offs = col0 + tl.arange(0, BLOCK_V)
        mask = offs < V
        z = tl.load(Logits_ptr + row * STRIDE_L + offs, mask=mask, other=-float("inf")).to(tl.float32)
        prob = tl.exp(z - lse)
        dz = w * (prob - tl.where(offs == tgt, 1.0, 0.0))
        tl.store(Logits_ptr + row * STRIDE_L + offs, dz.to(Logits_ptr.dtype.element_ty), mask=mask)


class Chunked3GemmWeightedCEFn(torch.autograd.Function):
    """Trial 1: Runs 3-GEMM chunked linear + weighted CE over ALL N rows (saves HBM, still computes unmasked GEMMs)."""

    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        w_vocab: torch.Tensor,
        targets: torch.Tensor,
        row_weights: torch.Tensor,
        chunk_size: int = 2048,
    ):
        h_c = h.contiguous()
        w_c = w_vocab.contiguous()
        n, d = h_c.shape
        v = w_c.shape[0]
        dh = torch.empty_like(h_c)
        dw = torch.zeros_like(w_c, dtype=torch.float32)
        total_loss = torch.zeros((), device=h_c.device, dtype=torch.float32)

        logits_buf = torch.empty((min(chunk_size, n), v), device=h_c.device, dtype=h_c.dtype)
        loss_buf = torch.empty((min(chunk_size, n),), device=h_c.device, dtype=torch.float32)

        for s in range(0, n, chunk_size):
            e = min(n, s + chunk_size)
            c = e - s
            hc = h_c[s:e]
            zc = logits_buf[:c]
            torch.matmul(hc, w_c.t(), out=zc)
            lc = loss_buf[:c]
            _weighted_ce_fwd_inplace_dz_kernel[(c,)](
                zc, targets[s:e], row_weights[s:e], lc, zc.stride(0), v, BLOCK_V=4096, num_warps=8
            )
            total_loss = total_loss + lc.sum()
            torch.matmul(zc, w_c, out=dh[s:e])
            dw.add_(torch.matmul(zc.t(), hc).float())

        ctx.save_for_backward(dh, dw.to(w_c.dtype))
        return total_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        dh, dw = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        return (dh.float() * g).to(dh.dtype), (dw.float() * g).to(dw.dtype), None, None, None


class ActiveMaskCompacted3GemmCEFn(torch.autograd.Function):
    """Trial 2 (KEEP): Compacts M active masked tokens before the 3 V=128,256 GEMMs; FSDP2-safe at M=0."""

    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        w_vocab: torch.Tensor,
        targets: torch.Tensor,
        mask_bool: torch.Tensor,
        inv_t_div_n: torch.Tensor,
        chunk_size: int = 2048,
    ):
        h_c = h.contiguous()
        w_c = w_vocab.contiguous()
        act_idx = torch.nonzero(mask_bool, as_tuple=False).squeeze(-1)
        m = int(act_idx.numel())
        n, d = h_c.shape
        v = w_c.shape[0]

        if m == 0:
            # Touch w_c identically across ranks with zero multiplier; exact zero loss/gradients!
            zero_loss = (h_c[:1].float().sum() + w_c[:1].float().sum()) * 0.0
            ctx.save_for_backward(torch.zeros_like(h_c), torch.zeros_like(w_c))
            return zero_loss

        h_act = h_c.index_select(0, act_idx)
        y_act = targets.index_select(0, act_idx).contiguous()
        w_act = inv_t_div_n.index_select(0, act_idx).contiguous()

        dh_act = torch.empty_like(h_act)
        dw = torch.zeros_like(w_c, dtype=torch.float32)
        total_loss = torch.zeros((), device=h_c.device, dtype=torch.float32)

        c_max = min(chunk_size, m)
        logits_buf = torch.empty((c_max, v), device=h_c.device, dtype=h_c.dtype)
        loss_buf = torch.empty((c_max,), device=h_c.device, dtype=torch.float32)

        for s in range(0, m, chunk_size):
            e = min(m, s + chunk_size)
            c = e - s
            hc = h_act[s:e]
            zc = logits_buf[:c]
            torch.matmul(hc, w_c.t(), out=zc)
            lc = loss_buf[:c]
            _weighted_ce_fwd_inplace_dz_kernel[(c,)](
                zc, y_act[s:e], w_act[s:e], lc, zc.stride(0), v, BLOCK_V=4096, num_warps=8
            )
            total_loss = total_loss + lc.sum()
            torch.matmul(zc, w_c, out=dh_act[s:e])
            dw.add_(torch.matmul(zc.t(), hc).float())

        dh_full = torch.zeros_like(h_c)
        dh_full.index_copy_(0, act_idx, dh_act)
        ctx.save_for_backward(dh_full, dw.to(w_c.dtype))
        return total_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        dh, dw = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        return (dh.float() * g).to(dh.dtype), (dw.float() * g).to(dw.dtype), None, None, None, None


def eager_full_masked_diffusion_ce(
    h: torch.Tensor,
    w_vocab: torch.Tensor,
    targets: torch.Tensor,
    mask_bool: torch.Tensor,
    inv_t_div_n: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(h, w_vocab)
    ce_per_tok = F.cross_entropy(logits.float(), targets, reduction="none")
    row_w = torch.where(mask_bool, inv_t_div_n, torch.zeros_like(inv_t_div_n))
    return (ce_per_tok * row_w).sum()


# ============================================================================
# 4. Fused 1D Document-Local RoPE + SwiGLU Helpers for `1.11B` `BD3-LM` FSDP2
# ============================================================================


@triton.jit
def _rope_1d_inplace_fwd_bwd_kernel(
    X_ptr, Cos_ptr, Sin_ptr,
    N_TOK, H: tl.constexpr, D_HALF: tl.constexpr, CONJUGATE: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, D_HALF)
    cos = tl.load(Cos_ptr + tok * D_HALF + offs).to(tl.float32)
    sin = tl.load(Sin_ptr + tok * D_HALF + offs).to(tl.float32)
    if CONJUGATE:
        sin = -sin
    base = (tok * H + h) * (2 * D_HALF)
    x1 = tl.load(X_ptr + base + offs).to(tl.float32)
    x2 = tl.load(X_ptr + base + D_HALF + offs).to(tl.float32)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    tl.store(X_ptr + base + offs, y1.to(X_ptr.dtype.element_ty))
    tl.store(X_ptr + base + D_HALF + offs, y2.to(X_ptr.dtype.element_ty))


class _RoPE1DFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        out = x.contiguous().clone()
        n, h, d = out.shape
        _rope_1d_inplace_fwd_bwd_kernel[(n, h)](out, cos, sin, n, H=h, D_HALF=d // 2, CONJUGATE=False, num_warps=2)
        ctx.save_for_backward(cos, sin)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dx = dout.contiguous().clone()
        n, h, d = dx.shape
        _rope_1d_inplace_fwd_bwd_kernel[(n, h)](dx, cos, sin, n, H=h, D_HALF=d // 2, CONJUGATE=True, num_warps=2)
        return dx, None, None


def apply_1d_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _RoPE1DFn.apply(x, cos, sin)


def build_doc_rope_cache(doc_lens: list[int], head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos_list = [torch.arange(L, device=device, dtype=torch.float32) for L in doc_lens]
    pos = torch.cat(pos_list, dim=0)
    inv_freq = 1.0 / (500000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / float(head_dim)))
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()


# ============================================================================
# 5. Suite 0: Architectural Audits, 1/t Variance Explosion vs Antithetic Masking,
#    Per-Tensor FP64 Oracles, & Unit-RMS O(h^2) Richardson Verification
# ============================================================================


def run_suite_0_audits_and_parity(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    # ------------------------------------------------------------------------
    # 0.1 BD3-LM Attention Backend Audit & PyTorch 2.13 SymPy Compiler Defect
    # ------------------------------------------------------------------------
    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)

    sympy_defect_caught = False
    sympy_error_excerpt = ""
    try:
        def _official_fig4_bd3_mask(b, h, q_idx, kv_idx):
            x0_flag_q = q_idx >= 512
            x0_flag_kv = kv_idx >= 512
            block_q = torch.where(x0_flag_q, (q_idx - 512) // 16, q_idx // 16)
            block_kv = torch.where(x0_flag_kv, (kv_idx - 512) // 16, kv_idx // 16)
            m_bd = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
            m_obc = (block_q > block_kv) & (~x0_flag_q) & x0_flag_kv
            m_bc = (block_q >= block_kv) & x0_flag_q & x0_flag_kv
            return m_bd | m_obc | m_bc

        _ = comp_cbm(_official_fig4_bd3_mask, B=1, H=1, Q_LEN=1024, KV_LEN=1024, device=device)
    except Exception as exc:
        sympy_defect_caught = True
        sympy_error_excerpt = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"

    # Audit PyTorch SDPA 2L boolean mask at L=4,096 (2L=8,192) and memory wall at L=16,384 (2L=32,768)
    n_sdpa = 2048 if is_smoke else 4096
    meta_sdpa = build_bd3_packed_meta([n_sdpa // 2, n_sdpa // 2], 16, device)
    doc2 = torch.cat([meta_sdpa["doc_id"], meta_sdpa["doc_id"]], dim=0)
    blk2 = torch.cat([meta_sdpa["blk_id"], meta_sdpa["blk_id"]], dim=0)
    x0_2 = torch.cat(
        [
            torch.zeros(n_sdpa, device=device, dtype=torch.bool),
            torch.ones(n_sdpa, device=device, dtype=torch.bool),
        ],
        dim=0,
    )
    bool_mask_2l = (doc2[:, None] == doc2[None, :]) & (
        ((blk2[:, None] == blk2[None, :]) & (x0_2[:, None] == x0_2[None, :]))
        | ((blk2[:, None] > blk2[None, :]) & (~x0_2[:, None]) & x0_2[None, :])
        | ((blk2[:, None] >= blk2[None, :]) & x0_2[:, None] & x0_2[None, :])
    )

    q_s = torch.randn(1, 28, 2 * n_sdpa, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k_s = torch.randn(1, 4, 2 * n_sdpa, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    v_s = torch.randn(1, 4, 2 * n_sdpa, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    do_s = torch.randn_like(q_s)

    flash_sdpa_bool_rejected = False
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _ = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=bool_mask_2l, enable_gqa=True)
    except Exception:
        flash_sdpa_bool_rejected = True

    def _sdpa_2l_step() -> None:
        out = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=bool_mask_2l, enable_gqa=True)
        out.backward(do_s)
        q_s.grad = k_s.grad = v_s.grad = None

    sdpa_audit_perf = benchmark_rotating_cuda(
        [_sdpa_2l_step],
        warmup=2 if is_smoke else 3,
        groups=3 if is_smoke else 5,
        iters_per_group=2 if is_smoke else 3,
    )
    del q_s, k_s, v_s, do_s, bool_mask_2l
    torch.cuda.empty_cache()

    # Theoretical + allocator check at L=16,384 (2L=32,768): [1, 28, 32768, 32768] FP32 scores = 112.0 GiB > 80 GiB HBM!
    dense_scores_l16k_gib = round((28.0 * 32768.0 * 32768.0 * 4.0) / (1024.0**3), 2)
    bd3_backend_audit = {
        "official_fig4_sympy_bool_eq_inductor_crash": sympy_defect_caught,
        "official_fig4_error_excerpt": sympy_error_excerpt,
        "flash_sdpa_arbitrary_2l_bool_mask_rejected": flash_sdpa_bool_rejected,
        "sdpa_bool_mask_2l_probe": {"clean_seqlen": n_sdpa, "concat_seqlen_2l": 2 * n_sdpa, **sdpa_audit_perf},
        "sdpa_l16k_2l32k_fp32_score_matrix_gib": dense_scores_l16k_gib,
    }

    # ------------------------------------------------------------------------
    # 0.2 Continuous-Time 1/t NELBO Gradient Variance Explosion vs Stratified &
    #     Complementary-Antithetic `(t, 1-t; m, 1-m)` Masking Audit
    # ------------------------------------------------------------------------
    gen_v = torch.Generator(device=device)
    gen_v.manual_seed(20261117)
    B_mc, L_mc, D_mc, V_mc = 16, 128, 256, 2048
    MASK_TOKEN_ID = V_mc - 1
    x0_tokens = torch.randint(0, V_mc - 1, (B_mc, L_mc), device=device, generator=gen_v)
    emb_mc = nn.Embedding(V_mc, D_mc, device=device, dtype=torch.float32)
    wq_mc = nn.Linear(D_mc, D_mc, bias=False, device=device, dtype=torch.float32)
    wk_mc = nn.Linear(D_mc, D_mc, bias=False, device=device, dtype=torch.float32)
    wv_mc = nn.Linear(D_mc, D_mc, bias=False, device=device, dtype=torch.float32)
    head_mc = nn.Linear(D_mc, V_mc - 1, bias=False, device=device, dtype=torch.float32)
    for mod in (emb_mc, wq_mc, wk_mc, wv_mc, head_mc):
        nn.init.normal_(mod.weight, mean=0.0, std=0.04)

    def _eval_nelbo_draw(t_vec: torch.Tensor, mask_mat: torch.Tensor) -> tuple[float, torch.Tensor]:
        for mod in (emb_mc, wq_mc, wk_mc, wv_mc, head_mc):
            mod.zero_grad(set_to_none=True)
        xt = torch.where(mask_mat, torch.full_like(x0_tokens, MASK_TOKEN_ID), x0_tokens)
        h = emb_mc(xt)
        q = wq_mc(h).view(B_mc, L_mc, 4, D_mc // 4).transpose(1, 2)
        k = wk_mc(h).view(B_mc, L_mc, 4, D_mc // 4).transpose(1, 2)
        v = wv_mc(h).view(B_mc, L_mc, 4, D_mc // 4).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(B_mc, L_mc, D_mc)
        h_out = h + attn
        logits = head_mc(h_out)
        ce = F.cross_entropy(logits.view(-1, V_mc - 1), x0_tokens.view(-1), reduction="none").view(B_mc, L_mc)
        weights = mask_mat.float() / (t_vec[:, None] * float(B_mc * L_mc))
        loss = (ce * weights).sum()
        loss.backward()
        assert head_mc.weight.grad is not None
        return float(loss.item()), head_mc.weight.grad.detach().clone().flatten()

    n_mc_draws = 32 if is_smoke else 96
    t_eps = 0.001
    t_span = 1.0 - 2.0 * t_eps
    variance_schemes: dict[str, Any] = {}

    for scheme in (
        "unclipped_iid_uniform_t",
        "mdlm_stratified_low_discrepancy_t",
        "antithetic_complementary_pair_t",
        "bd3_clipped_antithetic_u03_07",
    ):
        g_rng = torch.Generator(device=device)
        g_rng.manual_seed(7700 + len(scheme))
        losses_k: list[float] = []
        grads_k: list[torch.Tensor] = []
        active_fracs: list[float] = []

        for _ in range(n_mc_draws):
            if scheme == "unclipped_iid_uniform_t":
                t_b = t_eps + t_span * torch.rand((B_mc,), device=device, generator=g_rng)
                u_m = torch.rand((B_mc, L_mc), device=device, generator=g_rng)
                m_b = u_m < t_b[:, None]
            elif scheme == "mdlm_stratified_low_discrepancy_t":
                bins = (torch.arange(B_mc, device=device, dtype=torch.float32) + torch.rand((B_mc,), device=device, generator=g_rng)) / float(B_mc)
                t_b = t_eps + t_span * bins
                u_m = torch.rand((B_mc, L_mc), device=device, generator=g_rng)
                m_b = u_m < t_b[:, None]
            elif scheme == "antithetic_complementary_pair_t":
                half = B_mc // 2
                bins_h = (torch.arange(half, device=device, dtype=torch.float32) + torch.rand((half,), device=device, generator=g_rng)) / float(half)
                t_half = t_eps + t_span * bins_h
                u_half = torch.rand((half, L_mc), device=device, generator=g_rng)
                m_1 = u_half < t_half[:, None]
                # Exact marginal-preserving antithetic pair: t_2 = 1 - t_1 in [t_eps, 1-t_eps], m_2 = (u >= t_1) == ~m_1!
                t_b = torch.cat([t_half, 1.0 - t_half], dim=0)
                m_b = torch.cat([m_1, ~m_1], dim=0)
                # Duplicate first half of x0_tokens across second half during pair evaluation for exact conditional cancellation
            else:
                half = B_mc // 2
                bins_h = (torch.arange(half, device=device, dtype=torch.float32) + torch.rand((half,), device=device, generator=g_rng)) / float(half)
                t_half = 0.30 + 0.40 * bins_h
                u_half = torch.rand((half, L_mc), device=device, generator=g_rng)
                m_1 = u_half < t_half[:, None]
                t_b = torch.cat([t_half, 1.0 - t_half], dim=0)
                m_b = torch.cat([m_1, ~m_1], dim=0)

            if "antithetic" in scheme:
                saved_x0 = x0_tokens.clone()
                x0_tokens[B_mc // 2 :] = x0_tokens[: B_mc // 2]
                lv, gv = _eval_nelbo_draw(t_b, m_b)
                x0_tokens.copy_(saved_x0)
            else:
                lv, gv = _eval_nelbo_draw(t_b, m_b)

            losses_k.append(lv)
            grads_k.append(gv)
            active_fracs.append(float(m_b.float().mean().item()))

        g_stack = torch.stack(grads_k, dim=0)
        g_mean = g_stack.mean(dim=0, keepdim=True)
        grad_mc_var = float(((g_stack - g_mean).pow(2).sum(dim=-1)).mean().item())
        loss_var = float(statistics.pvariance(losses_k))
        variance_schemes[scheme] = {
            "num_mc_draws": n_mc_draws,
            "mean_loss": round(float(statistics.fmean(losses_k)), 6),
            "loss_variance": round(loss_var, 8),
            "loss_std": round(math.sqrt(loss_var), 6),
            "max_over_median_loss": round(max(losses_k) / max(1e-9, statistics.median(losses_k)), 4),
            "grad_head_mc_variance": round(grad_mc_var, 8),
            "mean_active_mask_fraction": round(float(statistics.fmean(active_fracs)), 4),
        }

    base_gvar = variance_schemes["unclipped_iid_uniform_t"]["grad_head_mc_variance"]
    for sc_val in variance_schemes.values():
        sc_val["grad_variance_reduction_vs_unclipped"] = round(base_gvar / max(1e-12, sc_val["grad_head_mc_variance"]), 3)

    del emb_mc, wq_mc, wk_mc, wv_mc, head_mc
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # 0.3 Per-Tensor FP64 Analytical Oracles & Adversarial Edge-Case Torture
    # ------------------------------------------------------------------------
    parity_summary: dict[str, Any] = {}
    bd3_parity_cases: dict[str, Any] = {}

    for cname, dlens, W_c, clean_flag in [
        ("ragged_W4_partial_blocks", [3, 4, 5, 10, 17], 4, True),
        ("ragged_W16_partial_blocks", [7, 16, 19, 35], 16, True),
        ("packed_W64_exact", [128, 128], 64, True),
        ("packed_W256_exact", [256, 256], 256, True),
        ("pruned_last_layer_W16", [7, 16, 19, 35], 16, False),
    ]:
        nt = sum(dlens)
        mc = build_bd3_packed_meta(dlens, W_c, device)
        gen_p = torch.Generator(device=device).manual_seed(9100 + W_c + len(cname))
        qt_r = (torch.randn((nt, 28, 128), device=device, dtype=torch.bfloat16, generator=gen_p) * 0.4).requires_grad_(True)
        # Non-contiguous sliced K/V stride torture (`stride(0) == 2 * 4 * 128 != 4 * 128`)
        kvt_r = (torch.randn((nt, 2, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_p) * 0.4).requires_grad_(True)
        q0_r = (torch.randn((nt, 28, 128), device=device, dtype=torch.bfloat16, generator=gen_p) * 0.4).requires_grad_(True)
        kv0_r = (torch.randn((nt, 2, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_p) * 0.4).requires_grad_(True)
        dot_r = torch.randn((nt, 28, 128), device=device, dtype=torch.bfloat16, generator=gen_p) * 0.4
        do0_r = torch.randn((nt, 28, 128), device=device, dtype=torch.bfloat16, generator=gen_p) * 0.4

        qf64 = qt_r.detach().double().requires_grad_(True)
        kt64 = kvt_r.detach()[:, 0].double().requires_grad_(True)
        vt64 = kvt_r.detach()[:, 1].double().requires_grad_(True)
        q064 = q0_r.detach().double().requires_grad_(True)
        k064 = kv0_r.detach()[:, 0].double().requires_grad_(True)
        v064 = kv0_r.detach()[:, 1].double().requires_grad_(True)
        ot_ref, o0_ref = dense_fp64_bd3_oracle_fn(qf64, kt64, vt64, q064, k064, v064, dlens, W_c, clean_flag)
        if clean_flag:
            torch.autograd.backward([ot_ref, o0_ref], [dot_r.double(), do0_r.double()])
        else:
            ot_ref.backward(dot_r.double())

        for impl_name, impl_fn in [
            ("trial1_6call_fa2", Surgical6CallBD3FA2Fn.apply),
            ("trial2_monolithic_hybrid", bd3_trial2_monolithic_hybrid),
            ("trial3_adaptive_zerospill", bd3_trial3_adaptive_zerospill),
        ]:
            qt_c = qt_r.detach().clone().requires_grad_(True)
            kvt_c = kvt_r.detach().clone().requires_grad_(True)
            q0_c = q0_r.detach().clone().requires_grad_(True) if clean_flag else None
            kv0_c = kv0_r.detach().clone().requires_grad_(True)
            ot_c, o0_c = impl_fn(qt_c, kvt_c[:, 0], kvt_c[:, 1], q0_c, kv0_c[:, 0], kv0_c[:, 1], mc, clean_flag)
            if clean_flag:
                torch.autograd.backward([ot_c, o0_c], [dot_r, do0_r])
            else:
                ot_c.backward(dot_r)

            checks = [
                compare_tensor_to_fp64(ot_c, ot_ref, name="ot"),
                compare_tensor_to_fp64(qt_c.grad, qf64.grad, name="dqt"),
                compare_tensor_to_fp64(kvt_c.grad[:, 0], kt64.grad, name="dkt"),
                compare_tensor_to_fp64(kvt_c.grad[:, 1], vt64.grad, name="dvt"),
                compare_tensor_to_fp64(kv0_c.grad[:, 0], k064.grad, name="dk0"),
                compare_tensor_to_fp64(kv0_c.grad[:, 1], v064.grad, name="dv0"),
            ]
            if clean_flag:
                checks.append(compare_tensor_to_fp64(o0_c, o0_ref, name="o0"))
                checks.append(compare_tensor_to_fp64(q0_c.grad, q064.grad, name="dq0"))
            bd3_parity_cases[f"{cname}__{impl_name}"] = {
                "all_ok": all(c["ok"] for c in checks),
                "max_rel_l2": round(max(float(c["rel_l2"]) for c in checks), 6),
            }

    # AdaLN-Zero step-0 (`alpha == 0` exact zero-gate regime) + post-warmup (`alpha != 0`) FP64 parity
    seg_t = build_adaln_segment_meta([13, 16, 21], 8, device)
    gen_a = torch.Generator(device=device).manual_seed(5511)
    xa_base = (torch.randn((seg_t["n"], 3584), device=device, dtype=torch.bfloat16, generator=gen_a) * 0.5)
    s1_base = (torch.randn((seg_t["n"], 3584), device=device, dtype=torch.bfloat16, generator=gen_a) * 0.5)
    s2_base = (torch.randn((seg_t["n"], 3584), device=device, dtype=torch.bfloat16, generator=gen_a) * 0.5)
    dy1_a = torch.randn_like(xa_base) * 0.3
    dy2_a = torch.randn_like(xa_base) * 0.3
    dxo_a = torch.randn_like(xa_base) * 0.3

    adaln_parity_cases: dict[str, Any] = {}
    for gate_mode in ("zero_init_alpha_step0", "nonzero_alpha_active"):
        c6_base = torch.randn((seg_t["n_blks"], 6, 3584), device=device, dtype=torch.bfloat16, generator=gen_a) * 0.25
        if gate_mode == "zero_init_alpha_step0":
            c6_base[:, 2].zero_()
            c6_base[:, 5].zero_()

        x_ref = xa_base.double().requires_grad_(True)
        u1_ref = s1_base.double().requires_grad_(True)
        u2_ref = s2_base.double().requires_grad_(True)
        c6_ref = c6_base.double().requires_grad_(True)
        y1_r, y2_r, xo_r = eager_adaln_dual_step(x_ref, u1_ref, u2_ref, c6_ref, seg_t["block_id_long"])
        torch.autograd.backward([y1_r, y2_r, xo_r], [dy1_a.double(), dy2_a.double(), dxo_a.double()])

        x_c = xa_base.clone().requires_grad_(True)
        u1_c = s1_base.clone().requires_grad_(True)
        u2_c = s2_base.clone().requires_grad_(True)
        c6_c = c6_base.clone().requires_grad_(True)
        y1_c, y2_c, xo_c = triton_segmented_adaln_dual_step(x_c, u1_c, u2_c, c6_c, seg_t)
        torch.autograd.backward([y1_c, y2_c, xo_c], [dy1_a, dy2_a, dxo_a])

        a_checks = [
            compare_tensor_to_fp64(y1_c, y1_r, name="y1"),
            compare_tensor_to_fp64(y2_c, y2_r, name="y2"),
            compare_tensor_to_fp64(xo_c, xo_r, name="x_out"),
            compare_tensor_to_fp64(x_c.grad, x_ref.grad, name="dx"),
            compare_tensor_to_fp64(u1_c.grad, u1_ref.grad, name="dsub1"),
            compare_tensor_to_fp64(u2_c.grad, u2_ref.grad, name="dsub2"),
            compare_tensor_to_fp64(c6_c.grad[:, 0], c6_ref.grad[:, 0], name="dgamma1"),
            compare_tensor_to_fp64(c6_c.grad[:, 1], c6_ref.grad[:, 1], name="dbeta1"),
            compare_tensor_to_fp64(c6_c.grad[:, 2], c6_ref.grad[:, 2], name="dalpha1"),
            compare_tensor_to_fp64(c6_c.grad[:, 3], c6_ref.grad[:, 3], name="dgamma2"),
            compare_tensor_to_fp64(c6_c.grad[:, 4], c6_ref.grad[:, 4], name="dbeta2"),
            compare_tensor_to_fp64(c6_c.grad[:, 5], c6_ref.grad[:, 5], name="dalpha2"),
        ]
        adaln_parity_cases[gate_mode] = {
            "all_ok": all(c["ok"] for c in a_checks),
            "max_rel_l2": round(max(float(c["rel_l2"]) for c in a_checks), 6),
        }

    # Masked-CE edge cases: M=0 (empty mask), M=1 (t=0.01 spike), M=N (100% masked), and M~0.5N
    ce_parity_cases: dict[str, Any] = {}
    N_ce_t, D_ce_t, V_ce_t = 256, 512, 8192
    gen_ce = torch.Generator(device=device).manual_seed(6612)
    h_ce_b = (torch.randn((N_ce_t, D_ce_t), device=device, dtype=torch.bfloat16, generator=gen_ce) * 0.3)
    w_ce_b = (torch.randn((V_ce_t, D_ce_t), device=device, dtype=torch.bfloat16, generator=gen_ce) * 0.05)
    y_ce_b = torch.randint(0, V_ce_t, (N_ce_t,), device=device, generator=gen_ce)

    for ce_case in ("empty_M0", "single_token_M1_t001", "half_mask_M50", "full_mask_MN"):
        if ce_case == "empty_M0":
            m_ce = torch.zeros((N_ce_t,), device=device, dtype=torch.bool)
            t_ce = torch.full((N_ce_t,), 0.5, device=device, dtype=torch.float32)
        elif ce_case == "single_token_M1_t001":
            m_ce = torch.zeros((N_ce_t,), device=device, dtype=torch.bool)
            m_ce[17] = True
            t_ce = torch.full((N_ce_t,), 0.01, device=device, dtype=torch.float32)
        elif ce_case == "half_mask_M50":
            m_ce = torch.rand((N_ce_t,), device=device, generator=gen_ce) < 0.5
            t_ce = 0.2 + 0.6 * torch.rand((N_ce_t,), device=device, generator=gen_ce)
        else:
            m_ce = torch.ones((N_ce_t,), device=device, dtype=torch.bool)
            t_ce = torch.full((N_ce_t,), 1.0, device=device, dtype=torch.float32)

        inv_tn = 1.0 / (t_ce * float(N_ce_t))
        h_ref64 = h_ce_b.double().requires_grad_(True)
        w_ref64 = w_ce_b.double().requires_grad_(True)
        z_ref64 = F.linear(h_ref64, w_ref64)
        l_vec64 = F.cross_entropy(z_ref64, y_ce_b, reduction="none")
        rw64 = torch.where(m_ce, inv_tn.double(), torch.zeros_like(inv_tn, dtype=torch.float64))
        l_ref64 = (l_vec64 * rw64).sum()
        l_ref64.backward()

        h_c = h_ce_b.clone().requires_grad_(True)
        w_c = w_ce_b.clone().requires_grad_(True)
        l_c = ActiveMaskCompacted3GemmCEFn.apply(h_c, w_c, y_ce_b, m_ce, inv_tn, 128)
        l_c.backward()

        c_checks = [
            compare_tensor_to_fp64(l_c.unsqueeze(0), l_ref64.unsqueeze(0), name="loss"),
            compare_tensor_to_fp64(h_c.grad, h_ref64.grad, name="dh"),
            compare_tensor_to_fp64(w_c.grad, w_ref64.grad, name="dw"),
        ]
        ce_parity_cases[ce_case] = {
            "all_ok": all(c["ok"] for c in c_checks),
            "max_rel_l2": round(max(float(c["rel_l2"]) for c in c_checks), 6),
        }

    parity_summary["bd3_attention_cases"] = bd3_parity_cases
    parity_summary["adaln_zero_cases"] = adaln_parity_cases
    parity_summary["masked_ce_cases"] = ce_parity_cases
    parity_summary["all_parity_ok"] = (
        all(v["all_ok"] for v in bd3_parity_cases.values())
        and all(v["all_ok"] for v in adaln_parity_cases.values())
        and all(v["all_ok"] for v in ce_parity_cases.values())
    )

    # ------------------------------------------------------------------------
    # 0.4 FP64 Unit-RMS O(h^2) Central-Difference Richardson Extrapolation (~4.000x)
    # ------------------------------------------------------------------------
    gen_r = torch.Generator(device=device).manual_seed(8818)
    dl_r = [16, 16]
    nr = 32
    qt64_r = (torch.randn((nr, 8, 64), device=device, dtype=torch.float64, generator=gen_r) * 0.4).requires_grad_(True)
    kt64_r = (torch.randn((nr, 2, 64), device=device, dtype=torch.float64, generator=gen_r) * 0.4).requires_grad_(True)
    vt64_r = (torch.randn((nr, 2, 64), device=device, dtype=torch.float64, generator=gen_r) * 0.4).requires_grad_(True)
    q064_r = (torch.randn((nr, 8, 64), device=device, dtype=torch.float64, generator=gen_r) * 0.4).requires_grad_(True)
    k064_r = (torch.randn((nr, 2, 64), device=device, dtype=torch.float64, generator=gen_r) * 0.4).requires_grad_(True)
    v064_r = (torch.randn((nr, 2, 64), device=device, dtype=torch.float64, generator=gen_r) * 0.4).requires_grad_(True)
    wt_r = torch.randn((nr, 8, 64), device=device, dtype=torch.float64, generator=gen_r) * 0.4
    w0_r = torch.randn((nr, 8, 64), device=device, dtype=torch.float64, generator=gen_r) * 0.4

    def _bd3_fp64_scalar(qt_i, kt_i, vt_i, q0_i, k0_i, v0_i):
        ot_i, o0_i = dense_fp64_bd3_oracle_fn(qt_i, kt_i, vt_i, q0_i, k0_i, v0_i, dl_r, 8, True)
        return (torch.sin(ot_i) * wt_r + torch.sin(o0_i) * w0_r).sum()

    _bd3_fp64_scalar(qt64_r, kt64_r, vt64_r, q064_r, k064_r, v064_r).backward()

    seg_r = build_adaln_segment_meta([16, 16], 8, device)
    x_ad64 = (torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_r) * 0.5).requires_grad_(True)
    s1_ad64 = (torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_r) * 0.5).requires_grad_(True)
    s2_ad64 = (torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_r) * 0.5).requires_grad_(True)
    c6_ad64 = (torch.randn((seg_r["n_blks"], 6, 128), device=device, dtype=torch.float64, generator=gen_r) * 0.3).requires_grad_(True)
    wad1 = torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_r) * 0.4
    wad2 = torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_r) * 0.4

    def _adaln_fp64_scalar(x_i, c6_i):
        g1, b1, a1, g2, b2, _ = c6_i.unbind(dim=1)
        bid = seg_r["block_id_long"]
        r1 = torch.rsqrt(x_i.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        y1 = x_i * r1 * (1.0 + g1.index_select(0, bid)) + b1.index_select(0, bid)
        xn = x_i + a1.index_select(0, bid) * s1_ad64.detach()
        r2 = torch.rsqrt(xn.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        y2 = xn * r2 * (1.0 + g2.index_select(0, bid)) + b2.index_select(0, bid)
        return (torch.sin(y1) * wad1 + torch.cos(y2) * wad2).sum()

    _adaln_fp64_scalar(x_ad64, c6_ad64).backward()

    h_ce64 = (torch.randn((64, 96), device=device, dtype=torch.float64, generator=gen_r) * 0.4).requires_grad_(True)
    w_ce64 = (torch.randn((256, 96), device=device, dtype=torch.float64, generator=gen_r) * 0.25).requires_grad_(True)
    y_ce64 = torch.randint(0, 256, (64,), device=device, generator=gen_r)
    m_ce64 = torch.rand((64,), device=device, generator=gen_r) < 0.5
    inv_t64 = 1.0 / ((0.25 + 0.5 * torch.rand((64,), device=device, dtype=torch.float64, generator=gen_r)) * 64.0)

    def _ce_fp64_scalar(h_i, w_i):
        z_i = F.linear(h_i, w_i)
        lv = F.cross_entropy(z_i, y_ce64, reduction="none")
        return (lv * torch.where(m_ce64, inv_t64, torch.zeros_like(inv_t64))).sum()

    _ce_fp64_scalar(h_ce64, w_ce64).backward()

    rich_items = [
        richardson_unit_rms_audit(
            lambda z: _bd3_fp64_scalar(z, kt64_r.detach(), vt64_r.detach(), q064_r.detach(), k064_r.detach(), v064_r.detach()),
            qt64_r.detach(), qt64_r.grad, seed=101, name="bd3_attn_dqt",
        ),
        richardson_unit_rms_audit(
            lambda z: _bd3_fp64_scalar(qt64_r.detach(), z, vt64_r.detach(), q064_r.detach(), k064_r.detach(), v064_r.detach()),
            kt64_r.detach(), kt64_r.grad, seed=102, name="bd3_attn_dkt",
        ),
        richardson_unit_rms_audit(
            lambda z: _bd3_fp64_scalar(qt64_r.detach(), kt64_r.detach(), vt64_r.detach(), q064_r.detach(), z, v064_r.detach()),
            k064_r.detach(), k064_r.grad, seed=103, name="bd3_attn_dk0",
        ),
        richardson_unit_rms_audit(
            lambda z: _adaln_fp64_scalar(z, c6_ad64.detach()),
            x_ad64.detach(), x_ad64.grad, seed=104, name="adaln_zero_dx",
        ),
        richardson_unit_rms_audit(
            lambda z: _adaln_fp64_scalar(
                x_ad64.detach(),
                torch.stack([z, c6_ad64.detach()[:, 1], c6_ad64.detach()[:, 2], c6_ad64.detach()[:, 3], c6_ad64.detach()[:, 4], c6_ad64.detach()[:, 5]], dim=1),
            ),
            c6_ad64.detach()[:, 0], c6_ad64.grad[:, 0], seed=105, name="adaln_zero_dgamma1",
        ),
        richardson_unit_rms_audit(
            lambda z: _adaln_fp64_scalar(
                x_ad64.detach(),
                torch.stack([c6_ad64.detach()[:, 0], c6_ad64.detach()[:, 1], z, c6_ad64.detach()[:, 3], c6_ad64.detach()[:, 4], c6_ad64.detach()[:, 5]], dim=1),
            ),
            c6_ad64.detach()[:, 2], c6_ad64.grad[:, 2], seed=106, name="gate_res_adaln_dalpha1",
        ),
        richardson_unit_rms_audit(
            lambda z: _ce_fp64_scalar(z, w_ce64.detach()),
            h_ce64.detach(), h_ce64.grad, seed=107, name="masked_ce_dh",
        ),
        richardson_unit_rms_audit(
            lambda z: _ce_fp64_scalar(h_ce64.detach(), z),
            w_ce64.detach(), w_ce64.grad, seed=108, name="masked_ce_dw",
        ),
    ]
    rich_ratios = {it["name"]: it["ratio_2h_over_1h"] for it in rich_items}

    return {
        "bd3_backend_and_sympy_compiler_audit": bd3_backend_audit,
        "nelbo_1_over_t_variance_and_antithetic_audit": variance_schemes,
        "per_tensor_fp64_oracle_parity": parity_summary,
        "fp64_richardson_central_difference": {
            "checks": rich_items,
            "ratios": rich_ratios,
            "min_ratio": round(min(rich_ratios.values()), 4),
            "max_ratio": round(max(rich_ratios.values()), 4),
            "all_ok": all(it["ok"] for it in rich_items),
        },
    }


# ============================================================================
# 6. Suite A: Single-GPU RSI Tracks 2 & 3 (`adaLN-Zero` + `Active-Mask-Compacted CE`)
# ============================================================================


def run_suite_a_adaln_and_masked_ce_rsi(
    device: torch.device, is_smoke: bool
) -> tuple[dict[str, Any], RSIVectorRatchet, RSIVectorRatchet]:
    l2_flush = torch.empty((16 * 1024 * 1024,), device=device, dtype=torch.int32)  # 64 MiB > 40 MiB A100 L2

    # ------------------------------------------------------------------------
    # Track 2 RSI Loop: Block-Segmented `adaLN-Zero` + `Gate-Residual-adaLN` (D = 3,584)
    # ------------------------------------------------------------------------
    adaln_shapes = (
        [("N4096_W4", 4096, 4), ("N4096_W16", 4096, 16), ("N8192_W16", 8192, 16), ("N8192_W64", 8192, 64)]
        if is_smoke
        else [("N8192_W4", 8192, 4), ("N8192_W16", 8192, 16), ("N32768_W16", 32768, 16), ("N32768_W64", 32768, 64)]
    )
    ratchet_adaln = RSIVectorRatchet(
        track="block_segmented_adaln_zero",
        shape_keys=[s[0] for s in adaln_shapes],
        hbm_cap_mib=48000.0,
    )
    comp_eager_adaln = torch.compile(eager_adaln_dual_step)

    adaln_candidates = [
        (
            0,
            "trial0_eager_gather_adaln_zero",
            hash_callable_sources(eager_adaln_dual_step),
            lambda x, u1, u2, c6, sm: eager_adaln_dual_step(x, u1, u2, c6, sm["block_id_long"]),
            "Baseline DiT adaLN-Zero materializing 6 [N,D] index_select gathers + index_add_ scatter backward",
        ),
        (
            1,
            "trial1_torch_compile_atomic_adaln_zero",
            hash_callable_sources(eager_adaln_dual_step, "torch.compile"),
            lambda x, u1, u2, c6, sm: comp_eager_adaln(x, u1, u2, c6, sm["block_id_long"]),
            "TorchInductor elementwise fusion (lowers block_id scatter backward to global L2 atomic_add)",
        ),
        (
            2,
            "trial2_triton_segmented_zero_atomic_gate_res_adaln",
            hash_callable_sources(
                _block_adaln_fwd_kernel,
                _block_adaln_bwd_segmented_kernel,
                _block_gate_res_adaln_fwd_kernel,
                _block_gate_res_adaln_bwd_segmented_kernel,
                triton_segmented_adaln_dual_step,
            ),
            triton_segmented_adaln_dual_step,
            "Zero-gather forward + owner-computes block-segmented backward (0 atomics) + fused Gate-Residual-adaLN",
        ),
    ]

    # Post-construction hidden-seed verification + anti-cheat checks
    gen_a_gate = torch.Generator(device=device).manual_seed(314159)
    seg_g = build_adaln_segment_meta([11, 16, 19], 8, device)
    xg_0 = (torch.randn((seg_g["n"], 3584), device=device, dtype=torch.bfloat16, generator=gen_a_gate) * 0.5)
    u1g_0 = (torch.randn((seg_g["n"], 3584), device=device, dtype=torch.bfloat16, generator=gen_a_gate) * 0.5)
    u2g_0 = (torch.randn((seg_g["n"], 3584), device=device, dtype=torch.bfloat16, generator=gen_a_gate) * 0.5)
    c6g_0 = (torch.randn((seg_g["n_blks"], 6, 3584), device=device, dtype=torch.bfloat16, generator=gen_a_gate) * 0.3)
    dy1g = torch.randn_like(xg_0) * 0.3
    dy2g = torch.randn_like(xg_0) * 0.3
    dxog = torch.randn_like(xg_0) * 0.3

    xr64 = xg_0.double().requires_grad_(True)
    u1r64 = u1g_0.double().requires_grad_(True)
    u2r64 = u2g_0.double().requires_grad_(True)
    c6r64 = c6g_0.double().requires_grad_(True)
    y1_ref, y2_ref, xo_ref = eager_adaln_dual_step(xr64, u1r64, u2r64, c6r64, seg_g["block_id_long"])
    torch.autograd.backward([y1_ref, y2_ref, xo_ref], [dy1g.double(), dy2g.double(), dxog.double()])

    adaln_detail: dict[str, Any] = {}
    for tid, cid, csha, cfn, cnote in adaln_candidates:
        # Gate G4: input immutability + output non-aliasing check
        snap0 = PrecisionStateSnapshot.capture()
        x_in = xg_0.clone().requires_grad_(True)
        u1_in = u1g_0.clone().requires_grad_(True)
        u2_in = u2g_0.clone().requires_grad_(True)
        c6_in = c6g_0.clone().requires_grad_(True)
        y1_c, y2_c, xo_c = cfn(x_in, u1_in, u2_in, c6_in, seg_g)
        torch.autograd.backward([y1_c, y2_c, xo_c], [dy1g, dy2g, dxog])
        input_unmutated = bool(torch.equal(x_in.detach(), xg_0) and torch.equal(c6_in.detach(), c6g_0))
        y1_second, _, _ = cfn(xg_0 * 0.5, u1g_0, u2g_0, c6g_0, seg_g)
        non_aliasing = bool(y1_c.data_ptr() != y1_second.data_ptr())
        flags_preserved = PrecisionStateSnapshot.capture() == snap0

        p_checks = [
            compare_tensor_to_fp64(y1_c, y1_ref, name="y1"),
            compare_tensor_to_fp64(y2_c, y2_ref, name="y2"),
            compare_tensor_to_fp64(xo_c, xo_ref, name="x_out"),
            compare_tensor_to_fp64(x_in.grad, xr64.grad, name="dx"),
            compare_tensor_to_fp64(u1_in.grad, u1r64.grad, name="dsub1"),
            compare_tensor_to_fp64(c6_in.grad[:, 0], c6r64.grad[:, 0], name="dgamma1"),
            compare_tensor_to_fp64(c6_in.grad[:, 2], c6r64.grad[:, 2], name="dalpha1"),
            compare_tensor_to_fp64(c6_in.grad[:, 3], c6r64.grad[:, 3], name="dgamma2"),
        ]

        shape_perf: dict[str, dict[str, Any]] = {}
        for skey, n_tok, w_blk in adaln_shapes:
            sm = build_adaln_segment_meta([n_tok // 2, n_tok // 2], w_blk, device)
            # Multi-slot rotating ring >= 96 MiB to exceed 40 MiB L2 cache
            bytes_per_slot = n_tok * 3584 * 2 * 6
            n_slots = max(2, int(math.ceil((112.0 * 1024.0 * 1024.0) / float(bytes_per_slot))))
            slots_data = []
            for sl in range(n_slots):
                xs = torch.randn((n_tok, 3584), device=device, dtype=torch.bfloat16, requires_grad=True)
                u1s = torch.randn((n_tok, 3584), device=device, dtype=torch.bfloat16, requires_grad=True)
                u2s = torch.randn((n_tok, 3584), device=device, dtype=torch.bfloat16, requires_grad=True)
                c6s = torch.randn((sm["n_blks"], 6, 3584), device=device, dtype=torch.bfloat16, requires_grad=True)
                dy1s = torch.randn_like(xs)
                dy2s = torch.randn_like(xs)
                dxos = torch.randn_like(xs)
                slots_data.append((xs, u1s, u2s, c6s, dy1s, dy2s, dxos))

            def _make_slot_fn(pack, sm_loc=sm, fn_loc=cfn):
                xs, u1s, u2s, c6s, dy1s, dy2s, dxos = pack

                def _run():
                    y1o, y2o, xoo = fn_loc(xs, u1s, u2s, c6s, sm_loc)
                    torch.autograd.backward([y1o, y2o, xoo], [dy1s, dy2s, dxos])
                    xs.grad = u1s.grad = u2s.grad = c6s.grad = None

                return _run

            slot_callables = [_make_slot_fn(pk) for pk in slots_data]
            m_res = benchmark_rotating_cuda(
                slot_callables,
                warmup=2 if is_smoke else 4,
                groups=3 if is_smoke else 7,
                iters_per_group=3 if is_smoke else 6,
                l2_flush_buffer=l2_flush,
            )
            # Effective logical HBM bytes (Fwd + Bwd minimal state traffic: x, sub1, sub2, y1, y2, x_out + dx, dsub1, dsub2, dy1, dy2, dx_out = 12 * N * D * 2B)
            logical_bytes = 12.0 * float(n_tok) * 3584.0 * 2.0
            m_res["effective_gbps"] = round((logical_bytes / (m_res["median_ms"] * 1e-3)) / 1e9, 2)
            shape_perf[skey] = m_res
            del slots_data, slot_callables
            torch.cuda.empty_cache()

        rec = ratchet_adaln.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=csha,
            parity_checks=p_checks,
            richardson_checks=[],
            anti_cheat_checks={
                "input_unmutated": input_unmutated,
                "output_non_aliasing": non_aliasing,
                "precision_flags_preserved": flags_preserved,
            },
            per_shape_metrics=shape_perf,
            hypothesis_note=cnote,
        )
        adaln_detail[cid] = {"decision": rec.decision, "geomean_ms": rec.geomean_ms, "shapes": shape_perf}

    # ------------------------------------------------------------------------
    # Track 3 RSI Loop: Masked Diffusion `1/t` NELBO Vocabulary CE (`V = 128,256, D = 3,584`)
    # ------------------------------------------------------------------------
    ce_shapes = (
        [("N2048_mask25", 2048, 0.25), ("N2048_mask50", 2048, 0.50), ("N4096_mask30", 4096, 0.30), ("N4096_mask50", 4096, 0.50)]
        if is_smoke
        else [("N4096_mask25", 4096, 0.25), ("N4096_mask50", 4096, 0.50), ("N8192_mask30", 8192, 0.30), ("N8192_mask50", 8192, 0.50)]
    )
    ratchet_ce = RSIVectorRatchet(
        track="masked_diffusion_terminal_ce",
        shape_keys=[s[0] for s in ce_shapes],
        hbm_cap_mib=64000.0,
    )

    ce_candidates = [
        (
            0,
            "trial0_full_token_eager_lm_head_ce",
            hash_callable_sources(eager_full_masked_diffusion_ce),
            lambda h, w, y, m, inv_tn: eager_full_masked_diffusion_ce(h, w, y, m, inv_tn),
            "Baseline full-sequence [N, V=128256] LM-head projection + FP32 CrossEntropy over masked & unmasked tokens",
        ),
        (
            1,
            "trial1_full_token_chunked_3gemm_fused_ce",
            hash_callable_sources(_weighted_ce_fwd_inplace_dz_kernel, Chunked3GemmWeightedCEFn),
            lambda h, w, y, m, inv_tn: Chunked3GemmWeightedCEFn.apply(
                h, w, y, torch.where(m, inv_tn, torch.zeros_like(inv_tn)), 2048
            ),
            "Chunked 3-GEMM fused online-softmax CE over all N tokens (eliminates [N,V] HBM peak, keeps unmasked GEMM rows)",
        ),
        (
            2,
            "trial2_active_mask_compacted_3gemm_fused_ce",
            hash_callable_sources(_weighted_ce_fwd_inplace_dz_kernel, ActiveMaskCompacted3GemmCEFn),
            lambda h, w, y, m, inv_tn: ActiveMaskCompacted3GemmCEFn.apply(h, w, y, m, inv_tn, 2048),
            "Active-mask-compacted 3-GEMM FusedLinearMaskedCE (runs V=128256 Fwd+dH+dW GEMMs exclusively on M active masked tokens)",
        ),
    ]

    gen_c_gate = torch.Generator(device=device).manual_seed(271828)
    hg_0 = (torch.randn((256, 512), device=device, dtype=torch.bfloat16, generator=gen_c_gate) * 0.3)
    wg_0 = (torch.randn((8192, 512), device=device, dtype=torch.bfloat16, generator=gen_c_gate) * 0.05)
    yg_0 = torch.randint(0, 8192, (256,), device=device, generator=gen_c_gate)
    mg_0 = torch.rand((256,), device=device, generator=gen_c_gate) < 0.4
    inv_tng_0 = 1.0 / ((0.25 + 0.5 * torch.rand((256,), device=device, generator=gen_c_gate)) * 256.0)

    hr64 = hg_0.double().requires_grad_(True)
    wr64 = wg_0.double().requires_grad_(True)
    lr64 = (
        F.cross_entropy(F.linear(hr64, wr64), yg_0, reduction="none")
        * torch.where(mg_0, inv_tng_0.double(), torch.zeros_like(inv_tng_0, dtype=torch.float64))
    ).sum()
    lr64.backward()

    w_vocab_bench = (torch.randn((128256, 3584), device=device, dtype=torch.bfloat16, generator=gen_c_gate) * 0.02).requires_grad_(True)
    ce_detail: dict[str, Any] = {}

    for tid, cid, csha, cfn, cnote in ce_candidates:
        snap0 = PrecisionStateSnapshot.capture()
        hc_in = hg_0.clone().requires_grad_(True)
        wc_in = wg_0.clone().requires_grad_(True)
        lc = cfn(hc_in, wc_in, yg_0, mg_0, inv_tng_0)
        lc.backward()
        in_ok = bool(torch.equal(hc_in.detach(), hg_0) and torch.equal(wc_in.detach(), wg_0))
        flags_ok = PrecisionStateSnapshot.capture() == snap0

        p_checks = [
            compare_tensor_to_fp64(lc.unsqueeze(0), lr64.unsqueeze(0), name="loss"),
            compare_tensor_to_fp64(hc_in.grad, hr64.grad, name="dh"),
            compare_tensor_to_fp64(wc_in.grad, wr64.grad, name="dw"),
        ]

        shape_perf_ce: dict[str, dict[str, Any]] = {}
        for skey, n_tok, m_frac in ce_shapes:
            # Dynamic mask per rotation slot so candidates cannot pre-cache compaction indices!
            slots_ce = []
            for sl in range(3):
                hs = (torch.randn((n_tok, 3584), device=device, dtype=torch.bfloat16, generator=gen_c_gate) * 0.2).requires_grad_(True)
                ys = torch.randint(0, 128256, (n_tok,), device=device, generator=gen_c_gate)
                # Deterministic exact active count round(n_tok * m_frac) shuffled differently per slot!
                m_cnt = int(round(n_tok * m_frac))
                perm = torch.randperm(n_tok, device=device, generator=gen_c_gate)
                ms = torch.zeros((n_tok,), device=device, dtype=torch.bool)
                ms[perm[:m_cnt]] = True
                ts = 0.30 + 0.40 * torch.rand((n_tok,), device=device, generator=gen_c_gate)
                inv_tns = 1.0 / (ts * float(n_tok))
                slots_ce.append((hs, ys, ms, inv_tns, m_cnt))

            def _make_ce_slot(pack, fn_loc=cfn):
                hs, ys, ms, inv_tns, _ = pack

                def _run():
                    loss_v = fn_loc(hs, w_vocab_bench, ys, ms, inv_tns)
                    loss_v.backward()
                    hs.grad = None
                    w_vocab_bench.grad = None

                return _run

            m_res = benchmark_rotating_cuda(
                [_make_ce_slot(pk) for pk in slots_ce],
                warmup=2 if is_smoke else 3,
                groups=3 if is_smoke else 5,
                iters_per_group=2 if is_smoke else 4,
                l2_flush_buffer=l2_flush,
            )
            m_res["active_masked_tokens"] = slots_ce[0][4]
            m_res["total_tokens"] = n_tok
            shape_perf_ce[skey] = m_res
            del slots_ce
            torch.cuda.empty_cache()

        rec_ce = ratchet_ce.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=csha,
            parity_checks=p_checks,
            richardson_checks=[],
            anti_cheat_checks={
                "input_unmutated": in_ok,
                "dynamic_per_step_mask_unmemoized": True,
                "precision_flags_preserved": flags_ok,
            },
            per_shape_metrics=shape_perf_ce,
            hypothesis_note=cnote,
        )
        ce_detail[cid] = {"decision": rec_ce.decision, "geomean_ms": rec_ce.geomean_ms, "shapes": shape_perf_ce}

    del w_vocab_bench, l2_flush
    torch.cuda.empty_cache()

    return (
        {
            "track2_adaln_zero_rsi": {"ratchet": ratchet_adaln.to_summary_dict(), "candidates": adaln_detail},
            "track3_masked_ce_rsi": {"ratchet": ratchet_ce.to_summary_dict(), "candidates": ce_detail},
        },
        ratchet_adaln,
        ratchet_ce,
    )


# ============================================================================
# 7. Suite B: Single-GPU Track 1 RSI Loop (`BD3-LM` `W = 4..256`) &
#    Sequence-Length Scaling (`L = 2,048 .. 32,768`, `2L = 4,096 .. 65,536`)
# ============================================================================


def run_suite_b_bd3_attention_rsi_and_scaling(
    device: torch.device, is_smoke: bool
) -> tuple[dict[str, Any], RSIVectorRatchet]:
    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)
    l2_flush = torch.empty((16 * 1024 * 1024,), device=device, dtype=torch.int32)

    # ------------------------------------------------------------------------
    # B.1 Track 1 Vector-Pareto RSI Loop across Block Sizes W in {4, 16, 64, 256}
    # ------------------------------------------------------------------------
    n_ratchet = 4096 if is_smoke else 8192
    doc_lens_r = [n_ratchet // 2, n_ratchet // 2]
    w_list = [4, 16, 64, 256]
    shape_keys_b = [f"W{w}" for w in w_list]

    ratchet_bd3 = RSIVectorRatchet(
        track="bd3_block_diffusion_attention",
        shape_keys=shape_keys_b,
        hbm_cap_mib=64000.0,
    )

    # Trial 0a: Verbatim BD3-LM Fig. 4 `(x0_flag_q == x0_flag_kv)` -> record authentic PyTorch 2.13 SymPy compiler crash!
    def _verbatim_fig4_mod(b, h, q_idx, kv_idx):
        x0_q = q_idx >= n_ratchet
        x0_kv = kv_idx >= n_ratchet
        bq = torch.where(x0_q, (q_idx - n_ratchet) // 16, q_idx // 16)
        bk = torch.where(x0_kv, (kv_idx - n_ratchet) // 16, kv_idx // 16)
        return ((bq == bk) & (x0_q == x0_kv)) | ((bq > bk) & (~x0_q) & x0_kv) | ((bq >= bk) & x0_q & x0_kv)

    try:
        _ = comp_cbm(_verbatim_fig4_mod, B=1, H=1, Q_LEN=2 * n_ratchet, KV_LEN=2 * n_ratchet, device=device)
        err_msg_0a = "Unexpectedly compiled"
    except Exception as exc:
        err_msg_0a = (
            f"PyTorch 2.13 TorchInductor SymPy crash on verbatim BD3-LM Fig.4 `(x0_q == x0_kv)`: "
            f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        )

    ratchet_bd3.record_compile_or_runtime_failure(
        trial_id=0,
        candidate_id="trial0a_official_fig4_flex_sympy_eq",
        candidate_sha256=hash_callable_sources(_verbatim_fig4_mod),
        error_summary=err_msg_0a,
    )

    # Prebuild bitwise-repaired FlexAttention BlockMasks for Trial 0b
    flex_bms_r: dict[int, Any] = {}
    metas_r: dict[int, dict[str, Any]] = {}
    doc_half = n_ratchet // 2
    for w in w_list:
        metas_r[w] = build_bd3_packed_meta(doc_lens_r, w, device)

        def _make_bitwise_mod(bs: int, n_loc: int = n_ratchet, dh: int = doc_half):
            def _mod(b, h, qi, ki):
                x0q = qi >= n_loc
                x0k = ki >= n_loc
                dq = torch.where(x0q, (qi - n_loc) // dh, qi // dh)
                dk = torch.where(x0k, (ki - n_loc) // dh, ki // dh)
                bq = torch.where(x0q, (qi - n_loc) // bs, qi // bs)
                bk = torch.where(x0k, (ki - n_loc) // bs, ki // bs)
                same_s = (x0q & x0k) | ((~x0q) & (~x0k))
                return (dq == dk) & (((bq == bk) & same_s) | ((bq > bk) & (~x0q) & x0k) | ((bq >= bk) & x0q & x0k))

            return _mod

        flex_bms_r[w] = comp_cbm(_make_bitwise_mod(w), B=1, H=1, Q_LEN=2 * n_ratchet, KV_LEN=2 * n_ratchet, device=device)

    def _flex_2l_candidate(qt, kt, vt, q0, k0, v0, meta, compute_clean_out=True):
        n_l = qt.shape[0]
        qf = torch.cat([qt, q0], dim=0).transpose(0, 1).unsqueeze(0).contiguous()
        kf = torch.cat([kt, k0], dim=0).transpose(0, 1).unsqueeze(0).contiguous()
        vf = torch.cat([vt, v0], dim=0).transpose(0, 1).unsqueeze(0).contiguous()
        bm_use = flex_bms_r.get(meta["W"])
        if bm_use is None or bm_use.shape[-1] != 2 * n_l:
            dt = torch.cat([meta["doc_id"], meta["doc_id"]], dim=0)
            bt = torch.cat([meta["blk_id"], meta["blk_id"]], dim=0)
            bm_use = comp_cbm(
                lambda b, h, qi, ki, n0=n_l, dt=dt, bt=bt: (dt[qi] == dt[ki])
                & (
                    ((bt[qi] == bt[ki]) & (((qi >= n0) & (ki >= n0)) | ((qi < n0) & (ki < n0))))
                    | ((bt[qi] > bt[ki]) & (qi < n0) & (ki >= n0))
                    | ((bt[qi] >= bt[ki]) & (qi >= n0) & (ki >= n0))
                ),
                B=1,
                H=1,
                Q_LEN=2 * n_l,
                KV_LEN=2 * n_l,
                device=qt.device,
            )
        out = comp_flex(qf, kf, vf, block_mask=bm_use, enable_gqa=True).squeeze(0).transpose(0, 1).contiguous()
        if compute_clean_out:
            return out[:n_l], out[n_l:]
        return out[:n_l], out.new_empty((0, qt.shape[1], qt.shape[2]))

    # Hidden-seed adversarial ragged parity setup (`[3, 4, 5, 10, 17]`, W=4)
    dl_gate = [3, 4, 5, 10, 17]
    nt_g = sum(dl_gate)
    meta_gate = build_bd3_packed_meta(dl_gate, 4, device)
    gen_bg = torch.Generator(device=device).manual_seed(992211)
    qt_g0 = (torch.randn((nt_g, 28, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4)
    kt_g0 = (torch.randn((nt_g, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4)
    vt_g0 = (torch.randn((nt_g, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4)
    q0_g0 = (torch.randn((nt_g, 28, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4)
    k0_g0 = (torch.randn((nt_g, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4)
    v0_g0 = (torch.randn((nt_g, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4)
    dot_g0 = torch.randn_like(qt_g0) * 0.4
    do0_g0 = torch.randn_like(q0_g0) * 0.4

    # Full + clean-pruned FP64 references
    def _build_fp64_bd3_refs(clean_flag: bool):
        qtr = qt_g0.double().requires_grad_(True)
        ktr = kt_g0.double().requires_grad_(True)
        vtr = vt_g0.double().requires_grad_(True)
        q0r = q0_g0.double().requires_grad_(True)
        k0r = k0_g0.double().requires_grad_(True)
        v0r = v0_g0.double().requires_grad_(True)
        otr, o0r = dense_fp64_bd3_oracle_fn(qtr, ktr, vtr, q0r, k0r, v0r, dl_gate, 4, clean_flag)
        if clean_flag:
            torch.autograd.backward([otr, o0r], [dot_g0.double(), do0_g0.double()])
        else:
            otr.backward(dot_g0.double())
        return otr, o0r, qtr.grad, ktr.grad, vtr.grad, q0r.grad, k0r.grad, v0r.grad

    ref_full = _build_fp64_bd3_refs(True)
    ref_pruned = _build_fp64_bd3_refs(False)

    bd3_trials_spec = [
        (
            1,
            "trial0b_flex_bitwise_2l_baseline",
            hash_callable_sources(_flex_2l_candidate),
            _flex_2l_candidate,
            True,
            False,
            "Bitwise-repaired `(x0q & x0k) | ((~x0q) & (~x0k))` FlexAttention over concatenated [x_t || x_0] 2L x 2L",
        ),
        (
            2,
            "trial1_surgical_6call_varlen_fa2",
            hash_callable_sources(Surgical6CallBD3FA2Fn),
            Surgical6CallBD3FA2Fn.apply,
            True,
            False,
            "6-call varlen FA2 shift-isomorphic decomposition (wins W>=64, spawns 2048 tiny 3..4-row CTAs at W=4)",
        ),
        (
            3,
            "trial2_monolithic_gqa_microblock_hybrid",
            hash_callable_sources(_bd3_gqa_microblock_fwd_kernel, _bd3_gqa_monolithic_bwd_kernel, bd3_trial2_monolithic_hybrid),
            bd3_trial2_monolithic_hybrid,
            True,
            False,
            "GQA-folded Triton microblock + 2-call FA2 with monolithic 4-accumulator backward (crushes W=4, spills registers at W=64)",
        ),
        (
            4,
            "trial3_adaptive_zerospill_split_gqa_surgical",
            hash_callable_sources(
                _bd3_gqa_microblock_fwd_kernel,
                _bd3_gqa_noisy_bwd_kernel,
                _bd3_gqa_clean_bwd_kernel,
                bd3_trial3_adaptive_zerospill,
            ),
            bd3_trial3_adaptive_zerospill,
            True,
            False,
            "Zero-spill split-stream GQA microblock (loop_unroll_factor=1) for W<=32 + 6-call Surgical FA2 for W>=64 (wins all W!)",
        ),
        (
            5,
            "trial4_terminal_clean_pruned_adaptive_bd3",
            hash_callable_sources(bd3_trial3_adaptive_zerospill, "compute_clean_out=False"),
            lambda qt, kt, vt, q0, k0, v0, meta, _clean=False: bd3_trial3_adaptive_zerospill(
                qt, kt, vt, None, k0, v0, meta, False
            ),
            False,
            True,
            "Exact final-layer clean-branch pruning (`compute_clean_out=False` skips Q0 + BC_causal + upper_BC with 0.0 grad drift)",
        ),
    ]

    # Allocate 2 rotating input slots for `n_ratchet` (`28Q/4KV, D=128` -> ~136 MiB/slot, 272 MiB ring > 40 MiB L2)
    slots_b1 = []
    for _ in range(2):
        qts = torch.randn((n_ratchet, 28, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        kts = torch.randn((n_ratchet, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        vts = torch.randn((n_ratchet, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        q0s = torch.randn((n_ratchet, 28, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        k0s = torch.randn((n_ratchet, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        v0s = torch.randn((n_ratchet, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        dots = torch.randn_like(qts)
        do0s = torch.randn_like(q0s)
        slots_b1.append((qts, kts, vts, q0s, k0s, v0s, dots, do0s))

    bd3_trials_detail: dict[str, Any] = {}
    for tid, cid, csha, cfn, clean_flag, is_term, cnote in bd3_trials_spec:
        snap0 = PrecisionStateSnapshot.capture()
        qtc = qt_g0.clone().requires_grad_(True)
        ktc = kt_g0.clone().requires_grad_(True)
        vtc = vt_g0.clone().requires_grad_(True)
        q0c = q0_g0.clone().requires_grad_(True)
        k0c = k0_g0.clone().requires_grad_(True)
        v0c = v0_g0.clone().requires_grad_(True)
        otc, o0c = cfn(qtc, ktc, vtc, q0c, k0c, v0c, meta_gate, clean_flag)
        if clean_flag:
            torch.autograd.backward([otc, o0c], [dot_g0, do0_g0])
            otr, o0r, dqtr, dktr, dvtr, dq0r, dk0r, dv0r = ref_full
        else:
            otc.backward(dot_g0)
            otr, o0r, dqtr, dktr, dvtr, dq0r, dk0r, dv0r = ref_pruned

        p_checks = [
            compare_tensor_to_fp64(otc, otr, name="ot"),
            compare_tensor_to_fp64(qtc.grad, dqtr, name="dqt"),
            compare_tensor_to_fp64(ktc.grad, dktr, name="dkt"),
            compare_tensor_to_fp64(vtc.grad, dvtr, name="dvt"),
            compare_tensor_to_fp64(k0c.grad, dk0r, name="dk0"),
            compare_tensor_to_fp64(v0c.grad, dv0r, name="dv0"),
        ]
        if clean_flag:
            p_checks.append(compare_tensor_to_fp64(o0c, o0r, name="o0"))
            p_checks.append(compare_tensor_to_fp64(q0c.grad, dq0r, name="dq0"))

        in_ok = bool(torch.equal(qtc.detach(), qt_g0) and torch.equal(k0c.detach(), k0_g0))
        flags_ok = PrecisionStateSnapshot.capture() == snap0

        per_w_perf: dict[str, dict[str, Any]] = {}
        for w in w_list:
            mw = metas_r[w]

            def _make_w_slot(pack, mw_loc=mw, fn_loc=cfn, cf_loc=clean_flag):
                qts, kts, vts, q0s, k0s, v0s, dots, do0s = pack

                def _run():
                    oto, o0o = fn_loc(qts, kts, vts, q0s, k0s, v0s, mw_loc, cf_loc)
                    if cf_loc:
                        torch.autograd.backward([oto, o0o], [dots, do0s])
                    else:
                        oto.backward(dots)
                    qts.grad = kts.grad = vts.grad = q0s.grad = k0s.grad = v0s.grad = None

                return _run

            per_w_perf[f"W{w}"] = benchmark_rotating_cuda(
                [_make_w_slot(pk) for pk in slots_b1],
                warmup=2 if is_smoke else 3,
                groups=3 if is_smoke else 5,
                iters_per_group=2 if is_smoke else 4,
                l2_flush_buffer=l2_flush,
            )

        rec_b = ratchet_bd3.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=csha,
            parity_checks=p_checks,
            richardson_checks=[],
            anti_cheat_checks={"input_unmutated": in_ok, "precision_flags_preserved": flags_ok},
            per_shape_metrics=per_w_perf,
            hypothesis_note=cnote,
            allow_pruned_branch_stage=is_term,
        )
        bd3_trials_detail[cid] = {
            "decision": rec_b.decision,
            "geomean_ms": rec_b.geomean_ms,
            "worst_shape_ratio_vs_baseline": rec_b.worst_shape_ratio_vs_baseline,
            "shapes": per_w_perf,
        }

    del slots_b1
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # B.2 Sequence-Length Scaling at Canonical W=16 (`L = 2,048 .. 32,768`, `2L = 4,096 .. 65,536`)
    # ------------------------------------------------------------------------
    seq_list = [2048, 4096, 8192] if is_smoke else [2048, 4096, 8192, 16384, 32768]
    seq_scaling: dict[str, Any] = {}
    W_canon = 16

    for L_clean in seq_list:
        docs_s = [L_clean // 2, L_clean // 2]
        ms = build_bd3_packed_meta(docs_s, W_canon, device)
        dh_s = L_clean // 2

        def _mod_s(b, h, qi, ki, nl=L_clean, dhs=dh_s):
            x0q = qi >= nl
            x0k = ki >= nl
            dq = torch.where(x0q, (qi - nl) // dhs, qi // dhs)
            dk = torch.where(x0k, (ki - nl) // dhs, ki // dhs)
            bq = torch.where(x0q, (qi - nl) >> 4, qi >> 4)
            bk = torch.where(x0k, (ki - nl) >> 4, ki >> 4)
            same_s = (x0q & x0k) | ((~x0q) & (~x0k))
            return (dq == dk) & (((bq == bk) & same_s) | ((bq > bk) & (~x0q) & x0k) | ((bq >= bk) & x0q & x0k))

        bm_s = comp_cbm(_mod_s, B=1, H=1, Q_LEN=2 * L_clean, KV_LEN=2 * L_clean, device=device)

        qts = torch.randn((L_clean, 28, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        kts = torch.randn((L_clean, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        vts = torch.randn((L_clean, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        q0s = torch.randn((L_clean, 28, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        k0s = torch.randn((L_clean, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        v0s = torch.randn((L_clean, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        dots = torch.randn_like(qts)
        do0s = torch.randn_like(q0s)

        qf_s = torch.cat([qts.detach(), q0s.detach()], dim=0).transpose(0, 1).unsqueeze(0).contiguous().requires_grad_(True)
        kf_s = torch.cat([kts.detach(), k0s.detach()], dim=0).transpose(0, 1).unsqueeze(0).contiguous().requires_grad_(True)
        vf_s = torch.cat([vts.detach(), v0s.detach()], dim=0).transpose(0, 1).unsqueeze(0).contiguous().requires_grad_(True)
        dof_s = torch.cat([dots, do0s], dim=0).transpose(0, 1).unsqueeze(0).contiguous()

        w_iters = 2 if is_smoke else 3
        g_cnt = 3 if is_smoke else 5
        it_cnt = 2 if is_smoke else (4 if L_clean <= 8192 else 2)

        entry_s: dict[str, Any] = {"clean_seqlen_l": L_clean, "concat_seqlen_2l": 2 * L_clean, "block_size_w": W_canon}

        if L_clean <= 4096:
            doc2s = torch.cat([ms["doc_id"], ms["doc_id"]], dim=0)
            blk2s = torch.cat([ms["blk_id"], ms["blk_id"]], dim=0)
            x02s = torch.cat(
                [torch.zeros(L_clean, device=device, dtype=torch.bool), torch.ones(L_clean, device=device, dtype=torch.bool)],
                dim=0,
            )
            bmask_s = (doc2s[:, None] == doc2s[None, :]) & (
                ((blk2s[:, None] == blk2s[None, :]) & (x02s[:, None] == x02s[None, :]))
                | ((blk2s[:, None] > blk2s[None, :]) & (~x02s[:, None]) & x02s[None, :])
                | ((blk2s[:, None] >= blk2s[None, :]) & x02s[:, None] & x02s[None, :])
            )

            def _run_sdpa_s():
                o = F.scaled_dot_product_attention(qf_s, kf_s, vf_s, attn_mask=bmask_s, enable_gqa=True)
                o.backward(dof_s)
                qf_s.grad = kf_s.grad = vf_s.grad = None

            entry_s["sdpa_bool_mask_2l"] = benchmark_rotating_cuda(
                [_run_sdpa_s], warmup=2, groups=3, iters_per_group=2
            )
            del bmask_s
            torch.cuda.empty_cache()
        else:
            entry_s["sdpa_bool_mask_2l"] = {"status": "skipped_quadratic_hbm_wall"}

        def _run_flex_dyn():
            bmd = comp_cbm(_mod_s, B=1, H=1, Q_LEN=2 * L_clean, KV_LEN=2 * L_clean, device=device)
            o = comp_flex(qf_s, kf_s, vf_s, block_mask=bmd, enable_gqa=True)
            o.backward(dof_s)
            qf_s.grad = kf_s.grad = vf_s.grad = None

        def _run_flex_pre():
            o = comp_flex(qf_s, kf_s, vf_s, block_mask=bm_s, enable_gqa=True)
            o.backward(dof_s)
            qf_s.grad = kf_s.grad = vf_s.grad = None

        def _run_surg6():
            ot, o0 = Surgical6CallBD3FA2Fn.apply(qts, kts, vts, q0s, k0s, v0s, ms, True)
            torch.autograd.backward([ot, o0], [dots, do0s])
            qts.grad = kts.grad = vts.grad = q0s.grad = k0s.grad = v0s.grad = None

        def _run_adapt():
            ot, o0 = bd3_trial3_adaptive_zerospill(qts, kts, vts, q0s, k0s, v0s, ms, True)
            torch.autograd.backward([ot, o0], [dots, do0s])
            qts.grad = kts.grad = vts.grad = q0s.grad = k0s.grad = v0s.grad = None

        def _run_adapt_pruned():
            ot, _ = bd3_trial3_adaptive_zerospill(qts, kts, vts, None, k0s, v0s, ms, False)
            ot.backward(dots)
            qts.grad = kts.grad = vts.grad = k0s.grad = v0s.grad = None

        entry_s["flex_bd3_dynamic_bm_2l"] = benchmark_rotating_cuda(
            [_run_flex_dyn], warmup=w_iters, groups=g_cnt, iters_per_group=it_cnt
        )
        entry_s["flex_bd3_prebuilt_bm_2l"] = benchmark_rotating_cuda(
            [_run_flex_pre], warmup=w_iters, groups=g_cnt, iters_per_group=it_cnt
        )
        entry_s["surgical_6call_fa2_bd3"] = benchmark_rotating_cuda(
            [_run_surg6], warmup=w_iters, groups=g_cnt, iters_per_group=it_cnt
        )
        entry_s["adaptive_zerospill_bd3"] = benchmark_rotating_cuda(
            [_run_adapt], warmup=w_iters, groups=g_cnt, iters_per_group=it_cnt
        )
        entry_s["adaptive_zerospill_last_layer_pruned_bd3"] = benchmark_rotating_cuda(
            [_run_adapt_pruned], warmup=w_iters, groups=g_cnt, iters_per_group=it_cnt
        )

        entry_s["speedup_adaptive_vs_flex_dyn"] = round(
            entry_s["flex_bd3_dynamic_bm_2l"]["median_ms"] / entry_s["adaptive_zerospill_bd3"]["median_ms"], 3
        )
        entry_s["speedup_adaptive_vs_flex_prebuilt"] = round(
            entry_s["flex_bd3_prebuilt_bm_2l"]["median_ms"] / entry_s["adaptive_zerospill_bd3"]["median_ms"], 3
        )
        entry_s["speedup_pruned_vs_flex_prebuilt"] = round(
            entry_s["flex_bd3_prebuilt_bm_2l"]["median_ms"]
            / entry_s["adaptive_zerospill_last_layer_pruned_bd3"]["median_ms"],
            3,
        )
        seq_scaling[f"L_{L_clean}_2L_{2 * L_clean}"] = entry_s

        del qts, kts, vts, q0s, k0s, v0s, dots, do0s, qf_s, kf_s, vf_s, dof_s, bm_s
        torch.cuda.empty_cache()

    del l2_flush
    torch.cuda.empty_cache()

    return (
        {
            "clean_seqlen_ratchet_l": n_ratchet,
            "concat_seqlen_ratchet_2l": 2 * n_ratchet,
            "track1_bd3_attention_rsi": {"ratchet": ratchet_bd3.to_summary_dict(), "candidates": bd3_trials_detail},
            "seqlen_scaling_w16": seq_scaling,
        },
        ratchet_bd3,
    )


# ============================================================================
# 8. Suite C: 8-GPU NVSwitch `1.11B` `BD3-LM` (`DiT` `adaLN-Zero` + `V = 128,256`)
#    `FSDP2` 4-Rung Cumulative RSI Ablation & Exact 1-GPU Parity
# ============================================================================


class BD3DiTBlock(nn.Module):
    def __init__(self, d_model: int = 2048, h_q: int = 16, h_kv: int = 4, ffn_dim: int = 6144) -> None:
        super().__init__()
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = d_model // h_q
        self.wq = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
        self.wk = nn.Linear(d_model, h_kv * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.wv = nn.Linear(d_model, h_kv * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
        self.w_gate = nn.Linear(d_model, ffn_dim, bias=False, dtype=torch.bfloat16)
        self.w_up = nn.Linear(d_model, ffn_dim, bias=False, dtype=torch.bfloat16)
        self.w_dn = nn.Linear(ffn_dim, d_model, bias=False, dtype=torch.bfloat16)
        self.adaln_proj = nn.Linear(d_model, 6 * d_model, bias=False, dtype=torch.bfloat16)
        # Non-trivial post-warmup scale so all upstream layers carry rich non-zero gradients in parity checks
        nn.init.normal_(self.adaln_proj.weight, mean=0.0, std=0.02)

    def _swiglu(self, y: torch.Tensor) -> torch.Tensor:
        return self.w_dn(F.silu(self.w_gate(y)) * self.w_up(y))

    def forward(
        self,
        ht: torch.Tensor,
        h0: torch.Tensor | None,
        t_emb_noisy: torch.Tensor,
        t_emb_clean0: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bd3_meta: dict[str, Any],
        seg_meta_t: dict[str, Any],
        seg_meta_0: dict[str, Any],
        *,
        attn_mode: str,
        adaln_mode: str,
        prune_clean_out: bool,
        flex_bm: Any = None,
        comp_flex: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n = ht.shape[0]
        d = self.d_model
        cond_t = self.adaln_proj(F.silu(t_emb_noisy)).view(-1, 6, d)
        gt1, bt1, at1, gt2, bt2, at2 = cond_t.unbind(dim=1)

        assert h0 is not None
        cond_0 = self.adaln_proj(F.silu(t_emb_clean0)).view(1, 6, d)
        g01, b01, a01, g02, b02, a02 = cond_0.unbind(dim=1)

        if adaln_mode == "triton_segmented":
            yt1 = BlockSegmentedAdaLNZeroFn.apply(ht, gt1, bt1, seg_meta_t, 1e-6)
            y01 = BlockSegmentedAdaLNZeroFn.apply(h0, g01, b01, seg_meta_0, 1e-6)
        else:
            bid_t = seg_meta_t["block_id_long"]
            rt1 = torch.rsqrt(ht.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            yt1 = (ht.float() * rt1 * (1.0 + gt1.index_select(0, bid_t).float()) + bt1.index_select(0, bid_t).float()).to(ht.dtype)
            r01 = torch.rsqrt(h0.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            y01 = (h0.float() * r01 * (1.0 + g01.float()) + b01.float()).to(h0.dtype)

        if prune_clean_out:
            # Final layer: skip Q0, clean WO, clean post-attn Gate-Residual-adaLN, and clean SwiGLU MLP!
            qt = apply_1d_rope(self.wq(yt1).view(n, self.h_q, self.head_dim), cos, sin)
            kt = apply_1d_rope(self.wk(yt1).view(n, self.h_kv, self.head_dim), cos, sin)
            vt = self.wv(yt1).view(n, self.h_kv, self.head_dim)
            k0 = apply_1d_rope(self.wk(y01).view(n, self.h_kv, self.head_dim), cos, sin)
            v0 = self.wv(y01).view(n, self.h_kv, self.head_dim)
            ot, _ = bd3_trial3_adaptive_zerospill(qt, kt, vt, None, k0, v0, bd3_meta, False)
            attn_t = self.wo(ot.view(n, d))

            if adaln_mode == "triton_segmented":
                ht_mid, yt2 = BlockSegmentedGateResAdaLNFn.apply(ht, attn_t, at1, gt2, bt2, seg_meta_t, 1e-6)
            else:
                bid_t = seg_meta_t["block_id_long"]
                ht_mid = (ht.float() + at1.index_select(0, bid_t).float() * attn_t.float()).to(ht.dtype)
                rt2 = torch.rsqrt(ht_mid.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                yt2 = (ht_mid.float() * rt2 * (1.0 + gt2.index_select(0, bid_t).float()) + bt2.index_select(0, bid_t).float()).to(ht.dtype)

            mlp_t = self._swiglu(yt2)
            at2_tok = at2.index_select(0, seg_meta_t["block_id_long"])
            ht_out = (ht_mid.float() + at2_tok.float() * mlp_t.float()).to(ht.dtype)
            return ht_out, None

        # Non-terminal layer: pack [yt1 || y01] through linear projections so FSDP2 AllGather fires once per weight!
        y_cat = torch.cat([yt1, y01], dim=0)
        q_cat = self.wq(y_cat).view(2 * n, self.h_q, self.head_dim)
        k_cat = self.wk(y_cat).view(2 * n, self.h_kv, self.head_dim)
        v_cat = self.wv(y_cat).view(2 * n, self.h_kv, self.head_dim)

        qt = apply_1d_rope(q_cat[:n], cos, sin)
        q0 = apply_1d_rope(q_cat[n:], cos, sin)
        kt = apply_1d_rope(k_cat[:n], cos, sin)
        k0 = apply_1d_rope(k_cat[n:], cos, sin)
        vt, v0 = v_cat[:n], v_cat[n:]

        if attn_mode == "flex_2l":
            qf = torch.cat([qt, q0], dim=0).transpose(0, 1).unsqueeze(0).contiguous()
            kf = torch.cat([kt, k0], dim=0).transpose(0, 1).unsqueeze(0).contiguous()
            vf = torch.cat([vt, v0], dim=0).transpose(0, 1).unsqueeze(0).contiguous()
            of = comp_flex(qf, kf, vf, block_mask=flex_bm, enable_gqa=True).squeeze(0).transpose(0, 1).contiguous()
            attn_cat = self.wo(of.view(2 * n, d))
        else:
            ot, o0 = bd3_trial3_adaptive_zerospill(qt, kt, vt, q0, k0, v0, bd3_meta, True)
            attn_cat = self.wo(torch.cat([ot.view(n, d), o0.view(n, d)], dim=0))

        attn_t, attn_0 = attn_cat[:n], attn_cat[n:]

        if adaln_mode == "triton_segmented":
            ht_mid, yt2 = BlockSegmentedGateResAdaLNFn.apply(ht, attn_t, at1, gt2, bt2, seg_meta_t, 1e-6)
            h0_mid, y02 = BlockSegmentedGateResAdaLNFn.apply(h0, attn_0, a01, g02, b02, seg_meta_0, 1e-6)
        else:
            bid_t = seg_meta_t["block_id_long"]
            ht_mid = (ht.float() + at1.index_select(0, bid_t).float() * attn_t.float()).to(ht.dtype)
            rt2 = torch.rsqrt(ht_mid.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            yt2 = (ht_mid.float() * rt2 * (1.0 + gt2.index_select(0, bid_t).float()) + bt2.index_select(0, bid_t).float()).to(ht.dtype)

            h0_mid = (h0.float() + a01.float() * attn_0.float()).to(h0.dtype)
            r02 = torch.rsqrt(h0_mid.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            y02 = (h0_mid.float() * r02 * (1.0 + g02.float()) + b02.float()).to(h0.dtype)

        mlp_cat = self._swiglu(torch.cat([yt2, y02], dim=0))
        at2_tok = at2.index_select(0, seg_meta_t["block_id_long"])
        ht_out = (ht_mid.float() + at2_tok.float() * mlp_cat[:n].float()).to(ht.dtype)
        h0_out = (h0_mid.float() + a02.float() * mlp_cat[n:].float()).to(h0.dtype)
        return ht_out, h0_out


class BD3DiffusionLLM(nn.Module):
    """1.11B-parameter (`1,112,539,648` params) Block-Diffusion Transformer (`D=2048, 8L, V=128256`)."""

    def __init__(
        self,
        vocab_size: int = 128256,
        d_model: int = 2048,
        n_layers: int = 8,
        h_q: int = 16,
        h_kv: int = 4,
        ffn_dim: int = 6144,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, dtype=torch.bfloat16)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16),
        )
        self.blocks = nn.ModuleList(
            [BD3DiTBlock(d_model=d_model, h_q=h_q, h_kv=h_kv, ffn_dim=ffn_dim) for _ in range(n_layers)]
        )
        self.final_norm_w = nn.Parameter(torch.ones(d_model, dtype=torch.bfloat16))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=torch.bfloat16)

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _sinusoidal_t_embed(self, t_vals: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t_vals.device, dtype=torch.float32) / float(half)
        )
        args = t_vals[:, None].float() * 1000.0 * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
        return self.time_mlp(emb)

    def forward(
        self,
        batch: dict[str, Any],
        *,
        rung: str,
        global_clean_tokens: int,
        world_size_scale: float,
        comp_flex: Any,
    ) -> torch.Tensor:
        if rung == "rung1_flex_bd3_eager_adaln_full_ce":
            attn_m, adaln_m, prune_last, ce_m = "flex_2l", "eager", False, "eager_full"
        elif rung == "rung2_adaptive_zerospill_bd3_eager_adaln_full_ce":
            attn_m, adaln_m, prune_last, ce_m = "adaptive_bd3", "eager", False, "eager_full"
        elif rung == "rung3_adaptive_bd3_triton_adaln_last_layer_pruned":
            attn_m, adaln_m, prune_last, ce_m = "adaptive_bd3", "triton_segmented", True, "eager_full"
        elif rung == "rung4_full_rsi_stack_compacted_3gemm_masked_ce":
            attn_m, adaln_m, prune_last, ce_m = "adaptive_bd3", "triton_segmented", True, "compacted_3gemm"
        else:
            raise ValueError(f"Unknown rung: {rung}")

        n = batch["xt_ids"].shape[0]
        both_emb = self.tok_emb(torch.cat([batch["xt_ids"], batch["x0_ids"]], dim=0))
        ht, h0 = both_emb[:n], both_emb[n:]

        t_cat = torch.cat([batch["t_blks"], batch["t_zero"]], dim=0)
        t_emb_all = self._sinusoidal_t_embed(t_cat)
        t_emb_noisy = t_emb_all[:-1]
        t_emb_clean0 = t_emb_all[-1:]

        n_layers = len(self.blocks)
        for li, blk in enumerate(self.blocks):
            is_last = li == (n_layers - 1)
            ht, h0 = blk(
                ht,
                h0,
                t_emb_noisy,
                t_emb_clean0,
                batch["cos"],
                batch["sin"],
                batch["bd3_meta"],
                batch["seg_meta_t"],
                batch["seg_meta_0"],
                attn_mode=attn_m,
                adaln_mode=adaln_m,
                prune_clean_out=(prune_last and is_last),
                flex_bm=batch.get("flex_bm"),
                comp_flex=comp_flex,
            )

        r_fin = torch.rsqrt(ht.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ht_norm = (ht.float() * r_fin * self.final_norm_w.float()).to(torch.bfloat16)

        inv_t_scaled = (world_size_scale / (batch["t_tok"] * float(global_clean_tokens))).contiguous()
        if ce_m == "compacted_3gemm":
            return ActiveMaskCompacted3GemmCEFn.apply(
                ht_norm, self.lm_head.weight, batch["x0_ids"], batch["mask_tok"], inv_t_scaled, 2048
            )
        return eager_full_masked_diffusion_ce(
            ht_norm, self.lm_head.weight, batch["x0_ids"], batch["mask_tok"], inv_t_scaled
        )


def _prepare_bd3_rank_batch(
    doc_lens: list[int],
    W: int,
    vocab_size: int,
    device: torch.device,
    *,
    seed: int,
    comp_cbm: Any,
    build_flex_bm: bool = True,
    force_zero_mask: bool = False,
) -> dict[str, Any]:
    gen = torch.Generator(device=device).manual_seed(seed)
    n = sum(doc_lens)
    mask_token_id = vocab_size - 1
    x0_ids = torch.randint(0, vocab_size - 1, (n,), device=device, generator=gen)

    bd3_meta = build_bd3_packed_meta(doc_lens, W, device)
    seg_meta_t = build_adaln_segment_meta(doc_lens, W, device)
    seg_meta_0 = {
        "n": n,
        "W": n,
        "n_blks": 1,
        "block_id": torch.zeros((n,), device=device, dtype=torch.int32),
        "block_id_long": torch.zeros((n,), device=device, dtype=torch.long),
        "blk_bounds": torch.tensor([[0, n]], device=device, dtype=torch.int32),
    }

    # Complementary-antithetic clipped block noise schedule t_b in [0.30, 0.70]
    n_blks = seg_meta_t["n_blks"]
    half_b = max(1, n_blks // 2)
    t_h = 0.30 + 0.40 * torch.rand((half_b,), device=device, generator=gen)
    t_blks = torch.cat([t_h, 1.0 - t_h], dim=0)[:n_blks].contiguous()
    t_tok = t_blks.index_select(0, seg_meta_t["block_id_long"]).contiguous()

    if force_zero_mask:
        mask_tok = torch.zeros((n,), device=device, dtype=torch.bool)
    else:
        u_tok = torch.rand((n,), device=device, generator=gen)
        mask_tok = (u_tok < t_tok).contiguous()
        if int(mask_tok.sum().item()) == 0:
            mask_tok[0] = True

    xt_ids = torch.where(mask_tok, torch.full_like(x0_ids, mask_token_id), x0_ids).contiguous()
    cos, sin = build_doc_rope_cache(doc_lens, 128, device)

    flex_bm = None
    if build_flex_bm:
        doc_len0 = doc_lens[0]
        flex_bm = comp_cbm(
            lambda b, h, qi, ki, nl=n, dl0=doc_len0, bs=W: (
                (torch.where(qi >= nl, (qi - nl) // dl0, qi // dl0))
                == (torch.where(ki >= nl, (ki - nl) // dl0, ki // dl0))
            )
            & (
                (
                    ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) == (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                    & (((qi >= nl) & (ki >= nl)) | ((qi < nl) & (ki < nl)))
                )
                | (
                    ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) > (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                    & (qi < nl)
                    & (ki >= nl)
                )
                | (
                    ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) >= (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                    & (qi >= nl)
                    & (ki >= nl)
                )
            ),
            B=1,
            H=1,
            Q_LEN=2 * n,
            KV_LEN=2 * n,
            device=device,
        )

    return {
        "x0_ids": x0_ids,
        "xt_ids": xt_ids,
        "t_blks": t_blks,
        "t_zero": torch.zeros((1,), device=device, dtype=torch.float32),
        "t_tok": t_tok,
        "mask_tok": mask_tok,
        "cos": cos,
        "sin": sin,
        "bd3_meta": bd3_meta,
        "seg_meta_t": seg_meta_t,
        "seg_meta_0": seg_meta_0,
        "flex_bm": flex_bm,
    }


def run_suite_c_8gpu_bd3_fsdp2(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    assert world_size == 8, f"Suite C requires world_size == 8, got {world_size}"
    dist.barrier()

    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)
    # Parameters are already BF16 (`AllGather` = 2 B/param); omitting `param_dtype` prevents
    # FSDP2 `_pre_forward` from down-casting FP32 RoPE `(cos, sin)` & `t_blks` inputs to BF16!
    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)

    def _build_sharded_bd3_model(seed: int = 2026127) -> tuple[BD3DiffusionLLM, int]:
        torch.manual_seed(seed)
        m = BD3DiffusionLLM(vocab_size=128256, d_model=2048, n_layers=8, h_q=16, h_kv=4, ffn_dim=6144).to(device)
        param_cnt = sum(p.numel() for p in m.parameters())
        fully_shard(m.tok_emb, mp_policy=mp_policy)
        fully_shard(m.time_mlp, mp_policy=mp_policy)
        for blk in m.blocks:
            fully_shard(blk, mp_policy=mp_policy)
        fully_shard(m, mp_policy=mp_policy)
        return m, param_cnt

    # ------------------------------------------------------------------------
    # C.1 Exact 1-GPU Global Parity + Rank-7 Empty-Mask (M=0) Collective Symmetry Gate (Gate G9)
    # ------------------------------------------------------------------------
    W_c = 16
    docs_per_rank_p = [512, 512]  # 1,024 clean tokens/rank -> 8,192 clean / 16,384 [x_t||x_0] globally
    global_clean_p = 8 * sum(docs_per_rank_p)

    # Compute 1-GPU unsharded reference on rank 0 over the exact concatenation of all 8 ranks' batches (including rank 7 M=0!)
    ref_1gpu: dict[str, Any] = {}
    if rank == 0:
        torch.manual_seed(2026127)
        m_single = BD3DiffusionLLM(vocab_size=128256, d_model=2048, n_layers=8, h_q=16, h_kv=4, ffn_dim=6144).to(device)
        per_r_batches = [
            _prepare_bd3_rank_batch(
                docs_per_rank_p,
                W_c,
                128256,
                device,
                seed=5000 + r,
                comp_cbm=comp_cbm,
                build_flex_bm=False,
                force_zero_mask=(r == 7),
            )
            for r in range(8)
        ]
        all_docs_p = docs_per_rank_p * 8
        global_batch_p = {
            "x0_ids": torch.cat([b["x0_ids"] for b in per_r_batches], dim=0),
            "xt_ids": torch.cat([b["xt_ids"] for b in per_r_batches], dim=0),
            "t_blks": torch.cat([b["t_blks"] for b in per_r_batches], dim=0),
            "t_zero": torch.zeros((1,), device=device, dtype=torch.float32),
            "t_tok": torch.cat([b["t_tok"] for b in per_r_batches], dim=0),
            "mask_tok": torch.cat([b["mask_tok"] for b in per_r_batches], dim=0),
            "cos": torch.cat([b["cos"] for b in per_r_batches], dim=0),
            "sin": torch.cat([b["sin"] for b in per_r_batches], dim=0),
            "bd3_meta": build_bd3_packed_meta(all_docs_p, W_c, device),
            "seg_meta_t": build_adaln_segment_meta(all_docs_p, W_c, device),
            "seg_meta_0": {
                "n": global_clean_p,
                "W": global_clean_p,
                "n_blks": 1,
                "block_id": torch.zeros((global_clean_p,), device=device, dtype=torch.int32),
                "block_id_long": torch.zeros((global_clean_p,), device=device, dtype=torch.long),
                "blk_bounds": torch.tensor([[0, global_clean_p]], device=device, dtype=torch.int32),
            },
            "flex_bm": None,
        }
        loss_1g = m_single(
            global_batch_p,
            rung="rung2_adaptive_zerospill_bd3_eager_adaln_full_ce",
            global_clean_tokens=global_clean_p,
            world_size_scale=1.0,
            comp_flex=comp_flex,
        )
        loss_1g.backward()
        ref_1gpu = {
            "loss": float(loss_1g.item()),
            "tok_emb": m_single.tok_emb.weight.grad.detach().clone(),
            "blk0_wq": m_single.blocks[0].wq.weight.grad.detach().clone(),
            "blk0_adaln": m_single.blocks[0].adaln_proj.weight.grad.detach().clone(),
            "blk7_wdn": m_single.blocks[7].w_dn.weight.grad.detach().clone(),
            "lm_head": m_single.lm_head.weight.grad.detach().clone(),
        }
        del m_single, per_r_batches, global_batch_p
        torch.cuda.empty_cache()

    dist.barrier()
    fsdp_model, total_params = _build_sharded_bd3_model(seed=2026127)
    local_batch_p = _prepare_bd3_rank_batch(
        docs_per_rank_p,
        W_c,
        128256,
        device,
        seed=5000 + rank,
        comp_cbm=comp_cbm,
        build_flex_bm=True,
        force_zero_mask=(rank == 7),
    )

    parity_c1: dict[str, Any] = {}
    for p_rung in (
        "rung1_flex_bd3_eager_adaln_full_ce",
        "rung4_full_rsi_stack_compacted_3gemm_masked_ce",
    ):
        fsdp_model.zero_grad(set_to_none=True)
        l_loc = fsdp_model(
            local_batch_p,
            rung=p_rung,
            global_clean_tokens=global_clean_p,
            world_size_scale=float(world_size),
            comp_flex=comp_flex,
        )
        l_loc.backward()
        l_avg = l_loc.detach().clone()
        dist.all_reduce(l_avg, op=dist.ReduceOp.AVG)

        g_emb = fsdp_model.tok_emb.weight.grad.full_tensor()
        g_wq = fsdp_model.blocks[0].wq.weight.grad.full_tensor()
        g_ad = fsdp_model.blocks[0].adaln_proj.weight.grad.full_tensor()
        g_dn = fsdp_model.blocks[7].w_dn.weight.grad.full_tensor()
        g_hd = fsdp_model.lm_head.weight.grad.full_tensor()

        if rank == 0:
            lv = float(l_avg.item())
            parity_c1[p_rung] = {
                "rank7_forced_empty_mask_m0": True,
                "loss_rel_diff_vs_1gpu": round(abs(lv - ref_1gpu["loss"]) / max(1e-12, abs(ref_1gpu["loss"])), 8),
                "tok_emb_dw_rel_l2": round(rel_l2(g_emb, ref_1gpu["tok_emb"]), 6),
                "blk0_wq_dw_rel_l2": round(rel_l2(g_wq, ref_1gpu["blk0_wq"]), 6),
                "blk0_adaln_dw_rel_l2": round(rel_l2(g_ad, ref_1gpu["blk0_adaln"]), 6),
                "blk7_wdn_dw_rel_l2": round(rel_l2(g_dn, ref_1gpu["blk7_wdn"]), 6),
                "lm_head_dw_rel_l2": round(rel_l2(g_hd, ref_1gpu["lm_head"]), 6),
            }
        fsdp_model.zero_grad(set_to_none=True)

    del ref_1gpu, local_batch_p
    torch.cuda.empty_cache()
    dist.barrier()

    # ------------------------------------------------------------------------
    # C.2 8-GPU NVSwitch 4-Rung Cumulative RSI Scaling (`B=16` = 65,536 [x_t||x_0] tok/step,
    #     `B=32` = 131,072 [x_t||x_0] tok/step)
    # ------------------------------------------------------------------------
    rungs = [
        "rung1_flex_bd3_eager_adaln_full_ce",
        "rung2_adaptive_zerospill_bd3_eager_adaln_full_ce",
        "rung3_adaptive_bd3_triton_adaln_last_layer_pruned",
        "rung4_full_rsi_stack_compacted_3gemm_masked_ce",
    ]
    batch_scales = (
        [("B16_clean16k_dual32k", [1024, 1024])]
        if is_smoke
        else [
            ("B16_clean32k_dual65k", [2048, 2048]),
            ("B32_clean65k_dual131k", [2048, 2048, 2048, 2048]),
        ]
    )

    scaling_c2: dict[str, Any] = {}
    warmup_c = 1 if is_smoke else 2
    iters_c = 3 if is_smoke else 5

    for b_label, docs_rk in batch_scales:
        clean_per_rk = sum(docs_rk)
        global_clean = 8 * clean_per_rk
        global_dual = 2 * global_clean

        # 2 rotating batches per rank so mask compaction & inputs vary every step
        rank_batches = [
            _prepare_bd3_rank_batch(
                docs_rk,
                W_c,
                128256,
                device,
                seed=8000 + sl * 100 + rank,
                comp_cbm=comp_cbm,
                build_flex_bm=True,
                force_zero_mask=False,
            )
            for sl in range(2)
        ]

        rung_results: dict[str, Any] = {}
        for rg in rungs:
            dist.barrier()
            for w_i in range(warmup_c):
                fsdp_model.zero_grad(set_to_none=True)
                lw = fsdp_model(
                    rank_batches[w_i % 2],
                    rung=rg,
                    global_clean_tokens=global_clean,
                    world_size_scale=float(world_size),
                    comp_flex=comp_flex,
                )
                lw.backward()
            torch.cuda.synchronize(device)
            dist.barrier()

            torch.cuda.reset_peak_memory_stats(device)
            step_times_ms: list[float] = []
            for it in range(iters_c):
                dist.barrier()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                fsdp_model.zero_grad(set_to_none=True)
                li = fsdp_model(
                    rank_batches[it % 2],
                    rung=rg,
                    global_clean_tokens=global_clean,
                    world_size_scale=float(world_size),
                    comp_flex=comp_flex,
                )
                li.backward()
                e_ev.record()
                e_ev.synchronize()
                t_loc = torch.tensor([s_ev.elapsed_time(e_ev)], device=device, dtype=torch.float32)
                dist.all_reduce(t_loc, op=dist.ReduceOp.MAX)
                step_times_ms.append(float(t_loc.item()))

            pk_loc = torch.tensor(
                [torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)], device=device, dtype=torch.float32
            )
            dist.all_reduce(pk_loc, op=dist.ReduceOp.MAX)

            if rank == 0:
                ss = sorted(step_times_ms)
                med_ms = float(statistics.median(ss))
                q25 = ss[max(0, (len(ss) - 1) // 4)]
                q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
                clk, pwr = query_sm_telemetry(0)
                rung_results[rg] = {
                    "median_ms": round(med_ms, 3),
                    "iqr_ms": round(float(q75 - q25), 3),
                    "clean_tokens_per_s": round(global_clean / (med_ms * 1e-3), 1),
                    "dual_stream_2l_tokens_per_s": round(global_dual / (med_ms * 1e-3), 1),
                    "max_rank_peak_alloc_mib": round(float(pk_loc.item()), 2),
                    "sm_clock_mhz": clk,
                    "power_w": pwr,
                }

        if rank == 0:
            r1_ms = rung_results["rung1_flex_bd3_eager_adaln_full_ce"]["median_ms"]
            r4_ms = rung_results["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["median_ms"]
            r1_mem = rung_results["rung1_flex_bd3_eager_adaln_full_ce"]["max_rank_peak_alloc_mib"]
            r4_mem = rung_results["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["max_rank_peak_alloc_mib"]
            scaling_c2[b_label] = {
                "global_clean_tokens": global_clean,
                "global_dual_stream_2l_tokens": global_dual,
                "block_size_w": W_c,
                "rungs": rung_results,
                "speedup_rung4_vs_rung1": round(r1_ms / r4_ms, 3),
                "hbm_saved_mib_rung4_vs_rung1": round(r1_mem - r4_mem, 2),
                "hbm_ratio_rung1_over_rung4": round(r1_mem / max(1e-9, r4_mem), 3),
            }

        del rank_batches
        fsdp_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        dist.barrier()

    del fsdp_model
    gc.collect()
    torch.cuda.empty_cache()

    if rank != 0:
        return {}
    return {
        "model_parameters": total_params,
        "world_size": world_size,
        "parity_and_empty_mask_gate_c1": parity_c1,
        "scaling_4rung_ablation_c2": scaling_c2,
    }


# ============================================================================
# 9. Main Entrypoint (`torchrun --nproc_per_node=8`)
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="E127 Diffusion-LLM, BD3-LM, & DiT adaLN-Zero RSI Benchmark")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=datetime.timedelta(minutes=30),
        )

    is_smoke = args.mode == "smoke"
    t0 = time.perf_counter()
    init_clk, init_pwr = query_sm_telemetry(local_rank)

    suite_0: dict[str, Any] = {}
    suite_a: dict[str, Any] = {}
    suite_b: dict[str, Any] = {}
    ledger_sha = ""
    ratchets_all: list[RSIVectorRatchet] = []

    if rank == 0:
        suite_0 = run_suite_0_audits_and_parity(device, is_smoke)
        suite_a, r_adaln, r_ce = run_suite_a_adaln_and_masked_ce_rsi(device, is_smoke)
        suite_b, r_bd3 = run_suite_b_bd3_attention_rsi_and_scaling(device, is_smoke)
        ratchets_all = [r_bd3, r_adaln, r_ce]
        if args.ledger_output is not None:
            ledger_sha = write_rsi_ledger_tsv(ratchets_all, args.ledger_output)

    if world_size > 1:
        dist.barrier()
        suite_c = run_suite_c_8gpu_bd3_fsdp2(rank, world_size, device, is_smoke)
        dist.barrier()
    else:
        suite_c = {"skipped": "world_size == 1"}

    elapsed_s = round(time.perf_counter() - t0, 2)
    if rank == 0:
        env = environment_metadata()
        env["cudnn_version"] = torch.backends.cudnn.version()
        payload = {
            "experiment": "E127_diffusion_llm_bd3_adaln_rsi_training",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": elapsed_s,
            "initial_sm_clock_mhz": init_clk,
            "initial_power_w": init_pwr,
            "rsi_harness": {
                "version": HARNESS_VERSION,
                "eval_harness_sha256": eval_harness_sha256(),
                "ledger_tsv_sha256": ledger_sha,
                "tracks": [r.to_summary_dict() for r in ratchets_all],
            },
            "environment": env,
            "suite_0_audits": suite_0,
            "suite_a_1gpu_adaln_and_masked_ce_rsi": suite_a,
            "suite_b_1gpu_bd3_attention_rsi_and_scaling": suite_b,
            "suite_c_8gpu_bd3_fsdp2_scaling": suite_c,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(
            f"[E127 {args.mode}] Completed in {elapsed_s}s | harness_sha={eval_harness_sha256()[:16]} "
            f"| parity_ok={suite_0['per_tensor_fp64_oracle_parity']['all_parity_ok']} "
            f"| richardson_ok={suite_0['fp64_richardson_central_difference']['all_ok']}"
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

