#!/usr/bin/env python3
"""
E140 (v131, Round-14 RSI): Kimi Linear & Kimi Delta Attention (KDA) Zero-Spill Kernels
======================================================================================

Reference Paper:
  "Kimi Linear: An Expressive, Efficient Attention Architecture"
  Moonshot AI / Kimi Team (arXiv:2510.26692)
  Official Repository: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda

Five Upstream Pathologies Resolved by Round-14 RSI:
  1. Pathology P1 (Upstream `safe_gate=True` `32,768`-NaN Explosion, Missing `lower_bound`
     Plumbing in `chunk_fwd.py:77` & `chunk_bwd.py:786`, & Paper Listing 8(b) Raw-`v_i` Bug):
     - Upstream `chunk_kda_fwd` and `chunk_kda_bwd` omit `lower_bound=lower_bound` when
       calling `chunk_kda_fwd_intra` / `chunk_kda_bwd_intra`, leaving `GATE_SHIFT` hardcoded
       to `2.0 * RCP_LN2` (`d = 15.0`). On diagonal 16x16 sub-chunks, factoring
       `exp(g_i - g_j) = exp(g_i - C) * exp(C - g_j)` around `C = g_0 - 15` evaluates
       `exp(+90.0) = +Inf` in FP32 when `g = -7.0`, corrupting 32,768 / 524,288 elements
       to `NaN` via `0 * Inf = NaN` in `tl.dot`.
     - Paper Listing 8(b) (`def chunk_kda`) omits `beta` on `Akk` and `(w, u)` and updates
       inter-chunk state `S` with raw `v_i` instead of pseudo-value `u_i - w_i @ S`.
     - Round-14 RSI uses Overflow-Free Non-Positive Causal Sub-Chunking (`max exp arg <= 0.0`
       across both diagonal and off-diagonal 16x16 sub-chunks), producing `0` NaNs across
       `524,288/524,288` elements and `0.0` future-token leakage.
  2. Pathology P2 (`156-3,648` SM80 Register Spills Across `36/36` Configs of
     `chunk_kda_bwd_kernel_wy_dqkg_fused`, plus `30` Spills in `chunk_kda_bwd_kernel_intra`
     and `4` Spills in `chunk_kda_fwd_kernel_inter_solve_fused`):
     - Round-14 RSI decouples the backward WY slice kernel (`BK=32, BV=32`, no `[64, 64]`
       `b_A`/`b_dA` held during the `i_v` loop) from the `[64, 64]` `dAkk` inverse-adjoint
       kernel, and stages intra-chunk forward/backward sub-chunk tiles to achieve `0` SM80
       cubin spills across all `16/16` Round-14 RSI Triton kernels.
  3. Pathology P3 (`10`-Launch Forward / `13`-Launch Backward Fragmentation):
     - Superfuses `QK-L2Norm + BetaSigmoid` and `KDA-Gate + Chunk-Cumsum` in forward and
       backward, eliminating separate global memory passes and host `.sum(0)` reductions.
  4. Pathology P4 (GVA `HV > H` Explicit `repeat_interleave` Memory Bloat):
     - Native GVA group-stride indexing `i_h = i_hv // (HV // H)` in all forward/backward
       kernels without `[B, T, HV, K]` Q/K expansion.
  5. Pathology P5 (8-GPU NVSwitch Channel-Affine Monoid Context Parallelism `KDA-CP = 8`):
     - Composes exact channel-gated affine transitions `P_c = Diag(exp(g_{c,BT})) - J_c^T W_c`
       and `H_c = J_c^T U_c` across 8 GPUs via a single NVSwitch `all_gather_into_tensor` and
       reuses `P_rank^T` in backward (`2x` smaller backward payload).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# =============================================================================
# 1. Exact Mathematical Oracles (FP64 / FP32 Reference & Paper Listing 8(b) Bug)
# =============================================================================

def kda_gate_ref(
    g_raw: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """
    Exact reference KDA channel-wise log-decay gate:
      - Unbounded (`lower_bound is None`): `g = -exp(A_log) * softplus(g_raw + dt_bias)`
      - Bounded (`lower_bound is not None`): `g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))`
    """
    HV, K = g_raw.shape[-2], g_raw.shape[-1]
    x = g_raw
    if dt_bias is not None:
        x = x + dt_bias.to(x.dtype).view(HV, K)
    scale_a = A_log.to(x.dtype).view(HV, 1).exp() if A_log is not None else 1.0
    if lower_bound is None:
        return -scale_a * F.softplus(x)
    return float(lower_bound) * torch.sigmoid(scale_a * x)


def kda_recurrent_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    eps: float = 1e-6,
    compute_dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact token-by-token mathematical oracle for Kimi Delta Attention (arXiv:2510.26692 Eq. 1):
      S_t = (I - beta_t k_t k_t^T) Diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T
      o_t = scale * S_t^T q_t
    """
    orig_dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    qf = q.to(compute_dtype)
    kf = k.to(compute_dtype)
    vf = v.to(compute_dtype)
    gf = g_raw.to(compute_dtype)
    bf = beta_raw.to(compute_dtype)

    if use_qk_l2norm:
        qf = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + eps)
        kf = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + eps)
    if use_beta_sigmoid:
        b_scale = 2.0 if allow_neg_eigval else 1.0
        bf = b_scale * torch.sigmoid(bf)
    if use_gate_in_kernel:
        gf = kda_gate_ref(
            gf,
            A_log.to(compute_dtype) if A_log is not None else None,
            dt_bias.to(compute_dtype) if dt_bias is not None else None,
            lower_bound=lower_bound,
        )

    if G > 1:
        qf = qf.repeat_interleave(G, dim=2)
        kf = kf.repeat_interleave(G, dim=2)

    S = torch.zeros(B, HV, K, V, device=q.device, dtype=compute_dtype)
    if initial_state is not None:
        S = S + initial_state.to(compute_dtype)

    out_list = []
    for t in range(T):
        qt = qf[:, t] * scale            # [B, HV, K]
        kt = kf[:, t]                    # [B, HV, K]
        vt = vf[:, t]                    # [B, HV, V]
        gt = gf[:, t]                    # [B, HV, K]
        bt = bf[:, t].unsqueeze(-1)      # [B, HV, 1]

        S = S * gt.exp().unsqueeze(-1)
        pred_v = (kt.unsqueeze(-1) * S).sum(dim=-2)  # [B, HV, V]
        S = S + (bt * kt).unsqueeze(-1) * (vt - pred_v).unsqueeze(-2)
        ot = (qt.unsqueeze(-1) * S).sum(dim=-2)      # [B, HV, V]
        out_list.append(ot)

    o = torch.stack(out_list, dim=1)
    if compute_dtype == torch.float64 and orig_dtype == torch.float64:
        return o, S
    return o.to(orig_dtype), S.to(torch.float32)


def paper_listing8b_buggy_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reproduces the two mathematical bugs in `arXiv:2510.26692` Listing 8(b) (`def chunk_kda`):
      1. Line 29 updates inter-chunk state `S += (k_i * decay)^T @ v_i` with raw `v_i`
         instead of pseudo-value `v_new = u_i - w_i @ S` (dropping the Householder erasure
         `-W_t S_t` across chunk boundaries!).
      2. Lines 15 & 20 omit `beta` weighting on `Akk` and `(w, u)`.
    """
    orig_dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5

    qf = (q.float().repeat_interleave(G, dim=2) * scale).view(B, NT, BT, HV, K).permute(0, 3, 1, 2, 4).contiguous()
    kf = k.float().repeat_interleave(G, dim=2).view(B, NT, BT, HV, K).permute(0, 3, 1, 2, 4).contiguous()
    vf = v.float().view(B, NT, BT, HV, V).permute(0, 3, 1, 2, 4).contiguous()
    gc = g_log.float().view(B, NT, BT, HV, K).permute(0, 3, 1, 2, 4).contiguous().cumsum(dim=-2)

    idx = torch.arange(BT, device=q.device)
    g_diff = gc[:, :, :, :, None, :] - gc[:, :, :, None, :, :]  # [B, HV, NT, BT, BT, K]
    causal_le = (idx[:, None] >= idx[None, :])[None, None, None, :, :, None]
    causal_lt = (idx[:, None] > idx[None, :])[None, None, None, :, :, None]

    s1 = torch.where(causal_le, g_diff.clamp(max=0.0).exp(), 0.0)
    s2 = torch.where(causal_lt, g_diff.clamp(max=0.0).exp(), 0.0)
    Aqk = (qf[:, :, :, :, None, :] * kf[:, :, :, None, :, :] * s1).sum(dim=-1)
    Akk = (kf[:, :, :, :, None, :] * kf[:, :, :, None, :, :] * s2).sum(dim=-1)  # Bug: omits beta!

    I_bt = torch.eye(BT, device=q.device, dtype=torch.float32)
    A = torch.linalg.solve_triangular(I_bt + Akk, I_bt, upper=False, unitriangular=True)
    w = A @ (gc.exp() * kf)  # Bug: omits beta!
    u = A @ vf               # Bug: omits beta!

    S = torch.zeros(B, HV, K, V, device=q.device, dtype=torch.float32)
    o = torch.zeros_like(vf)
    for i in range(NT):
        q_i, k_i, v_i, u_i, g_i, w_i = qf[:, :, i], kf[:, :, i], vf[:, :, i], u[:, :, i], gc[:, :, i], w[:, :, i]
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk[:, :, i] @ (u_i - w_i @ S)
        decay = (g_i[:, :, -1:, :] - g_i).exp()
        S = S * g_i[:, :, -1, :, None].exp()
        # Listing 8(b) Line 29 bug: uses raw `v_i` instead of `(u_i - w_i @ S)`!
        S = S + (k_i * decay).transpose(-1, -2) @ v_i

    o_out = o.permute(0, 2, 3, 1, 4).reshape(B, T, HV, V)
    return o_out.to(orig_dtype), S


def upstream_safe_gate_unclamped_subchunk_sim(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Reproduces Upstream `fla/ops/kda/chunk_intra.py:1430-1445` (`safe_gate=True`) where
    `chunk_fwd.py:77` omits `lower_bound=lower_bound`, leaving `GATE_SHIFT = 2.0` (`d = 15.0`)
    and factoring `exp(g_i - g_j) = exp(g_i - C) * exp(C - g_j)` with `C = g_0 - 15.0` inside
    each 16-token sub-chunk. When `g_log = -7.0`, `C - g_15 = +90.0 > 88.72`, overflowing FP32
    to `Inf` and producing `0 * Inf = NaN` in the matmul!
    """
    B, T, H, K = q.shape
    BC = 16
    NS = T // BC
    gc = g_log.float().view(B, T // chunk_size, chunk_size, H, K).cumsum(dim=2).view(B, NS, BC, H, K)
    qs = q.float().view(B, NS, BC, H, K)
    ks = k.float().view(B, NS, BC, H, K)
    g0 = gc[:, :, 0:1, :, :] - 15.0  # Upstream hardcoded C = g_0 - 0.5 * (16 - 1) * 2.0
    gq = (gc - g0).exp()
    gk = (g0 - gc).exp()  # Overflows FP32 to +Inf when g_0 - g_15 > 103.72!
    Aqk_diag = torch.einsum("bsihk,bsjhk->bshij", qs * gq, ks * gk)
    return Aqk_diag


def _kda_chunk_prep_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    scale: float,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    lower_bound: Optional[float],
    chunk_size: int,
):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    BT = chunk_size
    NT = T // BT

    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = g_raw.float()
    bf = beta_raw.float()

    if use_qk_l2norm:
        qf = F.normalize(qf, p=2.0, dim=-1, eps=1e-6)
        kf = F.normalize(kf, p=2.0, dim=-1, eps=1e-6)
    if use_beta_sigmoid:
        bf = (2.0 if allow_neg_eigval else 1.0) * torch.sigmoid(bf)
    if use_gate_in_kernel:
        gf = kda_gate_ref(gf, A_log, dt_bias, lower_bound=lower_bound)

    if G > 1:
        qf = qf.repeat_interleave(G, dim=2)
        kf = kf.repeat_interleave(G, dim=2)

    qc = (qf * scale).view(B, NT, BT, HV, K).permute(0, 3, 1, 2, 4).contiguous()
    kc = kf.view(B, NT, BT, HV, K).permute(0, 3, 1, 2, 4).contiguous()
    vc = vf.view(B, NT, BT, HV, V).permute(0, 3, 1, 2, 4).contiguous()
    gc = gf.view(B, NT, BT, HV, K).permute(0, 3, 1, 2, 4).contiguous().cumsum(dim=-2)
    bc = bf.view(B, NT, BT, HV).permute(0, 3, 1, 2).contiguous()

    idx = torch.arange(BT, device=q.device)
    m_le = (idx[:, None] >= idx[None, :])[None, None, None, :, :, None]
    m_lt = (idx[:, None] > idx[None, :])[None, None, None, :, :, None]
    g_diff = gc[:, :, :, :, None, :] - gc[:, :, :, None, :, :]
    g_diff_le = torch.where(m_le, g_diff, 0.0)
    g_diff_lt = torch.where(m_lt, g_diff, 0.0)

    Aqk = torch.where(m_le.squeeze(-1), (qc[:, :, :, :, None, :] * kc[:, :, :, None, :, :] * g_diff_le.exp()).sum(dim=-1), 0.0)
    L_mat = torch.where(m_lt.squeeze(-1), (kc[:, :, :, :, None, :] * kc[:, :, :, None, :, :] * g_diff_lt.exp()).sum(dim=-1) * bc[:, :, :, :, None], 0.0)
    return qc, kc, vc, gc, bc, Aqk, L_mat


_kda_chunk_prep_compiled = torch.compile(_kda_chunk_prep_eager, dynamic=True)


def eager_kda_chunk_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    chunk_size: int = 64,
    use_compiled_step: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unfused PyTorch eager chunkwise KDA baseline (`fla/ops/kda/naive.py` + `gate.py`).
    Materializes `[B, HV, NT, BT, BT, K]` 6D intra-chunk broadcast tensors in HBM and runs
    host-side Python loops over `c in range(NT)`.
    """
    orig_dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5

    prep_fn = _kda_chunk_prep_compiled if use_compiled_step else _kda_chunk_prep_eager
    qc, kc, vc, gc, bc, Aqk, L_mat = prep_fn(
        q, k, v, g_raw, beta_raw, A_log, dt_bias, float(scale),
        use_qk_l2norm, use_gate_in_kernel, use_beta_sigmoid, allow_neg_eigval, lower_bound, chunk_size,
    )

    I_bt = torch.eye(BT, device=q.device, dtype=torch.float32)
    A = torch.linalg.solve_triangular(I_bt + L_mat, I_bt, upper=False, unitriangular=True)
    w = A @ (gc.exp() * kc * bc.unsqueeze(-1))
    u = A @ (vc * bc.unsqueeze(-1))

    S = torch.zeros(B, HV, K, V, device=q.device, dtype=torch.float32)
    if initial_state is not None:
        S = S + initial_state.float()

    o_chunks = []
    for ci in range(NT):
        q_i = qc[:, :, ci]
        k_i = kc[:, :, ci]
        u_i = u[:, :, ci]
        w_i = w[:, :, ci]
        g_i = gc[:, :, ci]
        Aqk_i = Aqk[:, :, ci]
        v_new_i = u_i - w_i @ S
        o_i = (q_i * g_i.exp()) @ S + Aqk_i @ v_new_i
        o_chunks.append(o_i)
        g_last = g_i[:, :, -1:, :]
        k_decay = k_i * (g_last - g_i).exp()
        S = S * g_last.transpose(-1, -2).exp() + k_decay.transpose(-1, -2) @ v_new_i

    o = torch.stack(o_chunks, dim=2).permute(0, 2, 3, 1, 4).reshape(B, T, HV, V)
    return o.to(orig_dtype), (S if output_final_state else None)


# =============================================================================
# 2. Negative-Control Monolithic SM80 Spill Kernel (Reproduces Upstream P2)
# =============================================================================

@triton.jit
def _kda_monolithic_wy_dqkg_bwd_spill_kernel(
    q_ptr, k_ptr, v_ptr, v_new_ptr, g_ptr, beta_ptr, A_ptr, h_ptr, do_ptr, dh_ptr, dv_ptr,
    dq_ptr, dk_ptr, dv2_ptr, dg_ptr, db_ptr, dA_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Negative-control monolithic WY backward kernel mirroring upstream
    `fla/ops/kda/chunk_bwd.py:chunk_kda_bwd_kernel_wy_dqkg_fused` at `BT=64, BK=64, BV=64`:
    holds `[64, 64]` `b_A` and `b_dA` (`FP32`) across nested `i_k` and `i_v` loops alongside
    `b_dq, b_dk, b_dw` (`3 x [64, 64]` `FP32`) and `b_v_new, b_do, b_h, b_dh, b_dv`,
    causing massive SM80 register spills (`> 200` spills at 255 regs or `> 1,300` at 32 regs).
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)

    base_qk = (bos * H + i_h) * K
    base_v = (bos * HV + i_hv) * V
    base_g = (bos * HV + i_hv) * K
    base_b = bos * HV + i_hv
    base_A = (bos * HV + i_hv) * BT
    base_h = (i_tg * HV + i_hv) * K * V

    b_beta = tl.load(beta_ptr + base_b + o_t * HV).to(tl.float32)
    b_A = tl.load(A_ptr + base_A + o_A[:, None] + o_t[None, :] * (HV * BT))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        b_k = tl.load(k_ptr + base_qk + o_t[:, None] * (H * K) + o_k[None, :])
        b_g = tl.load(g_ptr + base_g + o_t[:, None] * (HV * K) + o_k[None, :]).to(tl.float32)
        b_gn = tl.load(g_ptr + base_g + (i_t * BT + BT - 1) * (HV * K) + o_k).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            b_v_new = tl.load(v_new_ptr + base_v + o_t[:, None] * (HV * V) + o_v[None, :])
            b_do = tl.load(do_ptr + base_v + o_t[:, None] * (HV * V) + o_v[None, :])
            b_h = tl.load(h_ptr + base_h + o_v[:, None] + o_k[None, :] * V)
            b_dh = tl.load(dh_ptr + base_h + o_v[:, None] + o_k[None, :] * V)
            b_dv = tl.load(dv_ptr + base_v + o_t[:, None] * (HV * V) + o_v[None, :])

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq)
            b_dk = tl.dot(b_v_new, b_dh.to(b_v_new.dtype), b_dk)
            b_dw = tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype), b_dw)
            if i_k == 0:
                b_v = tl.load(v_ptr + base_v + o_t[:, None] * (HV * V) + o_v[None, :])
                b_dA = tl.dot(b_dv, tl.trans(b_v), b_dA)
                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)
                tl.store(dv2_ptr + base_v + o_t[:, None] * (HV * V) + o_v[None, :], b_dv2.to(b_v.dtype))

        b_gk_exp = tl.exp(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= tl.exp(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.exp(b_gn[None, :] - b_g)
        b_kg = b_k * b_gk_exp
        b_dw = -b_dw.to(b_A.dtype)
        b_dA = tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)), b_dA)
        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg, 1)
        b_q = tl.load(q_ptr + base_qk + o_t[:, None] * (H * K) + o_k[None, :])
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        m_last = (o_t == (i_t * BT + BT - 1))[:, None]
        b_dg = b_q * b_dq - b_kdk + tl.where(m_last, b_dgk[None, :], 0.0) + b_kg * b_dkgb * b_beta[:, None]
        b_dk = b_dk + b_dkgb * b_gb
        tl.store(dq_ptr + base_g + o_t[:, None] * (HV * K) + o_k[None, :], b_dq)
        tl.store(dk_ptr + base_g + o_t[:, None] * (HV * K) + o_k[None, :], b_dk)
        tl.store(dg_ptr + base_g + o_t[:, None] * (HV * K) + o_k[None, :], b_dg)

    m_A = o_t[:, None] > o_t[None, :]
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0.0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0.0)
    tl.store(dA_ptr + base_A + o_t[:, None] * (HV * BT) + o_A[None, :], b_dA)
    tl.store(db_ptr + base_b + o_t * HV, b_db)


# =============================================================================
# 3. Round-14 RSI Zero-Spill Triton Kernels (16/16 Kernels at 0 Spills)
# =============================================================================

@triton.jit
def _kda_superfused_qk_l2norm_beta_fwd_kernel(
    q_ptr, k_ptr, beta_raw_ptr,
    q_norm_ptr, q_rstd_ptr, k_norm_ptr, k_rstd_ptr, beta_out_ptr,
    beta_scale, eps,
    N_ROWS,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Kernel 1: Fuses Q L2-normalization, K L2-normalization, and Beta Sigmoid
    in a single launch (`0` spills).
    Grid: `(N_ROWS,)` where `N_ROWS = B * T * HV`.
    """
    row_hv = tl.program_id(0).to(tl.int64)
    if row_hv >= N_ROWS:
        return
    G = HV // H
    bt_idx = row_hv // HV
    i_hv = row_hv % HV
    i_h = i_hv // G

    b_raw = tl.load(beta_raw_ptr + row_hv).to(tl.float32)
    tl.store(beta_out_ptr + row_hv, beta_scale * tl.sigmoid(b_raw))

    if (i_hv % G) == 0:
        row_h = bt_idx * H + i_h
        offs_k = tl.arange(0, BK)
        mask_k = offs_k < K
        base = row_h * K + offs_k
        q_val = tl.load(q_ptr + base, mask=mask_k, other=0.0).to(tl.float32)
        k_val = tl.load(k_ptr + base, mask=mask_k, other=0.0).to(tl.float32)
        q_rstd = tl.rsqrt(tl.sum(q_val * q_val, axis=0) + eps)
        k_rstd = tl.rsqrt(tl.sum(k_val * k_val, axis=0) + eps)
        tl.store(q_rstd_ptr + row_h, q_rstd)
        tl.store(k_rstd_ptr + row_h, k_rstd)
        tl.store(q_norm_ptr + base, (q_val * q_rstd).to(q_norm_ptr.dtype.element_ty), mask=mask_k)
        tl.store(k_norm_ptr + base, (k_val * k_rstd).to(k_norm_ptr.dtype.element_ty), mask=mask_k)


@triton.jit
def _kda_fused_gate_cumsum_fwd_kernel(
    g_raw_ptr, A_log_ptr, dt_bias_ptr, g_cumsum_ptr,
    lower_bound,
    T: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """
    Kernel 2: Fuses KDA channel-wise decay gate activation
      `g_act = -exp(A_log) * softplus(g_raw + dt_bias)` (or bounded `lower_bound * sigmoid(...)`)
    and chunk-local `tl.cumsum(g_act, axis=0)` in natural-log units (`0` spills).
    Grid: `(NT * cdiv(K, BS), B * HV)`.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NS = tl.cdiv(K, BS)
    i_t = pid0 // NS
    i_s = pid0 % NS

    i_b = i_bh // HV
    i_hv = i_bh % HV
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m_t = o_t < T
    m_s = o_s < K
    mask = m_t[:, None] & m_s[None, :]

    ptrs = (bos + o_t[:, None]) * (HV * K) + i_hv * K + o_s[None, :]
    b_g = tl.load(g_raw_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

    if USE_GATE_IN_KERNEL:
        if HAS_BIAS:
            b_bias = tl.load(dt_bias_ptr + i_hv * K + o_s, mask=m_s, other=0.0).to(tl.float32)
            b_g = b_g + b_bias[None, :]
        b_A = tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32)) if HAS_A else 1.0
        if USE_LOWER_BOUND:
            b_act = lower_bound * tl.sigmoid(b_A * b_g)
        else:
            # Numerically stable softplus(x) = max(x, 0) + log(1 + exp(-|x|))
            b_sp = tl.maximum(b_g, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(b_g)))
            b_act = -b_A * b_sp
        b_act = tl.where(mask, b_act, 0.0)
    else:
        b_act = tl.where(mask, b_g, 0.0)

    b_cs = tl.cumsum(b_act, axis=0)
    tl.store(g_cumsum_ptr + ptrs, b_cs, mask=mask)


@triton.jit
def _kda_overflow_free_intra_diag_subchunk_fwd_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr, Aqk_ptr, Akkd_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Kernel 3: Overflow-Free Non-Positive Causal Sub-Chunk Diagonal Kernel (`BC = 16`).
    Resolves Upstream Pathology P1 (`32,768` NaNs in `safe_gate=True`):
    For each 16x16 diagonal sub-chunk, computes `Aqk_diag[r, c]` (`r >= c`) and
    `L_diag[r, c]` (`r > c`) using strictly non-positive causal differences
    `g_r - g_c <= 0` (`exp(g_r - g_c) in (0, 1]`), then inverts `(I_16 + L_diag)^-1`
    in registers and writes `Akkd` (`[B, T, HV, 16]` FP32) and diagonal blocks of `Aqk`.
    Grid: `(NT, NC, B * HV)` where `NC = BT // BC = 4`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_sub = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T

    t_sub = i_t * BT + i_sub * BC
    o_i = tl.arange(0, BC)
    o_c = t_sub + o_i
    m_c = o_c < T

    b_beta = tl.load(beta_ptr + (bos + o_c) * HV + i_hv, mask=m_c, other=0.0).to(tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_ck = m_c[:, None] & m_k[None, :]
        qk_off = (bos + o_c[:, None]) * (H * K) + i_h * K + o_k[None, :]
        g_off = (bos + o_c[:, None]) * (HV * K) + i_hv * K + o_k[None, :]

        b_q = tl.load(q_ptr + qk_off, mask=m_ck, other=0.0).to(tl.float32)
        b_k = tl.load(k_ptr + qk_off, mask=m_ck, other=0.0).to(tl.float32)
        b_g = tl.load(g_ptr + g_off, mask=m_ck, other=0.0).to(tl.float32)

        for j in range(BC):
            p_kj = k_ptr + (bos + t_sub + j) * (H * K) + i_h * K + o_k
            p_gj = g_ptr + (bos + t_sub + j) * (HV * K) + i_hv * K + o_k
            b_kj = tl.load(p_kj, mask=m_k, other=0.0).to(tl.float32)
            b_gj = tl.load(p_gj, mask=m_k, other=0.0).to(tl.float32)

            m_le = (o_i >= j)[:, None] & m_ck
            # Pre-exponentiation causal clamp: strictly <= 0 so exp() in (0, 1]!
            diff = tl.where(m_le, tl.minimum(b_g - b_gj[None, :], 0.0), 0.0)
            decay = tl.where(m_le, tl.exp(diff), 0.0)

            dot_qk = tl.sum(b_q * b_kj[None, :] * decay, axis=1)
            dot_kk = tl.sum(b_k * b_kj[None, :] * decay, axis=1)
            col_mask = (o_i == j)[None, :]
            b_Aqk += tl.where(col_mask, dot_qk[:, None], 0.0)
            b_Akk += tl.where(col_mask, dot_kk[:, None], 0.0)

    m_tril = o_i[:, None] >= o_i[None, :]
    m_strict = o_i[:, None] > o_i[None, :]
    m_diag = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_tril, b_Aqk * scale, 0.0)
    b_L = tl.where(m_strict, b_Akk * b_beta[:, None], 0.0)

    # Store diagonal 16x16 block of Aqk
    aqk_ptrs = Aqk_ptr + (bos + o_c[:, None]) * (HV * BT) + i_hv * BT + (i_sub * BC + o_i)[None, :]
    tl.store(aqk_ptrs, b_Aqk.to(Aqk_ptr.dtype.element_ty), mask=m_c[:, None])

    # Invert unit-lower-triangular (I_16 + b_L)^-1 in registers via forward substitution
    b_Ai = -b_L
    for r in range(2, BC):
        row_mask = o_i == r
        row_l = tl.sum(tl.where(row_mask[:, None], b_L, 0.0), axis=0)  # [BC]
        # new_row[c] = -row_l[c] - sum_{m} row_l[m] * b_Ai[m, c]
        upd = -row_l - tl.sum(row_l[:, None] * b_Ai, axis=0)
        b_Ai = tl.where(row_mask[:, None], upd[None, :], b_Ai)
    b_Ai = tl.where(m_diag, 1.0, b_Ai)

    akkd_ptrs = Akkd_ptr + (bos + o_c[:, None]) * (HV * BC) + i_hv * BC + o_i[None, :]
    tl.store(akkd_ptrs, b_Ai, mask=m_c[:, None])


@triton.jit
def _kda_zero_spill_inter_subchunk_solve_fwd_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr, Aqk_ptr, Akkd_ptr, Akk_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Kernel 4: Zero-Spill Inter-Subchunk Off-Diagonal & 4x4 Block-Triangular Solve (`0` spills vs `4` spills in upstream).
    Uses exact sub-chunk boundary anchors `g[i_tc_r]` so all off-diagonal exponents are strictly `<= 0`.
    Stores `b_Aqk` blocks immediately before loading `b_Ai00..b_Ai33` to cut peak live registers.
    Grid: `(NT, B * HV)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T

    o_i = tl.arange(0, BC)
    o_c0 = i_t * BT + o_i
    o_c1 = i_t * BT + BC + o_i
    o_c2 = i_t * BT + 2 * BC + o_i
    o_c3 = i_t * BT + 3 * BC + o_i

    q_base = q_ptr + (bos * H + i_h) * K
    k_base = k_ptr + (bos * H + i_h) * K
    g_base = g_ptr + (bos * HV + i_hv) * K
    Aqk_base = Aqk_ptr + (bos * HV + i_hv) * BT
    Akk_base = Akk_ptr + (bos * HV + i_hv) * BT
    Akkd_base = Akkd_ptr + (bos * HV + i_hv) * BC

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        b_k0 = tl.load(k_base + o_c0[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
        b_g0 = tl.load(g_base + o_c0[:, None] * (HV * K) + o_k[None, :]).to(tl.float32)

        b_q1 = tl.load(q_base + o_c1[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
        b_k1 = tl.load(k_base + o_c1[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
        b_g1 = tl.load(g_base + o_c1[:, None] * (HV * K) + o_k[None, :]).to(tl.float32)
        b_gn1 = tl.load(g_base + (i_t * BT + BC) * (HV * K) + o_k).to(tl.float32)
        b_gqn1 = tl.exp(tl.minimum(b_g1 - b_gn1[None, :], 0.0))
        b_kgt0_1 = tl.trans(b_k0 * tl.exp(tl.minimum(b_gn1[None, :] - b_g0, 0.0)))
        b_Aqk10 = tl.dot(b_q1 * b_gqn1, b_kgt0_1, b_Aqk10, input_precision="ieee")
        b_Akk10 = tl.dot(b_k1 * b_gqn1, b_kgt0_1, b_Akk10, input_precision="ieee")

        b_q2 = tl.load(q_base + o_c2[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
        b_k2 = tl.load(k_base + o_c2[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
        b_g2 = tl.load(g_base + o_c2[:, None] * (HV * K) + o_k[None, :]).to(tl.float32)
        b_gn2 = tl.load(g_base + (i_t * BT + 2 * BC) * (HV * K) + o_k).to(tl.float32)
        b_gqn2 = tl.exp(tl.minimum(b_g2 - b_gn2[None, :], 0.0))
        b_qg2 = b_q2 * b_gqn2
        b_kg2 = b_k2 * b_gqn2
        b_kgt0_2 = tl.trans(b_k0 * tl.exp(tl.minimum(b_gn2[None, :] - b_g0, 0.0)))
        b_Aqk20 = tl.dot(b_qg2, b_kgt0_2, b_Aqk20, input_precision="ieee")
        b_Akk20 = tl.dot(b_kg2, b_kgt0_2, b_Akk20, input_precision="ieee")
        b_kgt1_2 = tl.trans(b_k1 * tl.exp(tl.minimum(b_gn2[None, :] - b_g1, 0.0)))
        b_Aqk21 = tl.dot(b_qg2, b_kgt1_2, b_Aqk21, input_precision="ieee")
        b_Akk21 = tl.dot(b_kg2, b_kgt1_2, b_Akk21, input_precision="ieee")

        b_q3 = tl.load(q_base + o_c3[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
        b_k3 = tl.load(k_base + o_c3[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
        b_g3 = tl.load(g_base + o_c3[:, None] * (HV * K) + o_k[None, :]).to(tl.float32)
        b_gn3 = tl.load(g_base + (i_t * BT + 3 * BC) * (HV * K) + o_k).to(tl.float32)
        b_gqn3 = tl.exp(tl.minimum(b_g3 - b_gn3[None, :], 0.0))
        b_qg3 = b_q3 * b_gqn3
        b_kg3 = b_k3 * b_gqn3
        b_kgt0_3 = tl.trans(b_k0 * tl.exp(tl.minimum(b_gn3[None, :] - b_g0, 0.0)))
        b_Aqk30 = tl.dot(b_qg3, b_kgt0_3, b_Aqk30, input_precision="ieee")
        b_Akk30 = tl.dot(b_kg3, b_kgt0_3, b_Akk30, input_precision="ieee")
        b_kgt1_3 = tl.trans(b_k1 * tl.exp(tl.minimum(b_gn3[None, :] - b_g1, 0.0)))
        b_Aqk31 = tl.dot(b_qg3, b_kgt1_3, b_Aqk31, input_precision="ieee")
        b_Akk31 = tl.dot(b_kg3, b_kgt1_3, b_Akk31, input_precision="ieee")
        b_kgt2_3 = tl.trans(b_k2 * tl.exp(tl.minimum(b_gn3[None, :] - b_g2, 0.0)))
        b_Aqk32 = tl.dot(b_qg3, b_kgt2_3, b_Aqk32, input_precision="ieee")
        b_Akk32 = tl.dot(b_kg3, b_kgt2_3, b_Akk32, input_precision="ieee")

    # Store off-diagonal Aqk blocks immediately so their registers are freed before loading b_Ai00..b_Ai33
    tl.store(Aqk_base + o_c1[:, None] * (HV * BT) + o_i[None, :], (b_Aqk10 * scale).to(Aqk_ptr.dtype.element_ty))
    tl.store(Aqk_base + o_c2[:, None] * (HV * BT) + o_i[None, :], (b_Aqk20 * scale).to(Aqk_ptr.dtype.element_ty))
    tl.store(Aqk_base + o_c2[:, None] * (HV * BT) + (o_i + BC)[None, :], (b_Aqk21 * scale).to(Aqk_ptr.dtype.element_ty))
    tl.store(Aqk_base + o_c3[:, None] * (HV * BT) + o_i[None, :], (b_Aqk30 * scale).to(Aqk_ptr.dtype.element_ty))
    tl.store(Aqk_base + o_c3[:, None] * (HV * BT) + (o_i + BC)[None, :], (b_Aqk31 * scale).to(Aqk_ptr.dtype.element_ty))
    tl.store(Aqk_base + o_c3[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :], (b_Aqk32 * scale).to(Aqk_ptr.dtype.element_ty))

    b_b1 = tl.load(beta_ptr + (bos + o_c1) * HV + i_hv).to(tl.float32)
    b_b2 = tl.load(beta_ptr + (bos + o_c2) * HV + i_hv).to(tl.float32)
    b_b3 = tl.load(beta_ptr + (bos + o_c3) * HV + i_hv).to(tl.float32)

    b_Akk10 = b_Akk10 * b_b1[:, None]
    b_Akk20 = b_Akk20 * b_b2[:, None]
    b_Akk21 = b_Akk21 * b_b2[:, None]
    b_Akk30 = b_Akk30 * b_b3[:, None]
    b_Akk31 = b_Akk31 * b_b3[:, None]
    b_Akk32 = b_Akk32 * b_b3[:, None]

    b_Ai00 = tl.load(Akkd_base + o_c0[:, None] * (HV * BC) + o_i[None, :]).to(tl.float32)
    b_Ai11 = tl.load(Akkd_base + o_c1[:, None] * (HV * BC) + o_i[None, :]).to(tl.float32)
    b_Ai22 = tl.load(Akkd_base + o_c2[:, None] * (HV * BC) + o_i[None, :]).to(tl.float32)
    b_Ai33 = tl.load(Akkd_base + o_c3[:, None] * (HV * BC) + o_i[None, :]).to(tl.float32)

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10, input_precision="ieee"), b_Ai00, input_precision="ieee")
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_Akk21, input_precision="ieee"), b_Ai11, input_precision="ieee")
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision="ieee") + tl.dot(b_Akk21, b_Ai10, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_Akk32, input_precision="ieee"), b_Ai22, input_precision="ieee")
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision="ieee") + tl.dot(b_Akk32, b_Ai21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision="ieee")
        + tl.dot(b_Akk31, b_Ai10, input_precision="ieee")
        + tl.dot(b_Akk32, b_Ai20, input_precision="ieee"),
        input_precision="ieee",
    )

    tl.store(Akk_base + o_c0[:, None] * (HV * BT) + o_i[None, :], b_Ai00.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c1[:, None] * (HV * BT) + o_i[None, :], b_Ai10.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c1[:, None] * (HV * BT) + (o_i + BC)[None, :], b_Ai11.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c2[:, None] * (HV * BT) + o_i[None, :], b_Ai20.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c2[:, None] * (HV * BT) + (o_i + BC)[None, :], b_Ai21.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c2[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :], b_Ai22.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c3[:, None] * (HV * BT) + o_i[None, :], b_Ai30.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c3[:, None] * (HV * BT) + (o_i + BC)[None, :], b_Ai31.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c3[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :], b_Ai32.to(Akk_ptr.dtype.element_ty))
    tl.store(Akk_base + o_c3[:, None] * (HV * BT) + (o_i + 3 * BC)[None, :], b_Ai33.to(Akk_ptr.dtype.element_ty))


@triton.jit
def _kda_wy_wu_qg_kg_fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_ptr,
    w_ptr, u_ptr, qg_ptr, kg_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 5: Row-Subtiled (`BM = 16`) UT-transformed WY vectors `W_c = A (beta * exp(g) * K)`,
    `U_c = A (beta * V)`, `Q_{g,c} = exp(g) * Q`, and `J_c = exp(g_{BT-1} - g) * K` (`0` spills).
    Grid: `(NT * (BT // BM), B * HV)`.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NSUB = BT // BM
    i_t = pid0 // NSUB
    i_sub = pid0 % NSUB

    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T

    o_m = i_t * BT + i_sub * BM + tl.arange(0, BM)
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)

    b_b_all = tl.load(beta_ptr + (bos + o_t) * HV + i_hv).to(tl.float32)
    b_A_sub = tl.load(A_ptr + (bos + o_m[:, None]) * (HV * BT) + i_hv * BT + o_A[None, :]).to(tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        v_all_off = (bos + o_t[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
        v_sub_off = (bos + o_m[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
        b_v = tl.load(v_ptr + v_all_off).to(tl.float32)
        b_vb = b_v * b_b_all[:, None]
        b_u = tl.dot(b_A_sub, b_vb, input_precision="ieee")
        tl.store(u_ptr + v_sub_off, b_u.to(u_ptr.dtype.element_ty))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        qk_sub_off = (bos + o_m[:, None]) * (H * K) + i_h * K + o_k[None, :]
        hv_sub_off = (bos + o_m[:, None]) * (HV * K) + i_hv * K + o_k[None, :]
        qk_all_off = (bos + o_t[:, None]) * (H * K) + i_h * K + o_k[None, :]
        hv_all_off = (bos + o_t[:, None]) * (HV * K) + i_hv * K + o_k[None, :]

        b_q_sub = tl.load(q_ptr + qk_sub_off).to(tl.float32)
        b_k_sub = tl.load(k_ptr + qk_sub_off).to(tl.float32)
        b_g_sub = tl.load(g_ptr + hv_sub_off).to(tl.float32)
        b_gn = tl.load(g_ptr + (bos + i_t * BT + BT - 1) * (HV * K) + i_hv * K + o_k).to(tl.float32)

        b_qg = b_q_sub * tl.exp(b_g_sub)
        b_kg = b_k_sub * tl.exp(tl.minimum(b_gn[None, :] - b_g_sub, 0.0))
        tl.store(qg_ptr + hv_sub_off, b_qg.to(qg_ptr.dtype.element_ty))
        tl.store(kg_ptr + hv_sub_off, b_kg.to(kg_ptr.dtype.element_ty))

        b_k_all = tl.load(k_ptr + qk_all_off).to(tl.float32)
        b_g_all = tl.load(g_ptr + hv_all_off).to(tl.float32)
        b_kb = b_k_all * tl.exp(b_g_all) * b_b_all[:, None]
        b_w = tl.dot(b_A_sub, b_kb, input_precision="ieee")
        tl.store(w_ptr + hv_sub_off, b_w.to(w_ptr.dtype.element_ty))


@triton.jit
def _kda_chunk_state_h_fwd_kernel(
    kg_ptr, w_ptr, u_ptr, g_ptr, h0_ptr,
    h_all_ptr, v_new_ptr, s_cur_ptr,
    T: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_H0: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    """
    Kernel 6: L2-Staged Zero-Spill Inter-chunk state recurrence (`BK = 32, BV = 32`, `0` spills):
      `v_new_c = U_c - W_c @ S_{c-1}`
      `S_c = Diag(exp(g_{c,BT-1})) S_{c-1} + J_c^T @ v_new_c`
    Grid: `(cdiv(V, BV), B * HV)`.
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    NT = T // BT
    bos = i_b * T
    o_v = i_v * BV + tl.arange(0, BV)
    s_base = (i_b * HV + i_hv) * (K * V)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        kv_off = o_k[:, None] * V + o_v[None, :]
        if HAS_H0:
            b_s = tl.load(h0_ptr + s_base + kv_off).to(tl.float32)
        else:
            b_s = tl.zeros([BK, BV], dtype=tl.float32)
        tl.store(s_cur_ptr + s_base + kv_off, b_s)

    o_t = tl.arange(0, BT)
    for i_t in range(NT):
        t_rows = bos + i_t * BT + o_t
        h_base = ((i_b * NT + i_t) * HV + i_hv) * (K * V)
        v_off = t_rows[:, None] * (HV * V) + i_hv * V + o_v[None, :]
        b_v_new = tl.load(u_ptr + v_off).to(tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            kv_off = o_k[:, None] * V + o_v[None, :]
            b_s = tl.load(s_cur_ptr + s_base + kv_off).to(tl.float32)
            tl.store(h_all_ptr + h_base + kv_off, b_s.to(h_all_ptr.dtype.element_ty))
            w_off = t_rows[:, None] * (HV * K) + i_hv * K + o_k[None, :]
            b_w = tl.load(w_ptr + w_off)
            if IS_FP32:
                b_v_new = b_v_new - tl.dot(b_w, b_s, input_precision="ieee")
            else:
                b_v_new = b_v_new - tl.dot(b_w, b_s.to(b_w.dtype))

        tl.store(v_new_ptr + v_off, b_v_new.to(v_new_ptr.dtype.element_ty))
        last_row = bos + i_t * BT + BT - 1

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            kv_off = o_k[:, None] * V + o_v[None, :]
            b_s = tl.load(s_cur_ptr + s_base + kv_off).to(tl.float32)
            b_gn = tl.load(g_ptr + last_row * (HV * K) + i_hv * K + o_k).to(tl.float32)
            w_off = t_rows[:, None] * (HV * K) + i_hv * K + o_k[None, :]
            b_kg = tl.load(kg_ptr + w_off)
            if IS_FP32:
                b_s = b_s * tl.exp(b_gn)[:, None] + tl.dot(tl.trans(b_kg), b_v_new, input_precision="ieee")
            else:
                b_s = b_s * tl.exp(b_gn)[:, None] + tl.dot(tl.trans(b_kg), b_v_new.to(b_kg.dtype))
            tl.store(s_cur_ptr + s_base + kv_off, b_s)


@triton.jit
def _kda_chunk_readout_o_fwd_kernel(
    qg_ptr, v_new_ptr, Aqk_ptr, h_all_ptr, o_ptr,
    scale,
    T: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 7: Parallel chunk output readout (`0` spills):
      `O_c = scale * Q_{g,c} S_{c-1} + A_{qk,c} V_{new,c}`
    Grid: `(cdiv(V, BV), NT, B * HV)`.
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    NT = T // BT
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    h_base = ((i_b * NT + i_t) * HV + i_hv) * (K * V)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        b_qg = tl.load(qg_ptr + (bos + o_t[:, None]) * (HV * K) + i_hv * K + o_k[None, :])
        b_h = tl.load(h_all_ptr + h_base + o_k[:, None] * V + o_v[None, :])
        b_o = tl.dot(b_qg, b_h.to(b_qg.dtype), b_o, input_precision="ieee")

    b_o = b_o * scale
    b_Aqk = tl.load(Aqk_ptr + (bos + o_t[:, None]) * (HV * BT) + i_hv * BT + o_A[None, :])
    m_tril = o_A[:, None] >= o_A[None, :]
    b_Aqk = tl.where(m_tril, b_Aqk, 0.0).to(v_new_ptr.dtype.element_ty)

    v_off = (bos + o_t[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
    b_vn = tl.load(v_new_ptr + v_off)
    b_o = tl.dot(b_Aqk, b_vn, b_o, input_precision="ieee")
    tl.store(o_ptr + v_off, b_o.to(o_ptr.dtype.element_ty))


@triton.jit
def _kda_bwd_dAv_kernel(
    v_new_ptr, Aqk_ptr, do_ptr, dv_intra_ptr, dAqk_ptr,
    scale,
    T: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 8: Row-Subtiled (`BM = 16, BV = 32`) Zero-Spill `dAqk = scale * Tril(dO @ V_new^T)`
    and `dV_new_intra = Aqk^T @ dO` (`0` spills in both FP32 and BF16).
    Grid: `(NT * (BT // BM), B * HV)`.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NSUB = BT // BM
    i_t = pid0 // NSUB
    i_sub = pid0 % NSUB

    i_b = i_bh // HV
    i_hv = i_bh % HV
    bos = i_b * T

    o_r = i_sub * BM + tl.arange(0, BM)
    o_m = i_t * BT + o_r
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)

    # Load Aqk^T[o_r, :]: Aqk[o_t, o_r] -> shape [BM, BT] = [16, 64]
    p_Aqk_T = Aqk_ptr + (bos + o_t[None, :]) * (HV * BT) + i_hv * BT + o_r[:, None]
    b_Aqk_T = tl.load(p_Aqk_T).to(tl.float32)
    m_upper = o_r[:, None] <= o_A[None, :]
    b_Aqk_T = tl.where(m_upper, b_Aqk_T, 0.0)

    b_dA = tl.zeros([BM, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        v_sub_off = (bos + o_m[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
        v_all_off = (bos + o_t[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
        b_do_sub = tl.load(do_ptr + v_sub_off).to(tl.float32)
        b_vn_all = tl.load(v_new_ptr + v_all_off).to(tl.float32)
        b_dA = tl.dot(b_do_sub, tl.trans(b_vn_all), b_dA, input_precision="ieee")

        b_do_all = tl.load(do_ptr + v_all_off).to(tl.float32)
        b_dv_sub = tl.dot(b_Aqk_T, b_do_all, input_precision="ieee")
        tl.store(dv_intra_ptr + v_sub_off, b_dv_sub.to(dv_intra_ptr.dtype.element_ty))

    m_tril = o_r[:, None] >= o_A[None, :]
    b_dA = tl.where(m_tril, b_dA * scale, 0.0)
    dAqk_off = (bos + o_m[:, None]) * (HV * BT) + i_hv * BT + o_A[None, :]
    tl.store(dAqk_ptr + dAqk_off, b_dA)


@triton.jit
def _kda_bwd_dhu_reverse_kernel(
    qg_ptr, kg_ptr, w_ptr, g_ptr, do_ptr, dv_intra_ptr, dht_ptr,
    dh_all_ptr, dv_total_ptr, ds_cur_ptr,
    scale,
    T: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_DHT: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    """
    Kernel 9: L2-Staged Zero-Spill Reverse inter-chunk state adjoint scan (`BK = 32, BV = 32`, `0` spills):
      `dV_new_c = dV_new_intra_c + J_c @ dS_c`
      `dS_{c-1} = Diag(exp(g_{c,BT-1})) dS_c + scale * Q_{g,c}^T @ dO_c - W_c^T @ dV_new_c`
    Grid: `(cdiv(V, BV), B * HV)`.
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    NT = T // BT
    bos = i_b * T
    o_v = i_v * BV + tl.arange(0, BV)
    s_base = (i_b * HV + i_hv) * (K * V)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        kv_off = o_k[:, None] * V + o_v[None, :]
        if HAS_DHT:
            b_ds = tl.load(dht_ptr + s_base + kv_off).to(tl.float32)
        else:
            b_ds = tl.zeros([BK, BV], dtype=tl.float32)
        tl.store(ds_cur_ptr + s_base + kv_off, b_ds)

    o_t = tl.arange(0, BT)
    for step in range(NT):
        i_t = NT - 1 - step
        t_rows = bos + i_t * BT + o_t
        dh_base = ((i_b * NT + i_t) * HV + i_hv) * (K * V)
        v_off = t_rows[:, None] * (HV * V) + i_hv * V + o_v[None, :]
        b_dv = tl.load(dv_intra_ptr + v_off).to(tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            kv_off = o_k[:, None] * V + o_v[None, :]
            b_ds = tl.load(ds_cur_ptr + s_base + kv_off).to(tl.float32)
            tl.store(dh_all_ptr + dh_base + kv_off, b_ds.to(dh_all_ptr.dtype.element_ty))
            k_off = t_rows[:, None] * (HV * K) + i_hv * K + o_k[None, :]
            b_kg = tl.load(kg_ptr + k_off)
            if IS_FP32:
                b_dv = b_dv + tl.dot(b_kg, b_ds, input_precision="ieee")
            else:
                b_dv = b_dv + tl.dot(b_kg, b_ds.to(b_kg.dtype))

        tl.store(dv_total_ptr + v_off, b_dv.to(dv_total_ptr.dtype.element_ty))
        b_do = tl.load(do_ptr + v_off).to(tl.float32) * scale
        last_row = bos + i_t * BT + BT - 1

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            kv_off = o_k[:, None] * V + o_v[None, :]
            b_ds = tl.load(ds_cur_ptr + s_base + kv_off).to(tl.float32)
            b_gn = tl.load(g_ptr + last_row * (HV * K) + i_hv * K + o_k).to(tl.float32)
            k_off = t_rows[:, None] * (HV * K) + i_hv * K + o_k[None, :]
            b_qg = tl.load(qg_ptr + k_off)
            b_w = tl.load(w_ptr + k_off)
            if IS_FP32:
                b_ds = b_ds * tl.exp(b_gn)[:, None] + tl.dot(tl.trans(b_qg), b_do, input_precision="ieee") - tl.dot(tl.trans(b_w), b_dv, input_precision="ieee")
            else:
                b_ds = b_ds * tl.exp(b_gn)[:, None] + tl.dot(tl.trans(b_qg), b_do.to(b_qg.dtype)) - tl.dot(tl.trans(b_w), b_dv.to(b_w.dtype))
            tl.store(ds_cur_ptr + s_base + kv_off, b_ds)


@triton.jit
def _kda_bwd_wy_dqkg_slice_kernel(
    q_ptr, k_ptr, v_new_ptr, g_ptr, h_all_ptr, do_ptr, dh_all_ptr, dv_total_ptr,
    dq_inter_ptr, dk_inter_ptr, dg_inter_ptr, dw_neg_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 10: Zero-Spill Split-`K` WY Slice Backward Kernel (`BK = 32, BV = 32`).
    Resolves Upstream Pathology P2 (`156-3,648` register spills across `36/36` configs
    of `chunk_kda_bwd_kernel_wy_dqkg_fused`) by computing per-slice `(dq, dk, dg, dw_neg)`
    without holding `[64, 64]` `b_A` and `b_dA` during the `i_v` loop (`0` spills!).
    Grid: `(NT, cdiv(K, BK), B * HV)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    NT = T // BT
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)

    qk_off = (bos + o_t[:, None]) * (H * K) + i_h * K + o_k[None, :]
    hv_off = (bos + o_t[:, None]) * (HV * K) + i_hv * K + o_k[None, :]
    h_base = ((i_b * NT + i_t) * HV + i_hv) * (K * V)

    b_q = tl.load(q_ptr + qk_off).to(tl.float32)
    b_k = tl.load(k_ptr + qk_off).to(tl.float32)
    b_g = tl.load(g_ptr + hv_off).to(tl.float32)
    b_gn = tl.load(g_ptr + (bos + i_t * BT + BT - 1) * (HV * K) + i_hv * K + o_k).to(tl.float32)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        v_off = (bos + o_t[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
        h_off = h_base + o_k[:, None] * V + o_v[None, :]

        b_do = tl.load(do_ptr + v_off)
        b_vn = tl.load(v_new_ptr + v_off)
        b_dv = tl.load(dv_total_ptr + v_off)
        b_h = tl.load(h_all_ptr + h_off)
        b_dh = tl.load(dh_all_ptr + h_off)

        b_dgk += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32), axis=1)
        b_h_T = tl.trans(b_h)
        b_dh_T = tl.trans(b_dh)
        b_dq = tl.dot(b_do, b_h_T.to(b_do.dtype), b_dq, input_precision="ieee")
        b_dk = tl.dot(b_vn, b_dh_T.to(b_vn.dtype), b_dk, input_precision="ieee")
        b_dw = tl.dot(b_dv.to(b_do.dtype), b_h_T.to(b_do.dtype), b_dw, input_precision="ieee")

    b_exp_g = tl.exp(b_g)
    b_dgk = b_dgk * tl.exp(b_gn)
    b_dq = b_dq * b_exp_g * scale
    b_dk = b_dk * tl.exp(tl.minimum(b_gn[None, :] - b_g, 0.0))

    b_kdk = b_k * b_dk
    b_dgk = b_dgk + tl.sum(b_kdk, axis=0)
    m_last = (o_t == (i_t * BT + BT - 1))[:, None]
    b_dg = b_q * b_dq - b_kdk + tl.where(m_last, b_dgk[None, :], 0.0)

    tl.store(dq_inter_ptr + hv_off, b_dq)
    tl.store(dk_inter_ptr + hv_off, b_dk)
    tl.store(dg_inter_ptr + hv_off, b_dg)
    tl.store(dw_neg_ptr + hv_off, (-b_dw).to(dw_neg_ptr.dtype.element_ty))


@triton.jit
def _kda_bwd_wy_dAkk_dv_db_kernel(
    k_ptr, v_ptr, g_ptr, beta_ptr, A_ptr, dv_total_ptr, dw_neg_ptr,
    dk_inter_ptr, dg_inter_ptr, dv_out_ptr, db_wy_ptr, dA_raw_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 11a: Row-Subtiled (`BM = 16`) Zero-Spill WY Adjoint `dA_raw + dV + dBeta_wy` Kernel (`0` spills).
    Grid: `(NT * (BT // BM), B * HV)`.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NSUB = BT // BM
    i_t = pid0 // NSUB
    i_sub = pid0 % NSUB

    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T

    o_r = i_sub * BM + tl.arange(0, BM)
    o_m = i_t * BT + o_r
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)

    b_beta_sub = tl.load(beta_ptr + (bos + o_m) * HV + i_hv).to(tl.float32)
    b_beta_all = tl.load(beta_ptr + (bos + o_t) * HV + i_hv).to(tl.float32)

    # Load A^T[o_r, :]: A[o_t, o_r] -> shape [BM, BT] = [16, 64]
    p_A_T = A_ptr + (bos + o_t[None, :]) * (HV * BT) + i_hv * BT + o_r[:, None]
    b_A_T_sub = tl.load(p_A_T).to(tl.float32)

    b_dA_sub = tl.zeros([BM, BT], dtype=tl.float32)
    b_db_sub = tl.zeros([BM], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        v_sub_off = (bos + o_m[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
        v_all_off = (bos + o_t[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
        b_v_sub = tl.load(v_ptr + v_sub_off).to(tl.float32)
        b_v_all = tl.load(v_ptr + v_all_off).to(tl.float32)
        b_dv_sub = tl.load(dv_total_ptr + v_sub_off).to(tl.float32)
        b_dv_all = tl.load(dv_total_ptr + v_all_off).to(tl.float32)

        b_dA_sub = tl.dot(b_dv_sub, tl.trans(b_v_all), b_dA_sub, input_precision="ieee")
        b_dvb_sub = tl.dot(b_A_T_sub, b_dv_all, input_precision="ieee")
        b_db_sub += tl.sum(b_dvb_sub * b_v_sub, axis=1)
        tl.store(dv_out_ptr + v_sub_off, (b_dvb_sub * b_beta_sub[:, None]).to(dv_out_ptr.dtype.element_ty))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        qk_sub_off = (bos + o_m[:, None]) * (H * K) + i_h * K + o_k[None, :]
        hv_sub_off = (bos + o_m[:, None]) * (HV * K) + i_hv * K + o_k[None, :]
        qk_all_off = (bos + o_t[:, None]) * (H * K) + i_h * K + o_k[None, :]
        hv_all_off = (bos + o_t[:, None]) * (HV * K) + i_hv * K + o_k[None, :]

        b_k_sub = tl.load(k_ptr + qk_sub_off).to(tl.float32)
        b_g_sub = tl.load(g_ptr + hv_sub_off).to(tl.float32)
        b_dw_sub = tl.load(dw_neg_ptr + hv_sub_off).to(tl.float32)
        b_exp_g_sub = tl.exp(b_g_sub)
        b_kg_sub = b_k_sub * b_exp_g_sub

        b_k_all = tl.load(k_ptr + qk_all_off).to(tl.float32)
        b_g_all = tl.load(g_ptr + hv_all_off).to(tl.float32)
        b_dw_all = tl.load(dw_neg_ptr + hv_all_off).to(tl.float32)
        b_kg_all = b_k_all * tl.exp(b_g_all)

        b_dA_sub = tl.dot(b_dw_sub, tl.trans(b_kg_all), b_dA_sub, input_precision="ieee")
        b_dkgb_sub = tl.dot(b_A_T_sub, b_dw_all, input_precision="ieee")
        b_db_sub += tl.sum(b_dkgb_sub * b_kg_sub, axis=1)

        b_dk_prev = tl.load(dk_inter_ptr + hv_sub_off).to(tl.float32)
        b_dg_prev = tl.load(dg_inter_ptr + hv_sub_off).to(tl.float32)
        b_gb_sub = b_exp_g_sub * b_beta_sub[:, None]
        tl.store(dk_inter_ptr + hv_sub_off, b_dk_prev + b_dkgb_sub * b_gb_sub)
        tl.store(dg_inter_ptr + hv_sub_off, b_dg_prev + b_kg_sub * b_dkgb_sub * b_beta_sub[:, None])

    m_strict = o_r[:, None] > o_A[None, :]
    b_dA_sub = tl.where(m_strict, b_dA_sub * b_beta_all[None, :], 0.0)
    dA_off = (bos + o_m[:, None]) * (HV * BT) + i_hv * BT + o_A[None, :]
    tl.store(dA_raw_ptr + dA_off, b_dA_sub)
    tl.store(db_wy_ptr + (bos + o_m) * HV + i_hv, b_db_sub)


@triton.jit
def _kda_bwd_wy_dAkk_solve_kernel(
    A_ptr, dA_raw_ptr, dAkk_ptr,
    T: tl.constexpr,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
):
    """
    Kernel 11b: Block-Subtiled (`16 x 16`) Zero-Spill Inverse-Triangular Adjoint Solve:
      `dAkk = -StrictTril(A^T @ dA_raw @ A^T)` (`0` spills).
    Grid: `(NT * (BT // BM), BT // BM, B * HV)`.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    j_sub = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NSUB = BT // BM
    i_t = pid0 // NSUB
    i_sub = pid0 % NSUB

    i_b = i_bh // HV
    i_hv = i_bh % HV
    bos = i_b * T

    o_r = i_sub * BM + tl.arange(0, BM)
    o_c = j_sub * BM + tl.arange(0, BM)
    o_m = i_t * BT + o_r
    o_n = i_t * BT + o_c
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)

    if i_sub < j_sub:
        b_zero = tl.zeros([BM, BM], dtype=tl.float32)
        tl.store(dAkk_ptr + (bos + o_m[:, None]) * (HV * BT) + i_hv * BT + o_c[None, :], b_zero)
        return

    # Load A^T[:, o_c] = A[o_n, o_A]^T -> shape [BT, BM] = [64, 16]
    b_AT_col = tl.load(A_ptr + (bos + o_n[None, :]) * (HV * BT) + i_hv * BT + o_A[:, None]).to(tl.float32)
    b_acc = tl.zeros([BM, BM], dtype=tl.float32)

    for k_sub in range(NSUB):
        o_k = k_sub * BM + tl.arange(0, BM)
        o_kt = i_t * BT + o_k
        # dA_raw[o_k, :] -> [16, 64]
        b_dA_blk = tl.load(dA_raw_ptr + (bos + o_kt[:, None]) * (HV * BT) + i_hv * BT + o_A[None, :]).to(tl.float32)
        # mid = dA_raw[o_k, :] @ A^T[:, o_c] -> [16, 16]
        b_mid = tl.dot(b_dA_blk, b_AT_col, input_precision="ieee")
        # A^T[o_r, o_k] = A[o_kt, o_r] -> [16, 16]
        b_AT_ik = tl.load(A_ptr + (bos + o_kt[None, :]) * (HV * BT) + i_hv * BT + o_r[:, None]).to(tl.float32)
        b_acc = tl.dot(b_AT_ik, b_mid, b_acc, input_precision="ieee")

    m_strict = o_r[:, None] > o_c[None, :]
    b_acc = tl.where(m_strict, -b_acc, 0.0)
    tl.store(dAkk_ptr + (bos + o_m[:, None]) * (HV * BT) + i_hv * BT + o_c[None, :], b_acc)


@triton.jit
def _kda_bwd_intra_subchunk_dqkgb_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr, dAqk_ptr, dAkk_ptr,
    dq_inter_ptr, dk_inter_ptr, dg_inter_ptr,
    dq_out_ptr, dk_out_ptr, dg_out_ptr, db_part_ptr,
    B,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    """
    Kernel 12: Zero-Spill Intra-Chunk Backward Kernel (`0` spills vs `30` spills in upstream
    `chunk_kda_bwd_kernel_intra`).
    Uses pre-masked non-positive causal differences `exp(min(g_i - g_j, 0))` across both
    off-diagonal and diagonal 16x16 sub-chunks.
    Grid: `(NT * NC, cdiv(K, BK), B * HV)`.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_t = pid0 // NC
    i_sub = pid0 % NC

    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T

    t_sub = i_t * BT + i_sub * BC
    o_i = tl.arange(0, BC)
    o_c = t_sub + o_i
    o_k = i_k * BK + tl.arange(0, BK)

    qk_off = (bos + o_c[:, None]) * (H * K) + i_h * K + o_k[None, :]
    hv_off = (bos + o_c[:, None]) * (HV * K) + i_hv * K + o_k[None, :]

    b_q = tl.load(q_ptr + qk_off).to(tl.float32)
    b_k = tl.load(k_ptr + qk_off).to(tl.float32)
    b_g = tl.load(g_ptr + hv_off).to(tl.float32)
    b_b = tl.load(beta_ptr + (bos + o_c) * HV + i_hv).to(tl.float32)

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

    # 1. Off-diagonal query-side sub-chunks (i_j < i_sub)
    if i_sub > 0:
        b_gn = tl.load(g_ptr + (bos + t_sub) * (HV * K) + i_hv * K + o_k).to(tl.float32)[None, :]
        for i_j in range(0, i_sub):
            o_j = i_t * BT + i_j * BC + o_i
            b_kj = tl.load(k_ptr + (bos + o_j[:, None]) * (H * K) + i_h * K + o_k[None, :]).to(tl.float32)
            b_gj = tl.load(g_ptr + (bos + o_j[:, None]) * (HV * K) + i_hv * K + o_k[None, :]).to(tl.float32)
            b_kg = b_kj * tl.exp(tl.minimum(b_gn - b_gj, 0.0))

            dA_off = (bos + o_c[:, None]) * (HV * BT) + i_hv * BT + (i_j * BC + o_i)[None, :]
            b_dAqk = tl.load(dAqk_ptr + dA_off).to(tl.float32)
            b_dAkk = tl.load(dAkk_ptr + dA_off).to(tl.float32)
            b_dq2 = tl.dot(b_dAqk, b_kg, b_dq2, input_precision="ieee")
            b_dk2 = tl.dot(b_dAkk, b_kg, b_dk2, input_precision="ieee")

        b_gqn = tl.exp(tl.minimum(b_g - b_gn, 0.0))
        b_dq2 = b_dq2 * b_gqn
        b_dk2 = b_dk2 * b_gqn

    # 2. Diagonal sub-chunk query-side (j <= o_i)
    o_dA_row = (bos + o_c) * (HV * BT) + i_hv * BT + i_sub * BC
    for j in range(BC):
        b_dAqk_j = tl.load(dAqk_ptr + o_dA_row + j).to(tl.float32)
        b_dAkk_j = tl.load(dAkk_ptr + o_dA_row + j).to(tl.float32)
        b_kj = tl.load(k_ptr + (bos + t_sub + j) * (H * K) + i_h * K + o_k).to(tl.float32)
        b_gj = tl.load(g_ptr + (bos + t_sub + j) * (HV * K) + i_hv * K + o_k).to(tl.float32)
        m_le = (o_i >= j)[:, None]
        m_lt = (o_i > j)[:, None]
        decay = tl.where(m_le, tl.exp(tl.minimum(b_g - b_gj[None, :], 0.0)), 0.0)
        b_dq2 += tl.where(m_le, b_dAqk_j[:, None] * b_kj[None, :] * decay, 0.0)
        b_dk2 += tl.where(m_lt, b_dAkk_j[:, None] * b_kj[None, :] * decay, 0.0)

    b_db_sub = tl.sum(b_dk2 * b_k, axis=1)
    b_dk2 = b_dk2 * b_b[:, None]
    b_dg2 = b_q * b_dq2

    tl.store(dq_out_ptr + hv_off, b_dq2 + tl.load(dq_inter_ptr + hv_off).to(tl.float32))
    tl.store(db_part_ptr + (i_k * B * T + bos + o_c) * HV + i_hv, b_db_sub)

    # 3. Off-diagonal key-side sub-chunks (i_j > i_sub)
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
    if i_sub < NC - 1:
        b_gn_end = tl.load(g_ptr + (bos + t_sub + BC - 1) * (HV * K) + i_hv * K + o_k).to(tl.float32)[None, :]
        for i_j in range(i_sub + 1, NC):
            o_j = i_t * BT + i_j * BC + o_i
            b_qj = tl.load(q_ptr + (bos + o_j[:, None]) * (H * K) + i_h * K + o_k[None, :]).to(tl.float32)
            b_kj = tl.load(k_ptr + (bos + o_j[:, None]) * (H * K) + i_h * K + o_k[None, :]).to(tl.float32)
            b_gj = tl.load(g_ptr + (bos + o_j[:, None]) * (HV * K) + i_hv * K + o_k[None, :]).to(tl.float32)
            b_bj = tl.load(beta_ptr + (bos + o_j) * HV + i_hv).to(tl.float32)

            # Transpose load: dA[o_j, i_sub * BC + o_i]
            dA_T_off = (bos + o_j[None, :]) * (HV * BT) + i_hv * BT + (i_sub * BC + o_i)[:, None]
            b_dAqk_T = tl.load(dAqk_ptr + dA_T_off).to(tl.float32)
            b_dAkk_T = tl.load(dAkk_ptr + dA_T_off).to(tl.float32)

            b_gkn = tl.exp(tl.minimum(b_gj - b_gn_end, 0.0))
            b_dkt = tl.dot(b_dAqk_T, b_qj * b_gkn, b_dkt, input_precision="ieee")
            b_dkt = tl.dot(b_dAkk_T, (b_kj * b_bj[:, None]) * b_gkn, b_dkt, input_precision="ieee")

        b_dkt = b_dkt * tl.exp(tl.minimum(b_gn_end - b_g, 0.0))

    # 4. Diagonal sub-chunk key-side (j >= o_i)
    o_dA_col_base = (bos + t_sub) * (HV * BT) + i_hv * BT + i_sub * BC + o_i
    for j in range(BC):
        b_dAqk_col = tl.load(dAqk_ptr + o_dA_col_base + j * (HV * BT)).to(tl.float32)
        b_dAkk_col = tl.load(dAkk_ptr + o_dA_col_base + j * (HV * BT)).to(tl.float32)
        b_qj = tl.load(q_ptr + (bos + t_sub + j) * (H * K) + i_h * K + o_k).to(tl.float32)
        b_kj = tl.load(k_ptr + (bos + t_sub + j) * (H * K) + i_h * K + o_k).to(tl.float32)
        b_gj = tl.load(g_ptr + (bos + t_sub + j) * (HV * K) + i_hv * K + o_k).to(tl.float32)
        b_bj = tl.load(beta_ptr + (bos + t_sub + j) * HV + i_hv).to(tl.float32)

        m_ge = (o_i <= j)[:, None]
        m_gt = (o_i < j)[:, None]
        decay = tl.where(m_ge, tl.exp(tl.minimum(b_gj[None, :] - b_g, 0.0)), 0.0)
        b_dkt += tl.where(m_ge, b_dAqk_col[:, None] * b_qj[None, :] * decay, 0.0)
        b_dkt += tl.where(m_gt, b_dAkk_col[:, None] * (b_kj * b_bj)[None, :] * decay, 0.0)

    b_dg_final = b_dg2 + (b_dk2 - b_dkt) * b_k + tl.load(dg_inter_ptr + hv_off).to(tl.float32)
    b_dk_final = b_dk2 + b_dkt + tl.load(dk_inter_ptr + hv_off).to(tl.float32)
    tl.store(dk_out_ptr + hv_off, b_dk_final)
    tl.store(dg_out_ptr + hv_off, b_dg_final)


@triton.jit
def _kda_bwd_fused_rcumsum_gate_kernel(
    g_raw_ptr, A_log_ptr, dt_bias_ptr, dg_cumsum_ptr,
    dg_raw_out_ptr, dA_part_ptr, dbias_part_ptr,
    lower_bound,
    T: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """
    Kernel 13: Fuses reverse chunk-local cumsum `dg_act = cumsum(dg_cumsum, reverse=True)`
    and KDA channel-gate VJP (`dg_raw, dA_log, ddt_bias`) (`0` spills).
    Grid: `(NT * cdiv(K, BS), B * HV)`.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NS = tl.cdiv(K, BS)
    i_t = pid0 // NS
    i_s = pid0 % NS
    NT = T // BT

    i_b = i_bh // HV
    i_hv = i_bh % HV
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m_t = o_t < T
    m_s = o_s < K
    mask = m_t[:, None] & m_s[None, :]

    ptrs = (bos + o_t[:, None]) * (HV * K) + i_hv * K + o_s[None, :]
    b_dyg_cs = tl.load(dg_cumsum_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    # Reverse cumsum via total - exclusive prefix sum
    b_total = tl.sum(b_dyg_cs, axis=0)[None, :]
    b_dyg = b_total - tl.cumsum(b_dyg_cs, axis=0) + b_dyg_cs
    b_dyg = tl.where(mask, b_dyg, 0.0)

    if not USE_GATE_IN_KERNEL:
        tl.store(dg_raw_out_ptr + ptrs, b_dyg.to(dg_raw_out_ptr.dtype.element_ty), mask=mask)
        return

    b_g = tl.load(g_raw_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_bias = tl.load(dt_bias_ptr + i_hv * K + o_s, mask=m_s, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    b_A = tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32)) if HAS_A else 1.0
    if USE_LOWER_BOUND:
        b_inner = b_A * b_g
        b_r = tl.exp(-tl.abs(b_inner))
        b_inv = 1.0 / (1.0 + b_r)
        b_dsig = b_r * (b_inv * b_inv)
        b_dg = b_dyg * (lower_bound * b_dsig * b_A)
        b_dA_val = tl.sum(b_dg * b_g) if HAS_A else 0.0
    else:
        b_sp = tl.maximum(b_g, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(b_g)))
        b_yg = -b_A * b_sp
        b_dg = -b_A * b_dyg * tl.sigmoid(b_g)
        b_dA_val = tl.sum(b_dyg * b_yg) if HAS_A else 0.0

    tl.store(dg_raw_out_ptr + ptrs, b_dg.to(dg_raw_out_ptr.dtype.element_ty), mask=mask)
    chunk_id = (i_b * NT + i_t) * NS + i_s
    if HAS_A:
        tl.store(dA_part_ptr + chunk_id * HV + i_hv, b_dA_val)
    if HAS_BIAS:
        b_dbias = tl.sum(b_dg, axis=0)
        tl.store(dbias_part_ptr + ((i_b * NT + i_t) * HV + i_hv) * K + o_s, b_dbias, mask=m_s)


@triton.jit
def _kda_bwd_qk_l2norm_beta_sigmoid_kernel(
    q_norm_ptr, q_rstd_ptr, k_norm_ptr, k_rstd_ptr, beta_ptr,
    dq_hv_ptr, dk_hv_ptr, db_wy_ptr, db_part_ptr,
    dq_out_ptr, dk_out_ptr, dbeta_raw_out_ptr,
    beta_scale,
    B,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    NK: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    USE_BETA_SIGMOID: tl.constexpr,
):
    """
    Kernel 14: Fuses GVA `(HV -> H)` reduction + Q/K L2Norm radial VJP + `db` NK reduction
    + Beta Sigmoid VJP (`0` spills).
    Grid: `(B * T, H)`.
    """
    bt_idx = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    G = HV // H
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K

    b_dq_sum = tl.zeros([BK], dtype=tl.float32)
    b_dk_sum = tl.zeros([BK], dtype=tl.float32)

    for g_idx in range(G):
        i_hv = i_h * G + g_idx
        row_hv = bt_idx * HV + i_hv
        hv_off = row_hv * K + offs_k
        b_dq_sum += tl.load(dq_hv_ptr + hv_off, mask=mask_k, other=0.0).to(tl.float32)
        b_dk_sum += tl.load(dk_hv_ptr + hv_off, mask=mask_k, other=0.0).to(tl.float32)

        b_db_tot = tl.load(db_wy_ptr + row_hv).to(tl.float32)
        for ik in range(NK):
            b_db_tot += tl.load(db_part_ptr + (ik * B * T + bt_idx) * HV + i_hv).to(tl.float32)

        if USE_BETA_SIGMOID:
            b_val = tl.load(beta_ptr + row_hv).to(tl.float32)
            sig = b_val / beta_scale
            b_db_tot = b_db_tot * beta_scale * sig * (1.0 - sig)
        tl.store(dbeta_raw_out_ptr + row_hv, b_db_tot.to(dbeta_raw_out_ptr.dtype.element_ty))

    row_h = bt_idx * H + i_h
    h_off = row_h * K + offs_k
    if USE_QK_L2NORM:
        b_qn = tl.load(q_norm_ptr + h_off, mask=mask_k, other=0.0).to(tl.float32)
        b_kn = tl.load(k_norm_ptr + h_off, mask=mask_k, other=0.0).to(tl.float32)
        q_rstd = tl.load(q_rstd_ptr + row_h).to(tl.float32)
        k_rstd = tl.load(k_rstd_ptr + row_h).to(tl.float32)
        q_dot = tl.sum(b_dq_sum * b_qn, axis=0)
        k_dot = tl.sum(b_dk_sum * b_kn, axis=0)
        b_dq_sum = (b_dq_sum - b_qn * q_dot) * q_rstd
        b_dk_sum = (b_dk_sum - b_kn * k_dot) * k_rstd

    tl.store(dq_out_ptr + h_off, b_dq_sum.to(dq_out_ptr.dtype.element_ty), mask=mask_k)
    tl.store(dk_out_ptr + h_off, b_dk_sum.to(dk_out_ptr.dtype.element_ty), mask=mask_k)


@triton.jit
def _kda_fused_shortconv_silu_fwd_kernel(
    x_ptr, w_ptr, y_ptr, pre_act_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """
    Kernel 15: Fused 4-tap depthwise causal `ShortConv1d(W=4) + Swish/SiLU` forward (`0` spills).
    Grid: `(cdiv(T, BT), cdiv(D, BD), B)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_d = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t = o_t < T
    m_d = o_d < D
    base_b = i_b * T * D

    w0 = tl.load(w_ptr + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)

    x3 = tl.load(x_ptr + base_b + o_t[:, None] * D + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base_b + (o_t[:, None] - 1) * D + o_d[None, :], mask=(o_t[:, None] >= 1) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base_b + (o_t[:, None] - 2) * D + o_d[None, :], mask=(o_t[:, None] >= 2) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + base_b + (o_t[:, None] - 3) * D + o_d[None, :], mask=(o_t[:, None] >= 3) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)

    pre = x0 * w0[None, :] + x1 * w1[None, :] + x2 * w2[None, :] + x3 * w3[None, :]
    out = pre * tl.sigmoid(pre)
    off = base_b + o_t[:, None] * D + o_d[None, :]
    tl.store(pre_act_ptr + off, pre, mask=m_t[:, None] & m_d[None, :])
    tl.store(y_ptr + off, out.to(y_ptr.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])


@triton.jit
def _kda_fused_shortconv_silu_bwd_kernel(
    x_ptr, w_ptr, pre_act_ptr, dy_ptr, dx_ptr, dw_part_ptr,
    B,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """
    Kernel 16: Fused 4-tap depthwise causal `ShortConv1d(W=4) + Swish/SiLU` backward (`0` spills).
    Grid: `(cdiv(T, BT), cdiv(D, BD), B)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_d = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)
    NT = tl.cdiv(T, BT)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t = o_t < T
    m_d = o_d < D
    base_b = i_b * T * D

    w0 = tl.load(w_ptr + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)

    m0 = m_t[:, None] & m_d[None, :]
    off0 = base_b + o_t[:, None] * D + o_d[None, :]
    pre0 = tl.load(pre_act_ptr + off0, mask=m0, other=0.0).to(tl.float32)
    dy0 = tl.load(dy_ptr + off0, mask=m0, other=0.0).to(tl.float32)
    sig0 = tl.sigmoid(pre0)
    dp0 = tl.where(m0, dy0 * sig0 * (1.0 + pre0 * (1.0 - sig0)), 0.0)

    t1 = o_t + 1
    m1 = (t1[:, None] < T) & m_d[None, :]
    off1 = base_b + t1[:, None] * D + o_d[None, :]
    pre1 = tl.load(pre_act_ptr + off1, mask=m1, other=0.0).to(tl.float32)
    dy1 = tl.load(dy_ptr + off1, mask=m1, other=0.0).to(tl.float32)
    sig1 = tl.sigmoid(pre1)
    dp1 = tl.where(m1, dy1 * sig1 * (1.0 + pre1 * (1.0 - sig1)), 0.0)

    t2 = o_t + 2
    m2 = (t2[:, None] < T) & m_d[None, :]
    off2 = base_b + t2[:, None] * D + o_d[None, :]
    pre2 = tl.load(pre_act_ptr + off2, mask=m2, other=0.0).to(tl.float32)
    dy2 = tl.load(dy_ptr + off2, mask=m2, other=0.0).to(tl.float32)
    sig2 = tl.sigmoid(pre2)
    dp2 = tl.where(m2, dy2 * sig2 * (1.0 + pre2 * (1.0 - sig2)), 0.0)

    t3 = o_t + 3
    m3 = (t3[:, None] < T) & m_d[None, :]
    off3 = base_b + t3[:, None] * D + o_d[None, :]
    pre3 = tl.load(pre_act_ptr + off3, mask=m3, other=0.0).to(tl.float32)
    dy3 = tl.load(dy_ptr + off3, mask=m3, other=0.0).to(tl.float32)
    sig3 = tl.sigmoid(pre3)
    dp3 = tl.where(m3, dy3 * sig3 * (1.0 + pre3 * (1.0 - sig3)), 0.0)

    dx = dp0 * w3[None, :] + dp1 * w2[None, :] + dp2 * w1[None, :] + dp3 * w0[None, :]
    tl.store(dx_ptr + base_b + o_t[:, None] * D + o_d[None, :], dx.to(dx_ptr.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])

    x3 = tl.load(x_ptr + base_b + o_t[:, None] * D + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base_b + (o_t[:, None] - 1) * D + o_d[None, :], mask=(o_t[:, None] >= 1) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base_b + (o_t[:, None] - 2) * D + o_d[None, :], mask=(o_t[:, None] >= 2) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + base_b + (o_t[:, None] - 3) * D + o_d[None, :], mask=(o_t[:, None] >= 3) & m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)

    dw0 = tl.sum(dp0 * x0, axis=0)
    dw1 = tl.sum(dp0 * x1, axis=0)
    dw2 = tl.sum(dp0 * x2, axis=0)
    dw3 = tl.sum(dp0 * x3, axis=0)

    part_idx = (i_b * NT + i_t) * D + o_d
    tl.store(dw_part_ptr + part_idx * 4 + 0, dw0, mask=m_d)
    tl.store(dw_part_ptr + part_idx * 4 + 1, dw1, mask=m_d)
    tl.store(dw_part_ptr + part_idx * 4 + 2, dw2, mask=m_d)
    tl.store(dw_part_ptr + part_idx * 4 + 3, dw3, mask=m_d)


@triton.jit
def _kda_fused_rmsnorm_sigmoid_gate_fwd_kernel(
    x_ptr, z_ptr, weight_ptr, y_ptr, rstd_ptr,
    eps,
    N_ROWS,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 17: Fused Kimi Linear output gate `y = Sigmoid(z) * RMSNorm(x, weight)` forward (`0` spills).
    Grid: `(N_ROWS,)` where `N_ROWS = B * T * HV`.
    """
    row = tl.program_id(0).to(tl.int64)
    if row >= N_ROWS:
        return
    offs = tl.arange(0, BV)
    mask = offs < V
    base = row * V + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / V + eps)
    y = (x * rstd * w) * tl.sigmoid(z)
    tl.store(rstd_ptr + row, rstd)
    tl.store(y_ptr + base, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _kda_fused_rmsnorm_sigmoid_gate_bwd_kernel(
    x_ptr, z_ptr, weight_ptr, rstd_ptr, dy_ptr,
    dx_ptr, dz_ptr, dw_part_ptr,
    N_ROWS,
    V: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Kernel 18: Fused Kimi Linear output gate `y = Sigmoid(z) * RMSNorm(x, weight)` backward (`0` spills).
    Grid: `(cdiv(N_ROWS, BLOCK_R),)`.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs_v = tl.arange(0, BV)
    mask_v = offs_v < V
    w = tl.load(weight_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BV], dtype=tl.float32)

    for r in range(BLOCK_R):
        row = pid * BLOCK_R + r
        if row < N_ROWS:
            base = row * V + offs_v
            x = tl.load(x_ptr + base, mask=mask_v, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + base, mask=mask_v, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + base, mask=mask_v, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + row).to(tl.float32)

            x_hat = x * rstd
            normed = x_hat * w
            sig = tl.sigmoid(z)
            dz = dy * normed * sig * (1.0 - sig)
            d_normed = dy * sig
            dw_acc += d_normed * x_hat
            dx_hat = d_normed * w
            dot = tl.sum(dx_hat * x_hat, axis=0) / V
            dx = (dx_hat - x_hat * dot) * rstd
            tl.store(dx_ptr + base, dx.to(dx_ptr.dtype.element_ty), mask=mask_v)
            tl.store(dz_ptr + base, dz.to(dz_ptr.dtype.element_ty), mask=mask_v)

    tl.store(dw_part_ptr + pid * V + offs_v, dw_acc, mask=mask_v)


# =============================================================================
# 4. Autograd Functions & High-Level Entry Points
# =============================================================================

class _KDAChunkRSIFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        A_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        initial_state: Optional[torch.Tensor],
        scale: float,
        use_qk_l2norm: bool,
        use_gate_in_kernel: bool,
        use_beta_sigmoid: bool,
        allow_neg_eigval: bool,
        lower_bound: Optional[float],
        output_final_state: bool,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        BT = chunk_size
        BC = 16
        NC = BT // BC
        NT = T // BT
        BK_full = triton.next_power_of_2(K)
        beta_scale = 2.0 if allow_neg_eigval else 1.0

        # 1. Superfused QK-L2Norm + Beta Sigmoid
        if use_qk_l2norm or use_beta_sigmoid:
            q_norm = torch.empty_like(q)
            k_norm = torch.empty_like(k)
            q_rstd = torch.empty(B * T * H, device=q.device, dtype=torch.float32)
            k_rstd = torch.empty(B * T * H, device=q.device, dtype=torch.float32)
            beta = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)
            n_rows_hv = B * T * HV
            _kda_superfused_qk_l2norm_beta_fwd_kernel[(n_rows_hv,)](
                q, k, beta_raw,
                q_norm, q_rstd, k_norm, k_rstd, beta,
                beta_scale if use_beta_sigmoid else 1.0,
                eps, n_rows_hv,
                H=H, HV=HV, K=K, BK=BK_full,
                num_warps=4,
            )
            if not use_qk_l2norm:
                q_norm = q
                k_norm = k
            if not use_beta_sigmoid:
                beta = beta_raw.float()
        else:
            q_norm, k_norm = q, k
            q_rstd, k_rstd = None, None
            beta = beta_raw.float()

        # 2. Fused KDA Gate + Chunk-Local Cumsum
        BS_gate = 32
        NS_gate = triton.cdiv(K, BS_gate)
        g_cumsum = torch.empty(B, T, HV, K, device=q.device, dtype=torch.float32)
        _kda_fused_gate_cumsum_fwd_kernel[(NT * NS_gate, B * HV)](
            g_raw, A_log, dt_bias, g_cumsum,
            float(lower_bound) if lower_bound is not None else 0.0,
            T=T, HV=HV, K=K, BT=BT, BS=BS_gate,
            HAS_A=A_log is not None,
            HAS_BIAS=dt_bias is not None,
            USE_GATE_IN_KERNEL=use_gate_in_kernel,
            USE_LOWER_BOUND=lower_bound is not None,
            num_warps=4,
        )

        # 3. Overflow-Free Non-Positive Causal Sub-Chunk Diagonal + 4x4 Block-Triangular Solve
        Aqk = torch.empty(B, T, HV, BT, device=q.device, dtype=torch.float32)
        Akkd = torch.empty(B, T, HV, BC, device=q.device, dtype=torch.float32)
        Akk = torch.zeros(B, T, HV, BT, device=q.device, dtype=torch.float32)

        _kda_overflow_free_intra_diag_subchunk_fwd_kernel[(NT, NC, B * HV)](
            q_norm, k_norm, g_cumsum, beta, Aqk, Akkd,
            scale,
            T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=32,
            num_warps=4, num_stages=1,
        )
        _kda_zero_spill_inter_subchunk_solve_fwd_kernel[(NT, B * HV)](
            q_norm, k_norm, g_cumsum, beta, Aqk, Akkd, Akk,
            scale,
            T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=32,
            num_warps=4, num_stages=1,
        )

        # 4. UT-Transform WY Vectors (w, u, qg, kg)
        w = torch.empty(B, T, HV, K, device=q.device, dtype=q.dtype)
        u = torch.empty_like(v)
        qg = torch.empty(B, T, HV, K, device=q.device, dtype=q.dtype)
        kg = torch.empty(B, T, HV, K, device=q.device, dtype=q.dtype)
        _kda_wy_wu_qg_kg_fwd_kernel[(NT * NC, B * HV)](
            q_norm, k_norm, v, g_cumsum, beta, Akk,
            w, u, qg, kg,
            T=T, H=H, HV=HV, K=K, V=V, BT=BT, BM=BC, BK=32, BV=32,
            num_warps=4, num_stages=1,
        )

        # 5. Inter-Chunk State Recurrence & Output Readout
        BV_h = 32
        h_all = torch.empty(B, NT, HV, K, V, device=q.device, dtype=q.dtype)
        v_new = torch.empty_like(v)
        s_cur = torch.empty(B, HV, K, V, device=q.device, dtype=torch.float32)
        _kda_chunk_state_h_fwd_kernel[(triton.cdiv(V, BV_h), B * HV)](
            kg, w, u, g_cumsum, initial_state,
            h_all, v_new, s_cur,
            T=T, HV=HV, K=K, V=V, BT=BT, BK=32, BV=BV_h,
            HAS_H0=initial_state is not None,
            IS_FP32=(q.dtype == torch.float32),
            num_warps=4, num_stages=1,
        )
        ht = s_cur if output_final_state else None

        o = torch.empty_like(v)
        _kda_chunk_readout_o_fwd_kernel[(triton.cdiv(V, 32), NT, B * HV)](
            qg, v_new, Aqk, h_all, o,
            scale,
            T=T, HV=HV, K=K, V=V, BT=BT, BK=32, BV=32,
            num_warps=4, num_stages=1,
        )

        ctx.save_for_backward(
            q_norm, q_rstd, k_norm, k_rstd, v, g_raw, g_cumsum, beta,
            A_log, dt_bias, initial_state, Aqk, Akk, w, qg, kg, v_new, h_all,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm = use_qk_l2norm
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid = use_beta_sigmoid
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.lower_bound = lower_bound
        ctx.chunk_size = chunk_size
        return o, ht

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        (
            q_norm, q_rstd, k_norm, k_rstd, v, g_raw, g_cumsum, beta,
            A_log, dt_bias, initial_state, Aqk, Akk, w, qg, kg, v_new, h_all,
        ) = ctx.saved_tensors

        B, T, H, K = q_norm.shape
        HV, V = v.shape[2], v.shape[3]
        BT = ctx.chunk_size
        BC = 16
        NC = BT // BC
        NT = T // BT
        BK = 32
        BV = 32
        BV_h = 16
        NK = triton.cdiv(K, BK)

        do = do.contiguous()

        # 1. dAqk & dV_new_intra
        dAqk = torch.empty(B, T, HV, BT, device=q_norm.device, dtype=torch.float32)
        dv_intra = torch.empty_like(v)
        _kda_bwd_dAv_kernel[(NT * NC, B * HV)](
            v_new, Aqk, do, dv_intra, dAqk,
            ctx.scale,
            T=T, HV=HV, V=V, BT=BT, BM=BC, BV=BV,
            num_warps=4, num_stages=1,
        )

        # 2. Reverse State Adjoint Scan (dh_all, dv_total, dh0)
        dh_all = torch.empty_like(h_all)
        dv_total = torch.empty_like(v)
        ds_cur = torch.empty(B, HV, K, V, device=q_norm.device, dtype=torch.float32)
        _kda_bwd_dhu_reverse_kernel[(triton.cdiv(V, BV), B * HV)](
            qg, kg, w, g_cumsum, do, dv_intra, dht,
            dh_all, dv_total, ds_cur,
            ctx.scale,
            T=T, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV,
            HAS_DHT=dht is not None,
            IS_FP32=(q_norm.dtype == torch.float32),
            num_warps=4, num_stages=1,
        )
        dh0 = ds_cur if initial_state is not None else None

        # 3. Zero-Spill Split-K/V WY Backward (Kernels 10, 11a, 11b)
        dq_inter = torch.empty(B, T, HV, K, device=q_norm.device, dtype=torch.float32)
        dk_inter = torch.empty(B, T, HV, K, device=q_norm.device, dtype=torch.float32)
        dg_inter = torch.empty(B, T, HV, K, device=q_norm.device, dtype=torch.float32)
        dw_neg = torch.empty(B, T, HV, K, device=q_norm.device, dtype=q_norm.dtype)

        _kda_bwd_wy_dqkg_slice_kernel[(NT, NK, B * HV)](
            q_norm, k_norm, v_new, g_cumsum, h_all, do, dh_all, dv_total,
            dq_inter, dk_inter, dg_inter, dw_neg,
            ctx.scale,
            T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV,
            num_warps=4, num_stages=1,
        )

        dv_out = torch.empty_like(v)
        db_wy = torch.empty(B, T, HV, device=q_norm.device, dtype=torch.float32)
        dA_raw = torch.empty(B, T, HV, BT, device=q_norm.device, dtype=torch.float32)
        dAkk = torch.empty(B, T, HV, BT, device=q_norm.device, dtype=torch.float32)
        _kda_bwd_wy_dAkk_dv_db_kernel[(NT * NC, B * HV)](
            k_norm, v, g_cumsum, beta, Akk, dv_total, dw_neg,
            dk_inter, dg_inter, dv_out, db_wy, dA_raw,
            T=T, H=H, HV=HV, K=K, V=V, BT=BT, BM=BC, BK=BK, BV=BV,
            num_warps=4, num_stages=1,
        )
        _kda_bwd_wy_dAkk_solve_kernel[(NT * NC, NC, B * HV)](
            Akk, dA_raw, dAkk,
            T=T, HV=HV, BT=BT, BM=BC,
            num_warps=4, num_stages=1,
        )

        # 4. Zero-Spill Intra-Chunk Sub-Chunk Backward (Kernel 12)
        dq_hv = torch.empty(B, T, HV, K, device=q_norm.device, dtype=torch.float32)
        dk_hv = torch.empty(B, T, HV, K, device=q_norm.device, dtype=torch.float32)
        dg_cs = torch.empty(B, T, HV, K, device=q_norm.device, dtype=torch.float32)
        db_part = torch.empty(NK, B, T, HV, device=q_norm.device, dtype=torch.float32)

        _kda_bwd_intra_subchunk_dqkgb_kernel[(NT * NC, NK, B * HV)](
            q_norm, k_norm, g_cumsum, beta, dAqk, dAkk,
            dq_inter, dk_inter, dg_inter,
            dq_hv, dk_hv, dg_cs, db_part,
            B,
            T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
            num_warps=4, num_stages=1,
        )

        # 5. Fused Reverse Cumsum + Gate VJP (Kernel 13)
        BS_gate = 32
        NS_gate = triton.cdiv(K, BS_gate)
        dg_raw = torch.empty_like(g_raw)
        dA_part = torch.empty(B * NT * NS_gate, HV, device=q_norm.device, dtype=torch.float32) if (ctx.use_gate_in_kernel and A_log is not None) else None
        dbias_part = torch.empty(B * NT, HV * K, device=q_norm.device, dtype=torch.float32) if (ctx.use_gate_in_kernel and dt_bias is not None) else None

        _kda_bwd_fused_rcumsum_gate_kernel[(NT * NS_gate, B * HV)](
            g_raw, A_log, dt_bias, dg_cs,
            dg_raw, dA_part, dbias_part,
            float(ctx.lower_bound) if ctx.lower_bound is not None else 0.0,
            T=T, HV=HV, K=K, BT=BT, BS=BS_gate,
            HAS_A=A_log is not None,
            HAS_BIAS=dt_bias is not None,
            USE_GATE_IN_KERNEL=ctx.use_gate_in_kernel,
            USE_LOWER_BOUND=ctx.lower_bound is not None,
            num_warps=4,
        )
        dA_log = dA_part.sum(dim=0).to(A_log.dtype).view_as(A_log) if dA_part is not None else None
        ddt_bias = dbias_part.sum(dim=0).to(dt_bias.dtype).view_as(dt_bias) if dbias_part is not None else None

        # 6. Fused GVA Reduction + QK-L2Norm Radial VJP + Beta Sigmoid VJP (Kernel 14)
        dq_out = torch.empty_like(q_norm)
        dk_out = torch.empty_like(k_norm)
        dbeta_out = torch.empty(B, T, HV, device=q_norm.device, dtype=beta_raw_dtype(beta, g_raw))
        beta_scale = 2.0 if ctx.allow_neg_eigval else 1.0
        _kda_bwd_qk_l2norm_beta_sigmoid_kernel[(B * T, H)](
            q_norm, q_rstd if ctx.use_qk_l2norm else q_norm,
            k_norm, k_rstd if ctx.use_qk_l2norm else k_norm,
            beta, dq_hv, dk_hv, db_wy, db_part,
            dq_out, dk_out, dbeta_out,
            beta_scale, B,
            T=T, H=H, HV=HV, K=K, BK=triton.next_power_of_2(K), NK=NK,
            USE_QK_L2NORM=ctx.use_qk_l2norm,
            USE_BETA_SIGMOID=ctx.use_beta_sigmoid,
            num_warps=4,
        )

        return (
            dq_out, dk_out, dv_out, dg_raw, dbeta_out, dA_log, ddt_bias, dh0,
            None, None, None, None, None, None, None, None, None,
        )


def beta_raw_dtype(beta: torch.Tensor, g_raw: torch.Tensor) -> torch.dtype:
    return g_raw.dtype


def chunk_kda_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _KDAChunkRSIFn.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g_raw.contiguous(),
        beta_raw.contiguous(),
        A_log.contiguous() if A_log is not None else None,
        dt_bias.contiguous() if dt_bias is not None else None,
        initial_state.contiguous() if initial_state is not None else None,
        float(scale),
        bool(use_qk_l2norm_in_kernel),
        bool(use_gate_in_kernel),
        bool(use_beta_sigmoid_in_kernel),
        bool(allow_neg_eigval),
        lower_bound,
        bool(output_final_state),
        int(chunk_size),
        1e-6,
    )


class _FusedKDAShortConvSiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        BT, BD = 64, 64
        x_c = x.contiguous()
        w_c = weight.contiguous()
        y = torch.empty_like(x_c)
        pre_act = torch.empty(B, T, D, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(T, BT), triton.cdiv(D, BD), B)
        _kda_fused_shortconv_silu_fwd_kernel[grid](
            x_c, w_c, y, pre_act, T=T, D=D, BT=BT, BD=BD, num_warps=4,
        )
        ctx.save_for_backward(x_c, w_c, pre_act)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, w_c, pre_act = ctx.saved_tensors
        B, T, D = x_c.shape
        BT, BD = 64, 64
        NT = triton.cdiv(T, BT)
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        dw_part = torch.empty(B * NT, D, 4, device=x_c.device, dtype=torch.float32)
        grid = (NT, triton.cdiv(D, BD), B)
        _kda_fused_shortconv_silu_bwd_kernel[grid](
            x_c, w_c, pre_act, dy_c, dx, dw_part,
            B, T=T, D=D, BT=BT, BD=BD, num_warps=4,
        )
        dw = dw_part.sum(dim=0).view_as(w_c).to(w_c.dtype)
        return dx, dw


class _FusedKDARMSNormSigmoidGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x_c = x.contiguous()
        z_c = z.contiguous()
        w_c = weight.contiguous()
        V = x_c.shape[-1]
        n_rows = x_c.numel() // V
        BV = triton.next_power_of_2(V)
        y = torch.empty_like(x_c)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        _kda_fused_rmsnorm_sigmoid_gate_fwd_kernel[(n_rows,)](
            x_c, z_c, w_c, y, rstd, eps, n_rows, V=V, BV=BV, num_warps=2,
        )
        ctx.save_for_backward(x_c, z_c, w_c, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, z_c, w_c, rstd = ctx.saved_tensors
        V = x_c.shape[-1]
        n_rows = x_c.numel() // V
        BV = triton.next_power_of_2(V)
        BLOCK_R = 16
        n_blocks = triton.cdiv(n_rows, BLOCK_R)
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        dz = torch.empty_like(z_c)
        dw_part = torch.empty(n_blocks, V, device=x_c.device, dtype=torch.float32)
        _kda_fused_rmsnorm_sigmoid_gate_bwd_kernel[(n_blocks,)](
            x_c, z_c, w_c, rstd, dy_c, dx, dz, dw_part,
            n_rows, V=V, BV=BV, BLOCK_R=BLOCK_R, num_warps=2,
        )
        dw = dw_part.sum(dim=0).to(w_c.dtype)
        return dx, dz, dw, None


# =============================================================================
# 5. Full KimiDeltaAttention Block (Baseline vs Round-14 RSI)
# =============================================================================

class KimiDeltaAttentionBlockBaseline(nn.Module):
    """
    Baseline `KimiDeltaAttention` block (`arXiv:2510.26692` Sec. 4 & `fla/layers/kda.py`)
    using PyTorch depthwise `Conv1d + SiLU`, `eager_kda_chunk_reference` (or upstream `fla`),
    and unfused `Sigmoid(g_proj) * RMSNorm(o)`.
    """
    def __init__(self, hidden_size: int = 1024, num_heads: int = 8, head_dim: int = 128, num_v_heads: Optional[int] = None, use_compiled_step: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads or num_heads
        self.head_dim = head_dim
        self.key_dim = self.num_heads * head_dim
        self.value_dim = self.num_v_heads * head_dim
        self.use_compiled_step = use_compiled_step

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.q_conv = nn.Conv1d(self.key_dim, self.key_dim, kernel_size=4, padding=0, groups=self.key_dim, bias=False)
        self.k_conv = nn.Conv1d(self.key_dim, self.key_dim, kernel_size=4, padding=0, groups=self.key_dim, bias=False)
        self.v_conv = nn.Conv1d(self.value_dim, self.value_dim, kernel_size=4, padding=0, groups=self.value_dim, bias=False)

        # Low-rank channel-decay projection W_alpha^up W_alpha^down and output gate W_g^up W_g^down
        self.f_down = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_up = nn.Linear(head_dim, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.full((self.value_dim,), -1.5, dtype=torch.float32))

        self.g_down = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_up = nn.Linear(head_dim, self.value_dim, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(head_dim))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, HV, K = self.num_heads, self.num_v_heads, self.head_dim

        q_lin = self.q_proj(x).transpose(1, 2)
        k_lin = self.k_proj(x).transpose(1, 2)
        v_lin = self.v_proj(x).transpose(1, 2)
        q = F.silu(F.conv1d(F.pad(q_lin, (3, 0)), self.q_conv.weight, groups=self.key_dim)).transpose(1, 2).view(B, T, H, K)
        k = F.silu(F.conv1d(F.pad(k_lin, (3, 0)), self.k_conv.weight, groups=self.key_dim)).transpose(1, 2).view(B, T, H, K)
        v = F.silu(F.conv1d(F.pad(v_lin, (3, 0)), self.v_conv.weight, groups=self.value_dim)).transpose(1, 2).view(B, T, HV, K)

        g_raw = self.f_up(self.f_down(x)).view(B, T, HV, K)
        beta_raw = self.b_proj(x)
        o, _ = eager_kda_chunk_reference(
            q, k, v, g_raw, beta_raw,
            A_log=self.A_log, dt_bias=self.dt_bias,
            use_qk_l2norm=True, use_gate_in_kernel=True, use_beta_sigmoid=True,
            lower_bound=-5.0, chunk_size=64, use_compiled_step=self.use_compiled_step,
        )
        z = self.g_up(self.g_down(x)).view(B, T, HV, K)
        of = o.float()
        rstd = torch.rsqrt((of * of).mean(dim=-1, keepdim=True) + 1e-5)
        gated = (of * rstd * self.norm_weight.float()) * torch.sigmoid(z.float())
        return self.o_proj(gated.to(x.dtype).reshape(B, T, self.value_dim))


class KimiDeltaAttentionBlockRSI(nn.Module):
    """
    Round-14 RSI `KimiDeltaAttention` block fusing 4-tap depthwise `ShortConv1d+SiLU`,
    `chunk_kda_rsi` (`0` cubin spills across all kernels), and `_FusedKDARMSNormSigmoidGateFn`.
    """
    def __init__(self, base: KimiDeltaAttentionBlockBaseline):
        super().__init__()
        self.hidden_size = base.hidden_size
        self.num_heads = base.num_heads
        self.num_v_heads = base.num_v_heads
        self.head_dim = base.head_dim
        self.key_dim = base.key_dim
        self.value_dim = base.value_dim

        self.q_proj = base.q_proj
        self.k_proj = base.k_proj
        self.v_proj = base.v_proj
        self.q_conv = base.q_conv
        self.k_conv = base.k_conv
        self.v_conv = base.v_conv
        self.f_down = base.f_down
        self.f_up = base.f_up
        self.b_proj = base.b_proj
        self.A_log = base.A_log
        self.dt_bias = base.dt_bias
        self.g_down = base.g_down
        self.g_up = base.g_up
        self.norm_weight = base.norm_weight
        self.o_proj = base.o_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, HV, K = self.num_heads, self.num_v_heads, self.head_dim

        q = _FusedKDAShortConvSiLUFn.apply(self.q_proj(x), self.q_conv.weight).view(B, T, H, K)
        k = _FusedKDAShortConvSiLUFn.apply(self.k_proj(x), self.k_conv.weight).view(B, T, H, K)
        v = _FusedKDAShortConvSiLUFn.apply(self.v_proj(x), self.v_conv.weight).view(B, T, HV, K)

        g_raw = self.f_up(self.f_down(x)).view(B, T, HV, K)
        beta_raw = self.b_proj(x)
        o, _ = chunk_kda_rsi(
            q, k, v, g_raw, beta_raw,
            A_log=self.A_log, dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            lower_bound=-5.0, chunk_size=64,
        )
        z = self.g_up(self.g_down(x)).view(B, T, HV, K)
        gated = _FusedKDARMSNormSigmoidGateFn.apply(o, z, self.norm_weight, 1e-5)
        return self.o_proj(gated.reshape(B, T, self.value_dim))


# =============================================================================
# 6. 8-GPU NVSwitch Channel-Affine Monoid Context Parallelism (KDA-CP = 8)
# =============================================================================

class _KDACP8AutogradFn(torch.autograd.Function):
    """
    8-GPU NVSwitch Channel-Affine Monoid Context Parallelism (`KDA-CP = 8`).
    For each rank `r in {0..P-1}` holding local shard `T_local = T_global // P`:
      - Every chunk `c` has exact channel-gated affine state transition:
          `S_c = P_c S_{c-1} + H_c`,
        where `P_c = Diag(exp(g_{c,BT-1})) - J_c^T W_c in R^{K x K}` and
        `H_c = J_c^T U_c in R^{K x V}`.
      - Composing across local chunks `c = 0..NT_loc-1` yields the rank-local affine
        summary `(P_rank(r), H_rank(r))`.
      - A single NVSwitch `all_gather_into_tensor` exchanges `(P_rank, H_rank)` across
        all 8 GPUs; rank `r` composes `H_prefix(q+1) = P_rank(q) H_prefix(q) + H_rank(q)`
        (`q = 0..r-1`) to obtain its exact initial state `S_init(r) = H_prefix(r)`.
      - In backward, because the homogenous adjoint transition across rank `q` is literally
        `P_rank(q)^T` (already saved from forward!), only the inhomogeneous local gradient
        `G_rank(q) = dh0_local(dht=0)` (`[B, HV, K, V]`, `2x` smaller payload!) is
        all-gathered over NVSwitch, composed via `dht(q-1) = P_rank(q)^T dht(q) + G_rank(q)`,
        and fed into the local zero-spill backward pass!
    """
    @staticmethod
    def forward(
        ctx,
        q_loc: torch.Tensor,
        k_loc: torch.Tensor,
        v_loc: torch.Tensor,
        g_raw_loc: torch.Tensor,
        beta_raw_loc: torch.Tensor,
        A_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        scale: float,
        lower_bound: Optional[float],
        chunk_size: int = 64,
        cp_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        world_size = dist.get_world_size(group=cp_group)
        rank = dist.get_rank(group=cp_group)
        B, T_loc, H, K = q_loc.shape
        HV, V = v_loc.shape[2], v_loc.shape[3]
        BT = chunk_size
        BC = 16
        NC = BT // BC
        NT = T_loc // BT
        BK_full = triton.next_power_of_2(K)

        # 1. Local QK-L2Norm + Beta Sigmoid + KDA Gate Cumsum
        q_norm = torch.empty_like(q_loc)
        k_norm = torch.empty_like(k_loc)
        q_rstd = torch.empty(B * T_loc * H, device=q_loc.device, dtype=torch.float32)
        k_rstd = torch.empty(B * T_loc * H, device=q_loc.device, dtype=torch.float32)
        beta = torch.empty(B, T_loc, HV, device=q_loc.device, dtype=torch.float32)
        n_rows_hv = B * T_loc * HV
        _kda_superfused_qk_l2norm_beta_fwd_kernel[(n_rows_hv,)](
            q_loc, k_loc, beta_raw_loc,
            q_norm, q_rstd, k_norm, k_rstd, beta,
            1.0, 1e-6, n_rows_hv,
            H=H, HV=HV, K=K, BK=BK_full, num_warps=4,
        )

        BS_gate = 32
        NS_gate = triton.cdiv(K, BS_gate)
        g_cumsum = torch.empty(B, T_loc, HV, K, device=q_loc.device, dtype=torch.float32)
        _kda_fused_gate_cumsum_fwd_kernel[(NT * NS_gate, B * HV)](
            g_raw_loc, A_log, dt_bias, g_cumsum,
            float(lower_bound) if lower_bound is not None else 0.0,
            T=T_loc, HV=HV, K=K, BT=BT, BS=BS_gate,
            HAS_A=A_log is not None,
            HAS_BIAS=dt_bias is not None,
            USE_GATE_IN_KERNEL=True,
            USE_LOWER_BOUND=lower_bound is not None,
            num_warps=4,
        )

        # 2. Local intra-chunk UT-transform WY (Aqk, Akk, w, u, qg, kg)
        Aqk = torch.empty(B, T_loc, HV, BT, device=q_loc.device, dtype=torch.float32)
        Akkd = torch.empty(B, T_loc, HV, BC, device=q_loc.device, dtype=torch.float32)
        Akk = torch.zeros(B, T_loc, HV, BT, device=q_loc.device, dtype=torch.float32)

        _kda_overflow_free_intra_diag_subchunk_fwd_kernel[(NT, NC, B * HV)](
            q_norm, k_norm, g_cumsum, beta, Aqk, Akkd,
            scale, T=T_loc, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=32, num_warps=4, num_stages=1,
        )
        _kda_zero_spill_inter_subchunk_solve_fwd_kernel[(NT, B * HV)](
            q_norm, k_norm, g_cumsum, beta, Aqk, Akkd, Akk,
            scale, T=T_loc, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=32, num_warps=4, num_stages=1,
        )

        w = torch.empty(B, T_loc, HV, K, device=q_loc.device, dtype=q_loc.dtype)
        u = torch.empty_like(v_loc)
        qg = torch.empty(B, T_loc, HV, K, device=q_loc.device, dtype=q_loc.dtype)
        kg = torch.empty(B, T_loc, HV, K, device=q_loc.device, dtype=q_loc.dtype)
        _kda_wy_wu_qg_kg_fwd_kernel[(NT * NC, B * HV)](
            q_norm, k_norm, v_loc, g_cumsum, beta, Akk,
            w, u, qg, kg,
            T=T_loc, H=H, HV=HV, K=K, V=V, BT=BT, BM=BC, BK=32, BV=32, num_warps=4, num_stages=1,
        )

        # 3. Build rank-local Channel-Affine Monoid summary `(P_rank, H_rank)` in FP32
        kg_c = kg.float().view(B, NT, BT, HV, K).permute(0, 3, 1, 2, 4)
        w_c = w.float().view(B, NT, BT, HV, K).permute(0, 3, 1, 2, 4)
        u_c = u.float().view(B, NT, BT, HV, V).permute(0, 3, 1, 2, 4)
        g_last_c = g_cumsum[:, BT - 1 :: BT].view(B, NT, HV, K).permute(0, 2, 1, 3).exp()

        I_k = torch.eye(K, device=q_loc.device, dtype=torch.float32)[None, None, :, :]
        P_loc = I_k.expand(B, HV, K, K).clone()
        H_loc = torch.zeros(B, HV, K, V, device=q_loc.device, dtype=torch.float32)

        for ci in range(NT):
            jc = kg_c[:, :, ci]           # [B, HV, BT, K]
            wc = w_c[:, :, ci]            # [B, HV, BT, K]
            uc = u_c[:, :, ci]            # [B, HV, BT, V]
            dc = g_last_c[:, :, ci]       # [B, HV, K]
            # P_c = Diag(dc) - jc^T @ wc
            P_c = torch.diag_embed(dc) - jc.transpose(-1, -2) @ wc
            H_c = jc.transpose(-1, -2) @ uc
            H_loc = P_c @ H_loc + H_c
            P_loc = P_c @ P_loc

        packed_local = torch.cat([P_loc, H_loc], dim=-1).contiguous()  # [B, HV, K, K + V]
        packed_all = torch.empty(world_size, B, HV, K, K + V, device=q_loc.device, dtype=torch.float32)
        dist.all_gather_into_tensor(packed_all, packed_local, group=cp_group)

        P_all = packed_all[..., :K].contiguous()
        H_all = packed_all[..., K:].contiguous()

        h0_rank = torch.zeros(B, HV, K, V, device=q_loc.device, dtype=torch.float32)
        for q_idx in range(rank):
            h0_rank = P_all[q_idx] @ h0_rank + H_all[q_idx]

        # 4. Run local chunk state recurrence and readout with exact `h0_rank`
        h_all = torch.empty(B, NT, HV, K, V, device=q_loc.device, dtype=q_loc.dtype)
        v_new = torch.empty_like(v_loc)
        s_cur = torch.empty(B, HV, K, V, device=q_loc.device, dtype=torch.float32)
        _kda_chunk_state_h_fwd_kernel[(triton.cdiv(V, 32), B * HV)](
            kg, w, u, g_cumsum, h0_rank if rank > 0 else None,
            h_all, v_new, s_cur,
            T=T_loc, HV=HV, K=K, V=V, BT=BT, BK=32, BV=32,
            HAS_H0=rank > 0,
            IS_FP32=(q_loc.dtype == torch.float32),
            num_warps=4, num_stages=1,
        )

        o_loc = torch.empty_like(v_loc)
        _kda_chunk_readout_o_fwd_kernel[(triton.cdiv(V, 32), NT, B * HV)](
            qg, v_new, Aqk, h_all, o_loc,
            scale, T=T_loc, HV=HV, K=K, V=V, BT=BT, BK=32, BV=32, num_warps=4, num_stages=1,
        )

        ctx.save_for_backward(
            q_norm, q_rstd, k_norm, k_rstd, v_loc, g_raw_loc, g_cumsum, beta,
            A_log, dt_bias, h0_rank, P_all, Aqk, Akk, w, qg, kg, v_new, h_all,
        )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.chunk_size = chunk_size
        ctx.cp_group = cp_group
        return o_loc

    @staticmethod
    def backward(ctx, do_loc: torch.Tensor):
        (
            q_norm, q_rstd, k_norm, k_rstd, v_loc, g_raw_loc, g_cumsum, beta,
            A_log, dt_bias, h0_rank, P_all, Aqk, Akk, w, qg, kg, v_new, h_all,
        ) = ctx.saved_tensors

        world_size = dist.get_world_size(group=ctx.cp_group)
        rank = dist.get_rank(group=ctx.cp_group)
        B, T_loc, H, K = q_norm.shape
        HV, V = v_loc.shape[2], v_loc.shape[3]
        BT = ctx.chunk_size
        BC = 16
        NC = BT // BC
        NT = T_loc // BT
        BK = 32
        BV = 32
        BV_h = 16
        NK = triton.cdiv(K, BK)

        do_loc = do_loc.contiguous()

        # 1. Local dAqk & dv_intra
        dAqk = torch.empty(B, T_loc, HV, BT, device=q_norm.device, dtype=torch.float32)
        dv_intra = torch.empty_like(v_loc)
        _kda_bwd_dAv_kernel[(NT * NC, B * HV)](
            v_new, Aqk, do_loc, dv_intra, dAqk,
            ctx.scale, T=T_loc, HV=HV, V=V, BT=BT, BM=BC, BV=BV, num_warps=4, num_stages=1,
        )

        # 2. Compute inhomogeneous local initial-state gradient `G_rank = dh0(dht=0)`
        dh_all_tmp = torch.empty_like(h_all)
        dv_total_tmp = torch.empty_like(v_loc)
        G_rank = torch.empty(B, HV, K, V, device=q_norm.device, dtype=torch.float32)
        _kda_bwd_dhu_reverse_kernel[(triton.cdiv(V, BV), B * HV)](
            qg, kg, w, g_cumsum, do_loc, dv_intra, None,
            dh_all_tmp, dv_total_tmp, G_rank,
            ctx.scale, T=T_loc, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV,
            HAS_DHT=False, IS_FP32=(q_norm.dtype == torch.float32), num_warps=4, num_stages=1,
        )

        # 3. Single NVSwitch AllGather of `G_rank` (2x smaller payload; reuses saved `P_all^T`!)
        G_all = torch.empty(world_size, B, HV, K, V, device=q_norm.device, dtype=torch.float32)
        dist.all_gather_into_tensor(G_all, G_rank, group=ctx.cp_group)

        dht_rank = torch.zeros(B, HV, K, V, device=q_norm.device, dtype=torch.float32)
        for q_idx in range(world_size - 1, rank, -1):
            dht_rank = P_all[q_idx].transpose(-1, -2) @ dht_rank + G_all[q_idx]

        # 4. Run local reverse state scan with exact `dht_rank` (only needed when rank < world_size - 1)
        if rank < world_size - 1:
            dh_all = torch.empty_like(h_all)
            dv_total = torch.empty_like(v_loc)
            ds_cur_tmp = torch.empty(B, HV, K, V, device=q_norm.device, dtype=torch.float32)
            _kda_bwd_dhu_reverse_kernel[(triton.cdiv(V, BV), B * HV)](
                qg, kg, w, g_cumsum, do_loc, dv_intra, dht_rank,
                dh_all, dv_total, ds_cur_tmp,
                ctx.scale, T=T_loc, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV,
                HAS_DHT=True, IS_FP32=(q_norm.dtype == torch.float32), num_warps=4, num_stages=1,
            )
        else:
            dh_all = dh_all_tmp
            dv_total = dv_total_tmp

        # 5. Zero-Spill Split-K/V WY Backward & Intra-Chunk Backward
        dq_inter = torch.empty(B, T_loc, HV, K, device=q_norm.device, dtype=torch.float32)
        dk_inter = torch.empty(B, T_loc, HV, K, device=q_norm.device, dtype=torch.float32)
        dg_inter = torch.empty(B, T_loc, HV, K, device=q_norm.device, dtype=torch.float32)
        dw_neg = torch.empty(B, T_loc, HV, K, device=q_norm.device, dtype=q_norm.dtype)

        _kda_bwd_wy_dqkg_slice_kernel[(NT, NK, B * HV)](
            q_norm, k_norm, v_new, g_cumsum, h_all, do_loc, dh_all, dv_total,
            dq_inter, dk_inter, dg_inter, dw_neg,
            ctx.scale, T=T_loc, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV, num_warps=4, num_stages=1,
        )

        dv_out = torch.empty_like(v_loc)
        db_wy = torch.empty(B, T_loc, HV, device=q_norm.device, dtype=torch.float32)
        dA_raw = torch.empty(B, T_loc, HV, BT, device=q_norm.device, dtype=torch.float32)
        dAkk = torch.empty(B, T_loc, HV, BT, device=q_norm.device, dtype=torch.float32)
        _kda_bwd_wy_dAkk_dv_db_kernel[(NT * NC, B * HV)](
            k_norm, v_loc, g_cumsum, beta, Akk, dv_total, dw_neg,
            dk_inter, dg_inter, dv_out, db_wy, dA_raw,
            T=T_loc, H=H, HV=HV, K=K, V=V, BT=BT, BM=BC, BK=BK, BV=BV, num_warps=4, num_stages=1,
        )
        _kda_bwd_wy_dAkk_solve_kernel[(NT * NC, NC, B * HV)](
            Akk, dA_raw, dAkk,
            T=T_loc, HV=HV, BT=BT, BM=BC, num_warps=4, num_stages=1,
        )

        dq_hv = torch.empty(B, T_loc, HV, K, device=q_norm.device, dtype=torch.float32)
        dk_hv = torch.empty(B, T_loc, HV, K, device=q_norm.device, dtype=torch.float32)
        dg_cs = torch.empty(B, T_loc, HV, K, device=q_norm.device, dtype=torch.float32)
        db_part = torch.empty(NK, B, T_loc, HV, device=q_norm.device, dtype=torch.float32)

        _kda_bwd_intra_subchunk_dqkgb_kernel[(NT * NC, NK, B * HV)](
            q_norm, k_norm, g_cumsum, beta, dAqk, dAkk,
            dq_inter, dk_inter, dg_inter,
            dq_hv, dk_hv, dg_cs, db_part,
            B, T=T_loc, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK, NC=NC, num_warps=4, num_stages=1,
        )

        BS_gate = 32
        NS_gate = triton.cdiv(K, BS_gate)
        dg_raw = torch.empty_like(g_raw_loc)
        dA_part = torch.empty(B * NT * NS_gate, HV, device=q_norm.device, dtype=torch.float32) if A_log is not None else None
        dbias_part = torch.empty(B * NT, HV * K, device=q_norm.device, dtype=torch.float32) if dt_bias is not None else None

        _kda_bwd_fused_rcumsum_gate_kernel[(NT * NS_gate, B * HV)](
            g_raw_loc, A_log, dt_bias, dg_cs,
            dg_raw, dA_part, dbias_part,
            float(ctx.lower_bound) if ctx.lower_bound is not None else 0.0,
            T=T_loc, HV=HV, K=K, BT=BT, BS=BS_gate,
            HAS_A=A_log is not None,
            HAS_BIAS=dt_bias is not None,
            USE_GATE_IN_KERNEL=True,
            USE_LOWER_BOUND=ctx.lower_bound is not None,
            num_warps=4,
        )
        dA_log = dA_part.sum(dim=0).to(A_log.dtype).view_as(A_log) if dA_part is not None else None
        ddt_bias = dbias_part.sum(dim=0).to(dt_bias.dtype).view_as(dt_bias) if dbias_part is not None else None
        if dA_log is not None:
            dist.all_reduce(dA_log, op=dist.ReduceOp.SUM, group=ctx.cp_group)
        if ddt_bias is not None:
            dist.all_reduce(ddt_bias, op=dist.ReduceOp.SUM, group=ctx.cp_group)

        dq_out = torch.empty_like(q_norm)
        dk_out = torch.empty_like(k_norm)
        dbeta_out = torch.empty(B, T_loc, HV, device=q_norm.device, dtype=g_raw_loc.dtype)
        _kda_bwd_qk_l2norm_beta_sigmoid_kernel[(B * T_loc, H)](
            q_norm, q_rstd, k_norm, k_rstd, beta,
            dq_hv, dk_hv, db_wy, db_part,
            dq_out, dk_out, dbeta_out,
            1.0, B, T=T_loc, H=H, HV=HV, K=K, BK=triton.next_power_of_2(K), NK=NK,
            USE_QK_L2NORM=True, USE_BETA_SIGMOID=True, num_warps=4,
        )

        return dq_out, dk_out, dv_out, dg_raw, dbeta_out, dA_log, ddt_bias, None, None, None, None


def kda_cp8_autostep(
    q_loc: torch.Tensor,
    k_loc: torch.Tensor,
    v_loc: torch.Tensor,
    g_raw_loc: torch.Tensor,
    beta_raw_loc: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = None,
    chunk_size: int = 64,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    if scale is None:
        scale = q_loc.shape[-1] ** -0.5
    return _KDACP8AutogradFn.apply(
        q_loc.contiguous(),
        k_loc.contiguous(),
        v_loc.contiguous(),
        g_raw_loc.contiguous(),
        beta_raw_loc.contiguous(),
        A_log,
        dt_bias,
        float(scale),
        lower_bound,
        int(chunk_size),
        cp_group,
    )


# =============================================================================
# 7. SM80 Cubin Register Spill & Compilation Auditor
# =============================================================================

RSI_ROUND14_KERNELS = [
    ("_kda_superfused_qk_l2norm_beta_fwd_kernel", _kda_superfused_qk_l2norm_beta_fwd_kernel),
    ("_kda_fused_gate_cumsum_fwd_kernel", _kda_fused_gate_cumsum_fwd_kernel),
    ("_kda_overflow_free_intra_diag_subchunk_fwd_kernel", _kda_overflow_free_intra_diag_subchunk_fwd_kernel),
    ("_kda_zero_spill_inter_subchunk_solve_fwd_kernel", _kda_zero_spill_inter_subchunk_solve_fwd_kernel),
    ("_kda_wy_wu_qg_kg_fwd_kernel", _kda_wy_wu_qg_kg_fwd_kernel),
    ("_kda_chunk_state_h_fwd_kernel", _kda_chunk_state_h_fwd_kernel),
    ("_kda_chunk_readout_o_fwd_kernel", _kda_chunk_readout_o_fwd_kernel),
    ("_kda_bwd_dAv_kernel", _kda_bwd_dAv_kernel),
    ("_kda_bwd_dhu_reverse_kernel", _kda_bwd_dhu_reverse_kernel),
    ("_kda_bwd_wy_dqkg_slice_kernel", _kda_bwd_wy_dqkg_slice_kernel),
    ("_kda_bwd_wy_dAkk_dv_db_kernel", _kda_bwd_wy_dAkk_dv_db_kernel),
    ("_kda_bwd_wy_dAkk_solve_kernel", _kda_bwd_wy_dAkk_solve_kernel),
    ("_kda_bwd_intra_subchunk_dqkgb_kernel", _kda_bwd_intra_subchunk_dqkgb_kernel),
    ("_kda_bwd_fused_rcumsum_gate_kernel", _kda_bwd_fused_rcumsum_gate_kernel),
    ("_kda_bwd_qk_l2norm_beta_sigmoid_kernel", _kda_bwd_qk_l2norm_beta_sigmoid_kernel),
    ("_kda_fused_shortconv_silu_fwd_kernel", _kda_fused_shortconv_silu_fwd_kernel),
    ("_kda_fused_shortconv_silu_bwd_kernel", _kda_fused_shortconv_silu_bwd_kernel),
    ("_kda_fused_rmsnorm_sigmoid_gate_fwd_kernel", _kda_fused_rmsnorm_sigmoid_gate_fwd_kernel),
    ("_kda_fused_rmsnorm_sigmoid_gate_bwd_kernel", _kda_fused_rmsnorm_sigmoid_gate_bwd_kernel),
]


def _extract_kernel_spills(jit_fn: Any) -> Tuple[int, int, int]:
    regs: List[int] = []
    spills: List[int] = []
    dev_caches = getattr(jit_fn, "device_caches", {})
    for _, tup in dev_caches.items():
        cache_dict = tup[0] if isinstance(tup, tuple) else tup
        if isinstance(cache_dict, dict):
            for _, c_bin in cache_dict.items():
                if hasattr(c_bin, "n_spills"):
                    regs.append(int(c_bin.n_regs))
                    spills.append(int(c_bin.n_spills))
    cache = getattr(jit_fn, "cache", {})
    if isinstance(cache, dict) and not spills:
        for _, v in cache.items():
            if hasattr(v, "n_spills"):
                regs.append(int(v.n_regs))
                spills.append(int(v.n_spills))
            elif isinstance(v, dict):
                for _, vv in v.items():
                    if hasattr(vv, "n_spills"):
                        regs.append(int(vv.n_regs))
                        spills.append(int(vv.n_spills))
    return len(spills), (max(regs) if regs else 0), sum(spills)


def audit_kda_triton_cubin_spills(device: str = "cuda:0") -> Dict[str, Any]:
    """
    Compiles all 18 Round-14 RSI Triton kernels across `K, V in {64, 128}` (`FP32` and `BF16`)
    plus the negative-control monolithic WY backward spill kernel, and verifies `0` cubin
    register spills across 100% of Round-14 RSI kernels.
    """
    for K_dim in (64, 128):
        for dt in (torch.float32, torch.bfloat16):
            B, T, H = 1, 128, 4
            q = torch.randn(B, T, H, K_dim, device=device, dtype=dt, requires_grad=True)
            k = torch.randn(B, T, H, K_dim, device=device, dtype=dt, requires_grad=True)
            v = torch.randn(B, T, H, K_dim, device=device, dtype=dt, requires_grad=True)
            g = torch.randn(B, T, H, K_dim, device=device, dtype=dt, requires_grad=True)
            b = torch.randn(B, T, H, device=device, dtype=dt, requires_grad=True)
            A_log = torch.zeros(H, device=device, dtype=torch.float32, requires_grad=True)
            dt_bias = torch.zeros(H * K_dim, device=device, dtype=torch.float32, requires_grad=True)
            o, _ = chunk_kda_rsi(q, k, v, g, b, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0, output_final_state=True)
            o.sum().backward()

        blk_base = KimiDeltaAttentionBlockBaseline(hidden_size=256, num_heads=2, head_dim=K_dim).to(device).bfloat16()
        blk_rsi = KimiDeltaAttentionBlockRSI(blk_base)
        x = torch.randn(1, 128, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
        blk_rsi(x).sum().backward()

    # Compile negative-control monolithic `[64, 64]` WY backward kernel at K=V=128
    B, T, H, K, V, BT = 1, 128, 2, 128, 128, 64
    NT = T // BT
    q_s = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k_s = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    v_s = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    vn_s = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g_s = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    b_s = torch.randn(B, T, H, device=device, dtype=torch.float32)
    A_s = torch.randn(B, T, H, BT, device=device, dtype=torch.bfloat16)
    h_s = torch.randn(B, NT, H, K, V, device=device, dtype=torch.bfloat16)
    do_s = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    dh_s = torch.randn(B, NT, H, K, V, device=device, dtype=torch.bfloat16)
    dv_s = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    dq_s = torch.empty(B, T, H, K, device=device, dtype=torch.float32)
    dk_s = torch.empty(B, T, H, K, device=device, dtype=torch.float32)
    dv2_s = torch.empty_like(v_s)
    dg_s = torch.empty_like(g_s)
    db_s = torch.empty_like(b_s)
    dA_s = torch.empty_like(A_s, dtype=torch.float32)

    _kda_monolithic_wy_dqkg_bwd_spill_kernel[(NT, B * H)](
        q_s, k_s, v_s, vn_s, g_s, b_s, A_s, h_s, do_s, dh_s, dv_s,
        dq_s, dk_s, dv2_s, dg_s, db_s, dA_s,
        K ** -0.5,
        T=T, H=H, HV=H, K=K, V=V, BT=BT, BK=64, BV=64,
        num_warps=4, num_stages=2,
    )
    _, neg_regs, neg_spills = _extract_kernel_spills(_kda_monolithic_wy_dqkg_bwd_spill_kernel)

    kernel_records = []
    total_specs = 0
    total_spills = 0
    compiled_count = 0
    for k_name, k_fn in RSI_ROUND14_KERNELS:
        n_specs, max_r, n_sp = _extract_kernel_spills(k_fn)
        if n_specs > 0:
            compiled_count += 1
        total_specs += n_specs
        total_spills += n_sp
        kernel_records.append({
            "kernel": k_name,
            "specializations": n_specs,
            "max_n_regs": max_r,
            "total_spills": n_sp,
        })

    return {
        "total_rsi_kernels": len(RSI_ROUND14_KERNELS),
        "compiled_rsi_kernels": compiled_count,
        "total_specializations": total_specs,
        "total_rsi_spills": total_spills,
        "negative_control_monolithic_wy_bwd_spills": neg_spills,
        "negative_control_monolithic_wy_bwd_regs": neg_regs,
        "all_zero_spills": bool(compiled_count == len(RSI_ROUND14_KERNELS) and total_spills == 0 and neg_spills > 0),
        "kernels": kernel_records,
    }
