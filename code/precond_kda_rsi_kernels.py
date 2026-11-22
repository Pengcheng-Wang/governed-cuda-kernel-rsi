#!/usr/bin/env python3
"""
E142 (v133, Round-16 RSI): Curvature-Aware Preconditioned Delta Attention (PrecondKDA) & Adaptive Diagonal Curvature Preconditioning (ATK) Triton Kernels
=========================================================================================================================================================

Source-verified against:
  - Moonshot AI / Kimi Team & FLA Team, "Preconditioned DeltaNet: Curvature-aware Sequence Modeling
    for Linear Recurrences" (arXiv:2604.21100, 2026)
  - Official reference repository: `fla-org/flash-linear-attention/tree/main/fla/ops/precond_kda`,
    `fla/ops/atk`, and `fla/layers/precond_kda.py`

Resolves the 4 upstream mathematical, numerical, and SM80 hardware bottlenecks verified on
8x NVIDIA A100-SXM4-80GB (`NV12` NVSwitch):
  1. Eliminates all `50,812` total cubin register spills (`318-322` winning-config spills/step) across
     `chunk_precond_kda_bwd_kernel_wy_dqkg` (`168-3,972` spills across `36/36` configs = `49,824`),
     `chunk_precond_kda_fwd_kernel_inter_solve_fused` (`62-246` spills across `3/3` configs = `430-438`),
     `_atk_backward_chunk_out` (`88` spills across `1/1` config), `recompute_w_u_fwd_kernel` (`44-78`
     spills across `6/18` configs = `364`), and `chunk_precond_kda_bwd_kernel_intra` (`32-34` spills
     across `3/12` configs = `98`) -> `0` cubin spills across all `24/24` Round-16 RSI Triton kernels.
  2. Eliminates all `401,408` `NaN`s (`o=8,192, dq=131,072, dk=131,072, dg=131,072` at `B=2, T=128,
     H=4, K=V=128`) in upstream's `safe_gate=True` shifted-anchor sub-chunk kernel (`b_gn = g[i_ti] -
     0.5*(BC-1)*GATE_SHIFT`, `exp2(-b_gm)`) via pre-masked non-positive causal differences `g_i - g_j <= 0`.
  3. Replaces upstream's 9-launch `64x64` matrix-materialized ATK pipeline (`chunk_atk_fwd` +
     `recompute_atk_fwd` + `chunk_atk_bwd`) with a 4-launch `16x16` sub-chunked ATK pipeline with saved
     per-subchunk boundary states `X_sub_init` (`0` recompute launches, `0` spills) plus superfused
     block-level frontend (`ShortConv1d+SiLU`, `QK-L2Norm+Beta/ATK-Gates`) and epilogue (`RMSNormGated`).
  4. Resolves upstream's `NotImplementedError` on Context Parallelism (`chunk.py:540-545`) via Two-Stage
     Hierarchically Coupled Curvature-State + Asymmetric Matrix-Monoid Context Parallelism (`PKDA-CP = 8`).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# =============================================================================
# 1. Ground-Truth FP64 / FP32 Sequential & Chunk Oracles
# =============================================================================

def precond_kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    log_atk_scale: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_A_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    use_qk_l2norm: bool = True,
    compute_dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Exact step-by-step sequential recurrence oracle for Preconditioned KDA in FP64/FP32:
      Stage 1 (ATK):
        X_{t+1} = exp(g_atk_t) * X_t + beta_atk_t * (k_t ** 2)
        ell_t   = log(X_{t+1} + eps)
        r_t     = ell_t - log_atk_scale
        s_t     = r_t / (1 + |r_t|)
        M_t     = exp(-log(x) * s_t)
        k_tilde = k_t * M_t
      Stage 2 (Asymmetric KDA):
        S_{t+1} = exp(g_t) * S_t + beta_t * k_tilde_t (v_t - (k_t^T (exp(g_t) * S_t)))^T
        o_t     = scale * S_{t+1}^T q_t
    """
    orig_dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    qf = q.to(compute_dtype)
    kf = k.to(compute_dtype)
    if use_qk_l2norm:
        qf = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-12)
        kf = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-12)
    vf = v.to(compute_dtype)
    gf = g.to(compute_dtype)
    gaf = g_atk.to(compute_dtype)
    baf = beta_atk.to(compute_dtype)
    bf = beta.to(compute_dtype)

    if log_atk_scale is None:
        center = torch.full((1, H, 1), -0.2, device=q.device, dtype=compute_dtype)
    else:
        center = log_atk_scale.to(compute_dtype).view(1, H, 1)

    logx = math.log(float(x))
    X = torch.zeros(B, H, K, device=q.device, dtype=compute_dtype)
    if initial_A_state is not None:
        X = X + initial_A_state.to(compute_dtype)

    S = torch.zeros(B, H, K, V, device=q.device, dtype=compute_dtype)
    if initial_state is not None:
        S = S + initial_state.to(compute_dtype)

    out_list = []
    for t in range(T):
        qt = qf[:, t] * scale
        kt = kf[:, t]
        vt = vf[:, t]
        gt = gf[:, t]
        gat = gaf[:, t]
        bat = baf[:, t]
        bt = bf[:, t]

        X = torch.exp(gat).unsqueeze(-1) * X + bat.unsqueeze(-1) * (kt * kt)
        ell = torch.log(X + eps)
        r = ell - center
        s = r / (1.0 + torch.abs(r))
        M = torch.exp(-logx * s)
        k_tilde = kt * M

        S_dec = S * torch.exp(gt).unsqueeze(-1)
        pred_v = (kt.unsqueeze(-1) * S_dec).sum(dim=-2)
        S = S_dec + (bt.unsqueeze(-1) * k_tilde).unsqueeze(-1) * (vt - pred_v).unsqueeze(-2)
        ot = (qt.unsqueeze(-1) * S).sum(dim=-2)
        out_list.append(ot)

    out_dtype = torch.float64 if compute_dtype == torch.float64 else orig_dtype
    state_dtype = torch.float64 if compute_dtype == torch.float64 else torch.float32
    o = torch.stack(out_list, dim=1).to(out_dtype)
    if not output_final_state:
        return o, None, None
    return o, S.to(state_dtype), X.to(state_dtype)


_COMPILED_PKDA_ATK_STEP = None
_COMPILED_PKDA_KDA_STEP = None


def _pkda_atk_chunk_step_fn(
    ga_ci: torch.Tensor,
    ba_ci: torch.Tensor,
    k_ci: torch.Tensor,
    X_cur: torch.Tensor,
    causal_tril: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    la_cum = ga_ci.cumsum(dim=-1)
    diff_a = (la_cum[:, :, :, None] - la_cum[:, :, None, :]).masked_fill(~causal_tril[None, None], -1e30)
    M_atk = torch.exp(diff_a) * causal_tril[None, None].float()
    U_atk = ba_ci.unsqueeze(-1) * (k_ci * k_ci)
    X_ci = torch.exp(la_cum).unsqueeze(-1) * X_cur.unsqueeze(-2) + torch.matmul(M_atk, U_atk)
    return X_ci, X_ci[:, :, -1, :]


def _pkda_kda_chunk_step_fn(
    u_ci: torch.Tensor,
    w_ci: torch.Tensor,
    qg_ci: torch.Tensor,
    Aqk_ci: torch.Tensor,
    kg_ci: torch.Tensor,
    g_last_ci: torch.Tensor,
    S_cur: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v_new = u_ci - torch.matmul(w_ci, S_cur)
    o_ci = torch.matmul(qg_ci * scale, S_cur) + torch.matmul(Aqk_ci, v_new)
    S_next = S_cur * torch.exp(g_last_ci.transpose(-1, -2)) + torch.matmul(kg_ci.transpose(-1, -2), v_new)
    return o_ci, S_next


def _get_compiled_pkda_steps():
    global _COMPILED_PKDA_ATK_STEP, _COMPILED_PKDA_KDA_STEP
    if _COMPILED_PKDA_ATK_STEP is None:
        _COMPILED_PKDA_ATK_STEP = torch.compile(_pkda_atk_chunk_step_fn)
        _COMPILED_PKDA_KDA_STEP = torch.compile(_pkda_kda_chunk_step_fn)
    return _COMPILED_PKDA_ATK_STEP, _COMPILED_PKDA_KDA_STEP


def precond_kda_eager_chunk_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    log_atk_scale: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_A_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    use_qk_l2norm: bool = True,
    chunk_size: int = 64,
    use_compiled_step: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Differentiable PyTorch eager chunkwise WY reference for Preconditioned KDA (Track 1 baseline).
    """
    orig_dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5

    qf = q.float()
    kf = k.float()
    if use_qk_l2norm:
        qf = F.normalize(qf, dim=-1, eps=1e-12)
        kf = F.normalize(kf, dim=-1, eps=1e-12)
    vf = v.float()
    gf = g.float()
    gaf = g_atk.float()
    baf = beta_atk.float()
    bf = beta.float()

    if log_atk_scale is None:
        center = torch.full((1, 1, H, 1), -0.2, device=q.device, dtype=torch.float32)
    else:
        center = log_atk_scale.float().view(1, 1, H, 1)
    logx = math.log(float(x))

    # Stage 1: Chunked ATK in PyTorch eager
    X_cur = torch.zeros(B, H, K, device=q.device, dtype=torch.float32)
    if initial_A_state is not None:
        X_cur = X_cur + initial_A_state.float()

    ga_c = gaf.reshape(B, NT, BT, H).permute(0, 3, 1, 2)
    ba_c = baf.reshape(B, NT, BT, H).permute(0, 3, 1, 2)
    k_c = kf.reshape(B, NT, BT, H, K).permute(0, 2, 1, 3, 4)  # [B, BT, NT, H, K] -> permute below
    k_c = kf.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)   # [B, H, NT, BT, K]

    idx = torch.arange(BT, device=q.device)
    strict_tril = idx[:, None] > idx[None, :]
    causal_tril = idx[:, None] >= idx[None, :]

    if use_compiled_step:
        atk_step_fn, kda_step_fn = _get_compiled_pkda_steps()
    else:
        atk_step_fn, kda_step_fn = _pkda_atk_chunk_step_fn, _pkda_kda_chunk_step_fn

    X_all = []
    for ci in range(NT):
        X_ci, X_cur = atk_step_fn(ga_c[:, :, ci, :], ba_c[:, :, ci, :], k_c[:, :, ci, :, :], X_cur, causal_tril)
        X_all.append(X_ci)

    X_seq = torch.stack(X_all, dim=2).permute(0, 2, 3, 1, 4).reshape(B, T, H, K)
    ell = torch.log(X_seq + eps)
    r = ell - center
    s = r / (1.0 + torch.abs(r))
    M_mult = torch.exp(-logx * s)
    k_tilde = kf * M_mult

    # Stage 2: Asymmetric KDA chunkwise WY
    g_c = gf.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    g_cum = g_c.cumsum(dim=3)
    q_c = qf.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    kt_c = k_tilde.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    v_c = vf.reshape(B, NT, BT, H, V).permute(0, 3, 1, 2, 4)
    b_c = bf.reshape(B, NT, BT, H).permute(0, 3, 1, 2)

    diff_g = g_cum[:, :, :, :, None, :] - g_cum[:, :, :, None, :, :]
    exp_causal = torch.exp(diff_g.masked_fill(~causal_tril[None, None, None, :, :, None], -1e30)) * causal_tril[None, None, None, :, :, None].float()
    exp_strict = torch.exp(diff_g.masked_fill(~strict_tril[None, None, None, :, :, None], -1e30)) * strict_tril[None, None, None, :, :, None].float()

    Aqk = ((q_c[:, :, :, :, None, :] * exp_causal) * kt_c[:, :, :, None, :, :]).sum(dim=-1) * scale
    Akk = ((k_c[:, :, :, :, None, :] * exp_strict) * kt_c[:, :, :, None, :, :]).sum(dim=-1) * b_c[:, :, :, :, None]

    eye_bt = torch.eye(BT, device=q.device, dtype=torch.float32)[None, None, None, :, :]
    M_inv = torch.linalg.solve_triangular(eye_bt + Akk, eye_bt, upper=False, unitriangular=True)

    w_wy = torch.matmul(M_inv, k_c * b_c.unsqueeze(-1) * torch.exp(g_cum))
    u_wy = torch.matmul(M_inv, v_c * b_c.unsqueeze(-1))
    qg = q_c * torch.exp(g_cum)
    g_last = g_cum[:, :, :, -1:, :]
    kg = kt_c * torch.exp(g_last - g_cum)

    S_cur = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    if initial_state is not None:
        S_cur = S_cur + initial_state.float()

    o_chunks = []
    for ci in range(NT):
        o_ci, S_cur = kda_step_fn(
            u_wy[:, :, ci], w_wy[:, :, ci], qg[:, :, ci], Aqk[:, :, ci],
            kg[:, :, ci], g_last[:, :, ci], S_cur, scale,
        )
        o_chunks.append(o_ci)

    o_out = torch.stack(o_chunks, dim=2).permute(0, 2, 3, 1, 4).reshape(B, T, H, V).to(orig_dtype)
    if not output_final_state:
        return o_out, None, None
    return o_out, S_cur, X_cur


# =============================================================================
# 2. Stage 1: Zero-Spill 16x16 Sub-Chunked ATK Curvature Preconditioner Kernels (Kernels 1-4)
# =============================================================================

@triton.jit
def _atk_fused_subchunk_state_fwd_kernel(
    k_ptr, beta_atk_ptr, g_atk_ptr, x0_ptr,
    x_sub_init_ptr, sa_sub_ptr, x_final_ptr, p_shard_ptr, a_shard_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    HAS_X0: tl.constexpr,
):
    """
    Kernel 1: Fused 16-Token Sub-Chunk ATK Summary + Cross-Subchunk Prefix Scan (`0` spills).
    Computes exact initial curvature state `X_sub_init[b, s, h, k]` at the start of every
    16-token sub-chunk `s in [0, T // BC)` plus final state `X_T` and CP shard monoid `(p_shard, a_shard)`.
    Grid: `(cdiv(K, BK), B * H)`.
    """
    i_k = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NS = T // BC
    bos = i_b * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_c = tl.arange(0, BC)

    if HAS_X0:
        b_x = tl.load(x0_ptr + (i_b * H + i_h) * K + o_k).to(tl.float32)
    else:
        b_x = tl.zeros([BK], dtype=tl.float32)

    b_p_shard = 1.0
    b_a_shard = tl.zeros([BK], dtype=tl.float32)

    for s in range(NS):
        t_idx = s * BC + o_c
        # Store sub-chunk initial curvature state X_sub_init[b, s, h, k]
        sub_off = ((i_b * NS + s) * H + i_h) * K + o_k
        tl.store(x_sub_init_ptr + sub_off, b_x)

        g_val = tl.load(g_atk_ptr + (bos + t_idx) * H + i_h).to(tl.float32)
        beta_val = tl.load(beta_atk_ptr + (bos + t_idx) * H + i_h).to(tl.float32)

        sa_val = tl.sum(g_val, axis=0)
        if i_k == 0:
            tl.store(sa_sub_ptr + (i_b * NS + s) * H + i_h, sa_val)

        # Suffix decay exp(sa_val - cumsum(g_val)) has exponent <= 0 because g_val <= 0
        sfx_decay = tl.exp(tl.minimum(sa_val - tl.cumsum(g_val, axis=0), 0.0))

        k_off = (bos + t_idx[:, None]) * (H * K) + i_h * K + o_k[None, :]
        k_val = tl.load(k_ptr + k_off).to(tl.float32)
        u_val = (beta_val * sfx_decay)[:, None] * (k_val * k_val)
        a_sub = tl.sum(u_val, axis=0)

        exp_sa = tl.exp(tl.minimum(sa_val, 0.0))
        b_x = exp_sa * b_x + a_sub
        b_p_shard = exp_sa * b_p_shard
        b_a_shard = exp_sa * b_a_shard + a_sub

    tl.store(x_final_ptr + (i_b * H + i_h) * K + o_k, b_x)
    if i_k == 0:
        tl.store(p_shard_ptr + i_b * H + i_h, b_p_shard)
    tl.store(a_shard_ptr + (i_b * H + i_h) * K + o_k, b_a_shard)


@triton.jit
def _atk_subchunk_squash_out_fwd_kernel(
    k_ptr, beta_atk_ptr, g_atk_ptr, x_sub_init_ptr, log_atk_scale_ptr,
    k_tilde_ptr,
    logx: tl.constexpr,
    eps: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Kernel 2: Parallel 16x16 Sub-Chunk ATK Evaluation & Symmetric Fast Squash (`0` spills).
    Evaluates X_{t+1} inside each 16-token sub-chunk using a pre-masked [16, 16] causal decay matrix
    M_sub (`16x` smaller than upstream's [64, 64] matrix) and writes `k_tilde = k * exp(-log(x)*s)`.
    Grid: `(T // BC, cdiv(K, BK), B * H)`.
    """
    i_s = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NS = T // BC
    bos = i_b * T

    o_c = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    t_idx = i_s * BC + o_c

    g_val = tl.load(g_atk_ptr + (bos + t_idx) * H + i_h).to(tl.float32)
    beta_val = tl.load(beta_atk_ptr + (bos + t_idx) * H + i_h).to(tl.float32)
    la_cum = tl.cumsum(g_val, axis=0)

    m_causal = o_c[:, None] >= o_c[None, :]
    diff_la = tl.where(m_causal, la_cum[:, None] - la_cum[None, :], -1.0e30)
    M_sub = tl.where(m_causal, tl.exp(diff_la), 0.0)
    base_decay = tl.exp(tl.minimum(la_cum, 0.0))

    x_init = tl.load(x_sub_init_ptr + ((i_b * NS + i_s) * H + i_h) * K + o_k).to(tl.float32)
    center = tl.load(log_atk_scale_ptr + i_h).to(tl.float32)

    k_off = (bos + t_idx[:, None]) * (H * K) + i_h * K + o_k[None, :]
    k_val = tl.load(k_ptr + k_off).to(tl.float32)
    u_val = beta_val[:, None] * (k_val * k_val)

    # Exact [16, 16] x [16, BK] sub-chunk state evolution
    x_t = base_decay[:, None] * x_init[None, :] + tl.dot(M_sub, u_val, allow_tf32=False)
    ell = tl.log(x_t + eps)
    r = ell - center
    s = r / (1.0 + tl.abs(r))
    m_mult = tl.exp(-logx * s)
    k_tilde = k_val * m_mult

    tl.store(k_tilde_ptr + k_off, k_tilde.to(k_tilde_ptr.dtype.element_ty))


@triton.jit
def _atk_subchunk_squash_bwd_kernel(
    k_ptr, beta_atk_ptr, g_atk_ptr, x_sub_init_ptr, log_atk_scale_ptr, dk_tilde_ptr,
    dk_atk_ptr, dg_atk_part_ptr, dbeta_atk_part_ptr, dx_sub_in_ptr, dcenter_part_ptr,
    logx: tl.constexpr,
    eps: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    WRONG_NEG_CENTER_SIGN: tl.constexpr = False,
):
    """
    Kernel 3: Parallel 16x16 Sub-Chunk ATK Backward Kernel (`0` spills vs upstream's `88` spills).
    Uses [16, 16] `M_sub` and `gM_sub` (`16x` smaller than upstream's [64, 64] matrices) and writes
    per-K-slice partial reductions so zero global atomic contention occurs across sequence chunks.
    Grid: `(T // BC, cdiv(K, BK), B * H)`.
    """
    i_s = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NS = T // BC
    NK = K // BK
    bos = i_b * T

    o_c = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    t_idx = i_s * BC + o_c

    g_val = tl.load(g_atk_ptr + (bos + t_idx) * H + i_h).to(tl.float32)
    beta_val = tl.load(beta_atk_ptr + (bos + t_idx) * H + i_h).to(tl.float32)
    la_cum = tl.cumsum(g_val, axis=0)

    m_causal = o_c[:, None] >= o_c[None, :]
    m_strict = o_c[:, None] > o_c[None, :]
    diff_la = tl.where(m_causal, la_cum[:, None] - la_cum[None, :], -1.0e30)
    M_sub = tl.where(m_causal, tl.exp(diff_la), 0.0)
    M_strict = tl.where(m_strict, M_sub, 0.0)
    base_decay = tl.exp(tl.minimum(la_cum, 0.0))

    x_init = tl.load(x_sub_init_ptr + ((i_b * NS + i_s) * H + i_h) * K + o_k).to(tl.float32)
    center = tl.load(log_atk_scale_ptr + i_h).to(tl.float32)

    k_off = (bos + t_idx[:, None]) * (H * K) + i_h * K + o_k[None, :]
    k_val = tl.load(k_ptr + k_off).to(tl.float32)
    dkp_val = tl.load(dk_tilde_ptr + k_off).to(tl.float32)

    k_sq = k_val * k_val
    u_val = beta_val[:, None] * k_sq
    x_t = base_decay[:, None] * x_init[None, :] + tl.dot(M_sub, u_val, allow_tf32=False)

    ell = tl.log(x_t + eps)
    r = ell - center
    one_plus_abs = 1.0 + tl.abs(r)
    s = r / one_plus_abs
    ds_dr = 1.0 / (one_plus_abs * one_plus_abs)
    m_mult = tl.exp(-logx * s)

    dk_local = dkp_val * m_mult
    g_m = dkp_val * k_val
    g_s = g_m * (-logx * m_mult)
    g_r = g_s * ds_dr

    if WRONG_NEG_CENTER_SIGN:
        g_center = tl.sum(g_r)
    else:
        g_center = -tl.sum(g_r)

    gx_t = g_r / (x_t + eps)
    gu_val = tl.dot(tl.trans(M_sub), gx_t, allow_tf32=False)

    dbeta_local = tl.sum(gu_val * k_sq, axis=1)
    dk_local += 2.0 * k_val * (gu_val * beta_val[:, None])

    dx_init_local = tl.sum(gx_t * base_decay[:, None], axis=0)
    g_base = tl.sum(gx_t * x_init[None, :], axis=1)
    gM_strict = tl.where(m_strict, tl.dot(gx_t, tl.trans(u_val), allow_tf32=False), 0.0)
    g_inner = M_strict * gM_strict

    gla_cum = g_base * base_decay + tl.sum(g_inner, axis=1) - tl.sum(g_inner, axis=0)
    dg_local = tl.cumsum(gla_cum, axis=0, reverse=True)

    tl.store(dk_atk_ptr + k_off, dk_local)
    tl.store(dx_sub_in_ptr + ((i_b * NS + i_s) * H + i_h) * K + o_k, dx_init_local)

    B_total = tl.num_programs(2).to(tl.int64) // H
    part_off = ((i_k * B_total + i_b) * T + t_idx) * H + i_h
    tl.store(dg_atk_part_ptr + part_off, dg_local)
    tl.store(dbeta_atk_part_ptr + part_off, dbeta_local)
    tl.store(dcenter_part_ptr + ((i_k * B_total + i_b) * NS + i_s) * H + i_h, g_center)


@triton.jit
def _atk_reverse_subchunk_scan_bwd_kernel(
    k_ptr, beta_atk_ptr, g_atk_ptr, x_sub_init_ptr, sa_sub_ptr,
    dx_sub_in_ptr, dxt_ptr,
    dk_atk_ptr, dg_atk_part_ptr, dbeta_atk_part_ptr, dx0_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    HAS_DXT: tl.constexpr,
):
    """
    Kernel 4: Reverse Cross-Subchunk Curvature Adjoint Scan (`0` spills).
    Propagates `dX_sub_out` backward across `NS = T // BC` sub-chunks (`s = NS-1 .. 0`),
    accumulating inter-subchunk contributions into `dk_atk`, `dg_atk_part`, `dbeta_atk_part`,
    and writing initial curvature-state gradient `dX_0`.
    Grid: `(cdiv(K, BK), B * H)`.
    """
    i_k = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NS = T // BC
    bos = i_b * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_c = tl.arange(0, BC)

    if HAS_DXT:
        b_dx_next = tl.load(dxt_ptr + (i_b * H + i_h) * K + o_k).to(tl.float32)
    else:
        b_dx_next = tl.zeros([BK], dtype=tl.float32)

    for s_rev in range(NS):
        s = NS - 1 - s_rev
        t_idx = s * BC + o_c
        sub_off = ((i_b * NS + s) * H + i_h) * K + o_k

        # b_dx_next is dL / dX_{s, end} entering from later sub-chunks (or terminal dX_T)
        g_val = tl.load(g_atk_ptr + (bos + t_idx) * H + i_h).to(tl.float32)
        beta_val = tl.load(beta_atk_ptr + (bos + t_idx) * H + i_h).to(tl.float32)
        sa_val = tl.load(sa_sub_ptr + (i_b * NS + s) * H + i_h).to(tl.float32)
        x_init = tl.load(x_sub_init_ptr + sub_off).to(tl.float32)

        exp_sa = tl.exp(tl.minimum(sa_val, 0.0))
        sfx_decay = tl.exp(tl.minimum(sa_val - tl.cumsum(g_val, axis=0), 0.0))

        k_off = (bos + t_idx[:, None]) * (H * K) + i_h * K + o_k[None, :]
        k_val = tl.load(k_ptr + k_off).to(tl.float32)
        k_sq = k_val * k_val
        u_val = beta_val[:, None] * k_sq

        # Gradient wrt a_sub is b_dx_next; gradient wrt sa_val from exp(sa)*x_init + sum(sfx*u)
        gsa_from_x = tl.sum(b_dx_next * x_init, axis=0) * exp_sa
        g_sfx = tl.sum(u_val * b_dx_next[None, :], axis=1) * sfx_decay
        gsa_total = gsa_from_x + tl.sum(g_sfx, axis=0)
        dg_inter = gsa_total - tl.cumsum(g_sfx, axis=0, reverse=True)

        gu_inter = sfx_decay[:, None] * b_dx_next[None, :]
        dbeta_inter = tl.sum(gu_inter * k_sq, axis=1)
        dk_inter = 2.0 * k_val * (gu_inter * beta_val[:, None])

        dk_prev = tl.load(dk_atk_ptr + k_off).to(tl.float32)
        tl.store(dk_atk_ptr + k_off, dk_prev + dk_inter)

        part_off = ((i_k * tl.num_programs(1) // H + i_b) * T + t_idx) * H + i_h
        dg_prev = tl.load(dg_atk_part_ptr + part_off).to(tl.float32)
        db_prev = tl.load(dbeta_atk_part_ptr + part_off).to(tl.float32)
        tl.store(dg_atk_part_ptr + part_off, dg_prev + dg_inter)
        tl.store(dbeta_atk_part_ptr + part_off, db_prev + dbeta_inter)

        dx_in = tl.load(dx_sub_in_ptr + sub_off).to(tl.float32)
        b_dx_next = exp_sa * b_dx_next + dx_in

    tl.store(dx0_ptr + (i_b * H + i_h) * K + o_k, b_dx_next)


# =============================================================================
# 3. Stage 2: Zero-Spill Asymmetric Preconditioned KDA Chunkwise Kernels (Kernels 5-16)
# =============================================================================

@triton.jit
def _pkda_fused_cumsum_g_fwd_kernel(
    g_in_ptr, g_cum_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Kernel 5: Chunk-Local Inclusive Cumulative Sum of KDA Log-Decay Gate `g` (`0` spills).
    Grid: `(T // BT, cdiv(K, BK), B * H)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    off = (bos + o_t[:, None]) * (H * K) + i_h * K + o_k[None, :]

    b_g = tl.load(g_in_ptr + off).to(tl.float32)
    b_gcum = tl.cumsum(b_g, axis=0)
    tl.store(g_cum_ptr + off, b_gcum)


@triton.jit
def _pkda_overflow_free_intra_diag_subchunk_fwd_kernel(
    q_ptr, k_ptr, k_tilde_ptr, g_cum_ptr, beta_ptr,
    Aqk_ptr, Akk_diag_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    SYM_AKK_BUG: tl.constexpr = False,
):
    """
    Kernel 6: Pre-Masked Non-Positive Causal 16x16 Diagonal Sub-Chunk Forward & Forward-Substitution
    Inversion (`0` spills, `0` `NaN`s across all non-positive decay rates).
      - `Aqk_diag[r, c] = scale * <q_r * exp(g_r - g_c), k_tilde_c>` for `r >= c`
      - `Akk_diag[r, c] = beta_r * <k_r * exp(g_r - g_c), k_tilde_c>` for `r > c`
      - Inverts `(I + Akk_diag)^{-1}` in-register via exact 16-step forward substitution.
    Grid: `(T // BT, BT // BC, B * H)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_sub = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    t_base = i_t * BT + i_sub * BC
    o_r = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    t_rows = t_base + o_r

    b_beta = tl.load(beta_ptr + (bos + t_rows) * H + i_h).to(tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        k_cols = i_k * BK + o_k
        off_rk = (bos + t_rows[:, None]) * (H * K) + i_h * K + k_cols[None, :]
        b_q = tl.load(q_ptr + off_rk).to(tl.float32)
        b_k = tl.load(k_ptr + off_rk).to(tl.float32)
        b_kt = b_k if SYM_AKK_BUG else tl.load(k_tilde_ptr + off_rk).to(tl.float32)
        b_kt_q = tl.load(k_tilde_ptr + off_rk).to(tl.float32)
        b_g = tl.load(g_cum_ptr + off_rk).to(tl.float32)

        for c in range(BC):
            tc = t_base + c
            off_ck = (bos + tc) * (H * K) + i_h * K + k_cols
            b_kt_c = tl.load(k_ptr + off_ck if SYM_AKK_BUG else k_tilde_ptr + off_ck).to(tl.float32)
            b_kt_qc = tl.load(k_tilde_ptr + off_ck).to(tl.float32)
            b_gc = tl.load(g_cum_ptr + off_ck).to(tl.float32)

            m_inc = o_r >= c
            m_exc = o_r > c
            diff_inc = tl.where(m_inc[:, None], b_g - b_gc[None, :], -1.0e30)
            exp_inc = tl.where(m_inc[:, None], tl.exp(diff_inc), 0.0)
            exp_exc = tl.where(m_exc[:, None], exp_inc, 0.0)

            s_qk = tl.sum((b_q * exp_inc) * b_kt_qc[None, :], axis=1)
            s_kk = tl.sum((b_k * exp_exc) * b_kt_c[None, :], axis=1)

            col_mask = o_r[None, :] == c
            b_Aqk += tl.where(col_mask, s_qk[:, None], 0.0)
            b_Akk += tl.where(col_mask, s_kk[:, None], 0.0)

    b_Aqk = b_Aqk * scale
    b_Akk = b_Akk * b_beta[:, None]

    # Store diagonal 16x16 block into Aqk
    aqk_off = (bos + t_rows[:, None]) * (H * BT) + i_h * BT + (i_sub * BC + o_r[None, :])
    tl.store(Aqk_ptr + aqk_off, b_Aqk)

    # Exact in-register 16x16 forward substitution for (I + Akk_diag)^{-1}
    m_I = o_r[:, None] == o_r[None, :]
    b_Ai = -b_Akk
    for r in range(2, BC):
        row_mask = o_r == r
        a_row = tl.sum(tl.where(row_mask[:, None], b_Ai, 0.0), axis=0)
        new_row = a_row + tl.sum(a_row[:, None] * b_Ai, axis=0)
        b_Ai = tl.where(row_mask[:, None], new_row[None, :], b_Ai)
    b_Ai = b_Ai + m_I.to(tl.float32)

    akk_diag_off = (bos + t_rows[:, None]) * (H * BC) + i_h * BC + o_r[None, :]
    tl.store(Akk_diag_ptr + akk_diag_off, b_Ai)


@triton.jit
def _pkda_zero_spill_inter_subchunk_Aqk_fwd_kernel(
    q_ptr, k_ptr, k_tilde_ptr, g_cum_ptr, beta_ptr, Aqk_ptr, Akk_inv_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    SYM_AKK_BUG: tl.constexpr = False,
):
    """
    Kernel 7: Zero-Spill Off-Diagonal 16x16 Asymmetric `Aqk` & `Akk` Sub-Chunk Forward Kernel (`0` spills).
    Anchored at `g_cum[i_t*BT + r_sub*BC - 1]` so `g_r - g_anc <= 0` and `g_anc - g_c <= 0` strictly!
    Grid: `(T // BT, B * H)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T
    t_chunk = i_t * BT
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    kt_src = k_ptr if SYM_AKK_BUG else k_tilde_ptr

    for r_sub in range(1, 4):
        t_r = t_chunk + r_sub * BC + o_i
        t_anc = t_chunk + r_sub * BC - 1
        b_br = tl.load(beta_ptr + (bos + t_r) * H + i_h).to(tl.float32)
        for c_sub in range(0, r_sub):
            t_c = t_chunk + c_sub * BC + o_i
            b_acc_qk = tl.zeros([BC, BC], dtype=tl.float32)
            b_acc_kk = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                k_cols = i_k * BK + o_k
                off_r = (bos + t_r[:, None]) * (H * K) + i_h * K + k_cols[None, :]
                off_c = (bos + t_c[:, None]) * (H * K) + i_h * K + k_cols[None, :]
                off_a = (bos + t_anc) * (H * K) + i_h * K + k_cols

                b_q = tl.load(q_ptr + off_r).to(tl.float32)
                b_kr = tl.load(k_ptr + off_r).to(tl.float32)
                b_ktc = tl.load(kt_src + off_c).to(tl.float32)
                b_gr = tl.load(g_cum_ptr + off_r).to(tl.float32)
                b_gc = tl.load(g_cum_ptr + off_c).to(tl.float32)
                b_ga = tl.load(g_cum_ptr + off_a).to(tl.float32)

                b_dec_r = tl.exp(tl.minimum(b_gr - b_ga[None, :], 0.0))
                b_ktg = tl.trans(b_ktc * tl.exp(tl.minimum(b_ga[None, :] - b_gc, 0.0)))
                b_acc_qk += tl.dot(b_q * b_dec_r, b_ktg, allow_tf32=False)
                b_acc_kk += tl.dot(b_kr * b_dec_r, b_ktg, allow_tf32=False)

            blk_off = (bos + t_r[:, None]) * (H * BT) + i_h * BT + (c_sub * BC + o_i[None, :])
            tl.store(Aqk_ptr + blk_off, b_acc_qk * scale)
            tl.store(Akk_inv_ptr + blk_off, b_acc_kk * b_br[:, None])


@triton.jit
def _pkda_zero_spill_inter_subchunk_Akk_solve_fwd_kernel(
    Akk_diag_ptr, Akk_inv_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """
    Kernel 8: Zero-Spill 4x4 Block-Triangular Merge Solve (`0` spills
    vs upstream's `62-246` spills in `chunk_precond_kda_fwd_kernel_inter_solve_fused`).
    Grid: `(T // BT, B * H)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T
    t_chunk = i_t * BT
    o_i = tl.arange(0, BC)

    t_c0 = t_chunk + o_i
    t_c1 = t_chunk + BC + o_i
    t_c2 = t_chunk + 2 * BC + o_i
    t_c3 = t_chunk + 3 * BC + o_i

    base_inv = Akk_inv_ptr + i_h * BT
    b_Ai00 = tl.load(Akk_diag_ptr + (bos + t_c0[:, None]) * (H * BC) + i_h * BC + o_i[None, :]).to(tl.float32)
    tl.store(base_inv + (bos + t_c0[:, None]) * (H * BT) + o_i[None, :], b_Ai00)

    b_Ai11 = tl.load(Akk_diag_ptr + (bos + t_c1[:, None]) * (H * BC) + i_h * BC + o_i[None, :]).to(tl.float32)
    b_Akk10 = tl.load(base_inv + (bos + t_c1[:, None]) * (H * BT) + o_i[None, :]).to(tl.float32)
    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10, allow_tf32=False), b_Ai00, allow_tf32=False)
    tl.store(base_inv + (bos + t_c1[:, None]) * (H * BT) + o_i[None, :], b_Ai10)
    tl.store(base_inv + (bos + t_c1[:, None]) * (H * BT) + (BC + o_i[None, :]), b_Ai11)

    b_Ai22 = tl.load(Akk_diag_ptr + (bos + t_c2[:, None]) * (H * BC) + i_h * BC + o_i[None, :]).to(tl.float32)
    b_Akk21 = tl.load(base_inv + (bos + t_c2[:, None]) * (H * BT) + (BC + o_i[None, :])).to(tl.float32)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_Akk21, allow_tf32=False), b_Ai11, allow_tf32=False)
    b_Akk20 = tl.load(base_inv + (bos + t_c2[:, None]) * (H * BT) + o_i[None, :]).to(tl.float32)
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, allow_tf32=False) + tl.dot(b_Akk21, b_Ai10, allow_tf32=False),
        allow_tf32=False,
    )
    tl.store(base_inv + (bos + t_c2[:, None]) * (H * BT) + o_i[None, :], b_Ai20)
    tl.store(base_inv + (bos + t_c2[:, None]) * (H * BT) + (BC + o_i[None, :]), b_Ai21)
    tl.store(base_inv + (bos + t_c2[:, None]) * (H * BT) + (2 * BC + o_i[None, :]), b_Ai22)

    b_Ai33 = tl.load(Akk_diag_ptr + (bos + t_c3[:, None]) * (H * BC) + i_h * BC + o_i[None, :]).to(tl.float32)
    b_Akk32 = tl.load(base_inv + (bos + t_c3[:, None]) * (H * BT) + (2 * BC + o_i[None, :])).to(tl.float32)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_Akk32, allow_tf32=False), b_Ai22, allow_tf32=False)
    b_Akk31 = tl.load(base_inv + (bos + t_c3[:, None]) * (H * BT) + (BC + o_i[None, :])).to(tl.float32)
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, allow_tf32=False) + tl.dot(b_Akk32, b_Ai21, allow_tf32=False),
        allow_tf32=False,
    )
    b_Akk30 = tl.load(base_inv + (bos + t_c3[:, None]) * (H * BT) + o_i[None, :]).to(tl.float32)
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, allow_tf32=False)
        + tl.dot(b_Akk31, b_Ai10, allow_tf32=False)
        + tl.dot(b_Akk32, b_Ai20, allow_tf32=False),
        allow_tf32=False,
    )
    tl.store(base_inv + (bos + t_c3[:, None]) * (H * BT) + o_i[None, :], b_Ai30)
    tl.store(base_inv + (bos + t_c3[:, None]) * (H * BT) + (BC + o_i[None, :]), b_Ai31)
    tl.store(base_inv + (bos + t_c3[:, None]) * (H * BT) + (2 * BC + o_i[None, :]), b_Ai32)
    tl.store(base_inv + (bos + t_c3[:, None]) * (H * BT) + (3 * BC + o_i[None, :]), b_Ai33)


@triton.jit
def _pkda_wy_wu_qg_kg_fwd_kernel(
    q_ptr, k_ptr, k_tilde_ptr, v_ptr, g_cum_ptr, beta_ptr, Akk_inv_ptr,
    w_ptr, u_ptr, qg_ptr, kg_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 9: Row-Subtiled (`BM = 16`) Asymmetric WY Forward Kernel (`0` spills vs upstream's `44-78` spills).
      - `w = M_inv @ (k * beta * exp(g))` (uses original `k`)
      - `u = M_inv @ (v * beta)`
      - `qg = q * exp(g)`
      - `kg = k_tilde * exp(g_last - g)` (uses curvature-preconditioned `k_tilde`)
    Grid: `((T // BT) * (BT // BM), B * H)`.
    """
    i_tm = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_m = i_tm % NSUB
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_m = i_t * BT + i_m * BM + tl.arange(0, BM)
    o_col = tl.arange(0, BT)
    b_M = tl.load(Akk_inv_ptr + (bos + o_m[:, None]) * (H * BT) + i_h * BT + o_col[None, :]).to(tl.float32)

    o_c = tl.arange(0, BM)
    for i_v in range(tl.cdiv(V, BV)):
        v_cols = i_v * BV + tl.arange(0, BV)
        b_u = tl.zeros([BM, BV], dtype=tl.float32)
        for s_col in range(0, i_m + 1):
            t_col = i_t * BT + s_col * BM + o_c
            b_Ms = tl.load(Akk_inv_ptr + (bos + o_m[:, None]) * (H * BT) + i_h * BT + (s_col * BM + o_c[None, :])).to(tl.float32)
            b_v = tl.load(v_ptr + (bos + t_col[:, None]) * (H * V) + i_h * V + v_cols[None, :]).to(tl.float32)
            b_b = tl.load(beta_ptr + (bos + t_col) * H + i_h).to(tl.float32)
            b_u += tl.dot(b_Ms, b_v * b_b[:, None], allow_tf32=False)
        tl.store(u_ptr + (bos + o_m[:, None]) * (H * V) + i_h * V + v_cols[None, :], b_u.to(u_ptr.dtype.element_ty))

    t_last = i_t * BT + BT - 1
    for i_k in range(tl.cdiv(K, BK)):
        k_cols = i_k * BK + tl.arange(0, BK)
        b_w = tl.zeros([BM, BK], dtype=tl.float32)
        for s_col in range(0, i_m + 1):
            t_col = i_t * BT + s_col * BM + o_c
            b_Ms = tl.load(Akk_inv_ptr + (bos + o_m[:, None]) * (H * BT) + i_h * BT + (s_col * BM + o_c[None, :])).to(tl.float32)
            off_ck = (bos + t_col[:, None]) * (H * K) + i_h * K + k_cols[None, :]
            b_k = tl.load(k_ptr + off_ck).to(tl.float32)
            b_g = tl.load(g_cum_ptr + off_ck).to(tl.float32)
            b_b = tl.load(beta_ptr + (bos + t_col) * H + i_h).to(tl.float32)
            b_kbg = b_k * b_b[:, None] * tl.exp(tl.minimum(b_g, 0.0))
            b_w += tl.dot(b_Ms, b_kbg, allow_tf32=False)

        off_mk = (bos + o_m[:, None]) * (H * K) + i_h * K + k_cols[None, :]
        tl.store(w_ptr + off_mk, b_w.to(w_ptr.dtype.element_ty))

        b_qm = tl.load(q_ptr + off_mk).to(tl.float32)
        b_ktm = tl.load(k_tilde_ptr + off_mk).to(tl.float32)
        b_gm = tl.load(g_cum_ptr + off_mk).to(tl.float32)
        b_glast = tl.load(g_cum_ptr + (bos + t_last) * (H * K) + i_h * K + k_cols).to(tl.float32)

        b_qg = b_qm * tl.exp(tl.minimum(b_gm, 0.0))
        b_kg = b_ktm * tl.exp(tl.minimum(b_glast[None, :] - b_gm, 0.0))
        tl.store(qg_ptr + off_mk, b_qg.to(qg_ptr.dtype.element_ty))
        tl.store(kg_ptr + off_mk, b_kg.to(kg_ptr.dtype.element_ty))


@triton.jit
def _pkda_chunk_state_h_fwd_kernel(
    kg_ptr, w_ptr, u_ptr, g_cum_ptr, h0_ptr,
    h_all_ptr, v_new_ptr, s_final_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_H0: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    """
    Kernel 10: L2-Staged Zero-Spill Inter-Chunk Asymmetric State Recurrence (`BK = 32, BV = 32`, `0` spills):
      `v_new = u - w @ S_c`, `S_{c+1} = Diag(exp(g_last)) S_c + kg^T @ v_new`.
    Grid: `(cdiv(V, BV), B * H)`.
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT
    bos = i_b * T
    o_v = i_v * BV + tl.arange(0, BV)
    s_base = (i_b * H + i_h) * (K * V)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        kv_off = o_k[:, None] * V + o_v[None, :]
        if HAS_H0:
            b_s = tl.load(h0_ptr + s_base + kv_off).to(tl.float32)
        else:
            b_s = tl.zeros([BK, BV], dtype=tl.float32)
        tl.store(s_final_ptr + s_base + kv_off, b_s)

    o_t = tl.arange(0, BT)
    for i_t in range(NT):
        t_rows = bos + i_t * BT + o_t
        h_base = ((i_b * NT + i_t) * H + i_h) * (K * V)
        v_off = t_rows[:, None] * (H * V) + i_h * V + o_v[None, :]
        b_vnew = tl.load(u_ptr + v_off).to(tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            kv_off = o_k[:, None] * V + o_v[None, :]
            b_s = tl.load(s_final_ptr + s_base + kv_off).to(tl.float32)
            tl.store(h_all_ptr + h_base + kv_off, b_s.to(h_all_ptr.dtype.element_ty))
            w_off = t_rows[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_w = tl.load(w_ptr + w_off).to(tl.float32)
            b_vnew = b_vnew - tl.dot(b_w, b_s, allow_tf32=not IS_FP32)

        tl.store(v_new_ptr + v_off, b_vnew.to(v_new_ptr.dtype.element_ty))
        last_row = bos + i_t * BT + BT - 1

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            kv_off = o_k[:, None] * V + o_v[None, :]
            b_s = tl.load(s_final_ptr + s_base + kv_off).to(tl.float32)
            b_gn = tl.load(g_cum_ptr + last_row * (H * K) + i_h * K + o_k).to(tl.float32)
            w_off = t_rows[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_kg = tl.load(kg_ptr + w_off).to(tl.float32)
            b_s = b_s * tl.exp(tl.minimum(b_gn, 0.0))[:, None] + tl.dot(tl.trans(b_kg), b_vnew, allow_tf32=not IS_FP32)
            tl.store(s_final_ptr + s_base + kv_off, b_s)


@triton.jit
def _pkda_chunk_readout_o_fwd_kernel(
    qg_ptr, v_new_ptr, Aqk_ptr, h_all_ptr, o_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 11: Row-Subtiled (`BM = 16, BK = 32, BV = 32`) Chunk Readout Kernel (`0` spills):
      `o = scale * (qg @ S_c) + Aqk @ v_new`.
    Grid: `(cdiv(V, BV), (T // BT) * (BT // BM), B * H)`.
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_tm = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_m = i_tm % NSUB
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT
    bos = i_b * T

    o_m = i_t * BT + i_m * BM + tl.arange(0, BM)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BM)
    o_k = tl.arange(0, BK)

    b_o = tl.zeros([BM, BV], dtype=tl.float32)
    h_base = ((i_b * NT + i_t) * H + i_h) * (K * V)

    for i_k in range(tl.cdiv(K, BK)):
        k_cols = i_k * BK + o_k
        b_qg = tl.load(qg_ptr + (bos + o_m[:, None]) * (H * K) + i_h * K + k_cols[None, :]).to(tl.float32)
        b_h = tl.load(h_all_ptr + h_base + k_cols[:, None] * V + o_v[None, :]).to(tl.float32)
        b_o += tl.dot(b_qg, b_h, allow_tf32=False) * scale

    for s_col in range(0, i_m + 1):
        t_col = i_t * BT + s_col * BM + o_c
        b_Aqk = tl.load(Aqk_ptr + (bos + o_m[:, None]) * (H * BT) + i_h * BT + (s_col * BM + o_c[None, :])).to(tl.float32)
        b_vn = tl.load(v_new_ptr + (bos + t_col[:, None]) * (H * V) + i_h * V + o_v[None, :]).to(tl.float32)
        b_o += tl.dot(b_Aqk, b_vn, allow_tf32=False)

    tl.store(o_ptr + (bos + o_m[:, None]) * (H * V) + i_h * V + o_v[None, :], b_o.to(o_ptr.dtype.element_ty))


@triton.jit
def _pkda_bwd_dAv_kernel(
    v_new_ptr, Aqk_ptr, do_ptr, dv_new_intra_ptr, dAqk_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 12: Row-Subtiled (`BM = 16`) Backward for `dAqk = tril(do @ v_new^T)` and `dv_new_intra = Aqk^T @ do` (`0` spills).
    Grid: `((T // BT) * (BT // BM), B * H)`.
    """
    i_tm = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_m = i_tm % NSUB
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_r = tl.arange(0, BM)
    o_c = tl.arange(0, BT)
    t_m = i_t * BT + i_m * BM + o_r
    t_col = i_t * BT + o_c

    b_dAqk = tl.zeros([BM, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        v_cols = i_v * BV + tl.arange(0, BV)
        b_do = tl.load(do_ptr + (bos + t_m[:, None]) * (H * V) + i_h * V + v_cols[None, :]).to(tl.float32)
        b_vn = tl.load(v_new_ptr + (bos + t_col[:, None]) * (H * V) + i_h * V + v_cols[None, :]).to(tl.float32)
        b_dAqk += tl.dot(b_do, tl.trans(b_vn), allow_tf32=False)

    m_causal = (i_m * BM + o_r[:, None]) >= o_c[None, :]
    b_dAqk = tl.where(m_causal, b_dAqk, 0.0)
    tl.store(dAqk_ptr + (bos + t_m[:, None]) * (H * BT) + i_h * BT + o_c[None, :], b_dAqk)

    # Also compute dv_new_intra[t_m, :] = sum_{r >= m} Aqk[r, t_m]^T @ do[r, :]
    b_Aqk_T = tl.where(
        o_c[:, None] >= (i_m * BM + o_r[None, :]),
        tl.load(Aqk_ptr + (bos + t_col[:, None]) * (H * BT) + i_h * BT + (i_m * BM + o_r[None, :])).to(tl.float32),
        0.0,
    )
    for i_v in range(tl.cdiv(V, BV)):
        v_cols = i_v * BV + tl.arange(0, BV)
        b_do_all = tl.load(do_ptr + (bos + t_col[:, None]) * (H * V) + i_h * V + v_cols[None, :]).to(tl.float32)
        b_dvn = tl.dot(tl.trans(b_Aqk_T), b_do_all, allow_tf32=False)
        tl.store(
            dv_new_intra_ptr + (bos + t_m[:, None]) * (H * V) + i_h * V + v_cols[None, :],
            b_dvn.to(dv_new_intra_ptr.dtype.element_ty),
        )


@triton.jit
def _pkda_bwd_dhu_reverse_kernel(
    qg_ptr, kg_ptr, w_ptr, g_cum_ptr, do_ptr, dv_new_intra_ptr, dht_ptr,
    dh_all_ptr, du_total_ptr, dh0_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_DHT: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    """
    Kernel 13: L2-Staged Zero-Spill Reverse Inter-Chunk State Adjoint (`dS_c`, `du`, `dS_0`) (`0` spills).
    Grid: `(cdiv(V, BV), B * H)`.
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT
    bos = i_b * T
    o_v = i_v * BV + tl.arange(0, BV)
    s_base = (i_b * H + i_h) * (K * V)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        kv_off = o_k[:, None] * V + o_v[None, :]
        if HAS_DHT:
            b_ds = tl.load(dht_ptr + s_base + kv_off).to(tl.float32)
        else:
            b_ds = tl.zeros([BK, BV], dtype=tl.float32)
        tl.store(dh0_ptr + s_base + kv_off, b_ds)

    o_t = tl.arange(0, BT)
    for step in range(NT):
        i_t = NT - 1 - step
        t_rows = bos + i_t * BT + o_t
        dh_base = ((i_b * NT + i_t) * H + i_h) * (K * V)
        v_off = t_rows[:, None] * (H * V) + i_h * V + o_v[None, :]
        b_du = tl.load(dv_new_intra_ptr + v_off).to(tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            kv_off = o_k[:, None] * V + o_v[None, :]
            b_ds = tl.load(dh0_ptr + s_base + kv_off).to(tl.float32)
            tl.store(dh_all_ptr + dh_base + kv_off, b_ds.to(dh_all_ptr.dtype.element_ty))
            k_off = t_rows[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_kg = tl.load(kg_ptr + k_off).to(tl.float32)
            b_du = b_du + tl.dot(b_kg, b_ds, allow_tf32=not IS_FP32)

        tl.store(du_total_ptr + v_off, b_du.to(du_total_ptr.dtype.element_ty))
        b_do = tl.load(do_ptr + v_off).to(tl.float32) * scale
        last_row = bos + i_t * BT + BT - 1

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            kv_off = o_k[:, None] * V + o_v[None, :]
            b_ds = tl.load(dh0_ptr + s_base + kv_off).to(tl.float32)
            b_gn = tl.load(g_cum_ptr + last_row * (H * K) + i_h * K + o_k).to(tl.float32)
            k_off = t_rows[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_qg = tl.load(qg_ptr + k_off).to(tl.float32)
            b_w = tl.load(w_ptr + k_off).to(tl.float32)
            b_ds = (
                b_ds * tl.exp(tl.minimum(b_gn, 0.0))[:, None]
                + tl.dot(tl.trans(b_qg), b_do, allow_tf32=not IS_FP32)
                - tl.dot(tl.trans(b_w), b_du, allow_tf32=not IS_FP32)
            )
            tl.store(dh0_ptr + s_base + kv_off, b_ds)


@triton.jit
def _pkda_bwd_wy_dqk_slice_kernel(
    v_new_ptr, g_cum_ptr, h_all_ptr, do_ptr, dh_all_ptr, du_total_ptr,
    dqg_ptr, dkg_ptr, dw_ptr, dg_last_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 14: Row-Subtiled (`BM = 16, BK = 32`) Inter-Chunk & WY Backward (`0` spills vs upstream's `168-3,972` spills!).
    Computes `dqg = scale * (do @ S_c^T)`, `dkg = v_new @ dS_{c+1}^T`, `dw = -du @ S_c^T`, and
    state-decay part of `dg_last = <S_c * exp(g_last), dS_{c+1}>`.
    Grid: `((T // BT) * (BT // BM), cdiv(K, BK), B * H)`.
    """
    i_tm = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_m = i_tm % NSUB
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT
    bos = i_b * T

    o_m = i_t * BT + i_m * BM + tl.arange(0, BM)
    o_k = i_k * BK + tl.arange(0, BK)

    b_dqg = tl.zeros([BM, BK], dtype=tl.float32)
    b_dkg = tl.zeros([BM, BK], dtype=tl.float32)
    b_dw = tl.zeros([BM, BK], dtype=tl.float32)
    b_dgl = tl.zeros([BK], dtype=tl.float32)

    h_base = ((i_b * NT + i_t) * H + i_h) * (K * V)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        b_h = tl.load(h_all_ptr + h_base + o_k[:, None] * V + o_v[None, :]).to(tl.float32)
        b_dh = tl.load(dh_all_ptr + h_base + o_k[:, None] * V + o_v[None, :]).to(tl.float32)
        b_do = tl.load(do_ptr + (bos + o_m[:, None]) * (H * V) + i_h * V + o_v[None, :]).to(tl.float32)
        b_vn = tl.load(v_new_ptr + (bos + o_m[:, None]) * (H * V) + i_h * V + o_v[None, :]).to(tl.float32)
        b_du = tl.load(du_total_ptr + (bos + o_m[:, None]) * (H * V) + i_h * V + o_v[None, :]).to(tl.float32)

        b_dqg += tl.dot(b_do, tl.trans(b_h), allow_tf32=False) * scale
        b_dkg += tl.dot(b_vn, tl.trans(b_dh), allow_tf32=False)
        b_dw -= tl.dot(b_du, tl.trans(b_h), allow_tf32=False)
        if i_m == 0:
            b_dgl += tl.sum(b_h * b_dh, axis=1)

    off_mk = (bos + o_m[:, None]) * (H * K) + i_h * K + o_k[None, :]
    tl.store(dqg_ptr + off_mk, b_dqg)
    tl.store(dkg_ptr + off_mk, b_dkg)
    tl.store(dw_ptr + off_mk, b_dw.to(dw_ptr.dtype.element_ty))

    if i_m == 0:
        t_last = i_t * BT + BT - 1
        b_gl = tl.load(g_cum_ptr + (bos + t_last) * (H * K) + i_h * K + o_k).to(tl.float32)
        b_dgl = b_dgl * tl.exp(tl.minimum(b_gl, 0.0))
        tl.store(dg_last_ptr + ((i_b * NT + i_t) * H + i_h) * K + o_k, b_dgl)


@triton.jit
def _pkda_bwd_wy_dAkk_dv_db_kernel(
    Akk_inv_ptr, k_ptr, v_ptr, g_cum_ptr, beta_ptr, dw_ptr, du_total_ptr,
    dv_out_ptr, dk_wy_ptr, db_wy_ptr, dM_raw_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Kernel 15: Row-Subtiled (`BM = 16`) WY Backward for `dv`, `dk_wy`, `db_wy`, and `dM_raw` (`0` spills).
    Grid: `((T // BT) * (BT // BM), B * H)`.
    """
    i_tm = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_m = i_tm % NSUB
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_r = tl.arange(0, BM)
    o_m = i_t * BT + i_m * BM + o_r
    b_b_m = tl.load(beta_ptr + (bos + o_m) * H + i_h).to(tl.float32)
    b_db_wy = tl.zeros([BM], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        v_cols = i_v * BV + tl.arange(0, BV)
        b_dvb = tl.zeros([BM, BV], dtype=tl.float32)
        for s_col in range(i_m, NSUB):
            t_col = i_t * BT + s_col * BM + o_r
            b_MT_s = tl.load(
                Akk_inv_ptr + (bos + t_col[None, :]) * (H * BT) + i_h * BT + (i_m * BM + o_r[:, None])
            ).to(tl.float32)
            b_du_s = tl.load(du_total_ptr + (bos + t_col[:, None]) * (H * V) + i_h * V + v_cols[None, :]).to(tl.float32)
            b_dvb += tl.dot(b_MT_s, b_du_s, allow_tf32=False)

        b_vm = tl.load(v_ptr + (bos + o_m[:, None]) * (H * V) + i_h * V + v_cols[None, :]).to(tl.float32)
        b_dv = b_dvb * b_b_m[:, None]
        b_db_wy += tl.sum(b_dvb * b_vm, axis=1)
        tl.store(dv_out_ptr + (bos + o_m[:, None]) * (H * V) + i_h * V + v_cols[None, :], b_dv.to(dv_out_ptr.dtype.element_ty))

    for i_k in range(tl.cdiv(K, BK)):
        k_cols = i_k * BK + tl.arange(0, BK)
        b_dkgb = tl.zeros([BM, BK], dtype=tl.float32)
        for s_col in range(i_m, NSUB):
            t_col = i_t * BT + s_col * BM + o_r
            b_MT_s = tl.load(
                Akk_inv_ptr + (bos + t_col[None, :]) * (H * BT) + i_h * BT + (i_m * BM + o_r[:, None])
            ).to(tl.float32)
            b_dw_s = tl.load(dw_ptr + (bos + t_col[:, None]) * (H * K) + i_h * K + k_cols[None, :]).to(tl.float32)
            b_dkgb += tl.dot(b_MT_s, b_dw_s, allow_tf32=False)

        off_mk = (bos + o_m[:, None]) * (H * K) + i_h * K + k_cols[None, :]
        b_km = tl.load(k_ptr + off_mk).to(tl.float32)
        b_gm = tl.load(g_cum_ptr + off_mk).to(tl.float32)
        b_exp_gm = tl.exp(tl.minimum(b_gm, 0.0))

        b_dk_wy = b_dkgb * (b_b_m[:, None] * b_exp_gm)
        b_db_wy += tl.sum(b_dkgb * (b_km * b_exp_gm), axis=1)
        tl.store(dk_wy_ptr + off_mk, b_dk_wy)

    tl.store(db_wy_ptr + (bos + o_m) * H + i_h, b_db_wy)

    for s_col in range(NSUB):
        t_col = i_t * BT + s_col * BM + o_r
        b_b_col = tl.load(beta_ptr + (bos + t_col) * H + i_h).to(tl.float32)
        b_dM_sub = tl.zeros([BM, BM], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            v_cols = i_v * BV + tl.arange(0, BV)
            b_du_m = tl.load(du_total_ptr + (bos + o_m[:, None]) * (H * V) + i_h * V + v_cols[None, :]).to(tl.float32)
            b_v_c = tl.load(v_ptr + (bos + t_col[:, None]) * (H * V) + i_h * V + v_cols[None, :]).to(tl.float32)
            b_dM_sub += tl.dot(b_du_m, tl.trans(b_v_c * b_b_col[:, None]), allow_tf32=False)
        for i_k in range(tl.cdiv(K, BK)):
            k_cols = i_k * BK + tl.arange(0, BK)
            b_dw_m = tl.load(dw_ptr + (bos + o_m[:, None]) * (H * K) + i_h * K + k_cols[None, :]).to(tl.float32)
            off_ck = (bos + t_col[:, None]) * (H * K) + i_h * K + k_cols[None, :]
            b_k_c = tl.load(k_ptr + off_ck).to(tl.float32)
            b_g_c = tl.load(g_cum_ptr + off_ck).to(tl.float32)
            b_kbg_c = b_k_c * (b_b_col[:, None] * tl.exp(tl.minimum(b_g_c, 0.0)))
            b_dM_sub += tl.dot(b_dw_m, tl.trans(b_kbg_c), allow_tf32=False)
        tl.store(dM_raw_ptr + (bos + o_m[:, None]) * (H * BT) + i_h * BT + (s_col * BM + o_r[None, :]), b_dM_sub)


@triton.jit
def _pkda_bwd_wy_dAkk_solve_kernel(
    Akk_inv_ptr, dM_raw_ptr, dAkk_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    OMIT_OUTER_STRICT_MASK: tl.constexpr = False,
):
    """
    Kernel 16: 16x16 Tile-Parallel Inverse Adjoint `dAkk = strictTril(-M_inv^T @ strictTril(dM_raw) @ M_inv^T)` (`0` spills).
    Grid: `((T // BT) * (BT // BM), BT // BM, B * H)`.
    """
    i_tm = tl.program_id(0).to(tl.int64)
    i_n = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NSUB = BT // BM
    i_t = i_tm // NSUB
    i_m = i_tm % NSUB
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_r = tl.arange(0, BM)
    o_c = tl.arange(0, BM)
    row_idx = i_m * BM + o_r
    col_idx = i_n * BM + o_c
    t_row = i_t * BT + row_idx

    if i_m < i_n and not OMIT_OUTER_STRICT_MASK:
        tl.store(dAkk_ptr + (bos + t_row[:, None]) * (H * BT) + i_h * BT + col_idx[None, :], tl.zeros([BM, BM], dtype=tl.float32))
        return

    o_full = tl.arange(0, BT)
    t_full = i_t * BT + o_full

    # MT_left = M_inv[t_full, row_idx]^T -> [BM, BT]
    b_MT_left = tl.load(
        Akk_inv_ptr + (bos + t_full[None, :]) * (H * BT) + i_h * BT + row_idx[:, None]
    ).to(tl.float32)

    # MT_right = M_inv[col_idx, t_full]^T -> [BT, BM]
    b_MT_right = tl.load(
        Akk_inv_ptr + (bos + (i_t * BT + col_idx)[None, :]) * (H * BT) + i_h * BT + o_full[:, None]
    ).to(tl.float32)

    b_tmp = tl.zeros([BM, BM], dtype=tl.float32)
    for s_sub in range(NSUB):
        s_idx = s_sub * BM + o_c
        t_s = i_t * BT + s_idx
        b_dM_col = tl.load(
            dM_raw_ptr + (bos + t_full[:, None]) * (H * BT) + i_h * BT + s_idx[None, :]
        ).to(tl.float32)
        m_strict_in = o_full[:, None] > s_idx[None, :]
        b_dM_col = tl.where(m_strict_in, b_dM_col, 0.0)
        b_L = tl.dot(b_MT_left, b_dM_col, allow_tf32=False)
        b_R = tl.load(
            Akk_inv_ptr + (bos + (i_t * BT + col_idx)[None, :]) * (H * BT) + i_h * BT + s_idx[:, None]
        ).to(tl.float32)
        b_tmp += tl.dot(b_L, b_R, allow_tf32=False)

    b_dAkk = -b_tmp
    if not OMIT_OUTER_STRICT_MASK:
        m_strict_out = row_idx[:, None] > col_idx[None, :]
        b_dAkk = tl.where(m_strict_out, b_dAkk, 0.0)

    tl.store(dAkk_ptr + (bos + t_row[:, None]) * (H * BT) + i_h * BT + col_idx[None, :], b_dAkk)


@triton.jit
def _pkda_bwd_intra_subchunk_asymmetric_kernel(
    q_ptr, k_ptr, k_tilde_ptr, g_cum_ptr, beta_ptr,
    dAqk_ptr, dAkk_ptr, dqg_ptr, dkg_ptr, dk_wy_ptr,
    dq_out_ptr, dk_stage2_ptr, dk_tilde_out_ptr, db_part_ptr, dg_sum_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Kernel 17: Pre-Masked Non-Positive Causal Asymmetric Sub-Chunk Backward (`0` spills):
      - Row of `dAqk` -> `dq`; Row of `dAkk` -> `dk_stage2` + `dbeta`
      - Column of BOTH `dAqk` and `dAkk` -> `dk_tilde`
      - Exact Asymmetric Homogeneous Euler Gate Identity:
        `dg_sum = q * dq + k * dk_stage2 - k_tilde * dk_tilde`
    Grid: `(T // BT, BT // BC, cdiv(K, BK) * B * H)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_sub = tl.program_id(1).to(tl.int64)
    i_kbh = tl.program_id(2).to(tl.int64)
    NK = K // BK
    i_k = i_kbh // (tl.num_programs(2) // NK)
    i_bh = i_kbh % (tl.num_programs(2) // NK)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T
    NSUB = BT // BC

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    t_sub = i_t * BT + i_sub * BC
    t_rows = t_sub + o_i
    off_rk = (bos + t_rows[:, None]) * (H * K) + i_h * K + o_k[None, :]

    b_q = tl.load(q_ptr + off_rk).to(tl.float32)
    b_k = tl.load(k_ptr + off_rk).to(tl.float32)
    b_kt = tl.load(k_tilde_ptr + off_rk).to(tl.float32)
    b_g = tl.load(g_cum_ptr + off_rk).to(tl.float32)
    b_beta = tl.load(beta_ptr + (bos + t_rows) * H + i_h).to(tl.float32)

    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    # 1. Off-diagonal left columns (j_sub < i_sub): column side uses k_tilde
    if i_sub > 0:
        b_ga = tl.load(g_cum_ptr + (bos + t_sub - 1) * (H * K) + i_h * K + o_k).to(tl.float32)
        for j_sub in range(0, i_sub):
            t_cols = i_t * BT + j_sub * BC + o_i
            off_ck = (bos + t_cols[:, None]) * (H * K) + i_h * K + o_k[None, :]
            b_kt_c = tl.load(k_tilde_ptr + off_ck).to(tl.float32)
            b_gc = tl.load(g_cum_ptr + off_ck).to(tl.float32)
            b_ktg = b_kt_c * tl.exp(tl.minimum(b_ga[None, :] - b_gc, 0.0))

            off_A = (bos + t_rows[:, None]) * (H * BT) + i_h * BT + (j_sub * BC + o_i[None, :])
            b_dAqk = tl.load(dAqk_ptr + off_A).to(tl.float32)
            b_dAkk = tl.load(dAkk_ptr + off_A).to(tl.float32)
            b_dq += tl.dot(b_dAqk, b_ktg, allow_tf32=False)
            b_dk += tl.dot(b_dAkk, b_ktg, allow_tf32=False)

        b_exp_r = tl.exp(tl.minimum(b_g - b_ga[None, :], 0.0))
        b_dq *= b_exp_r
        b_dk *= b_exp_r

    # 2. Diagonal sub-chunk (c in 0..BC-1): pre-masked non-positive causal differences
    for c in range(BC):
        tc = t_sub + c
        off_ck = (bos + tc) * (H * K) + i_h * K + o_k
        b_kt_c = tl.load(k_tilde_ptr + off_ck).to(tl.float32)
        b_gc = tl.load(g_cum_ptr + off_ck).to(tl.float32)

        m_inc = o_i >= c
        m_exc = o_i > c
        diff_inc = tl.where(m_inc[:, None], b_g - b_gc[None, :], -1.0e30)
        exp_inc = tl.where(m_inc[:, None], tl.exp(diff_inc), 0.0)
        exp_exc = tl.where(m_exc[:, None], exp_inc, 0.0)

        off_Ac = (bos + t_rows) * (H * BT) + i_h * BT + (i_sub * BC + c)
        b_dAqk_c = tl.load(dAqk_ptr + off_Ac).to(tl.float32)
        b_dAkk_c = tl.load(dAkk_ptr + off_Ac).to(tl.float32)

        b_dq += b_dAqk_c[:, None] * (b_kt_c[None, :] * exp_inc)
        b_dk += b_dAkk_c[:, None] * (b_kt_c[None, :] * exp_exc)

    b_db_intra = tl.sum(b_dk * b_k, axis=1)
    b_dk = b_dk * b_beta[:, None]
    b_dq = b_dq * scale

    # 3. Column-side gradient dk_tilde from lower blocks (j_sub > i_sub) and diagonal block
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
    if i_sub < NSUB - 1:
        b_ga_end = tl.load(g_cum_ptr + (bos + t_sub + BC - 1) * (H * K) + i_h * K + o_k).to(tl.float32)
        for j_sub in range(i_sub + 1, NSUB):
            t_r_low = i_t * BT + j_sub * BC + o_i
            off_low_k = (bos + t_r_low[:, None]) * (H * K) + i_h * K + o_k[None, :]
            b_q_low = tl.load(q_ptr + off_low_k).to(tl.float32) * scale
            b_k_low = tl.load(k_ptr + off_low_k).to(tl.float32)
            b_g_low = tl.load(g_cum_ptr + off_low_k).to(tl.float32)
            b_b_low = tl.load(beta_ptr + (bos + t_r_low) * H + i_h).to(tl.float32)

            b_exp_low = tl.exp(tl.minimum(b_g_low - b_ga_end[None, :], 0.0))
            b_qg_low = b_q_low * b_exp_low
            b_kg_low = (b_k_low * b_b_low[:, None]) * b_exp_low

            off_AT = (bos + t_r_low[None, :]) * (H * BT) + i_h * BT + (i_sub * BC + o_i[:, None])
            b_dAqk_T = tl.load(dAqk_ptr + off_AT).to(tl.float32)
            b_dAkk_T = tl.load(dAkk_ptr + off_AT).to(tl.float32)
            b_dkt += tl.dot(b_dAqk_T, b_qg_low, allow_tf32=False)
            b_dkt += tl.dot(b_dAkk_T, b_kg_low, allow_tf32=False)

        b_dkt *= tl.exp(tl.minimum(b_ga_end[None, :] - b_g, 0.0))

    for r in range(BC):
        tr = t_sub + r
        off_rk_single = (bos + tr) * (H * K) + i_h * K + o_k
        b_qr = tl.load(q_ptr + off_rk_single).to(tl.float32) * scale
        b_kr = tl.load(k_ptr + off_rk_single).to(tl.float32)
        b_gr = tl.load(g_cum_ptr + off_rk_single).to(tl.float32)
        b_br = tl.load(beta_ptr + (bos + tr) * H + i_h).to(tl.float32)

        m_inc = o_i <= r
        m_exc = o_i < r
        diff_inc = tl.where(m_inc[:, None], b_gr[None, :] - b_g, -1.0e30)
        exp_inc = tl.where(m_inc[:, None], tl.exp(diff_inc), 0.0)
        exp_exc = tl.where(m_exc[:, None], exp_inc, 0.0)

        off_Ar = (bos + tr) * (H * BT) + i_h * BT + (i_sub * BC + o_i)
        b_dAqk_r = tl.load(dAqk_ptr + off_Ar).to(tl.float32)
        b_dAkk_r = tl.load(dAkk_ptr + off_Ar).to(tl.float32)

        b_dkt += b_dAqk_r[:, None] * (b_qr[None, :] * exp_inc) + b_dAkk_r[:, None] * (b_kr[None, :] * b_br * exp_exc)

    # 4. Combine with inter-chunk & WY gradients and apply Asymmetric Euler Identity for dg
    t_last = i_t * BT + BT - 1
    b_glast = tl.load(g_cum_ptr + (bos + t_last) * (H * K) + i_h * K + o_k).to(tl.float32)
    b_sfx = tl.exp(tl.minimum(b_glast[None, :] - b_g, 0.0))
    b_pfx = tl.exp(tl.minimum(b_g, 0.0))

    b_dq_tot = b_dq + tl.load(dqg_ptr + off_rk).to(tl.float32) * b_pfx
    b_dk_tot = b_dk + tl.load(dk_wy_ptr + off_rk).to(tl.float32)
    b_dkt_tot = b_dkt + tl.load(dkg_ptr + off_rk).to(tl.float32) * b_sfx

    b_dg_sum = b_q * b_dq_tot + b_k * b_dk_tot - b_kt * b_dkt_tot

    tl.store(dq_out_ptr + off_rk, b_dq_tot.to(dq_out_ptr.dtype.element_ty))
    tl.store(dk_stage2_ptr + off_rk, b_dk_tot)
    tl.store(dk_tilde_out_ptr + off_rk, b_dkt_tot)
    tl.store(dg_sum_ptr + off_rk, b_dg_sum)

    B_total = tl.num_programs(2) // (NK * H)
    db_off = ((i_k * B_total + i_b) * T + t_rows) * H + i_h
    tl.store(db_part_ptr + db_off, b_db_intra)


@triton.jit
def _pkda_bwd_rcumsum_dg_kernel(
    dg_sum_ptr, dg_last_ptr, kg_ptr, dkg_ptr, dg_out_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Kernel 18: Reverse Chunk-Local Cumulative Sum for `dg = rcumsum(dg_sum) + dg_last + sum_t(kg * dkg)` (`0` spills).
    Grid: `(T // BT, cdiv(K, BK), B * H)`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    off = (bos + o_t[:, None]) * (H * K) + i_h * K + o_k[None, :]

    b_sum = tl.load(dg_sum_ptr + off).to(tl.float32)
    b_kg = tl.load(kg_ptr + off).to(tl.float32)
    b_dkg = tl.load(dkg_ptr + off).to(tl.float32)
    b_last = tl.load(dg_last_ptr + ((i_b * NT + i_t) * H + i_h) * K + o_k).to(tl.float32) + tl.sum(b_kg * b_dkg, axis=0)
    b_dg = tl.cumsum(b_sum, axis=0, reverse=True) + b_last[None, :]
    tl.store(dg_out_ptr + off, b_dg.to(dg_out_ptr.dtype.element_ty))


# =============================================================================
# 4. Block-Level Superfused PrecondKDA Frontend & Epilogue Kernels (Kernels 19-24)
# =============================================================================

@triton.jit
def _pkda_fused_shortconv_silu_fwd_kernel(
    x_ptr, w_ptr, y_ptr, pre_act_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """
    Kernel 19: Tile-Parallel (`[BT=64, BD=64]`) Fused Causal Depthwise `ShortConv1d(W=4) + SiLU` Forward (`0` spills).
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
def _pkda_fused_shortconv_silu_bwd_kernel(
    x_ptr, w_ptr, pre_act_ptr, dy_ptr, dx_ptr, dw_part_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """
    Kernel 20: Tile-Parallel (`[BT=64, BD=64]`) Fused Causal Depthwise `ShortConv1d(W=4) + SiLU` Backward (`0` spills).
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
    tl.store(dx_ptr + off0, dx.to(dx_ptr.dtype.element_ty), mask=m0)

    x3 = tl.load(x_ptr + off0, mask=m0, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base_b + (o_t[:, None] - 1) * D + o_d[None, :], mask=(o_t[:, None] >= 1) & m0, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base_b + (o_t[:, None] - 2) * D + o_d[None, :], mask=(o_t[:, None] >= 2) & m0, other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + base_b + (o_t[:, None] - 3) * D + o_d[None, :], mask=(o_t[:, None] >= 3) & m0, other=0.0).to(tl.float32)

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
def _pkda_superfused_qk_l2norm_gates_fwd_kernel(
    q_in_ptr, k_in_ptr, f_in_ptr, a_in_ptr, b_atk_in_ptr, b_in_ptr,
    A_log_ptr, dt_bias_ptr, A_log_atk_ptr, dt_bias_atk_ptr,
    q_out_ptr, k_out_ptr, q_rstd_ptr, k_rstd_ptr,
    g_out_ptr, g_atk_out_ptr, beta_atk_out_ptr, beta_out_ptr,
    H: tl.constexpr,
    K: tl.constexpr,
):
    """
    Kernel 21: Superfused `QK-L2Norm + KDA Softplus Gate + ATK Softplus Gate + Beta/BetaATK Sigmoid` Forward (`0` spills).
    Replaces 6 separate upstream kernel launches (`l2norm_fwd(q)`, `l2norm_fwd(k)`, `fused_kda_gate`,
    `softplus(a + dt_bias_atk) * -exp(A_log_atk)`, `sigmoid(b_atk)`, `sigmoid(b)`) in 1 pass.
    Grid: `(B * T, H)`.
    """
    i_bt = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_k = tl.arange(0, K)
    off_k = (i_bt * H + i_h) * K + o_k
    off_h = i_bt * H + i_h

    q_val = tl.load(q_in_ptr + off_k).to(tl.float32)
    k_val = tl.load(k_in_ptr + off_k).to(tl.float32)
    f_val = tl.load(f_in_ptr + off_k).to(tl.float32)

    q_rstd = tl.rsqrt(tl.sum(q_val * q_val, axis=0) + 1e-12)
    k_rstd = tl.rsqrt(tl.sum(k_val * k_val, axis=0) + 1e-12)
    tl.store(q_rstd_ptr + off_h, q_rstd)
    tl.store(k_rstd_ptr + off_h, k_rstd)
    tl.store(q_out_ptr + off_k, (q_val * q_rstd).to(q_out_ptr.dtype.element_ty))
    tl.store(k_out_ptr + off_k, (k_val * k_rstd).to(k_out_ptr.dtype.element_ty))

    A_exp = tl.exp(tl.load(A_log_ptr + i_h).to(tl.float32))
    dt_b = tl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)
    z_f = f_val + dt_b
    sp_f = tl.where(z_f > 20.0, z_f, tl.log(1.0 + tl.exp(z_f)))
    g_val = -A_exp * sp_f
    tl.store(g_out_ptr + off_k, g_val)

    a_raw = tl.load(a_in_ptr + off_h).to(tl.float32)
    A_atk_exp = tl.exp(tl.load(A_log_atk_ptr + i_h).to(tl.float32))
    dt_b_atk = tl.load(dt_bias_atk_ptr + i_h).to(tl.float32)
    z_a = a_raw + dt_b_atk
    sp_a = tl.where(z_a > 20.0, z_a, tl.log(1.0 + tl.exp(z_a)))
    g_atk_val = -A_atk_exp * sp_a
    tl.store(g_atk_out_ptr + off_h, g_atk_val)

    ba_raw = tl.load(b_atk_in_ptr + off_h).to(tl.float32)
    b_raw = tl.load(b_in_ptr + off_h).to(tl.float32)
    tl.store(beta_atk_out_ptr + off_h, tl.sigmoid(ba_raw).to(beta_atk_out_ptr.dtype.element_ty))
    tl.store(beta_out_ptr + off_h, tl.sigmoid(b_raw).to(beta_out_ptr.dtype.element_ty))


@triton.jit
def _pkda_superfused_qk_l2norm_gates_bwd_kernel(
    q_norm_ptr, k_norm_ptr, q_rstd_ptr, k_rstd_ptr,
    f_in_ptr, a_in_ptr, beta_atk_ptr, beta_ptr,
    A_log_ptr, dt_bias_ptr, A_log_atk_ptr, dt_bias_atk_ptr,
    dq_norm_ptr, dk_norm_ptr, dg_ptr, dg_atk_ptr, dbeta_atk_ptr, dbeta_ptr,
    dq_raw_ptr, dk_raw_ptr, df_raw_ptr, da_raw_ptr, db_atk_raw_ptr, db_raw_ptr,
    dA_log_part_ptr, ddt_bias_part_ptr, dA_log_atk_part_ptr, ddt_bias_atk_part_ptr,
    BT_total,
    H: tl.constexpr,
    K: tl.constexpr,
):
    """
    Kernel 22: Superfused `QK-L2Norm + KDA/ATK Softplus Gates + Beta/BetaATK Sigmoid` Backward (`0` spills).
    Grid: `(BT_total, H)`.
    """
    i_bt = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_k = tl.arange(0, K)
    off_k = (i_bt * H + i_h) * K + o_k
    off_h = i_bt * H + i_h

    qn = tl.load(q_norm_ptr + off_k).to(tl.float32)
    kn = tl.load(k_norm_ptr + off_k).to(tl.float32)
    qr = tl.load(q_rstd_ptr + off_h).to(tl.float32)
    kr = tl.load(k_rstd_ptr + off_h).to(tl.float32)
    dqn = tl.load(dq_norm_ptr + off_k).to(tl.float32)
    dkn = tl.load(dk_norm_ptr + off_k).to(tl.float32)

    dq_raw = (dqn - qn * tl.sum(qn * dqn, axis=0)) * qr
    dk_raw = (dkn - kn * tl.sum(kn * dkn, axis=0)) * kr
    tl.store(dq_raw_ptr + off_k, dq_raw.to(dq_raw_ptr.dtype.element_ty))
    tl.store(dk_raw_ptr + off_k, dk_raw.to(dk_raw_ptr.dtype.element_ty))

    f_val = tl.load(f_in_ptr + off_k).to(tl.float32)
    A_exp = tl.exp(tl.load(A_log_ptr + i_h).to(tl.float32))
    dt_b = tl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)
    z_f = f_val + dt_b
    sp_f = tl.where(z_f > 20.0, z_f, tl.log(1.0 + tl.exp(z_f)))
    sig_f = tl.sigmoid(z_f)
    dg_val = tl.load(dg_ptr + off_k).to(tl.float32)
    df_val = -A_exp * sig_f * dg_val
    dA_log_val = tl.sum(-A_exp * sp_f * dg_val, axis=0)
    tl.store(df_raw_ptr + off_k, df_val.to(df_raw_ptr.dtype.element_ty))
    tl.store(ddt_bias_part_ptr + off_k, df_val)
    tl.store(dA_log_part_ptr + off_h, dA_log_val)

    a_raw = tl.load(a_in_ptr + off_h).to(tl.float32)
    A_atk_exp = tl.exp(tl.load(A_log_atk_ptr + i_h).to(tl.float32))
    dt_b_atk = tl.load(dt_bias_atk_ptr + i_h).to(tl.float32)
    z_a = a_raw + dt_b_atk
    sp_a = tl.where(z_a > 20.0, z_a, tl.log(1.0 + tl.exp(z_a)))
    sig_a = tl.sigmoid(z_a)
    dga_val = tl.load(dg_atk_ptr + off_h).to(tl.float32)
    da_val = -A_atk_exp * sig_a * dga_val
    dA_atk_val = -A_atk_exp * sp_a * dga_val
    tl.store(da_raw_ptr + off_h, da_val.to(da_raw_ptr.dtype.element_ty))
    tl.store(ddt_bias_atk_part_ptr + off_h, da_val)
    tl.store(dA_log_atk_part_ptr + off_h, dA_atk_val)

    ba_val = tl.load(beta_atk_ptr + off_h).to(tl.float32)
    b_val = tl.load(beta_ptr + off_h).to(tl.float32)
    dba = tl.load(dbeta_atk_ptr + off_h).to(tl.float32) * ba_val * (1.0 - ba_val)
    db = tl.load(dbeta_ptr + off_h).to(tl.float32) * b_val * (1.0 - b_val)
    tl.store(db_atk_raw_ptr + off_h, dba.to(db_atk_raw_ptr.dtype.element_ty))
    tl.store(db_raw_ptr + off_h, db.to(db_raw_ptr.dtype.element_ty))


@triton.jit
def _pkda_fused_rmsnorm_sigmoid_gate_fwd_kernel(
    o_ptr, g_gate_ptr, w_ptr, y_ptr, rstd_ptr,
    eps: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """
    Kernel 23: Fused Output `RMSNorm + Sigmoid Gate` Forward (`0` spills).
    Grid: `(B * T, H)`.
    """
    i_bt = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_v = tl.arange(0, V)
    off = (i_bt * H + i_h) * V + o_v

    x = tl.load(o_ptr + off).to(tl.float32)
    g = tl.load(g_gate_ptr + off).to(tl.float32)
    w = tl.load(w_ptr + o_v).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / V + eps)
    tl.store(rstd_ptr + i_bt * H + i_h, rstd)
    y = (x * rstd * w) * tl.sigmoid(g)
    tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty))


@triton.jit
def _pkda_fused_rmsnorm_sigmoid_gate_bwd_kernel(
    o_ptr, g_gate_ptr, w_ptr, rstd_ptr, dy_ptr,
    do_ptr, dg_gate_ptr, dw_part_ptr,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """
    Kernel 24: Fused Output `RMSNorm + Sigmoid Gate` Backward (`0` spills).
    Grid: `(B * T, H)`.
    """
    i_bt = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_v = tl.arange(0, V)
    off = (i_bt * H + i_h) * V + o_v

    x = tl.load(o_ptr + off).to(tl.float32)
    g = tl.load(g_gate_ptr + off).to(tl.float32)
    w = tl.load(w_ptr + o_v).to(tl.float32)
    rstd = tl.load(rstd_ptr + i_bt * H + i_h).to(tl.float32)
    dy = tl.load(dy_ptr + off).to(tl.float32)

    sig = tl.sigmoid(g)
    x_hat = x * rstd
    normed = x_hat * w
    dg = dy * normed * sig * (1.0 - sig)
    dnorm = dy * sig
    dw = dnorm * x_hat
    dx_hat = dnorm * w
    do = rstd * (dx_hat - x_hat * (tl.sum(dx_hat * x_hat, axis=0) / V))

    tl.store(do_ptr + off, do.to(do_ptr.dtype.element_ty))
    tl.store(dg_gate_ptr + off, dg.to(dg_gate_ptr.dtype.element_ty))
    tl.store(dw_part_ptr + off, dw)


# =============================================================================
# 5. Upstream Negative-Control Spill & Overflow Reproduction Kernels
# =============================================================================

@triton.jit
def _pkda_upstream_safe_gate_subchunk_fwd_spill_kernel(
    q_ptr, k_ptr, k_tilde_ptr, g_cum_ptr, beta_ptr, Aqk_ptr, Akk_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    GATE_SHIFT: tl.constexpr,
):
    """
    Negative-Control 1: Reproduces upstream `fla/ops/precond_kda/chunk_intra.py:442-455`
    (`chunk_precond_kda_fwd_kernel_intra_sub_chunk`), which shifts the sub-chunk anchor by
    `b_gn = g[i_ti] - 0.5 * (BC - 1) * GATE_SHIFT` and computes `exp2(-b_gm)` before causal masking,
    overflowing FP32 to `+Inf` -> `NaN` via `0 * Inf` in `tl.dot` under steep decay.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_sub = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T
    i_ti = i_t * BT + i_sub * BC

    o_c = i_ti + tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    off = (bos + o_c[:, None]) * (H * K) + i_h * K + o_k[None, :]

    b_q = tl.load(q_ptr + off)
    b_k = tl.load(k_ptr + off)
    b_kp = tl.load(k_tilde_ptr + off)
    b_g = tl.load(g_cum_ptr + off).to(tl.float32) * 1.4426950408889634
    b_beta = tl.load(beta_ptr + (bos + o_c) * H + i_h).to(tl.float32)

    b_gn = tl.load(g_cum_ptr + (bos + i_ti) * (H * K) + i_h * K + o_k).to(tl.float32)[None, :] * 1.4426950408889634
    b_gn = b_gn - 0.5 * (BC - 1) * GATE_SHIFT
    b_gm = b_g - b_gn

    b_gq = tl.exp2(b_gm)
    b_gk = tl.exp2(-b_gm)
    b_kpgt = tl.trans((b_kp.to(tl.float32) * b_gk).to(b_kp.dtype))

    b_Aqk = tl.dot((b_q.to(tl.float32) * b_gq).to(b_q.dtype), b_kpgt) * scale
    b_Akk = tl.dot((b_k.to(tl.float32) * b_gk).to(b_k.dtype), b_kpgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    b_Aqk = tl.where(o_i[:, None] >= o_i[None, :], b_Aqk, 0.0)
    b_Akk = tl.where(o_i[:, None] > o_i[None, :], b_Akk, 0.0)
    tl.store(Aqk_ptr + (bos + o_c[:, None]) * (H * BT) + i_h * BT + (i_sub * BC + o_i[None, :]), b_Aqk)
    tl.store(Akk_ptr + (bos + o_c[:, None]) * (H * BC) + i_h * BC + o_i[None, :], b_Akk)


@triton.jit
def _pkda_upstream_wy_dqkg_bwd_spill_kernel(
    q_ptr, k_ptr, k_tilde_ptr, v_ptr, v_new_ptr, g_ptr, beta_ptr, A_ptr, h_ptr, do_ptr, dh_ptr, dv_ptr,
    dq_ptr, dk_ptr, dkg_ptr, dv2_ptr, dg_ptr, db_ptr, dA_ptr,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Negative-Control 2: Reproduces upstream `fla/ops/precond_kda/chunk_bwd.py:72-270`
    (`chunk_precond_kda_bwd_kernel_wy_dqkg`), which holds `[64, 64]` `b_A` and `b_dA` plus both
    `b_k` and `b_kp` (`k_tilde`), `b_dq`, `b_dk`, `b_dkg`, `b_dg`, `b_v`, `b_vn`, `b_h`, `b_dh`
    in a single monolithic CTA (`168-3,972` register spills on SM80).
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    b_beta = tl.load(beta_ptr + (bos + o_t) * H + i_h).to(tl.float32)
    b_A = tl.load(A_ptr + (bos + o_t[:, None]) * (H * BT) + i_h * BT + o_A[None, :]).to(tl.float32)
    b_A = tl.trans(b_A)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)
    h_base = ((i_b * NT + i_t) * H + i_h) * (K * V)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        off_tk = (bos + o_t[:, None]) * (H * K) + i_h * K + o_k[None, :]
        b_k = tl.load(k_ptr + off_tk).to(tl.float32)
        b_kp = tl.load(k_tilde_ptr + off_tk).to(tl.float32)
        b_g = tl.load(g_ptr + off_tk).to(tl.float32)
        b_gn = tl.load(g_ptr + (bos + i_t * BT + BT - 1) * (H * K) + i_h * K + o_k).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dkg_raw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            off_tv = (bos + o_t[:, None]) * (H * V) + i_h * V + o_v[None, :]
            b_v_new = tl.load(v_new_ptr + off_tv).to(tl.float32)
            b_do = tl.load(do_ptr + off_tv).to(tl.float32)
            b_dv = tl.load(dv_ptr + off_tv).to(tl.float32)
            b_h = tl.load(h_ptr + h_base + o_k[:, None] * V + o_v[None, :]).to(tl.float32)
            b_dh = tl.load(dh_ptr + h_base + o_k[:, None] * V + o_v[None, :]).to(tl.float32)

            b_dgk += tl.sum(b_h * b_dh, axis=1)
            b_dq += tl.dot(b_do, tl.trans(b_h))
            b_dkg_raw += tl.dot(b_v_new, tl.trans(b_dh))
            b_dw += tl.dot(b_dv, tl.trans(b_h))

            if i_k == 0:
                b_v = tl.load(v_ptr + off_tv).to(tl.float32)
                b_dA += tl.dot(b_dv, tl.trans(b_v))
                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, axis=1)
                tl.store(dv2_ptr + off_tv, b_dv2.to(dv2_ptr.dtype.element_ty))

        b_gk_exp = tl.exp(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= tl.exp(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_kg_orig = b_k * b_gk_exp
        b_dw = -b_dw
        b_dA += tl.dot(b_dw, tl.trans(b_kg_orig))
        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg_orig, axis=1)
        b_dk = b_dkgb * b_gb
        b_gn_g = tl.exp(b_gn[None, :] - b_g)
        b_dkg = b_dkg_raw * b_gn_g
        b_kg_precond = b_kp * b_gn_g
        b_kp_dkg = b_kg_precond * b_dkg_raw
        b_dgk += tl.sum(b_kp_dkg, axis=0)
        b_q = tl.load(q_ptr + off_tk).to(tl.float32)
        b_dg = b_q * b_dq - b_kp_dkg + b_kg_orig * b_dkgb * b_beta[:, None]
        tl.store(dq_ptr + off_tk, b_dq)
        tl.store(dk_ptr + off_tk, b_dk)
        tl.store(dkg_ptr + off_tk, b_dkg)
        tl.store(dg_ptr + off_tk, b_dg)

    m_A = o_t[:, None] > o_t[None, :]
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0.0)
    b_dA = tl.dot(b_dA, b_A)
    b_dA = tl.dot(b_A, b_dA)
    b_dA = tl.where(m_A, -b_dA, 0.0)
    tl.store(dA_ptr + (bos + o_t[:, None]) * (H * BT) + i_h * BT + o_A[None, :], b_dA)
    tl.store(db_ptr + (bos + o_t) * H + i_h, b_db)


# =============================================================================
# 6. Python Autograd Orchestration for Core Preconditioned KDA & Full Block
# =============================================================================

def chunk_precond_kda_rsi_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    log_atk_scale: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_A_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    sym_akk_bug: bool = False,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    BC = 16
    BM_sub = 16
    BK_sub = 32
    BV_sub = 32
    NT = T // BT
    NS = T // BC
    NK = K // BK_sub
    is_fp32 = q.dtype == torch.float32

    # Stage 1: 16-token sub-chunked ATK forward (2 zero-spill launches)
    x_sub_init = torch.empty(B, NS, H, K, device=q.device, dtype=torch.float32)
    sa_sub = torch.empty(B, NS, H, device=q.device, dtype=torch.float32)
    x_final = torch.empty(B, H, K, device=q.device, dtype=torch.float32)
    p_shard = torch.empty(B, H, device=q.device, dtype=torch.float32)
    a_shard = torch.empty(B, H, K, device=q.device, dtype=torch.float32)
    x0_arg = initial_A_state.contiguous() if initial_A_state is not None else x_final

    _atk_fused_subchunk_state_fwd_kernel[(NK, B * H)](
        k, beta_atk, g_atk, x0_arg,
        x_sub_init, sa_sub, x_final, p_shard, a_shard,
        T=T, H=H, K=K, BC=BC, BK=BK_sub,
        HAS_X0=(initial_A_state is not None), num_warps=4, num_stages=2,
    )

    k_tilde = torch.empty_like(k)
    logx = float(math.log(float(x)))
    _atk_subchunk_squash_out_fwd_kernel[(NS, NK, B * H)](
        k, beta_atk, g_atk, x_sub_init, log_atk_scale,
        k_tilde,
        logx=logx, eps=float(eps),
        T=T, H=H, K=K, BC=BC, BK=BK_sub, num_warps=4, num_stages=2,
    )

    # Stage 2: Zero-spill Asymmetric Preconditioned KDA forward
    g_cum = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    _pkda_fused_cumsum_g_fwd_kernel[(NT, NK, B * H)](
        g, g_cum, T=T, H=H, K=K, BT=BT, BK=BK_sub, num_warps=4, num_stages=1,
    )

    Aqk = torch.zeros(B, T, H, BT, device=q.device, dtype=torch.float32)
    Akk_diag = torch.empty(B, T, H, BC, device=q.device, dtype=torch.float32)
    _pkda_overflow_free_intra_diag_subchunk_fwd_kernel[(NT, BT // BC, B * H)](
        q, k, k_tilde, g_cum, beta, Aqk, Akk_diag, scale,
        T=T, H=H, K=K, BT=BT, BC=BC, BK=BK_sub, SYM_AKK_BUG=sym_akk_bug, num_warps=4, num_stages=1,
    )

    Akk_inv = torch.zeros(B, T, H, BT, device=q.device, dtype=torch.float32)
    _pkda_zero_spill_inter_subchunk_Aqk_fwd_kernel[(NT, B * H)](
        q, k, k_tilde, g_cum, beta, Aqk, Akk_inv, scale,
        T=T, H=H, K=K, BT=BT, BC=BC, BK=BK_sub, SYM_AKK_BUG=sym_akk_bug, num_warps=4, num_stages=1,
    )

    _pkda_zero_spill_inter_subchunk_Akk_solve_fwd_kernel[(NT, B * H)](
        Akk_diag, Akk_inv,
        T=T, H=H, BT=BT, BC=BC, num_warps=4, num_stages=1,
    )

    w_wy = torch.empty_like(k)
    u_wy = torch.empty_like(v)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k_tilde)
    _pkda_wy_wu_qg_kg_fwd_kernel[(NT * (BT // BM_sub), B * H)](
        q, k, k_tilde, v, g_cum, beta, Akk_inv,
        w_wy, u_wy, qg, kg,
        T=T, H=H, K=K, V=V, BT=BT, BM=BM_sub, BK=BK_sub, BV=BV_sub, num_warps=4, num_stages=1,
    )

    h_all = torch.empty(B, NT, H, K, V, device=q.device, dtype=q.dtype)
    v_new = torch.empty_like(v)
    s_final = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)
    h0_arg = initial_state.contiguous() if initial_state is not None else s_final

    _pkda_chunk_state_h_fwd_kernel[(triton.cdiv(V, BV_sub), B * H)](
        kg, w_wy, u_wy, g_cum, h0_arg,
        h_all, v_new, s_final,
        T=T, H=H, K=K, V=V, BT=BT, BK=BK_sub, BV=BV_sub,
        HAS_H0=(initial_state is not None), IS_FP32=is_fp32, num_warps=4, num_stages=1,
    )

    o = torch.empty_like(v)
    _pkda_chunk_readout_o_fwd_kernel[(triton.cdiv(V, BV_sub), NT * (BT // BM_sub), B * H)](
        qg, v_new, Aqk, h_all, o, scale,
        T=T, H=H, K=K, V=V, BT=BT, BM=BM_sub, BK=BK_sub, BV=BV_sub, num_warps=4, num_stages=1,
    )

    ht_out = s_final if output_final_state else None
    at_out = x_final if output_final_state else None
    cache = (x_sub_init, sa_sub, k_tilde, g_cum, Aqk, Akk_inv, w_wy, u_wy, qg, kg, h_all, v_new)
    return o, ht_out, at_out, cache


def chunk_precond_kda_rsi_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    log_atk_scale: torch.Tensor,
    cache: tuple,
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    dat: Optional[torch.Tensor],
    scale: float,
    x: float = 1.5,
    eps: float = 1e-6,
    omit_outer_strict_mask: bool = False,
    wrong_neg_center_sign: bool = False,
    omit_atk_dk: bool = False,
):
    x_sub_init, sa_sub, k_tilde, g_cum, Aqk, Akk_inv, w_wy, u_wy, qg, kg, h_all, v_new = cache
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    BC = 16
    BM_sub = 16
    BK_sub = 32
    BV_sub = 32
    NT = T // BT
    NS = T // BC
    NK = K // BK_sub
    is_fp32 = q.dtype == torch.float32

    dv_new_intra = torch.empty_like(v_new)
    dAqk = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
    _pkda_bwd_dAv_kernel[(NT * (BT // BM_sub), B * H)](
        v_new, Aqk, do, dv_new_intra, dAqk,
        T=T, H=H, V=V, BT=BT, BM=BM_sub, BV=BV_sub, num_warps=4, num_stages=1,
    )

    dh_all = torch.empty(B, NT, H, K, V, device=q.device, dtype=q.dtype)
    du_total = torch.empty_like(v_new)
    dh0 = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)
    dht_arg = dht.contiguous() if dht is not None else dh0

    _pkda_bwd_dhu_reverse_kernel[(triton.cdiv(V, BV_sub), B * H)](
        qg, kg, w_wy, g_cum, do, dv_new_intra, dht_arg,
        dh_all, du_total, dh0, scale,
        T=T, H=H, K=K, V=V, BT=BT, BK=BK_sub, BV=BV_sub,
        HAS_DHT=(dht is not None), IS_FP32=is_fp32, num_warps=4, num_stages=1,
    )

    dqg = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dkg = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dw = torch.empty_like(k)
    dg_last = torch.empty(B, NT, H, K, device=q.device, dtype=torch.float32)

    _pkda_bwd_wy_dqk_slice_kernel[(NT * (BT // BM_sub), NK, B * H)](
        v_new, g_cum, h_all, do, dh_all, du_total,
        dqg, dkg, dw, dg_last, scale,
        T=T, H=H, K=K, V=V, BT=BT, BM=BM_sub, BK=BK_sub, BV=BV_sub, num_warps=4, num_stages=1,
    )

    dv_out = torch.empty_like(v)
    dk_wy = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    db_wy = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    dM_raw = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)

    _pkda_bwd_wy_dAkk_dv_db_kernel[(NT * (BT // BM_sub), B * H)](
        Akk_inv, k, v, g_cum, beta, dw, du_total,
        dv_out, dk_wy, db_wy, dM_raw,
        T=T, H=H, K=K, V=V, BT=BT, BM=BM_sub, BK=BK_sub, BV=BV_sub, num_warps=4, num_stages=1,
    )

    dAkk = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
    _pkda_bwd_wy_dAkk_solve_kernel[(NT * (BT // BM_sub), BT // BM_sub, B * H)](
        Akk_inv, dM_raw, dAkk,
        T=T, H=H, BT=BT, BM=BM_sub, OMIT_OUTER_STRICT_MASK=omit_outer_strict_mask, num_warps=4, num_stages=1,
    )

    dq_out = torch.empty_like(q)
    dk_stage2 = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dk_tilde = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    db_part = torch.empty(NK, B, T, H, device=q.device, dtype=torch.float32)
    dg_sum = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)

    _pkda_bwd_intra_subchunk_asymmetric_kernel[(NT, BT // BC, NK * B * H)](
        q, k, k_tilde, g_cum, beta,
        dAqk, dAkk, dqg, dkg, dk_wy,
        dq_out, dk_stage2, dk_tilde, db_part, dg_sum, scale,
        T=T, H=H, K=K, BT=BT, BC=BC, BK=BK_sub, num_warps=4, num_stages=1,
    )

    dbeta_out = (db_wy + db_part.sum(dim=0)).to(beta.dtype)
    dg_out = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    _pkda_bwd_rcumsum_dg_kernel[(NT, NK, B * H)](
        dg_sum, dg_last, kg, dkg, dg_out,
        T=T, H=H, K=K, BT=BT, BK=BK_sub, num_warps=4, num_stages=1,
    )

    # Stage 1 backward: 2 zero-spill ATK sub-chunk backward launches
    dk_atk = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dg_atk_part = torch.empty(NK, B, T, H, device=q.device, dtype=torch.float32)
    dbeta_atk_part = torch.empty(NK, B, T, H, device=q.device, dtype=torch.float32)
    dx_sub_in = torch.empty(B, NS, H, K, device=q.device, dtype=torch.float32)
    dcenter_part = torch.empty(NK, B, NS, H, device=q.device, dtype=torch.float32)
    logx = float(math.log(float(x)))

    _atk_subchunk_squash_bwd_kernel[(NS, NK, B * H)](
        k, beta_atk, g_atk, x_sub_init, log_atk_scale, dk_tilde,
        dk_atk, dg_atk_part, dbeta_atk_part, dx_sub_in, dcenter_part,
        logx=logx, eps=float(eps),
        T=T, H=H, K=K, BC=BC, BK=BK_sub, WRONG_NEG_CENTER_SIGN=wrong_neg_center_sign, num_warps=4, num_stages=2,
    )

    dx0_atk = torch.empty(B, H, K, device=q.device, dtype=torch.float32)
    dat_arg = dat.contiguous() if dat is not None else dx0_atk
    _atk_reverse_subchunk_scan_bwd_kernel[(NK, B * H)](
        k, beta_atk, g_atk, x_sub_init, sa_sub,
        dx_sub_in, dat_arg,
        dk_atk, dg_atk_part, dbeta_atk_part, dx0_atk,
        T=T, H=H, K=K, BC=BC, BK=BK_sub, HAS_DXT=(dat is not None), num_warps=4, num_stages=2,
    )

    if omit_atk_dk:
        dk_total = dk_stage2.to(k.dtype)
    else:
        dk_total = (dk_stage2 + dk_atk).to(k.dtype)

    dg_atk_out = dg_atk_part.sum(dim=0)
    dbeta_atk_out = dbeta_atk_part.sum(dim=0).to(beta_atk.dtype)
    d_log_atk_scale = dcenter_part.sum(dim=(0, 1, 2))
    return dq_out, dk_total, dv_out, dg_out, dg_atk_out, dbeta_atk_out, dbeta_out, d_log_atk_scale, dh0, dx0_atk


class _ChunkPrecondKDARSIAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        g_atk: torch.Tensor,
        beta_atk: torch.Tensor,
        beta: torch.Tensor,
        log_atk_scale: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        initial_A_state: Optional[torch.Tensor],
        output_final_state: bool,
        x: float,
        eps: float,
        use_qk_l2norm: bool,
        sym_akk_bug: bool,
        omit_outer_strict_mask: bool,
        wrong_neg_center_sign: bool,
        omit_atk_dk: bool,
    ):
        if use_qk_l2norm:
            q_rstd = torch.rsqrt((q.float() * q.float()).sum(dim=-1) + 1e-12)
            k_rstd = torch.rsqrt((k.float() * k.float()).sum(dim=-1) + 1e-12)
            qn = (q.float() * q_rstd.unsqueeze(-1)).to(q.dtype).contiguous()
            kn = (k.float() * k_rstd.unsqueeze(-1)).to(k.dtype).contiguous()
        else:
            q_rstd = torch.empty(0, device=q.device)
            k_rstd = torch.empty(0, device=q.device)
            qn = q.contiguous()
            kn = k.contiguous()

        o, ht, at, cache = chunk_precond_kda_rsi_fwd(
            qn, kn, v.contiguous(), g.contiguous(), g_atk.contiguous(),
            beta_atk.contiguous(), beta.contiguous(), log_atk_scale.contiguous(),
            scale, initial_state=initial_state, initial_A_state=initial_A_state,
            output_final_state=output_final_state, x=x, eps=eps, sym_akk_bug=sym_akk_bug,
        )
        ctx.save_for_backward(qn, kn, q_rstd, k_rstd, v, g_atk, beta_atk, beta, log_atk_scale, *cache)
        ctx.scale = scale
        ctx.x = x
        ctx.eps = eps
        ctx.use_qk_l2norm = use_qk_l2norm
        ctx.has_h0 = initial_state is not None
        ctx.has_a0 = initial_A_state is not None
        ctx.omit_outer_strict_mask = omit_outer_strict_mask
        ctx.wrong_neg_center_sign = wrong_neg_center_sign
        ctx.omit_atk_dk = omit_atk_dk
        return o, ht, at

    @staticmethod
    def backward(ctx, do, dht, dat):
        saved = ctx.saved_tensors
        qn, kn, q_rstd, k_rstd, v, g_atk, beta_atk, beta, log_atk_scale = saved[:9]
        cache = saved[9:]
        dqn, dkn, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale, dh0, dx0 = chunk_precond_kda_rsi_bwd(
            qn, kn, v, g_atk, beta_atk, beta, log_atk_scale, cache,
            do.contiguous(), dht, dat, ctx.scale, x=ctx.x, eps=ctx.eps,
            omit_outer_strict_mask=ctx.omit_outer_strict_mask,
            wrong_neg_center_sign=ctx.wrong_neg_center_sign,
            omit_atk_dk=ctx.omit_atk_dk,
        )
        if ctx.use_qk_l2norm:
            qnf, knf = qn.float(), kn.float()
            dqf, dkf = dqn.float(), dkn.float()
            dq = ((dqf - qnf * (qnf * dqf).sum(dim=-1, keepdim=True)) * q_rstd.unsqueeze(-1)).to(qn.dtype)
            dk = ((dkf - knf * (knf * dkf).sum(dim=-1, keepdim=True)) * k_rstd.unsqueeze(-1)).to(kn.dtype)
        else:
            dq, dk = dqn, dkn

        return (
            dq, dk, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale,
            None,
            dh0 if ctx.has_h0 else None,
            dx0 if ctx.has_a0 else None,
            None, None, None, None, None, None, None, None,
        )


def chunk_precond_kda_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    log_atk_scale: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_A_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    use_qk_l2norm: bool = True,
    sym_akk_bug: bool = False,
    omit_outer_strict_mask: bool = False,
    wrong_neg_center_sign: bool = False,
    omit_atk_dk: bool = False,
):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if log_atk_scale is None:
        log_atk_scale = torch.full((q.shape[2],), -0.2, device=q.device, dtype=torch.float32)
    o, ht, at = _ChunkPrecondKDARSIAutogradFn.apply(
        q, k, v, g, g_atk, beta_atk, beta, log_atk_scale, float(scale),
        initial_state, initial_A_state, output_final_state, float(x), float(eps),
        use_qk_l2norm, sym_akk_bug, omit_outer_strict_mask, wrong_neg_center_sign, omit_atk_dk,
    )
    if output_final_state:
        return o, ht, at
    return o, None, None


# =============================================================================
# 7. Superfused Block-Level Wrappers & PrecondKDABlockRSI Module
# =============================================================================

class _PKDAShortConvSiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        BT = 32
        BD = 32
        y = torch.empty_like(x)
        pre_act = torch.empty(B, T, D, device=x.device, dtype=torch.float32)
        _pkda_fused_shortconv_silu_fwd_kernel[(triton.cdiv(T, BT), triton.cdiv(D, BD), B)](
            x.contiguous(), weight.contiguous(), y, pre_act,
            T=T, D=D, BT=BT, BD=BD, num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(x, weight, pre_act)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, pre_act = ctx.saved_tensors
        B, T, D = x.shape
        BT = 32
        BD = 32
        NT = triton.cdiv(T, BT)
        dx = torch.empty_like(x)
        dw_part = torch.empty(B * NT, D, 4, device=x.device, dtype=torch.float32)
        _pkda_fused_shortconv_silu_bwd_kernel[(NT, triton.cdiv(D, BD), B)](
            x.contiguous(), weight.contiguous(), pre_act, dy.contiguous(), dx, dw_part,
            T=T, D=D, BT=BT, BD=BD, num_warps=4, num_stages=1,
        )
        dw = dw_part.sum(dim=0).view_as(weight).to(weight.dtype)
        return dx, dw


def pkda_fused_shortconv_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _PKDAShortConvSiLUFn.apply(x, weight)


class _PKDASuperfusedQKL2NormGatesFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        f_in: torch.Tensor,
        a_in: torch.Tensor,
        b_atk_in: torch.Tensor,
        b_in: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        A_log_atk: torch.Tensor,
        dt_bias_atk: torch.Tensor,
    ):
        B, T, H, K = q_in.shape
        BT_total = B * T
        q_out = torch.empty_like(q_in)
        k_out = torch.empty_like(k_in)
        q_rstd = torch.empty(B, T, H, device=q_in.device, dtype=torch.float32)
        k_rstd = torch.empty(B, T, H, device=q_in.device, dtype=torch.float32)
        g_out = torch.empty(B, T, H, K, device=q_in.device, dtype=torch.float32)
        g_atk_out = torch.empty(B, T, H, device=q_in.device, dtype=torch.float32)
        beta_atk_out = torch.empty(B, T, H, device=q_in.device, dtype=q_in.dtype)
        beta_out = torch.empty(B, T, H, device=q_in.device, dtype=q_in.dtype)

        _pkda_superfused_qk_l2norm_gates_fwd_kernel[(BT_total, H)](
            q_in.contiguous(), k_in.contiguous(), f_in.contiguous(),
            a_in.contiguous(), b_atk_in.contiguous(), b_in.contiguous(),
            A_log.contiguous(), dt_bias.contiguous(), A_log_atk.contiguous(), dt_bias_atk.contiguous(),
            q_out, k_out, q_rstd, k_rstd, g_out, g_atk_out, beta_atk_out, beta_out,
            H=H, K=K, num_warps=2, num_stages=2,
        )
        ctx.save_for_backward(
            q_out, k_out, q_rstd, k_rstd, f_in, a_in, beta_atk_out, beta_out,
            A_log, dt_bias, A_log_atk, dt_bias_atk,
        )
        return q_out, k_out, g_out, g_atk_out, beta_atk_out, beta_out

    @staticmethod
    def backward(ctx, dq_out, dk_out, dg_out, dg_atk_out, dbeta_atk_out, dbeta_out):
        (
            q_out, k_out, q_rstd, k_rstd, f_in, a_in, beta_atk_out, beta_out,
            A_log, dt_bias, A_log_atk, dt_bias_atk,
        ) = ctx.saved_tensors
        B, T, H, K = q_out.shape
        BT_total = B * T
        dq_raw = torch.empty_like(q_out)
        dk_raw = torch.empty_like(k_out)
        df_raw = torch.empty_like(f_in)
        da_raw = torch.empty_like(a_in)
        db_atk_raw = torch.empty_like(beta_atk_out)
        db_raw = torch.empty_like(beta_out)
        dA_log_part = torch.empty(BT_total, H, device=q_out.device, dtype=torch.float32)
        ddt_bias_part = torch.empty(BT_total, H * K, device=q_out.device, dtype=torch.float32)
        dA_log_atk_part = torch.empty(BT_total, H, device=q_out.device, dtype=torch.float32)
        ddt_bias_atk_part = torch.empty(BT_total, H, device=q_out.device, dtype=torch.float32)

        _pkda_superfused_qk_l2norm_gates_bwd_kernel[(BT_total, H)](
            q_out, k_out, q_rstd, k_rstd, f_in.contiguous(), a_in.contiguous(), beta_atk_out, beta_out,
            A_log, dt_bias, A_log_atk, dt_bias_atk,
            dq_out.contiguous(), dk_out.contiguous(), dg_out.contiguous(),
            dg_atk_out.contiguous(), dbeta_atk_out.contiguous(), dbeta_out.contiguous(),
            dq_raw, dk_raw, df_raw, da_raw, db_atk_raw, db_raw,
            dA_log_part, ddt_bias_part, dA_log_atk_part, ddt_bias_atk_part,
            BT_total=BT_total, H=H, K=K, num_warps=2, num_stages=2,
        )
        return (
            dq_raw, dk_raw, df_raw, da_raw, db_atk_raw, db_raw,
            dA_log_part.sum(dim=0).to(A_log.dtype),
            ddt_bias_part.sum(dim=0).to(dt_bias.dtype),
            dA_log_atk_part.sum(dim=0).to(A_log_atk.dtype),
            ddt_bias_atk_part.sum(dim=0).to(dt_bias_atk.dtype),
        )


def pkda_superfused_qk_l2norm_gates(
    q_in, k_in, f_in, a_in, b_atk_in, b_in, A_log, dt_bias, A_log_atk, dt_bias_atk
):
    return _PKDASuperfusedQKL2NormGatesFn.apply(
        q_in, k_in, f_in, a_in, b_atk_in, b_in, A_log, dt_bias, A_log_atk, dt_bias_atk
    )


class _PKDARMSNormSigmoidGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o: torch.Tensor, g_gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        B, T, H, V = o.shape
        y = torch.empty_like(o)
        rstd = torch.empty(B, T, H, device=o.device, dtype=torch.float32)
        _pkda_fused_rmsnorm_sigmoid_gate_fwd_kernel[(B * T, H)](
            o.contiguous(), g_gate.contiguous(), weight.contiguous(), y, rstd,
            eps=float(eps), H=H, V=V, num_warps=2, num_stages=2,
        )
        ctx.save_for_backward(o, g_gate, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        o, g_gate, weight, rstd = ctx.saved_tensors
        B, T, H, V = o.shape
        do = torch.empty_like(o)
        dg = torch.empty_like(g_gate)
        dw_part = torch.empty(B * T, H, V, device=o.device, dtype=torch.float32)
        _pkda_fused_rmsnorm_sigmoid_gate_bwd_kernel[(B * T, H)](
            o.contiguous(), g_gate.contiguous(), weight.contiguous(), rstd, dy.contiguous(),
            do, dg, dw_part, H=H, V=V, num_warps=2, num_stages=2,
        )
        dw = dw_part.sum(dim=(0, 1)).to(weight.dtype)
        return do, dg, dw, None


def pkda_fused_rmsnorm_sigmoid_gate(o: torch.Tensor, g_gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return _PKDARMSNormSigmoidGateFn.apply(o, g_gate, weight, eps)


class PrecondKDABlockRSI(nn.Module):
    """
    Full Preconditioned KDA (`PrecondDeltaNet`) Block matching `fla/layers/precond_kda.py` (`arXiv:2604.21100`):
      - `q_proj, k_proj, v_proj` + causal depthwise `ShortConv1d(W=4) + SiLU`
      - Low-rank KDA gate MLP `f_proj`, ATK gate `a_proj`, ATK step `b_atk_proj`, KDA step `b_proj`
      - Learnable `A_log, dt_bias, A_log_atk, dt_bias_atk, log_atk_scale`
      - Stage 1 `ATK` curvature preconditioner + Stage 2 `Asymmetric Preconditioned KDA`
      - Output `FusedRMSNormGated` + `o_proj`
    """
    def __init__(self, hidden_size: int = 512, num_heads: int = 4, head_dim: int = 128, squash_x: float = 1.5, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.squash_x = squash_x
        self.eps = eps

        D = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, D, bias=False)
        self.k_proj = nn.Linear(hidden_size, D, bias=False)
        self.v_proj = nn.Linear(hidden_size, D, bias=False)

        self.q_conv_w = nn.Parameter(torch.randn(D, 4) * 0.02)
        self.k_conv_w = nn.Parameter(torch.randn(D, 4) * 0.02)
        self.v_conv_w = nn.Parameter(torch.randn(D, 4) * 0.02)

        self.f_proj1 = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_proj2 = nn.Linear(head_dim, D, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_atk_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(D, dtype=torch.float32))
        self.A_log_atk = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.dt_bias_atk = nn.Parameter(torch.full((num_heads,), -1.5, dtype=torch.float32))
        self.log_atk_scale = nn.Parameter(torch.full((num_heads,), -0.2, dtype=torch.float32))

        self.g_proj1 = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_proj2 = nn.Linear(head_dim, D, bias=True)
        self.o_norm_w = nn.Parameter(torch.ones(head_dim))
        self.o_proj = nn.Linear(D, hidden_size, bias=False)

    def _eager_shortconv_silu(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        y_t = F.conv1d(F.pad(x_t, (3, 0)), w.unsqueeze(1), groups=x.shape[-1])
        return F.silu(y_t.transpose(1, 2))

    def forward(
        self,
        x: torch.Tensor,
        use_rsi_kernels: bool = True,
        fusion_stage: int = 4,
        use_compiled_step: bool = False,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, K = self.num_heads, self.head_dim

        q_lin = self.q_proj(x)
        k_lin = self.k_proj(x)
        v_lin = self.v_proj(x)

        if use_rsi_kernels and fusion_stage >= 3:
            q_conv = pkda_fused_shortconv_silu(q_lin, self.q_conv_w).view(B, T, H, K)
            k_conv = pkda_fused_shortconv_silu(k_lin, self.k_conv_w).view(B, T, H, K)
            v_conv = pkda_fused_shortconv_silu(v_lin, self.v_conv_w).view(B, T, H, K)
        else:
            q_conv = self._eager_shortconv_silu(q_lin, self.q_conv_w).view(B, T, H, K)
            k_conv = self._eager_shortconv_silu(k_lin, self.k_conv_w).view(B, T, H, K)
            v_conv = self._eager_shortconv_silu(v_lin, self.v_conv_w).view(B, T, H, K)

        f_in = self.f_proj2(self.f_proj1(x)).view(B, T, H, K)
        a_in = self.a_proj(x)
        b_atk_in = self.b_atk_proj(x)
        b_in = self.b_proj(x)

        if use_rsi_kernels and fusion_stage >= 2:
            qn, kn, g, g_atk, beta_atk, beta = pkda_superfused_qk_l2norm_gates(
                q_conv, k_conv, f_in, a_in, b_atk_in, b_in,
                self.A_log, self.dt_bias, self.A_log_atk, self.dt_bias_atk,
            )
            o_core, _, _ = chunk_precond_kda_rsi(
                qn, kn, v_conv, g, g_atk, beta_atk, beta,
                log_atk_scale=self.log_atk_scale, x=self.squash_x, use_qk_l2norm=False,
            )
        elif use_rsi_kernels and fusion_stage == 1:
            qn = F.normalize(q_conv.float(), dim=-1, eps=1e-12).to(x.dtype)
            kn = F.normalize(k_conv.float(), dim=-1, eps=1e-12).to(x.dtype)
            g = -self.A_log.float().exp().view(1, 1, H, 1) * F.softplus(f_in.float() + self.dt_bias.float().view(1, 1, H, K))
            g_atk = -self.A_log_atk.float().exp().view(1, 1, H) * F.softplus(a_in.float() + self.dt_bias_atk.float().view(1, 1, H))
            beta_atk = torch.sigmoid(b_atk_in)
            beta = torch.sigmoid(b_in)
            o_core, _, _ = chunk_precond_kda_rsi(
                qn, kn, v_conv, g, g_atk, beta_atk, beta,
                log_atk_scale=self.log_atk_scale, x=self.squash_x, use_qk_l2norm=False,
            )
        else:
            g = -self.A_log.float().exp().view(1, 1, H, 1) * F.softplus(f_in.float() + self.dt_bias.float().view(1, 1, H, K))
            g_atk = -self.A_log_atk.float().exp().view(1, 1, H) * F.softplus(a_in.float() + self.dt_bias_atk.float().view(1, 1, H))
            beta_atk = torch.sigmoid(b_atk_in)
            beta = torch.sigmoid(b_in)
            o_core, _, _ = precond_kda_eager_chunk_reference(
                q_conv, k_conv, v_conv, g, g_atk, beta_atk, beta,
                log_atk_scale=self.log_atk_scale, x=self.squash_x,
                use_qk_l2norm=True, use_compiled_step=use_compiled_step,
            )

        g_gate = self.g_proj2(self.g_proj1(x)).view(B, T, H, K)
        if use_rsi_kernels and fusion_stage >= 4:
            y_norm = pkda_fused_rmsnorm_sigmoid_gate(o_core, g_gate, self.o_norm_w, eps=self.eps)
        else:
            of = o_core.float()
            rstd = torch.rsqrt((of * of).mean(dim=-1, keepdim=True) + self.eps)
            y_norm = ((of * rstd * self.o_norm_w.float()) * torch.sigmoid(g_gate.float())).to(x.dtype)

        return x + self.o_proj(y_norm.reshape(B, T, D))


# =============================================================================
# 8. 8-GPU NVSwitch Two-Stage Coupled Curvature-Matrix Context Parallelism (PKDA-CP = 8)
# =============================================================================

def _build_pkda_stage2_rank_affine_summary(
    qn: torch.Tensor,
    kn: torch.Tensor,
    k_tilde: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the exact Stage-2 Asymmetric KDA rank transition `(P_rank, H_rank)`:
      `S_{c+1} = (Diag(exp(g_last,c)) - K_tilde_{g,c}^T @ W_c) S_c + K_tilde_{g,c}^T @ U_c`
    using zero-spill Round-16 RSI Triton Kernels 5-9.
    """
    B, T_loc, H, K = qn.shape
    V = v.shape[-1]
    BT = 64
    BC = 16
    BM_sub = 16
    BK_sub = 32
    BV_sub = 32
    NT = T_loc // BT
    NK = K // BK_sub

    g_cum = torch.empty(B, T_loc, H, K, device=qn.device, dtype=torch.float32)
    _pkda_fused_cumsum_g_fwd_kernel[(NT, NK, B * H)](
        g, g_cum, T=T_loc, H=H, K=K, BT=BT, BK=BK_sub, num_warps=4, num_stages=1,
    )

    Aqk = torch.zeros(B, T_loc, H, BT, device=qn.device, dtype=torch.float32)
    Akk_diag = torch.empty(B, T_loc, H, BC, device=qn.device, dtype=torch.float32)
    _pkda_overflow_free_intra_diag_subchunk_fwd_kernel[(NT, BT // BC, B * H)](
        qn, kn, k_tilde, g_cum, beta, Aqk, Akk_diag, scale,
        T=T_loc, H=H, K=K, BT=BT, BC=BC, BK=BK_sub, SYM_AKK_BUG=False, num_warps=4, num_stages=1,
    )

    Akk_inv = torch.zeros(B, T_loc, H, BT, device=qn.device, dtype=torch.float32)
    _pkda_zero_spill_inter_subchunk_Aqk_fwd_kernel[(NT, B * H)](
        qn, kn, k_tilde, g_cum, beta, Aqk, Akk_inv, scale,
        T=T_loc, H=H, K=K, BT=BT, BC=BC, BK=BK_sub, SYM_AKK_BUG=False, num_warps=4, num_stages=1,
    )

    _pkda_zero_spill_inter_subchunk_Akk_solve_fwd_kernel[(NT, B * H)](
        Akk_diag, Akk_inv,
        T=T_loc, H=H, BT=BT, BC=BC, num_warps=4, num_stages=1,
    )

    w_wy = torch.empty_like(kn)
    u_wy = torch.empty_like(v)
    qg = torch.empty_like(qn)
    kg = torch.empty_like(k_tilde)
    _pkda_wy_wu_qg_kg_fwd_kernel[(NT * (BT // BM_sub), B * H)](
        qn, kn, k_tilde, v, g_cum, beta, Akk_inv,
        w_wy, u_wy, qg, kg,
        T=T_loc, H=H, K=K, V=V, BT=BT, BM=BM_sub, BK=BK_sub, BV=BV_sub, num_warps=4, num_stages=1,
    )

    g_c = g_cum.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    g_last = g_c[:, :, :, -1, :]
    W = w_wy.float().reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    U = u_wy.float().reshape(B, NT, BT, H, V).permute(0, 3, 1, 2, 4)
    KG = kg.float().reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)

    I_k = torch.eye(K, device=qn.device, dtype=torch.float32)[None, None, :, :]
    P_rank = I_k.expand(B, H, K, K).clone()
    H_rank = torch.zeros(B, H, K, V, device=qn.device, dtype=torch.float32)

    for ci in range(NT):
        d_vec = torch.exp(g_last[:, :, ci, :])
        P_c = torch.diag_embed(d_vec) - KG[:, :, ci].transpose(-1, -2) @ W[:, :, ci]
        H_c = KG[:, :, ci].transpose(-1, -2) @ U[:, :, ci]
        P_rank = P_c @ P_rank
        H_rank = P_c @ H_rank + H_c

    return P_rank, H_rank


class _PrecondKDACP8AutogradFn(torch.autograd.Function):
    """
    Two-Stage Hierarchically Coupled Curvature-State + Asymmetric Matrix-Monoid Context Parallelism (`PKDA-CP = 8`).
    Resolves upstream `fla/ops/precond_kda/chunk.py:540-545` (`NotImplementedError`).
    """
    @staticmethod
    def forward(
        ctx,
        q_loc: torch.Tensor,
        k_loc: torch.Tensor,
        v_loc: torch.Tensor,
        g_loc: torch.Tensor,
        g_atk_loc: torch.Tensor,
        beta_atk_loc: torch.Tensor,
        beta_loc: torch.Tensor,
        log_atk_scale: torch.Tensor,
        scale: float,
        x: float,
        eps: float,
        omit_handoff: bool = False,
    ):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        B, T_loc, H, K = q_loc.shape
        V = v_loc.shape[-1]
        BC = 16
        BK_sub = 32
        NS = T_loc // BC
        NK = K // BK_sub

        q_rstd = torch.rsqrt((q_loc.float() * q_loc.float()).sum(dim=-1) + 1e-12)
        k_rstd = torch.rsqrt((k_loc.float() * k_loc.float()).sum(dim=-1) + 1e-12)
        qn = (q_loc.float() * q_rstd.unsqueeze(-1)).to(q_loc.dtype).contiguous()
        kn = (k_loc.float() * k_rstd.unsqueeze(-1)).to(k_loc.dtype).contiguous()

        if omit_handoff or world_size == 1:
            o_loc, _, _, cache = chunk_precond_kda_rsi_fwd(
                qn, kn, v_loc.contiguous(), g_loc.contiguous(), g_atk_loc.contiguous(),
                beta_atk_loc.contiguous(), beta_loc.contiguous(), log_atk_scale.contiguous(),
                scale, initial_state=None, initial_A_state=None, output_final_state=False, x=x, eps=eps,
            )
            ctx.save_for_backward(qn, kn, q_rstd, k_rstd, v_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale, *cache)
            ctx.scale = scale
            ctx.x = x
            ctx.eps = eps
            ctx.omit_handoff = True
            return o_loc

        # Stage 1 (`ATK-CP`): Compute local scalar-vector curvature summary (p_atk_r, a_atk_r) with X_0 = 0
        x_sub_tmp = torch.empty(B, NS, H, K, device=q_loc.device, dtype=torch.float32)
        sa_sub_tmp = torch.empty(B, NS, H, device=q_loc.device, dtype=torch.float32)
        x_final_tmp = torch.empty(B, H, K, device=q_loc.device, dtype=torch.float32)
        p_atk_loc = torch.empty(B, H, device=q_loc.device, dtype=torch.float32)
        a_atk_loc = torch.empty(B, H, K, device=q_loc.device, dtype=torch.float32)

        _atk_fused_subchunk_state_fwd_kernel[(NK, B * H)](
            kn, beta_atk_loc.contiguous(), g_atk_loc.contiguous(), x_final_tmp,
            x_sub_tmp, sa_sub_tmp, x_final_tmp, p_atk_loc, a_atk_loc,
            T=T_loc, H=H, K=K, BC=BC, BK=BK_sub, HAS_X0=False, num_warps=4, num_stages=2,
        )

        atk_packed_loc = torch.cat([p_atk_loc.unsqueeze(-1), a_atk_loc], dim=-1).contiguous()
        atk_packed_all = torch.empty(world_size, B, H, K + 1, device=q_loc.device, dtype=torch.float32)
        dist.all_gather_into_tensor(atk_packed_all, atk_packed_loc)
        p_atk_all = atk_packed_all[:, :, :, 0]
        a_atk_all = atk_packed_all[:, :, :, 1:]

        X_prefix = torch.zeros(B, H, K, device=q_loc.device, dtype=torch.float32)
        for r in range(rank):
            X_prefix = p_atk_all[r].unsqueeze(-1) * X_prefix + a_atk_all[r]

        # Now compute exact local k_tilde using exact X_prefix
        x_sub_init = torch.empty(B, NS, H, K, device=q_loc.device, dtype=torch.float32)
        sa_sub = torch.empty(B, NS, H, device=q_loc.device, dtype=torch.float32)
        x_final = torch.empty(B, H, K, device=q_loc.device, dtype=torch.float32)
        _atk_fused_subchunk_state_fwd_kernel[(NK, B * H)](
            kn, beta_atk_loc.contiguous(), g_atk_loc.contiguous(), X_prefix,
            x_sub_init, sa_sub, x_final, p_atk_loc, a_atk_loc,
            T=T_loc, H=H, K=K, BC=BC, BK=BK_sub, HAS_X0=True, num_warps=4, num_stages=2,
        )
        k_tilde_loc = torch.empty_like(kn)
        _atk_subchunk_squash_out_fwd_kernel[(NS, NK, B * H)](
            kn, beta_atk_loc.contiguous(), g_atk_loc.contiguous(), x_sub_init, log_atk_scale.contiguous(),
            k_tilde_loc, logx=float(math.log(float(x))), eps=float(eps),
            T=T_loc, H=H, K=K, BC=BC, BK=BK_sub, num_warps=4, num_stages=2,
        )

        # Stage 2 (`KDA-CP`): Build exact Asymmetric Matrix Monoid (P_rank, H_rank) using exact k_tilde_loc
        P_rank, H_rank = _build_pkda_stage2_rank_affine_summary(
            qn, kn, k_tilde_loc, v_loc.contiguous(), g_loc.contiguous(), beta_loc.contiguous(), scale
        )
        mat_packed_loc = torch.cat([P_rank, H_rank], dim=-1).contiguous()
        mat_packed_all = torch.empty(world_size, B, H, K, K + V, device=q_loc.device, dtype=torch.float32)
        dist.all_gather_into_tensor(mat_packed_all, mat_packed_loc)
        P_all = mat_packed_all[:, :, :, :, :K]
        H_all = mat_packed_all[:, :, :, :, K:]

        S_prefix = torch.zeros(B, H, K, V, device=q_loc.device, dtype=torch.float32)
        for r in range(rank):
            S_prefix = P_all[r] @ S_prefix + H_all[r]

        o_loc, _, _, cache = chunk_precond_kda_rsi_fwd(
            qn, kn, v_loc.contiguous(), g_loc.contiguous(), g_atk_loc.contiguous(),
            beta_atk_loc.contiguous(), beta_loc.contiguous(), log_atk_scale.contiguous(),
            scale, initial_state=S_prefix, initial_A_state=X_prefix, output_final_state=False, x=x, eps=eps,
        )
        ctx.save_for_backward(
            qn, kn, q_rstd, k_rstd, v_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale,
            p_atk_all, P_all, *cache,
        )
        ctx.scale = scale
        ctx.x = x
        ctx.eps = eps
        ctx.omit_handoff = False
        return o_loc

    @staticmethod
    def backward(ctx, do_loc):
        if ctx.omit_handoff:
            saved = ctx.saved_tensors
            qn, kn, q_rstd, k_rstd, v_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale = saved[:9]
            cache = saved[9:]
            dqn, dkn, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale, _, _ = chunk_precond_kda_rsi_bwd(
                qn, kn, v_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale, cache,
                do_loc.contiguous(), None, None, ctx.scale, x=ctx.x, eps=ctx.eps,
            )
            qnf, knf = qn.float(), kn.float()
            dqf, dkf = dqn.float(), dkn.float()
            dq = ((dqf - qnf * (qnf * dqf).sum(dim=-1, keepdim=True)) * q_rstd.unsqueeze(-1)).to(qn.dtype)
            dk = ((dkf - knf * (knf * dkf).sum(dim=-1, keepdim=True)) * k_rstd.unsqueeze(-1)).to(kn.dtype)
            return dq, dk, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale, None, None, None, None

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        saved = ctx.saved_tensors
        qn, kn, q_rstd, k_rstd, v_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale, p_atk_all, P_all = saved[:11]
        cache = saved[11:]
        B, T_loc, H, K = qn.shape
        V = v_loc.shape[-1]

        # Pass 1: Stage-2 inhomogeneous matrix boundary gradient G_rank = dS_init(dht=0)
        _, _, _, _, _, _, _, _, G_rank, _ = chunk_precond_kda_rsi_bwd(
            qn, kn, v_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale, cache,
            do_loc.contiguous(), None, None, ctx.scale, x=ctx.x, eps=ctx.eps,
        )
        G_all = torch.empty(world_size, B, H, K, V, device=qn.device, dtype=torch.float32)
        dist.all_gather_into_tensor(G_all, G_rank.contiguous())

        dht = torch.zeros(B, H, K, V, device=qn.device, dtype=torch.float32)
        for q_idx in range(world_size - 1, rank, -1):
            dht = G_all[q_idx] + P_all[q_idx].transpose(-1, -2) @ dht

        # Pass 2: With exact dht, obtain exact Stage-1 inhomogeneous curvature boundary gradient gamma_atk = dX_init(dat=0)
        _, _, _, _, _, _, _, _, _, gamma_atk = chunk_precond_kda_rsi_bwd(
            qn, kn, v_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale, cache,
            do_loc.contiguous(), dht, None, ctx.scale, x=ctx.x, eps=ctx.eps,
        )
        gamma_all = torch.empty(world_size, B, H, K, device=qn.device, dtype=torch.float32)
        dist.all_gather_into_tensor(gamma_all, gamma_atk.contiguous())

        dat = torch.zeros(B, H, K, device=qn.device, dtype=torch.float32)
        for q_idx in range(world_size - 1, rank, -1):
            dat = gamma_all[q_idx] + p_atk_all[q_idx].unsqueeze(-1) * dat

        # Pass 3: Final local backward seeded with exact (dht, dat)
        dqn, dkn, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale, _, _ = chunk_precond_kda_rsi_bwd(
            qn, kn, v_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale, cache,
            do_loc.contiguous(), dht, dat, ctx.scale, x=ctx.x, eps=ctx.eps,
        )

        qnf, knf = qn.float(), kn.float()
        dqf, dkf = dqn.float(), dkn.float()
        dq = ((dqf - qnf * (qnf * dqf).sum(dim=-1, keepdim=True)) * q_rstd.unsqueeze(-1)).to(qn.dtype)
        dk = ((dkf - knf * (knf * dkf).sum(dim=-1, keepdim=True)) * k_rstd.unsqueeze(-1)).to(kn.dtype)
        return dq, dk, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale, None, None, None, None


def chunk_precond_kda_cp8_rsi(
    q_loc: torch.Tensor,
    k_loc: torch.Tensor,
    v_loc: torch.Tensor,
    g_loc: torch.Tensor,
    g_atk_loc: torch.Tensor,
    beta_atk_loc: torch.Tensor,
    beta_loc: torch.Tensor,
    log_atk_scale: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    x: float = 1.5,
    eps: float = 1e-6,
    use_qk_l2norm: bool = True,
    omit_handoff: bool = False,
) -> torch.Tensor:
    if scale is None:
        scale = q_loc.shape[-1] ** -0.5
    if log_atk_scale is None:
        log_atk_scale = torch.full((q_loc.shape[2],), -0.2, device=q_loc.device, dtype=torch.float32)
    return _PrecondKDACP8AutogradFn.apply(
        q_loc, k_loc, v_loc, g_loc, g_atk_loc, beta_atk_loc, beta_loc, log_atk_scale,
        float(scale), float(x), float(eps), omit_handoff,
    )


# =============================================================================
# 9. SM80 Cubin Register & Spill Audit Utility
# =============================================================================

def audit_precond_kda_rsi_kernels() -> Dict[str, object]:
    """
    Compiles all 24 Round-16 RSI Triton kernels across both FP32 and BF16 and head dims
    `K, V in {64, 128}` on SM80, plus the 2 upstream spill/overflow negative-control kernels,
    and verifies `0` cubin spills across 100% of Round-16 RSI specializations.
    """
    device = torch.device("cuda")
    records = []
    B, T, H = 2, 128, 4
    for dtype in (torch.float32, torch.bfloat16):
        for K in (64, 128):
            V = K
            q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
            g = (-F.softplus(torch.randn(B, T, H, K, device=device, dtype=torch.float32)) - 0.1).requires_grad_(True)
            g_atk = (-F.softplus(torch.randn(B, T, H, device=device, dtype=torch.float32)) - 0.1).requires_grad_(True)
            beta_atk = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid().detach().requires_grad_(True)
            beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid().detach().requires_grad_(True)
            log_atk_scale = torch.full((H,), -0.2, device=device, dtype=torch.float32, requires_grad=True)
            h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
            a0 = torch.rand(B, H, K, device=device, dtype=torch.float32, requires_grad=True)

            o, ht, at = chunk_precond_kda_rsi(
                q, k, v, g, g_atk, beta_atk, beta, log_atk_scale=log_atk_scale,
                initial_state=h0, initial_A_state=a0, output_final_state=True,
            )
            (o.float().sum() + ht.sum() + at.sum()).backward()

            blk = PrecondKDABlockRSI(hidden_size=H * K, num_heads=H, head_dim=K).to(device=device, dtype=dtype)
            x = torch.randn(B, T, H * K, device=device, dtype=dtype, requires_grad=True)
            y = blk(x, use_rsi_kernels=True, fusion_stage=4)
            y.float().sum().backward()

    rsi_kernels = [
        ("_atk_fused_subchunk_state_fwd_kernel", _atk_fused_subchunk_state_fwd_kernel),
        ("_atk_subchunk_squash_out_fwd_kernel", _atk_subchunk_squash_out_fwd_kernel),
        ("_atk_subchunk_squash_bwd_kernel", _atk_subchunk_squash_bwd_kernel),
        ("_atk_reverse_subchunk_scan_bwd_kernel", _atk_reverse_subchunk_scan_bwd_kernel),
        ("_pkda_fused_cumsum_g_fwd_kernel", _pkda_fused_cumsum_g_fwd_kernel),
        ("_pkda_overflow_free_intra_diag_subchunk_fwd_kernel", _pkda_overflow_free_intra_diag_subchunk_fwd_kernel),
        ("_pkda_zero_spill_inter_subchunk_Aqk_fwd_kernel", _pkda_zero_spill_inter_subchunk_Aqk_fwd_kernel),
        ("_pkda_zero_spill_inter_subchunk_Akk_solve_fwd_kernel", _pkda_zero_spill_inter_subchunk_Akk_solve_fwd_kernel),
        ("_pkda_wy_wu_qg_kg_fwd_kernel", _pkda_wy_wu_qg_kg_fwd_kernel),
        ("_pkda_chunk_state_h_fwd_kernel", _pkda_chunk_state_h_fwd_kernel),
        ("_pkda_chunk_readout_o_fwd_kernel", _pkda_chunk_readout_o_fwd_kernel),
        ("_pkda_bwd_dAv_kernel", _pkda_bwd_dAv_kernel),
        ("_pkda_bwd_dhu_reverse_kernel", _pkda_bwd_dhu_reverse_kernel),
        ("_pkda_bwd_wy_dqk_slice_kernel", _pkda_bwd_wy_dqk_slice_kernel),
        ("_pkda_bwd_wy_dAkk_dv_db_kernel", _pkda_bwd_wy_dAkk_dv_db_kernel),
        ("_pkda_bwd_wy_dAkk_solve_kernel", _pkda_bwd_wy_dAkk_solve_kernel),
        ("_pkda_bwd_intra_subchunk_asymmetric_kernel", _pkda_bwd_intra_subchunk_asymmetric_kernel),
        ("_pkda_bwd_rcumsum_dg_kernel", _pkda_bwd_rcumsum_dg_kernel),
        ("_pkda_fused_shortconv_silu_fwd_kernel", _pkda_fused_shortconv_silu_fwd_kernel),
        ("_pkda_fused_shortconv_silu_bwd_kernel", _pkda_fused_shortconv_silu_bwd_kernel),
        ("_pkda_superfused_qk_l2norm_gates_fwd_kernel", _pkda_superfused_qk_l2norm_gates_fwd_kernel),
        ("_pkda_superfused_qk_l2norm_gates_bwd_kernel", _pkda_superfused_qk_l2norm_gates_bwd_kernel),
        ("_pkda_fused_rmsnorm_sigmoid_gate_fwd_kernel", _pkda_fused_rmsnorm_sigmoid_gate_fwd_kernel),
        ("_pkda_fused_rmsnorm_sigmoid_gate_bwd_kernel", _pkda_fused_rmsnorm_sigmoid_gate_bwd_kernel),
    ]

    total_specs = 0
    total_spills = 0
    max_regs = 0
    for name, kfn in rsi_kernels:
        cache_dict = getattr(kfn, "device_caches", {})
        k_specs = 0
        k_spills = 0
        k_max_regs = 0
        for dev_cache in cache_dict.values():
            bin_map = dev_cache[0] if isinstance(dev_cache, tuple) else dev_cache
            for cbin in bin_map.values():
                n_regs = int(getattr(cbin, "n_regs", 0))
                n_spills = int(getattr(cbin, "n_spills", 0))
                k_specs += 1
                k_spills += n_spills
                k_max_regs = max(k_max_regs, n_regs)
        total_specs += k_specs
        total_spills += k_spills
        max_regs = max(max_regs, k_max_regs)
        records.append({
            "kernel": name,
            "specializations": k_specs,
            "max_regs": k_max_regs,
            "total_spills": k_spills,
        })

    # Compile negative-control upstream spill kernel at K=V=128
    K_nc = 128
    BT = 64
    q_nc = torch.randn(B, T, H, K_nc, device=device, dtype=torch.bfloat16)
    k_nc = torch.randn_like(q_nc)
    kt_nc = torch.randn_like(q_nc)
    v_nc = torch.randn_like(q_nc)
    vn_nc = torch.randn_like(q_nc)
    g_nc = torch.full((B, T, H, K_nc), -0.2, device=device, dtype=torch.float32)
    b_nc = torch.rand(B, T, H, device=device, dtype=torch.bfloat16)
    A_nc = torch.randn(B, T, H, BT, device=device, dtype=torch.bfloat16)
    h_nc = torch.randn(B, T // BT, H, K_nc, K_nc, device=device, dtype=torch.bfloat16)
    dh_nc = torch.randn_like(h_nc)
    do_nc = torch.randn_like(v_nc)
    dv_nc = torch.randn_like(v_nc)
    dq_nc = torch.empty(B, T, H, K_nc, device=device, dtype=torch.float32)
    dk_nc = torch.empty_like(dq_nc)
    dkg_nc = torch.empty_like(dq_nc)
    dv2_nc = torch.empty_like(v_nc)
    dg_nc = torch.empty_like(dq_nc)
    db_nc = torch.empty(B, T, H, device=device, dtype=torch.float32)
    dA_nc = torch.empty(B, T, H, BT, device=device, dtype=torch.float32)

    _pkda_upstream_wy_dqkg_bwd_spill_kernel[(T // BT, B * H)](
        q_nc, k_nc, kt_nc, v_nc, vn_nc, g_nc, b_nc, A_nc, h_nc, do_nc, dh_nc, dv_nc,
        dq_nc, dk_nc, dkg_nc, dv2_nc, dg_nc, db_nc, dA_nc, K_nc ** -0.5,
        T=T, H=H, K=K_nc, V=K_nc, BT=BT, BK=64, BV=64, num_warps=4, num_stages=2,
    )

    nc_spills = 0
    nc_regs = 0
    for dev_cache in getattr(_pkda_upstream_wy_dqkg_bwd_spill_kernel, "device_caches", {}).values():
        bin_map = dev_cache[0] if isinstance(dev_cache, tuple) else dev_cache
        for cbin in bin_map.values():
            nc_spills = max(nc_spills, int(getattr(cbin, "n_spills", 0)))
            nc_regs = max(nc_regs, int(getattr(cbin, "n_regs", 0)))

    return {
        "kernel_count": len(rsi_kernels),
        "total_specializations": total_specs,
        "total_spills": total_spills,
        "max_regs": max_regs,
        "all_zero_spill": (total_spills == 0),
        "upstream_wy_bwd_spill_repro_spills": nc_spills,
        "upstream_wy_bwd_spill_repro_regs": nc_regs,
        "kernels": records,
    }
