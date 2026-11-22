#!/usr/bin/env python3
"""
E144 (v135, Round-18 RSI): Momentum DeltaNet Zero-Spill Triton Kernels & Ulysses-WY Momentum-CP=8
==================================================================================================

Implements Round-18 RSI zero-spill SM80 Triton kernels for Momentum DeltaNet (`arXiv:2605.05838`;
official repository `fla-org/flash-linear-attention/tree/main/fla/ops/momentum_delta_rule` &
`fla/layers/momentum_deltanet.py`):
  1. Coupled Dual-State `[S_t, M_t]` Second-Order Heavy-Ball Momentum Test-Time Gradient Descent:
       v_new_t = v_t - S_{t-1}^T p_eff_t
       M_t     = mu_t M_{t-1} - k_eff_t (v_new_t)^T
       S_t     = alpha_t S_{t-1} - beta_t M_t
       o_t     = S_t^T q_eff_t
  2. Exact Bias-Free Two-Timescale Green's Function Precompute (`_mdr_chunk_gate_greens_fwd_kernel`):
       Gamma_q = D_alpha @ (diag(beta) * D_mu),   b_t = D_alpha @ (beta * m_cum)
     Eliminates upstream `triton_ascend`'s `(beta + 1e-6).log()` + `logcumsumexp` bias and eliminates
     redundant `[64, 64]` matrix multiplies inside `kkt_fwd`, `fwd_h`, and `fwd_o`.
  3. Zero-Spill 3-RHS Coupled WY Forward & Adjoint Factorization (`dX = Ai^T @ dV_new`):
       U_c = Ai @ V,   Y_c = Ai @ (bar_a_tm1 * P_eff),   Z_c = Ai @ (b_tm1 * P_eff)
       V_new = U_c - Y_c @ S_pre + Z_c @ M_pre
     In backward, solving a single `dX = Ai^T @ dV_new` yields:
       dV = dX,   dA = -strictTril(dX @ V_new^T),   dP_y = -dX @ S_pre^T,   dP_z = +dX @ M_pre^T
     without materializing `dY_c`, `dZ_c`, or `-Ai^T @ dAi @ Ai^T`, replacing upstream's slow
     `with torch.enable_grad(): torch.autograd.grad(...)` forward-replay backward with pure
     zero-spill Triton kernels (`0` cubin register spills across all 26 kernels).
  4. In-Register `[K=64, BV=32]` Dual-State Recurrence + L2-Synchronized `[BK=64, BV=32]` Recurrence
     for `K in {128, 256}` (avoiding upstream `triton_ascend`'s `4,624` spills at `K=64` and hard
     `SM80 OutOfResources` shared-memory crashes at `K=128` `223,768 B` and `K=256` `420,376 B`).
  5. Superfused `MomentumDeltaNetBlockRSI` (`3x ShortConv1d+SiLU`, Overflow-Safe Algebraic Symplectic
     4-Gate Fusion `sin^2(arctan r) = r^2/(1+r^2)` & `log cos^2(arctan r) = -log(1+r^2)`, Output
     Correction `q - exp(D_log)*k`, Triple `L2Norm`, and `FusedRMSNormGated`) + 8-GPU NVSwitch
     Ulysses-WY `Momentum-CP = 8`.
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


# ==================================================================================================
# 1. Exact FP64 Recurrent Oracle & Eager Chunk Reference
# ==================================================================================================

def momentum_deltanet_recurrent_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    p: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_p_times_alpha: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """True FP64 token-by-token recurrent oracle for Momentum DeltaNet (`arXiv:2605.05838`)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    if p is None:
        p = k
    if beta is None:
        beta = torch.ones(B, T, H, device=q.device, dtype=q.dtype)
    if eta is None:
        eta = torch.ones(B, T, H, device=q.device, dtype=q.dtype)

    qd = q.to(torch.float64)
    kd = k.to(torch.float64)
    vd = v.to(torch.float64)
    pd = p.to(torch.float64)
    la = log_alpha.to(torch.float64)
    lm = log_mu.to(torch.float64)
    bt = beta.to(torch.float64)
    et = eta.to(torch.float64)

    if use_qk_l2norm_in_kernel:
        qd = qd * torch.rsqrt((qd * qd).sum(dim=-1, keepdim=True) + 1e-12)
        kd = kd * torch.rsqrt((kd * kd).sum(dim=-1, keepdim=True) + 1e-12)
        pd = pd * torch.rsqrt((pd * pd).sum(dim=-1, keepdim=True) + 1e-12)

    q_eff = qd * float(scale)
    k_eff = kd * et.unsqueeze(-1)
    p_eff = pd * (la.exp().unsqueeze(-1) if use_p_times_alpha else 1.0)
    alpha = la.exp()
    mu = lm.exp()

    if initial_state is not None:
        S = initial_state[0].to(torch.float64).clone()
        M = initial_state[1].to(torch.float64).clone()
    else:
        S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float64)
        M = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float64)

    outs = []
    for t in range(T):
        vt_new = vd[:, t] - torch.einsum("bhkv,bhk->bhv", S, p_eff[:, t])
        M = mu[:, t, :, None, None] * M - torch.einsum("bhk,bhv->bhkv", k_eff[:, t], vt_new)
        S = alpha[:, t, :, None, None] * S - bt[:, t, :, None, None] * M
        ot = torch.einsum("bhkv,bhk->bhv", S, q_eff[:, t])
        outs.append(ot)

    o = torch.stack(outs, dim=1)
    final_state = torch.stack([S, M], dim=0) if output_final_state else None
    return o, final_state


def momentum_deltanet_chunk_eager_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    p: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_p_times_alpha: bool = True,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Differentiable PyTorch chunk reference matching upstream `fla/ops/momentum_delta_rule/naive.py`."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    if scale is None:
        scale = K ** -0.5
    if p is None:
        p = k
    if beta is None:
        beta = torch.ones(B, T, H, device=q.device, dtype=q.dtype)
    if eta is None:
        eta = torch.ones(B, T, H, device=q.device, dtype=q.dtype)

    orig_dtype = q.dtype
    qf, kf, vf, pf = q.float(), k.float(), v.float(), p.float()
    laf, lmf, betaf, etaf = log_alpha.float(), log_mu.float(), beta.float(), eta.float()

    if use_qk_l2norm_in_kernel:
        qf = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6)
        kf = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)
        pf = pf * torch.rsqrt((pf * pf).sum(dim=-1, keepdim=True) + 1e-6)

    q_eff = qf * float(scale)
    k_eff = kf * etaf.unsqueeze(-1)
    p_eff = pf * (laf.exp().unsqueeze(-1) if use_p_times_alpha else 1.0)

    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q_eff = F.pad(q_eff, (0, 0, 0, 0, 0, pad_len))
        k_eff = F.pad(k_eff, (0, 0, 0, 0, 0, pad_len))
        vf = F.pad(vf, (0, 0, 0, 0, 0, pad_len))
        p_eff = F.pad(p_eff, (0, 0, 0, 0, 0, pad_len))
        laf = F.pad(laf, (0, 0, 0, pad_len))
        lmf = F.pad(lmf, (0, 0, 0, pad_len))
        betaf = F.pad(betaf, (0, 0, 0, pad_len))

    T_pad = T + pad_len
    NT = T_pad // BT
    q_c = q_eff.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    k_c = k_eff.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    v_c = vf.reshape(B, NT, BT, H, V).permute(0, 3, 1, 2, 4)
    p_c = p_eff.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    la_c = laf.reshape(B, NT, BT, H).permute(0, 3, 1, 2)
    lm_c = lmf.reshape(B, NT, BT, H).permute(0, 3, 1, 2)
    beta_c = betaf.reshape(B, NT, BT, H).permute(0, 3, 1, 2)

    la_cum = la_c.cumsum(-1)
    lm_cum = lm_c.cumsum(-1)
    a_cum = la_cum.exp()
    m_cum = lm_cum.exp()
    bar_a_tm1 = torch.cat([torch.ones_like(a_cum[..., :1]), a_cum[..., :-1]], dim=-1)
    Da = (la_cum.unsqueeze(-1) - la_cum.unsqueeze(-2)).tril().exp().tril()
    Dm = (lm_cum.unsqueeze(-1) - lm_cum.unsqueeze(-2)).tril().exp().tril()
    Gamma_q = Da @ (beta_c.unsqueeze(-1) * Dm)
    b_t = (Da @ (beta_c * m_cum).unsqueeze(-1)).squeeze(-1)
    b_tm1 = torch.cat([torch.zeros_like(b_t[..., :1]), b_t[..., :-1]], dim=-1)
    Gamma = torch.cat([torch.zeros_like(Gamma_q[..., :1, :]), Gamma_q[..., :-1, :]], dim=-2).tril(-1)

    attn = (p_c @ k_c.transpose(-1, -2)) * Gamma
    attn_inv = -attn
    for i in range(1, BT):
        attn_inv[..., i, :i] = attn_inv[..., i, :i] + (attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()).sum(-2)
    attn_inv = attn_inv + torch.eye(BT, dtype=attn_inv.dtype, device=q.device)

    u_c = attn_inv @ v_c
    y_c = attn_inv @ (bar_a_tm1.unsqueeze(-1) * p_c)
    z_c = attn_inv @ (b_tm1.unsqueeze(-1) * p_c)

    if initial_state is not None:
        S_pre = initial_state[0].float()
        M_pre = initial_state[1].float()
    else:
        S_pre = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
        M_pre = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)

    o_list = []
    for c in range(NT):
        vn = u_c[:, :, c] - y_c[:, :, c] @ S_pre + z_c[:, :, c] @ M_pre
        Mqk = (q_c[:, :, c] @ k_c[:, :, c].transpose(-1, -2)) * Gamma_q[:, :, c]
        oc = (a_cum[:, :, c].unsqueeze(-1) * q_c[:, :, c]) @ S_pre - (b_t[:, :, c].unsqueeze(-1) * q_c[:, :, c]) @ M_pre + Mqk @ vn
        o_list.append(oc)
        ds = Gamma_q[:, :, c, -1].unsqueeze(-1)
        dm = Dm[:, :, c, -1].unsqueeze(-1)
        S_next = a_cum[:, :, c, -1, None, None] * S_pre - b_t[:, :, c, -1, None, None] * M_pre + (k_c[:, :, c] * ds).transpose(-1, -2) @ vn
        M_next = m_cum[:, :, c, -1, None, None] * M_pre - (k_c[:, :, c] * dm).transpose(-1, -2) @ vn
        S_pre, M_pre = S_next, M_next

    o = torch.stack(o_list, dim=2).permute(0, 2, 3, 1, 4).reshape(B, T_pad, H, V)[:, :T].to(orig_dtype)
    final_state = torch.stack([S_pre, M_pre], dim=0) if output_final_state else None
    return o, final_state


# ==================================================================================================
# 2. Zero-Spill Round-18 RSI Triton Kernels (Forward & Analytical Backward)
# ==================================================================================================

@triton.jit
def _mdr_triple_l2norm_scale_fwd_kernel(
    Q, K_in, P_in, Eta, LogAlpha,
    Q_eff, K_eff, P_eff, Q_rstd, K_rstd, P_rstd,
    scale,
    K_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_NORM: tl.constexpr,
    USE_P_ALPHA: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K_DIM

    b_q = tl.load(Q + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(K_in + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    b_p = tl.load(P_in + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)

    if USE_NORM:
        rq = tl.rsqrt(tl.sum(b_q * b_q, axis=0) + EPS)
        rk = tl.rsqrt(tl.sum(b_k * b_k, axis=0) + EPS)
        rp = tl.rsqrt(tl.sum(b_p * b_p, axis=0) + EPS)
        tl.store(Q_rstd + row, rq)
        tl.store(K_rstd + row, rk)
        tl.store(P_rstd + row, rp)
        b_q = b_q * rq
        b_k = b_k * rk
        b_p = b_p * rp

    b_eta = tl.load(Eta + row).to(tl.float32)
    b_la = tl.load(LogAlpha + row).to(tl.float32)
    p_mult = tl.exp(b_la) if USE_P_ALPHA else 1.0

    tl.store(Q_eff + row * K_DIM + offs_k, (b_q * scale).to(Q_eff.dtype.element_ty), mask=mask_k)
    tl.store(K_eff + row * K_DIM + offs_k, (b_k * b_eta).to(K_eff.dtype.element_ty), mask=mask_k)
    tl.store(P_eff + row * K_DIM + offs_k, (b_p * p_mult).to(P_eff.dtype.element_ty), mask=mask_k)


@triton.jit
def _mdr_triple_l2norm_scale_bwd_kernel(
    Q, K_in, P_in, Q_rstd, K_rstd, P_rstd, Eta, LogAlpha,
    dQ_eff, dK_eff, dP_eff, dLA_intra,
    dQ, dK, dP, dEta, dLA_out,
    scale,
    K_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_NORM: tl.constexpr,
    USE_P_ALPHA: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K_DIM

    b_q = tl.load(Q + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(K_in + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    b_p = tl.load(P_in + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)

    b_dqe = tl.load(dQ_eff + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    b_dke = tl.load(dK_eff + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    b_dpe = tl.load(dP_eff + row * K_DIM + offs_k, mask=mask_k, other=0.0).to(tl.float32)

    if USE_NORM:
        rq = tl.load(Q_rstd + row).to(tl.float32)
        rk = tl.load(K_rstd + row).to(tl.float32)
        rp = tl.load(P_rstd + row).to(tl.float32)
        qn = b_q * rq
        kn = b_k * rk
        pn = b_p * rp
    else:
        rq = 1.0
        rk = 1.0
        rp = 1.0
        qn = b_q
        kn = b_k
        pn = b_p

    b_eta = tl.load(Eta + row).to(tl.float32)
    b_la = tl.load(LogAlpha + row).to(tl.float32)
    alpha = tl.exp(b_la)
    p_mult = alpha if USE_P_ALPHA else 1.0

    deta_val = tl.sum(b_dke * kn, axis=0)
    dla_val = tl.load(dLA_intra + row).to(tl.float32)
    if USE_P_ALPHA:
        dla_val = dla_val + tl.sum(b_dpe * pn, axis=0) * alpha

    tl.store(dEta + row, deta_val.to(dEta.dtype.element_ty))
    tl.store(dLA_out + row, dla_val.to(dLA_out.dtype.element_ty))

    dq_n = b_dqe * scale
    dk_n = b_dke * b_eta
    dp_n = b_dpe * p_mult

    if USE_NORM:
        dq_out = rq * (dq_n - qn * tl.sum(qn * dq_n, axis=0))
        dk_out = rk * (dk_n - kn * tl.sum(kn * dk_n, axis=0))
        dp_out = rp * (dp_n - pn * tl.sum(pn * dp_n, axis=0))
    else:
        dq_out = dq_n
        dk_out = dk_n
        dp_out = dp_n

    tl.store(dQ + row * K_DIM + offs_k, dq_out.to(dQ.dtype.element_ty), mask=mask_k)
    tl.store(dK + row * K_DIM + offs_k, dk_out.to(dK.dtype.element_ty), mask=mask_k)
    tl.store(dP + row * K_DIM + offs_k, dp_out.to(dP.dtype.element_ty), mask=mask_k)


@triton.jit
def _mdr_chunk_gate_greens_fwd_kernel(
    LogAlpha, LogMu, Beta,
    LA_cum, LM_cum, A_cum, M_cum, BarA_tm1, B_t, B_tm1,
    DS_vec, DM_vec, A_last, B_last, M_last,
    Gamma_q,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    """
    1D Row-Recurrence Green's Function Precompute (`0` 2D [64, 64] matrices in registers!):
      For row i = 0 .. BT-1:
        row_dm[j] = (j < i) ? row_dm[j] * mu_i : ((j == i) ? 1.0 : 0.0)
        row_gq[j] = alpha_i * row_gq[j] + beta_i * row_dm[j]
        b_cur     = alpha_i * b_cur + beta_i * m_cum_i
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_j = tl.arange(0, BT)
    row_dm = tl.zeros([BT], dtype=tl.float32)
    row_gq = tl.zeros([BT], dtype=tl.float32)

    la_cum_i = 0.0
    lm_cum_i = 0.0
    a_prev = 1.0
    b_prev = 0.0

    vec_base = (i_bh * NT + i_t) * BT
    gq_base = Gamma_q + (i_bh * NT + i_t) * BT * BT

    for i in range(BT):
        t_cur = i_t * BT + i
        in_off = (i_b * T + t_cur) * H + i_h
        la_i = tl.load(LogAlpha + in_off).to(tl.float32)
        lm_i = tl.load(LogMu + in_off).to(tl.float32)
        beta_i = tl.load(Beta + in_off).to(tl.float32)

        la_cum_i = la_cum_i + la_i
        lm_cum_i = lm_cum_i + lm_i
        a_cum_i = tl.exp(la_cum_i)
        m_cum_i = tl.exp(lm_cum_i)
        alpha_i = tl.exp(la_i)
        mu_i = tl.exp(lm_i)

        row_dm = tl.where(o_j < i, row_dm * mu_i, tl.where(o_j == i, 1.0, 0.0))
        row_gq = alpha_i * row_gq + beta_i * row_dm
        b_cur = alpha_i * b_prev + beta_i * m_cum_i

        tl.store(LA_cum + vec_base + i, la_cum_i)
        tl.store(LM_cum + vec_base + i, lm_cum_i)
        tl.store(A_cum + vec_base + i, a_cum_i)
        tl.store(M_cum + vec_base + i, m_cum_i)
        tl.store(BarA_tm1 + vec_base + i, a_prev)
        tl.store(B_t + vec_base + i, b_cur)
        tl.store(B_tm1 + vec_base + i, b_prev)
        tl.store(gq_base + i * BT + o_j, row_gq)

        a_prev = a_cum_i
        b_prev = b_cur

    tl.store(DS_vec + vec_base + o_j, row_gq)
    tl.store(DM_vec + vec_base + o_j, row_dm)
    sc_off = i_bh * NT + i_t
    tl.store(A_last + sc_off, a_prev)
    tl.store(B_last + sc_off, b_prev)
    tl.store(M_last + sc_off, tl.exp(lm_cum_i))


@triton.jit
def _mdr_chunk_kkt_fwd_kernel(
    Q_eff, P_eff, K_eff, Gamma_q, A_out, Mqk_out,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    """Row-and-col-subtiled (BM=16, BC=32, BK=32) intra-chunk score builder for A and Mqk."""
    i_tr = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NSUB: tl.constexpr = BT // BM
    i_t = i_tr // NSUB
    i_sub = i_tr % NSUB
    NT = T // BT

    o_r = i_sub * BM + tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    t_r = i_t * BT + o_r
    r_prev = tl.maximum(o_r - 1, 0)
    base_bh = (i_b * T * H + i_h) * K
    gq_base = Gamma_q + (i_bh * NT + i_t) * BT * BT

    for ic in range(2):
        o_c = ic * 32 + tl.arange(0, 32)
        t_c = i_t * BT + o_c
        b_pk = tl.zeros([BM, 32], dtype=tl.float32)
        b_qk = tl.zeros([BM, 32], dtype=tl.float32)

        for ik in range(tl.cdiv(K, BK)):
            offs_k = ik * BK + o_k
            q_r = tl.load(Q_eff + base_bh + t_r[:, None] * (H * K) + offs_k[None, :])
            p_r = tl.load(P_eff + base_bh + t_r[:, None] * (H * K) + offs_k[None, :])
            k_c_T = tl.trans(tl.load(K_eff + base_bh + t_c[:, None] * (H * K) + offs_k[None, :]))
            b_pk = tl.dot(p_r, k_c_T, b_pk, input_precision=DOT_PREC)
            b_qk = tl.dot(q_r, k_c_T, b_qk, input_precision=DOT_PREC)

        b_gq = tl.load(gq_base + o_r[:, None] * BT + o_c[None, :])
        b_gamma = tl.load(gq_base + r_prev[:, None] * BT + o_c[None, :])
        m_strict = o_r[:, None] > o_c[None, :]
        m_tril = o_r[:, None] >= o_c[None, :]
        b_A = tl.where(m_strict, b_pk * b_gamma, 0.0)
        b_Mqk = tl.where(m_tril, b_qk * b_gq, 0.0)

        out_off = (i_bh * NT + i_t) * BT * BT + o_r[:, None] * BT + o_c[None, :]
        tl.store(A_out + out_off, b_A)
        tl.store(Mqk_out + out_off, b_Mqk)


@triton.jit
def _mdr_solve_tril_16x16(b_A):
    o_i = tl.arange(0, 16)
    m_lt = o_i[:, None] > o_i[None, :]
    b_inv = -tl.where(m_lt, b_A, 0.0)
    for i in range(2, 16):
        row_mask = o_i == i
        a_row = tl.sum(tl.where(row_mask[:, None], -b_A, 0.0), axis=0)
        contrib = tl.sum(a_row[:, None] * b_inv, axis=0)
        b_inv = tl.where(row_mask[:, None] & (o_i[None, :] < i), b_inv + contrib[None, :], b_inv)
    b_inv = b_inv + (o_i[:, None] == o_i[None, :]).to(tl.float32)
    return b_inv


@triton.jit
def _mdr_ut_solve_64x64_kernel(
    A_in, Ai_out,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = pid * 4096
    r = tl.arange(0, 16)
    c = tl.arange(0, 16)

    A00 = tl.load(A_in + base + (0 + r)[:, None] * 64 + (0 + c)[None, :]).to(tl.float32)
    A11 = tl.load(A_in + base + (16 + r)[:, None] * 64 + (16 + c)[None, :]).to(tl.float32)
    A22 = tl.load(A_in + base + (32 + r)[:, None] * 64 + (32 + c)[None, :]).to(tl.float32)
    A33 = tl.load(A_in + base + (48 + r)[:, None] * 64 + (48 + c)[None, :]).to(tl.float32)

    T00 = _mdr_solve_tril_16x16(A00)
    T11 = _mdr_solve_tril_16x16(A11)
    T22 = _mdr_solve_tril_16x16(A22)
    T33 = _mdr_solve_tril_16x16(A33)

    A10 = tl.load(A_in + base + (16 + r)[:, None] * 64 + (0 + c)[None, :]).to(tl.float32)
    A20 = tl.load(A_in + base + (32 + r)[:, None] * 64 + (0 + c)[None, :]).to(tl.float32)
    A21 = tl.load(A_in + base + (32 + r)[:, None] * 64 + (16 + c)[None, :]).to(tl.float32)
    A30 = tl.load(A_in + base + (48 + r)[:, None] * 64 + (0 + c)[None, :]).to(tl.float32)
    A31 = tl.load(A_in + base + (48 + r)[:, None] * 64 + (16 + c)[None, :]).to(tl.float32)
    A32 = tl.load(A_in + base + (48 + r)[:, None] * 64 + (32 + c)[None, :]).to(tl.float32)

    T10 = -tl.dot(tl.dot(T11, A10, input_precision="ieee"), T00, input_precision="ieee")
    T21 = -tl.dot(tl.dot(T22, A21, input_precision="ieee"), T11, input_precision="ieee")
    T20 = -tl.dot(T22, tl.dot(A20, T00, input_precision="ieee") + tl.dot(A21, T10, input_precision="ieee"), input_precision="ieee")
    T32 = -tl.dot(tl.dot(T33, A32, input_precision="ieee"), T22, input_precision="ieee")
    T31 = -tl.dot(T33, tl.dot(A31, T11, input_precision="ieee") + tl.dot(A32, T21, input_precision="ieee"), input_precision="ieee")
    T30 = -tl.dot(T33, tl.dot(A30, T00, input_precision="ieee") + tl.dot(A31, T10, input_precision="ieee") + tl.dot(A32, T20, input_precision="ieee"), input_precision="ieee")

    z = tl.zeros([16, 16], dtype=tl.float32)
    tl.store(Ai_out + base + (0 + r)[:, None] * 64 + (0 + c)[None, :], T00)
    tl.store(Ai_out + base + (0 + r)[:, None] * 64 + (16 + c)[None, :], z)
    tl.store(Ai_out + base + (0 + r)[:, None] * 64 + (32 + c)[None, :], z)
    tl.store(Ai_out + base + (0 + r)[:, None] * 64 + (48 + c)[None, :], z)

    tl.store(Ai_out + base + (16 + r)[:, None] * 64 + (0 + c)[None, :], T10)
    tl.store(Ai_out + base + (16 + r)[:, None] * 64 + (16 + c)[None, :], T11)
    tl.store(Ai_out + base + (16 + r)[:, None] * 64 + (32 + c)[None, :], z)
    tl.store(Ai_out + base + (16 + r)[:, None] * 64 + (48 + c)[None, :], z)

    tl.store(Ai_out + base + (32 + r)[:, None] * 64 + (0 + c)[None, :], T20)
    tl.store(Ai_out + base + (32 + r)[:, None] * 64 + (16 + c)[None, :], T21)
    tl.store(Ai_out + base + (32 + r)[:, None] * 64 + (32 + c)[None, :], T22)
    tl.store(Ai_out + base + (32 + r)[:, None] * 64 + (48 + c)[None, :], z)

    tl.store(Ai_out + base + (48 + r)[:, None] * 64 + (0 + c)[None, :], T30)
    tl.store(Ai_out + base + (48 + r)[:, None] * 64 + (16 + c)[None, :], T31)
    tl.store(Ai_out + base + (48 + r)[:, None] * 64 + (32 + c)[None, :], T32)
    tl.store(Ai_out + base + (48 + r)[:, None] * 64 + (48 + c)[None, :], T33)


@triton.jit
def _mdr_wy_u_fwd_kernel(
    Ai, V_in, U_out,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    iv = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_v = iv * BV + tl.arange(0, BV)
    ai_base = Ai + (i_bh * NT + i_t) * BT * BT
    v_base = V_in + (i_b * T * H + i_h) * V
    u_base = U_out + (i_b * T * H + i_h) * V

    for isub in range(2):
        o_r = isub * 32 + tl.arange(0, 32)
        t_r = i_t * BT + o_r
        b_u = tl.zeros([32, BV], dtype=tl.float32)
        for ic in range(2):
            o_c = ic * 32 + tl.arange(0, 32)
            t_c = i_t * BT + o_c
            b_Ai = tl.load(ai_base + o_r[:, None] * BT + o_c[None, :]).to(V_in.dtype.element_ty)
            b_v = tl.load(v_base + t_c[:, None] * (H * V) + o_v[None, :])
            b_u = tl.dot(b_Ai, b_v, b_u, input_precision=DOT_PREC)
        tl.store(u_base + t_r[:, None] * (H * V) + o_v[None, :], b_u.to(U_out.dtype.element_ty))


@triton.jit
def _mdr_wy_yz_fwd_kernel(
    Ai, P_eff, BarA_tm1, B_tm1, Y_out, Z_out,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    ik = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_i = tl.arange(0, BT)
    o_k = ik * BK + tl.arange(0, BK)
    t_r = i_t * BT + o_i

    ai_base = Ai + (i_bh * NT + i_t) * BT * BT
    p_base = P_eff + (i_b * T * H + i_h) * K
    vec_base = (i_bh * NT + i_t) * BT

    b_y = tl.zeros([BT, BK], dtype=tl.float32)
    b_z = tl.zeros([BT, BK], dtype=tl.float32)

    for ic in range(2):
        o_c = ic * 32 + tl.arange(0, 32)
        t_c = i_t * BT + o_c
        b_Ai = tl.load(ai_base + o_i[:, None] * BT + o_c[None, :]).to(P_eff.dtype.element_ty)
        b_p = tl.load(p_base + t_c[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
        b_bar_a = tl.load(BarA_tm1 + vec_base + o_c)
        b_btm1 = tl.load(B_tm1 + vec_base + o_c)
        b_pa = (b_p * b_bar_a[:, None]).to(P_eff.dtype.element_ty)
        b_pb = (b_p * b_btm1[:, None]).to(P_eff.dtype.element_ty)
        b_y = tl.dot(b_Ai, b_pa, b_y, input_precision=DOT_PREC)
        b_z = tl.dot(b_Ai, b_pb, b_z, input_precision=DOT_PREC)

    y_ptrs = Y_out + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + o_k[None, :]
    z_ptrs = Z_out + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + o_k[None, :]
    tl.store(y_ptrs, b_y.to(Y_out.dtype.element_ty))
    tl.store(z_ptrs, b_z.to(Z_out.dtype.element_ty))


@triton.jit
def _mdr_chunk_fwd_h_inreg_kernel(
    K_eff, U_c, Y_c, Z_c,
    DS_vec, DM_vec, A_last, B_last, M_last,
    S0, M0, S_pre_all, M_pre_all, V_new, ST, MT,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
):
    """Zero-spill BF16 Tensor Core in-register [K=64, BV=16] dual-state recurrence."""
    iv = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_k = tl.arange(0, 64)
    o_v = iv * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT)

    if HAS_INIT:
        s0_ptrs = S0 + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        m0_ptrs = M0 + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        b_S = tl.load(s0_ptrs).to(tl.float32)
        b_M = tl.load(m0_ptrs).to(tl.float32)
    else:
        b_S = tl.zeros([64, BV], dtype=tl.float32)
        b_M = tl.zeros([64, BV], dtype=tl.float32)

    for i_t in range(NT):
        st_off = (i_bh * NT + i_t) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(S_pre_all + st_off, b_S.to(S_pre_all.dtype.element_ty))
        tl.store(M_pre_all + st_off, b_M.to(M_pre_all.dtype.element_ty))

        t_idx = i_t * BT + o_i
        u_ptrs = U_c + (i_b * T * H + i_h) * V + t_idx[:, None] * (H * V) + o_v[None, :]
        y_ptrs = Y_c + (i_b * T * H + i_h) * K + t_idx[:, None] * (H * K) + o_k[None, :]
        z_ptrs = Z_c + (i_b * T * H + i_h) * K + t_idx[:, None] * (H * K) + o_k[None, :]

        b_u = tl.load(u_ptrs).to(tl.float32)
        b_y = tl.load(y_ptrs).to(tl.bfloat16)
        b_u = b_u - tl.dot(b_y, b_S.to(tl.bfloat16))
        b_z = tl.load(z_ptrs).to(tl.bfloat16)
        b_vn = b_u + tl.dot(b_z, b_M.to(tl.bfloat16))

        vn_ptrs = V_new + (i_b * T * H + i_h) * V + t_idx[:, None] * (H * V) + o_v[None, :]
        tl.store(vn_ptrs, b_vn.to(V_new.dtype.element_ty))

        sc_off = i_bh * NT + i_t
        a_last = tl.load(A_last + sc_off)
        b_last = tl.load(B_last + sc_off)
        m_last = tl.load(M_last + sc_off)

        vec_off = sc_off * BT + o_i
        b_ds = tl.load(DS_vec + vec_off)
        b_dm = tl.load(DM_vec + vec_off)

        k_ptrs = K_eff + (i_b * T * H + i_h) * K + t_idx[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(k_ptrs).to(tl.float32)
        b_vn_bf = b_vn.to(tl.bfloat16)

        b_ks_T = tl.trans((b_k * b_ds[:, None]).to(tl.bfloat16))
        b_S_new = a_last * b_S - b_last * b_M + tl.dot(b_ks_T, b_vn_bf)
        b_km_T = tl.trans((b_k * b_dm[:, None]).to(tl.bfloat16))
        b_M_new = m_last * b_M - tl.dot(b_km_T, b_vn_bf)
        b_S = b_S_new
        b_M = b_M_new

    if STORE_FINAL:
        st_ptrs = ST + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        mt_ptrs = MT + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(st_ptrs, b_S)
        tl.store(mt_ptrs, b_M)


@triton.jit
def _mdr_chunk_fwd_h_l2_kernel(
    K_eff, U_c, Y_c, Z_c,
    DS_vec, DM_vec, A_last, B_last, M_last,
    S0, M0, S_pre_all, M_pre_all, V_new, ST, MT,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    """Zero-spill L2-staged [BK=32, BV=16, BM=32] dual-state recurrence for arbitrary K, V, and FP32/BF16."""
    iv = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_k = tl.arange(0, BK)
    o_v = iv * BV + tl.arange(0, BV)

    for ik in range(tl.cdiv(K, BK)):
        offs_k = ik * BK + o_k
        st_off0 = (i_bh * NT + 0) * K * V + offs_k[:, None] * V + o_v[None, :]
        if HAS_INIT:
            s0_ptrs = S0 + i_bh * K * V + offs_k[:, None] * V + o_v[None, :]
            m0_ptrs = M0 + i_bh * K * V + offs_k[:, None] * V + o_v[None, :]
            tl.store(S_pre_all + st_off0, tl.load(s0_ptrs).to(S_pre_all.dtype.element_ty))
            tl.store(M_pre_all + st_off0, tl.load(m0_ptrs).to(M_pre_all.dtype.element_ty))
        else:
            z = tl.zeros([BK, BV], dtype=S_pre_all.dtype.element_ty)
            tl.store(S_pre_all + st_off0, z)
            tl.store(M_pre_all + st_off0, z)

    for i_t in range(NT):
        for isub in range(4):
            o_r = isub * 16 + tl.arange(0, 16)
            t_r = i_t * BT + o_r
            u_ptrs = U_c + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
            b_vn = tl.load(u_ptrs).to(tl.float32)

            for ik in range(tl.cdiv(K, BK)):
                offs_k = ik * BK + o_k
                st_off = (i_bh * NT + i_t) * K * V + offs_k[:, None] * V + o_v[None, :]
                b_S = tl.load(S_pre_all + st_off)
                y_ptrs = Y_c + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + offs_k[None, :]
                b_y = tl.load(y_ptrs)
                b_vn = b_vn - tl.dot(b_y, b_S, input_precision=DOT_PREC)

                b_M = tl.load(M_pre_all + st_off)
                z_ptrs = Z_c + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + offs_k[None, :]
                b_z = tl.load(z_ptrs)
                b_vn = tl.dot(b_z, b_M, b_vn, input_precision=DOT_PREC)

            vn_ptrs = V_new + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
            tl.store(vn_ptrs, b_vn.to(V_new.dtype.element_ty))

        sc_off = i_bh * NT + i_t
        a_last = tl.load(A_last + sc_off)
        b_last = tl.load(B_last + sc_off)
        m_last = tl.load(M_last + sc_off)

        for ik in range(tl.cdiv(K, BK)):
            offs_k = ik * BK + o_k
            st_off = (i_bh * NT + i_t) * K * V + offs_k[:, None] * V + o_v[None, :]
            b_S = tl.load(S_pre_all + st_off).to(tl.float32)
            b_M = tl.load(M_pre_all + st_off).to(tl.float32)
            b_S_new = a_last * b_S - b_last * b_M
            b_M_new = m_last * b_M
            for isub in range(4):
                o_r = isub * 16 + tl.arange(0, 16)
                t_r = i_t * BT + o_r
                vec_off = sc_off * BT + o_r
                b_ds = tl.load(DS_vec + vec_off)
                b_dm = tl.load(DM_vec + vec_off)
                k_ptrs = K_eff + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + offs_k[None, :]
                vn_ptrs = V_new + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
                b_k = tl.load(k_ptrs).to(tl.float32)
                b_vn_sub = tl.load(vn_ptrs)
                b_ks_T = tl.trans((b_k * b_ds[:, None]).to(V_new.dtype.element_ty))
                b_S_new = tl.dot(b_ks_T, b_vn_sub, b_S_new, input_precision=DOT_PREC)
                b_km_T = tl.trans((b_k * b_dm[:, None]).to(V_new.dtype.element_ty))
                b_M_new = b_M_new - tl.dot(b_km_T, b_vn_sub, input_precision=DOT_PREC)
            if i_t + 1 < NT:
                st_next = (i_bh * NT + (i_t + 1)) * K * V + offs_k[:, None] * V + o_v[None, :]
                tl.store(S_pre_all + st_next, b_S_new.to(S_pre_all.dtype.element_ty))
                tl.store(M_pre_all + st_next, b_M_new.to(M_pre_all.dtype.element_ty))
            elif STORE_FINAL:
                st_ptrs = ST + i_bh * K * V + offs_k[:, None] * V + o_v[None, :]
                mt_ptrs = MT + i_bh * K * V + offs_k[:, None] * V + o_v[None, :]
                tl.store(st_ptrs, b_S_new)
                tl.store(mt_ptrs, b_M_new)


@triton.jit
def _mdr_chunk_fwd_o_kernel(
    Q_eff, V_new, S_pre_all, M_pre_all,
    A_cum, B_t, Mqk_in, O_out,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    iv = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = iv * BV + tl.arange(0, BV)
    t_idx = i_t * BT + o_i

    vec_off = (i_bh * NT + i_t) * BT + o_i
    b_a_cum = tl.load(A_cum + vec_off)
    b_bt = tl.load(B_t + vec_off)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    for ik in range(tl.cdiv(K, BK)):
        offs_k = ik * BK + o_k
        q_ptrs = Q_eff + (i_b * T * H + i_h) * K + t_idx[:, None] * (H * K) + offs_k[None, :]
        b_q = tl.load(q_ptrs).to(tl.float32)

        st_off = (i_bh * NT + i_t) * K * V + offs_k[:, None] * V + o_v[None, :]
        b_S = tl.load(S_pre_all + st_off)
        b_qa = (b_q * b_a_cum[:, None]).to(b_S.dtype)
        b_o = tl.dot(b_qa, b_S, b_o, input_precision=DOT_PREC)

        b_M = tl.load(M_pre_all + st_off)
        b_qb = (b_q * b_bt[:, None]).to(b_M.dtype)
        b_o = b_o - tl.dot(b_qb, b_M, input_precision=DOT_PREC)

    mqk_base = Mqk_in + (i_bh * NT + i_t) * BT * BT
    vn_base = V_new + (i_b * T * H + i_h) * V
    for ic in range(2):
        o_c = ic * 32 + tl.arange(0, 32)
        t_c = i_t * BT + o_c
        b_Mqk = tl.load(mqk_base + o_i[:, None] * BT + o_c[None, :]).to(V_new.dtype.element_ty)
        b_vn = tl.load(vn_base + t_c[:, None] * (H * V) + o_v[None, :])
        b_o = tl.dot(b_Mqk, b_vn, b_o, input_precision=DOT_PREC)

    o_ptrs = O_out + (i_b * T * H + i_h) * V + t_idx[:, None] * (H * V) + o_v[None, :]
    tl.store(o_ptrs, b_o.to(O_out.dtype.element_ty))


# --------------------------------------------------------------------------------------------------
# Backward Kernels: Analytical Dual-State Adjoint + `dX = Ai^T @ dV_new` Factorization
# --------------------------------------------------------------------------------------------------

@triton.jit
def _mdr_bwd_dv_intra_kernel(
    Mqk_in, dO, dV_new_init,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    iv = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_v = iv * BV + tl.arange(0, BV)
    mqk_base = Mqk_in + (i_bh * NT + i_t) * BT * BT
    do_base = dO + (i_b * T * H + i_h) * V
    dvn_base = dV_new_init + (i_b * T * H + i_h) * V

    for isub in range(2):
        o_r = isub * 32 + tl.arange(0, 32)
        t_r = i_t * BT + o_r
        b_dvn = tl.zeros([32, BV], dtype=tl.float32)
        for ic in range(2):
            o_c = ic * 32 + tl.arange(0, 32)
            t_c = i_t * BT + o_c
            b_Mqk_T = tl.trans(tl.load(mqk_base + o_c[:, None] * BT + o_r[None, :])).to(dO.dtype.element_ty)
            b_do = tl.load(do_base + t_c[:, None] * (H * V) + o_v[None, :])
            b_dvn = tl.dot(b_Mqk_T, b_do, b_dvn, input_precision=DOT_PREC)
        tl.store(dvn_base + t_r[:, None] * (H * V) + o_v[None, :], b_dvn.to(dV_new_init.dtype.element_ty))


@triton.jit
def _mdr_chunk_bwd_dhu_inreg_kernel(
    Q_eff, K_eff, Y_c, Z_c, dO,
    A_cum, B_t, DS_vec, DM_vec, A_last, B_last, M_last,
    dST, dMT, dS_next_all, dM_next_all, dV_new, dS0, dM0,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    HAS_FINAL_GRAD: tl.constexpr,
    HAS_INIT_GRAD: tl.constexpr,
):
    """Zero-spill BF16 Tensor Core in-register [K=64, BV=16, BM=32] reverse dual-state recurrence."""
    iv = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_k = tl.arange(0, 64)
    o_v = iv * BV + tl.arange(0, BV)

    if HAS_FINAL_GRAD:
        dst_ptrs = dST + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        dmt_ptrs = dMT + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        b_dS = tl.load(dst_ptrs).to(tl.float32)
        b_dM = tl.load(dmt_ptrs).to(tl.float32)
    else:
        b_dS = tl.zeros([64, BV], dtype=tl.float32)
        b_dM = tl.zeros([64, BV], dtype=tl.float32)

    for i_step in range(NT):
        i_t = (NT - 1) - i_step
        st_off = (i_bh * NT + i_t) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(dS_next_all + st_off, b_dS.to(dS_next_all.dtype.element_ty))
        tl.store(dM_next_all + st_off, b_dM.to(dM_next_all.dtype.element_ty))

        sc_off = i_bh * NT + i_t
        a_last = tl.load(A_last + sc_off)
        b_last = tl.load(B_last + sc_off)
        m_last = tl.load(M_last + sc_off)

        b_dS_pre = a_last * b_dS
        b_dM_pre = m_last * b_dM - b_last * b_dS
        b_dS_bf = b_dS.to(tl.bfloat16)
        b_dM_bf = b_dM.to(tl.bfloat16)

        for isub in range(2):
            o_r = isub * 32 + tl.arange(0, 32)
            t_r = i_t * BT + o_r
            vec_off = sc_off * BT + o_r

            b_ds = tl.load(DS_vec + vec_off)
            b_dm = tl.load(DM_vec + vec_off)
            k_ptrs = K_eff + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + o_k[None, :]
            b_k = tl.load(k_ptrs).to(tl.float32)

            dvn_ptrs = dV_new + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
            b_dvn = tl.load(dvn_ptrs).to(tl.float32)
            b_ks = (b_k * b_ds[:, None]).to(tl.bfloat16)
            b_dvn = b_dvn + tl.dot(b_ks, b_dS_bf)
            b_km = (b_k * b_dm[:, None]).to(tl.bfloat16)
            b_dvn = b_dvn - tl.dot(b_km, b_dM_bf)
            tl.store(dvn_ptrs, b_dvn.to(dV_new.dtype.element_ty))

            b_a_cum = tl.load(A_cum + vec_off)
            b_bt = tl.load(B_t + vec_off)
            q_ptrs = Q_eff + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + o_k[None, :]
            y_ptrs = Y_c + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + o_k[None, :]
            z_ptrs = Z_c + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + o_k[None, :]
            do_ptrs = dO + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]

            b_q = tl.load(q_ptrs).to(tl.float32)
            b_do = tl.load(do_ptrs).to(tl.bfloat16)
            b_dvn_bf = b_dvn.to(tl.bfloat16)

            b_qa_T = tl.trans((b_q * b_a_cum[:, None]).to(tl.bfloat16))
            b_y_T = tl.trans(tl.load(y_ptrs).to(tl.bfloat16))
            b_dS_pre = b_dS_pre + tl.dot(b_qa_T, b_do) - tl.dot(b_y_T, b_dvn_bf)

            b_qb_T = tl.trans((b_q * b_bt[:, None]).to(tl.bfloat16))
            b_z_T = tl.trans(tl.load(z_ptrs).to(tl.bfloat16))
            b_dM_pre = b_dM_pre - tl.dot(b_qb_T, b_do) + tl.dot(b_z_T, b_dvn_bf)

        b_dS = b_dS_pre
        b_dM = b_dM_pre

    if HAS_INIT_GRAD:
        ds0_ptrs = dS0 + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        dm0_ptrs = dM0 + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(ds0_ptrs, b_dS)
        tl.store(dm0_ptrs, b_dM)


@triton.jit
def _mdr_chunk_bwd_dhu_l2_kernel(
    Q_eff, K_eff, Y_c, Z_c, dO,
    A_cum, B_t, DS_vec, DM_vec, A_last, B_last, M_last,
    dST, dMT, dS_next_all, dM_next_all, dV_new, dS0, dM0,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_FINAL_GRAD: tl.constexpr,
    HAS_INIT_GRAD: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    """Zero-spill L2-staged [BK=32, BV=16, BM=32] reverse dual-state recurrence for arbitrary K, V, and FP32/BF16."""
    iv = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_k = tl.arange(0, BK)
    o_v = iv * BV + tl.arange(0, BV)

    for ik in range(tl.cdiv(K, BK)):
        offs_k = ik * BK + o_k
        st_last = (i_bh * NT + (NT - 1)) * K * V + offs_k[:, None] * V + o_v[None, :]
        if HAS_FINAL_GRAD:
            dst_ptrs = dST + i_bh * K * V + offs_k[:, None] * V + o_v[None, :]
            dmt_ptrs = dMT + i_bh * K * V + offs_k[:, None] * V + o_v[None, :]
            tl.store(dS_next_all + st_last, tl.load(dst_ptrs).to(dS_next_all.dtype.element_ty))
            tl.store(dM_next_all + st_last, tl.load(dmt_ptrs).to(dM_next_all.dtype.element_ty))
        else:
            z = tl.zeros([BK, BV], dtype=dS_next_all.dtype.element_ty)
            tl.store(dS_next_all + st_last, z)
            tl.store(dM_next_all + st_last, z)

    for i_step in range(NT):
        i_t = (NT - 1) - i_step
        sc_off = i_bh * NT + i_t

        for isub in range(4):
            o_r = isub * 16 + tl.arange(0, 16)
            t_r = i_t * BT + o_r
            vec_off = sc_off * BT + o_r
            b_ds = tl.load(DS_vec + vec_off)
            b_dm = tl.load(DM_vec + vec_off)
            dvn_ptrs = dV_new + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
            b_dvn = tl.load(dvn_ptrs).to(tl.float32)

            for ik in range(tl.cdiv(K, BK)):
                offs_k = ik * BK + o_k
                st_off = (i_bh * NT + i_t) * K * V + offs_k[:, None] * V + o_v[None, :]
                k_ptrs = K_eff + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + offs_k[None, :]
                b_k = tl.load(k_ptrs).to(tl.float32)
                b_dS = tl.load(dS_next_all + st_off)
                b_ks = (b_k * b_ds[:, None]).to(b_dS.dtype)
                b_dvn = tl.dot(b_ks, b_dS, b_dvn, input_precision=DOT_PREC)

                b_dM = tl.load(dM_next_all + st_off)
                b_km = (b_k * b_dm[:, None]).to(b_dM.dtype)
                b_dvn = b_dvn - tl.dot(b_km, b_dM, input_precision=DOT_PREC)

            tl.store(dvn_ptrs, b_dvn.to(dV_new.dtype.element_ty))

        a_last = tl.load(A_last + sc_off)
        b_last = tl.load(B_last + sc_off)
        m_last = tl.load(M_last + sc_off)

        for ik in range(tl.cdiv(K, BK)):
            offs_k = ik * BK + o_k
            st_off = (i_bh * NT + i_t) * K * V + offs_k[:, None] * V + o_v[None, :]
            b_dS = tl.load(dS_next_all + st_off).to(tl.float32)
            b_dS_pre = a_last * b_dS

            for isub in range(4):
                o_r = isub * 16 + tl.arange(0, 16)
                t_r = i_t * BT + o_r
                vec_off = sc_off * BT + o_r
                b_a_cum = tl.load(A_cum + vec_off)
                q_ptrs = Q_eff + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + offs_k[None, :]
                y_ptrs = Y_c + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + offs_k[None, :]
                do_ptrs = dO + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
                dvn_ptrs = dV_new + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
                b_q = tl.load(q_ptrs).to(tl.float32)
                b_do = tl.load(do_ptrs)
                b_dvn_sub = tl.load(dvn_ptrs)
                b_qa_T = tl.trans((b_q * b_a_cum[:, None]).to(b_do.dtype))
                b_dS_pre = tl.dot(b_qa_T, b_do, b_dS_pre, input_precision=DOT_PREC)
                b_y_T = tl.trans(tl.load(y_ptrs))
                b_dS_pre = b_dS_pre - tl.dot(b_y_T, b_dvn_sub, input_precision=DOT_PREC)

            if i_t > 0:
                st_prev = (i_bh * NT + (i_t - 1)) * K * V + offs_k[:, None] * V + o_v[None, :]
                tl.store(dS_next_all + st_prev, b_dS_pre.to(dS_next_all.dtype.element_ty))
            elif HAS_INIT_GRAD:
                ds0_ptrs = dS0 + i_bh * K * V + offs_k[:, None] * V + o_v[None, :]
                tl.store(ds0_ptrs, b_dS_pre)

        for ik in range(tl.cdiv(K, BK)):
            offs_k = ik * BK + o_k
            st_off = (i_bh * NT + i_t) * K * V + offs_k[:, None] * V + o_v[None, :]
            b_dS = tl.load(dS_next_all + st_off).to(tl.float32)
            b_dM = tl.load(dM_next_all + st_off).to(tl.float32)
            b_dM_pre = m_last * b_dM - b_last * b_dS

            for isub in range(4):
                o_r = isub * 16 + tl.arange(0, 16)
                t_r = i_t * BT + o_r
                vec_off = sc_off * BT + o_r
                b_bt = tl.load(B_t + vec_off)
                q_ptrs = Q_eff + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + offs_k[None, :]
                z_ptrs = Z_c + (i_b * T * H + i_h) * K + t_r[:, None] * (H * K) + offs_k[None, :]
                do_ptrs = dO + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
                dvn_ptrs = dV_new + (i_b * T * H + i_h) * V + t_r[:, None] * (H * V) + o_v[None, :]
                b_q = tl.load(q_ptrs).to(tl.float32)
                b_do = tl.load(do_ptrs)
                b_dvn_sub = tl.load(dvn_ptrs)
                b_qb_T = tl.trans((b_q * b_bt[:, None]).to(b_do.dtype))
                b_dM_pre = b_dM_pre - tl.dot(b_qb_T, b_do, input_precision=DOT_PREC)
                b_z_T = tl.trans(tl.load(z_ptrs))
                b_dM_pre = tl.dot(b_z_T, b_dvn_sub, b_dM_pre, input_precision=DOT_PREC)

            if i_t > 0:
                st_prev = (i_bh * NT + (i_t - 1)) * K * V + offs_k[:, None] * V + o_v[None, :]
                tl.store(dM_next_all + st_prev, b_dM_pre.to(dM_next_all.dtype.element_ty))
            elif HAS_INIT_GRAD:
                dm0_ptrs = dM0 + i_bh * K * V + offs_k[:, None] * V + o_v[None, :]
                tl.store(dm0_ptrs, b_dM_pre)


@triton.jit
def _mdr_bwd_dx_dv_kernel(
    Ai, dV_new, dX_out,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    iv = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_v = iv * BV + tl.arange(0, BV)
    ai_base = Ai + (i_bh * NT + i_t) * BT * BT
    dvn_base = dV_new + (i_b * T * H + i_h) * V
    dx_base = dX_out + (i_b * T * H + i_h) * V

    for isub in range(2):
        o_r = isub * 32 + tl.arange(0, 32)
        t_r = i_t * BT + o_r
        b_dx = tl.zeros([32, BV], dtype=tl.float32)
        for ic in range(2):
            o_c = ic * 32 + tl.arange(0, 32)
            t_c = i_t * BT + o_c
            b_Ai_T = tl.trans(tl.load(ai_base + o_c[:, None] * BT + o_r[None, :])).to(dV_new.dtype.element_ty)
            b_dvn = tl.load(dvn_base + t_c[:, None] * (H * V) + o_v[None, :])
            b_dx = tl.dot(b_Ai_T, b_dvn, b_dx, input_precision=DOT_PREC)
        tl.store(dx_base + t_r[:, None] * (H * V) + o_v[None, :], b_dx.to(dX_out.dtype.element_ty))


@triton.jit
def _mdr_bwd_dA_dMqk_kernel(
    dX, dO, V_new, Q_eff, K_eff, P_eff,
    dA_out, dMqk_out, dGq_raw_out, dG_raw_out,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    """Row-subtiled (BM=16, BK=32, BV=32) backward score builder for dA, dMqk, dGq_raw, and dG_raw."""
    i_tr = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NSUB: tl.constexpr = BT // BM
    i_t = i_tr // NSUB
    i_sub = i_tr % NSUB
    NT = T // BT

    o_r = i_sub * BM + tl.arange(0, BM)
    o_c = tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    t_r = i_t * BT + o_r
    t_c = i_t * BT + o_c

    b_dA = tl.zeros([BM, BT], dtype=tl.float32)
    b_dMqk = tl.zeros([BM, BT], dtype=tl.float32)
    base_v = (i_b * T * H + i_h) * V

    for iv in range(tl.cdiv(V, BV)):
        offs_v = iv * BV + o_v
        b_dx = tl.load(dX + base_v + t_r[:, None] * (H * V) + offs_v[None, :])
        b_do = tl.load(dO + base_v + t_r[:, None] * (H * V) + offs_v[None, :])
        b_vn_T = tl.trans(tl.load(V_new + base_v + t_c[:, None] * (H * V) + offs_v[None, :]))
        b_dA = tl.dot(b_dx, b_vn_T, b_dA, input_precision=DOT_PREC)
        b_dMqk = tl.dot(b_do, b_vn_T, b_dMqk, input_precision=DOT_PREC)

    m_strict = o_r[:, None] > o_c[None, :]
    m_tril = o_r[:, None] >= o_c[None, :]
    b_dA = tl.where(m_strict, -b_dA, 0.0)
    b_dMqk = tl.where(m_tril, b_dMqk, 0.0)

    out_off = (i_bh * NT + i_t) * BT * BT + o_r[:, None] * BT + o_c[None, :]
    tl.store(dA_out + out_off, b_dA)
    tl.store(dMqk_out + out_off, b_dMqk)

    b_qk = tl.zeros([BM, BT], dtype=tl.float32)
    b_pk = tl.zeros([BM, BT], dtype=tl.float32)
    base_k = (i_b * T * H + i_h) * K

    for ik in range(tl.cdiv(K, BK)):
        offs_k = ik * BK + o_k
        b_q = tl.load(Q_eff + base_k + t_r[:, None] * (H * K) + offs_k[None, :])
        b_p = tl.load(P_eff + base_k + t_r[:, None] * (H * K) + offs_k[None, :])
        b_k_T = tl.trans(tl.load(K_eff + base_k + t_c[:, None] * (H * K) + offs_k[None, :]))
        b_qk = tl.dot(b_q, b_k_T, b_qk, input_precision=DOT_PREC)
        b_pk = tl.dot(b_p, b_k_T, b_pk, input_precision=DOT_PREC)

    tl.store(dGq_raw_out + out_off, tl.where(m_tril, b_dMqk * b_qk, 0.0))
    tl.store(dG_raw_out + out_off, tl.where(m_strict, b_dA * b_pk, 0.0))


@triton.jit
def _mdr_bwd_dq_dk_dp_inter_kernel(
    Q_eff, K_eff, P_eff, V_new, dO, dX,
    S_pre_all, M_pre_all, dS_next_all, dM_next_all,
    A_cum, B_t, BarA_tm1, B_tm1, DS_vec, DM_vec,
    Gamma_q, dA_in, dMqk_in,
    dQ_eff, dK_eff, dP_eff,
    dA_cum_part, dB_t_part, dK_vds_part, dK_vdm_part, dState_sc_part,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    """Sequential-staged (BM=16, BK=16, BV=16) inter/intra backward kernel (`<= 96` regs, `0` spills)."""
    ik = tl.program_id(0).to(tl.int64)
    i_tr = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NSUB: tl.constexpr = BT // BM
    i_t = i_tr // NSUB
    i_sub = i_tr % NSUB
    NT = T // BT
    NK = K // BK

    o_r = i_sub * BM + tl.arange(0, BM)
    o_k = ik * BK + tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    t_r = i_t * BT + o_r
    r_prev = tl.maximum(o_r - 1, 0)

    base_v = (i_b * T * H + i_h) * V
    base_k = (i_b * T * H + i_h) * K
    st_base = (i_bh * NT + i_t) * K * V
    mat_base = (i_bh * NT + i_t) * BT * BT
    vec_off = (i_bh * NT + i_t) * BT + o_r
    k_ptrs = base_k + t_r[:, None] * (H * K) + o_k[None, :]
    part_vec2_off = ((i_bh * NT + i_t) * NK + ik) * (2 * BT) + o_r
    part_vec_off = ((i_bh * NT + i_t) * NK + ik) * BT + o_r

    # Stage 1: dQ_eff and (q_qs, q_qm)
    b_qS = tl.zeros([BM, BK], dtype=tl.float32)
    b_qM = tl.zeros([BM, BK], dtype=tl.float32)
    for iv in range(tl.cdiv(V, BV)):
        offs_v = iv * BV + o_v
        b_do = tl.load(dO + base_v + t_r[:, None] * (H * V) + offs_v[None, :])
        st_ptrs = st_base + o_k[:, None] * V + offs_v[None, :]
        b_S_T = tl.trans(tl.load(S_pre_all + st_ptrs))
        b_M_T = tl.trans(tl.load(M_pre_all + st_ptrs))
        b_qS = tl.dot(b_do, b_S_T, b_qS, input_precision=DOT_PREC)
        b_qM = tl.dot(b_do, b_M_T, b_qM, input_precision=DOT_PREC)

    b_q = tl.load(Q_eff + k_ptrs).to(tl.float32)
    tl.store(dA_cum_part + part_vec2_off, tl.sum(b_q * b_qS, axis=1))
    tl.store(dB_t_part + part_vec2_off, -tl.sum(b_q * b_qM, axis=1))

    b_a_cum = tl.load(A_cum + vec_off)
    b_bt = tl.load(B_t + vec_off)
    b_dq = b_a_cum[:, None] * b_qS - b_bt[:, None] * b_qM
    for ic in range(4):
        o_c = ic * 16 + tl.arange(0, 16)
        t_c = i_t * BT + o_c
        b_Gq_rc = tl.load(Gamma_q + mat_base + o_r[:, None] * BT + o_c[None, :])
        b_dMqk_rc = (tl.load(dMqk_in + mat_base + o_r[:, None] * BT + o_c[None, :]) * b_Gq_rc).to(Q_eff.dtype.element_ty)
        b_kc = tl.load(K_eff + base_k + t_c[:, None] * (H * K) + o_k[None, :])
        b_dq = tl.dot(b_dMqk_rc, b_kc, b_dq, input_precision=DOT_PREC)
    tl.store(dQ_eff + k_ptrs, b_dq.to(dQ_eff.dtype.element_ty))

    # Stage 2: dP_eff and (p_dpy, p_dpz)
    b_dPy = tl.zeros([BM, BK], dtype=tl.float32)
    b_dPz = tl.zeros([BM, BK], dtype=tl.float32)
    for iv in range(tl.cdiv(V, BV)):
        offs_v = iv * BV + o_v
        b_dx = tl.load(dX + base_v + t_r[:, None] * (H * V) + offs_v[None, :])
        st_ptrs = st_base + o_k[:, None] * V + offs_v[None, :]
        b_S_T = tl.trans(tl.load(S_pre_all + st_ptrs))
        b_M_T = tl.trans(tl.load(M_pre_all + st_ptrs))
        b_dPy = tl.dot(b_dx, b_S_T, b_dPy, input_precision=DOT_PREC)
        b_dPz = tl.dot(b_dx, b_M_T, b_dPz, input_precision=DOT_PREC)
    b_dPy = -b_dPy

    b_p = tl.load(P_eff + k_ptrs).to(tl.float32)
    tl.store(dA_cum_part + part_vec2_off + BT, tl.sum(b_p * b_dPy, axis=1))
    tl.store(dB_t_part + part_vec2_off + BT, tl.sum(b_p * b_dPz, axis=1))

    b_bar_a = tl.load(BarA_tm1 + vec_off)
    b_btm1 = tl.load(B_tm1 + vec_off)
    b_dp = b_bar_a[:, None] * b_dPy + b_btm1[:, None] * b_dPz
    for ic in range(4):
        o_c = ic * 16 + tl.arange(0, 16)
        t_c = i_t * BT + o_c
        b_G_rc = tl.where(o_r[:, None] > o_c[None, :], tl.load(Gamma_q + mat_base + r_prev[:, None] * BT + o_c[None, :]), 0.0)
        b_dA_rc = (tl.load(dA_in + mat_base + o_r[:, None] * BT + o_c[None, :]) * b_G_rc).to(Q_eff.dtype.element_ty)
        b_kc = tl.load(K_eff + base_k + t_c[:, None] * (H * K) + o_k[None, :])
        b_dp = tl.dot(b_dA_rc, b_kc, b_dp, input_precision=DOT_PREC)
    tl.store(dP_eff + k_ptrs, b_dp.to(dP_eff.dtype.element_ty))

    # Stage 3: dK_eff and (kvds_vec, kvdm_vec, state scalars)
    b_vdS = tl.zeros([BM, BK], dtype=tl.float32)
    b_vdM = tl.zeros([BM, BK], dtype=tl.float32)
    b_s_ds = 0.0
    b_m_ds = 0.0
    b_m_dm = 0.0
    for iv in range(tl.cdiv(V, BV)):
        offs_v = iv * BV + o_v
        b_vn = tl.load(V_new + base_v + t_r[:, None] * (H * V) + offs_v[None, :])
        st_ptrs = st_base + o_k[:, None] * V + offs_v[None, :]
        b_dS = tl.load(dS_next_all + st_ptrs)
        b_dM = tl.load(dM_next_all + st_ptrs)
        b_vdS = tl.dot(b_vn, tl.trans(b_dS), b_vdS, input_precision=DOT_PREC)
        b_vdM = tl.dot(b_vn, tl.trans(b_dM), b_vdM, input_precision=DOT_PREC)
        if i_sub == 0:
            b_Sf = tl.load(S_pre_all + st_ptrs).to(tl.float32)
            b_Mf = tl.load(M_pre_all + st_ptrs).to(tl.float32)
            b_dSf = b_dS.to(tl.float32)
            b_dMf = b_dM.to(tl.float32)
            b_s_ds = b_s_ds + tl.sum(b_Sf * b_dSf)
            b_m_ds = b_m_ds + tl.sum(b_Mf * b_dSf)
            b_m_dm = b_m_dm + tl.sum(b_Mf * b_dMf)

    b_k = tl.load(K_eff + k_ptrs).to(tl.float32)
    tl.store(dK_vds_part + part_vec_off, tl.sum(b_k * b_vdS, axis=1))
    tl.store(dK_vdm_part + part_vec_off, tl.sum(b_k * b_vdM, axis=1))
    if i_sub == 0:
        part_sc_off = ((i_bh * NT + i_t) * NK + ik) * 3
        tl.store(dState_sc_part + part_sc_off + 0, b_s_ds)
        tl.store(dState_sc_part + part_sc_off + 1, b_m_ds)
        tl.store(dState_sc_part + part_sc_off + 2, b_m_dm)

    b_ds = tl.load(DS_vec + vec_off)
    b_dm = tl.load(DM_vec + vec_off)
    b_dk = b_ds[:, None] * b_vdS - b_dm[:, None] * b_vdM
    for ic in range(4):
        o_c = ic * 16 + tl.arange(0, 16)
        t_c = i_t * BT + o_c
        c_prev = tl.maximum(o_c - 1, 0)
        b_Gq_cr = tl.load(Gamma_q + mat_base + o_c[:, None] * BT + o_r[None, :])
        b_G_cr = tl.where(o_c[:, None] > o_r[None, :], tl.load(Gamma_q + mat_base + c_prev[:, None] * BT + o_r[None, :]), 0.0)
        b_dMqk_cr_T = tl.trans((tl.load(dMqk_in + mat_base + o_c[:, None] * BT + o_r[None, :]) * b_Gq_cr).to(Q_eff.dtype.element_ty))
        b_dA_cr_T = tl.trans((tl.load(dA_in + mat_base + o_c[:, None] * BT + o_r[None, :]) * b_G_cr).to(Q_eff.dtype.element_ty))
        b_qc = tl.load(Q_eff + base_k + t_c[:, None] * (H * K) + o_k[None, :])
        b_pc = tl.load(P_eff + base_k + t_c[:, None] * (H * K) + o_k[None, :])
        b_dk = tl.dot(b_dMqk_cr_T, b_qc, b_dk, input_precision=DOT_PREC)
        b_dk = tl.dot(b_dA_cr_T, b_pc, b_dk, input_precision=DOT_PREC)
    tl.store(dK_eff + k_ptrs, b_dk.to(dK_eff.dtype.element_ty))


@triton.jit
def _mdr_bwd_gate_greens_kernel(
    LogAlpha, LogMu, Beta,
    LA_cum, LM_cum, A_cum, M_cum,
    dGq_raw_in, dG_raw_in,
    dA_cum_part, dB_t_part, dK_vds_part, dK_vdm_part, dState_sc_part,
    dLA_intra, dLM_out, dBeta_out,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    NK: tl.constexpr,
):
    """
    Exact O(BT) 1D Vector-Recurrence Gate Adjoint Kernel (`0` 2D [64, 64] matrices in registers!):
      1. Pass 0: Assemble exact lower-triangular `dGq[i, j]` in `dGq_raw_in` and 1D `da_cum, db_t`.
      2. Pass 1 (Forward k = 0 .. BT-1):
           acc_dGq_Dm[i] = mu_k * acc_dGq_Dm[i] + dGq[i, k]
           dDa_col_k[i]  = (i >= k) ? (beta_k * acc_dGq_Dm[i] + db_t[i] * beta_k * m_cum_k) : 0
           Ga_col_k[i]   = (i > k)  ? (dDa_col_k[i] * exp(la_cum[i] - la_cum[k])) : 0
      3. Pass 2 (Backward k = BT-1 .. 0):
           rev_acc[j]    = dGq[k, j] + alpha_{k+1} * rev_acc[j]
           rev_db        = db_t[k]   + alpha_{k+1} * rev_db
           d_beta_Dm_k[j]= (j <= k)  ? rev_acc[j] : 0
           dbeta[k]      = sum_j(d_beta_Dm_k[j] * Dm[k, j]) + rev_db * m_cum_k
           dDm_row_k[j]  = (j <= k)  ? (beta_k * d_beta_Dm_k[j] - (kvdm[j] if k == BT-1 else 0)) : 0
           Gm_row_k[j]   = (j < k)   ? (dDm_row_k[j] * Dm[k, j]) : 0
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = T // BT

    o_j = tl.arange(0, BT)
    t_idx = i_t * BT + o_j

    q_qs = tl.zeros([BT], dtype=tl.float32)
    p_dpy = tl.zeros([BT], dtype=tl.float32)
    q_qm = tl.zeros([BT], dtype=tl.float32)
    p_dpz = tl.zeros([BT], dtype=tl.float32)
    kvds = tl.zeros([BT], dtype=tl.float32)
    kvdm = tl.zeros([BT], dtype=tl.float32)
    s_ds = 0.0
    m_ds = 0.0
    m_dm = 0.0

    for ik in range(NK):
        part_vec2_off = ((i_bh * NT + i_t) * NK + ik) * (2 * BT) + o_j
        q_qs = q_qs + tl.load(dA_cum_part + part_vec2_off)
        p_dpy = p_dpy + tl.load(dA_cum_part + part_vec2_off + BT)
        q_qm = q_qm + tl.load(dB_t_part + part_vec2_off)
        p_dpz = p_dpz + tl.load(dB_t_part + part_vec2_off + BT)

        part_vec_off = ((i_bh * NT + i_t) * NK + ik) * BT + o_j
        kvds = kvds + tl.load(dK_vds_part + part_vec_off)
        kvdm = kvdm + tl.load(dK_vdm_part + part_vec_off)

        part_sc_off = ((i_bh * NT + i_t) * NK + ik) * 3
        s_ds = s_ds + tl.load(dState_sc_part + part_sc_off + 0)
        m_ds = m_ds + tl.load(dState_sc_part + part_sc_off + 1)
        m_dm = m_dm + tl.load(dState_sc_part + part_sc_off + 2)

    vec_base = (i_bh * NT + i_t) * BT
    b_la_cum = tl.load(LA_cum + vec_base + o_j)
    b_lm_cum = tl.load(LM_cum + vec_base + o_j)
    b_a_cum = tl.load(A_cum + vec_base + o_j)
    b_m_cum = tl.load(M_cum + vec_base + o_j)

    in_off = (i_b * T + t_idx) * H + i_h
    b_beta = tl.load(Beta + in_off).to(tl.float32)

    mat_base = (i_bh * NT + i_t) * BT * BT
    da_cum = tl.zeros([BT], dtype=tl.float32)
    db_t = tl.zeros([BT], dtype=tl.float32)

    # Pass 0: Assemble dGq in dGq_raw_in and build 1D da_cum, db_t
    for i in range(BT):
        m_i = o_j == i
        m_ip1 = o_j == (i + 1)
        q_qs_i = tl.sum(tl.where(m_i, q_qs, 0.0), axis=0)
        q_qm_i = tl.sum(tl.where(m_i, q_qm, 0.0), axis=0)
        p_dpy_ip1 = tl.sum(tl.where(m_ip1, p_dpy, 0.0), axis=0) if i < (BT - 1) else s_ds
        p_dpz_ip1 = tl.sum(tl.where(m_ip1, p_dpz, 0.0), axis=0) if i < (BT - 1) else -m_ds
        da_cum = tl.where(m_i, q_qs_i + p_dpy_ip1, da_cum)
        db_t = tl.where(m_i, q_qm_i + p_dpz_ip1, db_t)

        dGq_i = tl.load(dGq_raw_in + mat_base + i * BT + o_j)
        if i < (BT - 1):
            dG_ip1 = tl.load(dG_raw_in + mat_base + (i + 1) * BT + o_j)
            dGq_i = dGq_i + dG_ip1
        else:
            dGq_i = dGq_i + kvds
        dGq_i = tl.where(o_j <= i, dGq_i, 0.0)
        tl.store(dGq_raw_in + mat_base + i * BT + o_j, dGq_i)

    # Pass 1: Forward recurrence k = 0 .. BT-1 for dDa and Ga = strictTril(dDa * Da)
    acc_dGq_Dm = tl.zeros([BT], dtype=tl.float32)
    ga_row_sum = tl.zeros([BT], dtype=tl.float32)
    ga_col_sum = tl.zeros([BT], dtype=tl.float32)

    for k in range(BT):
        t_k = i_t * BT + k
        in_k = (i_b * T + t_k) * H + i_h
        lm_k_step = tl.load(LogMu + in_k).to(tl.float32)
        mu_k = tl.exp(lm_k_step)
        beta_k = tl.load(Beta + in_k).to(tl.float32)
        la_cum_k = tl.load(LA_cum + vec_base + k)
        m_cum_k = tl.load(M_cum + vec_base + k)

        dGq_col_k = tl.load(dGq_raw_in + mat_base + o_j * BT + k)
        acc_dGq_Dm = mu_k * acc_dGq_Dm + dGq_col_k

        dDa_col_k = tl.where(o_j >= k, beta_k * acc_dGq_Dm + db_t * (beta_k * m_cum_k), 0.0)
        Da_col_k = tl.where(o_j >= k, tl.exp(b_la_cum - la_cum_k), 0.0)
        Ga_col_k = tl.where(o_j > k, dDa_col_k * Da_col_k, 0.0)

        ga_row_sum = ga_row_sum + Ga_col_k
        ga_col_k_sum = tl.sum(Ga_col_k, axis=0)
        ga_col_sum = tl.where(o_j == k, ga_col_k_sum, ga_col_sum)

    # Pass 2: Backward recurrence k = BT-1 .. 0 for d_beta_Dm, dbeta, dDm, and Gm = strictTril(dDm * Dm)
    rev_acc = tl.zeros([BT], dtype=tl.float32)
    rev_db = 0.0
    alpha_next = 0.0
    gm_row_sum = tl.zeros([BT], dtype=tl.float32)
    gm_col_sum = tl.zeros([BT], dtype=tl.float32)
    dm_cum = tl.zeros([BT], dtype=tl.float32)
    dbeta_vec = tl.zeros([BT], dtype=tl.float32)

    for step in range(BT):
        k = (BT - 1) - step
        t_k = i_t * BT + k
        in_k = (i_b * T + t_k) * H + i_h
        la_k_step = tl.load(LogAlpha + in_k).to(tl.float32)
        beta_k = tl.load(Beta + in_k).to(tl.float32)
        lm_cum_k = tl.load(LM_cum + vec_base + k)
        m_cum_k = tl.load(M_cum + vec_base + k)

        dGq_row_k = tl.load(dGq_raw_in + mat_base + k * BT + o_j)
        db_k = tl.sum(tl.where(o_j == k, db_t, 0.0), axis=0)

        rev_acc = dGq_row_k + alpha_next * rev_acc
        rev_db = db_k + alpha_next * rev_db
        alpha_next = tl.exp(la_k_step)

        d_beta_Dm_k = tl.where(o_j <= k, rev_acc, 0.0)
        Dm_row_k = tl.where(o_j <= k, tl.exp(lm_cum_k - b_lm_cum), 0.0)

        dbeta_k = tl.sum(d_beta_Dm_k * Dm_row_k, axis=0) + rev_db * m_cum_k
        dbeta_vec = tl.where(o_j == k, dbeta_k, dbeta_vec)

        dm_cum_k = rev_db * beta_k + tl.where(k == (BT - 1), m_dm, 0.0)
        dm_cum = tl.where(o_j == k, dm_cum_k, dm_cum)

        dDm_row_k = beta_k * d_beta_Dm_k - tl.where(k == (BT - 1), kvdm, 0.0)
        Gm_row_k = tl.where(o_j < k, dDm_row_k * Dm_row_k, 0.0)

        gm_row_k_sum = tl.sum(Gm_row_k, axis=0)
        gm_row_sum = tl.where(o_j == k, gm_row_k_sum, gm_row_sum)
        gm_col_sum = gm_col_sum + Gm_row_k

    dla_cum = b_a_cum * da_cum + ga_row_sum - ga_col_sum
    dlm_cum = b_m_cum * dm_cum + gm_row_sum - gm_col_sum

    # Reverse cumulative sum in 1D: sum(x) - cumsum(x) + x
    b_dla = tl.sum(dla_cum, axis=0) - tl.cumsum(dla_cum, axis=0) + dla_cum
    b_dlm = tl.sum(dlm_cum, axis=0) - tl.cumsum(dlm_cum, axis=0) + dlm_cum

    tl.store(dLA_intra + in_off, b_dla)
    tl.store(dLM_out + in_off, b_dlm.to(dLM_out.dtype.element_ty))
    tl.store(dBeta_out + in_off, dbeta_vec.to(dBeta_out.dtype.element_ty))


# ==================================================================================================
# 3. Superfused MomentumDeltaNetBlockRSI Kernels (ShortConv1d+SiLU, Symplectic 4-Gates, RMSNormGated)
# ==================================================================================================

@triton.jit
def _mdr_fused_shortconv_silu_fwd_kernel(
    X, W, Y, Z_pre,
    B, T, D,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    pid_d = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)

    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_t = pid_t * BT + tl.arange(0, BT)
    mask_d = offs_d < D
    mask_t = offs_t < T

    w0 = tl.load(W + offs_d * 4 + 0, mask=mask_d, other=0.0).to(tl.float32)
    w1 = tl.load(W + offs_d * 4 + 1, mask=mask_d, other=0.0).to(tl.float32)
    w2 = tl.load(W + offs_d * 4 + 2, mask=mask_d, other=0.0).to(tl.float32)
    w3 = tl.load(W + offs_d * 4 + 3, mask=mask_d, other=0.0).to(tl.float32)

    base_x = X + pid_b * T * D
    t0 = offs_t - 3
    t1 = offs_t - 2
    t2 = offs_t - 1
    t3 = offs_t

    x0 = tl.load(base_x + t0[:, None] * D + offs_d[None, :], mask=(t0[:, None] >= 0) & mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    x1 = tl.load(base_x + t1[:, None] * D + offs_d[None, :], mask=(t1[:, None] >= 0) & mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    x2 = tl.load(base_x + t2[:, None] * D + offs_d[None, :], mask=(t2[:, None] >= 0) & mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    x3 = tl.load(base_x + t3[:, None] * D + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    z = x0 * w0[None, :] + x1 * w1[None, :] + x2 * w2[None, :] + x3 * w3[None, :]
    sig = tl.sigmoid(z)
    y = z * sig

    out_off = pid_b * T * D + offs_t[:, None] * D + offs_d[None, :]
    mask_out = mask_t[:, None] & mask_d[None, :]
    tl.store(Z_pre + out_off, z, mask=mask_out)
    tl.store(Y + out_off, y.to(Y.dtype.element_ty), mask=mask_out)


@triton.jit
def _mdr_fused_shortconv_silu_bwd_dx_kernel(
    dY, Z_pre, W, dX, dZ_out,
    B, T, D,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    pid_d = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)

    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_t = pid_t * BT + tl.arange(0, BT)
    mask_d = offs_d < D
    mask_t = offs_t < T

    w0 = tl.load(W + offs_d * 4 + 0, mask=mask_d, other=0.0).to(tl.float32)
    w1 = tl.load(W + offs_d * 4 + 1, mask=mask_d, other=0.0).to(tl.float32)
    w2 = tl.load(W + offs_d * 4 + 2, mask=mask_d, other=0.0).to(tl.float32)
    w3 = tl.load(W + offs_d * 4 + 3, mask=mask_d, other=0.0).to(tl.float32)

    base = pid_b * T * D
    t0 = offs_t
    t1 = offs_t + 1
    t2 = offs_t + 2
    t3 = offs_t + 3

    m0 = (t0[:, None] < T) & mask_d[None, :]
    m1 = (t1[:, None] < T) & mask_d[None, :]
    m2 = (t2[:, None] < T) & mask_d[None, :]
    m3 = (t3[:, None] < T) & mask_d[None, :]

    dy0 = tl.load(dY + base + t0[:, None] * D + offs_d[None, :], mask=m0, other=0.0).to(tl.float32)
    z0 = tl.load(Z_pre + base + t0[:, None] * D + offs_d[None, :], mask=m0, other=0.0).to(tl.float32)
    s0 = tl.sigmoid(z0)
    dz0 = dy0 * s0 * (1.0 + z0 * (1.0 - s0))

    dy1 = tl.load(dY + base + t1[:, None] * D + offs_d[None, :], mask=m1, other=0.0).to(tl.float32)
    z1 = tl.load(Z_pre + base + t1[:, None] * D + offs_d[None, :], mask=m1, other=0.0).to(tl.float32)
    s1 = tl.sigmoid(z1)
    dz1 = dy1 * s1 * (1.0 + z1 * (1.0 - s1))

    dy2 = tl.load(dY + base + t2[:, None] * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
    z2 = tl.load(Z_pre + base + t2[:, None] * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
    s2 = tl.sigmoid(z2)
    dz2 = dy2 * s2 * (1.0 + z2 * (1.0 - s2))

    dy3 = tl.load(dY + base + t3[:, None] * D + offs_d[None, :], mask=m3, other=0.0).to(tl.float32)
    z3 = tl.load(Z_pre + base + t3[:, None] * D + offs_d[None, :], mask=m3, other=0.0).to(tl.float32)
    s3 = tl.sigmoid(z3)
    dz3 = dy3 * s3 * (1.0 + z3 * (1.0 - s3))

    dx = dz0 * w3[None, :] + dz1 * w2[None, :] + dz2 * w1[None, :] + dz3 * w0[None, :]
    out_off = base + t0[:, None] * D + offs_d[None, :]
    tl.store(dZ_out + out_off, dz0, mask=m0)
    tl.store(dX + out_off, dx.to(dX.dtype.element_ty), mask=m0)


@triton.jit
def _mdr_fused_shortconv_silu_bwd_dw_kernel(
    dZ, X, dW,
    B, T, D,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    pid_d = tl.program_id(0).to(tl.int64)
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    offs_t = tl.arange(0, BT)

    dw0 = tl.zeros([BD], dtype=tl.float32)
    dw1 = tl.zeros([BD], dtype=tl.float32)
    dw2 = tl.zeros([BD], dtype=tl.float32)
    dw3 = tl.zeros([BD], dtype=tl.float32)

    for ib in range(B):
        base = ib * T * D
        for it in range(0, T, BT):
            t_cur = it + offs_t
            m_t = t_cur < T
            dz = tl.load(dZ + base + t_cur[:, None] * D + offs_d[None, :], mask=m_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            t0 = t_cur - 3
            t1 = t_cur - 2
            t2 = t_cur - 1
            x0 = tl.load(X + base + t0[:, None] * D + offs_d[None, :], mask=(t0[:, None] >= 0) & m_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            x1 = tl.load(X + base + t1[:, None] * D + offs_d[None, :], mask=(t1[:, None] >= 0) & m_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            x2 = tl.load(X + base + t2[:, None] * D + offs_d[None, :], mask=(t2[:, None] >= 0) & m_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            x3 = tl.load(X + base + t_cur[:, None] * D + offs_d[None, :], mask=m_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            dw0 += tl.sum(dz * x0, axis=0)
            dw1 += tl.sum(dz * x1, axis=0)
            dw2 += tl.sum(dz * x2, axis=0)
            dw3 += tl.sum(dz * x3, axis=0)

    tl.store(dW + offs_d * 4 + 0, dw0.to(dW.dtype.element_ty), mask=mask_d)
    tl.store(dW + offs_d * 4 + 1, dw1.to(dW.dtype.element_ty), mask=mask_d)
    tl.store(dW + offs_d * 4 + 2, dw2.to(dW.dtype.element_ty), mask=mask_d)
    tl.store(dW + offs_d * 4 + 3, dw3.to(dW.dtype.element_ty), mask=mask_d)


@triton.jit
def _mdr_fused_symplectic_gates_fwd_kernel(
    ABME, A_log, Dt_bias, Mu_log, Mu_bias, Log_factor,
    LogAlpha_out, LogMu_out, Beta_out, Eta_out,
    inv_tau, min_log_mu,
    H: tl.constexpr,
    NEG_EIG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, H)

    base = row * (4 * H)
    a_val = tl.load(ABME + base + 0 * H + offs_h).to(tl.float32)
    b_val = tl.load(ABME + base + 1 * H + offs_h).to(tl.float32)
    m_val = tl.load(ABME + base + 2 * H + offs_h).to(tl.float32)
    e_val = tl.load(ABME + base + 3 * H + offs_h).to(tl.float32) * inv_tau

    a_log = tl.load(A_log + offs_h).to(tl.float32)
    dt_b = tl.load(Dt_bias + offs_h).to(tl.float32)
    mu_log = tl.load(Mu_log + offs_h).to(tl.float32)
    mu_b = tl.load(Mu_bias + offs_h).to(tl.float32)
    lf = tl.load(Log_factor + offs_h).to(tl.float32)

    xm = m_val + mu_b
    sp_m = tl.where(xm > 20.0, xm, tl.log(1.0 + tl.exp(xm)))
    log_mu_raw = -tl.exp(mu_log) * sp_m
    log_mu = tl.maximum(log_mu_raw, min_log_mu)

    xa = a_val + dt_b
    sp_a = tl.where(xa > 20.0, xa, tl.log(1.0 + tl.exp(xa)))
    log_alpha_base = -tl.exp(a_log) * sp_a

    # tanh(e) = 2 * sigmoid(2*e) - 1, so eta = tanh(e) + 1 = 2 * sigmoid(2*e)
    eta = 2.0 * tl.sigmoid(2.0 * e_val)
    sig_b = tl.sigmoid(b_val)

    # Overflow-safe algebraic symplectic coupling for r = eta * exp(lf):
    # sin^2(arctan(r)) = r^2 / (1 + r^2) if r <= 1 else 1 / (1 + (1/r)^2)
    # log(cos^2(arctan(r))) = -log(1 + r^2) if r <= 1 else -2*log(r) - log(1 + (1/r)^2)
    r = eta * tl.exp(lf)
    r_gt_1 = r > 1.0
    r_safe = tl.where(r_gt_1, r, 1.0)
    inv_r = 1.0 / r_safe
    t2 = tl.where(r_gt_1, inv_r * inv_r, r * r)
    one_plus_t2 = 1.0 + t2
    sin2 = tl.where(r_gt_1, 1.0 / one_plus_t2, t2 / one_plus_t2)
    log_cos2 = tl.where(r_gt_1, -2.0 * tl.log(r_safe) - tl.log(one_plus_t2), -tl.log(one_plus_t2))

    beta_mult = 2.0 if NEG_EIG else 1.0
    beta = sin2 * sig_b * beta_mult
    log_alpha = log_cos2 + log_alpha_base

    out_off = row * H + offs_h
    tl.store(LogAlpha_out + out_off, log_alpha)
    tl.store(LogMu_out + out_off, log_mu)
    tl.store(Beta_out + out_off, beta)
    tl.store(Eta_out + out_off, eta)


@triton.jit
def _mdr_fused_symplectic_gates_bwd_kernel(
    ABME, A_log, Dt_bias, Mu_log, Mu_bias, Log_factor,
    dLA, dLM, dBeta, dEta_in,
    dABME, dParams_out,
    N_ROWS, inv_tau, min_log_mu,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NEG_EIG: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, H)

    a_log = tl.load(A_log + offs_h).to(tl.float32)
    dt_b = tl.load(Dt_bias + offs_h).to(tl.float32)
    mu_log = tl.load(Mu_log + offs_h).to(tl.float32)
    mu_b = tl.load(Mu_bias + offs_h).to(tl.float32)
    lf = tl.load(Log_factor + offs_h).to(tl.float32)

    exp_a = tl.exp(a_log)
    exp_mu = tl.exp(mu_log)
    exp_lf = tl.exp(lf)
    beta_mult = 2.0 if NEG_EIG else 1.0

    da_log_acc = tl.zeros([H], dtype=tl.float32)
    ddt_b_acc = tl.zeros([H], dtype=tl.float32)
    dmu_log_acc = tl.zeros([H], dtype=tl.float32)
    dmu_b_acc = tl.zeros([H], dtype=tl.float32)
    dlf_acc = tl.zeros([H], dtype=tl.float32)

    r_start = pid * BLOCK_R
    for ir in range(BLOCK_R):
        row = r_start + ir
        m_row = row < N_ROWS
        base = row * (4 * H)
        a_val = tl.load(ABME + base + 0 * H + offs_h, mask=m_row, other=0.0).to(tl.float32)
        b_val = tl.load(ABME + base + 1 * H + offs_h, mask=m_row, other=0.0).to(tl.float32)
        m_val = tl.load(ABME + base + 2 * H + offs_h, mask=m_row, other=0.0).to(tl.float32)
        e_raw = tl.load(ABME + base + 3 * H + offs_h, mask=m_row, other=0.0).to(tl.float32)
        e_val = e_raw * inv_tau

        out_off = row * H + offs_h
        dla = tl.load(dLA + out_off, mask=m_row, other=0.0).to(tl.float32)
        dlm = tl.load(dLM + out_off, mask=m_row, other=0.0).to(tl.float32)
        dbeta = tl.load(dBeta + out_off, mask=m_row, other=0.0).to(tl.float32)
        deta_in = tl.load(dEta_in + out_off, mask=m_row, other=0.0).to(tl.float32)

        xm = m_val + mu_b
        sp_m = tl.where(xm > 20.0, xm, tl.log(1.0 + tl.exp(xm)))
        log_mu_raw = -exp_mu * sp_m
        active_mu = log_mu_raw > min_log_mu
        dlm_eff = tl.where(active_mu & m_row, dlm, 0.0)
        sig_xm = tl.sigmoid(xm)
        dm_out = -dlm_eff * exp_mu * sig_xm
        dmu_log_acc += -dlm_eff * exp_mu * sp_m
        dmu_b_acc += dm_out

        xa = a_val + dt_b
        sp_a = tl.where(xa > 20.0, xa, tl.log(1.0 + tl.exp(xa)))
        sig_xa = tl.sigmoid(xa)
        dla_eff = tl.where(m_row, dla, 0.0)
        da_out = -dla_eff * exp_a * sig_xa
        da_log_acc += -dla_eff * exp_a * sp_a
        ddt_b_acc += da_out

        sig_2e = tl.sigmoid(2.0 * e_val)
        eta = 2.0 * sig_2e
        sig_b = tl.sigmoid(b_val)

        r = eta * exp_lf
        r_gt_1 = r > 1.0
        r_safe = tl.where(r_gt_1, r, 1.0)
        inv_r = 1.0 / r_safe
        t2 = tl.where(r_gt_1, inv_r * inv_r, r * r)
        one_plus_t2 = 1.0 + t2
        sin2 = tl.where(r_gt_1, 1.0 / one_plus_t2, t2 / one_plus_t2)
        cos2 = tl.where(r_gt_1, t2 / one_plus_t2, 1.0 / one_plus_t2)

        dbeta_eff = tl.where(m_row, dbeta, 0.0)
        db_out = dbeta_eff * beta_mult * sin2 * sig_b * (1.0 - sig_b)
        dsin2 = dbeta_eff * beta_mult * sig_b

        # d(sin2)/dr = 2*r / (1+r^2)^2 = 2 * r * cos2^2; d(log_cos2)/dr = -2*r / (1+r^2) = -2 * r * cos2
        dr_sin2 = dsin2 * 2.0 * r * cos2 * cos2
        dr_lcos2 = -dla_eff * 2.0 * r * cos2
        dr = dr_sin2 + dr_lcos2

        dlf_acc += dr * r
        deta_total = tl.where(m_row, deta_in, 0.0) + dr * exp_lf
        # d(eta)/d(e_raw) = 4 * sig_2e * (1 - sig_2e) * inv_tau
        de_out = deta_total * 4.0 * sig_2e * (1.0 - sig_2e) * inv_tau

        tl.store(dABME + base + 0 * H + offs_h, da_out.to(dABME.dtype.element_ty), mask=m_row)
        tl.store(dABME + base + 1 * H + offs_h, db_out.to(dABME.dtype.element_ty), mask=m_row)
        tl.store(dABME + base + 2 * H + offs_h, dm_out.to(dABME.dtype.element_ty), mask=m_row)
        tl.store(dABME + base + 3 * H + offs_h, de_out.to(dABME.dtype.element_ty), mask=m_row)

    p_base = pid * (5 * H)
    tl.store(dParams_out + p_base + 0 * H + offs_h, da_log_acc)
    tl.store(dParams_out + p_base + 1 * H + offs_h, ddt_b_acc)
    tl.store(dParams_out + p_base + 2 * H + offs_h, dmu_log_acc)
    tl.store(dParams_out + p_base + 3 * H + offs_h, dmu_b_acc)
    tl.store(dParams_out + p_base + 4 * H + offs_h, dlf_acc)


@triton.jit
def _mdr_fused_output_corr_fwd_kernel(
    Q_in, K_in, D_log, Q_corr_out,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i_h = row % H
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    d_val = tl.exp(tl.load(D_log + i_h).to(tl.float32))
    q_val = tl.load(Q_in + row * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    k_val = tl.load(K_in + row * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    q_corr = q_val - d_val * k_val
    tl.store(Q_corr_out + row * K + offs_k, q_corr.to(Q_corr_out.dtype.element_ty), mask=mask_k)


@triton.jit
def _mdr_fused_output_corr_bwd_kernel(
    dQ_corr, K_in, dK_k, dK_p, D_log,
    dK_total_out, dD_log_part,
    N_TOKENS,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    d_val = tl.exp(tl.load(D_log + i_h).to(tl.float32))
    dd_acc = 0.0
    t_start = pid_r * BLOCK_R

    for ir in range(BLOCK_R):
        t_idx = t_start + ir
        m_t = t_idx < N_TOKENS
        row = t_idx * H + i_h
        m_all = m_t & mask_k
        dq_c = tl.load(dQ_corr + row * K + offs_k, mask=m_all, other=0.0).to(tl.float32)
        k_val = tl.load(K_in + row * K + offs_k, mask=m_all, other=0.0).to(tl.float32)
        dk_1 = tl.load(dK_k + row * K + offs_k, mask=m_all, other=0.0).to(tl.float32)
        dk_2 = tl.load(dK_p + row * K + offs_k, mask=m_all, other=0.0).to(tl.float32)

        dd_acc += -d_val * tl.sum(dq_c * k_val, axis=0)
        dk_tot = dk_1 + dk_2 - d_val * dq_c
        tl.store(dK_total_out + row * K + offs_k, dk_tot.to(dK_total_out.dtype.element_ty), mask=m_all)

    tl.store(dD_log_part + pid_r * H + i_h, dd_acc)


@triton.jit
def _mdr_fused_rmsnorm_gated_fwd_kernel(
    X, G, W, Y, Rstd,
    eps,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    x = tl.load(X + row * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    g = tl.load(G + row * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    w = tl.load(W + offs_v, mask=mask_v, other=0.0).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / V + eps)
    tl.store(Rstd + row, rstd)
    silu_g = g * tl.sigmoid(g)
    y = (x * rstd) * w * silu_g
    tl.store(Y + row * V + offs_v, y.to(Y.dtype.element_ty), mask=mask_v)


@triton.jit
def _mdr_fused_rmsnorm_gated_bwd_kernel(
    dY, X, G, W, Rstd,
    dX, dG, dW_part,
    N_ROWS,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0).to(tl.int64)
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    w = tl.load(W + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    r_start = pid_r * BLOCK_R

    for ir in range(BLOCK_R):
        row = r_start + ir
        m_row = row < N_ROWS
        m_all = m_row & mask_v

        dy = tl.load(dY + row * V + offs_v, mask=m_all, other=0.0).to(tl.float32)
        x = tl.load(X + row * V + offs_v, mask=m_all, other=0.0).to(tl.float32)
        g = tl.load(G + row * V + offs_v, mask=m_all, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row, mask=m_row, other=0.0).to(tl.float32)

        sig_g = tl.sigmoid(g)
        silu_g = g * sig_g
        dsilu_g = sig_g * (1.0 + g * (1.0 - sig_g))

        x_hat = x * rstd
        dg = dy * x_hat * w * dsilu_g
        dw_acc += dy * x_hat * silu_g

        dx_hat = dy * w * silu_g
        c1 = tl.sum(dx_hat * x_hat, axis=0) / V
        dx = rstd * (dx_hat - x_hat * c1)

        tl.store(dX + row * V + offs_v, dx.to(dX.dtype.element_ty), mask=m_all)
        tl.store(dG + row * V + offs_v, dg.to(dG.dtype.element_ty), mask=m_all)

    tl.store(dW_part + pid_r * V + offs_v, dw_acc, mask=mask_v)


# ==================================================================================================
# 4. Python Orchestration & Autograd Functions (`chunk_momentum_delta_rule_rsi`)
# ==================================================================================================

def _run_mdr_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    beta: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_p_times_alpha: bool,
    eps: float = 1e-6,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = T // BT
    dot_prec = "ieee" if q.dtype == torch.float32 else "tf32"
    work_dtype = torch.float32 if q.dtype == torch.float32 else torch.bfloat16

    N_rows = B * T * H
    BLOCK_K = triton.next_power_of_2(K)
    q_eff = torch.empty(B, T, H, K, device=q.device, dtype=work_dtype)
    k_eff = torch.empty(B, T, H, K, device=q.device, dtype=work_dtype)
    p_eff = torch.empty(B, T, H, K, device=q.device, dtype=work_dtype)
    q_rstd = torch.empty(N_rows, device=q.device, dtype=torch.float32) if use_qk_l2norm_in_kernel else q.new_empty(0, dtype=torch.float32)
    k_rstd = torch.empty(N_rows, device=q.device, dtype=torch.float32) if use_qk_l2norm_in_kernel else q.new_empty(0, dtype=torch.float32)
    p_rstd = torch.empty(N_rows, device=q.device, dtype=torch.float32) if use_qk_l2norm_in_kernel else q.new_empty(0, dtype=torch.float32)

    _mdr_triple_l2norm_scale_fwd_kernel[(N_rows,)](
        q, k, p, eta, log_alpha,
        q_eff, k_eff, p_eff, q_rstd, k_rstd, p_rstd,
        float(scale),
        K_DIM=K, BLOCK_K=BLOCK_K,
        USE_NORM=use_qk_l2norm_in_kernel,
        USE_P_ALPHA=use_p_times_alpha,
        EPS=eps,
        num_warps=4, num_stages=1,
    )

    n_chunks_total = B * H * NT
    la_cum = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    lm_cum = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    a_cum = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    m_cum = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    bar_a_tm1 = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    b_t = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    b_tm1 = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    ds_vec = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    dm_vec = torch.empty(n_chunks_total * BT, device=q.device, dtype=torch.float32)
    a_last = torch.empty(n_chunks_total, device=q.device, dtype=torch.float32)
    b_last = torch.empty(n_chunks_total, device=q.device, dtype=torch.float32)
    m_last = torch.empty(n_chunks_total, device=q.device, dtype=torch.float32)
    Gamma_q = torch.empty(n_chunks_total, BT, BT, device=q.device, dtype=torch.float32)

    _mdr_chunk_gate_greens_fwd_kernel[(NT, B * H)](
        log_alpha, log_mu, beta,
        la_cum, lm_cum, a_cum, m_cum, bar_a_tm1, b_t, b_tm1,
        ds_vec, dm_vec, a_last, b_last, m_last,
        Gamma_q,
        T=T, H=H, BT=BT,
        num_warps=4, num_stages=1,
    )

    A_mat = torch.empty(n_chunks_total, BT, BT, device=q.device, dtype=torch.float32)
    Mqk_mat = torch.empty(n_chunks_total, BT, BT, device=q.device, dtype=torch.float32)
    BM_sub = 16
    _mdr_chunk_kkt_fwd_kernel[(NT * (BT // BM_sub), B * H)](
        q_eff, p_eff, k_eff, Gamma_q, A_mat, Mqk_mat,
        T=T, H=H, K=K, BT=BT, BM=BM_sub, BK=32, DOT_PREC=dot_prec,
        num_warps=4, num_stages=1,
    )

    Ai_mat = torch.empty(n_chunks_total, BT, BT, device=q.device, dtype=torch.float32)
    _mdr_ut_solve_64x64_kernel[(n_chunks_total,)](
        A_mat, Ai_mat,
        NUM_CHUNKS=n_chunks_total,
        num_warps=4, num_stages=1,
    )

    BV_wy = 16
    BK_wy = 32
    u_c = torch.empty(B, T, H, V, device=q.device, dtype=work_dtype)
    y_c = torch.empty(B, T, H, K, device=q.device, dtype=work_dtype)
    z_c = torch.empty(B, T, H, K, device=q.device, dtype=work_dtype)

    _mdr_wy_u_fwd_kernel[(triton.cdiv(V, BV_wy), NT, B * H)](
        Ai_mat, v.to(work_dtype), u_c,
        T=T, H=H, V=V, BT=BT, BV=BV_wy, DOT_PREC=dot_prec,
        num_warps=4, num_stages=1,
    )
    _mdr_wy_yz_fwd_kernel[(triton.cdiv(K, BK_wy), NT, B * H)](
        Ai_mat, p_eff, bar_a_tm1, b_tm1, y_c, z_c,
        T=T, H=H, K=K, BT=BT, BK=BK_wy, DOT_PREC=dot_prec,
        num_warps=4, num_stages=1,
    )

    has_init = initial_state is not None
    S0 = initial_state[0].contiguous() if has_init else q.new_empty(0)
    M0 = initial_state[1].contiguous() if has_init else q.new_empty(0)
    S_pre_all = torch.empty(B * H * NT, K, V, device=q.device, dtype=work_dtype)
    M_pre_all = torch.empty(B * H * NT, K, V, device=q.device, dtype=work_dtype)
    v_new = torch.empty(B, T, H, V, device=q.device, dtype=work_dtype)
    ST = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if output_final_state else q.new_empty(0, dtype=torch.float32)
    MT = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if output_final_state else q.new_empty(0, dtype=torch.float32)

    BV_h = 16
    if K == 64 and q.dtype != torch.float32:
        BV_inreg = 16
        _mdr_chunk_fwd_h_inreg_kernel[(triton.cdiv(V, BV_inreg), B * H)](
            k_eff, u_c, y_c, z_c,
            ds_vec, dm_vec, a_last, b_last, m_last,
            S0, M0, S_pre_all, M_pre_all, v_new, ST, MT,
            T=T, H=H, K=K, V=V, BT=BT, BV=BV_inreg,
            HAS_INIT=has_init, STORE_FINAL=output_final_state,
            num_warps=4, num_stages=1,
        )
    else:
        _mdr_chunk_fwd_h_l2_kernel[(triton.cdiv(V, BV_h), B * H)](
            k_eff, u_c, y_c, z_c,
            ds_vec, dm_vec, a_last, b_last, m_last,
            S0, M0, S_pre_all, M_pre_all, v_new, ST, MT,
            T=T, H=H, K=K, V=V, BT=BT, BK=16, BV=BV_h,
            HAS_INIT=has_init, STORE_FINAL=output_final_state, DOT_PREC=dot_prec,
            num_warps=2, num_stages=1,
        )

    BV_o = 32
    o = torch.empty(B, T, H, V, device=q.device, dtype=q.dtype)
    _mdr_chunk_fwd_o_kernel[(triton.cdiv(V, BV_o), NT, B * H)](
        q_eff, v_new, S_pre_all, M_pre_all,
        a_cum, b_t, Mqk_mat, o,
        T=T, H=H, K=K, V=V, BT=BT, BK=32, BV=BV_o, DOT_PREC=dot_prec,
        num_warps=4, num_stages=1,
    )

    final_state = torch.stack([ST, MT], dim=0) if output_final_state else None
    saved = (
        q_eff, k_eff, p_eff, q_rstd, k_rstd, p_rstd,
        la_cum, lm_cum, a_cum, m_cum, bar_a_tm1, b_t, b_tm1,
        ds_vec, dm_vec, a_last, b_last, m_last,
        Gamma_q, Mqk_mat, Ai_mat, y_c, z_c, S_pre_all, M_pre_all, v_new,
    )
    return o, final_state, saved


def _run_mdr_bwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    beta: torch.Tensor,
    eta: torch.Tensor,
    do: torch.Tensor,
    d_final_state: Optional[torch.Tensor],
    saved: Tuple[torch.Tensor, ...],
    scale: float,
    has_init_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_p_times_alpha: bool,
):
    (
        q_eff, k_eff, p_eff, q_rstd, k_rstd, p_rstd,
        la_cum, lm_cum, a_cum, m_cum, bar_a_tm1, b_t, b_tm1,
        ds_vec, dm_vec, a_last, b_last, m_last,
        Gamma_q, Mqk_mat, Ai_mat, y_c, z_c, S_pre_all, M_pre_all, v_new,
    ) = saved

    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = T // BT
    n_chunks_total = B * H * NT
    dot_prec = "ieee" if q.dtype == torch.float32 else "tf32"
    work_dtype = q_eff.dtype
    do_w = do.to(work_dtype).contiguous()

    BV_h = 16
    dV_new = torch.empty(B, T, H, V, device=q.device, dtype=work_dtype)
    _mdr_bwd_dv_intra_kernel[(triton.cdiv(V, BV_h), NT, B * H)](
        Mqk_mat, do_w, dV_new,
        T=T, H=H, V=V, BT=BT, BV=BV_h, DOT_PREC=dot_prec,
        num_warps=4, num_stages=1,
    )

    has_final_grad = d_final_state is not None
    dST = d_final_state[0].contiguous() if has_final_grad else q.new_empty(0)
    dMT = d_final_state[1].contiguous() if has_final_grad else q.new_empty(0)
    dS_next_all = torch.empty(B * H * NT, K, V, device=q.device, dtype=work_dtype)
    dM_next_all = torch.empty(B * H * NT, K, V, device=q.device, dtype=work_dtype)
    dS0 = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if has_init_state else q.new_empty(0, dtype=torch.float32)
    dM0 = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if has_init_state else q.new_empty(0, dtype=torch.float32)

    if K == 64 and q.dtype != torch.float32:
        BV_inreg = 16
        _mdr_chunk_bwd_dhu_inreg_kernel[(triton.cdiv(V, BV_inreg), B * H)](
            q_eff, k_eff, y_c, z_c, do_w,
            a_cum, b_t, ds_vec, dm_vec, a_last, b_last, m_last,
            dST, dMT, dS_next_all, dM_next_all, dV_new, dS0, dM0,
            T=T, H=H, K=K, V=V, BT=BT, BV=BV_inreg,
            HAS_FINAL_GRAD=has_final_grad, HAS_INIT_GRAD=has_init_state,
            num_warps=4, num_stages=1,
        )
    else:
        _mdr_chunk_bwd_dhu_l2_kernel[(triton.cdiv(V, BV_h), B * H)](
            q_eff, k_eff, y_c, z_c, do_w,
            a_cum, b_t, ds_vec, dm_vec, a_last, b_last, m_last,
            dST, dMT, dS_next_all, dM_next_all, dV_new, dS0, dM0,
            T=T, H=H, K=K, V=V, BT=BT, BK=16, BV=BV_h,
            HAS_FINAL_GRAD=has_final_grad, HAS_INIT_GRAD=has_init_state, DOT_PREC=dot_prec,
            num_warps=2, num_stages=1,
        )

    dX = torch.empty(B, T, H, V, device=q.device, dtype=work_dtype)
    _mdr_bwd_dx_dv_kernel[(triton.cdiv(V, BV_h), NT, B * H)](
        Ai_mat, dV_new, dX,
        T=T, H=H, V=V, BT=BT, BV=BV_h, DOT_PREC=dot_prec,
        num_warps=4, num_stages=1,
    )

    BM_sub = 16
    dA_mat = torch.empty(n_chunks_total, BT, BT, device=q.device, dtype=torch.float32)
    dMqk_mat = torch.empty(n_chunks_total, BT, BT, device=q.device, dtype=torch.float32)
    dGq_raw = torch.empty(n_chunks_total, BT, BT, device=q.device, dtype=torch.float32)
    dG_raw = torch.empty(n_chunks_total, BT, BT, device=q.device, dtype=torch.float32)
    _mdr_bwd_dA_dMqk_kernel[(NT * (BT // BM_sub), B * H)](
        dX, do_w, v_new, q_eff, k_eff, p_eff,
        dA_mat, dMqk_mat, dGq_raw, dG_raw,
        T=T, H=H, K=K, V=V, BT=BT, BM=BM_sub, BK=32, BV=32, DOT_PREC=dot_prec,
        num_warps=4, num_stages=1,
    )

    BK_bwd = 16
    NK = K // BK_bwd
    dQ_eff = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dK_eff = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dP_eff = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dA_cum_part = torch.empty(n_chunks_total * NK * 2 * BT, device=q.device, dtype=torch.float32)
    dB_t_part = torch.empty(n_chunks_total * NK * 2 * BT, device=q.device, dtype=torch.float32)
    dK_vds_part = torch.empty(n_chunks_total * NK * BT, device=q.device, dtype=torch.float32)
    dK_vdm_part = torch.empty(n_chunks_total * NK * BT, device=q.device, dtype=torch.float32)
    dState_sc_part = torch.empty(n_chunks_total * NK * 3, device=q.device, dtype=torch.float32)

    _mdr_bwd_dq_dk_dp_inter_kernel[(NK, NT * (BT // BM_sub), B * H)](
        q_eff, k_eff, p_eff, v_new, do_w, dX,
        S_pre_all, M_pre_all, dS_next_all, dM_next_all,
        a_cum, b_t, bar_a_tm1, b_tm1, ds_vec, dm_vec,
        Gamma_q, dA_mat, dMqk_mat,
        dQ_eff, dK_eff, dP_eff,
        dA_cum_part, dB_t_part, dK_vds_part, dK_vdm_part, dState_sc_part,
        T=T, H=H, K=K, V=V, BT=BT, BM=BM_sub, BK=BK_bwd, BV=16, DOT_PREC=dot_prec,
        num_warps=4, num_stages=1,
    )

    dLA_intra = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    dLM_out = torch.empty(B, T, H, device=q.device, dtype=log_mu.dtype)
    dBeta_out = torch.empty(B, T, H, device=q.device, dtype=beta.dtype)

    _mdr_bwd_gate_greens_kernel[(NT, B * H)](
        log_alpha, log_mu, beta,
        la_cum, lm_cum, a_cum, m_cum,
        dGq_raw, dG_raw,
        dA_cum_part, dB_t_part, dK_vds_part, dK_vdm_part, dState_sc_part,
        dLA_intra, dLM_out, dBeta_out,
        T=T, H=H, BT=BT, NK=NK,
        num_warps=4, num_stages=1,
    )

    N_rows = B * T * H
    BLOCK_K = triton.next_power_of_2(K)
    dQ = torch.empty_like(q)
    dK = torch.empty_like(k)
    dP = torch.empty_like(p)
    dEta = torch.empty_like(eta)
    dLA_out = torch.empty_like(log_alpha)

    _mdr_triple_l2norm_scale_bwd_kernel[(N_rows,)](
        q, k, p, q_rstd, k_rstd, p_rstd, eta, log_alpha,
        dQ_eff, dK_eff, dP_eff, dLA_intra,
        dQ, dK, dP, dEta, dLA_out,
        float(scale),
        K_DIM=K, BLOCK_K=BLOCK_K,
        USE_NORM=use_qk_l2norm_in_kernel,
        USE_P_ALPHA=use_p_times_alpha,
        num_warps=4, num_stages=1,
    )

    dV = dX.to(v.dtype)
    d_init = torch.stack([dS0, dM0], dim=0) if has_init_state else None
    return dQ, dK, dV, dP, dLA_out, dLM_out, dBeta_out, dEta, d_init


class _ChunkMomentumDeltaRuleRSIFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        beta: torch.Tensor,
        eta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        use_p_times_alpha: bool,
    ):
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        p_c = p.contiguous()
        la_c = log_alpha.contiguous()
        lm_c = log_mu.contiguous()
        bt_c = beta.contiguous()
        et_c = eta.contiguous()

        o, final_state, saved = _run_mdr_fwd_impl(
            q_c, k_c, v_c, p_c, la_c, lm_c, bt_c, et_c,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_p_times_alpha=use_p_times_alpha,
        )
        ctx.save_for_backward(q_c, k_c, v_c, p_c, la_c, lm_c, bt_c, et_c, *saved)
        ctx.scale = scale
        ctx.has_init_state = initial_state is not None
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_p_times_alpha = use_p_times_alpha
        return o, final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, d_final_state: Optional[torch.Tensor]):
        all_saved = ctx.saved_tensors
        q_c, k_c, v_c, p_c, la_c, lm_c, bt_c, et_c = all_saved[:8]
        saved = all_saved[8:]
        dQ, dK, dV, dP, dLA, dLM, dBeta, dEta, d_init = _run_mdr_bwd_impl(
            q_c, k_c, v_c, p_c, la_c, lm_c, bt_c, et_c,
            do=do.contiguous(),
            d_final_state=d_final_state,
            saved=saved,
            scale=ctx.scale,
            has_init_state=ctx.has_init_state,
            use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
            use_p_times_alpha=ctx.use_p_times_alpha,
        )
        return dQ, dK, dV, dP, dLA, dLM, dBeta, dEta, None, d_init, None, None, None


def chunk_momentum_delta_rule_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    p: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_p_times_alpha: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Round-18 RSI Zero-Spill Momentum DeltaNet Chunk Operator (`arXiv:2605.05838`).
    Supports both `torch.bfloat16` and `torch.float32` (`IEEE` Tensor Core precision),
    arbitrary sequence lengths `T` (via exact zero-padded chunk boundary handling),
    and full analytical zero-spill Triton backward (`0` autograd replay).
    """
    B, T, H, K = q.shape
    if scale is None:
        scale = K ** -0.5
    if p is None:
        p = k
    if beta is None:
        beta = torch.ones(B, T, H, device=q.device, dtype=torch.float32)
    if eta is None:
        eta = torch.ones(B, T, H, device=q.device, dtype=torch.float32)

    BT = 64
    T_orig = T
    if T % BT != 0:
        pad = BT - (T % BT)
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        p = F.pad(p, (0, 0, 0, 0, 0, pad))
        log_alpha = F.pad(log_alpha, (0, 0, 0, pad))
        log_mu = F.pad(log_mu, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        eta = F.pad(eta, (0, 0, 0, pad))

    o, final_state = _ChunkMomentumDeltaRuleRSIFn.apply(
        q, k, v, p, log_alpha, log_mu, beta, eta,
        float(scale), initial_state, output_final_state,
        use_qk_l2norm_in_kernel, use_p_times_alpha,
    )
    if T_orig != q.shape[1]:
        o = o[:, :T_orig].contiguous()
    return o, final_state


# ==================================================================================================
# 5. Superfused Autograd Wrappers & `MomentumDeltaNetBlockRSI` (Stages A0..A4)
# ==================================================================================================

class _MDRShortConvSiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor):
        B, T, D = x.shape
        x_c = x.contiguous()
        w_c = w.squeeze(1).contiguous()
        y = torch.empty_like(x_c)
        z_pre = torch.empty(B, T, D, device=x.device, dtype=torch.float32)
        BT, BD = 16, 32
        grid = (triton.cdiv(D, BD), triton.cdiv(T, BT), B)
        _mdr_fused_shortconv_silu_fwd_kernel[grid](
            x_c, w_c, y, z_pre, B, T, D, BT=BT, BD=BD, num_warps=4, num_stages=1
        )
        ctx.save_for_backward(x_c, w_c, z_pre)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, w_c, z_pre = ctx.saved_tensors
        B, T, D = x_c.shape
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        dz = torch.empty(B, T, D, device=x_c.device, dtype=torch.float32)
        dw = torch.empty_like(w_c)
        BT, BD = 16, 32
        grid = (triton.cdiv(D, BD), triton.cdiv(T, BT), B)
        _mdr_fused_shortconv_silu_bwd_dx_kernel[grid](
            dy_c, z_pre, w_c, dx, dz, B, T, D, BT=BT, BD=BD, num_warps=4, num_stages=1
        )
        _mdr_fused_shortconv_silu_bwd_dw_kernel[(triton.cdiv(D, BD),)](
            dz, x_c, dw, B, T, D, BT=BT, BD=BD, num_warps=4, num_stages=1
        )
        return dx, dw.unsqueeze(1)


class _MDRSymplecticGatesFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        abme: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        Mu_log: torch.Tensor,
        mu_bias: torch.Tensor,
        log_factor: torch.Tensor,
        inv_tau: float,
        min_log_mu: float,
        allow_neg_eigval: bool,
    ):
        B, T, H4 = abme.shape
        H = H4 // 4
        N_rows = B * T
        abme_c = abme.contiguous()
        la = torch.empty(B, T, H, device=abme.device, dtype=torch.float32)
        lm = torch.empty(B, T, H, device=abme.device, dtype=torch.float32)
        beta = torch.empty(B, T, H, device=abme.device, dtype=torch.float32)
        eta = torch.empty(B, T, H, device=abme.device, dtype=torch.float32)

        _mdr_fused_symplectic_gates_fwd_kernel[(N_rows,)](
            abme_c, A_log, dt_bias, Mu_log, mu_bias, log_factor,
            la, lm, beta, eta,
            float(inv_tau), float(min_log_mu),
            H=H, NEG_EIG=allow_neg_eigval,
            num_warps=1, num_stages=1,
        )
        ctx.save_for_backward(abme_c, A_log, dt_bias, Mu_log, mu_bias, log_factor)
        ctx.inv_tau = float(inv_tau)
        ctx.min_log_mu = float(min_log_mu)
        ctx.allow_neg_eigval = allow_neg_eigval
        return la, lm, beta, eta

    @staticmethod
    def backward(ctx, dla: torch.Tensor, dlm: torch.Tensor, dbeta: torch.Tensor, deta: torch.Tensor):
        abme_c, A_log, dt_bias, Mu_log, mu_bias, log_factor = ctx.saved_tensors
        B, T, H4 = abme_c.shape
        H = H4 // 4
        N_rows = B * T
        BLOCK_R = 16
        n_blocks = triton.cdiv(N_rows, BLOCK_R)
        dabme = torch.empty_like(abme_c)
        dparams_part = torch.empty(n_blocks, 5, H, device=abme_c.device, dtype=torch.float32)

        _mdr_fused_symplectic_gates_bwd_kernel[(n_blocks,)](
            abme_c, A_log, dt_bias, Mu_log, mu_bias, log_factor,
            dla.contiguous(), dlm.contiguous(), dbeta.contiguous(), deta.contiguous(),
            dabme, dparams_part,
            N_rows, ctx.inv_tau, ctx.min_log_mu,
            H=H, BLOCK_R=BLOCK_R, NEG_EIG=ctx.allow_neg_eigval,
            num_warps=1, num_stages=1,
        )
        dp_sum = dparams_part.sum(dim=0)
        return (
            dabme,
            dp_sum[0].to(A_log.dtype),
            dp_sum[1].to(dt_bias.dtype),
            dp_sum[2].to(Mu_log.dtype),
            dp_sum[3].to(mu_bias.dtype),
            dp_sum[4].to(log_factor.dtype),
            None, None, None,
        )


class _MDROutputCorrSharedKP(torch.autograd.Function):
    """Fused output correction `q_corr = q - exp(D_log) * k` + shared `(k, p=k)` gradient merge."""
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, D_log: torch.Tensor):
        B, T, H, K = q.shape
        q_c = q.contiguous()
        k_c = k.contiguous()
        q_corr = torch.empty_like(q_c)
        N_rows = B * T * H
        BLOCK_K = triton.next_power_of_2(K)
        _mdr_fused_output_corr_fwd_kernel[(N_rows,)](
            q_c, k_c, D_log, q_corr, H=H, K=K, BLOCK_K=BLOCK_K, num_warps=2, num_stages=1
        )
        ctx.save_for_backward(k_c, D_log)
        return q_corr, k_c, k_c

    @staticmethod
    def backward(ctx, dq_corr: torch.Tensor, dk_k: torch.Tensor, dk_p: torch.Tensor):
        k_c, D_log = ctx.saved_tensors
        B, T, H, K = k_c.shape
        N_tokens = B * T
        BLOCK_R = 16
        BLOCK_K = triton.next_power_of_2(K)
        n_blocks = triton.cdiv(N_tokens, BLOCK_R)
        dk_total = torch.empty_like(k_c)
        dd_part = torch.empty(n_blocks, H, device=k_c.device, dtype=torch.float32)

        _mdr_fused_output_corr_bwd_kernel[(n_blocks, H)](
            dq_corr.contiguous(), k_c, dk_k.contiguous(), dk_p.contiguous(), D_log,
            dk_total, dd_part,
            N_tokens, H=H, K=K, BLOCK_K=BLOCK_K, BLOCK_R=BLOCK_R,
            num_warps=2, num_stages=1,
        )
        dD_log = dd_part.sum(dim=0).to(D_log.dtype)
        return dq_corr, dk_total, dD_log


class _MDRFusedRMSNormGatedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
        B, T, H, V = x.shape
        x_c = x.contiguous()
        g_c = g.contiguous()
        w_c = weight.contiguous()
        y = torch.empty_like(x_c)
        N_rows = B * T * H
        rstd = torch.empty(N_rows, device=x.device, dtype=torch.float32)
        BLOCK_V = triton.next_power_of_2(V)
        _mdr_fused_rmsnorm_gated_fwd_kernel[(N_rows,)](
            x_c, g_c, w_c,
            y, rstd, float(eps), V=V, BLOCK_V=BLOCK_V, num_warps=2, num_stages=1
        )
        ctx.save_for_backward(x_c, g_c, w_c, rstd)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, g_c, w_c, rstd = ctx.saved_tensors
        B, T, H, V = x_c.shape
        N_rows = B * T * H
        BLOCK_R = 16
        BLOCK_V = triton.next_power_of_2(V)
        n_blocks = triton.cdiv(N_rows, BLOCK_R)
        dx = torch.empty_like(x_c)
        dg = torch.empty_like(g_c)
        dw_part = torch.empty(n_blocks, V, device=x_c.device, dtype=torch.float32)

        _mdr_fused_rmsnorm_gated_bwd_kernel[(n_blocks,)](
            dy.contiguous(), x_c, g_c, w_c, rstd,
            dx, dg, dw_part,
            N_rows, V=V, BLOCK_V=BLOCK_V, BLOCK_R=BLOCK_R,
            num_warps=2, num_stages=1,
        )
        dw = dw_part.sum(dim=0).to(w_c.dtype)
        return dx, dg, dw, None


class MomentumDeltaNetBlockRSI(nn.Module):
    """
    Round-18 RSI Superfused MomentumDeltaNet Block (`arXiv:2605.05838`) supporting progressive
    ablation stages `A0..A4`:
      - Stage A0: Eager chunk reference + PyTorch autograd replay + unfused periphery
      - Stage A1: Zero-spill Triton 3-RHS Coupled WY forward + PyTorch autograd replay
      - Stage A2: Zero-spill Triton 3-RHS Coupled WY forward + Analytical `dX = Ai^T dV_new` backward
      - Stage A3: Stage A2 + Fused Algebraic Symplectic 4-Gates & Output Correction
      - Stage A4: Full Superfused Round-18 RSI Block (`ShortConv1d+SiLU` + Symplectic Gates + Output
                  Correction + Zero-Spill 3-RHS WY Fwd/Bwd + `FusedRMSNormGated` + Packed QKVG+ABME GEMM)
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        head_k_dim: int = 64,
        head_v_dim: int = 128,
        min_log_mu: float = -2.0,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        stage: str = "A4",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_heads * head_k_dim
        self.value_dim = num_heads * head_v_dim
        self.min_log_mu = min_log_mu
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps
        self.stage = stage
        self.tau = math.sqrt(float(d_model))

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)

        self.a_proj = nn.Linear(d_model, num_heads, bias=False)
        self.b_proj = nn.Linear(d_model, num_heads, bias=False)
        self.m_proj = nn.Linear(d_model, num_heads, bias=False)
        self.e_proj = nn.Linear(d_model, num_heads, bias=False)

        self.q_conv_weight = nn.Parameter(torch.randn(self.key_dim, 1, 4) * 0.02)
        self.k_conv_weight = nn.Parameter(torch.randn(self.key_dim, 1, 4) * 0.02)
        self.v_conv_weight = nn.Parameter(torch.randn(self.value_dim, 1, 4) * 0.02)

        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.full((num_heads,), -2.0, dtype=torch.float32))
        self.Mu_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.mu_bias = nn.Parameter(torch.full((num_heads,), -1.5, dtype=torch.float32))
        self.log_factor = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.D_log = nn.Parameter(torch.full((num_heads,), -1.0, dtype=torch.float32))

        self.norm_weight = nn.Parameter(torch.ones(head_v_dim, dtype=torch.float32))
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

    def _shortconv_eager(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x_p = F.pad(x_t, (3, 0))
        y = F.conv1d(x_p, w, groups=x.shape[-1]).transpose(1, 2)
        return F.silu(y)

    def _gates_eager(self, a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, e: torch.Tensor):
        af, bf, mf, ef = a.float(), b.float(), m.float(), e.float() / self.tau
        log_mu = (-self.Mu_log.float().exp() * F.softplus(mf + self.mu_bias.float())).clamp_min(self.min_log_mu)
        log_alpha_base = -self.A_log.float().exp() * F.softplus(af + self.dt_bias.float())
        eta = torch.tanh(ef) + 1.0
        beta_raw = torch.sigmoid(bf)
        theta = torch.arctan(eta * self.log_factor.float().exp())
        sin2 = torch.sin(theta) ** 2
        cos2 = torch.cos(theta) ** 2
        beta = sin2 * beta_raw * (2.0 if self.allow_neg_eigval else 1.0)
        log_alpha = cos2.log() + log_alpha_base
        return log_alpha, log_mu, beta, eta

    def _rmsnorm_gated_eager(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        gf = g.float()
        rstd = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + self.norm_eps)
        y = (xf * rstd) * self.norm_weight.float() * F.silu(gf)
        return y.to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, stage: Optional[str] = None) -> torch.Tensor:
        stg = stage if stage is not None else self.stage
        B, T, D = hidden_states.shape
        H, K, V = self.num_heads, self.head_k_dim, self.head_v_dim

        if stg == "A4":
            w_all = torch.cat([
                self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.g_proj.weight,
                self.a_proj.weight, self.b_proj.weight, self.m_proj.weight, self.e_proj.weight,
            ], dim=0)
            proj_all = F.linear(hidden_states, w_all)
            s0 = self.key_dim
            s1 = s0 + self.key_dim
            s2 = s1 + self.value_dim
            s3 = s2 + self.value_dim
            qkv_in = proj_all[..., :s2].contiguous()
            g = proj_all[..., s2:s3].reshape(B, T, H, V)
            abme = proj_all[..., s3:].contiguous()

            w_conv_all = torch.cat([self.q_conv_weight, self.k_conv_weight, self.v_conv_weight], dim=0)
            qkv_conv = _MDRShortConvSiLUFn.apply(qkv_in, w_conv_all)
            q = qkv_conv[..., :s0].reshape(B, T, H, K)
            k = qkv_conv[..., s0:s1].reshape(B, T, H, K)
            v = qkv_conv[..., s1:s2].reshape(B, T, H, V)
        else:
            q_raw = self.q_proj(hidden_states)
            k_raw = self.k_proj(hidden_states)
            v_raw = self.v_proj(hidden_states)
            g = self.g_proj(hidden_states).reshape(B, T, H, V)
            q = self._shortconv_eager(q_raw, self.q_conv_weight).reshape(B, T, H, K)
            k = self._shortconv_eager(k_raw, self.k_conv_weight).reshape(B, T, H, K)
            v = self._shortconv_eager(v_raw, self.v_conv_weight).reshape(B, T, H, V)

        if stg in ("A3", "A4"):
            if stg == "A3":
                abme = torch.cat([
                    self.a_proj(hidden_states), self.b_proj(hidden_states),
                    self.m_proj(hidden_states), self.e_proj(hidden_states),
                ], dim=-1)
            log_alpha, log_mu, beta, eta = _MDRSymplecticGatesFn.apply(
                abme, self.A_log, self.dt_bias, self.Mu_log, self.mu_bias, self.log_factor,
                1.0 / self.tau, self.min_log_mu, self.allow_neg_eigval,
            )
            q_corr, k_for_k, p_for_p = _MDROutputCorrSharedKP.apply(q, k, self.D_log)
        else:
            a = self.a_proj(hidden_states)
            b = self.b_proj(hidden_states)
            m = self.m_proj(hidden_states)
            e = self.e_proj(hidden_states)
            log_alpha, log_mu, beta, eta = self._gates_eager(a, b, m, e)
            q_corr = (q.float() - self.D_log.float().exp()[None, None, :, None] * k.float()).to(q.dtype)
            k_for_k, p_for_p = k, k

        if stg in ("A0", "A1"):
            o, _ = momentum_deltanet_chunk_eager_ref(
                q_corr, k_for_k, v, log_alpha, log_mu, p=p_for_p, beta=beta, eta=eta,
                use_qk_l2norm_in_kernel=True, use_p_times_alpha=True,
            )
            if stg == "A1":
                with torch.no_grad():
                    o_triton, _ = chunk_momentum_delta_rule_rsi(
                        q_corr, k_for_k, v, log_alpha, log_mu, p=p_for_p, beta=beta, eta=eta,
                        use_qk_l2norm_in_kernel=True, use_p_times_alpha=True,
                    )
                o = o + (o_triton - o).detach()
        else:
            o, _ = chunk_momentum_delta_rule_rsi(
                q_corr, k_for_k, v, log_alpha, log_mu, p=p_for_p, beta=beta, eta=eta,
                use_qk_l2norm_in_kernel=True, use_p_times_alpha=True,
            )

        if stg == "A4":
            o_normed = _MDRFusedRMSNormGatedFn.apply(o, g, self.norm_weight.to(o.dtype), self.norm_eps)
        else:
            o_normed = self._rmsnorm_gated_eager(o, g)

        return self.o_proj(o_normed.reshape(B, T, self.value_dim))


# ==================================================================================================
# 6. 8-GPU NVSwitch Ulysses-WY Coupled Dual-State Context Parallelism (`Momentum-CP = 8`)
# ==================================================================================================

def _all_to_all_seq_to_head(x: torch.Tensor, world_size: int) -> torch.Tensor:
    B, T_loc, H, D = x.shape
    H_loc = H // world_size
    if B == 1:
        x_send = x.view(T_loc, world_size, H_loc, D).transpose(0, 1).contiguous()
        x_recv = torch.empty_like(x_send)
        dist.all_to_all_single(x_recv, x_send)
        return x_recv.view(1, world_size * T_loc, H_loc, D)
    x_send = x.reshape(B, T_loc, world_size, H_loc, D).permute(2, 0, 1, 3, 4).contiguous()
    x_recv = torch.empty_like(x_send)
    dist.all_to_all_single(x_recv, x_send)
    return x_recv.permute(1, 0, 2, 3, 4).reshape(B, world_size * T_loc, H_loc, D).contiguous()


def _all_to_all_head_to_seq(x: torch.Tensor, world_size: int) -> torch.Tensor:
    B, T_glob, H_loc, D = x.shape
    T_loc = T_glob // world_size
    if B == 1:
        x_send = x.contiguous().view(world_size, T_loc, H_loc, D)
        x_recv = torch.empty_like(x_send)
        dist.all_to_all_single(x_recv, x_send)
        return x_recv.transpose(0, 1).contiguous().view(1, T_loc, world_size * H_loc, D)
    x_send = x.reshape(B, world_size, T_loc, H_loc, D).permute(1, 0, 2, 3, 4).contiguous()
    x_recv = torch.empty_like(x_send)
    dist.all_to_all_single(x_recv, x_send)
    return x_recv.permute(1, 2, 0, 3, 4).reshape(B, T_loc, world_size * H_loc, D).contiguous()


class _MomentumCP8AutogradFn(torch.autograd.Function):
    """
    Single-call NVSwitch `all_to_all_single` Ulysses-WY Context Parallelism (`Momentum-CP = 8`).
    Packs `(q, k, v, p, log_alpha, log_mu, beta, eta)` into a single `[B, T_local, H, 3K + V + 4]`
    tensor for a single NVSwitch `all_to_all_single` in forward and a single `all_to_all_single`
    in backward, running full-sequence Coupled Dual-State `[S, M]` Round-18 RSI on `H_local = H / P`
    heads per GPU with exact `0.0` cross-rank dual-state continuity error!
    """
    @staticmethod
    def forward(
        ctx,
        q_s: torch.Tensor,
        k_s: torch.Tensor,
        v_s: torch.Tensor,
        p_s: torch.Tensor,
        la_s: torch.Tensor,
        lm_s: torch.Tensor,
        beta_s: torch.Tensor,
        eta_s: torch.Tensor,
        world_size: int,
    ):
        B, T_loc, H, K = q_s.shape
        V = v_s.shape[-1]
        comm_dt = q_s.dtype
        packed_s = torch.cat([
            q_s, k_s, v_s, p_s,
            la_s.to(comm_dt).unsqueeze(-1), lm_s.to(comm_dt).unsqueeze(-1),
            beta_s.to(comm_dt).unsqueeze(-1), eta_s.to(comm_dt).unsqueeze(-1),
        ], dim=-1)
        packed_h = _all_to_all_seq_to_head(packed_s, world_size)

        c0 = K
        c1 = 2 * K
        c2 = 2 * K + V
        c3 = 3 * K + V
        q_h = packed_h[..., :c0].contiguous()
        k_h = packed_h[..., c0:c1].contiguous()
        v_h = packed_h[..., c1:c2].contiguous()
        p_h = packed_h[..., c2:c3].contiguous()
        la_h = packed_h[..., c3].float().contiguous()
        lm_h = packed_h[..., c3 + 1].float().contiguous()
        bt_h = packed_h[..., c3 + 2].float().contiguous()
        et_h = packed_h[..., c3 + 3].float().contiguous()

        scale = K ** -0.5
        o_h, _, saved = _run_mdr_fwd_impl(
            q_h, k_h, v_h, p_h, la_h, lm_h, bt_h, et_h,
            scale=scale, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True, use_p_times_alpha=True,
        )
        o_s = _all_to_all_head_to_seq(o_h, world_size)

        ctx.save_for_backward(q_h, k_h, v_h, p_h, la_h, lm_h, bt_h, et_h, *saved)
        ctx.scale = scale
        ctx.world_size = world_size
        ctx.K = K
        ctx.V = V
        ctx.gate_dt = la_s.dtype
        return o_s

    @staticmethod
    def backward(ctx, do_s: torch.Tensor):
        all_saved = ctx.saved_tensors
        q_h, k_h, v_h, p_h, la_h, lm_h, bt_h, et_h = all_saved[:8]
        saved = all_saved[8:]
        world_size = ctx.world_size
        K, V = ctx.K, ctx.V
        comm_dt = q_h.dtype
        gate_dt = ctx.gate_dt

        do_h = _all_to_all_seq_to_head(do_s.contiguous(), world_size)
        dQ_h, dK_h, dV_h, dP_h, dLA_h, dLM_h, dBeta_h, dEta_h, _ = _run_mdr_bwd_impl(
            q_h, k_h, v_h, p_h, la_h, lm_h, bt_h, et_h,
            do=do_h, d_final_state=None, saved=saved, scale=ctx.scale,
            has_init_state=False, use_qk_l2norm_in_kernel=True, use_p_times_alpha=True,
        )

        dpacked_h = torch.cat([
            dQ_h.to(comm_dt), dK_h.to(comm_dt), dV_h.to(comm_dt), dP_h.to(comm_dt),
            dLA_h.to(comm_dt).unsqueeze(-1), dLM_h.to(comm_dt).unsqueeze(-1),
            dBeta_h.to(comm_dt).unsqueeze(-1), dEta_h.to(comm_dt).unsqueeze(-1),
        ], dim=-1)
        dpacked_s = _all_to_all_head_to_seq(dpacked_h, world_size)

        c0 = K
        c1 = 2 * K
        c2 = 2 * K + V
        c3 = 3 * K + V
        return (
            dpacked_s[..., :c0],
            dpacked_s[..., c0:c1],
            dpacked_s[..., c1:c2],
            dpacked_s[..., c2:c3],
            dpacked_s[..., c3].to(gate_dt),
            dpacked_s[..., c3 + 1].to(gate_dt),
            dpacked_s[..., c3 + 2].to(gate_dt),
            dpacked_s[..., c3 + 3].to(gate_dt),
            None,
        )


def momentum_deltanet_cp8_step(
    q_s: torch.Tensor,
    k_s: torch.Tensor,
    v_s: torch.Tensor,
    p_s: torch.Tensor,
    la_s: torch.Tensor,
    lm_s: torch.Tensor,
    beta_s: torch.Tensor,
    eta_s: torch.Tensor,
    world_size: int = 8,
) -> torch.Tensor:
    return _MomentumCP8AutogradFn.apply(q_s, k_s, v_s, p_s, la_s, lm_s, beta_s, eta_s, world_size)


# ==================================================================================================
# 7. SM80 Cubin Register & Spill Audit Registry
# ==================================================================================================

ALL_ROUND18_RSI_KERNELS = [
    ("_mdr_triple_l2norm_scale_fwd_kernel", _mdr_triple_l2norm_scale_fwd_kernel),
    ("_mdr_triple_l2norm_scale_bwd_kernel", _mdr_triple_l2norm_scale_bwd_kernel),
    ("_mdr_chunk_gate_greens_fwd_kernel", _mdr_chunk_gate_greens_fwd_kernel),
    ("_mdr_chunk_kkt_fwd_kernel", _mdr_chunk_kkt_fwd_kernel),
    ("_mdr_ut_solve_64x64_kernel", _mdr_ut_solve_64x64_kernel),
    ("_mdr_wy_u_fwd_kernel", _mdr_wy_u_fwd_kernel),
    ("_mdr_wy_yz_fwd_kernel", _mdr_wy_yz_fwd_kernel),
    ("_mdr_chunk_fwd_h_inreg_kernel", _mdr_chunk_fwd_h_inreg_kernel),
    ("_mdr_chunk_fwd_h_l2_kernel", _mdr_chunk_fwd_h_l2_kernel),
    ("_mdr_chunk_fwd_o_kernel", _mdr_chunk_fwd_o_kernel),
    ("_mdr_bwd_dv_intra_kernel", _mdr_bwd_dv_intra_kernel),
    ("_mdr_chunk_bwd_dhu_inreg_kernel", _mdr_chunk_bwd_dhu_inreg_kernel),
    ("_mdr_chunk_bwd_dhu_l2_kernel", _mdr_chunk_bwd_dhu_l2_kernel),
    ("_mdr_bwd_dx_dv_kernel", _mdr_bwd_dx_dv_kernel),
    ("_mdr_bwd_dA_dMqk_kernel", _mdr_bwd_dA_dMqk_kernel),
    ("_mdr_bwd_dq_dk_dp_inter_kernel", _mdr_bwd_dq_dk_dp_inter_kernel),
    ("_mdr_bwd_gate_greens_kernel", _mdr_bwd_gate_greens_kernel),
    ("_mdr_fused_shortconv_silu_fwd_kernel", _mdr_fused_shortconv_silu_fwd_kernel),
    ("_mdr_fused_shortconv_silu_bwd_dx_kernel", _mdr_fused_shortconv_silu_bwd_dx_kernel),
    ("_mdr_fused_shortconv_silu_bwd_dw_kernel", _mdr_fused_shortconv_silu_bwd_dw_kernel),
    ("_mdr_fused_symplectic_gates_fwd_kernel", _mdr_fused_symplectic_gates_fwd_kernel),
    ("_mdr_fused_symplectic_gates_bwd_kernel", _mdr_fused_symplectic_gates_bwd_kernel),
    ("_mdr_fused_output_corr_fwd_kernel", _mdr_fused_output_corr_fwd_kernel),
    ("_mdr_fused_output_corr_bwd_kernel", _mdr_fused_output_corr_bwd_kernel),
    ("_mdr_fused_rmsnorm_gated_fwd_kernel", _mdr_fused_rmsnorm_gated_fwd_kernel),
    ("_mdr_fused_rmsnorm_gated_bwd_kernel", _mdr_fused_rmsnorm_gated_bwd_kernel),
]


def audit_round18_rsi_cubin_spills() -> Dict[str, Any]:
    per_kernel = {}
    total_spills = 0
    max_regs = 0
    total_specs = 0
    compiled_kernels = 0

    for name, kfn in ALL_ROUND18_RSI_KERNELS:
        specs = []
        if hasattr(kfn, "device_caches"):
            for dev_entry in kfn.device_caches.values():
                cache_dict = dev_entry[0] if isinstance(dev_entry, tuple) else dev_entry
                if isinstance(cache_dict, dict):
                    for ck in cache_dict.values():
                        if hasattr(ck, "n_regs") and hasattr(ck, "n_spills"):
                            specs.append({
                                "n_regs": int(ck.n_regs),
                                "n_spills": int(ck.n_spills),
                                "shared_bytes": int(getattr(getattr(ck, "metadata", None), "shared", 0)),
                            })
        k_spills = sum(s["n_spills"] for s in specs)
        k_max_regs = max((s["n_regs"] for s in specs), default=0)
        k_max_smem = max((s["shared_bytes"] for s in specs), default=0)
        per_kernel[name] = {
            "num_specializations": len(specs),
            "max_regs": k_max_regs,
            "total_spills": k_spills,
            "max_shared_bytes": k_max_smem,
        }
        total_spills += k_spills
        max_regs = max(max_regs, k_max_regs)
        total_specs += len(specs)
        if len(specs) > 0:
            compiled_kernels += 1

    return {
        "total_kernels": len(ALL_ROUND18_RSI_KERNELS),
        "compiled_kernels": compiled_kernels,
        "total_specializations": total_specs,
        "total_cubin_spills": total_spills,
        "max_regs": max_regs,
        "per_kernel": per_kernel,
    }


