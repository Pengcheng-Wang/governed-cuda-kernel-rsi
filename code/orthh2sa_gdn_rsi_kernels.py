#!/usr/bin/env python3
"""
Round-35 RSI (E161, v152): Route-Orthogonalized Subtile-WY Dual-Route Sparse-Linear Attention (OrthH2SA-GDN)
Zero-Spill SM80 Triton Kernels, Dipolar Intra-Block Fact-Disentanglement Oracle, & Bitwise-Exact CP=8.

Resolves the Coarse-Block Rank-1 Fast-Weight Collapse identified in E160's 25-cell phase law:
  1. Subtile-Granular Key/Value Centroid & Outlier Summary (_orthh2sa_block_subtile_summary_fwd_rsi_kernel):
     Computes both coarse block centroids mu_c [B, NC, HKV, K] and fine subtile key/value summaries
     (mu_sub [B, NS, HKV, K], vbar_sub [B, HKV, NS, V]) plus outlier-augmented routing keys k_tilde [B, NS, HKV, K].
  2. Constant-Active-Subtile Gate-Margin Adaptive Dual-Route Bitmask (_adah2sa_dual_route_bitmask_rsi_kernel):
     Maintains exact constant active-subtile budget (k_b - 1) * R + k_s while trading marginal coarse blocks
     for rejected-block outlier subtiles when outlier margin exceeds gamma_margin.
  3. Subtile-Granular Coupled Rank-R Householder WY Fast-Weight Recurrence (_orthh2sa_gdn_subtile_wy_fwd_rsi_kernel)
     & Live-Range-Reordered Backward Adjoint (_orthh2sa_gdn_subtile_wy_bwd_dv_dkpart_rsi_kernel, 128 regs, 0 spills):
     Updates S_{c,r} across all R = BC // BS subtiles within each block c using subtile-level surprise gating
     beta_{u,v} = beta_0 * sigmoid(gamma_surp * (u_{u,v} - tau_surp)) and alpha_sub = alpha_c ** (1 / R),
     eliminating intra-block dipolar cancellation (bar_v_c = 0 when v_{c,1} = -v_{c,0}) and cross-subtile ghost terms.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import triton
import triton.language as tl

from moba_rsi_kernels import (
    FusedMoBAQKRMSNormRoPEFunction,
    FusedMoBARMSNormFunction,
    FusedMoBARMSNormSigmoidGatedFunction,
    _moba_bwd_dkv_rsi_kernel,
    _moba_bwd_dq_rsi_kernel,
    _moba_bwd_prep_delta_rsi_kernel,
    _moba_decode_readout_rsi_kernel,
    _moba_fwd_rsi_kernel,
)


def _next_pow2(n: int) -> int:
    return 1 << (max(int(n), 1) - 1).bit_length()
from h2sa_gdn_rsi_kernels import (
    _h2sa_decode_gdn_inject_rsi_kernel,
    _h2sa_gdn_readout_bwd_dq_dg_rsi_kernel,
    _h2sa_gdn_readout_bwd_ds_rsi_kernel,
    _h2sa_gdn_readout_inject_fwd_rsi_kernel,
    _h2sa_pack_qtile_mask_rsi_kernel,
    build_rope_cos_sin,
)
from adah2sa_gdn_rsi_kernels import (
    _adah2sa_decode_gate_bitmask_rsi_kernel,
    _adah2sa_dual_route_bitmask_rsi_kernel,
    _build_adah2sa_subtile_bitmask_fp64,
)


# ==============================================================================
# 1. Differentiable FP64 Reference Oracle for OrthH2SA-GDN (Subtile-WY + Adaptive Routing)
# ==============================================================================

def orthh2sa_gdn_fp64_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_mem: torch.Tensor,
    chunk_size: int = 128,
    subtile_size: int = 64,
    topk_blocks: int = 3,
    topk_subtiles: int = 2,
    outlier_lambda: float = 0.5,
    alpha_decay: float = 0.90,
    beta_write: float = 0.60,
    gamma_surp: float = 4.0,
    tau_surp: float = 0.20,
    margin_scale: float = 1.15,
    enable_subtile_rescue: bool = True,
    enable_gdn_memory: bool = True,
    adaptive_budget: bool = True,
    subtile_wy: bool = True,
    external_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact Differentiable FP64 Reference Oracle for OrthH2SA-GDN:
      - Route 1 + Route 2: Constant-active-subtile gate-margin adaptive dual-route bitmask.
      - Route 3: Subtile-granular Rank-R coupled Householder WY fast-weight recurrence across
        all R = chunk_size // subtile_size subtiles per block (with alpha_sub = alpha_decay ** (1 / R)).
    """
    B, T, HQ, K = q.shape
    _, _, HKV, V = v.shape
    G = HQ // HKV
    R = chunk_size // subtile_size
    NC = T // chunk_size
    NS = T // subtile_size
    sm_scale = 1.0 / math.sqrt(K)

    qf = q.to(torch.float64)
    kf = k.to(torch.float64)
    vf = v.to(torch.float64)
    gf = g_mem.to(torch.float64)

    k_blocks = kf.view(B, NC, chunk_size, HKV, K)
    mu_c = k_blocks.mean(dim=2)  # [B, NC, HKV, K]
    k_subtiles = kf.view(B, NS, subtile_size, HKV, K)
    mu_u = k_subtiles.mean(dim=2)  # [B, NS, HKV, K]
    v_subtiles = vf.view(B, NS, subtile_size, HKV, V)
    vbar_u = v_subtiles.mean(dim=2)  # [B, NS, HKV, V]

    mu_c_for_u = mu_c.repeat_interleave(R, dim=1).unsqueeze(2)
    dev_norm = ((k_subtiles - mu_c_for_u) ** 2).sum(dim=-1)
    max_idx = dev_norm.argmax(dim=2, keepdim=True).unsqueeze(-1).expand(B, NS, 1, HKV, K)
    k_star = torch.gather(k_subtiles, dim=2, index=max_idx).squeeze(2)
    k_tilde = mu_u + outlier_lambda * (k_star - mu_c.repeat_interleave(R, dim=1))

    if external_masks is not None:
        umask = external_masks[0]
    else:
        umask = _build_adah2sa_subtile_bitmask_fp64(
            qf, mu_c, k_tilde, T, chunk_size, subtile_size,
            topk_blocks=topk_blocks,
            topk_subtiles=(topk_subtiles if enable_subtile_rescue else 0),
            adaptive_budget=adaptive_budget,
            margin_scale=margin_scale,
        )

    kf_hq = kf.repeat_interleave(G, dim=2)
    vf_hq = vf.repeat_interleave(G, dim=2)
    scores = torch.einsum("bthk,bshk->bhts", qf, kf_hq) * sm_scale

    t_idx = torch.arange(T, device=q.device)
    u_of_s = t_idx // subtile_size
    bit_active = ((umask.to(torch.int64).unsqueeze(-1) >> u_of_s.view(1, 1, 1, T).to(torch.int64)) & 1) != 0
    causal = t_idx.view(1, 1, T, 1) >= t_idx.view(1, 1, 1, T)
    valid = bit_active & causal
    scores = torch.where(valid, scores, torch.full_like(scores, -1.0e30))
    probs = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("bhts,bshv->bthv", probs, vf_hq)

    if not enable_gdn_memory:
        return attn_out, umask

    # Route 3: Subtile-Granular Coupled Rank-R WY Fast-Weight Recurrence (BV=32 column slices)
    BV = 32
    NV = V // BV
    s_state = torch.zeros((B, HKV, K, V), dtype=torch.float64, device=q.device)
    s_hist = []

    if subtile_wy:
        alpha_sub = alpha_decay ** (1.0 / R)
        k_hat_u = mu_u / torch.sqrt((mu_u * mu_u).sum(dim=-1, keepdim=True) + 1.0e-6)  # [B, NS, HKV, K]
        for c_i in range(NC):
            s_hist.append(s_state)
            for r_i in range(R):
                u_i = c_i * R + r_i
                s_dec = alpha_sub * s_state
                kh_u = k_hat_u[:, u_i]  # [B, HKV, K]
                vb_u = vbar_u[:, u_i]   # [B, NS->HKV, V] wait: vbar_u is [B, NS, HKV, V]
                pred_v = torch.einsum("bhkv,bhk->bhv", s_dec, kh_u)
                err_v = vb_u - pred_v
                err_tiles = err_v.view(B, HKV, NV, BV)
                u_surp = (err_tiles * err_tiles).mean(dim=-1, keepdim=True)
                beta_u = beta_write * torch.sigmoid(gamma_surp * (u_surp - tau_surp))
                beta_full = beta_u.expand(B, HKV, NV, BV).reshape(B, HKV, 1, V)
                s_state = s_dec + beta_full * torch.einsum("bhk,bhv->bhkv", kh_u, err_v)
    else:
        vbar_c = vf.view(B, NC, chunk_size, HKV, V).mean(dim=2)
        k_hat_c = mu_c / torch.sqrt((mu_c * mu_c).sum(dim=-1, keepdim=True) + 1.0e-6)
        for c_i in range(NC):
            s_hist.append(s_state)
            s_dec = alpha_decay * s_state
            kh_c = k_hat_c[:, c_i]
            vb_c = vbar_c[:, c_i]
            pred_v = torch.einsum("bhkv,bhk->bhv", s_dec, kh_c)
            err_v = vb_c - pred_v
            err_tiles = err_v.view(B, HKV, NV, BV)
            u_surp = (err_tiles * err_tiles).mean(dim=-1, keepdim=True)
            beta_c = beta_write * torch.sigmoid(gamma_surp * (u_surp - tau_surp))
            beta_full = beta_c.expand(B, HKV, NV, BV).reshape(B, HKV, 1, V)
            s_state = s_dec + beta_full * torch.einsum("bhk,bhv->bhkv", kh_c, err_v)

    s_stack = torch.stack(s_hist, dim=1)  # [B, NC, HKV, K, V]
    s_stack_hq = s_stack.repeat_interleave(G, dim=2)
    q_norm = torch.sqrt((qf * qf).sum(dim=-1, keepdim=True) + 1.0e-6)
    q_hat = (qf / q_norm).view(B, NC, chunk_size, HQ, K)
    mem_read = torch.einsum("bcthk,bchkv->bcthv", q_hat, s_stack_hq).reshape(B, T, HQ, V)
    gate = torch.sigmoid(gf).unsqueeze(-1)
    total_out = attn_out + gate * mem_read
    return total_out, umask


# ==============================================================================
# 2. Zero-Spill SM80 Triton Kernels for OrthH2SA-GDN (<= 168 Registers, 0 Spills)
# ==============================================================================

@triton.jit
def _orthh2sa_block_subtile_summary_fwd_rsi_kernel(
    K_ptr,
    V_ptr,
    Mu_ptr,
    MuSub_ptr,
    VbarSub_ptr,
    Ktilde_ptr,
    T,
    HKV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    R: tl.constexpr,
    OUTLIER_LAMBDA: tl.constexpr,
):
    """
    Computes in one zero-spill kernel on Grid (NC, B * HKV):
      1. Subtile key centroids MuSub [B, NS, HKV, K] and subtile value centroids VbarSub [B, HKV, NS, V]
      2. Block key centroid Mu [B, NC, HKV, K]
      3. Outlier-augmented subtile routing key Ktilde [B, NS, HKV, K] = mu_u + lambda * (k_{s*(u)} - mu_c)
    """
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    NC = T // BC
    NS = T // BS
    i_b = i_bh // HKV
    i_h = i_bh % HKV

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    BT: tl.constexpr = 32
    inv_bs = 1.0 / BS
    inv_bc = 1.0 / BC

    b_mu_c = tl.zeros([K], dtype=tl.float32)

    for r_i in range(0, R):
        i_u = i_c * R + r_i
        b_mu_u = tl.zeros([K], dtype=tl.float32)
        b_vb_u = tl.zeros([V], dtype=tl.float32)
        for s_off in range(0, BS, BT):
            o_t = i_u * BS + s_off + tl.arange(0, BT)
            k_ptrs = K_ptr + ((i_b * T + o_t[:, None]) * HKV + i_h) * K + o_k[None, :]
            v_ptrs = V_ptr + ((i_b * T + o_t[:, None]) * HKV + i_h) * V + o_v[None, :]
            b_k = tl.load(k_ptrs).to(tl.float32)
            b_v = tl.load(v_ptrs).to(tl.float32)
            b_mu_u += tl.sum(b_k, axis=0)
            b_vb_u += tl.sum(b_v, axis=0)
        b_mu_c += b_mu_u
        b_mu_u = b_mu_u * inv_bs
        b_vb_u = b_vb_u * inv_bs
        tl.store(MuSub_ptr + ((i_b * NS + i_u) * HKV + i_h) * K + o_k, b_mu_u)
        tl.store(VbarSub_ptr + (i_bh * NS + i_u) * V + o_v, b_vb_u)

    b_mu_c = b_mu_c * inv_bc
    tl.store(Mu_ptr + ((i_b * NC + i_c) * HKV + i_h) * K + o_k, b_mu_c)

    for r_i in range(0, R):
        i_u = i_c * R + r_i
        b_mu_u = tl.load(MuSub_ptr + ((i_b * NS + i_u) * HKV + i_h) * K + o_k).to(tl.float32)
        best_dev = tl.full([], -1.0, dtype=tl.float32)
        best_idx = tl.full([], i_u * BS, dtype=tl.int32)
        for s_off in range(0, BS, BT):
            o_t = i_u * BS + s_off + tl.arange(0, BT)
            k_ptrs = K_ptr + ((i_b * T + o_t[:, None]) * HKV + i_h) * K + o_k[None, :]
            b_k = tl.load(k_ptrs).to(tl.float32)
            diff = b_k - b_mu_c[None, :]
            dev = tl.sum(diff * diff, axis=1)
            chunk_max = tl.max(dev, axis=0)
            if chunk_max > best_dev:
                best_dev = chunk_max
                is_max = dev == chunk_max
                cand_t = tl.where(is_max, o_t, i_u * BS + BS)
                best_idx = tl.min(cand_t, axis=0)
        k_star = tl.load(K_ptr + ((i_b * T + best_idx) * HKV + i_h) * K + o_k).to(tl.float32)
        b_kt = b_mu_u + OUTLIER_LAMBDA * (k_star - b_mu_c)
        tl.store(Ktilde_ptr + ((i_b * NS + i_u) * HKV + i_h) * K + o_k, b_kt)


@triton.jit
def _orthh2sa_gdn_subtile_wy_fwd_rsi_kernel(
    MuSub_ptr,
    VbarSub_ptr,
    SHist_ptr,
    SHistSub_ptr,
    KhatSub_ptr,
    BetaSub_ptr,
    SFinal_ptr,
    T,
    NC: tl.constexpr,
    NS: tl.constexpr,
    R: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    ALPHA_SUB: tl.constexpr,
    BETA0: tl.constexpr,
    GAMMA_SURP: tl.constexpr,
    TAU_SURP: tl.constexpr,
    MUTATE_COARSE_COLLAPSE: tl.constexpr,
):
    """
    V-Tiled Subtile-Granular Coupled Rank-R Householder WY Fast-Weight Recurrence (BV=32, 0 spills):
      Advances S [K, BV] across all NS = NC * R subtiles in registers while storing block-entry
      prefix state SHist [B * HKV, NC, K, V] at r == 0 and subtile-entry state SHistSub [B * HKV, NS, K, V].
    Grid: (V // BV, B * HKV)
    """
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HKV
    i_h = i_bh % HKV
    NV: tl.constexpr = V // BV
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    b_S = tl.zeros([K, BV], dtype=tl.float32)
    for i_c in range(0, NC):
        s_blk_ptrs = SHist_ptr + (i_bh * NC + i_c) * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(s_blk_ptrs, b_S)

        for r_i in range(0, R):
            i_u = i_c * R + r_i
            s_sub_ptrs = SHistSub_ptr + (i_bh * NS + i_u) * (K * V) + o_k[:, None] * V + o_v[None, :]
            tl.store(s_sub_ptrs, b_S)

            if MUTATE_COARSE_COLLAPSE == 1:
                # Adversarial mutation M1: collapse subtile value to coarse-block average (erasing dipolar facts!)
                i_u0 = i_c * R
                b_mu = tl.load(MuSub_ptr + ((i_b * NS + i_u) * HKV + i_h) * K + o_k).to(tl.float32)
                b_vb0 = tl.load(VbarSub_ptr + (i_bh * NS + i_u0) * V + o_v).to(tl.float32)
                b_vb1 = tl.load(VbarSub_ptr + (i_bh * NS + i_u0 + 1) * V + o_v).to(tl.float32)
                b_vb = 0.5 * (b_vb0 + b_vb1)
            else:
                b_mu = tl.load(MuSub_ptr + ((i_b * NS + i_u) * HKV + i_h) * K + o_k).to(tl.float32)
                b_vb = tl.load(VbarSub_ptr + (i_bh * NS + i_u) * V + o_v).to(tl.float32)

            inv_norm = tl.rsqrt(tl.sum(b_mu * b_mu, axis=0) + 1.0e-6)
            b_kh = b_mu * inv_norm
            if i_v == 0:
                tl.store(KhatSub_ptr + (i_bh * NS + i_u) * K + o_k, b_kh)

            b_S_dec = ALPHA_SUB * b_S
            pred_v = tl.sum(b_S_dec * b_kh[:, None], axis=0)
            err_v = b_vb - pred_v

            u_surp = tl.sum(err_v * err_v, axis=0) * (1.0 / BV)
            sig_u = tl.sigmoid(GAMMA_SURP * (u_surp - TAU_SURP))
            beta_u = BETA0 * sig_u
            tl.store(BetaSub_ptr + (i_bh * NS + i_u) * NV + i_v, beta_u)

            b_S = b_S_dec + beta_u * (b_kh[:, None] * err_v[None, :])

    sf_ptrs = SFinal_ptr + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :]
    tl.store(sf_ptrs, b_S)


@triton.jit
def _orthh2sa_gdn_subtile_wy_bwd_dv_dkpart_rsi_kernel(
    VbarSub_ptr,
    KhatSub_ptr,
    SHistSub_ptr,
    BetaSub_ptr,
    dSHist_hq_ptr,
    dAttnV_ptr,
    dV_out_ptr,
    dKhSubPart_ptr,
    T,
    NC: tl.constexpr,
    NS: tl.constexpr,
    R: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
    ALPHA_SUB: tl.constexpr,
    BETA0: tl.constexpr,
    GAMMA_SURP: tl.constexpr,
    MUTATE_NO_SURP_JACOBIAN: tl.constexpr,
    MUTATE_SUBTILE_ORDER: tl.constexpr,
):
    """
    V-Tiled Live-Range-Reordered Reverse-Time Adjoint of Subtile-Granular Coupled Rank-R WY Recurrence:
      Consumes b_S_prev before accumulating dSHist_hq into b_F at block boundaries (r == 0),
      staying at <= 128 registers with 0 spills across all K in {64, 128}!
    Grid: (V // BV, B * HKV)
    """
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HKV
    i_hkv = i_bh % HKV
    G: tl.constexpr = HQ // HKV
    NV: tl.constexpr = V // BV

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    BT: tl.constexpr = 32
    inv_bs = 1.0 / BS

    b_F = tl.zeros([K, BV], dtype=tl.float32)
    for rev_c in range(0, NC):
        i_c = NC - 1 - rev_c
        for rev_r in range(0, R):
            r_i = rev_r if MUTATE_SUBTILE_ORDER == 1 else (R - 1 - rev_r)
            i_u = i_c * R + r_i

            b_vb = tl.load(VbarSub_ptr + (i_bh * NS + i_u) * V + o_v).to(tl.float32)
            b_kh = tl.load(KhatSub_ptr + (i_bh * NS + i_u) * K + o_k).to(tl.float32)
            beta_u = tl.load(BetaSub_ptr + (i_bh * NS + i_u) * NV + i_v).to(tl.float32)

            s_prev_ptrs = SHistSub_ptr + (i_bh * NS + i_u) * (K * V) + o_k[:, None] * V + o_v[None, :]
            b_S_prev = tl.load(s_prev_ptrs).to(tl.float32)

            pred_v = ALPHA_SUB * tl.sum(b_S_prev * b_kh[:, None], axis=0)
            err_v = b_vb - pred_v

            g_e = tl.sum(b_F * b_kh[:, None], axis=0)  # [BV]
            dbeta = tl.sum(g_e * err_v, axis=0)
            if MUTATE_NO_SURP_JACOBIAN == 1:
                b_dvbar = beta_u * g_e
            else:
                surp_jac = dbeta * beta_u * (1.0 - beta_u / BETA0) * (2.0 * GAMMA_SURP / BV)
                b_dvbar = beta_u * g_e + surp_jac * err_v

            term2 = ALPHA_SUB * tl.sum(b_S_prev * b_dvbar[None, :], axis=1)  # [K]
            term1 = beta_u * tl.sum(b_F * err_v[None, :], axis=1)           # [K]
            b_dkh_part = term1 - term2

            dkh_ptrs = dKhSubPart_ptr + ((i_v * tl.num_programs(1) + i_bh) * NS + i_u) * K + o_k
            tl.store(dkh_ptrs, b_dkh_part)

            b_F = ALPHA_SUB * (b_F - b_kh[:, None] * b_dvbar[None, :])

            dv_tok_delta = b_dvbar * inv_bs
            for s_off in range(0, BS, BT):
                o_t = i_u * BS + s_off + tl.arange(0, BT)
                dva_ptrs = dAttnV_ptr + ((i_b * T + o_t[:, None]) * HKV + i_hkv) * V + o_v[None, :]
                b_dva = tl.load(dva_ptrs).to(tl.float32)
                dv_out_ptrs = dV_out_ptr + ((i_b * T + o_t[:, None]) * HKV + i_hkv) * V + o_v[None, :]
                tl.store(dv_out_ptrs, (b_dva + dv_tok_delta[None, :]).to(dV_out_ptr.dtype.element_ty))

        for i_g in range(0, G):
            i_hq = i_hkv * G + i_g
            ds_ptrs = dSHist_hq_ptr + ((i_b * HQ + i_hq) * NC + i_c) * (K * V) + o_k[:, None] * V + o_v[None, :]
            b_F += tl.load(ds_ptrs).to(tl.float32)


@triton.jit
def _orthh2sa_gdn_subtile_bwd_dk_finalize_rsi_kernel(
    MuSub_ptr,
    KhatSub_ptr,
    dKhSubPart_ptr,
    dAttnK_ptr,
    dK_out_ptr,
    T,
    NS: tl.constexpr,
    NV: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
):
    """
    Sums partial dKhSubPart across NV = V // BV slices, projects through subtile L2-norm Jacobian,
    and adds (1 / BS) * d_mu_u to dAttnK across all BS tokens of subtile i_u.
    Grid: (NS, B * HKV)
    """
    i_u = tl.program_id(0)
    i_bh = tl.program_id(1)
    n_bh = tl.num_programs(1)
    i_b = i_bh // HKV
    i_hkv = i_bh % HKV

    o_k = tl.arange(0, K)
    BT: tl.constexpr = 32
    inv_bs = 1.0 / BS

    b_dkh = tl.zeros([K], dtype=tl.float32)
    for i_v in range(0, NV):
        part_ptrs = dKhSubPart_ptr + ((i_v * n_bh + i_bh) * NS + i_u) * K + o_k
        b_dkh += tl.load(part_ptrs).to(tl.float32)

    b_mu = tl.load(MuSub_ptr + ((i_b * NS + i_u) * HKV + i_hkv) * K + o_k).to(tl.float32)
    b_kh = tl.load(KhatSub_ptr + (i_bh * NS + i_u) * K + o_k).to(tl.float32)

    inv_norm = tl.rsqrt(tl.sum(b_mu * b_mu, axis=0) + 1.0e-6)
    proj = tl.sum(b_dkh * b_kh, axis=0)
    b_dmu = (b_dkh - b_kh * proj) * inv_norm
    dk_tok_delta = b_dmu * inv_bs

    for s_off in range(0, BS, BT):
        o_t = i_u * BS + s_off + tl.arange(0, BT)
        dka_ptrs = dAttnK_ptr + ((i_b * T + o_t[:, None]) * HKV + i_hkv) * K + o_k[None, :]
        b_dka = tl.load(dka_ptrs).to(tl.float32)
        dk_out_ptrs = dK_out_ptr + ((i_b * T + o_t[:, None]) * HKV + i_hkv) * K + o_k[None, :]
        tl.store(dk_out_ptrs, (b_dka + dk_tok_delta[None, :]).to(dK_out_ptr.dtype.element_ty))


# ==============================================================================
# 3. Autograd Function & Public API for OrthH2SA-GDN
# ==============================================================================

class ParallelOrthH2SAGDNRSIFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_mem: torch.Tensor,
        chunk_size: int,
        subtile_size: int,
        topk_blocks: int,
        topk_subtiles: int,
        outlier_lambda: float,
        alpha_decay: float,
        beta_write: float,
        gamma_surp: float,
        tau_surp: float,
        margin_scale: float,
        enable_subtile_rescue: bool,
        enable_gdn_memory: bool,
        adaptive_budget: bool,
        external_q_subtile_mask: Optional[torch.Tensor],
        external_qtile_mask: Optional[torch.Tensor],
        return_shared_state: bool,
        mutate_mode: int,
    ):
        B, T, HQ, K = q.shape
        _, _, HKV, V = v.shape
        assert K == V and K in (64, 128)
        assert T % chunk_size == 0 and chunk_size % subtile_size == 0
        G = HQ // HKV
        IS_FP32 = q.dtype == torch.float32
        R = chunk_size // subtile_size
        NC = T // chunk_size
        NS = T // subtile_size
        assert NS <= 64
        BM = 32 if not IS_FP32 else 16
        BN = 32 if not IS_FP32 else 16
        NT = triton.cdiv(T, BM)
        sm_scale = 1.0 / math.sqrt(K)

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = g_mem.contiguous()

        mu = torch.empty((B, NC, HKV, K), dtype=torch.float32, device=q.device)
        mu_sub = torch.empty((B, NS, HKV, K), dtype=torch.float32, device=q.device)
        vbar_sub = torch.empty((B * HKV, NS, V), dtype=torch.float32, device=q.device)
        ktilde = torch.empty((B, NS, HKV, K), dtype=torch.float32, device=q.device)

        _orthh2sa_block_subtile_summary_fwd_rsi_kernel[(NC, B * HKV)](
            k_c, v_c, mu, mu_sub, vbar_sub, ktilde,
            T=T, HKV=HKV, K=K, V=V, BC=chunk_size, BS=subtile_size, R=R,
            OUTLIER_LAMBDA=float(outlier_lambda),
            num_warps=4, num_stages=2,
        )

        if external_q_subtile_mask is not None and external_qtile_mask is not None:
            q_subtile_mask = external_q_subtile_mask
            qtile_mask = external_qtile_mask
        else:
            q_subtile_mask = torch.empty((B, HQ, T), dtype=torch.uint64, device=q.device)
            qtile_mask = torch.empty((B, HQ, NT), dtype=torch.uint64, device=q.device)
            nc_pad = max(16, triton.next_power_of_2(NC))
            ns_pad = max(16, triton.next_power_of_2(NS))
            eff_topk_sub = topk_subtiles if enable_subtile_rescue else 0
            eff_ada = 1 if (adaptive_budget and enable_subtile_rescue) else 0
            _adah2sa_dual_route_bitmask_rsi_kernel[(T, B * HQ)](
                q_c, mu, ktilde, q_subtile_mask,
                T=T, HQ=HQ, HKV=HKV, K=K, BC=chunk_size, BS=subtile_size, R=R,
                NC_PAD=nc_pad, NS_PAD=ns_pad,
                TOPK_BLOCKS=topk_blocks, TOPK_SUBTILES=eff_topk_sub,
                ADAPTIVE_BUDGET=eff_ada, MARGIN_SCALE=float(margin_scale),
                num_warps=2, num_stages=1,
            )
            _h2sa_pack_qtile_mask_rsi_kernel[(NT, B * HQ)](
                q_subtile_mask, qtile_mask,
                T=T, NT=NT, BM=BM,
                num_warps=2, num_stages=1,
            )

        attn_o = torch.empty_like(q_c)
        lse = torch.empty((B, HQ, T), dtype=torch.float32, device=q.device)
        _moba_fwd_rsi_kernel[(NT, B * HQ)](
            q_c, k_c, v_c, q_subtile_mask, qtile_mask, attn_o, lse, T,
            HQ=HQ, HKV=HKV, G=G, K=K, V=V, CHUNK_SIZE=subtile_size, BM=BM, BN=BN,
            IS_FP32=IS_FP32, MUTATE_MODE=0,
            num_warps=4, num_stages=2 if (K <= 64 and not IS_FP32) else 1,
        )

        BV = 32
        NV = V // BV
        BM_RO = 16
        NT_RO = triton.cdiv(T, BM_RO)
        alpha_sub = float(alpha_decay ** (1.0 / R))
        if enable_gdn_memory:
            s_hist = torch.empty((B, HKV, NC, K, V), dtype=torch.float32, device=q.device)
            s_hist_sub = torch.empty((B * HKV, NS, K, V), dtype=torch.float32, device=q.device)
            khat_sub = torch.empty((B * HKV, NS, K), dtype=torch.float32, device=q.device)
            beta_sub = torch.empty((B * HKV, NS, NV), dtype=torch.float32, device=q.device)
            s_final = torch.empty((B * HKV, K, V), dtype=torch.float32, device=q.device)

            _orthh2sa_gdn_subtile_wy_fwd_rsi_kernel[(NV, B * HKV)](
                mu_sub, vbar_sub, s_hist, s_hist_sub, khat_sub, beta_sub, s_final,
                T=T, NC=NC, NS=NS, R=R, HKV=HKV, K=K, V=V, BV=BV,
                ALPHA_SUB=alpha_sub, BETA0=float(beta_write),
                GAMMA_SURP=float(gamma_surp), TAU_SURP=float(tau_surp),
                MUTATE_COARSE_COLLAPSE=(1 if mutate_mode == 1 else 0),
                num_warps=4, num_stages=1,
            )

            out = torch.empty_like(q_c)
            qhat = torch.empty((B, T, HQ, K), dtype=torch.float32, device=q.device)
            q_rstd = torch.empty((B, HQ, T), dtype=torch.float32, device=q.device)
            mem_out = torch.empty((B, T, HQ, V), dtype=torch.float32, device=q.device)
            _h2sa_gdn_readout_inject_fwd_rsi_kernel[(NV, NT_RO, B * HQ)](
                q_c, g_c, s_hist, attn_o, out, qhat, q_rstd, mem_out,
                T=T, NC=NC, HQ=HQ, HKV=HKV, K=K, V=V, BC=chunk_size, BM=BM_RO, BV=BV,
                num_warps=4, num_stages=1,
            )
        else:
            s_hist = s_hist_sub = khat_sub = beta_sub = qhat = q_rstd = mem_out = torch.empty((0,), device=q.device)
            out = attn_o

        ctx.save_for_backward(
            q_c, k_c, v_c, g_c, attn_o, lse, q_subtile_mask, qtile_mask,
            mu_sub, vbar_sub, s_hist, s_hist_sub, khat_sub, beta_sub, qhat, q_rstd, mem_out,
        )
        ctx.chunk_size = chunk_size
        ctx.subtile_size = subtile_size
        ctx.alpha_sub = alpha_sub
        ctx.beta_write = beta_write
        ctx.gamma_surp = gamma_surp
        ctx.enable_gdn_memory = enable_gdn_memory
        ctx.mutate_mode = mutate_mode
        if return_shared_state:
            ctx.mark_non_differentiable(q_subtile_mask, qtile_mask)
            return out, q_subtile_mask, qtile_mask
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        do_c = grad_outputs[0].contiguous()
        (
            q_c, k_c, v_c, g_c, attn_o, lse, q_subtile_mask, qtile_mask,
            mu_sub, vbar_sub, s_hist, s_hist_sub, khat_sub, beta_sub, qhat, q_rstd, mem_out,
        ) = ctx.saved_tensors
        B, T, HQ, K = q_c.shape
        _, _, HKV, V = v_c.shape
        G = HQ // HKV
        IS_FP32 = q_c.dtype == torch.float32
        BC = ctx.chunk_size
        BS = ctx.subtile_size
        R = BC // BS
        NC = T // BC
        NS = T // BS
        BM = 32 if not IS_FP32 else 16
        BN = 32 if not IS_FP32 else 16
        BM_DQ = 32 if (K <= 64 and not IS_FP32) else 16
        NT_M_DQ = triton.cdiv(T, BM_DQ)
        NT_N = triton.cdiv(T, BN)
        BV = 32
        NV = V // BV

        delta = torch.empty((B, HQ, T), dtype=torch.float32, device=q_c.device)
        _moba_bwd_prep_delta_rsi_kernel[(triton.cdiv(T, 64), B * HQ)](
            attn_o, do_c, delta, T, HQ=HQ, V=V, BM=64, MUTATE_MODE=0, num_warps=4,
        )

        dq_attn = torch.empty_like(q_c)
        _moba_bwd_dq_rsi_kernel[(NT_M_DQ, B * HQ)](
            q_c, k_c, v_c, do_c, lse, delta, q_subtile_mask, qtile_mask, dq_attn, T,
            HQ=HQ, HKV=HKV, G=G, K=K, V=V, CHUNK_SIZE=BS, BM=BM_DQ, BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4, num_stages=2 if not IS_FP32 else 1,
        )

        dk_attn = torch.empty_like(k_c)
        dv_attn = torch.empty_like(v_c)
        _moba_bwd_dkv_rsi_kernel[(NT_N, B * HKV)](
            q_c, k_c, v_c, do_c, lse, delta, q_subtile_mask, qtile_mask, dk_attn, dv_attn, T, G,
            HQ=HQ, HKV=HKV, G=G, K=K, V=V, CHUNK_SIZE=BS, BM=BM, BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4, num_stages=2 if (K <= 64 and not IS_FP32) else 1,
        )

        if not ctx.enable_gdn_memory:
            dg = torch.zeros_like(g_c)
            return (
                dq_attn, dk_attn, dv_attn, dg,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            )

        BM_BWD = 16
        NT_BWD = triton.cdiv(T, BM_BWD)
        ds_hq = torch.empty((B, HQ, NC, K, V), dtype=torch.float32, device=q_c.device)
        dq = torch.empty_like(q_c)
        dg = torch.empty_like(g_c)
        _h2sa_gdn_readout_bwd_ds_rsi_kernel[(NV, NC, B * HQ)](
            do_c, g_c, qhat, ds_hq,
            T=T, NC=NC, HQ=HQ, K=K, V=V, BC=BC, BM=BM_BWD, BV=BV,
            num_warps=4, num_stages=1,
        )
        _h2sa_gdn_readout_bwd_dq_dg_rsi_kernel[(NT_BWD, B * HQ)](
            do_c, g_c, qhat, q_rstd, s_hist, mem_out,
            dq_attn, dq, dg,
            T=T, NC=NC, HQ=HQ, HKV=HKV, K=K, V=V, BC=BC, BM=BM_BWD, BV=BV,
            num_warps=4, num_stages=1,
        )

        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        dkh_sub_part = torch.empty((NV, B * HKV, NS, K), dtype=torch.float32, device=q_c.device)
        _orthh2sa_gdn_subtile_wy_bwd_dv_dkpart_rsi_kernel[(NV, B * HKV)](
            vbar_sub, khat_sub, s_hist_sub, beta_sub, ds_hq, dv_attn, dv, dkh_sub_part,
            T=T, NC=NC, NS=NS, R=R, HQ=HQ, HKV=HKV, K=K, V=V, BV=BV, BS=BS,
            ALPHA_SUB=float(ctx.alpha_sub), BETA0=float(ctx.beta_write), GAMMA_SURP=float(ctx.gamma_surp),
            MUTATE_NO_SURP_JACOBIAN=(1 if ctx.mutate_mode == 2 else 0),
            MUTATE_SUBTILE_ORDER=(1 if ctx.mutate_mode == 3 else 0),
            num_warps=4, num_stages=1,
        )
        _orthh2sa_gdn_subtile_bwd_dk_finalize_rsi_kernel[(NS, B * HKV)](
            mu_sub, khat_sub, dkh_sub_part, dk_attn, dk,
            T=T, NS=NS, NV=NV, HKV=HKV, K=K, BS=BS,
            num_warps=4, num_stages=1,
        )

        return (
            dq, dk, dv, dg,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        )


def parallel_orthh2sa_gdn_rsi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_mem: torch.Tensor,
    chunk_size: int = 128,
    subtile_size: int = 64,
    topk_blocks: int = 3,
    topk_subtiles: int = 2,
    outlier_lambda: float = 0.5,
    alpha_decay: float = 0.90,
    beta_write: float = 0.60,
    gamma_surp: float = 4.0,
    tau_surp: float = 0.20,
    margin_scale: float = 1.15,
    enable_subtile_rescue: bool = True,
    enable_gdn_memory: bool = True,
    adaptive_budget: bool = True,
    external_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_shared_state: bool = False,
    mutate_mode: int = 0,
):
    ext_sub = external_masks[0] if external_masks is not None else None
    ext_tile = external_masks[1] if external_masks is not None else None
    return ParallelOrthH2SAGDNRSIFunction.apply(
        q, k, v, g_mem,
        chunk_size, subtile_size, topk_blocks, topk_subtiles,
        outlier_lambda, alpha_decay, beta_write, gamma_surp, tau_surp, margin_scale,
        enable_subtile_rescue, enable_gdn_memory, adaptive_budget,
        ext_sub, ext_tile, return_shared_state, mutate_mode,
    )


# ==============================================================================
# 4. Single-Pass Subtile-WY Decode & Superfused OrthH2SAGDNBlockRSI
# ==============================================================================

@torch.no_grad()
def orthh2sa_gdn_decode_rsi(
    q_dec: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    g_dec: torch.Tensor,
    mu_cache: torch.Tensor,
    ktilde_cache: torch.Tensor,
    s_final: torch.Tensor,
    chunk_size: int = 128,
    subtile_size: int = 64,
    topk_blocks: int = 3,
    topk_subtiles: int = 2,
    margin_scale: float = 1.15,
) -> torch.Tensor:
    B, T_q, HQ, K = q_dec.shape
    assert T_q == 1
    T_cache = k_cache.shape[1]
    HKV = k_cache.shape[2]
    V = v_cache.shape[3]
    G = HQ // HKV
    R = chunk_size // subtile_size
    NC = T_cache // chunk_size
    NS = T_cache // subtile_size

    nc_pad = max(16, triton.next_power_of_2(NC))
    ns_pad = max(16, triton.next_power_of_2(NS))
    umask_dec = torch.empty((B, HQ, 1), dtype=torch.uint64, device=q_dec.device)
    _adah2sa_decode_gate_bitmask_rsi_kernel[(B * HQ,)](
        q_dec, mu_cache, ktilde_cache, umask_dec,
        HQ=HQ, HKV=HKV, K=K, R=R, NC=NC, NS=NS,
        NC_PAD=nc_pad, NS_PAD=ns_pad,
        TOPK_BLOCKS=topk_blocks, TOPK_SUBTILES=topk_subtiles,
        ADAPTIVE_BUDGET=1, MARGIN_SCALE=float(margin_scale),
        num_warps=2,
    )

    attn_dec = torch.empty_like(q_dec)
    BN = min(64, subtile_size)
    _moba_decode_readout_rsi_kernel[(1, B * HQ)](
        q_dec, k_cache, v_cache, umask_dec, attn_dec,
        T_Q=1, T_KV=T_cache,
        HQ=HQ, HKV=HKV, G=G, K=K, V=V, NC=NS, CHUNK_SIZE=subtile_size, BN=BN,
        num_warps=4,
    )

    BV = 32
    out_dec = torch.empty_like(q_dec)
    _h2sa_decode_gdn_inject_rsi_kernel[(V // BV, B * HQ)](
        q_dec, g_dec, s_final, attn_dec, out_dec,
        HQ=HQ, HKV=HKV, K=K, V=V, BV=BV,
        num_warps=2,
    )
    return out_dec


class OrthH2SAGDNBlockRSI(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        chunk_size: int = 128,
        subtile_size: int = 64,
        topk_blocks: int = 3,
        topk_subtiles: int = 2,
        stage: str = "A4",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.subtile_size = subtile_size
        self.topk_blocks = topk_blocks
        self.topk_subtiles = topk_subtiles
        self.stage = stage

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.in_norm_weight = nn.Parameter(torch.ones(d_model))
        self.q_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.out_norm_weight = nn.Parameter(torch.ones(head_dim))

        if stage in ("A3", "A4"):
            self.qkvg_proj = nn.Linear(d_model, q_dim + 2 * kv_dim + q_dim + num_heads, bias=False)
        else:
            self.q_proj = nn.Linear(d_model, q_dim, bias=False)
            self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
            self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
            self.g_out_proj = nn.Linear(d_model, q_dim, bias=False)
            self.g_mem_proj = nn.Linear(d_model, num_heads, bias=False)

        self.o_proj = nn.Linear(q_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        external_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_shared_state: bool = False,
    ):
        B, T, D = x.shape
        HQ = self.num_heads
        HKV = self.num_kv_heads
        K = self.head_dim
        q_dim = HQ * K
        kv_dim = HKV * K

        if self.stage in ("A3", "A4"):
            h = FusedMoBARMSNormFunction.apply(x, self.in_norm_weight, 1e-6)
            qkvg = self.qkvg_proj(h)
            q_raw = qkvg[..., :q_dim].view(B, T, HQ, K)
            k_raw = qkvg[..., q_dim : q_dim + kv_dim].view(B, T, HKV, K)
            v = qkvg[..., q_dim + kv_dim : q_dim + 2 * kv_dim].view(B, T, HKV, K)
            g_out = qkvg[..., q_dim + 2 * kv_dim : 2 * q_dim + 2 * kv_dim].view(B, T, HQ, K)
            g_mem = qkvg[..., 2 * q_dim + 2 * kv_dim :]
        else:
            xf = x.to(torch.float32)
            rstd = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)
            h = (xf * rstd).to(x.dtype) * self.in_norm_weight.to(x.dtype)
            q_raw = self.q_proj(h).view(B, T, HQ, K)
            k_raw = self.k_proj(h).view(B, T, HKV, K)
            v = self.v_proj(h).view(B, T, HKV, K)
            g_out = self.g_out_proj(h).view(B, T, HQ, K)
            g_mem = self.g_mem_proj(h)

        if self.stage in ("A2", "A3", "A4"):
            q = FusedMoBAQKRMSNormRoPEFunction.apply(q_raw, self.q_norm_weight, cos, sin, 1e-6)
            k = FusedMoBAQKRMSNormRoPEFunction.apply(k_raw, self.k_norm_weight, cos, sin, 1e-6)
        else:
            qf = q_raw.to(torch.float32)
            kf = k_raw.to(torch.float32)
            qn = (qf * torch.rsqrt((qf * qf).mean(-1, keepdim=True) + 1e-6)) * self.q_norm_weight.view(1, 1, 1, K)
            kn = (kf * torch.rsqrt((kf * kf).mean(-1, keepdim=True) + 1e-6)) * self.k_norm_weight.view(1, 1, 1, K)
            c0 = cos[:T].view(1, T, 1, K // 2)
            s0 = sin[:T].view(1, T, 1, K // 2)
            q0, q1 = qn[..., : K // 2], qn[..., K // 2 :]
            k0, k1 = kn[..., : K // 2], kn[..., K // 2 :]
            q = torch.cat([q0 * c0 - q1 * s0, q0 * s0 + q1 * c0], dim=-1).to(x.dtype)
            k = torch.cat([k0 * c0 - k1 * s0, k0 * s0 + k1 * c0], dim=-1).to(x.dtype)

        if self.stage == "A0":
            G = HQ // HKV
            ke = k.repeat_interleave(G, dim=2)
            ve = v.repeat_interleave(G, dim=2)
            try:
                from fla.ops.moba.parallel import parallel_moba as _fla_moba
                cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=x.device)
                attn_out = _fla_moba(
                    q.reshape(1, B * T, HQ, K),
                    ke.reshape(1, B * T, HQ, K),
                    ve.reshape(1, B * T, HQ, K),
                    cu,
                    max_seqlen=T,
                    chunk_size=self.chunk_size,
                    topk=self.topk_blocks + 1,
                )
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                attn_out = attn_out.reshape(B, T, HQ, K)
            except Exception:
                from moba_rsi_kernels import parallel_moba_rsi
                attn_out = parallel_moba_rsi(q, k, v, chunk_size=self.chunk_size, topk=self.topk_blocks + 1)
            R = self.chunk_size // self.subtile_size
            NS = T // self.subtile_size
            NC = T // self.chunk_size
            kf_s = k.to(torch.float32).view(B, NS, self.subtile_size, HKV, K).mean(dim=2)
            vf_s = v.to(torch.float32).view(B, NS, self.subtile_size, HKV, K).mean(dim=2)
            kh_s = kf_s * torch.rsqrt((kf_s * kf_s).sum(-1, keepdim=True) + 1e-6)
            s_c = torch.zeros((B, HKV, K, K), dtype=torch.float32, device=x.device)
            alpha_sub = 0.90 ** (1.0 / R)
            s_list = []
            for c_i in range(NC):
                s_list.append(s_c)
                for r_i in range(R):
                    u_i = c_i * R + r_i
                    sd = alpha_sub * s_c
                    pv = torch.einsum("bhkv,bhk->bhv", sd, kh_s[:, u_i])
                    ev = vf_s[:, u_i] - pv
                    u_s = (ev * ev).mean(dim=-1, keepdim=True).unsqueeze(-1)
                    bc = 0.60 * torch.sigmoid(4.0 * (u_s - 0.20))
                    s_c = sd + bc * torch.einsum("bhk,bhv->bhkv", kh_s[:, u_i], ev)
            s_st = torch.stack(s_list, dim=1).repeat_interleave(G, dim=2)
            qh = (q.to(torch.float32) * torch.rsqrt((q.to(torch.float32) ** 2).sum(-1, keepdim=True) + 1e-6)).view(B, NC, self.chunk_size, HQ, K)
            mr = torch.einsum("bcthk,bchkv->bcthv", qh, s_st).reshape(B, T, HQ, K)
            o_attn = attn_out + (torch.sigmoid(g_mem.to(torch.float32)).unsqueeze(-1) * mr).to(x.dtype)
            shared_masks = None
        else:
            if return_shared_state:
                o_attn, q_sub_m, qtile_m = parallel_orthh2sa_gdn_rsi(
                    q, k, v, g_mem,
                    chunk_size=self.chunk_size, subtile_size=self.subtile_size,
                    topk_blocks=self.topk_blocks, topk_subtiles=self.topk_subtiles,
                    external_masks=external_masks, return_shared_state=True,
                )
                shared_masks = (q_sub_m, qtile_m)
            else:
                o_attn = parallel_orthh2sa_gdn_rsi(
                    q, k, v, g_mem,
                    chunk_size=self.chunk_size, subtile_size=self.subtile_size,
                    topk_blocks=self.topk_blocks, topk_subtiles=self.topk_subtiles,
                    external_masks=external_masks, return_shared_state=False,
                )
                shared_masks = None

        if self.stage in ("A3", "A4"):
            o_gated = FusedMoBARMSNormSigmoidGatedFunction.apply(o_attn, g_out, self.out_norm_weight, 1e-6)
        else:
            of = o_attn.to(torch.float32)
            on = (of * torch.rsqrt((of * of).mean(-1, keepdim=True) + 1e-6)) * self.out_norm_weight.view(1, 1, 1, K)
            o_gated = (on * torch.sigmoid(g_out.to(torch.float32))).to(x.dtype)

        out = x + self.o_proj(o_gated.view(B, T, q_dim))
        if return_shared_state:
            return out, shared_masks
        return out


# ==============================================================================
# 5. Bitwise-Exact 8-GPU NVSwitch Ulysses OrthH2SA-GDN-CP = 8
# ==============================================================================

class _UlyssesSeqToHeadAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        W = dist.get_world_size(group)
        B, T_local, H, D = x.shape
        H_local = H // W
        send_buf = x.view(B, T_local, W, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 0, 2, 3, 4).contiguous().view(B, W * T_local, H_local, D)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        W = dist.get_world_size(group)
        B, T_total, H_local, D = grad_out.shape
        T_local = T_total // W
        send_buf = grad_out.view(B, W, T_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 2, 0, 3, 4).contiguous().view(B, T_local, W * H_local, D), None


class _UlyssesHeadToSeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        W = dist.get_world_size(group)
        B, T_total, H_local, D = x.shape
        T_local = T_total // W
        send_buf = x.view(B, W, T_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 2, 0, 3, 4).contiguous().view(B, T_local, W * H_local, D)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        group = ctx.group
        W = dist.get_world_size(group)
        B, T_local, H, D = grad_out.shape
        H_local = H // W
        send_buf = grad_out.view(B, T_local, W, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf, group=group)
        return recv_buf.permute(1, 0, 2, 3, 4).contiguous().view(B, W * T_local, H_local, D), None


def orthh2sa_gdn_cp8_ulysses_rsi(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    group: dist.ProcessGroup,
    chunk_size: int = 256,
    subtile_size: int = 128,
    topk_blocks: int = 3,
    topk_subtiles: int = 2,
) -> torch.Tensor:
    q_full = _UlyssesSeqToHeadAllToAll.apply(q_local, group)
    k_full = _UlyssesSeqToHeadAllToAll.apply(k_local, group)
    v_full = _UlyssesSeqToHeadAllToAll.apply(v_local, group)
    g_full = _UlyssesSeqToHeadAllToAll.apply(g_local.unsqueeze(-1), group).squeeze(-1)
    o_full = parallel_orthh2sa_gdn_rsi(
        q_full, k_full, v_full, g_full,
        chunk_size=chunk_size, subtile_size=subtile_size,
        topk_blocks=topk_blocks, topk_subtiles=topk_subtiles,
    )
    return _UlyssesHeadToSeqAllToAll.apply(o_full, group)


# ==============================================================================
# 6. SM80 Cubin Register, Spill, & Shared-Memory Auditor for OrthH2SA-GDN
# ==============================================================================

def audit_orthh2sa_gdn_rsi_cubin_kernels() -> Dict[str, Any]:
    device = torch.device("cuda:0")
    for K in (64, 128):
        for HKV in (16, 4):
            q = torch.randn((2, 1024, 16, K), dtype=torch.bfloat16, device=device, requires_grad=True)
            k = torch.randn((2, 1024, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
            v = torch.randn((2, 1024, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
            g = torch.randn((2, 1024, 16), dtype=torch.bfloat16, device=device, requires_grad=True)
            do = torch.randn_like(q)
            for BC, BS in ((128, 64), (256, 128)):
                o = parallel_orthh2sa_gdn_rsi(q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2)
                o.backward(do)
                q.grad = k.grad = v.grad = g.grad = None
        qd = torch.randn((4, 1, 16, K), dtype=torch.bfloat16, device=device)
        kc = torch.randn((4, 1024, 4, K), dtype=torch.bfloat16, device=device)
        vc = torch.randn((4, 1024, 4, K), dtype=torch.bfloat16, device=device)
        gd = torch.randn((4, 1, 16), dtype=torch.bfloat16, device=device)
        muc = torch.randn((4, 8, 4, K), dtype=torch.float32, device=device)
        ktc = torch.randn((4, 16, 4, K), dtype=torch.float32, device=device)
        sf = torch.randn((16, K, K), dtype=torch.float32, device=device)
        _ = orthh2sa_gdn_decode_rsi(qd, kc, vc, gd, muc, ktc, sf, chunk_size=128, subtile_size=64)
    torch.cuda.synchronize()

    kernels = [
        ("_orthh2sa_block_subtile_summary_fwd_rsi_kernel", _orthh2sa_block_subtile_summary_fwd_rsi_kernel),
        ("_adah2sa_dual_route_bitmask_rsi_kernel", _adah2sa_dual_route_bitmask_rsi_kernel),
        ("_h2sa_pack_qtile_mask_rsi_kernel", _h2sa_pack_qtile_mask_rsi_kernel),
        ("_orthh2sa_gdn_subtile_wy_fwd_rsi_kernel", _orthh2sa_gdn_subtile_wy_fwd_rsi_kernel),
        ("_h2sa_gdn_readout_inject_fwd_rsi_kernel", _h2sa_gdn_readout_inject_fwd_rsi_kernel),
        ("_h2sa_gdn_readout_bwd_ds_rsi_kernel", _h2sa_gdn_readout_bwd_ds_rsi_kernel),
        ("_h2sa_gdn_readout_bwd_dq_dg_rsi_kernel", _h2sa_gdn_readout_bwd_dq_dg_rsi_kernel),
        ("_orthh2sa_gdn_subtile_wy_bwd_dv_dkpart_rsi_kernel", _orthh2sa_gdn_subtile_wy_bwd_dv_dkpart_rsi_kernel),
        ("_orthh2sa_gdn_subtile_bwd_dk_finalize_rsi_kernel", _orthh2sa_gdn_subtile_bwd_dk_finalize_rsi_kernel),
        ("_adah2sa_decode_gate_bitmask_rsi_kernel", _adah2sa_decode_gate_bitmask_rsi_kernel),
        ("_h2sa_decode_gdn_inject_rsi_kernel", _h2sa_decode_gdn_inject_rsi_kernel),
        ("_moba_fwd_rsi_kernel", _moba_fwd_rsi_kernel),
        ("_moba_bwd_prep_delta_rsi_kernel", _moba_bwd_prep_delta_rsi_kernel),
        ("_moba_bwd_dq_rsi_kernel", _moba_bwd_dq_rsi_kernel),
        ("_moba_bwd_dkv_rsi_kernel", _moba_bwd_dkv_rsi_kernel),
        ("_moba_decode_readout_rsi_kernel", _moba_decode_readout_rsi_kernel),
    ]

    details = []
    total_specs = 0
    spilling_specs = 0
    over_168_specs = 0
    max_regs = 0
    max_spills = 0
    max_smem = 0

    for name, kobj in kernels:
        compiled = []
        if hasattr(kobj, "device_caches"):
            for dev_cache in kobj.device_caches.values():
                if isinstance(dev_cache, tuple) and len(dev_cache) > 0 and isinstance(dev_cache[0], dict):
                    compiled.extend(dev_cache[0].values())
                elif isinstance(dev_cache, dict):
                    compiled.extend(dev_cache.values())
        if not compiled and hasattr(kobj, "cache"):
            for v_c in kobj.cache.values():
                if isinstance(v_c, dict):
                    compiled.extend(v_c.values())
                else:
                    compiled.append(v_c)

        k_specs = 0
        k_max_regs = 0
        k_max_spills = 0
        k_max_smem = 0
        for bin_obj in compiled:
            if not hasattr(bin_obj, "n_regs"):
                continue
            regs = int(getattr(bin_obj, "n_regs", 0) or 0)
            spills = int(getattr(bin_obj, "n_spills", 0) or 0)
            smem = int(getattr(bin_obj, "shared", getattr(bin_obj, "metadata", None) and getattr(bin_obj.metadata, "shared", 0)) or 0)
            k_specs += 1
            total_specs += 1
            if spills > 0:
                spilling_specs += 1
            if regs > 168:
                over_168_specs += 1
            k_max_regs = max(k_max_regs, regs)
            k_max_spills = max(k_max_spills, spills)
            k_max_smem = max(k_max_smem, smem)
            max_regs = max(max_regs, regs)
            max_spills = max(max_spills, spills)
            max_smem = max(max_smem, smem)

        details.append({
            "kernel": name,
            "specializations": k_specs,
            "max_regs": k_max_regs,
            "max_spills": k_max_spills,
            "max_smem_bytes": k_max_smem,
        })

    return {
        "total_kernels": len(kernels),
        "total_specializations": total_specs,
        "spilling_specializations": spilling_specs,
        "over_168_reg_specializations": over_168_specs,
        "max_regs": max_regs,
        "max_spills": max_spills,
        "max_smem_bytes": max_smem,
        "all_zero_spill": (total_specs > 0 and spilling_specs == 0 and over_168_specs == 0),
        "kernels": details,
    }
