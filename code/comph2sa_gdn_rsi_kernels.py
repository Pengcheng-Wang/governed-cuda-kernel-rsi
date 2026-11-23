#!/usr/bin/env python3
"""
Round-37 RSI (E163, v154): Subspace-Complementary Orthogonal-Query Dual-Route Sparse-Linear Attention (CompH2SA-GDN)
Zero-Spill SM80 Triton Kernels, 4-Regime Compositional Duality Oracle, & Bitwise-Exact CP=8.

Resolves the Compositional Multi-Hop (Selected-Hit + Rejected-Miss) Interference in E159..E162:
  1. Selected-Route Subspace Orthogonalization:
     For each query token t in block c_q with active historical subtile bitmask A_t^{hist}, computes the
     positive-affinity-weighted selected-route key anchor:
       mu_sel,t = stop_grad( normalize( sum_{u in A_t^{hist}} [q_t . k_hat_u]_+ * k_hat_u ) )
     and projects q_t onto the orthogonal complement of mu_sel,t before querying fast-weight memory S_{c_q}:
       q_tilde_t = q_t - lambda_proj * (q_t . mu_sel,t) * mu_sel,t,   q_hat_perp,t = normalize(q_tilde_t)
  2. Unexplained-Query-Energy Modulated LSE Calibration:
     Computes the unexplained orthogonal query energy ratio rho_perp,t = ||q_tilde_t||_2 / ||q_t||_2 in [0, 1]
     and modulates LSE suppression:
       g_cal,t = g_t - eta_lse * (stop_grad(L_t) - tau_lse) * max(0, 1 - eta_perp * stop_grad(rho_perp,t))
       m_hat_t = sigmoid(g_cal,t)
     - Pure Selected-Hit (q_t || k_sel): rho_perp,t -> 0, L_t >> tau_lse -> m_hat_t -> 0 AND q_tilde_t -> 0.
     - Pure Rejected-Miss (q_t || k_rej _|_ mu_sel,t): rho_perp,t -> 1, L_t << tau_lse -> m_hat_t -> 1.
     - Compositional Hit+Miss (q_t = k_sel + k_rej): rho_perp,t ~ 0.707 >= 1/eta_perp lifts LSE suppression
       while q_hat_perp,t ~ k_rej queries S_{c_q} strictly along k_rej without triggering poison on k_sel!
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
from h2sa_gdn_rsi_kernels import (
    _h2sa_decode_gdn_inject_rsi_kernel,
    _h2sa_gdn_readout_bwd_dq_dg_rsi_kernel,
    _h2sa_gdn_readout_bwd_ds_rsi_kernel,
    _h2sa_pack_qtile_mask_rsi_kernel,
    build_rope_cos_sin,
)
from adah2sa_gdn_rsi_kernels import (
    _adah2sa_decode_gate_bitmask_rsi_kernel,
    _adah2sa_dual_route_bitmask_rsi_kernel,
    _build_adah2sa_subtile_bitmask_fp64,
)
from orthh2sa_gdn_rsi_kernels import (
    _orthh2sa_block_subtile_summary_fwd_rsi_kernel,
    _orthh2sa_gdn_subtile_bwd_dk_finalize_rsi_kernel,
    _orthh2sa_gdn_subtile_wy_bwd_dv_dkpart_rsi_kernel,
    _orthh2sa_gdn_subtile_wy_fwd_rsi_kernel,
)
from calh2sa_gdn_rsi_kernels import (
    _calh2sa_gdn_readout_inject_fwd_rsi_kernel,
    _calh2sa_scale_do_sel_rsi_kernel,
)


# ==============================================================================
# 1. Differentiable FP64 Reference Oracle for CompH2SA-GDN
# ==============================================================================

def comph2sa_gdn_fp64_oracle(
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
    eta_lse: float = 0.50,
    tau_lse: float = 4.0,
    rho_center: float = 0.25,
    lambda_proj: float = 1.00,
    eta_perp: float = 2.00,
    enable_subtile_rescue: bool = True,
    enable_gdn_memory: bool = True,
    adaptive_budget: bool = True,
    external_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact Differentiable FP64 Reference Oracle for CompH2SA-GDN:
      - Selected-Route Subspace Orthogonalization: q_tilde_t = q_t - lambda_proj * (q_t . mu_sel,t) * mu_sel,t
      - Unexplained-Energy Modulated LSE Gate:
          g_cal,t = g_t - eta_lse * (sg(L_t) - tau_lse) * max(0, 1 - eta_perp * sg(rho_perp,t))
      - Centered Residual Readout:
          y_t = o_sel,t + sigmoid(g_cal,t) * (S_{c_q}^T q_hat_perp,t - rho_center * o_sel,t)
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
    mu_c = k_blocks.mean(dim=2)
    k_subtiles = kf.view(B, NS, subtile_size, HKV, K)
    mu_u = k_subtiles.mean(dim=2)
    v_subtiles = vf.view(B, NS, subtile_size, HKV, V)
    vbar_u = v_subtiles.mean(dim=2)

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
    scores_masked = torch.where(valid, scores, torch.full_like(scores, -1.0e30))

    max_sc = scores_masked.max(dim=-1, keepdim=True).values
    lse_bht = (max_sc + torch.log(torch.exp(scores_masked - max_sc).sum(dim=-1, keepdim=True))).squeeze(-1)
    lse_bth = lse_bht.permute(0, 2, 1).detach()

    probs = torch.softmax(scores_masked, dim=-1)
    attn_out = torch.einsum("bhts,bshv->bthv", probs, vf_hq)

    if not enable_gdn_memory:
        return attn_out, umask

    BV = 32
    NV = V // BV
    s_state = torch.zeros((B, HKV, K, V), dtype=torch.float64, device=q.device)
    s_hist = []
    alpha_sub = alpha_decay ** (1.0 / R)
    k_hat_u = mu_u / torch.sqrt((mu_u * mu_u).sum(dim=-1, keepdim=True) + 1.0e-6)  # [B, NS, HKV, K]

    for c_i in range(NC):
        s_hist.append(s_state)
        for r_i in range(R):
            u_i = c_i * R + r_i
            s_dec = alpha_sub * s_state
            kh_u = k_hat_u[:, u_i]
            vb_u = vbar_u[:, u_i]
            pred_v = torch.einsum("bhkv,bhk->bhv", s_dec, kh_u)
            err_v = vb_u - pred_v
            err_tiles = err_v.view(B, HKV, NV, BV)
            u_surp = (err_tiles * err_tiles).mean(dim=-1, keepdim=True)
            beta_u = beta_write * torch.sigmoid(gamma_surp * (u_surp - tau_surp))
            beta_full = beta_u.expand(B, HKV, NV, BV).reshape(B, HKV, 1, V)
            s_state = s_dec + beta_full * torch.einsum("bhk,bhv->bhkv", kh_u, err_v)

    # Selected-Route Subspace Orthogonalization (stop_grad on mu_sel)
    k_hat_hq = k_hat_u.repeat_interleave(G, dim=2).detach()  # [B, NS, HQ, K]
    qf_det = qf.detach()
    aff = torch.einsum("bthk,buhk->bhtu", qf_det, k_hat_hq).clamp(min=0.0)  # [B, HQ, T, NS]
    u_idx = torch.arange(NS, device=q.device).view(1, 1, 1, NS)
    c_of_t = (t_idx // chunk_size).view(1, 1, T, 1)
    hist_sub = u_idx < (c_of_t * R)
    umask_sub = (((umask.to(torch.int64).unsqueeze(-1) >> u_idx.to(torch.int64)) & 1) != 0) & hist_sub
    aff_masked = torch.where(umask_sub, aff, torch.zeros_like(aff))  # [B, HQ, T, NS]
    mu_acc = torch.einsum("bhtu,buhk->bthk", aff_masked, k_hat_hq)  # [B, T, HQ, K]
    mu_sel = (mu_acc / torch.sqrt((mu_acc * mu_acc).sum(dim=-1, keepdim=True) + 1.0e-12)).detach()

    proj_coeff = (qf * mu_sel).sum(dim=-1, keepdim=True)
    q_tilde = qf - lambda_proj * proj_coeff * mu_sel
    q_tilde_norm = torch.sqrt((q_tilde * q_tilde).sum(dim=-1, keepdim=True) + 1.0e-6)
    q_hat_perp = (q_tilde / q_tilde_norm).view(B, NC, chunk_size, HQ, K)

    q_orig_norm = torch.sqrt((qf_det * qf_det).sum(dim=-1) + 1.0e-6)
    rho_perp = (q_tilde_norm.squeeze(-1).detach() / q_orig_norm).clamp(min=0.0, max=1.0)
    mod_factor = (1.0 - eta_perp * rho_perp).clamp(min=0.0)

    s_stack = torch.stack(s_hist, dim=1).repeat_interleave(G, dim=2)
    mem_read = torch.einsum("bcthk,bchkv->bcthv", q_hat_perp, s_stack).reshape(B, T, HQ, V)

    g_cal = gf - eta_lse * (lse_bth - tau_lse) * mod_factor
    m_hat = torch.sigmoid(g_cal).unsqueeze(-1)
    res_mem = mem_read - rho_center * attn_out
    total_out = attn_out + m_hat * res_mem
    return total_out, umask


# ==============================================================================
# 2. Zero-Spill SM80 Triton Kernels for CompH2SA-GDN (<= 168 Registers, 0 Spills)
# ==============================================================================

@triton.jit
def _comph2sa_ortho_query_prep_fwd_rsi_kernel(
    Q_ptr,
    KhatSub_ptr,
    QSubMask_ptr,
    GMem_ptr,
    LSE_ptr,
    MuSel_ptr,
    QTilde_ptr,
    GCal_ptr,
    Mhat_ptr,
    T,
    NS: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    ETA_LSE: tl.constexpr,
    TAU_LSE: tl.constexpr,
    LAMBDA_PROJ: tl.constexpr,
    ETA_PERP: tl.constexpr,
    MUTATE_NO_ORTHO_PROJ: tl.constexpr,
    MUTATE_NO_PERP_MOD: tl.constexpr,
):
    """
    Fused 1D Per-(Token, Head) Selected-Route Subspace Orthogonalization & Unexplained-Energy LSE Gate Prep:
      1. Accumulates mu_acc = sum_{u < c_q * R, u in A_t} max(0, q_t . k_hat_u) * k_hat_u
      2. Normalizes mu_sel = mu_acc / sqrt(||mu_acc||_2^2 + 1e-12)
      3. Projects q_tilde_t = q_t - LAMBDA_PROJ * (q_t . mu_sel) * mu_sel
      4. Modulates LSE gate with rho_perp = ||q_tilde_t||_2 / ||q_t||_2
    Grid: (T, B * HQ) -- uses < 48 registers and 0 spills!
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    G: tl.constexpr = HQ // HKV
    i_hkv = i_hq // G
    i_bhkv = i_b * HKV + i_hkv
    i_c = i_t // BC
    hist_lim = i_c * R

    o_k = tl.arange(0, K)
    q_ptrs = Q_ptr + ((i_b * T + i_t) * HQ + i_hq) * K + o_k
    b_q = tl.load(q_ptrs).to(tl.float32)

    umask = tl.load(QSubMask_ptr + i_bh * T + i_t)
    b_mu_acc = tl.zeros([K], dtype=tl.float32)

    for u_i in range(0, NS):
        if u_i < hist_lim:
            bit_on = ((umask >> u_i.to(tl.uint64)) & 1) != 0
            if bit_on:
                kh_ptrs = KhatSub_ptr + (i_bhkv * NS + u_i) * K + o_k
                b_kh = tl.load(kh_ptrs).to(tl.float32)
                aff = tl.maximum(0.0, tl.sum(b_q * b_kh, axis=0))
                b_mu_acc += aff * b_kh

    mu_norm_inv = tl.rsqrt(tl.sum(b_mu_acc * b_mu_acc, axis=0) + 1.0e-12)
    b_mu_sel = b_mu_acc * mu_norm_inv
    tl.store(MuSel_ptr + ((i_b * T + i_t) * HQ + i_hq) * K + o_k, b_mu_sel)

    if MUTATE_NO_ORTHO_PROJ == 1:
        b_q_tilde = b_q
    else:
        proj = tl.sum(b_q * b_mu_sel, axis=0)
        b_q_tilde = b_q - LAMBDA_PROJ * proj * b_mu_sel
    tl.store(QTilde_ptr + ((i_b * T + i_t) * HQ + i_hq) * K + o_k, b_q_tilde)

    q_norm = tl.sqrt(tl.sum(b_q * b_q, axis=0) + 1.0e-6)
    qt_norm = tl.sqrt(tl.sum(b_q_tilde * b_q_tilde, axis=0) + 1.0e-6)
    rho_perp = tl.minimum(1.0, tl.maximum(0.0, qt_norm / q_norm))

    if MUTATE_NO_PERP_MOD == 1:
        mod_factor = 1.0
    else:
        mod_factor = tl.maximum(0.0, 1.0 - ETA_PERP * rho_perp)

    b_g = tl.load(GMem_ptr + (i_b * T + i_t) * HQ + i_hq).to(tl.float32)
    b_lse = tl.load(LSE_ptr + i_bh * T + i_t).to(tl.float32) * 0.6931471805599453
    b_g_cal = b_g - ETA_LSE * (b_lse - TAU_LSE) * mod_factor
    b_sig = tl.sigmoid(b_g_cal)

    tl.store(GCal_ptr + (i_b * T + i_t) * HQ + i_hq, b_g_cal)
    tl.store(Mhat_ptr + i_bh * T + i_t, b_sig)


@triton.jit
def _comph2sa_ortho_query_bwd_dq_rsi_kernel(
    dQTilde_ptr,
    MuSel_ptr,
    dAttnQ_ptr,
    dQOut_ptr,
    T,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    LAMBDA_PROJ: tl.constexpr,
    MUTATE_NO_ORTHO_PROJ: tl.constexpr,
):
    """
    Propagates dQTilde through the symmetric subspace projector I - LAMBDA_PROJ * mu_sel * mu_sel^T
    and adds dAttnQ to form the total query gradient dQOut [B, T, HQ, K].
    Grid: (T // BM, B * HQ) -- uses < 48 registers and 0 spills!
    """
    i_m = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ

    o_m = i_m * BM + tl.arange(0, BM)
    o_k = tl.arange(0, K)
    offs = ((i_b * T + o_m[:, None]) * HQ + i_hq) * K + o_k[None, :]
    mask = o_m[:, None] < T

    b_dqt = tl.load(dQTilde_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b_dqa = tl.load(dAttnQ_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if MUTATE_NO_ORTHO_PROJ == 1:
        b_dq_mem = b_dqt
    else:
        b_mu = tl.load(MuSel_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        proj = tl.sum(b_dqt * b_mu, axis=1)
        b_dq_mem = b_dqt - LAMBDA_PROJ * proj[:, None] * b_mu

    tl.store(dQOut_ptr + offs, (b_dqa + b_dq_mem).to(dQOut_ptr.dtype.element_ty), mask=mask)


# ==============================================================================
# 3. Autograd Function & Public API for CompH2SA-GDN
# ==============================================================================

class ParallelCompH2SAGDNRSIFunction(torch.autograd.Function):
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
        eta_lse: float,
        tau_lse: float,
        rho_center: float,
        lambda_proj: float,
        eta_perp: float,
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
                MUTATE_COARSE_COLLAPSE=0,
                num_warps=4, num_stages=1,
            )

            mu_sel = torch.empty((B, T, HQ, K), dtype=torch.float32, device=q.device)
            q_tilde = torch.empty((B, T, HQ, K), dtype=torch.float32, device=q.device)
            gcal = torch.empty((B, T, HQ), dtype=torch.float32, device=q.device)
            mhat = torch.empty((B, HQ, T), dtype=torch.float32, device=q.device)

            _comph2sa_ortho_query_prep_fwd_rsi_kernel[(T, B * HQ)](
                q_c, khat_sub, q_subtile_mask, g_c, lse,
                mu_sel, q_tilde, gcal, mhat,
                T=T, NS=NS, R=R, BC=chunk_size, HQ=HQ, HKV=HKV, K=K,
                ETA_LSE=float(eta_lse), TAU_LSE=float(tau_lse),
                LAMBDA_PROJ=float(lambda_proj), ETA_PERP=float(eta_perp),
                MUTATE_NO_ORTHO_PROJ=(1 if mutate_mode == 1 else 0),
                MUTATE_NO_PERP_MOD=(1 if mutate_mode == 2 else 0),
                num_warps=2, num_stages=1,
            )

            out = torch.empty_like(q_c)
            qhat = torch.empty((B, T, HQ, K), dtype=torch.float32, device=q.device)
            q_rstd = torch.empty((B, HQ, T), dtype=torch.float32, device=q.device)
            mem_out = torch.empty((B, T, HQ, V), dtype=torch.float32, device=q.device)
            _calh2sa_gdn_readout_inject_fwd_rsi_kernel[(NV, NT_RO, B * HQ)](
                q_tilde, mhat, s_hist, attn_o, out, qhat, q_rstd, mem_out,
                T=T, NC=NC, HQ=HQ, HKV=HKV, K=K, V=V, BC=chunk_size, BM=BM_RO, BV=BV,
                RHO_CENTER=float(rho_center),
                num_warps=4, num_stages=1,
            )
        else:
            s_hist = s_hist_sub = khat_sub = beta_sub = mu_sel = qhat = q_rstd = mem_out = gcal = mhat = torch.empty((0,), device=q.device)
            out = attn_o

        ctx.save_for_backward(
            q_c, k_c, v_c, g_c, attn_o, lse, q_subtile_mask, qtile_mask,
            mu_sub, vbar_sub, s_hist, s_hist_sub, khat_sub, beta_sub, mu_sel, qhat, q_rstd, mem_out, gcal, mhat,
        )
        ctx.chunk_size = chunk_size
        ctx.subtile_size = subtile_size
        ctx.alpha_sub = alpha_sub
        ctx.beta_write = beta_write
        ctx.gamma_surp = gamma_surp
        ctx.rho_center = rho_center
        ctx.lambda_proj = lambda_proj
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
            mu_sub, vbar_sub, s_hist, s_hist_sub, khat_sub, beta_sub, mu_sel, qhat, q_rstd, mem_out, gcal, mhat,
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

        if ctx.enable_gdn_memory:
            do_sel = torch.empty_like(do_c)
            _calh2sa_scale_do_sel_rsi_kernel[(triton.cdiv(T, 32), B * HQ)](
                do_c, mhat, do_sel,
                T=T, HQ=HQ, V=V, BM=32,
                RHO_CENTER=float(ctx.rho_center),
                MUTATE_NO_CENTER_BWD=(1 if ctx.mutate_mode == 3 else 0),
                num_warps=4, num_stages=1,
            )
        else:
            do_sel = do_c

        delta = torch.empty((B, HQ, T), dtype=torch.float32, device=q_c.device)
        _moba_bwd_prep_delta_rsi_kernel[(triton.cdiv(T, 64), B * HQ)](
            attn_o, do_sel, delta, T, HQ=HQ, V=V, BM=64, MUTATE_MODE=0, num_warps=4,
        )

        dq_attn = torch.empty_like(q_c)
        _moba_bwd_dq_rsi_kernel[(NT_M_DQ, B * HQ)](
            q_c, k_c, v_c, do_sel, lse, delta, q_subtile_mask, qtile_mask, dq_attn, T,
            HQ=HQ, HKV=HKV, G=G, K=K, V=V, CHUNK_SIZE=BS, BM=BM_DQ, BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4, num_stages=2 if not IS_FP32 else 1,
        )

        dk_attn = torch.empty_like(k_c)
        dv_attn = torch.empty_like(v_c)
        _moba_bwd_dkv_rsi_kernel[(NT_N, B * HKV)](
            q_c, k_c, v_c, do_sel, lse, delta, q_subtile_mask, qtile_mask, dk_attn, dv_attn, T, G,
            HQ=HQ, HKV=HKV, G=G, K=K, V=V, CHUNK_SIZE=BS, BM=BM, BN=BN,
            IS_FP32=IS_FP32,
            num_warps=4, num_stages=2 if (K <= 64 and not IS_FP32) else 1,
        )

        if not ctx.enable_gdn_memory:
            dg = torch.zeros_like(g_c)
            return (
                dq_attn, dk_attn, dv_attn, dg,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            )

        BM_BWD = 16
        NT_BWD = triton.cdiv(T, BM_BWD)
        ds_hq = torch.empty((B, HQ, NC, K, V), dtype=torch.float32, device=q_c.device)
        zero_q = torch.zeros((B, T, HQ, K), dtype=torch.float32, device=q_c.device)
        dq_tilde = torch.empty((B, T, HQ, K), dtype=torch.float32, device=q_c.device)
        dg = torch.empty_like(g_c)

        _h2sa_gdn_readout_bwd_ds_rsi_kernel[(NV, NC, B * HQ)](
            do_c, gcal, qhat, ds_hq,
            T=T, NC=NC, HQ=HQ, K=K, V=V, BC=BC, BM=BM_BWD, BV=BV,
            num_warps=4, num_stages=1,
        )
        _h2sa_gdn_readout_bwd_dq_dg_rsi_kernel[(NT_BWD, B * HQ)](
            do_c, gcal, qhat, q_rstd, s_hist, mem_out,
            zero_q, dq_tilde, dg,
            T=T, NC=NC, HQ=HQ, HKV=HKV, K=K, V=V, BC=BC, BM=BM_BWD, BV=BV,
            num_warps=4, num_stages=1,
        )

        dq = torch.empty_like(q_c)
        _comph2sa_ortho_query_bwd_dq_rsi_kernel[(triton.cdiv(T, 16), B * HQ)](
            dq_tilde, mu_sel, dq_attn, dq,
            T=T, HQ=HQ, K=K, BM=16,
            LAMBDA_PROJ=float(ctx.lambda_proj),
            MUTATE_NO_ORTHO_PROJ=(1 if ctx.mutate_mode == 1 else 0),
            num_warps=4, num_stages=1,
        )

        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        dkh_sub_part = torch.empty((NV, B * HKV, NS, K), dtype=torch.float32, device=q_c.device)
        _orthh2sa_gdn_subtile_wy_bwd_dv_dkpart_rsi_kernel[(NV, B * HKV)](
            vbar_sub, khat_sub, s_hist_sub, beta_sub, ds_hq, dv_attn, dv, dkh_sub_part,
            T=T, NC=NC, NS=NS, R=R, HQ=HQ, HKV=HKV, K=K, V=V, BV=BV, BS=BS,
            ALPHA_SUB=float(ctx.alpha_sub), BETA0=float(ctx.beta_write), GAMMA_SURP=float(ctx.gamma_surp),
            MUTATE_NO_SURP_JACOBIAN=0,
            MUTATE_SUBTILE_ORDER=0,
            num_warps=4, num_stages=1,
        )
        _orthh2sa_gdn_subtile_bwd_dk_finalize_rsi_kernel[(NS, B * HKV)](
            mu_sub, khat_sub, dkh_sub_part, dk_attn, dk,
            T=T, NS=NS, NV=NV, HKV=HKV, K=K, BS=BS,
            num_warps=4, num_stages=1,
        )

        return (
            dq, dk, dv, dg,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        )


def parallel_comph2sa_gdn_rsi(
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
    eta_lse: float = 0.50,
    tau_lse: float = 4.0,
    rho_center: float = 0.25,
    lambda_proj: float = 1.00,
    eta_perp: float = 2.00,
    enable_subtile_rescue: bool = True,
    enable_gdn_memory: bool = True,
    adaptive_budget: bool = True,
    external_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_shared_state: bool = False,
    mutate_mode: int = 0,
):
    ext_sub = external_masks[0] if external_masks is not None else None
    ext_tile = external_masks[1] if external_masks is not None else None
    return ParallelCompH2SAGDNRSIFunction.apply(
        q, k, v, g_mem,
        chunk_size, subtile_size, topk_blocks, topk_subtiles,
        outlier_lambda, alpha_decay, beta_write, gamma_surp, tau_surp, margin_scale,
        eta_lse, tau_lse, rho_center, lambda_proj, eta_perp,
        enable_subtile_rescue, enable_gdn_memory, adaptive_budget,
        ext_sub, ext_tile, return_shared_state, mutate_mode,
    )


# ==============================================================================
# 4. Single-Pass Subtile-WY Decode & Superfused CompH2SAGDNBlockRSI
# ==============================================================================

@torch.no_grad()
def comph2sa_gdn_decode_rsi(
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


class CompH2SAGDNBlockRSI(nn.Module):
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
            o_attn = attn_out + (torch.sigmoid(g_mem.to(torch.float32)).unsqueeze(-1) * (mr - 0.25 * attn_out.to(torch.float32))).to(x.dtype)
            shared_masks = None
        else:
            if return_shared_state:
                o_attn, q_sub_m, qtile_m = parallel_comph2sa_gdn_rsi(
                    q, k, v, g_mem,
                    chunk_size=self.chunk_size, subtile_size=self.subtile_size,
                    topk_blocks=self.topk_blocks, topk_subtiles=self.topk_subtiles,
                    external_masks=external_masks, return_shared_state=True,
                )
                shared_masks = (q_sub_m, qtile_m)
            else:
                o_attn = parallel_comph2sa_gdn_rsi(
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
# 5. Bitwise-Exact 8-GPU NVSwitch Ulysses CompH2SA-GDN-CP = 8
# ==============================================================================

from calh2sa_gdn_rsi_kernels import _UlyssesHeadToSeqAllToAll, _UlyssesSeqToHeadAllToAll


def comph2sa_gdn_cp8_ulysses_rsi(
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
    o_full = parallel_comph2sa_gdn_rsi(
        q_full, k_full, v_full, g_full,
        chunk_size=chunk_size, subtile_size=subtile_size,
        topk_blocks=topk_blocks, topk_subtiles=topk_subtiles,
    )
    return _UlyssesHeadToSeqAllToAll.apply(o_full, group)


# ==============================================================================
# 6. SM80 Cubin Register, Spill, & Shared-Memory Auditor for CompH2SA-GDN
# ==============================================================================

def audit_comph2sa_gdn_rsi_cubin_kernels() -> Dict[str, Any]:
    device = torch.device("cuda:0")
    for K in (64, 128):
        for HKV in (16, 4):
            q = torch.randn((2, 1024, 16, K), dtype=torch.bfloat16, device=device, requires_grad=True)
            k = torch.randn((2, 1024, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
            v = torch.randn((2, 1024, HKV, K), dtype=torch.bfloat16, device=device, requires_grad=True)
            g = torch.randn((2, 1024, 16), dtype=torch.bfloat16, device=device, requires_grad=True)
            do = torch.randn_like(q)
            for BC, BS in ((128, 64), (256, 128)):
                o = parallel_comph2sa_gdn_rsi(q, k, v, g, chunk_size=BC, subtile_size=BS, topk_blocks=3, topk_subtiles=2)
                o.backward(do)
                q.grad = k.grad = v.grad = g.grad = None
        qd = torch.randn((4, 1, 16, K), dtype=torch.bfloat16, device=device)
        kc = torch.randn((4, 1024, 4, K), dtype=torch.bfloat16, device=device)
        vc = torch.randn((4, 1024, 4, K), dtype=torch.bfloat16, device=device)
        gd = torch.randn((4, 1, 16), dtype=torch.bfloat16, device=device)
        muc = torch.randn((4, 8, 4, K), dtype=torch.float32, device=device)
        ktc = torch.randn((4, 16, 4, K), dtype=torch.float32, device=device)
        sf = torch.randn((16, K, K), dtype=torch.float32, device=device)
        _ = comph2sa_gdn_decode_rsi(qd, kc, vc, gd, muc, ktc, sf, chunk_size=128, subtile_size=64)
    torch.cuda.synchronize()

    kernels = [
        ("_orthh2sa_block_subtile_summary_fwd_rsi_kernel", _orthh2sa_block_subtile_summary_fwd_rsi_kernel),
        ("_adah2sa_dual_route_bitmask_rsi_kernel", _adah2sa_dual_route_bitmask_rsi_kernel),
        ("_h2sa_pack_qtile_mask_rsi_kernel", _h2sa_pack_qtile_mask_rsi_kernel),
        ("_orthh2sa_gdn_subtile_wy_fwd_rsi_kernel", _orthh2sa_gdn_subtile_wy_fwd_rsi_kernel),
        ("_comph2sa_ortho_query_prep_fwd_rsi_kernel", _comph2sa_ortho_query_prep_fwd_rsi_kernel),
        ("_calh2sa_gdn_readout_inject_fwd_rsi_kernel", _calh2sa_gdn_readout_inject_fwd_rsi_kernel),
        ("_calh2sa_scale_do_sel_rsi_kernel", _calh2sa_scale_do_sel_rsi_kernel),
        ("_h2sa_gdn_readout_bwd_ds_rsi_kernel", _h2sa_gdn_readout_bwd_ds_rsi_kernel),
        ("_h2sa_gdn_readout_bwd_dq_dg_rsi_kernel", _h2sa_gdn_readout_bwd_dq_dg_rsi_kernel),
        ("_comph2sa_ortho_query_bwd_dq_rsi_kernel", _comph2sa_ortho_query_bwd_dq_rsi_kernel),
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
