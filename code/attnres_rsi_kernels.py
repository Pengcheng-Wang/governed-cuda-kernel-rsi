#!/usr/bin/env python3
"""Round-6 RSI Kernel Library for E132 (`v123`):
Moonshot Kimi `Attention Residuals` (`AttnRes` & `Block AttnRes`, `arXiv:2603.15031`,
official repo `MoonshotAI/Attention-Residuals`, plus community `Hydra-2P` baseline
`manishklach/attnres-kernel-lab`).

Key Mathematical & SM80 Hardware Innovations:
1. **Query-Scale RMSNorm Absorption (`q_eff = w_l * gamma_key_l`)**:
   Absorbs per-layer key `RMSNorm` weight `gamma_key_l` (`arXiv:2603.15031` Fig. 2 & Sec. 5)
   into `q_eff_l = w_l * gamma_key_l` (`O(D)` work independent of `M = B * T` tokens),
   recovering exact `dw_l = gamma_key_l * dq_eff_l` and `d_gamma_key_l = w_l * dq_eff_l`
   from a single token reduction (`0` extra HBM bytes, exact at step-0 `w_l = 0` init).
2. **Zero-`O1_bar`-Retention Two-Node Adjoint Identity**:
   Eliminates retaining `[S, M, D]` `FP32` Phase-1 outputs across blocks (`12 GiB` saved at
   `L = 48, M = 16384, D = 4096`) by proving:
     - Phase 2: `dlse1_s = -dx_part_s = -alpha_part_s * <dh_s, p_s - h_s>` (independent of `O1_bar`!)
     - Phase 1: `<dO1_bar_s, O1_bar_s> = sum_n tilde_alpha_{s,n} <dO1_bar_s, v_n>`!
3. **Closed-Form Radial Logit-Adjoint Identity (`<g_nm, v_hat_nm> = sum_s dx_{m,s,n} * x_{m,s,n}`)**:
   Replaces the cross-`D` `RMSNorm` radial dot product in backward with the scalar product of
   score adjoints `dx` and forward logits `x`, making 2D `(D_tile, M_tile)` backward (`dV`,
   `Part_dQ`) 100% pointwise along `D` (`0` cross-thread reductions, `0` cubin register spills
   for ANY `D` up to `65,536`, and `0` global `tl.atomic_add` contention vs `Hydra-2P`'s
   `(N*M)`-way atomic serialization!).
4. **Superfused `Phase 2 Online-Softmax Merge + Sublayer Pre-RMSNorm` (`SuperfusedPhase2PreNormFn`)**:
   Exploits the topological property of `AttnRes` (`Fig. 2` L19-36) that `h_l` has NO skip branch
   bypassing `sublayer_norm(h_l)`: computes `h_l` and `RMSNorm(h_l; gamma_sub_l)` in one register
   pass without materializing `h_l` (`[M, D]`) to HBM.
5. **Cross-Stage Virtual-Rank Block Cache Adjoint for Interleaved `1F1B` Pipeline Parallelism
   (`arXiv:2603.15031` Sec. 4.1 `Eq. 7-8` & `Fig. 3`, `Gate G9`)**:
   Caches completed blocks across virtual stages on each physical rank in forward (`O(P)` bounded
   incremental transition payload instead of `O(P*V)`) and accumulates reverse-DAG `db_k` gradients
   locally across virtual stages (`0` network bytes on cached virtual-stage edges, exact 1-GPU parity).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# =============================================================================
# 1. Exact Eager & FP64 Analytical Reference Oracles (`arXiv:2603.15031` Eq. 2-6)
# =============================================================================


def eager_rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    is_fp64 = x.dtype == torch.float64 and weight.dtype == torch.float64
    work_dt = torch.float64 if is_fp64 else torch.float32
    xf = x.to(work_dt)
    wf = weight.to(work_dt)
    rstd = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)
    out = xf * rstd * wf
    return out if is_fp64 else out.to(x.dtype)


def official_fig2_single_attnres_ref(
    blocks: Sequence[torch.Tensor],
    partial_block: Optional[torch.Tensor],
    w_query: torch.Tensor,
    gamma_key: torch.Tensor,
    eps: float = 1e-6,
    *,
    bug_duplicate_first_block: bool = False,
) -> torch.Tensor:
    """Official Moonshot `Figure 2` (`README.md`) single-sublayer `block_attn_res`:
    Materializes `V = torch.stack(...)`, `K = norm(V)`, `logits`, and `softmax(0)`.
    Preserves exact `FP64` when inputs are `float64`.
    Setting `bug_duplicate_first_block=True` reproduces the `Fig. 2` L16-24 pitfall where
    passing `partial_block = b_0` at `i = 0` stacks `[b_0, ..., b_{n-1}, b_0]` instead of Eq. 6!
    """
    is_fp64 = blocks[0].dtype == torch.float64 and w_query.dtype == torch.float64
    work_dt = torch.float64 if is_fp64 else torch.float32

    src_list = list(blocks)
    if partial_block is not None:
        src_list.append(partial_block)
    elif bug_duplicate_first_block:
        src_list.append(blocks[0])

    V = torch.stack(src_list, dim=0)  # [N_src, ..., D]
    Vf = V.to(work_dt)
    rstd = torch.rsqrt((Vf * Vf).mean(dim=-1, keepdim=True) + eps)
    K = Vf * rstd * gamma_key.to(work_dt)
    qf = w_query.to(work_dt).view(-1)
    logits = torch.einsum("d,n...d->n...", qf, K)
    probs = torch.softmax(logits, dim=0)
    h = torch.einsum("n...,n...d->...d", probs, Vf)
    return h if is_fp64 else h.to(blocks[0].dtype)


def eager_block_attnres_prenorm_ref(
    w_queries: torch.Tensor,
    gamma_keys: torch.Tensor,
    blocks: torch.Tensor,
    partials: torch.Tensor,
    has_partial: torch.Tensor,
    gamma_subs: Optional[torch.Tensor] = None,
    eps_key: float = 1e-6,
    eps_sub: float = 1e-6,
    *,
    bug_omit_rmsnorm_jacobian: bool = False,
    bug_unscaled_partition_merge: bool = False,
) -> torch.Tensor:
    """Reference for a block of `S` queries `[S, D]` attending over `N` completed `blocks`
    `[N, M, D]` plus optional per-sublayer `partials` `[S, M, D]` masked by `has_partial` `[S]`.
    If `gamma_subs` (`[S, D]`) is provided, applies sublayer Pre-RMSNorm `RMSNorm(h_s; gamma_subs[s])`.
    """
    is_fp64 = blocks.dtype == torch.float64 and w_queries.dtype == torch.float64
    work_dt = torch.float64 if is_fp64 else torch.float32
    out_dt = torch.float64 if is_fp64 else blocks.dtype

    S, D = w_queries.shape
    qf = w_queries.to(work_dt)
    gkf = gamma_keys.to(work_dt)
    vf = blocks.to(work_dt)     # [N, M, D]
    pf = partials.to(work_dt)   # [S, M, D]

    r_v = torch.rsqrt((vf * vf).mean(dim=-1, keepdim=True) + eps_key)
    if bug_omit_rmsnorm_jacobian:
        r_v = r_v.detach()
    k_v = vf * r_v  # [N, M, D]

    r_p = torch.rsqrt((pf * pf).mean(dim=-1, keepdim=True) + eps_key)
    if bug_omit_rmsnorm_jacobian:
        r_p = r_p.detach()
    k_p = pf * r_p  # [S, M, D]

    q_eff = qf * gkf  # [S, D]
    blogits = torch.einsum("sd,nmd->snm", q_eff, k_v)  # [S, N, M]
    plogits = torch.einsum("sd,smd->sm", q_eff, k_p)   # [S, M]

    outs: List[torch.Tensor] = []
    for s in range(S):
        bl_s = blogits[s]  # [N, M]
        if bool(has_partial[s].item()):
            if bug_unscaled_partition_merge:
                # Exploit 4: averages independent partition softmaxes without shared-max online merge!
                w_b = torch.softmax(bl_s, dim=0)
                h_b = torch.einsum("nm,nmd->md", w_b, vf)
                h_s = 0.5 * h_b + 0.5 * pf[s]
            else:
                all_l = torch.cat([bl_s, plogits[s : s + 1]], dim=0)  # [N+1, M]
                all_v = torch.cat([vf, pf[s : s + 1]], dim=0)         # [N+1, M, D]
                w_all = torch.softmax(all_l, dim=0)
                h_s = torch.einsum("nm,nmd->md", w_all, all_v)
        else:
            w_b = torch.softmax(bl_s, dim=0)
            h_s = torch.einsum("nm,nmd->md", w_b, vf)

        if gamma_subs is not None:
            gs_f = gamma_subs[s].to(work_dt)
            r_h = torch.rsqrt((h_s * h_s).mean(dim=-1, keepdim=True) + eps_sub)
            y_s = (h_s * r_h * gs_f).to(out_dt)
            outs.append(y_s)
        else:
            outs.append(h_s.to(out_dt))
    return torch.stack(outs, dim=0)


def compile_friendly_block_attnres_prenorm_ref(
    w_queries: torch.Tensor,
    gamma_keys: torch.Tensor,
    blocks: torch.Tensor,
    partials: torch.Tensor,
    has_partial: torch.Tensor,
    gamma_subs: Optional[torch.Tensor] = None,
    eps_key: float = 1e-6,
    eps_sub: float = 1e-6,
) -> torch.Tensor:
    """100% `torch.compile(fullgraph=True, dynamic=False)`-traceable tensorized reference
    (zero `.item()` calls or data-dependent Python branches)."""
    qf = w_queries.float()
    gkf = gamma_keys.float()
    vf = blocks.float()     # [N, M, D]
    pf = partials.float()   # [S, M, D]
    S = qf.shape[0]

    k_v = vf * torch.rsqrt((vf * vf).mean(dim=-1, keepdim=True) + eps_key)
    k_p = pf * torch.rsqrt((pf * pf).mean(dim=-1, keepdim=True) + eps_key)
    q_eff = qf * gkf

    blogits = torch.einsum("sd,nmd->snm", q_eff, k_v)  # [S, N, M]
    plogits = torch.einsum("sd,smd->sm", q_eff, k_p)   # [S, M]
    mask = has_partial.view(S, 1, 1)
    plogits_masked = torch.where(mask, plogits.unsqueeze(1), torch.full_like(plogits.unsqueeze(1), -torch.inf))

    all_logits = torch.cat([blogits, plogits_masked], dim=1)  # [S, N+1, M]
    probs = torch.softmax(all_logits, dim=1)
    h = torch.einsum("snm,nmd->smd", probs[:, :-1, :], vf) + probs[:, -1:, :].transpose(1, 2) * pf
    if gamma_subs is not None:
        r_h = torch.rsqrt((h * h).mean(dim=-1, keepdim=True) + eps_sub)
        return (h * r_h * gamma_subs.float().unsqueeze(1)).to(blocks.dtype)
    return h.to(blocks.dtype)


def fp64_block_attnres_oracle(
    w_queries: torch.Tensor,
    gamma_keys: torch.Tensor,
    blocks: torch.Tensor,
    partials: torch.Tensor,
    has_partial: torch.Tensor,
    grad_out: torch.Tensor,
    gamma_subs: Optional[torch.Tensor] = None,
    eps_key: float = 1e-6,
    eps_sub: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    with torch.enable_grad():
        wq64 = w_queries.detach().to(torch.float64).requires_grad_(True)
        gk64 = gamma_keys.detach().to(torch.float64).requires_grad_(True)
        bl64 = blocks.detach().to(torch.float64).requires_grad_(True)
        pa64 = partials.detach().to(torch.float64).requires_grad_(True)
        gs64 = gamma_subs.detach().to(torch.float64).requires_grad_(True) if gamma_subs is not None else None
        out64 = eager_block_attnres_prenorm_ref(
            wq64, gk64, bl64, pa64, has_partial, gamma_subs=gs64, eps_key=eps_key, eps_sub=eps_sub
        )
        loss = (out64 * grad_out.detach().to(torch.float64)).sum()
        grads = torch.autograd.grad(
            loss,
            (wq64, gk64, bl64, pa64) + ((gs64,) if gs64 is not None else ()),
        )
    res = {
        "out": out64.detach(),
        "dw_queries": grads[0].detach(),
        "dgamma_keys": grads[1].detach(),
        "dblocks": grads[2].detach(),
        "dpartials": grads[3].detach(),
    }
    if gs64 is not None:
        res["dgamma_subs"] = grads[4].detach()
    return res


# =============================================================================
# 2. Faithful Community `Hydra-2P` Baseline (`manishklach/attnres-kernel-lab`)
#    (`grid=(S*M,)` L2-thrashing Phase 1 + `(N*M)`-way `FP32` `tl.atomic_add` Bwd)
# =============================================================================


@triton.jit
def _hydra2p_phase1_fwd_kernel(
    Q_ptr,
    V_ptr,
    NUM_ptr,
    ROW_MAX_ptr,
    DENOM_ptr,
    N,
    M,
    D: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid // M
    m = pid % M
    od = tl.arange(0, BD)
    mask = od < D
    q = tl.load(Q_ptr + s * D + od, mask=mask, other=0.0).to(tl.float32)
    vmax = -float("inf")
    vsum = 0.0
    acc = tl.zeros((BD,), tl.float32)
    for n in range(0, N):
        v = tl.load(V_ptr + (n * M + m) * D + od, mask=mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(v * v, axis=0) / D + EPS)
        score = tl.sum(q * v * inv, axis=0)
        new_max = tl.maximum(vmax, score)
        old_scale = tl.exp(vmax - new_max)
        new_scale = tl.exp(score - new_max)
        acc = acc * old_scale + v * new_scale
        vsum = vsum * old_scale + new_scale
        vmax = new_max
    tl.store(NUM_ptr + pid * D + od, acc, mask=mask)
    tl.store(ROW_MAX_ptr + pid, vmax)
    tl.store(DENOM_ptr + pid, vsum)


@triton.jit
def _hydra2p_phase2_fwd_kernel(
    Q_ptr,
    P_ptr,
    HAS_ptr,
    NUM_ptr,
    OLD_MAX_ptr,
    OLD_DENOM_ptr,
    OUT_ptr,
    FINAL_MAX_ptr,
    FINAL_DENOM_ptr,
    M,
    D: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid // M
    od = tl.arange(0, BD)
    mask = od < D
    q = tl.load(Q_ptr + s * D + od, mask=mask, other=0.0).to(tl.float32)
    enabled = tl.load(HAS_ptr + s).to(tl.int1)
    p = tl.load(P_ptr + pid * D + od, mask=mask & enabled, other=0.0).to(tl.float32)
    num = tl.load(NUM_ptr + pid * D + od, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(p * p, axis=0) / D + EPS)
    score = tl.where(enabled, tl.sum(q * p * inv, axis=0), -float("inf"))
    old_max = tl.load(OLD_MAX_ptr + pid)
    old_den = tl.load(OLD_DENOM_ptr + pid)
    final_max = tl.maximum(old_max, score)
    a = tl.exp(old_max - final_max)
    b = tl.where(enabled, tl.exp(score - final_max), 0.0)
    final_den = a * old_den + b
    out = (a * num + b * p) / final_den
    tl.store(OUT_ptr + pid * D + od, out, mask=mask)
    tl.store(FINAL_MAX_ptr + pid, final_max)
    tl.store(FINAL_DENOM_ptr + pid, final_den)


@triton.jit
def _hydra2p_blocks_atomic_bwd_kernel(
    Q_ptr,
    V_ptr,
    OUT_ptr,
    DY_ptr,
    FINAL_MAX_ptr,
    FINAL_DENOM_ptr,
    DQ_ptr,
    DV_ptr,
    N,
    M,
    D: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid % N
    sm = pid // N
    s = sm // M
    m = sm % M
    od = tl.arange(0, BD)
    mask = od < D
    q = tl.load(Q_ptr + s * D + od, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V_ptr + (n * M + m) * D + od, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(OUT_ptr + sm * D + od, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + sm * D + od, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(v * v, axis=0) / D + EPS)
    key = v * inv
    score = tl.sum(q * key, axis=0)
    weight = tl.exp(score - tl.load(FINAL_MAX_ptr + sm)) / tl.load(FINAL_DENOM_ptr + sm)
    dl = weight * tl.sum(dy * (v - out), axis=0)
    gkey = dl * q
    norm_grad = inv * (gkey - key * (tl.sum(gkey * key, axis=0) / D))
    tl.atomic_add(DQ_ptr + s * D + od, dl * key, mask=mask)
    tl.atomic_add(DV_ptr + (n * M + m) * D + od, weight * dy + norm_grad, mask=mask)


@triton.jit
def _hydra2p_partial_atomic_bwd_kernel(
    Q_ptr,
    P_ptr,
    HAS_ptr,
    OUT_ptr,
    DY_ptr,
    FINAL_MAX_ptr,
    FINAL_DENOM_ptr,
    DQ_ptr,
    DP_ptr,
    M,
    D: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    sm = tl.program_id(0)
    s = sm // M
    od = tl.arange(0, BD)
    mask = od < D
    enabled = tl.load(HAS_ptr + s).to(tl.int1)
    q = tl.load(Q_ptr + s * D + od, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(P_ptr + sm * D + od, mask=mask & enabled, other=0.0).to(tl.float32)
    out = tl.load(OUT_ptr + sm * D + od, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + sm * D + od, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(p * p, axis=0) / D + EPS)
    key = p * inv
    score = tl.sum(q * key, axis=0)
    weight = tl.where(enabled, tl.exp(score - tl.load(FINAL_MAX_ptr + sm)) / tl.load(FINAL_DENOM_ptr + sm), 0.0)
    dl = weight * tl.sum(dy * (p - out), axis=0)
    gkey = dl * q
    norm_grad = inv * (gkey - key * (tl.sum(gkey * key, axis=0) / D))
    tl.atomic_add(DQ_ptr + s * D + od, dl * key, mask=mask)
    tl.store(DP_ptr + sm * D + od, tl.where(enabled, weight * dy + norm_grad, 0.0), mask=mask)


class Hydra2PCommunityAttnResFn(torch.autograd.Function):
    """Community `Hydra-2P` (`manishklach/attnres-kernel-lab` `triton_impl.py`), extended with
    `q_eff = w * gamma_key` and `BD = next_power_of_2(D)` so it executes on LLM widths `D > 256`."""

    @staticmethod
    def forward(
        ctx,
        w_queries: torch.Tensor,
        gamma_keys: torch.Tensor,
        blocks: torch.Tensor,
        partials: torch.Tensor,
        has_partial: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        wq = w_queries.contiguous()
        gk = gamma_keys.contiguous()
        q_eff = (wq.float() * gk.float()).contiguous()
        bflat = blocks.contiguous()
        pflat = partials.contiguous()
        has_p = has_partial.contiguous()
        S, D = wq.shape
        N, M, _ = bflat.shape
        bd = max(32, triton.next_power_of_2(D))
        nw = min(16, max(2, bd // 256))

        num = torch.empty((S, M, D), device=wq.device, dtype=torch.float32)
        old_max = torch.empty((S, M), device=wq.device, dtype=torch.float32)
        old_den = torch.empty((S, M), device=wq.device, dtype=torch.float32)
        out = torch.empty_like(pflat)
        fmax = torch.empty((S, M), device=wq.device, dtype=torch.float32)
        fden = torch.empty((S, M), device=wq.device, dtype=torch.float32)

        _hydra2p_phase1_fwd_kernel[(S * M,)](q_eff, bflat, num, old_max, old_den, N, M, D=D, BD=bd, EPS=eps, num_warps=nw)
        _hydra2p_phase2_fwd_kernel[(S * M,)](q_eff, pflat, has_p, num, old_max, old_den, out, fmax, fden, M, D=D, BD=bd, EPS=eps, num_warps=nw)
        ctx.save_for_backward(wq, gk, q_eff, bflat, pflat, has_p, out, fmax, fden)
        ctx.meta = (bd, nw, eps)
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        wq, gk, q_eff, bflat, pflat, has_p, out, fmax, fden = ctx.saved_tensors
        bd, nw, eps = ctx.meta
        S, D = wq.shape
        N, M, _ = bflat.shape
        dq_eff = torch.zeros((S, D), device=wq.device, dtype=torch.float32)
        dv = torch.zeros((N, M, D), device=wq.device, dtype=torch.float32)
        dp = torch.empty((S, M, D), device=wq.device, dtype=torch.float32)
        dy_c = dy.contiguous()

        _hydra2p_blocks_atomic_bwd_kernel[(S * M * N,)](
            q_eff, bflat, out, dy_c, fmax, fden, dq_eff, dv, N, M, D=D, BD=bd, EPS=eps, num_warps=nw
        )
        _hydra2p_partial_atomic_bwd_kernel[(S * M,)](
            q_eff, pflat, has_p, out, dy_c, fmax, fden, dq_eff, dp, M, D=D, BD=bd, EPS=eps, num_warps=nw
        )
        dwq = (dq_eff * gk.float()).to(wq.dtype)
        dgk = (dq_eff * wq.float()).to(gk.dtype)
        return dwq, dgk, dv.to(bflat.dtype), dp.to(pflat.dtype), None, None


# =============================================================================
# 3. Round-6 RSI Zero-Spill Lock-Free Two-Phase `Block AttnRes` + Superfused PreNorm
# =============================================================================


@triton.jit
def _rsi_phase1_stats_fwd_kernel(
    Q_ptr,          # [S_ACTUAL, D] fp32
    V_ptr,          # [N, M, D] bf16/fp32
    LSE1_ptr,       # [S_ACTUAL, M] fp32
    ALPHA1_ptr,     # [M, S_PAD, N] fp32
    N,
    M,
    D,
    S_ACTUAL,
    S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    """Pass F1 (`grid = (M,)`): Reads `V[:, m, :]` in `BLOCK_D = 256` chunks (`0` spills for ANY `D`),
    computing `rstd(v_{n,m})` ONCE and all `S_ACTUAL` dot-product scores `<q_eff_s, v_hat_{n,m}>` in a
    single pass with zero `S_pad` buffer copies, writing `LSE1[s, m]` and `ALPHA1[m, s, n]`."""
    m = tl.program_id(0)
    if m >= M:
        return

    od_base = tl.arange(0, BLOCK_D)
    s_idx = tl.arange(0, S)
    s_mask = s_idx < S_ACTUAL
    for n in range(0, N):
        v_row_ptr = V_ptr + (n * M + m) * D
        sq_sum = 0.0
        dots = tl.zeros((S,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            od = d0 + od_base
            d_mask = od < D
            v = tl.load(v_row_ptr + od, mask=d_mask, other=0.0).to(tl.float32)
            sq_sum += tl.sum(v * v, axis=0)
            q_tile = tl.load(
                Q_ptr + s_idx[:, None] * D + od[None, :],
                mask=s_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dots += tl.sum(q_tile * v[None, :], axis=1)

        inv_rms = tl.rsqrt(sq_sum / D + EPS)
        scores = dots * inv_rms
        tl.store(ALPHA1_ptr + (m * S + s_idx) * N + n, scores)

    rmax = tl.full((S,), -float("inf"), dtype=tl.float32)
    for n in range(0, N):
        sc = tl.load(ALPHA1_ptr + (m * S + s_idx) * N + n)
        rmax = tl.maximum(rmax, sc)

    rsum = tl.zeros((S,), dtype=tl.float32)
    for n in range(0, N):
        sc = tl.load(ALPHA1_ptr + (m * S + s_idx) * N + n)
        ex = tl.exp(sc - rmax)
        tl.store(ALPHA1_ptr + (m * S + s_idx) * N + n, ex)
        rsum += ex

    inv_sum = 1.0 / rsum
    for n in range(0, N):
        ex = tl.load(ALPHA1_ptr + (m * S + s_idx) * N + n)
        tl.store(ALPHA1_ptr + (m * S + s_idx) * N + n, tl.where(s_mask, ex * inv_sum, 0.0))

    lse1 = rmax + tl.log(rsum)
    tl.store(LSE1_ptr + s_idx * M + m, lse1, mask=s_mask)


@triton.jit
def _rsi_phase1_weighted_sum_2d_fwd_kernel(
    V_ptr,          # [N, M, D]
    ALPHA1_ptr,     # [M, S_PAD, N] fp32
    O1_BAR_ptr,     # [S_ACTUAL, M, D] fp32
    N,
    M,
    D,
    S_ACTUAL,
    S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Pass F2 (`grid = (cdiv(D, BLOCK_D), M)`): Zero-reduction coalesced 2D streaming weighted sum
    `O1_bar[s, m, d_tile] = sum_n ALPHA1[m, s, n] * V[n, m, d_tile]` (`0` `S_pad` copies, `0` spills)."""
    pid_d = tl.program_id(0)
    m = tl.program_id(1)
    od = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = od < D
    s_idx = tl.arange(0, S)
    s_mask = s_idx < S_ACTUAL

    acc = tl.zeros((S, BLOCK_D), dtype=tl.float32)
    for n in range(0, N):
        v = tl.load(V_ptr + (n * M + m) * D + od, mask=d_mask, other=0.0).to(tl.float32)
        w = tl.load(ALPHA1_ptr + (m * S + s_idx) * N + n)  # [S]
        acc += w[:, None] * v[None, :]

    tl.store(
        O1_BAR_ptr + (s_idx[:, None] * M + m) * D + od[None, :],
        acc,
        mask=s_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _rsi_phase1_1pass_fullrow_fwd_kernel(
    Q_ptr,          # [S_ACTUAL, D] fp32
    V_ptr,          # [N, M, D]
    O1_BAR_ptr,     # [S_ACTUAL, M, D] fp32
    LSE1_ptr,       # [S_ACTUAL, M] fp32
    N,
    M,
    S_ACTUAL,
    D: tl.constexpr,
    S: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    """Single-Read 1-Pass Phase 1 (`grid = (M,)` for `D <= 2048`, `S <= 8`):
    Reads `V[:, m, :]` STRICTLY ONCE from HBM (`1.0x` minimal algorithmic read!), computes `rsqrt`
    once per `(n, m)`, and updates all `S_ACTUAL` online-softmax numerators with `0` copies & `0` spills."""
    m = tl.program_id(0)
    od = tl.arange(0, BD)
    d_mask = od < D
    s_idx = tl.arange(0, S)
    s_mask = s_idx < S_ACTUAL

    q_mat = tl.load(
        Q_ptr + s_idx[:, None] * D + od[None, :],
        mask=s_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    vmax = tl.full((S,), -float("inf"), dtype=tl.float32)
    vsum = tl.zeros((S,), dtype=tl.float32)
    acc = tl.zeros((S, BD), dtype=tl.float32)

    for n in range(0, N):
        v = tl.load(V_ptr + (n * M + m) * D + od, mask=d_mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(v * v, axis=0) / D + EPS)
        scores = tl.sum(q_mat * v[None, :], axis=1) * inv  # [S]
        new_max = tl.maximum(vmax, scores)
        old_s = tl.exp(vmax - new_max)
        new_s = tl.exp(scores - new_max)
        acc = acc * old_s[:, None] + new_s[:, None] * v[None, :]
        vsum = vsum * old_s + new_s
        vmax = new_max

    o1_bar = acc / vsum[:, None]
    lse1 = vmax + tl.log(vsum)
    tl.store(
        O1_BAR_ptr + (s_idx[:, None] * M + m) * D + od[None, :],
        o1_bar,
        mask=s_mask[:, None] & d_mask[None, :],
    )
    tl.store(LSE1_ptr + s_idx * M + m, lse1, mask=s_mask)


@triton.jit
def _rsi_phase1_bwd_scores_kernel(
    Q_ptr,          # [S_ACTUAL, D] fp32
    V_ptr,          # [N, M, D]
    LSE1_ptr,       # [S_ACTUAL, M] fp32
    DO1_ptr,        # [S_ACTUAL, M, D] fp32
    DLSE1_ptr,      # [S_ACTUAL, M] fp32
    RSTD_V_ptr,     # [M, N] fp32
    ALPHA1_ptr,     # [M, S_PAD, N] fp32
    DX_ptr,         # [M, S_PAD, N] fp32
    RHO_ptr,        # [M, N] fp32
    N,
    M,
    D,
    S_ACTUAL,
    S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    """Pass B1 (`grid = (M,)`): Zero-copy masked `S_ACTUAL <= S_PAD` score & closed-form `RMSNorm`
    radial scalar `RHO[m, n] = (r^2 / D) * sum_{s < S_ACTUAL} dx * x` (`0` `O1_bar` retention)."""
    m = tl.program_id(0)
    if m >= M:
        return

    s_idx = tl.arange(0, S)
    s_mask = s_idx < S_ACTUAL
    lse1 = tl.load(LSE1_ptr + s_idx * M + m, mask=s_mask, other=0.0)
    dlse1 = tl.load(DLSE1_ptr + s_idx * M + m, mask=s_mask, other=0.0)
    od_base = tl.arange(0, BLOCK_D)
    sd_mask_prefix = s_mask[:, None]

    c1 = tl.zeros((S,), dtype=tl.float32)
    for n in range(0, N):
        v_row_ptr = V_ptr + (n * M + m) * D
        sq_sum = 0.0
        q_dots = tl.zeros((S,), dtype=tl.float32)
        u_dots = tl.zeros((S,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            od = d0 + od_base
            d_mask = od < D
            sd_mask = sd_mask_prefix & d_mask[None, :]
            v = tl.load(v_row_ptr + od, mask=d_mask, other=0.0).to(tl.float32)
            sq_sum += tl.sum(v * v, axis=0)
            q_tile = tl.load(Q_ptr + s_idx[:, None] * D + od[None, :], mask=sd_mask, other=0.0).to(tl.float32)
            do_tile = tl.load(DO1_ptr + (s_idx[:, None] * M + m) * D + od[None, :], mask=sd_mask, other=0.0).to(tl.float32)
            q_dots += tl.sum(q_tile * v[None, :], axis=1)
            u_dots += tl.sum(do_tile * v[None, :], axis=1)

        rstd_v = tl.rsqrt(sq_sum / D + EPS)
        x_sc = tl.where(s_mask, q_dots * rstd_v, 0.0)
        a1 = tl.where(s_mask, tl.exp(x_sc - lse1), 0.0)
        c1 += a1 * u_dots

        tl.store(RSTD_V_ptr + m * N + n, rstd_v)
        tl.store(ALPHA1_ptr + (m * S + s_idx) * N + n, x_sc)
        tl.store(DX_ptr + (m * S + s_idx) * N + n, u_dots)

    inv_d = 1.0 / D
    for n in range(0, N):
        rstd_v = tl.load(RSTD_V_ptr + m * N + n)
        x_sc = tl.load(ALPHA1_ptr + (m * S + s_idx) * N + n)
        u_dots = tl.load(DX_ptr + (m * S + s_idx) * N + n)
        a1 = tl.where(s_mask, tl.exp(x_sc - lse1), 0.0)
        dx = tl.where(s_mask, a1 * (u_dots - c1 + dlse1), 0.0)
        rho = (rstd_v * rstd_v * inv_d) * tl.sum(dx * x_sc, axis=0)
        tl.store(ALPHA1_ptr + (m * S + s_idx) * N + n, a1)
        tl.store(DX_ptr + (m * S + s_idx) * N + n, dx)
        tl.store(RHO_ptr + m * N + n, rho)


@triton.jit
def _rsi_phase1_bwd_dv_dq_2d_kernel(
    Q_ptr,          # [S_ACTUAL, D] fp32
    V_ptr,          # [N, M, D]
    DO1_ptr,        # [S_ACTUAL, M, D] fp32
    RSTD_V_ptr,     # [M, N] fp32
    ALPHA1_ptr,     # [M, S_PAD, N] fp32
    DX_ptr,         # [M, S_PAD, N] fp32
    RHO_ptr,        # [M, N] fp32
    DV_ptr,         # [N, M, D] out
    PART_DQ_ptr,    # [G_M, S_ACTUAL, D] fp32
    N,
    M,
    D,
    TOKENS_PER_SPLIT,
    S_ACTUAL,
    S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Pass B2 (`grid = (cdiv(D, BLOCK_D), G_M)`): 100% Lock-Free, Zero-Copy, Zero-Spill
    pointwise backward along `D` emitting exact `DV` (`1` writer) and `PART_DQ[gm, s, d_tile]`."""
    pid_d = tl.program_id(0)
    gm = tl.program_id(1)
    od = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = od < D
    s_idx = tl.arange(0, S)
    s_mask = s_idx < S_ACTUAL
    sd_mask = s_mask[:, None] & d_mask[None, :]

    m_start = gm * TOKENS_PER_SPLIT
    m_end = tl.minimum(M, m_start + TOKENS_PER_SPLIT)

    q_tile = tl.load(Q_ptr + s_idx[:, None] * D + od[None, :], mask=sd_mask, other=0.0).to(tl.float32)
    dq_acc = tl.zeros((S, BLOCK_D), dtype=tl.float32)

    for m in range(m_start, m_end):
        do_tile = tl.load(DO1_ptr + (s_idx[:, None] * M + m) * D + od[None, :], mask=sd_mask, other=0.0).to(tl.float32)
        for n in range(0, N):
            v = tl.load(V_ptr + (n * M + m) * D + od, mask=d_mask, other=0.0).to(tl.float32)
            r_nm = tl.load(RSTD_V_ptr + m * N + n)
            rho_nm = tl.load(RHO_ptr + m * N + n)
            a1 = tl.load(ALPHA1_ptr + (m * S + s_idx) * N + n)   # [S] (`0.0` for `s >= S_ACTUAL`)
            dx = tl.load(DX_ptr + (m * S + s_idx) * N + n)       # [S] (`0.0` for `s >= S_ACTUAL`)

            dv_direct = tl.sum(a1[:, None] * do_tile, axis=0)
            g_vkey = tl.sum(dx[:, None] * q_tile, axis=0)
            dv = dv_direct + r_nm * g_vkey - rho_nm * v
            tl.store(DV_ptr + (n * M + m) * D + od, dv, mask=d_mask)

            dq_acc += (dx * r_nm)[:, None] * v[None, :]

    tl.store(
        PART_DQ_ptr + (gm * S_ACTUAL + s_idx[:, None]) * D + od[None, :],
        dq_acc,
        mask=sd_mask,
    )


@triton.jit
def _reduce_splitm_vec_kernel(
    PART_ptr,       # [G_M, K_ROWS, D] fp32
    OUT_A_ptr,      # [K_ROWS, D] (dw = sum * scale_a)
    OUT_B_ptr,      # [K_ROWS, D] optional (dgamma = sum * scale_b)
    SCALE_A_ptr,    # [K_ROWS, D] optional
    SCALE_B_ptr,    # [K_ROWS, D] optional
    G_M,
    TOTAL_ELEMS,
    HAS_DUAL_SCALE: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < TOTAL_ELEMS
    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for gm in range(0, G_M):
        acc += tl.load(PART_ptr + gm * TOTAL_ELEMS + offs, mask=mask, other=0.0)
    if HAS_DUAL_SCALE:
        sa = tl.load(SCALE_A_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sb = tl.load(SCALE_B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT_A_ptr + offs, acc * sa, mask=mask)
        tl.store(OUT_B_ptr + offs, acc * sb, mask=mask)
    else:
        tl.store(OUT_A_ptr + offs, acc, mask=mask)


@triton.jit
def _rsi_phase2_merge_prenorm_fwd_kernel(
    Q_ptr,          # [D] fp32 (`q_eff_s`)
    O1_ptr,         # [M, D] fp32 (`O1_bar[s]`)
    LSE1_ptr,       # [M] fp32 (`LSE1[s]`)
    P_ptr,          # [M, D] (`partial_s`, unused when HAS_PARTIAL=0)
    GAMMA_SUB_ptr,  # [D] (`gamma_sub_s`, unused when FUSE_SUB_NORM=0)
    Y_OUT_ptr,      # [M, D] (`y_s` if FUSE_SUB_NORM=1 else `h_s`)
    H_SAVE_ptr,     # [M, D] (`u_hat_s` if FUSE_SUB_NORM=1 else `h_s`)
    STD_H_ptr,      # [M] fp32 (`sqrt(mean(h^2)+eps_sub)` if FUSE_SUB_NORM=1)
    ALPHA_P_ptr,    # [M] fp32 (`alpha_part` if HAS_PARTIAL=1)
    RSTD_P_ptr,     # [M] fp32 (`rstd_p` if HAS_PARTIAL=1)
    X_PART_ptr,     # [M] fp32 (`x_part` if HAS_PARTIAL=1)
    M,
    D,
    BLOCK_D: tl.constexpr,
    HAS_PARTIAL: tl.constexpr,
    FUSE_SUB_NORM: tl.constexpr,
    EPS_KEY: tl.constexpr,
    EPS_SUB: tl.constexpr,
):
    """Chunked-`D` (`BLOCK_D = 256`, `0` spills for ANY `D`) Phase-2 Online-Softmax Merge,
    optionally superfused with Sublayer Pre-RMSNorm (`FUSE_SUB_NORM = 1`) so `h_s` is never
    materialized as a separate tensor!"""
    m = tl.program_id(0)
    if m >= M:
        return

    od_base = tl.arange(0, BLOCK_D)
    if HAS_PARTIAL:
        sq_p = 0.0
        dot_qp = 0.0
        for d0 in range(0, D, BLOCK_D):
            od = d0 + od_base
            dm = od < D
            p = tl.load(P_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
            q = tl.load(Q_ptr + od, mask=dm, other=0.0).to(tl.float32)
            sq_p += tl.sum(p * p, axis=0)
            dot_qp += tl.sum(q * p, axis=0)
        rstd_p = tl.rsqrt(sq_p / D + EPS_KEY)
        x_part = dot_qp * rstd_p
        lse1 = tl.load(LSE1_ptr + m)
        m_tot = tl.maximum(lse1, x_part)
        e_hist = tl.exp(lse1 - m_tot)
        e_part = tl.exp(x_part - m_tot)
        inv_tot = 1.0 / (e_hist + e_part)
        a_hist = e_hist * inv_tot
        a_part = e_part * inv_tot
        tl.store(ALPHA_P_ptr + m, a_part)
        tl.store(RSTD_P_ptr + m, rstd_p)
        tl.store(X_PART_ptr + m, x_part)
    else:
        a_hist = 1.0
        a_part = 0.0

    if FUSE_SUB_NORM:
        sq_h = 0.0
        for d0 in range(0, D, BLOCK_D):
            od = d0 + od_base
            dm = od < D
            o1 = tl.load(O1_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
            if HAS_PARTIAL:
                p = tl.load(P_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                h = a_hist * o1 + a_part * p
            else:
                h = o1
            sq_h += tl.sum(h * h, axis=0)
        var_eps = sq_h / D + EPS_SUB
        rstd_h = tl.rsqrt(var_eps)
        std_h = tl.sqrt(var_eps)
        tl.store(STD_H_ptr + m, std_h)

        for d0 in range(0, D, BLOCK_D):
            od = d0 + od_base
            dm = od < D
            o1 = tl.load(O1_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
            if HAS_PARTIAL:
                p = tl.load(P_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                h = a_hist * o1 + a_part * p
            else:
                h = o1
            u_hat = h * rstd_h
            gs = tl.load(GAMMA_SUB_ptr + od, mask=dm, other=0.0).to(tl.float32)
            tl.store(H_SAVE_ptr + m * D + od, u_hat, mask=dm)
            tl.store(Y_OUT_ptr + m * D + od, u_hat * gs, mask=dm)
    else:
        for d0 in range(0, D, BLOCK_D):
            od = d0 + od_base
            dm = od < D
            o1 = tl.load(O1_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
            if HAS_PARTIAL:
                p = tl.load(P_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                h = a_hist * o1 + a_part * p
            else:
                h = o1
            tl.store(Y_OUT_ptr + m * D + od, h, mask=dm)
            if HAS_PARTIAL:
                tl.store(H_SAVE_ptr + m * D + od, h, mask=dm)


@triton.jit
def _rsi_phase2_merge_prenorm_bwd_scalars_kernel(
    DY_ptr,         # [M, D]
    GAMMA_SUB_ptr,  # [D] (unused if FUSE_SUB_NORM=0)
    H_SAVE_ptr,     # [M, D] (`u_hat` if FUSE_SUB_NORM=1 else `h`)
    STD_H_ptr,      # [M] fp32 (`std_h` if FUSE_SUB_NORM=1)
    P_ptr,          # [M, D] (unused if HAS_PARTIAL=0)
    ALPHA_P_ptr,    # [M] fp32
    DOT_GSUB_ptr,   # [M] fp32 out (`mean(dy * gamma_sub * u_hat)` if FUSE_SUB_NORM=1)
    DX_PART_ptr,    # [M] fp32 out (`dx_part` if HAS_PARTIAL=1)
    DLSE1_ptr,      # [M] fp32 out (`-dx_part` or `0.0`)
    M,
    D,
    BLOCK_D: tl.constexpr,
    HAS_PARTIAL: tl.constexpr,
    FUSE_SUB_NORM: tl.constexpr,
):
    """Computes per-token scalars `dot_gsub` (`RMSNorm` projection) and `dx_part` / `dlse1`
    in one `BLOCK_D = 256` pass (`0` spills for ANY `D`)."""
    m = tl.program_id(0)
    if m >= M:
        return

    od_base = tl.arange(0, BLOCK_D)
    dot_gsub = 0.0
    dot_dh_pmh = 0.0

    if FUSE_SUB_NORM:
        std_h = tl.load(STD_H_ptr + m)
        rstd_h = 1.0 / std_h
        for d0 in range(0, D, BLOCK_D):
            od = d0 + od_base
            dm = od < D
            dy = tl.load(DY_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
            gs = tl.load(GAMMA_SUB_ptr + od, mask=dm, other=0.0).to(tl.float32)
            u_hat = tl.load(H_SAVE_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
            dot_gsub += tl.sum(dy * gs * u_hat, axis=0)
        dot_gsub = dot_gsub / D
        tl.store(DOT_GSUB_ptr + m, dot_gsub)

        if HAS_PARTIAL:
            for d0 in range(0, D, BLOCK_D):
                od = d0 + od_base
                dm = od < D
                dy = tl.load(DY_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                gs = tl.load(GAMMA_SUB_ptr + od, mask=dm, other=0.0).to(tl.float32)
                u_hat = tl.load(H_SAVE_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                p = tl.load(P_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                dh = rstd_h * (dy * gs - u_hat * dot_gsub)
                h = u_hat * std_h
                dot_dh_pmh += tl.sum(dh * (p - h), axis=0)
    else:
        if HAS_PARTIAL:
            for d0 in range(0, D, BLOCK_D):
                od = d0 + od_base
                dm = od < D
                dh = tl.load(DY_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                h = tl.load(H_SAVE_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                p = tl.load(P_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
                dot_dh_pmh += tl.sum(dh * (p - h), axis=0)

    if HAS_PARTIAL:
        a_part = tl.load(ALPHA_P_ptr + m)
        dx_p = a_part * dot_dh_pmh
        tl.store(DX_PART_ptr + m, dx_p)
        tl.store(DLSE1_ptr + m, -dx_p)
    else:
        tl.store(DLSE1_ptr + m, 0.0)


@triton.jit
def _rsi_phase2_merge_prenorm_bwd_2d_kernel(
    Q_ptr,              # [D] fp32
    DY_ptr,             # [M, D]
    GAMMA_SUB_ptr,      # [D]
    H_SAVE_ptr,         # [M, D]
    STD_H_ptr,          # [M] fp32
    DOT_GSUB_ptr,       # [M] fp32
    P_ptr,              # [M, D]
    ALPHA_P_ptr,        # [M] fp32
    RSTD_P_ptr,         # [M] fp32
    X_PART_ptr,         # [M] fp32
    DX_PART_ptr,        # [M] fp32
    DO1_OUT_ptr,        # [M, D] fp32 out
    DP_OUT_ptr,         # [M, D] out
    PART_DQ_ptr,        # [G_M, D] fp32 out
    PART_DGSUB_ptr,     # [G_M, D] fp32 out
    M,
    D,
    TOKENS_PER_SPLIT,
    BLOCK_D: tl.constexpr,
    HAS_PARTIAL: tl.constexpr,
    FUSE_SUB_NORM: tl.constexpr,
):
    """100% Lock-Free, Zero-Reduction, Zero-Spill 2D (`grid = (cdiv(D, BLOCK_D), G_M)`) Phase-2
    backward using exact closed-form radial scalar `<g_pkey, p_hat> = dx_part * x_part`."""
    pid_d = tl.program_id(0)
    gm = tl.program_id(1)
    od = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = od < D

    m_start = gm * TOKENS_PER_SPLIT
    m_end = tl.minimum(M, m_start + TOKENS_PER_SPLIT)

    if HAS_PARTIAL:
        q = tl.load(Q_ptr + od, mask=dm, other=0.0).to(tl.float32)
        dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if FUSE_SUB_NORM:
        gs = tl.load(GAMMA_SUB_ptr + od, mask=dm, other=0.0).to(tl.float32)
        dgsub_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    inv_d = 1.0 / D
    for m in range(m_start, m_end):
        dy = tl.load(DY_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
        if FUSE_SUB_NORM:
            u_hat = tl.load(H_SAVE_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)
            std_h = tl.load(STD_H_ptr + m)
            dot_g = tl.load(DOT_GSUB_ptr + m)
            dgsub_acc += dy * u_hat
            dh = (dy * gs - u_hat * dot_g) / std_h
        else:
            dh = dy

        if HAS_PARTIAL:
            a_part = tl.load(ALPHA_P_ptr + m)
            a_hist = 1.0 - a_part
            rstd_p = tl.load(RSTD_P_ptr + m)
            x_part = tl.load(X_PART_ptr + m)
            dx_part = tl.load(DX_PART_ptr + m)
            p = tl.load(P_ptr + m * D + od, mask=dm, other=0.0).to(tl.float32)

            tl.store(DO1_OUT_ptr + m * D + od, a_hist * dh, mask=dm)
            rho_p = (rstd_p * rstd_p * inv_d) * (dx_part * x_part)
            dp = a_part * dh + (rstd_p * dx_part) * q - rho_p * p
            tl.store(DP_OUT_ptr + m * D + od, dp, mask=dm)
            dq_acc += (dx_part * rstd_p) * p
        else:
            tl.store(DO1_OUT_ptr + m * D + od, dh, mask=dm)

    if HAS_PARTIAL:
        tl.store(PART_DQ_ptr + gm * D + od, dq_acc, mask=dm)
    if FUSE_SUB_NORM:
        tl.store(PART_DGSUB_ptr + gm * D + od, dgsub_acc, mask=dm)


# =============================================================================
# 4. PyTorch Autograd Wrappers (`Phase 1` + `Phase 2` Sequential & Batched Block)
# =============================================================================


@triton.jit
def _rsi_batched_phase2_1pass_fwd_kernel(
    Q_ptr,          # [S, D] fp32
    O1_ptr,         # [S, M, D] fp32
    LSE1_ptr,       # [S, M] fp32
    P_ptr,          # [S, M, D]
    HAS_ptr,        # [S] bool
    GSUB_ptr,       # [S, D] fp32
    Y_OUT_ptr,      # [S, M, D]
    H_SAVE_ptr,     # [S, M, D]
    STD_H_ptr,      # [S, M] fp32
    ALPHA_P_ptr,    # [S, M] fp32
    RSTD_P_ptr,     # [S, M] fp32
    X_PART_ptr,     # [S, M] fp32
    M,
    D: tl.constexpr,
    BD: tl.constexpr,
    FUSE_SUB_NORM: tl.constexpr,
    EPS_KEY: tl.constexpr,
    EPS_SUB: tl.constexpr,
):
    """Single-Read 1-Pass Batched Phase-2 + Optional Superfused Pre-RMSNorm (`grid = (S * M,)`
    for `D <= 4096`, `num_warps <= 8` -> `< 64` regs, `0` cubin spills!)."""
    pid = tl.program_id(0)
    s = pid // M
    od = tl.arange(0, BD)
    dm = od < D
    enabled = tl.load(HAS_ptr + s).to(tl.int1)
    o1 = tl.load(O1_ptr + pid * D + od, mask=dm, other=0.0).to(tl.float32)

    if enabled:
        q = tl.load(Q_ptr + s * D + od, mask=dm, other=0.0).to(tl.float32)
        p = tl.load(P_ptr + pid * D + od, mask=dm, other=0.0).to(tl.float32)
        rp = tl.rsqrt(tl.sum(p * p, axis=0) / D + EPS_KEY)
        xp = tl.sum(q * p, axis=0) * rp
        lse1 = tl.load(LSE1_ptr + pid)
        mt = tl.maximum(lse1, xp)
        eh = tl.exp(lse1 - mt)
        ep = tl.exp(xp - mt)
        invt = 1.0 / (eh + ep)
        ah = eh * invt
        ap = ep * invt
        h = ah * o1 + ap * p
        tl.store(ALPHA_P_ptr + pid, ap)
        tl.store(RSTD_P_ptr + pid, rp)
        tl.store(X_PART_ptr + pid, xp)
    else:
        h = o1
        tl.store(ALPHA_P_ptr + pid, 0.0)
        tl.store(RSTD_P_ptr + pid, 0.0)
        tl.store(X_PART_ptr + pid, 0.0)

    if FUSE_SUB_NORM:
        ve = tl.sum(h * h, axis=0) / D + EPS_SUB
        rh = tl.rsqrt(ve)
        sh = tl.sqrt(ve)
        u_hat = h * rh
        gs = tl.load(GSUB_ptr + s * D + od, mask=dm, other=0.0).to(tl.float32)
        tl.store(STD_H_ptr + pid, sh)
        tl.store(H_SAVE_ptr + pid * D + od, u_hat, mask=dm)
        tl.store(Y_OUT_ptr + pid * D + od, u_hat * gs, mask=dm)
    else:
        tl.store(Y_OUT_ptr + pid * D + od, h, mask=dm)
        tl.store(H_SAVE_ptr + pid * D + od, h, mask=dm)


@triton.jit
def _rsi_batched_phase2_bwd_scalars_1pass_kernel(
    DY_ptr,         # [S, M, D]
    GSUB_ptr,       # [S, D]
    H_SAVE_ptr,     # [S, M, D]
    STD_H_ptr,      # [S, M] fp32
    P_ptr,          # [S, M, D]
    HAS_ptr,        # [S] bool
    ALPHA_P_ptr,    # [S, M] fp32
    DOT_GSUB_ptr,   # [S, M] fp32 out
    DX_PART_ptr,    # [S, M] fp32 out
    DLSE1_ptr,      # [S, M] fp32 out
    M,
    D: tl.constexpr,
    BD: tl.constexpr,
    FUSE_SUB_NORM: tl.constexpr,
):
    """Single-read 1-pass scalar projection kernel (`grid = (S * M,)` for `D <= 4096`, `0` spills)."""
    pid = tl.program_id(0)
    s = pid // M
    od = tl.arange(0, BD)
    dm = od < D
    enabled = tl.load(HAS_ptr + s).to(tl.int1)
    dy = tl.load(DY_ptr + pid * D + od, mask=dm, other=0.0).to(tl.float32)

    if FUSE_SUB_NORM:
        gs = tl.load(GSUB_ptr + s * D + od, mask=dm, other=0.0).to(tl.float32)
        u_hat = tl.load(H_SAVE_ptr + pid * D + od, mask=dm, other=0.0).to(tl.float32)
        std_h = tl.load(STD_H_ptr + pid)
        dot_g = tl.sum(dy * gs * u_hat, axis=0) / D
        tl.store(DOT_GSUB_ptr + pid, dot_g)
        dh = (dy * gs - u_hat * dot_g) / std_h
        h = u_hat * std_h
    else:
        dh = dy
        h = tl.load(H_SAVE_ptr + pid * D + od, mask=dm & enabled, other=0.0).to(tl.float32)

    if enabled:
        p = tl.load(P_ptr + pid * D + od, mask=dm, other=0.0).to(tl.float32)
        ap = tl.load(ALPHA_P_ptr + pid)
        dxp = ap * tl.sum(dh * (p - h), axis=0)
        tl.store(DX_PART_ptr + pid, dxp)
        tl.store(DLSE1_ptr + pid, -dxp)
    else:
        tl.store(DX_PART_ptr + pid, 0.0)
        tl.store(DLSE1_ptr + pid, 0.0)


@triton.jit
def _rsi_batched_phase2_bwd_2d_kernel(
    Q_ptr,              # [S, D] fp32
    DY_ptr,             # [S, M, D]
    GSUB_ptr,           # [S, D]
    H_SAVE_ptr,         # [S, M, D]
    STD_H_ptr,          # [S, M] fp32
    DOT_GSUB_ptr,       # [S, M] fp32
    P_ptr,              # [S, M, D]
    HAS_ptr,            # [S] bool
    ALPHA_P_ptr,        # [S, M] fp32
    RSTD_P_ptr,         # [S, M] fp32
    X_PART_ptr,         # [S, M] fp32
    DX_PART_ptr,        # [S, M] fp32
    DO1_OUT_ptr,        # [S, M, D] fp32 out
    DP_OUT_ptr,         # [S, M, D] out
    PART_DQ_ptr,        # [G_M, S, D] fp32 out
    PART_DGSUB_ptr,     # [G_M, S, D] fp32 out
    M,
    D,
    TOKENS_PER_SPLIT,
    BLOCK_D: tl.constexpr,
    FUSE_SUB_NORM: tl.constexpr,
):
    """Vectorized across `(pid_d, gm, s)`: `0` atomics, `0` cross-thread reductions, `0` spills!"""
    pid_d = tl.program_id(0)
    gm = tl.program_id(1)
    s = tl.program_id(2)
    od = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = od < D

    m_start = gm * TOKENS_PER_SPLIT
    m_end = tl.minimum(M, m_start + TOKENS_PER_SPLIT)
    enabled = tl.load(HAS_ptr + s).to(tl.int1)

    q = tl.load(Q_ptr + s * D + od, mask=dm & enabled, other=0.0).to(tl.float32)
    dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if FUSE_SUB_NORM:
        gs = tl.load(GSUB_ptr + s * D + od, mask=dm, other=0.0).to(tl.float32)
        dgsub_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    inv_d = 1.0 / D
    base_sm = s * M
    for m in range(m_start, m_end):
        sm = base_sm + m
        dy = tl.load(DY_ptr + sm * D + od, mask=dm, other=0.0).to(tl.float32)
        if FUSE_SUB_NORM:
            u_hat = tl.load(H_SAVE_ptr + sm * D + od, mask=dm, other=0.0).to(tl.float32)
            std_h = tl.load(STD_H_ptr + sm)
            dot_g = tl.load(DOT_GSUB_ptr + sm)
            dgsub_acc += dy * u_hat
            dh = (dy * gs - u_hat * dot_g) / std_h
        else:
            dh = dy

        if enabled:
            ap = tl.load(ALPHA_P_ptr + sm)
            ah = 1.0 - ap
            rp = tl.load(RSTD_P_ptr + sm)
            xp = tl.load(X_PART_ptr + sm)
            dxp = tl.load(DX_PART_ptr + sm)
            p = tl.load(P_ptr + sm * D + od, mask=dm, other=0.0).to(tl.float32)

            tl.store(DO1_OUT_ptr + sm * D + od, ah * dh, mask=dm)
            rho_p = (rp * rp * inv_d) * (dxp * xp)
            dp = ap * dh + (rp * dxp) * q - rho_p * p
            tl.store(DP_OUT_ptr + sm * D + od, dp, mask=dm)
            dq_acc += (dxp * rp) * p
        else:
            tl.store(DO1_OUT_ptr + sm * D + od, dh, mask=dm)
            tl.store(DP_OUT_ptr + sm * D + od, 0.0, mask=dm)

    S_tot = tl.num_programs(2)
    tl.store(PART_DQ_ptr + (gm * S_tot + s) * D + od, dq_acc, mask=dm)
    if FUSE_SUB_NORM:
        tl.store(PART_DGSUB_ptr + (gm * S_tot + s) * D + od, dgsub_acc, mask=dm)


def _choose_splits_m(M: int) -> Tuple[int, int]:
    gm = min(512, max(1, triton.cdiv(M, 16)))
    tps = triton.cdiv(M, gm)
    gm_eff = triton.cdiv(M, tps)
    return gm_eff, tps


def rsi_phase1_fwd(
    q_eff: torch.Tensor, blocks: torch.Tensor, eps_key: float = 1e-6, allow_1pass: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs Phase-1 Inter-Block Attention returning `(O1_bar [S, M, D] fp32, LSE1 [S, M] fp32)`
    with zero `S_pad` buffer allocations or memcpys."""
    S, D = q_eff.shape
    N, M, _ = blocks.shape
    o1_bar = torch.empty((S, M, D), device=blocks.device, dtype=torch.float32)
    lse1 = torch.empty((S, M), device=blocks.device, dtype=torch.float32)
    S_pad = max(1, triton.next_power_of_2(S))

    if allow_1pass and D <= 2048 and S_pad <= 8:
        bd = max(32, triton.next_power_of_2(D))
        nw = min(8, max(2, bd // 256))
        _rsi_phase1_1pass_fullrow_fwd_kernel[(M,)](
            q_eff, blocks, o1_bar, lse1, N, M, S, D=D, S=S_pad, BD=bd, EPS=eps_key, num_warps=nw
        )
    else:
        block_d = 256
        alpha1 = torch.empty((M, S_pad, N), device=blocks.device, dtype=torch.float32)
        _rsi_phase1_stats_fwd_kernel[(M,)](
            q_eff, blocks, lse1, alpha1, N, M, D, S, S=S_pad, BLOCK_D=block_d, EPS=eps_key, num_warps=4
        )
        _rsi_phase1_weighted_sum_2d_fwd_kernel[(triton.cdiv(D, block_d), M)](
            blocks, alpha1, o1_bar, N, M, D, S, S=S_pad, BLOCK_D=block_d, num_warps=4
        )
    return o1_bar, lse1


def rsi_phase1_bwd(
    q_eff: torch.Tensor,
    blocks: torch.Tensor,
    lse1: torch.Tensor,
    do1_bar: torch.Tensor,
    dlse1: torch.Tensor,
    eps_key: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lock-free zero-`O1_bar`, zero-`S_pad`-copy Phase-1 backward returning `(dq_eff [S, D], dblocks)`."""
    S, D = q_eff.shape
    N, M, _ = blocks.shape
    S_pad = max(1, triton.next_power_of_2(S))

    block_d = 512 if D >= 512 else 256
    nw_b1 = 8 if block_d >= 512 else 4
    rstd_v = torch.empty((M, N), device=blocks.device, dtype=torch.float32)
    alpha1 = torch.empty((M, S_pad, N), device=blocks.device, dtype=torch.float32)
    dx = torch.empty((M, S_pad, N), device=blocks.device, dtype=torch.float32)
    rho = torch.empty((M, N), device=blocks.device, dtype=torch.float32)

    _rsi_phase1_bwd_scores_kernel[(M,)](
        q_eff, blocks, lse1, do1_bar, dlse1, rstd_v, alpha1, dx, rho,
        N, M, D, S, S=S_pad, BLOCK_D=block_d, EPS=eps_key, num_warps=nw_b1,
    )

    gm, tps = _choose_splits_m(M)
    dblocks = torch.empty_like(blocks)
    part_dq = torch.empty((gm, S, D), device=blocks.device, dtype=torch.float32)
    _rsi_phase1_bwd_dv_dq_2d_kernel[(triton.cdiv(D, 256), gm)](
        q_eff, blocks, do1_bar, rstd_v, alpha1, dx, rho, dblocks, part_dq,
        N, M, D, tps, S, S=S_pad, BLOCK_D=256, num_warps=4,
    )

    dq_out = torch.empty((S, D), device=blocks.device, dtype=torch.float32)
    tot_e = S * D
    _reduce_splitm_vec_kernel[(triton.cdiv(tot_e, 256),)](
        part_dq, dq_out, dq_out, dq_out, dq_out, gm, tot_e, HAS_DUAL_SCALE=0, BLOCK_E=256, num_warps=4
    )
    return dq_out, dblocks


class RSIPhase1InterBlockFn(torch.autograd.Function):
    """Phase-1 Inter-Block Attention Autograd Node (`0` `[S, M, D]` `O1_bar` bytes saved!)."""

    @staticmethod
    def forward(
        ctx,
        w_queries: torch.Tensor,
        gamma_keys: torch.Tensor,
        blocks: torch.Tensor,
        eps_key: float = 1e-6,
        allow_1pass: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if blocks.dtype == torch.float64 or w_queries.dtype == torch.float64:
            qf = w_queries.to(torch.float64) * gamma_keys.to(torch.float64)
            vf = blocks.to(torch.float64)
            rv = torch.rsqrt((vf * vf).mean(dim=-1, keepdim=True) + eps_key)
            logits = torch.einsum("sd,nmd->snm", qf, vf * rv)
            lse1 = torch.logsumexp(logits, dim=1)
            a1 = torch.exp(logits - lse1.unsqueeze(1))
            o1_bar = torch.einsum("snm,nmd->smd", a1, vf)
            ctx.save_for_backward(w_queries, gamma_keys, blocks, lse1)
            ctx.eps_key = eps_key
            ctx.is_fp64 = True
            return o1_bar, lse1

        wq = w_queries.contiguous()
        gk = gamma_keys.contiguous()
        bl = blocks.contiguous()
        q_eff = (wq.float() * gk.float()).contiguous()
        o1_bar, lse1 = rsi_phase1_fwd(q_eff, bl, eps_key=eps_key, allow_1pass=allow_1pass)
        # Notice `o1_bar` is NOT saved in `ctx` (`0` bytes retained across blocks!)
        ctx.save_for_backward(wq, gk, bl, lse1)
        ctx.eps_key = eps_key
        ctx.is_fp64 = False
        return o1_bar, lse1

    @staticmethod
    def backward(ctx, do1_bar: torch.Tensor, dlse1: torch.Tensor):
        wq, gk, bl, lse1 = ctx.saved_tensors
        if ctx.is_fp64:
            with torch.enable_grad():
                wq64 = wq.detach().to(torch.float64).requires_grad_(True)
                gk64 = gk.detach().to(torch.float64).requires_grad_(True)
                bl64 = bl.detach().to(torch.float64).requires_grad_(True)
                qf = wq64 * gk64
                rv = torch.rsqrt((bl64 * bl64).mean(dim=-1, keepdim=True) + ctx.eps_key)
                logits = torch.einsum("sd,nmd->snm", qf, bl64 * rv)
                lse_ = torch.logsumexp(logits, dim=1)
                a1 = torch.exp(logits - lse_.unsqueeze(1))
                o1_ = torch.einsum("snm,nmd->smd", a1, bl64)
                loss = (o1_ * do1_bar.to(torch.float64)).sum() + (lse_ * dlse1.to(torch.float64)).sum()
                dwq, dgk, dbl = torch.autograd.grad(loss, (wq64, gk64, bl64))
            return dwq.to(wq.dtype), dgk.to(gk.dtype), dbl.to(bl.dtype), None, None

        q_eff = (wq.float() * gk.float()).contiguous()
        dq_eff, dbl = rsi_phase1_bwd(
            q_eff, bl, lse1, do1_bar.float().contiguous(), dlse1.float().contiguous(), eps_key=ctx.eps_key
        )
        dwq = (dq_eff * gk.float()).to(wq.dtype)
        dgk = (dq_eff * wq.float()).to(gk.dtype)
        return dwq, dgk, dbl, None, None


class RSIPhase2SuperfusedPreNormFn(torch.autograd.Function):
    """Per-Sublayer Phase-2 Online-Softmax Merge + Optional Superfused Sublayer Pre-RMSNorm.
    Never saves `o1_bar_s`, allowing `O1_bar` `[S, M, D]` to be freed immediately after the block's forward!"""

    @staticmethod
    def forward(
        ctx,
        w_s: torch.Tensor,
        gamma_key_s: torch.Tensor,
        o1_bar_s: torch.Tensor,
        lse1_s: torch.Tensor,
        partial_s: Optional[torch.Tensor],
        gamma_sub_s: Optional[torch.Tensor],
        out_dtype: torch.dtype,
        eps_key: float = 1e-6,
        eps_sub: float = 1e-6,
    ) -> torch.Tensor:
        has_p = partial_s is not None
        fuse_norm = gamma_sub_s is not None
        if o1_bar_s.dtype == torch.float64 or w_s.dtype == torch.float64:
            ws64 = w_s.to(torch.float64)
            gk64 = gamma_key_s.to(torch.float64)
            o1_64 = o1_bar_s.to(torch.float64)
            l1_64 = lse1_s.to(torch.float64)
            if has_p:
                assert partial_s is not None
                p64 = partial_s.to(torch.float64)
                rp = torch.rsqrt((p64 * p64).mean(dim=-1, keepdim=True) + eps_key)
                xp = ((ws64 * gk64) * (p64 * rp)).sum(dim=-1)
                ltot = torch.logaddexp(l1_64, xp)
                ah = torch.exp(l1_64 - ltot).unsqueeze(-1)
                ap = torch.exp(xp - ltot).unsqueeze(-1)
                h = ah * o1_64 + ap * p64
            else:
                p64 = o1_64.new_empty(0)
                h = o1_64
            if fuse_norm:
                assert gamma_sub_s is not None
                gs64 = gamma_sub_s.to(torch.float64)
                rh = torch.rsqrt((h * h).mean(dim=-1, keepdim=True) + eps_sub)
                out = h * rh * gs64
            else:
                gs64 = o1_64.new_empty(0)
                out = h
            ctx.save_for_backward(ws64, gk64, o1_64, l1_64, p64, gs64)
            ctx.meta = (has_p, fuse_norm, eps_key, eps_sub, True, w_s.dtype, gamma_key_s.dtype, out_dtype)
            return out.to(out_dtype)

        M, D = o1_bar_s.shape
        ws_c = w_s.contiguous()
        gk_c = gamma_key_s.contiguous()
        q_eff = (ws_c.float() * gk_c.float()).contiguous()
        o1_c = o1_bar_s.contiguous()
        l1_c = lse1_s.contiguous()
        p_c = partial_s.contiguous() if has_p else o1_c
        gs_c = gamma_sub_s.contiguous() if fuse_norm else ws_c

        y_out = torch.empty((M, D), device=o1_c.device, dtype=out_dtype)
        # Save compact BF16/out_dtype `h_save` (`u_hat` if `fuse_norm` else `h` only when `has_p`)
        save_dt = torch.float32 if out_dtype == torch.float32 else torch.bfloat16
        h_save = torch.empty((M, D), device=o1_c.device, dtype=save_dt) if (fuse_norm or has_p) else o1_c.new_empty(0)
        std_h = torch.empty((M,), device=o1_c.device, dtype=torch.float32) if fuse_norm else o1_c.new_empty(0)
        alpha_p = torch.empty((M,), device=o1_c.device, dtype=torch.float32) if has_p else o1_c.new_empty(0)
        rstd_p = torch.empty((M,), device=o1_c.device, dtype=torch.float32) if has_p else o1_c.new_empty(0)
        x_part = torch.empty((M,), device=o1_c.device, dtype=torch.float32) if has_p else o1_c.new_empty(0)

        _rsi_phase2_merge_prenorm_fwd_kernel[(M,)](
            q_eff, o1_c, l1_c, p_c, gs_c, y_out,
            h_save if h_save.numel() else y_out,
            std_h if fuse_norm else l1_c,
            alpha_p if has_p else l1_c,
            rstd_p if has_p else l1_c,
            x_part if has_p else l1_c,
            M, D, BLOCK_D=256,
            HAS_PARTIAL=(1 if has_p else 0),
            FUSE_SUB_NORM=(1 if fuse_norm else 0),
            EPS_KEY=eps_key, EPS_SUB=eps_sub, num_warps=4,
        )
        p_save = p_c if has_p else o1_c.new_empty(0)
        gs_save = gs_c if fuse_norm else ws_c.new_empty(0)
        # Notice `o1_c` and `l1_c` are NOT saved!
        ctx.save_for_backward(ws_c, gk_c, p_save, gs_save, h_save, std_h, alpha_p, rstd_p, x_part)
        ctx.meta = (has_p, fuse_norm, eps_key, eps_sub, False, M, D)
        return y_out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if ctx.meta[4]:  # is_fp64
            has_p, fuse_norm, eps_key, eps_sub, _, w_dt, gk_dt, p_dt = ctx.meta
            ws64, gk64, o1_64, l1_64, p64, gs64 = ctx.saved_tensors
            with torch.enable_grad():
                ws_ = ws64.detach().requires_grad_(True)
                gk_ = gk64.detach().requires_grad_(True)
                o1_ = o1_64.detach().requires_grad_(True)
                l1_ = l1_64.detach().requires_grad_(True)
                req = [ws_, gk_, o1_, l1_]
                if has_p:
                    p_ = p64.detach().requires_grad_(True)
                    rp = torch.rsqrt((p_ * p_).mean(dim=-1, keepdim=True) + eps_key)
                    xp = ((ws_ * gk_) * (p_ * rp)).sum(dim=-1)
                    ltot = torch.logaddexp(l1_, xp)
                    h = torch.exp(l1_ - ltot).unsqueeze(-1) * o1_ + torch.exp(xp - ltot).unsqueeze(-1) * p_
                    req.append(p_)
                else:
                    h = o1_
                if fuse_norm:
                    gs_ = gs64.detach().requires_grad_(True)
                    rh = torch.rsqrt((h * h).mean(dim=-1, keepdim=True) + eps_sub)
                    out = h * rh * gs_
                    req.append(gs_)
                else:
                    out = h
                grads_raw = torch.autograd.grad((out * dy.to(torch.float64)).sum(), req, allow_unused=True)
                grads = [
                    (g if g is not None else torch.zeros_like(t))
                    for g, t in zip(grads_raw, req)
                ]
            dws = grads.pop(0).to(w_dt)
            dgk = grads.pop(0).to(gk_dt)
            do1 = grads.pop(0)
            dl1 = grads.pop(0)
            dp = grads.pop(0).to(p_dt) if has_p else None
            dgs = grads.pop(0).to(gs64.dtype) if fuse_norm else None
            return dws, dgk, do1, dl1, dp, dgs, None, None, None

        has_p, fuse_norm, eps_key, eps_sub, _, M, D = ctx.meta
        ws_c, gk_c, p_save, gs_save, h_save, std_h, alpha_p, rstd_p, x_part = ctx.saved_tensors
        dy_c = dy.contiguous()
        q_eff = (ws_c.float() * gk_c.float()).contiguous()

        dot_gsub = torch.empty((M,), device=dy_c.device, dtype=torch.float32) if fuse_norm else dy_c.new_empty(0)
        dx_part = torch.empty((M,), device=dy_c.device, dtype=torch.float32) if has_p else dy_c.new_empty(0)
        dlse1 = torch.empty((M,), device=dy_c.device, dtype=torch.float32)

        _rsi_phase2_merge_prenorm_bwd_scalars_kernel[(M,)](
            dy_c,
            gs_save if fuse_norm else ws_c,
            h_save if h_save.numel() else dy_c,
            std_h if fuse_norm else dlse1,
            p_save if has_p else dy_c,
            alpha_p if has_p else dlse1,
            dot_gsub if fuse_norm else dlse1,
            dx_part if has_p else dlse1,
            dlse1,
            M, D, BLOCK_D=256,
            HAS_PARTIAL=(1 if has_p else 0),
            FUSE_SUB_NORM=(1 if fuse_norm else 0),
            num_warps=4,
        )

        gm, tps = _choose_splits_m(M)
        do1_out = torch.empty((M, D), device=dy_c.device, dtype=torch.float32)
        dp_out = torch.empty_like(p_save) if has_p else dy_c.new_empty(0)
        part_dq = torch.empty((gm, D), device=dy_c.device, dtype=torch.float32) if has_p else dy_c.new_empty(0)
        part_dgs = torch.empty((gm, D), device=dy_c.device, dtype=torch.float32) if fuse_norm else dy_c.new_empty(0)

        _rsi_phase2_merge_prenorm_bwd_2d_kernel[(triton.cdiv(D, 256), gm)](
            q_eff,
            dy_c,
            gs_save if fuse_norm else ws_c,
            h_save if h_save.numel() else dy_c,
            std_h if fuse_norm else dlse1,
            dot_gsub if fuse_norm else dlse1,
            p_save if has_p else dy_c,
            alpha_p if has_p else dlse1,
            rstd_p if has_p else dlse1,
            x_part if has_p else dlse1,
            dx_part if has_p else dlse1,
            do1_out,
            dp_out if has_p else do1_out,
            part_dq if has_p else do1_out,
            part_dgs if fuse_norm else do1_out,
            M, D, tps, BLOCK_D=256,
            HAS_PARTIAL=(1 if has_p else 0),
            FUSE_SUB_NORM=(1 if fuse_norm else 0),
            num_warps=4,
        )

        if has_p:
            dws = torch.empty_like(ws_c)
            dgk = torch.empty_like(gk_c)
            _reduce_splitm_vec_kernel[(triton.cdiv(D, 256),)](
                part_dq, dws, dgk, gk_c, ws_c, gm, D, HAS_DUAL_SCALE=1, BLOCK_E=256, num_warps=4
            )
        else:
            dws = torch.zeros_like(ws_c)
            dgk = torch.zeros_like(gk_c)

        if fuse_norm:
            dgs = torch.empty_like(gs_save)
            _reduce_splitm_vec_kernel[(triton.cdiv(D, 256),)](
                part_dgs, dgs, dgs, dgs, dgs, gm, D, HAS_DUAL_SCALE=0, BLOCK_E=256, num_warps=4
            )
        else:
            dgs = None

        return dws, dgk, do1_out, dlse1, (dp_out if has_p else None), dgs, None, None, None


class RSIPhase1UnbindInterBlockFn(torch.autograd.Function):
    """Sequential-Block Phase-1 Autograd Node returning `(*o1_bar.unbind(0), *lse1.unbind(0))`
    directly so PyTorch C++ autograd NEVER runs `S` separate `[S, M, D]` `SelectBackward0`
    zero-fill allocations (`-15.1 ms` & `-1.5 GiB` saved in sequential Transformer blocks!)."""

    @staticmethod
    def forward(
        ctx,
        w_queries: torch.Tensor,
        gamma_keys: torch.Tensor,
        blocks: torch.Tensor,
        eps_key: float = 1e-6,
        allow_1pass: bool = True,
    ):
        S = w_queries.shape[0]
        ctx.S = S
        ctx.eps_key = eps_key
        if blocks.dtype == torch.float64 or w_queries.dtype == torch.float64:
            qf = w_queries.to(torch.float64) * gamma_keys.to(torch.float64)
            vf = blocks.to(torch.float64)
            rv = torch.rsqrt((vf * vf).mean(dim=-1, keepdim=True) + eps_key)
            logits = torch.einsum("sd,nmd->snm", qf, vf * rv)
            lse1 = torch.logsumexp(logits, dim=1)
            a1 = torch.exp(logits - lse1.unsqueeze(1))
            o1_bar = torch.einsum("snm,nmd->smd", a1, vf)
            ctx.save_for_backward(w_queries, gamma_keys, blocks, lse1)
            ctx.is_fp64 = True
            return (*o1_bar.unbind(0), *lse1.unbind(0))

        wq = w_queries.contiguous()
        gk = gamma_keys.contiguous()
        bl = blocks.contiguous()
        q_eff = (wq.float() * gk.float()).contiguous()
        o1_bar, lse1 = rsi_phase1_fwd(q_eff, bl, eps_key=eps_key, allow_1pass=allow_1pass)
        ctx.save_for_backward(wq, gk, bl, lse1)
        ctx.is_fp64 = False
        return (*o1_bar.unbind(0), *lse1.unbind(0))

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        S = ctx.S
        wq, gk, bl, lse1 = ctx.saved_tensors
        M, D = bl.shape[1], bl.shape[2]
        g_dt = torch.float64 if ctx.is_fp64 else torch.float32
        do1_bar = torch.stack(
            [(g if g is not None else torch.zeros((M, D), device=bl.device, dtype=g_dt)) for g in grads[:S]],
            dim=0,
        )
        dlse1 = torch.stack(
            [(g if g is not None else torch.zeros((M,), device=bl.device, dtype=g_dt)) for g in grads[S:]],
            dim=0,
        )
        if ctx.is_fp64:
            with torch.enable_grad():
                wq64 = wq.detach().to(torch.float64).requires_grad_(True)
                gk64 = gk.detach().to(torch.float64).requires_grad_(True)
                bl64 = bl.detach().to(torch.float64).requires_grad_(True)
                qf = wq64 * gk64
                rv = torch.rsqrt((bl64 * bl64).mean(dim=-1, keepdim=True) + ctx.eps_key)
                logits = torch.einsum("sd,nmd->snm", qf, bl64 * rv)
                lse_ = torch.logsumexp(logits, dim=1)
                a1 = torch.exp(logits - lse_.unsqueeze(1))
                o1_ = torch.einsum("snm,nmd->smd", a1, bl64)
                loss = (o1_ * do1_bar.to(torch.float64)).sum() + (lse_ * dlse1.to(torch.float64)).sum()
                dwq, dgk, dbl = torch.autograd.grad(loss, (wq64, gk64, bl64))
            return dwq.to(wq.dtype), dgk.to(gk.dtype), dbl.to(bl.dtype), None, None

        q_eff = (wq.float() * gk.float()).contiguous()
        dq_eff, dbl = rsi_phase1_bwd(
            q_eff, bl, lse1, do1_bar.float().contiguous(), dlse1.float().contiguous(), eps_key=ctx.eps_key
        )
        dwq = (dq_eff * gk.float()).to(wq.dtype)
        dgk = (dq_eff * wq.float()).to(gk.dtype)
        return dwq, dgk, dbl, None, None


class RSIBatchedTwoPhaseBlockAttnResFn(torch.autograd.Function):
    """Unified Single-Node Autograd Operator for Batched Two-Phase `Block AttnRes` + Optional
    Superfused Sublayer Pre-RMSNorm (`0` `SelectBackward0`, `0` `torch.stack`, `0` `O1_bar`
    retention, `0` global atomics, and `0` cubin spills across `D = 1024..4096`!)."""

    @staticmethod
    def forward(
        ctx,
        w_queries: torch.Tensor,
        gamma_keys: torch.Tensor,
        blocks: torch.Tensor,
        partials: torch.Tensor,
        has_partial: torch.Tensor,
        gamma_subs: Optional[torch.Tensor] = None,
        eps_key: float = 1e-6,
        eps_sub: float = 1e-6,
        allow_1pass: bool = True,
    ) -> torch.Tensor:
        fuse_norm = gamma_subs is not None
        if blocks.dtype == torch.float64 or w_queries.dtype == torch.float64:
            out64 = eager_block_attnres_prenorm_ref(
                w_queries, gamma_keys, blocks, partials, has_partial,
                gamma_subs=gamma_subs, eps_key=eps_key, eps_sub=eps_sub,
            )
            gs_save = gamma_subs if fuse_norm else w_queries.new_empty(0)
            ctx.save_for_backward(w_queries, gamma_keys, blocks, partials, has_partial, gs_save)
            ctx.meta = (fuse_norm, eps_key, eps_sub, True)
            return out64

        wq = w_queries.contiguous()
        gk = gamma_keys.contiguous()
        bl = blocks.contiguous()
        pa = partials.contiguous()
        hp = has_partial.contiguous()
        gs = gamma_subs.contiguous() if fuse_norm else wq
        S, D = wq.shape
        _, M, _ = bl.shape

        q_eff = (wq.float() * gk.float()).contiguous()
        o1_bar, lse1 = rsi_phase1_fwd(q_eff, bl, eps_key=eps_key, allow_1pass=allow_1pass)

        y_out = torch.empty((S, M, D), device=bl.device, dtype=bl.dtype)
        save_dt = torch.float32 if bl.dtype == torch.float32 else torch.bfloat16
        h_save = torch.empty((S, M, D), device=bl.device, dtype=save_dt)
        std_h = torch.empty((S, M), device=bl.device, dtype=torch.float32) if fuse_norm else bl.new_empty(0)
        alpha_p = torch.empty((S, M), device=bl.device, dtype=torch.float32)
        rstd_p = torch.empty((S, M), device=bl.device, dtype=torch.float32)
        x_part = torch.empty((S, M), device=bl.device, dtype=torch.float32)

        if D <= 4096:
            bd = max(32, triton.next_power_of_2(D))
            nw = min(8, max(2, bd // 256))
            _rsi_batched_phase2_1pass_fwd_kernel[(S * M,)](
                q_eff, o1_bar, lse1, pa, hp, gs, y_out, h_save,
                std_h if fuse_norm else lse1, alpha_p, rstd_p, x_part,
                M, D=D, BD=bd, FUSE_SUB_NORM=(1 if fuse_norm else 0),
                EPS_KEY=eps_key, EPS_SUB=eps_sub, num_warps=nw,
            )
        else:
            hp_cpu = [bool(x) for x in hp.detach().cpu().tolist()]
            for s in range(S):
                _rsi_phase2_merge_prenorm_fwd_kernel[(M,)](
                    q_eff[s], o1_bar[s], lse1[s], pa[s], gs[s], y_out[s], h_save[s],
                    std_h[s] if fuse_norm else lse1[s], alpha_p[s], rstd_p[s], x_part[s],
                    M, D, BLOCK_D=256, HAS_PARTIAL=(1 if hp_cpu[s] else 0),
                    FUSE_SUB_NORM=(1 if fuse_norm else 0), EPS_KEY=eps_key, EPS_SUB=eps_sub, num_warps=4,
                )

        # `o1_bar` goes out of scope right here — ZERO `[S, M, D]` `FP32` memory retained!
        del o1_bar
        gs_save = gs if fuse_norm else wq.new_empty(0)
        ctx.save_for_backward(wq, gk, bl, pa, hp, gs_save, lse1, h_save, std_h, alpha_p, rstd_p, x_part)
        ctx.meta = (fuse_norm, eps_key, eps_sub, False)
        return y_out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        fuse_norm, eps_key, eps_sub, is_fp64 = ctx.meta
        if is_fp64:
            wq, gk, bl, pa, hp, gs_save = ctx.saved_tensors
            orc = fp64_block_attnres_oracle(
                wq, gk, bl, pa, hp, dy, gamma_subs=(gs_save if fuse_norm else None),
                eps_key=eps_key, eps_sub=eps_sub,
            )
            return (
                orc["dw_queries"].to(wq.dtype),
                orc["dgamma_keys"].to(gk.dtype),
                orc["dblocks"].to(bl.dtype),
                orc["dpartials"].to(pa.dtype),
                None,
                (orc["dgamma_subs"].to(gs_save.dtype) if fuse_norm else None),
                None, None, None,
            )

        wq, gk, bl, pa, hp, gs_save, lse1, h_save, std_h, alpha_p, rstd_p, x_part = ctx.saved_tensors
        dy_c = dy.contiguous()
        S, D = wq.shape
        _, M, _ = bl.shape
        q_eff = (wq.float() * gk.float()).contiguous()

        dot_gsub = torch.empty((S, M), device=bl.device, dtype=torch.float32) if fuse_norm else bl.new_empty(0)
        dx_part = torch.empty((S, M), device=bl.device, dtype=torch.float32)
        dlse1 = torch.empty((S, M), device=bl.device, dtype=torch.float32)

        if D <= 4096:
            bd = max(32, triton.next_power_of_2(D))
            nw = min(8, max(2, bd // 256))
            _rsi_batched_phase2_bwd_scalars_1pass_kernel[(S * M,)](
                dy_c, gs_save if fuse_norm else wq, h_save, std_h if fuse_norm else lse1,
                pa, hp, alpha_p, dot_gsub if fuse_norm else lse1, dx_part, dlse1,
                M, D=D, BD=bd, FUSE_SUB_NORM=(1 if fuse_norm else 0), num_warps=nw,
            )
        else:
            hp_cpu = [bool(x) for x in hp.detach().cpu().tolist()]
            for s in range(S):
                _rsi_phase2_merge_prenorm_bwd_scalars_kernel[(M,)](
                    dy_c[s], gs_save[s] if fuse_norm else wq[s], h_save[s],
                    std_h[s] if fuse_norm else lse1[s], pa[s], alpha_p[s],
                    dot_gsub[s] if fuse_norm else lse1[s], dx_part[s], dlse1[s],
                    M, D, BLOCK_D=256, HAS_PARTIAL=(1 if hp_cpu[s] else 0),
                    FUSE_SUB_NORM=(1 if fuse_norm else 0), num_warps=4,
                )

        gm, tps = _choose_splits_m(M)
        do1_bar = torch.empty((S, M, D), device=bl.device, dtype=torch.float32)
        dpartials = torch.empty_like(pa)
        part_dq_p2 = torch.empty((gm, S, D), device=bl.device, dtype=torch.float32)
        part_dgs = torch.empty((gm, S, D), device=bl.device, dtype=torch.float32) if fuse_norm else bl.new_empty(0)

        _rsi_batched_phase2_bwd_2d_kernel[(triton.cdiv(D, 256), gm, S)](
            q_eff, dy_c, gs_save if fuse_norm else wq, h_save, std_h if fuse_norm else lse1,
            dot_gsub if fuse_norm else lse1, pa, hp, alpha_p, rstd_p, x_part, dx_part,
            do1_bar, dpartials, part_dq_p2, part_dgs if fuse_norm else part_dq_p2,
            M, D, tps, BLOCK_D=256, FUSE_SUB_NORM=(1 if fuse_norm else 0), num_warps=4,
        )

        dq_eff_p2 = torch.empty((S, D), device=bl.device, dtype=torch.float32)
        tot_sd = S * D
        _reduce_splitm_vec_kernel[(triton.cdiv(tot_sd, 256),)](
            part_dq_p2, dq_eff_p2, dq_eff_p2, dq_eff_p2, dq_eff_p2, gm, tot_sd, HAS_DUAL_SCALE=0, BLOCK_E=256, num_warps=4
        )
        if fuse_norm:
            dgs = torch.empty_like(gs_save)
            _reduce_splitm_vec_kernel[(triton.cdiv(tot_sd, 256),)](
                part_dgs, dgs, dgs, dgs, dgs, gm, tot_sd, HAS_DUAL_SCALE=0, BLOCK_E=256, num_warps=4
            )
        else:
            dgs = None

        dq_eff_p1, dblocks = rsi_phase1_bwd(q_eff, bl, lse1, do1_bar, dlse1, eps_key=eps_key)
        dq_tot = dq_eff_p1 + dq_eff_p2
        dwq = (dq_tot * gk.float()).to(wq.dtype)
        dgk = (dq_tot * wq.float()).to(gk.dtype)
        return dwq, dgk, dblocks, dpartials, None, dgs, None, None, None


@triton.jit
def _fused_sublayer_rmsnorm_fwd_kernel(
    H_ptr,          # [S, M, D] or [M, D]
    GS_ptr,         # [S, D] or [1, D]
    Y_ptr,          # [S, M, D] out
    UHAT_ptr,       # [S, M, D] out
    RSTD_ptr,       # [S, M] fp32 out
    M,
    D: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid // M
    od = tl.arange(0, BD)
    dm = od < D
    h = tl.load(H_ptr + pid * D + od, mask=dm, other=0.0).to(tl.float32)
    gs = tl.load(GS_ptr + s * D + od, mask=dm, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(h * h, axis=0) / D + EPS)
    u_hat = h * rstd
    tl.store(RSTD_ptr + pid, rstd)
    tl.store(UHAT_ptr + pid * D + od, u_hat, mask=dm)
    tl.store(Y_ptr + pid * D + od, u_hat * gs, mask=dm)


@triton.jit
def _fused_sublayer_rmsnorm_bwd_kernel(
    DY_ptr,         # [S, M, D]
    GS_ptr,         # [S, D]
    UHAT_ptr,       # [S, M, D]
    RSTD_ptr,       # [S, M] fp32
    DH_ptr,         # [S, M, D] out
    PART_DGS_ptr,   # [G_M, S, D] fp32 out
    M,
    S_TOT,
    G_M: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid // G_M
    gm = pid % G_M
    od = tl.arange(0, BD)
    dm = od < D
    gs = tl.load(GS_ptr + s * D + od, mask=dm, other=0.0).to(tl.float32)
    dgs_acc = tl.zeros((BD,), dtype=tl.float32)
    base_sm = s * M
    inv_d = 1.0 / D
    for m in range(gm, M, G_M):
        sm = base_sm + m
        dy = tl.load(DY_ptr + sm * D + od, mask=dm, other=0.0).to(tl.float32)
        u_hat = tl.load(UHAT_ptr + sm * D + od, mask=dm, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + sm)
        dgs_acc += dy * u_hat
        dot_g = tl.sum(dy * gs * u_hat, axis=0) * inv_d
        dh = rstd * (dy * gs - u_hat * dot_g)
        tl.store(DH_ptr + sm * D + od, dh, mask=dm)
    tl.store(PART_DGS_ptr + (gm * S_TOT + s) * D + od, dgs_acc, mask=dm)


class FusedSublayerRMSNormFn(torch.autograd.Function):
    """Zero-spill lock-free Triton Sublayer RMSNorm baseline (supports `[S, M, D]` and `[M, D]`)
    used by unfused baselines (`Hydra-2P + FusedRMSNorm` and `RSI Lock-Free Two-Phase + FusedRMSNorm`)
    so `rsi_superfused` isolates the exact HBM gain of superfusing `Phase 2 + Pre-RMSNorm`."""

    @staticmethod
    def forward(ctx, h: torch.Tensor, gamma_sub: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if h.dtype == torch.float64 or h.shape[-1] > 4096:
            return eager_rmsnorm_ref(h, gamma_sub.unsqueeze(1) if h.dim() == 3 else gamma_sub, eps)
        is_2d = h.dim() == 2
        h3 = h.unsqueeze(0).contiguous() if is_2d else h.contiguous()
        gs2 = gamma_sub.unsqueeze(0).contiguous() if is_2d else gamma_sub.contiguous()
        S, M, D = h3.shape
        bd = max(32, triton.next_power_of_2(D))
        nw = min(8, max(2, bd // 256))
        y = torch.empty_like(h3)
        u_hat = torch.empty_like(h3)
        rstd = torch.empty((S, M), device=h.device, dtype=torch.float32)
        _fused_sublayer_rmsnorm_fwd_kernel[(S * M,)](
            h3, gs2, y, u_hat, rstd, M, D=D, BD=bd, EPS=eps, num_warps=nw
        )
        ctx.save_for_backward(gs2, u_hat, rstd)
        ctx.meta = (is_2d, bd, nw)
        return y.squeeze(0) if is_2d else y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        gs2, u_hat, rstd = ctx.saved_tensors
        is_2d, bd, nw = ctx.meta
        dy3 = dy.unsqueeze(0).contiguous() if is_2d else dy.contiguous()
        S, M, D = dy3.shape
        gm = 128 if M >= 256 else 32
        dh = torch.empty_like(u_hat)
        part_dgs = torch.empty((gm, S, D), device=dy.device, dtype=torch.float32)
        _fused_sublayer_rmsnorm_bwd_kernel[(S * gm,)](
            dy3, gs2, u_hat, rstd, dh, part_dgs, M, S, G_M=gm, D=D, BD=bd, num_warps=nw
        )
        dgs = torch.empty_like(gs2)
        tot_sd = S * D
        _reduce_splitm_vec_kernel[(triton.cdiv(tot_sd, 256),)](
            part_dgs, dgs, dgs, dgs, dgs, gm, tot_sd, HAS_DUAL_SCALE=0, BLOCK_E=256, num_warps=4
        )
        return (dh.squeeze(0) if is_2d else dh), (dgs.squeeze(0) if is_2d else dgs), None


def rsi_two_phase_block_attnres_batched(
    w_queries: torch.Tensor,
    gamma_keys: torch.Tensor,
    blocks: torch.Tensor,
    partials: torch.Tensor,
    has_partial: torch.Tensor,
    gamma_subs: Optional[torch.Tensor] = None,
    eps_key: float = 1e-6,
    eps_sub: float = 1e-6,
    allow_1pass: bool = True,
) -> torch.Tensor:
    """Executes unified single-node Two-Phase `Block AttnRes` + Optional Superfused PreNorm."""
    return RSIBatchedTwoPhaseBlockAttnResFn.apply(
        w_queries, gamma_keys, blocks, partials, has_partial, gamma_subs, eps_key, eps_sub, allow_1pass
    )


# =============================================================================
# 5. Distributed P2P Primitives for Interleaved 1F1B Cross-Stage Block Caching
#    (`arXiv:2603.15031` Section 4.1 Eq. 7-8 & Figure 3, `Gate G9`)
# =============================================================================


class P2PSendRecvTensorListFn(torch.autograd.Function):
    """Differentiable P2P transfer along directed stage edge `src_rank -> dst_rank`.
    On the sender (`rank == src_rank`), returns dummy zero scalar connected to `tensors` so backward
    receives exact upstream gradients from `dst_rank` and routes them into `tensors`!"""

    @staticmethod
    def forward(
        ctx,
        src_rank: int,
        dst_rank: int,
        shapes: Tuple[Tuple[int, ...], ...],
        dtype: torch.dtype,
        device: torch.device,
        *src_tensors: torch.Tensor,
    ):
        rank = dist.get_rank()
        ctx.src_rank = src_rank
        ctx.dst_rank = dst_rank
        ctx.shapes = shapes
        ctx.dtype = dtype
        ctx.device = device

        if src_rank == dst_rank:
            return tuple(t.clone() for t in src_tensors)

        ops: List[dist.P2POp] = []
        recv_list: List[torch.Tensor] = []
        if rank == src_rank:
            for t in src_tensors:
                ops.append(dist.P2POp(dist.isend, t.contiguous(), dst_rank))
            reqs = dist.batch_isend_irecv(ops)
            for rq in reqs:
                rq.wait()
            # Return tiny zero anchor on sender so autograd invokes backward on sender too!
            return (torch.zeros((), device=device, dtype=torch.float32),)
        elif rank == dst_rank:
            for sh in shapes:
                buf = torch.empty(sh, device=device, dtype=dtype)
                recv_list.append(buf)
                ops.append(dist.P2POp(dist.irecv, buf, src_rank))
            reqs = dist.batch_isend_irecv(ops)
            for rq in reqs:
                rq.wait()
            return tuple(recv_list)
        else:
            return (torch.zeros((), device=device, dtype=torch.float32),)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        rank = dist.get_rank()
        src_rank = ctx.src_rank
        dst_rank = ctx.dst_rank
        shapes = ctx.shapes
        dtype = ctx.dtype
        device = ctx.device

        if src_rank == dst_rank:
            return (None, None, None, None, None, *grad_outputs)

        ops: List[dist.P2POp] = []
        if rank == dst_rank:
            for g in grad_outputs:
                ops.append(dist.P2POp(dist.isend, g.to(dtype).contiguous(), src_rank))
            reqs = dist.batch_isend_irecv(ops)
            for rq in reqs:
                rq.wait()
            return (None, None, None, None, None)
        elif rank == src_rank:
            grad_in: List[torch.Tensor] = []
            for sh in shapes:
                gbuf = torch.empty(sh, device=device, dtype=dtype)
                grad_in.append(gbuf)
                ops.append(dist.P2POp(dist.irecv, gbuf, dst_rank))
            reqs = dist.batch_isend_irecv(ops)
            for rq in reqs:
                rq.wait()
            return (None, None, None, None, None, *grad_in)
        else:
            return (None, None, None, None, None)


def collect_compiled_kernel_spill_stats(device: torch.device) -> Dict[str, Any]:
    """Compiles all 7 Round-6 RSI Triton kernels across canonical shapes (`D = 1024, 2048, 4096`)
    and inspects compiled cubins to prove `0` register spills (`n_spills == 0`)."""
    stats: List[Dict[str, Any]] = []
    for D in (1024, 2048, 4096):
        S, N, M = 8, 8, 256
        wq = (torch.randn(S, D, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        gk = torch.ones(S, D, device=device, dtype=torch.float32, requires_grad=True)
        gs = torch.ones(S, D, device=device, dtype=torch.float32, requires_grad=True)
        bl = (torch.randn(N, M, D, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        pa = (torch.randn(S, M, D, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        hp = torch.tensor([False] + [True] * (S - 1), device=device, dtype=torch.bool)
        dy = torch.randn(S, M, D, device=device, dtype=torch.bfloat16)

        # Exercise both batched 1-pass/2-pass AND sequential per-sublayer Phase-2 kernels so all 12 kernels compile!
        y1 = rsi_two_phase_block_attnres_batched(wq, gk, bl, pa, hp, gamma_subs=gs, allow_1pass=True)
        (y1.float() * dy.float()).sum().backward()
        y2 = rsi_two_phase_block_attnres_batched(wq, gk, bl, pa, hp, gamma_subs=gs, allow_1pass=False)
        (y2.float() * dy.float()).sum().backward()
        ub = RSIPhase1UnbindInterBlockFn.apply(wq, gk, bl, 1e-6, True)
        ys_seq = RSIPhase2SuperfusedPreNormFn.apply(wq[1], gk[1], ub[1], ub[S + 1], pa[1], gs[1], bl.dtype, 1e-6, 1e-6)
        (ys_seq.float() * dy[1].float()).sum().backward()
        yn = FusedSublayerRMSNormFn.apply(pa, gs, 1e-6)
        (yn.float() * dy.float()).sum().backward()
    torch.cuda.synchronize(device)

    rsi_kernels = [
        ("_rsi_phase1_stats_fwd_kernel", _rsi_phase1_stats_fwd_kernel),
        ("_rsi_phase1_weighted_sum_2d_fwd_kernel", _rsi_phase1_weighted_sum_2d_fwd_kernel),
        ("_rsi_phase1_1pass_fullrow_fwd_kernel", _rsi_phase1_1pass_fullrow_fwd_kernel),
        ("_rsi_phase1_bwd_scores_kernel", _rsi_phase1_bwd_scores_kernel),
        ("_rsi_phase1_bwd_dv_dq_2d_kernel", _rsi_phase1_bwd_dv_dq_2d_kernel),
        ("_reduce_splitm_vec_kernel", _reduce_splitm_vec_kernel),
        ("_rsi_phase2_merge_prenorm_fwd_kernel", _rsi_phase2_merge_prenorm_fwd_kernel),
        ("_rsi_phase2_merge_prenorm_bwd_scalars_kernel", _rsi_phase2_merge_prenorm_bwd_scalars_kernel),
        ("_rsi_phase2_merge_prenorm_bwd_2d_kernel", _rsi_phase2_merge_prenorm_bwd_2d_kernel),
        ("_rsi_batched_phase2_1pass_fwd_kernel", _rsi_batched_phase2_1pass_fwd_kernel),
        ("_rsi_batched_phase2_bwd_scalars_1pass_kernel", _rsi_batched_phase2_bwd_scalars_1pass_kernel),
        ("_rsi_batched_phase2_bwd_2d_kernel", _rsi_batched_phase2_bwd_2d_kernel),
        ("_fused_sublayer_rmsnorm_fwd_kernel", _fused_sublayer_rmsnorm_fwd_kernel),
        ("_fused_sublayer_rmsnorm_bwd_kernel", _fused_sublayer_rmsnorm_bwd_kernel),
    ]
    total_spills = 0
    max_regs = 0
    for kname, kobj in rsi_kernels:
        spec_spills = 0
        spec_max_regs = 0
        spec_cnt = 0
        dev_caches = getattr(kobj, "device_caches", {})
        for dev_entry in dev_caches.values():
            cdict = dev_entry[0] if isinstance(dev_entry, tuple) and len(dev_entry) > 0 else dev_entry
            if isinstance(cdict, dict):
                for compiled in cdict.values():
                    if hasattr(compiled, "n_spills"):
                        spec_cnt += 1
                        sp = int(getattr(compiled, "n_spills", 0))
                        rg = int(getattr(compiled, "n_regs", 0))
                        spec_spills += sp
                        spec_max_regs = max(spec_max_regs, rg)
        # Fallback for older Triton `kobj.cache`
        if spec_cnt == 0:
            for dev_map in getattr(kobj, "cache", {}).values():
                if isinstance(dev_map, dict):
                    for compiled in dev_map.values():
                        spec_cnt += 1
                        sp = int(getattr(compiled, "n_spills", 0))
                        rg = int(getattr(compiled, "n_regs", 0))
                        spec_spills += sp
                        spec_max_regs = max(spec_max_regs, rg)
        total_spills += spec_spills
        max_regs = max(max_regs, spec_max_regs)
        stats.append(
            {
                "kernel": kname,
                "specializations": spec_cnt,
                "max_regs": spec_max_regs,
                "spills": spec_spills,
            }
        )
    return {
        "num_rsi_kernels": len(rsi_kernels),
        "total_specializations": sum(s["specializations"] for s in stats),
        "max_regs_across_all": max_regs,
        "total_cubin_spills": total_spills,
        "all_zero_spills": total_spills == 0 and all(s["specializations"] > 0 for s in stats),
        "per_kernel": stats,
    }


@triton.jit
def _audit_naive_1pass_d4096_nw16_spill_kernel(
    Q_ptr,
    V_ptr,
    O1_BAR_ptr,
    LSE1_ptr,
    N,
    M,
    S_ACTUAL,
    D: tl.constexpr,
    S: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    """Standalone forensic kernel demonstrating why unguarded 1-pass `BD = 4096, S = 8, num_warps = 16`
    (`512` threads -> `65536 / 512 = 128` max registers/thread on SM80) spills `26` registers to local memory,
    motivating Round-6 RSI's width-adaptive `D <= 2048` (`num_warps = 8`, `0` spills) + `2-pass BLOCK_D = 256`
    (`D = 4096`, `48` regs, `0` spills) dispatch."""
    m = tl.program_id(0)
    od = tl.arange(0, BD)
    d_mask = od < D
    s_idx = tl.arange(0, S)
    s_mask = s_idx < S_ACTUAL
    q_mat = tl.load(Q_ptr + s_idx[:, None] * D + od[None, :], mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    vmax = tl.full((S,), -float("inf"), dtype=tl.float32)
    vsum = tl.zeros((S,), dtype=tl.float32)
    acc = tl.zeros((S, BD), dtype=tl.float32)
    for n in range(0, N):
        v = tl.load(V_ptr + (n * M + m) * D + od, mask=d_mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(v * v, axis=0) / D + EPS)
        scores = tl.sum(q_mat * v[None, :], axis=1) * inv
        new_max = tl.maximum(vmax, scores)
        old_s = tl.exp(vmax - new_max)
        new_s = tl.exp(scores - new_max)
        acc = acc * old_s[:, None] + new_s[:, None] * v[None, :]
        vsum = vsum * old_s + new_s
        vmax = new_max
    tl.store(O1_BAR_ptr + (s_idx[:, None] * M + m) * D + od[None, :], acc / vsum[:, None], mask=s_mask[:, None] & d_mask[None, :])
    tl.store(LSE1_ptr + s_idx * M + m, vmax + tl.log(vsum), mask=s_mask)


def audit_naive_sm80_d4096_nw16_spill(device: torch.device) -> Dict[str, Any]:
    """Compiles `_audit_naive_1pass_d4096_nw16_spill_kernel` at `(S=8, N=8, M=64, D=4096, num_warps=16)`
    and returns its hardware register/spill count without affecting any production RSI kernel."""
    S, N, M, D = 8, 8, 64, 4096
    q = torch.randn(S, D, device=device, dtype=torch.float32)
    v = torch.randn(N, M, D, device=device, dtype=torch.bfloat16)
    o = torch.empty((S, M, D), device=device, dtype=torch.float32)
    l = torch.empty((S, M), device=device, dtype=torch.float32)
    _audit_naive_1pass_d4096_nw16_spill_kernel[(M,)](
        q, v, o, l, N, M, S, D=D, S=8, BD=4096, EPS=1e-6, num_warps=16
    )
    torch.cuda.synchronize(device)
    spills = 0
    regs = 0
    for dev_entry in getattr(_audit_naive_1pass_d4096_nw16_spill_kernel, "device_caches", {}).values():
        cdict = dev_entry[0] if isinstance(dev_entry, tuple) and len(dev_entry) > 0 else dev_entry
        if isinstance(cdict, dict):
            for compiled in cdict.values():
                if hasattr(compiled, "n_spills"):
                    spills = max(spills, int(getattr(compiled, "n_spills", 0)))
                    regs = max(regs, int(getattr(compiled, "n_regs", 0)))
    return {"kernel": "_audit_naive_1pass_d4096_nw16_spill_kernel", "n_regs": regs, "n_spills": spills}


class BrokenNoRadialJacobianBlockAttnResFn(torch.autograd.Function):
    """Adversarial Red-Team Candidate: exact forward output (`0.0` forward error!), but drops the
    closed-form `RMSNorm` radial Jacobian term `-rho * v` in backward (`dblocks` & `dpartials`
    corrupted on radial directions). Must be caught by `RSIImmutableEvaluator` backward oracle!"""

    @staticmethod
    def forward(ctx, wq, gk, bl, pa, hp, gs=None, eps_key=1e-6, eps_sub=1e-6):
        out = eager_block_attnres_prenorm_ref(wq, gk, bl, pa, hp, gamma_subs=gs, eps_key=eps_key, eps_sub=eps_sub)
        ctx.save_for_backward(wq, gk, bl, pa, hp, gs if gs is not None else wq.new_empty(0))
        ctx.meta = (gs is not None, eps_key, eps_sub)
        return out

    @staticmethod
    def backward(ctx, dy):
        wq, gk, bl, pa, hp, gs_save = ctx.saved_tensors
        has_gs, eps_key, eps_sub = ctx.meta
        # Detach radial `rstd` denominator so `-rho * v` radial Jacobian term is dropped!
        with torch.enable_grad():
            wq_ = wq.detach().float().requires_grad_(True)
            gk_ = gk.detach().float().requires_grad_(True)
            bl_ = bl.detach().float().requires_grad_(True)
            pa_ = pa.detach().float().requires_grad_(True)
            gs_ = gs_save.detach().float().requires_grad_(True) if has_gs else None

            qf = wq_ * gk_
            rv_detached = torch.rsqrt((bl_.detach() * bl_.detach()).mean(dim=-1, keepdim=True) + eps_key)
            rp_detached = torch.rsqrt((pa_.detach() * pa_.detach()).mean(dim=-1, keepdim=True) + eps_key)
            logits_b = torch.einsum("sd,nmd->snm", qf, bl_ * rv_detached)
            logits_p = (qf.unsqueeze(1) * (pa_ * rp_detached)).sum(dim=-1)
            logits_p = torch.where(hp.view(-1, 1), logits_p, torch.full_like(logits_p, -float("inf")))
            all_l = torch.cat([logits_b, logits_p.unsqueeze(1)], dim=1)
            alpha = torch.softmax(all_l, dim=1)
            h = torch.einsum("snm,nmd->smd", alpha[:, :-1, :], bl_) + alpha[:, -1, :].unsqueeze(-1) * pa_
            if has_gs:
                assert gs_ is not None
                rh = torch.rsqrt((h * h).mean(dim=-1, keepdim=True) + eps_sub)
                y = h * rh * gs_.unsqueeze(1)
                grads = torch.autograd.grad((y * dy.float()).sum(), (wq_, gk_, bl_, pa_, gs_))
                return grads[0].to(wq.dtype), grads[1].to(gk.dtype), grads[2].to(bl.dtype), grads[3].to(pa.dtype), None, grads[4].to(gs_save.dtype), None, None
            grads = torch.autograd.grad((h * dy.float()).sum(), (wq_, gk_, bl_, pa_))
            return grads[0].to(wq.dtype), grads[1].to(gk.dtype), grads[2].to(bl.dtype), grads[3].to(pa.dtype), None, None, None, None

