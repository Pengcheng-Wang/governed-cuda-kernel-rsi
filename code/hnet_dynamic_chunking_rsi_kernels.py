#!/usr/bin/env python3
"""Round-7 RSI Kernel Library for H-Net Dynamic Chunking (`arXiv:2507.07955`, `goombalab/hnet`).

Implements:
1. Exact `FP64` oracles for:
   - Adjacent Cosine-Similarity Boundary Routing (`Eq. 4`, exact PyTorch `F.normalize` `max(||x||, eps)`
     piecewise VJP + `argmax([1-p, p])` `p > 0.5` tie rule + empty-doc-safe forced `p_start = 1.0`),
   - Ragged Prefix-Scan Boundary Compaction (`Eq. 2`),
   - Exact Division-Free Scalar-Gate Affine EMA Smoothing (`Eq. 5`, zero `1e-4` clamp / `log` / `div` drift),
   - Confidence-Weighted Straight-Through Estimator (`STE`, `Eq. 6-9`) Upsampling + `FP32` Skip Residual (`Eq. 3`),
   - Token-Weighted Masked/Packed Ratio Loss (`Eq. 10`).
2. Official `goombalab/hnet` (`dc.py`, `hnet.py`, `train.py`) faithful baselines & upstream defect auditors:
   - `BF16` Surrogate `mamba_chunk_scan` `dt = -log(1 - p)` & `x = z / dt` quantization/amplification audit,
   - Variable-length `mask`-mode unmasked `load_balancing_loss` denominator & pad-gradient audit,
   - `ChunkLayer` `int(max())` host-device `DtoH` sync (`torch.compile(fullgraph=True)` graph break) &
     dual `O(B L log L)` `torch.argsort` + `GatherBackward0` global `scatter_add_` atomic contention.
3. Zero-Spill (`0` cubin spills) Lock-Free Round-7 RSI Triton Kernels:
   - `FusedHNetCosineRouterFn`: 1-pass scalar-norm saved-state forward + zero-reduction 2D `(d_blk, t)` backward,
   - `ZeroSortPrefixChunkPackFn`: `O(T)` prefix-sum compaction + span table (empty-doc safe, zero `argsort`),
   - `ParallelScalarAffineEMAFn`: 2-level (`BC = 64`) associative `(a, u)` forward & reverse-adjoint scan,
   - `FusedDechunkEMASmoothUpsampleSTEResidualFn`: superfused compact-`[M, D]` `z_bar` retention +
     lock-free span-segmented `d_z_bar` reduction (adaptive Split-Span for collapsed `M` regimes, `0` atomics) +
     fused per-byte `STE` confidence dot-product + reverse parallel EMA adjoint.
4. Distributed 8-GPU NVSwitch Whole-Document Cost-LPT `AllToAllv` + Global Token-Weighted Ratio Loss primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ==============================================================================
# 1. Exact Document-Boundary Metadata Kernel (Empty-Document Safe, Zero Sync)
# ==============================================================================


@triton.jit
def _hnet_doc_metadata_kernel(
    CU_ptr,
    DOC_ID_ptr,
    IS_START_ptr,
    IS_LAST_ptr,
    B_DOCS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    doc = tl.program_id(0)
    if doc >= B_DOCS:
        return
    s = tl.load(CU_ptr + doc).to(tl.int32)
    e = tl.load(CU_ptr + doc + 1).to(tl.int32)
    length = e - s
    if length <= 0:
        return
    offs = tl.arange(0, BLOCK_T)
    idx = 0
    while idx < length:
        cur = s + idx + offs
        m = (idx + offs) < length
        tl.store(DOC_ID_ptr + cur, doc, mask=m)
        tl.store(IS_START_ptr + cur, (idx + offs == 0).to(tl.int8), mask=m)
        tl.store(IS_LAST_ptr + cur, (idx + offs == (length - 1)).to(tl.int8), mask=m)
        idx += BLOCK_T


def build_packed_doc_metadata(cu_seqlens: torch.Tensor, total_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds `(doc_id[T], is_doc_start[T], is_doc_last[T])` without host DtoH syncs; safe on empty docs."""
    dev = cu_seqlens.device
    b_docs = int(cu_seqlens.numel() - 1)
    doc_id = torch.zeros((total_tokens,), dtype=torch.int32, device=dev)
    is_start = torch.zeros((total_tokens,), dtype=torch.int8, device=dev)
    is_last = torch.zeros((total_tokens,), dtype=torch.int8, device=dev)
    if total_tokens == 0 or b_docs <= 0:
        return doc_id, is_start, is_last
    cu_i32 = cu_seqlens.contiguous().to(torch.int32)
    if b_docs <= 256:
        _hnet_doc_metadata_kernel[(b_docs,)](
            cu_i32,
            doc_id,
            is_start,
            is_last,
            B_DOCS=b_docs,
            BLOCK_T=256,
            num_warps=4,
        )
    else:
        starts = cu_i32[:-1]
        ends = cu_i32[1:]
        non_empty = ends > starts
        valid_starts = starts[non_empty].to(torch.long)
        valid_lasts = (ends[non_empty] - 1).to(torch.long)
        is_start[valid_starts] = 1
        is_last[valid_lasts] = 1
        doc_id = torch.searchsorted(cu_i32[1:], torch.arange(total_tokens, device=dev, dtype=torch.int32), right=True).to(torch.int32)
    return doc_id, is_start, is_last


# ==============================================================================
# 2. Exact FP64 Reference Oracles & Locally-Anchored Differentiable Surrogates
# ==============================================================================


def fp64_hnet_router_oracle(
    u: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact FP64 `goombalab/hnet` `RoutingModule` (`dc.py` L90-137) on packed `[T, D]` sequences.

    Matches PyTorch `F.normalize(..., p=2, dim=-1, eps=1e-12)` (`x / max(||x||_2, eps)`)
    and `argmax(stack([1 - p, p], dim=-1))` (`boundary_mask = p > 0.5`), forcing `p = 1.0, b = True`
    on every non-empty document start `cu_seqlens[d]` (`cu[d+1] > cu[d]`).
    """
    u64 = u.to(torch.float64)
    v64 = v.to(torch.float64)
    t_len = u64.shape[0]
    if t_len == 0:
        empty = torch.zeros((0,), dtype=torch.float64, device=u.device)
        return empty, empty.to(torch.bool), empty

    u_hat = F.normalize(u64, p=2.0, dim=-1, eps=eps)
    v_hat = F.normalize(v64, p=2.0, dim=-1, eps=eps)
    if t_len > 1:
        cos_sim = torch.sum(u_hat[:-1] * v_hat[1:], dim=-1)
        p_tail = torch.clamp((1.0 - cos_sim) * 0.5, min=0.0, max=1.0)
        p = torch.cat([torch.ones((1,), dtype=torch.float64, device=u.device), p_tail], dim=0)
    else:
        p = torch.ones((1,), dtype=torch.float64, device=u.device)

    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    valid_starts = starts[ends > starts].to(torch.long)
    if valid_starts.numel() > 0:
        mask_force = torch.zeros((t_len,), dtype=torch.bool, device=u.device)
        mask_force[valid_starts] = True
        p = torch.where(mask_force, torch.ones_like(p), p)

    # PyTorch argmax([1 - p, p]): index 1 wins strictly when p > 1 - p <=> p > 0.5
    b_mask = p > 0.5
    c_conf = torch.where(b_mask, p, 1.0 - p)
    return p, b_mask, c_conf


def fp64_hnet_affine_ema_oracle(
    z_hat: torch.Tensor,
    p_sel: torch.Tensor,
    next_cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Exact FP64 Paper Eq. (5) Smoothing Module: `z_bar_j = P_j * z_hat_j + (1 - P_j) * z_bar_{j-1}` per doc."""
    z64 = z_hat.to(torch.float64)
    p64 = p_sel.to(torch.float64)
    m_len, d_dim = z64.shape
    if m_len == 0:
        return torch.zeros_like(z64)
    cu = next_cu_seqlens.detach().cpu().tolist()
    outs: list[torch.Tensor] = []
    for d_idx in range(len(cu) - 1):
        s, e = int(cu[d_idx]), int(cu[d_idx + 1])
        if e <= s:
            continue
        prev = torch.zeros((d_dim,), dtype=torch.float64, device=z64.device)
        for j in range(s, e):
            pj = p64[j]
            prev = pj * z64[j] + (1.0 - pj) * prev
            outs.append(prev)
    return torch.stack(outs, dim=0) if outs else torch.zeros_like(z64)


class _ExactSTEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(c)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        return grad_out


def fp64_hnet_dc_pipeline_oracle(
    x_enc: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_res: torch.Tensor,
    w_main: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    dechunk_round_dtype: torch.dtype | None = None,
    target_ratio_n: float = 4.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Complete FP64 oracle for `Router -> Pack -> Main -> EMA -> Upsample*STE + Residual` + Ratio Loss."""
    x64 = x_enc.to(torch.float64)
    wq64 = w_q.to(torch.float64)
    wk64 = w_k.to(torch.float64)
    wres64 = w_res.to(torch.float64)
    wmain64 = w_main.to(torch.float64)

    u = x64 @ wq64.t()
    v = x64 @ wk64.t()
    p, b_mask, c_conf = fp64_hnet_router_oracle(u, v, cu_seqlens, eps=eps)

    b_i64 = b_mask.to(torch.long)
    pref_pad = F.pad(torch.cumsum(b_i64, dim=0), (1, 0), value=0)
    next_cu = pref_pad[cu_seqlens.to(torch.long)]
    plug_idx = pref_pad[1:] - 1

    x_packed = x64[b_mask]
    p_sel = p[b_mask]
    # Nonlinear smooth main proxy in FP64 so 2nd/3rd derivatives are rich and non-degenerate
    z_hat = F.silu(x_packed) @ wmain64.t() + x_packed

    z_bar = fp64_hnet_affine_ema_oracle(z_hat, p_sel, next_cu)
    if dechunk_round_dtype is not None:
        z_bar_eff = z_bar.to(dechunk_round_dtype).to(torch.float64)
    else:
        z_bar_eff = z_bar

    z_tilde = z_bar_eff.index_select(0, plug_idx)
    ste_c = _ExactSTEFn.apply(c_conf).unsqueeze(-1)
    res = x64 @ wres64.t()
    z_out = z_tilde * ste_c + res

    f_frac = b_mask.to(torch.float64).mean().detach()
    g_mean = p.mean()
    ratio_loss = (target_ratio_n / (target_ratio_n - 1.0)) * (
        (target_ratio_n - 1.0) * f_frac * g_mean + (1.0 - f_frac) * (1.0 - g_mean)
    )
    return {
        "z_out": z_out,
        "p": p,
        "b_mask": b_mask,
        "z_bar": z_bar,
        "next_cu_seqlens": next_cu,
        "ratio_loss": ratio_loss,
    }


# ==============================================================================
# 3. Zero-Spill Round-7 RSI Triton Kernels
# ==============================================================================


@triton.jit
def _audit_naive_router_bwd_d4096_nw16_spill_kernel(
    U_ptr,
    V_ptr,
    DP_ptr,
    DU_ptr,
    DV_ptr,
    stride_t,
    T_LEN,
    BD: tl.constexpr,
):
    """Forensic audit kernel demonstrating SM80 register spills (`n_spills > 0`) when a naive monolithic
    8-token-per-CTA 1-pass `BD = 4096, num_warps = 16` (`512` threads -> `128` max regs/thread on SM80)
    fusion holds `[8, 4096]` `FP32` router+EMA tiles in registers across reductions, motivating Round-7 RSI's
    saved-scalar-norm 2D `(d_blk, t)` (`<= 32` regs, `0` spills) decomposition."""
    pid = tl.program_id(0)
    t0 = pid * 8 + 1
    if t0 + 8 >= T_LEN:
        return
    offs = tl.arange(0, BD)
    r_idx = tl.arange(0, 8)
    u_mat = tl.load(U_ptr + (t0 + r_idx[:, None] - 1) * stride_t + offs[None, :]).to(tl.float32)
    v_mat = tl.load(V_ptr + (t0 + r_idx[:, None]) * stride_t + offs[None, :]).to(tl.float32)
    nu = tl.maximum(tl.sqrt(tl.sum(u_mat * u_mat, axis=1)), 1e-12)
    nv = tl.maximum(tl.sqrt(tl.sum(v_mat * v_mat, axis=1)), 1e-12)
    uh = u_mat / nu[:, None]
    vh = v_mat / nv[:, None]
    cs = tl.sum(uh * vh, axis=1)
    dp = tl.load(DP_ptr + t0 + r_idx).to(tl.float32)
    dv_mat = (-0.5 * dp / nv)[:, None] * (uh - cs[:, None] * vh)
    du_mat = (-0.5 * dp / nu)[:, None] * (vh - cs[:, None] * uh)
    tl.store(DV_ptr + (t0 + r_idx[:, None]) * stride_t + offs[None, :], dv_mat.to(DV_ptr.dtype.element_ty))
    tl.store(DU_ptr + (t0 + r_idx[:, None] - 1) * stride_t + offs[None, :], du_mat.to(DU_ptr.dtype.element_ty))


@triton.jit
def _fused_hnet_cosine_router_fwd_kernel(
    U_ptr,
    V_ptr,
    IS_START_ptr,
    P_ptr,
    B_ptr,
    NORM_U_PREV_ptr,
    NORM_V_ptr,
    COS_ptr,
    RAW_IN_RANGE_ptr,
    stride_ut,
    stride_ud,
    stride_vt,
    stride_vd,
    D_DIM,
    EPS: tl.constexpr,
    BD: tl.constexpr,
):
    """Zero-spill tiled cosine router forward (`< 32` regs/thread for ANY `D`, `0` `[T, D]` HBM intermediates)."""
    t = tl.program_id(0)
    is_start = tl.load(IS_START_ptr + t).to(tl.int32)
    if is_start != 0 or t == 0:
        tl.store(P_ptr + t, 1.0)
        tl.store(B_ptr + t, 1)
        tl.store(NORM_U_PREV_ptr + t, 1.0)
        tl.store(NORM_V_ptr + t, 1.0)
        tl.store(COS_ptr + t, -1.0)
        tl.store(RAW_IN_RANGE_ptr + t, 0)
        return

    acc_uu = 0.0
    acc_vv = 0.0
    acc_uv = 0.0
    offs = tl.arange(0, BD)
    d0 = 0
    while d0 < D_DIM:
        cols = d0 + offs
        m = cols < D_DIM
        u = tl.load(U_ptr + (t - 1) * stride_ut + cols * stride_ud, mask=m, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + t * stride_vt + cols * stride_vd, mask=m, other=0.0).to(tl.float32)
        acc_uu += tl.sum(u * u, axis=0)
        acc_vv += tl.sum(v * v, axis=0)
        acc_uv += tl.sum(u * v, axis=0)
        d0 += BD

    norm_u = tl.sqrt(acc_uu)
    norm_v = tl.sqrt(acc_vv)
    denom_u = tl.maximum(norm_u, EPS)
    denom_v = tl.maximum(norm_v, EPS)
    cos_sim = (acc_uv / denom_u) / denom_v
    raw_p = (1.0 - cos_sim) * 0.5
    in_range = (raw_p >= 0.0) & (raw_p <= 1.0)
    p = tl.minimum(tl.maximum(raw_p, 0.0), 1.0)
    b = (p > 0.5).to(tl.int32)

    tl.store(P_ptr + t, p)
    tl.store(B_ptr + t, b)
    tl.store(NORM_U_PREV_ptr + t, norm_u)
    tl.store(NORM_V_ptr + t, norm_v)
    tl.store(COS_ptr + t, cos_sim)
    tl.store(RAW_IN_RANGE_ptr + t, in_range.to(tl.int8))


@triton.jit
def _fused_hnet_cosine_router_bwd_kernel(
    U_ptr,
    V_ptr,
    IS_START_ptr,
    IS_LAST_ptr,
    DP_ptr,
    NORM_U_PREV_ptr,
    NORM_V_ptr,
    COS_ptr,
    RAW_IN_RANGE_ptr,
    DU_ptr,
    DV_ptr,
    stride_ut,
    stride_ud,
    stride_vt,
    stride_vd,
    stride_dut,
    stride_dvt,
    T_LEN,
    D_DIM,
    EPS: tl.constexpr,
    OMIT_RADIAL_BUG: tl.constexpr,
    BD: tl.constexpr,
):
    """Lock-free 2D `(t, d_blk)` cosine router backward (`gridDim.x = T_LEN` up to `2^31 - 1`, `0` atomics, `0` spills)."""
    t = tl.program_id(0)
    d_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m = cols < D_DIM

    # 1. dV[t] comes from pair (t - 1, t) iff t > 0 and not a forced doc start
    is_start_t = tl.load(IS_START_ptr + t).to(tl.int32)
    dv = tl.zeros([BD], dtype=tl.float32)
    if (t > 0) and (is_start_t == 0):
        in_r_t = tl.load(RAW_IN_RANGE_ptr + t).to(tl.int32)
        if in_r_t != 0:
            dp_t = tl.load(DP_ptr + t).to(tl.float32)
            ds_t = -0.5 * dp_t
            nu_p = tl.load(NORM_U_PREV_ptr + t).to(tl.float32)
            nv_t = tl.load(NORM_V_ptr + t).to(tl.float32)
            du_den = tl.maximum(nu_p, EPS)
            dv_den = tl.maximum(nv_t, EPS)
            u_p = tl.load(U_ptr + (t - 1) * stride_ut + cols * stride_ud, mask=m, other=0.0).to(tl.float32)
            v_c = tl.load(V_ptr + t * stride_vt + cols * stride_vd, mask=m, other=0.0).to(tl.float32)
            u_hat = u_p / du_den
            v_hat = v_c / dv_den
            if OMIT_RADIAL_BUG:
                dv = (ds_t / dv_den) * u_hat
            else:
                if nv_t > EPS:
                    cos_t = tl.load(COS_ptr + t).to(tl.float32)
                    dv = (ds_t / nv_t) * (u_hat - cos_t * v_hat)
                else:
                    dv = (ds_t / EPS) * u_hat

    # 2. dU[t] comes from pair (t, t + 1) iff t + 1 < T_LEN and t is not doc_last and (t + 1) not doc_start
    is_last_t = tl.load(IS_LAST_ptr + t).to(tl.int32)
    du = tl.zeros([BD], dtype=tl.float32)
    if (t + 1 < T_LEN) and (is_last_t == 0):
        is_start_n = tl.load(IS_START_ptr + t + 1).to(tl.int32)
        in_r_n = tl.load(RAW_IN_RANGE_ptr + t + 1).to(tl.int32)
        if (is_start_n == 0) and (in_r_n != 0):
            dp_n = tl.load(DP_ptr + t + 1).to(tl.float32)
            ds_n = -0.5 * dp_n
            nu_t = tl.load(NORM_U_PREV_ptr + t + 1).to(tl.float32)
            nv_n = tl.load(NORM_V_ptr + t + 1).to(tl.float32)
            du_den_t = tl.maximum(nu_t, EPS)
            dv_den_n = tl.maximum(nv_n, EPS)
            u_c = tl.load(U_ptr + t * stride_ut + cols * stride_ud, mask=m, other=0.0).to(tl.float32)
            v_n = tl.load(V_ptr + (t + 1) * stride_vt + cols * stride_vd, mask=m, other=0.0).to(tl.float32)
            u_hat_c = u_c / du_den_t
            v_hat_n = v_n / dv_den_n
            if OMIT_RADIAL_BUG:
                du = (ds_n / du_den_t) * v_hat_n
            else:
                if nu_t > EPS:
                    cos_n = tl.load(COS_ptr + t + 1).to(tl.float32)
                    du = (ds_n / nu_t) * (v_hat_n - cos_n * u_hat_c)
                else:
                    du = (ds_n / EPS) * v_hat_n

    tl.store(DU_ptr + t * stride_dut + cols, du.to(DU_ptr.dtype.element_ty), mask=m)
    tl.store(DV_ptr + t * stride_dvt + cols, dv.to(DV_ptr.dtype.element_ty), mask=m)


class FusedHNetCosineRouterFn(torch.autograd.Function):
    """Exact zero-spill adjacent cosine-similarity router matching `goombalab/hnet` `RoutingModule`."""

    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        v: torch.Tensor,
        is_start: torch.Tensor,
        is_last: torch.Tensor,
        eps: float = 1e-12,
        omit_radial_bug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if u.dtype == torch.float64 or v.dtype == torch.float64:
            # FP64 path uses exact piecewise analytical expressions for Richardson verification
            t_len, d_dim = u.shape
            p_out = torch.empty((t_len,), dtype=torch.float64, device=u.device)
            b_out = torch.empty((t_len,), dtype=torch.int32, device=u.device)
            u64 = u.to(torch.float64)
            v64 = v.to(torch.float64)
            nu = torch.linalg.vector_norm(u64, ord=2, dim=-1)
            nv = torch.linalg.vector_norm(v64, ord=2, dim=-1)
            uh = u64 / nu.clamp_min(eps).unsqueeze(-1)
            vh = v64 / nv.clamp_min(eps).unsqueeze(-1)
            cos = torch.zeros((t_len,), dtype=torch.float64, device=u.device)
            raw_in = torch.zeros((t_len,), dtype=torch.int8, device=u.device)
            nu_prev = torch.ones((t_len,), dtype=torch.float64, device=u.device)
            if t_len > 1:
                c_tail = torch.sum(uh[:-1] * vh[1:], dim=-1)
                rp = (1.0 - c_tail) * 0.5
                cos[1:] = c_tail
                raw_in[1:] = ((rp >= 0.0) & (rp <= 1.0)).to(torch.int8)
                nu_prev[1:] = nu[:-1]
                p_out[1:] = torch.clamp(rp, 0.0, 1.0)
            if t_len > 0:
                p_out[0] = 1.0
                st_bool = is_start.to(torch.bool)
                p_out = torch.where(st_bool, torch.ones_like(p_out), p_out)
                raw_in = torch.where(st_bool, torch.zeros_like(raw_in), raw_in)
            b_out = (p_out > 0.5).to(torch.int32)
            ctx.save_for_backward(u64, v64, is_start, is_last, nu_prev, nv, cos, raw_in)
            ctx.eps = eps
            ctx.omit_radial_bug = omit_radial_bug
            ctx.is_fp64 = True
            return p_out, b_out

        t_len, d_dim = u.shape
        dev = u.device
        p_out = torch.empty((t_len,), dtype=torch.float32, device=dev)
        b_out = torch.empty((t_len,), dtype=torch.int32, device=dev)
        norm_u_prev = torch.empty((t_len,), dtype=torch.float32, device=dev)
        norm_v = torch.empty((t_len,), dtype=torch.float32, device=dev)
        cos_buf = torch.empty((t_len,), dtype=torch.float32, device=dev)
        raw_in_range = torch.empty((t_len,), dtype=torch.int8, device=dev)
        if t_len > 0:
            _fused_hnet_cosine_router_fwd_kernel[(t_len,)](
                u,
                v,
                is_start,
                p_out,
                b_out,
                norm_u_prev,
                norm_v,
                cos_buf,
                raw_in_range,
                u.stride(0),
                u.stride(1),
                v.stride(0),
                v.stride(1),
                d_dim,
                EPS=eps,
                BD=512,
                num_warps=4,
            )
        ctx.save_for_backward(u, v, is_start, is_last, norm_u_prev, norm_v, cos_buf, raw_in_range)
        ctx.eps = eps
        ctx.omit_radial_bug = omit_radial_bug
        ctx.is_fp64 = False
        return p_out, b_out

    @staticmethod
    def backward(ctx, dp: torch.Tensor, _db: Any):
        u, v, is_start, is_last, norm_u_prev, norm_v, cos_buf, raw_in_range = ctx.saved_tensors
        eps: float = ctx.eps
        omit_radial: bool = ctx.omit_radial_bug
        t_len, d_dim = u.shape
        if t_len == 0:
            return torch.zeros_like(u), torch.zeros_like(v), None, None, None, None

        if ctx.is_fp64:
            du = torch.zeros_like(u)
            dv = torch.zeros_like(v)
            if t_len > 1:
                active = (is_start[1:] == 0) & (raw_in_range[1:] != 0) & (is_last[:-1] == 0)
                ds = torch.where(active, -0.5 * dp[1:].to(torch.float64), torch.zeros_like(dp[1:], dtype=torch.float64))
                nu_p = norm_u_prev[1:]
                nv_n = norm_v[1:]
                uh = u[:-1] / nu_p.clamp_min(eps).unsqueeze(-1)
                vh = v[1:] / nv_n.clamp_min(eps).unsqueeze(-1)
                c = cos_buf[1:].unsqueeze(-1)
                if omit_radial:
                    dv[1:] = (ds / nv_n.clamp_min(eps)).unsqueeze(-1) * uh
                    du[:-1] = (ds / nu_p.clamp_min(eps)).unsqueeze(-1) * vh
                else:
                    dv_act = torch.where(
                        (nv_n > eps).unsqueeze(-1),
                        (ds / nv_n.clamp_min(eps)).unsqueeze(-1) * (uh - c * vh),
                        (ds / eps).unsqueeze(-1) * uh,
                    )
                    du_act = torch.where(
                        (nu_p > eps).unsqueeze(-1),
                        (ds / nu_p.clamp_min(eps)).unsqueeze(-1) * (vh - c * uh),
                        (ds / eps).unsqueeze(-1) * vh,
                    )
                    dv[1:] = dv_act
                    du[:-1] = du_act
            return du, dv, None, None, None, None

        du = torch.empty((t_len, d_dim), dtype=u.dtype, device=u.device)
        dv = torch.empty((t_len, d_dim), dtype=v.dtype, device=v.device)
        dp_c = dp.contiguous().to(torch.float32)
        bd = 256
        grid = (t_len, triton.cdiv(d_dim, bd))
        _fused_hnet_cosine_router_bwd_kernel[grid](
            u,
            v,
            is_start,
            is_last,
            dp_c,
            norm_u_prev,
            norm_v,
            cos_buf,
            raw_in_range,
            du,
            dv,
            u.stride(0),
            u.stride(1),
            v.stride(0),
            v.stride(1),
            du.stride(0),
            dv.stride(0),
            t_len,
            d_dim,
            EPS=eps,
            OMIT_RADIAL_BUG=omit_radial,
            BD=bd,
            num_warps=4,
        )
        return du, dv, None, None, None, None


@triton.jit
def _fused_hnet_merged_qk_router_1pass_fwd_kernel(
    QK_ptr,
    IS_START_ptr,
    P_ptr,
    B_ptr,
    NORM_U_PREV_ptr,
    NORM_V_ptr,
    COS_ptr,
    RAW_IN_RANGE_ptr,
    stride_qkt,
    D_DIM: tl.constexpr,
    EPS: tl.constexpr,
    BD: tl.constexpr,
):
    """Single-burst (`D <= 4096, num_warps=8`) merged-`QK` `[T, 2D]` cosine router forward (`0` slice views, `0` spills)."""
    t = tl.program_id(0)
    is_start = tl.load(IS_START_ptr + t).to(tl.int32)
    if is_start != 0 or t == 0:
        tl.store(P_ptr + t, 1.0)
        tl.store(B_ptr + t, 1)
        tl.store(NORM_U_PREV_ptr + t, 1.0)
        tl.store(NORM_V_ptr + t, 1.0)
        tl.store(COS_ptr + t, -1.0)
        tl.store(RAW_IN_RANGE_ptr + t, 0)
        return

    cols = tl.arange(0, BD)
    m = cols < D_DIM
    u = tl.load(QK_ptr + (t - 1) * stride_qkt + cols, mask=m, other=0.0).to(tl.float32)
    v = tl.load(QK_ptr + t * stride_qkt + D_DIM + cols, mask=m, other=0.0).to(tl.float32)
    acc_uu = tl.sum(u * u, axis=0)
    acc_vv = tl.sum(v * v, axis=0)
    acc_uv = tl.sum(u * v, axis=0)

    norm_u = tl.sqrt(acc_uu)
    norm_v = tl.sqrt(acc_vv)
    denom_u = tl.maximum(norm_u, EPS)
    denom_v = tl.maximum(norm_v, EPS)
    cos_sim = (acc_uv / denom_u) / denom_v
    raw_p = (1.0 - cos_sim) * 0.5
    in_range = (raw_p >= 0.0) & (raw_p <= 1.0)
    p = tl.minimum(tl.maximum(raw_p, 0.0), 1.0)
    b = (p > 0.5).to(tl.int32)

    tl.store(P_ptr + t, p)
    tl.store(B_ptr + t, b)
    tl.store(NORM_U_PREV_ptr + t, norm_u)
    tl.store(NORM_V_ptr + t, norm_v)
    tl.store(COS_ptr + t, cos_sim)
    tl.store(RAW_IN_RANGE_ptr + t, in_range.to(tl.int8))


@triton.jit
def _fused_hnet_merged_qk_router_bwd_kernel(
    QK_ptr,
    IS_START_ptr,
    IS_LAST_ptr,
    DP_ptr,
    NORM_U_PREV_ptr,
    NORM_V_ptr,
    COS_ptr,
    RAW_IN_RANGE_ptr,
    DQK_ptr,
    stride_qkt,
    stride_dqkt,
    T_LEN,
    D_DIM,
    EPS: tl.constexpr,
    OMIT_RADIAL_BUG: tl.constexpr,
    BD: tl.constexpr,
):
    """Writes `dU` and `dV` directly into unified `[T, 2D]` `DQK` (`gridDim.x = T_LEN` up to `2^31 - 1`, `0` copies!)."""
    t = tl.program_id(0)
    d_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m = cols < D_DIM

    is_start_t = tl.load(IS_START_ptr + t).to(tl.int32)
    dv = tl.zeros([BD], dtype=tl.float32)
    if (t > 0) and (is_start_t == 0):
        in_r_t = tl.load(RAW_IN_RANGE_ptr + t).to(tl.int32)
        if in_r_t != 0:
            dp_t = tl.load(DP_ptr + t).to(tl.float32)
            ds_t = -0.5 * dp_t
            nu_p = tl.load(NORM_U_PREV_ptr + t).to(tl.float32)
            nv_t = tl.load(NORM_V_ptr + t).to(tl.float32)
            du_den = tl.maximum(nu_p, EPS)
            dv_den = tl.maximum(nv_t, EPS)
            u_p = tl.load(QK_ptr + (t - 1) * stride_qkt + cols, mask=m, other=0.0).to(tl.float32)
            v_c = tl.load(QK_ptr + t * stride_qkt + D_DIM + cols, mask=m, other=0.0).to(tl.float32)
            u_hat = u_p / du_den
            v_hat = v_c / dv_den
            if OMIT_RADIAL_BUG:
                dv = (ds_t / dv_den) * u_hat
            else:
                if nv_t > EPS:
                    cos_t = tl.load(COS_ptr + t).to(tl.float32)
                    dv = (ds_t / nv_t) * (u_hat - cos_t * v_hat)
                else:
                    dv = (ds_t / EPS) * u_hat

    is_last_t = tl.load(IS_LAST_ptr + t).to(tl.int32)
    du = tl.zeros([BD], dtype=tl.float32)
    if (t + 1 < T_LEN) and (is_last_t == 0):
        is_start_n = tl.load(IS_START_ptr + t + 1).to(tl.int32)
        in_r_n = tl.load(RAW_IN_RANGE_ptr + t + 1).to(tl.int32)
        if (is_start_n == 0) and (in_r_n != 0):
            dp_n = tl.load(DP_ptr + t + 1).to(tl.float32)
            ds_n = -0.5 * dp_n
            nu_t = tl.load(NORM_U_PREV_ptr + t + 1).to(tl.float32)
            nv_n = tl.load(NORM_V_ptr + t + 1).to(tl.float32)
            du_den_t = tl.maximum(nu_t, EPS)
            dv_den_n = tl.maximum(nv_n, EPS)
            u_c = tl.load(QK_ptr + t * stride_qkt + cols, mask=m, other=0.0).to(tl.float32)
            v_n = tl.load(QK_ptr + (t + 1) * stride_qkt + D_DIM + cols, mask=m, other=0.0).to(tl.float32)
            u_hat_c = u_c / du_den_t
            v_hat_n = v_n / dv_den_n
            if OMIT_RADIAL_BUG:
                du = (ds_n / du_den_t) * v_hat_n
            else:
                if nu_t > EPS:
                    cos_n = tl.load(COS_ptr + t + 1).to(tl.float32)
                    du = (ds_n / nu_t) * (v_hat_n - cos_n * u_hat_c)
                else:
                    du = (ds_n / EPS) * v_hat_n

    tl.store(DQK_ptr + t * stride_dqkt + cols, du.to(DQK_ptr.dtype.element_ty), mask=m)
    tl.store(DQK_ptr + t * stride_dqkt + D_DIM + cols, dv.to(DQK_ptr.dtype.element_ty), mask=m)


class FusedHNetMergedQKCosineRouterFn(torch.autograd.Function):
    """Zero-slice-copy merged-`QK` `[T, 2D]` cosine router eliminating `SliceBackward0` HBM copies."""

    @staticmethod
    def forward(
        ctx,
        qk: torch.Tensor,
        is_start: torch.Tensor,
        is_last: torch.Tensor,
        eps: float = 1e-12,
        omit_radial_bug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_len, two_d = qk.shape
        d_dim = two_d // 2
        if qk.dtype == torch.float64:
            p, b = FusedHNetCosineRouterFn.apply(qk[:, :d_dim], qk[:, d_dim:], is_start, is_last, eps, omit_radial_bug)
            ctx.save_for_backward(qk, is_start, is_last)
            ctx.eps = eps
            ctx.omit_radial_bug = omit_radial_bug
            ctx.is_fp64 = True
            return p, b

        qk_c = qk.contiguous()
        dev = qk_c.device
        p_out = torch.empty((t_len,), dtype=torch.float32, device=dev)
        b_out = torch.empty((t_len,), dtype=torch.int32, device=dev)
        norm_u_prev = torch.empty((t_len,), dtype=torch.float32, device=dev)
        norm_v = torch.empty((t_len,), dtype=torch.float32, device=dev)
        cos_buf = torch.empty((t_len,), dtype=torch.float32, device=dev)
        raw_in_range = torch.empty((t_len,), dtype=torch.int8, device=dev)
        if t_len > 0:
            if d_dim <= 4096:
                bd = max(64, triton.next_power_of_2(d_dim))
                nw = 8 if bd >= 1024 else 4
                _fused_hnet_merged_qk_router_1pass_fwd_kernel[(t_len,)](
                    qk_c,
                    is_start,
                    p_out,
                    b_out,
                    norm_u_prev,
                    norm_v,
                    cos_buf,
                    raw_in_range,
                    qk_c.stride(0),
                    D_DIM=d_dim,
                    EPS=eps,
                    BD=bd,
                    num_warps=nw,
                )
            else:
                u = qk_c[:, :d_dim]
                v = qk_c[:, d_dim:]
                _fused_hnet_cosine_router_fwd_kernel[(t_len,)](
                    u,
                    v,
                    is_start,
                    p_out,
                    b_out,
                    norm_u_prev,
                    norm_v,
                    cos_buf,
                    raw_in_range,
                    u.stride(0),
                    u.stride(1),
                    v.stride(0),
                    v.stride(1),
                    d_dim,
                    EPS=eps,
                    BD=512,
                    num_warps=4,
                )
        ctx.save_for_backward(qk_c, is_start, is_last, norm_u_prev, norm_v, cos_buf, raw_in_range)
        ctx.eps = eps
        ctx.omit_radial_bug = omit_radial_bug
        ctx.is_fp64 = False
        return p_out, b_out

    @staticmethod
    def backward(ctx, dp: torch.Tensor, _db: Any):
        if ctx.is_fp64:
            qk, is_start, is_last = ctx.saved_tensors
            d_dim = qk.shape[1] // 2
            with torch.enable_grad():
                u = qk[:, :d_dim].detach().requires_grad_(True)
                v = qk[:, d_dim:].detach().requires_grad_(True)
                p, _ = FusedHNetCosineRouterFn.apply(u, v, is_start, is_last, ctx.eps, ctx.omit_radial_bug)
                du, dv = torch.autograd.grad(p, (u, v), dp)
            return torch.cat([du, dv], dim=-1), None, None, None, None

        qk_c, is_start, is_last, norm_u_prev, norm_v, cos_buf, raw_in_range = ctx.saved_tensors
        t_len, two_d = qk_c.shape
        d_dim = two_d // 2
        dqk = torch.empty_like(qk_c)
        if t_len > 0:
            dp_c = dp.contiguous().to(torch.float32)
            bd = 256
            _fused_hnet_merged_qk_router_bwd_kernel[(t_len, triton.cdiv(d_dim, bd))](
                qk_c,
                is_start,
                is_last,
                dp_c,
                norm_u_prev,
                norm_v,
                cos_buf,
                raw_in_range,
                dqk,
                qk_c.stride(0),
                dqk.stride(0),
                t_len,
                d_dim,
                EPS=ctx.eps,
                OMIT_RADIAL_BUG=ctx.omit_radial_bug,
                BD=bd,
                num_warps=4,
            )
        return dqk, None, None, None, None


# ==============================================================================
# 4. Zero-Sort Prefix-Sum Ragged Chunk Compaction + Span Metadata (`O(T)`)
# ==============================================================================


@triton.jit
def _zero_sort_prefix_scatter_meta_kernel(
    B_ptr,
    PREFIX_ptr,
    P_ptr,
    DOC_ID_ptr,
    SRC_POS_ptr,
    P_SEL_ptr,
    CHUNK_DOC_ptr,
    T_LEN,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    m = offs < T_LEN
    b = tl.load(B_ptr + offs, mask=m, other=0).to(tl.int32)
    sel = m & (b != 0)
    j = tl.load(PREFIX_ptr + offs, mask=sel, other=1).to(tl.int32) - 1
    p = tl.load(P_ptr + offs, mask=sel, other=0.0)
    doc = tl.load(DOC_ID_ptr + offs, mask=sel, other=0).to(tl.int32)
    tl.store(SRC_POS_ptr + j, offs, mask=sel)
    tl.store(P_SEL_ptr + j, p, mask=sel)
    tl.store(CHUNK_DOC_ptr + j, doc, mask=sel)


@triton.jit
def _build_chunk_span_end_and_doc_start_kernel(
    SRC_POS_ptr,
    CHUNK_DOC_ptr,
    SPAN_END_ptr,
    CHUNK_IS_START_ptr,
    M_LEN,
    T_LEN,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m = offs < M_LEN
    has_next = offs + 1 < M_LEN
    next_pos = tl.load(SRC_POS_ptr + offs + 1, mask=m & has_next, other=T_LEN).to(tl.int32)
    span_end = tl.where(has_next, next_pos, T_LEN)
    tl.store(SPAN_END_ptr + offs, span_end, mask=m)

    cur_doc = tl.load(CHUNK_DOC_ptr + offs, mask=m, other=0).to(tl.int32)
    prev_doc = tl.load(CHUNK_DOC_ptr + offs - 1, mask=m & (offs > 0), other=-1).to(tl.int32)
    is_st = (offs == 0) | (cur_doc != prev_doc)
    tl.store(CHUNK_IS_START_ptr + offs, is_st.to(tl.int8), mask=m)


@triton.jit
def _gather_selected_chunks_fwd_kernel(
    X_ptr,
    SRC_POS_ptr,
    X_PACK_ptr,
    stride_xt,
    stride_xd,
    stride_pt,
    D_DIM,
    BD: tl.constexpr,
):
    j = tl.program_id(0)
    d_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m = cols < D_DIM
    src_t = tl.load(SRC_POS_ptr + j).to(tl.int64)
    val = tl.load(X_ptr + src_t * stride_xt + cols * stride_xd, mask=m, other=0.0)
    tl.store(X_PACK_ptr + j * stride_pt + cols, val, mask=m)


@triton.jit
def _scatter_selected_chunks_bwd_kernel(
    DX_PACK_ptr,
    SRC_POS_ptr,
    DX_OUT_ptr,
    stride_pt,
    stride_ot,
    D_DIM,
    BD: tl.constexpr,
):
    j = tl.program_id(0)
    d_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m = cols < D_DIM
    src_t = tl.load(SRC_POS_ptr + j).to(tl.int64)
    val = tl.load(DX_PACK_ptr + j * stride_pt + cols, mask=m, other=0.0)
    tl.store(DX_OUT_ptr + src_t * stride_ot + cols, val, mask=m)


@dataclass
class PackedChunkMetadata:
    chunk_idx: torch.Tensor
    source_pos: torch.Tensor
    span_end: torch.Tensor
    chunk_doc_id: torch.Tensor
    chunk_is_start: torch.Tensor
    next_cu_seqlens: torch.Tensor
    m_total: int


class _GatherSelectedChunksFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_enc: torch.Tensor, source_pos: torch.Tensor) -> torch.Tensor:
        t_len, d_dim = x_enc.shape
        m_len = int(source_pos.numel())
        ctx.t_len = t_len
        ctx.save_for_backward(source_pos)
        if x_enc.dtype == torch.float64:
            ctx.is_fp64 = True
            return x_enc.index_select(0, source_pos.to(torch.long))
        ctx.is_fp64 = False
        out = torch.empty((m_len, d_dim), dtype=x_enc.dtype, device=x_enc.device)
        if m_len > 0:
            bd = 256
            _gather_selected_chunks_fwd_kernel[(m_len, triton.cdiv(d_dim, bd))](
                x_enc,
                source_pos,
                out,
                x_enc.stride(0),
                x_enc.stride(1),
                out.stride(0),
                d_dim,
                BD=bd,
                num_warps=4,
            )
        return out

    @staticmethod
    def backward(ctx, dx_pack: torch.Tensor):
        (source_pos,) = ctx.saved_tensors
        t_len = ctx.t_len
        m_len, d_dim = dx_pack.shape
        dx = torch.zeros((t_len, d_dim), dtype=dx_pack.dtype, device=dx_pack.device)
        if m_len == 0:
            return dx, None
        if ctx.is_fp64:
            dx.index_copy_(0, source_pos.to(torch.long), dx_pack)
            return dx, None
        dx_c = dx_pack.contiguous()
        bd = 256
        _scatter_selected_chunks_bwd_kernel[(m_len, triton.cdiv(d_dim, bd))](
            dx_c,
            source_pos,
            dx,
            dx_c.stride(0),
            dx.stride(0),
            d_dim,
            BD=bd,
            num_warps=4,
        )
        return dx, None


class _GatherSelectedProbsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p: torch.Tensor, source_pos: torch.Tensor) -> torch.Tensor:
        ctx.t_len = p.numel()
        ctx.save_for_backward(source_pos)
        return p.index_select(0, source_pos.to(torch.long))

    @staticmethod
    def backward(ctx, dp_sel: torch.Tensor):
        (source_pos,) = ctx.saved_tensors
        dp = torch.zeros((ctx.t_len,), dtype=dp_sel.dtype, device=dp_sel.device)
        if source_pos.numel() > 0:
            dp.index_copy_(0, source_pos.to(torch.long), dp_sel)
        return dp, None


def zero_sort_prefix_chunk_pack(
    x_enc: torch.Tensor,
    p: torch.Tensor,
    b_int: torch.Tensor,
    doc_id: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, PackedChunkMetadata]:
    """O(T) prefix-sum compaction & span table without `torch.argsort`; safe on empty documents."""
    dev = x_enc.device
    t_len = int(x_enc.shape[0])
    pref_pad = torch.zeros((t_len + 1,), dtype=torch.int32, device=dev)
    if t_len > 0:
        torch.cumsum(b_int.to(torch.int32), dim=0, dtype=torch.int32, out=pref_pad[1:])
    next_cu = pref_pad.index_select(0, cu_seqlens.to(torch.long))
    m_total = int(next_cu[-1].item()) if next_cu.numel() > 0 else 0
    chunk_idx = pref_pad[1:] - 1

    source_pos = torch.empty((m_total,), dtype=torch.int32, device=dev)
    p_sel_meta = torch.empty((m_total,), dtype=p.dtype, device=dev)
    chunk_doc_id = torch.empty((m_total,), dtype=torch.int32, device=dev)
    span_end = torch.empty((m_total,), dtype=torch.int32, device=dev)
    chunk_is_start = torch.empty((m_total,), dtype=torch.int8, device=dev)

    if t_len > 0 and m_total > 0:
        _zero_sort_prefix_scatter_meta_kernel[(triton.cdiv(t_len, 256),)](
            b_int,
            pref_pad[1:],
            p,
            doc_id,
            source_pos,
            p_sel_meta,
            chunk_doc_id,
            t_len,
            BLOCK_T=256,
            num_warps=4,
        )
        _build_chunk_span_end_and_doc_start_kernel[(triton.cdiv(m_total, 256),)](
            source_pos,
            chunk_doc_id,
            span_end,
            chunk_is_start,
            m_total,
            t_len,
            BLOCK_M=256,
            num_warps=4,
        )

    x_packed = _GatherSelectedChunksFn.apply(x_enc, source_pos)
    p_sel = _GatherSelectedProbsFn.apply(p, source_pos)
    meta = PackedChunkMetadata(
        chunk_idx=chunk_idx,
        source_pos=source_pos,
        span_end=span_end,
        chunk_doc_id=chunk_doc_id,
        chunk_is_start=chunk_is_start,
        next_cu_seqlens=next_cu,
        m_total=m_total,
    )
    return x_packed, p_sel, meta


# ==============================================================================
# 5. Two-Level Parallel Associative Scalar-Affine EMA + Lock-Free Upsample/STE
# ==============================================================================


@triton.jit
def _ema_fwd_block_summary_kernel(
    Z_HAT_ptr,
    P_SEL_ptr,
    IS_START_ptr,
    A_BLK_ptr,
    U_BLK_ptr,
    stride_zm,
    stride_zd,
    stride_um,
    M_LEN,
    D_DIM,
    BC: tl.constexpr,
    BD: tl.constexpr,
):
    d_blk = tl.program_id(0)
    c_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m_d = cols < D_DIM

    j_base = c_blk * BC
    a_acc = 1.0
    u_acc = tl.zeros([BD], dtype=tl.float32)

    for step in range(BC):
        j = j_base + step
        if j < M_LEN:
            is_st = tl.load(IS_START_ptr + j).to(tl.int32)
            pj = tl.load(P_SEL_ptr + j).to(tl.float32)
            aj = tl.where(is_st != 0, 0.0, 1.0 - pj)
            zj = tl.load(Z_HAT_ptr + j * stride_zm + cols * stride_zd, mask=m_d, other=0.0).to(tl.float32)
            a_acc = aj * a_acc
            u_acc = aj * u_acc + pj * zj

    if d_blk == 0:
        tl.store(A_BLK_ptr + c_blk, a_acc)
    tl.store(U_BLK_ptr + c_blk * stride_um + cols, u_acc, mask=m_d)


@triton.jit
def _ema_fwd_inter_and_intra_scan_kernel(
    Z_HAT_ptr,
    P_SEL_ptr,
    IS_START_ptr,
    A_BLK_ptr,
    U_BLK_ptr,
    Z_BAR_FP32_ptr,
    Z_BAR_OUT_ptr,
    stride_zm,
    stride_zd,
    stride_um,
    stride_bm,
    stride_om,
    M_LEN,
    D_DIM,
    BC: tl.constexpr,
    BD: tl.constexpr,
):
    d_blk = tl.program_id(0)
    c_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m_d = cols < D_DIM

    state = tl.zeros([BD], dtype=tl.float32)
    b_idx = 0
    while b_idx < c_blk:
        ab = tl.load(A_BLK_ptr + b_idx).to(tl.float32)
        ub = tl.load(U_BLK_ptr + b_idx * stride_um + cols, mask=m_d, other=0.0).to(tl.float32)
        state = ab * state + ub
        b_idx += 1

    j_base = c_blk * BC
    for step in range(BC):
        j = j_base + step
        if j < M_LEN:
            is_st = tl.load(IS_START_ptr + j).to(tl.int32)
            pj = tl.load(P_SEL_ptr + j).to(tl.float32)
            aj = tl.where(is_st != 0, 0.0, 1.0 - pj)
            zj = tl.load(Z_HAT_ptr + j * stride_zm + cols * stride_zd, mask=m_d, other=0.0).to(tl.float32)
            state = aj * state + pj * zj
            tl.store(Z_BAR_FP32_ptr + j * stride_bm + cols, state, mask=m_d)
            tl.store(Z_BAR_OUT_ptr + j * stride_om + cols, state.to(Z_BAR_OUT_ptr.dtype.element_ty), mask=m_d)


@triton.jit
def _ema_bwd_block_summary_kernel(
    DZ_BAR_ptr,
    P_SEL_ptr,
    IS_START_ptr,
    A_BWD_BLK_ptr,
    L_BWD_BLK_ptr,
    stride_dzm,
    stride_lm,
    M_LEN,
    D_DIM,
    BC: tl.constexpr,
    BD: tl.constexpr,
):
    """Reverse associative block summary for `lam_j = dz_bar_j + a_{j+1} * lam_{j+1}`."""
    d_blk = tl.program_id(0)
    c_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m_d = cols < D_DIM

    j_base = c_blk * BC
    a_acc = 1.0
    l_acc = tl.zeros([BD], dtype=tl.float32)

    for r_step in range(BC):
        j = j_base + (BC - 1 - r_step)
        if j < M_LEN:
            if j + 1 < M_LEN:
                is_st_n = tl.load(IS_START_ptr + j + 1).to(tl.int32)
                pn = tl.load(P_SEL_ptr + j + 1).to(tl.float32)
                a_next = tl.where(is_st_n != 0, 0.0, 1.0 - pn)
            else:
                a_next = 0.0
            dz = tl.load(DZ_BAR_ptr + j * stride_dzm + cols, mask=m_d, other=0.0).to(tl.float32)
            a_acc = a_next * a_acc
            l_acc = dz + a_next * l_acc

    if d_blk == 0:
        tl.store(A_BWD_BLK_ptr + c_blk, a_acc)
    tl.store(L_BWD_BLK_ptr + c_blk * stride_lm + cols, l_acc, mask=m_d)


@triton.jit
def _ema_bwd_inter_and_intra_scan_kernel(
    DZ_BAR_ptr,
    Z_HAT_ptr,
    Z_BAR_FP32_ptr,
    P_SEL_ptr,
    IS_START_ptr,
    A_BWD_BLK_ptr,
    L_BWD_BLK_ptr,
    DP_PART_OUT_ptr,
    DZ_HAT_OUT_ptr,
    stride_dzm,
    stride_zm,
    stride_zd,
    stride_bm,
    stride_lm,
    stride_dppm,
    stride_dzhm,
    M_LEN,
    N_BLKS,
    D_DIM,
    OMIT_EMA_ADJOINT_BUG: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
):
    """Fused reverse EMA scan + in-register partial `dP_j = <lam_j, z_hat_j - z_bar_{j-1}>` (`0` `lam_buf[M, D]` HBM!)."""
    d_blk = tl.program_id(0)
    c_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m_d = cols < D_DIM

    lam = tl.zeros([BD], dtype=tl.float32)
    if not OMIT_EMA_ADJOINT_BUG:
        b_idx = N_BLKS - 1
        while b_idx > c_blk:
            ab = tl.load(A_BWD_BLK_ptr + b_idx).to(tl.float32)
            lb = tl.load(L_BWD_BLK_ptr + b_idx * stride_lm + cols, mask=m_d, other=0.0).to(tl.float32)
            lam = lb + ab * lam
            b_idx -= 1

    j_base = c_blk * BC
    for r_step in range(BC):
        j = j_base + (BC - 1 - r_step)
        if j < M_LEN:
            if (not OMIT_EMA_ADJOINT_BUG) and (j + 1 < M_LEN):
                is_st_n = tl.load(IS_START_ptr + j + 1).to(tl.int32)
                pn = tl.load(P_SEL_ptr + j + 1).to(tl.float32)
                a_next = tl.where(is_st_n != 0, 0.0, 1.0 - pn)
            else:
                a_next = 0.0
            dz = tl.load(DZ_BAR_ptr + j * stride_dzm + cols, mask=m_d, other=0.0).to(tl.float32)
            lam = dz + a_next * lam
            pj = tl.load(P_SEL_ptr + j).to(tl.float32)
            dz_hat = pj * lam
            tl.store(DZ_HAT_OUT_ptr + j * stride_dzhm + cols, dz_hat.to(DZ_HAT_OUT_ptr.dtype.element_ty), mask=m_d)

            is_st_c = tl.load(IS_START_ptr + j).to(tl.int32)
            has_prev = (j > 0) & (is_st_c == 0)
            zh = tl.load(Z_HAT_ptr + j * stride_zm + cols * stride_zd, mask=m_d, other=0.0).to(tl.float32)
            zb_prev = tl.load(Z_BAR_FP32_ptr + (j - 1) * stride_bm + cols, mask=m_d & has_prev, other=0.0).to(tl.float32)
            dp_part = tl.sum(lam * (zh - zb_prev), axis=0)
            tl.store(DP_PART_OUT_ptr + d_blk * stride_dppm + j, dp_part)


@triton.jit
def _fused_upsample_ste_residual_fwd_kernel(
    Z_BAR_RND_ptr,
    CHUNK_IDX_ptr,
    RES_FP32_ptr,
    Z_OUT_ptr,
    stride_zbm,
    stride_rt,
    stride_rd,
    stride_ot,
    D_DIM,
    BD: tl.constexpr,
):
    """Superfused Upsample + `STE(c_t) == 1` + `FP32` Residual add (`gridDim.x = T_LEN` up to `2^31 - 1`)."""
    t = tl.program_id(0)
    d_blk = tl.program_id(1)
    cols = d_blk * BD + tl.arange(0, BD)
    m = cols < D_DIM
    j = tl.load(CHUNK_IDX_ptr + t).to(tl.int64)
    zb = tl.load(Z_BAR_RND_ptr + j * stride_zbm + cols, mask=m, other=0.0).to(tl.float32)
    r = tl.load(RES_FP32_ptr + t * stride_rt + cols * stride_rd, mask=m, other=0.0).to(tl.float32)
    out = zb + r
    tl.store(Z_OUT_ptr + t * stride_ot + cols, out.to(Z_OUT_ptr.dtype.element_ty), mask=m)


@triton.jit
def _upsample_ste_dot_bwd_kernel(
    GRAD_Z_ptr,
    Z_BAR_RND_ptr,
    CHUNK_IDX_ptr,
    B_ptr,
    DP_OUT_ptr,
    stride_gt,
    stride_gd,
    stride_zbm,
    D_DIM,
    BD: tl.constexpr,
):
    """Per-byte `STE` confidence gradient `dp_t = (2*b_t - 1) * <g_t, z_bar[chunk_idx[t]]>` (`0` spills)."""
    t = tl.program_id(0)
    j = tl.load(CHUNK_IDX_ptr + t).to(tl.int64)
    b = tl.load(B_ptr + t).to(tl.int32)
    sign = tl.where(b != 0, 1.0, -1.0)

    acc = 0.0
    offs = tl.arange(0, BD)
    d0 = 0
    while d0 < D_DIM:
        cols = d0 + offs
        m = cols < D_DIM
        g = tl.load(GRAD_Z_ptr + t * stride_gt + cols * stride_gd, mask=m, other=0.0).to(tl.float32)
        zb = tl.load(Z_BAR_RND_ptr + j * stride_zbm + cols, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(g * zb, axis=0)
        d0 += BD

    tl.store(DP_OUT_ptr + t, sign * acc)


@triton.jit
def _upsample_span_reduce_split_bwd_kernel(
    GRAD_Z_ptr,
    SRC_POS_ptr,
    SPAN_END_ptr,
    DZ_SPLIT_OUT_ptr,
    stride_gt,
    stride_gd,
    stride_sm,
    stride_sk,
    D_DIM,
    NUM_SPLITS: tl.constexpr,
    BD: tl.constexpr,
):
    """Lock-free adaptive Split-Span reduction `sum_{t in [s_j, e_j)} g_t` (`gridDim.x = M` up to `2^31 - 1`)."""
    j = tl.program_id(0)
    d_blk = tl.program_id(1)
    k_split = tl.program_id(2)
    cols = d_blk * BD + tl.arange(0, BD)
    m = cols < D_DIM

    s_j = tl.load(SRC_POS_ptr + j).to(tl.int32)
    e_j = tl.load(SPAN_END_ptr + j).to(tl.int32)
    span_len = e_j - s_j
    part = tl.cdiv(span_len, NUM_SPLITS)
    t_start = s_j + k_split * part
    t_end = tl.minimum(e_j, t_start + part)

    acc = tl.zeros([BD], dtype=tl.float32)
    t = t_start
    while t < t_end:
        acc += tl.load(GRAD_Z_ptr + t * stride_gt + cols * stride_gd, mask=m, other=0.0).to(tl.float32)
        t += 1

    tl.store(DZ_SPLIT_OUT_ptr + j * stride_sm + k_split * stride_sk + cols, acc, mask=m)


@triton.jit
def _add_bijective_dp_sel_kernel(
    DP_SEL_ptr,
    SRC_POS_ptr,
    DP_TOTAL_ptr,
    M_LEN,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m = offs < M_LEN
    src_t = tl.load(SRC_POS_ptr + offs, mask=m, other=0).to(tl.int64)
    dps = tl.load(DP_SEL_ptr + offs, mask=m, other=0.0).to(tl.float32)
    cur = tl.load(DP_TOTAL_ptr + src_t, mask=m, other=0.0).to(tl.float32)
    tl.store(DP_TOTAL_ptr + src_t, cur + dps, mask=m)


def _run_parallel_ema_fwd(
    z_hat: torch.Tensor,
    p_sel: torch.Tensor,
    chunk_is_start: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(z_bar_fp32[M, D], z_bar_rounded[M, D])` via 2-level parallel scalar-affine EMA."""
    m_len, d_dim = z_hat.shape
    dev = z_hat.device
    z_bar_fp32 = torch.empty((m_len, d_dim), dtype=torch.float32, device=dev)
    z_bar_rnd = torch.empty((m_len, d_dim), dtype=z_hat.dtype, device=dev)
    if m_len == 0:
        return z_bar_fp32, z_bar_rnd

    bc = 64
    bd = 128
    n_blks = triton.cdiv(m_len, bc)
    n_dblks = triton.cdiv(d_dim, bd)
    a_blk = torch.empty((n_blks,), dtype=torch.float32, device=dev)
    u_blk = torch.empty((n_blks, d_dim), dtype=torch.float32, device=dev)

    _ema_fwd_block_summary_kernel[(n_dblks, n_blks)](
        z_hat,
        p_sel,
        chunk_is_start,
        a_blk,
        u_blk,
        z_hat.stride(0),
        z_hat.stride(1),
        u_blk.stride(0),
        m_len,
        d_dim,
        BC=bc,
        BD=bd,
        num_warps=4,
    )
    _ema_fwd_inter_and_intra_scan_kernel[(n_dblks, n_blks)](
        z_hat,
        p_sel,
        chunk_is_start,
        a_blk,
        u_blk,
        z_bar_fp32,
        z_bar_rnd,
        z_hat.stride(0),
        z_hat.stride(1),
        u_blk.stride(0),
        z_bar_fp32.stride(0),
        z_bar_rnd.stride(0),
        m_len,
        d_dim,
        BC=bc,
        BD=bd,
        num_warps=4,
    )
    return z_bar_fp32, z_bar_rnd


def _run_parallel_ema_bwd(
    dz_bar: torch.Tensor,
    z_hat: torch.Tensor,
    p_sel: torch.Tensor,
    z_bar_fp32: torch.Tensor,
    chunk_is_start: torch.Tensor,
    *,
    omit_ema_adjoint_bug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(dz_hat[M, D], dp_sel[M])` via 2-level reverse parallel scalar-affine EMA scan (`0` `lam_buf` HBM)."""
    m_len, d_dim = z_hat.shape
    dev = z_hat.device
    dz_hat = torch.empty((m_len, d_dim), dtype=z_hat.dtype, device=dev)
    if m_len == 0:
        return dz_hat, torch.empty((0,), dtype=torch.float32, device=dev)

    bc = 64
    bd = 128
    n_blks = triton.cdiv(m_len, bc)
    n_dblks = triton.cdiv(d_dim, bd)
    a_bwd_blk = torch.empty((n_blks,), dtype=torch.float32, device=dev)
    l_bwd_blk = torch.empty((n_blks, d_dim), dtype=torch.float32, device=dev)
    dp_part = torch.empty((n_dblks, m_len), dtype=torch.float32, device=dev)

    _ema_bwd_block_summary_kernel[(n_dblks, n_blks)](
        dz_bar,
        p_sel,
        chunk_is_start,
        a_bwd_blk,
        l_bwd_blk,
        dz_bar.stride(0),
        l_bwd_blk.stride(0),
        m_len,
        d_dim,
        BC=bc,
        BD=bd,
        num_warps=4,
    )
    _ema_bwd_inter_and_intra_scan_kernel[(n_dblks, n_blks)](
        dz_bar,
        z_hat,
        z_bar_fp32,
        p_sel,
        chunk_is_start,
        a_bwd_blk,
        l_bwd_blk,
        dp_part,
        dz_hat,
        dz_bar.stride(0),
        z_hat.stride(0),
        z_hat.stride(1),
        z_bar_fp32.stride(0),
        l_bwd_blk.stride(0),
        dp_part.stride(0),
        dz_hat.stride(0),
        m_len,
        n_blks,
        d_dim,
        OMIT_EMA_ADJOINT_BUG=omit_ema_adjoint_bug,
        BC=bc,
        BD=bd,
        num_warps=4,
    )
    dp_sel = dp_part.sum(dim=0)
    return dz_hat, dp_sel


class ParallelScalarAffineEMAFn(torch.autograd.Function):
    """Standalone autograd wrapper for `ParallelScalarAffineEMA` (supports FP64 Richardson & BF16/FP32 Triton)."""

    @staticmethod
    def forward(ctx, z_hat: torch.Tensor, p_sel: torch.Tensor, chunk_is_start: torch.Tensor, next_cu_seqlens: torch.Tensor) -> torch.Tensor:
        if z_hat.dtype == torch.float64 or p_sel.dtype == torch.float64:
            z64 = z_hat.to(torch.float64)
            p64 = p_sel.to(torch.float64)
            z_bar = fp64_hnet_affine_ema_oracle(z64, p64, next_cu_seqlens)
            ctx.save_for_backward(z64, p64, z_bar, chunk_is_start)
            ctx.is_fp64 = True
            return z_bar
        ctx.is_fp64 = False
        p_f32 = p_sel.contiguous().to(torch.float32)
        z_bar_fp32, z_bar_rnd = _run_parallel_ema_fwd(z_hat, p_f32, chunk_is_start)
        ctx.save_for_backward(z_hat, p_f32, z_bar_fp32, chunk_is_start)
        return z_bar_rnd

    @staticmethod
    def backward(ctx, dz_bar_in: torch.Tensor):
        if ctx.is_fp64:
            z64, p64, z_bar, chunk_is_start = ctx.saved_tensors
            m_len, d_dim = z64.shape
            dz_hat = torch.zeros_like(z64)
            dp_sel = torch.zeros_like(p64)
            lam = torch.zeros((d_dim,), dtype=torch.float64, device=z64.device)
            st_list = chunk_is_start.detach().cpu().tolist()
            for j in range(m_len - 1, -1, -1):
                a_next = 0.0 if (j + 1 >= m_len or st_list[j + 1] != 0) else float((1.0 - p64[j + 1]).item())
                lam = dz_bar_in[j].to(torch.float64) + a_next * lam
                dz_hat[j] = p64[j] * lam
                prev_zb = torch.zeros_like(lam) if (j == 0 or st_list[j] != 0) else z_bar[j - 1]
                dp_sel[j] = torch.sum(lam * (z64[j] - prev_zb))
            return dz_hat, dp_sel, None, None

        z_hat, p_f32, z_bar_fp32, chunk_is_start = ctx.saved_tensors
        dz_bar_f32 = dz_bar_in.contiguous().to(torch.float32)
        dz_hat, dp_sel = _run_parallel_ema_bwd(dz_bar_f32, z_hat, p_f32, z_bar_fp32, chunk_is_start)
        return dz_hat, dp_sel.to(p_f32.dtype), None, None


class FusedDechunkEMASmoothUpsampleSTEResidualFn(torch.autograd.Function):
    """Superfused Round-7 RSI Dechunking (`Parallel Affine EMA + Upsample + STE(c_t) + FP32 Skip Residual`).

    Eliminates:
    - `goombalab/hnet` `BF16` surrogate `dt = -log(1 - p)` & `x = z / dt` drift and `mamba_ssm` dependency,
    - `O(B L log L)` `torch.argsort` in `DeChunkLayer`,
    - Expanded `[T, D]` `z_tilde` activation storage (saves compact `[M, D]` `z_bar`, ~4.6x-8x smaller),
    - `GatherBackward0` `scatter_add_` global atomic contention via lock-free Span-Segmented reduction.
    """

    @staticmethod
    def forward(
        ctx,
        z_hat: torch.Tensor,
        p_full: torch.Tensor,
        res_fp32: torch.Tensor,
        b_int: torch.Tensor,
        chunk_idx: torch.Tensor,
        source_pos: torch.Tensor,
        span_end: torch.Tensor,
        chunk_is_start: torch.Tensor,
        next_cu_seqlens: torch.Tensor,
        omit_ema_adjoint_bug: bool = False,
    ) -> torch.Tensor:
        t_len, d_dim = res_fp32.shape
        m_len = int(z_hat.shape[0])
        ctx.omit_ema_adjoint_bug = omit_ema_adjoint_bug

        if z_hat.dtype == torch.float64 or res_fp32.dtype == torch.float64:
            ctx.is_fp64 = True
            z64 = z_hat.to(torch.float64)
            p64 = p_full.to(torch.float64)
            r64 = res_fp32.to(torch.float64)
            p_sel = p64.index_select(0, source_pos.to(torch.long)) if m_len > 0 else torch.zeros((0,), dtype=torch.float64, device=z_hat.device)
            z_bar = fp64_hnet_affine_ema_oracle(z64, p_sel, next_cu_seqlens)
            z_tilde = z_bar.index_select(0, chunk_idx.to(torch.long)) if t_len > 0 else torch.zeros_like(r64)
            out = z_tilde + r64
            ctx.save_for_backward(z64, p_sel, z_bar, b_int, chunk_idx, source_pos, span_end, chunk_is_start)
            ctx.t_len = t_len
            ctx.p_dtype = p_full.dtype
            ctx.r_dtype = res_fp32.dtype
            return out

        ctx.is_fp64 = False
        p_sel = p_full.index_select(0, source_pos.to(torch.long)).contiguous().to(torch.float32) if m_len > 0 else torch.zeros((0,), dtype=torch.float32, device=z_hat.device)
        z_bar_fp32, z_bar_rnd = _run_parallel_ema_fwd(z_hat, p_sel, chunk_is_start)
        z_out = torch.empty((t_len, d_dim), dtype=z_hat.dtype, device=z_hat.device)
        if t_len > 0:
            bd = 256
            _fused_upsample_ste_residual_fwd_kernel[(t_len, triton.cdiv(d_dim, bd))](
                z_bar_rnd,
                chunk_idx,
                res_fp32,
                z_out,
                z_bar_rnd.stride(0),
                res_fp32.stride(0),
                res_fp32.stride(1),
                z_out.stride(0),
                d_dim,
                BD=bd,
                num_warps=4,
            )
        ctx.save_for_backward(z_hat, p_sel, z_bar_fp32, z_bar_rnd, b_int, chunk_idx, source_pos, span_end, chunk_is_start)
        ctx.t_len = t_len
        ctx.p_dtype = p_full.dtype
        ctx.r_dtype = res_fp32.dtype
        return z_out

    @staticmethod
    def backward(ctx, grad_z: torch.Tensor):
        t_len = ctx.t_len
        d_dim = grad_z.shape[1]
        if ctx.is_fp64:
            z64, p_sel, z_bar, b_int, chunk_idx, source_pos, span_end, chunk_is_start = ctx.saved_tensors
            g64 = grad_z.to(torch.float64)
            d_res = g64.to(ctx.r_dtype)
            m_len = int(z64.shape[0])
            if t_len == 0 or m_len == 0:
                return torch.zeros_like(z64), torch.zeros((t_len,), dtype=ctx.p_dtype, device=z64.device), d_res, None, None, None, None, None, None, None
            z_tilde = z_bar.index_select(0, chunk_idx.to(torch.long))
            sign = torch.where(b_int != 0, 1.0, -1.0).to(torch.float64)
            dp_total = sign * torch.sum(g64 * z_tilde, dim=-1)
            dz_bar = torch.zeros_like(z_bar)
            dz_bar.index_add_(0, chunk_idx.to(torch.long), g64)

            dz_hat = torch.zeros_like(z64)
            dp_sel = torch.zeros_like(p_sel)
            lam = torch.zeros((d_dim,), dtype=torch.float64, device=z64.device)
            st_list = chunk_is_start.detach().cpu().tolist()
            for j in range(m_len - 1, -1, -1):
                a_next = 0.0 if (ctx.omit_ema_adjoint_bug or j + 1 >= m_len or st_list[j + 1] != 0) else float((1.0 - p_sel[j + 1]).item())
                lam = dz_bar[j] + a_next * lam
                dz_hat[j] = p_sel[j] * lam
                prev_zb = torch.zeros_like(lam) if (j == 0 or st_list[j] != 0) else z_bar[j - 1]
                dp_sel[j] = torch.sum(lam * (z64[j] - prev_zb))
            dp_total.index_add_(0, source_pos.to(torch.long), dp_sel)
            return dz_hat, dp_total.to(ctx.p_dtype), d_res, None, None, None, None, None, None, None

        z_hat, p_sel, z_bar_fp32, z_bar_rnd, b_int, chunk_idx, source_pos, span_end, chunk_is_start = ctx.saved_tensors
        m_len = int(z_hat.shape[0])
        d_res = grad_z.to(ctx.r_dtype)
        dp_total = torch.zeros((t_len,), dtype=torch.float32, device=grad_z.device)
        if t_len == 0 or m_len == 0:
            return torch.zeros_like(z_hat), dp_total.to(ctx.p_dtype), d_res, None, None, None, None, None, None, None

        # 1. Per-byte STE confidence gradient
        _upsample_ste_dot_bwd_kernel[(t_len,)](
            grad_z,
            z_bar_rnd,
            chunk_idx,
            b_int,
            dp_total,
            grad_z.stride(0),
            grad_z.stride(1),
            z_bar_rnd.stride(0),
            d_dim,
            BD=512,
            num_warps=4,
        )

        # 2. Lock-free Span-Segmented reduction for dz_bar[M, D] (adaptive Split-Span for collapsed M)
        mean_span = t_len // max(1, m_len)
        num_splits = 16 if (m_len < 64 and mean_span > 64) else (4 if mean_span > 32 else 1)
        bd = 128
        n_dblks = triton.cdiv(d_dim, bd)
        if num_splits == 1:
            dz_bar = torch.empty((m_len, d_dim), dtype=torch.float32, device=grad_z.device)
            _upsample_span_reduce_split_bwd_kernel[(m_len, n_dblks, 1)](
                grad_z,
                source_pos,
                span_end,
                dz_bar,
                grad_z.stride(0),
                grad_z.stride(1),
                dz_bar.stride(0),
                dz_bar.stride(0),
                d_dim,
                NUM_SPLITS=1,
                BD=bd,
                num_warps=4,
            )
        else:
            dz_split = torch.empty((m_len, num_splits, d_dim), dtype=torch.float32, device=grad_z.device)
            _upsample_span_reduce_split_bwd_kernel[(m_len, n_dblks, num_splits)](
                grad_z,
                source_pos,
                span_end,
                dz_split,
                grad_z.stride(0),
                grad_z.stride(1),
                dz_split.stride(0),
                dz_split.stride(1),
                d_dim,
                NUM_SPLITS=num_splits,
                BD=bd,
                num_warps=4,
            )
            dz_bar = dz_split.sum(dim=1)

        # 3. Reverse 2-level parallel EMA adjoint scan + bijective dp_sel scatter-add
        dz_hat, dp_sel = _run_parallel_ema_bwd(
            dz_bar,
            z_hat,
            p_sel,
            z_bar_fp32,
            chunk_is_start,
            omit_ema_adjoint_bug=ctx.omit_ema_adjoint_bug,
        )
        _add_bijective_dp_sel_kernel[(triton.cdiv(m_len, 256),)](
            dp_sel,
            source_pos,
            dp_total,
            m_len,
            BLOCK_M=256,
            num_warps=4,
        )
        return dz_hat, dp_total.to(ctx.p_dtype), d_res, None, None, None, None, None, None, None


# ==============================================================================
# 6. Official `goombalab/hnet` Faithful Baselines & Upstream Defect Auditors
# ==============================================================================


def _pytorch_parallel_affine_scan(decay: torch.Tensor, u_step: torch.Tensor, *, max_len_bound: int | None = None) -> torch.Tensor:
    """Zero-branch vectorized Hillis-Steele `O(log T)` parallel associative affine scan.

    Avoids data-dependent `if m_len > stride:` Python branches when `max_len_bound` is supplied:
    in PyTorch, when `M <= stride`, `u[stride:]` and `u[:-stride]` both evaluate to empty `[0, D]` slices
    so `cat([u[:stride], empty], dim=0)` is an exact zero-branch no-op that never triggers TorchInductor
    dynamic-shape recompilations across variable `M` steps!
    """
    if u_step.ndim == 2:
        bound = max_len_bound if max_len_bound is not None else int(u_step.shape[0])
        if bound <= 1:
            return u_step
        a = decay
        u = u_step
        n_steps = int(math.ceil(math.log2(max(2, bound))))
        for k in range(n_steps):
            stride = 1 << k
            a_hi = a[stride:]
            u = torch.cat([u[:stride], u[stride:] + a_hi.unsqueeze(-1) * u[:-stride]], dim=0)
            a = torch.cat([a[:stride], a_hi * a[:-stride]], dim=0)
        return u

    bound = max_len_bound if max_len_bound is not None else int(u_step.shape[1])
    if bound <= 1:
        return u_step
    a = decay
    u = u_step
    n_steps = int(math.ceil(math.log2(max(2, bound))))
    for k in range(n_steps):
        stride = 1 << k
        a_hi = a[:, stride:]
        u = torch.cat([u[:, :stride], u[:, stride:] + a_hi.unsqueeze(-1) * u[:, :-stride]], dim=1)
        a = torch.cat([a[:, :stride], a_hi * a[:, :-stride]], dim=1)
    return u


def official_hnet_ssd_surrogate_ema(
    z_hat: torch.Tensor,
    p_raw: torch.Tensor,
    next_cu_seqlens: torch.Tensor | None = None,
    *,
    ssd_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Faithful reproduction of `goombalab/hnet/hnet/modules/dc.py` L257-293 `DeChunkLayer` SSD surrogate.

    Maps scalar EMA `z_bar_j = P_j z_hat_j + (1 - P_j) z_bar_{j-1}` onto Mamba-2 SSD discrete state dynamics:
      `p = clamp(p_raw.float(), min=1e-4, max=1 - 1e-4)`
      `dt = log(1 / (1 - p)).to(ssd_dtype)`
      `x  = (z_hat / dt[..., None]).to(ssd_dtype)`
      `b  = p.to(ssd_dtype)`
      `h_j = exp(-dt_j) * h_{j-1} + dt_j * b_j * x_j`
    """
    _ = chunk_size
    orig_dtype = z_hat.dtype
    p = torch.clamp(p_raw.float(), min=1e-4, max=1.0 - 1e-4)
    dt = torch.log(1.0 / (1.0 - p)).to(ssd_dtype)
    x = (z_hat / dt.unsqueeze(-1)).to(ssd_dtype)
    b = p.to(ssd_dtype)

    dt_f = dt.float()
    decay = torch.exp(-dt_f)
    u_step = (dt_f * b.float()).unsqueeze(-1) * x.float()

    if z_hat.ndim == 2:
        assert next_cu_seqlens is not None
        m_len = int(z_hat.shape[0])
        if m_len == 0:
            return torch.zeros_like(z_hat)
        _, is_st, _ = build_packed_doc_metadata(next_cu_seqlens, m_len)
        decay = torch.where(is_st != 0, torch.zeros_like(decay), decay)
        return _pytorch_parallel_affine_scan(decay, u_step).to(orig_dtype)

    if z_hat.shape[1] > 0:
        decay = decay.clone()
        decay[:, 0] = 0.0
    return _pytorch_parallel_affine_scan(decay, u_step).to(orig_dtype)


def official_hnet_eager_batched_dc_step(
    x_enc: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_res_fp32: torch.Tensor,
    w_main: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Faithful batched `goombalab/hnet` `RoutingModule + ChunkLayer(argsort) + DeChunkLayer(argsort+SSD+gather) + STE`."""
    # RoutingModule (`dc.py` L94-137)
    u = F.linear(x_enc[:, :-1], w_q)
    v = F.linear(x_enc[:, 1:], w_k)
    cos_sim = torch.einsum("bld,bld->bl", F.normalize(u, dim=-1), F.normalize(v, dim=-1))
    bprob = torch.clamp((1.0 - cos_sim) * 0.5, min=0.0, max=1.0)
    bprob = F.pad(bprob, (1, 0), "constant", 1.0)
    bprob_2 = torch.stack((1.0 - bprob, bprob), dim=-1)
    selected_idx = torch.argmax(bprob_2, dim=-1)
    b_mask = (selected_idx == 1) & mask
    selected_probs = bprob_2.gather(dim=-1, index=selected_idx.unsqueeze(-1))

    # ChunkLayer (`dc.py` L182-206): synchronous `.item()`/`int(max())` + first `torch.argsort`
    num_tokens = b_mask.sum(dim=-1)
    next_max_seqlen = int(num_tokens.max())
    bsz, l_len, d_dim = x_enc.shape
    token_idx = torch.arange(l_len, device=x_enc.device)[None, :] + (~b_mask).long() * l_len
    seq_sorted_indices = torch.argsort(token_idx, dim=1)
    next_hidden = torch.gather(
        x_enc,
        dim=1,
        index=seq_sorted_indices[:, :next_max_seqlen, None].expand(-1, -1, d_dim),
    )

    # Main stage proxy
    z_hat = F.linear(F.silu(next_hidden), w_main) + next_hidden

    # DeChunkLayer (`dc.py` L257-309): second `torch.argsort` + BF16 `dt = -log(1 - p)` SSD + `torch.gather`
    p = torch.clamp(bprob_2[..., -1].float(), min=1e-4, max=1.0 - 1e-4)
    token_idx_2 = torch.arange(l_len, device=x_enc.device)[None, :] + (~b_mask).long() * l_len
    seq_sorted_indices_2 = torch.argsort(token_idx_2, dim=1)
    p_gathered = torch.gather(p, dim=1, index=seq_sorted_indices_2[:, : z_hat.shape[1]])
    z_bar = official_hnet_ssd_surrogate_ema(z_hat, p_gathered, ssd_dtype=torch.bfloat16)
    plug_back_idx = torch.cumsum(b_mask, dim=1) - 1
    dechunk_out = torch.gather(z_bar, dim=1, index=plug_back_idx.unsqueeze(-1).expand(-1, -1, d_dim))

    # Residual + STE (`hnet.py` L252-283) + official `load_balancing_loss` (`train.py` L30-41)
    residual = F.linear(x_enc.float(), w_res_fp32)
    z_out = (dechunk_out.float() * _ExactSTEFn.apply(selected_probs) + residual).to(x_enc.dtype)
    true_ratio = b_mask.float().mean()
    avg_prob = bprob_2[..., -1].float().mean()
    ratio_loss = ((1.0 - true_ratio) * (1.0 - avg_prob) + true_ratio * avg_prob * 3.0) * (4.0 / 3.0)
    return z_out, ratio_loss


def eager_packed_exact_dc_step(
    x_enc: torch.Tensor,
    w_qk: torch.Tensor,
    w_res_fp32: torch.Tensor,
    w_main: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    target_ratio_n: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact packed PyTorch Eager DC baseline (no `BF16 log/div` drift, still uses PyTorch normalize + boolean mask + gather)."""
    d_dim = x_enc.shape[1]
    qk = F.linear(x_enc, w_qk)
    u, v = qk[:, :d_dim], qk[:, d_dim:]
    u_hat = F.normalize(u.float(), dim=-1, eps=1e-12)
    v_hat = F.normalize(v.float(), dim=-1, eps=1e-12)
    cos = torch.sum(u_hat[:-1] * v_hat[1:], dim=-1)
    p = F.pad(torch.clamp((1.0 - cos) * 0.5, 0.0, 1.0), (1, 0), value=1.0)
    starts = cu_seqlens[:-1][cu_seqlens[1:] > cu_seqlens[:-1]].to(torch.long)
    mask_st = torch.zeros_like(p, dtype=torch.bool)
    mask_st[starts] = True
    p = torch.where(mask_st, torch.ones_like(p), p)
    b_mask = p > 0.5
    c_conf = torch.where(b_mask, p, 1.0 - p)

    pref_pad = F.pad(b_mask.to(torch.long).cumsum(dim=0), (1, 0), value=0)
    next_cu = pref_pad[cu_seqlens.to(torch.long)]
    plug_idx = pref_pad[1:] - 1

    x_pack = x_enc[b_mask]
    p_sel = p[b_mask]
    z_hat = F.linear(F.silu(x_pack), w_main) + x_pack

    # Vectorized Hillis-Steele O(log T) parallel EMA scan in pure PyTorch
    t_bound = int(x_enc.shape[0])
    doc_id, is_st, _ = build_packed_doc_metadata(next_cu, int(x_pack.shape[0]))
    decay = torch.where(is_st != 0, torch.zeros_like(p_sel), 1.0 - p_sel.float())
    u_step = p_sel.float().unsqueeze(-1) * z_hat.float()
    z_bar = _pytorch_parallel_affine_scan(decay, u_step, max_len_bound=t_bound).to(x_enc.dtype)
    z_tilde = torch.gather(z_bar, dim=0, index=plug_idx.unsqueeze(-1).expand(-1, d_dim))
    res = F.linear(x_enc.float() if w_res_fp32.dtype == torch.float32 else x_enc, w_res_fp32)
    z_out = (z_tilde.float() * _ExactSTEFn.apply(c_conf).unsqueeze(-1) + res).to(x_enc.dtype)

    f_frac = b_mask.float().mean().detach()
    g_mean = p.mean()
    ratio_loss = (target_ratio_n / (target_ratio_n - 1.0)) * (
        (target_ratio_n - 1.0) * f_frac * g_mean + (1.0 - f_frac) * (1.0 - g_mean)
    )
    return z_out, ratio_loss


def official_dc_ops_batched(
    qk_batched: torch.Tensor,
    x_enc_batched: torch.Tensor,
    res_batched: torch.Tensor,
    mask: torch.Tensor,
    *,
    main_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isolated pure Dynamic Chunking operators matching `goombalab/hnet/hnet/modules/dc.py` (`Track 1` baseline)."""
    bsz, l_len, d_dim = x_enc_batched.shape
    u = qk_batched[:, :-1, :d_dim]
    v = qk_batched[:, 1:, d_dim:]
    cos_sim = torch.einsum("bld,bld->bl", F.normalize(u, dim=-1), F.normalize(v, dim=-1))
    bprob = torch.clamp((1.0 - cos_sim) * 0.5, min=0.0, max=1.0)
    bprob = F.pad(bprob, (1, 0), "constant", 1.0)
    bprob_2 = torch.stack((1.0 - bprob, bprob), dim=-1)
    selected_idx = torch.argmax(bprob_2, dim=-1)
    b_mask = (selected_idx == 1) & mask
    selected_probs = bprob_2.gather(dim=-1, index=selected_idx.unsqueeze(-1))

    num_tokens = b_mask.sum(dim=-1)
    next_max_seqlen = int(num_tokens.max())
    token_idx = torch.arange(l_len, device=x_enc_batched.device)[None, :] + (~b_mask).long() * l_len
    seq_sorted_indices = torch.argsort(token_idx, dim=1)
    x_packed = torch.gather(
        x_enc_batched,
        dim=1,
        index=seq_sorted_indices[:, :next_max_seqlen, None].expand(-1, -1, d_dim),
    )

    z_hat = F.silu(x_packed) * main_scale + x_packed

    p = torch.clamp(bprob_2[..., -1].float(), min=1e-4, max=1.0 - 1e-4)
    token_idx_2 = torch.arange(l_len, device=x_enc_batched.device)[None, :] + (~b_mask).long() * l_len
    seq_sorted_indices_2 = torch.argsort(token_idx_2, dim=1)
    p_gathered = torch.gather(p, dim=1, index=seq_sorted_indices_2[:, : z_hat.shape[1]])
    z_bar = official_hnet_ssd_surrogate_ema(z_hat, p_gathered, ssd_dtype=torch.bfloat16)
    plug_back_idx = torch.cumsum(b_mask, dim=1) - 1
    dechunk_out = torch.gather(z_bar, dim=1, index=plug_back_idx.unsqueeze(-1).expand(-1, -1, d_dim))

    z_out = (dechunk_out.float() * _ExactSTEFn.apply(selected_probs) + res_batched.float()).to(x_enc_batched.dtype)
    true_ratio = b_mask.float().mean()
    avg_prob = bprob_2[..., -1].float().mean()
    ratio_loss = ((1.0 - true_ratio) * (1.0 - avg_prob) + true_ratio * avg_prob * 3.0) * (4.0 / 3.0)
    return z_out, ratio_loss


def eager_dc_ops_packed(
    qk_packed: torch.Tensor,
    x_enc_packed: torch.Tensor,
    res_packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    main_scale: torch.Tensor,
    target_ratio_n: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact packed PyTorch Eager Dynamic Chunking operators (`F.normalize` + boolean pack + sequential EMA + `torch.gather`)."""
    d_dim = x_enc_packed.shape[1]
    u, v = qk_packed[:, :d_dim], qk_packed[:, d_dim:]
    u_hat = F.normalize(u.float(), dim=-1, eps=1e-12)
    v_hat = F.normalize(v.float(), dim=-1, eps=1e-12)
    cos = torch.sum(u_hat[:-1] * v_hat[1:], dim=-1)
    p = F.pad(torch.clamp((1.0 - cos) * 0.5, 0.0, 1.0), (1, 0), value=1.0)
    starts = cu_seqlens[:-1][cu_seqlens[1:] > cu_seqlens[:-1]].to(torch.long)
    mask_st = torch.zeros_like(p, dtype=torch.bool)
    mask_st[starts] = True
    p = torch.where(mask_st, torch.ones_like(p), p)
    b_mask = p > 0.5
    c_conf = torch.where(b_mask, p, 1.0 - p)

    pref_pad = F.pad(b_mask.to(torch.long).cumsum(dim=0), (1, 0), value=0)
    next_cu = pref_pad[cu_seqlens.to(torch.long)]
    plug_idx = pref_pad[1:] - 1

    x_pack = x_enc_packed[b_mask]
    p_sel = p[b_mask]
    z_hat = (F.silu(x_pack) * main_scale + x_pack).to(x_enc_packed.dtype)

    # Vectorized Hillis-Steele O(log T) parallel EMA scan (`<= 14` steps, dynamic-M compile friendly!)
    t_bound = int(x_enc_packed.shape[0])
    m_len = int(z_hat.shape[0])
    _, is_st, _ = build_packed_doc_metadata(next_cu, m_len)
    decay = torch.where(is_st != 0, torch.zeros_like(p_sel), 1.0 - p_sel.float())
    u_step = p_sel.float().unsqueeze(-1) * z_hat.float()
    z_bar = _pytorch_parallel_affine_scan(decay, u_step, max_len_bound=t_bound).to(x_enc_packed.dtype)

    z_tilde = torch.gather(z_bar, dim=0, index=plug_idx.unsqueeze(-1).expand(-1, d_dim))
    z_out = (z_tilde.float() * _ExactSTEFn.apply(c_conf).unsqueeze(-1) + res_packed.float()).to(x_enc_packed.dtype)

    f_frac = b_mask.float().mean().detach()
    g_mean = p.mean()
    ratio_loss = (target_ratio_n / (target_ratio_n - 1.0)) * (
        (target_ratio_n - 1.0) * f_frac * g_mean + (1.0 - f_frac) * (1.0 - g_mean)
    )
    return z_out, ratio_loss


def rsi_round7_superfused_dc_ops(
    qk_packed: torch.Tensor,
    x_enc_packed: torch.Tensor,
    res_packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    main_scale: torch.Tensor,
    doc_meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    target_ratio_n: float = 4.0,
    omit_radial_bug: bool = False,
    omit_ema_adjoint_bug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, PackedChunkMetadata]:
    """Isolated Round-7 RSI Zero-Spill Dynamic Chunking (`Merged-QK Router + Zero-Sort Pack + Superfused EMA/Upsample/STE/Residual`)."""
    t_len = x_enc_packed.shape[0]
    if doc_meta is None:
        doc_id, is_start, is_last = build_packed_doc_metadata(cu_seqlens, t_len)
    else:
        doc_id, is_start, is_last = doc_meta

    p, b_int = FusedHNetMergedQKCosineRouterFn.apply(qk_packed, is_start, is_last, 1e-12, omit_radial_bug)
    x_packed, _p_sel_unused, meta = zero_sort_prefix_chunk_pack(x_enc_packed, p, b_int, doc_id, cu_seqlens)
    z_hat = (F.silu(x_packed) * main_scale + x_packed).to(x_enc_packed.dtype)
    z_out = FusedDechunkEMASmoothUpsampleSTEResidualFn.apply(
        z_hat,
        p,
        res_packed,
        b_int,
        meta.chunk_idx,
        meta.source_pos,
        meta.span_end,
        meta.chunk_is_start,
        meta.next_cu_seqlens,
        omit_ema_adjoint_bug,
    )
    f_frac = float(meta.m_total) / float(max(1, t_len))
    g_mean = p.mean()
    ratio_loss = (target_ratio_n / (target_ratio_n - 1.0)) * (
        (target_ratio_n - 1.0) * f_frac * g_mean + (1.0 - f_frac) * (1.0 - g_mean)
    )
    return z_out, ratio_loss, meta


def rsi_round7_superfused_dc_step(
    x_enc: torch.Tensor,
    w_qk: torch.Tensor,
    w_res: torch.Tensor,
    w_main: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    doc_meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    target_ratio_n: float = 4.0,
    omit_radial_bug: bool = False,
    omit_ema_adjoint_bug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, PackedChunkMetadata]:
    """Full Round-7 RSI Zero-Spill Dynamic Chunking (`Merged-QK Router + Zero-Sort Pack + Main + Superfused EMA/Upsample/STE/Residual`)."""
    t_len, _d_dim = x_enc.shape
    if doc_meta is None:
        doc_id, is_start, is_last = build_packed_doc_metadata(cu_seqlens, t_len)
    else:
        doc_id, is_start, is_last = doc_meta

    qk = F.linear(x_enc, w_qk)
    p, b_int = FusedHNetMergedQKCosineRouterFn.apply(qk, is_start, is_last, 1e-12, omit_radial_bug)
    x_packed, _p_sel_unused, meta = zero_sort_prefix_chunk_pack(x_enc, p, b_int, doc_id, cu_seqlens)

    z_hat = F.linear(F.silu(x_packed), w_main) + x_packed
    res = F.linear(x_enc.float() if w_res.dtype == torch.float32 else x_enc, w_res)
    z_out = FusedDechunkEMASmoothUpsampleSTEResidualFn.apply(
        z_hat,
        p,
        res,
        b_int,
        meta.chunk_idx,
        meta.source_pos,
        meta.span_end,
        meta.chunk_is_start,
        meta.next_cu_seqlens,
        omit_ema_adjoint_bug,
    )

    f_frac = (float(meta.m_total) / float(max(1, t_len)))
    g_mean = p.mean()
    ratio_loss = (target_ratio_n / (target_ratio_n - 1.0)) * (
        (target_ratio_n - 1.0) * f_frac * g_mean + (1.0 - f_frac) * (1.0 - g_mean)
    )
    return z_out, ratio_loss, meta


# ==============================================================================
# 7. Distributed 8-GPU NVSwitch Whole-Document Cost-LPT `AllToAllv` Routing
# ==============================================================================


class _AllToAllvAutogradFn(torch.autograd.Function):
    """Differentiable unitary NVSwitch `all_to_all_single` over variable `[M_r, D]` compressed chunk buffers."""

    @staticmethod
    def forward(
        ctx,
        send_buf: torch.Tensor,
        send_splits: list[int],
        recv_splits: list[int],
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        ctx.send_splits = list(send_splits)
        ctx.recv_splits = list(recv_splits)
        ctx.group = group
        d_dim = send_buf.shape[1]
        recv_total = int(sum(recv_splits))
        recv_buf = torch.empty((recv_total, d_dim), dtype=send_buf.dtype, device=send_buf.device)
        dist.all_to_all_single(
            recv_buf,
            send_buf.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=group,
        )
        return recv_buf

    @staticmethod
    def backward(ctx, grad_recv: torch.Tensor):
        d_dim = grad_recv.shape[1]
        send_total = int(sum(ctx.send_splits))
        grad_send = torch.empty((send_total, d_dim), dtype=grad_recv.dtype, device=grad_recv.device)
        dist.all_to_all_single(
            grad_send,
            grad_recv.contiguous(),
            output_split_sizes=ctx.send_splits,
            input_split_sizes=ctx.recv_splits,
            group=ctx.group,
        )
        return grad_send, None, None, None


@dataclass
class CostLPTRoutePlan:
    send_perm_docs: list[int]
    inv_send_perm_chunks: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    recv_cu_seqlens: torch.Tensor
    recv_max_seqlen: int
    imbalance_before: float
    imbalance_after: float
    comm_bytes_per_rank: float


def plan_and_dispatch_cost_lpt_chunks(
    x_packed: torch.Tensor,
    next_cu_seqlens: torch.Tensor,
    *,
    d_main: int,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, CostLPTRoutePlan]:
    """Assigns complete compressed documents across 8 NVSwitch GPUs via Longest-Processing-Time (Cost-LPT).

    Preserves causal sequence integrity (`flash_attn_varlen_func` operates on whole documents!) and returns
    the exact inverse permutation so `combine_cost_lpt_chunks` restores each rank's local document order.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    dev = x_packed.device

    cu_i32 = next_cu_seqlens.to(torch.int32).contiguous()
    b_local = int(cu_i32.numel()) - 1
    gathered_cu = [torch.empty_like(cu_i32) for _ in range(world_size)]
    dist.all_gather(gathered_cu, cu_i32, group=group)
    all_cu = torch.stack(gathered_cu, dim=0).cpu().tolist()  # single tiny [W, B_local+1] host read
    cu_cpu = all_cu[rank]
    all_lens = [[int(all_cu[r][b + 1] - all_cu[r][b]) for b in range(b_local)] for r in range(world_size)]

    # Quadratic + linear attention+MLP cost model per document
    def _doc_cost(m_doc: int) -> float:
        return float(12 * d_main * m_doc + 4 * (m_doc * m_doc))

    raw_rank_costs = [sum(_doc_cost(int(m)) for m in all_lens[r]) for r in range(world_size)]
    mean_cost = max(1.0, sum(raw_rank_costs) / float(world_size))
    imbalance_before = max(raw_rank_costs) / mean_cost

    docs_global: list[tuple[float, int, int, int]] = []
    for r in range(world_size):
        for b in range(b_local):
            m_len = int(all_lens[r][b])
            docs_global.append((_doc_cost(m_len), r, b, m_len))
    # Deterministic LPT sort: descending cost, tie-broken by (orig_rank, doc_idx)
    docs_global.sort(key=lambda item: (-item[0], item[1], item[2]))

    worker_loads = [0.0 for _ in range(world_size)]
    dest_for_doc = [[0 for _ in range(b_local)] for _ in range(world_size)]
    recv_docs_by_src: list[list[list[tuple[int, int]]]] = [[[] for _ in range(world_size)] for _ in range(world_size)]

    for cost, orig_r, b_idx, m_len in docs_global:
        best_w = min(range(world_size), key=lambda w: (worker_loads[w], w))
        worker_loads[best_w] += cost
        dest_for_doc[orig_r][b_idx] = best_w
        recv_docs_by_src[best_w][orig_r].append((b_idx, m_len))

    imbalance_after = max(worker_loads) / mean_cost

    # Order local documents grouped by destination worker w = 0 .. W-1 (matching recv_docs_by_src[w][rank] order!)
    docs_by_dest: list[list[int]] = [[] for _ in range(world_size)]
    for b_idx, _m_len in [
        (bi, ml)
        for w in range(world_size)
        for (bi, ml) in recv_docs_by_src[w][rank]
    ]:
        docs_by_dest[dest_for_doc[rank][b_idx]].append(b_idx)

    send_perm_docs: list[int] = []
    send_splits: list[int] = []
    cpu_ranges: list[torch.Tensor] = []
    for w in range(world_size):
        tok_w = 0
        for b_idx in docs_by_dest[w]:
            send_perm_docs.append(b_idx)
            s, e = int(cu_cpu[b_idx]), int(cu_cpu[b_idx + 1])
            tok_w += (e - s)
            if e > s:
                cpu_ranges.append(torch.arange(s, e, dtype=torch.long))
        send_splits.append(tok_w)

    m_local_total = int(x_packed.shape[0])
    if send_perm_docs == list(range(b_local)):
        inv_chunk_perm = torch.arange(m_local_total, device=dev, dtype=torch.long)
        send_buf = x_packed
    elif cpu_ranges:
        chunk_perm_cpu = torch.cat(cpu_ranges, dim=0)
        inv_cpu = torch.empty_like(chunk_perm_cpu)
        inv_cpu[chunk_perm_cpu] = torch.arange(m_local_total, dtype=torch.long)
        chunk_perm = chunk_perm_cpu.to(device=dev, non_blocking=True)
        inv_chunk_perm = inv_cpu.to(device=dev, non_blocking=True)
        send_buf = x_packed.index_select(0, chunk_perm)
    else:
        inv_chunk_perm = torch.zeros((0,), device=dev, dtype=torch.long)
        send_buf = x_packed

    recv_splits: list[int] = []
    recv_doc_lens: list[int] = []
    for src_r in range(world_size):
        s_tok = 0
        for _bi, ml in recv_docs_by_src[rank][src_r]:
            s_tok += ml
            if ml > 0:
                recv_doc_lens.append(ml)
        recv_splits.append(s_tok)

    recv_cu_cpu = [0]
    for ml in recv_doc_lens:
        recv_cu_cpu.append(recv_cu_cpu[-1] + ml)
    recv_cu_seqlens = torch.tensor(recv_cu_cpu, dtype=torch.int32, device=dev)
    recv_max_seqlen = max(recv_doc_lens) if recv_doc_lens else 0

    x_recv = _AllToAllvAutogradFn.apply(send_buf, send_splits, recv_splits, group)
    elem_bytes = x_packed.element_size()
    # Forward dispatch + return + backward dispatch + return = 2 round-trips = 2 * (send + recv)
    comm_bytes = float((sum(send_splits) + sum(recv_splits)) * x_packed.shape[1] * elem_bytes * 2)
    plan = CostLPTRoutePlan(
        send_perm_docs=send_perm_docs,
        inv_send_perm_chunks=inv_chunk_perm,
        send_splits=send_splits,
        recv_splits=recv_splits,
        recv_cu_seqlens=recv_cu_seqlens,
        recv_max_seqlen=recv_max_seqlen,
        imbalance_before=imbalance_before,
        imbalance_after=imbalance_after,
        comm_bytes_per_rank=comm_bytes,
    )
    return x_recv, plan


def combine_cost_lpt_chunks(
    z_recv: torch.Tensor,
    plan: CostLPTRoutePlan,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Inverse `AllToAllv` + exact chunk-order unpermutation back to the owning rank."""
    z_back_perm = _AllToAllvAutogradFn.apply(z_recv, plan.recv_splits, plan.send_splits, group)
    if plan.inv_send_perm_chunks.numel() > 0:
        return z_back_perm.index_select(0, plan.inv_send_perm_chunks)
    return z_back_perm


# ==============================================================================
# 8. SM80 Cubin Register-Spill Auditor Across All 13 Round-7 RSI Triton Kernels
# ==============================================================================


def audit_hnet_triton_cubin_spills(device: torch.device | str = "cuda:0") -> dict[str, Any]:
    """Compiles & inspects SM80 SASS/PTX spill metadata (`n_spills`) across all 13 Round-7 RSI Triton kernels."""
    dev = torch.device(device)
    # Warm up representative specializations across D = 1024, 2048, 4096 and collapsed/normal M
    for d_dim in (1024, 2048, 4096):
        cu = torch.tensor([0, 256, 512], dtype=torch.int32, device=dev)
        x = torch.randn((512, d_dim), dtype=torch.bfloat16, device=dev, requires_grad=True)
        w_qk = torch.randn((2 * d_dim, d_dim), dtype=torch.bfloat16, device=dev, requires_grad=True) * 0.02
        w_res = torch.zeros((d_dim, d_dim), dtype=torch.float32, device=dev, requires_grad=True)
        w_main = torch.randn((d_dim, d_dim), dtype=torch.bfloat16, device=dev, requires_grad=True) * 0.02
        out, rloss, _ = rsi_round7_superfused_dc_step(x, w_qk, w_res, w_main, cu)
        (out.float().sum() + rloss).backward()

    # Also warm up multi-split span reduction (M < 64, mean_span > 64)
    cu1 = torch.tensor([0, 512], dtype=torch.int32, device=dev)
    x_rep = torch.ones((512, 1024), dtype=torch.bfloat16, device=dev, requires_grad=True)
    w_qk1 = torch.eye(1024, dtype=torch.bfloat16, device=dev).repeat(2, 1).detach().requires_grad_(True)
    w_res1 = torch.zeros((1024, 1024), dtype=torch.float32, device=dev, requires_grad=True)
    w_main1 = (torch.eye(1024, dtype=torch.bfloat16, device=dev) * 0.01).detach().requires_grad_(True)
    out1, rloss1, _ = rsi_round7_superfused_dc_step(x_rep, w_qk1, w_res1, w_main1, cu1)
    (out1.float().sum() + rloss1).backward()

    # Also warm up multi-branch 2-pass router (>4096 fallback) & standalone router kernels
    u_st = torch.randn((32, 256), dtype=torch.bfloat16, device=dev, requires_grad=True)
    v_st = torch.randn((32, 256), dtype=torch.bfloat16, device=dev, requires_grad=True)
    _, is_st0, is_la0 = build_packed_doc_metadata(torch.tensor([0, 32], dtype=torch.int32, device=dev), 32)
    p0, _ = FusedHNetCosineRouterFn.apply(u_st, v_st, is_st0, is_la0, 1e-12, False)
    p0.sum().backward()

    # Compile forensic naive 1-pass [8, BD=4096], nw=16 spill kernel (`512` threads -> `128` max regs/thread on SM80)
    u_sp = torch.randn((16, 4096), dtype=torch.bfloat16, device=dev)
    v_sp = torch.randn((16, 4096), dtype=torch.bfloat16, device=dev)
    dp_sp = torch.randn((16,), dtype=torch.float32, device=dev)
    du_sp = torch.empty_like(u_sp)
    dv_sp = torch.empty_like(v_sp)
    _audit_naive_router_bwd_d4096_nw16_spill_kernel[(1,)](
        u_sp, v_sp, dp_sp, du_sp, dv_sp, u_sp.stride(0), 16, BD=4096, num_warps=16
    )
    torch.cuda.synchronize(dev)

    def _collect_compiled(kobj):
        found = []
        for dev_entry in getattr(kobj, "device_caches", {}).values():
            cdict = dev_entry[0] if isinstance(dev_entry, tuple) and len(dev_entry) > 0 else dev_entry
            if isinstance(cdict, dict):
                for comp in cdict.values():
                    if hasattr(comp, "n_spills"):
                        found.append(comp)
        if not found:
            for cv in getattr(kobj, "cache", {}).values():
                entries = list(cv.values()) if isinstance(cv, dict) else [cv]
                for comp in entries:
                    if hasattr(comp, "n_spills"):
                        found.append(comp)
        return found

    rsi_kernels = [
        ("_hnet_doc_metadata_kernel", _hnet_doc_metadata_kernel),
        ("_fused_hnet_cosine_router_fwd_kernel", _fused_hnet_cosine_router_fwd_kernel),
        ("_fused_hnet_cosine_router_bwd_kernel", _fused_hnet_cosine_router_bwd_kernel),
        ("_fused_hnet_merged_qk_router_1pass_fwd_kernel", _fused_hnet_merged_qk_router_1pass_fwd_kernel),
        ("_fused_hnet_merged_qk_router_bwd_kernel", _fused_hnet_merged_qk_router_bwd_kernel),
        ("_zero_sort_prefix_scatter_meta_kernel", _zero_sort_prefix_scatter_meta_kernel),
        ("_build_chunk_span_end_and_doc_start_kernel", _build_chunk_span_end_and_doc_start_kernel),
        ("_gather_selected_chunks_fwd_kernel", _gather_selected_chunks_fwd_kernel),
        ("_scatter_selected_chunks_bwd_kernel", _scatter_selected_chunks_bwd_kernel),
        ("_ema_fwd_block_summary_kernel", _ema_fwd_block_summary_kernel),
        ("_ema_fwd_inter_and_intra_scan_kernel", _ema_fwd_inter_and_intra_scan_kernel),
        ("_ema_bwd_block_summary_kernel", _ema_bwd_block_summary_kernel),
        ("_ema_bwd_inter_and_intra_scan_kernel", _ema_bwd_inter_and_intra_scan_kernel),
        ("_fused_upsample_ste_residual_fwd_kernel", _fused_upsample_ste_residual_fwd_kernel),
        ("_upsample_ste_dot_bwd_kernel", _upsample_ste_dot_bwd_kernel),
        ("_upsample_span_reduce_split_bwd_kernel", _upsample_span_reduce_split_bwd_kernel),
        ("_add_bijective_dp_sel_kernel", _add_bijective_dp_sel_kernel),
    ]

    per_kernel: dict[str, dict[str, int]] = {}
    total_spills = 0
    total_specs = 0
    max_regs = 0
    for kname, kfn in rsi_kernels:
        compiled_entries = _collect_compiled(kfn)
        k_spills = 0
        k_regs = 0
        for ce in compiled_entries:
            sp = int(getattr(ce, "n_spills", 0) or 0)
            rg = int(getattr(ce, "n_regs", 0) or 0)
            k_spills += sp
            k_regs = max(k_regs, rg)
            total_specs += 1
        total_spills += k_spills
        max_regs = max(max_regs, k_regs)
        per_kernel[kname] = {
            "specializations": len(compiled_entries),
            "n_spills": k_spills,
            "max_n_regs": k_regs,
        }

    naive_spills = 0
    naive_regs = 0
    for ce in _collect_compiled(_audit_naive_router_bwd_d4096_nw16_spill_kernel):
        naive_spills = max(naive_spills, int(getattr(ce, "n_spills", 0) or 0))
        naive_regs = max(naive_regs, int(getattr(ce, "n_regs", 0) or 0))

    return {
        "num_rsi_kernels": len(rsi_kernels),
        "total_specializations": total_specs,
        "total_cubin_spills": total_spills,
        "max_n_regs": max_regs,
        "all_zero_spills": (total_spills == 0 and all(v["specializations"] > 0 for v in per_kernel.values())),
        "naive_1pass_router_d8192_nw16_spills": naive_spills,
        "naive_1pass_router_d8192_nw16_regs": naive_regs,
        "per_kernel": per_kernel,
    }
