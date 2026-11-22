#!/usr/bin/env python3
"""Round-3 SM80 Triton & Autograd Kernels for E129 (`v120`):
DeepSeek `mHC: Manifold-Constrained Hyper-Connections` (`arXiv:2512.24880`) Multi-Stream Residual Kernels.

Co-designed and adversarially audited via `/copilot-app-frontier-cowork` (`GPT 6 Astra, Max`):
1. Track 1 (`mhc_coefficient_and_sinkhorn_manifold`):
   - Exact `FP64` log-domain Sinkhorn-Knopp (`t_max = 20`) analytical oracle (`fp64_mhc_coeff_oracle`).
   - Single-Scan Tensor-Core Triton `[N, nC] @ diag(gamma)phi + Row-RMS` (`_single_scan_proj_rms_fwd_kernel`),
     eliminating the `[N, nC]` normalized intermediate HBM write/read (`Eq. 10-16`).
   - Zero-Spill On-Chip `FP32` Log-Domain Sinkhorn-Knopp (`t_max = 20`, `n in {2, 4, 8}`) Forward &
     5-Segment (`5 x 4`) Division-Free Reverse-Mode Adjoint (`0` cubin spills, immune to `+120` row-offset
     underflow `NaN`s where `exp(Z - col_max)` fails).
   - Consolidated 1-pass `dx = V @ A_cast^T - x * (a / (nC * r^3))` backward + exact `d_gamma = sum_j(dA * phi)`.
2. Track 2 (`mhc_stream_readout_pre_norm_and_post_res_merge`):
   - Superfused `F_pre + SublayerRMSNorm` (`X_l, H_pre, w_sub -> h_norm, rstd_sub`) preserving the exact
     `FP32 accum -> BF16 round -> FP32 RMS` boundary (`0` HBM bytes for `h_in`).
   - Vectorized `F_{post,res}` forward (`(3n+1)C -> (n+1)C` reads, `3nC -> nC` writes) and Unified
     Single-Read/Single-Write Sublayer-Sandwich Backward (`(X_l, y_l, H_pre, H_post, H_res, w_sub, rstd_sub,
     dh_norm, dX_{l+1}) -> (dX_l, dy_l, dH_pre, dH_post, dH_res, dw_sub)`) with Split-32 lock-free reductions.
3. Selective `L_r^*` Stream-Only Recompute (`mHC-Stream-Only-SAC`, DeepSeek Section 4.3.2 Eq. 20):
   - Stores `X_{l0} [N, n, C]` only every `L_r` sublayers + compact `y_l [N, C]` per sublayer, replaying
     only lightweight local `mHC` stream kernels in backward with zero extra Attention/MLP FLOPs and zero
     extra `FSDP2` `AllGather` collectives.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


# ============================================================================
# 0. Exact FP64 Oracles & Extreme-Logit / Red-Team Reference Functions
# ============================================================================


def log_sinkhorn_knopp_pytorch(z_res: torch.Tensor, t_max: int = 20) -> torch.Tensor:
    """Exact overflow/underflow-free log-domain Sinkhorn-Knopp (`M^(0)=exp(z)`, `M^(t)=T_r(T_c(M^(t-1)))`)."""
    L = z_res
    for _ in range(t_max):
        L = L - torch.logsumexp(L, dim=-2, keepdim=True)
        L = L - torch.logsumexp(L, dim=-1, keepdim=True)
    return torch.exp(L)


def naive_colmax_exp_sinkhorn_knopp(z_res: torch.Tensor, t_max: int = 20) -> torch.Tensor:
    """Broken candidate (`Trial 1`): column-max shift prevents overflow, NOT whole-row FP32 underflow (`0/0 = NaN`)."""
    m = torch.exp(z_res - z_res.amax(dim=-2, keepdim=True).detach())
    for _ in range(t_max):
        m = m / m.sum(dim=-2, keepdim=True)
        m = m / m.sum(dim=-1, keepdim=True)
    return m


def eager_mhc_coeff_eq7_eq9(
    X: torch.Tensor,
    phi: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-6,
    t_max: int = 20,
    use_broken_colmax_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Literal Eq. 7-9: RMSNorm over flattened `[N, nC]`, linear projection, and 20-step Sinkhorn-Knopp."""
    N, n_s, C = X.shape
    K = n_s * C
    x_flat = X.reshape(N, K)
    xf = x_flat.float()
    rstd = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    x_norm = (xf * rstd * gamma.float()).to(X.dtype)
    u = (x_norm @ phi.to(X.dtype)).float()

    zp = alpha[0].float() * u[:, :n_s] + b[:n_s].float()
    zo = alpha[1].float() * u[:, n_s : 2 * n_s] + b[n_s : 2 * n_s].float()
    zr = (alpha[2].float() * u[:, 2 * n_s :] + b[2 * n_s :].float()).view(N, n_s, n_s)

    h_pre = torch.sigmoid(zp)
    h_post = 2.0 * torch.sigmoid(zo)
    if use_broken_colmax_exp:
        h_res = naive_colmax_exp_sinkhorn_knopp(zr, t_max=t_max)
    else:
        h_res = log_sinkhorn_knopp_pytorch(zr, t_max=t_max)
    return h_pre, h_post, h_res


def fp64_mhc_coeff_forward(
    X: torch.Tensor,
    phi: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-6,
    t_max: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure FP64 analytical reference for mHC dynamic/static manifold coefficients (Gate G5 & G6)."""
    N, n_s, C = X.shape
    x64 = X.reshape(N, n_s * C).to(torch.float64)
    phi64 = phi.to(torch.float64)
    g64 = gamma.to(torch.float64)
    a64 = alpha.to(torch.float64)
    b64 = b.to(torch.float64)

    rstd = torch.rsqrt(torch.mean(x64 * x64, dim=-1, keepdim=True) + eps)
    u = (x64 * rstd * g64) @ phi64
    zp = a64[0] * u[:, :n_s] + b64[:n_s]
    zo = a64[1] * u[:, n_s : 2 * n_s] + b64[n_s : 2 * n_s]
    zr = (a64[2] * u[:, 2 * n_s :] + b64[2 * n_s :]).view(N, n_s, n_s)
    hp = torch.sigmoid(zp)
    ho = 2.0 * torch.sigmoid(zo)
    hr = log_sinkhorn_knopp_pytorch(zr, t_max=t_max)
    return hp, ho, hr


def fp64_mhc_coeff_oracle(
    X: torch.Tensor,
    phi: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    dhp: torch.Tensor,
    dho: torch.Tensor,
    dhr: torch.Tensor,
    *,
    eps: float = 1e-6,
    t_max: int = 20,
) -> dict[str, torch.Tensor]:
    X64 = X.detach().to(torch.float64).requires_grad_(True)
    phi64 = phi.detach().to(torch.float64).requires_grad_(True)
    g64 = gamma.detach().to(torch.float64).requires_grad_(True)
    a64 = alpha.detach().to(torch.float64).requires_grad_(True)
    b64 = b.detach().to(torch.float64).requires_grad_(True)
    hp, ho, hr = fp64_mhc_coeff_forward(X64, phi64, g64, a64, b64, eps=eps, t_max=t_max)
    loss = (
        (hp * dhp.to(torch.float64)).sum()
        + (ho * dho.to(torch.float64)).sum()
        + (hr * dhr.to(torch.float64)).sum()
    )
    loss.backward()
    return {
        "h_pre": hp.detach(),
        "h_post": ho.detach(),
        "h_res": hr.detach(),
        "dX": X64.grad.detach(),
        "d_phi": phi64.grad.detach(),
        "d_gamma": g64.grad.detach(),
        "d_alpha": a64.grad.detach(),
        "d_b": b64.grad.detach(),
    }


def eager_mhc_stream_sandwich(
    X: torch.Tensor,
    y: torch.Tensor,
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    w_sub: torch.Tensor,
    *,
    eps: float = 1e-6,
    round_hin_bf16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Table-2 Eager PyTorch stream readout + sublayer RMSNorm + post/res merge (`(5n+1)C` reads, `(3n+1)C` writes)."""
    h_in_f = torch.einsum("bi,bic->bc", h_pre.float(), X.float())
    if round_hin_bf16 and X.dtype == torch.bfloat16:
        h_in = h_in_f.to(torch.bfloat16)
        hf = h_in.float()
    else:
        hf = h_in_f
    rstd = torch.rsqrt(torch.mean(hf * hf, dim=-1, keepdim=True) + eps)
    h_norm = (hf * rstd * w_sub.float()).to(X.dtype)

    x_res = torch.einsum("bij,bjc->bic", h_res.float(), X.float()).to(X.dtype)
    x_post = (h_post.float().unsqueeze(-1) * y.float().unsqueeze(1)).to(X.dtype)
    X_next = x_res + x_post
    return h_norm, X_next


def fp64_mhc_stream_forward(
    X: torch.Tensor,
    y: torch.Tensor,
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    w_sub: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    X64 = X.to(torch.float64)
    y64 = y.to(torch.float64)
    hp64 = h_pre.to(torch.float64)
    ho64 = h_post.to(torch.float64)
    hr64 = h_res.to(torch.float64)
    w64 = w_sub.to(torch.float64)

    h_in = torch.einsum("bi,bic->bc", hp64, X64)
    rstd = torch.rsqrt(torch.mean(h_in * h_in, dim=-1, keepdim=True) + eps)
    h_norm = h_in * rstd * w64
    X_next = torch.einsum("bij,bjc->bic", hr64, X64) + ho64.unsqueeze(-1) * y64.unsqueeze(1)
    return h_norm, X_next


def fp64_mhc_stream_oracle(
    X: torch.Tensor,
    y: torch.Tensor,
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    w_sub: torch.Tensor,
    dh_norm: torch.Tensor,
    dX_next: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    X64 = X.detach().to(torch.float64).requires_grad_(True)
    y64 = y.detach().to(torch.float64).requires_grad_(True)
    hp64 = h_pre.detach().to(torch.float64).requires_grad_(True)
    ho64 = h_post.detach().to(torch.float64).requires_grad_(True)
    hr64 = h_res.detach().to(torch.float64).requires_grad_(True)
    w64 = w_sub.detach().to(torch.float64).requires_grad_(True)

    hn, xn = fp64_mhc_stream_forward(X64, y64, hp64, ho64, hr64, w64, eps=eps)
    loss = (hn * dh_norm.to(torch.float64)).sum() + (xn * dX_next.to(torch.float64)).sum()
    loss.backward()
    return {
        "h_norm": hn.detach(),
        "X_next": xn.detach(),
        "dX": X64.grad.detach(),
        "dy": y64.grad.detach(),
        "dh_pre": hp64.grad.detach(),
        "dh_post": ho64.grad.detach(),
        "dh_res": hr64.grad.detach(),
        "dw_sub": w64.grad.detach(),
    }


# ============================================================================
# 1. Track 1 SM80 Triton Kernels: Norm-After-Matmul + On-Chip Log-Sinkhorn-20
# ============================================================================


@triton.jit
def _row_rms_inv_kernel(
    X_ptr, InvR_ptr, stride_xn, K_DIM, eps, BLOCK_K: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K_DIM, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        m = offs < K_DIM
        x = tl.load(X_ptr + row * stride_xn + offs, mask=m, other=0.0).to(tl.float32)
        acc += x * x
    inv_r = tl.rsqrt(tl.sum(acc, axis=0) / K_DIM + eps)
    tl.store(InvR_ptr + row, inv_r)


@triton.jit
def _single_scan_proj_rms_fwd_kernel(
    X_ptr, A_ptr, U_ptr, InvR_ptr,
    stride_xn, stride_ak, stride_un,
    N_TOK, K_DIM, eps,
    M_DIM: tl.constexpr, M_PAD: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Single-pass Tensor-Core scan over `X [N, K]` computing both `U = X @ A` and `InvR = rsqrt(mean(X^2)+eps)`."""
    pid = tl.program_id(0)
    b_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < N_TOK
    m_offs = tl.arange(0, M_PAD)
    m_mask = m_offs < M_DIM

    acc_u = tl.zeros([BLOCK_B, M_PAD], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_B], dtype=tl.float32)

    for k0 in range(0, K_DIM, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_DIM
        x_tile = tl.load(
            X_ptr + b_offs[:, None].to(tl.int64) * stride_xn + k_offs[None, :],
            mask=b_mask[:, None] & k_mask[None, :], other=0.0,
        )
        xf = x_tile.to(tl.float32)
        acc_sq += tl.sum(xf * xf, axis=1)
        a_tile = tl.load(
            A_ptr + k_offs[:, None].to(tl.int64) * stride_ak + m_offs[None, :],
            mask=k_mask[:, None] & m_mask[None, :], other=0.0,
        )
        acc_u += tl.dot(x_tile, a_tile)

    inv_r = tl.rsqrt(acc_sq / K_DIM + eps)
    tl.store(InvR_ptr + b_offs, inv_r, mask=b_mask)
    tl.store(
        U_ptr + b_offs[:, None].to(tl.int64) * stride_un + m_offs[None, :],
        acc_u, mask=b_mask[:, None] & m_mask[None, :],
    )


@triton.jit
def _single_scan_dx_norm_proj_bwd_kernel(
    X_ptr, V_ptr, AT_ptr, Coeff_ptr, dX_ptr,
    stride_xn, stride_vn, stride_atm, stride_dxn,
    N_TOK, K_DIM, v_scale,
    M_DIM: tl.constexpr, M_PAD: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Single-pass Tensor-Core backward computing `dX = (V_norm @ A_cast^T)*v_scale - X * Coeff[:, None]`."""
    pid = tl.program_id(0)
    b_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < N_TOK
    m_offs = tl.arange(0, M_PAD)
    m_mask = m_offs < M_DIM

    v_tile = tl.load(
        V_ptr + b_offs[:, None].to(tl.int64) * stride_vn + m_offs[None, :],
        mask=b_mask[:, None] & m_mask[None, :], other=0.0,
    )
    v_cast = v_tile.to(AT_ptr.dtype.element_ty)
    coeff = tl.load(Coeff_ptr + b_offs, mask=b_mask, other=0.0).to(tl.float32)[:, None]

    for k0 in range(0, K_DIM, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_DIM
        at_tile = tl.load(
            AT_ptr + m_offs[:, None].to(tl.int64) * stride_atm + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        dx_proj = tl.dot(v_cast, at_tile) * v_scale
        x_tile = tl.load(
            X_ptr + b_offs[:, None].to(tl.int64) * stride_xn + k_offs[None, :],
            mask=b_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        dx_tot = dx_proj - x_tile * coeff
        tl.store(
            dX_ptr + b_offs[:, None].to(tl.int64) * stride_dxn + k_offs[None, :],
            dx_tot.to(dX_ptr.dtype.element_ty),
            mask=b_mask[:, None] & k_mask[None, :],
        )


@triton.jit
def _lse_col(L):
    m = tl.max(L, axis=1)[:, None, :]
    return m + tl.log(tl.sum(tl.exp(L - m), axis=1)[:, None, :])


@triton.jit
def _lse_row(L):
    m = tl.max(L, axis=2)[:, :, None]
    return m + tl.log(tl.sum(tl.exp(L - m), axis=2)[:, :, None])


@triton.jit
def _mhc_epilogue_log_sinkhorn20_fwd_kernel(
    U_ptr, InvR_ptr, Alpha_ptr, B_ptr,
    Hp_ptr, Ho_ptr, Hr_ptr, ZSave_ptr,
    stride_un, stride_zn, N_TOK,
    N_STREAM: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """Fuses Eq. 16-19 (`z = alpha*(u*inv_r)+b`, `sigmoid`, `2*sigmoid`, and 20-step Log-Sinkhorn-Knopp) in registers."""
    pid = tl.program_id(0)
    b_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < N_TOK
    inv_r = tl.load(InvR_ptr + b_offs, mask=b_mask, other=1.0).to(tl.float32)[:, None]

    ap = tl.load(Alpha_ptr + 0).to(tl.float32)
    ao = tl.load(Alpha_ptr + 1).to(tl.float32)
    ar = tl.load(Alpha_ptr + 2).to(tl.float32)

    offs_n = tl.arange(0, N_STREAM)
    up = tl.load(U_ptr + b_offs[:, None].to(tl.int64) * stride_un + offs_n[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
    bp = tl.load(B_ptr + offs_n).to(tl.float32)[None, :]
    zp = ap * (up * inv_r) + bp
    tl.store(ZSave_ptr + b_offs[:, None].to(tl.int64) * stride_zn + offs_n[None, :], zp, mask=b_mask[:, None])
    tl.store(Hp_ptr + b_offs[:, None].to(tl.int64) * N_STREAM + offs_n[None, :], tl.sigmoid(zp), mask=b_mask[:, None])

    uo = tl.load(U_ptr + b_offs[:, None].to(tl.int64) * stride_un + N_STREAM + offs_n[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
    bo = tl.load(B_ptr + N_STREAM + offs_n).to(tl.float32)[None, :]
    zo = ao * (uo * inv_r) + bo
    tl.store(ZSave_ptr + b_offs[:, None].to(tl.int64) * stride_zn + N_STREAM + offs_n[None, :], zo, mask=b_mask[:, None])
    tl.store(Ho_ptr + b_offs[:, None].to(tl.int64) * N_STREAM + offs_n[None, :], 2.0 * tl.sigmoid(zo), mask=b_mask[:, None])

    zr_off = (2 * N_STREAM) + offs_n[None, :, None] * N_STREAM + offs_n[None, None, :]
    ur = tl.load(U_ptr + b_offs[:, None, None].to(tl.int64) * stride_un + zr_off, mask=b_mask[:, None, None], other=0.0).to(tl.float32)
    br = tl.load(B_ptr + (2 * N_STREAM) + offs_n[:, None] * N_STREAM + offs_n[None, :]).to(tl.float32)[None, :, :]
    zr = ar * (ur * inv_r[:, :, None]) + br
    tl.store(ZSave_ptr + b_offs[:, None, None].to(tl.int64) * stride_zn + zr_off, zr, mask=b_mask[:, None, None])

    L = zr
    for _ in tl.static_range(20):
        L = L - _lse_col(L)
        L = L - _lse_row(L)
    hr = tl.exp(L)
    hr_base = Hr_ptr + (b_offs[:, None, None].to(tl.int64) * N_STREAM + offs_n[None, :, None]) * N_STREAM + offs_n[None, None, :]
    tl.store(hr_base, hr, mask=b_mask[:, None, None])


@triton.jit
def _mhc_epilogue_log_sinkhorn20_bwd_ckpt5_kernel(
    Z_ptr, U_ptr, InvR_ptr, Alpha_ptr, Hr_ptr,
    dHp_ptr, dHo_ptr, dHr_ptr,
    V_ptr, Coeff_ptr, dZ_ptr, PartAlpha_ptr,
    stride_zn, stride_un, stride_vn, N_TOK, K_DIM,
    DROP_DR_TERM: tl.constexpr,
    N_STREAM: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """5-segment (`5 x 4`) on-chip checkpointed division-free Log-Sinkhorn-20 VJP + `V = (dZ*alpha)*inv_r` & `Coeff`."""
    pid = tl.program_id(0)
    b_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < N_TOK
    offs_n = tl.arange(0, N_STREAM)

    inv_r = tl.load(InvR_ptr + b_offs, mask=b_mask, other=0.0).to(tl.float32)
    ap = tl.load(Alpha_ptr + 0).to(tl.float32)
    ao = tl.load(Alpha_ptr + 1).to(tl.float32)
    ar = tl.load(Alpha_ptr + 2).to(tl.float32)

    # 1) Sigmoid / 2*Sigmoid VJP
    zp = tl.load(Z_ptr + b_offs[:, None].to(tl.int64) * stride_zn + offs_n[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
    dhp = tl.load(dHp_ptr + b_offs[:, None].to(tl.int64) * N_STREAM + offs_n[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
    sp = tl.sigmoid(zp)
    dzp = tl.where(b_mask[:, None], dhp * sp * (1.0 - sp), 0.0)
    tl.store(dZ_ptr + b_offs[:, None].to(tl.int64) * stride_zn + offs_n[None, :], dzp, mask=b_mask[:, None])

    zo = tl.load(Z_ptr + b_offs[:, None].to(tl.int64) * stride_zn + N_STREAM + offs_n[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
    dho = tl.load(dHo_ptr + b_offs[:, None].to(tl.int64) * N_STREAM + offs_n[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
    so = tl.sigmoid(zo)
    dzo = tl.where(b_mask[:, None], dho * 2.0 * so * (1.0 - so), 0.0)
    tl.store(dZ_ptr + b_offs[:, None].to(tl.int64) * stride_zn + N_STREAM + offs_n[None, :], dzo, mask=b_mask[:, None])

    # 2) 5-segment x 4-step Log-Sinkhorn-20 VJP (`0` divisions, `0` spills!)
    zr_off = (2 * N_STREAM) + offs_n[None, :, None] * N_STREAM + offs_n[None, None, :]
    L0 = tl.load(Z_ptr + b_offs[:, None, None].to(tl.int64) * stride_zn + zr_off, mask=b_mask[:, None, None], other=0.0).to(tl.float32)

    cur = L0
    for _ in tl.static_range(4):
        cur = cur - _lse_col(cur)
        cur = cur - _lse_row(cur)
    c1 = cur
    for _ in tl.static_range(4):
        cur = cur - _lse_col(cur)
        cur = cur - _lse_row(cur)
    c2 = cur
    for _ in tl.static_range(4):
        cur = cur - _lse_col(cur)
        cur = cur - _lse_row(cur)
    c3 = cur
    for _ in tl.static_range(4):
        cur = cur - _lse_col(cur)
        cur = cur - _lse_row(cur)
    c4 = cur

    hr_base = (b_offs[:, None, None].to(tl.int64) * N_STREAM + offs_n[None, :, None]) * N_STREAM + offs_n[None, None, :]
    hr = tl.load(Hr_ptr + hr_base, mask=b_mask[:, None, None], other=0.0).to(tl.float32)
    dhr = tl.load(dHr_ptr + hr_base, mask=b_mask[:, None, None], other=0.0).to(tl.float32)
    g = tl.where(b_mask[:, None, None], dhr * hr, 0.0)

    for seg in tl.static_range(5):
        s_in = c4 if seg == 0 else (c3 if seg == 1 else (c2 if seg == 2 else (c1 if seg == 3 else L0)))
        lc0 = s_in - _lse_col(s_in); lr0 = lc0 - _lse_row(lc0)
        lc1 = lr0  - _lse_col(lr0);  lr1 = lc1 - _lse_row(lc1)
        lc2 = lr1  - _lse_col(lr1);  lr2 = lc2 - _lse_row(lc2)
        lc3 = lr2  - _lse_col(lr2);  lr3 = lc3 - _lse_row(lc3)

        g = g - tl.exp(lr3) * tl.sum(g, axis=2)[:, :, None]
        g = g - tl.exp(lc3) * tl.sum(g, axis=1)[:, None, :]
        g = g - tl.exp(lr2) * tl.sum(g, axis=2)[:, :, None]
        g = g - tl.exp(lc2) * tl.sum(g, axis=1)[:, None, :]
        g = g - tl.exp(lr1) * tl.sum(g, axis=2)[:, :, None]
        g = g - tl.exp(lc1) * tl.sum(g, axis=1)[:, None, :]
        g = g - tl.exp(lr0) * tl.sum(g, axis=2)[:, :, None]
        g = g - tl.exp(lc0) * tl.sum(g, axis=1)[:, None, :]

    dzr = tl.where(b_mask[:, None, None], g, 0.0)
    tl.store(dZ_ptr + b_offs[:, None, None].to(tl.int64) * stride_zn + zr_off, dzr, mask=b_mask[:, None, None])

    # 3) Compute V = (dZ * alpha) * inv_r, partial d_alpha [grid, 3], and Coeff = sum_j(q*u) * inv_r^3 / K_DIM
    up = tl.load(U_ptr + b_offs[:, None].to(tl.int64) * stride_un + offs_n[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
    uo = tl.load(U_ptr + b_offs[:, None].to(tl.int64) * stride_un + N_STREAM + offs_n[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
    ur = tl.load(U_ptr + b_offs[:, None, None].to(tl.int64) * stride_un + zr_off, mask=b_mask[:, None, None], other=0.0).to(tl.float32)

    dap = tl.sum(dzp * up * inv_r[:, None])
    dao = tl.sum(dzo * uo * inv_r[:, None])
    dar = tl.sum(dzr * ur * inv_r[:, None, None])
    tl.store(PartAlpha_ptr + pid * 3 + 0, dap)
    tl.store(PartAlpha_ptr + pid * 3 + 1, dao)
    tl.store(PartAlpha_ptr + pid * 3 + 2, dar)

    qp = dzp * ap
    qo = dzo * ao
    qr = dzr * ar
    tl.store(V_ptr + b_offs[:, None].to(tl.int64) * stride_vn + offs_n[None, :], qp * inv_r[:, None], mask=b_mask[:, None])
    tl.store(V_ptr + b_offs[:, None].to(tl.int64) * stride_vn + N_STREAM + offs_n[None, :], qo * inv_r[:, None], mask=b_mask[:, None])
    tl.store(V_ptr + b_offs[:, None, None].to(tl.int64) * stride_vn + zr_off, qr * inv_r[:, None, None], mask=b_mask[:, None, None])

    if DROP_DR_TERM:
        tl.store(Coeff_ptr + b_offs, tl.zeros([BLOCK_B], dtype=tl.float32), mask=b_mask)
    else:
        qu_sum = tl.sum(qp * up, axis=1) + tl.sum(qo * uo, axis=1) + tl.sum(tl.sum(qr * ur, axis=2), axis=1)
        coeff = qu_sum * (inv_r * inv_r * inv_r) / K_DIM
        tl.store(Coeff_ptr + b_offs, coeff, mask=b_mask)


class FusedMHCCoeffFn(torch.autograd.Function):
    """Unified autograd Function supporting both `Trial 3` (`cublas` proj + Triton RMS) and `Trial 4` (`Single-Scan` proj+RMS)."""

    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        phi: torch.Tensor,
        gamma: torch.Tensor,
        alpha: torch.Tensor,
        b: torch.Tensor,
        eps: float = 1e-6,
        single_scan: bool = True,
        drop_dr_exploit: bool = False,
    ):
        N, n_s, C = X.shape
        K = n_s * C
        M = n_s * n_s + 2 * n_s
        x_flat = X.contiguous().view(N, K)
        A_cast = (gamma.float().unsqueeze(1) * phi.float()).to(x_flat.dtype).contiguous()

        inv_r = torch.empty((N,), device=X.device, dtype=torch.float32)
        if single_scan:
            u = torch.empty((N, M), device=X.device, dtype=torch.float32)
            m_pad = max(16, triton.next_power_of_2(M))
            blk_b = 32 if m_pad >= 64 else 64
            blk_k = 64 if m_pad >= 64 else 128
            _single_scan_proj_rms_fwd_kernel[(triton.cdiv(N, blk_b),)](
                x_flat, A_cast, u, inv_r,
                x_flat.stride(0), A_cast.stride(0), u.stride(0),
                N, K, eps,
                M_DIM=M, M_PAD=m_pad, BLOCK_B=blk_b, BLOCK_K=blk_k,
                num_warps=4, num_stages=3,
            )
        else:
            u = (x_flat @ A_cast).float().contiguous()
            _row_rms_inv_kernel[(N,)](
                x_flat, inv_r, x_flat.stride(0), K, eps, BLOCK_K=2048, num_warps=8
            )

        hp = torch.empty((N, n_s), device=X.device, dtype=torch.float32)
        ho = torch.empty((N, n_s), device=X.device, dtype=torch.float32)
        hr = torch.empty((N, n_s, n_s), device=X.device, dtype=torch.float32)
        z_save = torch.empty((N, M), device=X.device, dtype=torch.float32)
        blk_ep = 16 if n_s == 2 else (8 if n_s == 4 else 2)
        alpha_f = alpha.detach().float().contiguous()
        b_f = b.detach().float().contiguous()
        _mhc_epilogue_log_sinkhorn20_fwd_kernel[(triton.cdiv(N, blk_ep),)](
            u, inv_r, alpha_f, b_f, hp, ho, hr, z_save,
            u.stride(0), z_save.stride(0), N,
            N_STREAM=n_s, BLOCK_B=blk_ep, num_warps=4,
        )

        ctx.save_for_backward(x_flat, phi, gamma, alpha_f, A_cast, u, inv_r, hr, z_save)
        ctx.shape_meta = (N, n_s, C, K, M)
        ctx.alpha_dtype = alpha.dtype
        ctx.b_dtype = b.dtype
        ctx.single_scan = single_scan
        ctx.drop_dr_exploit = drop_dr_exploit
        return hp, ho, hr

    @staticmethod
    def backward(ctx, dhp: torch.Tensor, dho: torch.Tensor, dhr: torch.Tensor):
        x_flat, phi, gamma, alpha_f, A_cast, u, inv_r, hr, z_save = ctx.saved_tensors
        N, n_s, C, K, M = ctx.shape_meta

        dhp_c = dhp.contiguous().float()
        dho_c = dho.contiguous().float()
        dhr_c = dhr.contiguous().float()

        blk_ep = 16 if n_s == 2 else (8 if n_s == 4 else 2)
        n_ep_grid = triton.cdiv(N, blk_ep)
        V = torch.empty((N, M), device=x_flat.device, dtype=torch.float32)
        coeff = torch.empty((N,), device=x_flat.device, dtype=torch.float32)
        dz = torch.empty((N, M), device=x_flat.device, dtype=torch.float32)
        part_alpha = torch.empty((n_ep_grid, 3), device=x_flat.device, dtype=torch.float32)

        _mhc_epilogue_log_sinkhorn20_bwd_ckpt5_kernel[(n_ep_grid,)](
            z_save, u, inv_r, alpha_f, hr,
            dhp_c, dho_c, dhr_c,
            V, coeff, dz, part_alpha,
            z_save.stride(0), u.stride(0), V.stride(0), N, K,
            DROP_DR_TERM=ctx.drop_dr_exploit,
            N_STREAM=n_s, BLOCK_B=blk_ep, num_warps=4,
        )

        d_alpha = part_alpha.sum(dim=0).to(ctx.alpha_dtype)
        d_b = dz.sum(dim=0).to(ctx.b_dtype)

        # Preserve full dynamic range when alpha is small (e.g., alpha = 0.01) by column-scaling V before BF16 Tensor-Core GEMM!
        v_amax = V.abs().amax(dim=0, keepdim=True).clamp_min(1e-30)
        v_scaled_bf16 = (V / v_amax).to(x_flat.dtype)
        dA = (x_flat.t() @ v_scaled_bf16).float() * v_amax

        d_phi = (gamma.float().unsqueeze(1) * dA).to(phi.dtype)
        d_gamma = torch.sum(dA * phi.float(), dim=1).to(gamma.dtype)

        dx_flat = torch.empty_like(x_flat)
        if ctx.single_scan:
            AT_contig = A_cast.t().contiguous()
            m_pad = max(16, triton.next_power_of_2(M))
            blk_b = 32 if m_pad >= 64 else 64
            blk_k = 64 if m_pad >= 64 else 128
            v_glob = float(V.abs().max().clamp_min(1e-30).item())
            V_norm = V / v_glob
            _single_scan_dx_norm_proj_bwd_kernel[(triton.cdiv(N, blk_b),)](
                x_flat, V_norm, AT_contig, coeff, dx_flat,
                x_flat.stride(0), V_norm.stride(0), AT_contig.stride(0), dx_flat.stride(0),
                N, K, v_glob, M_DIM=M, M_PAD=m_pad, BLOCK_B=blk_b, BLOCK_K=blk_k,
                num_warps=4, num_stages=3,
            )
        else:
            v_glob = V.abs().max().clamp_min(1e-30)
            dx_proj = ((V / v_glob).to(x_flat.dtype) @ A_cast.t()).float() * v_glob
            dx_flat = (dx_proj - x_flat.float() * coeff.unsqueeze(1)).to(x_flat.dtype)

        return dx_flat.view(N, n_s, C), d_phi, d_gamma, d_alpha, d_b, None, None, None


def b_dtype_matches(t: torch.Tensor) -> bool:
    return False


def fused_mhc_coeff(
    X: torch.Tensor,
    phi: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-6,
    single_scan: bool = True,
    drop_dr_exploit: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hp, ho, hr = FusedMHCCoeffFn.apply(X, phi, gamma, alpha, b, eps, single_scan, drop_dr_exploit)
    return hp, ho, hr


# ============================================================================
# 2. Track 2 SM80 Triton Kernels: DeepSeek 2-Kernel vs RSI Superfused Sandwich
# ============================================================================


@triton.jit
def _deepseek_fpre_fwd_kernel(
    X_ptr, Hp_ptr, HIn_ptr,
    stride_xb, stride_xs, stride_hn,
    C_DIM: tl.constexpr, N_STREAM: tl.constexpr, BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_DIM
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for i in tl.static_range(N_STREAM):
        hp_i = tl.load(Hp_ptr + b * N_STREAM + i).to(tl.float32)
        x_i = tl.load(X_ptr + b * stride_xb + i * stride_xs + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        acc += hp_i * x_i
    tl.store(HIn_ptr + b * stride_hn + c_offs, acc.to(HIn_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _sublayer_rmsnorm_fwd_kernel(
    HIn_ptr, W_ptr, HNorm_ptr, Rstd_ptr,
    stride_hn, C_DIM: tl.constexpr, BLOCK_C: tl.constexpr, eps: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_DIM
    h = tl.load(HIn_ptr + b * stride_hn + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(h * h, axis=0) / C_DIM + eps)
    tl.store(Rstd_ptr + b, rstd)
    w = tl.load(W_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(HNorm_ptr + b * stride_hn + c_offs, (h * rstd * w).to(HNorm_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _superfused_pre_rmsnorm_fwd_kernel(
    X_ptr, Hp_ptr, W_ptr, HNorm_ptr, Rstd_ptr,
    stride_xb, stride_xs, stride_hn,
    ROUND_BF16: tl.constexpr,
    C_DIM: tl.constexpr, N_STREAM: tl.constexpr, BLOCK_C: tl.constexpr, eps: tl.constexpr,
):
    """Superfuses `F_pre` readout + `SublayerRMSNorm` (`0` HBM bytes for `h_in`), preserving BF16 boundary."""
    b = tl.program_id(0).to(tl.int64)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_DIM

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for i in tl.static_range(N_STREAM):
        hp_i = tl.load(Hp_ptr + b * N_STREAM + i).to(tl.float32)
        x_i = tl.load(X_ptr + b * stride_xb + i * stride_xs + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        acc += hp_i * x_i

    if ROUND_BF16:
        h_val = tl.where(c_mask, acc.to(tl.bfloat16).to(tl.float32), 0.0)
    else:
        h_val = tl.where(c_mask, acc, 0.0)

    rstd = tl.rsqrt(tl.sum(h_val * h_val, axis=0) / C_DIM + eps)
    tl.store(Rstd_ptr + b, rstd)
    w = tl.load(W_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(HNorm_ptr + b * stride_hn + c_offs, (h_val * rstd * w).to(HNorm_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _fused_post_res_fwd_kernel(
    X_ptr, Y_ptr, Ho_ptr, Hr_ptr, XOut_ptr,
    stride_xb, stride_xs, stride_yb, stride_ob, stride_os,
    C_DIM, N_STREAM: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Fused `X_{l+1}[b, i, c] = sum_j H_res[b, i, j]*X_l[b, j, c] + H_post[b, i]*y_l[b, c]` (`(n+1)C` reads, `nC` writes)."""
    b = tl.program_id(0).to(tl.int64)
    c_tile = tl.program_id(1)
    c_offs = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_DIM
    s_offs = tl.arange(0, N_STREAM)

    # Load X_l[b, 0..n-1, c_offs] -> [N_STREAM, BLOCK_C] in FP32
    x_mat = tl.load(
        X_ptr + b * stride_xb + s_offs[:, None].to(tl.int64) * stride_xs + c_offs[None, :],
        mask=c_mask[None, :], other=0.0,
    ).to(tl.float32)
    y_vec = tl.load(Y_ptr + b * stride_yb + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    for i in tl.static_range(N_STREAM):
        ho_i = tl.load(Ho_ptr + b * N_STREAM + i).to(tl.float32)
        hr_row = tl.load(Hr_ptr + (b * N_STREAM + i) * N_STREAM + s_offs).to(tl.float32)
        out_i = tl.sum(hr_row[:, None] * x_mat, axis=0) + ho_i * y_vec
        tl.store(
            XOut_ptr + b * stride_ob + i * stride_os + c_offs,
            out_i.to(XOut_ptr.dtype.element_ty), mask=c_mask,
        )


@triton.jit
def _separate_post_res_bwd_kernel(
    X_ptr, Y_ptr, Ho_ptr, Hr_ptr, dXNext_ptr,
    dXRes_ptr, dY_ptr, dHo_ptr, dHr_ptr,
    stride_xb, stride_xs, stride_yb,
    C_DIM: tl.constexpr, N_STREAM: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """DeepSeek Trial-2 separate `F_{post,res}` backward (`1` CTA per token row)."""
    b = tl.program_id(0).to(tl.int64)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_DIM
    s_offs = tl.arange(0, N_STREAM)

    x_mat = tl.load(
        X_ptr + b * stride_xb + s_offs[:, None].to(tl.int64) * stride_xs + c_offs[None, :],
        mask=c_mask[None, :], other=0.0,
    ).to(tl.float32)
    dxn_mat = tl.load(
        dXNext_ptr + b * stride_xb + s_offs[:, None].to(tl.int64) * stride_xs + c_offs[None, :],
        mask=c_mask[None, :], other=0.0,
    ).to(tl.float32)
    y_vec = tl.load(Y_ptr + b * stride_yb + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    ho_vec = tl.load(Ho_ptr + b * N_STREAM + s_offs).to(tl.float32)

    dy_vec = tl.sum(ho_vec[:, None] * dxn_mat, axis=0)
    tl.store(dY_ptr + b * stride_yb + c_offs, dy_vec.to(dY_ptr.dtype.element_ty), mask=c_mask)

    dho_vec = tl.sum(dxn_mat * y_vec[None, :], axis=1)
    tl.store(dHo_ptr + b * N_STREAM + s_offs, dho_vec)

    for i in tl.static_range(N_STREAM):
        dxn_i = tl.sum(
            tl.where(s_offs[:, None] == i, dxn_mat, 0.0), axis=0
        )
        dhr_row = tl.sum(x_mat * dxn_i[None, :], axis=1)
        tl.store(dHr_ptr + (b * N_STREAM + i) * N_STREAM + s_offs, dhr_row)

    for j in tl.static_range(N_STREAM):
        hr_col = tl.load(Hr_ptr + (b * N_STREAM + s_offs) * N_STREAM + j).to(tl.float32)
        dx_j = tl.sum(hr_col[:, None] * dxn_mat, axis=0)
        tl.store(dXRes_ptr + b * stride_xb + j * stride_xs + c_offs, dx_j.to(dXRes_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _separate_pre_rmsnorm_bwd_split_kernel(
    HIn_ptr, X_ptr, Hp_ptr, W_ptr, Rstd_ptr, dHNorm_ptr, dXRes_ptr,
    dXOut_ptr, dHp_ptr, PartW_ptr,
    stride_xb, stride_xs, stride_hn,
    N_TOK, ROWS_PER_CTA: tl.constexpr,
    C_DIM: tl.constexpr, N_STREAM: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """DeepSeek Trial-2 separate `SublayerRMSNorm + F_pre` backward (requires reading `HIn` + `dXRes` + `X` and re-writing `dXOut`!)."""
    sc_id = tl.program_id(0).to(tl.int64)
    s_row = sc_id * ROWS_PER_CTA
    e_row = tl.minimum(N_TOK, s_row + ROWS_PER_CTA)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_DIM

    w = tl.load(W_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for b in range(s_row, e_row):
        hin = tl.load(HIn_ptr + b * stride_hn + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        dhn = tl.load(dHNorm_ptr + b * stride_hn + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + b).to(tl.float32)
        dw_acc += dhn * hin * rstd

        dyw = tl.where(c_mask, dhn * w, 0.0)
        c_dot = tl.sum(dyw * hin, axis=0) * (rstd * rstd / C_DIM)
        dhin = rstd * (dyw - hin * c_dot)

        for j in tl.static_range(N_STREAM):
            hp_j = tl.load(Hp_ptr + b * N_STREAM + j).to(tl.float32)
            xj = tl.load(X_ptr + b * stride_xb + j * stride_xs + c_offs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(dHp_ptr + b * N_STREAM + j, tl.sum(dhin * xj, axis=0))
            dxr = tl.load(dXRes_ptr + b * stride_xb + j * stride_xs + c_offs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(
                dXOut_ptr + b * stride_xb + j * stride_xs + c_offs,
                (dxr + hp_j * dhin).to(dXOut_ptr.dtype.element_ty), mask=c_mask,
            )

    tl.store(PartW_ptr + sc_id * stride_hn + c_offs, dw_acc, mask=c_mask)


@triton.jit
def _rsi_unified_sandwich_bwd_split_kernel(
    X_ptr, Y_ptr, Hp_ptr, Ho_ptr, Hr_ptr, W_ptr, Rstd_ptr,
    dHNorm_ptr, dXNext_ptr,
    dXOut_ptr, dY_ptr, dHp_ptr, dHo_ptr, dHr_ptr, PartW_ptr,
    stride_xb, stride_xs, stride_yb, stride_dxnb, stride_dxns, stride_dxob, stride_dxos, stride_hn,
    N_TOK, ROUND_BF16: tl.constexpr, ROWS_PER_CTA: tl.constexpr,
    C_DIM: tl.constexpr, N_STREAM: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Round-3 RSI Winner (`Trial 3` `KEEP`):
    Single-read/single-write Sublayer-Sandwich Backward with Split-CTA lock-free `dw_sub` reduction (`0` atomics, `0` spills).
    Reads `X_l` ONCE and writes `dX_l = H_res^T @ dX_{l+1} + H_pre^T * dh_in` ONCE, saving `3 * N * n * C` HBM elements!
    """
    sc_id = tl.program_id(0).to(tl.int64)
    s_row = sc_id * ROWS_PER_CTA
    e_row = tl.minimum(N_TOK, s_row + ROWS_PER_CTA)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_DIM
    s_offs = tl.arange(0, N_STREAM)

    w = tl.load(W_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for b in range(s_row, e_row):
        # 1) Load X_l[b, 0..n-1, :] and dX_{l+1}[b, 0..n-1, :] ONCE into registers
        x_mat = tl.load(
            X_ptr + b * stride_xb + s_offs[:, None].to(tl.int64) * stride_xs + c_offs[None, :],
            mask=c_mask[None, :], other=0.0,
        ).to(tl.float32)
        dxn_mat = tl.load(
            dXNext_ptr + b * stride_dxnb + s_offs[:, None].to(tl.int64) * stride_dxns + c_offs[None, :],
            mask=c_mask[None, :], other=0.0,
        ).to(tl.float32)

        hp_vec = tl.load(Hp_ptr + b * N_STREAM + s_offs).to(tl.float32)
        ho_vec = tl.load(Ho_ptr + b * N_STREAM + s_offs).to(tl.float32)

        # 2) Reconstruct exact BF16-rounded h_in on-chip (`0` HBM bytes!) & compute SublayerRMSNorm VJP
        h_acc = tl.sum(hp_vec[:, None] * x_mat, axis=0)
        if ROUND_BF16:
            h_val = tl.where(c_mask, h_acc.to(tl.bfloat16).to(tl.float32), 0.0)
        else:
            h_val = tl.where(c_mask, h_acc, 0.0)

        dhn = tl.load(dHNorm_ptr + b * stride_hn + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + b).to(tl.float32)
        dw_acc += dhn * h_val * rstd

        dyw = tl.where(c_mask, dhn * w, 0.0)
        c_dot = tl.sum(dyw * h_val, axis=0) * (rstd * rstd / C_DIM)
        dhin = rstd * (dyw - h_val * c_dot)

        dhp_vec = tl.sum(x_mat * dhin[None, :], axis=1)
        tl.store(dHp_ptr + b * N_STREAM + s_offs, dhp_vec)

        # 3) Post/Res VJP: dy, dH_post, dH_res, and single-write combined dX_l
        y_vec = tl.load(Y_ptr + b * stride_yb + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        dy_vec = tl.sum(ho_vec[:, None] * dxn_mat, axis=0)
        tl.store(dY_ptr + b * stride_yb + c_offs, dy_vec.to(dY_ptr.dtype.element_ty), mask=c_mask)

        dho_vec = tl.sum(dxn_mat * y_vec[None, :], axis=1)
        tl.store(dHo_ptr + b * N_STREAM + s_offs, dho_vec)

        for i in tl.static_range(N_STREAM):
            dxn_i = tl.sum(tl.where(s_offs[:, None] == i, dxn_mat, 0.0), axis=0)
            dhr_row = tl.sum(x_mat * dxn_i[None, :], axis=1)
            tl.store(dHr_ptr + (b * N_STREAM + i) * N_STREAM + s_offs, dhr_row)

        for j in tl.static_range(N_STREAM):
            hr_col = tl.load(Hr_ptr + (b * N_STREAM + s_offs) * N_STREAM + j).to(tl.float32)
            hp_j = tl.sum(tl.where(s_offs == j, hp_vec, 0.0), axis=0)
            dx_j = tl.sum(hr_col[:, None] * dxn_mat, axis=0) + hp_j * dhin
            tl.store(dXOut_ptr + b * stride_dxob + j * stride_dxos + c_offs, dx_j.to(dXOut_ptr.dtype.element_ty), mask=c_mask)

    tl.store(PartW_ptr + sc_id * stride_hn + c_offs, dw_acc, mask=c_mask)


class DeepSeek2KernelStreamSandwichFn(torch.autograd.Function):
    """Trial 2: Literal DeepSeek Section 4.3.1 split (`F_pre` -> separate `SublayerRMSNorm` -> fused `F_{post,res}`)."""

    @staticmethod
    def forward(ctx, X, y, h_pre, h_post, h_res, w_sub, eps: float = 1e-6):
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        hp_c = h_pre.contiguous().float()
        ho_c = h_post.contiguous().float()
        hr_c = h_res.contiguous().float()
        y_c = y.contiguous()
        w_c = w_sub.contiguous()

        h_in = torch.empty((N, C), device=X.device, dtype=X.dtype)
        h_norm = torch.empty((N, C), device=X.device, dtype=X.dtype)
        rstd = torch.empty((N,), device=X.device, dtype=torch.float32)
        X_next = torch.empty((N, n_s, C), device=X.device, dtype=X.dtype)

        _deepseek_fpre_fwd_kernel[(N,)](
            X, hp_c, h_in, X.stride(0), X.stride(1), h_in.stride(0),
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, num_warps=8,
        )
        _sublayer_rmsnorm_fwd_kernel[(N,)](
            h_in, w_c, h_norm, rstd, h_in.stride(0),
            C_DIM=C, BLOCK_C=block_c, eps=eps, num_warps=8,
        )
        c_tile = 512 if C >= 1024 else block_c
        _fused_post_res_fwd_kernel[(N, triton.cdiv(C, c_tile))](
            X, y_c, ho_c, hr_c, X_next,
            X.stride(0), X.stride(1), y_c.stride(0), X_next.stride(0), X_next.stride(1),
            C, N_STREAM=n_s, BLOCK_C=c_tile, num_warps=4,
        )
        ctx.save_for_backward(X, y_c, hp_c, ho_c, hr_c, w_c, h_in, rstd)
        return h_norm, X_next

    @staticmethod
    def backward(ctx, dh_norm, dX_next):
        X, y_c, hp_c, ho_c, hr_c, w_c, h_in, rstd = ctx.saved_tensors
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        Xc = X.contiguous()
        dhn_c = dh_norm.contiguous()
        dxn_c = dX_next.contiguous()

        dx_res = torch.empty_like(Xc)
        dy = torch.empty_like(y_c)
        dho = torch.empty_like(ho_c)
        dhr = torch.empty_like(hr_c)
        _separate_post_res_bwd_kernel[(N,)](
            Xc, y_c, ho_c, hr_c, dxn_c,
            dx_res, dy, dho, dhr,
            Xc.stride(0), Xc.stride(1), y_c.stride(0),
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, num_warps=8,
        )

        rows_per_cta = 16
        n_sc = triton.cdiv(N, rows_per_cta)
        dx_out = torch.empty_like(Xc)
        dhp = torch.empty_like(hp_c)
        part_w = torch.empty((n_sc, C), device=X.device, dtype=torch.float32)
        _separate_pre_rmsnorm_bwd_split_kernel[(n_sc,)](
            h_in, Xc, hp_c, w_c, rstd, dhn_c, dx_res,
            dx_out, dhp, part_w,
            Xc.stride(0), Xc.stride(1), h_in.stride(0),
            N, ROWS_PER_CTA=rows_per_cta,
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, num_warps=8,
        )
        dw_sub = part_w.sum(dim=0).to(w_c.dtype)
        return dx_out, dy, dhp.to(h_pre_dtype(hp_c)), dho, dhr, dw_sub, None


def h_pre_dtype(t: torch.Tensor) -> torch.dtype:
    return t.dtype


class RSISuperfusedStreamSandwichFn(torch.autograd.Function):
    """Trial 3 (`KEEP`): Superfused `Pre + SublayerRMSNorm` (`0` HBM bytes for `h_in`) + Unified Single-Read/Write Sandwich Bwd."""

    @staticmethod
    def forward(ctx, X, y, h_pre, h_post, h_res, w_sub, eps: float = 1e-6):
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        hp_c = h_pre.contiguous().float()
        ho_c = h_post.contiguous().float()
        hr_c = h_res.contiguous().float()
        y_c = y.contiguous()
        w_c = w_sub.contiguous()
        round_bf16 = X.dtype == torch.bfloat16

        h_norm = torch.empty((N, C), device=X.device, dtype=X.dtype)
        rstd = torch.empty((N,), device=X.device, dtype=torch.float32)
        X_next = torch.empty((N, n_s, C), device=X.device, dtype=X.dtype)

        _superfused_pre_rmsnorm_fwd_kernel[(N,)](
            X, hp_c, w_c, h_norm, rstd,
            X.stride(0), X.stride(1), h_norm.stride(0),
            ROUND_BF16=round_bf16,
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, eps=eps, num_warps=8,
        )
        c_tile = 512 if C >= 1024 else block_c
        _fused_post_res_fwd_kernel[(N, triton.cdiv(C, c_tile))](
            X, y_c, ho_c, hr_c, X_next,
            X.stride(0), X.stride(1), y_c.stride(0), X_next.stride(0), X_next.stride(1),
            C, N_STREAM=n_s, BLOCK_C=c_tile, num_warps=4,
        )
        ctx.save_for_backward(X, y_c, hp_c, ho_c, hr_c, w_c, rstd)
        ctx.round_bf16 = round_bf16
        return h_norm, X_next

    @staticmethod
    def backward(ctx, dh_norm, dX_next):
        X, y_c, hp_c, ho_c, hr_c, w_c, rstd = ctx.saved_tensors
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        dhn_c = dh_norm.contiguous()

        rows_per_cta = 16
        n_sc = triton.cdiv(N, rows_per_cta)
        dx_out = torch.empty((N, n_s, C), device=X.device, dtype=X.dtype)
        dy = torch.empty_like(y_c)
        dhp = torch.empty_like(hp_c)
        dho = torch.empty_like(ho_c)
        dhr = torch.empty_like(hr_c)
        part_w = torch.empty((n_sc, C), device=X.device, dtype=torch.float32)

        _rsi_unified_sandwich_bwd_split_kernel[(n_sc,)](
            X, y_c, hp_c, ho_c, hr_c, w_c, rstd,
            dhn_c, dX_next,
            dx_out, dy, dhp, dho, dhr, part_w,
            X.stride(0), X.stride(1), y_c.stride(0),
            dX_next.stride(0), dX_next.stride(1),
            dx_out.stride(0), dx_out.stride(1),
            dhn_c.stride(0),
            N, ROUND_BF16=ctx.round_bf16, ROWS_PER_CTA=rows_per_cta,
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, num_warps=8,
        )
        dw_sub = part_w.sum(dim=0).to(w_c.dtype)
        return dx_out, dy, dhp, dho, dhr, dw_sub, None


# Standalone fast Split-PreRMSNorm & FusedPostRes functions used in modular sublayer stacks (`y = F_l(h_norm)`)
class SuperfusedPreRMSNormFn(torch.autograd.Function):
    """Modular pre-sublayer `X_l, H_pre, w_sub -> h_norm` (`0` HBM bytes for `h_in`) when `y = F_l(h_norm)` runs between Pre & Post."""

    @staticmethod
    def forward(ctx, X: torch.Tensor, h_pre: torch.Tensor, w_sub: torch.Tensor, eps: float = 1e-6):
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        hp_c = h_pre.contiguous().float()
        w_c = w_sub.contiguous()
        round_bf16 = X.dtype == torch.bfloat16
        h_norm = torch.empty((N, C), device=X.device, dtype=X.dtype)
        rstd = torch.empty((N,), device=X.device, dtype=torch.float32)
        _superfused_pre_rmsnorm_fwd_kernel[(N,)](
            X, hp_c, w_c, h_norm, rstd,
            X.stride(0), X.stride(1), h_norm.stride(0),
            ROUND_BF16=round_bf16, C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, eps=eps, num_warps=8,
        )
        ctx.save_for_backward(X, hp_c, w_c, rstd)
        ctx.round_bf16 = round_bf16
        return h_norm

    @staticmethod
    def backward(ctx, dh_norm: torch.Tensor):
        X, hp_c, w_c, rstd = ctx.saved_tensors
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        dhn_c = dh_norm.contiguous()
        rows_per_cta = 16
        n_sc = triton.cdiv(N, rows_per_cta)
        dx = torch.empty((N, n_s, C), device=X.device, dtype=X.dtype)
        dhp = torch.empty_like(hp_c)
        part_w = torch.empty((n_sc, C), device=X.device, dtype=torch.float32)
        _modular_pre_rmsnorm_bwd_split_kernel[(n_sc,)](
            X, hp_c, w_c, rstd, dhn_c, dx, dhp, part_w,
            X.stride(0), X.stride(1), dx.stride(0), dx.stride(1), dhn_c.stride(0),
            N, ROUND_BF16=ctx.round_bf16, ROWS_PER_CTA=rows_per_cta,
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, num_warps=8,
        )
        return dx, dhp, part_w.sum(dim=0).to(w_c.dtype), None


@triton.jit
def _modular_pre_rmsnorm_bwd_split_kernel(
    X_ptr, Hp_ptr, W_ptr, Rstd_ptr, dHNorm_ptr,
    dXOut_ptr, dHp_ptr, PartW_ptr,
    stride_xb, stride_xs, stride_dxob, stride_dxos, stride_hn,
    N_TOK, ROUND_BF16: tl.constexpr, ROWS_PER_CTA: tl.constexpr,
    C_DIM: tl.constexpr, N_STREAM: tl.constexpr, BLOCK_C: tl.constexpr,
):
    sc_id = tl.program_id(0).to(tl.int64)
    s_row = sc_id * ROWS_PER_CTA
    e_row = tl.minimum(N_TOK, s_row + ROWS_PER_CTA)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_DIM
    s_offs = tl.arange(0, N_STREAM)
    w = tl.load(W_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for b in range(s_row, e_row):
        x_mat = tl.load(
            X_ptr + b * stride_xb + s_offs[:, None].to(tl.int64) * stride_xs + c_offs[None, :],
            mask=c_mask[None, :], other=0.0,
        ).to(tl.float32)
        hp_vec = tl.load(Hp_ptr + b * N_STREAM + s_offs).to(tl.float32)
        h_acc = tl.sum(hp_vec[:, None] * x_mat, axis=0)
        h_val = tl.where(c_mask, h_acc.to(tl.bfloat16).to(tl.float32) if ROUND_BF16 else h_acc, 0.0)
        dhn = tl.load(dHNorm_ptr + b * stride_hn + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + b).to(tl.float32)
        dw_acc += dhn * h_val * rstd
        dyw = tl.where(c_mask, dhn * w, 0.0)
        c_dot = tl.sum(dyw * h_val, axis=0) * (rstd * rstd / C_DIM)
        dhin = rstd * (dyw - h_val * c_dot)
        tl.store(dHp_ptr + b * N_STREAM + s_offs, tl.sum(x_mat * dhin[None, :], axis=1))
        for j in tl.static_range(N_STREAM):
            hp_j = tl.sum(tl.where(s_offs == j, hp_vec, 0.0), axis=0)
            tl.store(dXOut_ptr + b * stride_dxob + j * stride_dxos + c_offs, (hp_j * dhin).to(dXOut_ptr.dtype.element_ty), mask=c_mask)

    tl.store(PartW_ptr + sc_id * stride_hn + c_offs, dw_acc, mask=c_mask)


class FusedPostResMergeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, y: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor):
        N, n_s, C = X.shape
        ho_c = h_post.contiguous().float()
        hr_c = h_res.contiguous().float()
        y_c = y.contiguous()
        X_next = torch.empty((N, n_s, C), device=X.device, dtype=X.dtype)
        c_tile = 512 if C >= 1024 else triton.next_power_of_2(C)
        _fused_post_res_fwd_kernel[(N, triton.cdiv(C, c_tile))](
            X, y_c, ho_c, hr_c, X_next,
            X.stride(0), X.stride(1), y_c.stride(0), X_next.stride(0), X_next.stride(1),
            C, N_STREAM=n_s, BLOCK_C=c_tile, num_warps=4,
        )
        ctx.save_for_backward(X, y_c, ho_c, hr_c)
        return X_next

    @staticmethod
    def backward(ctx, dX_next: torch.Tensor):
        X, y_c, ho_c, hr_c = ctx.saved_tensors
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        Xc = X.contiguous()
        dxn_c = dX_next.contiguous()
        dx = torch.empty_like(Xc)
        dy = torch.empty_like(y_c)
        dho = torch.empty_like(ho_c)
        dhr = torch.empty_like(hr_c)
        _separate_post_res_bwd_kernel[(N,)](
            Xc, y_c, ho_c, hr_c, dxn_c, dx, dy, dho, dhr,
            Xc.stride(0), Xc.stride(1), y_c.stride(0),
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, num_warps=8,
        )
        return dx, dy, dho, dhr


# ============================================================================
# 3. Selective `L_r^*` Stream-Only Checkpointing (`mHC-Stream-Only-SAC`, Eq. 20)
# ============================================================================


class MHCStreamSegmentBank:
    """Implements DeepSeek Section 4.3.2 Table 3 / Eq. 20 selective stream-only recomputation (`L_r` stride).

    For each block of `L_r` consecutive sublayers `[l0, l0 + L_r - 1]`:
    - Forward stores ONLY the segment anchor `X_{l0} [N, n, C]` (`nC` elements) and per-sublayer compact
      `(y_l [N, C], h_post_l [N, n], h_res_l [N, n, n])` (discarding all `(L_r - 1)` intermediate `[N, n, C]`
      stream states `X_{l0+1 .. l0+L_r-1}`).
    - When backward enters the top of segment `seg_id`, `materialize_segment()` replays ONLY `_fused_post_res_fwd_kernel`
      across `0 .. len-2` (`0` extra Attention/MLP FLOPs and `0` extra `FSDP2` `AllGather` collectives!), and each
      `pop_step()` frees transient `[N, n, C]` tensors immediately after its sublayer backward completes.
    """

    def __init__(self, L_r: int) -> None:
        self.L_r = max(1, int(L_r))
        self.segments: list[dict[str, Any]] = []

    def record_sublayer(
        self,
        sublayer_idx: int,
        X_in: torch.Tensor,
        y_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> tuple[int, int]:
        seg_id = sublayer_idx // self.L_r
        offset = sublayer_idx % self.L_r
        if offset == 0:
            self.segments.append(
                {
                    "anchor_X": X_in.detach().contiguous(),
                    "steps": [],
                    "reconstructed": None,
                }
            )
        seg = self.segments[seg_id]
        seg["steps"].append(
            (
                y_out.detach().contiguous(),
                h_post.detach().contiguous().float(),
                h_res.detach().contiguous().float(),
            )
        )
        return seg_id, offset

    def get_reconstructed_X(self, seg_id: int, offset: int, release_ref: bool = False) -> torch.Tensor:
        seg = self.segments[seg_id]
        if self.L_r == 1:
            out = seg["anchor_X"]
            if release_ref:
                seg["anchor_X"] = None
                seg["steps"] = []
            return out

        if seg["reconstructed"] is None:
            steps = seg["steps"]
            L_seg = len(steps)
            xs: list[torch.Tensor | None] = [None] * L_seg
            cur = seg["anchor_X"]
            xs[0] = cur
            N, n_s, C = cur.shape
            c_tile = 512 if C >= 1024 else triton.next_power_of_2(C)
            for k in range(L_seg - 1):
                yk, hok, hrk = steps[k]
                nxt = torch.empty_like(cur)
                _fused_post_res_fwd_kernel[(N, triton.cdiv(C, c_tile))](
                    cur, yk, hok, hrk, nxt,
                    cur.stride(0), cur.stride(1), yk.stride(0), nxt.stride(0), nxt.stride(1),
                    C, N_STREAM=n_s, BLOCK_C=c_tile, num_warps=4,
                )
                xs[k + 1] = nxt
                cur = nxt
            # Materialize Table 3 / Eq. 20 transient segment norm/readout workspace `(2 * C * L_seg)` across active `L_r` block
            seg["transient_norm_ws"] = torch.empty((L_seg, 2, N, C), device=cur.device, dtype=cur.dtype)
            seg["reconstructed"] = xs

        out = seg["reconstructed"][offset]
        assert out is not None
        if release_ref:
            seg["reconstructed"][offset] = None
            if offset == 0:
                seg["anchor_X"] = None
                seg["steps"] = []
                seg["transient_norm_ws"] = None
                seg["reconstructed"] = None
        return out


class SACFusedMHCCoeffFn(torch.autograd.Function):
    """Stream-checkpointed counterpart of `FusedMHCCoeffFn` that does NOT retain `[N, nC]` `X` for non-anchor layers."""

    @staticmethod
    def forward(ctx, X, phi, gamma, alpha, b, bank: MHCStreamSegmentBank, seg_id: int, offset: int, eps: float = 1e-6):
        N, n_s, C = X.shape
        K = n_s * C
        M = n_s * n_s + 2 * n_s
        x_flat = X.contiguous().view(N, K)
        A_cast = (gamma.float().unsqueeze(1) * phi.float()).to(x_flat.dtype).contiguous()
        u = torch.empty((N, M), device=X.device, dtype=torch.float32)
        inv_r = torch.empty((N,), device=X.device, dtype=torch.float32)
        m_pad = max(16, triton.next_power_of_2(M))
        blk_b = 32 if m_pad >= 64 else 64
        blk_k = 64 if m_pad >= 64 else 128
        _single_scan_proj_rms_fwd_kernel[(triton.cdiv(N, blk_b),)](
            x_flat, A_cast, u, inv_r,
            x_flat.stride(0), A_cast.stride(0), u.stride(0),
            N, K, eps, M_DIM=M, M_PAD=m_pad, BLOCK_B=blk_b, BLOCK_K=blk_k, num_warps=4, num_stages=3,
        )
        hp = torch.empty((N, n_s), device=X.device, dtype=torch.float32)
        ho = torch.empty((N, n_s), device=X.device, dtype=torch.float32)
        hr = torch.empty((N, n_s, n_s), device=X.device, dtype=torch.float32)
        z_save = torch.empty((N, M), device=X.device, dtype=torch.float32)
        blk_ep = 16 if n_s == 2 else (8 if n_s == 4 else 2)
        alpha_f = alpha.detach().float().contiguous()
        b_f = b.detach().float().contiguous()
        _mhc_epilogue_log_sinkhorn20_fwd_kernel[(triton.cdiv(N, blk_ep),)](
            u, inv_r, alpha_f, b_f, hp, ho, hr, z_save,
            u.stride(0), z_save.stride(0), N, N_STREAM=n_s, BLOCK_B=blk_ep, num_warps=4,
        )
        # Notice: x_flat is NOT saved here! Only compact `[N, M]` / `[nC, M]` tensors are saved!
        ctx.save_for_backward(phi, gamma, alpha_f, A_cast, u, inv_r, hr, z_save)
        ctx.bank = bank
        ctx.seg_id = seg_id
        ctx.offset = offset
        ctx.alpha_dtype = alpha.dtype
        ctx.b_dtype = b.dtype
        ctx.shape_meta = (N, n_s, C, K, M)
        return hp, ho, hr

    @staticmethod
    def backward(ctx, dhp, dho, dhr):
        phi, gamma, alpha_f, A_cast, u, inv_r, hr, z_save = ctx.saved_tensors
        N, n_s, C, K, M = ctx.shape_meta
        X_rec = ctx.bank.get_reconstructed_X(ctx.seg_id, ctx.offset, release_ref=True)
        x_flat = X_rec.contiguous().view(N, K)

        blk_ep = 16 if n_s == 2 else (8 if n_s == 4 else 2)
        n_ep_grid = triton.cdiv(N, blk_ep)
        V = torch.empty((N, M), device=x_flat.device, dtype=torch.float32)
        coeff = torch.empty((N,), device=x_flat.device, dtype=torch.float32)
        dz = torch.empty((N, M), device=x_flat.device, dtype=torch.float32)
        part_alpha = torch.empty((n_ep_grid, 3), device=x_flat.device, dtype=torch.float32)

        _mhc_epilogue_log_sinkhorn20_bwd_ckpt5_kernel[(n_ep_grid,)](
            z_save, u, inv_r, alpha_f, hr,
            dhp.contiguous().float(), dho.contiguous().float(), dhr.contiguous().float(),
            V, coeff, dz, part_alpha,
            z_save.stride(0), u.stride(0), V.stride(0), N, K,
            DROP_DR_TERM=False, N_STREAM=n_s, BLOCK_B=blk_ep, num_warps=4,
        )
        d_alpha = part_alpha.sum(dim=0).to(ctx.alpha_dtype)
        d_b = dz.sum(dim=0).to(ctx.b_dtype)
        v_amax = V.abs().amax(dim=0, keepdim=True).clamp_min(1e-30)
        dA = (x_flat.t() @ (V / v_amax).to(x_flat.dtype)).float() * v_amax
        d_phi = (gamma.float().unsqueeze(1) * dA).to(phi.dtype)
        d_gamma = torch.sum(dA * phi.float(), dim=1).to(gamma.dtype)

        dx_flat = torch.empty_like(x_flat)
        AT_contig = A_cast.t().contiguous()
        m_pad = max(16, triton.next_power_of_2(M))
        blk_b = 32 if m_pad >= 64 else 64
        blk_k = 64 if m_pad >= 64 else 128
        v_glob = float(V.abs().max().clamp_min(1e-30).item())
        V_norm = V / v_glob
        _single_scan_dx_norm_proj_bwd_kernel[(triton.cdiv(N, blk_b),)](
            x_flat, V_norm, AT_contig, coeff, dx_flat,
            x_flat.stride(0), V_norm.stride(0), AT_contig.stride(0), dx_flat.stride(0),
            N, K, v_glob, M_DIM=M, M_PAD=m_pad, BLOCK_B=blk_b, BLOCK_K=blk_k, num_warps=4, num_stages=3,
        )
        return dx_flat.view(N, n_s, C), d_phi, d_gamma, d_alpha, d_b, None, None, None, None


class SACSuperfusedPreRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, h_pre, w_sub, bank: MHCStreamSegmentBank, seg_id: int, offset: int, eps: float = 1e-6):
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        hp_c = h_pre.contiguous().float()
        w_c = w_sub.contiguous()
        round_bf16 = X.dtype == torch.bfloat16
        h_norm = torch.empty((N, C), device=X.device, dtype=X.dtype)
        rstd = torch.empty((N,), device=X.device, dtype=torch.float32)
        _superfused_pre_rmsnorm_fwd_kernel[(N,)](
            X, hp_c, w_c, h_norm, rstd,
            X.stride(0), X.stride(1), h_norm.stride(0),
            ROUND_BF16=round_bf16, C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, eps=eps, num_warps=8,
        )
        # Do NOT save X!
        ctx.save_for_backward(hp_c, w_c, rstd)
        ctx.bank = bank
        ctx.seg_id = seg_id
        ctx.offset = offset
        ctx.round_bf16 = round_bf16
        return h_norm

    @staticmethod
    def backward(ctx, dh_norm):
        hp_c, w_c, rstd = ctx.saved_tensors
        X = ctx.bank.get_reconstructed_X(ctx.seg_id, ctx.offset, release_ref=False)
        N, n_s, C = X.shape
        block_c = triton.next_power_of_2(C)
        dhn_c = dh_norm.contiguous()
        rows_per_cta = 16
        n_sc = triton.cdiv(N, rows_per_cta)
        dx = torch.empty((N, n_s, C), device=X.device, dtype=X.dtype)
        dhp = torch.empty_like(hp_c)
        part_w = torch.empty((n_sc, C), device=X.device, dtype=torch.float32)
        _modular_pre_rmsnorm_bwd_split_kernel[(n_sc,)](
            X, hp_c, w_c, rstd, dhn_c, dx, dhp, part_w,
            X.stride(0), X.stride(1), dx.stride(0), dx.stride(1), dhn_c.stride(0),
            N, ROUND_BF16=ctx.round_bf16, ROWS_PER_CTA=rows_per_cta,
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, num_warps=8,
        )
        return dx, dhp, part_w.sum(dim=0).to(w_c.dtype), None, None, None, None


class SACFusedPostResMergeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, y, h_post, h_res, bank: MHCStreamSegmentBank, sublayer_idx: int):
        N, n_s, C = X.shape
        ho_c = h_post.contiguous().float()
        hr_c = h_res.contiguous().float()
        y_c = y.contiguous()
        seg_id, offset = bank.record_sublayer(sublayer_idx, X, y_c, ho_c, hr_c)
        X_next = torch.empty((N, n_s, C), device=X.device, dtype=X.dtype)
        c_tile = 512 if C >= 1024 else triton.next_power_of_2(C)
        _fused_post_res_fwd_kernel[(N, triton.cdiv(C, c_tile))](
            X, y_c, ho_c, hr_c, X_next,
            X.stride(0), X.stride(1), y_c.stride(0), X_next.stride(0), X_next.stride(1),
            C, N_STREAM=n_s, BLOCK_C=c_tile, num_warps=4,
        )
        # Do NOT save X! Only compact `[N, C]` y_c and tiny routing coeffs are saved!
        ctx.save_for_backward(y_c, ho_c, hr_c)
        ctx.bank = bank
        ctx.seg_id = seg_id
        ctx.offset = offset
        return X_next

    @staticmethod
    def backward(ctx, dX_next):
        y_c, ho_c, hr_c = ctx.saved_tensors
        Xc = ctx.bank.get_reconstructed_X(ctx.seg_id, ctx.offset, release_ref=False)
        N, n_s, C = Xc.shape
        block_c = triton.next_power_of_2(C)
        dxn_c = dX_next.contiguous()
        dx = torch.empty_like(Xc)
        dy = torch.empty_like(y_c)
        dho = torch.empty_like(ho_c)
        dhr = torch.empty_like(hr_c)
        _separate_post_res_bwd_kernel[(N,)](
            Xc, y_c, ho_c, hr_c, dxn_c, dx, dy, dho, dhr,
            Xc.stride(0), Xc.stride(1), y_c.stride(0),
            C_DIM=C, N_STREAM=n_s, BLOCK_C=block_c, num_warps=8,
        )
        return dx, dy, dho, dhr, None, None


def inspect_mhc_cubin_spills() -> dict[str, Any]:
    """Extracts compiled SM80 cubin register & spill counts across all promoted Round-3 mHC Triton kernels."""
    kernels = {
        "single_scan_proj_rms_fwd": _single_scan_proj_rms_fwd_kernel,
        "single_scan_dx_norm_proj_bwd": _single_scan_dx_norm_proj_bwd_kernel,
        "epilogue_log_sinkhorn20_fwd": _mhc_epilogue_log_sinkhorn20_fwd_kernel,
        "epilogue_log_sinkhorn20_bwd_ckpt5": _mhc_epilogue_log_sinkhorn20_bwd_ckpt5_kernel,
        "superfused_pre_rmsnorm_fwd": _superfused_pre_rmsnorm_fwd_kernel,
        "fused_post_res_fwd": _fused_post_res_fwd_kernel,
        "rsi_unified_sandwich_bwd_split": _rsi_unified_sandwich_bwd_split_kernel,
    }
    out: dict[str, Any] = {}
    max_spills = 0
    for kname, kobj in kernels.items():
        spills: list[int] = []
        regs: list[int] = []
        for dev_entry in getattr(kobj, "device_caches", {}).values():
            cd = dev_entry[0] if isinstance(dev_entry, tuple) else dev_entry
            for bin_v in cd.values():
                s = int(getattr(bin_v, "n_spills", 0))
                r = int(getattr(bin_v, "n_regs", 0))
                spills.append(s)
                regs.append(r)
                max_spills = max(max_spills, s)
        out[kname] = {"n_spills": spills, "n_regs": regs}
    out["max_winner_cubin_spills"] = max_spills
    return out
