"""Round-5 SM80 Triton, Checkpointed Tensor-Core Muon-NS5, & Distributed CP=8 Kernels for E131 (`v122`):
Nonlinear Fast-Weight Test-Time Training (`LaCT: Test-Time Training Done Right`,
Zhang et al., ICLR 2026, `arXiv:2505.23884`, official repo `a1600012888/LaCT`).

Key Mathematical & Hardware Components:
1. Independent `FP64` Analytical Oracles (`fp64_lact_chunk_grad_oracle`, `fp64_muon_ns5_oracle`,
   `fp64_lact_multichunk_trajectory_oracle`) for shifted block-causal `SwiGLU` fast-weight
   Apply + Inner Gradient + Second-Order `Fwd-of-Bwd` + Online Momentum + Newton-Schulz-5 `Muon`
   + Row-wise `L2` Weight Normalization (`Post-Norm` & `Pre-Norm`).
2. Ground-Truth Reproductions of Official `a1600012888/LaCT` Bugs & SM80 Spills:
   - Joint concatenated `NS5(cat([dw0, dw2], dim=1))` polar cross-contamination bug
     (`ttt_operation_fused_kernel.py` lines 164 & 268) vs exact independent `[3*B_h]` batched `Muon`.
   - Zero-scale `b_y / b_tgt_scale` `0/0 = NaN` & `sqrt(||w||^2 + eps)` vs `||w|| + eps` drift
     (`l2norm_triton_kernels.py` lines 82 & 153) vs division-free `L2Norm` backward.
   - `1,456`-word SM80 cubin register spills & `229,376 B > 166,912 B` SRAM `OutOfResources` in
     official 3-accumulator `_swiglu_three_bmm_with_lr_kernel` vs `0` cubin spills across all
     Round-5 RSI Triton kernels.
3. Round-5 RSI Kernels (`0` cubin spills, `0` global `tl.atomic_add`):
   - `FusedLaCTSwiGLUApplyFn`: Batched `cuBLAS` `[B_h, 2H, D] @ [B_h, D, C]` + `~26`-reg `0`-spill
     coalesced Triton `SwiGLU` forward & backward epilogues.
   - `FusedLaCTFastWeightGradFn`: Batched `cuBLAS` `(Y0_Y2, dH)` + `27`-reg `0`-spill forward
     epilogue + **100% Lock-Free (`0` `tl.atomic_add`) Split-`M` Second-Order `Fwd-of-Bwd`
     Hessian-Vector Triton Kernel** (`SiLU''(y0) = sig*(1-sig)*(2 + y0*(1 - 2*sig))`) supporting
     arbitrary `H < D`, `H == D`, `H > D`, and ragged non-multiple shapes.
   - `BatchedCheckpointedMuonNS5Fn`: Orientation-preserving `[3*B_h, R, K]` (`R <= K`) batched
     Tensor-Core Newton-Schulz-5 storing **only** `G_stacked` (`1` tensor vs `16` autograd
     intermediates/chunk) with exact analytical Fréchet VJP matching `b*A + (c*A) @ A` op order
     and `FP32` pre-normalization.
   - `FusedMomentumL2NormUpdateFn` & `ChunkCheckpointedLaCTSwiGLUFn`: Division-free `Momentum +
     L2Norm` state transition + **Chunk-Boundary Fast-Weight State Checkpointing** (`O(N_c * B_h * H * D)`
     boundary states instead of `O(N_c * B_h * H * C)` token activations).
   - `SubChunkCPLaCTSWALayer`: 8-GPU NVSwitch Within-Chunk Context Parallelism (`CP = 8`,
     `arXiv:2505.23884` Sec. 3.4 / Alg. 2–3) with 1-hop P2P boundary `SWA` KV halo, exact
     single-`AllReduce` replicated-adjoint backward (`Gate G9`), and dual-stream `SWA` / `LaCT`
     communication-compute overlap.
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

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


NS5_COEFFS: Tuple[Tuple[float, float, float], ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


# =============================================================================
# 1. Independent FP64 Reference Oracles & Official Repo Bug Reproductions
# =============================================================================


def zeropower_via_newtonschulz5_eager_ref(
    G: torch.Tensor, eps: float = 1e-7, ns5_fp32: bool = False
) -> torch.Tensor:
    """Exact official `a1600012888/LaCT` `zeropower_via_newtonschulz5` (`ttt_operation.py` lines 29–64).
    Preserves `FP64` when `G.dtype == torch.float64` so `Gate G5`/`G6` oracles are exact;
    setting `ns5_fp32=True` keeps FP32 precision so eager/compiled baselines satisfy `Gate G5` (`< 1.5e-2`).
    """
    assert G.ndim == 3
    if G.dtype == torch.float64:
        X = G
    elif ns5_fp32:
        X = G.float()
    else:
        X = G.bfloat16()
    trans = G.size(1) > G.size(2)
    if trans:
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    for a, b, c in NS5_COEFFS:
        A = X @ X.transpose(1, 2)
        B_mat = b * A + (c * A) @ A
        X = a * X + B_mat @ X
    if trans:
        X = X.transpose(1, 2)
    return X


def fp64_muon_ns5_oracle(
    G: torch.Tensor, dOut: torch.Tensor, eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        g = G.detach().to(torch.float64).requires_grad_(True)
        do = dOut.detach().to(torch.float64)
        y = zeropower_via_newtonschulz5_eager_ref(g, eps=eps)
        y.backward(do)
        assert g.grad is not None
        return y.detach(), g.grad.detach()


def eager_lact_swiglu_apply_ref(
    W0_W2: torch.Tensor, W1: torch.Tensor, Q: torch.Tensor
) -> torch.Tensor:
    """Reference Apply: O = (W1 @ [SiLU(W0 @ Q^T) * (W2 @ Q^T)])^T -> [B, N, D]."""
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.bmm(W0, Q.transpose(1, 2))
    Y2 = torch.bmm(W2, Q.transpose(1, 2))
    H_q = F.silu(Y0) * Y2
    return torch.bmm(W1, H_q).transpose(1, 2)


def eager_lact_fw_grad_ref(
    W0_W2: torch.Tensor,
    W1: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference Inner Fast-Weight Gradient (`lact_fw_grad.py` `reference_lact_swiglu_ffn_fast_weight_grads`).
    Supports arbitrary `W0_W2: [B, 2H, D]`, `W1: [B, D, H]`, `K, V: [B, N, D]`, `lr*: [B, N]`.
    """
    orig_dt = K.dtype
    compute_dt = torch.float64 if orig_dt == torch.float64 else torch.float32
    W0, W2 = W0_W2.chunk(2, dim=1)
    if orig_dt == torch.float64:
        Y0 = torch.bmm(W0, K.transpose(1, 2))
        Y2 = torch.bmm(W2, K.transpose(1, 2))
        dH = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))
    else:
        Y0 = torch.bmm(W0.to(orig_dt), K.transpose(1, 2)).to(compute_dt)
        Y2 = torch.bmm(W2.to(orig_dt), K.transpose(1, 2)).to(compute_dt)
        dH = torch.bmm(W1.to(orig_dt).transpose(1, 2), V.transpose(1, 2)).to(compute_dt)

    sig = torch.sigmoid(Y0)
    silu_y0 = Y0 * sig
    s_bp = sig * (1.0 + Y0 * (1.0 - sig))

    lr0_u = lr0.to(compute_dt).unsqueeze(1)
    lr1_u = lr1.to(compute_dt).unsqueeze(1)
    lr2_u = lr2.to(compute_dt).unsqueeze(1)

    dy0_lr = (dH * Y2 * s_bp * lr0_u).to(orig_dt)
    dy2_lr = (dH * silu_y0 * lr2_u).to(orig_dt)
    hid_lr = (Y2 * silu_y0 * lr1_u).to(orig_dt)

    dy02_lr = torch.cat([dy0_lr, dy2_lr], dim=1)
    dW0_W2 = torch.bmm(dy02_lr, K)
    dW1 = torch.bmm(V.transpose(1, 2), hid_lr.transpose(1, 2))
    return dW0_W2, dW1


def fp64_lact_chunk_grad_oracle(
    W0_W2: torch.Tensor,
    W1: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    g_dW0_W2: torch.Tensor,
    g_dW1: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    with torch.enable_grad():
        w02 = W0_W2.detach().to(torch.float64).requires_grad_(True)
        w1 = W1.detach().to(torch.float64).requires_grad_(True)
        k = K.detach().to(torch.float64).requires_grad_(True)
        v = V.detach().to(torch.float64).requires_grad_(True)
        l0 = lr0.detach().to(torch.float64).requires_grad_(True)
        l1 = lr1.detach().to(torch.float64).requires_grad_(True)
        l2 = lr2.detach().to(torch.float64).requires_grad_(True)

        dw02, dw1 = eager_lact_fw_grad_ref(w02, w1, k, v, l0, l1, l2)
        loss = (dw02 * g_dW0_W2.to(torch.float64)).sum() + (dw1 * g_dW1.to(torch.float64)).sum()
        loss.backward()
    return {
        "dW0_W2": dw02.detach(),
        "dW1": dw1.detach(),
        "grad_W0_W2": w02.grad.detach(),
        "grad_W1": w1.grad.detach(),
        "grad_K": k.grad.detach(),
        "grad_V": v.grad.detach(),
        "grad_lr0": l0.grad.detach(),
        "grad_lr1": l1.grad.detach(),
        "grad_lr2": l2.grad.detach(),
    }


def eager_block_causal_lact_swiglu_ref(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = True,
    momentum: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    concat_muon_bug: bool = False,
    normalize_fast_weights: bool = True,
    eps: float = 1e-5,
    ns5_fp32: bool = False,
) -> torch.Tensor:
    """Faithful reference for `ttt_operation.py` (`block_causal_lact_swiglu` / `prenorm_block_causal_lact_swiglu`).
    Preserves `FP64` when `w0.dtype == torch.float64` and `q.dtype == torch.float64`.
    Setting `concat_muon_bug=True` reproduces `ttt_operation_fused_kernel.py` lines 164 & 268.
    """
    is_fp64 = w0.dtype == torch.float64 and q.dtype == torch.float64
    mm_dt = torch.float64 if is_fp64 else torch.bfloat16

    w0_norm = torch.sqrt((w0.double() * w0.double()).sum(dim=2, keepdim=True) + 1e-24).to(w0.dtype) if is_fp64 else torch.sqrt((w0.float() * w0.float()).sum(dim=2, keepdim=True) + 1e-24).to(w0.dtype)
    w1_norm = torch.sqrt((w1.double() * w1.double()).sum(dim=2, keepdim=True) + 1e-24).to(w1.dtype) if is_fp64 else torch.sqrt((w1.float() * w1.float()).sum(dim=2, keepdim=True) + 1e-24).to(w1.dtype)
    w2_norm = torch.sqrt((w2.double() * w2.double()).sum(dim=2, keepdim=True) + 1e-24).to(w2.dtype) if is_fp64 else torch.sqrt((w2.float() * w2.float()).sum(dim=2, keepdim=True) + 1e-24).to(w2.dtype)

    w0_main, w1_main, w2_main = w0, w1, w2
    w0_cur, w1_cur, w2_cur = w0, w1, w2

    if momentum is not None:
        m0 = torch.zeros_like(w0_main)
        m1 = torch.zeros_like(w1_main)
        m2 = torch.zeros_like(w2_main)
    else:
        m0 = m1 = m2 = None

    B, seq_len, _ = q.shape
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    outs: List[torch.Tensor] = []

    for c_idx in range(num_chunks):
        s = c_idx * chunk_size
        e = min(seq_len, s + chunk_size)
        qi = q[:, s:e, :].to(mm_dt)
        w0_mm = w0_cur.to(mm_dt)
        w1_mm = w1_cur.to(mm_dt)
        w2_mm = w2_cur.to(mm_dt)

        # Apply
        y0_q = torch.bmm(w0_mm, qi.transpose(1, 2))
        y2_q = torch.bmm(w2_mm, qi.transpose(1, 2))
        if is_fp64:
            hq = F.silu(y0_q) * y2_q
        else:
            hq = (F.silu(y0_q.float()) * y2_q.float()).to(mm_dt)
        oi = torch.bmm(w1_mm, hq).transpose(1, 2)
        outs.append(oi)

        if c_idx == num_chunks - 1:
            break

        ki = k[:, s:e, :].to(mm_dt)
        vi = v[:, s:e, :].to(mm_dt)
        lr0i = lr0[:, s:e, 0] if lr0.ndim == 3 else lr0[:, s:e]
        lr1i = lr1[:, s:e, 0] if lr1.ndim == 3 else lr1[:, s:e]
        lr2i = lr2[:, s:e, 0] if lr2.ndim == 3 else lr2[:, s:e]

        w02_mm = torch.cat([w0_mm, w2_mm], dim=1)
        dw02, dw1 = eager_lact_fw_grad_ref(w02_mm, w1_mm, ki, vi, lr0i, lr1i, lr2i)
        dw0, dw2 = dw02.chunk(2, dim=1)

        dw0_s = dw0.to(w0_main.dtype)
        dw1_s = dw1.to(w1_main.dtype)
        dw2_s = dw2.to(w2_main.dtype)

        if momentum is not None:
            mi = momentum[:, s:e, :].mean(dim=1, keepdim=True)  # [B, 1, 1]
            m0 = m0 * mi + dw0_s
            m1 = m1 * mi + dw1_s
            m2 = m2 * mi + dw2_s
            u0_in, u1_in, u2_in = m0, m1, m2
        else:
            u0_in, u1_in, u2_in = dw0_s, dw1_s, dw2_s

        if use_muon:
            u1 = zeropower_via_newtonschulz5_eager_ref(u1_in, ns5_fp32=ns5_fp32).to(w1_main.dtype)
            if concat_muon_bug:
                u02 = zeropower_via_newtonschulz5_eager_ref(
                    torch.cat([u0_in, u2_in], dim=1), ns5_fp32=ns5_fp32
                ).to(w0_main.dtype)
                u0, u2 = u02.chunk(2, dim=1)
            else:
                u0 = zeropower_via_newtonschulz5_eager_ref(u0_in, ns5_fp32=ns5_fp32).to(w0_main.dtype)
                u2 = zeropower_via_newtonschulz5_eager_ref(u2_in, ns5_fp32=ns5_fp32).to(w2_main.dtype)
        else:
            u0, u1, u2 = u0_in, u1_in, u2_in

        if prenorm:
            w0_main = w0_main + u0
            w1_main = w1_main + u1
            w2_main = w2_main + u2
            if normalize_fast_weights:
                n0 = torch.sqrt((w0_main * w0_main).sum(dim=2, keepdim=True) + 1e-24) + eps
                n1 = torch.sqrt((w1_main * w1_main).sum(dim=2, keepdim=True) + 1e-24) + eps
                n2 = torch.sqrt((w2_main * w2_main).sum(dim=2, keepdim=True) + 1e-24) + eps
                w0_cur = (w0_main / n0) * w0_norm
                w1_cur = (w1_main / n1) * w1_norm
                w2_cur = (w2_main / n2) * w2_norm
            else:
                w0_cur, w1_cur, w2_cur = w0_main, w1_main, w2_main
        else:
            w0_cand = w0_cur + u0
            w1_cand = w1_cur + u1
            w2_cand = w2_cur + u2
            if normalize_fast_weights:
                n0 = torch.sqrt((w0_cand * w0_cand).sum(dim=2, keepdim=True) + 1e-24) + eps
                n1 = torch.sqrt((w1_cand * w1_cand).sum(dim=2, keepdim=True) + 1e-24) + eps
                n2 = torch.sqrt((w2_cand * w2_cand).sum(dim=2, keepdim=True) + 1e-24) + eps
                w0_cur = (w0_cand / n0) * w0_norm
                w1_cur = (w1_cand / n1) * w1_norm
                w2_cur = (w2_cand / n2) * w2_norm
            else:
                w0_cur, w1_cur, w2_cur = w0_cand, w1_cand, w2_cand

    return torch.cat(outs, dim=1)


# =============================================================================
# 2. Zero-Spill Coalesced Triton Kernels for SwiGLU Apply & Second-Order Fwd-of-Bwd
# =============================================================================


@triton.jit
def _lact_swiglu_apply_epilogue_kernel(
    y02_ptr,
    hid_ptr,
    B,
    H,
    N,
    BLOCK_N: tl.constexpr,
):
    """Computes Hidden[b, h, n] = SiLU(Y0[b, h, n]) * Y2[b, h, n] from [B, 2H, N] Y0_Y2 (`~24` regs, `0` spills)."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mn = offs_n < N
    y0 = tl.load(y02_ptr + (pid_b * 2 * H + pid_h) * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    y2 = tl.load(y02_ptr + (pid_b * 2 * H + H + pid_h) * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    sig = tl.sigmoid(y0)
    hid = (y0 * sig) * y2
    tl.store(hid_ptr + (pid_b * H + pid_h) * N + offs_n, hid.to(hid_ptr.dtype.element_ty), mask=mn)


@triton.jit
def _lact_swiglu_apply_bwd_epilogue_kernel(
    y02_ptr,
    dh_ptr,
    dy02_ptr,
    hid_ptr,
    B,
    H,
    N,
    BLOCK_N: tl.constexpr,
):
    """Computes DY0_DY2[b, 2H, n] and Hidden[b, H, n] from Y0_Y2 and dH = W1^T @ dOut^T (`~26` regs, `0` spills)."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mn = offs_n < N
    y0 = tl.load(y02_ptr + (pid_b * 2 * H + pid_h) * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    y2 = tl.load(y02_ptr + (pid_b * 2 * H + H + pid_h) * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + (pid_b * H + pid_h) * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    sig = tl.sigmoid(y0)
    silu = y0 * sig
    s_bp = sig * (1.0 + y0 * (1.0 - sig))
    hid = silu * y2
    dy0 = dh * y2 * s_bp
    dy2 = dh * silu
    out_dt = dy02_ptr.dtype.element_ty
    tl.store(dy02_ptr + (pid_b * 2 * H + pid_h) * N + offs_n, dy0.to(out_dt), mask=mn)
    tl.store(dy02_ptr + (pid_b * 2 * H + H + pid_h) * N + offs_n, dy2.to(out_dt), mask=mn)
    tl.store(hid_ptr + (pid_b * H + pid_h) * N + offs_n, hid.to(out_dt), mask=mn)


@triton.jit
def _lact_fw_grad_fwd_epilogue_kernel(
    y02_ptr,
    dh_ptr,
    lr0_ptr,
    lr1_ptr,
    lr2_ptr,
    dy02_ptr,
    hid_ptr,
    B,
    H,
    N,
    BLOCK_N: tl.constexpr,
):
    """Single-pass 27-reg 0-spill inner-gradient epilogue scaling DY0*lr0, DY2*lr2, Hidden*lr1."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mn = offs_n < N
    y0 = tl.load(y02_ptr + (pid_b * 2 * H + pid_h) * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    y2 = tl.load(y02_ptr + (pid_b * 2 * H + H + pid_h) * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + (pid_b * H + pid_h) * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    lr0 = tl.load(lr0_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    lr1 = tl.load(lr1_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    lr2 = tl.load(lr2_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    sig = tl.sigmoid(y0)
    silu = y0 * sig
    s_bp = sig * (1.0 + y0 * (1.0 - sig))
    hid = silu * y2 * lr1
    dy0 = dh * y2 * s_bp * lr0
    dy2 = dh * silu * lr2
    out_dt = dy02_ptr.dtype.element_ty
    tl.store(dy02_ptr + (pid_b * 2 * H + pid_h) * N + offs_n, dy0.to(out_dt), mask=mn)
    tl.store(dy02_ptr + (pid_b * 2 * H + H + pid_h) * N + offs_n, dy2.to(out_dt), mask=mn)
    tl.store(hid_ptr + (pid_b * H + pid_h) * N + offs_n, hid.to(out_dt), mask=mn)


@triton.jit
def _lact_fw_grad_bwd2_splitm_kernel(
    dh_ptr,
    y02_ptr,
    lr0_ptr,
    lr1_ptr,
    lr2_ptr,
    g_dy02_ptr,
    g_hid_lr1_ptr,
    out_gdh_ptr,
    out_gy02_ptr,
    part_glr_ptr,  # [num_splits, 3, B, N] fp32
    out_dy02_ptr,
    out_hid_lr1_ptr,
    B,
    H,
    N,
    CHUNK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """100% Lock-Free (`0` `tl.atomic_add`), `0`-Spill Split-`H` Second-Order `Fwd-of-Bwd` Kernel.
    Each CTA owns `(pid_n, pid_b, pid_sh)` covering `CHUNK_H` hidden channels and `BLOCK_N` tokens,
    computing exact `SiLU''` Hessian-vector terms and writing partial `(glr0, glr1, glr2)` to `part_glr_ptr`.
    Works identically across `H < D`, `H == D`, `H > D`, and non-divisible `H` or `N`!
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_sh = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mn = offs_n < N

    lr0 = tl.load(lr0_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    lr1 = tl.load(lr1_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)
    lr2 = tl.load(lr2_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)

    acc_glr0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_glr1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_glr2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    h_start = pid_sh * CHUNK_H
    h_end = tl.minimum(H, h_start + CHUNK_H)
    base_h = pid_b * H * N
    base_2h = pid_b * 2 * H * N
    out_dt = out_gdh_ptr.dtype.element_ty

    for h in range(h_start, h_end):
        off_h = base_h + h * N + offs_n
        off_y0 = base_2h + h * N + offs_n
        off_y2 = base_2h + (H + h) * N + offs_n

        dh = tl.load(dh_ptr + off_h, mask=mn, other=0.0).to(tl.float32)
        y0 = tl.load(y02_ptr + off_y0, mask=mn, other=0.0).to(tl.float32)
        y2 = tl.load(y02_ptr + off_y2, mask=mn, other=0.0).to(tl.float32)
        g_dy0 = tl.load(g_dy02_ptr + off_y0, mask=mn, other=0.0).to(tl.float32)
        g_dy2 = tl.load(g_dy02_ptr + off_y2, mask=mn, other=0.0).to(tl.float32)
        g_hlr1 = tl.load(g_hid_lr1_ptr + off_h, mask=mn, other=0.0).to(tl.float32)

        sig = tl.sigmoid(y0)
        om_sig = 1.0 - sig
        silu = y0 * sig
        s_bp = sig * (1.0 + y0 * om_sig)
        d2_silu = sig * om_sig * (2.0 + y0 * (1.0 - 2.0 * sig))

        g_dh = g_dy0 * lr0 * y2 * s_bp + g_dy2 * lr2 * silu
        g_y2 = g_dy0 * lr0 * dh * s_bp + g_hlr1 * lr1 * silu
        g_y0 = (g_dy2 * lr2 * dh + g_hlr1 * lr1 * y2) * s_bp + (g_dy0 * lr0 * dh * y2) * d2_silu

        acc_glr0 += g_dy0 * dh * y2 * s_bp
        acc_glr1 += g_hlr1 * y2 * silu
        acc_glr2 += g_dy2 * dh * silu

        dy0 = dh * y2 * s_bp * lr0
        dy2 = silu * dh * lr2
        hlr1 = lr1 * y2 * silu

        tl.store(out_gdh_ptr + off_h, g_dh.to(out_dt), mask=mn)
        tl.store(out_gy02_ptr + off_y0, g_y0.to(out_dt), mask=mn)
        tl.store(out_gy02_ptr + off_y2, g_y2.to(out_dt), mask=mn)
        tl.store(out_dy02_ptr + off_y0, dy0.to(out_dt), mask=mn)
        tl.store(out_dy02_ptr + off_y2, dy2.to(out_dt), mask=mn)
        tl.store(out_hid_lr1_ptr + off_h, hlr1.to(out_dt), mask=mn)

    stride_split = 3 * B * N
    stride_kind = B * N
    base_part = pid_sh * stride_split + pid_b * N + offs_n
    tl.store(part_glr_ptr + base_part + 0 * stride_kind, acc_glr0, mask=mn)
    tl.store(part_glr_ptr + base_part + 1 * stride_kind, acc_glr1, mask=mn)
    tl.store(part_glr_ptr + base_part + 2 * stride_kind, acc_glr2, mask=mn)


@triton.jit
def _reduce_splitm_glr_kernel(
    part_glr_ptr,  # [num_splits, 3, B, N] fp32
    glr0_ptr,
    glr1_ptr,
    glr2_ptr,
    num_splits,
    total_bn,
    BLOCK_BN: tl.constexpr,
):
    """Deterministic `0`-spill reduction of `part_glr_ptr` across `num_splits` into `(glr0, glr1, glr2)`."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_BN + tl.arange(0, BLOCK_BN)
    mask = offs < total_bn
    acc0 = tl.zeros((BLOCK_BN,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_BN,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_BN,), dtype=tl.float32)
    stride_split = 3 * total_bn
    for s in range(0, num_splits):
        base = s * stride_split + offs
        acc0 += tl.load(part_glr_ptr + base, mask=mask, other=0.0)
        acc1 += tl.load(part_glr_ptr + base + total_bn, mask=mask, other=0.0)
        acc2 += tl.load(part_glr_ptr + base + 2 * total_bn, mask=mask, other=0.0)
    out_dt = glr0_ptr.dtype.element_ty
    tl.store(glr0_ptr + offs, acc0.to(out_dt), mask=mask)
    tl.store(glr1_ptr + offs, acc1.to(out_dt), mask=mask)
    tl.store(glr2_ptr + offs, acc2.to(out_dt), mask=mask)


# =============================================================================
# 3. Division-Free Fused Momentum + Row-Wise L2 Weight Normalization Kernels
# =============================================================================


@triton.jit
def _fused_l2norm_update_fwd_kernel(
    w_base_ptr,     # [M_rows, D] fp32 or bf16
    u_step_ptr,     # [M_rows, D] fp32 or bf16 (Muon output or momentum/raw step)
    tgt_scale_ptr,  # [M_rows] fp32
    w_main_out_ptr, # [M_rows, D] (written only if WRITE_MAIN == 1)
    w_norm_out_ptr, # [M_rows, D] bf16 or fp32
    u_hat_out_ptr,  # [M_rows, D] fp32 unit-normalized vector (w_cand * inv_denom) -> zero division in bwd!
    inv_denom_ptr,  # [M_rows] fp32 1 / (||w_cand||_2 + eps)
    eps,
    M_rows,
    D,
    WRITE_MAIN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Division-free row-wise L2-norm update matching `w_cand / (||w_cand||_2 + eps) * tgt_scale` (`0` spills)."""
    row = tl.program_id(0)
    if row >= M_rows:
        return
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    off = row * D + cols

    wb = tl.load(w_base_ptr + off, mask=mask, other=0.0).to(tl.float32)
    us = tl.load(u_step_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tscale = tl.load(tgt_scale_ptr + row).to(tl.float32)

    wcand = wb + us
    if WRITE_MAIN:
        tl.store(w_main_out_ptr + off, wcand.to(w_main_out_ptr.dtype.element_ty), mask=mask)

    sq_sum = tl.sum(wcand * wcand, axis=0)
    norm_val = tl.sqrt(sq_sum + 1e-24)
    inv_denom = 1.0 / (norm_val + eps)
    u_hat = wcand * inv_denom
    w_out = u_hat * tscale

    tl.store(u_hat_out_ptr + off, u_hat, mask=mask)
    tl.store(inv_denom_ptr + row, inv_denom)
    tl.store(w_norm_out_ptr + off, w_out.to(w_norm_out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_l2norm_update_bwd_kernel(
    dw_norm_ptr,    # [M_rows, D]
    dw_main_ptr,    # [M_rows, D] (read only if HAS_DMAIN == 1)
    u_hat_ptr,      # [M_rows, D] fp32
    inv_denom_ptr,  # [M_rows] fp32
    tgt_scale_ptr,  # [M_rows] fp32
    eps,
    grad_w_base_ptr,
    grad_u_step_ptr,
    grad_tscale_ptr,
    M_rows,
    D,
    HAS_DMAIN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Exact division-free backward for `y = (wcand / (||wcand||_2 + eps)) * tgt_scale`.
    Note: Let `r = ||wcand||_2`, `denom = r + eps`, `u_hat = wcand / denom`, so `r * inv_denom = 1 - eps * inv_denom`.
    Wait: for exact derivative wrt `wcand` when `eps > 0`:
      d(wcand / (r + eps)) = dw / denom - wcand * (dw . wcand) / (r * denom^2)
                           = inv_denom * [dw - u_hat * (dw . (wcand / r))].
    Since `u_hat = wcand / denom` and `denom / r = 1 / (1 - eps * inv_denom)` when `r > 0`,
    `u_unit = wcand / r`: we compute `u_hat_norm_sq = sum(u_hat^2) = r^2 / denom^2`, and
    `dot_unit_uhat = sum(dw * u_hat) / max(sqrt(u_hat_norm_sq), 1e-12) * sqrt(u_hat_norm_sq)`?
    Even cleaner & 100% division-free: `wcand / r = u_hat / sqrt(sum(u_hat^2) + 1e-30)`, so
    `dw_cand = inv_denom * tscale * (dw_norm - u_hat * (sum(dw_norm * u_hat) / tl.maximum(tl.sqrt(tl.sum(u_hat * u_hat, axis=0)), 1e-12)))`!
    Wait: check algebra! `u_unit = wcand / r`, `u_hat = (r / denom) * u_unit`.
    `(dw . wcand) / (r * denom^2) * wcand = inv_denom * (dw . u_unit) * u_hat`!
    YES! Because `||u_hat||_2 = r / denom`, `u_unit = u_hat / ||u_hat||_2` whenever `r > 0` (and `u_hat == 0` when `r == 0`!).
    """
    row = tl.program_id(0)
    if row >= M_rows:
        return
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    off = row * D + cols

    dwn = tl.load(dw_norm_ptr + off, mask=mask, other=0.0).to(tl.float32)
    u_hat = tl.load(u_hat_ptr + off, mask=mask, other=0.0).to(tl.float32)
    inv_d = tl.load(inv_denom_ptr + row).to(tl.float32)
    tscale = tl.load(tgt_scale_ptr + row).to(tl.float32)

    dot_uhat = tl.sum(dwn * u_hat, axis=0)
    tl.store(grad_tscale_ptr + row, dot_uhat.to(grad_tscale_ptr.dtype.element_ty))

    uhat_norm = tl.sqrt(tl.sum(u_hat * u_hat, axis=0))
    inv_uhat_norm = tl.where(uhat_norm > 1e-12, 1.0 / uhat_norm, 0.0)
    dot_uunit = dot_uhat * inv_uhat_norm

    g_wcand = inv_d * tscale * (dwn - u_hat * dot_uunit)
    if HAS_DMAIN:
        dwm = tl.load(dw_main_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g_wcand = g_wcand + dwm

    tl.store(grad_w_base_ptr + off, g_wcand.to(grad_w_base_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_u_step_ptr + off, g_wcand.to(grad_u_step_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# 4. Autograd Operators: Apply, Second-Order Fwd-of-Bwd, Batched Muon-NS5, L2Norm
# =============================================================================


def _run_lact_apply_fwd(
    W0_W2: torch.Tensor, W1: torch.Tensor, Q: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if Q.dtype == torch.float64 or W0_W2.dtype == torch.float64:
        res = eager_lact_swiglu_apply_ref(W0_W2, W1, Q)
        if out is not None:
            out.copy_(res)
            return out
        return res
    B, two_H, D = W0_W2.shape
    H = two_H // 2
    N = Q.shape[1]
    y02 = torch.bmm(W0_W2, Q.transpose(1, 2))
    hid = torch.empty((B, H, N), device=Q.device, dtype=Q.dtype)
    block_n = 256 if N >= 256 else 128
    _lact_swiglu_apply_epilogue_kernel[(triton.cdiv(N, block_n), B * H)](
        y02, hid, B, H, N, BLOCK_N=block_n, num_warps=4
    )
    if out is not None:
        return torch.bmm(hid.transpose(1, 2), W1.transpose(1, 2), out=out)
    return torch.bmm(hid.transpose(1, 2), W1.transpose(1, 2))


def _run_lact_apply_bwd(
    W0_W2: torch.Tensor,
    W1: torch.Tensor,
    Q: torch.Tensor,
    grad_out: torch.Tensor,
    w_grad_dtype: Optional[torch.dtype] = None,
    out_grad_Q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if Q.dtype == torch.float64 or W0_W2.dtype == torch.float64:
        with torch.enable_grad():
            w02 = W0_W2.detach().requires_grad_(True)
            w1 = W1.detach().requires_grad_(True)
            q = Q.detach().requires_grad_(True)
            o = eager_lact_swiglu_apply_ref(w02, w1, q)
            o.backward(grad_out)
            if out_grad_Q is not None:
                out_grad_Q.copy_(q.grad.detach())
                return w02.grad.detach(), w1.grad.detach(), out_grad_Q
            return w02.grad.detach(), w1.grad.detach(), q.grad.detach()

    B, two_H, D = W0_W2.shape
    H = two_H // 2
    N = Q.shape[1]
    go = grad_out if grad_out.stride(-1) == 1 else grad_out.contiguous()
    y02 = torch.bmm(W0_W2, Q.transpose(1, 2))
    dh = torch.bmm(W1.transpose(1, 2), go.transpose(1, 2))
    dy02 = torch.empty((B, two_H, N), device=Q.device, dtype=Q.dtype)
    hid = torch.empty((B, H, N), device=Q.device, dtype=Q.dtype)
    block_n = 256 if N >= 256 else 128
    _lact_swiglu_apply_bwd_epilogue_kernel[(triton.cdiv(N, block_n), B * H)](
        y02, dh, dy02, hid, B, H, N, BLOCK_N=block_n, num_warps=4
    )
    wg_dt = W0_W2.dtype if w_grad_dtype is None else w_grad_dtype
    if wg_dt == torch.float32 and Q.dtype == torch.bfloat16:
        grad_W1 = torch.bmm(go.transpose(1, 2), hid.transpose(1, 2), out_dtype=torch.float32)
        grad_W0_W2 = torch.bmm(dy02, Q, out_dtype=torch.float32)
    else:
        grad_W1 = torch.bmm(go.transpose(1, 2), hid.transpose(1, 2)).to(wg_dt)
        grad_W0_W2 = torch.bmm(dy02, Q).to(wg_dt)
    if out_grad_Q is not None:
        torch.bmm(dy02.transpose(1, 2), W0_W2, out=out_grad_Q)
        grad_Q = out_grad_Q
    else:
        grad_Q = torch.bmm(dy02.transpose(1, 2), W0_W2)
    return grad_W0_W2, grad_W1, grad_Q


class FusedLaCTSwiGLUApplyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W0_W2: torch.Tensor, W1: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        w02 = W0_W2.contiguous()
        w1 = W1.contiguous()
        q = Q.contiguous()
        ctx.save_for_backward(w02, w1, q)
        return _run_lact_apply_fwd(w02, w1, q)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        w02, w1, q = ctx.saved_tensors
        return _run_lact_apply_bwd(w02, w1, q, grad_out)


def _run_lact_fw_grad_fwd(
    W0_W2: torch.Tensor,
    W1: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if K.dtype == torch.float64 or W0_W2.dtype == torch.float64:
        return eager_lact_fw_grad_ref(W0_W2, W1, K, V, lr0, lr1, lr2)
    B, two_H, D = W0_W2.shape
    H = two_H // 2
    N = K.shape[1]
    y02 = torch.bmm(W0_W2, K.transpose(1, 2))
    dh = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))
    dy02 = torch.empty((B, two_H, N), device=K.device, dtype=K.dtype)
    hid = torch.empty((B, H, N), device=K.device, dtype=K.dtype)
    block_n = 256 if N >= 256 else 128
    _lact_fw_grad_fwd_epilogue_kernel[(triton.cdiv(N, block_n), B * H)](
        y02, dh, lr0, lr1, lr2, dy02, hid, B, H, N, BLOCK_N=block_n, num_warps=4
    )
    if K.dtype == torch.bfloat16:
        dW0_W2 = torch.bmm(dy02, K, out_dtype=torch.float32)
        dW1 = torch.bmm(V.transpose(1, 2), hid.transpose(1, 2), out_dtype=torch.float32)
    else:
        dW0_W2 = torch.bmm(dy02, K)
        dW1 = torch.bmm(V.transpose(1, 2), hid.transpose(1, 2))
    return dW0_W2, dW1


def _run_lact_fw_grad_bwd2(
    W0_W2: torch.Tensor,
    W1: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    grad_dW0_W2: torch.Tensor,
    grad_dW1: torch.Tensor,
    w_grad_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if K.dtype == torch.float64 or W0_W2.dtype == torch.float64:
        orc = fp64_lact_chunk_grad_oracle(W0_W2, W1, K, V, lr0, lr1, lr2, grad_dW0_W2, grad_dW1)
        return (
            orc["grad_W0_W2"],
            orc["grad_W1"],
            orc["grad_K"],
            orc["grad_V"],
            orc["grad_lr0"],
            orc["grad_lr1"],
            orc["grad_lr2"],
        )

    B, two_H, D = W0_W2.shape
    H = two_H // 2
    N = K.shape[1]
    mm_dt = K.dtype
    gdw02 = grad_dW0_W2.to(mm_dt).contiguous()
    gdw1 = grad_dW1.to(mm_dt).contiguous()

    y02 = torch.bmm(W0_W2, K.transpose(1, 2))
    dh = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))
    g_dy02 = torch.bmm(gdw02, K.transpose(1, 2))
    g_hid_lr1 = torch.bmm(gdw1.transpose(1, 2), V.transpose(1, 2))

    out_gdh = torch.empty_like(dh)
    out_gy02 = torch.empty_like(y02)
    out_dy02 = torch.empty_like(y02)
    out_hid_lr1 = torch.empty_like(dh)

    chunk_h = 32 if H >= 128 else 16
    num_splits = triton.cdiv(H, chunk_h)
    block_n = 256 if N >= 256 else 128
    part_glr = torch.empty((num_splits, 3, B, N), device=K.device, dtype=torch.float32)

    _lact_fw_grad_bwd2_splitm_kernel[(triton.cdiv(N, block_n), B, num_splits)](
        dh,
        y02,
        lr0,
        lr1,
        lr2,
        g_dy02,
        g_hid_lr1,
        out_gdh,
        out_gy02,
        part_glr,
        out_dy02,
        out_hid_lr1,
        B,
        H,
        N,
        CHUNK_H=chunk_h,
        BLOCK_N=block_n,
        num_warps=4,
    )

    glr0 = torch.empty((B, N), device=K.device, dtype=lr0.dtype)
    glr1 = torch.empty((B, N), device=K.device, dtype=lr1.dtype)
    glr2 = torch.empty((B, N), device=K.device, dtype=lr2.dtype)
    total_bn = B * N
    _reduce_splitm_glr_kernel[(triton.cdiv(total_bn, 256),)](
        part_glr, glr0, glr1, glr2, num_splits, total_bn, BLOCK_BN=256, num_warps=4
    )

    # Pure vendor cuBLAS Tensor-Core GEMMs for input & fast-weight Hessians
    grad_K = torch.bmm(out_dy02.transpose(1, 2), gdw02)
    grad_K.add_(torch.bmm(out_gy02.transpose(1, 2), W0_W2))

    grad_V = torch.bmm(out_gdh.transpose(1, 2), W1.transpose(1, 2))
    grad_V.add_(torch.bmm(out_hid_lr1.transpose(1, 2), gdw1.transpose(1, 2)))

    wg_dt = W0_W2.dtype if w_grad_dtype is None else w_grad_dtype
    if wg_dt == torch.float32 and K.dtype == torch.bfloat16:
        grad_W1 = torch.bmm(V.transpose(1, 2), out_gdh.transpose(1, 2), out_dtype=torch.float32)
        grad_W0_W2 = torch.bmm(out_gy02, K, out_dtype=torch.float32)
    else:
        grad_W1 = torch.bmm(V.transpose(1, 2), out_gdh.transpose(1, 2)).to(wg_dt)
        grad_W0_W2 = torch.bmm(out_gy02, K).to(wg_dt)
    return grad_W0_W2, grad_W1, grad_K, grad_V, glr0, glr1, glr2


class FusedLaCTFastWeightGradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        W0_W2: torch.Tensor,
        W1: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        lr0: torch.Tensor,
        lr1: torch.Tensor,
        lr2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w02 = W0_W2.contiguous()
        w1 = W1.contiguous()
        k = K.contiguous()
        v = V.contiguous()
        l0 = lr0.contiguous()
        l1 = lr1.contiguous()
        l2 = lr2.contiguous()
        ctx.save_for_backward(w02, w1, k, v, l0, l1, l2)
        return _run_lact_fw_grad_fwd(w02, w1, k, v, l0, l1, l2)

    @staticmethod
    def backward(ctx, grad_dW0_W2: torch.Tensor, grad_dW1: torch.Tensor):
        w02, w1, k, v, l0, l1, l2 = ctx.saved_tensors
        return _run_lact_fw_grad_bwd2(w02, w1, k, v, l0, l1, l2, grad_dW0_W2, grad_dW1)


def _batched_ns5_fwd_only(G_rk: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Forward-only NS5 on `[M_b, R, K]` with `R <= K`.
    Executes in FP32 (or FP64) with cuBLAS-epilogue-fused `torch.baddbmm` (`15` launches instead of `40`),
    eliminating the `~2.7e-2` BF16 5-step cubic mantissa drift (`41x` perturbation amplification)
    while batching all 3 fast-weight matrices `(U0, U2, U1)` into 1 launch per step.
    """
    is_fp64 = G_rk.dtype == torch.float64
    work_dt = torch.float64 if is_fp64 else torch.float32
    Gf = G_rk if is_fp64 else G_rk.float()
    fnorm = torch.sqrt((Gf * Gf).sum(dim=(1, 2), keepdim=True) + 1e-30)
    X = (Gf / (fnorm + eps)).to(work_dt)
    for a, b, c in NS5_COEFFS:
        A = torch.bmm(X, X.transpose(1, 2))
        B_mat = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B_mat, X, beta=a, alpha=1.0)
    return X


def _batched_ns5_recompute_and_vjp(
    G_rk: torch.Tensor, dOut_rk: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Recomputes 5-step NS5 chain from `G_rk` (`R <= K`) in FP32/FP64 with `torch.baddbmm` epilogue fusion."""
    is_fp64 = G_rk.dtype == torch.float64
    work_dt = torch.float64 if is_fp64 else torch.float32
    Gf = G_rk if is_fp64 else G_rk.float()
    fnorm = torch.sqrt((Gf * Gf).sum(dim=(1, 2), keepdim=True) + 1e-30)
    denom = fnorm + eps
    X0 = (Gf / denom).to(work_dt)

    xs = [X0]
    As: List[torch.Tensor] = []
    Bs: List[torch.Tensor] = []
    X = X0
    for a, b, c in NS5_COEFFS:
        A = torch.bmm(X, X.transpose(1, 2))
        B_mat = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B_mat, X, beta=a, alpha=1.0)
        As.append(A)
        Bs.append(B_mat)
        xs.append(X)

    dX = dOut_rk.to(work_dt)
    for t in range(4, -1, -1):
        a, b, c = NS5_COEFFS[t]
        Xt = xs[t]
        At = As[t]
        Bt = Bs[t]
        dBt = torch.bmm(dX, Xt.transpose(1, 2))
        dXt = torch.baddbmm(dX, Bt.transpose(1, 2), dX, beta=a, alpha=1.0)
        At_T = At.transpose(1, 2)
        dAt = torch.baddbmm(dBt, dBt, At_T, beta=b, alpha=c)
        dAt = torch.baddbmm(dAt, At_T, dBt, beta=1.0, alpha=c)
        dAt_sym = dAt + dAt.transpose(1, 2)
        dX = torch.baddbmm(dXt, dAt_sym, Xt, beta=1.0, alpha=1.0)

    dX_f = dX if is_fp64 else dX.float()
    dot = (dX_f * Gf).sum(dim=(1, 2), keepdim=True)
    inv_norm = torch.where(fnorm > 1e-14, 1.0 / fnorm, torch.zeros_like(fnorm))
    dGf = dX_f / denom - Gf * (dot * inv_norm / (denom * denom))
    return dGf.to(G_rk.dtype)


class BatchedCheckpointedMuonNS5Fn(torch.autograd.Function):
    """Exact Independent 3-Matrix Batched Checkpointed `Muon` (`NS5`) for `(U0_in, U2_in, U1_in)`:
    - `U0_in, U2_in`: `[B, H, D]`, `U1_in`: `[B, D, H]`.
    - Preserves exact `R <= K` orientation of three independent `zeropower_via_newtonschulz5` calls:
      - When `H > D`: `[U0_in^T, U2_in^T, U1_in]` has shape `[3B, D, H]` (`D < H`).
      - When `H == D`: `[U0_in, U2_in, U1_in]` has shape `[3B, D, D]`.
      - When `H < D`: `[U0_in, U2_in, U1_in^T]` has shape `[3B, H, D]` (`H < D`).
    - Saves ONLY `G_rk` (`1` tensor instead of `16` NS5 intermediates/chunk) and evaluates exact Fréchet VJP.
    """

    @staticmethod
    def forward(
        ctx,
        U0_in: torch.Tensor,
        U2_in: torch.Tensor,
        U1_in: torch.Tensor,
        eps: float = 1e-7,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, D = U0_in.shape
        assert U2_in.shape == (B, H, D) and U1_in.shape == (B, D, H)
        ctx.eps = eps
        ctx.H = H
        ctx.D = D
        ctx.in_dtype = U0_in.dtype

        if H > D:
            G_rk = torch.cat([U0_in.transpose(1, 2), U2_in.transpose(1, 2), U1_in], dim=0).contiguous()
        elif H == D:
            G_rk = torch.cat([U0_in, U2_in, U1_in], dim=0).contiguous()
        else:
            G_rk = torch.cat([U0_in, U2_in, U1_in.transpose(1, 2)], dim=0).contiguous()

        ctx.save_for_backward(G_rk)
        Y_rk = _batched_ns5_fwd_only(G_rk, eps=eps).to(U0_in.dtype)
        y0, y2, y1 = Y_rk.chunk(3, dim=0)
        if H > D:
            return y0.transpose(1, 2).contiguous(), y2.transpose(1, 2).contiguous(), y1.contiguous()
        elif H == D:
            return y0.contiguous(), y2.contiguous(), y1.contiguous()
        else:
            return y0.contiguous(), y2.contiguous(), y1.transpose(1, 2).contiguous()

    @staticmethod
    def backward(ctx, dU0: torch.Tensor, dU2: torch.Tensor, dU1: torch.Tensor):
        (G_rk,) = ctx.saved_tensors
        H, D = ctx.H, ctx.D
        if H > D:
            dO_rk = torch.cat([dU0.transpose(1, 2), dU2.transpose(1, 2), dU1], dim=0).contiguous()
        elif H == D:
            dO_rk = torch.cat([dU0, dU2, dU1], dim=0).contiguous()
        else:
            dO_rk = torch.cat([dU0, dU2, dU1.transpose(1, 2)], dim=0).contiguous()

        dG_rk = _batched_ns5_recompute_and_vjp(G_rk, dO_rk, eps=ctx.eps).to(ctx.in_dtype)
        g0, g2, g1 = dG_rk.chunk(3, dim=0)
        if H > D:
            return g0.transpose(1, 2).contiguous(), g2.transpose(1, 2).contiguous(), g1.contiguous(), None
        elif H == D:
            return g0.contiguous(), g2.contiguous(), g1.contiguous(), None
        else:
            return g0.contiguous(), g2.contiguous(), g1.transpose(1, 2).contiguous(), None


def fused_l2norm_step_fwd(
    w_base: torch.Tensor,
    u_step: torch.Tensor,
    tgt_scale: torch.Tensor,
    out_dtype: torch.dtype,
    write_main: bool,
    eps: float = 1e-5,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs `_fused_l2norm_update_fwd_kernel` (or exact FP64 path when `w_base.dtype == float64`)."""
    B, nR, D = w_base.shape
    M_rows = B * nR
    if w_base.dtype == torch.float64:
        wcand = w_base + u_step
        w_main_out = wcand if write_main else None
        norm_v = torch.sqrt((wcand * wcand).sum(dim=-1, keepdim=True) + 1e-24)
        inv_d = 1.0 / (norm_v + eps)
        u_hat = wcand * inv_d
        w_norm = (u_hat * tgt_scale.view(B, nR, 1)).to(out_dtype)
        return w_main_out, w_norm, u_hat, inv_d.squeeze(-1)

    wb = w_base.contiguous()
    us = u_step.contiguous()
    ts = tgt_scale.contiguous()
    w_main_out = torch.empty_like(wb) if write_main else wb  # unused pointer when WRITE_MAIN=0
    w_norm_out = torch.empty((B, nR, D), device=wb.device, dtype=out_dtype)
    u_hat_out = torch.empty((B, nR, D), device=wb.device, dtype=torch.float32)
    inv_denom = torch.empty((B, nR), device=wb.device, dtype=torch.float32)

    block_d = max(32, triton.next_power_of_2(D))
    nw = min(8, max(1, block_d // 128))
    _fused_l2norm_update_fwd_kernel[(M_rows,)](
        wb,
        us,
        ts,
        w_main_out,
        w_norm_out,
        u_hat_out,
        inv_denom,
        eps,
        M_rows,
        D,
        WRITE_MAIN=(1 if write_main else 0),
        BLOCK_D=block_d,
        num_warps=nw,
    )
    return (w_main_out if write_main else None), w_norm_out, u_hat_out, inv_denom


def fused_l2norm_step_bwd(
    dw_norm: torch.Tensor,
    dw_main: Optional[torch.Tensor],
    u_hat: torch.Tensor,
    inv_denom: torch.Tensor,
    tgt_scale: torch.Tensor,
    w_base_dtype: torch.dtype,
    u_step_dtype: torch.dtype,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, nR, D = dw_norm.shape
    M_rows = B * nR
    if u_hat.dtype == torch.float64:
        dwn = dw_norm.to(torch.float64)
        ts = tgt_scale.view(B, nR, 1).to(torch.float64)
        inv_d = inv_denom.view(B, nR, 1).to(torch.float64)
        dot_uhat = (dwn * u_hat).sum(dim=-1, keepdim=True)
        uhat_n = torch.sqrt((u_hat * u_hat).sum(dim=-1, keepdim=True))
        inv_un = torch.where(uhat_n > 1e-15, 1.0 / uhat_n, torch.zeros_like(uhat_n))
        g_wc = inv_d * ts * (dwn - u_hat * (dot_uhat * inv_un))
        if dw_main is not None:
            g_wc = g_wc + dw_main.to(torch.float64)
        return g_wc.to(w_base_dtype), g_wc.to(u_step_dtype), dot_uhat.squeeze(-1).to(tgt_scale.dtype)

    dwn_c = dw_norm.contiguous()
    has_dm = dw_main is not None
    dwm_c = dw_main.contiguous() if has_dm else dwn_c
    gw = torch.empty((B, nR, D), device=dwn_c.device, dtype=w_base_dtype)
    gu = torch.empty((B, nR, D), device=dwn_c.device, dtype=u_step_dtype)
    gts = torch.empty((B, nR), device=dwn_c.device, dtype=tgt_scale.dtype)

    block_d = max(32, triton.next_power_of_2(D))
    nw = min(8, max(1, block_d // 128))
    _fused_l2norm_update_bwd_kernel[(M_rows,)](
        dwn_c,
        dwm_c,
        u_hat.contiguous(),
        inv_denom.contiguous(),
        tgt_scale.contiguous(),
        eps,
        gw,
        gu,
        gts,
        M_rows,
        D,
        HAS_DMAIN=(1 if has_dm else 0),
        BLOCK_D=block_d,
        num_warps=nw,
    )
    return gw, gu, gts


class FusedL2NormStepFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w_base: torch.Tensor,
        u_step: torch.Tensor,
        tgt_scale: torch.Tensor,
        out_dtype: torch.dtype,
        write_main: bool,
        eps: float = 1e-5,
    ):
        ctx.w_base_dtype = w_base.dtype
        ctx.u_step_dtype = u_step.dtype
        ctx.write_main = write_main
        ctx.eps = eps
        ts = tgt_scale.squeeze(-1) if tgt_scale.ndim == 3 else tgt_scale
        w_main_out, w_norm_out, u_hat, inv_d = fused_l2norm_step_fwd(
            w_base, u_step, ts, out_dtype=out_dtype, write_main=write_main, eps=eps
        )
        ctx.save_for_backward(u_hat, inv_d, ts)
        if write_main:
            assert w_main_out is not None
            return w_main_out, w_norm_out
        return w_norm_out

    @staticmethod
    def backward(ctx, *grads):
        u_hat, inv_d, ts = ctx.saved_tensors
        if ctx.write_main:
            dw_main, dw_norm = grads
        else:
            dw_main = None
            (dw_norm,) = grads
        gw, gu, gts = fused_l2norm_step_bwd(
            dw_norm, dw_main, u_hat, inv_d, ts, ctx.w_base_dtype, ctx.u_step_dtype, eps=ctx.eps
        )
        return gw, gu, gts, None, None, None


# =============================================================================
# 5. Multi-Chunk LaCT Trajectories: Uncheckpointed RSI vs Chunk-State Checkpointing
# =============================================================================


def _compute_row_norm_safe(w: torch.Tensor) -> torch.Tensor:
    wf = w if w.dtype == torch.float64 else w.float()
    return torch.sqrt((wf * wf).sum(dim=2) + 1e-24).to(w.dtype)


def _compute_row_norm_vjp(w: torch.Tensor, w_norm: torch.Tensor, g_wnorm: torch.Tensor) -> torch.Tensor:
    wf = w if w.dtype == torch.float64 else w.float()
    nf = w_norm.unsqueeze(-1).to(wf.dtype)
    gf = g_wnorm.unsqueeze(-1).to(wf.dtype)
    inv_n = torch.where(nf > 1e-12, 1.0 / nf, torch.zeros_like(nf))
    return (wf * inv_n * gf).to(w.dtype)


def rsi_block_causal_lact_swiglu_uncheckpointed(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = True,
    momentum: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    normalize_fast_weights: bool = True,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Round-5 RSI Multi-Chunk `LaCT` (Uncheckpointed Chunks):
    Uses `FusedLaCTSwiGLUApplyFn` + `FusedLaCTFastWeightGradFn` + `BatchedCheckpointedMuonNS5Fn`
    + `FusedL2NormStepFn`.
    """
    is_fp64 = w0.dtype == torch.float64 and q.dtype == torch.float64
    mm_dt = torch.float64 if is_fp64 else torch.bfloat16
    state_dt = w0.dtype

    w02_init = torch.cat([w0, w2], dim=1).contiguous()
    w1_init = w1.contiguous()
    w02_norm = _compute_row_norm_safe(w02_init)
    w1_norm = _compute_row_norm_safe(w1_init)

    w02_main = w02_init
    w1_main = w1_init
    w02_cur = w02_init
    w1_cur = w1_init

    has_mom = momentum is not None
    if has_mom:
        m02 = torch.zeros_like(w02_init)
        m1 = torch.zeros_like(w1_init)
    else:
        m02 = m1 = None

    seq_len = q.shape[1]
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    outs: List[torch.Tensor] = []

    for c_idx in range(num_chunks):
        s = c_idx * chunk_size
        e = min(seq_len, s + chunk_size)
        qi = q[:, s:e, :].to(mm_dt)

        w02_mm = w02_cur if w02_cur.dtype == mm_dt else w02_cur.to(mm_dt)
        w1_mm = w1_cur if w1_cur.dtype == mm_dt else w1_cur.to(mm_dt)
        outs.append(FusedLaCTSwiGLUApplyFn.apply(w02_mm, w1_mm, qi))

        if c_idx == num_chunks - 1:
            break

        ki = k[:, s:e, :].to(mm_dt)
        vi = v[:, s:e, :].to(mm_dt)
        l0i = lr0[:, s:e, 0] if lr0.ndim == 3 else lr0[:, s:e]
        l1i = lr1[:, s:e, 0] if lr1.ndim == 3 else lr1[:, s:e]
        l2i = lr2[:, s:e, 0] if lr2.ndim == 3 else lr2[:, s:e]

        dw02, dw1 = FusedLaCTFastWeightGradFn.apply(w02_mm, w1_mm, ki, vi, l0i, l1i, l2i)
        dw02_s = dw02.to(state_dt)
        dw1_s = dw1.to(state_dt)

        if has_mom:
            assert momentum is not None and m02 is not None and m1 is not None
            mi = momentum[:, s:e, :].mean(dim=1, keepdim=True).to(state_dt)
            m02 = m02 * mi + dw02_s
            m1 = m1 * mi + dw1_s
            u02_in, u1_in = m02, m1
        else:
            u02_in, u1_in = dw02_s, dw1_s

        if use_muon:
            u0_in, u2_in = u02_in.chunk(2, dim=1)
            u0, u2, u1 = BatchedCheckpointedMuonNS5Fn.apply(u0_in.contiguous(), u2_in.contiguous(), u1_in.contiguous(), 1e-7)
            u02 = torch.cat([u0, u2], dim=1)
        else:
            u02, u1 = u02_in, u1_in

        if prenorm:
            if normalize_fast_weights:
                w02_main, w02_cur = FusedL2NormStepFn.apply(w02_main, u02, w02_norm, state_dt, True, eps)
                w1_main, w1_cur = FusedL2NormStepFn.apply(w1_main, u1, w1_norm, state_dt, True, eps)
            else:
                w02_main = w02_main + u02
                w1_main = w1_main + u1
                w02_cur = w02_main
                w1_cur = w1_main
        else:
            if normalize_fast_weights:
                w02_cur = FusedL2NormStepFn.apply(w02_cur, u02, w02_norm, state_dt, False, eps)
                w1_cur = FusedL2NormStepFn.apply(w1_cur, u1, w1_norm, state_dt, False, eps)
            else:
                w02_cur = w02_cur + u02
                w1_cur = w1_cur + u1

    return torch.cat(outs, dim=1)


class ChunkCheckpointedLaCTSwiGLUFn(torch.autograd.Function):
    """Chunk-Boundary Fast-Weight State Checkpointing for Multi-Chunk `LaCT` (`Track 2` `t4`):
    Stores ONLY chunk-boundary fast-weight & momentum states `(w02_state[c], w1_state[c], m02[c], m1[c])`
    of size `O(N_c * B_h * H * D)` instead of `O(N_c * B_h * H * C)` token-level GEMM intermediates,
    replaying exactly one chunk at a time in reverse order during backward (`0` RNG/dropout drift).
    """

    @staticmethod
    def forward(
        ctx,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lr0: torch.Tensor,
        lr1: torch.Tensor,
        lr2: torch.Tensor,
        momentum: Optional[torch.Tensor],
        chunk_size: int,
        use_muon: bool,
        prenorm: bool,
        normalize_fast_weights: bool,
        eps: float,
    ) -> torch.Tensor:
        is_fp64 = w0.dtype == torch.float64 and q.dtype == torch.float64
        mm_dt = torch.float64 if is_fp64 else torch.bfloat16
        state_dt = w0.dtype

        w02_init = torch.cat([w0.detach(), w2.detach()], dim=1).contiguous()
        w1_init = w1.detach().contiguous()
        w02_norm = _compute_row_norm_safe(w02_init)
        w1_norm = _compute_row_norm_safe(w1_init)

        has_mom = momentum is not None
        B, seq_len, D = q.shape
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # Boundary state banks (ONLY compact O(Bh * H * D) fast-weight matrices per chunk — ZERO O(C) token tensors!)
        w02_cur_bank: List[torch.Tensor] = []
        w1_cur_bank: List[torch.Tensor] = []
        m02_bank: List[torch.Tensor] = []
        m1_bank: List[torch.Tensor] = []
        mi_bank: List[torch.Tensor] = []
        g_rk_bank: List[Optional[torch.Tensor]] = []
        uhat02_bank: List[Optional[torch.Tensor]] = []
        invd02_bank: List[Optional[torch.Tensor]] = []
        uhat1_bank: List[Optional[torch.Tensor]] = []
        invd1_bank: List[Optional[torch.Tensor]] = []

        w02_main = w02_init
        w1_main = w1_init
        w02_cur = w02_init
        w1_cur = w1_init
        m02 = torch.zeros_like(w02_init) if has_mom else w02_init.new_empty(0)
        m1 = torch.zeros_like(w1_init) if has_mom else w1_init.new_empty(0)

        q_mm = q if (q.dtype == mm_dt and q.stride(-1) == 1) else q.to(mm_dt).contiguous()
        k_mm = k if (k.dtype == mm_dt and k.stride(-1) == 1) else k.to(mm_dt).contiguous()
        v_mm = v if (v.dtype == mm_dt and v.stride(-1) == 1) else v.to(mm_dt).contiguous()
        out = torch.empty((B, seq_len, D), device=q.device, dtype=mm_dt)

        for c_idx in range(num_chunks):
            s = c_idx * chunk_size
            e = min(seq_len, s + chunk_size)

            w02_cur_bank.append(w02_cur)
            w1_cur_bank.append(w1_cur)
            if has_mom and c_idx < num_chunks - 1:
                m02_bank.append(m02)
                m1_bank.append(m1)

            qi = q_mm[:, s:e, :]
            w02_mm = w02_cur if w02_cur.dtype == mm_dt else w02_cur.to(mm_dt)
            w1_mm = w1_cur if w1_cur.dtype == mm_dt else w1_cur.to(mm_dt)
            _run_lact_apply_fwd(w02_mm, w1_mm, qi, out=out[:, s:e, :])

            if c_idx == num_chunks - 1:
                break

            ki = k_mm[:, s:e, :]
            vi = v_mm[:, s:e, :]
            l0i = (lr0[:, s:e, 0] if lr0.ndim == 3 else lr0[:, s:e]).contiguous()
            l1i = (lr1[:, s:e, 0] if lr1.ndim == 3 else lr1[:, s:e]).contiguous()
            l2i = (lr2[:, s:e, 0] if lr2.ndim == 3 else lr2[:, s:e]).contiguous()

            dw02, dw1 = _run_lact_fw_grad_fwd(w02_mm, w1_mm, ki, vi, l0i, l1i, l2i)
            dw02_s = dw02 if dw02.dtype == state_dt else dw02.to(state_dt)
            dw1_s = dw1 if dw1.dtype == state_dt else dw1.to(state_dt)

            if has_mom:
                assert momentum is not None
                mi = momentum[:, s:e, :].mean(dim=1, keepdim=True).to(state_dt)
                mi_bank.append(mi)
                m02 = m02 * mi + dw02_s
                m1 = m1 * mi + dw1_s
                u02_in, u1_in = m02, m1
            else:
                u02_in, u1_in = dw02_s, dw1_s

            if use_muon:
                u0_in, u2_in = u02_in.chunk(2, dim=1)
                H_dim = u0_in.shape[1]
                if H_dim > D:
                    G_rk = torch.cat([u0_in.transpose(1, 2), u2_in.transpose(1, 2), u1_in], dim=0).contiguous()
                    Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                    y0, y2, u1 = Y_rk.chunk(3, dim=0)
                    u02 = torch.cat([y0.transpose(1, 2), y2.transpose(1, 2)], dim=1).contiguous()
                    u1 = u1.contiguous()
                elif H_dim == D:
                    G_rk = torch.cat([u0_in, u2_in, u1_in], dim=0).contiguous()
                    Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                    y0, y2, u1 = Y_rk.chunk(3, dim=0)
                    u02 = torch.cat([y0, y2], dim=1).contiguous()
                    u1 = u1.contiguous()
                else:
                    G_rk = torch.cat([u0_in, u2_in, u1_in.transpose(1, 2)], dim=0).contiguous()
                    Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                    y0, y2, y1 = Y_rk.chunk(3, dim=0)
                    u02 = torch.cat([y0, y2], dim=1).contiguous()
                    u1 = y1.transpose(1, 2).contiguous()
                g_rk_bank.append(G_rk)
            else:
                g_rk_bank.append(None)
                u02, u1 = u02_in, u1_in

            if prenorm:
                if normalize_fast_weights:
                    w02_m_next, w02_cur, uhat02, invd02 = fused_l2norm_step_fwd(w02_main, u02, w02_norm, state_dt, True, eps)
                    w1_m_next, w1_cur, uhat1, invd1 = fused_l2norm_step_fwd(w1_main, u1, w1_norm, state_dt, True, eps)
                    assert w02_m_next is not None and w1_m_next is not None
                    w02_main, w1_main = w02_m_next, w1_m_next
                    uhat02_bank.append(uhat02)
                    invd02_bank.append(invd02)
                    uhat1_bank.append(uhat1)
                    invd1_bank.append(invd1)
                else:
                    w02_main = w02_main + u02
                    w1_main = w1_main + u1
                    w02_cur = w02_main
                    w1_cur = w1_main
                    uhat02_bank.append(None)
                    invd02_bank.append(None)
                    uhat1_bank.append(None)
                    invd1_bank.append(None)
            else:
                if normalize_fast_weights:
                    _, w02_cur, uhat02, invd02 = fused_l2norm_step_fwd(w02_cur, u02, w02_norm, state_dt, False, eps)
                    _, w1_cur, uhat1, invd1 = fused_l2norm_step_fwd(w1_cur, u1, w1_norm, state_dt, False, eps)
                    uhat02_bank.append(uhat02)
                    invd02_bank.append(invd02)
                    uhat1_bank.append(uhat1)
                    invd1_bank.append(invd1)
                else:
                    w02_cur = w02_cur + u02
                    w1_cur = w1_cur + u1
                    uhat02_bank.append(None)
                    invd02_bank.append(None)
                    uhat1_bank.append(None)
                    invd1_bank.append(None)

        ctx.chunk_size = chunk_size
        ctx.use_muon = use_muon
        ctx.prenorm = prenorm
        ctx.normalize_fast_weights = normalize_fast_weights
        ctx.eps = eps
        ctx.has_mom = has_mom
        ctx.lr_ndim = lr0.ndim
        ctx.w02_cur_bank = w02_cur_bank
        ctx.w1_cur_bank = w1_cur_bank
        ctx.m02_bank = m02_bank
        ctx.m1_bank = m1_bank
        ctx.mi_bank = mi_bank
        ctx.g_rk_bank = g_rk_bank
        ctx.uhat02_bank = uhat02_bank
        ctx.invd02_bank = invd02_bank
        ctx.uhat1_bank = uhat1_bank
        inv_save = invd1_bank
        ctx.invd1_bank = inv_save

        mom_save = momentum if has_mom else q.new_empty(0)
        ctx.save_for_backward(w02_init, w1_init, w02_norm, w1_norm, q_mm, k_mm, v_mm, lr0, lr1, lr2, mom_save)
        ctx.q_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.v_dtype = v.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        w02_init, w1_init, w02_norm, w1_norm, q_mm, k_mm, v_mm, lr0, lr1, lr2, mom_save = ctx.saved_tensors
        has_mom = ctx.has_mom
        momentum = mom_save if has_mom else None
        chunk_size = ctx.chunk_size
        use_muon = ctx.use_muon
        prenorm = ctx.prenorm
        normalize_fw = ctx.normalize_fast_weights
        eps = ctx.eps

        is_fp64 = w02_init.dtype == torch.float64 and q_mm.dtype == torch.float64
        mm_dt = torch.float64 if is_fp64 else torch.bfloat16
        state_dt = w02_init.dtype
        B, seq_len, D = q_mm.shape
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        go_mm = grad_out if (grad_out.dtype == mm_dt and grad_out.stride(-1) == 1) else grad_out.to(mm_dt).contiguous()

        grad_q = torch.empty_like(q_mm)
        last_s = (num_chunks - 1) * chunk_size
        grad_k = torch.empty_like(k_mm)
        grad_k[:, last_s:, :].zero_()
        grad_v = torch.empty_like(v_mm)
        grad_v[:, last_s:, :].zero_()
        grad_lr0 = torch.zeros_like(lr0)
        grad_lr1 = torch.zeros_like(lr1)
        grad_lr2 = torch.zeros_like(lr2)
        grad_mom = torch.zeros_like(momentum) if has_mom else None

        g_w02_cur = torch.zeros_like(ctx.w02_cur_bank[-1])
        g_w1_cur = torch.zeros_like(ctx.w1_cur_bank[-1])
        g_w02_main = torch.zeros_like(w02_init) if prenorm else None
        g_w1_main = torch.zeros_like(w1_init) if prenorm else None
        g_m02 = torch.zeros_like(w02_init) if has_mom else None
        g_m1 = torch.zeros_like(w1_init) if has_mom else None

        g_w02_norm_acc = torch.zeros_like(w02_norm)
        g_w1_norm_acc = torch.zeros_like(w1_norm)

        for c_idx in range(num_chunks - 1, -1, -1):
            s = c_idx * chunk_size
            e = min(seq_len, s + chunk_size)
            clen = e - s

            # Step 1: If c_idx < num_chunks - 1, backprop through state transition c_idx -> c_idx + 1
            if c_idx < num_chunks - 1:
                ki = k_mm[:, s:e, :]
                vi = v_mm[:, s:e, :]
                l0i = (lr0[:, s:e, 0] if ctx.lr_ndim == 3 else lr0[:, s:e]).contiguous()
                l1i = (lr1[:, s:e, 0] if ctx.lr_ndim == 3 else lr1[:, s:e]).contiguous()
                l2i = (lr2[:, s:e, 0] if ctx.lr_ndim == 3 else lr2[:, s:e]).contiguous()

                w02_c = ctx.w02_cur_bank[c_idx]
                w1_c = ctx.w1_cur_bank[c_idx]
                w02_mm = w02_c if w02_c.dtype == mm_dt else w02_c.to(mm_dt)
                w1_mm = w1_c if w1_c.dtype == mm_dt else w1_c.to(mm_dt)

                if prenorm:
                    assert g_w02_main is not None and g_w1_main is not None
                    if normalize_fw:
                        uhat02 = ctx.uhat02_bank[c_idx]
                        invd02 = ctx.invd02_bank[c_idx]
                        uhat1 = ctx.uhat1_bank[c_idx]
                        invd1 = ctx.invd1_bank[c_idx]
                        assert uhat02 is not None and invd02 is not None and uhat1 is not None and invd1 is not None
                        g_w02_main, g_u02, gts02 = fused_l2norm_step_bwd(
                            g_w02_cur, g_w02_main, uhat02, invd02, w02_norm, state_dt, state_dt, eps=eps
                        )
                        g_w1_main, g_u1, gts1 = fused_l2norm_step_bwd(
                            g_w1_cur, g_w1_main, uhat1, invd1, w1_norm, state_dt, state_dt, eps=eps
                        )
                        g_w02_norm_acc.add_(gts02)
                        g_w1_norm_acc.add_(gts1)
                    else:
                        g_w02_main = g_w02_main + g_w02_cur.to(state_dt)
                        g_w1_main = g_w1_main + g_w1_cur.to(state_dt)
                        g_u02 = g_w02_main
                        g_u1 = g_w1_main
                    g_w02_cur = torch.zeros_like(w02_c)
                    g_w1_cur = torch.zeros_like(w1_c)
                else:
                    if normalize_fw:
                        uhat02 = ctx.uhat02_bank[c_idx]
                        invd02 = ctx.invd02_bank[c_idx]
                        uhat1 = ctx.uhat1_bank[c_idx]
                        invd1 = ctx.invd1_bank[c_idx]
                        assert uhat02 is not None and invd02 is not None and uhat1 is not None and invd1 is not None
                        g_w02_cur, g_u02, gts02 = fused_l2norm_step_bwd(
                            g_w02_cur, None, uhat02, invd02, w02_norm, state_dt, state_dt, eps=eps
                        )
                        g_w1_cur, g_u1, gts1 = fused_l2norm_step_bwd(
                            g_w1_cur, None, uhat1, invd1, w1_norm, state_dt, state_dt, eps=eps
                        )
                        g_w02_norm_acc.add_(gts02)
                        g_w1_norm_acc.add_(gts1)
                    else:
                        g_u02 = g_w02_cur.to(state_dt)
                        g_u1 = g_w1_cur.to(state_dt)

                # Backprop through Muon-NS5 if enabled (single Fréchet VJP recompute from compact G_rk!)
                if use_muon:
                    G_rk = ctx.g_rk_bank[c_idx]
                    assert G_rk is not None
                    gu0, gu2 = g_u02.chunk(2, dim=1)
                    H_dim = gu0.shape[1]
                    if H_dim > D:
                        dO_rk = torch.cat([gu0.transpose(1, 2), gu2.transpose(1, 2), g_u1], dim=0).contiguous()
                        dG_rk = _batched_ns5_recompute_and_vjp(G_rk, dO_rk, eps=1e-7).to(state_dt)
                        dg0, dg2, g_u1_in = dG_rk.chunk(3, dim=0)
                        g_u02_in = torch.cat([dg0.transpose(1, 2), dg2.transpose(1, 2)], dim=1).contiguous()
                    elif H_dim == D:
                        dO_rk = torch.cat([gu0, gu2, g_u1], dim=0).contiguous()
                        dG_rk = _batched_ns5_recompute_and_vjp(G_rk, dO_rk, eps=1e-7).to(state_dt)
                        dg0, dg2, g_u1_in = dG_rk.chunk(3, dim=0)
                        g_u02_in = torch.cat([dg0, dg2], dim=1).contiguous()
                    else:
                        dO_rk = torch.cat([gu0, gu2, g_u1.transpose(1, 2)], dim=0).contiguous()
                        dG_rk = _batched_ns5_recompute_and_vjp(G_rk, dO_rk, eps=1e-7).to(state_dt)
                        dg0, dg2, dg1 = dG_rk.chunk(3, dim=0)
                        g_u02_in = torch.cat([dg0, dg2], dim=1).contiguous()
                        g_u1_in = dg1.transpose(1, 2).contiguous()
                else:
                    g_u02_in, g_u1_in = g_u02, g_u1

                # Backprop through Momentum if enabled
                if has_mom:
                    m02_prev = ctx.m02_bank[c_idx]
                    m1_prev = ctx.m1_bank[c_idx]
                    mi = ctx.mi_bank[c_idx]
                    assert g_m02 is not None and g_m1 is not None and grad_mom is not None
                    g_m02_tot = g_m02 + g_u02_in
                    g_m1_tot = g_m1 + g_u1_in
                    g_mi = (g_m02_tot * m02_prev).sum(dim=(1, 2), keepdim=True) + (g_m1_tot * m1_prev).sum(dim=(1, 2), keepdim=True)
                    grad_mom[:, s:e, :] = (g_mi / float(clen)).expand(B, clen, 1).to(grad_mom.dtype)
                    g_m02 = g_m02_tot * mi
                    g_m1 = g_m1_tot * mi
                    g_dw02 = g_m02_tot
                    g_dw1 = g_m1_tot
                else:
                    g_dw02, g_dw1 = g_u02_in, g_u1_in

                # Backprop through Second-Order Fwd-of-Bwd inner gradient!
                gw02_fg, gw1_fg, gk_c, gv_c, gl0_c, gl1_c, gl2_c = _run_lact_fw_grad_bwd2(
                    w02_mm, w1_mm, ki, vi, l0i, l1i, l2i, g_dw02, g_dw1, w_grad_dtype=torch.float32
                )
                g_w02_cur = g_w02_cur + gw02_fg.to(g_w02_cur.dtype)
                g_w1_cur = g_w1_cur + gw1_fg.to(g_w1_cur.dtype)
                grad_k[:, s:e, :] = gk_c
                grad_v[:, s:e, :] = gv_c
                if ctx.lr_ndim == 3:
                    grad_lr0[:, s:e, 0] = gl0_c.to(grad_lr0.dtype)
                    grad_lr1[:, s:e, 0] = gl1_c.to(grad_lr1.dtype)
                    grad_lr2[:, s:e, 0] = gl2_c.to(grad_lr2.dtype)
                else:
                    grad_lr0[:, s:e] = gl0_c.to(grad_lr0.dtype)
                    grad_lr1[:, s:e] = gl1_c.to(grad_lr1.dtype)
                    grad_lr2[:, s:e] = gl2_c.to(grad_lr2.dtype)

            # Step 2: Backprop through Apply on chunk `c_idx`
            qi = q_mm[:, s:e, :]
            go_i = go_mm[:, s:e, :]
            w02_c = ctx.w02_cur_bank[c_idx]
            w1_c = ctx.w1_cur_bank[c_idx]
            w02_mm = w02_c if w02_c.dtype == mm_dt else w02_c.to(mm_dt)
            w1_mm = w1_c if w1_c.dtype == mm_dt else w1_c.to(mm_dt)

            gw02_ap, gw1_ap, _ = _run_lact_apply_bwd(
                w02_mm, w1_mm, qi, go_i, w_grad_dtype=torch.float32, out_grad_Q=grad_q[:, s:e, :]
            )
            g_w02_cur = g_w02_cur + gw02_ap.to(g_w02_cur.dtype)
            g_w1_cur = g_w1_cur + gw1_ap.to(g_w1_cur.dtype)

        # Combine initial fast-weight gradients + row-norm target-scale gradients
        if prenorm:
            assert g_w02_main is not None and g_w1_main is not None
            g_w02_init = g_w02_main + g_w02_cur.to(state_dt)
            g_w1_init = g_w1_main + g_w1_cur.to(state_dt)
        else:
            g_w02_init = g_w02_cur.to(state_dt)
            g_w1_init = g_w1_cur.to(state_dt)

        if normalize_fw and num_chunks > 1:
            g_w02_init = g_w02_init + _compute_row_norm_vjp(w02_init, w02_norm, g_w02_norm_acc)
            g_w1_init = g_w1_init + _compute_row_norm_vjp(w1_init, w1_norm, g_w1_norm_acc)

        g_w0, g_w2 = g_w02_init.chunk(2, dim=1)
        return (
            g_w0.contiguous(),
            g_w1_init.contiguous(),
            g_w2.contiguous(),
            grad_q.to(ctx.q_dtype),
            grad_k.to(ctx.k_dtype),
            grad_v.to(ctx.v_dtype),
            grad_lr0,
            grad_lr1,
            grad_lr2,
            grad_mom,
            None,
            None,
            None,
            None,
            None,
        )


def chunk_checkpointed_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = True,
    momentum: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    normalize_fast_weights: bool = True,
    eps: float = 1e-5,
) -> torch.Tensor:
    return ChunkCheckpointedLaCTSwiGLUFn.apply(
        w0, w1, w2, q, k, v, lr0, lr1, lr2, momentum, chunk_size, use_muon, prenorm, normalize_fast_weights, eps
    )


# =============================================================================
# 6. 8-GPU NVSwitch Within-Chunk Context Parallelism (`CP = 8`, `Gate G9`)
# =============================================================================


class SubChunkCPLaCTAutogradFn(torch.autograd.Function):
    """Exact Distributed Within-Chunk Context Parallelism (`arXiv:2505.23884` Sec. 3.4 / Alg. 3)
    for Multi-Chunk Shifted Block-Causal `LaCT` with `Momentum + Muon-NS5 + L2Norm`.

    Sharding layout (`Gate G9` exact 1-GPU equivalence):
    - Global sequence `S = num_chunks * C_global`.
    - Within each chunk `c in 0..num_chunks-1`, rank `r in 0..P-1` holds the contiguous sub-slice
      `[c * C_global + r * C_sub, c * C_global + (r + 1) * C_sub)` where `C_sub = C_global // P`.
      Thus `q_local, k_local, v_local` on rank `r` have shape `[B_h, num_chunks * C_sub, D]`.
    - Replicated-Adjoint Backward Rule:
      `s_{c+1} = F(s_c, sum_r d_{c,r})` is identically replicated on all `P` ranks.
      During backward, maintain ONE logically global adjoint `(g_w02_cur, g_w1_cur, g_m02, g_m1)` on all ranks:
      1. Run `VJP_F` once locally on replicated `(g_w02_cur, g_w1_cur)` -> `(g_w_base, g_d_c)`.
      2. Run local shard `VJP_{grad}(g_d_c)` and `VJP_{apply}(go_{c,r})` -> `(gw_shard_r, gk_r, gv_r, gq_r)`.
      3. `AllReduceSum_r(gw_shard_r)` across `cp_group` and add `g_w_base` **ONCE** (never `P`x unless intentional
         `bug_double_allreduce_base=True` audit!).
    """

    @staticmethod
    def forward(
        ctx,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        lr0_local: torch.Tensor,
        lr1_local: torch.Tensor,
        lr2_local: torch.Tensor,
        mom_local: Optional[torch.Tensor],
        c_sub: int,
        use_muon: bool,
        eps: float,
        cp_group: dist.ProcessGroup,
        bug_double_allreduce_base: bool = False,
    ) -> torch.Tensor:
        world_size = dist.get_world_size(cp_group)
        mm_dt = torch.bfloat16
        state_dt = w0.dtype
        has_mom = mom_local is not None

        w02_init = torch.cat([w0.detach(), w2.detach()], dim=1).contiguous()
        w1_init = w1.detach().contiguous()
        w02_norm = _compute_row_norm_safe(w02_init)
        w1_norm = _compute_row_norm_safe(w1_init)

        B, local_len, D = q_local.shape
        num_chunks = local_len // c_sub
        assert num_chunks * c_sub == local_len

        w02_cur_bank: List[torch.Tensor] = []
        w1_cur_bank: List[torch.Tensor] = []
        m02_bank: List[torch.Tensor] = []
        m1_bank: List[torch.Tensor] = []
        d_global_bank: List[Tuple[torch.Tensor, torch.Tensor]] = []
        mi_global_bank: List[torch.Tensor] = []

        w02_cur = w02_init
        w1_cur = w1_init
        m02 = torch.zeros_like(w02_init) if has_mom else w02_init.new_empty(0)
        m1 = torch.zeros_like(w1_init) if has_mom else w1_init.new_empty(0)

        out_local = torch.empty((B, local_len, D), device=q_local.device, dtype=mm_dt)

        for c_idx in range(num_chunks):
            s = c_idx * c_sub
            e = s + c_sub
            w02_cur_bank.append(w02_cur)
            w1_cur_bank.append(w1_cur)
            if has_mom and c_idx < num_chunks - 1:
                m02_bank.append(m02)
                m1_bank.append(m1)

            qi = q_local[:, s:e, :].to(mm_dt).contiguous()
            w02_mm = w02_cur.to(mm_dt)
            w1_mm = w1_cur.to(mm_dt)
            out_local[:, s:e, :] = _run_lact_apply_fwd(w02_mm, w1_mm, qi)

            if c_idx == num_chunks - 1:
                break

            ki = k_local[:, s:e, :].to(mm_dt).contiguous()
            vi = v_local[:, s:e, :].to(mm_dt).contiguous()
            l0i = (lr0_local[:, s:e, 0] if lr0_local.ndim == 3 else lr0_local[:, s:e]).contiguous()
            l1i = (lr1_local[:, s:e, 0] if lr1_local.ndim == 3 else lr1_local[:, s:e]).contiguous()
            l2i = (lr2_local[:, s:e, 0] if lr2_local.ndim == 3 else lr2_local[:, s:e]).contiguous()

            dw02_loc, dw1_loc = _run_lact_fw_grad_fwd(w02_mm, w1_mm, ki, vi, l0i, l1i, l2i)
            dw02_s = dw02_loc.to(torch.float32)
            dw1_s = dw1_loc.to(torch.float32)

            # Single coalesced NVSwitch AllReduce(SUM) across CP shards (Eq. 11)!
            if has_mom:
                assert mom_local is not None
                mi_loc = mom_local[:, s:e, :].mean(dim=1, keepdim=True).to(torch.float32) / float(world_size)
                flat = torch.cat([dw02_s.reshape(-1), dw1_s.reshape(-1), mi_loc.reshape(-1)], dim=0)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=cp_group)
                n02 = dw02_s.numel()
                n1 = dw1_s.numel()
                dw02_g = flat[:n02].view_as(dw02_s).to(state_dt)
                dw1_g = flat[n02 : n02 + n1].view_as(dw1_s).to(state_dt)
                mi_g = flat[n02 + n1 :].view(B, 1, 1).to(state_dt)
                mi_global_bank.append(mi_g)
                m02 = m02 * mi_g + dw02_g
                m1 = m1 * mi_g + dw1_g
                u02_in, u1_in = m02, m1
            else:
                flat = torch.cat([dw02_s.reshape(-1), dw1_s.reshape(-1)], dim=0)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=cp_group)
                n02 = dw02_s.numel()
                dw02_g = flat[:n02].view_as(dw02_s).to(state_dt)
                dw1_g = flat[n02:].view_as(dw1_s).to(state_dt)
                u02_in, u1_in = dw02_g, dw1_g

            d_global_bank.append((dw02_g, dw1_g))

            if use_muon:
                u0_in, u2_in = u02_in.chunk(2, dim=1)
                H_dim = u0_in.shape[1]
                if H_dim > D:
                    G_rk = torch.cat([u0_in.transpose(1, 2), u2_in.transpose(1, 2), u1_in], dim=0).contiguous()
                    Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                    y0, y2, u1 = Y_rk.chunk(3, dim=0)
                    u02 = torch.cat([y0.transpose(1, 2), y2.transpose(1, 2)], dim=1).contiguous()
                    u1 = u1.contiguous()
                elif H_dim == D:
                    G_rk = torch.cat([u0_in, u2_in, u1_in], dim=0).contiguous()
                    Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                    y0, y2, u1 = Y_rk.chunk(3, dim=0)
                    u02 = torch.cat([y0, y2], dim=1).contiguous()
                    u1 = u1.contiguous()
                else:
                    G_rk = torch.cat([u0_in, u2_in, u1_in.transpose(1, 2)], dim=0).contiguous()
                    Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                    y0, y2, y1 = Y_rk.chunk(3, dim=0)
                    u02 = torch.cat([y0, y2], dim=1).contiguous()
                    u1 = y1.transpose(1, 2).contiguous()
            else:
                u02, u1 = u02_in, u1_in

            _, w02_cur, _, _ = fused_l2norm_step_fwd(w02_cur, u02, w02_norm, state_dt, False, eps)
            _, w1_cur, _, _ = fused_l2norm_step_fwd(w1_cur, u1, w1_norm, state_dt, False, eps)

        ctx.c_sub = c_sub
        ctx.use_muon = use_muon
        ctx.eps = eps
        ctx.has_mom = has_mom
        ctx.lr_ndim = lr0_local.ndim
        ctx.cp_group = cp_group
        ctx.bug_double_allreduce_base = bug_double_allreduce_base
        ctx.w02_cur_bank = w02_cur_bank
        ctx.w1_cur_bank = w1_cur_bank
        ctx.m02_bank = m02_bank
        ctx.m1_bank = m1_bank
        ctx.d_global_bank = d_global_bank
        ctx.mi_global_bank = mi_global_bank

        mom_save = mom_local if has_mom else q_local.new_empty(0)
        ctx.save_for_backward(w02_init, w1_init, w02_norm, w1_norm, q_local, k_local, v_local, lr0_local, lr1_local, lr2_local, mom_save)
        return out_local

    @staticmethod
    def backward(ctx, grad_out_local: torch.Tensor):
        w02_init, w1_init, w02_norm, w1_norm, q_local, k_local, v_local, lr0_local, lr1_local, lr2_local, mom_save = ctx.saved_tensors
        cp_group = ctx.cp_group
        world_size = dist.get_world_size(cp_group)
        has_mom = ctx.has_mom
        c_sub = ctx.c_sub
        use_muon = ctx.use_muon
        eps = ctx.eps
        mm_dt = torch.bfloat16
        state_dt = w02_init.dtype

        B, local_len, D = q_local.shape
        num_chunks = local_len // c_sub

        grad_q = torch.empty_like(q_local)
        grad_k = torch.zeros_like(k_local)
        grad_v = torch.zeros_like(v_local)
        grad_lr0 = torch.zeros_like(lr0_local)
        grad_lr1 = torch.zeros_like(lr1_local)
        grad_lr2 = torch.zeros_like(lr2_local)
        grad_mom = torch.zeros_like(mom_save) if has_mom else None

        # Logically global state adjoints replicated identically on all CP ranks!
        g_w02_cur = torch.zeros_like(ctx.w02_cur_bank[-1])
        g_w1_cur = torch.zeros_like(ctx.w1_cur_bank[-1])
        g_m02 = torch.zeros_like(w02_init) if has_mom else None
        g_m1 = torch.zeros_like(w1_init) if has_mom else None
        g_w02_norm_acc = torch.zeros_like(w02_norm)
        g_w1_norm_acc = torch.zeros_like(w1_norm)

        for c_idx in range(num_chunks - 1, -1, -1):
            s = c_idx * c_sub
            e = s + c_sub
            w02_c = ctx.w02_cur_bank[c_idx]
            w1_c = ctx.w1_cur_bank[c_idx]
            w02_mm = w02_c.to(mm_dt)
            w1_mm = w1_c.to(mm_dt)

            if c_idx < num_chunks - 1:
                dw02_g, dw1_g = ctx.d_global_bank[c_idx]
                if has_mom:
                    m02_prev = ctx.m02_bank[c_idx]
                    m1_prev = ctx.m1_bank[c_idx]
                    mi_g = ctx.mi_global_bank[c_idx]
                    u02_in = m02_prev * mi_g + dw02_g
                    u1_in = m1_prev * mi_g + dw1_g
                else:
                    m02_prev = m1_prev = mi_g = None
                    u02_in, u1_in = dw02_g, dw1_g

                if use_muon:
                    u0_in, u2_in = u02_in.chunk(2, dim=1)
                    H_dim = u0_in.shape[1]
                    if H_dim > D:
                        G_rk = torch.cat([u0_in.transpose(1, 2), u2_in.transpose(1, 2), u1_in], dim=0).contiguous()
                        Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                        y0, y2, u1 = Y_rk.chunk(3, dim=0)
                        u02 = torch.cat([y0.transpose(1, 2), y2.transpose(1, 2)], dim=1).contiguous()
                        u1 = u1.contiguous()
                    elif H_dim == D:
                        G_rk = torch.cat([u0_in, u2_in, u1_in], dim=0).contiguous()
                        Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                        y0, y2, u1 = Y_rk.chunk(3, dim=0)
                        u02 = torch.cat([y0, y2], dim=1).contiguous()
                        u1 = u1.contiguous()
                    else:
                        G_rk = torch.cat([u0_in, u2_in, u1_in.transpose(1, 2)], dim=0).contiguous()
                        Y_rk = _batched_ns5_fwd_only(G_rk, eps=1e-7).to(state_dt)
                        y0, y2, y1 = Y_rk.chunk(3, dim=0)
                        u02 = torch.cat([y0, y2], dim=1).contiguous()
                        u1 = y1.transpose(1, 2).contiguous()
                else:
                    G_rk = None
                    u02, u1 = u02_in, u1_in

                _, _, uhat02, invd02 = fused_l2norm_step_fwd(w02_c, u02, w02_norm, state_dt, False, eps)
                _, _, uhat1, invd1 = fused_l2norm_step_fwd(w1_c, u1, w1_norm, state_dt, False, eps)
                g_w02_base, g_u02, gts02 = fused_l2norm_step_bwd(
                    g_w02_cur, None, uhat02, invd02, w02_norm, state_dt, state_dt, eps=eps
                )
                g_w1_base, g_u1, gts1 = fused_l2norm_step_bwd(
                    g_w1_cur, None, uhat1, invd1, w1_norm, state_dt, state_dt, eps=eps
                )
                g_w02_norm_acc.add_(gts02)
                g_w1_norm_acc.add_(gts1)

                if use_muon:
                    assert G_rk is not None
                    gu0, gu2 = g_u02.chunk(2, dim=1)
                    H_dim = gu0.shape[1]
                    if H_dim > D:
                        dO_rk = torch.cat([gu0.transpose(1, 2), gu2.transpose(1, 2), g_u1], dim=0).contiguous()
                        dG_rk = _batched_ns5_recompute_and_vjp(G_rk, dO_rk, eps=1e-7).to(state_dt)
                        dg0, dg2, g_u1_in = dG_rk.chunk(3, dim=0)
                        g_u02_in = torch.cat([dg0.transpose(1, 2), dg2.transpose(1, 2)], dim=1).contiguous()
                    elif H_dim == D:
                        dO_rk = torch.cat([gu0, gu2, g_u1], dim=0).contiguous()
                        dG_rk = _batched_ns5_recompute_and_vjp(G_rk, dO_rk, eps=1e-7).to(state_dt)
                        dg0, dg2, g_u1_in = dG_rk.chunk(3, dim=0)
                        g_u02_in = torch.cat([dg0, dg2], dim=1).contiguous()
                    else:
                        dO_rk = torch.cat([gu0, gu2, g_u1.transpose(1, 2)], dim=0).contiguous()
                        dG_rk = _batched_ns5_recompute_and_vjp(G_rk, dO_rk, eps=1e-7).to(state_dt)
                        dg0, dg2, dg1 = dG_rk.chunk(3, dim=0)
                        g_u02_in = torch.cat([dg0, dg2], dim=1).contiguous()
                        g_u1_in = dg1.transpose(1, 2).contiguous()
                else:
                    g_u02_in, g_u1_in = g_u02, g_u1

                if has_mom:
                    assert g_m02 is not None and g_m1 is not None and m02_prev is not None and m1_prev is not None and mi_g is not None and grad_mom is not None
                    g_m02_tot = g_m02 + g_u02_in
                    g_m1_tot = g_m1 + g_u1_in
                    g_mi = (g_m02_tot * m02_prev).sum(dim=(1, 2), keepdim=True) + (g_m1_tot * m1_prev).sum(dim=(1, 2), keepdim=True)
                    grad_mom[:, s:e, :] = (g_mi / float(world_size * c_sub)).expand(B, c_sub, 1).to(grad_mom.dtype)
                    g_m02 = g_m02_tot * mi_g
                    g_m1 = g_m1_tot * mi_g
                    g_dw02_g = g_m02_tot
                    g_dw1_g = g_m1_tot
                else:
                    g_dw02_g, g_dw1_g = g_u02_in, g_u1_in

                ki = k_local[:, s:e, :].to(mm_dt).contiguous()
                vi = v_local[:, s:e, :].to(mm_dt).contiguous()
                l0i = (lr0_local[:, s:e, 0] if ctx.lr_ndim == 3 else lr0_local[:, s:e]).contiguous()
                l1i = (lr1_local[:, s:e, 0] if ctx.lr_ndim == 3 else lr1_local[:, s:e]).contiguous()
                l2i = (lr2_local[:, s:e, 0] if ctx.lr_ndim == 3 else lr2_local[:, s:e]).contiguous()

                gw02_fg, gw1_fg, gk_c, gv_c, gl0_c, gl1_c, gl2_c = _run_lact_fw_grad_bwd2(
                    w02_mm, w1_mm, ki, vi, l0i, l1i, l2i, g_dw02_g, g_dw1_g, w_grad_dtype=torch.float32
                )
                grad_k[:, s:e, :] = gk_c.to(grad_k.dtype)
                grad_v[:, s:e, :] = gv_c.to(grad_v.dtype)
                if ctx.lr_ndim == 3:
                    grad_lr0[:, s:e, 0] = gl0_c.to(grad_lr0.dtype)
                    grad_lr1[:, s:e, 0] = gl1_c.to(grad_lr1.dtype)
                    grad_lr2[:, s:e, 0] = gl2_c.to(grad_lr2.dtype)
                else:
                    grad_lr0[:, s:e] = gl0_c.to(grad_lr0.dtype)
                    grad_lr1[:, s:e] = gl1_c.to(grad_lr1.dtype)
                    grad_lr2[:, s:e] = gl2_c.to(grad_lr2.dtype)
            else:
                g_w02_base = torch.zeros_like(w02_c)
                g_w1_base = torch.zeros_like(w1_c)
                gw02_fg = torch.zeros_like(w02_c)
                gw1_fg = torch.zeros_like(w1_c)

            qi = q_local[:, s:e, :].to(mm_dt).contiguous()
            go_i = grad_out_local[:, s:e, :].to(mm_dt).contiguous()
            gw02_ap, gw1_ap, gq_c = _run_lact_apply_bwd(
                w02_mm, w1_mm, qi, go_i, w_grad_dtype=torch.float32
            )
            grad_q[:, s:e, :] = gq_c.to(grad_q.dtype)

            # Sum shard-local gradients across CP ranks via ONE coalesced AllReduce(SUM),
            # then add replicated `g_w_base` ONCE!
            gw02_loc = gw02_fg.to(torch.float32) + gw02_ap.to(torch.float32)
            gw1_loc = gw1_fg.to(torch.float32) + gw1_ap.to(torch.float32)
            if ctx.bug_double_allreduce_base:
                gw02_loc = gw02_loc + g_w02_base.to(torch.float32)
                gw1_loc = gw1_loc + g_w1_base.to(torch.float32)
            flat_b = torch.cat([gw02_loc.reshape(-1), gw1_loc.reshape(-1)], dim=0)
            dist.all_reduce(flat_b, op=dist.ReduceOp.SUM, group=cp_group)
            n02 = gw02_loc.numel()
            gw02_sum = flat_b[:n02].view_as(gw02_c := w02_c).to(state_dt)
            gw1_sum = flat_b[n02:].view_as(w1_c).to(state_dt)

            if ctx.bug_double_allreduce_base:
                g_w02_cur = gw02_sum
                g_w1_cur = gw1_sum
            else:
                g_w02_cur = g_w02_base + gw02_sum
                g_w1_cur = g_w1_base + gw1_sum

        g_w02_init = g_w02_cur.to(state_dt)
        g_w1_init = g_w1_cur.to(state_dt)
        if num_chunks > 1:
            g_w02_init = g_w02_init + _compute_row_norm_vjp(w02_init, w02_norm, g_w02_norm_acc)
            g_w1_init = g_w1_init + _compute_row_norm_vjp(w1_init, w1_norm, g_w1_norm_acc)

        g_w0, g_w2 = g_w02_init.chunk(2, dim=1)
        return (
            g_w0.contiguous(),
            g_w1_init.contiguous(),
            g_w2.contiguous(),
            grad_q,
            grad_k,
            grad_v,
            grad_lr0,
            grad_lr1,
            grad_lr2,
            grad_mom,
            None,
            None,
            None,
            None,
            None,
        )


class P2PHaloCausalSWAAutogradFn(torch.autograd.Function):
    """Exact 1-Hop P2P Boundary KV-Halo Causal Sliding-Window Attention (`W_swa <= C_sub`)
    for Sub-Chunk Context Parallelism (`CP = 8`, `num_chunks >= 1`):
    - On rank `r` and chunk `c`:
      - For `r > 0`: predecessor slice of length `W_swa` is the tail of chunk `c` on `rank - 1`.
      - For `r == 0` and `c > 0`: predecessor slice of length `W_swa` is the tail of chunk `c - 1` on `rank = P - 1`!
      - For `r == 0` and `c == 0`: start of sequence -> no predecessor halo (`causal=True` alone!).
    - Exchanges ALL `num_chunks` halos in a single batched ring P2P `batch_isend_irecv`!
    """

    @staticmethod
    def forward(
        ctx,
        q_local: torch.Tensor,  # [B, num_chunks * c_sub, H_q, D_h] bf16
        k_local: torch.Tensor,  # [B, num_chunks * c_sub, H_kv, D_h] bf16
        v_local: torch.Tensor,  # [B, num_chunks * c_sub, H_kv, D_h] bf16
        c_sub: int,
        w_swa: int,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        assert flash_attn_func is not None, "flash_attn_func required"
        assert w_swa <= c_sub
        rank = dist.get_rank(cp_group)
        world_size = dist.get_world_size(cp_group)
        B, local_len, H_q, D_h = q_local.shape
        H_kv = k_local.shape[2]
        num_chunks = local_len // c_sub

        kv_chunks_k = k_local.view(B, num_chunks, c_sub, H_kv, D_h)
        kv_chunks_v = v_local.view(B, num_chunks, c_sub, H_kv, D_h)

        # Tail of each chunk on this rank: [2, B, num_chunks, w_swa, H_kv, D_h]
        send_tails = torch.stack(
            [kv_chunks_k[:, :, -w_swa:, :, :], kv_chunks_v[:, :, -w_swa:, :, :]], dim=0
        ).contiguous()
        recv_tails = torch.empty_like(send_tails)

        if world_size > 1:
            dst = (rank + 1) % world_size
            src = (rank - 1 + world_size) % world_size
            ops = [
                dist.P2POp(dist.isend, send_tails, dst, group=cp_group),
                dist.P2POp(dist.irecv, recv_tails, src, group=cp_group),
            ]
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        else:
            recv_tails.copy_(send_tails)

        q_ch = q_local.view(B, num_chunks, c_sub, H_q, D_h)
        out_ch = torch.empty_like(q_ch)

        for c_idx in range(num_chunks):
            qc = q_ch[:, c_idx]
            kc = kv_chunks_k[:, c_idx]
            vc = kv_chunks_v[:, c_idx]
            if rank == 0 and c_idx == 0:
                out_ch[:, c_idx] = flash_attn_func(qc, kc, vc, causal=True, window_size=(w_swa - 1, 0))
            else:
                # For rank > 0, halo is chunk `c_idx` from `rank - 1`.
                # For rank == 0 and c_idx > 0, halo is chunk `c_idx - 1` from `rank = world_size - 1`!
                src_c = c_idx if rank > 0 else (c_idx - 1)
                k_ext = torch.cat([recv_tails[0, :, src_c], kc], dim=1)
                v_ext = torch.cat([recv_tails[1, :, src_c], vc], dim=1)
                # With `w_swa` prepended tokens and `c_sub` queries, `flash_attn_func` aligns the last
                # query token with the last key token when `causal=True`!
                out_ch[:, c_idx] = flash_attn_func(qc, k_ext, v_ext, causal=True, window_size=(w_swa - 1, 0))

        ctx.c_sub = c_sub
        ctx.w_swa = w_swa
        ctx.cp_group = cp_group
        ctx.save_for_backward(q_local, k_local, v_local, recv_tails)
        return out_ch.reshape(B, local_len, H_q, D_h)

    @staticmethod
    def backward(ctx, grad_out_local: torch.Tensor):
        q_local, k_local, v_local, recv_tails = ctx.saved_tensors
        c_sub = ctx.c_sub
        w_swa = ctx.w_swa
        cp_group = ctx.cp_group
        rank = dist.get_rank(cp_group)
        world_size = dist.get_world_size(cp_group)
        B, local_len, H_q, D_h = q_local.shape
        H_kv = k_local.shape[2]
        num_chunks = local_len // c_sub

        q_ch = q_local.view(B, num_chunks, c_sub, H_q, D_h)
        k_ch = k_local.view(B, num_chunks, c_sub, H_kv, D_h)
        v_ch = v_local.view(B, num_chunks, c_sub, H_kv, D_h)
        go_ch = grad_out_local.contiguous().view(B, num_chunks, c_sub, H_q, D_h)

        dq_ch = torch.empty_like(q_ch)
        dk_ch = torch.zeros_like(k_ch)
        dv_ch = torch.zeros_like(v_ch)
        d_halo_send = torch.zeros_like(recv_tails)

        with torch.enable_grad():
            for c_idx in range(num_chunks):
                qc = q_ch[:, c_idx].detach().requires_grad_(True)
                kc = k_ch[:, c_idx].detach()
                vc = v_ch[:, c_idx].detach()
                goc = go_ch[:, c_idx]
                if rank == 0 and c_idx == 0:
                    kc_r = kc.requires_grad_(True)
                    vc_r = vc.requires_grad_(True)
                    oc = flash_attn_func(qc, kc_r, vc_r, causal=True, window_size=(w_swa - 1, 0))
                    oc.backward(goc)
                    dq_ch[:, c_idx] = qc.grad
                    dk_ch[:, c_idx] = kc_r.grad
                    dv_ch[:, c_idx] = vc_r.grad
                else:
                    src_c = c_idx if rank > 0 else (c_idx - 1)
                    k_ext = torch.cat([recv_tails[0, :, src_c], kc], dim=1).detach().requires_grad_(True)
                    v_ext = torch.cat([recv_tails[1, :, src_c], vc], dim=1).detach().requires_grad_(True)
                    oc = flash_attn_func(qc, k_ext, v_ext, causal=True, window_size=(w_swa - 1, 0))
                    oc.backward(goc)
                    dq_ch[:, c_idx] = qc.grad
                    dk_ext = k_ext.grad
                    dv_ext = v_ext.grad
                    d_halo_send[0, :, src_c] = dk_ext[:, :w_swa]
                    d_halo_send[1, :, src_c] = dv_ext[:, :w_swa]
                    dk_ch[:, c_idx] = dk_ext[:, w_swa:]
                    dv_ch[:, c_idx] = dv_ext[:, w_swa:]

        # Reverse 1-hop P2P halo gradient return (`rank -> (rank - 1) % world_size`)
        d_halo_recv = torch.empty_like(d_halo_send)
        if world_size > 1:
            dst = (rank - 1 + world_size) % world_size
            src = (rank + 1) % world_size
            ops = [
                dist.P2POp(dist.isend, d_halo_send.contiguous(), dst, group=cp_group),
                dist.P2POp(dist.irecv, d_halo_recv, src, group=cp_group),
            ]
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        else:
            d_halo_recv.copy_(d_halo_send)

        # Accumulate returned halo gradients onto local chunk tails
        # Note: on rank == world_size - 1, chunk `num_chunks - 1` has no successor chunk (`d_halo_recv[:, :, -1] == 0`).
        dk_ch[:, :, -w_swa:, :, :].add_(d_halo_recv[0])
        dv_ch[:, :, -w_swa:, :, :].add_(d_halo_recv[1])

        return (
            dq_ch.reshape(B, local_len, H_q, D_h),
            dk_ch.reshape(B, local_len, H_kv, D_h),
            dv_ch.reshape(B, local_len, H_kv, D_h),
            None,
            None,
            None,
        )


# =============================================================================
# 6B. Zero-Spill Fused LaCT Output `RMSNorm(x) * SiLU(g)` Gate (`layer_lact_swiglu.py` L213)
# =============================================================================


@triton.jit
def _lact_rmsnorm_silu_gate_fwd_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    M_rows,
    D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M_rows:
        return
    cols = tl.arange(0, BLOCK_D)
    m = cols < D
    off = row * D + cols
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    tl.store(rstd_ptr + row, rstd)
    silu_g = g * tl.sigmoid(g)
    y = (x * rstd * w) * silu_g
    tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _lact_rmsnorm_silu_gate_bwd_kernel(
    dy_ptr,
    x_ptr,
    g_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    dg_ptr,
    part_dw_ptr,
    M_rows,
    D,
    BLOCK_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    m = cols < D
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    acc_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    r_start = pid * ROWS_PER_PROG
    r_end = tl.minimum(M_rows, r_start + ROWS_PER_PROG)
    out_x_dt = dx_ptr.dtype.element_ty
    out_g_dt = dg_ptr.dtype.element_ty
    for row in range(r_start, r_end):
        off = row * D + cols
        dy = tl.load(dy_ptr + off, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)
        sig = tl.sigmoid(g)
        silu_g = g * sig
        s_bp = sig * (1.0 + g * (1.0 - sig))
        x_hat = x * rstd
        dg = dy * (x_hat * w) * s_bp
        dnorm = dy * silu_g * w
        c1 = tl.sum(dnorm * x_hat, axis=0) / D
        dx = (dnorm - x_hat * c1) * rstd
        acc_dw += dy * silu_g * x_hat
        tl.store(dx_ptr + off, dx.to(out_x_dt), mask=m)
        tl.store(dg_ptr + off, dg.to(out_g_dt), mask=m)
    tl.store(part_dw_ptr + pid * D + cols, acc_dw, mask=m)


def eager_lact_rmsnorm_silu_gate_ref(
    x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    is_fp64 = x.dtype == torch.float64
    xf = x if is_fp64 else x.float()
    gf = g if is_fp64 else g.float()
    wf = weight if is_fp64 else weight.float()
    rstd = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)
    y = (xf * rstd * wf) * F.silu(gf)
    return y if is_fp64 else y.to(x.dtype)


class FusedLaCTRMSNormGateFn(torch.autograd.Function):
    """Zero-spill fused `RMSNorm(x) * SiLU(g)` output gate (`layer_lact_swiglu.py` line 213)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if x.dtype == torch.float64:
            with torch.enable_grad():
                xr = x.detach().requires_grad_(True)
                gr = g.detach().requires_grad_(True)
                wr = weight.detach().requires_grad_(True)
                ctx.fp64_mode = True
                ctx.eps = eps
                ctx.save_for_backward(xr, gr, wr)
                return eager_lact_rmsnorm_silu_gate_ref(xr, gr, wr, eps=eps).detach()
        ctx.fp64_mode = False
        xc = x.contiguous()
        gc = g.contiguous()
        wc = weight.contiguous()
        D = xc.shape[-1]
        M_rows = xc.numel() // D
        y = torch.empty_like(xc)
        rstd = torch.empty((M_rows,), device=xc.device, dtype=torch.float32)
        bd = max(32, triton.next_power_of_2(D))
        nw = min(8, max(2, bd // 128))
        _lact_rmsnorm_silu_gate_fwd_kernel[(M_rows,)](
            xc, gc, wc, y, rstd, M_rows, D, eps, BLOCK_D=bd, num_warps=nw
        )
        ctx.save_for_backward(xc, gc, wc, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if ctx.fp64_mode:
            xr, gr, wr = ctx.saved_tensors
            with torch.enable_grad():
                y = eager_lact_rmsnorm_silu_gate_ref(xr, gr, wr, eps=ctx.eps)
                y.backward(dy)
                return xr.grad.detach(), gr.grad.detach(), wr.grad.detach(), None
        xc, gc, wc, rstd = ctx.saved_tensors
        dyc = dy.contiguous()
        D = xc.shape[-1]
        M_rows = xc.numel() // D
        dx = torch.empty_like(xc)
        dg = torch.empty_like(gc)
        rpp = 32
        n_progs = triton.cdiv(M_rows, rpp)
        part_dw = torch.empty((n_progs, D), device=xc.device, dtype=torch.float32)
        bd = max(32, triton.next_power_of_2(D))
        nw = min(8, max(2, bd // 128))
        _lact_rmsnorm_silu_gate_bwd_kernel[(n_progs,)](
            dyc, xc, gc, wc, rstd, dx, dg, part_dw, M_rows, D, BLOCK_D=bd, ROWS_PER_PROG=rpp, num_warps=nw
        )
        dw = part_dw.sum(dim=0).to(wc.dtype)
        return dx, dg, dw, None


# =============================================================================
# 7. Cubin Register-Spill & Official LaCT 3-Acc Spill Auditor
# =============================================================================


@triton.jit
def _official_lact_3acc_audit_kernel(
    w0_w2_ptr,
    w1_ptr,
    x_ptr,
    v_ptr,
    lr0_ptr,
    lr1_ptr,
    lr2_ptr,
    dy0_dy2_ptr,
    hidden_ptr,
    B,
    M: tl.constexpr,
    N,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Exact reproduction of `a1600012888/LaCT` `_swiglu_three_bmm_with_lr_kernel` (`triton_swiglu_bwd_with_lr.py`)
    used in `Suite 0` & `Suite A` `Track 1` (`t2`) to measure real SM80 cubin register spills.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc_y0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_y2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_dh = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w0_b = w0_w2_ptr + pid_b * 2 * M * K
    w2_b = w0_b + M * K
    w1_b = w1_ptr + pid_b * K * M
    x_b = x_ptr + pid_b * N * K
    v_b = v_ptr + pid_b * N * K
    for ko in range(0, tl.cdiv(K, BLOCK_K)):
        k_ids = ko * BLOCK_K + offs_k
        mk = k_ids < K
        mm = offs_m < M
        mn = offs_n < N
        a0 = tl.load(w0_b + offs_m[:, None] * K + k_ids[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        a2 = tl.load(w2_b + offs_m[:, None] * K + k_ids[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        a1 = tl.load(w1_b + k_ids[None, :] * M + offs_m[:, None], mask=mm[:, None] & mk[None, :], other=0.0)
        bx = tl.load(x_b + offs_n[None, :] * K + k_ids[:, None], mask=mk[:, None] & mn[None, :], other=0.0)
        bv = tl.load(v_b + offs_n[None, :] * K + k_ids[:, None], mask=mk[:, None] & mn[None, :], other=0.0)
        acc_y0 += tl.dot(a0, bx, out_dtype=tl.float32)
        acc_y2 += tl.dot(a2, bx, out_dtype=tl.float32)
        acc_dh += tl.dot(a1, bv, out_dtype=tl.float32)
    sigma = tl.sigmoid(acc_y0)
    swish = sigma * acc_y0
    mn = offs_n < N
    hid = swish * acc_y2 * tl.load(lr1_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)[None, :]
    dy0 = sigma * acc_y2 * acc_dh * (1.0 + acc_y0 * (1.0 - sigma)) * tl.load(lr0_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)[None, :]
    dy2 = swish * acc_dh * tl.load(lr2_ptr + pid_b * N + offs_n, mask=mn, other=0.0).to(tl.float32)[None, :]
    mo = (offs_m[:, None] < M) & mn[None, :]
    tl.store(dy0_dy2_ptr + pid_b * 2 * M * N + offs_m[:, None] * N + offs_n[None, :], dy0.to(tl.bfloat16), mask=mo)
    tl.store(dy0_dy2_ptr + pid_b * 2 * M * N + (M + offs_m[:, None]) * N + offs_n[None, :], dy2.to(tl.bfloat16), mask=mo)
    tl.store(hidden_ptr + pid_b * M * N + offs_m[:, None] * N + offs_n[None, :], hid.to(tl.bfloat16), mask=mo)


def audit_lact_triton_cubin_spills() -> Dict[str, Any]:
    """Audits all compiled specializations of Round-5 RSI Triton kernels for `0` cubin spills."""
    rsi_kernels = [
        ("_lact_swiglu_apply_epilogue_kernel", _lact_swiglu_apply_epilogue_kernel),
        ("_lact_swiglu_apply_bwd_epilogue_kernel", _lact_swiglu_apply_bwd_epilogue_kernel),
        ("_lact_fw_grad_fwd_epilogue_kernel", _lact_fw_grad_fwd_epilogue_kernel),
        ("_lact_fw_grad_bwd2_splitm_kernel", _lact_fw_grad_bwd2_splitm_kernel),
        ("_reduce_splitm_glr_kernel", _reduce_splitm_glr_kernel),
        ("_fused_l2norm_update_fwd_kernel", _fused_l2norm_update_fwd_kernel),
        ("_fused_l2norm_update_bwd_kernel", _fused_l2norm_update_bwd_kernel),
        ("_lact_rmsnorm_silu_gate_fwd_kernel", _lact_rmsnorm_silu_gate_fwd_kernel),
        ("_lact_rmsnorm_silu_gate_bwd_kernel", _lact_rmsnorm_silu_gate_bwd_kernel),
    ]
    per_kernel: Dict[str, Any] = {}
    total_specs = 0
    total_spills = 0
    max_regs = 0
    for name, kfn in rsi_kernels:
        cache_vals: List[Any] = []
        for dev_map in getattr(kfn, "device_caches", {}).values():
            if isinstance(dev_map, tuple) and len(dev_map) > 0 and isinstance(dev_map[0], dict):
                cache_vals.extend(dev_map[0].values())
            elif isinstance(dev_map, dict):
                cache_vals.extend(dev_map.values())
        k_spills = 0
        k_regs = 0
        n_comp = 0
        for ck in cache_vals:
            if hasattr(ck, "n_spills"):
                n_comp += 1
                k_spills += int(ck.n_spills)
                k_regs = max(k_regs, int(getattr(ck, "n_regs", 0)))
        per_kernel[name] = {"compiled_specs": n_comp, "max_regs": k_regs, "spills": k_spills}
        total_specs += n_comp
        total_spills += k_spills
        max_regs = max(max_regs, k_regs)
    return {
        "total_compiled_specializations": total_specs,
        "total_cubin_spills": total_spills,
        "max_registers_per_thread": max_regs,
        "per_kernel": per_kernel,
    }
