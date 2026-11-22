#!/usr/bin/env python3
"""Round-2 SM80 Triton & Autograd Kernels for E128 (`v119`):
1. Adaptive Direct-`dc6` / Split-Subchunk 3-Point & Unified `adaLN-Zero` (`D = 3,584`, >1,000 GB/s).
2. Disjoint-Lifetime Zero-Spill `K_TILE=16` + Zero-Copy `OBC` `BD3-LM` `[x_t || x_0]` Attention (`0` cubin spills).
3. Merged `W_qkv` Zero-Clone Dual-Stream `QKV-Split-RoPE` & Merged `W_gate_up` Triton `FusedSwiGLU`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)

import bench_diffusion_llm_bd3_adaln_training as e127


# ============================================================================
# 1. Track A Round-2 Kernels: Adaptive Direct-`dc6` / Split-Subchunk `adaLN-Zero`
# ============================================================================


def build_split_adaln_meta(
    doc_lens: list[int], W: int, rows_per_cta: int, device: torch.device
) -> dict[str, Any]:
    base = e127.build_adaln_segment_meta(doc_lens, W, device)
    sc_blk: list[int] = []
    sc_s: list[int] = []
    sc_e: list[int] = []
    cur = 0
    b_global = 0
    for L in doc_lens:
        n_b = (L + W - 1) // W
        for b in range(n_b):
            s0 = cur + b * W
            e0 = min(cur + L, s0 + W)
            for s in range(s0, e0, rows_per_cta):
                sc_blk.append(b_global)
                sc_s.append(s)
                sc_e.append(min(e0, s + rows_per_cta))
            b_global += 1
        cur += L

    blk_sc_s = [0] * b_global
    blk_sc_e = [0] * b_global
    idx = 0
    for sc_i, b_id in enumerate(sc_blk):
        if sc_i == 0 or sc_blk[sc_i - 1] != b_id:
            blk_sc_s[b_id] = idx
        idx += 1
        blk_sc_e[b_id] = idx

    n_sc = len(sc_blk)
    return {
        **base,
        "rows_per_cta": rows_per_cta,
        "n_subchunks": n_sc,
        "one_cta_per_blk": (n_sc == b_global),
        "sc_blk": torch.tensor(sc_blk, device=device, dtype=torch.int32).contiguous(),
        "sc_s": torch.tensor(sc_s, device=device, dtype=torch.int32).contiguous(),
        "sc_e": torch.tensor(sc_e, device=device, dtype=torch.int32).contiguous(),
        "blk_sc_s": torch.tensor(blk_sc_s, device=device, dtype=torch.int32).contiguous(),
        "blk_sc_e": torch.tensor(blk_sc_e, device=device, dtype=torch.int32).contiguous(),
    }


@triton.jit
def _reduce_subchunks_kernel(
    Part_ptr, Out_ptr, BlkScS_ptr, BlkScE_ptr,
    stride_psc, stride_pk, stride_ob, stride_ok,
    K_PARAMS: tl.constexpr, D_MODEL: tl.constexpr, BLOCK_D: tl.constexpr,
):
    blk_id = tl.program_id(0).to(tl.int64)
    k_id = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    m = offs_d < D_MODEL
    s = tl.load(BlkScS_ptr + blk_id)
    e = tl.load(BlkScE_ptr + blk_id)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for sc in range(s, e):
        acc += tl.load(Part_ptr + sc * stride_psc + k_id * stride_pk + offs_d, mask=m, other=0.0)
    tl.store(Out_ptr + blk_id * stride_ob + k_id * stride_ok + offs_d, acc.to(Out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _adaln_p1_bwd_split_kernel(
    X_ptr, dY_ptr, Gamma_ptr, Rstd_ptr,
    ScBlk_ptr, ScS_ptr, ScE_ptr,
    dX_ptr, OutG_ptr, OutB_ptr,
    stride_xn, stride_gb, stride_out0,
    DIRECT_WRITE: tl.constexpr, D_MODEL: tl.constexpr, BLOCK_D: tl.constexpr,
):
    sc_id = tl.program_id(0).to(tl.int64)
    blk_id = tl.load(ScBlk_ptr + sc_id).to(tl.int64)
    s_row = tl.load(ScS_ptr + sc_id).to(tl.int64)
    e_row = tl.load(ScE_ptr + sc_id).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    m = offs_d < D_MODEL

    gamma = tl.load(Gamma_ptr + blk_id * stride_gb + offs_d, mask=m, other=0.0).to(tl.float32)
    scale = 1.0 + gamma
    dg_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    inv_d = 1.0 / float(D_MODEL)

    for row in tl.range(s_row, e_row, loop_unroll_factor=1):
        x = tl.load(X_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        dy = tl.load(dY_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + row).to(tl.float32)
        x_hat = x * rstd
        dg_acc += dy * x_hat
        db_acc += dy
        g = dy * scale
        c = tl.sum(g * x_hat, axis=0) * inv_d
        dx = rstd * (g - x_hat * c)
        tl.store(dX_ptr + row * stride_xn + offs_d, dx.to(dX_ptr.dtype.element_ty), mask=m)

    dst = blk_id if DIRECT_WRITE else sc_id
    tl.store(OutG_ptr + dst * stride_out0 + offs_d, dg_acc.to(OutG_ptr.dtype.element_ty), mask=m)
    tl.store(OutB_ptr + dst * stride_out0 + offs_d, db_acc.to(OutB_ptr.dtype.element_ty), mask=m)


@triton.jit
def _adaln_p2_bwd_split_kernel(
    XNext_ptr, Sub1_ptr, dXNextUp_ptr, dY2_ptr,
    Alpha1_ptr, Gamma2_ptr, Rstd2_ptr,
    ScBlk_ptr, ScS_ptr, ScE_ptr,
    dX_ptr, dSub1_ptr, OutA1_ptr, OutG2_ptr, OutB2_ptr,
    stride_xn, stride_gb, stride_out0,
    DIRECT_WRITE: tl.constexpr, D_MODEL: tl.constexpr, BLOCK_D: tl.constexpr,
):
    sc_id = tl.program_id(0).to(tl.int64)
    blk_id = tl.load(ScBlk_ptr + sc_id).to(tl.int64)
    s_row = tl.load(ScS_ptr + sc_id).to(tl.int64)
    e_row = tl.load(ScE_ptr + sc_id).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    m = offs_d < D_MODEL

    a1 = tl.load(Alpha1_ptr + blk_id * stride_gb + offs_d, mask=m, other=0.0).to(tl.float32)
    g2 = tl.load(Gamma2_ptr + blk_id * stride_gb + offs_d, mask=m, other=0.0).to(tl.float32)
    scale2 = 1.0 + g2
    da1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    dg2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    db2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    inv_d = 1.0 / float(D_MODEL)

    for row in tl.range(s_row, e_row, loop_unroll_factor=1):
        xn = tl.load(XNext_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        s1 = tl.load(Sub1_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        dxn_up = tl.load(dXNextUp_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        dy2 = tl.load(dY2_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        r2 = tl.load(Rstd2_ptr + row).to(tl.float32)
        xn_hat = xn * r2
        dg2 += dy2 * xn_hat
        db2 += dy2
        wdy2 = dy2 * scale2
        c2 = tl.sum(wdy2 * xn_hat, axis=0) * inv_d
        dxn = dxn_up + r2 * (wdy2 - xn_hat * c2)
        da1 += dxn * s1
        ds1 = dxn * a1
        tl.store(dX_ptr + row * stride_xn + offs_d, dxn.to(dX_ptr.dtype.element_ty), mask=m)
        tl.store(dSub1_ptr + row * stride_xn + offs_d, ds1.to(dSub1_ptr.dtype.element_ty), mask=m)

    dst = blk_id if DIRECT_WRITE else sc_id
    tl.store(OutA1_ptr + dst * stride_out0 + offs_d, da1.to(OutA1_ptr.dtype.element_ty), mask=m)
    tl.store(OutG2_ptr + dst * stride_out0 + offs_d, dg2.to(OutG2_ptr.dtype.element_ty), mask=m)
    tl.store(OutB2_ptr + dst * stride_out0 + offs_d, db2.to(OutB2_ptr.dtype.element_ty), mask=m)


@triton.jit
def _adaln_p3_fwd_row_kernel(
    XNext_ptr, Sub2_ptr, Alpha2_ptr, BlockId_ptr, XOut_ptr,
    stride_xn, stride_ab, D_MODEL: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.load(BlockId_ptr + row).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    m = offs_d < D_MODEL
    xn = tl.load(XNext_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(Sub2_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
    a2 = tl.load(Alpha2_ptr + b * stride_ab + offs_d, mask=m, other=0.0).to(tl.float32)
    xo = (xn + a2 * s2).to(XOut_ptr.dtype.element_ty)
    tl.store(XOut_ptr + row * stride_xn + offs_d, xo, mask=m)


@triton.jit
def _adaln_p3_bwd_split_kernel(
    dXOut_ptr, Sub2_ptr, Alpha2_ptr,
    ScBlk_ptr, ScS_ptr, ScE_ptr,
    dSub2_ptr, OutA2_ptr,
    stride_xn, stride_ab, stride_out0,
    DIRECT_WRITE: tl.constexpr, D_MODEL: tl.constexpr, BLOCK_D: tl.constexpr,
):
    sc_id = tl.program_id(0).to(tl.int64)
    blk_id = tl.load(ScBlk_ptr + sc_id).to(tl.int64)
    s_row = tl.load(ScS_ptr + sc_id).to(tl.int64)
    e_row = tl.load(ScE_ptr + sc_id).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    m = offs_d < D_MODEL
    a2 = tl.load(Alpha2_ptr + blk_id * stride_ab + offs_d, mask=m, other=0.0).to(tl.float32)
    da2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    for row in tl.range(s_row, e_row, loop_unroll_factor=1):
        dxo = tl.load(dXOut_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        s2 = tl.load(Sub2_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        da2 += dxo * s2
        ds2 = dxo * a2
        tl.store(dSub2_ptr + row * stride_xn + offs_d, ds2.to(dSub2_ptr.dtype.element_ty), mask=m)
    dst = blk_id if DIRECT_WRITE else sc_id
    tl.store(OutA2_ptr + dst * stride_out0 + offs_d, da2.to(OutA2_ptr.dtype.element_ty), mask=m)


class SplitAdaLNP1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, meta: dict[str, Any], eps: float = 1e-6):
        x, gamma, beta = x.contiguous(), gamma.contiguous(), beta.contiguous()
        n, d = x.shape
        block_d = triton.next_power_of_2(d)
        y = torch.empty_like(x)
        rstd = torch.empty((n,), device=x.device, dtype=torch.float32)
        e127._block_adaln_fwd_kernel[(n,)](
            x, gamma, beta, meta["block_id"], y, rstd, D=d, BLOCK_D=block_d, EPS=eps, num_warps=8
        )
        ctx.save_for_backward(x, gamma, rstd)
        ctx.meta = meta
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, gamma, rstd = ctx.saved_tensors
        meta = ctx.meta
        dy = dy.contiguous()
        d = x.shape[1]
        block_d = triton.next_power_of_2(d)
        dx = torch.empty_like(x)
        n_sc = meta["n_subchunks"]
        n_blks = meta["n_blks"]
        if meta["one_cta_per_blk"]:
            dg = torch.empty_like(gamma)
            db = torch.empty_like(gamma)
            _adaln_p1_bwd_split_kernel[(n_sc,)](
                x, dy, gamma, rstd, meta["sc_blk"], meta["sc_s"], meta["sc_e"],
                dx, dg, db, x.stride(0), gamma.stride(0), dg.stride(0),
                DIRECT_WRITE=True, D_MODEL=d, BLOCK_D=block_d, num_warps=8,
            )
            return dx, dg, db, None, None

        part = torch.empty((n_sc, 2, d), device=x.device, dtype=torch.float32)
        _adaln_p1_bwd_split_kernel[(n_sc,)](
            x, dy, gamma, rstd, meta["sc_blk"], meta["sc_s"], meta["sc_e"],
            dx, part[:, 0], part[:, 1], x.stride(0), gamma.stride(0), part.stride(0),
            DIRECT_WRITE=False, D_MODEL=d, BLOCK_D=block_d, num_warps=8,
        )
        out2 = torch.empty((n_blks, 2, d), device=x.device, dtype=gamma.dtype)
        _reduce_subchunks_kernel[(n_blks, 2)](
            part, out2, meta["blk_sc_s"], meta["blk_sc_e"],
            part.stride(0), part.stride(1), out2.stride(0), out2.stride(1),
            K_PARAMS=2, D_MODEL=d, BLOCK_D=block_d, num_warps=4,
        )
        return dx, out2[:, 0], out2[:, 1], None, None


class SplitGateResAdaLNP2Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, sub1: torch.Tensor, alpha1: torch.Tensor,
        gamma2: torch.Tensor, beta2: torch.Tensor, meta: dict[str, Any], eps: float = 1e-6
    ):
        x, sub1 = x.contiguous(), sub1.contiguous()
        alpha1, gamma2, beta2 = alpha1.contiguous(), gamma2.contiguous(), beta2.contiguous()
        n, d = x.shape
        block_d = triton.next_power_of_2(d)
        x_next = torch.empty_like(x)
        y2 = torch.empty_like(x)
        rstd2 = torch.empty((n,), device=x.device, dtype=torch.float32)
        e127._block_gate_res_adaln_fwd_kernel[(n,)](
            x, sub1, alpha1, gamma2, beta2, meta["block_id"],
            x_next, y2, rstd2, D=d, BLOCK_D=block_d, EPS=eps, num_warps=8,
        )
        ctx.save_for_backward(x_next, sub1, alpha1, gamma2, rstd2)
        ctx.meta = meta
        return x_next, y2

    @staticmethod
    def backward(ctx, dx_next_up: torch.Tensor, dy2: torch.Tensor):
        x_next, sub1, alpha1, gamma2, rstd2 = ctx.saved_tensors
        meta = ctx.meta
        dx_next_up, dy2 = dx_next_up.contiguous(), dy2.contiguous()
        d = x_next.shape[1]
        block_d = triton.next_power_of_2(d)
        dx = torch.empty_like(x_next)
        ds1 = torch.empty_like(sub1)
        n_sc = meta["n_subchunks"]
        n_blks = meta["n_blks"]
        if meta["one_cta_per_blk"]:
            da1 = torch.empty_like(alpha1)
            dg2 = torch.empty_like(gamma2)
            db2 = torch.empty_like(gamma2)
            _adaln_p2_bwd_split_kernel[(n_sc,)](
                x_next, sub1, dx_next_up, dy2, alpha1, gamma2, rstd2,
                meta["sc_blk"], meta["sc_s"], meta["sc_e"],
                dx, ds1, da1, dg2, db2, x_next.stride(0), alpha1.stride(0), da1.stride(0),
                DIRECT_WRITE=True, D_MODEL=d, BLOCK_D=block_d, num_warps=8,
            )
            return dx, ds1, da1, dg2, db2, None, None

        part = torch.empty((n_sc, 3, d), device=x_next.device, dtype=torch.float32)
        _adaln_p2_bwd_split_kernel[(n_sc,)](
            x_next, sub1, dx_next_up, dy2, alpha1, gamma2, rstd2,
            meta["sc_blk"], meta["sc_s"], meta["sc_e"],
            dx, ds1, part[:, 0], part[:, 1], part[:, 2],
            x_next.stride(0), alpha1.stride(0), part.stride(0),
            DIRECT_WRITE=False, D_MODEL=d, BLOCK_D=block_d, num_warps=8,
        )
        out3 = torch.empty((n_blks, 3, d), device=x_next.device, dtype=alpha1.dtype)
        _reduce_subchunks_kernel[(n_blks, 3)](
            part, out3, meta["blk_sc_s"], meta["blk_sc_e"],
            part.stride(0), part.stride(1), out3.stride(0), out3.stride(1),
            K_PARAMS=3, D_MODEL=d, BLOCK_D=block_d, num_warps=4,
        )
        return dx, ds1, out3[:, 0], out3[:, 1], out3[:, 2], None, None


class SplitGateResP3Fn(torch.autograd.Function):
    """Fused Point-3 `x_out = x_next + alpha2[b] * sub2` with zero-copy `dx_next = dx_out` and zero atomics."""

    @staticmethod
    def forward(ctx, x_next: torch.Tensor, sub2: torch.Tensor, alpha2: torch.Tensor, meta: dict[str, Any]):
        x_next, sub2, alpha2 = x_next.contiguous(), sub2.contiguous(), alpha2.contiguous()
        n, d = x_next.shape
        block_d = triton.next_power_of_2(d)
        x_out = torch.empty_like(x_next)
        _adaln_p3_fwd_row_kernel[(n,)](
            x_next, sub2, alpha2, meta["block_id"], x_out,
            x_next.stride(0), alpha2.stride(0), D_MODEL=d, BLOCK_D=block_d, num_warps=8,
        )
        ctx.save_for_backward(sub2, alpha2)
        ctx.meta = meta
        return x_out

    @staticmethod
    def backward(ctx, dx_out: torch.Tensor):
        sub2, alpha2 = ctx.saved_tensors
        meta = ctx.meta
        dx_out = dx_out.contiguous()
        d = sub2.shape[1]
        block_d = triton.next_power_of_2(d)
        ds2 = torch.empty_like(sub2)
        n_sc = meta["n_subchunks"]
        n_blks = meta["n_blks"]
        if meta["one_cta_per_blk"]:
            da2 = torch.empty_like(alpha2)
            _adaln_p3_bwd_split_kernel[(n_sc,)](
                dx_out, sub2, alpha2, meta["sc_blk"], meta["sc_s"], meta["sc_e"],
                ds2, da2, sub2.stride(0), alpha2.stride(0), da2.stride(0),
                DIRECT_WRITE=True, D_MODEL=d, BLOCK_D=block_d, num_warps=8,
            )
            return dx_out, ds2, da2, None

        part = torch.empty((n_sc, 1, d), device=sub2.device, dtype=torch.float32)
        _adaln_p3_bwd_split_kernel[(n_sc,)](
            dx_out, sub2, alpha2, meta["sc_blk"], meta["sc_s"], meta["sc_e"],
            ds2, part[:, 0], sub2.stride(0), alpha2.stride(0), part.stride(0),
            DIRECT_WRITE=False, D_MODEL=d, BLOCK_D=block_d, num_warps=8,
        )
        out1 = torch.empty((n_blks, 1, d), device=sub2.device, dtype=alpha2.dtype)
        _reduce_subchunks_kernel[(n_blks, 1)](
            part, out1, meta["blk_sc_s"], meta["blk_sc_e"],
            part.stride(0), part.stride(1), out1.stride(0), out1.stride(1),
            K_PARAMS=1, D_MODEL=d, BLOCK_D=block_d, num_warps=4,
        )
        return dx_out, ds2, out1[:, 0], None


def staged3_split_adaln_dual_step(
    x: torch.Tensor, sub1: torch.Tensor, sub2: torch.Tensor, cond6: torch.Tensor, meta: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g1, b1, a1, g2, b2, a2 = cond6.unbind(dim=1)
    y1 = SplitAdaLNP1Fn.apply(x, g1, b1, meta, 1e-6)
    xn, y2 = SplitGateResAdaLNP2Fn.apply(x, sub1, a1, g2, b2, meta, 1e-6)
    xo = SplitGateResP3Fn.apply(xn, sub2, a2, meta)
    return y1, y2, xo


@triton.jit
def _adaln_unified_fwd_row_kernel(
    X_ptr, Sub1_ptr, Sub2_ptr, C6_ptr, BlockId_ptr,
    Y1_ptr, Y2_ptr, XOut_ptr, R1_ptr, R2_ptr,
    stride_xn, stride_cb, stride_ck,
    D_MODEL: tl.constexpr, BLOCK_D: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.load(BlockId_ptr + row).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    m = offs_d < D_MODEL
    inv_d = 1.0 / float(D_MODEL)

    cb = C6_ptr + b * stride_cb + offs_d
    g1 = tl.load(cb + 0 * stride_ck, mask=m, other=0.0).to(tl.float32)
    b1 = tl.load(cb + 1 * stride_ck, mask=m, other=0.0).to(tl.float32)
    a1 = tl.load(cb + 2 * stride_ck, mask=m, other=0.0).to(tl.float32)
    g2 = tl.load(cb + 3 * stride_ck, mask=m, other=0.0).to(tl.float32)
    b2 = tl.load(cb + 4 * stride_ck, mask=m, other=0.0).to(tl.float32)
    a2 = tl.load(cb + 5 * stride_ck, mask=m, other=0.0).to(tl.float32)

    x = tl.load(X_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(Sub1_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(Sub2_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)

    r1 = tl.rsqrt(tl.sum(x * x, axis=0) * inv_d + EPS)
    y1 = (x * r1 * (1.0 + g1) + b1).to(Y1_ptr.dtype.element_ty)

    # Exact BF16 intermediate cast boundary matching unfused reference (per GPT-6-Astra audit!)
    xn_bf = (x + a1 * s1).to(Y1_ptr.dtype.element_ty)
    xn = xn_bf.to(tl.float32)
    r2 = tl.rsqrt(tl.sum(xn * xn, axis=0) * inv_d + EPS)
    y2 = (xn * r2 * (1.0 + g2) + b2).to(Y2_ptr.dtype.element_ty)
    xo = (xn + a2 * s2).to(XOut_ptr.dtype.element_ty)

    tl.store(Y1_ptr + row * stride_xn + offs_d, y1, mask=m)
    tl.store(Y2_ptr + row * stride_xn + offs_d, y2, mask=m)
    tl.store(XOut_ptr + row * stride_xn + offs_d, xo, mask=m)
    tl.store(R1_ptr + row, r1)
    tl.store(R2_ptr + row, r2)


@triton.jit
def _adaln_unified_bwd_split_kernel(
    X_ptr, Sub1_ptr, Sub2_ptr, C6_ptr, R1_ptr, R2_ptr,
    dY1_ptr, dY2_ptr, dXO_ptr,
    ScBlk_ptr, ScS_ptr, ScE_ptr,
    dX_ptr, dSub1_ptr, dSub2_ptr, OutC6_ptr,
    stride_xn, stride_cb, stride_ck, stride_out_sc, stride_out_k,
    DIRECT_WRITE: tl.constexpr, D_MODEL: tl.constexpr, BLOCK_D: tl.constexpr,
):
    sc_id = tl.program_id(0).to(tl.int64)
    blk_id = tl.load(ScBlk_ptr + sc_id).to(tl.int64)
    s_row = tl.load(ScS_ptr + sc_id).to(tl.int64)
    e_row = tl.load(ScE_ptr + sc_id).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    m = offs_d < D_MODEL
    inv_d = 1.0 / float(D_MODEL)

    cb = C6_ptr + blk_id * stride_cb + offs_d
    g1 = tl.load(cb + 0 * stride_ck, mask=m, other=0.0).to(tl.float32)
    a1 = tl.load(cb + 2 * stride_ck, mask=m, other=0.0).to(tl.float32)
    g2 = tl.load(cb + 3 * stride_ck, mask=m, other=0.0).to(tl.float32)
    a2 = tl.load(cb + 5 * stride_ck, mask=m, other=0.0).to(tl.float32)

    dg1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    db1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    da1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    dg2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    db2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    da2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for row in tl.range(s_row, e_row, loop_unroll_factor=1):
        x = tl.load(X_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        s1 = tl.load(Sub1_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        s2 = tl.load(Sub2_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        dy1 = tl.load(dY1_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        dy2 = tl.load(dY2_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        dxo = tl.load(dXO_ptr + row * stride_xn + offs_d, mask=m, other=0.0).to(tl.float32)
        r1 = tl.load(R1_ptr + row).to(tl.float32)
        r2 = tl.load(R2_ptr + row).to(tl.float32)

        # Reproduce exact BF16 rounding boundary for x_next so r2 matches saved statistics identically
        xn = (x + a1 * s1).to(dX_ptr.dtype.element_ty).to(tl.float32)
        xn_hat = xn * r2
        x_hat = x * r1

        da2 += dxo * s2
        ds2 = dxo * a2

        dg2 += dy2 * xn_hat
        db2 += dy2
        wdy2 = dy2 * (1.0 + g2)
        c2 = tl.sum(wdy2 * xn_hat, axis=0) * inv_d
        dxn = dxo + r2 * (wdy2 - xn_hat * c2)

        da1 += dxn * s1
        ds1 = dxn * a1

        dg1 += dy1 * x_hat
        db1 += dy1
        wdy1 = dy1 * (1.0 + g1)
        c1 = tl.sum(wdy1 * x_hat, axis=0) * inv_d
        dx = dxn + r1 * (wdy1 - x_hat * c1)

        tl.store(dX_ptr + row * stride_xn + offs_d, dx.to(dX_ptr.dtype.element_ty), mask=m)
        tl.store(dSub1_ptr + row * stride_xn + offs_d, ds1.to(dSub1_ptr.dtype.element_ty), mask=m)
        tl.store(dSub2_ptr + row * stride_xn + offs_d, ds2.to(dSub2_ptr.dtype.element_ty), mask=m)

    dst = blk_id if DIRECT_WRITE else sc_id
    ob = OutC6_ptr + dst * stride_out_sc + offs_d
    tl.store(ob + 0 * stride_out_k, dg1.to(OutC6_ptr.dtype.element_ty), mask=m)
    tl.store(ob + 1 * stride_out_k, db1.to(OutC6_ptr.dtype.element_ty), mask=m)
    tl.store(ob + 2 * stride_out_k, da1.to(OutC6_ptr.dtype.element_ty), mask=m)
    tl.store(ob + 3 * stride_out_k, dg2.to(OutC6_ptr.dtype.element_ty), mask=m)
    tl.store(ob + 4 * stride_out_k, db2.to(OutC6_ptr.dtype.element_ty), mask=m)
    tl.store(ob + 5 * stride_out_k, da2.to(OutC6_ptr.dtype.element_ty), mask=m)


class UnifiedAdaptiveSplitAdaLNFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, sub1: torch.Tensor, sub2: torch.Tensor,
        c6: torch.Tensor, meta: dict[str, Any], force_scratch: bool = False, eps: float = 1e-6,
    ):
        x, sub1, sub2, c6 = x.contiguous(), sub1.contiguous(), sub2.contiguous(), c6.contiguous()
        n, d = x.shape
        block_d = triton.next_power_of_2(d)
        y1 = torch.empty_like(x)
        y2 = torch.empty_like(x)
        xo = torch.empty_like(x)
        r1 = torch.empty((n,), device=x.device, dtype=torch.float32)
        r2 = torch.empty((n,), device=x.device, dtype=torch.float32)
        _adaln_unified_fwd_row_kernel[(n,)](
            x, sub1, sub2, c6, meta["block_id"], y1, y2, xo, r1, r2,
            x.stride(0), c6.stride(0), c6.stride(1),
            D_MODEL=d, BLOCK_D=block_d, EPS=eps, num_warps=8,
        )
        ctx.save_for_backward(x, sub1, sub2, c6, r1, r2)
        ctx.meta = meta
        ctx.force_scratch = force_scratch
        return y1, y2, xo

    @staticmethod
    def backward(ctx, dy1: torch.Tensor, dy2: torch.Tensor, dxo: torch.Tensor):
        x, sub1, sub2, c6, r1, r2 = ctx.saved_tensors
        meta = ctx.meta
        dy1, dy2, dxo = dy1.contiguous(), dy2.contiguous(), dxo.contiguous()
        d = x.shape[1]
        block_d = triton.next_power_of_2(d)
        dx = torch.empty_like(x)
        ds1 = torch.empty_like(sub1)
        ds2 = torch.empty_like(sub2)
        n_sc = meta["n_subchunks"]
        n_blks = meta["n_blks"]
        if meta["one_cta_per_blk"] and not ctx.force_scratch:
            dc6 = torch.empty_like(c6)
            _adaln_unified_bwd_split_kernel[(n_sc,)](
                x, sub1, sub2, c6, r1, r2, dy1, dy2, dxo,
                meta["sc_blk"], meta["sc_s"], meta["sc_e"],
                dx, ds1, ds2, dc6,
                x.stride(0), c6.stride(0), c6.stride(1), dc6.stride(0), dc6.stride(1),
                DIRECT_WRITE=True, D_MODEL=d, BLOCK_D=block_d, num_warps=8,
            )
            return dx, ds1, ds2, dc6, None, None, None

        part6 = torch.empty((n_sc, 6, d), device=x.device, dtype=torch.float32)
        _adaln_unified_bwd_split_kernel[(n_sc,)](
            x, sub1, sub2, c6, r1, r2, dy1, dy2, dxo,
            meta["sc_blk"], meta["sc_s"], meta["sc_e"],
            dx, ds1, ds2, part6,
            x.stride(0), c6.stride(0), c6.stride(1), part6.stride(0), part6.stride(1),
            DIRECT_WRITE=False, D_MODEL=d, BLOCK_D=block_d, num_warps=8,
        )
        dc6 = torch.empty_like(c6)
        _reduce_subchunks_kernel[(n_blks, 6)](
            part6, dc6, meta["blk_sc_s"], meta["blk_sc_e"],
            part6.stride(0), part6.stride(1), dc6.stride(0), dc6.stride(1),
            K_PARAMS=6, D_MODEL=d, BLOCK_D=block_d, num_warps=4,
        )
        return dx, ds1, ds2, dc6, None, None, None


# ============================================================================
# 2. Track B Round-2 Kernels: Zero-Spill Disjoint-Lifetime `BD3` Dual-Stream Attention
# ============================================================================


@triton.jit
def _bd3_ktile_fwd_kernel(
    Qt_ptr, Kt_ptr, Vt_ptr, Q0_ptr, K0_ptr, V0_ptr,
    O_obc_ptr, LSE_obc_ptr, O_bc_ptr, LSE_bc_ptr,
    Ot_out_ptr, LSEt_out_ptr, BlkMeta_ptr,
    N_TOTAL, N_OBC, sm_scale,
    W: tl.constexpr, K_TILE: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    COMPUTE_CLEAN: tl.constexpr,
):
    b_idx = tl.program_id(0).to(tl.int64)
    q_tile = tl.program_id(1)
    h_kv = tl.program_id(2).to(tl.int64)

    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0).to(tl.int64)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    prev_start = tl.load(BlkMeta_ptr + b_idx * 6 + 2).to(tl.int64)
    obc_off = tl.load(BlkMeta_ptr + b_idx * 6 + 4).to(tl.int64)

    q_base = q_tile * Q_STEP
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
    offs_d = tl.arange(0, D)

    qt = tl.load(
        Qt_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :],
        mask=valid_m[:, None], other=0.0,
    )
    m_acc = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    has_prev = prev_start >= 0
    for k_off in tl.range(0, W, K_TILE, loop_unroll_factor=1):
        offs_k = k_off + tl.arange(0, K_TILE)
        valid_kt = offs_k < blk_len
        kt_tok = blk_start + offs_k
        kt = tl.load(Kt_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
        vt = tl.load(Vt_ptr + (kt_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
        st = tl.dot(qt, tl.trans(kt)) * sm_scale
        mask_bd = valid_m[:, None] & valid_kt[None, :]
        st = tl.where(mask_bd, st, -float("inf"))
        m_tile = tl.max(st, axis=1)
        m_new = tl.maximum(m_acc, m_tile)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(m_acc == -float("inf"), 0.0, tl.exp(m_acc - m_new_safe))
        pt = tl.where(mask_bd, tl.exp(st - m_new_safe[:, None]), 0.0)
        l_acc = l_acc * alpha + tl.sum(pt, axis=1)
        o_acc = o_acc * alpha[:, None] + tl.dot(pt.to(vt.dtype), vt)
        m_acc = m_new

        if has_prev:
            valid_kp = offs_k < W
            kp_tok = tl.maximum(prev_start, 0) + offs_k
            kp = tl.load(K0_ptr + (kp_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kp[:, None], other=0.0)
            vp = tl.load(V0_ptr + (kp_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_kp[:, None], other=0.0)
            sp = tl.dot(qt, tl.trans(kp)) * sm_scale
            up_mask = valid_m[:, None] & valid_kp[None, :] & (offs_k[None, :] > q_pos[:, None])
            sp = tl.where(up_mask, sp, -float("inf"))
            m_p = tl.max(sp, axis=1)
            m_new2 = tl.maximum(m_acc, m_p)
            m_new2_safe = tl.where(m_new2 == -float("inf"), 0.0, m_new2)
            alpha2 = tl.where(m_acc == -float("inf"), 0.0, tl.exp(m_acc - m_new2_safe))
            pp = tl.where(up_mask, tl.exp(sp - m_new2_safe[:, None]), 0.0)
            l_acc = l_acc * alpha2 + tl.sum(pp, axis=1)
            o_acc = o_acc * alpha2[:, None] + tl.dot(pp.to(vp.dtype), vp)
            m_acc = m_new2

    m_safe = tl.where(m_acc == -float("inf"), 0.0, m_acc)
    if has_prev:
        obc_u = obc_off + q_pos
        lse_c = tl.load(LSE_obc_ptr + hq_vec * N_OBC + obc_u, mask=valid_m, other=-float("inf")).to(tl.float32)
        o_c = tl.load(O_obc_ptr + (obc_u[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
        lse_micro = tl.where(l_acc > 0.0, m_safe + tl.log(tl.maximum(l_acc, 1e-30)), -float("inf"))
        m_tot = tl.maximum(lse_micro, lse_c)
        m_tot_safe = tl.where(m_tot == -float("inf"), 0.0, m_tot)
        w_micro = tl.where(l_acc > 0.0, tl.exp(m_safe - m_tot_safe), 0.0)
        w_c = tl.where(lse_c == -float("inf"), 0.0, tl.exp(lse_c - m_tot_safe))
        denom = l_acc * w_micro + w_c
        o_final = (o_acc * w_micro[:, None] + o_c * w_c[:, None]) / tl.maximum(denom[:, None], 1e-30)
        lse_final = tl.where(denom > 0.0, m_tot_safe + tl.log(tl.maximum(denom, 1e-30)), 0.0)
    else:
        o_final = o_acc / tl.maximum(l_acc[:, None], 1e-30)
        lse_final = tl.where(l_acc > 0.0, m_safe + tl.log(tl.maximum(l_acc, 1e-30)), 0.0)

    tl.store(Ot_out_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :], o_final.to(Ot_out_ptr.dtype.element_ty), mask=valid_m[:, None])
    tl.store(LSEt_out_ptr + hq_vec * N_TOTAL + q_tok, lse_final, mask=valid_m)

    if COMPUTE_CLEAN:
        has_ubc = valid_m & (q_pos < (blk_len - 1))
        q0 = tl.load(Q0_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :], mask=has_ubc[:, None], other=0.0)
        m0_acc = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l0_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        o0_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        for k_off in tl.range(0, W, K_TILE, loop_unroll_factor=1):
            offs_k = k_off + tl.arange(0, K_TILE)
            valid_k0 = offs_k < blk_len
            k0_tok = blk_start + offs_k
            k0 = tl.load(K0_ptr + (k0_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_k0[:, None], other=0.0)
            v0 = tl.load(V0_ptr + (k0_tok[:, None] * H_KV + h_kv) * D + offs_d[None, :], mask=valid_k0[:, None], other=0.0)
            s0 = tl.dot(q0, tl.trans(k0)) * sm_scale
            ubc_mask = has_ubc[:, None] & valid_k0[None, :] & (offs_k[None, :] > q_pos[:, None])
            s0 = tl.where(ubc_mask, s0, -float("inf"))
            m0_t = tl.max(s0, axis=1)
            m0_new = tl.maximum(m0_acc, m0_t)
            m0_new_safe = tl.where(m0_new == -float("inf"), 0.0, m0_new)
            a0 = tl.where(m0_acc == -float("inf"), 0.0, tl.exp(m0_acc - m0_new_safe))
            p0 = tl.where(ubc_mask, tl.exp(s0 - m0_new_safe[:, None]), 0.0)
            l0_acc = l0_acc * a0 + tl.sum(p0, axis=1)
            o0_acc = o0_acc * a0[:, None] + tl.dot(p0.to(v0.dtype), v0)
            m0_acc = m0_new

        m0_safe = tl.where(m0_acc == -float("inf"), 0.0, m0_acc)
        lse_bc = tl.load(LSE_bc_ptr + hq_vec * N_TOTAL + q_tok, mask=has_ubc, other=-float("inf")).to(tl.float32)
        o_bc = tl.load(O_bc_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :], mask=has_ubc[:, None], other=0.0).to(tl.float32)
        lse0_micro = tl.where(l0_acc > 0.0, m0_safe + tl.log(tl.maximum(l0_acc, 1e-30)), -float("inf"))
        m0_tot = tl.maximum(lse_bc, lse0_micro)
        m0_tot_safe = tl.where(m0_tot == -float("inf"), 0.0, m0_tot)
        w0_micro = tl.where(l0_acc > 0.0, tl.exp(m0_safe - m0_tot_safe), 0.0)
        w0_bc = tl.where(lse_bc == -float("inf"), 0.0, tl.exp(lse_bc - m0_tot_safe))
        denom0 = l0_acc * w0_micro + w0_bc
        o0_final = (o0_acc * w0_micro[:, None] + o_bc * w0_bc[:, None]) / tl.maximum(denom0[:, None], 1e-30)
        lse0_final = tl.where(denom0 > 0.0, m0_tot_safe + tl.log(tl.maximum(denom0, 1e-30)), 0.0)

        tl.store(O_bc_ptr + (q_tok[:, None] * H_Q + hq_vec[:, None]) * D + offs_d[None, :], o0_final.to(O_bc_ptr.dtype.element_ty), mask=has_ubc[:, None])
        tl.store(LSE_bc_ptr + hq_vec * N_TOTAL + q_tok, lse0_final, mask=has_ubc)


@triton.jit
def _bd3_ktile_noisy_bwd_disjoint_kernel(
    Qt_ptr, Kt_ptr, Vt_ptr, K0_ptr, V0_ptr,
    Ot_ptr, LSEt_ptr, dOt_ptr, dQt_ptr, dKt_ptr, dVt_ptr,
    BlkMeta_ptr, N_TOTAL, sm_scale,
    W: tl.constexpr, K_TILE: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
):
    b_idx = tl.program_id(0).to(tl.int64)
    h_kv = tl.program_id(1).to(tl.int64)
    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0).to(tl.int64)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    prev_start = tl.load(BlkMeta_ptr + b_idx * 6 + 2).to(tl.int64)
    has_prev = prev_start >= 0

    BLOCK_M: tl.constexpr = Q_STEP * G_PAD
    offs_m = tl.arange(0, BLOCK_M)
    q_sub = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    hq_vec = h_kv * G + g_idx
    offs_d = tl.arange(0, D)

    for k_off in tl.range(0, W, K_TILE, loop_unroll_factor=1):
        offs_k = k_off + tl.arange(0, K_TILE)
        valid_kt = offs_k < blk_len
        kt_row = ((blk_start + offs_k) * H_KV + h_kv).to(tl.int64) * D
        kt = tl.load(Kt_ptr + kt_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
        vt = tl.load(Vt_ptr + kt_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
        dkt_acc = tl.zeros([K_TILE, D], dtype=tl.float32)
        dvt_acc = tl.zeros([K_TILE, D], dtype=tl.float32)

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

            if k_off == 0:
                tl.store(dQt_ptr + q_row[:, None] + offs_d[None, :], dqt_tile.to(dQt_ptr.dtype.element_ty), mask=valid_m[:, None])
            else:
                dqt_prev = tl.load(dQt_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
                tl.store(dQt_ptr + q_row[:, None] + offs_d[None, :], (dqt_prev + dqt_tile).to(dQt_ptr.dtype.element_ty), mask=valid_m[:, None])

        tl.store(dKt_ptr + kt_row[:, None] + offs_d[None, :], dkt_acc.to(dKt_ptr.dtype.element_ty), mask=valid_kt[:, None])
        tl.store(dVt_ptr + kt_row[:, None] + offs_d[None, :], dvt_acc.to(dVt_ptr.dtype.element_ty), mask=valid_kt[:, None])

    if has_prev:
        for q_base in tl.range(0, W, Q_STEP, loop_unroll_factor=1):
            q_pos = q_base + q_sub
            valid_m = (q_pos < blk_len) & (q_pos < (W - 1)) & (g_idx < G)
            q_tok = blk_start + q_pos
            q_row = (q_tok * H_Q + hq_vec).to(tl.int64) * D
            qt = tl.load(Qt_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0)
            ot = tl.load(Ot_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
            dot = tl.load(dOt_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0)
            lset = tl.load(LSEt_ptr + hq_vec.to(tl.int64) * N_TOTAL + q_tok.to(tl.int64), mask=valid_m, other=0.0).to(tl.float32)
            delta_t = tl.sum(dot.to(tl.float32) * ot, axis=1)
            dqt_acc = tl.load(dQt_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
            for k_off in tl.range(0, W, K_TILE, loop_unroll_factor=1):
                offs_k = k_off + tl.arange(0, K_TILE)
                valid_kp = offs_k < W
                kp_row = ((prev_start + offs_k) * H_KV + h_kv).to(tl.int64) * D
                kp = tl.load(K0_ptr + kp_row[:, None] + offs_d[None, :], mask=valid_kp[:, None], other=0.0)
                vp = tl.load(V0_ptr + kp_row[:, None] + offs_d[None, :], mask=valid_kp[:, None], other=0.0)
                sp = tl.dot(qt, tl.trans(kp)) * sm_scale
                up_mask = valid_m[:, None] & valid_kp[None, :] & (offs_k[None, :] > q_pos[:, None])
                pp = tl.where(up_mask, tl.exp(sp - lset[:, None]), 0.0)
                dp_p = tl.dot(dot, tl.trans(vp))
                ds_p = pp * (dp_p - delta_t[:, None]) * sm_scale
                dqt_acc += tl.dot(ds_p.to(kp.dtype), kp)
            tl.store(dQt_ptr + q_row[:, None] + offs_d[None, :], dqt_acc.to(dQt_ptr.dtype.element_ty), mask=valid_m[:, None])


@triton.jit
def _bd3_ktile_clean_mbc_bwd_kernel(
    Q0_ptr, K0_ptr, V0_ptr, O0_ptr, LSE0_ptr, dO0_ptr,
    dQ0_ptr, dK0_ptr, dV0_ptr, BlkMeta_ptr, N_TOTAL, sm_scale,
    W: tl.constexpr, K_TILE: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
):
    b_idx = tl.program_id(0).to(tl.int64)
    h_kv = tl.program_id(1).to(tl.int64)
    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0).to(tl.int64)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    if blk_len <= 1:
        return

    BLOCK_M: tl.constexpr = Q_STEP * G_PAD
    offs_m = tl.arange(0, BLOCK_M)
    q_sub = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    hq_vec = h_kv * G + g_idx
    offs_d = tl.arange(0, D)

    for k_off in tl.range(0, W, K_TILE, loop_unroll_factor=1):
        offs_k = k_off + tl.arange(0, K_TILE)
        valid_k0 = offs_k < blk_len
        k0_row = ((blk_start + offs_k) * H_KV + h_kv).to(tl.int64) * D
        k0 = tl.load(K0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_k0[:, None], other=0.0)
        v0 = tl.load(V0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_k0[:, None], other=0.0)
        dk0_acc = tl.zeros([K_TILE, D], dtype=tl.float32)
        dv0_acc = tl.zeros([K_TILE, D], dtype=tl.float32)

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
            ubc_mask = has_ubc[:, None] & valid_k0[None, :] & (offs_k[None, :] > q_pos[:, None])
            p0 = tl.where(ubc_mask, tl.exp(s0 - lse0[:, None]), 0.0)
            dv0_acc += tl.dot(tl.trans(p0.to(do0.dtype)), do0)
            dp0 = tl.dot(do0, tl.trans(v0))
            ds0 = p0 * (dp0 - delta0[:, None]) * sm_scale
            dq0_add = tl.dot(ds0.to(k0.dtype), k0)
            dk0_acc += tl.dot(tl.trans(ds0.to(q0.dtype)), q0)

            dq0_prev = tl.load(dQ0_ptr + q_row[:, None] + offs_d[None, :], mask=has_ubc[:, None], other=0.0).to(tl.float32)
            tl.store(dQ0_ptr + q_row[:, None] + offs_d[None, :], (dq0_prev + dq0_add).to(dQ0_ptr.dtype.element_ty), mask=has_ubc[:, None])

        dk0_prev = tl.load(dK0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_k0[:, None], other=0.0).to(tl.float32)
        dv0_prev = tl.load(dV0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_k0[:, None], other=0.0).to(tl.float32)
        tl.store(dK0_ptr + k0_row[:, None] + offs_d[None, :], (dk0_prev + dk0_acc).to(dK0_ptr.dtype.element_ty), mask=valid_k0[:, None])
        tl.store(dV0_ptr + k0_row[:, None] + offs_d[None, :], (dv0_prev + dv0_acc).to(dV0_ptr.dtype.element_ty), mask=valid_k0[:, None])


@triton.jit
def _bd3_ktile_clean_uobc_bwd_kernel(
    Qt_ptr, Ot_ptr, LSEt_ptr, dOt_ptr, K0_ptr, V0_ptr,
    dK0_ptr, dV0_ptr, BlkMeta_ptr, N_TOTAL, sm_scale,
    W: tl.constexpr, K_TILE: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
):
    b_idx = tl.program_id(0).to(tl.int64)
    h_kv = tl.program_id(1).to(tl.int64)
    next_start = tl.load(BlkMeta_ptr + b_idx * 6 + 3).to(tl.int64)
    if next_start < 0:
        return
    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0).to(tl.int64)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    next_len = tl.load(BlkMeta_ptr + b_idx * 6 + 5)

    BLOCK_M: tl.constexpr = Q_STEP * G_PAD
    offs_m = tl.arange(0, BLOCK_M)
    q_sub = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    hq_vec = h_kv * G + g_idx
    offs_d = tl.arange(0, D)

    for k_off in tl.range(0, W, K_TILE, loop_unroll_factor=1):
        offs_k = k_off + tl.arange(0, K_TILE)
        valid_k0 = offs_k < blk_len
        k0_row = ((blk_start + offs_k) * H_KV + h_kv).to(tl.int64) * D
        k0 = tl.load(K0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_k0[:, None], other=0.0)
        v0 = tl.load(V0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_k0[:, None], other=0.0)
        dk0_acc2 = tl.zeros([K_TILE, D], dtype=tl.float32)
        dv0_acc2 = tl.zeros([K_TILE, D], dtype=tl.float32)

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
            uobc_mask = valid_qn[:, None] & valid_k0[None, :] & (offs_k[None, :] > q_pos[:, None])
            pn = tl.where(uobc_mask, tl.exp(sn - lsetn[:, None]), 0.0)
            dv0_acc2 += tl.dot(tl.trans(pn.to(dotn.dtype)), dotn)
            dpn = tl.dot(dotn, tl.trans(v0))
            dsn = pn * (dpn - deltatn[:, None]) * sm_scale
            dk0_acc2 += tl.dot(tl.trans(dsn.to(qtn.dtype)), qtn)

        dk0_prev2 = tl.load(dK0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_k0[:, None], other=0.0).to(tl.float32)
        dv0_prev2 = tl.load(dV0_ptr + k0_row[:, None] + offs_d[None, :], mask=valid_k0[:, None], other=0.0).to(tl.float32)
        tl.store(dK0_ptr + k0_row[:, None] + offs_d[None, :], (dk0_prev2 + dk0_acc2).to(dK0_ptr.dtype.element_ty), mask=valid_k0[:, None])
        tl.store(dV0_ptr + k0_row[:, None] + offs_d[None, :], (dv0_prev2 + dv0_acc2).to(dV0_ptr.dtype.element_ty), mask=valid_k0[:, None])


@triton.jit
def _bd3_ktile32_coupled_spilling_noisy_bwd_kernel(
    Qt_ptr, Kt_ptr, Vt_ptr, K0_ptr, V0_ptr,
    Ot_ptr, LSEt_ptr, dOt_ptr, dQt_ptr, dKt_ptr, dVt_ptr,
    BlkMeta_ptr, N_TOTAL, sm_scale,
    W: tl.constexpr, K_TILE: tl.constexpr, Q_STEP: tl.constexpr,
    G: tl.constexpr, G_PAD: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
):
    """Intermediate RSI Trial 2 candidate (`K_TILE=32`, `num_warps=8` coupled lifetimes -> 454 cubin spills!)."""
    b_idx = tl.program_id(0).to(tl.int64)
    h_kv = tl.program_id(1).to(tl.int64)
    blk_start = tl.load(BlkMeta_ptr + b_idx * 6 + 0).to(tl.int64)
    blk_len = tl.load(BlkMeta_ptr + b_idx * 6 + 1)
    prev_start = tl.load(BlkMeta_ptr + b_idx * 6 + 2).to(tl.int64)
    has_prev = prev_start >= 0
    BLOCK_M: tl.constexpr = Q_STEP * G_PAD
    offs_m = tl.arange(0, BLOCK_M)
    q_sub = offs_m // G_PAD
    g_idx = offs_m % G_PAD
    hq_vec = h_kv * G + g_idx
    offs_d = tl.arange(0, D)

    for k_off in tl.range(0, W, K_TILE, loop_unroll_factor=1):
        offs_k = k_off + tl.arange(0, K_TILE)
        valid_kt = offs_k < blk_len
        kt_row = ((blk_start + offs_k) * H_KV + h_kv).to(tl.int64) * D
        kt = tl.load(Kt_ptr + kt_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
        vt = tl.load(Vt_ptr + kt_row[:, None] + offs_d[None, :], mask=valid_kt[:, None], other=0.0)
        dkt_acc = tl.zeros([K_TILE, D], dtype=tl.float32)
        dvt_acc = tl.zeros([K_TILE, D], dtype=tl.float32)
        valid_kp = offs_k < W
        kp_row = ((tl.maximum(prev_start, 0) + offs_k) * H_KV + h_kv).to(tl.int64) * D
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
                up_mask = valid_m[:, None] & valid_kp[None, :] & (offs_k[None, :] > q_pos[:, None])
                pp = tl.where(up_mask, tl.exp(sp - lset[:, None]), 0.0)
                dp_p = tl.dot(dot, tl.trans(vp))
                ds_p = pp * (dp_p - delta_t[:, None]) * sm_scale
                dqt_tile += tl.dot(ds_p.to(kp.dtype), kp)
            if k_off == 0:
                tl.store(dQt_ptr + q_row[:, None] + offs_d[None, :], dqt_tile.to(dQt_ptr.dtype.element_ty), mask=valid_m[:, None])
            else:
                dqt_prev = tl.load(dQt_ptr + q_row[:, None] + offs_d[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
                tl.store(dQt_ptr + q_row[:, None] + offs_d[None, :], (dqt_prev + dqt_tile).to(dQt_ptr.dtype.element_ty), mask=valid_m[:, None])
        tl.store(dKt_ptr + kt_row[:, None] + offs_d[None, :], dkt_acc.to(dKt_ptr.dtype.element_ty), mask=valid_kt[:, None])
        tl.store(dVt_ptr + kt_row[:, None] + offs_d[None, :], dvt_acc.to(dVt_ptr.dtype.element_ty), mask=valid_kt[:, None])


class Round2KTile16ZeroCopyBD3Fn(torch.autograd.Function):
    """Round-2 Winner (`KEEP`): `K_TILE=16`, `Q_STEP_FWD=8`, `Q_STEP_BWD=4`, disjoint register lifetimes (`0` cubin spills),
    guarded `L > W` zero-copy `OBC` slice (`len(doc_lens)==1`) or block-cat (`len(doc_lens)<=32`), and direct global-(Out_t, LSE_t) FA2 bwd.
    """

    @staticmethod
    def forward(
        ctx, qt: torch.Tensor, kt: torch.Tensor, vt: torch.Tensor,
        q0: torch.Tensor | None, k0: torch.Tensor, v0: torch.Tensor,
        meta: dict[str, Any], compute_clean_out: bool = True, spilling_trial2_mode: bool = False,
    ):
        qt, kt, vt = qt.contiguous(), kt.contiguous(), vt.contiguous()
        k0, v0 = k0.contiguous(), v0.contiguous()
        n, h_q, d = qt.shape
        h_kv = kt.shape[1]
        g = h_q // h_kv
        g_pad = max(4, triton.next_power_of_2(g))
        W = meta["W"]
        k_tile = (32 if W >= 32 else 16) if spilling_trial2_mode else 16
        q_step_fwd = min(W, max(4, 64 // g_pad))
        n_q_tiles = (W + q_step_fwd - 1) // q_step_fwd
        sm = d ** -0.5

        n_obc = meta["n_obc"]
        doc_lens = meta["doc_lens"]
        single_doc_zerocopy = (len(doc_lens) == 1) and (doc_lens[0] > W) and (not spilling_trial2_mode)
        slice_cat_mode = (not single_doc_zerocopy) and (len(doc_lens) == 2) and (not spilling_trial2_mode)

        if n_obc > 0:
            if single_doc_zerocopy:
                L0 = doc_lens[0]
                q_obc = qt[W:L0]
                k_obc = k0[: L0 - W]
                v_obc = v0[: L0 - W]
            elif slice_cat_mode:
                q_c, k_c, v_c = [], [], []
                cur = 0
                for L in doc_lens:
                    if L > W:
                        q_c.append(qt[cur + W : cur + L])
                        k_c.append(k0[cur : cur + L - W])
                        v_c.append(v0[cur : cur + L - W])
                    cur += L
                q_obc = torch.cat(q_c, dim=0)
                k_obc = torch.cat(k_c, dim=0)
                v_obc = torch.cat(v_c, dim=0)
            else:
                q_obc = torch.empty((n_obc, h_q, d), device=qt.device, dtype=qt.dtype)
                k_obc = torch.empty((n_obc, h_kv, d), device=kt.device, dtype=kt.dtype)
                v_obc = torch.empty((n_obc, h_kv, d), device=vt.device, dtype=vt.dtype)
                e127._rev_gather_qkv_masked_kernel[(n_obc, h_q + 2 * h_kv)](
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
        if (W <= 32) and (not spilling_trial2_mode):
            block_n_fast = max(16, triton.next_power_of_2(W))
            e127._bd3_gqa_microblock_fwd_kernel[(meta["n_blks"], n_q_tiles, h_kv)](
                qt, kt, vt, q0 if compute_clean_out else qt, k0, v0,
                o_obc, lse_obc, o0, lse0, ot, lset,
                meta["blk_meta"], n, max(1, n_obc), sm,
                W=W, BLOCK_N=block_n_fast, Q_STEP=q_step_fwd,
                G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                COMPUTE_CLEAN=compute_clean_out,
                num_warps=4, num_stages=2,
            )
        else:
            _bd3_ktile_fwd_kernel[(meta["n_blks"], n_q_tiles, h_kv)](
                qt, kt, vt, q0 if compute_clean_out else qt, k0, v0,
                o_obc, lse_obc, o0, lse0, ot, lset,
                meta["blk_meta"], n, max(1, n_obc), sm,
                W=W, K_TILE=k_tile, Q_STEP=q_step_fwd,
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
        ctx.single_doc_zerocopy = single_doc_zerocopy
        ctx.slice_cat_mode = slice_cat_mode
        ctx.spilling_trial2_mode = spilling_trial2_mode
        return ot, o0

    @staticmethod
    def backward(ctx, dot: torch.Tensor, do0: torch.Tensor):
        meta = ctx.meta
        W = meta["W"]
        doc_lens = meta["doc_lens"]
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
        if ctx.spilling_trial2_mode:
            k_tile_b = 32 if W >= 32 else 16
            q_step_b = min(W, max(4, 64 // g_pad))
            b_warps = 8 if W >= 64 else 4
            _bd3_ktile32_coupled_spilling_noisy_bwd_kernel[(meta["n_blks"], h_kv)](
                qt, kt, vt, k0, v0, ot, lset, dot, dqt, dkt, dvt,
                meta["blk_meta"], n, sm,
                W=W, K_TILE=k_tile_b, Q_STEP=q_step_b,
                G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                num_warps=b_warps, num_stages=1,
            )
            if ctx.compute_clean_out:
                _bd3_ktile_clean_mbc_bwd_kernel[(meta["n_blks"], h_kv)](
                    q0, k0, v0, o0, lse0, do0, dq0, dk0, dv0,
                    meta["blk_meta"], n, sm,
                    W=W, K_TILE=k_tile_b, Q_STEP=min(W, 4),
                    G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                    num_warps=4, num_stages=1,
                )
            _bd3_ktile_clean_uobc_bwd_kernel[(meta["n_blks"], h_kv)](
                qt, ot, lset, dot, k0, v0, dk0, dv0,
                meta["blk_meta"], n, sm,
                W=W, K_TILE=k_tile_b, Q_STEP=min(W, 4),
                G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                num_warps=4, num_stages=1,
            )
        else:
            if W <= 32:
                block_n_fast = max(16, triton.next_power_of_2(W))
                q_step_fast = min(W, max(4, 64 // g_pad))
                e127._bd3_gqa_noisy_bwd_kernel[(meta["n_blks"], h_kv)](
                    qt, kt, vt, k0, v0, ot, lset, dot, dqt, dkt, dvt,
                    meta["blk_meta"], n, sm,
                    W=W, BLOCK_N=block_n_fast, Q_STEP=q_step_fast,
                    G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                    num_warps=4, num_stages=1,
                )
                e127._bd3_gqa_clean_bwd_kernel[(meta["n_blks"], h_kv)](
                    qt, ot, lset, dot, q0, k0, v0, o0, lse0, do0,
                    dq0 if ctx.compute_clean_out else dqt, dk0, dv0,
                    meta["blk_meta"], n, sm,
                    W=W, BLOCK_N=block_n_fast, Q_STEP=q_step_fast,
                    G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                    COMPUTE_CLEAN=ctx.compute_clean_out,
                    num_warps=4, num_stages=1,
                )
            else:
                q_step_b = min(W, 4)
                _bd3_ktile_noisy_bwd_disjoint_kernel[(meta["n_blks"], h_kv)](
                    qt, kt, vt, k0, v0, ot, lset, dot, dqt, dkt, dvt,
                    meta["blk_meta"], n, sm,
                    W=W, K_TILE=16, Q_STEP=q_step_b,
                    G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                    num_warps=4, num_stages=1,
                )
                if ctx.compute_clean_out:
                    _bd3_ktile_clean_mbc_bwd_kernel[(meta["n_blks"], h_kv)](
                        q0, k0, v0, o0, lse0, do0, dq0, dk0, dv0,
                        meta["blk_meta"], n, sm,
                        W=W, K_TILE=16, Q_STEP=q_step_b,
                        G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                        num_warps=4, num_stages=1,
                    )
                _bd3_ktile_clean_uobc_bwd_kernel[(meta["n_blks"], h_kv)](
                    qt, ot, lset, dot, k0, v0, dk0, dv0,
                    meta["blk_meta"], n, sm,
                    W=W, K_TILE=16, Q_STEP=q_step_b,
                    G=g, G_PAD=g_pad, H_Q=h_q, H_KV=h_kv, D=d,
                    num_warps=4, num_stages=1,
                )

        n_obc = meta["n_obc"]
        if n_obc > 0:
            if ctx.single_doc_zerocopy:
                L0 = doc_lens[0]
                do_obc = dot[W:L0]
                mo_obc = ot[W:L0]
                ml_obc = lset[:, W:L0].contiguous()
                dq_obc = torch.empty_like(q_obc)
                dk_obc = torch.empty_like(k_obc)
                dv_obc = torch.empty_like(v_obc)
                _flash_attn_varlen_backward(
                    dout=do_obc, q=q_obc, k=k_obc, v=v_obc, out=mo_obc, softmax_lse=ml_obc,
                    dq=dq_obc, dk=dk_obc, dv=dv_obc,
                    cu_seqlens_q=meta["cu_obc"], cu_seqlens_k=meta["cu_obc"],
                    max_seqlen_q=meta["max_obc"], max_seqlen_k=meta["max_obc"],
                    dropout_p=0.0, softmax_scale=sm, causal=True,
                    window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
                )
                dqt[W:L0].add_(dq_obc)
                dk0[: L0 - W].add_(dk_obc)
                dv0[: L0 - W].add_(dv_obc)
            elif ctx.slice_cat_mode:
                do_c, mo_c, ml_c = [], [], []
                cur = 0
                for L in doc_lens:
                    if L > W:
                        do_c.append(dot[cur + W : cur + L])
                        mo_c.append(ot[cur + W : cur + L])
                        ml_c.append(lset[:, cur + W : cur + L])
                    cur += L
                do_obc = torch.cat(do_c, dim=0)
                mo_obc = torch.cat(mo_c, dim=0)
                ml_obc = torch.cat(ml_c, dim=1).contiguous()
                dq_obc = torch.empty_like(q_obc)
                dk_obc = torch.empty_like(k_obc)
                dv_obc = torch.empty_like(v_obc)
                _flash_attn_varlen_backward(
                    dout=do_obc, q=q_obc, k=k_obc, v=v_obc, out=mo_obc, softmax_lse=ml_obc,
                    dq=dq_obc, dk=dk_obc, dv=dv_obc,
                    cu_seqlens_q=meta["cu_obc"], cu_seqlens_k=meta["cu_obc"],
                    max_seqlen_q=meta["max_obc"], max_seqlen_k=meta["max_obc"],
                    dropout_p=0.0, softmax_scale=sm, causal=True,
                    window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=False,
                )
                cur = 0
                off = 0
                for L in doc_lens:
                    if L > W:
                        ln = L - W
                        dqt[cur + W : cur + L].add_(dq_obc[off : off + ln])
                        dk0[cur : cur + ln].add_(dk_obc[off : off + ln])
                        dv0[cur : cur + ln].add_(dv_obc[off : off + ln])
                        off += ln
                    cur += L
            else:
                do_obc = torch.empty_like(q_obc)
                mo_obc = torch.empty_like(q_obc)
                ml_obc = torch.empty((h_q, n_obc), device=qt.device, dtype=torch.float32)
                e127._gather_bwd_indexed_masked_kernel[(n_obc, h_q)](
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
                e127._scatter_add_rev_grads_masked_kernel[(n_obc, h_q + 2 * h_kv)](
                    dqt, dk0, dv0, dq_obc, dk_obc, dv_obc, meta["obc_pair"], H_Q=h_q, H_KV=h_kv, D=d
                )
        return dqt, dkt, dvt, dq0, dk0, dv0, None, None, None


def naive_autograd_fa2_lse_merge_bd3(
    qt: torch.Tensor, kt: torch.Tensor, vt: torch.Tensor,
    q0: torch.Tensor, k0: torch.Tensor, v0: torch.Tensor,
    meta: dict[str, Any], compute_clean_out: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Realistic broken RSI candidate (`Trial 1`): merges PyTorch autograd `flash_attn_varlen_func(..., return_attn_probs=True)`
    outputs via `torch.logaddexp`. `FlashAttnVarlenFunc.backward` silently ignores `grad_softmax_lse`, corrupting `dQ`/`dK` by ~22%!
    """
    sm = qt.shape[-1] ** -0.5
    o_bd, lse_bd, _ = flash_attn_varlen_func(
        qt, kt, vt, meta["cu_blks"], meta["cu_blks"], meta["max_blk"], meta["max_blk"],
        softmax_scale=sm, causal=False, return_attn_probs=True,
    )
    o0, _, _ = flash_attn_varlen_func(
        q0, k0, v0, meta["cu_docs"], meta["cu_docs"], meta["max_doc"], meta["max_doc"],
        softmax_scale=sm, causal=True, return_attn_probs=True,
    )
    if meta["n_obc"] == 0:
        return o_bd, o0
    # Naive block-shifted causal FA2 merge on first document / packed segments (drops upper_OBC AND drops grad_softmax_lse!)
    q_parts, k_parts, v_parts = [], [], []
    W = meta["W"]
    cur = 0
    for L in meta["doc_lens"]:
        if L > W:
            q_parts.append(qt[cur + W : cur + L])
            k_parts.append(k0[cur : cur + L - W])
            v_parts.append(v0[cur : cur + L - W])
        cur += L
    qo = torch.cat(q_parts, dim=0)
    ko = torch.cat(k_parts, dim=0)
    vo = torch.cat(v_parts, dim=0)
    o_ob, lse_ob, _ = flash_attn_varlen_func(
        qo, ko, vo, meta["cu_obc"], meta["cu_obc"], meta["max_obc"], meta["max_obc"],
        softmax_scale=sm, causal=True, return_attn_probs=True,
    )
    ot_list = []
    cur = 0
    off = 0
    l_bd_t = lse_bd.transpose(0, 1).float()
    l_ob_t = lse_ob.transpose(0, 1).float()
    for L in meta["doc_lens"]:
        head_len = min(W, L)
        ot_list.append(o_bd[cur : cur + head_len])
        if L > W:
            ln = L - W
            la = l_bd_t[cur + W : cur + L]
            lb = l_ob_t[off : off + ln]
            ltot = torch.logaddexp(la, lb)
            wa = torch.exp(la - ltot).unsqueeze(-1)
            wb = torch.exp(lb - ltot).unsqueeze(-1)
            merged = (wa * o_bd[cur + W : cur + L].float() + wb * o_ob[off : off + ln].float()).to(qt.dtype)
            ot_list.append(merged)
            off += ln
        cur += L
    return torch.cat(ot_list, dim=0), o0


def inspect_bd3_cubin_spills() -> dict[str, Any]:
    report: dict[str, Any] = {}
    max_spills = 0
    for kname, kobj in (
        ("fwd", _bd3_ktile_fwd_kernel),
        ("noisy_bwd_disjoint", _bd3_ktile_noisy_bwd_disjoint_kernel),
        ("clean_mbc_bwd_disjoint", _bd3_ktile_clean_mbc_bwd_kernel),
        ("clean_uobc_bwd_disjoint", _bd3_ktile_clean_uobc_bwd_kernel),
    ):
        spills: list[int] = []
        regs: list[int] = []
        for dev_entry in getattr(kobj, "device_caches", {}).values():
            cache_dict = dev_entry[0] if isinstance(dev_entry, tuple) else dev_entry
            for bin_v in cache_dict.values():
                s = int(getattr(bin_v, "n_spills", 0))
                r = int(getattr(bin_v, "n_regs", 0))
                spills.append(s)
                regs.append(r)
                max_spills = max(max_spills, s)
        report[kname] = {"n_spills": spills, "n_regs": regs}
    report["max_disjoint_spills"] = max_spills

    spilling_t2: list[int] = []
    for dev_entry in getattr(_bd3_ktile32_coupled_spilling_noisy_bwd_kernel, "device_caches", {}).values():
        cache_dict = dev_entry[0] if isinstance(dev_entry, tuple) else dev_entry
        for bin_v in cache_dict.values():
            spilling_t2.append(int(getattr(bin_v, "n_spills", 0)))
    report["trial2_coupled_warps8_spills"] = spilling_t2
    return report


# ============================================================================
# 3. Track C Round-2 Backbone Kernels: `FusedSwiGLU` & Zero-Clone `QKV-Split-RoPE`
# ============================================================================


@triton.jit
def _swiglu_fwd_kernel(GateU_ptr, Out_ptr, stride_gu_n, stride_o_n, I_DIM: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    m = col < I_DIM
    g = tl.load(GateU_ptr + row * stride_gu_n + col, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GateU_ptr + row * stride_gu_n + I_DIM + col, mask=m, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    out = (g * sig) * u
    tl.store(Out_ptr + row * stride_o_n + col, out.to(Out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _swiglu_bwd_kernel(
    GateU_ptr, dOut_ptr, dGateU_ptr, stride_gu_n, stride_do_n, stride_dgu_n, I_DIM: tl.constexpr, BLOCK_I: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    m = col < I_DIM
    g = tl.load(GateU_ptr + row * stride_gu_n + col, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GateU_ptr + row * stride_gu_n + I_DIM + col, mask=m, other=0.0).to(tl.float32)
    do = tl.load(dOut_ptr + row * stride_do_n + col, mask=m, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu_g = g * sig
    du = do * silu_g
    dg = do * u * sig * (1.0 + g * (1.0 - sig))
    tl.store(dGateU_ptr + row * stride_dgu_n + col, dg.to(dGateU_ptr.dtype.element_ty), mask=m)
    tl.store(dGateU_ptr + row * stride_dgu_n + I_DIM + col, du.to(dGateU_ptr.dtype.element_ty), mask=m)


class FusedSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        orig_shape = gate_up.shape
        i_dim = orig_shape[-1] // 2
        gu2d = gate_up.contiguous().view(-1, 2 * i_dim)
        n = gu2d.shape[0]
        out = torch.empty((n, i_dim), device=gu2d.device, dtype=gu2d.dtype)
        block_i = 1024
        _swiglu_fwd_kernel[(n, triton.cdiv(i_dim, block_i))](
            gu2d, out, gu2d.stride(0), out.stride(0), I_DIM=i_dim, BLOCK_I=block_i, num_warps=4
        )
        ctx.save_for_backward(gu2d)
        ctx.orig_shape = orig_shape
        ctx.i_dim = i_dim
        return out.view(*orig_shape[:-1], i_dim)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (gu2d,) = ctx.saved_tensors
        i_dim = ctx.i_dim
        do2d = dout.contiguous().view(-1, i_dim)
        n = gu2d.shape[0]
        dgu = torch.empty_like(gu2d)
        block_i = 1024
        _swiglu_bwd_kernel[(n, triton.cdiv(i_dim, block_i))](
            gu2d, do2d, dgu, gu2d.stride(0), do2d.stride(0), dgu.stride(0), I_DIM=i_dim, BLOCK_I=block_i, num_warps=4
        )
        return dgu.view(ctx.orig_shape)


@triton.jit
def _qkv_split_rope_fwd_kernel(
    QKV_ptr, Cos_ptr, Sin_ptr, QOut_ptr, KOut_ptr, VOut_ptr,
    N_CLEAN, stride_qkv_n,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D_HALF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h_id = tl.program_id(1).to(tl.int64)
    tok_pos = row % N_CLEAN
    d_full: tl.constexpr = 2 * D_HALF
    offs = tl.arange(0, D_HALF)
    base_in = row * stride_qkv_n

    if h_id < H_Q:
        cos = tl.load(Cos_ptr + tok_pos * D_HALF + offs).to(tl.float32)
        sin = tl.load(Sin_ptr + tok_pos * D_HALF + offs).to(tl.float32)
        off_h = h_id * d_full
        x1 = tl.load(QKV_ptr + base_in + off_h + offs).to(tl.float32)
        x2 = tl.load(QKV_ptr + base_in + off_h + D_HALF + offs).to(tl.float32)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        out_base = (row * H_Q + h_id) * d_full
        tl.store(QOut_ptr + out_base + offs, y1.to(QOut_ptr.dtype.element_ty))
        tl.store(QOut_ptr + out_base + D_HALF + offs, y2.to(QOut_ptr.dtype.element_ty))
    elif h_id < H_Q + H_KV:
        hk = h_id - H_Q
        cos = tl.load(Cos_ptr + tok_pos * D_HALF + offs).to(tl.float32)
        sin = tl.load(Sin_ptr + tok_pos * D_HALF + offs).to(tl.float32)
        off_h = (H_Q + hk) * d_full
        x1 = tl.load(QKV_ptr + base_in + off_h + offs).to(tl.float32)
        x2 = tl.load(QKV_ptr + base_in + off_h + D_HALF + offs).to(tl.float32)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        out_base = (row * H_KV + hk) * d_full
        tl.store(KOut_ptr + out_base + offs, y1.to(KOut_ptr.dtype.element_ty))
        tl.store(KOut_ptr + out_base + D_HALF + offs, y2.to(KOut_ptr.dtype.element_ty))
    else:
        hv = h_id - H_Q - H_KV
        off_h = (H_Q + H_KV + hv) * d_full
        v1 = tl.load(QKV_ptr + base_in + off_h + offs)
        v2 = tl.load(QKV_ptr + base_in + off_h + D_HALF + offs)
        out_base = (row * H_KV + hv) * d_full
        tl.store(VOut_ptr + out_base + offs, v1)
        tl.store(VOut_ptr + out_base + D_HALF + offs, v2)


@triton.jit
def _qkv_split_rope_bwd_kernel(
    dQ_ptr, dK_ptr, dV_ptr, Cos_ptr, Sin_ptr, dQKV_ptr,
    N_CLEAN, stride_dqkv_n,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D_HALF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h_id = tl.program_id(1).to(tl.int64)
    tok_pos = row % N_CLEAN
    d_full: tl.constexpr = 2 * D_HALF
    offs = tl.arange(0, D_HALF)
    base_out = row * stride_dqkv_n

    if h_id < H_Q:
        cos = tl.load(Cos_ptr + tok_pos * D_HALF + offs).to(tl.float32)
        sin = -tl.load(Sin_ptr + tok_pos * D_HALF + offs).to(tl.float32)
        in_base = (row * H_Q + h_id) * d_full
        dy1 = tl.load(dQ_ptr + in_base + offs).to(tl.float32)
        dy2 = tl.load(dQ_ptr + in_base + D_HALF + offs).to(tl.float32)
        dx1 = dy1 * cos - dy2 * sin
        dx2 = dy1 * sin + dy2 * cos
        off_h = h_id * d_full
        tl.store(dQKV_ptr + base_out + off_h + offs, dx1.to(dQKV_ptr.dtype.element_ty))
        tl.store(dQKV_ptr + base_out + off_h + D_HALF + offs, dx2.to(dQKV_ptr.dtype.element_ty))
    elif h_id < H_Q + H_KV:
        hk = h_id - H_Q
        cos = tl.load(Cos_ptr + tok_pos * D_HALF + offs).to(tl.float32)
        sin = -tl.load(Sin_ptr + tok_pos * D_HALF + offs).to(tl.float32)
        in_base = (row * H_KV + hk) * d_full
        dy1 = tl.load(dK_ptr + in_base + offs).to(tl.float32)
        dy2 = tl.load(dK_ptr + in_base + D_HALF + offs).to(tl.float32)
        dx1 = dy1 * cos - dy2 * sin
        dx2 = dy1 * sin + dy2 * cos
        off_h = (H_Q + hk) * d_full
        tl.store(dQKV_ptr + base_out + off_h + offs, dx1.to(dQKV_ptr.dtype.element_ty))
        tl.store(dQKV_ptr + base_out + off_h + D_HALF + offs, dx2.to(dQKV_ptr.dtype.element_ty))
    else:
        hv = h_id - H_Q - H_KV
        in_base = (row * H_KV + hv) * d_full
        dv1 = tl.load(dV_ptr + in_base + offs)
        dv2 = tl.load(dV_ptr + in_base + D_HALF + offs)
        off_h = (H_Q + H_KV + hv) * d_full
        tl.store(dQKV_ptr + base_out + off_h + offs, dv1)
        tl.store(dQKV_ptr + base_out + off_h + D_HALF + offs, dv2)


class DualStreamQKVSplitRoPEFn(torch.autograd.Function):
    """Splits packed `qkv_cat [M_rows, (H_Q + 2*H_KV)*D_h]` (`M_rows = N` or `2*N`) into `(Q, K, V)` with in-flight RoPE and zero clones."""

    @staticmethod
    def forward(ctx, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, h_q: int, h_kv: int, head_dim: int):
        qkv = qkv.contiguous()
        m_rows = qkv.shape[0]
        n_clean = cos.shape[0]
        q_out = torch.empty((m_rows, h_q, head_dim), device=qkv.device, dtype=qkv.dtype)
        k_out = torch.empty((m_rows, h_kv, head_dim), device=qkv.device, dtype=qkv.dtype)
        v_out = torch.empty((m_rows, h_kv, head_dim), device=qkv.device, dtype=qkv.dtype)
        _qkv_split_rope_fwd_kernel[(m_rows, h_q + 2 * h_kv)](
            qkv, cos, sin, q_out, k_out, v_out, n_clean, qkv.stride(0),
            H_Q=h_q, H_KV=h_kv, D_HALF=head_dim // 2, num_warps=2,
        )
        ctx.save_for_backward(cos, sin)
        ctx.h_q = h_q
        ctx.h_kv = h_kv
        ctx.head_dim = head_dim
        return q_out, k_out, v_out

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
        m_rows = dq.shape[0]
        n_clean = cos.shape[0]
        h_q, h_kv, head_dim = ctx.h_q, ctx.h_kv, ctx.head_dim
        dqkv = torch.empty((m_rows, (h_q + 2 * h_kv) * head_dim), device=dq.device, dtype=dq.dtype)
        _qkv_split_rope_bwd_kernel[(m_rows, h_q + 2 * h_kv)](
            dq, dk, dv, cos, sin, dqkv, n_clean, dqkv.stride(0),
            H_Q=h_q, H_KV=h_kv, D_HALF=head_dim // 2, num_warps=2,
        )
        return dqkv, None, None, None, None, None
