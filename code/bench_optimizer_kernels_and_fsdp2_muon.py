from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
import triton
import triton.language as tl

from common import environment_metadata

try:
    import pynvml  # type: ignore

    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False


# ============================================================================
# 1. Multi-Tensor Pointer-Table Triton Kernels (Clip-Norm + Low-Bit/SR AdamW)
# ============================================================================


@triton.jit
def _mt_grad_norm_sq_kernel(
    g_ptrs_ptr,
    numels_ptr,
    cum_blocks_ptr,
    sq_sum_ptr,
    n_tensors: tl.int32,
    G_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-launch multi-tensor L2^2 reduction across arbitrary variable-sized
    gradient tensors via 9-step binary search over `cum_blocks` + CTA FP32 sum."""
    pid = tl.program_id(0)

    lo = 0
    hi = n_tensors - 1
    for _ in tl.static_range(0, 9):
        mid = (lo + hi) // 2
        cb = tl.load(cum_blocks_ptr + mid)
        if pid < cb:
            hi = mid
        else:
            lo = mid + 1

    t_idx = lo
    if t_idx >= n_tensors:
        return
    prev_b = 0 if t_idx == 0 else tl.load(cum_blocks_ptr + t_idx - 1)
    local_block = pid - prev_b

    g_addr = tl.load(g_ptrs_ptr + t_idx)
    numel = tl.load(numels_ptr + t_idx)
    offs = (local_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offs < numel

    if G_BF16:
        g_p = g_addr.to(tl.pointer_type(tl.bfloat16))
    else:
        g_p = g_addr.to(tl.pointer_type(tl.float32))

    g = tl.load(g_p + offs, mask=mask, other=0.0).to(tl.float32)
    cta_sq = tl.sum(g * g, axis=0)
    tl.atomic_add(sq_sum_ptr, cta_sq)


@triton.jit
def _mt_adamw_dense_inclip_kernel(
    meta_ptr,          # int64 [N_tensors, 5]: w_ptr, g_ptr, m_ptr, v_ptr, numel
    cum_blocks_ptr,    # int32 [N_tensors]
    sq_sum_ptr,        # float32 [1]: global L2^2 gradient norm
    max_norm,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bc1_inv,
    bc2_inv,
    seed: tl.int32,
    n_tensors: tl.int32,
    W_BF16: tl.constexpr,
    G_BF16: tl.constexpr,
    MV_BF16: tl.constexpr,
    USE_SR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-launch multi-tensor AdamW with in-register global gradient clipping
    (`clip_coef` computed directly from `sq_sum_ptr` with zero host sync and zero
    in-place `grad *= clip_coef` HBM write+read) and optional unbiased Philox-32
    Stochastic Rounding (`FP32 -> BF16`) on master weights."""
    pid = tl.program_id(0)

    lo = 0
    hi = n_tensors - 1
    for _ in tl.static_range(0, 9):
        mid = (lo + hi) // 2
        cb = tl.load(cum_blocks_ptr + mid)
        if pid < cb:
            hi = mid
        else:
            lo = mid + 1

    t_idx = lo
    if t_idx >= n_tensors:
        return
    prev_b = 0 if t_idx == 0 else tl.load(cum_blocks_ptr + t_idx - 1)
    local_block = pid - prev_b

    base_meta = meta_ptr + t_idx * 5
    w_addr = tl.load(base_meta + 0)
    g_addr = tl.load(base_meta + 1)
    m_addr = tl.load(base_meta + 2)
    v_addr = tl.load(base_meta + 3)
    numel = tl.load(base_meta + 4)

    offs = (local_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offs < numel

    total_norm = tl.sqrt(tl.load(sq_sum_ptr))
    clip_c = tl.minimum(1.0, max_norm / (total_norm + 1e-6))

    if W_BF16:
        w_p = w_addr.to(tl.pointer_type(tl.bfloat16))
    else:
        w_p = w_addr.to(tl.pointer_type(tl.float32))

    if G_BF16:
        g_p = g_addr.to(tl.pointer_type(tl.bfloat16))
    else:
        g_p = g_addr.to(tl.pointer_type(tl.float32))

    if MV_BF16:
        m_p = m_addr.to(tl.pointer_type(tl.bfloat16))
        v_p = v_addr.to(tl.pointer_type(tl.bfloat16))
    else:
        m_p = m_addr.to(tl.pointer_type(tl.float32))
        v_p = v_addr.to(tl.pointer_type(tl.float32))

    w = tl.load(w_p + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_p + offs, mask=mask, other=0.0).to(tl.float32) * clip_c
    m = tl.load(m_p + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_p + offs, mask=mask, other=0.0).to(tl.float32)

    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g * g)

    m_hat = m_new * bc1_inv
    v_hat = v_new * bc2_inv
    w_new = w * (1.0 - lr * weight_decay) - lr * (m_hat / (tl.sqrt(v_hat) + eps))

    if W_BF16:
        if USE_SR:
            rng_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            r_u32 = tl.randint(seed, rng_idx).to(tl.uint32)
            w_bits = w_new.to(tl.uint32, bitcast=True)
            w_sr_bits = (w_bits + (r_u32 & 0xFFFF)) & 0xFFFF0000
            w_sr_f32 = w_sr_bits.to(tl.float32, bitcast=True)
            tl.store(w_p + offs, w_sr_f32.to(tl.bfloat16), mask=mask)
        else:
            tl.store(w_p + offs, w_new.to(tl.bfloat16), mask=mask)
    else:
        tl.store(w_p + offs, w_new, mask=mask)

    if MV_BF16:
        tl.store(m_p + offs, m_new.to(tl.bfloat16), mask=mask)
        tl.store(v_p + offs, v_new.to(tl.bfloat16), mask=mask)
    else:
        tl.store(m_p + offs, m_new, mask=mask)
        tl.store(v_p + offs, v_new, mask=mask)


@triton.jit
def _mt_adamw_int8mv_inclip_kernel(
    meta_ptr,          # int64 [N_tensors, 7]: w_ptr, g_ptr, qm_ptr, qsv_ptr, sm_ptr, ssv_ptr, numel
    cum_blocks_ptr,    # int32 [N_tensors], each block = 1024 elems = 8 x 128 quant blocks
    sq_sum_ptr,        # float32 [1]
    max_norm,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bc1_inv,
    sqrt_bc2_inv,
    n_tensors: tl.int32,
    G_BF16: tl.constexpr,
):
    """Single-launch multi-tensor 8-bit AdamW (`FP32` master weights + 128-element
    blockwise dynamic `INT8` `m` + square-root-compressed `UINT8` `sqrt(v)` +
    `FP32` block absmax scales, `2.0625 bytes/param` moments = `3.88x` compression)
    with in-register global gradient clipping."""
    pid = tl.program_id(0)

    lo = 0
    hi = n_tensors - 1
    for _ in tl.static_range(0, 9):
        mid = (lo + hi) // 2
        cb = tl.load(cum_blocks_ptr + mid)
        if pid < cb:
            hi = mid
        else:
            lo = mid + 1

    t_idx = lo
    if t_idx >= n_tensors:
        return
    prev_b = 0 if t_idx == 0 else tl.load(cum_blocks_ptr + t_idx - 1)
    local_block = pid - prev_b  # Each CTA covers 8 x 128 = 1024 elements

    base_meta = meta_ptr + t_idx * 7
    w_addr = tl.load(base_meta + 0)
    g_addr = tl.load(base_meta + 1)
    qm_addr = tl.load(base_meta + 2)
    qsv_addr = tl.load(base_meta + 3)
    sm_addr = tl.load(base_meta + 4)
    ssv_addr = tl.load(base_meta + 5)
    numel = tl.load(base_meta + 6)

    w_p = w_addr.to(tl.pointer_type(tl.float32))
    g_p = (
        g_addr.to(tl.pointer_type(tl.bfloat16))
        if G_BF16
        else g_addr.to(tl.pointer_type(tl.float32))
    )
    qm_p = qm_addr.to(tl.pointer_type(tl.int8))
    qsv_p = qsv_addr.to(tl.pointer_type(tl.uint8))
    sm_p = sm_addr.to(tl.pointer_type(tl.float32))
    ssv_p = ssv_addr.to(tl.pointer_type(tl.float32))

    total_qblocks = (numel + 127) // 128
    qb_rows = (local_block * 8 + tl.arange(0, 8)).to(tl.int64)
    qb_mask = qb_rows < total_qblocks

    offs_2d = qb_rows[:, None] * 128 + tl.arange(0, 128)[None, :].to(tl.int64)
    elem_mask = offs_2d < numel

    total_norm = tl.sqrt(tl.load(sq_sum_ptr))
    clip_c = tl.minimum(1.0, max_norm / (total_norm + 1e-6))

    w = tl.load(w_p + offs_2d, mask=elem_mask, other=0.0)
    g = tl.load(g_p + offs_2d, mask=elem_mask, other=0.0).to(tl.float32) * clip_c

    old_qm = tl.load(qm_p + offs_2d, mask=elem_mask, other=0).to(tl.float32)
    old_qsv = tl.load(qsv_p + offs_2d, mask=elem_mask, other=0).to(tl.float32)
    old_sm = tl.load(sm_p + qb_rows, mask=qb_mask, other=0.0)
    old_ssv = tl.load(ssv_p + qb_rows, mask=qb_mask, other=0.0)

    m_old = old_qm * (old_sm[:, None] * (1.0 / 127.0))
    sv_old = old_qsv * (old_ssv[:, None] * (1.0 / 255.0))
    v_old = sv_old * sv_old

    m_new = beta1 * m_old + (1.0 - beta1) * g
    v_new = beta2 * v_old + (1.0 - beta2) * (g * g)
    sv_new = tl.sqrt(v_new)

    m_hat = m_new * bc1_inv
    sqrt_v_hat = sv_new * sqrt_bc2_inv
    w_new = w * (1.0 - lr * weight_decay) - lr * (m_hat / (sqrt_v_hat + eps))

    # Zero masked lanes before per-128-row absmax reduction
    m_masked = tl.where(elem_mask, m_new, 0.0)
    sv_masked = tl.where(elem_mask, sv_new, 0.0)
    new_sm = tl.maximum(tl.max(tl.abs(m_masked), axis=1), 1e-30)
    new_ssv = tl.maximum(tl.max(sv_masked, axis=1), 1e-30)

    qm_f = tl.extra.cuda.libdevice.rint(m_masked * (127.0 / new_sm[:, None]))
    qm_i8 = tl.minimum(127.0, tl.maximum(-127.0, qm_f)).to(tl.int8)

    qsv_f = tl.extra.cuda.libdevice.rint(sv_masked * (255.0 / new_ssv[:, None]))
    qsv_u8 = tl.minimum(255.0, tl.maximum(0.0, qsv_f)).to(tl.uint8)

    tl.store(w_p + offs_2d, w_new, mask=elem_mask)
    tl.store(qm_p + offs_2d, qm_i8, mask=elem_mask)
    tl.store(qsv_p + offs_2d, qsv_u8, mask=elem_mask)
    tl.store(sm_p + qb_rows, new_sm, mask=qb_mask)
    tl.store(ssv_p + qb_rows, new_ssv, mask=qb_mask)


class TritonMultiTensorAdamW:
    """Production-style multi-tensor Triton AdamW optimizer supporting:
      - `fp32w_bf16g_fp32mv`: FP32 master + BF16 grad (`grad_dtype=bf16`) + FP32 moments
      - `fp32w_bf16g_bf16mv`: FP32 master + BF16 grad + BF16 moments
      - `fp32w_int8mv`:       FP32 master + BF16/FP32 grad + 128-block INT8/UINT8 moments
      - `bf16_rtn`:           BF16 master (round-to-nearest negative control) + BF16 moments
      - `sr_bf16`:            Unbiased Philox Stochastic-Rounding BF16 master + BF16 moments
    Fuses global L2 gradient clipping directly into the optimizer load pass."""

    BLOCK_SIZE = 1024

    def __init__(
        self,
        params: list[torch.Tensor],
        *,
        variant: str,
        lr: float = 1.5e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        max_norm: float = 1.0,
        distributed_norm: bool = False,
        seed: int = 20261115,
    ) -> None:
        self.raw_params = list(params)
        self.local_params = [
            p._local_tensor if isinstance(p, DTensor) else p for p in self.raw_params
        ]
        self.variant = variant
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_norm = max_norm
        self.distributed_norm = distributed_norm
        self.seed = seed
        self.step_num = 0
        self.device = self.local_params[0].device

        self.w_bf16 = variant in ("bf16_rtn", "sr_bf16")
        self.use_sr = variant == "sr_bf16"
        self.is_int8 = variant == "fp32w_int8mv"
        self.mv_bf16 = variant in ("fp32w_bf16g_bf16mv", "bf16_rtn", "sr_bf16")

        self.sq_sum_buf = torch.zeros((1,), device=self.device, dtype=torch.float32)

        # Allocate persistent optimizer states once
        self.m_list: list[torch.Tensor] = []
        self.v_list: list[torch.Tensor] = []
        self.sm_list: list[torch.Tensor] = []
        self.ssv_list: list[torch.Tensor] = []

        cum_b: list[int] = []
        numels: list[int] = []
        tot_b = 0
        for lp in self.local_params:
            n = lp.numel()
            numels.append(n)
            tot_b += triton.cdiv(n, self.BLOCK_SIZE)
            cum_b.append(tot_b)
            if self.is_int8:
                qblocks = triton.cdiv(n, 128)
                self.m_list.append(torch.zeros(n, device=self.device, dtype=torch.int8))
                self.v_list.append(torch.zeros(n, device=self.device, dtype=torch.uint8))
                self.sm_list.append(
                    torch.zeros(qblocks, device=self.device, dtype=torch.float32)
                )
                self.ssv_list.append(
                    torch.zeros(qblocks, device=self.device, dtype=torch.float32)
                )
            else:
                mv_dtype = torch.bfloat16 if self.mv_bf16 else torch.float32
                self.m_list.append(torch.zeros_like(lp, dtype=mv_dtype))
                self.v_list.append(torch.zeros_like(lp, dtype=mv_dtype))

        self.total_blocks = tot_b
        self.n_tensors = len(self.local_params)
        self.numels_t = torch.tensor(numels, device=self.device, dtype=torch.int64)
        self.cum_blocks_t = torch.tensor(cum_b, device=self.device, dtype=torch.int32)
        self._cached_grad_ptrs_key: tuple[int, ...] | None = None
        self._g_ptrs_t: torch.Tensor | None = None
        self._meta_t: torch.Tensor | None = None
        self._g_bf16: bool = True

    def state_bytes_local(self) -> int:
        total = 0
        for t in self.m_list + self.v_list + self.sm_list + self.ssv_list:
            total += t.numel() * t.element_size()
        return total

    def _refresh_pointer_tables(self) -> None:
        local_grads = []
        for rp in self.raw_params:
            g = rp.grad
            assert g is not None, "Expected non-None grad on all parameters"
            lg = g._local_tensor if isinstance(g, DTensor) else g
            local_grads.append(lg)

        key = tuple(lg.data_ptr() for lg in local_grads) + tuple(
            lp.data_ptr() for lp in self.local_params
        )
        if key == self._cached_grad_ptrs_key:
            return

        self._g_bf16 = local_grads[0].dtype == torch.bfloat16
        g_addrs = [lg.data_ptr() for lg in local_grads]
        self._g_ptrs_t = torch.tensor(g_addrs, device=self.device, dtype=torch.int64)

        rows = []
        for idx, lp in enumerate(self.local_params):
            lg = local_grads[idx]
            if self.is_int8:
                rows.append(
                    [
                        lp.data_ptr(),
                        lg.data_ptr(),
                        self.m_list[idx].data_ptr(),
                        self.v_list[idx].data_ptr(),
                        self.sm_list[idx].data_ptr(),
                        self.ssv_list[idx].data_ptr(),
                        lp.numel(),
                    ]
                )
            else:
                rows.append(
                    [
                        lp.data_ptr(),
                        lg.data_ptr(),
                        self.m_list[idx].data_ptr(),
                        self.v_list[idx].data_ptr(),
                        lp.numel(),
                    ]
                )
        self._meta_t = torch.tensor(rows, device=self.device, dtype=torch.int64)
        self._cached_grad_ptrs_key = key

    def zero_grad(self, set_to_none: bool = True) -> None:
        for rp in self.raw_params:
            if set_to_none:
                rp.grad = None
            elif rp.grad is not None:
                rp.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self._refresh_pointer_tables()
        self.step_num += 1
        bc1_inv = 1.0 / (1.0 - (self.beta1**self.step_num))
        bc2_inv = 1.0 / (1.0 - (self.beta2**self.step_num))

        self.sq_sum_buf.zero_()
        _mt_grad_norm_sq_kernel[(self.total_blocks,)](
            self._g_ptrs_t,
            self.numels_t,
            self.cum_blocks_t,
            self.sq_sum_buf,
            self.n_tensors,
            G_BF16=self._g_bf16,
            BLOCK_SIZE=self.BLOCK_SIZE,
            num_warps=4,
        )
        if self.distributed_norm:
            dist.all_reduce(self.sq_sum_buf, op=dist.ReduceOp.SUM)

        if self.is_int8:
            _mt_adamw_int8mv_inclip_kernel[(self.total_blocks,)](
                self._meta_t,
                self.cum_blocks_t,
                self.sq_sum_buf,
                self.max_norm,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                bc1_inv,
                math.sqrt(bc2_inv),
                self.n_tensors,
                G_BF16=self._g_bf16,
                num_warps=8,
            )
        else:
            _mt_adamw_dense_inclip_kernel[(self.total_blocks,)](
                self._meta_t,
                self.cum_blocks_t,
                self.sq_sum_buf,
                self.max_norm,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                bc1_inv,
                bc2_inv,
                self.seed + self.step_num * 7919,
                self.n_tensors,
                W_BF16=self.w_bf16,
                G_BF16=self._g_bf16,
                MV_BF16=self.mv_bf16,
                USE_SR=self.use_sr,
                BLOCK_SIZE=self.BLOCK_SIZE,
                num_warps=4,
            )


# ============================================================================
# 2. Fused Triton + Batched Tensor-Core Muon (1-GPU & 8-GPU Owner-Sharded FSDP2)
# ============================================================================


@triton.jit
def _muon_nesterov_pack_bf16_kernel(
    g_ptrs_ptr,
    m_ptrs_ptr,
    out_buf_ptr,
    slice_numel: tl.int64,
    momentum,
    G_BF16: tl.constexpr,
    M_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Updates Nesterov momentum across `num_mats` equal-sized matrices/shards
    in a single kernel launch and writes contiguous `BF16` `send_buf[mat, idx]`."""
    mat_id = tl.program_id(1)
    offs = (tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offs < slice_numel

    g_addr = tl.load(g_ptrs_ptr + mat_id)
    m_addr = tl.load(m_ptrs_ptr + mat_id)

    g_p = (
        g_addr.to(tl.pointer_type(tl.bfloat16))
        if G_BF16
        else g_addr.to(tl.pointer_type(tl.float32))
    )
    m_p = (
        m_addr.to(tl.pointer_type(tl.bfloat16))
        if M_BF16
        else m_addr.to(tl.pointer_type(tl.float32))
    )

    g = tl.load(g_p + offs, mask=mask, other=0.0).to(tl.float32)
    m_old = tl.load(m_p + offs, mask=mask, other=0.0).to(tl.float32)
    # Match torch.optim.Muon lerp_: m_new = m_old + (1 - mom) * (g - m_old), u = g + mom * (m_new - g)
    m_new = m_old + (1.0 - momentum) * (g - m_old)
    u = g + momentum * (m_new - g)

    if M_BF16:
        tl.store(m_p + offs, m_new.to(tl.bfloat16), mask=mask)
    else:
        tl.store(m_p + offs, m_new, mask=mask)

    out_base = out_buf_ptr + mat_id.to(tl.int64) * slice_numel
    tl.store(out_base + offs, u.to(tl.bfloat16), mask=mask)


@triton.jit
def _muon_wd_update_kernel(
    w_ptrs_ptr,
    upd_buf_ptr,
    slice_numel: tl.int64,
    wd_keep,
    neg_adj_lr,
    W_BF16_SR: tl.constexpr,
    seed: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Applies decoupled weight decay and `-adj_lr * ortho_update` across all
    matrices in a shape bucket in one kernel launch."""
    mat_id = tl.program_id(1)
    offs = (tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offs < slice_numel

    w_addr = tl.load(w_ptrs_ptr + mat_id)
    upd_base = upd_buf_ptr + mat_id.to(tl.int64) * slice_numel
    u = tl.load(upd_base + offs, mask=mask, other=0.0).to(tl.float32)

    if W_BF16_SR:
        w_p = w_addr.to(tl.pointer_type(tl.bfloat16))
        w = tl.load(w_p + offs, mask=mask, other=0.0).to(tl.float32)
        w_new = w * wd_keep + neg_adj_lr * u
        rng_idx = (mat_id * 1048573 + tl.program_id(0) * BLOCK_SIZE) + tl.arange(
            0, BLOCK_SIZE
        )
        r_u32 = tl.randint(seed, rng_idx).to(tl.uint32)
        w_bits = w_new.to(tl.uint32, bitcast=True)
        w_sr = ((w_bits + (r_u32 & 0xFFFF)) & 0xFFFF0000).to(tl.float32, bitcast=True)
        tl.store(w_p + offs, w_sr.to(tl.bfloat16), mask=mask)
    else:
        w_p = w_addr.to(tl.pointer_type(tl.float32))
        w = tl.load(w_p + offs, mask=mask, other=0.0)
        w_new = w * wd_keep + neg_adj_lr * u
        tl.store(w_p + offs, w_new, mask=mask)


def batched_ns5_bf16(
    u_batch_bf16: torch.Tensor, ns_steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Batched 5-step quintic Newton-Schulz on `[B, R, C]` BF16 Tensor Cores."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = u_batch_bf16
    transposed = x.size(1) > x.size(2)
    if transposed:
        x = x.transpose(1, 2).contiguous()
    norms = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True).clamp(min=eps)
    x = x / norms
    for _ in range(ns_steps):
        gram = torch.bmm(x, x.transpose(1, 2))
        gram_upd = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, gram_upd, x, beta=a, alpha=1.0)
    if transposed:
        x = x.transpose(1, 2).contiguous()
    return x


class BatchedAndOwnerShardedMuon:
    """Supports:
      - `single_gpu`: Shape-bucketed batched `bmm` + fused Triton pre/post Muon
      - `fsdp2_owner_seq`: 8-GPU Owner-Sharded `AllToAll` + batched `bmm` Muon
      - `fsdp2_owner_pipelined`: 2-stream overlapped `AllToAll` + batched `bmm` Muon
      - `fsdp2_shard_local_wrong`: 0-comm shard-local NS5 negative control
    """

    BLOCK_SIZE = 1024

    def __init__(
        self,
        params_2d: list[torch.Tensor],
        *,
        mode: str,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        bf16_momentum: bool = False,
    ) -> None:
        self.raw_params = list(params_2d)
        self.mode = mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.bf16_momentum = bf16_momentum
        self.step_num = 0
        self.world_size = dist.get_world_size() if mode != "single_gpu" else 1

        first_loc = (
            self.raw_params[0]._local_tensor
            if isinstance(self.raw_params[0], DTensor)
            else self.raw_params[0]
        )
        self.device = first_loc.device
        self.w_bf16_sr = first_loc.dtype == torch.bfloat16

        # Group parameters by global `(R, C)` shape
        grouped: dict[tuple[int, int], list[torch.Tensor]] = {}
        for p in self.raw_params:
            grouped.setdefault((int(p.shape[0]), int(p.shape[1])), []).append(p)

        self.buckets: list[dict[str, Any]] = []
        for (rows_g, cols_g), plist in grouped.items():
            locs = [p._local_tensor if isinstance(p, DTensor) else p for p in plist]
            m_dtype = torch.bfloat16 if bf16_momentum else locs[0].dtype
            moms = [torch.zeros_like(lp, dtype=m_dtype) for lp in locs]
            w_ptrs = torch.tensor(
                [lp.data_ptr() for lp in locs], device=self.device, dtype=torch.int64
            )
            m_ptrs = torch.tensor(
                [mb.data_ptr() for mb in moms], device=self.device, dtype=torch.int64
            )
            slice_numel = locs[0].numel()
            r_loc = locs[0].shape[0]
            num_mats = len(plist)
            pad_mats = (
                triton.cdiv(num_mats, self.world_size) * self.world_size
                if self.mode != "single_gpu"
                else num_mats
            )
            adj_lr = lr * 0.2 * math.sqrt(max(rows_g, cols_g))

            send_buf = torch.zeros(
                (pad_mats, r_loc, cols_g), device=self.device, dtype=torch.bfloat16
            )
            recv_buf = torch.zeros_like(send_buf)
            back_send = torch.zeros_like(send_buf)
            back_recv = torch.zeros_like(send_buf)

            self.buckets.append(
                {
                    "global_shape": (rows_g, cols_g),
                    "raw_params": plist,
                    "locs": locs,
                    "moms": moms,
                    "w_ptrs": w_ptrs,
                    "m_ptrs": m_ptrs,
                    "slice_numel": slice_numel,
                    "r_loc": r_loc,
                    "cols_g": cols_g,
                    "num_mats": num_mats,
                    "pad_mats": pad_mats,
                    "adj_lr": adj_lr,
                    "send_buf": send_buf,
                    "recv_buf": recv_buf,
                    "back_send": back_send,
                    "back_recv": back_recv,
                    "cached_g_key": None,
                    "g_ptrs": None,
                    "g_bf16": False,
                }
            )

    def state_bytes_local(self) -> int:
        total = 0
        for b in self.buckets:
            for mb in b["moms"]:
                total += mb.numel() * mb.element_size()
        return total

    def _ensure_grad_ptrs(self, b: dict[str, Any]) -> None:
        lgs = [
            p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
            for p in b["raw_params"]
        ]
        key = tuple(lg.data_ptr() for lg in lgs)
        if key != b["cached_g_key"]:
            b["g_ptrs"] = torch.tensor(
                [lg.data_ptr() for lg in lgs], device=self.device, dtype=torch.int64
            )
            b["g_bf16"] = lgs[0].dtype == torch.bfloat16
            b["cached_g_key"] = key

    def _launch_nesterov_pack(self, b: dict[str, Any]) -> None:
        self._ensure_grad_ptrs(b)
        grid = (triton.cdiv(b["slice_numel"], self.BLOCK_SIZE), b["num_mats"])
        _muon_nesterov_pack_bf16_kernel[grid](
            b["g_ptrs"],
            b["m_ptrs"],
            b["send_buf"],
            b["slice_numel"],
            self.momentum,
            G_BF16=b["g_bf16"],
            M_BF16=(b["moms"][0].dtype == torch.bfloat16),
            BLOCK_SIZE=self.BLOCK_SIZE,
            num_warps=4,
        )

    def _launch_wd_update(self, b: dict[str, Any], upd_buf: torch.Tensor) -> None:
        grid = (triton.cdiv(b["slice_numel"], self.BLOCK_SIZE), b["num_mats"])
        _muon_wd_update_kernel[grid](
            b["w_ptrs"],
            upd_buf,
            b["slice_numel"],
            1.0 - self.lr * self.weight_decay,
            -b["adj_lr"],
            W_BF16_SR=self.w_bf16_sr,
            seed=70001 + self.step_num * 313,
            BLOCK_SIZE=self.BLOCK_SIZE,
            num_warps=4,
        )

    @torch.no_grad()
    def step(self) -> None:
        self.step_num += 1
        w_sz = self.world_size

        if self.mode == "single_gpu":
            for b in self.buckets:
                self._launch_nesterov_pack(b)
                ortho = batched_ns5_bf16(b["send_buf"])
                self._launch_wd_update(b, ortho)
            return

        if self.mode == "fsdp2_shard_local_wrong":
            # Negative control: orthogonalizes each rank's [R/8, C] shard independently (0 collectives!)
            for b in self.buckets:
                self._launch_nesterov_pack(b)
                ortho_wrong = batched_ns5_bf16(b["send_buf"][: b["num_mats"]])
                self._launch_wd_update(b, ortho_wrong)
            return

        if self.mode == "fsdp2_owner_seq":
            for b in self.buckets:
                pad_mats = b["pad_mats"]
                per_rank = pad_mats // w_sz
                rows_g, cols_g = b["global_shape"]
                r_loc = b["r_loc"]

                self._launch_nesterov_pack(b)
                dist.all_to_all_single(b["recv_buf"], b["send_buf"])
                full_owned = (
                    b["recv_buf"]
                    .view(w_sz, per_rank, r_loc, cols_g)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                    .view(per_rank, rows_g, cols_g)
                )
                ortho_owned = batched_ns5_bf16(full_owned)
                b["back_send"].view(w_sz, per_rank, r_loc, cols_g).copy_(
                    ortho_owned.view(per_rank, w_sz, r_loc, cols_g).permute(1, 0, 2, 3)
                )
                dist.all_to_all_single(b["back_recv"], b["back_send"])
                self._launch_wd_update(b, b["back_recv"])
            return

        if self.mode == "fsdp2_owner_pipelined":
            # 2-stream overlapped schedule: overlap ncclStream AllToAll(b+1) & BackAllToAll(b-1)
            # behind default_stream Tensor-Core batched_ns5_bf16(b)!
            nb = len(self.buckets)
            for b in self.buckets:
                self._launch_nesterov_pack(b)

            wf_handles: list[Any] = [None] * nb
            wb_handles: list[Any] = [None] * nb
            wf_handles[0] = dist.all_to_all_single(
                self.buckets[0]["recv_buf"],
                self.buckets[0]["send_buf"],
                async_op=True,
            )

            for idx, b in enumerate(self.buckets):
                wf_handles[idx].wait()
                if idx + 1 < nb:
                    nxt = self.buckets[idx + 1]
                    wf_handles[idx + 1] = dist.all_to_all_single(
                        nxt["recv_buf"], nxt["send_buf"], async_op=True
                    )
                per_rank = b["pad_mats"] // w_sz
                rows_g, cols_g = b["global_shape"]
                r_loc = b["r_loc"]
                full_owned = (
                    b["recv_buf"]
                    .view(w_sz, per_rank, r_loc, cols_g)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                    .view(per_rank, rows_g, cols_g)
                )
                ortho_owned = batched_ns5_bf16(full_owned)
                b["back_send"].view(w_sz, per_rank, r_loc, cols_g).copy_(
                    ortho_owned.view(per_rank, w_sz, r_loc, cols_g).permute(1, 0, 2, 3)
                )
                wb_handles[idx] = dist.all_to_all_single(
                    b["back_recv"], b["back_send"], async_op=True
                )

            for idx, b in enumerate(self.buckets):
                wb_handles[idx].wait()
                self._launch_wd_update(b, b["back_recv"])
            return

        raise ValueError(f"Unknown Muon mode: {self.mode}")


# ============================================================================
# 3. Transformer Model & Timing / Telemetry Helpers
# ============================================================================


class GQASwiGLUBlock(nn.Module):
    def __init__(
        self, hidden: int, inter_dim: int, num_q_heads: int, num_kv_heads: int
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_q = num_q_heads
        self.num_kv = num_kv_heads
        self.head_dim = hidden // num_q_heads
        kv_dim = num_kv_heads * self.head_dim

        self.attn_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.wq = nn.Linear(hidden, hidden, bias=False)
        self.wk = nn.Linear(hidden, kv_dim, bias=False)
        self.wv = nn.Linear(hidden, kv_dim, bias=False)
        self.wo = nn.Linear(hidden, hidden, bias=False)

        self.ffn_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.w_gate = nn.Linear(hidden, inter_dim, bias=False)
        self.w_up = nn.Linear(hidden, inter_dim, bias=False)
        self.w_down = nn.Linear(inter_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).view(bsz, seqlen, self.num_q, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(bsz, seqlen, self.num_kv, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(bsz, seqlen, self.num_kv, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=(self.num_q != self.num_kv)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden)
        x = x + self.wo(attn_out)

        h2 = self.ffn_norm(x)
        x = x + self.w_down(F.silu(self.w_gate(h2)) * self.w_up(h2))
        return x


class GQATransformerModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden: int,
        inter_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GQASwiGLUBlock(hidden, inter_dim, num_q_heads, num_kv_heads)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


def sample_gpu_clocks(device_idx: int) -> dict[str, float]:
    if not HAS_PYNVML:
        return {"sm_clock_mhz": 0.0, "power_w": 0.0}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        return {
            "sm_clock_mhz": float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
            "power_w": float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0,
        }
    except Exception:
        return {"sm_clock_mhz": 0.0, "power_w": 0.0}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-8)).item()
    )


def time_cuda_fn(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    device_idx: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device_idx)
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        s_ev.record()
        fn()
        e_ev.record()
        torch.cuda.synchronize(device_idx)
        samples.append(float(s_ev.elapsed_time(e_ev)))
    nv = sample_gpu_clocks(device_idx)
    ss = sorted(samples)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    return {
        "median_ms": round(statistics.median(ss), 4),
        "iqr_ms": round(q75 - q25, 4),
        "min_ms": round(ss[0], 4),
        "max_ms": round(ss[-1], 4),
        "sm_clock_after_mhz": nv["sm_clock_mhz"],
        "power_after_w": round(nv["power_w"], 2),
    }


# ============================================================================
# 4. Suite 0: PyTorch 2.13 grad_dtype, Sub-ULP SR, and 1-GPU Muon Launch Audit
# ============================================================================


def run_suite_0_audit(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    # 1. Audit PyTorch 2.13 `p.grad_dtype = torch.bfloat16` across all 3 AdamW backends
    grad_dtype_audit: dict[str, Any] = {}
    gen = torch.Generator(device=device)
    gen.manual_seed(11800)
    w0 = torch.randn((1024, 1024), generator=gen, device=device, dtype=torch.float32)
    g0_bf16 = torch.randn((1024, 1024), generator=gen, device=device, dtype=torch.bfloat16)

    for bk_name, kwargs in (
        ("single_tensor", {"foreach": False, "fused": False}),
        ("foreach", {"foreach": True, "fused": False}),
        ("fused", {"foreach": False, "fused": True}),
    ):
        p = w0.clone().requires_grad_(True)
        p.grad_dtype = torch.bfloat16
        p.grad = g0_bf16.clone()
        rejected = False
        err_msg = ""
        try:
            opt = torch.optim.AdamW([p], lr=1e-3, **kwargs)
            opt.step()
        except RuntimeError as exc:
            rejected = True
            err_msg = str(exc)[:180]
        grad_dtype_audit[bk_name] = {
            "rejected_fp32w_bf16_grad_dtype": rejected,
            "error_excerpt": err_msg,
        }

    # Verify our Triton `fp32w_bf16g_fp32mv` handles `p.grad_dtype = torch.bfloat16` cleanly & matches FP32 ref!
    p_ref = w0.clone().requires_grad_(True)
    p_ref.grad = g0_bf16.float()
    torch.nn.utils.clip_grad_norm_([p_ref], max_norm=0.5, foreach=True)
    opt_ref = torch.optim.AdamW(
        [p_ref], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True
    )
    opt_ref.step()

    p_tri = w0.clone().requires_grad_(True)
    p_tri.grad_dtype = torch.bfloat16
    p_tri.grad = g0_bf16.clone()
    opt_tri = TritonMultiTensorAdamW(
        [p_tri],
        variant="fp32w_bf16g_fp32mv",
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        max_norm=0.5,
    )
    opt_tri.step()
    triton_grad_dtype_rel_l2 = rel_l2(p_tri, p_ref)

    # Also verify 8-bit moment AdamW single-step parity vs FP32 ref
    p_i8 = w0.clone().requires_grad_(True)
    p_i8.grad_dtype = torch.bfloat16
    p_i8.grad = g0_bf16.clone()
    opt_i8 = TritonMultiTensorAdamW(
        [p_i8],
        variant="fp32w_int8mv",
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        max_norm=0.5,
    )
    opt_i8.step()
    int8_delta_cos = float(
        F.cosine_similarity(
            (p_i8 - w0).flatten(), (p_ref - w0).flatten(), dim=0
        ).item()
    )

    # 2. Sub-ULP BF16 Stagnation vs Philox Stochastic Rounding Unbiasedness Proof
    # Take 1,048,576 BF16 parameters initialized at +1.0 and -1.0 (half positive, half negative!),
    # apply 1,000 steps of lr = 1e-4, wd = 0.0, constant normalized step (+1e-4 on negative half, -1e-4 on positive half).
    # Exact target magnitude after N steps: 1.0 - N * 1e-4!
    n_sub = 262144 if is_smoke else 1048576
    steps_sub = 200 if is_smoke else 1000
    lr_sub = 1e-4  # 0.0128 * ulp_bf16(1.0) -> 78x smaller than BF16 ULP!
    target_mag = 1.0 - steps_sub * lr_sub

    signs = torch.where(
        torch.arange(n_sub, device=device) % 2 == 0,
        torch.ones(1, device=device, dtype=torch.bfloat16),
        -torch.ones(1, device=device, dtype=torch.bfloat16),
    )
    w_rtn = signs.clone().requires_grad_(True)
    w_sr = signs.clone().requires_grad_(True)
    g_const = (signs * 0.01).contiguous()  # Constant sign gradient -> m_hat / sqrt(v_hat) == sign!

    opt_rtn = TritonMultiTensorAdamW(
        [w_rtn],
        variant="bf16_rtn",
        lr=lr_sub,
        betas=(0.9, 0.95),
        eps=1e-12,
        weight_decay=0.0,
        max_norm=1e9,
    )
    opt_sr = TritonMultiTensorAdamW(
        [w_sr],
        variant="sr_bf16",
        lr=lr_sub,
        betas=(0.9, 0.95),
        eps=1e-12,
        weight_decay=0.0,
        max_norm=1e9,
    )
    for _ in range(steps_sub):
        w_rtn.grad = g_const
        w_sr.grad = g_const
        opt_rtn.step()
        opt_sr.step()

    rtn_mag_mean = float(w_rtn.float().abs().mean().item())
    sr_pos_mean = float(w_sr[0::2].float().mean().item())
    sr_neg_mean = float(w_sr[1::2].float().mean().item())
    sr_mag_mean = float(w_sr.float().abs().mean().item())
    rtn_stagnant_frac = float((w_rtn == signs).float().mean().item())
    sr_stagnant_frac = float((w_sr == signs).float().mean().item())
    expected_delta = steps_sub * lr_sub
    sr_delta_rel_err = abs((1.0 - sr_mag_mean) - expected_delta) / expected_delta

    # 3. Single-GPU PyTorch 2.13 `torch.optim.Muon` vs Shape-Bucketed Batched Triton+bmm Muon Audit
    num_layers_m = 4 if is_smoke else 16
    h_m = 512 if is_smoke else 2048
    inter_m = 1536 if is_smoke else 8192
    kv_m = h_m // 4
    warmup = 2 if is_smoke else 5
    iters = 5 if is_smoke else 15

    muon_foreach_rejected = False
    try:
        dummy_p = torch.randn((64, 64), device=device, requires_grad=True)
        dummy_p.grad = torch.randn_like(dummy_p)
        from torch.optim._muon import muon as functional_muon

        functional_muon(
            [dummy_p],
            [dummy_p.grad],
            [torch.zeros_like(dummy_p)],
            foreach=True,
            lr=1e-3,
            weight_decay=0.1,
            momentum=0.95,
            nesterov=True,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            ns_steps=5,
            eps=1e-7,
            adjust_lr_fn="match_rms_adamw",
            has_complex=False,
        )
    except RuntimeError:
        muon_foreach_rejected = True

    shapes_per_layer = [
        (h_m, h_m),
        (kv_m, h_m),
        (kv_m, h_m),
        (h_m, h_m),
        (inter_m, h_m),
        (inter_m, h_m),
        (h_m, inter_m),
    ]
    params_pt_m: list[torch.Tensor] = []
    params_bt_m: list[torch.Tensor] = []
    for _ in range(num_layers_m):
        for r_s, c_s in shapes_per_layer:
            w_init = torch.randn((r_s, c_s), generator=gen, device=device, dtype=torch.float32) * 0.02
            g_init = torch.randn((r_s, c_s), generator=gen, device=device, dtype=torch.float32) * 0.01
            p1 = w_init.clone().requires_grad_(True)
            p1.grad = g_init.clone()
            p2 = w_init.clone().requires_grad_(True)
            p2.grad = g_init.clone()
            params_pt_m.append(p1)
            params_bt_m.append(p2)

    opt_pt_muon = torch.optim.Muon(
        params_pt_m,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )
    opt_bt_muon = BatchedAndOwnerShardedMuon(
        params_bt_m,
        mode="single_gpu",
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
    )
    opt_pt_muon.step()
    opt_bt_muon.step()
    muon_1gpu_max_rel_l2 = max(
        rel_l2(pb, pp) for pb, pp in zip(params_bt_m, params_pt_m)
    )

    with profile(activities=[ProfilerActivity.CUDA]) as prof_pt_m:
        opt_pt_muon.step()
        torch.cuda.synchronize(device)
    pt_muon_launches = sum(
        int(e.count) for e in prof_pt_m.key_averages() if e.self_device_time_total > 0
    )

    with profile(activities=[ProfilerActivity.CUDA]) as prof_bt_m:
        opt_bt_muon.step()
        torch.cuda.synchronize(device)
    bt_muon_launches = sum(
        int(e.count) for e in prof_bt_m.key_averages() if e.self_device_time_total > 0
    )

    t_pt_m = time_cuda_fn(
        opt_pt_muon.step, warmup=warmup, iters=iters, device_idx=device.index
    )
    t_bt_m = time_cuda_fn(
        opt_bt_muon.step, warmup=warmup, iters=iters, device_idx=device.index
    )

    return {
        "grad_dtype_audit": {
            "backends": grad_dtype_audit,
            "triton_fp32w_bf16g_vs_fp32_ref_rel_l2": round(
                triton_grad_dtype_rel_l2, 7
            ),
            "triton_int8mv_step1_delta_cosine": round(int8_delta_cos, 6),
        },
        "sub_ulp_stochastic_rounding_proof": {
            "num_elements": n_sub,
            "steps": steps_sub,
            "lr_sub_ulp": lr_sub,
            "bf16_ulp_at_1p0": 0.0078125,
            "target_magnitude": round(target_mag, 6),
            "rtn_final_magnitude_mean": round(rtn_mag_mean, 6),
            "rtn_stagnant_fraction": round(rtn_stagnant_frac, 6),
            "sr_final_pos_mean": round(sr_pos_mean, 6),
            "sr_final_neg_mean": round(sr_neg_mean, 6),
            "sr_final_magnitude_mean": round(sr_mag_mean, 6),
            "sr_stagnant_fraction": round(sr_stagnant_frac, 6),
            "sr_accumulated_delta_rel_err": round(sr_delta_rel_err, 5),
        },
        "single_gpu_muon_launch_audit": {
            "num_2d_matrices": len(params_pt_m),
            "foreach_rejected": muon_foreach_rejected,
            "pytorch_muon_cuda_launches": pt_muon_launches,
            "batched_triton_muon_cuda_launches": bt_muon_launches,
            "launch_reduction_factor": round(
                float(pt_muon_launches) / max(float(bt_muon_launches), 1.0), 2
            ),
            "max_param_rel_l2_vs_pytorch_muon": round(muon_1gpu_max_rel_l2, 7),
            "pytorch_muon_median_ms": t_pt_m["median_ms"],
            "batched_triton_muon_median_ms": t_bt_m["median_ms"],
            "speedup_vs_pytorch_muon": round(
                t_pt_m["median_ms"] / max(t_bt_m["median_ms"], 1e-6), 4
            ),
        },
    }


# ============================================================================
# 5. Suite A: Single-GPU Multi-Tensor AdamW + Clip Bandwidth & State Roofline
# ============================================================================


def run_suite_a_adamw_roofline(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    # 64 Transformer-shaped tensors (16 layers x 4 projections) across 3 model scales
    if is_smoke:
        scales = [("smoke_16m", 1024, 1024)]
        warmup, iters = 2, 5
    else:
        scales = [
            ("slm_67m", 1024, 1024),       # 64 x 1,048,576 = 67,108,864 params
            ("mid_268m", 2048, 2048),      # 64 x 4,194,304 = 268,435,456 params
            ("llm_1p07b", 4096, 4096),     # 64 x 16,777,216 = 1,073,741,824 params
        ]
        warmup, iters = 6, 20

    num_tensors = 64
    results: dict[str, Any] = {}

    for scale_label, r_dim, c_dim in scales:
        total_params = num_tensors * r_dim * c_dim
        gen = torch.Generator(device=device)
        gen.manual_seed(11820 + r_dim)

        # Base FP32 and BF16 gradients (fixed norm > max_norm=1.0 so clipping is active!)
        w_base_f32 = [
            torch.randn((r_dim, c_dim), generator=gen, device=device, dtype=torch.float32) * 0.02
            for _ in range(num_tensors)
        ]
        g_base_bf16 = [
            torch.randn((r_dim, c_dim), generator=gen, device=device, dtype=torch.bfloat16) * 0.05
            for _ in range(num_tensors)
        ]

        modes_res: dict[str, Any] = {}

        # Mode 1: PyTorch single-tensor clip + single-tensor AdamW (only up to mid_268m or all 3; 64*7=448 launches is fast enough!)
        for m_name, clip_foreach, opt_foreach, fused_flag, phys_bytes_per_p in (
            ("pytorch_single_tensor_fp32_clip", False, False, False, 60.0),
            ("pytorch_foreach_fp32_clip", True, True, False, 56.0),
            ("pytorch_fused_fp32_clip", True, False, True, 40.0),
        ):
            ps = [w.clone().requires_grad_(True) for w in w_base_f32]
            gs_f32 = [g.float() for g in g_base_bf16]
            for p, g in zip(ps, gs_f32):
                p.grad = g.clone()
            opt = torch.optim.AdamW(
                ps,
                lr=1.5e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                foreach=opt_foreach,
                fused=fused_flag,
            )

            def _step_pt() -> None:
                # Copy fresh unclipped gradient (or scale in place as real backward produces fresh grads)
                torch.nn.utils.clip_grad_norm_(
                    ps, max_norm=1.0, foreach=clip_foreach
                )
                opt.step()

            tm = time_cuda_fn(_step_pt, warmup=warmup, iters=iters, device_idx=device.index)
            gparams_s = (float(total_params) / (tm["median_ms"] * 1e-3)) / 1e9
            hbm_gbps = (
                float(total_params) * phys_bytes_per_p / (tm["median_ms"] * 1e-3)
            ) / 1e9
            modes_res[m_name] = {
                **tm,
                "gparams_per_s": round(gparams_s, 3),
                "physical_hbm_bytes_per_param": phys_bytes_per_p,
                "achieved_hbm_gbps": round(hbm_gbps, 2),
                "resident_bytes_per_param": 16.0,
                "optimizer_state_bytes_per_param": 8.0,
            }
            del ps, gs_f32, opt
            torch.cuda.empty_cache()

        # Modes 4..7: Custom Multi-Tensor Triton In-Register-Clip AdamW Variants
        for m_name, var_key, w_is_bf16, phys_bytes_per_p, res_bytes, state_bytes in (
            ("triton_fp32w_bf16g_fp32mv_inclip", "fp32w_bf16g_fp32mv", False, 28.0, 14.0, 8.0),
            ("triton_fp32w_bf16g_bf16mv_inclip", "fp32w_bf16g_bf16mv", False, 20.0, 10.0, 4.0),
            ("triton_fp32w_bf16g_int8mv_inclip", "fp32w_int8mv", False, 16.125, 8.0625, 2.0625),
            ("triton_sr_bf16_inclip", "sr_bf16", True, 16.0, 8.0, 4.0),
        ):
            if w_is_bf16:
                ps = [w.bfloat16().requires_grad_(True) for w in w_base_f32]
            else:
                ps = [w.clone().requires_grad_(True) for w in w_base_f32]
                for p in ps:
                    p.grad_dtype = torch.bfloat16
            for p, g in zip(ps, g_base_bf16):
                p.grad = g

            opt_t = TritonMultiTensorAdamW(
                ps,
                variant=var_key,
                lr=1.5e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                max_norm=1.0,
            )
            tm = time_cuda_fn(
                opt_t.step, warmup=warmup, iters=iters, device_idx=device.index
            )
            gparams_s = (float(total_params) / (tm["median_ms"] * 1e-3)) / 1e9
            hbm_gbps = (
                float(total_params) * phys_bytes_per_p / (tm["median_ms"] * 1e-3)
            ) / 1e9
            modes_res[m_name] = {
                **tm,
                "gparams_per_s": round(gparams_s, 3),
                "physical_hbm_bytes_per_param": phys_bytes_per_p,
                "achieved_hbm_gbps": round(hbm_gbps, 2),
                "resident_bytes_per_param": res_bytes,
                "optimizer_state_bytes_per_param": state_bytes,
            }
            del ps, opt_t
            torch.cuda.empty_cache()

        results[scale_label] = {
            "num_tensors": num_tensors,
            "tensor_shape": [r_dim, c_dim],
            "total_params": total_params,
            "modes": modes_res,
        }
    return results


# ============================================================================
# 6. Suite B: Multi-Step Transformer Loss Convergence & BF16 Stagnation Proof
# ============================================================================


def run_suite_b_convergence(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    vocab_size = 128
    num_layers = 2 if is_smoke else 4
    hidden = 256 if is_smoke else 512
    inter_dim = 768 if is_smoke else 1536
    q_heads = 4 if is_smoke else 8
    kv_heads = 2
    steps = 32 if is_smoke else 120
    bsz = 4 if is_smoke else 8
    seqlen = 64 if is_smoke else 128
    lr_peak = 1.1e-4
    lr_min = 4.0e-5

    class CausalSLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, hidden)
            self.core = GQATransformerModel(
                num_layers, hidden, inter_dim, q_heads, kv_heads
            )
            self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
            for p in self.parameters():
                if p.ndim == 2:
                    nn.init.normal_(p, mean=0.0, std=0.04)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.core(self.tok_emb(ids)))

    gen = torch.Generator(device=device)
    gen.manual_seed(11850)
    num_batches = 1
    token_batches = [
        torch.randint(
            0, vocab_size, (bsz, seqlen + 1), generator=gen, device=device
        )
        for _ in range(num_batches)
    ]

    torch.manual_seed(11855)
    init_model = CausalSLM().to(device=device, dtype=torch.float32)
    init_sd = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    del init_model

    trajectories: dict[str, Any] = {}
    for opt_label in (
        "fp32_master_adamw_ref",
        "triton_fp32w_bf16mv",
        "triton_fp32w_int8mv",
        "triton_bf16_rtn_stagnant",
        "triton_sr_bf16",
        "batched_muon_plus_adamw",
    ):
        use_bf16_weights = opt_label in ("triton_bf16_rtn_stagnant", "triton_sr_bf16")
        m = CausalSLM().to(
            device=device, dtype=torch.bfloat16 if use_bf16_weights else torch.float32
        )
        with torch.no_grad():
            for k, p in m.named_parameters():
                p.copy_(init_sd[k].bfloat16() if use_bf16_weights else init_sd[k])
                if not use_bf16_weights and opt_label in (
                    "triton_fp32w_bf16mv",
                    "triton_fp32w_int8mv",
                ):
                    p.grad_dtype = torch.bfloat16

        params_all = list(m.parameters())
        if opt_label == "fp32_master_adamw_ref":
            opt_main: Any = torch.optim.AdamW(
                params_all, lr=lr_peak, betas=(0.9, 0.95), weight_decay=0.05, fused=True
            )
            opt_aux = None
        elif opt_label == "triton_fp32w_bf16mv":
            opt_main = TritonMultiTensorAdamW(
                params_all,
                variant="fp32w_bf16g_bf16mv",
                lr=lr_peak,
                weight_decay=0.05,
                max_norm=1.0,
            )
            opt_aux = None
        elif opt_label == "triton_fp32w_int8mv":
            opt_main = TritonMultiTensorAdamW(
                params_all,
                variant="fp32w_int8mv",
                lr=lr_peak,
                weight_decay=0.05,
                max_norm=1.0,
            )
            opt_aux = None
        elif opt_label == "triton_bf16_rtn_stagnant":
            opt_main = TritonMultiTensorAdamW(
                params_all,
                variant="bf16_rtn",
                lr=lr_peak,
                weight_decay=0.05,
                max_norm=1.0,
            )
            opt_aux = None
        elif opt_label == "triton_sr_bf16":
            opt_main = TritonMultiTensorAdamW(
                params_all,
                variant="sr_bf16",
                lr=lr_peak,
                weight_decay=0.05,
                max_norm=1.0,
            )
            opt_aux = None
        elif opt_label == "batched_muon_plus_adamw":
            # Standard Muon partitioning: hidden 2D matrices -> Muon, embeddings/head/1D -> AdamW
            p2d = [
                p
                for n, p in m.named_parameters()
                if p.ndim == 2 and "tok_emb" not in n and "lm_head" not in n
            ]
            p_other = [
                p
                for n, p in m.named_parameters()
                if not (p.ndim == 2 and "tok_emb" not in n and "lm_head" not in n)
            ]
            opt_main = BatchedAndOwnerShardedMuon(
                p2d,
                mode="single_gpu",
                lr=lr_peak * 5.0,
                weight_decay=0.05,
                momentum=0.95,
            )
            opt_aux = torch.optim.AdamW(
                p_other, lr=lr_peak, betas=(0.9, 0.95), weight_decay=0.05, fused=True
            )
        else:
            raise ValueError(opt_label)

        losses: list[float] = []
        stagnant_fracs: list[float] = []
        for st in range(steps):
            # Cosine LR schedule from lr_peak -> lr_min
            cos_t = 0.5 * (1.0 + math.cos(math.pi * st / max(steps - 1, 1)))
            cur_lr = lr_min + (lr_peak - lr_min) * cos_t
            if isinstance(opt_main, torch.optim.Optimizer):
                for pg in opt_main.param_groups:
                    pg["lr"] = cur_lr
            elif isinstance(opt_main, TritonMultiTensorAdamW):
                opt_main.lr = cur_lr
            else:
                opt_main.lr = cur_lr * 5.0
                for b_info in opt_main.buckets:
                    rg, cg = b_info["global_shape"]
                    b_info["adj_lr"] = opt_main.lr * 0.2 * math.sqrt(max(rg, cg))
                if opt_aux is not None:
                    for pg in opt_aux.param_groups:
                        pg["lr"] = cur_lr

            tb = token_batches[st % num_batches]
            inp, tgt = tb[:, :-1], tb[:, 1:]
            for p in params_all:
                p.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = m(inp)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size).float(), tgt.reshape(-1)
                )
            loss.backward()
            losses.append(float(loss.item()))

            with torch.no_grad():
                snaps = [p.detach().clone() for p in params_all]
            if opt_label == "fp32_master_adamw_ref":
                torch.nn.utils.clip_grad_norm_(params_all, max_norm=1.0, foreach=True)
                opt_main.step()
            elif opt_label == "batched_muon_plus_adamw":
                torch.nn.utils.clip_grad_norm_(params_all, max_norm=1.0, foreach=True)
                opt_main.step()
                assert opt_aux is not None
                opt_aux.step()
            else:
                opt_main.step()

            with torch.no_grad():
                unchanged = 0
                tot_el = 0
                for p, sn in zip(params_all, snaps):
                    unchanged += int((p == sn).sum().item())
                    tot_el += p.numel()
                stagnant_fracs.append(float(unchanged) / max(float(tot_el), 1.0))

        trajectories[opt_label] = {
            "initial_loss": round(losses[0], 6),
            "mid_loss": round(losses[len(losses) // 2], 6),
            "final_loss": round(losses[-1], 6),
            "loss_reduction_ratio": round(losses[0] / max(losses[-1], 1e-8), 4),
            "mean_step_stagnant_param_fraction": round(
                statistics.mean(stagnant_fracs), 5
            ),
            "tail_step_stagnant_param_fraction": round(
                statistics.mean(stagnant_fracs[len(stagnant_fracs) // 2 :]), 5
            ),
        }
        del m, opt_main, opt_aux, params_all
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "steps": steps,
        "lr_peak": lr_peak,
        "lr_min": lr_min,
        "optimizers": trajectories,
    }


# ============================================================================
# 7. Suite C: 8-GPU FSDP2 Llama-3-Width Distributed Muon & SR-BF16/INT8 AdamW
# ============================================================================


def run_suite_c_fsdp2(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    mesh = init_device_mesh("cuda", (world_size,))

    # ------------------------------------------------------------------------
    # Part C.1: 16-Layer Llama-3-8B-Width (3.49B, 112 2D + 33 1D = 145 DTensors)
    #           Isolated Distributed Optimizer + Clip Step on 8xA100
    # ------------------------------------------------------------------------
    num_layers_c1 = 8 if is_smoke else 16
    h_c1 = 512 if is_smoke else 4096
    inter_c1 = 1536 if is_smoke else 14336
    kv_c1 = h_c1 // 4
    warmup_c1 = 2 if is_smoke else 5
    iters_c1 = 5 if is_smoke else 15

    shapes_2d = [
        (h_c1, h_c1),
        (kv_c1, h_c1),
        (kv_c1, h_c1),
        (h_c1, h_c1),
        (inter_c1, h_c1),
        (inter_c1, h_c1),
        (h_c1, inter_c1),
    ]

    # Build deterministic Shard(0) DTensor parameters + gradients directly on each rank
    # (for verification on first layer's 7 matrices, also verify against exact 1-GPU Muon!)
    gen_ver = torch.Generator(device=device)
    gen_ver.manual_seed(11870)
    ver_ref_2d: list[torch.Tensor] = []
    ver_dt_2d: list[DTensor] = []
    ver_own_seq_2d: list[DTensor] = []
    ver_own_pipe_2d: list[DTensor] = []
    ver_wrong_2d: list[DTensor] = []

    for _ in range(8):  # 8 layers x 7 = 56 matrices for exact 1-GPU vs 8-GPU verification
        for r_s, c_s in shapes_2d:
            # Use compact shape in verification if full H=4096 so 56 full matrices don't take 20 GiB on every rank!
            rv = min(r_s, 1024)
            cv = min(c_s, 2048)
            wf = torch.randn((rv, cv), generator=gen_ver, device=device, dtype=torch.float32) * 0.02
            gf = torch.randn((rv, cv), generator=gen_ver, device=device, dtype=torch.float32) * 0.01

            pr = wf.clone().requires_grad_(True)
            pr.grad = gf.clone()
            ver_ref_2d.append(pr)

            for lst in (ver_dt_2d, ver_own_seq_2d, ver_own_pipe_2d, ver_wrong_2d):
                ps = distribute_tensor(wf.clone(), mesh, [Shard(0)]).requires_grad_(True)
                ps.grad = distribute_tensor(gf.clone(), mesh, [Shard(0)])
                lst.append(ps)

    opt_ver_ref = torch.optim.Muon(
        ver_ref_2d, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"
    )
    opt_ver_dt = torch.optim.Muon(
        ver_dt_2d, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"
    )
    opt_ver_seq = BatchedAndOwnerShardedMuon(
        ver_own_seq_2d, mode="fsdp2_owner_seq", lr=1e-3, weight_decay=0.1, momentum=0.95
    )
    opt_ver_pipe = BatchedAndOwnerShardedMuon(
        ver_own_pipe_2d, mode="fsdp2_owner_pipelined", lr=1e-3, weight_decay=0.1, momentum=0.95
    )
    opt_ver_wrong = BatchedAndOwnerShardedMuon(
        ver_wrong_2d, mode="fsdp2_shard_local_wrong", lr=1e-3, weight_decay=0.1, momentum=0.95
    )

    # Run 2 steps so momentum buffer is non-zero in step 2!
    for _ in range(2):
        opt_ver_ref.step()
        opt_ver_dt.step()
        opt_ver_seq.step()
        opt_ver_pipe.step()
        opt_ver_wrong.step()

    dt_vs_1g_err = max(rel_l2(p.full_tensor(), r) for p, r in zip(ver_dt_2d, ver_ref_2d))
    seq_vs_1g_err = max(rel_l2(p.full_tensor(), r) for p, r in zip(ver_own_seq_2d, ver_ref_2d))
    pipe_vs_1g_err = max(rel_l2(p.full_tensor(), r) for p, r in zip(ver_own_pipe_2d, ver_ref_2d))
    wrong_vs_1g_err = max(rel_l2(p.full_tensor(), r) for p, r in zip(ver_wrong_2d, ver_ref_2d))
    # Also compute pure update delta error `||(W_2 - W_0) - (W_ref_2 - W_0)|| / ||W_ref_2 - W_0||` for shard_local_wrong!
    gen_ver2 = torch.Generator(device=device)
    gen_ver2.manual_seed(11870)
    wrong_update_err = 0.0
    pipe_update_err = 0.0
    for pr, pw, pp in zip(ver_ref_2d, ver_wrong_2d, ver_own_pipe_2d):
        rv, cv = pr.shape
        w_init = torch.randn((rv, cv), generator=gen_ver2, device=device, dtype=torch.float32) * 0.02
        _ = torch.randn((rv, cv), generator=gen_ver2, device=device, dtype=torch.float32)
        d_ref = pr - w_init
        d_wrong = pw.full_tensor() - w_init
        d_pipe = pp.full_tensor() - w_init
        wrong_update_err = max(wrong_update_err, rel_l2(d_wrong, d_ref))
        pipe_update_err = max(pipe_update_err, rel_l2(d_pipe, d_ref))

    del ver_ref_2d, ver_dt_2d, ver_own_seq_2d, ver_own_pipe_2d, ver_wrong_2d
    torch.cuda.empty_cache()

    # Now benchmark the full 3.49B Llama-3-8B-width (`num_layers_c1` layers = 112 2D + 33 1D DTensors) on 8 GPUs!
    def _make_sharded_llama_params(dtype_w: torch.dtype) -> tuple[list[DTensor], list[DTensor]]:
        g_loc = torch.Generator(device=device)
        g_loc.manual_seed(11880 + rank)
        p2d_list: list[DTensor] = []
        p1d_list: list[DTensor] = []
        for _ in range(num_layers_c1):
            for r_s, c_s in shapes_2d:
                r_loc = r_s // world_size
                w_loc = torch.randn((r_loc, c_s), generator=g_loc, device=device, dtype=dtype_w) * 0.02
                gr_loc = torch.randn((r_loc, c_s), generator=g_loc, device=device, dtype=dtype_w) * 0.02
                dt_w = DTensor.from_local(w_loc, mesh, [Shard(0)], run_check=False).requires_grad_(True)
                dt_w.grad = DTensor.from_local(gr_loc, mesh, [Shard(0)], run_check=False)
                p2d_list.append(dt_w)
            for _ in range(2):
                n_loc = h_c1 // world_size
                wn = torch.ones((n_loc,), device=device, dtype=dtype_w)
                gn = torch.randn((n_loc,), generator=g_loc, device=device, dtype=dtype_w) * 0.01
                dt_n = DTensor.from_local(wn, mesh, [Shard(0)], run_check=False).requires_grad_(True)
                dt_n.grad = DTensor.from_local(gn, mesh, [Shard(0)], run_check=False)
                p1d_list.append(dt_n)
        n_loc = h_c1 // world_size
        wn = torch.ones((n_loc,), device=device, dtype=dtype_w)
        gn = torch.randn((n_loc,), generator=g_loc, device=device, dtype=dtype_w) * 0.01
        dt_n = DTensor.from_local(wn, mesh, [Shard(0)], run_check=False).requires_grad_(True)
        dt_n.grad = DTensor.from_local(gn, mesh, [Shard(0)], run_check=False)
        p1d_list.append(dt_n)
        return p2d_list, p1d_list

    total_params_c1 = num_layers_c1 * sum(r * c for r, c in shapes_2d) + (2 * num_layers_c1 + 1) * h_c1
    c1_modes: dict[str, Any] = {}

    def _bench_c1_mode(c1_name: str) -> dict[str, Any]:
        w_dt = torch.bfloat16 if c1_name == "fsdp2_triton_sr_bf16_inclip" else torch.float32
        p2d, p1d = _make_sharded_llama_params(w_dt)
        p_all = p2d + p1d

        if c1_name == "fsdp2_adamw_foreach_clip":
            opt_a = torch.optim.AdamW(p_all, lr=1.5e-4, foreach=True, fused=False)

            def _run_c1() -> None:
                torch.nn.utils.clip_grad_norm_(p_all, max_norm=1.0, foreach=True)
                opt_a.step()

            _run_c1()
            state_b = sum(
                sum(v._local_tensor.numel() * v._local_tensor.element_size() for v in st.values() if isinstance(v, DTensor))
                for st in opt_a.state.values()
            )
        elif c1_name == "fsdp2_adamw_fused_clip":
            opt_a = torch.optim.AdamW(p_all, lr=1.5e-4, foreach=False, fused=True)

            def _run_c1() -> None:
                torch.nn.utils.clip_grad_norm_(p_all, max_norm=1.0, foreach=True)
                opt_a.step()

            _run_c1()
            state_b = sum(
                sum(v._local_tensor.numel() * v._local_tensor.element_size() for v in st.values() if isinstance(v, DTensor))
                for st in opt_a.state.values()
            )
        elif c1_name in ("fsdp2_triton_int8mv_inclip", "fsdp2_triton_sr_bf16_inclip"):
            v_key = "fp32w_int8mv" if c1_name == "fsdp2_triton_int8mv_inclip" else "sr_bf16"
            opt_t = TritonMultiTensorAdamW(
                p_all, variant=v_key, lr=1.5e-4, max_norm=1.0, distributed_norm=True
            )
            _run_c1 = opt_t.step
            _run_c1()
            state_b = opt_t.state_bytes_local()
        elif c1_name == "fsdp2_pytorch_dtensor_muon":
            opt_m = torch.optim.Muon(
                p2d, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"
            )
            opt_1d = torch.optim.AdamW(p1d, lr=1e-3 * 0.15, fused=True)

            def _run_c1() -> None:
                opt_m.step()
                opt_1d.step()

            _run_c1()
            state_b = sum(
                sum(v._local_tensor.numel() * v._local_tensor.element_size() for v in st.values() if isinstance(v, DTensor))
                for st in list(opt_m.state.values()) + list(opt_1d.state.values())
            )
        else:
            m_mode = "fsdp2_owner_seq" if c1_name == "fsdp2_owner_seq_muon" else "fsdp2_owner_pipelined"
            opt_om = BatchedAndOwnerShardedMuon(
                p2d, mode=m_mode, lr=1e-3, weight_decay=0.1, momentum=0.95, bf16_momentum=True
            )
            opt_1d = torch.optim.AdamW(p1d, lr=1.5e-4, fused=True)

            def _run_c1() -> None:
                opt_om.step()
                opt_1d.step()

            _run_c1()
            state_b = opt_om.state_bytes_local() + sum(
                sum(v._local_tensor.numel() * v._local_tensor.element_size() for v in st.values() if isinstance(v, DTensor))
                for st in opt_1d.state.values()
            )

        param_grad_b = sum(
            p._local_tensor.numel() * p._local_tensor.element_size()
            + p.grad._local_tensor.numel() * p.grad._local_tensor.element_size()
            for p in p_all
        )

        dist.barrier()
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CUDA]) as prof_c1:
            _run_c1()
            torch.cuda.synchronize(device)
        nccl_calls = sum(
            int(e.count)
            for e in prof_c1.key_averages()
            if "nccl" in e.key.lower()
        )

        iters_this = max(3, iters_c1 // 3) if c1_name == "fsdp2_pytorch_dtensor_muon" else iters_c1
        warmup_this = min(2, warmup_c1) if c1_name == "fsdp2_pytorch_dtensor_muon" else warmup_c1
        for _ in range(warmup_this):
            _run_c1()
        dist.barrier()
        torch.cuda.synchronize(device)

        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        samples: list[float] = []
        for _ in range(iters_this):
            dist.barrier()
            s_ev.record()
            _run_c1()
            e_ev.record()
            torch.cuda.synchronize(device)
            dur = torch.tensor(
                [float(s_ev.elapsed_time(e_ev))], device=device, dtype=torch.float64
            )
            dist.all_reduce(dur, op=dist.ReduceOp.MAX)
            samples.append(float(dur.item()))

        med_ms = statistics.median(samples)
        ss = sorted(samples)
        iqr_ms = ss[(3 * (len(ss) - 1)) // 4] - ss[(len(ss) - 1) // 4]
        return {
            "step_ms_median": round(med_ms, 4),
            "step_ms_iqr": round(iqr_ms, 4),
            "cluster_gparams_per_s": round(
                (float(total_params_c1) / (med_ms * 1e-3)) / 1e9, 3
            ),
            "nccl_collective_launches": nccl_calls,
            "per_rank_state_mib": round(state_b / (1024.0**2), 2),
            "per_rank_param_grad_state_mib": round(
                (param_grad_b + state_b) / (1024.0**2), 2
            ),
        }

    for c1_name in (
        "fsdp2_adamw_foreach_clip",
        "fsdp2_adamw_fused_clip",
        "fsdp2_triton_int8mv_inclip",
        "fsdp2_triton_sr_bf16_inclip",
        "fsdp2_pytorch_dtensor_muon",
        "fsdp2_owner_seq_muon",
        "fsdp2_owner_pipelined_muon",
    ):
        c1_modes[c1_name] = _bench_c1_mode(c1_name)
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # Part C.2: End-to-End 8-GPU FSDP2 Training Step (`Fwd + Bwd + Clip + Opt`)
    #           16-Layer 1.01B GQA Transformer (`H = 2048, FFN = 8192, B=2, S=2048`)
    # ------------------------------------------------------------------------
    num_l_e2e = 8 if is_smoke else 16
    h_e2e = 512 if is_smoke else 2048
    i_e2e = 1536 if is_smoke else 8192
    qh_e2e = 8 if is_smoke else 16
    kvh_e2e = 2 if is_smoke else 4
    b_e2e = 2
    s_e2e = 128 if is_smoke else 2048
    warmup_e2e = 2 if is_smoke else 4
    iters_e2e = 4 if is_smoke else 12

    gen_x = torch.Generator(device=device)
    gen_x.manual_seed(11890 + rank)
    x_e2e = torch.randn(
        (b_e2e, s_e2e, h_e2e), generator=gen_x, device=device, dtype=torch.bfloat16
    )

    e2e_modes: dict[str, Any] = {}
    e2e_total_params = 0

    def _bench_e2e_mode(e2e_name: str) -> tuple[int, dict[str, Any]]:
        use_bf16_master = e2e_name == "e2e_fsdp2_triton_sr_bf16"
        red_dtype = (
            torch.bfloat16
            if e2e_name
            in (
                "e2e_fsdp2_triton_int8mv",
                "e2e_fsdp2_triton_sr_bf16",
                "e2e_fsdp2_owner_pipelined_muon",
            )
            else torch.float32
        )
        torch.manual_seed(11895)
        model = GQATransformerModel(
            num_l_e2e, h_e2e, i_e2e, qh_e2e, kvh_e2e
        ).to(device=device, dtype=torch.bfloat16 if use_bf16_master else torch.float32)
        tot_p = sum(p.numel() for p in model.parameters())

        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=red_dtype,
            output_dtype=torch.bfloat16,
        )
        for blk in model.layers:
            fully_shard(blk, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
        fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp)

        params_m = list(model.parameters())
        p2d_m = [p for p in params_m if p.ndim == 2]
        p1d_m = [p for p in params_m if p.ndim != 2]

        if e2e_name == "e2e_fsdp2_adamw_foreach":
            opt_e = torch.optim.AdamW(params_m, lr=1.5e-4, foreach=True, fused=False)

            def _opt_call() -> None:
                torch.nn.utils.clip_grad_norm_(params_m, max_norm=1.0, foreach=True)
                opt_e.step()

        elif e2e_name == "e2e_fsdp2_adamw_fused":
            opt_e = torch.optim.AdamW(params_m, lr=1.5e-4, foreach=False, fused=True)

            def _opt_call() -> None:
                torch.nn.utils.clip_grad_norm_(params_m, max_norm=1.0, foreach=True)
                opt_e.step()

        elif e2e_name in ("e2e_fsdp2_triton_int8mv", "e2e_fsdp2_triton_sr_bf16"):
            vk = "fp32w_int8mv" if e2e_name == "e2e_fsdp2_triton_int8mv" else "sr_bf16"
            opt_te = TritonMultiTensorAdamW(
                params_m, variant=vk, lr=1.5e-4, max_norm=1.0, distributed_norm=True
            )
            _opt_call = opt_te.step
        elif e2e_name == "e2e_fsdp2_pytorch_dtensor_muon":
            opt_me = torch.optim.Muon(
                p2d_m, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"
            )
            opt_1de = torch.optim.AdamW(p1d_m, lr=1.5e-4, fused=True)

            def _opt_call() -> None:
                opt_me.step()
                opt_1de.step()

        else:
            opt_ome = BatchedAndOwnerShardedMuon(
                p2d_m, mode="fsdp2_owner_pipelined", lr=1e-3, weight_decay=0.1, momentum=0.95, bf16_momentum=True
            )
            opt_1de = torch.optim.AdamW(p1d_m, lr=1.5e-4, fused=True)

            def _opt_call() -> None:
                opt_ome.step()
                opt_1de.step()

        def _full_step() -> tuple[float, float]:
            for p in params_m:
                p.grad = None
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev2 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = model(x_e2e)
            loss = out.float().square().mean()
            loss.backward()
            ev1.record()
            _opt_call()
            ev2.record()
            torch.cuda.synchronize(device)
            return float(ev0.elapsed_time(ev2)), float(ev1.elapsed_time(ev2))

        it_this = max(3, iters_e2e // 2) if e2e_name == "e2e_fsdp2_pytorch_dtensor_muon" else iters_e2e
        wm_this = min(2, warmup_e2e) if e2e_name == "e2e_fsdp2_pytorch_dtensor_muon" else warmup_e2e
        for _ in range(wm_this):
            _full_step()
        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        _full_step()
        peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        # Drop gradient shards before measuring persistent post-step parameter+optimizer footprint
        for p in params_m:
            p.grad = None
        steady_alloc_mib = torch.cuda.memory_allocated(device) / (1024.0**2)
        mem_vec = torch.tensor(
            [peak_alloc_mib, steady_alloc_mib], device=device, dtype=torch.float64
        )
        dist.all_reduce(mem_vec, op=dist.ReduceOp.MAX)

        tot_samples: list[float] = []
        opt_samples: list[float] = []
        for _ in range(it_this):
            dist.barrier()
            tot_ms, opt_ms = _full_step()
            t_vec = torch.tensor([tot_ms, opt_ms], device=device, dtype=torch.float64)
            dist.all_reduce(t_vec, op=dist.ReduceOp.MAX)
            tot_samples.append(float(t_vec[0].item()))
            opt_samples.append(float(t_vec[1].item()))

        med_tot = statistics.median(tot_samples)
        med_opt = statistics.median(opt_samples)
        global_tokens = world_size * b_e2e * s_e2e
        return tot_p, {
            "total_step_ms_median": round(med_tot, 4),
            "opt_clip_ms_median": round(med_opt, 4),
            "opt_fraction_of_step": round(med_opt / max(med_tot, 1e-6), 4),
            "cluster_tokens_per_s": round(float(global_tokens) / (med_tot * 1e-3), 1),
            "peak_allocated_mib": round(float(mem_vec[0].item()), 2),
            "steady_post_step_mib": round(float(mem_vec[1].item()), 2),
        }

    for e2e_name in (
        "e2e_fsdp2_adamw_foreach",
        "e2e_fsdp2_adamw_fused",
        "e2e_fsdp2_triton_int8mv",
        "e2e_fsdp2_triton_sr_bf16",
        "e2e_fsdp2_pytorch_dtensor_muon",
        "e2e_fsdp2_owner_pipelined_muon",
    ):
        gc.collect()
        torch.cuda.empty_cache()
        e2e_total_params, e2e_modes[e2e_name] = _bench_e2e_mode(e2e_name)
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "muon_8gpu_vs_1gpu_verification": {
            "pytorch_dtensor_muon_rel_l2": round(dt_vs_1g_err, 7),
            "owner_seq_muon_rel_l2": round(seq_vs_1g_err, 7),
            "owner_pipelined_muon_rel_l2": round(pipe_vs_1g_err, 7),
            "owner_pipelined_update_delta_rel_l2": round(pipe_update_err, 7),
            "shard_local_wrong_param_rel_l2": round(wrong_vs_1g_err, 6),
            "shard_local_wrong_update_delta_rel_l2": round(wrong_update_err, 5),
        },
        "isolated_3p49b_fsdp2_optimizer_step": {
            "num_layers": num_layers_c1,
            "hidden": h_c1,
            "inter_dim": inter_c1,
            "total_params": total_params_c1,
            "modes": c1_modes,
        },
        "end_to_end_1p01b_fsdp2_training_step": {
            "num_layers": num_l_e2e,
            "hidden": h_e2e,
            "inter_dim": i_e2e,
            "total_params": e2e_total_params,
            "global_tokens_per_step": world_size * b_e2e * s_e2e,
            "modes": e2e_modes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E118: Fused Optimizer Kernels, SR-BF16/INT8 AdamW, and FSDP2 Owner-Sharded Muon"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(minutes=30), device_id=device
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"Expected 8 GPUs, got {world_size}"

    t0 = time.perf_counter()
    is_smoke = args.mode == "smoke"

    s0_res: dict[str, Any] = {}
    sa_res: dict[str, Any] = {}
    sb_res: dict[str, Any] = {}

    if rank == 0:
        s0_res = run_suite_0_audit(device, is_smoke)
        print(f"[E118] Suite 0 complete at +{time.perf_counter() - t0:.1f}s", flush=True)
    dist.barrier()

    if rank == 0:
        sa_res = run_suite_a_adamw_roofline(device, is_smoke)
        print(f"[E118] Suite A complete at +{time.perf_counter() - t0:.1f}s", flush=True)
    dist.barrier()

    if rank == 0:
        sb_res = run_suite_b_convergence(device, is_smoke)
        print(f"[E118] Suite B complete at +{time.perf_counter() - t0:.1f}s", flush=True)
    dist.barrier()

    sc_res = run_suite_c_fsdp2(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = time.perf_counter() - t0

    if rank == 0:
        print(f"[E118] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        payload = {
            "experiment": "E118_optimizer_kernels_and_fsdp2_muon",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": round(elapsed_s, 2),
            "environment": environment_metadata(),
            "suite_0_audits": s0_res,
            "suite_a_adamw_roofline": sa_res,
            "suite_b_convergence_and_stagnation": sb_res,
            "suite_c_fsdp2_muon_and_adamw": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
