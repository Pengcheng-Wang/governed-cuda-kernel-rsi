from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import pynvml
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from flash_attn import flash_attn_func
import flash_attn.flash_attn_interface as fai
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.debug import (
    _clear_fast_path_sharding_prop_cache,
    _clear_python_sharding_prop_cache,
)
from torch.distributed.tensor.experimental import _attention as torch_cp_attn
from torch.profiler import ProfilerActivity, profile

from common import environment_metadata


HIDDEN_SIZE = 4096
FFN_SIZE = 14336
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
LOSS_SCALE_CHECK = 1.75
LLAMA3_8B_LAYERS = 32

_wrapped_flash_fwd = fai._wrapped_flash_attn_forward
_wrapped_flash_bwd = fai._wrapped_flash_attn_backward


def write_json_lf(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(output)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalGpuTelemetry:
    def __init__(self, device_index: int) -> None:
        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            parts = [p.strip() for p in visible.split(",") if p.strip()]
            nvml_index = (
                int(parts[device_index])
                if device_index < len(parts) and parts[device_index].isdigit()
                else device_index
            )
        else:
            nvml_index = device_index
        self.nvml_index = nvml_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)

    def sample(self) -> dict[str, float | int]:
        return {
            "gpu_index": self.nvml_index,
            "sm_clock_mhz": int(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            ),
            "mem_clock_mhz": int(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            ),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0, 2),
            "temp_c": int(
                pynvml.nvmlDeviceGetTemperature(
                    self.handle, pynvml.NVML_TEMPERATURE_GPU
                )
            ),
            "pstate": int(pynvml.nvmlDeviceGetPerformanceState(self.handle)),
        }


def rel_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm().item()
    denom = max(expected.float().norm().item(), 1e-8)
    return float(diff / denom)


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def benchmark_distributed_group(
    fn: Callable[[], Any],
    cp_group: dist.ProcessGroup,
    *,
    warmup: int,
    groups: int,
    iterations_per_group: int,
) -> tuple[dict[str, float], float]:
    dist.barrier(group=cp_group)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=cp_group)

    torch.cuda.reset_peak_memory_stats()
    base_alloc_bytes = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    local_inc_mib = max(0.0, (torch.cuda.max_memory_allocated() - base_alloc_bytes) / (1024.0**2))
    mem_tensor = torch.tensor([local_inc_mib], device="cuda", dtype=torch.float64)
    dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX, group=cp_group)
    max_rank_inc_mib = float(mem_tensor.item())

    dist.barrier(group=cp_group)
    crit_samples: list[float] = []
    mean_samples: list[float] = []
    rank_tensor = torch.zeros(1, device="cuda", dtype=torch.float64)

    for _ in range(groups):
        dist.barrier(group=cp_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations_per_group):
            fn()
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end) / iterations_per_group)

        rank_tensor.fill_(elapsed_ms)
        max_tensor = rank_tensor.clone()
        sum_tensor = rank_tensor.clone()
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX, group=cp_group)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM, group=cp_group)
        crit_samples.append(float(max_tensor.item()))
        mean_samples.append(float(sum_tensor.item()) / dist.get_world_size(cp_group))

    stats = {
        "median_ms": round(float(statistics.median(crit_samples)), 4),
        "min_ms": round(float(min(crit_samples)), 4),
        "max_ms": round(float(max(crit_samples)), 4),
        "mean_ms": round(float(statistics.fmean(crit_samples)), 4),
        "rank_mean_median_ms": round(float(statistics.median(mean_samples)), 4),
        "warmup": warmup,
        "groups": groups,
        "iterations_per_group": iterations_per_group,
    }
    return stats, round(max_rank_inc_mib, 2)


# ---------------------------------------------------------------------------
# Fused Triton Online-Softmax LSE Merger (in-place FP32 accumulator)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_lse_init_kernel(
    out_acc_ptr,
    lse_acc_ptr,
    block_out_ptr,
    block_lse_ptr,
    s_total: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid_bt = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    b = pid_bt // s_total
    t = pid_bt % s_total

    lse_idx = (b * n_heads + pid_h) * s_total + t
    l_val = tl.load(block_lse_ptr + lse_idx).to(tl.float32)
    tl.store(lse_acc_ptr + lse_idx, l_val)

    offs_d = tl.arange(0, head_dim).to(tl.int64)
    out_base = ((b * s_total + t) * n_heads + pid_h) * head_dim + offs_d
    o_val = tl.load(block_out_ptr + out_base).to(tl.float32)
    tl.store(out_acc_ptr + out_base, o_val)


@triton.jit
def _fused_lse_merge_kernel(
    out_acc_ptr,
    lse_acc_ptr,
    block_out_ptr,
    block_lse_ptr,
    q_offset,
    s_total: tl.constexpr,
    s_sub: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid_bt = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    b = pid_bt // s_sub
    t = pid_bt % s_sub
    q_pos = q_offset + t

    acc_lse_idx = (b * n_heads + pid_h) * s_total + q_pos
    blk_lse_idx = (b * n_heads + pid_h) * s_sub + t

    l0 = tl.load(lse_acc_ptr + acc_lse_idx).to(tl.float32)
    l1 = tl.load(block_lse_ptr + blk_lse_idx).to(tl.float32)

    delta = l1 - l0
    sigma = tl.sigmoid(delta)
    max_l = tl.maximum(l0, l1)
    l_new = max_l + tl.log(1.0 + tl.exp(-tl.abs(delta)))
    tl.store(lse_acc_ptr + acc_lse_idx, l_new)

    offs_d = tl.arange(0, head_dim).to(tl.int64)
    acc_out_base = ((b * s_total + q_pos) * n_heads + pid_h) * head_dim + offs_d
    blk_out_base = ((b * s_sub + t) * n_heads + pid_h) * head_dim + offs_d

    o0 = tl.load(out_acc_ptr + acc_out_base).to(tl.float32)
    o1 = tl.load(block_out_ptr + blk_out_base).to(tl.float32)
    o_new = o0 + sigma * (o1 - o0)
    tl.store(out_acc_ptr + acc_out_base, o_new)


def triton_lse_init(
    out_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> None:
    bsz, s_total, n_heads, head_dim = out_acc.shape
    grid = (bsz * s_total, n_heads)
    _fused_lse_init_kernel[grid](
        out_acc,
        lse_acc,
        block_out,
        block_lse,
        s_total=s_total,
        n_heads=n_heads,
        head_dim=head_dim,
        num_warps=2,
    )


def triton_lse_merge(
    out_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    q_offset: int,
) -> None:
    bsz, s_total, n_heads, head_dim = out_acc.shape
    s_sub = block_out.shape[1]
    grid = (bsz * s_sub, n_heads)
    _fused_lse_merge_kernel[grid](
        out_acc,
        lse_acc,
        block_out,
        block_lse,
        q_offset,
        s_total=s_total,
        s_sub=s_sub,
        n_heads=n_heads,
        head_dim=head_dim,
        num_warps=2,
    )


def eager_lse_merge(
    out_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    q_offset: int,
) -> None:
    s_sub = block_out.shape[1]
    l0 = lse_acc[:, :, q_offset : q_offset + s_sub]
    l1 = block_lse.float()
    sigma = torch.sigmoid(l1 - l0).transpose(1, 2).unsqueeze(-1)
    o0 = out_acc[:, q_offset : q_offset + s_sub]
    o1 = block_out.float()
    out_acc[:, q_offset : q_offset + s_sub] = o0 + sigma * (o1 - o0)
    lse_acc[:, :, q_offset : q_offset + s_sub] = l0 - F.logsigmoid(l0 - l1)


# ---------------------------------------------------------------------------
# Sequence Sharding / Unsharding Utilities (Contiguous, Zig-Zag, and 2D USP)
# ---------------------------------------------------------------------------


def shard_contiguous(x_global: torch.Tensor, rank: int, cp_size: int) -> torch.Tensor:
    s_local = x_global.shape[1] // cp_size
    return x_global[:, rank * s_local : (rank + 1) * s_local].contiguous()


def unshard_contiguous(x_local: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    torch.cuda.synchronize()
    cp_size = dist.get_world_size(cp_group)
    gathered = [torch.empty_like(x_local) for _ in range(cp_size)]
    dist.all_gather(gathered, x_local.contiguous(), group=cp_group)
    return torch.cat(gathered, dim=1)


def shard_usp_2d(
    x_global: torch.Tensor,
    ring_rank: int,
    ulysses_rank: int,
    ring_size: int,
    ulysses_size: int,
) -> torch.Tensor:
    """
    Partition global sequence [B, S, ...] across a 2D (ring_size, ulysses_size) mesh:
    - Outer Ring dimension splits S into 2 * ring_size Zig-Zag macro-chunks of size M_c = S / (2 * ring_size),
      assigning macro-chunks (ring_rank, 2 * ring_size - 1 - ring_rank) to ring_rank.
    - Inner Ulysses dimension splits each macro-chunk contiguously into ulysses_size sub-slices of size
      s_u = M_c / ulysses_size = S / (2 * ring_size * ulysses_size).
    Note:
    - For (ulysses_size=1, ring_size=CP), this equals standard 1D Zig-Zag Head-Tail sharding.
    - For (ulysses_size=CP, ring_size=1), macro-chunks (0, 1) are the two halves of S; for pure 1D Ulysses
      we also provide contiguous 1D sharding so both are clean.
    """
    bsz, seq_len = x_global.shape[:2]
    rest = x_global.shape[2:]
    macro_c = seq_len // (2 * ring_size)
    sub_u = macro_c // ulysses_size
    chunks = x_global.view(bsz, 2 * ring_size, ulysses_size, sub_u, *rest)
    head_slice = chunks[:, ring_rank, ulysses_rank]
    tail_slice = chunks[:, 2 * ring_size - 1 - ring_rank, ulysses_rank]
    return torch.cat([head_slice, tail_slice], dim=1).contiguous()


def unshard_usp_2d(
    x_local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    ring_size: int,
    ulysses_size: int,
) -> torch.Tensor:
    torch.cuda.synchronize()
    cp_size = dist.get_world_size(cp_group)
    assert cp_size == ring_size * ulysses_size
    bsz, s_local = x_local.shape[:2]
    rest = x_local.shape[2:]
    sub_u = s_local // 2
    gathered = [torch.empty_like(x_local) for _ in range(cp_size)]
    dist.all_gather(gathered, x_local.contiguous(), group=cp_group)

    macro_chunks: list[list[torch.Tensor]] = [
        [None for _ in range(ulysses_size)] for _ in range(2 * ring_size)  # type: ignore[list-item]
    ]
    for g_rank in range(cp_size):
        r = g_rank // ulysses_size
        u = g_rank % ulysses_size
        piece = gathered[g_rank]
        macro_chunks[r][u] = piece[:, :sub_u]
        macro_chunks[2 * ring_size - 1 - r][u] = piece[:, sub_u:]

    ordered = [
        torch.cat(macro_chunks[m_idx], dim=1) for m_idx in range(2 * ring_size)
    ]
    return torch.cat(ordered, dim=1).reshape(bsz, cp_size * s_local, *rest)


# ---------------------------------------------------------------------------
# Contiguous & Zig-Zag Ring FlashAttention-2 Autograd Functions
# ---------------------------------------------------------------------------


class _ContiguousRingFlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ring_pg: dist.ProcessGroup,
        overlap_comm: bool,
    ) -> torch.Tensor:
        bsz, s_local, h_q, d = q.shape
        h_kv = k.shape[2]
        scale = d ** (-0.5)
        ring_rank = dist.get_rank(ring_pg)
        ring_size = dist.get_world_size(ring_pg)
        send_dst = (ring_rank + 1) % ring_size
        recv_src = (ring_rank - 1) % ring_size

        q_c = q.contiguous()
        kv_curr = torch.stack([k.contiguous(), v.contiguous()], dim=0)
        kv_next = torch.empty_like(kv_curr) if ring_size > 1 else kv_curr

        out_acc = torch.empty((bsz, s_local, h_q, d), device=q.device, dtype=torch.float32)
        lse_acc = torch.empty((bsz, h_q, s_local), device=q.device, dtype=torch.float32)

        for s in range(ring_size):
            do_comm = s < ring_size - 1
            reqs: list[Any] = []
            if do_comm:
                ops = [
                    dist.P2POp(dist.isend, kv_curr, group=ring_pg, group_peer=send_dst),
                    dist.P2POp(dist.irecv, kv_next, group=ring_pg, group_peer=recv_src),
                ]
                reqs = dist.batch_isend_irecv(ops)
                if not overlap_comm:
                    for req in reqs:
                        req.wait()

            if s == 0:
                blk_out, blk_lse, _, _ = _wrapped_flash_fwd(
                    q_c, kv_curr[0], kv_curr[1], 0.0, scale, True, -1, -1, 0.0, None, False
                )
                triton_lse_init(out_acc, lse_acc, blk_out, blk_lse)
            elif s <= ring_rank:
                blk_out, blk_lse, _, _ = _wrapped_flash_fwd(
                    q_c, kv_curr[0], kv_curr[1], 0.0, scale, False, -1, -1, 0.0, None, False
                )
                triton_lse_merge(out_acc, lse_acc, blk_out, blk_lse, 0)

            if do_comm:
                if overlap_comm:
                    for req in reqs:
                        req.wait()
                kv_curr, kv_next = kv_next, kv_curr

        out = out_acc.to(q.dtype)
        ctx.save_for_backward(q_c, k.contiguous(), v.contiguous(), out, lse_acc)
        ctx.ring_pg = ring_pg
        ctx.overlap_comm = overlap_comm
        ctx.scale = scale
        ctx.h_kv = h_kv
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q_c, k_c, v_c, out_c, lse_global = ctx.saved_tensors
        ring_pg: dist.ProcessGroup = ctx.ring_pg
        overlap_comm: bool = ctx.overlap_comm
        scale: float = ctx.scale

        ring_rank = dist.get_rank(ring_pg)
        ring_size = dist.get_world_size(ring_pg)
        send_dst = (ring_rank + 1) % ring_size
        recv_src = (ring_rank - 1) % ring_size

        dout_c = dout.contiguous()
        kv_curr = torch.stack([k_c, v_c], dim=0)
        kv_next = torch.empty_like(kv_curr) if ring_size > 1 else kv_curr
        dkv_send = torch.empty_like(kv_curr)
        dkv_recv = torch.empty_like(kv_curr)

        dq_acc = torch.zeros_like(q_c, dtype=torch.float32)
        dq_step = torch.empty_like(q_c)
        dk_step = torch.empty_like(k_c)
        dv_step = torch.empty_like(v_c)

        for s in range(ring_size):
            ops: list[dist.P2POp] = []
            if s < ring_size - 1:
                ops.append(dist.P2POp(dist.isend, kv_curr, group=ring_pg, group_peer=send_dst))
                ops.append(dist.P2POp(dist.irecv, kv_next, group=ring_pg, group_peer=recv_src))
            if s >= 1:
                ops.append(dist.P2POp(dist.isend, dkv_send, group=ring_pg, group_peer=send_dst))
                ops.append(dist.P2POp(dist.irecv, dkv_recv, group=ring_pg, group_peer=recv_src))

            reqs: list[Any] = []
            if ops:
                reqs = dist.batch_isend_irecv(ops)
                if not overlap_comm:
                    for req in reqs:
                        req.wait()

            did_compute = s <= ring_rank
            if did_compute:
                is_causal = s == 0
                _wrapped_flash_bwd(
                    dout_c,
                    q_c,
                    kv_curr[0],
                    kv_curr[1],
                    out_c,
                    lse_global,
                    dq_step,
                    dk_step,
                    dv_step,
                    0.0,
                    scale,
                    is_causal,
                    -1,
                    -1,
                    0.0,
                    None,
                    False,
                    None,
                )
                dq_acc.add_(dq_step)

            if ops and overlap_comm:
                for req in reqs:
                    req.wait()

            if s == 0:
                dkv_send[0].copy_(dk_step)
                dkv_send[1].copy_(dv_step)
            else:
                if did_compute:
                    dkv_send[0].copy_((dkv_recv[0].float() + dk_step.float()).to(k_c.dtype))
                    dkv_send[1].copy_((dkv_recv[1].float() + dv_step.float()).to(v_c.dtype))
                else:
                    dkv_send.copy_(dkv_recv)

            if s < ring_size - 1:
                kv_curr, kv_next = kv_next, kv_curr

        if ring_size > 1:
            reqs = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, dkv_send, group=ring_pg, group_peer=send_dst),
                    dist.P2POp(dist.irecv, dkv_recv, group=ring_pg, group_peer=recv_src),
                ]
            )
            for req in reqs:
                req.wait()
            dk_final, dv_final = dkv_recv[0], dkv_recv[1]
        else:
            dk_final, dv_final = dkv_send[0], dkv_send[1]

        return dq_acc.to(q_c.dtype), dk_final, dv_final, None, None


class _ZigZagRingFlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ring_pg: dist.ProcessGroup,
        overlap_comm: bool,
        use_triton_lse: bool,
    ) -> torch.Tensor:
        bsz, s_local, h_q, d = q.shape
        c = s_local // 2
        scale = d ** (-0.5)
        ring_rank = dist.get_rank(ring_pg)
        ring_size = dist.get_world_size(ring_pg)
        send_dst = (ring_rank + 1) % ring_size
        recv_src = (ring_rank - 1) % ring_size

        q_c = q.contiguous()
        kv_curr = torch.stack([k.contiguous(), v.contiguous()], dim=0)
        kv_next = torch.empty_like(kv_curr) if ring_size > 1 else kv_curr

        out_acc = torch.empty((bsz, s_local, h_q, d), device=q.device, dtype=torch.float32)
        lse_acc = torch.empty((bsz, h_q, s_local), device=q.device, dtype=torch.float32)

        for s in range(ring_size):
            do_comm = s < ring_size - 1
            reqs: list[Any] = []
            if do_comm:
                ops = [
                    dist.P2POp(dist.isend, kv_curr, group=ring_pg, group_peer=send_dst),
                    dist.P2POp(dist.irecv, kv_next, group=ring_pg, group_peer=recv_src),
                ]
                reqs = dist.batch_isend_irecv(ops)
                if not overlap_comm:
                    for req in reqs:
                        req.wait()

            if s == 0:
                blk_out, blk_lse, _, _ = _wrapped_flash_fwd(
                    q_c, kv_curr[0], kv_curr[1], 0.0, scale, True, -1, -1, 0.0, None, False
                )
                if use_triton_lse:
                    triton_lse_init(out_acc, lse_acc, blk_out, blk_lse)
                else:
                    out_acc.copy_(blk_out.float())
                    lse_acc.copy_(blk_lse.float())
            elif s <= ring_rank:
                blk_out, blk_lse, _, _ = _wrapped_flash_fwd(
                    q_c,
                    kv_curr[0, :, :c],
                    kv_curr[1, :, :c],
                    0.0,
                    scale,
                    False,
                    -1,
                    -1,
                    0.0,
                    None,
                    False,
                )
                if use_triton_lse:
                    triton_lse_merge(out_acc, lse_acc, blk_out, blk_lse, 0)
                else:
                    eager_lse_merge(out_acc, lse_acc, blk_out, blk_lse, 0)
            else:
                blk_out, blk_lse, _, _ = _wrapped_flash_fwd(
                    q_c[:, c:],
                    kv_curr[0],
                    kv_curr[1],
                    0.0,
                    scale,
                    False,
                    -1,
                    -1,
                    0.0,
                    None,
                    False,
                )
                if use_triton_lse:
                    triton_lse_merge(out_acc, lse_acc, blk_out, blk_lse, c)
                else:
                    eager_lse_merge(out_acc, lse_acc, blk_out, blk_lse, c)

            if do_comm:
                if overlap_comm:
                    for req in reqs:
                        req.wait()
                kv_curr, kv_next = kv_next, kv_curr

        out = out_acc.to(q.dtype)
        ctx.save_for_backward(q_c, k.contiguous(), v.contiguous(), out, lse_acc)
        ctx.ring_pg = ring_pg
        ctx.overlap_comm = overlap_comm
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q_c, k_c, v_c, out_c, lse_global = ctx.saved_tensors
        ring_pg: dist.ProcessGroup = ctx.ring_pg
        overlap_comm: bool = ctx.overlap_comm
        scale: float = ctx.scale

        bsz, s_local, h_q, d = q_c.shape
        h_kv = k_c.shape[2]
        c = s_local // 2
        ring_rank = dist.get_rank(ring_pg)
        ring_size = dist.get_world_size(ring_pg)
        send_dst = (ring_rank + 1) % ring_size
        recv_src = (ring_rank - 1) % ring_size

        dout_c = dout.contiguous()
        lse_tail_c = lse_global[:, :, c:].contiguous()

        kv_curr = torch.stack([k_c, v_c], dim=0)
        kv_next = torch.empty_like(kv_curr) if ring_size > 1 else kv_curr
        dkv_send = torch.empty_like(kv_curr)
        dkv_recv = torch.empty_like(kv_curr)

        dq_acc = torch.zeros_like(q_c, dtype=torch.float32)
        dq_full = torch.empty((bsz, s_local, h_q, d), device=q_c.device, dtype=q_c.dtype)
        dq_half = torch.empty((bsz, c, h_q, d), device=q_c.device, dtype=q_c.dtype)
        dk_full = torch.empty((bsz, s_local, h_kv, d), device=k_c.device, dtype=k_c.dtype)
        dv_full = torch.empty((bsz, s_local, h_kv, d), device=v_c.device, dtype=v_c.dtype)
        dk_half = torch.empty((bsz, c, h_kv, d), device=k_c.device, dtype=k_c.dtype)
        dv_half = torch.empty((bsz, c, h_kv, d), device=v_c.device, dtype=v_c.dtype)

        for s in range(ring_size):
            ops: list[dist.P2POp] = []
            if s < ring_size - 1:
                ops.append(dist.P2POp(dist.isend, kv_curr, group=ring_pg, group_peer=send_dst))
                ops.append(dist.P2POp(dist.irecv, kv_next, group=ring_pg, group_peer=recv_src))
            if s >= 1:
                ops.append(dist.P2POp(dist.isend, dkv_send, group=ring_pg, group_peer=send_dst))
                ops.append(dist.P2POp(dist.irecv, dkv_recv, group=ring_pg, group_peer=recv_src))

            reqs: list[Any] = []
            if ops:
                reqs = dist.batch_isend_irecv(ops)
                if not overlap_comm:
                    for req in reqs:
                        req.wait()

            if s == 0:
                _wrapped_flash_bwd(
                    dout_c,
                    q_c,
                    kv_curr[0],
                    kv_curr[1],
                    out_c,
                    lse_global,
                    dq_full,
                    dk_full,
                    dv_full,
                    0.0,
                    scale,
                    True,
                    -1,
                    -1,
                    0.0,
                    None,
                    False,
                    None,
                )
                dq_acc.add_(dq_full)
            elif s <= ring_rank:
                _wrapped_flash_bwd(
                    dout_c,
                    q_c,
                    kv_curr[0, :, :c],
                    kv_curr[1, :, :c],
                    out_c,
                    lse_global,
                    dq_full,
                    dk_half,
                    dv_half,
                    0.0,
                    scale,
                    False,
                    -1,
                    -1,
                    0.0,
                    None,
                    False,
                    None,
                )
                dq_acc.add_(dq_full)
            else:
                _wrapped_flash_bwd(
                    dout_c[:, c:],
                    q_c[:, c:],
                    kv_curr[0],
                    kv_curr[1],
                    out_c[:, c:],
                    lse_tail_c,
                    dq_half,
                    dk_full,
                    dv_full,
                    0.0,
                    scale,
                    False,
                    -1,
                    -1,
                    0.0,
                    None,
                    False,
                    None,
                )
                dq_acc[:, c:].add_(dq_half)

            if ops and overlap_comm:
                for req in reqs:
                    req.wait()

            if s == 0:
                dkv_send[0].copy_(dk_full)
                dkv_send[1].copy_(dv_full)
            elif s <= ring_rank:
                dkv_send[0, :, :c].copy_(
                    (dkv_recv[0, :, :c].float() + dk_half.float()).to(k_c.dtype)
                )
                dkv_send[1, :, :c].copy_(
                    (dkv_recv[1, :, :c].float() + dv_half.float()).to(v_c.dtype)
                )
                dkv_send[:, :, c:].copy_(dkv_recv[:, :, c:])
            else:
                dkv_send[0].copy_((dkv_recv[0].float() + dk_full.float()).to(k_c.dtype))
                dkv_send[1].copy_((dkv_recv[1].float() + dv_full.float()).to(v_c.dtype))

            if s < ring_size - 1:
                kv_curr, kv_next = kv_next, kv_curr

        if ring_size > 1:
            reqs = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, dkv_send, group=ring_pg, group_peer=send_dst),
                    dist.P2POp(dist.irecv, dkv_recv, group=ring_pg, group_peer=recv_src),
                ]
            )
            for req in reqs:
                req.wait()
            dk_final, dv_final = dkv_recv[0], dkv_recv[1]
        else:
            dk_final, dv_final = dkv_send[0], dkv_send[1]

        return dq_acc.to(q_c.dtype), dk_final, dv_final, None, None, None


# ---------------------------------------------------------------------------
# Ulysses & Hybrid 2D USP All-to-All Autograd Functions
# ---------------------------------------------------------------------------


def _seq_to_head_1d_raw(x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    u = dist.get_world_size(pg)
    if u == 1:
        return x
    bsz, s_local, h, d = x.shape
    h_u = h // u
    send_buf = x.view(bsz, s_local, u, h_u, d).permute(2, 0, 1, 3, 4).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=pg)
    if bsz == 1:
        return recv_buf.view(1, u * s_local, h_u, d)
    return recv_buf.permute(1, 0, 2, 3, 4).contiguous().view(bsz, u * s_local, h_u, d)


def _head_to_seq_1d_raw(x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    u = dist.get_world_size(pg)
    if u == 1:
        return x
    bsz, s_full, h_u, d = x.shape
    s_local = s_full // u
    if bsz == 1:
        send_buf = x.contiguous().view(u, 1, s_local, h_u, d)
    else:
        send_buf = x.view(bsz, u, s_local, h_u, d).permute(1, 0, 2, 3, 4).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=pg)
    return recv_buf.permute(1, 2, 0, 3, 4).contiguous().view(bsz, s_local, u * h_u, d)


def _seq_to_head_kv_1d_raw(
    k: torch.Tensor, v: torch.Tensor, pg: dist.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor]:
    u = dist.get_world_size(pg)
    if u == 1:
        return k, v
    bsz, s_local, h_kv, d = k.shape
    h_u = h_kv // u
    kv = torch.stack([k, v], dim=2).view(bsz, s_local, 2, u, h_u, d)
    send_buf = kv.permute(3, 2, 0, 1, 4, 5).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=pg)
    if bsz == 1:
        kv_out = recv_buf.permute(1, 2, 0, 3, 4, 5).contiguous().view(2, 1, u * s_local, h_u, d)
    else:
        kv_out = recv_buf.permute(1, 2, 0, 3, 4, 5).contiguous().view(2, bsz, u * s_local, h_u, d)
    return kv_out[0], kv_out[1]


def _head_to_seq_kv_1d_raw(
    dk: torch.Tensor, dv: torch.Tensor, pg: dist.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor]:
    u = dist.get_world_size(pg)
    if u == 1:
        return dk, dv
    bsz, s_full, h_u, d = dk.shape
    s_local = s_full // u
    dkv = torch.stack([dk, dv], dim=0).view(2, bsz, u, s_local, h_u, d)
    send_buf = dkv.permute(2, 0, 1, 3, 4, 5).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=pg)
    out = recv_buf.permute(1, 2, 3, 0, 4, 5).contiguous().view(2, bsz, s_local, u * h_u, d)
    return out[0], out[1]


class _SeqToHead1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
        ctx.pg = pg
        return _seq_to_head_1d_raw(x, pg)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _head_to_seq_1d_raw(grad_out, ctx.pg), None


class _HeadToSeq1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
        ctx.pg = pg
        return _head_to_seq_1d_raw(x, pg)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _seq_to_head_1d_raw(grad_out, ctx.pg), None


class _SeqToHeadKV1D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, k: torch.Tensor, v: torch.Tensor, pg: dist.ProcessGroup
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.pg = pg
        return _seq_to_head_kv_1d_raw(k, v, pg)

    @staticmethod
    def backward(ctx, dk: torch.Tensor, dv: torch.Tensor):
        gk, gv = _head_to_seq_kv_1d_raw(dk, dv, ctx.pg)
        return gk, gv, None


def _seq_to_head_2half_raw(x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    u = dist.get_world_size(pg)
    if u == 1:
        return x
    bsz, s_local, h, d = x.shape
    s_u = s_local // 2
    h_u = h // u
    send_buf = x.view(bsz, 2, s_u, u, h_u, d).permute(3, 0, 1, 2, 4, 5).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=pg)
    return recv_buf.permute(1, 2, 0, 3, 4, 5).contiguous().view(bsz, u * s_local, h_u, d)


def _head_to_seq_2half_raw(x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    u = dist.get_world_size(pg)
    if u == 1:
        return x
    bsz, s_ring, h_u, d = x.shape
    s_u = s_ring // (2 * u)
    send_buf = x.view(bsz, 2, u, s_u, h_u, d).permute(2, 0, 1, 3, 4, 5).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=pg)
    return recv_buf.permute(1, 2, 3, 0, 4, 5).contiguous().view(bsz, 2 * s_u, u * h_u, d)


def _seq_to_head_kv_2half_raw(
    k: torch.Tensor, v: torch.Tensor, pg: dist.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor]:
    u = dist.get_world_size(pg)
    if u == 1:
        return k, v
    bsz, s_local, h_kv, d = k.shape
    s_u = s_local // 2
    h_u = h_kv // u
    kv = torch.stack([k, v], dim=0).view(2, bsz, 2, s_u, u, h_u, d)
    send_buf = kv.permute(4, 0, 1, 2, 3, 5, 6).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=pg)
    kv_out = recv_buf.permute(1, 2, 3, 0, 4, 5, 6).contiguous().view(2, bsz, u * s_local, h_u, d)
    return kv_out[0], kv_out[1]


def _head_to_seq_kv_2half_raw(
    dk: torch.Tensor, dv: torch.Tensor, pg: dist.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor]:
    u = dist.get_world_size(pg)
    if u == 1:
        return dk, dv
    bsz, s_ring, h_u, d = dk.shape
    s_u = s_ring // (2 * u)
    dkv = torch.stack([dk, dv], dim=0).view(2, bsz, 2, u, s_u, h_u, d)
    send_buf = dkv.permute(3, 0, 1, 2, 4, 5, 6).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=pg)
    out = recv_buf.permute(1, 2, 3, 4, 0, 5, 6).contiguous().view(2, bsz, 2 * s_u, u * h_u, d)
    return out[0], out[1]


class _SeqToHead2Half(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
        ctx.pg = pg
        return _seq_to_head_2half_raw(x, pg)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _head_to_seq_2half_raw(grad_out, ctx.pg), None


class _HeadToSeq2Half(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
        ctx.pg = pg
        return _head_to_seq_2half_raw(x, pg)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _seq_to_head_2half_raw(grad_out, ctx.pg), None


class _SeqToHeadKV2Half(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, k: torch.Tensor, v: torch.Tensor, pg: dist.ProcessGroup
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.pg = pg
        return _seq_to_head_kv_2half_raw(k, v, pg)

    @staticmethod
    def backward(ctx, dk: torch.Tensor, dv: torch.Tensor):
        gk, gv = _head_to_seq_kv_2half_raw(dk, dv, ctx.pg)
        return gk, gv, None


def ulysses_attention_1d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ulysses_pg: dist.ProcessGroup,
    *,
    expand_kv_to_q_heads: bool = False,
) -> torch.Tensor:
    u = dist.get_world_size(ulysses_pg)
    h_q = q.shape[2]
    h_kv = k.shape[2]
    if expand_kv_to_q_heads:
        k_in = k.repeat_interleave(h_q // h_kv, dim=2)
        v_in = v.repeat_interleave(h_q // h_kv, dim=2)
    elif h_kv < u:
        rep = u // h_kv
        k_in = k.repeat_interleave(rep, dim=2)
        v_in = v.repeat_interleave(rep, dim=2)
    else:
        k_in, v_in = k, v

    q_h = _SeqToHead1D.apply(q, ulysses_pg)
    k_h, v_h = _SeqToHeadKV1D.apply(k_in, v_in, ulysses_pg)
    out_h = flash_attn_func(q_h, k_h, v_h, causal=True)
    return _HeadToSeq1D.apply(out_h, ulysses_pg)


def usp_hybrid_attention_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ulysses_pg: dist.ProcessGroup,
    ring_pg: dist.ProcessGroup,
) -> torch.Tensor:
    q_h = _SeqToHead2Half.apply(q, ulysses_pg)
    k_h, v_h = _SeqToHeadKV2Half.apply(k, v, ulysses_pg)
    ring_size = dist.get_world_size(ring_pg)
    if ring_size == 1:
        out_h = flash_attn_func(q_h, k_h, v_h, causal=True)
    else:
        out_h = _ZigZagRingFlashAttn.apply(q_h, k_h, v_h, ring_pg, True, True)
    return _HeadToSeq2Half.apply(out_h, ulysses_pg)


# ---------------------------------------------------------------------------
# Sub-communicator Cache & Eager Warmup
# ---------------------------------------------------------------------------


class TopologyGroups:
    def __init__(self, cp_group: dist.ProcessGroup) -> None:
        self.cp_group = cp_group
        self.cp_size = dist.get_world_size(cp_group)
        self.cp_rank = dist.get_rank(cp_group)
        self.usp_groups: dict[
            tuple[int, int], tuple[dist.ProcessGroup | None, dist.ProcessGroup | None]
        ] = {}

        dummy = torch.zeros(4, device="cuda", dtype=torch.float32)
        dist.all_reduce(dummy, group=cp_group)
        torch.cuda.synchronize()

        singleton_pg = None
        for rk in range(self.cp_size):
            pg = dist.new_group(ranks=[rk])
            if rk == self.cp_rank:
                singleton_pg = pg
        assert singleton_pg is not None

        for u in (1, 2, 4, 8):
            if u > self.cp_size or self.cp_size % u != 0:
                continue
            r = self.cp_size // u
            if u == 1:
                u_pg = singleton_pg
                r_pg = cp_group
            elif r == 1:
                u_pg = cp_group
                r_pg = singleton_pg
            else:
                my_u_pg = None
                for r_idx in range(r):
                    ranks = [r_idx * u + u_idx for u_idx in range(u)]
                    pg = dist.new_group(ranks=ranks)
                    if self.cp_rank in ranks:
                        my_u_pg = pg
                my_r_pg = None
                for u_idx in range(u):
                    ranks = [r_idx * u + u_idx for r_idx in range(r)]
                    pg = dist.new_group(ranks=ranks)
                    if self.cp_rank in ranks:
                        my_r_pg = pg
                assert my_u_pg is not None and my_r_pg is not None
                u_pg, r_pg = my_u_pg, my_r_pg

            if u > 1:
                dist.all_reduce(dummy, group=u_pg)
            if r > 1:
                dist.all_reduce(dummy, group=r_pg)
                # Also warm up P2P communicator on r_pg eagerly across all ring peers
                p2p_send = torch.zeros(4, device="cuda", dtype=torch.bfloat16)
                p2p_recv = torch.zeros(4, device="cuda", dtype=torch.bfloat16)
                r_rk = dist.get_rank(r_pg)
                reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, p2p_send, group=r_pg, group_peer=(r_rk + 1) % r),
                        dist.P2POp(dist.irecv, p2p_recv, group=r_pg, group_peer=(r_rk - 1) % r),
                    ]
                )
                for req in reqs:
                    req.wait()
            torch.cuda.synchronize()
            self.usp_groups[(u, r)] = (u_pg, r_pg)


# ---------------------------------------------------------------------------
# Full Long-Context Llama-3-8B GQA Transformer Block (with RoPE)
# ---------------------------------------------------------------------------


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        500000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.bfloat16).view(1, seq_len, 1, head_dim)
    sin = emb.sin().to(torch.bfloat16).view(1, seq_len, 1, head_dim)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _rmsnorm_rope_qkv_swiglu_helpers():
    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        return (xf * inv).to(x.dtype) * weight

    def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return (F.silu(gate.float()) * up.float()).to(gate.dtype)

    return torch.compile(rms_norm), torch.compile(swiglu), torch.compile(apply_rope)


class LongContextLlamaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1_weight = nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))
        self.wq = nn.Linear(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM, bias=False, dtype=torch.bfloat16)
        self.wk = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False, dtype=torch.bfloat16)
        self.wv = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False, dtype=torch.bfloat16)
        self.norm2_weight = nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))
        self.w_gate = nn.Linear(HIDDEN_SIZE, FFN_SIZE, bias=False, dtype=torch.bfloat16)
        self.w_up = nn.Linear(HIDDEN_SIZE, FFN_SIZE, bias=False, dtype=torch.bfloat16)
        self.w_down = nn.Linear(FFN_SIZE, HIDDEN_SIZE, bias=False, dtype=torch.bfloat16)

    def forward_with_attn(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rms_fn: Callable,
        swiglu_fn: Callable,
        rope_fn: Callable,
        attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        bsz, s_len, _ = x.shape
        h1 = rms_fn(x, self.norm1_weight)
        q = self.wq(h1).view(bsz, s_len, NUM_Q_HEADS, HEAD_DIM)
        k = self.wk(h1).view(bsz, s_len, NUM_KV_HEADS, HEAD_DIM)
        v = self.wv(h1).view(bsz, s_len, NUM_KV_HEADS, HEAD_DIM)
        q = rope_fn(q, cos, sin)
        k = rope_fn(k, cos, sin)
        attn_out = attn_fn(q, k, v).reshape(bsz, s_len, HIDDEN_SIZE)
        x = x + self.wo(attn_out)
        h2 = rms_fn(x, self.norm2_weight)
        ffn_act = swiglu_fn(self.w_gate(h2), self.w_up(h2))
        return x + self.w_down(ffn_act)


# ---------------------------------------------------------------------------
# Suite 0: PyTorch 2.13 Experimental context_parallel Audit
# ---------------------------------------------------------------------------


def run_suite_0_pytorch_cp_audit(
    topo: TopologyGroups,
    cp_mesh: Any,
    *,
    smoke: bool,
) -> dict[str, Any]:
    cp_size = topo.cp_size
    cp_rank = topo.cp_rank
    cp_group = topo.cp_group
    seq_len = 4096 if smoke else 32768
    warmup, groups, iters = (2, 3, 2) if smoke else (4, 7, 3)

    # 1. Verify empty-buffers IndexError pitfall in PyTorch 2.13 context_parallel
    empty_buffers_raises_index_error = False
    empty_buffers_error_type = "none"
    try:
        with torch_cp_attn.context_parallel(cp_mesh):
            pass
    except IndexError as exc:
        empty_buffers_raises_index_error = True
        empty_buffers_error_type = f"IndexError: {exc}"
    except Exception as exc:
        empty_buffers_error_type = f"{type(exc).__name__}: {exc}"

    default_rotate_method = torch_cp_attn._cp_options.rotate_method.name

    # Reference tensors on global sequence
    torch.manual_seed(20261115)
    q_glob = torch.randn((1, seq_len, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
    k_glob = torch.randn((1, seq_len, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
    v_glob = torch.randn((1, seq_len, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
    do_glob = torch.randn((1, seq_len, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25

    q_ref = q_glob.detach().clone().requires_grad_(True)
    k_ref = k_glob.detach().clone().requires_grad_(True)
    v_ref = v_glob.detach().clone().requires_grad_(True)
    out_ref = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
    out_ref.backward(do_glob)
    dq_ref = q_ref.grad.detach()
    dk_ref = k_ref.grad.detach()
    dv_ref = v_ref.grad.detach()
    del q_ref, k_ref, v_ref
    torch.cuda.synchronize()

    # Zig-Zag local shards (shared by PyTorch _HeadTailLoadBalancer and our Zig-Zag Ring)
    q_zz = shard_usp_2d(q_glob, cp_rank, 0, cp_size, 1)
    k_zz = shard_usp_2d(k_glob, cp_rank, 0, cp_size, 1)
    v_zz = shard_usp_2d(v_glob, cp_rank, 0, cp_size, 1)
    do_zz = shard_usp_2d(do_glob, cp_rank, 0, cp_size, 1)

    q_bhsd_base = q_zz.transpose(1, 2).contiguous()
    k_bhsd_base = k_zz.transpose(1, 2).contiguous()
    v_bhsd_base = v_zz.transpose(1, 2).contiguous()
    do_bhsd_base = do_zz.transpose(1, 2).contiguous()

    causal_flops = 7.0 * 1 * NUM_Q_HEADS * (seq_len**2) * HEAD_DIM
    records: list[dict[str, Any]] = []

    for rot_mode in ("allgather", "alltoall"):
        torch_cp_attn.set_rotate_method(rot_mode)
        dummy_buf = torch.zeros((1, 1, 2 * cp_size, 1), device="cuda", dtype=torch.bfloat16)
        with torch_cp_attn.context_parallel(
            cp_mesh,
            buffers=[dummy_buf],
            buffer_seq_dims=[2],
            no_restore_buffers={dummy_buf},
        ):
            # Correctness + chain-rule gate
            q_in = q_bhsd_base.detach().clone().requires_grad_(True)
            k_in = k_bhsd_base.detach().clone().requires_grad_(True)
            v_in = v_bhsd_base.detach().clone().requires_grad_(True)
            out_bhsd = F.scaled_dot_product_attention(
                q_in, k_in, v_in, is_causal=True, enable_gqa=True
            )
            out_bhsd.backward(do_bhsd_base)

            out_full = unshard_usp_2d(out_bhsd.detach().transpose(1, 2).contiguous(), cp_group, cp_size, 1)
            dq_full = unshard_usp_2d(q_in.grad.detach().transpose(1, 2).contiguous(), cp_group, cp_size, 1)
            dk_full = unshard_usp_2d(k_in.grad.detach().transpose(1, 2).contiguous(), cp_group, cp_size, 1)
            dv_full = unshard_usp_2d(v_in.grad.detach().transpose(1, 2).contiguous(), cp_group, cp_size, 1)

            q_sc = q_bhsd_base.detach().clone().requires_grad_(True)
            k_sc = k_bhsd_base.detach().clone().requires_grad_(True)
            v_sc = v_bhsd_base.detach().clone().requires_grad_(True)
            out_sc = F.scaled_dot_product_attention(
                q_sc, k_sc, v_sc, is_causal=True, enable_gqa=True
            )
            out_sc.backward(do_bhsd_base * LOSS_SCALE_CHECK)
            chain_dq_err = rel_l2_error(q_sc.grad.detach(), q_in.grad.detach() * LOSS_SCALE_CHECK)
            chain_dk_err = rel_l2_error(k_sc.grad.detach(), k_in.grad.detach() * LOSS_SCALE_CHECK)

            # Profiler collective byte audit (1 fwd+bwd step)
            dist.barrier(group=cp_group)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                q_p = q_bhsd_base.detach().requires_grad_(True)
                k_p = k_bhsd_base.detach().requires_grad_(True)
                v_p = v_bhsd_base.detach().requires_grad_(True)
                o_p = F.scaled_dot_product_attention(
                    q_p, k_p, v_p, is_causal=True, enable_gqa=True
                )
                o_p.backward(do_bhsd_base)
                torch.cuda.synchronize()

            allgather_calls = 0
            alltoall_calls = 0
            aten_repeat_interleave_calls = 0
            comm_input_elems = 0
            for evt in prof.key_averages(group_by_input_shape=True):
                key_name = evt.key
                if "c10d::allgather" in key_name or "_c10d_functional::all_gather" in key_name:
                    allgather_calls += int(evt.count)
                    if evt.input_shapes:
                        for shp in evt.input_shapes:
                            if shp:
                                comm_input_elems += int(math.prod(shp)) * int(evt.count)
                                break
                elif "all_to_all" in key_name:
                    alltoall_calls += int(evt.count)
                    if evt.input_shapes:
                        for shp in evt.input_shapes:
                            if shp:
                                comm_input_elems += int(math.prod(shp)) * int(evt.count)
                                break
                elif "expand" in key_name or "repeat_interleave" in key_name:
                    aten_repeat_interleave_calls += int(evt.count)

            def _step_pytorch_cp() -> None:
                q_t = q_bhsd_base.detach().requires_grad_(True)
                k_t = k_bhsd_base.detach().requires_grad_(True)
                v_t = v_bhsd_base.detach().requires_grad_(True)
                o_t = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, is_causal=True, enable_gqa=True
                )
                o_t.backward(do_bhsd_base)

            stats, inc_mib = benchmark_distributed_group(
                _step_pytorch_cp, cp_group, warmup=warmup, groups=groups, iterations_per_group=iters
            )

        _clear_fast_path_sharding_prop_cache()
        _clear_python_sharding_prop_cache()

        records.append(
            {
                "mode": f"pytorch_cp_{rot_mode}",
                "cp_size": cp_size,
                "seq_len": seq_len,
                "rotate_method": rot_mode,
                "allgather_calls": allgather_calls,
                "alltoall_calls": alltoall_calls,
                "expand_or_repeat_calls": aten_repeat_interleave_calls,
                "comm_input_elems": comm_input_elems,
                "peak_incremental_mib": inc_mib,
                "cluster_tflops": round(causal_flops / (stats["median_ms"] * 1e-3) / 1e12, 2),
                "out_rel_l2": round(rel_l2_error(out_full, out_ref), 6),
                "dq_rel_l2": round(rel_l2_error(dq_full, dq_ref), 6),
                "dk_rel_l2": round(rel_l2_error(dk_full, dk_ref), 6),
                "dv_rel_l2": round(rel_l2_error(dv_full, dv_ref), 6),
                "chain_rule_err": round(max(chain_dq_err, chain_dk_err), 6),
                **stats,
            }
        )

    torch_cp_attn.set_rotate_method("allgather")

    # Compare directly against our Zig-Zag Ring (eager LSE merge vs fused Triton LSE merge) and GQA Ulysses
    for mode_name, use_triton in (
        ("ring_zigzag_eager_lse", False),
        ("ring_zigzag_triton_lse", True),
    ):
        q_in = q_zz.detach().clone().requires_grad_(True)
        k_in = k_zz.detach().clone().requires_grad_(True)
        v_in = v_zz.detach().clone().requires_grad_(True)
        out_zz = _ZigZagRingFlashAttn.apply(
            q_in, k_in, v_in, cp_group, True, use_triton
        )
        out_zz.backward(do_zz)
        out_full = unshard_usp_2d(out_zz.detach(), cp_group, cp_size, 1)
        dq_full = unshard_usp_2d(q_in.grad.detach(), cp_group, cp_size, 1)
        dk_full = unshard_usp_2d(k_in.grad.detach(), cp_group, cp_size, 1)
        dv_full = unshard_usp_2d(v_in.grad.detach(), cp_group, cp_size, 1)

        q_sc = q_zz.detach().clone().requires_grad_(True)
        k_sc = k_zz.detach().clone().requires_grad_(True)
        v_sc = v_zz.detach().clone().requires_grad_(True)
        out_sc = _ZigZagRingFlashAttn.apply(
            q_sc, k_sc, v_sc, cp_group, True, use_triton
        )
        out_sc.backward(do_zz * LOSS_SCALE_CHECK)
        zz_chain_err = max(
            rel_l2_error(q_sc.grad.detach(), q_in.grad.detach() * LOSS_SCALE_CHECK),
            rel_l2_error(k_sc.grad.detach(), k_in.grad.detach() * LOSS_SCALE_CHECK),
        )
        del q_sc, k_sc, v_sc, out_sc

        def _step_zz(ut: bool = use_triton) -> None:
            qt = q_zz.detach().requires_grad_(True)
            kt = k_zz.detach().requires_grad_(True)
            vt = v_zz.detach().requires_grad_(True)
            ot = _ZigZagRingFlashAttn.apply(qt, kt, vt, cp_group, True, ut)
            ot.backward(do_zz)

        stats, inc_mib = benchmark_distributed_group(
            _step_zz, cp_group, warmup=warmup, groups=groups, iterations_per_group=iters
        )
        records.append(
            {
                "mode": mode_name,
                "cp_size": cp_size,
                "seq_len": seq_len,
                "peak_incremental_mib": inc_mib,
                "cluster_tflops": round(causal_flops / (stats["median_ms"] * 1e-3) / 1e12, 2),
                "out_rel_l2": round(rel_l2_error(out_full, out_ref), 6),
                "dq_rel_l2": round(rel_l2_error(dq_full, dq_ref), 6),
                "dk_rel_l2": round(rel_l2_error(dk_full, dk_ref), 6),
                "dv_rel_l2": round(rel_l2_error(dv_full, dv_ref), 6),
                "chain_rule_err": round(zz_chain_err, 6),
                **stats,
            }
        )

    # GQA-native Ulysses on same shape
    q_ct = shard_contiguous(q_glob, cp_rank, cp_size)
    k_ct = shard_contiguous(k_glob, cp_rank, cp_size)
    v_ct = shard_contiguous(v_glob, cp_rank, cp_size)
    do_ct = shard_contiguous(do_glob, cp_rank, cp_size)

    q_u = q_ct.detach().clone().requires_grad_(True)
    k_u = k_ct.detach().clone().requires_grad_(True)
    v_u = v_ct.detach().clone().requires_grad_(True)
    out_u = ulysses_attention_1d(q_u, k_u, v_u, cp_group, expand_kv_to_q_heads=False)
    out_u.backward(do_ct)
    out_u_full = unshard_contiguous(out_u.detach(), cp_group)
    dq_u_full = unshard_contiguous(q_u.grad.detach(), cp_group)
    dk_u_full = unshard_contiguous(k_u.grad.detach(), cp_group)
    dv_u_full = unshard_contiguous(v_u.grad.detach(), cp_group)

    q_u2 = q_ct.detach().clone().requires_grad_(True)
    k_u2 = k_ct.detach().clone().requires_grad_(True)
    v_u2 = v_ct.detach().clone().requires_grad_(True)
    out_u2 = ulysses_attention_1d(q_u2, k_u2, v_u2, cp_group, expand_kv_to_q_heads=False)
    out_u2.backward(do_ct * LOSS_SCALE_CHECK)
    u_chain_err = max(
        rel_l2_error(q_u2.grad.detach(), q_u.grad.detach() * LOSS_SCALE_CHECK),
        rel_l2_error(k_u2.grad.detach(), k_u.grad.detach() * LOSS_SCALE_CHECK),
    )
    del q_u2, k_u2, v_u2, out_u2

    def _step_ulysses() -> None:
        qt = q_ct.detach().requires_grad_(True)
        kt = k_ct.detach().requires_grad_(True)
        vt = v_ct.detach().requires_grad_(True)
        ot = ulysses_attention_1d(qt, kt, vt, cp_group, expand_kv_to_q_heads=False)
        ot.backward(do_ct)

    stats_u, inc_u = benchmark_distributed_group(
        _step_ulysses, cp_group, warmup=warmup, groups=groups, iterations_per_group=iters
    )
    records.append(
        {
            "mode": "ulysses_gqa_native",
            "cp_size": cp_size,
            "seq_len": seq_len,
            "peak_incremental_mib": inc_u,
            "cluster_tflops": round(causal_flops / (stats_u["median_ms"] * 1e-3) / 1e12, 2),
            "out_rel_l2": round(rel_l2_error(out_u_full, out_ref), 6),
            "dq_rel_l2": round(rel_l2_error(dq_u_full, dq_ref), 6),
            "dk_rel_l2": round(rel_l2_error(dk_u_full, dk_ref), 6),
            "dv_rel_l2": round(rel_l2_error(dv_u_full, dv_ref), 6),
            "chain_rule_err": round(u_chain_err, 6),
            **stats_u,
        }
    )

    return {
        "cp_size": cp_size,
        "seq_len": seq_len,
        "default_rotate_method": default_rotate_method,
        "empty_buffers_raises_index_error": empty_buffers_raises_index_error,
        "empty_buffers_error": empty_buffers_error_type,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Suite A: Sequence Scaling (8K..128K) across Ring, Zig-Zag, Ulysses & 2D USP
# ---------------------------------------------------------------------------


def _schedule_for_seq(seq_len: int, smoke: bool) -> tuple[int, int, int]:
    if smoke:
        return (2, 3, 2)
    if seq_len <= 8192:
        return (6, 7, 6)
    if seq_len <= 16384:
        return (5, 7, 4)
    if seq_len <= 32768:
        return (4, 7, 3)
    if seq_len <= 65536:
        return (3, 5, 2)
    return (2, 5, 1)


def run_single_gpu_attention_baselines(smoke: bool) -> list[dict[str, Any]]:
    seq_list = [2048, 4096] if smoke else [8192, 16384, 32768, 65536, 131072]
    out_records: list[dict[str, Any]] = []
    for seq_len in seq_list:
        warmup, groups, iters = _schedule_for_seq(seq_len, smoke)
        if seq_len >= 131072 and not smoke:
            warmup, groups, iters = (1, 3, 1)
        torch.manual_seed(20261115 + seq_len)
        q = torch.randn((1, seq_len, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
        k = torch.randn((1, seq_len, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
        v = torch.randn((1, seq_len, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
        do = torch.randn((1, seq_len, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25

        def _fn() -> None:
            qt = q.detach().requires_grad_(True)
            kt = k.detach().requires_grad_(True)
            vt = v.detach().requires_grad_(True)
            ot = flash_attn_func(qt, kt, vt, causal=True)
            ot.backward(do)

        for _ in range(warmup):
            _fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_b = torch.cuda.memory_allocated()
        _fn()
        torch.cuda.synchronize()
        inc_mib = round(max(0.0, (torch.cuda.max_memory_allocated() - base_b) / (1024.0**2)), 2)

        samples: list[float] = []
        for _ in range(groups):
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            for _ in range(iters):
                _fn()
            e_ev.record()
            e_ev.synchronize()
            samples.append(float(s_ev.elapsed_time(e_ev) / iters))

        med_ms = round(float(statistics.median(samples)), 4)
        causal_flops = 7.0 * 1 * NUM_Q_HEADS * (seq_len**2) * HEAD_DIM
        out_records.append(
            {
                "mode": "single_gpu_flash_attn2",
                "cp_size": 1,
                "seq_len": seq_len,
                "median_ms": med_ms,
                "min_ms": round(float(min(samples)), 4),
                "max_ms": round(float(max(samples)), 4),
                "mean_ms": round(float(statistics.fmean(samples)), 4),
                "peak_incremental_mib": inc_mib,
                "cluster_tflops": round(causal_flops / (med_ms * 1e-3) / 1e12, 2),
                "per_gpu_tflops": round(causal_flops / (med_ms * 1e-3) / 1e12, 2),
            }
        )
        del q, k, v, do
        torch.cuda.empty_cache()
    return out_records


def run_suite_a_sequence_scaling(
    topo: TopologyGroups,
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    cp_size = topo.cp_size
    cp_rank = topo.cp_rank
    cp_group = topo.cp_group
    if smoke:
        seq_list = [2048, 4096]
    elif cp_size == 8:
        seq_list = [8192, 16384, 32768, 65536, 131072]
    else:
        seq_list = [8192, 16384, 32768, 65536]

    records: list[dict[str, Any]] = []
    for seq_len in seq_list:
        warmup, groups, iters = _schedule_for_seq(seq_len, smoke)
        causal_flops = 7.0 * 1 * NUM_Q_HEADS * (seq_len**2) * HEAD_DIM

        # Generate identical global tensors across ranks via deterministic seed; verify against CP=1 up to 64K
        do_ref_check = seq_len <= 65536
        if do_ref_check:
            torch.manual_seed(20261115 + seq_len)
            q_glob = torch.randn((1, seq_len, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
            k_glob = torch.randn((1, seq_len, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
            v_glob = torch.randn((1, seq_len, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
            do_glob = torch.randn((1, seq_len, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25

            qr = q_glob.detach().clone().requires_grad_(True)
            kr = k_glob.detach().clone().requires_grad_(True)
            vr = v_glob.detach().clone().requires_grad_(True)
            or_ref = flash_attn_func(qr, kr, vr, causal=True)
            or_ref.backward(do_glob)
            out_ref = or_ref.detach()
            dq_ref = qr.grad.detach()
            dk_ref = kr.grad.detach()
            dv_ref = vr.grad.detach()
            del qr, kr, vr
        else:
            q_glob = k_glob = v_glob = do_glob = None
            out_ref = dq_ref = dk_ref = dv_ref = None

        s_local = seq_len // cp_size

        def _make_contiguous_shards() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if q_glob is not None:
                return (
                    shard_contiguous(q_glob, cp_rank, cp_size),
                    shard_contiguous(k_glob, cp_rank, cp_size),
                    shard_contiguous(v_glob, cp_rank, cp_size),
                    shard_contiguous(do_glob, cp_rank, cp_size),
                )
            torch.manual_seed(20261199 + seq_len + cp_rank)
            return (
                torch.randn((1, s_local, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25,
                torch.randn((1, s_local, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25,
                torch.randn((1, s_local, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25,
                torch.randn((1, s_local, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25,
            )

        def _make_usp_shards(u_deg: int, r_deg: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            r_rk = cp_rank // u_deg
            u_rk = cp_rank % u_deg
            if q_glob is not None:
                return (
                    shard_usp_2d(q_glob, r_rk, u_rk, r_deg, u_deg),
                    shard_usp_2d(k_glob, r_rk, u_rk, r_deg, u_deg),
                    shard_usp_2d(v_glob, r_rk, u_rk, r_deg, u_deg),
                    shard_usp_2d(do_glob, r_rk, u_rk, r_deg, u_deg),
                )
            torch.manual_seed(20261299 + seq_len + cp_rank)
            return (
                torch.randn((1, s_local, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25,
                torch.randn((1, s_local, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25,
                torch.randn((1, s_local, NUM_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25,
                torch.randn((1, s_local, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25,
            )

        # Build mode list for this CP size
        mode_specs: list[tuple[str, int, int, bool, bool]] = [
            ("ring_contiguous_overlap", 1, cp_size, True, False),
            ("ring_zigzag_seq", 1, cp_size, False, False),
            ("ring_zigzag_overlap", 1, cp_size, True, False),
            ("ulysses_kv_expanded", cp_size, 1, True, True),
            ("ulysses_gqa_native", cp_size, 1, True, False),
        ]
        if cp_size == 4:
            mode_specs.append(("usp_u2_r2", 2, 2, True, False))
        elif cp_size == 8:
            mode_specs.append(("usp_u2_r4", 2, 4, True, False))
            mode_specs.append(("usp_u4_r2", 4, 2, True, False))

        for mode_name, u_deg, r_deg, use_overlap, expand_kv in mode_specs:
            if mode_name == "ring_contiguous_overlap" or r_deg == 1:
                ql, kl, vl, dol = _make_contiguous_shards()
            else:
                ql, kl, vl, dol = _make_usp_shards(u_deg, r_deg)

            u_pg, r_pg = topo.usp_groups[(u_deg, r_deg)]

            def _run_once(
                q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
            ) -> torch.Tensor:
                if mode_name == "ring_contiguous_overlap":
                    return _ContiguousRingFlashAttn.apply(q_in, k_in, v_in, cp_group, use_overlap)
                if u_deg == 1:
                    return _ZigZagRingFlashAttn.apply(q_in, k_in, v_in, cp_group, use_overlap, True)
                if r_deg == 1:
                    return ulysses_attention_1d(q_in, k_in, v_in, cp_group, expand_kv_to_q_heads=expand_kv)
                return usp_hybrid_attention_2d(q_in, k_in, v_in, u_pg, r_pg)

            # Correctness & 1.75x chain-rule check
            q1 = ql.detach().clone().requires_grad_(True)
            k1 = kl.detach().clone().requires_grad_(True)
            v1 = vl.detach().clone().requires_grad_(True)
            o1 = _run_once(q1, k1, v1)
            o1.backward(dol)

            q2 = ql.detach().clone().requires_grad_(True)
            k2 = kl.detach().clone().requires_grad_(True)
            v2 = vl.detach().clone().requires_grad_(True)
            o2 = _run_once(q2, k2, v2)
            o2.backward(dol * LOSS_SCALE_CHECK)
            chain_err = max(
                rel_l2_error(q2.grad.detach(), q1.grad.detach() * LOSS_SCALE_CHECK),
                rel_l2_error(k2.grad.detach(), k1.grad.detach() * LOSS_SCALE_CHECK),
                rel_l2_error(v2.grad.detach(), v1.grad.detach() * LOSS_SCALE_CHECK),
            )

            if do_ref_check and out_ref is not None:
                if mode_name == "ring_contiguous_overlap" or r_deg == 1:
                    out_f = unshard_contiguous(o1.detach(), cp_group)
                    dq_f = unshard_contiguous(q1.grad.detach(), cp_group)
                    dk_f = unshard_contiguous(k1.grad.detach(), cp_group)
                    dv_f = unshard_contiguous(v1.grad.detach(), cp_group)
                else:
                    out_f = unshard_usp_2d(o1.detach(), cp_group, r_deg, u_deg)
                    dq_f = unshard_usp_2d(q1.grad.detach(), cp_group, r_deg, u_deg)
                    dk_f = unshard_usp_2d(k1.grad.detach(), cp_group, r_deg, u_deg)
                    dv_f = unshard_usp_2d(v1.grad.detach(), cp_group, r_deg, u_deg)
                out_err = round(rel_l2_error(out_f, out_ref), 6)
                dq_err = round(rel_l2_error(dq_f, dq_ref), 6)
                dk_err = round(rel_l2_error(dk_f, dk_ref), 6)
                dv_err = round(rel_l2_error(dv_f, dv_ref), 6)
                del out_f, dq_f, dk_f, dv_f
            else:
                out_err = dq_err = dk_err = dv_err = 0.0

            del q1, k1, v1, o1, q2, k2, v2, o2

            def _bench_step() -> None:
                qt = ql.detach().requires_grad_(True)
                kt = kl.detach().requires_grad_(True)
                vt = vl.detach().requires_grad_(True)
                ot = _run_once(qt, kt, vt)
                ot.backward(dol)

            stats, inc_mib = benchmark_distributed_group(
                _bench_step, cp_group, warmup=warmup, groups=groups, iterations_per_group=iters
            )
            cluster_tflops = round(causal_flops / (stats["median_ms"] * 1e-3) / 1e12, 2)
            records.append(
                {
                    "mode": mode_name,
                    "cp_size": cp_size,
                    "ulysses_degree": u_deg,
                    "ring_degree": r_deg,
                    "seq_len": seq_len,
                    "s_local": s_local,
                    "peak_incremental_mib": inc_mib,
                    "cluster_tflops": cluster_tflops,
                    "per_gpu_tflops": round(cluster_tflops / cp_size, 2),
                    "out_rel_l2": out_err,
                    "dq_rel_l2": dq_err,
                    "dk_rel_l2": dk_err,
                    "dv_rel_l2": dv_err,
                    "chain_rule_err": round(chain_err, 6),
                    "ref_checked": do_ref_check,
                    **stats,
                }
            )
            del ql, kl, vl, dol

        del q_glob, k_glob, v_glob, do_glob, out_ref, dq_ref, dk_ref, dv_ref
        torch.cuda.empty_cache()

    return records


# ---------------------------------------------------------------------------
# Suite B: Breaking the CP > H_kv GQA Head-Sharding Wall (CP = 8, H_kv = 8,4,2,1)
# ---------------------------------------------------------------------------


def run_suite_b_sparse_gqa_wall(
    topo: TopologyGroups,
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    cp_size = topo.cp_size
    assert cp_size == 8
    cp_rank = topo.cp_rank
    cp_group = topo.cp_group
    seq_list = [4096] if smoke else [32768, 65536]
    kv_heads_list = [8, 4, 2, 1]

    records: list[dict[str, Any]] = []
    for seq_len in seq_list:
        warmup, groups, iters = _schedule_for_seq(seq_len, smoke)
        causal_flops = 7.0 * 1 * NUM_Q_HEADS * (seq_len**2) * HEAD_DIM

        for h_kv in kv_heads_list:
            torch.manual_seed(20261500 + seq_len + h_kv)
            q_glob = torch.randn((1, seq_len, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
            k_glob = torch.randn((1, seq_len, h_kv, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
            v_glob = torch.randn((1, seq_len, h_kv, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25
            do_glob = torch.randn((1, seq_len, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16) * 0.25

            qr = q_glob.detach().clone().requires_grad_(True)
            kr = k_glob.detach().clone().requires_grad_(True)
            vr = v_glob.detach().clone().requires_grad_(True)
            or_ref = flash_attn_func(qr, kr, vr, causal=True)
            or_ref.backward(do_glob)
            out_ref = or_ref.detach()
            dq_ref = qr.grad.detach()
            dk_ref = kr.grad.detach()
            dv_ref = vr.grad.detach()
            del qr, kr, vr

            max_norep_u = min(cp_size, h_kv)
            matched_r = cp_size // max_norep_u

            candidates = [
                ("ulysses_u8_kv_replicated", 8, 1),
                ("ring_zigzag_u1_r8", 1, 8),
                ("usp_matched_norep", max_norep_u, matched_r),
            ]
            for label, u_deg, r_deg in candidates:
                if r_deg == 1:
                    ql = shard_contiguous(q_glob, cp_rank, cp_size)
                    kl = shard_contiguous(k_glob, cp_rank, cp_size)
                    vl = shard_contiguous(v_glob, cp_rank, cp_size)
                    dol = shard_contiguous(do_glob, cp_rank, cp_size)
                else:
                    r_rk = cp_rank // u_deg
                    u_rk = cp_rank % u_deg
                    ql = shard_usp_2d(q_glob, r_rk, u_rk, r_deg, u_deg)
                    kl = shard_usp_2d(k_glob, r_rk, u_rk, r_deg, u_deg)
                    vl = shard_usp_2d(v_glob, r_rk, u_rk, r_deg, u_deg)
                    dol = shard_usp_2d(do_glob, r_rk, u_rk, r_deg, u_deg)

                u_pg, r_pg = topo.usp_groups[(u_deg, r_deg)]
                kv_rep_factor = max(1, u_deg // h_kv)

                def _exec(qi: torch.Tensor, ki: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
                    if r_deg == 1:
                        return ulysses_attention_1d(qi, ki, vi, cp_group, expand_kv_to_q_heads=False)
                    if u_deg == 1:
                        return _ZigZagRingFlashAttn.apply(qi, ki, vi, cp_group, True, True)
                    return usp_hybrid_attention_2d(qi, ki, vi, u_pg, r_pg)

                q1 = ql.detach().clone().requires_grad_(True)
                k1 = kl.detach().clone().requires_grad_(True)
                v1 = vl.detach().clone().requires_grad_(True)
                o1 = _exec(q1, k1, v1)
                o1.backward(dol)

                if r_deg == 1:
                    out_f = unshard_contiguous(o1.detach(), cp_group)
                    dq_f = unshard_contiguous(q1.grad.detach(), cp_group)
                    dk_f = unshard_contiguous(k1.grad.detach(), cp_group)
                    dv_f = unshard_contiguous(v1.grad.detach(), cp_group)
                else:
                    out_f = unshard_usp_2d(o1.detach(), cp_group, r_deg, u_deg)
                    dq_f = unshard_usp_2d(q1.grad.detach(), cp_group, r_deg, u_deg)
                    dk_f = unshard_usp_2d(k1.grad.detach(), cp_group, r_deg, u_deg)
                    dv_f = unshard_usp_2d(v1.grad.detach(), cp_group, r_deg, u_deg)

                out_err = round(rel_l2_error(out_f, out_ref), 6)
                dq_err = round(rel_l2_error(dq_f, dq_ref), 6)
                dk_err = round(rel_l2_error(dk_f, dk_ref), 6)
                dv_err = round(rel_l2_error(dv_f, dv_ref), 6)
                del q1, k1, v1, o1, out_f, dq_f, dk_f, dv_f

                def _step_b() -> None:
                    qt = ql.detach().requires_grad_(True)
                    kt = kl.detach().requires_grad_(True)
                    vt = vl.detach().requires_grad_(True)
                    ot = _exec(qt, kt, vt)
                    ot.backward(dol)

                stats, inc_mib = benchmark_distributed_group(
                    _step_b, cp_group, warmup=warmup, groups=groups, iterations_per_group=iters
                )
                records.append(
                    {
                        "mode": label,
                        "cp_size": cp_size,
                        "num_q_heads": NUM_Q_HEADS,
                        "num_kv_heads": h_kv,
                        "gqa_group_size": NUM_Q_HEADS // h_kv,
                        "ulysses_degree": u_deg,
                        "ring_degree": r_deg,
                        "kv_replication_factor": kv_rep_factor,
                        "seq_len": seq_len,
                        "peak_incremental_mib": inc_mib,
                        "cluster_tflops": round(causal_flops / (stats["median_ms"] * 1e-3) / 1e12, 2),
                        "out_rel_l2": out_err,
                        "dq_rel_l2": dq_err,
                        "dk_rel_l2": dk_err,
                        "dv_rel_l2": dv_err,
                        **stats,
                    }
                )
                del ql, kl, vl, dol

            del q_glob, k_glob, v_glob, do_glob, out_ref, dq_ref, dk_ref, dv_ref
            torch.cuda.empty_cache()

    return records


# ---------------------------------------------------------------------------
# Suite C: End-to-End Long-Context Llama-3-8B Transformer Block (with RoPE)
# ---------------------------------------------------------------------------


def _block_total_flops(seq_len: int) -> tuple[float, float]:
    linear_fwd = 2.0 * seq_len * (
        2 * HIDDEN_SIZE * (NUM_Q_HEADS * HEAD_DIM)
        + 2 * HIDDEN_SIZE * (NUM_KV_HEADS * HEAD_DIM)
        + 3 * HIDDEN_SIZE * FFN_SIZE
    )
    linear_train = 3.0 * linear_fwd
    attn_train = 7.0 * NUM_Q_HEADS * (seq_len**2) * HEAD_DIM
    return linear_train + attn_train, attn_train


def run_single_gpu_block_baselines(smoke: bool) -> list[dict[str, Any]]:
    seq_list = [2048] if smoke else [16384, 32768, 65536]
    rms_fn, swiglu_fn, rope_fn = _rmsnorm_rope_qkv_swiglu_helpers()
    torch.manual_seed(20261800)
    block = LongContextLlamaBlock().cuda()

    records: list[dict[str, Any]] = []
    for seq_len in seq_list:
        warmup, groups, iters = (2, 3, 2) if smoke else _schedule_for_seq(seq_len, False)
        if seq_len >= 65536 and not smoke:
            warmup, groups, iters = (2, 5, 1)

        torch.manual_seed(20261900 + seq_len)
        x = torch.randn((1, seq_len, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16) * 0.2
        dy = torch.randn((1, seq_len, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16) * 0.2
        cos, sin = build_rope_cache(seq_len, HEAD_DIM, x.device)

        def _attn_1gpu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attn_func(q, k, v, causal=True)

        def _step_1gpu() -> None:
            block.zero_grad(set_to_none=True)
            xt = x.detach().requires_grad_(True)
            yt = block.forward_with_attn(xt, cos, sin, rms_fn, swiglu_fn, rope_fn, _attn_1gpu)
            yt.backward(dy)

        for _ in range(warmup):
            _step_1gpu()
        torch.cuda.synchronize()

        block.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        base_b = torch.cuda.memory_allocated()
        _step_1gpu()
        torch.cuda.synchronize()
        inc_mib = round(max(0.0, (torch.cuda.max_memory_allocated() - base_b) / (1024.0**2)), 2)

        samples: list[float] = []
        for _ in range(groups):
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            for _ in range(iters):
                _step_1gpu()
            e_ev.record()
            e_ev.synchronize()
            samples.append(float(s_ev.elapsed_time(e_ev) / iters))

        med_ms = round(float(statistics.median(samples)), 4)
        total_flops, attn_flops = _block_total_flops(seq_len)
        records.append(
            {
                "mode": "single_gpu_llama3_block",
                "cp_size": 1,
                "seq_len": seq_len,
                "median_ms": med_ms,
                "min_ms": round(float(min(samples)), 4),
                "max_ms": round(float(max(samples)), 4),
                "mean_ms": round(float(statistics.fmean(samples)), 4),
                "peak_incremental_mib": inc_mib,
                "projected_32layer_activation_gib": round(inc_mib * LLAMA3_8B_LAYERS / 1024.0, 2),
                "cluster_tflops": round(total_flops / (med_ms * 1e-3) / 1e12, 2),
                "attn_flop_fraction": round(attn_flops / total_flops, 4),
                "tokens_per_sec": round(seq_len / (med_ms * 1e-3), 1),
            }
        )
        block.zero_grad(set_to_none=True)
        del x, dy, cos, sin
        torch.cuda.empty_cache()

    del block
    torch.cuda.empty_cache()
    return records


def run_suite_c_llama3_block_cp(
    topo: TopologyGroups,
    cp_mesh: Any,
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    cp_size = topo.cp_size
    assert cp_size == 8
    cp_rank = topo.cp_rank
    cp_group = topo.cp_group
    seq_list = [2048, 4096] if smoke else [16384, 32768, 65536, 131072]

    rms_fn, swiglu_fn, rope_fn = _rmsnorm_rope_qkv_swiglu_helpers()
    torch.manual_seed(20261800)
    block = LongContextLlamaBlock().cuda()
    param_list = list(block.parameters())

    records: list[dict[str, Any]] = []
    for seq_len in seq_list:
        warmup, groups, iters = _schedule_for_seq(seq_len, smoke)
        s_local = seq_len // cp_size
        total_flops, attn_flops = _block_total_flops(seq_len)
        cos_glob, sin_glob = build_rope_cache(seq_len, HEAD_DIM, torch.device("cuda"))

        # Single-GPU exact reference at S <= 16384 (or smoke) for full out, dx, and all 9 parameter dW checks!
        do_full_ref = smoke or seq_len == 16384
        if do_full_ref:
            torch.manual_seed(20261900 + seq_len)
            x_glob = torch.randn((1, seq_len, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16) * 0.2
            dy_glob = torch.randn((1, seq_len, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16) * 0.2

            block.zero_grad(set_to_none=True)
            xr = x_glob.detach().clone().requires_grad_(True)
            yr = block.forward_with_attn(
                xr,
                cos_glob,
                sin_glob,
                rms_fn,
                swiglu_fn,
                rope_fn,
                lambda q, k, v: flash_attn_func(q, k, v, causal=True),
            )
            yr.backward(dy_glob)
            y_ref = yr.detach().clone()
            dx_ref = xr.grad.detach().clone()
            dw_ref = {name: p.grad.detach().clone() for name, p in block.named_parameters()}
            block.zero_grad(set_to_none=True)
            del xr, yr
        else:
            x_glob = dy_glob = y_ref = dx_ref = None
            dw_ref = None

        mode_defs = [
            ("ring_contiguous_block", 1, 8),
            ("ring_zigzag_block", 1, 8),
            ("ulysses_gqa_block", 8, 1),
            ("usp_u4_r2_block", 4, 2),
            ("pytorch_cp_allgather_block", 1, 8),
        ]

        for mode_name, u_deg, r_deg in mode_defs:
            is_contig = mode_name in ("ring_contiguous_block", "ulysses_gqa_block")
            if x_glob is not None:
                if is_contig:
                    xl = shard_contiguous(x_glob, cp_rank, cp_size)
                    dyl = shard_contiguous(dy_glob, cp_rank, cp_size)
                    cosl = shard_contiguous(cos_glob, cp_rank, cp_size)
                    sinl = shard_contiguous(sin_glob, cp_rank, cp_size)
                else:
                    r_rk = cp_rank // u_deg
                    u_rk = cp_rank % u_deg
                    xl = shard_usp_2d(x_glob, r_rk, u_rk, r_deg, u_deg)
                    dyl = shard_usp_2d(dy_glob, r_rk, u_rk, r_deg, u_deg)
                    cosl = shard_usp_2d(cos_glob, r_rk, u_rk, r_deg, u_deg)
                    sinl = shard_usp_2d(sin_glob, r_rk, u_rk, r_deg, u_deg)
            else:
                torch.manual_seed(20262100 + seq_len + cp_rank)
                xl = torch.randn((1, s_local, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16) * 0.2
                dyl = torch.randn((1, s_local, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16) * 0.2
                if is_contig:
                    cosl = shard_contiguous(cos_glob, cp_rank, cp_size)
                    sinl = shard_contiguous(sin_glob, cp_rank, cp_size)
                else:
                    r_rk = cp_rank // u_deg
                    u_rk = cp_rank % u_deg
                    cosl = shard_usp_2d(cos_glob, r_rk, u_rk, r_deg, u_deg)
                    sinl = shard_usp_2d(sin_glob, r_rk, u_rk, r_deg, u_deg)

            u_pg, r_pg = topo.usp_groups[(u_deg, r_deg)]

            def _make_attn_fn() -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
                if mode_name == "ring_contiguous_block":
                    return lambda q, k, v: _ContiguousRingFlashAttn.apply(
                        q, k, v, cp_group, True
                    )
                if mode_name == "ring_zigzag_block":
                    return lambda q, k, v: _ZigZagRingFlashAttn.apply(
                        q, k, v, cp_group, True, True
                    )
                if mode_name == "ulysses_gqa_block":
                    return lambda q, k, v: ulysses_attention_1d(
                        q, k, v, cp_group, expand_kv_to_q_heads=False
                    )
                if mode_name == "usp_u4_r2_block":
                    return lambda q, k, v: usp_hybrid_attention_2d(
                        q, k, v, u_pg, r_pg
                    )

                def _pt_cp_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    qb = q.transpose(1, 2).contiguous()
                    kb = k.transpose(1, 2).contiguous()
                    vb = v.transpose(1, 2).contiguous()
                    ob = F.scaled_dot_product_attention(
                        qb, kb, vb, is_causal=True, enable_gqa=True
                    )
                    return ob.transpose(1, 2).contiguous()

                return _pt_cp_attn

            attn_fn = _make_attn_fn()

            def _sync_grads() -> None:
                flat = torch.cat([p.grad.view(-1) for p in param_list])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=cp_group)
                offset = 0
                for p in param_list:
                    n = p.numel()
                    p.grad.copy_(flat[offset : offset + n].view_as(p))
                    offset += n

            def _run_block_with_context(run_body: Callable[[], None]) -> None:
                if mode_name == "pytorch_cp_allgather_block":
                    torch_cp_attn.set_rotate_method("allgather")
                    dummy = torch.zeros((1, 1, 2 * cp_size, 1), device="cuda", dtype=torch.bfloat16)
                    with torch_cp_attn.context_parallel(
                        cp_mesh,
                        buffers=[dummy],
                        buffer_seq_dims=[2],
                        no_restore_buffers={dummy},
                    ):
                        run_body()
                    _clear_fast_path_sharding_prop_cache()
                    _clear_python_sharding_prop_cache()
                else:
                    run_body()

            # Correctness & 1.75x chain-rule verification
            out_err = dx_err = dw_max_err = chain_err = 0.0
            if do_full_ref and y_ref is not None and dx_ref is not None and dw_ref is not None:
                def _verify_body() -> None:
                    nonlocal out_err, dx_err, dw_max_err, chain_err
                    block.zero_grad(set_to_none=True)
                    x1 = xl.detach().clone().requires_grad_(True)
                    y1 = block.forward_with_attn(x1, cosl, sinl, rms_fn, swiglu_fn, rope_fn, attn_fn)
                    y1.backward(dyl)
                    _sync_grads()
                    g1_wq = block.wq.weight.grad.detach().clone()
                    dx1 = x1.grad.detach().clone()

                    if is_contig:
                        yf = unshard_contiguous(y1.detach(), cp_group)
                        dxf = unshard_contiguous(dx1, cp_group)
                    else:
                        yf = unshard_usp_2d(y1.detach(), cp_group, r_deg, u_deg)
                        dxf = unshard_usp_2d(dx1, cp_group, r_deg, u_deg)

                    out_err = round(rel_l2_error(yf, y_ref), 6)
                    dx_err = round(rel_l2_error(dxf, dx_ref), 6)
                    dw_max_err = round(
                        max(rel_l2_error(p.grad.detach(), dw_ref[n]) for n, p in block.named_parameters()),
                        6,
                    )

                    block.zero_grad(set_to_none=True)
                    x2 = xl.detach().clone().requires_grad_(True)
                    y2 = block.forward_with_attn(x2, cosl, sinl, rms_fn, swiglu_fn, rope_fn, attn_fn)
                    y2.backward(dyl * LOSS_SCALE_CHECK)
                    _sync_grads()
                    chain_err = round(
                        max(
                            rel_l2_error(x2.grad.detach(), dx1 * LOSS_SCALE_CHECK),
                            rel_l2_error(block.wq.weight.grad.detach(), g1_wq * LOSS_SCALE_CHECK),
                        ),
                        6,
                    )
                    block.zero_grad(set_to_none=True)

                _run_block_with_context(_verify_body)

            def _bench_pure_cp_step() -> None:
                block.zero_grad(set_to_none=True)
                xt = xl.detach().requires_grad_(True)
                yt = block.forward_with_attn(xt, cosl, sinl, rms_fn, swiglu_fn, rope_fn, attn_fn)
                yt.backward(dyl)

            def _bench_full_dp_sync_step() -> None:
                block.zero_grad(set_to_none=True)
                xt = xl.detach().requires_grad_(True)
                yt = block.forward_with_attn(xt, cosl, sinl, rms_fn, swiglu_fn, rope_fn, attn_fn)
                yt.backward(dyl)
                _sync_grads()

            def _run_benchmarks() -> tuple[dict[str, float], float, dict[str, float]]:
                pure_s, inc_m = benchmark_distributed_group(
                    _bench_pure_cp_step, cp_group, warmup=warmup, groups=groups, iterations_per_group=iters
                )
                sync_s, _ = benchmark_distributed_group(
                    _bench_full_dp_sync_step, cp_group, warmup=warmup, groups=groups, iterations_per_group=iters
                )
                return pure_s, inc_m, sync_s

            if mode_name == "pytorch_cp_allgather_block":
                torch_cp_attn.set_rotate_method("allgather")
                dummy = torch.zeros((1, 1, 2 * cp_size, 1), device="cuda", dtype=torch.bfloat16)
                with torch_cp_attn.context_parallel(
                    cp_mesh,
                    buffers=[dummy],
                    buffer_seq_dims=[2],
                    no_restore_buffers={dummy},
                ):
                    pure_stats, inc_mib, sync_stats = _run_benchmarks()
                _clear_fast_path_sharding_prop_cache()
                _clear_python_sharding_prop_cache()
            else:
                pure_stats, inc_mib, sync_stats = _run_benchmarks()

            block.zero_grad(set_to_none=True)
            records.append(
                {
                    "mode": mode_name,
                    "cp_size": cp_size,
                    "ulysses_degree": u_deg,
                    "ring_degree": r_deg,
                    "seq_len": seq_len,
                    "s_local": s_local,
                    "pure_cp_median_ms": pure_stats["median_ms"],
                    "with_dw_sync_median_ms": sync_stats["median_ms"],
                    "dw_sync_overhead_ms": round(
                        max(0.0, sync_stats["median_ms"] - pure_stats["median_ms"]), 4
                    ),
                    "peak_incremental_mib": inc_mib,
                    "projected_32layer_activation_gib": round(inc_mib * LLAMA3_8B_LAYERS / 1024.0, 2),
                    "cluster_tflops_pure_cp": round(
                        total_flops / (pure_stats["median_ms"] * 1e-3) / 1e12, 2
                    ),
                    "cluster_tflops_with_dw_sync": round(
                        total_flops / (sync_stats["median_ms"] * 1e-3) / 1e12, 2
                    ),
                    "attn_flop_fraction": round(attn_flops / total_flops, 4),
                    "tokens_per_sec_with_dw_sync": round(
                        seq_len / (sync_stats["median_ms"] * 1e-3), 1
                    ),
                    "out_rel_l2": out_err,
                    "dx_rel_l2": dx_err,
                    "dw_max_rel_l2": dw_max_err,
                    "chain_rule_err": chain_err,
                    "ref_checked": do_full_ref,
                    **pure_stats,
                }
            )
            del xl, dyl, cosl, sinl

        del x_glob, dy_glob, y_ref, dx_ref, dw_ref, cos_glob, sin_glob
        torch.cuda.empty_cache()

    del block
    torch.cuda.empty_cache()
    return records


# ---------------------------------------------------------------------------
# Worker & Coordinator Entry Points
# ---------------------------------------------------------------------------


def worker_main(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    cp_size = dist.get_world_size()
    cp_rank = dist.get_rank()
    cp_mesh = init_device_mesh("cuda", (cp_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh.get_group("cp")

    telemetry = LocalGpuTelemetry(local_rank)
    pre_sample = telemetry.sample()

    topo = TopologyGroups(cp_group)

    single_gpu_attn: list[dict[str, Any]] = []
    single_gpu_block: list[dict[str, Any]] = []
    if cp_size == 8:
        if cp_rank == 0:
            single_gpu_attn = run_single_gpu_attention_baselines(args.smoke)
            single_gpu_block = run_single_gpu_block_baselines(args.smoke)
        dist.barrier(group=cp_group)

    suite_0 = run_suite_0_pytorch_cp_audit(topo, cp_mesh, smoke=args.smoke)
    suite_a = run_suite_a_sequence_scaling(topo, smoke=args.smoke)
    suite_b = run_suite_b_sparse_gqa_wall(topo, smoke=args.smoke) if cp_size == 8 else []
    suite_c = run_suite_c_llama3_block_cp(topo, cp_mesh, smoke=args.smoke) if cp_size == 8 else []

    post_sample = telemetry.sample()
    gathered_pre: list[dict[str, Any]] = [{} for _ in range(cp_size)]
    gathered_post: list[dict[str, Any]] = [{} for _ in range(cp_size)]
    dist.all_gather_object(gathered_pre, pre_sample, group=cp_group)
    dist.all_gather_object(gathered_post, post_sample, group=cp_group)

    if cp_rank == 0:
        worker_payload = {
            "cp_size": cp_size,
            "smoke": args.smoke,
            "telemetry_pre": gathered_pre,
            "telemetry_post": gathered_post,
            "single_gpu_attention": single_gpu_attn,
            "single_gpu_block": single_gpu_block,
            "suite_0_pytorch_cp_audit": suite_0,
            "suite_a_sequence_scaling": suite_a,
            "suite_b_sparse_gqa_wall": suite_b,
            "suite_c_llama3_block": suite_c,
        }
        write_json_lf(args.worker_output, worker_payload)

    dist.barrier()
    dist.destroy_process_group()


def coordinator_main(args: argparse.Namespace) -> None:
    cp_sizes = [2, 8] if args.smoke else [2, 4, 8]
    meta = environment_metadata()
    meta["hostname"] = "sanitized-a100-nvswitch-node"

    runs: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="e115_cp_workers_") as td:
        td_path = Path(td)
        for cp_size in cp_sizes:
            port = find_free_port()
            worker_json = td_path / f"worker_cp{cp_size}.json"
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={cp_size}",
                f"--master_port={port}",
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-output",
                str(worker_json),
            ]
            if args.smoke:
                cmd.append("--smoke")
            print(f"[E115] Launching torchrun CP={cp_size} (smoke={args.smoke}) ...", flush=True)
            subprocess.run(cmd, check=True)
            runs[f"cp{cp_size}"] = json.loads(worker_json.read_text(encoding="utf-8"))

    payload = {
        "experiment": "E115_context_parallel_long_context_attention",
        "smoke": args.smoke,
        "environment": meta,
        "model_geometry": {
            "hidden_size": HIDDEN_SIZE,
            "ffn_size": FFN_SIZE,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "reference_layers": LLAMA3_8B_LAYERS,
        },
        "runs": runs,
    }
    write_json_lf(args.output, payload)
    print(f"[E115] Wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E115: Context Parallelism (Ring, Zig-Zag Ring, GQA Ulysses, Hybrid 2D USP, and PyTorch CP) benchmark."
    )
    parser.add_argument("--output", type=str, default="results/raw/context_parallel_attention/smoke.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.worker:
        worker_main(cli_args)
    else:
        coordinator_main(cli_args)
