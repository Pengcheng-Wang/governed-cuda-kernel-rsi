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
import types
import warnings
from pathlib import Path
from typing import Any, Callable

import pynvml
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory as symm_mem
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.utils import fresh_cache
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.profiler import ProfilerActivity, profile

from common import environment_metadata


HIDDEN_SIZE = 4096
FFN_SIZE = 14336
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
LOSS_SCALE_CHECK = 1.75


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
            nvml_index = int(parts[device_index]) if device_index < len(parts) and parts[device_index].isdigit() else device_index
        else:
            nvml_index = device_index
        self.nvml_index = nvml_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)

    def sample(self) -> dict[str, float | int]:
        return {
            "gpu_index": self.nvml_index,
            "sm_clock_mhz": int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)),
            "mem_clock_mhz": int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0, 2),
            "temp_c": int(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)),
            "pstate": int(pynvml.nvmlDeviceGetPerformanceState(self.handle)),
        }


def wait_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, funcol.AsyncCollectiveTensor):
        return tensor.wait()
    return torch.ops._c10d_functional.wait_tensor(tensor) if torch._C._is_tracing() else tensor


def rel_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm().item()
    denom = max(expected.float().norm().item(), 1e-8)
    return float(diff / denom)


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def benchmark_distributed_group(
    fn: Callable[[], Any],
    tp_group: dist.ProcessGroup,
    *,
    warmup: int,
    groups: int,
    iterations_per_group: int,
    between_groups_fn: Callable[[], None] | None = None,
) -> tuple[list[float], list[float]]:
    dist.barrier(group=tp_group)
    torch.cuda.synchronize()
    for _ in range(warmup):
        if between_groups_fn is not None:
            between_groups_fn()
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)

    crit_samples: list[float] = []
    local_samples: list[float] = []
    scratch = torch.empty(1, dtype=torch.float64, device="cuda")
    for _ in range(groups):
        if between_groups_fn is not None:
            between_groups_fn()
            torch.cuda.synchronize()
        dist.barrier(group=tp_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations_per_group):
            fn()
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end) / iterations_per_group)
        local_samples.append(elapsed_ms)
        scratch.fill_(elapsed_ms)
        dist.all_reduce(scratch, op=dist.ReduceOp.MAX, group=tp_group)
        crit_samples.append(float(scratch.item()))
    return crit_samples, local_samples


def summarize_samples(crit_samples: list[float], local_samples: list[float], warmup: int, iters_per_group: int) -> dict[str, float | int]:
    return {
        "median_ms": round(statistics.median(crit_samples), 6),
        "min_ms": round(min(crit_samples), 6),
        "max_ms": round(max(crit_samples), 6),
        "mean_ms": round(statistics.fmean(crit_samples), 6),
        "rank0_median_ms": round(statistics.median(local_samples), 6),
        "groups": len(crit_samples),
        "iterations_per_group": iters_per_group,
        "warmup": warmup,
    }


class _CopyToTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
        ctx.tp_group = tp_group
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        red = funcol.all_reduce(grad_output.contiguous(), "sum", group=ctx.tp_group)
        return wait_if_needed(red), None


class _ReduceFromTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
        red = funcol.all_reduce(x.contiguous(), "sum", group=tp_group)
        return wait_if_needed(red)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class _GatherFromSPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x_shard: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
        ctx.tp_group = tp_group
        gathered = funcol.all_gather_single(x_shard.contiguous(), gather_dim=1, group=tp_group)
        return wait_if_needed(gathered)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        scattered = funcol.reduce_scatter_single(grad_output.contiguous(), "sum", scatter_dim=1, group=ctx.tp_group)
        return wait_if_needed(scattered), None


class _ReduceScatterToSPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x_partial: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
        ctx.tp_group = tp_group
        scattered = funcol.reduce_scatter_single(x_partial.contiguous(), "sum", scatter_dim=1, group=tp_group)
        return wait_if_needed(scattered)

    @staticmethod
    def backward(ctx: Any, grad_shard: torch.Tensor) -> tuple[torch.Tensor, None]:
        gathered = funcol.all_gather_single(grad_shard.contiguous(), gather_dim=1, group=ctx.tp_group)
        return wait_if_needed(gathered), None


class AttentionSubmodule(nn.Module):
    def __init__(self, h: int = HIDDEN_SIZE, n_heads: int = NUM_Q_HEADS, n_kv: int = NUM_KV_HEADS, head_dim: int = HEAD_DIM) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.wq = nn.Linear(h, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(h, n_kv * head_dim, bias=False)
        self.wv = nn.Linear(h, n_kv * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, h, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)
        bsz, seqlen, _ = q.shape
        q = q.view(bsz, seqlen, -1, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, -1, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, -1, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=(q.size(1) != k.size(1)),
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(attn_out)


class FeedForwardSubmodule(nn.Module):
    def __init__(self, h: int = HIDDEN_SIZE, ffn: int = FFN_SIZE) -> None:
        super().__init__()
        self.w_gate = nn.Linear(h, ffn, bias=False)
        self.w_up = nn.Linear(h, ffn, bias=False)
        self.w_down = nn.Linear(ffn, h, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class CanonicalTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(HIDDEN_SIZE, eps=1e-5)
        self.attn = AttentionSubmodule()
        self.ffn_norm = nn.RMSNorm(HIDDEN_SIZE, eps=1e-5)
        self.ffn = FeedForwardSubmodule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FunctionalMegatronBlock(nn.Module):
    def __init__(self, base_state: dict[str, torch.Tensor], rank: int, world_size: int, tp_group: dist.ProcessGroup, sequence_parallel: bool) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.sequence_parallel = sequence_parallel
        self.head_dim = HEAD_DIM

        q_rows = (NUM_Q_HEADS * HEAD_DIM) // world_size
        kv_rows = (NUM_KV_HEADS * HEAD_DIM) // world_size
        ffn_rows = FFN_SIZE // world_size

        self.attn_norm = nn.RMSNorm(HIDDEN_SIZE, eps=1e-5)
        self.ffn_norm = nn.RMSNorm(HIDDEN_SIZE, eps=1e-5)
        self.wq = nn.Linear(HIDDEN_SIZE, q_rows, bias=False)
        self.wk = nn.Linear(HIDDEN_SIZE, kv_rows, bias=False)
        self.wv = nn.Linear(HIDDEN_SIZE, kv_rows, bias=False)
        self.wo = nn.Linear(q_rows, HIDDEN_SIZE, bias=False)
        self.w_gate = nn.Linear(HIDDEN_SIZE, ffn_rows, bias=False)
        self.w_up = nn.Linear(HIDDEN_SIZE, ffn_rows, bias=False)
        self.w_down = nn.Linear(ffn_rows, HIDDEN_SIZE, bias=False)

        with torch.no_grad():
            self.attn_norm.weight.copy_(base_state["attn_norm.weight"])
            self.ffn_norm.weight.copy_(base_state["ffn_norm.weight"])
            self.wq.weight.copy_(base_state["attn.wq.weight"][rank * q_rows : (rank + 1) * q_rows])
            self.wk.weight.copy_(base_state["attn.wk.weight"][rank * kv_rows : (rank + 1) * kv_rows])
            self.wv.weight.copy_(base_state["attn.wv.weight"][rank * kv_rows : (rank + 1) * kv_rows])
            self.wo.weight.copy_(base_state["attn.wo.weight"][:, rank * q_rows : (rank + 1) * q_rows])
            self.w_gate.weight.copy_(base_state["ffn.w_gate.weight"][rank * ffn_rows : (rank + 1) * ffn_rows])
            self.w_up.weight.copy_(base_state["ffn.w_up.weight"][rank * ffn_rows : (rank + 1) * ffn_rows])
            self.w_down.weight.copy_(base_state["ffn.w_down.weight"][:, rank * ffn_rows : (rank + 1) * ffn_rows])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        if self.sequence_parallel:
            h_full = _GatherFromSPRegion.apply(h, self.tp_group)
        else:
            h_full = _CopyToTPRegion.apply(h, self.tp_group)
        q = self.wq(h_full)
        k = self.wk(h_full)
        v = self.wv(h_full)
        bsz, seqlen, _ = q.shape
        q = q.view(bsz, seqlen, -1, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, -1, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, -1, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=(q.size(1) != k.size(1)),
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        proj = self.wo(attn_out)
        if self.sequence_parallel:
            proj_out = _ReduceScatterToSPRegion.apply(proj, self.tp_group)
        else:
            proj_out = _ReduceFromTPRegion.apply(proj, self.tp_group)
        x = x + proj_out

        h2 = self.ffn_norm(x)
        if self.sequence_parallel:
            h2_full = _GatherFromSPRegion.apply(h2, self.tp_group)
        else:
            h2_full = _CopyToTPRegion.apply(h2, self.tp_group)
        ffn_partial = self.w_down(F.silu(self.w_gate(h2_full)) * self.w_up(h2_full))
        if self.sequence_parallel:
            ffn_out = _ReduceScatterToSPRegion.apply(ffn_partial, self.tp_group)
        else:
            ffn_out = _ReduceFromTPRegion.apply(ffn_partial, self.tp_group)
        return x + ffn_out


def build_base_state(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(20261114)
    ref = CanonicalTransformerBlock().to(device=device, dtype=torch.bfloat16)
    return {k: v.detach().clone() for k, v in ref.state_dict().items()}


class SBHAttentionSubmodule(nn.Module):
    def __init__(self, h: int = HIDDEN_SIZE, n_heads: int = NUM_Q_HEADS, n_kv: int = NUM_KV_HEADS, head_dim: int = HEAD_DIM) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.wq = nn.Linear(h, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(h, n_kv * head_dim, bias=False)
        self.wv = nn.Linear(h, n_kv * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, h, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)
        seqlen, bsz, _ = q.shape
        q = q.view(seqlen, bsz, -1, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(seqlen, bsz, -1, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(seqlen, bsz, -1, self.head_dim).permute(1, 2, 0, 3)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=(q.size(1) != k.size(1)),
        )
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(seqlen, bsz, -1)
        return self.wo(attn_out)


class SequenceFirstTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(HIDDEN_SIZE, eps=1e-5)
        self.attn = SBHAttentionSubmodule()
        self.ffn_norm = nn.RMSNorm(HIDDEN_SIZE, eps=1e-5)
        self.ffn = FeedForwardSubmodule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


def make_dtensor_block(
    base_state: dict[str, torch.Tensor],
    mesh: Any,
    *,
    naive_per_linear: bool,
    sequence_first: bool = False,
) -> nn.Module:
    if sequence_first:
        blk: nn.Module = SequenceFirstTransformerBlock().to(device="cuda", dtype=torch.bfloat16)
        blk.load_state_dict(base_state)
        plan = {
            "attn_norm": SequenceParallel(sequence_dim=0),
            "attn": PrepareModuleInput(
                input_layouts=(Shard(0),),
                desired_input_layouts=(Replicate(),),
            ),
            "attn.wq": ColwiseParallel(),
            "attn.wk": ColwiseParallel(),
            "attn.wv": ColwiseParallel(),
            "attn.wo": RowwiseParallel(output_layouts=Shard(0)),
            "ffn_norm": SequenceParallel(sequence_dim=0),
            "ffn": PrepareModuleInput(
                input_layouts=(Shard(0),),
                desired_input_layouts=(Replicate(),),
            ),
            "ffn.w_gate": ColwiseParallel(),
            "ffn.w_up": ColwiseParallel(),
            "ffn.w_down": RowwiseParallel(output_layouts=Shard(0)),
        }
        return parallelize_module(blk, mesh, plan)

    blk = CanonicalTransformerBlock().to(device="cuda", dtype=torch.bfloat16)
    blk.load_state_dict(base_state)
    if naive_per_linear:
        plan = {
            "attn_norm": SequenceParallel(),
            "attn.wq": ColwiseParallel(input_layouts=Shard(1), use_local_output=True),
            "attn.wk": ColwiseParallel(input_layouts=Shard(1), use_local_output=True),
            "attn.wv": ColwiseParallel(input_layouts=Shard(1), use_local_output=True),
            "attn.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "ffn.w_gate": ColwiseParallel(input_layouts=Shard(1), use_local_output=True),
            "ffn.w_up": ColwiseParallel(input_layouts=Shard(1), use_local_output=True),
            "ffn.w_down": RowwiseParallel(output_layouts=Shard(1)),
        }
    else:
        plan = {
            "attn_norm": SequenceParallel(),
            "attn": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "attn.wq": ColwiseParallel(),
            "attn.wk": ColwiseParallel(),
            "attn.wv": ColwiseParallel(),
            "attn.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "ffn": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "ffn.w_gate": ColwiseParallel(),
            "ffn.w_up": ColwiseParallel(),
            "ffn.w_down": RowwiseParallel(output_layouts=Shard(1)),
        }
    return parallelize_module(blk, mesh, plan)


def profile_async_tp_op_counts(step_fn: Callable[[], Any]) -> dict[str, int]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        step_fn()
        torch.cuda.synchronize()
    counts = {
        "symm_fused_ag_mm_calls": 0,
        "symm_fused_mm_rs_calls": 0,
        "c10d_ag_calls": 0,
        "c10d_rs_calls": 0,
        "cat_split_Copy_kernels": 0,
    }
    for evt in prof.key_averages():
        key = evt.key
        c = int(evt.count)
        if "symm_mem::fused_all_gather_matmul" in key:
            counts["symm_fused_ag_mm_calls"] += c
        elif "symm_mem::fused_matmul_reduce_scatter" in key:
            counts["symm_fused_mm_rs_calls"] += c
        elif "_c10d_functional::all_gather_into_tensor" in key:
            counts["c10d_ag_calls"] += c
        elif "_c10d_functional::reduce_scatter_tensor" in key:
            counts["c10d_rs_calls"] += c
        elif "cat_split" in key:
            counts["cat_split_Copy_kernels"] += c
    return counts


def run_deprecation_gate_audit(
    base_state: dict[str, torch.Tensor],
    mesh: Any,
    tp_group: dist.ProcessGroup,
    *,
    smoke: bool,
) -> dict[str, Any]:
    rank = dist.get_rank(group=tp_group)
    world_size = dist.get_world_size(group=tp_group)
    group_name = tp_group.group_name

    bsz, seqlen = (2, 4096) if smoke else (4, 4096)
    s_shard = seqlen // world_size
    torch.manual_seed(3100)
    x_global = torch.randn(seqlen, bsz, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.5
    dy_global = torch.randn(seqlen, bsz, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.5
    x_shard = x_global[rank * s_shard : (rank + 1) * s_shard].contiguous().detach().requires_grad_(True)
    dy_shard = dy_global[rank * s_shard : (rank + 1) * s_shard].contiguous().detach()

    default_enabled = bool(symm_mem.is_symm_mem_enabled_for_group(group_name))
    warmup = 2 if smoke else 4
    groups = 2 if smoke else 6
    iters = 2 if smoke else 4

    # Use a shared temporary cache dir across Steps 1 & 2 (per-rank isolated) to test FX cache poisoning
    with tempfile.TemporaryDirectory(prefix=f"e114_cache_poison_r{rank}_") as shared_cache_dir:
        with torch._inductor.config.patch({"fx_graph_cache": True}):
            old_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = shared_cache_dir
            try:
                # 1. Before enable_symm_mem_for_group (obeying the FutureWarning)
                torch._dynamo.reset()
                torch._inductor.config._micro_pipeline_tp = True
                blk_noop = make_dtensor_block(base_state, mesh, naive_per_linear=False, sequence_first=True)
                cblk_noop = torch.compile(blk_noop, fullgraph=True)

                def step_noop() -> None:
                    blk_noop.zero_grad(set_to_none=True)
                    if x_shard.grad is not None:
                        x_shard.grad = None
                    out = cblk_noop(x_shard)
                    torch.autograd.backward(out, dy_shard)

                for _ in range(2):
                    step_noop()
                torch.cuda.synchronize()
                noop_counts = profile_async_tp_op_counts(step_noop)
                noop_crit, noop_local = benchmark_distributed_group(
                    step_noop, tp_group, warmup=warmup, groups=groups, iterations_per_group=iters
                )
                del cblk_noop, blk_noop
                torch.cuda.empty_cache()

                # 2. Call enable_symm_mem_for_group and recompile in the SAME Inductor cache dir
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    symm_mem.enable_symm_mem_for_group(group_name)
                future_warning_messages = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
                after_enabled = bool(symm_mem.is_symm_mem_enabled_for_group(group_name))

                torch._dynamo.reset()
                torch._inductor.config._micro_pipeline_tp = True
                blk_poisoned = make_dtensor_block(base_state, mesh, naive_per_linear=False, sequence_first=True)
                cblk_poisoned = torch.compile(blk_poisoned, fullgraph=True)

                def step_poisoned() -> None:
                    blk_poisoned.zero_grad(set_to_none=True)
                    if x_shard.grad is not None:
                        x_shard.grad = None
                    out = cblk_poisoned(x_shard)
                    torch.autograd.backward(out, dy_shard)

                for _ in range(2):
                    step_poisoned()
                torch.cuda.synchronize()
                poisoned_counts = profile_async_tp_op_counts(step_poisoned)
                del cblk_poisoned, blk_poisoned
                torch.cuda.empty_cache()
            finally:
                if old_cache_dir is None:
                    os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
                else:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = old_cache_dir

    # 3. Compile in a clean fresh_cache() AFTER enable_symm_mem_for_group
    with fresh_cache():
        torch._dynamo.reset()
        torch._inductor.config._micro_pipeline_tp = True
        blk_active = make_dtensor_block(base_state, mesh, naive_per_linear=False, sequence_first=True)
        cblk_active = torch.compile(blk_active, fullgraph=True)

        def step_active() -> None:
            blk_active.zero_grad(set_to_none=True)
            if x_shard.grad is not None:
                x_shard.grad = None
            out = cblk_active(x_shard)
            torch.autograd.backward(out, dy_shard)

        for _ in range(2):
            step_active()
        torch.cuda.synchronize()
        active_counts = profile_async_tp_op_counts(step_active)
        act_crit, act_local = benchmark_distributed_group(
            step_active, tp_group, warmup=warmup, groups=groups, iterations_per_group=iters
        )
        del cblk_active, blk_active
        torch.cuda.empty_cache()

    noop_stats = summarize_samples(noop_crit, noop_local, warmup, iters)
    act_stats = summarize_samples(act_crit, act_local, warmup, iters)
    return {
        "tp_size": world_size,
        "batch_size": bsz,
        "seq_len": seqlen,
        "global_tokens": bsz * seqlen,
        "default_is_symm_mem_enabled": default_enabled,
        "future_warning_emitted": len(future_warning_messages) > 0,
        "future_warning_text": future_warning_messages[0] if future_warning_messages else "",
        "post_enable_is_symm_mem_enabled": after_enabled,
        "without_enable_call": {
            **noop_stats,
            **noop_counts,
        },
        "same_cache_after_enable_call": {
            **poisoned_counts,
            "fx_cache_poisoned": (
                poisoned_counts["symm_fused_ag_mm_calls"] == 0
                and poisoned_counts["symm_fused_mm_rs_calls"] == 0
            ),
        },
        "with_enable_and_fresh_cache": {
            **act_stats,
            **active_counts,
        },
        "speedup_from_enabling_symm_mem": round(float(noop_stats["median_ms"]) / float(act_stats["median_ms"]), 6),
    }


def run_allreduce_suite(tp_group: dist.ProcessGroup, *, smoke: bool) -> list[dict[str, Any]]:
    rank = dist.get_rank(group=tp_group)
    world_size = dist.get_world_size(group=tp_group)
    group_name = tp_group.group_name

    sizes_bytes = [64 * 1024, 4 * 1024 * 1024] if smoke else [
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
    ]
    ring_slots = 4 if smoke else 8
    warmup = 2 if smoke else 6
    half_groups = 2 if smoke else 4
    iters_per_group = ring_slots

    records: list[dict[str, Any]] = []
    for n_bytes in sizes_bytes:
        numel = n_bytes // 2
        torch.manual_seed(4100 + rank * 17 + numel % 97)
        init_input = (torch.randn(numel, dtype=torch.bfloat16, device="cuda") * 0.125).contiguous()

        ref_fp32 = init_input.float().clone()
        dist.all_reduce(ref_fp32, op=dist.ReduceOp.SUM, group=tp_group)

        nccl_ring = [init_input.clone() for _ in range(ring_slots)]
        plain_inputs = [init_input.clone() for _ in range(ring_slots)]
        symm_one_ring = [symm_mem.empty(numel, dtype=torch.bfloat16, device="cuda") for _ in range(ring_slots)]
        symm_copy_bufs = [symm_mem.empty(numel, dtype=torch.bfloat16, device="cuda") for _ in range(ring_slots)]
        symm_two_ring = [symm_mem.empty(numel, dtype=torch.bfloat16, device="cuda") for _ in range(ring_slots)]
        out_ring = [torch.empty_like(init_input) for _ in range(ring_slots)]

        for slot in range(ring_slots):
            symm_mem.rendezvous(symm_one_ring[slot], group=tp_group)
            symm_mem.rendezvous(symm_copy_bufs[slot], group=tp_group)
            symm_mem.rendezvous(symm_two_ring[slot], group=tp_group)
            symm_one_ring[slot].copy_(init_input)
            symm_two_ring[slot].copy_(init_input)
        dist.barrier(group=tp_group)

        # Single-call numerical accuracy gates against FP32 sum reference
        nccl_check = wait_if_needed(funcol.all_reduce(init_input.clone(), "sum", group=tp_group))
        one_shot_check = torch.ops.symm_mem.one_shot_all_reduce_out(symm_one_ring[0], "sum", group_name, torch.empty_like(init_input))
        one_shot_copy_check = torch.ops.symm_mem.one_shot_all_reduce_copy(symm_copy_bufs[0], init_input, "sum", group_name)
        symm_two_ring[0].copy_(init_input)
        dist.barrier(group=tp_group)
        torch.ops.symm_mem.two_shot_all_reduce_(symm_two_ring[0], "sum", group_name)
        two_shot_check = symm_two_ring[0].clone()

        errors = {
            "nccl_all_reduce": (max_abs_error(nccl_check, ref_fp32), rel_l2_error(nccl_check, ref_fp32)),
            "symm_one_shot_staged": (max_abs_error(one_shot_check, ref_fp32), rel_l2_error(one_shot_check, ref_fp32)),
            "symm_one_shot_with_copy": (max_abs_error(one_shot_copy_check, ref_fp32), rel_l2_error(one_shot_copy_check, ref_fp32)),
            "symm_two_shot_staged": (max_abs_error(two_shot_check, ref_fp32), rel_l2_error(two_shot_check, ref_fp32)),
        }

        idx = {"nccl": 0, "one": 0, "copy": 0, "two": 0}

        def reset_two_shot() -> None:
            idx["two"] = 0
            for slot in range(ring_slots):
                symm_two_ring[slot].copy_(init_input)

        def reset_nccl() -> None:
            idx["nccl"] = 0

        def reset_one() -> None:
            idx["one"] = 0

        def reset_copy() -> None:
            idx["copy"] = 0

        def fn_nccl() -> None:
            s = idx["nccl"] % ring_slots
            idx["nccl"] += 1
            out_ring[s] = wait_if_needed(funcol.all_reduce(nccl_ring[s], "sum", group=tp_group))

        def fn_one_shot() -> None:
            s = idx["one"] % ring_slots
            idx["one"] += 1
            torch.ops.symm_mem.one_shot_all_reduce_out(symm_one_ring[s], "sum", group_name, out_ring[s])

        def fn_one_shot_copy() -> None:
            s = idx["copy"] % ring_slots
            idx["copy"] += 1
            out_ring[s] = torch.ops.symm_mem.one_shot_all_reduce_copy(symm_copy_bufs[s], plain_inputs[s], "sum", group_name)

        def fn_two_shot() -> None:
            s = idx["two"] % ring_slots
            idx["two"] += 1
            torch.ops.symm_mem.two_shot_all_reduce_(symm_two_ring[s], "sum", group_name)

        methods = [
            ("nccl_all_reduce", fn_nccl, reset_nccl, 2.0 * (world_size - 1) / world_size),
            ("symm_one_shot_staged", fn_one_shot, reset_one, float(world_size - 1)),
            ("symm_one_shot_with_copy", fn_one_shot_copy, reset_copy, float(world_size - 1)),
            ("symm_two_shot_staged", fn_two_shot, reset_two_shot, 2.0 * (world_size - 1) / world_size),
        ]

        crit_accum: dict[str, list[float]] = {m[0]: [] for m in methods}
        local_accum: dict[str, list[float]] = {m[0]: [] for m in methods}
        for pass_methods in (methods, list(reversed(methods))):
            for name, fn, reset_fn, _ in pass_methods:
                c_s, l_s = benchmark_distributed_group(
                    fn,
                    tp_group,
                    warmup=warmup,
                    groups=half_groups,
                    iterations_per_group=iters_per_group,
                    between_groups_fn=reset_fn,
                )
                crit_accum[name].extend(c_s)
                local_accum[name].extend(l_s)

        nccl_med = statistics.median(crit_accum["nccl_all_reduce"])
        for name, _, _, p2p_read_mult in methods:
            stats = summarize_samples(crit_accum[name], local_accum[name], warmup, iters_per_group)
            med_s = float(stats["median_ms"]) / 1000.0
            alg_bw = (n_bytes / 1e9) / med_s
            bus_bw = alg_bw * (2.0 * (world_size - 1) / world_size)
            max_e, rel_e = errors[name]
            records.append(
                {
                    "tp_size": world_size,
                    "bytes": n_bytes,
                    "size_kib": n_bytes // 1024,
                    "implementation": name,
                    **stats,
                    "speedup_vs_nccl": round(nccl_med / float(stats["median_ms"]), 6),
                    "alg_bw_gbps": round(alg_bw, 4),
                    "bus_bw_gbps": round(bus_bw, 4),
                    "nvlink_read_byte_multiplier": round(p2p_read_mult, 4),
                    "max_abs_error": round(max_e, 6),
                    "rel_l2_error": round(rel_e, 6),
                }
            )

        del nccl_ring, plain_inputs, symm_one_ring, symm_copy_bufs, symm_two_ring, out_ring
        torch.cuda.empty_cache()

    return records


def run_primitive_overlap_suite(tp_group: dist.ProcessGroup, *, smoke: bool) -> list[dict[str, Any]]:
    rank = dist.get_rank(group=tp_group)
    world_size = dist.get_world_size(group=tp_group)
    group_name = tp_group.group_name

    m_globals = [2048, 16384] if smoke else [2048, 4096, 8192, 16384, 32768]
    warmup = 2 if smoke else 5
    half_groups = 2 if smoke else 4
    iters = 3 if smoke else 6

    ffn_shard = FFN_SIZE // world_size
    scale_h = 1.0 / math.sqrt(HIDDEN_SIZE)
    scale_ffn = 1.0 / math.sqrt(FFN_SIZE)

    records: list[dict[str, Any]] = []
    for m_global in m_globals:
        m_shard = m_global // world_size
        torch.manual_seed(5100 + rank * 31 + m_global)
        a_shard = torch.randn(m_shard, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda").contiguous()
        a_shard_restrided = symm_mem.restride_A_shard_for_fused_all_gather_matmul(a_shard, gather_dim=0)
        w_gate = (torch.randn(HIDDEN_SIZE, ffn_shard, dtype=torch.bfloat16, device="cuda") * scale_h).contiguous()
        w_up = (torch.randn(HIDDEN_SIZE, ffn_shard, dtype=torch.bfloat16, device="cuda") * scale_h).contiguous()
        a_full_dummy = torch.randn(m_global, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda").contiguous()

        def ag_only() -> torch.Tensor:
            return wait_if_needed(funcol.all_gather_single(a_shard, gather_dim=0, group=tp_group))

        def dual_mm_only() -> tuple[torch.Tensor, torch.Tensor]:
            return torch.mm(a_full_dummy, w_gate), torch.mm(a_full_dummy, w_up)

        def seq_ag_dual_mm() -> tuple[torch.Tensor, torch.Tensor]:
            full = wait_if_needed(funcol.all_gather_single(a_shard, gather_dim=0, group=tp_group))
            return torch.mm(full, w_gate), torch.mm(full, w_up)

        def symm_ag_dual_mm() -> list[torch.Tensor]:
            _, outs = torch.ops.symm_mem.fused_all_gather_matmul(
                a_shard_restrided,
                [w_gate, w_up],
                gather_dim=0,
                group_name=group_name,
                return_A=False,
            )
            return outs

        ref_g, ref_u = seq_ag_dual_mm()
        sym_outs = symm_ag_dual_mm()
        ag_rel_err = max(rel_l2_error(sym_outs[0], ref_g), rel_l2_error(sym_outs[1], ref_u))
        ag_max_err = max(max_abs_error(sym_outs[0], ref_g), max_abs_error(sym_outs[1], ref_u))

        ag_methods = [
            ("ag_only", ag_only),
            ("dual_mm_only", dual_mm_only),
            ("seq_ag_dual_mm", seq_ag_dual_mm),
            ("symm_ag_dual_mm", symm_ag_dual_mm),
        ]
        ag_crit: dict[str, list[float]] = {k: [] for k, _ in ag_methods}
        ag_loc: dict[str, list[float]] = {k: [] for k, _ in ag_methods}
        for pass_m in (ag_methods, list(reversed(ag_methods))):
            for k, fn in pass_m:
                c_s, l_s = benchmark_distributed_group(fn, tp_group, warmup=warmup, groups=half_groups, iterations_per_group=iters)
                ag_crit[k].extend(c_s)
                ag_loc[k].extend(l_s)

        t_ag = statistics.median(ag_crit["ag_only"])
        t_mm2 = statistics.median(ag_crit["dual_mm_only"])
        t_seq_ag = statistics.median(ag_crit["seq_ag_dual_mm"])
        t_sym_ag = statistics.median(ag_crit["symm_ag_dual_mm"])
        flops_ag_per_gpu = 2.0 * 2.0 * m_global * HIDDEN_SIZE * ffn_shard
        records.append(
            {
                "tp_size": world_size,
                "primitive": "all_gather_dual_matmul",
                "m_global": m_global,
                "m_shard": m_shard,
                "comm_only_ms": round(t_ag, 6),
                "mm_only_ms": round(t_mm2, 6),
                "nccl_seq_ms": round(t_seq_ag, 6),
                "symm_fused_ms": round(t_sym_ag, 6),
                "speedup_vs_seq": round(t_seq_ag / t_sym_ag, 6),
                "comm_fraction_of_seq": round(t_ag / t_seq_ag, 6),
                "effective_hidden_comm_fraction": round((t_seq_ag - t_sym_ag) / max(t_ag, 1e-6), 6),
                "seq_tflops_per_gpu": round((flops_ag_per_gpu / 1e12) / (t_seq_ag / 1000.0), 4),
                "symm_tflops_per_gpu": round((flops_ag_per_gpu / 1e12) / (t_sym_ag / 1000.0), 4),
                "max_abs_error": round(ag_max_err, 6),
                "rel_l2_error": round(ag_rel_err, 6),
            }
        )

        # Row-parallel GEMM + ReduceScatter: [M_global, FFN_shard] @ [FFN_shard, HIDDEN_SIZE] -> [M_shard, HIDDEN_SIZE]
        x_part = torch.randn(m_global, ffn_shard, dtype=torch.bfloat16, device="cuda").contiguous()
        x_part_restrided = symm_mem.restride_A_for_fused_matmul_reduce_scatter(x_part, scatter_dim=0)
        w_down = (torch.randn(ffn_shard, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * scale_ffn).contiguous()
        y_dummy = torch.randn(m_global, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda").contiguous()

        def down_mm_only() -> torch.Tensor:
            return torch.mm(x_part, w_down)

        def rs_only() -> torch.Tensor:
            return wait_if_needed(funcol.reduce_scatter_single(y_dummy, "sum", scatter_dim=0, group=tp_group))

        def seq_mm_rs() -> torch.Tensor:
            y = torch.mm(x_part, w_down)
            return wait_if_needed(funcol.reduce_scatter_single(y, "sum", scatter_dim=0, group=tp_group))

        def symm_mm_rs() -> torch.Tensor:
            return torch.ops.symm_mem.fused_matmul_reduce_scatter(
                x_part_restrided,
                w_down,
                "sum",
                scatter_dim=0,
                group_name=group_name,
            )

        ref_rs = seq_mm_rs()
        sym_rs = symm_mm_rs()
        rs_rel_err = rel_l2_error(sym_rs, ref_rs)
        rs_max_err = max_abs_error(sym_rs, ref_rs)

        rs_methods = [
            ("down_mm_only", down_mm_only),
            ("rs_only", rs_only),
            ("seq_mm_rs", seq_mm_rs),
            ("symm_mm_rs", symm_mm_rs),
        ]
        rs_crit: dict[str, list[float]] = {k: [] for k, _ in rs_methods}
        rs_loc: dict[str, list[float]] = {k: [] for k, _ in rs_methods}
        for pass_m in (rs_methods, list(reversed(rs_methods))):
            for k, fn in pass_m:
                c_s, l_s = benchmark_distributed_group(fn, tp_group, warmup=warmup, groups=half_groups, iterations_per_group=iters)
                rs_crit[k].extend(c_s)
                rs_loc[k].extend(l_s)

        t_mm1 = statistics.median(rs_crit["down_mm_only"])
        t_rs = statistics.median(rs_crit["rs_only"])
        t_seq_rs = statistics.median(rs_crit["seq_mm_rs"])
        t_sym_rs = statistics.median(rs_crit["symm_mm_rs"])
        flops_rs_per_gpu = 2.0 * m_global * ffn_shard * HIDDEN_SIZE
        records.append(
            {
                "tp_size": world_size,
                "primitive": "matmul_reduce_scatter",
                "m_global": m_global,
                "m_shard": m_shard,
                "comm_only_ms": round(t_rs, 6),
                "mm_only_ms": round(t_mm1, 6),
                "nccl_seq_ms": round(t_seq_rs, 6),
                "symm_fused_ms": round(t_sym_rs, 6),
                "speedup_vs_seq": round(t_seq_rs / t_sym_rs, 6),
                "comm_fraction_of_seq": round(t_rs / t_seq_rs, 6),
                "effective_hidden_comm_fraction": round((t_seq_rs - t_sym_rs) / max(t_rs, 1e-6), 6),
                "seq_tflops_per_gpu": round((flops_rs_per_gpu / 1e12) / (t_seq_rs / 1000.0), 4),
                "symm_tflops_per_gpu": round((flops_rs_per_gpu / 1e12) / (t_sym_rs / 1000.0), 4),
                "max_abs_error": round(rs_max_err, 6),
                "rel_l2_error": round(rs_rel_err, 6),
            }
        )

        del a_shard, a_shard_restrided, w_gate, w_up, a_full_dummy, x_part, x_part_restrided, w_down, y_dummy
        torch.cuda.empty_cache()

    return records


def compute_reference_shards(
    base_state: dict[str, torch.Tensor],
    x_global: torch.Tensor,
    dy_global: torch.Tensor,
    rank: int,
    world_size: int,
) -> dict[str, torch.Tensor]:
    ref_blk = CanonicalTransformerBlock().to(device="cuda", dtype=torch.bfloat16)
    ref_blk.load_state_dict(base_state)
    x_ref = x_global.detach().clone().requires_grad_(True)
    y_ref = ref_blk(x_ref)
    torch.autograd.backward(y_ref, dy_global)

    seqlen = x_global.size(1)
    s_shard = seqlen // world_size
    q_rows = (NUM_Q_HEADS * HEAD_DIM) // world_size
    kv_rows = (NUM_KV_HEADS * HEAD_DIM) // world_size
    ffn_rows = FFN_SIZE // world_size

    out = {
        "y_shard": y_ref[:, rank * s_shard : (rank + 1) * s_shard].detach().clone(),
        "dx_shard": x_ref.grad[:, rank * s_shard : (rank + 1) * s_shard].detach().clone(),
        "dw_wq": ref_blk.attn.wq.weight.grad[rank * q_rows : (rank + 1) * q_rows].detach().clone(),
        "dw_wk": ref_blk.attn.wk.weight.grad[rank * kv_rows : (rank + 1) * kv_rows].detach().clone(),
        "dw_wo": ref_blk.attn.wo.weight.grad[:, rank * q_rows : (rank + 1) * q_rows].detach().clone(),
        "dw_w_gate": ref_blk.ffn.w_gate.weight.grad[rank * ffn_rows : (rank + 1) * ffn_rows].detach().clone(),
        "dw_w_down": ref_blk.ffn.w_down.weight.grad[:, rank * ffn_rows : (rank + 1) * ffn_rows].detach().clone(),
    }
    del ref_blk, x_ref, y_ref
    torch.cuda.empty_cache()
    return out


def block_training_flops_global(bsz: int, seqlen: int) -> float:
    bt = float(bsz * seqlen)
    qkv_proj = 2.0 * bt * HIDDEN_SIZE * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    o_proj = 2.0 * bt * (NUM_Q_HEADS * HEAD_DIM) * HIDDEN_SIZE
    mlp_proj = 3.0 * (2.0 * bt * HIDDEN_SIZE * FFN_SIZE)
    linear_fwd = qkv_proj + o_proj + mlp_proj
    # Causal attention forward FLOPs: 2 * B * Q_heads * (S^2 / 2) * D * 2 (QK^T + PV) = 2 * B * H_q * S^2 * D
    attn_fwd = 2.0 * bsz * NUM_Q_HEADS * (float(seqlen) ** 2) * HEAD_DIM
    # Forward + backward (2x linear bwd, 2.5x causal FlashAttention bwd)
    return 3.0 * linear_fwd + 3.5 * attn_fwd


def extract_local_weight_grads(module: nn.Module, is_dtensor: bool) -> dict[str, torch.Tensor]:
    def to_loc(t: torch.Tensor) -> torch.Tensor:
        return t.to_local() if hasattr(t, "to_local") else t

    if is_dtensor:
        return {
            "dw_wq": to_loc(module.attn.wq.weight.grad),
            "dw_wk": to_loc(module.attn.wk.weight.grad),
            "dw_wo": to_loc(module.attn.wo.weight.grad),
            "dw_w_gate": to_loc(module.ffn.w_gate.weight.grad),
            "dw_w_down": to_loc(module.ffn.w_down.weight.grad),
        }
    return {
        "dw_wq": module.wq.weight.grad,
        "dw_wk": module.wk.weight.grad,
        "dw_wo": module.wo.weight.grad,
        "dw_w_gate": module.w_gate.weight.grad,
        "dw_w_down": module.w_down.weight.grad,
    }


def run_transformer_block_suite(
    base_state: dict[str, torch.Tensor],
    mesh: Any,
    tp_group: dist.ProcessGroup,
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    rank = dist.get_rank(group=tp_group)
    world_size = dist.get_world_size(group=tp_group)

    shapes = [(1, 4096), (4, 4096)] if smoke else [(1, 4096), (2, 4096), (4, 4096), (8, 4096)]
    warmup = 2 if smoke else 4
    half_groups = 2 if smoke else 4
    iters = 2 if smoke else 4

    # (impl_name, use_sp, is_dtensor, naive_sp, do_compile, sequence_first, async_tp)
    mode_specs = [
        ("megatron_tp_func_eager", False, False, False, False, False, False),
        ("megatron_sp_func_eager", True, False, False, False, False, False),
        ("dtensor_sp_naive_eager", True, True, True, False, False, False),
        ("dtensor_sp_shared_eager", True, True, False, False, False, False),
        ("dtensor_bsh_sp_compiled", True, True, False, True, False, False),
        ("dtensor_bsh_async_tp_compiled", True, True, False, True, False, True),
        ("dtensor_sbh_sp_compiled", True, True, False, True, True, False),
        ("dtensor_sbh_async_tp_compiled", True, True, False, True, True, True),
    ]

    records: list[dict[str, Any]] = []
    for bsz, seqlen in shapes:
        s_shard = seqlen // world_size
        torch.manual_seed(6100 + bsz * 13 + seqlen)
        x_global = (torch.randn(bsz, seqlen, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
        dy_global = (torch.randn(bsz, seqlen, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
        x_sbh_global = x_global.transpose(0, 1).contiguous()
        dy_sbh_global = dy_global.transpose(0, 1).contiguous()

        ref_shards = compute_reference_shards(base_state, x_global, dy_global, rank, world_size)
        dist.barrier(group=tp_group)

        torch._dynamo.reset()
        built_runners: list[tuple[str, Callable[[], None], dict[str, Any]]] = []
        for impl_name, use_sp, is_dtensor, naive_sp, do_compile, seq_first, async_tp in mode_specs:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            dist.barrier(group=tp_group)

            if is_dtensor:
                raw_mod = make_dtensor_block(base_state, mesh, naive_per_linear=naive_sp, sequence_first=seq_first)
            else:
                raw_mod = FunctionalMegatronBlock(base_state, rank, world_size, tp_group, sequence_parallel=use_sp).to(
                    device="cuda", dtype=torch.bfloat16
                )

            if seq_first:
                x_in = x_sbh_global[rank * s_shard : (rank + 1) * s_shard].contiguous().detach().requires_grad_(True)
                dy_in = dy_sbh_global[rank * s_shard : (rank + 1) * s_shard].contiguous().detach()
            elif use_sp:
                x_in = x_global[:, rank * s_shard : (rank + 1) * s_shard].contiguous().detach().requires_grad_(True)
                dy_in = dy_global[:, rank * s_shard : (rank + 1) * s_shard].contiguous().detach()
            else:
                x_in = x_global.clone().detach().requires_grad_(True)
                dy_in = dy_global.clone().detach()

            compile_seconds = 0.0
            op_counts = {
                "symm_fused_ag_mm_calls": 0,
                "symm_fused_mm_rs_calls": 0,
                "c10d_ag_calls": 0,
                "c10d_rs_calls": 0,
                "cat_split_Copy_kernels": 0,
            }
            if do_compile:
                dist.barrier(group=tp_group)
                def _call_template(inp: torch.Tensor) -> torch.Tensor:
                    return raw_mod(inp)

                unique_fn = types.FunctionType(
                    _call_template.__code__.replace(co_name=f"compiled_entry_{impl_name}_{bsz}x{seqlen}"),
                    _call_template.__globals__,
                    name=f"compiled_entry_{impl_name}_{bsz}x{seqlen}",
                    argdefs=_call_template.__defaults__,
                    closure=_call_template.__closure__,
                )
                with fresh_cache(), torch._inductor.config.patch({"_micro_pipeline_tp": async_tp}):
                    t0 = time.perf_counter()
                    exec_mod: Any = torch.compile(unique_fn, fullgraph=True, dynamic=False)
                    raw_mod.zero_grad(set_to_none=True)
                    y_warm = exec_mod(x_in)
                    torch.autograd.backward(y_warm, dy_in)
                    torch.cuda.synchronize()
                    dist.barrier(group=tp_group)
                    compile_seconds = time.perf_counter() - t0
                    del y_warm

                    def _prof_step() -> None:
                        raw_mod.zero_grad(set_to_none=True)
                        if x_in.grad is not None:
                            x_in.grad = None
                        yp = exec_mod(x_in)
                        torch.autograd.backward(yp, dy_in)

                    op_counts = profile_async_tp_op_counts(_prof_step)
                if impl_name in ("dtensor_bsh_sp_compiled", "dtensor_sbh_sp_compiled"):
                    if op_counts["symm_fused_ag_mm_calls"] != 0 or op_counts["symm_fused_mm_rs_calls"] != 0:
                        raise RuntimeError(f"{impl_name} unexpectedly invoked symm_mem ops: {op_counts}")
                if impl_name == "dtensor_sbh_async_tp_compiled":
                    if op_counts["symm_fused_ag_mm_calls"] != 4 or op_counts["symm_fused_mm_rs_calls"] != 2:
                        raise RuntimeError(f"{impl_name} failed to fuse expected 4 AG+MM and 2 MM+RS ops: {op_counts}")
            else:
                exec_mod = raw_mod

            # Correctness + 1.75x chain-rule check + peak incremental step memory
            raw_mod.zero_grad(set_to_none=True)
            if x_in.grad is not None:
                x_in.grad = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            base_mem = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            y_out = exec_mod(x_in)
            torch.autograd.backward(y_out, dy_in)
            torch.cuda.synchronize()
            inc_mib_local = (torch.cuda.max_memory_allocated() - base_mem) / (1024.0 ** 2)
            mem_tensor = torch.tensor([inc_mib_local], dtype=torch.float64, device="cuda")
            dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX, group=tp_group)
            inc_mib = float(mem_tensor.item())

            if seq_first:
                y_shard = y_out.detach().transpose(0, 1).contiguous()
                dx_shard = x_in.grad.detach().transpose(0, 1).contiguous()
            elif use_sp:
                y_shard = y_out.detach().clone()
                dx_shard = x_in.grad.detach().clone()
            else:
                y_shard = y_out[:, rank * s_shard : (rank + 1) * s_shard].detach().clone()
                dx_shard = x_in.grad[:, rank * s_shard : (rank + 1) * s_shard].detach().clone()

            w_grads_1x = {k: v.detach().clone() for k, v in extract_local_weight_grads(raw_mod, is_dtensor).items()}
            out_err = rel_l2_error(y_shard, ref_shards["y_shard"])
            dx_err = rel_l2_error(dx_shard, ref_shards["dx_shard"])
            max_dw_err = max(rel_l2_error(w_grads_1x[k], ref_shards[k]) for k in w_grads_1x)

            # 1.75x upstream scalar loss-scale chain-rule gate
            raw_mod.zero_grad(set_to_none=True)
            x_in.grad = None
            y_scaled = exec_mod(x_in)
            torch.autograd.backward(y_scaled, dy_in * LOSS_SCALE_CHECK)
            torch.cuda.synchronize()
            if seq_first:
                dx_scaled = x_in.grad.detach().transpose(0, 1).contiguous()
            elif use_sp:
                dx_scaled = x_in.grad.detach()
            else:
                dx_scaled = x_in.grad[:, rank * s_shard : (rank + 1) * s_shard].detach()
            chain_err = rel_l2_error(dx_scaled, dx_shard * LOSS_SCALE_CHECK)

            err_tensor = torch.tensor([out_err, dx_err, max_dw_err, chain_err, compile_seconds], dtype=torch.float64, device="cuda")
            dist.all_reduce(err_tensor, op=dist.ReduceOp.MAX, group=tp_group)

            def make_step_runner(m_raw: nn.Module, m_exec: Any, x_t: torch.Tensor, dy_t: torch.Tensor) -> Callable[[], None]:
                def _step() -> None:
                    m_raw.zero_grad(set_to_none=True)
                    if x_t.grad is not None:
                        x_t.grad = None
                    out_t = m_exec(x_t)
                    torch.autograd.backward(out_t, dy_t)
                return _step

            meta = {
                "sequence_layout": "SBH_Shard0" if seq_first else ("BSH_Shard1" if use_sp else "BSH_Replicated"),
                "compile_seconds": round(float(err_tensor[4].item()), 4),
                "incremental_peak_mib": round(inc_mib, 2),
                "output_rel_l2_error": round(float(err_tensor[0].item()), 6),
                "dx_rel_l2_error": round(float(err_tensor[1].item()), 6),
                "max_weight_grad_rel_l2_error": round(float(err_tensor[2].item()), 6),
                "chain_rule_1p75x_rel_error": round(float(err_tensor[3].item()), 6),
                **op_counts,
            }
            built_runners.append((impl_name, make_step_runner(raw_mod, exec_mod, x_in, dy_in), meta))
            del y_out, y_scaled, w_grads_1x

        # Palindromic timing across all 8 modes while keeping compiled graphs warm
        crit_map: dict[str, list[float]] = {name: [] for name, _, _ in built_runners}
        loc_map: dict[str, list[float]] = {name: [] for name, _, _ in built_runners}
        for pass_runners in (built_runners, list(reversed(built_runners))):
            for name, step_fn, _ in pass_runners:
                c_s, l_s = benchmark_distributed_group(
                    step_fn, tp_group, warmup=warmup, groups=half_groups, iterations_per_group=iters
                )
                crit_map[name].extend(c_s)
                loc_map[name].extend(l_s)

        # Post-timing integrity check: verify compiled graphs stayed bound to their exact kernels
        runner_lookup = {name: step_fn for name, step_fn, _ in built_runners}
        post_sp_counts = profile_async_tp_op_counts(runner_lookup["dtensor_sbh_sp_compiled"])
        post_async_counts = profile_async_tp_op_counts(runner_lookup["dtensor_sbh_async_tp_compiled"])
        if post_sp_counts["symm_fused_ag_mm_calls"] != 0 or post_sp_counts["symm_fused_mm_rs_calls"] != 0:
            raise RuntimeError(f"Post-timing dtensor_sbh_sp_compiled contaminated: {post_sp_counts}")
        if post_async_counts["symm_fused_ag_mm_calls"] != 4 or post_async_counts["symm_fused_mm_rs_calls"] != 2:
            raise RuntimeError(f"Post-timing dtensor_sbh_async_tp_compiled lost fusion: {post_async_counts}")

        total_flops = block_training_flops_global(bsz, seqlen)
        tp_eager_ms = statistics.median(crit_map["megatron_tp_func_eager"])
        sbh_sp_comp_ms = statistics.median(crit_map["dtensor_sbh_sp_compiled"])

        for name, _, meta in built_runners:
            stats = summarize_samples(crit_map[name], loc_map[name], warmup, iters)
            med_ms = float(stats["median_ms"])
            med_s = med_ms / 1000.0
            records.append(
                {
                    "tp_size": world_size,
                    "batch_size": bsz,
                    "seq_len": seqlen,
                    "global_tokens": bsz * seqlen,
                    "tokens_per_rank_sp": (bsz * seqlen) // world_size,
                    "implementation": name,
                    **stats,
                    **meta,
                    "tokens_per_s": round((bsz * seqlen) / med_s, 2),
                    "achieved_tflops_per_gpu": round(((total_flops / world_size) / 1e12) / med_s, 4),
                    "speedup_vs_megatron_tp_eager": round(tp_eager_ms / med_ms, 6),
                    "speedup_vs_sbh_sp_compiled": round(sbh_sp_comp_ms / med_ms, 6),
                }
            )

        del built_runners, ref_shards, x_global, dy_global
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    return records


def worker_main(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    telemetry = LocalGpuTelemetry(local_rank)
    pre_sample = telemetry.sample()

    mesh = init_device_mesh("cuda", (world_size,))
    tp_group = mesh.get_group()
    base_state = build_base_state(torch.device("cuda"))

    # Suite 0 MUST run before symm_mem.enable_symm_mem_for_group is invoked
    dep_audit = run_deprecation_gate_audit(base_state, mesh, tp_group, smoke=args.smoke)

    # Suites A, B, and C run with SymmetricMemory enabled on tp_group
    allreduce_records = run_allreduce_suite(tp_group, smoke=args.smoke)
    primitive_records = run_primitive_overlap_suite(tp_group, smoke=args.smoke)
    block_records = run_transformer_block_suite(base_state, mesh, tp_group, smoke=args.smoke)

    post_sample = telemetry.sample()
    gathered_telemetry: list[dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[list-item]
    dist.all_gather_object(
        gathered_telemetry,
        {"rank": rank, "pre": pre_sample, "post": post_sample},
        group=tp_group,
    )

    if rank == 0:
        write_json_lf(
            args.worker_out,
            {
                "tp_size": world_size,
                "telemetry": gathered_telemetry,
                "deprecation_gate_audit": dep_audit,
                "allreduce_records": allreduce_records,
                "primitive_overlap_records": primitive_records,
                "transformer_block_records": block_records,
            },
        )

    dist.barrier(group=tp_group)
    dist.destroy_process_group()


def try_lock_all_gpus(target_mhz: int) -> dict[str, Any]:
    pynvml.nvmlInit()
    count = int(pynvml.nvmlDeviceGetCount())
    locked_by_nvml = False
    for idx in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        try:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_mhz, target_mhz)
            locked_by_nvml = True
        except pynvml.NVMLError:
            pass
    observed = [
        int(pynvml.nvmlDeviceGetClockInfo(pynvml.nvmlDeviceGetHandleByIndex(i), pynvml.NVML_CLOCK_SM))
        for i in range(count)
    ]
    return {
        "gpu_count": count,
        "target_sm_clock_mhz": target_mhz,
        "locked_in_process": locked_by_nvml,
        "initial_sm_clocks_mhz": observed,
    }


def coordinator_main(args: argparse.Namespace) -> None:
    tp_sizes = [2, 8] if args.smoke else [2, 4, 8]
    clock_info = try_lock_all_gpus(args.lock_sm_clock_mhz)

    env = environment_metadata()
    env["hostname"] = "node-0"
    env["nvshmem_available"] = bool(symm_mem.is_nvshmem_available())

    deprecation_audits: list[dict[str, Any]] = []
    allreduce_records: list[dict[str, Any]] = []
    primitive_records: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    rank_telemetry_by_tp: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="e114_tp_shards_") as tmp_dir:
        for tp in tp_sizes:
            shard_path = Path(tmp_dir) / f"tp_{tp}.json"
            port = find_free_port()
            cmd = [
                "torchrun",
                "--standalone",
                f"--master_port={port}",
                f"--nproc_per_node={tp}",
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-out",
                str(shard_path),
            ]
            if args.smoke:
                cmd.append("--smoke")
            subprocess.run(cmd, check=True)
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            rank_telemetry_by_tp[f"tp{tp}"] = shard["telemetry"]
            deprecation_audits.append(shard["deprecation_gate_audit"])
            allreduce_records.extend(shard["allreduce_records"])
            primitive_records.extend(shard["primitive_overlap_records"])
            block_records.extend(shard["transformer_block_records"])

    payload = {
        "experiment_id": "E114",
        "workload": "symmetric_memory_async_tp",
        "smoke": bool(args.smoke),
        "repetition": int(args.repetition),
        "environment": env,
        "clock_lock": clock_info,
        "model_config": {
            "hidden_size": HIDDEN_SIZE,
            "ffn_size": FFN_SIZE,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "loss_scale_check": LOSS_SCALE_CHECK,
        },
        "tp_sizes": tp_sizes,
        "telemetry_by_tp": rank_telemetry_by_tp,
        "deprecation_gate_records": deprecation_audits,
        "allreduce_records": allreduce_records,
        "primitive_overlap_records": primitive_records,
        "transformer_block_records": block_records,
    }
    write_json_lf(args.out, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E114 NVSwitch SymmetricMemory, Sequence Parallelism, and Async-TP benchmark")
    parser.add_argument("--out", type=str, default="results/raw/symmetric_memory_async_tp/smoke_symmetric_memory_async_tp.json")
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--lock-sm-clock-mhz", type=int, default=1410)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-out", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        worker_main(parsed)
    else:
        coordinator_main(parsed)
