from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pynvml
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleZBVZeroBubble,
)
from torch.distributed.pipelining.schedules import generate_stage_to_rank_mapping
from torch.utils.checkpoint import checkpoint

from common import environment_metadata


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


# ---------------------------------------------------------------------------
# Functional Transformer Primitives (Eager + @torch.compile variants)
# ---------------------------------------------------------------------------


def rmsnorm_eager(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd).to(x.dtype) * weight


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).to(torch.bfloat16).view(1, seq_len, 1, head_dim // 2)
    sin = torch.sin(freqs).to(torch.bfloat16).view(1, seq_len, 1, head_dim // 2)
    return cos, sin


def apply_rope_qkv_eager(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len, _ = qkv.shape
    q_dim = n_q * head_dim
    kv_dim = n_kv * head_dim
    q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
    q = q.view(bsz, seq_len, n_q, head_dim)
    k = k.view(bsz, seq_len, n_kv, head_dim)
    v = v.view(bsz, seq_len, n_kv, head_dim)

    q1, q2 = q[..., : head_dim // 2], q[..., head_dim // 2 :]
    k1, k2 = k[..., : head_dim // 2], k[..., head_dim // 2 :]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot, v


def swiglu_eager(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


_rmsnorm_compiled = torch.compile(rmsnorm_eager, dynamic=False)
_apply_rope_qkv_compiled = torch.compile(apply_rope_qkv_eager, dynamic=False)
_swiglu_compiled = torch.compile(swiglu_eager, dynamic=False)


# ---------------------------------------------------------------------------
# Zero-Copy Split dX / dW Linear Autograd Function & Stage Queue
# ---------------------------------------------------------------------------


class SplitDWLinearFn(torch.autograd.Function):
    """Computes only dX = grad_output @ weight during BACKWARD_INPUT (critical path),
    deferring in-place cuBLAS dW accumulation (w.grad.addmm_(dy^T, x)) to BACKWARD_WEIGHT.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        dw_queue: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x.detach(), weight)
        ctx.dw_queue = dw_queue
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None]:
        x, weight = ctx.saved_tensors
        dx = torch.matmul(grad_output, weight) if ctx.needs_input_grad[0] else None
        if weight.requires_grad and ctx.dw_queue is not None:
            ctx.dw_queue.append(
                (
                    x.reshape(-1, x.shape[-1]),
                    grad_output.detach().reshape(-1, grad_output.shape[-1]),
                    weight,
                )
            )
        return dx, None, None


# ---------------------------------------------------------------------------
# Llama-3 GQA Transformer Block & Pipeline Stage Module
# ---------------------------------------------------------------------------


class ModelSpec:
    def __init__(
        self,
        *,
        num_layers: int,
        hidden_size: int,
        ffn_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim

    def flops_breakdown_per_step(self, global_tokens: int, seq_len: int) -> dict[str, float]:
        # Per-layer linear forward FLOPs: qkv + o + gate_up + down
        lin_fwd_layer = 2.0 * global_tokens * (
            self.hidden_size * self.qkv_dim
            + self.hidden_size * self.hidden_size
            + self.hidden_size * (2 * self.ffn_size)
            + self.ffn_size * self.hidden_size
        )
        attn_fwd_layer = 2.0 * global_tokens * (seq_len + 1) * self.num_q_heads * self.head_dim
        attn_bwd_layer = 2.5 * attn_fwd_layer

        fwd_total = self.num_layers * (lin_fwd_layer + attn_fwd_layer)
        bx_total = self.num_layers * (lin_fwd_layer + attn_bwd_layer)
        bw_total = self.num_layers * lin_fwd_layer
        total = fwd_total + bx_total + bw_total
        return {
            "fwd_flops": fwd_total,
            "bx_flops": bx_total,
            "bw_flops": bw_total,
            "total_train_flops": total,
            "attn_share_of_total": (self.num_layers * (attn_fwd_layer + attn_bwd_layer)) / total,
            "bx_to_bw_flop_ratio": bx_total / bw_total,
        }


def init_deterministic_layer_weights(
    layer_idx: int, spec: ModelSpec, device: torch.device
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(20000 + layer_idx * 97)
    scale_h = 1.0 / math.sqrt(spec.hidden_size)
    scale_ffn = 1.0 / math.sqrt(spec.ffn_size)

    norm1 = torch.ones(spec.hidden_size, device=device, dtype=torch.bfloat16)
    wqkv = torch.randn(spec.qkv_dim, spec.hidden_size, generator=gen, device=device, dtype=torch.bfloat16) * scale_h
    wo = torch.randn(spec.hidden_size, spec.hidden_size, generator=gen, device=device, dtype=torch.bfloat16) * scale_h
    norm2 = torch.ones(spec.hidden_size, device=device, dtype=torch.bfloat16)
    wgu = torch.randn(2 * spec.ffn_size, spec.hidden_size, generator=gen, device=device, dtype=torch.bfloat16) * scale_h
    wd = torch.randn(spec.hidden_size, spec.ffn_size, generator=gen, device=device, dtype=torch.bfloat16) * scale_ffn
    return {
        "norm1": norm1,
        "wqkv": wqkv,
        "wo": wo,
        "norm2": norm2,
        "wgu": wgu,
        "wd": wd,
    }


class GqaTransformerLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        spec: ModelSpec,
        device: torch.device,
        *,
        use_split_dw: bool = False,
        use_compiled_ops: bool = True,
        ac_mode: str = "no_ac",
        dw_queue: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.spec = spec
        self.use_split_dw = use_split_dw
        self.use_compiled_ops = use_compiled_ops
        self.ac_mode = ac_mode
        self.dw_queue = dw_queue

        init = init_deterministic_layer_weights(layer_idx, spec, device)
        self.norm1 = nn.Parameter(init["norm1"])
        self.wqkv = nn.Parameter(init["wqkv"])
        self.wo = nn.Parameter(init["wo"])
        self.norm2 = nn.Parameter(init["norm2"])
        self.wgu = nn.Parameter(init["wgu"])
        self.wd = nn.Parameter(init["wd"])

    def _rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.use_compiled_ops:
            return _rmsnorm_compiled(x, w)
        return rmsnorm_eager(x, w)

    def _rope_qkv(
        self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_compiled_ops:
            return _apply_rope_qkv_compiled(
                qkv, cos, sin, self.spec.num_q_heads, self.spec.num_kv_heads, self.spec.head_dim
            )
        return apply_rope_qkv_eager(
            qkv, cos, sin, self.spec.num_q_heads, self.spec.num_kv_heads, self.spec.head_dim
        )

    def _swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self.use_compiled_ops:
            return _swiglu_compiled(gate_up)
        return swiglu_eager(gate_up)

    def _linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.use_split_dw:
            return SplitDWLinearFn.apply(x, w, self.dw_queue)
        return F.linear(x, w)

    def _attn_branch(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n1 = self._rmsnorm(x, self.norm1)
        qkv = self._linear(n1, self.wqkv)
        q, k, v = self._rope_qkv(qkv, cos, sin)
        attn_out = flash_attn_func(q, k, v, causal=True)
        attn_flat = attn_out.reshape(x.shape[0], x.shape[1], self.spec.hidden_size)
        return self._linear(attn_flat, self.wo)

    def _mlp_branch(self, h: torch.Tensor) -> torch.Tensor:
        n2 = self._rmsnorm(h, self.norm2)
        gu = self._linear(n2, self.wgu)
        act = self._swiglu(gu)
        return self._linear(act, self.wd)

    def _full_body(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = x + self._attn_branch(x, cos, sin)
        return h + self._mlp_branch(h)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.ac_mode == "no_ac":
            return self._full_body(x, cos, sin)
        if self.ac_mode == "full_ac":
            return checkpoint(self._full_body, x, cos, sin, use_reentrant=True)
        if self.ac_mode == "sac_mlp_recomp":
            # Save expensive FlashAttention-2 + QKV/O GEMMs; recompute norm2 + 28672-wide SwiGLU MLP!
            n1 = checkpoint(self._rmsnorm, x, self.norm1, use_reentrant=True)
            qkv = self._linear(n1, self.wqkv)
            q, k, v = checkpoint(self._rope_qkv, qkv, cos, sin, use_reentrant=True)
            attn_out = flash_attn_func(q, k, v, causal=True)
            attn_flat = attn_out.reshape(x.shape[0], x.shape[1], self.spec.hidden_size)
            h = x + self._linear(attn_flat, self.wo)
            return h + checkpoint(self._mlp_branch, h, use_reentrant=True)
        if self.ac_mode == "sac_nonlin_only":
            # Equivalent to MUST_SAVE(all mm + FA2): recomputes only RMSNorm, RoPE, and SwiGLU,
            # leaving 28672-wide gate_up GEMM output resident across in-flight microbatches!
            n1 = checkpoint(self._rmsnorm, x, self.norm1, use_reentrant=True)
            qkv = self._linear(n1, self.wqkv)
            q, k, v = checkpoint(self._rope_qkv, qkv, cos, sin, use_reentrant=True)
            attn_out = flash_attn_func(q, k, v, causal=True)
            attn_flat = attn_out.reshape(x.shape[0], x.shape[1], self.spec.hidden_size)
            h = x + self._linear(attn_flat, self.wo)
            n2 = checkpoint(self._rmsnorm, h, self.norm2, use_reentrant=True)
            gu = self._linear(n2, self.wgu)
            act = checkpoint(self._swiglu, gu, use_reentrant=True)
            return h + self._linear(act, self.wd)
        raise ValueError(f"Unknown ac_mode: {self.ac_mode}")


class PipelineStageModule(nn.Module):
    def __init__(
        self,
        layer_indices: list[int],
        spec: ModelSpec,
        seq_len: int,
        device: torch.device,
        *,
        use_split_dw: bool = False,
        use_compiled_ops: bool = True,
        ac_mode: str = "no_ac",
    ) -> None:
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.use_split_dw = use_split_dw
        self.dw_queue: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        cos, sin = build_rope_cache(seq_len, spec.head_dim, device)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.layers = nn.ModuleList(
            [
                GqaTransformerLayer(
                    idx,
                    spec,
                    device,
                    use_split_dw=use_split_dw,
                    use_compiled_ops=use_compiled_ops,
                    ac_mode=ac_mode,
                    dw_queue=self.dw_queue if use_split_dw else None,
                )
                for idx in self.layer_indices
            ]
        )

    def clear_dw_queue(self) -> None:
        self.dw_queue.clear()

    def preallocate_and_zero_grads(self) -> None:
        self.dw_queue.clear()
        for p in self.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            else:
                p.grad.zero_()

    def pop_dw_runner(self) -> Callable[[], None]:
        tasks = list(self.dw_queue)
        self.dw_queue.clear()

        def runner() -> None:
            for x2d, dy2d, w in tasks:
                if w.grad is None:
                    w.grad = torch.matmul(dy2d.t(), x2d)
                else:
                    w.grad.addmm_(dy2d.t(), x2d, beta=1.0, alpha=1.0)

        return runner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure stage-0 input participates in autograd even if P2P/data input has requires_grad=False
        h = x
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        return h


def make_data_tensors(
    global_batch: int,
    seq_len: int,
    hidden_size: int,
    device: torch.device,
    *,
    seed: int = 7777,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn(global_batch, seq_len, hidden_size, generator=gen, device=device, dtype=torch.bfloat16)
    target = torch.randn(global_batch, seq_len, hidden_size, generator=gen, device=device, dtype=torch.bfloat16)
    return x, target


def pipeline_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output.float(), target.float(), reduction="mean")


# ---------------------------------------------------------------------------
# Single-GPU (PP = 1) Exact Microbatch-Accumulated Reference & Timing
# ---------------------------------------------------------------------------


def run_pp1_baseline(
    spec: ModelSpec,
    *,
    n_microbatches: int,
    microbatch_size: int,
    seq_len: int,
    device: torch.device,
    warmup: int,
    groups: int,
) -> tuple[dict[str, Any], float, dict[int, dict[str, torch.Tensor]]]:
    global_batch = n_microbatches * microbatch_size
    x_full, y_full = make_data_tensors(global_batch, seq_len, spec.hidden_size, device)
    x_mbs = list(x_full.chunk(n_microbatches, dim=0))
    y_mbs = list(y_full.chunk(n_microbatches, dim=0))

    mod = PipelineStageModule(
        list(range(spec.num_layers)),
        spec,
        seq_len,
        device,
        use_split_dw=False,
        use_compiled_ops=True,
        ac_mode="no_ac",
    )
    mod.preallocate_and_zero_grads()

    def step_fn() -> float:
        mod.preallocate_and_zero_grads()
        loss_sum = 0.0
        for xm, ym in zip(x_mbs, y_mbs):
            xm_in = xm.detach().requires_grad_(True)
            out = mod(xm_in)
            loss = pipeline_mse_loss(out, ym) / n_microbatches
            loss.backward()
            loss_sum += float(loss.detach().item())
        return loss_sum

    torch.cuda.synchronize()
    ref_loss = step_fn()
    torch.cuda.synchronize()

    cpu_ref_grads: dict[int, dict[str, torch.Tensor]] = {}
    for li, layer in enumerate(mod.layers):
        cpu_ref_grads[li] = {
            name: p.grad.detach().to(torch.bfloat16).cpu()
            for name, p in layer.named_parameters()
            if p.grad is not None
        }

    for _ in range(max(0, warmup - 1)):
        step_fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    base_bytes = torch.cuda.memory_allocated()
    step_fn()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    inc_mib = max(0.0, (peak_bytes - base_bytes) / (1024.0**2))
    total_peak_mib = peak_bytes / (1024.0**2)

    samples_ms: list[float] = []
    for _ in range(groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    med_ms = float(statistics.median(samples_ms))
    global_tokens = global_batch * seq_len
    flops_info = spec.flops_breakdown_per_step(global_tokens, seq_len)
    tflops = (flops_info["total_train_flops"] / (med_ms * 1e-3)) / 1e12

    del mod, x_full, y_full, x_mbs, y_mbs
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "median_ms": round(med_ms, 4),
        "min_ms": round(float(min(samples_ms)), 4),
        "max_ms": round(float(max(samples_ms)), 4),
        "tokens_per_s": round(global_tokens / (med_ms * 1e-3), 2),
        "achieved_tflops": round(tflops, 2),
        "activation_inc_mib": round(inc_mib, 2),
        "total_peak_allocated_mib": round(total_peak_mib, 2),
        "loss": round(ref_loss, 8),
        "flops_breakdown": {k: round(v, 6) if "ratio" in k or "share" in k else v for k, v in flops_info.items()},
    }
    return result, ref_loss, cpu_ref_grads


def broadcast_ref_grads_to_all_ranks(
    spec: ModelSpec,
    rank0_ref_grads: dict[int, dict[str, torch.Tensor]] | None,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    rank = dist.get_rank()
    param_shapes = {
        "norm1": (spec.hidden_size,),
        "wqkv": (spec.qkv_dim, spec.hidden_size),
        "wo": (spec.hidden_size, spec.hidden_size),
        "norm2": (spec.hidden_size,),
        "wgu": (2 * spec.ffn_size, spec.hidden_size),
        "wd": (spec.hidden_size, spec.ffn_size),
    }
    out: dict[int, dict[str, torch.Tensor]] = {}
    for li in range(spec.num_layers):
        out[li] = {}
        for pname, shape in param_shapes.items():
            if rank == 0:
                assert rank0_ref_grads is not None
                gpu_buf = rank0_ref_grads[li][pname].to(device=device, dtype=torch.bfloat16).contiguous()
            else:
                gpu_buf = torch.empty(shape, device=device, dtype=torch.bfloat16)
            dist.broadcast(gpu_buf, src=0)
            out[li][pname] = gpu_buf.cpu()
            del gpu_buf
    torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# Distributed Pipeline Execution & Verification Helper
# ---------------------------------------------------------------------------


def evaluate_pipeline_schedule(
    *,
    schedule_key: str,
    pp_group: dist.ProcessGroup,
    spec: ModelSpec,
    n_microbatches: int,
    microbatch_size: int,
    seq_len: int,
    dw_backend: str,
    ac_mode: str,
    device: torch.device,
    warmup: int,
    groups: int,
    ref_loss: float,
    cpu_ref_grads: dict[int, dict[str, torch.Tensor]] | None,
    check_chain_rule: bool = False,
) -> dict[str, Any]:
    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)
    global_batch = n_microbatches * microbatch_size
    global_tokens = global_batch * seq_len

    is_v2 = schedule_key in ("interleaved_1f1b", "interleaved_zb", "zbv", "dualpipe_v")
    style = "v" if schedule_key in ("zbv", "dualpipe_v") else "loop"
    num_stages = 2 * pp_size if is_v2 else pp_size
    assert spec.num_layers % num_stages == 0 and (spec.num_layers // num_stages) >= 1, (
        f"Invalid stage split: num_layers={spec.num_layers}, num_stages={num_stages}"
    )
    layers_per_stage = spec.num_layers // num_stages

    is_split_schedule = schedule_key in ("zb1p", "interleaved_zb", "zbv", "dualpipe_v")
    if is_split_schedule:
        use_split_dw = dw_backend in ("custom_dw_eager", "custom_dw_compiled_ops")
        use_compiled_ops = dw_backend == "custom_dw_compiled_ops"
        use_module_compile = False
    else:
        use_split_dw = False
        use_compiled_ops = dw_backend == "compiled_ops"
        use_module_compile = dw_backend == "module_compile"

    if is_v2:
        s2r = generate_stage_to_rank_mapping(pp_size, num_stages, style=style)
        my_stage_ids = [s for s, r in s2r.items() if r == pp_rank]
    else:
        s2r = {r: r for r in range(pp_size)}
        my_stage_ids = [pp_rank]

    first_rank = s2r[0]
    last_rank = s2r[num_stages - 1]

    local_stages: list[PipelineStage] = []
    local_mods: dict[int, PipelineStageModule] = {}

    for sid in my_stage_ids:
        layer_ids = list(range(sid * layers_per_stage, (sid + 1) * layers_per_stage))
        raw_mod = PipelineStageModule(
            layer_ids,
            spec,
            seq_len,
            device,
            use_split_dw=use_split_dw,
            use_compiled_ops=use_compiled_ops,
            ac_mode=ac_mode,
        )
        raw_mod.preallocate_and_zero_grads()
        stage_submod: nn.Module = torch.compile(raw_mod, dynamic=False) if use_module_compile else raw_mod
        # Always pass static input_args / output_args to prevent _prepare_backward_infra
        # from polluting custom dw_builder queues or running extra shape-inference passes.
        in_ex = torch.empty(
            microbatch_size,
            seq_len,
            spec.hidden_size,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        out_ex = torch.empty(
            microbatch_size,
            seq_len,
            spec.hidden_size,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        st = PipelineStage(
            stage_submod,
            stage_index=sid,
            num_stages=num_stages,
            device=device,
            input_args=(in_ex,),
            output_args=out_ex,
            group=pp_group,
            dw_builder=raw_mod.pop_dw_runner if use_split_dw else None,
        )
        local_stages.append(st)
        local_mods[sid] = raw_mod

    if schedule_key == "gpipe":
        sched = ScheduleGPipe(local_stages[0], n_microbatches=n_microbatches, loss_fn=pipeline_mse_loss)
    elif schedule_key == "1f1b":
        sched = Schedule1F1B(local_stages[0], n_microbatches=n_microbatches, loss_fn=pipeline_mse_loss)
    elif schedule_key == "zb1p":
        sched = ScheduleInterleavedZeroBubble(local_stages, n_microbatches=n_microbatches, loss_fn=pipeline_mse_loss)
    elif schedule_key == "interleaved_1f1b":
        sched = ScheduleInterleaved1F1B(local_stages, n_microbatches=n_microbatches, loss_fn=pipeline_mse_loss)
    elif schedule_key == "interleaved_zb":
        sched = ScheduleInterleavedZeroBubble(local_stages, n_microbatches=n_microbatches, loss_fn=pipeline_mse_loss)
    elif schedule_key == "zbv":
        sched = ScheduleZBVZeroBubble(local_stages, n_microbatches=n_microbatches, loss_fn=pipeline_mse_loss)
    elif schedule_key == "dualpipe_v":
        sched = ScheduleDualPipeV(local_stages, n_microbatches=n_microbatches, loss_fn=pipeline_mse_loss)
    else:
        raise ValueError(f"Unsupported schedule_key: {schedule_key}")

    x_full, y_full = make_data_tensors(global_batch, seq_len, spec.hidden_size, device)
    arg_x = x_full if pp_rank == first_rank else None
    arg_y = y_full if pp_rank == last_rank else None

    def run_one_step(loss_scale: float = 1.0) -> list[torch.Tensor] | None:
        for mod in local_mods.values():
            mod.preallocate_and_zero_grads()
        losses: list[torch.Tensor] | None = [] if pp_rank == last_rank else None
        x_in = arg_x.detach().requires_grad_(True) if arg_x is not None else None
        if loss_scale != 1.0:
            sched._loss_fn = lambda o, t: LOSS_SCALE_CHECK * pipeline_mse_loss(o, t)
        else:
            sched._loss_fn = pipeline_mse_loss

        if pp_rank == first_rank and pp_rank == last_rank:
            sched.step(x_in, target=arg_y, losses=losses)
        elif pp_rank == first_rank:
            sched.step(x_in)
        elif pp_rank == last_rank:
            sched.step(target=arg_y, losses=losses)
        else:
            sched.step()
        return losses

    # Step 1: Correctness & gradient check on first clean execution
    dist.barrier(group=pp_group)
    torch.cuda.synchronize()
    first_losses = run_one_step(1.0)
    torch.cuda.synchronize()

    loss_tensor = torch.zeros(1, device=device, dtype=torch.float64)
    if pp_rank == last_rank and first_losses:
        loss_tensor.fill_(sum(float(t.detach().item()) for t in first_losses) / len(first_losses))
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM, group=pp_group)
    measured_loss = float(loss_tensor.item())
    rel_loss_err = abs(measured_loss - ref_loss) / max(abs(ref_loss), 1e-8)

    saved_unscaled_grads: dict[tuple[int, str], torch.Tensor] = {}
    if check_chain_rule:
        for sid, mod in local_mods.items():
            for li_local, layer in enumerate(mod.layers):
                global_li = sid * layers_per_stage + li_local
                for pname, p in layer.named_parameters():
                    saved_unscaled_grads[(global_li, pname)] = p.grad.detach().float().clone()

    max_dw_err_local = 0.0
    if cpu_ref_grads is not None:
        for sid, mod in local_mods.items():
            for li_local, layer in enumerate(mod.layers):
                global_li = sid * layers_per_stage + li_local
                for pname, p in layer.named_parameters():
                    err = rel_l2_error(p.grad.detach().float().cpu(), cpu_ref_grads[global_li][pname].float())
                    max_dw_err_local = max(max_dw_err_local, err)

    dw_err_tensor = torch.tensor([max_dw_err_local], device=device, dtype=torch.float64)
    dist.all_reduce(dw_err_tensor, op=dist.ReduceOp.MAX, group=pp_group)
    max_dw_rel_l2 = float(dw_err_tensor.item())

    chain_rule_err = 0.0
    if check_chain_rule:
        run_one_step(LOSS_SCALE_CHECK)
        torch.cuda.synchronize()
        local_cr = 0.0
        for sid, mod in local_mods.items():
            for li_local, layer in enumerate(mod.layers):
                global_li = sid * layers_per_stage + li_local
                for pname, p in layer.named_parameters():
                    expected_scaled = saved_unscaled_grads[(global_li, pname)] * LOSS_SCALE_CHECK
                    local_cr = max(local_cr, rel_l2_error(p.grad.detach().float(), expected_scaled))
        cr_tensor = torch.tensor([local_cr], device=device, dtype=torch.float64)
        dist.all_reduce(cr_tensor, op=dist.ReduceOp.MAX, group=pp_group)
        chain_rule_err = float(cr_tensor.item())
        saved_unscaled_grads.clear()

    # Step 2: Warmup remaining iterations
    for _ in range(max(0, warmup - 1)):
        run_one_step(1.0)
    torch.cuda.synchronize()
    dist.barrier(group=pp_group)

    # Step 3: Per-rank activation & total memory profile on one clean step
    torch.cuda.reset_peak_memory_stats()
    base_bytes = torch.cuda.memory_allocated()
    run_one_step(1.0)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    local_inc_mib = max(0.0, (peak_bytes - base_bytes) / (1024.0**2))
    local_total_mib = peak_bytes / (1024.0**2)
    local_base_mib = base_bytes / (1024.0**2)

    inc_gather = [torch.zeros(1, device=device, dtype=torch.float64) for _ in range(pp_size)]
    tot_gather = [torch.zeros(1, device=device, dtype=torch.float64) for _ in range(pp_size)]
    dist.all_gather(inc_gather, torch.tensor([local_inc_mib], device=device, dtype=torch.float64), group=pp_group)
    dist.all_gather(tot_gather, torch.tensor([local_total_mib], device=device, dtype=torch.float64), group=pp_group)
    per_rank_inc_mib = [round(float(t.item()), 2) for t in inc_gather]
    per_rank_tot_mib = [round(float(t.item()), 2) for t in tot_gather]

    # Step 4: Timed steady-state groups
    crit_samples: list[float] = []
    mean_samples: list[float] = []
    rank_ms_tensor = torch.zeros(1, device=device, dtype=torch.float64)

    for _ in range(groups):
        dist.barrier(group=pp_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_one_step(1.0)
        end.record()
        end.synchronize()
        rank_ms_tensor.fill_(float(start.elapsed_time(end)))
        max_t = rank_ms_tensor.clone()
        sum_t = rank_ms_tensor.clone()
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX, group=pp_group)
        dist.all_reduce(sum_t, op=dist.ReduceOp.SUM, group=pp_group)
        crit_samples.append(float(max_t.item()))
        mean_samples.append(float(sum_t.item()) / pp_size)

    med_ms = float(statistics.median(crit_samples))
    flops_info = spec.flops_breakdown_per_step(global_tokens, seq_len)
    total_tflops = (flops_info["total_train_flops"] / (med_ms * 1e-3)) / 1e12

    # Count P2P ops per step across all stages (forward + backward)
    cross_rank_transitions = 0
    for s in range(num_stages - 1):
        if s2r[s] != s2r[s + 1]:
            cross_rank_transitions += 1
    bytes_per_mb_activation = microbatch_size * seq_len * spec.hidden_size * 2
    total_p2p_mib = (2 * n_microbatches * cross_rank_transitions * bytes_per_mb_activation) / (1024.0**2)

    max_inc = max(per_rank_inc_mib)
    min_inc = max(min(per_rank_inc_mib), 0.01)

    del sched, local_stages, local_mods, x_full, y_full
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "schedule": schedule_key,
        "dw_backend": dw_backend,
        "ac_mode": ac_mode,
        "pp_size": pp_size,
        "num_stages": num_stages,
        "virtual_stages_per_rank": num_stages // pp_size,
        "n_microbatches": n_microbatches,
        "microbatch_size": microbatch_size,
        "seq_len": seq_len,
        "global_tokens": global_tokens,
        "median_ms": round(med_ms, 4),
        "min_ms": round(float(min(crit_samples)), 4),
        "max_ms": round(float(max(crit_samples)), 4),
        "rank_mean_median_ms": round(float(statistics.median(mean_samples)), 4),
        "tokens_per_s": round(global_tokens / (med_ms * 1e-3), 2),
        "achieved_tflops": round(total_tflops, 2),
        "per_GPU_tflops": round(total_tflops / pp_size, 2),
        "static_base_mib_rank0": round(local_base_mib, 2) if pp_rank == 0 else 0.0,
        "per_rank_activation_inc_mib": per_rank_inc_mib,
        "per_rank_total_peak_mib": per_rank_tot_mib,
        "max_rank_activation_inc_mib": round(max_inc, 2),
        "min_rank_activation_inc_mib": round(min(per_rank_inc_mib), 2),
        "rank0_activation_inc_mib": per_rank_inc_mib[0],
        "last_rank_activation_inc_mib": per_rank_inc_mib[-1],
        "rank_mem_imbalance_ratio": round(max_inc / min_inc, 3),
        "max_rank_total_peak_mib": round(max(per_rank_tot_mib), 2),
        "cross_rank_stage_transitions": cross_rank_transitions,
        "total_p2p_payload_mib": round(total_p2p_mib, 2),
        "loss": round(measured_loss, 8),
        "rel_loss_err": round(rel_loss_err, 8),
        "max_dw_rel_l2": round(max_dw_rel_l2, 6),
        "chain_rule_rel_l2": round(chain_rule_err, 6),
    }


# ---------------------------------------------------------------------------
# Suite 0 Programmatic PyTorch 2.13 Audits & Isolated t_F / t_Bx / t_Bw Profile
# ---------------------------------------------------------------------------


def run_pytorch_pipelining_audits(
    pp_group: dist.ProcessGroup,
    spec: ModelSpec,
    device: torch.device,
    *,
    is_smoke: bool,
) -> dict[str, Any]:
    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)

    # Audit 1: Verify ScheduleZBVZeroBubble & ScheduleDualPipeV reject torch.compile(nn.Module)
    # EVEN WHEN a custom dw_builder is supplied.
    s2r_v = generate_stage_to_rank_mapping(pp_size, 2 * pp_size, style="v")
    my_v_sids = [s for s, r in s2r_v.items() if r == pp_rank]
    audit_stages: list[PipelineStage] = []
    for sid in my_v_sids:
        tiny = nn.Linear(64, 64, bias=False, device=device, dtype=torch.bfloat16)
        comp_tiny = torch.compile(tiny, dynamic=False)
        in_ex = torch.empty(1, 32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        st = PipelineStage(
            comp_tiny,
            stage_index=sid,
            num_stages=2 * pp_size,
            device=device,
            input_args=(in_ex,),
            output_args=in_ex,
            group=pp_group,
            dw_builder=lambda: (lambda: None),
        )
        audit_stages.append(st)

    zbv_rejected = False
    zbv_err_msg = ""
    try:
        ScheduleZBVZeroBubble(audit_stages, n_microbatches=2 * pp_size, loss_fn=pipeline_mse_loss)
    except RuntimeError as exc:
        zbv_rejected = True
        zbv_err_msg = str(exc)

    dpv_rejected = False
    try:
        ScheduleDualPipeV(audit_stages, n_microbatches=2 * pp_size, loss_fn=pipeline_mse_loss)
    except RuntimeError:
        dpv_rejected = True

    # Audit 2: Verify omitting input_args/output_args causes _prepare_backward_infra to push
    # phantom shape-inference dW tasks into a custom dw_builder queue on non-first stages (sid > 0),
    # silently corrupting weight gradients on step 1 unless static input_args/output_args are supplied.
    aud_dim = 64
    aud_mb = 2 * pp_size
    gen_aud = torch.Generator(device=device)
    gen_aud.manual_seed(31415)
    aud_weights = [
        torch.randn(aud_dim, aud_dim, generator=gen_aud, device=device, dtype=torch.float32)
        * (1.0 / math.sqrt(aud_dim))
        for _ in range(2 * pp_size)
    ]
    x_aud = torch.randn(aud_mb, 16, aud_dim, generator=gen_aud, device=device, dtype=torch.float32)
    y_aud = torch.randn(aud_mb, 16, aud_dim, generator=gen_aud, device=device, dtype=torch.float32)

    # Exact microbatch-accumulated FP32 single-device reference
    ref_aud_ws = [nn.Parameter(w.clone()) for w in aud_weights]
    for xm_a, ym_a in zip(x_aud.chunk(aud_mb, dim=0), y_aud.chunk(aud_mb, dim=0)):
        h_aud = xm_a.detach().requires_grad_(True)
        for rw in ref_aud_ws:
            h_aud = h_aud + torch.tanh(F.linear(h_aud + 0.5, rw))
        (pipeline_mse_loss(h_aud, ym_a) / aud_mb).backward()
    ref_aud_grads = [rw.grad.detach().float() for rw in ref_aud_ws]

    class _AuditStage(nn.Module):
        def __init__(self, w_init: torch.Tensor) -> None:
            super().__init__()
            self.w = nn.Parameter(w_init.clone())
            self.q: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            self.total_popped = 0

        def pop_dw(self) -> Callable[[], None]:
            items = list(self.q)
            self.q.clear()
            self.total_popped += len(items)

            def _run() -> None:
                for x2d, dy2d, wp in items:
                    if wp.grad is None:
                        wp.grad = torch.matmul(dy2d.t(), x2d)
                    else:
                        wp.grad.addmm_(dy2d.t(), x2d, beta=1.0, alpha=1.0)

            return _run

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + torch.tanh(SplitDWLinearFn.apply(x + 0.5, self.w, self.q))

    def _run_audit_case(pass_static_shapes: bool) -> tuple[int, float]:
        mods: dict[int, _AuditStage] = {}
        st_list: list[PipelineStage] = []
        for sid in my_v_sids:
            m = _AuditStage(aud_weights[sid])
            mods[sid] = m
            if pass_static_shapes:
                ex = torch.empty(1, 16, aud_dim, device=device, dtype=torch.float32, requires_grad=True)
                st_list.append(
                    PipelineStage(
                        m,
                        stage_index=sid,
                        num_stages=2 * pp_size,
                        device=device,
                        input_args=(ex,),
                        output_args=ex,
                        group=pp_group,
                        dw_builder=m.pop_dw,
                    )
                )
            else:
                st_list.append(
                    PipelineStage(
                        m,
                        stage_index=sid,
                        num_stages=2 * pp_size,
                        device=device,
                        group=pp_group,
                        dw_builder=m.pop_dw,
                    )
                )
        sc = ScheduleZBVZeroBubble(st_list, n_microbatches=aud_mb, loss_fn=pipeline_mse_loss)
        if pp_rank == 0:
            sc.step(x_aud.clone(), target=y_aud.clone(), losses=[])
        else:
            sc.step()
        extra_tasks = max(m.total_popped - aud_mb for m in mods.values())
        local_err = max(rel_l2_error(m.w.grad.detach().float(), ref_aud_grads[sid]) for sid, m in mods.items())
        t_extra = torch.tensor([extra_tasks], device=device, dtype=torch.int64)
        t_err = torch.tensor([local_err], device=device, dtype=torch.float64)
        dist.all_reduce(t_extra, op=dist.ReduceOp.MAX, group=pp_group)
        dist.all_reduce(t_err, op=dist.ReduceOp.MAX, group=pp_group)
        return int(t_extra.item()), round(float(t_err.item()), 6)

    phantom_extra_tasks, naive_shape_inf_dw_err = _run_audit_case(pass_static_shapes=False)
    static_extra_tasks, static_shape_inf_dw_err = _run_audit_case(pass_static_shapes=True)

    # Audit 3: DualPipeV microbatch count constraint (M < 2 * PP raises ValueError)
    dpv_min_mb_enforced = False
    try:
        valid_stages = []
        for sid in my_v_sids:
            m = nn.Linear(64, 64, bias=False, device=device, dtype=torch.bfloat16)
            ex = torch.empty(1, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
            valid_stages.append(
                PipelineStage(
                    m,
                    stage_index=sid,
                    num_stages=2 * pp_size,
                    device=device,
                    input_args=(ex,),
                    output_args=ex,
                    group=pp_group,
                )
            )
        ScheduleDualPipeV(valid_stages, n_microbatches=pp_size, loss_fn=pipeline_mse_loss)
    except ValueError:
        dpv_min_mb_enforced = True

    # Audit 4: Isolated single-stage (2 layers on full, 1 layer on smoke) t_F / t_Bx / t_Bw profile on rank 0
    seq_grid = [512, 1024] if is_smoke else [1024, 2048, 4096, 8192]
    stage_layers = spec.num_layers // pp_size
    isolated_profiles: dict[str, Any] = {}

    if pp_rank == 0:
        reps = 3 if is_smoke else 8
        warm = 2 if is_smoke else 3
        for slen in seq_grid:
            smod = PipelineStageModule(
                list(range(stage_layers)),
                spec,
                slen,
                device,
                use_split_dw=True,
                use_compiled_ops=True,
                ac_mode="no_ac",
            )
            smod.preallocate_and_zero_grads()
            fmod = PipelineStageModule(
                list(range(stage_layers)),
                spec,
                slen,
                device,
                use_split_dw=False,
                use_compiled_ops=True,
                ac_mode="no_ac",
            )
            fmod.preallocate_and_zero_grads()

            x_mb = torch.randn(1, slen, spec.hidden_size, device=device, dtype=torch.bfloat16)
            dy_mb = torch.randn(1, slen, spec.hidden_size, device=device, dtype=torch.bfloat16)

            for _ in range(warm):
                smod.preallocate_and_zero_grads()
                xi = x_mb.detach().requires_grad_(True)
                out = smod(xi)
                out.backward(dy_mb)
                smod.pop_dw_runner()()

                fmod.preallocate_and_zero_grads()
                xi2 = x_mb.detach().requires_grad_(True)
                out2 = fmod(xi2)
                out2.backward(dy_mb)
            torch.cuda.synchronize()

            tf_list, tbx_list, tbw_list, tbfull_list = [], [], [], []
            for _ in range(reps):
                smod.preallocate_and_zero_grads()
                xi = x_mb.detach().requires_grad_(True)
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                e3 = torch.cuda.Event(enable_timing=True)

                e0.record()
                out = smod(xi)
                e1.record()
                out.backward(dy_mb)
                dw_run = smod.pop_dw_runner()
                e2.record()
                dw_run()
                e3.record()
                e3.synchronize()

                tf_list.append(float(e0.elapsed_time(e1)))
                tbx_list.append(float(e1.elapsed_time(e2)))
                tbw_list.append(float(e2.elapsed_time(e3)))

                fmod.preallocate_and_zero_grads()
                xi2 = x_mb.detach().requires_grad_(True)
                ef0 = torch.cuda.Event(enable_timing=True)
                ef1 = torch.cuda.Event(enable_timing=True)
                out2 = fmod(xi2)
                torch.cuda.synchronize()
                ef0.record()
                out2.backward(dy_mb)
                ef1.record()
                ef1.synchronize()
                tbfull_list.append(float(ef0.elapsed_time(ef1)))

            tf_m = float(statistics.median(tf_list))
            tbx_m = float(statistics.median(tbx_list))
            tbw_m = float(statistics.median(tbw_list))
            tbf_m = float(statistics.median(tbfull_list))
            isolated_profiles[str(slen)] = {
                "stage_layers": stage_layers,
                "t_F_ms": round(tf_m, 4),
                "t_Bx_ms": round(tbx_m, 4),
                "t_Bw_ms": round(tbw_m, 4),
                "t_Bx_plus_Bw_ms": round(tbx_m + tbw_m, 4),
                "t_B_full_ms": round(tbf_m, 4),
                "bx_to_f_ratio": round(tbx_m / max(tf_m, 1e-6), 4),
                "bw_to_f_ratio": round(tbw_m / max(tf_m, 1e-6), 4),
                "bx_to_bw_time_ratio": round(tbx_m / max(tbw_m, 1e-6), 4),
            }
            del smod, fmod, x_mb, dy_mb
            gc.collect()
            torch.cuda.empty_cache()

    dist.barrier(group=pp_group)
    return {
        "optimized_module_rejected_in_zbv": zbv_rejected,
        "optimized_module_rejected_in_dualpipe_v": dpv_rejected,
        "optimized_module_error_excerpt": zbv_err_msg[:180],
        "dynamic_shape_inference_phantom_dw_entries": phantom_extra_tasks,
        "dynamic_shape_inference_dw_rel_l2": naive_shape_inf_dw_err,
        "static_shape_args_extra_dw_entries": static_extra_tasks,
        "static_shape_args_dw_rel_l2": static_shape_inf_dw_err,
        "dualpipe_v_requires_m_ge_2pp": dpv_min_mb_enforced,
        "isolated_stage_timings": isolated_profiles,
    }


# ---------------------------------------------------------------------------
# Distributed Worker Entry Point
# ---------------------------------------------------------------------------


def worker_main(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Important: omit device_id=... here so ProcessGroupNCCL uses lazy per-peer P2P
    # communicators rather than eager-init collective-stream P2P serialization.
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"Expected 8 GPUs, got {world_size}"

    is_smoke = args.mode == "smoke"
    spec = (
        ModelSpec(
            num_layers=16,
            hidden_size=512,
            ffn_size=1408,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=128,
        )
        if is_smoke
        else ModelSpec(
            num_layers=16,
            hidden_size=4096,
            ffn_size=14336,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )
    )

    warmup = 1 if is_smoke else 2
    groups = 2 if is_smoke else 5

    telemetry = LocalGpuTelemetry(local_rank)
    pre_telem = telemetry.sample()
    pre_gather = [None] * world_size
    dist.all_gather_object(pre_gather, pre_telem)

    pp8_group = dist.new_group(list(range(8)))
    pp4_active_group = dist.new_group([0, 1, 2, 3])
    pp4_idle_group = dist.new_group([4, 5, 6, 7])

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Suite 0: PyTorch 2.13 Audit + dX/dW Backend Comparison (PP = 8)
    # ------------------------------------------------------------------
    audits = run_pytorch_pipelining_audits(pp8_group, spec, device, is_smoke=is_smoke)
    s0_seq = 512 if is_smoke else 2048
    s0_mb = 16
    if rank == 0:
        s0_pp1, s0_ref_loss, s0_ref_grads_r0 = run_pp1_baseline(
            spec,
            n_microbatches=s0_mb,
            microbatch_size=1,
            seq_len=s0_seq,
            device=device,
            warmup=warmup,
            groups=groups,
        )
    else:
        s0_pp1, s0_ref_loss, s0_ref_grads_r0 = {}, 0.0, None
    loss_bcast = torch.tensor([s0_ref_loss], device=device, dtype=torch.float64)
    dist.broadcast(loss_bcast, src=0)
    s0_ref_loss = float(loss_bcast.item())
    s0_ref_grads = broadcast_ref_grads_to_all_ranks(spec, s0_ref_grads_r0, device)
    del s0_ref_grads_r0

    s0_backends: dict[str, Any] = {}
    s0_matrix = [
        ("1f1b_eager", "1f1b", "eager"),
        ("1f1b_module_compile", "1f1b", "module_compile"),
        ("1f1b_compiled_ops", "1f1b", "compiled_ops"),
        ("zb1p_autograd_split", "zb1p", "autograd_split"),
        ("zb1p_custom_dw_eager", "zb1p", "custom_dw_eager"),
        ("zb1p_custom_dw_compiled_ops", "zb1p", "custom_dw_compiled_ops"),
        ("zbv_autograd_split", "zbv", "autograd_split"),
        ("zbv_custom_dw_eager", "zbv", "custom_dw_eager"),
        ("zbv_custom_dw_compiled_ops", "zbv", "custom_dw_compiled_ops"),
        ("dualpipe_v_autograd_split", "dualpipe_v", "autograd_split"),
        ("dualpipe_v_custom_dw_eager", "dualpipe_v", "custom_dw_eager"),
        ("dualpipe_v_custom_dw_compiled_ops", "dualpipe_v", "custom_dw_compiled_ops"),
    ]
    for label, skey, bkey in s0_matrix:
        res = evaluate_pipeline_schedule(
            schedule_key=skey,
            pp_group=pp8_group,
            spec=spec,
            n_microbatches=s0_mb,
            microbatch_size=1,
            seq_len=s0_seq,
            dw_backend=bkey,
            ac_mode="no_ac",
            device=device,
            warmup=warmup,
            groups=groups,
            ref_loss=s0_ref_loss,
            cpu_ref_grads=s0_ref_grads,
            check_chain_rule=bkey in ("compiled_ops", "custom_dw_compiled_ops"),
        )
        s0_backends[label] = res
    if rank == 0:
        print(f"[E116] Suite 0 complete at +{time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Suite A: Microbatch Bubble & Rank-Memory Sweep (PP = 1, 4, 8)
    # ------------------------------------------------------------------
    sa_seq = 512 if is_smoke else 2048
    sa_mb_grid = [8, 16] if is_smoke else [8, 16, 32]
    suite_a: dict[str, Any] = {"pp1": {}, "pp4": {}, "pp8": {}}
    cached_sc_pp1: dict[str, Any] | None = None
    cached_sc_loss: float = 0.0
    cached_sc_grads: dict[int, dict[str, torch.Tensor]] | None = None

    for mb in sa_mb_grid:
        if mb == s0_mb and sa_seq == s0_seq:
            pp1_info = s0_pp1
            mb_ref_loss = s0_ref_loss
            mb_ref_grads = s0_ref_grads
        else:
            if rank == 0:
                pp1_info, mb_ref_loss, mb_ref_grads_r0 = run_pp1_baseline(
                    spec,
                    n_microbatches=mb,
                    microbatch_size=1,
                    seq_len=sa_seq,
                    device=device,
                    warmup=warmup,
                    groups=groups,
                )
            else:
                pp1_info, mb_ref_loss, mb_ref_grads_r0 = {}, 0.0, None
            lb = torch.tensor([mb_ref_loss], device=device, dtype=torch.float64)
            dist.broadcast(lb, src=0)
            mb_ref_loss = float(lb.item())
            mb_ref_grads = broadcast_ref_grads_to_all_ranks(spec, mb_ref_grads_r0, device)
            del mb_ref_grads_r0

        if rank == 0:
            suite_a["pp1"][str(mb)] = pp1_info
        pp1_ms_t = torch.tensor([pp1_info.get("median_ms", 0.0) if rank == 0 else 0.0], device=device, dtype=torch.float64)
        dist.broadcast(pp1_ms_t, src=0)
        pp1_ms = float(pp1_ms_t.item())

        # PP = 4 on ranks 0..3
        pp4_modes: dict[str, Any] = {}
        if rank < 4:
            for skey in ["gpipe", "1f1b", "zb1p", "interleaved_1f1b", "zbv", "dualpipe_v"]:
                bkey = "custom_dw_compiled_ops" if skey in ("zb1p", "zbv", "dualpipe_v") else "compiled_ops"
                out4 = evaluate_pipeline_schedule(
                    schedule_key=skey,
                    pp_group=pp4_active_group,
                    spec=spec,
                    n_microbatches=mb,
                    microbatch_size=1,
                    seq_len=sa_seq,
                    dw_backend=bkey,
                    ac_mode="no_ac",
                    device=device,
                    warmup=warmup,
                    groups=groups,
                    ref_loss=mb_ref_loss,
                    cpu_ref_grads=mb_ref_grads,
                    check_chain_rule=(mb == sa_mb_grid[0]),
                )
                out4["speedup_vs_pp1"] = round(pp1_ms / out4["median_ms"], 4)
                out4["strong_scaling_efficiency"] = round((pp1_ms / out4["median_ms"]) / 4.0, 4)
                out4["measured_bubble_fraction"] = round(1.0 - (pp1_ms / 4.0) / out4["median_ms"], 4)
                out4["theoretical_1f1b_bubble_fraction"] = round(3.0 / (mb + 3.0), 4)
                pp4_modes[skey] = out4
        dist.barrier()
        if rank == 0:
            suite_a["pp4"][str(mb)] = {"modes": pp4_modes}

        # PP = 8 on all 8 ranks (note DualPipeV requires M >= 2 * PP = 16)
        pp8_modes: dict[str, Any] = {}
        pp8_scheds = ["gpipe", "1f1b", "zb1p", "interleaved_1f1b", "zbv"]
        if mb >= 16:
            pp8_scheds.append("dualpipe_v")
        for skey in pp8_scheds:
            bkey = "custom_dw_compiled_ops" if skey in ("zb1p", "zbv", "dualpipe_v") else "compiled_ops"
            out8 = evaluate_pipeline_schedule(
                schedule_key=skey,
                pp_group=pp8_group,
                spec=spec,
                n_microbatches=mb,
                microbatch_size=1,
                seq_len=sa_seq,
                dw_backend=bkey,
                ac_mode="no_ac",
                device=device,
                warmup=warmup,
                groups=groups,
                ref_loss=mb_ref_loss,
                cpu_ref_grads=mb_ref_grads,
                check_chain_rule=(mb == sa_mb_grid[0]),
            )
            out8["speedup_vs_pp1"] = round(pp1_ms / out8["median_ms"], 4)
            out8["strong_scaling_efficiency"] = round((pp1_ms / out8["median_ms"]) / 8.0, 4)
            out8["measured_bubble_fraction"] = round(1.0 - (pp1_ms / 8.0) / out8["median_ms"], 4)
            out8["theoretical_1f1b_bubble_fraction"] = round(7.0 / (mb + 7.0), 4)
            pp8_modes[skey] = out8
        if rank == 0:
            suite_a["pp8"][str(mb)] = {"modes": pp8_modes}
            print(f"[E116] Suite A mb={mb} complete at +{time.perf_counter() - t0:.1f}s", flush=True)

        sc_target_mb = 16 if is_smoke else 32
        if mb == sc_target_mb and sa_seq == (512 if is_smoke else 2048):
            cached_sc_pp1 = pp1_info
            cached_sc_loss = mb_ref_loss
            cached_sc_grads = mb_ref_grads
        elif mb != s0_mb:
            mb_ref_grads = None
            gc.collect()

    # Keep s0_ref_grads only until Suite B consumes slen == s0_seq
    gc.collect()

    # ------------------------------------------------------------------
    # Suite B: Sequence-Length t_F : t_Bx : t_Bw Skew Frontier (PP = 8, M = 16)
    # ------------------------------------------------------------------
    sb_mb = 16
    sb_seq_grid = [512, 1024] if is_smoke else [1024, 2048, 4096, 8192]
    suite_b: dict[str, Any] = {}

    for slen in sb_seq_grid:
        if slen == s0_seq and sb_mb == s0_mb and s0_ref_grads is not None:
            sb_pp1 = s0_pp1
            sb_loss = s0_ref_loss
            sb_grads = s0_ref_grads
        else:
            pp1_groups = groups if slen <= 2048 else max(2, groups - 2)
            pp1_warm = warmup if slen <= 2048 else 1
            if rank == 0:
                sb_pp1, sb_loss, sb_grads_r0 = run_pp1_baseline(
                    spec,
                    n_microbatches=sb_mb,
                    microbatch_size=1,
                    seq_len=slen,
                    device=device,
                    warmup=pp1_warm,
                    groups=pp1_groups,
                )
            else:
                sb_pp1, sb_loss, sb_grads_r0 = {}, 0.0, None
            lb = torch.tensor([sb_loss], device=device, dtype=torch.float64)
            dist.broadcast(lb, src=0)
            sb_loss = float(lb.item())
            sb_grads = broadcast_ref_grads_to_all_ranks(spec, sb_grads_r0, device)
            del sb_grads_r0

        pp1_ms_t = torch.tensor([sb_pp1.get("median_ms", 0.0) if rank == 0 else 0.0], device=device, dtype=torch.float64)
        dist.broadcast(pp1_ms_t, src=0)
        sb_pp1_ms = float(pp1_ms_t.item())

        sb_modes: dict[str, Any] = {}
        for skey in ["1f1b", "interleaved_1f1b", "zb1p", "zbv", "dualpipe_v"]:
            bkey = "custom_dw_compiled_ops" if skey in ("zb1p", "zbv", "dualpipe_v") else "compiled_ops"
            out_b = evaluate_pipeline_schedule(
                schedule_key=skey,
                pp_group=pp8_group,
                spec=spec,
                n_microbatches=sb_mb,
                microbatch_size=1,
                seq_len=slen,
                dw_backend=bkey,
                ac_mode="no_ac",
                device=device,
                warmup=warmup,
                groups=groups if slen <= 4096 else max(2, groups - 2),
                ref_loss=sb_loss,
                cpu_ref_grads=sb_grads,
                check_chain_rule=(slen == sb_seq_grid[-1]),
            )
            out_b["speedup_vs_pp1"] = round(sb_pp1_ms / out_b["median_ms"], 4)
            out_b["strong_scaling_efficiency"] = round((sb_pp1_ms / out_b["median_ms"]) / 8.0, 4)
            out_b["measured_bubble_fraction"] = round(1.0 - (sb_pp1_ms / 8.0) / out_b["median_ms"], 4)
            sb_modes[skey] = out_b

        if rank == 0:
            suite_b[str(slen)] = {
                "pp1": sb_pp1,
                "modes": sb_modes,
            }
            print(f"[E116] Suite B S={slen} complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        if slen == s0_seq and sb_grads is not cached_sc_grads:
            s0_ref_grads = None
        if sb_grads is not cached_sc_grads:
            sb_grads = None
        gc.collect()

    # ------------------------------------------------------------------
    # Suite C: Activation Checkpointing & SAC Frontier (PP = 8, M = 32)
    # ------------------------------------------------------------------
    sc_mb = 16 if is_smoke else 32
    sc_seq = 512 if is_smoke else 2048
    if cached_sc_grads is not None and cached_sc_pp1 is not None:
        sc_pp1 = cached_sc_pp1
        sc_loss = cached_sc_loss
        sc_grads = cached_sc_grads
        cached_sc_grads = None
    else:
        if rank == 0:
            sc_pp1, sc_loss, sc_grads_r0 = run_pp1_baseline(
                spec,
                n_microbatches=sc_mb,
                microbatch_size=1,
                seq_len=sc_seq,
                device=device,
                warmup=warmup,
                groups=groups,
            )
        else:
            sc_pp1, sc_loss, sc_grads_r0 = {}, 0.0, None
        lb = torch.tensor([sc_loss], device=device, dtype=torch.float64)
        dist.broadcast(lb, src=0)
        sc_loss = float(lb.item())
        sc_grads = broadcast_ref_grads_to_all_ranks(spec, sc_grads_r0, device)
        del sc_grads_r0

    suite_c: dict[str, Any] = {}
    for skey in ["gpipe", "1f1b", "zbv", "dualpipe_v"]:
        suite_c[skey] = {}
        bkey = "custom_dw_compiled_ops" if skey in ("zbv", "dualpipe_v") else "compiled_ops"
        for ac in ["no_ac", "sac_nonlin_only", "sac_mlp_recomp", "full_ac"]:
            out_c = evaluate_pipeline_schedule(
                schedule_key=skey,
                pp_group=pp8_group,
                spec=spec,
                n_microbatches=sc_mb,
                microbatch_size=1,
                seq_len=sc_seq,
                dw_backend=bkey,
                ac_mode=ac,
                device=device,
                warmup=warmup,
                groups=groups,
                ref_loss=sc_loss,
                cpu_ref_grads=sc_grads,
                check_chain_rule=(ac == "sac_mlp_recomp"),
            )
            suite_c[skey][ac] = out_c

    sc_grads = None
    gc.collect()

    post_telem = telemetry.sample()
    post_gather = [None] * world_size
    dist.all_gather_object(post_gather, post_telem)
    wall_s = round(time.perf_counter() - t0, 2)

    if rank == 0:
        env = environment_metadata()
        payload = {
            "experiment": "E116_pipeline_parallel_zero_bubble_sac",
            "mode": args.mode,
            "wall_seconds": wall_s,
            "environment": env,
            "model_spec": {
                "num_layers": spec.num_layers,
                "hidden_size": spec.hidden_size,
                "ffn_size": spec.ffn_size,
                "num_q_heads": spec.num_q_heads,
                "num_kv_heads": spec.num_kv_heads,
                "head_dim": spec.head_dim,
                "dtype": "bfloat16",
            },
            "telemetry": {
                "pre": pre_gather,
                "post": post_gather,
            },
            "suite_0_audit_and_dw_backends": {
                "audits": audits,
                "seq_len": s0_seq,
                "n_microbatches": s0_mb,
                "pp1_reference": s0_pp1,
                "backends": s0_backends,
            },
            "suite_a_bubble_and_memory_sweep": suite_a,
            "suite_b_seq_skew_frontier": suite_b,
            "suite_c_selective_checkpointing": {
                "n_microbatches": sc_mb,
                "seq_len": sc_seq,
                "pp1_reference": sc_pp1,
                "schedules": suite_c,
            },
        }
        write_json_lf(args.output, payload)
        print(f"Wrote {args.output} in {wall_s:.2f}s")

    dist.barrier()
    dist.destroy_process_group()


def orchestrate(args: argparse.Namespace) -> None:
    port = find_free_port()
    cache_dir = Path("/scratch/cuda_kernel_optimization_20261110_c7e4a29d/inductor_e116")
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    env["OMP_NUM_THREADS"] = "4"

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        f"--master_port={port}",
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        args.mode,
        "--output",
        str(Path(args.output).resolve()),
    ]
    subprocess.run(cmd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E116: Pipeline Parallelism, Zero-Bubble & DualPipeV dX/dW Splitting, and Selective Activation Checkpointing"
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        worker_main(parsed)
    else:
        orchestrate(parsed)
