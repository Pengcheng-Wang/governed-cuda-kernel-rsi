from __future__ import annotations

import argparse
import atexit
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import pynvml
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import checkpoint
import triton
import triton.language as tl

from common import environment_metadata, write_result


class GpuTelemetry:
    def __init__(self) -> None:
        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        nvml_index = int(visible) if visible.isdigit() else 0
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        self.nvml_index = nvml_index

    def max_clocks(self) -> dict[str, int]:
        return {
            "default_app_sm_clock_mhz": int(
                pynvml.nvmlDeviceGetDefaultApplicationsClock(
                    self.handle,
                    pynvml.NVML_CLOCK_SM,
                )
            ),
            "max_sm_clock_mhz": int(
                pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            ),
            "max_mem_clock_mhz": int(
                pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            ),
        }

    def lock_sm_clock(self, mhz: int) -> None:
        torch.cuda.synchronize()
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, int(mhz), int(mhz))
        time.sleep(0.05)

    def reset_clocks(self) -> None:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)
        except Exception:
            pass

    def sample(self) -> dict[str, float | int]:
        return {
            "sm_clock_mhz": int(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            ),
            "mem_clock_mhz": int(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            ),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0, 2),
            "temp_c": int(
                pynvml.nvmlDeviceGetTemperature(
                    self.handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            ),
            "pstate": int(pynvml.nvmlDeviceGetPerformanceState(self.handle)),
        }


@triton.jit
def fused_cross_entropy_fwd_bwd_inplace_kernel(
    logits_ptr,
    stride_logits_row,
    target_ptr,
    loss_ptr,
    inv_n_ptr,
    vocab_size,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    y = tl.load(target_ptr + row)
    row_ptr = logits_ptr + row * stride_logits_row
    if y == ignore_index:
        tl.store(loss_ptr + row, 0.0)
        for offset in range(0, vocab_size, BLOCK_V):
            cols = offset + tl.arange(0, BLOCK_V)
            tl.store(
                row_ptr + cols,
                tl.zeros([BLOCK_V], dtype=tl.bfloat16),
                mask=cols < vocab_size,
            )
        return

    inv_n = tl.load(inv_n_ptr)
    m = -float("inf")
    d = 0.0
    for offset in range(0, vocab_size, BLOCK_V):
        cols = offset + tl.arange(0, BLOCK_V)
        z = tl.load(
            row_ptr + cols,
            mask=cols < vocab_size,
            other=-float("inf"),
            cache_modifier=".ca",
        ).to(tl.float32)
        b_max = tl.max(z, axis=0)
        m_new = tl.maximum(m, b_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(z - m_new), axis=0)
        m = m_new

    lse = m + tl.log(d)
    z_y = tl.load(row_ptr + y).to(tl.float32)
    tl.store(loss_ptr + row, (lse - z_y) * inv_n)

    for offset in range(0, vocab_size, BLOCK_V):
        cols = offset + tl.arange(0, BLOCK_V)
        mask = cols < vocab_size
        z = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        prob = tl.exp(z - lse)
        grad = tl.where(cols == y, prob - 1.0, prob) * inv_n
        tl.store(row_ptr + cols, grad.to(tl.bfloat16), mask=mask)


def select_ce_kernel_meta(num_rows: int, vocab_size: int) -> tuple[int, int]:
    if vocab_size < 32768:
        return 2048, 8
    if num_rows <= 256:
        return 8192, 16
    return 16384, 32


class FusedLinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        chunk_size: int,
        fp32_dw_accum: bool,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        num_tokens, hidden = x.shape
        vocab_size, weight_hidden = weight.shape
        if hidden != weight_hidden:
            raise ValueError("x and weight hidden dimensions must match")

        inv_n = 1.0 / (target != ignore_index).sum().clamp_min(1).to(torch.float32)
        losses = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)
        dx = torch.empty_like(x)
        dw_acc = torch.zeros(
            (vocab_size, hidden),
            device=weight.device,
            dtype=torch.float32 if fp32_dw_accum else weight.dtype,
        )

        # Reuse one [min(chunk_size, num_tokens), vocab_size] buffer across all chunks
        tile_rows = min(chunk_size, num_tokens)
        logits_buf = torch.empty(
            (tile_rows, vocab_size),
            device=x.device,
            dtype=x.dtype,
        )
        w_t = weight.t()

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            rows = end - start
            x_c = x[start:end]
            y_c = target[start:end]
            logits_c = logits_buf[:rows]

            torch.mm(x_c, w_t, out=logits_c)
            block_v, num_warps = select_ce_kernel_meta(rows, vocab_size)
            fused_cross_entropy_fwd_bwd_inplace_kernel[(rows,)](
                logits_c,
                logits_c.stride(0),
                y_c,
                losses[start:end],
                inv_n,
                vocab_size,
                ignore_index,
                BLOCK_V=block_v,
                num_warps=num_warps,
            )
            torch.mm(logits_c, weight, out=dx[start:end])
            if fp32_dw_accum:
                torch.addmm(
                    dw_acc,
                    logits_c.t(),
                    x_c,
                    out_dtype=torch.float32,
                    out=dw_acc,
                )
            else:
                dw_acc.addmm_(logits_c.t(), x_c)

        del logits_buf
        dw_out = dw_acc.to(weight.dtype) if fp32_dw_accum else dw_acc
        del dw_acc
        ctx.save_for_backward(dx, dw_out)
        return losses.sum()

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
        dx, dw = ctx.saved_tensors
        scale_x = grad_output.to(dx.dtype)
        scale_w = grad_output.to(dw.dtype)
        dx.mul_(scale_x)
        dw.mul_(scale_w)
        return dx, dw, None, None, None, None


def triton_fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
    fp32_dw_accum: bool,
    ignore_index: int = -100,
) -> torch.Tensor:
    return FusedLinearCrossEntropyFn.apply(
        x,
        weight,
        target,
        chunk_size,
        fp32_dw_accum,
        ignore_index,
    )


def eager_fp32_logits_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(F.linear(x, weight).float(), target, ignore_index=-100)


def eager_bf16_logits_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(F.linear(x, weight), target, ignore_index=-100)


def _chunk_ce_sum_fn(
    x_chunk: torch.Tensor,
    weight: torch.Tensor,
    target_chunk: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        F.linear(x_chunk, weight).float(),
        target_chunk,
        ignore_index=-100,
        reduction="sum",
    )


def make_checkpointed_chunked_ce(
    compiled_chunk_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    chunk_size: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def run_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        valid = (target != -100).sum().clamp_min(1).to(torch.float32)
        total = torch.zeros((), device=x.device, dtype=torch.float32)
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            total = total + checkpoint(
                compiled_chunk_fn,
                x[start:end],
                weight,
                target[start:end],
                use_reentrant=False,
            )
        return total / valid

    return run_fn


def run_step(
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.nn.Parameter,
    weight: torch.nn.Parameter,
    target: torch.Tensor,
) -> torch.Tensor:
    x.grad = None
    weight.grad = None
    loss = loss_fn(x, weight, target)
    loss.backward()
    return loss


def measure_step_memory(
    step_fn: Callable[[], Any],
    x: torch.nn.Parameter,
    weight: torch.nn.Parameter,
) -> dict[str, int]:
    x.grad = None
    weight.grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    pre_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step_fn()
    torch.cuda.synchronize()
    post_bytes = torch.cuda.memory_allocated()
    peak_bytes = torch.cuda.max_memory_allocated()
    return {
        "peak_incremental_memory_bytes": max(0, peak_bytes - pre_bytes),
        "persistent_grad_bytes": max(0, post_bytes - pre_bytes),
        "peak_transient_workspace_bytes": max(0, peak_bytes - post_bytes),
    }


def benchmark_with_telemetry(
    fn: Callable[[], Any],
    telemetry: GpuTelemetry,
    *,
    warmup: int,
    groups: int,
    iterations_per_group: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    pre_warmup = telemetry.sample()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    post_warmup = telemetry.sample()

    samples_ms: list[float] = []
    clocks_before: list[int] = []
    clocks_after: list[int] = []
    powers_w: list[float] = []
    temps_c: list[int] = []

    for _ in range(groups):
        before = telemetry.sample()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations_per_group):
            fn()
        end.record()
        end.synchronize()
        after = telemetry.sample()

        samples_ms.append(start.elapsed_time(end) / iterations_per_group)
        clocks_before.append(int(before["sm_clock_mhz"]))
        clocks_after.append(int(after["sm_clock_mhz"]))
        powers_w.append(0.5 * (float(before["power_w"]) + float(after["power_w"])))
        temps_c.append(max(int(before["temp_c"]), int(after["temp_c"])))

    all_clocks = clocks_before + clocks_after
    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "stdev_ms": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "groups": groups,
        "iterations_per_group": iterations_per_group,
        "warmup": warmup,
        "group_samples_ms": [round(val, 6) for val in samples_ms],
        "pre_warmup_sm_clock_mhz": int(pre_warmup["sm_clock_mhz"]),
        "post_warmup_sm_clock_mhz": int(post_warmup["sm_clock_mhz"]),
        "group_sm_clocks_before_mhz": clocks_before,
        "group_sm_clocks_after_mhz": clocks_after,
        "median_sm_clock_mhz": float(statistics.median(clocks_after)),
        "min_sm_clock_mhz": min(all_clocks),
        "max_sm_clock_mhz": max(all_clocks),
        "mean_power_w": round(statistics.fmean(powers_w), 2),
        "max_temp_c": max(temps_c),
    }


def merge_two_passes(
    pass_a: dict[str, Any],
    pass_b: dict[str, Any],
) -> dict[str, Any]:
    samples = list(pass_a["group_samples_ms"]) + list(pass_b["group_samples_ms"])
    clocks_before = (
        list(pass_a["group_sm_clocks_before_mhz"])
        + list(pass_b["group_sm_clocks_before_mhz"])
    )
    clocks_after = (
        list(pass_a["group_sm_clocks_after_mhz"])
        + list(pass_b["group_sm_clocks_after_mhz"])
    )
    all_clocks = clocks_before + clocks_after
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "groups": int(pass_a["groups"]) + int(pass_b["groups"]),
        "iterations_per_group": int(pass_a["iterations_per_group"]),
        "warmup": int(pass_a["warmup"]) + int(pass_b["warmup"]),
        "group_samples_ms": [round(val, 6) for val in samples],
        "pre_warmup_sm_clock_mhz": min(
            int(pass_a["pre_warmup_sm_clock_mhz"]),
            int(pass_b["pre_warmup_sm_clock_mhz"]),
        ),
        "post_warmup_sm_clock_mhz": min(
            int(pass_a["post_warmup_sm_clock_mhz"]),
            int(pass_b["post_warmup_sm_clock_mhz"]),
        ),
        "group_sm_clocks_before_mhz": clocks_before,
        "group_sm_clocks_after_mhz": clocks_after,
        "median_sm_clock_mhz": float(statistics.median(clocks_after)),
        "min_sm_clock_mhz": min(all_clocks),
        "max_sm_clock_mhz": max(all_clocks),
        "mean_power_w": round(
            0.5 * (float(pass_a["mean_power_w"]) + float(pass_b["mean_power_w"])),
            2,
        ),
        "max_temp_c": max(int(pass_a["max_temp_c"]), int(pass_b["max_temp_c"])),
    }


def compute_exact_fp32_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ref_chunk: int = 2048,
) -> tuple[float, torch.Tensor, torch.Tensor, float, float]:
    # Exact FP32 chunked reference (mathematically identical to monolithic FP32 linear + FP32 CE)
    num_tokens, hidden = x.shape
    vocab_size, _ = weight.shape
    x_f32 = x.detach().float()
    w_f32 = weight.detach().float()
    w_t_f32 = w_f32.t()

    valid = float((target != -100).sum().clamp_min(1).item())
    total_loss = 0.0
    dx_f32 = torch.empty((num_tokens, hidden), device=x.device, dtype=torch.float32)
    dw_f32 = torch.zeros((vocab_size, hidden), device=x.device, dtype=torch.float32)

    for start in range(0, num_tokens, ref_chunk):
        end = min(start + ref_chunk, num_tokens)
        xc = x_f32[start:end].detach().requires_grad_(True)
        wc_logits = torch.mm(xc, w_t_f32).requires_grad_(True)
        loss_c = F.cross_entropy(
            wc_logits,
            target[start:end],
            ignore_index=-100,
            reduction="sum",
        ) / valid
        dlogits_c = torch.autograd.grad(loss_c, wc_logits)[0]
        total_loss += float(loss_c.detach().item())
        torch.mm(dlogits_c, w_f32, out=dx_f32[start:end])
        dw_f32.addmm_(dlogits_c.t(), xc.detach())
        del xc, wc_logits, loss_c, dlogits_c

    dx_norm = float(dx_f32.norm().item())
    dw_norm = float(dw_f32.norm().item())
    del x_f32, w_f32, w_t_f32
    torch.cuda.empty_cache()
    return total_loss, dx_f32, dw_f32, dx_norm, dw_norm


def evaluate_correctness(
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.nn.Parameter,
    weight: torch.nn.Parameter,
    target: torch.Tensor,
    ref_loss: float,
    ref_dx_f32: torch.Tensor,
    ref_dw_f32: torch.Tensor,
    ref_dx_norm: float,
    ref_dw_norm: float,
) -> dict[str, float]:
    loss_t = run_step(loss_fn, x, weight, target)
    torch.cuda.synchronize()
    loss_val = float(loss_t.detach().item())
    assert x.grad is not None and weight.grad is not None
    dx_1 = x.grad.detach().float()
    dw_1 = weight.grad.detach().float()

    dx_rel = float((dx_1 - ref_dx_f32).norm().item()) / max(ref_dx_norm, 1e-12)
    dw_rel = float((dw_1 - ref_dw_f32).norm().item()) / max(ref_dw_norm, 1e-12)
    dx_max_abs = float((dx_1 - ref_dx_f32).abs().max().item())
    dw_max_abs = float((dw_1 - ref_dw_f32).abs().max().item())

    # Verify arbitrary upstream scalar chain-rule scaling (alpha = 1.75)
    scale = 1.75
    x.grad = None
    weight.grad = None
    scaled_loss = loss_fn(x, weight, target) * scale
    scaled_loss.backward()
    torch.cuda.synchronize()
    assert x.grad is not None and weight.grad is not None
    dx_scaled_err = float((x.grad.detach().float() - dx_1 * scale).norm().item()) / max(
        float((dx_1 * scale).norm().item()),
        1e-12,
    )
    dw_scaled_err = float(
        (weight.grad.detach().float() - dw_1 * scale).norm().item()
    ) / max(float((dw_1 * scale).norm().item()), 1e-12)

    del dx_1, dw_1
    x.grad = None
    weight.grad = None
    return {
        "loss": loss_val,
        "loss_abs_error_vs_fp32": abs(loss_val - ref_loss),
        "dx_l2_rel_error_vs_fp32": dx_rel,
        "dw_l2_rel_error_vs_fp32": dw_rel,
        "dx_max_abs_error_vs_fp32": dx_max_abs,
        "dw_max_abs_error_vs_fp32": dw_max_abs,
        "scaled_backward_chain_rule_rel_error": max(dx_scaled_err, dw_scaled_err),
    }


def classify_ce_cuda_kernel(name: str) -> str:
    lowered = name.lower()
    if any(
        tok in lowered
        for tok in ("gemm", "nvjet", "cutlass", "cublas", "splitkreduce", "bmm")
    ):
        return "gemm"
    return "ce_softmax_cast"


def run_scaling_and_attribution(
    profile_name: str,
    device: torch.device,
    dtype: torch.dtype,
    telemetry: GpuTelemetry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if profile_name == "smoke":
        shapes = [(512, 512, 16384)]
        profiled_shapes = {(512, 512, 16384)}
        default_chunk = 128
        warmup, groups, iters = 2, 2, 2
        prof_steps = 2
    else:
        shapes = [
            (4096, 1024, 128256),
            (8192, 1024, 128256),
            (16384, 1024, 128256),
            (8192, 2048, 128256),
            (8192, 4096, 128256),
        ]
        profiled_shapes = {
            (8192, 1024, 128256),
            (8192, 4096, 128256),
        }
        default_chunk = 1024
        warmup, groups, iters = 6, 4, 6
        prof_steps = 6

    boost_mhz = telemetry.max_clocks()["max_sm_clock_mhz"]
    scaling_records: list[dict[str, Any]] = []
    kernel_records: list[dict[str, Any]] = []

    old_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    try:
        telemetry.lock_sm_clock(boost_mhz)
        with tempfile.TemporaryDirectory(prefix="e112_scaling_cache_") as tmpdir:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(tmpdir) / "cache")
            torch._dynamo.reset()

            for shape_tuple in shapes:
                num_tokens, hidden, vocab_size = shape_tuple
                shape_list = list(shape_tuple)
                useful_gemm_flops = 6.0 * float(num_tokens) * float(hidden) * float(vocab_size)

                torch.manual_seed(31)
                x = torch.nn.Parameter(
                    torch.randn((num_tokens, hidden), device=device, dtype=dtype)
                )
                weight = torch.nn.Parameter(
                    torch.randn((vocab_size, hidden), device=device, dtype=dtype) * 0.02
                )
                target = torch.randint(
                    0,
                    vocab_size,
                    (num_tokens,),
                    device=device,
                    dtype=torch.int64,
                )
                if profile_name == "smoke":
                    target[7] = -100

                ref_loss, ref_dx, ref_dw, ref_dx_norm, ref_dw_norm = (
                    compute_exact_fp32_reference(x, weight, target)
                )

                # Compile unchunked and chunked functions for this shape
                t0 = time.perf_counter()
                compiled_full_fn = torch.compile(
                    eager_fp32_logits_ce,
                    fullgraph=True,
                    mode="default",
                )
                _ = run_step(compiled_full_fn, x, weight, target)
                torch.cuda.synchronize()
                compile_full_setup_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                compiled_chunk_sum = torch.compile(
                    _chunk_ce_sum_fn,
                    fullgraph=True,
                    mode="default",
                )
                ckpt_chunk_fn = make_checkpointed_chunked_ce(
                    compiled_chunk_sum,
                    default_chunk,
                )
                _ = run_step(ckpt_chunk_fn, x, weight, target)
                torch.cuda.synchronize()
                ckpt_setup_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                triton_bf16_fn = lambda x_in, w_in, y_in, cs=default_chunk: triton_fused_linear_cross_entropy(
                    x_in,
                    w_in,
                    y_in,
                    chunk_size=cs,
                    fp32_dw_accum=False,
                )
                _ = run_step(triton_bf16_fn, x, weight, target)
                torch.cuda.synchronize()
                triton_bf16_setup_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                triton_fp32_fn = lambda x_in, w_in, y_in, cs=default_chunk: triton_fused_linear_cross_entropy(
                    x_in,
                    w_in,
                    y_in,
                    chunk_size=cs,
                    fp32_dw_accum=True,
                )
                _ = run_step(triton_fp32_fn, x, weight, target)
                torch.cuda.synchronize()
                triton_fp32_setup_s = time.perf_counter() - t0

                specs = [
                    ("eager_fp32_logits_ce", eager_fp32_logits_ce, 0.0, num_tokens, True),
                    ("eager_bf16_logits_ce", eager_bf16_logits_ce, 0.0, num_tokens, False),
                    (
                        "torch_compile_full_ce",
                        compiled_full_fn,
                        compile_full_setup_s,
                        num_tokens,
                        True,
                    ),
                    (
                        "checkpointed_chunked_compile_ce",
                        ckpt_chunk_fn,
                        ckpt_setup_s,
                        default_chunk,
                        True,
                    ),
                    (
                        "triton_fused_linear_ce_bf16_dw",
                        triton_bf16_fn,
                        triton_bf16_setup_s,
                        default_chunk,
                        True,
                    ),
                    (
                        "triton_fused_linear_ce_fp32_dw",
                        triton_fp32_fn,
                        triton_fp32_setup_s,
                        default_chunk,
                        False,
                    ),
                ]

                candidates: list[dict[str, Any]] = []
                for impl_name, loss_fn, setup_s, chunk_used, prof_flag in specs:
                    corr = evaluate_correctness(
                        loss_fn,
                        x,
                        weight,
                        target,
                        ref_loss,
                        ref_dx,
                        ref_dw,
                        ref_dx_norm,
                        ref_dw_norm,
                    )
                    candidates.append(
                        {
                            "name": impl_name,
                            "step_fn": lambda lf=loss_fn, xp=x, wp=weight, yp=target: run_step(
                                lf,
                                xp,
                                wp,
                                yp,
                            ),
                            "setup_seconds": setup_s,
                            "chunk_size": chunk_used,
                            "correctness": corr,
                            "profile_kernels": prof_flag,
                        }
                    )

                del ref_dx, ref_dw
                torch.cuda.empty_cache()

                # Palindromic timing (1..K then K..1) at locked boost clock
                telemetry.lock_sm_clock(boost_mhz)
                pass_a: dict[str, dict[str, Any]] = {}
                pass_b: dict[str, dict[str, Any]] = {}
                for cand in candidates:
                    pass_a[cand["name"]] = benchmark_with_telemetry(
                        cand["step_fn"],
                        telemetry,
                        warmup=warmup,
                        groups=groups,
                        iterations_per_group=iters,
                    )
                for cand in reversed(candidates):
                    pass_b[cand["name"]] = benchmark_with_telemetry(
                        cand["step_fn"],
                        telemetry,
                        warmup=warmup,
                        groups=groups,
                        iterations_per_group=iters,
                    )

                if shape_tuple in profiled_shapes:
                    for cand in candidates:
                        if not cand["profile_kernels"]:
                            continue
                        for _ in range(3):
                            cand["step_fn"]()
                        torch.cuda.synchronize()
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        ) as prof:
                            for _ in range(prof_steps):
                                cand["step_fn"]()
                            torch.cuda.synchronize()

                        cat_us = {"gemm": 0.0, "ce_softmax_cast": 0.0}
                        cat_cnt = {"gemm": 0.0, "ce_softmax_cast": 0.0}
                        cpu_launches = 0.0
                        top_rows: list[dict[str, Any]] = []

                        for evt in prof.key_averages():
                            dev_str = str(getattr(evt, "device_type", ""))
                            if dev_str == "DeviceType.CPU" and evt.key in (
                                "cudaLaunchKernel",
                                "cuLaunchKernel",
                                "cudaLaunchKernelExC",
                            ):
                                cpu_launches += evt.count / prof_steps
                            elif dev_str == "DeviceType.CUDA":
                                us = (
                                    float(getattr(evt, "self_device_time_total", 0.0))
                                    / prof_steps
                                )
                                cnt = float(evt.count) / prof_steps
                                cat = classify_ce_cuda_kernel(evt.key)
                                cat_us[cat] += us
                                cat_cnt[cat] += cnt
                                top_rows.append(
                                    {
                                        "kernel": evt.key[:140],
                                        "category": cat,
                                        "calls_per_step": round(cnt, 2),
                                        "cuda_us_per_step": round(us, 3),
                                    }
                                )

                        top_rows.sort(key=lambda item: item["cuda_us_per_step"], reverse=True)
                        tot_us = sum(cat_us.values())
                        denom = max(tot_us, 1e-9)
                        kernel_records.append(
                            {
                                "experiment": "cuda_kernel_attribution",
                                "clock_policy": f"locked_{boost_mhz}mhz",
                                "implementation": cand["name"],
                                "shape_bt_h_v": shape_list,
                                "chunk_size": cand["chunk_size"],
                                "dtype": str(dtype),
                                "profiled_steps": prof_steps,
                                "total_cuda_kernel_us_per_step": round(tot_us, 3),
                                "gemm_us_per_step": round(cat_us["gemm"], 3),
                                "ce_softmax_cast_us_per_step": round(
                                    cat_us["ce_softmax_cast"],
                                    3,
                                ),
                                "gemm_share": round(cat_us["gemm"] / denom, 6),
                                "ce_softmax_cast_share": round(
                                    cat_us["ce_softmax_cast"] / denom,
                                    6,
                                ),
                                "cuda_kernels_per_step": round(sum(cat_cnt.values()), 2),
                                "cpu_launch_calls_per_step": round(cpu_launches, 2),
                                "top_cuda_kernels": top_rows[:8],
                            }
                        )

                # Measure clean step memory decomposition after profiling
                for cand in candidates:
                    merged = merge_two_passes(pass_a[cand["name"]], pass_b[cand["name"]])
                    mem_stats = measure_step_memory(cand["step_fn"], x, weight)
                    sec = merged["median_ms"] / 1000.0
                    scaling_records.append(
                        {
                            "experiment": "slm_llm_vocab_scaling",
                            "protocol": "locked_boost_palindromic_balanced",
                            "clock_policy": f"locked_{boost_mhz}mhz",
                            "implementation": cand["name"],
                            "shape_bt_h_v": shape_list,
                            "chunk_size": cand["chunk_size"],
                            "dtype": str(dtype),
                            "setup_seconds": cand["setup_seconds"],
                            **cand["correctness"],
                            **merged,
                            **mem_stats,
                            "tokens_per_second": num_tokens / sec,
                            "effective_3gemm_tflop_s": useful_gemm_flops / sec / 1e12,
                        }
                    )

                del candidates, x, weight, target
                torch._dynamo.reset()
                torch.cuda.empty_cache()
    finally:
        telemetry.reset_clocks()
        if old_cache_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = old_cache_dir

    return scaling_records, kernel_records


def run_chunk_size_frontier(
    profile_name: str,
    device: torch.device,
    dtype: torch.dtype,
    telemetry: GpuTelemetry,
) -> list[dict[str, Any]]:
    if profile_name == "smoke":
        shapes = [(512, 512, 16384)]
        chunk_sizes = [128, 256, 512]
        warmup, groups, iters = 2, 2, 2
    else:
        shapes = [
            (8192, 1024, 128256),
            (8192, 4096, 128256),
        ]
        chunk_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
        warmup, groups, iters = 6, 4, 6

    boost_mhz = telemetry.max_clocks()["max_sm_clock_mhz"]
    records: list[dict[str, Any]] = []

    try:
        telemetry.lock_sm_clock(boost_mhz)
        for num_tokens, hidden, vocab_size in shapes:
            shape_list = [num_tokens, hidden, vocab_size]
            useful_gemm_flops = (
                6.0 * float(num_tokens) * float(hidden) * float(vocab_size)
            )
            torch.manual_seed(43)
            x = torch.nn.Parameter(
                torch.randn((num_tokens, hidden), device=device, dtype=dtype)
            )
            weight = torch.nn.Parameter(
                torch.randn((vocab_size, hidden), device=device, dtype=dtype) * 0.02
            )
            target = torch.randint(
                0,
                vocab_size,
                (num_tokens,),
                device=device,
                dtype=torch.int64,
            )

            ref_loss, ref_dx, ref_dw, ref_dx_norm, ref_dw_norm = (
                compute_exact_fp32_reference(x, weight, target)
            )

            candidates: list[dict[str, Any]] = []
            for fp32_dw in (False, True):
                accum_tag = "fp32_dw" if fp32_dw else "bf16_dw"
                impl_name = f"triton_fused_linear_ce_{accum_tag}"
                for chunk_size in chunk_sizes:
                    loss_fn = (
                        lambda x_in, w_in, y_in, cs=chunk_size, fp=fp32_dw: triton_fused_linear_cross_entropy(
                            x_in,
                            w_in,
                            y_in,
                            chunk_size=cs,
                            fp32_dw_accum=fp,
                        )
                    )
                    corr = evaluate_correctness(
                        loss_fn,
                        x,
                        weight,
                        target,
                        ref_loss,
                        ref_dx,
                        ref_dw,
                        ref_dx_norm,
                        ref_dw_norm,
                    )
                    key = f"{impl_name}_c{chunk_size}"
                    candidates.append(
                        {
                            "key": key,
                            "implementation": impl_name,
                            "dw_accum_dtype": "float32" if fp32_dw else "bfloat16",
                            "chunk_size": chunk_size,
                            "num_chunks": math.ceil(num_tokens / chunk_size),
                            "step_fn": lambda lf=loss_fn, xp=x, wp=weight, yp=target: run_step(
                                lf,
                                xp,
                                wp,
                                yp,
                            ),
                            "correctness": corr,
                        }
                    )

            del ref_dx, ref_dw
            torch.cuda.empty_cache()

            telemetry.lock_sm_clock(boost_mhz)
            pass_a: dict[str, dict[str, Any]] = {}
            pass_b: dict[str, dict[str, Any]] = {}
            for cand in candidates:
                pass_a[cand["key"]] = benchmark_with_telemetry(
                    cand["step_fn"],
                    telemetry,
                    warmup=warmup,
                    groups=groups,
                    iterations_per_group=iters,
                )
            for cand in reversed(candidates):
                pass_b[cand["key"]] = benchmark_with_telemetry(
                    cand["step_fn"],
                    telemetry,
                    warmup=warmup,
                    groups=groups,
                    iterations_per_group=iters,
                )

            for cand in candidates:
                merged = merge_two_passes(pass_a[cand["key"]], pass_b[cand["key"]])
                mem_stats = measure_step_memory(cand["step_fn"], x, weight)
                sec = merged["median_ms"] / 1000.0
                records.append(
                    {
                        "experiment": "chunk_size_frontier",
                        "protocol": "locked_boost_palindromic_balanced",
                        "clock_policy": f"locked_{boost_mhz}mhz",
                        "implementation": cand["implementation"],
                        "dw_accum_dtype": cand["dw_accum_dtype"],
                        "shape_bt_h_v": shape_list,
                        "chunk_size": cand["chunk_size"],
                        "num_chunks": cand["num_chunks"],
                        "dtype": str(dtype),
                        **cand["correctness"],
                        **merged,
                        **mem_stats,
                        "tokens_per_second": num_tokens / sec,
                        "effective_3gemm_tflop_s": useful_gemm_flops / sec / 1e12,
                    }
                )

            del candidates, x, weight, target
            torch.cuda.empty_cache()
    finally:
        telemetry.reset_clocks()

    return records


def run(profile_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    telemetry = GpuTelemetry()
    atexit.register(telemetry.reset_clocks)
    telemetry.reset_clocks()

    env = environment_metadata()
    if env.get("hostname"):
        env["hostname"] = "node-0"
    env["nvml_max_clocks"] = telemetry.max_clocks()
    env["initial_telemetry"] = telemetry.sample()

    try:
        scaling_records, kernel_records = run_scaling_and_attribution(
            profile_name,
            device,
            dtype,
            telemetry,
        )
        chunk_records = run_chunk_size_frontier(
            profile_name,
            device,
            dtype,
            telemetry,
        )
    finally:
        telemetry.reset_clocks()

    time.sleep(0.2)
    env["final_unlocked_telemetry"] = telemetry.sample()

    return {
        "suite": "fused_linear_cross_entropy",
        "profile": profile_name,
        "environment": env,
        "records": scaling_records + chunk_records,
        "kernel_attribution_records": kernel_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile))


if __name__ == "__main__":
    main()
