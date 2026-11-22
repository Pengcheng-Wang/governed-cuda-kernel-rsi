from __future__ import annotations

import argparse
import atexit
import copy
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

from bench_transformer_training import TransformerBlock, training_step
from common import environment_metadata, measure_peak_memory, write_result


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

    group_samples_ms: list[float] = []
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

        group_samples_ms.append(start.elapsed_time(end) / iterations_per_group)
        clocks_before.append(int(before["sm_clock_mhz"]))
        clocks_after.append(int(after["sm_clock_mhz"]))
        powers_w.append(0.5 * (float(before["power_w"]) + float(after["power_w"])))
        temps_c.append(max(int(before["temp_c"]), int(after["temp_c"])))

    all_clocks = clocks_before + clocks_after
    return {
        "median_ms": statistics.median(group_samples_ms),
        "min_ms": min(group_samples_ms),
        "max_ms": max(group_samples_ms),
        "mean_ms": statistics.fmean(group_samples_ms),
        "stdev_ms": statistics.pstdev(group_samples_ms) if len(group_samples_ms) > 1 else 0.0,
        "groups": groups,
        "iterations_per_group": iterations_per_group,
        "warmup": warmup,
        "group_samples_ms": [round(val, 6) for val in group_samples_ms],
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


def snapshot_gradients(model: torch.nn.Module) -> list[torch.Tensor]:
    grads: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("Expected populated parameter gradient")
        grads.append(parameter.grad.detach().float().clone())
    return grads


def compare_gradients(
    actual: list[torch.Tensor],
    reference: list[torch.Tensor],
) -> dict[str, float]:
    diff_sq = 0.0
    ref_sq = 0.0
    act_sq = 0.0
    max_abs_diff = 0.0
    max_abs_val = 0.0
    for act_g, ref_g in zip(actual, reference, strict=True):
        delta = act_g - ref_g
        diff_sq += delta.square().sum().item()
        ref_sq += ref_g.square().sum().item()
        act_sq += act_g.square().sum().item()
        max_abs_diff = max(max_abs_diff, delta.abs().max().item())
        max_abs_val = max(max_abs_val, act_g.abs().max().item())
    ref_norm = math.sqrt(ref_sq)
    act_norm = math.sqrt(act_sq)
    return {
        "gradient_l2_norm": act_norm,
        "gradient_l2_relative_error": abs(act_norm - ref_norm) / max(ref_norm, 1e-12),
        "gradient_elementwise_l2_relative_error": math.sqrt(diff_sq)
        / max(ref_norm, 1e-12),
        "gradient_elementwise_max_abs_error": max_abs_diff,
        "gradient_max_abs": max_abs_val,
    }


def capture_full_step_training_graph(
    base_model: torch.nn.Module,
    sample_x: torch.Tensor,
    *,
    compile_default_forward_backward: bool = False,
) -> tuple[Callable[[], torch.Tensor], torch.nn.Module, float]:
    started = time.perf_counter()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        graph_model = copy.deepcopy(base_model)
        static_x = sample_x.clone()
        forward_callable: Callable[[torch.Tensor], torch.Tensor]
        if compile_default_forward_backward:
            forward_callable = torch.compile(
                graph_model,
                fullgraph=True,
                mode="default",
            )
        else:
            forward_callable = graph_model
        for _ in range(3):
            graph_model.zero_grad(set_to_none=False)
            warmup_out = forward_callable(static_x)
            warmup_loss = warmup_out.float().square().mean()
            warmup_loss.backward()
            del warmup_out, warmup_loss
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_model.zero_grad(set_to_none=False)
        static_out = forward_callable(static_x)
        static_loss = static_out.float().square().mean()
        static_loss.backward()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started

    def replay() -> torch.Tensor:
        static_x.copy_(sample_x)
        graph.replay()
        return static_loss

    return replay, graph_model, setup_seconds


def classify_cuda_kernel(name: str) -> str:
    lowered = name.lower()
    if any(tok in lowered for tok in ("flash", "fmha", "attention", "cudnn_sdpa", "sdp")):
        return "attention"
    if any(
        tok in lowered
        for tok in (
            "gemm",
            "nvjet",
            "cutlass",
            "cublas",
            "triton_mm",
            "splitkreduce",
            "bmm",
        )
    ):
        return "gemm"
    return "norm_pointwise_loss"


def run_dvfs_cache_order_ablation(
    profile_name: str,
    device: torch.device,
    dtype: torch.dtype,
    telemetry: GpuTelemetry,
) -> list[dict[str, Any]]:
    if profile_name == "smoke":
        shapes = [(1, 128, 512, 8, 1408)]
        settle_s = 0.25
        sustained_warmup, iso_warmup, iso_groups, iso_iters = 4, 2, 2, 2
    else:
        shapes = [
            (4, 1024, 1024, 16, 2816),
            (1, 4096, 1024, 16, 2816),
        ]
        settle_s = 2.5
        sustained_warmup, iso_warmup, iso_groups, iso_iters = 30, 15, 5, 20

    clock_caps = telemetry.max_clocks()
    base_sm_mhz = clock_caps["default_app_sm_clock_mhz"]
    boost_sm_mhz = clock_caps["max_sm_clock_mhz"]

    old_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    old_fx_cache = os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE")
    records: list[dict[str, Any]] = []

    try:
        for batch, sequence, hidden, heads, intermediate in shapes:
            shape_list = [batch, sequence, hidden, heads, intermediate]
            tokens = batch * sequence
            with tempfile.TemporaryDirectory(prefix="e111_inductor_ablation_") as tmpdir:
                shared_cache = Path(tmpdir) / "cache"
                shared_cache.mkdir(parents=True, exist_ok=True)
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(shared_cache)
                os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

                torch.manual_seed(23)
                eager_model = TransformerBlock(hidden, heads, intermediate).to(
                    device=device,
                    dtype=dtype,
                )
                cold_base = copy.deepcopy(eager_model)
                warm_base = copy.deepcopy(eager_model)
                x = torch.randn((batch, sequence, hidden), device=device, dtype=dtype)

                # Reference correctness snapshot under default unlocked clock policy
                telemetry.reset_clocks()
                ref_loss_t = training_step(eager_model, x)
                torch.cuda.synchronize()
                ref_loss = ref_loss_t.detach().item()
                ref_grads = snapshot_gradients(eager_model)
                zero_grad_metrics = compare_gradients(ref_grads, ref_grads)

                # 1. Exact legacy sequence with COLD Inductor cache (unlocked DVFS)
                torch._dynamo.reset()
                torch.cuda.synchronize()
                time.sleep(settle_s)
                idle_before_cold = telemetry.sample()

                _ = training_step(eager_model, x)
                torch.cuda.synchronize()
                cold_started = time.perf_counter()
                cold_compiled = torch.compile(cold_base, fullgraph=True, mode="default")
                cold_loss_t = training_step(cold_compiled, x)
                torch.cuda.synchronize()
                cold_compile_s = time.perf_counter() - cold_started
                post_cold_compile_tel = telemetry.sample()
                cold_loss = cold_loss_t.detach().item()
                cold_grad_metrics = compare_gradients(
                    snapshot_gradients(cold_base),
                    ref_grads,
                )

                eager_fn = lambda model=eager_model, inp=x: training_step(model, inp)
                cold_fn = lambda model=cold_compiled, inp=x: training_step(model, inp)

                t_cold_eager = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=3,
                    groups=5,
                    iterations_per_group=5,
                )
                mem_cold_eager = measure_peak_memory(eager_fn)
                t_cold_comp = benchmark_with_telemetry(
                    cold_fn,
                    telemetry,
                    warmup=3,
                    groups=5,
                    iterations_per_group=5,
                )
                mem_cold_comp = measure_peak_memory(cold_fn)

                for impl_name, timing, setup_s, loss_v, g_metrics, peak_mem in [
                    (
                        "torch_eager",
                        t_cold_eager,
                        0.0,
                        ref_loss,
                        zero_grad_metrics,
                        mem_cold_eager,
                    ),
                    (
                        "compile_default",
                        t_cold_comp,
                        cold_compile_s,
                        cold_loss,
                        cold_grad_metrics,
                        mem_cold_comp,
                    ),
                ]:
                    records.append(
                        {
                            "experiment": "dvfs_cache_order_ablation",
                            "protocol": "legacy_cold_cache_eager_first",
                            "clock_policy": "unlocked_dvfs",
                            "inductor_cache_state": "cold",
                            "execution_order": "eager_then_compile",
                            "implementation": impl_name,
                            "shape": shape_list,
                            "dtype": str(dtype),
                            "setup_seconds": setup_s,
                            "idle_before_protocol_sm_clock_mhz": int(
                                idle_before_cold["sm_clock_mhz"]
                            ),
                            "post_compile_sm_clock_mhz": int(
                                post_cold_compile_tel["sm_clock_mhz"]
                            ),
                            "loss": loss_v,
                            "loss_abs_error": abs(loss_v - ref_loss),
                            **g_metrics,
                            **timing,
                            "peak_incremental_memory_bytes": peak_mem,
                            "tokens_per_second": tokens / (timing["median_ms"] / 1000.0),
                        }
                    )

                # Measure the exact cold-compiled artifact at hardware-locked 1410 MHz BEFORE dynamo.reset()
                telemetry.lock_sm_clock(boost_sm_mhz)
                t_iso_cold_art_a = benchmark_with_telemetry(
                    cold_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_iso_cold_art_b = benchmark_with_telemetry(
                    cold_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_iso_cold_art = merge_two_passes(t_iso_cold_art_a, t_iso_cold_art_b)
                telemetry.reset_clocks()

                # 2. Exact legacy sequence with WARM Inductor cache (unlocked DVFS)
                del cold_compiled
                torch._dynamo.reset()
                torch.cuda.synchronize()
                time.sleep(settle_s)
                idle_before_warm = telemetry.sample()

                _ = training_step(eager_model, x)
                torch.cuda.synchronize()
                warm_started = time.perf_counter()
                warm_compiled = torch.compile(warm_base, fullgraph=True, mode="default")
                warm_loss_t = training_step(warm_compiled, x)
                torch.cuda.synchronize()
                warm_compile_s = time.perf_counter() - warm_started
                post_warm_compile_tel = telemetry.sample()
                warm_loss = warm_loss_t.detach().item()
                warm_grad_metrics = compare_gradients(
                    snapshot_gradients(warm_base),
                    ref_grads,
                )

                warm_fn = lambda model=warm_compiled, inp=x: training_step(model, inp)

                t_warm_eager = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=3,
                    groups=5,
                    iterations_per_group=5,
                )
                mem_warm_eager = measure_peak_memory(eager_fn)
                t_warm_comp = benchmark_with_telemetry(
                    warm_fn,
                    telemetry,
                    warmup=3,
                    groups=5,
                    iterations_per_group=5,
                )
                mem_warm_comp = measure_peak_memory(warm_fn)

                for impl_name, timing, setup_s, loss_v, g_metrics, peak_mem in [
                    (
                        "torch_eager",
                        t_warm_eager,
                        0.0,
                        ref_loss,
                        zero_grad_metrics,
                        mem_warm_eager,
                    ),
                    (
                        "compile_default",
                        t_warm_comp,
                        warm_compile_s,
                        warm_loss,
                        warm_grad_metrics,
                        mem_warm_comp,
                    ),
                ]:
                    records.append(
                        {
                            "experiment": "dvfs_cache_order_ablation",
                            "protocol": "legacy_warm_cache_eager_first",
                            "clock_policy": "unlocked_dvfs",
                            "inductor_cache_state": "warm",
                            "execution_order": "eager_then_compile",
                            "implementation": impl_name,
                            "shape": shape_list,
                            "dtype": str(dtype),
                            "setup_seconds": setup_s,
                            "idle_before_protocol_sm_clock_mhz": int(
                                idle_before_warm["sm_clock_mhz"]
                            ),
                            "post_compile_sm_clock_mhz": int(
                                post_warm_compile_tel["sm_clock_mhz"]
                            ),
                            "loss": loss_v,
                            "loss_abs_error": abs(loss_v - ref_loss),
                            **g_metrics,
                            **timing,
                            "peak_incremental_memory_bytes": peak_mem,
                            "tokens_per_second": tokens / (timing["median_ms"] / 1000.0),
                        }
                    )

                # 3. Reversed execution order from idle (compile first, eager second; unlocked DVFS)
                torch.cuda.synchronize()
                time.sleep(settle_s)
                idle_before_rev = telemetry.sample()

                t_rev_comp = benchmark_with_telemetry(
                    warm_fn,
                    telemetry,
                    warmup=3,
                    groups=5,
                    iterations_per_group=5,
                )
                _ = measure_peak_memory(warm_fn)
                t_rev_eager = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=3,
                    groups=5,
                    iterations_per_group=5,
                )
                _ = measure_peak_memory(eager_fn)

                for impl_name, timing, setup_s, loss_v, g_metrics, peak_mem in [
                    (
                        "compile_default",
                        t_rev_comp,
                        warm_compile_s,
                        warm_loss,
                        warm_grad_metrics,
                        mem_warm_comp,
                    ),
                    (
                        "torch_eager",
                        t_rev_eager,
                        0.0,
                        ref_loss,
                        zero_grad_metrics,
                        mem_warm_eager,
                    ),
                ]:
                    records.append(
                        {
                            "experiment": "dvfs_cache_order_ablation",
                            "protocol": "idle_warm_cache_compile_first",
                            "clock_policy": "unlocked_dvfs",
                            "inductor_cache_state": "warm",
                            "execution_order": "compile_then_eager",
                            "implementation": impl_name,
                            "shape": shape_list,
                            "dtype": str(dtype),
                            "setup_seconds": setup_s,
                            "idle_before_protocol_sm_clock_mhz": int(
                                idle_before_rev["sm_clock_mhz"]
                            ),
                            "post_compile_sm_clock_mhz": int(
                                idle_before_rev["sm_clock_mhz"]
                            ),
                            "loss": loss_v,
                            "loss_abs_error": abs(loss_v - ref_loss),
                            **g_metrics,
                            **timing,
                            "peak_incremental_memory_bytes": peak_mem,
                            "tokens_per_second": tokens / (timing["median_ms"] / 1000.0),
                        }
                    )

                # 4. Unlocked sustained-warmup ABBA (tests whether longer warmup alone cures DVFS drift)
                torch.cuda.synchronize()
                time.sleep(settle_s)
                idle_before_sust = telemetry.sample()
                t_sust_e1 = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=sustained_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_sust_w1 = benchmark_with_telemetry(
                    warm_fn,
                    telemetry,
                    warmup=sustained_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_sust_w2 = benchmark_with_telemetry(
                    warm_fn,
                    telemetry,
                    warmup=sustained_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_sust_e2 = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=sustained_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                for impl_name, timing, setup_s, loss_v, g_metrics, peak_mem in [
                    (
                        "torch_eager",
                        merge_two_passes(t_sust_e1, t_sust_e2),
                        0.0,
                        ref_loss,
                        zero_grad_metrics,
                        mem_warm_eager,
                    ),
                    (
                        "compile_default",
                        merge_two_passes(t_sust_w1, t_sust_w2),
                        warm_compile_s,
                        warm_loss,
                        warm_grad_metrics,
                        mem_warm_comp,
                    ),
                ]:
                    records.append(
                        {
                            "experiment": "dvfs_cache_order_ablation",
                            "protocol": "unlocked_sustained_abba",
                            "clock_policy": "unlocked_dvfs",
                            "inductor_cache_state": "warm",
                            "execution_order": "abba_balanced",
                            "implementation": impl_name,
                            "shape": shape_list,
                            "dtype": str(dtype),
                            "setup_seconds": setup_s,
                            "idle_before_protocol_sm_clock_mhz": int(
                                idle_before_sust["sm_clock_mhz"]
                            ),
                            "post_compile_sm_clock_mhz": int(
                                idle_before_sust["sm_clock_mhz"]
                            ),
                            "loss": loss_v,
                            "loss_abs_error": abs(loss_v - ref_loss),
                            **g_metrics,
                            **timing,
                            "peak_incremental_memory_bytes": peak_mem,
                            "tokens_per_second": tokens / (timing["median_ms"] / 1000.0),
                        }
                    )

                # 5. Hardware-locked 1140 MHz base application clock ABBA
                telemetry.lock_sm_clock(base_sm_mhz)
                locked_base_tel = telemetry.sample()
                t_b1140_e1 = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_b1140_w1 = benchmark_with_telemetry(
                    warm_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_b1140_w2 = benchmark_with_telemetry(
                    warm_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_b1140_e2 = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                for impl_name, timing, setup_s, loss_v, g_metrics, peak_mem in [
                    (
                        "torch_eager",
                        merge_two_passes(t_b1140_e1, t_b1140_e2),
                        0.0,
                        ref_loss,
                        zero_grad_metrics,
                        mem_warm_eager,
                    ),
                    (
                        "compile_default",
                        merge_two_passes(t_b1140_w1, t_b1140_w2),
                        warm_compile_s,
                        warm_loss,
                        warm_grad_metrics,
                        mem_warm_comp,
                    ),
                ]:
                    records.append(
                        {
                            "experiment": "dvfs_cache_order_ablation",
                            "protocol": "locked_base_clock_abba",
                            "clock_policy": f"locked_{base_sm_mhz}mhz",
                            "inductor_cache_state": "warm",
                            "execution_order": "abba_balanced",
                            "implementation": impl_name,
                            "shape": shape_list,
                            "dtype": str(dtype),
                            "setup_seconds": setup_s,
                            "idle_before_protocol_sm_clock_mhz": int(
                                locked_base_tel["sm_clock_mhz"]
                            ),
                            "post_compile_sm_clock_mhz": int(
                                locked_base_tel["sm_clock_mhz"]
                            ),
                            "loss": loss_v,
                            "loss_abs_error": abs(loss_v - ref_loss),
                            **g_metrics,
                            **timing,
                            "peak_incremental_memory_bytes": peak_mem,
                            "tokens_per_second": tokens / (timing["median_ms"] / 1000.0),
                        }
                    )

                # 6. Hardware-locked 1410 MHz boost clock ABBA (+ cold-vs-warm artifact check)
                telemetry.lock_sm_clock(boost_sm_mhz)
                locked_boost_tel = telemetry.sample()
                t_b1410_e1 = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_b1410_w1 = benchmark_with_telemetry(
                    warm_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_b1410_w2 = benchmark_with_telemetry(
                    warm_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                t_b1410_e2 = benchmark_with_telemetry(
                    eager_fn,
                    telemetry,
                    warmup=iso_warmup,
                    groups=iso_groups,
                    iterations_per_group=iso_iters,
                )
                for impl_name, cache_tag, timing, setup_s, loss_v, g_metrics, peak_mem in [
                    (
                        "torch_eager",
                        "n/a",
                        merge_two_passes(t_b1410_e1, t_b1410_e2),
                        0.0,
                        ref_loss,
                        zero_grad_metrics,
                        mem_warm_eager,
                    ),
                    (
                        "compile_default",
                        "warm",
                        merge_two_passes(t_b1410_w1, t_b1410_w2),
                        warm_compile_s,
                        warm_loss,
                        warm_grad_metrics,
                        mem_warm_comp,
                    ),
                    (
                        "compile_default_cold_artifact",
                        "cold",
                        t_iso_cold_art,
                        cold_compile_s,
                        cold_loss,
                        cold_grad_metrics,
                        mem_cold_comp,
                    ),
                ]:
                    records.append(
                        {
                            "experiment": "dvfs_cache_order_ablation",
                            "protocol": "locked_boost_clock_abba",
                            "clock_policy": f"locked_{boost_sm_mhz}mhz",
                            "inductor_cache_state": cache_tag,
                            "execution_order": "abba_balanced",
                            "implementation": impl_name,
                            "shape": shape_list,
                            "dtype": str(dtype),
                            "setup_seconds": setup_s,
                            "idle_before_protocol_sm_clock_mhz": int(
                                locked_boost_tel["sm_clock_mhz"]
                            ),
                            "post_compile_sm_clock_mhz": int(
                                locked_boost_tel["sm_clock_mhz"]
                            ),
                            "loss": loss_v,
                            "loss_abs_error": abs(loss_v - ref_loss),
                            **g_metrics,
                            **timing,
                            "peak_incremental_memory_bytes": peak_mem,
                            "tokens_per_second": tokens / (timing["median_ms"] / 1000.0),
                        }
                    )

                telemetry.reset_clocks()
                del eager_model, cold_base, warm_base, warm_compiled, x, ref_grads
                torch._dynamo.reset()
                torch.cuda.empty_cache()
    finally:
        telemetry.reset_clocks()
        if old_cache_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = old_cache_dir
        if old_fx_cache is None:
            os.environ.pop("TORCHINDUCTOR_FX_GRAPH_CACHE", None)
        else:
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = old_fx_cache

    return records


def run_compiler_mode_frontier_and_attribution(
    profile_name: str,
    device: torch.device,
    dtype: torch.dtype,
    telemetry: GpuTelemetry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if profile_name == "smoke":
        shapes = [(1, 128, 512, 8, 1408)]
        warmup, groups, iters = 2, 2, 2
        prof_steps = 2
    else:
        shapes = [
            (1, 512, 1024, 16, 2816),
            (4, 1024, 1024, 16, 2816),
            (1, 4096, 1024, 16, 2816),
        ]
        warmup, groups, iters = 15, 5, 20
        prof_steps = 10

    boost_sm_mhz = telemetry.max_clocks()["max_sm_clock_mhz"]
    mode_records: list[dict[str, Any]] = []
    kernel_records: list[dict[str, Any]] = []

    old_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    try:
        telemetry.lock_sm_clock(boost_sm_mhz)
        with tempfile.TemporaryDirectory(prefix="e111_mode_frontier_") as tmpdir:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(tmpdir) / "cache")
            torch._dynamo.reset()

            for shape_tuple in shapes:
                batch, sequence, hidden, heads, intermediate = shape_tuple
                shape_list = list(shape_tuple)
                tokens = batch * sequence

                torch.manual_seed(23)
                eager_model = TransformerBlock(hidden, heads, intermediate).to(
                    device=device,
                    dtype=dtype,
                )
                x = torch.randn((batch, sequence, hidden), device=device, dtype=dtype)

                ref_loss_t = training_step(eager_model, x)
                torch.cuda.synchronize()
                ref_loss = ref_loss_t.detach().item()
                ref_grads = snapshot_gradients(eager_model)

                candidates: list[dict[str, Any]] = [
                    {
                        "name": "torch_eager",
                        "fn": lambda m=eager_model, inp=x: training_step(m, inp),
                        "setup_seconds": 0.0,
                        "loss": ref_loss,
                        "grad_metrics": compare_gradients(ref_grads, ref_grads),
                        "profile_kernels": True,
                    }
                ]

                # Explicit CUDA Graph around eager forward + loss + backward
                graph_fn, graph_model, graph_setup_s = capture_full_step_training_graph(
                    eager_model,
                    x,
                    compile_default_forward_backward=False,
                )
                graph_loss_t = graph_fn()
                torch.cuda.synchronize()
                graph_loss = graph_loss_t.detach().item()
                graph_grad_metrics = compare_gradients(
                    snapshot_gradients(graph_model),
                    ref_grads,
                )
                candidates.append(
                    {
                        "name": "explicit_cuda_graph_eager",
                        "fn": graph_fn,
                        "setup_seconds": graph_setup_s,
                        "loss": graph_loss,
                        "grad_metrics": graph_grad_metrics,
                        "profile_kernels": False,
                    }
                )

                compile_specs = [
                    ("compile_default", "default", False, True),
                    (
                        "compile_max_autotune_no_cudagraphs",
                        "max-autotune-no-cudagraphs",
                        False,
                        True,
                    ),
                    ("compile_reduce_overhead", "reduce-overhead", True, False),
                    ("compile_max_autotune", "max-autotune", True, False),
                ]
                kept_models: list[Any] = [eager_model, graph_model]

                for impl_name, mode_name, uses_cudagraph_trees, prof_flag in compile_specs:
                    base_m = copy.deepcopy(eager_model)
                    started = time.perf_counter()
                    comp_m = torch.compile(base_m, fullgraph=True, mode=mode_name)
                    warmup_calls = 3 if uses_cudagraph_trees else 1
                    last_loss_val = 0.0
                    last_grads: list[torch.Tensor] = []
                    for call_idx in range(warmup_calls):
                        if uses_cudagraph_trees:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss_t = training_step(comp_m, x)
                        torch.cuda.synchronize()
                        if call_idx == warmup_calls - 1:
                            last_loss_val = loss_t.detach().item()
                            last_grads = snapshot_gradients(base_m)
                    setup_s = time.perf_counter() - started
                    grad_metrics = compare_gradients(last_grads, ref_grads)

                    if uses_cudagraph_trees:
                        def step_fn(cm=comp_m, inp=x) -> torch.Tensor:
                            torch.compiler.cudagraph_mark_step_begin()
                            return training_step(cm, inp)
                    else:
                        def step_fn(cm=comp_m, inp=x) -> torch.Tensor:
                            return training_step(cm, inp)

                    candidates.append(
                        {
                            "name": impl_name,
                            "fn": step_fn,
                            "setup_seconds": setup_s,
                            "loss": last_loss_val,
                            "grad_metrics": grad_metrics,
                            "profile_kernels": prof_flag,
                        }
                    )
                    kept_models.extend([base_m, comp_m])

                # Explicit full-step CUDA Graph around Inductor default-compiled forward+backward
                cgraph_fn, cgraph_model, cgraph_setup_s = capture_full_step_training_graph(
                    eager_model,
                    x,
                    compile_default_forward_backward=True,
                )
                cgraph_loss_t = cgraph_fn()
                torch.cuda.synchronize()
                cgraph_loss = cgraph_loss_t.detach().item()
                cgraph_grad_metrics = compare_gradients(
                    snapshot_gradients(cgraph_model),
                    ref_grads,
                )
                candidates.append(
                    {
                        "name": "explicit_cuda_graph_compiled",
                        "fn": cgraph_fn,
                        "setup_seconds": cgraph_setup_s,
                        "loss": cgraph_loss,
                        "grad_metrics": cgraph_grad_metrics,
                        "profile_kernels": False,
                    }
                )
                kept_models.append(cgraph_model)

                # Palindromic order-balanced timing (1..N then N..1) at locked boost clock
                telemetry.lock_sm_clock(boost_sm_mhz)
                pass_a_map: dict[str, dict[str, Any]] = {}
                pass_b_map: dict[str, dict[str, Any]] = {}

                for cand in candidates:
                    pass_a_map[cand["name"]] = benchmark_with_telemetry(
                        cand["fn"],
                        telemetry,
                        warmup=warmup,
                        groups=groups,
                        iterations_per_group=iters,
                    )

                for cand in reversed(candidates):
                    pass_b_map[cand["name"]] = benchmark_with_telemetry(
                        cand["fn"],
                        telemetry,
                        warmup=warmup,
                        groups=groups,
                        iterations_per_group=iters,
                    )

                # CUDA kernel attribution on non-CUDAGraph implementations at locked boost clock
                for cand in candidates:
                    if not cand["profile_kernels"]:
                        continue
                    for _ in range(5):
                        cand["fn"]()
                    torch.cuda.synchronize()
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    ) as prof:
                        for _ in range(prof_steps):
                            cand["fn"]()
                        torch.cuda.synchronize()

                    cat_us = {
                        "gemm": 0.0,
                        "attention": 0.0,
                        "norm_pointwise_loss": 0.0,
                    }
                    cat_launches = {
                        "gemm": 0.0,
                        "attention": 0.0,
                        "norm_pointwise_loss": 0.0,
                    }
                    cpu_launch_calls = 0.0
                    kernel_rows: list[dict[str, Any]] = []

                    for evt in prof.key_averages():
                        dev_type_str = str(getattr(evt, "device_type", ""))
                        if dev_type_str == "DeviceType.CPU" and evt.key in (
                            "cudaLaunchKernel",
                            "cuLaunchKernel",
                            "cudaLaunchKernelExC",
                        ):
                            cpu_launch_calls += evt.count / prof_steps
                        elif dev_type_str == "DeviceType.CUDA":
                            us_per_step = (
                                float(getattr(evt, "self_device_time_total", 0.0))
                                / prof_steps
                            )
                            cnt_per_step = float(evt.count) / prof_steps
                            category = classify_cuda_kernel(evt.key)
                            cat_us[category] += us_per_step
                            cat_launches[category] += cnt_per_step
                            kernel_rows.append(
                                {
                                    "kernel": evt.key[:140],
                                    "category": category,
                                    "calls_per_step": round(cnt_per_step, 2),
                                    "cuda_us_per_step": round(us_per_step, 3),
                                }
                            )

                    kernel_rows.sort(
                        key=lambda item: item["cuda_us_per_step"],
                        reverse=True,
                    )
                    total_us = sum(cat_us.values())
                    total_cuda_kernels = sum(cat_launches.values())
                    denom = max(total_us, 1e-9)
                    kernel_records.append(
                        {
                            "experiment": "cuda_kernel_attribution",
                            "clock_policy": f"locked_{boost_sm_mhz}mhz",
                            "implementation": cand["name"],
                            "shape": shape_list,
                            "dtype": str(dtype),
                            "profiled_steps": prof_steps,
                            "total_cuda_kernel_us_per_step": round(total_us, 3),
                            "gemm_us_per_step": round(cat_us["gemm"], 3),
                            "attention_us_per_step": round(cat_us["attention"], 3),
                            "norm_pointwise_loss_us_per_step": round(
                                cat_us["norm_pointwise_loss"],
                                3,
                            ),
                            "gemm_share": round(cat_us["gemm"] / denom, 6),
                            "attention_share": round(cat_us["attention"] / denom, 6),
                            "norm_pointwise_loss_share": round(
                                cat_us["norm_pointwise_loss"] / denom,
                                6,
                            ),
                            "cuda_kernels_per_step": round(total_cuda_kernels, 2),
                            "cpu_launch_calls_per_step": round(cpu_launch_calls, 2),
                            "top_cuda_kernels": kernel_rows[:8],
                        }
                    )

                # Peak incremental memory measured after all timing and profiling
                for cand in candidates:
                    merged = merge_two_passes(
                        pass_a_map[cand["name"]],
                        pass_b_map[cand["name"]],
                    )
                    peak_mem = measure_peak_memory(cand["fn"])
                    mode_records.append(
                        {
                            "experiment": "compiler_mode_frontier",
                            "protocol": "locked_boost_palindromic_balanced",
                            "clock_policy": f"locked_{boost_sm_mhz}mhz",
                            "implementation": cand["name"],
                            "shape": shape_list,
                            "dtype": str(dtype),
                            "setup_seconds": cand["setup_seconds"],
                            "loss": cand["loss"],
                            "loss_abs_error": abs(cand["loss"] - ref_loss),
                            **cand["grad_metrics"],
                            **merged,
                            "peak_incremental_memory_bytes": peak_mem,
                            "tokens_per_second": tokens / (merged["median_ms"] / 1000.0),
                        }
                    )

                del (
                    candidates,
                    kept_models,
                    eager_model,
                    graph_model,
                    cgraph_model,
                    x,
                    ref_grads,
                )
                torch._dynamo.reset()
                torch.cuda.empty_cache()
    finally:
        telemetry.reset_clocks()
        if old_cache_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = old_cache_dir

    return mode_records, kernel_records


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
        ablation_records = run_dvfs_cache_order_ablation(
            profile_name,
            device,
            dtype,
            telemetry,
        )
        mode_records, kernel_records = run_compiler_mode_frontier_and_attribution(
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
        "suite": "transformer_compile_diagnosis",
        "profile": profile_name,
        "environment": env,
        "records": ablation_records + mode_records,
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
