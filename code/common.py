from __future__ import annotations

import json
import os
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch


def environment_metadata() -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0)
    try:
        driver_version = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        driver_version = None
    metadata: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_driver": driver_version,
        "gpu_name": gpu.name,
        "gpu_count": torch.cuda.device_count(),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_bytes": gpu.total_memory,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    try:
        import triton

        metadata["triton"] = triton.__version__
    except ImportError:
        metadata["triton"] = None
    try:
        import flash_attn

        metadata["flash_attn"] = flash_attn.__version__
    except ImportError:
        metadata["flash_attn"] = None
    return metadata


def benchmark_cuda(
    fn: Callable[[], Any],
    *,
    warmup: int = 20,
    groups: int = 9,
    iterations_per_group: int = 20,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations_per_group):
            fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / iterations_per_group)

    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "groups": groups,
        "iterations_per_group": iterations_per_group,
        "warmup": warmup,
    }


def measure_peak_memory(fn: Callable[[], Any]) -> int:
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return max(0, torch.cuda.max_memory_allocated() - baseline)


def compile_with_timing(
    fn: Callable[..., Any],
    sample_args: tuple[Any, ...],
    *,
    mode: str = "default",
) -> tuple[Callable[..., Any], float]:
    started = time.perf_counter()
    compiled = torch.compile(fn, fullgraph=True, mode=mode)
    compiled(*sample_args)
    torch.cuda.synchronize()
    return compiled, time.perf_counter() - started


def write_result(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(output)


def allclose_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_error = (actual_f - expected_f).abs()
    denominator = expected_f.abs().clamp_min(1e-6)
    rel_error = abs_error / denominator
    return {
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
        "max_rel_error": rel_error.max().item(),
        "mean_rel_error": rel_error.mean().item(),
    }
