from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import (
    allclose_metrics,
    benchmark_cuda,
    compile_with_timing,
    environment_metadata,
    write_result,
)


@triton.jit
def _swiglu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-a))
    tl.store(out_ptr + offsets, a * sigmoid * b, mask=mask)


def triton_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    n_elements = a.numel()
    grid = (triton.cdiv(n_elements, 256),)
    _swiglu_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=256)
    return out


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    x_row_stride,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    variance = tl.sum(x_f32 * x_f32, axis=0) / n_cols
    normalized = x_f32 * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * x_row_stride + offsets, normalized * weight, mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert x.is_contiguous()
    n_cols = x.shape[-1]
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError(f"RMSNorm hidden size {n_cols} is too large for this teaching kernel")
    num_warps = 8 if block_size >= 4096 else 4
    out = torch.empty_like(x)
    rows = x.numel() // n_cols
    _rmsnorm_kernel[(rows,)](
        x,
        weight,
        out,
        x.stride(-2),
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out


def eager_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.silu(a) * b


def eager_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(x.dtype)


def _timed_record(
    name: str,
    fn,
    *,
    useful_bytes: int,
    warmup: int,
    groups: int,
    iterations: int,
) -> dict[str, Any]:
    timing = benchmark_cuda(
        fn,
        warmup=warmup,
        groups=groups,
        iterations_per_group=iterations,
    )
    seconds = timing["median_ms"] / 1_000.0
    return {
        "implementation": name,
        **timing,
        "effective_gb_s": useful_bytes / seconds / 1e9,
    }


def run(profile: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if profile == "smoke":
        swiglu_sizes = [1 << 20]
        rms_shapes = [(256, 4096)]
        warmup, groups, iterations = 5, 3, 5
    else:
        swiglu_sizes = [1 << 20, 1 << 23, 1 << 25]
        rms_shapes = [(2048, 4096), (4096, 8192)]
        warmup, groups, iterations = 20, 9, 20

    records: list[dict[str, Any]] = []

    for n_elements in swiglu_sizes:
        a = torch.randn(n_elements, device=device, dtype=dtype)
        b = torch.randn_like(a)
        reference = eager_swiglu(a, b)

        compile_started = time.perf_counter()
        compiled_swiglu, compile_seconds = compile_with_timing(
            eager_swiglu,
            (a, b),
            mode="default",
        )
        compile_seconds = max(compile_seconds, time.perf_counter() - compile_started)

        implementations = [
            ("torch_eager", lambda: eager_swiglu(a, b)),
            ("torch_compile", lambda: compiled_swiglu(a, b)),
            ("triton", lambda: triton_swiglu(a, b)),
        ]
        useful_bytes = n_elements * a.element_size() * 3
        for name, fn in implementations:
            actual = fn()
            torch.cuda.synchronize()
            records.append(
                {
                    "experiment": "swiglu",
                    "shape": [n_elements],
                    "dtype": str(dtype),
                    "correctness": allclose_metrics(actual, reference),
                    "compile_seconds": compile_seconds if name == "torch_compile" else 0.0,
                    **_timed_record(
                        name,
                        fn,
                        useful_bytes=useful_bytes,
                        warmup=warmup,
                        groups=groups,
                        iterations=iterations,
                    ),
                }
            )
        del a, b, reference
        torch.cuda.empty_cache()

    for rows, hidden in rms_shapes:
        x = torch.randn(rows, hidden, device=device, dtype=dtype)
        weight = torch.randn(hidden, device=device, dtype=dtype)
        reference = eager_rmsnorm(x, weight)
        compiled_rmsnorm, compile_seconds = compile_with_timing(
            eager_rmsnorm,
            (x, weight),
            mode="default",
        )
        implementations = [
            ("torch_eager", lambda: eager_rmsnorm(x, weight)),
            ("torch_compile", lambda: compiled_rmsnorm(x, weight)),
            ("triton", lambda: triton_rmsnorm(x, weight)),
        ]
        useful_bytes = (x.numel() * 2 + weight.numel()) * x.element_size()
        for name, fn in implementations:
            actual = fn()
            torch.cuda.synchronize()
            records.append(
                {
                    "experiment": "rmsnorm",
                    "shape": [rows, hidden],
                    "dtype": str(dtype),
                    "correctness": allclose_metrics(actual, reference),
                    "compile_seconds": compile_seconds if name == "torch_compile" else 0.0,
                    **_timed_record(
                        name,
                        fn,
                        useful_bytes=useful_bytes,
                        warmup=warmup,
                        groups=groups,
                        iterations=iterations,
                    ),
                }
            )
        del x, weight, reference
        torch.cuda.empty_cache()

    return {
        "suite": "fused_ops",
        "profile": profile,
        "environment": environment_metadata(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile))


if __name__ == "__main__":
    main()

