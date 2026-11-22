from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from common import allclose_metrics, benchmark_cuda, environment_metadata, write_result


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_start * BLOCK_K + offs_k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Teaching kernel expects contiguous row-major inputs")
    m, k = a.shape
    k_b, n = b.shape
    if k != k_b:
        raise ValueError("Incompatible matrix shapes")
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def run(profile: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    if profile == "smoke":
        shapes = [(256, 1024, 1024)]
        warmup, groups, iterations = 3, 3, 3
    else:
        shapes = [
            (512, 4096, 4096),
            (4096, 4096, 4096),
            (4096, 11008, 4096),
            (8192, 4096, 11008),
        ]
        warmup, groups, iterations = 10, 7, 10

    records: list[dict[str, Any]] = []
    for m, n, k in shapes:
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((k, n), device=device, dtype=dtype)
        reference = torch.mm(a, b)

        autotune_started = time.perf_counter()
        triton_output = triton_matmul(a, b)
        torch.cuda.synchronize()
        autotune_seconds = time.perf_counter() - autotune_started

        implementations = [
            ("torch_cublas", lambda: torch.mm(a, b), 0.0),
            ("triton_autotuned", lambda: triton_matmul(a, b), autotune_seconds),
        ]
        flop_count = 2.0 * m * n * k
        for name, fn, setup_seconds in implementations:
            actual = fn()
            torch.cuda.synchronize()
            timing = benchmark_cuda(
                fn,
                warmup=warmup,
                groups=groups,
                iterations_per_group=iterations,
            )
            seconds = timing["median_ms"] / 1_000.0
            records.append(
                {
                    "experiment": "bf16_matmul",
                    "implementation": name,
                    "shape_mnk": [m, n, k],
                    "dtype": str(dtype),
                    "correctness": allclose_metrics(actual, reference),
                    "autotune_or_setup_seconds": setup_seconds,
                    **timing,
                    "tflop_s": flop_count / seconds / 1e12,
                }
            )
        del a, b, reference, triton_output
        torch.cuda.empty_cache()

    return {
        "suite": "matmul",
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

