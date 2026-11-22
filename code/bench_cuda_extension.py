from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from common import allclose_metrics, benchmark_cuda, environment_metadata, write_result


def load_extension(source_root: Path):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    started = time.perf_counter()
    extension = load(
        name="cuda_kernel_lab_swiglu",
        sources=[
            str(source_root / "swiglu_ext.cpp"),
            str(source_root / "swiglu_ext.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=True,
    )
    return extension, time.perf_counter() - started


def run(profile: str, source_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    extension, build_seconds = load_extension(source_root)
    torch.manual_seed(19)
    dtype = torch.bfloat16
    sizes = [1 << 20] if profile == "smoke" else [1 << 20, 1 << 23, 1 << 25]
    warmup, groups, iterations = (3, 3, 3) if profile == "smoke" else (20, 9, 20)
    records: list[dict[str, Any]] = []

    for n_elements in sizes:
        a = torch.randn(n_elements, device="cuda", dtype=dtype)
        b = torch.randn_like(a)
        reference = F.silu(a) * b
        actual = extension.swiglu(a, b)
        torch.cuda.synchronize()
        timing = benchmark_cuda(
            lambda: extension.swiglu(a, b),
            warmup=warmup,
            groups=groups,
            iterations_per_group=iterations,
        )
        useful_bytes = n_elements * a.element_size() * 3
        seconds = timing["median_ms"] / 1_000.0
        records.append(
            {
                "experiment": "swiglu",
                "implementation": "native_cuda_bf16x2",
                "shape": [n_elements],
                "dtype": str(dtype),
                "correctness": allclose_metrics(actual, reference),
                "extension_build_seconds": build_seconds,
                **timing,
                "effective_gb_s": useful_bytes / seconds / 1e9,
            }
        )

    return {
        "suite": "cuda_extension",
        "profile": profile,
        "environment": environment_metadata(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile, args.source_root))


if __name__ == "__main__":
    main()

