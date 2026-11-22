from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import (
    allclose_metrics,
    benchmark_cuda,
    compile_with_timing,
    environment_metadata,
    measure_peak_memory,
    write_result,
)


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: SDPBackend,
    causal: bool,
) -> torch.Tensor:
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _flash_attn(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    return flash_attn_func(q_bshd, k_bshd, v_bshd, causal=causal)


def _benchmark_implementation(
    name: str,
    fn: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    *,
    flop_count: float,
    tokens: int,
    warmup: int,
    groups: int,
    iterations: int,
    setup_seconds: float = 0.0,
) -> dict[str, Any]:
    actual = fn()
    torch.cuda.synchronize()
    correctness = allclose_metrics(actual, reference)
    timing = benchmark_cuda(
        fn,
        warmup=warmup,
        groups=groups,
        iterations_per_group=iterations,
    )
    peak_memory = measure_peak_memory(fn)
    seconds = timing["median_ms"] / 1_000.0
    return {
        "implementation": name,
        "correctness": correctness,
        "setup_seconds": setup_seconds,
        **timing,
        "peak_incremental_memory_bytes": peak_memory,
        "attention_tflop_s": flop_count / seconds / 1e12,
        "tokens_per_second": tokens / seconds,
    }


def run(profile: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    causal = True
    if profile == "smoke":
        shapes = [(1, 256, 8, 64)]
        warmup, groups, iterations = 3, 3, 3
    else:
        shapes = [
            (4, 512, 16, 128),
            (2, 2048, 16, 128),
            (1, 8192, 16, 128),
        ]
        warmup, groups, iterations = 10, 7, 10

    records: list[dict[str, Any]] = []
    for batch, sequence, heads, head_dim in shapes:
        q_bshd = torch.randn(
            (batch, sequence, heads, head_dim),
            device=device,
            dtype=dtype,
        )
        k_bshd = torch.randn_like(q_bshd)
        v_bshd = torch.randn_like(q_bshd)
        q = q_bshd.transpose(1, 2)
        k = k_bshd.transpose(1, 2)
        v = v_bshd.transpose(1, 2)

        if sequence <= 2048:
            reference = _sdpa(q, k, v, SDPBackend.MATH, causal)
        else:
            reference = _sdpa(q, k, v, SDPBackend.FLASH_ATTENTION, causal)

        implementations: list[tuple[str, Callable[[], torch.Tensor], float]] = []
        if sequence <= 2048:
            implementations.append(
                (
                    "torch_sdpa_math",
                    lambda: _sdpa(q, k, v, SDPBackend.MATH, causal),
                    0.0,
                )
            )
        implementations.append(
            (
                "torch_sdpa_flash",
                lambda: _sdpa(q, k, v, SDPBackend.FLASH_ATTENTION, causal),
                0.0,
            )
        )
        implementations.append(
            (
                "flash_attn",
                lambda: _flash_attn(q_bshd, k_bshd, v_bshd, causal).transpose(1, 2),
                0.0,
            )
        )

        def flash_sdpa_no_context(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            return F.scaled_dot_product_attention(
                q_in,
                k_in,
                v_in,
                is_causal=causal,
            )

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            compiled_sdpa, compile_seconds = compile_with_timing(
                flash_sdpa_no_context,
                (q, k, v),
                mode="default",
            )
        implementations.append(
            (
                "torch_compile_sdpa",
                lambda: compiled_sdpa(q, k, v),
                compile_seconds,
            )
        )

        causal_fraction = 0.5 if causal else 1.0
        flop_count = (
            4.0
            * batch
            * heads
            * sequence
            * sequence
            * head_dim
            * causal_fraction
        )
        tokens = batch * sequence
        for name, fn, setup_seconds in implementations:
            record = _benchmark_implementation(
                name,
                fn,
                reference,
                flop_count=flop_count,
                tokens=tokens,
                warmup=warmup,
                groups=groups,
                iterations=iterations,
                setup_seconds=setup_seconds,
            )
            record.update(
                {
                    "experiment": "causal_attention_forward",
                    "shape_bshd": [batch, sequence, heads, head_dim],
                    "dtype": str(dtype),
                }
            )
            records.append(record)

        del q_bshd, k_bshd, v_bshd, q, k, v, reference
        torch.cuda.empty_cache()

    return {
        "suite": "attention",
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
