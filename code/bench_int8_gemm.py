from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from common import allclose_metrics, benchmark_cuda, environment_metadata, write_result


def quantize_symmetric(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_scale = a.abs().max().clamp_min(1e-8) / 127.0
    b_scale = b.abs().amax(dim=0).clamp_min(1e-8) / 127.0
    a_q = torch.round(a / a_scale).clamp(-127, 127).to(torch.int8)
    b_q = torch.round(b / b_scale).clamp(-127, 127).to(torch.int8)
    output_scale = a_scale * b_scale
    return a_q.contiguous(), b_q.contiguous(), output_scale


def quantization_quality(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    error = actual_f - reference_f
    normalized_rmse = (
        error.square().mean().sqrt() / reference_f.square().mean().sqrt().clamp_min(1e-8)
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(),
        reference_f.flatten(),
        dim=0,
    )
    return {
        **allclose_metrics(actual, reference),
        "normalized_rmse": normalized_rmse.item(),
        "cosine_similarity": cosine.item(),
    }


def run(profile: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("This PyTorch build does not expose torch._int_mm")
    torch.manual_seed(29)
    device = torch.device("cuda")
    if profile == "smoke":
        shapes = [(256, 1024, 1024)]
        warmup, groups, iterations = 3, 3, 3
    else:
        shapes = [
            (32, 4096, 4096),
            (128, 4096, 4096),
            (512, 4096, 4096),
            (4096, 4096, 4096),
        ]
        warmup, groups, iterations = 10, 7, 10

    records: list[dict[str, Any]] = []
    for m, n, k in shapes:
        a_fp32 = torch.randn((m, k), device=device, dtype=torch.float32)
        b_fp32 = torch.randn((k, n), device=device, dtype=torch.float32)
        a_bf16 = a_fp32.to(torch.bfloat16)
        b_bf16 = b_fp32.to(torch.bfloat16)
        a_q, b_q, output_scale = quantize_symmetric(a_fp32, b_fp32)
        reference = torch.mm(a_bf16, b_bf16)

        def bf16_mm() -> torch.Tensor:
            return torch.mm(a_bf16, b_bf16)

        def int8_mm() -> torch.Tensor:
            return torch._int_mm(a_q, b_q)

        def int8_dequant() -> torch.Tensor:
            return (torch._int_mm(a_q, b_q).float() * output_scale).to(torch.bfloat16)

        implementations = [
            ("bf16_cublas", bf16_mm, "tflop_s"),
            ("int8_int32_kernel", int8_mm, "tops"),
            ("int8_with_unfused_dequant", int8_dequant, "tops"),
        ]
        operation_count = 2.0 * m * n * k
        dequantized = int8_dequant()
        quality = quantization_quality(dequantized, reference)
        for name, fn, rate_name in implementations:
            timing = benchmark_cuda(
                fn,
                warmup=warmup,
                groups=groups,
                iterations_per_group=iterations,
            )
            seconds = timing["median_ms"] / 1_000.0
            record: dict[str, Any] = {
                "experiment": "symmetric_w8a8_gemm",
                "implementation": name,
                "shape_mnk": [m, n, k],
                "quantization_quality": quality if name.startswith("int8") else None,
                **timing,
                rate_name: operation_count / seconds / 1e12,
            }
            records.append(record)

        del a_fp32, b_fp32, a_bf16, b_bf16, a_q, b_q, output_scale, reference
        torch.cuda.empty_cache()

    return {
        "suite": "int8_gemm",
        "profile": profile,
        "environment": environment_metadata(),
        "notes": [
            "INT8 tensors are prequantized; quantization cost is excluded.",
            "torch._int_mm in this build requires M > 16, so the smallest case uses M=32.",
            "The unfused dequant path materializes INT32 and FP32 intermediates.",
            "Production W8A8 kernels normally fuse scaling and output conversion.",
        ],
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
