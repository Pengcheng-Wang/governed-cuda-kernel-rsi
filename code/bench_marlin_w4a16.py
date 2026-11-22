from __future__ import annotations

import argparse
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    check_marlin_supported,
    check_marlin_supports_shape,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

from common import benchmark_cuda, environment_metadata, write_result


QUANT_TYPE = scalar_types.uint4b8


def quality_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    normalized_rmse = (
        error.square().mean().sqrt()
        / expected_f.square().mean().sqrt().clamp_min(1e-8)
    )
    cosine = F.cosine_similarity(
        actual_f.flatten(),
        expected_f.flatten(),
        dim=0,
    )
    return {
        "max_abs_error": error.abs().max().item(),
        "mean_abs_error": error.abs().mean().item(),
        "normalized_rmse": normalized_rmse.item(),
        "cosine_similarity": cosine.item(),
    }


def benchmark_record(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    operation_count: float,
    weight_bytes: int,
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
        "nominal_tflop_s": operation_count / seconds / 1e12,
        "weight_gb_s": weight_bytes / seconds / 1e9,
    }


def profiler_summary(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int = 3,
    top_k: int = 20,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        fn()
    torch.cuda.synchronize()
    events: list[dict[str, Any]] = []
    for event in prof.key_averages():
        self_device_us = float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
        device_us = float(
            getattr(
                event,
                "device_time_total",
                getattr(event, "cuda_time_total", 0.0),
            )
        )
        if self_device_us <= 0.0 and device_us <= 0.0:
            continue
        events.append(
            {
                "name": event.key,
                "count": event.count,
                "self_device_time_us": self_device_us,
                "device_time_us": device_us,
            }
        )
    events.sort(key=lambda item: item["self_device_time_us"], reverse=True)
    return {
        "top_cuda_events": events[:top_k],
        "table": prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=top_k,
        ),
    }


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def run(profile_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if profile_name == "smoke":
        shapes = [
            {
                "name": "smoke_square",
                "k": 1024,
                "n": 1024,
                "m_values": [1, 8, 32],
            }
        ]
        group_sizes = [-1, 128]
        warmup, groups, iterations = 3, 3, 5
    else:
        shapes = [
            {
                "name": "square_4096",
                "k": 4096,
                "n": 4096,
                "m_values": [1, 8, 32, 128, 512, 2048],
            },
            {
                "name": "mlp_up",
                "k": 4096,
                "n": 11008,
                "m_values": [1, 8, 32, 128, 512, 2048],
            },
            {
                "name": "mlp_down",
                "k": 11008,
                "n": 4096,
                "m_values": [1, 8, 32, 128, 512, 2048],
            },
            {
                "name": "vocab_projection",
                "k": 4096,
                "n": 32000,
                "m_values": [1, 8, 32, 128, 512],
            },
        ]
        group_sizes = [-1, 128]
        warmup, groups, iterations = 10, 7, 10

    records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    empty_zp = torch.empty(0, device=device, dtype=torch.int32)

    for shape_index, shape in enumerate(shapes):
        k = shape["k"]
        n = shape["n"]
        support = {
            group_size: {
                "kernel": check_marlin_supported(
                    QUANT_TYPE,
                    group_size,
                    False,
                    80,
                ),
                "shape": check_marlin_supports_shape(
                    n,
                    k,
                    k,
                    group_size,
                ),
            }
            for group_size in group_sizes
        }
        generator = torch.Generator(device=device)
        generator.manual_seed(1_000 + shape_index)
        weight = (
            torch.randn(
                (k, n),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / math.sqrt(k)
        )
        original_weight_bytes = tensor_bytes(weight)

        for group_size in group_sizes:
            if (
                not support[group_size]["kernel"]
                or not support[group_size]["shape"][0]
            ):
                weight_records.append(
                    {
                        "shape": shape["name"],
                        "k": k,
                        "n": n,
                        "group_size": group_size,
                        "supported": False,
                        "support": support[group_size],
                    }
                )
                continue

            quantize_started = time.perf_counter()
            (
                dequant_weight,
                packed_weight,
                scales,
                g_idx,
                sort_indices,
                permutation,
            ) = marlin_quantize(
                weight,
                QUANT_TYPE,
                group_size,
                False,
            )
            torch.cuda.synchronize()
            quantize_seconds = time.perf_counter() - quantize_started
            workspace = marlin_make_workspace_new(device)
            packed_weight_bytes = (
                tensor_bytes(packed_weight)
                + tensor_bytes(scales)
                + tensor_bytes(g_idx)
                + tensor_bytes(sort_indices)
            )
            weight_quality = quality_metrics(
                dequant_weight,
                weight,
            )
            weight_records.append(
                {
                    "shape": shape["name"],
                    "k": k,
                    "n": n,
                    "group_size": group_size,
                    "supported": True,
                    "quantize_seconds": quantize_seconds,
                    "weight_quality": weight_quality,
                    "original_weight_bytes": original_weight_bytes,
                    "packed_weight_and_metadata_bytes": packed_weight_bytes,
                    "compression_ratio": original_weight_bytes
                    / packed_weight_bytes,
                    "packed_weight_shape": list(packed_weight.shape),
                    "scale_shape": list(scales.shape),
                    "workspace_bytes": tensor_bytes(workspace),
                    "permutation_elements": permutation.numel(),
                }
            )

            for m in shape["m_values"]:
                activation_generator = torch.Generator(device=device)
                activation_generator.manual_seed(
                    10_000
                    + shape_index * 1_000
                    + (0 if group_size == -1 else group_size)
                    + m
                )
                activation = torch.randn(
                    (m, k),
                    device=device,
                    dtype=dtype,
                    generator=activation_generator,
                )
                original_reference = torch.mm(
                    activation,
                    weight,
                )
                quantized_reference = torch.mm(
                    activation,
                    dequant_weight,
                )
                marlin_fn = lambda: apply_gptq_marlin_linear(
                    activation,
                    packed_weight,
                    scales,
                    empty_zp,
                    g_idx,
                    sort_indices,
                    workspace,
                    QUANT_TYPE,
                    n,
                    k,
                    True,
                    input_dtype=None,
                )
                output = marlin_fn()
                torch.cuda.synchronize()
                kernel_correctness = quality_metrics(
                    output,
                    quantized_reference,
                )
                quantization_quality = quality_metrics(
                    output,
                    original_reference,
                )
                operation_count = 2.0 * m * n * k
                bf16_record = benchmark_record(
                    "bf16_cublas",
                    lambda: torch.mm(activation, weight),
                    operation_count=operation_count,
                    weight_bytes=original_weight_bytes,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                )
                marlin_record = benchmark_record(
                    "marlin_w4a16",
                    marlin_fn,
                    operation_count=operation_count,
                    weight_bytes=packed_weight_bytes,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                )
                speedup = (
                    bf16_record["median_ms"]
                    / marlin_record["median_ms"]
                )
                for record in (bf16_record, marlin_record):
                    record.update(
                        {
                            "experiment": "w4a16_gemm",
                            "shape": shape["name"],
                            "m": m,
                            "n": n,
                            "k": k,
                            "group_size": group_size,
                            "dtype": str(dtype),
                            "speedup_vs_bf16": (
                                1.0
                                if record["implementation"]
                                == "bf16_cublas"
                                else speedup
                            ),
                            "kernel_correctness_vs_dequantized": (
                                kernel_correctness
                                if record["implementation"]
                                == "marlin_w4a16"
                                else None
                            ),
                            "quantization_quality_vs_original": (
                                quantization_quality
                                if record["implementation"]
                                == "marlin_w4a16"
                                else None
                            ),
                            "original_weight_bytes": original_weight_bytes,
                            "packed_weight_and_metadata_bytes": packed_weight_bytes,
                            "compression_ratio": original_weight_bytes
                            / packed_weight_bytes,
                        }
                    )
                    records.append(record)

                if (
                    profile_name == "full"
                    and shape["name"] == "mlp_up"
                    and group_size == 128
                    and m == 8
                ):
                    profiles.append(
                        {
                            "shape": shape["name"],
                            "m": m,
                            "n": n,
                            "k": k,
                            "group_size": group_size,
                            "implementation": "marlin_w4a16",
                            **profiler_summary(marlin_fn),
                        }
                    )

                del (
                    activation,
                    original_reference,
                    quantized_reference,
                    output,
                )
                torch.cuda.empty_cache()

            del (
                dequant_weight,
                packed_weight,
                scales,
                g_idx,
                sort_indices,
                permutation,
                workspace,
            )
            torch.cuda.empty_cache()

        del weight
        torch.cuda.empty_cache()

    return {
        "suite": "marlin_w4a16",
        "profile": profile_name,
        "environment": {
            **environment_metadata(),
            "vllm": importlib.metadata.version("vllm"),
        },
        "quantization": {
            "type": repr(QUANT_TYPE),
            "bits": QUANT_TYPE.size_bits,
            "has_zero_point": False,
            "activation_order": False,
            "activation_dtype": str(dtype),
        },
        "notes": [
            "Weights are quantized and packed before timing; offline quantization/repack cost is excluded from kernel latency.",
            "BF16 cuBLAS uses the original weight; Marlin output quality is reported against both the original and dequantized quantized weight.",
            "Nominal TFLOP/s counts dense-equivalent 2MNK operations and should not be compared directly to physical INT4 instruction throughput.",
            "Weight GB/s divides the resident weight representation by latency and is most meaningful at decode-like M.",
        ],
        "weight_records": weight_records,
        "records": records,
        "profiler_summaries": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile))


if __name__ == "__main__":
    main()

