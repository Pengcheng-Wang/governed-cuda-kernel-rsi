from __future__ import annotations

import argparse
import importlib.metadata
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    moe_align_block_size,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)

from common import (
    benchmark_cuda,
    environment_metadata,
    write_result,
)


def reference_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del hidden_states
    scores = torch.softmax(gating_output.float(), dim=-1)
    weights, ids = torch.topk(scores, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    assignment_indices = torch.arange(
        ids.numel(),
        device=ids.device,
        dtype=torch.int32,
    ).reshape_as(ids)
    return weights, ids.to(torch.int32), assignment_indices


def topk_correctness(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float | bool]:
    actual_weights, actual_ids, _ = actual
    expected_weights, expected_ids, _ = expected
    return {
        "ids_equal": bool(torch.equal(actual_ids, expected_ids)),
        "max_weight_abs_error": (
            actual_weights.float() - expected_weights.float()
        ).abs().max().item(),
        "mean_weight_abs_error": (
            actual_weights.float() - expected_weights.float()
        ).abs().mean().item(),
    }


def output_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    return {
        "max_abs_error": error.abs().max().item(),
        "mean_abs_error": error.abs().mean().item(),
        "normalized_rmse": (
            error.square().mean().sqrt()
            / expected_f.square().mean().sqrt().clamp_min(1e-8)
        ).item(),
        "cosine_similarity": F.cosine_similarity(
            actual_f.flatten(),
            expected_f.flatten(),
            dim=0,
        ).item(),
    }


def timed(
    fn: Callable[[], Any],
    *,
    warmup: int,
    groups: int,
    iterations: int,
) -> dict[str, float]:
    return benchmark_cuda(
        fn,
        warmup=warmup,
        groups=groups,
        iterations_per_group=iterations,
    )


def profiler_summary(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    top_n: int = 30,
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
        "top_cuda_events": events[:top_n],
        "table": prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=top_n,
        ),
    }


def run(profile_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(97)
    torch.cuda.manual_seed_all(97)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden = 1024
    intermediate = 2816
    top_k = 2
    if profile_name == "smoke":
        expert_values = [8]
        token_values = [32, 128]
        warmup, groups, iterations = 1, 2, 2
    else:
        expert_values = [8, 32]
        token_values = [32, 128, 512, 2048, 8192]
        warmup, groups, iterations = 5, 5, 10

    records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for num_experts in expert_values:
        generator = torch.Generator(device=device)
        generator.manual_seed(6_000 + num_experts)
        gate_weight = (
            torch.randn(
                (num_experts, hidden),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / math.sqrt(hidden)
        )
        w1 = (
            torch.randn(
                (num_experts, 2 * intermediate, hidden),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / math.sqrt(hidden)
        )
        w2 = (
            torch.randn(
                (num_experts, hidden, intermediate),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / math.sqrt(intermediate)
        )

        for num_tokens in token_values:
            hidden_states = torch.randn(
                (num_tokens, hidden),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            tie_breaker = (
                torch.arange(
                    num_experts,
                    device=device,
                    dtype=torch.float32,
                )
                * 1e-6
            )

            def router_projection() -> torch.Tensor:
                return (
                    F.linear(hidden_states, gate_weight).float()
                    + tie_breaker
                )

            gating_output = router_projection()
            torch.cuda.synchronize()

            reference = reference_topk(
                hidden_states,
                gating_output,
                top_k,
            )
            fused = fused_topk(
                hidden_states,
                gating_output,
                top_k,
                True,
            )
            correctness = topk_correctness(fused, reference)
            if not correctness["ids_equal"]:
                raise RuntimeError(
                    f"Fused top-k ids differ for E={num_experts}, "
                    f"T={num_tokens}"
                )

            topk_weights, topk_ids, _ = fused
            config = try_get_optimal_moe_config(
                tuple(w1.shape),
                tuple(w2.shape),
                top_k,
                "bfloat16",
                num_tokens,
            )
            block_size_m = config["BLOCK_SIZE_M"]

            router_projection_fn = router_projection
            torch_topk_fn = lambda: reference_topk(
                hidden_states,
                gating_output,
                top_k,
            )
            fused_topk_fn = lambda: fused_topk(
                hidden_states,
                gating_output,
                top_k,
                True,
            )
            align_fn = lambda: moe_align_block_size(
                topk_ids,
                block_size_m,
                num_experts,
            )
            experts_fn = lambda: fused_experts(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
            )

            def full_fused_router() -> torch.Tensor:
                logits = router_projection()
                weights, ids, _ = fused_topk(
                    hidden_states,
                    logits,
                    top_k,
                    True,
                )
                return fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    weights,
                    ids,
                )

            def full_torch_router() -> torch.Tensor:
                logits = router_projection()
                weights, ids, _ = reference_topk(
                    hidden_states,
                    logits,
                    top_k,
                )
                return fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    weights,
                    ids,
                )

            expert_output = experts_fn()
            fused_full_output = full_fused_router()
            torch_full_output = full_torch_router()
            torch.cuda.synchronize()
            full_fused_quality = output_quality(
                fused_full_output,
                expert_output,
            )
            full_torch_quality = output_quality(
                torch_full_output,
                expert_output,
            )

            timings = {
                "router_projection": timed(
                    router_projection_fn,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                ),
                "torch_topk": timed(
                    torch_topk_fn,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                ),
                "vllm_fused_topk": timed(
                    fused_topk_fn,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                ),
                "moe_align_block_size": timed(
                    align_fn,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                ),
                "fused_experts_precomputed": timed(
                    experts_fn,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                ),
                "full_vllm_router_experts": timed(
                    full_fused_router,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                ),
                "full_torch_router_experts": timed(
                    full_torch_router,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                ),
            }
            expert_ms = timings["fused_experts_precomputed"]["median_ms"]
            fused_full_ms = timings["full_vllm_router_experts"]["median_ms"]
            torch_full_ms = timings["full_torch_router_experts"]["median_ms"]
            records.append(
                {
                    "num_tokens": num_tokens,
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "num_experts": num_experts,
                    "top_k": top_k,
                    "dtype": str(dtype),
                    "block_size_m": block_size_m,
                    "config": config,
                    "topk_correctness": correctness,
                    "full_fused_quality_vs_precomputed": (
                        full_fused_quality
                    ),
                    "full_torch_quality_vs_precomputed": (
                        full_torch_quality
                    ),
                    "timings": timings,
                    "vllm_topk_speedup_vs_torch": (
                        timings["torch_topk"]["median_ms"]
                        / timings["vllm_fused_topk"]["median_ms"]
                    ),
                    "fused_router_overhead_fraction": (
                        (fused_full_ms - expert_ms) / fused_full_ms
                    ),
                    "torch_router_overhead_fraction": (
                        (torch_full_ms - expert_ms) / torch_full_ms
                    ),
                    "full_vllm_speedup_vs_torch_router": (
                        torch_full_ms / fused_full_ms
                    ),
                }
            )

            if (
                profile_name == "full"
                and num_experts == 32
                and num_tokens == 512
            ):
                profiles.append(
                    {
                        "num_experts": num_experts,
                        "num_tokens": num_tokens,
                        "top_k": top_k,
                        **profiler_summary(full_fused_router),
                    }
                )

            del (
                hidden_states,
                gating_output,
                reference,
                fused,
                topk_weights,
                topk_ids,
                expert_output,
                fused_full_output,
                torch_full_output,
            )
            torch.cuda.empty_cache()

        del gate_weight, w1, w2
        torch.cuda.empty_cache()

    return {
        "suite": "moe_router",
        "profile": profile_name,
        "environment": {
            **environment_metadata(),
            "vllm": importlib.metadata.version("vllm"),
        },
        "notes": [
            "Router projection uses BF16 hidden states and BF16 gate weights; fused_topk internally computes softmax/top-k and returns FP32 weights plus INT32 expert IDs.",
            "A deterministic FP32 expert-index epsilon of 1e-6 is added after BF16 projection so exact ties have the same ordering across PyTorch and vLLM backends.",
            "The PyTorch reference computes FP32 softmax, torch.topk, and top-k renormalization.",
            "moe_align_block_size measures vLLM's production expert-assignment sorting and padding preparation separately.",
            "Fused expert timing includes its own internal alignment; the separate alignment timing is explanatory rather than additive.",
            "Full router overhead is derived from independently timed full and precomputed-expert paths, so small negative fractions can occur within timing variance.",
        ],
        "records": records,
        "profiler_summaries": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="smoke",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile))


if __name__ == "__main__":
    main()
