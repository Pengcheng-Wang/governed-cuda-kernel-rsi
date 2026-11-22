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

from common import (
    benchmark_cuda,
    environment_metadata,
    measure_peak_memory,
    write_result,
)


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


def make_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    pattern: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | list[int]]]:
    tokens = torch.arange(num_tokens, device=device, dtype=torch.int64)
    ids = torch.empty(
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
    )
    if pattern == "uniform":
        for slot in range(top_k):
            ids[:, slot] = ((tokens * top_k + slot) % num_experts).to(
                torch.int32
            )
    elif pattern == "hot80":
        hot_tokens = int(round(0.8 * num_tokens))
        ids[:hot_tokens, 0] = 0
        if hot_tokens < num_tokens:
            ids[hot_tokens:, 0] = (
                1
                + (
                    torch.arange(
                        num_tokens - hot_tokens,
                        device=device,
                    )
                    % (num_experts - 1)
                )
            ).to(torch.int32)
        for slot in range(1, top_k):
            candidates = (
                1
                + (
                    tokens
                    + slot
                    - 1
                )
                % (num_experts - 1)
            ).to(torch.int32)
            ids[:, slot] = candidates
            duplicate = ids[:, slot] == ids[:, 0]
            ids[duplicate, slot] = (
                (ids[duplicate, slot].to(torch.int64) % num_experts) + 1
            ).remainder(num_experts).to(torch.int32)
    else:
        raise ValueError(f"Unknown routing pattern: {pattern}")

    logits = torch.randn(
        (num_tokens, top_k),
        device=device,
        dtype=torch.float32,
    )
    weights = torch.softmax(logits, dim=-1).to(dtype)
    counts = torch.bincount(
        ids.to(torch.int64).flatten(),
        minlength=num_experts,
    )
    counts_f = counts.float()
    mean = counts_f.mean().item()
    std = counts_f.std(unbiased=False).item()
    metadata: dict[str, float | int | list[int]] = {
        "counts": counts.tolist(),
        "active_experts": int((counts > 0).sum().item()),
        "mean_assignments_per_expert": mean,
        "max_assignments": int(counts.max().item()),
        "min_assignments": int(counts.min().item()),
        "coefficient_of_variation": std / mean if mean > 0 else 0.0,
        "max_to_mean": counts.max().item() / mean if mean > 0 else 0.0,
    }
    return ids, weights, metadata


def expert_batched_loop(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_experts = w1.shape[0]
    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    assignment_indices = torch.arange(
        num_tokens * top_k,
        device=hidden_states.device,
        dtype=torch.int64,
    )
    token_indices = assignment_indices // top_k
    output = torch.zeros(
        (num_tokens, hidden),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    for expert in range(num_experts):
        expert_assignments = torch.nonzero(
            flat_ids == expert,
            as_tuple=False,
        ).squeeze(-1)
        if expert_assignments.numel() == 0:
            continue
        expert_tokens = token_indices.index_select(
            0,
            expert_assignments,
        )
        expert_input = hidden_states.index_select(
            0,
            expert_tokens,
        )
        gate_up = F.linear(expert_input, w1[expert])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(
            F.silu(gate) * up,
            w2[expert],
        )
        expert_weights = flat_weights.index_select(
            0,
            expert_assignments,
        ).unsqueeze(-1)
        output.index_add_(
            0,
            expert_tokens,
            expert_output * expert_weights,
        )
    return output


def benchmark_record(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    useful_flops: float,
    num_tokens: int,
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
    peak_memory = measure_peak_memory(fn)
    seconds = timing["median_ms"] / 1_000.0
    return {
        "implementation": name,
        **timing,
        "useful_tflop_s": useful_flops / seconds / 1e12,
        "tokens_per_second": num_tokens / seconds,
        "peak_incremental_memory_bytes": peak_memory,
    }


def profiler_summary(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int = 3,
    top_k: int = 30,
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


def run(profile_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(53)
    torch.cuda.manual_seed_all(53)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if profile_name == "smoke":
        model_shapes = [
            {
                "name": "smoke",
                "hidden": 256,
                "intermediate": 512,
                "experts": [4, 8],
                "tokens": [32],
                "top_k": [1, 2],
            }
        ]
        patterns = ["uniform", "hot80"]
        warmup, groups, iterations = 1, 2, 2
    else:
        model_shapes = [
            {
                "name": "slm",
                "hidden": 1024,
                "intermediate": 2816,
                "experts": [8, 32],
                "tokens": [32, 128, 512, 2048],
                "top_k": [1, 2],
            },
            {
                "name": "llm",
                "hidden": 4096,
                "intermediate": 11008,
                "experts": [8],
                "tokens": [32, 128, 512],
                "top_k": [2],
            },
        ]
        patterns = ["uniform", "hot80"]
        warmup, groups, iterations = 5, 5, 5

    records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(model_shapes):
        hidden = shape["hidden"]
        intermediate = shape["intermediate"]
        for num_experts in shape["experts"]:
            generator = torch.Generator(device=device)
            generator.manual_seed(2_000 + shape_index * 100 + num_experts)
            w1 = (
                torch.randn(
                    (
                        num_experts,
                        2 * intermediate,
                        hidden,
                    ),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                / math.sqrt(hidden)
            )
            w2 = (
                torch.randn(
                    (
                        num_experts,
                        hidden,
                        intermediate,
                    ),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                / math.sqrt(intermediate)
            )
            weight_bytes = (
                w1.numel() * w1.element_size()
                + w2.numel() * w2.element_size()
            )
            for num_tokens in shape["tokens"]:
                hidden_states = torch.randn(
                    (num_tokens, hidden),
                    device=device,
                    dtype=dtype,
                )
                for top_k in shape["top_k"]:
                    config = try_get_optimal_moe_config(
                        tuple(w1.shape),
                        tuple(w2.shape),
                        top_k,
                        "bfloat16",
                        num_tokens,
                    )
                    block_size_m = config["BLOCK_SIZE_M"]
                    for pattern in patterns:
                        topk_ids, topk_weights, routing = make_routing(
                            num_tokens,
                            num_experts,
                            top_k,
                            pattern,
                            device=device,
                            dtype=dtype,
                        )
                        _, _, padded_count = moe_align_block_size(
                            topk_ids,
                            block_size_m,
                            num_experts,
                        )
                        padded_assignments = int(
                            padded_count.item()
                        )
                        assignments = num_tokens * top_k
                        reference_fn = lambda: expert_batched_loop(
                            hidden_states,
                            w1,
                            w2,
                            topk_weights,
                            topk_ids,
                        )
                        fused_fn = lambda: fused_experts(
                            hidden_states,
                            w1,
                            w2,
                            topk_weights,
                            topk_ids,
                        )
                        reference = reference_fn()
                        fused_output = fused_fn()
                        torch.cuda.synchronize()
                        correctness = quality_metrics(
                            fused_output,
                            reference,
                        )
                        useful_flops = (
                            6.0
                            * assignments
                            * hidden
                            * intermediate
                        )
                        reference_record = benchmark_record(
                            "expert_batched_loop",
                            reference_fn,
                            useful_flops=useful_flops,
                            num_tokens=num_tokens,
                            warmup=warmup,
                            groups=groups,
                            iterations=iterations,
                        )
                        fused_record = benchmark_record(
                            "vllm_fused_experts",
                            fused_fn,
                            useful_flops=useful_flops,
                            num_tokens=num_tokens,
                            warmup=warmup,
                            groups=groups,
                            iterations=iterations,
                        )
                        speedup = (
                            reference_record["median_ms"]
                            / fused_record["median_ms"]
                        )
                        common = {
                            "experiment": "local_fused_moe",
                            "model_shape": shape["name"],
                            "num_tokens": num_tokens,
                            "hidden": hidden,
                            "intermediate": intermediate,
                            "num_experts": num_experts,
                            "top_k": top_k,
                            "routing_pattern": pattern,
                            "routing": routing,
                            "config": config,
                            "block_size_m": block_size_m,
                            "assignments": assignments,
                            "padded_assignments": padded_assignments,
                            "padding_ratio": padded_assignments
                            / assignments,
                            "useful_flops": useful_flops,
                            "weight_bytes": weight_bytes,
                            "dtype": str(dtype),
                            "fused_correctness": correctness,
                        }
                        for record in (
                            reference_record,
                            fused_record,
                        ):
                            record.update(common)
                            record["speedup_vs_loop"] = (
                                1.0
                                if record["implementation"]
                                == "expert_batched_loop"
                                else speedup
                            )
                            records.append(record)

                        if (
                            profile_name == "full"
                            and shape["name"] == "slm"
                            and num_experts == 32
                            and num_tokens == 512
                            and top_k == 2
                            and pattern == "hot80"
                        ):
                            profiles.append(
                                {
                                    "model_shape": shape["name"],
                                    "num_tokens": num_tokens,
                                    "num_experts": num_experts,
                                    "top_k": top_k,
                                    "routing_pattern": pattern,
                                    **profiler_summary(fused_fn),
                                }
                            )

                        del (
                            topk_ids,
                            topk_weights,
                            reference,
                            fused_output,
                        )
                        torch.cuda.empty_cache()
                del hidden_states
                torch.cuda.empty_cache()
            del w1, w2
            torch.cuda.empty_cache()

    return {
        "suite": "fused_moe",
        "profile": profile_name,
        "environment": {
            **environment_metadata(),
            "vllm": importlib.metadata.version("vllm"),
        },
        "notes": [
            "The expert_batched_loop groups assignments by expert but issues separate PyTorch GEMMs and index operations per active expert.",
            "Useful FLOPs count two dense SwiGLU expert linears per token-expert assignment: 6*H*I.",
            "Padding ratio is the actual token-assignment padding returned by vLLM moe_align_block_size for the selected BLOCK_SIZE_M.",
            "Routing and top-k selection are precomputed outside the timed expert execution.",
        ],
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

