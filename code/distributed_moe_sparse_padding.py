from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    moe_align_block_size,
    try_get_optimal_moe_config,
)

from distributed_moe_capacity_frontier import (
    gather_segment_counts,
)
from distributed_moe_end_to_end import (
    TensorDict,
    critical_path_summary,
    gather_scalar,
    output_quality,
    route_and_pack,
    routing_bias,
    timed_schedule,
)
from distributed_moe_packed_dispatch import timed_crossover
from distributed_moe_padded_capacity import (
    apply_host_capacity,
    build_padded_route,
    dispatch_padded,
    masked_global_reference,
    return_padded,
    run_padded_route,
    run_variable_route,
)


@triton.jit
def _zero_invalid_rows_kernel(
    output_ptr,
    expert_ids_ptr,
    output_row_stride,
    expert_id_row_stride,
    hidden: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    invalid = (
        tl.load(
            expert_ids_ptr + row * expert_id_row_stride
        )
        < 0
    )
    tl.store(
        output_ptr + row * output_row_stride + offsets,
        0.0,
        mask=(offsets < hidden) & invalid,
    )


def zero_invalid_rows(
    output: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    hidden = output.shape[1]
    block_size = triton.next_power_of_2(hidden)
    if block_size > 65536:
        raise ValueError(
            f"Hidden size {hidden} is too large for row cleanup"
        )
    _zero_invalid_rows_kernel[(output.shape[0],)](
        output,
        expert_ids,
        output.stride(0),
        expert_ids.stride(0),
        hidden=hidden,
        block_size=block_size,
        num_warps=4 if block_size <= 2048 else 8,
    )
    return output


def profile_configurations(
    profile: str,
) -> tuple[
    int,
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    int,
    int,
]:
    if profile == "smoke":
        return (
            128,
            (
                ("balanced", 0.0),
                ("global_bias1", 1.0),
            ),
            (("threshold_exact", 1.0),),
            1,
            2,
        )
    return (
        512,
        (
            ("balanced", 0.0),
            ("global_bias1", 1.0),
            ("global_bias4", 4.0),
        ),
        (
            ("threshold_exact", 1.0),
            ("threshold_1125pct", 1.125),
        ),
        1,
        4 if profile == "full" else 2,
    )


def build_sparse_padded_route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    *,
    top_k: int,
    experts_per_rank: int,
    world_size: int,
    capacity: int,
) -> TensorDict:
    route = build_padded_route(
        hidden_states,
        gate_weight,
        bias,
        tie_breaker,
        top_k=top_k,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
        capacity=capacity,
    )
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    assert isinstance(send_local_ids, torch.Tensor)
    assert isinstance(send_weights, torch.Tensor)
    route["send_local_ids"] = send_local_ids.masked_fill(
        send_weights == 0,
        -1,
    )
    return route


def mapped_sparse_local_experts(
    dispatched: TensorDict,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    expert_map: torch.Tensor,
) -> torch.Tensor:
    recv_hidden = dispatched["recv_hidden"]
    recv_local_ids = dispatched["recv_local_ids"]
    recv_weights = dispatched["recv_weights"]
    assert isinstance(recv_hidden, torch.Tensor)
    assert isinstance(recv_local_ids, torch.Tensor)
    assert isinstance(recv_weights, torch.Tensor)
    return fused_experts(
        recv_hidden,
        local_w1,
        local_w2,
        recv_weights,
        recv_local_ids,
        global_num_experts=local_w1.shape[0],
        expert_map=expert_map,
    )


def masked_sparse_local_experts(
    dispatched: TensorDict,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
) -> torch.Tensor:
    recv_hidden = dispatched["recv_hidden"]
    recv_local_ids = dispatched["recv_local_ids"]
    recv_weights = dispatched["recv_weights"]
    assert isinstance(recv_hidden, torch.Tensor)
    assert isinstance(recv_local_ids, torch.Tensor)
    assert isinstance(recv_weights, torch.Tensor)
    output = fused_experts(
        recv_hidden,
        local_w1,
        local_w2,
        recv_weights,
        recv_local_ids,
    )
    return zero_invalid_rows(
        output,
        recv_local_ids,
    )


def run_mapped_sparse_padded_route(
    route: TensorDict,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    expert_map: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    dispatched = dispatch_padded(route)
    expert_output = mapped_sparse_local_experts(
        dispatched,
        local_w1,
        local_w2,
        expert_map,
    )
    return return_padded(
        expert_output,
        route,
        num_tokens=num_tokens,
    )


def run_masked_sparse_padded_route(
    route: TensorDict,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    dispatched = dispatch_padded(route)
    expert_output = masked_sparse_local_experts(
        dispatched,
        local_w1,
        local_w2,
    )
    return return_padded(
        expert_output,
        route,
        num_tokens=num_tokens,
    )


def alignment_metadata(
    dispatched: TensorDict,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> tuple[int, int]:
    recv_hidden = dispatched["recv_hidden"]
    recv_local_ids = dispatched["recv_local_ids"]
    assert isinstance(recv_hidden, torch.Tensor)
    assert isinstance(recv_local_ids, torch.Tensor)
    config = try_get_optimal_moe_config(
        tuple(local_w1.shape),
        tuple(local_w2.shape),
        1,
        "bfloat16",
        recv_hidden.shape[0],
    )
    block_size_m = config["BLOCK_SIZE_M"]
    _, _, aligned_rows = moe_align_block_size(
        recv_local_ids,
        block_size_m,
        local_w1.shape[0],
        expert_map,
        ignore_invalid_experts=True,
    )
    return block_size_m, int(aligned_rows.item())


def gather_int(value: int, device: torch.device) -> list[int]:
    return [
        int(item)
        for item in gather_scalar(float(value), device)
    ]


def max_quality_error(
    qualities: tuple[dict[str, float], ...],
) -> float:
    return max(quality["max_abs_error"] for quality in qualities)


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    (
        num_tokens,
        routes,
        capacity_variants,
        warmup,
        iterations,
    ) = profile_configurations(args.profile)
    assignments_per_rank = num_tokens * args.top_k
    num_experts = world_size * args.experts_per_rank

    generator = torch.Generator(device=device)
    generator.manual_seed(7_000)
    gate_weight = (
        torch.randn(
            (num_experts, args.hidden),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.hidden)
    )
    global_w1 = (
        torch.randn(
            (
                num_experts,
                2 * args.intermediate,
                args.hidden,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.hidden)
    )
    global_w2 = (
        torch.randn(
            (
                num_experts,
                args.hidden,
                args.intermediate,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.intermediate)
    )
    local_start = rank * args.experts_per_rank
    local_end = local_start + args.experts_per_rank
    local_w1 = global_w1[local_start:local_end].clone()
    local_w2 = global_w2[local_start:local_end].clone()
    expert_map = torch.arange(
        args.experts_per_rank,
        device=device,
        dtype=torch.int32,
    )
    tie_breaker = (
        torch.arange(
            num_experts,
            device=device,
            dtype=torch.float32,
        )
        * 1e-6
    )

    records: list[dict[str, Any]] = []
    for route_name, hot_bias in routes:
        bias = routing_bias(
            "global_biased",
            rank=rank,
            num_experts=num_experts,
            experts_per_rank=args.experts_per_rank,
            hot_bias=hot_bias,
            device=device,
        )
        input_generator = torch.Generator(device=device)
        input_generator.manual_seed(
            60_000
            + rank * 10_003
            + {
                "balanced": 0,
                "global_bias1": 1_000_003,
                "global_bias4": 2_000_006,
            }[route_name]
        )
        hidden = torch.randn(
            (num_tokens, args.hidden),
            device=device,
            dtype=dtype,
            generator=input_generator,
        )
        uncapped_route = route_and_pack(
            hidden,
            gate_weight,
            bias,
            tie_breaker,
            top_k=args.top_k,
            experts_per_rank=args.experts_per_rank,
            world_size=world_size,
        )
        send_counts = uncapped_route["send_counts"]
        assert isinstance(send_counts, list)
        count_matrix = gather_segment_counts(send_counts, device)
        flat_counts = [
            count
            for source_counts in count_matrix
            for count in source_counts
        ]
        minimum_lossless_capacity = max(flat_counts)
        mean_capacity = assignments_per_rank / world_size

        for capacity_name, capacity_scale in capacity_variants:
            capacity = math.ceil(
                minimum_lossless_capacity * capacity_scale
            )

            def build_variable() -> TensorDict:
                route = route_and_pack(
                    hidden,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                )
                return apply_host_capacity(route, capacity)

            def build_dense() -> TensorDict:
                return build_padded_route(
                    hidden,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                    capacity=capacity,
                )

            def build_sparse() -> TensorDict:
                return build_sparse_padded_route(
                    hidden,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                    capacity=capacity,
                )

            variable_route = build_variable()
            dense_route = build_dense()
            sparse_route = build_sparse()
            dense_dispatched = dispatch_padded(dense_route)
            sparse_dispatched = dispatch_padded(sparse_route)

            dense_expert_output = fused_experts(
                dense_dispatched["recv_hidden"],
                local_w1,
                local_w2,
                dense_dispatched["recv_weights"],
                dense_dispatched["recv_local_ids"],
            )
            mapped_sparse_expert_output = (
                mapped_sparse_local_experts(
                    sparse_dispatched,
                    local_w1,
                    local_w2,
                    expert_map,
                )
            )
            masked_sparse_expert_output = (
                masked_sparse_local_experts(
                    sparse_dispatched,
                    local_w1,
                    local_w2,
                )
            )
            variable_output = run_variable_route(
                variable_route,
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )
            dense_output = return_padded(
                dense_expert_output,
                dense_route,
                num_tokens=num_tokens,
            )
            mapped_sparse_output = return_padded(
                mapped_sparse_expert_output,
                sparse_route,
                num_tokens=num_tokens,
            )
            masked_sparse_output = return_padded(
                masked_sparse_expert_output,
                sparse_route,
                num_tokens=num_tokens,
            )
            masked_reference = masked_global_reference(
                hidden,
                global_w1,
                global_w2,
                dense_route,
            )
            torch.cuda.synchronize()

            variable_selected = variable_route[
                "selected_assignment_indices"
            ]
            dense_selected = dense_route[
                "selected_assignment_indices"
            ]
            sparse_selected = sparse_route[
                "selected_assignment_indices"
            ]
            assert isinstance(variable_selected, torch.Tensor)
            assert isinstance(dense_selected, torch.Tensor)
            assert isinstance(sparse_selected, torch.Tensor)
            local_selection_equal = (
                torch.equal(
                    torch.sort(variable_selected).values,
                    torch.sort(dense_selected).values,
                )
                and torch.equal(
                    torch.sort(dense_selected).values,
                    torch.sort(sparse_selected).values,
                )
            )
            selection_equal = min(
                gather_scalar(float(local_selection_equal), device)
            )

            variable_quality = output_quality(
                variable_output,
                masked_reference,
            )
            dense_quality = output_quality(
                dense_output,
                masked_reference,
            )
            mapped_sparse_quality = output_quality(
                mapped_sparse_output,
                masked_reference,
            )
            masked_sparse_quality = output_quality(
                masked_sparse_output,
                masked_reference,
            )
            dense_vs_mapped_sparse = output_quality(
                dense_output,
                mapped_sparse_output,
            )
            dense_vs_masked_sparse = output_quality(
                dense_output,
                masked_sparse_output,
            )
            mapped_vs_masked_sparse = output_quality(
                mapped_sparse_output,
                masked_sparse_output,
            )
            variable_vs_mapped_sparse = output_quality(
                variable_output,
                mapped_sparse_output,
            )
            variable_vs_masked_sparse = output_quality(
                variable_output,
                masked_sparse_output,
            )

            sparse_ids = sparse_dispatched["recv_local_ids"]
            sparse_weights = sparse_dispatched["recv_weights"]
            assert isinstance(sparse_ids, torch.Tensor)
            assert isinstance(sparse_weights, torch.Tensor)
            invalid_mask = sparse_ids.reshape(-1) < 0
            local_mapped_invalid_output_max = (
                mapped_sparse_expert_output[invalid_mask]
                .float()
                .abs()
                .max()
                .item()
                if bool(invalid_mask.any().item())
                else 0.0
            )
            local_masked_invalid_output_max = (
                masked_sparse_expert_output[invalid_mask]
                .float()
                .abs()
                .max()
                .item()
                if bool(invalid_mask.any().item())
                else 0.0
            )
            mapped_invalid_output_max = max(
                gather_scalar(
                    local_mapped_invalid_output_max,
                    device,
                )
            )
            masked_invalid_output_max = max(
                gather_scalar(
                    local_masked_invalid_output_max,
                    device,
                )
            )
            exact_qualities = (
                variable_quality,
                dense_quality,
                mapped_sparse_quality,
                masked_sparse_quality,
                dense_vs_mapped_sparse,
                dense_vs_masked_sparse,
                mapped_vs_masked_sparse,
                variable_vs_mapped_sparse,
                variable_vs_masked_sparse,
            )
            if (
                selection_equal != 1.0
                or max_quality_error(exact_qualities) != 0.0
                or mapped_invalid_output_max != 0.0
                or masked_invalid_output_max != 0.0
            ):
                raise RuntimeError(
                    "Sparse padding correctness failed: "
                    f"selection_equal={selection_equal}, "
                    f"qualities={exact_qualities}, "
                    "invalid_output_max="
                    f"{mapped_invalid_output_max}/"
                    f"{masked_invalid_output_max}"
                )

            dense_block_size, dense_aligned_rows = (
                alignment_metadata(
                    dense_dispatched,
                    local_w1,
                    local_w2,
                    None,
                )
            )
            sparse_block_size, sparse_aligned_rows = (
                alignment_metadata(
                    sparse_dispatched,
                    local_w1,
                    local_w2,
                    expert_map,
                )
            )
            if dense_block_size != sparse_block_size:
                raise RuntimeError(
                    "Dense and sparse block sizes differ: "
                    f"{dense_block_size} != {sparse_block_size}"
                )
            fixed_rows = sparse_ids.shape[0]
            local_valid_rows = int(
                (sparse_weights.reshape(-1) != 0).sum().item()
            )
            valid_rows_by_rank = gather_int(
                local_valid_rows,
                device,
            )
            dense_aligned_rows_by_rank = gather_int(
                dense_aligned_rows,
                device,
            )
            sparse_aligned_rows_by_rank = gather_int(
                sparse_aligned_rows,
                device,
            )
            total_valid_rows = sum(valid_rows_by_rank)
            planned_global = world_size * assignments_per_rank
            if total_valid_rows != planned_global:
                raise RuntimeError(
                    "Lossless sparse capacity dropped assignments: "
                    f"valid={total_valid_rows}, "
                    f"planned={planned_global}"
                )
            if (
                any(
                    sparse > dense
                    for sparse, dense in zip(
                        sparse_aligned_rows_by_rank,
                        dense_aligned_rows_by_rank,
                        strict=True,
                    )
                )
                or sum(sparse_aligned_rows_by_rank)
                >= sum(dense_aligned_rows_by_rank)
            ):
                raise RuntimeError(
                    "Sparse alignment did not reduce aggregate rows: "
                    f"dense={dense_aligned_rows_by_rank}, "
                    f"sparse={sparse_aligned_rows_by_rank}"
                )

            def dense_experts_prebuilt() -> torch.Tensor:
                return fused_experts(
                    dense_dispatched["recv_hidden"],
                    local_w1,
                    local_w2,
                    dense_dispatched["recv_weights"],
                    dense_dispatched["recv_local_ids"],
                )

            def mapped_sparse_experts_prebuilt() -> torch.Tensor:
                return mapped_sparse_local_experts(
                    sparse_dispatched,
                    local_w1,
                    local_w2,
                    expert_map,
                )

            def masked_sparse_experts_prebuilt() -> torch.Tensor:
                return masked_sparse_local_experts(
                    sparse_dispatched,
                    local_w1,
                    local_w2,
                )

            def variable_end_to_end() -> torch.Tensor:
                return run_variable_route(
                    build_variable(),
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            def dense_end_to_end() -> torch.Tensor:
                return run_padded_route(
                    build_dense(),
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            def mapped_sparse_end_to_end() -> torch.Tensor:
                return run_mapped_sparse_padded_route(
                    build_sparse(),
                    local_w1,
                    local_w2,
                    expert_map,
                    num_tokens=num_tokens,
                )

            def masked_sparse_end_to_end() -> torch.Tensor:
                return run_masked_sparse_padded_route(
                    build_sparse(),
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            schedules: dict[str, Callable[[], Any]] = {
                "dense_route_capacity": build_dense,
                "sparse_route_capacity": build_sparse,
                "dense_experts_prebuilt": dense_experts_prebuilt,
                "mapped_sparse_experts_prebuilt": (
                    mapped_sparse_experts_prebuilt
                ),
                "masked_sparse_experts_prebuilt": (
                    masked_sparse_experts_prebuilt
                ),
                "variable_end_to_end": variable_end_to_end,
                "dense_end_to_end": dense_end_to_end,
                "mapped_sparse_end_to_end": (
                    mapped_sparse_end_to_end
                ),
                "masked_sparse_end_to_end": (
                    masked_sparse_end_to_end
                ),
            }
            timings = {
                name: critical_path_summary(
                    timed_schedule(
                        schedule,
                        warmup=warmup,
                        iterations=iterations,
                        local_rank=local_rank,
                    ),
                    device,
                )
                for name, schedule in schedules.items()
            }
            expert_crossover = None
            end_to_end_crossover = None
            if args.profile == "crossover":
                expert_crossover = timed_crossover(
                    (
                        (
                            "dense_experts_prebuilt",
                            dense_experts_prebuilt,
                        ),
                        (
                            "mapped_sparse_experts_prebuilt",
                            mapped_sparse_experts_prebuilt,
                        ),
                        (
                            "masked_sparse_experts_prebuilt",
                            masked_sparse_experts_prebuilt,
                        ),
                    ),
                    warmup_cycles=6,
                    measured_cycles=args.crossover_cycles,
                    local_rank=local_rank,
                    device=device,
                )
                end_to_end_crossover = timed_crossover(
                    (
                        (
                            "variable_end_to_end",
                            variable_end_to_end,
                        ),
                        ("dense_end_to_end", dense_end_to_end),
                        (
                            "mapped_sparse_end_to_end",
                            mapped_sparse_end_to_end,
                        ),
                        (
                            "masked_sparse_end_to_end",
                            masked_sparse_end_to_end,
                        ),
                    ),
                    warmup_cycles=6,
                    measured_cycles=args.crossover_cycles,
                    local_rank=local_rank,
                    device=device,
                )

            records.append(
                {
                    "tokens_per_rank": num_tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts_per_rank": args.experts_per_rank,
                    "num_experts": num_experts,
                    "top_k": args.top_k,
                    "route": route_name,
                    "hot_bias": hot_bias,
                    "capacity_variant": capacity_name,
                    "capacity_per_destination": capacity,
                    "effective_capacity_factor": (
                        capacity / mean_capacity
                    ),
                    "frontier": {
                        "minimum_lossless_capacity": (
                            minimum_lossless_capacity
                        ),
                        "minimum_lossless_factor": (
                            minimum_lossless_capacity
                            / mean_capacity
                        ),
                        "source_destination_count_min": min(
                            flat_counts
                        ),
                        "source_destination_count_median": (
                            statistics.median(flat_counts)
                        ),
                        "source_destination_count_max": max(
                            flat_counts
                        ),
                    },
                    "padding": {
                        "fixed_rows_per_rank": fixed_rows,
                        "valid_rows_by_rank": valid_rows_by_rank,
                        "padding_fraction_by_rank": [
                            1.0 - valid / fixed_rows
                            for valid in valid_rows_by_rank
                        ],
                        "global_padding_fraction": (
                            1.0
                            - total_valid_rows
                            / (world_size * fixed_rows)
                        ),
                        "critical_rank_valid_rows": max(
                            valid_rows_by_rank
                        ),
                        "critical_rank_padding_fraction": (
                            1.0
                            - max(valid_rows_by_rank)
                            / fixed_rows
                        ),
                    },
                    "gpu_compaction": {
                        "padding_expert_id": -1,
                        "host_visible_count": False,
                        "block_size_m": dense_block_size,
                        "mapped_zero_strategy": (
                            "identity expert map with cache zero"
                        ),
                        "masked_zero_strategy": (
                            "row-selective Triton cleanup"
                        ),
                        "dense_aligned_rows_by_rank": (
                            dense_aligned_rows_by_rank
                        ),
                        "sparse_aligned_rows_by_rank": (
                            sparse_aligned_rows_by_rank
                        ),
                        "critical_aligned_row_reduction": (
                            1.0
                            - max(sparse_aligned_rows_by_rank)
                            / max(dense_aligned_rows_by_rank)
                        ),
                        "aggregate_aligned_row_reduction": (
                            1.0
                            - sum(sparse_aligned_rows_by_rank)
                            / sum(dense_aligned_rows_by_rank)
                        ),
                    },
                    "correctness": {
                        "selection_equal": (
                            selection_equal == 1.0
                        ),
                        "variable_vs_masked_reference": (
                            variable_quality
                        ),
                        "dense_vs_masked_reference": (
                            dense_quality
                        ),
                        "mapped_sparse_vs_masked_reference": (
                            mapped_sparse_quality
                        ),
                        "masked_sparse_vs_masked_reference": (
                            masked_sparse_quality
                        ),
                        "dense_vs_mapped_sparse": (
                            dense_vs_mapped_sparse
                        ),
                        "dense_vs_masked_sparse": (
                            dense_vs_masked_sparse
                        ),
                        "mapped_vs_masked_sparse": (
                            mapped_vs_masked_sparse
                        ),
                        "variable_vs_mapped_sparse": (
                            variable_vs_mapped_sparse
                        ),
                        "variable_vs_masked_sparse": (
                            variable_vs_masked_sparse
                        ),
                        "mapped_invalid_output_max": (
                            mapped_invalid_output_max
                        ),
                        "masked_invalid_output_max": (
                            masked_invalid_output_max
                        ),
                    },
                    "timings": timings,
                    "grouped_speedups": {
                        "mapped_vs_dense_experts": (
                            timings["dense_experts_prebuilt"][
                                "critical_path_ms"
                            ]
                            / timings[
                                "mapped_sparse_experts_prebuilt"
                            ][
                                "critical_path_ms"
                            ]
                        ),
                        "masked_vs_dense_experts": (
                            timings["dense_experts_prebuilt"][
                                "critical_path_ms"
                            ]
                            / timings[
                                "masked_sparse_experts_prebuilt"
                            ][
                                "critical_path_ms"
                            ]
                        ),
                        "masked_vs_mapped_experts": (
                            timings[
                                "mapped_sparse_experts_prebuilt"
                            ][
                                "critical_path_ms"
                            ]
                            / timings[
                                "masked_sparse_experts_prebuilt"
                            ][
                                "critical_path_ms"
                            ]
                        ),
                        "dense_vs_variable_end_to_end": (
                            timings["variable_end_to_end"][
                                "critical_path_ms"
                            ]
                            / timings["dense_end_to_end"][
                                "critical_path_ms"
                            ]
                        ),
                        "mapped_vs_dense_end_to_end": (
                            timings["dense_end_to_end"][
                                "critical_path_ms"
                            ]
                            / timings[
                                "mapped_sparse_end_to_end"
                            ][
                                "critical_path_ms"
                            ]
                        ),
                        "masked_vs_dense_end_to_end": (
                            timings["dense_end_to_end"][
                                "critical_path_ms"
                            ]
                            / timings[
                                "masked_sparse_end_to_end"
                            ][
                                "critical_path_ms"
                            ]
                        ),
                        "masked_vs_mapped_end_to_end": (
                            timings[
                                "mapped_sparse_end_to_end"
                            ][
                                "critical_path_ms"
                            ]
                            / timings[
                                "masked_sparse_end_to_end"
                            ][
                                "critical_path_ms"
                            ]
                        ),
                        "mapped_vs_variable_end_to_end": (
                            timings["variable_end_to_end"][
                                "critical_path_ms"
                            ]
                            / timings[
                                "mapped_sparse_end_to_end"
                            ][
                                "critical_path_ms"
                            ]
                        ),
                        "masked_vs_variable_end_to_end": (
                            timings["variable_end_to_end"][
                                "critical_path_ms"
                            ]
                            / timings[
                                "masked_sparse_end_to_end"
                            ][
                                "critical_path_ms"
                            ]
                        ),
                    },
                    "expert_crossover": expert_crossover,
                    "end_to_end_crossover": (
                        end_to_end_crossover
                    ),
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )
            del (
                variable_route,
                dense_route,
                sparse_route,
                dense_dispatched,
                sparse_dispatched,
                dense_expert_output,
                mapped_sparse_expert_output,
                masked_sparse_expert_output,
                variable_output,
                dense_output,
                mapped_sparse_output,
                masked_sparse_output,
                masked_reference,
            )
            torch.cuda.empty_cache()

        del uncapped_route, hidden
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_sparse_padding",
            "profile": args.profile,
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "Dense static padding assigns zero-weight slots to expert zero, so vLLM alignment and expert kernels still process them.",
                "Both sparse variants assign zero-weight slots expert ID -1, so vLLM ignores invalid IDs during device-side assignment alignment.",
                "The mapped variant passes an identity expert map, which makes vLLM zero the full output cache before its second expert kernel.",
                "The masked variant avoids that full-cache zero and instead clears only invalid output rows with one row-selective Triton kernel.",
                "Transport tensor shapes and collective sizes remain static and identical; no receive count returns to Python.",
                "The critical rank can have much less padding than the global average under receiver skew, so aligned-row and speedup statistics are reported by rank.",
            ],
            "records": records,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(output)

    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["smoke", "full", "crossover"],
        default="smoke",
    )
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--crossover-cycles", type=int, default=20)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
