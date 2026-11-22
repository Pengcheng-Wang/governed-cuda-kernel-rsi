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
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)

from distributed_moe_capacity_frontier import (
    gather_segment_counts,
)
from distributed_moe_end_to_end import (
    TensorDict,
    critical_path_summary,
    gather_scalar,
    local_experts,
    output_quality,
    route_and_pack,
    routing_bias,
    timed_schedule,
)
from distributed_moe_packed_dispatch import (
    build_metadata_packet,
    timed_crossover,
)
from distributed_moe_padded_capacity import (
    masked_global_reference,
    run_variable_route,
)


CapacityMatrix = list[list[int]]


def profile_configurations(
    profile: str,
) -> tuple[
    int,
    tuple[tuple[str, float], ...],
    int,
    int,
    int,
]:
    if profile == "smoke":
        return (
            100,
            (
                ("balanced", 0.0),
                ("global_bias1", 1.0),
            ),
            4,
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
        8,
        1,
        4 if profile == "full" else 2,
    )


def route_seed(
    route_name: str,
    rank: int,
    batch: int,
    *,
    calibration: bool,
) -> int:
    route_offset = {
        "balanced": 0,
        "global_bias1": 1_000_003,
        "global_bias4": 2_000_006,
    }[route_name]
    phase_offset = 10_000_019 if calibration else 0
    return (
        60_000
        + phase_offset
        + route_offset
        + rank * 10_003
        + batch * 1_000_033
    )


def make_hidden(
    num_tokens: int,
    hidden: int,
    *,
    route_name: str,
    rank: int,
    batch: int,
    calibration: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        route_seed(
            route_name,
            rank,
            batch,
            calibration=calibration,
        )
    )
    return torch.randn(
        (num_tokens, hidden),
        device=device,
        dtype=dtype,
        generator=generator,
    )


def matrix_max(
    left: CapacityMatrix,
    right: CapacityMatrix,
) -> CapacityMatrix:
    return [
        [
            max(left_value, right_value)
            for left_value, right_value in zip(
                left_row,
                right_row,
                strict=True,
            )
        ]
        for left_row, right_row in zip(
            left,
            right,
            strict=True,
        )
    ]


def scale_matrix(
    matrix: CapacityMatrix,
    scale: float,
) -> CapacityMatrix:
    return [
        [
            math.ceil(value * scale)
            for value in row
        ]
        for row in matrix
    ]


def margin_matrix(
    matrix: CapacityMatrix,
    margin: int,
) -> CapacityMatrix:
    return [
        [
            value + margin
            for value in row
        ]
        for row in matrix
    ]


def uniform_matrix(
    matrix: CapacityMatrix,
) -> CapacityMatrix:
    width = len(matrix)
    capacity = max(
        value
        for row in matrix
        for value in row
    )
    return [
        [capacity for _ in range(width)]
        for _ in range(width)
    ]


def matrix_row_sums(matrix: CapacityMatrix) -> list[int]:
    return [sum(row) for row in matrix]


def matrix_column_sums(
    matrix: CapacityMatrix,
) -> list[int]:
    return [
        sum(row[column] for row in matrix)
        for column in range(len(matrix))
    ]


def calibrate_capacity_matrix(
    *,
    route_name: str,
    calibration_batches: int,
    num_tokens: int,
    hidden_size: int,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    top_k: int,
    experts_per_rank: int,
    world_size: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> CapacityMatrix:
    maximum = [
        [0 for _ in range(world_size)]
        for _ in range(world_size)
    ]
    for batch in range(calibration_batches):
        hidden = make_hidden(
            num_tokens,
            hidden_size,
            route_name=route_name,
            rank=rank,
            batch=batch,
            calibration=True,
            device=device,
            dtype=dtype,
        )
        route = route_and_pack(
            hidden,
            gate_weight,
            bias,
            tie_breaker,
            top_k=top_k,
            experts_per_rank=experts_per_rank,
            world_size=world_size,
        )
        send_counts = route["send_counts"]
        assert isinstance(send_counts, list)
        maximum = matrix_max(
            maximum,
            gather_segment_counts(send_counts, device),
        )
        del hidden, route
    return maximum


def apply_host_capacity_row(
    route: TensorDict,
    capacity_row: list[int],
) -> TensorDict:
    send_hidden = route["send_hidden"]
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    sorted_token_indices = route["sorted_token_indices"]
    sorted_assignment_indices = route[
        "sorted_assignment_indices"
    ]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_local_ids, torch.Tensor)
    assert isinstance(send_weights, torch.Tensor)
    assert isinstance(sorted_token_indices, torch.Tensor)
    assert isinstance(sorted_assignment_indices, torch.Tensor)
    assert isinstance(send_counts, list)

    selected_segments: list[torch.Tensor] = []
    accepted_counts: list[int] = []
    offset = 0
    flat_weights = send_weights.reshape(-1)
    for count, capacity in zip(
        send_counts,
        capacity_row,
        strict=True,
    ):
        accepted = min(count, capacity)
        accepted_counts.append(accepted)
        if accepted == count:
            relative = torch.arange(
                count,
                device=send_hidden.device,
                dtype=torch.int64,
            )
        elif accepted:
            relative = torch.topk(
                flat_weights[offset : offset + count].float(),
                accepted,
                sorted=False,
            ).indices
            relative = torch.sort(relative).values
        else:
            relative = torch.empty(
                0,
                device=send_hidden.device,
                dtype=torch.int64,
            )
        selected_segments.append(relative + offset)
        offset += count
    selected = torch.cat(selected_segments)

    capped = dict(route)
    capped.update(
        {
            "send_hidden": send_hidden.index_select(
                0,
                selected,
            ).contiguous(),
            "send_local_ids": send_local_ids.index_select(
                0,
                selected,
            ).contiguous(),
            "send_weights": send_weights.index_select(
                0,
                selected,
            ).contiguous(),
            "sorted_token_indices": (
                sorted_token_indices.index_select(0, selected)
            ),
            "selected_assignment_indices": (
                sorted_assignment_indices.index_select(
                    0,
                    selected,
                )
            ),
            "send_counts_tensor": torch.tensor(
                accepted_counts,
                device=send_hidden.device,
                dtype=torch.int64,
            ),
            "send_counts": accepted_counts,
        }
    )
    return capped


def build_static_matrix_route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    *,
    top_k: int,
    experts_per_rank: int,
    world_size: int,
    capacity_row: list[int],
) -> TensorDict:
    gating_output = (
        F.linear(hidden_states, gate_weight).float()
        + bias
        + tie_breaker
    )
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states,
        gating_output,
        top_k,
        True,
    )
    topk_weights = topk_weights.to(hidden_states.dtype)

    token_indices = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
        dtype=torch.int64,
    ).repeat_interleave(top_k)
    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    destinations = torch.div(
        flat_ids,
        experts_per_rank,
        rounding_mode="floor",
    )
    local_ids = torch.remainder(
        flat_ids,
        experts_per_rank,
    )

    weight_order = torch.argsort(
        flat_weights.float(),
        descending=True,
        stable=True,
    )
    destinations_by_weight = destinations.index_select(
        0,
        weight_order,
    )
    destination_order = torch.argsort(
        destinations_by_weight,
        stable=True,
    )
    grouped_order = weight_order.index_select(
        0,
        destination_order,
    )
    grouped_destinations = destinations.index_select(
        0,
        grouped_order,
    )
    planned_counts = torch.bincount(
        grouped_destinations,
        minlength=world_size,
    ).to(torch.int64)
    planned_offsets = (
        torch.cumsum(planned_counts, dim=0)
        - planned_counts
    )
    positions = (
        torch.arange(
            grouped_order.numel(),
            device=hidden_states.device,
            dtype=torch.int64,
        )
        - planned_offsets.index_select(
            0,
            grouped_destinations,
        )
    )
    capacity_tensor = torch.tensor(
        capacity_row,
        device=hidden_states.device,
        dtype=torch.int64,
    )
    accepted_mask = positions < capacity_tensor.index_select(
        0,
        grouped_destinations,
    )
    selected_assignment_indices = grouped_order[
        accepted_mask
    ]
    selected_destinations = grouped_destinations[
        accepted_mask
    ]
    selected_positions = positions[accepted_mask]
    capacity_offsets = (
        torch.cumsum(capacity_tensor, dim=0)
        - capacity_tensor
    )
    fixed_indices = (
        capacity_offsets.index_select(
            0,
            selected_destinations,
        )
        + selected_positions
    )
    selected_token_indices = token_indices.index_select(
        0,
        selected_assignment_indices,
    )
    selected_local_ids = local_ids.index_select(
        0,
        selected_assignment_indices,
    )
    selected_weights = flat_weights.index_select(
        0,
        selected_assignment_indices,
    )
    selected_hidden = hidden_states.index_select(
        0,
        selected_token_indices,
    )

    fixed_rows = sum(capacity_row)
    send_hidden = torch.zeros(
        (fixed_rows, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    send_local_ids = torch.zeros(
        (fixed_rows, 1),
        device=hidden_states.device,
        dtype=torch.int32,
    )
    send_weights = torch.zeros(
        (fixed_rows, 1),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    fixed_token_indices = torch.zeros(
        fixed_rows,
        device=hidden_states.device,
        dtype=torch.int64,
    )
    send_hidden.index_copy_(
        0,
        fixed_indices,
        selected_hidden,
    )
    send_local_ids.index_copy_(
        0,
        fixed_indices,
        selected_local_ids.to(torch.int32).reshape(-1, 1),
    )
    send_weights.index_copy_(
        0,
        fixed_indices,
        selected_weights.reshape(-1, 1),
    )
    fixed_token_indices.index_copy_(
        0,
        fixed_indices,
        selected_token_indices,
    )

    return {
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "send_hidden": send_hidden,
        "send_local_ids": send_local_ids,
        "send_weights": send_weights,
        "fixed_token_indices": fixed_token_indices,
        "selected_token_indices": selected_token_indices,
        "selected_weights": selected_weights,
        "selected_assignment_indices": (
            selected_assignment_indices
        ),
        "accepted_counts_tensor": torch.minimum(
            planned_counts,
            capacity_tensor,
        ),
        "send_counts_tensor": capacity_tensor,
        "send_counts": list(capacity_row),
    }


def dispatch_static_matrix(
    route: TensorDict,
    recv_counts: list[int],
) -> TensorDict:
    send_hidden = route["send_hidden"]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts, list)
    metadata = build_metadata_packet(route)
    recv_total = sum(recv_counts)
    recv_hidden = torch.empty(
        (recv_total, send_hidden.shape[1]),
        device=send_hidden.device,
        dtype=send_hidden.dtype,
    )
    recv_metadata = torch.empty(
        (recv_total, 2),
        device=send_hidden.device,
        dtype=send_hidden.dtype,
    )
    dist.all_to_all_single(
        recv_hidden,
        send_hidden,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    dist.all_to_all_single(
        recv_metadata,
        metadata,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    return {
        "recv_hidden": recv_hidden,
        "recv_local_ids": (
            recv_metadata[:, :1]
            .to(torch.int32)
            .contiguous()
        ),
        "recv_weights": (
            recv_metadata[:, 1:].contiguous()
        ),
        "recv_counts": list(recv_counts),
    }


def return_static_matrix(
    expert_output: torch.Tensor,
    route: TensorDict,
    recv_counts: list[int],
    *,
    num_tokens: int,
) -> torch.Tensor:
    send_counts = route["send_counts"]
    fixed_token_indices = route["fixed_token_indices"]
    assert isinstance(send_counts, list)
    assert isinstance(fixed_token_indices, torch.Tensor)
    returned = torch.empty(
        (sum(send_counts), expert_output.shape[1]),
        device=expert_output.device,
        dtype=expert_output.dtype,
    )
    dist.all_to_all_single(
        returned,
        expert_output,
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts,
    )
    combined = torch.zeros(
        (num_tokens, returned.shape[1]),
        device=returned.device,
        dtype=torch.float32,
    )
    combined.index_add_(
        0,
        fixed_token_indices,
        returned.float(),
    )
    return combined.to(returned.dtype)


def run_static_matrix_route(
    route: TensorDict,
    recv_counts: list[int],
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    dispatched = dispatch_static_matrix(
        route,
        recv_counts,
    )
    expert_output = local_experts(
        dispatched,
        local_w1,
        local_w2,
    )
    return return_static_matrix(
        expert_output,
        route,
        recv_counts,
        num_tokens=num_tokens,
    )


def gather_int(
    value: int,
    device: torch.device,
) -> list[int]:
    return [
        int(item)
        for item in gather_scalar(float(value), device)
    ]


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
        calibration_batches,
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
        calibration_matrix = calibrate_capacity_matrix(
            route_name=route_name,
            calibration_batches=calibration_batches,
            num_tokens=num_tokens,
            hidden_size=args.hidden,
            gate_weight=gate_weight,
            bias=bias,
            tie_breaker=tie_breaker,
            top_k=args.top_k,
            experts_per_rank=args.experts_per_rank,
            world_size=world_size,
            rank=rank,
            device=device,
            dtype=dtype,
        )
        calibrated_headroom_matrix = scale_matrix(
            calibration_matrix,
            args.calibration_headroom,
        )
        calibrated_margin_matrix = margin_matrix(
            calibration_matrix,
            args.calibration_margin,
        )
        hidden = make_hidden(
            num_tokens,
            args.hidden,
            route_name=route_name,
            rank=rank,
            batch=0,
            calibration=False,
            device=device,
            dtype=dtype,
        )
        evaluation_route = route_and_pack(
            hidden,
            gate_weight,
            bias,
            tie_breaker,
            top_k=args.top_k,
            experts_per_rank=args.experts_per_rank,
            world_size=world_size,
        )
        evaluation_send_counts = evaluation_route[
            "send_counts"
        ]
        assert isinstance(evaluation_send_counts, list)
        evaluation_matrix = gather_segment_counts(
            evaluation_send_counts,
            device,
        )
        uniform_exact_matrix = uniform_matrix(
            evaluation_matrix,
        )
        profile_matrices: list[
            tuple[str, str, CapacityMatrix]
        ] = [
            (
                "uniform_exact",
                "evaluation global maximum",
                uniform_exact_matrix,
            ),
            (
                "calibrated_1125",
                (
                    f"{calibration_batches} calibration batches "
                    f"with {args.calibration_headroom:.3f}x headroom"
                ),
                calibrated_headroom_matrix,
            ),
            (
                "calibrated_plus16",
                (
                    f"{calibration_batches} calibration batches "
                    f"with +{args.calibration_margin} rows"
                ),
                calibrated_margin_matrix,
            ),
            (
                "oracle_matrix",
                "evaluation source-destination counts",
                evaluation_matrix,
            ),
        ]
        if args.reverse_profiles:
            profile_matrices.reverse()
        uniform_fixed_rows = sum(
            sum(row)
            for row in uniform_exact_matrix
        )
        uniform_critical_recv = max(
            matrix_column_sums(
                uniform_exact_matrix
            )
        )

        for (
            profile_name,
            profile_origin,
            capacity_matrix,
        ) in profile_matrices:
            capacity_row = capacity_matrix[rank]
            recv_counts = [
                capacity_matrix[source][rank]
                for source in range(world_size)
            ]

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
                return apply_host_capacity_row(
                    route,
                    capacity_row,
                )

            def build_static() -> TensorDict:
                return build_static_matrix_route(
                    hidden,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                    capacity_row=capacity_row,
                )

            variable_route = build_variable()
            static_route = build_static()
            variable_output = run_variable_route(
                variable_route,
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )
            static_output = run_static_matrix_route(
                static_route,
                recv_counts,
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )
            masked_reference = masked_global_reference(
                hidden,
                global_w1,
                global_w2,
                static_route,
            )
            topk_ids = static_route["topk_ids"]
            topk_weights = static_route["topk_weights"]
            assert isinstance(topk_ids, torch.Tensor)
            assert isinstance(topk_weights, torch.Tensor)
            uncapped_reference = fused_experts(
                hidden,
                global_w1,
                global_w2,
                topk_weights,
                topk_ids,
            )
            torch.cuda.synchronize()

            variable_selected = variable_route[
                "selected_assignment_indices"
            ]
            static_selected = static_route[
                "selected_assignment_indices"
            ]
            assert isinstance(variable_selected, torch.Tensor)
            assert isinstance(static_selected, torch.Tensor)
            local_selection_equal = torch.equal(
                torch.sort(variable_selected).values,
                torch.sort(static_selected).values,
            )
            selection_equal = min(
                gather_scalar(
                    float(local_selection_equal),
                    device,
                )
            )
            variable_quality = output_quality(
                variable_output,
                masked_reference,
            )
            static_quality = output_quality(
                static_output,
                masked_reference,
            )
            static_vs_variable = output_quality(
                static_output,
                variable_output,
            )
            quality_vs_uncapped = output_quality(
                static_output,
                uncapped_reference,
            )
            if (
                selection_equal != 1.0
                or variable_quality["max_abs_error"] != 0.0
                or static_quality["max_abs_error"] != 0.0
                or static_vs_variable["max_abs_error"] != 0.0
            ):
                raise RuntimeError(
                    "Profiled transport correctness failed: "
                    f"selection_equal={selection_equal}, "
                    f"variable={variable_quality}, "
                    f"static={static_quality}, "
                    f"static_vs_variable={static_vs_variable}"
                )

            accepted_counts_tensor = static_route[
                "accepted_counts_tensor"
            ]
            selected_token_indices = static_route[
                "selected_token_indices"
            ]
            selected_weights = static_route[
                "selected_weights"
            ]
            assert isinstance(
                accepted_counts_tensor,
                torch.Tensor,
            )
            assert isinstance(
                selected_token_indices,
                torch.Tensor,
            )
            assert isinstance(
                selected_weights,
                torch.Tensor,
            )
            accepted_row = [
                int(value)
                for value in (
                    accepted_counts_tensor.cpu().tolist()
                )
            ]
            accepted_matrix = gather_segment_counts(
                accepted_row,
                device,
            )
            accepted_global = sum(
                sum(row)
                for row in accepted_matrix
            )
            planned_global = (
                world_size * assignments_per_rank
            )
            fixed_send_rows = matrix_row_sums(
                capacity_matrix
            )
            fixed_recv_rows = matrix_column_sums(
                capacity_matrix
            )
            valid_recv_rows = matrix_column_sums(
                accepted_matrix
            )
            fixed_rows_global = sum(fixed_send_rows)
            critical_receiver_rank = max(
                range(world_size),
                key=lambda index: fixed_recv_rows[index],
            )
            token_assignment_counts = torch.bincount(
                selected_token_indices,
                minlength=num_tokens,
            )
            fully_dropped_tokens = sum(
                gather_scalar(
                    float(
                        (
                            token_assignment_counts == 0
                        ).sum().item()
                    ),
                    device,
                )
            )
            planned_weight_global = sum(
                gather_scalar(
                    float(
                        topk_weights.float().sum().item()
                    ),
                    device,
                )
            )
            accepted_weight_global = sum(
                gather_scalar(
                    float(
                        selected_weights.float().sum().item()
                    ),
                    device,
                )
            )
            if (
                profile_name
                in {"uniform_exact", "oracle_matrix"}
                and accepted_global != planned_global
            ):
                raise RuntimeError(
                    "Required lossless profile dropped work: "
                    f"profile={profile_name}, "
                    f"accepted={accepted_global}, "
                    f"planned={planned_global}"
                )

            def variable_payload_prebuilt() -> torch.Tensor:
                return run_variable_route(
                    variable_route,
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            def static_payload_prebuilt() -> torch.Tensor:
                return run_static_matrix_route(
                    static_route,
                    recv_counts,
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            def variable_end_to_end() -> torch.Tensor:
                return run_variable_route(
                    build_variable(),
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            def static_end_to_end() -> torch.Tensor:
                route = build_static()
                return run_static_matrix_route(
                    route,
                    recv_counts,
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            schedules: dict[str, Callable[[], Any]] = {
                "variable_route_capacity": build_variable,
                "static_route_capacity": build_static,
                "variable_payload_prebuilt": (
                    variable_payload_prebuilt
                ),
                "static_payload_prebuilt": (
                    static_payload_prebuilt
                ),
                "variable_end_to_end": variable_end_to_end,
                "static_end_to_end": static_end_to_end,
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
            payload_crossover = None
            end_to_end_crossover = None
            if args.profile == "crossover":
                payload_crossover = timed_crossover(
                    (
                        (
                            "variable_payload_prebuilt",
                            variable_payload_prebuilt,
                        ),
                        (
                            "static_payload_prebuilt",
                            static_payload_prebuilt,
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
                        (
                            "static_end_to_end",
                            static_end_to_end,
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
                    "profile_name": profile_name,
                    "profile_origin": profile_origin,
                    "calibration_batches": (
                        calibration_batches
                    ),
                    "calibration_headroom": (
                        args.calibration_headroom
                    ),
                    "calibration_margin": (
                        args.calibration_margin
                    ),
                    "capacity_matrix": capacity_matrix,
                    "evaluation_count_matrix": (
                        evaluation_matrix
                    ),
                    "accepted_count_matrix": (
                        accepted_matrix
                    ),
                    "capacity": {
                        "planned_assignments": planned_global,
                        "accepted_assignments": accepted_global,
                        "dropped_assignment_fraction": (
                            1.0
                            - accepted_global / planned_global
                        ),
                        "retained_router_weight_fraction": (
                            accepted_weight_global
                            / planned_weight_global
                        ),
                        "fully_dropped_token_fraction": (
                            fully_dropped_tokens
                            / (world_size * num_tokens)
                        ),
                        "lossless": (
                            accepted_global == planned_global
                        ),
                    },
                    "transport": {
                        "per_batch_count_exchange": False,
                        "per_batch_python_split_extraction": False,
                        "fixed_split_lists": True,
                        "fixed_send_rows_by_rank": (
                            fixed_send_rows
                        ),
                        "fixed_recv_rows_by_rank": (
                            fixed_recv_rows
                        ),
                        "valid_recv_rows_by_rank": (
                            valid_recv_rows
                        ),
                        "global_fixed_rows": (
                            fixed_rows_global
                        ),
                        "global_padding_fraction": (
                            1.0
                            - accepted_global
                            / fixed_rows_global
                        ),
                        "critical_receiver_rank": (
                            critical_receiver_rank
                        ),
                        "critical_receiver_padding_fraction": (
                            1.0
                            - valid_recv_rows[
                                critical_receiver_rank
                            ]
                            / fixed_recv_rows[
                                critical_receiver_rank
                            ]
                        ),
                        "row_reduction_vs_uniform": (
                            1.0
                            - fixed_rows_global
                            / uniform_fixed_rows
                        ),
                        "critical_recv_reduction_vs_uniform": (
                            1.0
                            - max(fixed_recv_rows)
                            / uniform_critical_recv
                        ),
                    },
                    "correctness": {
                        "selection_equal": (
                            selection_equal == 1.0
                        ),
                        "variable_vs_masked_reference": (
                            variable_quality
                        ),
                        "static_vs_masked_reference": (
                            static_quality
                        ),
                        "static_vs_variable": (
                            static_vs_variable
                        ),
                    },
                    "quality_vs_uncapped": (
                        quality_vs_uncapped
                    ),
                    "timings": timings,
                    "grouped_speedups": {
                        "static_vs_variable_payload": (
                            timings[
                                "variable_payload_prebuilt"
                            ]["critical_path_ms"]
                            / timings[
                                "static_payload_prebuilt"
                            ]["critical_path_ms"]
                        ),
                        "static_vs_variable_end_to_end": (
                            timings["variable_end_to_end"][
                                "critical_path_ms"
                            ]
                            / timings["static_end_to_end"][
                                "critical_path_ms"
                            ]
                        ),
                    },
                    "payload_crossover": payload_crossover,
                    "end_to_end_crossover": (
                        end_to_end_crossover
                    ),
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )
            del (
                variable_route,
                static_route,
                variable_output,
                static_output,
                masked_reference,
                uncapped_reference,
            )
            torch.cuda.empty_cache()

        del evaluation_route, hidden
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_profiled_transport",
            "profile": args.profile,
            "profile_order": (
                "descending"
                if args.reverse_profiles
                else "ascending"
            ),
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
                "Fixed source-destination split matrices are calibrated outside the measured step and reused without per-batch count exchange or CPU split extraction.",
                "Oracle matrix capacities equal the held-out evaluation counts and are a zero-padding upper bound, not a deployable profile.",
                "Calibrated capacities use the elementwise maximum over disjoint calibration batches plus fixed headroom, then run on a held-out batch.",
                "The additive-margin calibration is a robust control for low-count cells where multiplicative headroom remains small in absolute rows.",
                "Uniform exact capacity reproduces the v26 global maximum slot design inside the same benchmark.",
                "Static and dynamic controls retain the same highest router weights for every source-destination segment and must match exactly.",
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
    parser.add_argument(
        "--calibration-headroom",
        type=float,
        default=1.125,
    )
    parser.add_argument(
        "--calibration-margin",
        type=int,
        default=16,
    )
    parser.add_argument("--crossover-cycles", type=int, default=20)
    parser.add_argument(
        "--reverse-profiles",
        action="store_true",
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
