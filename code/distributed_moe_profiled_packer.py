from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
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
    masked_global_reference,
    run_variable_route,
)
from distributed_moe_profiled_transport import (
    CapacityMatrix,
    apply_host_capacity_row,
    calibrate_capacity_matrix,
    gather_segment_counts,
    make_hidden,
    margin_matrix,
    matrix_column_sums,
    matrix_row_sums,
    run_static_matrix_route,
)


@triton.jit
def _atomic_profile_pack_kernel(
    hidden_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    capacity_ptr,
    capacity_offsets_ptr,
    counters_ptr,
    overflow_ptr,
    send_hidden_ptr,
    send_local_ids_ptr,
    send_weights_ptr,
    fixed_token_indices_ptr,
    hidden_row_stride,
    send_row_stride,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    experts_per_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    assignment = tl.program_id(0)
    expert_id = tl.load(topk_ids_ptr + assignment).to(tl.int32)
    destination = expert_id // experts_per_rank
    local_id = expert_id - destination * experts_per_rank
    position = tl.atomic_add(counters_ptr + destination, 1)
    capacity = tl.load(capacity_ptr + destination)
    accepted = position < capacity
    fixed_index = (
        tl.load(capacity_offsets_ptr + destination)
        + position
    )
    token_index = assignment // top_k

    offsets = tl.arange(0, BLOCK_SIZE)
    hidden_mask = accepted & (offsets < hidden_size)
    hidden = tl.load(
        hidden_ptr
        + token_index * hidden_row_stride
        + offsets,
        mask=hidden_mask,
        other=0.0,
    )
    tl.store(
        send_hidden_ptr
        + fixed_index * send_row_stride
        + offsets,
        hidden,
        mask=hidden_mask,
    )
    weight = tl.load(topk_weights_ptr + assignment)
    tl.store(
        send_local_ids_ptr + fixed_index,
        local_id,
        mask=accepted,
    )
    tl.store(
        send_weights_ptr + fixed_index,
        weight,
        mask=accepted,
    )
    tl.store(
        fixed_token_indices_ptr + fixed_index,
        token_index,
        mask=accepted,
    )
    tl.atomic_add(
        overflow_ptr,
        1,
        mask=~accepted,
    )


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


def router_topk(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return topk_ids, topk_weights.to(hidden_states.dtype)


def profile_tensors(
    capacity_row: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor | int | list[int]]:
    capacity_i64 = torch.tensor(
        capacity_row,
        device=device,
        dtype=torch.int64,
    )
    offsets_i64 = (
        torch.cumsum(capacity_i64, dim=0)
        - capacity_i64
    )
    return {
        "capacity_row": list(capacity_row),
        "capacity_i64": capacity_i64,
        "offsets_i64": offsets_i64,
        "capacity_i32": capacity_i64.to(torch.int32),
        "offsets_i32": offsets_i64.to(torch.int32),
        "fixed_rows": sum(capacity_row),
    }


def assignment_tensors(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    top_k = topk_ids.shape[1]
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
    return (
        token_indices,
        flat_ids,
        flat_weights,
        destinations,
        local_ids,
    )


def materialize_sorted_route(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_indices: torch.Tensor,
    flat_weights: torch.Tensor,
    destinations: torch.Tensor,
    local_ids: torch.Tensor,
    grouped_order: torch.Tensor,
    profile: dict[str, torch.Tensor | int | list[int]],
) -> TensorDict:
    capacity_i64 = profile["capacity_i64"]
    offsets_i64 = profile["offsets_i64"]
    capacity_row = profile["capacity_row"]
    fixed_rows = profile["fixed_rows"]
    assert isinstance(capacity_i64, torch.Tensor)
    assert isinstance(offsets_i64, torch.Tensor)
    assert isinstance(capacity_row, list)
    assert isinstance(fixed_rows, int)

    grouped_destinations = destinations.index_select(
        0,
        grouped_order,
    )
    planned_counts = torch.bincount(
        grouped_destinations,
        minlength=capacity_i64.numel(),
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
    accepted_mask = positions < capacity_i64.index_select(
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
    fixed_indices = (
        offsets_i64.index_select(
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

    overflow_count = torch.clamp(
        planned_counts - capacity_i64,
        min=0,
    ).sum()
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
            capacity_i64,
        ),
        "send_counts_tensor": capacity_i64,
        "send_counts": list(capacity_row),
        "overflow_count": overflow_count,
    }


def pack_two_sort(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    profile: dict[str, torch.Tensor | int | list[int]],
) -> TensorDict:
    (
        token_indices,
        _,
        flat_weights,
        destinations,
        local_ids,
    ) = assignment_tensors(
        hidden_states,
        topk_ids,
        topk_weights,
        experts_per_rank=experts_per_rank,
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
    return materialize_sorted_route(
        hidden_states,
        topk_ids,
        topk_weights,
        token_indices,
        flat_weights,
        destinations,
        local_ids,
        grouped_order,
        profile,
    )


def pack_destination_sort(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    profile: dict[str, torch.Tensor | int | list[int]],
) -> TensorDict:
    (
        token_indices,
        _,
        flat_weights,
        destinations,
        local_ids,
    ) = assignment_tensors(
        hidden_states,
        topk_ids,
        topk_weights,
        experts_per_rank=experts_per_rank,
    )
    grouped_order = torch.argsort(
        destinations,
        stable=True,
    )
    return materialize_sorted_route(
        hidden_states,
        topk_ids,
        topk_weights,
        token_indices,
        flat_weights,
        destinations,
        local_ids,
        grouped_order,
        profile,
    )


def pack_triton_atomic(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    profile: dict[str, torch.Tensor | int | list[int]],
) -> TensorDict:
    capacity_row = profile["capacity_row"]
    capacity_i64 = profile["capacity_i64"]
    capacity_i32 = profile["capacity_i32"]
    offsets_i32 = profile["offsets_i32"]
    fixed_rows = profile["fixed_rows"]
    assert isinstance(capacity_row, list)
    assert isinstance(capacity_i64, torch.Tensor)
    assert isinstance(capacity_i32, torch.Tensor)
    assert isinstance(offsets_i32, torch.Tensor)
    assert isinstance(fixed_rows, int)

    num_tokens, top_k = topk_ids.shape
    num_assignments = topk_ids.numel()
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
    counters = torch.zeros(
        len(capacity_row),
        device=hidden_states.device,
        dtype=torch.int32,
    )
    overflow_count = torch.zeros(
        (),
        device=hidden_states.device,
        dtype=torch.int32,
    )
    _atomic_profile_pack_kernel[(num_assignments,)](
        hidden_states,
        topk_ids,
        topk_weights,
        capacity_i32,
        offsets_i32,
        counters,
        overflow_count,
        send_hidden,
        send_local_ids,
        send_weights,
        fixed_token_indices,
        hidden_states.stride(0),
        send_hidden.stride(0),
        hidden_size=hidden_states.shape[1],
        top_k=top_k,
        experts_per_rank=experts_per_rank,
        BLOCK_SIZE=triton.next_power_of_2(
            hidden_states.shape[1]
        ),
        num_warps=4,
    )

    selected_assignment_indices = torch.arange(
        num_assignments,
        device=hidden_states.device,
        dtype=torch.int64,
    )
    selected_token_indices = torch.arange(
        num_tokens,
        device=hidden_states.device,
        dtype=torch.int64,
    ).repeat_interleave(top_k)
    selected_weights = topk_weights.reshape(-1)
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
            counters.to(torch.int64),
            capacity_i64,
        ),
        "send_counts_tensor": capacity_i64,
        "send_counts": list(capacity_row),
        "overflow_count": overflow_count.to(torch.int64),
    }


def build_with_router(
    packer: Callable[..., TensorDict],
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    *,
    top_k: int,
    experts_per_rank: int,
    profile: dict[str, torch.Tensor | int | list[int]],
) -> TensorDict:
    topk_ids, topk_weights = router_topk(
        hidden_states,
        gate_weight,
        bias,
        tie_breaker,
        top_k=top_k,
    )
    return packer(
        hidden_states,
        topk_ids,
        topk_weights,
        experts_per_rank=experts_per_rank,
        profile=profile,
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
    num_experts = world_size * args.experts_per_rank
    assignments_per_rank = num_tokens * args.top_k

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

    packers: tuple[
        tuple[str, Callable[..., TensorDict]],
        ...,
    ] = (
        ("two_sort", pack_two_sort),
        ("destination_sort", pack_destination_sort),
        ("triton_atomic", pack_triton_atomic),
    )
    if args.reverse_packers:
        packers = tuple(reversed(packers))

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
        capacity_matrix = margin_matrix(
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
        capacity_row = capacity_matrix[rank]
        recv_counts = [
            capacity_matrix[source][rank]
            for source in range(world_size)
        ]
        profile = profile_tensors(
            capacity_row,
            device,
        )
        prebuilt_topk_ids, prebuilt_topk_weights = router_topk(
            hidden,
            gate_weight,
            bias,
            tie_breaker,
            top_k=args.top_k,
        )

        def build_dynamic() -> TensorDict:
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

        pack_only_functions: dict[
            str,
            Callable[[], TensorDict],
        ] = {}
        route_functions: dict[
            str,
            Callable[[], TensorDict],
        ] = {}
        end_to_end_functions: dict[
            str,
            Callable[[], torch.Tensor],
        ] = {}
        for packer_name, packer in packers:
            def pack_only(
                selected_packer: Callable[..., TensorDict] = packer,
            ) -> TensorDict:
                return selected_packer(
                    hidden,
                    prebuilt_topk_ids,
                    prebuilt_topk_weights,
                    experts_per_rank=args.experts_per_rank,
                    profile=profile,
                )

            def route_capacity(
                selected_packer: Callable[..., TensorDict] = packer,
            ) -> TensorDict:
                return build_with_router(
                    selected_packer,
                    hidden,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    profile=profile,
                )

            def end_to_end(
                selected_route: Callable[
                    [], TensorDict
                ] = route_capacity,
            ) -> torch.Tensor:
                return run_static_matrix_route(
                    selected_route(),
                    recv_counts,
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            pack_only_functions[packer_name] = pack_only
            route_functions[packer_name] = route_capacity
            end_to_end_functions[packer_name] = end_to_end

        def dynamic_end_to_end() -> torch.Tensor:
            return run_variable_route(
                build_dynamic(),
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )

        prepared_dynamic = build_dynamic()
        prepared_routes = {
            name: function()
            for name, function in route_functions.items()
        }
        overflow_by_packer = {
            name: max(
                gather_int(
                    int(
                        route["overflow_count"].item()
                    ),
                    device,
                )
            )
            for name, route in prepared_routes.items()
        }
        if any(overflow_by_packer.values()):
            raise RuntimeError(
                "Lossless profile overflowed: "
                f"{overflow_by_packer}"
            )

        dynamic_output = run_variable_route(
            prepared_dynamic,
            local_w1,
            local_w2,
            num_tokens=num_tokens,
        )
        static_outputs = {
            name: run_static_matrix_route(
                route,
                recv_counts,
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )
            for name, route in prepared_routes.items()
        }
        masked_reference = masked_global_reference(
            hidden,
            global_w1,
            global_w2,
            prepared_routes["two_sort"],
        )
        uncapped_reference = fused_experts(
            hidden,
            global_w1,
            global_w2,
            prebuilt_topk_weights,
            prebuilt_topk_ids,
        )
        torch.cuda.synchronize()

        dynamic_selected = prepared_dynamic[
            "selected_assignment_indices"
        ]
        assert isinstance(dynamic_selected, torch.Tensor)
        selection_equal: dict[str, bool] = {}
        correctness: dict[str, dict[str, float]] = {
            "dynamic_vs_masked_reference": output_quality(
                dynamic_output,
                masked_reference,
            )
        }
        for name, route in prepared_routes.items():
            selected = route[
                "selected_assignment_indices"
            ]
            assert isinstance(selected, torch.Tensor)
            local_equal = torch.equal(
                torch.sort(selected).values,
                torch.sort(dynamic_selected).values,
            )
            selection_equal[name] = (
                min(
                    gather_scalar(
                        float(local_equal),
                        device,
                    )
                )
                == 1.0
            )
            correctness[
                f"{name}_vs_masked_reference"
            ] = output_quality(
                static_outputs[name],
                masked_reference,
            )
            correctness[
                f"{name}_vs_dynamic"
            ] = output_quality(
                static_outputs[name],
                dynamic_output,
            )
        correctness[
            "triton_atomic_vs_two_sort"
        ] = output_quality(
            static_outputs["triton_atomic"],
            static_outputs["two_sort"],
        )
        correctness[
            "destination_sort_vs_two_sort"
        ] = output_quality(
            static_outputs["destination_sort"],
            static_outputs["two_sort"],
        )
        if (
            not all(selection_equal.values())
            or any(
                quality["max_abs_error"] != 0.0
                for quality in correctness.values()
            )
        ):
            raise RuntimeError(
                "Profiled packer correctness failed: "
                f"selection={selection_equal}, "
                f"correctness={correctness}"
            )

        accepted_counts_tensor = prepared_routes[
            "two_sort"
        ]["accepted_counts_tensor"]
        selected_weights = prepared_routes[
            "two_sort"
        ]["selected_weights"]
        assert isinstance(
            accepted_counts_tensor,
            torch.Tensor,
        )
        assert isinstance(selected_weights, torch.Tensor)
        accepted_row = [
            int(value)
            for value in accepted_counts_tensor.cpu().tolist()
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
        if accepted_global != planned_global:
            raise RuntimeError(
                "Calibrated +margin profile was not lossless: "
                f"accepted={accepted_global}, "
                f"planned={planned_global}"
            )
        planned_weight_global = sum(
            gather_scalar(
                float(
                    prebuilt_topk_weights.float().sum().item()
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

        pack_only_timings = {
            name: critical_path_summary(
                timed_schedule(
                    function,
                    warmup=warmup,
                    iterations=iterations,
                    local_rank=local_rank,
                ),
                device,
            )
            for name, function in pack_only_functions.items()
        }
        route_timings = {
            "dynamic": critical_path_summary(
                timed_schedule(
                    build_dynamic,
                    warmup=warmup,
                    iterations=iterations,
                    local_rank=local_rank,
                ),
                device,
            ),
            **{
                name: critical_path_summary(
                    timed_schedule(
                        function,
                        warmup=warmup,
                        iterations=iterations,
                        local_rank=local_rank,
                    ),
                    device,
                )
                for name, function in route_functions.items()
            },
        }
        end_to_end_timings = {
            "dynamic": critical_path_summary(
                timed_schedule(
                    dynamic_end_to_end,
                    warmup=warmup,
                    iterations=iterations,
                    local_rank=local_rank,
                ),
                device,
            ),
            **{
                name: critical_path_summary(
                    timed_schedule(
                        function,
                        warmup=warmup,
                        iterations=iterations,
                        local_rank=local_rank,
                    ),
                    device,
                )
                for name, function in (
                    end_to_end_functions.items()
                )
            },
        }

        pack_only_crossover = None
        route_crossover = None
        end_to_end_crossover = None
        if args.profile == "crossover":
            pack_only_crossover = timed_crossover(
                tuple(pack_only_functions.items()),
                warmup_cycles=6,
                measured_cycles=args.crossover_cycles,
                local_rank=local_rank,
                device=device,
            )
            route_crossover = timed_crossover(
                (
                    ("dynamic", build_dynamic),
                    *tuple(route_functions.items()),
                ),
                warmup_cycles=8,
                measured_cycles=args.crossover_cycles,
                local_rank=local_rank,
                device=device,
            )
            end_to_end_crossover = timed_crossover(
                (
                    ("dynamic", dynamic_end_to_end),
                    *tuple(end_to_end_functions.items()),
                ),
                warmup_cycles=8,
                measured_cycles=args.crossover_cycles,
                local_rank=local_rank,
                device=device,
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
                "profile_name": "calibrated_plus_margin",
                "calibration_batches": calibration_batches,
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
                    "dropped_assignment_fraction": 0.0,
                    "retained_router_weight_fraction": (
                        accepted_weight_global
                        / planned_weight_global
                    ),
                    "lossless": True,
                },
                "transport": {
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
                },
                "packers": {
                    "two_sort": (
                        "stable weight sort followed by stable destination sort"
                    ),
                    "destination_sort": (
                        "lossless-only stable destination sort with no weight ordering"
                    ),
                    "triton_atomic": (
                        "lossless-only fused Triton atomic histogram and hidden/metadata scatter"
                    ),
                },
                "overflow_by_packer": (
                    overflow_by_packer
                ),
                "correctness": {
                    "selection_equal": (
                        selection_equal
                    ),
                    "output": correctness,
                },
                "quality_vs_uncapped": output_quality(
                    static_outputs["two_sort"],
                    uncapped_reference,
                ),
                "timings": {
                    "pack_only": pack_only_timings,
                    "route_capacity": route_timings,
                    "end_to_end": end_to_end_timings,
                },
                "grouped_speedups": {
                    "destination_sort_vs_two_sort_pack": (
                        pack_only_timings["two_sort"][
                            "critical_path_ms"
                        ]
                        / pack_only_timings[
                            "destination_sort"
                        ]["critical_path_ms"]
                    ),
                    "triton_atomic_vs_two_sort_pack": (
                        pack_only_timings["two_sort"][
                            "critical_path_ms"
                        ]
                        / pack_only_timings[
                            "triton_atomic"
                        ]["critical_path_ms"]
                    ),
                    "destination_sort_vs_two_sort_route": (
                        route_timings["two_sort"][
                            "critical_path_ms"
                        ]
                        / route_timings[
                            "destination_sort"
                        ]["critical_path_ms"]
                    ),
                    "triton_atomic_vs_two_sort_route": (
                        route_timings["two_sort"][
                            "critical_path_ms"
                        ]
                        / route_timings[
                            "triton_atomic"
                        ]["critical_path_ms"]
                    ),
                    "destination_sort_vs_two_sort_e2e": (
                        end_to_end_timings["two_sort"][
                            "critical_path_ms"
                        ]
                        / end_to_end_timings[
                            "destination_sort"
                        ]["critical_path_ms"]
                    ),
                    "triton_atomic_vs_two_sort_e2e": (
                        end_to_end_timings["two_sort"][
                            "critical_path_ms"
                        ]
                        / end_to_end_timings[
                            "triton_atomic"
                        ]["critical_path_ms"]
                    ),
                    "two_sort_vs_dynamic_e2e": (
                        end_to_end_timings["dynamic"][
                            "critical_path_ms"
                        ]
                        / end_to_end_timings[
                            "two_sort"
                        ]["critical_path_ms"]
                    ),
                    "destination_sort_vs_dynamic_e2e": (
                        end_to_end_timings["dynamic"][
                            "critical_path_ms"
                        ]
                        / end_to_end_timings[
                            "destination_sort"
                        ]["critical_path_ms"]
                    ),
                    "triton_atomic_vs_dynamic_e2e": (
                        end_to_end_timings["dynamic"][
                            "critical_path_ms"
                        ]
                        / end_to_end_timings[
                            "triton_atomic"
                        ]["critical_path_ms"]
                    ),
                },
                "crossover": {
                    "pack_only": pack_only_crossover,
                    "route_capacity": route_crossover,
                    "end_to_end": end_to_end_crossover,
                },
                "warmup": warmup,
                "iterations": iterations,
            }
        )
        del (
            evaluation_route,
            hidden,
            prebuilt_topk_ids,
            prebuilt_topk_weights,
            prepared_dynamic,
            prepared_routes,
            dynamic_output,
            static_outputs,
            masked_reference,
            uncapped_reference,
        )
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_profiled_packer",
            "profile": args.profile,
            "packer_order": (
                "descending"
                if args.reverse_packers
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
                "The experiment uses the lossless calibrated additive-margin matrix from the profiled transport study.",
                "The two-sort baseline preserves the v28 highest-weight clipping implementation even though the measured profile is lossless.",
                "Destination sort removes weight ordering and is valid only when every source-destination count fits its fixed profile.",
                "The Triton path uses one program per assignment, an atomic destination counter, and fused hidden/ID/weight/token-index scatter.",
                "Packing-only timing reuses router outputs; route timing includes router projection and fused top-k; end-to-end timing adds fixed dispatch, experts, return, and combine.",
                "All optimized routes must report zero overflow, retain the same assignments, and match the dynamic and masked outputs exactly.",
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
        "--calibration-margin",
        type=int,
        default=16,
    )
    parser.add_argument("--crossover-cycles", type=int, default=24)
    parser.add_argument(
        "--reverse-packers",
        action="store_true",
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
