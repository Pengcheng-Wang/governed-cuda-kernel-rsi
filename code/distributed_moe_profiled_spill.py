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

import numpy as np
import torch
import torch.distributed as dist

from analyze_router_profile_drift import (
    build_profile,
    destination_steps,
)
from distributed_moe_end_to_end import (
    TensorDict,
    gather_scalar,
    local_experts,
    output_quality,
)
from distributed_moe_packed_dispatch import (
    build_metadata_packet,
    dispatch_assignments_metadata_packed,
    timed_crossover,
)
from distributed_moe_profiled_packer import (
    profile_tensors,
)
from distributed_moe_profiled_packer_scaling import (
    pack_triton_tiled,
)


def reconstruct_topk_ids(
    expert_counts: list[int],
    *,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    remaining = [int(value) for value in expert_counts]
    if sum(remaining) != num_tokens * top_k:
        raise ValueError("Expert counts do not match token assignments")
    rows: list[list[int]] = []
    num_experts = len(remaining)
    for token in range(num_tokens):
        order = sorted(
            range(num_experts),
            key=lambda expert: (
                -remaining[expert],
                (expert - token) % num_experts,
            ),
        )
        selected = [
            expert
            for expert in order
            if remaining[expert] > 0
        ][:top_k]
        if len(selected) != top_k:
            raise RuntimeError(
                "Could not reconstruct unique top-k assignments"
            )
        rows.append(selected)
        for expert in selected:
            remaining[expert] -= 1
    if any(remaining):
        raise RuntimeError(
            f"Unassigned expert counts remain: {remaining}"
        )
    return torch.tensor(rows, dtype=torch.int64)


def remap_expert_ids(
    expert_ids: torch.Tensor,
    *,
    num_trace_experts: int,
    world_size: int,
) -> tuple[torch.Tensor, int]:
    mapping = [
        min(
            expert * world_size // num_trace_experts,
            world_size - 1,
        )
        for expert in range(num_trace_experts)
    ]
    local_index: dict[int, int] = {}
    per_destination: dict[int, int] = {}
    for expert, destination in enumerate(mapping):
        local_index[expert] = per_destination.get(
            destination,
            0,
        )
        per_destination[destination] = (
            local_index[expert] + 1
        )
    experts_per_rank = max(per_destination.values())
    remap = torch.tensor(
        [
            mapping[expert] * experts_per_rank
            + local_index[expert]
            for expert in range(num_trace_experts)
        ],
        dtype=torch.int64,
    )
    return remap.index_select(
        0,
        expert_ids.reshape(-1),
    ).reshape_as(expert_ids), experts_per_rank


def trace_step_indices(
    trace: dict[str, Any],
    world_size: int,
) -> tuple[list[str], list[list[int]]]:
    labels: list[str] = []
    sample_indices: list[list[int]] = []
    for domain_range in trace["domain_ranges"]:
        domain = domain_range["domain"]
        start = int(domain_range["start"])
        stop = int(domain_range["stop"])
        for offset in range(start, stop, world_size):
            labels.append(domain)
            sample_indices.append(
                list(range(offset, offset + world_size))
            )
    return labels, sample_indices


def pack_dynamic_from_topk(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    world_size: int,
) -> TensorDict:
    num_tokens, top_k = topk_ids.shape
    token_indices = torch.arange(
        num_tokens,
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
    order = torch.argsort(destinations, stable=True)
    sorted_destinations = destinations.index_select(
        0,
        order,
    )
    sorted_tokens = token_indices.index_select(0, order)
    send_counts_tensor = torch.bincount(
        sorted_destinations,
        minlength=world_size,
    ).to(torch.int64)
    return {
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "send_hidden": hidden_states.index_select(
            0,
            sorted_tokens,
        ).contiguous(),
        "send_local_ids": local_ids.index_select(
            0,
            order,
        ).to(torch.int32).reshape(-1, 1).contiguous(),
        "send_weights": flat_weights.index_select(
            0,
            order,
        ).reshape(-1, 1).contiguous(),
        "sorted_token_indices": sorted_tokens,
        "sorted_assignment_indices": order,
        "send_counts_tensor": send_counts_tensor,
        "send_counts": [
            int(value)
            for value in send_counts_tensor.cpu().tolist()
        ],
    }


def build_spill_route(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fixed_indices: torch.Tensor,
    *,
    experts_per_rank: int,
    world_size: int,
) -> TensorDict:
    top_k = topk_ids.shape[1]
    assignment_indices = torch.nonzero(
        fixed_indices < 0,
        as_tuple=False,
    ).reshape(-1)
    flat_ids = topk_ids.reshape(-1).index_select(
        0,
        assignment_indices,
    )
    flat_weights = topk_weights.reshape(-1).index_select(
        0,
        assignment_indices,
    )
    token_indices = torch.div(
        assignment_indices,
        top_k,
        rounding_mode="floor",
    )
    destinations = torch.div(
        flat_ids,
        experts_per_rank,
        rounding_mode="floor",
    )
    local_ids = torch.remainder(
        flat_ids,
        experts_per_rank,
    )
    order = torch.argsort(destinations, stable=True)
    sorted_destinations = destinations.index_select(
        0,
        order,
    )
    sorted_tokens = token_indices.index_select(0, order)
    sorted_assignments = assignment_indices.index_select(
        0,
        order,
    )
    send_counts_tensor = torch.bincount(
        sorted_destinations,
        minlength=world_size,
    ).to(torch.int64)
    return {
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "send_hidden": hidden_states.index_select(
            0,
            sorted_tokens,
        ).contiguous(),
        "send_local_ids": local_ids.index_select(
            0,
            order,
        ).to(torch.int32).reshape(-1, 1).contiguous(),
        "send_weights": flat_weights.index_select(
            0,
            order,
        ).reshape(-1, 1).contiguous(),
        "sorted_token_indices": sorted_tokens,
        "sorted_assignment_indices": sorted_assignments,
        "send_counts_tensor": send_counts_tensor,
        "send_counts": [
            int(value)
            for value in send_counts_tensor.cpu().tolist()
        ],
    }


def dispatch_static_metadata(
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
            recv_metadata[:, :1].to(torch.int32).contiguous()
        ),
        "recv_weights": recv_metadata[:, 1:].contiguous(),
        "recv_counts": list(recv_counts),
    }


def return_static_fp32(
    expert_output: torch.Tensor,
    route: TensorDict,
    recv_counts: list[int],
    *,
    num_tokens: int,
) -> torch.Tensor:
    send_hidden = route["send_hidden"]
    send_counts = route["send_counts"]
    fixed_token_indices = route["fixed_token_indices"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts, list)
    assert isinstance(fixed_token_indices, torch.Tensor)
    returned = torch.empty_like(send_hidden)
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
    return combined


def return_dynamic_fp32(
    expert_output: torch.Tensor,
    route: TensorDict,
    dispatched: TensorDict,
    *,
    num_tokens: int,
) -> torch.Tensor:
    send_hidden = route["send_hidden"]
    sorted_token_indices = route["sorted_token_indices"]
    send_counts = route["send_counts"]
    recv_counts = dispatched["recv_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(sorted_token_indices, torch.Tensor)
    assert isinstance(send_counts, list)
    assert isinstance(recv_counts, list)
    returned = torch.empty_like(send_hidden)
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
        sorted_token_indices,
        returned.float(),
    )
    return combined


def run_dynamic(
    route: TensorDict,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    dispatched = dispatch_assignments_metadata_packed(route)
    expert_output = local_experts(
        dispatched,
        local_w1,
        local_w2,
    )
    return return_dynamic_fp32(
        expert_output,
        route,
        dispatched,
        num_tokens=num_tokens,
    ).to(local_w1.dtype)


def run_hybrid(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    world_size: int,
    profile: dict[str, torch.Tensor | int | list[int]],
    recv_counts: list[int],
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    tile_size: int,
) -> tuple[torch.Tensor, int]:
    fixed_route = pack_triton_tiled(
        hidden_states,
        topk_ids,
        topk_weights,
        experts_per_rank=experts_per_rank,
        profile=profile,
        tile_size=tile_size,
    )
    fixed_indices = fixed_route["fixed_indices"]
    assert isinstance(fixed_indices, torch.Tensor)
    spill_route = build_spill_route(
        hidden_states,
        topk_ids,
        topk_weights,
        fixed_indices,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
    )
    dispatched = dispatch_static_metadata(
        fixed_route,
        recv_counts,
    )
    fixed_expert_output = local_experts(
        dispatched,
        local_w1,
        local_w2,
    )
    combined = return_static_fp32(
        fixed_expert_output,
        fixed_route,
        recv_counts,
        num_tokens=hidden_states.shape[0],
    )

    local_spill = spill_route["send_hidden"].shape[0]
    global_spill = torch.tensor(
        local_spill,
        device=hidden_states.device,
        dtype=torch.int64,
    )
    dist.all_reduce(global_spill, op=dist.ReduceOp.SUM)
    global_spill_count = int(global_spill.item())
    if global_spill_count:
        spill_dispatched = (
            dispatch_assignments_metadata_packed(
                spill_route
            )
        )
        spill_expert_output = local_experts(
            spill_dispatched,
            local_w1,
            local_w2,
        )
        combined += return_dynamic_fp32(
            spill_expert_output,
            spill_route,
            spill_dispatched,
            num_tokens=hidden_states.shape[0],
        )
    return combined.to(hidden_states.dtype), global_spill_count


def profile_schedules(
    matrices: np.ndarray,
    *,
    calibration_steps: int,
    selected_steps: list[int],
    assignments_per_source: int,
    additive_margin: int,
    adaptive_window: int,
) -> dict[str, list[np.ndarray]]:
    num_layers = matrices.shape[1]
    schedules: dict[str, list[np.ndarray]] = {
        "additive_spill": [],
        "risk0_spill": [],
        "risk8_spill": [],
    }
    for layer in range(num_layers):
        calibration = matrices[:calibration_steps, layer]
        additive = calibration.max(axis=0) + additive_margin
        history = [
            matrix.copy()
            for matrix in calibration
        ]
        step_profiles: dict[
            int,
            tuple[np.ndarray, np.ndarray],
        ] = {}
        for step in range(calibration_steps, matrices.shape[0]):
            window = np.stack(
                history[-adaptive_window:],
                axis=0,
            )
            step_profiles[step] = (
                build_profile(
                    window,
                    assignments_per_source=(
                        assignments_per_source
                    ),
                    budget_fraction=0.0,
                    risk_weighted=True,
                ),
                build_profile(
                    window,
                    assignments_per_source=(
                        assignments_per_source
                    ),
                    budget_fraction=0.08,
                    risk_weighted=True,
                ),
            )
            history.append(matrices[step, layer].copy())
        for step in selected_steps:
            risk0, risk8 = step_profiles[step]
            schedules["additive_spill"].append(additive)
            schedules["risk0_spill"].append(risk0)
            schedules["risk8_spill"].append(risk8)
    return schedules


def summarize_profile_rows(
    matrices: np.ndarray,
    *,
    selected_layers: list[int],
    selected_steps: list[int],
    schedules: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method, profiles in schedules.items():
        fixed_ratios = []
        spill_fractions = []
        spill_steps = []
        index = 0
        for layer in selected_layers:
            for step in selected_steps:
                actual = matrices[step, layer]
                profile = profiles[index]
                overflow = np.maximum(actual - profile, 0)
                assignments = int(actual.sum())
                fixed = int(profile.sum())
                spilled = int(overflow.sum())
                fixed_ratios.append(fixed / assignments)
                spill_fractions.append(spilled / assignments)
                spill_steps.append(float(spilled > 0))
                index += 1
        summary[method] = {
            "mean_fixed_row_ratio": statistics.fmean(
                fixed_ratios
            ),
            "mean_spill_fraction": statistics.fmean(
                spill_fractions
            ),
            "spill_step_fraction": statistics.fmean(
                spill_steps
            ),
            "mean_total_row_ratio": statistics.fmean(
                fixed + spill
                for fixed, spill in zip(
                    fixed_ratios,
                    spill_fractions,
                    strict=True,
                )
            ),
        }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    trace = json.loads(Path(args.trace).read_text())
    labels, matrices = destination_steps(
        trace,
        world_size,
    )
    _, sample_indices = trace_step_indices(
        trace,
        world_size,
    )
    steps_per_domain = (
        int(trace["samples_per_domain"]) // world_size
    )
    calibration_steps = (
        args.calibration_domains * steps_per_domain
    )
    all_evaluation_steps = list(
        range(calibration_steps, len(labels))
    )
    if args.profile == "smoke":
        selected_layers = [0]
        selected_steps = all_evaluation_steps[:2]
        warmup_cycles, measured_cycles = 1, 2
    elif args.profile == "crossover":
        selected_layers = [0, 8, 15]
        selected_steps = all_evaluation_steps
        warmup_cycles, measured_cycles = 1, 4
    else:
        selected_layers = [0, 4, 8, 12, 15]
        selected_steps = all_evaluation_steps
        warmup_cycles, measured_cycles = 1, 3

    assignments_per_source = (
        int(trace["sequence_length"])
        * int(trace["top_k"])
    )
    adaptive_window = (
        args.adaptive_window_domains
        * steps_per_domain
    )
    additive_margin = (
        args.margin_8
        if world_size <= 8
        else args.margin_24
    )
    full_schedules = profile_schedules(
        matrices,
        calibration_steps=calibration_steps,
        selected_steps=selected_steps,
        assignments_per_source=assignments_per_source,
        additive_margin=additive_margin,
        adaptive_window=adaptive_window,
    )
    schedules = {
        method: [
            profiles[
                layer * len(selected_steps)
                + step_index
            ]
            for layer in selected_layers
            for step_index, _ in enumerate(selected_steps)
        ]
        for method, profiles in full_schedules.items()
    }

    trace_counts = trace["expert_counts"]
    num_tokens = int(trace["sequence_length"])
    top_k = int(trace["top_k"])
    num_trace_experts = int(trace["num_experts"])
    remapped_example, experts_per_rank = remap_expert_ids(
        torch.arange(num_trace_experts).reshape(-1, 1),
        num_trace_experts=num_trace_experts,
        world_size=world_size,
    )
    del remapped_example

    generator = torch.Generator(device=device)
    generator.manual_seed(7_000)
    local_w1 = (
        torch.randn(
            (
                experts_per_rank,
                2 * args.intermediate,
                args.hidden,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.hidden)
    )
    local_w2 = (
        torch.randn(
            (
                experts_per_rank,
                args.hidden,
                args.intermediate,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.intermediate)
    )

    cases: list[dict[str, Any]] = []
    schedule_index = 0
    for layer in selected_layers:
        for step in selected_steps:
            sample = sample_indices[step][rank]
            original_ids = reconstruct_topk_ids(
                trace_counts[sample][layer],
                num_tokens=num_tokens,
                top_k=top_k,
            )
            topk_ids, actual_experts_per_rank = (
                remap_expert_ids(
                    original_ids,
                    num_trace_experts=num_trace_experts,
                    world_size=world_size,
                )
            )
            if actual_experts_per_rank != experts_per_rank:
                raise RuntimeError(
                    "Inconsistent local expert count"
                )
            topk_ids = topk_ids.to(device)
            topk_weights = torch.full(
                topk_ids.shape,
                1.0 / top_k,
                device=device,
                dtype=dtype,
            )
            hidden_generator = torch.Generator(device=device)
            hidden_generator.manual_seed(
                100_000
                + rank * 10_003
                + layer * 1_003
                + step
            )
            hidden = torch.randn(
                (num_tokens, args.hidden),
                device=device,
                dtype=dtype,
                generator=hidden_generator,
            )
            method_profiles = {}
            for method, profiles in schedules.items():
                matrix = profiles[schedule_index]
                capacity_row = [
                    int(value)
                    for value in matrix[rank].tolist()
                ]
                method_profiles[method] = {
                    "profile": profile_tensors(
                        capacity_row,
                        device,
                    ),
                    "recv_counts": [
                        int(matrix[source, rank])
                        for source in range(world_size)
                    ],
                }
            cases.append(
                {
                    "layer": layer,
                    "step": step,
                    "domain": labels[step],
                    "hidden": hidden,
                    "topk_ids": topk_ids,
                    "topk_weights": topk_weights,
                    "profiles": method_profiles,
                }
            )
            schedule_index += 1

    quality: dict[str, dict[str, float]] = {}
    partition_exact = True
    for case in cases:
        dynamic_route = pack_dynamic_from_topk(
            case["hidden"],
            case["topk_ids"],
            case["topk_weights"],
            experts_per_rank=experts_per_rank,
            world_size=world_size,
        )
        expected = run_dynamic(
            dynamic_route,
            local_w1,
            local_w2,
            num_tokens=num_tokens,
        )
        for method in (
            "additive_spill",
            "risk0_spill",
            "risk8_spill",
        ):
            profile_data = case["profiles"][method]
            partition_route = pack_triton_tiled(
                case["hidden"],
                case["topk_ids"],
                case["topk_weights"],
                experts_per_rank=experts_per_rank,
                profile=profile_data["profile"],
                tile_size=args.tile_size,
            )
            fixed_indices = partition_route["fixed_indices"]
            overflow_count = partition_route["overflow_count"]
            assert isinstance(fixed_indices, torch.Tensor)
            assert isinstance(overflow_count, torch.Tensor)
            accepted_assignments = torch.nonzero(
                fixed_indices >= 0,
                as_tuple=False,
            ).reshape(-1)
            spill_partition = build_spill_route(
                case["hidden"],
                case["topk_ids"],
                case["topk_weights"],
                fixed_indices,
                experts_per_rank=experts_per_rank,
                world_size=world_size,
            )
            spill_assignments = spill_partition[
                "sorted_assignment_indices"
            ]
            assert isinstance(
                spill_assignments,
                torch.Tensor,
            )
            partition = torch.sort(
                torch.cat(
                    (
                        accepted_assignments,
                        spill_assignments,
                    )
                )
            ).values
            local_partition_exact = (
                int(overflow_count.item())
                == spill_assignments.numel()
                and torch.equal(
                    partition,
                    torch.arange(
                        case["topk_ids"].numel(),
                        device=device,
                        dtype=torch.int64,
                    ),
                )
            )
            partition_exact = (
                partition_exact
                and min(
                    gather_scalar(
                        float(local_partition_exact),
                        device,
                    )
                )
                == 1.0
            )
            actual, _ = run_hybrid(
                case["hidden"],
                case["topk_ids"],
                case["topk_weights"],
                experts_per_rank=experts_per_rank,
                world_size=world_size,
                profile=profile_data["profile"],
                recv_counts=profile_data["recv_counts"],
                local_w1=local_w1,
                local_w2=local_w2,
                tile_size=args.tile_size,
            )
            local_quality = output_quality(actual, expected)
            key = (
                f"{method}_layer{case['layer']}_"
                f"step{case['step']}"
            )
            quality[key] = {
                "max_abs_error": max(
                    gather_scalar(
                        local_quality["max_abs_error"],
                        device,
                    )
                ),
                "normalized_rmse": max(
                    gather_scalar(
                        local_quality["normalized_rmse"],
                        device,
                    )
                ),
                "cosine_similarity": min(
                    gather_scalar(
                        local_quality["cosine_similarity"],
                        device,
                    )
                ),
            }
        del dynamic_route, expected
    max_error = max(
        value["max_abs_error"]
        for value in quality.values()
    )
    min_cosine = min(
        value["cosine_similarity"]
        for value in quality.values()
    )
    if max_error > args.max_abs_error or min_cosine < 0.999:
        raise RuntimeError(
            "Spill correctness failed: "
            f"max_error={max_error}, min_cosine={min_cosine}"
        )

    def dynamic_cycle() -> torch.Tensor:
        output = torch.empty(
            0,
            device=device,
            dtype=dtype,
        )
        for case in cases:
            route = pack_dynamic_from_topk(
                case["hidden"],
                case["topk_ids"],
                case["topk_weights"],
                experts_per_rank=experts_per_rank,
                world_size=world_size,
            )
            output = run_dynamic(
                route,
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )
        return output

    def hybrid_cycle(method: str) -> torch.Tensor:
        output = torch.empty(
            0,
            device=device,
            dtype=dtype,
        )
        for case in cases:
            profile_data = case["profiles"][method]
            output, _ = run_hybrid(
                case["hidden"],
                case["topk_ids"],
                case["topk_weights"],
                experts_per_rank=experts_per_rank,
                world_size=world_size,
                profile=profile_data["profile"],
                recv_counts=profile_data["recv_counts"],
                local_w1=local_w1,
                local_w2=local_w2,
                tile_size=args.tile_size,
            )
        return output

    methods: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("dynamic", dynamic_cycle),
        (
            "additive_spill",
            lambda: hybrid_cycle("additive_spill"),
        ),
        (
            "risk0_spill",
            lambda: hybrid_cycle("risk0_spill"),
        ),
        (
            "risk8_spill",
            lambda: hybrid_cycle("risk8_spill"),
        ),
    )
    if args.reverse_methods:
        methods = tuple(reversed(methods))
    timings = timed_crossover(
        methods,
        warmup_cycles=warmup_cycles,
        measured_cycles=measured_cycles,
        local_rank=local_rank,
        device=device,
    )

    result = None
    if rank == 0:
        profile_rows = summarize_profile_rows(
            matrices,
            selected_layers=selected_layers,
            selected_steps=selected_steps,
            schedules=schedules,
        )
        dynamic_ms = timings["dynamic"]["critical_path_ms"]
        result = {
            "suite": "distributed_moe_profiled_spill",
            "profile": args.profile,
            "method_order": (
                "descending"
                if args.reverse_methods
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
            "trace_model": trace["model"],
            "trace_corpus_sha256": trace["corpus_sha256"],
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "num_tokens": num_tokens,
            "top_k": top_k,
            "trace_experts": num_trace_experts,
            "experts_per_rank": experts_per_rank,
            "selected_layers": selected_layers,
            "selected_steps": selected_steps,
            "selected_domains": [
                labels[step]
                for step in selected_steps
            ],
            "calibration_steps": calibration_steps,
            "adaptive_window": adaptive_window,
            "additive_margin": additive_margin,
            "tile_size": args.tile_size,
            "num_cycle_cases": len(cases),
            "timings": timings,
            "speedup_vs_dynamic": {
                method: (
                    dynamic_ms
                    / timing["critical_path_ms"]
                )
                for method, timing in timings.items()
                if method != "dynamic"
            },
            "profile_rows": profile_rows,
            "correctness": {
                "partition_exact": partition_exact,
                "max_abs_error": max_error,
                "min_cosine_similarity": min_cosine,
                "max_allowed_abs_error": args.max_abs_error,
                "cases": quality,
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "Router assignments are reconstructed from compact OLMoE trained-router count traces; hidden rows and expert weights are deterministic synthetic tensors.",
                "Fixed profiles use tile-512 packing and fixed metadata-packed transport. Assignments beyond capacity are preserved in a variable-split metadata-packed spill side channel.",
                "A global overflow all-reduce guards the spill path. Profile construction is causal but precomputed outside timed execution.",
                "Dynamic and spill paths use the same reconstructed top-k assignments and expert weights.",
            ],
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
            + "\n",
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
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--calibration-domains",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--adaptive-window-domains",
        type=int,
        default=2,
    )
    parser.add_argument("--margin-8", type=int, default=18)
    parser.add_argument("--margin-24", type=int, default=16)
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=0.03125,
    )
    parser.add_argument(
        "--reverse-methods",
        action="store_true",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
