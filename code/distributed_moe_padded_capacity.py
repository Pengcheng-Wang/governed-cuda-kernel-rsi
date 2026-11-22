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
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
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
from distributed_moe_variable_int8_overlap import (
    finish_dispatch,
    finish_return,
    launch_dispatch,
    launch_return,
    prepare_dispatch,
)


def profile_configurations(
    profile: str,
    world_size: int,
) -> tuple[
    int,
    tuple[tuple[str, float, float], ...],
    int,
    int,
]:
    if profile == "smoke":
        return (
            128,
            (
                ("balanced", 0.0, 1.0),
                ("balanced", 0.0, 2.0),
                ("global_bias1", 1.0, 1.0),
                ("global_bias1", 1.0, 2.0),
            ),
            1,
            2,
        )
    if profile == "crossover":
        bias1_drop = 2.0 if world_size <= 8 else 4.0
        bias1_lossless = 4.0 if world_size <= 8 else 8.0
        bias4_lossless = 8.0 if world_size <= 8 else 24.0
        return (
            512,
            (
                ("balanced", 0.0, 1.0),
                ("balanced", 0.0, 2.0),
                ("global_bias1", 1.0, bias1_drop),
                ("global_bias1", 1.0, bias1_lossless),
                ("global_bias4", 4.0, 2.0),
                ("global_bias4", 4.0, bias4_lossless),
            ),
            1,
            2,
        )
    return (
        512,
        tuple(
            (route, bias, capacity_factor)
            for route, bias in (
                ("balanced", 0.0),
                ("global_bias1", 1.0),
                ("global_bias4", 4.0),
            )
            for capacity_factor in (1.0, 2.0, 4.0, 8.0, 24.0)
        ),
        1,
        4,
    )


def apply_host_capacity(
    route: TensorDict,
    capacity: int,
) -> TensorDict:
    send_hidden = route["send_hidden"]
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    sorted_token_indices = route["sorted_token_indices"]
    sorted_assignment_indices = route["sorted_assignment_indices"]
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
    for count in send_counts:
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


def build_padded_route(
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
    local_ids = torch.remainder(flat_ids, experts_per_rank)

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
    offsets = torch.cumsum(planned_counts, dim=0) - planned_counts
    positions = (
        torch.arange(
            grouped_order.numel(),
            device=hidden_states.device,
            dtype=torch.int64,
        )
        - offsets.index_select(0, grouped_destinations)
    )
    accepted_mask = positions < capacity
    selected_assignment_indices = grouped_order[accepted_mask]
    selected_destinations = grouped_destinations[accepted_mask]
    selected_positions = positions[accepted_mask]
    fixed_indices = (
        selected_destinations * capacity + selected_positions
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

    fixed_rows = world_size * capacity
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
    send_hidden.index_copy_(0, fixed_indices, selected_hidden)
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
        "selected_assignment_indices": selected_assignment_indices,
        "accepted_counts_tensor": torch.clamp(
            planned_counts,
            max=capacity,
        ),
        "capacity": torch.tensor(
            capacity,
            device=hidden_states.device,
            dtype=torch.int64,
        ),
    }


def masked_global_reference(
    hidden_states: torch.Tensor,
    global_w1: torch.Tensor,
    global_w2: torch.Tensor,
    route: TensorDict,
) -> torch.Tensor:
    topk_ids = route["topk_ids"]
    topk_weights = route["topk_weights"]
    selected_assignment_indices = route[
        "selected_assignment_indices"
    ]
    selected_weights = route["selected_weights"]
    assert isinstance(topk_ids, torch.Tensor)
    assert isinstance(topk_weights, torch.Tensor)
    assert isinstance(selected_assignment_indices, torch.Tensor)
    assert isinstance(selected_weights, torch.Tensor)
    masked_weights = torch.zeros_like(topk_weights)
    masked_weights.reshape(-1).scatter_(
        0,
        selected_assignment_indices,
        selected_weights,
    )
    return fused_experts(
        hidden_states,
        global_w1,
        global_w2,
        masked_weights,
        topk_ids,
    )


def dispatch_padded(route: TensorDict) -> TensorDict:
    send_hidden = route["send_hidden"]
    assert isinstance(send_hidden, torch.Tensor)
    metadata = build_metadata_packet(route)
    recv_hidden = torch.empty_like(send_hidden)
    recv_metadata = torch.empty_like(metadata)
    dist.all_to_all_single(recv_hidden, send_hidden)
    dist.all_to_all_single(recv_metadata, metadata)
    return {
        "recv_hidden": recv_hidden,
        "recv_local_ids": (
            recv_metadata[:, :1].to(torch.int32).contiguous()
        ),
        "recv_weights": recv_metadata[:, 1:].contiguous(),
    }


def return_padded(
    expert_output: torch.Tensor,
    route: TensorDict,
    *,
    num_tokens: int,
) -> torch.Tensor:
    returned = torch.empty_like(expert_output)
    dist.all_to_all_single(returned, expert_output)
    fixed_token_indices = route["fixed_token_indices"]
    assert isinstance(fixed_token_indices, torch.Tensor)
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


def run_variable_route(
    route: TensorDict,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    prepared = prepare_dispatch(route)
    dispatched = finish_dispatch(launch_dispatch(prepared))
    expert_output = local_experts(
        dispatched,
        local_w1,
        local_w2,
    )
    send_hidden = route["send_hidden"]
    assert isinstance(send_hidden, torch.Tensor)
    return finish_return(
        launch_return(
            "bf16",
            expert_output,
            prepared,
            {"bf16": torch.empty_like(send_hidden)},
            packet_alignment=16,
            num_tokens=num_tokens,
        )
    )


def run_padded_route(
    route: TensorDict,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    dispatched = dispatch_padded(route)
    expert_output = local_experts(
        dispatched,
        local_w1,
        local_w2,
    )
    return return_padded(
        expert_output,
        route,
        num_tokens=num_tokens,
    )


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
        configurations,
        warmup,
        iterations,
    ) = profile_configurations(args.profile, world_size)
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
    for (
        route_name,
        hot_bias,
        capacity_factor,
    ) in configurations:
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
        capacity = int(
            math.ceil(
                capacity_factor
                * assignments_per_rank
                / world_size
            )
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

        def build_padded() -> TensorDict:
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

        variable_route = build_variable()
        padded_route = build_padded()
        variable_output = run_variable_route(
            variable_route,
            local_w1,
            local_w2,
            num_tokens=num_tokens,
        )
        padded_output = run_padded_route(
            padded_route,
            local_w1,
            local_w2,
            num_tokens=num_tokens,
        )
        masked_reference = masked_global_reference(
            hidden,
            global_w1,
            global_w2,
            padded_route,
        )
        topk_ids = padded_route["topk_ids"]
        topk_weights = padded_route["topk_weights"]
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
        padded_selected = padded_route[
            "selected_assignment_indices"
        ]
        assert isinstance(variable_selected, torch.Tensor)
        assert isinstance(padded_selected, torch.Tensor)
        local_selection_equal = torch.equal(
            torch.sort(variable_selected).values,
            torch.sort(padded_selected).values,
        )
        selection_equal = min(
            gather_scalar(float(local_selection_equal), device)
        )
        variable_correctness = output_quality(
            variable_output,
            masked_reference,
        )
        padded_correctness = output_quality(
            padded_output,
            masked_reference,
        )
        padded_vs_variable = output_quality(
            padded_output,
            variable_output,
        )
        quality_vs_uncapped = output_quality(
            padded_output,
            uncapped_reference,
        )
        if (
            selection_equal != 1.0
            or padded_vs_variable["max_abs_error"] != 0.0
            or variable_correctness["max_abs_error"] != 0.0
            or padded_correctness["max_abs_error"] != 0.0
        ):
            raise RuntimeError(
                "Padded capacity correctness failed: "
                f"selection_equal={selection_equal}, "
                f"variable={variable_correctness}, "
                f"padded={padded_correctness}, "
                f"padded_vs_variable={padded_vs_variable}"
            )

        selected_token_indices = padded_route[
            "selected_token_indices"
        ]
        selected_weights = padded_route["selected_weights"]
        assert isinstance(selected_token_indices, torch.Tensor)
        assert isinstance(selected_weights, torch.Tensor)
        accepted_local = selected_token_indices.numel()
        planned_global = world_size * assignments_per_rank
        accepted_global = sum(
            gather_scalar(float(accepted_local), device)
        )
        fixed_rows_per_rank = world_size * capacity
        fixed_rows_global = world_size * fixed_rows_per_rank
        expected_lossless_factor = {
            "balanced": 2.0,
            "global_bias1": (
                4.0 if world_size <= 8 else 8.0
            ),
            "global_bias4": (
                8.0 if world_size <= 8 else 24.0
            ),
        }[route_name]
        if (
            args.profile == "crossover"
            and capacity_factor == expected_lossless_factor
            and accepted_global != planned_global
        ):
            raise RuntimeError(
                f"Crossover capacity was not lossless: "
                f"route={route_name}, factor={capacity_factor}, "
                f"accepted={accepted_global}, "
                f"planned={planned_global}"
            )
        planned_weight_global = sum(
            gather_scalar(
                float(topk_weights.float().sum().item()),
                device,
            )
        )
        accepted_weight_global = sum(
            gather_scalar(
                float(selected_weights.float().sum().item()),
                device,
            )
        )
        token_assignment_counts = torch.bincount(
            selected_token_indices,
            minlength=num_tokens,
        )
        fully_dropped_tokens = sum(
            gather_scalar(
                float(
                    (token_assignment_counts == 0).sum().item()
                ),
                device,
            )
        )

        def variable_end_to_end() -> torch.Tensor:
            return run_variable_route(
                build_variable(),
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )

        def padded_end_to_end() -> torch.Tensor:
            return run_padded_route(
                build_padded(),
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )

        schedules: dict[str, Callable[[], Any]] = {
            "variable_route_capacity": build_variable,
            "padded_route_capacity": build_padded,
            "variable_end_to_end": variable_end_to_end,
            "padded_end_to_end": padded_end_to_end,
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
        crossover = None
        if args.profile == "crossover":
            crossover = timed_crossover(
                (
                    ("variable_end_to_end", variable_end_to_end),
                    ("padded_end_to_end", padded_end_to_end),
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
                "capacity_factor": capacity_factor,
                "capacity_per_destination": capacity,
                "fixed_rows_per_rank": fixed_rows_per_rank,
                "capacity": {
                    "planned_assignments": planned_global,
                    "accepted_assignments": accepted_global,
                    "dropped_assignment_fraction": (
                        1.0 - accepted_global / planned_global
                    ),
                    "retained_router_weight_fraction": (
                        accepted_weight_global
                        / planned_weight_global
                    ),
                    "fully_dropped_token_fraction": (
                        fully_dropped_tokens
                        / (world_size * num_tokens)
                    ),
                    "fixed_padding_fraction": (
                        1.0
                        - accepted_global / fixed_rows_global
                    ),
                    "fixed_row_inflation_vs_accepted": (
                        fixed_rows_global / accepted_global
                    ),
                    "lossless": (
                        accepted_global == planned_global
                    ),
                },
                "transport": {
                    "variable_count_exchange": True,
                    "variable_python_split_lists": True,
                    "padded_count_exchange": False,
                    "padded_python_split_lists": False,
                    "padded_dense_expert_rows": (
                        fixed_rows_per_rank
                    ),
                },
                "correctness": {
                    "selection_equal": selection_equal == 1.0,
                    "variable_vs_masked_reference": (
                        variable_correctness
                    ),
                    "padded_vs_masked_reference": (
                        padded_correctness
                    ),
                    "padded_vs_variable": padded_vs_variable,
                },
                "quality_vs_uncapped": quality_vs_uncapped,
                "timings": timings,
                "speedup": (
                    timings["variable_end_to_end"][
                        "critical_path_ms"
                    ]
                    / timings["padded_end_to_end"][
                        "critical_path_ms"
                    ]
                ),
                "crossover": crossover,
                "warmup": warmup,
                "iterations": iterations,
            }
        )
        del (
            variable_route,
            padded_route,
            variable_output,
            padded_output,
            masked_reference,
            uncapped_reference,
        )
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_padded_capacity",
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
            "torch_cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "The variable control uses count exchange and Python split lists; the padded path uses equal-size all-to-all without count exchange.",
                "Static capacity is ceil(capacity_factor * assignments_per_rank / world_size) for every source-destination segment.",
                "The padded path keeps a fixed tensor shape and computes zero-weight padding rows instead of using data-dependent compaction.",
                "Both paths retain the same highest router weights per source-destination segment and must match exactly.",
                "Capacity factors trade host synchronization for padding, dense expert work, assignment drops, and output quality.",
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
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=2816)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--crossover-cycles", type=int, default=18)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
