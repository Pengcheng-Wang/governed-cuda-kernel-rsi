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
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from distributed_moe_end_to_end import (
    TensorDict,
    critical_path_summary,
    dispatch_assignments,
    gather_scalar,
    local_experts,
    output_quality,
    return_and_combine,
    route_and_pack,
    routing_bias,
    timed_schedule,
)


def apply_local_destination_capacity(
    route: TensorDict,
    *,
    capacity_factor: float | None,
    num_tokens: int,
    top_k: int,
    world_size: int,
) -> tuple[TensorDict, dict[str, float | int | None]]:
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

    planned_assignments = send_hidden.shape[0]
    local_capacity = None
    if capacity_factor is None:
        selected = torch.arange(
            planned_assignments,
            device=send_hidden.device,
            dtype=torch.int64,
        )
        accepted_counts = send_counts[:]
    else:
        local_capacity = int(
            math.ceil(
                capacity_factor
                * num_tokens
                * top_k
                / world_size
            )
        )
        selected_segments: list[torch.Tensor] = []
        accepted_counts = []
        offset = 0
        flat_weights = send_weights.reshape(-1)
        for count in send_counts:
            accepted = min(count, local_capacity)
            accepted_counts.append(accepted)
            if accepted == count:
                indices = torch.arange(
                    offset,
                    offset + count,
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
                indices = relative + offset
            else:
                indices = torch.empty(
                    (0,),
                    device=send_hidden.device,
                    dtype=torch.int64,
                )
            selected_segments.append(indices)
            offset += count
        selected = torch.cat(selected_segments)

    accepted_hidden = send_hidden.index_select(0, selected).contiguous()
    accepted_local_ids = send_local_ids.index_select(
        0,
        selected,
    ).contiguous()
    accepted_weights = send_weights.index_select(
        0,
        selected,
    ).contiguous()
    accepted_token_indices = sorted_token_indices.index_select(
        0,
        selected,
    )
    accepted_assignment_indices = sorted_assignment_indices.index_select(
        0,
        selected,
    )
    accepted_counts_tensor = torch.tensor(
        accepted_counts,
        device=send_hidden.device,
        dtype=torch.int64,
    )
    token_assignment_counts = torch.bincount(
        accepted_token_indices,
        minlength=num_tokens,
    )

    accepted_assignments = accepted_hidden.shape[0]
    planned_weight_mass = send_weights.float().sum().item()
    accepted_weight_mass = accepted_weights.float().sum().item()
    metadata: dict[str, float | int | None] = {
        "local_destination_capacity": local_capacity,
        "planned_assignments": planned_assignments,
        "accepted_assignments": accepted_assignments,
        "dropped_assignments": (
            planned_assignments - accepted_assignments
        ),
        "planned_router_weight_mass": planned_weight_mass,
        "accepted_router_weight_mass": accepted_weight_mass,
        "dropped_router_weight_mass": (
            planned_weight_mass - accepted_weight_mass
        ),
        "fully_dropped_tokens": int(
            (token_assignment_counts == 0).sum().item()
        ),
        "partially_retained_tokens": int(
            (
                (token_assignment_counts > 0)
                & (token_assignment_counts < top_k)
            )
            .sum()
            .item()
        ),
        "fully_retained_tokens": int(
            (token_assignment_counts == top_k).sum().item()
        ),
    }

    capped_route = dict(route)
    capped_route.update(
        {
            "send_hidden": accepted_hidden,
            "send_local_ids": accepted_local_ids,
            "send_weights": accepted_weights,
            "sorted_token_indices": accepted_token_indices,
            "selected_assignment_indices": (
                accepted_assignment_indices
            ),
            "send_counts_tensor": accepted_counts_tensor,
            "send_counts": accepted_counts,
        }
    )
    return capped_route, metadata


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
    accepted_weights = route["send_weights"]
    assert isinstance(topk_ids, torch.Tensor)
    assert isinstance(topk_weights, torch.Tensor)
    assert isinstance(selected_assignment_indices, torch.Tensor)
    assert isinstance(accepted_weights, torch.Tensor)

    masked_weights = torch.zeros_like(topk_weights)
    masked_weights.reshape(-1).scatter_(
        0,
        selected_assignment_indices,
        accepted_weights.reshape(-1),
    )
    return fused_experts(
        hidden_states,
        global_w1,
        global_w2,
        masked_weights,
        topk_ids,
    )


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    if args.profile == "smoke":
        token_values = [32]
        capacity_factors: list[float | None] = [None, 1.25]
        warmup, iterations = 1, 3
    else:
        token_values = [512, 2048]
        capacity_factors = [None, 1.25, 2.0, 4.0, 8.0]
        warmup, iterations = 3, 10

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
    bias = routing_bias(
        "global_biased",
        rank=rank,
        num_experts=num_experts,
        experts_per_rank=args.experts_per_rank,
        hot_bias=args.hot_bias,
        device=device,
    )

    records: list[dict[str, Any]] = []
    for num_tokens in token_values:
        input_generator = torch.Generator(device=device)
        input_generator.manual_seed(10_000 + rank * 101 + num_tokens)
        hidden_states = torch.randn(
            (num_tokens, args.hidden),
            device=device,
            dtype=dtype,
            generator=input_generator,
        )

        base_route = route_and_pack(
            hidden_states,
            gate_weight,
            bias,
            tie_breaker,
            top_k=args.top_k,
            experts_per_rank=args.experts_per_rank,
            world_size=world_size,
        )
        topk_ids = base_route["topk_ids"]
        topk_weights = base_route["topk_weights"]
        assert isinstance(topk_ids, torch.Tensor)
        assert isinstance(topk_weights, torch.Tensor)
        uncapped_reference = fused_experts(
            hidden_states,
            global_w1,
            global_w2,
            topk_weights,
            topk_ids,
        )

        for capacity_factor in capacity_factors:
            def build_route() -> tuple[
                TensorDict,
                dict[str, float | int | None],
            ]:
                route = route_and_pack(
                    hidden_states,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                )
                return apply_local_destination_capacity(
                    route,
                    capacity_factor=capacity_factor,
                    num_tokens=num_tokens,
                    top_k=args.top_k,
                    world_size=world_size,
                )

            prepared_route, local_capacity = build_route()
            prepared_dispatch = dispatch_assignments(prepared_route)
            prepared_expert_output = local_experts(
                prepared_dispatch,
                local_w1,
                local_w2,
            )
            distributed_output = return_and_combine(
                prepared_expert_output,
                prepared_route,
                prepared_dispatch,
                num_tokens=num_tokens,
            )
            masked_reference = masked_global_reference(
                hidden_states,
                global_w1,
                global_w2,
                prepared_route,
            )
            torch.cuda.synchronize()

            correctness = output_quality(
                distributed_output,
                masked_reference,
            )
            if (
                correctness["normalized_rmse"] > args.max_nrmse
                or correctness["cosine_similarity"] < args.min_cosine
            ):
                raise RuntimeError(
                    f"Capacity correctness failed for T={num_tokens}, "
                    f"capacity={capacity_factor}: {correctness}"
                )
            quality_vs_uncapped = output_quality(
                distributed_output,
                uncapped_reference,
            )

            send_counts = prepared_route["send_counts"]
            assert isinstance(send_counts, list)
            send_counts_tensor = torch.tensor(
                send_counts,
                device=device,
                dtype=torch.int64,
            )
            gathered_counts = [
                torch.empty_like(send_counts_tensor)
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_counts, send_counts_tensor)
            send_matrix = [
                tensor.cpu().tolist()
                for tensor in gathered_counts
            ]
            receive_totals = [
                sum(
                    send_matrix[source][destination]
                    for source in range(world_size)
                )
                for destination in range(world_size)
            ]
            mean_receive = statistics.fmean(receive_totals)

            planned_assignments = sum(
                gather_scalar(
                    float(local_capacity["planned_assignments"]),
                    device,
                )
            )
            accepted_assignments = sum(
                gather_scalar(
                    float(local_capacity["accepted_assignments"]),
                    device,
                )
            )
            planned_weight_mass = sum(
                gather_scalar(
                    float(
                        local_capacity[
                            "planned_router_weight_mass"
                        ]
                    ),
                    device,
                )
            )
            accepted_weight_mass = sum(
                gather_scalar(
                    float(
                        local_capacity[
                            "accepted_router_weight_mass"
                        ]
                    ),
                    device,
                )
            )
            fully_dropped_tokens = sum(
                gather_scalar(
                    float(local_capacity["fully_dropped_tokens"]),
                    device,
                )
            )
            partially_retained_tokens = sum(
                gather_scalar(
                    float(
                        local_capacity[
                            "partially_retained_tokens"
                        ]
                    ),
                    device,
                )
            )
            total_tokens = world_size * num_tokens

            def route_capacity_only() -> TensorDict:
                route, _ = build_route()
                return route

            def dispatch_only() -> TensorDict:
                return dispatch_assignments(prepared_route)

            def expert_only() -> torch.Tensor:
                return local_experts(
                    prepared_dispatch,
                    local_w1,
                    local_w2,
                )

            def return_combine_only() -> torch.Tensor:
                return return_and_combine(
                    prepared_expert_output,
                    prepared_route,
                    prepared_dispatch,
                    num_tokens=num_tokens,
                )

            def distributed_end_to_end() -> torch.Tensor:
                route, _ = build_route()
                dispatched = dispatch_assignments(route)
                expert_output = local_experts(
                    dispatched,
                    local_w1,
                    local_w2,
                )
                return return_and_combine(
                    expert_output,
                    route,
                    dispatched,
                    num_tokens=num_tokens,
                )

            timings: dict[str, dict[str, Any]] = {}
            schedules: tuple[
                tuple[str, Callable[[], Any]],
                ...,
            ] = (
                ("router_pack_capacity", route_capacity_only),
                ("dispatch", dispatch_only),
                ("local_experts", expert_only),
                ("return_combine", return_combine_only),
                ("distributed_end_to_end", distributed_end_to_end),
            )
            for name, schedule in schedules:
                timings[name] = critical_path_summary(
                    timed_schedule(
                        schedule,
                        warmup=warmup,
                        iterations=iterations,
                        local_rank=local_rank,
                    ),
                    device,
                )

            records.append(
                {
                    "tokens_per_rank": num_tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts_per_rank": args.experts_per_rank,
                    "num_experts": num_experts,
                    "top_k": args.top_k,
                    "routing_pattern": "global_biased",
                    "hot_bias": args.hot_bias,
                    "capacity_factor": capacity_factor,
                    "local_destination_capacity": local_capacity[
                        "local_destination_capacity"
                    ],
                    "capacity_policy": (
                        "highest router weights per source-destination "
                        "segment; no post-drop renormalization"
                    ),
                    "dtype": str(dtype),
                    "capacity": {
                        "planned_assignments": planned_assignments,
                        "accepted_assignments": accepted_assignments,
                        "dropped_assignments": (
                            planned_assignments
                            - accepted_assignments
                        ),
                        "dropped_assignment_fraction": (
                            1.0
                            - accepted_assignments
                            / planned_assignments
                        ),
                        "planned_router_weight_mass": (
                            planned_weight_mass
                        ),
                        "accepted_router_weight_mass": (
                            accepted_weight_mass
                        ),
                        "retained_router_weight_fraction": (
                            accepted_weight_mass
                            / planned_weight_mass
                        ),
                        "fully_dropped_token_fraction": (
                            fully_dropped_tokens / total_tokens
                        ),
                        "partially_retained_token_fraction": (
                            partially_retained_tokens / total_tokens
                        ),
                    },
                    "routing": {
                        "send_matrix": send_matrix,
                        "rank_receive_assignments": receive_totals,
                        "max_to_mean_receive": (
                            max(receive_totals) / mean_receive
                            if mean_receive
                            else 0.0
                        ),
                        "receive_cv": (
                            statistics.pstdev(receive_totals)
                            / mean_receive
                            if mean_receive
                            else 0.0
                        ),
                    },
                    "correctness_vs_masked_reference": {
                        "max_abs_error": max(
                            gather_scalar(
                                correctness["max_abs_error"],
                                device,
                            )
                        ),
                        "max_normalized_rmse": max(
                            gather_scalar(
                                correctness["normalized_rmse"],
                                device,
                            )
                        ),
                        "min_cosine_similarity": min(
                            gather_scalar(
                                correctness["cosine_similarity"],
                                device,
                            )
                        ),
                    },
                    "quality_vs_uncapped": {
                        "max_abs_error": max(
                            gather_scalar(
                                quality_vs_uncapped["max_abs_error"],
                                device,
                            )
                        ),
                        "max_normalized_rmse": max(
                            gather_scalar(
                                quality_vs_uncapped[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_cosine_similarity": min(
                            gather_scalar(
                                quality_vs_uncapped[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                    },
                    "timings": timings,
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )

            del (
                prepared_route,
                prepared_dispatch,
                prepared_expert_output,
                distributed_output,
                masked_reference,
            )
            torch.cuda.empty_cache()

        del base_route, uncapped_reference
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_capacity",
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
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "The workload reuses the integrated global-biased router, token permutation, variable dispatch, local fused experts, return all-to-all, and weighted combine path.",
                "Each source independently caps every destination at ceil(capacity_factor * local_assignments / world_size) and retains the highest router weights in that source-destination segment.",
                "This source-fair policy bounds aggregate destination capacity without a global selection collective, but can underfill capacity when sources have different split vectors.",
                "Dropped weights are not renormalized. Quality is reported against the uncapped output, while correctness is checked against an independently masked global-expert reference.",
                "Separate stage timings reuse prepared tensors and are diagnostic rather than additive. distributed_end_to_end is authoritative.",
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
        choices=["smoke", "full"],
        default="smoke",
    )
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=2816)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hot-bias", type=float, default=4.0)
    parser.add_argument("--max-nrmse", type=float, default=5e-3)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
