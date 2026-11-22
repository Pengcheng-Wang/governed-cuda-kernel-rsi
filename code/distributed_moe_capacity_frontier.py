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
    masked_global_reference,
    run_padded_route,
    run_variable_route,
)


def profile_configurations(
    profile: str,
) -> tuple[
    int,
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
            1,
            2,
        )
    return (
        512,
        (
            ("balanced", 0.0),
            ("global_bias1", 1.0),
        ),
        1,
        4 if profile == "full" else 2,
    )


def frontier_capacities(
    minimum_lossless_capacity: int,
    profile: str,
) -> tuple[tuple[str, int], ...]:
    if profile == "smoke":
        candidates = (
            (
                "threshold_minus_1",
                max(1, minimum_lossless_capacity - 1),
            ),
            ("threshold_exact", minimum_lossless_capacity),
            (
                "threshold_125pct",
                math.ceil(1.25 * minimum_lossless_capacity),
            ),
        )
    else:
        candidates = (
            (
                "threshold_75pct",
                max(1, math.floor(0.75 * minimum_lossless_capacity)),
            ),
            (
                "threshold_875pct",
                max(1, math.floor(0.875 * minimum_lossless_capacity)),
            ),
            (
                "threshold_minus_1",
                max(1, minimum_lossless_capacity - 1),
            ),
            ("threshold_exact", minimum_lossless_capacity),
            (
                "threshold_1125pct",
                math.ceil(1.125 * minimum_lossless_capacity),
            ),
            (
                "threshold_125pct",
                math.ceil(1.25 * minimum_lossless_capacity),
            ),
        )

    unique: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, capacity in candidates:
        if capacity not in seen:
            unique.append((label, capacity))
            seen.add(capacity)
    return tuple(unique)


def gather_segment_counts(
    send_counts: list[int],
    device: torch.device,
) -> list[list[int]]:
    local_counts = torch.tensor(
        send_counts,
        device=device,
        dtype=torch.int64,
    )
    gathered = [
        torch.empty_like(local_counts)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered, local_counts)
    return [
        [int(value) for value in counts.cpu().tolist()]
        for counts in gathered
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
        input_generator = torch.Generator(device=device)
        input_generator.manual_seed(
            60_000
            + rank * 10_003
            + {
                "balanced": 0,
                "global_bias1": 1_000_003,
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

        capacity_candidates = frontier_capacities(
            minimum_lossless_capacity,
            args.profile,
        )
        if args.reverse_capacities:
            capacity_candidates = tuple(
                reversed(capacity_candidates)
            )

        for candidate_label, capacity in capacity_candidates:
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
                    "Capacity frontier correctness failed: "
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
            lossless = accepted_global == planned_global
            expected_lossless = (
                capacity >= minimum_lossless_capacity
            )
            if lossless != expected_lossless:
                raise RuntimeError(
                    "Exact lossless threshold failed: "
                    f"route={route_name}, capacity={capacity}, "
                    f"minimum={minimum_lossless_capacity}, "
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
            fixed_rows_per_rank = world_size * capacity
            fixed_rows_global = world_size * fixed_rows_per_rank

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
                        (
                            "variable_end_to_end",
                            variable_end_to_end,
                        ),
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
                    "candidate": candidate_label,
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
                        "capacity_ratio_to_threshold": (
                            capacity
                            / minimum_lossless_capacity
                        ),
                        "capacity_distance_from_threshold": (
                            capacity
                            - minimum_lossless_capacity
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
                        "source_destination_counts": count_matrix,
                    },
                    "fixed_rows_per_rank": fixed_rows_per_rank,
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
                        "fixed_padding_fraction": (
                            1.0
                            - accepted_global / fixed_rows_global
                        ),
                        "fixed_row_inflation_vs_accepted": (
                            fixed_rows_global / accepted_global
                        ),
                        "lossless": lossless,
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
                        "selection_equal": (
                            selection_equal == 1.0
                        ),
                        "variable_vs_masked_reference": (
                            variable_correctness
                        ),
                        "padded_vs_masked_reference": (
                            padded_correctness
                        ),
                        "padded_vs_variable": (
                            padded_vs_variable
                        ),
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

        del uncapped_route, hidden
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_capacity_frontier",
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
            "capacity_order": (
                "descending"
                if args.reverse_capacities
                else "ascending"
            ),
            "notes": [
                "The exact lossless capacity is the global maximum source-destination assignment count for the deterministic router output.",
                "Every point compares the same retained assignments in the host-split variable path and fixed-shape padded path.",
                "The threshold-minus-one point must drop work, while the exact threshold and all larger capacities must be lossless.",
                "Position-balanced crossover timings are authoritative for close end-to-end comparisons.",
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
    parser.add_argument("--crossover-cycles", type=int, default=18)
    parser.add_argument(
        "--reverse-capacities",
        action="store_true",
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
