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
    timed_schedule,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def destination_bias(
    *,
    hot_destination: int | None,
    hot_bias: float,
    experts_per_rank: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    bias = torch.zeros(num_experts, device=device, dtype=torch.float32)
    if hot_destination is not None:
        start = hot_destination * experts_per_rank
        bias[start : start + experts_per_rank] = hot_bias
    return bias


def gather_step_critical_paths(
    local_steps_ms: list[list[float]],
    device: torch.device,
) -> list[list[float]]:
    local = torch.tensor(
        local_steps_ms,
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    stacked = torch.stack(gathered)
    return stacked.max(dim=0).values.cpu().tolist()


def profile_step_latencies(
    fn: Callable[[int], torch.Tensor],
    *,
    steps: int,
    warmup_cycles: int,
    iterations: int,
    local_rank: int,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(warmup_cycles):
        outputs = [fn(step) for step in range(steps)]
        torch.cuda.synchronize()
        del outputs

    local_runs: list[list[float]] = []
    for _ in range(iterations):
        dist.barrier(device_ids=[local_rank])
        run: list[float] = []
        for step in range(steps):
            started = time.perf_counter()
            output = fn(step)
            torch.cuda.synchronize()
            run.append((time.perf_counter() - started) * 1_000.0)
            del output
        dist.barrier(device_ids=[local_rank])
        local_runs.append(run)

    critical_runs = gather_step_critical_paths(local_runs, device)
    position_medians = [
        statistics.median(
            critical_runs[iteration][step]
            for iteration in range(iterations)
        )
        for step in range(steps)
    ]
    all_samples = [
        sample
        for iteration in critical_runs
        for sample in iteration
    ]
    return {
        "critical_step_samples_ms": critical_runs,
        "position_median_ms": position_medians,
        "p50_ms": percentile(all_samples, 0.50),
        "p95_ms": percentile(all_samples, 0.95),
        "p99_ms": percentile(all_samples, 0.99),
        "max_ms": max(all_samples),
        "mean_ms": statistics.fmean(all_samples),
        "cv": (
            statistics.pstdev(all_samples)
            / statistics.fmean(all_samples)
        ),
        "iterations": iterations,
    }


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
        schedules = (
            ("steady_balanced", "balanced", 0.0),
            ("rotating_bias1", "rotating", 1.0),
            ("persistent_bias1", "persistent", 1.0),
        )
        warmup, iterations = 1, 2
        step_warmup, step_iterations = 1, 2
    else:
        token_values = [512, 2048]
        schedules = (
            ("steady_balanced", "balanced", 0.0),
            ("rotating_bias1", "rotating", 1.0),
            ("rotating_bias2", "rotating", 2.0),
            ("persistent_bias1", "persistent", 1.0),
            ("persistent_bias2", "persistent", 2.0),
        )
        warmup, iterations = 2, 5
        step_warmup, step_iterations = 1, 3

    steps = world_size
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
    for num_tokens in token_values:
        hidden_states: list[torch.Tensor] = []
        for step in range(steps):
            input_generator = torch.Generator(device=device)
            input_generator.manual_seed(
                10_000
                + rank * 10_003
                + num_tokens * 31
                + step
            )
            hidden_states.append(
                torch.randn(
                    (num_tokens, args.hidden),
                    device=device,
                    dtype=dtype,
                    generator=input_generator,
                )
            )

        balanced_ids: list[torch.Tensor] = []
        balanced_outputs: list[torch.Tensor] = []
        for step in range(steps):
            bias = destination_bias(
                hot_destination=None,
                hot_bias=0.0,
                experts_per_rank=args.experts_per_rank,
                num_experts=num_experts,
                device=device,
            )
            route = route_and_pack(
                hidden_states[step],
                gate_weight,
                bias,
                tie_breaker,
                top_k=args.top_k,
                experts_per_rank=args.experts_per_rank,
                world_size=world_size,
            )
            ids = route["topk_ids"]
            weights = route["topk_weights"]
            assert isinstance(ids, torch.Tensor)
            assert isinstance(weights, torch.Tensor)
            balanced_ids.append(ids)
            balanced_outputs.append(
                fused_experts(
                    hidden_states[step],
                    global_w1,
                    global_w2,
                    weights,
                    ids,
                )
            )

        for schedule_name, schedule_kind, hot_bias in schedules:
            hot_destinations: list[int | None] = []
            for step in range(steps):
                if schedule_kind == "balanced":
                    hot_destinations.append(None)
                elif schedule_kind == "rotating":
                    hot_destinations.append(step % world_size)
                elif schedule_kind == "persistent":
                    hot_destinations.append(0)
                else:
                    raise ValueError(schedule_kind)

            def build_route(step: int) -> TensorDict:
                bias = destination_bias(
                    hot_destination=hot_destinations[step],
                    hot_bias=hot_bias,
                    experts_per_rank=args.experts_per_rank,
                    num_experts=num_experts,
                    device=device,
                )
                return route_and_pack(
                    hidden_states[step],
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                )

            prepared_routes = [
                build_route(step)
                for step in range(steps)
            ]
            prepared_dispatches = [
                dispatch_assignments(route)
                for route in prepared_routes
            ]
            prepared_expert_outputs = [
                local_experts(
                    dispatched,
                    local_w1,
                    local_w2,
                )
                for dispatched in prepared_dispatches
            ]

            aggregate_send_matrix = [
                [0] * world_size
                for _ in range(world_size)
            ]
            step_metrics: list[dict[str, Any]] = []
            local_max_correctness_nrmse = 0.0
            local_min_correctness_cosine = 1.0
            local_set_matches: list[float] = []
            local_output_cosines: list[float] = []

            for step in range(steps):
                distributed_output = return_and_combine(
                    prepared_expert_outputs[step],
                    prepared_routes[step],
                    prepared_dispatches[step],
                    num_tokens=num_tokens,
                )
                topk_ids = prepared_routes[step]["topk_ids"]
                topk_weights = prepared_routes[step][
                    "topk_weights"
                ]
                send_counts = prepared_routes[step]["send_counts"]
                assert isinstance(topk_ids, torch.Tensor)
                assert isinstance(topk_weights, torch.Tensor)
                assert isinstance(send_counts, list)
                reference_output = fused_experts(
                    hidden_states[step],
                    global_w1,
                    global_w2,
                    topk_weights,
                    topk_ids,
                )
                correctness = output_quality(
                    distributed_output,
                    reference_output,
                )
                local_max_correctness_nrmse = max(
                    local_max_correctness_nrmse,
                    correctness["normalized_rmse"],
                )
                local_min_correctness_cosine = min(
                    local_min_correctness_cosine,
                    correctness["cosine_similarity"],
                )
                if (
                    correctness["normalized_rmse"] > args.max_nrmse
                    or correctness["cosine_similarity"]
                    < args.min_cosine
                ):
                    raise RuntimeError(
                        f"Temporal correctness failed for "
                        f"schedule={schedule_name}, step={step}: "
                        f"{correctness}"
                    )

                route_set_match = (
                    torch.sort(topk_ids, dim=-1).values
                    == torch.sort(
                        balanced_ids[step],
                        dim=-1,
                    ).values
                ).all(dim=-1).float().mean().item()
                drift = output_quality(
                    reference_output,
                    balanced_outputs[step],
                )
                local_set_matches.append(route_set_match)
                local_output_cosines.append(
                    drift["cosine_similarity"]
                )

                send_counts_tensor = torch.tensor(
                    send_counts,
                    device=device,
                    dtype=torch.int64,
                )
                gathered_counts = [
                    torch.empty_like(send_counts_tensor)
                    for _ in range(world_size)
                ]
                dist.all_gather(
                    gathered_counts,
                    send_counts_tensor,
                )
                send_matrix = [
                    tensor.cpu().tolist()
                    for tensor in gathered_counts
                ]
                for source in range(world_size):
                    for destination in range(world_size):
                        aggregate_send_matrix[source][destination] += (
                            send_matrix[source][destination]
                        )
                receive_totals = [
                    sum(
                        send_matrix[source][destination]
                        for source in range(world_size)
                    )
                    for destination in range(world_size)
                ]
                total_assignments = sum(receive_totals)
                mean_receive = statistics.fmean(receive_totals)
                hottest_destination = max(
                    range(world_size),
                    key=receive_totals.__getitem__,
                )
                step_metrics.append(
                    {
                        "step": step,
                        "configured_hot_destination": (
                            hot_destinations[step]
                        ),
                        "observed_hot_destination": (
                            hottest_destination
                        ),
                        "hottest_receive_fraction": (
                            receive_totals[hottest_destination]
                            / total_assignments
                        ),
                        "max_to_mean_receive": (
                            max(receive_totals) / mean_receive
                        ),
                    }
                )

                del distributed_output, reference_output

            aggregate_receive_totals = [
                sum(
                    aggregate_send_matrix[source][destination]
                    for source in range(world_size)
                )
                for destination in range(world_size)
            ]
            aggregate_total = sum(aggregate_receive_totals)
            aggregate_mean = statistics.fmean(
                aggregate_receive_totals
            )
            aggregate_probabilities = [
                count / aggregate_total
                for count in aggregate_receive_totals
                if count
            ]
            aggregate_entropy = -sum(
                probability * math.log(probability)
                for probability in aggregate_probabilities
            )

            route_set_match = statistics.fmean(
                gather_scalar(
                    statistics.fmean(local_set_matches),
                    device,
                )
            )
            min_output_cosine = min(
                gather_scalar(
                    min(local_output_cosines),
                    device,
                )
            )

            def router_cycle() -> list[TensorDict]:
                return [
                    build_route(step)
                    for step in range(steps)
                ]

            def dispatch_cycle() -> list[TensorDict]:
                return [
                    dispatch_assignments(prepared_routes[step])
                    for step in range(steps)
                ]

            def expert_cycle() -> list[torch.Tensor]:
                return [
                    local_experts(
                        prepared_dispatches[step],
                        local_w1,
                        local_w2,
                    )
                    for step in range(steps)
                ]

            def return_cycle() -> list[torch.Tensor]:
                return [
                    return_and_combine(
                        prepared_expert_outputs[step],
                        prepared_routes[step],
                        prepared_dispatches[step],
                        num_tokens=num_tokens,
                    )
                    for step in range(steps)
                ]

            def integrated_step(step: int) -> torch.Tensor:
                route = build_route(step)
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

            def integrated_cycle() -> list[torch.Tensor]:
                return [
                    integrated_step(step)
                    for step in range(steps)
                ]

            timings: dict[str, dict[str, Any]] = {}
            schedules_to_time: tuple[
                tuple[str, Callable[[], Any]],
                ...,
            ] = (
                ("router_cycle", router_cycle),
                ("dispatch_cycle", dispatch_cycle),
                ("expert_cycle", expert_cycle),
                ("return_cycle", return_cycle),
                ("integrated_cycle", integrated_cycle),
            )
            for name, schedule in schedules_to_time:
                timings[name] = critical_path_summary(
                    timed_schedule(
                        schedule,
                        warmup=warmup,
                        iterations=iterations,
                        local_rank=local_rank,
                    ),
                    device,
                )

            step_latency = profile_step_latencies(
                integrated_step,
                steps=steps,
                warmup_cycles=step_warmup,
                iterations=step_iterations,
                local_rank=local_rank,
                device=device,
            )

            records.append(
                {
                    "tokens_per_rank_per_step": num_tokens,
                    "steps": steps,
                    "total_tokens_per_rank": num_tokens * steps,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts_per_rank": args.experts_per_rank,
                    "num_experts": num_experts,
                    "top_k": args.top_k,
                    "schedule": schedule_name,
                    "schedule_kind": schedule_kind,
                    "hot_bias": hot_bias,
                    "dtype": str(dtype),
                    "aggregate_routing": {
                        "send_matrix": aggregate_send_matrix,
                        "rank_receive_assignments": (
                            aggregate_receive_totals
                        ),
                        "max_to_mean_receive": (
                            max(aggregate_receive_totals)
                            / aggregate_mean
                        ),
                        "receive_cv": (
                            statistics.pstdev(
                                aggregate_receive_totals
                            )
                            / aggregate_mean
                        ),
                        "normalized_destination_entropy": (
                            aggregate_entropy
                            / math.log(world_size)
                        ),
                        "effective_destinations": math.exp(
                            aggregate_entropy
                        ),
                    },
                    "step_routing": {
                        "records": step_metrics,
                        "median_max_to_mean_receive": (
                            statistics.median(
                                item["max_to_mean_receive"]
                                for item in step_metrics
                            )
                        ),
                        "max_max_to_mean_receive": max(
                            item["max_to_mean_receive"]
                            for item in step_metrics
                        ),
                        "median_hottest_receive_fraction": (
                            statistics.median(
                                item[
                                    "hottest_receive_fraction"
                                ]
                                for item in step_metrics
                            )
                        ),
                    },
                    "route_set_match_vs_balanced": (
                        route_set_match
                    ),
                    "min_output_cosine_vs_balanced": (
                        min_output_cosine
                    ),
                    "correctness": {
                        "max_normalized_rmse": max(
                            gather_scalar(
                                local_max_correctness_nrmse,
                                device,
                            )
                        ),
                        "min_cosine_similarity": min(
                            gather_scalar(
                                local_min_correctness_cosine,
                                device,
                            )
                        ),
                    },
                    "timings": timings,
                    "step_latency": step_latency,
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )

            del (
                prepared_routes,
                prepared_dispatches,
                prepared_expert_outputs,
            )
            torch.cuda.empty_cache()

        del hidden_states, balanced_ids, balanced_outputs
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_temporal_skew",
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
                "Each temporal cycle has world_size microbatches. Rotating schedules bias destination step mod world_size, so every rank is hot once and aggregate receiver load is balanced over the cycle.",
                "Persistent schedules bias rank 0 in every microbatch. Steady balanced uses zero bias.",
                "The full path includes router projection, fused top-k, token duplication and sort, variable hidden/id/weight dispatch, local fused experts, return all-to-all, and weighted combine.",
                "Cycle timings are authoritative for throughput. Per-step synchronized samples expose p50/p95/p99/max tail latency.",
                "Every microbatch is checked against its corresponding global-expert reference; route/output drift is measured against the same hidden states under zero bias.",
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
    parser.add_argument("--max-nrmse", type=float, default=5e-3)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
