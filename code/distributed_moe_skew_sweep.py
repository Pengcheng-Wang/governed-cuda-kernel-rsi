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
        bias_values = [0.0, 1.0, 2.0, 4.0]
        warmup, iterations = 1, 3
    else:
        token_values = [512, 2048]
        bias_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
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
        zero_bias = routing_bias(
            "global_biased",
            rank=rank,
            num_experts=num_experts,
            experts_per_rank=args.experts_per_rank,
            hot_bias=0.0,
            device=device,
        )
        zero_route = route_and_pack(
            hidden_states,
            gate_weight,
            zero_bias,
            tie_breaker,
            top_k=args.top_k,
            experts_per_rank=args.experts_per_rank,
            world_size=world_size,
        )
        zero_ids = zero_route["topk_ids"]
        zero_weights = zero_route["topk_weights"]
        assert isinstance(zero_ids, torch.Tensor)
        assert isinstance(zero_weights, torch.Tensor)
        zero_output = fused_experts(
            hidden_states,
            global_w1,
            global_w2,
            zero_weights,
            zero_ids,
        )

        for hot_bias in bias_values:
            bias = routing_bias(
                "global_biased",
                rank=rank,
                num_experts=num_experts,
                experts_per_rank=args.experts_per_rank,
                hot_bias=hot_bias,
                device=device,
            )

            def build_route() -> TensorDict:
                return route_and_pack(
                    hidden_states,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                )

            prepared_route = build_route()
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

            topk_ids = prepared_route["topk_ids"]
            topk_weights = prepared_route["topk_weights"]
            send_counts = prepared_route["send_counts"]
            assert isinstance(topk_ids, torch.Tensor)
            assert isinstance(topk_weights, torch.Tensor)
            assert isinstance(send_counts, list)
            reference_output = fused_experts(
                hidden_states,
                global_w1,
                global_w2,
                topk_weights,
                topk_ids,
            )
            torch.cuda.synchronize()
            correctness = output_quality(
                distributed_output,
                reference_output,
            )
            if (
                correctness["normalized_rmse"] > args.max_nrmse
                or correctness["cosine_similarity"] < args.min_cosine
            ):
                raise RuntimeError(
                    f"Skew correctness failed for T={num_tokens}, "
                    f"bias={hot_bias}: {correctness}"
                )
            output_drift = output_quality(
                reference_output,
                zero_output,
            )

            local_slot_match = (
                topk_ids == zero_ids
            ).float().mean().item()
            local_set_match = (
                torch.sort(topk_ids, dim=-1).values
                == torch.sort(zero_ids, dim=-1).values
            ).all(dim=-1).float().mean().item()
            slot_match = statistics.fmean(
                gather_scalar(local_slot_match, device)
            )
            set_match = statistics.fmean(
                gather_scalar(local_set_match, device)
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
            total_assignments = sum(receive_totals)
            mean_receive = statistics.fmean(receive_totals)
            diagonal_assignments = sum(
                send_matrix[source][source]
                for source in range(world_size)
            )
            probabilities = [
                count / total_assignments
                for count in receive_totals
                if count
            ]
            destination_entropy = -sum(
                probability * math.log(probability)
                for probability in probabilities
            )

            expert_counts = torch.bincount(
                topk_ids.reshape(-1).to(torch.int64),
                minlength=num_experts,
            ).to(torch.float64)
            dist.all_reduce(expert_counts)
            expert_totals = expert_counts.cpu().tolist()
            mean_expert = statistics.fmean(expert_totals)

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
                route = build_route()
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
                ("router_pack", build_route),
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
                    "routing_pattern": "global_bias_sweep",
                    "hot_bias": hot_bias,
                    "dtype": str(dtype),
                    "routing": {
                        "send_matrix": send_matrix,
                        "rank_receive_assignments": receive_totals,
                        "total_assignments": total_assignments,
                        "rank0_receive_fraction": (
                            receive_totals[0] / total_assignments
                        ),
                        "max_to_mean_receive": (
                            max(receive_totals) / mean_receive
                        ),
                        "receive_cv": (
                            statistics.pstdev(receive_totals)
                            / mean_receive
                        ),
                        "locality_fraction": (
                            diagonal_assignments / total_assignments
                        ),
                        "normalized_destination_entropy": (
                            destination_entropy
                            / math.log(world_size)
                        ),
                        "effective_destinations": math.exp(
                            destination_entropy
                        ),
                        "hot_expert_fraction": (
                            sum(
                                expert_totals[
                                    : args.experts_per_rank
                                ]
                            )
                            / total_assignments
                        ),
                        "max_to_mean_expert_load": (
                            max(expert_totals) / mean_expert
                        ),
                        "expert_load_cv": (
                            statistics.pstdev(expert_totals)
                            / mean_expert
                        ),
                    },
                    "route_change_vs_zero_bias": {
                        "topk_slot_match_fraction": slot_match,
                        "topk_set_match_fraction": set_match,
                    },
                    "output_drift_vs_zero_bias": {
                        "max_abs_error": max(
                            gather_scalar(
                                output_drift["max_abs_error"],
                                device,
                            )
                        ),
                        "max_normalized_rmse": max(
                            gather_scalar(
                                output_drift["normalized_rmse"],
                                device,
                            )
                        ),
                        "min_cosine_similarity": min(
                            gather_scalar(
                                output_drift["cosine_similarity"],
                                device,
                            )
                        ),
                    },
                    "correctness": {
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
                reference_output,
            )
            torch.cuda.empty_cache()

        del zero_route, zero_output
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_skew_sweep",
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
                "The sweep adds a common FP32 bias to rank 0's experts before fused top-k while keeping inputs, gate weights, and expert weights fixed.",
                "Bias 0 is the balanced reference; increasing bias measures the onset of a learned-router destination hotspot rather than a hand-authored split vector.",
                "The integrated path includes router projection, fused top-k, token duplication and sort, variable hidden/id/weight dispatch, local fused experts, return all-to-all, and weighted combine.",
                "Route match and output drift are measured against bias 0. Distributed correctness is independently checked against a global-expert reference for every bias.",
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
    parser.add_argument("--max-nrmse", type=float, default=5e-3)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
