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
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)


TensorDict = dict[str, torch.Tensor | list[int]]


def gather_scalar(value: float, device: torch.device) -> list[float]:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [item.item() for item in gathered]


def critical_path_summary(
    samples_ms: list[float],
    device: torch.device,
) -> dict[str, Any]:
    local_median_ms = statistics.median(samples_ms)
    rank_medians = gather_scalar(local_median_ms, device)
    mean_rank_median = statistics.fmean(rank_medians)
    return {
        "critical_path_ms": max(rank_medians),
        "rank_median_ms": rank_medians,
        "rank_median_spread_ms": max(rank_medians) - min(rank_medians),
        "rank_median_cv": (
            statistics.pstdev(rank_medians) / mean_rank_median
            if mean_rank_median
            else 0.0
        ),
        "local_samples_ms": samples_ms,
    }


def timed_schedule(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    local_rank: int,
) -> list[float]:
    for _ in range(warmup):
        output = fn()
        torch.cuda.synchronize()
        del output

    samples_ms: list[float] = []
    for _ in range(iterations):
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1_000.0)
        del output
    dist.barrier(device_ids=[local_rank])
    return samples_ms


def output_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    return {
        "max_abs_error": error.abs().max().item(),
        "mean_abs_error": error.abs().mean().item(),
        "normalized_rmse": (
            error.square().mean().sqrt()
            / expected_f.square().mean().sqrt().clamp_min(1e-8)
        ).item(),
        "cosine_similarity": F.cosine_similarity(
            actual_f.flatten(),
            expected_f.flatten(),
            dim=0,
        ).item(),
    }


def routing_bias(
    pattern: str,
    *,
    rank: int,
    num_experts: int,
    experts_per_rank: int,
    hot_bias: float,
    device: torch.device,
) -> torch.Tensor:
    bias = torch.zeros(num_experts, device=device, dtype=torch.float32)
    if pattern == "global_biased":
        bias[:experts_per_rank] = hot_bias
    elif pattern == "locality_biased":
        start = rank * experts_per_rank
        bias[start : start + experts_per_rank] = hot_bias
    elif pattern != "balanced":
        raise ValueError(f"Unknown routing pattern: {pattern}")
    return bias


def route_and_pack(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    *,
    top_k: int,
    experts_per_rank: int,
    world_size: int,
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
    destinations = torch.div(
        flat_ids,
        experts_per_rank,
        rounding_mode="floor",
    )
    local_ids = torch.remainder(flat_ids, experts_per_rank)
    permutation = torch.argsort(destinations, stable=True)

    sorted_destinations = destinations.index_select(0, permutation)
    sorted_token_indices = token_indices.index_select(0, permutation)
    sorted_local_ids = local_ids.index_select(0, permutation)
    sorted_weights = topk_weights.reshape(-1).index_select(
        0,
        permutation,
    )
    send_hidden = hidden_states.index_select(0, sorted_token_indices)
    send_counts_tensor = torch.bincount(
        sorted_destinations,
        minlength=world_size,
    ).to(torch.int64)
    send_counts = send_counts_tensor.cpu().tolist()

    return {
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "send_hidden": send_hidden.contiguous(),
        "send_local_ids": sorted_local_ids.to(torch.int32)
        .reshape(-1, 1)
        .contiguous(),
        "send_weights": sorted_weights.reshape(-1, 1).contiguous(),
        "sorted_token_indices": sorted_token_indices,
        "sorted_assignment_indices": permutation,
        "send_counts_tensor": send_counts_tensor,
        "send_counts": send_counts,
    }


def dispatch_assignments(route: TensorDict) -> TensorDict:
    send_hidden = route["send_hidden"]
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    send_counts_tensor = route["send_counts_tensor"]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_local_ids, torch.Tensor)
    assert isinstance(send_weights, torch.Tensor)
    assert isinstance(send_counts_tensor, torch.Tensor)
    assert isinstance(send_counts, list)

    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    recv_counts = recv_counts_tensor.cpu().tolist()
    recv_total = sum(recv_counts)

    recv_hidden = torch.empty(
        (recv_total, send_hidden.shape[1]),
        device=send_hidden.device,
        dtype=send_hidden.dtype,
    )
    recv_local_ids = torch.empty(
        (recv_total, 1),
        device=send_hidden.device,
        dtype=torch.int32,
    )
    recv_weights = torch.empty(
        (recv_total, 1),
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
        recv_local_ids,
        send_local_ids,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    dist.all_to_all_single(
        recv_weights,
        send_weights,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    return {
        "recv_hidden": recv_hidden,
        "recv_local_ids": recv_local_ids,
        "recv_weights": recv_weights,
        "recv_counts": recv_counts,
    }


def local_experts(
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
    if recv_hidden.shape[0] == 0:
        return torch.empty_like(recv_hidden)
    return fused_experts(
        recv_hidden,
        local_w1,
        local_w2,
        recv_weights,
        recv_local_ids,
    )


def return_and_combine(
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
    return combined.to(returned.dtype)


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
        warmup, iterations = 1, 3
    else:
        token_values = [128, 512, 2048]
        warmup, iterations = 3, 10
    patterns = ["balanced", "global_biased", "locality_biased"]
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

        for pattern in patterns:
            bias = routing_bias(
                pattern,
                rank=rank,
                num_experts=num_experts,
                experts_per_rank=args.experts_per_rank,
                hot_bias=args.hot_bias,
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
            assert isinstance(topk_ids, torch.Tensor)
            assert isinstance(topk_weights, torch.Tensor)
            reference_output = fused_experts(
                hidden_states,
                global_w1,
                global_w2,
                topk_weights,
                topk_ids,
            )
            torch.cuda.synchronize()
            quality = output_quality(
                distributed_output,
                reference_output,
            )
            if (
                quality["normalized_rmse"] > args.max_nrmse
                or quality["cosine_similarity"] < args.min_cosine
            ):
                raise RuntimeError(
                    f"Distributed MoE correctness failed for "
                    f"T={num_tokens}, pattern={pattern}: {quality}"
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
                sum(send_matrix[source][destination] for source in range(world_size))
                for destination in range(world_size)
            ]
            total_assignments = sum(receive_totals)
            mean_receive = statistics.fmean(receive_totals)
            diagonal_assignments = sum(
                send_matrix[source][source]
                for source in range(world_size)
            )

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

            def local_reference() -> torch.Tensor:
                route = build_route()
                ids = route["topk_ids"]
                weights = route["topk_weights"]
                assert isinstance(ids, torch.Tensor)
                assert isinstance(weights, torch.Tensor)
                return fused_experts(
                    hidden_states,
                    global_w1,
                    global_w2,
                    weights,
                    ids,
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
                ("local_reference", local_reference),
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

            local_quality = quality
            records.append(
                {
                    "tokens_per_rank": num_tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts_per_rank": args.experts_per_rank,
                    "num_experts": num_experts,
                    "top_k": args.top_k,
                    "routing_pattern": pattern,
                    "hot_bias": args.hot_bias,
                    "dtype": str(dtype),
                    "routing": {
                        "send_matrix": send_matrix,
                        "rank_receive_assignments": receive_totals,
                        "total_assignments": total_assignments,
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
                        "locality_fraction": (
                            diagonal_assignments / total_assignments
                            if total_assignments
                            else 0.0
                        ),
                        "rank0_receive_fraction": (
                            receive_totals[0] / total_assignments
                            if total_assignments
                            else 0.0
                        ),
                    },
                    "correctness": {
                        "max_abs_error": max(
                            gather_scalar(
                                local_quality["max_abs_error"],
                                device,
                            )
                        ),
                        "max_mean_abs_error": max(
                            gather_scalar(
                                local_quality["mean_abs_error"],
                                device,
                            )
                        ),
                        "max_normalized_rmse": max(
                            gather_scalar(
                                local_quality["normalized_rmse"],
                                device,
                            )
                        ),
                        "min_cosine_similarity": min(
                            gather_scalar(
                                local_quality["cosine_similarity"],
                                device,
                            )
                        ),
                    },
                    "timings": timings,
                    "distributed_vs_local_reference": (
                        timings["distributed_end_to_end"][
                            "critical_path_ms"
                        ]
                        / timings["local_reference"]["critical_path_ms"]
                    ),
                    "router_pack_fraction_of_end_to_end": (
                        timings["router_pack"]["critical_path_ms"]
                        / timings["distributed_end_to_end"][
                            "critical_path_ms"
                        ]
                    ),
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

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_end_to_end",
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
                "The integrated path includes BF16 router projection, vLLM fused top-k, token duplication and destination sort, count exchange, variable-split hidden/id/weight dispatch, local vLLM fused experts, variable-split return, and FP32 weighted index-add combine.",
                "Every rank owns experts_per_rank experts; global expert weights are retained only to build a correctness and local-reference baseline.",
                "global_biased adds the same positive bias to rank 0's experts on every source; locality_biased adds it to each source rank's own experts.",
                "Separate stage timings reuse prepared tensors and are diagnostic rather than additive. distributed_end_to_end is the authoritative integrated critical path.",
                "Critical path is the slowest rank's median CPU wall time with CUDA synchronization and a rank barrier around each measured iteration.",
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
