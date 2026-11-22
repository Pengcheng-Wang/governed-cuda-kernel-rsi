from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def gather_scalar(value: float, device: torch.device) -> list[float]:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [item.item() for item in gathered]


def planned_send_counts(
    total_tokens: int,
    world_size: int,
    rank: int,
    pattern: str,
) -> list[int]:
    if pattern == "balanced":
        base, remainder = divmod(total_tokens, world_size)
        return [
            base + (1 if destination < remainder else 0)
            for destination in range(world_size)
        ]

    hot_tokens = int(round(0.8 * total_tokens))
    hot_destination = 0 if pattern == "global_hot80" else rank
    counts = [0] * world_size
    counts[hot_destination] = hot_tokens
    remaining = total_tokens - hot_tokens
    other_destinations = [
        destination
        for destination in range(world_size)
        if destination != hot_destination
    ]
    base, remainder = divmod(remaining, len(other_destinations))
    for index, destination in enumerate(other_destinations):
        counts[destination] = base + (1 if index < remainder else 0)
    return counts


def accepted_send_matrix(
    planned: list[list[int]],
    capacity_factor: float | None,
) -> tuple[list[list[int]], int | None]:
    world_size = len(planned)
    if capacity_factor is None:
        return [row[:] for row in planned], None

    total_assignments = sum(sum(row) for row in planned)
    average_receive = total_assignments / world_size
    capacity = int(math.ceil(capacity_factor * average_receive))
    accepted = [[0] * world_size for _ in range(world_size)]
    for destination in range(world_size):
        remaining = capacity
        for source in range(world_size):
            accepted_count = min(planned[source][destination], remaining)
            accepted[source][destination] = accepted_count
            remaining -= accepted_count
    return accepted, capacity


def exchange_counts(
    send_counts: torch.Tensor,
) -> torch.Tensor:
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    return recv_counts


def validate_markers(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    rank: int,
    send_counts: list[int],
    recv_counts: list[int],
) -> float:
    offset = 0
    for destination, count in enumerate(send_counts):
        if count:
            input_tensor[offset : offset + count, 0].fill_(
                rank + destination / 128.0
            )
        offset += count

    dist.all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    torch.cuda.synchronize()

    max_error = 0.0
    offset = 0
    for source, count in enumerate(recv_counts):
        if count:
            expected = source + rank / 128.0
            error = (
                output_tensor[offset, 0]
                .float()
                .sub(expected)
                .abs()
                .item()
            )
            max_error = max(max_error, error)
        offset += count
    return max_error


def timed_variable_alltoall(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    send_counts: list[int],
    recv_counts: list[int],
    warmup: int,
    iterations: int,
    local_rank: int,
) -> list[float]:
    for _ in range(warmup):
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])

    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1_000.0)
    dist.barrier(device_ids=[local_rank])
    return samples_ms


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
        patterns = ["balanced", "global_hot80", "rotated_hot80"]
        capacity_factors: list[float | None] = [None, 1.25]
        warmup, iterations = 2, 4
    else:
        token_values = [256, 1024, 4096]
        patterns = ["balanced", "global_hot80", "rotated_hot80"]
        capacity_factors = [None, 1.25]
        warmup, iterations = 10, 30

    element_size = torch.empty((), dtype=dtype).element_size()
    records: list[dict[str, Any]] = []
    for tokens_per_rank in token_values:
        for pattern in patterns:
            local_planned = planned_send_counts(
                tokens_per_rank,
                world_size,
                rank,
                pattern,
            )
            local_planned_tensor = torch.tensor(
                local_planned,
                device=device,
                dtype=torch.int64,
            )
            gathered_planned = [
                torch.empty_like(local_planned_tensor)
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_planned, local_planned_tensor)
            planned_matrix = [
                tensor.cpu().tolist()
                for tensor in gathered_planned
            ]

            for capacity_factor in capacity_factors:
                accepted_matrix, destination_capacity = (
                    accepted_send_matrix(
                        planned_matrix,
                        capacity_factor,
                    )
                )
                send_counts = accepted_matrix[rank]
                send_counts_tensor = torch.tensor(
                    send_counts,
                    device=device,
                    dtype=torch.int64,
                )
                recv_counts_tensor = exchange_counts(send_counts_tensor)
                recv_counts = recv_counts_tensor.cpu().tolist()

                send_tokens = sum(send_counts)
                recv_tokens = sum(recv_counts)
                input_tensor = torch.randn(
                    (send_tokens, args.hidden),
                    device=device,
                    dtype=dtype,
                )
                output_tensor = torch.empty(
                    (recv_tokens, args.hidden),
                    device=device,
                    dtype=dtype,
                )
                local_marker_error = validate_markers(
                    input_tensor,
                    output_tensor,
                    rank=rank,
                    send_counts=send_counts,
                    recv_counts=recv_counts,
                )
                samples_ms = timed_variable_alltoall(
                    input_tensor,
                    output_tensor,
                    send_counts=send_counts,
                    recv_counts=recv_counts,
                    warmup=warmup,
                    iterations=iterations,
                    local_rank=local_rank,
                )
                local_median_ms = statistics.median(samples_ms)
                rank_medians = gather_scalar(local_median_ms, device)
                send_totals = gather_scalar(float(send_tokens), device)
                recv_totals = gather_scalar(float(recv_tokens), device)
                marker_errors = gather_scalar(
                    local_marker_error,
                    device,
                )
                critical_path_ms = max(rank_medians)

                planned_total = sum(sum(row) for row in planned_matrix)
                accepted_total = sum(
                    sum(row) for row in accepted_matrix
                )
                dropped_total = planned_total - accepted_total
                max_recv = max(recv_totals)
                mean_recv = statistics.fmean(recv_totals)
                total_payload_bytes = accepted_total * args.hidden * element_size
                records.append(
                    {
                        "tokens_per_rank": tokens_per_rank,
                        "hidden": args.hidden,
                        "dtype": str(dtype),
                        "routing_pattern": pattern,
                        "capacity_factor": capacity_factor,
                        "destination_capacity": destination_capacity,
                        "planned_assignments": planned_total,
                        "accepted_assignments": accepted_total,
                        "dropped_assignments": dropped_total,
                        "dropped_fraction": (
                            dropped_total / planned_total
                            if planned_total
                            else 0.0
                        ),
                        "send_counts": send_counts,
                        "recv_counts": recv_counts,
                        "rank_send_tokens": send_totals,
                        "rank_recv_tokens": recv_totals,
                        "max_to_mean_receive": (
                            max_recv / mean_recv
                            if mean_recv
                            else 0.0
                        ),
                        "receive_cv": (
                            statistics.pstdev(recv_totals) / mean_recv
                            if mean_recv
                            else 0.0
                        ),
                        "critical_path_ms": critical_path_ms,
                        "rank_median_ms": rank_medians,
                        "rank_median_spread_ms": (
                            max(rank_medians) - min(rank_medians)
                        ),
                        "rank_median_cv": (
                            statistics.pstdev(rank_medians)
                            / statistics.fmean(rank_medians)
                        ),
                        "aggregate_logical_payload_gb_s": (
                            total_payload_bytes
                            / (critical_path_ms / 1_000.0)
                            / 1e9
                        ),
                        "max_validation_error": max(marker_errors),
                        "warmup": warmup,
                        "iterations": iterations,
                    }
                )

                del input_tensor, output_tensor
                torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_variable_alltoall",
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
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "balanced sends equal counts to every destination.",
                "global_hot80 sends 80% of every source rank's tokens to destination rank 0.",
                "rotated_hot80 sends 80% to the source rank's matching destination, preserving global receiver balance while skewing each sender's split vector.",
                "Capacity clipping uses a deterministic source-rank order and a per-destination capacity equal to ceil(capacity_factor * average planned receive count).",
                "Aggregate logical payload bandwidth counts accepted BF16 token bytes once and is not a physical fabric bus-bandwidth normalization.",
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
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
