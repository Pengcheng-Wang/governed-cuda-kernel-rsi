from __future__ import annotations

import argparse
import json
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


def validate_alltoall(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    elements_per_peer: int,
) -> float:
    for destination in range(world_size):
        start = destination * elements_per_peer
        end = start + elements_per_peer
        input_tensor[start:end].fill_(
            rank + destination / 128.0
        )
    dist.all_to_all_single(output_tensor, input_tensor)
    torch.cuda.synchronize()
    max_error = 0.0
    for source in range(world_size):
        start = source * elements_per_peer
        expected = source + rank / 128.0
        error = (
            output_tensor[start]
            .float()
            .sub(expected)
            .abs()
            .item()
        )
        max_error = max(max_error, error)
    return max_error


def timed_alltoall(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    local_rank: int,
) -> list[float]:
    for _ in range(warmup):
        dist.all_to_all_single(output_tensor, input_tensor)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_to_all_single(output_tensor, input_tensor)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    dist.barrier(device_ids=[local_rank])
    return samples


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    hidden = args.hidden
    element_size = torch.empty((), dtype=dtype).element_size()
    if args.profile == "smoke":
        tokens_per_peer_values = [1, 16]
        warmup, iterations = 3, 5
    else:
        tokens_per_peer_values = [
            1,
            4,
            16,
            64,
            256,
            1024,
            4096,
        ]
        warmup, iterations = 10, 30

    records: list[dict[str, Any]] = []
    for tokens_per_peer in tokens_per_peer_values:
        elements_per_peer = tokens_per_peer * hidden
        total_elements = elements_per_peer * world_size
        input_tensor = torch.randn(
            total_elements,
            device=device,
            dtype=dtype,
        )
        output_tensor = torch.empty_like(input_tensor)
        validation_error = validate_alltoall(
            input_tensor,
            output_tensor,
            rank=rank,
            world_size=world_size,
            elements_per_peer=elements_per_peer,
        )
        max_errors = gather_scalar(validation_error, device)
        samples_ms = timed_alltoall(
            input_tensor,
            output_tensor,
            warmup=warmup,
            iterations=iterations,
            local_rank=local_rank,
        )
        local_median_ms = statistics.median(samples_ms)
        rank_medians = gather_scalar(local_median_ms, device)
        critical_path_ms = max(rank_medians)
        per_peer_bytes = elements_per_peer * element_size
        per_rank_send_bytes = per_peer_bytes * world_size
        per_rank_injection_gb_s = (
            per_rank_send_bytes
            / (critical_path_ms / 1_000.0)
            / 1e9
        )
        records.append(
            {
                "tokens_per_peer": tokens_per_peer,
                "hidden": hidden,
                "dtype": str(dtype),
                "bytes_per_token": hidden * element_size,
                "per_peer_bytes": per_peer_bytes,
                "per_rank_send_bytes": per_rank_send_bytes,
                "critical_path_ms": critical_path_ms,
                "rank_median_ms": rank_medians,
                "rank_median_spread_ms": max(rank_medians)
                - min(rank_medians),
                "rank_median_cv": (
                    statistics.pstdev(rank_medians)
                    / statistics.fmean(rank_medians)
                ),
                "per_rank_injection_gb_s": per_rank_injection_gb_s,
                "aggregate_cluster_payload_gb_s": per_rank_injection_gb_s
                * world_size,
                "max_validation_error": max(max_errors),
                "warmup": warmup,
                "iterations": iterations,
            }
        )
        del input_tensor, output_tensor
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_alltoall",
            "profile": args.profile,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "Each rank sends an equal BF16 token payload to every rank.",
                "Per-rank injection bandwidth divides all bytes sent by one rank by the slowest-rank median latency.",
                "Aggregate cluster payload bandwidth is a logical sum, not a physical fabric bus-bandwidth normalization.",
            ],
            "records": records,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

