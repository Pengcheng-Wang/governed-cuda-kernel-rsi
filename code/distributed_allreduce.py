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


def timed_all_reduce(tensor: torch.Tensor, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_reduce(tensor)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    dist.barrier()
    return samples


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    sizes_mb = [1, 16] if args.profile == "smoke" else [1, 16, 64, 256]
    warmup = 3 if args.profile == "smoke" else 10
    iterations = 5 if args.profile == "smoke" else 30
    records: list[dict[str, Any]] = []
    for size_mb in sizes_mb:
        elements = size_mb * 1024 * 1024 // 2
        tensor = torch.ones(elements, dtype=torch.bfloat16, device=device)
        samples_ms = timed_all_reduce(tensor, warmup, iterations)
        median_ms = statistics.median(samples_ms)
        bytes_count = tensor.numel() * tensor.element_size()
        algorithm_gb_s = bytes_count / (median_ms / 1_000.0) / 1e9
        bus_gb_s = algorithm_gb_s * 2.0 * (world_size - 1) / world_size
        records.append(
            {
                "size_mb": size_mb,
                "median_ms": median_ms,
                "min_ms": min(samples_ms),
                "max_ms": max(samples_ms),
                "algorithm_gb_s": algorithm_gb_s,
                "bus_gb_s": bus_gb_s,
            }
        )
        del tensor

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_allreduce",
            "profile": args.profile,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "records": records,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

