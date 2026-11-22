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
from torch.nn.parallel import DistributedDataParallel

from bench_transformer_training import TransformerBlock


class TransformerStack(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        hidden: int,
        heads: int,
        intermediate: int,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            TransformerBlock(hidden, heads, intermediate) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def train_step(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    output = model(x)
    loss = output.float().square().mean()
    loss.backward()
    return loss


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(31)

    if args.profile == "smoke":
        depth, batch, sequence, hidden, heads, intermediate = (2, 1, 128, 512, 8, 1408)
        warmup, iterations = 2, 3
    else:
        depth, batch, sequence, hidden, heads, intermediate = (
            8,
            1,
            512,
            1024,
            16,
            2816,
        )
        warmup, iterations = 3, 10

    model = TransformerStack(depth, hidden, heads, intermediate).to(
        device=device,
        dtype=torch.bfloat16,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
        static_graph=True,
        bucket_cap_mb=25,
    )
    x = torch.randn(
        (batch, sequence, hidden),
        device=device,
        dtype=torch.bfloat16,
    )

    for _ in range(warmup):
        train_step(ddp, x)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])

    samples_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_step(ddp, x)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
    dist.barrier(device_ids=[local_rank])

    local_median = torch.tensor(
        [statistics.median(samples_ms)],
        device=device,
        dtype=torch.float64,
    )
    medians = [torch.zeros_like(local_median) for _ in range(world_size)]
    dist.all_gather(medians, local_median)
    rank_medians = [item.item() for item in medians]
    global_step_ms = max(rank_medians)

    result = None
    if rank == 0:
        global_tokens = batch * sequence * world_size
        result = {
            "suite": "distributed_transformer_training",
            "profile": args.profile,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "model": {
                "depth": depth,
                "hidden": hidden,
                "heads": heads,
                "intermediate": intermediate,
                "parameter_count": parameter_count,
            },
            "per_gpu_batch": batch,
            "sequence": sequence,
            "global_tokens_per_step": global_tokens,
            "rank_median_ms": rank_medians,
            "global_step_ms": global_step_ms,
            "global_tokens_per_second": global_tokens / (global_step_ms / 1_000.0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
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

