from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import ProfilerActivity, profile

from distributed_training import TransformerStack


STRATEGIES = (
    "ddp",
    "fsdp_root",
    "fsdp_layerwise_reshard",
    "fsdp_layerwise_no_reshard",
)


def local_parameter_bytes(module: torch.nn.Module) -> int:
    total = 0
    for parameter in module.parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        total += local.numel() * local.element_size()
    return total


def make_model(
    *,
    strategy: str,
    world_size: int,
    device: torch.device,
    depth: int,
    hidden: int,
    heads: int,
    intermediate: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = TransformerStack(depth, hidden, heads, intermediate).to(
        device=device,
        dtype=torch.float32,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata: dict[str, Any] = {
        "parameter_count": parameter_count,
        "master_parameter_dtype": "torch.float32",
        "compute_dtype": "torch.bfloat16",
        "reduce_dtype": "torch.float32",
    }

    if strategy == "ddp":
        wrapped = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            gradient_as_bucket_view=True,
            static_graph=True,
            bucket_cap_mb=25,
        )
        metadata["fsdp_groups"] = 0
        metadata["reshard_after_forward"] = None
        return wrapped, metadata

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    if strategy == "fsdp_root":
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=mp_policy,
        )
        metadata["fsdp_groups"] = 1
        metadata["reshard_after_forward"] = True
        return model, metadata

    reshard_after_forward = strategy == "fsdp_layerwise_reshard"
    for block in model.blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=False,
        mp_policy=mp_policy,
    )
    metadata["fsdp_groups"] = depth
    metadata["reshard_after_forward"] = reshard_after_forward
    return model, metadata


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(x)
        loss = output.float().square().mean()
    loss.backward()
    optimizer.step()
    return loss


def validation_step(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(x)
        loss = output.float().square().mean()
    loss.backward()
    return loss


def profiler_summary(
    step: Callable[[], torch.Tensor],
    rank: int,
    local_rank: int,
    top_k: int = 30,
) -> dict[str, Any] | None:
    dist.barrier(device_ids=[local_rank])
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        step()
    dist.barrier(device_ids=[local_rank])
    if rank != 0:
        return None

    events: list[dict[str, Any]] = []
    for event in prof.key_averages():
        self_device_us = float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
        device_us = float(
            getattr(
                event,
                "device_time_total",
                getattr(event, "cuda_time_total", 0.0),
            )
        )
        if self_device_us <= 0.0 and device_us <= 0.0:
            continue
        events.append(
            {
                "name": event.key,
                "count": event.count,
                "self_device_time_us": self_device_us,
                "device_time_us": device_us,
            }
        )
    events.sort(key=lambda item: item["self_device_time_us"], reverse=True)
    communication = [
        item
        for item in events
        if any(
            token in item["name"].lower()
            for token in ("nccl", "all_gather", "reduce_scatter", "allreduce")
        )
    ]
    return {
        "top_cuda_events": events[:top_k],
        "communication_events": communication[:top_k],
        "table": prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=top_k,
        ),
    }


def gather_scalar(value: float, device: torch.device) -> list[float]:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [item.item() for item in gathered]


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(41)
    torch.cuda.manual_seed_all(41)
    torch.backends.cuda.matmul.allow_tf32 = False

    if args.profile == "smoke":
        depth, batch, sequence, hidden, heads, intermediate = (
            2,
            1,
            128,
            512,
            8,
            1408,
        )
        warmup, iterations = 1, 2
    else:
        depth, batch, sequence, hidden, heads, intermediate = (
            12,
            1,
            256,
            2048,
            16,
            5504,
        )
        warmup, iterations = 2, 5

    model, model_metadata = make_model(
        strategy=args.strategy,
        world_size=world_size,
        device=device,
        depth=depth,
        hidden=hidden,
        heads=heads,
        intermediate=intermediate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    x = torch.randn(
        (batch, sequence, hidden),
        device=device,
        dtype=torch.bfloat16,
    )

    initial_loss = validation_step(model, x)
    torch.cuda.synchronize()
    initial_loss_value = initial_loss.detach().item()
    optimizer.zero_grad(set_to_none=True)

    for _ in range(warmup):
        train_step(model, optimizer, x)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])

    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    samples_ms: list[float] = []
    final_loss_value = 0.0
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        final_loss = train_step(model, optimizer, x)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
        final_loss_value = final_loss.detach().item()
    dist.barrier(device_ids=[local_rank])

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    rank_median_ms = gather_scalar(statistics.median(samples_ms), device)
    rank_initial_losses = gather_scalar(initial_loss_value, device)
    rank_final_losses = gather_scalar(final_loss_value, device)
    rank_baseline_allocated = gather_scalar(float(baseline_allocated), device)
    rank_baseline_reserved = gather_scalar(float(baseline_reserved), device)
    rank_peak_allocated = gather_scalar(float(peak_allocated), device)
    rank_peak_reserved = gather_scalar(float(peak_reserved), device)
    rank_local_parameter_bytes = gather_scalar(
        float(local_parameter_bytes(model)),
        device,
    )
    global_step_ms = max(rank_median_ms)
    global_tokens = batch * sequence * world_size

    profile_result = None
    if args.include_profiler:
        profile_result = profiler_summary(
            lambda: train_step(model, optimizer, x),
            rank,
            local_rank,
        )

    result = None
    if rank == 0:
        result = {
            "suite": "fsdp2_training",
            "profile": args.profile,
            "strategy": args.strategy,
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
                **model_metadata,
            },
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-4,
                "betas": [0.9, 0.95],
                "weight_decay": 0.1,
            },
            "per_gpu_batch": batch,
            "sequence": sequence,
            "global_tokens_per_step": global_tokens,
            "rank_median_ms": rank_median_ms,
            "global_step_ms": global_step_ms,
            "global_tokens_per_second": global_tokens / (global_step_ms / 1_000.0),
            "rank_initial_losses": rank_initial_losses,
            "rank_final_losses": rank_final_losses,
            "rank_local_parameter_bytes": rank_local_parameter_bytes,
            "rank_baseline_allocated_bytes": rank_baseline_allocated,
            "rank_baseline_reserved_bytes": rank_baseline_reserved,
            "rank_peak_allocated_bytes": rank_peak_allocated,
            "rank_peak_reserved_bytes": rank_peak_reserved,
            "max_peak_allocated_bytes": max(rank_peak_allocated),
            "max_peak_reserved_bytes": max(rank_peak_reserved),
            "max_peak_incremental_allocated_bytes": max(
                peak - baseline
                for peak, baseline in zip(
                    rank_peak_allocated,
                    rank_baseline_allocated,
                )
            ),
            "profiler_summary": profile_result,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
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
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--include-profiler", action="store_true")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

