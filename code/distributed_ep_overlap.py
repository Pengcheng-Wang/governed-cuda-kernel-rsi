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

from bench_fused_moe import make_routing, quality_metrics
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def parse_int_values(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"Expected positive comma-separated integers: {value}")
    return values


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
    return {
        "critical_path_ms": max(rank_medians),
        "rank_median_ms": rank_medians,
        "rank_median_spread_ms": max(rank_medians) - min(rank_medians),
        "rank_median_cv": (
            statistics.pstdev(rank_medians)
            / statistics.fmean(rank_medians)
        ),
        "local_samples_ms": samples_ms,
    }


def timed_schedule(
    fn: Callable[[], list[torch.Tensor] | None],
    *,
    warmup: int,
    iterations: int,
    local_rank: int,
) -> list[float]:
    for _ in range(warmup):
        outputs = fn()
        torch.cuda.synchronize()
        del outputs

    samples_ms: list[float] = []
    for _ in range(iterations):
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1_000.0)
        del outputs
    dist.barrier(device_ids=[local_rank])
    return samples_ms


def launch_alltoall(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
) -> dist.Work:
    return dist.all_to_all_single(
        output_tensor,
        input_tensor,
        async_op=True,
    )


def validate_dispatch_markers(
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    tokens_per_peer: int,
) -> float:
    max_error = 0.0
    for input_tensor, output_tensor in zip(inputs, outputs):
        for destination in range(world_size):
            start = destination * tokens_per_peer
            end = start + tokens_per_peer
            input_tensor[start:end, 0].fill_(
                rank + destination / 128.0
            )
        dist.all_to_all_single(output_tensor, input_tensor)
        torch.cuda.synchronize()
        for source in range(world_size):
            start = source * tokens_per_peer
            expected = source + rank / 128.0
            error = (
                output_tensor[start, 0]
                .float()
                .sub(expected)
                .abs()
                .item()
            )
            max_error = max(max_error, error)
    return max_error


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    torch.manual_seed(71)
    torch.cuda.manual_seed_all(71)

    if args.profile == "smoke":
        default_tokens_per_peer_values = [1, 4]
        default_chunks = 2
        warmup, iterations = 1, 3
    else:
        default_tokens_per_peer_values = [4, 16, 64, 256, 1024]
        default_chunks = 4
        warmup, iterations = 3, 10

    chunk_count_values = parse_int_values(args.chunk_count_values)
    total_tokens_per_peer_values = parse_int_values(
        args.total_tokens_per_peer_values
    )
    if (chunk_count_values is None) != (
        total_tokens_per_peer_values is None
    ):
        raise ValueError(
            "--chunk-count-values and --total-tokens-per-peer-values "
            "must be supplied together"
        )
    if chunk_count_values is None:
        configurations = [
            (
                default_chunks,
                tokens_per_peer,
                default_chunks * tokens_per_peer,
            )
            for tokens_per_peer in default_tokens_per_peer_values
        ]
        fixed_total_sweep = False
    else:
        configurations = []
        for total_tokens_per_peer in total_tokens_per_peer_values:
            for chunks in chunk_count_values:
                if total_tokens_per_peer % chunks != 0:
                    raise ValueError(
                        f"{total_tokens_per_peer} total tokens per peer "
                        f"is not divisible by {chunks} chunks"
                    )
                configurations.append(
                    (
                        chunks,
                        total_tokens_per_peer // chunks,
                        total_tokens_per_peer,
                    )
                )
        fixed_total_sweep = True

    generator = torch.Generator(device=device)
    generator.manual_seed(4_000)
    w1 = (
        torch.randn(
            (
                args.num_experts,
                2 * args.intermediate,
                args.hidden,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.hidden)
    )
    w2 = (
        torch.randn(
            (
                args.num_experts,
                args.hidden,
                args.intermediate,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.intermediate)
    )

    records: list[dict[str, Any]] = []
    for chunks, tokens_per_peer, total_tokens_per_peer in configurations:
        local_tokens = world_size * tokens_per_peer
        inputs: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        for chunk in range(chunks):
            chunk_generator = torch.Generator(device=device)
            chunk_generator.manual_seed(
                10_000 + rank * 100 + chunk
            )
            inputs.append(
                torch.randn(
                    (local_tokens, args.hidden),
                    device=device,
                    dtype=dtype,
                    generator=chunk_generator,
                )
            )
            outputs.append(torch.empty_like(inputs[-1]))

        marker_error = validate_dispatch_markers(
            inputs,
            outputs,
            rank=rank,
            world_size=world_size,
            tokens_per_peer=tokens_per_peer,
        )
        topk_ids, topk_weights, routing = make_routing(
            local_tokens,
            args.num_experts,
            args.top_k,
            args.routing,
            device=device,
            dtype=dtype,
        )

        def compute(tensor: torch.Tensor) -> torch.Tensor:
            return fused_experts(
                tensor,
                w1,
                w2,
                topk_weights,
                topk_ids,
            )

        def communication_only() -> None:
            for input_tensor, output_tensor in zip(inputs, outputs):
                work = launch_alltoall(output_tensor, input_tensor)
                work.wait()
            return None

        def compute_only() -> list[torch.Tensor]:
            return [compute(output_tensor) for output_tensor in outputs]

        def sequential() -> list[torch.Tensor]:
            expert_outputs: list[torch.Tensor] = []
            for input_tensor, output_tensor in zip(inputs, outputs):
                work = launch_alltoall(output_tensor, input_tensor)
                work.wait()
                expert_outputs.append(compute(output_tensor))
            return expert_outputs

        def pipelined() -> list[torch.Tensor]:
            expert_outputs: list[torch.Tensor] = []
            current_work = launch_alltoall(outputs[0], inputs[0])
            current_work.wait()
            for chunk in range(chunks - 1):
                next_work = launch_alltoall(
                    outputs[chunk + 1],
                    inputs[chunk + 1],
                )
                expert_outputs.append(compute(outputs[chunk]))
                next_work.wait()
                current_work = next_work
            expert_outputs.append(compute(outputs[-1]))
            del current_work
            return expert_outputs

        sequential_reference = sequential()
        pipelined_output = pipelined()
        torch.cuda.synchronize()
        local_max_abs_error = 0.0
        local_min_cosine = 1.0
        for actual, expected in zip(
            pipelined_output,
            sequential_reference,
        ):
            quality = quality_metrics(actual, expected)
            local_max_abs_error = max(
                local_max_abs_error,
                quality["max_abs_error"],
            )
            local_min_cosine = min(
                local_min_cosine,
                quality["cosine_similarity"],
            )
        del sequential_reference, pipelined_output

        communication = critical_path_summary(
            timed_schedule(
                communication_only,
                warmup=warmup,
                iterations=iterations,
                local_rank=local_rank,
            ),
            device,
        )
        expert_compute = critical_path_summary(
            timed_schedule(
                compute_only,
                warmup=warmup,
                iterations=iterations,
                local_rank=local_rank,
            ),
            device,
        )
        sequential_timing = critical_path_summary(
            timed_schedule(
                sequential,
                warmup=warmup,
                iterations=iterations,
                local_rank=local_rank,
            ),
            device,
        )
        pipelined_timing = critical_path_summary(
            timed_schedule(
                pipelined,
                warmup=warmup,
                iterations=iterations,
                local_rank=local_rank,
            ),
            device,
        )

        communication_ms = communication["critical_path_ms"]
        compute_ms = expert_compute["critical_path_ms"]
        sequential_ms = sequential_timing["critical_path_ms"]
        pipelined_ms = pipelined_timing["critical_path_ms"]
        communication_per_chunk_ms = communication_ms / chunks
        compute_per_chunk_ms = compute_ms / chunks
        ideal_pipeline_ms = (
            communication_per_chunk_ms
            + compute_per_chunk_ms
            + (chunks - 1)
            * max(communication_per_chunk_ms, compute_per_chunk_ms)
        )
        hideable_ms = max(0.0, sequential_ms - ideal_pipeline_ms)
        hidden_ms = max(0.0, sequential_ms - pipelined_ms)
        global_marker_error = max(
            gather_scalar(marker_error, device)
        )
        global_max_abs_error = max(
            gather_scalar(local_max_abs_error, device)
        )
        global_min_cosine = min(
            gather_scalar(local_min_cosine, device)
        )
        per_rank_send_bytes = (
            local_tokens
            * args.hidden
            * torch.empty((), dtype=dtype).element_size()
        )
        useful_flops_per_chunk = (
            6.0
            * local_tokens
            * args.top_k
            * args.hidden
            * args.intermediate
        )
        records.append(
            {
                "tokens_per_peer": tokens_per_peer,
                "total_tokens_per_peer": total_tokens_per_peer,
                "local_tokens_per_chunk": local_tokens,
                "total_local_tokens": local_tokens * chunks,
                "chunks": chunks,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "routing_pattern": args.routing,
                "routing": routing,
                "dtype": str(dtype),
                "per_rank_send_bytes_per_chunk": per_rank_send_bytes,
                "useful_flops_per_chunk": useful_flops_per_chunk,
                "communication_only": communication,
                "compute_only": expert_compute,
                "sequential": sequential_timing,
                "pipelined": pipelined_timing,
                "pipeline_speedup": sequential_ms / pipelined_ms,
                "saved_critical_path_ms": hidden_ms,
                "saved_critical_path_fraction": hidden_ms / sequential_ms,
                "ideal_pipeline_ms": ideal_pipeline_ms,
                "overlap_efficiency": (
                    hidden_ms / hideable_ms
                    if hideable_ms > 0.0
                    else 0.0
                ),
                "communication_fraction_of_separate_work": (
                    communication_ms / (communication_ms + compute_ms)
                ),
                "useful_compute_tflop_s": (
                    useful_flops_per_chunk
                    * chunks
                    / (compute_ms / 1_000.0)
                    / 1e12
                ),
                "max_dispatch_marker_error": global_marker_error,
                "pipeline_vs_sequential_max_abs_error": (
                    global_max_abs_error
                ),
                "pipeline_vs_sequential_min_cosine": (
                    global_min_cosine
                ),
                "warmup": warmup,
                "iterations": iterations,
            }
        )

        del inputs, outputs, topk_ids, topk_weights
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_ep_overlap",
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
                "Each microbatch dispatches equal BF16 token splits with NCCL all-to-all, then runs vLLM fused experts locally.",
                "The pipelined schedule overlaps dispatch of microbatch i+1 with fused-expert compute for microbatch i.",
                "Routing is precomputed and excluded; dispatch permutation, combine all-to-all, and router computation are outside this benchmark.",
                "Critical path is the slowest rank's median CPU wall time with CUDA synchronization before and after each schedule.",
            ],
            "fixed_total_chunk_sweep": fixed_total_sweep,
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
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--routing",
        choices=["uniform", "hot80"],
        default="uniform",
    )
    parser.add_argument(
        "--chunk-count-values",
        help="Comma-separated chunk counts for a fixed-total sweep.",
    )
    parser.add_argument(
        "--total-tokens-per-peer-values",
        help=(
            "Comma-separated total token counts per peer; each value must "
            "be divisible by every chunk count."
        ),
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
