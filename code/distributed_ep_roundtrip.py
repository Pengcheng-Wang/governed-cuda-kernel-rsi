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

from bench_fused_moe import make_routing, quality_metrics


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


def validate_roundtrip_markers(
    inputs: list[torch.Tensor],
    dispatch_outputs: list[torch.Tensor],
    combine_outputs: list[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    tokens_per_peer: int,
) -> float:
    max_error = 0.0
    for input_tensor, dispatch_output, combine_output in zip(
        inputs,
        dispatch_outputs,
        combine_outputs,
    ):
        for destination in range(world_size):
            start = destination * tokens_per_peer
            end = start + tokens_per_peer
            input_tensor[start:end, 0].fill_(
                rank + destination / 128.0
            )
        dist.all_to_all_single(dispatch_output, input_tensor)
        dist.all_to_all_single(combine_output, dispatch_output)
        torch.cuda.synchronize()
        for destination in range(world_size):
            start = destination * tokens_per_peer
            expected = rank + destination / 128.0
            error = (
                combine_output[start, 0]
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

    torch.manual_seed(83)
    torch.cuda.manual_seed_all(83)

    if args.profile == "smoke":
        configurations = [(16, 2)]
        warmup, iterations = 1, 3
    else:
        configurations = [(1024, 4), (4096, 8)]
        warmup, iterations = 3, 10

    generator = torch.Generator(device=device)
    generator.manual_seed(5_000)
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
    for total_tokens_per_peer, chunks in configurations:
        if total_tokens_per_peer % chunks != 0:
            raise ValueError(
                f"{total_tokens_per_peer} total tokens per peer is not "
                f"divisible by {chunks} chunks"
            )
        tokens_per_peer = total_tokens_per_peer // chunks
        local_tokens = world_size * tokens_per_peer
        inputs: list[torch.Tensor] = []
        dispatch_outputs: list[torch.Tensor] = []
        combine_outputs: list[torch.Tensor] = []
        for chunk in range(chunks):
            chunk_generator = torch.Generator(device=device)
            chunk_generator.manual_seed(
                20_000 + rank * 100 + chunk
            )
            inputs.append(
                torch.randn(
                    (local_tokens, args.hidden),
                    device=device,
                    dtype=dtype,
                    generator=chunk_generator,
                )
            )
            dispatch_outputs.append(torch.empty_like(inputs[-1]))
            combine_outputs.append(torch.empty_like(inputs[-1]))

        marker_error = validate_roundtrip_markers(
            inputs,
            dispatch_outputs,
            combine_outputs,
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

        def dispatch_only() -> None:
            for input_tensor, dispatch_output in zip(
                inputs,
                dispatch_outputs,
            ):
                work = launch_alltoall(dispatch_output, input_tensor)
                work.wait()
            return None

        def roundtrip_communication_only() -> None:
            for input_tensor, dispatch_output, combine_output in zip(
                inputs,
                dispatch_outputs,
                combine_outputs,
            ):
                dispatch_work = launch_alltoall(
                    dispatch_output,
                    input_tensor,
                )
                dispatch_work.wait()
                combine_work = launch_alltoall(
                    combine_output,
                    dispatch_output,
                )
                combine_work.wait()
            return None

        def compute_only() -> list[torch.Tensor]:
            return [
                compute(dispatch_output)
                for dispatch_output in dispatch_outputs
            ]

        def sequential() -> list[torch.Tensor]:
            expert_outputs: list[torch.Tensor] = []
            for input_tensor, dispatch_output, combine_output in zip(
                inputs,
                dispatch_outputs,
                combine_outputs,
            ):
                dispatch_work = launch_alltoall(
                    dispatch_output,
                    input_tensor,
                )
                dispatch_work.wait()
                expert_output = compute(dispatch_output)
                expert_outputs.append(expert_output)
                combine_work = launch_alltoall(
                    combine_output,
                    expert_output,
                )
                combine_work.wait()
            return combine_outputs

        def pipelined() -> list[torch.Tensor]:
            expert_outputs: list[torch.Tensor] = []
            first_dispatch = launch_alltoall(
                dispatch_outputs[0],
                inputs[0],
            )
            first_dispatch.wait()

            if chunks == 1:
                expert_output = compute(dispatch_outputs[0])
                expert_outputs.append(expert_output)
                combine_work = launch_alltoall(
                    combine_outputs[0],
                    expert_output,
                )
                combine_work.wait()
                return combine_outputs

            next_dispatch = launch_alltoall(
                dispatch_outputs[1],
                inputs[1],
            )
            first_expert_output = compute(dispatch_outputs[0])
            expert_outputs.append(first_expert_output)
            previous_combine = launch_alltoall(
                combine_outputs[0],
                first_expert_output,
            )
            next_dispatch.wait()

            for chunk in range(1, chunks):
                following_dispatch = None
                if chunk + 1 < chunks:
                    following_dispatch = launch_alltoall(
                        dispatch_outputs[chunk + 1],
                        inputs[chunk + 1],
                    )

                expert_output = compute(dispatch_outputs[chunk])
                expert_outputs.append(expert_output)
                current_combine = launch_alltoall(
                    combine_outputs[chunk],
                    expert_output,
                )

                previous_combine.wait()
                if following_dispatch is not None:
                    following_dispatch.wait()
                if chunk + 1 == chunks:
                    current_combine.wait()
                previous_combine = current_combine

            return combine_outputs

        sequential_reference = [
            output.clone()
            for output in sequential()
        ]
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

        timings: dict[str, dict[str, Any]] = {}
        schedules: tuple[
            tuple[str, Callable[[], list[torch.Tensor] | None]],
            ...,
        ] = (
            ("dispatch_only", dispatch_only),
            (
                "roundtrip_communication_only",
                roundtrip_communication_only,
            ),
            ("compute_only", compute_only),
            ("sequential", sequential),
            ("pipelined", pipelined),
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

        roundtrip_communication_ms = timings[
            "roundtrip_communication_only"
        ]["critical_path_ms"]
        compute_ms = timings["compute_only"]["critical_path_ms"]
        sequential_ms = timings["sequential"]["critical_path_ms"]
        pipelined_ms = timings["pipelined"]["critical_path_ms"]
        communication_per_chunk_ms = (
            roundtrip_communication_ms / chunks
        )
        compute_per_chunk_ms = compute_ms / chunks
        ideal_pipeline_ms = (
            communication_per_chunk_ms
            + compute_per_chunk_ms
            + (chunks - 1)
            * max(
                communication_per_chunk_ms,
                compute_per_chunk_ms,
            )
        )
        hidden_ms = max(0.0, sequential_ms - pipelined_ms)
        hideable_ms = max(0.0, sequential_ms - ideal_pipeline_ms)

        records.append(
            {
                "total_tokens_per_peer": total_tokens_per_peer,
                "tokens_per_peer": tokens_per_peer,
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
                "timings": timings,
                "pipeline_speedup": sequential_ms / pipelined_ms,
                "saved_critical_path_ms": hidden_ms,
                "saved_critical_path_fraction": hidden_ms / sequential_ms,
                "ideal_pipeline_ms": ideal_pipeline_ms,
                "overlap_efficiency": (
                    hidden_ms / hideable_ms
                    if hideable_ms > 0.0
                    else 0.0
                ),
                "max_roundtrip_marker_error": max(
                    gather_scalar(marker_error, device)
                ),
                "pipeline_vs_sequential_max_abs_error": max(
                    gather_scalar(local_max_abs_error, device)
                ),
                "pipeline_vs_sequential_min_cosine": min(
                    gather_scalar(local_min_cosine, device)
                ),
                "warmup": warmup,
                "iterations": iterations,
            }
        )

        del (
            inputs,
            dispatch_outputs,
            combine_outputs,
            topk_ids,
            topk_weights,
        )
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_ep_roundtrip",
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
                "Each microbatch performs equal-split BF16 dispatch all-to-all, vLLM fused expert compute, and combine all-to-all back to the original rank.",
                "The pipeline overlaps combine of microbatch i-1 and dispatch of microbatch i+1 with expert compute for microbatch i.",
                "NCCL collectives share one process group and retain identical launch ordering on every rank.",
                "Routing is precomputed and excluded; token permutation, router computation, capacity factors, and dropped tokens are outside this isolated benchmark.",
                "Critical path is the slowest rank's median CPU wall time with CUDA synchronization before and after each schedule.",
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
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--routing",
        choices=["uniform", "hot80"],
        default="uniform",
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
