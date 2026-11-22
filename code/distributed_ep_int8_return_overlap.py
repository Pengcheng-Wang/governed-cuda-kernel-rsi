from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from bench_fused_moe import make_routing, quality_metrics
from distributed_ep_roundtrip import (
    critical_path_summary,
    gather_scalar,
    launch_alltoall,
    timed_schedule,
    validate_roundtrip_markers,
)
from distributed_moe_int8_roundtrip import (
    aligned_packet_size,
    dequantize_packet_triton,
    dequantize_scale_triton,
    quantize_packet_triton,
    quantize_scale_triton,
)
from distributed_moe_packed_dispatch import timed_crossover


ReturnState = dict[str, Any]


def profile_settings(
    profile: str,
) -> tuple[list[tuple[int, int]], int, int]:
    if profile == "smoke":
        return [(128, 2)], 1, 2
    if profile == "crossover":
        return [(1024, 4), (4096, 8)], 1, 2
    return [(1024, 4), (4096, 8)], 3, 8


def launch_bf16_return(
    expert_output: torch.Tensor,
    returned: torch.Tensor,
) -> ReturnState:
    return {
        "kind": "bf16",
        "works": [launch_alltoall(returned, expert_output)],
        "inputs": [expert_output],
        "returned": returned,
    }


def launch_two_collective_return(
    expert_output: torch.Tensor,
    returned_quantized: torch.Tensor,
    returned_scales: torch.Tensor,
) -> ReturnState:
    quantized, scales = quantize_scale_triton(expert_output)
    return {
        "kind": "two_collective",
        "works": [
            launch_alltoall(returned_quantized, quantized),
            launch_alltoall(returned_scales, scales),
        ],
        "inputs": [expert_output, quantized, scales],
        "returned_quantized": returned_quantized,
        "returned_scales": returned_scales,
        "dtype": expert_output.dtype,
    }


def launch_packet_return(
    expert_output: torch.Tensor,
    returned_packet: torch.Tensor,
    *,
    alignment: int,
) -> ReturnState:
    packet = quantize_packet_triton(
        expert_output,
        alignment=alignment,
    )
    return {
        "kind": "packet",
        "works": [launch_alltoall(returned_packet, packet)],
        "inputs": [expert_output, packet],
        "returned_packet": returned_packet,
        "dtype": expert_output.dtype,
        "width": expert_output.shape[1],
    }


def finish_return(state: ReturnState) -> torch.Tensor:
    for work in state["works"]:
        work.wait()
    kind = state["kind"]
    if kind == "bf16":
        return state["returned"]
    if kind == "two_collective":
        return dequantize_scale_triton(
            state["returned_quantized"],
            state["returned_scales"],
            state["dtype"],
        )
    if kind == "packet":
        return dequantize_packet_triton(
            state["returned_packet"],
            state["dtype"],
            state["width"],
        )
    raise ValueError(f"Unknown return kind: {kind}")


def quality_across_chunks(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    local_max_abs_error = 0.0
    local_max_nrmse = 0.0
    local_min_cosine = 1.0
    for actual_tensor, expected_tensor in zip(actual, expected):
        quality = quality_metrics(actual_tensor, expected_tensor)
        local_max_abs_error = max(
            local_max_abs_error,
            quality["max_abs_error"],
        )
        local_max_nrmse = max(
            local_max_nrmse,
            quality["normalized_rmse"],
        )
        local_min_cosine = min(
            local_min_cosine,
            quality["cosine_similarity"],
        )
    return {
        "max_abs_error": max(
            gather_scalar(local_max_abs_error, device)
        ),
        "max_normalized_rmse": max(
            gather_scalar(local_max_nrmse, device)
        ),
        "min_cosine_similarity": min(
            gather_scalar(local_min_cosine, device)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    configurations, warmup, iterations = profile_settings(
        args.profile
    )
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

    packet_size = aligned_packet_size(
        args.hidden,
        args.packet_alignment,
    )
    records: list[dict[str, Any]] = []
    for total_tokens_per_peer, chunks in configurations:
        if total_tokens_per_peer % chunks:
            raise ValueError(
                f"{total_tokens_per_peer} is not divisible by "
                f"{chunks}"
            )
        tokens_per_peer = total_tokens_per_peer // chunks
        local_tokens = world_size * tokens_per_peer
        inputs: list[torch.Tensor] = []
        dispatch_outputs: list[torch.Tensor] = []
        bf16_return_buffers: list[torch.Tensor] = []
        int8_return_buffers: list[torch.Tensor] = []
        scale_return_buffers: list[torch.Tensor] = []
        packet_return_buffers: list[torch.Tensor] = []
        for chunk in range(chunks):
            chunk_generator = torch.Generator(device=device)
            chunk_generator.manual_seed(
                20_000 + rank * 100 + chunk
            )
            input_tensor = torch.randn(
                (local_tokens, args.hidden),
                device=device,
                dtype=dtype,
                generator=chunk_generator,
            )
            inputs.append(input_tensor)
            dispatch_outputs.append(torch.empty_like(input_tensor))
            bf16_return_buffers.append(torch.empty_like(input_tensor))
            int8_return_buffers.append(
                torch.empty(
                    input_tensor.shape,
                    device=device,
                    dtype=torch.int8,
                )
            )
            scale_return_buffers.append(
                torch.empty(
                    (local_tokens, 1),
                    device=device,
                    dtype=dtype,
                )
            )
            packet_return_buffers.append(
                torch.empty(
                    (local_tokens, packet_size),
                    device=device,
                    dtype=torch.uint8,
                )
            )

        marker_error = validate_roundtrip_markers(
            inputs,
            dispatch_outputs,
            bf16_return_buffers,
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

        def launch_return(
            kind: str,
            chunk: int,
            expert_output: torch.Tensor,
        ) -> ReturnState:
            if kind == "bf16":
                return launch_bf16_return(
                    expert_output,
                    bf16_return_buffers[chunk],
                )
            if kind == "two_collective":
                return launch_two_collective_return(
                    expert_output,
                    int8_return_buffers[chunk],
                    scale_return_buffers[chunk],
                )
            if kind == "packet":
                return launch_packet_return(
                    expert_output,
                    packet_return_buffers[chunk],
                    alignment=args.packet_alignment,
                )
            raise ValueError(f"Unknown return kind: {kind}")

        def dispatch_only() -> None:
            for input_tensor, dispatch_output in zip(
                inputs,
                dispatch_outputs,
            ):
                work = launch_alltoall(
                    dispatch_output,
                    input_tensor,
                )
                work.wait()
            return None

        def compute_only() -> list[torch.Tensor]:
            return [
                compute(dispatch_output)
                for dispatch_output in dispatch_outputs
            ]

        prepared_expert_outputs = compute_only()

        def return_only(kind: str) -> list[torch.Tensor]:
            outputs: list[torch.Tensor] = []
            for chunk, expert_output in enumerate(
                prepared_expert_outputs
            ):
                outputs.append(
                    finish_return(
                        launch_return(
                            kind,
                            chunk,
                            expert_output,
                        )
                    )
                )
            return outputs

        def sequential(kind: str) -> list[torch.Tensor]:
            outputs: list[torch.Tensor] = []
            for chunk, (input_tensor, dispatch_output) in enumerate(
                zip(inputs, dispatch_outputs)
            ):
                dispatch_work = launch_alltoall(
                    dispatch_output,
                    input_tensor,
                )
                dispatch_work.wait()
                expert_output = compute(dispatch_output)
                outputs.append(
                    finish_return(
                        launch_return(
                            kind,
                            chunk,
                            expert_output,
                        )
                    )
                )
            return outputs

        def pipelined(kind: str) -> list[torch.Tensor]:
            outputs: list[torch.Tensor | None] = [
                None for _ in range(chunks)
            ]
            first_dispatch = launch_alltoall(
                dispatch_outputs[0],
                inputs[0],
            )
            first_dispatch.wait()

            if chunks == 1:
                expert_output = compute(dispatch_outputs[0])
                outputs[0] = finish_return(
                    launch_return(kind, 0, expert_output)
                )
                return [outputs[0]]

            next_dispatch = launch_alltoall(
                dispatch_outputs[1],
                inputs[1],
            )
            first_expert_output = compute(dispatch_outputs[0])
            previous_return = launch_return(
                kind,
                0,
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
                current_return = launch_return(
                    kind,
                    chunk,
                    expert_output,
                )
                outputs[chunk - 1] = finish_return(
                    previous_return
                )
                if following_dispatch is not None:
                    following_dispatch.wait()
                if chunk + 1 == chunks:
                    outputs[chunk] = finish_return(
                        current_return
                    )
                previous_return = current_return

            return [
                output
                for output in outputs
                if output is not None
            ]

        schedules: dict[str, Callable[[], Any]] = {
            "dispatch_only": dispatch_only,
            "compute_only": compute_only,
            "bf16_return_only": lambda: return_only("bf16"),
            "two_return_only": lambda: return_only(
                "two_collective"
            ),
            "packet_return_only": lambda: return_only("packet"),
            "bf16_sequential": lambda: sequential("bf16"),
            "bf16_pipelined": lambda: pipelined("bf16"),
            "two_sequential": lambda: sequential(
                "two_collective"
            ),
            "two_pipelined": lambda: pipelined(
                "two_collective"
            ),
            "packet_sequential": lambda: sequential("packet"),
            "packet_pipelined": lambda: pipelined("packet"),
        }

        bf16_reference = [
            output.clone()
            for output in schedules["bf16_sequential"]()
        ]
        quality_outputs = {
            name: schedule()
            for name, schedule in schedules.items()
            if name.endswith("_pipelined")
            or name in ("two_sequential", "packet_sequential")
        }
        torch.cuda.synchronize()
        quality = {
            name: quality_across_chunks(
                output,
                bf16_reference,
                device,
            )
            for name, output in quality_outputs.items()
        }
        exact_quality = {
            "bf16_pipeline_vs_sequential": (
                quality_across_chunks(
                    quality_outputs["bf16_pipelined"],
                    bf16_reference,
                    device,
                )
            ),
            "two_pipeline_vs_sequential": (
                quality_across_chunks(
                    quality_outputs["two_pipelined"],
                    quality_outputs["two_sequential"],
                    device,
                )
            ),
            "packet_pipeline_vs_sequential": (
                quality_across_chunks(
                    quality_outputs["packet_pipelined"],
                    quality_outputs["packet_sequential"],
                    device,
                )
            ),
            "packet_vs_two_sequential": quality_across_chunks(
                quality_outputs["packet_sequential"],
                quality_outputs["two_sequential"],
                device,
            ),
            "packet_vs_two_pipelined": quality_across_chunks(
                quality_outputs["packet_pipelined"],
                quality_outputs["two_pipelined"],
                device,
            ),
        }
        if any(
            record["max_abs_error"] != 0.0
            for record in exact_quality.values()
        ):
            raise RuntimeError(
                f"Compressed return paths disagreed: {exact_quality}"
            )
        for name in (
            "two_sequential",
            "two_pipelined",
            "packet_sequential",
            "packet_pipelined",
        ):
            if (
                quality[name]["max_normalized_rmse"]
                > args.max_output_nrmse
                or quality[name]["min_cosine_similarity"]
                < args.min_output_cosine
            ):
                raise RuntimeError(
                    f"Quality failed for {name}: {quality[name]}"
                )

        timings = {
            name: critical_path_summary(
                timed_schedule(
                    schedule,
                    warmup=warmup,
                    iterations=iterations,
                    local_rank=local_rank,
                ),
                device,
            )
            for name, schedule in schedules.items()
        }

        crossover = None
        if args.profile == "crossover":
            crossover = timed_crossover(
                (
                    ("bf16_sequential", schedules["bf16_sequential"]),
                    ("bf16_pipelined", schedules["bf16_pipelined"]),
                    ("two_sequential", schedules["two_sequential"]),
                    ("two_pipelined", schedules["two_pipelined"]),
                    (
                        "packet_sequential",
                        schedules["packet_sequential"],
                    ),
                    (
                        "packet_pipelined",
                        schedules["packet_pipelined"],
                    ),
                ),
                warmup_cycles=6,
                measured_cycles=args.crossover_cycles,
                local_rank=local_rank,
                device=device,
            )

        def timing_speedup(
            baseline: str,
            candidate: str,
        ) -> float:
            return (
                timings[baseline]["critical_path_ms"]
                / timings[candidate]["critical_path_ms"]
            )

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
                "payload": {
                    "bf16_return_bytes_per_token": args.hidden * 2,
                    "two_collective_return_bytes_per_token": (
                        args.hidden + 2
                    ),
                    "packet_return_bytes_per_token": packet_size,
                    "packet_alignment": args.packet_alignment,
                    "packet_byte_reduction_fraction": (
                        1 - packet_size / (args.hidden * 2)
                    ),
                },
                "timings": timings,
                "speedups": {
                    "bf16_pipeline": timing_speedup(
                        "bf16_sequential",
                        "bf16_pipelined",
                    ),
                    "two_pipeline": timing_speedup(
                        "two_sequential",
                        "two_pipelined",
                    ),
                    "packet_pipeline": timing_speedup(
                        "packet_sequential",
                        "packet_pipelined",
                    ),
                    "two_sequential_vs_bf16": timing_speedup(
                        "bf16_sequential",
                        "two_sequential",
                    ),
                    "packet_sequential_vs_bf16": timing_speedup(
                        "bf16_sequential",
                        "packet_sequential",
                    ),
                    "packet_sequential_vs_two": timing_speedup(
                        "two_sequential",
                        "packet_sequential",
                    ),
                    "two_pipelined_vs_bf16": timing_speedup(
                        "bf16_pipelined",
                        "two_pipelined",
                    ),
                    "packet_pipelined_vs_bf16": timing_speedup(
                        "bf16_pipelined",
                        "packet_pipelined",
                    ),
                    "packet_pipelined_vs_two": timing_speedup(
                        "two_pipelined",
                        "packet_pipelined",
                    ),
                },
                "quality_vs_bf16": quality,
                "exact_quality": exact_quality,
                "max_roundtrip_marker_error": max(
                    gather_scalar(marker_error, device)
                ),
                "crossover": crossover,
                "warmup": warmup,
                "iterations": iterations,
            }
        )

        del (
            inputs,
            dispatch_outputs,
            bf16_return_buffers,
            int8_return_buffers,
            scale_return_buffers,
            packet_return_buffers,
            topk_ids,
            topk_weights,
            bf16_reference,
            quality_outputs,
            prepared_expert_outputs,
        )
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_ep_int8_return_overlap",
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
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "Each fixed-work microbatch uses equal-split BF16 dispatch, vLLM fused experts, and one of three return paths.",
                "Return paths are BF16, INT8 values plus BF16 scales in two collectives, and one 16-byte-aligned uint8 packet collective.",
                "The pipeline overlaps return of chunk i-1 and dispatch of chunk i+1 with expert compute and return quantization for chunk i.",
                "Return dequantization occurs only after the asynchronous collective completes.",
                "Routing is precomputed; token permutation, router computation, capacity, and variable split counts are outside this isolated benchmark.",
                "Packet and two-collective quantized outputs must match bit-for-bit.",
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
        choices=["smoke", "full", "crossover"],
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
    parser.add_argument("--packet-alignment", type=int, default=16)
    parser.add_argument("--crossover-cycles", type=int, default=18)
    parser.add_argument(
        "--max-output-nrmse",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--min-output-cosine",
        type=float,
        default=0.9998,
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
