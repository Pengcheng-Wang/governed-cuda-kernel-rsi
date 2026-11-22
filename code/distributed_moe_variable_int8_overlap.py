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

from distributed_ep_int8_return_overlap import (
    quality_across_chunks,
)
from distributed_ep_roundtrip import (
    critical_path_summary,
    gather_scalar,
    timed_schedule,
)
from distributed_moe_end_to_end import (
    TensorDict,
    local_experts,
    route_and_pack,
    routing_bias,
)
from distributed_moe_int8_roundtrip import (
    aligned_packet_size,
    combine_returned,
    dequantize_packet_triton,
    dequantize_scale_triton,
    quantize_packet_triton,
    quantize_scale_triton,
)
from distributed_moe_packed_dispatch import (
    build_metadata_packet,
    timed_crossover,
)


State = dict[str, Any]


def profile_settings(
    profile: str,
) -> tuple[int, int, tuple[tuple[str, float], ...], int, int]:
    if profile == "smoke":
        return (
            256,
            2,
            (("balanced", 0.0), ("global_bias1", 1.0)),
            1,
            2,
        )
    return (
        2048,
        4,
        (
            ("balanced", 0.0),
            ("global_bias1", 1.0),
            ("global_bias4", 4.0),
        ),
        2 if profile == "full" else 1,
        6 if profile == "full" else 2,
    )


def prepare_dispatch(
    route: TensorDict,
) -> dict[str, Any]:
    send_hidden = route["send_hidden"]
    send_counts_tensor = route["send_counts_tensor"]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts_tensor, torch.Tensor)
    assert isinstance(send_counts, list)
    metadata = build_metadata_packet(route)
    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    recv_counts = recv_counts_tensor.cpu().tolist()
    recv_total = sum(recv_counts)
    return {
        "route": route,
        "metadata": metadata,
        "recv_counts": recv_counts,
        "recv_hidden": torch.empty(
            (recv_total, send_hidden.shape[1]),
            device=send_hidden.device,
            dtype=send_hidden.dtype,
        ),
        "recv_metadata": torch.empty(
            (recv_total, 2),
            device=send_hidden.device,
            dtype=send_hidden.dtype,
        ),
    }


def launch_dispatch(prepared: dict[str, Any]) -> State:
    route = prepared["route"]
    send_hidden = route["send_hidden"]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts, list)
    return {
        "kind": "dispatch",
        "prepared": prepared,
        "works": [
            dist.all_to_all_single(
                prepared["recv_hidden"],
                send_hidden,
                output_split_sizes=prepared["recv_counts"],
                input_split_sizes=send_counts,
                async_op=True,
            ),
            dist.all_to_all_single(
                prepared["recv_metadata"],
                prepared["metadata"],
                output_split_sizes=prepared["recv_counts"],
                input_split_sizes=send_counts,
                async_op=True,
            ),
        ],
    }


def finish_dispatch(state: State) -> TensorDict:
    for work in state["works"]:
        work.wait()
    prepared = state["prepared"]
    recv_metadata = prepared["recv_metadata"]
    return {
        "recv_hidden": prepared["recv_hidden"],
        "recv_local_ids": (
            recv_metadata[:, :1].to(torch.int32).contiguous()
        ),
        "recv_weights": recv_metadata[:, 1:].contiguous(),
        "recv_counts": prepared["recv_counts"],
    }


def launch_return(
    kind: str,
    expert_output: torch.Tensor,
    prepared: dict[str, Any],
    buffers: dict[str, torch.Tensor],
    *,
    packet_alignment: int,
    num_tokens: int,
) -> State:
    route = prepared["route"]
    send_counts = route["send_counts"]
    recv_counts = prepared["recv_counts"]
    assert isinstance(send_counts, list)
    if kind == "bf16":
        return {
            "kind": kind,
            "route": route,
            "num_tokens": num_tokens,
            "inputs": [expert_output],
            "works": [
                dist.all_to_all_single(
                    buffers["bf16"],
                    expert_output,
                    output_split_sizes=send_counts,
                    input_split_sizes=recv_counts,
                    async_op=True,
                )
            ],
            "returned": buffers["bf16"],
        }
    if kind == "two_collective":
        quantized, scales = quantize_scale_triton(expert_output)
        return {
            "kind": kind,
            "route": route,
            "num_tokens": num_tokens,
            "inputs": [expert_output, quantized, scales],
            "works": [
                dist.all_to_all_single(
                    buffers["int8"],
                    quantized,
                    output_split_sizes=send_counts,
                    input_split_sizes=recv_counts,
                    async_op=True,
                ),
                dist.all_to_all_single(
                    buffers["scale"],
                    scales,
                    output_split_sizes=send_counts,
                    input_split_sizes=recv_counts,
                    async_op=True,
                ),
            ],
            "returned_quantized": buffers["int8"],
            "returned_scales": buffers["scale"],
            "dtype": expert_output.dtype,
        }
    if kind == "packet":
        packet = quantize_packet_triton(
            expert_output,
            alignment=packet_alignment,
        )
        return {
            "kind": kind,
            "route": route,
            "num_tokens": num_tokens,
            "inputs": [expert_output, packet],
            "works": [
                dist.all_to_all_single(
                    buffers["packet"],
                    packet,
                    output_split_sizes=send_counts,
                    input_split_sizes=recv_counts,
                    async_op=True,
                )
            ],
            "returned_packet": buffers["packet"],
            "dtype": expert_output.dtype,
            "width": expert_output.shape[1],
        }
    raise ValueError(f"Unknown return kind: {kind}")


def finish_return(state: State) -> torch.Tensor:
    for work in state["works"]:
        work.wait()
    kind = state["kind"]
    if kind == "bf16":
        returned = state["returned"]
    elif kind == "two_collective":
        returned = dequantize_scale_triton(
            state["returned_quantized"],
            state["returned_scales"],
            state["dtype"],
        )
    elif kind == "packet":
        returned = dequantize_packet_triton(
            state["returned_packet"],
            state["dtype"],
            state["width"],
        )
    else:
        raise ValueError(f"Unknown return kind: {kind}")
    return combine_returned(
        returned,
        state["route"],
        num_tokens=state["num_tokens"],
    )


def aggregate_routing(
    routes: list[TensorDict],
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    matrices: list[list[list[int]]] = []
    receive_vectors: list[list[int]] = []
    for route in routes:
        send_counts = route["send_counts"]
        assert isinstance(send_counts, list)
        send_counts_tensor = torch.tensor(
            send_counts,
            device=device,
            dtype=torch.int64,
        )
        gathered = [
            torch.empty_like(send_counts_tensor)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered, send_counts_tensor)
        matrix = [tensor.cpu().tolist() for tensor in gathered]
        matrices.append(matrix)
        receive_vectors.append(
            [
                sum(
                    matrix[source][destination]
                    for source in range(world_size)
                )
                for destination in range(world_size)
            ]
        )
    aggregate = [
        sum(vector[rank] for vector in receive_vectors)
        for rank in range(world_size)
    ]
    aggregate_mean = statistics.fmean(aggregate)
    step_imbalances = [
        max(vector) / statistics.fmean(vector)
        for vector in receive_vectors
    ]
    return {
        "send_matrices": matrices,
        "receive_assignments_per_chunk": receive_vectors,
        "aggregate_receive_assignments": aggregate,
        "aggregate_max_to_mean": max(aggregate) / aggregate_mean,
        "max_step_max_to_mean": max(step_imbalances),
        "median_step_max_to_mean": statistics.median(
            step_imbalances
        ),
        "rank0_fraction_per_chunk": [
            vector[0] / sum(vector)
            for vector in receive_vectors
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    (
        total_tokens_per_rank,
        chunks,
        routes_to_test,
        warmup,
        iterations,
    ) = profile_settings(args.profile)
    if total_tokens_per_rank % chunks:
        raise ValueError("Total tokens must divide chunks")
    tokens_per_chunk = total_tokens_per_rank // chunks
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
    packet_size = aligned_packet_size(
        args.hidden,
        args.packet_alignment,
    )

    records: list[dict[str, Any]] = []
    for route_index, (route_name, hot_bias) in enumerate(
        routes_to_test
    ):
        bias = routing_bias(
            "global_biased",
            rank=rank,
            num_experts=num_experts,
            experts_per_rank=args.experts_per_rank,
            hot_bias=hot_bias,
            device=device,
        )
        hidden_chunks: list[torch.Tensor] = []
        routes: list[TensorDict] = []
        prepared_dispatches: list[dict[str, Any]] = []
        return_buffers: list[dict[str, torch.Tensor]] = []
        reference_outputs: list[torch.Tensor] = []
        for chunk in range(chunks):
            input_generator = torch.Generator(device=device)
            input_generator.manual_seed(
                50_000
                + rank * 10_003
                + route_index * 1_000_003
                + chunk
            )
            hidden = torch.randn(
                (tokens_per_chunk, args.hidden),
                device=device,
                dtype=dtype,
                generator=input_generator,
            )
            route = route_and_pack(
                hidden,
                gate_weight,
                bias,
                tie_breaker,
                top_k=args.top_k,
                experts_per_rank=args.experts_per_rank,
                world_size=world_size,
            )
            prepared = prepare_dispatch(route)
            send_hidden = route["send_hidden"]
            assert isinstance(send_hidden, torch.Tensor)
            hidden_chunks.append(hidden)
            routes.append(route)
            prepared_dispatches.append(prepared)
            return_buffers.append(
                {
                    "bf16": torch.empty_like(send_hidden),
                    "int8": torch.empty(
                        send_hidden.shape,
                        device=device,
                        dtype=torch.int8,
                    ),
                    "scale": torch.empty(
                        (send_hidden.shape[0], 1),
                        device=device,
                        dtype=dtype,
                    ),
                    "packet": torch.empty(
                        (send_hidden.shape[0], packet_size),
                        device=device,
                        dtype=torch.uint8,
                    ),
                }
            )
            topk_ids = route["topk_ids"]
            topk_weights = route["topk_weights"]
            assert isinstance(topk_ids, torch.Tensor)
            assert isinstance(topk_weights, torch.Tensor)
            reference_outputs.append(
                fused_experts(
                    hidden,
                    global_w1,
                    global_w2,
                    topk_weights,
                    topk_ids,
                )
            )

        routing = aggregate_routing(routes, world_size, device)

        def compute(dispatched: TensorDict) -> torch.Tensor:
            return local_experts(
                dispatched,
                local_w1,
                local_w2,
            )

        def sequential(kind: str) -> list[torch.Tensor]:
            outputs: list[torch.Tensor] = []
            for chunk, prepared in enumerate(
                prepared_dispatches
            ):
                dispatched = finish_dispatch(
                    launch_dispatch(prepared)
                )
                expert_output = compute(dispatched)
                outputs.append(
                    finish_return(
                        launch_return(
                            kind,
                            expert_output,
                            prepared,
                            return_buffers[chunk],
                            packet_alignment=(
                                args.packet_alignment
                            ),
                            num_tokens=tokens_per_chunk,
                        )
                    )
                )
            return outputs

        def pipelined(kind: str) -> list[torch.Tensor]:
            outputs: list[torch.Tensor | None] = [
                None for _ in range(chunks)
            ]
            first_dispatched = finish_dispatch(
                launch_dispatch(prepared_dispatches[0])
            )
            if chunks == 1:
                expert_output = compute(first_dispatched)
                outputs[0] = finish_return(
                    launch_return(
                        kind,
                        expert_output,
                        prepared_dispatches[0],
                        return_buffers[0],
                        packet_alignment=args.packet_alignment,
                        num_tokens=tokens_per_chunk,
                    )
                )
                return [outputs[0]]

            next_dispatch = launch_dispatch(
                prepared_dispatches[1]
            )
            first_expert_output = compute(first_dispatched)
            previous_return = launch_return(
                kind,
                first_expert_output,
                prepared_dispatches[0],
                return_buffers[0],
                packet_alignment=args.packet_alignment,
                num_tokens=tokens_per_chunk,
            )
            current_dispatched = finish_dispatch(next_dispatch)

            for chunk in range(1, chunks):
                following_dispatch = None
                if chunk + 1 < chunks:
                    following_dispatch = launch_dispatch(
                        prepared_dispatches[chunk + 1]
                    )
                expert_output = compute(current_dispatched)
                current_return = launch_return(
                    kind,
                    expert_output,
                    prepared_dispatches[chunk],
                    return_buffers[chunk],
                    packet_alignment=args.packet_alignment,
                    num_tokens=tokens_per_chunk,
                )
                outputs[chunk - 1] = finish_return(
                    previous_return
                )
                if following_dispatch is not None:
                    current_dispatched = finish_dispatch(
                        following_dispatch
                    )
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

        schedules: dict[str, Callable[[], list[torch.Tensor]]] = {
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

        bf16_sequential = schedules["bf16_sequential"]()
        quality_outputs = {
            name: schedule()
            for name, schedule in schedules.items()
            if name != "bf16_sequential"
        }
        torch.cuda.synchronize()
        quality_vs_reference = {
            "bf16_sequential": quality_across_chunks(
                bf16_sequential,
                reference_outputs,
                device,
            ),
            **{
                name: quality_across_chunks(
                    output,
                    reference_outputs,
                    device,
                )
                for name, output in quality_outputs.items()
            },
        }
        exact_quality = {
            "bf16_pipeline_vs_sequential": (
                quality_across_chunks(
                    quality_outputs["bf16_pipelined"],
                    bf16_sequential,
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
        if quality_vs_reference["bf16_sequential"][
            "max_abs_error"
        ] != 0.0:
            raise RuntimeError(
                "BF16 variable pipeline disagrees with global reference"
            )
        if any(
            record["max_abs_error"] != 0.0
            for record in exact_quality.values()
        ):
            raise RuntimeError(
                f"Variable compressed paths disagreed: "
                f"{exact_quality}"
            )
        for name in (
            "two_sequential",
            "two_pipelined",
            "packet_sequential",
            "packet_pipelined",
        ):
            record = quality_vs_reference[name]
            if (
                record["max_normalized_rmse"]
                > args.max_output_nrmse
                or record["min_cosine_similarity"]
                < args.min_output_cosine
            ):
                raise RuntimeError(
                    f"Variable quality failed for {name}: {record}"
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
                tuple(schedules.items()),
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
                "total_tokens_per_rank": total_tokens_per_rank,
                "tokens_per_chunk": tokens_per_chunk,
                "chunks": chunks,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "experts_per_rank": args.experts_per_rank,
                "num_experts": num_experts,
                "top_k": args.top_k,
                "route": route_name,
                "hot_bias": hot_bias,
                "routing": routing,
                "payload": {
                    "bf16_return_bytes_per_assignment": (
                        args.hidden * 2
                    ),
                    "two_collective_return_bytes_per_assignment": (
                        args.hidden + 2
                    ),
                    "packet_return_bytes_per_assignment": (
                        packet_size
                    ),
                    "packet_alignment": args.packet_alignment,
                    "packet_byte_reduction_fraction": (
                        1 - packet_size / (args.hidden * 2)
                    ),
                    "count_exchange_in_timed_region": False,
                    "router_pack_in_timed_region": False,
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
                    "packet_sequential_vs_bf16": timing_speedup(
                        "bf16_sequential",
                        "packet_sequential",
                    ),
                    "packet_sequential_vs_two": timing_speedup(
                        "two_sequential",
                        "packet_sequential",
                    ),
                    "packet_pipelined_vs_bf16": timing_speedup(
                        "bf16_pipelined",
                        "packet_pipelined",
                    ),
                    "packet_pipelined_vs_two": timing_speedup(
                        "two_pipelined",
                        "packet_pipelined",
                    ),
                    "packet_pipelined_vs_bf16_sequential": (
                        timing_speedup(
                            "bf16_sequential",
                            "packet_pipelined",
                        )
                    ),
                },
                "quality_vs_reference": quality_vs_reference,
                "exact_quality": exact_quality,
                "crossover": crossover,
                "warmup": warmup,
                "iterations": iterations,
            }
        )

        del (
            hidden_chunks,
            routes,
            prepared_dispatches,
            return_buffers,
            reference_outputs,
            bf16_sequential,
            quality_outputs,
        )
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_variable_int8_overlap",
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
                "Real fused-router outputs determine variable send and receive counts for every microbatch.",
                "Router projection, top-k, token permutation, count exchange, and CPU count extraction are prepared outside the timed region.",
                "The timed path includes metadata-packed BF16 hidden/ID/weight dispatch, local fused experts, BF16 or compressed return, dequantization, and weighted inverse permutation.",
                "The pipeline overlaps return of chunk i-1 and dispatch of chunk i+1 with expert compute and return quantization for chunk i.",
                "BF16 distributed output must match the global-expert reference exactly; packet and two-collective INT8 must match bit-for-bit.",
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
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
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
