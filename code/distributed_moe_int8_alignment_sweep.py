from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from distributed_moe_end_to_end import (
    critical_path_summary,
    gather_scalar,
    output_quality,
    timed_schedule,
)
from distributed_moe_int8_roundtrip import (
    aligned_packet_size,
    dequantize_packet_triton,
    dequantize_scale_triton,
    quantize_packet_triton,
    quantize_scale_triton,
    return_packet_views,
)
from distributed_moe_packed_dispatch import timed_crossover


def distribute_evenly(
    total: int,
    destinations: list[int],
    *,
    start: int,
) -> list[int]:
    counts = [0 for _ in destinations]
    base, remainder = divmod(total, len(destinations))
    for index in range(len(destinations)):
        counts[index] = base
    for offset in range(remainder):
        counts[(start + offset) % len(destinations)] += 1
    return counts


def build_send_matrix(
    world_size: int,
    assignments_per_rank: int,
    route: str,
) -> tuple[list[list[int]], float]:
    if route == "balanced":
        hot_fraction = 1.0 / world_size
    elif route == "moderate_hot":
        hot_fraction = 0.20 if world_size >= 8 else 0.65
    elif route == "extreme_hot":
        hot_fraction = 0.95 if world_size >= 8 else 0.90
    else:
        raise ValueError(f"Unknown route: {route}")

    matrix: list[list[int]] = []
    for source in range(world_size):
        if route == "balanced":
            matrix.append(
                distribute_evenly(
                    assignments_per_rank,
                    list(range(world_size)),
                    start=source,
                )
            )
            continue

        hot_count = round(assignments_per_rank * hot_fraction)
        remaining = assignments_per_rank - hot_count
        non_hot = list(range(1, world_size))
        distributed = distribute_evenly(
            remaining,
            non_hot,
            start=source % len(non_hot),
        )
        counts = [0 for _ in range(world_size)]
        counts[0] = hot_count
        for destination, count in zip(non_hot, distributed):
            counts[destination] = count
        matrix.append(counts)
    return matrix, hot_fraction


def return_bf16(
    expert_output: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
) -> torch.Tensor:
    returned = torch.empty(
        (sum(send_counts), expert_output.shape[1]),
        device=expert_output.device,
        dtype=expert_output.dtype,
    )
    dist.all_to_all_single(
        returned,
        expert_output,
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts,
    )
    return returned


def return_two_collective(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    returned_quantized = torch.empty(
        (sum(send_counts), quantized.shape[1]),
        device=quantized.device,
        dtype=torch.int8,
    )
    returned_scales = torch.empty(
        (sum(send_counts), 1),
        device=quantized.device,
        dtype=dtype,
    )
    dist.all_to_all_single(
        returned_quantized,
        quantized,
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts,
    )
    dist.all_to_all_single(
        returned_scales,
        scales,
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts,
    )
    return dequantize_scale_triton(
        returned_quantized,
        returned_scales,
        dtype,
    )


def return_packet(
    packet: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    dtype: torch.dtype,
    width: int,
) -> torch.Tensor:
    returned_packet = torch.empty(
        (sum(send_counts), packet.shape[1]),
        device=packet.device,
        dtype=torch.uint8,
    )
    dist.all_to_all_single(
        returned_packet,
        packet,
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts,
    )
    return dequantize_packet_triton(
        returned_packet,
        dtype,
        width,
    )


def profile_settings(
    profile: str,
) -> tuple[list[int], list[int], list[int], list[str], int, int]:
    if profile == "smoke":
        return (
            [510, 512],
            [2, 16],
            [64],
            ["balanced", "moderate_hot"],
            1,
            2,
        )
    if profile == "crossover":
        return (
            [512, 1024, 4096],
            [2, 4, 8, 16, 32, 64],
            [1024],
            ["balanced", "moderate_hot", "extreme_hot"],
            1,
            2,
        )
    return (
        [510, 512, 1022, 1024, 4094, 4096],
        [2, 4, 8, 16, 32, 64],
        [512],
        ["balanced", "moderate_hot", "extreme_hot"],
        1,
        3,
    )


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    (
        default_widths,
        default_alignments,
        default_tokens,
        routes,
        warmup,
        iterations,
    ) = profile_settings(args.profile)
    widths = args.hidden_values or default_widths
    alignments = args.alignments or default_alignments
    token_values = args.token_values or default_tokens
    if any(alignment <= 0 or alignment % 2 for alignment in alignments):
        raise ValueError("Alignments must be positive even integers")
    if len(set(alignments)) != len(alignments):
        raise ValueError("Alignments must be unique")

    records: list[dict[str, Any]] = []
    for width in widths:
        for tokens_per_rank in token_values:
            assignments_per_rank = tokens_per_rank * args.top_k
            for route_index, route in enumerate(routes):
                send_matrix, requested_hot_fraction = build_send_matrix(
                    world_size,
                    assignments_per_rank,
                    route,
                )
                send_counts = send_matrix[rank]
                recv_counts = [
                    send_matrix[source][rank]
                    for source in range(world_size)
                ]
                receive_totals = [
                    sum(
                        send_matrix[source][destination]
                        for source in range(world_size)
                    )
                    for destination in range(world_size)
                ]
                total_assignments = sum(receive_totals)
                mean_receive = statistics.fmean(receive_totals)

                generator = torch.Generator(device=device)
                generator.manual_seed(
                    90_000
                    + rank * 10_003
                    + width * 17
                    + tokens_per_rank
                    + route_index * 1_000_003
                )
                expert_output = torch.randn(
                    (sum(recv_counts), width),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )

                quantized, scales = quantize_scale_triton(
                    expert_output
                )
                packets = {
                    alignment: quantize_packet_triton(
                        expert_output,
                        alignment=alignment,
                    )
                    for alignment in alignments
                }
                returned_bf16 = return_bf16(
                    expert_output,
                    send_counts,
                    recv_counts,
                )
                returned_two = return_two_collective(
                    quantized,
                    scales,
                    send_counts,
                    recv_counts,
                    dtype,
                )
                returned_packets = {
                    alignment: return_packet(
                        packet,
                        send_counts,
                        recv_counts,
                        dtype,
                        width,
                    )
                    for alignment, packet in packets.items()
                }

                packet_quality: dict[str, dict[str, Any]] = {}
                for alignment, packet in packets.items():
                    packet_quantized, packet_scales = (
                        return_packet_views(
                            packet,
                            width,
                            dtype,
                        )
                    )
                    local_payload_equal = (
                        torch.equal(packet_quantized, quantized)
                        and torch.equal(packet_scales, scales)
                    )
                    raw_size = width + 2
                    local_padding_zero = (
                        packet.shape[1] == raw_size
                        or packet[:, raw_size:]
                        .count_nonzero()
                        .item()
                        == 0
                    )
                    payload_equal = min(
                        gather_scalar(
                            float(local_payload_equal),
                            device,
                        )
                    )
                    padding_zero = min(
                        gather_scalar(
                            float(local_padding_zero),
                            device,
                        )
                    )
                    versus_two = output_quality(
                        returned_packets[alignment],
                        returned_two,
                    )
                    versus_bf16 = output_quality(
                        returned_packets[alignment],
                        returned_bf16,
                    )
                    max_vs_two_error = max(
                        gather_scalar(
                            versus_two["max_abs_error"],
                            device,
                        )
                    )
                    max_bf16_nrmse = max(
                        gather_scalar(
                            versus_bf16["normalized_rmse"],
                            device,
                        )
                    )
                    min_bf16_cosine = min(
                        gather_scalar(
                            versus_bf16["cosine_similarity"],
                            device,
                        )
                    )
                    if (
                        payload_equal != 1.0
                        or padding_zero != 1.0
                        or max_vs_two_error != 0.0
                        or max_bf16_nrmse > args.max_output_nrmse
                        or min_bf16_cosine < args.min_output_cosine
                    ):
                        raise RuntimeError(
                            f"Alignment quality failed for width={width}, "
                            f"route={route}, alignment={alignment}"
                        )
                    packet_quality[str(alignment)] = {
                        "payload_equal": payload_equal == 1.0,
                        "padding_zero": padding_zero == 1.0,
                        "max_vs_two_collective_error": (
                            max_vs_two_error
                        ),
                        "max_output_nrmse_vs_bf16": (
                            max_bf16_nrmse
                        ),
                        "min_output_cosine_vs_bf16": (
                            min_bf16_cosine
                        ),
                    }

                two_quality = output_quality(
                    returned_two,
                    returned_bf16,
                )

                def quantize_two() -> Any:
                    return quantize_scale_triton(expert_output)

                def dequantize_two() -> torch.Tensor:
                    return dequantize_scale_triton(
                        quantized,
                        scales,
                        dtype,
                    )

                def bf16_return() -> torch.Tensor:
                    return return_bf16(
                        expert_output,
                        send_counts,
                        recv_counts,
                    )

                def two_prebuilt() -> torch.Tensor:
                    return return_two_collective(
                        quantized,
                        scales,
                        send_counts,
                        recv_counts,
                        dtype,
                    )

                def two_with_quantization() -> torch.Tensor:
                    current_quantized, current_scales = (
                        quantize_scale_triton(expert_output)
                    )
                    return return_two_collective(
                        current_quantized,
                        current_scales,
                        send_counts,
                        recv_counts,
                        dtype,
                    )

                schedules: list[tuple[str, Callable[[], Any]]] = [
                    ("quantize_two", quantize_two),
                    ("dequantize_two", dequantize_two),
                    ("bf16_return", bf16_return),
                    ("two_prebuilt", two_prebuilt),
                    (
                        "two_with_quantization",
                        two_with_quantization,
                    ),
                ]

                for alignment in alignments:
                    packet = packets[alignment]

                    def packet_quantize(
                        current_alignment: int = alignment,
                    ) -> torch.Tensor:
                        return quantize_packet_triton(
                            expert_output,
                            alignment=current_alignment,
                        )

                    def packet_dequantize(
                        current_packet: torch.Tensor = packet,
                    ) -> torch.Tensor:
                        return dequantize_packet_triton(
                            current_packet,
                            dtype,
                            width,
                        )

                    def packet_prebuilt(
                        current_packet: torch.Tensor = packet,
                    ) -> torch.Tensor:
                        return return_packet(
                            current_packet,
                            send_counts,
                            recv_counts,
                            dtype,
                            width,
                        )

                    def packet_with_quantization(
                        current_alignment: int = alignment,
                    ) -> torch.Tensor:
                        current_packet = quantize_packet_triton(
                            expert_output,
                            alignment=current_alignment,
                        )
                        return return_packet(
                            current_packet,
                            send_counts,
                            recv_counts,
                            dtype,
                            width,
                        )

                    schedules.extend(
                        [
                            (
                                f"packet_{alignment}_quantize",
                                packet_quantize,
                            ),
                            (
                                f"packet_{alignment}_dequantize",
                                packet_dequantize,
                            ),
                            (
                                f"packet_{alignment}_prebuilt",
                                packet_prebuilt,
                            ),
                            (
                                f"packet_{alignment}_with_quantization",
                                packet_with_quantization,
                            ),
                        ]
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
                    for name, schedule in schedules
                }

                crossover = None
                if args.profile == "crossover":
                    crossover_schedules: list[
                        tuple[str, Callable[[], Any]]
                    ] = [
                        ("bf16", bf16_return),
                        (
                            "two_collective",
                            two_with_quantization,
                        ),
                    ]
                    for alignment in alignments:

                        def crossover_packet(
                            current_alignment: int = alignment,
                        ) -> torch.Tensor:
                            current_packet = (
                                quantize_packet_triton(
                                    expert_output,
                                    alignment=current_alignment,
                                )
                            )
                            return return_packet(
                                current_packet,
                                send_counts,
                                recv_counts,
                                dtype,
                                width,
                            )

                        crossover_schedules.append(
                            (
                                f"packet_{alignment}",
                                crossover_packet,
                            )
                        )
                    crossover = timed_crossover(
                        tuple(crossover_schedules),
                        warmup_cycles=len(crossover_schedules),
                        measured_cycles=args.crossover_cycles,
                        local_rank=local_rank,
                        device=device,
                    )

                two_ms = timings["two_with_quantization"][
                    "critical_path_ms"
                ]
                bf16_ms = timings["bf16_return"][
                    "critical_path_ms"
                ]
                alignment_results: dict[str, dict[str, float]] = {}
                for alignment in alignments:
                    packet_ms = timings[
                        f"packet_{alignment}_with_quantization"
                    ]["critical_path_ms"]
                    alignment_results[str(alignment)] = {
                        "speedup_vs_two_collective": (
                            two_ms / packet_ms
                        ),
                        "speedup_vs_bf16": bf16_ms / packet_ms,
                    }

                records.append(
                    {
                        "hidden": width,
                        "tokens_per_rank": tokens_per_rank,
                        "top_k": args.top_k,
                        "assignments_per_rank": (
                            assignments_per_rank
                        ),
                        "route": route,
                        "requested_hot_fraction": (
                            requested_hot_fraction
                        ),
                        "routing": {
                            "send_matrix": send_matrix,
                            "rank_receive_assignments": (
                                receive_totals
                            ),
                            "rank0_receive_fraction": (
                                receive_totals[0]
                                / total_assignments
                            ),
                            "max_to_mean_receive": (
                                max(receive_totals)
                                / mean_receive
                            ),
                        },
                        "payload": {
                            "bf16_bytes_per_assignment": width * 2,
                            "two_collective_bytes_per_assignment": (
                                width + 2
                            ),
                            "packet_sizes": {
                                str(alignment): (
                                    aligned_packet_size(
                                        width,
                                        alignment,
                                    )
                                )
                                for alignment in alignments
                            },
                            "byte_reduction_fractions": {
                                str(alignment): (
                                    1
                                    - aligned_packet_size(
                                        width,
                                        alignment,
                                    )
                                    / (width * 2)
                                )
                                for alignment in alignments
                            },
                        },
                        "quality": {
                            "two_collective_max_output_nrmse": max(
                                gather_scalar(
                                    two_quality[
                                        "normalized_rmse"
                                    ],
                                    device,
                                )
                            ),
                            "two_collective_min_output_cosine": min(
                                gather_scalar(
                                    two_quality[
                                        "cosine_similarity"
                                    ],
                                    device,
                                )
                            ),
                            "packets": packet_quality,
                        },
                        "timings": timings,
                        "alignment_speedups": alignment_results,
                        "crossover": crossover,
                        "warmup": warmup,
                        "iterations": iterations,
                    }
                )

                del (
                    expert_output,
                    quantized,
                    scales,
                    packets,
                    returned_bf16,
                    returned_two,
                    returned_packets,
                )
                torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_int8_alignment_sweep",
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
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "This transport-only benchmark isolates the expert-output return path without allocating router or expert weights.",
                "Balanced, moderate-hot, and extreme-hot deterministic split matrices preserve the variable-split return direction used by expert parallelism.",
                "The two-collective baseline sends INT8 values and one BF16 scale separately.",
                "Packet alignments round each uint8 row size upward while fused Triton writes INT8 values, the BF16 scale, and zero padding in one kernel.",
                "Widths H-2 create naturally aligned raw packets and are paired with standard transformer widths to control for payload size.",
                "Every packet must reproduce the two-collective quantized return bit-for-bit.",
            ],
            "alignments": alignments,
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
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--hidden-values",
        nargs="*",
        type=int,
    )
    parser.add_argument(
        "--alignments",
        nargs="*",
        type=int,
    )
    parser.add_argument(
        "--token-values",
        nargs="*",
        type=int,
    )
    parser.add_argument(
        "--crossover-cycles",
        type=int,
        default=16,
    )
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
