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

from distributed_moe_end_to_end import (
    TensorDict,
    critical_path_summary,
    dispatch_assignments,
    gather_scalar,
    local_experts,
    output_quality,
    return_and_combine,
    route_and_pack,
    routing_bias,
    timed_schedule,
)


def build_packed_packet(route: TensorDict) -> torch.Tensor:
    send_hidden = route["send_hidden"]
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_local_ids, torch.Tensor)
    assert isinstance(send_weights, torch.Tensor)
    return torch.cat(
        (
            send_hidden,
            send_local_ids.to(send_hidden.dtype),
            send_weights,
        ),
        dim=1,
    ).contiguous()


def build_metadata_packet(route: TensorDict) -> torch.Tensor:
    send_hidden = route["send_hidden"]
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_local_ids, torch.Tensor)
    assert isinstance(send_weights, torch.Tensor)
    return torch.cat(
        (
            send_local_ids.to(send_hidden.dtype),
            send_weights,
        ),
        dim=1,
    ).contiguous()


def dispatch_assignments_packed(
    route: TensorDict,
    *,
    packet: torch.Tensor | None = None,
) -> TensorDict:
    send_hidden = route["send_hidden"]
    send_counts_tensor = route["send_counts_tensor"]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts_tensor, torch.Tensor)
    assert isinstance(send_counts, list)
    if packet is None:
        packet = build_packed_packet(route)

    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    recv_counts = recv_counts_tensor.cpu().tolist()
    recv_total = sum(recv_counts)
    recv_packet = torch.empty(
        (recv_total, packet.shape[1]),
        device=packet.device,
        dtype=packet.dtype,
    )
    dist.all_to_all_single(
        recv_packet,
        packet,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    hidden = send_hidden.shape[1]
    return {
        "recv_hidden": recv_packet[:, :hidden].contiguous(),
        "recv_local_ids": (
            recv_packet[:, hidden : hidden + 1]
            .to(torch.int32)
            .contiguous()
        ),
        "recv_weights": recv_packet[:, hidden + 1 :].contiguous(),
        "recv_counts": recv_counts,
        "recv_packet": recv_packet,
    }


def dispatch_assignments_metadata_packed(
    route: TensorDict,
    *,
    metadata_packet: torch.Tensor | None = None,
) -> TensorDict:
    send_hidden = route["send_hidden"]
    send_counts_tensor = route["send_counts_tensor"]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts_tensor, torch.Tensor)
    assert isinstance(send_counts, list)
    if metadata_packet is None:
        metadata_packet = build_metadata_packet(route)

    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    recv_counts = recv_counts_tensor.cpu().tolist()
    recv_total = sum(recv_counts)
    recv_hidden = torch.empty(
        (recv_total, send_hidden.shape[1]),
        device=send_hidden.device,
        dtype=send_hidden.dtype,
    )
    recv_metadata = torch.empty(
        (recv_total, 2),
        device=send_hidden.device,
        dtype=send_hidden.dtype,
    )
    dist.all_to_all_single(
        recv_hidden,
        send_hidden,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    dist.all_to_all_single(
        recv_metadata,
        metadata_packet,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    return {
        "recv_hidden": recv_hidden,
        "recv_local_ids": (
            recv_metadata[:, :1].to(torch.int32).contiguous()
        ),
        "recv_weights": recv_metadata[:, 1:].contiguous(),
        "recv_counts": recv_counts,
        "recv_metadata": recv_metadata,
    }


def timed_crossover(
    schedules: tuple[tuple[str, Callable[[], Any]], ...],
    *,
    warmup_cycles: int,
    measured_cycles: int,
    local_rank: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    names = tuple(name for name, _ in schedules)
    functions = dict(schedules)
    orders = tuple(
        names[offset:] + names[:offset]
        for offset in range(len(names))
    )

    for cycle in range(warmup_cycles):
        for name in orders[cycle % len(orders)]:
            output = functions[name]()
            torch.cuda.synchronize()
            del output

    samples = {name: [] for name in names}
    for cycle in range(measured_cycles):
        for name in orders[cycle % len(orders)]:
            dist.barrier(device_ids=[local_rank])
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = functions[name]()
            torch.cuda.synchronize()
            samples[name].append(
                (time.perf_counter() - started) * 1_000.0
            )
            del output
    dist.barrier(device_ids=[local_rank])
    return {
        name: critical_path_summary(samples[name], device)
        for name in names
    }


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
        routes = (
            ("balanced", 0.0),
            ("global_bias1", 1.0),
        )
        warmup, iterations = 1, 3
    elif args.profile == "focused":
        token_values = [2048]
        routes = (("balanced", 0.0),)
        warmup, iterations = 10, 30
    elif args.profile == "crossover":
        token_values = [2048]
        routes = (("balanced", 0.0),)
        warmup, iterations = 2, 5
    else:
        token_values = [128, 512, 2048]
        routes = (
            ("balanced", 0.0),
            ("global_bias1", 1.0),
            ("global_bias4", 4.0),
        )
        warmup, iterations = 3, 10

    num_experts = world_size * args.experts_per_rank
    if args.experts_per_rank > 256:
        raise ValueError(
            "BF16 metadata packing requires experts_per_rank <= 256"
        )

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

    records: list[dict[str, Any]] = []
    for num_tokens in token_values:
        input_generator = torch.Generator(device=device)
        input_generator.manual_seed(10_000 + rank * 101 + num_tokens)
        hidden_states = torch.randn(
            (num_tokens, args.hidden),
            device=device,
            dtype=dtype,
            generator=input_generator,
        )

        for route_name, hot_bias in routes:
            bias = routing_bias(
                "global_biased",
                rank=rank,
                num_experts=num_experts,
                experts_per_rank=args.experts_per_rank,
                hot_bias=hot_bias,
                device=device,
            )

            def build_route() -> TensorDict:
                return route_and_pack(
                    hidden_states,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                )

            prepared_route = build_route()
            prepared_all_packet = build_packed_packet(prepared_route)
            prepared_metadata_packet = build_metadata_packet(
                prepared_route
            )
            split_dispatch = dispatch_assignments(prepared_route)
            all_in_one_dispatch = dispatch_assignments_packed(
                prepared_route,
                packet=prepared_all_packet,
            )
            metadata_dispatch = (
                dispatch_assignments_metadata_packed(
                    prepared_route,
                    metadata_packet=prepared_metadata_packet,
                )
            )

            split_expert_output = local_experts(
                split_dispatch,
                local_w1,
                local_w2,
            )
            all_in_one_expert_output = local_experts(
                all_in_one_dispatch,
                local_w1,
                local_w2,
            )
            metadata_expert_output = local_experts(
                metadata_dispatch,
                local_w1,
                local_w2,
            )
            split_output = return_and_combine(
                split_expert_output,
                prepared_route,
                split_dispatch,
                num_tokens=num_tokens,
            )
            all_in_one_output = return_and_combine(
                all_in_one_expert_output,
                prepared_route,
                all_in_one_dispatch,
                num_tokens=num_tokens,
            )
            metadata_output = return_and_combine(
                metadata_expert_output,
                prepared_route,
                metadata_dispatch,
                num_tokens=num_tokens,
            )

            topk_ids = prepared_route["topk_ids"]
            topk_weights = prepared_route["topk_weights"]
            send_counts = prepared_route["send_counts"]
            assert isinstance(topk_ids, torch.Tensor)
            assert isinstance(topk_weights, torch.Tensor)
            assert isinstance(send_counts, list)
            reference_output = fused_experts(
                hidden_states,
                global_w1,
                global_w2,
                topk_weights,
                topk_ids,
            )
            torch.cuda.synchronize()

            all_in_one_hidden_quality = output_quality(
                all_in_one_dispatch["recv_hidden"],
                split_dispatch["recv_hidden"],
            )
            all_in_one_weight_quality = output_quality(
                all_in_one_dispatch["recv_weights"],
                split_dispatch["recv_weights"],
            )
            metadata_hidden_quality = output_quality(
                metadata_dispatch["recv_hidden"],
                split_dispatch["recv_hidden"],
            )
            metadata_weight_quality = output_quality(
                metadata_dispatch["recv_weights"],
                split_dispatch["recv_weights"],
            )
            all_in_one_ids_equal = torch.equal(
                all_in_one_dispatch["recv_local_ids"],
                split_dispatch["recv_local_ids"],
            )
            metadata_ids_equal = torch.equal(
                metadata_dispatch["recv_local_ids"],
                split_dispatch["recv_local_ids"],
            )
            all_in_one_output_quality = output_quality(
                all_in_one_output,
                split_output,
            )
            metadata_output_quality = output_quality(
                metadata_output,
                split_output,
            )
            split_quality = output_quality(
                split_output,
                reference_output,
            )
            all_in_one_quality = output_quality(
                all_in_one_output,
                reference_output,
            )
            metadata_quality = output_quality(
                metadata_output,
                reference_output,
            )
            if (
                not all_in_one_ids_equal
                or not metadata_ids_equal
                or all_in_one_hidden_quality["max_abs_error"] != 0.0
                or all_in_one_weight_quality["max_abs_error"] != 0.0
                or metadata_hidden_quality["max_abs_error"] != 0.0
                or metadata_weight_quality["max_abs_error"] != 0.0
                or all_in_one_output_quality["max_abs_error"] != 0.0
                or metadata_output_quality["max_abs_error"] != 0.0
                or all_in_one_quality["normalized_rmse"]
                > args.max_nrmse
                or metadata_quality["normalized_rmse"]
                > args.max_nrmse
                or all_in_one_quality["cosine_similarity"]
                < args.min_cosine
                or metadata_quality["cosine_similarity"]
                < args.min_cosine
            ):
                raise RuntimeError(
                    f"Packed dispatch correctness failed for "
                    f"T={num_tokens}, route={route_name}"
                )

            send_counts_tensor = torch.tensor(
                send_counts,
                device=device,
                dtype=torch.int64,
            )
            gathered_counts = [
                torch.empty_like(send_counts_tensor)
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_counts, send_counts_tensor)
            send_matrix = [
                tensor.cpu().tolist()
                for tensor in gathered_counts
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

            def pack_all_only() -> torch.Tensor:
                return build_packed_packet(prepared_route)

            def pack_metadata_only() -> torch.Tensor:
                return build_metadata_packet(prepared_route)

            def split_dispatch_only() -> TensorDict:
                return dispatch_assignments(prepared_route)

            def all_in_one_dispatch_prebuilt() -> TensorDict:
                return dispatch_assignments_packed(
                    prepared_route,
                    packet=prepared_all_packet,
                )

            def all_in_one_dispatch_with_pack() -> TensorDict:
                return dispatch_assignments_packed(prepared_route)

            def metadata_dispatch_prebuilt() -> TensorDict:
                return dispatch_assignments_metadata_packed(
                    prepared_route,
                    metadata_packet=prepared_metadata_packet,
                )

            def metadata_dispatch_with_pack() -> TensorDict:
                return dispatch_assignments_metadata_packed(
                    prepared_route
                )

            def split_end_to_end() -> torch.Tensor:
                route = build_route()
                dispatched = dispatch_assignments(route)
                expert_output = local_experts(
                    dispatched,
                    local_w1,
                    local_w2,
                )
                return return_and_combine(
                    expert_output,
                    route,
                    dispatched,
                    num_tokens=num_tokens,
                )

            def all_in_one_end_to_end() -> torch.Tensor:
                route = build_route()
                packet = build_packed_packet(route)
                dispatched = dispatch_assignments_packed(
                    route,
                    packet=packet,
                )
                expert_output = local_experts(
                    dispatched,
                    local_w1,
                    local_w2,
                )
                return return_and_combine(
                    expert_output,
                    route,
                    dispatched,
                    num_tokens=num_tokens,
                )

            def metadata_end_to_end() -> torch.Tensor:
                route = build_route()
                metadata_packet = build_metadata_packet(route)
                dispatched = dispatch_assignments_metadata_packed(
                    route,
                    metadata_packet=metadata_packet,
                )
                expert_output = local_experts(
                    dispatched,
                    local_w1,
                    local_w2,
                )
                return return_and_combine(
                    expert_output,
                    route,
                    dispatched,
                    num_tokens=num_tokens,
                )

            timings: dict[str, dict[str, Any]] = {}
            schedules: tuple[
                tuple[str, Callable[[], Any]],
                ...,
            ] = (
                ("pack_all_only", pack_all_only),
                ("pack_metadata_only", pack_metadata_only),
                ("split_dispatch", split_dispatch_only),
                (
                    "all_in_one_dispatch_prebuilt",
                    all_in_one_dispatch_prebuilt,
                ),
                (
                    "all_in_one_dispatch_with_pack",
                    all_in_one_dispatch_with_pack,
                ),
                (
                    "metadata_dispatch_prebuilt",
                    metadata_dispatch_prebuilt,
                ),
                (
                    "metadata_dispatch_with_pack",
                    metadata_dispatch_with_pack,
                ),
                ("split_end_to_end", split_end_to_end),
                (
                    "all_in_one_end_to_end",
                    all_in_one_end_to_end,
                ),
                ("metadata_end_to_end", metadata_end_to_end),
            )
            if args.schedule_order == "reversed":
                schedules = tuple(reversed(schedules))
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

            crossover = None
            if args.profile == "crossover":
                crossover_dispatch = timed_crossover(
                    (
                        ("split", split_dispatch_only),
                        (
                            "all_in_one",
                            all_in_one_dispatch_with_pack,
                        ),
                        (
                            "metadata",
                            metadata_dispatch_with_pack,
                        ),
                    ),
                    warmup_cycles=6,
                    measured_cycles=30,
                    local_rank=local_rank,
                    device=device,
                )
                crossover_end_to_end = timed_crossover(
                    (
                        ("split", split_end_to_end),
                        (
                            "all_in_one",
                            all_in_one_end_to_end,
                        ),
                        ("metadata", metadata_end_to_end),
                    ),
                    warmup_cycles=6,
                    measured_cycles=30,
                    local_rank=local_rank,
                    device=device,
                )
                crossover = {
                    "method": (
                        "Latin-square rotation over split, "
                        "all-in-one, and metadata-packed"
                    ),
                    "warmup_cycles": 6,
                    "measured_cycles": 30,
                    "dispatch": crossover_dispatch,
                    "end_to_end": crossover_end_to_end,
                    "dispatch_speedups": {
                        "all_in_one": (
                            crossover_dispatch["split"][
                                "critical_path_ms"
                            ]
                            / crossover_dispatch["all_in_one"][
                                "critical_path_ms"
                            ]
                        ),
                        "metadata": (
                            crossover_dispatch["split"][
                                "critical_path_ms"
                            ]
                            / crossover_dispatch["metadata"][
                                "critical_path_ms"
                            ]
                        ),
                    },
                    "end_to_end_speedups": {
                        "all_in_one": (
                            crossover_end_to_end["split"][
                                "critical_path_ms"
                            ]
                            / crossover_end_to_end["all_in_one"][
                                "critical_path_ms"
                            ]
                        ),
                        "metadata": (
                            crossover_end_to_end["split"][
                                "critical_path_ms"
                            ]
                            / crossover_end_to_end["metadata"][
                                "critical_path_ms"
                            ]
                        ),
                    },
                }

            split_dispatch_ms = timings["split_dispatch"][
                "critical_path_ms"
            ]
            all_in_one_prebuilt_ms = timings[
                "all_in_one_dispatch_prebuilt"
            ]["critical_path_ms"]
            all_in_one_with_pack_ms = timings[
                "all_in_one_dispatch_with_pack"
            ]["critical_path_ms"]
            metadata_prebuilt_ms = timings[
                "metadata_dispatch_prebuilt"
            ]["critical_path_ms"]
            metadata_with_pack_ms = timings[
                "metadata_dispatch_with_pack"
            ]["critical_path_ms"]
            split_end_to_end_ms = timings["split_end_to_end"][
                "critical_path_ms"
            ]
            all_in_one_end_to_end_ms = timings[
                "all_in_one_end_to_end"
            ]["critical_path_ms"]
            metadata_end_to_end_ms = timings[
                "metadata_end_to_end"
            ]["critical_path_ms"]
            element_size = torch.empty(
                (),
                dtype=dtype,
            ).element_size()
            records.append(
                {
                    "tokens_per_rank": num_tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts_per_rank": args.experts_per_rank,
                    "num_experts": num_experts,
                    "top_k": args.top_k,
                    "route": route_name,
                    "hot_bias": hot_bias,
                    "dtype": str(dtype),
                    "routing": {
                        "send_matrix": send_matrix,
                        "rank_receive_assignments": receive_totals,
                        "rank0_receive_fraction": (
                            receive_totals[0] / total_assignments
                        ),
                        "max_to_mean_receive": (
                            max(receive_totals) / mean_receive
                        ),
                    },
                    "payload": {
                        "split_payload_collectives": 3,
                        "all_in_one_payload_collectives": 1,
                        "metadata_payload_collectives": 2,
                        "split_bytes_per_assignment": (
                            args.hidden * element_size
                            + torch.empty(
                                (),
                                dtype=torch.int32,
                            ).element_size()
                            + element_size
                        ),
                        "all_in_one_bytes_per_assignment": (
                            (args.hidden + 2) * element_size
                        ),
                        "metadata_bytes_per_assignment": (
                            (args.hidden + 2) * element_size
                        ),
                        "all_in_one_width_overhead_fraction": (
                            2 / args.hidden
                        ),
                    },
                    "correctness": {
                        "all_in_one_ids_equal": (
                            all_in_one_ids_equal
                        ),
                        "metadata_ids_equal": metadata_ids_equal,
                        "max_all_in_one_dispatch_hidden_error": max(
                            gather_scalar(
                                all_in_one_hidden_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_all_in_one_dispatch_weight_error": max(
                            gather_scalar(
                                all_in_one_weight_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_metadata_dispatch_hidden_error": max(
                            gather_scalar(
                                metadata_hidden_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_metadata_dispatch_weight_error": max(
                            gather_scalar(
                                metadata_weight_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_all_in_one_vs_split_output_error": max(
                            gather_scalar(
                                all_in_one_output_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_metadata_vs_split_output_error": max(
                            gather_scalar(
                                metadata_output_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_split_reference_nrmse": max(
                            gather_scalar(
                                split_quality["normalized_rmse"],
                                device,
                            )
                        ),
                        "max_all_in_one_reference_nrmse": max(
                            gather_scalar(
                                all_in_one_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_metadata_reference_nrmse": max(
                            gather_scalar(
                                metadata_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_all_in_one_reference_cosine": min(
                            gather_scalar(
                                all_in_one_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                        "min_metadata_reference_cosine": min(
                            gather_scalar(
                                metadata_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                    },
                    "timings": timings,
                    "crossover": crossover,
                    "all_in_one_dispatch_speedup_prebuilt": (
                        split_dispatch_ms / all_in_one_prebuilt_ms
                    ),
                    "all_in_one_dispatch_speedup_with_pack": (
                        split_dispatch_ms / all_in_one_with_pack_ms
                    ),
                    "metadata_dispatch_speedup_prebuilt": (
                        split_dispatch_ms / metadata_prebuilt_ms
                    ),
                    "metadata_dispatch_speedup_with_pack": (
                        split_dispatch_ms / metadata_with_pack_ms
                    ),
                    "all_in_one_end_to_end_speedup": (
                        split_end_to_end_ms
                        / all_in_one_end_to_end_ms
                    ),
                    "metadata_end_to_end_speedup": (
                        split_end_to_end_ms / metadata_end_to_end_ms
                    ),
                    "pack_all_fraction_of_end_to_end": (
                        timings["pack_all_only"]["critical_path_ms"]
                        / all_in_one_end_to_end_ms
                    ),
                    "pack_metadata_fraction_of_end_to_end": (
                        timings["pack_metadata_only"][
                            "critical_path_ms"
                        ]
                        / metadata_end_to_end_ms
                    ),
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )

            del (
                prepared_route,
                prepared_all_packet,
                prepared_metadata_packet,
                split_dispatch,
                all_in_one_dispatch,
                metadata_dispatch,
                split_expert_output,
                all_in_one_expert_output,
                metadata_expert_output,
                split_output,
                all_in_one_output,
                metadata_output,
                reference_output,
            )
            torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_packed_dispatch",
            "profile": args.profile,
            "schedule_order": args.schedule_order,
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
                "The split baseline exchanges hidden BF16, local expert int32, and router-weight BF16 payloads with three variable all-to-all collectives after one count exchange.",
                "The all-in-one path appends local expert ID and router weight as two BF16 columns, then uses one payload collective. Because the widened row stride is incompatible with the fused expert kernel, unpacking materializes a contiguous hidden-state tensor.",
                "The metadata-packed path keeps hidden BF16 contiguous and combines only local expert ID plus router weight into one BF16 metadata packet, reducing payload collectives from three to two without copying hidden activations.",
                "Both packed layouts use hidden+2 BF16 values per assignment instead of hidden BF16 + int32 ID + BF16 weight. With two local experts, BF16 IDs are represented exactly.",
                "Both packed variants and the split baseline are required to match exactly before timing.",
                "Dispatch timings include count exchange and unpack conversion. End-to-end timings include routing, packing when applicable, dispatch, local fused experts, return all-to-all, and weighted combine.",
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
        choices=["smoke", "full", "focused", "crossover"],
        default="smoke",
    )
    parser.add_argument(
        "--schedule-order",
        choices=["normal", "reversed"],
        default="normal",
    )
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=2816)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-nrmse", type=float, default=5e-3)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
