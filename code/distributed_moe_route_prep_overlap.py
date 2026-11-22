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
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)

from distributed_ep_int8_return_overlap import quality_across_chunks
from distributed_ep_roundtrip import (
    critical_path_summary,
    timed_schedule,
)
from distributed_moe_end_to_end import (
    TensorDict,
    local_experts,
    route_and_pack,
    routing_bias,
)
from distributed_moe_int8_roundtrip import aligned_packet_size
from distributed_moe_packed_dispatch import (
    build_metadata_packet,
    timed_crossover,
)
from distributed_moe_variable_int8_overlap import (
    aggregate_routing,
    finish_dispatch,
    finish_return,
    launch_dispatch,
    launch_return,
    prepare_dispatch,
    profile_settings,
)


State = dict[str, Any]


def route_and_pack_device(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    *,
    top_k: int,
    experts_per_rank: int,
    world_size: int,
) -> TensorDict:
    gating_output = (
        F.linear(hidden_states, gate_weight).float()
        + bias
        + tie_breaker
    )
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states,
        gating_output,
        top_k,
        True,
    )
    topk_weights = topk_weights.to(hidden_states.dtype)

    token_indices = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
        dtype=torch.int64,
    ).repeat_interleave(top_k)
    flat_ids = topk_ids.reshape(-1)
    destinations = torch.div(
        flat_ids,
        experts_per_rank,
        rounding_mode="floor",
    )
    local_ids = torch.remainder(flat_ids, experts_per_rank)
    permutation = torch.argsort(destinations, stable=True)

    sorted_destinations = destinations.index_select(0, permutation)
    sorted_token_indices = token_indices.index_select(0, permutation)
    sorted_local_ids = local_ids.index_select(0, permutation)
    sorted_weights = topk_weights.reshape(-1).index_select(
        0,
        permutation,
    )
    send_hidden = hidden_states.index_select(0, sorted_token_indices)
    send_counts_tensor = torch.bincount(
        sorted_destinations,
        minlength=world_size,
    ).to(torch.int64)
    return {
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "send_hidden": send_hidden.contiguous(),
        "send_local_ids": sorted_local_ids.to(torch.int32)
        .reshape(-1, 1)
        .contiguous(),
        "send_weights": sorted_weights.reshape(-1, 1).contiguous(),
        "sorted_token_indices": sorted_token_indices,
        "sorted_assignment_indices": permutation,
        "send_counts_tensor": send_counts_tensor,
    }


def start_prepare(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    *,
    top_k: int,
    experts_per_rank: int,
    world_size: int,
    count_group: dist.ProcessGroup,
    prep_stream: torch.cuda.Stream,
    send_counts_host: torch.Tensor,
    route_ready: torch.cuda.Event,
) -> State:
    with torch.cuda.stream(prep_stream):
        route = route_and_pack_device(
            hidden,
            gate_weight,
            bias,
            tie_breaker,
            top_k=top_k,
            experts_per_rank=experts_per_rank,
            world_size=world_size,
        )
        metadata = build_metadata_packet(route)
        send_counts_tensor = route["send_counts_tensor"]
        assert isinstance(send_counts_tensor, torch.Tensor)
        send_counts_host.copy_(
            send_counts_tensor,
            non_blocking=True,
        )
        route_ready.record(prep_stream)

    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    return {
        "route": route,
        "metadata": metadata,
        "send_counts_host": send_counts_host,
        "recv_counts_tensor": recv_counts_tensor,
        "count_group": count_group,
        "route_ready": route_ready,
    }


def finish_prepare(state: State) -> dict[str, Any]:
    state["route_ready"].synchronize()
    route = state["route"]
    send_counts_tensor = route["send_counts_tensor"]
    assert isinstance(send_counts_tensor, torch.Tensor)
    dist.all_to_all_single(
        state["recv_counts_tensor"],
        send_counts_tensor,
        group=state["count_group"],
    )
    send_counts = [
        int(value)
        for value in state["send_counts_host"].tolist()
    ]
    recv_counts = [
        int(value)
        for value in state["recv_counts_tensor"].cpu().tolist()
    ]
    route["send_counts"] = send_counts
    send_hidden = route["send_hidden"]
    assert isinstance(send_hidden, torch.Tensor)
    consumer_stream = torch.cuda.current_stream(
        send_hidden.device
    )
    for value in route.values():
        if isinstance(value, torch.Tensor):
            value.record_stream(consumer_stream)
    state["metadata"].record_stream(consumer_stream)
    recv_total = sum(recv_counts)
    if (
        any(count < 0 for count in send_counts + recv_counts)
        or sum(send_counts) != send_hidden.shape[0]
        or recv_total > send_hidden.shape[0] * len(recv_counts)
    ):
        raise RuntimeError(
            "Invalid online split counts: "
            f"send={send_counts}, recv={recv_counts}"
        )
    return {
        "route": route,
        "metadata": state["metadata"],
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


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    count_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="nccl",
    )
    count_group_warmup = torch.ones(
        1,
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(count_group_warmup, group=count_group)
    del count_group_warmup

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
    assignments_per_chunk = tokens_per_chunk * args.top_k
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
    prep_stream = torch.cuda.Stream(device=device)

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
        precomputed_routes: list[TensorDict] = []
        precomputed_dispatches: list[dict[str, Any]] = []
        reference_outputs: list[torch.Tensor] = []
        return_buffers: list[dict[str, torch.Tensor]] = []
        send_count_hosts: list[torch.Tensor] = []
        route_events: list[torch.cuda.Event] = []

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
            hidden_chunks.append(hidden)
            precomputed_routes.append(route)
            precomputed_dispatches.append(
                prepare_dispatch(route)
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
            return_buffers.append(
                {
                    "bf16": torch.empty(
                        (assignments_per_chunk, args.hidden),
                        device=device,
                        dtype=dtype,
                    ),
                    "int8": torch.empty(
                        (assignments_per_chunk, args.hidden),
                        device=device,
                        dtype=torch.int8,
                    ),
                    "scale": torch.empty(
                        (assignments_per_chunk, 1),
                        device=device,
                        dtype=dtype,
                    ),
                    "packet": torch.empty(
                        (assignments_per_chunk, packet_size),
                        device=device,
                        dtype=torch.uint8,
                    ),
                }
            )
            send_count_hosts.append(
                torch.empty(
                    world_size,
                    dtype=torch.int64,
                    pin_memory=True,
                )
            )
            route_events.append(torch.cuda.Event())

        routing = aggregate_routing(
            precomputed_routes,
            world_size,
            device,
        )

        def start_chunk(chunk: int) -> State:
            return start_prepare(
                hidden_chunks[chunk],
                gate_weight,
                bias,
                tie_breaker,
                top_k=args.top_k,
                experts_per_rank=args.experts_per_rank,
                world_size=world_size,
                count_group=count_group,
                prep_stream=prep_stream,
                send_counts_host=send_count_hosts[chunk],
                route_ready=route_events[chunk],
            )

        def compute(dispatched: TensorDict) -> torch.Tensor:
            return local_experts(
                dispatched,
                local_w1,
                local_w2,
            )

        def payload_pipeline(
            kind: str,
            prepared_dispatches: list[dict[str, Any]],
        ) -> list[torch.Tensor]:
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

        def online_staged(kind: str) -> list[torch.Tensor]:
            prepared = [
                finish_prepare(start_chunk(chunk))
                for chunk in range(chunks)
            ]
            return payload_pipeline(kind, prepared)

        def online_overlapped(kind: str) -> list[torch.Tensor]:
            outputs: list[torch.Tensor | None] = [
                None for _ in range(chunks)
            ]
            prepared_current = finish_prepare(start_chunk(0))
            first_dispatch = launch_dispatch(prepared_current)
            pending_prepare = (
                start_chunk(1)
                if chunks > 1
                else None
            )
            current_dispatched = finish_dispatch(first_dispatch)
            if chunks == 1:
                expert_output = compute(current_dispatched)
                outputs[0] = finish_return(
                    launch_return(
                        kind,
                        expert_output,
                        prepared_current,
                        return_buffers[0],
                        packet_alignment=args.packet_alignment,
                        num_tokens=tokens_per_chunk,
                    )
                )
                return [outputs[0]]

            assert pending_prepare is not None
            prepared_next = finish_prepare(pending_prepare)
            next_dispatch = launch_dispatch(prepared_next)
            pending_prepare = (
                start_chunk(2)
                if chunks > 2
                else None
            )
            first_expert_output = compute(current_dispatched)
            previous_return = launch_return(
                kind,
                first_expert_output,
                prepared_current,
                return_buffers[0],
                packet_alignment=args.packet_alignment,
                num_tokens=tokens_per_chunk,
            )
            current_dispatched = finish_dispatch(next_dispatch)
            prepared_current = prepared_next

            for chunk in range(1, chunks):
                following_dispatch = None
                prepared_next = None
                if chunk + 1 < chunks:
                    assert pending_prepare is not None
                    prepared_next = finish_prepare(
                        pending_prepare
                    )
                    following_dispatch = launch_dispatch(
                        prepared_next
                    )
                    pending_prepare = (
                        start_chunk(chunk + 2)
                        if chunk + 2 < chunks
                        else None
                    )
                expert_output = compute(current_dispatched)
                current_return = launch_return(
                    kind,
                    expert_output,
                    prepared_current,
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
                    assert prepared_next is not None
                    prepared_current = prepared_next
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

        def online_prepare_only() -> list[dict[str, Any]]:
            return [
                finish_prepare(start_chunk(chunk))
                for chunk in range(chunks)
            ]

        schedules: dict[str, Callable[[], list[torch.Tensor]]] = {
            "bf16_precomputed": lambda: payload_pipeline(
                "bf16",
                precomputed_dispatches,
            ),
            "bf16_online_staged": lambda: online_staged("bf16"),
            "bf16_online_overlapped": lambda: online_overlapped(
                "bf16"
            ),
            "packet_precomputed": lambda: payload_pipeline(
                "packet",
                precomputed_dispatches,
            ),
            "packet_online_staged": lambda: online_staged(
                "packet"
            ),
            "packet_online_overlapped": lambda: online_overlapped(
                "packet"
            ),
        }

        quality_outputs = {
            name: schedule()
            for name, schedule in schedules.items()
        }
        torch.cuda.synchronize()
        quality_vs_reference = {
            name: quality_across_chunks(
                output,
                reference_outputs,
                device,
            )
            for name, output in quality_outputs.items()
        }
        exact_quality = {
            "bf16_staged_vs_precomputed": quality_across_chunks(
                quality_outputs["bf16_online_staged"],
                quality_outputs["bf16_precomputed"],
                device,
            ),
            "bf16_overlapped_vs_precomputed": quality_across_chunks(
                quality_outputs["bf16_online_overlapped"],
                quality_outputs["bf16_precomputed"],
                device,
            ),
            "packet_staged_vs_precomputed": quality_across_chunks(
                quality_outputs["packet_online_staged"],
                quality_outputs["packet_precomputed"],
                device,
            ),
            "packet_overlapped_vs_precomputed": quality_across_chunks(
                quality_outputs["packet_online_overlapped"],
                quality_outputs["packet_precomputed"],
                device,
            ),
        }
        for name in (
            "bf16_precomputed",
            "bf16_online_staged",
            "bf16_online_overlapped",
        ):
            if quality_vs_reference[name]["max_abs_error"] != 0.0:
                raise RuntimeError(
                    f"Online BF16 preparation disagreed: "
                    f"{name}={quality_vs_reference[name]}"
                )
        if any(
            record["max_abs_error"] != 0.0
            for record in exact_quality.values()
        ):
            raise RuntimeError(
                f"Online preparation changed output: {exact_quality}"
            )
        for name in (
            "packet_precomputed",
            "packet_online_staged",
            "packet_online_overlapped",
        ):
            quality = quality_vs_reference[name]
            if (
                quality["max_normalized_rmse"]
                > args.max_output_nrmse
                or quality["min_cosine_similarity"]
                < args.min_output_cosine
            ):
                raise RuntimeError(
                    f"Packet quality failed for {name}: {quality}"
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
        prepare_timing = critical_path_summary(
            timed_schedule(
                online_prepare_only,
                warmup=warmup,
                iterations=iterations,
                local_rank=local_rank,
            ),
            device,
        )
        crossover = None
        if args.profile == "crossover":
            crossover = timed_crossover(
                tuple(schedules.items()),
                warmup_cycles=6,
                measured_cycles=args.crossover_cycles,
                local_rank=local_rank,
                device=device,
            )

        def speedup(baseline: str, candidate: str) -> float:
            return (
                timings[baseline]["critical_path_ms"]
                / timings[candidate]["critical_path_ms"]
            )

        def recovery_fraction(
            precomputed: str,
            staged: str,
            overlapped: str,
        ) -> float:
            preparation_tax = (
                timings[staged]["critical_path_ms"]
                - timings[precomputed]["critical_path_ms"]
            )
            if preparation_tax <= 0:
                return 0.0
            return (
                timings[staged]["critical_path_ms"]
                - timings[overlapped]["critical_path_ms"]
            ) / preparation_tax

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
                "preparation": {
                    "router_pack_in_timed_region": True,
                    "count_exchange_in_timed_region": True,
                    "send_count_host_copy": (
                        "pinned_nonblocking"
                    ),
                    "recv_count_host_copy": (
                        "blocking_tensor_cpu"
                    ),
                    "count_collective_group": (
                        "dedicated_nccl"
                    ),
                    "count_collective_overlap": False,
                    "prepare_only_timing": prepare_timing,
                },
                "timings": timings,
                "speedups": {
                    "bf16_overlap_vs_staged": speedup(
                        "bf16_online_staged",
                        "bf16_online_overlapped",
                    ),
                    "packet_overlap_vs_staged": speedup(
                        "packet_online_staged",
                        "packet_online_overlapped",
                    ),
                    "bf16_online_vs_precomputed": speedup(
                        "bf16_precomputed",
                        "bf16_online_overlapped",
                    ),
                    "packet_online_vs_precomputed": speedup(
                        "packet_precomputed",
                        "packet_online_overlapped",
                    ),
                    "packet_online_vs_bf16_online": speedup(
                        "bf16_online_overlapped",
                        "packet_online_overlapped",
                    ),
                    "bf16_recovery_fraction": recovery_fraction(
                        "bf16_precomputed",
                        "bf16_online_staged",
                        "bf16_online_overlapped",
                    ),
                    "packet_recovery_fraction": (
                        recovery_fraction(
                            "packet_precomputed",
                            "packet_online_staged",
                            "packet_online_overlapped",
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
            precomputed_routes,
            precomputed_dispatches,
            reference_outputs,
            return_buffers,
            quality_outputs,
        )
        torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_route_prep_overlap",
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
                "Online schedules include router projection, fused top-k, token permutation, metadata packing, count exchange, CPU split extraction, variable dispatch, local experts, return, and weighted combine.",
                "Send counts use a pinned nonblocking copy; count exchange and receive-count extraction run after the route event because NCCL did not inherit the external prep-stream dependency reliably.",
                "The overlapped schedule double-buffers future device router/packing work while count exchange and Python split extraction remain exposed.",
                "Precomputed schedules reproduce the v23 payload-only lower bound; staged schedules expose all online preparation before payload pipelining.",
                "BF16 online outputs must match the global-expert reference exactly, and online preparation must not change either BF16 or packet output.",
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

    dist.destroy_process_group(count_group)
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
