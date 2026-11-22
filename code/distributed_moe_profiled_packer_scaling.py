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
import triton
import triton.language as tl

from distributed_moe_end_to_end import (
    TensorDict,
    critical_path_summary,
    gather_scalar,
    route_and_pack,
    routing_bias,
    timed_schedule,
)
from distributed_moe_packed_dispatch import timed_crossover
from distributed_moe_profiled_packer import (
    pack_triton_atomic,
    pack_two_sort,
    profile_tensors,
    router_topk,
)
from distributed_moe_profiled_transport import (
    gather_segment_counts,
    make_hidden,
)


@triton.jit
def _reserve_profile_slots_kernel(
    topk_ids_ptr,
    capacity_ptr,
    capacity_offsets_ptr,
    counters_ptr,
    fixed_indices_ptr,
    overflow_ptr,
    experts_per_rank: tl.constexpr,
):
    assignment = tl.program_id(0)
    expert_id = tl.load(
        topk_ids_ptr + assignment
    ).to(tl.int32)
    destination = expert_id // experts_per_rank
    position = tl.atomic_add(
        counters_ptr + destination,
        1,
    )
    capacity = tl.load(
        capacity_ptr + destination
    )
    accepted = position < capacity
    fixed_index = (
        tl.load(
            capacity_offsets_ptr + destination
        )
        + position
    )
    tl.store(
        fixed_indices_ptr + assignment,
        tl.where(accepted, fixed_index, -1),
    )
    tl.atomic_add(
        overflow_ptr,
        1,
        mask=~accepted,
    )


@triton.jit
def _scatter_profile_tiles_kernel(
    hidden_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    fixed_indices_ptr,
    send_hidden_ptr,
    send_local_ids_ptr,
    send_weights_ptr,
    fixed_token_indices_ptr,
    hidden_row_stride,
    send_row_stride,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    experts_per_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    assignment = tl.program_id(0)
    tile = tl.program_id(1)
    fixed_index = tl.load(
        fixed_indices_ptr + assignment
    )
    accepted = fixed_index >= 0
    token_index = assignment // top_k

    offsets = (
        tile * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    mask = accepted & (offsets < hidden_size)
    hidden = tl.load(
        hidden_ptr
        + token_index * hidden_row_stride
        + offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(
        send_hidden_ptr
        + fixed_index * send_row_stride
        + offsets,
        hidden,
        mask=mask,
    )

    metadata_mask = accepted & (tile == 0)
    expert_id = tl.load(
        topk_ids_ptr + assignment
    ).to(tl.int32)
    local_id = expert_id % experts_per_rank
    weight = tl.load(
        topk_weights_ptr + assignment
    )
    tl.store(
        send_local_ids_ptr + fixed_index,
        local_id,
        mask=metadata_mask,
    )
    tl.store(
        send_weights_ptr + fixed_index,
        weight,
        mask=metadata_mask,
    )
    tl.store(
        fixed_token_indices_ptr + fixed_index,
        token_index,
        mask=metadata_mask,
    )


def profile_configurations(
    profile: str,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[str, float], ...],
    int,
    int,
]:
    if profile == "smoke":
        return (
            (
                (100, 512),
                (100, 4096),
            ),
            (
                ("balanced", 0.0),
                ("global_bias4", 4.0),
            ),
            1,
            2,
        )
    if profile == "crossover":
        return (
            (
                (512, 512),
                (8192, 512),
                (512, 4096),
                (8192, 4096),
            ),
            (
                ("balanced", 0.0),
                ("global_bias1", 1.0),
                ("global_bias4", 4.0),
            ),
            1,
            2,
        )
    return (
        tuple(
            (num_tokens, hidden)
            for hidden in (512, 1024, 4096)
            for num_tokens in (512, 2048, 8192)
        ),
        (
            ("balanced", 0.0),
            ("global_bias1", 1.0),
            ("global_bias4", 4.0),
        ),
        2,
        5,
    )


def pack_triton_tiled(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    profile: dict[str, torch.Tensor | int | list[int]],
    tile_size: int,
) -> TensorDict:
    capacity_row = profile["capacity_row"]
    capacity_i64 = profile["capacity_i64"]
    capacity_i32 = profile["capacity_i32"]
    offsets_i32 = profile["offsets_i32"]
    fixed_rows = profile["fixed_rows"]
    assert isinstance(capacity_row, list)
    assert isinstance(capacity_i64, torch.Tensor)
    assert isinstance(capacity_i32, torch.Tensor)
    assert isinstance(offsets_i32, torch.Tensor)
    assert isinstance(fixed_rows, int)

    num_tokens, top_k = topk_ids.shape
    num_assignments = topk_ids.numel()
    send_hidden = torch.zeros(
        (fixed_rows, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    send_local_ids = torch.zeros(
        (fixed_rows, 1),
        device=hidden_states.device,
        dtype=torch.int32,
    )
    send_weights = torch.zeros(
        (fixed_rows, 1),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    fixed_token_indices = torch.zeros(
        fixed_rows,
        device=hidden_states.device,
        dtype=torch.int64,
    )
    counters = torch.zeros(
        len(capacity_row),
        device=hidden_states.device,
        dtype=torch.int32,
    )
    fixed_indices = torch.empty(
        num_assignments,
        device=hidden_states.device,
        dtype=torch.int32,
    )
    overflow_count = torch.zeros(
        (),
        device=hidden_states.device,
        dtype=torch.int32,
    )

    _reserve_profile_slots_kernel[(num_assignments,)](
        topk_ids,
        capacity_i32,
        offsets_i32,
        counters,
        fixed_indices,
        overflow_count,
        experts_per_rank=experts_per_rank,
        num_warps=4,
    )
    _scatter_profile_tiles_kernel[
        (
            num_assignments,
            triton.cdiv(
                hidden_states.shape[1],
                tile_size,
            ),
        )
    ](
        hidden_states,
        topk_ids,
        topk_weights,
        fixed_indices,
        send_hidden,
        send_local_ids,
        send_weights,
        fixed_token_indices,
        hidden_states.stride(0),
        send_hidden.stride(0),
        hidden_size=hidden_states.shape[1],
        top_k=top_k,
        experts_per_rank=experts_per_rank,
        BLOCK_SIZE=tile_size,
        num_warps=4,
    )

    return {
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "send_hidden": send_hidden,
        "send_local_ids": send_local_ids,
        "send_weights": send_weights,
        "fixed_token_indices": fixed_token_indices,
        "fixed_indices": fixed_indices,
        "selected_token_indices": torch.arange(
            num_tokens,
            device=hidden_states.device,
            dtype=torch.int64,
        ).repeat_interleave(top_k),
        "selected_weights": topk_weights.reshape(-1),
        "selected_assignment_indices": torch.arange(
            num_assignments,
            device=hidden_states.device,
            dtype=torch.int64,
        ),
        "accepted_counts_tensor": torch.minimum(
            counters.to(torch.int64),
            capacity_i64,
        ),
        "send_counts_tensor": capacity_i64,
        "send_counts": list(capacity_row),
        "overflow_count": overflow_count.to(torch.int64),
    }


def fixed_row_masks(
    capacity_i64: torch.Tensor,
    accepted_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    destinations = torch.repeat_interleave(
        torch.arange(
            capacity_i64.numel(),
            device=capacity_i64.device,
            dtype=torch.int64,
        ),
        capacity_i64,
    )
    offsets = (
        torch.cumsum(capacity_i64, dim=0)
        - capacity_i64
    )
    positions = (
        torch.arange(
            destinations.numel(),
            device=capacity_i64.device,
            dtype=torch.int64,
        )
        - offsets.index_select(
            0,
            destinations,
        )
    )
    valid = positions < accepted_counts.index_select(
        0,
        destinations,
    )
    return destinations, valid


def packed_correctness(
    route: TensorDict,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    num_experts: int,
    profile: dict[str, torch.Tensor | int | list[int]],
) -> dict[str, float | bool | int]:
    capacity_i64 = profile["capacity_i64"]
    assert isinstance(capacity_i64, torch.Tensor)
    accepted_counts = route["accepted_counts_tensor"]
    send_hidden = route["send_hidden"]
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    fixed_token_indices = route[
        "fixed_token_indices"
    ]
    overflow_count = route["overflow_count"]
    assert isinstance(accepted_counts, torch.Tensor)
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_local_ids, torch.Tensor)
    assert isinstance(send_weights, torch.Tensor)
    assert isinstance(
        fixed_token_indices,
        torch.Tensor,
    )
    assert isinstance(overflow_count, torch.Tensor)

    destinations, valid = fixed_row_masks(
        capacity_i64,
        accepted_counts,
    )
    packed_tokens = fixed_token_indices[valid]
    packed_experts = (
        destinations[valid] * experts_per_rank
        + send_local_ids.reshape(-1)[valid].to(
            torch.int64
        )
    )
    packed_keys = (
        packed_tokens * num_experts
        + packed_experts
    )
    packed_order = torch.argsort(packed_keys)

    top_k = topk_ids.shape[1]
    reference_tokens = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
        dtype=torch.int64,
    ).repeat_interleave(top_k)
    reference_experts = topk_ids.reshape(-1).to(
        torch.int64
    )
    reference_keys = (
        reference_tokens * num_experts
        + reference_experts
    )
    reference_order = torch.argsort(reference_keys)
    sorted_reference_tokens = (
        reference_tokens.index_select(
            0,
            reference_order,
        )
    )
    sorted_reference_weights = (
        topk_weights.reshape(-1).index_select(
            0,
            reference_order,
        )
    )
    sorted_packed_tokens = packed_tokens.index_select(
        0,
        packed_order,
    )
    sorted_packed_weights = (
        send_weights.reshape(-1)[valid].index_select(
            0,
            packed_order,
        )
    )
    sorted_packed_hidden = send_hidden[
        valid
    ].index_select(
        0,
        packed_order,
    )
    sorted_reference_hidden = hidden_states.index_select(
        0,
        sorted_reference_tokens,
    )
    padded = ~valid
    padding_hidden_max = (
        send_hidden[padded].abs().max().item()
        if padded.any()
        else 0.0
    )
    padding_weight_max = (
        send_weights.reshape(-1)[padded]
        .abs()
        .max()
        .item()
        if padded.any()
        else 0.0
    )
    return {
        "overflow_count": int(
            overflow_count.item()
        ),
        "valid_rows": int(valid.sum().item()),
        "expected_rows": int(topk_ids.numel()),
        "keys_equal": torch.equal(
            packed_keys.index_select(
                0,
                packed_order,
            ),
            reference_keys.index_select(
                0,
                reference_order,
            ),
        ),
        "tokens_equal": torch.equal(
            sorted_packed_tokens,
            sorted_reference_tokens,
        ),
        "weights_equal": torch.equal(
            sorted_packed_weights,
            sorted_reference_weights,
        ),
        "hidden_equal": torch.equal(
            sorted_packed_hidden,
            sorted_reference_hidden,
        ),
        "padding_hidden_max": (
            padding_hidden_max
        ),
        "padding_weight_max": (
            padding_weight_max
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

    (
        shapes,
        routes,
        warmup,
        iterations,
    ) = profile_configurations(args.profile)
    num_experts = (
        world_size * args.experts_per_rank
    )
    tie_breaker = (
        torch.arange(
            num_experts,
            device=device,
            dtype=torch.float32,
        )
        * 1e-6
    )

    records: list[dict[str, Any]] = []
    for num_tokens, hidden_size in shapes:
        generator = torch.Generator(device=device)
        generator.manual_seed(
            7_000 + hidden_size
        )
        gate_weight = (
            torch.randn(
                (num_experts, hidden_size),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / math.sqrt(hidden_size)
        )
        for route_name, hot_bias in routes:
            bias = routing_bias(
                "global_biased",
                rank=rank,
                num_experts=num_experts,
                experts_per_rank=(
                    args.experts_per_rank
                ),
                hot_bias=hot_bias,
                device=device,
            )
            hidden = make_hidden(
                num_tokens,
                hidden_size,
                route_name=route_name,
                rank=rank,
                batch=0,
                calibration=False,
                device=device,
                dtype=dtype,
            )
            topk_ids, topk_weights = router_topk(
                hidden,
                gate_weight,
                bias,
                tie_breaker,
                top_k=args.top_k,
            )
            destinations = torch.div(
                topk_ids.reshape(-1),
                args.experts_per_rank,
                rounding_mode="floor",
            )
            evaluation_counts = torch.bincount(
                destinations,
                minlength=world_size,
            ).to(torch.int64)
            capacity_row = [
                int(value) + args.padding_margin
                for value in (
                    evaluation_counts.cpu().tolist()
                )
            ]
            evaluation_matrix = gather_segment_counts(
                [
                    int(value)
                    for value in (
                        evaluation_counts.cpu().tolist()
                    )
                ],
                device,
            )
            capacity_matrix = [
                [
                    value + args.padding_margin
                    for value in row
                ]
                for row in evaluation_matrix
            ]
            profile = profile_tensors(
                capacity_row,
                device,
            )

            methods: list[
                tuple[
                    str,
                    Callable[[], TensorDict],
                ]
            ] = [
                (
                    "two_sort",
                    lambda: pack_two_sort(
                        hidden,
                        topk_ids,
                        topk_weights,
                        experts_per_rank=(
                            args.experts_per_rank
                        ),
                        profile=profile,
                    ),
                ),
            ]
            if hidden_size <= args.one_pass_max_hidden:
                methods.append(
                    (
                        "one_pass_atomic",
                        lambda: pack_triton_atomic(
                            hidden,
                            topk_ids,
                            topk_weights,
                            experts_per_rank=(
                                args.experts_per_rank
                            ),
                            profile=profile,
                        ),
                    )
                )
            methods.append(
                (
                    "tiled_atomic",
                    lambda: pack_triton_tiled(
                        hidden,
                        topk_ids,
                        topk_weights,
                        experts_per_rank=(
                            args.experts_per_rank
                        ),
                        profile=profile,
                        tile_size=args.tile_size,
                    ),
                )
            )
            if args.reverse_methods:
                methods.reverse()
            pack_functions = dict(methods)

            def dynamic_route() -> TensorDict:
                return route_and_pack(
                    hidden,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=(
                        args.experts_per_rank
                    ),
                    world_size=world_size,
                )

            route_functions: dict[
                str,
                Callable[[], TensorDict],
            ] = {"dynamic": dynamic_route}
            for method_name in pack_functions:
                def route_and_fixed_pack(
                    name: str = method_name,
                ) -> TensorDict:
                    ids, weights = router_topk(
                        hidden,
                        gate_weight,
                        bias,
                        tie_breaker,
                        top_k=args.top_k,
                    )
                    if name == "two_sort":
                        return pack_two_sort(
                            hidden,
                            ids,
                            weights,
                            experts_per_rank=(
                                args.experts_per_rank
                            ),
                            profile=profile,
                        )
                    if name == "one_pass_atomic":
                        return pack_triton_atomic(
                            hidden,
                            ids,
                            weights,
                            experts_per_rank=(
                                args.experts_per_rank
                            ),
                            profile=profile,
                        )
                    return pack_triton_tiled(
                        hidden,
                        ids,
                        weights,
                        experts_per_rank=(
                            args.experts_per_rank
                        ),
                        profile=profile,
                        tile_size=args.tile_size,
                    )

                route_functions[
                    method_name
                ] = route_and_fixed_pack

            correctness: dict[
                str,
                dict[str, float | bool | int],
            ] = {}
            for method_name, function in (
                pack_functions.items()
            ):
                packed = function()
                local = packed_correctness(
                    packed,
                    hidden,
                    topk_ids,
                    topk_weights,
                    experts_per_rank=(
                        args.experts_per_rank
                    ),
                    num_experts=num_experts,
                    profile=profile,
                )
                local_pass = (
                    local["overflow_count"] == 0
                    and local["valid_rows"]
                    == local["expected_rows"]
                    and bool(local["keys_equal"])
                    and bool(local["tokens_equal"])
                    and bool(local["weights_equal"])
                    and bool(local["hidden_equal"])
                    and local["padding_hidden_max"]
                    == 0.0
                    and local["padding_weight_max"]
                    == 0.0
                )
                global_pass = (
                    min(
                        gather_scalar(
                            float(local_pass),
                            device,
                        )
                    )
                    == 1.0
                )
                if not global_pass:
                    raise RuntimeError(
                        "Tiled packer correctness failed: "
                        f"method={method_name}, "
                        f"shape={(num_tokens, hidden_size)}, "
                        f"route={route_name}, "
                        f"local={local}"
                    )
                correctness[method_name] = local
                del packed

            pack_timings = {
                name: critical_path_summary(
                    timed_schedule(
                        function,
                        warmup=warmup,
                        iterations=iterations,
                        local_rank=local_rank,
                    ),
                    device,
                )
                for name, function in (
                    pack_functions.items()
                )
            }
            route_timings = {
                name: critical_path_summary(
                    timed_schedule(
                        function,
                        warmup=warmup,
                        iterations=iterations,
                        local_rank=local_rank,
                    ),
                    device,
                )
                for name, function in (
                    route_functions.items()
                )
            }

            pack_crossover = None
            route_crossover = None
            if args.profile == "crossover":
                pack_crossover = timed_crossover(
                    tuple(pack_functions.items()),
                    warmup_cycles=6,
                    measured_cycles=(
                        args.crossover_cycles
                    ),
                    local_rank=local_rank,
                    device=device,
                )
                route_crossover = timed_crossover(
                    tuple(route_functions.items()),
                    warmup_cycles=8,
                    measured_cycles=(
                        args.crossover_cycles
                    ),
                    local_rank=local_rank,
                    device=device,
                )

            global_assignments = (
                world_size
                * num_tokens
                * args.top_k
            )
            global_fixed_rows = sum(
                sum(row)
                for row in capacity_matrix
            )
            speedups = {
                "tiled_vs_two_sort_pack": (
                    pack_timings["two_sort"][
                        "critical_path_ms"
                    ]
                    / pack_timings["tiled_atomic"][
                        "critical_path_ms"
                    ]
                ),
                "tiled_vs_two_sort_route": (
                    route_timings["two_sort"][
                        "critical_path_ms"
                    ]
                    / route_timings["tiled_atomic"][
                        "critical_path_ms"
                    ]
                ),
                "tiled_vs_dynamic_route": (
                    route_timings["dynamic"][
                        "critical_path_ms"
                    ]
                    / route_timings["tiled_atomic"][
                        "critical_path_ms"
                    ]
                ),
            }
            if "one_pass_atomic" in pack_timings:
                speedups.update(
                    {
                        "one_pass_vs_two_sort_pack": (
                            pack_timings["two_sort"][
                                "critical_path_ms"
                            ]
                            / pack_timings[
                                "one_pass_atomic"
                            ]["critical_path_ms"]
                        ),
                        "tiled_vs_one_pass_pack": (
                            pack_timings[
                                "one_pass_atomic"
                            ]["critical_path_ms"]
                            / pack_timings[
                                "tiled_atomic"
                            ]["critical_path_ms"]
                        ),
                        "tiled_vs_one_pass_route": (
                            route_timings[
                                "one_pass_atomic"
                            ]["critical_path_ms"]
                            / route_timings[
                                "tiled_atomic"
                            ]["critical_path_ms"]
                        ),
                    }
                )
            records.append(
                {
                    "tokens_per_rank": num_tokens,
                    "assignments_per_rank": (
                        num_tokens * args.top_k
                    ),
                    "hidden": hidden_size,
                    "experts_per_rank": (
                        args.experts_per_rank
                    ),
                    "num_experts": num_experts,
                    "top_k": args.top_k,
                    "route": route_name,
                    "hot_bias": hot_bias,
                    "padding_margin": (
                        args.padding_margin
                    ),
                    "tile_size": args.tile_size,
                    "one_pass_max_hidden": (
                        args.one_pass_max_hidden
                    ),
                    "supported_methods": list(
                        pack_functions
                    ),
                    "evaluation_count_matrix": (
                        evaluation_matrix
                    ),
                    "capacity_matrix": capacity_matrix,
                    "transport": {
                        "global_assignments": (
                            global_assignments
                        ),
                        "global_fixed_rows": (
                            global_fixed_rows
                        ),
                        "global_padding_fraction": (
                            1.0
                            - global_assignments
                            / global_fixed_rows
                        ),
                    },
                    "correctness": correctness,
                    "timings": {
                        "pack_only": pack_timings,
                        "route_pack": route_timings,
                    },
                    "grouped_speedups": speedups,
                    "crossover": {
                        "pack_only": pack_crossover,
                        "route_pack": route_crossover,
                    },
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )
            del hidden, topk_ids, topk_weights
            torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": (
                "distributed_moe_profiled_packer_scaling"
            ),
            "profile": args.profile,
            "method_order": (
                "descending"
                if args.reverse_methods
                else "ascending"
            ),
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version(
                "triton"
            ),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "The profile is the held-out source-destination count matrix plus a fixed absolute margin, ensuring a controlled lossless packer study.",
                "Two-sort is the v28/v29 static baseline; one-pass atomic writes a complete hidden row in one Triton program; tiled atomic separates slot reservation from BLOCK_SIZE-wide scatter.",
                "One-pass atomic is intentionally limited by --one-pass-max-hidden because one program spans the complete row.",
                "Correctness sorts valid fixed rows by token/expert key and requires exact hidden, weight, token, and padding equality with zero overflow.",
                "Pack-only timing reuses router outputs; route-pack timing includes router projection and fused top-k. No collective or expert kernel is included.",
            ],
            "records": records,
        }
        output = Path(args.output)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temp = output.with_suffix(
            output.suffix + ".tmp"
        )
        temp.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
            + "\n",
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
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--padding-margin",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=[128, 256, 512],
        default=256,
    )
    parser.add_argument(
        "--one-pass-max-hidden",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--crossover-cycles",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--reverse-methods",
        action="store_true",
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
