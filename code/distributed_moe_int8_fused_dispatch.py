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
import triton
import triton.language as tl
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from distributed_moe_end_to_end import (
    TensorDict,
    critical_path_summary,
    gather_scalar,
    local_experts,
    output_quality,
    return_and_combine,
    route_and_pack,
    routing_bias,
    timed_schedule,
)
from distributed_moe_int8_dispatch import (
    dispatch_assignments_int8,
    quantize_hidden_per_token,
)
from distributed_moe_packed_dispatch import (
    build_metadata_packet,
    dispatch_assignments_metadata_packed,
    timed_crossover,
)


@triton.jit
def _quantize_pack_kernel(
    hidden_ptr,
    local_ids_ptr,
    weights_ptr,
    quantized_ptr,
    metadata_ptr,
    hidden_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    hidden = tl.load(
        hidden_ptr + row * hidden_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    max_abs = tl.max(
        tl.where(mask, tl.abs(hidden), 0.0),
        axis=0,
    )
    scale = tl.maximum(max_abs / 127.0, 1e-8)
    transmitted_scale = scale.to(tl.bfloat16)
    quant_scale = transmitted_scale.to(tl.float32)
    scaled = hidden / quant_scale
    rounded = tl.where(
        scaled >= 0.0,
        tl.floor(scaled + 0.5),
        tl.ceil(scaled - 0.5),
    )
    clipped = tl.maximum(-127.0, tl.minimum(127.0, rounded))
    tl.store(
        quantized_ptr + row * n_cols + offsets,
        clipped,
        mask=mask,
    )

    local_id = tl.load(local_ids_ptr + row).to(tl.float32)
    weight = tl.load(weights_ptr + row).to(tl.float32)
    tl.store(metadata_ptr + row * 3, local_id)
    tl.store(metadata_ptr + row * 3 + 1, weight)
    tl.store(metadata_ptr + row * 3 + 2, transmitted_scale)


@triton.jit
def _dequantize_unpack_kernel(
    quantized_ptr,
    metadata_ptr,
    hidden_ptr,
    local_ids_ptr,
    weights_ptr,
    scales_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    metadata_base = row * 3
    local_id = tl.load(metadata_ptr + metadata_base)
    weight = tl.load(metadata_ptr + metadata_base + 1)
    scale = tl.load(metadata_ptr + metadata_base + 2)
    quantized = tl.load(
        quantized_ptr + row * n_cols + offsets,
        mask=mask,
        other=0,
    ).to(tl.float32)
    tl.store(
        hidden_ptr + row * n_cols + offsets,
        quantized * scale.to(tl.float32),
        mask=mask,
    )
    tl.store(local_ids_ptr + row, local_id)
    tl.store(weights_ptr + row, weight)
    tl.store(scales_ptr + row, scale)


def quantize_pack_triton(
    route: TensorDict,
) -> tuple[torch.Tensor, torch.Tensor]:
    send_hidden = route["send_hidden"]
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_local_ids, torch.Tensor)
    assert isinstance(send_weights, torch.Tensor)
    rows, hidden = send_hidden.shape
    quantized = torch.empty(
        (rows, hidden),
        device=send_hidden.device,
        dtype=torch.int8,
    )
    metadata = torch.empty(
        (rows, 3),
        device=send_hidden.device,
        dtype=send_hidden.dtype,
    )
    if rows == 0:
        return quantized, metadata
    block_size = triton.next_power_of_2(hidden)
    _quantize_pack_kernel[(rows,)](
        send_hidden,
        send_local_ids,
        send_weights,
        quantized,
        metadata,
        send_hidden.stride(0),
        n_cols=hidden,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return quantized, metadata


def dequantize_unpack_triton(
    quantized: torch.Tensor,
    metadata: torch.Tensor,
    dtype: torch.dtype,
) -> TensorDict:
    rows, hidden = quantized.shape
    recv_hidden = torch.empty(
        (rows, hidden),
        device=quantized.device,
        dtype=dtype,
    )
    recv_local_ids = torch.empty(
        (rows, 1),
        device=quantized.device,
        dtype=torch.int32,
    )
    recv_weights = torch.empty(
        (rows, 1),
        device=quantized.device,
        dtype=dtype,
    )
    recv_scale = torch.empty(
        (rows, 1),
        device=quantized.device,
        dtype=dtype,
    )
    if rows == 0:
        return {
            "recv_hidden": recv_hidden,
            "recv_local_ids": recv_local_ids,
            "recv_weights": recv_weights,
            "recv_scale": recv_scale,
        }
    block_size = triton.next_power_of_2(hidden)
    _dequantize_unpack_kernel[(rows,)](
        quantized,
        metadata,
        recv_hidden,
        recv_local_ids,
        recv_weights,
        recv_scale,
        n_cols=hidden,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return {
        "recv_hidden": recv_hidden,
        "recv_local_ids": recv_local_ids,
        "recv_weights": recv_weights,
        "recv_scale": recv_scale,
    }


def unpack_dequantize_torch(
    quantized: torch.Tensor,
    metadata: torch.Tensor,
    dtype: torch.dtype,
) -> TensorDict:
    return {
        "recv_hidden": (
            quantized.float()
            * metadata[:, 2:3].float()
        ).to(dtype).contiguous(),
        "recv_local_ids": (
            metadata[:, :1].to(torch.int32).contiguous()
        ),
        "recv_weights": metadata[:, 1:2].contiguous(),
        "recv_scale": metadata[:, 2:3].contiguous(),
    }


def dispatch_assignments_int8_triton(
    route: TensorDict,
    *,
    quantized: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
) -> TensorDict:
    send_hidden = route["send_hidden"]
    send_counts_tensor = route["send_counts_tensor"]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts_tensor, torch.Tensor)
    assert isinstance(send_counts, list)
    if quantized is None or metadata is None:
        quantized, metadata = quantize_pack_triton(route)

    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    recv_counts = recv_counts_tensor.cpu().tolist()
    recv_total = sum(recv_counts)
    recv_quantized = torch.empty(
        (recv_total, send_hidden.shape[1]),
        device=send_hidden.device,
        dtype=torch.int8,
    )
    recv_metadata = torch.empty(
        (recv_total, 3),
        device=send_hidden.device,
        dtype=send_hidden.dtype,
    )
    dist.all_to_all_single(
        recv_quantized,
        quantized,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    dist.all_to_all_single(
        recv_metadata,
        metadata,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    unpacked = dequantize_unpack_triton(
        recv_quantized,
        recv_metadata,
        send_hidden.dtype,
    )
    unpacked["recv_counts"] = recv_counts
    unpacked["recv_quantized"] = recv_quantized
    return unpacked


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
    elif args.profile == "crossover":
        token_values = [2048]
        routes = (
            ("balanced", 0.0),
            ("global_bias1", 1.0),
            ("global_bias4", 4.0),
        )
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
        input_generator.manual_seed(30_000 + rank * 101 + num_tokens)
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
            prepared_bf16_metadata = build_metadata_packet(
                prepared_route
            )
            prepared_torch_quantized, prepared_torch_metadata = (
                quantize_hidden_per_token(prepared_route)
            )
            prepared_triton_quantized, prepared_triton_metadata = (
                quantize_pack_triton(prepared_route)
            )
            bf16_dispatch = dispatch_assignments_metadata_packed(
                prepared_route,
                metadata_packet=prepared_bf16_metadata,
            )
            torch_dispatch = dispatch_assignments_int8(
                prepared_route,
                quantized=prepared_torch_quantized,
                metadata=prepared_torch_metadata,
            )
            triton_dispatch = dispatch_assignments_int8_triton(
                prepared_route,
                quantized=prepared_triton_quantized,
                metadata=prepared_triton_metadata,
            )
            bf16_expert_output = local_experts(
                bf16_dispatch,
                local_w1,
                local_w2,
            )
            torch_expert_output = local_experts(
                torch_dispatch,
                local_w1,
                local_w2,
            )
            triton_expert_output = local_experts(
                triton_dispatch,
                local_w1,
                local_w2,
            )
            bf16_output = return_and_combine(
                bf16_expert_output,
                prepared_route,
                bf16_dispatch,
                num_tokens=num_tokens,
            )
            torch_output = return_and_combine(
                torch_expert_output,
                prepared_route,
                torch_dispatch,
                num_tokens=num_tokens,
            )
            triton_output = return_and_combine(
                triton_expert_output,
                prepared_route,
                triton_dispatch,
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

            torch_hidden_quality = output_quality(
                torch_dispatch["recv_hidden"],
                bf16_dispatch["recv_hidden"],
            )
            triton_hidden_quality = output_quality(
                triton_dispatch["recv_hidden"],
                bf16_dispatch["recv_hidden"],
            )
            triton_vs_torch_hidden_quality = output_quality(
                triton_dispatch["recv_hidden"],
                torch_dispatch["recv_hidden"],
            )
            torch_output_quality = output_quality(
                torch_output,
                bf16_output,
            )
            triton_output_quality = output_quality(
                triton_output,
                bf16_output,
            )
            triton_vs_torch_output_quality = output_quality(
                triton_output,
                torch_output,
            )
            triton_reference_quality = output_quality(
                triton_output,
                reference_output,
            )
            torch_ids_equal = torch.equal(
                torch_dispatch["recv_local_ids"],
                bf16_dispatch["recv_local_ids"],
            )
            triton_ids_equal = torch.equal(
                triton_dispatch["recv_local_ids"],
                bf16_dispatch["recv_local_ids"],
            )
            triton_weight_quality = output_quality(
                triton_dispatch["recv_weights"],
                bf16_dispatch["recv_weights"],
            )
            if (
                not torch_ids_equal
                or not triton_ids_equal
                or triton_weight_quality["max_abs_error"] != 0.0
                or triton_hidden_quality["normalized_rmse"]
                > args.max_hidden_nrmse
                or triton_hidden_quality["cosine_similarity"]
                < args.min_hidden_cosine
                or triton_output_quality["normalized_rmse"]
                > args.max_output_nrmse
                or triton_output_quality["cosine_similarity"]
                < args.min_output_cosine
            ):
                raise RuntimeError(
                    f"Fused INT8 quality failed for "
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

            def torch_quantize_pack_only() -> Any:
                return quantize_hidden_per_token(prepared_route)

            def triton_quantize_pack_only() -> Any:
                return quantize_pack_triton(prepared_route)

            def torch_dequantize_unpack_only() -> Any:
                return unpack_dequantize_torch(
                    prepared_torch_quantized,
                    prepared_torch_metadata,
                    dtype,
                )

            def triton_dequantize_unpack_only() -> Any:
                return dequantize_unpack_triton(
                    prepared_triton_quantized,
                    prepared_triton_metadata,
                    dtype,
                )

            def bf16_dispatch_with_pack() -> TensorDict:
                return dispatch_assignments_metadata_packed(
                    prepared_route
                )

            def torch_dispatch_with_quant() -> TensorDict:
                return dispatch_assignments_int8(prepared_route)

            def triton_dispatch_with_quant() -> TensorDict:
                return dispatch_assignments_int8_triton(
                    prepared_route
                )

            def bf16_end_to_end() -> torch.Tensor:
                route = build_route()
                dispatched = dispatch_assignments_metadata_packed(
                    route
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

            def torch_end_to_end() -> torch.Tensor:
                route = build_route()
                dispatched = dispatch_assignments_int8(route)
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

            def triton_end_to_end() -> torch.Tensor:
                route = build_route()
                dispatched = dispatch_assignments_int8_triton(
                    route
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

            schedules: tuple[
                tuple[str, Callable[[], Any]],
                ...,
            ] = (
                (
                    "torch_quantize_pack_only",
                    torch_quantize_pack_only,
                ),
                (
                    "triton_quantize_pack_only",
                    triton_quantize_pack_only,
                ),
                (
                    "torch_dequantize_unpack_only",
                    torch_dequantize_unpack_only,
                ),
                (
                    "triton_dequantize_unpack_only",
                    triton_dequantize_unpack_only,
                ),
                ("bf16_dispatch", bf16_dispatch_with_pack),
                ("torch_int8_dispatch", torch_dispatch_with_quant),
                (
                    "triton_int8_dispatch",
                    triton_dispatch_with_quant,
                ),
                ("bf16_end_to_end", bf16_end_to_end),
                ("torch_int8_end_to_end", torch_end_to_end),
                (
                    "triton_int8_end_to_end",
                    triton_end_to_end,
                ),
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
                crossover_dispatch = timed_crossover(
                    (
                        ("bf16", bf16_dispatch_with_pack),
                        ("torch_int8", torch_dispatch_with_quant),
                        (
                            "triton_int8",
                            triton_dispatch_with_quant,
                        ),
                    ),
                    warmup_cycles=6,
                    measured_cycles=30,
                    local_rank=local_rank,
                    device=device,
                )
                crossover_end_to_end = timed_crossover(
                    (
                        ("bf16", bf16_end_to_end),
                        ("torch_int8", torch_end_to_end),
                        ("triton_int8", triton_end_to_end),
                    ),
                    warmup_cycles=6,
                    measured_cycles=30,
                    local_rank=local_rank,
                    device=device,
                )
                crossover = {
                    "method": (
                        "Latin-square rotation over BF16, "
                        "PyTorch INT8, and Triton INT8"
                    ),
                    "warmup_cycles": 6,
                    "measured_cycles": 30,
                    "dispatch": crossover_dispatch,
                    "end_to_end": crossover_end_to_end,
                    "dispatch_speedups_vs_bf16": {
                        "torch_int8": (
                            crossover_dispatch["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_dispatch["torch_int8"][
                                "critical_path_ms"
                            ]
                        ),
                        "triton_int8": (
                            crossover_dispatch["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_dispatch["triton_int8"][
                                "critical_path_ms"
                            ]
                        ),
                    },
                    "end_to_end_speedups_vs_bf16": {
                        "torch_int8": (
                            crossover_end_to_end["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_end_to_end["torch_int8"][
                                "critical_path_ms"
                            ]
                        ),
                        "triton_int8": (
                            crossover_end_to_end["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_end_to_end["triton_int8"][
                                "critical_path_ms"
                            ]
                        ),
                    },
                }

            bf16_dispatch_ms = timings["bf16_dispatch"][
                "critical_path_ms"
            ]
            torch_dispatch_ms = timings["torch_int8_dispatch"][
                "critical_path_ms"
            ]
            triton_dispatch_ms = timings["triton_int8_dispatch"][
                "critical_path_ms"
            ]
            bf16_end_to_end_ms = timings["bf16_end_to_end"][
                "critical_path_ms"
            ]
            torch_end_to_end_ms = timings[
                "torch_int8_end_to_end"
            ]["critical_path_ms"]
            triton_end_to_end_ms = timings[
                "triton_int8_end_to_end"
            ]["critical_path_ms"]
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
                        "byte_reduction_fraction": (
                            1
                            - (args.hidden + 6)
                            / (args.hidden * 2 + 4)
                        ),
                        "quantization": (
                            "symmetric per-token INT8 with BF16 scale"
                        ),
                        "return_path": "BF16",
                    },
                    "quality": {
                        "torch_ids_equal": torch_ids_equal,
                        "triton_ids_equal": triton_ids_equal,
                        "max_triton_weight_error": max(
                            gather_scalar(
                                triton_weight_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_torch_hidden_nrmse": max(
                            gather_scalar(
                                torch_hidden_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_triton_hidden_nrmse": max(
                            gather_scalar(
                                triton_hidden_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_triton_hidden_cosine": min(
                            gather_scalar(
                                triton_hidden_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                        "max_triton_vs_torch_hidden_nrmse": max(
                            gather_scalar(
                                triton_vs_torch_hidden_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_torch_output_nrmse": max(
                            gather_scalar(
                                torch_output_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_triton_output_nrmse": max(
                            gather_scalar(
                                triton_output_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_triton_output_cosine": min(
                            gather_scalar(
                                triton_output_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                        "max_triton_vs_torch_output_nrmse": max(
                            gather_scalar(
                                triton_vs_torch_output_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_triton_reference_nrmse": max(
                            gather_scalar(
                                triton_reference_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                    },
                    "timings": timings,
                    "crossover": crossover,
                    "torch_dispatch_speedup_vs_bf16": (
                        bf16_dispatch_ms / torch_dispatch_ms
                    ),
                    "triton_dispatch_speedup_vs_bf16": (
                        bf16_dispatch_ms / triton_dispatch_ms
                    ),
                    "triton_dispatch_speedup_vs_torch": (
                        torch_dispatch_ms / triton_dispatch_ms
                    ),
                    "torch_end_to_end_speedup_vs_bf16": (
                        bf16_end_to_end_ms / torch_end_to_end_ms
                    ),
                    "triton_end_to_end_speedup_vs_bf16": (
                        bf16_end_to_end_ms / triton_end_to_end_ms
                    ),
                    "triton_end_to_end_speedup_vs_torch": (
                        torch_end_to_end_ms / triton_end_to_end_ms
                    ),
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )

            del (
                prepared_route,
                prepared_bf16_metadata,
                prepared_torch_quantized,
                prepared_torch_metadata,
                prepared_triton_quantized,
                prepared_triton_metadata,
                bf16_dispatch,
                torch_dispatch,
                triton_dispatch,
                bf16_expert_output,
                torch_expert_output,
                triton_expert_output,
                bf16_output,
                torch_output,
                triton_output,
                reference_output,
            )
            torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_int8_fused_dispatch",
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
                "The Triton quantization kernel fuses per-token absolute-max reduction, BF16 scale rounding, INT8 quantization, and expert-ID/router-weight/scale metadata packing.",
                "The Triton receive kernel fuses INT8 dequantization with expert-ID, router-weight, and scale unpacking into contiguous tensors for the unchanged fused expert kernel.",
                "BF16 metadata-packed, eager PyTorch INT8, and fused Triton INT8 paths share routing, collectives, expert weights, BF16 return communication, and weighted combine.",
                "Crossovers rotate all three paths through timing positions to separate low-single-digit conversion gains from schedule-order effects.",
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
    parser.add_argument("--max-hidden-nrmse", type=float, default=0.02)
    parser.add_argument("--min-hidden-cosine", type=float, default=0.999)
    parser.add_argument("--max-output-nrmse", type=float, default=0.05)
    parser.add_argument("--min-output-cosine", type=float, default=0.998)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
