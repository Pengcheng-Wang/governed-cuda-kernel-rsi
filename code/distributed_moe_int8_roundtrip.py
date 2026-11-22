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
from distributed_moe_int8_fused_dispatch import (
    dispatch_assignments_int8_triton,
    quantize_pack_triton,
)
from distributed_moe_packed_dispatch import (
    build_metadata_packet,
    dispatch_assignments_metadata_packed,
    timed_crossover,
)


@triton.jit
def _quantize_scale_kernel(
    hidden_ptr,
    quantized_ptr,
    scales_ptr,
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
    scaled = hidden / transmitted_scale.to(tl.float32)
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
    tl.store(scales_ptr + row, transmitted_scale)


@triton.jit
def _dequantize_scale_kernel(
    quantized_ptr,
    scales_ptr,
    hidden_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    scale = tl.load(scales_ptr + row).to(tl.float32)
    quantized = tl.load(
        quantized_ptr + row * n_cols + offsets,
        mask=mask,
        other=0,
    ).to(tl.float32)
    tl.store(
        hidden_ptr + row * n_cols + offsets,
        quantized * scale,
        mask=mask,
    )


@triton.jit
def _quantize_packet_kernel(
    hidden_ptr,
    packet_ptr,
    quantized_ptr,
    scales_ptr,
    hidden_stride,
    packet_stride,
    quantized_stride,
    scales_stride,
    n_cols: tl.constexpr,
    PACKET_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
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
    scaled = hidden / transmitted_scale.to(tl.float32)
    rounded = tl.where(
        scaled >= 0.0,
        tl.floor(scaled + 0.5),
        tl.ceil(scaled - 0.5),
    )
    clipped = tl.maximum(-127.0, tl.minimum(127.0, rounded))
    tl.store(
        quantized_ptr + row * quantized_stride + offsets,
        clipped,
        mask=mask,
    )
    tl.store(
        scales_ptr + row * scales_stride,
        transmitted_scale,
    )
    padding_offsets = tl.arange(0, PAD_BLOCK)
    padding_size = PACKET_SIZE - n_cols - 2
    tl.store(
        packet_ptr
        + row * packet_stride
        + n_cols
        + 2
        + padding_offsets,
        0,
        mask=padding_offsets < padding_size,
    )


@triton.jit
def _dequantize_packet_kernel(
    quantized_ptr,
    scales_ptr,
    hidden_ptr,
    quantized_stride,
    scales_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    scale = tl.load(
        scales_ptr + row * scales_stride,
    ).to(tl.float32)
    quantized = tl.load(
        quantized_ptr + row * quantized_stride + offsets,
        mask=mask,
        other=0,
    ).to(tl.float32)
    tl.store(
        hidden_ptr + row * n_cols + offsets,
        quantized * scale,
        mask=mask,
    )


def quantize_scale_triton(
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = hidden.shape
    quantized = torch.empty(
        (rows, width),
        device=hidden.device,
        dtype=torch.int8,
    )
    scales = torch.empty(
        (rows, 1),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    if rows == 0:
        return quantized, scales
    block_size = triton.next_power_of_2(width)
    _quantize_scale_kernel[(rows,)](
        hidden,
        quantized,
        scales,
        hidden.stride(0),
        n_cols=width,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return quantized, scales


def dequantize_scale_triton(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows, width = quantized.shape
    hidden = torch.empty(
        (rows, width),
        device=quantized.device,
        dtype=dtype,
    )
    if rows == 0:
        return hidden
    block_size = triton.next_power_of_2(width)
    _dequantize_scale_kernel[(rows,)](
        quantized,
        scales,
        hidden,
        n_cols=width,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return hidden


def return_packet_views(
    packet: torch.Tensor,
    width: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if packet.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 packet, got {packet.dtype}")
    if packet.ndim != 2 or packet.shape[1] < width + 2:
        raise ValueError(
            f"Expected packet shape [rows, >= {width + 2}], "
            f"got {tuple(packet.shape)}"
        )
    quantized = packet[:, :width].view(torch.int8)
    scales = packet[:, width : width + 2].view(dtype)
    return quantized, scales


def aligned_packet_size(width: int, alignment: int) -> int:
    if alignment <= 0 or alignment % 2:
        raise ValueError(
            f"Packet alignment must be a positive even value, got {alignment}"
        )
    raw_size = width + 2
    return ((raw_size + alignment - 1) // alignment) * alignment


def quantize_packet_triton(
    hidden: torch.Tensor,
    *,
    alignment: int = 2,
) -> torch.Tensor:
    rows, width = hidden.shape
    packet_size = aligned_packet_size(width, alignment)
    packet = torch.empty(
        (rows, packet_size),
        device=hidden.device,
        dtype=torch.uint8,
    )
    if rows == 0:
        return packet
    quantized, scales = return_packet_views(
        packet,
        width,
        hidden.dtype,
    )
    block_size = triton.next_power_of_2(width)
    pad_block = triton.next_power_of_2(
        max(1, packet_size - width - 2)
    )
    _quantize_packet_kernel[(rows,)](
        hidden,
        packet,
        quantized,
        scales,
        hidden.stride(0),
        packet.stride(0),
        quantized.stride(0),
        scales.stride(0),
        n_cols=width,
        PACKET_SIZE=packet_size,
        BLOCK_SIZE=block_size,
        PAD_BLOCK=pad_block,
        num_warps=8,
    )
    return packet


def dequantize_packet_triton(
    packet: torch.Tensor,
    dtype: torch.dtype,
    width: int,
) -> torch.Tensor:
    rows = packet.shape[0]
    hidden = torch.empty(
        (rows, width),
        device=packet.device,
        dtype=dtype,
    )
    if rows == 0:
        return hidden
    quantized, scales = return_packet_views(
        packet,
        width,
        dtype,
    )
    block_size = triton.next_power_of_2(width)
    _dequantize_packet_kernel[(rows,)](
        quantized,
        scales,
        hidden,
        quantized.stride(0),
        scales.stride(0),
        n_cols=width,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return hidden


def combine_returned(
    returned: torch.Tensor,
    route: TensorDict,
    *,
    num_tokens: int,
) -> torch.Tensor:
    sorted_token_indices = route["sorted_token_indices"]
    assert isinstance(sorted_token_indices, torch.Tensor)
    combined = torch.zeros(
        (num_tokens, returned.shape[1]),
        device=returned.device,
        dtype=torch.float32,
    )
    combined.index_add_(
        0,
        sorted_token_indices,
        returned.float(),
    )
    return combined.to(returned.dtype)


def return_and_combine_int8_triton(
    expert_output: torch.Tensor,
    route: TensorDict,
    dispatched: TensorDict,
    *,
    num_tokens: int,
    quantized: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    send_hidden = route["send_hidden"]
    send_counts = route["send_counts"]
    recv_counts = dispatched["recv_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts, list)
    assert isinstance(recv_counts, list)
    if quantized is None or scales is None:
        quantized, scales = quantize_scale_triton(expert_output)

    returned_quantized = torch.empty(
        send_hidden.shape,
        device=expert_output.device,
        dtype=torch.int8,
    )
    returned_scales = torch.empty(
        (send_hidden.shape[0], 1),
        device=expert_output.device,
        dtype=expert_output.dtype,
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
    returned = dequantize_scale_triton(
        returned_quantized,
        returned_scales,
        expert_output.dtype,
    )
    return combine_returned(
        returned,
        route,
        num_tokens=num_tokens,
    )


def return_and_combine_int8_packed_triton(
    expert_output: torch.Tensor,
    route: TensorDict,
    dispatched: TensorDict,
    *,
    num_tokens: int,
    packet: torch.Tensor | None = None,
    packet_alignment: int = 2,
) -> torch.Tensor:
    send_hidden = route["send_hidden"]
    send_counts = route["send_counts"]
    recv_counts = dispatched["recv_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts, list)
    assert isinstance(recv_counts, list)
    if packet is None:
        packet = quantize_packet_triton(
            expert_output,
            alignment=packet_alignment,
        )

    returned_packet = torch.empty(
        (send_hidden.shape[0], packet.shape[1]),
        device=expert_output.device,
        dtype=torch.uint8,
    )
    dist.all_to_all_single(
        returned_packet,
        packet,
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts,
    )
    returned = dequantize_packet_triton(
        returned_packet,
        expert_output.dtype,
        expert_output.shape[1],
    )
    return combine_returned(
        returned,
        route,
        num_tokens=num_tokens,
    )


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
    elif args.profile in ("crossover", "alignment"):
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
        input_generator.manual_seed(40_000 + rank * 101 + num_tokens)
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
            prepared_forward_quantized, prepared_forward_metadata = (
                quantize_pack_triton(prepared_route)
            )
            bf16_dispatch = dispatch_assignments_metadata_packed(
                prepared_route,
                metadata_packet=prepared_bf16_metadata,
            )
            int8_dispatch = dispatch_assignments_int8_triton(
                prepared_route,
                quantized=prepared_forward_quantized,
                metadata=prepared_forward_metadata,
            )
            bf16_expert_output = local_experts(
                bf16_dispatch,
                local_w1,
                local_w2,
            )
            int8_expert_output = local_experts(
                int8_dispatch,
                local_w1,
                local_w2,
            )
            prepared_return_quantized, prepared_return_scales = (
                quantize_scale_triton(int8_expert_output)
            )
            prepared_return_packet = quantize_packet_triton(
                int8_expert_output
            )
            prepared_aligned_return_packet = quantize_packet_triton(
                int8_expert_output,
                alignment=16,
            )
            bf16_output = return_and_combine(
                bf16_expert_output,
                prepared_route,
                bf16_dispatch,
                num_tokens=num_tokens,
            )
            forward_only_output = return_and_combine(
                int8_expert_output,
                prepared_route,
                int8_dispatch,
                num_tokens=num_tokens,
            )
            roundtrip_output = return_and_combine_int8_triton(
                int8_expert_output,
                prepared_route,
                int8_dispatch,
                num_tokens=num_tokens,
                quantized=prepared_return_quantized,
                scales=prepared_return_scales,
            )
            packed_roundtrip_output = (
                return_and_combine_int8_packed_triton(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                    packet=prepared_return_packet,
                )
            )
            aligned_roundtrip_output = (
                return_and_combine_int8_packed_triton(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                    packet=prepared_aligned_return_packet,
                    packet_alignment=16,
                )
            )
            isolated_return_output = (
                return_and_combine_int8_triton(
                    bf16_expert_output,
                    prepared_route,
                    bf16_dispatch,
                    num_tokens=num_tokens,
                )
            )
            isolated_packed_return_output = (
                return_and_combine_int8_packed_triton(
                    bf16_expert_output,
                    prepared_route,
                    bf16_dispatch,
                    num_tokens=num_tokens,
                )
            )
            isolated_aligned_return_output = (
                return_and_combine_int8_packed_triton(
                    bf16_expert_output,
                    prepared_route,
                    bf16_dispatch,
                    num_tokens=num_tokens,
                    packet_alignment=16,
                )
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

            forward_quality = output_quality(
                forward_only_output,
                bf16_output,
            )
            roundtrip_quality = output_quality(
                roundtrip_output,
                bf16_output,
            )
            return_only_quality = output_quality(
                isolated_return_output,
                bf16_output,
            )
            packed_return_only_quality = output_quality(
                isolated_packed_return_output,
                bf16_output,
            )
            aligned_return_only_quality = output_quality(
                isolated_aligned_return_output,
                bf16_output,
            )
            roundtrip_vs_forward_quality = output_quality(
                roundtrip_output,
                forward_only_output,
            )
            packed_roundtrip_quality = output_quality(
                packed_roundtrip_output,
                bf16_output,
            )
            packed_vs_two_collective_quality = output_quality(
                packed_roundtrip_output,
                roundtrip_output,
            )
            aligned_roundtrip_quality = output_quality(
                aligned_roundtrip_output,
                bf16_output,
            )
            aligned_vs_two_collective_quality = output_quality(
                aligned_roundtrip_output,
                roundtrip_output,
            )
            reference_quality = output_quality(
                roundtrip_output,
                reference_output,
            )
            packed_reference_quality = output_quality(
                packed_roundtrip_output,
                reference_output,
            )
            aligned_reference_quality = output_quality(
                aligned_roundtrip_output,
                reference_output,
            )
            if any(
                quality["normalized_rmse"] > args.max_output_nrmse
                or quality["cosine_similarity"]
                < args.min_output_cosine
                for quality in (
                    roundtrip_quality,
                    packed_roundtrip_quality,
                    aligned_roundtrip_quality,
                )
            ):
                raise RuntimeError(
                    f"INT8 roundtrip quality failed for "
                    f"T={num_tokens}, route={route_name}"
                )
            if packed_vs_two_collective_quality["max_abs_error"] != 0.0:
                raise RuntimeError(
                    f"Packed return disagrees with two-collective "
                    f"return for T={num_tokens}, route={route_name}: "
                    f"{packed_vs_two_collective_quality}"
                )
            if aligned_vs_two_collective_quality["max_abs_error"] != 0.0:
                raise RuntimeError(
                    f"Aligned return disagrees with two-collective "
                    f"return for T={num_tokens}, route={route_name}: "
                    f"{aligned_vs_two_collective_quality}"
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

            def return_quantize_only() -> Any:
                return quantize_scale_triton(int8_expert_output)

            def packed_return_quantize_only() -> torch.Tensor:
                return quantize_packet_triton(int8_expert_output)

            def aligned_return_quantize_only() -> torch.Tensor:
                return quantize_packet_triton(
                    int8_expert_output,
                    alignment=16,
                )

            def return_dequantize_only() -> torch.Tensor:
                return dequantize_scale_triton(
                    prepared_return_quantized,
                    prepared_return_scales,
                    dtype,
                )

            def packed_return_dequantize_only() -> torch.Tensor:
                return dequantize_packet_triton(
                    prepared_return_packet,
                    dtype,
                    args.hidden,
                )

            def aligned_return_dequantize_only() -> torch.Tensor:
                return dequantize_packet_triton(
                    prepared_aligned_return_packet,
                    dtype,
                    args.hidden,
                )

            def bf16_return_only() -> torch.Tensor:
                return return_and_combine(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                )

            def int8_return_prebuilt() -> torch.Tensor:
                return return_and_combine_int8_triton(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                    quantized=prepared_return_quantized,
                    scales=prepared_return_scales,
                )

            def int8_return_with_quant() -> torch.Tensor:
                return return_and_combine_int8_triton(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                )

            def packed_return_prebuilt() -> torch.Tensor:
                return return_and_combine_int8_packed_triton(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                    packet=prepared_return_packet,
                )

            def packed_return_with_quant() -> torch.Tensor:
                return return_and_combine_int8_packed_triton(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                )

            def aligned_return_prebuilt() -> torch.Tensor:
                return return_and_combine_int8_packed_triton(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                    packet=prepared_aligned_return_packet,
                    packet_alignment=16,
                )

            def aligned_return_with_quant() -> torch.Tensor:
                return return_and_combine_int8_packed_triton(
                    int8_expert_output,
                    prepared_route,
                    int8_dispatch,
                    num_tokens=num_tokens,
                    packet_alignment=16,
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

            def forward_only_end_to_end() -> torch.Tensor:
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

            def roundtrip_end_to_end() -> torch.Tensor:
                route = build_route()
                dispatched = dispatch_assignments_int8_triton(
                    route
                )
                expert_output = local_experts(
                    dispatched,
                    local_w1,
                    local_w2,
                )
                return return_and_combine_int8_triton(
                    expert_output,
                    route,
                    dispatched,
                    num_tokens=num_tokens,
                )

            def packed_roundtrip_end_to_end() -> torch.Tensor:
                route = build_route()
                dispatched = dispatch_assignments_int8_triton(
                    route
                )
                expert_output = local_experts(
                    dispatched,
                    local_w1,
                    local_w2,
                )
                return return_and_combine_int8_packed_triton(
                    expert_output,
                    route,
                    dispatched,
                    num_tokens=num_tokens,
                )

            def aligned_roundtrip_end_to_end() -> torch.Tensor:
                route = build_route()
                dispatched = dispatch_assignments_int8_triton(
                    route
                )
                expert_output = local_experts(
                    dispatched,
                    local_w1,
                    local_w2,
                )
                return return_and_combine_int8_packed_triton(
                    expert_output,
                    route,
                    dispatched,
                    num_tokens=num_tokens,
                    packet_alignment=16,
                )

            schedules: tuple[
                tuple[str, Callable[[], Any]],
                ...,
            ] = (
                ("return_quantize_only", return_quantize_only),
                (
                    "packed_return_quantize_only",
                    packed_return_quantize_only,
                ),
                (
                    "aligned_return_quantize_only",
                    aligned_return_quantize_only,
                ),
                (
                    "return_dequantize_only",
                    return_dequantize_only,
                ),
                (
                    "packed_return_dequantize_only",
                    packed_return_dequantize_only,
                ),
                (
                    "aligned_return_dequantize_only",
                    aligned_return_dequantize_only,
                ),
                ("bf16_return_only", bf16_return_only),
                ("int8_return_prebuilt", int8_return_prebuilt),
                (
                    "int8_return_with_quant",
                    int8_return_with_quant,
                ),
                (
                    "packed_return_prebuilt",
                    packed_return_prebuilt,
                ),
                (
                    "packed_return_with_quant",
                    packed_return_with_quant,
                ),
                (
                    "aligned_return_prebuilt",
                    aligned_return_prebuilt,
                ),
                (
                    "aligned_return_with_quant",
                    aligned_return_with_quant,
                ),
                ("bf16_end_to_end", bf16_end_to_end),
                (
                    "forward_only_end_to_end",
                    forward_only_end_to_end,
                ),
                (
                    "roundtrip_end_to_end",
                    roundtrip_end_to_end,
                ),
                (
                    "packed_roundtrip_end_to_end",
                    packed_roundtrip_end_to_end,
                ),
                (
                    "aligned_roundtrip_end_to_end",
                    aligned_roundtrip_end_to_end,
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
            if args.profile in ("crossover", "alignment"):
                crossover_return = timed_crossover(
                    (
                        ("bf16", bf16_return_only),
                        (
                            "two_collective",
                            int8_return_with_quant,
                        ),
                        ("packed", packed_return_with_quant),
                        ("aligned", aligned_return_with_quant),
                    ),
                    warmup_cycles=6,
                    measured_cycles=30,
                    local_rank=local_rank,
                    device=device,
                )
                crossover_end_to_end = timed_crossover(
                    (
                        ("bf16", bf16_end_to_end),
                        (
                            "forward_only",
                            forward_only_end_to_end,
                        ),
                        (
                            "roundtrip_two_collective",
                            roundtrip_end_to_end,
                        ),
                        (
                            "roundtrip_packed",
                            packed_roundtrip_end_to_end,
                        ),
                        (
                            "roundtrip_aligned",
                            aligned_roundtrip_end_to_end,
                        ),
                    ),
                    warmup_cycles=6,
                    measured_cycles=30,
                    local_rank=local_rank,
                    device=device,
                )
                crossover = {
                    "method": (
                        "Position-balanced rotation over BF16, "
                        "two-collective INT8, and packed INT8 "
                        "return paths; end-to-end timing also "
                        "includes forward-only INT8"
                    ),
                    "warmup_cycles": 6,
                    "measured_cycles": 30,
                    "return": crossover_return,
                    "end_to_end": crossover_end_to_end,
                    "return_speedups_vs_bf16": {
                        "two_collective": (
                            crossover_return["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_return[
                                "two_collective"
                            ]["critical_path_ms"]
                        ),
                        "packed": (
                            crossover_return["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_return["packed"][
                                "critical_path_ms"
                            ]
                        ),
                        "aligned": (
                            crossover_return["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_return["aligned"][
                                "critical_path_ms"
                            ]
                        ),
                    },
                    "end_to_end_speedups_vs_bf16": {
                        "forward_only": (
                            crossover_end_to_end["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_end_to_end[
                                "forward_only"
                            ]["critical_path_ms"]
                        ),
                        "roundtrip_two_collective": (
                            crossover_end_to_end["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_end_to_end[
                                "roundtrip_two_collective"
                            ]["critical_path_ms"]
                        ),
                        "roundtrip_packed": (
                            crossover_end_to_end["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_end_to_end[
                                "roundtrip_packed"
                            ]["critical_path_ms"]
                        ),
                        "roundtrip_aligned": (
                            crossover_end_to_end["bf16"][
                                "critical_path_ms"
                            ]
                            / crossover_end_to_end[
                                "roundtrip_aligned"
                            ]["critical_path_ms"]
                        ),
                    },
                }

            bf16_return_ms = timings["bf16_return_only"][
                "critical_path_ms"
            ]
            int8_return_prebuilt_ms = timings[
                "int8_return_prebuilt"
            ]["critical_path_ms"]
            int8_return_with_quant_ms = timings[
                "int8_return_with_quant"
            ]["critical_path_ms"]
            packed_return_prebuilt_ms = timings[
                "packed_return_prebuilt"
            ]["critical_path_ms"]
            packed_return_with_quant_ms = timings[
                "packed_return_with_quant"
            ]["critical_path_ms"]
            aligned_return_prebuilt_ms = timings[
                "aligned_return_prebuilt"
            ]["critical_path_ms"]
            aligned_return_with_quant_ms = timings[
                "aligned_return_with_quant"
            ]["critical_path_ms"]
            bf16_end_to_end_ms = timings["bf16_end_to_end"][
                "critical_path_ms"
            ]
            forward_only_end_to_end_ms = timings[
                "forward_only_end_to_end"
            ]["critical_path_ms"]
            roundtrip_end_to_end_ms = timings[
                "roundtrip_end_to_end"
            ]["critical_path_ms"]
            packed_roundtrip_end_to_end_ms = timings[
                "packed_roundtrip_end_to_end"
            ]["critical_path_ms"]
            aligned_roundtrip_end_to_end_ms = timings[
                "aligned_roundtrip_end_to_end"
            ]["critical_path_ms"]
            bf16_return_bytes = args.hidden * 2
            int8_return_bytes = args.hidden + 2
            aligned_int8_return_bytes = aligned_packet_size(
                args.hidden,
                16,
            )
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
                        "forward_byte_reduction_fraction": (
                            1
                            - (args.hidden + 6)
                            / (args.hidden * 2 + 4)
                        ),
                        "return_byte_reduction_fraction": (
                            1
                            - int8_return_bytes / bf16_return_bytes
                        ),
                        "bf16_return_collectives": 1,
                        "int8_return_collectives": 2,
                        "packed_int8_return_collectives": 1,
                        "packed_packet_dtype": "torch.uint8",
                        "packed_packet_bytes_per_assignment": (
                            int8_return_bytes
                        ),
                        "aligned_int8_return_collectives": 1,
                        "aligned_packet_bytes_per_assignment": (
                            aligned_int8_return_bytes
                        ),
                        "aligned_return_byte_reduction_fraction": (
                            1
                            - aligned_int8_return_bytes
                            / bf16_return_bytes
                        ),
                        "aligned_packet_alignment_bytes": 16,
                    },
                    "quality": {
                        "max_forward_output_nrmse": max(
                            gather_scalar(
                                forward_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_return_only_output_nrmse": max(
                            gather_scalar(
                                return_only_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_packed_return_only_output_nrmse": max(
                            gather_scalar(
                                packed_return_only_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_aligned_return_only_output_nrmse": max(
                            gather_scalar(
                                aligned_return_only_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_roundtrip_output_nrmse": max(
                            gather_scalar(
                                roundtrip_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_roundtrip_output_cosine": min(
                            gather_scalar(
                                roundtrip_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                        "max_packed_roundtrip_output_nrmse": max(
                            gather_scalar(
                                packed_roundtrip_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_packed_roundtrip_output_cosine": min(
                            gather_scalar(
                                packed_roundtrip_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                        "max_aligned_roundtrip_output_nrmse": max(
                            gather_scalar(
                                aligned_roundtrip_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_aligned_roundtrip_output_cosine": min(
                            gather_scalar(
                                aligned_roundtrip_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                        "max_packed_vs_two_collective_error": max(
                            gather_scalar(
                                packed_vs_two_collective_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_aligned_vs_two_collective_error": max(
                            gather_scalar(
                                aligned_vs_two_collective_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_roundtrip_vs_forward_nrmse": max(
                            gather_scalar(
                                roundtrip_vs_forward_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_roundtrip_reference_nrmse": max(
                            gather_scalar(
                                reference_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_packed_reference_nrmse": max(
                            gather_scalar(
                                packed_reference_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_aligned_reference_nrmse": max(
                            gather_scalar(
                                aligned_reference_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                    },
                    "timings": timings,
                    "crossover": crossover,
                    "return_speedup_prebuilt": (
                        bf16_return_ms / int8_return_prebuilt_ms
                    ),
                    "return_speedup_with_quantization": (
                        bf16_return_ms / int8_return_with_quant_ms
                    ),
                    "packed_return_speedup_prebuilt": (
                        bf16_return_ms / packed_return_prebuilt_ms
                    ),
                    "packed_return_speedup_with_quantization": (
                        bf16_return_ms / packed_return_with_quant_ms
                    ),
                    "packed_return_speedup_vs_two_collective": (
                        int8_return_with_quant_ms
                        / packed_return_with_quant_ms
                    ),
                    "aligned_return_speedup_prebuilt": (
                        bf16_return_ms / aligned_return_prebuilt_ms
                    ),
                    "aligned_return_speedup_with_quantization": (
                        bf16_return_ms / aligned_return_with_quant_ms
                    ),
                    "aligned_return_speedup_vs_two_collective": (
                        int8_return_with_quant_ms
                        / aligned_return_with_quant_ms
                    ),
                    "aligned_return_speedup_vs_unaligned": (
                        packed_return_with_quant_ms
                        / aligned_return_with_quant_ms
                    ),
                    "forward_only_end_to_end_speedup": (
                        bf16_end_to_end_ms
                        / forward_only_end_to_end_ms
                    ),
                    "roundtrip_end_to_end_speedup": (
                        bf16_end_to_end_ms
                        / roundtrip_end_to_end_ms
                    ),
                    "roundtrip_speedup_vs_forward_only": (
                        forward_only_end_to_end_ms
                        / roundtrip_end_to_end_ms
                    ),
                    "packed_roundtrip_end_to_end_speedup": (
                        bf16_end_to_end_ms
                        / packed_roundtrip_end_to_end_ms
                    ),
                    "packed_roundtrip_speedup_vs_forward_only": (
                        forward_only_end_to_end_ms
                        / packed_roundtrip_end_to_end_ms
                    ),
                    "packed_roundtrip_speedup_vs_two_collective": (
                        roundtrip_end_to_end_ms
                        / packed_roundtrip_end_to_end_ms
                    ),
                    "aligned_roundtrip_end_to_end_speedup": (
                        bf16_end_to_end_ms
                        / aligned_roundtrip_end_to_end_ms
                    ),
                    "aligned_roundtrip_speedup_vs_forward_only": (
                        forward_only_end_to_end_ms
                        / aligned_roundtrip_end_to_end_ms
                    ),
                    "aligned_roundtrip_speedup_vs_two_collective": (
                        roundtrip_end_to_end_ms
                        / aligned_roundtrip_end_to_end_ms
                    ),
                    "aligned_roundtrip_speedup_vs_unaligned": (
                        packed_roundtrip_end_to_end_ms
                        / aligned_roundtrip_end_to_end_ms
                    ),
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )

            del (
                prepared_route,
                prepared_bf16_metadata,
                prepared_forward_quantized,
                prepared_forward_metadata,
                bf16_dispatch,
                int8_dispatch,
                bf16_expert_output,
                int8_expert_output,
                prepared_return_quantized,
                prepared_return_scales,
                prepared_return_packet,
                prepared_aligned_return_packet,
                bf16_output,
                forward_only_output,
                roundtrip_output,
                packed_roundtrip_output,
                aligned_roundtrip_output,
                isolated_return_output,
                isolated_packed_return_output,
                isolated_aligned_return_output,
                reference_output,
            )
            torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_int8_packed_return",
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
                "The BF16 baseline uses metadata-packed BF16 forward dispatch and one BF16 return all-to-all.",
                "The forward-only path uses fused Triton INT8 dispatch and retains the BF16 return path.",
                "The two-collective roundtrip path quantizes every expert-output row to INT8 with one BF16 scale and sends payload and scale separately.",
                "The packed path writes INT8 values and the BF16 scale into disjoint typed views of one uint8 allocation, sends one return collective, and dequantizes before weighted combine.",
                "The aligned packed control pads each 1026-byte row to 1040 bytes so every row and destination segment starts on a 16-byte boundary.",
                "Quality is measured against BF16, and both packed paths must match the two-collective INT8 result exactly.",
                "Crossovers balance BF16, forward-only INT8, two-collective roundtrip INT8, unaligned packed INT8, and aligned packed INT8 across timing positions.",
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
        choices=["smoke", "full", "crossover", "alignment"],
        default="smoke",
    )
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=2816)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-output-nrmse", type=float, default=0.05)
    parser.add_argument("--min-output-cosine", type=float, default=0.998)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
