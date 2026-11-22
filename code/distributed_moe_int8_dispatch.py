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
    gather_scalar,
    local_experts,
    output_quality,
    return_and_combine,
    route_and_pack,
    routing_bias,
    timed_schedule,
)
from distributed_moe_packed_dispatch import (
    build_metadata_packet,
    dispatch_assignments_metadata_packed,
    timed_crossover,
)


def quantize_hidden_per_token(
    route: TensorDict,
) -> tuple[torch.Tensor, torch.Tensor]:
    send_hidden = route["send_hidden"]
    send_local_ids = route["send_local_ids"]
    send_weights = route["send_weights"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_local_ids, torch.Tensor)
    assert isinstance(send_weights, torch.Tensor)

    scale = (
        send_hidden.float().abs().amax(dim=1, keepdim=True)
        / 127.0
    ).clamp_min(1e-8)
    transmitted_scale = scale.to(send_hidden.dtype)
    quantized = (
        torch.round(
            send_hidden.float() / transmitted_scale.float()
        )
        .clamp_(-127, 127)
        .to(torch.int8)
        .contiguous()
    )
    metadata = torch.cat(
        (
            send_local_ids.to(send_hidden.dtype),
            send_weights,
            transmitted_scale,
        ),
        dim=1,
    ).contiguous()
    return quantized, metadata


def dequantize_hidden(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (
        quantized.float() * scale.float()
    ).to(dtype).contiguous()


def dispatch_assignments_int8(
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
        quantized, metadata = quantize_hidden_per_token(route)

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
    recv_hidden = dequantize_hidden(
        recv_quantized,
        recv_metadata[:, 2:3],
        send_hidden.dtype,
    )
    return {
        "recv_hidden": recv_hidden,
        "recv_local_ids": (
            recv_metadata[:, :1].to(torch.int32).contiguous()
        ),
        "recv_weights": recv_metadata[:, 1:2].contiguous(),
        "recv_counts": recv_counts,
        "recv_quantized": recv_quantized,
        "recv_scale": recv_metadata[:, 2:3].contiguous(),
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
        input_generator.manual_seed(20_000 + rank * 101 + num_tokens)
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
            prepared_quantized, prepared_int8_metadata = (
                quantize_hidden_per_token(prepared_route)
            )
            bf16_dispatch = dispatch_assignments_metadata_packed(
                prepared_route,
                metadata_packet=prepared_bf16_metadata,
            )
            int8_dispatch = dispatch_assignments_int8(
                prepared_route,
                quantized=prepared_quantized,
                metadata=prepared_int8_metadata,
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
            bf16_output = return_and_combine(
                bf16_expert_output,
                prepared_route,
                bf16_dispatch,
                num_tokens=num_tokens,
            )
            int8_output = return_and_combine(
                int8_expert_output,
                prepared_route,
                int8_dispatch,
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

            hidden_quality = output_quality(
                int8_dispatch["recv_hidden"],
                bf16_dispatch["recv_hidden"],
            )
            output_quality_vs_bf16 = output_quality(
                int8_output,
                bf16_output,
            )
            bf16_reference_quality = output_quality(
                bf16_output,
                reference_output,
            )
            int8_reference_quality = output_quality(
                int8_output,
                reference_output,
            )
            ids_equal = torch.equal(
                int8_dispatch["recv_local_ids"],
                bf16_dispatch["recv_local_ids"],
            )
            weight_quality = output_quality(
                int8_dispatch["recv_weights"],
                bf16_dispatch["recv_weights"],
            )
            if (
                not ids_equal
                or weight_quality["max_abs_error"] != 0.0
                or hidden_quality["normalized_rmse"]
                > args.max_hidden_nrmse
                or hidden_quality["cosine_similarity"]
                < args.min_hidden_cosine
                or output_quality_vs_bf16["normalized_rmse"]
                > args.max_output_nrmse
                or output_quality_vs_bf16["cosine_similarity"]
                < args.min_output_cosine
            ):
                raise RuntimeError(
                    f"INT8 dispatch quality failed for "
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

            def quantize_only() -> tuple[torch.Tensor, torch.Tensor]:
                return quantize_hidden_per_token(prepared_route)

            def dequantize_only() -> torch.Tensor:
                return dequantize_hidden(
                    prepared_quantized,
                    prepared_int8_metadata[:, 2:3],
                    dtype,
                )

            def bf16_dispatch_prebuilt() -> TensorDict:
                return dispatch_assignments_metadata_packed(
                    prepared_route,
                    metadata_packet=prepared_bf16_metadata,
                )

            def bf16_dispatch_with_pack() -> TensorDict:
                return dispatch_assignments_metadata_packed(
                    prepared_route
                )

            def int8_dispatch_prebuilt() -> TensorDict:
                return dispatch_assignments_int8(
                    prepared_route,
                    quantized=prepared_quantized,
                    metadata=prepared_int8_metadata,
                )

            def int8_dispatch_with_quant() -> TensorDict:
                return dispatch_assignments_int8(prepared_route)

            def bf16_end_to_end() -> torch.Tensor:
                route = build_route()
                metadata = build_metadata_packet(route)
                dispatched = dispatch_assignments_metadata_packed(
                    route,
                    metadata_packet=metadata,
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

            def int8_end_to_end() -> torch.Tensor:
                route = build_route()
                quantized, metadata = quantize_hidden_per_token(
                    route
                )
                dispatched = dispatch_assignments_int8(
                    route,
                    quantized=quantized,
                    metadata=metadata,
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
                ("quantize_only", quantize_only),
                ("dequantize_only", dequantize_only),
                (
                    "bf16_dispatch_prebuilt",
                    bf16_dispatch_prebuilt,
                ),
                (
                    "bf16_dispatch_with_pack",
                    bf16_dispatch_with_pack,
                ),
                (
                    "int8_dispatch_prebuilt",
                    int8_dispatch_prebuilt,
                ),
                (
                    "int8_dispatch_with_quant",
                    int8_dispatch_with_quant,
                ),
                ("bf16_end_to_end", bf16_end_to_end),
                ("int8_end_to_end", int8_end_to_end),
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
                        ("int8", int8_dispatch_with_quant),
                    ),
                    warmup_cycles=6,
                    measured_cycles=30,
                    local_rank=local_rank,
                    device=device,
                )
                crossover_end_to_end = timed_crossover(
                    (
                        ("bf16", bf16_end_to_end),
                        ("int8", int8_end_to_end),
                    ),
                    warmup_cycles=6,
                    measured_cycles=30,
                    local_rank=local_rank,
                    device=device,
                )
                crossover = {
                    "method": (
                        "Alternating-order crossover over BF16 "
                        "and per-token INT8"
                    ),
                    "warmup_cycles": 6,
                    "measured_cycles": 30,
                    "dispatch": crossover_dispatch,
                    "end_to_end": crossover_end_to_end,
                    "dispatch_speedup": (
                        crossover_dispatch["bf16"][
                            "critical_path_ms"
                        ]
                        / crossover_dispatch["int8"][
                            "critical_path_ms"
                        ]
                    ),
                    "end_to_end_speedup": (
                        crossover_end_to_end["bf16"][
                            "critical_path_ms"
                        ]
                        / crossover_end_to_end["int8"][
                            "critical_path_ms"
                        ]
                    ),
                }

            bf16_dispatch_prebuilt_ms = timings[
                "bf16_dispatch_prebuilt"
            ]["critical_path_ms"]
            bf16_dispatch_with_pack_ms = timings[
                "bf16_dispatch_with_pack"
            ]["critical_path_ms"]
            int8_dispatch_prebuilt_ms = timings[
                "int8_dispatch_prebuilt"
            ]["critical_path_ms"]
            int8_dispatch_with_quant_ms = timings[
                "int8_dispatch_with_quant"
            ]["critical_path_ms"]
            bf16_end_to_end_ms = timings["bf16_end_to_end"][
                "critical_path_ms"
            ]
            int8_end_to_end_ms = timings["int8_end_to_end"][
                "critical_path_ms"
            ]
            bf16_bytes = args.hidden * 2 + 4
            int8_bytes = args.hidden + 6
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
                        "bf16_payload_collectives": 2,
                        "int8_payload_collectives": 2,
                        "bf16_bytes_per_assignment": bf16_bytes,
                        "int8_bytes_per_assignment": int8_bytes,
                        "byte_reduction_fraction": (
                            1 - int8_bytes / bf16_bytes
                        ),
                        "quantization": (
                            "symmetric per-token INT8 with BF16 scale"
                        ),
                        "return_path": "BF16",
                    },
                    "quality": {
                        "ids_equal": ids_equal,
                        "max_weight_error": max(
                            gather_scalar(
                                weight_quality["max_abs_error"],
                                device,
                            )
                        ),
                        "max_hidden_abs_error": max(
                            gather_scalar(
                                hidden_quality["max_abs_error"],
                                device,
                            )
                        ),
                        "max_hidden_nrmse": max(
                            gather_scalar(
                                hidden_quality["normalized_rmse"],
                                device,
                            )
                        ),
                        "min_hidden_cosine": min(
                            gather_scalar(
                                hidden_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                        "max_output_abs_error_vs_bf16": max(
                            gather_scalar(
                                output_quality_vs_bf16[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "max_output_nrmse_vs_bf16": max(
                            gather_scalar(
                                output_quality_vs_bf16[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_output_cosine_vs_bf16": min(
                            gather_scalar(
                                output_quality_vs_bf16[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                        "max_bf16_reference_nrmse": max(
                            gather_scalar(
                                bf16_reference_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "max_int8_reference_nrmse": max(
                            gather_scalar(
                                int8_reference_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "min_int8_reference_cosine": min(
                            gather_scalar(
                                int8_reference_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                    },
                    "timings": timings,
                    "crossover": crossover,
                    "dispatch_speedup_prebuilt": (
                        bf16_dispatch_prebuilt_ms
                        / int8_dispatch_prebuilt_ms
                    ),
                    "dispatch_speedup_with_quantization": (
                        bf16_dispatch_with_pack_ms
                        / int8_dispatch_with_quant_ms
                    ),
                    "end_to_end_speedup": (
                        bf16_end_to_end_ms / int8_end_to_end_ms
                    ),
                    "quantize_fraction_of_int8_end_to_end": (
                        timings["quantize_only"][
                            "critical_path_ms"
                        ]
                        / int8_end_to_end_ms
                    ),
                    "dequantize_fraction_of_int8_end_to_end": (
                        timings["dequantize_only"][
                            "critical_path_ms"
                        ]
                        / int8_end_to_end_ms
                    ),
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )

            del (
                prepared_route,
                prepared_bf16_metadata,
                prepared_quantized,
                prepared_int8_metadata,
                bf16_dispatch,
                int8_dispatch,
                bf16_expert_output,
                int8_expert_output,
                bf16_output,
                int8_output,
                reference_output,
            )
            torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_int8_dispatch",
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
                "The BF16 baseline keeps hidden activations contiguous and packs expert ID plus router weight into one BF16 metadata collective.",
                "The compressed path quantizes each routed hidden row symmetrically to INT8, sends a BF16 per-token scale with expert ID and router weight, and dequantizes back to BF16 before the unchanged fused expert kernel.",
                "Both paths use one hidden-payload and one metadata all-to-all after count exchange; the experiment isolates byte reduction versus quantize/dequantize overhead.",
                "The return all-to-all remains BF16, so reported byte savings apply only to forward expert dispatch.",
                "Quality is measured against both the BF16 distributed path and the global-expert BF16 reference before timing.",
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
