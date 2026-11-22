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
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
)

from distributed_moe_end_to_end import (
    TensorDict,
    critical_path_summary,
    gather_scalar,
    output_quality,
    route_and_pack,
    routing_bias,
    timed_schedule,
)
from distributed_moe_packed_dispatch import timed_crossover
from distributed_moe_padded_capacity import (
    masked_global_reference,
    run_variable_route,
)
from distributed_moe_profiled_packer import (
    pack_two_sort,
    profile_tensors,
    router_topk,
)
from distributed_moe_profiled_packer_scaling import (
    pack_triton_tiled,
)
from distributed_moe_profiled_transport import (
    calibrate_capacity_matrix,
    gather_segment_counts,
    make_hidden,
    margin_matrix,
    matrix_column_sums,
    matrix_row_sums,
    run_static_matrix_route,
)


def profile_configurations(
    profile: str,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[str, float], ...],
    int,
    int,
    int,
]:
    if profile == "smoke":
        return (
            (100,),
            (
                ("balanced", 0.0),
                ("global_bias1", 1.0),
            ),
            4,
            1,
            1,
        )
    if profile == "crossover":
        return (
            (512,),
            (
                ("balanced", 0.0),
                ("global_bias1", 1.0),
                ("global_bias4", 4.0),
            ),
            8,
            1,
            2,
        )
    return (
        (128, 512),
        (
            ("balanced", 0.0),
            ("global_bias1", 1.0),
            ("global_bias4", 4.0),
        ),
        8,
        1,
        3,
    )


def build_fixed_route(
    method: str,
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    tie_breaker: torch.Tensor,
    *,
    top_k: int,
    experts_per_rank: int,
    profile: dict[str, torch.Tensor | int | list[int]],
    tile_size: int,
) -> TensorDict:
    topk_ids, topk_weights = router_topk(
        hidden_states,
        gate_weight,
        bias,
        tie_breaker,
        top_k=top_k,
    )
    if method == "two_sort":
        return pack_two_sort(
            hidden_states,
            topk_ids,
            topk_weights,
            experts_per_rank=experts_per_rank,
            profile=profile,
        )
    if method == "tiled_atomic":
        return pack_triton_tiled(
            hidden_states,
            topk_ids,
            topk_weights,
            experts_per_rank=experts_per_rank,
            profile=profile,
            tile_size=tile_size,
        )
    raise ValueError(f"Unknown fixed packer: {method}")


def gather_int(
    value: int,
    device: torch.device,
) -> list[int]:
    return [
        int(item)
        for item in gather_scalar(float(value), device)
    ]


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    if args.calibration_margin is None:
        args.calibration_margin = (
            18 if world_size == 8 else 16
        )

    (
        token_values,
        routes,
        calibration_batches,
        warmup,
        iterations,
    ) = profile_configurations(args.profile)
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
    global_w1: torch.Tensor | None = None
    global_w2: torch.Tensor | None = None
    local_w1: torch.Tensor | None = None
    local_w2: torch.Tensor | None = None
    if not args.capacity_only:
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

    fixed_methods = (
        ("two_sort", "fixed two-sort"),
        ("tiled_atomic", "fixed tiled atomic"),
    )
    if args.reverse_methods:
        fixed_methods = tuple(reversed(fixed_methods))

    records: list[dict[str, Any]] = []
    for num_tokens in token_values:
        assignments_per_rank = num_tokens * args.top_k
        for route_name, hot_bias in routes:
            bias = routing_bias(
                "global_biased",
                rank=rank,
                num_experts=num_experts,
                experts_per_rank=args.experts_per_rank,
                hot_bias=hot_bias,
                device=device,
            )
            calibration_matrix = calibrate_capacity_matrix(
                route_name=route_name,
                calibration_batches=calibration_batches,
                num_tokens=num_tokens,
                hidden_size=args.hidden,
                gate_weight=gate_weight,
                bias=bias,
                tie_breaker=tie_breaker,
                top_k=args.top_k,
                experts_per_rank=args.experts_per_rank,
                world_size=world_size,
                rank=rank,
                device=device,
                dtype=dtype,
            )
            capacity_matrix = margin_matrix(
                calibration_matrix,
                args.calibration_margin,
            )
            capacity_row = capacity_matrix[rank]
            recv_counts = [
                capacity_matrix[source][rank]
                for source in range(world_size)
            ]
            profile = profile_tensors(
                capacity_row,
                device,
            )
            hidden = make_hidden(
                num_tokens,
                args.hidden,
                route_name=route_name,
                rank=rank,
                batch=0,
                calibration=False,
                device=device,
                dtype=dtype,
            )

            def build_dynamic() -> TensorDict:
                return route_and_pack(
                    hidden,
                    gate_weight,
                    bias,
                    tie_breaker,
                    top_k=args.top_k,
                    experts_per_rank=args.experts_per_rank,
                    world_size=world_size,
                )

            fixed_builders: dict[
                str,
                Callable[[], TensorDict],
            ] = {}
            for method, _ in fixed_methods:
                def build_fixed(
                    selected_method: str = method,
                ) -> TensorDict:
                    return build_fixed_route(
                        selected_method,
                        hidden,
                        gate_weight,
                        bias,
                        tie_breaker,
                        top_k=args.top_k,
                        experts_per_rank=(
                            args.experts_per_rank
                        ),
                        profile=profile,
                        tile_size=args.tile_size,
                    )

                fixed_builders[method] = build_fixed

            dynamic_route = build_dynamic()
            dynamic_send_counts = dynamic_route["send_counts"]
            assert isinstance(dynamic_send_counts, list)
            actual_count_matrix = gather_segment_counts(
                dynamic_send_counts,
                device,
            )
            minimum_lossless_margin = max(
                0,
                max(
                    actual - calibrated
                    for actual_row, calibration_row in zip(
                        actual_count_matrix,
                        calibration_matrix,
                        strict=True,
                    )
                    for actual, calibrated in zip(
                        actual_row,
                        calibration_row,
                        strict=True,
                    )
                ),
            )
            if args.capacity_only:
                records.append(
                    {
                        "tokens_per_rank": num_tokens,
                        "assignments_per_rank": (
                            assignments_per_rank
                        ),
                        "hidden": args.hidden,
                        "experts_per_rank": (
                            args.experts_per_rank
                        ),
                        "num_experts": num_experts,
                        "top_k": args.top_k,
                        "route": route_name,
                        "hot_bias": hot_bias,
                        "calibration_batches": (
                            calibration_batches
                        ),
                        "requested_calibration_margin": (
                            args.calibration_margin
                        ),
                        "minimum_lossless_margin": (
                            minimum_lossless_margin
                        ),
                        "calibration_count_matrix": (
                            calibration_matrix
                        ),
                        "actual_count_matrix": (
                            actual_count_matrix
                        ),
                        "capacity_matrix": capacity_matrix,
                    }
                )
                del hidden, dynamic_route
                torch.cuda.empty_cache()
                continue

            assert global_w1 is not None
            assert global_w2 is not None
            assert local_w1 is not None
            assert local_w2 is not None
            fixed_routes = {
                name: builder()
                for name, builder in fixed_builders.items()
            }
            overflow_by_method = {
                name: max(
                    gather_int(
                        int(
                            route[
                                "overflow_count"
                            ].item()
                        ),
                        device,
                    )
                )
                for name, route in fixed_routes.items()
            }
            if any(overflow_by_method.values()):
                raise RuntimeError(
                    "Wide profiled route overflowed: "
                    f"tokens={num_tokens}, route={route_name}, "
                    f"requested_margin={args.calibration_margin}, "
                    f"minimum_lossless_margin="
                    f"{minimum_lossless_margin}, "
                    f"overflow={overflow_by_method}"
                )

            dynamic_output = run_variable_route(
                dynamic_route,
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )
            fixed_outputs = {
                name: run_static_matrix_route(
                    route,
                    recv_counts,
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )
                for name, route in fixed_routes.items()
            }
            tiled_route = fixed_routes["tiled_atomic"]
            masked_reference = masked_global_reference(
                hidden,
                global_w1,
                global_w2,
                tiled_route,
            )
            topk_ids = tiled_route["topk_ids"]
            topk_weights = tiled_route["topk_weights"]
            assert isinstance(topk_ids, torch.Tensor)
            assert isinstance(topk_weights, torch.Tensor)
            uncapped_reference = fused_experts(
                hidden,
                global_w1,
                global_w2,
                topk_weights,
                topk_ids,
            )
            torch.cuda.synchronize()

            dynamic_selected = dynamic_route[
                "sorted_assignment_indices"
            ]
            assert isinstance(dynamic_selected, torch.Tensor)
            selection_equal: dict[str, bool] = {}
            correctness = {
                "dynamic_vs_masked_reference": (
                    output_quality(
                        dynamic_output,
                        masked_reference,
                    )
                )
            }
            for name, route in fixed_routes.items():
                selected = route[
                    "selected_assignment_indices"
                ]
                assert isinstance(selected, torch.Tensor)
                local_equal = torch.equal(
                    torch.sort(selected).values,
                    torch.sort(dynamic_selected).values,
                )
                selection_equal[name] = (
                    min(
                        gather_scalar(
                            float(local_equal),
                            device,
                        )
                    )
                    == 1.0
                )
                correctness[
                    f"{name}_vs_masked_reference"
                ] = output_quality(
                    fixed_outputs[name],
                    masked_reference,
                )
                correctness[
                    f"{name}_vs_dynamic"
                ] = output_quality(
                    fixed_outputs[name],
                    dynamic_output,
                )
            correctness[
                "tiled_atomic_vs_two_sort"
            ] = output_quality(
                fixed_outputs["tiled_atomic"],
                fixed_outputs["two_sort"],
            )
            if (
                not all(selection_equal.values())
                or any(
                    quality["max_abs_error"] != 0.0
                    for quality in correctness.values()
                )
            ):
                raise RuntimeError(
                    "Wide layer correctness failed: "
                    f"selection={selection_equal}, "
                    f"correctness={correctness}"
                )

            accepted_counts_tensor = tiled_route[
                "accepted_counts_tensor"
            ]
            selected_weights = tiled_route[
                "selected_weights"
            ]
            assert isinstance(
                accepted_counts_tensor,
                torch.Tensor,
            )
            assert isinstance(selected_weights, torch.Tensor)
            accepted_matrix = gather_segment_counts(
                [
                    int(value)
                    for value in (
                        accepted_counts_tensor.cpu().tolist()
                    )
                ],
                device,
            )
            accepted_global = sum(
                sum(row)
                for row in accepted_matrix
            )
            planned_global = (
                world_size * assignments_per_rank
            )
            if accepted_global != planned_global:
                raise RuntimeError(
                    "Calibrated wide profile was not lossless: "
                    f"accepted={accepted_global}, "
                    f"planned={planned_global}"
                )

            def dynamic_payload() -> torch.Tensor:
                return run_variable_route(
                    dynamic_route,
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            payload_functions: dict[
                str,
                Callable[[], torch.Tensor],
            ] = {"dynamic": dynamic_payload}
            route_functions: dict[
                str,
                Callable[[], TensorDict],
            ] = {"dynamic": build_dynamic}
            end_to_end_functions: dict[
                str,
                Callable[[], torch.Tensor],
            ] = {}
            for method, _ in fixed_methods:
                def fixed_payload(
                    selected_method: str = method,
                ) -> torch.Tensor:
                    return run_static_matrix_route(
                        fixed_routes[selected_method],
                        recv_counts,
                        local_w1,
                        local_w2,
                        num_tokens=num_tokens,
                    )

                def fixed_end_to_end(
                    selected_method: str = method,
                ) -> torch.Tensor:
                    return run_static_matrix_route(
                        fixed_builders[selected_method](),
                        recv_counts,
                        local_w1,
                        local_w2,
                        num_tokens=num_tokens,
                    )

                payload_functions[method] = fixed_payload
                route_functions[method] = (
                    fixed_builders[method]
                )
                end_to_end_functions[method] = (
                    fixed_end_to_end
                )

            def dynamic_end_to_end() -> torch.Tensor:
                return run_variable_route(
                    build_dynamic(),
                    local_w1,
                    local_w2,
                    num_tokens=num_tokens,
                )

            end_to_end_functions = {
                "dynamic": dynamic_end_to_end,
                **end_to_end_functions,
            }
            stage_functions: dict[
                str,
                dict[str, Callable[[], Any]],
            ] = {
                "route_pack": route_functions,
                "payload_prebuilt": payload_functions,
                "end_to_end": end_to_end_functions,
            }
            timings = {
                stage: {
                    name: critical_path_summary(
                        timed_schedule(
                            function,
                            warmup=warmup,
                            iterations=iterations,
                            local_rank=local_rank,
                        ),
                        device,
                    )
                    for name, function in functions.items()
                }
                for stage, functions in (
                    stage_functions.items()
                )
            }
            crossovers = {
                "route_pack": None,
                "payload_prebuilt": None,
                "end_to_end": None,
            }
            if args.profile == "crossover":
                crossovers = {
                    stage: timed_crossover(
                        tuple(functions.items()),
                        warmup_cycles=6,
                        measured_cycles=(
                            args.crossover_cycles
                        ),
                        local_rank=local_rank,
                        device=device,
                    )
                    for stage, functions in (
                        stage_functions.items()
                    )
                }

            fixed_send_rows = matrix_row_sums(
                capacity_matrix
            )
            fixed_recv_rows = matrix_column_sums(
                capacity_matrix
            )
            valid_recv_rows = matrix_column_sums(
                accepted_matrix
            )
            fixed_rows_global = sum(fixed_send_rows)
            planned_weight_global = sum(
                gather_scalar(
                    float(
                        topk_weights.float().sum().item()
                    ),
                    device,
                )
            )
            accepted_weight_global = sum(
                gather_scalar(
                    float(
                        selected_weights.float().sum().item()
                    ),
                    device,
                )
            )
            speedups = {}
            for stage in (
                "route_pack",
                "payload_prebuilt",
                "end_to_end",
            ):
                stage_timing = timings[stage]
                speedups[
                    f"tiled_vs_two_sort_{stage}"
                ] = (
                    stage_timing["two_sort"][
                        "critical_path_ms"
                    ]
                    / stage_timing["tiled_atomic"][
                        "critical_path_ms"
                    ]
                )
                speedups[
                    f"two_sort_vs_dynamic_{stage}"
                ] = (
                    stage_timing["dynamic"][
                        "critical_path_ms"
                    ]
                    / stage_timing["two_sort"][
                        "critical_path_ms"
                    ]
                )
                speedups[
                    f"tiled_vs_dynamic_{stage}"
                ] = (
                    stage_timing["dynamic"][
                        "critical_path_ms"
                    ]
                    / stage_timing["tiled_atomic"][
                        "critical_path_ms"
                    ]
                )

            records.append(
                {
                    "tokens_per_rank": num_tokens,
                    "assignments_per_rank": (
                        assignments_per_rank
                    ),
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts_per_rank": (
                        args.experts_per_rank
                    ),
                    "num_experts": num_experts,
                    "top_k": args.top_k,
                    "route": route_name,
                    "hot_bias": hot_bias,
                    "tile_size": args.tile_size,
                    "calibration_batches": (
                        calibration_batches
                    ),
                    "calibration_margin": (
                        args.calibration_margin
                    ),
                    "minimum_lossless_margin": (
                        minimum_lossless_margin
                    ),
                    "calibration_count_matrix": (
                        calibration_matrix
                    ),
                    "actual_count_matrix": (
                        actual_count_matrix
                    ),
                    "capacity_matrix": capacity_matrix,
                    "accepted_count_matrix": (
                        accepted_matrix
                    ),
                    "capacity": {
                        "planned_assignments": (
                            planned_global
                        ),
                        "accepted_assignments": (
                            accepted_global
                        ),
                        "dropped_assignment_fraction": 0.0,
                        "retained_router_weight_fraction": (
                            accepted_weight_global
                            / planned_weight_global
                        ),
                        "lossless": True,
                    },
                    "transport": {
                        "fixed_send_rows_by_rank": (
                            fixed_send_rows
                        ),
                        "fixed_recv_rows_by_rank": (
                            fixed_recv_rows
                        ),
                        "valid_recv_rows_by_rank": (
                            valid_recv_rows
                        ),
                        "global_fixed_rows": (
                            fixed_rows_global
                        ),
                        "global_padding_fraction": (
                            1.0
                            - accepted_global
                            / fixed_rows_global
                        ),
                    },
                    "overflow_by_method": (
                        overflow_by_method
                    ),
                    "correctness": {
                        "selection_equal": (
                            selection_equal
                        ),
                        "output": correctness,
                    },
                    "quality_vs_uncapped": output_quality(
                        fixed_outputs["tiled_atomic"],
                        uncapped_reference,
                    ),
                    "timings": timings,
                    "grouped_speedups": speedups,
                    "crossover": crossovers,
                    "warmup": warmup,
                    "iterations": iterations,
                }
            )
            del (
                hidden,
                dynamic_route,
                fixed_routes,
                dynamic_output,
                fixed_outputs,
                masked_reference,
                uncapped_reference,
            )
            torch.cuda.empty_cache()

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_wide_profiled_layer",
            "profile": args.profile,
            "capacity_only": args.capacity_only,
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
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "The experiment integrates the shape-gated tile-512 fixed-profile packer into a hidden-4096/intermediate-4096 expert-parallel MoE layer.",
                "The dynamic control uses online variable split lists. Fixed two-sort and fixed tiled routes reuse an eight-batch calibrated source-destination matrix with the requested additive rows per cell.",
                "Route timing includes router projection, fused top-k, and packing. Payload timing reuses prepared routes and includes dispatch, experts, return, and combine. End-to-end timing includes both.",
                "All methods must retain the same assignments, report zero overflow, and match the independently masked global-expert output exactly.",
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
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--calibration-margin",
        type=int,
        default=None,
        help=(
            "Additive rows per source-destination cell. "
            "Defaults to the measured lossless margin: "
            "18 for 8 GPUs and 16 otherwise."
        ),
    )
    parser.add_argument(
        "--capacity-only",
        action="store_true",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=[128, 256, 512],
        default=512,
    )
    parser.add_argument(
        "--crossover-cycles",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--reverse-methods",
        action="store_true",
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
