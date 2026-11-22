from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from analyze_router_profile_drift import (
    build_profile,
    destination_steps,
)
from distributed_moe_end_to_end import (
    gather_scalar,
    local_experts,
    output_quality,
)
from distributed_moe_packed_dispatch import timed_crossover
from distributed_moe_profiled_packer import profile_tensors
from distributed_moe_profiled_packer_scaling import pack_triton_tiled
from distributed_moe_profiled_spill import (
    dispatch_static_metadata,
    pack_dynamic_from_topk,
    reconstruct_topk_ids,
    remap_expert_ids,
    return_static_fp32,
    run_dynamic,
    trace_step_indices,
)


def build_candidate_specs(
    matrices: np.ndarray,
    *,
    calibration_steps: int,
    evaluation_steps: list[int],
    selected_layers: list[int],
    additive_margin: int,
    profile_policy: str,
    assignments_per_source: int,
    adaptive_budget: float,
    adaptive_window: int,
    minimum_per_kind: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, float | int],
]:
    candidates: dict[str, list[dict[str, Any]]] = {
        "hit": [],
        "miss": [],
    }
    all_hits: list[bool] = []
    profiles: dict[tuple[int, int], np.ndarray] = {}
    for layer in selected_layers:
        calibration = matrices[:calibration_steps, layer]
        if profile_policy == "additive":
            additive = calibration.max(axis=0) + additive_margin
            for step in evaluation_steps:
                profiles[(layer, step)] = additive
            continue
        history = [
            matrix.copy()
            for matrix in calibration
        ]
        for step in evaluation_steps:
            window = np.stack(
                history[-adaptive_window:],
                axis=0,
            )
            profiles[(layer, step)] = build_profile(
                window,
                assignments_per_source=assignments_per_source,
                budget_fraction=adaptive_budget,
                risk_weighted=False,
            )
            history.append(matrices[step, layer].copy())
    for step in evaluation_steps:
        for layer in selected_layers:
            profile = profiles[(layer, step)]
            hit = bool(
                np.all(matrices[step, layer] <= profile)
            )
            all_hits.append(hit)
            candidates["hit" if hit else "miss"].append(
                {
                    "kind": "hit" if hit else "miss",
                    "layer": layer,
                    "step": step,
                    "matrix": profile.copy(),
                    "synthetic": False,
                }
            )

    reference_layer = selected_layers[0]
    reference_step = evaluation_steps[0]
    actual = matrices[reference_step, reference_layer]
    while len(candidates["hit"]) < minimum_per_kind:
        candidates["hit"].append(
            {
                "kind": "hit",
                "layer": reference_layer,
                "step": reference_step,
                "matrix": np.maximum(
                    profiles[
                        (
                            reference_layer,
                            reference_step,
                        )
                    ],
                    actual,
                ),
                "synthetic": True,
            }
        )
    while len(candidates["miss"]) < minimum_per_kind:
        matrix = actual.copy()
        source, destination = np.argwhere(actual > 0)[0]
        matrix[source, destination] -= 1
        candidates["miss"].append(
            {
                "kind": "miss",
                "layer": reference_layer,
                "step": reference_step,
                "matrix": matrix,
                "synthetic": True,
            }
        )
    return candidates, {
        "observed_hit_rate": statistics.fmean(all_hits),
        "observed_hits": sum(all_hits),
        "observed_steps": len(all_hits),
    }


def mixed_specs(
    candidates: dict[str, list[dict[str, Any]]],
    *,
    queue_depth: int,
    hit_rate: float,
) -> list[dict[str, Any]]:
    hit_count = min(
        queue_depth,
        max(0, round(queue_depth * hit_rate)),
    )
    miss_count = queue_depth - hit_count
    if queue_depth > 1:
        hit_count = min(queue_depth - 1, max(1, hit_count))
        miss_count = queue_depth - hit_count
    hits = candidates["hit"][:hit_count]
    misses = candidates["miss"][:miss_count]
    result: list[dict[str, Any]] = []
    while hits or misses:
        if hits:
            result.append(hits.pop(0))
        if misses:
            result.append(misses.pop(0))
    return result


def queue_specifications(
    candidates: dict[str, list[dict[str, Any]]],
    *,
    queue_depths: list[int],
    observed_hit_rate: float,
    smoke: bool,
    patterns: set[str],
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    result = []
    for depth in queue_depths:
        if "hit" in patterns:
            result.append(
                ("hit", depth, candidates["hit"][:depth])
            )
        if "miss" in patterns:
            result.append(
                ("miss", depth, candidates["miss"][:depth])
            )
        if (
            "mixed" in patterns
            and (depth > 1 or not smoke)
        ):
            result.append(
                (
                    "mixed",
                    depth,
                    mixed_specs(
                        candidates,
                        queue_depth=depth,
                        hit_rate=observed_hit_rate,
                    ),
                )
            )
    return result


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    trace = json.loads(Path(args.trace).read_text())
    labels, matrices = destination_steps(trace, world_size)
    _, sample_indices = trace_step_indices(trace, world_size)
    steps_per_domain = (
        int(trace["samples_per_domain"]) // world_size
    )
    calibration_steps = (
        args.calibration_domains * steps_per_domain
    )
    evaluation_steps = list(
        range(calibration_steps, len(labels))
    )
    queue_depths = sorted(
        {
            int(value)
            for value in args.queue_depths.split(",")
            if value
        }
    )
    if (
        not queue_depths
        or queue_depths[0] < 1
        or queue_depths[-1] > 16
    ):
        raise ValueError("queue depths must be within 1..16")
    patterns = {
        value
        for value in args.patterns.split(",")
        if value
    }
    if not patterns or not patterns <= {
        "hit",
        "miss",
        "mixed",
    }:
        raise ValueError(
            "patterns must use hit,miss,mixed"
        )
    if args.profile == "smoke":
        selected_layers = [0]
        queue_depths = [
            depth
            for depth in queue_depths
            if depth <= 2
        ]
        warmup_cycles, measured_cycles = 1, 4
        controller_warmup, controller_cycles = 2, 10
    elif args.profile == "crossover":
        selected_layers = [0, 8, 15]
        warmup_cycles, measured_cycles = 1, 8
        controller_warmup, controller_cycles = 3, 20
    else:
        selected_layers = [0, 4, 8, 12, 15]
        warmup_cycles, measured_cycles = 2, 12
        controller_warmup, controller_cycles = 5, 30
    if args.measured_cycles is not None:
        measured_cycles = args.measured_cycles
    if args.controller_cycles is not None:
        controller_cycles = args.controller_cycles

    additive_margin = (
        args.margin_8
        if world_size <= 8
        else args.margin_24
    )
    adaptive_budget = (
        args.adaptive_budget_8
        if world_size <= 8
        else args.adaptive_budget_24
    )
    adaptive_window = (
        args.adaptive_window_domains
        * steps_per_domain
    )
    assignments_per_source = (
        int(trace["sequence_length"])
        * int(trace["top_k"])
    )
    candidates, profile_statistics = build_candidate_specs(
        matrices,
        calibration_steps=calibration_steps,
        evaluation_steps=evaluation_steps,
        selected_layers=selected_layers,
        additive_margin=additive_margin,
        profile_policy=args.profile_policy,
        assignments_per_source=assignments_per_source,
        adaptive_budget=adaptive_budget,
        adaptive_window=adaptive_window,
        minimum_per_kind=max(queue_depths),
    )
    queue_specs = queue_specifications(
        candidates,
        queue_depths=queue_depths,
        observed_hit_rate=profile_statistics[
            "observed_hit_rate"
        ],
        smoke=args.profile == "smoke",
        patterns=patterns,
    )

    num_tokens = int(trace["sequence_length"])
    top_k = int(trace["top_k"])
    num_trace_experts = int(trace["num_experts"])
    _, experts_per_rank = remap_expert_ids(
        torch.arange(num_trace_experts).reshape(-1, 1),
        num_trace_experts=num_trace_experts,
        world_size=world_size,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(7_000)
    local_w1 = (
        torch.randn(
            (
                experts_per_rank,
                2 * args.intermediate,
                args.hidden,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.hidden)
    )
    local_w2 = (
        torch.randn(
            (
                experts_per_rank,
                args.hidden,
                args.intermediate,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(args.intermediate)
    )

    all_specs = {
        (
            spec["kind"],
            spec["layer"],
            spec["step"],
            spec["matrix"].tobytes(),
        ): spec
        for _, _, specs in queue_specs
        for spec in specs
    }
    trace_counts = trace["expert_counts"]
    cases_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, spec in all_specs.items():
        layer = spec["layer"]
        step = spec["step"]
        sample = sample_indices[step][rank]
        original_ids = reconstruct_topk_ids(
            trace_counts[sample][layer],
            num_tokens=num_tokens,
            top_k=top_k,
        )
        topk_ids, actual_experts_per_rank = remap_expert_ids(
            original_ids,
            num_trace_experts=num_trace_experts,
            world_size=world_size,
        )
        if actual_experts_per_rank != experts_per_rank:
            raise RuntimeError("Inconsistent local expert count")
        topk_ids = topk_ids.to(device)
        topk_weights = torch.full(
            topk_ids.shape,
            1.0 / top_k,
            device=device,
            dtype=dtype,
        )
        hidden_generator = torch.Generator(device=device)
        hidden_generator.manual_seed(
            400_000
            + rank * 10_003
            + layer * 1_003
            + step
        )
        hidden = torch.randn(
            (num_tokens, args.hidden),
            device=device,
            dtype=dtype,
            generator=hidden_generator,
        )
        matrix = spec["matrix"]
        actual_matrix = matrices[step, layer]
        expected_hit = bool(np.all(actual_matrix <= matrix))
        if expected_hit != (spec["kind"] == "hit"):
            raise RuntimeError("Case fit label is inconsistent")
        capacity_row = [
            int(value)
            for value in matrix[rank].tolist()
        ]
        fixed_rows = sum(capacity_row)
        route_bytes = (
            fixed_rows
            * (
                args.hidden * torch.tensor(
                    [],
                    dtype=dtype,
                ).element_size()
                + 4
                + torch.tensor(
                    [],
                    dtype=dtype,
                ).element_size()
                + 8
            )
            + topk_ids.numel() * 4
        )
        cases_by_key[key] = {
            **spec,
            "domain": labels[step],
            "hidden": hidden,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "profile": profile_tensors(
                capacity_row,
                device,
            ),
            "send_counts": capacity_row,
            "recv_counts": [
                int(matrix[source, rank])
                for source in range(world_size)
            ],
            "expected_hit": expected_hit,
            "local_overflow": int(
                np.any(actual_matrix[rank] > matrix[rank])
            ),
            "route_buffer_bytes_estimate": route_bytes,
        }

    methods = (
        "dynamic",
        "ideal_precomputed",
        "scalar_host",
        "vector_host",
    )
    if args.reverse_methods:
        methods = tuple(reversed(methods))
    records: list[dict[str, Any]] = []
    quality: dict[str, dict[str, dict[str, float]]] = {}
    fit_decisions_exact = True

    for pattern, queue_depth, specs in queue_specs:
        cases = [
            cases_by_key[
                (
                    spec["kind"],
                    spec["layer"],
                    spec["step"],
                    spec["matrix"].tobytes(),
                )
            ]
            for spec in specs
        ]
        expected_decisions = [
            case["expected_hit"]
            for case in cases
        ]
        local_flags = torch.tensor(
            [
                case["local_overflow"]
                for case in cases
            ],
            device=device,
            dtype=torch.int64,
        )

        def dynamic_case(
            case: dict[str, Any],
        ) -> torch.Tensor:
            route = pack_dynamic_from_topk(
                case["hidden"],
                case["topk_ids"],
                case["topk_weights"],
                experts_per_rank=experts_per_rank,
                world_size=world_size,
            )
            return run_dynamic(
                route,
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )

        def static_case(
            case: dict[str, Any],
            route: dict[str, Any],
        ) -> torch.Tensor:
            dispatched = dispatch_static_metadata(
                route,
                case["recv_counts"],
            )
            expert_output = local_experts(
                dispatched,
                local_w1,
                local_w2,
            )
            return return_static_fp32(
                expert_output,
                route,
                case["recv_counts"],
                num_tokens=num_tokens,
            ).to(dtype)

        def pack_static(
            case: dict[str, Any],
        ) -> dict[str, Any]:
            return pack_triton_tiled(
                case["hidden"],
                case["topk_ids"],
                case["topk_weights"],
                experts_per_rank=experts_per_rank,
                profile=case["profile"],
                tile_size=args.tile_size,
            )

        def execute(
            method: str,
            *,
            collect: bool,
        ) -> tuple[
            torch.Tensor | list[torch.Tensor],
            list[bool],
        ]:
            outputs: list[torch.Tensor] = []
            decisions: list[bool] = []
            last = torch.empty(0, device=device, dtype=dtype)
            if method == "vector_host":
                routes = [
                    pack_static(case)
                    for case in cases
                ]
                flags = torch.stack(
                    [
                        route["overflow_count"]
                        for route in routes
                    ]
                )
                dist.all_reduce(
                    flags,
                    op=dist.ReduceOp.SUM,
                )
                decisions = [
                    value == 0
                    for value in flags.cpu().tolist()
                ]
                for case, route, hit in zip(
                    cases,
                    routes,
                    decisions,
                ):
                    last = (
                        static_case(case, route)
                        if hit
                        else dynamic_case(case)
                    )
                    if collect:
                        outputs.append(last)
            else:
                for case in cases:
                    if method == "dynamic":
                        hit = case["expected_hit"]
                        last = dynamic_case(case)
                    elif method == "ideal_precomputed":
                        hit = case["expected_hit"]
                        last = (
                            static_case(
                                case,
                                pack_static(case),
                            )
                            if hit
                            else dynamic_case(case)
                        )
                    elif method == "scalar_host":
                        route = pack_static(case)
                        overflow = route[
                            "overflow_count"
                        ].clone()
                        dist.all_reduce(
                            overflow,
                            op=dist.ReduceOp.SUM,
                        )
                        hit = int(overflow.item()) == 0
                        last = (
                            static_case(case, route)
                            if hit
                            else dynamic_case(case)
                        )
                    else:
                        raise ValueError(
                            f"Unknown method: {method}"
                        )
                    decisions.append(hit)
                    if collect:
                        outputs.append(last)
            return (
                outputs if collect else last,
                decisions,
            )

        def scalar_controller() -> torch.Tensor:
            last = local_flags[0]
            for index in range(queue_depth):
                last = local_flags[index].clone()
                dist.all_reduce(
                    last,
                    op=dist.ReduceOp.SUM,
                )
                int(last.item())
            return last

        def vector_controller() -> torch.Tensor:
            flags = local_flags.clone()
            dist.all_reduce(
                flags,
                op=dist.ReduceOp.SUM,
            )
            flags.cpu().tolist()
            return flags

        eager_warmup, _ = execute(
            "dynamic",
            collect=False,
        )
        torch.cuda.synchronize()
        del eager_warmup

        expected_outputs, _ = execute(
            "dynamic",
            collect=True,
        )
        assert isinstance(expected_outputs, list)
        quality_key = f"{pattern}_q{queue_depth}"
        quality[quality_key] = {}
        for method in (
            "ideal_precomputed",
            "scalar_host",
            "vector_host",
        ):
            actual_outputs, decisions = execute(
                method,
                collect=True,
            )
            assert isinstance(actual_outputs, list)
            fit_decisions_exact = (
                fit_decisions_exact
                and decisions == expected_decisions
            )
            method_quality = []
            for actual, expected in zip(
                actual_outputs,
                expected_outputs,
            ):
                local_quality = output_quality(
                    actual,
                    expected,
                )
                method_quality.append(
                    {
                        "max_abs_error": max(
                            gather_scalar(
                                local_quality[
                                    "max_abs_error"
                                ],
                                device,
                            )
                        ),
                        "normalized_rmse": max(
                            gather_scalar(
                                local_quality[
                                    "normalized_rmse"
                                ],
                                device,
                            )
                        ),
                        "cosine_similarity": min(
                            gather_scalar(
                                local_quality[
                                    "cosine_similarity"
                                ],
                                device,
                            )
                        ),
                    }
                )
            quality[quality_key][method] = {
                "max_abs_error": max(
                    item["max_abs_error"]
                    for item in method_quality
                ),
                "normalized_rmse": max(
                    item["normalized_rmse"]
                    for item in method_quality
                ),
                "cosine_similarity": min(
                    item["cosine_similarity"]
                    for item in method_quality
                ),
            }
        del expected_outputs

        timing_functions = {
            method: (
                lambda selected=method: execute(
                    selected,
                    collect=False,
                )[0]
            )
            for method in methods
        }
        timings = timed_crossover(
            tuple(
                (method, timing_functions[method])
                for method in methods
            ),
            warmup_cycles=warmup_cycles,
            measured_cycles=measured_cycles,
            local_rank=local_rank,
            device=device,
        )
        controller_timings = timed_crossover(
            (
                (
                    "scalar_controller",
                    scalar_controller,
                ),
                (
                    "vector_controller",
                    vector_controller,
                ),
            ),
            warmup_cycles=controller_warmup,
            measured_cycles=controller_cycles,
            local_rank=local_rank,
            device=device,
        )

        def peak_extra(method: str) -> float:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output, _ = execute(
                method,
                collect=False,
            )
            torch.cuda.synchronize()
            peak = (
                torch.cuda.max_memory_allocated()
                - baseline
            )
            del output
            return max(gather_scalar(peak, device))

        scalar_peak = peak_extra("scalar_host")
        vector_peak = peak_extra("vector_host")
        if rank == 0:
            dynamic_ms = timings["dynamic"][
                "critical_path_ms"
            ]
            scalar_controller_ms = controller_timings[
                "scalar_controller"
            ]["critical_path_ms"]
            vector_controller_ms = controller_timings[
                "vector_controller"
            ]["critical_path_ms"]
            records.append(
                {
                    "pattern": pattern,
                    "queue_depth": queue_depth,
                    "hit_fraction": statistics.fmean(
                        expected_decisions
                    ),
                    "synthetic_cases": sum(
                        bool(case["synthetic"])
                        for case in cases
                    ),
                    "layers": [
                        case["layer"]
                        for case in cases
                    ],
                    "steps": [
                        case["step"]
                        for case in cases
                    ],
                    "domains": [
                        case["domain"]
                        for case in cases
                    ],
                    "route_buffer_bytes_estimate": sum(
                        case[
                            "route_buffer_bytes_estimate"
                        ]
                        for case in cases
                    ),
                    "scalar_peak_extra_bytes": scalar_peak,
                    "vector_peak_extra_bytes": vector_peak,
                    "timings": timings,
                    "speedup_vs_dynamic": {
                        method: (
                            dynamic_ms
                            / timing["critical_path_ms"]
                        )
                        for method, timing in timings.items()
                        if method != "dynamic"
                    },
                    "vector_vs_scalar": (
                        timings["scalar_host"][
                            "critical_path_ms"
                        ]
                        / timings["vector_host"][
                            "critical_path_ms"
                        ]
                    ),
                    "vector_vs_ideal": (
                        timings["ideal_precomputed"][
                            "critical_path_ms"
                        ]
                        / timings["vector_host"][
                            "critical_path_ms"
                        ]
                    ),
                    "controller_timings": (
                        controller_timings
                    ),
                    "controller_amortization": (
                        scalar_controller_ms
                        / vector_controller_ms
                    ),
                    "controller_per_case_us": {
                        "scalar": (
                            scalar_controller_ms
                            * 1_000.0
                            / queue_depth
                        ),
                        "vector": (
                            vector_controller_ms
                            * 1_000.0
                            / queue_depth
                        ),
                    },
                }
            )

    max_error = max(
        metrics["max_abs_error"]
        for config in quality.values()
        for metrics in config.values()
    )
    min_cosine = min(
        metrics["cosine_similarity"]
        for config in quality.values()
        for metrics in config.values()
    )
    if (
        not fit_decisions_exact
        or max_error > args.max_abs_error
        or min_cosine < 0.999
    ):
        raise RuntimeError(
            "Persistent fit correctness failed: "
            f"fit_exact={fit_decisions_exact}, "
            f"max_error={max_error}, "
            f"min_cosine={min_cosine}"
        )

    result = None
    if rank == 0:
        result = {
            "suite": "distributed_moe_persistent_fit",
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
            "trace_model": trace["model"],
            "trace_corpus_sha256": trace["corpus_sha256"],
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "num_tokens": num_tokens,
            "top_k": top_k,
            "trace_experts": num_trace_experts,
            "experts_per_rank": experts_per_rank,
            "selected_layers": selected_layers,
            "calibration_steps": calibration_steps,
            "additive_margin": additive_margin,
            "profile_policy": args.profile_policy,
            "adaptive_budget": adaptive_budget,
            "adaptive_window": adaptive_window,
            "tile_size": args.tile_size,
            "queue_depths": queue_depths,
            "patterns": sorted(patterns),
            "profile_statistics": profile_statistics,
            "records": records,
            "correctness": {
                "fit_decisions_exact": (
                    fit_decisions_exact
                ),
                "max_abs_error": max_error,
                "min_cosine_similarity": min_cosine,
                "max_allowed_abs_error": (
                    args.max_abs_error
                ),
                "cases": quality,
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "The persistent controller batches profile-fit flags for several independent ready microbatches into one vector NCCL all-reduce and one CPU synchronization.",
                "scalar_host performs the same pack, global fit reduction, host decision, and exact static-or-dynamic branch independently for every microbatch.",
                "vector_host retains all speculative fixed routes until the batched decision is available, trading memory for fewer collective launches and host synchronizations.",
                "ideal_precomputed is an upper bound with known fit decisions. Misses execute the original dynamic variable-split path rather than the count-free gathered fallback.",
                "Queue patterns use natural trained-trace profile hits and misses when available; the smoke gate may synthesize a fit boundary if one class is absent.",
            ],
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
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
        choices=["smoke", "crossover", "full"],
        default="smoke",
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--queue-depths",
        default="1,2,4,8",
    )
    parser.add_argument(
        "--patterns",
        default="hit,miss,mixed",
    )
    parser.add_argument(
        "--measured-cycles",
        type=int,
    )
    parser.add_argument(
        "--controller-cycles",
        type=int,
    )
    parser.add_argument(
        "--calibration-domains",
        type=int,
        default=4,
    )
    parser.add_argument("--margin-8", type=int, default=18)
    parser.add_argument("--margin-24", type=int, default=16)
    parser.add_argument(
        "--profile-policy",
        choices=["additive", "adaptive"],
        default="additive",
    )
    parser.add_argument(
        "--adaptive-window-domains",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--adaptive-budget-8",
        type=float,
        default=0.16,
    )
    parser.add_argument(
        "--adaptive-budget-24",
        type=float,
        default=0.48,
    )
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=0.03125,
    )
    parser.add_argument(
        "--reverse-methods",
        action="store_true",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
