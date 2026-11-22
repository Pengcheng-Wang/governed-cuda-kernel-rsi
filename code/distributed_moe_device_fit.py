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

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from analyze_router_profile_drift import destination_steps
from distributed_moe_end_to_end import (
    gather_scalar,
    output_quality,
)
from distributed_moe_packed_dispatch import timed_crossover
from distributed_moe_profiled_p2p import run_static_alltoall
from distributed_moe_profiled_packer import profile_tensors
from distributed_moe_profiled_packer_scaling import pack_triton_tiled
from distributed_moe_profiled_spill import (
    pack_dynamic_from_topk,
    reconstruct_topk_ids,
    remap_expert_ids,
    run_dynamic,
    trace_step_indices,
)


torch._dynamo.config.capture_dynamic_output_shape_ops = True


def run_static_functional(
    send_hidden: torch.Tensor,
    send_local_ids: torch.Tensor,
    send_weights: torch.Tensor,
    fixed_token_indices: torch.Tensor,
    *,
    send_counts: list[int],
    recv_counts: list[int],
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    metadata = torch.cat(
        (
            send_local_ids.to(send_hidden.dtype),
            send_weights,
        ),
        dim=1,
    ).contiguous()
    recv_hidden = funcol.all_to_all_single(
        send_hidden,
        recv_counts,
        send_counts,
        dist.group.WORLD,
    )
    recv_metadata = funcol.all_to_all_single(
        metadata,
        recv_counts,
        send_counts,
        dist.group.WORLD,
    )
    expert_output = fused_experts(
        recv_hidden,
        local_w1,
        local_w2,
        recv_metadata[:, 1:].contiguous(),
        recv_metadata[:, :1].to(torch.int32).contiguous(),
    )
    returned = funcol.all_to_all_single(
        expert_output,
        send_counts,
        recv_counts,
        dist.group.WORLD,
    )
    combined = torch.zeros(
        (num_tokens, returned.shape[1]),
        device=returned.device,
        dtype=torch.float32,
    )
    return torch.index_add(
        combined,
        0,
        fixed_token_indices,
        returned.float(),
    ).to(send_hidden.dtype)


def run_gather_fallback(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    *,
    rank: int,
    experts_per_rank: int,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    top_k = topk_ids.shape[1]
    metadata = torch.cat(
        (
            topk_ids.to(hidden_states.dtype),
            topk_weights,
        ),
        dim=1,
    ).contiguous()
    gathered_hidden = funcol.all_gather_tensor(
        hidden_states,
        0,
        dist.group.WORLD,
    )
    gathered_metadata = funcol.all_gather_tensor(
        metadata,
        0,
        dist.group.WORLD,
    )
    gathered_ids = gathered_metadata[:, :top_k].to(
        torch.int64
    )
    gathered_weights = gathered_metadata[:, top_k:]
    flat_ids = gathered_ids.reshape(-1)
    destinations = torch.div(
        flat_ids,
        experts_per_rank,
        rounding_mode="floor",
    )
    selected_assignments = torch.nonzero(
        destinations == rank,
        as_tuple=False,
    ).reshape(-1)
    selected_tokens = torch.div(
        selected_assignments,
        top_k,
        rounding_mode="floor",
    )
    selected_hidden = gathered_hidden.index_select(
        0,
        selected_tokens,
    )
    selected_local_ids = torch.remainder(
        flat_ids.index_select(0, selected_assignments),
        experts_per_rank,
    ).to(torch.int32).reshape(-1, 1)
    selected_weights = (
        gathered_weights.reshape(-1)
        .index_select(0, selected_assignments)
        .reshape(-1, 1)
    )
    expert_output = fused_experts(
        selected_hidden,
        local_w1,
        local_w2,
        selected_weights,
        selected_local_ids,
    )
    contributions = torch.zeros(
        (
            gathered_hidden.shape[0],
            hidden_states.shape[1],
        ),
        device=hidden_states.device,
        dtype=accumulation_dtype,
    )
    contributions = torch.index_add(
        contributions,
        0,
        selected_tokens,
        expert_output.to(accumulation_dtype),
    )
    return funcol.reduce_scatter_tensor(
        contributions,
        "sum",
        0,
        dist.group.WORLD,
    ).to(hidden_states.dtype)


def build_case_specs(
    matrices: np.ndarray,
    *,
    calibration_steps: int,
    evaluation_steps: list[int],
    selected_layers: list[int],
    additive_margin: int,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    all_hits: list[bool] = []
    for layer in selected_layers:
        additive = (
            matrices[:calibration_steps, layer].max(axis=0)
            + additive_margin
        )
        layer_hits = [
            bool(np.all(matrices[step, layer] <= additive))
            for step in evaluation_steps
        ]
        all_hits.extend(layer_hits)
        if smoke:
            step = evaluation_steps[0]
            actual = matrices[step, layer]
            specs.append(
                {
                    "kind": "hit",
                    "layer": layer,
                    "step": step,
                    "matrix": np.maximum(
                        additive,
                        actual,
                    ),
                    "synthetic": True,
                }
            )
            miss_matrix = actual.copy()
            source, destination = np.argwhere(
                actual > 0
            )[0]
            miss_matrix[source, destination] -= 1
            specs.append(
                {
                    "kind": "miss",
                    "layer": layer,
                    "step": step,
                    "matrix": miss_matrix,
                    "synthetic": True,
                }
            )
            continue

        for kind, desired in (("hit", True), ("miss", False)):
            candidates = [
                step
                for step, hit in zip(
                    evaluation_steps,
                    layer_hits,
                )
                if hit == desired
            ]
            if candidates:
                specs.append(
                    {
                        "kind": kind,
                        "layer": layer,
                        "step": candidates[0],
                        "matrix": additive.copy(),
                        "synthetic": False,
                    }
                )
                continue
            step = evaluation_steps[0]
            actual = matrices[step, layer]
            matrix = np.maximum(additive, actual)
            if not desired:
                matrix = actual.copy()
                source, destination = np.argwhere(
                    actual > 0
                )[0]
                matrix[source, destination] -= 1
            specs.append(
                {
                    "kind": kind,
                    "layer": layer,
                    "step": step,
                    "matrix": matrix,
                    "synthetic": True,
                }
            )
    return specs, {
        "observed_hit_rate": statistics.fmean(all_hits),
        "observed_hits": sum(all_hits),
        "observed_steps": len(all_hits),
    }


def make_conditional_runner(
    case: dict[str, Any],
    *,
    rank: int,
    experts_per_rank: int,
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    tile_size: int,
    accumulation_dtype: torch.dtype,
) -> Callable[[], torch.Tensor]:
    profile = case["profile"]
    send_counts = case["send_counts"]
    recv_counts = case["recv_counts"]
    hidden = case["hidden"]
    topk_ids = case["topk_ids"]
    topk_weights = case["topk_weights"]
    num_tokens = hidden.shape[0]

    def static_branch(
        send_hidden: torch.Tensor,
        send_local_ids: torch.Tensor,
        send_weights: torch.Tensor,
        fixed_token_indices: torch.Tensor,
        branch_hidden: torch.Tensor,
        branch_ids: torch.Tensor,
        branch_weights: torch.Tensor,
        branch_w1: torch.Tensor,
        branch_w2: torch.Tensor,
    ) -> torch.Tensor:
        return run_static_functional(
            send_hidden,
            send_local_ids,
            send_weights,
            fixed_token_indices,
            send_counts=send_counts,
            recv_counts=recv_counts,
            local_w1=branch_w1,
            local_w2=branch_w2,
            num_tokens=num_tokens,
        )

    def fallback_branch(
        send_hidden: torch.Tensor,
        send_local_ids: torch.Tensor,
        send_weights: torch.Tensor,
        fixed_token_indices: torch.Tensor,
        branch_hidden: torch.Tensor,
        branch_ids: torch.Tensor,
        branch_weights: torch.Tensor,
        branch_w1: torch.Tensor,
        branch_w2: torch.Tensor,
    ) -> torch.Tensor:
        return run_gather_fallback(
            branch_hidden,
            branch_ids,
            branch_weights,
            branch_w1,
            branch_w2,
            rank=rank,
            experts_per_rank=experts_per_rank,
            accumulation_dtype=accumulation_dtype,
        )

    def execute() -> torch.Tensor:
        route = pack_triton_tiled(
            hidden,
            topk_ids,
            topk_weights,
            experts_per_rank=experts_per_rank,
            profile=profile,
            tile_size=tile_size,
        )
        overflow = funcol.all_reduce(
            route["overflow_count"],
            "sum",
            dist.group.WORLD,
        )
        return torch.cond(
            overflow == 0,
            static_branch,
            fallback_branch,
            (
                route["send_hidden"],
                route["send_local_ids"],
                route["send_weights"],
                route["fixed_token_indices"],
                hidden,
                topk_ids,
                topk_weights,
                local_w1,
                local_w2,
            ),
        )

    return execute


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    accumulation_dtype = (
        torch.bfloat16
        if args.fallback_accumulation == "bf16"
        else torch.float32
    )

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
    if args.profile == "smoke":
        selected_layers = [0]
        warmup_cycles, measured_cycles = 1, 2
    elif args.profile == "crossover":
        selected_layers = [0, 8, 15]
        warmup_cycles, measured_cycles = 1, 6
    else:
        selected_layers = [0, 4, 8, 12, 15]
        warmup_cycles, measured_cycles = 2, 12
    additive_margin = (
        args.margin_8
        if world_size <= 8
        else args.margin_24
    )
    case_specs, profile_statistics = build_case_specs(
        matrices,
        calibration_steps=calibration_steps,
        evaluation_steps=evaluation_steps,
        selected_layers=selected_layers,
        additive_margin=additive_margin,
        smoke=args.profile == "smoke",
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

    trace_counts = trace["expert_counts"]
    cases: list[dict[str, Any]] = []
    for spec in case_specs:
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
            300_000
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
        expected_hit = bool(
            np.all(matrices[step, layer] <= matrix)
        )
        if expected_hit != (spec["kind"] == "hit"):
            raise RuntimeError("Case fit label is inconsistent")
        capacity_row = [
            int(value)
            for value in matrix[rank].tolist()
        ]
        cases.append(
            {
                **spec,
                "domain": labels[step],
                "expected_hit": expected_hit,
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
            }
        )

    records: list[dict[str, Any]] = []
    quality_cases: dict[str, dict[str, dict[str, float]]] = {}
    method_names = (
        "dynamic",
        "ideal_precomputed",
        "gather_fallback",
        "precomputed_gather",
        "device_host_gather",
        "device_cond_gather",
    )
    if args.reverse_methods:
        method_names = tuple(reversed(method_names))

    for case in cases:
        hidden = case["hidden"]
        topk_ids = case["topk_ids"]
        topk_weights = case["topk_weights"]
        profile = case["profile"]
        recv_counts = case["recv_counts"]
        expected_hit = case["expected_hit"]

        def dynamic_case() -> torch.Tensor:
            route = pack_dynamic_from_topk(
                hidden,
                topk_ids,
                topk_weights,
                experts_per_rank=experts_per_rank,
                world_size=world_size,
            )
            return run_dynamic(
                route,
                local_w1,
                local_w2,
                num_tokens=num_tokens,
            )

        def static_case() -> torch.Tensor:
            return run_static_alltoall(
                hidden,
                topk_ids,
                topk_weights,
                experts_per_rank=experts_per_rank,
                profile=profile,
                recv_counts=recv_counts,
                local_w1=local_w1,
                local_w2=local_w2,
                tile_size=args.tile_size,
            )

        def gather_case() -> torch.Tensor:
            return run_gather_fallback(
                hidden,
                topk_ids,
                topk_weights,
                local_w1,
                local_w2,
                rank=rank,
                experts_per_rank=experts_per_rank,
                accumulation_dtype=accumulation_dtype,
            )

        def ideal_precomputed() -> torch.Tensor:
            return static_case() if expected_hit else dynamic_case()

        def precomputed_gather() -> torch.Tensor:
            return static_case() if expected_hit else gather_case()

        def device_host_gather() -> torch.Tensor:
            route = pack_triton_tiled(
                hidden,
                topk_ids,
                topk_weights,
                experts_per_rank=experts_per_rank,
                profile=profile,
                tile_size=args.tile_size,
            )
            overflow = funcol.all_reduce(
                route["overflow_count"],
                "sum",
                dist.group.WORLD,
            )
            if int(overflow.item()) != 0:
                return gather_case()
            return run_static_functional(
                route["send_hidden"],
                route["send_local_ids"],
                route["send_weights"],
                route["fixed_token_indices"],
                send_counts=case["send_counts"],
                recv_counts=recv_counts,
                local_w1=local_w1,
                local_w2=local_w2,
                num_tokens=num_tokens,
            )

        device_cond_gather = make_conditional_runner(
            case,
            rank=rank,
            experts_per_rank=experts_per_rank,
            local_w1=local_w1,
            local_w2=local_w2,
            tile_size=args.tile_size,
            accumulation_dtype=accumulation_dtype,
        )

        eager_warmup = dynamic_case()
        torch.cuda.synchronize()
        del eager_warmup
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize()
        compile_started = time.perf_counter()
        compiled_output = device_cond_gather()
        torch.cuda.synchronize()
        compile_ms = (
            time.perf_counter() - compile_started
        ) * 1_000.0
        compile_critical_ms = max(
            gather_scalar(compile_ms, device)
        )
        del compiled_output

        functions = {
            "dynamic": dynamic_case,
            "ideal_precomputed": ideal_precomputed,
            "gather_fallback": gather_case,
            "precomputed_gather": precomputed_gather,
            "device_host_gather": device_host_gather,
            "device_cond_gather": device_cond_gather,
        }
        expected = dynamic_case()
        case_key = (
            f"{case['kind']}_layer{case['layer']}"
            f"_step{case['step']}"
        )
        quality_cases[case_key] = {}
        for name in method_names:
            actual = functions[name]()
            local_quality = output_quality(actual, expected)
            quality_cases[case_key][name] = {
                "max_abs_error": max(
                    gather_scalar(
                        local_quality["max_abs_error"],
                        device,
                    )
                ),
                "normalized_rmse": max(
                    gather_scalar(
                        local_quality["normalized_rmse"],
                        device,
                    )
                ),
                "cosine_similarity": min(
                    gather_scalar(
                        local_quality["cosine_similarity"],
                        device,
                    )
                ),
            }
        del expected

        timings = timed_crossover(
            tuple(
                (
                    name,
                    functions[name],
                )
                for name in method_names
            ),
            warmup_cycles=warmup_cycles,
            measured_cycles=measured_cycles,
            local_rank=local_rank,
            device=device,
        )
        if rank == 0:
            dynamic_ms = timings["dynamic"][
                "critical_path_ms"
            ]
            records.append(
                {
                    "kind": case["kind"],
                    "layer": case["layer"],
                    "step": case["step"],
                    "domain": case["domain"],
                    "synthetic": case["synthetic"],
                    "expected_hit": expected_hit,
                    "profile_rows_per_source": sum(
                        case["send_counts"]
                    ),
                    "compile_critical_ms": (
                        compile_critical_ms
                    ),
                    "timings": timings,
                    "speedup_vs_dynamic": {
                        name: (
                            dynamic_ms
                            / timing["critical_path_ms"]
                        )
                        for name, timing in timings.items()
                        if name != "dynamic"
                    },
                }
            )

    max_error = max(
        metrics["max_abs_error"]
        for case_quality in quality_cases.values()
        for metrics in case_quality.values()
    )
    min_cosine = min(
        metrics["cosine_similarity"]
        for case_quality in quality_cases.values()
        for metrics in case_quality.values()
    )
    if (
        max_error > args.max_abs_error
        or min_cosine < 0.999
    ):
        raise RuntimeError(
            "Device-fit fallback correctness failed: "
            f"max_error={max_error}, "
            f"min_cosine={min_cosine}"
        )

    result = None
    if rank == 0:
        hit_records = [
            record
            for record in records
            if record["expected_hit"]
        ]
        miss_records = [
            record
            for record in records
            if not record["expected_hit"]
        ]

        def mean_speed(
            selected: list[dict[str, Any]],
            method: str,
        ) -> float:
            return statistics.fmean(
                record["speedup_vs_dynamic"][method]
                for record in selected
            )

        hit_rate = profile_statistics[
            "observed_hit_rate"
        ]
        conditional_hit_ms = statistics.fmean(
            record["timings"]["device_cond_gather"][
                "critical_path_ms"
            ]
            for record in hit_records
        )
        conditional_miss_ms = statistics.fmean(
            record["timings"]["device_cond_gather"][
                "critical_path_ms"
            ]
            for record in miss_records
        )
        dynamic_hit_ms = statistics.fmean(
            record["timings"]["dynamic"][
                "critical_path_ms"
            ]
            for record in hit_records
        )
        dynamic_miss_ms = statistics.fmean(
            record["timings"]["dynamic"][
                "critical_path_ms"
            ]
            for record in miss_records
        )
        expected_conditional_ms = (
            hit_rate * conditional_hit_ms
            + (1.0 - hit_rate) * conditional_miss_ms
        )
        expected_dynamic_ms = (
            hit_rate * dynamic_hit_ms
            + (1.0 - hit_rate) * dynamic_miss_ms
        )
        result = {
            "suite": "distributed_moe_device_fit",
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
            "tile_size": args.tile_size,
            "fallback_accumulation": (
                args.fallback_accumulation
            ),
            "fallback_dispatch_row_ratio": (
                world_size / top_k
            ),
            "fallback_return_byte_ratio": (
                world_size
                / top_k
                * (
                    1.0
                    if accumulation_dtype == torch.bfloat16
                    else 2.0
                )
            ),
            "profile_statistics": profile_statistics,
            "records": records,
            "mean_speedup_vs_dynamic": {
                "hit": {
                    method: mean_speed(
                        hit_records,
                        method,
                    )
                    for method in method_names
                    if method != "dynamic"
                },
                "miss": {
                    method: mean_speed(
                        miss_records,
                        method,
                    )
                    for method in method_names
                    if method != "dynamic"
                },
            },
            "expected_mix": {
                "hit_rate": hit_rate,
                "device_cond_gather_ms": (
                    expected_conditional_ms
                ),
                "dynamic_ms": expected_dynamic_ms,
                "speedup_vs_dynamic": (
                    expected_dynamic_ms
                    / expected_conditional_ms
                ),
            },
            "correctness": {
                "max_abs_error": max_error,
                "min_cosine_similarity": min_cosine,
                "max_allowed_abs_error": (
                    args.max_abs_error
                ),
                "cases": quality_cases,
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "The device predicate reuses the tiled packer's overflow scalar and a functional NCCL all-reduce; torch.cond consumes the global scalar without Python item() synchronization.",
                "The exact fallback all-gathers each source token once, filters assignments for local experts on device, and reduce-scatters token contributions back to source ranks.",
                "The fallback uses fixed collective signatures and requires no Python split extraction. Its hidden-row ratio versus dynamic assignment transport is world_size/top_k.",
                "ideal_precomputed matches the v33 assumption: host-known profile hits use fixed all-to-all and misses use the ordinary dynamic split path.",
                "The first device_cond_gather invocation is reported separately as compile_critical_ms and excluded from steady-state timing.",
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
        "--calibration-domains",
        type=int,
        default=4,
    )
    parser.add_argument("--margin-8", type=int, default=18)
    parser.add_argument("--margin-24", type=int, default=16)
    parser.add_argument(
        "--fallback-accumulation",
        choices=["bf16", "fp32"],
        default="bf16",
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
