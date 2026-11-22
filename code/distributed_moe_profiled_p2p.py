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

from analyze_router_profile_drift import (
    build_profile,
    destination_steps,
)
from distributed_moe_end_to_end import (
    TensorDict,
    gather_scalar,
    local_experts,
    output_quality,
)
from distributed_moe_packed_dispatch import (
    build_metadata_packet,
    timed_crossover,
)
from distributed_moe_profiled_packer import (
    profile_tensors,
)
from distributed_moe_profiled_packer_scaling import (
    pack_triton_tiled,
)
from distributed_moe_profiled_spill import (
    dispatch_static_metadata,
    pack_dynamic_from_topk,
    reconstruct_topk_ids,
    remap_expert_ids,
    return_static_fp32,
    run_dynamic,
    trace_step_indices,
)


def offsets(counts: list[int]) -> list[int]:
    result: list[int] = []
    current = 0
    for count in counts:
        result.append(current)
        current += count
    return result


def dispatch_static_p2p(
    route: TensorDict,
    recv_counts: list[int],
) -> TensorDict:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    send_hidden = route["send_hidden"]
    send_counts = route["send_counts"]
    assert isinstance(send_hidden, torch.Tensor)
    assert isinstance(send_counts, list)
    if len(send_counts) != world_size:
        raise ValueError("Invalid P2P send-count vector")

    send_metadata = build_metadata_packet(route)
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
    send_offsets = offsets(send_counts)
    recv_offsets = offsets(recv_counts)
    operations: list[dist.P2POp] = []
    for peer in range(world_size):
        send_count = send_counts[peer]
        recv_count = recv_counts[peer]
        send_offset = send_offsets[peer]
        recv_offset = recv_offsets[peer]
        if peer == rank:
            if send_count != recv_count:
                raise RuntimeError(
                    "Self P2P counts must match"
                )
            recv_hidden[
                recv_offset : recv_offset + recv_count
            ].copy_(
                send_hidden[
                    send_offset : send_offset + send_count
                ]
            )
            recv_metadata[
                recv_offset : recv_offset + recv_count
            ].copy_(
                send_metadata[
                    send_offset : send_offset + send_count
                ]
            )
            continue
        if send_count:
            operations.append(
                dist.P2POp(
                    dist.isend,
                    send_hidden[
                        send_offset : send_offset + send_count
                    ],
                    peer,
                )
            )
        if recv_count:
            operations.append(
                dist.P2POp(
                    dist.irecv,
                    recv_hidden[
                        recv_offset : recv_offset + recv_count
                    ],
                    peer,
                )
            )
        if send_count:
            operations.append(
                dist.P2POp(
                    dist.isend,
                    send_metadata[
                        send_offset : send_offset + send_count
                    ],
                    peer,
                )
            )
        if recv_count:
            operations.append(
                dist.P2POp(
                    dist.irecv,
                    recv_metadata[
                        recv_offset : recv_offset + recv_count
                    ],
                    peer,
                )
            )
    if operations:
        requests = dist.batch_isend_irecv(operations)
        for request in requests:
            request.wait()
    return {
        "recv_hidden": recv_hidden,
        "recv_local_ids": (
            recv_metadata[:, :1].to(torch.int32).contiguous()
        ),
        "recv_weights": recv_metadata[:, 1:].contiguous(),
        "recv_counts": list(recv_counts),
    }


def return_static_p2p_fp32(
    expert_output: torch.Tensor,
    route: TensorDict,
    recv_counts: list[int],
    *,
    num_tokens: int,
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    send_counts = route["send_counts"]
    fixed_token_indices = route["fixed_token_indices"]
    assert isinstance(send_counts, list)
    assert isinstance(fixed_token_indices, torch.Tensor)

    returned = torch.empty(
        (sum(send_counts), expert_output.shape[1]),
        device=expert_output.device,
        dtype=expert_output.dtype,
    )
    output_offsets = offsets(send_counts)
    expert_offsets = offsets(recv_counts)
    operations: list[dist.P2POp] = []
    for peer in range(world_size):
        expert_count = recv_counts[peer]
        returned_count = send_counts[peer]
        expert_offset = expert_offsets[peer]
        output_offset = output_offsets[peer]
        if peer == rank:
            if expert_count != returned_count:
                raise RuntimeError(
                    "Self return counts must match"
                )
            returned[
                output_offset : output_offset + returned_count
            ].copy_(
                expert_output[
                    expert_offset : expert_offset + expert_count
                ]
            )
            continue
        if expert_count:
            operations.append(
                dist.P2POp(
                    dist.isend,
                    expert_output[
                        expert_offset : expert_offset
                        + expert_count
                    ],
                    peer,
                )
            )
        if returned_count:
            operations.append(
                dist.P2POp(
                    dist.irecv,
                    returned[
                        output_offset : output_offset
                        + returned_count
                    ],
                    peer,
                )
            )
    if operations:
        requests = dist.batch_isend_irecv(operations)
        for request in requests:
            request.wait()

    combined = torch.zeros(
        (num_tokens, returned.shape[1]),
        device=returned.device,
        dtype=torch.float32,
    )
    combined.index_add_(
        0,
        fixed_token_indices,
        returned.float(),
    )
    return combined


def run_static_alltoall(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    profile: dict[str, torch.Tensor | int | list[int]],
    recv_counts: list[int],
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    route = pack_triton_tiled(
        hidden_states,
        topk_ids,
        topk_weights,
        experts_per_rank=experts_per_rank,
        profile=profile,
        tile_size=tile_size,
    )
    dispatched = dispatch_static_metadata(
        route,
        recv_counts,
    )
    expert_output = local_experts(
        dispatched,
        local_w1,
        local_w2,
    )
    return return_static_fp32(
        expert_output,
        route,
        recv_counts,
        num_tokens=hidden_states.shape[0],
    ).to(hidden_states.dtype)


def run_static_p2p(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    profile: dict[str, torch.Tensor | int | list[int]],
    recv_counts: list[int],
    local_w1: torch.Tensor,
    local_w2: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    route = pack_triton_tiled(
        hidden_states,
        topk_ids,
        topk_weights,
        experts_per_rank=experts_per_rank,
        profile=profile,
        tile_size=tile_size,
    )
    dispatched = dispatch_static_p2p(
        route,
        recv_counts,
    )
    expert_output = local_experts(
        dispatched,
        local_w1,
        local_w2,
    )
    return return_static_p2p_fp32(
        expert_output,
        route,
        recv_counts,
        num_tokens=hidden_states.shape[0],
    ).to(hidden_states.dtype)


def build_profile_schedules(
    matrices: np.ndarray,
    *,
    calibration_steps: int,
    selected_layers: list[int],
    selected_steps: list[int],
    assignments_per_source: int,
    additive_margin: int,
    adaptive_budget: float,
    adaptive_window: int,
) -> dict[str, list[np.ndarray]]:
    schedules = {
        "additive": [],
        "adaptive": [],
    }
    for layer in selected_layers:
        calibration = matrices[:calibration_steps, layer]
        additive = calibration.max(axis=0) + additive_margin
        history = [
            matrix.copy()
            for matrix in calibration
        ]
        adaptive_profiles: dict[int, np.ndarray] = {}
        for step in range(calibration_steps, matrices.shape[0]):
            window = np.stack(
                history[-adaptive_window:],
                axis=0,
            )
            adaptive_profiles[step] = build_profile(
                window,
                assignments_per_source=assignments_per_source,
                budget_fraction=adaptive_budget,
                risk_weighted=False,
            )
            history.append(matrices[step, layer].copy())
        for step in selected_steps:
            schedules["additive"].append(additive)
            schedules["adaptive"].append(
                adaptive_profiles[step]
            )
    return schedules


def profile_statistics(
    matrices: np.ndarray,
    *,
    selected_layers: list[int],
    selected_steps: list[int],
    schedules: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for policy, profiles in schedules.items():
        hits: list[float] = []
        served_ratios: list[float] = []
        hit_fixed_ratios: list[float] = []
        index = 0
        for layer in selected_layers:
            for step in selected_steps:
                actual = matrices[step, layer]
                profile = profiles[index]
                hit = bool(np.all(actual <= profile))
                actual_rows = int(actual.sum())
                fixed_rows = int(profile.sum())
                hits.append(float(hit))
                served_ratios.append(
                    (
                        fixed_rows
                        if hit
                        else actual_rows
                    )
                    / actual_rows
                )
                if hit:
                    hit_fixed_ratios.append(
                        fixed_rows / actual_rows
                    )
                index += 1
        result[policy] = {
            "hit_rate": statistics.fmean(hits),
            "fallback_rate": 1.0 - statistics.fmean(hits),
            "mean_served_row_ratio": statistics.fmean(
                served_ratios
            ),
            "mean_hit_fixed_row_ratio": (
                statistics.fmean(hit_fixed_ratios)
                if hit_fixed_ratios
                else 0.0
            ),
        }
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
    labels, matrices = destination_steps(
        trace,
        world_size,
    )
    _, sample_indices = trace_step_indices(
        trace,
        world_size,
    )
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
        selected_steps = evaluation_steps[:2]
        warmup_cycles, measured_cycles = 1, 2
    elif args.profile == "crossover":
        selected_layers = [0, 8, 15]
        selected_steps = evaluation_steps
        warmup_cycles, measured_cycles = 1, 4
    else:
        selected_layers = [0, 4, 8, 12, 15]
        selected_steps = evaluation_steps
        warmup_cycles, measured_cycles = 1, 3

    assignments_per_source = (
        int(trace["sequence_length"])
        * int(trace["top_k"])
    )
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
    schedules = build_profile_schedules(
        matrices,
        calibration_steps=calibration_steps,
        selected_layers=selected_layers,
        selected_steps=selected_steps,
        assignments_per_source=assignments_per_source,
        additive_margin=additive_margin,
        adaptive_budget=adaptive_budget,
        adaptive_window=adaptive_window,
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

    cases: list[dict[str, Any]] = []
    schedule_index = 0
    trace_counts = trace["expert_counts"]
    for layer in selected_layers:
        for step in selected_steps:
            sample = sample_indices[step][rank]
            original_ids = reconstruct_topk_ids(
                trace_counts[sample][layer],
                num_tokens=num_tokens,
                top_k=top_k,
            )
            topk_ids, actual_experts_per_rank = (
                remap_expert_ids(
                    original_ids,
                    num_trace_experts=num_trace_experts,
                    world_size=world_size,
                )
            )
            if actual_experts_per_rank != experts_per_rank:
                raise RuntimeError(
                    "Inconsistent local expert count"
                )
            topk_ids = topk_ids.to(device)
            topk_weights = torch.full(
                topk_ids.shape,
                1.0 / top_k,
                device=device,
                dtype=dtype,
            )
            hidden_generator = torch.Generator(device=device)
            hidden_generator.manual_seed(
                200_000
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
            policy_data = {}
            actual_matrix = matrices[step, layer]
            for policy, profiles in schedules.items():
                matrix = profiles[schedule_index]
                hit = bool(np.all(actual_matrix <= matrix))
                capacity_row = [
                    int(value)
                    for value in matrix[rank].tolist()
                ]
                policy_data[policy] = {
                    "hit": hit,
                    "profile": profile_tensors(
                        capacity_row,
                        device,
                    ),
                    "recv_counts": [
                        int(matrix[source, rank])
                        for source in range(world_size)
                    ],
                }
            cases.append(
                {
                    "layer": layer,
                    "step": step,
                    "domain": labels[step],
                    "hidden": hidden,
                    "topk_ids": topk_ids,
                    "topk_weights": topk_weights,
                    "policies": policy_data,
                }
            )
            schedule_index += 1

    def dynamic_case(case: dict[str, Any]) -> torch.Tensor:
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

    def policy_case(
        case: dict[str, Any],
        *,
        policy: str,
        p2p: bool,
    ) -> torch.Tensor:
        policy_data = case["policies"][policy]
        if not policy_data["hit"]:
            return dynamic_case(case)
        runner = (
            run_static_p2p
            if p2p
            else run_static_alltoall
        )
        return runner(
            case["hidden"],
            case["topk_ids"],
            case["topk_weights"],
            experts_per_rank=experts_per_rank,
            profile=policy_data["profile"],
            recv_counts=policy_data["recv_counts"],
            local_w1=local_w1,
            local_w2=local_w2,
            tile_size=args.tile_size,
        )

    quality: dict[str, dict[str, float]] = {}
    for case in cases:
        expected = dynamic_case(case)
        for name, policy, p2p in (
            ("additive_alltoall", "additive", False),
            ("additive_p2p", "additive", True),
            ("adaptive_alltoall", "adaptive", False),
            ("adaptive_p2p", "adaptive", True),
        ):
            actual = policy_case(
                case,
                policy=policy,
                p2p=p2p,
            )
            local_quality = output_quality(actual, expected)
            quality[
                f"{name}_layer{case['layer']}_step{case['step']}"
            ] = {
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
    max_error = max(
        item["max_abs_error"]
        for item in quality.values()
    )
    min_cosine = min(
        item["cosine_similarity"]
        for item in quality.values()
    )
    if (
        max_error > args.max_abs_error
        or min_cosine < 0.999
    ):
        raise RuntimeError(
            "P2P transport correctness failed: "
            f"max_error={max_error}, "
            f"min_cosine={min_cosine}"
        )

    def cycle(
        method: str,
    ) -> torch.Tensor:
        output = torch.empty(
            0,
            device=device,
            dtype=dtype,
        )
        for case in cases:
            if method == "dynamic":
                output = dynamic_case(case)
            elif method == "additive_alltoall":
                output = policy_case(
                    case,
                    policy="additive",
                    p2p=False,
                )
            elif method == "additive_p2p":
                output = policy_case(
                    case,
                    policy="additive",
                    p2p=True,
                )
            elif method == "adaptive_alltoall":
                output = policy_case(
                    case,
                    policy="adaptive",
                    p2p=False,
                )
            elif method == "adaptive_p2p":
                output = policy_case(
                    case,
                    policy="adaptive",
                    p2p=True,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        return output

    method_names = (
        "dynamic",
        "additive_alltoall",
        "additive_p2p",
        "adaptive_alltoall",
        "adaptive_p2p",
    )
    if args.reverse_methods:
        method_names = tuple(reversed(method_names))
    timings = timed_crossover(
        tuple(
            (
                method,
                lambda selected=method: cycle(selected),
            )
            for method in method_names
        ),
        warmup_cycles=warmup_cycles,
        measured_cycles=measured_cycles,
        local_rank=local_rank,
        device=device,
    )

    result = None
    if rank == 0:
        profile_stats = profile_statistics(
            matrices,
            selected_layers=selected_layers,
            selected_steps=selected_steps,
            schedules=schedules,
        )
        dynamic_ms = timings["dynamic"]["critical_path_ms"]
        result = {
            "suite": "distributed_moe_profiled_p2p",
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
            "selected_steps": selected_steps,
            "selected_domains": [
                labels[step]
                for step in selected_steps
            ],
            "calibration_steps": calibration_steps,
            "adaptive_window": adaptive_window,
            "adaptive_budget": adaptive_budget,
            "additive_margin": additive_margin,
            "tile_size": args.tile_size,
            "num_cycle_cases": len(cases),
            "profile_statistics": profile_stats,
            "timings": timings,
            "speedup_vs_dynamic": {
                method: (
                    dynamic_ms
                    / timing["critical_path_ms"]
                )
                for method, timing in timings.items()
                if method != "dynamic"
            },
            "p2p_vs_alltoall": {
                "additive": (
                    timings["additive_alltoall"][
                        "critical_path_ms"
                    ]
                    / timings["additive_p2p"][
                        "critical_path_ms"
                    ]
                ),
                "adaptive": (
                    timings["adaptive_alltoall"][
                        "critical_path_ms"
                    ]
                    / timings["adaptive_p2p"][
                        "critical_path_ms"
                    ]
                ),
            },
            "correctness": {
                "max_abs_error": max_error,
                "min_cosine_similarity": min_cosine,
                "max_allowed_abs_error": args.max_abs_error,
                "cases": quality,
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "vllm": importlib.metadata.version("vllm"),
            "gpu_name": torch.cuda.get_device_name(0),
            "notes": [
                "Oversized NCCL receives with smaller sends were rejected by a prior timeout gate; grouped P2P therefore uses exact fixed profile sizes only.",
                "On a profile miss the method executes the fully dynamic metadata-packed all-to-all path rather than clipping or spilling.",
                "Grouped P2P batches fixed-size hidden and metadata sends/receives into one dispatch group and one return group. Self traffic is copied locally.",
                "Router inference, profile construction, and profile-hit decisions are precomputed outside timed execution.",
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
        choices=["smoke", "full", "crossover"],
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
    parser.add_argument("--margin-8", type=int, default=18)
    parser.add_argument("--margin-24", type=int, default=16)
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
