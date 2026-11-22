from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def allocate_integer(
    weights: np.ndarray,
    budget: int,
) -> np.ndarray:
    if budget <= 0:
        return np.zeros_like(weights, dtype=np.int64)
    total = float(weights.sum())
    if total <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64)
        total = float(weights.sum())
    exact = weights.astype(np.float64) * budget / total
    allocation = np.floor(exact).astype(np.int64)
    remainder = budget - int(allocation.sum())
    if remainder:
        order = np.argsort(
            -(exact - allocation),
            kind="stable",
        )
        allocation[order[:remainder]] += 1
    return allocation


def build_profile(
    history: np.ndarray,
    *,
    assignments_per_source: int,
    budget_fraction: float,
    risk_weighted: bool,
) -> np.ndarray:
    base = history.max(axis=0).astype(np.int64)
    budget = math.ceil(
        assignments_per_source * budget_fraction
    )
    profile = base.copy()
    for source in range(profile.shape[0]):
        if risk_weighted:
            counts = history[:, source, :]
            weights = (
                counts.std(axis=0)
                + np.sqrt(counts.mean(axis=0) + 1.0)
            )
        else:
            weights = np.ones(
                profile.shape[1],
                dtype=np.float64,
            )
        profile[source] += allocate_integer(
            weights,
            budget,
        )
    return profile


def expert_to_destination(
    num_experts: int,
    world_size: int,
) -> np.ndarray:
    expert_ids = np.arange(num_experts, dtype=np.int64)
    return np.minimum(
        expert_ids * world_size // num_experts,
        world_size - 1,
    )


def destination_steps(
    trace: dict[str, Any],
    world_size: int,
) -> tuple[list[str], np.ndarray]:
    expert_counts = np.asarray(
        trace["expert_counts"],
        dtype=np.int64,
    )
    num_samples, num_layers, num_experts = (
        expert_counts.shape
    )
    mapping = expert_to_destination(
        num_experts,
        world_size,
    )
    destination_counts = np.zeros(
        (
            num_samples,
            num_layers,
            world_size,
        ),
        dtype=np.int64,
    )
    for expert, destination in enumerate(mapping):
        destination_counts[:, :, destination] += (
            expert_counts[:, :, expert]
        )

    labels: list[str] = []
    matrices: list[np.ndarray] = []
    for domain_range in trace["domain_ranges"]:
        domain = domain_range["domain"]
        start = int(domain_range["start"])
        stop = int(domain_range["stop"])
        if (stop - start) % world_size:
            raise ValueError(
                f"{domain} sample count is not divisible "
                f"by world size {world_size}"
            )
        for offset in range(start, stop, world_size):
            batch = destination_counts[
                offset : offset + world_size
            ]
            matrices.append(
                np.transpose(batch, (1, 0, 2))
            )
            labels.append(domain)
    return labels, np.stack(matrices, axis=0)


def evaluate_policy(
    matrices: np.ndarray,
    labels: list[str],
    *,
    calibration_steps: int,
    assignments_per_source: int,
    mode: str,
    budget_fraction: float = 0.0,
    additive_margin: int = 0,
    adaptive_window: int = 0,
) -> dict[str, Any]:
    _, num_layers, world_size, _ = matrices.shape
    records: list[dict[str, Any]] = []
    for layer in range(num_layers):
        calibration = matrices[:calibration_steps, layer]
        if mode == "additive":
            static_profile = (
                calibration.max(axis=0) + additive_margin
            )
        elif mode in {"risk_static", "uniform_static"}:
            static_profile = build_profile(
                calibration,
                assignments_per_source=(
                    assignments_per_source
                ),
                budget_fraction=budget_fraction,
                risk_weighted=mode == "risk_static",
            )
        else:
            static_profile = None

        history = [
            matrix.copy()
            for matrix in calibration
        ]
        for step in range(calibration_steps, len(labels)):
            actual = matrices[step, layer]
            if mode in {"risk_adaptive", "uniform_adaptive"}:
                window = np.stack(
                    history[-adaptive_window:],
                    axis=0,
                )
                profile = build_profile(
                    window,
                    assignments_per_source=(
                        assignments_per_source
                    ),
                    budget_fraction=budget_fraction,
                    risk_weighted=(
                        mode == "risk_adaptive"
                    ),
                )
            else:
                assert static_profile is not None
                profile = static_profile

            excess = np.maximum(actual - profile, 0)
            hit = int(excess.sum()) == 0
            actual_rows = int(actual.sum())
            fixed_rows = int(profile.sum())
            records.append(
                {
                    "layer": layer,
                    "step": step,
                    "domain": labels[step],
                    "hit": hit,
                    "miss_cells": int((excess > 0).sum()),
                    "overflow_assignments": int(excess.sum()),
                    "max_cell_overflow": int(excess.max()),
                    "actual_rows": actual_rows,
                    "fixed_rows": fixed_rows,
                    "planned_padding_fraction": (
                        1.0 - actual_rows / fixed_rows
                    ),
                    "served_rows": (
                        fixed_rows if hit else actual_rows
                    ),
                }
            )
            history.append(actual.copy())

    misses = [record for record in records if not record["hit"]]
    hits = [record for record in records if record["hit"]]
    overflow_values = [
        record["overflow_assignments"]
        for record in records
    ]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for record in records:
        by_domain[record["domain"]].append(record)
    return {
        "mode": mode,
        "budget_fraction": budget_fraction,
        "additive_margin": additive_margin,
        "adaptive_window": adaptive_window,
        "layer_steps": len(records),
        "hit_rate": statistics.fmean(
            float(record["hit"])
            for record in records
        ),
        "fallback_rate": statistics.fmean(
            float(not record["hit"])
            for record in records
        ),
        "mean_planned_padding_fraction": statistics.fmean(
            record["planned_padding_fraction"]
            for record in records
        ),
        "mean_hit_padding_fraction": (
            statistics.fmean(
                record["planned_padding_fraction"]
                for record in hits
            )
            if hits
            else None
        ),
        "mean_served_row_ratio": statistics.fmean(
            record["served_rows"] / record["actual_rows"]
            for record in records
        ),
        "mean_spill_row_ratio": statistics.fmean(
            (
                record["fixed_rows"]
                + record["overflow_assignments"]
            )
            / record["actual_rows"]
            for record in records
        ),
        "mean_overflow_fraction": statistics.fmean(
            record["overflow_assignments"]
            / record["actual_rows"]
            for record in records
        ),
        "p95_overflow_assignments": float(
            np.quantile(
                np.asarray(
                    overflow_values,
                    dtype=np.float64,
                ),
                0.95,
            )
        ),
        "max_overflow_assignments": max(
            record["overflow_assignments"]
            for record in records
        ),
        "mean_miss_overflow_assignments": (
            statistics.fmean(
                record["overflow_assignments"]
                for record in misses
            )
            if misses
            else 0.0
        ),
        "max_cell_overflow": max(
            record["max_cell_overflow"]
            for record in records
        ),
        "domain_fallback_rate": {
            domain: statistics.fmean(
                float(not record["hit"])
                for record in domain_records
            )
            for domain, domain_records in by_domain.items()
        },
        "records": records,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    trace = json.loads(Path(args.trace).read_text())
    valid_tokens = trace["valid_tokens"]
    if len(set(valid_tokens)) != 1:
        raise ValueError(
            "Trace samples must have a fixed valid-token count"
        )
    assignments_per_source = (
        int(valid_tokens[0]) * int(trace["top_k"])
    )
    calibration_domains = int(args.calibration_domains)
    policies = []
    for world_size in args.world_sizes:
        labels, matrices = destination_steps(
            trace,
            world_size,
        )
        steps_per_domain = (
            int(trace["samples_per_domain"]) // world_size
        )
        calibration_steps = (
            calibration_domains * steps_per_domain
        )
        adaptive_window = (
            args.adaptive_window_domains
            * steps_per_domain
        )
        world_policies = [
            evaluate_policy(
                matrices,
                labels,
                calibration_steps=calibration_steps,
                assignments_per_source=(
                    assignments_per_source
                ),
                mode="additive",
                additive_margin=(
                    args.margin_8
                    if world_size == 8
                    else args.margin_24
                ),
            )
        ]
        for budget_fraction in args.budget_fractions:
            for mode in (
                "uniform_static",
                "risk_static",
                "uniform_adaptive",
                "risk_adaptive",
            ):
                world_policies.append(
                    evaluate_policy(
                        matrices,
                        labels,
                        calibration_steps=(
                            calibration_steps
                        ),
                        assignments_per_source=(
                            assignments_per_source
                        ),
                        mode=mode,
                        budget_fraction=budget_fraction,
                        adaptive_window=adaptive_window,
                    )
                )
        policies.append(
            {
                "world_size": world_size,
                "steps": len(labels),
                "steps_per_domain": steps_per_domain,
                "calibration_steps": calibration_steps,
                "evaluation_steps": (
                    len(labels) - calibration_steps
                ),
                "adaptive_window": adaptive_window,
                "expert_to_destination": (
                    expert_to_destination(
                        int(trace["num_experts"]),
                        world_size,
                    ).tolist()
                ),
                "policies": world_policies,
            }
        )

    result = {
        "suite": "trained_router_profile_drift",
        "trace_suite": trace["suite"],
        "model": trace["model"],
        "corpus_sha256": trace["corpus_sha256"],
        "num_samples": trace["num_samples"],
        "num_layers": trace["num_layers"],
        "num_experts": trace["num_experts"],
        "top_k": trace["top_k"],
        "sequence_length": trace["sequence_length"],
        "assignments_per_source": assignments_per_source,
        "calibration_domains": calibration_domains,
        "adaptive_window_domains": (
            args.adaptive_window_domains
        ),
        "budget_fractions": args.budget_fractions,
        "worlds": policies,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--world-sizes",
        type=int,
        nargs="+",
        default=[8, 24],
    )
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
        "--budget-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.04, 0.08, 0.16],
    )
    parser.add_argument("--margin-8", type=int, default=18)
    parser.add_argument("--margin-24", type=int, default=16)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
