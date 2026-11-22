from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    MAX_BATCH,
    aggregate_metrics,
    load_service_models,
    make_arrivals,
)
from simulate_prefill_decode_scheduling import (
    PREFIX_MODES,
    estimate_capacity as estimate_fixed_capacity,
    load_prefill_service_model,
    make_prefill_workload,
    prefill_chunk_service,
    simulate,
)


FIXED_POLICIES = {
    "fixed_monolithic": "monolithic_prefill_first",
    "fixed_2048": "chunk_2048_prefill_first",
    "fixed_1024": "chunk_1024_prefill_first",
    "fixed_512": "chunk_512_prefill_first",
}
ADAPTIVE_POLICIES = {
    "adaptive_50ms_age": {
        "base_budget_ms": 50.0,
        "middle_age_ms": 500.0,
        "middle_budget_ms": 100.0,
        "high_age_ms": 2_000.0,
        "high_budget_ms": 150.0,
    },
    "adaptive_100ms_age": {
        "base_budget_ms": 100.0,
        "middle_age_ms": 1_000.0,
        "middle_budget_ms": 150.0,
        "high_age_ms": 4_000.0,
        "high_budget_ms": 300.0,
    },
}
CHUNK_CANDIDATES = (
    6144,
    4096,
    2048,
    1024,
    512,
    256,
    128,
)


def percentile(
    values: list[float],
    q: float,
) -> float:
    return float(np.percentile(values, q))


def make_selector(
    *,
    policy: dict[str, float],
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], int],
    list[dict[str, float]],
]:
    decisions: list[dict[str, float]] = []

    def select(state: dict[str, Any]) -> int:
        remaining = int(
            state["remaining_tokens"]
        )
        prefix = int(state["prefix_tokens"])
        has_decode_work = (
            int(state["active_decode_requests"])
            + int(state["decode_ready_requests"])
            > 0
        )
        queue_age_ms = float(
            state["queue_age_ms"]
        )
        if not has_decode_work:
            query = remaining
            service_ms = prefill_chunk_service(
                prefill_model,
                query_tokens=query,
                prefix_tokens=prefix,
            )["total_ms"]
            decisions.append(
                {
                    "query_tokens": float(query),
                    "service_ms": service_ms,
                    "budget_ms": math.inf,
                    "level": -1.0,
                    "queue_age_ms": queue_age_ms,
                    "has_decode_work": 0.0,
                    "budget_violation": 0.0,
                }
            )
            return query

        if queue_age_ms >= policy["high_age_ms"]:
            budget_ms = policy["high_budget_ms"]
            level = 2
        elif queue_age_ms >= policy["middle_age_ms"]:
            budget_ms = policy[
                "middle_budget_ms"
            ]
            level = 1
        else:
            budget_ms = policy["base_budget_ms"]
            level = 0

        candidates = sorted(
            {
                min(remaining, candidate)
                for candidate in CHUNK_CANDIDATES
                if min(remaining, candidate) >= 1
            },
            reverse=True,
        )
        selected = candidates[-1]
        selected_service_ms = (
            prefill_chunk_service(
                prefill_model,
                query_tokens=selected,
                prefix_tokens=prefix,
            )["total_ms"]
        )
        for candidate in candidates:
            service_ms = prefill_chunk_service(
                prefill_model,
                query_tokens=candidate,
                prefix_tokens=prefix,
            )["total_ms"]
            if service_ms <= budget_ms:
                selected = candidate
                selected_service_ms = service_ms
                break
        decisions.append(
            {
                "query_tokens": float(selected),
                "service_ms": selected_service_ms,
                "budget_ms": budget_ms,
                "level": float(level),
                "queue_age_ms": queue_age_ms,
                "has_decode_work": 1.0,
                "budget_violation": float(
                    selected_service_ms
                    > budget_ms + 1e-12
                ),
            }
        )
        return selected

    return select, decisions


def decision_metrics(
    decisions: list[dict[str, float]],
) -> dict[str, float]:
    if not decisions:
        raise RuntimeError(
            "Adaptive policy made no decisions"
        )
    active = [
        decision
        for decision in decisions
        if decision["has_decode_work"] == 1.0
    ]
    idle = len(decisions) - len(active)
    if not active:
        raise RuntimeError(
            "Adaptive policy saw no decode work"
        )
    return {
        "mean_selected_chunk_tokens": float(
            statistics.mean(
                decision["query_tokens"]
                for decision in decisions
            )
        ),
        "p95_selected_chunk_tokens": (
            percentile(
                [
                    decision["query_tokens"]
                    for decision in decisions
                ],
                95,
            )
        ),
        "mean_active_slice_ms": float(
            statistics.mean(
                decision["service_ms"]
                for decision in active
            )
        ),
        "p95_active_slice_ms": percentile(
            [
                decision["service_ms"]
                for decision in active
            ],
            95,
        ),
        "p99_active_slice_ms": percentile(
            [
                decision["service_ms"]
                for decision in active
            ],
            99,
        ),
        "active_decode_slice_fraction": (
            len(active) / len(decisions)
        ),
        "idle_monolithic_fraction": (
            idle / len(decisions)
        ),
        "middle_escalation_fraction": float(
            np.mean(
                [
                    decision["level"] == 1.0
                    for decision in active
                ]
            )
        ),
        "high_escalation_fraction": float(
            np.mean(
                [
                    decision["level"] == 2.0
                    for decision in active
                ]
            )
        ),
        "any_escalation_fraction": float(
            np.mean(
                [
                    decision["level"] > 0.0
                    for decision in active
                ]
            )
        ),
        "slice_budget_violation_fraction": (
            float(
                np.mean(
                    [
                        decision[
                            "budget_violation"
                        ]
                        for decision in active
                    ]
                )
            )
        ),
        "mean_active_queue_age_ms": float(
            statistics.mean(
                decision["queue_age_ms"]
                for decision in active
            )
        ),
        "p95_active_queue_age_ms": percentile(
            [
                decision["queue_age_ms"]
                for decision in active
            ],
            95,
        ),
    }


def simulate_policy(
    *,
    arrivals: np.ndarray,
    workload: dict[str, np.ndarray],
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    policy_name: str,
    kv_budget_tokens: int,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    if policy_name in FIXED_POLICIES:
        return simulate(
            arrivals=arrivals,
            workload=workload,
            decode_model=decode_model,
            decode_service_models=(
                decode_service_models
            ),
            prefill_model=prefill_model,
            policy_name=FIXED_POLICIES[
                policy_name
            ],
            kv_budget_tokens=kv_budget_tokens,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )

    selector, decisions = make_selector(
        policy=ADAPTIVE_POLICIES[policy_name],
        prefill_model=prefill_model,
    )
    metrics = simulate(
        arrivals=arrivals,
        workload=workload,
        decode_model=decode_model,
        decode_service_models=(
            decode_service_models
        ),
        prefill_model=prefill_model,
        policy_name="chunk_1024_prefill_first",
        kv_budget_tokens=kv_budget_tokens,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        chunk_selector=selector,
    )
    metrics.update(
        decision_metrics(decisions)
    )
    return metrics


def estimate_capacity(
    *,
    num_requests: int,
    seed: int,
    prefix_mode: str,
    policy_name: str,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    kv_budget_tokens: int,
) -> dict[str, float]:
    if policy_name in FIXED_POLICIES:
        return estimate_fixed_capacity(
            num_requests=num_requests,
            seed=seed,
            prefix_mode=prefix_mode,
            policy_name=FIXED_POLICIES[
                policy_name
            ],
            decode_model=decode_model,
            decode_service_models=(
                decode_service_models
            ),
            prefill_model=prefill_model,
            kv_budget_tokens=kv_budget_tokens,
        )
    workload = make_prefill_workload(
        num_requests=num_requests,
        seed=seed,
        prefix_mode=prefix_mode,
    )
    metrics = simulate_policy(
        arrivals=np.zeros(
            num_requests,
            dtype=np.float64,
        ),
        workload=workload,
        decode_model=decode_model,
        decode_service_models=(
            decode_service_models
        ),
        prefill_model=prefill_model,
        policy_name=policy_name,
        kv_budget_tokens=kv_budget_tokens,
        analysis_start=0,
        analysis_end=num_requests,
    )
    busy_ms = (
        metrics["prefill_busy_ms"]
        + metrics["decode_busy_ms"]
    )
    return {
        "requests_per_second": (
            num_requests
            * 1_000.0
            / busy_ms
        ),
        "output_tokens_per_second": (
            metrics["completed_output_tokens"]
            * 1_000.0
            / busy_ms
        ),
        "prefill_chunk_overhead": metrics[
            "prefill_chunk_overhead"
        ],
        "prefix_compute_ratio_vs_no_cache": (
            metrics[
                "prefix_compute_ratio_vs_no_cache"
            ]
        ),
        "prefill_busy_fraction": metrics[
            "prefill_busy_fraction"
        ],
        "mean_prefill_chunks": metrics[
            "mean_prefill_chunks"
        ],
        "mean_selected_chunk_tokens": metrics[
            "mean_selected_chunk_tokens"
        ],
        "p95_active_slice_ms": metrics[
            "p95_active_slice_ms"
        ],
        "any_escalation_fraction": metrics[
            "any_escalation_fraction"
        ],
    }


def v40_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "prefill_decode_scheduling"
        / "prefill_decode_scheduling_full.json"
    )
    result = json.loads(
        path.read_text(encoding="utf-8")
    )
    return {
        (
            record["prefix_mode"],
            record[
                "offered_load_vs_decode_capacity"
            ],
            record["policy"],
        ): record
        for record in result["records"]
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    decode_service_models = load_service_models(
        args.root
    )
    prefill_model = load_prefill_service_model(
        args.root
    )
    kv_budget_tokens = int(
        args.kv_budget_gib
        * (1024**3)
        / KV_BYTES_PER_TOKEN
    )
    offered_loads = [
        float(value)
        for value in args.offered_loads.split(",")
    ]
    seeds = [
        args.seed + index
        for index in range(args.seeds)
    ]
    policy_names = (
        *FIXED_POLICIES,
        *ADAPTIVE_POLICIES,
    )

    decode_capacities = {}
    capacities: dict[str, dict[str, Any]] = {}
    for prefix_mode in PREFIX_MODES:
        capacities[prefix_mode] = {}
        for replicate_index, decode_model in enumerate(
            decode_service_models[
                "attention_replicates"
            ]
        ):
            source = decode_model["source"]
            from simulate_decode_continuous_batching import (
                simulate as simulate_decode_only,
            )
            from simulate_decode_continuous_batching import (
                make_workload,
            )

            decode_workload = make_workload(
                name="mixed_chat",
                num_requests=args.capacity_requests,
                seed=(
                    args.seed
                    + 500_000
                    + replicate_index * 10_000
                    + 1_000
                ),
            )
            decode_metrics = simulate_decode_only(
                arrivals=np.zeros(
                    args.capacity_requests,
                    dtype=np.float64,
                ),
                workload=decode_workload,
                model=decode_model,
                service_models=(
                    decode_service_models
                ),
                policy_name="ragged_fcfs",
                kv_budget_tokens=kv_budget_tokens,
                ttft_target_ms=250.0,
                analysis_start=0,
                analysis_end=(
                    args.capacity_requests
                ),
            )
            decode_capacities[source] = (
                args.capacity_requests
                * 1_000.0
                / decode_metrics["busy_ms"]
            )
            capacities[prefix_mode][source] = {}
            for policy_name in policy_names:
                capacities[prefix_mode][source][
                    policy_name
                ] = estimate_capacity(
                    num_requests=(
                        args.capacity_requests
                    ),
                    seed=(
                        args.seed
                        + 600_000
                        + replicate_index
                        * 10_000
                        + (
                            0
                            if prefix_mode == "none"
                            else 1_000
                        )
                    ),
                    prefix_mode=prefix_mode,
                    policy_name=policy_name,
                    decode_model=decode_model,
                    decode_service_models=(
                        decode_service_models
                    ),
                    prefill_model=prefill_model,
                    kv_budget_tokens=(
                        kv_budget_tokens
                    ),
                )

    records = []
    scenario_count = 0
    for prefix_mode in PREFIX_MODES:
        for offered_load in offered_loads:
            for policy_name in policy_names:
                scenarios = []
                for replicate_index, decode_model in enumerate(
                    decode_service_models[
                        "attention_replicates"
                    ]
                ):
                    source = decode_model["source"]
                    request_rate = (
                        decode_capacities[source]
                        * offered_load
                    )
                    for seed in seeds:
                        workload_seed = (
                            seed
                            + replicate_index
                            * 100_000
                            + (
                                0
                                if prefix_mode == "none"
                                else 10_000
                            )
                        )
                        workload = (
                            make_prefill_workload(
                                num_requests=(
                                    args.num_requests
                                ),
                                seed=workload_seed,
                                prefix_mode=(
                                    prefix_mode
                                ),
                            )
                        )
                        arrivals = make_arrivals(
                            num_requests=(
                                args.num_requests
                            ),
                            requests_per_second=(
                                request_rate
                            ),
                            seed=workload_seed + 1,
                        )
                        metrics = simulate_policy(
                            arrivals=arrivals,
                            workload=workload,
                            decode_model=decode_model,
                            decode_service_models=(
                                decode_service_models
                            ),
                            prefill_model=prefill_model,
                            policy_name=policy_name,
                            kv_budget_tokens=(
                                kv_budget_tokens
                            ),
                            analysis_start=(
                                args.warmup_requests
                            ),
                            analysis_end=(
                                args.num_requests
                                - args.tail_guard_requests
                            ),
                        )
                        scenarios.append(
                            {
                                "service_replicate": (
                                    source
                                ),
                                "seed": seed,
                                "request_rate_per_second": (
                                    request_rate
                                ),
                                "system_capacity_requests_s": (
                                    capacities[
                                        prefix_mode
                                    ][source][
                                        policy_name
                                    ][
                                        "requests_per_second"
                                    ]
                                ),
                                "offered_load_vs_system_capacity": (
                                    request_rate
                                    / capacities[
                                        prefix_mode
                                    ][source][
                                        policy_name
                                    ][
                                        "requests_per_second"
                                    ]
                                ),
                                **metrics,
                            }
                        )
                        scenario_count += 1
                records.append(
                    {
                        "prefix_mode": prefix_mode,
                        "offered_load_vs_decode_capacity": (
                            offered_load
                        ),
                        "policy": policy_name,
                        "configuration": (
                            FIXED_POLICIES.get(
                                policy_name
                            )
                            if policy_name
                            in FIXED_POLICIES
                            else ADAPTIVE_POLICIES[
                                policy_name
                            ]
                        ),
                        "metrics": aggregate_metrics(
                            scenarios
                        ),
                        "scenarios": scenarios,
                    }
                )

    reference = v40_records(args.root)
    equivalence = []
    equivalence_enabled = (
        args.num_requests == 2_000
        and args.warmup_requests == 200
        and args.tail_guard_requests == 100
        and args.capacity_requests == 1_500
        and args.seeds == 3
        and args.seed == 40_000
    )
    if equivalence_enabled:
        for record in records:
            policy_name = record["policy"]
            if policy_name not in FIXED_POLICIES:
                continue
            expected_record = reference[
                (
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    FIXED_POLICIES[policy_name],
                )
            ]
            expected_scenarios = {
                (
                    scenario["service_replicate"],
                    scenario["seed"],
                ): scenario
                for scenario in expected_record[
                    "scenarios"
                ]
            }
            for scenario in record["scenarios"]:
                key = (
                    scenario["service_replicate"],
                    scenario["seed"],
                )
                expected = expected_scenarios[key]
                maximum_difference = max(
                    abs(
                        scenario[metric]
                        - expected[metric]
                    )
                    for metric in (
                        "throughput_requests_s",
                        "throughput_output_tokens_s",
                        "p99_prefill_latency_ms",
                        "p99_ttft_ms",
                        "p99_tpot_ms",
                        "p99_max_intertoken_gap_ms",
                        "p99_e2e_ms",
                        "mean_batch_size",
                        "mean_prefill_chunks",
                        "prefill_chunk_overhead",
                        "prefill_busy_ms",
                        "decode_busy_ms",
                        "peak_kv_gib",
                        "completed_output_tokens",
                    )
                )
                if maximum_difference > 1e-9:
                    raise RuntimeError(
                        "v40 fixed-policy equivalence failed: "
                        f"{policy_name=} {key=} "
                        f"{maximum_difference=}"
                    )
                equivalence.append(
                    {
                        "prefix_mode": record[
                            "prefix_mode"
                        ],
                        "offered_load": record[
                            "offered_load_vs_decode_capacity"
                        ],
                        "policy": policy_name,
                        "service_replicate": key[0],
                        "seed": key[1],
                        "maximum_absolute_difference": (
                            maximum_difference
                        ),
                    }
                )

    result = {
        "suite": (
            "adaptive_prefill_time_slice_simulation"
        ),
        "model": (
            "The v40 measured-service prefill/decode "
            "model with online chunk selection from "
            "predicted prefill service time and oldest "
            "prefill queue age."
        ),
        "prefill_service_model": prefill_model,
        "decode_service_models": (
            decode_service_models
        ),
        "workload": {
            "base": "mixed_chat",
            "prefix_modes": PREFIX_MODES,
        },
        "fixed_policies": FIXED_POLICIES,
        "adaptive_policies": ADAPTIVE_POLICIES,
        "chunk_candidates": CHUNK_CANDIDATES,
        "simulation": {
            "num_requests": args.num_requests,
            "warmup_requests": args.warmup_requests,
            "tail_guard_requests": (
                args.tail_guard_requests
            ),
            "capacity_requests": (
                args.capacity_requests
            ),
            "seeds": seeds,
            "offered_loads_vs_decode_capacity": (
                offered_loads
            ),
            "max_batch": MAX_BATCH,
            "kv_budget_gib": args.kv_budget_gib,
            "kv_budget_tokens": kv_budget_tokens,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        },
        "decode_only_capacities": (
            decode_capacities
        ),
        "system_capacities": capacities,
        "validation": {
            "scenario_count": scenario_count,
            "v40_equivalence_enabled": (
                equivalence_enabled
            ),
            "v40_equivalence_checks": (
                equivalence
            ),
            "v40_maximum_absolute_difference": (
                max(
                    item[
                        "maximum_absolute_difference"
                    ]
                    for item in equivalence
                )
                if equivalence
                else None
            ),
            "accounting": (
                "The v40 simulator continues to assert "
                "exact prefill/decode tokens, exact "
                "causal pairs, finite completion, max "
                "batch 64, and KV within budget. The "
                "adaptive selector additionally asserts "
                "valid upward service-time choices and "
                "zero slice-budget violations."
            ),
        },
        "records": records,
        "limitations": [
            "The underlying prefill/decode service model inherits all v40 limitations and is not a live serving-engine measurement.",
            "The controller uses exact queue age and predicted kernel service from the fitted model. Production estimates will have noise and CPU overhead.",
            "When no decode work is ready, the controller uses the remaining prompt monolithically. It does not reserve periodic decode slots for future arrivals.",
            "Queue-age thresholds and service budgets are hand-designed controls, not optimized against a training trace or formal SLO objective.",
            "All policies remain FCFS within the prefill queue and use one prefill action per decode action. Tenant fairness and request priorities are excluded.",
        ],
    }
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = args.output.with_suffix(
        args.output.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--offered-loads",
        default="0.1,0.2,0.3,0.4",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--tail-guard-requests",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--capacity-requests",
        type=int,
        default=1_500,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=40_000,
    )
    parser.add_argument(
        "--kv-budget-gib",
        type=float,
        default=32.0,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
