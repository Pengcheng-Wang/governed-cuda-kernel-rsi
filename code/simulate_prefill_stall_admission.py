from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simulate_adaptive_prefill_scheduling import (
    ADAPTIVE_POLICIES,
    CHUNK_CANDIDATES,
    decision_metrics,
    make_selector,
)
from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    MAX_BATCH,
    aggregate_metrics,
    load_service_models,
    make_arrivals,
)
from simulate_prefill_decode_scheduling import (
    PREFIX_MODES,
    load_prefill_service_model,
    make_prefill_workload,
    prefill_chunk_service,
    simulate,
)


POLICIES = {
    "fixed_1024": {
        "kind": "fixed",
        "base_policy": "chunk_1024_prefill_first",
    },
    "adaptive_100ms_age": {
        "kind": "adaptive",
        "selector": ADAPTIVE_POLICIES[
            "adaptive_100ms_age"
        ],
    },
    "hard_100ms": {
        "kind": "hard",
        "slice_budget_ms": 100.0,
        "backlog_limit_ms": None,
    },
    "hard_100ms_admit_2s": {
        "kind": "hard",
        "slice_budget_ms": 100.0,
        "backlog_limit_ms": 2_000.0,
    },
    "hard_100ms_admit_4s": {
        "kind": "hard",
        "slice_budget_ms": 100.0,
        "backlog_limit_ms": 4_000.0,
    },
}


def hard_selector_config(
    budget_ms: float,
) -> dict[str, float]:
    return {
        "base_budget_ms": budget_ms,
        "middle_age_ms": math.inf,
        "middle_budget_ms": budget_ms,
        "high_age_ms": math.inf,
        "high_budget_ms": budget_ms,
    }


def choose_chunk(
    *,
    remaining_tokens: int,
    prefix_tokens: int,
    budget_ms: float,
    prefill_model: dict[str, Any],
) -> tuple[int, float]:
    candidates = sorted(
        {
            min(remaining_tokens, candidate)
            for candidate in CHUNK_CANDIDATES
            if min(remaining_tokens, candidate)
            >= 1
        },
        reverse=True,
    )
    selected = candidates[-1]
    selected_service_ms = (
        prefill_chunk_service(
            prefill_model,
            query_tokens=selected,
            prefix_tokens=prefix_tokens,
        )["total_ms"]
    )
    for candidate in candidates:
        service_ms = prefill_chunk_service(
            prefill_model,
            query_tokens=candidate,
            prefix_tokens=prefix_tokens,
        )["total_ms"]
        if service_ms <= budget_ms:
            selected = candidate
            selected_service_ms = service_ms
            break
    return selected, selected_service_ms


def remaining_prefill_service_ms(
    *,
    remaining_tokens: int,
    prefix_tokens: int,
    budget_ms: float,
    prefill_model: dict[str, Any],
) -> float:
    total_ms = 0.0
    while remaining_tokens:
        query, service_ms = choose_chunk(
            remaining_tokens=remaining_tokens,
            prefix_tokens=prefix_tokens,
            budget_ms=budget_ms,
            prefill_model=prefill_model,
        )
        total_ms += service_ms
        remaining_tokens -= query
        prefix_tokens += query
    return total_ms


def make_admission_selector(
    *,
    backlog_limit_ms: float,
    slice_budget_ms: float,
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[dict[str, float]],
]:
    decisions: list[dict[str, float]] = []

    def select(state: dict[str, Any]) -> bool:
        backlog_ms = 0.0
        for request in state["prefill_queue"]:
            processed = int(
                state["prefill_processed"][
                    request
                ]
            )
            remaining = (
                int(
                    state["uncached_tokens"][
                        request
                    ]
                )
                - processed
            )
            prefix = (
                int(
                    state["cached_tokens"][
                        request
                    ]
                )
                + processed
            )
            backlog_ms += (
                remaining_prefill_service_ms(
                    remaining_tokens=remaining,
                    prefix_tokens=prefix,
                    budget_ms=slice_budget_ms,
                    prefill_model=prefill_model,
                )
            )

        request = int(state["request"])
        candidate_ms = remaining_prefill_service_ms(
            remaining_tokens=int(
                state["uncached_tokens"][
                    request
                ]
            ),
            prefix_tokens=int(
                state["cached_tokens"][
                    request
                ]
            ),
            budget_ms=slice_budget_ms,
            prefill_model=prefill_model,
        )
        predicted_completion_ms = (
            backlog_ms + candidate_ms
        )
        allow = (
            predicted_completion_ms
            <= backlog_limit_ms
            and int(
                state["reservation_tokens"][
                    request
                ]
            )
            <= int(state["kv_budget_tokens"])
        )
        decisions.append(
            {
                "predicted_backlog_ms": (
                    backlog_ms
                ),
                "candidate_prefill_ms": (
                    candidate_ms
                ),
                "predicted_completion_ms": (
                    predicted_completion_ms
                ),
                "backlog_limit_ms": (
                    backlog_limit_ms
                ),
                "allowed": float(allow),
                "prompt_tokens": float(
                    state["prompt_tokens"][
                        request
                    ]
                ),
                "output_tokens": float(
                    state["output_tokens"][
                        request
                    ]
                ),
                "queue_depth": float(
                    len(state["prefill_queue"])
                ),
            }
        )
        return allow

    return select, decisions


def admission_metrics(
    decisions: list[dict[str, float]],
) -> dict[str, float]:
    if not decisions:
        raise RuntimeError(
            "Admission policy made no decisions"
        )
    accepted = [
        decision
        for decision in decisions
        if decision["allowed"] == 1.0
    ]
    rejected = [
        decision
        for decision in decisions
        if decision["allowed"] == 0.0
    ]
    return {
        "admission_decision_acceptance": (
            len(accepted) / len(decisions)
        ),
        "p95_predicted_completion_ms": (
            float(
                np.percentile(
                    [
                        decision[
                            "predicted_completion_ms"
                        ]
                        for decision in decisions
                    ],
                    95,
                )
            )
        ),
        "p99_predicted_completion_ms": (
            float(
                np.percentile(
                    [
                        decision[
                            "predicted_completion_ms"
                        ]
                        for decision in decisions
                    ],
                    99,
                )
            )
        ),
        "mean_accepted_predicted_completion_ms": (
            statistics.mean(
                decision[
                    "predicted_completion_ms"
                ]
                for decision in accepted
            )
            if accepted
            else 0.0
        ),
        "p99_accepted_predicted_completion_ms": (
            float(
                np.percentile(
                    [
                        decision[
                            "predicted_completion_ms"
                        ]
                        for decision in accepted
                    ],
                    99,
                )
            )
            if accepted
            else 0.0
        ),
        "mean_rejected_predicted_completion_ms": (
            statistics.mean(
                decision[
                    "predicted_completion_ms"
                ]
                for decision in rejected
            )
            if rejected
            else 0.0
        ),
        "p01_rejected_predicted_completion_ms": (
            float(
                np.percentile(
                    [
                        decision[
                            "predicted_completion_ms"
                        ]
                        for decision in rejected
                    ],
                    1,
                )
            )
            if rejected
            else 0.0
        ),
        "mean_accepted_prompt_tokens_control": (
            statistics.mean(
                decision["prompt_tokens"]
                for decision in accepted
            )
            if accepted
            else 0.0
        ),
        "mean_rejected_prompt_tokens_control": (
            statistics.mean(
                decision["prompt_tokens"]
                for decision in rejected
            )
            if rejected
            else 0.0
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
    policy = POLICIES[policy_name]
    if policy["kind"] == "fixed":
        metrics = simulate(
            arrivals=arrivals,
            workload=workload,
            decode_model=decode_model,
            decode_service_models=(
                decode_service_models
            ),
            prefill_model=prefill_model,
            policy_name=policy["base_policy"],
            kv_budget_tokens=kv_budget_tokens,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    else:
        selector_config = (
            policy["selector"]
            if policy["kind"] == "adaptive"
            else hard_selector_config(
                policy["slice_budget_ms"]
            )
        )
        selector, slice_decisions = (
            make_selector(
                policy=selector_config,
                prefill_model=prefill_model,
            )
        )
        admission_selector = None
        admission_decisions = []
        if policy.get("backlog_limit_ms"):
            (
                admission_selector,
                admission_decisions,
            ) = make_admission_selector(
                backlog_limit_ms=policy[
                    "backlog_limit_ms"
                ],
                slice_budget_ms=policy[
                    "slice_budget_ms"
                ],
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
            policy_name=(
                "chunk_1024_prefill_first"
            ),
            kv_budget_tokens=kv_budget_tokens,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            chunk_selector=selector,
            admission_selector=(
                admission_selector
            ),
        )
        metrics.update(
            decision_metrics(slice_decisions)
        )
        if admission_decisions:
            metrics.update(
                admission_metrics(
                    admission_decisions
                )
            )

    metrics.setdefault(
        "accepted_fraction",
        1.0,
    )
    metrics.setdefault(
        "rejected_fraction",
        0.0,
    )
    metrics.setdefault(
        "accepted_output_token_fraction",
        1.0,
    )
    metrics.setdefault(
        "mean_accepted_prompt_tokens",
        float(np.mean(workload["prompt_tokens"])),
    )
    metrics.setdefault(
        "mean_rejected_prompt_tokens",
        0.0,
    )
    metrics.setdefault(
        "mean_accepted_output_tokens",
        float(np.mean(workload["output_tokens"])),
    )
    metrics.setdefault(
        "mean_rejected_output_tokens",
        0.0,
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
    capacity_policy = (
        "hard_100ms"
        if POLICIES[policy_name].get(
            "backlog_limit_ms"
        )
        else policy_name
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
        policy_name=capacity_policy,
        kv_budget_tokens=kv_budget_tokens,
        analysis_start=0,
        analysis_end=num_requests,
    )
    busy_ms = (
        metrics["prefill_busy_ms"]
        + metrics["decode_busy_ms"]
    )
    return {
        "capacity_policy": capacity_policy,
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
        "mean_prefill_chunks": metrics[
            "mean_prefill_chunks"
        ],
    }


def v41_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "adaptive_prefill_scheduling"
        / "adaptive_prefill_scheduling_full.json"
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
                make_workload,
            )
            from simulate_decode_continuous_batching import (
                simulate as simulate_decode_only,
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
            for policy_name in POLICIES:
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
            for policy_name in POLICIES:
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
                        "configuration": POLICIES[
                            policy_name
                        ],
                        "metrics": aggregate_metrics(
                            scenarios
                        ),
                        "scenarios": scenarios,
                    }
                )

    reference = v41_records(args.root)
    equivalence = []
    mapping = {
        "fixed_1024": "fixed_1024",
        "adaptive_100ms_age": (
            "adaptive_100ms_age"
        ),
    }
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
            if policy_name not in mapping:
                continue
            expected_record = reference[
                (
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    mapping[policy_name],
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
                        "v41 control equivalence failed: "
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
            "prefill_stall_capped_admission_simulation"
        ),
        "model": (
            "The v41 measured-service prefill/decode "
            "model with a non-escalating 100 ms "
            "prefill slice and optional 2/4-second "
            "predicted prefill-service backlog admission."
        ),
        "prefill_service_model": prefill_model,
        "decode_service_models": (
            decode_service_models
        ),
        "workload": {
            "base": "mixed_chat",
            "prefix_modes": PREFIX_MODES,
        },
        "policies": POLICIES,
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
            "v41_equivalence_enabled": (
                equivalence_enabled
            ),
            "v41_equivalence_checks": (
                equivalence
            ),
            "v41_maximum_absolute_difference": (
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
                "The v40 base asserts exact accepted "
                "prefill/decode tokens, exact causal "
                "pairs, finite completion only for "
                "accepted requests, max batch 64, and "
                "KV within budget. Hard slices report "
                "zero budget violations; admission "
                "classifies every request exactly once."
            ),
        },
        "records": records,
        "limitations": [
            "The underlying prefill/decode service model inherits all v40/v41 limitations and is not a live serving-engine measurement.",
            "Admission predicts only queued prefill GPU service under the hard 100 ms chunk policy. It excludes decode work between chunks, future arrivals, CPU overhead, and KV contention.",
            "The 2/4-second backlog thresholds are controls, not production SLOs. Actual TTFT can exceed them because decode and scheduling remain outside the predictor.",
            "Rejected requests leave immediately and are not retried, queued elsewhere, or assigned tenant priorities.",
            "The admission policy is FCFS and does not explicitly equalize prompt length, output length, or tenant token budgets.",
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
