from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simulate_adaptive_prefill_scheduling import (
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
    simulate,
)
from simulate_prefill_stall_admission import (
    hard_selector_config,
    make_admission_selector,
)


NOMINAL_BACKLOG_LIMIT_MS = 2_000.0
POLICIES = {
    "hard_100ms": {
        "safety_factor": None,
        "prompt_credit": False,
    },
    "raw_2s": {
        "safety_factor": 1.0,
        "prompt_credit": False,
    },
    "safety_2x": {
        "safety_factor": 2.0,
        "prompt_credit": False,
    },
    "safety_4x": {
        "safety_factor": 4.0,
        "prompt_credit": False,
    },
    "safety_8x": {
        "safety_factor": 8.0,
        "prompt_credit": False,
    },
    "safety_8x_sqrt_prompt_credit": {
        "safety_factor": 8.0,
        "prompt_credit": True,
    },
}


def make_safety_admission_selector(
    *,
    safety_factor: float,
    prompt_credit: bool,
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[dict[str, float]],
]:
    base_limit_ms = (
        NOMINAL_BACKLOG_LIMIT_MS
        / safety_factor
    )
    decisions: list[dict[str, float]] = []

    def select(state: dict[str, Any]) -> bool:
        request = int(state["request"])
        prompt_tokens = int(
            state["prompt_tokens"][request]
        )
        prompt_multiplier = (
            math.sqrt(prompt_tokens / 512.0)
            if prompt_credit
            else 1.0
        )
        effective_limit_ms = (
            base_limit_ms * prompt_multiplier
        )
        base_selector, base_decisions = (
            make_admission_selector(
                backlog_limit_ms=(
                    effective_limit_ms
                ),
                slice_budget_ms=100.0,
                prefill_model=prefill_model,
            )
        )
        allow = base_selector(state)
        decision = base_decisions[0]
        decisions.append(
            {
                **decision,
                "request": float(request),
                "nominal_backlog_limit_ms": (
                    NOMINAL_BACKLOG_LIMIT_MS
                ),
                "safety_factor": safety_factor,
                "prompt_credit_multiplier": (
                    prompt_multiplier
                ),
                "effective_limit_ms": (
                    effective_limit_ms
                ),
                "safety_adjusted_completion_ms": (
                    decision[
                        "predicted_completion_ms"
                    ]
                    * safety_factor
                    / prompt_multiplier
                ),
            }
        )
        return allow

    return select, decisions


def safety_admission_metrics(
    decisions: list[dict[str, float]],
    *,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    selected = [
        decision
        for decision in decisions
        if analysis_start
        <= int(decision["request"])
        < analysis_end
    ]
    if not selected:
        raise RuntimeError(
            "Admission policy made no decisions"
        )
    accepted = [
        decision
        for decision in selected
        if decision["allowed"] == 1.0
    ]
    rejected = [
        decision
        for decision in selected
        if decision["allowed"] == 0.0
    ]
    metrics = {
        "admission_decision_acceptance": (
            len(accepted) / len(selected)
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
        "p99_accepted_safety_adjusted_completion_ms": (
            float(
                np.percentile(
                    [
                        decision[
                            "safety_adjusted_completion_ms"
                        ]
                        for decision in accepted
                    ],
                    99,
                )
            )
            if accepted
            else 0.0
        ),
        "p01_rejected_safety_adjusted_completion_ms": (
            float(
                np.percentile(
                    [
                        decision[
                            "safety_adjusted_completion_ms"
                        ]
                        for decision in rejected
                    ],
                    1,
                )
            )
            if rejected
            else 0.0
        ),
        "mean_effective_limit_ms": float(
            statistics.mean(
                decision["effective_limit_ms"]
                for decision in selected
            )
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
    prompt_acceptance = {}
    for prompt in (512, 2048, 6144):
        members = [
            decision
            for decision in selected
            if decision["prompt_tokens"]
            == prompt
        ]
        if not members:
            raise RuntimeError(
                f"Missing prompt class {prompt}"
            )
        prompt_acceptance[prompt] = (
            sum(
                decision["allowed"]
                for decision in members
            )
            / len(members)
        )
        metrics[
            f"prompt_{prompt}_acceptance"
        ] = prompt_acceptance[prompt]
    metrics[
        "long_vs_short_prompt_acceptance"
    ] = (
        prompt_acceptance[6144]
        / max(
            prompt_acceptance[512],
            1e-12,
        )
    )
    acceptance_values = np.asarray(
        list(prompt_acceptance.values()),
        dtype=np.float64,
    )
    denominator = (
        acceptance_values.size
        * float(
            np.sum(
                acceptance_values
                * acceptance_values
            )
        )
    )
    metrics["prompt_acceptance_jain_index"] = (
        float(
            np.sum(acceptance_values) ** 2
            / denominator
        )
        if denominator > 0.0
        else 1.0
    )
    return metrics


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
    selector, slice_decisions = make_selector(
        policy=hard_selector_config(100.0),
        prefill_model=prefill_model,
    )
    configuration = POLICIES[policy_name]
    admission_selector = None
    admission_decisions = []
    if configuration["safety_factor"]:
        (
            admission_selector,
            admission_decisions,
        ) = make_safety_admission_selector(
            safety_factor=configuration[
                "safety_factor"
            ],
            prompt_credit=configuration[
                "prompt_credit"
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
        policy_name="chunk_1024_prefill_first",
        kv_budget_tokens=kv_budget_tokens,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        chunk_selector=selector,
        admission_selector=admission_selector,
    )
    metrics.update(
        decision_metrics(slice_decisions)
    )
    if admission_decisions:
        metrics.update(
            safety_admission_metrics(
                admission_decisions,
                analysis_start=analysis_start,
                analysis_end=analysis_end,
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
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    kv_budget_tokens: int,
) -> dict[str, float]:
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
        policy_name="hard_100ms",
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
        "mean_prefill_chunks": metrics[
            "mean_prefill_chunks"
        ],
    }


def v42_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "prefill_stall_admission"
        / "prefill_stall_admission_full.json"
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
            capacities[prefix_mode][source] = (
                estimate_capacity(
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
                    decode_model=decode_model,
                    decode_service_models=(
                        decode_service_models
                    ),
                    prefill_model=prefill_model,
                    kv_budget_tokens=(
                        kv_budget_tokens
                    ),
                )
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
                                        "requests_per_second"
                                    ]
                                ),
                                "offered_load_vs_system_capacity": (
                                    request_rate
                                    / capacities[
                                        prefix_mode
                                    ][source][
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

    reference = v42_records(args.root)
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
        mapping = {
            "hard_100ms": "hard_100ms",
            "raw_2s": (
                "hard_100ms_admit_2s"
            ),
        }
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
                        "accepted_fraction",
                        "rejected_fraction",
                        (
                            "accepted_output_"
                            "token_fraction"
                        ),
                        "throughput_requests_s",
                        (
                            "throughput_"
                            "output_tokens_s"
                        ),
                        "p99_prefill_latency_ms",
                        "p99_ttft_ms",
                        "p99_tpot_ms",
                        (
                            "p99_max_"
                            "intertoken_gap_ms"
                        ),
                        "p99_e2e_ms",
                        "mean_batch_size",
                        "mean_prefill_chunks",
                        "prefill_chunk_overhead",
                        "prefill_busy_ms",
                        "decode_busy_ms",
                        "peak_kv_gib",
                        "completed_output_tokens",
                        (
                            "mean_accepted_"
                            "prompt_tokens"
                        ),
                        (
                            "mean_rejected_"
                            "prompt_tokens"
                        ),
                        (
                            "mean_accepted_"
                            "output_tokens"
                        ),
                        (
                            "mean_rejected_"
                            "output_tokens"
                        ),
                    )
                )
                if maximum_difference > 1e-9:
                    raise RuntimeError(
                        "v42 control equivalence failed: "
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
            "prefill_admission_safety_simulation"
        ),
        "model": (
            "The v42 hard-100ms prefill admission "
            "model with 1/2/4/8x structural safety "
            "margins and an 8x sqrt-prompt-credit "
            "fairness control."
        ),
        "prefill_service_model": prefill_model,
        "decode_service_models": (
            decode_service_models
        ),
        "workload": {
            "base": "mixed_chat",
            "prefix_modes": PREFIX_MODES,
        },
        "nominal_backlog_limit_ms": (
            NOMINAL_BACKLOG_LIMIT_MS
        ),
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
        "hard_100ms_capacities": capacities,
        "validation": {
            "scenario_count": scenario_count,
            "v42_equivalence_enabled": (
                equivalence_enabled
            ),
            "v42_equivalence_checks": (
                equivalence
            ),
            "v42_maximum_absolute_difference": (
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
                "KV within budget. Every admitted "
                "request satisfies its safety-adjusted "
                "threshold and hard slices report zero "
                "budget violations."
            ),
        },
        "records": records,
        "limitations": [
            "The underlying prefill/decode service model inherits all v40-v42 limitations and is not a live serving-engine measurement.",
            "Safety factors multiply the same queued-prefill-service estimate. They do not add decode occupancy, future-arrival prediction, CPU overhead, or uncertainty distributions.",
            "The sqrt prompt credit is a hand-designed fairness control. It is not derived from a tenant contract, utility function, or formal weighted-fair-queue objective.",
            "Rejected requests leave immediately and are not retried or routed to another replica.",
            "Prompt-class Jain metrics weight 512/2048/6144 prompt classes equally rather than by offered traffic.",
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
