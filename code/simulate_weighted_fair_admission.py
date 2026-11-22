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
    make_workload,
)
from simulate_decode_continuous_batching import (
    simulate as simulate_decode_only,
)
from simulate_prefill_admission_safety import (
    NOMINAL_BACKLOG_LIMIT_MS,
    estimate_capacity,
    make_safety_admission_selector,
    safety_admission_metrics,
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


PROMPT_CLASSES = (512, 2048, 6144)
BASE_SAFETY_FACTOR = 8.0
BASE_LIMIT_MS = (
    NOMINAL_BACKLOG_LIMIT_MS
    / BASE_SAFETY_FACTOR
)
PRIOR_COUNT = 32.0
PRIOR_ACCEPTANCE = 0.85
MIN_CREDIT = 0.5
MAX_CREDIT = 4.0

POLICIES = {
    "safety_8x": {
        "kind": "static",
        "safety_factor": 8.0,
        "prompt_credit": False,
    },
    "safety_8x_sqrt_prompt_credit": {
        "kind": "static",
        "safety_factor": 8.0,
        "prompt_credit": True,
    },
    "fair_gain_2": {
        "kind": "weighted_fair",
        "gain": 2.0,
    },
    "fair_gain_4": {
        "kind": "weighted_fair",
        "gain": 4.0,
    },
    "fair_gain_8": {
        "kind": "weighted_fair",
        "gain": 8.0,
    },
}


def make_weighted_fair_selector(
    *,
    gain: float,
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[dict[str, float]],
]:
    offered = {
        prompt: 0
        for prompt in PROMPT_CLASSES
    }
    accepted = {
        prompt: 0
        for prompt in PROMPT_CLASSES
    }
    decisions: list[dict[str, float]] = []

    def select(state: dict[str, Any]) -> bool:
        request = int(state["request"])
        prompt = int(
            state["prompt_tokens"][request]
        )
        if prompt not in offered:
            raise RuntimeError(
                f"Unexpected prompt class {prompt}"
            )
        ratios = {
            member: (
                accepted[member]
                + PRIOR_ACCEPTANCE
                * PRIOR_COUNT
            )
            / (
                offered[member]
                + PRIOR_COUNT
            )
            for member in PROMPT_CLASSES
        }
        equal_class_mean = statistics.mean(
            ratios.values()
        )
        class_deficit = (
            equal_class_mean - ratios[prompt]
        )
        credit = min(
            MAX_CREDIT,
            max(
                MIN_CREDIT,
                math.exp(gain * class_deficit),
            ),
        )
        effective_limit_ms = (
            BASE_LIMIT_MS * credit
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
        offered[prompt] += 1
        accepted[prompt] += int(allow)
        decisions.append(
            {
                **decision,
                "request": float(request),
                "nominal_backlog_limit_ms": (
                    NOMINAL_BACKLOG_LIMIT_MS
                ),
                "safety_factor": (
                    BASE_SAFETY_FACTOR
                ),
                "prompt_credit_multiplier": (
                    credit
                ),
                "effective_limit_ms": (
                    effective_limit_ms
                ),
                "safety_adjusted_completion_ms": (
                    decision[
                        "predicted_completion_ms"
                    ]
                    * BASE_SAFETY_FACTOR
                    / credit
                ),
                "controller_gain": gain,
                "smoothed_class_acceptance_before": (
                    ratios[prompt]
                ),
                "smoothed_equal_class_mean_before": (
                    equal_class_mean
                ),
                "class_deficit_before": (
                    class_deficit
                ),
                "class_offered_before": float(
                    offered[prompt] - 1
                ),
                "class_accepted_before": float(
                    accepted[prompt] - int(allow)
                ),
            }
        )
        return allow

    return select, decisions


def weighted_fair_metrics(
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
            "Weighted-fair policy made no decisions"
        )
    credits = np.asarray(
        [
            decision[
                "prompt_credit_multiplier"
            ]
            for decision in selected
        ],
        dtype=np.float64,
    )
    deficits = np.asarray(
        [
            abs(
                decision[
                    "class_deficit_before"
                ]
            )
            for decision in selected
        ],
        dtype=np.float64,
    )
    metrics = {
        "mean_prompt_credit_multiplier": float(
            np.mean(credits)
        ),
        "p05_prompt_credit_multiplier": float(
            np.percentile(credits, 5)
        ),
        "p95_prompt_credit_multiplier": float(
            np.percentile(credits, 95)
        ),
        "minimum_credit_saturation_fraction": (
            float(np.mean(credits <= MIN_CREDIT))
        ),
        "maximum_credit_saturation_fraction": (
            float(np.mean(credits >= MAX_CREDIT))
        ),
        "mean_absolute_class_deficit_before": (
            float(np.mean(deficits))
        ),
        "p95_absolute_class_deficit_before": (
            float(np.percentile(deficits, 95))
        ),
    }
    for prompt in PROMPT_CLASSES:
        class_credits = np.asarray(
            [
                decision[
                    "prompt_credit_multiplier"
                ]
                for decision in selected
                if int(
                    decision["prompt_tokens"]
                )
                == prompt
            ],
            dtype=np.float64,
        )
        if class_credits.size < 1:
            raise RuntimeError(
                f"Missing prompt class {prompt}"
            )
        metrics[
            f"prompt_{prompt}_mean_credit"
        ] = float(np.mean(class_credits))
        metrics[
            f"prompt_{prompt}_p95_credit"
        ] = float(
            np.percentile(
                class_credits,
                95,
            )
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
    chunk_selector, slice_decisions = (
        make_selector(
            policy=hard_selector_config(
                100.0
            ),
            prefill_model=prefill_model,
        )
    )
    configuration = POLICIES[policy_name]
    if configuration["kind"] == "static":
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
    else:
        (
            admission_selector,
            admission_decisions,
        ) = make_weighted_fair_selector(
            gain=configuration["gain"],
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
        chunk_selector=chunk_selector,
        admission_selector=admission_selector,
    )
    metrics.update(
        decision_metrics(slice_decisions)
    )
    metrics.update(
        safety_admission_metrics(
            admission_decisions,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    if configuration["kind"] == (
        "weighted_fair"
    ):
        metrics.update(
            weighted_fair_metrics(
                admission_decisions,
                analysis_start=analysis_start,
                analysis_end=analysis_end,
            )
        )
    return metrics


def e52_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "prefill_admission_safety"
        / "prefill_admission_safety_full.json"
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
            decode_workload = make_workload(
                name="mixed_chat",
                num_requests=(
                    args.capacity_requests
                ),
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
                kv_budget_tokens=(
                    kv_budget_tokens
                ),
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
                for (
                    replicate_index,
                    decode_model,
                ) in enumerate(
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
                                if prefix_mode
                                == "none"
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
                        if (
                            metrics[
                                "slice_budget_"
                                "violation_fraction"
                            ]
                            != 0.0
                        ):
                            raise RuntimeError(
                                "Hard slice budget failed"
                            )
                        if (
                            metrics[
                                "p99_accepted_"
                                "safety_adjusted_"
                                "completion_ms"
                            ]
                            > (
                                NOMINAL_BACKLOG_LIMIT_MS
                                + 1e-9
                            )
                        ):
                            raise RuntimeError(
                                "Admission threshold failed"
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

    reference = e52_records(args.root)
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
        baseline_policies = (
            "safety_8x",
            "safety_8x_sqrt_prompt_credit",
        )
        for record in records:
            policy_name = record["policy"]
            if policy_name not in baseline_policies:
                continue
            expected_record = reference[
                (
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    policy_name,
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
                numeric_metrics = [
                    metric
                    for metric, value in (
                        expected.items()
                    )
                    if isinstance(
                        value,
                        (int, float),
                    )
                    and not isinstance(value, bool)
                ]
                maximum_difference = max(
                    abs(
                        scenario[metric]
                        - expected[metric]
                    )
                    for metric in numeric_metrics
                )
                if maximum_difference > 1e-9:
                    raise RuntimeError(
                        "E52 control equivalence "
                        "failed: "
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
            "weighted_fair_prefill_"
            "admission_simulation"
        ),
        "model": (
            "The E52 hard-100ms, 8x-safety "
            "controller with explicit online "
            "equal-class acceptance deficits."
        ),
        "prefill_service_model": prefill_model,
        "decode_service_models": (
            decode_service_models
        ),
        "workload": {
            "base": "mixed_chat",
            "prefix_modes": PREFIX_MODES,
            "prompt_classes": PROMPT_CLASSES,
        },
        "controller": {
            "objective": (
                "Equalize smoothed acceptance "
                "rates across prompt classes."
            ),
            "base_safety_factor": (
                BASE_SAFETY_FACTOR
            ),
            "base_limit_ms": BASE_LIMIT_MS,
            "prior_count": PRIOR_COUNT,
            "prior_acceptance": (
                PRIOR_ACCEPTANCE
            ),
            "minimum_credit": MIN_CREDIT,
            "maximum_credit": MAX_CREDIT,
            "credit_rule": (
                "clip(exp(gain * "
                "(equal_class_mean - "
                "class_acceptance)), 0.5, 4.0)"
            ),
        },
        "policies": POLICIES,
        "simulation": {
            "num_requests": args.num_requests,
            "warmup_requests": (
                args.warmup_requests
            ),
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
            "kv_budget_gib": (
                args.kv_budget_gib
            ),
            "kv_budget_tokens": (
                kv_budget_tokens
            ),
            "kv_bytes_per_token": (
                KV_BYTES_PER_TOKEN
            ),
        },
        "decode_only_capacities": (
            decode_capacities
        ),
        "hard_100ms_capacities": capacities,
        "validation": {
            "scenario_count": scenario_count,
            "e52_equivalence_enabled": (
                equivalence_enabled
            ),
            "e52_equivalence_checks": (
                equivalence
            ),
            "e52_maximum_absolute_difference": (
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
                "The v40 base asserts exact "
                "accepted prefill/decode tokens, "
                "causal pairs, finite completion "
                "only for accepted requests, max "
                "batch 64, and KV within budget. "
                "Every admitted request satisfies "
                "its dynamic safety-adjusted "
                "threshold and hard slices report "
                "zero budget violations."
            ),
        },
        "records": records,
        "limitations": [
            "The service model inherits all E49-E52 limitations and is not a live serving-engine measurement.",
            "The controller optimizes equal prompt-class acceptance, not tenant, token, or utility-weighted fairness.",
            "Cumulative smoothed feedback assumes a stationary class mixture and does not model abrupt tenant or workload drift.",
            "Gain and credit bounds are a sensitivity sweep, not an automatically tuned production controller.",
            "Rejected requests leave immediately and are not retried, routed to another replica, or queued in a reservation class.",
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
