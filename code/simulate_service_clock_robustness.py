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
from simulate_decayed_fair_admission import (
    decay_metrics,
)
from simulate_decayed_fair_admission import (
    simulate_decayed_policy as simulate_e55_policy,
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
from simulate_nonstationary_fair_admission import (
    PHASES,
    make_nonstationary_workload,
    phase_metrics,
)
from simulate_nonstationary_fair_admission import (
    simulate_policy as simulate_e54_policy,
)
from simulate_prefill_admission_safety import (
    NOMINAL_BACKLOG_LIMIT_MS,
    estimate_capacity,
    safety_admission_metrics,
)
from simulate_prefill_decode_scheduling import (
    PREFIX_MODES,
    load_prefill_service_model,
    simulate,
)
from simulate_prefill_stall_admission import (
    hard_selector_config,
    make_admission_selector,
    remaining_prefill_service_ms,
)
from simulate_weighted_fair_admission import (
    BASE_LIMIT_MS,
    BASE_SAFETY_FACTOR,
    MAX_CREDIT,
    MIN_CREDIT,
    PRIOR_ACCEPTANCE,
    PRIOR_COUNT,
    PROMPT_CLASSES,
    weighted_fair_metrics,
)


HALF_LIFE_MS = 40_000.0
POLICIES = {
    "fair_gain_4_window_512": {
        "kind": "e54_control",
    },
    "fair_gain_4_service_half_40s": {
        "kind": "e55_control",
    },
    "service40_uniform_075x": {
        "kind": "biased_service",
        "global_multiplier": 0.75,
    },
    "service40_uniform_125x": {
        "kind": "biased_service",
        "global_multiplier": 1.25,
    },
    "service40_long_075x": {
        "kind": "biased_service",
        "class_multipliers": {
            512: 1.0,
            2048: 1.0,
            6144: 0.75,
        },
    },
    "service40_long_125x": {
        "kind": "biased_service",
        "class_multipliers": {
            512: 1.0,
            2048: 1.0,
            6144: 1.25,
        },
    },
    "service40_cache_blind": {
        "kind": "biased_service",
        "cache_blind": True,
    },
}

CONTROL_POLICIES = (
    "fair_gain_4_window_512",
    "fair_gain_4_service_half_40s",
)


def make_biased_service_selector(
    *,
    configuration: dict[str, Any],
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[dict[str, float]],
]:
    offered = {
        prompt: 0.0
        for prompt in PROMPT_CLASSES
    }
    accepted = {
        prompt: 0.0
        for prompt in PROMPT_CLASSES
    }
    decisions: list[dict[str, float]] = []

    def select(state: dict[str, Any]) -> bool:
        request = int(state["request"])
        prompt = int(
            state["prompt_tokens"][request]
        )
        predictor, predictor_decisions = (
            make_admission_selector(
                backlog_limit_ms=1e30,
                slice_budget_ms=100.0,
                prefill_model=prefill_model,
            )
        )
        predictor_allow = predictor(state)
        decision = predictor_decisions[0]
        true_clock_delta_ms = decision[
            "candidate_prefill_ms"
        ]
        if configuration.get(
            "cache_blind",
            False,
        ):
            clock_delta_ms = (
                remaining_prefill_service_ms(
                    remaining_tokens=prompt,
                    prefix_tokens=0,
                    budget_ms=100.0,
                    prefill_model=prefill_model,
                )
            )
        else:
            multiplier = configuration.get(
                "global_multiplier",
                configuration.get(
                    "class_multipliers",
                    {},
                ).get(prompt, 1.0),
            )
            clock_delta_ms = (
                true_clock_delta_ms
                * multiplier
            )
        if (
            true_clock_delta_ms <= 0.0
            or clock_delta_ms <= 0.0
        ):
            raise RuntimeError(
                "Invalid service-clock delta"
            )
        clock_ratio = (
            clock_delta_ms
            / true_clock_delta_ms
        )
        decay_factor = 2.0 ** (
            -clock_delta_ms / HALF_LIFE_MS
        )
        if not (
            0.0 < decay_factor <= 1.0
        ):
            raise RuntimeError(
                "Invalid decay factor"
            )
        for member in PROMPT_CLASSES:
            offered[member] *= decay_factor
            accepted[member] *= decay_factor
            if (
                accepted[member]
                > offered[member] + 1e-12
                or offered[member] < 0.0
            ):
                raise RuntimeError(
                    "Decayed accounting failed"
                )
        history_weight_before = sum(
            offered.values()
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
                math.exp(
                    4.0 * class_deficit
                ),
            ),
        )
        effective_limit_ms = (
            BASE_LIMIT_MS * credit
        )
        allow = (
            predictor_allow
            and decision[
                "predicted_completion_ms"
            ]
            <= effective_limit_ms
        )
        offered[prompt] += 1.0
        accepted[prompt] += float(allow)
        decisions.append(
            {
                **decision,
                "request": float(request),
                "allowed": float(allow),
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
                "controller_gain": 4.0,
                "decay_clock": 1.0,
                "decay_half_life_ms": (
                    HALF_LIFE_MS
                ),
                "decay_clock_delta_ms": (
                    clock_delta_ms
                ),
                "true_decay_clock_delta_ms": (
                    true_clock_delta_ms
                ),
                "clock_error_ratio": (
                    clock_ratio
                ),
                "clock_relative_error": (
                    clock_ratio - 1.0
                ),
                "decay_factor": (
                    decay_factor
                ),
                "effective_history_weight_before": (
                    history_weight_before
                ),
                "smoothed_class_acceptance_before": (
                    ratios[prompt]
                ),
                "smoothed_equal_class_mean_before": (
                    equal_class_mean
                ),
                "class_deficit_before": (
                    class_deficit
                ),
                "class_offered_before": (
                    offered[prompt] - 1.0
                ),
                "class_accepted_before": (
                    accepted[prompt]
                    - float(allow)
                ),
            }
        )
        return allow

    return select, decisions


def clock_error_metrics(
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
            "Clock policy made no decisions"
        )
    ratios = np.asarray(
        [
            decision["clock_error_ratio"]
            for decision in selected
        ],
        dtype=np.float64,
    )
    relative = ratios - 1.0
    metrics = {
        "mean_clock_error_ratio": float(
            np.mean(ratios)
        ),
        "p05_clock_error_ratio": float(
            np.percentile(ratios, 5)
        ),
        "p95_clock_error_ratio": float(
            np.percentile(ratios, 95)
        ),
        "mean_absolute_clock_relative_error": (
            float(np.mean(np.abs(relative)))
        ),
        "maximum_absolute_clock_relative_error": (
            float(np.max(np.abs(relative)))
        ),
    }
    for prompt in PROMPT_CLASSES:
        members = [
            decision
            for decision in selected
            if int(
                decision["prompt_tokens"]
            )
            == prompt
        ]
        if not members:
            raise RuntimeError(
                f"Missing prompt class {prompt}"
            )
        metrics[
            f"prompt_{prompt}_mean_clock_error_ratio"
        ] = statistics.mean(
            decision["clock_error_ratio"]
            for decision in members
        )
        metrics[
            f"prompt_{prompt}_mean_true_clock_ms"
        ] = statistics.mean(
            decision[
                "true_decay_clock_delta_ms"
            ]
            for decision in members
        )
        metrics[
            f"prompt_{prompt}_mean_biased_clock_ms"
        ] = statistics.mean(
            decision[
                "decay_clock_delta_ms"
            ]
            for decision in members
        )
    return metrics


def simulate_biased_policy(
    *,
    arrivals: np.ndarray,
    workload: dict[str, np.ndarray],
    phase_ids: np.ndarray,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    policy_name: str,
    kv_budget_tokens: int,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    (
        admission_selector,
        admission_decisions,
    ) = make_biased_service_selector(
        configuration=POLICIES[policy_name],
        prefill_model=prefill_model,
    )
    chunk_selector, slice_decisions = (
        make_selector(
            policy=hard_selector_config(
                100.0
            ),
            prefill_model=prefill_model,
        )
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
    metrics.update(
        weighted_fair_metrics(
            admission_decisions,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    metrics.update(
        decay_metrics(
            admission_decisions,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    metrics.update(
        clock_error_metrics(
            admission_decisions,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    metrics.update(
        phase_metrics(
            admission_decisions,
            phase_ids=phase_ids,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    return metrics


def e55_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "decayed_fair_admission"
        / "decayed_fair_admission_full.json"
    )
    result = json.loads(
        path.read_text(encoding="utf-8")
    )
    return {
        (
            record["workload_mode"],
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
    phase_ranges_reference = None
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
                                if prefix_mode == "none"
                                else 10_000
                            )
                        )
                        (
                            workload,
                            phase_ids,
                            phase_ranges,
                        ) = make_nonstationary_workload(
                            num_requests=(
                                args.num_requests
                            ),
                            seed=workload_seed,
                            prefix_mode=prefix_mode,
                        )
                        if (
                            phase_ranges_reference
                            is None
                        ):
                            phase_ranges_reference = (
                                phase_ranges
                            )
                        elif [
                            (
                                item[
                                    "start_request"
                                ],
                                item[
                                    "end_request"
                                ],
                                item[
                                    "prompt_class_counts"
                                ],
                            )
                            for item
                            in phase_ranges_reference
                        ] != [
                            (
                                item[
                                    "start_request"
                                ],
                                item[
                                    "end_request"
                                ],
                                item[
                                    "prompt_class_counts"
                                ],
                            )
                            for item
                            in phase_ranges
                        ]:
                            raise RuntimeError(
                                "Phase composition drifted"
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
                        if (
                            policy_name
                            == "fair_gain_4_window_512"
                        ):
                            metrics = (
                                simulate_e54_policy(
                                    arrivals=arrivals,
                                    workload=workload,
                                    phase_ids=phase_ids,
                                    decode_model=(
                                        decode_model
                                    ),
                                    decode_service_models=(
                                        decode_service_models
                                    ),
                                    prefill_model=(
                                        prefill_model
                                    ),
                                    policy_name=(
                                        policy_name
                                    ),
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
                            )
                        elif (
                            policy_name
                            == "fair_gain_4_service_half_40s"
                        ):
                            metrics = (
                                simulate_e55_policy(
                                    arrivals=arrivals,
                                    workload=workload,
                                    phase_ids=phase_ids,
                                    decode_model=(
                                        decode_model
                                    ),
                                    decode_service_models=(
                                        decode_service_models
                                    ),
                                    prefill_model=(
                                        prefill_model
                                    ),
                                    policy_name=(
                                        policy_name
                                    ),
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
                            )
                        else:
                            metrics = (
                                simulate_biased_policy(
                                    arrivals=arrivals,
                                    workload=workload,
                                    phase_ids=phase_ids,
                                    decode_model=(
                                        decode_model
                                    ),
                                    decode_service_models=(
                                        decode_service_models
                                    ),
                                    prefill_model=(
                                        prefill_model
                                    ),
                                    policy_name=(
                                        policy_name
                                    ),
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
                        "workload_mode": (
                            "rotating_prompt_mix"
                        ),
                        "prefix_mode": (
                            prefix_mode
                        ),
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

    record_index = {
        (
            record["prefix_mode"],
            record[
                "offered_load_vs_decode_capacity"
            ],
            record["policy"],
        ): record
        for record in records
    }
    for prefix_mode in PREFIX_MODES:
        for offered_load in offered_loads:
            exact = record_index[
                (
                    prefix_mode,
                    offered_load,
                    (
                        "fair_gain_4_"
                        "service_half_40s"
                    ),
                )
            ]["scenarios"]
            under = record_index[
                (
                    prefix_mode,
                    offered_load,
                    (
                        "service40_"
                        "uniform_075x"
                    ),
                )
            ]["scenarios"]
            over = record_index[
                (
                    prefix_mode,
                    offered_load,
                    (
                        "service40_"
                        "uniform_125x"
                    ),
                )
            ]["scenarios"]
            for exact_item, under_item, over_item in zip(
                exact,
                under,
                over,
            ):
                if not (
                    under_item[
                        "mean_effective_history_weight"
                    ]
                    > exact_item[
                        "mean_effective_history_weight"
                    ]
                    > over_item[
                        "mean_effective_history_weight"
                    ]
                ):
                    raise RuntimeError(
                        "Global clock-bias memory "
                        "ordering failed"
                    )

    reference = e55_records(args.root)
    equivalence = []
    cache_blind_equivalence = []
    equivalence_enabled = (
        args.num_requests == 2_000
        and args.warmup_requests == 200
        and args.tail_guard_requests == 100
        and args.capacity_requests == 1_500
        and args.seeds == 3
        and args.seed == 40_000
        and offered_loads == [0.3, 0.4]
    )
    if equivalence_enabled:
        for record in records:
            if (
                record["policy"]
                not in CONTROL_POLICIES
            ):
                continue
            expected_record = reference[
                (
                    "rotating_prompt_mix",
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    record["policy"],
                )
            ]
            expected_scenarios = {
                (
                    scenario[
                        "service_replicate"
                    ],
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
                    for metric, value
                    in expected.items()
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
                        "E55 control equivalence "
                        "failed: "
                        f"{record['policy']=} "
                        f"{key=} "
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
                        "policy": record[
                            "policy"
                        ],
                        "service_replicate": key[0],
                        "seed": key[1],
                        "maximum_absolute_difference": (
                            maximum_difference
                        ),
                    }
                )

        for offered_load in offered_loads:
            exact_record = record_index[
                (
                    "none",
                    offered_load,
                    (
                        "fair_gain_4_"
                        "service_half_40s"
                    ),
                )
            ]
            blind_record = record_index[
                (
                    "none",
                    offered_load,
                    "service40_cache_blind",
                )
            ]
            exact_scenarios = {
                (
                    scenario[
                        "service_replicate"
                    ],
                    scenario["seed"],
                ): scenario
                for scenario in exact_record[
                    "scenarios"
                ]
            }
            for scenario in blind_record[
                "scenarios"
            ]:
                key = (
                    scenario["service_replicate"],
                    scenario["seed"],
                )
                expected = exact_scenarios[key]
                common_metrics = [
                    metric
                    for metric, value
                    in expected.items()
                    if isinstance(
                        value,
                        (int, float),
                    )
                    and not isinstance(value, bool)
                    and metric in scenario
                ]
                maximum_difference = max(
                    abs(
                        scenario[metric]
                        - expected[metric]
                    )
                    for metric in common_metrics
                )
                if maximum_difference > 1e-9:
                    raise RuntimeError(
                        "No-cache cache-blind "
                        "equivalence failed: "
                        f"{key=} "
                        f"{maximum_difference=}"
                    )
                cache_blind_equivalence.append(
                    {
                        "offered_load": (
                            offered_load
                        ),
                        "service_replicate": key[0],
                        "seed": key[1],
                        "maximum_absolute_difference": (
                            maximum_difference
                        ),
                    }
                )

    result = {
        "suite": (
            "service_clock_robustness_"
            "simulation"
        ),
        "model": (
            "The E55 rotating-prompt service-"
            "decay controller with isolated "
            "uniform, class-specific, and "
            "cache-blind fairness-clock errors."
        ),
        "prefill_service_model": prefill_model,
        "decode_service_models": (
            decode_service_models
        ),
        "workload": {
            "nonstationary_phases": PHASES,
            "phase_ranges": (
                phase_ranges_reference
            ),
            "prefix_modes": PREFIX_MODES,
        },
        "controller": {
            "gain": 4.0,
            "service_half_life_ms": (
                HALF_LIFE_MS
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
            "clock_errors_only": (
                "Admission backlog prediction is "
                "unchanged; only fairness-state "
                "decay uses the biased service "
                "clock."
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
            "e55_equivalence_enabled": (
                equivalence_enabled
            ),
            "e55_equivalence_checks": (
                equivalence
            ),
            "e55_maximum_absolute_difference": (
                max(
                    item[
                        "maximum_absolute_difference"
                    ]
                    for item in equivalence
                )
                if equivalence
                else None
            ),
            "cache_blind_no_cache_checks": (
                cache_blind_equivalence
            ),
            "cache_blind_no_cache_maximum_absolute_difference": (
                max(
                    item[
                        "maximum_absolute_difference"
                    ]
                    for item
                    in cache_blind_equivalence
                )
                if cache_blind_equivalence
                else None
            ),
            "accounting": (
                "The v40 base asserts exact "
                "accepted prefill/decode tokens, "
                "causal pairs, finite completion "
                "only for accepted requests, max "
                "batch 64, and KV within budget. "
                "Phase counts are exact, decayed "
                "accepted weight never exceeds "
                "offered weight, global bias "
                "orders effective history "
                "monotonically, every admitted "
                "request satisfies its unchanged "
                "admission threshold, and hard "
                "slices report zero violations."
            ),
        },
        "records": records,
        "limitations": [
            "The service model inherits all E49-E55 limitations and is not a live serving-engine measurement.",
            "Clock errors are synthetic sensitivity controls and do not represent a calibrated production predictor-error distribution.",
            "Class-specific bias is applied only to the 6144-token prompt class; real errors also vary with batch, prefix position, hardware state, and decode occupancy.",
            "Cache-blind service uses no-cache candidate prefill cost and isolates missed prefix reuse in the fairness clock, not in admission or execution.",
            "The controller still optimizes equal prompt-class acceptance rather than tenant, token, or utility-weighted fairness.",
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
        default="0.3,0.4",
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
