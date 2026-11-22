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
    make_prefill_workload,
    simulate,
)
from simulate_prefill_stall_admission import (
    hard_selector_config,
    make_admission_selector,
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


POLICIES = {
    "fair_gain_4_cumulative": {
        "kind": "e54_control",
    },
    "fair_gain_4_window_512": {
        "kind": "e54_control",
    },
    "fair_gain_4_wall_half_20s": {
        "kind": "decay",
        "clock": "wall_time",
        "half_life_ms": 20_000.0,
    },
    "fair_gain_4_wall_half_40s": {
        "kind": "decay",
        "clock": "wall_time",
        "half_life_ms": 40_000.0,
    },
    "fair_gain_4_service_half_20s": {
        "kind": "decay",
        "clock": "prefill_service",
        "half_life_ms": 20_000.0,
    },
    "fair_gain_4_service_half_40s": {
        "kind": "decay",
        "clock": "prefill_service",
        "half_life_ms": 40_000.0,
    },
}

CONTROL_POLICIES = (
    "fair_gain_4_cumulative",
    "fair_gain_4_window_512",
)
STATIONARY_CONTROL_POLICIES = (
    "fair_gain_4_cumulative",
)


def make_decayed_fair_selector(
    *,
    clock: str,
    half_life_ms: float,
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[dict[str, float]],
]:
    if clock not in (
        "wall_time",
        "prefill_service",
    ):
        raise ValueError(clock)
    if half_life_ms <= 0.0:
        raise ValueError(half_life_ms)
    offered = {
        prompt: 0.0
        for prompt in PROMPT_CLASSES
    }
    accepted = {
        prompt: 0.0
        for prompt in PROMPT_CLASSES
    }
    previous_now_ms: float | None = None
    decisions: list[dict[str, float]] = []

    def select(state: dict[str, Any]) -> bool:
        nonlocal previous_now_ms
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
        now_ms = float(state["now_ms"])
        if clock == "wall_time":
            clock_delta_ms = (
                0.0
                if previous_now_ms is None
                else max(
                    0.0,
                    now_ms - previous_now_ms,
                )
            )
            previous_now_ms = now_ms
        else:
            clock_delta_ms = decision[
                "candidate_prefill_ms"
            ]
        decay_factor = 2.0 ** (
            -clock_delta_ms / half_life_ms
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
                "decay_clock": (
                    0.0
                    if clock == "wall_time"
                    else 1.0
                ),
                "decay_half_life_ms": (
                    half_life_ms
                ),
                "decay_clock_delta_ms": (
                    clock_delta_ms
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


def decay_metrics(
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
            "Decay policy made no decisions"
        )
    factors = np.asarray(
        [
            decision["decay_factor"]
            for decision in selected
        ],
        dtype=np.float64,
    )
    deltas = np.asarray(
        [
            decision[
                "decay_clock_delta_ms"
            ]
            for decision in selected
        ],
        dtype=np.float64,
    )
    history = np.asarray(
        [
            decision[
                "effective_history_weight_before"
            ]
            for decision in selected
        ],
        dtype=np.float64,
    )
    metrics = {
        "mean_decay_factor": float(
            np.mean(factors)
        ),
        "p05_decay_factor": float(
            np.percentile(factors, 5)
        ),
        "mean_decay_clock_delta_ms": float(
            np.mean(deltas)
        ),
        "p95_decay_clock_delta_ms": float(
            np.percentile(deltas, 95)
        ),
        "mean_effective_history_weight": float(
            np.mean(history)
        ),
        "p95_effective_history_weight": float(
            np.percentile(history, 95)
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
            f"prompt_{prompt}_mean_decay_factor"
        ] = statistics.mean(
            decision["decay_factor"]
            for decision in members
        )
        metrics[
            f"prompt_{prompt}_mean_history_weight"
        ] = statistics.mean(
            decision[
                "effective_history_weight_before"
            ]
            for decision in members
        )
    return metrics


def simulate_decayed_policy(
    *,
    arrivals: np.ndarray,
    workload: dict[str, np.ndarray],
    phase_ids: np.ndarray | None,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    policy_name: str,
    kv_budget_tokens: int,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    configuration = POLICIES[policy_name]
    (
        admission_selector,
        admission_decisions,
    ) = make_decayed_fair_selector(
        clock=configuration["clock"],
        half_life_ms=configuration[
            "half_life_ms"
        ],
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
    if phase_ids is not None:
        metrics.update(
            phase_metrics(
                admission_decisions,
                phase_ids=phase_ids,
                analysis_start=analysis_start,
                analysis_end=analysis_end,
            )
        )
    return metrics


def e54_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "nonstationary_fair_admission"
        / "nonstationary_fair_admission_full.json"
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
    stationary_loads = [
        float(value)
        for value in args.stationary_loads.split(
            ","
        )
    ]
    shifted_loads = [
        float(value)
        for value in args.shifted_loads.split(",")
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
    matrices = (
        (
            "stationary_mixed",
            stationary_loads,
            STATIONARY_CONTROL_POLICIES,
        ),
        (
            "rotating_prompt_mix",
            shifted_loads,
            tuple(POLICIES),
        ),
    )
    for (
        workload_mode,
        offered_loads,
        policy_names,
    ) in matrices:
        for prefix_mode in PREFIX_MODES:
            for offered_load in offered_loads:
                for policy_name in policy_names:
                    scenarios = []
                    for (
                        replicate_index,
                        decode_model,
                    ) in enumerate(
                        decode_service_models[
                            "attention_replicates"
                        ]
                    ):
                        source = decode_model[
                            "source"
                        ]
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
                            if (
                                workload_mode
                                == "stationary_mixed"
                            ):
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
                                phase_ids = None
                            else:
                                (
                                    workload,
                                    phase_ids,
                                    phase_ranges,
                                ) = (
                                    make_nonstationary_workload(
                                        num_requests=(
                                            args.num_requests
                                        ),
                                        seed=workload_seed,
                                        prefix_mode=(
                                            prefix_mode
                                        ),
                                    )
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
                                seed=(
                                    workload_seed + 1
                                ),
                            )
                            if (
                                policy_name
                                in CONTROL_POLICIES
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
                            else:
                                metrics = (
                                    simulate_decayed_policy(
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
                                workload_mode
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

    reference = e54_records(args.root)
    equivalence = []
    equivalence_enabled = (
        args.num_requests == 2_000
        and args.warmup_requests == 200
        and args.tail_guard_requests == 100
        and args.capacity_requests == 1_500
        and args.seeds == 3
        and args.seed == 40_000
        and stationary_loads
        == [0.1, 0.2, 0.3, 0.4]
        and shifted_loads == [0.3, 0.4]
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
                    record["workload_mode"],
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
                        "E54 control equivalence "
                        "failed: "
                        f"{record['policy']=} "
                        f"{key=} "
                        f"{maximum_difference=}"
                    )
                equivalence.append(
                    {
                        "workload_mode": record[
                            "workload_mode"
                        ],
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

    result = {
        "suite": (
            "decayed_weighted_fair_"
            "admission_simulation"
        ),
        "model": (
            "The E54 rotating-prompt study with "
            "gain-4 fairness state decayed by wall "
            "time or predicted prefill GPU service."
        ),
        "prefill_service_model": prefill_model,
        "decode_service_models": (
            decode_service_models
        ),
        "workload": {
            "stationary": "mixed_chat",
            "nonstationary_phases": PHASES,
            "phase_ranges": (
                phase_ranges_reference
            ),
            "prefix_modes": PREFIX_MODES,
        },
        "controller": {
            "gain": 4.0,
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
            "wall_half_lives_ms": (
                20_000.0,
                40_000.0,
            ),
            "service_half_lives_ms": (
                20_000.0,
                40_000.0,
            ),
            "decay_rule": (
                "observed class counts multiply by "
                "2**(-clock_delta / half_life) "
                "before each admission decision"
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
            "stationary_loads_vs_decode_capacity": (
                stationary_loads
            ),
            "shifted_loads_vs_decode_capacity": (
                shifted_loads
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
            "e54_equivalence_enabled": (
                equivalence_enabled
            ),
            "e54_equivalence_checks": (
                equivalence
            ),
            "e54_maximum_absolute_difference": (
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
                "Phase counts are exact, decayed "
                "accepted weight never exceeds "
                "offered weight, every admitted "
                "request satisfies its dynamic "
                "threshold, and hard slices report "
                "zero budget violations."
            ),
        },
        "records": records,
        "limitations": [
            "The service model inherits all E49-E54 limitations and is not a live serving-engine measurement.",
            "Wall-time and service half-lives are hand-selected sensitivity points calibrated from this synthetic load range.",
            "Service decay advances on offered candidate prefill work, not realized total GPU occupancy or end-to-end request cost.",
            "The controller still optimizes equal prompt-class acceptance rather than tenant, token, or utility-weighted fairness.",
            "Rejected requests leave immediately and are not retried, routed to another replica, or placed in a reservation class.",
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
        "--stationary-loads",
        default="0.1,0.2,0.3,0.4",
    )
    parser.add_argument(
        "--shifted-loads",
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
