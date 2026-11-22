from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
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
from simulate_weighted_fair_admission import (
    BASE_LIMIT_MS,
    BASE_SAFETY_FACTOR,
    MAX_CREDIT,
    MIN_CREDIT,
    PRIOR_ACCEPTANCE,
    PRIOR_COUNT,
    PROMPT_CLASSES,
    make_weighted_fair_selector,
    weighted_fair_metrics,
)


PHASES = (
    {
        "name": "short_initial",
        "prompt_probabilities": (
            0.80,
            0.15,
            0.05,
        ),
        "dominant_prompt": 512,
    },
    {
        "name": "long_surge",
        "prompt_probabilities": (
            0.15,
            0.20,
            0.65,
        ),
        "dominant_prompt": 6144,
    },
    {
        "name": "medium_surge",
        "prompt_probabilities": (
            0.15,
            0.70,
            0.15,
        ),
        "dominant_prompt": 2048,
    },
    {
        "name": "short_return",
        "prompt_probabilities": (
            0.80,
            0.15,
            0.05,
        ),
        "dominant_prompt": 512,
    },
)

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
    "fair_gain_4_cumulative": {
        "kind": "cumulative",
        "gain": 4.0,
    },
    "fair_gain_4_window_128": {
        "kind": "rolling",
        "gain": 4.0,
        "window_requests": 128,
    },
    "fair_gain_4_window_256": {
        "kind": "rolling",
        "gain": 4.0,
        "window_requests": 256,
    },
    "fair_gain_4_window_512": {
        "kind": "rolling",
        "gain": 4.0,
        "window_requests": 512,
    },
}

STATIONARY_POLICIES = (
    "safety_8x",
    "safety_8x_sqrt_prompt_credit",
    "fair_gain_4_cumulative",
)


def exact_class_counts(
    *,
    length: int,
    probabilities: tuple[float, ...],
) -> list[int]:
    raw = np.asarray(
        probabilities,
        dtype=np.float64,
    ) * length
    counts = np.floor(raw).astype(np.int64)
    remainder = length - int(np.sum(counts))
    order = np.argsort(
        -(raw - counts),
        kind="stable",
    )
    for index in order[:remainder]:
        counts[index] += 1
    if int(np.sum(counts)) != length:
        raise RuntimeError(
            "Phase class-count allocation failed"
        )
    return [
        int(value)
        for value in counts
    ]


def make_nonstationary_workload(
    *,
    num_requests: int,
    seed: int,
    prefix_mode: str,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    list[dict[str, Any]],
]:
    base = make_workload(
        name="mixed_chat",
        num_requests=num_requests,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 91)
    phase_ids = np.empty(
        num_requests,
        dtype=np.int64,
    )
    prompt_tokens = np.empty(
        num_requests,
        dtype=np.int64,
    )
    phase_lengths = [
        num_requests // len(PHASES)
        for _ in PHASES
    ]
    for index in range(
        num_requests % len(PHASES)
    ):
        phase_lengths[index] += 1
    phase_ranges = []
    start = 0
    for phase_index, (
        phase,
        length,
    ) in enumerate(
        zip(PHASES, phase_lengths)
    ):
        end = start + length
        counts = exact_class_counts(
            length=length,
            probabilities=phase[
                "prompt_probabilities"
            ],
        )
        values = np.concatenate(
            [
                np.full(
                    count,
                    prompt,
                    dtype=np.int64,
                )
                for prompt, count in zip(
                    PROMPT_CLASSES,
                    counts,
                )
            ]
        )
        rng.shuffle(values)
        prompt_tokens[start:end] = values
        phase_ids[start:end] = phase_index
        phase_ranges.append(
            {
                **phase,
                "index": phase_index,
                "start_request": start,
                "end_request": end,
                "request_count": length,
                "prompt_class_counts": {
                    str(prompt): count
                    for prompt, count in zip(
                        PROMPT_CLASSES,
                        counts,
                    )
                },
            }
        )
        start = end
    if start != num_requests:
        raise RuntimeError(
            "Phase partition did not cover workload"
        )

    specification = PREFIX_MODES[prefix_mode]
    cache_rng = np.random.default_rng(seed + 47)
    hits = (
        cache_rng.random(num_requests)
        < specification[
            "request_hit_probability"
        ]
    )
    cached_tokens = np.zeros(
        num_requests,
        dtype=np.int64,
    )
    if specification[
        "cached_prompt_fraction"
    ] > 0.0:
        cached_values = (
            prompt_tokens
            * specification[
                "cached_prompt_fraction"
            ]
        ).astype(np.int64)
        cached_values = (
            cached_values // 16
        ) * 16
        cached_tokens[hits] = (
            cached_values[hits]
        )
    uncached_tokens = (
        prompt_tokens - cached_tokens
    )
    if np.any(uncached_tokens < 1):
        raise RuntimeError(
            "Prefix mode cached the full prompt"
        )
    output_tokens = base["output_tokens"]
    workload = {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "reservation_tokens": (
            prompt_tokens + output_tokens
        ),
        "cached_tokens": cached_tokens,
        "uncached_tokens": uncached_tokens,
        "prefix_hits": hits,
    }
    return workload, phase_ids, phase_ranges


def make_rolling_fair_selector(
    *,
    gain: float,
    window_requests: int,
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
    history: deque[tuple[int, int]] = (
        deque()
    )
    decisions: list[dict[str, float]] = []

    def select(state: dict[str, Any]) -> bool:
        request = int(state["request"])
        prompt = int(
            state["prompt_tokens"][request]
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
        history.append(
            (prompt, int(allow))
        )
        if len(history) > window_requests:
            old_prompt, old_allow = (
                history.popleft()
            )
            offered[old_prompt] -= 1
            accepted[old_prompt] -= (
                old_allow
            )
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
                "controller_window_requests": (
                    float(window_requests)
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
                "class_offered_before": float(
                    offered[prompt] - 1
                ),
                "class_accepted_before": float(
                    accepted[prompt]
                    - int(allow)
                ),
                "history_size_after": float(
                    len(history)
                ),
            }
        )
        return allow

    return select, decisions


def jain_index(
    values: list[float],
) -> float:
    array = np.asarray(
        values,
        dtype=np.float64,
    )
    denominator = (
        array.size
        * float(np.sum(array * array))
    )
    return (
        float(np.sum(array) ** 2 / denominator)
        if denominator > 0.0
        else 1.0
    )


def segment_metrics(
    decisions: list[dict[str, float]],
    *,
    dominant_prompt: int,
) -> dict[str, float]:
    if not decisions:
        raise RuntimeError(
            "Empty phase segment"
        )
    acceptance = {}
    for prompt in PROMPT_CLASSES:
        members = [
            decision
            for decision in decisions
            if int(
                decision["prompt_tokens"]
            )
            == prompt
        ]
        if not members:
            acceptance[prompt] = 0.0
        else:
            acceptance[prompt] = (
                sum(
                    decision["allowed"]
                    for decision in members
                )
                / len(members)
            )
    dominant = [
        decision
        for decision in decisions
        if int(decision["prompt_tokens"])
        == dominant_prompt
    ]
    return {
        "accepted_fraction": (
            sum(
                decision["allowed"]
                for decision in decisions
            )
            / len(decisions)
        ),
        "prompt_acceptance_jain_index": (
            jain_index(
                list(acceptance.values())
            )
        ),
        "dominant_acceptance": (
            sum(
                decision["allowed"]
                for decision in dominant
            )
            / len(dominant)
            if dominant
            else 0.0
        ),
        "dominant_mean_credit": (
            statistics.mean(
                decision[
                    "prompt_credit_multiplier"
                ]
                for decision in dominant
            )
            if dominant
            else 0.0
        ),
    }


def phase_metrics(
    decisions: list[dict[str, float]],
    *,
    phase_ids: np.ndarray,
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
    metrics: dict[str, float] = {}
    phase_jain = []
    dominant_acceptance = []
    dominant_credit = []
    for phase_index, phase in enumerate(
        PHASES
    ):
        members = [
            decision
            for decision in selected
            if int(
                phase_ids[
                    int(decision["request"])
                ]
            )
            == phase_index
        ]
        if not members:
            raise RuntimeError(
                f"No analyzed requests in phase {phase_index}"
            )
        name = phase["name"]
        dominant = int(
            phase["dominant_prompt"]
        )
        summary = segment_metrics(
            members,
            dominant_prompt=dominant,
        )
        for key, value in summary.items():
            metrics[
                f"phase_{name}_{key}"
            ] = value
        for prompt in PROMPT_CLASSES:
            prompt_members = [
                decision
                for decision in members
                if int(
                    decision["prompt_tokens"]
                )
                == prompt
            ]
            if not prompt_members:
                raise RuntimeError(
                    "Phase lost a prompt class"
                )
            metrics[
                f"phase_{name}_prompt_{prompt}_offered_fraction"
            ] = (
                len(prompt_members)
                / len(members)
            )
            metrics[
                f"phase_{name}_prompt_{prompt}_acceptance"
            ] = (
                sum(
                    decision["allowed"]
                    for decision
                    in prompt_members
                )
                / len(prompt_members)
            )
            metrics[
                f"phase_{name}_prompt_{prompt}_mean_credit"
            ] = statistics.mean(
                decision[
                    "prompt_credit_multiplier"
                ]
                for decision
                in prompt_members
            )
        segment_size = min(
            64,
            max(1, len(members) // 4),
        )
        first = segment_metrics(
            members[:segment_size],
            dominant_prompt=dominant,
        )
        last = segment_metrics(
            members[-segment_size:],
            dominant_prompt=dominant,
        )
        for key, value in first.items():
            metrics[
                f"phase_{name}_first_segment_{key}"
            ] = value
        for key, value in last.items():
            metrics[
                f"phase_{name}_last_segment_{key}"
            ] = value
        metrics[
            f"phase_{name}_dominant_credit_delta"
        ] = (
            last["dominant_mean_credit"]
            - first["dominant_mean_credit"]
        )
        phase_jain.append(
            summary[
                "prompt_acceptance_jain_index"
            ]
        )
        dominant_acceptance.append(
            summary["dominant_acceptance"]
        )
        dominant_credit.append(
            summary["dominant_mean_credit"]
        )
    metrics[
        "phase_mean_prompt_acceptance_jain_index"
    ] = statistics.mean(phase_jain)
    metrics[
        "phase_minimum_prompt_acceptance_jain_index"
    ] = min(phase_jain)
    metrics[
        "phase_dominant_acceptance_range"
    ] = (
        max(dominant_acceptance)
        - min(dominant_acceptance)
    )
    metrics[
        "phase_dominant_credit_range"
    ] = (
        max(dominant_credit)
        - min(dominant_credit)
    )
    metrics[
        "short_return_vs_initial_acceptance"
    ] = (
        metrics[
            "phase_short_return_"
            "dominant_acceptance"
        ]
        - metrics[
            "phase_short_initial_"
            "dominant_acceptance"
        ]
    )
    metrics[
        "short_return_vs_initial_credit"
    ] = (
        metrics[
            "phase_short_return_"
            "dominant_mean_credit"
        ]
        - metrics[
            "phase_short_initial_"
            "dominant_mean_credit"
        ]
    )
    return metrics


def simulate_policy(
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
    elif configuration["kind"] == (
        "cumulative"
    ):
        (
            admission_selector,
            admission_decisions,
        ) = make_weighted_fair_selector(
            gain=configuration["gain"],
            prefill_model=prefill_model,
        )
    else:
        (
            admission_selector,
            admission_decisions,
        ) = make_rolling_fair_selector(
            gain=configuration["gain"],
            window_requests=configuration[
                "window_requests"
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
    if configuration["kind"] in (
        "cumulative",
        "rolling",
    ):
        metrics.update(
            weighted_fair_metrics(
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


def e53_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "weighted_fair_admission"
        / "weighted_fair_admission_full.json"
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
            STATIONARY_POLICIES,
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
                            metrics = simulate_policy(
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

    reference = e53_records(args.root)
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
    )
    if equivalence_enabled:
        policy_mapping = {
            "safety_8x": "safety_8x",
            "safety_8x_sqrt_prompt_credit": (
                "safety_8x_sqrt_prompt_credit"
            ),
            "fair_gain_4_cumulative": (
                "fair_gain_4"
            ),
        }
        for record in records:
            if (
                record["workload_mode"]
                != "stationary_mixed"
            ):
                continue
            expected_record = reference[
                (
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    policy_mapping[
                        record["policy"]
                    ],
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
                        "E53 control equivalence "
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

    result = {
        "suite": (
            "nonstationary_weighted_fair_"
            "admission_simulation"
        ),
        "model": (
            "The E53 hard-100ms, 8x-safety "
            "controller under rotating prompt "
            "mixtures, comparing cumulative and "
            "bounded rolling fairness state."
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
            "rolling_windows": (
                128,
                256,
                512,
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
            "e53_equivalence_enabled": (
                equivalence_enabled
            ),
            "e53_equivalence_checks": (
                equivalence
            ),
            "e53_maximum_absolute_difference": (
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
                "Phase prompt counts are allocated "
                "exactly, rolling histories never "
                "exceed their bound, every admitted "
                "request satisfies its dynamic "
                "threshold, and hard slices report "
                "zero budget violations."
            ),
        },
        "records": records,
        "limitations": [
            "The service model inherits all E49-E53 limitations and is not a live serving-engine measurement.",
            "The phase sequence is synthetic, fixed-length, and changes prompt mix rather than tenant identity or request utility.",
            "Rolling state is bounded by request count, not wall time, token cost, or per-tenant service.",
            "The controller still optimizes equal prompt-class acceptance rather than token, tenant, or utility-weighted fairness.",
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
