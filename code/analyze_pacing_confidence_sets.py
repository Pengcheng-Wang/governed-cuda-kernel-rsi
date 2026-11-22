from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analyze_pacing_policy_compression import (
    FAMILIES,
    group_dict,
    group_key,
    index_records,
    load_e62,
    oracle_windows_from_e62,
    state_dict,
)
from analyze_pacing_state_uncertainty import (
    ERROR_PROFILES,
    estimate_distribution,
    evaluate_strategy,
    hamming1_mapping,
    hamming_distance,
    load_e63,
    robust_mapping_from_e63,
)


CONFIDENCE_PROFILES = {
    "oracle": {
        "mistake_flag_sensitivity": 1.0,
        "correct_flag_false_positive_rate": 0.0,
    },
    "high_recall_clean": {
        "mistake_flag_sensitivity": 0.95,
        "correct_flag_false_positive_rate": 0.05,
    },
    "high_recall_noisy": {
        "mistake_flag_sensitivity": 0.95,
        "correct_flag_false_positive_rate": 0.20,
    },
    "moderate_clean": {
        "mistake_flag_sensitivity": 0.80,
        "correct_flag_false_positive_rate": 0.05,
    },
    "weak_noisy": {
        "mistake_flag_sensitivity": 0.60,
        "correct_flag_false_positive_rate": 0.20,
    },
    "always_flag": {
        "mistake_flag_sensitivity": 1.0,
        "correct_flag_false_positive_rate": 1.0,
    },
}

BASELINE_STRATEGIES = (
    "naive",
    "hamming1_safe",
)

COMPARISON_METRICS = (
    "mean_expected_throughput_ratio_vs_oracle",
    "geomean_expected_throughput_ratio_vs_oracle",
    "worst_expected_state_throughput_ratio_vs_oracle",
    "mean_expected_acceptance_delta_vs_oracle",
    "worst_expected_acceptance_delta_vs_oracle",
    "expected_scenario_slo_violation_fraction",
    "unsafe_path_probability",
    "maximum_slo_excess_ms",
    "expected_selected_window_ms",
    "expected_key_error_count",
)


def load_e64(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_state_uncertainty"
        / "pacing_state_uncertainty_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def confidence_flag_distribution(
    *,
    true_key: tuple[Any, ...],
    estimated_key: tuple[Any, ...],
    sensitivity: float,
    false_positive_rate: float,
) -> list[tuple[tuple[bool, ...], float]]:
    if len(true_key) != len(estimated_key):
        raise ValueError("Confidence key length mismatch")
    outcomes = []
    for flags in itertools.product(
        (False, True),
        repeat=len(true_key),
    ):
        probability = 1.0
        for true_value, estimated_value, flagged in zip(
            true_key,
            estimated_key,
            flags,
        ):
            flag_rate = (
                sensitivity
                if true_value != estimated_value
                else false_positive_rate
            )
            probability *= (
                flag_rate
                if flagged
                else 1.0 - flag_rate
            )
        if probability > 0.0:
            outcomes.append((flags, probability))
    total = sum(probability for _, probability in outcomes)
    if abs(total - 1.0) > 1e-12:
        raise RuntimeError("Confidence distribution failed")
    return outcomes


def confidence_set_mapping(
    *,
    family: str,
    slo_ms: float,
    states: list[tuple[str, float, str]],
    windows: list[float],
    lookup: dict[
        tuple[str, float, str, float],
        dict[str, Any],
    ],
) -> tuple[
    dict[tuple[tuple[Any, ...], tuple[bool, ...]], float],
    list[dict[str, Any]],
]:
    dimensions = FAMILIES[family]
    keys = sorted(
        {
            group_key(state, dimensions)
            for state in states
        }
    )
    mapping = {}
    details = []
    for estimated_key in keys:
        for flags in itertools.product(
            (False, True),
            repeat=len(dimensions),
        ):
            neighborhood = [
                state
                for state in states
                if all(
                    flagged
                    or value == estimated_value
                    for value, estimated_value, flagged in zip(
                        group_key(state, dimensions),
                        estimated_key,
                        flags,
                    )
                )
            ]
            feasible = [
                window
                for window in windows
                if all(
                    lookup[(*state, window)]["metrics"][
                        "p99_ttft_ms"
                    ]["maximum"]
                    <= slo_ms + 1e-9
                    for state in neighborhood
                )
            ]
            if not feasible:
                raise RuntimeError(
                    "Confidence set became infeasible: "
                    f"{family=} {slo_ms=} {estimated_key=} {flags=}"
                )
            selected = max(
                feasible,
                key=lambda window: (
                    sum(
                        lookup[(*state, window)]["metrics"][
                            "throughput_output_tokens_s"
                        ]["median"]
                        for state in neighborhood
                    ),
                    -max(
                        lookup[(*state, window)]["metrics"][
                            "p99_ttft_ms"
                        ]["maximum"]
                        for state in neighborhood
                    ),
                    -window,
                ),
            )
            mapping[(estimated_key, flags)] = selected
            covered_keys = sorted(
                {
                    group_key(state, dimensions)
                    for state in neighborhood
                }
            )
            details.append(
                {
                    "estimated_key": group_dict(
                        dimensions,
                        estimated_key,
                    ),
                    "uncertain_dimensions": [
                        dimension
                        for dimension, flagged in zip(
                            dimensions,
                            flags,
                        )
                        if flagged
                    ],
                    "confidence_set_key_count": len(covered_keys),
                    "confidence_set_keys": [
                        group_dict(dimensions, key)
                        for key in covered_keys
                    ],
                    "member_state_count": len(neighborhood),
                    "selected_window_ms": selected,
                    "maximum_confidence_set_p99_ttft_ms": max(
                        lookup[(*state, selected)]["metrics"][
                            "p99_ttft_ms"
                        ]["maximum"]
                        for state in neighborhood
                    ),
                }
            )
    return mapping, details


def evaluate_confidence_profile(
    *,
    family: str,
    slo_ms: float,
    error_profile: str,
    confidence_profile: str,
    mapping: dict[
        tuple[tuple[Any, ...], tuple[bool, ...]],
        float,
    ],
    states: list[tuple[str, float, str]],
    lookup: dict[
        tuple[str, float, str, float],
        dict[str, Any],
    ],
    oracle_windows: dict[
        tuple[str, float, str],
        float,
    ],
    baselines: dict[str, dict[str, Any]],
    confidence_parameters: (
        dict[str, float] | None
    ) = None,
) -> dict[str, Any]:
    dimensions = FAMILIES[family]
    error_probabilities = ERROR_PROFILES[error_profile]
    confidence = (
        CONFIDENCE_PROFILES[
            confidence_profile
        ]
        if confidence_parameters is None
        else confidence_parameters
    )
    sensitivity = float(
        confidence["mistake_flag_sensitivity"]
    )
    false_positive_rate = float(
        confidence["correct_flag_false_positive_rate"]
    )
    state_weight = 1.0 / len(states)
    state_records = []
    expected_ratios = []
    expected_acceptance_deltas = []
    scenario_violation_probability = 0.0
    unsafe_path_probability = 0.0
    confidence_miss_probability = 0.0
    any_false_alarm_probability = 0.0
    expected_missed_key_error_count = 0.0
    expected_false_alarm_count = 0.0
    expected_uncertain_dimension_count = 0.0
    expected_confidence_set_key_count = 0.0
    expected_selected_window_ms = 0.0
    expected_key_error_count = 0.0
    no_flag_probability = 0.0
    all_key_dimensions_flagged_probability = 0.0
    maximum_excess_ms = 0.0
    for true_state in states:
        true_key = group_key(true_state, dimensions)
        oracle_window = oracle_windows[true_state]
        oracle_record = lookup[(*true_state, oracle_window)]
        oracle_throughput = oracle_record["metrics"][
            "throughput_output_tokens_s"
        ]["median"]
        oracle_acceptance = oracle_record["metrics"][
            "accepted_fraction"
        ]["median"]
        state_expected_ratio = 0.0
        state_expected_acceptance_delta = 0.0
        state_scenario_violation_probability = 0.0
        state_unsafe_path_probability = 0.0
        state_confidence_miss_probability = 0.0
        state_any_false_alarm_probability = 0.0
        state_expected_missed_key_errors = 0.0
        state_expected_false_alarms = 0.0
        state_expected_uncertain_dimensions = 0.0
        state_expected_confidence_set_keys = 0.0
        state_expected_window = 0.0
        state_expected_key_errors = 0.0
        state_no_flag_probability = 0.0
        state_all_flagged_probability = 0.0
        state_maximum_excess = 0.0
        window_distribution: dict[float, float] = {}
        for (
            estimated_state,
            estimate_probability,
            _,
        ) in estimate_distribution(
            true_state=true_state,
            probabilities=error_probabilities,
        ):
            estimated_key = group_key(
                estimated_state,
                dimensions,
            )
            mismatch = tuple(
                true_value != estimated_value
                for true_value, estimated_value in zip(
                    true_key,
                    estimated_key,
                )
            )
            key_error_count = sum(mismatch)
            for (
                flags,
                confidence_probability,
            ) in confidence_flag_distribution(
                true_key=true_key,
                estimated_key=estimated_key,
                sensitivity=sensitivity,
                false_positive_rate=false_positive_rate,
            ):
                probability = (
                    estimate_probability
                    * confidence_probability
                )
                window = mapping[(estimated_key, flags)]
                record = lookup[(*true_state, window)]
                throughput = record["metrics"][
                    "throughput_output_tokens_s"
                ]["median"]
                ratio = throughput / oracle_throughput
                acceptance_delta = (
                    record["metrics"]["accepted_fraction"][
                        "median"
                    ]
                    - oracle_acceptance
                )
                violations = [
                    max(
                        0.0,
                        float(scenario["p99_ttft_ms"])
                        - slo_ms,
                    )
                    for scenario in record["scenarios"]
                ]
                violation_fraction = (
                    sum(excess > 0.0 for excess in violations)
                    / len(violations)
                )
                path_maximum_excess = max(violations)
                unsafe = path_maximum_excess > 0.0
                missed_errors = sum(
                    wrong and not flagged
                    for wrong, flagged in zip(
                        mismatch,
                        flags,
                    )
                )
                false_alarms = sum(
                    not wrong and flagged
                    for wrong, flagged in zip(
                        mismatch,
                        flags,
                    )
                )
                confidence_miss = missed_errors > 0
                any_false_alarm = false_alarms > 0
                if not confidence_miss and unsafe:
                    raise RuntimeError(
                        "Covered confidence set violated SLO: "
                        f"{family=} {slo_ms=} "
                        f"{true_key=} {estimated_key=} {flags=}"
                    )
                flagged_count = sum(flags)
                set_key_count = 2**flagged_count
                all_flagged = (
                    bool(dimensions)
                    and flagged_count == len(dimensions)
                )
                state_expected_ratio += probability * ratio
                state_expected_acceptance_delta += (
                    probability * acceptance_delta
                )
                state_scenario_violation_probability += (
                    probability * violation_fraction
                )
                state_unsafe_path_probability += (
                    probability * float(unsafe)
                )
                state_confidence_miss_probability += (
                    probability * float(confidence_miss)
                )
                state_any_false_alarm_probability += (
                    probability * float(any_false_alarm)
                )
                state_expected_missed_key_errors += (
                    probability * missed_errors
                )
                state_expected_false_alarms += (
                    probability * false_alarms
                )
                state_expected_uncertain_dimensions += (
                    probability * flagged_count
                )
                state_expected_confidence_set_keys += (
                    probability * set_key_count
                )
                state_expected_window += probability * window
                state_expected_key_errors += (
                    probability * key_error_count
                )
                state_no_flag_probability += (
                    probability * float(flagged_count == 0)
                )
                state_all_flagged_probability += (
                    probability * float(all_flagged)
                )
                state_maximum_excess = max(
                    state_maximum_excess,
                    path_maximum_excess,
                )
                window_distribution[window] = (
                    window_distribution.get(window, 0.0)
                    + probability
                )
        state_records.append(
            {
                **state_dict(true_state),
                "oracle_window_ms": oracle_window,
                "expected_throughput_ratio_vs_oracle": (
                    state_expected_ratio
                ),
                "expected_acceptance_delta_vs_oracle": (
                    state_expected_acceptance_delta
                ),
                "expected_scenario_slo_violation_fraction": (
                    state_scenario_violation_probability
                ),
                "unsafe_path_probability": (
                    state_unsafe_path_probability
                ),
                "confidence_miss_probability": (
                    state_confidence_miss_probability
                ),
                "any_false_alarm_probability": (
                    state_any_false_alarm_probability
                ),
                "expected_missed_key_error_count": (
                    state_expected_missed_key_errors
                ),
                "expected_false_alarm_count": (
                    state_expected_false_alarms
                ),
                "expected_uncertain_dimension_count": (
                    state_expected_uncertain_dimensions
                ),
                "expected_confidence_set_key_count": (
                    state_expected_confidence_set_keys
                ),
                "expected_selected_window_ms": (
                    state_expected_window
                ),
                "expected_key_error_count": (
                    state_expected_key_errors
                ),
                "no_flag_probability": state_no_flag_probability,
                "all_key_dimensions_flagged_probability": (
                    state_all_flagged_probability
                ),
                "maximum_slo_excess_ms": state_maximum_excess,
                "selected_window_distribution": [
                    {
                        "window_ms": window,
                        "probability": probability,
                    }
                    for window, probability in sorted(
                        window_distribution.items()
                    )
                ],
            }
        )
        expected_ratios.append(state_expected_ratio)
        expected_acceptance_deltas.append(
            state_expected_acceptance_delta
        )
        scenario_violation_probability += (
            state_weight
            * state_scenario_violation_probability
        )
        unsafe_path_probability += (
            state_weight
            * state_unsafe_path_probability
        )
        confidence_miss_probability += (
            state_weight
            * state_confidence_miss_probability
        )
        any_false_alarm_probability += (
            state_weight
            * state_any_false_alarm_probability
        )
        expected_missed_key_error_count += (
            state_weight
            * state_expected_missed_key_errors
        )
        expected_false_alarm_count += (
            state_weight
            * state_expected_false_alarms
        )
        expected_uncertain_dimension_count += (
            state_weight
            * state_expected_uncertain_dimensions
        )
        expected_confidence_set_key_count += (
            state_weight
            * state_expected_confidence_set_keys
        )
        expected_selected_window_ms += (
            state_weight * state_expected_window
        )
        expected_key_error_count += (
            state_weight * state_expected_key_errors
        )
        no_flag_probability += (
            state_weight * state_no_flag_probability
        )
        all_key_dimensions_flagged_probability += (
            state_weight * state_all_flagged_probability
        )
        maximum_excess_ms = max(
            maximum_excess_ms,
            state_maximum_excess,
        )
    record = {
        "family": family,
        "dimensions": list(dimensions),
        "key_count": len(
            {
                group_key(state, dimensions)
                for state in states
            }
        ),
        "strategy": "confidence_set_safe",
        "ttft_slo_ms": slo_ms,
        "error_profile": error_profile,
        "error_probabilities": error_probabilities,
        "confidence_profile": confidence_profile,
        "mistake_flag_sensitivity": sensitivity,
        "correct_flag_false_positive_rate": (
            false_positive_rate
        ),
        "mean_expected_throughput_ratio_vs_oracle": (
            statistics.mean(expected_ratios)
        ),
        "geomean_expected_throughput_ratio_vs_oracle": (
            math.exp(
                statistics.mean(
                    math.log(value)
                    for value in expected_ratios
                )
            )
        ),
        "worst_expected_state_throughput_ratio_vs_oracle": (
            min(expected_ratios)
        ),
        "mean_expected_acceptance_delta_vs_oracle": (
            statistics.mean(expected_acceptance_deltas)
        ),
        "worst_expected_acceptance_delta_vs_oracle": (
            min(expected_acceptance_deltas)
        ),
        "expected_scenario_slo_violation_fraction": (
            scenario_violation_probability
        ),
        "unsafe_path_probability": unsafe_path_probability,
        "confidence_miss_probability": (
            confidence_miss_probability
        ),
        "any_false_alarm_probability": (
            any_false_alarm_probability
        ),
        "expected_missed_key_error_count": (
            expected_missed_key_error_count
        ),
        "expected_false_alarm_count": (
            expected_false_alarm_count
        ),
        "expected_uncertain_dimension_count": (
            expected_uncertain_dimension_count
        ),
        "expected_confidence_set_key_count": (
            expected_confidence_set_key_count
        ),
        "expected_selected_window_ms": (
            expected_selected_window_ms
        ),
        "expected_key_error_count": expected_key_error_count,
        "no_flag_probability": no_flag_probability,
        "all_key_dimensions_flagged_probability": (
            all_key_dimensions_flagged_probability
        ),
        "maximum_slo_excess_ms": maximum_excess_ms,
        "states": state_records,
    }
    for baseline_name, baseline in baselines.items():
        prefix = f"{baseline_name}_baseline"
        record[
            f"{prefix}_mean_throughput_ratio_vs_oracle"
        ] = baseline[
            "mean_expected_throughput_ratio_vs_oracle"
        ]
        record[
            f"{prefix}_scenario_slo_violation_fraction"
        ] = baseline[
            "expected_scenario_slo_violation_fraction"
        ]
        record[
            f"{prefix}_unsafe_path_probability"
        ] = baseline["unsafe_path_probability"]
        record[
            f"mean_throughput_delta_vs_{baseline_name}"
        ] = (
            record[
                "mean_expected_throughput_ratio_vs_oracle"
            ]
            - baseline[
                "mean_expected_throughput_ratio_vs_oracle"
            ]
        )
        record[
            f"scenario_violation_delta_vs_{baseline_name}"
        ] = (
            record[
                "expected_scenario_slo_violation_fraction"
            ]
            - baseline[
                "expected_scenario_slo_violation_fraction"
            ]
        )
    return record


def compare_records(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, float]:
    return {
        metric: abs(float(left[metric]) - float(right[metric]))
        for metric in COMPARISON_METRICS
    }


def analytic_confidence_metrics(
    *,
    dimensions: tuple[str, ...],
    error_probabilities: dict[str, float],
    sensitivity: float,
    false_positive_rate: float,
) -> dict[str, float]:
    flag_probabilities = [
        float(error_probabilities[dimension]) * sensitivity
        + (
            1.0 - float(error_probabilities[dimension])
        )
        * false_positive_rate
        for dimension in dimensions
    ]
    return {
        "confidence_miss_probability": (
            1.0
            - math.prod(
                1.0
                - float(error_probabilities[dimension])
                * (1.0 - sensitivity)
                for dimension in dimensions
            )
        ),
        "any_false_alarm_probability": (
            1.0
            - math.prod(
                1.0
                - (
                    1.0
                    - float(error_probabilities[dimension])
                )
                * false_positive_rate
                for dimension in dimensions
            )
        ),
        "expected_missed_key_error_count": sum(
            float(error_probabilities[dimension])
            * (1.0 - sensitivity)
            for dimension in dimensions
        ),
        "expected_false_alarm_count": sum(
            (
                1.0
                - float(error_probabilities[dimension])
            )
            * false_positive_rate
            for dimension in dimensions
        ),
        "expected_uncertain_dimension_count": sum(
            flag_probabilities
        ),
        "expected_confidence_set_key_count": math.prod(
            1.0 + probability
            for probability in flag_probabilities
        ),
        "expected_key_error_count": sum(
            float(error_probabilities[dimension])
            for dimension in dimensions
        ),
        "no_flag_probability": math.prod(
            1.0 - probability
            for probability in flag_probabilities
        ),
        "all_key_dimensions_flagged_probability": (
            math.prod(flag_probabilities)
            if dimensions
            else 0.0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e62 = load_e62(args.root)
    e63 = load_e63(args.root)
    e64 = load_e64(args.root)
    states, windows, lookup = index_records(e62)
    slos_ms = [
        float(value)
        for value in args.ttft_slos_ms.split(",")
    ]
    families = [
        value.strip()
        for value in args.families.split(",")
        if value.strip()
    ]
    error_profiles = [
        value.strip()
        for value in args.error_profiles.split(",")
        if value.strip()
    ]
    confidence_profiles = [
        value.strip()
        for value in args.confidence_profiles.split(",")
        if value.strip()
    ]
    unknown_families = set(families) - set(FAMILIES)
    if unknown_families:
        raise ValueError(
            f"Unknown families: {sorted(unknown_families)}"
        )
    unknown_error_profiles = set(error_profiles) - set(
        ERROR_PROFILES
    )
    if unknown_error_profiles:
        raise ValueError(
            "Unknown error profiles: "
            f"{sorted(unknown_error_profiles)}"
        )
    unknown_confidence_profiles = set(
        confidence_profiles
    ) - set(CONFIDENCE_PROFILES)
    if unknown_confidence_profiles:
        raise ValueError(
            "Unknown confidence profiles: "
            f"{sorted(unknown_confidence_profiles)}"
        )
    e64_index = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            record["strategy"],
        ): record
        for record in e64["records"]
    }
    records = []
    mapping_records = []
    baseline_checks = []
    oracle_safety_checks = []
    all_flag_safety_checks = []
    no_flag_mapping_checks = []
    all_flag_mapping_checks = []
    probability_checks = []
    for slo_ms in slos_ms:
        oracle_windows = oracle_windows_from_e62(
            e62=e62,
            slo_ms=slo_ms,
        )
        universal_mapping = robust_mapping_from_e63(
            e63=e63,
            family="universal",
            slo_ms=slo_ms,
        )
        universal_window = next(
            iter(universal_mapping.values())
        )
        for family in families:
            naive_mapping = robust_mapping_from_e63(
                e63=e63,
                family=family,
                slo_ms=slo_ms,
            )
            (
                hamming_mapping,
                _,
            ) = hamming1_mapping(
                family=family,
                slo_ms=slo_ms,
                states=states,
                windows=windows,
                lookup=lookup,
            )
            (
                confidence_mapping,
                confidence_mapping_details,
            ) = confidence_set_mapping(
                family=family,
                slo_ms=slo_ms,
                states=states,
                windows=windows,
                lookup=lookup,
            )
            mapping_records.append(
                {
                    "ttft_slo_ms": slo_ms,
                    "family": family,
                    "dimensions": list(FAMILIES[family]),
                    "confidence_set_mapping": (
                        confidence_mapping_details
                    ),
                }
            )
            empty_flags = (False,) * len(FAMILIES[family])
            all_flags = (True,) * len(FAMILIES[family])
            for key, window in naive_mapping.items():
                difference = abs(
                    confidence_mapping[(key, empty_flags)]
                    - window
                )
                if difference > 1e-12:
                    raise RuntimeError(
                        "No-flag mapping did not reproduce E63"
                    )
                no_flag_mapping_checks.append(
                    {
                        "ttft_slo_ms": slo_ms,
                        "family": family,
                        "key": group_dict(
                            FAMILIES[family],
                            key,
                        ),
                        "absolute_difference_ms": difference,
                    }
                )
                all_flag_difference = abs(
                    confidence_mapping[(key, all_flags)]
                    - universal_window
                )
                if all_flag_difference > 1e-12:
                    raise RuntimeError(
                        "All-flag mapping did not reproduce "
                        "the universal robust window"
                    )
                all_flag_mapping_checks.append(
                    {
                        "ttft_slo_ms": slo_ms,
                        "family": family,
                        "key": group_dict(
                            FAMILIES[family],
                            key,
                        ),
                        "absolute_difference_ms": (
                            all_flag_difference
                        ),
                    }
                )
            for error_profile in error_profiles:
                baselines = {}
                for strategy, mapping in (
                    ("naive", naive_mapping),
                    ("hamming1_safe", hamming_mapping),
                ):
                    evaluated = evaluate_strategy(
                        family=family,
                        strategy=strategy,
                        slo_ms=slo_ms,
                        error_profile=error_profile,
                        mapping=mapping,
                        states=states,
                        lookup=lookup,
                        oracle_windows=oracle_windows,
                    )
                    expected = e64_index[
                        (
                            slo_ms,
                            family,
                            error_profile,
                            strategy,
                        )
                    ]
                    differences = compare_records(
                        evaluated,
                        expected,
                    )
                    maximum_difference = max(
                        differences.values()
                    )
                    if maximum_difference > 1e-12:
                        raise RuntimeError(
                            "E64 baseline reproduction failed: "
                            f"{family=} {slo_ms=} "
                            f"{error_profile=} {strategy=}"
                        )
                    baseline_checks.append(
                        {
                            "ttft_slo_ms": slo_ms,
                            "family": family,
                            "error_profile": error_profile,
                            "strategy": strategy,
                            "differences": differences,
                            "maximum_absolute_difference": (
                                maximum_difference
                            ),
                        }
                    )
                    baselines[strategy] = evaluated
                for confidence_profile in confidence_profiles:
                    evaluated = evaluate_confidence_profile(
                        family=family,
                        slo_ms=slo_ms,
                        error_profile=error_profile,
                        confidence_profile=confidence_profile,
                        mapping=confidence_mapping,
                        states=states,
                        lookup=lookup,
                        oracle_windows=oracle_windows,
                        baselines=baselines,
                    )
                    analytic = analytic_confidence_metrics(
                        dimensions=FAMILIES[family],
                        error_probabilities=ERROR_PROFILES[
                            error_profile
                        ],
                        sensitivity=CONFIDENCE_PROFILES[
                            confidence_profile
                        ]["mistake_flag_sensitivity"],
                        false_positive_rate=CONFIDENCE_PROFILES[
                            confidence_profile
                        ][
                            "correct_flag_false_positive_rate"
                        ],
                    )
                    probability_differences = {
                        metric: abs(
                            float(evaluated[metric])
                            - expected
                        )
                        for metric, expected in analytic.items()
                    }
                    maximum_probability_difference = max(
                        probability_differences.values(),
                        default=0.0,
                    )
                    if maximum_probability_difference > 1e-12:
                        raise RuntimeError(
                            "Confidence probability validation failed: "
                            f"{family=} {slo_ms=} "
                            f"{error_profile=} {confidence_profile=}"
                        )
                    if (
                        evaluated["unsafe_path_probability"]
                        > evaluated[
                            "confidence_miss_probability"
                        ]
                        + 1e-12
                        or evaluated[
                            "expected_scenario_slo_violation_fraction"
                        ]
                        > evaluated["unsafe_path_probability"]
                        + 1e-12
                    ):
                        raise RuntimeError(
                            "Confidence safety probability ordering failed"
                        )
                    probability_checks.append(
                        {
                            "ttft_slo_ms": slo_ms,
                            "family": family,
                            "error_profile": error_profile,
                            "confidence_profile": (
                                confidence_profile
                            ),
                            "differences": (
                                probability_differences
                            ),
                            "maximum_absolute_difference": (
                                maximum_probability_difference
                            ),
                        }
                    )
                    records.append(evaluated)
                    if confidence_profile == "oracle":
                        safety = max(
                            evaluated[
                                "expected_scenario_slo_violation_fraction"
                            ],
                            evaluated[
                                "unsafe_path_probability"
                            ],
                            evaluated[
                                "confidence_miss_probability"
                            ],
                        )
                        if safety > 1e-15:
                            raise RuntimeError(
                                "Oracle confidence was not safe"
                            )
                        oracle_safety_checks.append(
                            {
                                "ttft_slo_ms": slo_ms,
                                "family": family,
                                "error_profile": error_profile,
                                "maximum_safety_metric": safety,
                            }
                        )
                    if confidence_profile == "always_flag":
                        safety = max(
                            evaluated[
                                "expected_scenario_slo_violation_fraction"
                            ],
                            evaluated[
                                "unsafe_path_probability"
                            ],
                            evaluated[
                                "confidence_miss_probability"
                            ],
                        )
                        if safety > 1e-15:
                            raise RuntimeError(
                                "Always-flag confidence was not safe"
                            )
                        all_flag_safety_checks.append(
                            {
                                "ttft_slo_ms": slo_ms,
                                "family": family,
                                "error_profile": error_profile,
                                "maximum_safety_metric": safety,
                            }
                        )
    expected_record_count = (
        len(slos_ms)
        * len(families)
        * len(error_profiles)
        * len(confidence_profiles)
    )
    if len(records) != expected_record_count:
        raise RuntimeError("Unexpected E65 record count")
    result = {
        "suite": "pacing_confidence_set_analysis",
        "model": (
            "Condition an uncertainty-safe pacing lookup on "
            "per-feature confidence flags. A flagged feature "
            "may take either binary value; an unflagged feature "
            "is fixed to its estimate. The selected window must "
            "be SLO-feasible for every keyed state in that "
            "confidence set."
        ),
        "source": {
            "slo_pacing": (
                "results/raw/slo_ingress_pacing/"
                "slo_ingress_pacing_full.json"
            ),
            "policy_compression": (
                "results/raw/pacing_policy_compression/"
                "pacing_policy_compression_full.json"
            ),
            "state_uncertainty": (
                "results/raw/pacing_state_uncertainty/"
                "pacing_state_uncertainty_full.json"
            ),
        },
        "states": [state_dict(state) for state in states],
        "candidate_windows_ms": windows,
        "ttft_slos_ms": slos_ms,
        "families": {
            family: {
                "dimensions": list(FAMILIES[family]),
                "key_count": len(
                    {
                        group_key(state, FAMILIES[family])
                        for state in states
                    }
                ),
            }
            for family in families
        },
        "error_profiles": {
            profile: ERROR_PROFILES[profile]
            for profile in error_profiles
        },
        "confidence_profiles": {
            profile: CONFIDENCE_PROFILES[profile]
            for profile in confidence_profiles
        },
        "strategy": {
            "confidence_set_safe": (
                "Use confidence flags to form an axis-aligned "
                "set of possible true keys, then select the "
                "maximum-throughput window that is robustly "
                "SLO-feasible for every state in the set."
            ),
            "naive_baseline": (
                "E64 estimated-key lookup with no safety set."
            ),
            "hamming1_safe_baseline": (
                "E64 static envelope covering every key within "
                "Hamming distance one, regardless of which "
                "feature is low confidence."
            ),
        },
        "mappings": mapping_records,
        "records": records,
        "validation": {
            "e64_baseline_checks": baseline_checks,
            "e64_baseline_maximum_absolute_difference": max(
                item["maximum_absolute_difference"]
                for item in baseline_checks
            ),
            "no_flag_e63_mapping_checks": (
                no_flag_mapping_checks
            ),
            "no_flag_e63_mapping_maximum_absolute_difference_ms": max(
                item["absolute_difference_ms"]
                for item in no_flag_mapping_checks
            ),
            "all_flag_universal_mapping_checks": (
                all_flag_mapping_checks
            ),
            "all_flag_universal_mapping_maximum_absolute_difference_ms": max(
                item["absolute_difference_ms"]
                for item in all_flag_mapping_checks
            ),
            "oracle_confidence_safety_checks": (
                oracle_safety_checks
            ),
            "oracle_confidence_maximum_safety_metric": max(
                item["maximum_safety_metric"]
                for item in oracle_safety_checks
            ),
            "always_flag_safety_checks": (
                all_flag_safety_checks
            ),
            "always_flag_maximum_safety_metric": max(
                item["maximum_safety_metric"]
                for item in all_flag_safety_checks
            ),
            "analytic_confidence_probability_checks": (
                probability_checks
            ),
            "analytic_confidence_probability_maximum_absolute_difference": max(
                item["maximum_absolute_difference"]
                for item in probability_checks
            ),
            "safety_probability_ordering": (
                "For every record, scenario violation probability "
                "is no greater than unsafe-path probability, which "
                "is no greater than confidence-miss probability."
            ),
            "expected_record_count": expected_record_count,
        },
        "limitations": [
            "The confidence model uses conditional-independent binary flags with shared sensitivity and false-positive rates, not calibrated production confidence scores.",
            "The analysis inherits E62-E64's finite binary state matrix, pacing-window grid, and fitted-service simulation.",
            "True states are equally weighted; a production traffic distribution can change expected throughput and violation probability.",
            "Safety is deterministic only when every mistaken keyed feature is flagged; missed low-confidence errors retain residual risk.",
            "The lookup has no temporal hysteresis, switching cost, reject/defer action, or online confidence calibration.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ttft-slos-ms",
        default="650,800,1000",
    )
    parser.add_argument(
        "--families",
        default=(
            "universal,burst,load,cache,"
            "burst_load,burst_cache,load_cache,full"
        ),
    )
    parser.add_argument(
        "--error-profiles",
        default="all_5pct,all_10pct,all_20pct",
    )
    parser.add_argument(
        "--confidence-profiles",
        default=(
            "oracle,high_recall_clean,"
            "high_recall_noisy,moderate_clean,"
            "weak_noisy,always_flag"
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
