from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

from analyze_pacing_policy_compression import FAMILIES
from analyze_pacing_state_uncertainty import ERROR_PROFILES


CALIBRATION_SCENARIOS = {
    "balanced_margin": {
        "state_error_fraction_of_cap": 0.80,
        "mistake_flag_recall": 0.99,
        "false_alarm_fraction_of_cap": 0.50,
    },
    "very_high_recall": {
        "state_error_fraction_of_cap": 0.80,
        "mistake_flag_recall": 0.995,
        "false_alarm_fraction_of_cap": 0.50,
    },
    "tight_recall": {
        "state_error_fraction_of_cap": 0.80,
        "mistake_flag_recall": 0.975,
        "false_alarm_fraction_of_cap": 0.50,
    },
    "near_error_boundary": {
        "state_error_fraction_of_cap": 0.95,
        "mistake_flag_recall": 0.99,
        "false_alarm_fraction_of_cap": 0.50,
    },
    "near_false_alarm_boundary": {
        "state_error_fraction_of_cap": 0.80,
        "mistake_flag_recall": 0.99,
        "false_alarm_fraction_of_cap": 0.90,
    },
    "perfect_recall": {
        "state_error_fraction_of_cap": 0.80,
        "mistake_flag_recall": 1.00,
        "false_alarm_fraction_of_cap": 0.50,
    },
}

CONFIDENCE_LEVELS = {
    "joint_95pct": 0.95,
    "joint_99pct": 0.99,
}

DEFAULT_RISK_BUDGETS = (
    0.0,
    0.001,
    0.0025,
    0.005,
    0.01,
)

DEFAULT_FALSE_POSITIVE_RATE_CAP = 0.05


def load_e66(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_confidence_frontier"
        / "pacing_confidence_frontier_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float_csv(value: str) -> list[float]:
    return [
        float(item)
        for item in value.split(",")
        if item.strip()
    ]


def wilson_bounds(
    *,
    successes: float,
    trials: float,
    z: float,
) -> tuple[float, float]:
    if trials <= 0.0:
        return 0.0, 1.0
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (
        proportion
        + z_squared / (2.0 * trials)
    ) / denominator
    margin = (
        z
        / denominator
        * math.sqrt(
            proportion
            * (1.0 - proportion)
            / trials
            + z_squared
            / (4.0 * trials * trials)
        )
    )
    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def minimum_requests(
    *,
    predicate: Callable[[int], bool],
    maximum_requests: int,
) -> int | None:
    if predicate(1):
        return 1
    low = 1
    high = 2
    while (
        high <= maximum_requests
        and not predicate(high)
    ):
        low = high
        high *= 2
    if high > maximum_requests:
        high = maximum_requests
        if not predicate(high):
            return None
    while low + 1 < high:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle
    return high


def state_error_upper_bound(
    *,
    requests: int,
    actual_state_error: float,
    z: float,
) -> float:
    return wilson_bounds(
        successes=requests
        * actual_state_error,
        trials=float(requests),
        z=z,
    )[1]


def recall_lower_bound(
    *,
    requests: int,
    actual_state_error: float,
    actual_recall: float,
    z: float,
) -> float:
    mistake_trials = (
        requests * actual_state_error
    )
    return wilson_bounds(
        successes=mistake_trials
        * actual_recall,
        trials=mistake_trials,
        z=z,
    )[0]


def false_alarm_upper_bound(
    *,
    requests: int,
    actual_state_error: float,
    actual_false_alarm_rate: float,
    z: float,
) -> float:
    correct_trials = requests * (
        1.0 - actual_state_error
    )
    return wilson_bounds(
        successes=correct_trials
        * actual_false_alarm_rate,
        trials=correct_trials,
        z=z,
    )[1]


def certify_operating_point(
    *,
    dimensions: tuple[str, ...],
    state_error_cap: float,
    required_recall: float,
    false_alarm_cap: float,
    actual_state_error: float,
    actual_recall: float,
    actual_false_alarm_rate: float,
    joint_confidence: float,
    maximum_requests: int,
) -> dict[str, Any]:
    if not dimensions:
        return {
            "status": "structural_safe",
            "required_labeled_requests": 0,
            "required_requests_by_metric": {
                "state_error": 0,
                "mistake_flag_recall": 0,
                "false_alarm_rate": 0,
            },
            "limiting_metrics": [],
            "per_bound_confidence": 1.0,
            "one_sided_z": 0.0,
            "certified_bounds": {
                "state_error_upper": 0.0,
                "mistake_flag_recall_lower": 1.0,
                "false_alarm_rate_upper": 0.0,
            },
        }
    bound_count = 3 * len(dimensions)
    total_alpha = 1.0 - joint_confidence
    per_bound_alpha = (
        total_alpha / bound_count
    )
    z = NormalDist().inv_cdf(
        1.0 - per_bound_alpha
    )
    impossible_reasons = []
    if actual_state_error >= state_error_cap:
        impossible_reasons.append(
            "actual_state_error_not_below_cap"
        )
    if actual_recall <= required_recall:
        impossible_reasons.append(
            "actual_recall_not_above_requirement"
        )
    if (
        actual_false_alarm_rate
        >= false_alarm_cap
    ):
        impossible_reasons.append(
            "actual_false_alarm_not_below_cap"
        )
    if impossible_reasons:
        return {
            "status": "uncertifiable",
            "reason": ",".join(
                impossible_reasons
            ),
            "required_labeled_requests": None,
            "required_requests_by_metric": {
                "state_error": None,
                "mistake_flag_recall": None,
                "false_alarm_rate": None,
            },
            "limiting_metrics": [],
            "per_bound_confidence": (
                1.0 - per_bound_alpha
            ),
            "one_sided_z": z,
            "certified_bounds": None,
        }
    requirements = {
        "state_error": minimum_requests(
            predicate=lambda requests: (
                state_error_upper_bound(
                    requests=requests,
                    actual_state_error=(
                        actual_state_error
                    ),
                    z=z,
                )
                <= state_error_cap + 1e-15
            ),
            maximum_requests=maximum_requests,
        ),
        "mistake_flag_recall": (
            minimum_requests(
                predicate=lambda requests: (
                    recall_lower_bound(
                        requests=requests,
                        actual_state_error=(
                            actual_state_error
                        ),
                        actual_recall=(
                            actual_recall
                        ),
                        z=z,
                    )
                    >= required_recall
                    - 1e-15
                ),
                maximum_requests=(
                    maximum_requests
                ),
            )
        ),
        "false_alarm_rate": minimum_requests(
            predicate=lambda requests: (
                false_alarm_upper_bound(
                    requests=requests,
                    actual_state_error=(
                        actual_state_error
                    ),
                    actual_false_alarm_rate=(
                        actual_false_alarm_rate
                    ),
                    z=z,
                )
                <= false_alarm_cap + 1e-15
            ),
            maximum_requests=maximum_requests,
        ),
    }
    if any(
        value is None
        for value in requirements.values()
    ):
        return {
            "status": (
                "beyond_search_limit"
            ),
            "required_labeled_requests": None,
            "required_requests_by_metric": (
                requirements
            ),
            "limiting_metrics": [],
            "per_bound_confidence": (
                1.0 - per_bound_alpha
            ),
            "one_sided_z": z,
            "certified_bounds": None,
        }
    required_requests = max(
        int(value)
        for value in requirements.values()
        if value is not None
    )
    limiting_metrics = [
        metric
        for metric, value
        in requirements.items()
        if value == required_requests
    ]
    bounds = {
        "state_error_upper": (
            state_error_upper_bound(
                requests=required_requests,
                actual_state_error=(
                    actual_state_error
                ),
                z=z,
            )
        ),
        "mistake_flag_recall_lower": (
            recall_lower_bound(
                requests=required_requests,
                actual_state_error=(
                    actual_state_error
                ),
                actual_recall=actual_recall,
                z=z,
            )
        ),
        "false_alarm_rate_upper": (
            false_alarm_upper_bound(
                requests=required_requests,
                actual_state_error=(
                    actual_state_error
                ),
                actual_false_alarm_rate=(
                    actual_false_alarm_rate
                ),
                z=z,
            )
        ),
    }
    previous_passes = False
    if required_requests > 1:
        previous_passes = (
            state_error_upper_bound(
                requests=required_requests - 1,
                actual_state_error=(
                    actual_state_error
                ),
                z=z,
            )
            <= state_error_cap + 1e-15
            and recall_lower_bound(
                requests=required_requests - 1,
                actual_state_error=(
                    actual_state_error
                ),
                actual_recall=actual_recall,
                z=z,
            )
            >= required_recall - 1e-15
            and false_alarm_upper_bound(
                requests=required_requests - 1,
                actual_state_error=(
                    actual_state_error
                ),
                actual_false_alarm_rate=(
                    actual_false_alarm_rate
                ),
                z=z,
            )
            <= false_alarm_cap + 1e-15
        )
    if previous_passes:
        raise RuntimeError(
            "Certification request count was "
            "not minimal"
        )
    return {
        "status": "certified",
        "required_labeled_requests": (
            required_requests
        ),
        "required_requests_by_metric": (
            requirements
        ),
        "limiting_metrics": (
            limiting_metrics
        ),
        "per_bound_confidence": (
            1.0 - per_bound_alpha
        ),
        "one_sided_z": z,
        "certified_bounds": bounds,
        "expected_labeled_mistakes": (
            required_requests
            * actual_state_error
        ),
        "expected_correct_feature_labels": (
            required_requests
            * (1.0 - actual_state_error)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e66 = load_e66(args.root)
    slos_ms = parse_float_csv(
        args.ttft_slos_ms
    )
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
    risk_budgets = parse_float_csv(
        args.risk_budgets
    )
    scenarios = [
        value.strip()
        for value in args.scenarios.split(",")
        if value.strip()
    ]
    confidence_levels = [
        value.strip()
        for value
        in args.confidence_levels.split(",")
        if value.strip()
    ]
    unknown_families = set(families) - set(FAMILIES)
    if unknown_families:
        raise ValueError(
            f"Unknown families: {sorted(unknown_families)}"
        )
    unknown_error_profiles = set(
        error_profiles
    ) - set(ERROR_PROFILES)
    if unknown_error_profiles:
        raise ValueError(
            "Unknown error profiles: "
            f"{sorted(unknown_error_profiles)}"
        )
    unknown_scenarios = set(
        scenarios
    ) - set(CALIBRATION_SCENARIOS)
    if unknown_scenarios:
        raise ValueError(
            "Unknown scenarios: "
            f"{sorted(unknown_scenarios)}"
        )
    unknown_confidence_levels = set(
        confidence_levels
    ) - set(CONFIDENCE_LEVELS)
    if unknown_confidence_levels:
        raise ValueError(
            "Unknown confidence levels: "
            f"{sorted(unknown_confidence_levels)}"
        )
    false_alarm_cap = float(
        args.false_alarm_cap
    )
    fixed_index = {
        (
            float(group["ttft_slo_ms"]),
            group["family"],
            group["error_profile"],
            float(
                false_positive_slice[
                    "correct_flag_false_positive_rate"
                ]
            ),
            float(selection["risk_budget"]),
        ): selection
        for group in e66[
            "fixed_false_positive_rate_budget_selections"
        ]
        for false_positive_slice in group[
            "false_positive_rate_slices"
        ]
        for selection in false_positive_slice[
            "selections"
        ]
    }
    records = []
    selection_checks = []
    for slo_ms in slos_ms:
        for family in families:
            dimensions = FAMILIES[family]
            for error_profile in error_profiles:
                error_probabilities = (
                    ERROR_PROFILES[
                        error_profile
                    ]
                )
                keyed_errors = [
                    float(
                        error_probabilities[
                            dimension
                        ]
                    )
                    for dimension in dimensions
                ]
                state_error_cap = (
                    max(keyed_errors)
                    if keyed_errors
                    else 0.0
                )
                for risk_budget in risk_budgets:
                    selection = fixed_index[
                        (
                            slo_ms,
                            family,
                            error_profile,
                            false_alarm_cap,
                            risk_budget,
                        )
                    ]
                    if (
                        selection["status"]
                        != "feasible"
                        or selection[
                            "expected_scenario_slo_violation_fraction"
                        ]
                        > risk_budget + 1e-12
                    ):
                        raise RuntimeError(
                            "Invalid E66 selection"
                        )
                    required_recall = float(
                        selection[
                            "mistake_flag_sensitivity"
                        ]
                    )
                    selection_checks.append(
                        {
                            "ttft_slo_ms": slo_ms,
                            "family": family,
                            "error_profile": (
                                error_profile
                            ),
                            "risk_budget": (
                                risk_budget
                            ),
                            "selected_recall": (
                                required_recall
                            ),
                            "actual_violation_probability": (
                                selection[
                                    "expected_scenario_slo_violation_fraction"
                                ]
                            ),
                        }
                    )
                    for scenario_name in scenarios:
                        scenario = (
                            CALIBRATION_SCENARIOS[
                                scenario_name
                            ]
                        )
                        actual_state_error = (
                            state_error_cap
                            * scenario[
                                "state_error_fraction_of_cap"
                            ]
                        )
                        actual_recall = float(
                            scenario[
                                "mistake_flag_recall"
                            ]
                        )
                        actual_false_alarm = (
                            false_alarm_cap
                            * scenario[
                                "false_alarm_fraction_of_cap"
                            ]
                        )
                        for confidence_name in (
                            confidence_levels
                        ):
                            joint_confidence = (
                                CONFIDENCE_LEVELS[
                                    confidence_name
                                ]
                            )
                            certification = (
                                certify_operating_point(
                                    dimensions=(
                                        dimensions
                                    ),
                                    state_error_cap=(
                                        state_error_cap
                                    ),
                                    required_recall=(
                                        required_recall
                                    ),
                                    false_alarm_cap=(
                                        false_alarm_cap
                                    ),
                                    actual_state_error=(
                                        actual_state_error
                                    ),
                                    actual_recall=(
                                        actual_recall
                                    ),
                                    actual_false_alarm_rate=(
                                        actual_false_alarm
                                    ),
                                    joint_confidence=(
                                        joint_confidence
                                    ),
                                    maximum_requests=(
                                        args.maximum_requests
                                    ),
                                )
                            )
                            records.append(
                                {
                                    "ttft_slo_ms": (
                                        slo_ms
                                    ),
                                    "family": family,
                                    "dimensions": list(
                                        dimensions
                                    ),
                                    "dimension_count": (
                                        len(dimensions)
                                    ),
                                    "error_profile": (
                                        error_profile
                                    ),
                                    "state_error_cap": (
                                        state_error_cap
                                    ),
                                    "false_alarm_cap": (
                                        false_alarm_cap
                                    ),
                                    "risk_budget": (
                                        risk_budget
                                    ),
                                    "e66_required_recall": (
                                        required_recall
                                    ),
                                    "e66_mean_throughput_ratio_vs_oracle": (
                                        selection[
                                            "mean_expected_throughput_ratio_vs_oracle"
                                        ]
                                    ),
                                    "e66_actual_violation_probability": (
                                        selection[
                                            "expected_scenario_slo_violation_fraction"
                                        ]
                                    ),
                                    "calibration_scenario": (
                                        scenario_name
                                    ),
                                    "actual_state_error": (
                                        actual_state_error
                                    ),
                                    "actual_mistake_flag_recall": (
                                        actual_recall
                                    ),
                                    "actual_false_alarm_rate": (
                                        actual_false_alarm
                                    ),
                                    "confidence_profile": (
                                        confidence_name
                                    ),
                                    "joint_confidence": (
                                        joint_confidence
                                    ),
                                    **certification,
                                }
                            )
    expected_record_count = (
        len(slos_ms)
        * len(families)
        * len(error_profiles)
        * len(risk_budgets)
        * len(scenarios)
        * len(confidence_levels)
    )
    if len(records) != expected_record_count:
        raise RuntimeError(
            "Unexpected E67 record count"
        )
    index = {
        (
            record["ttft_slo_ms"],
            record["family"],
            record["error_profile"],
            record["risk_budget"],
            record["calibration_scenario"],
            record["confidence_profile"],
        ): record
        for record in records
    }
    confidence_monotonicity_checks = []
    for key in sorted(
        {
            item[:-1]
            for item in index
        }
    ):
        lower = index[
            (*key, "joint_95pct")
        ]
        higher = index[
            (*key, "joint_99pct")
        ]
        if (
            lower["status"] == "certified"
            and higher["status"]
            == "certified"
        ):
            difference = (
                higher[
                    "required_labeled_requests"
                ]
                - lower[
                    "required_labeled_requests"
                ]
            )
            if difference < 0:
                raise RuntimeError(
                    "Higher confidence required fewer labels"
                )
            confidence_monotonicity_checks.append(
                {
                    "key": list(key),
                    "additional_requests_for_99pct": (
                        difference
                    ),
                }
            )
    structural_records = [
        record
        for record in records
        if record["family"] == "universal"
    ]
    if any(
        record["status"]
        != "structural_safe"
        or record[
            "required_labeled_requests"
        ]
        != 0
        for record in structural_records
    ):
        raise RuntimeError(
            "Universal policy should be structural safe"
        )
    recall_one_nonuniversal = [
        record
        for record in records
        if record["family"] != "universal"
        and record[
            "e66_required_recall"
        ]
        >= 1.0 - 1e-15
    ]
    if any(
        record["status"] != "uncertifiable"
        for record
        in recall_one_nonuniversal
    ):
        raise RuntimeError(
            "Finite data certified exact recall one"
        )
    result = {
        "suite": (
            "pacing_confidence_calibration_"
            "certification_analysis"
        ),
        "model": (
            "Use one-sided Wilson score bounds with "
            "Bonferroni correction across every keyed "
            "feature's state-error, mistake-recall, and "
            "false-alarm constraints. Solve the minimum "
            "labeled-request count on the conservative "
            "expected-count trajectory needed to certify "
            "an E66 operating point."
        ),
        "source": {
            "confidence_frontier": (
                "results/raw/"
                "pacing_confidence_frontier/"
                "pacing_confidence_frontier_full.json"
            )
        },
        "ttft_slos_ms": slos_ms,
        "families": {
            family: {
                "dimensions": list(
                    FAMILIES[family]
                ),
                "dimension_count": len(
                    FAMILIES[family]
                ),
            }
            for family in families
        },
        "error_profiles": {
            profile: ERROR_PROFILES[
                profile
            ]
            for profile in error_profiles
        },
        "false_alarm_cap": false_alarm_cap,
        "risk_budgets": risk_budgets,
        "calibration_scenarios": {
            scenario: (
                CALIBRATION_SCENARIOS[
                    scenario
                ]
            )
            for scenario in scenarios
        },
        "confidence_levels": {
            name: CONFIDENCE_LEVELS[name]
            for name in confidence_levels
        },
        "records": records,
        "validation": {
            "e66_selection_checks": (
                selection_checks
            ),
            "confidence_monotonicity_checks": (
                confidence_monotonicity_checks
            ),
            "structural_safe_record_count": (
                len(structural_records)
            ),
            "recall_one_uncertifiable_record_count": (
                len(
                    recall_one_nonuniversal
                )
            ),
            "expected_record_count": (
                expected_record_count
            ),
            "minimality": (
                "For every certified record, "
                "required_labeled_requests - 1 "
                "fails at least one bound."
            ),
        },
        "limitations": [
            "Wilson bounds are evaluated on expected fractional counts, so the result is a nominal sample-size plan rather than an exact binomial power calculation.",
            "Bonferroni correction assumes each keyed feature is certified separately and controls family-wide confidence conservatively.",
            "The analysis assumes labeled ground truth for state mistakes and confidence flags; delayed or biased labels can invalidate the certificate.",
            "Rates are stationary within the calibration window. Correlated bursts, drift, and adaptive threshold changes require sequential or time-uniform confidence methods.",
            "Many strict zero-risk E66 points require recall one and cannot be certified from finite monitoring data; some loose-SLO compact policies are insensitive to the remaining mistakes.",
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
        "--ttft-slos-ms",
        default="650,800,1000",
    )
    parser.add_argument(
        "--families",
        default=(
            "universal,burst,load,cache,"
            "burst_load,burst_cache,"
            "load_cache,full"
        ),
    )
    parser.add_argument(
        "--error-profiles",
        default=(
            "all_5pct,all_10pct,all_20pct"
        ),
    )
    parser.add_argument(
        "--risk-budgets",
        default=",".join(
            str(value)
            for value in DEFAULT_RISK_BUDGETS
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(
            CALIBRATION_SCENARIOS
        ),
    )
    parser.add_argument(
        "--confidence-levels",
        default=",".join(
            CONFIDENCE_LEVELS
        ),
    )
    parser.add_argument(
        "--false-alarm-cap",
        type=float,
        default=(
            DEFAULT_FALSE_POSITIVE_RATE_CAP
        ),
    )
    parser.add_argument(
        "--maximum-requests",
        type=int,
        default=1_000_000_000,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
