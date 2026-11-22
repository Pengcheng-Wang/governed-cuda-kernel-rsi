from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from analyze_pacing_policy_compression import FAMILIES


DEFAULT_SCENARIOS = (
    "balanced_margin",
    "very_high_recall",
    "near_error_boundary",
    "near_false_alarm_boundary",
)

DRIFT_DELTAS = {
    "state_error_absolute_minimum": 0.02,
    "state_error_relative": 0.25,
    "mistake_recall_absolute": 0.025,
    "false_alarm_absolute": 0.025,
}


def load_e67(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_calibration_certification"
        / "pacing_calibration_certification_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float_csv(value: str) -> list[float]:
    return [
        float(item)
        for item in value.split(",")
        if item.strip()
    ]


def anytime_radius(
    *,
    trials: int,
    stream_alpha: float,
) -> float:
    if trials <= 0:
        return 1.0
    spending_alpha = (
        stream_alpha
        * 6.0
        / (
            math.pi
            * math.pi
            * trials
            * trials
        )
    )
    return math.sqrt(
        math.log(1.0 / spending_alpha)
        / (2.0 * trials)
    )


def anytime_bounds(
    *,
    proportion: float,
    trials: int,
    stream_alpha: float,
) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    radius = anytime_radius(
        trials=trials,
        stream_alpha=stream_alpha,
    )
    return (
        max(0.0, proportion - radius),
        min(1.0, proportion + radius),
    )


def effective_trials(
    *,
    requests: int,
    probability: float,
) -> int:
    return max(
        0,
        math.floor(
            requests * probability
            + 1e-12
        ),
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


def stream_alpha(
    *,
    dimension_count: int,
    joint_confidence: float,
) -> float:
    return (
        (1.0 - joint_confidence)
        / (3.0 * dimension_count)
    )


def certification_bounds(
    *,
    requests: int,
    actual_state_error: float,
    actual_recall: float,
    actual_false_alarm_rate: float,
    per_stream_alpha: float,
) -> dict[str, Any]:
    recall_trials = effective_trials(
        requests=requests,
        probability=actual_state_error,
    )
    false_alarm_trials = effective_trials(
        requests=requests,
        probability=(
            1.0 - actual_state_error
        ),
    )
    state_bounds = anytime_bounds(
        proportion=actual_state_error,
        trials=requests,
        stream_alpha=per_stream_alpha,
    )
    recall_bounds = anytime_bounds(
        proportion=actual_recall,
        trials=recall_trials,
        stream_alpha=per_stream_alpha,
    )
    false_alarm_bounds = anytime_bounds(
        proportion=actual_false_alarm_rate,
        trials=false_alarm_trials,
        stream_alpha=per_stream_alpha,
    )
    return {
        "state_error_lower": (
            state_bounds[0]
        ),
        "state_error_upper": (
            state_bounds[1]
        ),
        "mistake_flag_recall_lower": (
            recall_bounds[0]
        ),
        "mistake_flag_recall_upper": (
            recall_bounds[1]
        ),
        "false_alarm_rate_lower": (
            false_alarm_bounds[0]
        ),
        "false_alarm_rate_upper": (
            false_alarm_bounds[1]
        ),
        "state_error_trials": requests,
        "mistake_flag_recall_trials": (
            recall_trials
        ),
        "false_alarm_rate_trials": (
            false_alarm_trials
        ),
    }


def certify_anytime(
    *,
    dimension_count: int,
    state_error_cap: float,
    required_recall: float,
    false_alarm_cap: float,
    actual_state_error: float,
    actual_recall: float,
    actual_false_alarm_rate: float,
    joint_confidence: float,
    maximum_requests: int,
) -> dict[str, Any]:
    if dimension_count == 0:
        return {
            "status": "structural_safe",
            "required_labeled_requests": 0,
            "required_requests_by_metric": {
                "state_error": 0,
                "mistake_flag_recall": 0,
                "false_alarm_rate": 0,
            },
            "limiting_metrics": [],
            "stream_alpha": 1.0,
            "certified_bounds": {
                "state_error_upper": 0.0,
                "mistake_flag_recall_lower": 1.0,
                "false_alarm_rate_upper": 0.0,
            },
        }
    alpha = stream_alpha(
        dimension_count=dimension_count,
        joint_confidence=joint_confidence,
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
    if actual_false_alarm_rate >= false_alarm_cap:
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
            "stream_alpha": alpha,
            "certified_bounds": None,
        }

    def metric_passes(
        requests: int,
        metric: str,
    ) -> bool:
        bounds = certification_bounds(
            requests=requests,
            actual_state_error=(
                actual_state_error
            ),
            actual_recall=actual_recall,
            actual_false_alarm_rate=(
                actual_false_alarm_rate
            ),
            per_stream_alpha=alpha,
        )
        if metric == "state_error":
            return (
                bounds["state_error_upper"]
                <= state_error_cap + 1e-15
            )
        if metric == "mistake_flag_recall":
            return (
                bounds[
                    "mistake_flag_recall_lower"
                ]
                >= required_recall - 1e-15
            )
        if metric == "false_alarm_rate":
            return (
                bounds[
                    "false_alarm_rate_upper"
                ]
                <= false_alarm_cap + 1e-15
            )
        raise ValueError(metric)

    requirements = {
        metric: minimum_requests(
            predicate=lambda requests, metric=metric: (
                metric_passes(
                    requests,
                    metric,
                )
            ),
            maximum_requests=maximum_requests,
        )
        for metric in (
            "state_error",
            "mistake_flag_recall",
            "false_alarm_rate",
        )
    }
    if any(
        value is None
        for value in requirements.values()
    ):
        return {
            "status": "beyond_search_limit",
            "required_labeled_requests": None,
            "required_requests_by_metric": (
                requirements
            ),
            "limiting_metrics": [],
            "stream_alpha": alpha,
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
    bounds = certification_bounds(
        requests=required_requests,
        actual_state_error=actual_state_error,
        actual_recall=actual_recall,
        actual_false_alarm_rate=(
            actual_false_alarm_rate
        ),
        per_stream_alpha=alpha,
    )
    if required_requests > 1:
        previous_bounds = certification_bounds(
            requests=required_requests - 1,
            actual_state_error=(
                actual_state_error
            ),
            actual_recall=actual_recall,
            actual_false_alarm_rate=(
                actual_false_alarm_rate
            ),
            per_stream_alpha=alpha,
        )
        previous_passes = (
            previous_bounds[
                "state_error_upper"
            ]
            <= state_error_cap + 1e-15
            and previous_bounds[
                "mistake_flag_recall_lower"
            ]
            >= required_recall - 1e-15
            and previous_bounds[
                "false_alarm_rate_upper"
            ]
            <= false_alarm_cap + 1e-15
        )
        if previous_passes:
            raise RuntimeError(
                "Anytime certification was "
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
        "stream_alpha": alpha,
        "certified_bounds": bounds,
    }


def detect_breach(
    *,
    dimension_count: int,
    metric: str,
    cap: float,
    actual_rate: float,
    conditioning_probability: float,
    joint_confidence: float,
    maximum_requests: int,
) -> dict[str, Any]:
    if dimension_count == 0:
        return {
            "status": "not_applicable",
            "required_post_change_requests": 0,
            "stream_alpha": 1.0,
            "detection_bound": None,
        }
    alpha = stream_alpha(
        dimension_count=dimension_count,
        joint_confidence=joint_confidence,
    )
    if metric in (
        "state_error",
        "false_alarm_rate",
    ):
        if actual_rate <= cap:
            return {
                "status": "no_breach",
                "required_post_change_requests": None,
                "stream_alpha": alpha,
                "detection_bound": None,
            }
    elif metric == "mistake_flag_recall":
        if actual_rate >= cap:
            return {
                "status": "no_breach",
                "required_post_change_requests": None,
                "stream_alpha": alpha,
                "detection_bound": None,
            }
    else:
        raise ValueError(metric)

    def bound_for(
        requests: int,
    ) -> tuple[float, int]:
        trials = (
            requests
            if metric == "state_error"
            else effective_trials(
                requests=requests,
                probability=(
                    conditioning_probability
                ),
            )
        )
        lower, upper = anytime_bounds(
            proportion=actual_rate,
            trials=trials,
            stream_alpha=alpha,
        )
        return (
            (
                lower
                if metric
                in (
                    "state_error",
                    "false_alarm_rate",
                )
                else upper
            ),
            trials,
        )

    def detected(requests: int) -> bool:
        bound, _ = bound_for(requests)
        return (
            bound > cap + 1e-15
            if metric
            in (
                "state_error",
                "false_alarm_rate",
            )
            else bound < cap - 1e-15
        )

    required = minimum_requests(
        predicate=detected,
        maximum_requests=maximum_requests,
    )
    if required is None:
        return {
            "status": "beyond_search_limit",
            "required_post_change_requests": None,
            "stream_alpha": alpha,
            "detection_bound": None,
        }
    bound, trials = bound_for(required)
    if required > 1 and detected(required - 1):
        raise RuntimeError(
            "Drift detection was not minimal"
        )
    return {
        "status": "detected",
        "required_post_change_requests": (
            required
        ),
        "stream_alpha": alpha,
        "detection_bound": bound,
        "effective_trials": trials,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e67 = load_e67(args.root)
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
    confidence_profiles = [
        value.strip()
        for value
        in args.confidence_profiles.split(",")
        if value.strip()
    ]
    e67_index = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            float(record["risk_budget"]),
            record["calibration_scenario"],
            record["confidence_profile"],
        ): record
        for record in e67["records"]
    }
    certification_records = []
    fixed_baseline_checks = []
    for slo_ms in slos_ms:
        for family in families:
            dimension_count = len(
                FAMILIES[family]
            )
            for error_profile in error_profiles:
                for risk_budget in risk_budgets:
                    for scenario in scenarios:
                        for confidence_profile in (
                            confidence_profiles
                        ):
                            fixed = e67_index[
                                (
                                    slo_ms,
                                    family,
                                    error_profile,
                                    risk_budget,
                                    scenario,
                                    confidence_profile,
                                )
                            ]
                            anytime = certify_anytime(
                                dimension_count=(
                                    dimension_count
                                ),
                                state_error_cap=float(
                                    fixed[
                                        "state_error_cap"
                                    ]
                                ),
                                required_recall=float(
                                    fixed[
                                        "e66_required_recall"
                                    ]
                                ),
                                false_alarm_cap=float(
                                    fixed[
                                        "false_alarm_cap"
                                    ]
                                ),
                                actual_state_error=float(
                                    fixed[
                                        "actual_state_error"
                                    ]
                                ),
                                actual_recall=float(
                                    fixed[
                                        "actual_mistake_flag_recall"
                                    ]
                                ),
                                actual_false_alarm_rate=float(
                                    fixed[
                                        "actual_false_alarm_rate"
                                    ]
                                ),
                                joint_confidence=float(
                                    fixed[
                                        "joint_confidence"
                                    ]
                                ),
                                maximum_requests=(
                                    args.maximum_requests
                                ),
                            )
                            fixed_requests = fixed[
                                "required_labeled_requests"
                            ]
                            anytime_requests = anytime[
                                "required_labeled_requests"
                            ]
                            overhead_ratio = (
                                anytime_requests
                                / fixed_requests
                                if (
                                    fixed_requests
                                    not in (None, 0)
                                    and anytime_requests
                                    is not None
                                )
                                else None
                            )
                            certification_records.append(
                                {
                                    "ttft_slo_ms": (
                                        slo_ms
                                    ),
                                    "family": family,
                                    "dimensions": list(
                                        FAMILIES[
                                            family
                                        ]
                                    ),
                                    "dimension_count": (
                                        dimension_count
                                    ),
                                    "error_profile": (
                                        error_profile
                                    ),
                                    "risk_budget": (
                                        risk_budget
                                    ),
                                    "calibration_scenario": (
                                        scenario
                                    ),
                                    "confidence_profile": (
                                        confidence_profile
                                    ),
                                    "joint_confidence": (
                                        fixed[
                                            "joint_confidence"
                                        ]
                                    ),
                                    "state_error_cap": (
                                        fixed[
                                            "state_error_cap"
                                        ]
                                    ),
                                    "required_recall": (
                                        fixed[
                                            "e66_required_recall"
                                        ]
                                    ),
                                    "false_alarm_cap": (
                                        fixed[
                                            "false_alarm_cap"
                                        ]
                                    ),
                                    "actual_state_error": (
                                        fixed[
                                            "actual_state_error"
                                        ]
                                    ),
                                    "actual_mistake_flag_recall": (
                                        fixed[
                                            "actual_mistake_flag_recall"
                                        ]
                                    ),
                                    "actual_false_alarm_rate": (
                                        fixed[
                                            "actual_false_alarm_rate"
                                        ]
                                    ),
                                    "fixed_wilson_status": (
                                        fixed["status"]
                                    ),
                                    "fixed_wilson_required_labeled_requests": (
                                        fixed_requests
                                    ),
                                    "anytime_overhead_ratio_vs_fixed": (
                                        overhead_ratio
                                    ),
                                    **anytime,
                                }
                            )
                            fixed_baseline_checks.append(
                                {
                                    "key": [
                                        slo_ms,
                                        family,
                                        error_profile,
                                        risk_budget,
                                        scenario,
                                        confidence_profile,
                                    ],
                                    "fixed_status": (
                                        fixed["status"]
                                    ),
                                    "fixed_required_labeled_requests": (
                                        fixed_requests
                                    ),
                                }
                            )

    drift_records = []
    for slo_ms in slos_ms:
        for family in families:
            dimensions = FAMILIES[family]
            dimension_count = len(dimensions)
            for error_profile in error_profiles:
                for risk_budget in risk_budgets:
                    for confidence_profile in (
                        confidence_profiles
                    ):
                        reference = e67_index[
                            (
                                slo_ms,
                                family,
                                error_profile,
                                risk_budget,
                                "balanced_margin",
                                confidence_profile,
                            )
                        ]
                        state_error_cap = float(
                            reference[
                                "state_error_cap"
                            ]
                        )
                        required_recall = float(
                            reference[
                                "e66_required_recall"
                            ]
                        )
                        false_alarm_cap = float(
                            reference[
                                "false_alarm_cap"
                            ]
                        )
                        joint_confidence = float(
                            reference[
                                "joint_confidence"
                            ]
                        )
                        nominal_error = (
                            0.8 * state_error_cap
                        )
                        breaches = {
                            "state_error": {
                                "cap": state_error_cap,
                                "actual_rate": min(
                                    1.0,
                                    state_error_cap
                                    + max(
                                        DRIFT_DELTAS[
                                            "state_error_absolute_minimum"
                                        ],
                                        DRIFT_DELTAS[
                                            "state_error_relative"
                                        ]
                                        * state_error_cap,
                                    ),
                                ),
                                "conditioning_probability": (
                                    1.0
                                ),
                            },
                            "mistake_flag_recall": {
                                "cap": required_recall,
                                "actual_rate": max(
                                    0.0,
                                    required_recall
                                    - DRIFT_DELTAS[
                                        "mistake_recall_absolute"
                                    ],
                                ),
                                "conditioning_probability": (
                                    nominal_error
                                ),
                            },
                            "false_alarm_rate": {
                                "cap": false_alarm_cap,
                                "actual_rate": min(
                                    1.0,
                                    false_alarm_cap
                                    + DRIFT_DELTAS[
                                        "false_alarm_absolute"
                                    ],
                                ),
                                "conditioning_probability": (
                                    1.0
                                    - nominal_error
                                ),
                            },
                        }
                        for metric, values in (
                            breaches.items()
                        ):
                            detection = detect_breach(
                                dimension_count=(
                                    dimension_count
                                ),
                                metric=metric,
                                cap=float(
                                    values["cap"]
                                ),
                                actual_rate=float(
                                    values[
                                        "actual_rate"
                                    ]
                                ),
                                conditioning_probability=float(
                                    values[
                                        "conditioning_probability"
                                    ]
                                ),
                                joint_confidence=(
                                    joint_confidence
                                ),
                                maximum_requests=(
                                    args.maximum_requests
                                ),
                            )
                            drift_records.append(
                                {
                                    "ttft_slo_ms": (
                                        slo_ms
                                    ),
                                    "family": family,
                                    "dimensions": list(
                                        dimensions
                                    ),
                                    "dimension_count": (
                                        dimension_count
                                    ),
                                    "error_profile": (
                                        error_profile
                                    ),
                                    "risk_budget": (
                                        risk_budget
                                    ),
                                    "confidence_profile": (
                                        confidence_profile
                                    ),
                                    "joint_confidence": (
                                        joint_confidence
                                    ),
                                    "breach_metric": (
                                        metric
                                    ),
                                    "safe_cap": (
                                        values["cap"]
                                    ),
                                    "post_change_actual_rate": (
                                        values[
                                            "actual_rate"
                                        ]
                                    ),
                                    "conditioning_probability": (
                                        values[
                                            "conditioning_probability"
                                        ]
                                    ),
                                    **detection,
                                }
                            )

    expected_certification_count = (
        len(slos_ms)
        * len(families)
        * len(error_profiles)
        * len(risk_budgets)
        * len(scenarios)
        * len(confidence_profiles)
    )
    expected_drift_count = (
        len(slos_ms)
        * len(families)
        * len(error_profiles)
        * len(risk_budgets)
        * len(confidence_profiles)
        * 3
    )
    if (
        len(certification_records)
        != expected_certification_count
        or len(drift_records)
        != expected_drift_count
    ):
        raise RuntimeError(
            "Unexpected E68 record count"
        )
    certification_index = {
        (
            record["ttft_slo_ms"],
            record["family"],
            record["error_profile"],
            record["risk_budget"],
            record["calibration_scenario"],
            record["confidence_profile"],
        ): record
        for record in certification_records
    }
    confidence_monotonicity_checks = []
    for key in sorted(
        {
            item[:-1]
            for item in certification_index
        }
    ):
        lower = certification_index[
            (*key, "joint_95pct")
        ]
        higher = certification_index[
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
                    "Higher confidence required "
                    "fewer anytime labels"
                )
            confidence_monotonicity_checks.append(
                {
                    "key": list(key),
                    "additional_requests_for_99pct": (
                        difference
                    ),
                }
            )
    fixed_match_count = sum(
        1
        for record in certification_records
        if (
            record["fixed_wilson_status"]
            == record["status"]
            or (
                record[
                    "fixed_wilson_status"
                ]
                == "certified"
                and record["status"]
                == "beyond_search_limit"
            )
        )
    )
    finite_overhead_ratios = [
        float(
            record[
                "anytime_overhead_ratio_vs_fixed"
            ]
        )
        for record in certification_records
        if record[
            "anytime_overhead_ratio_vs_fixed"
        ]
        is not None
    ]
    if any(
        ratio < 1.0 - 1e-12
        for ratio in finite_overhead_ratios
    ):
        raise RuntimeError(
            "Anytime certificate beat fixed "
            "one-shot baseline unexpectedly"
        )
    sorted_overhead = sorted(
        finite_overhead_ratios
    )
    drift_index = {
        (
            record["ttft_slo_ms"],
            record["family"],
            record["error_profile"],
            record["risk_budget"],
            record["breach_metric"],
            record["confidence_profile"],
        ): record
        for record in drift_records
    }
    drift_confidence_monotonicity_checks = []
    for key in sorted(
        {
            item[:-1]
            for item in drift_index
        }
    ):
        lower = drift_index[
            (*key, "joint_95pct")
        ]
        higher = drift_index[
            (*key, "joint_99pct")
        ]
        if (
            lower["status"] == "detected"
            and higher["status"]
            == "detected"
        ):
            difference = (
                higher[
                    "required_post_change_requests"
                ]
                - lower[
                    "required_post_change_requests"
                ]
            )
            if difference < 0:
                raise RuntimeError(
                    "Higher confidence detected "
                    "drift sooner"
                )
            drift_confidence_monotonicity_checks.append(
                {
                    "key": list(key),
                    "additional_requests_for_99pct": (
                        difference
                    ),
                }
            )
    result = {
        "suite": (
            "pacing_anytime_monitoring_"
            "analysis"
        ),
        "model": (
            "Replace E67's one-shot Wilson "
            "certificate with an anytime-valid "
            "Hoeffding confidence sequence. Each "
            "metric-feature stream receives a "
            "Bonferroni share of joint alpha and "
            "spends it over trial count m as "
            "6 alpha / (pi^2 m^2)."
        ),
        "source": {
            "fixed_calibration": (
                "results/raw/"
                "pacing_calibration_certification/"
                "pacing_calibration_certification_full.json"
            )
        },
        "scenarios": scenarios,
        "confidence_profiles": (
            confidence_profiles
        ),
        "drift_deltas": DRIFT_DELTAS,
        "certification_records": (
            certification_records
        ),
        "drift_detection_records": (
            drift_records
        ),
        "repeated_look_risk_examples": [
            {
                "nominal_single_look_confidence": (
                    confidence
                ),
                "looks": looks,
                "independence_false_alarm_probability": (
                    1.0
                    - confidence**looks
                ),
                "union_bound_false_alarm_probability": min(
                    1.0,
                    looks
                    * (1.0 - confidence),
                ),
            }
            for confidence in (
                0.95,
                0.99,
            )
            for looks in (
                24,
                168,
                720,
            )
        ],
        "validation": {
            "fixed_wilson_baseline_checks": (
                fixed_baseline_checks
            ),
            "fixed_status_match_count": (
                fixed_match_count
            ),
            "confidence_monotonicity_checks": (
                confidence_monotonicity_checks
            ),
            "drift_confidence_monotonicity_checks": (
                drift_confidence_monotonicity_checks
            ),
            "finite_anytime_overhead_ratio_summary": {
                "minimum": min(
                    finite_overhead_ratios
                ),
                "median": statistics.median(
                    sorted_overhead
                ),
                "p90": sorted_overhead[
                    int(
                        0.9
                        * (
                            len(
                                sorted_overhead
                            )
                            - 1
                        )
                    )
                ],
                "maximum": max(
                    finite_overhead_ratios
                ),
                "count": len(
                    finite_overhead_ratios
                ),
            },
            "expected_certification_record_count": (
                expected_certification_count
            ),
            "expected_drift_record_count": (
                expected_drift_count
            ),
            "minimality": (
                "For every finite anytime "
                "certification or drift detection, "
                "one fewer labeled request fails."
            ),
        },
        "limitations": [
            "The confidence sequence uses a transparent union-bound Hoeffding construction rather than a tighter mixture or empirical-Bernstein boundary.",
            "Sample sizes follow deterministic expected proportions and integer effective trial counts, not stochastic stopping-time simulations.",
            "Drift detection starts a fresh post-change audit; mixing pre-change data would delay invalidation and requires a change-point or rolling-window design.",
            "The analysis assumes independent Bernoulli labels within each stream. Correlated traffic and delayed labels require martingale-safe or block-aware methods.",
            "Time-uniform validity protects repeated peeking but can cost substantially more labels than E67's one-shot Wilson certificate.",
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
        default="0,0.001,0.0025,0.005,0.01",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(
            DEFAULT_SCENARIOS
        ),
    )
    parser.add_argument(
        "--confidence-profiles",
        default="joint_95pct,joint_99pct",
    )
    parser.add_argument(
        "--maximum-requests",
        type=int,
        default=1_000_000_000,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
