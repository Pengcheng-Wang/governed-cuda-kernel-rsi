from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from analyze_pacing_anytime_monitoring import (
    effective_trials,
)


LOOK_SCHEDULES = {
    "geometric_1p10": 1.10,
    "geometric_1p25": 1.25,
    "geometric_1p50": 1.50,
    "geometric_2p00": 2.00,
}


def load_e68(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_anytime_monitoring"
        / "pacing_anytime_monitoring_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def geometric_looks(
    *,
    ratio: float,
    maximum_requests: int,
) -> list[int]:
    if ratio <= 1.0:
        raise ValueError(
            "Geometric ratio must exceed one"
        )
    looks = [1]
    while looks[-1] < maximum_requests:
        next_request = max(
            looks[-1] + 1,
            math.ceil(
                looks[-1] * ratio
                - 1e-12
            ),
        )
        looks.append(
            min(
                next_request,
                maximum_requests,
            )
        )
        if looks[-1] == maximum_requests:
            break
    return looks


def look_alpha(
    *,
    stream_alpha: float,
    look_index: int,
) -> float:
    if look_index <= 0:
        raise ValueError(
            "Look index must be positive"
        )
    return (
        stream_alpha
        * 6.0
        / (
            math.pi
            * math.pi
            * look_index
            * look_index
        )
    )


def look_radius(
    *,
    trials: int,
    stream_alpha: float,
    look_index: int,
) -> float:
    if trials <= 0:
        return 1.0
    alpha = look_alpha(
        stream_alpha=stream_alpha,
        look_index=look_index,
    )
    return math.sqrt(
        math.log(1.0 / alpha)
        / (2.0 * trials)
    )


def look_bounds(
    *,
    proportion: float,
    trials: int,
    stream_alpha: float,
    look_index: int,
) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    radius = look_radius(
        trials=trials,
        stream_alpha=stream_alpha,
        look_index=look_index,
    )
    return (
        max(0.0, proportion - radius),
        min(1.0, proportion + radius),
    )


def certification_bounds(
    *,
    requests: int,
    look_index: int,
    actual_state_error: float,
    actual_recall: float,
    actual_false_alarm_rate: float,
    stream_alpha: float,
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
    state_bounds = look_bounds(
        proportion=actual_state_error,
        trials=requests,
        stream_alpha=stream_alpha,
        look_index=look_index,
    )
    recall_bounds = look_bounds(
        proportion=actual_recall,
        trials=recall_trials,
        stream_alpha=stream_alpha,
        look_index=look_index,
    )
    false_alarm_bounds = look_bounds(
        proportion=actual_false_alarm_rate,
        trials=false_alarm_trials,
        stream_alpha=stream_alpha,
        look_index=look_index,
    )
    return {
        "state_error_upper": state_bounds[1],
        "mistake_flag_recall_lower": (
            recall_bounds[0]
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


def certification_passes(
    *,
    bounds: dict[str, Any],
    state_error_cap: float,
    required_recall: float,
    false_alarm_cap: float,
) -> bool:
    return (
        bounds["state_error_upper"]
        <= state_error_cap + 1e-15
        and bounds[
            "mistake_flag_recall_lower"
        ]
        >= required_recall - 1e-15
        and bounds[
            "false_alarm_rate_upper"
        ]
        <= false_alarm_cap + 1e-15
    )


def minimum_requests(
    *,
    low: int,
    high: int,
    predicate: Callable[[int], bool],
) -> int:
    if predicate(low):
        return low
    if not predicate(high):
        raise RuntimeError(
            "Scheduled endpoint did not pass"
        )
    while low + 1 < high:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle
    return high


def alpha_spent_fraction(
    look_count: int,
) -> float:
    return (
        6.0
        / (math.pi * math.pi)
        * sum(
            1.0 / (index * index)
            for index in range(
                1,
                look_count + 1,
            )
        )
    )


def evaluate_certification_schedule(
    *,
    baseline: dict[str, Any],
    schedule_name: str,
    ratio: float,
    looks: list[int],
) -> dict[str, Any]:
    common = {
        "ttft_slo_ms": (
            baseline["ttft_slo_ms"]
        ),
        "family": baseline["family"],
        "error_profile": (
            baseline["error_profile"]
        ),
        "risk_budget": (
            baseline["risk_budget"]
        ),
        "calibration_scenario": (
            baseline["calibration_scenario"]
        ),
        "confidence_profile": (
            baseline["confidence_profile"]
        ),
        "schedule": schedule_name,
        "geometric_ratio": ratio,
        "fixed_wilson_status": (
            baseline["fixed_wilson_status"]
        ),
        "fixed_wilson_required_labeled_requests": (
            baseline[
                "fixed_wilson_required_labeled_requests"
            ]
        ),
        "every_request_anytime_status": (
            baseline["status"]
        ),
        "every_request_anytime_required_labeled_requests": (
            baseline[
                "required_labeled_requests"
            ]
        ),
    }
    if baseline["status"] in (
        "structural_safe",
        "uncertifiable",
        "beyond_search_limit",
    ):
        return {
            **common,
            "status": baseline["status"],
            "required_labeled_requests": (
                baseline[
                    "required_labeled_requests"
                ]
            ),
            "look_index": 0,
            "previous_checkpoint_requests": (
                0
            ),
            "checkpoint_gap_requests": 0,
            "within_look_quantization_delay_requests": (
                0
            ),
            "alpha_spent_fraction": 0.0,
            "ratio_vs_fixed_wilson": None,
            "ratio_vs_every_request_anytime": (
                None
            ),
            "request_savings_vs_every_request_anytime": (
                None
            ),
            "certified_bounds": (
                baseline[
                    "certified_bounds"
                ]
            ),
        }
    stream_alpha = float(
        baseline["stream_alpha"]
    )
    state_error_cap = float(
        baseline["state_error_cap"]
    )
    required_recall = float(
        baseline["required_recall"]
    )
    false_alarm_cap = float(
        baseline["false_alarm_cap"]
    )
    actual_state_error = float(
        baseline["actual_state_error"]
    )
    actual_recall = float(
        baseline[
            "actual_mistake_flag_recall"
        ]
    )
    actual_false_alarm_rate = float(
        baseline["actual_false_alarm_rate"]
    )
    selected = None
    previous_request = 0
    for look_index, request_count in enumerate(
        looks,
        start=1,
    ):
        bounds = certification_bounds(
            requests=request_count,
            look_index=look_index,
            actual_state_error=(
                actual_state_error
            ),
            actual_recall=actual_recall,
            actual_false_alarm_rate=(
                actual_false_alarm_rate
            ),
            stream_alpha=stream_alpha,
        )
        if certification_passes(
            bounds=bounds,
            state_error_cap=state_error_cap,
            required_recall=required_recall,
            false_alarm_cap=false_alarm_cap,
        ):
            selected = (
                look_index,
                request_count,
                previous_request,
                bounds,
            )
            break
        previous_request = request_count
    if selected is None:
        return {
            **common,
            "status": "beyond_search_limit",
            "required_labeled_requests": None,
            "look_index": None,
            "previous_checkpoint_requests": (
                looks[-1]
            ),
            "checkpoint_gap_requests": None,
            "within_look_quantization_delay_requests": (
                None
            ),
            "alpha_spent_fraction": (
                alpha_spent_fraction(
                    len(looks)
                )
            ),
            "ratio_vs_fixed_wilson": None,
            "ratio_vs_every_request_anytime": (
                None
            ),
            "request_savings_vs_every_request_anytime": (
                None
            ),
            "certified_bounds": None,
        }
    (
        look_index,
        request_count,
        previous_request,
        bounds,
    ) = selected
    threshold_request = minimum_requests(
        low=previous_request,
        high=request_count,
        predicate=lambda requests: (
            certification_passes(
                bounds=certification_bounds(
                    requests=requests,
                    look_index=look_index,
                    actual_state_error=(
                        actual_state_error
                    ),
                    actual_recall=(
                        actual_recall
                    ),
                    actual_false_alarm_rate=(
                        actual_false_alarm_rate
                    ),
                    stream_alpha=(
                        stream_alpha
                    ),
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
            )
        ),
    )
    if previous_request > 0:
        previous_bounds = certification_bounds(
            requests=previous_request,
            look_index=look_index - 1,
            actual_state_error=(
                actual_state_error
            ),
            actual_recall=actual_recall,
            actual_false_alarm_rate=(
                actual_false_alarm_rate
            ),
            stream_alpha=stream_alpha,
        )
        if certification_passes(
            bounds=previous_bounds,
            state_error_cap=state_error_cap,
            required_recall=required_recall,
            false_alarm_cap=false_alarm_cap,
        ):
            raise RuntimeError(
                "Previous checkpoint already "
                "certified"
            )
    fixed_requests = baseline[
        "fixed_wilson_required_labeled_requests"
    ]
    every_request = baseline[
        "required_labeled_requests"
    ]
    return {
        **common,
        "status": "certified",
        "required_labeled_requests": (
            request_count
        ),
        "look_index": look_index,
        "previous_checkpoint_requests": (
            previous_request
        ),
        "checkpoint_gap_requests": (
            request_count
            - previous_request
        ),
        "within_look_threshold_requests": (
            threshold_request
        ),
        "within_look_quantization_delay_requests": (
            request_count
            - threshold_request
        ),
        "alpha_spent_fraction": (
            alpha_spent_fraction(
                look_index
            )
        ),
        "look_alpha": look_alpha(
            stream_alpha=stream_alpha,
            look_index=look_index,
        ),
        "ratio_vs_fixed_wilson": (
            request_count
            / fixed_requests
        ),
        "ratio_vs_every_request_anytime": (
            request_count
            / every_request
        ),
        "request_savings_vs_every_request_anytime": (
            1.0
            - request_count
            / every_request
        ),
        "certified_bounds": bounds,
    }


def drift_bound(
    *,
    requests: int,
    look_index: int,
    metric: str,
    actual_rate: float,
    conditioning_probability: float,
    stream_alpha: float,
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
    lower, upper = look_bounds(
        proportion=actual_rate,
        trials=trials,
        stream_alpha=stream_alpha,
        look_index=look_index,
    )
    return (
        (
            lower
            if metric in (
                "state_error",
                "false_alarm_rate",
            )
            else upper
        ),
        trials,
    )


def drift_detected(
    *,
    bound: float,
    metric: str,
    cap: float,
) -> bool:
    return (
        bound > cap + 1e-15
        if metric in (
            "state_error",
            "false_alarm_rate",
        )
        else bound < cap - 1e-15
    )


def evaluate_drift_schedule(
    *,
    baseline: dict[str, Any],
    schedule_name: str,
    ratio: float,
    looks: list[int],
) -> dict[str, Any]:
    common = {
        "ttft_slo_ms": (
            baseline["ttft_slo_ms"]
        ),
        "family": baseline["family"],
        "error_profile": (
            baseline["error_profile"]
        ),
        "risk_budget": (
            baseline["risk_budget"]
        ),
        "confidence_profile": (
            baseline["confidence_profile"]
        ),
        "breach_metric": (
            baseline["breach_metric"]
        ),
        "schedule": schedule_name,
        "geometric_ratio": ratio,
        "every_request_anytime_status": (
            baseline["status"]
        ),
        "every_request_anytime_required_post_change_requests": (
            baseline[
                "required_post_change_requests"
            ]
        ),
    }
    if baseline["status"] in (
        "not_applicable",
        "no_breach",
        "beyond_search_limit",
    ):
        return {
            **common,
            "status": baseline["status"],
            "required_post_change_requests": (
                baseline[
                    "required_post_change_requests"
                ]
            ),
            "look_index": 0,
            "previous_checkpoint_requests": (
                0
            ),
            "checkpoint_gap_requests": 0,
            "within_look_quantization_delay_requests": (
                0
            ),
            "alpha_spent_fraction": 0.0,
            "ratio_vs_every_request_anytime": (
                None
            ),
            "request_savings_vs_every_request_anytime": (
                None
            ),
            "detection_bound": (
                baseline["detection_bound"]
            ),
            "effective_trials": baseline.get(
                "effective_trials"
            ),
        }
    stream_alpha = float(
        baseline["stream_alpha"]
    )
    metric = baseline["breach_metric"]
    cap = float(baseline["safe_cap"])
    actual_rate = float(
        baseline["post_change_actual_rate"]
    )
    conditioning_probability = float(
        baseline[
            "conditioning_probability"
        ]
    )
    selected = None
    previous_request = 0
    for look_index, request_count in enumerate(
        looks,
        start=1,
    ):
        bound, trials = drift_bound(
            requests=request_count,
            look_index=look_index,
            metric=metric,
            actual_rate=actual_rate,
            conditioning_probability=(
                conditioning_probability
            ),
            stream_alpha=stream_alpha,
        )
        if drift_detected(
            bound=bound,
            metric=metric,
            cap=cap,
        ):
            selected = (
                look_index,
                request_count,
                previous_request,
                bound,
                trials,
            )
            break
        previous_request = request_count
    if selected is None:
        return {
            **common,
            "status": "beyond_search_limit",
            "required_post_change_requests": (
                None
            ),
            "look_index": None,
            "previous_checkpoint_requests": (
                looks[-1]
            ),
            "checkpoint_gap_requests": None,
            "within_look_quantization_delay_requests": (
                None
            ),
            "alpha_spent_fraction": (
                alpha_spent_fraction(
                    len(looks)
                )
            ),
            "ratio_vs_every_request_anytime": (
                None
            ),
            "request_savings_vs_every_request_anytime": (
                None
            ),
            "detection_bound": None,
            "effective_trials": None,
        }
    (
        look_index,
        request_count,
        previous_request,
        bound,
        trials,
    ) = selected
    threshold_request = minimum_requests(
        low=previous_request,
        high=request_count,
        predicate=lambda requests: (
            drift_detected(
                bound=drift_bound(
                    requests=requests,
                    look_index=look_index,
                    metric=metric,
                    actual_rate=actual_rate,
                    conditioning_probability=(
                        conditioning_probability
                    ),
                    stream_alpha=(
                        stream_alpha
                    ),
                )[0],
                metric=metric,
                cap=cap,
            )
        ),
    )
    if previous_request > 0:
        previous_bound, _ = drift_bound(
            requests=previous_request,
            look_index=look_index - 1,
            metric=metric,
            actual_rate=actual_rate,
            conditioning_probability=(
                conditioning_probability
            ),
            stream_alpha=stream_alpha,
        )
        if drift_detected(
            bound=previous_bound,
            metric=metric,
            cap=cap,
        ):
            raise RuntimeError(
                "Previous checkpoint already "
                "detected drift"
            )
    every_request = baseline[
        "required_post_change_requests"
    ]
    return {
        **common,
        "status": "detected",
        "required_post_change_requests": (
            request_count
        ),
        "look_index": look_index,
        "previous_checkpoint_requests": (
            previous_request
        ),
        "checkpoint_gap_requests": (
            request_count
            - previous_request
        ),
        "within_look_threshold_requests": (
            threshold_request
        ),
        "within_look_quantization_delay_requests": (
            request_count
            - threshold_request
        ),
        "alpha_spent_fraction": (
            alpha_spent_fraction(
                look_index
            )
        ),
        "look_alpha": look_alpha(
            stream_alpha=stream_alpha,
            look_index=look_index,
        ),
        "ratio_vs_every_request_anytime": (
            request_count
            / every_request
        ),
        "request_savings_vs_every_request_anytime": (
            1.0
            - request_count
            / every_request
        ),
        "detection_bound": bound,
        "effective_trials": trials,
    }


def summarize_ratios(
    values: list[float],
) -> dict[str, Any]:
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(
            values
        ),
        "p90": sorted_values[
            int(
                0.9
                * (len(sorted_values) - 1)
            )
        ],
        "maximum": max(values),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e68 = load_e68(args.root)
    schedules = {
        name: ratio
        for name, ratio
        in LOOK_SCHEDULES.items()
        if name in {
            value.strip()
            for value
            in args.schedules.split(",")
            if value.strip()
        }
    }
    unknown = {
        value.strip()
        for value
        in args.schedules.split(",")
        if value.strip()
    } - set(LOOK_SCHEDULES)
    if unknown:
        raise ValueError(
            f"Unknown schedules: {sorted(unknown)}"
        )
    look_tables = {
        name: geometric_looks(
            ratio=ratio,
            maximum_requests=(
                args.maximum_requests
            ),
        )
        for name, ratio in schedules.items()
    }
    certification_records = []
    for baseline in e68[
        "certification_records"
    ]:
        for name, ratio in (
            schedules.items()
        ):
            certification_records.append(
                evaluate_certification_schedule(
                    baseline=baseline,
                    schedule_name=name,
                    ratio=ratio,
                    looks=look_tables[name],
                )
            )
    drift_records = []
    for baseline in e68[
        "drift_detection_records"
    ]:
        for name, ratio in (
            schedules.items()
        ):
            drift_records.append(
                evaluate_drift_schedule(
                    baseline=baseline,
                    schedule_name=name,
                    ratio=ratio,
                    looks=look_tables[name],
                )
            )
    expected_certification_count = (
        len(
            e68["certification_records"]
        )
        * len(schedules)
    )
    expected_drift_count = (
        len(
            e68["drift_detection_records"]
        )
        * len(schedules)
    )
    if (
        len(certification_records)
        != expected_certification_count
        or len(drift_records)
        != expected_drift_count
    ):
        raise RuntimeError(
            "Unexpected E69 record count"
        )
    finite_certification = [
        record
        for record in certification_records
        if record["status"] == "certified"
    ]
    finite_drift = [
        record
        for record in drift_records
        if record["status"] == "detected"
    ]
    for record in (
        finite_certification
        + finite_drift
    ):
        if (
            record[
                "ratio_vs_every_request_anytime"
            ]
            > 1.0 + 1e-12
        ):
            raise RuntimeError(
                "Checkpoint monitoring used more "
                "requests than every-request "
                "monitoring"
            )
        if (
            record[
                "within_look_quantization_delay_requests"
            ]
            < 0
            or record[
                "within_look_quantization_delay_requests"
            ]
            >= record[
                "checkpoint_gap_requests"
            ]
        ):
            raise RuntimeError(
                "Invalid checkpoint "
                "quantization delay"
            )
        if (
            record[
                "alpha_spent_fraction"
            ]
            > 1.0 + 1e-12
        ):
            raise RuntimeError(
                "Finite alpha spend exceeded "
                "stream budget"
            )
    certification_summaries = {}
    drift_summaries = {}
    for name in schedules:
        certification_subset = [
            record
            for record
            in certification_records
            if (
                record["schedule"] == name
                and record["status"]
                == "certified"
            )
        ]
        drift_subset = [
            record
            for record in drift_records
            if (
                record["schedule"] == name
                and record["status"]
                == "detected"
            )
        ]
        certification_summaries[name] = {
            "finite_record_count": len(
                certification_subset
            ),
            "ratio_vs_fixed_wilson": (
                summarize_ratios(
                    [
                        float(
                            record[
                                "ratio_vs_fixed_wilson"
                            ]
                        )
                        for record
                        in certification_subset
                    ]
                )
            ),
            "ratio_vs_every_request_anytime": (
                summarize_ratios(
                    [
                        float(
                            record[
                                "ratio_vs_every_request_anytime"
                            ]
                        )
                        for record
                        in certification_subset
                    ]
                )
            ),
            "quantization_delay_requests": (
                summarize_ratios(
                    [
                        float(
                            record[
                                "within_look_quantization_delay_requests"
                            ]
                        )
                        for record
                        in certification_subset
                    ]
                )
            ),
        }
        drift_summaries[name] = {
            "finite_record_count": len(
                drift_subset
            ),
            "ratio_vs_every_request_anytime": (
                summarize_ratios(
                    [
                        float(
                            record[
                                "ratio_vs_every_request_anytime"
                            ]
                        )
                        for record
                        in drift_subset
                    ]
                )
            ),
            "quantization_delay_requests": (
                summarize_ratios(
                    [
                        float(
                            record[
                                "within_look_quantization_delay_requests"
                            ]
                        )
                        for record
                        in drift_subset
                    ]
                )
            ),
        }
    schedule_validation = {
        name: {
            "ratio": schedules[name],
            "look_count_to_search_limit": len(
                look_tables[name]
            ),
            "last_look_requests": (
                look_tables[name][-1]
            ),
            "infinite_alpha_budget_fraction": (
                1.0
            ),
            "finite_alpha_spent_fraction": (
                alpha_spent_fraction(
                    len(look_tables[name])
                )
            ),
        }
        for name in schedules
    }
    for values in (
        schedule_validation.values()
    ):
        if (
            values[
                "finite_alpha_spent_fraction"
            ]
            > 1.0 + 1e-12
        ):
            raise RuntimeError(
                "Checkpoint alpha budget exceeded"
            )
    certification_groups: dict[
        tuple[Any, ...],
        list[dict[str, Any]],
    ] = {}
    for record in finite_certification:
        key = (
            record["ttft_slo_ms"],
            record["family"],
            record["error_profile"],
            record["risk_budget"],
            record[
                "calibration_scenario"
            ],
            record[
                "confidence_profile"
            ],
        )
        certification_groups.setdefault(
            key,
            [],
        ).append(record)
    drift_groups: dict[
        tuple[Any, ...],
        list[dict[str, Any]],
    ] = {}
    for record in finite_drift:
        key = (
            record["ttft_slo_ms"],
            record["family"],
            record["error_profile"],
            record["risk_budget"],
            record[
                "confidence_profile"
            ],
            record["breach_metric"],
        )
        drift_groups.setdefault(
            key,
            [],
        ).append(record)
    certification_winner_counts = {
        name: 0
        for name in schedules
    }
    for records in (
        certification_groups.values()
    ):
        minimum = min(
            record[
                "required_labeled_requests"
            ]
            for record in records
        )
        for record in records:
            if (
                record[
                    "required_labeled_requests"
                ]
                == minimum
            ):
                certification_winner_counts[
                    record["schedule"]
                ] += 1
    drift_winner_counts = {
        name: 0
        for name in schedules
    }
    for records in drift_groups.values():
        minimum = min(
            record[
                "required_post_change_requests"
            ]
            for record in records
        )
        for record in records:
            if (
                record[
                    "required_post_change_requests"
                ]
                == minimum
            ):
                drift_winner_counts[
                    record["schedule"]
                ] += 1
    result = {
        "suite": (
            "pacing_checkpoint_monitoring_"
            "analysis"
        ),
        "model": (
            "Monitor only at geometric request "
            "checkpoints. Spend stream alpha over "
            "look index k as 6 alpha / (pi^2 k^2), "
            "then use fixed-look Hoeffding bounds. "
            "Validity holds at scheduled checkpoints; "
            "the controller must not peek or act "
            "between them."
        ),
        "source": {
            "every_request_anytime": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            )
        },
        "schedules": schedule_validation,
        "certification_records": (
            certification_records
        ),
        "drift_detection_records": (
            drift_records
        ),
        "certification_summaries": (
            certification_summaries
        ),
        "drift_summaries": (
            drift_summaries
        ),
        "schedule_winner_counts": {
            "certification": (
                certification_winner_counts
            ),
            "drift_detection": (
                drift_winner_counts
            ),
        },
        "validation": {
            "expected_certification_record_count": (
                expected_certification_count
            ),
            "expected_drift_record_count": (
                expected_drift_count
            ),
            "minimality": (
                "The previous scheduled checkpoint "
                "fails every finite certification or "
                "drift detection. Within the selected "
                "look's alpha, one fewer request than "
                "the reported threshold also fails."
            ),
            "alpha_budget": (
                "For each metric-feature stream, "
                "the infinite sum of checkpoint "
                "alphas equals stream alpha."
            ),
        },
        "limitations": [
            "Validity holds only at the declared checkpoints. Peeking or acting between checkpoints invalidates the stated alpha budget.",
            "Geometric request schedules are not aligned to effective mistake-label counts, so recall monitoring can have irregular information growth.",
            "The same conservative Hoeffding fixed-look bounds are used as E68; tighter binomial, mixture, or empirical-Bernstein boundaries remain future work.",
            "The analysis follows deterministic expected proportions rather than simulating random checkpoint crossing times.",
            "Coarser schedules reduce statistical overhead but increase action latency and can leave the safe lane active longer after a certificate becomes feasible.",
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
        "--schedules",
        default=",".join(
            LOOK_SCHEDULES
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
