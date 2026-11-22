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
from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
    alpha_spent_fraction,
    certification_bounds as hoeffding_certification_bounds,
    drift_bound as hoeffding_drift_bound,
    geometric_looks,
    look_alpha,
)


def load_e68(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_anytime_monitoring"
        / "pacing_anytime_monitoring_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_e69(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_checkpoint_monitoring"
        / "pacing_checkpoint_monitoring_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def binary_kl(
    empirical: float,
    candidate: float,
) -> float:
    if not (
        0.0 <= empirical <= 1.0
        and 0.0 <= candidate <= 1.0
    ):
        raise ValueError(
            "Bernoulli probability outside [0, 1]"
        )
    if empirical == candidate:
        return 0.0
    first = (
        0.0
        if empirical == 0.0
        else (
            math.inf
            if candidate == 0.0
            else empirical
            * math.log(
                empirical / candidate
            )
        )
    )
    complement = 1.0 - empirical
    candidate_complement = (
        1.0 - candidate
    )
    second = (
        0.0
        if complement == 0.0
        else (
            math.inf
            if candidate_complement
            == 0.0
            else complement
            * math.log(
                complement
                / candidate_complement
            )
        )
    )
    return first + second


def kl_upper_bound(
    *,
    empirical: float,
    trials: int,
    alpha: float,
) -> float:
    if trials <= 0:
        return 1.0
    if empirical >= 1.0:
        return 1.0
    target = math.log(1.0 / alpha) / trials
    low = empirical
    high = 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if (
            binary_kl(
                empirical,
                middle,
            )
            <= target
        ):
            low = middle
        else:
            high = middle
    return low


def kl_lower_bound(
    *,
    empirical: float,
    trials: int,
    alpha: float,
) -> float:
    if trials <= 0:
        return 0.0
    if empirical <= 0.0:
        return 0.0
    target = math.log(1.0 / alpha) / trials
    low = 0.0
    high = empirical
    for _ in range(80):
        middle = (low + high) / 2.0
        if (
            binary_kl(
                empirical,
                middle,
            )
            <= target
        ):
            high = middle
        else:
            low = middle
    return high


def certification_bounds(
    *,
    requests: int,
    look_index: int,
    actual_state_error: float,
    actual_recall: float,
    actual_false_alarm_rate: float,
    stream_alpha: float,
) -> dict[str, Any]:
    alpha = look_alpha(
        stream_alpha=stream_alpha,
        look_index=look_index,
    )
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
    return {
        "state_error_upper": (
            kl_upper_bound(
                empirical=actual_state_error,
                trials=requests,
                alpha=alpha,
            )
        ),
        "mistake_flag_recall_lower": (
            kl_lower_bound(
                empirical=actual_recall,
                trials=recall_trials,
                alpha=alpha,
            )
        ),
        "false_alarm_rate_upper": (
            kl_upper_bound(
                empirical=actual_false_alarm_rate,
                trials=false_alarm_trials,
                alpha=alpha,
            )
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
            "Selected checkpoint did not pass"
        )
    while low + 1 < high:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle
    return high


def certify_kl(
    *,
    baseline: dict[str, Any],
    hoeffding: dict[str, Any],
    schedule: str,
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
        "schedule": schedule,
        "geometric_ratio": ratio,
        "fixed_wilson_status": (
            baseline["fixed_wilson_status"]
        ),
        "fixed_wilson_required_labeled_requests": (
            baseline[
                "fixed_wilson_required_labeled_requests"
            ]
        ),
        "every_request_anytime_required_labeled_requests": (
            baseline[
                "required_labeled_requests"
            ]
        ),
        "checkpoint_hoeffding_status": (
            hoeffding["status"]
        ),
        "checkpoint_hoeffding_required_labeled_requests": (
            hoeffding[
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
            "previous_checkpoint_requests": 0,
            "checkpoint_gap_requests": 0,
            "within_look_quantization_delay_requests": 0,
            "alpha_spent_fraction": 0.0,
            "ratio_vs_fixed_wilson": None,
            "ratio_vs_every_request_anytime": None,
            "ratio_vs_checkpoint_hoeffding": None,
            "certified_bounds": (
                baseline["certified_bounds"]
            ),
            "hoeffding_containment_margins": None,
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
            "within_look_quantization_delay_requests": None,
            "alpha_spent_fraction": (
                alpha_spent_fraction(
                    len(looks)
                )
            ),
            "ratio_vs_fixed_wilson": None,
            "ratio_vs_every_request_anytime": None,
            "ratio_vs_checkpoint_hoeffding": None,
            "certified_bounds": None,
            "hoeffding_containment_margins": None,
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
                    actual_recall=actual_recall,
                    actual_false_alarm_rate=(
                        actual_false_alarm_rate
                    ),
                    stream_alpha=stream_alpha,
                ),
                state_error_cap=state_error_cap,
                required_recall=required_recall,
                false_alarm_cap=false_alarm_cap,
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
                "Previous KL checkpoint "
                "already certified"
            )
    hoeffding_bounds = (
        hoeffding_certification_bounds(
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
    )
    containment = {
        "state_error_upper_reduction": (
            hoeffding_bounds[
                "state_error_upper"
            ]
            - bounds["state_error_upper"]
        ),
        "mistake_recall_lower_increase": (
            bounds[
                "mistake_flag_recall_lower"
            ]
            - hoeffding_bounds[
                "mistake_flag_recall_lower"
            ]
        ),
        "false_alarm_upper_reduction": (
            hoeffding_bounds[
                "false_alarm_rate_upper"
            ]
            - bounds[
                "false_alarm_rate_upper"
            ]
        ),
    }
    if min(containment.values()) < -1e-12:
        raise RuntimeError(
            "KL interval was wider than "
            "Hoeffding interval"
        )
    hoeffding_requests = hoeffding[
        "required_labeled_requests"
    ]
    if (
        request_count
        > hoeffding_requests
    ):
        raise RuntimeError(
            "KL checkpoint certified later "
            "than Hoeffding"
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
            request_count - previous_request
        ),
        "within_look_threshold_requests": (
            threshold_request
        ),
        "within_look_quantization_delay_requests": (
            request_count - threshold_request
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
            request_count / fixed_requests
        ),
        "ratio_vs_every_request_anytime": (
            request_count / every_request
        ),
        "ratio_vs_checkpoint_hoeffding": (
            request_count
            / hoeffding_requests
        ),
        "request_savings_vs_checkpoint_hoeffding": (
            1.0
            - request_count
            / hoeffding_requests
        ),
        "certified_bounds": bounds,
        "hoeffding_containment_margins": (
            containment
        ),
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
    alpha = look_alpha(
        stream_alpha=stream_alpha,
        look_index=look_index,
    )
    bound = (
        kl_lower_bound(
            empirical=actual_rate,
            trials=trials,
            alpha=alpha,
        )
        if metric in (
            "state_error",
            "false_alarm_rate",
        )
        else kl_upper_bound(
            empirical=actual_rate,
            trials=trials,
            alpha=alpha,
        )
    )
    return bound, trials


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


def detect_kl(
    *,
    baseline: dict[str, Any],
    hoeffding: dict[str, Any],
    schedule: str,
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
        "schedule": schedule,
        "geometric_ratio": ratio,
        "every_request_anytime_status": (
            baseline["status"]
        ),
        "every_request_anytime_required_post_change_requests": (
            baseline[
                "required_post_change_requests"
            ]
        ),
        "checkpoint_hoeffding_status": (
            hoeffding["status"]
        ),
        "checkpoint_hoeffding_required_post_change_requests": (
            hoeffding[
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
            "previous_checkpoint_requests": 0,
            "checkpoint_gap_requests": 0,
            "within_look_quantization_delay_requests": 0,
            "alpha_spent_fraction": 0.0,
            "ratio_vs_every_request_anytime": None,
            "ratio_vs_checkpoint_hoeffding": None,
            "detection_bound": (
                baseline["detection_bound"]
            ),
            "effective_trials": baseline.get(
                "effective_trials"
            ),
            "hoeffding_containment_margin": None,
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
            "required_post_change_requests": None,
            "look_index": None,
            "previous_checkpoint_requests": (
                looks[-1]
            ),
            "checkpoint_gap_requests": None,
            "within_look_quantization_delay_requests": None,
            "alpha_spent_fraction": (
                alpha_spent_fraction(
                    len(looks)
                )
            ),
            "ratio_vs_every_request_anytime": None,
            "ratio_vs_checkpoint_hoeffding": None,
            "detection_bound": None,
            "effective_trials": None,
            "hoeffding_containment_margin": None,
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
                "Previous KL checkpoint "
                "already detected drift"
            )
    hoeffding_bound, _ = (
        hoeffding_drift_bound(
            requests=request_count,
            look_index=look_index,
            metric=metric,
            actual_rate=actual_rate,
            conditioning_probability=(
                conditioning_probability
            ),
            stream_alpha=stream_alpha,
        )
    )
    containment = (
        bound - hoeffding_bound
        if metric in (
            "state_error",
            "false_alarm_rate",
        )
        else hoeffding_bound - bound
    )
    if containment < -1e-12:
        raise RuntimeError(
            "KL drift bound was wider "
            "than Hoeffding"
        )
    hoeffding_requests = hoeffding[
        "required_post_change_requests"
    ]
    if request_count > hoeffding_requests:
        raise RuntimeError(
            "KL checkpoint detected later "
            "than Hoeffding"
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
            request_count - previous_request
        ),
        "within_look_threshold_requests": (
            threshold_request
        ),
        "within_look_quantization_delay_requests": (
            request_count - threshold_request
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
            request_count / every_request
        ),
        "ratio_vs_checkpoint_hoeffding": (
            request_count
            / hoeffding_requests
        ),
        "request_savings_vs_checkpoint_hoeffding": (
            1.0
            - request_count
            / hoeffding_requests
        ),
        "detection_bound": bound,
        "effective_trials": trials,
        "hoeffding_containment_margin": (
            containment
        ),
    }


def summarize(
    values: list[float],
) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(
            values
        ),
        "p90": ordered[
            int(
                0.9 * (len(ordered) - 1)
            )
        ],
        "maximum": max(values),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e68 = load_e68(args.root)
    e69 = load_e69(args.root)
    schedule_names = [
        value.strip()
        for value in args.schedules.split(",")
        if value.strip()
    ]
    unknown = set(
        schedule_names
    ) - set(LOOK_SCHEDULES)
    if unknown:
        raise ValueError(
            f"Unknown schedules: {sorted(unknown)}"
        )
    schedules = {
        name: LOOK_SCHEDULES[name]
        for name in schedule_names
    }
    look_tables = {
        name: geometric_looks(
            ratio=ratio,
            maximum_requests=(
                args.maximum_requests
            ),
        )
        for name, ratio in schedules.items()
    }
    e69_certification = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            float(record["risk_budget"]),
            record["calibration_scenario"],
            record["confidence_profile"],
            record["schedule"],
        ): record
        for record in e69[
            "certification_records"
        ]
    }
    e69_drift = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            float(record["risk_budget"]),
            record["confidence_profile"],
            record["breach_metric"],
            record["schedule"],
        ): record
        for record in e69[
            "drift_detection_records"
        ]
    }
    certification_records = []
    for baseline in e68[
        "certification_records"
    ]:
        base_key = (
            float(
                baseline["ttft_slo_ms"]
            ),
            baseline["family"],
            baseline["error_profile"],
            float(
                baseline["risk_budget"]
            ),
            baseline[
                "calibration_scenario"
            ],
            baseline[
                "confidence_profile"
            ],
        )
        for schedule, ratio in (
            schedules.items()
        ):
            certification_records.append(
                certify_kl(
                    baseline=baseline,
                    hoeffding=e69_certification[
                        (*base_key, schedule)
                    ],
                    schedule=schedule,
                    ratio=ratio,
                    looks=look_tables[
                        schedule
                    ],
                )
            )
    drift_records = []
    for baseline in e68[
        "drift_detection_records"
    ]:
        base_key = (
            float(
                baseline["ttft_slo_ms"]
            ),
            baseline["family"],
            baseline["error_profile"],
            float(
                baseline["risk_budget"]
            ),
            baseline[
                "confidence_profile"
            ],
            baseline["breach_metric"],
        )
        for schedule, ratio in (
            schedules.items()
        ):
            drift_records.append(
                detect_kl(
                    baseline=baseline,
                    hoeffding=e69_drift[
                        (*base_key, schedule)
                    ],
                    schedule=schedule,
                    ratio=ratio,
                    looks=look_tables[
                        schedule
                    ],
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
            "Unexpected E70 record count"
        )
    certification_summaries = {}
    drift_summaries = {}
    for schedule in schedules:
        cert = [
            record
            for record
            in certification_records
            if (
                record["schedule"] == schedule
                and record["status"]
                == "certified"
            )
        ]
        drift = [
            record
            for record in drift_records
            if (
                record["schedule"] == schedule
                and record["status"]
                == "detected"
            )
        ]
        certification_summaries[
            schedule
        ] = {
            "finite_record_count": len(cert),
            "ratio_vs_fixed_wilson": summarize(
                [
                    float(
                        record[
                            "ratio_vs_fixed_wilson"
                        ]
                    )
                    for record in cert
                ]
            ),
            "ratio_vs_every_request_anytime": summarize(
                [
                    float(
                        record[
                            "ratio_vs_every_request_anytime"
                        ]
                    )
                    for record in cert
                ]
            ),
            "ratio_vs_checkpoint_hoeffding": summarize(
                [
                    float(
                        record[
                            "ratio_vs_checkpoint_hoeffding"
                        ]
                    )
                    for record in cert
                ]
            ),
            "quantization_delay_requests": summarize(
                [
                    float(
                        record[
                            "within_look_quantization_delay_requests"
                        ]
                    )
                    for record in cert
                ]
            ),
        }
        drift_summaries[schedule] = {
            "finite_record_count": len(
                drift
            ),
            "ratio_vs_every_request_anytime": summarize(
                [
                    float(
                        record[
                            "ratio_vs_every_request_anytime"
                        ]
                    )
                    for record in drift
                ]
            ),
            "ratio_vs_checkpoint_hoeffding": summarize(
                [
                    float(
                        record[
                            "ratio_vs_checkpoint_hoeffding"
                        ]
                    )
                    for record in drift
                ]
            ),
            "quantization_delay_requests": summarize(
                [
                    float(
                        record[
                            "within_look_quantization_delay_requests"
                        ]
                    )
                    for record in drift
                ]
            ),
        }
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
    containment_checks = {
        "certification_count": len(
            finite_certification
        ),
        "drift_count": len(finite_drift),
        "minimum_certification_margin": min(
            min(
                record[
                    "hoeffding_containment_margins"
                ].values()
            )
            for record
            in finite_certification
        ),
        "minimum_drift_margin": min(
            float(
                record[
                    "hoeffding_containment_margin"
                ]
            )
            for record in finite_drift
        ),
        "maximum_certification_ratio_vs_hoeffding": max(
            float(
                record[
                    "ratio_vs_checkpoint_hoeffding"
                ]
            )
            for record
            in finite_certification
        ),
        "maximum_drift_ratio_vs_hoeffding": max(
            float(
                record[
                    "ratio_vs_checkpoint_hoeffding"
                ]
            )
            for record in finite_drift
        ),
    }
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
        schedule: 0
        for schedule in schedules
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
        schedule: 0
        for schedule in schedules
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
            "pacing_kl_checkpoint_"
            "monitoring_analysis"
        ),
        "model": (
            "Replace E69 fixed-look "
            "Hoeffding intervals with one-sided "
            "Bernoulli-KL Chernoff intervals at "
            "the same preregistered geometric "
            "checkpoints and alpha allocation."
        ),
        "source": {
            "every_request_anytime": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            ),
            "checkpoint_hoeffding": (
                "results/raw/"
                "pacing_checkpoint_monitoring/"
                "pacing_checkpoint_monitoring_full.json"
            ),
        },
        "schedules": {
            schedule: e69[
                "schedules"
            ][schedule]
            for schedule in schedules
        },
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
            "containment_checks": (
                containment_checks
            ),
            "minimality": (
                "The previous scheduled checkpoint "
                "fails, and one fewer request than "
                "the selected look's same-alpha "
                "threshold also fails."
            ),
            "validity": (
                "At a fixed look, Bernoulli "
                "Chernoff tails bound a one-sided "
                "failure by the allocated alpha. "
                "The summable look-index schedule "
                "and Bonferroni stream split "
                "provide joint scheduled-look "
                "validity."
            ),
        },
        "limitations": [
            "The binary-KL Chernoff interval is still conservative relative to exact binomial inversion or optimized mixture confidence sequences.",
            "Validity holds only at the declared checkpoints; any intermediate peek or action requires additional alpha allocation.",
            "The analysis uses deterministic expected proportions and integer effective trial counts rather than stochastic stopping-time simulation.",
            "Recall trials remain sparse because only state mistakes carry recall labels; tighter intervals cannot remove that information bottleneck.",
            "Correlated traffic and delayed labels require martingale-safe or block-aware extensions.",
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
