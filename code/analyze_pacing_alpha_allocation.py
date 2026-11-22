from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
    alpha_spent_fraction,
    geometric_looks,
)
from analyze_pacing_kl_monitoring import (
    binary_kl,
    drift_bound,
    drift_detected,
)


METRICS = (
    "state_error",
    "mistake_flag_recall",
    "false_alarm_rate",
)

ALLOCATION_UNITS = 20
MINIMUM_UNITS = 1


def load_e68(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_anytime_monitoring"
        / "pacing_anytime_monitoring_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_e70(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_kl_monitoring"
        / "pacing_kl_monitoring_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def allocation_grid() -> list[dict[str, Any]]:
    allocations = [
        {
            "allocation": "equal",
            "weights": {
                metric: 1.0 / 3.0
                for metric in METRICS
            },
            "distance_from_equal": 0.0,
            "grid_units": None,
        }
    ]
    for state_units in range(
        MINIMUM_UNITS,
        ALLOCATION_UNITS,
    ):
        for recall_units in range(
            MINIMUM_UNITS,
            ALLOCATION_UNITS
            - state_units,
        ):
            false_alarm_units = (
                ALLOCATION_UNITS
                - state_units
                - recall_units
            )
            if (
                false_alarm_units
                < MINIMUM_UNITS
            ):
                continue
            units = {
                "state_error": state_units,
                "mistake_flag_recall": (
                    recall_units
                ),
                "false_alarm_rate": (
                    false_alarm_units
                ),
            }
            weights = {
                metric: (
                    value
                    / ALLOCATION_UNITS
                )
                for metric, value
                in units.items()
            }
            allocations.append(
                {
                    "allocation": (
                        f"u{state_units}_"
                        f"{recall_units}_"
                        f"{false_alarm_units}"
                    ),
                    "weights": weights,
                    "distance_from_equal": sum(
                        abs(
                            weight - 1.0 / 3.0
                        )
                        for weight
                        in weights.values()
                    ),
                    "grid_units": units,
                }
            )
    return allocations


def look_alpha(
    *,
    total_alpha: float,
    metric_weight: float,
    dimension_count: int,
    look_index: int,
) -> float:
    stream_alpha = (
        total_alpha
        * metric_weight
        / dimension_count
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


def metric_specification(
    *,
    record: dict[str, Any],
    metric: str,
    drift: bool,
) -> tuple[float, float, float]:
    if not drift:
        if metric == "state_error":
            return (
                float(
                    record[
                        "actual_state_error"
                    ]
                ),
                float(
                    record[
                        "state_error_cap"
                    ]
                ),
                1.0,
            )
        if metric == "mistake_flag_recall":
            return (
                float(
                    record[
                        "actual_mistake_flag_recall"
                    ]
                ),
                float(
                    record[
                        "required_recall"
                    ]
                ),
                float(
                    record[
                        "actual_state_error"
                    ]
                ),
            )
        if metric == "false_alarm_rate":
            return (
                float(
                    record[
                        "actual_false_alarm_rate"
                    ]
                ),
                float(
                    record[
                        "false_alarm_cap"
                    ]
                ),
                1.0
                - float(
                    record[
                        "actual_state_error"
                    ]
                ),
            )
    else:
        if metric != record[
            "breach_metric"
        ]:
            raise ValueError(
                "Drift metric mismatch"
            )
        return (
            float(
                record[
                    "post_change_actual_rate"
                ]
            ),
            float(record["safe_cap"]),
            float(
                record[
                    "conditioning_probability"
                ]
            ),
        )
    raise ValueError(metric)


def effective_trials(
    *,
    requests: int,
    conditioning_probability: float,
) -> int:
    return max(
        0,
        math.floor(
            requests
            * conditioning_probability
            + 1e-12
        ),
    )


def metric_passes(
    *,
    requests: int,
    look_index: int,
    empirical: float,
    threshold: float,
    conditioning_probability: float,
    total_alpha: float,
    metric_weight: float,
    dimension_count: int,
) -> bool:
    trials = (
        requests
        if conditioning_probability == 1.0
        else effective_trials(
            requests=requests,
            conditioning_probability=(
                conditioning_probability
            ),
        )
    )
    if trials <= 0:
        return False
    alpha = look_alpha(
        total_alpha=total_alpha,
        metric_weight=metric_weight,
        dimension_count=dimension_count,
        look_index=look_index,
    )
    return (
        trials
        * binary_kl(
            empirical,
            threshold,
        )
        >= math.log(1.0 / alpha)
        - 1e-12
    )


def first_look_index(
    *,
    looks: list[int],
    empirical: float,
    threshold: float,
    conditioning_probability: float,
    total_alpha: float,
    metric_weight: float,
    dimension_count: int,
) -> int | None:
    if empirical == threshold:
        return None
    for look_index, requests in enumerate(
        looks,
        start=1,
    ):
        if metric_passes(
            requests=requests,
            look_index=look_index,
            empirical=empirical,
            threshold=threshold,
            conditioning_probability=(
                conditioning_probability
            ),
            total_alpha=total_alpha,
            metric_weight=metric_weight,
            dimension_count=dimension_count,
        ):
            return look_index
    return None


def weighted_drift_crossing(
    *,
    record: dict[str, Any],
    looks: list[int],
    metric_weight: float,
) -> dict[str, Any] | None:
    metric = record["breach_metric"]
    actual_rate = float(
        record["post_change_actual_rate"]
    )
    conditioning_probability = float(
        record["conditioning_probability"]
    )
    cap = float(record["safe_cap"])
    stream_alpha = (
        (
            1.0
            - float(
                record["joint_confidence"]
            )
        )
        * metric_weight
        / int(record["dimension_count"])
    )
    previous_request = 0
    for look_index, requests in enumerate(
        looks,
        start=1,
    ):
        bound, trials = drift_bound(
            requests=requests,
            look_index=look_index,
            metric=metric,
            actual_rate=actual_rate,
            conditioning_probability=(
                conditioning_probability
            ),
            stream_alpha=stream_alpha,
        )
        if not drift_detected(
            bound=bound,
            metric=metric,
            cap=cap,
        ):
            previous_request = requests
            continue
        threshold_request = minimum_requests(
            low=previous_request,
            high=requests,
            predicate=lambda request_count: drift_detected(
                bound=drift_bound(
                    requests=request_count,
                    look_index=look_index,
                    metric=metric,
                    actual_rate=actual_rate,
                    conditioning_probability=(
                        conditioning_probability
                    ),
                    stream_alpha=stream_alpha,
                )[0],
                metric=metric,
                cap=cap,
            ),
        )
        return {
            "look_index": look_index,
            "required_requests": requests,
            "within_look_threshold_requests": (
                threshold_request
            ),
            "within_look_quantization_delay_requests": (
                requests
                - threshold_request
            ),
            "detection_bound": bound,
            "effective_trials": trials,
            "stream_alpha": stream_alpha,
        }
    return None


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
            "Selected look did not pass"
        )
    while low + 1 < high:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle
    return high


def precompute_metric_looks(
    *,
    record: dict[str, Any],
    looks: list[int],
    allocations: list[dict[str, Any]],
) -> dict[
    str,
    dict[float, int | None],
]:
    dimension_count = int(
        record["dimension_count"]
    )
    total_alpha = (
        1.0
        - float(
            record["joint_confidence"]
        )
    )
    unique_weights = sorted(
        {
            float(
                allocation["weights"][
                    metric
                ]
            )
            for allocation in allocations
            for metric in METRICS
        }
    )
    table = {}
    for metric in METRICS:
        empirical, threshold, conditioning = (
            metric_specification(
                record=record,
                metric=metric,
                drift=False,
            )
        )
        table[metric] = {
            weight: first_look_index(
                looks=looks,
                empirical=empirical,
                threshold=threshold,
                conditioning_probability=(
                    conditioning
                ),
                total_alpha=total_alpha,
                metric_weight=weight,
                dimension_count=(
                    dimension_count
                ),
            )
            for weight in unique_weights
        }
    return table


def evaluate_allocation(
    *,
    allocation: dict[str, Any],
    metric_looks: dict[
        str,
        dict[float, int | None],
    ],
    looks: list[int],
) -> dict[str, Any] | None:
    indices = {
        metric: metric_looks[metric][
            float(
                allocation[
                    "weights"
                ][metric]
            )
        ]
        for metric in METRICS
    }
    if any(
        value is None
        for value in indices.values()
    ):
        return None
    selected_index = max(
        int(value)
        for value in indices.values()
        if value is not None
    )
    return {
        "allocation": (
            allocation["allocation"]
        ),
        "weights": allocation["weights"],
        "distance_from_equal": (
            allocation[
                "distance_from_equal"
            ]
        ),
        "grid_units": (
            allocation["grid_units"]
        ),
        "metric_look_indices": indices,
        "look_index": selected_index,
        "required_requests": (
            looks[selected_index - 1]
        ),
        "limiting_metrics": [
            metric
            for metric, value
            in indices.items()
            if value == selected_index
        ],
    }


def select_best_allocation(
    *,
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    return min(
        evaluations,
        key=lambda item: (
            item["required_requests"],
            item["distance_from_equal"],
            sum(
                int(value)
                for value
                in item[
                    "metric_look_indices"
                ].values()
            ),
            item["allocation"],
        ),
    )


def certification_record(
    *,
    baseline: dict[str, Any],
    equal_record: dict[str, Any],
    schedule: str,
    looks: list[int],
    allocations: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
]:
    if (
        baseline["status"]
        != equal_record["status"]
    ):
        raise RuntimeError(
            "E68/E70 status mismatch"
        )
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
        "dimension_count": (
            baseline["dimension_count"]
        ),
        "schedule": schedule,
        "equal_kl_status": (
            equal_record["status"]
        ),
        "equal_kl_required_labeled_requests": (
            equal_record[
                "required_labeled_requests"
            ]
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
    }
    if baseline["status"] in (
        "structural_safe",
        "uncertifiable",
        "beyond_search_limit",
    ):
        return (
            {
                **common,
                "status": baseline["status"],
                "selected_allocation": (
                    "equal"
                ),
                "selected_weights": {
                    metric: 1.0 / 3.0
                    for metric in METRICS
                },
                "required_labeled_requests": (
                    baseline[
                        "required_labeled_requests"
                    ]
                ),
                "look_index": 0,
                "limiting_metrics": [],
                "ratio_vs_equal_kl": None,
                "ratio_vs_fixed_wilson": None,
                "ratio_vs_every_request_anytime": (
                    None
                ),
                "within_look_quantization_delay_requests": (
                    0
                ),
                "alpha_spent_fraction": 0.0,
            },
            None,
        )
    metric_looks = precompute_metric_looks(
        record=baseline,
        looks=looks,
        allocations=allocations,
    )
    evaluations = [
        evaluated
        for allocation in allocations
        if (
            evaluated
            := evaluate_allocation(
                allocation=allocation,
                metric_looks=metric_looks,
                looks=looks,
            )
        )
        is not None
    ]
    equal = next(
        item
        for item in evaluations
        if item["allocation"] == "equal"
    )
    if (
        equal["required_requests"]
        != equal_record[
            "required_labeled_requests"
        ]
    ):
        raise RuntimeError(
            "Equal alpha did not reproduce E70"
        )
    selected = select_best_allocation(
        evaluations=evaluations
    )
    if (
        selected["required_requests"]
        > equal["required_requests"]
    ):
        raise RuntimeError(
            "Selected alpha was worse than equal"
        )
    selected_index = int(
        selected["look_index"]
    )
    previous_request = (
        0
        if selected_index == 1
        else looks[selected_index - 2]
    )
    thresholds = {}
    for metric in METRICS:
        empirical, threshold, conditioning = (
            metric_specification(
                record=baseline,
                metric=metric,
                drift=False,
            )
        )
        weight = float(
            selected["weights"][metric]
        )
        thresholds[metric] = minimum_requests(
            low=previous_request,
            high=selected[
                "required_requests"
            ],
            predicate=lambda requests, empirical=empirical, threshold=threshold, conditioning=conditioning, weight=weight: metric_passes(
                requests=requests,
                look_index=selected_index,
                empirical=empirical,
                threshold=threshold,
                conditioning_probability=(
                    conditioning
                ),
                total_alpha=(
                    1.0
                    - float(
                        baseline[
                            "joint_confidence"
                        ]
                    )
                ),
                metric_weight=weight,
                dimension_count=int(
                    baseline[
                        "dimension_count"
                    ]
                ),
            ),
        )
    within_look_threshold = max(
        thresholds.values()
    )
    fixed_requests = baseline[
        "fixed_wilson_required_labeled_requests"
    ]
    anytime_requests = baseline[
        "required_labeled_requests"
    ]
    equal_requests = equal[
        "required_requests"
    ]
    return (
        {
            **common,
            "status": "certified",
            "selected_allocation": (
                selected["allocation"]
            ),
            "selected_weights": (
                selected["weights"]
            ),
            "selected_grid_units": (
                selected["grid_units"]
            ),
            "allocation_distance_from_equal": (
                selected[
                    "distance_from_equal"
                ]
            ),
            "required_labeled_requests": (
                selected[
                    "required_requests"
                ]
            ),
            "look_index": selected_index,
            "metric_look_indices": (
                selected[
                    "metric_look_indices"
                ]
            ),
            "limiting_metrics": (
                selected[
                    "limiting_metrics"
                ]
            ),
            "within_look_metric_threshold_requests": (
                thresholds
            ),
            "within_look_quantization_delay_requests": (
                selected[
                    "required_requests"
                ]
                - within_look_threshold
            ),
            "alpha_spent_fraction": (
                alpha_spent_fraction(
                    selected_index
                )
            ),
            "ratio_vs_equal_kl": (
                selected[
                    "required_requests"
                ]
                / equal_requests
            ),
            "ratio_vs_fixed_wilson": (
                selected[
                    "required_requests"
                ]
                / fixed_requests
            ),
            "ratio_vs_every_request_anytime": (
                selected[
                    "required_requests"
                ]
                / anytime_requests
            ),
            "request_savings_vs_equal_kl": (
                1.0
                - selected[
                    "required_requests"
                ]
                / equal_requests
            ),
            "candidate_allocation_count": len(
                evaluations
            ),
        },
        {
            "metric_looks": metric_looks,
            "equal": equal,
            "selected": selected,
        },
    )


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
    e70 = load_e70(args.root)
    allocations = allocation_grid()
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
    looks = {
        name: geometric_looks(
            ratio=ratio,
            maximum_requests=(
                args.maximum_requests
            ),
        )
        for name, ratio in schedules.items()
    }
    e70_certification = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            float(record["risk_budget"]),
            record["calibration_scenario"],
            record["confidence_profile"],
            record["schedule"],
        ): record
        for record in e70[
            "certification_records"
        ]
    }
    e70_drift = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            float(record["risk_budget"]),
            record["confidence_profile"],
            record["breach_metric"],
            record["schedule"],
        ): record
        for record in e70[
            "drift_detection_records"
        ]
    }
    certification_records = []
    case_tables: dict[
        tuple[Any, ...],
        dict[str, Any],
    ] = {}
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
        for schedule in schedule_names:
            record, table = (
                certification_record(
                    baseline=baseline,
                    equal_record=(
                        e70_certification[
                            (*base_key, schedule)
                        ]
                    ),
                    schedule=schedule,
                    looks=looks[schedule],
                    allocations=allocations,
                )
            )
            certification_records.append(
                record
            )
            if table is not None:
                case_tables[
                    (*base_key, schedule)
                ] = table
    expected_certification_count = (
        len(
            e68["certification_records"]
        )
        * len(schedule_names)
    )
    if (
        len(certification_records)
        != expected_certification_count
    ):
        raise RuntimeError(
            "Unexpected E71 certification count"
        )
    global_sweeps = {}
    global_selections = {}
    global_minimax_selections = {}
    globally_nonworsening_counts = {}
    for schedule in schedule_names:
        finite_keys = [
            key
            for key in case_tables
            if key[-1] == schedule
        ]
        sweep = []
        for allocation in allocations:
            ratios = []
            requests = []
            feasible = True
            for key in finite_keys:
                table = case_tables[key]
                evaluated = evaluate_allocation(
                    allocation=allocation,
                    metric_looks=table[
                        "metric_looks"
                    ],
                    looks=looks[schedule],
                )
                if evaluated is None:
                    feasible = False
                    break
                equal_requests = table[
                    "equal"
                ]["required_requests"]
                requests.append(
                    evaluated[
                        "required_requests"
                    ]
                )
                ratios.append(
                    evaluated[
                        "required_requests"
                    ]
                    / equal_requests
                )
            if not feasible:
                continue
            summary = summarize(ratios)
            sweep.append(
                {
                    "allocation": (
                        allocation[
                            "allocation"
                        ]
                    ),
                    "weights": (
                        allocation["weights"]
                    ),
                    "distance_from_equal": (
                        allocation[
                            "distance_from_equal"
                        ]
                    ),
                    "finite_case_count": len(
                        ratios
                    ),
                    "ratio_vs_equal_kl": (
                        summary
                    ),
                    "mean_required_requests": (
                        statistics.mean(
                            requests
                        )
                    ),
                    "geomean_ratio_vs_equal_kl": (
                        math.exp(
                            statistics.mean(
                                math.log(value)
                                for value
                                in ratios
                            )
                        )
                    ),
                    "improved_fraction": (
                        sum(
                            value
                            < 1.0 - 1e-12
                            for value in ratios
                        )
                        / len(ratios)
                    ),
                    "unchanged_fraction": (
                        sum(
                            abs(value - 1.0)
                            <= 1e-12
                            for value in ratios
                        )
                        / len(ratios)
                    ),
                    "worsened_fraction": (
                        sum(
                            value
                            > 1.0 + 1e-12
                            for value in ratios
                        )
                        / len(ratios)
                    ),
                }
            )
        selected = min(
            sweep,
            key=lambda item: (
                item[
                    "ratio_vs_equal_kl"
                ]["median"],
                item[
                    "ratio_vs_equal_kl"
                ]["p90"],
                item[
                    "geomean_ratio_vs_equal_kl"
                ],
                item[
                    "ratio_vs_equal_kl"
                ]["maximum"],
                item[
                    "distance_from_equal"
                ],
                item["allocation"],
            ),
        )
        minimax = min(
            sweep,
            key=lambda item: (
                item[
                    "ratio_vs_equal_kl"
                ]["maximum"],
                item[
                    "ratio_vs_equal_kl"
                ]["p90"],
                item[
                    "ratio_vs_equal_kl"
                ]["median"],
                item[
                    "geomean_ratio_vs_equal_kl"
                ],
                item[
                    "distance_from_equal"
                ],
                item["allocation"],
            ),
        )
        global_sweeps[schedule] = sweep
        global_selections[schedule] = (
            selected
        )
        global_minimax_selections[
            schedule
        ] = minimax
        globally_nonworsening_counts[
            schedule
        ] = sum(
            item[
                "ratio_vs_equal_kl"
            ]["maximum"]
            <= 1.0 + 1e-12
            for item in sweep
        )
    selected_balanced = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            float(record["risk_budget"]),
            record["confidence_profile"],
            record["schedule"],
        ): record
        for record in certification_records
        if (
            record[
                "calibration_scenario"
            ]
            == "balanced_margin"
            and record["status"]
            == "certified"
        )
    }
    drift_records = []
    equal_drift_reproduction_count = 0
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
        )
        for schedule in schedule_names:
            equal = e70_drift[
                (
                    *base_key,
                    baseline[
                        "breach_metric"
                    ],
                    schedule,
                )
            ]
            allocation_record = (
                selected_balanced.get(
                    (*base_key, schedule)
                )
            )
            common = {
                "ttft_slo_ms": (
                    baseline[
                        "ttft_slo_ms"
                    ]
                ),
                "family": (
                    baseline["family"]
                ),
                "error_profile": (
                    baseline[
                        "error_profile"
                    ]
                ),
                "risk_budget": (
                    baseline["risk_budget"]
                ),
                "confidence_profile": (
                    baseline[
                        "confidence_profile"
                    ]
                ),
                "breach_metric": (
                    baseline[
                        "breach_metric"
                    ]
                ),
                "schedule": schedule,
                "equal_kl_status": (
                    equal["status"]
                ),
                "equal_kl_required_post_change_requests": (
                    equal[
                        "required_post_change_requests"
                    ]
                ),
            }
            if baseline["status"] == (
                "not_applicable"
            ):
                drift_records.append(
                    {
                        **common,
                        "status": "not_applicable",
                        "selected_allocation": (
                            None
                        ),
                        "selected_weights": (
                            None
                        ),
                        "required_post_change_requests": (
                            0
                        ),
                        "ratio_vs_equal_kl": None,
                        "within_look_quantization_delay_requests": (
                            0
                        ),
                    }
                )
                continue
            equal_crossing = (
                weighted_drift_crossing(
                    record=baseline,
                    looks=looks[schedule],
                    metric_weight=1.0 / 3.0,
                )
            )
            if equal_crossing is None:
                raise RuntimeError(
                    "Equal drift did not cross"
                )
            if (
                equal_crossing[
                    "required_requests"
                ]
                != equal[
                    "required_post_change_requests"
                ]
            ):
                raise RuntimeError(
                    "Equal alpha did not reproduce "
                    "E70 drift"
                )
            equal_drift_reproduction_count += 1
            if allocation_record is None:
                drift_records.append(
                    {
                        **common,
                        "status": (
                            "inactive_uncertifiable"
                        ),
                        "selected_allocation": None,
                        "selected_weights": None,
                        "required_post_change_requests": None,
                        "ratio_vs_equal_kl": None,
                        "within_look_quantization_delay_requests": 0,
                    }
                )
                continue
            metric = baseline[
                "breach_metric"
            ]
            weight = float(
                allocation_record[
                    "selected_weights"
                ][metric]
            )
            crossing = weighted_drift_crossing(
                record=baseline,
                looks=looks[schedule],
                metric_weight=weight,
            )
            if crossing is None:
                raise RuntimeError(
                    "Allocated drift did not cross"
                )
            requests = crossing[
                "required_requests"
            ]
            equal_requests = equal[
                "required_post_change_requests"
            ]
            drift_records.append(
                {
                    **common,
                    "status": "detected",
                    "selected_allocation": (
                        allocation_record[
                            "selected_allocation"
                        ]
                    ),
                    "selected_weights": (
                        allocation_record[
                            "selected_weights"
                        ]
                    ),
                    "metric_alpha_weight": (
                        weight
                    ),
                    "required_post_change_requests": (
                        requests
                    ),
                    "look_index": (
                        crossing["look_index"]
                    ),
                    "within_look_quantization_delay_requests": (
                        crossing[
                            "within_look_quantization_delay_requests"
                        ]
                    ),
                    "detection_bound": (
                        crossing[
                            "detection_bound"
                        ]
                    ),
                    "effective_trials": (
                        crossing[
                            "effective_trials"
                        ]
                    ),
                    "stream_alpha": (
                        crossing[
                            "stream_alpha"
                        ]
                    ),
                    "ratio_vs_equal_kl": (
                        requests
                        / equal_requests
                    ),
                    "request_delta_vs_equal_kl": (
                        requests
                        - equal_requests
                    ),
                }
            )
    expected_drift_count = (
        len(
            e68["drift_detection_records"]
        )
        * len(schedule_names)
    )
    if len(drift_records) != (
        expected_drift_count
    ):
        raise RuntimeError(
            "Unexpected E71 drift count"
        )
    certification_summaries = {}
    drift_summaries = {}
    for schedule in schedule_names:
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
            "ratio_vs_equal_kl": summarize(
                [
                    float(
                        record[
                            "ratio_vs_equal_kl"
                        ]
                    )
                    for record in cert
                ]
            ),
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
            "selected_allocation_counts": dict(
                Counter(
                    record[
                        "selected_allocation"
                    ]
                    for record in cert
                )
            ),
            "limiting_metric_counts": dict(
                Counter(
                    metric
                    for record in cert
                    for metric
                    in record[
                        "limiting_metrics"
                    ]
                )
            ),
        }
        drift_summaries[schedule] = {
            "finite_record_count": len(
                drift
            ),
            "ratio_vs_equal_kl": summarize(
                [
                    float(
                        record[
                            "ratio_vs_equal_kl"
                        ]
                    )
                    for record in drift
                ]
            ),
            "slower_than_equal_fraction": (
                sum(
                    record[
                        "ratio_vs_equal_kl"
                    ]
                    > 1.0 + 1e-12
                    for record in drift
                )
                / len(drift)
            ),
            "faster_than_equal_fraction": (
                sum(
                    record[
                        "ratio_vs_equal_kl"
                    ]
                    < 1.0 - 1e-12
                    for record in drift
                )
                / len(drift)
            ),
            "ratio_by_metric": {
                metric: summarize(
                    [
                        float(
                            record[
                                "ratio_vs_equal_kl"
                            ]
                        )
                        for record in drift
                        if record[
                            "breach_metric"
                        ]
                        == metric
                    ]
                )
                for metric in METRICS
            },
        }
    equal_reproduction_checks = [
        {
            "key": [
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
                record["schedule"],
            ],
            "required_labeled_requests": (
                record[
                    "equal_kl_required_labeled_requests"
                ]
            ),
        }
        for record in certification_records
    ]
    result = {
        "suite": (
            "pacing_metric_alpha_"
            "allocation_analysis"
        ),
        "model": (
            "Keep E70's KL checkpoints and "
            "joint alpha, but replace equal "
            "Bonferroni metric shares with "
            "preregistered state-error, recall, "
            "and false-alarm weights. Each metric "
            "weight is divided equally across the "
            "family's keyed dimensions."
        ),
        "source": {
            "every_request_anytime": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            ),
            "equal_kl": (
                "results/raw/"
                "pacing_kl_monitoring/"
                "pacing_kl_monitoring_full.json"
            ),
        },
        "allocation_grid": {
            "unit_count": ALLOCATION_UNITS,
            "minimum_units_per_metric": (
                MINIMUM_UNITS
            ),
            "grid_allocation_count": (
                len(allocations) - 1
            ),
            "includes_exact_equal_split": (
                True
            ),
            "allocations": allocations,
        },
        "schedules": {
            schedule: {
                "ratio": schedules[
                    schedule
                ],
                "look_count": len(
                    looks[schedule]
                ),
                "alpha_spent_fraction": (
                    alpha_spent_fraction(
                        len(
                            looks[schedule]
                        )
                    )
                ),
            }
            for schedule in schedule_names
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
        "global_allocation_sweeps": (
            global_sweeps
        ),
        "global_selected_allocations": (
            global_selections
        ),
        "global_minimax_allocations": (
            global_minimax_selections
        ),
        "globally_nonworsening_allocation_counts": (
            globally_nonworsening_counts
        ),
        "validation": {
            "equal_split_reproduction_checks": (
                equal_reproduction_checks
            ),
            "equal_split_reproduction_count": (
                len(
                    equal_reproduction_checks
                )
            ),
            "equal_drift_reproduction_count": (
                equal_drift_reproduction_count
            ),
            "expected_certification_record_count": (
                expected_certification_count
            ),
            "expected_drift_record_count": (
                expected_drift_count
            ),
            "alpha_sum": (
                "Every allocation has positive "
                "weights summing to one; each "
                "metric share is divided equally "
                "across keyed dimensions."
            ),
            "certification_dominance": (
                "Per-case selection includes the "
                "exact equal split, so it never "
                "requires more requests than E70."
            ),
            "global_minimax": (
                "Global minimax selection is "
                "evaluated separately from the "
                "median-optimized allocation."
            ),
        },
        "limitations": [
            "Per-case allocations use assumed calibration rates and are scenario-conditioned offline oracles, not data-adaptive allocations.",
            "Selecting alpha weights from the same monitoring data would invalidate the guarantee; production weights must be preregistered or chosen with separate data.",
            "The 5% grid cannot represent every continuous allocation and reserves at least 5% of alpha for each metric family.",
            "Certification-optimized weights can delay detection of a de-emphasized metric; E71 reports that drift tradeoff but does not optimize a joint utility.",
            "All earlier independence, expected-proportion, checkpoint-only, and delayed-label limitations remain.",
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
