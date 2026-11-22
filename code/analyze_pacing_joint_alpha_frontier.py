from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analyze_pacing_alpha_allocation import (
    METRICS,
    allocation_grid,
    load_e68,
    load_e70,
    summarize,
    weighted_drift_crossing,
)
from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
    geometric_looks,
)


DRIFT_CAPS = (
    1.00,
    1.05,
    1.10,
    1.25,
    1.50,
    2.00,
)
EPSILON = 1e-12


def load_e71(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_alpha_allocation"
        / "pacing_alpha_allocation_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def geometric_mean(values: list[float]) -> float:
    return math.exp(
        statistics.mean(
            math.log(value)
            for value in values
        )
    )


def cap_name(cap: float) -> str:
    return (
        f"max_{cap:.2f}"
        .replace(".", "p")
    )


def drift_allocation_sweep(
    *,
    baselines: list[dict[str, Any]],
    equal_records: dict[
        tuple[Any, ...],
        dict[str, Any],
    ],
    allocations: list[dict[str, Any]],
    schedule: str,
    looks: list[int],
) -> tuple[list[dict[str, Any]], int]:
    trackers = {
        allocation["allocation"]: {
            "ratios": [],
            "requests": [],
            "ratio_by_metric": {
                metric: []
                for metric in METRICS
            },
            "feasible": True,
        }
        for allocation in allocations
    }
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
    equal_reproduction_count = 0
    for baseline in baselines:
        key = (
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
            schedule,
        )
        equal = equal_records[key]
        if (
            baseline["status"]
            != equal["status"]
        ):
            raise RuntimeError(
                "E68/E70 drift status mismatch"
            )
        if baseline["status"] == (
            "not_applicable"
        ):
            continue
        if baseline["status"] != "detected":
            raise RuntimeError(
                "Unexpected E68 drift status"
            )
        metric = baseline["breach_metric"]
        requests_by_weight = {}
        for weight in unique_weights:
            crossing = weighted_drift_crossing(
                record=baseline,
                looks=looks,
                metric_weight=weight,
            )
            requests_by_weight[weight] = (
                None
                if crossing is None
                else crossing[
                    "required_requests"
                ]
            )
        equal_requests = int(
            equal[
                "required_post_change_requests"
            ]
        )
        reproduced = requests_by_weight[
            1.0 / 3.0
        ]
        if reproduced != equal_requests:
            raise RuntimeError(
                "Equal alpha did not reproduce "
                "E70 drift"
            )
        equal_reproduction_count += 1
        for allocation in allocations:
            tracker = trackers[
                allocation["allocation"]
            ]
            weight = float(
                allocation["weights"][metric]
            )
            requests = requests_by_weight[
                weight
            ]
            if requests is None:
                tracker["feasible"] = False
                continue
            ratio = requests / equal_requests
            tracker["ratios"].append(ratio)
            tracker["requests"].append(
                requests
            )
            tracker["ratio_by_metric"][
                metric
            ].append(ratio)
    sweep = []
    for allocation in allocations:
        tracker = trackers[
            allocation["allocation"]
        ]
        if not tracker["feasible"]:
            continue
        ratios = tracker["ratios"]
        if not ratios:
            raise RuntimeError(
                "No finite drift records"
            )
        by_metric = {}
        for metric in METRICS:
            metric_ratios = tracker[
                "ratio_by_metric"
            ][metric]
            by_metric[metric] = {
                "ratio_vs_equal_kl": (
                    summarize(
                        metric_ratios
                    )
                ),
                "geomean_ratio_vs_equal_kl": (
                    geometric_mean(
                        metric_ratios
                    )
                ),
                "slower_fraction": (
                    sum(
                        value
                        > 1.0 + EPSILON
                        for value
                        in metric_ratios
                    )
                    / len(metric_ratios)
                ),
                "faster_fraction": (
                    sum(
                        value
                        < 1.0 - EPSILON
                        for value
                        in metric_ratios
                    )
                    / len(metric_ratios)
                ),
            }
        sweep.append(
            {
                "allocation": (
                    allocation["allocation"]
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
                    summarize(ratios)
                ),
                "geomean_ratio_vs_equal_kl": (
                    geometric_mean(ratios)
                ),
                "mean_required_requests": (
                    statistics.mean(
                        tracker["requests"]
                    )
                ),
                "slower_fraction": (
                    sum(
                        value
                        > 1.0 + EPSILON
                        for value in ratios
                    )
                    / len(ratios)
                ),
                "unchanged_fraction": (
                    sum(
                        abs(value - 1.0)
                        <= EPSILON
                        for value in ratios
                    )
                    / len(ratios)
                ),
                "faster_fraction": (
                    sum(
                        value
                        < 1.0 - EPSILON
                        for value in ratios
                    )
                    / len(ratios)
                ),
                "ratio_by_metric": by_metric,
            }
        )
    return (
        sweep,
        equal_reproduction_count,
    )


def join_sweeps(
    *,
    certification: list[dict[str, Any]],
    drift: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cert_by_allocation = {
        item["allocation"]: item
        for item in certification
    }
    joined = []
    for drift_item in drift:
        cert_item = cert_by_allocation[
            drift_item["allocation"]
        ]
        cert_geomean = float(
            cert_item[
                "geomean_ratio_vs_equal_kl"
            ]
        )
        drift_geomean = float(
            drift_item[
                "geomean_ratio_vs_equal_kl"
            ]
        )
        cert_count = int(
            cert_item["finite_case_count"]
        )
        drift_count = int(
            drift_item["finite_case_count"]
        )
        joined.append(
            {
                "allocation": (
                    drift_item["allocation"]
                ),
                "weights": (
                    drift_item["weights"]
                ),
                "distance_from_equal": (
                    drift_item[
                        "distance_from_equal"
                    ]
                ),
                "certification": (
                    cert_item
                ),
                "drift": drift_item,
                "joint": {
                    "balanced_domain_geomean_ratio": (
                        math.sqrt(
                            cert_geomean
                            * drift_geomean
                        )
                    ),
                    "record_weighted_geomean_ratio": (
                        math.exp(
                            (
                                cert_count
                                * math.log(
                                    cert_geomean
                                )
                                + drift_count
                                * math.log(
                                    drift_geomean
                                )
                            )
                            / (
                                cert_count
                                + drift_count
                            )
                        )
                    ),
                    "maximum_ratio": max(
                        float(
                            cert_item[
                                "ratio_vs_equal_kl"
                            ]["maximum"]
                        ),
                        float(
                            drift_item[
                                "ratio_vs_equal_kl"
                            ]["maximum"]
                        ),
                    ),
                    "p90_maximum_ratio": max(
                        float(
                            cert_item[
                                "ratio_vs_equal_kl"
                            ]["p90"]
                        ),
                        float(
                            drift_item[
                                "ratio_vs_equal_kl"
                            ]["p90"]
                        ),
                    ),
                    "certification_nonworsening": (
                        cert_item[
                            "ratio_vs_equal_kl"
                        ]["maximum"]
                        <= 1.0 + EPSILON
                    ),
                    "drift_nonworsening": (
                        drift_item[
                            "ratio_vs_equal_kl"
                        ]["maximum"]
                        <= 1.0 + EPSILON
                    ),
                },
            }
        )
    return joined


def selection_snapshot(
    item: dict[str, Any],
) -> dict[str, Any]:
    certification = item[
        "certification"
    ]
    drift = item["drift"]
    return {
        "allocation": item["allocation"],
        "weights": item["weights"],
        "distance_from_equal": (
            item["distance_from_equal"]
        ),
        "certification": {
            "ratio_vs_equal_kl": (
                certification[
                    "ratio_vs_equal_kl"
                ]
            ),
            "geomean_ratio_vs_equal_kl": (
                certification[
                    "geomean_ratio_vs_equal_kl"
                ]
            ),
            "improved_fraction": (
                certification[
                    "improved_fraction"
                ]
            ),
            "worsened_fraction": (
                certification[
                    "worsened_fraction"
                ]
            ),
        },
        "drift": {
            "ratio_vs_equal_kl": (
                drift[
                    "ratio_vs_equal_kl"
                ]
            ),
            "geomean_ratio_vs_equal_kl": (
                drift[
                    "geomean_ratio_vs_equal_kl"
                ]
            ),
            "faster_fraction": (
                drift["faster_fraction"]
            ),
            "slower_fraction": (
                drift["slower_fraction"]
            ),
            "ratio_by_metric": (
                drift["ratio_by_metric"]
            ),
        },
        "joint": item["joint"],
    }


def certification_key(
    item: dict[str, Any],
) -> tuple[Any, ...]:
    ratio = item["certification"][
        "ratio_vs_equal_kl"
    ]
    return (
        ratio["median"],
        ratio["p90"],
        item["certification"][
            "geomean_ratio_vs_equal_kl"
        ],
        ratio["maximum"],
        item["drift"][
            "ratio_vs_equal_kl"
        ]["maximum"],
        item["distance_from_equal"],
        item["allocation"],
    )


def pareto_front(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    front = []
    for item in items:
        cert = item["certification"][
            "geomean_ratio_vs_equal_kl"
        ]
        drift = item["drift"][
            "geomean_ratio_vs_equal_kl"
        ]
        dominated = any(
            (
                other["certification"][
                    "geomean_ratio_vs_equal_kl"
                ]
                <= cert + EPSILON
                and other["drift"][
                    "geomean_ratio_vs_equal_kl"
                ]
                <= drift + EPSILON
                and (
                    other["certification"][
                        "geomean_ratio_vs_equal_kl"
                    ]
                    < cert - EPSILON
                    or other["drift"][
                        "geomean_ratio_vs_equal_kl"
                    ]
                    < drift - EPSILON
                )
            )
            for other in items
            if other is not item
        )
        if not dominated:
            front.append(item)
    return sorted(
        front,
        key=lambda item: (
            item["certification"][
                "geomean_ratio_vs_equal_kl"
            ],
            item["drift"][
                "geomean_ratio_vs_equal_kl"
            ],
            item["distance_from_equal"],
            item["allocation"],
        ),
    )


def cap_selections(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    selections = {}
    for cap in DRIFT_CAPS:
        eligible = [
            item
            for item in items
            if item["drift"][
                "ratio_vs_equal_kl"
            ]["maximum"]
            <= cap + EPSILON
        ]
        nonworsening = [
            item
            for item in eligible
            if item["certification"][
                "ratio_vs_equal_kl"
            ]["maximum"]
            <= 1.0 + EPSILON
        ]
        if not eligible or not nonworsening:
            raise RuntimeError(
                "Equal alpha must satisfy every "
                "drift cap"
            )
        selections[cap_name(cap)] = {
            "maximum_drift_ratio": cap,
            "eligible_allocation_count": len(
                eligible
            ),
            "certification_nonworsening_count": (
                len(nonworsening)
            ),
            "certification_first": (
                selection_snapshot(
                    min(
                        eligible,
                        key=certification_key,
                    )
                )
            ),
            "certification_nonworsening": (
                selection_snapshot(
                    min(
                        nonworsening,
                        key=certification_key,
                    )
                )
            ),
        }
    return selections


def run(args: argparse.Namespace) -> dict[str, Any]:
    e68 = load_e68(args.root)
    e70 = load_e70(args.root)
    e71 = load_e71(args.root)
    allocations = allocation_grid()
    schedule_names = [
        value.strip()
        for value
        in args.schedules.split(",")
        if value.strip()
    ]
    unknown = (
        set(schedule_names)
        - set(LOOK_SCHEDULES)
    )
    if unknown:
        raise ValueError(
            f"Unknown schedules: {sorted(unknown)}"
        )
    looks = {
        schedule: geometric_looks(
            ratio=LOOK_SCHEDULES[
                schedule
            ],
            maximum_requests=(
                args.maximum_requests
            ),
        )
        for schedule in schedule_names
    }
    equal_drift = {
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
    allocation_summaries = {}
    pareto_fronts = {}
    selections = {}
    equal_reproduction_count = 0
    expected_finite_drift = sum(
        record["status"] == "detected"
        for record in e68[
            "drift_detection_records"
        ]
    )
    for schedule in schedule_names:
        drift_sweep, reproduced = (
            drift_allocation_sweep(
                baselines=e68[
                    "drift_detection_records"
                ],
                equal_records=equal_drift,
                allocations=allocations,
                schedule=schedule,
                looks=looks[schedule],
            )
        )
        equal_reproduction_count += (
            reproduced
        )
        if reproduced != (
            expected_finite_drift
        ):
            raise RuntimeError(
                "Unexpected equal drift "
                "reproduction count"
            )
        certification_sweep = e71[
            "global_allocation_sweeps"
        ][schedule]
        joined = join_sweeps(
            certification=(
                certification_sweep
            ),
            drift=drift_sweep,
        )
        if len(joined) != len(allocations):
            raise RuntimeError(
                "Unexpected joint allocation "
                "count"
            )
        equal = next(
            item
            for item in joined
            if item["allocation"] == "equal"
        )
        e71_selected_name = e71[
            "global_selected_allocations"
        ][schedule]["allocation"]
        e71_minimax_name = e71[
            "global_minimax_allocations"
        ][schedule]["allocation"]
        joint_geomean = min(
            joined,
            key=lambda item: (
                item["joint"][
                    "balanced_domain_geomean_ratio"
                ],
                item["joint"][
                    "maximum_ratio"
                ],
                item[
                    "distance_from_equal"
                ],
                item["allocation"],
            ),
        )
        joint_minimax = min(
            joined,
            key=lambda item: (
                item["joint"][
                    "maximum_ratio"
                ],
                item["joint"][
                    "p90_maximum_ratio"
                ],
                item["joint"][
                    "balanced_domain_geomean_ratio"
                ],
                item[
                    "distance_from_equal"
                ],
                item["allocation"],
            ),
        )
        front = pareto_front(joined)
        allocation_summaries[
            schedule
        ] = joined
        pareto_fronts[schedule] = [
            selection_snapshot(item)
            for item in front
        ]
        selections[schedule] = {
            "equal": (
                selection_snapshot(equal)
            ),
            "e71_certification_only": (
                selection_snapshot(
                    next(
                        item
                        for item in joined
                        if item["allocation"]
                        == e71_selected_name
                    )
                )
            ),
            "e71_certification_minimax": (
                selection_snapshot(
                    next(
                        item
                        for item in joined
                        if item["allocation"]
                        == e71_minimax_name
                    )
                )
            ),
            "joint_geomean": (
                selection_snapshot(
                    joint_geomean
                )
            ),
            "joint_minimax": (
                selection_snapshot(
                    joint_minimax
                )
            ),
            "by_worst_drift_cap": (
                cap_selections(joined)
            ),
            "pareto_allocation_count": len(
                front
            ),
            "certification_nonworsening_count": (
                sum(
                    item["joint"][
                        "certification_nonworsening"
                    ]
                    for item in joined
                )
            ),
            "drift_nonworsening_count": (
                sum(
                    item["joint"][
                        "drift_nonworsening"
                    ]
                    for item in joined
                )
            ),
            "joint_nonworsening_count": (
                sum(
                    item["joint"][
                        "certification_nonworsening"
                    ]
                    and item["joint"][
                        "drift_nonworsening"
                    ]
                    for item in joined
                )
            ),
        }
    result = {
        "suite": (
            "pacing_joint_alpha_frontier"
        ),
        "model": (
            "Evaluate every E71 metric-alpha "
            "allocation as one preregistered "
            "global controller configuration. "
            "Join its E71 certification cost "
            "with fresh E70-style drift response "
            "for every active breach, then select "
            "allocations under explicit worst-case "
            "drift-slowdown caps."
        ),
        "source": {
            "anytime_baseline": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            ),
            "equal_kl": (
                "results/raw/"
                "pacing_kl_monitoring/"
                "pacing_kl_monitoring_full.json"
            ),
            "metric_alpha": (
                "results/raw/"
                "pacing_alpha_allocation/"
                "pacing_alpha_allocation_full.json"
            ),
        },
        "allocation_count": len(
            allocations
        ),
        "worst_drift_ratio_caps": list(
            DRIFT_CAPS
        ),
        "schedules": {
            schedule: {
                "ratio": LOOK_SCHEDULES[
                    schedule
                ],
                "look_count": len(
                    looks[schedule]
                ),
            }
            for schedule in schedule_names
        },
        "allocation_summaries": (
            allocation_summaries
        ),
        "pareto_fronts": pareto_fronts,
        "selections": selections,
        "validation": {
            "equal_drift_reproduction_count": (
                equal_reproduction_count
            ),
            "expected_equal_drift_reproduction_count": (
                expected_finite_drift
                * len(schedule_names)
            ),
            "allocation_count_per_schedule": {
                schedule: len(
                    allocation_summaries[
                        schedule
                    ]
                )
                for schedule in schedule_names
            },
            "equal_control": (
                "Equal alpha reproduces every "
                "finite E70 drift crossing exactly "
                "and is feasible under every cap."
            ),
            "cap_feasibility": (
                "Each cap-selected allocation has "
                "maximum drift ratio no greater "
                "than the declared cap."
            ),
            "pareto_definition": (
                "An allocation is retained when no "
                "other allocation has no-larger "
                "certification and drift geometric "
                "means with at least one strict "
                "improvement."
            ),
        },
        "limitations": [
            "The joint frontier is offline and assumes the same scenario weighting as E68-E71; production traffic prevalence can change the aggregate optimum.",
            "Certification and drift geometric means are descriptive objectives, not a utility function. Operational selection should use an explicit worst-drift cap.",
            "Worst-case ratios are checkpoint-quantized and can change discontinuously when a boundary crosses one scheduled look.",
            "The 5% positive-weight grid excludes continuous optima and reserves alpha for every metric family.",
            "Weights remain valid only when preregistered or selected from independent data; monitored outcomes cannot choose their own allocation without additional multiplicity control.",
            "All earlier independence, expected-proportion, delayed-label, and checkpoint-only limitations remain.",
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
