from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_pacing_alpha_allocation import (
    allocation_grid,
    evaluate_allocation,
    load_e68,
    load_e70,
    precompute_metric_looks,
    summarize,
    weighted_drift_crossing,
)
from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
    geometric_looks,
)
from analyze_pacing_checkpoint_phase_sensitivity import (
    PHASE_DENOMINATOR,
    ROUNDING_MODES,
    rounded_request,
)


EPSILON = 1e-12


def direct_phase_looks(
    *,
    ratio: float,
    phase: float,
    rounding: str,
    maximum_requests: int,
) -> list[int]:
    if ratio <= 1.0:
        raise ValueError(
            "Geometric ratio must exceed one"
        )
    if not 0.0 <= phase < 1.0:
        raise ValueError(
            "Phase must be in [0, 1)"
        )
    looks = [1]
    exponent = 1.0 + phase
    while looks[-1] < maximum_requests:
        raw = ratio**exponent
        candidate = (
            maximum_requests
            if raw >= maximum_requests
            else max(
                1,
                rounded_request(
                    raw,
                    rounding,
                ),
            )
        )
        if candidate > looks[-1]:
            looks.append(
                min(
                    candidate,
                    maximum_requests,
                )
            )
        if looks[-1] == maximum_requests:
            break
        exponent += 1.0
    return looks


def schedule_grids(
    *,
    schedule: str,
    ratio: float,
    maximum_requests: int,
) -> list[dict[str, Any]]:
    by_looks: dict[
        tuple[int, ...],
        dict[str, Any],
    ] = {}

    def add(
        looks: list[int],
        alias: dict[str, Any],
    ) -> None:
        key = tuple(looks)
        if key not in by_looks:
            by_looks[key] = {
                "grid": (
                    f"{schedule}_grid_"
                    f"{len(by_looks):02d}"
                ),
                "looks": looks,
                "aliases": [],
                "is_e70_base_grid": False,
            }
        by_looks[key]["aliases"].append(
            alias
        )

    base = geometric_looks(
        ratio=ratio,
        maximum_requests=(
            maximum_requests
        ),
    )
    add(
        base,
        {
            "construction": (
                "recursive_ceil"
            ),
            "phase_units": 0,
            "phase_denominator": (
                PHASE_DENOMINATOR
            ),
            "phase": 0.0,
            "rounding": "ceil",
        },
    )
    base_key = tuple(base)
    for phase_units in range(
        PHASE_DENOMINATOR
    ):
        phase = (
            phase_units
            / PHASE_DENOMINATOR
        )
        for rounding in ROUNDING_MODES:
            add(
                direct_phase_looks(
                    ratio=ratio,
                    phase=phase,
                    rounding=rounding,
                    maximum_requests=(
                        maximum_requests
                    ),
                ),
                {
                    "construction": (
                        "direct_log_phase"
                    ),
                    "phase_units": (
                        phase_units
                    ),
                    "phase_denominator": (
                        PHASE_DENOMINATOR
                    ),
                    "phase": phase,
                    "rounding": rounding,
                },
            )
    by_looks[base_key][
        "is_e70_base_grid"
    ] = True
    return list(by_looks.values())


def summarize_costs(
    *,
    values: list[int],
    grids: list[dict[str, Any]],
    baseline: int,
    reference: int,
) -> dict[str, Any]:
    minimum = min(values)
    maximum = max(values)
    median = statistics.median(values)
    min_index = values.index(minimum)
    max_index = values.index(maximum)
    return {
        "grid_count": len(values),
        "base_requests": baseline,
        "minimum_requests": minimum,
        "median_requests": median,
        "maximum_requests": maximum,
        "minimum_grid": (
            grids[min_index]["grid"]
        ),
        "minimum_grid_aliases": (
            grids[min_index]["aliases"]
        ),
        "maximum_grid": (
            grids[max_index]["grid"]
        ),
        "maximum_grid_aliases": (
            grids[max_index]["aliases"]
        ),
        "phase_oracle_ratio_vs_base": (
            minimum / baseline
        ),
        "median_ratio_vs_base": (
            median / baseline
        ),
        "robust_ratio_vs_base": (
            maximum / baseline
        ),
        "grid_spread_ratio": (
            maximum / minimum
        ),
        "base_ratio_vs_reference": (
            baseline / reference
        ),
        "robust_ratio_vs_reference": (
            maximum / reference
        ),
    }


def aggregate_schedule(
    records: list[dict[str, Any]],
    *,
    domain: str,
    schedule: str,
) -> dict[str, Any]:
    values = [
        record["schedules"][schedule]
        for record in records
    ]
    return {
        "case_count": len(values),
        "phase_oracle_ratio_vs_base": (
            summarize(
                [
                    value[
                        "phase_oracle_ratio_vs_base"
                    ]
                    for value in values
                ]
            )
        ),
        "median_ratio_vs_base": (
            summarize(
                [
                    value[
                        "median_ratio_vs_base"
                    ]
                    for value in values
                ]
            )
        ),
        "robust_ratio_vs_base": (
            summarize(
                [
                    value[
                        "robust_ratio_vs_base"
                    ]
                    for value in values
                ]
            )
        ),
        "grid_spread_ratio": (
            summarize(
                [
                    value[
                        "grid_spread_ratio"
                    ]
                    for value in values
                ]
            )
        ),
        "base_ratio_vs_reference": (
            summarize(
                [
                    value[
                        "base_ratio_vs_reference"
                    ]
                    for value in values
                ]
            )
        ),
        "robust_ratio_vs_reference": (
            summarize(
                [
                    value[
                        "robust_ratio_vs_reference"
                    ]
                    for value in values
                ]
            )
        ),
        "worst_phase_penalty_case": max(
            (
                {
                    "key": record["key"],
                    "ratio": record[
                        "schedules"
                    ][schedule][
                        "robust_ratio_vs_base"
                    ],
                    "base_requests": record[
                        "schedules"
                    ][schedule][
                        "base_requests"
                    ],
                    "maximum_requests": record[
                        "schedules"
                    ][schedule][
                        "maximum_requests"
                    ],
                    "maximum_grid": record[
                        "schedules"
                    ][schedule][
                        "maximum_grid"
                    ],
                    "maximum_grid_aliases": (
                        record[
                            "schedules"
                        ][schedule][
                            "maximum_grid_aliases"
                        ]
                    ),
                }
                for record in records
            ),
            key=lambda item: (
                item["ratio"],
                tuple(
                    str(value)
                    for value in item["key"]
                ),
            ),
        ),
        "domain": domain,
    }


def winner_summary(
    records: list[dict[str, Any]],
    schedules: list[str],
) -> dict[str, Any]:
    nominal_membership = Counter()
    robust_membership = Counter()
    nominal_unique = Counter()
    robust_unique = Counter()
    retention = 0
    for record in records:
        nominal = record[
            "nominal_winner_schedules"
        ]
        robust = record[
            "robust_winner_schedules"
        ]
        nominal_membership.update(nominal)
        robust_membership.update(robust)
        if len(nominal) == 1:
            nominal_unique[nominal[0]] += 1
        if len(robust) == 1:
            robust_unique[robust[0]] += 1
        retention += bool(
            set(nominal) & set(robust)
        )
    return {
        "case_count": len(records),
        "nominal_winner_membership_counts": {
            schedule: (
                nominal_membership[
                    schedule
                ]
            )
            for schedule in schedules
        },
        "robust_winner_membership_counts": {
            schedule: (
                robust_membership[schedule]
            )
            for schedule in schedules
        },
        "nominal_unique_winner_counts": {
            schedule: nominal_unique[
                schedule
            ]
            for schedule in schedules
        },
        "robust_unique_winner_counts": {
            schedule: robust_unique[
                schedule
            ]
            for schedule in schedules
        },
        "nominal_winner_retention_fraction": (
            retention / len(records)
        ),
    }


def fixed_schedule_regret(
    records: list[dict[str, Any]],
    schedules: list[str],
) -> dict[str, Any]:
    results = {}
    for schedule in schedules:
        ratios = []
        for record in records:
            robust_oracle = min(
                record["schedules"][
                    candidate
                ]["maximum_requests"]
                for candidate in schedules
            )
            ratios.append(
                record["schedules"][
                    schedule
                ]["maximum_requests"]
                / robust_oracle
            )
        results[schedule] = {
            "ratio_vs_per_case_robust_oracle": (
                summarize(ratios)
            ),
            "geomean_ratio_vs_per_case_robust_oracle": (
                math.exp(
                    statistics.mean(
                        math.log(value)
                        for value in ratios
                    )
                )
            ),
        }
    return results


def make_case_record(
    *,
    key: list[Any],
    costs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    nominal_minimum = min(
        value["base_requests"]
        for value in costs.values()
    )
    robust_minimum = min(
        value["maximum_requests"]
        for value in costs.values()
    )
    return {
        "key": key,
        "schedules": costs,
        "nominal_winner_schedules": [
            schedule
            for schedule, value
            in costs.items()
            if value["base_requests"]
            == nominal_minimum
        ],
        "robust_winner_schedules": [
            schedule
            for schedule, value
            in costs.items()
            if value["maximum_requests"]
            == robust_minimum
        ],
        "nominal_minimum_requests": (
            nominal_minimum
        ),
        "robust_minimum_requests": (
            robust_minimum
        ),
        "robust_oracle_ratio_vs_nominal_oracle": (
            robust_minimum
            / nominal_minimum
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e68 = load_e68(args.root)
    e70 = load_e70(args.root)
    equal = next(
        allocation
        for allocation in allocation_grid()
        if allocation["allocation"]
        == "equal"
    )
    schedule_names = list(
        LOOK_SCHEDULES
    )
    grids = {
        schedule: schedule_grids(
            schedule=schedule,
            ratio=LOOK_SCHEDULES[
                schedule
            ],
            maximum_requests=(
                args.maximum_requests
            ),
        )
        for schedule in schedule_names
    }
    e70_certification = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            float(record["risk_budget"]),
            record[
                "calibration_scenario"
            ],
            record[
                "confidence_profile"
            ],
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
            record[
                "confidence_profile"
            ],
            record["breach_metric"],
            record["schedule"],
        ): record
        for record in e70[
            "drift_detection_records"
        ]
    }
    certification_records = []
    certification_reproduction = 0
    for baseline in e68[
        "certification_records"
    ]:
        if baseline["status"] != (
            "certified"
        ):
            continue
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
        costs = {}
        for schedule in schedule_names:
            requests = []
            base_requests = None
            for grid in grids[schedule]:
                metric_looks = (
                    precompute_metric_looks(
                        record=baseline,
                        looks=grid["looks"],
                        allocations=[equal],
                    )
                )
                evaluated = evaluate_allocation(
                    allocation=equal,
                    metric_looks=metric_looks,
                    looks=grid["looks"],
                )
                if evaluated is None:
                    raise RuntimeError(
                        "Certification did not "
                        "cross"
                    )
                value = int(
                    evaluated[
                        "required_requests"
                    ]
                )
                requests.append(value)
                if grid[
                    "is_e70_base_grid"
                ]:
                    base_requests = value
            if base_requests is None:
                raise RuntimeError(
                    "Missing base certification "
                    "grid"
                )
            expected = e70_certification[
                (*base_key, schedule)
            ]
            if (
                base_requests
                != expected[
                    "required_labeled_requests"
                ]
            ):
                raise RuntimeError(
                    "Base certification did not "
                    "reproduce E70"
                )
            certification_reproduction += 1
            costs[schedule] = (
                summarize_costs(
                    values=requests,
                    grids=grids[schedule],
                    baseline=base_requests,
                    reference=int(
                        baseline[
                            "fixed_wilson_"
                            "required_labeled_"
                            "requests"
                        ]
                    ),
                )
            )
        certification_records.append(
            make_case_record(
                key=list(base_key),
                costs=costs,
            )
        )
    drift_records = []
    drift_reproduction = 0
    for baseline in e68[
        "drift_detection_records"
    ]:
        if baseline["status"] != "detected":
            continue
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
        costs = {}
        for schedule in schedule_names:
            requests = []
            base_requests = None
            for grid in grids[schedule]:
                crossing = (
                    weighted_drift_crossing(
                        record=baseline,
                        looks=grid["looks"],
                        metric_weight=1.0 / 3.0,
                    )
                )
                if crossing is None:
                    raise RuntimeError(
                        "Drift did not cross"
                    )
                value = int(
                    crossing[
                        "required_requests"
                    ]
                )
                requests.append(value)
                if grid[
                    "is_e70_base_grid"
                ]:
                    base_requests = value
            if base_requests is None:
                raise RuntimeError(
                    "Missing base drift grid"
                )
            expected = e70_drift[
                (*base_key, schedule)
            ]
            if (
                base_requests
                != expected[
                    "required_post_change_requests"
                ]
            ):
                raise RuntimeError(
                    "Base drift did not "
                    "reproduce E70"
                )
            drift_reproduction += 1
            costs[schedule] = (
                summarize_costs(
                    values=requests,
                    grids=grids[schedule],
                    baseline=base_requests,
                    reference=int(
                        baseline[
                            "required_post_change_"
                            "requests"
                        ]
                    ),
                )
            )
        drift_records.append(
            make_case_record(
                key=list(base_key),
                costs=costs,
            )
        )
    if len(certification_records) != 1918:
        raise RuntimeError(
            "Unexpected certification count"
        )
    if len(drift_records) != 1890:
        raise RuntimeError(
            "Unexpected drift count"
        )
    certification_summary = {
        schedule: aggregate_schedule(
            certification_records,
            domain="certification",
            schedule=schedule,
        )
        for schedule in schedule_names
    }
    drift_summary = {
        schedule: aggregate_schedule(
            drift_records,
            domain="drift",
            schedule=schedule,
        )
        for schedule in schedule_names
    }
    certification_regret = (
        fixed_schedule_regret(
            certification_records,
            schedule_names,
        )
    )
    drift_regret = fixed_schedule_regret(
        drift_records,
        schedule_names,
    )
    certification_global_minimax = min(
        schedule_names,
        key=lambda schedule: (
            certification_regret[
                schedule
            ][
                "ratio_vs_per_case_"
                "robust_oracle"
            ]["maximum"],
            certification_regret[
                schedule
            ][
                "ratio_vs_per_case_"
                "robust_oracle"
            ]["p90"],
            certification_regret[
                schedule
            ][
                "geomean_ratio_vs_per_"
                "case_robust_oracle"
            ],
            schedule,
        ),
    )
    drift_global_minimax = min(
        schedule_names,
        key=lambda schedule: (
            drift_regret[schedule][
                "ratio_vs_per_case_"
                "robust_oracle"
            ]["maximum"],
            drift_regret[schedule][
                "ratio_vs_per_case_"
                "robust_oracle"
            ]["p90"],
            drift_regret[schedule][
                "geomean_ratio_vs_per_"
                "case_robust_oracle"
            ],
            schedule,
        ),
    )
    result = {
        "suite": (
            "pacing_cadence_robustness"
        ),
        "model": (
            "Hold equal metric alpha fixed and "
            "compare 1.10x, 1.25x, 1.50x, and "
            "2.00x checkpoint cadences under the "
            "original recursive E70 grid plus "
            "direct logarithmic phase grids with "
            "ceil, nearest, and floor rounding. "
            "Robust cost is the maximum request "
            "count across valid grid constructions."
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
            "cross_grid_alpha": (
                "results/raw/"
                "pacing_cross_grid_alpha/"
                "pacing_cross_grid_alpha_full.json"
            ),
        },
        "schedules": {
            schedule: {
                "ratio": (
                    LOOK_SCHEDULES[
                        schedule
                    ]
                ),
                "requested_grid_count": (
                    1
                    + PHASE_DENOMINATOR
                    * len(
                        ROUNDING_MODES
                    )
                ),
                "unique_grid_count": len(
                    grids[schedule]
                ),
                "grids": [
                    {
                        "grid": grid["grid"],
                        "aliases": (
                            grid["aliases"]
                        ),
                        "look_count": len(
                            grid["looks"]
                        ),
                        "first_looks": (
                            grid["looks"][:12]
                        ),
                        "last_looks": (
                            grid["looks"][-4:]
                        ),
                        "is_e70_base_grid": (
                            grid[
                                "is_e70_base_grid"
                            ]
                        ),
                    }
                    for grid in grids[
                        schedule
                    ]
                ],
            }
            for schedule in schedule_names
        },
        "certification_records": (
            certification_records
        ),
        "drift_records": drift_records,
        "certification_schedule_summaries": (
            certification_summary
        ),
        "drift_schedule_summaries": (
            drift_summary
        ),
        "certification_winners": (
            winner_summary(
                certification_records,
                schedule_names,
            )
        ),
        "drift_winners": winner_summary(
            drift_records,
            schedule_names,
        ),
        "certification_fixed_schedule_regret": (
            certification_regret
        ),
        "drift_fixed_schedule_regret": (
            drift_regret
        ),
        "certification_global_minimax_schedule": (
            certification_global_minimax
        ),
        "drift_global_minimax_schedule": (
            drift_global_minimax
        ),
        "validation": {
            "equal_alpha": (
                "All schedules use exact equal "
                "metric alpha divided equally "
                "across keyed dimensions."
            ),
            "certification_base_reproduction_count": (
                certification_reproduction
            ),
            "expected_certification_base_reproduction_count": (
                1918
                * len(schedule_names)
            ),
            "drift_base_reproduction_count": (
                drift_reproduction
            ),
            "expected_drift_base_reproduction_count": (
                1890
                * len(schedule_names)
            ),
            "grid_uniqueness": {
                schedule: len(
                    {
                        tuple(grid["looks"])
                        for grid in grids[
                            schedule
                        ]
                    }
                )
                for schedule in schedule_names
            },
            "winner_definition": (
                "Nominal winners minimize the "
                "E70 base-grid request count; "
                "robust winners minimize the "
                "maximum request count across "
                "each cadence's valid grids."
            ),
        },
        "limitations": [
            "Alternative grids use direct logarithmic phases while the E70 control uses recursive ceil multiplication; this intentionally tests common schedule-construction differences rather than one implementation only.",
            "The audit holds equal alpha fixed after E74 established it as the unique robust allocation on the tested simplex.",
            "Robust cost uses worst request count across 25 requested constructions per cadence, not a probability distribution over implementations.",
            "Request-count robustness does not include wall-clock batching, label delay, traffic correlation, or alert-processing latency.",
            "All checkpoint grids must be preregistered; peeking or acting between declared looks invalidates coverage.",
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
        "--maximum-requests",
        type=int,
        default=1_000_000_000,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
