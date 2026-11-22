from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_pacing_alpha_allocation import (
    allocation_grid,
    evaluate_allocation,
    load_e68,
    precompute_metric_looks,
    summarize,
    weighted_drift_crossing,
)
from analyze_pacing_cadence_robustness import (
    schedule_grids,
)
from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
)


DEFAULT_LOOK_OVERHEADS = (
    0,
    1,
    2,
    3,
    4,
    5,
    10,
    20,
    43,
    44,
    45,
    50,
    100,
    200,
    344,
    345,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
)


def load_e75(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_cadence_robustness"
        / "pacing_cadence_robustness_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def parse_overheads(
    value: str,
) -> list[int]:
    overheads = sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )
    if not overheads:
        raise ValueError(
            "At least one overhead is required"
        )
    if overheads[0] != 0:
        raise ValueError(
            "Overhead sweep must include zero"
        )
    if any(
        overhead < 0
        for overhead in overheads
    ):
        raise ValueError(
            "Overheads must be nonnegative"
        )
    return overheads


def profile_summary(
    profiles: list[dict[str, Any]],
    overhead: int,
) -> dict[str, Any]:
    costs = [
        profile["requests"]
        + overhead
        * profile["look_index"]
        for profile in profiles
    ]
    maximum = max(costs)
    index = costs.index(maximum)
    selected = profiles[index]
    return {
        "composite_cost": maximum,
        "requests": selected["requests"],
        "look_index": (
            selected["look_index"]
        ),
        "look_overhead_cost": (
            overhead
            * selected["look_index"]
        ),
        "overhead_fraction": (
            0.0
            if maximum == 0
            else (
                overhead
                * selected["look_index"]
                / maximum
            )
        ),
        "grid": selected["grid"],
        "grid_aliases": (
            selected["grid_aliases"]
        ),
        "is_e75_max_request_grid": (
            selected[
                "is_e75_max_request_grid"
            ]
        ),
    }


def fixed_schedule_regret(
    case_costs: list[
        dict[str, int]
    ],
    schedules: list[str],
) -> dict[str, Any]:
    result = {}
    for schedule in schedules:
        ratios = []
        for costs in case_costs:
            oracle = min(costs.values())
            ratios.append(
                costs[schedule] / oracle
            )
        result[schedule] = {
            "ratio_vs_per_case_oracle": (
                summarize(ratios)
            ),
            "geomean_ratio_vs_per_case_oracle": (
                math.exp(
                    statistics.mean(
                        math.log(value)
                        for value in ratios
                    )
                )
            ),
        }
    return result


def overhead_summary(
    *,
    cases: list[dict[str, Any]],
    overhead: int,
    schedules: list[str],
    domain: str,
) -> dict[str, Any]:
    membership = Counter()
    unique = Counter()
    by_slo: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)
    by_group: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)
    retention = 0
    oracle_ratios = []
    case_costs = []
    schedule_composite_vs_request = {
        schedule: []
        for schedule in schedules
    }
    schedule_overhead_fractions = {
        schedule: []
        for schedule in schedules
    }
    for case in cases:
        summaries = {
            schedule: profile_summary(
                case["profiles"][schedule],
                overhead,
            )
            for schedule in schedules
        }
        costs = {
            schedule: summary[
                "composite_cost"
            ]
            for schedule, summary
            in summaries.items()
        }
        case_costs.append(costs)
        minimum = min(costs.values())
        winners = [
            schedule
            for schedule, cost
            in costs.items()
            if cost == minimum
        ]
        membership.update(winners)
        if len(winners) == 1:
            unique[winners[0]] += 1
        retention += bool(
            set(winners)
            & set(
                case[
                    "request_only_winners"
                ]
            )
        )
        oracle_ratios.append(
            minimum
            / case[
                "request_only_oracle_cost"
            ]
        )
        slo = str(case["key"][0])
        group = (
            str(case["key"][-1])
            if domain == "drift"
            else str(case["key"][1])
        )
        for winner in winners:
            by_slo[slo][winner] += 1
            by_group[group][winner] += 1
        for schedule, summary in (
            summaries.items()
        ):
            request_only = max(
                profile["requests"]
                for profile
                in case["profiles"][
                    schedule
                ]
            )
            schedule_composite_vs_request[
                schedule
            ].append(
                summary[
                    "composite_cost"
                ]
                / request_only
            )
            schedule_overhead_fractions[
                schedule
            ].append(
                summary[
                    "overhead_fraction"
                ]
            )
    regret = fixed_schedule_regret(
        case_costs,
        schedules,
    )
    minimax = min(
        schedules,
        key=lambda schedule: (
            regret[schedule][
                "ratio_vs_per_case_oracle"
            ]["maximum"],
            regret[schedule][
                "ratio_vs_per_case_oracle"
            ]["p90"],
            regret[schedule][
                "geomean_ratio_vs_per_case_oracle"
            ],
            schedule,
        ),
    )
    return {
        "look_overhead_request_equivalents": (
            overhead
        ),
        "case_count": len(cases),
        "winner_membership_counts": {
            schedule: membership[schedule]
            for schedule in schedules
        },
        "unique_winner_counts": {
            schedule: unique[schedule]
            for schedule in schedules
        },
        "request_only_winner_retention_fraction": (
            retention / len(cases)
        ),
        "per_case_oracle_cost_ratio_vs_request_only": (
            summarize(oracle_ratios)
        ),
        "fixed_schedule_regret": regret,
        "global_minimax_schedule": minimax,
        "winner_counts_by_slo": {
            group: {
                schedule: counts[
                    schedule
                ]
                for schedule in schedules
            }
            for group, counts
            in sorted(by_slo.items())
        },
        "winner_counts_by_group": {
            group: {
                schedule: counts[
                    schedule
                ]
                for schedule in schedules
            }
            for group, counts
            in sorted(by_group.items())
        },
        "schedule_composite_ratio_vs_request_only": {
            schedule: summarize(values)
            for schedule, values
            in schedule_composite_vs_request.items()
        },
        "schedule_overhead_fraction": {
            schedule: summarize(values)
            for schedule, values
            in schedule_overhead_fractions.items()
        },
    }


def build_certification_profiles(
    *,
    e68: dict[str, Any],
    e75: dict[str, Any],
    grids: dict[
        str,
        list[dict[str, Any]],
    ],
    equal: dict[str, Any],
    schedules: list[str],
) -> tuple[
    list[dict[str, Any]],
    int,
]:
    e75_records = {
        tuple(record["key"]): record
        for record in e75[
            "certification_records"
        ]
    }
    cases = []
    reproduction_count = 0
    for baseline in e68[
        "certification_records"
    ]:
        if baseline["status"] != (
            "certified"
        ):
            continue
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
                "calibration_scenario"
            ],
            baseline[
                "confidence_profile"
            ],
        )
        e75_record = e75_records[key]
        profiles = {}
        for schedule in schedules:
            values = []
            e75_maximum = int(
                e75_record["schedules"][
                    schedule
                ]["maximum_requests"]
            )
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
                requests = int(
                    evaluated[
                        "required_requests"
                    ]
                )
                values.append(
                    {
                        "grid": grid["grid"],
                        "grid_aliases": (
                            grid["aliases"]
                        ),
                        "requests": requests,
                        "look_index": int(
                            evaluated[
                                "look_index"
                            ]
                        ),
                        "is_e75_max_request_grid": (
                            requests
                            == e75_maximum
                        ),
                    }
                )
            if (
                max(
                    value["requests"]
                    for value in values
                )
                != e75_maximum
            ):
                raise RuntimeError(
                    "Certification did not "
                    "reproduce E75"
                )
            reproduction_count += 1
            profiles[schedule] = values
        cases.append(
            {
                "key": list(key),
                "profiles": profiles,
                "request_only_winners": (
                    e75_record[
                        "robust_winner_schedules"
                    ]
                ),
                "request_only_oracle_cost": (
                    e75_record[
                        "robust_minimum_requests"
                    ]
                ),
            }
        )
    return cases, reproduction_count


def build_drift_profiles(
    *,
    e68: dict[str, Any],
    e75: dict[str, Any],
    grids: dict[
        str,
        list[dict[str, Any]],
    ],
    schedules: list[str],
) -> tuple[
    list[dict[str, Any]],
    int,
]:
    e75_records = {
        tuple(record["key"]): record
        for record in e75[
            "drift_records"
        ]
    }
    cases = []
    reproduction_count = 0
    for baseline in e68[
        "drift_detection_records"
    ]:
        if baseline["status"] != "detected":
            continue
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
        )
        e75_record = e75_records[key]
        profiles = {}
        for schedule in schedules:
            values = []
            e75_maximum = int(
                e75_record["schedules"][
                    schedule
                ]["maximum_requests"]
            )
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
                requests = int(
                    crossing[
                        "required_requests"
                    ]
                )
                values.append(
                    {
                        "grid": grid["grid"],
                        "grid_aliases": (
                            grid["aliases"]
                        ),
                        "requests": requests,
                        "look_index": int(
                            crossing[
                                "look_index"
                            ]
                        ),
                        "is_e75_max_request_grid": (
                            requests
                            == e75_maximum
                        ),
                    }
                )
            if (
                max(
                    value["requests"]
                    for value in values
                )
                != e75_maximum
            ):
                raise RuntimeError(
                    "Drift did not reproduce "
                    "E75"
                )
            reproduction_count += 1
            profiles[schedule] = values
        cases.append(
            {
                "key": list(key),
                "profiles": profiles,
                "request_only_winners": (
                    e75_record[
                        "robust_winner_schedules"
                    ]
                ),
                "request_only_oracle_cost": (
                    e75_record[
                        "robust_minimum_requests"
                    ]
                ),
            }
        )
    return cases, reproduction_count


def transition_summary(
    overheads: list[int],
    summaries: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    minimax = [
        summaries[str(overhead)][
            "global_minimax_schedule"
        ]
        for overhead in overheads
    ]
    segments = []
    start = 0
    for index in range(
        1,
        len(overheads) + 1,
    ):
        if (
            index == len(overheads)
            or minimax[index]
            != minimax[start]
        ):
            segments.append(
                {
                    "schedule": (
                        minimax[start]
                    ),
                    "first_overhead": (
                        overheads[start]
                    ),
                    "last_overhead": (
                        overheads[
                            index - 1
                        ]
                    ),
                }
            )
            start = index
    return {
        "segments": segments,
        "first_non_1p10_overhead": next(
            (
                overhead
                for overhead, schedule
                in zip(
                    overheads,
                    minimax,
                    strict=True,
                )
                if schedule
                != "geometric_1p10"
            ),
            None,
        ),
    }


def minimax_schedule_for_overhead(
    *,
    cases: list[dict[str, Any]],
    overhead: int,
    schedules: list[str],
) -> str:
    case_costs = []
    for case in cases:
        case_costs.append(
            {
                schedule: profile_summary(
                    case["profiles"][
                        schedule
                    ],
                    overhead,
                )["composite_cost"]
                for schedule in schedules
            }
        )
    regret = fixed_schedule_regret(
        case_costs,
        schedules,
    )
    return min(
        schedules,
        key=lambda schedule: (
            regret[schedule][
                "ratio_vs_per_case_oracle"
            ]["maximum"],
            regret[schedule][
                "ratio_vs_per_case_oracle"
            ]["p90"],
            regret[schedule][
                "geomean_ratio_vs_per_case_oracle"
            ],
            schedule,
        ),
    )


def exact_transition_scan(
    *,
    cases: list[dict[str, Any]],
    maximum_overhead: int,
    schedules: list[str],
) -> dict[str, Any]:
    overheads = list(
        range(maximum_overhead + 1)
    )
    minimax = [
        minimax_schedule_for_overhead(
            cases=cases,
            overhead=overhead,
            schedules=schedules,
        )
        for overhead in overheads
    ]
    segments = []
    start = 0
    for index in range(
        1,
        len(overheads) + 1,
    ):
        if (
            index == len(overheads)
            or minimax[index]
            != minimax[start]
        ):
            segments.append(
                {
                    "schedule": (
                        minimax[start]
                    ),
                    "first_overhead": (
                        overheads[start]
                    ),
                    "last_overhead": (
                        overheads[
                            index - 1
                        ]
                    ),
                }
            )
            start = index
    return {
        "maximum_scanned_overhead": (
            maximum_overhead
        ),
        "segments": segments,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e68 = load_e68(args.root)
    e75 = load_e75(args.root)
    overheads = parse_overheads(
        args.look_overheads
    )
    schedules = list(
        LOOK_SCHEDULES
    )
    equal = next(
        allocation
        for allocation in allocation_grid()
        if allocation["allocation"]
        == "equal"
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
        for schedule in schedules
    }
    certification_cases, cert_reproduction = (
        build_certification_profiles(
            e68=e68,
            e75=e75,
            grids=grids,
            equal=equal,
            schedules=schedules,
        )
    )
    drift_cases, drift_reproduction = (
        build_drift_profiles(
            e68=e68,
            e75=e75,
            grids=grids,
            schedules=schedules,
        )
    )
    certification_summaries = {
        str(overhead): overhead_summary(
            cases=certification_cases,
            overhead=overhead,
            schedules=schedules,
            domain="certification",
        )
        for overhead in overheads
    }
    drift_summaries = {
        str(overhead): overhead_summary(
            cases=drift_cases,
            overhead=overhead,
            schedules=schedules,
            domain="drift",
        )
        for overhead in overheads
    }
    certification_exact_scan = (
        exact_transition_scan(
            cases=certification_cases,
            maximum_overhead=(
                args.transition_scan_maximum
            ),
            schedules=schedules,
        )
    )
    drift_exact_scan = (
        exact_transition_scan(
            cases=drift_cases,
            maximum_overhead=(
                args.transition_scan_maximum
            ),
            schedules=schedules,
        )
    )
    zero_cert = certification_summaries[
        "0"
    ]
    zero_drift = drift_summaries["0"]
    if (
        zero_cert[
            "unique_winner_counts"
        ]
        != e75[
            "certification_winners"
        ][
            "robust_unique_winner_counts"
        ]
    ):
        raise RuntimeError(
            "Zero-overhead certification did "
            "not reproduce E75"
        )
    if (
        zero_drift[
            "unique_winner_counts"
        ]
        != e75["drift_winners"][
            "robust_unique_winner_counts"
        ]
    ):
        raise RuntimeError(
            "Zero-overhead drift did not "
            "reproduce E75"
        )
    for summaries in (
        certification_summaries,
        drift_summaries,
    ):
        previous = None
        for overhead in overheads:
            current = summaries[
                str(overhead)
            ][
                "per_case_oracle_"
                "cost_ratio_vs_request_only"
            ]["median"]
            if (
                previous is not None
                and current
                < previous - 1e-12
            ):
                raise RuntimeError(
                    "Composite oracle cost was "
                    "not monotone"
                )
            previous = current
    result = {
        "suite": (
            "pacing_look_cost_frontier"
        ),
        "model": (
            "Add a synchronous request-"
            "equivalent cost to every statistical "
            "checkpoint. For each cadence and "
            "case, robust composite cost is the "
            "maximum of requests plus overhead "
            "times look index across E75's valid "
            "grid constructions."
        ),
        "source": {
            "anytime_baseline": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            ),
            "cadence_robustness": (
                "results/raw/"
                "pacing_cadence_robustness/"
                "pacing_cadence_robustness_full.json"
            ),
        },
        "look_overheads_request_equivalents": (
            overheads
        ),
        "schedules": {
            schedule: {
                "ratio": LOOK_SCHEDULES[
                    schedule
                ],
                "unique_grid_count": len(
                    grids[schedule]
                ),
            }
            for schedule in schedules
        },
        "certification_overhead_summaries": (
            certification_summaries
        ),
        "drift_overhead_summaries": (
            drift_summaries
        ),
        "certification_minimax_transitions": (
            transition_summary(
                overheads,
                certification_summaries,
            )
        ),
        "drift_minimax_transitions": (
            transition_summary(
                overheads,
                drift_summaries,
            )
        ),
        "certification_exact_transition_scan": (
            certification_exact_scan
        ),
        "drift_exact_transition_scan": (
            drift_exact_scan
        ),
        "validation": {
            "certification_case_count": len(
                certification_cases
            ),
            "drift_case_count": len(
                drift_cases
            ),
            "certification_schedule_reproduction_count": (
                cert_reproduction
            ),
            "expected_certification_schedule_reproduction_count": (
                1918 * len(schedules)
            ),
            "drift_schedule_reproduction_count": (
                drift_reproduction
            ),
            "expected_drift_schedule_reproduction_count": (
                1890 * len(schedules)
            ),
            "zero_overhead_reproduction": (
                "Unique robust winner counts "
                "match E75 exactly."
            ),
            "monotonicity": (
                "Median per-case robust oracle "
                "composite cost is nondecreasing "
                "with per-look overhead."
            ),
            "exact_transition_scan": (
                "Global minimax cadence is "
                "evaluated at every integer "
                "request-equivalent overhead "
                f"from zero through "
                f"{args.transition_scan_maximum}."
            ),
        },
        "limitations": [
            "Look overhead is expressed in request-equivalent units. Converting CPU time, label-query cost, or synchronous blocking into this unit is deployment-specific.",
            "The model assumes each look's overhead is cumulative and synchronous. Asynchronous or batched monitoring can have lower request-path cost.",
            "Composite cost adds request count and monitoring overhead linearly; it is a planning frontier rather than a wall-clock simulator.",
            "The audit holds equal alpha fixed and reuses E75's valid checkpoint constructions.",
            "All earlier independence, traffic-weighting, delayed-label, and preregistration limitations remain.",
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
        "--look-overheads",
        default=",".join(
            str(value)
            for value
            in DEFAULT_LOOK_OVERHEADS
        ),
    )
    parser.add_argument(
        "--maximum-requests",
        type=int,
        default=1_000_000_000,
    )
    parser.add_argument(
        "--transition-scan-maximum",
        type=int,
        default=500,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
