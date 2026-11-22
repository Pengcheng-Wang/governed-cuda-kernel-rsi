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
    weighted_drift_crossing,
)
from analyze_pacing_checkpoint_phase_sensitivity import (
    EPSILON,
    PHASE_DENOMINATOR,
    RATIO,
    ROUNDING_MODES,
    certification_summaries,
    compare_summary,
    geometric_mean,
    load_e72,
    phased_geometric_looks,
    ratio_summary,
    update_worst,
)


DRIFT_CAPS = (
    1.00,
    1.10,
    1.25,
    1.50,
    2.00,
)


def parse_phase_units(
    value: str,
) -> list[int]:
    units = sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )
    if not units:
        raise ValueError(
            "At least one phase is required"
        )
    if any(
        unit < 0
        or unit >= PHASE_DENOMINATOR
        for unit in units
    ):
        raise ValueError(
            "Phase units must be in "
            f"[0, {PHASE_DENOMINATOR - 1}]"
        )
    return units


def grid_variants(
    *,
    maximum_requests: int,
    phase_units: list[int],
) -> list[dict[str, Any]]:
    by_looks: dict[
        tuple[int, ...],
        dict[str, Any],
    ] = {}
    for phase_unit in phase_units:
        phase = (
            phase_unit
            / PHASE_DENOMINATOR
        )
        for rounding in ROUNDING_MODES:
            looks = phased_geometric_looks(
                ratio=RATIO,
                phase=phase,
                rounding=rounding,
                maximum_requests=(
                    maximum_requests
                ),
            )
            key = tuple(looks)
            alias = {
                "phase_units": phase_unit,
                "phase_denominator": (
                    PHASE_DENOMINATOR
                ),
                "phase": phase,
                "rounding": rounding,
            }
            if key not in by_looks:
                by_looks[key] = {
                    "variant": (
                        f"grid_{len(by_looks):02d}"
                    ),
                    "looks": looks,
                    "aliases": [],
                }
            by_looks[key]["aliases"].append(
                alias
            )
    return list(by_looks.values())


def efficient_drift_summaries(
    *,
    baselines: list[dict[str, Any]],
    looks: list[int],
    allocations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    trackers = {
        item["allocation"]: {
            "ratios": [],
            "requests": [],
            "worst": None,
        }
        for item in allocations
    }
    weights_by_metric = {
        metric: sorted(
            {
                float(
                    allocation["weights"][
                        metric
                    ]
                )
                for allocation
                in allocations
            }
        )
        for metric in METRICS
    }
    finite_count = 0
    for baseline in baselines:
        if baseline["status"] != "detected":
            continue
        finite_count += 1
        metric = baseline["breach_metric"]
        crossings = {}
        for weight in weights_by_metric[
            metric
        ]:
            value = weighted_drift_crossing(
                record=baseline,
                looks=looks,
                metric_weight=weight,
            )
            if value is None:
                raise RuntimeError(
                    "Candidate drift did not "
                    "cross"
                )
            crossings[weight] = int(
                value["required_requests"]
            )
        equal_requests = crossings[
            1.0 / 3.0
        ]
        key = [
            baseline["ttft_slo_ms"],
            baseline["family"],
            baseline["error_profile"],
            baseline["risk_budget"],
            baseline[
                "confidence_profile"
            ],
            metric,
        ]
        for allocation in allocations:
            name = allocation["allocation"]
            weight = float(
                allocation["weights"][
                    metric
                ]
            )
            requests = crossings[weight]
            ratio = (
                requests / equal_requests
            )
            tracker = trackers[name]
            tracker["ratios"].append(
                ratio
            )
            tracker["requests"].append(
                requests
            )
            tracker["worst"] = (
                update_worst(
                    tracker["worst"],
                    ratio=ratio,
                    key=key,
                    required_requests=(
                        requests
                    ),
                    equal_requests=(
                        equal_requests
                    ),
                )
            )
    if finite_count != 1890:
        raise RuntimeError(
            "Unexpected finite drift count"
        )
    return {
        name: ratio_summary(
            tracker["ratios"],
            tracker["requests"],
            tracker["worst"],
        )
        for name, tracker
        in trackers.items()
    }


def base_power_of_two_looks(
    maximum_requests: int,
) -> list[int]:
    looks = []
    request = 1
    while request < maximum_requests:
        looks.append(request)
        request *= 2
    looks.append(maximum_requests)
    return looks


def robust_pareto(
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    front = []
    for item in values:
        dominated = any(
            (
                other[
                    "maximum_certification_ratio"
                ]
                <= item[
                    "maximum_certification_ratio"
                ]
                + EPSILON
                and other[
                    "maximum_drift_ratio"
                ]
                <= item[
                    "maximum_drift_ratio"
                ]
                + EPSILON
                and other[
                    "certification_geomean_across_grids"
                ]
                <= item[
                    "certification_geomean_across_grids"
                ]
                + EPSILON
                and other[
                    "drift_geomean_across_grids"
                ]
                <= item[
                    "drift_geomean_across_grids"
                ]
                + EPSILON
                and (
                    other[
                        "maximum_certification_ratio"
                    ]
                    < item[
                        "maximum_certification_ratio"
                    ]
                    - EPSILON
                    or other[
                        "maximum_drift_ratio"
                    ]
                    < item[
                        "maximum_drift_ratio"
                    ]
                    - EPSILON
                    or other[
                        "certification_geomean_across_grids"
                    ]
                    < item[
                        "certification_geomean_across_grids"
                    ]
                    - EPSILON
                    or other[
                        "drift_geomean_across_grids"
                    ]
                    < item[
                        "drift_geomean_across_grids"
                    ]
                    - EPSILON
                )
            )
            for other in values
            if other is not item
        )
        if not dominated:
            front.append(item)
    return sorted(
        front,
        key=lambda item: (
            item[
                "maximum_certification_ratio"
            ],
            item["maximum_drift_ratio"],
            item[
                "certification_geomean_across_grids"
            ],
            item[
                "drift_geomean_across_grids"
            ],
            item["distance_from_equal"],
            item["allocation"],
        ),
    )


def cap_name(cap: float) -> str:
    return (
        f"max_{cap:.2f}"
        .replace(".", "p")
    )


def cap_selections(
    values: list[dict[str, Any]],
) -> dict[str, Any]:
    selections = {}
    for cap in DRIFT_CAPS:
        eligible = [
            item
            for item in values
            if item[
                "maximum_drift_ratio"
            ]
            <= cap + EPSILON
        ]
        if not eligible:
            raise RuntimeError(
                "Equal alpha must satisfy "
                "every drift cap"
            )
        nonworsening = [
            item
            for item in eligible
            if item[
                "maximum_certification_ratio"
            ]
            <= 1.0 + EPSILON
        ]
        selections[cap_name(cap)] = {
            "maximum_drift_ratio": cap,
            "eligible_allocation_count": len(
                eligible
            ),
            "certification_nonworsening_count": (
                len(nonworsening)
            ),
            "certification_first": min(
                eligible,
                key=lambda item: (
                    item[
                        "certification_geomean_across_grids"
                    ],
                    item[
                        "maximum_certification_ratio"
                    ],
                    item[
                        "drift_geomean_across_grids"
                    ],
                    item[
                        "maximum_drift_ratio"
                    ],
                    item[
                        "distance_from_equal"
                    ],
                    item["allocation"],
                ),
            ),
            "certification_nonworsening": (
                None
                if not nonworsening
                else min(
                    nonworsening,
                    key=lambda item: (
                        item[
                            "certification_geomean_across_grids"
                        ],
                        item[
                            "drift_geomean_across_grids"
                        ],
                        item[
                            "distance_from_equal"
                        ],
                        item["allocation"],
                    ),
                )
            ),
        }
    return selections


def run(args: argparse.Namespace) -> dict[str, Any]:
    e68 = load_e68(args.root)
    e72 = load_e72(args.root)
    allocations = allocation_grid()
    phases = parse_phase_units(
        args.phase_units
    )
    variants = grid_variants(
        maximum_requests=(
            args.maximum_requests
        ),
        phase_units=phases,
    )
    grid_records = []
    for variant in variants:
        certification = (
            certification_summaries(
                baselines=e68[
                    "certification_records"
                ],
                looks=variant["looks"],
                allocations=allocations,
            )
        )
        drift = efficient_drift_summaries(
            baselines=e68[
                "drift_detection_records"
            ],
            looks=variant["looks"],
            allocations=allocations,
        )
        grid_records.append(
            {
                "variant": (
                    variant["variant"]
                ),
                "aliases": (
                    variant["aliases"]
                ),
                "look_count": len(
                    variant["looks"]
                ),
                "first_looks": (
                    variant["looks"][:12]
                ),
                "last_looks": (
                    variant["looks"][-4:]
                ),
                "allocations": {
                    allocation[
                        "allocation"
                    ]: {
                        "certification": (
                            certification[
                                allocation[
                                    "allocation"
                                ]
                            ]
                        ),
                        "drift": drift[
                            allocation[
                                "allocation"
                            ]
                        ],
                    }
                    for allocation
                    in allocations
                },
            }
        )
    base_looks = base_power_of_two_looks(
        args.maximum_requests
    )
    base_record = next(
        (
            record
            for record in grid_records
            if (
                record["first_looks"]
                == base_looks[:12]
                and record["last_looks"]
                == base_looks[-4:]
                and record["look_count"]
                == len(base_looks)
            )
        ),
        None,
    )
    if 0 in phases and base_record is None:
        raise RuntimeError(
            "Base power-of-two grid missing"
        )
    if base_record is not None:
        e72_base = {
            item["allocation"]: item
            for item in e72[
                "allocation_summaries"
            ]["geometric_2p00"]
        }
        for allocation in allocations:
            name = allocation["allocation"]
            compare_summary(
                base_record[
                    "allocations"
                ][name]["certification"],
                e72_base[name][
                    "certification"
                ],
                prefix=(
                    f"{name} certification"
                ),
            )
            compare_summary(
                base_record[
                    "allocations"
                ][name]["drift"],
                e72_base[name]["drift"],
                prefix=f"{name} drift",
            )
    robustness = []
    for allocation in allocations:
        name = allocation["allocation"]
        cert_geomeans = [
            record["allocations"][name][
                "certification"
            ][
                "geomean_ratio_vs_equal"
            ]
            for record in grid_records
        ]
        drift_geomeans = [
            record["allocations"][name][
                "drift"
            ][
                "geomean_ratio_vs_equal"
            ]
            for record in grid_records
        ]
        max_certification = max(
            record["allocations"][name][
                "certification"
            ]["ratio_vs_equal"]["maximum"]
            for record in grid_records
        )
        max_drift = max(
            record["allocations"][name][
                "drift"
            ]["ratio_vs_equal"]["maximum"]
            for record in grid_records
        )
        joint_count = sum(
            record["allocations"][name][
                "certification"
            ]["ratio_vs_equal"]["maximum"]
            <= 1.0 + EPSILON
            and record["allocations"][name][
                "drift"
            ]["ratio_vs_equal"]["maximum"]
            <= 1.0 + EPSILON
            for record in grid_records
        )
        worst_certification = max(
            grid_records,
            key=lambda record: (
                record["allocations"][name][
                    "certification"
                ]["ratio_vs_equal"]["maximum"],
                record["variant"],
            ),
        )
        worst_drift = max(
            grid_records,
            key=lambda record: (
                record["allocations"][name][
                    "drift"
                ]["ratio_vs_equal"]["maximum"],
                record["variant"],
            ),
        )
        robustness.append(
            {
                "allocation": name,
                "weights": (
                    allocation["weights"]
                ),
                "distance_from_equal": (
                    allocation[
                        "distance_from_equal"
                    ]
                ),
                "grid_count": len(
                    grid_records
                ),
                "jointly_nonworsening_grid_count": (
                    joint_count
                ),
                "jointly_nonworsening_grid_fraction": (
                    joint_count
                    / len(grid_records)
                ),
                "all_grids_jointly_nonworsening": (
                    joint_count
                    == len(grid_records)
                ),
                "certification_geomean_across_grids": (
                    geometric_mean(
                        cert_geomeans
                    )
                ),
                "drift_geomean_across_grids": (
                    geometric_mean(
                        drift_geomeans
                    )
                ),
                "maximum_certification_ratio": (
                    max_certification
                ),
                "maximum_drift_ratio": (
                    max_drift
                ),
                "maximum_joint_ratio": max(
                    max_certification,
                    max_drift,
                ),
                "worst_certification_grid": {
                    "variant": (
                        worst_certification[
                            "variant"
                        ]
                    ),
                    "aliases": (
                        worst_certification[
                            "aliases"
                        ]
                    ),
                    "summary": (
                        worst_certification[
                            "allocations"
                        ][name][
                            "certification"
                        ]
                    ),
                },
                "worst_drift_grid": {
                    "variant": (
                        worst_drift[
                            "variant"
                        ]
                    ),
                    "aliases": (
                        worst_drift[
                            "aliases"
                        ]
                    ),
                    "summary": (
                        worst_drift[
                            "allocations"
                        ][name]["drift"]
                    ),
                },
            }
        )
    robust_minimax = min(
        robustness,
        key=lambda item: (
            item["maximum_joint_ratio"],
            item["maximum_drift_ratio"],
            item[
                "maximum_certification_ratio"
            ],
            item[
                "certification_geomean_across_grids"
            ]
            * item[
                "drift_geomean_across_grids"
            ],
            item["distance_from_equal"],
            item["allocation"],
        ),
    )
    all_grid_nonworsening = [
        item
        for item in robustness
        if item[
            "all_grids_jointly_nonworsening"
        ]
    ]
    pareto = robust_pareto(
        robustness
    )
    requested_grid_count = (
        len(phases)
        * len(ROUNDING_MODES)
    )
    result = {
        "suite": (
            "pacing_cross_grid_alpha_"
            "robustness"
        ),
        "model": (
            "Evaluate all 172 positive E71 "
            "alpha allocations over every unique "
            "E73 2.00x checkpoint phase/rounding "
            "grid. Select by cross-grid worst "
            "certification/drift ratio rather "
            "than one exact checkpoint lattice."
        ),
        "source": {
            "anytime_baseline": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            ),
            "joint_alpha_frontier": (
                "results/raw/"
                "pacing_joint_alpha_frontier/"
                "pacing_joint_alpha_frontier_full.json"
            ),
            "phase_sensitivity": (
                "results/raw/"
                "pacing_checkpoint_phase_sensitivity/"
                "pacing_checkpoint_phase_sensitivity_full.json"
            ),
        },
        "ratio": RATIO,
        "phase_units": phases,
        "phase_denominator": (
            PHASE_DENOMINATOR
        ),
        "rounding_modes": list(
            ROUNDING_MODES
        ),
        "requested_grid_count": (
            requested_grid_count
        ),
        "unique_grid_count": len(
            grid_records
        ),
        "allocation_count": len(
            allocations
        ),
        "grid_records": grid_records,
        "allocation_robustness": (
            robustness
        ),
        "all_grid_jointly_nonworsening_allocations": (
            all_grid_nonworsening
        ),
        "robust_minimax_allocation": (
            robust_minimax
        ),
        "robust_pareto_front": pareto,
        "by_cross_grid_worst_drift_cap": (
            cap_selections(
                robustness
            )
        ),
        "validation": {
            "requested_grid_count": (
                requested_grid_count
            ),
            "unique_grid_count": len(
                grid_records
            ),
            "allocation_count": len(
                allocations
            ),
            "base_grid_present": (
                base_record is not None
            ),
            "base_grid_variant": (
                None
                if base_record is None
                else base_record["variant"]
            ),
            "base_grid_reproduction": (
                None
                if base_record is None
                else (
                    "All 172 allocations "
                    "reproduce E72 2.00x "
                    "certification and drift "
                    "summaries."
                )
            ),
            "finite_certification_cases_per_grid": (
                1918
            ),
            "finite_drift_cases_per_grid": (
                1890
            ),
            "equal_control": (
                "Equal alpha has ratio one in "
                "every certification and drift "
                "case on every grid."
            ),
        },
        "limitations": [
            "The sweep is exhaustive over E71's positive 5% grid, not the continuous allocation simplex.",
            "Only the 2.00x cadence is phase-perturbed because E72's non-equal exact-grid exception occurred there.",
            "Log phase and integer rounding do not cover batching, delayed labels, wall-clock scheduling, or correlated traffic.",
            "Cross-grid geometric means retain E68-E72 scenario weighting and are descriptive rather than a production utility.",
            "Every exact checkpoint sequence and allocation must still be preregistered; peeking between looks invalidates coverage.",
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
        "--phase-units",
        default=",".join(
            str(value)
            for value
            in range(PHASE_DENOMINATOR)
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
