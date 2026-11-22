from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    policy_inventory,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    load_e83,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
    prevalence_profiles,
    profile_metrics,
    winner,
)
from analyze_pacing_coalescing_order_prevalence_certification import (
    CONFIDENCE_LEVELS,
    COVERAGE_TARGETS,
    PREVALENCE_DENOMINATOR,
    REPRESENTATIVE_ACCURACIES,
    SAMPLE_BUDGETS,
    TOLERANCE,
    coverage_thresholds,
    pairwise_boundary_distance_l1,
    quantile,
    radius_summary,
    required_samples,
    validate_sample_bound,
)


REGRET_TOLERANCES_PPM = (
    0,
    1,
    5,
    10,
    25,
    50,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_e89(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_prevalence"
        / "pacing_coalescing_order_prevalence_full.json"
    )


def load_e90(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_prevalence_certification"
        / "pacing_coalescing_order_prevalence_certification_full.json"
    )


def tolerance_key(
    tolerance_ppm: int,
) -> str:
    return str(tolerance_ppm)


def radius_kind(
    radius: float,
) -> str:
    if math.isinf(radius):
        return "infinite"
    if radius <= TOLERANCE:
        return "zero"
    return "finite"


def certification_record(
    *,
    policy: str,
    radius: float,
    acceptable_policy_count: int,
    category_count: int,
) -> dict[str, Any]:
    kind = radius_kind(radius)
    if kind == "zero":
        radius = 0.0
    samples = {
        f"{confidence:.2f}": (
            required_samples(
                radius=radius,
                confidence=confidence,
                category_count=(
                    category_count
                ),
            )
        )
        for confidence
        in CONFIDENCE_LEVELS
    }
    for confidence in CONFIDENCE_LEVELS:
        key = f"{confidence:.2f}"
        validate_sample_bound(
            radius=radius,
            confidence=confidence,
            category_count=(
                category_count
            ),
            samples=samples[key],
        )
    return {
        "policy": policy,
        "radius_l1": radius,
        "radius_kind": kind,
        "required_samples": samples,
        "acceptable_policy_count": (
            acceptable_policy_count
        ),
    }


def regret_summary(
    *,
    records: list[dict[str, Any]],
    profile_count: int,
) -> dict[str, Any]:
    result = radius_summary(
        records=records,
        profile_count=profile_count,
    )
    acceptable_counts = [
        record[
            "acceptable_policy_count"
        ]
        for record in records
    ]
    result[
        "acceptable_policy_count"
    ] = {
        "minimum": min(
            acceptable_counts
        ),
        "p50": quantile(
            acceptable_counts,
            0.50,
        ),
        "p90": quantile(
            acceptable_counts,
            0.90,
        ),
        "maximum": max(
            acceptable_counts
        ),
        "mean": (
            sum(acceptable_counts)
            / len(acceptable_counts)
        ),
    }
    return result


def radius_strictly_larger(
    candidate: float,
    baseline: float,
) -> bool:
    if math.isinf(candidate):
        return not math.isinf(
            baseline
        )
    if math.isinf(baseline):
        return False
    return (
        candidate
        > baseline + TOLERANCE
    )


def comparison_to_identity(
    *,
    identity_records: list[
        dict[str, Any]
    ],
    regret_records: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    if (
        len(identity_records)
        != len(regret_records)
    ):
        raise RuntimeError(
            "Comparison record count changed"
        )
    profile_count = len(
        identity_records
    )
    resolved = sum(
        identity["radius_kind"] == "zero"
        and regret["radius_kind"]
        != "zero"
        for identity, regret
        in zip(
            identity_records,
            regret_records,
        )
    )
    improved = sum(
        radius_strictly_larger(
            regret["radius_l1"],
            identity["radius_l1"],
        )
        for identity, regret
        in zip(
            identity_records,
            regret_records,
        )
    )
    confidence = {}
    for level in CONFIDENCE_LEVELS:
        key = f"{level:.2f}"
        budget_gains = {}
        for budget in SAMPLE_BUDGETS:
            identity_covered = sum(
                record[
                    "required_samples"
                ][key]
                is not None
                and record[
                    "required_samples"
                ][key]
                <= budget
                for record
                in identity_records
            )
            regret_covered = sum(
                record[
                    "required_samples"
                ][key]
                is not None
                and record[
                    "required_samples"
                ][key]
                <= budget
                for record
                in regret_records
            )
            budget_gains[
                str(budget)
            ] = {
                "count": (
                    regret_covered
                    - identity_covered
                ),
                "fraction": (
                    (
                        regret_covered
                        - identity_covered
                    )
                    / profile_count
                ),
            }
        paired_ratios = []
        for identity, regret in zip(
            identity_records,
            regret_records,
        ):
            identity_samples = identity[
                "required_samples"
            ][key]
            regret_samples = regret[
                "required_samples"
            ][key]
            if (
                identity_samples
                in (None, 0)
                or regret_samples
                in (None, 0)
            ):
                continue
            paired_ratios.append(
                regret_samples
                / identity_samples
            )
        confidence[key] = {
            "budget_coverage_gain": (
                budget_gains
            ),
            "common_finite_sample_ratio": {
                "count": len(
                    paired_ratios
                ),
                "p50": quantile(
                    paired_ratios,
                    0.50,
                ),
                "p90": quantile(
                    paired_ratios,
                    0.90,
                ),
                "maximum": (
                    max(paired_ratios)
                    if paired_ratios
                    else None
                ),
            },
        }
    return {
        "identity_zero_resolved_count": (
            resolved
        ),
        "identity_zero_resolved_fraction": (
            resolved / profile_count
        ),
        "strict_radius_improvement_count": (
            improved
        ),
        "strict_radius_improvement_fraction": (
            improved / profile_count
        ),
        "confidence": confidence,
    }


def normalized_sample_count(
    value: int | None,
) -> float:
    if value is None:
        return math.inf
    return float(value)


def validate_monotonicity(
    *,
    records_by_tolerance: dict[
        str,
        list[dict[str, Any]],
    ],
) -> None:
    previous = None
    for tolerance_ppm in (
        REGRET_TOLERANCES_PPM
    ):
        key = tolerance_key(
            tolerance_ppm
        )
        current = records_by_tolerance[
            key
        ]
        if previous is not None:
            for before, after in zip(
                previous,
                current,
            ):
                before_radius = before[
                    "radius_l1"
                ]
                after_radius = after[
                    "radius_l1"
                ]
                if (
                    not math.isinf(
                        after_radius
                    )
                    and (
                        math.isinf(
                            before_radius
                        )
                        or after_radius
                        + TOLERANCE
                        < before_radius
                    )
                ):
                    raise RuntimeError(
                        "Regret radius is not "
                        "monotone in tolerance"
                    )
                for confidence in (
                    CONFIDENCE_LEVELS
                ):
                    confidence_key = (
                        f"{confidence:.2f}"
                    )
                    before_samples = (
                        normalized_sample_count(
                            before[
                                "required_samples"
                            ][confidence_key]
                        )
                    )
                    after_samples = (
                        normalized_sample_count(
                            after[
                                "required_samples"
                            ][confidence_key]
                        )
                    )
                    if (
                        after_samples
                        > before_samples
                    ):
                        raise RuntimeError(
                            "Sample requirement "
                            "increased with tolerance"
                        )
        previous = current


def tolerance_coverage_thresholds(
    *,
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for tolerance_ppm in (
        REGRET_TOLERANCES_PPM
    ):
        key = tolerance_key(
            tolerance_ppm
        )
        result[key] = (
            coverage_thresholds(
                grid=[
                    {
                        "accuracy": row[
                            "accuracy"
                        ],
                        "summary": row[
                            "tolerances"
                        ][key],
                    }
                    for row in grid
                ]
            )
        )
    return result


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e90 = load_e90(args.root)
    drift = e83["headline"][
        "drift_by_marginal_fraction_and_order"
    ]
    costs = sorted(
        drift,
        key=float,
    )
    inventory = policy_inventory(
        root=args.root,
        costs=costs,
    )
    profiles = prevalence_profiles(
        denominator=(
            PREVALENCE_DENOMINATOR
        ),
        cost_count=len(costs),
    )
    if len(profiles) != 3003:
        raise RuntimeError(
            "Unexpected prevalence profile count"
        )
    accuracy_grid = [
        index / 100.0
        for index in range(101)
    ]
    kernels = {}
    e90_reproduced = True
    all_zero_regret_positive = True
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        e89_grid = {
            row["accuracy"]: row
            for row in e89["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        e90_grid = {
            row["accuracy"]: row
            for row in e90["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        grid = []
        representatives = {}
        for accuracy in accuracy_grid:
            per_policy_costs = {
                name: per_cost_metrics(
                    drift=drift,
                    costs=costs,
                    mapping=policy[
                        "mapping"
                    ],
                    accuracy=accuracy,
                    offset_weights=kernel[
                        "offset_weights"
                    ],
                )
                for name, policy
                in inventory.items()
            }
            comparison_counts = [
                row[
                    "comparison_count"
                ]
                for rows
                in per_policy_costs.values()
                for row in rows
            ]
            if (
                max(comparison_counts)
                - min(comparison_counts)
                > 1e-6
            ):
                raise RuntimeError(
                    "Prevalence regression risk "
                    "is not linear"
                )
            identity_records = []
            records_by_tolerance = {
                tolerance_key(
                    tolerance_ppm
                ): []
                for tolerance_ppm
                in REGRET_TOLERANCES_PPM
            }
            winner_counts = {
                name: 0
                for name in inventory
            }
            for profile in profiles:
                metrics = {
                    name: profile_metrics(
                        per_cost=rows,
                        weights=profile[
                            "weights"
                        ],
                    )
                    for name, rows
                    in per_policy_costs.items()
                }
                selected = winner(
                    metrics=metrics,
                    inventory=inventory,
                )
                winner_counts[selected] += 1
                selected_rows = (
                    per_policy_costs[
                        selected
                    ]
                )
                selected_risk = [
                    row[
                        "regression_fraction"
                    ]
                    for row
                    in selected_rows
                ]
                selected_profile_risk = (
                    metrics[selected][
                        "regression_fraction"
                    ]
                )
                competitors = []
                for competitor, rows in (
                    per_policy_costs.items()
                ):
                    if competitor == selected:
                        continue
                    differences = [
                        row[
                            "regression_fraction"
                        ]
                        - selected_risk[index]
                        for index, row
                        in enumerate(rows)
                    ]
                    regret = sum(
                        weight
                        * difference
                        for weight, difference
                        in zip(
                            profile[
                                "weights"
                            ],
                            differences,
                        )
                    )
                    metric_regret = (
                        metrics[competitor][
                            "regression_fraction"
                        ]
                        - selected_profile_risk
                    )
                    if (
                        abs(
                            regret
                            - metric_regret
                        )
                        > 1e-12
                    ):
                        raise RuntimeError(
                            "Linear prevalence "
                            "regret changed"
                        )
                    identity_distance = (
                        pairwise_boundary_distance_l1(
                            prevalence=profile[
                                "weights"
                            ],
                            differences=(
                                differences
                            ),
                        )
                    )
                    safe_distance = (
                        pairwise_boundary_distance_l1(
                            prevalence=profile[
                                "weights"
                            ],
                            differences=(
                                differences
                            ),
                            include_ties=True,
                        )
                    )
                    competitors.append(
                        {
                            "regret": max(
                                0.0,
                                regret,
                            ),
                            "identity_distance": (
                                identity_distance
                            ),
                            "safe_distance": (
                                safe_distance
                            ),
                        }
                    )
                identity_radius = min(
                    competitor[
                        "identity_distance"
                    ]
                    for competitor
                    in competitors
                )
                identity_records.append(
                    certification_record(
                        policy=selected,
                        radius=(
                            identity_radius
                        ),
                        acceptable_policy_count=1,
                        category_count=(
                            len(costs)
                        ),
                    )
                )
                previous_radius = None
                for tolerance_ppm in (
                    REGRET_TOLERANCES_PPM
                ):
                    key = tolerance_key(
                        tolerance_ppm
                    )
                    tolerance = (
                        tolerance_ppm
                        / 1_000_000.0
                    )
                    unacceptable = [
                        competitor
                        for competitor
                        in competitors
                        if competitor[
                            "regret"
                        ]
                        > (
                            tolerance
                            + TOLERANCE
                        )
                    ]
                    radius = min(
                        (
                            competitor[
                                "safe_distance"
                            ]
                            for competitor
                            in unacceptable
                        ),
                        default=math.inf,
                    )
                    if (
                        previous_radius
                        is not None
                        and
                        not math.isinf(
                            radius
                        )
                        and (
                            math.isinf(
                                previous_radius
                            )
                            or radius
                            + TOLERANCE
                            < previous_radius
                        )
                    ):
                        raise RuntimeError(
                            "Regret tolerance "
                            "reduced radius"
                        )
                    previous_radius = radius
                    records_by_tolerance[
                        key
                    ].append(
                        certification_record(
                            policy=selected,
                            radius=radius,
                            acceptable_policy_count=(
                                len(
                                    inventory
                                )
                                - len(
                                    unacceptable
                                )
                            ),
                            category_count=(
                                len(costs)
                            ),
                        )
                    )
            if (
                winner_counts
                != e89_grid[accuracy][
                    "winner_counts"
                ]
            ):
                raise RuntimeError(
                    "E89 winner counts were "
                    "not reproduced"
                )
            identity_summary = (
                radius_summary(
                    records=(
                        identity_records
                    ),
                    profile_count=len(
                        profiles
                    ),
                )
            )
            if (
                identity_summary
                != e90_grid[accuracy][
                    "summary"
                ]
            ):
                e90_reproduced = False
                raise RuntimeError(
                    "E90 identity summary "
                    "was not reproduced"
                )
            validate_monotonicity(
                records_by_tolerance=(
                    records_by_tolerance
                )
            )
            tolerance_summaries = {}
            representative_rows = {}
            for tolerance_ppm in (
                REGRET_TOLERANCES_PPM
            ):
                key = tolerance_key(
                    tolerance_ppm
                )
                records = (
                    records_by_tolerance[
                        key
                    ]
                )
                summary = regret_summary(
                    records=records,
                    profile_count=len(
                        profiles
                    ),
                )
                tolerance_summaries[
                    key
                ] = summary
                if tolerance_ppm == 0:
                    if (
                        summary[
                            "zero_radius_count"
                        ]
                        != 0
                    ):
                        all_zero_regret_positive = (
                            False
                        )
                if accuracy in (
                    REPRESENTATIVE_ACCURACIES
                ):
                    representative_rows[
                        key
                    ] = {
                        "summary": summary,
                        "comparison_to_identity": (
                            comparison_to_identity(
                                identity_records=(
                                    identity_records
                                ),
                                regret_records=(
                                    records
                                ),
                            )
                        ),
                    }
            row = {
                "accuracy": accuracy,
                "tolerances": (
                    tolerance_summaries
                ),
            }
            grid.append(row)
            if accuracy in (
                REPRESENTATIVE_ACCURACIES
            ):
                representatives[
                    f"{accuracy:g}"
                ] = {
                    "accuracy": accuracy,
                    "identity_summary": (
                        identity_summary
                    ),
                    "tolerances": (
                        representative_rows
                    ),
                }
        kernels[kernel_name] = {
            "description": kernel[
                "description"
            ],
            "accuracy_grid": grid,
            "representative_accuracies": (
                representatives
            ),
            "coverage_thresholds": (
                tolerance_coverage_thresholds(
                    grid=grid
                )
            ),
        }
    if not all_zero_regret_positive:
        raise RuntimeError(
            "Zero-regret certification "
            "retained a zero radius"
        )
    result = {
        "suite": (
            "pacing_coalescing_order_regret_certification"
        ),
        "model": (
            "Select a policy from empirical "
            "six-cell prevalence while "
            "guaranteeing that its true weighted "
            "regression is within a declared "
            "absolute tolerance of the true "
            "prevalence winner. The pairwise-safe "
            "radius is the exact minimum L1 "
            "distance at which any policy whose "
            "true regret exceeds the tolerance can "
            "tie the true winner."
        ),
        "guarantee": {
            "event": (
                "empirical winner true regret <= "
                "declared tolerance"
            ),
            "radius": (
                "minimum pairwise L1 distance to "
                "an unacceptable competitor tying "
                "the true winner"
            ),
            "conservatism": (
                "A competitor tying the true "
                "winner need not become the global "
                "empirical winner, so the radius "
                "is safe but can be smaller than "
                "the exact bad-winner-region "
                "distance."
            ),
        },
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            ),
            "prevalence_audit": (
                "results/raw/"
                "pacing_coalescing_order_prevalence/"
                "pacing_coalescing_order_prevalence_full.json"
            ),
            "identity_certification": (
                "results/raw/"
                "pacing_coalescing_order_prevalence_certification/"
                "pacing_coalescing_order_prevalence_certification_full.json"
            ),
        },
        "confidence_bound": {
            "name": (
                "multinomial L1 concentration"
            ),
            "formula": (
                "P(||p_hat-p||_1 >= eps) <= "
                "(2^k-2) * exp(-n*eps^2/2)"
            ),
            "category_count": len(costs),
            "confidence_levels": list(
                CONFIDENCE_LEVELS
            ),
            "sample_budgets": list(
                SAMPLE_BUDGETS
            ),
            "coverage_targets": list(
                COVERAGE_TARGETS
            ),
        },
        "regret_tolerances": [
            {
                "key": tolerance_key(
                    tolerance_ppm
                ),
                "excess_regressions_per_million": (
                    tolerance_ppm
                ),
                "regression_fraction": (
                    tolerance_ppm
                    / 1_000_000.0
                ),
            }
            for tolerance_ppm
            in REGRET_TOLERANCES_PPM
        ],
        "costs": costs,
        "prevalence_grid": {
            "denominator": (
                PREVALENCE_DENOMINATOR
            ),
            "profile_count": len(
                profiles
            ),
        },
        "accuracy_grid": accuracy_grid,
        "policy_inventory": {
            name: {
                "configuration_segment_count": (
                    policy[
                        "configuration_segment_count"
                    ]
                ),
                "segments": policy[
                    "segments"
                ],
            }
            for name, policy
            in inventory.items()
        },
        "kernels": kernels,
        "headline": {
            kernel_name: {
                "coverage_thresholds": values[
                    "coverage_thresholds"
                ],
                "representative_accuracies": (
                    values[
                        "representative_accuracies"
                    ]
                ),
            }
            for kernel_name, values
            in kernels.items()
        },
        "validation": {
            "kernel_count": len(
                kernels
            ),
            "accuracy_grid_count": len(
                accuracy_grid
            ),
            "prevalence_profile_count": (
                len(profiles)
            ),
            "regret_tolerance_count": len(
                REGRET_TOLERANCES_PPM
            ),
            "e89_winner_counts_reproduced": (
                True
            ),
            "e90_identity_summaries_reproduced": (
                e90_reproduced
            ),
            "per_cost_comparison_counts_equal": (
                True
            ),
            "pairwise_radii_monotone": True,
            "sample_requirements_monotone": (
                True
            ),
            "zero_regret_uses_tie_safe_boundaries": (
                True
            ),
            "zero_regret_zero_radius_count": 0,
            "sample_bound_inversion_valid": (
                True
            ),
        },
        "limitations": [
            "The pairwise-safe radius is conservative because an unacceptable competitor can tie the true winner without becoming the global empirical winner.",
            "Regret tolerances are absolute regression fractions, not utility-, latency-, or SLO-weighted deployment costs.",
            "The multinomial concentration bound is distribution-free and conservative; exact confidence regions can require fewer samples.",
            "The 10% prevalence simplex and E88 sensitivity kernels are not calibrated deployment distributions.",
            "The audit reuses compact E83 path summaries and is not new GPU evidence.",
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
            allow_nan=False,
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
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
