from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    policy_inventory,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    DRIFT_METRICS,
    load_e83,
)
from analyze_pacing_coalescing_order_path_allocation import (
    BUDGET_FACTORS,
    CONFIDENCE_LEVELS,
    COVERAGE_TARGETS,
    PREVALENCE_DENOMINATOR,
    TOLERANCE_TARGETS,
    budget_key,
    comparison_summary,
    coverage_requirements,
    load_json,
    planned_counts,
    worst_accuracy_summary,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
    prevalence_profiles,
)
from analyze_pacing_coalescing_order_prevalence_certification import (
    quantile,
)
from analyze_pacing_coalescing_order_risk_alpha_allocation import (
    allocation_weights,
    geometry_scores,
    interval_summary,
    load_e89,
    weighted_intervals,
)
from analyze_pacing_coalescing_order_risk_uncertainty import (
    base_cells,
    policy_coefficients,
    regret_summary,
    robust_regret,
    selected_policies,
)


DESIGN_CONFIDENCE = 0.95
METHOD = (
    "waterfilled_mean_absolute_"
    "contrast_paths_mean_absolute_"
    "contrast_alpha"
)
BASELINE_METHOD = (
    "mean_absolute_contrast_paths_"
    "mean_absolute_contrast_alpha"
)


def load_e98(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_path_allocation"
        / "pacing_coalescing_order_path_allocation_full.json"
    )


def surrogate_coefficients(
    *,
    contrast: np.ndarray,
    alpha_weights: np.ndarray,
) -> np.ndarray:
    family_alpha = (
        1.0
        - DESIGN_CONFIDENCE
    )
    return (
        contrast
        * np.sqrt(
            np.log(
                2.0
                / (
                    family_alpha
                    * alpha_weights
                )
            )
            / 2.0
        )
    )


def continuous_waterfill(
    *,
    base_totals: np.ndarray,
    target_budget: int,
    coefficients: np.ndarray,
) -> dict[str, Any]:
    base = base_totals.astype(
        np.float64
    )
    if target_budget < int(
        np.sum(base_totals)
    ):
        raise RuntimeError(
            "Target budget shrinks evidence"
        )
    target_score = np.power(
        coefficients,
        2.0 / 3.0,
    )
    positive = target_score > 0.0
    if not np.any(positive):
        raise RuntimeError(
            "Water-fill surrogate has no mass"
        )
    if target_budget == int(
        np.sum(base_totals)
    ):
        totals = base.copy()
        scale = 0.0
    else:
        lower = 0.0
        upper = float(
            np.max(
                base[positive]
                / target_score[
                    positive
                ]
            )
            * 2.0
        )
        while (
            float(
                np.sum(
                    np.maximum(
                        base,
                        upper
                        * target_score,
                    )
                )
            )
            < target_budget
        ):
            upper *= 2.0
        for _ in range(120):
            middle = (
                lower + upper
            ) / 2.0
            current = float(
                np.sum(
                    np.maximum(
                        base,
                        middle
                        * target_score,
                    )
                )
            )
            if current < target_budget:
                lower = middle
            else:
                upper = middle
        scale = upper
        totals = np.maximum(
            base,
            scale * target_score,
        )
    active = totals > base + 1e-7
    ratios = (
        totals[active]
        / target_score[active]
        if np.any(active)
        else np.asarray(
            [],
            dtype=np.float64,
        )
    )
    active_scale_relative_spread = (
        0.0
        if not len(ratios)
        else (
            float(
                np.max(ratios)
                - np.min(ratios)
            )
            / float(
                np.mean(ratios)
            )
        )
    )
    inactive_violation = float(
        np.max(
            np.maximum(
                0.0,
                scale
                * target_score[
                    ~active
                ]
                - base[~active],
            )
        )
    )
    return {
        "totals": totals,
        "target_score": target_score,
        "scale": scale,
        "active": active,
        "active_scale_relative_spread": (
            active_scale_relative_spread
        ),
        "inactive_constraint_maximum_violation": (
            inactive_violation
        ),
    }


def integer_waterfill(
    *,
    base_successes: np.ndarray,
    base_totals: np.ndarray,
    budget_factor: float,
    coefficients: np.ndarray,
) -> dict[str, Any]:
    base_budget = int(
        np.sum(base_totals)
    )
    target_budget = int(
        round(
            base_budget
            * budget_factor
        )
    )
    continuous = continuous_waterfill(
        base_totals=base_totals,
        target_budget=target_budget,
        coefficients=coefficients,
    )
    continuous_additional = (
        continuous["totals"]
        - base_totals
    )
    additional_totals = np.floor(
        continuous_additional
    ).astype(np.int64)
    remaining = (
        target_budget
        - base_budget
        - int(
            np.sum(
                additional_totals
            )
        )
    )
    fractions = (
        continuous_additional
        - additional_totals
    )
    order = np.lexsort(
        (
            np.arange(
                len(fractions)
            ),
            -fractions,
        )
    )
    additional_totals[
        order[:remaining]
    ] += 1
    totals = (
        base_totals
        + additional_totals
    )
    rates = (
        base_successes
        / base_totals
    )
    expected_additional = (
        rates
        * additional_totals
    )
    additional_successes = np.floor(
        expected_additional + 0.5
    ).astype(np.int64)
    additional_successes = np.minimum(
        additional_successes,
        additional_totals,
    )
    successes = (
        base_successes
        + additional_successes
    )
    projected_rates = (
        successes / totals
    )
    objective = float(
        np.sum(
            coefficients
            / np.sqrt(totals)
        )
    )
    continuous_objective = float(
        np.sum(
            coefficients
            / np.sqrt(
                continuous[
                    "totals"
                ]
            )
        )
    )
    add_benefit = (
        coefficients
        * (
            1.0 / np.sqrt(totals)
            - 1.0
            / np.sqrt(
                totals + 1
            )
        )
    )
    donors = totals > base_totals
    remove_cost = (
        coefficients[donors]
        * (
            1.0
            / np.sqrt(
                totals[donors]
                - 1
            )
            - 1.0
            / np.sqrt(
                totals[donors]
            )
        )
    )
    single_transfer_maximum_improvement = (
        0.0
        if not np.any(donors)
        else max(
            0.0,
            float(
                np.max(add_benefit)
                - np.min(remove_cost)
            ),
        )
    )
    return {
        "successes": successes,
        "totals": totals,
        "additional_successes": (
            additional_successes
        ),
        "additional_totals": (
            additional_totals
        ),
        "target_budget": target_budget,
        "additional_budget": (
            target_budget
            - base_budget
        ),
        "objective": objective,
        "continuous_objective": (
            continuous_objective
        ),
        "integer_objective_ratio": (
            objective
            / continuous_objective
        ),
        "single_transfer_maximum_improvement": (
            single_transfer_maximum_improvement
        ),
        "active_cell_count": int(
            np.count_nonzero(
                additional_totals
            )
        ),
        "active_scale_relative_spread": (
            continuous[
                "active_scale_relative_spread"
            ]
        ),
        "inactive_constraint_maximum_violation": (
            continuous[
                "inactive_constraint_maximum_violation"
            ]
        ),
        "maximum_rate_rounding_error_ppm": (
            float(
                np.max(
                    np.abs(
                        projected_rates
                        - rates
                    )
                )
            )
            * 1_000_000.0
        ),
    }


def allocation_summary(
    *,
    allocation: dict[str, Any],
    labels: list[dict[str, Any]],
    baseline_objective: float,
    uniform_objective: float,
) -> dict[str, Any]:
    additional = allocation[
        "additional_totals"
    ]
    totals = allocation["totals"]
    order = np.argsort(
        additional
    )[::-1]
    additional_sum = int(
        np.sum(additional)
    )
    return {
        "target_budget": int(
            allocation[
                "target_budget"
            ]
        ),
        "additional_budget": (
            additional_sum
        ),
        "minimum_total_per_cell": int(
            np.min(totals)
        ),
        "median_total_per_cell": (
            quantile(
                totals.tolist(),
                0.50,
            )
        ),
        "maximum_total_per_cell": int(
            np.max(totals)
        ),
        "active_cell_count": (
            allocation[
                "active_cell_count"
            ]
        ),
        "top_10_additional_share": (
            0.0
            if not additional_sum
            else float(
                np.sum(
                    additional[
                        order[:10]
                    ]
                )
                / additional_sum
            )
        ),
        "surrogate_objective": (
            allocation["objective"]
        ),
        "objective_ratio_vs_e98_linear": (
            allocation["objective"]
            / baseline_objective
        ),
        "objective_ratio_vs_uniform": (
            allocation["objective"]
            / uniform_objective
        ),
        "integer_objective_ratio_vs_continuous_optimum": (
            allocation[
                "integer_objective_ratio"
            ]
        ),
        "single_transfer_maximum_improvement": (
            allocation[
                "single_transfer_maximum_improvement"
            ]
        ),
        "active_scale_relative_spread": (
            allocation[
                "active_scale_relative_spread"
            ]
        ),
        "inactive_constraint_maximum_violation": (
            allocation[
                "inactive_constraint_maximum_violation"
            ]
        ),
        "maximum_rate_rounding_error_ppm": (
            allocation[
                "maximum_rate_rounding_error_ppm"
            ]
        ),
        "top_cells": [
            {
                **labels[int(index)],
                "additional_comparisons": (
                    int(
                        additional[
                            index
                        ]
                    )
                ),
                "total_comparisons": int(
                    totals[index]
                ),
            }
            for index in order[:10]
        ],
    }


def clopper_pearson_validation(
    *,
    intervals: dict[str, Any],
) -> dict[str, Any]:
    checks = 0
    maximum_error = 0.0
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        family_alpha = (
            1.0
            - confidence
        )
        for factor in (
            BUDGET_FACTORS
        ):
            row = intervals[
                confidence_key
            ][budget_key(factor)]
            successes = row[
                "successes"
            ]
            totals = row["totals"]
            weights = row[
                "alpha_weights"
            ]
            tail = (
                family_alpha
                * weights
                / 2.0
            )
            lower = row["lower"]
            upper = row["upper"]
            positive = successes > 0
            not_full = successes < totals
            if np.any(positive):
                probability = beta.cdf(
                    lower[positive],
                    successes[positive],
                    (
                        totals[positive]
                        - successes[positive]
                        + 1
                    ),
                )
                maximum_error = max(
                    maximum_error,
                    float(
                        np.max(
                            np.abs(
                                probability
                                - tail[
                                    positive
                                ]
                            )
                        )
                    ),
                )
                checks += int(
                    np.count_nonzero(
                        positive
                    )
                )
            if np.any(not_full):
                probability = beta.sf(
                    upper[not_full],
                    (
                        successes[not_full]
                        + 1
                    ),
                    (
                        totals[not_full]
                        - successes[
                            not_full
                        ]
                    ),
                )
                maximum_error = max(
                    maximum_error,
                    float(
                        np.max(
                            np.abs(
                                probability
                                - tail[
                                    not_full
                                ]
                            )
                        )
                    ),
                )
                checks += int(
                    np.count_nonzero(
                        not_full
                    )
                )
    return {
        "checks": checks,
        "maximum_probability_error": (
            maximum_error
        ),
    }


def global_comparison_summary(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    def ratio(
        numerator: float,
        denominator: float,
    ) -> float | None:
        if denominator <= 0.0:
            return (
                1.0
                if numerator <= 0.0
                else None
            )
        return (
            numerator
            / denominator
        )

    return {
        "p50_regret_ratio": ratio(
            candidate[
                "robust_regret_ppm"
            ]["p50"],
            baseline[
                "robust_regret_ppm"
            ]["p50"],
        ),
        "p90_regret_ratio": ratio(
            candidate[
                "robust_regret_ppm"
            ]["p90"],
            baseline[
                "robust_regret_ppm"
            ]["p90"],
        ),
        "maximum_regret_ratio": ratio(
            candidate[
                "robust_regret_ppm"
            ]["maximum"],
            baseline[
                "robust_regret_ppm"
            ]["maximum"],
        ),
        "ten_ppm_coverage_delta": (
            candidate[
                "tolerance_coverage"
            ]["10"]
            - baseline[
                "tolerance_coverage"
            ]["10"]
        ),
        "twenty_five_ppm_coverage_delta": (
            candidate[
                "tolerance_coverage"
            ]["25"]
            - baseline[
                "tolerance_coverage"
            ]["25"]
        ),
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e98 = load_e98(args.root)
    drift = e83["headline"][
        "drift_by_marginal_fraction_and_order"
    ]
    costs = sorted(
        drift,
        key=float,
    )
    orders = sorted(
        next(
            iter(
                drift.values()
            )
        )
    )
    metrics = list(
        DRIFT_METRICS
    )
    cells = base_cells(
        drift=drift,
        costs=costs,
        orders=orders,
        metrics=metrics,
    )
    point = cells["point"].reshape(
        (
            len(costs),
            len(orders),
            len(metrics),
        )
    )
    inventory = policy_inventory(
        root=args.root,
        costs=costs,
    )
    names = sorted(
        inventory
    )
    scores = geometry_scores(
        inventory=inventory,
        names=names,
        costs=costs,
        orders=orders,
        metrics=metrics,
    )
    weights = allocation_weights(
        scores=scores,
        cell_count=len(
            cells["labels"]
        ),
    )
    contrast = scores[
        "mean_absolute_contrast"
    ].reshape(-1)
    alpha_weights = weights[
        "mean_absolute_contrast"
    ]
    coefficients = (
        surrogate_coefficients(
            contrast=contrast,
            alpha_weights=alpha_weights,
        )
    )

    allocations = {}
    allocation_summaries = {}
    intervals = {
        f"{confidence:.2f}": {}
        for confidence in (
            CONFIDENCE_LEVELS
        )
    }
    for factor in BUDGET_FACTORS:
        key = budget_key(
            factor
        )
        allocation = integer_waterfill(
            base_successes=(
                cells["successes"]
            ),
            base_totals=(
                cells["totals"]
            ),
            budget_factor=factor,
            coefficients=coefficients,
        )
        linear = planned_counts(
            base_successes=(
                cells["successes"]
            ),
            base_totals=(
                cells["totals"]
            ),
            budget_factor=factor,
            path_weights=alpha_weights,
        )
        uniform = planned_counts(
            base_successes=(
                cells["successes"]
            ),
            base_totals=(
                cells["totals"]
            ),
            budget_factor=factor,
            path_weights=weights[
                "equal"
            ],
        )
        linear_objective = float(
            np.sum(
                coefficients
                / np.sqrt(
                    linear["totals"]
                )
            )
        )
        uniform_objective = float(
            np.sum(
                coefficients
                / np.sqrt(
                    uniform["totals"]
                )
            )
        )
        allocations[key] = allocation
        allocation_summaries[key] = (
            allocation_summary(
                allocation=allocation,
                labels=cells["labels"],
                baseline_objective=(
                    linear_objective
                ),
                uniform_objective=(
                    uniform_objective
                ),
            )
        )
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            lower, upper = (
                weighted_intervals(
                    successes=allocation[
                        "successes"
                    ],
                    totals=allocation[
                        "totals"
                    ],
                    confidence=confidence,
                    weights=alpha_weights,
                    family_divisor=1,
                )
            )
            intervals[
                confidence_key
            ][key] = {
                "lower": lower,
                "upper": upper,
                "successes": allocation[
                    "successes"
                ],
                "totals": allocation[
                    "totals"
                ],
                "alpha_weights": (
                    alpha_weights
                ),
                "summary": (
                    interval_summary(
                        lower=lower,
                        upper=upper,
                    )
                ),
            }

    profiles = prevalence_profiles(
        denominator=(
            PREVALENCE_DENOMINATOR
        ),
        cost_count=len(costs),
    )
    prevalence = np.asarray(
        [
            profile["weights"]
            for profile in profiles
        ],
        dtype=np.float64,
    )
    accuracy_grid = [
        index / 100.0
        for index in range(101)
    ]
    kernels = {}
    reconstruction_maximum_error = (
        0.0
    )
    current_reproduced = True
    p90_wins = 0
    p90_losses = 0
    p90_ties = 0
    comparison_count = 0
    comparison_by_condition = {
        f"{confidence:.2f}": {
            budget_key(factor): {
                "wins": 0,
                "losses": 0,
                "ties": 0,
            }
            for factor in BUDGET_FACTORS
            if factor > 1.0
        }
        for confidence in (
            CONFIDENCE_LEVELS
        )
    }
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        e89_grid = {
            row["accuracy"]: row
            for row in e89["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        e98_grid = e98["kernels"][
            kernel_name
        ]["accuracy_grid"]
        grid = []
        for accuracy_index, accuracy in enumerate(
            accuracy_grid
        ):
            coefficients_by_policy = (
                policy_coefficients(
                    inventory=inventory,
                    names=names,
                    costs=costs,
                    orders=orders,
                    metrics=metrics,
                    accuracy=accuracy,
                    offset_weights=kernel[
                        "offset_weights"
                    ],
                )
            )
            per_policy_costs = {
                name: per_cost_metrics(
                    drift=drift,
                    costs=costs,
                    mapping=inventory[
                        name
                    ]["mapping"],
                    accuracy=accuracy,
                    offset_weights=kernel[
                        "offset_weights"
                    ],
                )
                for name in names
            }
            reconstructed = np.sum(
                coefficients_by_policy
                * point[
                    None,
                    :,
                    :,
                    :,
                ],
                axis=(2, 3),
            )
            expected = np.asarray(
                [
                    [
                        row[
                            "regression_fraction"
                        ]
                        for row
                        in per_policy_costs[
                            name
                        ]
                    ]
                    for name in names
                ],
                dtype=np.float64,
            )
            reconstruction_maximum_error = max(
                reconstruction_maximum_error,
                float(
                    np.max(
                        np.abs(
                            reconstructed
                            - expected
                        )
                    )
                ),
            )
            selected = (
                selected_policies(
                    profiles=profiles,
                    per_policy_costs=(
                        per_policy_costs
                    ),
                    inventory=inventory,
                    names=names,
                )
            )
            winner_counts = {
                name: int(
                    np.count_nonzero(
                        selected == index
                    )
                )
                for index, name
                in enumerate(names)
            }
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
            confidence_rows = {}
            for confidence in (
                CONFIDENCE_LEVELS
            ):
                confidence_key = (
                    f"{confidence:.2f}"
                )
                confidence_rows[
                    confidence_key
                ] = {}
                for factor in (
                    BUDGET_FACTORS
                ):
                    key = budget_key(
                        factor
                    )
                    interval = intervals[
                        confidence_key
                    ][key]
                    regret, _ = (
                        robust_regret(
                            prevalence=(
                                prevalence
                            ),
                            selected=selected,
                            coefficients=(
                                coefficients_by_policy
                            ),
                            lower=interval[
                                "lower"
                            ].reshape(
                                point.shape
                            ),
                            upper=interval[
                                "upper"
                            ].reshape(
                                point.shape
                            ),
                        )
                    )
                    summary = (
                        regret_summary(
                            regret
                        )
                    )
                    baseline = (
                        e98_grid[
                            accuracy_index
                        ]["confidence"][
                            confidence_key
                        ][key][
                            BASELINE_METHOD
                        ]
                    )
                    confidence_rows[
                        confidence_key
                    ][key] = {
                        METHOD: summary,
                        "vs_e98_linear": (
                            comparison_summary(
                                baseline=baseline,
                                candidate=summary,
                            )
                        ),
                    }
                    difference = (
                        summary[
                            "robust_regret_ppm"
                        ]["p90"]
                        - baseline[
                            "robust_regret_ppm"
                        ]["p90"]
                    )
                    if factor > 1.0:
                        comparison_count += 1
                        if difference < -1e-12:
                            p90_wins += 1
                            comparison_by_condition[
                                confidence_key
                            ][key]["wins"] += 1
                        elif difference > 1e-12:
                            p90_losses += 1
                            comparison_by_condition[
                                confidence_key
                            ][key]["losses"] += 1
                        else:
                            p90_ties += 1
                            comparison_by_condition[
                                confidence_key
                            ][key]["ties"] += 1
                    if factor == 1.0:
                        for quantile_name in (
                            "p50",
                            "p90",
                            "p95",
                            "p99",
                            "maximum",
                        ):
                            current_reproduced &= (
                                abs(
                                    summary[
                                        "robust_regret_ppm"
                                    ][quantile_name]
                                    - baseline[
                                        "robust_regret_ppm"
                                    ][quantile_name]
                                )
                                <= 1e-9
                            )
                        for tolerance in (
                            TOLERANCE_TARGETS
                        ):
                            current_reproduced &= (
                                summary[
                                    "tolerance_coverage"
                                ][tolerance][
                                    "count"
                                ]
                                == baseline[
                                    "tolerance_coverage"
                                ][tolerance][
                                    "count"
                                ]
                            )
            grid.append(
                {
                    "accuracy": accuracy,
                    "winner_counts": (
                        winner_counts
                    ),
                    "confidence": (
                        confidence_rows
                    ),
                }
            )
        kernels[kernel_name] = {
            "accuracy_grid": grid,
            "worst_accuracy": (
                worst_accuracy_summary(
                    grid=grid,
                    method=METHOD,
                )
            ),
            "coverage_requirements": (
                coverage_requirements(
                    grid=grid,
                    method=METHOD,
                )
            ),
        }

    global_worst = {}
    global_requirements = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        global_worst[
            confidence_key
        ] = {}
        global_requirements[
            confidence_key
        ] = {}
        for factor in (
            BUDGET_FACTORS
        ):
            key = budget_key(
                factor
            )
            rows = [
                kernels[kernel][
                    "worst_accuracy"
                ][confidence_key][key]
                for kernel in KERNELS
            ]
            global_worst[
                confidence_key
            ][key] = {
                "robust_regret_ppm": {
                    quantile_name: max(
                        row[
                            "robust_regret_ppm"
                        ][quantile_name]
                        for row in rows
                    )
                    for quantile_name in (
                        "p50",
                        "p90",
                        "p95",
                        "p99",
                        "maximum",
                    )
                },
                "tolerance_coverage": {
                    tolerance: min(
                        row[
                            "tolerance_coverage"
                        ][tolerance]
                        for row in rows
                    )
                    for tolerance in (
                        TOLERANCE_TARGETS
                    )
                },
            }
        for tolerance in (
            TOLERANCE_TARGETS
        ):
            global_requirements[
                confidence_key
            ][tolerance] = {}
            for target in (
                COVERAGE_TARGETS
            ):
                target_key = (
                    f"{target:.2f}"
                )
                values = [
                    kernels[kernel][
                        "coverage_requirements"
                    ][confidence_key][
                        tolerance
                    ][target_key][
                        "maximum_required_budget_factor"
                    ]
                    for kernel in KERNELS
                ]
                global_requirements[
                    confidence_key
                ][tolerance][
                    target_key
                ] = {
                    "maximum_required_budget_factor": (
                        None
                        if any(
                            value is None
                            for value in values
                        )
                        else max(
                            values
                        )
                    )
                }

    monotone_p90 = True
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        previous = None
        for factor in (
            BUDGET_FACTORS
        ):
            value = global_worst[
                confidence_key
            ][budget_key(factor)][
                "robust_regret_ppm"
            ]["p90"]
            if (
                previous is not None
                and value
                > previous + 1e-9
            ):
                monotone_p90 = False
            previous = value
    global_vs_baseline = {
        confidence_key: {
            key: global_comparison_summary(
                baseline=e98[
                    "headline"
                ][
                    "global_worst_accuracy"
                ][BASELINE_METHOD][
                    confidence_key
                ][key],
                candidate=global_worst[
                    confidence_key
                ][key],
            )
            for key in (
                budget_key(factor)
                for factor in (
                    BUDGET_FACTORS
                )
            )
        }
        for confidence_key in (
            f"{confidence:.2f}"
            for confidence in (
                CONFIDENCE_LEVELS
            )
        )
    }
    cp_validation = (
        clopper_pearson_validation(
            intervals=intervals,
        )
    )
    exact_budgets = all(
        int(
            np.sum(
                allocations[
                    budget_key(factor)
                ]["totals"]
            )
        )
        == int(
            round(
                np.sum(
                    cells["totals"]
                )
                * factor
            )
        )
        for factor in BUDGET_FACTORS
    )
    preserves_evidence = all(
        np.all(
            allocations[
                budget_key(factor)
            ]["totals"]
            >= cells["totals"]
        )
        for factor in BUDGET_FACTORS
    )
    kkt_valid = all(
        allocation_summaries[
            budget_key(factor)
        ][
            "active_scale_relative_spread"
        ]
        <= 1e-10
        and allocation_summaries[
            budget_key(factor)
        ][
            "inactive_constraint_maximum_violation"
        ]
        <= 1e-6
        and allocation_summaries[
            budget_key(factor)
        ][
            "integer_objective_ratio_vs_continuous_optimum"
        ]
        <= 1.000000001
        and allocation_summaries[
            budget_key(factor)
        ][
            "single_transfer_maximum_improvement"
        ]
        <= 1e-18
        for factor in BUDGET_FACTORS
    )
    surrogate_dominates = all(
        allocation_summaries[
            budget_key(factor)
        ][
            "objective_ratio_vs_e98_linear"
        ]
        <= 1.0 + 1e-12
        for factor in BUDGET_FACTORS
    )
    if not (
        current_reproduced
        and exact_budgets
        and preserves_evidence
        and kkt_valid
        and surrogate_dominates
        and monotone_p90
        and reconstruction_maximum_error
        <= 5e-18
        and cp_validation[
            "maximum_probability_error"
        ]
        <= 1e-10
    ):
        raise RuntimeError(
            "E99 validation failed"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "path_waterfill"
        ),
        "source": {
            "e83": (
                "pacing_coalescing_order_full.json"
            ),
            "e89": (
                "pacing_coalescing_order_"
                "prevalence_full.json"
            ),
            "e98": (
                "pacing_coalescing_order_"
                "path_allocation_full.json"
            ),
        },
        "model": (
            "At 95% family confidence with "
            "E97 mean-contrast alpha weights, "
            "minimize the outcome-independent "
            "Hoeffding-width surrogate sum_i "
            "c_i sqrt(log(2/(alpha w_i))/"
            "(2 n_i)) subject to retaining all "
            "existing E83 counts and an exact "
            "total path budget. KKT conditions "
            "give active final counts "
            "proportional to the two-thirds "
            "power of the confidence-adjusted "
            "policy contrast."
        ),
        "method": METHOD,
        "baseline_method": (
            BASELINE_METHOD
        ),
        "design_confidence": (
            DESIGN_CONFIDENCE
        ),
        "budget_factors": list(
            BUDGET_FACTORS
        ),
        "confidence_levels": list(
            CONFIDENCE_LEVELS
        ),
        "allocations": (
            allocation_summaries
        ),
        "headline": {
            "global_worst_accuracy": (
                global_worst
            ),
            "coverage_requirements": (
                global_requirements
            ),
            "p90_comparison": {
                "conditions": (
                    comparison_count
                ),
                "wins": p90_wins,
                "losses": p90_losses,
                "ties": p90_ties,
            },
            "p90_comparison_by_confidence_and_budget": (
                comparison_by_condition
            ),
            "global_vs_e98_linear": (
                global_vs_baseline
            ),
            "e98_baseline": {
                "global_worst_accuracy": (
                    e98["headline"][
                        "global_worst_accuracy"
                    ][BASELINE_METHOD]
                ),
                "coverage_requirements": (
                    e98["headline"][
                        "coverage_requirements"
                    ][BASELINE_METHOD]
                ),
            },
        },
        "kernels": kernels,
        "validation": {
            "e89_winner_counts_reproduced": (
                True
            ),
            "e98_current_evidence_reproduced": (
                current_reproduced
            ),
            "total_budgets_exact": (
                exact_budgets
            ),
            "no_cell_loses_existing_evidence": (
                preserves_evidence
            ),
            "continuous_kkt_conditions_valid": (
                kkt_valid
            ),
            "surrogate_no_worse_than_e98_linear": (
                surrogate_dominates
            ),
            "global_p90_monotone": (
                monotone_p90
            ),
            "policy_coefficient_reconstruction_maximum_error": (
                reconstruction_maximum_error
            ),
            "clopper_pearson": (
                cp_validation
            ),
        },
        "limitations": [
            (
                "The optimized objective is a "
                "worst-case Hoeffding-width "
                "surrogate averaged through "
                "policy contrast, not exact "
                "Clopper-Pearson robust regret."
            ),
            (
                "Projected future successes "
                "retain E83 rates and round to "
                "integer counts; they are "
                "planning sensitivity, not new "
                "measured paths."
            ),
            (
                "The design is preregistered at "
                "95% family confidence with E97 "
                "mean-contrast alpha. The same "
                "path allocation is evaluated, "
                "not retuned, at 99%."
            ),
            (
                "Cellwise binomial and "
                "rectangular-confidence "
                "limitations from E96 remain."
            ),
            (
                "The audit is local statistical "
                "analysis and is not new GPU "
                "evidence."
            ),
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
