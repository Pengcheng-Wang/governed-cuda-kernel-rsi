from __future__ import annotations

import argparse
import json
import math
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


CONFIDENCE_LEVELS = (
    0.95,
    0.99,
)
BUDGET_FACTORS = (
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
)
PATH_SCHEMES = (
    "uniform",
    "mean_absolute_contrast",
)
ALPHA_SCHEMES = (
    "equal",
    "mean_absolute_contrast",
)
METHODS = tuple(
    f"{path}_paths_{alpha}_alpha"
    for path in PATH_SCHEMES
    for alpha in ALPHA_SCHEMES
)
TOLERANCE_TARGETS = (
    "5",
    "10",
    "25",
    "50",
)
COVERAGE_TARGETS = (
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
)
PREVALENCE_DENOMINATOR = 10


def budget_key(
    factor: float,
) -> str:
    return (
        str(int(factor))
        if float(factor).is_integer()
        else f"{factor:g}"
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_e96(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_risk_uncertainty"
        / "pacing_coalescing_order_risk_uncertainty_full.json"
    )


def load_e97(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_risk_alpha_allocation"
        / "pacing_coalescing_order_risk_alpha_allocation_full.json"
    )


def largest_remainder(
    *,
    total: int,
    weights: np.ndarray,
) -> np.ndarray:
    raw = weights * total
    result = np.floor(
        raw
    ).astype(np.int64)
    remaining = int(
        total
        - int(
            np.sum(result)
        )
    )
    if remaining:
        fractions = raw - result
        order = np.lexsort(
            (
                np.arange(
                    len(weights)
                ),
                -fractions,
            )
        )
        result[
            order[:remaining]
        ] += 1
    if (
        np.any(result < 0)
        or int(
            np.sum(result)
        )
        != total
    ):
        raise RuntimeError(
            "Invalid largest-remainder allocation"
        )
    return result


def planned_counts(
    *,
    base_successes: np.ndarray,
    base_totals: np.ndarray,
    budget_factor: float,
    path_weights: np.ndarray,
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
    additional_budget = (
        target_budget
        - base_budget
    )
    if additional_budget < 0:
        raise RuntimeError(
            "Budget factor shrinks evidence"
        )
    additional_totals = (
        largest_remainder(
            total=additional_budget,
            weights=path_weights,
        )
    )
    rates = (
        base_successes
        / base_totals
    )
    expected_additional = (
        rates
        * additional_totals
    )
    additional_successes = (
        np.floor(
            expected_additional
            + 0.5
        ).astype(np.int64)
    )
    additional_successes = np.minimum(
        additional_successes,
        additional_totals,
    )
    totals = (
        base_totals
        + additional_totals
    )
    successes = (
        base_successes
        + additional_successes
    )
    projected_rates = (
        successes
        / totals
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
            additional_budget
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


def path_summary(
    *,
    allocation: dict[str, Any],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    additional = allocation[
        "additional_totals"
    ]
    totals = allocation["totals"]
    order = np.argsort(
        additional
    )[::-1]
    return {
        "target_budget": int(
            allocation[
                "target_budget"
            ]
        ),
        "additional_budget": int(
            allocation[
                "additional_budget"
            ]
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
        "top_10_additional_share": (
            0.0
            if not int(
                np.sum(additional)
            )
            else float(
                np.sum(
                    additional[
                        order[:10]
                    ]
                )
                / np.sum(additional)
            )
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


def coverage_requirements(
    *,
    grid: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    result = {}
    factor_keys = [
        budget_key(factor)
        for factor in BUDGET_FACTORS
    ]
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        result[confidence_key] = {}
        for tolerance in (
            TOLERANCE_TARGETS
        ):
            result[confidence_key][
                tolerance
            ] = {}
            for target in (
                COVERAGE_TARGETS
            ):
                per_accuracy = []
                for row in grid:
                    coverage = [
                        row["confidence"][
                            confidence_key
                        ][key][method][
                            "tolerance_coverage"
                        ][tolerance][
                            "fraction"
                        ]
                        for key in factor_keys
                    ]
                    selected = None
                    for index, value in enumerate(
                        coverage
                    ):
                        if (
                            value >= target
                            and all(
                                later >= target
                                for later
                                in coverage[
                                    index:
                                ]
                            )
                        ):
                            selected = (
                                BUDGET_FACTORS[
                                    index
                                ]
                            )
                            break
                    per_accuracy.append(
                        selected
                    )
                finite = [
                    value
                    for value in per_accuracy
                    if value is not None
                ]
                result[confidence_key][
                    tolerance
                ][f"{target:.2f}"] = {
                    "maximum_required_budget_factor": (
                        None
                        if len(finite)
                        != len(
                            per_accuracy
                        )
                        else max(
                            finite
                        )
                    ),
                    "accuracy_fraction_certified_at_16x": (
                        len(finite)
                        / len(
                            per_accuracy
                        )
                    ),
                }
    return result


def worst_accuracy_summary(
    *,
    grid: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    result = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        result[confidence_key] = {}
        for factor in (
            BUDGET_FACTORS
        ):
            key = budget_key(
                factor
            )
            rows = [
                row["confidence"][
                    confidence_key
                ][key][method]
                for row in grid
            ]
            result[confidence_key][
                key
            ] = {
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
                        ][tolerance][
                            "fraction"
                        ]
                        for row in rows
                    )
                    for tolerance in (
                        TOLERANCE_TARGETS
                    )
                },
            }
    return result


def comparison_summary(
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
            ]["10"]["fraction"]
            - baseline[
                "tolerance_coverage"
            ]["10"]["fraction"]
        ),
        "twenty_five_ppm_coverage_delta": (
            candidate[
                "tolerance_coverage"
            ]["25"]["fraction"]
            - baseline[
                "tolerance_coverage"
            ]["25"]["fraction"]
        ),
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
        alpha = (
            1.0
            - confidence
        )
        for factor in (
            BUDGET_FACTORS
        ):
            factor_key = (
                budget_key(
                    factor
                )
            )
            for method in METHODS:
                row = intervals[
                    confidence_key
                ][factor_key][method]
                successes = row[
                    "successes"
                ]
                totals = row["totals"]
                weights = row[
                    "alpha_weights"
                ]
                tail = (
                    alpha
                    * weights
                    / 2.0
                )
                lower = row["lower"]
                upper = row["upper"]
                positive = (
                    successes > 0
                )
                not_full = (
                    successes < totals
                )
                if np.any(positive):
                    observed = beta.cdf(
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
                                    observed
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
                    observed = beta.sf(
                        upper[not_full],
                        (
                            successes[
                                not_full
                            ]
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
                                    observed
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


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e96 = load_e96(args.root)
    e97 = load_e97(args.root)
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
    path_weights = {
        "uniform": weights["equal"],
        "mean_absolute_contrast": (
            weights[
                "mean_absolute_contrast"
            ]
        ),
    }
    alpha_weights = {
        "equal": weights["equal"],
        "mean_absolute_contrast": (
            weights[
                "mean_absolute_contrast"
            ]
        ),
    }

    allocations = {}
    path_summaries = {}
    for factor in BUDGET_FACTORS:
        factor_key = budget_key(
            factor
        )
        allocations[factor_key] = {}
        path_summaries[factor_key] = {}
        for scheme in PATH_SCHEMES:
            allocation = planned_counts(
                base_successes=(
                    cells["successes"]
                ),
                base_totals=(
                    cells["totals"]
                ),
                budget_factor=factor,
                path_weights=(
                    path_weights[scheme]
                ),
            )
            allocations[factor_key][
                scheme
            ] = allocation
            path_summaries[factor_key][
                scheme
            ] = path_summary(
                allocation=allocation,
                labels=cells["labels"],
            )

    intervals = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        intervals[confidence_key] = {}
        for factor in (
            BUDGET_FACTORS
        ):
            factor_key = budget_key(
                factor
            )
            intervals[
                confidence_key
            ][factor_key] = {}
            for path_scheme in (
                PATH_SCHEMES
            ):
                allocation = (
                    allocations[
                        factor_key
                    ][path_scheme]
                )
                for alpha_scheme in (
                    ALPHA_SCHEMES
                ):
                    method = (
                        f"{path_scheme}_paths_"
                        f"{alpha_scheme}_alpha"
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
                            weights=(
                                alpha_weights[
                                    alpha_scheme
                                ]
                            ),
                            family_divisor=1,
                        )
                    )
                    intervals[
                        confidence_key
                    ][factor_key][
                        method
                    ] = {
                        "lower": lower,
                        "upper": upper,
                        "successes": (
                            allocation[
                                "successes"
                            ]
                        ),
                        "totals": (
                            allocation[
                                "totals"
                            ]
                        ),
                        "alpha_weights": (
                            alpha_weights[
                                alpha_scheme
                            ]
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
    e96_reproduced = True
    e97_reproduced = True
    path_win_counts = {
        alpha: 0
        for alpha in (
            ALPHA_SCHEMES
        )
    }
    path_condition_count = 0
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        e89_grid = {
            row["accuracy"]: row
            for row in e89["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        e96_grid = e96["kernels"][
            kernel_name
        ]["accuracy_grid"]
        e97_grid = e97["kernels"][
            kernel_name
        ]["accuracy_grid"]
        grid = []
        for accuracy_index, accuracy in enumerate(
            accuracy_grid
        ):
            coefficients = (
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
                coefficients
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
                    factor_key = (
                        budget_key(
                            factor
                        )
                    )
                    method_rows = {}
                    for method in METHODS:
                        interval = intervals[
                            confidence_key
                        ][factor_key][
                            method
                        ]
                        regret, _ = (
                            robust_regret(
                                prevalence=(
                                    prevalence
                                ),
                                selected=(
                                    selected
                                ),
                                coefficients=(
                                    coefficients
                                ),
                                lower=(
                                    interval[
                                        "lower"
                                    ].reshape(
                                        point.shape
                                    )
                                ),
                                upper=(
                                    interval[
                                        "upper"
                                    ].reshape(
                                        point.shape
                                    )
                                ),
                            )
                        )
                        method_rows[
                            method
                        ] = regret_summary(
                            regret
                        )
                    method_rows[
                        "geometry_paths_vs_uniform_equal_alpha"
                    ] = comparison_summary(
                        baseline=method_rows[
                            "uniform_paths_equal_alpha"
                        ],
                        candidate=method_rows[
                            "mean_absolute_contrast_paths_equal_alpha"
                        ],
                    )
                    method_rows[
                        "geometry_paths_vs_uniform_mean_alpha"
                    ] = comparison_summary(
                        baseline=method_rows[
                            "uniform_paths_mean_absolute_contrast_alpha"
                        ],
                        candidate=method_rows[
                            "mean_absolute_contrast_paths_mean_absolute_contrast_alpha"
                        ],
                    )
                    for alpha_scheme in (
                        ALPHA_SCHEMES
                    ):
                        uniform_method = (
                            "uniform_paths_"
                            f"{alpha_scheme}_alpha"
                        )
                        geometry_method = (
                            "mean_absolute_contrast_paths_"
                            f"{alpha_scheme}_alpha"
                        )
                        if (
                            method_rows[
                                geometry_method
                            ][
                                "robust_regret_ppm"
                            ]["p90"]
                            < method_rows[
                                uniform_method
                            ][
                                "robust_regret_ppm"
                            ]["p90"]
                            - 1e-12
                        ):
                            path_win_counts[
                                alpha_scheme
                            ] += 1
                    if factor > 1.0:
                        path_condition_count += 1
                    confidence_rows[
                        confidence_key
                    ][factor_key] = (
                        method_rows
                    )
                    if factor_key in {
                        "1",
                        "2",
                        "4",
                        "8",
                        "16",
                    }:
                        expected_e96 = (
                            e96_grid[
                                accuracy_index
                            ]["confidence"][
                                confidence_key
                            ][factor_key]
                        )
                        actual_e96 = (
                            method_rows[
                                "uniform_paths_equal_alpha"
                            ]
                        )
                        for quantile_name in (
                            "p50",
                            "p90",
                            "p95",
                            "p99",
                            "maximum",
                        ):
                            e96_reproduced &= (
                                abs(
                                    actual_e96[
                                        "robust_regret_ppm"
                                    ][quantile_name]
                                    - expected_e96[
                                        "robust_regret_ppm"
                                    ][quantile_name]
                                )
                                <= 1e-9
                            )
                        for tolerance in (
                            TOLERANCE_TARGETS
                        ):
                            e96_reproduced &= (
                                actual_e96[
                                    "tolerance_coverage"
                                ][tolerance][
                                    "count"
                                ]
                                == expected_e96[
                                    "tolerance_coverage"
                                ][tolerance][
                                    "count"
                                ]
                            )
                    if factor == 1.0:
                        expected_e97 = (
                            e97_grid[
                                accuracy_index
                            ]["confidence"][
                                confidence_key
                            ]["1"]
                        )
                        for alpha_scheme in (
                            ALPHA_SCHEMES
                        ):
                            actual_method = (
                                "uniform_paths_"
                                f"{alpha_scheme}_alpha"
                            )
                            expected_method = (
                                alpha_scheme
                            )
                            actual_e97 = (
                                method_rows[
                                    actual_method
                                ]
                            )
                            expected_row = (
                                expected_e97[
                                    expected_method
                                ]
                            )
                            for quantile_name in (
                                "p50",
                                "p90",
                                "p95",
                                "p99",
                                "maximum",
                            ):
                                e97_reproduced &= (
                                    abs(
                                        actual_e97[
                                            "robust_regret_ppm"
                                        ][quantile_name]
                                        - expected_row[
                                            "robust_regret_ppm"
                                        ][quantile_name]
                                    )
                                    <= 1e-9
                                )
                            for tolerance in (
                                TOLERANCE_TARGETS
                            ):
                                e97_reproduced &= (
                                    actual_e97[
                                        "tolerance_coverage"
                                    ][tolerance][
                                        "count"
                                    ]
                                    == expected_row[
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
            "worst_accuracy": {
                method: (
                    worst_accuracy_summary(
                        grid=grid,
                        method=method,
                    )
                )
                for method in METHODS
            },
            "coverage_requirements": {
                method: (
                    coverage_requirements(
                        grid=grid,
                        method=method,
                    )
                )
                for method in METHODS
            },
        }

    global_worst = {}
    for method in METHODS:
        global_worst[method] = {}
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            global_worst[method][
                confidence_key
            ] = {}
            for factor in (
                BUDGET_FACTORS
            ):
                factor_key = (
                    budget_key(
                        factor
                    )
                )
                rows = [
                    kernels[kernel][
                        "worst_accuracy"
                    ][method][
                        confidence_key
                    ][factor_key]
                    for kernel in KERNELS
                ]
                global_worst[method][
                    confidence_key
                ][factor_key] = {
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

    global_requirements = {}
    for method in METHODS:
        global_requirements[
            method
        ] = {}
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            global_requirements[
                method
            ][confidence_key] = {}
            for tolerance in (
                TOLERANCE_TARGETS
            ):
                global_requirements[
                    method
                ][confidence_key][
                    tolerance
                ] = {}
                for target in (
                    COVERAGE_TARGETS
                ):
                    key = f"{target:.2f}"
                    values = [
                        kernels[kernel][
                            "coverage_requirements"
                        ][method][
                            confidence_key
                        ][tolerance][key][
                            "maximum_required_budget_factor"
                        ]
                        for kernel in KERNELS
                    ]
                    global_requirements[
                        method
                    ][confidence_key][
                        tolerance
                    ][key] = {
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

    cp_validation = (
        clopper_pearson_validation(
            intervals=intervals,
        )
    )
    total_budget_valid = all(
        int(
            np.sum(
                allocations[
                    budget_key(factor)
                ][scheme]["totals"]
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
        for scheme in PATH_SCHEMES
    )
    positive_allocations = all(
        np.all(
            allocations[
                budget_key(factor)
            ][scheme]["totals"]
            >= cells["totals"]
        )
        for factor in BUDGET_FACTORS
        for scheme in PATH_SCHEMES
    )
    if not (
        e96_reproduced
        and e97_reproduced
        and total_budget_valid
        and positive_allocations
        and reconstruction_maximum_error
        <= 5e-18
        and cp_validation[
            "maximum_probability_error"
        ]
        <= 1e-10
    ):
        raise RuntimeError(
            "E98 validation failed"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "path_allocation"
        ),
        "source": {
            "e83": (
                "pacing_coalescing_order_full.json"
            ),
            "e89": (
                "pacing_coalescing_order_"
                "prevalence_full.json"
            ),
            "e96": (
                "pacing_coalescing_order_"
                "risk_uncertainty_full.json"
            ),
            "e97": (
                "pacing_coalescing_order_"
                "risk_alpha_allocation_full.json"
            ),
        },
        "model": (
            "Allocate future path comparisons "
            "with outcome-independent E97 "
            "mean-absolute policy geometry. "
            "Projected successes use the E83 "
            "observed rates only for unchanged-"
            "rate planning; actual future "
            "certification must use newly "
            "measured integer outcomes."
        ),
        "budget_factors": list(
            BUDGET_FACTORS
        ),
        "confidence_levels": list(
            CONFIDENCE_LEVELS
        ),
        "methods": list(
            METHODS
        ),
        "path_allocations": (
            path_summaries
        ),
        "headline": {
            "global_worst_accuracy": (
                global_worst
            ),
            "coverage_requirements": (
                global_requirements
            ),
            "path_p90_win_counts": (
                path_win_counts
            ),
            "path_comparison_condition_count": (
                path_condition_count
            ),
        },
        "kernels": kernels,
        "validation": {
            "e89_winner_counts_reproduced": (
                True
            ),
            "e96_uniform_reproduced": (
                e96_reproduced
            ),
            "e97_current_alpha_reproduced": (
                e97_reproduced
            ),
            "total_budgets_exact": (
                total_budget_valid
            ),
            "no_cell_loses_existing_evidence": (
                positive_allocations
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
                "Projected future successes "
                "hold E83 event rates fixed and "
                "round to integer counts; they "
                "are planning sensitivity, not "
                "new measured paths."
            ),
            (
                "Geometry-aware path allocation "
                "is valid for future evidence "
                "only if preregistered before "
                "collecting those outcomes."
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
