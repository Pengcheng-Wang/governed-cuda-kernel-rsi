from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
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
    PREVALENCE_DENOMINATOR,
    TOLERANCE_TARGETS,
    budget_key,
    comparison_summary,
    coverage_requirements,
    load_json,
    planned_counts,
    worst_accuracy_summary,
)
from analyze_pacing_coalescing_order_path_waterfill import (
    BASELINE_METHOD,
    integer_waterfill,
    load_e98,
    surrogate_coefficients,
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
    "all_pair_cost_vertex_minimax_"
    "paths_mean_absolute_contrast_alpha"
)
INITIAL_ACTIVE_COUNT = 12
ADD_ACTIVE_COUNT = 8
SEPARATION_TOLERANCE = 1e-7


def load_e99(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_path_waterfill"
        / "pacing_coalescing_order_path_waterfill_full.json"
    )


def ranked_indices(
    values: np.ndarray,
) -> np.ndarray:
    return np.lexsort(
        (
            np.arange(
                len(values)
            ),
            values,
        )
    )


def build_constraint_matrix(
    *,
    inventory: dict[
        str,
        dict[str, Any],
    ],
    names: list[str],
    costs: list[str],
    orders: list[str],
    metrics: list[str],
    alpha_weights: np.ndarray,
    base_totals: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    if not np.all(
        base_totals
        == base_totals[0]
    ):
        raise RuntimeError(
            "Minimax ratio parameterization "
            "requires equal base totals"
        )
    family_alpha = (
        1.0
        - DESIGN_CONFIDENCE
    )
    width_scale = (
        np.sqrt(
            np.log(
                2.0
                / (
                    family_alpha
                    * alpha_weights
                )
            )
            / 2.0
        )
        * 1_000_000.0
        / np.sqrt(
            base_totals
        )
    )
    pairs = [
        (left, right)
        for left in range(
            len(names)
        )
        for right in range(left)
    ]
    rows = []
    metadata = []
    cells_per_cost = (
        len(orders)
        * len(metrics)
    )
    for kernel_index, kernel in enumerate(
        KERNELS.values()
    ):
        for accuracy_index in range(101):
            coefficients = (
                policy_coefficients(
                    inventory=inventory,
                    names=names,
                    costs=costs,
                    orders=orders,
                    metrics=metrics,
                    accuracy=(
                        accuracy_index
                        / 100.0
                    ),
                    offset_weights=kernel[
                        "offset_weights"
                    ],
                )
            )
            for left, right in pairs:
                difference = np.abs(
                    coefficients[left]
                    - coefficients[right]
                )
                for cost_index in range(
                    len(costs)
                ):
                    row = np.zeros(
                        len(base_totals),
                        dtype=np.float64,
                    )
                    start = (
                        cost_index
                        * cells_per_cost
                    )
                    row[
                        start:
                        start
                        + cells_per_cost
                    ] = difference[
                        cost_index
                    ].reshape(-1)
                    row *= width_scale
                    if np.any(row > 0.0):
                        rows.append(row)
                        metadata.append(
                            (
                                kernel_index,
                                accuracy_index,
                                left,
                                right,
                                cost_index,
                            )
                        )
    return (
        np.asarray(
            rows,
            dtype=np.float64,
        ),
        metadata,
    )


def solve_minimax(
    *,
    constraints: np.ndarray,
    budget_factor: float,
    initial_ratios: np.ndarray,
) -> dict[str, Any]:
    budget = (
        len(initial_ratios)
        * budget_factor
    )
    ratios = initial_ratios.copy()
    values = constraints @ (
        1.0 / np.sqrt(ratios)
    )
    active = [
        int(index)
        for index in ranked_indices(
            values
        )[-INITIAL_ACTIVE_COUNT:]
    ]
    trace = []
    result = None
    for iteration in range(40):
        active_matrix = constraints[
            active
        ]
        initial_epigraph = float(
            np.max(
                active_matrix
                @ (
                    1.0
                    / np.sqrt(ratios)
                )
            )
        )
        initial = np.concatenate(
            (
                ratios,
                [initial_epigraph],
            )
        )

        def objective(
            values_: np.ndarray,
        ) -> float:
            return float(
                values_[-1]
            )

        def objective_jacobian(
            values_: np.ndarray,
        ) -> np.ndarray:
            gradient = np.zeros_like(
                values_
            )
            gradient[-1] = 1.0
            return gradient

        def budget_constraint(
            values_: np.ndarray,
        ) -> float:
            return float(
                np.sum(
                    values_[:-1]
                )
                - budget
            )

        def budget_jacobian(
            values_: np.ndarray,
        ) -> np.ndarray:
            return np.concatenate(
                (
                    np.ones(
                        len(
                            values_
                        )
                        - 1
                    ),
                    [0.0],
                )
            )

        def epigraph_constraints(
            values_: np.ndarray,
        ) -> np.ndarray:
            return (
                values_[-1]
                - active_matrix
                @ (
                    1.0
                    / np.sqrt(
                        values_[:-1]
                    )
                )
            )

        def epigraph_jacobian(
            values_: np.ndarray,
        ) -> np.ndarray:
            return np.column_stack(
                (
                    0.5
                    * active_matrix
                    * np.power(
                        values_[:-1],
                        -1.5,
                    ),
                    np.ones(
                        len(
                            active_matrix
                        )
                    ),
                )
            )

        result = minimize(
            objective,
            initial,
            jac=objective_jacobian,
            bounds=(
                [(1.0, None)]
                * len(ratios)
                + [(0.0, None)]
            ),
            constraints=[
                {
                    "type": "eq",
                    "fun": (
                        budget_constraint
                    ),
                    "jac": (
                        budget_jacobian
                    ),
                },
                {
                    "type": "ineq",
                    "fun": (
                        epigraph_constraints
                    ),
                    "jac": (
                        epigraph_jacobian
                    ),
                },
            ],
            method="SLSQP",
            options={
                "ftol": 1e-10,
                "maxiter": 1000,
                "disp": False,
            },
        )
        if not result.success:
            raise RuntimeError(
                "Minimax active-set solve "
                f"failed: {result.message}"
            )
        ratios = result.x[:-1]
        values = constraints @ (
            1.0 / np.sqrt(ratios)
        )
        full_objective = float(
            np.max(values)
        )
        lower_bound = float(
            result.x[-1]
        )
        separation_gap = (
            full_objective
            - lower_bound
        )
        trace.append(
            {
                "iteration": iteration,
                "active_constraint_count": (
                    len(active)
                ),
                "active_lower_bound_ppm": (
                    lower_bound
                ),
                "full_objective_ppm": (
                    full_objective
                ),
                "separation_gap_ppm": (
                    separation_gap
                ),
            }
        )
        if (
            separation_gap
            <= SEPARATION_TOLERANCE
        ):
            break
        top = ranked_indices(
            values
        )[-ADD_ACTIVE_COUNT:]
        for index in top:
            integer = int(index)
            if integer not in active:
                active.append(integer)
    else:
        raise RuntimeError(
            "Minimax cutting plane did "
            "not converge"
        )
    if result is None:
        raise RuntimeError(
            "Minimax solve produced no result"
        )
    return {
        "ratios": ratios,
        "full_objective_ppm": (
            float(
                np.max(values)
            )
        ),
        "active_lower_bound_ppm": (
            float(
                result.x[-1]
            )
        ),
        "separation_gap_ppm": (
            float(
                np.max(values)
                - result.x[-1]
            )
        ),
        "active_constraints": (
            active
        ),
        "constraint_values": values,
        "trace": trace,
    }


def integerize_allocation(
    *,
    base_successes: np.ndarray,
    base_totals: np.ndarray,
    budget_factor: float,
    ratios: np.ndarray,
) -> dict[str, Any]:
    target_budget = int(
        round(
            np.sum(base_totals)
            * budget_factor
        )
    )
    continuous_totals = (
        ratios * base_totals
    )
    continuous_additional = (
        continuous_totals
        - base_totals
    )
    additional = np.floor(
        np.maximum(
            0.0,
            continuous_additional,
        )
    ).astype(np.int64)
    remaining = (
        target_budget
        - int(
            np.sum(base_totals)
        )
        - int(
            np.sum(additional)
        )
    )
    fractions = (
        continuous_additional
        - additional
    )
    order = np.lexsort(
        (
            np.arange(
                len(fractions)
            ),
            -fractions,
        )
    )
    additional[
        order[:remaining]
    ] += 1
    totals = base_totals + additional
    rates = (
        base_successes
        / base_totals
    )
    additional_successes = np.floor(
        rates * additional + 0.5
    ).astype(np.int64)
    additional_successes = np.minimum(
        additional_successes,
        additional,
    )
    successes = (
        base_successes
        + additional_successes
    )
    projected_rates = (
        successes / totals
    )
    return {
        "successes": successes,
        "totals": totals,
        "additional_successes": (
            additional_successes
        ),
        "additional_totals": (
            additional
        ),
        "target_budget": target_budget,
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
    continuous: dict[str, Any],
    constraints: np.ndarray,
    metadata: list[
        tuple[int, int, int, int, int]
    ],
    labels: list[dict[str, Any]],
    names: list[str],
    costs: list[str],
    e99_ratios: np.ndarray,
    e98_ratios: np.ndarray,
    base_totals: np.ndarray,
) -> dict[str, Any]:
    ratios = (
        allocation["totals"]
        / base_totals
    )
    integer_values = (
        constraints
        @ (
            1.0 / np.sqrt(ratios)
        )
    )
    integer_objective = float(
        np.max(integer_values)
    )
    e99_objective = float(
        np.max(
            constraints
            @ (
                1.0
                / np.sqrt(
                    e99_ratios
                )
            )
        )
    )
    e98_objective = float(
        np.max(
            constraints
            @ (
                1.0
                / np.sqrt(
                    e98_ratios
                )
            )
        )
    )
    additional = allocation[
        "additional_totals"
    ]
    order = np.argsort(
        additional
    )[::-1]
    constraint_order = (
        ranked_indices(
            integer_values
        )[::-1]
    )
    active_constraints = []
    for index in constraint_order[:20]:
        (
            kernel_index,
            accuracy_index,
            left,
            right,
            cost_index,
        ) = metadata[int(index)]
        active_constraints.append(
            {
                "kernel": list(
                    KERNELS
                )[kernel_index],
                "accuracy": (
                    accuracy_index
                    / 100.0
                ),
                "left_policy": names[left],
                "right_policy": (
                    names[right]
                ),
                "actual_cost": (
                    costs[cost_index]
                ),
                "surrogate_ppm": float(
                    integer_values[
                        index
                    ]
                ),
            }
        )
    additional_sum = int(
        np.sum(additional)
    )
    return {
        "target_budget": int(
            allocation[
                "target_budget"
            ]
        ),
        "minimum_total_per_cell": int(
            np.min(
                allocation["totals"]
            )
        ),
        "median_total_per_cell": (
            quantile(
                allocation[
                    "totals"
                ].tolist(),
                0.50,
            )
        ),
        "maximum_total_per_cell": int(
            np.max(
                allocation["totals"]
            )
        ),
        "active_cell_count": int(
            np.count_nonzero(
                additional
            )
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
        "continuous_objective_ppm": (
            continuous[
                "full_objective_ppm"
            ]
        ),
        "active_lower_bound_ppm": (
            continuous[
                "active_lower_bound_ppm"
            ]
        ),
        "separation_gap_ppm": (
            continuous[
                "separation_gap_ppm"
            ]
        ),
        "integer_objective_ppm": (
            integer_objective
        ),
        "integer_to_continuous_ratio": (
            integer_objective
            / continuous[
                "full_objective_ppm"
            ]
        ),
        "objective_ratio_vs_e99": (
            integer_objective
            / e99_objective
        ),
        "objective_ratio_vs_e98": (
            integer_objective
            / e98_objective
        ),
        "cutting_plane_iterations": (
            len(
                continuous["trace"]
            )
        ),
        "active_constraint_count": (
            len(
                continuous[
                    "active_constraints"
                ]
            )
        ),
        "maximum_rate_rounding_error_ppm": (
            allocation[
                "maximum_rate_rounding_error_ppm"
            ]
        ),
        "trace": continuous["trace"],
        "top_constraints": (
            active_constraints
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
                    allocation["totals"][
                        index
                    ]
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
            1.0 - confidence
        )
        for factor in BUDGET_FACTORS:
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


def global_comparison(
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
    e99 = load_e99(args.root)
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
    names = sorted(inventory)
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
    alpha_weights = weights[
        "mean_absolute_contrast"
    ]
    constraints, metadata = (
        build_constraint_matrix(
            inventory=inventory,
            names=names,
            costs=costs,
            orders=orders,
            metrics=metrics,
            alpha_weights=alpha_weights,
            base_totals=(
                cells["totals"]
            ),
        )
    )
    mean_coefficients = (
        surrogate_coefficients(
            contrast=scores[
                "mean_absolute_contrast"
            ].reshape(-1),
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
        factor_key = budget_key(
            factor
        )
        e99_allocation = (
            integer_waterfill(
                base_successes=(
                    cells["successes"]
                ),
                base_totals=(
                    cells["totals"]
                ),
                budget_factor=factor,
                coefficients=(
                    mean_coefficients
                ),
            )
        )
        e98_allocation = planned_counts(
            base_successes=(
                cells["successes"]
            ),
            base_totals=(
                cells["totals"]
            ),
            budget_factor=factor,
            path_weights=alpha_weights,
        )
        if factor == 1.0:
            continuous = {
                "ratios": np.ones(
                    len(
                        cells["totals"]
                    ),
                    dtype=np.float64,
                ),
                "full_objective_ppm": (
                    float(
                        np.max(
                            constraints
                            @ np.ones(
                                len(
                                    cells[
                                        "totals"
                                    ]
                                )
                            )
                        )
                    )
                ),
                "active_lower_bound_ppm": (
                    float(
                        np.max(
                            constraints
                            @ np.ones(
                                len(
                                    cells[
                                        "totals"
                                    ]
                                )
                            )
                        )
                    )
                ),
                "separation_gap_ppm": 0.0,
                "active_constraints": [],
                "constraint_values": (
                    constraints
                    @ np.ones(
                        len(
                            cells["totals"]
                        )
                    )
                ),
                "trace": [],
            }
        else:
            continuous = solve_minimax(
                constraints=constraints,
                budget_factor=factor,
                initial_ratios=(
                    e99_allocation[
                        "totals"
                    ]
                    / cells["totals"]
                ),
            )
        allocation = integerize_allocation(
            base_successes=(
                cells["successes"]
            ),
            base_totals=(
                cells["totals"]
            ),
            budget_factor=factor,
            ratios=continuous[
                "ratios"
            ],
        )
        allocations[factor_key] = (
            allocation
        )
        allocation_summaries[
            factor_key
        ] = allocation_summary(
            allocation=allocation,
            continuous=continuous,
            constraints=constraints,
            metadata=metadata,
            labels=cells["labels"],
            names=names,
            costs=costs,
            e99_ratios=(
                e99_allocation["totals"]
                / cells["totals"]
            ),
            e98_ratios=(
                e98_allocation["totals"]
                / cells["totals"]
            ),
            base_totals=(
                cells["totals"]
            ),
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
            ][factor_key] = {
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
    local_comparison = {
        "conditions": 0,
        "better_than_e99_p90": 0,
        "better_than_e99_maximum": 0,
        "better_than_e98_p90": 0,
        "better_than_e98_maximum": 0,
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
        e99_grid = e99["kernels"][
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
                    interval = intervals[
                        confidence_key
                    ][factor_key]
                    regret, _ = (
                        robust_regret(
                            prevalence=(
                                prevalence
                            ),
                            selected=selected,
                            coefficients=(
                                coefficients
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
                    baseline_e99 = (
                        e99_grid[
                            accuracy_index
                        ]["confidence"][
                            confidence_key
                        ][factor_key][
                            e99["method"]
                        ]
                    )
                    baseline_e98 = (
                        e98_grid[
                            accuracy_index
                        ]["confidence"][
                            confidence_key
                        ][factor_key][
                            BASELINE_METHOD
                        ]
                    )
                    confidence_rows[
                        confidence_key
                    ][factor_key] = {
                        METHOD: summary,
                        "vs_e99": (
                            comparison_summary(
                                baseline=(
                                    baseline_e99
                                ),
                                candidate=summary,
                            )
                        ),
                        "vs_e98": (
                            comparison_summary(
                                baseline=(
                                    baseline_e98
                                ),
                                candidate=summary,
                            )
                        ),
                    }
                    if factor > 1.0:
                        local_comparison[
                            "conditions"
                        ] += 1
                        if (
                            summary[
                                "robust_regret_ppm"
                            ]["p90"]
                            < baseline_e99[
                                "robust_regret_ppm"
                            ]["p90"]
                            - 1e-12
                        ):
                            local_comparison[
                                "better_than_e99_p90"
                            ] += 1
                        if (
                            summary[
                                "robust_regret_ppm"
                            ]["maximum"]
                            < baseline_e99[
                                "robust_regret_ppm"
                            ]["maximum"]
                            - 1e-12
                        ):
                            local_comparison[
                                "better_than_e99_maximum"
                            ] += 1
                        if (
                            summary[
                                "robust_regret_ppm"
                            ]["p90"]
                            < baseline_e98[
                                "robust_regret_ppm"
                            ]["p90"]
                            - 1e-12
                        ):
                            local_comparison[
                                "better_than_e98_p90"
                            ] += 1
                        if (
                            summary[
                                "robust_regret_ppm"
                            ]["maximum"]
                            < baseline_e98[
                                "robust_regret_ppm"
                            ]["maximum"]
                            - 1e-12
                        ):
                            local_comparison[
                                "better_than_e98_maximum"
                            ] += 1
                    if factor == 1.0:
                        current_reproduced &= (
                            all(
                                abs(
                                    summary[
                                        "robust_regret_ppm"
                                    ][name]
                                    - baseline_e99[
                                        "robust_regret_ppm"
                                    ][name]
                                )
                                <= 1e-9
                                for name in (
                                    "p50",
                                    "p90",
                                    "p95",
                                    "p99",
                                    "maximum",
                                )
                            )
                            and all(
                                summary[
                                    "tolerance_coverage"
                                ][tolerance][
                                    "count"
                                ]
                                == baseline_e99[
                                    "tolerance_coverage"
                                ][tolerance][
                                    "count"
                                ]
                                for tolerance in (
                                    TOLERANCE_TARGETS
                                )
                            )
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
        for factor in BUDGET_FACTORS:
            factor_key = budget_key(
                factor
            )
            rows = [
                kernels[kernel][
                    "worst_accuracy"
                ][confidence_key][
                    factor_key
                ]
                for kernel in KERNELS
            ]
            global_worst[
                confidence_key
            ][factor_key] = {
                "robust_regret_ppm": {
                    name: max(
                        row[
                            "robust_regret_ppm"
                        ][name]
                        for row in rows
                    )
                    for name in (
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
                "0.50",
                "0.75",
                "0.90",
                "0.95",
                "0.99",
                "1.00",
            ):
                values = [
                    kernels[kernel][
                        "coverage_requirements"
                    ][confidence_key][
                        tolerance
                    ][target][
                        "maximum_required_budget_factor"
                    ]
                    for kernel in KERNELS
                ]
                global_requirements[
                    confidence_key
                ][tolerance][
                    target
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

    global_comparisons = {
        confidence_key: {
            factor_key: {
                "vs_e99": global_comparison(
                    baseline=e99[
                        "headline"
                    ][
                        "global_worst_accuracy"
                    ][confidence_key][
                        factor_key
                    ],
                    candidate=global_worst[
                        confidence_key
                    ][factor_key],
                ),
                "vs_e98": global_comparison(
                    baseline=e98[
                        "headline"
                    ][
                        "global_worst_accuracy"
                    ][BASELINE_METHOD][
                        confidence_key
                    ][factor_key],
                    candidate=global_worst[
                        confidence_key
                    ][factor_key],
                ),
            }
            for factor_key in (
                budget_key(factor)
                for factor in BUDGET_FACTORS
            )
        }
        for confidence_key in (
            f"{confidence:.2f}"
            for confidence in (
                CONFIDENCE_LEVELS
            )
        )
    }
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
    separation_valid = all(
        allocation_summaries[
            budget_key(factor)
        ]["separation_gap_ppm"]
        <= SEPARATION_TOLERANCE
        + 1e-9
        for factor in BUDGET_FACTORS
    )
    integer_rounding_valid = all(
        allocation_summaries[
            budget_key(factor)
        ][
            "integer_to_continuous_ratio"
        ]
        <= 1.000001
        for factor in BUDGET_FACTORS
    )
    surrogate_dominates = all(
        allocation_summaries[
            budget_key(factor)
        ][
            "objective_ratio_vs_e99"
        ]
        <= 1.0 + 1e-12
        and allocation_summaries[
            budget_key(factor)
        ][
            "objective_ratio_vs_e98"
        ]
        <= 1.0 + 1e-12
        for factor in BUDGET_FACTORS
    )
    monotone_p90 = all(
        all(
            later <= earlier + 1e-9
            for earlier, later in zip(
                [
                    global_worst[
                        f"{confidence:.2f}"
                    ][budget_key(factor)][
                        "robust_regret_ppm"
                    ]["p90"]
                    for factor in (
                        BUDGET_FACTORS
                    )
                ],
                [
                    global_worst[
                        f"{confidence:.2f}"
                    ][budget_key(factor)][
                        "robust_regret_ppm"
                    ]["p90"]
                    for factor in (
                        BUDGET_FACTORS[
                            1:
                        ]
                    )
                ],
            )
        )
        for confidence in (
            CONFIDENCE_LEVELS
        )
    )
    cp_validation = (
        clopper_pearson_validation(
            intervals=intervals,
        )
    )
    validation_state = {
        "current_reproduced": (
            current_reproduced
        ),
        "exact_budgets": exact_budgets,
        "separation_valid": (
            separation_valid
        ),
        "integer_rounding_valid": (
            integer_rounding_valid
        ),
        "surrogate_dominates": (
            surrogate_dominates
        ),
        "monotone_p90": monotone_p90,
        "reconstruction": (
            reconstruction_maximum_error
            <= 5e-18
        ),
        "clopper_pearson": (
            cp_validation[
                "maximum_probability_error"
            ]
            <= 1e-10
        ),
    }
    if not all(
        validation_state.values()
    ):
        raise RuntimeError(
            "E101 validation failed: "
            f"{validation_state}"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "path_minimax"
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
            "e99": (
                "pacing_coalescing_order_"
                "path_waterfill_full.json"
            ),
        },
        "model": (
            "At 95% family confidence with "
            "E97 mean-contrast alpha, minimize "
            "the maximum symmetric Hoeffding "
            "width over every policy pair, "
            "kernel, accuracy, and actual-cost "
            "vertex. The design is fully "
            "outcome-independent but protects "
            "all geometrically possible pairs, "
            "including policies and vertices "
            "that do not control the measured "
            "selection problem."
        ),
        "method": METHOD,
        "design_confidence": (
            DESIGN_CONFIDENCE
        ),
        "constraint_count": int(
            len(constraints)
        ),
        "policy_count": len(names),
        "policy_pair_count": (
            len(names)
            * (
                len(names) - 1
            )
            // 2
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
            "global_comparisons": (
                global_comparisons
            ),
            "local_comparison": (
                local_comparison
            ),
            "e99_baseline": {
                "global_worst_accuracy": (
                    e99["headline"][
                        "global_worst_accuracy"
                    ]
                ),
            },
            "e98_baseline": {
                "global_worst_accuracy": (
                    e98["headline"][
                        "global_worst_accuracy"
                    ][BASELINE_METHOD]
                ),
            },
        },
        "kernels": kernels,
        "validation": {
            "e89_winner_counts_reproduced": (
                True
            ),
            "current_evidence_reproduced": (
                current_reproduced
            ),
            "total_budgets_exact": (
                exact_budgets
            ),
            "cutting_plane_separation_valid": (
                separation_valid
            ),
            "integer_rounding_negligible": (
                integer_rounding_valid
            ),
            "surrogate_no_worse_than_e98_or_e99": (
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
                "The minimax objective protects "
                "all policy pairs and cost "
                "vertices without using observed "
                "selection outcomes. This can "
                "spend evidence on irrelevant "
                "or unreachable comparisons."
            ),
            (
                "The optimized objective is a "
                "symmetric Hoeffding-width "
                "surrogate, not exact "
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
