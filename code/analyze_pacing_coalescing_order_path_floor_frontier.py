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
from analyze_pacing_coalescing_order_path_allocation import (
    BUDGET_FACTORS,
    CONFIDENCE_LEVELS,
    COVERAGE_TARGETS,
    PREVALENCE_DENOMINATOR,
    TOLERANCE_TARGETS,
    budget_key,
    largest_remainder,
    load_json,
    planned_counts,
)
from analyze_pacing_coalescing_order_path_waterfill import (
    BASELINE_METHOD,
    continuous_waterfill,
    integer_waterfill,
    load_e98,
    surrogate_coefficients,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
    prevalence_profiles,
)
from analyze_pacing_coalescing_order_risk_alpha_allocation import (
    allocation_weights,
    geometry_scores,
    load_e89,
    weighted_intervals,
)
from analyze_pacing_coalescing_order_risk_uncertainty import (
    base_cells,
    policy_coefficients,
    selected_policies,
)


FLOOR_SHARES = (
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
)
QUANTILES = (
    ("p50", 0.50),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)


def share_key(
    share: float,
) -> str:
    return (
        str(int(share))
        if float(share).is_integer()
        else f"{share:g}"
    )


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


def floored_waterfill(
    *,
    base_successes: np.ndarray,
    base_totals: np.ndarray,
    budget_factor: float,
    floor_share: float,
    linear_weights: np.ndarray,
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
    additional_budget = (
        target_budget
        - base_budget
    )
    reserved_budget = int(
        math.floor(
            additional_budget
            * floor_share
            + 0.5
        )
    )
    reserved = largest_remainder(
        total=reserved_budget,
        weights=linear_weights,
    )
    lower_bounds = (
        base_totals + reserved
    )
    continuous = continuous_waterfill(
        base_totals=lower_bounds,
        target_budget=target_budget,
        coefficients=coefficients,
    )
    continuous_residual = (
        continuous["totals"]
        - lower_bounds
    )
    residual = np.floor(
        continuous_residual
    ).astype(np.int64)
    remaining = (
        target_budget
        - int(
            np.sum(
                lower_bounds
            )
        )
        - int(
            np.sum(
                residual
            )
        )
    )
    fractions = (
        continuous_residual
        - residual
    )
    order = np.lexsort(
        (
            np.arange(
                len(fractions)
            ),
            -fractions,
        )
    )
    residual[
        order[:remaining]
    ] += 1
    totals = (
        lower_bounds + residual
    )
    rates = (
        base_successes
        / base_totals
    )
    additional = (
        totals - base_totals
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
    donors = totals > lower_bounds
    remove_cost = (
        coefficients[donors]
        * (
            1.0
            / np.sqrt(
                totals[donors] - 1
            )
            - 1.0
            / np.sqrt(
                totals[donors]
            )
        )
    )
    single_transfer_improvement = (
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
    projected_rates = (
        successes / totals
    )
    return {
        "successes": successes,
        "totals": totals,
        "lower_bounds": lower_bounds,
        "additional_totals": (
            additional
        ),
        "reserved_totals": reserved,
        "target_budget": target_budget,
        "additional_budget": (
            additional_budget
        ),
        "reserved_budget": (
            reserved_budget
        ),
        "residual_budget": int(
            np.sum(residual)
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
            single_transfer_improvement
        ),
        "active_cell_count": int(
            np.count_nonzero(
                additional
            )
        ),
        "residual_active_cell_count": int(
            np.count_nonzero(
                residual
            )
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
        "reserved_budget": int(
            allocation[
                "reserved_budget"
            ]
        ),
        "residual_budget": int(
            allocation[
                "residual_budget"
            ]
        ),
        "minimum_total_per_cell": int(
            np.min(totals)
        ),
        "median_total_per_cell": float(
            np.median(totals)
        ),
        "maximum_total_per_cell": int(
            np.max(totals)
        ),
        "active_cell_count": (
            allocation[
                "active_cell_count"
            ]
        ),
        "residual_active_cell_count": (
            allocation[
                "residual_active_cell_count"
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


def robust_regret_many(
    *,
    prevalence: np.ndarray,
    selected: np.ndarray,
    coefficients: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> np.ndarray:
    scheme_count = len(lowers)
    result = np.zeros(
        (
            scheme_count,
            len(prevalence),
        ),
        dtype=np.float64,
    )
    for selected_policy in range(
        len(coefficients)
    ):
        profile_indices = np.flatnonzero(
            selected
            == selected_policy
        )
        if not len(profile_indices):
            continue
        difference = (
            coefficients[
                selected_policy
            ][None, ...]
            - coefficients
        )
        contributions = np.sum(
            np.where(
                difference[
                    None,
                    ...,
                ]
                >= 0.0,
                difference[
                    None,
                    ...,
                ]
                * uppers[
                    :,
                    None,
                    ...,
                ],
                difference[
                    None,
                    ...,
                ]
                * lowers[
                    :,
                    None,
                    ...,
                ],
            ),
            axis=(3, 4),
        )
        risks = np.einsum(
            "nc,spc->spn",
            prevalence[
                profile_indices
            ],
            contributions,
        )
        result[
            :,
            profile_indices,
        ] = np.maximum(
            0.0,
            np.max(
                risks,
                axis=1,
            ),
        )
    return result


def regret_summaries_many(
    regrets: np.ndarray,
) -> list[dict[str, Any]]:
    ppm = regrets * 1_000_000.0
    ordered = np.sort(
        ppm,
        axis=1,
    )
    count = ordered.shape[1]
    summaries = []
    for scheme_index in range(
        len(ordered)
    ):
        robust = {
            name: float(
                ordered[
                    scheme_index,
                    min(
                        count - 1,
                        max(
                            0,
                            math.ceil(
                                probability
                                * count
                            )
                            - 1,
                        ),
                    ),
                ]
            )
            for name, probability
            in QUANTILES
        }
        robust["maximum"] = float(
            ordered[
                scheme_index,
                -1,
            ]
        )
        summaries.append(
            {
                "robust_regret_ppm": (
                    robust
                ),
                "tolerance_coverage": {
                    tolerance: {
                        "count": int(
                            np.count_nonzero(
                                ppm[
                                    scheme_index
                                ]
                                <= float(
                                    tolerance
                                )
                                + 1e-12
                            )
                        ),
                        "fraction": float(
                            np.mean(
                                ppm[
                                    scheme_index
                                ]
                                <= float(
                                    tolerance
                                )
                                + 1e-12
                            )
                        ),
                    }
                    for tolerance in (
                        TOLERANCE_TARGETS
                    )
                },
            }
        )
    return summaries


def update_worst(
    *,
    aggregate: dict[str, Any] | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if aggregate is None:
        return {
            "robust_regret_ppm": dict(
                summary[
                    "robust_regret_ppm"
                ]
            ),
            "tolerance_coverage": {
                tolerance: summary[
                    "tolerance_coverage"
                ][tolerance][
                    "fraction"
                ]
                for tolerance in (
                    TOLERANCE_TARGETS
                )
            },
        }
    for name in (
        "p50",
        "p90",
        "p95",
        "p99",
        "maximum",
    ):
        aggregate[
            "robust_regret_ppm"
        ][name] = max(
            aggregate[
                "robust_regret_ppm"
            ][name],
            summary[
                "robust_regret_ppm"
            ][name],
        )
    for tolerance in (
        TOLERANCE_TARGETS
    ):
        aggregate[
            "tolerance_coverage"
        ][tolerance] = min(
            aggregate[
                "tolerance_coverage"
            ][tolerance],
            summary[
                "tolerance_coverage"
            ][tolerance][
                "fraction"
            ],
        )
    return aggregate


def pareto_indices(
    p90: np.ndarray,
    maximum: np.ndarray,
) -> list[int]:
    result = []
    for left in range(len(p90)):
        dominated = any(
            right != left
            and p90[right]
            <= p90[left] + 1e-12
            and maximum[right]
            <= maximum[left] + 1e-12
            and (
                p90[right]
                < p90[left] - 1e-12
                or maximum[right]
                < maximum[left] - 1e-12
            )
            for right in range(
                len(p90)
            )
        )
        if not dominated:
            result.append(left)
    return result


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
            factor_key = budget_key(
                factor
            )
            for share in FLOOR_SHARES:
                row = intervals[
                    confidence_key
                ][factor_key][
                    share_key(share)
                ]
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


def coverage_requirements(
    *,
    global_worst: dict[str, Any],
    floor_key: str,
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
            coverage = [
                global_worst[
                    confidence_key
                ][factor_key][
                    floor_key
                ][
                    "tolerance_coverage"
                ][tolerance]
                for factor_key in (
                    factor_keys
                )
            ]
            for target in (
                COVERAGE_TARGETS
            ):
                selected = None
                for index, value in enumerate(
                    coverage
                ):
                    if (
                        value >= target
                        and all(
                            later >= target
                            for later in coverage[
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
                result[confidence_key][
                    tolerance
                ][f"{target:.2f}"] = {
                    "maximum_required_budget_factor": (
                        selected
                    )
                }
    return result


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
    coefficients = (
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
    floor_zero_reproduced = True
    floor_one_reproduced = True
    for factor in BUDGET_FACTORS:
        factor_key = budget_key(
            factor
        )
        allocations[factor_key] = {}
        allocation_summaries[
            factor_key
        ] = {}
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            intervals[
                f"{confidence:.2f}"
            ][factor_key] = {}
        for share in FLOOR_SHARES:
            key = share_key(share)
            allocation = (
                floored_waterfill(
                    base_successes=(
                        cells["successes"]
                    ),
                    base_totals=(
                        cells["totals"]
                    ),
                    budget_factor=factor,
                    floor_share=share,
                    linear_weights=(
                        alpha_weights
                    ),
                    coefficients=(
                        coefficients
                    ),
                )
            )
            allocations[
                factor_key
            ][key] = allocation
            allocation_summaries[
                factor_key
            ][key] = allocation_summary(
                allocation=allocation,
                labels=cells["labels"],
            )
            if share == 0.0:
                expected = (
                    integer_waterfill(
                        base_successes=(
                            cells[
                                "successes"
                            ]
                        ),
                        base_totals=(
                            cells["totals"]
                        ),
                        budget_factor=(
                            factor
                        ),
                        coefficients=(
                            coefficients
                        ),
                    )
                )
                floor_zero_reproduced &= (
                    np.array_equal(
                        allocation["totals"],
                        expected["totals"],
                    )
                    and np.array_equal(
                        allocation[
                            "successes"
                        ],
                        expected[
                            "successes"
                        ],
                    )
                )
            if share == 1.0:
                expected = planned_counts(
                    base_successes=(
                        cells["successes"]
                    ),
                    base_totals=(
                        cells["totals"]
                    ),
                    budget_factor=factor,
                    path_weights=(
                        alpha_weights
                    ),
                )
                floor_one_reproduced &= (
                    np.array_equal(
                        allocation["totals"],
                        expected["totals"],
                    )
                    and np.array_equal(
                        allocation[
                            "successes"
                        ],
                        expected[
                            "successes"
                        ],
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
                ][factor_key][key] = {
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
                        alpha_weights
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
    kernel_worst = {
        kernel: {
            f"{confidence:.2f}": {
                budget_key(factor): {
                    share_key(share): None
                    for share in (
                        FLOOR_SHARES
                    )
                }
                for factor in (
                    BUDGET_FACTORS
                )
            }
            for confidence in (
                CONFIDENCE_LEVELS
            )
        }
        for kernel in KERNELS
    }
    local_frontier = {
        "conditions": 0,
        "p90_winner_counts": {
            share_key(share): 0
            for share in FLOOR_SHARES
        },
        "maximum_winner_counts": {
            share_key(share): 0
            for share in FLOOR_SHARES
        },
        "pareto_membership_counts": {
            share_key(share): 0
            for share in FLOOR_SHARES
        },
        "endpoint_only_pareto_count": 0,
        "intermediate_dominated_by_endpoint_counts": {
            share_key(share): 0
            for share in (
                FLOOR_SHARES[1:-1]
            )
        },
    }
    reconstruction_maximum_error = (
        0.0
    )
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        e89_grid = {
            row["accuracy"]: row
            for row in e89["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        for accuracy in (
            index / 100.0
            for index in range(101)
        ):
            policy_coefficients_array = (
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
                policy_coefficients_array
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
            for confidence in (
                CONFIDENCE_LEVELS
            ):
                confidence_key = (
                    f"{confidence:.2f}"
                )
                for factor in (
                    BUDGET_FACTORS
                ):
                    factor_key = (
                        budget_key(
                            factor
                        )
                    )
                    lowers = np.asarray(
                        [
                            intervals[
                                confidence_key
                            ][factor_key][
                                share_key(
                                    share
                                )
                            ]["lower"].reshape(
                                point.shape
                            )
                            for share in (
                                FLOOR_SHARES
                            )
                        ]
                    )
                    uppers = np.asarray(
                        [
                            intervals[
                                confidence_key
                            ][factor_key][
                                share_key(
                                    share
                                )
                            ]["upper"].reshape(
                                point.shape
                            )
                            for share in (
                                FLOOR_SHARES
                            )
                        ]
                    )
                    regrets = (
                        robust_regret_many(
                            prevalence=(
                                prevalence
                            ),
                            selected=selected,
                            coefficients=(
                                policy_coefficients_array
                            ),
                            lowers=lowers,
                            uppers=uppers,
                        )
                    )
                    summaries = (
                        regret_summaries_many(
                            regrets
                        )
                    )
                    for index, share in enumerate(
                        FLOOR_SHARES
                    ):
                        key = share_key(
                            share
                        )
                        kernel_worst[
                            kernel_name
                        ][confidence_key][
                            factor_key
                        ][key] = update_worst(
                            aggregate=(
                                kernel_worst[
                                    kernel_name
                                ][confidence_key][
                                    factor_key
                                ][key]
                            ),
                            summary=(
                                summaries[index]
                            ),
                        )
                    if factor > 1.0:
                        local_frontier[
                            "conditions"
                        ] += 1
                        p90 = np.asarray(
                            [
                                summary[
                                    "robust_regret_ppm"
                                ]["p90"]
                                for summary
                                in summaries
                            ]
                        )
                        maximum = np.asarray(
                            [
                                summary[
                                    "robust_regret_ppm"
                                ]["maximum"]
                                for summary
                                in summaries
                            ]
                        )
                        p90_minimum = float(
                            np.min(p90)
                        )
                        maximum_minimum = (
                            float(
                                np.min(
                                    maximum
                                )
                            )
                        )
                        for index in np.flatnonzero(
                            np.abs(
                                p90
                                - p90_minimum
                            )
                            <= 1e-12
                        ):
                            local_frontier[
                                "p90_winner_counts"
                            ][
                                share_key(
                                    FLOOR_SHARES[
                                        int(index)
                                    ]
                                )
                            ] += 1
                        for index in np.flatnonzero(
                            np.abs(
                                maximum
                                - maximum_minimum
                            )
                            <= 1e-12
                        ):
                            local_frontier[
                                "maximum_winner_counts"
                            ][
                                share_key(
                                    FLOOR_SHARES[
                                        int(index)
                                    ]
                                )
                            ] += 1
                        pareto = pareto_indices(
                            p90,
                            maximum,
                        )
                        for index in pareto:
                            local_frontier[
                                "pareto_membership_counts"
                            ][
                                share_key(
                                    FLOOR_SHARES[
                                        index
                                    ]
                                )
                            ] += 1
                        if set(pareto).issubset(
                            {
                                0,
                                len(
                                    FLOOR_SHARES
                                )
                                - 1,
                            }
                        ):
                            local_frontier[
                                "endpoint_only_pareto_count"
                            ] += 1
                        for index, share in enumerate(
                            FLOOR_SHARES[
                                1:-1
                            ],
                            start=1,
                        ):
                            dominated = any(
                                p90[endpoint]
                                <= p90[index]
                                + 1e-12
                                and maximum[
                                    endpoint
                                ]
                                <= maximum[
                                    index
                                ]
                                + 1e-12
                                and (
                                    p90[
                                        endpoint
                                    ]
                                    < p90[index]
                                    - 1e-12
                                    or maximum[
                                        endpoint
                                    ]
                                    < maximum[
                                        index
                                    ]
                                    - 1e-12
                                )
                                for endpoint in (
                                    0,
                                    len(
                                        FLOOR_SHARES
                                    )
                                    - 1,
                                )
                            )
                            if dominated:
                                local_frontier[
                                    "intermediate_dominated_by_endpoint_counts"
                                ][
                                    share_key(
                                        share
                                    )
                                ] += 1

    global_worst = {
        f"{confidence:.2f}": {
            budget_key(factor): {
                share_key(share): None
                for share in FLOOR_SHARES
            }
            for factor in BUDGET_FACTORS
        }
        for confidence in (
            CONFIDENCE_LEVELS
        )
    }
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        for factor in BUDGET_FACTORS:
            factor_key = budget_key(
                factor
            )
            for share in FLOOR_SHARES:
                key = share_key(share)
                rows = [
                    kernel_worst[kernel][
                        confidence_key
                    ][factor_key][key]
                    for kernel in KERNELS
                ]
                global_worst[
                    confidence_key
                ][factor_key][key] = {
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

    global_frontier = {
        f"{confidence:.2f}": {}
        for confidence in (
            CONFIDENCE_LEVELS
        )
    }
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        for factor in BUDGET_FACTORS:
            factor_key = budget_key(
                factor
            )
            p90 = np.asarray(
                [
                    global_worst[
                        confidence_key
                    ][factor_key][
                        share_key(share)
                    ][
                        "robust_regret_ppm"
                    ]["p90"]
                    for share in (
                        FLOOR_SHARES
                    )
                ]
            )
            maximum = np.asarray(
                [
                    global_worst[
                        confidence_key
                    ][factor_key][
                        share_key(share)
                    ][
                        "robust_regret_ppm"
                    ]["maximum"]
                    for share in (
                        FLOOR_SHARES
                    )
                ]
            )
            global_frontier[
                confidence_key
            ][factor_key] = {
                "pareto_floor_shares": [
                    share_key(
                        FLOOR_SHARES[index]
                    )
                    for index in (
                        pareto_indices(
                            p90,
                            maximum,
                        )
                    )
                ],
                "p90_winner_floor_shares": [
                    share_key(
                        FLOOR_SHARES[
                            int(index)
                        ]
                    )
                    for index in np.flatnonzero(
                        np.abs(
                            p90
                            - np.min(p90)
                        )
                        <= 1e-12
                    )
                ],
                "maximum_winner_floor_shares": [
                    share_key(
                        FLOOR_SHARES[
                            int(index)
                        ]
                    )
                    for index in np.flatnonzero(
                        np.abs(
                            maximum
                            - np.min(
                                maximum
                            )
                        )
                        <= 1e-12
                    )
                ],
            }

    requirements = {
        share_key(share): (
            coverage_requirements(
                global_worst=(
                    global_worst
                ),
                floor_key=(
                    share_key(share)
                ),
            )
        )
        for share in FLOOR_SHARES
    }
    monotone_p90 = all(
        all(
            later
            <= earlier + 1e-9
            for earlier, later in zip(
                [
                    global_worst[
                        f"{confidence:.2f}"
                    ][budget_key(factor)][
                        share_key(share)
                    ][
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
                        share_key(share)
                    ][
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
        for share in FLOOR_SHARES
    )
    e99_reproduced = True
    e98_reproduced = True
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        for factor in BUDGET_FACTORS:
            factor_key = budget_key(
                factor
            )
            for actual, expected in (
                (
                    global_worst[
                        confidence_key
                    ][factor_key]["0"],
                    e99["headline"][
                        "global_worst_accuracy"
                    ][confidence_key][
                        factor_key
                    ],
                ),
                (
                    global_worst[
                        confidence_key
                    ][factor_key]["1"],
                    e98["headline"][
                        "global_worst_accuracy"
                    ][BASELINE_METHOD][
                        confidence_key
                    ][factor_key],
                ),
            ):
                matched = all(
                    abs(
                        actual[
                            "robust_regret_ppm"
                        ][name]
                        - expected[
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
                ) and all(
                    abs(
                        actual[
                            "tolerance_coverage"
                        ][tolerance]
                        - expected[
                            "tolerance_coverage"
                        ][tolerance]
                    )
                    <= 1e-15
                    for tolerance in (
                        TOLERANCE_TARGETS
                    )
                )
                if expected is e99[
                    "headline"
                ][
                    "global_worst_accuracy"
                ][confidence_key][
                    factor_key
                ]:
                    e99_reproduced &= matched
                else:
                    e98_reproduced &= matched

    exact_budgets = all(
        int(
            np.sum(
                allocations[
                    budget_key(factor)
                ][share_key(share)][
                    "totals"
                ]
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
        for share in FLOOR_SHARES
    )
    preserves_evidence = all(
        np.all(
            allocations[
                budget_key(factor)
            ][share_key(share)][
                "totals"
            ]
            >= cells["totals"]
        )
        for factor in BUDGET_FACTORS
        for share in FLOOR_SHARES
    )
    residual_optimality = all(
        allocation_summaries[
            budget_key(factor)
        ][share_key(share)][
            "integer_objective_ratio_vs_continuous_optimum"
        ]
        <= 1.000000001
        and allocation_summaries[
            budget_key(factor)
        ][share_key(share)][
            "single_transfer_maximum_improvement"
        ]
        <= 1e-17
        for factor in BUDGET_FACTORS
        for share in FLOOR_SHARES
    )
    cp_validation = (
        clopper_pearson_validation(
            intervals=intervals,
        )
    )
    validation_state = {
        "floor_zero_allocations": (
            floor_zero_reproduced
        ),
        "floor_one_allocations": (
            floor_one_reproduced
        ),
        "e99_global": e99_reproduced,
        "e98_global": e98_reproduced,
        "exact_budgets": exact_budgets,
        "preserves_evidence": (
            preserves_evidence
        ),
        "residual_optimality": (
            residual_optimality
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
            "E100 validation failed: "
            f"{validation_state}"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "path_floor_frontier"
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
            "Reserve a preregistered share "
            "of every additional path budget "
            "using E98 linear mean-contrast "
            "weights, then minimize E99's "
            "mean-width surrogate over the "
            "remaining budget. Zero floor "
            "reproduces E99 and full floor "
            "reproduces E98."
        ),
        "floor_shares": list(
            FLOOR_SHARES
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
                requirements
            ),
            "global_frontier": (
                global_frontier
            ),
            "local_frontier": (
                local_frontier
            ),
        },
        "kernels": kernel_worst,
        "validation": {
            "e89_winner_counts_reproduced": (
                True
            ),
            "floor_zero_reproduces_e99": (
                floor_zero_reproduced
                and e99_reproduced
            ),
            "floor_one_reproduces_e98": (
                floor_one_reproduced
                and e98_reproduced
            ),
            "total_budgets_exact": (
                exact_budgets
            ),
            "no_cell_loses_existing_evidence": (
                preserves_evidence
            ),
            "residual_waterfill_optimality": (
                residual_optimality
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
                "Floor candidates are "
                "predeclared alternatives. "
                "Choosing a floor after "
                "inspecting the same outcomes "
                "does not create a jointly "
                "selectable confidence family."
            ),
            (
                "Projected future successes "
                "retain E83 rates and round to "
                "integer counts; they are "
                "planning sensitivity, not new "
                "measured paths."
            ),
            (
                "The residual optimization "
                "minimizes E99's mean-width "
                "surrogate, not exact p90 or "
                "maximum robust regret."
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
