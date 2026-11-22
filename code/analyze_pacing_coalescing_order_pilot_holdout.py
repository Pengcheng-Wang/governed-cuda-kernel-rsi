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
    COVERAGE_TARGETS,
    PREVALENCE_DENOMINATOR,
    TOLERANCE_TARGETS,
    comparison_summary,
    largest_remainder,
    load_json,
)
from analyze_pacing_coalescing_order_path_waterfill import (
    continuous_waterfill,
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


CONFIDENCE_LEVELS = (
    0.95,
    0.99,
)
FUTURE_FACTORS = (
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    11.0,
    15.0,
)
COMBINED_TOTAL_FACTORS = {
    0.25: "1.25",
    0.5: "1.5",
    1.0: "2",
    2.0: "3",
    3.0: "4",
    5.0: "6",
    7.0: "8",
    11.0: "12",
    15.0: "16",
}
METHODS = (
    "uniform_holdout",
    "geometry_holdout",
    "pilot_selected_holdout",
    "pilot_active_holdout",
)
EQUAL_FLOOR_SHARE = 0.10


def factor_key(
    factor: float,
) -> str:
    return (
        str(int(factor))
        if float(factor).is_integer()
        else f"{factor:g}"
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


def score_summary(
    *,
    score: np.ndarray,
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    flat = score.reshape(-1)
    total = float(
        np.sum(flat)
    )
    if total <= 0.0:
        raise RuntimeError(
            "Pilot score has no mass"
        )
    weights = flat / total
    order = np.argsort(
        weights
    )[::-1]
    return {
        "minimum_weight": float(
            np.min(weights)
        ),
        "maximum_weight": float(
            np.max(weights)
        ),
        "effective_cell_count": (
            1.0
            / float(
                weights @ weights
            )
        ),
        "top_10_share": float(
            np.sum(
                weights[
                    order[:10]
                ]
            )
        ),
        "top_cells": [
            {
                **labels[int(index)],
                "weight": float(
                    weights[index]
                ),
            }
            for index in order[:10]
        ],
    }


def holdout_waterfill(
    *,
    point: np.ndarray,
    base_budget: int,
    future_factor: float,
    score: np.ndarray,
    alpha_weights: np.ndarray,
    equal_weights: np.ndarray,
) -> dict[str, Any]:
    target_budget = int(
        round(
            base_budget
            * future_factor
        )
    )
    floor_budget = int(
        round(
            target_budget
            * EQUAL_FLOOR_SHARE
        )
    )
    floor = largest_remainder(
        total=floor_budget,
        weights=equal_weights,
    )
    coefficients = (
        surrogate_coefficients(
            contrast=score.reshape(-1),
            alpha_weights=alpha_weights,
        )
    )
    continuous = continuous_waterfill(
        base_totals=floor,
        target_budget=target_budget,
        coefficients=coefficients,
    )
    continuous_additional = (
        continuous["totals"]
        - floor
    )
    additional = np.floor(
        continuous_additional
    ).astype(np.int64)
    remaining = (
        target_budget
        - int(
            np.sum(floor)
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
    totals = floor + additional
    successes = np.floor(
        point * totals + 0.5
    ).astype(np.int64)
    successes = np.minimum(
        successes,
        totals,
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
    return {
        "successes": successes,
        "totals": totals,
        "target_budget": target_budget,
        "floor_budget": floor_budget,
        "floor_totals": floor,
        "additional_totals": (
            additional
        ),
        "objective": objective,
        "continuous_objective": (
            continuous_objective
        ),
        "integer_objective_ratio": (
            objective
            / continuous_objective
        ),
        "maximum_rate_rounding_error_ppm": (
            float(
                np.max(
                    np.abs(
                        projected_rates
                        - point
                    )
                )
            )
            * 1_000_000.0
        ),
    }


def uniform_holdout(
    *,
    point: np.ndarray,
    base_budget: int,
    future_factor: float,
    equal_weights: np.ndarray,
) -> dict[str, Any]:
    target_budget = int(
        round(
            base_budget
            * future_factor
        )
    )
    totals = largest_remainder(
        total=target_budget,
        weights=equal_weights,
    )
    successes = np.floor(
        point * totals + 0.5
    ).astype(np.int64)
    successes = np.minimum(
        successes,
        totals,
    )
    projected_rates = (
        successes / totals
    )
    return {
        "successes": successes,
        "totals": totals,
        "target_budget": target_budget,
        "floor_budget": target_budget,
        "floor_totals": totals,
        "additional_totals": (
            np.zeros_like(totals)
        ),
        "objective": None,
        "continuous_objective": None,
        "integer_objective_ratio": None,
        "maximum_rate_rounding_error_ppm": (
            float(
                np.max(
                    np.abs(
                        projected_rates
                        - point
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
    totals = allocation["totals"]
    order = np.argsort(
        totals
    )[::-1]
    return {
        "target_budget": int(
            allocation[
                "target_budget"
            ]
        ),
        "floor_budget": int(
            allocation["floor_budget"]
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
        "active_cell_count": int(
            np.count_nonzero(totals)
        ),
        "top_10_total_share": float(
            np.sum(
                totals[
                    order[:10]
                ]
            )
            / np.sum(totals)
        ),
        "surrogate_objective": (
            allocation["objective"]
        ),
        "integer_objective_ratio_vs_continuous_optimum": (
            allocation[
                "integer_objective_ratio"
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
                "holdout_comparisons": (
                    int(totals[index])
                ),
            }
            for index in order[:10]
        ],
    }


def build_pilot_scores(
    *,
    inventory: dict[
        str,
        dict[str, Any],
    ],
    names: list[str],
    costs: list[str],
    orders: list[str],
    metrics: list[str],
    drift: dict[str, Any],
    profiles: list[dict[str, Any]],
    prevalence: np.ndarray,
    pilot_lower: np.ndarray,
    pilot_upper: np.ndarray,
    e89: dict[str, Any],
) -> dict[str, Any]:
    selected_score = np.zeros_like(
        pilot_lower,
        dtype=np.float64,
    )
    active_score = np.zeros_like(
        pilot_lower,
        dtype=np.float64,
    )
    cache: dict[
        str,
        list[dict[str, Any]],
    ] = {
        kernel: []
        for kernel in KERNELS
    }
    condition_count = 0
    pair_counts: dict[
        tuple[int, int],
        int,
    ] = {}
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        e89_grid = {
            row["accuracy"]: row
            for row in e89["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        for accuracy_index in range(101):
            accuracy = (
                accuracy_index / 100.0
            )
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
            _, worst = robust_regret(
                prevalence=prevalence,
                selected=selected,
                coefficients=coefficients,
                lower=pilot_lower,
                upper=pilot_upper,
            )
            for selected_policy in range(
                len(names)
            ):
                selected_indices = (
                    np.flatnonzero(
                        selected
                        == selected_policy
                    )
                )
                if not len(
                    selected_indices
                ):
                    continue
                mass = np.sum(
                    prevalence[
                        selected_indices
                    ],
                    axis=0,
                )
                mean_competitor_contrast = (
                    np.mean(
                        np.abs(
                            coefficients[
                                selected_policy
                            ][None, ...]
                            - coefficients
                        ),
                        axis=0,
                    )
                )
                selected_score += (
                    mass[
                        :,
                        None,
                        None,
                    ]
                    * mean_competitor_contrast
                )
                for competitor in range(
                    len(names)
                ):
                    pair_indices = np.flatnonzero(
                        (
                            selected
                            == selected_policy
                        )
                        & (
                            worst
                            == competitor
                        )
                    )
                    if not len(
                        pair_indices
                    ):
                        continue
                    pair_mass = np.sum(
                        prevalence[
                            pair_indices
                        ],
                        axis=0,
                    )
                    active_score += (
                        pair_mass[
                            :,
                            None,
                            None,
                        ]
                        * np.abs(
                            coefficients[
                                selected_policy
                            ]
                            - coefficients[
                                competitor
                            ]
                        )
                    )
                    pair_counts[
                        (
                            selected_policy,
                            competitor,
                        )
                    ] = (
                        pair_counts.get(
                            (
                                selected_policy,
                                competitor,
                            ),
                            0,
                        )
                        + len(
                            pair_indices
                        )
                    )
            condition_count += len(
                profiles
            )
            cache[kernel_name].append(
                {
                    "accuracy": accuracy,
                    "coefficients": (
                        coefficients
                    ),
                    "selected": selected,
                    "winner_counts": (
                        winner_counts
                    ),
                }
            )
    selected_score /= condition_count
    active_score /= condition_count
    pair_order = sorted(
        pair_counts,
        key=lambda pair: (
            -pair_counts[pair],
            pair,
        ),
    )
    return {
        "selected_score": (
            selected_score
        ),
        "active_score": active_score,
        "condition_count": (
            condition_count
        ),
        "cache": cache,
        "active_pair_counts": [
            {
                "selected_policy": (
                    names[pair[0]]
                ),
                "competitor": (
                    names[pair[1]]
                ),
                "count": int(
                    pair_counts[pair]
                ),
            }
            for pair in pair_order[:20]
        ],
    }


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
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
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


def coverage_requirements(
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
        for tolerance in (
            TOLERANCE_TARGETS
        ):
            result[confidence_key][
                tolerance
            ] = {}
            for target in (
                COVERAGE_TARGETS
            ):
                values = []
                for row in grid:
                    coverage = [
                        row["confidence"][
                            confidence_key
                        ][factor_key(factor)][
                            method
                        ][
                            "tolerance_coverage"
                        ][tolerance][
                            "fraction"
                        ]
                        for factor in (
                            FUTURE_FACTORS
                        )
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
                                FUTURE_FACTORS[
                                    index
                                ]
                            )
                            break
                    values.append(
                        selected
                    )
                finite = [
                    value
                    for value in values
                    if value is not None
                ]
                result[confidence_key][
                    tolerance
                ][f"{target:.2f}"] = {
                    "maximum_required_future_factor": (
                        None
                        if len(finite)
                        != len(values)
                        else max(finite)
                    ),
                    "accuracy_fraction_certified_at_15x": (
                        len(finite)
                        / len(values)
                    ),
                }
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
        alpha = 1.0 - confidence
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
            for method in METHODS:
                row = intervals[
                    confidence_key
                ][key][method]
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


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e97 = load_e97(args.root)
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
    point = cells["point"]
    point_shaped = point.reshape(
        (
            len(costs),
            len(orders),
            len(metrics),
        )
    )
    base_budget = int(
        np.sum(
            cells["totals"]
        )
    )
    inventory = policy_inventory(
        root=args.root,
        costs=costs,
    )
    names = sorted(inventory)
    geometry = geometry_scores(
        inventory=inventory,
        names=names,
        costs=costs,
        orders=orders,
        metrics=metrics,
    )
    weights = allocation_weights(
        scores=geometry,
        cell_count=len(
            cells["labels"]
        ),
    )
    alpha_weights = weights[
        "mean_absolute_contrast"
    ]
    equal_weights = weights["equal"]
    pilot_lower, pilot_upper = (
        weighted_intervals(
            successes=cells[
                "successes"
            ],
            totals=cells["totals"],
            confidence=0.95,
            weights=alpha_weights,
            family_divisor=1,
        )
    )
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
    pilot = build_pilot_scores(
        inventory=inventory,
        names=names,
        costs=costs,
        orders=orders,
        metrics=metrics,
        drift=drift,
        profiles=profiles,
        prevalence=prevalence,
        pilot_lower=(
            pilot_lower.reshape(
                point_shaped.shape
            )
        ),
        pilot_upper=(
            pilot_upper.reshape(
                point_shaped.shape
            )
        ),
        e89=e89,
    )
    scores = {
        "geometry_holdout": (
            geometry[
                "mean_absolute_contrast"
            ]
        ),
        "pilot_selected_holdout": (
            pilot["selected_score"]
        ),
        "pilot_active_holdout": (
            pilot["active_score"]
        ),
    }
    allocations = {}
    allocation_summaries = {}
    intervals = {
        f"{confidence:.2f}": {}
        for confidence in (
            CONFIDENCE_LEVELS
        )
    }
    for factor in FUTURE_FACTORS:
        key = factor_key(factor)
        allocations[key] = {
            "uniform_holdout": (
                uniform_holdout(
                    point=point,
                    base_budget=base_budget,
                    future_factor=factor,
                    equal_weights=(
                        equal_weights
                    ),
                )
            )
        }
        for method, score in (
            scores.items()
        ):
            allocations[key][method] = (
                holdout_waterfill(
                    point=point,
                    base_budget=base_budget,
                    future_factor=factor,
                    score=score,
                    alpha_weights=(
                        alpha_weights
                    ),
                    equal_weights=(
                        equal_weights
                    ),
                )
            )
        allocation_summaries[key] = {
            method: allocation_summary(
                allocation=allocation,
                labels=cells["labels"],
            )
            for method, allocation in (
                allocations[key].items()
            )
        }
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            intervals[
                confidence_key
            ][key] = {}
            for method in METHODS:
                allocation = (
                    allocations[key][
                        method
                    ]
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
                ][key][method] = {
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
                    "summary": (
                        interval_summary(
                            lower=lower,
                            upper=upper,
                        )
                    ),
                }

    kernels = {}
    local_comparison = {
        "conditions": 0,
        "pilot_selected_p90_wins_vs_geometry": 0,
        "pilot_selected_maximum_wins_vs_geometry": 0,
        "pilot_active_p90_wins_vs_geometry": 0,
        "pilot_active_maximum_wins_vs_geometry": 0,
    }
    reconstruction_maximum_error = (
        0.0
    )
    for kernel_name in KERNELS:
        grid = []
        for row in pilot["cache"][
            kernel_name
        ]:
            accuracy = row["accuracy"]
            coefficients = row[
                "coefficients"
            ]
            selected = row["selected"]
            per_policy_costs = {
                name: per_cost_metrics(
                    drift=drift,
                    costs=costs,
                    mapping=inventory[
                        name
                    ]["mapping"],
                    accuracy=accuracy,
                    offset_weights=KERNELS[
                        kernel_name
                    ]["offset_weights"],
                )
                for name in names
            }
            reconstructed = np.sum(
                coefficients
                * point_shaped[
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
                        item[
                            "regression_fraction"
                        ]
                        for item
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
                    FUTURE_FACTORS
                ):
                    key = factor_key(
                        factor
                    )
                    method_rows = {}
                    for method in METHODS:
                        interval = intervals[
                            confidence_key
                        ][key][method]
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
                                lower=interval[
                                    "lower"
                                ].reshape(
                                    point_shaped.shape
                                ),
                                upper=interval[
                                    "upper"
                                ].reshape(
                                    point_shaped.shape
                                ),
                            )
                        )
                        method_rows[
                            method
                        ] = regret_summary(
                            regret
                        )
                    method_rows[
                        "pilot_selected_vs_geometry"
                    ] = comparison_summary(
                        baseline=method_rows[
                            "geometry_holdout"
                        ],
                        candidate=method_rows[
                            "pilot_selected_holdout"
                        ],
                    )
                    method_rows[
                        "pilot_active_vs_geometry"
                    ] = comparison_summary(
                        baseline=method_rows[
                            "geometry_holdout"
                        ],
                        candidate=method_rows[
                            "pilot_active_holdout"
                        ],
                    )
                    local_comparison[
                        "conditions"
                    ] += 1
                    if (
                        method_rows[
                            "pilot_selected_holdout"
                        ][
                            "robust_regret_ppm"
                        ]["p90"]
                        < method_rows[
                            "geometry_holdout"
                        ][
                            "robust_regret_ppm"
                        ]["p90"]
                        - 1e-12
                    ):
                        local_comparison[
                            "pilot_selected_p90_wins_vs_geometry"
                        ] += 1
                    if (
                        method_rows[
                            "pilot_selected_holdout"
                        ][
                            "robust_regret_ppm"
                        ]["maximum"]
                        < method_rows[
                            "geometry_holdout"
                        ][
                            "robust_regret_ppm"
                        ]["maximum"]
                        - 1e-12
                    ):
                        local_comparison[
                            "pilot_selected_maximum_wins_vs_geometry"
                        ] += 1
                    if (
                        method_rows[
                            "pilot_active_holdout"
                        ][
                            "robust_regret_ppm"
                        ]["p90"]
                        < method_rows[
                            "geometry_holdout"
                        ][
                            "robust_regret_ppm"
                        ]["p90"]
                        - 1e-12
                    ):
                        local_comparison[
                            "pilot_active_p90_wins_vs_geometry"
                        ] += 1
                    if (
                        method_rows[
                            "pilot_active_holdout"
                        ][
                            "robust_regret_ppm"
                        ]["maximum"]
                        < method_rows[
                            "geometry_holdout"
                        ][
                            "robust_regret_ppm"
                        ]["maximum"]
                        - 1e-12
                    ):
                        local_comparison[
                            "pilot_active_maximum_wins_vs_geometry"
                        ] += 1
                    confidence_rows[
                        confidence_key
                    ][key] = method_rows
            grid.append(
                {
                    "accuracy": accuracy,
                    "winner_counts": (
                        row[
                            "winner_counts"
                        ]
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
    global_requirements = {}
    for method in METHODS:
        global_worst[method] = {}
        global_requirements[method] = {}
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            global_worst[method][
                confidence_key
            ] = {}
            global_requirements[method][
                confidence_key
            ] = {}
            for factor in (
                FUTURE_FACTORS
            ):
                key = factor_key(
                    factor
                )
                rows = [
                    kernels[kernel][
                        "worst_accuracy"
                    ][method][
                        confidence_key
                    ][key]
                    for kernel in KERNELS
                ]
                global_worst[method][
                    confidence_key
                ][key] = {
                    "robust_regret_ppm": {
                        name: max(
                            item[
                                "robust_regret_ppm"
                            ][name]
                            for item in rows
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
                            item[
                                "tolerance_coverage"
                            ][tolerance]
                            for item in rows
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
                    method
                ][confidence_key][
                    tolerance
                ] = {}
                for target in (
                    COVERAGE_TARGETS
                ):
                    target_key = (
                        f"{target:.2f}"
                    )
                    values = [
                        kernels[kernel][
                            "coverage_requirements"
                        ][method][
                            confidence_key
                        ][tolerance][
                            target_key
                        ][
                            "maximum_required_future_factor"
                        ]
                        for kernel in KERNELS
                    ]
                    global_requirements[
                        method
                    ][confidence_key][
                        tolerance
                    ][target_key] = {
                        "maximum_required_future_factor": (
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

    combined_reuse_gap = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        combined_reuse_gap[
            confidence_key
        ] = {}
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
            total_key = (
                COMBINED_TOTAL_FACTORS[
                    factor
                ]
            )
            holdout = global_worst[
                "pilot_selected_holdout"
            ][confidence_key][key]
            combined = e99["headline"][
                "global_worst_accuracy"
            ][confidence_key][total_key]
            combined_reuse_gap[
                confidence_key
            ][key] = {
                "combined_total_factor": (
                    float(total_key)
                ),
                "holdout_only_p90_ratio": (
                    holdout[
                        "robust_regret_ppm"
                    ]["p90"]
                    / combined[
                        "robust_regret_ppm"
                    ]["p90"]
                ),
                "holdout_only_maximum_ratio": (
                    holdout[
                        "robust_regret_ppm"
                    ]["maximum"]
                    / combined[
                        "robust_regret_ppm"
                    ]["maximum"]
                ),
                "ten_ppm_coverage_delta": (
                    holdout[
                        "tolerance_coverage"
                    ]["10"]
                    - combined[
                        "tolerance_coverage"
                    ]["10"]
                ),
            }

    future_only_valid = all(
        np.array_equal(
            allocation["successes"],
            np.minimum(
                allocation["totals"],
                np.floor(
                    point
                    * allocation["totals"]
                    + 0.5
                ).astype(np.int64),
            ),
        )
        for factor in FUTURE_FACTORS
        for allocation in (
            allocations[
                factor_key(factor)
            ].values()
        )
    )
    exact_budgets = all(
        int(
            np.sum(
                allocation["totals"]
            )
        )
        == int(
            round(
                base_budget * factor
            )
        )
        for factor in FUTURE_FACTORS
        for allocation in (
            allocations[
                factor_key(factor)
            ].values()
        )
    )
    positive_counts = all(
        np.all(
            allocation["totals"] > 0
        )
        for factor in FUTURE_FACTORS
        for allocation in (
            allocations[
                factor_key(factor)
            ].values()
        )
    )
    e97_reproduced = True
    for kernel in KERNELS:
        for index, row in enumerate(
            kernels[kernel][
                "accuracy_grid"
            ]
        ):
            expected = e97["kernels"][
                kernel
            ]["accuracy_grid"][index][
                "confidence"
            ]
            for confidence in (
                CONFIDENCE_LEVELS
            ):
                confidence_key = (
                    f"{confidence:.2f}"
                )
                actual = row[
                    "confidence"
                ][confidence_key]["1"][
                    "uniform_holdout"
                ]
                target = expected[
                    confidence_key
                ]["1"][
                    "mean_absolute_contrast"
                ]
                e97_reproduced &= (
                    all(
                        abs(
                            actual[
                                "robust_regret_ppm"
                            ][name]
                            - target[
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
                        actual[
                            "tolerance_coverage"
                        ][tolerance][
                            "count"
                        ]
                        == target[
                            "tolerance_coverage"
                        ][tolerance][
                            "count"
                        ]
                        for tolerance in (
                            TOLERANCE_TARGETS
                        )
                    )
                )
    cp_validation = (
        clopper_pearson_validation(
            intervals=intervals,
        )
    )
    monotone_p90 = all(
        all(
            later <= earlier + 1e-9
            for earlier, later in zip(
                [
                    global_worst[method][
                        f"{confidence:.2f}"
                    ][factor_key(factor)][
                        "robust_regret_ppm"
                    ]["p90"]
                    for factor in (
                        FUTURE_FACTORS
                    )
                ],
                [
                    global_worst[method][
                        f"{confidence:.2f}"
                    ][factor_key(factor)][
                        "robust_regret_ppm"
                    ]["p90"]
                    for factor in (
                        FUTURE_FACTORS[
                            1:
                        ]
                    )
                ],
            )
        )
        for method in METHODS
        for confidence in (
            CONFIDENCE_LEVELS
        )
    )
    if not (
        future_only_valid
        and exact_budgets
        and positive_counts
        and e97_reproduced
        and monotone_p90
        and reconstruction_maximum_error
        <= 5e-18
        and cp_validation[
            "maximum_probability_error"
        ]
        <= 1e-10
    ):
        raise RuntimeError(
            "E102 validation failed"
        )

    geometry_score = geometry[
        "mean_absolute_contrast"
    ]
    result = {
        "suite": (
            "pacing_coalescing_order_"
            "pilot_holdout"
        ),
        "source": {
            "pilot": (
                "pacing_coalescing_order_full.json"
            ),
            "policy_selection": (
                "pacing_coalescing_order_"
                "prevalence_full.json"
            ),
            "alpha_allocation": (
                "pacing_coalescing_order_"
                "risk_alpha_allocation_full.json"
            ),
            "combined_data_baseline": (
                "pacing_coalescing_order_"
                "path_waterfill_full.json"
            ),
        },
        "model": (
            "Use E83 as a fixed pilot to "
            "select deployment policies and "
            "derive task-aware future sampling. "
            "Construct final simultaneous "
            "Clopper-Pearson intervals from "
            "future holdout counts only. "
            "Conditional on the pilot, the "
            "allocation and selected policies "
            "are fixed before holdout outcomes."
        ),
        "future_factors": list(
            FUTURE_FACTORS
        ),
        "confidence_levels": list(
            CONFIDENCE_LEVELS
        ),
        "equal_floor_share": (
            EQUAL_FLOOR_SHARE
        ),
        "methods": list(METHODS),
        "pilot_scores": {
            "condition_count": (
                pilot[
                    "condition_count"
                ]
            ),
            "geometry": score_summary(
                score=geometry_score,
                labels=cells["labels"],
            ),
            "pilot_selected": (
                score_summary(
                    score=pilot[
                        "selected_score"
                    ],
                    labels=cells[
                        "labels"
                    ],
                )
            ),
            "pilot_active": (
                score_summary(
                    score=pilot[
                        "active_score"
                    ],
                    labels=cells[
                        "labels"
                    ],
                )
            ),
            "selected_geometry_correlation": (
                float(
                    np.corrcoef(
                        pilot[
                            "selected_score"
                        ].reshape(-1),
                        geometry_score.reshape(
                            -1
                        ),
                    )[0, 1]
                )
            ),
            "active_geometry_correlation": (
                float(
                    np.corrcoef(
                        pilot[
                            "active_score"
                        ].reshape(-1),
                        geometry_score.reshape(
                            -1
                        ),
                    )[0, 1]
                )
            ),
            "active_pair_counts": (
                pilot[
                    "active_pair_counts"
                ]
            ),
        },
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
            "local_comparison": (
                local_comparison
            ),
            "combined_reuse_gap": (
                combined_reuse_gap
            ),
        },
        "kernels": kernels,
        "validation": {
            "future_intervals_use_holdout_counts_only": (
                future_only_valid
            ),
            "total_future_budgets_exact": (
                exact_budgets
            ),
            "all_cells_receive_holdout_counts": (
                positive_counts
            ),
            "e89_pilot_winner_counts_reproduced": (
                True
            ),
            "uniform_one_x_reproduces_e97_mean_alpha": (
                e97_reproduced
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
                "The holdout study projects "
                "future counts at E83 rates; "
                "these are planning sensitivity, "
                "not new measured paths."
            ),
            (
                "Pilot-selected policies and "
                "scores are valid for future-only "
                "certification because no E83 "
                "pilot counts enter the final "
                "intervals. Reusing the pilot "
                "counts would require adaptive "
                "or selection-adjusted inference."
            ),
            (
                "The selected-policy score "
                "averages contrast to every "
                "competitor; the active score "
                "uses only the current robust "
                "competitor and can overfocus."
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
