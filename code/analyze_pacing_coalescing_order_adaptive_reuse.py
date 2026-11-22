from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri_exp
from scipy.stats import beta, binom, norm

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
    load_json,
)
from analyze_pacing_coalescing_order_pilot_holdout import (
    CONFIDENCE_LEVELS,
    FUTURE_FACTORS,
    build_pilot_scores,
    coverage_requirements,
    factor_key,
    holdout_waterfill,
    worst_accuracy_summary,
)
from analyze_pacing_coalescing_order_prevalence import (
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
    regret_summary,
    robust_regret,
)


METHODS = (
    "future_only_cp",
    "fisher_product",
    "budget_weighted_stouffer",
    "pooled_cp_oracle",
)
VALID_METHODS = METHODS[:-1]
ROOT_EPSILON = 1e-15
ROOT_ABSOLUTE_TOLERANCE = float(
    np.nextafter(0.0, 1.0)
)
ROOT_RELATIVE_TOLERANCE = 1e-14


def load_e102(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_pilot_holdout"
        / "pacing_coalescing_order_pilot_holdout_full.json"
    )


def fisher_product_threshold(
    alpha: float,
) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "Fisher alpha must be in (0, 1)"
        )
    return float(
        brentq(
            lambda value: (
                value
                * (
                    1.0
                    - math.log(value)
                )
                - alpha
            ),
            np.finfo(
                np.float64
            ).tiny,
            alpha,
            xtol=1e-15,
            rtol=1e-14,
        )
    )


def fisher_combined_p(
    first: float,
    second: float,
) -> float:
    product = first * second
    if product <= 0.0:
        return 0.0
    return min(
        1.0,
        product
        * (
            1.0
            - math.log(product)
        ),
    )


def clipped_inverse_survival(
    log_probability: float,
) -> float:
    return float(
        np.clip(
            -ndtri_exp(
                log_probability
            ),
            -40.0,
            40.0,
        )
    )


def stouffer_combined_p(
    first: float,
    second: float,
    future_factor: float,
) -> float:
    first_clipped = min(
        1.0 - 1e-15,
        max(1e-300, first),
    )
    second_clipped = min(
        1.0 - 1e-15,
        max(1e-300, second),
    )
    future_weight = math.sqrt(
        future_factor
    )
    statistic = (
        norm.isf(first_clipped)
        + future_weight
        * norm.isf(
            second_clipped
        )
    ) / math.sqrt(
        1.0 + future_factor
    )
    return float(
        norm.sf(statistic)
    )


def fisher_interval(
    *,
    pilot_successes: int,
    pilot_total: int,
    future_successes: int,
    future_total: int,
    directional_alpha: float,
) -> tuple[float, float, float]:
    threshold = (
        fisher_product_threshold(
            directional_alpha
        )
    )
    log_threshold = math.log(
        threshold
    )

    def upper_tail(
        probability: float,
    ) -> float:
        return (
            binom.logsf(
                pilot_successes - 1,
                pilot_total,
                probability,
            )
            + binom.logsf(
                future_successes - 1,
                future_total,
                probability,
            )
            - log_threshold
        )

    def lower_tail(
        probability: float,
    ) -> float:
        return (
            binom.logcdf(
                pilot_successes,
                pilot_total,
                probability,
            )
            + binom.logcdf(
                future_successes,
                future_total,
                probability,
            )
            - log_threshold
        )

    lower_at_minimum = upper_tail(
        ROOT_EPSILON
    )
    lower_at_maximum = upper_tail(
        1.0 - ROOT_EPSILON
    )
    if lower_at_minimum >= 0.0:
        lower = 0.0
    elif lower_at_maximum <= 0.0:
        lower = 1.0
    else:
        lower = float(
            brentq(
                upper_tail,
                ROOT_EPSILON,
                1.0
                - ROOT_EPSILON,
                xtol=(
                    ROOT_ABSOLUTE_TOLERANCE
                ),
                rtol=(
                    ROOT_RELATIVE_TOLERANCE
                ),
            )
        )

    upper_at_minimum = lower_tail(
        ROOT_EPSILON
    )
    upper_at_maximum = lower_tail(
        1.0 - ROOT_EPSILON
    )
    if upper_at_maximum >= 0.0:
        upper = 1.0
    elif upper_at_minimum <= 0.0:
        upper = 0.0
    else:
        upper = float(
            brentq(
                lower_tail,
                ROOT_EPSILON,
                1.0
                - ROOT_EPSILON,
                xtol=(
                    ROOT_ABSOLUTE_TOLERANCE
                ),
                rtol=(
                    ROOT_RELATIVE_TOLERANCE
                ),
            )
        )

    maximum_residual = 0.0
    if 0.0 < lower < 1.0:
        maximum_residual = max(
            maximum_residual,
            abs(
                upper_tail(lower)
            ),
        )
    if 0.0 < upper < 1.0:
        maximum_residual = max(
            maximum_residual,
            abs(
                lower_tail(upper)
            ),
        )
    return lower, upper, maximum_residual


def stouffer_interval(
    *,
    pilot_successes: int,
    pilot_total: int,
    future_successes: int,
    future_total: int,
    directional_alpha: float,
    future_factor: float,
) -> tuple[float, float, float]:
    future_weight = math.sqrt(
        future_factor
    )
    denominator = math.sqrt(
        1.0 + future_factor
    )
    threshold = norm.isf(
        directional_alpha
    )

    def upper_tail(
        probability: float,
    ) -> float:
        pilot_z = (
            clipped_inverse_survival(
                binom.logsf(
                    pilot_successes
                    - 1,
                    pilot_total,
                    probability,
                )
            )
        )
        future_z = (
            clipped_inverse_survival(
                binom.logsf(
                    future_successes
                    - 1,
                    future_total,
                    probability,
                )
            )
        )
        return (
            (
                pilot_z
                + future_weight
                * future_z
            )
            / denominator
            - threshold
        )

    def lower_tail(
        probability: float,
    ) -> float:
        pilot_z = (
            clipped_inverse_survival(
                binom.logcdf(
                    pilot_successes,
                    pilot_total,
                    probability,
                )
            )
        )
        future_z = (
            clipped_inverse_survival(
                binom.logcdf(
                    future_successes,
                    future_total,
                    probability,
                )
            )
        )
        return (
            (
                pilot_z
                + future_weight
                * future_z
            )
            / denominator
            - threshold
        )

    lower_at_minimum = upper_tail(
        ROOT_EPSILON
    )
    lower_at_maximum = upper_tail(
        1.0 - ROOT_EPSILON
    )
    if lower_at_minimum <= 0.0:
        lower = 0.0
    elif lower_at_maximum >= 0.0:
        lower = 1.0
    else:
        lower = float(
            brentq(
                upper_tail,
                ROOT_EPSILON,
                1.0
                - ROOT_EPSILON,
                xtol=(
                    ROOT_ABSOLUTE_TOLERANCE
                ),
                rtol=(
                    ROOT_RELATIVE_TOLERANCE
                ),
            )
        )

    upper_at_minimum = lower_tail(
        ROOT_EPSILON
    )
    upper_at_maximum = lower_tail(
        1.0 - ROOT_EPSILON
    )
    if upper_at_maximum <= 0.0:
        upper = 1.0
    elif upper_at_minimum >= 0.0:
        upper = 0.0
    else:
        upper = float(
            brentq(
                lower_tail,
                ROOT_EPSILON,
                1.0
                - ROOT_EPSILON,
                xtol=(
                    ROOT_ABSOLUTE_TOLERANCE
                ),
                rtol=(
                    ROOT_RELATIVE_TOLERANCE
                ),
            )
        )

    maximum_residual = 0.0
    if 0.0 < lower < 1.0:
        maximum_residual = max(
            maximum_residual,
            abs(
                upper_tail(lower)
            ),
        )
    if 0.0 < upper < 1.0:
        maximum_residual = max(
            maximum_residual,
            abs(
                lower_tail(upper)
            ),
        )
    return lower, upper, maximum_residual


def exact_adaptive_validation() -> dict[str, Any]:
    alpha = 0.05
    checks = 0
    maximum_excess = {
        "fisher_product": 0.0,
        "budget_weighted_stouffer": (
            0.0
        ),
    }

    def future_total(
        rule: int,
        pilot_successes: int,
        pilot_total: int,
    ) -> int:
        if rule == 0:
            return (
                1 + pilot_successes
            )
        if rule == 1:
            return (
                5
                if 2
                * pilot_successes
                >= pilot_total
                else 1
            )
        return (
            2
            + (
                3
                * pilot_successes
                + pilot_total
            )
            % 4
        )

    for pilot_total in (
        2,
        3,
        4,
        5,
    ):
        for rule in range(3):
            for probability in (
                0.1,
                0.3,
                0.5,
                0.8,
            ):
                for future_factor in (
                    0.5,
                    1.0,
                    2.0,
                    5.0,
                ):
                    rejection = {
                        "fisher_product": {
                            "upper": 0.0,
                            "lower": 0.0,
                        },
                        "budget_weighted_stouffer": {
                            "upper": 0.0,
                            "lower": 0.0,
                        },
                    }
                    for pilot_successes in range(
                        pilot_total + 1
                    ):
                        pilot_mass = binom.pmf(
                            pilot_successes,
                            pilot_total,
                            probability,
                        )
                        holdout_total = (
                            future_total(
                                rule,
                                pilot_successes,
                                pilot_total,
                            )
                        )
                        for future_successes in range(
                            holdout_total + 1
                        ):
                            mass = (
                                pilot_mass
                                * binom.pmf(
                                    future_successes,
                                    holdout_total,
                                    probability,
                                )
                            )
                            pilot_upper = binom.sf(
                                pilot_successes
                                - 1,
                                pilot_total,
                                probability,
                            )
                            future_upper = (
                                binom.sf(
                                    future_successes
                                    - 1,
                                    holdout_total,
                                    probability,
                                )
                            )
                            pilot_lower = (
                                binom.cdf(
                                    pilot_successes,
                                    pilot_total,
                                    probability,
                                )
                            )
                            future_lower = (
                                binom.cdf(
                                    future_successes,
                                    holdout_total,
                                    probability,
                                )
                            )
                            if (
                                fisher_combined_p(
                                    pilot_upper,
                                    future_upper,
                                )
                                <= alpha
                            ):
                                rejection[
                                    "fisher_product"
                                ]["upper"] += mass
                            if (
                                fisher_combined_p(
                                    pilot_lower,
                                    future_lower,
                                )
                                <= alpha
                            ):
                                rejection[
                                    "fisher_product"
                                ]["lower"] += mass
                            if (
                                stouffer_combined_p(
                                    pilot_upper,
                                    future_upper,
                                    future_factor,
                                )
                                <= alpha
                            ):
                                rejection[
                                    "budget_weighted_stouffer"
                                ]["upper"] += mass
                            if (
                                stouffer_combined_p(
                                    pilot_lower,
                                    future_lower,
                                    future_factor,
                                )
                                <= alpha
                            ):
                                rejection[
                                    "budget_weighted_stouffer"
                                ]["lower"] += mass
                    for method in (
                        "fisher_product",
                        "budget_weighted_stouffer",
                    ):
                        for direction in (
                            "upper",
                            "lower",
                        ):
                            maximum_excess[
                                method
                            ] = max(
                                maximum_excess[
                                    method
                                ],
                                rejection[
                                    method
                                ][direction]
                                - alpha,
                            )
                            checks += 1
    return {
        "checks": checks,
        "maximum_size_excess": (
            maximum_excess
        ),
    }


def root_intervals(
    *,
    pilot_successes: np.ndarray,
    pilot_totals: np.ndarray,
    future_successes: np.ndarray,
    future_totals: np.ndarray,
    confidence: float,
    alpha_weights: np.ndarray,
    future_factor: float,
) -> dict[str, Any]:
    fisher_lower = []
    fisher_upper = []
    stouffer_lower = []
    stouffer_upper = []
    fisher_residual = 0.0
    stouffer_residual = 0.0
    directional_alpha = (
        (
            1.0 - confidence
        )
        * alpha_weights
        / 2.0
    )
    for (
        pilot_success,
        pilot_total,
        future_success,
        future_total,
        alpha,
    ) in zip(
        pilot_successes,
        pilot_totals,
        future_successes,
        future_totals,
        directional_alpha,
    ):
        lower, upper, residual = (
            fisher_interval(
                pilot_successes=int(
                    pilot_success
                ),
                pilot_total=int(
                    pilot_total
                ),
                future_successes=int(
                    future_success
                ),
                future_total=int(
                    future_total
                ),
                directional_alpha=float(
                    alpha
                ),
            )
        )
        fisher_lower.append(lower)
        fisher_upper.append(upper)
        fisher_residual = max(
            fisher_residual,
            residual,
        )
        lower, upper, residual = (
            stouffer_interval(
                pilot_successes=int(
                    pilot_success
                ),
                pilot_total=int(
                    pilot_total
                ),
                future_successes=int(
                    future_success
                ),
                future_total=int(
                    future_total
                ),
                directional_alpha=float(
                    alpha
                ),
                future_factor=(
                    future_factor
                ),
            )
        )
        stouffer_lower.append(lower)
        stouffer_upper.append(upper)
        stouffer_residual = max(
            stouffer_residual,
            residual,
        )
    return {
        "fisher_product": {
            "lower": np.asarray(
                fisher_lower,
                dtype=np.float64,
            ),
            "upper": np.asarray(
                fisher_upper,
                dtype=np.float64,
            ),
            "maximum_root_residual": (
                fisher_residual
            ),
        },
        "budget_weighted_stouffer": {
            "lower": np.asarray(
                stouffer_lower,
                dtype=np.float64,
            ),
            "upper": np.asarray(
                stouffer_upper,
                dtype=np.float64,
            ),
            "maximum_root_residual": (
                stouffer_residual
            ),
        },
    }


def global_comparison(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "p90_regret_ratio": (
            candidate[
                "robust_regret_ppm"
            ]["p90"]
            / baseline[
                "robust_regret_ppm"
            ]["p90"]
        ),
        "maximum_regret_ratio": (
            candidate[
                "robust_regret_ppm"
            ]["maximum"]
            / baseline[
                "robust_regret_ppm"
            ]["maximum"]
        ),
        "ten_ppm_coverage_delta": (
            candidate[
                "tolerance_coverage"
            ]["10"]
            - baseline[
                "tolerance_coverage"
            ]["10"]
        ),
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e102 = load_e102(args.root)
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
    point_shape = (
        len(costs),
        len(orders),
        len(metrics),
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
                point_shape
            )
        ),
        pilot_upper=(
            pilot_upper.reshape(
                point_shape
            )
        ),
        e89=e89,
    )

    allocations = {}
    intervals = {
        f"{confidence:.2f}": {}
        for confidence in (
            CONFIDENCE_LEVELS
        )
    }
    maximum_root_residual = {
        "fisher_product": 0.0,
        "budget_weighted_stouffer": (
            0.0
        ),
    }
    for factor in FUTURE_FACTORS:
        key = factor_key(factor)
        allocation = holdout_waterfill(
            point=point,
            base_budget=int(
                np.sum(
                    cells["totals"]
                )
            ),
            future_factor=factor,
            score=pilot[
                "selected_score"
            ],
            alpha_weights=alpha_weights,
            equal_weights=equal_weights,
        )
        allocations[key] = allocation
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            future_lower, future_upper = (
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
            pooled_successes = (
                cells["successes"]
                + allocation[
                    "successes"
                ]
            )
            pooled_totals = (
                cells["totals"]
                + allocation[
                    "totals"
                ]
            )
            pooled_lower, pooled_upper = (
                weighted_intervals(
                    successes=(
                        pooled_successes
                    ),
                    totals=pooled_totals,
                    confidence=confidence,
                    weights=alpha_weights,
                    family_divisor=1,
                )
            )
            roots = root_intervals(
                pilot_successes=(
                    cells["successes"]
                ),
                pilot_totals=(
                    cells["totals"]
                ),
                future_successes=(
                    allocation[
                        "successes"
                    ]
                ),
                future_totals=(
                    allocation[
                        "totals"
                    ]
                ),
                confidence=confidence,
                alpha_weights=(
                    alpha_weights
                ),
                future_factor=factor,
            )
            intervals[
                confidence_key
            ][key] = {
                "future_only_cp": {
                    "lower": (
                        future_lower
                    ),
                    "upper": (
                        future_upper
                    ),
                },
                "fisher_product": (
                    roots[
                        "fisher_product"
                    ]
                ),
                "budget_weighted_stouffer": (
                    roots[
                        "budget_weighted_stouffer"
                    ]
                ),
                "pooled_cp_oracle": {
                    "lower": (
                        pooled_lower
                    ),
                    "upper": (
                        pooled_upper
                    ),
                },
            }
            for method in (
                "fisher_product",
                "budget_weighted_stouffer",
            ):
                maximum_root_residual[
                    method
                ] = max(
                    maximum_root_residual[
                        method
                    ],
                    roots[method][
                        "maximum_root_residual"
                    ],
                )

    kernels = {}
    local_comparison = {
        "conditions": 0,
        "stouffer_p90_wins_vs_future": 0,
        "stouffer_maximum_wins_vs_future": 0,
        "fisher_p90_wins_vs_future": 0,
        "fisher_maximum_wins_vs_future": 0,
        "stouffer_p90_wins_vs_fisher": 0,
        "stouffer_maximum_wins_vs_fisher": 0,
    }
    for kernel_name in KERNELS:
        grid = []
        for row in pilot["cache"][
            kernel_name
        ]:
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
                                selected=row[
                                    "selected"
                                ],
                                coefficients=row[
                                    "coefficients"
                                ],
                                lower=interval[
                                    "lower"
                                ].reshape(
                                    point_shape
                                ),
                                upper=interval[
                                    "upper"
                                ].reshape(
                                    point_shape
                                ),
                            )
                        )
                        method_rows[
                            method
                        ] = regret_summary(
                            regret
                        )
                    method_rows[
                        "stouffer_vs_future"
                    ] = comparison_summary(
                        baseline=method_rows[
                            "future_only_cp"
                        ],
                        candidate=method_rows[
                            "budget_weighted_stouffer"
                        ],
                    )
                    method_rows[
                        "fisher_vs_future"
                    ] = comparison_summary(
                        baseline=method_rows[
                            "future_only_cp"
                        ],
                        candidate=method_rows[
                            "fisher_product"
                        ],
                    )
                    method_rows[
                        "stouffer_vs_oracle"
                    ] = comparison_summary(
                        baseline=method_rows[
                            "pooled_cp_oracle"
                        ],
                        candidate=method_rows[
                            "budget_weighted_stouffer"
                        ],
                    )
                    local_comparison[
                        "conditions"
                    ] += 1
                    comparisons = (
                        (
                            "budget_weighted_stouffer",
                            "future_only_cp",
                            "stouffer",
                            "future",
                        ),
                        (
                            "fisher_product",
                            "future_only_cp",
                            "fisher",
                            "future",
                        ),
                        (
                            "budget_weighted_stouffer",
                            "fisher_product",
                            "stouffer",
                            "fisher",
                        ),
                    )
                    for (
                        candidate,
                        baseline,
                        candidate_label,
                        baseline_label,
                    ) in comparisons:
                        if (
                            method_rows[
                                candidate
                            ][
                                "robust_regret_ppm"
                            ]["p90"]
                            < method_rows[
                                baseline
                            ][
                                "robust_regret_ppm"
                            ]["p90"]
                            - 1e-12
                        ):
                            local_comparison[
                                f"{candidate_label}_p90_wins_vs_{baseline_label}"
                            ] += 1
                        if (
                            method_rows[
                                candidate
                            ][
                                "robust_regret_ppm"
                            ]["maximum"]
                            < method_rows[
                                baseline
                            ][
                                "robust_regret_ppm"
                            ]["maximum"]
                            - 1e-12
                        ):
                            local_comparison[
                                f"{candidate_label}_maximum_wins_vs_{baseline_label}"
                            ] += 1
                    confidence_rows[
                        confidence_key
                    ][key] = method_rows
            grid.append(
                {
                    "accuracy": row[
                        "accuracy"
                    ],
                    "winner_counts": row[
                        "winner_counts"
                    ],
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

    global_comparisons = {
        confidence_key: {
            key: {
                "stouffer_vs_future": (
                    global_comparison(
                        baseline=global_worst[
                            "future_only_cp"
                        ][confidence_key][key],
                        candidate=global_worst[
                            "budget_weighted_stouffer"
                        ][confidence_key][key],
                    )
                ),
                "fisher_vs_future": (
                    global_comparison(
                        baseline=global_worst[
                            "future_only_cp"
                        ][confidence_key][key],
                        candidate=global_worst[
                            "fisher_product"
                        ][confidence_key][key],
                    )
                ),
                "stouffer_vs_oracle": (
                    global_comparison(
                        baseline=global_worst[
                            "pooled_cp_oracle"
                        ][confidence_key][key],
                        candidate=global_worst[
                            "budget_weighted_stouffer"
                        ][confidence_key][key],
                    )
                ),
            }
            for key in (
                factor_key(factor)
                for factor in FUTURE_FACTORS
            )
        }
        for confidence_key in (
            f"{confidence:.2f}"
            for confidence in (
                CONFIDENCE_LEVELS
            )
        )
    }

    future_reproduced = True
    for kernel in KERNELS:
        for index, row in enumerate(
            kernels[kernel][
                "accuracy_grid"
            ]
        ):
            expected_row = e102[
                "kernels"
            ][kernel][
                "accuracy_grid"
            ][index]
            for confidence in (
                CONFIDENCE_LEVELS
            ):
                confidence_key = (
                    f"{confidence:.2f}"
                )
                for factor in (
                    FUTURE_FACTORS
                ):
                    key = factor_key(
                        factor
                    )
                    actual = row[
                        "confidence"
                    ][confidence_key][key][
                        "future_only_cp"
                    ]
                    expected = expected_row[
                        "confidence"
                    ][confidence_key][key][
                        "pilot_selected_holdout"
                    ]
                    future_reproduced &= (
                        all(
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
                        )
                        and all(
                            actual[
                                "tolerance_coverage"
                            ][tolerance][
                                "count"
                            ]
                            == expected[
                                "tolerance_coverage"
                            ][tolerance][
                                "count"
                            ]
                            for tolerance in (
                                TOLERANCE_TARGETS
                            )
                        )
                    )
    adaptive_validation = (
        exact_adaptive_validation()
    )
    threshold_error = max(
        abs(
            fisher_product_threshold(
                alpha
            )
            * (
                1.0
                - math.log(
                    fisher_product_threshold(
                        alpha
                    )
                )
            )
            - alpha
        )
        for alpha in (
            1e-6,
            1e-4,
            0.001,
            0.01,
            0.05,
        )
    )
    family_budget_error = max(
        abs(
            float(
                np.sum(
                    (
                        1.0
                        - confidence
                    )
                    * alpha_weights
                )
            )
            - (
                1.0
                - confidence
            )
        )
        for confidence in (
            CONFIDENCE_LEVELS
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
    validation_state = {
        "future_reproduced": (
            future_reproduced
        ),
        "adaptive_enumeration": (
            max(
                adaptive_validation[
                    "maximum_size_excess"
                ].values()
            )
            <= 1e-12
        ),
        "root_residual": (
            max(
                maximum_root_residual.values()
            )
            <= 1e-7
        ),
        "fisher_threshold": (
            threshold_error <= 1e-12
        ),
        "family_budget": (
            family_budget_error
            <= 1e-12
        ),
        "monotone_p90": monotone_p90,
    }
    if not all(
        validation_state.values()
    ):
        raise RuntimeError(
            "E103 validation failed: "
            f"{validation_state}; "
            f"adaptive={adaptive_validation}; "
            f"roots={maximum_root_residual}"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "adaptive_reuse"
        ),
        "source": {
            "pilot": (
                "pacing_coalescing_order_full.json"
            ),
            "pilot_holdout": (
                "pacing_coalescing_order_"
                "pilot_holdout_full.json"
            ),
        },
        "model": (
            "Combine exact one-sided pilot "
            "and conditional future binomial "
            "p-values. Conditional on the "
            "pilot, future sample sizes and "
            "weights are fixed before future "
            "outcomes. Fisher product and "
            "future-budget-weighted Stouffer "
            "tests are calibrated under "
            "independent uniforms and remain "
            "conservative for sequentially "
            "superuniform exact p-values."
        ),
        "methods": list(METHODS),
        "valid_methods": list(
            VALID_METHODS
        ),
        "invalid_reference": (
            "pooled_cp_oracle"
        ),
        "future_factors": list(
            FUTURE_FACTORS
        ),
        "confidence_levels": list(
            CONFIDENCE_LEVELS
        ),
        "stouffer_weights": {
            factor_key(factor): {
                "pilot": 1.0,
                "future": math.sqrt(
                    factor
                ),
                "normalizer": math.sqrt(
                    1.0 + factor
                ),
            }
            for factor in FUTURE_FACTORS
        },
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
        },
        "kernels": kernels,
        "validation": {
            "future_only_reproduces_e102": (
                future_reproduced
            ),
            "exact_adaptive_design": (
                adaptive_validation
            ),
            "maximum_root_residual": (
                maximum_root_residual
            ),
            "fisher_threshold_maximum_error": (
                threshold_error
            ),
            "family_alpha_budget_maximum_error": (
                family_budget_error
            ),
            "global_p90_monotone": (
                monotone_p90
            ),
        },
        "limitations": [
            (
                "The pooled fixed-sample "
                "Clopper-Pearson oracle is not "
                "valid after pilot-adaptive "
                "sample allocation; it is shown "
                "only as an optimistic reference."
            ),
            (
                "Stouffer weights use the "
                "predeclared global future "
                "budget factor, not pilot-"
                "dependent per-cell sample "
                "sizes. Pilot-dependent weights "
                "would need separate validity "
                "analysis."
            ),
            (
                "The two-stage combinations are "
                "valid for the declared pilot "
                "and one future stage, not "
                "arbitrary repeated peeking."
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
