from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom, norm

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    policy_inventory,
)
from analyze_pacing_coalescing_order_adaptive_reuse import (
    stouffer_combined_p,
    stouffer_interval,
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
    factor_key,
    holdout_waterfill,
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


MULTIPLIERS = (
    1.0 / math.sqrt(2.0),
    1.0,
    math.sqrt(2.0),
    2.0,
    3.0,
)
BASELINE_MULTIPLIER = 1.0
ALPHA = 0.05


def multiplier_key(
    multiplier: float,
) -> str:
    labels = {
        MULTIPLIERS[0]: "inv_sqrt2",
        1.0: "one",
        MULTIPLIERS[2]: "sqrt2",
        2.0: "two",
        3.0: "three",
    }
    return labels[multiplier]


def load_e103(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_adaptive_reuse"
        / "pacing_coalescing_order_adaptive_reuse_full.json"
    )


def exact_fixed_weight_validation() -> dict[str, Any]:
    checks = 0
    maximum_excess = {
        multiplier_key(
            multiplier
        ): 0.0
        for multiplier in MULTIPLIERS
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
                    0.25,
                    0.5,
                    1.0,
                    2.0,
                    5.0,
                    15.0,
                ):
                    rejection = {
                        multiplier_key(
                            multiplier
                        ): {
                            "upper": 0.0,
                            "lower": 0.0,
                        }
                        for multiplier
                        in MULTIPLIERS
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
                            for multiplier in (
                                MULTIPLIERS
                            ):
                                key = (
                                    multiplier_key(
                                        multiplier
                                    )
                                )
                                effective_factor = (
                                    multiplier
                                    * multiplier
                                    * future_factor
                                )
                                if (
                                    stouffer_combined_p(
                                        pilot_upper,
                                        future_upper,
                                        effective_factor,
                                    )
                                    <= ALPHA
                                ):
                                    rejection[key][
                                        "upper"
                                    ] += mass
                                if (
                                    stouffer_combined_p(
                                        pilot_lower,
                                        future_lower,
                                        effective_factor,
                                    )
                                    <= ALPHA
                                ):
                                    rejection[key][
                                        "lower"
                                    ] += mass
                    for multiplier in (
                        MULTIPLIERS
                    ):
                        key = multiplier_key(
                            multiplier
                        )
                        for direction in (
                            "upper",
                            "lower",
                        ):
                            maximum_excess[
                                key
                            ] = max(
                                maximum_excess[
                                    key
                                ],
                                rejection[key][
                                    direction
                                ]
                                - ALPHA,
                            )
                            checks += 1
    return {
        "checks": checks,
        "maximum_size_excess": (
            maximum_excess
        ),
    }


def adaptive_realized_weight_stress() -> dict[str, Any]:
    maximum = {
        "size_excess": (
            -float("inf")
        ),
        "rejection_probability": (
            None
        ),
        "pilot_total": None,
        "threshold": None,
        "large_future_total": None,
        "orientation": None,
        "probability": None,
        "direction": None,
    }
    checks = 0
    for pilot_total in range(2, 13):
        for threshold in range(
            pilot_total + 1
        ):
            for large_future_total in (
                10,
                30,
                100,
            ):
                for orientation in (
                    "small_on_high",
                    "large_on_high",
                ):
                    for probability in np.linspace(
                        0.05,
                        0.95,
                        19,
                    ):
                        rejection = {
                            "upper": 0.0,
                            "lower": 0.0,
                        }
                        for pilot_successes in range(
                            pilot_total + 1
                        ):
                            high = (
                                pilot_successes
                                >= threshold
                            )
                            if (
                                orientation
                                == "small_on_high"
                            ):
                                future_total = (
                                    1
                                    if high
                                    else large_future_total
                                )
                            else:
                                future_total = (
                                    large_future_total
                                    if high
                                    else 1
                                )
                            future_successes = (
                                np.arange(
                                    future_total + 1
                                )
                            )
                            pilot_mass = binom.pmf(
                                pilot_successes,
                                pilot_total,
                                probability,
                            )
                            future_mass = binom.pmf(
                                future_successes,
                                future_total,
                                probability,
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
                                    future_total,
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
                                    future_total,
                                    probability,
                                )
                            )
                            realized_factor = (
                                future_total
                                / pilot_total
                            )
                            weight = math.sqrt(
                                realized_factor
                            )
                            denominator = math.sqrt(
                                1.0
                                + weight * weight
                            )
                            pilot_upper_clip = (
                                np.clip(
                                    pilot_upper,
                                    1e-300,
                                    1.0 - 1e-15,
                                )
                            )
                            future_upper_clip = (
                                np.clip(
                                    future_upper,
                                    1e-300,
                                    1.0 - 1e-15,
                                )
                            )
                            pilot_lower_clip = (
                                np.clip(
                                    pilot_lower,
                                    1e-300,
                                    1.0 - 1e-15,
                                )
                            )
                            future_lower_clip = (
                                np.clip(
                                    future_lower,
                                    1e-300,
                                    1.0 - 1e-15,
                                )
                            )
                            upper_combined = norm.sf(
                                (
                                    norm.isf(
                                        pilot_upper_clip
                                    )
                                    + weight
                                    * norm.isf(
                                        future_upper_clip
                                    )
                                )
                                / denominator
                            )
                            lower_combined = norm.sf(
                                (
                                    norm.isf(
                                        pilot_lower_clip
                                    )
                                    + weight
                                    * norm.isf(
                                        future_lower_clip
                                    )
                                )
                                / denominator
                            )
                            rejection[
                                "upper"
                            ] += (
                                pilot_mass
                                * float(
                                    np.sum(
                                        future_mass
                                        * (
                                            upper_combined
                                            <= ALPHA
                                        )
                                    )
                                )
                            )
                            rejection[
                                "lower"
                            ] += (
                                pilot_mass
                                * float(
                                    np.sum(
                                        future_mass
                                        * (
                                            lower_combined
                                            <= ALPHA
                                        )
                                    )
                                )
                            )
                        for direction in (
                            "upper",
                            "lower",
                        ):
                            excess = (
                                rejection[
                                    direction
                                ]
                                - ALPHA
                            )
                            if (
                                excess
                                > maximum[
                                    "size_excess"
                                ]
                            ):
                                maximum = {
                                    "size_excess": (
                                        excess
                                    ),
                                    "rejection_probability": (
                                        rejection[
                                            direction
                                        ]
                                    ),
                                    "pilot_total": (
                                        pilot_total
                                    ),
                                    "threshold": (
                                        threshold
                                    ),
                                    "large_future_total": (
                                        large_future_total
                                    ),
                                    "orientation": (
                                        orientation
                                    ),
                                    "probability": (
                                        float(
                                            probability
                                        )
                                    ),
                                    "direction": (
                                        direction
                                    ),
                                }
                            checks += 1
    return {
        "checks": checks,
        "maximum": maximum,
    }


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


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e103 = load_e103(args.root)
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
        "individual": {
            multiplier_key(
                multiplier
            ): 0.0
            for multiplier in (
                MULTIPLIERS
            )
        },
        "selection_safe": {
            multiplier_key(
                multiplier
            ): 0.0
            for multiplier in (
                MULTIPLIERS
            )
        },
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
            intervals[
                confidence_key
            ][key] = {
                "individual": {},
                "selection_safe": {},
            }
            for multiplier in MULTIPLIERS:
                multiplier_name = (
                    multiplier_key(
                        multiplier
                    )
                )
                effective_factor = (
                    multiplier
                    * multiplier
                    * factor
                )
                for mode, divisor in (
                    ("individual", 1),
                    (
                        "selection_safe",
                        len(MULTIPLIERS),
                    ),
                ):
                    lower = []
                    upper = []
                    residual = 0.0
                    for (
                        pilot_success,
                        pilot_total,
                        future_success,
                        future_total,
                        alpha_weight,
                    ) in zip(
                        cells["successes"],
                        cells["totals"],
                        allocation[
                            "successes"
                        ],
                        allocation["totals"],
                        alpha_weights,
                    ):
                        (
                            cell_lower,
                            cell_upper,
                            cell_residual,
                        ) = stouffer_interval(
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
                            directional_alpha=(
                                (
                                    1.0
                                    - confidence
                                )
                                * float(
                                    alpha_weight
                                )
                                / (
                                    2.0
                                    * divisor
                                )
                            ),
                            future_factor=(
                                effective_factor
                            ),
                        )
                        lower.append(
                            cell_lower
                        )
                        upper.append(
                            cell_upper
                        )
                        residual = max(
                            residual,
                            cell_residual,
                        )
                    maximum_root_residual[
                        mode
                    ][multiplier_name] = max(
                        maximum_root_residual[
                            mode
                        ][multiplier_name],
                        residual,
                    )
                    intervals[
                        confidence_key
                    ][key][mode][
                        multiplier_name
                    ] = {
                        "lower": np.asarray(
                            lower,
                            dtype=np.float64,
                        ),
                        "upper": np.asarray(
                            upper,
                            dtype=np.float64,
                        ),
                    }

    kernels = {}
    individual_wins = {
        multiplier_key(
            multiplier
        ): {
            "p90": 0,
            "maximum": 0,
        }
        for multiplier in MULTIPLIERS
    }
    selection_counts = {
        multiplier_key(
            multiplier
        ): 0
        for multiplier in MULTIPLIERS
    }
    condition_count = 0
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
                    candidate_rows = {}
                    safe_regrets = []
                    safe_names = []
                    for multiplier in (
                        MULTIPLIERS
                    ):
                        name = multiplier_key(
                            multiplier
                        )
                        individual = intervals[
                            confidence_key
                        ][key][
                            "individual"
                        ][name]
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
                                lower=individual[
                                    "lower"
                                ].reshape(
                                    point_shape
                                ),
                                upper=individual[
                                    "upper"
                                ].reshape(
                                    point_shape
                                ),
                            )
                        )
                        candidate_rows[
                            name
                        ] = regret_summary(
                            regret
                        )
                        safe_interval = intervals[
                            confidence_key
                        ][key][
                            "selection_safe"
                        ][name]
                        safe_regret, _ = (
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
                                lower=safe_interval[
                                    "lower"
                                ].reshape(
                                    point_shape
                                ),
                                upper=safe_interval[
                                    "upper"
                                ].reshape(
                                    point_shape
                                ),
                            )
                        )
                        safe_regrets.append(
                            safe_regret
                        )
                        safe_names.append(name)
                    p90_values = np.asarray(
                        [
                            candidate_rows[name][
                                "robust_regret_ppm"
                            ]["p90"]
                            for name in safe_names
                        ]
                    )
                    maximum_values = np.asarray(
                        [
                            candidate_rows[name][
                                "robust_regret_ppm"
                            ]["maximum"]
                            for name in safe_names
                        ]
                    )
                    for index in np.flatnonzero(
                        np.abs(
                            p90_values
                            - np.min(p90_values)
                        )
                        <= 1e-12
                    ):
                        individual_wins[
                            safe_names[
                                int(index)
                            ]
                        ]["p90"] += 1
                    for index in np.flatnonzero(
                        np.abs(
                            maximum_values
                            - np.min(
                                maximum_values
                            )
                        )
                        <= 1e-12
                    ):
                        individual_wins[
                            safe_names[
                                int(index)
                            ]
                        ]["maximum"] += 1
                    safe_array = np.asarray(
                        safe_regrets
                    )
                    safe_choice = np.argmin(
                        safe_array,
                        axis=0,
                    )
                    safe_best = safe_array[
                        safe_choice,
                        np.arange(
                            len(profiles)
                        ),
                    ]
                    safe_summary = (
                        regret_summary(
                            safe_best
                        )
                    )
                    safe_summary[
                        "selected_multiplier_counts"
                    ] = {
                        name: int(
                            np.count_nonzero(
                                safe_choice
                                == index
                            )
                        )
                        for index, name
                        in enumerate(
                            safe_names
                        )
                    }
                    for index, name in enumerate(
                        safe_names
                    ):
                        selection_counts[
                            name
                        ] += int(
                            np.count_nonzero(
                                safe_choice
                                == index
                            )
                        )
                    candidate_rows[
                        "selection_safe_best"
                    ] = safe_summary
                    candidate_rows[
                        "sqrt2_vs_baseline"
                    ] = comparison_summary(
                        baseline=candidate_rows[
                            "one"
                        ],
                        candidate=candidate_rows[
                            "sqrt2"
                        ],
                    )
                    confidence_rows[
                        confidence_key
                    ][key] = candidate_rows
                    condition_count += 1
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
        method_names = [
            multiplier_key(multiplier)
            for multiplier in MULTIPLIERS
        ] + [
            "selection_safe_best"
        ]
        kernels[kernel_name] = {
            "accuracy_grid": grid,
            "worst_accuracy": {
                method: (
                    worst_accuracy_summary(
                        grid=grid,
                        method=method,
                    )
                )
                for method in method_names
            },
            "coverage_requirements": {
                method: (
                    coverage_requirements(
                        grid=grid,
                        method=method,
                    )
                )
                for method in method_names
            },
        }

    method_names = [
        multiplier_key(multiplier)
        for multiplier in MULTIPLIERS
    ] + [
        "selection_safe_best"
    ]
    global_worst = {}
    global_requirements = {}
    for method in method_names:
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

    baseline_reproduced = True
    for kernel in KERNELS:
        for index, row in enumerate(
            kernels[kernel][
                "accuracy_grid"
            ]
        ):
            expected_row = e103[
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
                        "one"
                    ]
                    expected = expected_row[
                        "confidence"
                    ][confidence_key][key][
                        "budget_weighted_stouffer"
                    ]
                    baseline_reproduced &= (
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
    fixed_validation = (
        exact_fixed_weight_validation()
    )
    adaptive_stress = (
        adaptive_realized_weight_stress()
    )
    maximum_residual = max(
        residual
        for mode in (
            maximum_root_residual.values()
        )
        for residual in mode.values()
    )
    family_budget_error = max(
        abs(
            (
                1.0
                - confidence
            )
            / len(MULTIPLIERS)
            * len(MULTIPLIERS)
            - (
                1.0 - confidence
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
        for method in method_names
        for confidence in (
            CONFIDENCE_LEVELS
        )
    )
    if not (
        baseline_reproduced
        and max(
            fixed_validation[
                "maximum_size_excess"
            ].values()
        )
        <= 1e-12
        and adaptive_stress[
            "maximum"
        ]["size_excess"]
        >= 1e-3
        and maximum_residual <= 1e-7
        and family_budget_error
        <= 1e-12
        and monotone_p90
    ):
        raise RuntimeError(
            "E104 validation failed"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "stage_weights"
        ),
        "source": {
            "pilot": (
                "pacing_coalescing_order_full.json"
            ),
            "adaptive_reuse": (
                "pacing_coalescing_order_"
                "adaptive_reuse_full.json"
            ),
        },
        "model": (
            "Compare fixed Stouffer future-"
            "weight multipliers applied to "
            "sqrt(global future factor). Each "
            "fixed multiplier is a valid "
            "predeclared alternative. A best-"
            "of-five certificate divides alpha "
            "across candidates. A separate "
            "exact stress test demonstrates "
            "that weights based on pilot-"
            "dependent realized per-cell sample "
            "sizes can exceed nominal type-I "
            "error."
        ),
        "multipliers": [
            {
                "key": multiplier_key(
                    multiplier
                ),
                "value": multiplier,
            }
            for multiplier in MULTIPLIERS
        ],
        "baseline_multiplier": (
            multiplier_key(
                BASELINE_MULTIPLIER
            )
        ),
        "future_factors": list(
            FUTURE_FACTORS
        ),
        "confidence_levels": list(
            CONFIDENCE_LEVELS
        ),
        "headline": {
            "global_worst_accuracy": (
                global_worst
            ),
            "coverage_requirements": (
                global_requirements
            ),
            "individual_winner_counts": (
                individual_wins
            ),
            "selection_safe_profile_choices": (
                selection_counts
            ),
            "condition_count": (
                condition_count
            ),
        },
        "kernels": kernels,
        "validation": {
            "baseline_reproduces_e103": (
                baseline_reproduced
            ),
            "fixed_weight_exact_adaptive_design": (
                fixed_validation
            ),
            "adaptive_realized_weight_stress": (
                adaptive_stress
            ),
            "maximum_root_residual": (
                maximum_root_residual
            ),
            "selection_safe_family_alpha_error": (
                family_budget_error
            ),
            "global_p90_monotone": (
                monotone_p90
            ),
        },
        "limitations": [
            (
                "Individual multiplier results "
                "are valid only when the "
                "multiplier is selected before "
                "observing the pilot. The same "
                "pilot cannot be used to choose "
                "a full-alpha candidate."
            ),
            (
                "The selection-safe best-of-five "
                "divides family alpha across "
                "candidates and pays a width "
                "tax."
            ),
            (
                "The realized-sample-weight "
                "stress grid gives a concrete "
                "counterexample, not the maximum "
                "possible inflation over all "
                "adaptive rules."
            ),
            (
                "Projected future successes "
                "retain E83 rates and round to "
                "integer counts; they are "
                "planning sensitivity, not new "
                "measured paths."
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
