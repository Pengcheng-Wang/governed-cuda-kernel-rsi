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
    REPRESENTATIVE_ACCURACIES,
    quantile,
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
EVIDENCE_MULTIPLIERS = (
    1,
    2,
    4,
    8,
    16,
)
SCHEMES = (
    "equal",
    "mean_absolute_contrast",
    "rms_contrast",
    "maximum_contrast",
)
GEOMETRY_SHARE = 0.90
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


def geometry_scores(
    *,
    inventory: dict[
        str,
        dict[str, Any],
    ],
    names: list[str],
    costs: list[str],
    orders: list[str],
    metrics: list[str],
) -> dict[str, np.ndarray]:
    absolute = []
    for kernel in KERNELS.values():
        for accuracy_index in range(
            101
        ):
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
            for left in range(
                len(names)
            ):
                for right in range(
                    left
                ):
                    absolute.append(
                        np.abs(
                            coefficients[left]
                            - coefficients[
                                right
                            ]
                        )
                    )
    values = np.asarray(
        absolute,
        dtype=np.float64,
    )
    return {
        "mean_absolute_contrast": (
            np.mean(
                values,
                axis=0,
            )
        ),
        "rms_contrast": np.sqrt(
            np.mean(
                values * values,
                axis=0,
            )
        ),
        "maximum_contrast": np.max(
            values,
            axis=0,
        ),
    }


def allocation_weights(
    *,
    scores: dict[str, np.ndarray],
    cell_count: int,
) -> dict[str, np.ndarray]:
    equal = np.full(
        cell_count,
        1.0 / cell_count,
        dtype=np.float64,
    )
    result = {
        "equal": equal,
    }
    for name, score in (
        scores.items()
    ):
        flat = score.reshape(-1)
        total = float(
            np.sum(flat)
        )
        if total <= 0.0:
            raise RuntimeError(
                "Geometry score has no mass"
            )
        geometry = flat / total
        result[name] = (
            (
                1.0
                - GEOMETRY_SHARE
            )
            * equal
            + GEOMETRY_SHARE
            * geometry
        )
    for name, weights in (
        result.items()
    ):
        if (
            np.any(
                weights <= 0.0
            )
            or abs(
                float(
                    np.sum(weights)
                )
                - 1.0
            )
            > 1e-12
        ):
            raise RuntimeError(
                f"Invalid alpha weights: "
                f"{name}"
            )
    return result


def weighted_intervals(
    *,
    successes: np.ndarray,
    totals: np.ndarray,
    confidence: float,
    weights: np.ndarray,
    family_divisor: int,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = (
        1.0
        - confidence
    )
    directional_tail = (
        alpha
        * weights
        / (
            2.0
            * family_divisor
        )
    )
    lower = np.zeros(
        len(successes),
        dtype=np.float64,
    )
    upper = np.ones(
        len(successes),
        dtype=np.float64,
    )
    positive = successes > 0
    not_full = successes < totals
    lower[positive] = beta.ppf(
        directional_tail[
            positive
        ],
        successes[positive],
        (
            totals[positive]
            - successes[positive]
            + 1
        ),
    )
    upper[not_full] = beta.ppf(
        1.0
        - directional_tail[
            not_full
        ],
        successes[
            not_full
        ]
        + 1,
        (
            totals[not_full]
            - successes[
                not_full
            ]
        ),
    )
    return lower, upper


def weight_summary(
    *,
    weights: np.ndarray,
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    order = np.argsort(
        weights
    )[::-1]
    return {
        "minimum": float(
            np.min(weights)
        ),
        "maximum": float(
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
                **labels[
                    int(index)
                ],
                "weight": float(
                    weights[
                        index
                    ]
                ),
            }
            for index in order[:10]
        ],
    }


def interval_summary(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    width = (
        upper - lower
    ) * 1_000_000.0
    return {
        "width_ppm": {
            "p50": quantile(
                width.tolist(),
                0.50,
            ),
            "p90": quantile(
                width.tolist(),
                0.90,
            ),
            "maximum": float(
                np.max(width)
            ),
        },
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
                per_accuracy = []
                for row in grid:
                    selected = next(
                        (
                            multiplier
                            for multiplier
                            in (
                                EVIDENCE_MULTIPLIERS
                            )
                            if row[
                                "confidence"
                            ][confidence_key][
                                str(
                                    multiplier
                                )
                            ][method][
                                "tolerance_coverage"
                            ][tolerance][
                                "fraction"
                            ]
                            >= target
                        ),
                        None,
                    )
                    per_accuracy.append(
                        selected
                    )
                result[confidence_key][
                    tolerance
                ][f"{target:.2f}"] = {
                    "maximum_required_multiplier": (
                        None
                        if any(
                            value is None
                            for value
                            in per_accuracy
                        )
                        else max(
                            per_accuracy
                        )
                    ),
                    "accuracy_fraction_certified_at_16x": (
                        sum(
                            value is not None
                            for value
                            in per_accuracy
                        )
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
        for multiplier in (
            EVIDENCE_MULTIPLIERS
        ):
            rows = [
                row["confidence"][
                    confidence_key
                ][str(multiplier)][
                    method
                ]
                for row in grid
            ]
            result[confidence_key][
                str(multiplier)
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


def improvement_summary(
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
            baseline[
                "robust_regret_ppm"
            ]["p50"]
            ,
            candidate[
                "robust_regret_ppm"
            ]["p50"],
        ),
        "p90_regret_ratio": ratio(
            baseline[
                "robust_regret_ppm"
            ]["p90"]
            ,
            candidate[
                "robust_regret_ppm"
            ]["p90"],
        ),
        "maximum_regret_ratio": ratio(
            baseline[
                "robust_regret_ppm"
            ]["maximum"]
            ,
            candidate[
                "robust_regret_ppm"
            ]["maximum"],
        ),
        "coverage_gain": {
            tolerance: (
                candidate[
                    "tolerance_coverage"
                ][tolerance][
                    "fraction"
                ]
                - baseline[
                    "tolerance_coverage"
                ][tolerance][
                    "fraction"
                ]
            )
            for tolerance in (
                TOLERANCE_TARGETS
            )
        },
    }


def global_headline(
    *,
    kernels: dict[str, Any],
) -> dict[str, Any]:
    methods = (
        *SCHEMES,
        "selection_safe_best",
    )
    result = {
        "worst_accuracy": {},
        "coverage_requirements": {},
        "representative_accuracy_0p5": {},
        "individual_scheme_wins": {
            scheme: 0
            for scheme in SCHEMES
        },
        "selection_safe_profile_choices": {
            scheme: 0
            for scheme in SCHEMES
        },
    }
    for method in methods:
        result[
            "worst_accuracy"
        ][method] = {}
        result[
            "coverage_requirements"
        ][method] = {}
        result[
            "representative_accuracy_0p5"
        ][method] = {}
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            result[
                "worst_accuracy"
            ][method][
                confidence_key
            ] = {}
            result[
                "coverage_requirements"
            ][method][
                confidence_key
            ] = {}
            result[
                "representative_accuracy_0p5"
            ][method][
                confidence_key
            ] = {}
            for multiplier in (
                EVIDENCE_MULTIPLIERS
            ):
                multiplier_key = str(
                    multiplier
                )
                rows = [
                    values[
                        "worst_accuracy"
                    ][method][
                        confidence_key
                    ][multiplier_key]
                    for values
                    in kernels.values()
                ]
                result[
                    "worst_accuracy"
                ][method][
                    confidence_key
                ][multiplier_key] = {
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
                representative_rows = [
                    values[
                        "representative_accuracies"
                    ]["0.5"]["confidence"][
                        confidence_key
                    ][multiplier_key][
                        method
                    ]
                    for values
                    in kernels.values()
                ]
                result[
                    "representative_accuracy_0p5"
                ][method][
                    confidence_key
                ][multiplier_key] = {
                    "robust_regret_ppm": {
                        quantile_name: max(
                            row[
                                "robust_regret_ppm"
                            ][quantile_name]
                            for row
                            in representative_rows
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
                            for row
                            in representative_rows
                        )
                        for tolerance in (
                            TOLERANCE_TARGETS
                        )
                    },
                }
            for tolerance in (
                TOLERANCE_TARGETS
            ):
                result[
                    "coverage_requirements"
                ][method][
                    confidence_key
                ][tolerance] = {}
                for target in (
                    COVERAGE_TARGETS
                ):
                    target_key = (
                        f"{target:.2f}"
                    )
                    rows = [
                        values[
                            "coverage_requirements"
                        ][method][
                            confidence_key
                        ][tolerance][
                            target_key
                        ]
                        for values
                        in kernels.values()
                    ]
                    multipliers = [
                        row[
                            "maximum_required_multiplier"
                        ]
                        for row in rows
                    ]
                    result[
                        "coverage_requirements"
                    ][method][
                        confidence_key
                    ][tolerance][
                        target_key
                    ] = {
                        "maximum_required_multiplier": (
                            None
                            if any(
                                value is None
                                for value
                                in multipliers
                            )
                            else max(
                                multipliers
                            )
                        ),
                        "minimum_kernel_accuracy_fraction_certified_at_16x": (
                            min(
                                row[
                                    "accuracy_fraction_certified_at_16x"
                                ]
                                for row in rows
                            )
                        ),
                    }
    for values in kernels.values():
        for row in values[
            "accuracy_grid"
        ]:
            for confidence in (
                CONFIDENCE_LEVELS
            ):
                confidence_key = (
                    f"{confidence:.2f}"
                )
                for multiplier in (
                    EVIDENCE_MULTIPLIERS
                ):
                    multiplier_key = str(
                        multiplier
                    )
                    candidate_rows = (
                        row["confidence"][
                            confidence_key
                        ][multiplier_key]
                    )
                    best = min(
                        SCHEMES,
                        key=lambda scheme: (
                            candidate_rows[
                                scheme
                            ][
                                "robust_regret_ppm"
                            ]["p90"],
                            candidate_rows[
                                scheme
                            ][
                                "robust_regret_ppm"
                            ]["maximum"],
                            scheme,
                        ),
                    )
                    result[
                        "individual_scheme_wins"
                    ][best] += 1
                    for scheme, count in (
                        candidate_rows[
                            "selection_safe_best"
                        ][
                            "selected_scheme_counts"
                        ].items()
                    ):
                        result[
                            "selection_safe_profile_choices"
                        ][scheme] += count
    return result


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e96 = load_e96(args.root)
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
    interval_grid = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        interval_grid[
            confidence_key
        ] = {}
        for multiplier in (
            EVIDENCE_MULTIPLIERS
        ):
            interval_grid[
                confidence_key
            ][str(multiplier)] = {
                "individual": {},
                "selection_safe": {},
            }
            successes = (
                cells["successes"]
                * multiplier
            )
            totals = (
                cells["totals"]
                * multiplier
            )
            for scheme in SCHEMES:
                for mode, divisor in (
                    ("individual", 1),
                    (
                        "selection_safe",
                        len(SCHEMES),
                    ),
                ):
                    lower, upper = (
                        weighted_intervals(
                            successes=(
                                successes
                            ),
                            totals=totals,
                            confidence=(
                                confidence
                            ),
                            weights=weights[
                                scheme
                            ],
                            family_divisor=(
                                divisor
                            ),
                        )
                    )
                    if (
                        np.any(
                            cells["point"]
                            < lower
                            - 1e-15
                        )
                        or np.any(
                            cells["point"]
                            > upper
                            + 1e-15
                        )
                    ):
                        raise RuntimeError(
                            "Point estimate outside "
                            "weighted interval"
                        )
                    interval_grid[
                        confidence_key
                    ][str(multiplier)][
                        mode
                    ][scheme] = {
                        "lower": lower.reshape(
                            point.shape
                        ),
                        "upper": upper.reshape(
                            point.shape
                        ),
                        "summary": (
                            interval_summary(
                                lower=lower,
                                upper=upper,
                            )
                        ),
                        "tail_budget_sum": float(
                            (
                                1.0
                                - confidence
                            )
                            / divisor
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
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        e89_grid = {
            row["accuracy"]: row
            for row in e89["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        grid = []
        representatives = {}
        for accuracy in accuracy_grid:
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
            reconstruction_error = float(
                np.max(
                    np.abs(
                        reconstructed
                        - expected
                    )
                )
            )
            reconstruction_maximum_error = max(
                reconstruction_maximum_error,
                reconstruction_error,
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
                for multiplier in (
                    EVIDENCE_MULTIPLIERS
                ):
                    multiplier_key = str(
                        multiplier
                    )
                    row = {}
                    safe_regrets = []
                    safe_scheme_names = []
                    for scheme in SCHEMES:
                        individual_interval = (
                            interval_grid[
                                confidence_key
                            ][multiplier_key][
                                "individual"
                            ][scheme]
                        )
                        individual_regret, _ = (
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
                                    individual_interval[
                                        "lower"
                                    ]
                                ),
                                upper=(
                                    individual_interval[
                                        "upper"
                                    ]
                                ),
                            )
                        )
                        row[scheme] = (
                            regret_summary(
                                individual_regret
                            )
                        )
                        safe_interval = (
                            interval_grid[
                                confidence_key
                            ][multiplier_key][
                                "selection_safe"
                            ][scheme]
                        )
                        safe_regret, _ = (
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
                                    safe_interval[
                                        "lower"
                                    ]
                                ),
                                upper=(
                                    safe_interval[
                                        "upper"
                                    ]
                                ),
                            )
                        )
                        safe_regrets.append(
                            safe_regret
                        )
                        safe_scheme_names.append(
                            scheme
                        )
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
                        "selected_scheme_counts"
                    ] = {
                        safe_scheme_names[
                            index
                        ]: int(
                            np.count_nonzero(
                                safe_choice
                                == index
                            )
                        )
                        for index in range(
                            len(
                                safe_scheme_names
                            )
                        )
                    }
                    row[
                        "selection_safe_best"
                    ] = safe_summary
                    row[
                        "mean_vs_equal"
                    ] = (
                        improvement_summary(
                            baseline=row[
                                "equal"
                            ],
                            candidate=row[
                                "mean_absolute_contrast"
                            ],
                        )
                    )
                    row[
                        "selection_safe_vs_equal"
                    ] = (
                        improvement_summary(
                            baseline=row[
                                "equal"
                            ],
                            candidate=row[
                                "selection_safe_best"
                            ],
                        )
                    )
                    confidence_rows[
                        confidence_key
                    ][multiplier_key] = row
            result_row = {
                "accuracy": accuracy,
                "winner_counts": (
                    winner_counts
                ),
                "confidence": (
                    confidence_rows
                ),
            }
            grid.append(
                result_row
            )
            e96_row = (
                e96["kernels"][
                    kernel_name
                ]["accuracy_grid"][
                    int(
                        round(
                            accuracy
                            * 100
                        )
                    )
                ]
            )
            for confidence in (
                CONFIDENCE_LEVELS
            ):
                confidence_key = (
                    f"{confidence:.2f}"
                )
                for multiplier in (
                    EVIDENCE_MULTIPLIERS
                ):
                    expected_equal = (
                        e96_row[
                            "confidence"
                        ][confidence_key][
                            str(multiplier)
                        ]
                    )
                    actual_equal = (
                        confidence_rows[
                            confidence_key
                        ][str(multiplier)][
                            "equal"
                        ]
                    )
                    regret_difference = max(
                        abs(
                            actual_equal[
                                "robust_regret_ppm"
                            ][quantile_name]
                            - expected_equal[
                                "robust_regret_ppm"
                            ][quantile_name]
                        )
                        for quantile_name in (
                            "p50",
                            "p90",
                            "p95",
                            "p99",
                            "maximum",
                        )
                    )
                    coverage_equal = all(
                        actual_equal[
                            "tolerance_coverage"
                        ][tolerance][
                            "count"
                        ]
                        == expected_equal[
                            "tolerance_coverage"
                        ][tolerance][
                            "count"
                        ]
                        for tolerance in (
                            actual_equal[
                                "tolerance_coverage"
                            ]
                        )
                    )
                    if (
                        regret_difference
                        > 1e-9
                        or not coverage_equal
                    ):
                        raise RuntimeError(
                            "Equal allocation did "
                            "not reproduce E96"
                        )
            if accuracy in (
                REPRESENTATIVE_ACCURACIES
            ):
                representatives[
                    f"{accuracy:g}"
                ] = result_row
        methods = (
            *SCHEMES,
            "selection_safe_best",
        )
        kernels[kernel_name] = {
            "description": kernel[
                "description"
            ],
            "accuracy_grid": grid,
            "representative_accuracies": (
                representatives
            ),
            "worst_accuracy": {
                method: (
                    worst_accuracy_summary(
                        grid=grid,
                        method=method,
                    )
                )
                for method in methods
            },
            "coverage_requirements": {
                method: (
                    coverage_requirements(
                        grid=grid,
                        method=method,
                    )
                )
                for method in methods
            },
        }

    result = {
        "suite": (
            "pacing_coalescing_order_risk_alpha_allocation"
        ),
        "model": (
            "Allocate the familywise error "
            "budget across 72 lower and 72 "
            "upper E83 risk bounds using only "
            "declared policy geometry. Four "
            "candidate allocations are evaluated "
            "individually; a best-of-four "
            "certificate divides alpha across "
            "candidates so profilewise selection "
            "remains familywise valid."
        ),
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            ),
            "risk_uncertainty": (
                "results/raw/"
                "pacing_coalescing_order_risk_uncertainty/"
                "pacing_coalescing_order_risk_uncertainty_full.json"
            ),
        },
        "geometry_share": (
            GEOMETRY_SHARE
        ),
        "candidate_count": len(
            SCHEMES
        ),
        "confidence_levels": list(
            CONFIDENCE_LEVELS
        ),
        "evidence_multipliers": list(
            EVIDENCE_MULTIPLIERS
        ),
        "schemes": {
            scheme: {
                "weight_summary": (
                    weight_summary(
                        weights=weights[
                            scheme
                        ],
                        labels=cells[
                            "labels"
                        ],
                    )
                ),
                "interval_summary": {
                    confidence_key: {
                        multiplier: {
                            mode: (
                                values[
                                    mode
                                ][scheme][
                                    "summary"
                                ]
                            )
                            for mode in (
                                "individual",
                                "selection_safe",
                            )
                        }
                        for multiplier, values
                        in confidence_values.items()
                    }
                    for confidence_key, confidence_values
                    in interval_grid.items()
                },
            }
            for scheme in SCHEMES
        },
        "prevalence_grid": {
            "denominator": (
                PREVALENCE_DENOMINATOR
            ),
            "profile_count": len(
                profiles
            ),
        },
        "accuracy_grid": accuracy_grid,
        "kernels": kernels,
        "headline": {
            "global": global_headline(
                kernels=kernels
            ),
            "kernels": {
                kernel_name: {
                    "worst_accuracy": values[
                        "worst_accuracy"
                    ],
                    "coverage_requirements": (
                        values[
                            "coverage_requirements"
                        ]
                    ),
                    "representative_accuracies": (
                        values[
                            "representative_accuracies"
                        ]
                    ),
                }
                for kernel_name, values
                in kernels.items()
            },
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
            "candidate_count": len(
                SCHEMES
            ),
            "tail_weight_sums_valid": (
                True
            ),
            "point_estimates_inside_all_intervals": (
                True
            ),
            "policy_coefficient_reconstruction_maximum_error": (
                reconstruction_maximum_error
            ),
            "e89_winner_counts_reproduced": (
                True
            ),
            "equal_allocation_reproduces_e96": (
                True
            ),
            "selection_safe_family_error": (
                {
                    f"{confidence:.2f}": (
                        1.0
                        - confidence
                    )
                    for confidence
                    in CONFIDENCE_LEVELS
                }
            ),
            "rectangular_maximization_exact": (
                True
            ),
        },
        "limitations": [
            "Geometry-weighted allocations are outcome-independent, but individually reported candidate intervals are alternatives rather than one jointly selectable family.",
            "The selection-safe best-of-four divides alpha across candidates and is valid for data-dependent profilewise choice, but pays a multiple-candidate width tax.",
            "Cellwise binomial and rectangular-confidence limitations from E96 remain unchanged.",
            "Evidence multipliers are unchanged-rate planning sensitivity rather than new measured paths.",
            "The audit is local statistical analysis and is not new GPU evidence.",
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
