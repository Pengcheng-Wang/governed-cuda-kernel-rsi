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
    estimate_distribution,
    policy_inventory,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    DRIFT_METRICS,
    load_e83,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
    prevalence_profiles,
    profile_metrics,
    winner,
)
from analyze_pacing_coalescing_order_prevalence_certification import (
    REPRESENTATIVE_ACCURACIES,
    TOLERANCE,
    quantile,
)


CONFIDENCE_LEVELS = (
    0.95,
    0.99,
)
REPLICATION_MULTIPLIERS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
)
REGRET_TOLERANCES_PPM = (
    0,
    1,
    5,
    10,
    25,
    50,
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


def tolerance_key(
    tolerance_ppm: int,
) -> str:
    return str(
        tolerance_ppm
    )


def base_cells(
    *,
    drift: dict[str, Any],
    costs: list[str],
    orders: list[str],
    metrics: list[str],
) -> dict[str, Any]:
    labels = []
    successes = []
    totals = []
    reported_lower = []
    reported_upper = []
    for cost in costs:
        for order in orders:
            for metric in metrics:
                row = drift[cost][
                    order
                ]["by_breach_metric"][
                    metric
                ]
                slower = row["slower"]
                labels.append(
                    {
                        "cost": cost,
                        "order": order,
                        "metric": metric,
                    }
                )
                successes.append(
                    slower["successes"]
                )
                totals.append(
                    row[
                        "comparison_count"
                    ]
                )
                reported_lower.append(
                    slower["lower_95pct"]
                )
                reported_upper.append(
                    slower["upper_95pct"]
                )
    successes_array = np.asarray(
        successes,
        dtype=np.int64,
    )
    totals_array = np.asarray(
        totals,
        dtype=np.int64,
    )
    return {
        "labels": labels,
        "successes": successes_array,
        "totals": totals_array,
        "point": (
            successes_array
            / totals_array
        ),
        "reported_lower_95": np.asarray(
            reported_lower,
            dtype=np.float64,
        ),
        "reported_upper_95": np.asarray(
            reported_upper,
            dtype=np.float64,
        ),
    }


def clopper_pearson_family(
    *,
    successes: np.ndarray,
    totals: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - confidence
    tail = (
        alpha
        / (
            2.0
            * len(successes)
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
        tail,
        successes[positive],
        (
            totals[positive]
            - successes[positive]
            + 1
        ),
    )
    upper[not_full] = beta.ppf(
        1.0 - tail,
        successes[not_full] + 1,
        (
            totals[not_full]
            - successes[not_full]
        ),
    )
    if (
        np.any(
            lower < 0.0
        )
        or np.any(
            upper > 1.0
        )
        or np.any(
            lower > upper
        )
    ):
        raise RuntimeError(
            "Invalid Clopper-Pearson "
            "interval"
        )
    return lower, upper


def interval_summary(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    width = (
        upper - lower
    ) * 1_000_000.0
    half_width = (
        upper - lower
    ) * 500_000.0
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
        "half_width_ppm": {
            "p50": quantile(
                half_width.tolist(),
                0.50,
            ),
            "p90": quantile(
                half_width.tolist(),
                0.90,
            ),
            "maximum": float(
                np.max(
                    half_width
                )
            ),
        },
    }


def policy_coefficients(
    *,
    inventory: dict[
        str,
        dict[str, Any],
    ],
    names: list[str],
    costs: list[str],
    orders: list[str],
    metrics: list[str],
    accuracy: float,
    offset_weights: dict[int, float],
) -> np.ndarray:
    coefficients = np.zeros(
        (
            len(names),
            len(costs),
            len(orders),
            len(metrics),
        ),
        dtype=np.float64,
    )
    for policy_index, name in enumerate(
        names
    ):
        mapping = inventory[name][
            "mapping"
        ]
        for cost_index in range(
            len(costs)
        ):
            distribution = (
                estimate_distribution(
                    costs=costs,
                    actual_index=(
                        cost_index
                    ),
                    accuracy=accuracy,
                    offset_weights=(
                        offset_weights
                    ),
                )
            )
            for metric_index, metric in (
                enumerate(metrics)
            ):
                for estimated_cost, probability in (
                    distribution.items()
                ):
                    order = mapping[
                        (
                            estimated_cost,
                            metric,
                        )
                    ]
                    coefficients[
                        policy_index,
                        cost_index,
                        orders.index(
                            order
                        ),
                        metric_index,
                    ] += (
                        probability
                        / len(metrics)
                    )
    sums = np.sum(
        coefficients,
        axis=(2, 3),
    )
    if not np.allclose(
        sums,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Policy coefficients do not "
            "normalize per actual cost"
        )
    return coefficients


def selected_policies(
    *,
    profiles: list[dict[str, Any]],
    per_policy_costs: dict[
        str,
        list[dict[str, Any]],
    ],
    inventory: dict[
        str,
        dict[str, Any],
    ],
    names: list[str],
) -> np.ndarray:
    name_to_index = {
        name: index
        for index, name
        in enumerate(names)
    }
    return np.asarray(
        [
            name_to_index[
                winner(
                    metrics={
                        name: profile_metrics(
                            per_cost=rows,
                            weights=profile[
                                "weights"
                            ],
                        )
                        for name, rows
                        in per_policy_costs.items()
                    },
                    inventory=inventory,
                )
            ]
            for profile in profiles
        ],
        dtype=np.int64,
    )


def robust_regret(
    *,
    prevalence: np.ndarray,
    selected: np.ndarray,
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros(
        len(prevalence),
        dtype=np.float64,
    )
    worst_competitor = np.zeros(
        len(prevalence),
        dtype=np.int64,
    )
    for selected_policy in range(
        len(coefficients)
    ):
        profile_indices = np.flatnonzero(
            selected
            == selected_policy
        )
        if not len(
            profile_indices
        ):
            continue
        worst = []
        for competitor in range(
            len(coefficients)
        ):
            difference = (
                coefficients[
                    selected_policy
                ]
                - coefficients[
                    competitor
                ]
            )
            cost_contribution = np.sum(
                np.where(
                    difference >= 0.0,
                    difference * upper,
                    difference * lower,
                ),
                axis=(1, 2),
            )
            worst.append(
                prevalence[
                    profile_indices
                ]
                @ cost_contribution
            )
        worst_array = np.asarray(
            worst
        )
        competitor = np.argmax(
            worst_array,
            axis=0,
        )
        result[
            profile_indices
        ] = np.maximum(
            0.0,
            worst_array[
                competitor,
                np.arange(
                    len(
                        profile_indices
                    )
                ),
            ],
        )
        worst_competitor[
            profile_indices
        ] = competitor
    return result, worst_competitor


def regret_summary(
    regret: np.ndarray,
) -> dict[str, Any]:
    ppm = (
        regret
        * 1_000_000.0
    )
    return {
        "robust_regret_ppm": {
            "p50": quantile(
                ppm.tolist(),
                0.50,
            ),
            "p90": quantile(
                ppm.tolist(),
                0.90,
            ),
            "p95": quantile(
                ppm.tolist(),
                0.95,
            ),
            "p99": quantile(
                ppm.tolist(),
                0.99,
            ),
            "maximum": float(
                np.max(ppm)
            ),
        },
        "tolerance_coverage": {
            tolerance_key(
                tolerance_ppm
            ): {
                "count": int(
                    np.count_nonzero(
                        ppm
                        <= (
                            tolerance_ppm
                            + 1e-9
                        )
                    )
                ),
                "fraction": float(
                    np.mean(
                        ppm
                        <= (
                            tolerance_ppm
                            + 1e-9
                        )
                    )
                ),
            }
            for tolerance_ppm
            in REGRET_TOLERANCES_PPM
        },
    }


def replication_requirements(
    *,
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        result[confidence_key] = {}
        for tolerance_ppm in (
            REGRET_TOLERANCES_PPM
        ):
            tolerance = tolerance_key(
                tolerance_ppm
            )
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
                                REPLICATION_MULTIPLIERS
                            )
                            if row[
                                "confidence"
                            ][confidence_key][
                                str(
                                    multiplier
                                )
                            ][
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
                available = [
                    value
                    for value
                    in per_accuracy
                    if value is not None
                ]
                result[confidence_key][
                    tolerance
                ][f"{target:.2f}"] = {
                    "maximum_required_multiplier": (
                        max(available)
                        if (
                            len(available)
                            == len(
                                per_accuracy
                            )
                        )
                        else None
                    ),
                    "accuracy_points_certified_at_256x": (
                        len(available)
                    ),
                    "accuracy_fraction_certified_at_256x": (
                        len(available)
                        / len(
                            per_accuracy
                        )
                    ),
                }
    return result


def current_evidence_summary(
    *,
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        rows = [
            row["confidence"][
                confidence_key
            ]["1"]
            for row in grid
        ]
        result[confidence_key] = {
            "worst_accuracy_robust_regret_ppm": {
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
            "worst_accuracy_tolerance_coverage": {
                tolerance_key(
                    tolerance_ppm
                ): min(
                    row[
                        "tolerance_coverage"
                    ][
                        tolerance_key(
                            tolerance_ppm
                        )
                    ]["fraction"]
                    for row in rows
                )
                for tolerance_ppm
                in REGRET_TOLERANCES_PPM
            },
        }
    return result


def global_headline(
    *,
    kernels: dict[str, Any],
) -> dict[str, Any]:
    current = {}
    representative = {}
    requirements = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        current[
            confidence_key
        ] = {
            "worst_accuracy_robust_regret_ppm": {
                quantile_name: max(
                    values[
                        "current_evidence"
                    ][confidence_key][
                        "worst_accuracy_robust_regret_ppm"
                    ][quantile_name]
                    for values
                    in kernels.values()
                )
                for quantile_name in (
                    "p50",
                    "p90",
                    "p95",
                    "p99",
                    "maximum",
                )
            },
            "worst_accuracy_tolerance_coverage": {
                tolerance_key(
                    tolerance_ppm
                ): min(
                    values[
                        "current_evidence"
                    ][confidence_key][
                        "worst_accuracy_tolerance_coverage"
                    ][
                        tolerance_key(
                            tolerance_ppm
                        )
                    ]
                    for values
                    in kernels.values()
                )
                for tolerance_ppm
                in REGRET_TOLERANCES_PPM
            },
            "worst_case": max(
                (
                    values[
                        "current_evidence"
                    ]["worst_cases"][
                        confidence_key
                    ]
                    for values
                    in kernels.values()
                ),
                key=lambda row: row[
                    "robust_regret_ppm"
                ],
            ),
        }
        representative_rows = [
            values[
                "representative_accuracies"
            ]["0.5"]["confidence"][
                confidence_key
            ]["1"]
            for values
            in kernels.values()
        ]
        representative[
            confidence_key
        ] = {
            "accuracy": 0.5,
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
                tolerance_key(
                    tolerance_ppm
                ): min(
                    row[
                        "tolerance_coverage"
                    ][
                        tolerance_key(
                            tolerance_ppm
                        )
                    ]["fraction"]
                    for row
                    in representative_rows
                )
                for tolerance_ppm
                in REGRET_TOLERANCES_PPM
            },
        }
        requirements[
            confidence_key
        ] = {}
        for tolerance_ppm in (
            REGRET_TOLERANCES_PPM
        ):
            tolerance = tolerance_key(
                tolerance_ppm
            )
            requirements[
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
                        "replication_requirements"
                    ][confidence_key][
                        tolerance
                    ][target_key]
                    for values
                    in kernels.values()
                ]
                multipliers = [
                    row[
                        "maximum_required_multiplier"
                    ]
                    for row in rows
                ]
                requirements[
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
                    "minimum_kernel_accuracy_fraction_certified_at_256x": (
                        min(
                            row[
                                "accuracy_fraction_certified_at_256x"
                            ]
                            for row in rows
                        )
                    ),
                }
    return {
        "current_evidence": current,
        "representative_accuracy_0p5": (
            representative
        ),
        "replication_requirements": (
            requirements
        ),
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
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
        previous_lower = None
        previous_upper = None
        for multiplier in (
            REPLICATION_MULTIPLIERS
        ):
            lower, upper = (
                clopper_pearson_family(
                    successes=(
                        cells[
                            "successes"
                        ]
                        * multiplier
                    ),
                    totals=(
                        cells["totals"]
                        * multiplier
                    ),
                    confidence=confidence,
                )
            )
            if (
                np.any(
                    point
                    .reshape(-1)
                    < lower
                    - 1e-15
                )
                or np.any(
                    point
                    .reshape(-1)
                    > upper
                    + 1e-15
                )
            ):
                raise RuntimeError(
                    "Point estimate is outside "
                    "its confidence interval"
                )
            if (
                previous_lower is not None
                and (
                    np.any(
                        lower
                        + 1e-15
                        < previous_lower
                    )
                    or np.any(
                        upper
                        - 1e-15
                        > previous_upper
                    )
                )
            ):
                raise RuntimeError(
                    "Confidence interval did not "
                    "shrink with replication"
                )
            previous_lower = lower
            previous_upper = upper
            interval_grid[
                confidence_key
            ][str(multiplier)] = {
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
            }
    for multiplier in (
        REPLICATION_MULTIPLIERS
    ):
        lower_95 = interval_grid[
            "0.95"
        ][str(multiplier)]["lower"]
        upper_95 = interval_grid[
            "0.95"
        ][str(multiplier)]["upper"]
        lower_99 = interval_grid[
            "0.99"
        ][str(multiplier)]["lower"]
        upper_99 = interval_grid[
            "0.99"
        ][str(multiplier)]["upper"]
        if (
            np.any(
                lower_99
                > lower_95
                + 1e-15
            )
            or np.any(
                upper_99
                < upper_95
                - 1e-15
            )
        ):
            raise RuntimeError(
                "99% interval is not wider "
                "than 95% interval"
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
        current_worst_cases = {
            f"{confidence:.2f}": None
            for confidence
            in CONFIDENCE_LEVELS
        }
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
            if (
                reconstruction_error
                > 1e-12
            ):
                raise RuntimeError(
                    "Policy coefficient "
                    "reconstruction failed"
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
            regret_by_confidence = {}
            for confidence in (
                CONFIDENCE_LEVELS
            ):
                confidence_key = (
                    f"{confidence:.2f}"
                )
                confidence_rows[
                    confidence_key
                ] = {}
                regret_by_confidence[
                    confidence_key
                ] = {}
                previous_regret = None
                for multiplier in (
                    REPLICATION_MULTIPLIERS
                ):
                    intervals = (
                        interval_grid[
                            confidence_key
                        ][str(multiplier)]
                    )
                    (
                        regret,
                        worst_competitor,
                    ) = robust_regret(
                        prevalence=(
                            prevalence
                        ),
                        selected=selected,
                        coefficients=(
                            coefficients
                        ),
                        lower=intervals[
                            "lower"
                        ],
                        upper=intervals[
                            "upper"
                        ],
                    )
                    if (
                        previous_regret
                        is not None
                        and np.any(
                            regret
                            > previous_regret
                            + 1e-12
                        )
                    ):
                        raise RuntimeError(
                            "Robust regret increased "
                            "with replication"
                        )
                    previous_regret = regret
                    if multiplier == 1:
                        profile_index = int(
                            np.argmax(
                                regret
                            )
                        )
                        candidate = {
                            "accuracy": accuracy,
                            "profile_index": (
                                profile_index
                            ),
                            "prevalence_weights": list(
                                profiles[
                                    profile_index
                                ]["weights"]
                            ),
                            "selected_policy": names[
                                int(
                                    selected[
                                        profile_index
                                    ]
                                )
                            ],
                            "worst_competitor": names[
                                int(
                                    worst_competitor[
                                        profile_index
                                    ]
                                )
                            ],
                            "robust_regret_ppm": (
                                float(
                                    regret[
                                        profile_index
                                    ]
                                    * 1_000_000.0
                                )
                            ),
                        }
                        current = (
                            current_worst_cases[
                                confidence_key
                            ]
                        )
                        if (
                            current is None
                            or candidate[
                                "robust_regret_ppm"
                            ]
                            > current[
                                "robust_regret_ppm"
                            ]
                        ):
                            current_worst_cases[
                                confidence_key
                            ] = candidate
                    regret_by_confidence[
                        confidence_key
                    ][str(multiplier)] = (
                        regret
                    )
                    confidence_rows[
                        confidence_key
                    ][str(multiplier)] = (
                        regret_summary(
                            regret
                        )
                    )
            for multiplier in (
                REPLICATION_MULTIPLIERS
            ):
                if np.any(
                    regret_by_confidence[
                        "0.99"
                    ][str(multiplier)]
                    + 1e-12
                    < regret_by_confidence[
                        "0.95"
                    ][str(multiplier)]
                ):
                    raise RuntimeError(
                        "99% robust regret is "
                        "below 95% regret"
                    )
            row = {
                "accuracy": accuracy,
                "winner_counts": (
                    winner_counts
                ),
                "confidence": (
                    confidence_rows
                ),
            }
            grid.append(row)
            if accuracy in (
                REPRESENTATIVE_ACCURACIES
            ):
                representatives[
                    f"{accuracy:g}"
                ] = row
        kernels[kernel_name] = {
            "description": kernel[
                "description"
            ],
            "accuracy_grid": grid,
            "representative_accuracies": (
                representatives
            ),
            "current_evidence": (
                {
                    **current_evidence_summary(
                        grid=grid
                    ),
                    "worst_cases": (
                        current_worst_cases
                    ),
                }
            ),
            "replication_requirements": (
                replication_requirements(
                    grid=grid
                )
            ),
        }

    reported_width = (
        cells[
            "reported_upper_95"
        ]
        - cells[
            "reported_lower_95"
        ]
    ) * 1_000_000.0
    simultaneous_width = (
        interval_grid["0.95"]["1"][
            "upper"
        ].reshape(-1)
        - interval_grid["0.95"]["1"][
            "lower"
        ].reshape(-1)
    ) * 1_000_000.0
    width_ratio = (
        simultaneous_width
        / reported_width
    )
    result = {
        "suite": (
            "pacing_coalescing_order_risk_uncertainty"
        ),
        "model": (
            "Treat each of E83's 72 slower-"
            "event aggregates as a binomial "
            "risk cell. Construct simultaneous "
            "Bonferroni Clopper-Pearson intervals "
            "and maximize the point-selected "
            "policy's true regret over the "
            "resulting rectangular confidence "
            "set. Integer replication multipliers "
            "are unchanged-rate planning "
            "sensitivity, not new evidence."
        ),
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
        },
        "base_cells": {
            "count": len(
                cells["labels"]
            ),
            "labels": cells[
                "labels"
            ],
            "comparison_count": int(
                cells["totals"][0]
            ),
            "comparison_counts_equal": bool(
                np.all(
                    cells["totals"]
                    == cells[
                        "totals"
                    ][0]
                )
            ),
            "zero_success_count": int(
                np.count_nonzero(
                    cells[
                        "successes"
                    ]
                    == 0
                )
            ),
            "maximum_success_count": int(
                np.max(
                    cells[
                        "successes"
                    ]
                )
            ),
            "reported_marginal_wilson_95_width_ppm": {
                "p50": quantile(
                    reported_width.tolist(),
                    0.50,
                ),
                "p90": quantile(
                    reported_width.tolist(),
                    0.90,
                ),
                "maximum": float(
                    np.max(
                        reported_width
                    )
                ),
            },
            "simultaneous_cp95_to_reported_wilson95_width_ratio": {
                "p50": quantile(
                    width_ratio.tolist(),
                    0.50,
                ),
                "p90": quantile(
                    width_ratio.tolist(),
                    0.90,
                ),
                "maximum": float(
                    np.max(
                        width_ratio
                    )
                ),
            },
            "simultaneous_interval_summary": {
                confidence_key: {
                    multiplier: (
                        values["summary"]
                    )
                    for multiplier, values
                    in confidence_values.items()
                }
                for confidence_key, confidence_values
                in interval_grid.items()
            },
            "zero_success_upper_ppm": {
                confidence_key: {
                    multiplier: float(
                        np.max(
                            values[
                                "upper"
                            ].reshape(-1)[
                                cells[
                                    "successes"
                                ]
                                == 0
                            ]
                        )
                        * 1_000_000.0
                    )
                    for multiplier, values
                    in confidence_values.items()
                }
                for confidence_key, confidence_values
                in interval_grid.items()
            },
        },
        "confidence_levels": list(
            CONFIDENCE_LEVELS
        ),
        "replication_multipliers": list(
            REPLICATION_MULTIPLIERS
        ),
        "regret_tolerances_ppm": list(
            REGRET_TOLERANCES_PPM
        ),
        "coverage_targets": list(
            COVERAGE_TARGETS
        ),
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
                    "current_evidence": values[
                        "current_evidence"
                    ],
                    "replication_requirements": (
                        values[
                            "replication_requirements"
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
            "base_cell_count": len(
                cells["labels"]
            ),
            "point_estimates_inside_all_intervals": (
                True
            ),
            "intervals_shrink_with_replication": (
                True
            ),
            "ninety_nine_intervals_contain_ninety_five": (
                True
            ),
            "policy_coefficient_reconstruction_maximum_error": (
                reconstruction_maximum_error
            ),
            "e89_winner_counts_reproduced": (
                True
            ),
            "robust_regret_nonincreasing_with_replication": (
                True
            ),
            "ninety_nine_regret_no_smaller_than_ninety_five": (
                True
            ),
            "rectangular_maximization_exact": (
                True
            ),
        },
        "limitations": [
            "Clopper-Pearson treats each aggregate E83 cell as binomial; heterogeneous simulation strata and shared paths across cells are not modeled.",
            "The simultaneous rectangular confidence set ignores cross-cell dependence and is therefore conservative when the marginal interval model is appropriate.",
            "Replication multipliers scale successes and trials at unchanged empirical rates; they are planning sensitivity, not observed additional paths.",
            "The audit holds traffic prevalence and cost-estimator confusion fixed and studies uncertainty only in E83 slower-event probabilities.",
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
