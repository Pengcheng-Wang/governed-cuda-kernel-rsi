from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom

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
    PREVALENCE_DENOMINATOR,
    TOLERANCE_TARGETS,
    load_json,
)
from analyze_pacing_coalescing_order_path_floor_frontier import (
    regret_summaries_many,
    robust_regret_many,
    update_worst,
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
)


PLATEAU_THRESHOLDS = (
    0.0025,
    0.005,
    0.01,
)
ALPHA = 0.05


def multiplier_grid() -> tuple[float, ...]:
    values = {
        round(
            0.5 + 0.25 * index,
            12,
        )
        for index in range(11)
    }
    values.update(
        round(
            1.0 + 0.1 * index,
            12,
        )
        for index in range(11)
    )
    values.add(
        1.0 / math.sqrt(2.0)
    )
    values.add(
        math.sqrt(2.0)
    )
    return tuple(sorted(values))


MULTIPLIERS = multiplier_grid()


def multiplier_key(
    multiplier: float,
) -> str:
    if math.isclose(
        multiplier,
        1.0 / math.sqrt(2.0),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "inv_sqrt2"
    if math.isclose(
        multiplier,
        math.sqrt(2.0),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "sqrt2"
    return (
        "m_"
        + f"{multiplier:.2f}".replace(
            ".",
            "p",
        )
    )


def load_e104(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_stage_weights"
        / "pacing_coalescing_order_stage_weights_full.json"
    )


def exact_fixed_weight_validation() -> dict[str, Any]:
    maximum_excess = {
        multiplier_key(
            multiplier
        ): 0.0
        for multiplier in MULTIPLIERS
    }
    checks = 0

    def future_total(
        rule: int,
        pilot_successes: int,
        pilot_total: int,
    ) -> int:
        if rule == 0:
            return (
                1 + pilot_successes
            )
        return (
            6
            if 2
            * pilot_successes
            >= pilot_total
            else 1
        )

    for pilot_total in (
        2,
        3,
        4,
    ):
        for rule in range(2):
            for probability in (
                0.1,
                0.3,
                0.5,
                0.8,
            ):
                for future_factor in (
                    0.25,
                    1.0,
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


def safe_ratio(
    value: float,
    optimum: float,
) -> float:
    if optimum <= 0.0:
        return (
            1.0
            if value <= 0.0
            else float("inf")
        )
    return value / optimum


def quantile(
    values: list[float],
    probability: float,
) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(
            0,
            math.ceil(
                probability
                * len(ordered)
            )
            - 1,
        ),
    )
    return float(ordered[index])


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e104 = load_e104(args.root)
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

    intervals = {
        f"{confidence:.2f}": {}
        for confidence in (
            CONFIDENCE_LEVELS
        )
    }
    maximum_root_residual = {
        multiplier_key(
            multiplier
        ): 0.0
        for multiplier in MULTIPLIERS
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
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            intervals[
                confidence_key
            ][key] = {}
            for multiplier in MULTIPLIERS:
                multiplier_name = (
                    multiplier_key(
                        multiplier
                    )
                )
                lower = []
                upper = []
                residual = 0.0
                effective_factor = (
                    multiplier
                    * multiplier
                    * factor
                )
                for (
                    pilot_success,
                    pilot_total,
                    future_success,
                    future_total,
                    alpha_weight,
                ) in zip(
                    cells["successes"],
                    cells["totals"],
                    allocation["successes"],
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
                            / 2.0
                        ),
                        future_factor=(
                            effective_factor
                        ),
                    )
                    lower.append(cell_lower)
                    upper.append(cell_upper)
                    residual = max(
                        residual,
                        cell_residual,
                    )
                maximum_root_residual[
                    multiplier_name
                ] = max(
                    maximum_root_residual[
                        multiplier_name
                    ],
                    residual,
                )
                intervals[
                    confidence_key
                ][key][
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

    candidate_names = [
        multiplier_key(multiplier)
        for multiplier in MULTIPLIERS
    ]
    kernel_worst = {
        kernel: {
            f"{confidence:.2f}": {
                factor_key(factor): {
                    name: None
                    for name in (
                        candidate_names
                    )
                }
                for factor in (
                    FUTURE_FACTORS
                )
            }
            for confidence in (
                CONFIDENCE_LEVELS
            )
        }
        for kernel in KERNELS
    }
    local_wins = {
        name: {
            "p90": 0,
            "maximum": 0,
        }
        for name in candidate_names
    }
    local_relative = {
        name: {
            "p90": [],
            "maximum": [],
        }
        for name in candidate_names
    }
    condition_count = 0
    for kernel_name in KERNELS:
        for row in pilot["cache"][
            kernel_name
        ]:
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
                    lowers = np.asarray(
                        [
                            intervals[
                                confidence_key
                            ][key][name][
                                "lower"
                            ].reshape(
                                point_shape
                            )
                            for name in (
                                candidate_names
                            )
                        ]
                    )
                    uppers = np.asarray(
                        [
                            intervals[
                                confidence_key
                            ][key][name][
                                "upper"
                            ].reshape(
                                point_shape
                            )
                            for name in (
                                candidate_names
                            )
                        ]
                    )
                    regrets = (
                        robust_regret_many(
                            prevalence=(
                                prevalence
                            ),
                            selected=row[
                                "selected"
                            ],
                            coefficients=row[
                                "coefficients"
                            ],
                            lowers=lowers,
                            uppers=uppers,
                        )
                    )
                    summaries = (
                        regret_summaries_many(
                            regrets
                        )
                    )
                    p90_values = np.asarray(
                        [
                            summary[
                                "robust_regret_ppm"
                            ]["p90"]
                            for summary
                            in summaries
                        ]
                    )
                    maximum_values = np.asarray(
                        [
                            summary[
                                "robust_regret_ppm"
                            ]["maximum"]
                            for summary
                            in summaries
                        ]
                    )
                    p90_optimum = float(
                        np.min(p90_values)
                    )
                    maximum_optimum = float(
                        np.min(
                            maximum_values
                        )
                    )
                    for index, name in enumerate(
                        candidate_names
                    ):
                        kernel_worst[
                            kernel_name
                        ][confidence_key][
                            key
                        ][name] = update_worst(
                            aggregate=(
                                kernel_worst[
                                    kernel_name
                                ][confidence_key][
                                    key
                                ][name]
                            ),
                            summary=(
                                summaries[index]
                            ),
                        )
                        local_relative[name][
                            "p90"
                        ].append(
                            safe_ratio(
                                p90_values[
                                    index
                                ],
                                p90_optimum,
                            )
                        )
                        local_relative[name][
                            "maximum"
                        ].append(
                            safe_ratio(
                                maximum_values[
                                    index
                                ],
                                maximum_optimum,
                            )
                        )
                    for index in np.flatnonzero(
                        np.abs(
                            p90_values
                            - p90_optimum
                        )
                        <= 1e-12
                    ):
                        local_wins[
                            candidate_names[
                                int(index)
                            ]
                        ]["p90"] += 1
                    for index in np.flatnonzero(
                        np.abs(
                            maximum_values
                            - maximum_optimum
                        )
                        <= 1e-12
                    ):
                        local_wins[
                            candidate_names[
                                int(index)
                            ]
                        ]["maximum"] += 1
                    condition_count += 1

    global_worst = {
        name: {
            f"{confidence:.2f}": {}
            for confidence in (
                CONFIDENCE_LEVELS
            )
        }
        for name in candidate_names
    }
    global_plateaus = {
        f"{confidence:.2f}": {}
        for confidence in (
            CONFIDENCE_LEVELS
        )
    }
    global_ratios = {
        name: []
        for name in candidate_names
    }
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
            for name in candidate_names:
                rows = [
                    kernel_worst[kernel][
                        confidence_key
                    ][key][name]
                    for kernel in KERNELS
                ]
                global_worst[name][
                    confidence_key
                ][key] = {
                    "robust_regret_ppm": {
                        metric: max(
                            item[
                                "robust_regret_ppm"
                            ][metric]
                            for item in rows
                        )
                        for metric in (
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
            p90_values = {
                name: global_worst[name][
                    confidence_key
                ][key][
                    "robust_regret_ppm"
                ]["p90"]
                for name in candidate_names
            }
            maximum_values = {
                name: global_worst[name][
                    confidence_key
                ][key][
                    "robust_regret_ppm"
                ]["maximum"]
                for name in candidate_names
            }
            p90_optimum = min(
                p90_values.values()
            )
            maximum_optimum = min(
                maximum_values.values()
            )
            global_plateaus[
                confidence_key
            ][key] = {
                "p90_optimum_ppm": (
                    p90_optimum
                ),
                "maximum_optimum_ppm": (
                    maximum_optimum
                ),
                "p90_winners": [
                    name
                    for name, value
                    in p90_values.items()
                    if abs(
                        value - p90_optimum
                    )
                    <= 1e-12
                ],
                "maximum_winners": [
                    name
                    for name, value
                    in maximum_values.items()
                    if abs(
                        value
                        - maximum_optimum
                    )
                    <= 1e-12
                ],
                "plateaus": {
                    f"{threshold:.4f}": {
                        "p90": [
                            name
                            for name, value
                            in p90_values.items()
                            if value
                            <= p90_optimum
                            * (
                                1.0
                                + threshold
                            )
                            + 1e-12
                        ],
                        "maximum": [
                            name
                            for name, value
                            in maximum_values.items()
                            if value
                            <= maximum_optimum
                            * (
                                1.0
                                + threshold
                            )
                            + 1e-12
                        ],
                    }
                    for threshold in (
                        PLATEAU_THRESHOLDS
                    )
                },
            }
            for name in candidate_names:
                global_ratios[name].extend(
                    (
                        safe_ratio(
                            p90_values[name],
                            p90_optimum,
                        ),
                        safe_ratio(
                            maximum_values[
                                name
                            ],
                            maximum_optimum,
                        ),
                    )
                )

    robust_choice_rows = []
    for multiplier, name in zip(
        MULTIPLIERS,
        candidate_names,
    ):
        ratios = global_ratios[name]
        robust_choice_rows.append(
            {
                "key": name,
                "multiplier": multiplier,
                "maximum_relative_regret": (
                    max(ratios)
                ),
                "p90_relative_regret": (
                    quantile(
                        ratios,
                        0.90,
                    )
                ),
                "geometric_mean_relative_regret": (
                    float(
                        math.exp(
                            np.mean(
                                np.log(
                                    ratios
                                )
                            )
                        )
                    )
                ),
            }
        )
    robust_choice_rows.sort(
        key=lambda row: (
            row[
                "maximum_relative_regret"
            ],
            row[
                "geometric_mean_relative_regret"
            ],
            abs(
                row["multiplier"]
                - math.sqrt(2.0)
            ),
            row["multiplier"],
        )
    )
    robust_choice = (
        robust_choice_rows[0]
    )
    local_relative_summary = {
        name: {
            metric: {
                "p50": quantile(
                    values,
                    0.50,
                ),
                "p90": quantile(
                    values,
                    0.90,
                ),
                "p99": quantile(
                    values,
                    0.99,
                ),
                "maximum": max(values),
            }
            for metric, values in (
                local_relative[name].items()
            )
        }
        for name in candidate_names
    }

    anchor_map = {
        "inv_sqrt2": (
            "inv_sqrt2"
        ),
        "m_1p00": "one",
        "sqrt2": "sqrt2",
        "m_2p00": "two",
        "m_3p00": "three",
    }
    anchors_reproduced = True
    for dense_name, e104_name in (
        anchor_map.items()
    ):
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
                actual = global_worst[
                    dense_name
                ][confidence_key][key]
                expected = e104[
                    "headline"
                ][
                    "global_worst_accuracy"
                ][e104_name][
                    confidence_key
                ][key]
                anchors_reproduced &= (
                    all(
                        abs(
                            actual[
                                "robust_regret_ppm"
                            ][metric]
                            - expected[
                                "robust_regret_ppm"
                            ][metric]
                        )
                        <= 1e-9
                        for metric in (
                            "p50",
                            "p90",
                            "p95",
                            "p99",
                            "maximum",
                        )
                    )
                    and all(
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
                )

    exact_validation = (
        exact_fixed_weight_validation()
    )
    maximum_residual = max(
        maximum_root_residual.values()
    )
    monotone_p90 = all(
        all(
            later <= earlier + 1e-9
            for earlier, later in zip(
                [
                    global_worst[name][
                        f"{confidence:.2f}"
                    ][factor_key(factor)][
                        "robust_regret_ppm"
                    ]["p90"]
                    for factor in (
                        FUTURE_FACTORS
                    )
                ],
                [
                    global_worst[name][
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
        for name in candidate_names
        for confidence in (
            CONFIDENCE_LEVELS
        )
    )
    robust_minimality = all(
        robust_choice[
            "maximum_relative_regret"
        ]
        <= row[
            "maximum_relative_regret"
        ]
        + 1e-15
        for row in robust_choice_rows
    )
    if not (
        anchors_reproduced
        and max(
            exact_validation[
                "maximum_size_excess"
            ].values()
        )
        <= 1e-12
        and maximum_residual <= 1e-7
        and monotone_p90
        and robust_minimality
    ):
        raise RuntimeError(
            "E105 validation failed"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "stage_weight_plateau"
        ),
        "source": {
            "pilot": (
                "pacing_coalescing_order_full.json"
            ),
            "stage_weight_frontier": (
                "pacing_coalescing_order_"
                "stage_weights_full.json"
            ),
        },
        "model": (
            "Evaluate 21 fixed full-alpha "
            "Stouffer future-stage multipliers. "
            "Each candidate is individually "
            "valid only if preregistered before "
            "the pilot. Dense-grid optima and "
            "the robust single multiplier are "
            "planning results for a future "
            "independent pilot, not permission "
            "to retune the existing E83 pilot."
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
            "global_plateaus": (
                global_plateaus
            ),
            "robust_choice": (
                robust_choice
            ),
            "robust_choice_ranking": (
                robust_choice_rows
            ),
            "local_winner_counts": (
                local_wins
            ),
            "local_relative_regret": (
                local_relative_summary
            ),
            "condition_count": (
                condition_count
            ),
        },
        "kernels": kernel_worst,
        "validation": {
            "e104_anchor_results_reproduced": (
                anchors_reproduced
            ),
            "fixed_weight_exact_adaptive_design": (
                exact_validation
            ),
            "maximum_root_residual": (
                maximum_root_residual
            ),
            "global_p90_monotone": (
                monotone_p90
            ),
            "robust_choice_minimality": (
                robust_minimality
            ),
        },
        "limitations": [
            (
                "Dense-grid optima and the "
                "robust multiplier are selected "
                "after inspecting E83-derived "
                "results. They are planning "
                "recommendations for a future "
                "independent pilot, not valid "
                "full-alpha choices for E83."
            ),
            (
                "The grid is finite. Plateau "
                "bounds describe tested "
                "multipliers and do not prove a "
                "continuous optimum."
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
