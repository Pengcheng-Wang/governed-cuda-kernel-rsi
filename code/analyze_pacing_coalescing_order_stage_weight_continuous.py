from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri_exp
from scipy.stats import binom, norm

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    policy_inventory,
)
from analyze_pacing_coalescing_order_adaptive_reuse import (
    ROOT_EPSILON,
    stouffer_combined_p,
    stouffer_interval,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    DRIFT_METRICS,
    load_e83,
)
from analyze_pacing_coalescing_order_path_allocation import (
    PREVALENCE_DENOMINATOR,
    load_json,
)
from analyze_pacing_coalescing_order_path_floor_frontier import (
    robust_regret_many,
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
from analyze_pacing_coalescing_order_stage_weight_plateau import (
    MULTIPLIERS as E105_MULTIPLIERS,
)


DOMAIN = (0.5, 3.0)
FINE_DOMAIN = (1.2, 1.7)
MESH_STEPS = (0.04, 0.02, 0.01, 0.005)
PLATEAU_THRESHOLDS = (0.0025, 0.005, 0.01)
ALPHA = 0.05
BISECTION_ITERATIONS = 56


def load_e105(root: Path) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_stage_weight_plateau"
        / "pacing_coalescing_order_stage_weight_plateau_full.json"
    )


def prepare_context(root: Path) -> dict[str, Any]:
    e83 = load_e83(root)
    e89 = load_e89(root)
    drift = e83["headline"][
        "drift_by_marginal_fraction_and_order"
    ]
    costs = sorted(drift, key=float)
    orders = sorted(next(iter(drift.values())))
    metrics = list(DRIFT_METRICS)
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
        root=root,
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
        cell_count=len(cells["labels"]),
    )
    alpha_weights = weights[
        "mean_absolute_contrast"
    ]
    equal_weights = weights["equal"]
    pilot_lower, pilot_upper = weighted_intervals(
        successes=cells["successes"],
        totals=cells["totals"],
        confidence=0.95,
        weights=alpha_weights,
        family_divisor=1,
    )
    profiles = prevalence_profiles(
        denominator=PREVALENCE_DENOMINATOR,
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
        pilot_lower=pilot_lower.reshape(
            point_shape
        ),
        pilot_upper=pilot_upper.reshape(
            point_shape
        ),
        e89=e89,
    )
    allocations = {}
    for factor in FUTURE_FACTORS:
        allocations[factor_key(factor)] = (
            holdout_waterfill(
                point=point,
                base_budget=int(
                    np.sum(cells["totals"])
                ),
                future_factor=factor,
                score=pilot["selected_score"],
                alpha_weights=alpha_weights,
                equal_weights=equal_weights,
            )
        )
    conditions = []
    for confidence in CONFIDENCE_LEVELS:
        for factor in FUTURE_FACTORS:
            for metric in ("p90", "maximum"):
                conditions.append(
                    {
                        "confidence": confidence,
                        "future_factor": factor,
                        "metric": metric,
                    }
                )
    return {
        "cells": cells,
        "point_shape": point_shape,
        "alpha_weights": alpha_weights,
        "prevalence": prevalence,
        "pilot": pilot,
        "allocations": allocations,
        "conditions": conditions,
    }


def inverse_survival(
    log_probability: np.ndarray,
) -> np.ndarray:
    return np.clip(
        -ndtri_exp(log_probability),
        -40.0,
        40.0,
    )


def vectorized_stouffer_intervals(
    *,
    multipliers: np.ndarray,
    pilot_successes: np.ndarray,
    pilot_totals: np.ndarray,
    future_successes: np.ndarray,
    future_totals: np.ndarray,
    directional_alpha: np.ndarray,
    future_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    multiplier_column = multipliers[:, None]
    future_weight = (
        multiplier_column
        * math.sqrt(future_factor)
    )
    denominator = np.sqrt(
        1.0
        + multiplier_column
        * multiplier_column
        * future_factor
    )
    threshold = norm.isf(
        directional_alpha
    )[None, :]
    pilot_successes = pilot_successes[
        None, :
    ]
    pilot_totals = pilot_totals[None, :]
    future_successes = future_successes[
        None, :
    ]
    future_totals = future_totals[None, :]

    def statistic(
        probability: np.ndarray,
        *,
        lower_tail: bool,
    ) -> np.ndarray:
        if lower_tail:
            pilot_log_probability = binom.logcdf(
                pilot_successes,
                pilot_totals,
                probability,
            )
            future_log_probability = binom.logcdf(
                future_successes,
                future_totals,
                probability,
            )
        else:
            pilot_log_probability = binom.logsf(
                pilot_successes - 1,
                pilot_totals,
                probability,
            )
            future_log_probability = binom.logsf(
                future_successes - 1,
                future_totals,
                probability,
            )
        return (
            (
                inverse_survival(
                    pilot_log_probability
                )
                + future_weight
                * inverse_survival(
                    future_log_probability
                )
            )
            / denominator
            - threshold
        )

    shape = (
        len(multipliers),
        pilot_successes.shape[1],
    )
    minimum = np.full(
        shape,
        ROOT_EPSILON,
        dtype=np.float64,
    )
    maximum = np.full(
        shape,
        1.0 - ROOT_EPSILON,
        dtype=np.float64,
    )

    lower_at_minimum = statistic(
        minimum,
        lower_tail=False,
    )
    lower_at_maximum = statistic(
        maximum,
        lower_tail=False,
    )
    lower_zero = lower_at_minimum <= 0.0
    lower_one = lower_at_maximum >= 0.0
    lower_active = ~(
        lower_zero | lower_one
    )
    lower_left = minimum.copy()
    lower_right = maximum.copy()
    for _ in range(BISECTION_ITERATIONS):
        middle = (
            lower_left + lower_right
        ) / 2.0
        value = statistic(
            middle,
            lower_tail=False,
        )
        move_left = (
            lower_active & (value > 0.0)
        )
        lower_left = np.where(
            move_left,
            middle,
            lower_left,
        )
        lower_right = np.where(
            lower_active & ~move_left,
            middle,
            lower_right,
        )
    lower = np.where(
        lower_zero,
        0.0,
        np.where(
            lower_one,
            1.0,
            (
                lower_left + lower_right
            )
            / 2.0,
        ),
    )

    upper_at_minimum = statistic(
        minimum,
        lower_tail=True,
    )
    upper_at_maximum = statistic(
        maximum,
        lower_tail=True,
    )
    upper_one = upper_at_maximum <= 0.0
    upper_zero = upper_at_minimum >= 0.0
    upper_active = ~(
        upper_one | upper_zero
    )
    upper_left = minimum.copy()
    upper_right = maximum.copy()
    for _ in range(BISECTION_ITERATIONS):
        middle = (
            upper_left + upper_right
        ) / 2.0
        value = statistic(
            middle,
            lower_tail=True,
        )
        move_left = (
            upper_active & (value < 0.0)
        )
        upper_left = np.where(
            move_left,
            middle,
            upper_left,
        )
        upper_right = np.where(
            upper_active & ~move_left,
            middle,
            upper_right,
        )
    upper = np.where(
        upper_one,
        1.0,
        np.where(
            upper_zero,
            0.0,
            (
                upper_left + upper_right
            )
            / 2.0,
        ),
    )

    lower_residual = np.where(
        lower_active,
        np.abs(
            statistic(
                lower,
                lower_tail=False,
            )
        ),
        0.0,
    )
    upper_residual = np.where(
        upper_active,
        np.abs(
            statistic(
                upper,
                lower_tail=True,
            )
        ),
        0.0,
    )
    residual = np.maximum(
        np.max(
            lower_residual,
            axis=1,
        ),
        np.max(
            upper_residual,
            axis=1,
        ),
    )
    return lower, upper, residual


class Evaluator:
    def __init__(
        self,
        context: dict[str, Any],
    ) -> None:
        self.context = context
        self.cache: dict[
            float,
            tuple[np.ndarray, float],
        ] = {}

    @staticmethod
    def key(multiplier: float) -> float:
        return round(float(multiplier), 14)

    def evaluate(
        self,
        multipliers: list[float]
        | tuple[float, ...]
        | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        requested = [
            self.key(value)
            for value in multipliers
        ]
        missing = sorted(
            {
                value
                for value in requested
                if value not in self.cache
            }
        )
        if missing:
            values, residuals = (
                self.evaluate_batch(
                    np.asarray(
                        missing,
                        dtype=np.float64,
                    )
                )
            )
            for index, value in enumerate(
                missing
            ):
                self.cache[value] = (
                    values[index],
                    float(
                        residuals[index]
                    ),
                )
        return (
            np.asarray(
                [
                    self.cache[value][0]
                    for value in requested
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    self.cache[value][1]
                    for value in requested
                ],
                dtype=np.float64,
            ),
        )

    def evaluate_batch(
        self,
        multipliers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        context = self.context
        cells = context["cells"]
        point_shape = context[
            "point_shape"
        ]
        prevalence = context["prevalence"]
        pilot = context["pilot"]
        alpha_weights = context[
            "alpha_weights"
        ]
        values = np.empty(
            (
                len(multipliers),
                len(
                    context["conditions"]
                ),
            ),
            dtype=np.float64,
        )
        residuals = np.zeros(
            len(multipliers),
            dtype=np.float64,
        )
        condition_index = 0
        for confidence in CONFIDENCE_LEVELS:
            directional_alpha = (
                (1.0 - confidence)
                * alpha_weights
                / 2.0
            )
            for factor in FUTURE_FACTORS:
                allocation = context[
                    "allocations"
                ][factor_key(factor)]
                lower, upper, residual = (
                    vectorized_stouffer_intervals(
                        multipliers=multipliers,
                        pilot_successes=np.asarray(
                            cells["successes"],
                            dtype=np.int64,
                        ),
                        pilot_totals=np.asarray(
                            cells["totals"],
                            dtype=np.int64,
                        ),
                        future_successes=np.asarray(
                            allocation[
                                "successes"
                            ],
                            dtype=np.int64,
                        ),
                        future_totals=np.asarray(
                            allocation["totals"],
                            dtype=np.int64,
                        ),
                        directional_alpha=(
                            directional_alpha
                        ),
                        future_factor=factor,
                    )
                )
                residuals = np.maximum(
                    residuals,
                    residual,
                )
                lower = lower.reshape(
                    (
                        len(multipliers),
                        *point_shape,
                    )
                )
                upper = upper.reshape(
                    (
                        len(multipliers),
                        *point_shape,
                    )
                )
                global_p90 = np.zeros(
                    len(multipliers),
                    dtype=np.float64,
                )
                global_maximum = np.zeros(
                    len(multipliers),
                    dtype=np.float64,
                )
                for kernel_name in KERNELS:
                    kernel_p90 = np.zeros(
                        len(multipliers),
                        dtype=np.float64,
                    )
                    kernel_maximum = np.zeros(
                        len(multipliers),
                        dtype=np.float64,
                    )
                    for row in pilot["cache"][
                        kernel_name
                    ]:
                        regrets = robust_regret_many(
                            prevalence=prevalence,
                            selected=row[
                                "selected"
                            ],
                            coefficients=row[
                                "coefficients"
                            ],
                            lowers=lower,
                            uppers=upper,
                        )
                        ppm = (
                            regrets
                            * 1_000_000.0
                        )
                        p90_index = min(
                            ppm.shape[1] - 1,
                            max(
                                0,
                                math.ceil(
                                    0.90
                                    * ppm.shape[1]
                                )
                                - 1,
                            ),
                        )
                        p90 = np.partition(
                            ppm,
                            p90_index,
                            axis=1,
                        )[:, p90_index]
                        maximum = np.max(
                            ppm,
                            axis=1,
                        )
                        kernel_p90 = (
                            np.maximum(
                                kernel_p90,
                                p90,
                            )
                        )
                        kernel_maximum = (
                            np.maximum(
                                kernel_maximum,
                                maximum,
                            )
                        )
                    global_p90 = np.maximum(
                        global_p90,
                        kernel_p90,
                    )
                    global_maximum = np.maximum(
                        global_maximum,
                        kernel_maximum,
                    )
                values[
                    :,
                    condition_index,
                ] = global_p90
                condition_index += 1
                values[
                    :,
                    condition_index,
                ] = global_maximum
                condition_index += 1
        return values, residuals


def uniform_grid(
    lower: float,
    upper: float,
    step: float,
) -> np.ndarray:
    count = int(
        round(
            (upper - lower) / step
        )
    )
    return np.linspace(
        lower,
        upper,
        count + 1,
        dtype=np.float64,
    )


def parabolic_vertex(
    left_x: float,
    middle_x: float,
    right_x: float,
    left_y: float,
    middle_y: float,
    right_y: float,
) -> float:
    denominator = (
        left_y
        - 2.0 * middle_y
        + right_y
    )
    if (
        abs(denominator) <= 1e-20
        or not math.isclose(
            middle_x - left_x,
            right_x - middle_x,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return middle_x
    step = middle_x - left_x
    offset = (
        0.5
        * step
        * (left_y - right_y)
        / denominator
    )
    return min(
        right_x,
        max(
            left_x,
            middle_x + offset,
        ),
    )


def mesh_optima(
    multipliers: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argmin(
        values,
        axis=0,
    )
    optimum_multipliers = (
        multipliers[indices]
    )
    optimum_values = values[
        indices,
        np.arange(values.shape[1]),
    ]
    return (
        optimum_multipliers,
        optimum_values,
    )


def exact_fixed_weight_validation(
    multipliers: list[float],
) -> dict[str, Any]:
    maximum_excess = {
        f"{multiplier:.12f}": 0.0
        for multiplier in multipliers
    }
    checks = 0

    def future_total(
        rule: int,
        pilot_successes: int,
        pilot_total: int,
    ) -> int:
        if rule == 0:
            return 1 + pilot_successes
        return (
            6
            if 2 * pilot_successes
            >= pilot_total
            else 1
        )

    for pilot_total in (2, 3, 4):
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
                        f"{multiplier:.12f}": {
                            "upper": 0.0,
                            "lower": 0.0,
                        }
                        for multiplier
                        in multipliers
                    }
                    for pilot_successes in range(
                        pilot_total + 1
                    ):
                        pilot_mass = binom.pmf(
                            pilot_successes,
                            pilot_total,
                            probability,
                        )
                        holdout_total = future_total(
                            rule,
                            pilot_successes,
                            pilot_total,
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
                                pilot_successes - 1,
                                pilot_total,
                                probability,
                            )
                            future_upper = binom.sf(
                                future_successes - 1,
                                holdout_total,
                                probability,
                            )
                            pilot_lower = binom.cdf(
                                pilot_successes,
                                pilot_total,
                                probability,
                            )
                            future_lower = binom.cdf(
                                future_successes,
                                holdout_total,
                                probability,
                            )
                            for multiplier in multipliers:
                                key = (
                                    f"{multiplier:.12f}"
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
                    for multiplier in multipliers:
                        key = (
                            f"{multiplier:.12f}"
                        )
                        for direction in (
                            "upper",
                            "lower",
                        ):
                            maximum_excess[key] = max(
                                maximum_excess[key],
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


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    context = prepare_context(args.root)
    evaluator = Evaluator(context)
    e105 = load_e105(args.root)

    guard = uniform_grid(
        DOMAIN[0],
        DOMAIN[1],
        0.1,
    )
    fine = uniform_grid(
        FINE_DOMAIN[0],
        FINE_DOMAIN[1],
        MESH_STEPS[-1],
    )
    initial = np.asarray(
        sorted(
            {
                *guard.tolist(),
                *fine.tolist(),
                *E105_MULTIPLIERS,
                math.sqrt(2.0),
            }
        ),
        dtype=np.float64,
    )
    initial_values, initial_residuals = (
        evaluator.evaluate(initial)
    )

    mesh_convergence = []
    for step in MESH_STEPS:
        mesh = uniform_grid(
            FINE_DOMAIN[0],
            FINE_DOMAIN[1],
            step,
        )
        mesh_values, _ = evaluator.evaluate(
            mesh
        )
        (
            mesh_condition_multipliers,
            mesh_condition_values,
        ) = mesh_optima(
            mesh,
            mesh_values,
        )
        ratios = (
            mesh_values
            / mesh_condition_values[
                None, :
            ]
        )
        envelope = np.max(
            ratios,
            axis=1,
        )
        best_index = int(
            np.argmin(envelope)
        )
        mesh_convergence.append(
            {
                "step": step,
                "robust_multiplier": float(
                    mesh[best_index]
                ),
                "maximum_relative_regret": (
                    float(
                        envelope[
                            best_index
                        ]
                    )
                ),
                "condition_optimum_minimum": (
                    float(
                        np.min(
                            mesh_condition_multipliers
                        )
                    )
                ),
                "condition_optimum_maximum": (
                    float(
                        np.max(
                            mesh_condition_multipliers
                        )
                    )
                ),
            }
        )

    finest = uniform_grid(
        FINE_DOMAIN[0],
        FINE_DOMAIN[1],
        MESH_STEPS[-1],
    )
    finest_values, _ = evaluator.evaluate(
        finest
    )
    finest_indices = np.argmin(
        finest_values,
        axis=0,
    )
    vertex_candidates = []
    for condition_index, index in enumerate(
        finest_indices
    ):
        if 0 < index < len(finest) - 1:
            vertex_candidates.append(
                parabolic_vertex(
                    float(finest[index - 1]),
                    float(finest[index]),
                    float(finest[index + 1]),
                    float(
                        finest_values[
                            index - 1,
                            condition_index,
                        ]
                    ),
                    float(
                        finest_values[
                            index,
                            condition_index,
                        ]
                    ),
                    float(
                        finest_values[
                            index + 1,
                            condition_index,
                        ]
                    ),
                )
            )
    vertex_candidates = sorted(
        {
            round(value, 14)
            for value in vertex_candidates
        }
    )
    if vertex_candidates:
        evaluator.evaluate(vertex_candidates)

    condition_optima = []
    condition_values = np.empty(
        len(context["conditions"]),
        dtype=np.float64,
    )
    condition_multipliers = np.empty(
        len(context["conditions"]),
        dtype=np.float64,
    )
    candidate_multipliers = sorted(
        {
            *finest.tolist(),
            *vertex_candidates,
        }
    )
    candidate_values, _ = evaluator.evaluate(
        candidate_multipliers
    )
    candidate_multipliers_array = (
        np.asarray(
            candidate_multipliers,
            dtype=np.float64,
        )
    )
    for index, condition in enumerate(
        context["conditions"]
    ):
        best_index = int(
            np.argmin(
                candidate_values[:, index]
            )
        )
        condition_values[index] = (
            candidate_values[
                best_index,
                index,
            ]
        )
        condition_multipliers[index] = (
            candidate_multipliers_array[
                best_index
            ]
        )
        condition_optima.append(
            {
                **condition,
                "multiplier": float(
                    condition_multipliers[
                        index
                    ]
                ),
                "robust_regret_ppm": float(
                    condition_values[index]
                ),
            }
        )

    envelope_multipliers = sorted(
        {
            *initial.tolist(),
            *vertex_candidates,
        }
    )
    envelope_multipliers_array = (
        np.asarray(
            envelope_multipliers,
            dtype=np.float64,
        )
    )
    all_candidate_values, _ = (
        evaluator.evaluate(
            envelope_multipliers
        )
    )
    all_candidate_ratios = (
        all_candidate_values
        / condition_values[None, :]
    )
    all_candidate_envelope = np.max(
        all_candidate_ratios,
        axis=1,
    )
    preliminary_index = int(
        np.argmin(
            all_candidate_envelope
        )
    )
    preliminary_multiplier = float(
        envelope_multipliers_array[
            preliminary_index
        ]
    )
    nearest_index = int(
        np.argmin(
            np.abs(
                finest
                - preliminary_multiplier
            )
        )
    )
    nearest_index = min(
        len(finest) - 2,
        max(1, nearest_index),
    )
    robust_vertex = parabolic_vertex(
        float(finest[nearest_index - 1]),
        float(finest[nearest_index]),
        float(finest[nearest_index + 1]),
        float(
            np.max(
                finest_values[
                    nearest_index - 1
                ]
                / condition_values
            )
        ),
        float(
            np.max(
                finest_values[
                    nearest_index
                ]
                / condition_values
            )
        ),
        float(
            np.max(
                finest_values[
                    nearest_index + 1
                ]
                / condition_values
            )
        ),
    )
    intersection_candidates = []
    bracket_left = float(
        finest[
            max(
                0,
                nearest_index - 1,
            )
        ]
    )
    bracket_right = float(
        finest[
            min(
                len(finest) - 1,
                nearest_index + 1,
            )
        ]
    )
    bracket_values, _ = evaluator.evaluate(
        [
            bracket_left,
            preliminary_multiplier,
            bracket_right,
        ]
    )
    bracket_ratios = (
        bracket_values
        / condition_values[None, :]
    )
    near_active = sorted(
        {
            int(index)
            for row in bracket_ratios
            for index in np.argsort(
                row
            )[-6:]
        }
    )

    def relative_difference(
        multiplier: float,
        first: int,
        second: int,
    ) -> float:
        values, _ = evaluator.evaluate(
            [multiplier]
        )
        return float(
            values[0, first]
            / condition_values[first]
            - values[0, second]
            / condition_values[second]
        )

    for first_position, first in enumerate(
        near_active
    ):
        for second in near_active[
            first_position + 1:
        ]:
            left_difference = (
                bracket_ratios[0, first]
                - bracket_ratios[0, second]
            )
            right_difference = (
                bracket_ratios[-1, first]
                - bracket_ratios[-1, second]
            )
            if (
                left_difference
                * right_difference
                > 0.0
            ):
                continue
            if abs(left_difference) <= 1e-14:
                root = bracket_left
            elif abs(
                right_difference
            ) <= 1e-14:
                root = bracket_right
            else:
                root = float(
                    brentq(
                        lambda value: (
                            relative_difference(
                                value,
                                first,
                                second,
                            )
                        ),
                        bracket_left,
                        bracket_right,
                        xtol=1e-10,
                        rtol=1e-12,
                    )
                )
            intersection_candidates.append(
                root
            )
    robust_candidates = sorted(
        {
            preliminary_multiplier,
            robust_vertex,
            math.sqrt(2.0),
            *intersection_candidates,
            float(
                finest[
                    nearest_index - 1
                ]
            ),
            float(finest[nearest_index]),
            float(
                finest[
                    nearest_index + 1
                ]
            ),
        }
    )
    robust_values, robust_residuals = (
        evaluator.evaluate(
            robust_candidates
        )
    )
    robust_ratios = (
        robust_values
        / condition_values[None, :]
    )
    robust_envelope = np.max(
        robust_ratios,
        axis=1,
    )
    robust_index = int(
        np.argmin(robust_envelope)
    )
    robust_multiplier = float(
        robust_candidates[
            robust_index
        ]
    )
    robust_ratio_row = robust_ratios[
        robust_index
    ]
    robust_choice = {
        "multiplier": robust_multiplier,
        "maximum_relative_regret": float(
            np.max(robust_ratio_row)
        ),
        "p90_relative_regret": float(
            np.quantile(
                robust_ratio_row,
                0.90,
                method="higher",
            )
        ),
        "geometric_mean_relative_regret": (
            float(
                math.exp(
                    np.mean(
                        np.log(
                            robust_ratio_row
                        )
                    )
                )
            )
        ),
        "active_conditions": [
            {
                **context["conditions"][
                    int(index)
                ],
                "relative_regret": float(
                    robust_ratio_row[
                        index
                    ]
                ),
            }
            for index in np.flatnonzero(
                np.abs(
                    robust_ratio_row
                    - np.max(
                        robust_ratio_row
                    )
                )
                <= 1e-8
            )
        ],
    }

    def relative_envelope(
        multiplier: float,
    ) -> float:
        values, _ = evaluator.evaluate(
            [multiplier]
        )
        return float(
            np.max(
                values[0]
                / condition_values
            )
        )

    envelope_grid = (
        all_candidate_envelope
    )
    plateau_intervals = {}
    plateau_root_residual = 0.0
    for threshold in PLATEAU_THRESHOLDS:
        target = 1.0 + threshold
        inside = (
            envelope_grid <= target
        )
        robust_position = int(
            np.argmin(
                np.abs(
                    envelope_multipliers_array
                    - robust_multiplier
                )
            )
        )
        if not inside[robust_position]:
            raise RuntimeError(
                "Robust choice outside plateau"
            )
        left_index = robust_position
        while (
            left_index > 0
            and inside[left_index - 1]
        ):
            left_index -= 1
        right_index = robust_position
        while (
            right_index
            < len(inside) - 1
            and inside[right_index + 1]
        ):
            right_index += 1

        if left_index == 0:
            lower = float(
                envelope_multipliers_array[0]
            )
        else:
            left = float(
                envelope_multipliers_array[
                    left_index - 1
                ]
            )
            right = float(
                envelope_multipliers_array[
                    left_index
                ]
            )
            lower = float(
                brentq(
                    lambda value: (
                        relative_envelope(
                            value
                        )
                        - target
                    ),
                    left,
                    right,
                    xtol=1e-10,
                    rtol=1e-12,
                )
            )
            plateau_root_residual = max(
                plateau_root_residual,
                abs(
                    relative_envelope(
                        lower
                    )
                    - target
                ),
            )
        if right_index == len(inside) - 1:
            upper = float(
                envelope_multipliers_array[
                    -1
                ]
            )
        else:
            left = float(
                envelope_multipliers_array[
                    right_index
                ]
            )
            right = float(
                envelope_multipliers_array[
                    right_index + 1
                ]
            )
            upper = float(
                brentq(
                    lambda value: (
                        relative_envelope(
                            value
                        )
                        - target
                    ),
                    left,
                    right,
                    xtol=1e-10,
                    rtol=1e-12,
                )
            )
            plateau_root_residual = max(
                plateau_root_residual,
                abs(
                    relative_envelope(
                        upper
                    )
                    - target
                ),
            )
        plateau_intervals[
            f"{threshold:.4f}"
        ] = {
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "contains_sqrt2": (
                lower
                <= math.sqrt(2.0)
                <= upper
            ),
        }

    sqrt2_values, _ = evaluator.evaluate(
        [math.sqrt(2.0)]
    )
    sqrt2_ratios = (
        sqrt2_values[0]
        / condition_values
    )
    sqrt2_summary = {
        "multiplier": math.sqrt(2.0),
        "maximum_relative_regret": float(
            np.max(sqrt2_ratios)
        ),
        "p90_relative_regret": float(
            np.quantile(
                sqrt2_ratios,
                0.90,
                method="higher",
            )
        ),
        "geometric_mean_relative_regret": (
            float(
                math.exp(
                    np.mean(
                        np.log(
                            sqrt2_ratios
                        )
                    )
                )
            )
        ),
    }

    e105_anchor_values, _ = (
        evaluator.evaluate(
            E105_MULTIPLIERS
        )
    )
    maximum_anchor_error = 0.0
    condition_lookup = {
        (
            condition["confidence"],
            condition["future_factor"],
            condition["metric"],
        ): index
        for index, condition in enumerate(
            context["conditions"]
        )
    }
    e105_keys = {
        round(
            item["value"],
            14,
        ): item["key"]
        for item in e105["multipliers"]
    }
    for multiplier_index, multiplier in enumerate(
        E105_MULTIPLIERS
    ):
        name = e105_keys[
            round(multiplier, 14)
        ]
        for confidence in CONFIDENCE_LEVELS:
            confidence_key = (
                f"{confidence:.2f}"
            )
            for factor in FUTURE_FACTORS:
                key = factor_key(factor)
                for metric in (
                    "p90",
                    "maximum",
                ):
                    condition_index = (
                        condition_lookup[
                            (
                                confidence,
                                factor,
                                metric,
                            )
                        ]
                    )
                    expected = e105[
                        "headline"
                    ][
                        "global_worst_accuracy"
                    ][name][
                        confidence_key
                    ][key][
                        "robust_regret_ppm"
                    ][metric]
                    maximum_anchor_error = max(
                        maximum_anchor_error,
                        abs(
                            e105_anchor_values[
                                multiplier_index,
                                condition_index,
                            ]
                            - expected
                        ),
                    )

    scalar_checks = []
    scalar_maximum_error = 0.0
    sample_indices = (
        0,
        len(
            context["cells"][
                "successes"
            ]
        )
        // 2,
        len(
            context["cells"][
                "successes"
            ]
        )
        - 1,
    )
    for multiplier in (
        math.sqrt(2.0),
        robust_multiplier,
    ):
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            factor = FUTURE_FACTORS[
                0
                if confidence
                == CONFIDENCE_LEVELS[0]
                else -1
            ]
            allocation = context[
                "allocations"
            ][factor_key(factor)]
            directional_alpha = (
                (1.0 - confidence)
                * context[
                    "alpha_weights"
                ]
                / 2.0
            )
            vector_lower, vector_upper, _ = (
                vectorized_stouffer_intervals(
                    multipliers=np.asarray(
                        [multiplier]
                    ),
                    pilot_successes=np.asarray(
                        context["cells"][
                            "successes"
                        ]
                    ),
                    pilot_totals=np.asarray(
                        context["cells"][
                            "totals"
                        ]
                    ),
                    future_successes=np.asarray(
                        allocation[
                            "successes"
                        ]
                    ),
                    future_totals=np.asarray(
                        allocation["totals"]
                    ),
                    directional_alpha=(
                        directional_alpha
                    ),
                    future_factor=factor,
                )
            )
            for cell_index in sample_indices:
                lower, upper, _ = (
                    stouffer_interval(
                        pilot_successes=int(
                            context["cells"][
                                "successes"
                            ][cell_index]
                        ),
                        pilot_total=int(
                            context["cells"][
                                "totals"
                            ][cell_index]
                        ),
                        future_successes=int(
                            allocation[
                                "successes"
                            ][cell_index]
                        ),
                        future_total=int(
                            allocation["totals"][
                                cell_index
                            ]
                        ),
                        directional_alpha=float(
                            directional_alpha[
                                cell_index
                            ]
                        ),
                        future_factor=(
                            multiplier
                            * multiplier
                            * factor
                        ),
                    )
                )
                error = max(
                    abs(
                        vector_lower[
                            0,
                            cell_index,
                        ]
                        - lower
                    ),
                    abs(
                        vector_upper[
                            0,
                            cell_index,
                        ]
                        - upper
                    ),
                )
                scalar_maximum_error = max(
                    scalar_maximum_error,
                    error,
                )
                scalar_checks.append(
                    {
                        "multiplier": multiplier,
                        "confidence": confidence,
                        "future_factor": factor,
                        "cell_index": cell_index,
                        "maximum_absolute_error": (
                            error
                        ),
                    }
                )

    exact_candidates = sorted(
        {
            robust_multiplier,
            math.sqrt(2.0),
            *(
                interval[boundary]
                for interval
                in plateau_intervals.values()
                for boundary
                in ("lower", "upper")
            ),
        }
    )
    exact_validation = (
        exact_fixed_weight_validation(
            exact_candidates
        )
    )
    mesh_multiplier_shift = abs(
        mesh_convergence[-1][
            "robust_multiplier"
        ]
        - mesh_convergence[-2][
            "robust_multiplier"
        ]
    )
    mesh_objective_shift = abs(
        mesh_convergence[-1][
            "maximum_relative_regret"
        ]
        - mesh_convergence[-2][
            "maximum_relative_regret"
        ]
    )
    neighbor_delta = MESH_STEPS[-1] / 4.0
    neighbor_multipliers = [
        robust_multiplier
        - neighbor_delta,
        robust_multiplier
        + neighbor_delta,
    ]
    neighbor_values, _ = evaluator.evaluate(
        neighbor_multipliers
    )
    neighbor_envelope = np.max(
        neighbor_values
        / condition_values[None, :],
        axis=1,
    )
    robust_neighbor_minimality = all(
        value
        >= robust_choice[
            "maximum_relative_regret"
        ]
        - 1e-10
        for value in neighbor_envelope
    )
    boundary_values, _ = evaluator.evaluate(
        list(DOMAIN)
    )
    boundary_envelope = np.max(
        boundary_values
        / condition_values[None, :],
        axis=1,
    )
    boundary_margin = float(
        np.min(boundary_envelope)
        - robust_choice[
            "maximum_relative_regret"
        ]
    )
    maximum_root_residual = max(
        residual
        for _, residual
        in evaluator.cache.values()
    )

    maximum_exact_excess = max(
        exact_validation[
            "maximum_size_excess"
        ].values()
    )
    if not (
        maximum_anchor_error <= 2e-8
        and scalar_maximum_error <= 2e-10
        and maximum_root_residual <= 2e-7
        and plateau_root_residual <= 2e-8
        and maximum_exact_excess <= 1e-12
        and robust_neighbor_minimality
        and boundary_margin > 0.0
    ):
        raise RuntimeError(
            "E106 validation failed"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "stage_weight_continuous"
        ),
        "source": {
            "pilot": (
                "pacing_coalescing_order_full.json"
            ),
            "dense_frontier": (
                "pacing_coalescing_order_"
                "stage_weight_plateau_full.json"
            ),
        },
        "model": (
            "Use vectorized exact binomial-tail "
            "bisection, nested meshes, parabolic "
            "candidate refinement, and direct "
            "plateau root solving to estimate the "
            "continuous fixed Stouffer multiplier "
            "frontier on [0.5, 3.0]. Results are "
            "post-hoc planning for preregistration "
            "in a future independent pilot."
        ),
        "domain": list(DOMAIN),
        "fine_domain": list(FINE_DOMAIN),
        "mesh_steps": list(MESH_STEPS),
        "headline": {
            "robust_choice": robust_choice,
            "sqrt2": sqrt2_summary,
            "continuous_plateaus": (
                plateau_intervals
            ),
            "condition_optimum_range": {
                "minimum": float(
                    np.min(
                        condition_multipliers
                    )
                ),
                "maximum": float(
                    np.max(
                        condition_multipliers
                    )
                ),
            },
            "condition_optima": (
                condition_optima
            ),
            "mesh_convergence": (
                mesh_convergence
            ),
            "continuous_improvement_over_sqrt2": {
                "maximum_relative_regret": (
                    sqrt2_summary[
                        "maximum_relative_regret"
                    ]
                    - robust_choice[
                        "maximum_relative_regret"
                    ]
                ),
                "multiplier_difference": (
                    robust_multiplier
                    - math.sqrt(2.0)
                ),
            },
        },
        "validation": {
            "e105_anchor_maximum_absolute_error": (
                maximum_anchor_error
            ),
            "scalar_interval_checks": (
                scalar_checks
            ),
            "scalar_interval_maximum_absolute_error": (
                scalar_maximum_error
            ),
            "maximum_interval_root_residual": (
                maximum_root_residual
            ),
            "plateau_root_maximum_residual": (
                plateau_root_residual
            ),
            "fixed_weight_exact_adaptive_design": (
                exact_validation
            ),
            "mesh_finest_multiplier_shift": (
                mesh_multiplier_shift
            ),
            "mesh_finest_objective_shift": (
                mesh_objective_shift
            ),
            "robust_neighbor_minimality": (
                robust_neighbor_minimality
            ),
            "domain_boundary_margin": (
                boundary_margin
            ),
            "evaluated_multiplier_count": (
                len(evaluator.cache)
            ),
            "intersection_candidate_count": (
                len(
                    set(
                        round(
                            value,
                            12,
                        )
                        for value
                        in intersection_candidates
                    )
                )
            ),
        },
        "limitations": [
            (
                "The numerical continuous limit "
                "uses nested meshes plus local "
                "candidate and root refinement. It "
                "is not a symbolic proof of a "
                "unique global optimum."
            ),
            (
                "The multiplier is selected after "
                "inspecting E83-derived outcomes. "
                "It may be preregistered for a "
                "future independent pilot but may "
                "not retune E83 at full alpha."
            ),
            (
                "Projected future successes retain "
                "E83 rates and integer rounding. "
                "This is planning sensitivity, not "
                "new measured paths."
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
