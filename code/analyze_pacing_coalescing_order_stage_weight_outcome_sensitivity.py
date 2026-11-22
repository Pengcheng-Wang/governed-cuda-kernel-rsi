from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pacing_coalescing_order_adaptive_reuse import (
    stouffer_interval,
)
from analyze_pacing_coalescing_order_path_allocation import (
    load_json,
)
from analyze_pacing_coalescing_order_pilot_holdout import (
    FUTURE_FACTORS,
    factor_key,
)
from analyze_pacing_coalescing_order_stage_weight_continuous import (
    Evaluator,
    prepare_context,
    vectorized_stouffer_intervals,
)


POINT_BINOMIAL_REPLICATES = 32
POSTERIOR_PREDICTIVE_REPLICATES = 32
POINT_BINOMIAL_SEED = 10720261114
POSTERIOR_PREDICTIVE_SEED = 10720261115
TIE_TOLERANCE = 1e-10
CANDIDATES = (
    ("one", 1.0),
    ("sqrt2", math.sqrt(2.0)),
    ("continuous", 1.428316903144539),
    ("one_point_five", 1.5),
    ("one_point_six", 1.6),
)


def load_e106(root: Path) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_stage_weight_continuous"
        / "pacing_coalescing_order_stage_weight_continuous_full.json"
    )


def quantile(
    values: list[float],
    probability: float,
) -> float:
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


def balanced_round(
    expected: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    successes = np.floor(
        expected
    ).astype(np.int64)
    target = int(
        math.floor(
            float(np.sum(expected))
            + 0.5
        )
    )
    remaining = target - int(
        np.sum(successes)
    )
    if remaining:
        fractions = expected - successes
        order = np.lexsort(
            (
                np.arange(
                    len(fractions)
                ),
                -fractions,
            )
        )
        successes[
            order[:remaining]
        ] += 1
    return np.minimum(
        successes,
        totals,
    )


def structured_scenarios(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    point = np.asarray(
        context["cells"]["point"],
        dtype=np.float64,
    )
    modes = (
        "floor",
        "nearest",
        "balanced",
        "ceil",
    )
    scenarios = []
    for mode in modes:
        successes = {}
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
            totals = np.asarray(
                context["allocations"][
                    key
                ]["totals"],
                dtype=np.int64,
            )
            expected = point * totals
            if mode == "floor":
                counts = np.floor(
                    expected
                ).astype(np.int64)
            elif mode == "nearest":
                counts = np.floor(
                    expected + 0.5
                ).astype(np.int64)
            elif mode == "balanced":
                counts = balanced_round(
                    expected,
                    totals,
                )
            else:
                counts = np.ceil(
                    expected
                ).astype(np.int64)
            successes[key] = np.minimum(
                counts,
                totals,
            )
        scenarios.append(
            {
                "name": mode,
                "family": "integer_projection",
                "successes": successes,
            }
        )
    return scenarios


def point_binomial_scenarios(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    point = np.asarray(
        context["cells"]["point"],
        dtype=np.float64,
    )
    rng = np.random.default_rng(
        POINT_BINOMIAL_SEED
    )
    scenarios = []
    for replicate in range(
        POINT_BINOMIAL_REPLICATES
    ):
        successes = {}
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
            totals = np.asarray(
                context["allocations"][
                    key
                ]["totals"],
                dtype=np.int64,
            )
            successes[key] = rng.binomial(
                totals,
                point,
            ).astype(np.int64)
        scenarios.append(
            {
                "name": (
                    "point_binomial_"
                    f"{replicate:02d}"
                ),
                "family": "point_binomial",
                "successes": successes,
            }
        )
    return scenarios


def posterior_predictive_scenarios(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    pilot_successes = np.asarray(
        context["cells"]["successes"],
        dtype=np.float64,
    )
    pilot_totals = np.asarray(
        context["cells"]["totals"],
        dtype=np.float64,
    )
    rng = np.random.default_rng(
        POSTERIOR_PREDICTIVE_SEED
    )
    scenarios = []
    for replicate in range(
        POSTERIOR_PREDICTIVE_REPLICATES
    ):
        probability = rng.beta(
            pilot_successes + 0.5,
            pilot_totals
            - pilot_successes
            + 0.5,
        )
        successes = {}
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
            totals = np.asarray(
                context["allocations"][
                    key
                ]["totals"],
                dtype=np.int64,
            )
            successes[key] = rng.binomial(
                totals,
                probability,
            ).astype(np.int64)
        scenarios.append(
            {
                "name": (
                    "posterior_predictive_"
                    f"{replicate:02d}"
                ),
                "family": (
                    "posterior_predictive"
                ),
                "successes": successes,
            }
        )
    return scenarios


def scenario_context(
    context: dict[str, Any],
    successes: dict[str, np.ndarray],
) -> dict[str, Any]:
    result = dict(context)
    result["allocations"] = {
        key: {
            **allocation,
            "successes": successes[key],
        }
        for key, allocation in context[
            "allocations"
        ].items()
    }
    return result


def candidate_summary(
    ratios: list[float],
) -> dict[str, float]:
    values = np.asarray(
        ratios,
        dtype=np.float64,
    )
    return {
        "p50": quantile(
            ratios,
            0.50,
        ),
        "p90": quantile(
            ratios,
            0.90,
        ),
        "p99": quantile(
            ratios,
            0.99,
        ),
        "maximum": float(
            np.max(values)
        ),
        "geometric_mean": float(
            math.exp(
                np.mean(
                    np.log(values)
                )
            )
        ),
    }


def pairwise_summary(
    continuous: list[float],
    sqrt2: list[float],
) -> dict[str, Any]:
    continuous_array = np.asarray(
        continuous,
        dtype=np.float64,
    )
    sqrt2_array = np.asarray(
        sqrt2,
        dtype=np.float64,
    )
    delta = (
        continuous_array
        - sqrt2_array
    )
    ratio = (
        continuous_array
        / sqrt2_array
    )
    return {
        "continuous_lower_count": int(
            np.count_nonzero(
                delta < -TIE_TOLERANCE
            )
        ),
        "sqrt2_lower_count": int(
            np.count_nonzero(
                delta > TIE_TOLERANCE
            )
        ),
        "tie_count": int(
            np.count_nonzero(
                np.abs(delta)
                <= TIE_TOLERANCE
            )
        ),
        "continuous_minus_sqrt2_ppm": {
            "minimum": float(
                np.min(delta)
            ),
            "p10": quantile(
                delta.tolist(),
                0.10,
            ),
            "p50": quantile(
                delta.tolist(),
                0.50,
            ),
            "p90": quantile(
                delta.tolist(),
                0.90,
            ),
            "maximum": float(
                np.max(delta)
            ),
        },
        "continuous_over_sqrt2": {
            "minimum": float(
                np.min(ratio)
            ),
            "p50": quantile(
                ratio.tolist(),
                0.50,
            ),
            "p90": quantile(
                ratio.tolist(),
                0.90,
            ),
            "maximum": float(
                np.max(ratio)
            ),
        },
    }


def scalar_interval_validation(
    context: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    scenario_by_name = {
        scenario["name"]: scenario
        for scenario in scenarios
    }
    selected_names = [
        name
        for name in (
            "floor",
            "ceil",
            "point_binomial_00",
            "posterior_predictive_00",
        )
        if name in scenario_by_name
    ]
    multipliers = np.asarray(
        [
            value
            for _, value in CANDIDATES
        ],
        dtype=np.float64,
    )
    pilot_successes = np.asarray(
        context["cells"]["successes"],
        dtype=np.int64,
    )
    pilot_totals = np.asarray(
        context["cells"]["totals"],
        dtype=np.int64,
    )
    maximum_error = 0.0
    checks = 0
    for scenario_name in selected_names:
        scenario = scenario_by_name[
            scenario_name
        ]
        for confidence in (
            0.95,
            0.99,
        ):
            directional_alpha = (
                (1.0 - confidence)
                * context[
                    "alpha_weights"
                ]
                / 2.0
            )
            for factor in (
                FUTURE_FACTORS[0],
                FUTURE_FACTORS[-1],
            ):
                key = factor_key(factor)
                future_successes = np.asarray(
                    scenario["successes"][
                        key
                    ],
                    dtype=np.int64,
                )
                future_totals = np.asarray(
                    context["allocations"][
                        key
                    ]["totals"],
                    dtype=np.int64,
                )
                lower, upper, _ = (
                    vectorized_stouffer_intervals(
                        multipliers=multipliers,
                        pilot_successes=(
                            pilot_successes
                        ),
                        pilot_totals=pilot_totals,
                        future_successes=(
                            future_successes
                        ),
                        future_totals=future_totals,
                        directional_alpha=(
                            directional_alpha
                        ),
                        future_factor=factor,
                    )
                )
                boundary = np.flatnonzero(
                    (future_successes == 0)
                    | (
                        future_successes
                        == future_totals
                    )
                )
                cell_indices = sorted(
                    {
                        0,
                        len(
                            future_successes
                        )
                        // 2,
                        len(
                            future_successes
                        )
                        - 1,
                        *boundary[:6].tolist(),
                    }
                )
                for multiplier_index, multiplier in enumerate(
                    multipliers
                ):
                    for cell_index in cell_indices:
                        (
                            scalar_lower,
                            scalar_upper,
                            _,
                        ) = stouffer_interval(
                            pilot_successes=int(
                                pilot_successes[
                                    cell_index
                                ]
                            ),
                            pilot_total=int(
                                pilot_totals[
                                    cell_index
                                ]
                            ),
                            future_successes=int(
                                future_successes[
                                    cell_index
                                ]
                            ),
                            future_total=int(
                                future_totals[
                                    cell_index
                                ]
                            ),
                            directional_alpha=float(
                                directional_alpha[
                                    cell_index
                                ]
                            ),
                            future_factor=(
                                float(multiplier)
                                * float(
                                    multiplier
                                )
                                * factor
                            ),
                        )
                        maximum_error = max(
                            maximum_error,
                            abs(
                                lower[
                                    multiplier_index,
                                    cell_index,
                                ]
                                - scalar_lower
                            ),
                            abs(
                                upper[
                                    multiplier_index,
                                    cell_index,
                                ]
                                - scalar_upper
                            ),
                        )
                        checks += 1
    return {
        "checks": checks,
        "maximum_absolute_error": (
            maximum_error
        ),
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    context = prepare_context(args.root)
    e106 = load_e106(args.root)
    candidates = [
        value
        for _, value in CANDIDATES
    ]
    names = [
        name
        for name, _ in CANDIDATES
    ]
    scenarios = [
        *structured_scenarios(
            context
        ),
        *point_binomial_scenarios(
            context
        ),
        *posterior_predictive_scenarios(
            context
        ),
    ]
    ratios = {
        name: []
        for name in names
    }
    family_ratios = {
        family: {
            name: []
            for name in names
        }
        for family in (
            "integer_projection",
            "point_binomial",
            "posterior_predictive",
        )
    }
    winner_counts = {
        name: 0
        for name in names
    }
    family_winner_counts = {
        family: {
            name: 0
            for name in names
        }
        for family in family_ratios
    }
    scenario_robust_choice_counts = {
        name: 0
        for name in names
    }
    family_robust_choice_counts = {
        family: {
            name: 0
            for name in names
        }
        for family in family_ratios
    }
    pairwise_values = {
        "all": {
            "continuous": [],
            "sqrt2": [],
        },
        **{
            family: {
                "continuous": [],
                "sqrt2": [],
            }
            for family in family_ratios
        },
    }
    scenario_rows = []
    maximum_root_residual = 0.0
    nearest_values = None
    for scenario in scenarios:
        evaluator = Evaluator(
            scenario_context(
                context,
                scenario["successes"],
            )
        )
        values, residuals = (
            evaluator.evaluate(
                candidates
            )
        )
        maximum_root_residual = max(
            maximum_root_residual,
            float(
                np.max(residuals)
            ),
        )
        if scenario["name"] == "nearest":
            nearest_values = values.copy()
        optimum = np.min(
            values,
            axis=0,
        )
        relative = values / optimum[
            None, :
        ]
        scenario_maximum = np.max(
            relative,
            axis=1,
        )
        robust_optimum = float(
            np.min(scenario_maximum)
        )
        scenario_robust_winners = [
            names[index]
            for index in np.flatnonzero(
                np.abs(
                    scenario_maximum
                    - robust_optimum
                )
                <= TIE_TOLERANCE
            )
        ]
        for name in (
            scenario_robust_winners
        ):
            scenario_robust_choice_counts[
                name
            ] += 1
            family_robust_choice_counts[
                scenario["family"]
            ][name] += 1
        scenario_wins = {
            name: 0
            for name in names
        }
        for condition_index in range(
            values.shape[1]
        ):
            condition_optimum = optimum[
                condition_index
            ]
            for candidate_index, name in enumerate(
                names
            ):
                value = relative[
                    candidate_index,
                    condition_index,
                ]
                ratios[name].append(
                    float(value)
                )
                family_ratios[
                    scenario["family"]
                ][name].append(
                    float(value)
                )
                if (
                    abs(
                        values[
                            candidate_index,
                            condition_index,
                        ]
                        - condition_optimum
                    )
                    <= TIE_TOLERANCE
                ):
                    winner_counts[
                        name
                    ] += 1
                    family_winner_counts[
                        scenario[
                            "family"
                        ]
                    ][name] += 1
                    scenario_wins[
                        name
                    ] += 1
        continuous_index = names.index(
            "continuous"
        )
        sqrt2_index = names.index(
            "sqrt2"
        )
        for bucket in (
            "all",
            scenario["family"],
        ):
            pairwise_values[bucket][
                "continuous"
            ].extend(
                values[
                    continuous_index
                ].tolist()
            )
            pairwise_values[bucket][
                "sqrt2"
            ].extend(
                values[
                    sqrt2_index
                ].tolist()
            )
        scenario_rows.append(
            {
                "name": scenario["name"],
                "family": scenario["family"],
                "robust_winners": (
                    scenario_robust_winners
                ),
                "maximum_relative_regret": {
                    name: float(
                        scenario_maximum[
                            index
                        ]
                    )
                    for index, name
                    in enumerate(names)
                },
                "condition_wins": (
                    scenario_wins
                ),
            }
        )

    condition_count = len(
        context["conditions"]
    )
    total_comparisons = (
        len(scenarios)
        * condition_count
    )
    candidate_summaries = {
        name: {
            **candidate_summary(
                ratios[name]
            ),
            "condition_wins": (
                winner_counts[name]
            ),
            "condition_win_fraction": (
                winner_counts[name]
                / total_comparisons
            ),
            "within_0p1_percent_fraction": (
                float(
                    np.mean(
                        np.asarray(
                            ratios[name]
                        )
                        <= 1.001
                    )
                )
            ),
            "within_0p25_percent_fraction": (
                float(
                    np.mean(
                        np.asarray(
                            ratios[name]
                        )
                        <= 1.0025
                    )
                )
            ),
            "within_0p5_percent_fraction": (
                float(
                    np.mean(
                        np.asarray(
                            ratios[name]
                        )
                        <= 1.005
                    )
                )
            ),
            "scenario_robust_wins": (
                scenario_robust_choice_counts[
                    name
                ]
            ),
        }
        for name in names
    }
    robust_name = min(
        names,
        key=lambda name: (
            candidate_summaries[name][
                "maximum"
            ],
            candidate_summaries[name][
                "geometric_mean"
            ],
            candidates[
                names.index(name)
            ],
        ),
    )
    family_summaries = {}
    for family in family_ratios:
        family_total = sum(
            1
            for scenario in scenarios
            if scenario["family"]
            == family
        ) * condition_count
        family_summaries[family] = {
            "scenario_count": int(
                family_total
                / condition_count
            ),
            "candidate_relative_regret": {
                name: {
                    **candidate_summary(
                        family_ratios[
                            family
                        ][name]
                    ),
                    "condition_wins": (
                        family_winner_counts[
                            family
                        ][name]
                    ),
                    "condition_win_fraction": (
                        family_winner_counts[
                            family
                        ][name]
                        / family_total
                    ),
                    "scenario_robust_wins": (
                        family_robust_choice_counts[
                            family
                        ][name]
                    ),
                }
                for name in names
            },
            "continuous_vs_sqrt2": (
                pairwise_summary(
                    pairwise_values[
                        family
                    ]["continuous"],
                    pairwise_values[
                        family
                    ]["sqrt2"],
                )
            ),
        }

    if nearest_values is None:
        raise RuntimeError(
            "Missing nearest scenario"
        )
    e106_optima = np.asarray(
        [
            row["robust_regret_ppm"]
            for row in e106["headline"][
                "condition_optima"
            ]
        ],
        dtype=np.float64,
    )
    continuous_index = names.index(
        "continuous"
    )
    sqrt2_index = names.index("sqrt2")
    nearest_continuous = float(
        np.max(
            nearest_values[
                continuous_index
            ]
            / e106_optima
        )
    )
    nearest_sqrt2 = float(
        np.max(
            nearest_values[
                sqrt2_index
            ]
            / e106_optima
        )
    )
    e106_anchor_error = max(
        abs(
            nearest_continuous
            - e106["headline"][
                "robust_choice"
            ][
                "maximum_relative_regret"
            ]
        ),
        abs(
            nearest_sqrt2
            - e106["headline"][
                "sqrt2"
            ][
                "maximum_relative_regret"
            ]
        ),
    )
    all_successes_valid = all(
        np.all(
            scenario["successes"][
                factor_key(factor)
            ]
            >= 0
        )
        and np.all(
            scenario["successes"][
                factor_key(factor)
            ]
            <= context["allocations"][
                factor_key(factor)
            ]["totals"]
        )
        for scenario in scenarios
        for factor in FUTURE_FACTORS
    )
    scalar_validation = (
        scalar_interval_validation(
            context,
            scenarios,
        )
    )
    if not (
        e106_anchor_error <= 2e-8
        and maximum_root_residual
        <= 0.01
        and scalar_validation[
            "maximum_absolute_error"
        ]
        <= 2e-10
        and all_successes_valid
        and total_comparisons
        == len(scenarios)
        * condition_count
    ):
        raise RuntimeError(
            "E107 validation failed"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "stage_weight_outcome_sensitivity"
        ),
        "source": {
            "pilot": (
                "pacing_coalescing_order_full.json"
            ),
            "continuous_frontier": (
                "pacing_coalescing_order_"
                "stage_weight_continuous_full.json"
            ),
        },
        "model": (
            "Compare five fixed stage weights "
            "under four deterministic integer "
            "projections, point-rate binomial "
            "future outcomes, and independent "
            "Jeffreys posterior-predictive future "
            "outcomes. This tests planning "
            "stability, not inferential validity "
            "or a new full-alpha choice for E83."
        ),
        "candidates": [
            {
                "name": name,
                "multiplier": value,
            }
            for name, value in CANDIDATES
        ],
        "scenario_design": {
            "integer_projection": [
                "floor",
                "nearest",
                "balanced",
                "ceil",
            ],
            "point_binomial_replicates": (
                POINT_BINOMIAL_REPLICATES
            ),
            "point_binomial_seed": (
                POINT_BINOMIAL_SEED
            ),
            "posterior_predictive_replicates": (
                POSTERIOR_PREDICTIVE_REPLICATES
            ),
            "posterior_predictive_seed": (
                POSTERIOR_PREDICTIVE_SEED
            ),
            "posterior": (
                "Independent Jeffreys "
                "Beta(successes+0.5, "
                "failures+0.5) per E83 risk cell"
            ),
        },
        "headline": {
            "robust_candidate": (
                robust_name
            ),
            "candidate_relative_regret": (
                candidate_summaries
            ),
            "continuous_vs_sqrt2": (
                pairwise_summary(
                    pairwise_values[
                        "all"
                    ]["continuous"],
                    pairwise_values[
                        "all"
                    ]["sqrt2"],
                )
            ),
            "scenario_robust_choice_counts": (
                scenario_robust_choice_counts
            ),
            "family_summaries": (
                family_summaries
            ),
            "scenario_count": len(
                scenarios
            ),
            "conditions_per_scenario": (
                condition_count
            ),
            "candidate_comparisons": (
                total_comparisons
            ),
        },
        "scenarios": scenario_rows,
        "validation": {
            "e106_nearest_anchor_maximum_error": (
                e106_anchor_error
            ),
            "maximum_interval_root_residual": (
                maximum_root_residual
            ),
            "scalar_interval_validation": (
                scalar_validation
            ),
            "all_success_counts_valid": (
                all_successes_valid
            ),
            "deterministic_seeds": [
                POINT_BINOMIAL_SEED,
                POSTERIOR_PREDICTIVE_SEED,
            ],
        },
        "limitations": [
            (
                "The simulated future counts are "
                "planning sensitivity, not new "
                "measurements or a frequentist "
                "coverage experiment."
            ),
            (
                "The Jeffreys posterior treats "
                "E83 cells as independent and "
                "inherits the rectangular/binomial "
                "limitations of earlier audits."
            ),
            (
                "Candidate-relative regret compares "
                "five fixed weights. It does not "
                "reoptimize a continuous multiplier "
                "inside every simulated outcome."
            ),
            (
                "All candidate conclusions are "
                "post-hoc for E83 and apply only "
                "to preregistration in a future "
                "independent pilot."
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
