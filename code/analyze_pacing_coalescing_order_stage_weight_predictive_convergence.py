from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtri_exp
from scipy.stats import beta, binom, norm

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
)
from analyze_pacing_coalescing_order_path_allocation import (
    load_json,
)
from analyze_pacing_coalescing_order_path_floor_frontier import (
    robust_regret_many,
)
from analyze_pacing_coalescing_order_pilot_holdout import (
    CONFIDENCE_LEVELS,
    FUTURE_FACTORS,
    factor_key,
)
from analyze_pacing_coalescing_order_stage_weight_continuous import (
    BISECTION_ITERATIONS,
    ROOT_EPSILON,
    Evaluator,
    prepare_context,
)
from analyze_pacing_coalescing_order_stage_weight_outcome_sensitivity import (
    pairwise_summary,
    scenario_context,
)


SEEDS = (
    10720261115,
    10720261116,
    10720261117,
    10720261118,
)
REPLICATES_PER_SEED = 32
TIE_TOLERANCE = 1e-10
CANDIDATES = (
    ("sqrt2", math.sqrt(2.0)),
    ("continuous", 1.428316903144539),
)


def load_e107(root: Path) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_stage_weight_outcome_sensitivity"
        / "pacing_coalescing_order_stage_weight_outcome_sensitivity_full.json"
    )


def inverse_survival(
    log_probability: np.ndarray,
) -> np.ndarray:
    return np.clip(
        -ndtri_exp(log_probability),
        -40.0,
        40.0,
    )


def batched_stouffer_intervals(
    *,
    multipliers: np.ndarray,
    pilot_successes: np.ndarray,
    pilot_totals: np.ndarray,
    future_successes: np.ndarray,
    future_totals: np.ndarray,
    directional_alpha: np.ndarray,
    future_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    multiplier = multipliers[
        None, :, None
    ]
    future_weight = (
        multiplier
        * math.sqrt(future_factor)
    )
    denominator = np.sqrt(
        1.0
        + multiplier
        * multiplier
        * future_factor
    )
    threshold = norm.isf(
        directional_alpha
    )[None, None, :]
    pilot_successes = pilot_successes[
        None, None, :
    ]
    pilot_totals = pilot_totals[
        None, None, :
    ]
    future_successes = future_successes[
        :, None, :
    ]
    future_totals = future_totals[
        None, None, :
    ]

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
        future_successes.shape[0],
        len(multipliers),
        pilot_successes.shape[2],
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
    left = minimum.copy()
    right = maximum.copy()
    for _ in range(BISECTION_ITERATIONS):
        middle = (left + right) / 2.0
        value = statistic(
            middle,
            lower_tail=False,
        )
        move_left = (
            lower_active & (value > 0.0)
        )
        left = np.where(
            move_left,
            middle,
            left,
        )
        right = np.where(
            lower_active & ~move_left,
            middle,
            right,
        )
    lower = np.where(
        lower_zero,
        0.0,
        np.where(
            lower_one,
            1.0,
            (left + right) / 2.0,
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
    left = minimum.copy()
    right = maximum.copy()
    for _ in range(BISECTION_ITERATIONS):
        middle = (left + right) / 2.0
        value = statistic(
            middle,
            lower_tail=True,
        )
        move_left = (
            upper_active & (value < 0.0)
        )
        left = np.where(
            move_left,
            middle,
            left,
        )
        right = np.where(
            upper_active & ~move_left,
            middle,
            right,
        )
    upper = np.where(
        upper_one,
        1.0,
        np.where(
            upper_zero,
            0.0,
            (left + right) / 2.0,
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
            axis=2,
        ),
        np.max(
            upper_residual,
            axis=2,
        ),
    )
    return lower, upper, residual


def posterior_scenarios(
    context: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    list[int],
]:
    pilot_successes = np.asarray(
        context["cells"]["successes"],
        dtype=np.float64,
    )
    pilot_totals = np.asarray(
        context["cells"]["totals"],
        dtype=np.float64,
    )
    by_factor = {
        factor_key(factor): []
        for factor in FUTURE_FACTORS
    }
    block_ids = []
    for block, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        for _ in range(
            REPLICATES_PER_SEED
        ):
            probability = rng.beta(
                pilot_successes + 0.5,
                pilot_totals
                - pilot_successes
                + 0.5,
            )
            for factor in FUTURE_FACTORS:
                key = factor_key(factor)
                totals = np.asarray(
                    context["allocations"][
                        key
                    ]["totals"],
                    dtype=np.int64,
                )
                by_factor[key].append(
                    rng.binomial(
                        totals,
                        probability,
                    ).astype(np.int64)
                )
            block_ids.append(block)
    return (
        {
            key: np.asarray(
                values,
                dtype=np.int64,
            )
            for key, values
            in by_factor.items()
        },
        block_ids,
    )


def evaluate_scenarios(
    context: dict[str, Any],
    successes: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    multipliers = np.asarray(
        [
            value
            for _, value in CANDIDATES
        ],
        dtype=np.float64,
    )
    scenario_count = next(
        iter(successes.values())
    ).shape[0]
    values = np.empty(
        (
            scenario_count,
            len(multipliers),
            len(context["conditions"]),
        ),
        dtype=np.float64,
    )
    residuals = np.zeros(
        (
            scenario_count,
            len(multipliers),
        ),
        dtype=np.float64,
    )
    condition_index = 0
    for confidence in CONFIDENCE_LEVELS:
        directional_alpha = (
            (1.0 - confidence)
            * context["alpha_weights"]
            / 2.0
        )
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
            lower, upper, residual = (
                batched_stouffer_intervals(
                    multipliers=multipliers,
                    pilot_successes=np.asarray(
                        context["cells"][
                            "successes"
                        ],
                        dtype=np.int64,
                    ),
                    pilot_totals=np.asarray(
                        context["cells"][
                            "totals"
                        ],
                        dtype=np.int64,
                    ),
                    future_successes=(
                        successes[key]
                    ),
                    future_totals=np.asarray(
                        context["allocations"][
                            key
                        ]["totals"],
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
            scheme_count = (
                scenario_count
                * len(multipliers)
            )
            lower = lower.reshape(
                (
                    scheme_count,
                    *context["point_shape"],
                )
            )
            upper = upper.reshape(
                (
                    scheme_count,
                    *context["point_shape"],
                )
            )
            global_p90 = np.zeros(
                scheme_count,
                dtype=np.float64,
            )
            global_maximum = np.zeros(
                scheme_count,
                dtype=np.float64,
            )
            for kernel_name in KERNELS:
                kernel_p90 = np.zeros(
                    scheme_count,
                    dtype=np.float64,
                )
                kernel_maximum = np.zeros(
                    scheme_count,
                    dtype=np.float64,
                )
                for row in context["pilot"][
                    "cache"
                ][kernel_name]:
                    regrets = robust_regret_many(
                        prevalence=context[
                            "prevalence"
                        ],
                        selected=row["selected"],
                        coefficients=row[
                            "coefficients"
                        ],
                        lowers=lower,
                        uppers=upper,
                    )
                    ppm = (
                        regrets * 1_000_000.0
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
                    kernel_p90 = np.maximum(
                        kernel_p90,
                        p90,
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
                :,
                condition_index,
            ] = global_p90.reshape(
                scenario_count,
                len(multipliers),
            )
            condition_index += 1
            values[
                :,
                :,
                condition_index,
            ] = global_maximum.reshape(
                scenario_count,
                len(multipliers),
            )
            condition_index += 1
    return values, residuals


def clopper_pearson(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(
            beta.ppf(
                alpha / 2.0,
                successes,
                total - successes + 1,
            )
        )
    )
    upper = (
        1.0
        if successes == total
        else float(
            beta.ppf(
                1.0 - alpha / 2.0,
                successes + 1,
                total - successes,
            )
        )
    )
    return {
        "fraction": successes / total,
        "lower": lower,
        "upper": upper,
    }


def maximum_numeric_difference(
    left: Any,
    right: Any,
) -> float:
    if isinstance(left, dict):
        return max(
            maximum_numeric_difference(
                left[key],
                right[key],
            )
            for key in left
        )
    if isinstance(left, list):
        return max(
            maximum_numeric_difference(
                first,
                second,
            )
            for first, second
            in zip(left, right)
        )
    if isinstance(left, (int, float)):
        return abs(
            float(left) - float(right)
        )
    return 0.0 if left == right else float(
        "inf"
    )


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    context = prepare_context(args.root)
    e107 = load_e107(args.root)
    successes, block_ids = (
        posterior_scenarios(context)
    )
    values, residuals = evaluate_scenarios(
        context,
        successes,
    )
    sqrt2 = values[:, 0, :]
    continuous = values[:, 1, :]
    delta = continuous - sqrt2
    sqrt2_lower = delta > TIE_TOLERANCE
    continuous_lower = (
        delta < -TIE_TOLERANCE
    )
    ties = np.abs(delta) <= TIE_TOLERANCE

    optimum = np.minimum(
        sqrt2,
        continuous,
    )
    ratios = values / optimum[:, None, :]
    scenario_maximum = np.max(
        ratios,
        axis=2,
    )
    scenario_delta = (
        scenario_maximum[:, 1]
        - scenario_maximum[:, 0]
    )
    sqrt2_scenario_wins = (
        scenario_delta > TIE_TOLERANCE
    )
    continuous_scenario_wins = (
        scenario_delta < -TIE_TOLERANCE
    )
    scenario_ties = (
        np.abs(scenario_delta)
        <= TIE_TOLERANCE
    )

    block_rows = []
    block_ids_array = np.asarray(
        block_ids,
        dtype=np.int64,
    )
    for block, seed in enumerate(SEEDS):
        mask = block_ids_array == block
        non_ties = int(
            np.count_nonzero(
                ~scenario_ties[mask]
            )
        )
        wins = int(
            np.count_nonzero(
                sqrt2_scenario_wins[
                    mask
                ]
            )
        )
        block_rows.append(
            {
                "seed": seed,
                "scenario_count": int(
                    np.count_nonzero(mask)
                ),
                "sqrt2_scenario_wins": (
                    wins
                ),
                "continuous_scenario_wins": int(
                    np.count_nonzero(
                        continuous_scenario_wins[
                            mask
                        ]
                    )
                ),
                "ties": int(
                    np.count_nonzero(
                        scenario_ties[mask]
                    )
                ),
                "sqrt2_win_rate": (
                    clopper_pearson(
                        wins,
                        non_ties,
                    )
                ),
                "sqrt2_lower_condition_count": int(
                    np.count_nonzero(
                        sqrt2_lower[mask]
                    )
                ),
                "continuous_lower_condition_count": int(
                    np.count_nonzero(
                        continuous_lower[
                            mask
                        ]
                    )
                ),
            }
        )

    convergence = []
    for scenario_count in (
        32,
        64,
        96,
        128,
    ):
        prefix_ties = scenario_ties[
            :scenario_count
        ]
        non_ties = int(
            np.count_nonzero(
                ~prefix_ties
            )
        )
        wins = int(
            np.count_nonzero(
                sqrt2_scenario_wins[
                    :scenario_count
                ]
            )
        )
        convergence.append(
            {
                "scenario_count": (
                    scenario_count
                ),
                "sqrt2_scenario_wins": (
                    wins
                ),
                "continuous_scenario_wins": int(
                    np.count_nonzero(
                        continuous_scenario_wins[
                            :scenario_count
                        ]
                    )
                ),
                "ties": int(
                    np.count_nonzero(
                        prefix_ties
                    )
                ),
                "sqrt2_win_rate": (
                    clopper_pearson(
                        wins,
                        non_ties,
                    )
                ),
            }
        )

    first_block_pairwise = pairwise_summary(
        continuous[:32].reshape(
            -1
        ).tolist(),
        sqrt2[:32].reshape(
            -1
        ).tolist(),
    )
    expected_first_block = e107[
        "headline"
    ]["family_summaries"][
        "posterior_predictive"
    ]["continuous_vs_sqrt2"]
    first_block_error = (
        maximum_numeric_difference(
            first_block_pairwise,
            expected_first_block,
        )
    )

    first_successes = {
        key: value[0]
        for key, value in successes.items()
    }
    single_values, _ = Evaluator(
        scenario_context(
            context,
            first_successes,
        )
    ).evaluate(
        [
            value
            for _, value in CANDIDATES
        ]
    )
    first_scenario_error = float(
        np.max(
            np.abs(
                single_values
                - values[0]
            )
        )
    )
    total_non_ties = int(
        np.count_nonzero(
            ~scenario_ties
        )
    )
    sqrt2_scenario_win_count = int(
        np.count_nonzero(
            sqrt2_scenario_wins
        )
    )
    all_successes_valid = all(
        np.all(counts >= 0)
        and np.all(
            counts
            <= context["allocations"][
                key
            ]["totals"][None, :]
        )
        for key, counts
        in successes.items()
    )
    maximum_residual = float(
        np.max(residuals)
    )
    if not (
        first_block_error <= 2e-12
        and first_scenario_error
        <= 2e-9
        and maximum_residual <= 0.01
        and all_successes_valid
        and len(block_ids)
        == len(SEEDS)
        * REPLICATES_PER_SEED
    ):
        raise RuntimeError(
            "E108 validation failed"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "stage_weight_predictive_convergence"
        ),
        "source": {
            "pilot": (
                "pacing_coalescing_order_full.json"
            ),
            "outcome_sensitivity": (
                "pacing_coalescing_order_"
                "stage_weight_outcome_sensitivity_full.json"
            ),
        },
        "model": (
            "Compare sqrt(2) and 1.428317 "
            "across four independent fixed-seed "
            "blocks of 32 Jeffreys posterior-"
            "predictive future datasets. Batched "
            "interval inversion and regret "
            "propagation quantify Monte Carlo "
            "convergence of E107's recommendation."
        ),
        "candidates": [
            {
                "name": name,
                "multiplier": value,
            }
            for name, value in CANDIDATES
        ],
        "scenario_design": {
            "seeds": list(SEEDS),
            "replicates_per_seed": (
                REPLICATES_PER_SEED
            ),
            "scenario_count": len(
                block_ids
            ),
            "posterior": (
                "Independent Jeffreys "
                "Beta(successes+0.5, "
                "failures+0.5) per E83 risk cell"
            ),
        },
        "headline": {
            "scenario_count": len(
                block_ids
            ),
            "conditions_per_scenario": (
                values.shape[2]
            ),
            "pairwise_condition_count": int(
                values.shape[0]
                * values.shape[2]
            ),
            "sqrt2_lower_condition_count": int(
                np.count_nonzero(
                    sqrt2_lower
                )
            ),
            "continuous_lower_condition_count": int(
                np.count_nonzero(
                    continuous_lower
                )
            ),
            "condition_tie_count": int(
                np.count_nonzero(ties)
            ),
            "scenario_robust_wins": {
                "sqrt2": (
                    sqrt2_scenario_win_count
                ),
                "continuous": int(
                    np.count_nonzero(
                        continuous_scenario_wins
                    )
                ),
                "ties": int(
                    np.count_nonzero(
                        scenario_ties
                    )
                ),
            },
            "sqrt2_scenario_win_rate": (
                clopper_pearson(
                    sqrt2_scenario_win_count,
                    total_non_ties,
                )
            ),
            "continuous_vs_sqrt2": (
                pairwise_summary(
                    continuous.reshape(
                        -1
                    ).tolist(),
                    sqrt2.reshape(
                        -1
                    ).tolist(),
                )
            ),
            "candidate_relative_regret": {
                name: {
                    "geometric_mean": float(
                        math.exp(
                            np.mean(
                                np.log(
                                    ratios[
                                        :,
                                        index,
                                        :,
                                    ]
                                )
                            )
                        )
                    ),
                    "p90": float(
                        np.quantile(
                            ratios[
                                :,
                                index,
                                :,
                            ],
                            0.90,
                            method="higher",
                        )
                    ),
                    "p99": float(
                        np.quantile(
                            ratios[
                                :,
                                index,
                                :,
                            ],
                            0.99,
                            method="higher",
                        )
                    ),
                    "maximum": float(
                        np.max(
                            ratios[
                                :,
                                index,
                                :,
                            ]
                        )
                    ),
                }
                for index, (name, _)
                in enumerate(CANDIDATES)
            },
            "convergence": convergence,
            "seed_blocks": block_rows,
        },
        "validation": {
            "e107_first_seed_block_maximum_error": (
                first_block_error
            ),
            "single_scenario_evaluator_maximum_error": (
                first_scenario_error
            ),
            "maximum_interval_root_residual": (
                maximum_residual
            ),
            "all_success_counts_valid": (
                all_successes_valid
            ),
        },
        "limitations": [
            (
                "The 128 scenarios reduce but do "
                "not eliminate Monte Carlo error. "
                "Confidence intervals summarize "
                "simulation frequency, not policy "
                "coverage."
            ),
            (
                "The Jeffreys posterior treats "
                "E83 risk cells independently and "
                "inherits earlier binomial and "
                "rectangular limitations."
            ),
            (
                "Only sqrt(2) and 1.428317 are "
                "compared. The study confirms the "
                "specific E107 recommendation; it "
                "does not reopen the five-candidate "
                "selection problem."
            ),
            (
                "Both candidates remain post-hoc "
                "for E83 and apply only to a future "
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
