from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, binom, norm

from analyze_pacing_coalescing_order_path_allocation import (
    load_json,
)
from analyze_pacing_coalescing_order_pilot_holdout import (
    FUTURE_FACTORS,
    factor_key,
)
from analyze_pacing_coalescing_order_stage_weight_continuous import (
    prepare_context,
)
from analyze_pacing_coalescing_order_stage_weight_outcome_sensitivity import (
    pairwise_summary,
)
from analyze_pacing_coalescing_order_stage_weight_predictive_convergence import (
    CANDIDATES,
    clopper_pearson,
    evaluate_scenarios,
    posterior_scenarios,
)


SCENARIOS_PER_PROFILE = 32
HALF_SCENARIOS = SCENARIOS_PER_PROFILE // 2
LATENT_SEED = 10920261114
COUNT_SEED_BASE = 10920261200
TIE_TOLERANCE = 1e-10
PROFILES = (
    (
        "global_0p50",
        {
            "global": 0.50,
            "cost": 0.0,
            "order": 0.0,
            "metric": 0.0,
        },
    ),
    (
        "cost_0p50",
        {
            "global": 0.0,
            "cost": 0.50,
            "order": 0.0,
            "metric": 0.0,
        },
    ),
    (
        "order_0p50",
        {
            "global": 0.0,
            "cost": 0.0,
            "order": 0.50,
            "metric": 0.0,
        },
    ),
    (
        "metric_0p50",
        {
            "global": 0.0,
            "cost": 0.0,
            "order": 0.0,
            "metric": 0.50,
        },
    ),
    (
        "mixed_0p75",
        {
            "global": 0.10,
            "cost": 0.25,
            "order": 0.20,
            "metric": 0.20,
        },
    ),
)


def load_e108(root: Path) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_stage_weight_predictive_convergence"
        / "pacing_coalescing_order_stage_weight_predictive_convergence_full.json"
    )


def group_indices(
    labels: list[dict[str, Any]],
    field: str,
) -> tuple[np.ndarray, list[str]]:
    values = sorted(
        {
            str(label[field])
            for label in labels
        }
    )
    index = {
        value: position
        for position, value
        in enumerate(values)
    }
    return (
        np.asarray(
            [
                index[str(label[field])]
                for label in labels
            ],
            dtype=np.int64,
        ),
        values,
    )


def latent_factors(
    context: dict[str, Any],
) -> dict[str, np.ndarray]:
    labels = context["cells"]["labels"]
    cost_index, cost_values = (
        group_indices(labels, "cost")
    )
    order_index, order_values = (
        group_indices(labels, "order")
    )
    metric_index, metric_values = (
        group_indices(labels, "metric")
    )
    rng = np.random.default_rng(
        LATENT_SEED
    )
    half = HALF_SCENARIOS
    factors = {
        "global": rng.normal(
            size=(half, 1)
        ),
        "cost": rng.normal(
            size=(
                half,
                len(cost_values),
            )
        )[:, cost_index],
        "order": rng.normal(
            size=(
                half,
                len(order_values),
            )
        )[:, order_index],
        "metric": rng.normal(
            size=(
                half,
                len(metric_values),
            )
        )[:, metric_index],
        "residual": rng.normal(
            size=(
                half,
                len(labels),
            )
        ),
    }
    return {
        name: np.concatenate(
            (
                values,
                -values,
            ),
            axis=0,
        )
        for name, values
        in factors.items()
    }


def correlated_profiles(
    context: dict[str, Any],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    pilot_successes = np.asarray(
        context["cells"]["successes"],
        dtype=np.float64,
    )
    pilot_totals = np.asarray(
        context["cells"]["totals"],
        dtype=np.float64,
    )
    alpha = pilot_successes + 0.5
    beta_parameter = (
        pilot_totals
        - pilot_successes
        + 0.5
    )
    factors = latent_factors(context)
    results = {}
    profile_validation = {}
    for profile_index, (
        name,
        weights,
    ) in enumerate(PROFILES):
        shared_variance = sum(
            weights.values()
        )
        if shared_variance > 1.0:
            raise RuntimeError(
                f"Invalid profile {name}"
            )
        z = (
            math.sqrt(
                weights["global"]
            )
            * factors["global"]
            + math.sqrt(
                weights["cost"]
            )
            * factors["cost"]
            + math.sqrt(
                weights["order"]
            )
            * factors["order"]
            + math.sqrt(
                weights["metric"]
            )
            * factors["metric"]
            + math.sqrt(
                1.0
                - shared_variance
            )
            * factors["residual"]
        )
        uniform = np.clip(
            norm.cdf(z),
            1e-12,
            1.0 - 1e-12,
        )
        probability = beta.ppf(
            uniform,
            alpha[None, :],
            beta_parameter[None, :],
        )
        roundtrip_error = float(
            np.max(
                np.abs(
                    beta.cdf(
                        probability,
                        alpha[None, :],
                        beta_parameter[
                            None, :
                        ],
                    )
                    - uniform
                )
            )
        )
        rng = np.random.default_rng(
            COUNT_SEED_BASE
            + profile_index
        )
        results[name] = {}
        for factor in FUTURE_FACTORS:
            key = factor_key(factor)
            totals = np.asarray(
                context["allocations"][
                    key
                ]["totals"],
                dtype=np.int64,
            )
            results[name][key] = (
                rng.binomial(
                    totals[None, :],
                    probability,
                ).astype(np.int64)
            )
        profile_validation[name] = {
            "shared_variance": (
                shared_variance
            ),
            "residual_variance": (
                1.0
                - shared_variance
            ),
            "beta_cdf_roundtrip_maximum_error": (
                roundtrip_error
            ),
        }
    return results, profile_validation


def candidate_relative_summary(
    ratios: np.ndarray,
) -> dict[str, float]:
    return {
        "geometric_mean": float(
            math.exp(
                np.mean(
                    np.log(ratios)
                )
            )
        ),
        "p90": float(
            np.quantile(
                ratios,
                0.90,
                method="higher",
            )
        ),
        "p99": float(
            np.quantile(
                ratios,
                0.99,
                method="higher",
            )
        ),
        "maximum": float(
            np.max(ratios)
        ),
    }


def profile_summary(
    values: np.ndarray,
) -> dict[str, Any]:
    sqrt2 = values[:, 0, :]
    continuous = values[:, 1, :]
    delta = continuous - sqrt2
    optimum = np.minimum(
        sqrt2,
        continuous,
    )
    ratios = values / optimum[
        :, None, :
    ]
    scenario_maximum = np.max(
        ratios,
        axis=2,
    )
    scenario_delta = (
        scenario_maximum[:, 1]
        - scenario_maximum[:, 0]
    )
    sqrt2_wins = int(
        np.count_nonzero(
            scenario_delta
            > TIE_TOLERANCE
        )
    )
    continuous_wins = int(
        np.count_nonzero(
            scenario_delta
            < -TIE_TOLERANCE
        )
    )
    ties = int(
        np.count_nonzero(
            np.abs(scenario_delta)
            <= TIE_TOLERANCE
        )
    )
    non_ties = sqrt2_wins + continuous_wins
    return {
        "scenario_robust_wins": {
            "sqrt2": sqrt2_wins,
            "continuous": (
                continuous_wins
            ),
            "ties": ties,
        },
        "sqrt2_scenario_win_rate": (
            clopper_pearson(
                sqrt2_wins,
                non_ties,
            )
        ),
        "one_sided_p_value_vs_half": (
            float(
                binom.sf(
                    sqrt2_wins - 1,
                    non_ties,
                    0.5,
                )
            )
        ),
        "sqrt2_lower_condition_count": int(
            np.count_nonzero(
                delta > TIE_TOLERANCE
            )
        ),
        "continuous_lower_condition_count": int(
            np.count_nonzero(
                delta < -TIE_TOLERANCE
            )
        ),
        "condition_tie_count": int(
            np.count_nonzero(
                np.abs(delta)
                <= TIE_TOLERANCE
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
            name: (
                candidate_relative_summary(
                    ratios[:, index, :]
                )
            )
            for index, (name, _)
            in enumerate(CANDIDATES)
        },
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    context = prepare_context(args.root)
    e108 = load_e108(args.root)
    independent_successes, _ = (
        posterior_scenarios(context)
    )
    profile_successes, profile_validation = (
        correlated_profiles(context)
    )
    all_profiles = {
        "independent": {
            key: values[
                :SCENARIOS_PER_PROFILE
            ]
            for key, values
            in independent_successes.items()
        },
        **profile_successes,
    }
    profile_names = list(
        all_profiles
    )
    combined_successes = {
        key: np.concatenate(
            [
                all_profiles[name][key]
                for name in profile_names
            ],
            axis=0,
        )
        for key in all_profiles[
            profile_names[0]
        ]
    }
    values, residuals = evaluate_scenarios(
        context,
        combined_successes,
    )
    profile_summaries = {}
    for profile_index, name in enumerate(
        profile_names
    ):
        start = (
            profile_index
            * SCENARIOS_PER_PROFILE
        )
        end = (
            start
            + SCENARIOS_PER_PROFILE
        )
        profile_summaries[name] = (
            profile_summary(
                values[start:end]
            )
        )
    correlated_values = values[
        SCENARIOS_PER_PROFILE:
    ]
    overall_correlated = (
        profile_summary(
            correlated_values
        )
    )
    overall_all = profile_summary(
        values
    )

    independent_expected = e108[
        "headline"
    ]["seed_blocks"][0]
    independent_actual = (
        profile_summaries[
            "independent"
        ]
    )
    independent_error = max(
        abs(
            independent_actual[
                "scenario_robust_wins"
            ]["sqrt2"]
            - independent_expected[
                "sqrt2_scenario_wins"
            ]
        ),
        abs(
            independent_actual[
                "scenario_robust_wins"
            ]["continuous"]
            - independent_expected[
                "continuous_scenario_wins"
            ]
        ),
        abs(
            independent_actual[
                "sqrt2_lower_condition_count"
            ]
            - independent_expected[
                "sqrt2_lower_condition_count"
            ]
        ),
        abs(
            independent_actual[
                "continuous_lower_condition_count"
            ]
            - independent_expected[
                "continuous_lower_condition_count"
            ]
        ),
    )
    all_profiles_favor_sqrt2 = all(
        summary[
            "scenario_robust_wins"
        ]["sqrt2"]
        > summary[
            "scenario_robust_wins"
        ]["continuous"]
        for summary in (
            profile_summaries.values()
        )
    )
    worst_profile = min(
        profile_summaries,
        key=lambda name: (
            profile_summaries[name][
                "sqrt2_scenario_win_rate"
            ]["fraction"],
            name,
        ),
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
        in combined_successes.items()
    )
    maximum_residual = float(
        np.max(residuals)
    )
    maximum_roundtrip_error = max(
        item[
            "beta_cdf_roundtrip_maximum_error"
        ]
        for item
        in profile_validation.values()
    )
    if not (
        independent_error == 0
        and all_successes_valid
        and maximum_residual <= 0.01
        and maximum_roundtrip_error
        <= 1e-9
    ):
        raise RuntimeError(
            "E109 validation failed"
        )

    result = {
        "suite": (
            "pacing_coalescing_order_"
            "stage_weight_dependence_sensitivity"
        ),
        "source": {
            "pilot": (
                "pacing_coalescing_order_full.json"
            ),
            "predictive_convergence": (
                "pacing_coalescing_order_"
                "stage_weight_predictive_convergence_full.json"
            ),
        },
        "model": (
            "Compare sqrt(2) and 1.428317 "
            "under the E108 independent Jeffreys "
            "posterior and five Gaussian-copula "
            "dependence profiles with global, "
            "cost, order, metric, or mixed latent "
            "factors. Antithetic latent draws "
            "reduce marginal Monte Carlo imbalance."
        ),
        "candidates": [
            {
                "name": name,
                "multiplier": value,
            }
            for name, value in CANDIDATES
        ],
        "scenario_design": {
            "scenarios_per_profile": (
                SCENARIOS_PER_PROFILE
            ),
            "latent_seed": LATENT_SEED,
            "count_seed_base": (
                COUNT_SEED_BASE
            ),
            "profiles": {
                "independent": {
                    "source": (
                        "E108 first seed block"
                    ),
                },
                **{
                    name: {
                        **weights,
                        "residual": (
                            1.0
                            - sum(
                                weights.values()
                            )
                        ),
                    }
                    for name, weights
                    in PROFILES
                },
            },
        },
        "headline": {
            "profile_count": len(
                profile_names
            ),
            "scenario_count": int(
                values.shape[0]
            ),
            "conditions_per_scenario": int(
                values.shape[2]
            ),
            "all_profiles_favor_sqrt2": (
                all_profiles_favor_sqrt2
            ),
            "worst_profile_for_sqrt2": (
                worst_profile
            ),
            "profiles": profile_summaries,
            "correlated_profiles_combined": (
                overall_correlated
            ),
            "all_profiles_combined": (
                overall_all
            ),
        },
        "validation": {
            "e108_independent_block_maximum_count_error": (
                independent_error
            ),
            "maximum_interval_root_residual": (
                maximum_residual
            ),
            "maximum_beta_cdf_roundtrip_error": (
                maximum_roundtrip_error
            ),
            "profile_generation": (
                profile_validation
            ),
            "all_success_counts_valid": (
                all_successes_valid
            ),
        },
        "limitations": [
            (
                "Gaussian-copula profiles are "
                "structured stress tests, not an "
                "estimated dependence model for "
                "production traffic."
            ),
            (
                "Positive latent correlations are "
                "tested; adversarial negative or "
                "non-Gaussian dependence remains "
                "outside scope."
            ),
            (
                "Exact intervals describe "
                "simulation win frequency, not "
                "policy confidence-set coverage."
            ),
            (
                "Only sqrt(2) and 1.428317 are "
                "compared, and both remain post-hoc "
                "for E83."
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
