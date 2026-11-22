from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    policy_inventory,
)
from analyze_pacing_coalescing_order_certificate_calibration import (
    policy_arrays,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    load_e83,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
    prevalence_profiles,
    profile_metrics,
    winner,
)
from analyze_pacing_coalescing_order_prevalence_certification import (
    CONFIDENCE_LEVELS,
    COVERAGE_TARGETS,
    PREVALENCE_DENOMINATOR,
    REPRESENTATIVE_ACCURACIES,
    SAMPLE_BUDGETS,
    TOLERANCE,
    quantile,
)


REGRET_TOLERANCES_PPM = (
    0,
    1,
    5,
    10,
    25,
    50,
)


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


def load_e91(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_regret_certification"
        / "pacing_coalescing_order_regret_certification_full.json"
    )


def load_e92(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_certificate_calibration"
        / "pacing_coalescing_order_certificate_calibration_full.json"
    )


def tolerance_key(
    tolerance_ppm: int,
) -> str:
    return str(
        tolerance_ppm
    )


def tilted_mean(
    *,
    log_probability: np.ndarray,
    differences: np.ndarray,
    multiplier: np.ndarray,
) -> np.ndarray:
    log_weight = (
        log_probability
        - multiplier[:, None]
        * differences
    )
    normalized = np.exp(
        log_weight
        - logsumexp(
            log_weight,
            axis=1,
            keepdims=True,
        )
    )
    return np.sum(
        normalized
        * differences,
        axis=1,
    )


def chernoff_rates(
    *,
    prevalence: np.ndarray,
    differences: np.ndarray,
) -> np.ndarray:
    means = (
        prevalence
        @ differences.T
    )
    rates = np.zeros_like(
        means,
        dtype=np.float64,
    )
    supported = (
        prevalence > 0.0
    )
    minimum_supported = np.min(
        np.where(
            supported[:, None, :],
            differences[None, :, :],
            math.inf,
        ),
        axis=2,
    )
    positive = means > TOLERANCE
    impossible = (
        positive
        & (
            minimum_supported
            > TOLERANCE
        )
    )
    rates[impossible] = math.inf

    nonnegative = (
        positive
        & ~impossible
        & (
            minimum_supported
            >= -TOLERANCE
        )
    )
    if np.any(nonnegative):
        zero_mask = (
            np.abs(
                differences
            )
            <= TOLERANCE
        )
        zero_mass = (
            prevalence
            @ zero_mask.T
        )
        rows, columns = np.nonzero(
            nonnegative
        )
        mass = zero_mass[
            rows,
            columns,
        ]
        rates[
            rows,
            columns,
        ] = np.where(
            mass > 0.0,
            -np.log(mass),
            math.inf,
        )

    negative = (
        positive
        & (
            minimum_supported
            < -TOLERANCE
        )
    )
    rows, columns = np.nonzero(
        negative
    )
    if len(rows):
        probability = prevalence[
            rows
        ]
        pair_difference = differences[
            columns
        ]
        log_probability = np.where(
            probability > 0.0,
            np.log(
                np.where(
                    probability > 0.0,
                    probability,
                    1.0,
                )
            ),
            -math.inf,
        )
        lower = np.zeros(
            len(rows),
            dtype=np.float64,
        )
        upper = np.ones(
            len(rows),
            dtype=np.float64,
        )
        for _ in range(80):
            mean = tilted_mean(
                log_probability=(
                    log_probability
                ),
                differences=(
                    pair_difference
                ),
                multiplier=upper,
            )
            active = mean > 0.0
            if not np.any(active):
                break
            upper[active] *= 2.0
        else:
            raise RuntimeError(
                "Could not bracket Chernoff "
                "tilt"
            )
        for _ in range(70):
            middle = (
                lower + upper
            ) / 2.0
            mean = tilted_mean(
                log_probability=(
                    log_probability
                ),
                differences=(
                    pair_difference
                ),
                multiplier=middle,
            )
            active = mean > 0.0
            lower[active] = middle[
                active
            ]
            upper[~active] = middle[
                ~active
            ]
        multiplier = (
            lower + upper
        ) / 2.0
        log_mgf = logsumexp(
            log_probability
            - multiplier[:, None]
            * pair_difference,
            axis=1,
        )
        calculated = -log_mgf
        if np.any(
            calculated
            <= 0.0
        ):
            raise RuntimeError(
                "Chernoff rate is not "
                "positive"
            )
        rates[
            rows,
            columns,
        ] = calculated
    return rates


def log_union_bound(
    *,
    samples: np.ndarray,
    rates: np.ndarray,
    included: np.ndarray,
) -> np.ndarray:
    safe_rates = np.where(
        np.isfinite(rates),
        rates,
        0.0,
    )
    exponent = np.where(
        included,
        -samples[:, None]
        * safe_rates,
        -math.inf,
    )
    return logsumexp(
        exponent,
        axis=1,
    )


def required_union_samples(
    *,
    rates: np.ndarray,
    included: np.ndarray,
    confidence: float,
) -> np.ndarray:
    finite_included = (
        included
        & np.isfinite(rates)
    )
    invalid = (
        finite_included
        & (
            rates <= 0.0
        )
    )
    if np.any(invalid):
        raise RuntimeError(
            "Included Chernoff event has "
            "nonpositive rate"
        )
    has_event = np.any(
        finite_included,
        axis=1,
    )
    result = np.zeros(
        len(rates),
        dtype=np.int64,
    )
    if not np.any(has_event):
        return result
    alpha_log = math.log(
        1.0 - confidence
    )
    lower = np.zeros(
        len(rates),
        dtype=np.int64,
    )
    upper = np.ones(
        len(rates),
        dtype=np.int64,
    )
    active = has_event.copy()
    for _ in range(63):
        bound = log_union_bound(
            samples=upper,
            rates=rates,
            included=finite_included,
        )
        active = (
            has_event
            & (
                bound > alpha_log
            )
        )
        if not np.any(active):
            break
        if np.any(
            upper[active]
            > np.iinfo(
                np.int64
            ).max // 2
        ):
            raise RuntimeError(
                "Chernoff sample requirement "
                "overflow"
            )
        upper[active] *= 2
    else:
        raise RuntimeError(
            "Could not bracket Chernoff "
            "sample requirement"
        )
    while np.any(
        upper[has_event]
        - lower[has_event]
        > 1
    ):
        middle = (
            lower
            + (
                upper - lower
            ) // 2
        )
        bound = log_union_bound(
            samples=middle,
            rates=rates,
            included=finite_included,
        )
        too_small = (
            has_event
            & (
                bound > alpha_log
            )
        )
        lower[too_small] = middle[
            too_small
        ]
        sufficient = (
            has_event
            & ~too_small
            & (
                upper - lower
                > 1
            )
        )
        upper[sufficient] = middle[
            sufficient
        ]
    result[has_event] = upper[
        has_event
    ]
    final_bound = log_union_bound(
        samples=result,
        rates=rates,
        included=finite_included,
    )
    if np.any(
        has_event
        & (
            final_bound
            > alpha_log
            + 1e-12
        )
    ):
        raise RuntimeError(
            "Chernoff sample inversion "
            "failed"
        )
    return result


def sample_summary(
    *,
    minimum_rates: np.ndarray,
    bad_policy_counts: np.ndarray,
    samples_by_confidence: dict[
        str,
        np.ndarray,
    ],
) -> dict[str, Any]:
    profile_count = len(
        minimum_rates
    )
    positive_rates = [
        float(value)
        for value
        in minimum_rates
        if (
            value > 0.0
            and math.isfinite(
                float(value)
            )
        )
    ]
    result = {
        "profile_count": profile_count,
        "unconditional_count": int(
            np.count_nonzero(
                bad_policy_counts == 0
            )
        ),
        "unconditional_fraction": (
            float(
                np.count_nonzero(
                    bad_policy_counts
                    == 0
                )
            )
            / profile_count
        ),
        "impossible_bad_event_count": int(
            np.count_nonzero(
                (
                    bad_policy_counts > 0
                )
                & np.isinf(
                    minimum_rates
                )
            )
        ),
        "minimum_pairwise_rate_quantiles": {
            "p10": quantile(
                positive_rates,
                0.10,
            ),
            "p50": quantile(
                positive_rates,
                0.50,
            ),
            "p90": quantile(
                positive_rates,
                0.90,
            ),
            "minimum": (
                min(positive_rates)
                if positive_rates
                else None
            ),
            "maximum": (
                max(positive_rates)
                if positive_rates
                else None
            ),
        },
        "bad_policy_count": {
            "p50": quantile(
                bad_policy_counts.tolist(),
                0.50,
            ),
            "p90": quantile(
                bad_policy_counts.tolist(),
                0.90,
            ),
            "maximum": int(
                np.max(
                    bad_policy_counts
                )
            ),
            "mean": float(
                np.mean(
                    bad_policy_counts
                )
            ),
        },
        "confidence": {},
    }
    for confidence in CONFIDENCE_LEVELS:
        key = f"{confidence:.2f}"
        samples = (
            samples_by_confidence[
                key
            ]
        )
        positive_samples = [
            int(value)
            for value in samples
            if value > 0
        ]
        result["confidence"][
            key
        ] = {
            "sample_quantiles": {
                "p50": quantile(
                    positive_samples,
                    0.50,
                ),
                "p90": quantile(
                    positive_samples,
                    0.90,
                ),
                "p95": quantile(
                    positive_samples,
                    0.95,
                ),
                "p99": quantile(
                    positive_samples,
                    0.99,
                ),
                "maximum": (
                    max(positive_samples)
                    if positive_samples
                    else None
                ),
            },
            "budget_coverage": {
                str(budget): {
                    "count": int(
                        np.count_nonzero(
                            samples
                            <= budget
                        )
                    ),
                    "fraction": float(
                        np.count_nonzero(
                            samples
                            <= budget
                        )
                        / profile_count
                    ),
                }
                for budget
                in SAMPLE_BUDGETS
            },
        }
    return result


def compare_to_e91(
    *,
    chernoff: dict[str, Any],
    e91: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "confidence": {},
    }
    for confidence in CONFIDENCE_LEVELS:
        key = f"{confidence:.2f}"
        chernoff_confidence = (
            chernoff[
                "confidence"
            ][key]
        )
        e91_confidence = (
            e91[
                "confidence"
            ][key]
        )
        ratios = {}
        for quantile_name in (
            "p50",
            "p90",
            "p95",
            "p99",
            "maximum",
        ):
            baseline = e91_confidence[
                "finite_sample_quantiles"
            ][quantile_name]
            candidate = (
                chernoff_confidence[
                    "sample_quantiles"
                ][quantile_name]
            )
            ratios[quantile_name] = (
                None
                if (
                    baseline is None
                    or candidate in (
                        None,
                        0,
                    )
                )
                else (
                    baseline
                    / candidate
                )
            )
            if (
                baseline is not None
                and candidate is not None
                and candidate > baseline
            ):
                raise RuntimeError(
                    "Chernoff quantile is "
                    "worse than E91"
                )
        coverage_gain = {}
        for budget in SAMPLE_BUDGETS:
            budget_key = str(
                budget
            )
            candidate = (
                chernoff_confidence[
                    "budget_coverage"
                ][budget_key][
                    "fraction"
                ]
            )
            baseline = (
                e91_confidence[
                    "budget_coverage"
                ][budget_key][
                    "fraction"
                ]
            )
            if (
                candidate
                + 1e-15
                < baseline
            ):
                raise RuntimeError(
                    "Chernoff coverage is "
                    "worse than E91"
                )
            coverage_gain[
                budget_key
            ] = {
                "fraction": (
                    candidate
                    - baseline
                ),
                "count": round(
                    (
                        candidate
                        - baseline
                    )
                    * chernoff[
                        "profile_count"
                    ]
                ),
            }
        result["confidence"][key] = {
            "e91_to_chernoff_sample_ratio": (
                ratios
            ),
            "budget_coverage_gain": (
                coverage_gain
            ),
        }
    return result


def coverage_thresholds(
    *,
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for tolerance_ppm in (
        REGRET_TOLERANCES_PPM
    ):
        tolerance = tolerance_key(
            tolerance_ppm
        )
        result[tolerance] = {}
        for confidence in (
            CONFIDENCE_LEVELS
        ):
            confidence_key = (
                f"{confidence:.2f}"
            )
            result[tolerance][
                confidence_key
            ] = {}
            for budget in (
                SAMPLE_BUDGETS
            ):
                budget_key = str(
                    budget
                )
                result[tolerance][
                    confidence_key
                ][budget_key] = {}
                for target in (
                    COVERAGE_TARGETS
                ):
                    sustained = next(
                        (
                            row["accuracy"]
                            for index, row
                            in enumerate(grid)
                            if all(
                                later[
                                    "tolerances"
                                ][tolerance][
                                    "summary"
                                ]["confidence"][
                                    confidence_key
                                ][
                                    "budget_coverage"
                                ][budget_key][
                                    "fraction"
                                ]
                                >= target
                                for later
                                in grid[index:]
                            )
                        ),
                        None,
                    )
                    result[tolerance][
                        confidence_key
                    ][budget_key][
                        f"{target:.2f}"
                    ] = {
                        "sustained_accuracy": (
                            sustained
                        ),
                    }
    return result


def ratio_summary(
    values: list[float],
) -> dict[str, Any]:
    return {
        "p10": quantile(
            values,
            0.10,
        ),
        "p50": quantile(
            values,
            0.50,
        ),
        "p90": quantile(
            values,
            0.90,
        ),
        "maximum": max(
            values
        ),
    }


def calibration_comparison(
    *,
    root: Path,
    drift: dict[str, Any],
    costs: list[str],
    inventory: dict[
        str,
        dict[str, Any],
    ],
    e92: dict[str, Any],
) -> dict[str, Any]:
    cache = {}
    records = []
    for case in e92["cases"]:
        cache_key = (
            case["kernel"],
            case["accuracy"],
        )
        if cache_key not in cache:
            kernel = KERNELS[
                cache_key[0]
            ]
            cache[cache_key] = {
                name: per_cost_metrics(
                    drift=drift,
                    costs=costs,
                    mapping=policy[
                        "mapping"
                    ],
                    accuracy=(
                        cache_key[1]
                    ),
                    offset_weights=kernel[
                        "offset_weights"
                    ],
                )
                for name, policy
                in inventory.items()
            }
        per_policy_costs = cache[
            cache_key
        ]
        prevalence = np.asarray(
            case["weights"],
            dtype=np.float64,
        )
        selected = case[
            "true_winner"
        ]
        selected_risk = np.asarray(
            [
                row[
                    "regression_fraction"
                ]
                for row
                in per_policy_costs[
                    selected
                ]
            ],
            dtype=np.float64,
        )
        differences = []
        bad = []
        tolerance = (
            case[
                "tolerance_ppm"
            ]
            / 1_000_000.0
        )
        for competitor, rows in (
            per_policy_costs.items()
        ):
            if competitor == selected:
                continue
            difference = (
                np.asarray(
                    [
                        row[
                            "regression_fraction"
                        ]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                - selected_risk
            )
            differences.append(
                difference
            )
            bad.append(
                float(
                    prevalence
                    @ difference
                )
                > (
                    tolerance
                    + TOLERANCE
                )
            )
        rates = chernoff_rates(
            prevalence=prevalence[
                None,
                :,
            ],
            differences=np.asarray(
                differences
            ),
        )
        chernoff_samples = int(
            required_union_samples(
                rates=rates,
                included=np.asarray(
                    bad,
                    dtype=bool,
                )[None, :],
                confidence=0.95,
            )[0]
        )
        empirical = case[
            "first_empirically_certified"
        ]
        if empirical is None:
            raise RuntimeError(
                "E92 case was not "
                "empirically certified"
            )
        empirical_samples = int(
            empirical["samples"]
        )
        records.append(
            {
                "kernel": case[
                    "kernel"
                ],
                "accuracy": case[
                    "accuracy"
                ],
                "tolerance_ppm": (
                    case[
                        "tolerance_ppm"
                    ]
                ),
                "sample_requirement_quantile": (
                    case[
                        "sample_requirement_quantile"
                    ]
                ),
                "e91_samples": int(
                    case[
                        "bound_samples"
                    ]
                ),
                "chernoff_samples": (
                    chernoff_samples
                ),
                "e92_first_tested_samples": (
                    empirical_samples
                ),
            }
        )

    def summarize(
        selected_records: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        e91_to_chernoff = [
            record[
                "e91_samples"
            ]
            / max(
                1,
                record[
                    "chernoff_samples"
                ],
            )
            for record
            in selected_records
        ]
        chernoff_to_e92 = [
            record[
                "chernoff_samples"
            ]
            / max(
                1,
                record[
                    "e92_first_tested_samples"
                ],
            )
            for record
            in selected_records
        ]
        largest = max(
            selected_records,
            key=lambda record: (
                record[
                    "chernoff_samples"
                ]
                / max(
                    1,
                    record[
                        "e92_first_tested_samples"
                    ],
                )
            ),
        )
        return {
            "case_count": len(
                selected_records
            ),
            "e91_to_chernoff_ratio": (
                ratio_summary(
                    e91_to_chernoff
                )
            ),
            "chernoff_to_e92_first_tested_ratio": (
                ratio_summary(
                    chernoff_to_e92
                )
            ),
            "chernoff_no_more_than_e92_first_tested_count": (
                sum(
                    value <= 1.0
                    for value
                    in chernoff_to_e92
                )
            ),
            "largest_residual_case": {
                **largest,
                "chernoff_to_e92_first_tested_ratio": (
                    largest[
                        "chernoff_samples"
                    ]
                    / max(
                        1,
                        largest[
                            "e92_first_tested_samples"
                        ],
                    )
                ),
            },
        }

    return {
        "overall": summarize(
            records
        ),
        "by_tolerance_ppm": {
            str(tolerance_ppm): summarize(
                [
                    record
                    for record in records
                    if record[
                        "tolerance_ppm"
                    ]
                    == tolerance_ppm
                ]
            )
            for tolerance_ppm in (
                0,
                1,
                10,
                50,
            )
        },
        "by_profile_sample_quantile": {
            f"{probability:.2f}": summarize(
                [
                    record
                    for record in records
                    if record[
                        "sample_requirement_quantile"
                    ]
                    == probability
                ]
            )
            for probability in (
                0.10,
                0.50,
                0.90,
            )
        },
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
    e91 = load_e91(args.root)
    e92 = load_e92(args.root)
    drift = e83["headline"][
        "drift_by_marginal_fraction_and_order"
    ]
    costs = sorted(
        drift,
        key=float,
    )
    inventory = policy_inventory(
        root=args.root,
        costs=costs,
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
    accuracy_grid = [
        index / 100.0
        for index in range(101)
    ]
    kernels = {}
    total_pairwise_rates = 0
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        e89_grid = {
            row["accuracy"]: row
            for row in e89["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        e91_grid = {
            row["accuracy"]: row
            for row in e91["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        grid = []
        representatives = {}
        for accuracy in accuracy_grid:
            per_policy_costs = {
                name: per_cost_metrics(
                    drift=drift,
                    costs=costs,
                    mapping=policy[
                        "mapping"
                    ],
                    accuracy=accuracy,
                    offset_weights=kernel[
                        "offset_weights"
                    ],
                )
                for name, policy
                in inventory.items()
            }
            arrays = policy_arrays(
                per_policy_costs=(
                    per_policy_costs
                ),
                inventory=inventory,
            )
            name_to_index = {
                name: index
                for index, name
                in enumerate(
                    arrays["names"]
                )
            }
            selected = np.asarray(
                [
                    name_to_index[
                        winner(
                            metrics={
                                name: (
                                    profile_metrics(
                                        per_cost=rows,
                                        weights=profile[
                                            "weights"
                                        ],
                                    )
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
            winner_counts = {
                name: int(
                    np.count_nonzero(
                        selected == index
                    )
                )
                for index, name
                in enumerate(
                    arrays["names"]
                )
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
            true_risks = (
                prevalence
                @ arrays[
                    "regression"
                ].T
            )
            selected_risks = (
                true_risks[
                    np.arange(
                        len(profiles)
                    ),
                    selected,
                ]
            )
            regret = (
                true_risks
                - selected_risks[
                    :,
                    None,
                ]
            )
            all_rates = np.full(
                regret.shape,
                math.inf,
                dtype=np.float64,
            )
            for policy_index in range(
                len(arrays["names"])
            ):
                rows = np.flatnonzero(
                    selected
                    == policy_index
                )
                if not len(rows):
                    continue
                competitors = np.asarray(
                    [
                        index
                        for index in range(
                            len(
                                arrays[
                                    "names"
                                ]
                            )
                        )
                        if index
                        != policy_index
                    ],
                    dtype=np.int64,
                )
                differences = (
                    arrays["regression"][
                        competitors
                    ]
                    - arrays["regression"][
                        policy_index
                    ][None, :]
                )
                rates = chernoff_rates(
                    prevalence=prevalence[
                        rows
                    ],
                    differences=(
                        differences
                    ),
                )
                all_rates[
                    rows[:, None],
                    competitors[
                        None, :
                    ],
                ] = rates
                total_pairwise_rates += (
                    rates.size
                )
            tolerance_rows = {}
            representative_rows = {}
            for tolerance_ppm in (
                REGRET_TOLERANCES_PPM
            ):
                key = tolerance_key(
                    tolerance_ppm
                )
                tolerance = (
                    tolerance_ppm
                    / 1_000_000.0
                )
                bad = (
                    regret
                    > (
                        tolerance
                        + TOLERANCE
                    )
                )
                bad[
                    np.arange(
                        len(profiles)
                    ),
                    selected,
                ] = False
                bad_counts = np.sum(
                    bad,
                    axis=1,
                )
                minimum_rates = np.min(
                    np.where(
                        bad,
                        all_rates,
                        math.inf,
                    ),
                    axis=1,
                )
                samples_by_confidence = {
                    f"{confidence:.2f}": (
                        required_union_samples(
                            rates=all_rates,
                            included=bad,
                            confidence=(
                                confidence
                            ),
                        )
                    )
                    for confidence
                    in CONFIDENCE_LEVELS
                }
                summary = sample_summary(
                    minimum_rates=(
                        minimum_rates
                    ),
                    bad_policy_counts=(
                        bad_counts
                    ),
                    samples_by_confidence=(
                        samples_by_confidence
                    ),
                )
                baseline = (
                    e91_grid[
                        accuracy
                    ]["tolerances"][
                        key
                    ]
                )
                comparison = (
                    compare_to_e91(
                        chernoff=summary,
                        e91=baseline,
                    )
                )
                tolerance_rows[key] = {
                    "summary": summary,
                    "comparison_to_e91": (
                        comparison
                    ),
                }
                if accuracy in (
                    REPRESENTATIVE_ACCURACIES
                ):
                    representative_rows[
                        key
                    ] = {
                        "summary": (
                            summary
                        ),
                        "comparison_to_e91": (
                            comparison
                        ),
                    }
            row = {
                "accuracy": accuracy,
                "tolerances": (
                    tolerance_rows
                ),
            }
            grid.append(row)
            if accuracy in (
                REPRESENTATIVE_ACCURACIES
            ):
                representatives[
                    f"{accuracy:g}"
                ] = {
                    "accuracy": accuracy,
                    "tolerances": (
                        representative_rows
                    ),
                }
        kernels[kernel_name] = {
            "description": kernel[
                "description"
            ],
            "accuracy_grid": grid,
            "representative_accuracies": (
                representatives
            ),
            "coverage_thresholds": (
                coverage_thresholds(
                    grid=grid
                )
            ),
        }
    result = {
        "suite": (
            "pacing_coalescing_order_chernoff_certification"
        ),
        "model": (
            "For every true over-tolerance "
            "competitor, the empirical policy "
            "selector can choose it only if its "
            "sample mean regression difference "
            "from the true winner is nonpositive. "
            "The exact categorical Chernoff rate "
            "bounds that event; a union bound over "
            "only bad policies gives a finite-"
            "sample certificate."
        ),
        "bound": {
            "pairwise_event": (
                "P(mean_n[d_j(X)] <= 0) <= "
                "exp(-n * sup_lambda>=0 "
                "-log E[exp(-lambda d_j(X))])"
            ),
            "policy_union": (
                "sum over true-regret-above-"
                "tolerance competitors"
            ),
            "confidence_levels": list(
                CONFIDENCE_LEVELS
            ),
            "sample_budgets": list(
                SAMPLE_BUDGETS
            ),
        },
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
            "l1_regret_certification": (
                "results/raw/"
                "pacing_coalescing_order_regret_certification/"
                "pacing_coalescing_order_regret_certification_full.json"
            ),
            "empirical_calibration": (
                "results/raw/"
                "pacing_coalescing_order_certificate_calibration/"
                "pacing_coalescing_order_certificate_calibration_full.json"
            ),
        },
        "regret_tolerances": [
            {
                "key": tolerance_key(
                    tolerance_ppm
                ),
                "excess_regressions_per_million": (
                    tolerance_ppm
                ),
                "regression_fraction": (
                    tolerance_ppm
                    / 1_000_000.0
                ),
            }
            for tolerance_ppm
            in REGRET_TOLERANCES_PPM
        ],
        "costs": costs,
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
        "calibration_comparison": (
            calibration_comparison(
                root=args.root,
                drift=drift,
                costs=costs,
                inventory=inventory,
                e92=e92,
            )
        ),
        "headline": {
            kernel_name: {
                "coverage_thresholds": values[
                    "coverage_thresholds"
                ],
                "representative_accuracies": (
                    values[
                        "representative_accuracies"
                    ]
                ),
            }
            for kernel_name, values
            in kernels.items()
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
            "regret_tolerance_count": len(
                REGRET_TOLERANCES_PPM
            ),
            "pairwise_rate_count": (
                total_pairwise_rates
            ),
            "e89_winner_counts_reproduced": (
                True
            ),
            "chernoff_quantiles_no_worse_than_e91": (
                True
            ),
            "chernoff_coverage_no_worse_than_e91": (
                True
            ),
            "sample_bound_inversion_valid": (
                True
            ),
            "e92_calibration_case_count": (
                len(e92["cases"])
            ),
        },
        "limitations": [
            "The union bound is policy-specific and finite-sample valid, but competitors' empirical non-worse events overlap and can still make the result conservative.",
            "The certificate uses the compact E83 regression model and does not include uncertainty in those path-level regression estimates.",
            "Regret tolerances are absolute regression fractions rather than deployment utility or SLO cost.",
            "The 10% prevalence simplex and E88 estimator kernels are sensitivity models, not calibrated field distributions.",
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
