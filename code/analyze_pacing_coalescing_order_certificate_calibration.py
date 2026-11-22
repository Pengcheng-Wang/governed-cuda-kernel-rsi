from __future__ import annotations

import argparse
import hashlib
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
    load_e83,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
    prevalence_profiles,
    profile_metrics,
    winner,
)
from analyze_pacing_coalescing_order_prevalence_certification import (
    PREVALENCE_DENOMINATOR,
    TOLERANCE,
    pairwise_boundary_distance_l1,
    quantile,
    required_samples,
)


SIMULATION_ACCURACIES = (
    0.0,
    0.5,
    0.9,
    1.0,
)
REGRET_TOLERANCES_PPM = (
    0,
    1,
    10,
    50,
)
PROFILE_SAMPLE_QUANTILES = (
    0.10,
    0.50,
    0.90,
)
SAMPLE_FRACTIONS = (
    0.001,
    0.003,
    0.01,
    0.03,
    0.10,
    0.30,
    1.0,
)
CERTIFICATION_CONFIDENCE = 0.95
MONTE_CARLO_CONFIDENCE = 0.95
DEFAULT_REPLICATES = 20_000


def stable_seed(
    *parts: object,
) -> int:
    payload = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")
    digest = hashlib.sha256(
        payload
    ).digest()
    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    )


def binomial_upper_bound(
    *,
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    if successes >= trials:
        return 1.0
    return float(
        beta.ppf(
            confidence,
            successes + 1,
            trials - successes,
        )
    )


def multinomial_l1_bound(
    *,
    samples: int,
    radius: float,
    category_count: int,
) -> float:
    return min(
        1.0,
        (
            2**category_count - 2
        )
        * math.exp(
            -samples
            * radius
            * radius
            / 2.0
        ),
    )


def policy_arrays(
    *,
    per_policy_costs: dict[
        str,
        list[dict[str, Any]],
    ],
    inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    names = sorted(
        per_policy_costs
    )
    return {
        "names": names,
        "regression": np.asarray(
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
        ),
        "delay": np.asarray(
            [
                [
                    row[
                        "positive_delay_burden_requests_per_comparison"
                    ]
                    for row
                    in per_policy_costs[
                        name
                    ]
                ]
                for name in names
            ],
            dtype=np.float64,
        ),
        "faster": np.asarray(
            [
                [
                    row[
                        "faster_fraction"
                    ]
                    for row
                    in per_policy_costs[
                        name
                    ]
                ]
                for name in names
            ],
            dtype=np.float64,
        ),
        "segments": np.asarray(
            [
                inventory[name][
                    "configuration_segment_count"
                ]
                for name in names
            ],
            dtype=np.int64,
        ),
        "name_rank": np.arange(
            len(names),
            dtype=np.int64,
        ),
    }


def vectorized_winners(
    *,
    prevalence: np.ndarray,
    arrays: dict[str, Any],
) -> np.ndarray:
    regression = (
        prevalence
        @ arrays["regression"].T
    )
    delay = (
        prevalence
        @ arrays["delay"].T
    )
    faster = (
        prevalence
        @ arrays["faster"].T
    )
    shape = regression.shape
    segments = np.broadcast_to(
        arrays["segments"],
        shape,
    )
    name_rank = np.broadcast_to(
        arrays["name_rank"],
        shape,
    )
    order = np.lexsort(
        (
            name_rank,
            -faster,
            segments,
            delay,
            regression,
        ),
        axis=1,
    )
    return order[:, 0]


def validate_vectorized_winners(
    *,
    arrays: dict[str, Any],
    per_policy_costs: dict[
        str,
        list[dict[str, Any]],
    ],
    inventory: dict[str, dict[str, Any]],
    seed: int,
) -> int:
    rng = np.random.default_rng(
        seed
    )
    prevalence = rng.dirichlet(
        np.ones(
            arrays["regression"].shape[1]
        ),
        size=100,
    )
    observed = vectorized_winners(
        prevalence=prevalence,
        arrays=arrays,
    )
    for index, weights in enumerate(
        prevalence
    ):
        metrics = {
            name: profile_metrics(
                per_cost=rows,
                weights=tuple(
                    float(value)
                    for value in weights
                ),
            )
            for name, rows
            in per_policy_costs.items()
        }
        expected = winner(
            metrics=metrics,
            inventory=inventory,
        )
        actual = arrays["names"][
            int(observed[index])
        ]
        if actual != expected:
            raise RuntimeError(
                "Vectorized winner did not "
                "match scalar winner"
            )
    return len(prevalence)


def profile_records(
    *,
    profiles: list[dict[str, Any]],
    per_policy_costs: dict[
        str,
        list[dict[str, Any]],
    ],
    inventory: dict[str, dict[str, Any]],
    arrays: dict[str, Any],
    tolerance_ppm: int,
) -> list[dict[str, Any]]:
    tolerance = (
        tolerance_ppm
        / 1_000_000.0
    )
    name_to_index = {
        name: index
        for index, name
        in enumerate(
            arrays["names"]
        )
    }
    records = []
    for profile_index, profile in (
        enumerate(profiles)
    ):
        metrics = {
            name: profile_metrics(
                per_cost=rows,
                weights=profile[
                    "weights"
                ],
            )
            for name, rows
            in per_policy_costs.items()
        }
        selected = winner(
            metrics=metrics,
            inventory=inventory,
        )
        selected_index = (
            name_to_index[selected]
        )
        true_risks = (
            arrays["regression"]
            @ np.asarray(
                profile["weights"],
                dtype=np.float64,
            )
        )
        selected_risk = float(
            true_risks[
                selected_index
            ]
        )
        bad_policy_indices = []
        bad_distances = []
        for competitor_index, competitor in (
            enumerate(
                arrays["names"]
            )
        ):
            if competitor == selected:
                continue
            regret = (
                float(
                    true_risks[
                        competitor_index
                    ]
                )
                - selected_risk
            )
            if (
                regret
                <= tolerance
                + TOLERANCE
            ):
                continue
            bad_policy_indices.append(
                competitor_index
            )
            bad_distances.append(
                pairwise_boundary_distance_l1(
                    prevalence=profile[
                        "weights"
                    ],
                    differences=[
                        arrays[
                            "regression"
                        ][
                            competitor_index,
                            cost_index,
                        ]
                        - arrays[
                            "regression"
                        ][
                            selected_index,
                            cost_index,
                        ]
                        for cost_index
                        in range(
                            arrays[
                                "regression"
                            ].shape[1]
                        )
                    ],
                    include_ties=True,
                )
            )
        radius = min(
            bad_distances,
            default=math.inf,
        )
        samples = required_samples(
            radius=radius,
            confidence=(
                CERTIFICATION_CONFIDENCE
            ),
            category_count=len(
                profile["weights"]
            ),
        )
        if samples in (None, 0):
            continue
        records.append(
            {
                "profile_index": (
                    profile_index
                ),
                "weights": profile[
                    "weights"
                ],
                "true_winner": selected,
                "true_winner_index": (
                    selected_index
                ),
                "true_risks": true_risks,
                "bad_policy_indices": (
                    bad_policy_indices
                ),
                "bad_policy_count": len(
                    bad_policy_indices
                ),
                "radius_l1": radius,
                "bound_samples": samples,
            }
        )
    return records


def select_profile_quantiles(
    *,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda record: (
            record["bound_samples"],
            record["profile_index"],
        ),
    )
    result = []
    for probability in (
        PROFILE_SAMPLE_QUANTILES
    ):
        index = math.ceil(
            probability
            * len(ordered)
        ) - 1
        index = min(
            len(ordered) - 1,
            max(0, index),
        )
        selected = dict(
            ordered[index]
        )
        selected[
            "sample_requirement_quantile"
        ] = probability
        result.append(selected)
    return result


def simulate_case(
    *,
    kernel_name: str,
    accuracy: float,
    tolerance_ppm: int,
    record: dict[str, Any],
    arrays: dict[str, Any],
    replicates: int,
) -> dict[str, Any]:
    weights = np.asarray(
        record["weights"],
        dtype=np.float64,
    )
    true_risks = np.asarray(
        record["true_risks"],
        dtype=np.float64,
    )
    true_winner_risk = float(
        true_risks[
            record[
                "true_winner_index"
            ]
        ]
    )
    true_regrets = np.maximum(
        0.0,
        true_risks
        - true_winner_risk,
    )
    tolerance = (
        tolerance_ppm
        / 1_000_000.0
    )
    by_samples = {}
    requested_samples = []
    for fraction in SAMPLE_FRACTIONS:
        samples = max(
            1,
            math.ceil(
                record[
                    "bound_samples"
                ]
                * fraction
            ),
        )
        requested_samples.append(
            (
                fraction,
                samples,
            )
        )
        if samples in by_samples:
            by_samples[samples][
                "requested_fractions"
            ].append(fraction)
            continue
        rng = np.random.default_rng(
            stable_seed(
                "e92",
                kernel_name,
                f"{accuracy:.2f}",
                tolerance_ppm,
                record[
                    "profile_index"
                ],
                samples,
                replicates,
            )
        )
        counts = rng.multinomial(
            samples,
            weights,
            size=replicates,
        )
        empirical = (
            counts
            / samples
        )
        winners = vectorized_winners(
            prevalence=empirical,
            arrays=arrays,
        )
        failures = (
            true_regrets[
                winners
            ]
            > tolerance
            + TOLERANCE
        )
        l1_error = np.abs(
            empirical - weights
        ).sum(axis=1)
        radius_exceeded = (
            l1_error
            >= record["radius_l1"]
            - 1e-12
        )
        violations = int(
            np.count_nonzero(
                failures
                & ~radius_exceeded
            )
        )
        if violations:
            raise RuntimeError(
                "Observed a bad winner inside "
                "the certified L1 radius"
            )
        failure_count = int(
            np.count_nonzero(
                failures
            )
        )
        radius_count = int(
            np.count_nonzero(
                radius_exceeded
            )
        )
        by_samples[samples] = {
            "samples": samples,
            "actual_fraction_of_bound": (
                samples
                / record[
                    "bound_samples"
                ]
            ),
            "requested_fractions": [
                fraction
            ],
            "failure_count": (
                failure_count
            ),
            "failure_fraction": (
                failure_count
                / replicates
            ),
            "failure_fraction_upper_95": (
                binomial_upper_bound(
                    successes=(
                        failure_count
                    ),
                    trials=replicates,
                    confidence=(
                        MONTE_CARLO_CONFIDENCE
                    ),
                )
            ),
            "radius_exceedance_count": (
                radius_count
            ),
            "radius_exceedance_fraction": (
                radius_count
                / replicates
            ),
            "multinomial_l1_bound": (
                multinomial_l1_bound(
                    samples=samples,
                    radius=record[
                        "radius_l1"
                    ],
                    category_count=len(
                        weights
                    ),
                )
            ),
            "inside_radius_failure_count": (
                violations
            ),
        }
    simulations = [
        by_samples[samples]
        for samples in sorted(
            by_samples
        )
    ]
    certified = next(
        (
            row
            for row in simulations
            if row[
                "failure_fraction_upper_95"
            ]
            <= (
                1.0
                - CERTIFICATION_CONFIDENCE
            )
        ),
        None,
    )
    bound_row = by_samples[
        record["bound_samples"]
    ]
    if (
        bound_row[
            "multinomial_l1_bound"
        ]
        > (
            1.0
            - CERTIFICATION_CONFIDENCE
            + 1e-12
        )
    ):
        raise RuntimeError(
            "E91 bound samples did not "
            "invert the concentration bound"
        )
    return {
        "kernel": kernel_name,
        "accuracy": accuracy,
        "tolerance_ppm": (
            tolerance_ppm
        ),
        "sample_requirement_quantile": (
            record[
                "sample_requirement_quantile"
            ]
        ),
        "profile_index": record[
            "profile_index"
        ],
        "weights": list(
            record["weights"]
        ),
        "true_winner": record[
            "true_winner"
        ],
        "bad_policy_count": record[
            "bad_policy_count"
        ],
        "radius_l1": record[
            "radius_l1"
        ],
        "bound_samples": record[
            "bound_samples"
        ],
        "simulations": simulations,
        "bound_sample_result": (
            bound_row
        ),
        "first_empirically_certified": (
            None
            if certified is None
            else {
                "samples": certified[
                    "samples"
                ],
                "actual_fraction_of_bound": (
                    certified[
                        "actual_fraction_of_bound"
                    ]
                ),
                "bound_reduction_factor": (
                    record[
                        "bound_samples"
                    ]
                    / certified[
                        "samples"
                    ]
                ),
                "failure_fraction": (
                    certified[
                        "failure_fraction"
                    ]
                ),
                "failure_fraction_upper_95": (
                    certified[
                        "failure_fraction_upper_95"
                    ]
                ),
            }
        ),
    }


def aggregate_cases(
    *,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    bound_failures = [
        case[
            "bound_sample_result"
        ]["failure_fraction"]
        for case in cases
    ]
    bound_upper = [
        case[
            "bound_sample_result"
        ][
            "failure_fraction_upper_95"
        ]
        for case in cases
    ]
    bound_radius_exceedance = [
        case[
            "bound_sample_result"
        ][
            "radius_exceedance_fraction"
        ]
        for case in cases
    ]
    failure_to_radius = [
        (
            case[
                "bound_sample_result"
            ]["failure_fraction"]
            / case[
                "bound_sample_result"
            ][
                "radius_exceedance_fraction"
            ]
        )
        for case in cases
        if case[
            "bound_sample_result"
        ][
            "radius_exceedance_fraction"
        ]
        > 0.0
    ]
    fractions = [
        case[
            "first_empirically_certified"
        ][
            "actual_fraction_of_bound"
        ]
        for case in cases
        if case[
            "first_empirically_certified"
        ]
        is not None
    ]
    reductions = [
        case[
            "first_empirically_certified"
        ][
            "bound_reduction_factor"
        ]
        for case in cases
        if case[
            "first_empirically_certified"
        ]
        is not None
    ]
    return {
        "case_count": len(cases),
        "empirically_certified_case_count": (
            len(fractions)
        ),
        "empirically_certified_fraction": (
            len(fractions)
            / len(cases)
        ),
        "at_e91_bound": {
            "failure_fraction_p50": (
                quantile(
                    bound_failures,
                    0.50,
                )
            ),
            "failure_fraction_p90": (
                quantile(
                    bound_failures,
                    0.90,
                )
            ),
            "failure_fraction_maximum": (
                max(bound_failures)
            ),
            "failure_upper_95_maximum": (
                max(bound_upper)
            ),
            "zero_failure_case_count": sum(
                case[
                    "bound_sample_result"
                ]["failure_count"]
                == 0
                for case in cases
            ),
            "radius_exceedance_fraction_p50": (
                quantile(
                    bound_radius_exceedance,
                    0.50,
                )
            ),
            "radius_exceedance_fraction_p90": (
                quantile(
                    bound_radius_exceedance,
                    0.90,
                )
            ),
            "radius_exceedance_fraction_maximum": (
                max(
                    bound_radius_exceedance
                )
            ),
            "zero_radius_exceedance_case_count": (
                sum(
                    value == 0.0
                    for value
                    in bound_radius_exceedance
                )
            ),
            "failure_to_radius_exceedance_ratio": {
                "p50": quantile(
                    failure_to_radius,
                    0.50,
                ),
                "p90": quantile(
                    failure_to_radius,
                    0.90,
                ),
                "maximum": max(
                    failure_to_radius
                ),
            },
        },
        "tested_fraction_of_e91_bound": {
            "p10": quantile(
                fractions,
                0.10,
            ),
            "p50": quantile(
                fractions,
                0.50,
            ),
            "p90": quantile(
                fractions,
                0.90,
            ),
            "maximum": (
                max(fractions)
                if fractions
                else None
            ),
        },
        "tested_bound_reduction_factor": {
            "p10": quantile(
                reductions,
                0.10,
            ),
            "p50": quantile(
                reductions,
                0.50,
            ),
            "p90": quantile(
                reductions,
                0.90,
            ),
            "maximum": (
                max(reductions)
                if reductions
                else None
            ),
        },
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
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
    cases = []
    winner_validation_count = 0
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        for accuracy in (
            SIMULATION_ACCURACIES
        ):
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
            winner_validation_count += (
                validate_vectorized_winners(
                    arrays=arrays,
                    per_policy_costs=(
                        per_policy_costs
                    ),
                    inventory=inventory,
                    seed=stable_seed(
                        "winner-validation",
                        kernel_name,
                        accuracy,
                    ),
                )
            )
            for tolerance_ppm in (
                REGRET_TOLERANCES_PPM
            ):
                records = profile_records(
                    profiles=profiles,
                    per_policy_costs=(
                        per_policy_costs
                    ),
                    inventory=inventory,
                    arrays=arrays,
                    tolerance_ppm=(
                        tolerance_ppm
                    ),
                )
                selected = (
                    select_profile_quantiles(
                        records=records
                    )
                )
                for record in selected:
                    cases.append(
                        simulate_case(
                            kernel_name=(
                                kernel_name
                            ),
                            accuracy=accuracy,
                            tolerance_ppm=(
                                tolerance_ppm
                            ),
                            record=record,
                            arrays=arrays,
                            replicates=(
                                args.replicates
                            ),
                        )
                    )
    aggregate = aggregate_cases(
        cases=cases
    )
    by_tolerance = {}
    for tolerance_ppm in (
        REGRET_TOLERANCES_PPM
    ):
        key = str(
            tolerance_ppm
        )
        by_tolerance[key] = (
            aggregate_cases(
                cases=[
                    case
                    for case in cases
                    if case[
                        "tolerance_ppm"
                    ]
                    == tolerance_ppm
                ]
            )
        )
    by_difficulty = {}
    for probability in (
        PROFILE_SAMPLE_QUANTILES
    ):
        key = f"{probability:.2f}"
        by_difficulty[key] = (
            aggregate_cases(
                cases=[
                    case
                    for case in cases
                    if case[
                        "sample_requirement_quantile"
                    ]
                    == probability
                ]
            )
        )
    result = {
        "suite": (
            "pacing_coalescing_order_certificate_calibration"
        ),
        "model": (
            "Deterministically simulate finite "
            "multinomial prevalence samples and "
            "the full lexicographic policy "
            "selector at representative easy, "
            "median, and hard E91 profiles. "
            "Compare observed over-tolerance "
            "selection with the conservative "
            "distribution-free L1 certificate."
        ),
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            ),
            "regret_certification": (
                "results/raw/"
                "pacing_coalescing_order_regret_certification/"
                "pacing_coalescing_order_regret_certification_full.json"
            ),
        },
        "simulation": {
            "accuracies": list(
                SIMULATION_ACCURACIES
            ),
            "regret_tolerances_ppm": list(
                REGRET_TOLERANCES_PPM
            ),
            "profile_sample_quantiles": list(
                PROFILE_SAMPLE_QUANTILES
            ),
            "sample_fractions_of_e91_bound": list(
                SAMPLE_FRACTIONS
            ),
            "replicates_per_sample_point": (
                args.replicates
            ),
            "certification_confidence": (
                CERTIFICATION_CONFIDENCE
            ),
            "monte_carlo_upper_bound_confidence": (
                MONTE_CARLO_CONFIDENCE
            ),
            "random_seed": (
                "SHA256-derived per case and sample size"
            ),
        },
        "cases": cases,
        "headline": {
            "overall": aggregate,
            "by_tolerance_ppm": (
                by_tolerance
            ),
            "by_profile_sample_quantile": (
                by_difficulty
            ),
        },
        "validation": {
            "case_count": len(cases),
            "expected_case_count": (
                len(KERNELS)
                * len(
                    SIMULATION_ACCURACIES
                )
                * len(
                    REGRET_TOLERANCES_PPM
                )
                * len(
                    PROFILE_SAMPLE_QUANTILES
                )
            ),
            "vectorized_winner_matches": (
                winner_validation_count
            ),
            "inside_radius_failure_count": sum(
                row[
                    "inside_radius_failure_count"
                ]
                for case in cases
                for row in case[
                    "simulations"
                ]
            ),
            "all_bound_samples_valid": all(
                case[
                    "bound_sample_result"
                ][
                    "multinomial_l1_bound"
                ]
                <= (
                    1.0
                    - CERTIFICATION_CONFIDENCE
                    + 1e-12
                )
                for case in cases
            ),
            "all_cases_empirically_certified_by_bound": all(
                case[
                    "bound_sample_result"
                ][
                    "failure_fraction_upper_95"
                ]
                <= (
                    1.0
                    - CERTIFICATION_CONFIDENCE
                )
                for case in cases
            ),
        },
        "limitations": [
            "Monte Carlo calibration estimates the tested E83/E88 policy-selection model; it does not replace the distribution-free E91 guarantee.",
            "The 192 cases cover four representative estimator accuracies and three E91 sample-requirement quantiles, not every prevalence profile.",
            "The first empirically certified sample is selected from a coarse seven-point fraction grid and is not an exact minimum.",
            "Clopper-Pearson bounds cover Monte Carlo uncertainty only, not model misspecification or traffic nonstationarity.",
            "The audit reuses compact E83 path summaries and is not new GPU evidence.",
        ],
    }
    if (
        result["validation"][
            "case_count"
        ]
        != result["validation"][
            "expected_case_count"
        ]
    ):
        raise RuntimeError(
            "Unexpected calibration case count"
        )
    if (
        result["validation"][
            "inside_radius_failure_count"
        ]
        != 0
    ):
        raise RuntimeError(
            "Calibration violated E91 radius"
        )
    if not result["validation"][
        "all_bound_samples_valid"
    ]:
        raise RuntimeError(
            "An E91 bound sample was invalid"
        )
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
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
    )
    args = parser.parse_args()
    if args.replicates <= 0:
        raise ValueError(
            "replicates must be positive"
        )
    run(args)


if __name__ == "__main__":
    main()
