from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    policy_inventory,
)
from analyze_pacing_coalescing_order_chernoff_certification import (
    chernoff_rates,
    required_union_samples,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    load_e83,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
)
from analyze_pacing_coalescing_order_prevalence_certification import (
    TOLERANCE,
    quantile,
)


CONFIDENCE = 0.95


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def pairwise_tilt(
    *,
    prevalence: np.ndarray,
    differences: np.ndarray,
) -> tuple[float, float | None]:
    support = prevalence > 0.0
    probability = prevalence[
        support
    ]
    values = differences[
        support
    ]
    mean = float(
        probability @ values
    )
    if mean <= TOLERANCE:
        return 0.0, 0.0
    minimum = float(
        np.min(values)
    )
    if minimum > TOLERANCE:
        return math.inf, None
    if minimum >= -TOLERANCE:
        zero_mass = float(
            np.sum(
                probability[
                    np.abs(values)
                    <= TOLERANCE
                ]
            )
        )
        if zero_mass <= 0.0:
            return math.inf, None
        return -math.log(
            zero_mass
        ), None
    log_probability = np.log(
        probability
    )

    def tilted_mean(
        multiplier: float,
    ) -> float:
        log_weight = (
            log_probability
            - multiplier
            * values
        )
        weight = np.exp(
            log_weight
            - logsumexp(
                log_weight
            )
        )
        return float(
            weight @ values
        )

    lower = 0.0
    upper = 1.0
    for _ in range(80):
        if (
            tilted_mean(
                upper
            )
            <= 0.0
        ):
            break
        upper *= 2.0
    else:
        raise RuntimeError(
            "Could not bracket pairwise "
            "Chernoff tilt"
        )
    for _ in range(80):
        middle = (
            lower + upper
        ) / 2.0
        if (
            tilted_mean(
                middle
            )
            > 0.0
        ):
            lower = middle
        else:
            upper = middle
    multiplier = (
        lower + upper
    ) / 2.0
    log_mgf = float(
        logsumexp(
            log_probability
            - multiplier
            * values
        )
    )
    return -log_mgf, multiplier


def projected_gradient_violation(
    *,
    point: np.ndarray,
    gradient: np.ndarray,
) -> float:
    active = point > 1e-8
    values = []
    if np.any(active):
        values.append(
            float(
                np.max(
                    np.abs(
                        gradient[
                            active
                        ]
                    )
                )
            )
        )
    if np.any(~active):
        values.append(
            float(
                np.max(
                    np.maximum(
                        0.0,
                        -gradient[
                            ~active
                        ],
                    )
                )
            )
        )
    return max(
        values,
        default=0.0,
    )


def global_minimizer_rate(
    *,
    prevalence: np.ndarray,
    risk: np.ndarray,
    candidate: int,
    true_winner: int,
) -> dict[str, Any]:
    support = prevalence > 0.0
    probability = prevalence[
        support
    ]
    supported_risk = risk[
        :,
        support,
    ]
    others = [
        index
        for index in range(
            len(risk)
        )
        if index != candidate
    ]
    differences = (
        supported_risk[
            candidate
        ][None, :]
        - supported_risk[
            others
        ]
    ).T
    feasibility = linprog(
        np.zeros(
            len(probability)
        ),
        A_ub=differences.T,
        b_ub=np.zeros(
            len(others)
        ),
        A_eq=np.ones(
            (
                1,
                len(probability),
            )
        ),
        b_eq=[1.0],
        bounds=[
            (0.0, 1.0)
            for _ in probability
        ],
        method="highs",
    )
    true_difference = (
        risk[candidate]
        - risk[true_winner]
    )
    pairwise_rate, pairwise_multiplier = (
        pairwise_tilt(
            prevalence=prevalence,
            differences=(
                true_difference
            ),
        )
    )
    if not feasibility.success:
        return {
            "rate": math.inf,
            "pairwise_rate": (
                pairwise_rate
            ),
            "rate_gain": math.inf,
            "event_impossible": True,
            "optimizer_success": True,
            "projected_gradient_violation": (
                0.0
            ),
        }
    if math.isinf(pairwise_rate):
        return {
            "rate": math.inf,
            "pairwise_rate": (
                pairwise_rate
            ),
            "rate_gain": 1.0,
            "event_impossible": True,
            "optimizer_success": True,
            "projected_gradient_violation": (
                0.0
            ),
        }

    log_probability = np.log(
        probability
    )

    def objective(
        multiplier: np.ndarray,
    ) -> float:
        return float(
            logsumexp(
                log_probability
                - differences
                @ multiplier
            )
        )

    def gradient(
        multiplier: np.ndarray,
    ) -> np.ndarray:
        log_weight = (
            log_probability
            - differences
            @ multiplier
        )
        weight = np.exp(
            log_weight
            - logsumexp(
                log_weight
            )
        )
        return -(
            differences.T
            @ weight
        )

    seeds = [
        np.zeros(
            len(others),
            dtype=np.float64,
        )
    ]
    if pairwise_multiplier is not None:
        pairwise_seed = np.zeros(
            len(others),
            dtype=np.float64,
        )
        pairwise_seed[
            others.index(
                true_winner
            )
        ] = pairwise_multiplier
        seeds.append(
            pairwise_seed
        )
    best_rate = pairwise_rate
    best_point = seeds[-1]
    optimizer_success = False
    for seed in seeds:
        seed_rate = max(
            0.0,
            -objective(seed),
        )
        if seed_rate > best_rate:
            best_rate = seed_rate
            best_point = seed
        optimized = minimize(
            objective,
            seed,
            jac=gradient,
            bounds=[
                (0.0, None)
                for _ in seed
            ],
            method="L-BFGS-B",
            options={
                "ftol": 1e-15,
                "gtol": 1e-12,
                "maxiter": 5_000,
                "maxls": 100,
                "maxcor": 20,
            },
        )
        optimizer_success = (
            optimizer_success
            or bool(
                optimized.success
            )
        )
        optimized_rate = max(
            0.0,
            -objective(
                optimized.x
            ),
        )
        if (
            optimized_rate
            > best_rate
        ):
            best_rate = (
                optimized_rate
            )
            best_point = (
                optimized.x
            )
    if (
        best_rate
        + 1e-10
        < pairwise_rate
    ):
        raise RuntimeError(
            "Global Chernoff rate is "
            "below pairwise rate"
        )
    violation = (
        projected_gradient_violation(
            point=best_point,
            gradient=gradient(
                best_point
            ),
        )
    )
    return {
        "rate": best_rate,
        "pairwise_rate": (
            pairwise_rate
        ),
        "rate_gain": (
            None
            if pairwise_rate
            <= 0.0
            else (
                best_rate
                / pairwise_rate
            )
        ),
        "event_impossible": False,
        "optimizer_success": (
            optimizer_success
        ),
        "projected_gradient_violation": (
            violation
        ),
    }


def summarize_ratios(
    values: list[float],
) -> dict[str, float]:
    return {
        "p10": float(
            quantile(
                values,
                0.10,
            )
        ),
        "p50": float(
            quantile(
                values,
                0.50,
            )
        ),
        "p90": float(
            quantile(
                values,
                0.90,
            )
        ),
        "maximum": max(
            values
        ),
    }


def aggregate_cases(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    e93_to_global = [
        case["e93_samples"]
        / max(
            1,
            case[
                "global_samples"
            ],
        )
        for case in cases
    ]
    global_to_e92 = [
        case["global_samples"]
        / max(
            1,
            case[
                "e92_first_tested_samples"
            ],
        )
        for case in cases
    ]
    largest = max(
        cases,
        key=lambda case: (
            case["global_samples"]
            / max(
                1,
                case[
                    "e92_first_tested_samples"
                ],
            )
        ),
    )
    return {
        "case_count": len(cases),
        "e93_to_global_sample_ratio": (
            summarize_ratios(
                e93_to_global
            )
        ),
        "global_to_e92_first_tested_ratio": (
            summarize_ratios(
                global_to_e92
            )
        ),
        "global_no_more_than_e92_first_tested_count": (
            sum(
                value <= 1.0
                for value
                in global_to_e92
            )
        ),
        "global_strictly_improves_e93_count": (
            sum(
                case[
                    "global_samples"
                ]
                < case[
                    "e93_samples"
                ]
                for case in cases
            )
        ),
        "largest_residual_case": {
            "kernel": largest[
                "kernel"
            ],
            "accuracy": largest[
                "accuracy"
            ],
            "tolerance_ppm": (
                largest[
                    "tolerance_ppm"
                ]
            ),
            "sample_requirement_quantile": (
                largest[
                    "sample_requirement_quantile"
                ]
            ),
            "e91_samples": largest[
                "e91_samples"
            ],
            "e93_samples": largest[
                "e93_samples"
            ],
            "global_samples": largest[
                "global_samples"
            ],
            "e92_first_tested_samples": (
                largest[
                    "e92_first_tested_samples"
                ]
            ),
            "global_to_e92_first_tested_ratio": (
                largest[
                    "global_samples"
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


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
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
    names = sorted(
        inventory
    )
    cache = {}
    cases = []
    policy_event_count = 0
    impossible_event_count = 0
    optimizer_success_count = 0
    maximum_projected_violation = 0.0
    for source_case in e92[
        "cases"
    ]:
        cache_key = (
            source_case["kernel"],
            source_case["accuracy"],
        )
        if cache_key not in cache:
            kernel = KERNELS[
                cache_key[0]
            ]
            per_policy_costs = {
                name: per_cost_metrics(
                    drift=drift,
                    costs=costs,
                    mapping=inventory[
                        name
                    ]["mapping"],
                    accuracy=(
                        cache_key[1]
                    ),
                    offset_weights=kernel[
                        "offset_weights"
                    ],
                )
                for name in names
            }
            cache[cache_key] = (
                np.asarray(
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
            )
        risk = cache[
            cache_key
        ]
        prevalence = np.asarray(
            source_case["weights"],
            dtype=np.float64,
        )
        true_winner = names.index(
            source_case[
                "true_winner"
            ]
        )
        true_risk = (
            risk @ prevalence
        )
        tolerance = (
            source_case[
                "tolerance_ppm"
            ]
            / 1_000_000.0
        )
        bad = (
            true_risk
            - true_risk[
                true_winner
            ]
            > (
                tolerance
                + TOLERANCE
            )
        )
        bad[
            true_winner
        ] = False

        pairwise_differences = (
            risk[
                np.arange(
                    len(names)
                )
                != true_winner
            ]
            - risk[
                true_winner
            ][None, :]
        )
        pairwise_rates = (
            chernoff_rates(
                prevalence=prevalence[
                    None,
                    :,
                ],
                differences=(
                    pairwise_differences
                ),
            )
        )
        pairwise_bad = np.delete(
            bad,
            true_winner,
        )[None, :]
        e93_samples = int(
            required_union_samples(
                rates=pairwise_rates,
                included=pairwise_bad,
                confidence=CONFIDENCE,
            )[0]
        )

        policy_events = []
        global_rates = []
        for candidate in np.flatnonzero(
            bad
        ):
            event = (
                global_minimizer_rate(
                    prevalence=(
                        prevalence
                    ),
                    risk=risk,
                    candidate=int(
                        candidate
                    ),
                    true_winner=(
                        true_winner
                    ),
                )
            )
            policy_event_count += 1
            impossible_event_count += int(
                event[
                    "event_impossible"
                ]
            )
            optimizer_success_count += int(
                event[
                    "optimizer_success"
                ]
            )
            maximum_projected_violation = max(
                maximum_projected_violation,
                event[
                    "projected_gradient_violation"
                ],
            )
            global_rates.append(
                event["rate"]
            )
            serialized_event = {
                key: (
                    None
                    if (
                        isinstance(
                            value,
                            float,
                        )
                        and not math.isfinite(
                            value
                        )
                    )
                    else value
                )
                for key, value
                in event.items()
            }
            policy_events.append(
                {
                    "policy": names[
                        int(candidate)
                    ],
                    **serialized_event,
                }
            )
        global_samples = int(
            required_union_samples(
                rates=np.asarray(
                    global_rates,
                    dtype=np.float64,
                )[None, :],
                included=np.ones(
                    (
                        1,
                        len(
                            global_rates
                        ),
                    ),
                    dtype=bool,
                ),
                confidence=CONFIDENCE,
            )[0]
        ) if global_rates else 0
        if global_samples > e93_samples:
            raise RuntimeError(
                "Global-minimizer bound is "
                "worse than E93"
            )
        empirical = source_case[
            "first_empirically_certified"
        ]
        if empirical is None:
            raise RuntimeError(
                "E92 case was not "
                "empirically certified"
            )
        cases.append(
            {
                "kernel": source_case[
                    "kernel"
                ],
                "accuracy": source_case[
                    "accuracy"
                ],
                "tolerance_ppm": (
                    source_case[
                        "tolerance_ppm"
                    ]
                ),
                "sample_requirement_quantile": (
                    source_case[
                        "sample_requirement_quantile"
                    ]
                ),
                "profile_index": (
                    source_case[
                        "profile_index"
                    ]
                ),
                "weights": source_case[
                    "weights"
                ],
                "true_winner": (
                    source_case[
                        "true_winner"
                    ]
                ),
                "bad_policy_count": int(
                    np.count_nonzero(
                        bad
                    )
                ),
                "e91_samples": int(
                    source_case[
                        "bound_samples"
                    ]
                ),
                "e93_samples": (
                    e93_samples
                ),
                "global_samples": (
                    global_samples
                ),
                "e92_first_tested_samples": int(
                    empirical[
                        "samples"
                    ]
                ),
                "policy_events": (
                    policy_events
                ),
            }
        )
    overall = aggregate_cases(
        cases
    )
    result = {
        "suite": (
            "pacing_coalescing_order_global_chernoff"
        ),
        "model": (
            "For each true over-tolerance "
            "policy, bound the event that it "
            "becomes a global empirical "
            "regression minimizer. The "
            "multivariate exponential tilt "
            "enforces nonpositive empirical "
            "risk difference against every "
            "tested policy, then a union bound "
            "combines bad-policy events."
        ),
        "bound": {
            "event": (
                "candidate empirical regression "
                "<= every tested policy"
            ),
            "dual": (
                "sup_lambda>=0 -log E[exp("
                "-lambda dot D_candidate(X))]"
            ),
            "confidence": CONFIDENCE,
            "pairwise_warm_start": True,
            "dominance_guarantee": (
                "global rate >= E93 pairwise "
                "true-winner rate"
            ),
        },
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            ),
            "empirical_calibration": (
                "results/raw/"
                "pacing_coalescing_order_certificate_calibration/"
                "pacing_coalescing_order_certificate_calibration_full.json"
            ),
            "pairwise_chernoff": (
                "results/raw/"
                "pacing_coalescing_order_chernoff_certification/"
                "pacing_coalescing_order_chernoff_certification_full.json"
            ),
        },
        "cases": cases,
        "headline": {
            "overall": overall,
            "by_tolerance_ppm": {
                str(tolerance_ppm): (
                    aggregate_cases(
                        [
                            case
                            for case in cases
                            if case[
                                "tolerance_ppm"
                            ]
                            == tolerance_ppm
                        ]
                    )
                )
                for tolerance_ppm
                in (
                    0,
                    1,
                    10,
                    50,
                )
            },
            "by_profile_sample_quantile": {
                f"{probability:.2f}": (
                    aggregate_cases(
                        [
                            case
                            for case in cases
                            if case[
                                "sample_requirement_quantile"
                            ]
                            == probability
                        ]
                    )
                )
                for probability
                in (
                    0.10,
                    0.50,
                    0.90,
                )
            },
            "by_kernel": {
                kernel_name: (
                    aggregate_cases(
                        [
                            case
                            for case in cases
                            if case[
                                "kernel"
                            ]
                            == kernel_name
                        ]
                    )
                )
                for kernel_name
                in KERNELS
            },
        },
        "validation": {
            "case_count": len(
                cases
            ),
            "expected_case_count": len(
                e92["cases"]
            ),
            "policy_event_count": (
                policy_event_count
            ),
            "impossible_event_count": (
                impossible_event_count
            ),
            "optimizer_success_count": (
                optimizer_success_count
            ),
            "maximum_projected_gradient_violation": (
                maximum_projected_violation
            ),
            "global_samples_no_worse_than_e93": (
                all(
                    case[
                        "global_samples"
                    ]
                    <= case[
                        "e93_samples"
                    ]
                    for case in cases
                )
            ),
            "sample_bound_inversion_valid": (
                True
            ),
        },
        "limitations": [
            "The global-minimizer event ignores secondary delay, complexity, faster-rate, and name tie-breaking, so it remains a superset of actual bad-policy selection.",
            "The 192 cases are representative E92 profiles rather than the full 1.2-million profile-accuracy-tolerance grid.",
            "The union bound still ignores overlap among distinct bad policies' global-minimizer events.",
            "The analysis treats compact E83 regression risks as fixed and does not model uncertainty in those estimates.",
            "The audit is local statistical analysis and is not new GPU evidence.",
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
            "Unexpected E94 case count"
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
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
