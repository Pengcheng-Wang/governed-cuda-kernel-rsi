from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.special import gammaln, logsumexp

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    policy_inventory,
)
from analyze_pacing_coalescing_order_certificate_calibration import (
    policy_arrays,
    vectorized_winners,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    load_e83,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
    profile_metrics,
    winner,
)
from analyze_pacing_coalescing_order_prevalence_certification import (
    TOLERANCE,
    quantile,
)


FAILURE_TARGET = 0.05
MAX_ENUMERATION_STATES = 1_000_000
MAX_CUMULATIVE_SCAN_STATES = 1_000_000


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_e94(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_global_chernoff"
        / "pacing_coalescing_order_global_chernoff_full.json"
    )


def compositions(
    total: int,
    parts: int,
    prefix: tuple[int, ...] = (),
) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield prefix + (
            total,
        )
        return
    for value in range(
        total + 1
    ):
        yield from compositions(
            total - value,
            parts - 1,
            prefix + (
                value,
            ),
        )


def state_count(
    *,
    samples: int,
    support_size: int,
) -> int:
    if samples == 0:
        return 1
    return math.comb(
        samples
        + support_size
        - 1,
        support_size - 1,
    )


def cumulative_state_count(
    *,
    samples: int,
    support_size: int,
) -> int:
    if samples <= 0:
        return 0
    return (
        math.comb(
            samples
            + support_size,
            support_size,
        )
        - 1
    )


def exact_failure_probability(
    *,
    samples: int,
    prevalence: np.ndarray,
    bad_policies: np.ndarray,
    arrays: dict[str, Any],
) -> dict[str, Any]:
    if samples <= 0:
        return {
            "samples": samples,
            "state_count": 1,
            "failure_probability": (
                0.0
            ),
            "normalization_error": (
                0.0
            ),
        }
    support = np.flatnonzero(
        prevalence > 0.0
    )
    supported_probability = (
        prevalence[
            support
        ]
    )
    counts = np.asarray(
        list(
            compositions(
                samples,
                len(support),
            )
        ),
        dtype=np.int64,
    )
    empirical = np.zeros(
        (
            len(counts),
            len(prevalence),
        ),
        dtype=np.float64,
    )
    empirical[
        :,
        support,
    ] = counts / samples
    winners = vectorized_winners(
        prevalence=empirical,
        arrays=arrays,
    )
    failures = bad_policies[
        winners
    ]
    log_probability = (
        gammaln(
            samples + 1
        )
        - np.sum(
            gammaln(
                counts + 1
            ),
            axis=1,
        )
        + counts
        @ np.log(
            supported_probability
        )
    )
    total_log_probability = float(
        logsumexp(
            log_probability
        )
    )
    failure_probability = (
        0.0
        if not np.any(
            failures
        )
        else float(
            math.exp(
                logsumexp(
                    log_probability[
                        failures
                    ]
                )
            )
        )
    )
    return {
        "samples": samples,
        "state_count": len(
            counts
        ),
        "failure_probability": (
            failure_probability
        ),
        "normalization_error": abs(
            math.exp(
                total_log_probability
            )
            - 1.0
        ),
    }


def validate_vectorized_winners(
    *,
    per_policy_costs: dict[
        str,
        list[dict[str, Any]],
    ],
    inventory: dict[str, dict[str, Any]],
    arrays: dict[str, Any],
) -> int:
    counts = np.asarray(
        list(
            compositions(
                6,
                arrays[
                    "regression"
                ].shape[1],
            )
        ),
        dtype=np.int64,
    )
    prevalence = counts / 6.0
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
    return len(
        counts
    )


def ratio_summary(
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
    scanned = [
        case
        for case in cases
        if case[
            "exact_scan"
        ]
        is not None
    ]
    exact_samples = [
        case[
            "exact_scan"
        ][
            "sustained_failure_target_samples"
        ]
        for case in scanned
    ]

    def ratios(
        field: str,
    ) -> list[float]:
        return [
            case[field]
            / max(
                1,
                case[
                    "exact_scan"
                ][
                    "sustained_failure_target_samples"
                ],
            )
            for case in scanned
        ]

    point_names = (
        "e92_first_tested",
        "e93",
        "e94",
    )
    point_summary = {}
    for point_name in point_names:
        checked = [
            case[
                "point_checks"
            ][point_name]
            for case in cases
            if case[
                "point_checks"
            ][point_name]
            is not None
        ]
        point_summary[
            point_name
        ] = {
            "checked_count": len(
                checked
            ),
            "passing_count": sum(
                row[
                    "failure_probability"
                ]
                <= (
                    FAILURE_TARGET
                    + 1e-14
                )
                for row in checked
            ),
            "maximum_failure_probability": (
                max(
                    row[
                        "failure_probability"
                    ]
                    for row in checked
                )
                if checked
                else None
            ),
        }
    return {
        "case_count": len(cases),
        "exact_scan_count": len(
            scanned
        ),
        "unscanned_count": (
            len(cases)
            - len(scanned)
        ),
        "exact_sustained_sample_quantiles": {
            "p10": quantile(
                exact_samples,
                0.10,
            ),
            "p50": quantile(
                exact_samples,
                0.50,
            ),
            "p90": quantile(
                exact_samples,
                0.90,
            ),
            "maximum": (
                max(exact_samples)
                if exact_samples
                else None
            ),
        },
        "e91_to_exact_ratio": (
            ratio_summary(
                ratios(
                    "e91_samples"
                )
            )
            if scanned
            else None
        ),
        "e93_to_exact_ratio": (
            ratio_summary(
                ratios(
                    "e93_samples"
                )
            )
            if scanned
            else None
        ),
        "e94_to_exact_ratio": (
            ratio_summary(
                ratios(
                    "global_samples"
                )
            )
            if scanned
            else None
        ),
        "e92_to_exact_ratio": (
            ratio_summary(
                ratios(
                    "e92_first_tested_samples"
                )
            )
            if scanned
            else None
        ),
        "exact_no_more_than_e92_count": (
            sum(
                case[
                    "exact_scan"
                ][
                    "sustained_failure_target_samples"
                ]
                <= case[
                    "e92_first_tested_samples"
                ]
                for case in scanned
            )
        ),
        "nonmonotone_case_count": sum(
            case[
                "exact_scan"
            ][
                "nonmonotone_increase_count"
            ]
            > 0
            for case in scanned
        ),
        "maximum_nonmonotone_increase": (
            max(
                case[
                    "exact_scan"
                ][
                    "maximum_nonmonotone_increase"
                ]
                for case in scanned
            )
            if scanned
            else None
        ),
        "point_checks": point_summary,
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e94 = load_e94(args.root)
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
    cache = {}
    cases = []
    vectorized_winner_checks = 0
    maximum_normalization_error = 0.0
    enumerated_state_count = 0
    for source_case in e94[
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
            arrays = policy_arrays(
                per_policy_costs=(
                    per_policy_costs
                ),
                inventory=inventory,
            )
            vectorized_winner_checks += (
                validate_vectorized_winners(
                    per_policy_costs=(
                        per_policy_costs
                    ),
                    inventory=inventory,
                    arrays=arrays,
                )
            )
            cache[cache_key] = (
                arrays
            )
        arrays = cache[
            cache_key
        ]
        prevalence = np.asarray(
            source_case["weights"],
            dtype=np.float64,
        )
        true_winner = (
            arrays["names"].index(
                source_case[
                    "true_winner"
                ]
            )
        )
        true_risk = (
            arrays["regression"]
            @ prevalence
        )
        tolerance = (
            source_case[
                "tolerance_ppm"
            ]
            / 1_000_000.0
        )
        bad_policies = (
            true_risk
            - true_risk[
                true_winner
            ]
            > (
                tolerance
                + TOLERANCE
            )
        )
        bad_policies[
            true_winner
        ] = False
        support_size = int(
            np.count_nonzero(
                prevalence > 0.0
            )
        )
        probability_cache = {}

        def exact(
            samples: int,
        ) -> dict[str, Any]:
            nonlocal maximum_normalization_error
            nonlocal enumerated_state_count
            if samples not in (
                probability_cache
            ):
                row = (
                    exact_failure_probability(
                        samples=samples,
                        prevalence=(
                            prevalence
                        ),
                        bad_policies=(
                            bad_policies
                        ),
                        arrays=arrays,
                    )
                )
                probability_cache[
                    samples
                ] = row
                maximum_normalization_error = max(
                    maximum_normalization_error,
                    row[
                        "normalization_error"
                    ],
                )
                enumerated_state_count += (
                    row[
                        "state_count"
                    ]
                )
            return probability_cache[
                samples
            ]

        global_samples = int(
            source_case[
                "global_samples"
            ]
        )
        scan = None
        if (
            cumulative_state_count(
                samples=(
                    global_samples
                ),
                support_size=(
                    support_size
                ),
            )
            <= (
                MAX_CUMULATIVE_SCAN_STATES
            )
        ):
            probabilities = [
                exact(
                    samples
                )[
                    "failure_probability"
                ]
                for samples in range(
                    1,
                    global_samples + 1,
                )
            ]
            failures = [
                index + 1
                for index, probability
                in enumerate(
                    probabilities
                )
                if probability
                > (
                    FAILURE_TARGET
                    + 1e-14
                )
            ]
            sustained = (
                0
                if not np.any(
                    bad_policies
                )
                else (
                    max(failures) + 1
                    if failures
                    else 1
                )
            )
            increases = [
                max(
                    0.0,
                    probabilities[index]
                    - probabilities[
                        index - 1
                    ],
                )
                for index in range(
                    1,
                    len(
                        probabilities
                    ),
                )
            ]
            scan = {
                "cumulative_state_count": (
                    cumulative_state_count(
                        samples=(
                            global_samples
                        ),
                        support_size=(
                            support_size
                        ),
                    )
                ),
                "sustained_failure_target_samples": (
                    sustained
                ),
                "last_failing_samples": (
                    max(failures)
                    if failures
                    else None
                ),
                "failure_probability_at_sustained": (
                    0.0
                    if sustained == 0
                    else exact(
                        sustained
                    )[
                        "failure_probability"
                    ]
                ),
                "failure_probability_before_sustained": (
                    None
                    if sustained <= 1
                    else exact(
                        sustained - 1
                    )[
                        "failure_probability"
                    ]
                ),
                "nonmonotone_increase_count": (
                    sum(
                        increase
                        > 1e-12
                        for increase
                        in increases
                    )
                ),
                "maximum_nonmonotone_increase": (
                    max(
                        increases,
                        default=0.0,
                    )
                ),
            }

        point_checks = {}
        for point_name, samples in (
            (
                "e92_first_tested",
                int(
                    source_case[
                        "e92_first_tested_samples"
                    ]
                ),
            ),
            (
                "e93",
                int(
                    source_case[
                        "e93_samples"
                    ]
                ),
            ),
            (
                "e94",
                global_samples,
            ),
        ):
            if (
                state_count(
                    samples=samples,
                    support_size=(
                        support_size
                    ),
                )
                <= (
                    MAX_ENUMERATION_STATES
                )
            ):
                point_checks[
                    point_name
                ] = exact(
                    samples
                )
            else:
                point_checks[
                    point_name
                ] = None
        for point_name in (
            "e93",
            "e94",
        ):
            row = point_checks[
                point_name
            ]
            if (
                row is not None
                and row[
                    "failure_probability"
                ]
                > (
                    FAILURE_TARGET
                    + 1e-12
                )
            ):
                raise RuntimeError(
                    f"{point_name} exact "
                    "failure exceeds target"
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
                "support_size": (
                    support_size
                ),
                "bad_policy_count": int(
                    np.count_nonzero(
                        bad_policies
                    )
                ),
                "e91_samples": int(
                    source_case[
                        "e91_samples"
                    ]
                ),
                "e92_first_tested_samples": int(
                    source_case[
                        "e92_first_tested_samples"
                    ]
                ),
                "e93_samples": int(
                    source_case[
                        "e93_samples"
                    ]
                ),
                "global_samples": (
                    global_samples
                ),
                "exact_scan": scan,
                "point_checks": (
                    point_checks
                ),
            }
        )
    overall = aggregate_cases(
        cases
    )
    result = {
        "suite": (
            "pacing_coalescing_order_exact_multinomial"
        ),
        "model": (
            "Enumerate all multinomial count "
            "vectors for tractable representative "
            "E94 cases, apply the full "
            "lexicographic policy selector, and "
            "sum exact count probabilities. "
            "Scanned cases report the first "
            "sample after the last exact failure "
            "above 5%; E94 guarantees all later "
            "sample sizes."
        ),
        "enumeration": {
            "failure_target": (
                FAILURE_TARGET
            ),
            "maximum_point_states": (
                MAX_ENUMERATION_STATES
            ),
            "maximum_cumulative_scan_states": (
                MAX_CUMULATIVE_SCAN_STATES
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
            "global_chernoff": (
                "results/raw/"
                "pacing_coalescing_order_global_chernoff/"
                "pacing_coalescing_order_global_chernoff_full.json"
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
            "by_support_size": {
                str(support_size): (
                    aggregate_cases(
                        [
                            case
                            for case in cases
                            if case[
                                "support_size"
                            ]
                            == support_size
                        ]
                    )
                )
                for support_size
                in range(
                    1,
                    len(costs) + 1,
                )
                if any(
                    case[
                        "support_size"
                    ]
                    == support_size
                    for case in cases
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
        },
        "validation": {
            "case_count": len(
                cases
            ),
            "expected_case_count": len(
                e94["cases"]
            ),
            "vectorized_winner_matches": (
                vectorized_winner_checks
            ),
            "enumerated_state_count": (
                enumerated_state_count
            ),
            "maximum_probability_normalization_error": (
                maximum_normalization_error
            ),
            "e93_exact_points_within_target": (
                True
            ),
            "e94_exact_points_within_target": (
                True
            ),
            "sustained_threshold_uses_last_failure": (
                True
            ),
        },
        "limitations": [
            "Exact scans cover cases with at most one million cumulative count states; larger representative cases retain E94/E93 analytic certificates.",
            "Point checks are omitted when a single sample size exceeds one million multinomial count states.",
            "The exact probabilities are conditional on fixed E83 path-risk estimates and the declared categorical prevalence model.",
            "The audit covers E92's 192 representative cases rather than the full prevalence/accuracy/tolerance grid.",
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
            "Unexpected E95 case count"
        )
    if (
        maximum_normalization_error
        > 1e-10
    ):
        raise RuntimeError(
            "Multinomial probability did "
            "not normalize"
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
