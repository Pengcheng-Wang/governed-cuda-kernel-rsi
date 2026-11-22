from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analyze_pacing_policy_compression import (
    FAMILIES,
    group_dict,
    group_key,
    index_records,
    load_e62,
    oracle_windows_from_e62,
    state_dict,
)


ALL_DIMENSIONS = (
    "burst_mode",
    "offered_load",
    "prefix_mode",
)

ERROR_PROFILES = {
    "perfect": {
        "burst_mode": 0.0,
        "offered_load": 0.0,
        "prefix_mode": 0.0,
    },
    "all_5pct": {
        "burst_mode": 0.05,
        "offered_load": 0.05,
        "prefix_mode": 0.05,
    },
    "all_10pct": {
        "burst_mode": 0.10,
        "offered_load": 0.10,
        "prefix_mode": 0.10,
    },
    "all_20pct": {
        "burst_mode": 0.20,
        "offered_load": 0.20,
        "prefix_mode": 0.20,
    },
    "burst_10pct": {
        "burst_mode": 0.10,
        "offered_load": 0.0,
        "prefix_mode": 0.0,
    },
    "load_10pct": {
        "burst_mode": 0.0,
        "offered_load": 0.10,
        "prefix_mode": 0.0,
    },
    "cache_10pct": {
        "burst_mode": 0.0,
        "offered_load": 0.0,
        "prefix_mode": 0.10,
    },
}

STRATEGIES = (
    "naive",
    "hamming1_safe",
)


def load_e63(
    root: Path,
) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_policy_compression"
        / "pacing_policy_compression_full.json"
    )
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def state_from_dict(
    values: dict[str, Any],
) -> tuple[str, float, str]:
    return (
        values["prefix_mode"],
        float(values["offered_load"]),
        values["burst_mode"],
    )


def key_tuple(
    dimensions: tuple[str, ...],
    values: dict[str, Any],
) -> tuple[Any, ...]:
    return tuple(
        values[dimension]
        for dimension in dimensions
    )


def robust_mapping_from_e63(
    *,
    e63: dict[str, Any],
    family: str,
    slo_ms: float,
) -> dict[tuple[Any, ...], float]:
    record = next(
        record
        for record in e63["records"]
        if record["family"] == family
        and record["selection_mode"]
        == "robust"
        and float(
            record["ttft_slo_ms"]
        )
        == slo_ms
    )
    dimensions = FAMILIES[family]
    return {
        key_tuple(
            dimensions,
            group["key"],
        ): float(
            group["selected_window_ms"]
        )
        for group in record["groups"]
    }


def hamming_distance(
    left: tuple[Any, ...],
    right: tuple[Any, ...],
) -> int:
    if len(left) != len(right):
        raise ValueError(
            "Hamming length mismatch"
        )
    return sum(
        first != second
        for first, second
        in zip(left, right)
    )


def hamming1_mapping(
    *,
    family: str,
    slo_ms: float,
    states: list[
        tuple[str, float, str]
    ],
    windows: list[float],
    lookup: dict[
        tuple[str, float, str, float],
        dict[str, Any],
    ],
) -> tuple[
    dict[tuple[Any, ...], float],
    list[dict[str, Any]],
]:
    dimensions = FAMILIES[family]
    keys = sorted(
        {
            group_key(
                state,
                dimensions,
            )
            for state in states
        }
    )
    mapping = {}
    records = []
    for estimated_key in keys:
        neighborhood = [
            state
            for state in states
            if hamming_distance(
                group_key(
                    state,
                    dimensions,
                ),
                estimated_key,
            )
            <= 1
        ]
        feasible = [
            window
            for window in windows
            if all(
                lookup[
                    (*state, window)
                ]["metrics"][
                    "p99_ttft_ms"
                ]["maximum"]
                <= slo_ms + 1e-9
                for state in neighborhood
            )
        ]
        if not feasible:
            raise RuntimeError(
                "Hamming-1 envelope became "
                f"infeasible: {family=} "
                f"{slo_ms=} {estimated_key=}"
            )
        selected = max(
            feasible,
            key=lambda window: (
                sum(
                    lookup[
                        (*state, window)
                    ]["metrics"][
                        "throughput_output_"
                        "tokens_s"
                    ]["median"]
                    for state
                    in neighborhood
                ),
                -max(
                    lookup[
                        (*state, window)
                    ]["metrics"][
                        "p99_ttft_ms"
                    ]["maximum"]
                    for state
                    in neighborhood
                ),
                -window,
            ),
        )
        mapping[estimated_key] = (
            selected
        )
        records.append(
            {
                "estimated_key": (
                    group_dict(
                        dimensions,
                        estimated_key,
                    )
                ),
                "selected_window_ms": (
                    selected
                ),
                "neighborhood_states": [
                    state_dict(state)
                    for state
                    in neighborhood
                ],
                "maximum_neighborhood_p99_ttft_ms": (
                    max(
                        lookup[
                            (*state, selected)
                        ]["metrics"][
                            "p99_ttft_ms"
                        ]["maximum"]
                        for state
                        in neighborhood
                    )
                ),
            }
        )
    return mapping, records


def flip_value(
    *,
    dimension: str,
    value: Any,
) -> Any:
    if dimension == "burst_mode":
        return (
            "burst_16"
            if value == "burst_4"
            else "burst_4"
        )
    if dimension == "offered_load":
        return (
            0.4
            if float(value) == 0.3
            else 0.3
        )
    if dimension == "prefix_mode":
        return (
            "half_requests_75pct_cached"
            if value == "none"
            else "none"
        )
    raise ValueError(dimension)


def estimate_distribution(
    *,
    true_state: tuple[str, float, str],
    probabilities: dict[str, float],
) -> list[
    tuple[
        tuple[str, float, str],
        float,
        int,
    ]
]:
    true_values = state_dict(
        true_state
    )
    estimates = []
    for flips in itertools.product(
        (False, True),
        repeat=len(ALL_DIMENSIONS),
    ):
        probability = 1.0
        values = dict(true_values)
        flip_count = 0
        for dimension, flipped in zip(
            ALL_DIMENSIONS,
            flips,
        ):
            error = float(
                probabilities[dimension]
            )
            probability *= (
                error
                if flipped
                else 1.0 - error
            )
            if flipped:
                values[dimension] = (
                    flip_value(
                        dimension=dimension,
                        value=values[
                            dimension
                        ],
                    )
                )
                flip_count += 1
        if probability <= 0.0:
            continue
        estimates.append(
            (
                state_from_dict(values),
                probability,
                flip_count,
            )
        )
    total = sum(
        probability
        for _, probability, _
        in estimates
    )
    if abs(total - 1.0) > 1e-12:
        raise RuntimeError(
            "Estimator distribution failed"
        )
    return estimates


def evaluate_strategy(
    *,
    family: str,
    strategy: str,
    slo_ms: float,
    error_profile: str,
    mapping: dict[
        tuple[Any, ...],
        float,
    ],
    states: list[
        tuple[str, float, str]
    ],
    lookup: dict[
        tuple[str, float, str, float],
        dict[str, Any],
    ],
    oracle_windows: dict[
        tuple[str, float, str],
        float,
    ],
) -> dict[str, Any]:
    dimensions = FAMILIES[family]
    probabilities = ERROR_PROFILES[
        error_profile
    ]
    state_records = []
    mean_ratios = []
    state_expected_ratios = []
    state_expected_acceptance_delta = []
    scenario_violation_probability = 0.0
    unsafe_path_probability = 0.0
    maximum_excess_ms = 0.0
    expected_selected_window_ms = 0.0
    expected_key_error_count = 0.0
    state_weight = 1.0 / len(states)
    for true_state in states:
        oracle_window = oracle_windows[
            true_state
        ]
        oracle_record = lookup[
            (*true_state, oracle_window)
        ]
        oracle_throughput = (
            oracle_record["metrics"][
                "throughput_output_tokens_s"
            ]["median"]
        )
        oracle_acceptance = (
            oracle_record["metrics"][
                "accepted_fraction"
            ]["median"]
        )
        expected_ratio = 0.0
        expected_acceptance_delta = 0.0
        state_violation_probability = (
            0.0
        )
        state_unsafe_path_probability = (
            0.0
        )
        state_maximum_excess = 0.0
        state_expected_window = 0.0
        state_expected_key_errors = 0.0
        paths = []
        true_key = group_key(
            true_state,
            dimensions,
        )
        for (
            estimated_state,
            probability,
            full_flip_count,
        ) in estimate_distribution(
            true_state=true_state,
            probabilities=probabilities,
        ):
            estimated_key = group_key(
                estimated_state,
                dimensions,
            )
            key_error_count = (
                hamming_distance(
                    true_key,
                    estimated_key,
                )
            )
            window = mapping[
                estimated_key
            ]
            record = lookup[
                (*true_state, window)
            ]
            throughput = (
                record["metrics"][
                    "throughput_output_"
                    "tokens_s"
                ]["median"]
            )
            ratio = (
                throughput
                / oracle_throughput
            )
            acceptance_delta = (
                record["metrics"][
                    "accepted_fraction"
                ]["median"]
                - oracle_acceptance
            )
            violations = [
                max(
                    0.0,
                    float(
                        scenario[
                            "p99_ttft_ms"
                        ]
                    )
                    - slo_ms,
                )
                for scenario
                in record["scenarios"]
            ]
            violation_fraction = (
                sum(
                    excess > 0.0
                    for excess
                    in violations
                )
                / len(violations)
            )
            path_maximum_excess = max(
                violations
            )
            unsafe = (
                path_maximum_excess > 0.0
            )
            expected_ratio += (
                probability * ratio
            )
            expected_acceptance_delta += (
                probability
                * acceptance_delta
            )
            state_violation_probability += (
                probability
                * violation_fraction
            )
            state_unsafe_path_probability += (
                probability * float(unsafe)
            )
            state_maximum_excess = max(
                state_maximum_excess,
                path_maximum_excess,
            )
            state_expected_window += (
                probability * window
            )
            state_expected_key_errors += (
                probability
                * key_error_count
            )
            paths.append(
                {
                    "estimated_state": (
                        state_dict(
                            estimated_state
                        )
                    ),
                    "estimated_key": (
                        group_dict(
                            dimensions,
                            estimated_key,
                        )
                    ),
                    "probability": (
                        probability
                    ),
                    "full_flip_count": (
                        full_flip_count
                    ),
                    "key_error_count": (
                        key_error_count
                    ),
                    "selected_window_ms": (
                        window
                    ),
                    "throughput_ratio_vs_oracle": (
                        ratio
                    ),
                    "acceptance_delta_vs_oracle": (
                        acceptance_delta
                    ),
                    "scenario_slo_violation_fraction": (
                        violation_fraction
                    ),
                    "maximum_slo_excess_ms": (
                        path_maximum_excess
                    ),
                }
            )
        state_records.append(
            {
                **state_dict(true_state),
                "oracle_window_ms": (
                    oracle_window
                ),
                "expected_throughput_ratio_vs_oracle": (
                    expected_ratio
                ),
                "expected_acceptance_delta_vs_oracle": (
                    expected_acceptance_delta
                ),
                "expected_scenario_slo_violation_fraction": (
                    state_violation_probability
                ),
                "unsafe_path_probability": (
                    state_unsafe_path_probability
                ),
                "maximum_slo_excess_ms": (
                    state_maximum_excess
                ),
                "expected_selected_window_ms": (
                    state_expected_window
                ),
                "expected_key_error_count": (
                    state_expected_key_errors
                ),
                "paths": paths,
            }
        )
        mean_ratios.append(expected_ratio)
        state_expected_ratios.append(
            expected_ratio
        )
        state_expected_acceptance_delta.append(
            expected_acceptance_delta
        )
        scenario_violation_probability += (
            state_weight
            * state_violation_probability
        )
        unsafe_path_probability += (
            state_weight
            * state_unsafe_path_probability
        )
        maximum_excess_ms = max(
            maximum_excess_ms,
            state_maximum_excess,
        )
        expected_selected_window_ms += (
            state_weight
            * state_expected_window
        )
        expected_key_error_count += (
            state_weight
            * state_expected_key_errors
        )
    return {
        "family": family,
        "dimensions": list(dimensions),
        "key_count": len(mapping),
        "strategy": strategy,
        "ttft_slo_ms": slo_ms,
        "error_profile": error_profile,
        "error_probabilities": (
            probabilities
        ),
        "mean_expected_throughput_ratio_vs_oracle": (
            statistics.mean(
                mean_ratios
            )
        ),
        "geomean_expected_throughput_ratio_vs_oracle": (
            math.exp(
                statistics.mean(
                    math.log(value)
                    for value
                    in mean_ratios
                )
            )
        ),
        "worst_expected_state_throughput_ratio_vs_oracle": (
            min(
                state_expected_ratios
            )
        ),
        "mean_expected_acceptance_delta_vs_oracle": (
            statistics.mean(
                state_expected_acceptance_delta
            )
        ),
        "worst_expected_acceptance_delta_vs_oracle": (
            min(
                state_expected_acceptance_delta
            )
        ),
        "expected_scenario_slo_violation_fraction": (
            scenario_violation_probability
        ),
        "unsafe_path_probability": (
            unsafe_path_probability
        ),
        "maximum_slo_excess_ms": (
            maximum_excess_ms
        ),
        "expected_selected_window_ms": (
            expected_selected_window_ms
        ),
        "expected_key_error_count": (
            expected_key_error_count
        ),
        "mapping": [
            {
                "key": group_dict(
                    dimensions,
                    key,
                ),
                "selected_window_ms": (
                    window
                ),
            }
            for key, window
            in sorted(mapping.items())
        ],
        "states": state_records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e62 = load_e62(args.root)
    e63 = load_e63(args.root)
    states, windows, lookup = (
        index_records(e62)
    )
    slos_ms = [
        float(value)
        for value in args.ttft_slos_ms.split(",")
    ]
    families = [
        value.strip()
        for value in args.families.split(",")
        if value.strip()
    ]
    profiles = [
        value.strip()
        for value in args.error_profiles.split(",")
        if value.strip()
    ]
    unknown_families = set(
        families
    ) - set(FAMILIES)
    if unknown_families:
        raise ValueError(
            f"Unknown families: "
            f"{sorted(unknown_families)}"
        )
    unknown_profiles = set(
        profiles
    ) - set(ERROR_PROFILES)
    if unknown_profiles:
        raise ValueError(
            f"Unknown error profiles: "
            f"{sorted(unknown_profiles)}"
        )
    records = []
    envelope_records = []
    validation = []
    for slo_ms in slos_ms:
        oracle_windows = (
            oracle_windows_from_e62(
                e62=e62,
                slo_ms=slo_ms,
            )
        )
        for family in families:
            naive_mapping = (
                robust_mapping_from_e63(
                    e63=e63,
                    family=family,
                    slo_ms=slo_ms,
                )
            )
            (
                safe_mapping,
                safe_details,
            ) = hamming1_mapping(
                family=family,
                slo_ms=slo_ms,
                states=states,
                windows=windows,
                lookup=lookup,
            )
            envelope_records.append(
                {
                    "ttft_slo_ms": (
                        slo_ms
                    ),
                    "family": family,
                    "dimensions": list(
                        FAMILIES[family]
                    ),
                    "naive_mapping": [
                        {
                            "key": group_dict(
                                FAMILIES[
                                    family
                                ],
                                key,
                            ),
                            "selected_window_ms": (
                                window
                            ),
                        }
                        for key, window
                        in sorted(
                            naive_mapping.items()
                        )
                    ],
                    "hamming1_safe_mapping": (
                        safe_details
                    ),
                }
            )
            for profile in profiles:
                for strategy, mapping in (
                    (
                        "naive",
                        naive_mapping,
                    ),
                    (
                        "hamming1_safe",
                        safe_mapping,
                    ),
                ):
                    evaluated = (
                        evaluate_strategy(
                            family=family,
                            strategy=strategy,
                            slo_ms=slo_ms,
                            error_profile=profile,
                            mapping=mapping,
                            states=states,
                            lookup=lookup,
                            oracle_windows=(
                                oracle_windows
                            ),
                        )
                    )
                    records.append(evaluated)
                    if (
                        profile == "perfect"
                        and strategy == "naive"
                    ):
                        e63_record = next(
                            record
                            for record
                            in e63["records"]
                            if record[
                                "family"
                            ]
                            == family
                            and record[
                                "selection_mode"
                            ]
                            == "robust"
                            and float(
                                record[
                                    "ttft_slo_ms"
                                ]
                            )
                            == slo_ms
                        )
                        differences = {
                            "mean": abs(
                                evaluated[
                                    "mean_expected_throughput_ratio_vs_oracle"
                                ]
                                - e63_record[
                                    "mean_throughput_ratio_vs_oracle"
                                ]
                            ),
                            "geomean": abs(
                                evaluated[
                                    "geomean_expected_throughput_ratio_vs_oracle"
                                ]
                                - e63_record[
                                    "geomean_throughput_ratio_vs_oracle"
                                ]
                            ),
                            "worst": abs(
                                evaluated[
                                    "worst_expected_state_throughput_ratio_vs_oracle"
                                ]
                                - e63_record[
                                    "worst_throughput_ratio_vs_oracle"
                                ]
                            ),
                        }
                        maximum = max(
                            differences.values()
                        )
                        if maximum > 1e-12:
                            raise RuntimeError(
                                "E63 zero-error "
                                "reproduction failed"
                            )
                        validation.append(
                            {
                                "ttft_slo_ms": (
                                    slo_ms
                                ),
                                "family": family,
                                "differences": (
                                    differences
                                ),
                                "maximum_absolute_difference": (
                                    maximum
                                ),
                            }
                        )
                    if (
                        strategy
                        == "hamming1_safe"
                        and profile
                        in (
                            "burst_10pct",
                            "load_10pct",
                            "cache_10pct",
                        )
                        and evaluated[
                            "expected_scenario_slo_violation_fraction"
                        ]
                        > 1e-15
                    ):
                        raise RuntimeError(
                            "One-feature safe "
                            "envelope violated SLO "
                            f"for {family=} "
                            f"{slo_ms=} {profile=}"
                        )

    result = {
        "suite": (
            "pacing_state_uncertainty_"
            "analysis"
        ),
        "model": (
            "Apply independent burst/load/cache "
            "classification errors to E63 robust "
            "lookup families and compare naive "
            "lookup with a Hamming-radius-1 "
            "uncertainty-safe envelope."
        ),
        "source": {
            "slo_pacing": (
                "results/raw/"
                "slo_ingress_pacing/"
                "slo_ingress_pacing_full.json"
            ),
            "policy_compression": (
                "results/raw/"
                "pacing_policy_compression/"
                "pacing_policy_compression_full.json"
            ),
        },
        "state_dimensions": list(
            ALL_DIMENSIONS
        ),
        "states": [
            state_dict(state)
            for state in states
        ],
        "candidate_windows_ms": windows,
        "ttft_slos_ms": slos_ms,
        "families": {
            family: {
                "dimensions": list(
                    FAMILIES[family]
                ),
                "key_count": len(
                    {
                        group_key(
                            state,
                            FAMILIES[family],
                        )
                        for state in states
                    }
                ),
            }
            for family in families
        },
        "error_profiles": {
            profile: ERROR_PROFILES[
                profile
            ]
            for profile in profiles
        },
        "strategies": {
            "naive": (
                "Use the E63 robust window for "
                "the estimated policy key."
            ),
            "hamming1_safe": (
                "For each estimated key, select "
                "the maximum-throughput window "
                "that is SLO-feasible for every "
                "true state whose keyed features "
                "differ in at most one bit."
            ),
        },
        "envelopes": envelope_records,
        "records": records,
        "validation": {
            "e63_zero_error_checks": (
                validation
            ),
            "e63_zero_error_maximum_absolute_difference": (
                max(
                    item[
                        "maximum_absolute_difference"
                    ]
                    for item in validation
                )
            ),
            "expected_record_count": (
                len(slos_ms)
                * len(families)
                * len(profiles)
                * len(STRATEGIES)
            ),
            "one_feature_safety": (
                "Every Hamming-1-safe policy "
                "has zero violations for each "
                "single-dimension 10% error "
                "profile. Under simultaneous "
                "independent errors, residual "
                "violations can occur only on "
                "two-or-more keyed-feature "
                "mistakes."
            ),
        },
        "limitations": [
            "The analysis inherits E62-E63's finite state/window grid and fitted-service simulation.",
            "Estimator errors are independent binary flips rather than a calibrated production classifier or continuous state estimator.",
            "True states are equally weighted; production traffic prevalence can change expected throughput and violation probability.",
            "The Hamming-1 envelope guarantees one keyed-feature mistake but not simultaneous two- or three-feature mistakes.",
            "The safe envelope optimizes equal-weight neighborhood throughput rather than tenant utility, cost, or switching hysteresis.",
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
        "--ttft-slos-ms",
        default="650,800,1000",
    )
    parser.add_argument(
        "--families",
        default=(
            "universal,burst,load,cache,"
            "burst_load,burst_cache,"
            "load_cache,full"
        ),
    )
    parser.add_argument(
        "--error-profiles",
        default=(
            "perfect,all_5pct,all_10pct,"
            "all_20pct,burst_10pct,"
            "load_10pct,cache_10pct"
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
