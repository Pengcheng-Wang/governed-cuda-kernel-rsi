from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_pacing_confidence_sets import (
    COMPARISON_METRICS,
    CONFIDENCE_PROFILES,
    confidence_set_mapping,
    evaluate_confidence_profile,
)
from analyze_pacing_policy_compression import (
    FAMILIES,
    group_key,
    index_records,
    load_e62,
    oracle_windows_from_e62,
)
from analyze_pacing_state_uncertainty import (
    ERROR_PROFILES,
    load_e63,
)


DEFAULT_RECALLS = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
    1.00,
)

DEFAULT_FALSE_POSITIVE_RATES = (
    0.0,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    1.00,
)

DEFAULT_RISK_BUDGETS = (
    0.0,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.05,
)

E65_PROFILE_POINTS = {
    name: (
        float(values["mistake_flag_sensitivity"]),
        float(
            values[
                "correct_flag_false_positive_rate"
            ]
        ),
    )
    for name, values in CONFIDENCE_PROFILES.items()
}

POINT_METRICS = (
    "mean_expected_throughput_ratio_vs_oracle",
    "geomean_expected_throughput_ratio_vs_oracle",
    "worst_expected_state_throughput_ratio_vs_oracle",
    "mean_expected_acceptance_delta_vs_oracle",
    "worst_expected_acceptance_delta_vs_oracle",
    "expected_scenario_slo_violation_fraction",
    "unsafe_path_probability",
    "confidence_miss_probability",
    "any_false_alarm_probability",
    "expected_missed_key_error_count",
    "expected_false_alarm_count",
    "expected_uncertain_dimension_count",
    "expected_confidence_set_key_count",
    "expected_selected_window_ms",
    "expected_key_error_count",
    "no_flag_probability",
    "all_key_dimensions_flagged_probability",
    "maximum_slo_excess_ms",
    "mean_throughput_delta_vs_naive",
    "mean_throughput_delta_vs_hamming1_safe",
    "scenario_violation_delta_vs_naive",
    "scenario_violation_delta_vs_hamming1_safe",
)


def load_e64(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_state_uncertainty"
        / "pacing_state_uncertainty_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_e65(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_confidence_sets"
        / "pacing_confidence_sets_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float_csv(value: str) -> list[float]:
    return [
        float(item)
        for item in value.split(",")
        if item.strip()
    ]


def compact_point(
    evaluated: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mistake_flag_sensitivity": (
            evaluated[
                "mistake_flag_sensitivity"
            ]
        ),
        "correct_flag_false_positive_rate": (
            evaluated[
                "correct_flag_false_positive_rate"
            ]
        ),
        **{
            metric: evaluated[metric]
            for metric in POINT_METRICS
        },
    }


def pareto_frontier(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frontier = []
    for point in points:
        dominated = any(
            other[
                "mean_expected_throughput_ratio_vs_oracle"
            ]
            >= point[
                "mean_expected_throughput_ratio_vs_oracle"
            ]
            - 1e-15
            and other[
                "expected_scenario_slo_violation_fraction"
            ]
            <= point[
                "expected_scenario_slo_violation_fraction"
            ]
            + 1e-15
            and (
                other[
                    "mean_expected_throughput_ratio_vs_oracle"
                ]
                > point[
                    "mean_expected_throughput_ratio_vs_oracle"
                ]
                + 1e-15
                or other[
                    "expected_scenario_slo_violation_fraction"
                ]
                < point[
                    "expected_scenario_slo_violation_fraction"
                ]
                - 1e-15
            )
            for other in points
            if other is not point
        )
        if not dominated:
            frontier.append(point)
    return sorted(
        frontier,
        key=lambda point: (
            point[
                "expected_scenario_slo_violation_fraction"
            ],
            -point[
                "mean_expected_throughput_ratio_vs_oracle"
            ],
            -point[
                "mistake_flag_sensitivity"
            ],
            point[
                "correct_flag_false_positive_rate"
            ],
        ),
    )


def select_for_budget(
    *,
    points: list[dict[str, Any]],
    budget: float,
) -> dict[str, Any]:
    feasible = [
        point
        for point in points
        if point[
            "expected_scenario_slo_violation_fraction"
        ]
        <= budget + 1e-12
    ]
    if not feasible:
        return {
            "risk_budget": budget,
            "status": "infeasible",
        }
    selected = max(
        feasible,
        key=lambda point: (
            point[
                "mean_expected_throughput_ratio_vs_oracle"
            ],
            -point[
                "expected_scenario_slo_violation_fraction"
            ],
            -point[
                "unsafe_path_probability"
            ],
            -point[
                "expected_confidence_set_key_count"
            ],
            -point[
                "correct_flag_false_positive_rate"
            ],
            point[
                "mistake_flag_sensitivity"
            ],
        ),
    )
    return {
        "risk_budget": budget,
        "status": "feasible",
        **selected,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e62 = load_e62(args.root)
    e63 = load_e63(args.root)
    e64 = load_e64(args.root)
    e65 = load_e65(args.root)
    states, windows, lookup = index_records(e62)
    slos_ms = parse_float_csv(
        args.ttft_slos_ms
    )
    families = [
        value.strip()
        for value in args.families.split(",")
        if value.strip()
    ]
    error_profiles = [
        value.strip()
        for value in args.error_profiles.split(",")
        if value.strip()
    ]
    recalls = parse_float_csv(
        args.recalls
    )
    false_positive_rates = (
        parse_float_csv(
            args.false_positive_rates
        )
    )
    risk_budgets = parse_float_csv(
        args.risk_budgets
    )
    unknown_families = set(families) - set(FAMILIES)
    if unknown_families:
        raise ValueError(
            f"Unknown families: {sorted(unknown_families)}"
        )
    unknown_error_profiles = set(
        error_profiles
    ) - set(ERROR_PROFILES)
    if unknown_error_profiles:
        raise ValueError(
            "Unknown error profiles: "
            f"{sorted(unknown_error_profiles)}"
        )
    for name, values in (
        ("recalls", recalls),
        (
            "false_positive_rates",
            false_positive_rates,
        ),
    ):
        if any(
            value < 0.0 or value > 1.0
            for value in values
        ):
            raise ValueError(
                f"{name} must be in [0, 1]"
            )
    if any(value < 0.0 for value in risk_budgets):
        raise ValueError(
            "Risk budgets must be nonnegative"
        )
    e64_index = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            record["strategy"],
        ): record
        for record in e64["records"]
    }
    e65_index = {
        (
            float(record["ttft_slo_ms"]),
            record["family"],
            record["error_profile"],
            record["confidence_profile"],
        ): record
        for record in e65["records"]
    }
    surfaces = []
    budget_selections = []
    fixed_fpr_budget_selections = []
    deployable_profile_budget_selections = []
    profile_checks = []
    ordering_checks = []
    for slo_ms in slos_ms:
        oracle_windows = oracle_windows_from_e62(
            e62=e62,
            slo_ms=slo_ms,
        )
        for family in families:
            (
                confidence_mapping,
                _,
            ) = confidence_set_mapping(
                family=family,
                slo_ms=slo_ms,
                states=states,
                windows=windows,
                lookup=lookup,
            )
            for error_profile in error_profiles:
                baselines = {
                    strategy: e64_index[
                        (
                            slo_ms,
                            family,
                            error_profile,
                            strategy,
                        )
                    ]
                    for strategy in (
                        "naive",
                        "hamming1_safe",
                    )
                }
                points = []
                point_index = {}
                for recall in recalls:
                    for false_positive_rate in (
                        false_positive_rates
                    ):
                        evaluated = (
                            evaluate_confidence_profile(
                                family=family,
                                slo_ms=slo_ms,
                                error_profile=(
                                    error_profile
                                ),
                                confidence_profile=(
                                    "operating_point"
                                ),
                                mapping=(
                                    confidence_mapping
                                ),
                                states=states,
                                lookup=lookup,
                                oracle_windows=(
                                    oracle_windows
                                ),
                                baselines=baselines,
                                confidence_parameters={
                                    "mistake_flag_sensitivity": (
                                        recall
                                    ),
                                    "correct_flag_false_positive_rate": (
                                        false_positive_rate
                                    ),
                                },
                            )
                        )
                        point = compact_point(
                            evaluated
                        )
                        points.append(point)
                        point_index[
                            (
                                recall,
                                false_positive_rate,
                            )
                        ] = point
                        if (
                            point[
                                "expected_scenario_slo_violation_fraction"
                            ]
                            > point[
                                "unsafe_path_probability"
                            ]
                            + 1e-12
                            or point[
                                "unsafe_path_probability"
                            ]
                            > point[
                                "confidence_miss_probability"
                            ]
                            + 1e-12
                        ):
                            raise RuntimeError(
                                "Safety probability ordering failed"
                            )
                        ordering_checks.append(
                            {
                                "ttft_slo_ms": slo_ms,
                                "family": family,
                                "error_profile": (
                                    error_profile
                                ),
                                "mistake_flag_sensitivity": (
                                    recall
                                ),
                                "correct_flag_false_positive_rate": (
                                    false_positive_rate
                                ),
                                "scenario_minus_unsafe": (
                                    point[
                                        "expected_scenario_slo_violation_fraction"
                                    ]
                                    - point[
                                        "unsafe_path_probability"
                                    ]
                                ),
                                "unsafe_minus_miss": (
                                    point[
                                        "unsafe_path_probability"
                                    ]
                                    - point[
                                        "confidence_miss_probability"
                                    ]
                                ),
                            }
                        )
                for profile, (
                    recall,
                    false_positive_rate,
                ) in E65_PROFILE_POINTS.items():
                    if (
                        recall,
                        false_positive_rate,
                    ) not in point_index:
                        continue
                    expected = e65_index[
                        (
                            slo_ms,
                            family,
                            error_profile,
                            profile,
                        )
                    ]
                    actual = point_index[
                        (
                            recall,
                            false_positive_rate,
                        )
                    ]
                    differences = {
                        metric: abs(
                            float(actual[metric])
                            - float(expected[metric])
                        )
                        for metric in (
                            COMPARISON_METRICS
                            + (
                                "confidence_miss_probability",
                                "any_false_alarm_probability",
                                "expected_confidence_set_key_count",
                            )
                        )
                    }
                    maximum_difference = max(
                        differences.values()
                    )
                    if maximum_difference > 1e-12:
                        raise RuntimeError(
                            "E65 profile reproduction failed"
                        )
                    profile_checks.append(
                        {
                            "ttft_slo_ms": slo_ms,
                            "family": family,
                            "error_profile": (
                                error_profile
                            ),
                            "confidence_profile": (
                                profile
                            ),
                            "differences": (
                                differences
                            ),
                            "maximum_absolute_difference": (
                                maximum_difference
                            ),
                        }
                    )
                frontier = pareto_frontier(points)
                selections = [
                    select_for_budget(
                        points=points,
                        budget=budget,
                    )
                    for budget in risk_budgets
                ]
                fpr_selections = [
                    {
                        "correct_flag_false_positive_rate": (
                            false_positive_rate
                        ),
                        "selections": [
                            select_for_budget(
                                points=[
                                    point
                                    for point in points
                                    if abs(
                                        point[
                                            "correct_flag_false_positive_rate"
                                        ]
                                        - false_positive_rate
                                    )
                                    <= 1e-12
                                ],
                                budget=budget,
                            )
                            for budget in risk_budgets
                        ],
                    }
                    for false_positive_rate
                    in false_positive_rates
                ]
                named_points = [
                    {
                        "confidence_profile": profile,
                        **point_index[
                            (
                                recall,
                                false_positive_rate,
                            )
                        ],
                    }
                    for profile, (
                        recall,
                        false_positive_rate,
                    ) in E65_PROFILE_POINTS.items()
                    if profile != "oracle"
                    if (
                        recall,
                        false_positive_rate,
                    )
                    in point_index
                ]
                named_selections = [
                    select_for_budget(
                        points=named_points,
                        budget=budget,
                    )
                    for budget in risk_budgets
                ]
                constrained_selections = (
                    selections
                    + [
                        selection
                        for item in fpr_selections
                        for selection in item[
                            "selections"
                        ]
                    ]
                    + named_selections
                )
                for selection in constrained_selections:
                    if (
                        selection["status"]
                        == "feasible"
                        and selection[
                            "expected_scenario_slo_violation_fraction"
                        ]
                        > selection["risk_budget"]
                        + 1e-12
                    ):
                        raise RuntimeError(
                            "Risk-budget selector violated budget"
                        )
                surfaces.append(
                    {
                        "ttft_slo_ms": slo_ms,
                        "family": family,
                        "dimensions": list(
                            FAMILIES[family]
                        ),
                        "error_profile": (
                            error_profile
                        ),
                        "grid_point_count": (
                            len(points)
                        ),
                        "points": points,
                        "pareto_frontier": (
                            frontier
                        ),
                        "pareto_point_count": (
                            len(frontier)
                        ),
                    }
                )
                budget_selections.append(
                    {
                        "ttft_slo_ms": slo_ms,
                        "family": family,
                        "error_profile": (
                            error_profile
                        ),
                        "selections": selections,
                    }
                )
                fixed_fpr_budget_selections.append(
                    {
                        "ttft_slo_ms": slo_ms,
                        "family": family,
                        "error_profile": (
                            error_profile
                        ),
                        "false_positive_rate_slices": (
                            fpr_selections
                        ),
                    }
                )
                deployable_profile_budget_selections.append(
                    {
                        "ttft_slo_ms": slo_ms,
                        "family": family,
                        "error_profile": (
                            error_profile
                        ),
                        "profiles": [
                            point[
                                "confidence_profile"
                            ]
                            for point in named_points
                        ],
                        "selections": (
                            named_selections
                        ),
                    }
                )
    expected_surface_count = (
        len(slos_ms)
        * len(families)
        * len(error_profiles)
    )
    expected_grid_point_count = (
        len(recalls)
        * len(false_positive_rates)
    )
    expected_profile_check_count = (
        expected_surface_count
        * len(E65_PROFILE_POINTS)
    )
    if len(surfaces) != expected_surface_count:
        raise RuntimeError(
            "Unexpected E66 surface count"
        )
    if any(
        surface["grid_point_count"]
        != expected_grid_point_count
        for surface in surfaces
    ):
        raise RuntimeError(
            "Unexpected E66 grid size"
        )
    if (
        len(profile_checks)
        != expected_profile_check_count
    ):
        raise RuntimeError(
            "Unexpected E65 profile check count"
        )
    result = {
        "suite": (
            "pacing_confidence_operating_"
            "frontier_analysis"
        ),
        "model": (
            "Sweep confidence mistake recall and false-positive "
            "rate independently, then select the maximum-throughput "
            "operating point satisfying explicit expected scenario "
            "SLO-violation budgets."
        ),
        "source": {
            "slo_pacing": (
                "results/raw/slo_ingress_pacing/"
                "slo_ingress_pacing_full.json"
            ),
            "policy_compression": (
                "results/raw/pacing_policy_compression/"
                "pacing_policy_compression_full.json"
            ),
            "state_uncertainty": (
                "results/raw/pacing_state_uncertainty/"
                "pacing_state_uncertainty_full.json"
            ),
            "confidence_sets": (
                "results/raw/pacing_confidence_sets/"
                "pacing_confidence_sets_full.json"
            ),
        },
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
            for profile in error_profiles
        },
        "mistake_flag_sensitivities": (
            recalls
        ),
        "correct_flag_false_positive_rates": (
            false_positive_rates
        ),
        "scenario_slo_violation_budgets": (
            risk_budgets
        ),
        "surfaces": surfaces,
        "risk_budget_selections": (
            budget_selections
        ),
        "fixed_false_positive_rate_budget_selections": (
            fixed_fpr_budget_selections
        ),
        "deployable_profile_budget_selections": (
            deployable_profile_budget_selections
        ),
        "validation": {
            "e65_profile_checks": (
                profile_checks
            ),
            "e65_profile_maximum_absolute_difference": max(
                item[
                    "maximum_absolute_difference"
                ]
                for item in profile_checks
            ),
            "safety_ordering_checks": (
                ordering_checks
            ),
            "maximum_scenario_minus_unsafe": max(
                item[
                    "scenario_minus_unsafe"
                ]
                for item in ordering_checks
            ),
            "maximum_unsafe_minus_miss": max(
                item["unsafe_minus_miss"]
                for item in ordering_checks
            ),
            "expected_surface_count": (
                expected_surface_count
            ),
            "expected_grid_point_count_per_surface": (
                expected_grid_point_count
            ),
            "expected_total_grid_points": (
                expected_surface_count
                * expected_grid_point_count
            ),
            "expected_e65_profile_check_count": (
                expected_profile_check_count
            ),
            "risk_budget_feasibility": (
                "Every selected operating point "
                "satisfies its expected scenario "
                "SLO-violation budget."
            ),
        },
        "limitations": [
            "Recall and false-positive rate are swept independently; a real detector exposes a coupled ROC curve and may not reach every grid point.",
            "The unconstrained budget selector is an idealized upper bound. Fixed-false-positive slices and named E65 profile selections are the deployable comparisons unless a measured ROC is supplied.",
            "Risk budgets constrain expected scenario violation probability under equally weighted states, not per-tenant, per-hour, or worst-path risk.",
            "The analysis inherits E62-E65's binary state matrix, finite pacing-window grid, and fitted-service simulation.",
            "Confidence errors are conditionally independent and stationary; temporal correlation or distribution shift can invalidate the selected operating point.",
            "The selector has no reject/defer action, online recalibration, switching hysteresis, or multi-replica safe lane.",
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
            "all_5pct,all_10pct,all_20pct"
        ),
    )
    parser.add_argument(
        "--recalls",
        default=",".join(
            str(value)
            for value in DEFAULT_RECALLS
        ),
    )
    parser.add_argument(
        "--false-positive-rates",
        default=",".join(
            str(value)
            for value in (
                DEFAULT_FALSE_POSITIVE_RATES
            )
        ),
    )
    parser.add_argument(
        "--risk-budgets",
        default=",".join(
            str(value)
            for value in DEFAULT_RISK_BUDGETS
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
