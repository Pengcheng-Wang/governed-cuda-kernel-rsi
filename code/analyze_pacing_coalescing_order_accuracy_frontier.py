from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from analyze_pacing_coalescing_order_cost_uncertainty import (
    DRIFT_METRICS,
    expand_segments,
    load_e83,
    load_e84,
    load_e85,
    mapping_segments,
)


KERNELS = {
    "symmetric_radius_1": {
        "offset_weights": {
            -1: 0.5,
            1: 0.5,
        },
        "description": (
            "Non-exact mass is split equally "
            "between adjacent lower and higher "
            "cost cells."
        ),
    },
    "symmetric_radius_2": {
        "offset_weights": {
            -2: 0.25,
            -1: 0.25,
            1: 0.25,
            2: 0.25,
        },
        "description": (
            "Non-exact mass is split equally "
            "across one- and two-cell errors "
            "in either direction."
        ),
    },
    "underestimate_radius_2": {
        "offset_weights": {
            -2: 0.5,
            -1: 0.5,
        },
        "description": (
            "Non-exact mass is split equally "
            "between one- and two-cell "
            "underestimation."
        ),
    },
    "overestimate_radius_2": {
        "offset_weights": {
            1: 0.5,
            2: 0.5,
        },
        "description": (
            "Non-exact mass is split equally "
            "between one- and two-cell "
            "overestimation."
        ),
    },
}

REPRESENTATIVE_ACCURACIES = (
    0.0,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    1.0,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_e86(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_cost_uncertainty"
        / "pacing_coalescing_order_cost_uncertainty_full.json"
    )


def load_e87(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_uncertainty_caps"
        / "pacing_coalescing_order_uncertainty_caps_full.json"
    )


def flatten_mapping(
    *,
    costs: list[str],
    nested: dict[
        str,
        dict[str, str],
    ],
) -> dict[tuple[str, str], str]:
    return {
        (cost, metric): nested[cost][
            metric
        ]
        for cost in costs
        for metric in DRIFT_METRICS
    }


def policy_inventory(
    *,
    root: Path,
    costs: list[str],
) -> dict[str, dict[str, Any]]:
    e84 = load_e84(root)
    e85 = load_e85(root)
    e86 = load_e86(root)
    e87 = load_e87(root)
    mappings = {
        "fixed_newest_first": {
            (
                cost,
                metric,
            ): "newest_first"
            for cost in costs
            for metric in DRIFT_METRICS
        },
        "e84_nominal_metric_cost": (
            flatten_mapping(
                costs=costs,
                nested=e84[
                    "drift_policy_candidates"
                ][
                    "metric_cost_aware"
                ]["mapping"],
            )
        ),
        "e85_compressed_8key": (
            expand_segments(
                costs=costs,
                segments=e85["headline"][
                    "unconstrained_oracle_recovery_policy"
                ]["segments"],
            )
        ),
        "radius_1_robust": (
            flatten_mapping(
                costs=costs,
                nested=e86["radii"]["1"][
                    "policies"
                ]["uncertainty_robust"][
                    "mapping"
                ],
            )
        ),
        "radius_2_robust": (
            flatten_mapping(
                costs=costs,
                nested=e86["radii"]["2"][
                    "policies"
                ]["uncertainty_robust"][
                    "mapping"
                ],
            )
        ),
        "radius_1_cap_10500": (
            flatten_mapping(
                costs=costs,
                nested=e87["radii"]["1"][
                    "policies_by_delay_cap"
                ]["10500"]["mapping"],
            )
        ),
        "radius_1_cap_19500": (
            flatten_mapping(
                costs=costs,
                nested=e87["radii"]["1"][
                    "policies_by_delay_cap"
                ]["19500"]["mapping"],
            )
        ),
        "radius_2_cap_10500": (
            flatten_mapping(
                costs=costs,
                nested=e87["radii"]["2"][
                    "policies_by_delay_cap"
                ]["10500"]["mapping"],
            )
        ),
        "radius_2_cap_19500": (
            flatten_mapping(
                costs=costs,
                nested=e87["radii"]["2"][
                    "policies_by_delay_cap"
                ]["19500"]["mapping"],
            )
        ),
    }
    return {
        name: {
            "mapping": mapping,
            "segments": mapping_segments(
                costs=costs,
                mapping=mapping,
            ),
            "configuration_segment_count": (
                sum(
                    len(metric_segments)
                    for metric_segments
                    in mapping_segments(
                        costs=costs,
                        mapping=mapping,
                    ).values()
                )
            ),
        }
        for name, mapping in mappings.items()
    }


def estimate_distribution(
    *,
    costs: list[str],
    actual_index: int,
    accuracy: float,
    offset_weights: dict[int, float],
) -> dict[str, float]:
    if not (
        0.0 <= accuracy <= 1.0
    ):
        raise ValueError(
            "accuracy must be in [0, 1]"
        )
    distribution = {
        costs[actual_index]: accuracy
    }
    error_mass = 1.0 - accuracy
    for offset, weight in (
        offset_weights.items()
    ):
        estimated_index = max(
            0,
            min(
                len(costs) - 1,
                actual_index + offset,
            ),
        )
        estimated_cost = costs[
            estimated_index
        ]
        distribution[estimated_cost] = (
            distribution.get(
                estimated_cost,
                0.0,
            )
            + error_mass * weight
        )
    total = sum(distribution.values())
    if abs(total - 1.0) > 1e-12:
        raise RuntimeError(
            "Estimator distribution does "
            "not normalize"
        )
    return {
        cost: probability
        for cost, probability
        in distribution.items()
        if probability > 0.0
    }


def evaluate_policy(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    costs: list[str],
    mapping: dict[
        tuple[str, str],
        str,
    ],
    accuracy: float,
    offset_weights: dict[int, float],
) -> dict[str, Any]:
    comparison_count = 0.0
    regression_successes = 0.0
    faster_successes = 0.0
    total_positive_delay = 0.0
    total_request_savings = 0.0
    maximum_delay = 0.0
    support_pair_count = 0
    for actual_index, actual_cost in (
        enumerate(costs)
    ):
        distribution = estimate_distribution(
            costs=costs,
            actual_index=actual_index,
            accuracy=accuracy,
            offset_weights=offset_weights,
        )
        for estimated_cost, probability in (
            distribution.items()
        ):
            support_pair_count += 1
            for metric in DRIFT_METRICS:
                order = mapping[
                    (
                        estimated_cost,
                        metric,
                    )
                ]
                value = drift[
                    actual_cost
                ][order][
                    "by_breach_metric"
                ][metric]
                comparison_count += (
                    probability
                    * value[
                        "comparison_count"
                    ]
                )
                regression_successes += (
                    probability
                    * value["slower"][
                        "successes"
                    ]
                )
                faster_successes += (
                    probability
                    * value["faster"][
                        "successes"
                    ]
                )
                total_positive_delay += (
                    probability
                    * value[
                        "positive_delay_requests"
                    ]["count"]
                    * (
                        value[
                            "positive_delay_requests"
                        ]["mean"]
                        or 0.0
                    )
                )
                total_request_savings += (
                    probability
                    * value[
                        "request_savings"
                    ]["count"]
                    * (
                        value[
                            "request_savings"
                        ]["mean"]
                        or 0.0
                    )
                )
                maximum_delay = max(
                    maximum_delay,
                    value[
                        "positive_delay_requests"
                    ]["maximum"]
                    or 0.0,
                )
    return {
        "accuracy": accuracy,
        "support_pair_count": (
            support_pair_count
        ),
        "comparison_count": (
            comparison_count
        ),
        "regression_successes": (
            regression_successes
        ),
        "regression_fraction": (
            regression_successes
            / comparison_count
        ),
        "faster_successes": (
            faster_successes
        ),
        "faster_fraction": (
            faster_successes
            / comparison_count
        ),
        "positive_delay_burden_requests_per_comparison": (
            total_positive_delay
            / comparison_count
        ),
        "conditional_positive_delay_mean_requests": (
            total_positive_delay
            / regression_successes
            if regression_successes > 0
            else 0.0
        ),
        "request_savings_per_comparison": (
            total_request_savings
            / comparison_count
        ),
        "conditional_request_savings_mean": (
            total_request_savings
            / faster_successes
            if faster_successes > 0
            else 0.0
        ),
        "maximum_positive_delay_support_requests": (
            maximum_delay
        ),
    }


def policy_key(
    *,
    name: str,
    values: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    return (
        values["regression_fraction"],
        values[
            "positive_delay_burden_requests_per_comparison"
        ],
        inventory[name][
            "configuration_segment_count"
        ],
        -values["faster_fraction"],
        name,
    )


def winner(
    *,
    values: dict[
        str,
        dict[str, Any],
    ],
    inventory: dict[str, dict[str, Any]],
) -> str:
    return min(
        values,
        key=lambda name: policy_key(
            name=name,
            values=values[name],
            inventory=inventory,
        ),
    )


def affine_intersections(
    *,
    endpoints: dict[
        str,
        dict[str, dict[str, Any]],
    ],
) -> list[float]:
    points = {
        0.0,
        1.0,
    }
    for left, right in (
        itertools.combinations(
            endpoints,
            2,
        )
    ):
        left_zero = endpoints[left][
            "accuracy_0"
        ]["regression_successes"]
        left_one = endpoints[left][
            "accuracy_1"
        ]["regression_successes"]
        right_zero = endpoints[right][
            "accuracy_0"
        ]["regression_successes"]
        right_one = endpoints[right][
            "accuracy_1"
        ]["regression_successes"]
        intercept = (
            left_zero - right_zero
        )
        slope = (
            (left_one - left_zero)
            - (right_one - right_zero)
        )
        if abs(slope) < 1e-12:
            continue
        point = -intercept / slope
        if (
            0.0 < point < 1.0
        ):
            points.add(round(point, 14))
    return sorted(points)


def exact_winner_regions(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    costs: list[str],
    inventory: dict[str, dict[str, Any]],
    offset_weights: dict[int, float],
    endpoints: dict[
        str,
        dict[str, dict[str, Any]],
    ],
) -> dict[str, Any]:
    points = affine_intersections(
        endpoints=endpoints
    )
    elementary = []
    for start, end in zip(
        points[:-1],
        points[1:],
    ):
        midpoint = (
            start + end
        ) / 2.0
        values = {
            name: evaluate_policy(
                drift=drift,
                costs=costs,
                mapping=policy[
                    "mapping"
                ],
                accuracy=midpoint,
                offset_weights=(
                    offset_weights
                ),
            )
            for name, policy
            in inventory.items()
        }
        elementary.append(
            {
                "start_accuracy": start,
                "end_accuracy": end,
                "winner": winner(
                    values=values,
                    inventory=inventory,
                ),
                "midpoint_accuracy": (
                    midpoint
                ),
            }
        )
    merged = []
    for region in elementary:
        if (
            merged
            and merged[-1]["winner"]
            == region["winner"]
            and abs(
                merged[-1][
                    "end_accuracy"
                ]
                - region[
                    "start_accuracy"
                ]
            )
            < 1e-12
        ):
            merged[-1][
                "end_accuracy"
            ] = region["end_accuracy"]
        else:
            merged.append(
                dict(region)
            )
    for region in merged:
        midpoint = (
            region["start_accuracy"]
            + region["end_accuracy"]
        ) / 2.0
        values = {
            name: evaluate_policy(
                drift=drift,
                costs=costs,
                mapping=policy[
                    "mapping"
                ],
                accuracy=midpoint,
                offset_weights=(
                    offset_weights
                ),
            )
            for name, policy
            in inventory.items()
        }
        region[
            "midpoint_accuracy"
        ] = midpoint
        region[
            "winner_metrics_at_midpoint"
        ] = values[region["winner"]]
    point_winners = {}
    for point in points:
        values = {
            name: evaluate_policy(
                drift=drift,
                costs=costs,
                mapping=policy[
                    "mapping"
                ],
                accuracy=point,
                offset_weights=(
                    offset_weights
                ),
            )
            for name, policy
            in inventory.items()
        }
        point_winners[
            f"{point:.14g}"
        ] = {
            "winner": winner(
                values=values,
                inventory=inventory,
            ),
            "winner_metrics": values[
                winner(
                    values=values,
                    inventory=inventory,
                )
            ],
        }
    return {
        "candidate_crossover_points": (
            points
        ),
        "open_interval_winners": (
            merged
        ),
        "point_winners": point_winners,
    }


def covered_winner(
    *,
    accuracy: float,
    regions: dict[str, Any],
) -> str:
    for point, values in regions[
        "point_winners"
    ].items():
        if abs(
            float(point) - accuracy
        ) < 1e-12:
            return values["winner"]
    for region in regions[
        "open_interval_winners"
    ]:
        if (
            region["start_accuracy"]
            < accuracy
            < region["end_accuracy"]
        ):
            return region["winner"]
    raise RuntimeError(
        "Accuracy is not covered by exact "
        "winner regions"
    )


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
    expected_comparison_count = sum(
        drift[cost]["oldest_first"][
            "by_breach_metric"
        ][metric][
            "comparison_count"
        ]
        for cost in costs
        for metric in DRIFT_METRICS
    )
    kernels = {}
    accuracy_grid = [
        index / 100.0
        for index in range(101)
    ]
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        offset_weights = kernel[
            "offset_weights"
        ]
        if (
            abs(
                sum(
                    offset_weights.values()
                )
                - 1.0
            )
            > 1e-12
        ):
            raise RuntimeError(
                "Kernel offset weights do "
                "not normalize"
            )
        endpoints = {}
        for policy_name, policy in (
            inventory.items()
        ):
            endpoints[policy_name] = {
                "accuracy_0": (
                    evaluate_policy(
                        drift=drift,
                        costs=costs,
                        mapping=policy[
                            "mapping"
                        ],
                        accuracy=0.0,
                        offset_weights=(
                            offset_weights
                        ),
                    )
                ),
                "accuracy_1": (
                    evaluate_policy(
                        drift=drift,
                        costs=costs,
                        mapping=policy[
                            "mapping"
                        ],
                        accuracy=1.0,
                        offset_weights=(
                            offset_weights
                        ),
                    )
                ),
            }
            for endpoint in (
                "accuracy_0",
                "accuracy_1",
            ):
                if (
                    abs(
                        endpoints[
                            policy_name
                        ][endpoint][
                            "comparison_count"
                        ]
                        - expected_comparison_count
                    )
                    > 1e-6
                ):
                    raise RuntimeError(
                        "Expected comparison count "
                        "changed"
                    )
        exact_regions = exact_winner_regions(
            drift=drift,
            costs=costs,
            inventory=inventory,
            offset_weights=offset_weights,
            endpoints=endpoints,
        )
        grid_winners = []
        for accuracy in accuracy_grid:
            values = {
                name: evaluate_policy(
                    drift=drift,
                    costs=costs,
                    mapping=policy[
                        "mapping"
                    ],
                    accuracy=accuracy,
                    offset_weights=(
                        offset_weights
                    ),
                )
                for name, policy
                in inventory.items()
            }
            grid_winners.append(
                {
                    "accuracy": accuracy,
                    "winner": winner(
                        values=values,
                        inventory=inventory,
                    ),
                }
            )
        for row in grid_winners:
            if (
                covered_winner(
                    accuracy=row["accuracy"],
                    regions=exact_regions,
                )
                != row["winner"]
            ):
                raise RuntimeError(
                    "Exact crossover regions did "
                    "not reproduce grid winner"
                )
        representatives = {}
        for accuracy in (
            REPRESENTATIVE_ACCURACIES
        ):
            values = {
                name: evaluate_policy(
                    drift=drift,
                    costs=costs,
                    mapping=policy[
                        "mapping"
                    ],
                    accuracy=accuracy,
                    offset_weights=(
                        offset_weights
                    ),
                )
                for name, policy
                in inventory.items()
            }
            representatives[
                f"{accuracy:g}"
            ] = {
                "winner": winner(
                    values=values,
                    inventory=inventory,
                ),
                "policies": values,
            }
        kernels[kernel_name] = {
            "description": kernel[
                "description"
            ],
            "offset_weights": {
                str(offset): weight
                for offset, weight
                in offset_weights.items()
            },
            "policy_affine_endpoints": (
                endpoints
            ),
            "exact_winner_regions": (
                exact_regions
            ),
            "accuracy_grid_winners": (
                grid_winners
            ),
            "representative_accuracies": (
                representatives
            ),
        }
    headline = {
        kernel_name: {
            "winner_regions": [
                {
                    key: region[key]
                    for key in (
                        "start_accuracy",
                        "end_accuracy",
                        "winner",
                    )
                }
                for region in values[
                    "exact_winner_regions"
                ]["open_interval_winners"]
            ],
            "boundary_winners": {
                point: point_values[
                    "winner"
                ]
                for point, point_values
                in values[
                    "exact_winner_regions"
                ]["point_winners"].items()
                if (
                    float(point)
                    not in (0.0, 1.0)
                    and any(
                        abs(
                            float(point)
                            - region[
                                "end_accuracy"
                            ]
                        )
                        < 1e-12
                        for region in values[
                            "exact_winner_regions"
                        ][
                            "open_interval_winners"
                        ][:-1]
                    )
                )
            },
            "representative_winners": {
                accuracy: values[
                    "representative_accuracies"
                ][accuracy]["winner"]
                for accuracy in (
                    "0",
                    "0.5",
                    "0.75",
                    "0.9",
                    "1",
                )
            },
        }
        for kernel_name, values
        in kernels.items()
    }
    exact_outcomes = {
        policy_name: kernels[
            "symmetric_radius_1"
        ]["policy_affine_endpoints"][
            policy_name
        ]["accuracy_1"]
        for policy_name in inventory
    }
    if (
        exact_outcomes[
            "e84_nominal_metric_cost"
        ]["regression_successes"]
        != exact_outcomes[
            "e85_compressed_8key"
        ]["regression_successes"]
    ):
        raise RuntimeError(
            "E84/E85 exact outcomes differ"
        )
    fixed_endpoints = [
        kernels[kernel_name][
            "policy_affine_endpoints"
        ]["fixed_newest_first"]
        for kernel_name in kernels
    ]
    for values in fixed_endpoints[1:]:
        for endpoint in (
            "accuracy_0",
            "accuracy_1",
        ):
            for key in (
                "comparison_count",
                "regression_successes",
                "faster_successes",
                "maximum_positive_delay_support_requests",
            ):
                if (
                    values[endpoint][key]
                    != fixed_endpoints[0][
                        endpoint
                    ][key]
                ):
                    raise RuntimeError(
                        "Fixed policy changed across "
                        "estimator kernels"
                    )
    result = {
        "suite": (
            "pacing_coalescing_order_accuracy_frontier"
        ),
        "model": (
            "Sweep exact cost-classification "
            "accuracy from 0 to 1 under "
            "boundary-clamped symmetric and "
            "one-sided ordinal error kernels. "
            "Expected risk weights each actual "
            "cost and breach metric by its full "
            "E83 comparison count."
        ),
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            ),
            "robust_policy_audit": (
                "results/raw/"
                "pacing_coalescing_order_robustness/"
                "pacing_coalescing_order_robustness_full.json"
            ),
            "compressed_policy_audit": (
                "results/raw/"
                "pacing_coalescing_order_compression/"
                "pacing_coalescing_order_compression_full.json"
            ),
            "cost_uncertainty_audit": (
                "results/raw/"
                "pacing_coalescing_order_cost_uncertainty/"
                "pacing_coalescing_order_cost_uncertainty_full.json"
            ),
            "joint_uncertainty_cap_audit": (
                "results/raw/"
                "pacing_coalescing_order_uncertainty_caps/"
                "pacing_coalescing_order_uncertainty_caps_full.json"
            ),
        },
        "selection_objective": [
            "minimum expected regression fraction",
            "minimum expected positive-delay burden per comparison",
            "minimum contiguous configuration segments",
            "maximum expected faster-action fraction",
            "deterministic policy name",
        ],
        "costs": costs,
        "accuracy_grid": accuracy_grid,
        "policy_inventory": {
            name: {
                "configuration_segment_count": (
                    policy[
                        "configuration_segment_count"
                    ]
                ),
                "segments": policy[
                    "segments"
                ],
            }
            for name, policy
            in inventory.items()
        },
        "kernels": kernels,
        "headline": headline,
        "validation": {
            "kernel_count": len(
                kernels
            ),
            "policy_count": len(
                inventory
            ),
            "accuracy_grid_count": len(
                accuracy_grid
            ),
            "expected_comparison_count": (
                expected_comparison_count
            ),
            "all_kernel_probabilities_normalized": (
                True
            ),
            "accuracy_one_exact_cost_support": (
                True
            ),
            "e84_e85_exact_regression_reproduced": (
                True
            ),
            "fixed_policy_kernel_invariant": (
                True
            ),
            "exact_crossover_coverage": True,
        },
        "limitations": [
            "The estimator kernels are sensitivity models, not calibrated production error distributions.",
            "Actual cost cells receive equal E83 comparison weight; deployment prevalence can change expected winners.",
            "Boundary clamping accumulates impossible under/over-estimation mass at the edge cost cell.",
            "Expected regression is the primary objective; another utility or worst-cell constraint can select a different policy.",
            "The audit reuses compact E83 path summaries and is not new GPU evidence.",
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
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
