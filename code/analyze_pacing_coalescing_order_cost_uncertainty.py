from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_pacing_coalescing_order import (
    ORDER_POLICIES,
    ORDER_TIE_RANK,
)
from analyze_pacing_coalescing_order_robustness import (
    DRIFT_METRICS,
    load_e83,
)


UNCERTAINTY_RADII = (
    0,
    1,
    2,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_e84(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_robustness"
        / "pacing_coalescing_order_robustness_full.json"
    )


def load_e85(
    root: Path,
) -> dict[str, Any]:
    return load_json(
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_compression"
        / "pacing_coalescing_order_compression_full.json"
    )


def uncertainty_costs(
    *,
    costs: list[str],
    center_index: int,
    radius: int,
) -> list[str]:
    start = max(
        0,
        center_index - radius,
    )
    end = min(
        len(costs),
        center_index + radius + 1,
    )
    return costs[start:end]


def expand_segments(
    *,
    costs: list[str],
    segments: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[tuple[str, str], str]:
    mapping = {}
    for metric, metric_segments in (
        segments.items()
    ):
        for segment in metric_segments:
            start = costs.index(
                segment["start_cost"]
            )
            end = costs.index(
                segment["end_cost"]
            )
            for index in range(
                start,
                end + 1,
            ):
                key = (
                    costs[index],
                    metric,
                )
                if key in mapping:
                    raise RuntimeError(
                        "Overlapping compressed segments"
                    )
                mapping[key] = segment[
                    "order"
                ]
    expected = {
        (cost, metric)
        for cost in costs
        for metric in DRIFT_METRICS
    }
    if set(mapping) != expected:
        raise RuntimeError(
            "Compressed mapping is incomplete"
        )
    return mapping


def nested_mapping(
    *,
    costs: list[str],
    mapping: dict[
        tuple[str, str],
        str,
    ],
) -> dict[str, dict[str, str]]:
    return {
        cost: {
            metric: mapping[
                (cost, metric)
            ]
            for metric in DRIFT_METRICS
        }
        for cost in costs
    }


def mapping_segments(
    *,
    costs: list[str],
    mapping: dict[
        tuple[str, str],
        str,
    ],
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for metric in DRIFT_METRICS:
        segments = []
        start = 0
        for index in range(
            1,
            len(costs) + 1,
        ):
            if (
                index == len(costs)
                or mapping[
                    (
                        costs[index],
                        metric,
                    )
                ]
                != mapping[
                    (
                        costs[start],
                        metric,
                    )
                ]
            ):
                segments.append(
                    {
                        "start_cost": (
                            costs[start]
                        ),
                        "end_cost": costs[
                            index - 1
                        ],
                        "order": mapping[
                            (
                                costs[start],
                                metric,
                            )
                        ],
                        "cost_count": (
                            index - start
                        ),
                    }
                )
                start = index
        result[metric] = segments
    return result


def robust_mapping(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    costs: list[str],
    radius: int,
) -> tuple[
    dict[tuple[str, str], str],
    dict[str, Any],
]:
    mapping = {}
    selections = {}
    for estimated_index, estimated_cost in (
        enumerate(costs)
    ):
        actual_costs = uncertainty_costs(
            costs=costs,
            center_index=estimated_index,
            radius=radius,
        )
        selections[estimated_cost] = {
            "actual_cost_uncertainty_set": (
                actual_costs
            ),
            "metrics": {},
        }
        for metric in DRIFT_METRICS:
            keys = {}
            for order in ORDER_POLICIES:
                values = [
                    drift[actual_cost][
                        order
                    ]["by_breach_metric"][
                        metric
                    ]
                    for actual_cost
                    in actual_costs
                ]
                key = (
                    max(
                        value["slower"][
                            "fraction"
                        ]
                        for value in values
                    ),
                    max(
                        value[
                            "positive_delay_requests"
                        ]["maximum"]
                        or 0.0
                        for value in values
                    ),
                    sum(
                        value["slower"][
                            "successes"
                        ]
                        for value in values
                    ),
                    -sum(
                        value["faster"][
                            "successes"
                        ]
                        for value in values
                    ),
                    ORDER_TIE_RANK[order],
                )
                keys[order] = key
            selected = min(
                ORDER_POLICIES,
                key=lambda order: keys[
                    order
                ],
            )
            mapping[
                (
                    estimated_cost,
                    metric,
                )
            ] = selected
            selections[estimated_cost][
                "metrics"
            ][metric] = {
                "order": selected,
                "selection_key": list(
                    keys[selected][:-1]
                ),
            }
    return mapping, selections


def cell_record(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    actual_cost: str,
    estimated_cost: str,
    metric: str,
    order: str,
) -> dict[str, Any]:
    value = drift[actual_cost][order][
        "by_breach_metric"
    ][metric]
    return {
        "actual_cost": actual_cost,
        "estimated_cost": estimated_cost,
        "metric": metric,
        "order": order,
        "comparison_count": value[
            "comparison_count"
        ],
        "faster_successes": value[
            "faster"
        ]["successes"],
        "faster_fraction": value[
            "faster"
        ]["fraction"],
        "regression_successes": value[
            "slower"
        ]["successes"],
        "regression_fraction": value[
            "slower"
        ]["fraction"],
        "maximum_positive_delay_requests": (
            value[
                "positive_delay_requests"
            ]["maximum"]
            or 0.0
        ),
    }


def aggregate_cells(
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    comparison_count = sum(
        cell["comparison_count"]
        for cell in cells
    )
    faster_successes = sum(
        cell["faster_successes"]
        for cell in cells
    )
    regression_successes = sum(
        cell["regression_successes"]
        for cell in cells
    )
    maximum_delay = max(
        cell[
            "maximum_positive_delay_requests"
        ]
        for cell in cells
    )
    maximum_regression = max(
        cells,
        key=lambda cell: (
            cell["regression_fraction"],
            cell[
                "maximum_positive_delay_requests"
            ],
            cell["regression_successes"],
            -cell["faster_successes"],
            -ORDER_TIE_RANK[
                cell["order"]
            ],
        ),
    )
    return {
        "cell_count": len(cells),
        "comparison_count": (
            comparison_count
        ),
        "faster_successes": (
            faster_successes
        ),
        "faster_fraction": (
            faster_successes
            / comparison_count
        ),
        "regression_successes": (
            regression_successes
        ),
        "regression_fraction": (
            regression_successes
            / comparison_count
        ),
        "maximum_positive_delay_requests": (
            maximum_delay
        ),
        "maximum_cell_regression": {
            key: maximum_regression[key]
            for key in (
                "actual_cost",
                "estimated_cost",
                "metric",
                "order",
                "comparison_count",
                "regression_successes",
                "regression_fraction",
                "maximum_positive_delay_requests",
            )
        },
    }


def evaluate_policy(
    *,
    name: str,
    drift: dict[
        str,
        dict[str, Any],
    ],
    costs: list[str],
    radius: int,
    mapping: dict[
        tuple[str, str],
        str,
    ],
) -> dict[str, Any]:
    cells = []
    by_actual = {}
    by_estimated = {}
    for actual_index, actual_cost in (
        enumerate(costs)
    ):
        estimated_costs = (
            uncertainty_costs(
                costs=costs,
                center_index=actual_index,
                radius=radius,
            )
        )
        for estimated_cost in (
            estimated_costs
        ):
            for metric in DRIFT_METRICS:
                order = mapping[
                    (
                        estimated_cost,
                        metric,
                    )
                ]
                cell = cell_record(
                    drift=drift,
                    actual_cost=actual_cost,
                    estimated_cost=(
                        estimated_cost
                    ),
                    metric=metric,
                    order=order,
                )
                cells.append(cell)
                by_actual.setdefault(
                    actual_cost,
                    [],
                ).append(cell)
                by_estimated.setdefault(
                    estimated_cost,
                    [],
                ).append(cell)
    adversarial_cells = []
    for actual_index, actual_cost in (
        enumerate(costs)
    ):
        estimated_costs = (
            uncertainty_costs(
                costs=costs,
                center_index=actual_index,
                radius=radius,
            )
        )
        for metric in DRIFT_METRICS:
            candidates = [
                cell_record(
                    drift=drift,
                    actual_cost=actual_cost,
                    estimated_cost=(
                        estimated_cost
                    ),
                    metric=metric,
                    order=mapping[
                        (
                            estimated_cost,
                            metric,
                        )
                    ],
                )
                for estimated_cost
                in estimated_costs
            ]
            selected = max(
                candidates,
                key=lambda cell: (
                    cell[
                        "regression_fraction"
                    ],
                    cell[
                        "maximum_positive_delay_requests"
                    ],
                    cell[
                        "regression_successes"
                    ],
                    -cell[
                        "faster_successes"
                    ],
                    -costs.index(
                        cell[
                            "estimated_cost"
                        ]
                    ),
                ),
            )
            adversarial_cells.append(
                selected
            )
    return {
        "name": name,
        "uncertainty_radius": radius,
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
        "segments": mapping_segments(
            costs=costs,
            mapping=mapping,
        ),
        "mapping": nested_mapping(
            costs=costs,
            mapping=mapping,
        ),
        "all_admissible_pairs": (
            aggregate_cells(cells)
        ),
        "adversarial_estimate_per_actual_metric": (
            aggregate_cells(
                adversarial_cells
            )
        ),
        "by_actual_cost": {
            cost: aggregate_cells(
                cost_cells
            )
            for cost, cost_cells
            in by_actual.items()
        },
        "by_estimated_cost": {
            cost: aggregate_cells(
                cost_cells
            )
            for cost, cost_cells
            in by_estimated.items()
        },
    }


def relative_reduction(
    *,
    value: float,
    baseline: float,
) -> float:
    if baseline == 0:
        return 0.0
    return 1.0 - value / baseline


def policy_comparison(
    *,
    policy: dict[str, Any],
    fixed: dict[str, Any],
    nominal: dict[str, Any],
) -> dict[str, Any]:
    pair = policy[
        "all_admissible_pairs"
    ]
    fixed_pair = fixed[
        "all_admissible_pairs"
    ]
    nominal_pair = nominal[
        "all_admissible_pairs"
    ]
    adversarial = policy[
        "adversarial_estimate_per_actual_metric"
    ]
    fixed_adversarial = fixed[
        "adversarial_estimate_per_actual_metric"
    ]
    nominal_adversarial = nominal[
        "adversarial_estimate_per_actual_metric"
    ]
    return {
        "pair_regression_reduction_vs_fixed": (
            relative_reduction(
                value=pair[
                    "regression_fraction"
                ],
                baseline=fixed_pair[
                    "regression_fraction"
                ],
            )
        ),
        "pair_regression_reduction_vs_e84_nominal": (
            relative_reduction(
                value=pair[
                    "regression_fraction"
                ],
                baseline=nominal_pair[
                    "regression_fraction"
                ],
            )
        ),
        "adversarial_regression_reduction_vs_fixed": (
            relative_reduction(
                value=adversarial[
                    "regression_fraction"
                ],
                baseline=fixed_adversarial[
                    "regression_fraction"
                ],
            )
        ),
        "adversarial_regression_reduction_vs_e84_nominal": (
            relative_reduction(
                value=adversarial[
                    "regression_fraction"
                ],
                baseline=nominal_adversarial[
                    "regression_fraction"
                ],
            )
        ),
    }


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e84 = load_e84(args.root)
    e85 = load_e85(args.root)
    drift = e83["headline"][
        "drift_by_marginal_fraction_and_order"
    ]
    costs = sorted(
        drift,
        key=float,
    )
    fixed_mapping = {
        (
            cost,
            metric,
        ): "newest_first"
        for cost in costs
        for metric in DRIFT_METRICS
    }
    e84_nested = e84[
        "drift_policy_candidates"
    ]["metric_cost_aware"]["mapping"]
    e84_mapping = {
        (
            cost,
            metric,
        ): e84_nested[cost][metric]
        for cost in costs
        for metric in DRIFT_METRICS
    }
    e85_mapping = expand_segments(
        costs=costs,
        segments=e85["headline"][
            "unconstrained_oracle_recovery_policy"
        ]["segments"],
    )
    radii = {}
    expected_pair_cells = {
        0: 18,
        1: 48,
        2: 72,
    }
    fixed_adversarial_reference = None
    for radius in UNCERTAINTY_RADII:
        uncertainty_mapping, selections = (
            robust_mapping(
                drift=drift,
                costs=costs,
                radius=radius,
            )
        )
        policies = {
            "fixed_newest_first": (
                evaluate_policy(
                    name=(
                        "fixed_newest_first"
                    ),
                    drift=drift,
                    costs=costs,
                    radius=radius,
                    mapping=fixed_mapping,
                )
            ),
            "e84_nominal_metric_cost": (
                evaluate_policy(
                    name=(
                        "e84_nominal_metric_cost"
                    ),
                    drift=drift,
                    costs=costs,
                    radius=radius,
                    mapping=e84_mapping,
                )
            ),
            "e85_compressed_8key": (
                evaluate_policy(
                    name=(
                        "e85_compressed_8key"
                    ),
                    drift=drift,
                    costs=costs,
                    radius=radius,
                    mapping=e85_mapping,
                )
            ),
            "uncertainty_robust": (
                evaluate_policy(
                    name="uncertainty_robust",
                    drift=drift,
                    costs=costs,
                    radius=radius,
                    mapping=(
                        uncertainty_mapping
                    ),
                )
            ),
        }
        fixed = policies[
            "fixed_newest_first"
        ]
        nominal = policies[
            "e84_nominal_metric_cost"
        ]
        for policy in policies.values():
            policy["comparisons"] = (
                policy_comparison(
                    policy=policy,
                    fixed=fixed,
                    nominal=nominal,
                )
            )
            if (
                policy[
                    "all_admissible_pairs"
                ]["cell_count"]
                != expected_pair_cells[
                    radius
                ]
            ):
                raise RuntimeError(
                    "Unexpected admissible pair count"
                )
        fixed_adversarial = fixed[
            "adversarial_estimate_per_actual_metric"
        ]
        if fixed_adversarial_reference is None:
            fixed_adversarial_reference = {
                key: fixed_adversarial[key]
                for key in (
                    "comparison_count",
                    "faster_successes",
                    "regression_successes",
                    "maximum_positive_delay_requests",
                )
            }
        elif fixed_adversarial_reference != {
            key: fixed_adversarial[key]
            for key in (
                "comparison_count",
                "faster_successes",
                "regression_successes",
                "maximum_positive_delay_requests",
            )
        }:
            raise RuntimeError(
                "Fixed policy changed under "
                "cost-estimate uncertainty"
            )
        radii[str(radius)] = {
            "radius": radius,
            "estimated_cost_uncertainty_sets": {
                cost: uncertainty_costs(
                    costs=costs,
                    center_index=index,
                    radius=radius,
                )
                for index, cost
                in enumerate(costs)
            },
            "uncertainty_robust_selections": (
                selections
            ),
            "policies": policies,
        }
    if (
        radii["0"]["policies"][
            "uncertainty_robust"
        ]["mapping"]
        != nested_mapping(
            costs=costs,
            mapping=e84_mapping,
        )
    ):
        raise RuntimeError(
            "Radius-zero robust mapping did not "
            "reproduce E84"
        )
    radius_zero_e84 = radii["0"][
        "policies"
    ]["e84_nominal_metric_cost"]
    radius_zero_e85 = radii["0"][
        "policies"
    ]["e85_compressed_8key"]
    for scope in (
        "all_admissible_pairs",
        "adversarial_estimate_per_actual_metric",
    ):
        for key in (
            "comparison_count",
            "faster_successes",
            "regression_successes",
            "maximum_positive_delay_requests",
        ):
            if (
                radius_zero_e84[scope][key]
                != radius_zero_e85[
                    scope
                ][key]
            ):
                raise RuntimeError(
                    "E85 eight-key policy did not "
                    "reproduce E84 outcomes"
                )
    headline = {}
    for radius in (
        1,
        2,
    ):
        values = radii[str(radius)][
            "policies"
        ]
        headline[str(radius)] = {
            name: {
                "all_pairs": policy[
                    "all_admissible_pairs"
                ],
                "adversarial": policy[
                    "adversarial_estimate_per_actual_metric"
                ],
                "comparisons": policy[
                    "comparisons"
                ],
            }
            for name, policy
            in values.items()
        }
    result = {
        "suite": (
            "pacing_coalescing_order_cost_uncertainty"
        ),
        "model": (
            "Treat marginal snapshot cost as an "
            "ordinal estimate on E83's six-point "
            "grid. Radius r admits actual and "
            "estimated cost indices at distance "
            "at most r. Compare fixed, E84 "
            "nominal, E85 compressed, and "
            "uncertainty-aware minimax mappings."
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
        },
        "robust_selection_objective": [
            "minimum maximum cell regression fraction over the uncertainty set",
            "minimum maximum positive delay over the uncertainty set",
            "minimum aggregate regression successes over the uncertainty set",
            "maximum aggregate faster-action successes over the uncertainty set",
            "chronological deterministic order rank",
        ],
        "costs": costs,
        "uncertainty_radii": list(
            UNCERTAINTY_RADII
        ),
        "radii": radii,
        "headline": headline,
        "validation": {
            "radius_zero_e84_mapping_reproduced": (
                True
            ),
            "radius_zero_e85_outcomes_reproduce_e84": (
                True
            ),
            "fixed_policy_adversarial_invariant": (
                True
            ),
            "expected_pair_cell_counts": {
                str(radius): count
                for radius, count
                in expected_pair_cells.items()
            },
            "all_uncertainty_sets_contain_estimated_cost": (
                all(
                    cost
                    in radii[str(radius)][
                        "estimated_cost_uncertainty_sets"
                    ][cost]
                    for radius
                    in UNCERTAINTY_RADII
                    for cost in costs
                )
            ),
            "all_uncertainty_sets_respect_radius": (
                all(
                    abs(
                        costs.index(actual)
                        - costs.index(estimated)
                    )
                    <= radius
                    for radius
                    in UNCERTAINTY_RADII
                    for estimated in costs
                    for actual in radii[
                        str(radius)
                    ][
                        "estimated_cost_uncertainty_sets"
                    ][estimated]
                )
            ),
        },
        "limitations": [
            "Cost uncertainty is ordinal on E83's six tested points; it is not a continuous calibration model.",
            "All admissible-pair aggregation weights every replayed comparison, while the adversarial envelope chooses one estimate per actual cost and metric.",
            "The robust mapping minimizes worst cell regression before delay and aggregate risk; another priority order can select different mappings.",
            "Observed maximum delay is not a future probabilistic tail guarantee.",
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
