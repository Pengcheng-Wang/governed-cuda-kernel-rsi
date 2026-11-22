from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_pacing_coalescing_order import (
    ORDER_POLICIES,
    ORDER_TIE_RANK,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    DRIFT_METRICS,
    evaluate_policy,
    load_e83,
    uncertainty_costs,
)


UNCERTAINTY_RADII = (
    1,
    2,
)


def load_e86(
    root: Path,
) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_cost_uncertainty"
        / "pacing_coalescing_order_cost_uncertainty_full.json"
    )
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def order_values(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    actual_costs: list[str],
    metric: str,
    order: str,
) -> list[dict[str, Any]]:
    return [
        drift[actual_cost][order][
            "by_breach_metric"
        ][metric]
        for actual_cost in actual_costs
    ]


def order_delay(
    values: list[dict[str, Any]],
) -> float:
    return max(
        value[
            "positive_delay_requests"
        ]["maximum"]
        or 0.0
        for value in values
    )


def selection_key(
    *,
    values: list[dict[str, Any]],
    order: str,
) -> tuple[Any, ...]:
    return (
        max(
            value["slower"][
                "fraction"
            ]
            for value in values
        ),
        order_delay(values),
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


def derive_caps(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    costs: list[str],
    radius: int,
) -> tuple[float, list[float]]:
    possible_delays = set()
    minimum_feasible = 0.0
    for estimated_index, _ in enumerate(
        costs
    ):
        actual_costs = uncertainty_costs(
            costs=costs,
            center_index=estimated_index,
            radius=radius,
        )
        for metric in DRIFT_METRICS:
            delays = []
            for order in ORDER_POLICIES:
                delay = order_delay(
                    order_values(
                        drift=drift,
                        actual_costs=actual_costs,
                        metric=metric,
                        order=order,
                    )
                )
                delays.append(delay)
                possible_delays.add(delay)
            minimum_feasible = max(
                minimum_feasible,
                min(delays),
            )
    caps = sorted(
        delay
        for delay in possible_delays
        if delay >= minimum_feasible
    )
    return minimum_feasible, caps


def capped_mapping(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    costs: list[str],
    radius: int,
    delay_cap: float,
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
            candidates = {}
            for order in ORDER_POLICIES:
                values = order_values(
                    drift=drift,
                    actual_costs=actual_costs,
                    metric=metric,
                    order=order,
                )
                delay = order_delay(values)
                if delay <= delay_cap:
                    candidates[order] = (
                        selection_key(
                            values=values,
                            order=order,
                        )
                    )
            if not candidates:
                raise RuntimeError(
                    "No order satisfies a globally "
                    "feasible delay cap"
                )
            selected = min(
                candidates,
                key=lambda order: candidates[
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
                    candidates[selected][:-1]
                ),
                "feasible_orders": (
                    sorted(
                        candidates,
                        key=lambda order: (
                            ORDER_TIE_RANK[
                                order
                            ]
                        ),
                    )
                ),
            }
    return mapping, selections


def relative_reduction(
    *,
    value: float,
    baseline: float,
) -> float:
    if baseline == 0:
        return 0.0
    return 1.0 - value / baseline


def compact_scope(
    scope: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: scope[key]
        for key in (
            "cell_count",
            "comparison_count",
            "faster_successes",
            "faster_fraction",
            "regression_successes",
            "regression_fraction",
            "maximum_positive_delay_requests",
            "maximum_cell_regression",
        )
    }


def compact_policy(
    *,
    policy: dict[str, Any],
    delay_cap: float,
    fixed: dict[str, Any],
) -> dict[str, Any]:
    pair = policy[
        "all_admissible_pairs"
    ]
    adversarial = policy[
        "adversarial_estimate_per_actual_metric"
    ]
    fixed_pair = fixed[
        "all_admissible_pairs"
    ]
    fixed_adversarial = fixed[
        "adversarial_estimate_per_actual_metric"
    ]
    return {
        "configured_delay_cap_requests": (
            delay_cap
        ),
        "configuration_segment_count": (
            policy[
                "configuration_segment_count"
            ]
        ),
        "segments": policy["segments"],
        "mapping": policy["mapping"],
        "all_admissible_pairs": (
            compact_scope(pair)
        ),
        "adversarial_estimate_per_actual_metric": (
            compact_scope(adversarial)
        ),
        "comparisons_vs_fixed": {
            "all_pair_regression_reduction": (
                relative_reduction(
                    value=pair[
                        "regression_fraction"
                    ],
                    baseline=fixed_pair[
                        "regression_fraction"
                    ],
                )
            ),
            "adversarial_regression_reduction": (
                relative_reduction(
                    value=adversarial[
                        "regression_fraction"
                    ],
                    baseline=fixed_adversarial[
                        "regression_fraction"
                    ],
                )
            ),
            "worst_cell_regression_change": (
                adversarial[
                    "maximum_cell_regression"
                ]["regression_fraction"]
                / fixed_adversarial[
                    "maximum_cell_regression"
                ]["regression_fraction"]
                - 1.0
            ),
            "maximum_delay_reduction": (
                relative_reduction(
                    value=adversarial[
                        "maximum_positive_delay_requests"
                    ],
                    baseline=fixed_adversarial[
                        "maximum_positive_delay_requests"
                    ],
                )
            ),
        },
    }


def outcomes_equal(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
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
                left[scope][key]
                != right[scope][key]
            ):
                return False
        if (
            left[scope][
                "maximum_cell_regression"
            ]["regression_fraction"]
            != right[scope][
                "maximum_cell_regression"
            ]["regression_fraction"]
        ):
            return False
    return True


def pareto_front(
    policies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    front = []
    for policy in policies:
        adversarial = policy[
            "adversarial_estimate_per_actual_metric"
        ]
        pair = policy[
            "all_admissible_pairs"
        ]
        objectives = (
            adversarial[
                "maximum_cell_regression"
            ]["regression_fraction"],
            adversarial[
                "regression_fraction"
            ],
            pair["regression_fraction"],
            adversarial[
                "maximum_positive_delay_requests"
            ],
            policy[
                "configuration_segment_count"
            ],
            -pair["faster_fraction"],
        )
        dominated = False
        for other in policies:
            if other is policy:
                continue
            other_adversarial = other[
                "adversarial_estimate_per_actual_metric"
            ]
            other_pair = other[
                "all_admissible_pairs"
            ]
            other_objectives = (
                other_adversarial[
                    "maximum_cell_regression"
                ]["regression_fraction"],
                other_adversarial[
                    "regression_fraction"
                ],
                other_pair[
                    "regression_fraction"
                ],
                other_adversarial[
                    "maximum_positive_delay_requests"
                ],
                other[
                    "configuration_segment_count"
                ],
                -other_pair[
                    "faster_fraction"
                ],
            )
            if (
                all(
                    left <= right
                    for left, right
                    in zip(
                        other_objectives,
                        objectives,
                    )
                )
                and any(
                    left < right
                    for left, right
                    in zip(
                        other_objectives,
                        objectives,
                    )
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(policy)
    return sorted(
        front,
        key=lambda policy: (
            policy[
                "adversarial_estimate_per_actual_metric"
            ][
                "maximum_positive_delay_requests"
            ],
            policy[
                "adversarial_estimate_per_actual_metric"
            ][
                "maximum_cell_regression"
            ]["regression_fraction"],
            policy[
                "adversarial_estimate_per_actual_metric"
            ]["regression_fraction"],
            policy[
                "all_admissible_pairs"
            ]["regression_fraction"],
        ),
    )


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e86 = load_e86(args.root)
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
    radii = {}
    for radius in UNCERTAINTY_RADII:
        fixed = evaluate_policy(
            name="fixed_newest_first",
            drift=drift,
            costs=costs,
            radius=radius,
            mapping=fixed_mapping,
        )
        minimum_cap, caps = derive_caps(
            drift=drift,
            costs=costs,
            radius=radius,
        )
        policies = {}
        previous_worst_cell = None
        for cap in caps:
            mapping, selections = (
                capped_mapping(
                    drift=drift,
                    costs=costs,
                    radius=radius,
                    delay_cap=cap,
                )
            )
            evaluated = evaluate_policy(
                name=(
                    "uncertainty_delay_capped"
                ),
                drift=drift,
                costs=costs,
                radius=radius,
                mapping=mapping,
            )
            if (
                evaluated[
                    "all_admissible_pairs"
                ][
                    "maximum_positive_delay_requests"
                ]
                > cap
                or evaluated[
                    "adversarial_estimate_per_actual_metric"
                ][
                    "maximum_positive_delay_requests"
                ]
                > cap
            ):
                raise RuntimeError(
                    "Selected policy violates cap"
                )
            worst_cell = evaluated[
                "adversarial_estimate_per_actual_metric"
            ][
                "maximum_cell_regression"
            ]["regression_fraction"]
            if (
                previous_worst_cell
                is not None
                and worst_cell
                > previous_worst_cell
                + 1e-15
            ):
                raise RuntimeError(
                    "Worst-cell regression worsened "
                    "under a looser cap"
                )
            previous_worst_cell = worst_cell
            compact = compact_policy(
                policy=evaluated,
                delay_cap=cap,
                fixed=fixed,
            )
            compact[
                "selections"
            ] = selections
            policies[f"{cap:g}"] = compact
        maximum_cap_key = max(
            policies,
            key=float,
        )
        e86_uncapped = e86["radii"][
            str(radius)
        ]["policies"][
            "uncertainty_robust"
        ]
        if not outcomes_equal(
            left={
                "all_admissible_pairs": (
                    policies[
                        maximum_cap_key
                    ][
                        "all_admissible_pairs"
                    ]
                ),
                "adversarial_estimate_per_actual_metric": (
                    policies[
                        maximum_cap_key
                    ][
                        "adversarial_estimate_per_actual_metric"
                    ]
                ),
            },
            right=e86_uncapped,
        ):
            raise RuntimeError(
                "Maximum cap did not reproduce E86"
            )
        policy_values = list(
            policies.values()
        )
        first_pair = next(
            policy
            for policy in policy_values
            if policy[
                "comparisons_vs_fixed"
            ][
                "all_pair_regression_reduction"
            ]
            > 0
        )
        first_adversarial = next(
            policy
            for policy in policy_values
            if policy[
                "comparisons_vs_fixed"
            ][
                "adversarial_regression_reduction"
            ]
            > 0
        )
        first_both = next(
            policy
            for policy in policy_values
            if (
                policy[
                    "comparisons_vs_fixed"
                ][
                    "all_pair_regression_reduction"
                ]
                > 0
                and policy[
                    "comparisons_vs_fixed"
                ][
                    "adversarial_regression_reduction"
                ]
                > 0
            )
        )
        best_pair = min(
            policy_values,
            key=lambda policy: (
                policy[
                    "all_admissible_pairs"
                ]["regression_fraction"],
                policy[
                    "adversarial_estimate_per_actual_metric"
                ]["regression_fraction"],
                policy[
                    "adversarial_estimate_per_actual_metric"
                ][
                    "maximum_positive_delay_requests"
                ],
            ),
        )
        best_adversarial = min(
            policy_values,
            key=lambda policy: (
                policy[
                    "adversarial_estimate_per_actual_metric"
                ]["regression_fraction"],
                policy[
                    "all_admissible_pairs"
                ]["regression_fraction"],
                policy[
                    "adversarial_estimate_per_actual_metric"
                ][
                    "maximum_positive_delay_requests"
                ],
            ),
        )
        fixed_worst_cell = fixed[
            "adversarial_estimate_per_actual_metric"
        ][
            "maximum_cell_regression"
        ]["regression_fraction"]
        first_fixed_worst_cell = next(
            policy
            for policy in policy_values
            if policy[
                "adversarial_estimate_per_actual_metric"
            ][
                "maximum_cell_regression"
            ]["regression_fraction"]
            <= fixed_worst_cell
        )
        radii[str(radius)] = {
            "radius": radius,
            "minimum_feasible_delay_cap_requests": (
                minimum_cap
            ),
            "delay_caps_requests": caps,
            "fixed_newest_first": (
                compact_policy(
                    policy=fixed,
                    delay_cap=fixed[
                        "adversarial_estimate_per_actual_metric"
                    ][
                        "maximum_positive_delay_requests"
                    ],
                    fixed=fixed,
                )
            ),
            "policies_by_delay_cap": policies,
            "headline": {
                "first_cap_beating_fixed_all_pairs": (
                    first_pair
                ),
                "first_cap_beating_fixed_adversarially": (
                    first_adversarial
                ),
                "first_cap_beating_fixed_in_both": (
                    first_both
                ),
                "best_all_pair_policy": (
                    best_pair
                ),
                "best_adversarial_policy": (
                    best_adversarial
                ),
                "first_cap_matching_fixed_worst_cell": (
                    first_fixed_worst_cell
                ),
                "pareto_front": (
                    pareto_front(
                        policy_values
                    )
                ),
            },
        }
    result = {
        "suite": (
            "pacing_coalescing_order_uncertainty_caps"
        ),
        "model": (
            "Jointly select member order under "
            "ordinal cost uncertainty and an "
            "observed hard maximum-delay cap. "
            "Within each cap, minimize worst cell "
            "regression, then delay, aggregate "
            "regression, acceleration, and "
            "chronological tie rank."
        ),
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            ),
            "cost_uncertainty_audit": (
                "results/raw/"
                "pacing_coalescing_order_cost_uncertainty/"
                "pacing_coalescing_order_cost_uncertainty_full.json"
            ),
        },
        "costs": costs,
        "uncertainty_radii": list(
            UNCERTAINTY_RADII
        ),
        "radii": radii,
        "validation": {
            "all_caps_satisfied": True,
            "uncapped_e86_reproduced": True,
            "worst_cell_nonincreasing_with_cap": (
                True
            ),
            "radius_pair_cell_counts": {
                radius: radii[radius][
                    "fixed_newest_first"
                ][
                    "all_admissible_pairs"
                ]["cell_count"]
                for radius in radii
            },
            "radius_adversarial_cell_counts": {
                radius: radii[radius][
                    "fixed_newest_first"
                ][
                    "adversarial_estimate_per_actual_metric"
                ]["cell_count"]
                for radius in radii
            },
        },
        "limitations": [
            "The audit uses ordinal radius-1/radius-2 uncertainty on E83's six cost points, not a continuous error model.",
            "Hard caps use observed path maxima and do not guarantee future tail latency.",
            "Worst-cell-first selection can sacrifice aggregate risk, creating non-monotone aggregate behavior as the cap loosens.",
            "The tested cap set contains only delay values induced by the four order policies.",
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
