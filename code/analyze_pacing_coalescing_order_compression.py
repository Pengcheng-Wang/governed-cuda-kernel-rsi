from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from analyze_pacing_coalescing_order import (
    ORDER_POLICIES,
    ORDER_TIE_RANK,
)
from analyze_pacing_coalescing_order_robustness import (
    DRIFT_METRICS,
    evaluate_drift_policy,
    load_e83,
)


GAIN_TARGETS = (
    0.90,
    0.95,
    0.99,
    1.00,
)


def load_e84(
    root: Path,
) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_robustness"
        / "pacing_coalescing_order_robustness_full.json"
    )
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def mapping_segments(
    *,
    costs: list[str],
    orders: tuple[str, ...],
) -> list[dict[str, Any]]:
    segments = []
    start = 0
    for index in range(
        1,
        len(costs) + 1,
    ):
        if (
            index == len(costs)
            or orders[index]
            != orders[start]
        ):
            segments.append(
                {
                    "start_cost": costs[start],
                    "end_cost": costs[
                        index - 1
                    ],
                    "order": orders[start],
                    "cost_count": (
                        index - start
                    ),
                }
            )
            start = index
    return segments


def metric_mapping_key(
    *,
    regression_successes: int,
    maximum_delay: float,
    faster_successes: int,
    segment_count: int,
    orders: tuple[str, ...],
) -> tuple[Any, ...]:
    return (
        regression_successes,
        maximum_delay,
        segment_count,
        -faster_successes,
        tuple(
            ORDER_TIE_RANK[order]
            for order in orders
        ),
    )


def enumerate_metric_options(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    costs: list[str],
    metric: str,
    delay_cap: float,
) -> dict[int, dict[str, Any]]:
    best_by_segment_count = {}
    for orders in itertools.product(
        ORDER_POLICIES,
        repeat=len(costs),
    ):
        values = [
            drift[cost][order][
                "by_breach_metric"
            ][metric]
            for cost, order in zip(
                costs,
                orders,
            )
        ]
        maximum_delay = max(
            value[
                "positive_delay_requests"
            ]["maximum"]
            or 0.0
            for value in values
        )
        if maximum_delay > delay_cap:
            continue
        regression_successes = sum(
            value["slower"][
                "successes"
            ]
            for value in values
        )
        faster_successes = sum(
            value["faster"][
                "successes"
            ]
            for value in values
        )
        segments = mapping_segments(
            costs=costs,
            orders=orders,
        )
        segment_count = len(segments)
        candidate = {
            "metric": metric,
            "segment_count": (
                segment_count
            ),
            "regression_successes": (
                regression_successes
            ),
            "faster_successes": (
                faster_successes
            ),
            "maximum_positive_delay_requests": (
                maximum_delay
            ),
            "orders": list(orders),
            "segments": segments,
        }
        key = metric_mapping_key(
            regression_successes=(
                regression_successes
            ),
            maximum_delay=maximum_delay,
            faster_successes=(
                faster_successes
            ),
            segment_count=segment_count,
            orders=orders,
        )
        previous = (
            best_by_segment_count.get(
                segment_count
            )
        )
        if (
            previous is None
            or key
            < previous["selection_key"]
        ):
            candidate[
                "selection_key"
            ] = key
            best_by_segment_count[
                segment_count
            ] = candidate
    return best_by_segment_count


def combined_key(
    *,
    regression_successes: int,
    maximum_delay: float,
    configuration_key_count: int,
    faster_successes: int,
    segment_counts: tuple[int, ...],
    order_signatures: tuple[
        tuple[int, ...],
        ...,
    ],
) -> tuple[Any, ...]:
    return (
        regression_successes,
        maximum_delay,
        configuration_key_count,
        -faster_successes,
        segment_counts,
        order_signatures,
    )


def best_policy_for_budget(
    *,
    metric_options: dict[
        str,
        dict[int, dict[str, Any]],
    ],
    costs: list[str],
    key_budget: int,
) -> dict[str, Any] | None:
    best = None
    for segment_counts in (
        itertools.product(
            range(
                1,
                len(costs) + 1,
            ),
            repeat=len(DRIFT_METRICS),
        )
    ):
        configuration_key_count = (
            sum(segment_counts)
        )
        if (
            configuration_key_count
            > key_budget
        ):
            continue
        selected = []
        for metric, segment_count in zip(
            DRIFT_METRICS,
            segment_counts,
        ):
            option = metric_options[
                metric
            ].get(segment_count)
            if option is None:
                break
            selected.append(option)
        if (
            len(selected)
            != len(DRIFT_METRICS)
        ):
            continue
        regression_successes = sum(
            option[
                "regression_successes"
            ]
            for option in selected
        )
        faster_successes = sum(
            option["faster_successes"]
            for option in selected
        )
        maximum_delay = max(
            option[
                "maximum_positive_delay_requests"
            ]
            for option in selected
        )
        order_signatures = tuple(
            tuple(
                ORDER_TIE_RANK[order]
                for order
                in option["orders"]
            )
            for option in selected
        )
        key = combined_key(
            regression_successes=(
                regression_successes
            ),
            maximum_delay=maximum_delay,
            configuration_key_count=(
                configuration_key_count
            ),
            faster_successes=(
                faster_successes
            ),
            segment_counts=(
                segment_counts
            ),
            order_signatures=(
                order_signatures
            ),
        )
        if (
            best is None
            or key < best["selection_key"]
        ):
            mapping = {}
            segments = {}
            for metric, option in zip(
                DRIFT_METRICS,
                selected,
            ):
                segments[metric] = option[
                    "segments"
                ]
                for cost, order in zip(
                    costs,
                    option["orders"],
                ):
                    mapping[
                        (cost, metric)
                    ] = order
            best = {
                "selection_key": key,
                "configuration_key_count": (
                    configuration_key_count
                ),
                "key_budget": key_budget,
                "mapping": mapping,
                "segments": segments,
            }
    return best


def derive_delay_caps(
    *,
    drift: dict[
        str,
        dict[str, Any],
    ],
    costs: list[str],
) -> tuple[float, list[float]]:
    delays = {
        (
            drift[cost][order][
                "by_breach_metric"
            ][metric][
                "positive_delay_requests"
            ]["maximum"]
            or 0.0
        )
        for cost in costs
        for metric in DRIFT_METRICS
        for order in ORDER_POLICIES
    }
    minimum_feasible_cap = max(
        min(
            (
                drift[cost][order][
                    "by_breach_metric"
                ][metric][
                    "positive_delay_requests"
                ]["maximum"]
                or 0.0
            )
            for order in ORDER_POLICIES
        )
        for cost in costs
        for metric in DRIFT_METRICS
    )
    caps = sorted(
        delay
        for delay in delays
        if delay >= minimum_feasible_cap
    )
    return minimum_feasible_cap, caps


def compact_policy(
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "key_budget": policy[
            "key_budget"
        ],
        "configuration_key_count": (
            policy[
                "configuration_key_count"
            ]
        ),
        "regression_successes": policy[
            "regression_successes"
        ],
        "comparison_count": policy[
            "comparison_count"
        ],
        "overall_regression_fraction": (
            policy[
                "overall_regression_fraction"
            ]
        ),
        "regression_reduction_vs_fixed_newest_first": (
            policy[
                "regression_reduction_vs_fixed_newest_first"
            ]
        ),
        "oracle_gain_retention_fraction": (
            policy[
                "oracle_gain_retention_fraction"
            ]
        ),
        "maximum_positive_delay_requests": (
            policy[
                "maximum_positive_delay_requests"
            ]
        ),
        "faster_successes": policy[
            "faster_successes"
        ],
        "overall_faster_fraction": policy[
            "overall_faster_fraction"
        ],
        "worst_cost": policy[
            "worst_cost"
        ],
        "worst_cost_regression_fraction": (
            policy[
                "worst_cost_regression_fraction"
            ]
        ),
        "worst_metric": policy[
            "worst_metric"
        ],
        "worst_metric_cost": policy[
            "worst_metric_cost"
        ],
        "worst_metric_cost_regression_fraction": (
            policy[
                "worst_metric_cost_regression_fraction"
            ]
        ),
        "segments": policy["segments"],
    }
    if (
        "configured_delay_cap_requests"
        in policy
    ):
        result[
            "configured_delay_cap_requests"
        ] = policy[
            "configured_delay_cap_requests"
        ]
    return result


def pareto_front(
    policies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique = {}
    for policy in policies:
        signature = tuple(
            tuple(
                policy["mapping"][cost][
                    metric
                ]
                for cost in policy[
                    "costs"
                ]
            )
            for metric in DRIFT_METRICS
        )
        key = (
            signature,
            policy[
                "configuration_key_count"
            ],
        )
        previous = unique.get(key)
        if (
            previous is None
            or (
                policy[
                    "configured_delay_cap_requests"
                ],
                policy["key_budget"],
            )
            < (
                previous[
                    "configured_delay_cap_requests"
                ],
                previous["key_budget"],
            )
        ):
            unique[key] = policy
    values = list(unique.values())
    front = []
    for policy in values:
        objectives = (
            policy[
                "regression_successes"
            ],
            policy[
                "maximum_positive_delay_requests"
            ],
            policy[
                "configuration_key_count"
            ],
            -policy[
                "faster_successes"
            ],
        )
        dominated = False
        for other in values:
            if other is policy:
                continue
            other_objectives = (
                other[
                    "regression_successes"
                ],
                other[
                    "maximum_positive_delay_requests"
                ],
                other[
                    "configuration_key_count"
                ],
                -other[
                    "faster_successes"
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
                "maximum_positive_delay_requests"
            ],
            policy[
                "configuration_key_count"
            ],
            policy[
                "regression_successes"
            ],
        ),
    )


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e84 = load_e84(args.root)
    drift = e83["headline"][
        "drift_by_marginal_fraction_and_order"
    ]
    costs = sorted(
        drift,
        key=float,
    )
    candidates = e84[
        "drift_policy_candidates"
    ]
    baseline = candidates[
        "fixed_newest_first"
    ]
    oracle = candidates[
        "metric_cost_aware"
    ]
    baseline_regressions = baseline[
        "regression_successes"
    ]
    oracle_regressions = oracle[
        "regression_successes"
    ]
    available_oracle_gain = (
        baseline_regressions
        - oracle_regressions
    )
    if available_oracle_gain <= 0:
        raise RuntimeError(
            "E84 oracle does not improve baseline"
        )
    minimum_feasible_cap, delay_caps = (
        derive_delay_caps(
            drift=drift,
            costs=costs,
        )
    )
    policies_by_cap = {}
    all_policies = []
    mapping_enumeration_count = 0
    for delay_cap in delay_caps:
        metric_options = {}
        for metric in DRIFT_METRICS:
            metric_options[metric] = (
                enumerate_metric_options(
                    drift=drift,
                    costs=costs,
                    metric=metric,
                    delay_cap=delay_cap,
                )
            )
            mapping_enumeration_count += (
                len(ORDER_POLICIES)
                ** len(costs)
            )
        cap_key = f"{delay_cap:g}"
        policies_by_cap[cap_key] = {
            "configured_delay_cap_requests": (
                delay_cap
            ),
            "by_key_budget": {},
        }
        previous_regressions = None
        for key_budget in range(
            len(DRIFT_METRICS),
            (
                len(DRIFT_METRICS)
                * len(costs)
            )
            + 1,
        ):
            compact = best_policy_for_budget(
                metric_options=metric_options,
                costs=costs,
                key_budget=key_budget,
            )
            if compact is None:
                raise RuntimeError(
                    "No feasible compressed policy"
                )
            evaluated = evaluate_drift_policy(
                name=(
                    "delay_capped_compressed_"
                    f"{cap_key}_{key_budget}"
                ),
                mapping=compact[
                    "mapping"
                ],
                by_cost=drift,
                configuration_key_count=(
                    compact[
                        "configuration_key_count"
                    ]
                ),
                decision_dimensions=[
                    "breach_metric",
                    "marginal_fraction_bucket",
                ],
            )
            if (
                evaluated[
                    "maximum_positive_delay_requests"
                ]
                > delay_cap
            ):
                raise RuntimeError(
                    "Policy violates delay cap"
                )
            evaluated["key_budget"] = (
                key_budget
            )
            evaluated[
                "configured_delay_cap_requests"
            ] = delay_cap
            evaluated["segments"] = compact[
                "segments"
            ]
            evaluated["costs"] = costs
            evaluated[
                "regression_reduction_vs_fixed_newest_first"
            ] = (
                baseline_regressions
                - evaluated[
                    "regression_successes"
                ]
            )
            evaluated[
                "oracle_gain_retention_fraction"
            ] = (
                evaluated[
                    "regression_reduction_vs_fixed_newest_first"
                ]
                / available_oracle_gain
            )
            if (
                evaluated[
                    "oracle_gain_retention_fraction"
                ]
                > 1.0 + 1e-12
            ):
                raise RuntimeError(
                    "Compressed policy beat E84 oracle"
                )
            if (
                previous_regressions
                is not None
                and evaluated[
                    "regression_successes"
                ]
                > previous_regressions
            ):
                raise RuntimeError(
                    "Larger key budget worsened "
                    "best regression count"
                )
            previous_regressions = (
                evaluated[
                    "regression_successes"
                ]
            )
            policies_by_cap[cap_key][
                "by_key_budget"
            ][str(key_budget)] = evaluated
            all_policies.append(evaluated)
        milestones = {}
        for target in GAIN_TARGETS:
            selected = next(
                (
                    policy
                    for policy in (
                        policies_by_cap[
                            cap_key
                        ][
                            "by_key_budget"
                        ].values()
                    )
                    if policy[
                        "oracle_gain_retention_fraction"
                    ]
                    >= target - 1e-12
                ),
                None,
            )
            milestones[
                f"{target:.2f}"
            ] = (
                None
                if selected is None
                else compact_policy(
                    policy=selected
                )
            )
        first_baseline_improvement = next(
            (
                policy
                for policy in (
                    policies_by_cap[
                        cap_key
                    ]["by_key_budget"].values()
                )
                if policy[
                    "regression_successes"
                ]
                < baseline_regressions
            ),
            None,
        )
        policies_by_cap[cap_key][
            "gain_milestones"
        ] = milestones
        policies_by_cap[cap_key][
            "minimum_budget_beating_fixed_newest_first"
        ] = (
            None
            if first_baseline_improvement
            is None
            else compact_policy(
                policy=(
                    first_baseline_improvement
                )
            )
        )
    cap_keys = sorted(
        policies_by_cap,
        key=float,
    )
    for key_budget in range(
        len(DRIFT_METRICS),
        (
            len(DRIFT_METRICS)
            * len(costs)
        )
        + 1,
    ):
        previous_regressions = None
        for cap_key in cap_keys:
            policy = policies_by_cap[
                cap_key
            ]["by_key_budget"][
                str(key_budget)
            ]
            if (
                previous_regressions
                is not None
                and policy[
                    "regression_successes"
                ]
                > previous_regressions
            ):
                raise RuntimeError(
                    "Looser delay cap worsened "
                    "best regression count"
                )
            previous_regressions = policy[
                "regression_successes"
            ]
    maximum_cap_key = cap_keys[-1]
    unconstrained = policies_by_cap[
        maximum_cap_key
    ]["by_key_budget"]
    oracle_recovery = next(
        (
            policy
            for policy
            in unconstrained.values()
            if policy[
                "regression_successes"
            ]
            == oracle_regressions
        ),
        None,
    )
    if oracle_recovery is None:
        raise RuntimeError(
            "Compression never recovered E84 oracle"
        )
    maximum_budget_policy = unconstrained[
        str(
            len(DRIFT_METRICS)
            * len(costs)
        )
    ]
    if (
        maximum_budget_policy[
            "regression_successes"
        ]
        != oracle_regressions
    ):
        raise RuntimeError(
            "Maximum budget did not recover "
            "E84 oracle"
        )
    if (
        oracle_recovery[
            "configuration_key_count"
        ]
        != 8
    ):
        raise RuntimeError(
            "Expected eight-key oracle recovery"
        )
    lowest_delay_improvement = min(
        (
            policy
            for policy in all_policies
            if policy[
                "regression_successes"
            ]
            < baseline_regressions
        ),
        key=lambda policy: (
            policy[
                "maximum_positive_delay_requests"
            ],
            policy[
                "configuration_key_count"
            ],
            policy[
                "regression_successes"
            ],
        ),
    )
    front = pareto_front(all_policies)
    selected_caps = {}
    for delay_cap in (
        4999.0,
        8000.0,
        10000.0,
        10500.0,
        19500.0,
        22500.0,
    ):
        cap_key = f"{delay_cap:g}"
        selected_caps[cap_key] = (
            compact_policy(
                policy=policies_by_cap[
                    cap_key
                ]["by_key_budget"]["18"]
            )
        )
    published_policies_by_cap = {
        cap_key: {
            "configured_delay_cap_requests": (
                values[
                    "configured_delay_cap_requests"
                ]
            ),
            "by_key_budget": {
                key_budget: compact_policy(
                    policy=policy
                )
                for key_budget, policy
                in values[
                    "by_key_budget"
                ].items()
            },
            "gain_milestones": values[
                "gain_milestones"
            ],
            "minimum_budget_beating_fixed_newest_first": (
                values[
                    "minimum_budget_beating_fixed_newest_first"
                ]
            ),
        }
        for cap_key, values
        in policies_by_cap.items()
    }
    result = {
        "suite": (
            "pacing_coalescing_order_compression"
        ),
        "model": (
            "Compress E84 drift member-order "
            "policies into contiguous "
            "metric-specific marginal-cost "
            "buckets under observed hard "
            "maximum-delay caps. Selection is "
            "outcome-independent and uses "
            "lexicographic regression, delay, "
            "complexity, acceleration, and "
            "chronological tie objectives."
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
        },
        "objective_definition": [
            "minimum exact regression successes",
            "minimum realized maximum positive delay",
            "minimum contiguous configuration keys",
            "maximum faster-action successes",
            "chronological deterministic tie rank",
        ],
        "costs": costs,
        "delay_caps_requests": (
            delay_caps
        ),
        "minimum_globally_feasible_delay_cap_requests": (
            minimum_feasible_cap
        ),
        "baselines": {
            "fixed_newest_first": {
                "regression_successes": (
                    baseline_regressions
                ),
                "faster_successes": baseline[
                    "faster_successes"
                ],
                "maximum_positive_delay_requests": (
                    baseline[
                        "maximum_positive_delay_requests"
                    ]
                ),
            },
            "metric_cost_oracle": {
                "regression_successes": (
                    oracle_regressions
                ),
                "faster_successes": oracle[
                    "faster_successes"
                ],
                "maximum_positive_delay_requests": (
                    oracle[
                        "maximum_positive_delay_requests"
                    ]
                ),
                "available_regression_reduction": (
                    available_oracle_gain
                ),
            },
        },
        "policies_by_delay_cap": (
            published_policies_by_cap
        ),
        "headline": {
            "minimum_keys_recovering_full_oracle_gain": (
                oracle_recovery[
                    "configuration_key_count"
                ]
            ),
            "unconstrained_gain_milestones": (
                policies_by_cap[
                    maximum_cap_key
                ]["gain_milestones"]
            ),
            "unconstrained_three_key_policy": (
                compact_policy(
                    policy=unconstrained[
                        "3"
                    ]
                )
            ),
            "unconstrained_oracle_recovery_policy": (
                compact_policy(
                    policy=oracle_recovery
                )
            ),
            "lowest_delay_policy_beating_fixed_newest_first": (
                compact_policy(
                    policy=(
                        lowest_delay_improvement
                    )
                )
            ),
            "selected_delay_cap_max_budget_tradeoffs": (
                selected_caps
            ),
            "pareto_policy_count": len(
                front
            ),
            "pareto_front": [
                compact_policy(
                    policy=policy
                )
                for policy in front
            ],
        },
        "validation": {
            "cost_count": len(costs),
            "metric_count": len(
                DRIFT_METRICS
            ),
            "order_count": len(
                ORDER_POLICIES
            ),
            "mapping_enumeration_count": (
                mapping_enumeration_count
            ),
            "delay_cap_count": len(
                delay_caps
            ),
            "key_budget_count": (
                len(DRIFT_METRICS)
                * len(costs)
                - len(DRIFT_METRICS)
                + 1
            ),
            "all_delay_caps_satisfied": True,
            "regression_nonincreasing_with_budget": (
                True
            ),
            "regression_nonincreasing_with_cap": (
                True
            ),
            "maximum_budget_recovers_oracle": (
                True
            ),
            "minimum_oracle_recovery_keys": (
                oracle_recovery[
                    "configuration_key_count"
                ]
            ),
            "negative_gain_retention_allowed": (
                True
            ),
        },
        "limitations": [
            "The audit reuses E83's six marginal-cost points and compact path summaries rather than rerunning paths.",
            "Cost buckets are contiguous on the tested grid; interpolation and unseen costs are not certified.",
            "Hard caps use observed maximum positive delay, not a probabilistic future-tail guarantee.",
            "Configuration-key count is a proxy for implementation and calibration burden.",
            "The aggregate regression objective can sacrifice acceleration or per-profile behavior unless separately gated.",
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
