from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_pacing_coalescing_order import (
    ORDER_POLICIES,
    ORDER_TIE_RANK,
)


DRIFT_METRICS = (
    "state_error",
    "mistake_flag_recall",
    "false_alarm_rate",
)

OBJECTIVES = (
    "aggregate_risk",
    "worst_profile_risk",
    "maximum_delay",
)


def load_e83(
    root: Path,
) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order"
        / "pacing_coalescing_order_full.json"
    )
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def per_cost_key(
    *,
    values: dict[str, Any],
    objective: str,
    order: str,
) -> tuple[Any, ...]:
    aggregate = values["combined"][
        "slower"
    ]["fraction"]
    profile = values[
        "maximum_profile_regression"
    ]["slower"]["fraction"]
    delay = (
        values["combined"][
            "positive_delay_requests"
        ]["maximum"]
        or 0.0
    )
    faster = values["combined"][
        "faster"
    ]["fraction"]
    if objective == "aggregate_risk":
        return (
            aggregate,
            profile,
            delay,
            -faster,
            ORDER_TIE_RANK[order],
        )
    if objective == "worst_profile_risk":
        return (
            profile,
            aggregate,
            delay,
            -faster,
            ORDER_TIE_RANK[order],
        )
    if objective == "maximum_delay":
        return (
            delay,
            aggregate,
            profile,
            -faster,
            ORDER_TIE_RANK[order],
        )
    raise ValueError(
        f"Unknown objective: {objective}"
    )


def robust_fixed_key(
    *,
    by_cost: dict[
        str,
        dict[str, Any],
    ],
    objective: str,
    order: str,
) -> tuple[Any, ...]:
    values = [
        by_order[order]
        for by_order in by_cost.values()
    ]
    max_aggregate = max(
        value["combined"][
            "slower"
        ]["fraction"]
        for value in values
    )
    max_profile = max(
        value[
            "maximum_profile_regression"
        ]["slower"]["fraction"]
        for value in values
    )
    max_delay = max(
        value["combined"][
            "positive_delay_requests"
        ]["maximum"]
        or 0.0
        for value in values
    )
    min_faster = min(
        value["combined"][
            "faster"
        ]["fraction"]
        for value in values
    )
    if objective == "aggregate_risk":
        return (
            max_aggregate,
            max_profile,
            max_delay,
            -min_faster,
            ORDER_TIE_RANK[order],
        )
    if objective == "worst_profile_risk":
        return (
            max_profile,
            max_aggregate,
            max_delay,
            -min_faster,
            ORDER_TIE_RANK[order],
        )
    if objective == "maximum_delay":
        return (
            max_delay,
            max_aggregate,
            max_profile,
            -min_faster,
            ORDER_TIE_RANK[order],
        )
    raise ValueError(
        f"Unknown objective: {objective}"
    )


def fixed_order_summary(
    *,
    by_cost: dict[
        str,
        dict[str, Any],
    ],
    order: str,
) -> dict[str, Any]:
    total = 0
    faster = 0
    slower = 0
    worst_cost = None
    worst_cost_fraction = -1.0
    maximum_profile = 0.0
    maximum_delay = 0.0
    minimum_faster = 1.0
    for cost, by_order in by_cost.items():
        value = by_order[order]
        combined = value["combined"]
        count = combined[
            "comparison_count"
        ]
        total += count
        faster += combined[
            "faster"
        ]["successes"]
        slower += combined[
            "slower"
        ]["successes"]
        regression = combined[
            "slower"
        ]["fraction"]
        if regression > worst_cost_fraction:
            worst_cost_fraction = (
                regression
            )
            worst_cost = cost
        maximum_profile = max(
            maximum_profile,
            value[
                "maximum_profile_regression"
            ]["slower"]["fraction"],
        )
        maximum_delay = max(
            maximum_delay,
            combined[
                "positive_delay_requests"
            ]["maximum"]
            or 0.0,
        )
        minimum_faster = min(
            minimum_faster,
            combined[
                "faster"
            ]["fraction"],
        )
    return {
        "order": order,
        "comparison_count": total,
        "faster_successes": faster,
        "regression_successes": slower,
        "overall_faster_fraction": (
            faster / total
        ),
        "overall_regression_fraction": (
            slower / total
        ),
        "worst_cost": worst_cost,
        "worst_cost_regression_fraction": (
            worst_cost_fraction
        ),
        "worst_profile_regression_fraction": (
            maximum_profile
        ),
        "maximum_positive_delay_requests": (
            maximum_delay
        ),
        "minimum_cost_faster_fraction": (
            minimum_faster
        ),
    }


def per_cost_oracles(
    *,
    by_cost: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    result = {}
    for cost, by_order in sorted(
        by_cost.items(),
        key=lambda item: float(
            item[0]
        ),
    ):
        result[cost] = {}
        for objective in OBJECTIVES:
            selected = min(
                ORDER_POLICIES,
                key=lambda order: (
                    per_cost_key(
                        values=by_order[
                            order
                        ],
                        objective=objective,
                        order=order,
                    )
                ),
            )
            result[cost][objective] = {
                "order": selected,
                "selection_key": list(
                    per_cost_key(
                        values=by_order[
                            selected
                        ],
                        objective=objective,
                        order=selected,
                    )[:-1]
                ),
            }
    return result


def robust_fixed_orders(
    *,
    by_cost: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    summaries = {
        order: fixed_order_summary(
            by_cost=by_cost,
            order=order,
        )
        for order in ORDER_POLICIES
    }
    selections = {}
    for objective in OBJECTIVES:
        selected = min(
            ORDER_POLICIES,
            key=lambda order: (
                robust_fixed_key(
                    by_cost=by_cost,
                    objective=objective,
                    order=order,
                )
            ),
        )
        selections[objective] = {
            "order": selected,
            "selection_key": list(
                robust_fixed_key(
                    by_cost=by_cost,
                    objective=objective,
                    order=selected,
                )[:-1]
            ),
        }
    return {
        "selections": selections,
        "order_summaries": summaries,
    }


def metric_order_key(
    *,
    by_cost: dict[
        str,
        dict[str, Any],
    ],
    metric: str,
    order: str,
) -> tuple[Any, ...]:
    values = [
        by_order[order][
            "by_breach_metric"
        ][metric]
        for by_order in by_cost.values()
    ]
    return (
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
        -min(
            value["faster"][
                "fraction"
            ]
            for value in values
        ),
        ORDER_TIE_RANK[order],
    )


def metric_cost_order(
    *,
    values: dict[str, Any],
) -> str:
    return min(
        ORDER_POLICIES,
        key=lambda order: (
            values[order][
                "slower"
            ]["fraction"],
            values[order][
                "positive_delay_requests"
            ]["maximum"]
            or 0.0,
            -values[order][
                "faster"
            ]["fraction"],
            ORDER_TIE_RANK[order],
        ),
    )


def evaluate_drift_policy(
    *,
    name: str,
    mapping: dict[
        tuple[str, str],
        str,
    ],
    by_cost: dict[
        str,
        dict[str, Any],
    ],
    configuration_key_count: int,
    decision_dimensions: list[str],
) -> dict[str, Any]:
    total = 0
    faster = 0
    slower = 0
    worst_cost = None
    worst_cost_fraction = -1.0
    worst_metric = None
    worst_metric_cost = None
    worst_metric_fraction = -1.0
    maximum_delay = 0.0
    by_cost_result = {}
    for cost, by_order in sorted(
        by_cost.items(),
        key=lambda item: float(
            item[0]
        ),
    ):
        cost_total = 0
        cost_faster = 0
        cost_slower = 0
        metric_result = {}
        for metric in DRIFT_METRICS:
            order = mapping[
                (cost, metric)
            ]
            value = by_order[order][
                "by_breach_metric"
            ][metric]
            count = value[
                "comparison_count"
            ]
            metric_faster = value[
                "faster"
            ]["successes"]
            metric_slower = value[
                "slower"
            ]["successes"]
            total += count
            faster += metric_faster
            slower += metric_slower
            cost_total += count
            cost_faster += metric_faster
            cost_slower += metric_slower
            regression = value[
                "slower"
            ]["fraction"]
            if (
                regression
                > worst_metric_fraction
            ):
                worst_metric_fraction = (
                    regression
                )
                worst_metric = metric
                worst_metric_cost = cost
            maximum_delay = max(
                maximum_delay,
                value[
                    "positive_delay_requests"
                ]["maximum"]
                or 0.0,
            )
            metric_result[metric] = {
                "order": order,
                "comparison_count": (
                    count
                ),
                "faster_fraction": (
                    value["faster"][
                        "fraction"
                    ]
                ),
                "regression_fraction": (
                    regression
                ),
                "maximum_positive_delay_requests": (
                    value[
                        "positive_delay_requests"
                    ]["maximum"]
                    or 0.0
                ),
            }
        cost_regression = (
            cost_slower / cost_total
        )
        if (
            cost_regression
            > worst_cost_fraction
        ):
            worst_cost_fraction = (
                cost_regression
            )
            worst_cost = cost
        by_cost_result[cost] = {
            "comparison_count": (
                cost_total
            ),
            "faster_fraction": (
                cost_faster / cost_total
            ),
            "regression_fraction": (
                cost_regression
            ),
            "metrics": metric_result,
        }
    return {
        "name": name,
        "configuration_key_count": (
            configuration_key_count
        ),
        "effective_unique_order_count": (
            len(set(mapping.values()))
        ),
        "decision_dimensions": (
            decision_dimensions
        ),
        "comparison_count": total,
        "faster_successes": faster,
        "regression_successes": slower,
        "overall_faster_fraction": (
            faster / total
        ),
        "overall_regression_fraction": (
            slower / total
        ),
        "worst_cost": worst_cost,
        "worst_cost_regression_fraction": (
            worst_cost_fraction
        ),
        "worst_metric": worst_metric,
        "worst_metric_cost": (
            worst_metric_cost
        ),
        "worst_metric_cost_regression_fraction": (
            worst_metric_fraction
        ),
        "maximum_positive_delay_requests": (
            maximum_delay
        ),
        "mapping": {
            cost: {
                metric: mapping[
                    (cost, metric)
                ]
                for metric in DRIFT_METRICS
            }
            for cost in sorted(
                by_cost,
                key=float,
            )
        },
        "by_cost": by_cost_result,
    }


def attach_metric_cost_regret(
    *,
    candidate: dict[str, Any],
    by_cost: dict[
        str,
        dict[str, Any],
    ],
) -> None:
    rows = {}
    normalized_regrets = []
    excess_successes = 0
    oracle_successes = 0
    for cost in sorted(
        by_cost,
        key=float,
    ):
        rows[cost] = {}
        for metric in DRIFT_METRICS:
            order_values = {
                order: by_cost[cost][
                    order
                ]["by_breach_metric"][
                    metric
                ]
                for order in ORDER_POLICIES
            }
            oracle_order = metric_cost_order(
                values=order_values
            )
            chosen_order = candidate[
                "mapping"
            ][cost][metric]
            oracle_value = order_values[
                oracle_order
            ]
            chosen_value = order_values[
                chosen_order
            ]
            best_fraction = oracle_value[
                "slower"
            ]["fraction"]
            worst_fraction = max(
                value["slower"][
                    "fraction"
                ]
                for value
                in order_values.values()
            )
            chosen_fraction = (
                chosen_value["slower"][
                    "fraction"
                ]
            )
            spread = (
                worst_fraction
                - best_fraction
            )
            normalized = (
                (
                    chosen_fraction
                    - best_fraction
                )
                / spread
                if spread > 0
                else 0.0
            )
            if not (
                -1e-12
                <= normalized
                <= 1.0 + 1e-12
            ):
                raise RuntimeError(
                    "Normalized oracle regret "
                    "outside [0, 1]"
                )
            normalized = min(
                1.0,
                max(0.0, normalized),
            )
            chosen_successes = (
                chosen_value["slower"][
                    "successes"
                ]
            )
            oracle_cell_successes = (
                oracle_value["slower"][
                    "successes"
                ]
            )
            if (
                chosen_successes
                < oracle_cell_successes
            ):
                raise RuntimeError(
                    "Chosen policy beat its "
                    "metric-cost oracle"
                )
            excess = (
                chosen_successes
                - oracle_cell_successes
            )
            excess_successes += excess
            oracle_successes += (
                oracle_cell_successes
            )
            normalized_regrets.append(
                normalized
            )
            rows[cost][metric] = {
                "chosen_order": (
                    chosen_order
                ),
                "oracle_order": (
                    oracle_order
                ),
                "chosen_regression_successes": (
                    chosen_successes
                ),
                "oracle_regression_successes": (
                    oracle_cell_successes
                ),
                "excess_regression_successes": (
                    excess
                ),
                "chosen_regression_fraction": (
                    chosen_fraction
                ),
                "oracle_regression_fraction": (
                    best_fraction
                ),
                "normalized_regret": (
                    normalized
                ),
            }
    candidate[
        "metric_cost_oracle_regret"
    ] = {
        "oracle_regression_successes": (
            oracle_successes
        ),
        "excess_regression_successes": (
            excess_successes
        ),
        "maximum_normalized_regret": max(
            normalized_regrets
        ),
        "unweighted_mean_normalized_regret": (
            sum(normalized_regrets)
            / len(normalized_regrets)
        ),
        "cells": rows,
    }


def pareto_front(
    candidates: dict[
        str,
        dict[str, Any],
    ],
) -> list[str]:
    front = []
    for name, value in candidates.items():
        objectives = (
            value[
                "overall_regression_fraction"
            ],
            value[
                "worst_cost_regression_fraction"
            ],
            value[
                "worst_metric_cost_regression_fraction"
            ],
            value[
                "maximum_positive_delay_requests"
            ],
            -value[
                "overall_faster_fraction"
            ],
            value[
                "configuration_key_count"
            ],
        )
        dominated = False
        for other_name, other in (
            candidates.items()
        ):
            if other_name == name:
                continue
            other_objectives = (
                other[
                    "overall_regression_fraction"
                ],
                other[
                    "worst_cost_regression_fraction"
                ],
                other[
                    "worst_metric_cost_regression_fraction"
                ],
                other[
                    "maximum_positive_delay_requests"
                ],
                -other[
                    "overall_faster_fraction"
                ],
                other[
                    "configuration_key_count"
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
            front.append(name)
    return sorted(front)


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    headline = e83["headline"]
    certification = headline[
        "certification_by_marginal_fraction_and_order"
    ]
    drift = headline[
        "drift_by_marginal_fraction_and_order"
    ]
    costs = sorted(
        drift,
        key=float,
    )
    if set(certification) != set(
        drift
    ):
        raise RuntimeError(
            "Certification/drift cost grids "
            "differ"
        )
    for cost in costs:
        if set(
            certification[cost]
        ) != set(ORDER_POLICIES):
            raise RuntimeError(
                "Certification order inventory "
                "mismatch"
            )
        if set(
            drift[cost]
        ) != set(ORDER_POLICIES):
            raise RuntimeError(
                "Drift order inventory mismatch"
            )
    certification_oracles = (
        per_cost_oracles(
            by_cost=certification
        )
    )
    drift_oracles = per_cost_oracles(
        by_cost=drift
    )
    certification_robust = (
        robust_fixed_orders(
            by_cost=certification
        )
    )
    drift_robust = robust_fixed_orders(
        by_cost=drift
    )
    e83_reproduction_checks = 0
    for cost in costs:
        for domain, oracles in (
            (
                "certification",
                certification_oracles,
            ),
            (
                "drift",
                drift_oracles,
            ),
        ):
            expected = headline[
                "order_selections"
            ][cost][domain][
                "aggregate_regression_minimizer"
            ]
            if (
                oracles[cost][
                    "aggregate_risk"
                ]["order"]
                != expected
            ):
                raise RuntimeError(
                    "Per-cost aggregate oracle "
                    "did not reproduce E83"
                )
            e83_reproduction_checks += 1
    zero_cost_tie_checks = 0
    for domain, oracles in (
        (
            "certification",
            certification_oracles,
        ),
        ("drift", drift_oracles),
    ):
        for objective in OBJECTIVES:
            if (
                oracles["0"][
                    objective
                ]["order"]
                != "oldest_first"
            ):
                raise RuntimeError(
                    "Zero-cost tie did not "
                    "preserve chronology"
                )
            zero_cost_tie_checks += 1
    fixed_mappings = {
        order: {
            (cost, metric): order
            for cost in costs
            for metric in DRIFT_METRICS
        }
        for order in ORDER_POLICIES
    }
    cost_mapping = {
        (
            cost,
            metric,
        ): drift_oracles[cost][
            "aggregate_risk"
        ]["order"]
        for cost in costs
        for metric in DRIFT_METRICS
    }
    robust_metric_orders = {
        metric: min(
            ORDER_POLICIES,
            key=lambda order: (
                metric_order_key(
                    by_cost=drift,
                    metric=metric,
                    order=order,
                )
            ),
        )
        for metric in DRIFT_METRICS
    }
    metric_mapping = {
        (
            cost,
            metric,
        ): robust_metric_orders[
            metric
        ]
        for cost in costs
        for metric in DRIFT_METRICS
    }
    metric_cost_mapping = {
        (
            cost,
            metric,
        ): metric_cost_order(
            values={
                order: drift[cost][
                    order
                ][
                    "by_breach_metric"
                ][metric]
                for order
                in ORDER_POLICIES
            }
        )
        for cost in costs
        for metric in DRIFT_METRICS
    }
    candidates = {}
    for order, mapping in (
        fixed_mappings.items()
    ):
        name = f"fixed_{order}"
        candidates[name] = (
            evaluate_drift_policy(
                name=name,
                mapping=mapping,
                by_cost=drift,
                configuration_key_count=1,
                decision_dimensions=[],
            )
        )
    candidates[
        "cost_aware_aggregate"
    ] = evaluate_drift_policy(
        name="cost_aware_aggregate",
        mapping=cost_mapping,
        by_cost=drift,
        configuration_key_count=len(
            costs
        ),
        decision_dimensions=[
            "marginal_fraction"
        ],
    )
    candidates[
        "metric_aware_robust"
    ] = evaluate_drift_policy(
        name="metric_aware_robust",
        mapping=metric_mapping,
        by_cost=drift,
        configuration_key_count=len(
            DRIFT_METRICS
        ),
        decision_dimensions=[
            "breach_metric"
        ],
    )
    candidates[
        "metric_cost_aware"
    ] = evaluate_drift_policy(
        name="metric_cost_aware",
        mapping=metric_cost_mapping,
        by_cost=drift,
        configuration_key_count=(
            len(costs)
            * len(DRIFT_METRICS)
        ),
        decision_dimensions=[
            "marginal_fraction",
            "breach_metric",
        ],
    )
    for value in candidates.values():
        attach_metric_cost_regret(
            candidate=value,
            by_cost=drift,
        )
    baseline = candidates[
        "fixed_newest_first"
    ]
    for value in candidates.values():
        value[
            "regression_reduction_vs_fixed_newest_first"
        ] = (
            1.0
            - value[
                "overall_regression_fraction"
            ]
            / baseline[
                "overall_regression_fraction"
            ]
            if baseline[
                "overall_regression_fraction"
            ]
            > 0
            else 0.0
        )
        value[
            "faster_fraction_delta_vs_fixed_newest_first"
        ] = (
            value[
                "overall_faster_fraction"
            ]
            - baseline[
                "overall_faster_fraction"
            ]
        )
    result = {
        "suite": (
            "pacing_coalescing_order_robustness"
        ),
        "model": (
            "Select deployable exact-batch "
            "member-order policies from E83 "
            "without live outcome dependence. "
            "Objectives are lexicographic and "
            "weight-free."
        ),
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            )
        },
        "objective_definitions": {
            "aggregate_risk": [
                "aggregate regression fraction",
                "worst-profile regression fraction",
                "maximum positive delay",
                "negative faster-action fraction",
                "chronological tie rank",
            ],
            "worst_profile_risk": [
                "worst-profile regression fraction",
                "aggregate regression fraction",
                "maximum positive delay",
                "negative faster-action fraction",
                "chronological tie rank",
            ],
            "maximum_delay": [
                "maximum positive delay",
                "aggregate regression fraction",
                "worst-profile regression fraction",
                "negative faster-action fraction",
                "chronological tie rank",
            ],
        },
        "marginal_fractions": costs,
        "per_cost_oracles": {
            "certification": (
                certification_oracles
            ),
            "drift": drift_oracles,
        },
        "robust_fixed_orders": {
            "certification": (
                certification_robust
            ),
            "drift": drift_robust,
        },
        "domain_aware_aggregate_policy": {
            "configuration_key_count": 2,
            "certification_order": (
                certification_robust[
                    "selections"
                ]["aggregate_risk"][
                    "order"
                ]
            ),
            "drift_order": (
                drift_robust[
                    "selections"
                ]["aggregate_risk"][
                    "order"
                ]
            ),
            "certification_summary": (
                certification_robust[
                    "order_summaries"
                ][
                    certification_robust[
                        "selections"
                    ]["aggregate_risk"][
                        "order"
                    ]
                ]
            ),
            "drift_summary": (
                drift_robust[
                    "order_summaries"
                ][
                    drift_robust[
                        "selections"
                    ]["aggregate_risk"][
                        "order"
                    ]
                ]
            ),
        },
        "robust_metric_orders": (
            robust_metric_orders
        ),
        "drift_policy_candidates": (
            candidates
        ),
        "drift_policy_pareto_front": (
            pareto_front(candidates)
        ),
        "validation": {
            "marginal_fraction_count": (
                len(costs)
            ),
            "order_count": len(
                ORDER_POLICIES
            ),
            "drift_metric_count": len(
                DRIFT_METRICS
            ),
            "per_cost_e83_aggregate_reproduction_checks": (
                e83_reproduction_checks
            ),
            "zero_cost_chronological_tie_checks": (
                zero_cost_tie_checks
            ),
            "candidate_policy_count": (
                len(candidates)
            ),
            "metric_aware_collapsed_to_fixed_order": (
                len(
                    set(
                        robust_metric_orders.values()
                    )
                )
                == 1
            ),
            "metric_cost_mapping_entry_count": (
                len(metric_cost_mapping)
            ),
            "normalized_oracle_regret_bounds": {
                name: {
                    "minimum": min(
                        cell[
                            "normalized_regret"
                        ]
                        for by_metric
                        in value[
                            "metric_cost_oracle_regret"
                        ]["cells"].values()
                        for cell
                        in by_metric.values()
                    ),
                    "maximum": max(
                        cell[
                            "normalized_regret"
                        ]
                        for by_metric
                        in value[
                            "metric_cost_oracle_regret"
                        ]["cells"].values()
                        for cell
                        in by_metric.values()
                    ),
                }
                for name, value
                in candidates.items()
            },
        },
        "limitations": [
            "The audit reuses E83's six marginal-cost points and compact summaries rather than rerunning paths.",
            "Lexicographic objectives avoid arbitrary utility weights but encode explicit priority order.",
            "Metric-aware policies condition only on the preregistered breach metric, never live monitor outcomes.",
            "Marginal cost must be known or bucketed before selecting a cost-aware order.",
            "The candidate Pareto front treats configuration-key count as an objective; operational implementation cost may differ.",
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
