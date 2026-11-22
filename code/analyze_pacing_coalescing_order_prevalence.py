from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    estimate_distribution,
    policy_inventory,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    DRIFT_METRICS,
    load_e83,
)


PREVALENCE_DENOMINATOR = 10
REPRESENTATIVE_ACCURACIES = (
    0.0,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
)
E85_COVERAGE_TARGETS = (
    0.50,
    0.75,
    0.90,
    1.00,
)


def load_e88(
    root: Path,
) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_accuracy_frontier"
        / "pacing_coalescing_order_accuracy_frontier_full.json"
    )
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def integer_compositions(
    *,
    total: int,
    parts: int,
) -> list[tuple[int, ...]]:
    compositions = []
    for bars in itertools.combinations(
        range(total + parts - 1),
        parts - 1,
    ):
        values = []
        previous = -1
        for position in (
            bars
            + (total + parts - 1,)
        ):
            values.append(
                position - previous - 1
            )
            previous = position
        if sum(values) != total:
            raise RuntimeError(
                "Composition did not preserve total"
            )
        compositions.append(
            tuple(values)
        )
    return compositions


def prevalence_profiles(
    *,
    denominator: int,
    cost_count: int,
) -> list[dict[str, Any]]:
    profiles = []
    for counts in integer_compositions(
        total=denominator,
        parts=cost_count,
    ):
        weights = tuple(
            count / denominator
            for count in counts
        )
        if abs(
            sum(weights) - 1.0
        ) > 1e-12:
            raise RuntimeError(
                "Prevalence weights do not "
                "normalize"
            )
        profiles.append(
            {
                "counts": counts,
                "weights": weights,
                "support_size": sum(
                    count > 0
                    for count in counts
                ),
            }
        )
    return profiles


def per_cost_metrics(
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
) -> list[dict[str, Any]]:
    rows = []
    for actual_index, actual_cost in (
        enumerate(costs)
    ):
        distribution = estimate_distribution(
            costs=costs,
            actual_index=actual_index,
            accuracy=accuracy,
            offset_weights=offset_weights,
        )
        comparison_count = 0.0
        regression_successes = 0.0
        faster_successes = 0.0
        positive_delay = 0.0
        request_savings = 0.0
        maximum_delay = 0.0
        for estimated_cost, probability in (
            distribution.items()
        ):
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
                positive_delay += (
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
                request_savings += (
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
        rows.append(
            {
                "cost": actual_cost,
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
                    positive_delay
                    / comparison_count
                ),
                "request_savings_per_comparison": (
                    request_savings
                    / comparison_count
                ),
                "maximum_positive_delay_support_requests": (
                    maximum_delay
                ),
            }
        )
    return rows


def profile_metrics(
    *,
    per_cost: list[dict[str, Any]],
    weights: tuple[float, ...],
) -> dict[str, Any]:
    comparison_count = sum(
        weight
        * row["comparison_count"]
        for weight, row in zip(
            weights,
            per_cost,
        )
    )
    regression_successes = sum(
        weight
        * row["regression_successes"]
        for weight, row in zip(
            weights,
            per_cost,
        )
    )
    faster_successes = sum(
        weight
        * row["faster_successes"]
        for weight, row in zip(
            weights,
            per_cost,
        )
    )
    positive_delay = sum(
        weight
        * row[
            "positive_delay_burden_requests_per_comparison"
        ]
        * row["comparison_count"]
        for weight, row in zip(
            weights,
            per_cost,
        )
    )
    request_savings = sum(
        weight
        * row[
            "request_savings_per_comparison"
        ]
        * row["comparison_count"]
        for weight, row in zip(
            weights,
            per_cost,
        )
    )
    return {
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
            positive_delay
            / comparison_count
        ),
        "request_savings_per_comparison": (
            request_savings
            / comparison_count
        ),
    }


def winner_key(
    *,
    name: str,
    metrics: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    return (
        metrics["regression_fraction"],
        metrics[
            "positive_delay_burden_requests_per_comparison"
        ],
        inventory[name][
            "configuration_segment_count"
        ],
        -metrics["faster_fraction"],
        name,
    )


def winner(
    *,
    metrics: dict[
        str,
        dict[str, Any],
    ],
    inventory: dict[str, dict[str, Any]],
) -> str:
    return min(
        metrics,
        key=lambda name: winner_key(
            name=name,
            metrics=metrics[name],
            inventory=inventory,
        ),
    )


def minimax_regret(
    *,
    per_policy_costs: dict[
        str,
        list[dict[str, Any]],
    ],
    inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cost_oracles = [
        min(
            values[index][
                "regression_fraction"
            ]
            for values
            in per_policy_costs.values()
        )
        for index in range(
            len(
                next(
                    iter(
                        per_policy_costs.values()
                    )
                )
            )
        )
    ]
    summaries = {}
    for name, rows in (
        per_policy_costs.items()
    ):
        regrets = [
            row["regression_fraction"]
            - cost_oracles[index]
            for index, row
            in enumerate(rows)
        ]
        summaries[name] = {
            "maximum_per_cost_regret": (
                max(regrets)
            ),
            "mean_per_cost_regret": (
                sum(regrets)
                / len(regrets)
            ),
            "maximum_per_cost_delay_burden": (
                max(
                    row[
                        "positive_delay_burden_requests_per_comparison"
                    ]
                    for row in rows
                )
            ),
            "mean_per_cost_faster_fraction": (
                sum(
                    row["faster_fraction"]
                    for row in rows
                )
                / len(rows)
            ),
            "per_cost_regret": {
                row["cost"]: regret
                for row, regret
                in zip(
                    rows,
                    regrets,
                )
            },
        }
    selected = min(
        summaries,
        key=lambda name: (
            summaries[name][
                "maximum_per_cost_regret"
            ],
            summaries[name][
                "mean_per_cost_regret"
            ],
            summaries[name][
                "maximum_per_cost_delay_burden"
            ],
            inventory[name][
                "configuration_segment_count"
            ],
            -summaries[name][
                "mean_per_cost_faster_fraction"
            ],
            name,
        ),
    )
    return {
        "policy": selected,
        "summary": summaries[selected],
        "all_policies": summaries,
    }


def compress_grid_intervals(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    intervals = []
    for row in rows:
        state = (
            row["majority_winner"],
            row[
                "minimax_regret_policy"
            ],
        )
        if (
            intervals
            and intervals[-1]["state"]
            == state
        ):
            intervals[-1][
                "end_accuracy"
            ] = row["accuracy"]
        else:
            intervals.append(
                {
                    "start_accuracy": (
                        row["accuracy"]
                    ),
                    "end_accuracy": (
                        row["accuracy"]
                    ),
                    "state": state,
                    "majority_winner": (
                        state[0]
                    ),
                    "minimax_regret_policy": (
                        state[1]
                    ),
                }
            )
    for interval in intervals:
        del interval["state"]
    return intervals


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e88 = load_e88(args.root)
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
    expected_profile_count = 3003
    if len(profiles) != expected_profile_count:
        raise RuntimeError(
            "Unexpected prevalence profile count"
        )
    vertex_count = sum(
        profile["support_size"] == 1
        for profile in profiles
    )
    if vertex_count != len(costs):
        raise RuntimeError(
            "Simplex vertices are incomplete"
        )
    accuracy_grid = [
        index / 100.0
        for index in range(101)
    ]
    uniform_weights = tuple(
        1.0 / len(costs)
        for _ in costs
    )
    kernels = {}
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        grid_rows = []
        representatives = {}
        e88_winners = {
            row["accuracy"]: row[
                "winner"
            ]
            for row in e88["kernels"][
                kernel_name
            ]["accuracy_grid_winners"]
        }
        for accuracy in accuracy_grid:
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
            winner_counts = {
                name: 0
                for name in inventory
            }
            regression_optimal_counts = {
                name: 0
                for name in inventory
            }
            for profile in profiles:
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
                winner_counts[selected] += 1
                best_regression = min(
                    value[
                        "regression_fraction"
                    ]
                    for value
                    in metrics.values()
                )
                for name, value in (
                    metrics.items()
                ):
                    if abs(
                        value[
                            "regression_fraction"
                        ]
                        - best_regression
                    ) < 1e-15:
                        regression_optimal_counts[
                            name
                        ] += 1
            uniform_metrics = {
                name: profile_metrics(
                    per_cost=rows,
                    weights=uniform_weights,
                )
                for name, rows
                in per_policy_costs.items()
            }
            uniform_winner = winner(
                metrics=uniform_metrics,
                inventory=inventory,
            )
            if (
                uniform_winner
                != e88_winners[accuracy]
            ):
                raise RuntimeError(
                    "Uniform prevalence did not "
                    "reproduce E88"
                )
            majority_winner = min(
                winner_counts,
                key=lambda name: (
                    -winner_counts[name],
                    inventory[name][
                        "configuration_segment_count"
                    ],
                    name,
                ),
            )
            minimax = minimax_regret(
                per_policy_costs=(
                    per_policy_costs
                ),
                inventory=inventory,
            )
            row = {
                "accuracy": accuracy,
                "uniform_winner": (
                    uniform_winner
                ),
                "majority_winner": (
                    majority_winner
                ),
                "minimax_regret_policy": (
                    minimax["policy"]
                ),
                "winner_counts": (
                    winner_counts
                ),
                "winner_fractions": {
                    name: count
                    / len(profiles)
                    for name, count
                    in winner_counts.items()
                },
                "regression_optimal_counts": (
                    regression_optimal_counts
                ),
                "regression_optimal_fractions": {
                    name: count
                    / len(profiles)
                    for name, count
                    in regression_optimal_counts.items()
                },
                "minimax_regret": {
                    "policy": minimax[
                        "policy"
                    ],
                    "summary": minimax[
                        "summary"
                    ],
                },
                "e85_maximum_per_cost_regret": (
                    minimax[
                        "all_policies"
                    ][
                        "e85_compressed_8key"
                    ][
                        "maximum_per_cost_regret"
                    ]
                ),
            }
            grid_rows.append(row)
            if accuracy in (
                REPRESENTATIVE_ACCURACIES
            ):
                representatives[
                    f"{accuracy:g}"
                ] = row
        e85_coverage = {}
        for target in (
            E85_COVERAGE_TARGETS
        ):
            selected = next(
                (
                    row
                    for row in grid_rows
                    if row[
                        "winner_fractions"
                    ][
                        "e85_compressed_8key"
                    ]
                    >= target
                ),
                None,
            )
            e85_coverage[
                f"{target:.2f}"
            ] = (
                None
                if selected is None
                else selected["accuracy"]
            )
        maximum_e85 = max(
            grid_rows,
            key=lambda row: (
                row[
                    "winner_fractions"
                ][
                    "e85_compressed_8key"
                ],
                row["accuracy"],
            ),
        )
        kernels[kernel_name] = {
            "description": kernel[
                "description"
            ],
            "grid_intervals": (
                compress_grid_intervals(
                    grid_rows
                )
            ),
            "accuracy_grid": grid_rows,
            "representative_accuracies": (
                representatives
            ),
            "e85_winner_coverage_thresholds": (
                e85_coverage
            ),
            "maximum_e85_winner_coverage": {
                "accuracy": (
                    maximum_e85[
                        "accuracy"
                    ]
                ),
                "winner_count": (
                    maximum_e85[
                        "winner_counts"
                    ][
                        "e85_compressed_8key"
                    ]
                ),
                "winner_fraction": (
                    maximum_e85[
                        "winner_fractions"
                    ][
                        "e85_compressed_8key"
                    ]
                ),
                "regression_optimal_count": (
                    maximum_e85[
                        "regression_optimal_counts"
                    ][
                        "e85_compressed_8key"
                    ]
                ),
                "regression_optimal_fraction": (
                    maximum_e85[
                        "regression_optimal_fractions"
                    ][
                        "e85_compressed_8key"
                    ]
                ),
            },
        }
    result = {
        "suite": (
            "pacing_coalescing_order_prevalence"
        ),
        "model": (
            "Sweep every six-cell cost "
            "prevalence vector on the 10% "
            "simplex for each E88 estimator "
            "kernel and 1% accuracy point. "
            "Report prevalence-grid winners "
            "and distribution-free maximum "
            "per-cost regret."
        ),
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            ),
            "accuracy_frontier": (
                "results/raw/"
                "pacing_coalescing_order_accuracy_frontier/"
                "pacing_coalescing_order_accuracy_frontier_full.json"
            ),
        },
        "selection_objective": [
            "minimum expected regression fraction",
            "minimum expected positive-delay burden",
            "minimum contiguous configuration segments",
            "maximum expected faster-action fraction",
            "deterministic policy name",
        ],
        "minimax_regret_objective": [
            "minimum maximum per-cost regression regret",
            "minimum mean per-cost regression regret",
            "minimum maximum per-cost delay burden",
            "minimum contiguous configuration segments",
            "maximum mean per-cost faster-action fraction",
            "deterministic policy name",
        ],
        "costs": costs,
        "prevalence_grid": {
            "denominator": (
                PREVALENCE_DENOMINATOR
            ),
            "step": (
                1.0
                / PREVALENCE_DENOMINATOR
            ),
            "profile_count": len(
                profiles
            ),
            "vertex_count": vertex_count,
            "support_size_histogram": {
                str(size): sum(
                    profile[
                        "support_size"
                    ]
                    == size
                    for profile in profiles
                )
                for size in range(
                    1,
                    len(costs) + 1,
                )
            },
        },
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
        "headline": {
            kernel_name: {
                "grid_intervals": values[
                    "grid_intervals"
                ],
                "e85_winner_coverage_thresholds": (
                    values[
                        "e85_winner_coverage_thresholds"
                    ]
                ),
                "maximum_e85_winner_coverage": (
                    values[
                        "maximum_e85_winner_coverage"
                    ]
                ),
                "representative_winners": {
                    accuracy: {
                        "uniform_winner": row[
                            "uniform_winner"
                        ],
                        "majority_winner": row[
                            "majority_winner"
                        ],
                        "minimax_regret_policy": (
                            row[
                                "minimax_regret_policy"
                            ]
                        ),
                        "e85_winner_fraction": (
                            row[
                                "winner_fractions"
                            ][
                                "e85_compressed_8key"
                            ]
                        ),
                        "e85_regression_optimal_fraction": (
                            row[
                                "regression_optimal_fractions"
                            ][
                                "e85_compressed_8key"
                            ]
                        ),
                    }
                    for accuracy, row
                    in values[
                        "representative_accuracies"
                    ].items()
                },
            }
            for kernel_name, values
            in kernels.items()
        },
        "validation": {
            "prevalence_profile_count": (
                len(profiles)
            ),
            "simplex_vertices": (
                vertex_count
            ),
            "all_profiles_normalized": (
                True
            ),
            "uniform_prevalence_reproduces_e88": (
                True
            ),
            "accuracy_grid_count": len(
                accuracy_grid
            ),
            "kernel_count": len(
                kernels
            ),
            "policy_count": len(
                inventory
            ),
            "winner_counts_sum_to_profile_count": (
                all(
                    sum(
                        row[
                            "winner_counts"
                        ].values()
                    )
                    == len(profiles)
                    for values
                    in kernels.values()
                    for row in values[
                        "accuracy_grid"
                    ]
                )
            ),
            "regression_optimal_counts_bounded": (
                all(
                    all(
                        0 <= count
                        <= len(profiles)
                        for count in row[
                            "regression_optimal_counts"
                        ].values()
                    )
                    for values
                    in kernels.values()
                    for row in values[
                        "accuracy_grid"
                    ]
                )
            ),
            "e85_zero_max_regret_at_accuracy_one": (
                all(
                    abs(
                        values[
                            "accuracy_grid"
                        ][-1][
                            "e85_maximum_per_cost_regret"
                        ]
                    )
                    < 1e-15
                    for values
                    in kernels.values()
                )
            ),
        },
        "limitations": [
            "The 10% simplex grid is exhaustive at that resolution but not a continuous prevalence-volume calculation.",
            "Winner fractions count lattice profiles equally rather than weighting them by a deployment prior.",
            "The estimator kernels remain sensitivity models rather than calibrated confusion matrices.",
            "Expected regression is primary; worst-cell constraints can select different policies.",
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
