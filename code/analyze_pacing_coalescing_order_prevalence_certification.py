from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_pacing_coalescing_order_accuracy_frontier import (
    KERNELS,
    policy_inventory,
)
from analyze_pacing_coalescing_order_cost_uncertainty import (
    load_e83,
)
from analyze_pacing_coalescing_order_prevalence import (
    per_cost_metrics,
    prevalence_profiles,
    profile_metrics,
    winner,
)


def validate_sample_bound(
    *,
    radius: float,
    confidence: float,
    category_count: int,
    samples: int | None,
) -> None:
    if samples in (None, 0):
        return
    alpha = 1.0 - confidence
    prefactor = (
        2**category_count - 2
    )
    bound = prefactor * math.exp(
        -samples
        * radius
        * radius
        / 2.0
    )
    if bound > alpha + 1e-12:
        raise RuntimeError(
            "Concentration inversion failed"
        )


CONFIDENCE_LEVELS = (
    0.95,
    0.99,
)
SAMPLE_BUDGETS = (
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
)
COVERAGE_TARGETS = (
    0.50,
    0.75,
    0.90,
)
REPRESENTATIVE_ACCURACIES = (
    0.0,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
)
PREVALENCE_DENOMINATOR = 10
TOLERANCE = 1e-14


def load_e89(
    root: Path,
) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_coalescing_order_prevalence"
        / "pacing_coalescing_order_prevalence_full.json"
    )
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def pairwise_boundary_distance_l1(
    *,
    prevalence: tuple[float, ...],
    differences: list[float],
    include_ties: bool = False,
) -> float:
    margin = sum(
        weight * difference
        for weight, difference
        in zip(
            prevalence,
            differences,
        )
    )
    if margin < -TOLERANCE:
        raise RuntimeError(
            "Selected policy is not "
            "regression-optimal"
        )
    minimum = min(differences)
    if (
        minimum > TOLERANCE
        or (
            not include_ties
            and minimum
            >= -TOLERANCE
        )
    ):
        return math.inf
    if margin <= TOLERANCE:
        return 0.0
    sources = [
        [
            differences[index],
            prevalence[index],
        ]
        for index
        in range(len(prevalence))
        if prevalence[index] > 0.0
    ]
    destinations = [
        [
            differences[index],
            1.0 - prevalence[index],
        ]
        for index
        in range(len(prevalence))
        if prevalence[index] < 1.0
    ]
    sources.sort(
        key=lambda item: item[0],
        reverse=True,
    )
    destinations.sort(
        key=lambda item: item[0]
    )
    source_index = 0
    destination_index = 0
    remaining_margin = margin
    moved_mass = 0.0
    while (
        remaining_margin > TOLERANCE
        and source_index
        < len(sources)
        and destination_index
        < len(destinations)
    ):
        source_value, source_mass = (
            sources[source_index]
        )
        (
            destination_value,
            destination_capacity,
        ) = destinations[
            destination_index
        ]
        reduction_per_mass = (
            source_value
            - destination_value
        )
        if reduction_per_mass <= TOLERANCE:
            break
        moved = min(
            source_mass,
            destination_capacity,
            (
                remaining_margin
                / reduction_per_mass
            ),
        )
        remaining_margin -= (
            moved
            * reduction_per_mass
        )
        moved_mass += moved
        sources[source_index][1] -= moved
        destinations[
            destination_index
        ][1] -= moved
        if (
            sources[source_index][1]
            <= TOLERANCE
        ):
            source_index += 1
        if (
            destinations[
                destination_index
            ][1]
            <= TOLERANCE
        ):
            destination_index += 1
    if remaining_margin > 1e-10:
        raise RuntimeError(
            "Regression boundary should be "
            "reachable but was not"
        )
    distance = 2.0 * moved_mass
    if (
        not math.isinf(distance)
        and not (
            -TOLERANCE
            <= distance
            <= 2.0 + TOLERANCE
        )
    ):
        raise RuntimeError(
            "L1 stability radius is invalid"
        )
    return distance


def boundary_distance_l1(
    *,
    prevalence: tuple[float, ...],
    selected: str,
    per_policy_costs: dict[
        str,
        list[dict[str, Any]],
    ],
) -> float:
    selected_risk = [
        row["regression_fraction"]
        for row in per_policy_costs[
            selected
        ]
    ]
    best = math.inf
    for competitor, rows in (
        per_policy_costs.items()
    ):
        if competitor == selected:
            continue
        differences = [
            row["regression_fraction"]
            - selected_risk[index]
            for index, row
            in enumerate(rows)
        ]
        best = min(
            best,
            pairwise_boundary_distance_l1(
                prevalence=prevalence,
                differences=differences,
            ),
        )
    return best


def required_samples(
    *,
    radius: float,
    confidence: float,
    category_count: int,
) -> int | None:
    if math.isinf(radius):
        return 0
    if radius <= 0.0:
        return None
    alpha = 1.0 - confidence
    prefactor = (
        2**category_count - 2
    )
    return math.ceil(
        (
            2.0
            / (radius * radius)
        )
        * math.log(
            prefactor / alpha
        )
    )


def quantile(
    values: list[float | int],
    probability: float,
) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(
        probability * len(ordered)
    ) - 1
    index = min(
        len(ordered) - 1,
        max(0, index),
    )
    return ordered[index]


def radius_summary(
    *,
    records: list[dict[str, Any]],
    profile_count: int,
) -> dict[str, Any]:
    finite_radii = [
        record["radius_l1"]
        for record in records
        if record["radius_kind"]
        == "finite"
    ]
    zero_count = sum(
        record["radius_kind"] == "zero"
        for record in records
    )
    infinite_count = sum(
        record["radius_kind"]
        == "infinite"
        for record in records
    )
    result = {
        "profile_count": profile_count,
        "zero_radius_count": zero_count,
        "zero_radius_fraction": (
            zero_count / profile_count
        ),
        "finite_radius_count": len(
            finite_radii
        ),
        "finite_radius_fraction": (
            len(finite_radii)
            / profile_count
        ),
        "infinite_radius_count": (
            infinite_count
        ),
        "infinite_radius_fraction": (
            infinite_count
            / profile_count
        ),
        "finite_radius_quantiles": {
            "p10": quantile(
                finite_radii,
                0.10,
            ),
            "p50": quantile(
                finite_radii,
                0.50,
            ),
            "p90": quantile(
                finite_radii,
                0.90,
            ),
            "p95": quantile(
                finite_radii,
                0.95,
            ),
            "minimum": (
                min(finite_radii)
                if finite_radii
                else None
            ),
            "maximum": (
                max(finite_radii)
                if finite_radii
                else None
            ),
        },
        "confidence": {},
    }
    for confidence in CONFIDENCE_LEVELS:
        key = f"{confidence:.2f}"
        sample_values = [
            record["required_samples"][
                key
            ]
            for record in records
            if (
                record["required_samples"][
                    key
                ]
                not in (None, 0)
            )
        ]
        coverage = {}
        for budget in SAMPLE_BUDGETS:
            covered = sum(
                (
                    record[
                        "required_samples"
                    ][key]
                    is not None
                    and record[
                        "required_samples"
                    ][key]
                    <= budget
                )
                for record in records
            )
            coverage[str(budget)] = {
                "count": covered,
                "fraction": (
                    covered / profile_count
                ),
            }
        result["confidence"][key] = {
            "finite_sample_quantiles": {
                "p50": quantile(
                    sample_values,
                    0.50,
                ),
                "p90": quantile(
                    sample_values,
                    0.90,
                ),
                "p95": quantile(
                    sample_values,
                    0.95,
                ),
                "p99": quantile(
                    sample_values,
                    0.99,
                ),
                "maximum": (
                    max(sample_values)
                    if sample_values
                    else None
                ),
            },
            "budget_coverage": coverage,
        }
    return result


def policy_summary(
    *,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    policies = sorted(
        {
            record["policy"]
            for record in records
        }
    )
    for policy in policies:
        selected = [
            record
            for record in records
            if record["policy"] == policy
        ]
        result[policy] = (
            radius_summary(
                records=selected,
                profile_count=len(selected),
            )
        )
    return result


def coverage_thresholds(
    *,
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for confidence in (
        CONFIDENCE_LEVELS
    ):
        confidence_key = (
            f"{confidence:.2f}"
        )
        result[confidence_key] = {}
        for budget in SAMPLE_BUDGETS:
            result[confidence_key][
                str(budget)
            ] = {}
            for target in (
                COVERAGE_TARGETS
            ):
                selected = next(
                    (
                        row["accuracy"]
                        for row in grid
                        if row["summary"][
                            "confidence"
                        ][confidence_key][
                            "budget_coverage"
                        ][str(budget)][
                            "fraction"
                        ]
                        >= target
                    ),
                    None,
                )
                sustained = next(
                    (
                        row["accuracy"]
                        for index, row
                        in enumerate(grid)
                        if all(
                            later["summary"][
                                "confidence"
                            ][confidence_key][
                                "budget_coverage"
                            ][str(budget)][
                                "fraction"
                            ]
                            >= target
                            for later
                            in grid[index:]
                        )
                    ),
                    None,
                )
                result[confidence_key][
                    str(budget)
                ][f"{target:.2f}"] = {
                    "first_accuracy": (
                        selected
                    ),
                    "sustained_accuracy": (
                        sustained
                    ),
                }
    return result


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e83 = load_e83(args.root)
    e89 = load_e89(args.root)
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
    if len(profiles) != 3003:
        raise RuntimeError(
            "Unexpected prevalence profile count"
        )
    accuracy_grid = [
        index / 100.0
        for index in range(101)
    ]
    kernels = {}
    for kernel_name, kernel in (
        KERNELS.items()
    ):
        e89_grid = {
            row["accuracy"]: row
            for row in e89["kernels"][
                kernel_name
            ]["accuracy_grid"]
        }
        grid = []
        representatives = {}
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
            records = []
            winner_counts = {
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
                radius = boundary_distance_l1(
                    prevalence=profile[
                        "weights"
                    ],
                    selected=selected,
                    per_policy_costs=(
                        per_policy_costs
                    ),
                )
                if math.isinf(radius):
                    radius_kind = "infinite"
                elif radius <= TOLERANCE:
                    radius = 0.0
                    radius_kind = "zero"
                else:
                    radius_kind = "finite"
                records.append(
                    {
                        "policy": selected,
                        "radius_l1": radius,
                        "radius_kind": (
                            radius_kind
                        ),
                        "required_samples": {
                            f"{confidence:.2f}": (
                                required_samples(
                                    radius=radius,
                                    confidence=(
                                        confidence
                                    ),
                                    category_count=(
                                        len(costs)
                                    ),
                                )
                            )
                            for confidence
                            in CONFIDENCE_LEVELS
                        },
                    }
                )
                for confidence in (
                    CONFIDENCE_LEVELS
                ):
                    confidence_key = (
                        f"{confidence:.2f}"
                    )
                    validate_sample_bound(
                        radius=radius,
                        confidence=(
                            confidence
                        ),
                        category_count=(
                            len(costs)
                        ),
                        samples=records[-1][
                            "required_samples"
                        ][confidence_key],
                    )
            if (
                winner_counts
                != e89_grid[accuracy][
                    "winner_counts"
                ]
            ):
                raise RuntimeError(
                    "E89 winner counts were "
                    "not reproduced"
                )
            summary = radius_summary(
                records=records,
                profile_count=len(
                    profiles
                ),
            )
            by_policy = policy_summary(
                records=records
            )
            row = {
                "accuracy": accuracy,
                "summary": summary,
            }
            grid.append(row)
            if accuracy in (
                REPRESENTATIVE_ACCURACIES
            ):
                representatives[
                    f"{accuracy:g}"
                ] = {
                    "accuracy": accuracy,
                    "summary": summary,
                    "by_policy": by_policy,
                }
        kernels[kernel_name] = {
            "description": kernel[
                "description"
            ],
            "accuracy_grid": grid,
            "representative_accuracies": (
                representatives
            ),
            "coverage_thresholds": (
                coverage_thresholds(
                    grid=grid
                )
            ),
        }
    perfect = kernels[
        "symmetric_radius_1"
    ]["representative_accuracies"]["1"][
        "summary"
    ]
    if (
        perfect[
            "infinite_radius_count"
        ]
        != 2827
        or perfect["zero_radius_count"]
        != 176
    ):
        raise RuntimeError(
            "Perfect-accuracy certification "
            "partition changed"
        )
    result = {
        "suite": (
            "pacing_coalescing_order_prevalence_certification"
        ),
        "model": (
            "Certify the E89 prevalence-selected "
            "policy from a finite multinomial "
            "sample. The stability radius is the "
            "exact minimum L1 distance from the "
            "true six-cell prevalence to a "
            "competitor regression boundary."
        ),
        "source": {
            "member_order_frontier": (
                "results/raw/"
                "pacing_coalescing_order/"
                "pacing_coalescing_order_full.json"
            ),
            "prevalence_audit": (
                "results/raw/"
                "pacing_coalescing_order_prevalence/"
                "pacing_coalescing_order_prevalence_full.json"
            ),
        },
        "confidence_bound": {
            "name": (
                "multinomial L1 concentration"
            ),
            "formula": (
                "P(||p_hat-p||_1 >= eps) <= "
                "(2^k-2) * exp(-n*eps^2/2)"
            ),
            "category_count": len(costs),
            "confidence_levels": list(
                CONFIDENCE_LEVELS
            ),
            "sample_budgets": list(
                SAMPLE_BUDGETS
            ),
        },
        "costs": costs,
        "prevalence_grid": {
            "denominator": (
                PREVALENCE_DENOMINATOR
            ),
            "profile_count": len(
                profiles
            ),
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
                "coverage_thresholds": values[
                    "coverage_thresholds"
                ],
                "representative_accuracies": {
                    accuracy: {
                        "summary": row[
                            "summary"
                        ],
                        "e85": row[
                            "by_policy"
                        ].get(
                            "e85_compressed_8key"
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
            "kernel_count": len(
                kernels
            ),
            "accuracy_grid_count": len(
                accuracy_grid
            ),
            "prevalence_profile_count": (
                len(profiles)
            ),
            "simplex_vertex_count": sum(
                profile["support_size"] == 1
                for profile in profiles
            ),
            "e89_winner_counts_reproduced": (
                True
            ),
            "all_radii_in_valid_range": True,
            "sample_bound_inversion_valid": (
                True
            ),
            "perfect_accuracy_infinite_radius_count": (
                perfect[
                    "infinite_radius_count"
                ]
            ),
            "perfect_accuracy_zero_radius_count": (
                perfect[
                    "zero_radius_count"
                ]
            ),
            "perfect_accuracy_partition_matches_e89": (
                True
            ),
        },
        "limitations": [
            "The multinomial concentration bound is distribution-free and conservative; exact multinomial confidence regions can require fewer samples.",
            "Policy identity is uncertifiable at zero-radius prevalence boundaries even when several policies have identical regression.",
            "Infinite radius means no tested competitor has lower regression at any cost prevalence, not that every untested policy is dominated.",
            "The 10% prevalence simplex and E88 sensitivity kernels are not calibrated deployment distributions.",
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
