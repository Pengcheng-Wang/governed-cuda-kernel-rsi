from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable


FAMILIES: dict[
    str,
    tuple[str, ...],
] = {
    "universal": (),
    "burst": ("burst_mode",),
    "load": ("offered_load",),
    "cache": ("prefix_mode",),
    "burst_load": (
        "burst_mode",
        "offered_load",
    ),
    "burst_cache": (
        "burst_mode",
        "prefix_mode",
    ),
    "load_cache": (
        "offered_load",
        "prefix_mode",
    ),
    "full": (
        "burst_mode",
        "offered_load",
        "prefix_mode",
    ),
}

SELECTION_MODES = (
    "robust",
    "median",
)


def load_e62(
    root: Path,
) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "slo_ingress_pacing"
        / "slo_ingress_pacing_full.json"
    )
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def state_key(
    record: dict[str, Any],
) -> tuple[str, float, str]:
    return (
        record["prefix_mode"],
        float(
            record[
                "offered_load_vs_decode_capacity"
            ]
        ),
        record["burst_mode"],
    )


def state_dict(
    state: tuple[str, float, str],
) -> dict[str, Any]:
    return {
        "prefix_mode": state[0],
        "offered_load": state[1],
        "burst_mode": state[2],
    }


def group_key(
    state: tuple[str, float, str],
    dimensions: tuple[str, ...],
) -> tuple[Any, ...]:
    values = state_dict(state)
    return tuple(
        values[dimension]
        for dimension in dimensions
    )


def group_dict(
    dimensions: tuple[str, ...],
    key: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        dimension: value
        for dimension, value in zip(
            dimensions,
            key,
        )
    }


def index_records(
    e62: dict[str, Any],
) -> tuple[
    list[tuple[str, float, str]],
    list[float],
    dict[
        tuple[str, float, str, float],
        dict[str, Any],
    ],
]:
    records = e62["records"]
    states = sorted(
        {
            state_key(record)
            for record in records
        }
    )
    windows = sorted(
        {
            float(
                record[
                    "requested_pacing_window_ms"
                ]
            )
            for record in records
        }
    )
    lookup = {
        (
            *state_key(record),
            float(
                record[
                    "requested_pacing_window_ms"
                ]
            ),
        ): record
        for record in records
    }
    return states, windows, lookup


def select_family(
    *,
    family: str,
    mode: str,
    slo_ms: float,
    states: list[
        tuple[str, float, str]
    ],
    windows: list[float],
    lookup: dict[
        tuple[str, float, str, float],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    dimensions = FAMILIES[family]
    groups: dict[
        tuple[Any, ...],
        list[tuple[str, float, str]],
    ] = {}
    for state in states:
        groups.setdefault(
            group_key(
                state,
                dimensions,
            ),
            [],
        ).append(state)
    selected: dict[
        tuple[Any, ...],
        float,
    ] = {}
    group_records = []
    statistic = (
        "maximum"
        if mode == "robust"
        else "median"
    )
    for key, members in groups.items():
        feasible = [
            window
            for window in windows
            if all(
                lookup[
                    (*state, window)
                ]["metrics"][
                    "p99_ttft_ms"
                ][statistic]
                <= slo_ms + 1e-9
                for state in members
            )
        ]
        if not feasible:
            return {
                "family": family,
                "dimensions": list(
                    dimensions
                ),
                "selection_mode": mode,
                "ttft_slo_ms": slo_ms,
                "status": "infeasible",
                "key_count": len(groups),
            }
        best = max(
            feasible,
            key=lambda window: (
                sum(
                    lookup[
                        (*state, window)
                    ]["metrics"][
                        "throughput_output_"
                        "tokens_s"
                    ]["median"]
                    for state in members
                ),
                -max(
                    lookup[
                        (*state, window)
                    ]["metrics"][
                        "p99_ttft_ms"
                    ]["maximum"]
                    for state in members
                ),
                -window,
            ),
        )
        selected[key] = best
        group_records.append(
            {
                "key": group_dict(
                    dimensions,
                    key,
                ),
                "member_states": [
                    state_dict(state)
                    for state in members
                ],
                "selected_window_ms": best,
                "selection_statistic": (
                    statistic
                ),
                "selection_p99_ttft_ms": max(
                    lookup[
                        (*state, best)
                    ]["metrics"][
                        "p99_ttft_ms"
                    ][statistic]
                    for state in members
                ),
                "maximum_p99_ttft_ms": max(
                    lookup[
                        (*state, best)
                    ]["metrics"][
                        "p99_ttft_ms"
                    ]["maximum"]
                    for state in members
                ),
                "median_throughput_sum": sum(
                    lookup[
                        (*state, best)
                    ]["metrics"][
                        "throughput_output_"
                        "tokens_s"
                    ]["median"]
                    for state in members
                ),
            }
        )
    return {
        "family": family,
        "dimensions": list(dimensions),
        "selection_mode": mode,
        "ttft_slo_ms": slo_ms,
        "status": "feasible",
        "key_count": len(groups),
        "groups": group_records,
        "selected": selected,
    }


def evaluate_family(
    *,
    selection: dict[str, Any],
    oracle_windows: dict[
        tuple[str, float, str],
        float,
    ],
    states: list[
        tuple[str, float, str]
    ],
    lookup: dict[
        tuple[str, float, str, float],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    if selection["status"] != "feasible":
        return selection
    dimensions = tuple(
        selection["dimensions"]
    )
    selected = selection.pop("selected")
    state_records = []
    throughput_ratios = []
    acceptance_deltas = []
    violation_count = 0
    scenario_count = 0
    violating_states = 0
    maximum_excess_ms = 0.0
    for state in states:
        key = group_key(
            state,
            dimensions,
        )
        window = selected[key]
        oracle_window = oracle_windows[
            state
        ]
        record = lookup[
            (*state, window)
        ]
        oracle = lookup[
            (*state, oracle_window)
        ]
        throughput = record["metrics"][
            "throughput_output_tokens_s"
        ]["median"]
        oracle_throughput = oracle[
            "metrics"
        ][
            "throughput_output_tokens_s"
        ]["median"]
        throughput_ratio = (
            throughput / oracle_throughput
        )
        throughput_ratios.append(
            throughput_ratio
        )
        acceptance_delta = (
            record["metrics"][
                "accepted_fraction"
            ]["median"]
            - oracle["metrics"][
                "accepted_fraction"
            ]["median"]
        )
        acceptance_deltas.append(
            acceptance_delta
        )
        state_violations = 0
        state_maximum_excess = 0.0
        for scenario in record[
            "scenarios"
        ]:
            excess = max(
                0.0,
                float(
                    scenario[
                        "p99_ttft_ms"
                    ]
                )
                - selection[
                    "ttft_slo_ms"
                ],
            )
            state_violations += int(
                excess > 0.0
            )
            state_maximum_excess = max(
                state_maximum_excess,
                excess,
            )
        violation_count += (
            state_violations
        )
        scenario_count += len(
            record["scenarios"]
        )
        violating_states += int(
            state_violations > 0
        )
        maximum_excess_ms = max(
            maximum_excess_ms,
            state_maximum_excess,
        )
        state_records.append(
            {
                **state_dict(state),
                "selected_window_ms": (
                    window
                ),
                "oracle_window_ms": (
                    oracle_window
                ),
                "throughput_output_tokens_s": (
                    throughput
                ),
                "oracle_throughput_output_tokens_s": (
                    oracle_throughput
                ),
                "throughput_ratio_vs_oracle": (
                    throughput_ratio
                ),
                "accepted_fraction": (
                    record["metrics"][
                        "accepted_fraction"
                    ]["median"]
                ),
                "oracle_accepted_fraction": (
                    oracle["metrics"][
                        "accepted_fraction"
                    ]["median"]
                ),
                "acceptance_delta_vs_oracle": (
                    acceptance_delta
                ),
                "median_p99_ttft_ms": (
                    record["metrics"][
                        "p99_ttft_ms"
                    ]["median"]
                ),
                "maximum_p99_ttft_ms": (
                    record["metrics"][
                        "p99_ttft_ms"
                    ]["maximum"]
                ),
                "state_scenario_violation_count": (
                    state_violations
                ),
                "state_maximum_slo_excess_ms": (
                    state_maximum_excess
                ),
            }
        )
    return {
        **selection,
        "mean_throughput_ratio_vs_oracle": (
            statistics.mean(
                throughput_ratios
            )
        ),
        "geomean_throughput_ratio_vs_oracle": (
            math.exp(
                statistics.mean(
                    math.log(value)
                    for value
                    in throughput_ratios
                )
            )
        ),
        "worst_throughput_ratio_vs_oracle": (
            min(throughput_ratios)
        ),
        "mean_acceptance_delta_vs_oracle": (
            statistics.mean(
                acceptance_deltas
            )
        ),
        "worst_acceptance_delta_vs_oracle": (
            min(acceptance_deltas)
        ),
        "scenario_count": scenario_count,
        "scenario_slo_violation_count": (
            violation_count
        ),
        "scenario_slo_violation_fraction": (
            violation_count
            / scenario_count
        ),
        "violating_state_count": (
            violating_states
        ),
        "maximum_slo_excess_ms": (
            maximum_excess_ms
        ),
        "states": state_records,
    }


def oracle_windows_from_e62(
    *,
    e62: dict[str, Any],
    slo_ms: float,
) -> dict[
    tuple[str, float, str],
    float,
]:
    matches = [
        item
        for item in e62["slo_frontier"]
        if float(
            item["ttft_slo_ms"]
        )
        == slo_ms
    ]
    return {
        (
            item["prefix_mode"],
            float(
                item[
                    "offered_load_vs_decode_capacity"
                ]
            ),
            item["burst_mode"],
        ): float(
            item[
                "requested_pacing_window_ms"
            ]
        )
        for item in matches
        if item["status"] == "feasible"
    }


def pareto_frontier(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    feasible = [
        record
        for record in records
        if record["status"] == "feasible"
    ]
    frontier = []
    for record in feasible:
        dominated = any(
            other["key_count"]
            <= record["key_count"]
            and other[
                "mean_throughput_ratio_vs_oracle"
            ]
            >= record[
                "mean_throughput_ratio_vs_oracle"
            ]
            and other[
                "worst_throughput_ratio_vs_oracle"
            ]
            >= record[
                "worst_throughput_ratio_vs_oracle"
            ]
            and (
                other["key_count"]
                < record["key_count"]
                or other[
                    "mean_throughput_ratio_vs_oracle"
                ]
                > record[
                    "mean_throughput_ratio_vs_oracle"
                ]
                or other[
                    "worst_throughput_ratio_vs_oracle"
                ]
                > record[
                    "worst_throughput_ratio_vs_oracle"
                ]
            )
            for other in feasible
            if other is not record
        )
        if not dominated:
            frontier.append(
                {
                    "family": record[
                        "family"
                    ],
                    "key_count": record[
                        "key_count"
                    ],
                    "mean_throughput_ratio_vs_oracle": (
                        record[
                            "mean_throughput_ratio_vs_oracle"
                        ]
                    ),
                    "worst_throughput_ratio_vs_oracle": (
                        record[
                            "worst_throughput_ratio_vs_oracle"
                        ]
                    ),
                }
            )
    return sorted(
        frontier,
        key=lambda item: (
            item["key_count"],
            -item[
                "mean_throughput_ratio_vs_oracle"
            ],
            -item[
                "worst_throughput_ratio_vs_oracle"
            ],
            item["family"],
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    e62 = load_e62(args.root)
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
    unknown_families = set(
        families
    ) - set(FAMILIES)
    if unknown_families:
        raise ValueError(
            f"Unknown families: "
            f"{sorted(unknown_families)}"
        )
    records = []
    compression_frontiers = {}
    oracle_validation = []
    for slo_ms in slos_ms:
        oracle_windows = (
            oracle_windows_from_e62(
                e62=e62,
                slo_ms=slo_ms,
            )
        )
        if set(oracle_windows) != set(
            states
        ):
            raise RuntimeError(
                "E62 oracle state mismatch"
            )
        robust_records = []
        for mode in SELECTION_MODES:
            for family in families:
                selection = select_family(
                    family=family,
                    mode=mode,
                    slo_ms=slo_ms,
                    states=states,
                    windows=windows,
                    lookup=lookup,
                )
                evaluated = evaluate_family(
                    selection=selection,
                    oracle_windows=(
                        oracle_windows
                    ),
                    states=states,
                    lookup=lookup,
                )
                records.append(evaluated)
                if mode == "robust":
                    robust_records.append(
                        evaluated
                    )
                if (
                    mode == "robust"
                    and family == "full"
                ):
                    if (
                        evaluated["status"]
                        != "feasible"
                    ):
                        raise RuntimeError(
                            "Full robust policy "
                            "became infeasible"
                        )
                    for state_record in (
                        evaluated["states"]
                    ):
                        state = (
                            state_record[
                                "prefix_mode"
                            ],
                            float(
                                state_record[
                                    "offered_load"
                                ]
                            ),
                            state_record[
                                "burst_mode"
                            ],
                        )
                        expected = (
                            oracle_windows[
                                state
                            ]
                        )
                        difference = abs(
                            state_record[
                                "selected_window_ms"
                            ]
                            - expected
                        )
                        if difference > 1e-12:
                            raise RuntimeError(
                                "E62 oracle "
                                "reproduction failed"
                            )
                        oracle_validation.append(
                            {
                                "ttft_slo_ms": (
                                    slo_ms
                                ),
                                **state_dict(
                                    state
                                ),
                                "selected_window_ms": (
                                    state_record[
                                        "selected_window_ms"
                                    ]
                                ),
                                "expected_window_ms": (
                                    expected
                                ),
                                "absolute_difference_ms": (
                                    difference
                                ),
                            }
                        )
        compression_frontiers[
            str(int(slo_ms))
        ] = pareto_frontier(
            robust_records
        )

    result = {
        "suite": (
            "pacing_policy_compression_"
            "analysis"
        ),
        "model": (
            "Compress the E62 robust per-state "
            "pacing oracle into universal, "
            "one-dimensional, two-dimensional, "
            "and full lookup policies; compare "
            "robust maximum-p99 selection with "
            "median-p99 tuning."
        ),
        "source": {
            "path": (
                "results/raw/"
                "slo_ingress_pacing/"
                "slo_ingress_pacing_full.json"
            ),
            "suite": e62["suite"],
        },
        "state_dimensions": [
            "prefix_mode",
            "offered_load",
            "burst_mode",
        ],
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
        "selection_modes": {
            "robust": (
                "Every state in a policy key "
                "must have maximum p99 TTFT "
                "within the SLO."
            ),
            "median": (
                "Every state in a policy key "
                "must have median p99 TTFT "
                "within the SLO; all six "
                "scenarios are then evaluated "
                "for hidden violations."
            ),
        },
        "records": records,
        "compression_frontiers": (
            compression_frontiers
        ),
        "validation": {
            "e62_oracle_checks": (
                oracle_validation
            ),
            "e62_oracle_maximum_window_difference_ms": (
                max(
                    item[
                        "absolute_difference_ms"
                    ]
                    for item
                    in oracle_validation
                )
            ),
            "expected_record_count": (
                len(slos_ms)
                * len(SELECTION_MODES)
                * len(families)
            ),
            "accounting": (
                "Every selected state uses an "
                "existing E62 candidate window. "
                "Robust policies cannot violate "
                "the SLO by construction. "
                "Median policies are evaluated "
                "against all six underlying "
                "scenarios. Throughput retention "
                "is measured against the exact "
                "E62 robust full-state oracle."
            ),
        },
        "limitations": [
            "The analysis inherits E62's finite candidate grid, synthetic states, and fitted-service simulation.",
            "Policy keys are exact categorical states rather than noisy online estimates of load, cache rate, or burst size.",
            "The group objective maximizes the sum of median throughput, not tenant utility, cost, or weighted traffic prevalence.",
            "The three SLOs and eight E62 states are a compact systems study, not a production policy-training dataset.",
            "Median tuning is a deliberate unsafe control; a production controller should use confidence bounds or online safety margins.",
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
