from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analyze_pacing_alpha_allocation import (
    allocation_grid,
    evaluate_allocation,
    load_e68,
    precompute_metric_looks,
    summarize,
    weighted_drift_crossing,
)


RATIO = 2.0
PHASE_DENOMINATOR = 8
ROUNDING_MODES = (
    "ceil",
    "nearest",
    "floor",
)
EPSILON = 1e-12


def load_e72(root: Path) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "raw"
        / "pacing_joint_alpha_frontier"
        / "pacing_joint_alpha_frontier_full.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def rounded_request(
    value: float,
    mode: str,
) -> int:
    if mode == "ceil":
        return math.ceil(value - 1e-12)
    if mode == "floor":
        return math.floor(value + 1e-12)
    if mode == "nearest":
        return math.floor(value + 0.5)
    raise ValueError(mode)


def phased_geometric_looks(
    *,
    ratio: float,
    phase: float,
    rounding: str,
    maximum_requests: int,
) -> list[int]:
    if ratio <= 1.0:
        raise ValueError(
            "Geometric ratio must exceed one"
        )
    if not 0.0 <= phase < 1.0:
        raise ValueError(
            "Phase must be in [0, 1)"
        )
    looks = [1]
    exponent = phase
    while looks[-1] < maximum_requests:
        raw = ratio**exponent
        if raw >= maximum_requests:
            candidate = maximum_requests
        else:
            candidate = max(
                1,
                rounded_request(
                    raw,
                    rounding,
                ),
            )
        if candidate > looks[-1]:
            looks.append(
                min(
                    candidate,
                    maximum_requests,
                )
            )
        if looks[-1] == maximum_requests:
            break
        exponent += 1.0
    return looks


def grid_variants(
    maximum_requests: int,
) -> list[dict[str, Any]]:
    by_looks: dict[
        tuple[int, ...],
        dict[str, Any],
    ] = {}
    for phase_units in range(
        PHASE_DENOMINATOR
    ):
        phase = (
            phase_units
            / PHASE_DENOMINATOR
        )
        for rounding in ROUNDING_MODES:
            looks = phased_geometric_looks(
                ratio=RATIO,
                phase=phase,
                rounding=rounding,
                maximum_requests=(
                    maximum_requests
                ),
            )
            key = tuple(looks)
            alias = {
                "phase_units": phase_units,
                "phase_denominator": (
                    PHASE_DENOMINATOR
                ),
                "phase": phase,
                "rounding": rounding,
            }
            if key not in by_looks:
                by_looks[key] = {
                    "variant": (
                        f"grid_{len(by_looks):02d}"
                    ),
                    "looks": looks,
                    "aliases": [],
                }
            by_looks[key]["aliases"].append(
                alias
            )
    return list(by_looks.values())


def allocation_roles(
    e72: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[str]],
]:
    schedule = "geometric_2p00"
    summaries = {
        item["allocation"]: item
        for item in e72[
            "allocation_summaries"
        ][schedule]
    }
    roles: dict[str, list[str]] = {}

    def add(
        allocation: str,
        role: str,
    ) -> None:
        roles.setdefault(
            allocation,
            [],
        ).append(role)

    for item in summaries.values():
        if (
            item["joint"][
                "certification_nonworsening"
            ]
            and item["joint"][
                "drift_nonworsening"
            ]
        ):
            add(
                item["allocation"],
                "e72_jointly_nonworsening",
            )
    selections = e72["selections"][
        schedule
    ]
    add(
        selections[
            "e71_certification_only"
        ]["allocation"],
        "e71_certification_only",
    )
    add(
        selections["joint_geomean"][
            "allocation"
        ],
        "e72_joint_geomean",
    )
    grid = {
        item["allocation"]: item
        for item in allocation_grid()
    }
    names = sorted(
        roles,
        key=lambda name: (
            name != "equal",
            name,
        ),
    )
    return (
        [grid[name] for name in names],
        roles,
    )


def geometric_mean(
    values: list[float],
) -> float:
    return math.exp(
        statistics.mean(
            math.log(value)
            for value in values
        )
    )


def ratio_summary(
    ratios: list[float],
    requests: list[int],
    worst: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ratio_vs_equal": summarize(
            ratios
        ),
        "geomean_ratio_vs_equal": (
            geometric_mean(ratios)
        ),
        "mean_required_requests": (
            statistics.mean(requests)
        ),
        "improved_fraction": (
            sum(
                value < 1.0 - EPSILON
                for value in ratios
            )
            / len(ratios)
        ),
        "unchanged_fraction": (
            sum(
                abs(value - 1.0)
                <= EPSILON
                for value in ratios
            )
            / len(ratios)
        ),
        "worsened_fraction": (
            sum(
                value > 1.0 + EPSILON
                for value in ratios
            )
            / len(ratios)
        ),
        "worst_case": worst,
    }


def update_worst(
    current: dict[str, Any] | None,
    *,
    ratio: float,
    key: list[Any],
    required_requests: int,
    equal_requests: int,
) -> dict[str, Any]:
    candidate = {
        "ratio_vs_equal": ratio,
        "key": key,
        "required_requests": (
            required_requests
        ),
        "equal_requests": equal_requests,
    }
    if current is None:
        return candidate
    if (
        ratio
        > current["ratio_vs_equal"]
        + EPSILON
    ):
        return candidate
    if (
        abs(
            ratio
            - current["ratio_vs_equal"]
        )
        <= EPSILON
        and tuple(str(value) for value in key)
        < tuple(
            str(value)
            for value
            in current["key"]
        )
    ):
        return candidate
    return current


def certification_summaries(
    *,
    baselines: list[dict[str, Any]],
    looks: list[int],
    allocations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    trackers = {
        item["allocation"]: {
            "ratios": [],
            "requests": [],
            "worst": None,
        }
        for item in allocations
    }
    finite_count = 0
    for baseline in baselines:
        if baseline["status"] != (
            "certified"
        ):
            continue
        finite_count += 1
        metric_looks = (
            precompute_metric_looks(
                record=baseline,
                looks=looks,
                allocations=allocations,
            )
        )
        evaluated = {
            allocation["allocation"]: (
                evaluate_allocation(
                    allocation=allocation,
                    metric_looks=(
                        metric_looks
                    ),
                    looks=looks,
                )
            )
            for allocation in allocations
        }
        equal = evaluated["equal"]
        if equal is None:
            raise RuntimeError(
                "Equal certification did not "
                "cross"
            )
        equal_requests = int(
            equal["required_requests"]
        )
        key = [
            baseline["ttft_slo_ms"],
            baseline["family"],
            baseline["error_profile"],
            baseline["risk_budget"],
            baseline[
                "calibration_scenario"
            ],
            baseline[
                "confidence_profile"
            ],
        ]
        for allocation in allocations:
            name = allocation["allocation"]
            value = evaluated[name]
            if value is None:
                raise RuntimeError(
                    "Candidate certification did "
                    "not cross"
                )
            requests = int(
                value["required_requests"]
            )
            ratio = (
                requests / equal_requests
            )
            tracker = trackers[name]
            tracker["ratios"].append(
                ratio
            )
            tracker["requests"].append(
                requests
            )
            tracker["worst"] = (
                update_worst(
                    tracker["worst"],
                    ratio=ratio,
                    key=key,
                    required_requests=(
                        requests
                    ),
                    equal_requests=(
                        equal_requests
                    ),
                )
            )
    if finite_count != 1918:
        raise RuntimeError(
            "Unexpected finite certification "
            "count"
        )
    return {
        name: ratio_summary(
            tracker["ratios"],
            tracker["requests"],
            tracker["worst"],
        )
        for name, tracker
        in trackers.items()
    }


def drift_summaries(
    *,
    baselines: list[dict[str, Any]],
    looks: list[int],
    allocations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    trackers = {
        item["allocation"]: {
            "ratios": [],
            "requests": [],
            "worst": None,
        }
        for item in allocations
    }
    finite_count = 0
    for baseline in baselines:
        if baseline["status"] != "detected":
            continue
        finite_count += 1
        equal = weighted_drift_crossing(
            record=baseline,
            looks=looks,
            metric_weight=1.0 / 3.0,
        )
        if equal is None:
            raise RuntimeError(
                "Equal drift did not cross"
            )
        equal_requests = int(
            equal["required_requests"]
        )
        key = [
            baseline["ttft_slo_ms"],
            baseline["family"],
            baseline["error_profile"],
            baseline["risk_budget"],
            baseline[
                "confidence_profile"
            ],
            baseline["breach_metric"],
        ]
        for allocation in allocations:
            name = allocation["allocation"]
            weight = float(
                allocation["weights"][
                    baseline[
                        "breach_metric"
                    ]
                ]
            )
            value = weighted_drift_crossing(
                record=baseline,
                looks=looks,
                metric_weight=weight,
            )
            if value is None:
                raise RuntimeError(
                    "Candidate drift did not "
                    "cross"
                )
            requests = int(
                value["required_requests"]
            )
            ratio = (
                requests / equal_requests
            )
            tracker = trackers[name]
            tracker["ratios"].append(
                ratio
            )
            tracker["requests"].append(
                requests
            )
            tracker["worst"] = (
                update_worst(
                    tracker["worst"],
                    ratio=ratio,
                    key=key,
                    required_requests=(
                        requests
                    ),
                    equal_requests=(
                        equal_requests
                    ),
                )
            )
    if finite_count != 1890:
        raise RuntimeError(
            "Unexpected finite drift count"
        )
    return {
        name: ratio_summary(
            tracker["ratios"],
            tracker["requests"],
            tracker["worst"],
        )
        for name, tracker
        in trackers.items()
    }


def compare_summary(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    prefix: str,
) -> None:
    actual_ratio = actual[
        "ratio_vs_equal"
    ]
    expected_ratio = expected[
        "ratio_vs_equal_kl"
    ]
    for field in (
        "count",
        "minimum",
        "median",
        "p90",
        "maximum",
    ):
        if not math.isclose(
            float(actual_ratio[field]),
            float(expected_ratio[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{prefix} {field} did not "
                "reproduce E72"
            )
    if not math.isclose(
        float(
            actual[
                "geomean_ratio_vs_equal"
            ]
        ),
        float(
            expected[
                "geomean_ratio_vs_equal_kl"
            ]
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{prefix} geomean did not "
            "reproduce E72"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    e68 = load_e68(args.root)
    e72 = load_e72(args.root)
    allocations, roles = (
        allocation_roles(e72)
    )
    variants = grid_variants(
        args.maximum_requests
    )
    base_looks = [
        1 << exponent
        for exponent in range(30)
    ] + [args.maximum_requests]
    base_looks = [
        value
        for index, value
        in enumerate(base_looks)
        if (
            value <= args.maximum_requests
            and (
                index == 0
                or value
                != base_looks[index - 1]
            )
        )
    ]
    if base_looks[-1] != (
        args.maximum_requests
    ):
        base_looks.append(
            args.maximum_requests
        )
    base_variant = next(
        (
            variant
            for variant in variants
            if variant["looks"] == base_looks
        ),
        None,
    )
    if base_variant is None:
        raise RuntimeError(
            "Base power-of-two grid missing"
        )
    records = []
    for variant in variants:
        certification = (
            certification_summaries(
                baselines=e68[
                    "certification_records"
                ],
                looks=variant["looks"],
                allocations=allocations,
            )
        )
        drift = drift_summaries(
            baselines=e68[
                "drift_detection_records"
            ],
            looks=variant["looks"],
            allocations=allocations,
        )
        allocation_results = {}
        for allocation in allocations:
            name = allocation["allocation"]
            cert = certification[name]
            drift_value = drift[name]
            allocation_results[name] = {
                "weights": (
                    allocation["weights"]
                ),
                "roles": roles[name],
                "certification": cert,
                "drift": drift_value,
                "joint_nonworsening": (
                    cert["ratio_vs_equal"][
                        "maximum"
                    ]
                    <= 1.0 + EPSILON
                    and drift_value[
                        "ratio_vs_equal"
                    ]["maximum"]
                    <= 1.0 + EPSILON
                ),
                "maximum_ratio": max(
                    cert["ratio_vs_equal"][
                        "maximum"
                    ],
                    drift_value[
                        "ratio_vs_equal"
                    ]["maximum"],
                ),
            }
        records.append(
            {
                "variant": (
                    variant["variant"]
                ),
                "aliases": (
                    variant["aliases"]
                ),
                "look_count": len(
                    variant["looks"]
                ),
                "first_looks": (
                    variant["looks"][:12]
                ),
                "last_looks": (
                    variant["looks"][-4:]
                ),
                "is_e72_base_grid": (
                    variant is base_variant
                ),
                "allocations": (
                    allocation_results
                ),
            }
        )
    base_record = next(
        record
        for record in records
        if record["is_e72_base_grid"]
    )
    for record in records:
        equal = record["allocations"][
            "equal"
        ]
        for domain in (
            "certification",
            "drift",
        ):
            summary = equal[domain]
            if (
                summary[
                    "geomean_ratio_vs_equal"
                ]
                != 1.0
                or any(
                    summary[
                        "ratio_vs_equal"
                    ][field]
                    != 1.0
                    for field in (
                        "minimum",
                        "median",
                        "p90",
                        "maximum",
                    )
                )
            ):
                raise RuntimeError(
                    "Equal control was not one"
                )
    e72_base = {
        item["allocation"]: item
        for item in e72[
            "allocation_summaries"
        ]["geometric_2p00"]
    }
    for allocation in allocations:
        name = allocation["allocation"]
        compare_summary(
            base_record["allocations"][
                name
            ]["certification"],
            e72_base[name][
                "certification"
            ],
            prefix=(
                f"{name} certification"
            ),
        )
        compare_summary(
            base_record["allocations"][
                name
            ]["drift"],
            e72_base[name]["drift"],
            prefix=f"{name} drift",
        )
    robustness = {}
    for allocation in allocations:
        name = allocation["allocation"]
        values = [
            record["allocations"][name]
            for record in records
        ]
        cert_geomeans = [
            value["certification"][
                "geomean_ratio_vs_equal"
            ]
            for value in values
        ]
        drift_geomeans = [
            value["drift"][
                "geomean_ratio_vs_equal"
            ]
            for value in values
        ]
        worst_record = max(
            records,
            key=lambda record: (
                record["allocations"][name][
                    "maximum_ratio"
                ],
                record["variant"],
            ),
        )
        robustness[name] = {
            "weights": (
                allocation["weights"]
            ),
            "roles": roles[name],
            "unique_variant_count": len(
                records
            ),
            "jointly_nonworsening_variant_count": (
                sum(
                    value[
                        "joint_nonworsening"
                    ]
                    for value in values
                )
            ),
            "jointly_nonworsening_variant_fraction": (
                sum(
                    value[
                        "joint_nonworsening"
                    ]
                    for value in values
                )
                / len(values)
            ),
            "all_variants_jointly_nonworsening": (
                all(
                    value[
                        "joint_nonworsening"
                    ]
                    for value in values
                )
            ),
            "certification_geomean_across_variants": (
                geometric_mean(
                    cert_geomeans
                )
            ),
            "drift_geomean_across_variants": (
                geometric_mean(
                    drift_geomeans
                )
            ),
            "maximum_certification_ratio_across_variants": (
                max(
                    value["certification"][
                        "ratio_vs_equal"
                    ]["maximum"]
                    for value in values
                )
            ),
            "maximum_drift_ratio_across_variants": (
                max(
                    value["drift"][
                        "ratio_vs_equal"
                    ]["maximum"]
                    for value in values
                )
            ),
            "worst_variant": (
                worst_record["variant"]
            ),
            "worst_variant_aliases": (
                worst_record["aliases"]
            ),
            "worst_variant_result": (
                worst_record[
                    "allocations"
                ][name]
            ),
        }
    selected = min(
        robustness,
        key=lambda name: (
            max(
                robustness[name][
                    "maximum_certification_ratio_across_variants"
                ],
                robustness[name][
                    "maximum_drift_ratio_across_variants"
                ],
            ),
            robustness[name][
                "certification_geomean_across_variants"
            ]
            * robustness[name][
                "drift_geomean_across_variants"
            ],
            name != "equal",
            name,
        ),
    )
    requested_grid_count = (
        PHASE_DENOMINATOR
        * len(ROUNDING_MODES)
    )
    result = {
        "suite": (
            "pacing_checkpoint_phase_"
            "sensitivity"
        ),
        "model": (
            "Perturb one logarithmic phase "
            "cycle of the 2.00x checkpoint "
            "schedule under ceil, nearest, and "
            "floor integer rounding. Re-evaluate "
            "the E72 jointly nonworsening splits "
            "plus certification/joint-average "
            "negative controls against an equal-"
            "alpha baseline on each unique grid."
        ),
        "source": {
            "anytime_baseline": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            ),
            "joint_alpha_frontier": (
                "results/raw/"
                "pacing_joint_alpha_frontier/"
                "pacing_joint_alpha_frontier_full.json"
            ),
        },
        "ratio": RATIO,
        "phase_denominator": (
            PHASE_DENOMINATOR
        ),
        "rounding_modes": list(
            ROUNDING_MODES
        ),
        "requested_grid_count": (
            requested_grid_count
        ),
        "unique_grid_count": len(
            records
        ),
        "candidate_allocations": [
            {
                "allocation": (
                    allocation["allocation"]
                ),
                "weights": (
                    allocation["weights"]
                ),
                "roles": roles[
                    allocation["allocation"]
                ],
            }
            for allocation in allocations
        ],
        "variant_records": records,
        "allocation_robustness": (
            robustness
        ),
        "robust_minimax_candidate": {
            "allocation": selected,
            **robustness[selected],
        },
        "validation": {
            "base_grid_variant": (
                base_variant["variant"]
            ),
            "base_grid_aliases": (
                base_variant["aliases"]
            ),
            "base_grid_reproduction": (
                "Every candidate allocation "
                "reproduces its E72 2.00x "
                "certification and drift summary."
            ),
            "finite_certification_cases_per_grid": (
                1918
            ),
            "finite_drift_cases_per_grid": (
                1890
            ),
            "requested_grid_count": (
                requested_grid_count
            ),
            "unique_grid_count": len(
                records
            ),
            "equal_control": (
                "Equal alpha has ratio one in "
                "every certification and drift "
                "case on every grid."
            ),
        },
        "limitations": [
            "The audit perturbs only the 2.00x cadence because E72's non-equal jointly nonworsening result occurred only there.",
            "It evaluates the E72 jointly nonworsening candidates and two negative controls rather than re-optimizing all 172 allocations on every grid.",
            "Logarithmic phase and integer rounding are discrete implementation perturbations; request batching, delayed labels, and wall-clock scheduling can add further offsets.",
            "Every grid remains preregistered and spends alpha by look index. Peeking or acting between its declared looks invalidates the guarantee.",
            "All E68-E72 independence, expected-proportion, traffic-weighting, and detector-model limitations remain.",
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
        "--maximum-requests",
        type=int,
        default=1_000_000_000,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
