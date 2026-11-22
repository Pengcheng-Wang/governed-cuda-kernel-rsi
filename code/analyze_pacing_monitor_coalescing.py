from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pacing_alpha_allocation import (
    allocation_grid,
    effective_trials,
    load_e68,
    look_alpha,
    summarize,
)
from analyze_pacing_cadence_robustness import (
    schedule_grids,
)
from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
)
from analyze_pacing_look_cost_frontier import (
    DEFAULT_LOOK_OVERHEADS,
    build_certification_profiles,
    build_drift_profiles,
    load_e75,
    parse_overheads,
)
from analyze_pacing_monitor_workers import (
    WORKER_POLICIES,
    action_costs,
    completion_times,
    cost_summary,
    prepare_domain,
    synchronous_costs,
)


FINITE_WORKERS = {
    policy: workers
    for policy, workers
    in WORKER_POLICIES.items()
    if workers is not None
}


def coalesced_completion_times(
    *,
    looks: list[int],
    service_cost: int,
    workers: int,
) -> tuple[
    np.ndarray,
    int,
]:
    running: list[
        tuple[int, int]
    ] = []
    pending: int | None = None
    completion = np.full(
        len(looks),
        -1,
        dtype=np.int64,
    )

    def finish_until(
        time_limit: int | None,
    ) -> None:
        nonlocal pending
        while running and (
            time_limit is None
            or running[0][0]
            <= time_limit
        ):
            finished, job_index = (
                heapq.heappop(running)
            )
            completion[job_index] = (
                finished
            )
            if pending is not None:
                next_job = pending
                pending = None
                heapq.heappush(
                    running,
                    (
                        finished
                        + service_cost,
                        next_job,
                    ),
                )

    for job_index, arrival in enumerate(
        looks
    ):
        finish_until(arrival)
        if len(running) < workers:
            heapq.heappush(
                running,
                (
                    arrival + service_cost,
                    job_index,
                ),
            )
        else:
            pending = job_index
    finish_until(None)
    if pending is not None:
        raise RuntimeError(
            "Pending monitor job did not drain"
        )
    processed = int(
        (completion >= 0).sum()
    )
    if completion[-1] < 0:
        raise RuntimeError(
            "Final checkpoint was not processed"
        )
    return completion, processed


def next_processed_completion(
    completion: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(completion)
    next_value = -1
    for index in range(
        len(completion) - 1,
        -1,
        -1,
    ):
        if completion[index] >= 0:
            next_value = int(
                completion[index]
            )
        if next_value < 0:
            raise RuntimeError(
                "No processed checkpoint after "
                "crossing"
            )
        result[index] = next_value
    return result


def coalesced_action_costs(
    *,
    prepared: dict[str, Any],
    grids: dict[
        str,
        list[dict[str, Any]],
    ],
    schedules: list[str],
    service_cost: int,
    workers: int,
) -> tuple[
    np.ndarray,
    dict[str, dict[str, Any]],
]:
    costs = []
    job_stats = {}
    for schedule in schedules:
        indices = prepared[
            "indices"
        ][schedule]
        robust = np.zeros(
            prepared["case_count"],
            dtype=np.int64,
        )
        fractions = []
        processed_counts = []
        for grid_index, grid in enumerate(
            grids[schedule]
        ):
            completion, processed = (
                coalesced_completion_times(
                    looks=grid["looks"],
                    service_cost=service_cost,
                    workers=workers,
                )
            )
            next_completion = (
                next_processed_completion(
                    completion
                )
            )
            robust = np.maximum(
                robust,
                next_completion[
                    indices[
                        :,
                        grid_index,
                    ]
                ],
            )
            processed_counts.append(
                processed
            )
            fractions.append(
                processed
                / len(grid["looks"])
            )
        job_stats[schedule] = {
            "grid_count": len(
                grids[schedule]
            ),
            "processed_job_count": (
                summarize(
                    [
                        float(value)
                        for value
                        in processed_counts
                    ]
                )
            ),
            "processed_job_fraction": (
                summarize(fractions)
            ),
            "dropped_job_fraction": (
                summarize(
                    [
                        1.0 - value
                        for value in fractions
                    ]
                )
            ),
        }
        costs.append(robust)
    return (
        np.stack(
            costs,
            axis=1,
        ),
        job_stats,
    )


def array_summary(
    values: np.ndarray,
) -> dict[str, Any]:
    return summarize(
        [
            float(value)
            for value in values.tolist()
        ]
    )


def comparison_summary(
    *,
    coalesced: np.ndarray,
    fifo: np.ndarray,
    schedules: list[str],
) -> dict[str, Any]:
    coalesced_oracle = (
        coalesced.min(axis=1)
    )
    fifo_oracle = fifo.min(axis=1)
    oracle_ratio = (
        coalesced_oracle
        / fifo_oracle
    )
    oracle_delta = (
        coalesced_oracle
        - fifo_oracle
    )
    return {
        "per_case_oracle_ratio_vs_fifo": (
            array_summary(oracle_ratio)
        ),
        "per_case_oracle_delta_vs_fifo": (
            array_summary(oracle_delta)
        ),
        "oracle_faster_fraction": float(
            (
                coalesced_oracle
                < fifo_oracle
            ).mean()
        ),
        "oracle_equal_fraction": float(
            (
                coalesced_oracle
                == fifo_oracle
            ).mean()
        ),
        "oracle_slower_fraction": float(
            (
                coalesced_oracle
                > fifo_oracle
            ).mean()
        ),
        "ratio_vs_fifo_by_schedule": {
            schedule: array_summary(
                coalesced[:, index]
                / fifo[:, index]
            )
            for index, schedule
            in enumerate(schedules)
        },
        "delta_vs_fifo_by_schedule": {
            schedule: array_summary(
                coalesced[:, index]
                - fifo[:, index]
            )
            for index, schedule
            in enumerate(schedules)
        },
    }


def exact_transition_scan(
    *,
    prepared: dict[str, Any],
    grids: dict[
        str,
        list[dict[str, Any]],
    ],
    schedules: list[str],
    workers: int,
    maximum_service_cost: int,
) -> dict[str, Any]:
    services = list(
        range(maximum_service_cost + 1)
    )
    minimax = []
    faster = 0
    equal = 0
    slower = 0
    maximum_savings = 0
    selection_reason_counts = {
        "maximum_regret": 0,
        "p90_regret": 0,
        "geometric_mean_regret": 0,
        "lexical_tie_break": 0,
    }
    transition_reason_counts = {
        "maximum_regret": 0,
        "p90_regret": 0,
        "geometric_mean_regret": 0,
        "lexical_tie_break": 0,
    }
    maximum_regret_gaps = []

    def selection_reason(
        preferred: tuple[
            float,
            float,
            float,
            str,
        ],
        alternative: tuple[
            float,
            float,
            float,
            str,
        ],
    ) -> str:
        for index, reason in enumerate(
            (
                "maximum_regret",
                "p90_regret",
                "geometric_mean_regret",
            )
        ):
            if not math.isclose(
                preferred[index],
                alternative[index],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return reason
        return "lexical_tie_break"

    for service_cost in services:
        costs, _ = coalesced_action_costs(
            prepared=prepared,
            grids=grids,
            schedules=schedules,
            service_cost=service_cost,
            workers=workers,
        )
        fifo = action_costs(
            prepared=prepared,
            grids=grids,
            schedules=schedules,
            service_cost=service_cost,
            workers=workers,
        )
        if np.any(costs > fifo):
            raise RuntimeError(
                "Coalescing was slower than FIFO "
                "for an exact-scan schedule"
            )
        coalesced_oracle = (
            costs.min(axis=1)
        )
        fifo_oracle = fifo.min(axis=1)
        faster += int(
            (
                coalesced_oracle
                < fifo_oracle
            ).sum()
        )
        equal += int(
            (
                coalesced_oracle
                == fifo_oracle
            ).sum()
        )
        slower += int(
            (
                coalesced_oracle
                > fifo_oracle
            ).sum()
        )
        maximum_savings = max(
            maximum_savings,
            int(
                (
                    fifo_oracle
                    - coalesced_oracle
                ).max()
            ),
        )
        oracle = costs.min(axis=1)
        regret = {}
        for index, schedule in enumerate(
            schedules
        ):
            ratios = (
                costs[:, index] / oracle
            )
            ordered = np.sort(ratios)
            regret[schedule] = (
                float(ratios.max()),
                float(
                    ordered[
                        int(
                            0.9
                            * (
                                len(ordered)
                                - 1
                            )
                        )
                    ]
                ),
                float(
                    np.exp(
                        np.log(
                            ratios
                        ).mean()
                    )
                ),
                schedule,
            )
        ranked = sorted(
            schedules,
            key=lambda schedule: (
                regret[schedule]
            ),
        )
        selected = ranked[0]
        runner_up = ranked[1]
        reason = selection_reason(
            regret[selected],
            regret[runner_up],
        )
        selection_reason_counts[reason] += 1
        maximum_regret_gaps.append(
            regret[runner_up][0]
            - regret[selected][0]
        )
        if (
            minimax
            and selected != minimax[-1]
        ):
            transition_reason = (
                selection_reason(
                    regret[selected],
                    regret[minimax[-1]],
                )
            )
            transition_reason_counts[
                transition_reason
            ] += 1
        minimax.append(selected)
    segments = []
    start = 0
    for index in range(
        1,
        len(services) + 1,
    ):
        if (
            index == len(services)
            or minimax[index]
            != minimax[start]
        ):
            segments.append(
                {
                    "schedule": (
                        minimax[start]
                    ),
                    "first_service_cost": (
                        services[start]
                    ),
                    "last_service_cost": (
                        services[
                            index - 1
                        ]
                    ),
                }
            )
            start = index
    return {
        "maximum_scanned_service_cost": (
            maximum_service_cost
        ),
        "segments": segments,
        "transition_count": max(
            0,
            len(segments) - 1,
        ),
        "service_count_by_schedule": {
            schedule: minimax.count(
                schedule
            )
            for schedule in schedules
        },
        "selection_diagnostics": {
            "service_count_by_deciding_component": (
                selection_reason_counts
            ),
            "transition_count_by_deciding_component": (
                transition_reason_counts
            ),
            "best_vs_runner_up_maximum_regret_gap": (
                summarize(
                    maximum_regret_gaps
                )
            ),
        },
        "oracle_comparison_vs_fifo": {
            "case_service_count": (
                faster + equal + slower
            ),
            "faster_count": faster,
            "equal_count": equal,
            "slower_count": slower,
            "faster_fraction": (
                faster
                / (faster + equal + slower)
            ),
            "equal_fraction": (
                equal
                / (faster + equal + slower)
            ),
            "slower_fraction": (
                slower
                / (faster + equal + slower)
            ),
            "maximum_request_savings": (
                maximum_savings
            ),
        },
    }


def threshold_sequence(
    *,
    looks: list[int],
    total_alpha: float,
    dimension_count: int,
    conditioning_probability: float,
) -> list[float]:
    values = []
    for look_index, requests in enumerate(
        looks,
        start=1,
    ):
        trials = (
            requests
            if conditioning_probability == 1.0
            else effective_trials(
                requests=requests,
                conditioning_probability=(
                    conditioning_probability
                ),
            )
        )
        if trials <= 0:
            values.append(
                math.inf
            )
            continue
        alpha = look_alpha(
            total_alpha=total_alpha,
            metric_weight=1.0 / 3.0,
            dimension_count=(
                dimension_count
            ),
            look_index=look_index,
        )
        values.append(
            math.log(1.0 / alpha)
            / trials
        )
    return values


def validate_crossing_monotonicity(
    *,
    e68: dict[str, Any],
    certification_cases: list[
        dict[str, Any]
    ],
    drift_cases: list[
        dict[str, Any]
    ],
    grids: dict[
        str,
        list[dict[str, Any]],
    ],
    schedules: list[str],
) -> dict[str, int]:
    certification_baselines = [
        baseline
        for baseline in e68[
            "certification_records"
        ]
        if baseline["status"]
        == "certified"
    ]
    drift_baselines = [
        baseline
        for baseline in e68[
            "drift_detection_records"
        ]
        if baseline["status"]
        == "detected"
    ]
    if (
        len(certification_baselines)
        != len(certification_cases)
        or len(drift_baselines)
        != len(drift_cases)
    ):
        raise RuntimeError(
            "Monotonicity baseline count "
            "mismatch"
        )
    certification_checks = 0
    for baseline, case in zip(
        certification_baselines,
        certification_cases,
        strict=True,
    ):
        total_alpha = (
            1.0
            - float(
                baseline[
                    "joint_confidence"
                ]
            )
        )
        dimension_count = int(
            baseline["dimension_count"]
        )
        conditionings = (
            1.0,
            float(
                baseline[
                    "actual_state_error"
                ]
            ),
            1.0
            - float(
                baseline[
                    "actual_state_error"
                ]
            ),
        )
        for schedule in schedules:
            for grid_index, grid in enumerate(
                grids[schedule]
            ):
                crossing = (
                    case["profiles"][
                        schedule
                    ][grid_index][
                        "look_index"
                    ]
                    - 1
                )
                for conditioning in (
                    conditionings
                ):
                    thresholds = (
                        threshold_sequence(
                            looks=grid["looks"],
                            total_alpha=(
                                total_alpha
                            ),
                            dimension_count=(
                                dimension_count
                            ),
                            conditioning_probability=(
                                conditioning
                            ),
                        )
                    )
                    active = thresholds[
                        crossing:
                    ]
                    if any(
                        later
                        > earlier + 1e-12
                        for earlier, later
                        in zip(
                            active,
                            active[1:],
                        )
                    ):
                        raise RuntimeError(
                            "Certification confidence "
                            "threshold widened after "
                            "crossing"
                        )
                    certification_checks += 1
    drift_checks = 0
    for baseline, case in zip(
        drift_baselines,
        drift_cases,
        strict=True,
    ):
        total_alpha = (
            1.0
            - float(
                baseline[
                    "joint_confidence"
                ]
            )
        )
        dimension_count = int(
            baseline["dimension_count"]
        )
        conditioning = (
            1.0
            if baseline[
                "breach_metric"
            ]
            == "state_error"
            else float(
                baseline[
                    "conditioning_probability"
                ]
            )
        )
        for schedule in schedules:
            for grid_index, grid in enumerate(
                grids[schedule]
            ):
                crossing = (
                    case["profiles"][
                        schedule
                    ][grid_index][
                        "look_index"
                    ]
                    - 1
                )
                thresholds = (
                    threshold_sequence(
                        looks=grid["looks"],
                        total_alpha=total_alpha,
                        dimension_count=(
                            dimension_count
                        ),
                        conditioning_probability=(
                            conditioning
                        ),
                    )
                )
                active = thresholds[
                    crossing:
                ]
                if any(
                    later
                    > earlier + 1e-12
                    for earlier, later
                    in zip(
                        active,
                        active[1:],
                    )
                ):
                    raise RuntimeError(
                        "Drift confidence threshold "
                        "widened after crossing"
                    )
                drift_checks += 1
    return {
        "certification_metric_grid_checks": (
            certification_checks
        ),
        "drift_metric_grid_checks": (
            drift_checks
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e68 = load_e68(args.root)
    e75 = load_e75(args.root)
    schedules = list(
        LOOK_SCHEDULES
    )
    service_costs = parse_overheads(
        args.service_costs
    )
    equal = next(
        allocation
        for allocation in allocation_grid()
        if allocation["allocation"]
        == "equal"
    )
    grids = {
        schedule: schedule_grids(
            schedule=schedule,
            ratio=LOOK_SCHEDULES[
                schedule
            ],
            maximum_requests=(
                args.maximum_requests
            ),
        )
        for schedule in schedules
    }
    certification_cases, cert_reproduction = (
        build_certification_profiles(
            e68=e68,
            e75=e75,
            grids=grids,
            equal=equal,
            schedules=schedules,
        )
    )
    drift_cases, drift_reproduction = (
        build_drift_profiles(
            e68=e68,
            e75=e75,
            grids=grids,
            schedules=schedules,
        )
    )
    monotonicity = (
        validate_crossing_monotonicity(
            e68=e68,
            certification_cases=(
                certification_cases
            ),
            drift_cases=drift_cases,
            grids=grids,
            schedules=schedules,
        )
    )
    certification = prepare_domain(
        cases=certification_cases,
        grids=grids,
        schedules=schedules,
    )
    drift = prepare_domain(
        cases=drift_cases,
        grids=grids,
        schedules=schedules,
    )
    sampled = {
        "certification": {},
        "drift": {},
    }
    for service_cost in service_costs:
        cert_sync = synchronous_costs(
            prepared=certification,
            schedules=schedules,
            service_cost=service_cost,
        )
        drift_sync = synchronous_costs(
            prepared=drift,
            schedules=schedules,
            service_cost=service_cost,
        )
        for policy, workers in (
            FINITE_WORKERS.items()
        ):
            cert_coalesced, cert_jobs = (
                coalesced_action_costs(
                    prepared=certification,
                    grids=grids,
                    schedules=schedules,
                    service_cost=service_cost,
                    workers=workers,
                )
            )
            drift_coalesced, drift_jobs = (
                coalesced_action_costs(
                    prepared=drift,
                    grids=grids,
                    schedules=schedules,
                    service_cost=service_cost,
                    workers=workers,
                )
            )
            cert_fifo = action_costs(
                prepared=certification,
                grids=grids,
                schedules=schedules,
                service_cost=service_cost,
                workers=workers,
            )
            drift_fifo = action_costs(
                prepared=drift,
                grids=grids,
                schedules=schedules,
                service_cost=service_cost,
                workers=workers,
            )
            sampled["certification"].setdefault(
                policy,
                {},
            )[str(service_cost)] = {
                "coalesced": cost_summary(
                    costs=cert_coalesced,
                    prepared=certification,
                    schedules=schedules,
                    synchronous=cert_sync,
                ),
                "comparison_vs_fifo": (
                    comparison_summary(
                        coalesced=(
                            cert_coalesced
                        ),
                        fifo=cert_fifo,
                        schedules=schedules,
                    )
                ),
                "jobs": cert_jobs,
            }
            sampled["drift"].setdefault(
                policy,
                {},
            )[str(service_cost)] = {
                "coalesced": cost_summary(
                    costs=drift_coalesced,
                    prepared=drift,
                    schedules=schedules,
                    synchronous=drift_sync,
                ),
                "comparison_vs_fifo": (
                    comparison_summary(
                        coalesced=(
                            drift_coalesced
                        ),
                        fifo=drift_fifo,
                        schedules=schedules,
                    )
                ),
                "jobs": drift_jobs,
            }
    exact_scans = {
        "certification": {
            policy: exact_transition_scan(
                prepared=certification,
                grids=grids,
                schedules=schedules,
                workers=workers,
                maximum_service_cost=(
                    args.transition_scan_maximum
                ),
            )
            for policy, workers
            in FINITE_WORKERS.items()
        },
        "drift": {
            policy: exact_transition_scan(
                prepared=drift,
                grids=grids,
                schedules=schedules,
                workers=workers,
                maximum_service_cost=(
                    args.transition_scan_maximum
                ),
            )
            for policy, workers
            in FINITE_WORKERS.items()
        },
    }
    for domain, prepared in (
        ("certification", certification),
        ("drift", drift),
    ):
        for policy, workers in (
            FINITE_WORKERS.items()
        ):
            zero = sampled[domain][
                policy
            ]["0"]
            if (
                zero["comparison_vs_fifo"][
                    "oracle_equal_fraction"
                ]
                != 1.0
            ):
                raise RuntimeError(
                    "Zero-service coalescing did "
                    "not reproduce FIFO"
                )
            for service_cost in service_costs:
                values = sampled[domain][
                    policy
                ][str(service_cost)]
                if any(
                    summary["maximum"]
                    > 1.0 + 1e-12
                    for summary in values[
                        "coalesced"
                    ][
                        "ratio_vs_synchronous_"
                        "upper_bound"
                    ].values()
                ):
                    raise RuntimeError(
                        "Coalescing exceeded "
                        "synchronous upper bound"
                    )
                for schedule in schedules:
                    fraction = values[
                        "jobs"
                    ][schedule][
                        "processed_job_fraction"
                    ]
                    if not (
                        0.0
                        < fraction["minimum"]
                        <= fraction["maximum"]
                        <= 1.0
                    ):
                        raise RuntimeError(
                            "Invalid processed-job "
                            "fraction"
                        )
    result = {
        "suite": (
            "pacing_monitor_coalescing"
        ),
        "model": (
            "Use nonpreemptive latest-pending "
            "coalescing for asynchronous FIFO "
            "monitor workers. Running checks "
            "finish; whenever all workers are "
            "busy, a new checkpoint replaces the "
            "older queued checkpoint. Skipped "
            "looks leave alpha unused, and a "
            "monotone confidence-threshold gate "
            "verifies that the next processed look "
            "after a dropped crossing remains "
            "statistically sufficient."
        ),
        "source": {
            "anytime_baseline": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            ),
            "cadence_robustness": (
                "results/raw/"
                "pacing_cadence_robustness/"
                "pacing_cadence_robustness_full.json"
            ),
            "monitor_workers": (
                "results/raw/"
                "pacing_monitor_workers/"
                "pacing_monitor_workers_full.json"
            ),
        },
        "worker_policies": {
            policy: {
                "workers": workers,
            }
            for policy, workers
            in FINITE_WORKERS.items()
        },
        "service_costs_request_equivalents": (
            service_costs
        ),
        "sampled_summaries": sampled,
        "exact_transition_scans": (
            exact_scans
        ),
        "validation": {
            "certification_case_count": (
                certification[
                    "case_count"
                ]
            ),
            "drift_case_count": (
                drift["case_count"]
            ),
            "certification_schedule_reproduction_count": (
                cert_reproduction
            ),
            "expected_certification_schedule_reproduction_count": (
                1918 * len(schedules)
            ),
            "drift_schedule_reproduction_count": (
                drift_reproduction
            ),
            "expected_drift_schedule_reproduction_count": (
                1890 * len(schedules)
            ),
            "crossing_monotonicity": (
                monotonicity
            ),
            "zero_service_control": (
                "Every worker policy exactly "
                "reproduces FIFO per-case oracle "
                "action cost at zero service."
            ),
            "alpha_validity": (
                "Coalescing depends only on look "
                "arrival and worker state, not "
                "monitor outcomes; skipped looks "
                "leave their alpha unspent."
            ),
            "synchronous_bound": (
                "Every coalesced action cost is no "
                "greater than E76's synchronous "
                "blocking upper bound."
            ),
            "fifo_dominance": (
                "Across every integer service cost "
                "from zero through the exact scan, "
                "coalesced per-schedule action cost "
                "is no greater than FIFO."
            ),
        },
        "limitations": [
            "Running jobs are nonpreemptive; only queued jobs are replaced. Preemptive latest-look service can behave differently.",
            "The confidence-threshold monotonicity gate uses E68's expected fixed empirical rates. Production estimates can fluctuate between looks.",
            "Jobs remain identical-cost FIFO work and the pending queue holds only the latest look. Variable cost, priority, cancellation overhead, and multiple metric services remain unmodeled.",
            "Coalescing is outcome-independent and conservative here. Data-dependent look selection would require separate optional-stopping control.",
            "All earlier independence, traffic-weighting, delayed-label, and preregistration limitations remain.",
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
        "--service-costs",
        default=",".join(
            str(value)
            for value
            in DEFAULT_LOOK_OVERHEADS
        ),
    )
    parser.add_argument(
        "--transition-scan-maximum",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--maximum-requests",
        type=int,
        default=1_000_000_000,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
