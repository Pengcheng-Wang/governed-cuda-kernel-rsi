from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pacing_alpha_allocation import (
    allocation_grid,
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
from analyze_pacing_alpha_allocation import (
    load_e68,
)


WORKER_POLICIES = {
    "workers_1": 1,
    "workers_2": 2,
    "workers_4": 4,
    "workers_8": 8,
    "unlimited": None,
}


def completion_times(
    *,
    looks: list[int],
    service_cost: int,
    workers: int | None,
) -> np.ndarray:
    if workers is None:
        return np.asarray(
            [
                request + service_cost
                for request in looks
            ],
            dtype=np.int64,
        )
    available = [0] * workers
    heapq.heapify(available)
    completion = []
    for request in looks:
        earliest = heapq.heappop(
            available
        )
        finished = (
            max(request, earliest)
            + service_cost
        )
        completion.append(finished)
        heapq.heappush(
            available,
            finished,
        )
    return np.asarray(
        completion,
        dtype=np.int64,
    )


def prepare_domain(
    *,
    cases: list[dict[str, Any]],
    grids: dict[
        str,
        list[dict[str, Any]],
    ],
    schedules: list[str],
) -> dict[str, Any]:
    indices = {}
    request_only = {}
    synchronous_requests = {}
    for schedule in schedules:
        expected_grids = [
            grid["grid"]
            for grid in grids[schedule]
        ]
        schedule_indices = []
        schedule_requests = []
        for case in cases:
            profiles = case["profiles"][
                schedule
            ]
            if [
                profile["grid"]
                for profile in profiles
            ] != expected_grids:
                raise RuntimeError(
                    "Profile/grid order mismatch"
                )
            schedule_indices.append(
                [
                    int(
                        profile["look_index"]
                    )
                    - 1
                    for profile in profiles
                ]
            )
            schedule_requests.append(
                [
                    int(
                        profile["requests"]
                    )
                    for profile in profiles
                ]
            )
        index_array = np.asarray(
            schedule_indices,
            dtype=np.int64,
        )
        request_array = np.asarray(
            schedule_requests,
            dtype=np.int64,
        )
        indices[schedule] = index_array
        request_only[schedule] = (
            request_array.max(axis=1)
        )
        synchronous_requests[
            schedule
        ] = request_array
    return {
        "case_count": len(cases),
        "indices": indices,
        "request_only": request_only,
        "synchronous_requests": (
            synchronous_requests
        ),
        "request_only_winners": [
            tuple(
                case[
                    "request_only_winners"
                ]
            )
            for case in cases
        ],
        "request_only_oracle": np.asarray(
            [
                int(
                    case[
                        "request_only_oracle_cost"
                    ]
                )
                for case in cases
            ],
            dtype=np.int64,
        ),
    }


def action_costs(
    *,
    prepared: dict[str, Any],
    grids: dict[
        str,
        list[dict[str, Any]],
    ],
    schedules: list[str],
    service_cost: int,
    workers: int | None,
) -> np.ndarray:
    costs = []
    for schedule in schedules:
        indices = prepared[
            "indices"
        ][schedule]
        robust = np.zeros(
            prepared["case_count"],
            dtype=np.int64,
        )
        for grid_index, grid in enumerate(
            grids[schedule]
        ):
            completion = completion_times(
                looks=grid["looks"],
                service_cost=service_cost,
                workers=workers,
            )
            robust = np.maximum(
                robust,
                completion[
                    indices[
                        :,
                        grid_index,
                    ]
                ],
            )
        costs.append(robust)
    return np.stack(
        costs,
        axis=1,
    )


def synchronous_costs(
    *,
    prepared: dict[str, Any],
    schedules: list[str],
    service_cost: int,
) -> np.ndarray:
    costs = []
    for schedule in schedules:
        request_array = prepared[
            "synchronous_requests"
        ][schedule]
        indices = prepared[
            "indices"
        ][schedule]
        costs.append(
            (
                request_array
                + service_cost
                * (indices + 1)
            ).max(axis=1)
        )
    return np.stack(
        costs,
        axis=1,
    )


def array_summary(
    values: np.ndarray,
) -> dict[str, Any]:
    as_list = [
        float(value)
        for value in values.tolist()
    ]
    return summarize(as_list)


def cost_summary(
    *,
    costs: np.ndarray,
    prepared: dict[str, Any],
    schedules: list[str],
    synchronous: np.ndarray,
) -> dict[str, Any]:
    oracle = costs.min(axis=1)
    winners = (
        costs == oracle[:, None]
    )
    winner_counts = {}
    unique_counts = {}
    unique_mask = (
        winners.sum(axis=1) == 1
    )
    for index, schedule in enumerate(
        schedules
    ):
        winner_counts[schedule] = int(
            winners[:, index].sum()
        )
        unique_counts[schedule] = int(
            (
                winners[:, index]
                & unique_mask
            ).sum()
        )
    retention = statistics.mean(
        any(
            winners[
                case_index,
                schedules.index(schedule),
            ]
            for schedule in (
                prepared[
                    "request_only_winners"
                ][case_index]
            )
        )
        for case_index in range(
            prepared["case_count"]
        )
    )
    regret = {}
    for index, schedule in enumerate(
        schedules
    ):
        ratios = (
            costs[:, index] / oracle
        )
        regret[schedule] = {
            "ratio_vs_per_case_oracle": (
                array_summary(ratios)
            ),
            "geomean_ratio_vs_per_case_oracle": (
                float(
                    np.exp(
                        np.log(
                            ratios
                        ).mean()
                    )
                )
            ),
        }
    minimax = min(
        schedules,
        key=lambda schedule: (
            regret[schedule][
                "ratio_vs_per_case_oracle"
            ]["maximum"],
            regret[schedule][
                "ratio_vs_per_case_oracle"
            ]["p90"],
            regret[schedule][
                "geomean_ratio_vs_per_case_oracle"
            ],
            schedule,
        ),
    )
    request_only_oracle = prepared[
        "request_only_oracle"
    ]
    return {
        "case_count": (
            prepared["case_count"]
        ),
        "winner_membership_counts": (
            winner_counts
        ),
        "unique_winner_counts": (
            unique_counts
        ),
        "request_only_winner_retention_fraction": (
            retention
        ),
        "per_case_oracle_action_ratio_vs_request_only": (
            array_summary(
                oracle
                / request_only_oracle
            )
        ),
        "per_case_oracle_action_delay_requests": (
            array_summary(
                oracle
                - request_only_oracle
            )
        ),
        "fixed_schedule_regret": regret,
        "global_minimax_schedule": minimax,
        "ratio_vs_synchronous_upper_bound": {
            schedule: array_summary(
                costs[:, index]
                / synchronous[:, index]
            )
            for index, schedule
            in enumerate(schedules)
        },
        "action_ratio_vs_request_only_by_schedule": {
            schedule: array_summary(
                costs[:, index]
                / prepared[
                    "request_only"
                ][schedule]
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
    workers: int | None,
    maximum_service_cost: int,
) -> dict[str, Any]:
    services = list(
        range(maximum_service_cost + 1)
    )
    minimax = []
    for service_cost in services:
        costs = action_costs(
            prepared=prepared,
            grids=grids,
            schedules=schedules,
            service_cost=service_cost,
            workers=workers,
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
        minimax.append(
            min(
                schedules,
                key=lambda schedule: (
                    regret[schedule]
                ),
            )
        )
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
            WORKER_POLICIES.items()
        ):
            cert_costs = action_costs(
                prepared=certification,
                grids=grids,
                schedules=schedules,
                service_cost=service_cost,
                workers=workers,
            )
            drift_costs = action_costs(
                prepared=drift,
                grids=grids,
                schedules=schedules,
                service_cost=service_cost,
                workers=workers,
            )
            sampled["certification"].setdefault(
                policy,
                {},
            )[str(service_cost)] = (
                cost_summary(
                    costs=cert_costs,
                    prepared=certification,
                    schedules=schedules,
                    synchronous=cert_sync,
                )
            )
            sampled["drift"].setdefault(
                policy,
                {},
            )[str(service_cost)] = (
                cost_summary(
                    costs=drift_costs,
                    prepared=drift,
                    schedules=schedules,
                    synchronous=drift_sync,
                )
            )
    exact_scans = {
        "certification": {},
        "drift": {},
    }
    for policy, workers in (
        WORKER_POLICIES.items()
    ):
        exact_scans[
            "certification"
        ][policy] = exact_transition_scan(
            prepared=certification,
            grids=grids,
            schedules=schedules,
            workers=workers,
            maximum_service_cost=(
                args.transition_scan_maximum
            ),
        )
        exact_scans["drift"][
            policy
        ] = exact_transition_scan(
            prepared=drift,
            grids=grids,
            schedules=schedules,
            workers=workers,
            maximum_service_cost=(
                args.transition_scan_maximum
            ),
        )
    for domain, prepared in (
        ("certification", certification),
        ("drift", drift),
    ):
        zero = sampled[domain]
        expected = (
            e75[
                "certification_winners"
            ][
                "robust_unique_winner_counts"
            ]
            if domain == "certification"
            else e75["drift_winners"][
                "robust_unique_winner_counts"
            ]
        )
        for policy in WORKER_POLICIES:
            if (
                zero[policy]["0"][
                    "unique_winner_counts"
                ]
                != expected
            ):
                raise RuntimeError(
                    "Zero-service worker policy "
                    "did not reproduce E75"
                )
        for service_cost in service_costs:
            unlimited = zero[
                "unlimited"
            ][str(service_cost)]
            expected_delay = service_cost
            delay = unlimited[
                "per_case_oracle_action_delay_requests"
            ]
            if not all(
                math.isclose(
                    float(delay[field]),
                    float(expected_delay),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for field in (
                    "minimum",
                    "median",
                    "p90",
                    "maximum",
                )
            ):
                raise RuntimeError(
                    "Unlimited workers did not "
                    "add exactly one service cost"
                )
    for domain in (
        "certification",
        "drift",
    ):
        for service_cost in service_costs:
            previous = None
            for policy in (
                "unlimited",
                "workers_8",
                "workers_4",
                "workers_2",
                "workers_1",
            ):
                median = sampled[domain][
                    policy
                ][str(service_cost)][
                    "per_case_oracle_action_"
                    "delay_requests"
                ]["median"]
                if (
                    previous is not None
                    and median
                    < previous - 1e-12
                ):
                    raise RuntimeError(
                        "Worker monotonicity failed"
                    )
                previous = median
                ratios = sampled[domain][
                    policy
                ][str(service_cost)][
                    "ratio_vs_synchronous_"
                    "upper_bound"
                ]
                if any(
                    summary["maximum"]
                    > 1.0 + 1e-12
                    for summary
                    in ratios.values()
                ):
                    raise RuntimeError(
                        "Async action exceeded "
                        "synchronous upper bound"
                    )
    result = {
        "suite": (
            "pacing_monitor_worker_"
            "frontier"
        ),
        "model": (
            "Replay every declared checkpoint "
            "through asynchronous FIFO monitor "
            "worker pools. Each look is one job "
            "with fixed request-equivalent service "
            "time; alert action occurs when the "
            "first statistically crossing look "
            "finishes processing. Compare one, "
            "two, four, eight, and unlimited "
            "workers against E76's synchronous "
            "blocking upper bound."
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
            "synchronous_look_cost": (
                "results/raw/"
                "pacing_look_cost_frontier/"
                "pacing_look_cost_frontier_full.json"
            ),
        },
        "worker_policies": {
            policy: {
                "workers": workers,
            }
            for policy, workers
            in WORKER_POLICIES.items()
        },
        "service_costs_request_equivalents": (
            service_costs
        ),
        "schedules": {
            schedule: {
                "ratio": LOOK_SCHEDULES[
                    schedule
                ],
                "unique_grid_count": len(
                    grids[schedule]
                ),
            }
            for schedule in schedules
        },
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
            "zero_service_reproduction": (
                "Every worker policy reproduces "
                "E75 robust unique winner counts."
            ),
            "unlimited_workers": (
                "Unlimited workers add exactly "
                "one service cost to the request-"
                "only per-case action oracle."
            ),
            "worker_monotonicity": (
                "Median action delay is "
                "nondecreasing as worker count "
                "falls."
            ),
            "synchronous_bound": (
                "Every asynchronous worker-pool "
                "action cost is no greater than "
                "E76's synchronous blocking cost."
            ),
        },
        "limitations": [
            "Monitor jobs are FIFO, identical-cost, and nonpreemptive. Production queries can have variable cost, priority, cancellation, or coalescing.",
            "Worker service is expressed in request-equivalent time and assumes a stationary arrival-rate conversion.",
            "The model processes every declared look. Dropping superseded checks can reduce queueing but changes the operational policy and requires a separate validity audit.",
            "Unlimited workers are a lower-bound concurrency model; E76 synchronous blocking is the upper bound.",
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
