from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
    geometric_looks,
)
from analyze_pacing_coalescing_stochastic import (
    WORKERS,
    action_comparison,
    action_times,
    aggregate_records,
    certification_key,
    combined_comparison,
    drift_key,
    maximum_aggregate_regression,
    maximum_regression_record,
    metric_comparisons,
    null_alert_summary,
    public_profile,
    select_profiles,
    simulate_certification,
    simulate_drift,
    stable_seed,
    summarize,
    validate_crossing_equivalence,
    wilson_interval,
)
from analyze_pacing_kl_monitoring import (
    load_e68,
)
from analyze_pacing_monitor_coalescing import (
    coalesced_completion_times,
)
from analyze_pacing_monitor_workers import (
    completion_times,
)


DEFAULT_MARGINAL_FRACTIONS = (
    0.0,
    0.0001,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.25,
    0.50,
    1.0,
)


def fraction_key(
    value: float,
) -> str:
    return (
        f"{value:.4f}"
        .rstrip("0")
        .rstrip(".")
    )


def parse_fractions(
    value: str,
) -> list[float]:
    fractions = sorted(
        {
            float(item)
            for item in value.split(",")
            if item.strip()
        }
    )
    if (
        not fractions
        or fractions[0] < 0.0
        or fractions[-1] > 1.0
    ):
        raise ValueError(
            "Marginal fractions must be in [0, 1]"
        )
    return fractions


def batch_service_cost(
    *,
    service_cost: int,
    batch_size: int,
    marginal_fraction: float,
) -> int:
    if service_cost < 0:
        raise ValueError(
            "Service cost must be nonnegative"
        )
    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive"
        )
    return int(
        math.ceil(
            service_cost
            * (
                1.0
                + marginal_fraction
                * (batch_size - 1)
            )
            - 1e-12
        )
    )


def merged_pending_completion_times(
    *,
    looks: list[int],
    service_cost: int,
    workers: int,
    marginal_fraction: float,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    running: list[
        tuple[int, int]
    ] = []
    batch_members: dict[
        int,
        list[int],
    ] = {}
    pending: list[int] = []
    completion = np.full(
        len(looks),
        -1,
        dtype=np.int64,
    )
    batch_sizes = []
    batch_costs = []
    next_batch_id = 0

    def start_batch(
        *,
        started: int,
        members: list[int],
    ) -> None:
        nonlocal next_batch_id
        batch_id = next_batch_id
        next_batch_id += 1
        size = len(members)
        cost = batch_service_cost(
            service_cost=service_cost,
            batch_size=size,
            marginal_fraction=(
                marginal_fraction
            ),
        )
        batch_members[batch_id] = (
            members
        )
        batch_sizes.append(size)
        batch_costs.append(cost)
        heapq.heappush(
            running,
            (
                started + cost,
                batch_id,
            ),
        )

    def finish_until(
        time_limit: int | None,
    ) -> None:
        while running and (
            time_limit is None
            or running[0][0]
            <= time_limit
        ):
            finished, batch_id = (
                heapq.heappop(running)
            )
            members = batch_members.pop(
                batch_id
            )
            completion[members] = finished
            if pending:
                merged = list(pending)
                pending.clear()
                start_batch(
                    started=finished,
                    members=merged,
                )

    for look_index, arrival in enumerate(
        looks
    ):
        finish_until(arrival)
        if len(running) < workers:
            start_batch(
                started=arrival,
                members=[look_index],
            )
        else:
            pending.append(look_index)
    finish_until(None)
    if pending or batch_members:
        raise RuntimeError(
            "Merged pending queue did not drain"
        )
    if np.any(completion < 0):
        raise RuntimeError(
            "Not every look was evaluated"
        )
    if completion[-1] < 0:
        raise RuntimeError(
            "Final look was not evaluated"
        )
    batch_sizes_array = np.asarray(
        batch_sizes,
        dtype=np.float64,
    )
    return completion, {
        "look_count": len(looks),
        "batch_count": len(batch_sizes),
        "launch_fraction_vs_fifo": (
            len(batch_sizes)
            / len(looks)
        ),
        "launch_savings_fraction": (
            1.0
            - len(batch_sizes)
            / len(looks)
        ),
        "total_service_work": int(
            sum(batch_costs)
        ),
        "fifo_total_service_work": (
            service_cost * len(looks)
        ),
        "service_work_fraction_vs_fifo": (
            (
                sum(batch_costs)
                / (service_cost * len(looks))
            )
            if service_cost > 0
            else 1.0
        ),
        "batch_size": (
            summarize(
                batch_sizes_array
            )
        ),
        "maximum_batch_size": int(
            batch_sizes_array.max()
        ),
    }


def records_for_fraction(
    *,
    records: list[dict[str, Any]],
    marginal_fraction: float,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["marginal_fraction"]
        == marginal_fraction
    ]


def fraction_summary(
    *,
    records: list[dict[str, Any]],
    queue_profiles: dict[
        tuple[float, str, int, str],
        dict[str, Any],
    ],
    marginal_fraction: float,
    profiles: list[dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    selected = records_for_fraction(
        records=records,
        marginal_fraction=(
            marginal_fraction
        ),
    )
    aggregate = aggregate_records(
        selected
    )
    queues = [
        queue
        for (
            fraction,
            _,
            _,
            _,
        ), queue in queue_profiles.items()
        if fraction == marginal_fraction
    ]
    offered = sum(
        queue["look_count"]
        for queue in queues
    )
    batches = sum(
        queue["batch_count"]
        for queue in queues
    )
    service_work = sum(
        queue["total_service_work"]
        for queue in queues
    )
    fifo_work = sum(
        queue["fifo_total_service_work"]
        for queue in queues
    )
    delayed_count = sum(
        record["comparison"][
            "positive_delay_requests"
        ]["count"]
        for record in selected
    )
    delayed_sum = sum(
        (
            record["comparison"][
                "positive_delay_requests"
            ]["mean"]
            or 0.0
        )
        * record["comparison"][
            "positive_delay_requests"
        ]["count"]
        for record in selected
    )
    maximum_delay_record = max(
        selected,
        key=lambda record: (
            record["comparison"][
                "positive_delay_requests"
            ]["maximum"]
            or -1.0,
            record["profile_id"],
            record["schedule"],
            record["worker_policy"],
            record["service_cost"],
        ),
    )
    result = {
        "combined": combined_comparison(
            selected
        ),
        "maximum_aggregate_regression": (
            maximum_aggregate_regression(
                aggregate
            )
        ),
        "maximum_profile_regression": (
            maximum_regression_record(
                selected
            )
        ),
        "positive_delay": {
            "delayed_path_count": (
                delayed_count
            ),
            "weighted_mean_requests": (
                delayed_sum
                / delayed_count
                if delayed_count
                else None
            ),
            "maximum_record": {
                "profile_id": (
                    maximum_delay_record[
                        "profile_id"
                    ]
                ),
                "schedule": (
                    maximum_delay_record[
                        "schedule"
                    ]
                ),
                "worker_policy": (
                    maximum_delay_record[
                        "worker_policy"
                    ]
                ),
                "service_cost": (
                    maximum_delay_record[
                        "service_cost"
                    ]
                ),
                "summary": (
                    maximum_delay_record[
                        "comparison"
                    ][
                        "positive_delay_requests"
                    ]
                ),
            },
        },
        "queue_work": {
            "queue_profile_count": (
                len(queues)
            ),
            "offered_look_count": (
                offered
            ),
            "batch_count": batches,
            "launch_fraction_vs_fifo": (
                batches / offered
            ),
            "launch_savings_fraction": (
                1.0 - batches / offered
            ),
            "total_service_work": (
                service_work
            ),
            "fifo_total_service_work": (
                fifo_work
            ),
            "service_work_fraction_vs_fifo": (
                service_work / fifo_work
            ),
            "maximum_batch_size": max(
                queue["maximum_batch_size"]
                for queue in queues
            ),
        },
    }
    if profiles is not None:
        result["by_breach_metric"] = (
            metric_comparisons(
                records=selected,
                profiles=profiles,
            )
        )
    return result


def marginal_cap_selections(
    summaries: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    caps = {
        "aggregate_0p01pct": (
            "combined",
            0.0001,
        ),
        "aggregate_0p1pct": (
            "combined",
            0.001,
        ),
        "aggregate_1pct": (
            "combined",
            0.01,
        ),
        "worst_profile_0p1pct": (
            "maximum_profile_regression",
            0.001,
        ),
        "worst_profile_1pct": (
            "maximum_profile_regression",
            0.01,
        ),
        "worst_profile_10pct": (
            "maximum_profile_regression",
            0.10,
        ),
    }
    result = {}
    ordered = sorted(
        (
            float(key),
            values,
        )
        for key, values
        in summaries.items()
    )
    for name, (
        source,
        cap,
    ) in caps.items():
        eligible = [
            (
                fraction,
                values,
            )
            for fraction, values
            in ordered
            if (
                (
                    values["combined"]
                    if source == "combined"
                    else values[
                        "maximum_profile_regression"
                    ]
                )[
                    "regression_among_fifo_detected"
                ]["fraction"]
                <= cap
            )
        ]
        if not eligible:
            result[name] = None
            continue
        fraction, values = max(
            eligible,
            key=lambda item: (
                item[0],
                item[1]["combined"][
                    "faster_among_fifo_detected"
                ]["fraction"],
            ),
        )
        result[name] = {
            "maximum_marginal_fraction": (
                fraction
            ),
            "regression_fraction": (
                (
                    values["combined"]
                    if source == "combined"
                    else values[
                        "maximum_profile_regression"
                    ]
                )[
                    "regression_among_fifo_detected"
                ]["fraction"]
            ),
            "faster_fraction": (
                values["combined"][
                    "faster_among_fifo_detected"
                ]["fraction"]
            ),
            "launch_savings_fraction": (
                values["queue_work"][
                    "launch_savings_fraction"
                ]
            ),
            "service_work_fraction_vs_fifo": (
                values["queue_work"][
                    "service_work_fraction_vs_fifo"
                ]
            ),
        }
    return result


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    e68 = load_e68(args.root)
    certification_profiles = (
        select_profiles(
            records=e68[
                "certification_records"
            ],
            status="certified",
            key_function=certification_key,
            strata=(
                "confidence_profile",
                "calibration_scenario",
            ),
            rank_field=(
                "required_labeled_requests"
            ),
            prefix="cert",
        )
    )
    drift_profiles = select_profiles(
        records=e68[
            "drift_detection_records"
        ],
        status="detected",
        key_function=drift_key,
        strata=(
            "confidence_profile",
            "breach_metric",
        ),
        rank_field=(
            "required_post_change_requests"
        ),
        prefix="drift",
    )
    equivalence_checks = (
        validate_crossing_equivalence(
            profiles=(
                certification_profiles
                + drift_profiles
            ),
        )
    )
    schedules = {
        schedule: geometric_looks(
            ratio=ratio,
            maximum_requests=(
                args.maximum_requests
            ),
        )
        for schedule, ratio
        in LOOK_SCHEDULES.items()
    }
    queue_profiles: dict[
        tuple[float, str, int, str],
        dict[str, Any],
    ] = {}
    all_look_checks = 0
    final_look_checks = 0
    zero_service_checks = 0
    marginal_one_work_checks = 0
    zero_marginal_e78_checks = 0
    single_look_checks = 0
    for marginal_fraction in (
        args.marginal_fractions
    ):
        for service_cost in (
            args.service_costs
        ):
            for workers in WORKERS.values():
                completion, stats = (
                    merged_pending_completion_times(
                        looks=[1],
                        service_cost=service_cost,
                        workers=workers,
                        marginal_fraction=(
                            marginal_fraction
                        ),
                    )
                )
                if (
                    completion[0]
                    != 1 + service_cost
                    or stats["batch_count"] != 1
                ):
                    raise RuntimeError(
                        "Single-look batch cost "
                        "did not match FIFO"
                    )
                single_look_checks += 1
        for schedule, looks in (
            schedules.items()
        ):
            for worker_policy, workers in (
                WORKERS.items()
            ):
                fifo_zero = completion_times(
                    looks=looks,
                    service_cost=0,
                    workers=workers,
                )
                merged_zero, _ = (
                    merged_pending_completion_times(
                        looks=looks,
                        service_cost=0,
                        workers=workers,
                        marginal_fraction=(
                            marginal_fraction
                        ),
                    )
                )
                if not np.array_equal(
                    fifo_zero,
                    merged_zero,
                ):
                    raise RuntimeError(
                        "Zero-service batching did "
                        "not reproduce FIFO"
                    )
                zero_service_checks += 1
            for service_cost in (
                args.service_costs
            ):
                for (
                    worker_policy,
                    workers,
                ) in WORKERS.items():
                    fifo = completion_times(
                        looks=looks,
                        service_cost=service_cost,
                        workers=workers,
                    )
                    (
                        completion,
                        stats,
                    ) = (
                        merged_pending_completion_times(
                            looks=looks,
                            service_cost=(
                                service_cost
                            ),
                            workers=workers,
                            marginal_fraction=(
                                marginal_fraction
                            ),
                        )
                    )
                    if (
                        len(completion)
                        != len(looks)
                        or np.any(
                            completion < 0
                        )
                    ):
                        raise RuntimeError(
                            "Not every look was "
                            "assigned exactly once"
                        )
                    all_look_checks += 1
                    if completion[-1] < 0:
                        raise RuntimeError(
                            "Final look was not "
                            "evaluated"
                        )
                    final_look_checks += 1
                    if marginal_fraction == 1.0:
                        if (
                            stats[
                                "total_service_work"
                            ]
                            != stats[
                                "fifo_total_service_work"
                            ]
                        ):
                            raise RuntimeError(
                                "Marginal-one total "
                                "work did not match "
                                "FIFO"
                            )
                        marginal_one_work_checks += 1
                    if marginal_fraction == 0.0:
                        (
                            e78_completion,
                            e78_processed,
                        ) = (
                            coalesced_completion_times(
                                looks=looks,
                                service_cost=(
                                    service_cost
                                ),
                                workers=workers,
                            )
                        )
                        processed_mask = (
                            e78_completion >= 0
                        )
                        if (
                            stats["batch_count"]
                            != e78_processed
                            or not np.array_equal(
                                completion[
                                    processed_mask
                                ],
                                e78_completion[
                                    processed_mask
                                ],
                            )
                        ):
                            raise RuntimeError(
                                "Zero-marginal batching "
                                "did not reproduce E78 "
                                "launch completions"
                            )
                        zero_marginal_e78_checks += 1
                    queue_profiles[
                        (
                            marginal_fraction,
                            schedule,
                            service_cost,
                            worker_policy,
                        )
                    ] = {
                        "fifo": fifo,
                        "completion": (
                            completion
                        ),
                        **stats,
                    }
    certification_records = []
    drift_records = []
    drift_null_records = []
    no_missed_action_checks = 0
    null_equality_checks = 0
    for domain, profiles in (
        (
            "certification",
            certification_profiles,
        ),
        (
            "drift",
            drift_profiles,
        ),
    ):
        for profile in profiles:
            for schedule, looks in (
                schedules.items()
            ):
                rng = np.random.default_rng(
                    stable_seed(
                        args.seed,
                        domain,
                        profile["profile_id"],
                        schedule,
                        "alternative",
                    )
                )
                crossings = (
                    simulate_certification(
                        profile=profile,
                        looks=looks,
                        paths=args.paths,
                        rng=rng,
                    )
                    if domain
                    == "certification"
                    else simulate_drift(
                        profile=profile,
                        looks=looks,
                        paths=args.paths,
                        rng=rng,
                        actual_rate=profile[
                            "post_change_actual_rate"
                        ],
                    )
                )
                for service_cost in (
                    args.service_costs
                ):
                    for (
                        worker_policy,
                        workers,
                    ) in WORKERS.items():
                        for marginal_fraction in (
                            args.marginal_fractions
                        ):
                            queue = queue_profiles[
                                (
                                    marginal_fraction,
                                    schedule,
                                    service_cost,
                                    worker_policy,
                                )
                            ]
                            (
                                comparison,
                                _,
                            ) = action_comparison(
                                crossings=(
                                    crossings
                                ),
                                fifo_completion=(
                                    queue["fifo"]
                                ),
                                coalesced_completion=(
                                    queue[
                                        "completion"
                                    ]
                                ),
                            )
                            if (
                                comparison[
                                    "missed_among_fifo_detected"
                                ]["successes"]
                                != 0
                            ):
                                raise RuntimeError(
                                    "Exact batching missed "
                                    "an eventual action"
                                )
                            no_missed_action_checks += 1
                            record = {
                                "profile_id": (
                                    profile[
                                        "profile_id"
                                    ]
                                ),
                                "marginal_fraction": (
                                    marginal_fraction
                                ),
                                "schedule": (
                                    schedule
                                ),
                                "worker_policy": (
                                    worker_policy
                                ),
                                "workers": workers,
                                "service_cost": (
                                    service_cost
                                ),
                                "look_count": (
                                    queue[
                                        "look_count"
                                    ]
                                ),
                                "batch_count": (
                                    queue[
                                        "batch_count"
                                    ]
                                ),
                                "launch_savings_fraction": (
                                    queue[
                                        "launch_savings_fraction"
                                    ]
                                ),
                                "service_work_fraction_vs_fifo": (
                                    queue[
                                        "service_work_fraction_vs_fifo"
                                    ]
                                ),
                                "comparison": (
                                    comparison
                                ),
                            }
                            (
                                certification_records
                                if domain
                                == "certification"
                                else drift_records
                            ).append(record)
                if domain != "drift":
                    continue
                null_rng = (
                    np.random.default_rng(
                        stable_seed(
                            args.seed,
                            domain,
                            profile[
                                "profile_id"
                            ],
                            schedule,
                            "null",
                        )
                    )
                )
                null_crossings = simulate_drift(
                    profile=profile,
                    looks=looks,
                    paths=args.paths,
                    rng=null_rng,
                    actual_rate=profile[
                        "safe_cap"
                    ],
                )
                for service_cost in (
                    args.service_costs
                ):
                    for (
                        worker_policy,
                        workers,
                    ) in WORKERS.items():
                        for marginal_fraction in (
                            args.marginal_fractions
                        ):
                            queue = queue_profiles[
                                (
                                    marginal_fraction,
                                    schedule,
                                    service_cost,
                                    worker_policy,
                                )
                            ]
                            (
                                fifo_times,
                                fifo_alert,
                            ) = action_times(
                                crossings=(
                                    null_crossings
                                ),
                                completion=queue[
                                    "fifo"
                                ],
                            )
                            (
                                merged_times,
                                merged_alert,
                            ) = action_times(
                                crossings=(
                                    null_crossings
                                ),
                                completion=queue[
                                    "completion"
                                ],
                            )
                            del fifo_times
                            del merged_times
                            if not np.array_equal(
                                fifo_alert,
                                merged_alert,
                            ):
                                raise RuntimeError(
                                    "Exact batching "
                                    "changed null alert "
                                    "membership"
                                )
                            null_equality_checks += 1
                            drift_null_records.append(
                                {
                                    "profile_id": (
                                        profile[
                                            "profile_id"
                                        ]
                                    ),
                                    "marginal_fraction": (
                                        marginal_fraction
                                    ),
                                    "schedule": (
                                        schedule
                                    ),
                                    "worker_policy": (
                                        worker_policy
                                    ),
                                    "workers": workers,
                                    "service_cost": (
                                        service_cost
                                    ),
                                    "fifo_false_alert": (
                                        wilson_interval(
                                            successes=int(
                                                fifo_alert.sum()
                                            ),
                                            total=args.paths,
                                        )
                                    ),
                                    "coalesced_false_alert": (
                                        wilson_interval(
                                            successes=int(
                                                merged_alert.sum()
                                            ),
                                            total=args.paths,
                                        )
                                    ),
                                }
                            )
    cert_by_fraction = {
        fraction_key(fraction): (
            fraction_summary(
                records=certification_records,
                queue_profiles=queue_profiles,
                marginal_fraction=fraction,
            )
        )
        for fraction
        in args.marginal_fractions
    }
    drift_by_fraction = {
        fraction_key(fraction): (
            fraction_summary(
                records=drift_records,
                queue_profiles=queue_profiles,
                marginal_fraction=fraction,
                profiles=drift_profiles,
            )
        )
        for fraction
        in args.marginal_fractions
    }
    null_by_fraction = {
        fraction_key(fraction): (
            null_alert_summary(
                records=[
                    record
                    for record
                    in drift_null_records
                    if record[
                        "marginal_fraction"
                    ]
                    == fraction
                ],
                profiles=drift_profiles,
            )
        )
        for fraction
        in args.marginal_fractions
    }
    result = {
        "suite": (
            "pacing_coalescing_batch"
        ),
        "model": (
            "Merge all pending declared looks "
            "into one nonpreemptive monitor job. "
            "The batch preserves every sufficient-"
            "statistic snapshot and completes all "
            "member looks together. One-look cost "
            "equals FIFO; each additional look "
            "costs a configurable fraction of a "
            "standalone check."
        ),
        "source": {
            "stochastic_audit": (
                "results/raw/"
                "pacing_coalescing_stochastic/"
                "pacing_coalescing_stochastic_full.json"
            ),
            "depth_frontier": (
                "results/raw/"
                "pacing_coalescing_depth/"
                "pacing_coalescing_depth_full.json"
            ),
        },
        "simulation": {
            "seed": args.seed,
            "paths_per_profile_schedule": (
                args.paths
            ),
            "maximum_requests": (
                args.maximum_requests
            ),
            "marginal_fractions": (
                args.marginal_fractions
            ),
            "service_costs": (
                args.service_costs
            ),
            "worker_policies": WORKERS,
            "schedules": {
                schedule: {
                    "ratio": (
                        LOOK_SCHEDULES[
                            schedule
                        ]
                    ),
                    "look_count": len(looks),
                }
                for schedule, looks
                in schedules.items()
            },
        },
        "selected_profiles": {
            "certification": [
                public_profile(
                    profile,
                    "certification",
                )
                for profile
                in certification_profiles
            ],
            "drift": [
                public_profile(
                    profile,
                    "drift",
                )
                for profile
                in drift_profiles
            ],
        },
        "records": {
            "certification": (
                certification_records
            ),
            "drift": drift_records,
            "drift_null": (
                drift_null_records
            ),
        },
        "headline": {
            "certification_by_marginal_fraction": (
                cert_by_fraction
            ),
            "drift_by_marginal_fraction": (
                drift_by_fraction
            ),
            "drift_null_by_marginal_fraction": (
                null_by_fraction
            ),
            "drift_marginal_cap_selections": (
                marginal_cap_selections(
                    drift_by_fraction
                )
            ),
            "exact_statistical_equivalence": {
                "all_looks_evaluated": True,
                "no_missed_actions": True,
                "null_alert_membership_equal_to_fifo": (
                    True
                ),
                    "zero_marginal_launch_schedule_equal_to_e78": (
                        True
                    ),
            },
        },
        "validation": {
            "certification_profile_count": (
                len(certification_profiles)
            ),
            "drift_profile_count": (
                len(drift_profiles)
            ),
            "direct_kl_inversion_equivalence_checks": (
                equivalence_checks
            ),
            "single_look_fifo_cost_checks": (
                single_look_checks
            ),
            "zero_service_fifo_reproduction_checks": (
                zero_service_checks
            ),
            "all_look_assignment_checks": (
                all_look_checks
            ),
            "final_look_checks": (
                final_look_checks
            ),
            "marginal_one_total_work_checks": (
                marginal_one_work_checks
            ),
            "zero_marginal_e78_launch_equivalence_checks": (
                zero_marginal_e78_checks
            ),
            "no_missed_action_checks": (
                no_missed_action_checks
            ),
            "null_alert_membership_equality_checks": (
                null_equality_checks
            ),
        },
        "limitations": [
            "The batch service model fixes standalone cost and assigns a linear marginal cost to each additional look; real query engines may have nonlinear scan and serialization costs.",
            "All member looks complete at batch end. Streaming partial results from within a batch could reduce action delay.",
            "The audit reuses E79's stratified profiles, independent Bernoulli paths, and recursive E70 grids.",
            "One pending batch is launched on the first free worker; pending work is not partitioned across simultaneously idle workers.",
            "Aggregate comparisons repeat stochastic paths across queue and marginal-cost configurations and are not independent Monte Carlo samples.",
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
        "--paths",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026111301,
    )
    parser.add_argument(
        "--maximum-requests",
        type=int,
        default=1_000_000_000,
    )
    parser.add_argument(
        "--service-costs",
        type=lambda value: [
            int(item)
            for item in value.split(",")
            if item.strip()
        ],
        default=[10, 100, 500],
    )
    parser.add_argument(
        "--marginal-fractions",
        type=parse_fractions,
        default=list(
            DEFAULT_MARGINAL_FRACTIONS
        ),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
