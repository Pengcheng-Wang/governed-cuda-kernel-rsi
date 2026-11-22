from __future__ import annotations

import argparse
import heapq
import json
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
    aggregate_records,
    combined_comparison,
    drift_key,
    certification_key,
    maximum_aggregate_regression,
    maximum_regression_record,
    metric_comparisons,
    null_alert_summary,
    public_profile,
    select_profiles,
    simulate_certification,
    simulate_drift,
    stable_seed,
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


DEFAULT_PENDING_DEPTHS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
)


def freshest_k_completion_times(
    *,
    looks: list[int],
    service_cost: int,
    workers: int,
    pending_depth: int,
) -> tuple[np.ndarray, int]:
    if pending_depth <= 0:
        raise ValueError(
            "Pending depth must be positive"
        )
    running: list[
        tuple[int, int]
    ] = []
    pending: list[int] = []
    completion = np.full(
        len(looks),
        -1,
        dtype=np.int64,
    )

    def finish_until(
        time_limit: int | None,
    ) -> None:
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
            if pending:
                next_job = pending.pop(0)
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
            continue
        pending.append(job_index)
        if len(pending) > pending_depth:
            pending.pop(0)
    finish_until(None)
    if pending:
        raise RuntimeError(
            "Pending queue did not drain"
        )
    processed = int(
        (completion >= 0).sum()
    )
    if completion[-1] < 0:
        raise RuntimeError(
            "Final checkpoint was not processed"
        )
    return completion, processed


def parse_depths(
    value: str,
) -> list[int]:
    depths = sorted(
        {
            int(item)
            for item in value.split(",")
            if item.strip()
        }
    )
    if not depths or depths[0] <= 0:
        raise ValueError(
            "Pending depths must be positive"
        )
    return depths


def records_for_depth(
    *,
    records: list[dict[str, Any]],
    pending_depth: int,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["pending_depth"]
        == pending_depth
    ]


def queue_depth_summary(
    *,
    records: list[dict[str, Any]],
    queue_profiles: dict[
        tuple[int, str, int, str],
        dict[str, Any],
    ],
    pending_depth: int,
    profiles: list[dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    depth_records = records_for_depth(
        records=records,
        pending_depth=pending_depth,
    )
    aggregate = aggregate_records(
        depth_records
    )
    queue_values = [
        value
        for (
            depth,
            _,
            _,
            _,
        ), value in queue_profiles.items()
        if depth == pending_depth
    ]
    processed = sum(
        value["processed_look_count"]
        for value in queue_values
    )
    offered = sum(
        value["look_count"]
        for value in queue_values
    )
    result = {
        "combined": combined_comparison(
            depth_records
        ),
        "maximum_aggregate_regression": (
            maximum_aggregate_regression(
                aggregate
            )
        ),
        "maximum_profile_regression": (
            maximum_regression_record(
                depth_records
            )
        ),
        "queue_work": {
            "queue_profile_count": (
                len(queue_values)
            ),
            "processed_look_count": (
                processed
            ),
            "offered_look_count": offered,
            "processed_look_fraction": (
                processed / offered
            ),
            "dropped_look_fraction": (
                1.0 - processed / offered
            ),
            "queue_profiles_with_drops": (
                sum(
                    value[
                        "dropped_look_count"
                    ]
                    > 0
                    for value in queue_values
                )
            ),
        },
    }
    if profiles is not None:
        result["by_breach_metric"] = (
            metric_comparisons(
                records=depth_records,
                profiles=profiles,
            )
        )
    return result


def risk_cap_selections(
    *,
    depth_summaries: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    caps = {
        "aggregate_0p1pct": (
            "combined",
            0.001,
        ),
        "aggregate_0p01pct": (
            "combined",
            0.0001,
        ),
        "aggregate_zero_observed": (
            "combined",
            0.0,
        ),
        "worst_profile_0p1pct": (
            "maximum_profile_regression",
            0.001,
        ),
        "worst_profile_0p01pct": (
            "maximum_profile_regression",
            0.0001,
        ),
        "worst_profile_zero_observed": (
            "maximum_profile_regression",
            0.0,
        ),
    }
    selections = {}
    ordered = sorted(
        (
            int(depth),
            values,
        )
        for depth, values
        in depth_summaries.items()
    )
    for name, (
        source,
        cap,
    ) in caps.items():
        selection = None
        for depth, values in ordered:
            if source == "combined":
                fraction = values[
                    "combined"
                ][
                    "regression_among_fifo_detected"
                ]["fraction"]
            else:
                fraction = values[
                    "maximum_profile_regression"
                ][
                    "regression_among_fifo_detected"
                ]["fraction"]
            if (
                fraction is not None
                and fraction <= cap
            ):
                selection = {
                    "pending_depth": depth,
                    "regression_fraction": (
                        fraction
                    ),
                    "faster_fraction": (
                        values["combined"][
                            "faster_among_fifo_detected"
                        ]["fraction"]
                    ),
                    "processed_look_fraction": (
                        values["queue_work"][
                            "processed_look_fraction"
                        ]
                    ),
                    "fifo_equivalent": (
                        values["queue_work"][
                            "dropped_look_fraction"
                        ]
                        == 0.0
                    ),
                }
                break
        selections[name] = selection
    return selections


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
        tuple[int, str, int, str],
        dict[str, Any],
    ] = {}
    depth_one_checks = 0
    fifo_limit_checks = 0
    final_look_checks = 0
    depth_monotonicity_checks = 0
    adversarial_counterexamples = {
        str(depth): 0
        for depth in args.pending_depths
    }
    for schedule, looks in (
        schedules.items()
    ):
        for service_cost in (
            args.service_costs
        ):
            for worker_policy, workers in (
                WORKERS.items()
            ):
                fifo = completion_times(
                    looks=looks,
                    service_cost=service_cost,
                    workers=workers,
                )
                depth_one, _ = (
                    coalesced_completion_times(
                        looks=looks,
                        service_cost=service_cost,
                        workers=workers,
                    )
                )
                previous_processed = 0
                for pending_depth in (
                    args.pending_depths
                ):
                    (
                        completion,
                        processed,
                    ) = freshest_k_completion_times(
                        looks=looks,
                        service_cost=service_cost,
                        workers=workers,
                        pending_depth=(
                            pending_depth
                        ),
                    )
                    if pending_depth == 1:
                        if not np.array_equal(
                            completion,
                            depth_one,
                        ):
                            raise RuntimeError(
                                "Depth one did not "
                                "reproduce E78"
                            )
                        depth_one_checks += 1
                    if (
                        completion[-1] < 0
                        or processed
                        != int(
                            (
                                completion >= 0
                            ).sum()
                        )
                    ):
                        raise RuntimeError(
                            "Queue accounting failed"
                        )
                    final_look_checks += 1
                    if processed < previous_processed:
                        raise RuntimeError(
                            "Processed work decreased "
                            "with pending depth"
                        )
                    if previous_processed:
                        depth_monotonicity_checks += 1
                    previous_processed = processed
                    dropped = (
                        len(looks) - processed
                    )
                    if dropped > 0:
                        adversarial_counterexamples[
                            str(pending_depth)
                        ] += 1
                    queue_profiles[
                        (
                            pending_depth,
                            schedule,
                            service_cost,
                            worker_policy,
                        )
                    ] = {
                        "fifo": fifo,
                        "completion": (
                            completion
                        ),
                        "look_count": len(
                            looks
                        ),
                        "processed_look_count": (
                            processed
                        ),
                        "dropped_look_count": (
                            dropped
                        ),
                    }
                (
                    fifo_limit,
                    fifo_limit_processed,
                ) = freshest_k_completion_times(
                    looks=looks,
                    service_cost=service_cost,
                    workers=workers,
                    pending_depth=len(looks),
                )
                if (
                    fifo_limit_processed
                    != len(looks)
                    or not np.array_equal(
                        fifo_limit,
                        fifo,
                    )
                ):
                    raise RuntimeError(
                        "Large pending depth did not "
                        "reproduce FIFO"
                    )
                fifo_limit_checks += 1
    certification_records = []
    drift_records = []
    drift_null_records = []
    null_subset_checks = 0
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
                        for pending_depth in (
                            args.pending_depths
                        ):
                            queue = queue_profiles[
                                (
                                    pending_depth,
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
                            record = {
                                "profile_id": (
                                    profile[
                                        "profile_id"
                                    ]
                                ),
                                "pending_depth": (
                                    pending_depth
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
                                "processed_look_count": (
                                    queue[
                                        "processed_look_count"
                                    ]
                                ),
                                "processed_look_fraction": (
                                    queue[
                                        "processed_look_count"
                                    ]
                                    / queue[
                                        "look_count"
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
                        fifo = queue_profiles[
                            (
                                args.pending_depths[
                                    0
                                ],
                                schedule,
                                service_cost,
                                worker_policy,
                            )
                        ]["fifo"]
                        fifo_alert = (
                            action_comparison(
                                crossings=(
                                    null_crossings
                                ),
                                fifo_completion=fifo,
                                coalesced_completion=(
                                    fifo
                                ),
                            )[0][
                                "fifo_detection"
                            ]
                        )
                        for pending_depth in (
                            args.pending_depths
                        ):
                            queue = queue_profiles[
                                (
                                    pending_depth,
                                    schedule,
                                    service_cost,
                                    worker_policy,
                                )
                            ]
                            comparison, _ = (
                                action_comparison(
                                    crossings=(
                                        null_crossings
                                    ),
                                    fifo_completion=(
                                        fifo
                                    ),
                                    coalesced_completion=(
                                        queue[
                                            "completion"
                                        ]
                                    ),
                                )
                            )
                            if (
                                comparison[
                                    "coalesced_detection"
                                ]["successes"]
                                > fifo_alert[
                                    "successes"
                                ]
                            ):
                                raise RuntimeError(
                                    "Null alert subset "
                                    "failed"
                                )
                            null_subset_checks += 1
                            drift_null_records.append(
                                {
                                    "profile_id": (
                                        profile[
                                            "profile_id"
                                        ]
                                    ),
                                    "pending_depth": (
                                        pending_depth
                                    ),
                                    "schedule": (
                                        schedule
                                    ),
                                    "worker_policy": (
                                        worker_policy
                                    ),
                                    "workers": (
                                        workers
                                    ),
                                    "service_cost": (
                                        service_cost
                                    ),
                                    "fifo_false_alert": (
                                        fifo_alert
                                    ),
                                    "coalesced_false_alert": (
                                        comparison[
                                            "coalesced_detection"
                                        ]
                                    ),
                                }
                            )
    certification_by_depth = {
        str(depth): queue_depth_summary(
            records=certification_records,
            queue_profiles=queue_profiles,
            pending_depth=depth,
        )
        for depth in args.pending_depths
    }
    drift_by_depth = {
        str(depth): queue_depth_summary(
            records=drift_records,
            queue_profiles=queue_profiles,
            pending_depth=depth,
            profiles=drift_profiles,
        )
        for depth in args.pending_depths
    }
    null_by_depth = {
        str(depth): null_alert_summary(
            records=[
                record
                for record
                in drift_null_records
                if record[
                    "pending_depth"
                ]
                == depth
            ],
            profiles=drift_profiles,
        )
        for depth in args.pending_depths
    }
    fifo_equivalent_depths = [
        depth
        for depth in args.pending_depths
        if adversarial_counterexamples[
            str(depth)
        ]
        == 0
    ]
    result = {
        "suite": (
            "pacing_coalescing_depth"
        ),
        "model": (
            "Keep the newest K pending "
            "checkpoints while running monitor "
            "jobs finish nonpreemptively. When "
            "the bounded pending queue is full, "
            "drop its oldest checkpoint. Process "
            "retained work in checkpoint order."
        ),
        "source": {
            "stochastic_audit": (
                "results/raw/"
                "pacing_coalescing_stochastic/"
                "pacing_coalescing_stochastic_full.json"
            ),
            "monitor_coalescing": (
                "results/raw/"
                "pacing_monitor_coalescing/"
                "pacing_monitor_coalescing_full.json"
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
            "pending_depths": (
                args.pending_depths
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
            "certification_by_pending_depth": (
                certification_by_depth
            ),
            "drift_by_pending_depth": (
                drift_by_depth
            ),
            "drift_null_by_pending_depth": (
                null_by_depth
            ),
            "drift_risk_cap_selections": (
                risk_cap_selections(
                    depth_summaries=(
                        drift_by_depth
                    )
                )
            ),
            "strict_pathwise_dominance": {
                "requirement": (
                    "Any outcome-independent "
                    "policy that drops a declared "
                    "look admits an adversarial "
                    "path where only that look "
                    "crosses, so strict FIFO "
                    "action dominance requires "
                    "processing every look."
                ),
                "queue_profiles_with_adversarial_counterexample_by_depth": (
                    adversarial_counterexamples
                ),
                "every_dropping_depth_has_counterexample": (
                    all(
                        count > 0
                        for depth, count in (
                            (
                                depth,
                                adversarial_counterexamples[
                                    str(depth)
                                ],
                            )
                            for depth
                            in args.pending_depths
                        )
                        if depth
                        not in fifo_equivalent_depths
                    )
                ),
                "fifo_equivalent_depths": (
                    fifo_equivalent_depths
                ),
                "smallest_fifo_equivalent_depth": (
                    min(
                        fifo_equivalent_depths
                    )
                    if fifo_equivalent_depths
                    else None
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
            "depth_one_e78_reproduction_checks": (
                depth_one_checks
            ),
            "large_depth_fifo_reproduction_checks": (
                fifo_limit_checks
            ),
            "queue_final_look_and_accounting_checks": (
                final_look_checks
            ),
            "processed_work_depth_monotonicity_checks": (
                depth_monotonicity_checks
            ),
            "null_alert_subset_checks": (
                null_subset_checks
            ),
        },
        "limitations": [
            "The audit reuses E79's stratified profiles, independent Bernoulli paths, and recursive E70 grids.",
            "Bounded freshest-K policies remain outcome-independent and alpha-safe, but no finite depth that drops a look can guarantee strict pathwise FIFO latency dominance.",
            "Service cost is fixed, running jobs are nonpreemptive, and retained pending jobs are processed oldest first.",
            "Aggregate comparisons repeat stochastic paths across queue depths and configurations and are not independent Monte Carlo samples.",
            "Correlated traffic, variable query costs, preemption, and E75 phase/rounding grids remain unmodeled.",
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
        "--pending-depths",
        type=parse_depths,
        default=list(
            DEFAULT_PENDING_DEPTHS
        ),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
