from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
    geometric_looks,
)
from analyze_pacing_coalescing_batch import (
    batch_service_cost,
    fraction_key,
    fraction_summary,
    marginal_cap_selections,
    merged_pending_completion_times,
    parse_fractions,
)
from analyze_pacing_coalescing_stochastic import (
    WORKERS,
    action_comparison,
    action_times,
    certification_key,
    combined_comparison,
    drift_key,
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
from analyze_pacing_monitor_workers import (
    completion_times,
)


STREAMING_MARGINAL_FRACTIONS = (
    0.0,
    0.0001,
    0.0005,
    0.001,
    0.002,
    0.0025,
    0.003,
    0.004,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.25,
    0.50,
    1.0,
)


def streaming_merged_completion_times(
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
        member_elapsed = np.asarray(
            [
                batch_service_cost(
                    service_cost=service_cost,
                    batch_size=index + 1,
                    marginal_fraction=(
                        marginal_fraction
                    ),
                )
                for index in range(size)
            ],
            dtype=np.int64,
        )
        completion[members] = (
            started + member_elapsed
        )
        batch_sizes.append(size)
        batch_costs.append(
            int(member_elapsed[-1])
        )
        heapq.heappush(
            running,
            (
                started
                + int(member_elapsed[-1]),
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
            finished, _ = heapq.heappop(
                running
            )
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
    if pending or running:
        raise RuntimeError(
            "Streaming batch queue did not drain"
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
        "batch_size": summarize(
            batch_sizes_array
        ),
        "maximum_batch_size": int(
            batch_sizes_array.max()
        ),
    }


def comparison_field_summary(
    *,
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    return combined_comparison(
        [
            {
                "comparison": (
                    record[field]
                )
            }
            for record in records
        ]
    )


def streaming_fraction_summary(
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
    result = fraction_summary(
        records=records,
        queue_profiles=queue_profiles,
        marginal_fraction=(
            marginal_fraction
        ),
        profiles=profiles,
    )
    selected = [
        record
        for record in records
        if record["marginal_fraction"]
        == marginal_fraction
    ]
    result["comparison_vs_batch_fence"] = (
        comparison_field_summary(
            records=selected,
            field=(
                "comparison_vs_batch_fence"
            ),
        )
    )
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
    queue_work_equality_checks = 0
    fence_dominance_checks = 0
    zero_marginal_fence_equality_checks = 0
    for marginal_fraction in (
        args.marginal_fractions
    ):
        for schedule, looks in (
            schedules.items()
        ):
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
                        streaming,
                        streaming_stats,
                    ) = (
                        streaming_merged_completion_times(
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
                    (
                        fence,
                        fence_stats,
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
                        len(streaming)
                        != len(looks)
                        or np.any(streaming < 0)
                    ):
                        raise RuntimeError(
                            "Not every look was "
                            "streamed exactly once"
                        )
                    all_look_checks += 1
                    if streaming[-1] < 0:
                        raise RuntimeError(
                            "Final look was not "
                            "streamed"
                        )
                    final_look_checks += 1
                    if (
                        streaming_stats
                        != fence_stats
                    ):
                        raise RuntimeError(
                            "Streaming changed queue "
                            "or work accounting"
                        )
                    queue_work_equality_checks += 1
                    if np.any(
                        streaming > fence
                    ):
                        raise RuntimeError(
                            "Streaming member result "
                            "was later than the batch "
                            "fence"
                        )
                    fence_dominance_checks += 1
                    if (
                        marginal_fraction == 0.0
                        and not np.array_equal(
                            streaming,
                            fence,
                        )
                    ):
                        raise RuntimeError(
                            "Zero-marginal streaming "
                            "did not reproduce the "
                            "batch fence"
                        )
                    if marginal_fraction == 0.0:
                        zero_marginal_fence_equality_checks += 1
                    queue_profiles[
                        (
                            marginal_fraction,
                            schedule,
                            service_cost,
                            worker_policy,
                        )
                    ] = {
                        "fifo": fifo,
                        "streaming": streaming,
                        "fence": fence,
                        **streaming_stats,
                    }
    certification_records = []
    drift_records = []
    drift_null_records = []
    no_missed_fifo_checks = 0
    no_later_fence_checks = 0
    null_membership_checks = 0
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
                                        "streaming"
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
                                    "Streaming missed an "
                                    "eventual FIFO action"
                                )
                            no_missed_fifo_checks += 1
                            (
                                fence_comparison,
                                _,
                            ) = action_comparison(
                                crossings=(
                                    crossings
                                ),
                                fifo_completion=(
                                    queue["fence"]
                                ),
                                coalesced_completion=(
                                    queue[
                                        "streaming"
                                    ]
                                ),
                            )
                            if (
                                fence_comparison[
                                    "regression_among_fifo_detected"
                                ]["successes"]
                                != 0
                            ):
                                raise RuntimeError(
                                    "Streaming action was "
                                    "later than the batch "
                                    "fence"
                                )
                            no_later_fence_checks += 1
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
                                "comparison_vs_batch_fence": (
                                    fence_comparison
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
                            alerts = []
                            for completion in (
                                queue["fifo"],
                                queue["fence"],
                                queue[
                                    "streaming"
                                ],
                            ):
                                _, detected = (
                                    action_times(
                                        crossings=(
                                            null_crossings
                                        ),
                                        completion=(
                                            completion
                                        ),
                                    )
                                )
                                alerts.append(
                                    detected
                                )
                            if not (
                                np.array_equal(
                                    alerts[0],
                                    alerts[1],
                                )
                                and np.array_equal(
                                    alerts[0],
                                    alerts[2],
                                )
                            ):
                                raise RuntimeError(
                                    "Streaming changed "
                                    "null alert membership"
                                )
                            null_membership_checks += 1
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
                                                alerts[
                                                    0
                                                ].sum()
                                            ),
                                            total=args.paths,
                                        )
                                    ),
                                    "coalesced_false_alert": (
                                        wilson_interval(
                                            successes=int(
                                                alerts[
                                                    2
                                                ].sum()
                                            ),
                                            total=args.paths,
                                        )
                                    ),
                                }
                            )
    cert_by_fraction = {
        fraction_key(fraction): (
            streaming_fraction_summary(
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
            streaming_fraction_summary(
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
            "pacing_coalescing_streaming"
        ),
        "model": (
            "Use E81's exact merged pending "
            "batches and identical total service "
            "work, but emit each cumulative look "
            "as soon as its incremental snapshot "
            "finishes in chronological order "
            "instead of waiting for the batch "
            "fence."
        ),
        "source": {
            "exact_batch": (
                "results/raw/"
                "pacing_coalescing_batch/"
                "pacing_coalescing_batch_full.json"
            ),
            "stochastic_audit": (
                "results/raw/"
                "pacing_coalescing_stochastic/"
                "pacing_coalescing_stochastic_full.json"
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
                "queue_and_work_equal_to_batch_fence": (
                    True
                ),
                "per_look_completion_no_later_than_batch_fence": (
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
            "all_look_assignment_checks": (
                all_look_checks
            ),
            "final_look_checks": (
                final_look_checks
            ),
            "queue_and_work_equality_checks": (
                queue_work_equality_checks
            ),
            "per_look_batch_fence_dominance_checks": (
                fence_dominance_checks
            ),
            "zero_marginal_fence_equality_checks": (
                zero_marginal_fence_equality_checks
            ),
            "no_missed_fifo_action_checks": (
                no_missed_fifo_checks
            ),
            "no_later_batch_fence_action_checks": (
                no_later_fence_checks
            ),
            "null_alert_membership_equality_checks": (
                null_membership_checks
            ),
        },
        "limitations": [
            "Member snapshots are processed chronologically inside one batch; alternative ordering can change action latency.",
            "Incremental completion uses the same linear marginal service model as E81 and leaves the worker occupied until the final member completes.",
            "One pending batch is launched on the first free worker; pending work is not partitioned across simultaneously idle workers.",
            "The audit reuses E79's stratified profiles, independent Bernoulli paths, and recursive E70 grids.",
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
            STREAMING_MARGINAL_FRACTIONS
        ),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
