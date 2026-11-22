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
from analyze_pacing_coalescing_batch import (
    batch_service_cost,
    fraction_key,
    parse_fractions,
)
from analyze_pacing_coalescing_stochastic import (
    WORKERS,
    action_times,
    certification_key,
    drift_key,
    public_profile,
    select_profiles,
    simulate_certification,
    simulate_drift,
    stable_seed,
    summarize,
    validate_crossing_equivalence,
    wilson_interval,
)
from analyze_pacing_coalescing_streaming import (
    streaming_merged_completion_times,
)
from analyze_pacing_kl_monitoring import (
    load_e68,
)
from analyze_pacing_monitor_workers import (
    completion_times,
)


ORDER_POLICIES = (
    "oldest_first",
    "newest_first",
    "oldest_newest",
    "newest_oldest",
)

ORDER_TIE_RANK = {
    "oldest_first": 0,
    "oldest_newest": 1,
    "newest_oldest": 2,
    "newest_first": 3,
}

DEFAULT_MARGINAL_FRACTIONS = (
    0.0,
    0.0025,
    0.01,
    0.10,
    0.50,
    1.0,
)


def member_order(
    *,
    size: int,
    policy: str,
) -> list[int]:
    if size <= 0:
        raise ValueError(
            "Batch size must be positive"
        )
    if policy == "oldest_first":
        return list(range(size))
    if policy == "newest_first":
        return list(
            range(size - 1, -1, -1)
        )
    result = []
    low = 0
    high = size - 1
    newest_first = (
        policy == "newest_oldest"
    )
    if policy not in (
        "oldest_newest",
        "newest_oldest",
    ):
        raise ValueError(
            f"Unknown order policy: {policy}"
        )
    while low <= high:
        if newest_first:
            result.append(high)
            high -= 1
            if low <= high:
                result.append(low)
                low += 1
        else:
            result.append(low)
            low += 1
            if low <= high:
                result.append(high)
                high -= 1
    return result


def ordered_streaming_completion_times(
    *,
    looks: list[int],
    service_cost: int,
    workers: int,
    marginal_fraction: float,
    order_policy: str,
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
        order = member_order(
            size=size,
            policy=order_policy,
        )
        elapsed = np.asarray(
            [
                batch_service_cost(
                    service_cost=service_cost,
                    batch_size=rank + 1,
                    marginal_fraction=(
                        marginal_fraction
                    ),
                )
                for rank in range(size)
            ],
            dtype=np.int64,
        )
        ordered_members = [
            members[position]
            for position in order
        ]
        completion[
            ordered_members
        ] = started + elapsed
        batch_sizes.append(size)
        batch_costs.append(
            int(elapsed[-1])
        )
        heapq.heappush(
            running,
            (
                started + int(elapsed[-1]),
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
                members = list(pending)
                pending.clear()
                start_batch(
                    started=finished,
                    members=members,
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
            "Ordered streaming queue "
            "did not drain"
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


def new_accumulator(
    *,
    keep_values: bool = False,
) -> dict[str, Any]:
    return {
        "total": 0,
        "faster": 0,
        "equal": 0,
        "slower": 0,
        "missed": 0,
        "positive_count": 0,
        "positive_sum": 0.0,
        "positive_maximum": None,
        "savings_count": 0,
        "savings_sum": 0.0,
        "savings_maximum": None,
        "positive_values": (
            []
            if keep_values
            else None
        ),
        "savings_values": (
            []
            if keep_values
            else None
        ),
    }


def update_accumulator(
    *,
    accumulator: dict[str, Any],
    reference_times: np.ndarray,
    reference_detected: np.ndarray,
    candidate_times: np.ndarray,
    candidate_detected: np.ndarray,
) -> np.ndarray:
    if not np.array_equal(
        reference_detected,
        candidate_detected,
    ):
        raise RuntimeError(
            "Exact order policy changed "
            "eventual action membership"
        )
    delta = (
        candidate_times[
            reference_detected
        ]
        - reference_times[
            reference_detected
        ]
    )
    faster = delta < 0
    equal = delta == 0
    slower = delta > 0
    accumulator["total"] += int(
        delta.size
    )
    accumulator["faster"] += int(
        faster.sum()
    )
    accumulator["equal"] += int(
        equal.sum()
    )
    accumulator["slower"] += int(
        slower.sum()
    )
    positive = delta[slower]
    savings = -delta[faster]
    if positive.size:
        accumulator[
            "positive_count"
        ] += int(positive.size)
        accumulator[
            "positive_sum"
        ] += float(positive.sum())
        maximum = float(
            positive.max()
        )
        accumulator[
            "positive_maximum"
        ] = max(
            maximum,
            accumulator[
                "positive_maximum"
            ]
            if accumulator[
                "positive_maximum"
            ]
            is not None
            else maximum,
        )
        if (
            accumulator[
                "positive_values"
            ]
            is not None
        ):
            accumulator[
                "positive_values"
            ].extend(
                positive.tolist()
            )
    if savings.size:
        accumulator[
            "savings_count"
        ] += int(savings.size)
        accumulator[
            "savings_sum"
        ] += float(savings.sum())
        maximum = float(
            savings.max()
        )
        accumulator[
            "savings_maximum"
        ] = max(
            maximum,
            accumulator[
                "savings_maximum"
            ]
            if accumulator[
                "savings_maximum"
            ]
            is not None
            else maximum,
        )
        if (
            accumulator[
                "savings_values"
            ]
            is not None
        ):
            accumulator[
                "savings_values"
            ].extend(
                savings.tolist()
            )
    return delta


def finalize_accumulator(
    accumulator: dict[str, Any],
) -> dict[str, Any]:
    total = accumulator["total"]
    positive_values = accumulator[
        "positive_values"
    ]
    savings_values = accumulator[
        "savings_values"
    ]
    positive_summary = (
        summarize(
            positive_values
        )
        if positive_values is not None
        else {
            "count": accumulator[
                "positive_count"
            ],
            "mean": (
                accumulator[
                    "positive_sum"
                ]
                / accumulator[
                    "positive_count"
                ]
                if accumulator[
                    "positive_count"
                ]
                else None
            ),
            "maximum": accumulator[
                "positive_maximum"
            ],
        }
    )
    savings_summary = (
        summarize(
            savings_values
        )
        if savings_values is not None
        else {
            "count": accumulator[
                "savings_count"
            ],
            "mean": (
                accumulator[
                    "savings_sum"
                ]
                / accumulator[
                    "savings_count"
                ]
                if accumulator[
                    "savings_count"
                ]
                else None
            ),
            "maximum": accumulator[
                "savings_maximum"
            ],
        }
    )
    return {
        "comparison_count": total,
        "faster": wilson_interval(
            successes=accumulator[
                "faster"
            ],
            total=total,
        ),
        "equal": wilson_interval(
            successes=accumulator[
                "equal"
            ],
            total=total,
        ),
        "slower": wilson_interval(
            successes=accumulator[
                "slower"
            ],
            total=total,
        ),
        "missed": wilson_interval(
            successes=accumulator[
                "missed"
            ],
            total=total,
        ),
        "positive_delay_requests": (
            positive_summary
        ),
        "request_savings": (
            savings_summary
        ),
    }


def record_summary(
    delta: np.ndarray,
) -> dict[str, Any]:
    positive = delta[
        delta > 0
    ]
    savings = -delta[
        delta < 0
    ]
    return {
        "comparison_count": int(
            delta.size
        ),
        "faster": wilson_interval(
            successes=int(
                (delta < 0).sum()
            ),
            total=int(delta.size),
        ),
        "equal": wilson_interval(
            successes=int(
                (delta == 0).sum()
            ),
            total=int(delta.size),
        ),
        "slower": wilson_interval(
            successes=int(
                (delta > 0).sum()
            ),
            total=int(delta.size),
        ),
        "positive_delay_requests": (
            summarize(positive)
        ),
        "request_savings": (
            summarize(savings)
        ),
    }


def summary_for_order(
    *,
    domain: str,
    order_policy: str,
    marginal_fraction: float,
    combined: dict[
        tuple[str, str, float],
        dict[str, Any],
    ],
    versus_oldest: dict[
        tuple[str, str, float],
        dict[str, Any],
    ],
    configurations: dict[
        tuple[
            str,
            str,
            float,
            str,
            str,
            int,
        ],
        dict[str, Any],
    ],
    maximum_profiles: dict[
        tuple[str, str, float],
        dict[str, Any],
    ],
    metrics: dict[
        tuple[str, float, str],
        dict[str, Any],
    ],
    queue_work: dict[
        float,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    configuration_values = [
        (
            key,
            finalize_accumulator(
                accumulator
            ),
        )
        for key, accumulator
        in configurations.items()
        if key[0] == domain
        and key[1] == order_policy
        and key[2] == marginal_fraction
    ]
    max_key, maximum_configuration = max(
        configuration_values,
        key=lambda item: (
            item[1]["slower"][
                "fraction"
            ]
            or 0.0,
            item[1][
                "positive_delay_requests"
            ]["maximum"]
            or -1.0,
            item[0],
        ),
    )
    result = {
        "combined": finalize_accumulator(
            combined[
                (
                    domain,
                    order_policy,
                    marginal_fraction,
                )
            ]
        ),
        "comparison_vs_oldest_first": (
            finalize_accumulator(
                versus_oldest[
                    (
                        domain,
                        order_policy,
                        marginal_fraction,
                    )
                ]
            )
        ),
        "maximum_aggregate_regression": {
            "schedule": max_key[3],
            "worker_policy": max_key[4],
            "service_cost": max_key[5],
            **maximum_configuration,
        },
        "maximum_profile_regression": (
            maximum_profiles[
                (
                    domain,
                    order_policy,
                    marginal_fraction,
                )
            ]
        ),
        "queue_work": queue_work[
            marginal_fraction
        ],
    }
    if domain == "drift":
        result[
            "by_breach_metric"
        ] = {
            metric: (
                finalize_accumulator(
                    metrics[
                        (
                            order_policy,
                            marginal_fraction,
                            metric,
                        )
                    ]
                )
            )
            for metric in (
                "state_error",
                "mistake_flag_recall",
                "false_alarm_rate",
            )
        }
    return result


def select_orders(
    summaries: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    aggregate = min(
        summaries,
        key=lambda policy: (
            summaries[policy][
                "combined"
            ]["slower"]["fraction"],
            summaries[policy][
                "maximum_profile_regression"
            ]["slower"]["fraction"],
            -summaries[policy][
                "combined"
            ]["faster"]["fraction"],
            ORDER_TIE_RANK[policy],
        ),
    )
    minimax = min(
        summaries,
        key=lambda policy: (
            summaries[policy][
                "maximum_profile_regression"
            ]["slower"]["fraction"],
            summaries[policy][
                "combined"
            ]["slower"]["fraction"],
            summaries[policy][
                "maximum_profile_regression"
            ][
                "positive_delay_requests"
            ]["maximum"]
            or -1.0,
            -summaries[policy][
                "combined"
            ]["faster"]["fraction"],
            ORDER_TIE_RANK[policy],
        ),
    )
    delay_aware = min(
        summaries,
        key=lambda policy: (
            summaries[policy][
                "maximum_profile_regression"
            ]["slower"]["fraction"],
            summaries[policy][
                "combined"
            ][
                "positive_delay_requests"
            ]["maximum"]
            or -1.0,
            summaries[policy][
                "combined"
            ]["slower"]["fraction"],
            -summaries[policy][
                "combined"
            ]["faster"]["fraction"],
            ORDER_TIE_RANK[policy],
        ),
    )
    return {
        "aggregate_regression_minimizer": (
            aggregate
        ),
        "worst_profile_minimizer": (
            minimax
        ),
        "delay_aware_minimizer": (
            delay_aware
        ),
    }


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
    queue_profiles = {}
    queue_work = {}
    all_look_checks = 0
    final_look_checks = 0
    oldest_e82_checks = 0
    work_invariance_checks = 0
    zero_cost_order_equality_checks = 0
    for marginal_fraction in (
        args.marginal_fractions
    ):
        work_totals = {
            "look_count": 0,
            "batch_count": 0,
            "total_service_work": 0,
            "fifo_total_service_work": 0,
            "maximum_batch_size": 0,
        }
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
                    e82, e82_stats = (
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
                    completions = {}
                    reference_stats = None
                    for order_policy in (
                        ORDER_POLICIES
                    ):
                        completion, stats = (
                            ordered_streaming_completion_times(
                                looks=looks,
                                service_cost=(
                                    service_cost
                                ),
                                workers=workers,
                                marginal_fraction=(
                                    marginal_fraction
                                ),
                                order_policy=(
                                    order_policy
                                ),
                            )
                        )
                        if np.any(
                            completion < 0
                        ):
                            raise RuntimeError(
                                "Order did not assign "
                                "every look"
                            )
                        all_look_checks += 1
                        if completion[-1] < 0:
                            raise RuntimeError(
                                "Order did not process "
                                "the final look"
                            )
                        final_look_checks += 1
                        if reference_stats is None:
                            reference_stats = stats
                        elif stats != reference_stats:
                            raise RuntimeError(
                                "Member order changed "
                                "queue or work"
                            )
                        work_invariance_checks += 1
                        if (
                            order_policy
                            == "oldest_first"
                            and (
                                not np.array_equal(
                                    completion,
                                    e82,
                                )
                                or stats
                                != e82_stats
                            )
                        ):
                            raise RuntimeError(
                                "Oldest-first order "
                                "did not reproduce E82"
                            )
                        if (
                            order_policy
                            == "oldest_first"
                        ):
                            oldest_e82_checks += 1
                        completions[
                            order_policy
                        ] = completion
                    if marginal_fraction == 0.0:
                        oldest = completions[
                            "oldest_first"
                        ]
                        for order_policy in (
                            ORDER_POLICIES
                        ):
                            if not np.array_equal(
                                completions[
                                    order_policy
                                ],
                                oldest,
                            ):
                                raise RuntimeError(
                                    "Zero-cost member "
                                    "orders differed"
                                )
                            zero_cost_order_equality_checks += 1
                    queue_profiles[
                        (
                            marginal_fraction,
                            schedule,
                            service_cost,
                            worker_policy,
                        )
                    ] = {
                        "fifo": fifo,
                        "completions": (
                            completions
                        ),
                        "stats": (
                            reference_stats
                        ),
                    }
                    work_totals[
                        "look_count"
                    ] += reference_stats[
                        "look_count"
                    ]
                    work_totals[
                        "batch_count"
                    ] += reference_stats[
                        "batch_count"
                    ]
                    work_totals[
                        "total_service_work"
                    ] += reference_stats[
                        "total_service_work"
                    ]
                    work_totals[
                        "fifo_total_service_work"
                    ] += reference_stats[
                        "fifo_total_service_work"
                    ]
                    work_totals[
                        "maximum_batch_size"
                    ] = max(
                        work_totals[
                            "maximum_batch_size"
                        ],
                        reference_stats[
                            "maximum_batch_size"
                        ],
                    )
        queue_work[
            marginal_fraction
        ] = {
            **work_totals,
            "launch_savings_fraction": (
                1.0
                - work_totals[
                    "batch_count"
                ]
                / work_totals[
                    "look_count"
                ]
            ),
            "service_work_fraction_vs_fifo": (
                work_totals[
                    "total_service_work"
                ]
                / work_totals[
                    "fifo_total_service_work"
                ]
            ),
        }
    combined = defaultdict(
        new_accumulator
    )
    versus_oldest = defaultdict(
        new_accumulator
    )
    configurations = defaultdict(
        new_accumulator
    )
    metrics = defaultdict(
        new_accumulator
    )
    maximum_profiles = {}
    null_membership_checks = 0
    no_missed_action_checks = 0
    profile_metric = {
        profile["profile_id"]: (
            profile["breach_metric"]
        )
        for profile in drift_profiles
    }
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
                                fifo_times,
                                fifo_detected,
                            ) = action_times(
                                crossings=crossings,
                                completion=queue[
                                    "fifo"
                                ],
                            )
                            order_actions = {}
                            for order_policy in (
                                ORDER_POLICIES
                            ):
                                (
                                    times,
                                    detected,
                                ) = action_times(
                                    crossings=(
                                        crossings
                                    ),
                                    completion=queue[
                                        "completions"
                                    ][order_policy],
                                )
                                if not np.array_equal(
                                    fifo_detected,
                                    detected,
                                ):
                                    raise RuntimeError(
                                        "Order changed "
                                        "eventual action "
                                        "membership"
                                    )
                                no_missed_action_checks += 1
                                delta = update_accumulator(
                                    accumulator=combined[
                                        (
                                            domain,
                                            order_policy,
                                            marginal_fraction,
                                        )
                                    ],
                                    reference_times=(
                                        fifo_times
                                    ),
                                    reference_detected=(
                                        fifo_detected
                                    ),
                                    candidate_times=(
                                        times
                                    ),
                                    candidate_detected=(
                                        detected
                                    ),
                                )
                                update_accumulator(
                                    accumulator=(
                                        configurations[
                                            (
                                                domain,
                                                order_policy,
                                                marginal_fraction,
                                                schedule,
                                                worker_policy,
                                                service_cost,
                                            )
                                        ]
                                    ),
                                    reference_times=(
                                        fifo_times
                                    ),
                                    reference_detected=(
                                        fifo_detected
                                    ),
                                    candidate_times=(
                                        times
                                    ),
                                    candidate_detected=(
                                        detected
                                    ),
                                )
                                if domain == "drift":
                                    update_accumulator(
                                        accumulator=(
                                            metrics[
                                                (
                                                    order_policy,
                                                    marginal_fraction,
                                                    profile_metric[
                                                        profile[
                                                            "profile_id"
                                                        ]
                                                    ],
                                                )
                                            ]
                                        ),
                                        reference_times=(
                                            fifo_times
                                        ),
                                        reference_detected=(
                                            fifo_detected
                                        ),
                                        candidate_times=(
                                            times
                                        ),
                                        candidate_detected=(
                                            detected
                                        ),
                                    )
                                record = {
                                    "profile_id": (
                                        profile[
                                            "profile_id"
                                        ]
                                    ),
                                    "schedule": (
                                        schedule
                                    ),
                                    "worker_policy": (
                                        worker_policy
                                    ),
                                    "service_cost": (
                                        service_cost
                                    ),
                                    **record_summary(
                                        delta
                                    ),
                                }
                                key = (
                                    domain,
                                    order_policy,
                                    marginal_fraction,
                                )
                                existing = (
                                    maximum_profiles.get(
                                        key
                                    )
                                )
                                if (
                                    existing is None
                                    or (
                                        record["slower"][
                                            "fraction"
                                        ],
                                        record[
                                            "positive_delay_requests"
                                        ]["maximum"]
                                        or -1.0,
                                        record[
                                            "profile_id"
                                        ],
                                    )
                                    > (
                                        existing[
                                            "slower"
                                        ]["fraction"],
                                        existing[
                                            "positive_delay_requests"
                                        ]["maximum"]
                                        or -1.0,
                                        existing[
                                            "profile_id"
                                        ],
                                    )
                                ):
                                    maximum_profiles[
                                        key
                                    ] = record
                                order_actions[
                                    order_policy
                                ] = (
                                    times,
                                    detected,
                                )
                            oldest_times, oldest_detected = (
                                order_actions[
                                    "oldest_first"
                                ]
                            )
                            for order_policy in (
                                ORDER_POLICIES
                            ):
                                times, detected = (
                                    order_actions[
                                        order_policy
                                    ]
                                )
                                update_accumulator(
                                    accumulator=(
                                        versus_oldest[
                                            (
                                                domain,
                                                order_policy,
                                                marginal_fraction,
                                            )
                                        ]
                                    ),
                                    reference_times=(
                                        oldest_times
                                    ),
                                    reference_detected=(
                                        oldest_detected
                                    ),
                                    candidate_times=(
                                        times
                                    ),
                                    candidate_detected=(
                                        detected
                                    ),
                                )
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
                            reference = None
                            for order_policy in (
                                ORDER_POLICIES
                            ):
                                _, detected = (
                                    action_times(
                                        crossings=(
                                            null_crossings
                                        ),
                                        completion=queue[
                                            "completions"
                                        ][order_policy],
                                    )
                                )
                                if reference is None:
                                    reference = detected
                                elif not np.array_equal(
                                    reference,
                                    detected,
                                ):
                                    raise RuntimeError(
                                        "Member order changed "
                                        "null alert membership"
                                    )
                                null_membership_checks += 1
    certification_headline = {}
    drift_headline = {}
    order_selections = {}
    for marginal_fraction in (
        args.marginal_fractions
    ):
        cert_summaries = {}
        drift_summaries = {}
        for order_policy in (
            ORDER_POLICIES
        ):
            cert_summaries[
                order_policy
            ] = summary_for_order(
                domain="certification",
                order_policy=order_policy,
                marginal_fraction=(
                    marginal_fraction
                ),
                combined=combined,
                versus_oldest=(
                    versus_oldest
                ),
                configurations=(
                    configurations
                ),
                maximum_profiles=(
                    maximum_profiles
                ),
                metrics=metrics,
                queue_work=queue_work,
            )
            drift_summaries[
                order_policy
            ] = summary_for_order(
                domain="drift",
                order_policy=order_policy,
                marginal_fraction=(
                    marginal_fraction
                ),
                combined=combined,
                versus_oldest=(
                    versus_oldest
                ),
                configurations=(
                    configurations
                ),
                maximum_profiles=(
                    maximum_profiles
                ),
                metrics=metrics,
                queue_work=queue_work,
            )
        key = fraction_key(
            marginal_fraction
        )
        certification_headline[
            key
        ] = cert_summaries
        drift_headline[
            key
        ] = drift_summaries
        order_selections[key] = {
            "certification": (
                select_orders(
                    cert_summaries
                )
            ),
            "drift": select_orders(
                drift_summaries
            ),
        }
    result = {
        "suite": (
            "pacing_coalescing_order"
        ),
        "model": (
            "Hold E82 exact streaming batch "
            "formation, launches, and total work "
            "fixed while changing the outcome-"
            "independent member evaluation order."
        ),
        "source": {
            "streaming_batch": (
                "results/raw/"
                "pacing_coalescing_streaming/"
                "pacing_coalescing_streaming_full.json"
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
            "order_policies": (
                ORDER_POLICIES
            ),
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
        "order_policy_definitions": {
            "oldest_first": (
                "Evaluate retained members in "
                "checkpoint order; reproduces E82."
            ),
            "newest_first": (
                "Evaluate newest checkpoint first, "
                "then move backward."
            ),
            "oldest_newest": (
                "Alternate oldest, newest, second "
                "oldest, second newest."
            ),
            "newest_oldest": (
                "Alternate newest, oldest, second "
                "newest, second oldest."
            ),
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
        "headline": {
            "queue_work_by_marginal_fraction": {
                fraction_key(
                    fraction
                ): values
                for fraction, values
                in queue_work.items()
            },
            "certification_by_marginal_fraction_and_order": (
                certification_headline
            ),
            "drift_by_marginal_fraction_and_order": (
                drift_headline
            ),
            "order_selections": (
                order_selections
            ),
            "exact_statistical_equivalence": {
                "all_looks_evaluated": True,
                "no_missed_actions": True,
                "null_alert_membership_equal_across_orders": (
                    True
                ),
                "queue_and_work_equal_across_orders": (
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
            "oldest_first_e82_reproduction_checks": (
                oldest_e82_checks
            ),
            "queue_and_work_order_invariance_checks": (
                work_invariance_checks
            ),
            "zero_cost_all_order_equality_checks": (
                zero_cost_order_equality_checks
            ),
            "no_missed_action_checks": (
                no_missed_action_checks
            ),
            "null_alert_membership_equality_checks": (
                null_membership_checks
            ),
        },
        "limitations": [
            "Only four simple outcome-independent member orders are tested.",
            "Batch formation, worker assignment, and linear marginal service cost remain fixed to E82.",
            "Compact output retains aggregate, configuration, metric, and worst-profile summaries rather than every per-profile record.",
            "The audit reuses E79's stratified profiles, independent Bernoulli paths, and recursive E70 grids.",
            "Aggregate comparisons repeat stochastic paths across orders and configurations and are not independent Monte Carlo samples.",
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
