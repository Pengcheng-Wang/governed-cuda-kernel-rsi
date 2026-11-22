from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pacing_checkpoint_monitoring import (
    LOOK_SCHEDULES,
    geometric_looks,
    look_alpha,
)
from analyze_pacing_kl_monitoring import (
    kl_lower_bound,
    kl_upper_bound,
    load_e68,
)
from analyze_pacing_monitor_coalescing import (
    coalesced_completion_times,
    next_processed_completion,
)
from analyze_pacing_monitor_workers import (
    completion_times,
)


WORKERS = {
    "workers_1": 1,
    "workers_2": 2,
    "workers_4": 4,
    "workers_8": 8,
}

QUANTILES = (
    ("low", 0.0),
    ("median", 0.5),
    ("high", 1.0),
)


def summarize(
    values: np.ndarray | list[float],
) -> dict[str, Any]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )
    if array.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def wilson_interval(
    *,
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> dict[str, Any]:
    if total <= 0:
        return {
            "successes": successes,
            "total": total,
            "fraction": None,
            "lower_95pct": None,
            "upper_95pct": None,
        }
    fraction = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (
        fraction + z2 / (2.0 * total)
    ) / denominator
    half = (
        z
        * math.sqrt(
            fraction
            * (1.0 - fraction)
            / total
            + z2
            / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "fraction": fraction,
        "lower_95pct": max(
            0.0,
            center - half,
        ),
        "upper_95pct": min(
            1.0,
            center + half,
        ),
    }


def stable_seed(
    base_seed: int,
    *parts: str,
) -> int:
    digest = hashlib.sha256(
        (
            str(base_seed)
            + "|"
            + "|".join(parts)
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    )


def certification_key(
    record: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        record["confidence_profile"],
        record["calibration_scenario"],
        record["dimension_count"],
        record["actual_state_error"],
        record[
            "actual_mistake_flag_recall"
        ],
        record["actual_false_alarm_rate"],
        record["state_error_cap"],
        record["required_recall"],
        record["false_alarm_cap"],
        record["stream_alpha"],
    )


def drift_key(
    record: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        record["confidence_profile"],
        record["breach_metric"],
        record["dimension_count"],
        record["conditioning_probability"],
        record["post_change_actual_rate"],
        record["safe_cap"],
        record["stream_alpha"],
    )


def select_profiles(
    *,
    records: list[dict[str, Any]],
    status: str,
    key_function: Any,
    strata: tuple[str, ...],
    rank_field: str,
    prefix: str,
) -> list[dict[str, Any]]:
    unique = {}
    for record in records:
        if record["status"] != status:
            continue
        key = key_function(record)
        unique.setdefault(
            key,
            record,
        )
    grouped: dict[
        tuple[Any, ...],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for record in unique.values():
        grouped[
            tuple(
                record[field]
                for field in strata
            )
        ].append(record)
    selected = []
    for stratum, values in sorted(
        grouped.items(),
        key=lambda item: tuple(
            str(value)
            for value in item[0]
        ),
    ):
        ordered = sorted(
            values,
            key=lambda record: (
                record[rank_field],
                key_function(record),
            ),
        )
        used = set()
        for quantile_name, quantile in (
            QUANTILES
        ):
            index = int(
                round(
                    quantile
                    * (len(ordered) - 1)
                )
            )
            record = ordered[index]
            key = key_function(record)
            if key in used:
                continue
            used.add(key)
            profile = dict(record)
            profile["profile_id"] = (
                f"{prefix}_"
                + "_".join(
                    str(value)
                    for value in stratum
                )
                + f"_{quantile_name}"
            )
            profile[
                "selection_quantile"
            ] = quantile_name
            selected.append(profile)
    return selected


def binary_kl_array(
    empirical: np.ndarray,
    candidate: float,
) -> np.ndarray:
    empirical = np.asarray(
        empirical,
        dtype=np.float64,
    )
    if not 0.0 <= candidate <= 1.0:
        raise ValueError(
            "Candidate outside [0, 1]"
        )
    result = np.zeros_like(empirical)
    if candidate == 0.0:
        result[empirical > 0.0] = math.inf
        return result
    if candidate == 1.0:
        result[empirical < 1.0] = math.inf
        return result
    positive = empirical > 0.0
    result[positive] += (
        empirical[positive]
        * np.log(
            empirical[positive]
            / candidate
        )
    )
    complement = 1.0 - empirical
    positive_complement = complement > 0.0
    result[positive_complement] += (
        complement[positive_complement]
        * np.log(
            complement[
                positive_complement
            ]
            / (1.0 - candidate)
        )
    )
    return result


def directional_crossing(
    *,
    successes: np.ndarray,
    trials: np.ndarray,
    alpha: float,
    cap: float,
    direction: str,
    strict: bool,
) -> np.ndarray:
    if (
        not strict
        and direction == "upper"
        and cap == 1.0
    ):
        return np.ones(
            trials.shape,
            dtype=np.bool_,
        )
    if (
        not strict
        and direction == "lower"
        and cap == 0.0
    ):
        return np.ones(
            trials.shape,
            dtype=np.bool_,
        )
    valid = trials > 0
    empirical = np.zeros(
        trials.shape,
        dtype=np.float64,
    )
    empirical[valid] = (
        successes[valid]
        / trials[valid]
    )
    divergence = np.zeros_like(
        empirical
    )
    divergence[valid] = (
        binary_kl_array(
            empirical[valid],
            cap,
        )
        * trials[valid]
    )
    target = math.log(1.0 / alpha)
    separated = (
        divergence
        > target + 1e-12
        if strict
        else divergence
        >= target - 1e-12
    )
    if direction == "upper":
        directional = (
            empirical < cap
            if strict
            else empirical <= cap
        )
    elif direction == "lower":
        directional = (
            empirical > cap
            if strict
            else empirical >= cap
        )
    else:
        raise ValueError(
            f"Unknown direction: {direction}"
        )
    return (
        valid
        & directional
        & separated
    )


def validate_crossing_equivalence(
    *,
    profiles: list[dict[str, Any]],
) -> int:
    caps = {
        0.0,
        0.05,
        0.5,
        0.95,
        1.0,
    }
    alphas = {
        look_alpha(
            stream_alpha=profile[
                "stream_alpha"
            ],
            look_index=look_index,
        )
        for profile in profiles
        for look_index in (
            1,
            2,
            5,
            10,
            25,
            50,
            100,
        )
    }
    for profile in profiles:
        for field in (
            "state_error_cap",
            "required_recall",
            "false_alarm_cap",
            "safe_cap",
        ):
            if field in profile:
                caps.add(
                    float(profile[field])
                )
    checks = 0
    for trials in (
        1,
        2,
        5,
        10,
        100,
        1000,
    ):
        successes_values = sorted(
            {
                0,
                min(1, trials),
                trials // 4,
                trials // 2,
                max(0, trials - 1),
                trials,
            }
        )
        for successes in successes_values:
            empirical = successes / trials
            success_array = np.asarray(
                [successes],
                dtype=np.int64,
            )
            trial_array = np.asarray(
                [trials],
                dtype=np.int64,
            )
            for cap in sorted(caps):
                for alpha in sorted(alphas):
                    target = (
                        trials
                        * float(
                            binary_kl_array(
                                np.asarray(
                                    [empirical]
                                ),
                                cap,
                            )[0]
                        )
                    )
                    threshold = math.log(
                        1.0 / alpha
                    )
                    if math.isclose(
                        target,
                        threshold,
                        rel_tol=0.0,
                        abs_tol=1e-10,
                    ):
                        continue
                    upper = directional_crossing(
                        successes=success_array,
                        trials=trial_array,
                        alpha=alpha,
                        cap=cap,
                        direction="upper",
                        strict=False,
                    )[0]
                    expected_upper = (
                        kl_upper_bound(
                            empirical=empirical,
                            trials=trials,
                            alpha=alpha,
                        )
                        <= cap + 1e-15
                    )
                    if bool(upper) != bool(
                        expected_upper
                    ):
                        raise RuntimeError(
                            "Direct upper crossing "
                            "did not match KL inversion"
                        )
                    lower = directional_crossing(
                        successes=success_array,
                        trials=trial_array,
                        alpha=alpha,
                        cap=cap,
                        direction="lower",
                        strict=False,
                    )[0]
                    expected_lower = (
                        kl_lower_bound(
                            empirical=empirical,
                            trials=trials,
                            alpha=alpha,
                        )
                        >= cap - 1e-15
                    )
                    if bool(lower) != bool(
                        expected_lower
                    ):
                        raise RuntimeError(
                            "Direct lower crossing "
                            "did not match KL inversion"
                        )
                    checks += 2
    return checks


def simulate_certification(
    *,
    profile: dict[str, Any],
    looks: list[int],
    paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    state_errors = np.zeros(
        paths,
        dtype=np.int64,
    )
    recalled_errors = np.zeros_like(
        state_errors
    )
    false_alarms = np.zeros_like(
        state_errors
    )
    crossings = np.zeros(
        (
            len(looks),
            paths,
        ),
        dtype=np.bool_,
    )
    previous = 0
    for look_index, requests in enumerate(
        looks,
        start=1,
    ):
        increment = requests - previous
        new_errors = rng.binomial(
            increment,
            profile["actual_state_error"],
            size=paths,
        )
        state_errors += new_errors
        recalled_errors += rng.binomial(
            new_errors,
            profile[
                "actual_mistake_flag_recall"
            ],
        )
        new_non_errors = (
            increment - new_errors
        )
        false_alarms += rng.binomial(
            new_non_errors,
            profile[
                "actual_false_alarm_rate"
            ],
        )
        non_errors = (
            requests - state_errors
        )
        alpha = look_alpha(
            stream_alpha=profile[
                "stream_alpha"
            ],
            look_index=look_index,
        )
        state_passes = (
            directional_crossing(
                successes=state_errors,
                trials=np.full(
                    paths,
                    requests,
                    dtype=np.int64,
                ),
                alpha=alpha,
                cap=profile[
                    "state_error_cap"
                ],
                direction="upper",
                strict=False,
            )
        )
        recall_passes = (
            directional_crossing(
                successes=recalled_errors,
                trials=state_errors,
                alpha=alpha,
                cap=profile[
                    "required_recall"
                ],
                direction="lower",
                strict=False,
            )
        )
        false_alarm_passes = (
            directional_crossing(
                successes=false_alarms,
                trials=non_errors,
                alpha=alpha,
                cap=profile[
                    "false_alarm_cap"
                ],
                direction="upper",
                strict=False,
            )
        )
        crossings[
            look_index - 1
        ] = (
            state_passes
            & recall_passes
            & false_alarm_passes
        )
        previous = requests
    return crossings


def simulate_drift(
    *,
    profile: dict[str, Any],
    looks: list[int],
    paths: int,
    rng: np.random.Generator,
    actual_rate: float,
) -> np.ndarray:
    eligible = np.zeros(
        paths,
        dtype=np.int64,
    )
    successes = np.zeros_like(eligible)
    crossings = np.zeros(
        (
            len(looks),
            paths,
        ),
        dtype=np.bool_,
    )
    previous = 0
    direction = (
        "upper"
        if profile["breach_metric"]
        == "mistake_flag_recall"
        else "lower"
    )
    for look_index, requests in enumerate(
        looks,
        start=1,
    ):
        increment = requests - previous
        new_eligible = rng.binomial(
            increment,
            profile[
                "conditioning_probability"
            ],
            size=paths,
        )
        eligible += new_eligible
        successes += rng.binomial(
            new_eligible,
            actual_rate,
        )
        alpha = look_alpha(
            stream_alpha=profile[
                "stream_alpha"
            ],
            look_index=look_index,
        )
        crossings[
            look_index - 1
        ] = directional_crossing(
            successes=successes,
            trials=eligible,
            alpha=alpha,
            cap=profile["safe_cap"],
            direction=direction,
            strict=True,
        )
        previous = requests
    return crossings


def action_times(
    *,
    crossings: np.ndarray,
    completion: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    processed = completion >= 0
    sentinel = np.iinfo(
        np.int64
    ).max
    candidates = np.where(
        crossings
        & processed[:, None],
        completion[:, None],
        sentinel,
    )
    times = candidates.min(axis=0)
    detected = times != sentinel
    return times, detected


def action_comparison(
    *,
    crossings: np.ndarray,
    fifo_completion: np.ndarray,
    coalesced_completion: np.ndarray,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
]:
    fifo_times, fifo_detected = (
        action_times(
            crossings=crossings,
            completion=fifo_completion,
        )
    )
    coalesced_times, coalesced_detected = (
        action_times(
            crossings=crossings,
            completion=(
                coalesced_completion
            ),
        )
    )
    if np.any(
        coalesced_detected
        & ~fifo_detected
    ):
        raise RuntimeError(
            "Coalesced action was not a subset "
            "of FIFO crossings"
        )
    both = (
        fifo_detected
        & coalesced_detected
    )
    delta = (
        coalesced_times[both]
        - fifo_times[both]
    )
    fifo_count = int(
        fifo_detected.sum()
    )
    coalesced_count = int(
        coalesced_detected.sum()
    )
    faster = int(
        (delta < 0).sum()
    )
    equal = int(
        (delta == 0).sum()
    )
    slower = int(
        (delta > 0).sum()
    )
    missed = int(
        (
            fifo_detected
            & ~coalesced_detected
        ).sum()
    )
    any_crossing = crossings.any(
        axis=0
    )
    first_crossing = np.argmax(
        crossings,
        axis=0,
    )
    dropped_first = (
        any_crossing
        & (
            coalesced_completion[
                first_crossing
            ]
            < 0
        )
    )
    next_completion = (
        next_processed_completion(
            coalesced_completion
        )
    )
    next_processed_index = np.empty(
        len(coalesced_completion),
        dtype=np.int64,
    )
    next_index = -1
    for index in range(
        len(coalesced_completion) - 1,
        -1,
        -1,
    ):
        if (
            coalesced_completion[index]
            >= 0
        ):
            next_index = index
        next_processed_index[index] = (
            next_index
        )
    if np.any(
        next_completion
        != coalesced_completion[
            next_processed_index
        ]
    ):
        raise RuntimeError(
            "Next-processed completion mismatch"
        )
    dropped_indices = np.where(
        dropped_first
    )[0]
    if dropped_indices.size:
        next_indices = (
            next_processed_index[
                first_crossing[
                    dropped_indices
                ]
            ]
        )
        persisted = crossings[
            next_indices,
            dropped_indices,
        ]
    else:
        persisted = np.asarray(
            [],
            dtype=np.bool_,
        )
    regression = slower + missed
    return (
        {
            "path_count": (
                crossings.shape[1]
            ),
            "fifo_detection": (
                wilson_interval(
                    successes=fifo_count,
                    total=crossings.shape[1],
                )
            ),
            "coalesced_detection": (
                wilson_interval(
                    successes=coalesced_count,
                    total=crossings.shape[1],
                )
            ),
            "faster_among_fifo_detected": (
                wilson_interval(
                    successes=faster,
                    total=fifo_count,
                )
            ),
            "equal_among_fifo_detected": (
                wilson_interval(
                    successes=equal,
                    total=fifo_count,
                )
            ),
            "slower_among_fifo_detected": (
                wilson_interval(
                    successes=slower,
                    total=fifo_count,
                )
            ),
            "missed_among_fifo_detected": (
                wilson_interval(
                    successes=missed,
                    total=fifo_count,
                )
            ),
            "regression_among_fifo_detected": (
                wilson_interval(
                    successes=regression,
                    total=fifo_count,
                )
            ),
            "action_delta_requests": (
                summarize(delta)
            ),
            "positive_delay_requests": (
                summarize(
                    delta[delta > 0]
                )
            ),
            "request_savings": (
                summarize(
                    -delta[delta < 0]
                )
            ),
            "dropped_first_crossing": (
                wilson_interval(
                    successes=int(
                        dropped_first.sum()
                    ),
                    total=int(
                        any_crossing.sum()
                    ),
                )
            ),
            "next_processed_persistence": (
                wilson_interval(
                    successes=int(
                        persisted.sum()
                    ),
                    total=int(
                        persisted.size
                    ),
                )
            ),
        },
        {
            "fifo_detected": (
                fifo_detected
            ),
            "coalesced_detected": (
                coalesced_detected
            ),
            "delta": delta,
            "faster": delta < 0,
            "equal": delta == 0,
            "slower": delta > 0,
            "missed": (
                fifo_detected
                & ~coalesced_detected
            ),
            "dropped_first": (
                dropped_first
            ),
            "persisted": persisted,
        },
    )


def public_profile(
    profile: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    fields = (
        (
            "profile_id",
            "selection_quantile",
            "confidence_profile",
            "calibration_scenario",
            "dimension_count",
            "actual_state_error",
            "actual_mistake_flag_recall",
            "actual_false_alarm_rate",
            "state_error_cap",
            "required_recall",
            "false_alarm_cap",
            "stream_alpha",
            "required_labeled_requests",
        )
        if domain == "certification"
        else (
            "profile_id",
            "selection_quantile",
            "confidence_profile",
            "breach_metric",
            "dimension_count",
            "conditioning_probability",
            "post_change_actual_rate",
            "safe_cap",
            "stream_alpha",
            "required_post_change_requests",
        )
    )
    return {
        field: profile[field]
        for field in fields
    }


def aggregate_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[
        tuple[str, int, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["worker_policy"],
                record["service_cost"],
                record["schedule"],
            )
        ].append(record)
    result = {}
    for key, values in sorted(
        grouped.items()
    ):
        worker, service, schedule = key
        fifo_total = sum(
            value["comparison"][
                "fifo_detection"
            ]["successes"]
            for value in values
        )
        path_total = sum(
            value["comparison"][
                "fifo_detection"
            ]["total"]
            for value in values
        )
        faster = sum(
            value["comparison"][
                "faster_among_fifo_detected"
            ]["successes"]
            for value in values
        )
        slower = sum(
            value["comparison"][
                "slower_among_fifo_detected"
            ]["successes"]
            for value in values
        )
        missed = sum(
            value["comparison"][
                "missed_among_fifo_detected"
            ]["successes"]
            for value in values
        )
        equal = (
            fifo_total
            - faster
            - slower
            - missed
        )
        dropped = sum(
            value["comparison"][
                "dropped_first_crossing"
            ]["successes"]
            for value in values
        )
        crossings = sum(
            value["comparison"][
                "dropped_first_crossing"
            ]["total"]
            for value in values
        )
        persisted = sum(
            value["comparison"][
                "next_processed_persistence"
            ]["successes"]
            for value in values
        )
        persistence_total = sum(
            value["comparison"][
                "next_processed_persistence"
            ]["total"]
            for value in values
        )
        result[
            f"{worker}|{service}|{schedule}"
        ] = {
            "profile_count": len(values),
            "path_count": path_total,
            "fifo_detection": (
                wilson_interval(
                    successes=fifo_total,
                    total=path_total,
                )
            ),
            "faster_among_fifo_detected": (
                wilson_interval(
                    successes=faster,
                    total=fifo_total,
                )
            ),
            "equal_among_fifo_detected": (
                wilson_interval(
                    successes=equal,
                    total=fifo_total,
                )
            ),
            "slower_among_fifo_detected": (
                wilson_interval(
                    successes=slower,
                    total=fifo_total,
                )
            ),
            "missed_among_fifo_detected": (
                wilson_interval(
                    successes=missed,
                    total=fifo_total,
                )
            ),
            "regression_among_fifo_detected": (
                wilson_interval(
                    successes=(
                        slower + missed
                    ),
                    total=fifo_total,
                )
            ),
            "dropped_first_crossing": (
                wilson_interval(
                    successes=dropped,
                    total=crossings,
                )
            ),
            "next_processed_persistence": (
                wilson_interval(
                    successes=persisted,
                    total=persistence_total,
                )
            ),
        }
    return result


def combined_comparison(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    path_count = sum(
        record["comparison"][
            "fifo_detection"
        ]["total"]
        for record in records
    )
    fifo_detected = sum(
        record["comparison"][
            "fifo_detection"
        ]["successes"]
        for record in records
    )
    faster = sum(
        record["comparison"][
            "faster_among_fifo_detected"
        ]["successes"]
        for record in records
    )
    slower = sum(
        record["comparison"][
            "slower_among_fifo_detected"
        ]["successes"]
        for record in records
    )
    missed = sum(
        record["comparison"][
            "missed_among_fifo_detected"
        ]["successes"]
        for record in records
    )
    equal = (
        fifo_detected
        - faster
        - slower
        - missed
    )
    return {
        "path_comparison_count": path_count,
        "fifo_detected_count": fifo_detected,
        "faster_among_fifo_detected": (
            wilson_interval(
                successes=faster,
                total=fifo_detected,
            )
        ),
        "equal_among_fifo_detected": (
            wilson_interval(
                successes=equal,
                total=fifo_detected,
            )
        ),
        "slower_among_fifo_detected": (
            wilson_interval(
                successes=slower,
                total=fifo_detected,
            )
        ),
        "missed_among_fifo_detected": (
            wilson_interval(
                successes=missed,
                total=fifo_detected,
            )
        ),
        "regression_among_fifo_detected": (
            wilson_interval(
                successes=(
                    slower + missed
                ),
                total=fifo_detected,
            )
        ),
    }


def maximum_regression_record(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    record = max(
        records,
        key=lambda value: (
            value["comparison"][
                "regression_among_fifo_detected"
            ]["fraction"]
            or 0.0,
            value["profile_id"],
            value["schedule"],
            value["worker_policy"],
            value["service_cost"],
        ),
    )
    return {
        "profile_id": record["profile_id"],
        "schedule": record["schedule"],
        "worker_policy": (
            record["worker_policy"]
        ),
        "service_cost": (
            record["service_cost"]
        ),
        "regression_among_fifo_detected": (
            record["comparison"][
                "regression_among_fifo_detected"
            ]
        ),
        "faster_among_fifo_detected": (
            record["comparison"][
                "faster_among_fifo_detected"
            ]
        ),
        "positive_delay_requests": (
            record["comparison"][
                "positive_delay_requests"
            ]
        ),
        "next_processed_persistence": (
            record["comparison"][
                "next_processed_persistence"
            ]
        ),
    }


def maximum_aggregate_regression(
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    key, value = max(
        aggregate.items(),
        key=lambda item: (
            item[1][
                "regression_among_fifo_detected"
            ]["fraction"]
            or 0.0,
            item[0],
        ),
    )
    worker, service, schedule = (
        key.split("|")
    )
    return {
        "worker_policy": worker,
        "service_cost": int(service),
        "schedule": schedule,
        "profile_count": (
            value["profile_count"]
        ),
        "regression_among_fifo_detected": (
            value[
                "regression_among_fifo_detected"
            ]
        ),
        "faster_among_fifo_detected": (
            value[
                "faster_among_fifo_detected"
            ]
        ),
        "dropped_first_crossing": (
            value[
                "dropped_first_crossing"
            ]
        ),
        "next_processed_persistence": (
            value[
                "next_processed_persistence"
            ]
        ),
    }


def metric_comparisons(
    *,
    records: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_by_profile = {
        profile["profile_id"]: (
            profile["breach_metric"]
        )
        for profile in profiles
    }
    grouped = defaultdict(list)
    for record in records:
        grouped[
            metric_by_profile[
                record["profile_id"]
            ]
        ].append(record)
    return {
        metric: combined_comparison(
            values
        )
        for metric, values
        in sorted(grouped.items())
    }


def null_alert_summary(
    *,
    records: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    profile_by_id = {
        profile["profile_id"]: profile
        for profile in profiles
    }
    budget_checks = 0
    budget_violations = 0
    maximum_ratio = 0.0
    maximum_record = None
    for record in records:
        profile = profile_by_id[
            record["profile_id"]
        ]
        budget = profile["stream_alpha"]
        for policy in (
            "fifo_false_alert",
            "coalesced_false_alert",
        ):
            summary = record[policy]
            fraction = summary["fraction"] or 0.0
            ratio = fraction / budget
            budget_checks += 1
            if (
                summary["upper_95pct"]
                is not None
                and summary["upper_95pct"]
                > budget
            ):
                budget_violations += 1
            if ratio > maximum_ratio:
                maximum_ratio = ratio
                maximum_record = {
                    "profile_id": (
                        record["profile_id"]
                    ),
                    "schedule": (
                        record["schedule"]
                    ),
                    "worker_policy": (
                        record["worker_policy"]
                    ),
                    "service_cost": (
                        record["service_cost"]
                    ),
                    "policy": policy,
                    "stream_alpha": budget,
                    "false_alert": summary,
                    "fraction_of_stream_alpha": (
                        ratio
                    ),
                }
    return {
        "budget_check_count": (
            budget_checks
        ),
        "upper_95pct_budget_violation_count": (
            budget_violations
        ),
        "maximum_fraction_of_stream_alpha": (
            maximum_ratio
        ),
        "maximum_record": maximum_record,
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
    final_look_checks = 0
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
                coalesced, processed = (
                    coalesced_completion_times(
                        looks=looks,
                        service_cost=service_cost,
                        workers=workers,
                    )
                )
                if (
                    coalesced[-1] < 0
                    or processed
                    != int(
                        (
                            coalesced >= 0
                        ).sum()
                    )
                ):
                    raise RuntimeError(
                        "Coalesced queue accounting "
                        "failed"
                    )
                final_look_checks += 1
                queue_profiles[
                    (
                        schedule,
                        service_cost,
                        worker_policy,
                    )
                ] = {
                    "fifo": fifo,
                    "coalesced": coalesced,
                    "processed": processed,
                }
    certification_records = []
    drift_records = []
    drift_null_records = []
    subset_checks = 0
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
                        queue = queue_profiles[
                            (
                                schedule,
                                service_cost,
                                worker_policy,
                            )
                        ]
                        (
                            comparison,
                            _,
                        ) = action_comparison(
                            crossings=crossings,
                            fifo_completion=queue[
                                "fifo"
                            ],
                            coalesced_completion=(
                                queue[
                                    "coalesced"
                                ]
                            ),
                        )
                        record = {
                            "profile_id": (
                                profile[
                                    "profile_id"
                                ]
                            ),
                            "schedule": schedule,
                            "worker_policy": (
                                worker_policy
                            ),
                            "workers": workers,
                            "service_cost": (
                                service_cost
                            ),
                            "look_count": len(
                                looks
                            ),
                            "processed_look_count": (
                                queue["processed"]
                            ),
                            "processed_look_fraction": (
                                queue["processed"]
                                / len(looks)
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
                        queue = queue_profiles[
                            (
                                schedule,
                                service_cost,
                                worker_policy,
                            )
                        ]
                        fifo_times, fifo_alert = (
                            action_times(
                                crossings=(
                                    null_crossings
                                ),
                                completion=queue[
                                    "fifo"
                                ],
                            )
                        )
                        (
                            coalesced_times,
                            coalesced_alert,
                        ) = action_times(
                            crossings=(
                                null_crossings
                            ),
                            completion=queue[
                                "coalesced"
                            ],
                        )
                        del fifo_times
                        del coalesced_times
                        if np.any(
                            coalesced_alert
                            & ~fifo_alert
                        ):
                            raise RuntimeError(
                                "Coalesced null alert "
                                "was not a FIFO subset"
                            )
                        subset_checks += 1
                        drift_null_records.append(
                            {
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
                                            coalesced_alert.sum()
                                        ),
                                        total=args.paths,
                                    )
                                ),
                            }
                        )
    certification_aggregate = (
        aggregate_records(
            certification_records
        )
    )
    drift_aggregate = aggregate_records(
        drift_records
    )
    null_summary = null_alert_summary(
        records=drift_null_records,
        profiles=drift_profiles,
    )
    result = {
        "suite": (
            "pacing_coalescing_stochastic"
        ),
        "model": (
            "Replay realized Bernoulli "
            "sufficient-statistic paths at the "
            "original recursive E70 checkpoints. "
            "Certification jointly samples state "
            "errors, conditional mistake recall, "
            "and conditional false alarms. Drift "
            "samples random eligible labels and "
            "post-change outcomes. FIFO evaluates "
            "every look; latest-pending coalescing "
            "evaluates only its deterministic "
            "outcome-independent subset."
        ),
        "source": {
            "anytime_monitoring": (
                "results/raw/"
                "pacing_anytime_monitoring/"
                "pacing_anytime_monitoring_full.json"
            ),
            "kl_monitoring": (
                "results/raw/"
                "pacing_kl_monitoring/"
                "pacing_kl_monitoring_full.json"
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
            "checkpoint_grids": (
                "original recursive E70 grids"
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
            "service_costs": (
                args.service_costs
            ),
            "worker_policies": WORKERS,
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
        "aggregate": {
            "certification": (
                certification_aggregate
            ),
            "drift": drift_aggregate,
        },
        "headline": {
            "certification": {
                "combined": (
                    combined_comparison(
                        certification_records
                    )
                ),
                "maximum_aggregate_regression": (
                    maximum_aggregate_regression(
                        certification_aggregate
                    )
                ),
                "maximum_profile_regression": (
                    maximum_regression_record(
                        certification_records
                    )
                ),
            },
            "drift": {
                "combined": (
                    combined_comparison(
                        drift_records
                    )
                ),
                "by_breach_metric": (
                    metric_comparisons(
                        records=drift_records,
                        profiles=drift_profiles,
                    )
                ),
                "maximum_aggregate_regression": (
                    maximum_aggregate_regression(
                        drift_aggregate
                    )
                ),
                "maximum_profile_regression": (
                    maximum_regression_record(
                        drift_records
                    )
                ),
            },
            "drift_null": null_summary,
        },
        "validation": {
            "certification_profile_count": (
                len(certification_profiles)
            ),
            "expected_certification_profile_count": (
                24
            ),
            "drift_profile_count": (
                len(drift_profiles)
            ),
            "expected_drift_profile_count": 18,
            "direct_kl_inversion_equivalence_checks": (
                equivalence_checks
            ),
            "queue_final_look_and_accounting_checks": (
                final_look_checks
            ),
            "null_alert_subset_checks": (
                subset_checks
            ),
            "null_alert_budget_checks": (
                null_summary[
                    "budget_check_count"
                ]
            ),
            "null_alert_upper_95pct_budget_violation_count": (
                null_summary[
                    "upper_95pct_budget_violation_count"
                ]
            ),
            "outcome_independent_selection": (
                "Queue selection depends only on "
                "checkpoint arrivals, service "
                "cost, and worker availability."
            ),
        },
        "limitations": [
            "The audit uses stratified E68 parameter profiles rather than every duplicated policy record.",
            "Paths use independent Bernoulli request outcomes and the original recursive E70 checkpoint grids; traffic correlation and E75 phase/rounding alternatives remain unmodeled.",
            "Certification simulates one aggregate stream with dimension count represented through alpha allocation, matching E68-E70 rather than modeling heterogeneous per-dimension rates.",
            "Service cost is fixed, running jobs are nonpreemptive, and the pending queue holds only the latest look.",
            "Monte Carlo intervals quantify simulation uncertainty, not uncertainty in the underlying measured-service model.",
        ],
    }
    if (
        result["validation"][
            "certification_profile_count"
        ]
        != result["validation"][
            "expected_certification_profile_count"
        ]
        or result["validation"][
            "drift_profile_count"
        ]
        != result["validation"][
            "expected_drift_profile_count"
        ]
    ):
        raise RuntimeError(
            "Stratified profile count mismatch"
        )
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
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
