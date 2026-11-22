from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simulate_adaptive_prefill_scheduling import (
    decision_metrics,
    make_selector,
)
from simulate_causal_service_fairness import (
    CAUSAL_HALF_LIFE_MS,
    PRIOR_TOTAL_SERVICE_MS,
    causal_service_metrics,
    fitted_decode_request_costs,
    jain_index,
    make_service_ledger,
    target_weights,
)
from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    MAX_BATCH,
    aggregate_metrics,
    load_service_models,
    make_arrivals,
    make_workload,
)
from simulate_decode_continuous_batching import (
    simulate as simulate_decode_only,
)
from simulate_prefill_admission_safety import (
    NOMINAL_BACKLOG_LIMIT_MS,
    safety_admission_metrics,
)
from simulate_prefill_decode_scheduling import (
    PREFIX_MODES,
    load_prefill_service_model,
    simulate,
)
from simulate_prefill_stall_admission import (
    hard_selector_config,
    make_admission_selector,
)
from simulate_tenant_reservation_fairness import (
    TENANTS,
    TENANT_BY_ID,
    estimate_tenant_capacity,
    make_tenant_workload,
    standalone_decode_lookup,
    tenant_metrics,
    tenant_specifications,
)
from simulate_weighted_fair_admission import (
    BASE_LIMIT_MS,
    BASE_SAFETY_FACTOR,
    MAX_CREDIT,
    MIN_CREDIT,
)


ARRIVAL_MODES = {
    "poisson": {
        "kind": "poisson",
        "burst_size": 1,
    },
    "burst_4": {
        "kind": "fixed_burst",
        "burst_size": 4,
    },
    "burst_16": {
        "kind": "fixed_burst",
        "burst_size": 16,
    },
}

POLICIES = {
    "tenant_equal_actual_service40_gain16": {
        "reservation": "equal",
        "commitment": "actual_only",
        "gain": 16.0,
    },
    "tenant_equal_prefill_commit40_gain16": {
        "reservation": "equal",
        "commitment": "prefill",
        "gain": 16.0,
    },
    "tenant_equal_batch32_commit40_gain16": {
        "reservation": "equal",
        "commitment": "prefill_decode",
        "decode_batch": 32,
        "gain": 16.0,
    },
    "tenant_equal_batch1_commit40_gain16": {
        "reservation": "equal",
        "commitment": "prefill_decode",
        "decode_batch": 1,
        "gain": 16.0,
    },
    "tenant_reserved_actual_service40_gain16": {
        "reservation": "weighted",
        "commitment": "actual_only",
        "gain": 16.0,
    },
    "tenant_reserved_prefill_commit40_gain16": {
        "reservation": "weighted",
        "commitment": "prefill",
        "gain": 16.0,
    },
    "tenant_reserved_batch32_commit40_gain16": {
        "reservation": "weighted",
        "commitment": "prefill_decode",
        "decode_batch": 32,
        "gain": 16.0,
    },
    "tenant_reserved_batch1_commit40_gain16": {
        "reservation": "weighted",
        "commitment": "prefill_decode",
        "decode_batch": 1,
        "gain": 16.0,
    },
}

ACTUAL_POLICY_REFERENCE = {
    "tenant_equal_actual_service40_gain16": (
        "tenant_equal_causal_service40_gain16"
    ),
    "tenant_reserved_actual_service40_gain16": (
        "tenant_reserved_causal_service40_gain16"
    ),
}


def make_arrival_trace(
    *,
    mode: str,
    num_requests: int,
    requests_per_second: float,
    seed: int,
) -> np.ndarray:
    configuration = ARRIVAL_MODES[mode]
    if configuration["kind"] == "poisson":
        return make_arrivals(
            num_requests=num_requests,
            requests_per_second=(
                requests_per_second
            ),
            seed=seed,
        )
    burst_size = int(
        configuration["burst_size"]
    )
    interval_ms = (
        burst_size
        * 1_000.0
        / requests_per_second
    )
    return (
        np.arange(
            num_requests,
            dtype=np.int64,
        )
        // burst_size
    ).astype(np.float64) * interval_ms


def fixed_batch_decode_lookup(
    *,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prompt_values: np.ndarray,
    output_values: np.ndarray,
    batch_size: int,
) -> dict[tuple[int, int], float]:
    lookup = {}
    for prompt in sorted(
        {
            int(value)
            for value in prompt_values
        }
    ):
        for output in sorted(
            {
                int(value)
                for value in output_values
            }
        ):
            total_ms = sum(
                float(
                    fitted_decode_request_costs(
                        contexts=np.asarray(
                            [
                                prompt
                                + generated
                            ],
                            dtype=np.float64,
                        ),
                        decode_model=decode_model,
                        decode_service_models=(
                            decode_service_models
                        ),
                        allocation_batch_size=(
                            batch_size
                        ),
                    )[0]
                )
                for generated in range(output)
            )
            if total_ms <= 0.0:
                raise RuntimeError(
                    "Invalid fixed-batch decode "
                    "commitment"
                )
            lookup[(prompt, output)] = total_ms
    return lookup


def make_commitment_ledger(
    *,
    tenant_ids: np.ndarray,
    uncached_tokens: np.ndarray,
    output_tokens: np.ndarray,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    target: np.ndarray,
    commitment_mode: str,
    decode_batch: int | None,
) -> tuple[
    Callable[[dict[str, Any]], None],
    Callable[[dict[str, Any]], None],
    Callable[[int, float, float], None],
    dict[str, Any],
]:
    (
        base_prefill_observer,
        base_decode_observer,
        ledger,
    ) = make_service_ledger(
        tenant_ids=tenant_ids,
        decode_model=decode_model,
        decode_service_models=(
            decode_service_models
        ),
        target=target,
    )
    num_requests = int(
        tenant_ids.size
    )
    pending_prefill = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    pending_decode = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    predicted_prefill = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    predicted_decode = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    decode_steps = np.zeros(
        num_requests,
        dtype=np.int64,
    )
    prefill_tokens = np.zeros(
        num_requests,
        dtype=np.int64,
    )
    ledger.update(
        {
            "pending_service_ms": np.zeros(
                len(TENANTS),
                dtype=np.float64,
            ),
            "pending_prefill_by_request": (
                pending_prefill
            ),
            "pending_decode_by_request": (
                pending_decode
            ),
            "predicted_prefill_by_request": (
                predicted_prefill
            ),
            "predicted_decode_by_request": (
                predicted_decode
            ),
            "decode_commitment_steps": (
                decode_steps
            ),
            "prefill_commitment_tokens": (
                prefill_tokens
            ),
            "commitment_count": 0,
            "maximum_commitment_underflow_ms": (
                0.0
            ),
            "prefill_reconciliation_ms": (
                0.0
            ),
            "decode_reconciliation_ms": 0.0,
        }
    )

    def consume(
        request: int,
        amount_ms: float,
        pending: np.ndarray,
    ) -> None:
        remaining = float(
            pending[request]
        )
        underflow = max(
            0.0,
            amount_ms - remaining,
        )
        ledger[
            "maximum_commitment_underflow_ms"
        ] = max(
            ledger[
                "maximum_commitment_underflow_ms"
            ],
            underflow,
        )
        consumed = min(
            amount_ms,
            remaining,
        )
        pending[request] = max(
            0.0,
            remaining - consumed,
        )
        tenant_id = int(
            tenant_ids[request]
        )
        ledger[
            "pending_service_ms"
        ][tenant_id] -= consumed
        if (
            ledger["pending_service_ms"][
                tenant_id
            ]
            < -1e-8
        ):
            raise RuntimeError(
                "Tenant commitment underflow"
            )
        ledger[
            "pending_service_ms"
        ][tenant_id] = max(
            0.0,
            ledger["pending_service_ms"][
                tenant_id
            ],
        )

    def reconcile(
        request: int,
        pending: np.ndarray,
        *,
        key: str,
    ) -> None:
        residual = float(
            pending[request]
        )
        if residual <= 0.0:
            return
        tenant_id = int(
            tenant_ids[request]
        )
        pending[request] = 0.0
        ledger[
            "pending_service_ms"
        ][tenant_id] -= residual
        if (
            ledger["pending_service_ms"][
                tenant_id
            ]
            < -1e-8
        ):
            raise RuntimeError(
                "Commitment reconciliation "
                "underflow"
            )
        ledger[
            "pending_service_ms"
        ][tenant_id] = max(
            0.0,
            ledger["pending_service_ms"][
                tenant_id
            ],
        )
        ledger[key] += residual

    def commit_request(
        request: int,
        prefill_ms: float,
        decode_ms: float,
    ) -> None:
        if commitment_mode == "actual_only":
            if (
                prefill_ms != 0.0
                or decode_ms != 0.0
            ):
                raise RuntimeError(
                    "Actual-only policy received "
                    "a commitment"
                )
            return
        if (
            pending_prefill[request] != 0.0
            or pending_decode[request] != 0.0
        ):
            raise RuntimeError(
                "Duplicate request commitment"
            )
        if prefill_ms <= 0.0:
            raise RuntimeError(
                "Missing prefill commitment"
            )
        if (
            commitment_mode
            == "prefill_decode"
            and decode_ms <= 0.0
        ):
            raise RuntimeError(
                "Missing decode commitment"
            )
        pending_prefill[request] = (
            prefill_ms
        )
        predicted_prefill[request] = (
            prefill_ms
        )
        pending_decode[request] = (
            decode_ms
        )
        predicted_decode[request] = (
            decode_ms
        )
        tenant_id = int(
            tenant_ids[request]
        )
        ledger[
            "pending_service_ms"
        ][tenant_id] += (
            prefill_ms + decode_ms
        )
        ledger["commitment_count"] += 1

    def observe_prefill(
        event: dict[str, Any],
    ) -> None:
        base_prefill_observer(event)
        if commitment_mode == "actual_only":
            return
        consume(
            int(event["request"]),
            float(event["service_ms"]),
            pending_prefill,
        )
        request = int(event["request"])
        prefill_tokens[request] += int(
            event["query_tokens"]
        )
        if (
            prefill_tokens[request]
            > uncached_tokens[request]
        ):
            raise RuntimeError(
                "Too many prefill commitment "
                "tokens"
            )
        if (
            prefill_tokens[request]
            == uncached_tokens[request]
        ):
            reconcile(
                request,
                pending_prefill,
                key=(
                    "prefill_reconciliation_ms"
                ),
            )

    def observe_decode(
        event: dict[str, Any],
    ) -> None:
        base_decode_observer(event)
        if commitment_mode != (
            "prefill_decode"
        ):
            return
        if decode_batch is None:
            raise RuntimeError(
                "Decode commitment lost its "
                "batch assumption"
            )
        requests = np.asarray(
            event["active_requests"],
            dtype=np.int64,
        )
        contexts = np.asarray(
            event["contexts"],
            dtype=np.float64,
        )
        predicted = (
            fitted_decode_request_costs(
                contexts=contexts,
                decode_model=decode_model,
                decode_service_models=(
                    decode_service_models
                ),
                allocation_batch_size=(
                    decode_batch
                ),
            )
        )
        for request, amount_ms in zip(
            requests,
            predicted,
        ):
            member = int(request)
            consume(
                member,
                float(amount_ms),
                pending_decode,
            )
            decode_steps[member] += 1
            if (
                decode_steps[member]
                > output_tokens[member]
            ):
                raise RuntimeError(
                    "Too many decode commitment "
                    "steps"
                )
            if (
                decode_steps[member]
                == output_tokens[member]
            ):
                reconcile(
                    member,
                    pending_decode,
                    key=(
                        "decode_reconciliation_ms"
                    ),
                )

    return (
        observe_prefill,
        observe_decode,
        commit_request,
        ledger,
    )


def make_commitment_selector(
    *,
    tenant_ids: np.ndarray,
    standalone_lookup: dict[
        tuple[int, int],
        float,
    ],
    commitment_lookup: dict[
        tuple[int, int],
        float,
    ]
    | None,
    policy_name: str,
    ledger: dict[str, Any],
    commit_request: Callable[
        [int, float, float],
        None,
    ],
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[dict[str, float]],
]:
    configuration = POLICIES[policy_name]
    target = target_weights(
        configuration["reservation"]
    )
    gain = float(
        configuration["gain"]
    )
    decisions: list[dict[str, float]] = []

    def select(state: dict[str, Any]) -> bool:
        request = int(state["request"])
        tenant_id = int(
            tenant_ids[request]
        )
        prompt = int(
            state["prompt_tokens"][request]
        )
        output = int(
            state["output_tokens"][request]
        )
        predictor, predictor_decisions = (
            make_admission_selector(
                backlog_limit_ms=1e30,
                slice_budget_ms=100.0,
                prefill_model=prefill_model,
            )
        )
        predictor_allow = predictor(state)
        decision = predictor_decisions[0]
        actual = np.asarray(
            ledger["decayed_service_ms"],
            dtype=np.float64,
        )
        pending = np.asarray(
            ledger["pending_service_ms"],
            dtype=np.float64,
        )
        view = actual + pending
        actual_total = float(
            np.sum(actual)
        )
        view_total = float(np.sum(view))
        if (
            actual_total <= 0.0
            or view_total <= 0.0
        ):
            raise RuntimeError(
                "Empty commitment controller "
                "ledger"
            )
        actual_shares = (
            actual / actual_total
        )
        view_shares = view / view_total
        deficit = (
            float(target[tenant_id])
            - float(view_shares[tenant_id])
        )
        credit = min(
            MAX_CREDIT,
            max(
                MIN_CREDIT,
                math.exp(gain * deficit),
            ),
        )
        effective_limit_ms = (
            BASE_LIMIT_MS * credit
        )
        allow = (
            predictor_allow
            and decision[
                "predicted_completion_ms"
            ]
            <= effective_limit_ms
        )
        standalone_decode_ms = (
            standalone_lookup[
                (prompt, output)
            ]
        )
        commitment_prefill_ms = (
            0.0
            if configuration["commitment"]
            == "actual_only"
            else float(
                decision[
                    "candidate_prefill_ms"
                ]
            )
        )
        commitment_decode_ms = (
            0.0
            if commitment_lookup is None
            else commitment_lookup[
                (prompt, output)
            ]
        )
        record = {
            **decision,
            "request": float(request),
            "arrival_ms": float(
                state["arrival_ms"]
            ),
            "admission_now_ms": float(
                state["now_ms"]
            ),
            "allowed": float(allow),
            "tenant_id": float(
                tenant_id
            ),
            "nominal_backlog_limit_ms": (
                NOMINAL_BACKLOG_LIMIT_MS
            ),
            "safety_factor": (
                BASE_SAFETY_FACTOR
            ),
            "prompt_credit_multiplier": (
                credit
            ),
            "effective_limit_ms": (
                effective_limit_ms
            ),
            "safety_adjusted_completion_ms": (
                decision[
                    "predicted_completion_ms"
                ]
                * BASE_SAFETY_FACTOR
                / credit
            ),
            "controller_gain": gain,
            "reservation_multiplier": (
                TENANT_BY_ID[
                    tenant_id
                ][
                    "reservation_multiplier"
                ]
                if configuration[
                    "reservation"
                ]
                == "weighted"
                else 1.0
            ),
            "candidate_decode_ms": (
                standalone_decode_ms
            ),
            "candidate_total_service_ms": (
                decision[
                    "candidate_prefill_ms"
                ]
                + standalone_decode_ms
            ),
            "causal_service_target_share": (
                float(target[tenant_id])
            ),
            "causal_service_share_before": (
                float(
                    actual_shares[tenant_id]
                )
            ),
            "causal_service_deficit_before": (
                float(target[tenant_id])
                - float(
                    actual_shares[tenant_id]
                )
            ),
            "causal_service_ledger_total_ms": (
                actual_total
            ),
            "controller_view_share_before": (
                float(view_shares[tenant_id])
            ),
            "controller_view_deficit_before": (
                deficit
            ),
            "controller_view_total_ms": (
                view_total
            ),
            "pending_commitment_before_ms": (
                float(np.sum(pending))
            ),
            "commitment_prefill_ms": (
                commitment_prefill_ms
            ),
            "commitment_decode_ms": (
                commitment_decode_ms
            ),
            "commitment_total_ms": (
                commitment_prefill_ms
                + commitment_decode_ms
            ),
        }
        for member in range(len(TENANTS)):
            record[
                f"causal_service_share_tenant_{member}_before"
            ] = float(actual_shares[member])
            record[
                f"controller_view_share_tenant_{member}_before"
            ] = float(view_shares[member])
            record[
                f"causal_service_target_tenant_{member}"
            ] = float(target[member])
        decisions.append(record)
        if allow:
            commit_request(
                request,
                commitment_prefill_ms,
                commitment_decode_ms,
            )
        record[
            "pending_commitment_after_ms"
        ] = float(
            np.sum(
                ledger[
                    "pending_service_ms"
                ]
            )
        )
        return allow

    return select, decisions


def admission_batch_metrics(
    *,
    decisions: list[dict[str, float]],
    workload: dict[str, np.ndarray],
    ledger: dict[str, Any],
    reservation: str,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    selected = [
        decision
        for decision in decisions
        if analysis_start
        <= int(decision["request"])
        < analysis_end
    ]
    groups: list[
        list[dict[str, float]]
    ] = []
    for decision in selected:
        if (
            not groups
            or groups[-1][0][
                "admission_now_ms"
            ]
            != decision[
                "admission_now_ms"
            ]
        ):
            groups.append([decision])
        else:
            groups[-1].append(decision)
    sizes = [
        len(group)
        for group in groups
    ]
    transitions = 0
    stale = 0
    view_changes = []
    accepted_counts = []
    actual_target_l1 = []
    actual_normalized_jain = []
    actual_service = (
        ledger["prefill_by_request"]
        + ledger["decode_by_request"]
    )
    tenant_ids = workload["tenant_ids"]
    target = target_weights(reservation)
    for group in groups:
        for left, right in zip(
            group,
            group[1:],
        ):
            left_shares = np.asarray(
                [
                    left[
                        f"controller_view_share_tenant_{member}_before"
                    ]
                    for member
                    in range(len(TENANTS))
                ],
                dtype=np.float64,
            )
            right_shares = np.asarray(
                [
                    right[
                        f"controller_view_share_tenant_{member}_before"
                    ]
                    for member
                    in range(len(TENANTS))
                ],
                dtype=np.float64,
            )
            change = 0.5 * float(
                np.sum(
                    np.abs(
                        right_shares
                        - left_shares
                    )
                )
            )
            transitions += 1
            stale += int(change <= 1e-15)
            view_changes.append(change)
        if len(group) < 2:
            continue
        accepted_requests = np.asarray(
            [
                int(decision["request"])
                for decision in group
                if decision["allowed"] == 1.0
            ],
            dtype=np.int64,
        )
        accepted_counts.append(
            int(accepted_requests.size)
        )
        if accepted_requests.size < 1:
            continue
        values = []
        for tenant in TENANTS:
            tenant_id = int(tenant["id"])
            mask = (
                tenant_ids[
                    accepted_requests
                ]
                == tenant_id
            )
            values.append(
                float(
                    np.sum(
                        actual_service[
                            accepted_requests[
                                mask
                            ]
                        ]
                    )
                )
            )
        total = sum(values)
        if total <= 0.0:
            raise RuntimeError(
                "Accepted admission batch "
                "consumed no service"
            )
        shares = [
            value / total
            for value in values
        ]
        actual_target_l1.append(
            0.5
            * sum(
                abs(
                    share
                    - float(target[index])
                )
                for index, share
                in enumerate(shares)
            )
        )
        actual_normalized_jain.append(
            jain_index(
                [
                    share
                    / float(target[index])
                    for index, share
                    in enumerate(shares)
                ]
            )
        )
    return {
        "admission_batch_count": float(
            len(groups)
        ),
        "mean_admission_batch_size": (
            statistics.mean(sizes)
        ),
        "p95_admission_batch_size": float(
            np.percentile(sizes, 95)
        ),
        "maximum_admission_batch_size": float(
            max(sizes)
        ),
        "multi_request_admission_batch_fraction": (
            sum(size > 1 for size in sizes)
            / len(sizes)
        ),
        "within_batch_transition_count": float(
            transitions
        ),
        "within_batch_stale_transition_fraction": (
            stale / transitions
            if transitions
            else 0.0
        ),
        "within_batch_mean_view_l1_change": (
            statistics.mean(view_changes)
            if view_changes
            else 0.0
        ),
        "mean_multi_request_batch_accepted_count": (
            statistics.mean(
                accepted_counts
            )
            if accepted_counts
            else 0.0
        ),
        "mean_multi_request_batch_actual_target_l1": (
            statistics.mean(
                actual_target_l1
            )
            if actual_target_l1
            else 0.0
        ),
        "p95_multi_request_batch_actual_target_l1": (
            float(
                np.percentile(
                    actual_target_l1,
                    95,
                )
            )
            if actual_target_l1
            else 0.0
        ),
        "mean_multi_request_batch_actual_normalized_jain": (
            statistics.mean(
                actual_normalized_jain
            )
            if actual_normalized_jain
            else 1.0
        ),
        "minimum_multi_request_batch_actual_normalized_jain": (
            min(actual_normalized_jain)
            if actual_normalized_jain
            else 1.0
        ),
    }


def commitment_metrics(
    *,
    workload: dict[str, np.ndarray],
    decisions: list[dict[str, float]],
    ledger: dict[str, Any],
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    if (
        np.max(
            np.abs(
                ledger[
                    "pending_service_ms"
                ]
            )
        )
        > 1e-8
        or np.max(
            ledger[
                "pending_prefill_by_request"
            ]
        )
        > 1e-8
        or np.max(
            ledger[
                "pending_decode_by_request"
            ]
        )
        > 1e-8
    ):
        raise RuntimeError(
            "Commitment ledger did not drain: "
            f"tenant_max={float(np.max(np.abs(ledger['pending_service_ms'])))} "
            f"prefill_max={float(np.max(ledger['pending_prefill_by_request']))} "
            f"decode_max={float(np.max(ledger['pending_decode_by_request']))}"
        )
    window = np.arange(
        analysis_start,
        analysis_end,
        dtype=np.int64,
    )
    actual_prefill = ledger[
        "prefill_by_request"
    ][window]
    actual_decode = ledger[
        "decode_by_request"
    ][window]
    actual_total = (
        actual_prefill + actual_decode
    )
    accepted = actual_total > 0.0
    selected = window[accepted]
    if selected.size < 1:
        raise RuntimeError(
            "No accepted commitment requests"
        )
    predicted_prefill = ledger[
        "predicted_prefill_by_request"
    ][selected]
    predicted_decode = ledger[
        "predicted_decode_by_request"
    ][selected]
    predicted_total = (
        predicted_prefill
        + predicted_decode
    )
    selected_actual_prefill = ledger[
        "prefill_by_request"
    ][selected]
    selected_actual_decode = ledger[
        "decode_by_request"
    ][selected]
    selected_actual_total = (
        selected_actual_prefill
        + selected_actual_decode
    )
    committed = predicted_total > 0.0
    selected_decisions = [
        decision
        for decision in decisions
        if analysis_start
        <= int(decision["request"])
        < analysis_end
    ]
    pending_before = [
        decision[
            "pending_commitment_before_ms"
        ]
        for decision in selected_decisions
    ]
    view_l1 = []
    for decision in selected_decisions:
        actual_shares = np.asarray(
            [
                decision[
                    f"causal_service_share_tenant_{member}_before"
                ]
                for member
                in range(len(TENANTS))
            ],
            dtype=np.float64,
        )
        view_shares = np.asarray(
            [
                decision[
                    f"controller_view_share_tenant_{member}_before"
                ]
                for member
                in range(len(TENANTS))
            ],
            dtype=np.float64,
        )
        view_l1.append(
            0.5
            * float(
                np.sum(
                    np.abs(
                        view_shares
                        - actual_shares
                    )
                )
            )
        )
    metrics = {
        "commitment_count": float(
            ledger["commitment_count"]
        ),
        "maximum_commitment_underflow_ms": float(
            ledger[
                "maximum_commitment_underflow_ms"
            ]
        ),
        "prefill_commitment_reconciliation_ms": float(
            ledger[
                "prefill_reconciliation_ms"
            ]
        ),
        "decode_commitment_reconciliation_ms": float(
            ledger[
                "decode_reconciliation_ms"
            ]
        ),
        "committed_accepted_fraction": (
            float(np.mean(committed))
        ),
        "commitment_predicted_prefill_ms": (
            float(np.sum(predicted_prefill))
        ),
        "commitment_predicted_decode_ms": (
            float(np.sum(predicted_decode))
        ),
        "commitment_predicted_total_ms": (
            float(np.sum(predicted_total))
        ),
        "commitment_actual_prefill_ms": (
            float(
                np.sum(
                    selected_actual_prefill
                )
            )
        ),
        "commitment_actual_decode_ms": (
            float(
                np.sum(
                    selected_actual_decode
                )
            )
        ),
        "commitment_actual_total_ms": (
            float(
                np.sum(
                    selected_actual_total
                )
            )
        ),
        "commitment_prefill_over_actual_ratio": (
            float(
                np.sum(predicted_prefill)
                / np.sum(
                    selected_actual_prefill
                )
            )
            if np.any(committed)
            else 0.0
        ),
        "commitment_decode_over_actual_ratio": (
            float(
                np.sum(predicted_decode)
                / np.sum(
                    selected_actual_decode
                )
            )
            if np.sum(predicted_decode) > 0.0
            else 0.0
        ),
        "commitment_total_over_actual_ratio": (
            float(
                np.sum(predicted_total)
                / np.sum(
                    selected_actual_total
                )
            )
            if np.any(committed)
            else 0.0
        ),
        "mean_pending_commitment_before_ms": (
            statistics.mean(
                pending_before
            )
        ),
        "p95_pending_commitment_before_ms": (
            float(
                np.percentile(
                    pending_before,
                    95,
                )
            )
        ),
        "maximum_pending_commitment_before_ms": (
            max(pending_before)
        ),
        "mean_actual_vs_controller_view_l1": (
            statistics.mean(view_l1)
        ),
    }
    if np.any(committed):
        ratios = (
            predicted_total[committed]
            / selected_actual_total[
                committed
            ]
        )
        metrics[
            "mean_request_commitment_over_actual_ratio"
        ] = float(np.mean(ratios))
        metrics[
            "p95_request_commitment_over_actual_ratio"
        ] = float(
            np.percentile(ratios, 95)
        )
    else:
        metrics[
            "mean_request_commitment_over_actual_ratio"
        ] = 0.0
        metrics[
            "p95_request_commitment_over_actual_ratio"
        ] = 0.0
    tenant_ids = workload["tenant_ids"]
    predicted_values = []
    actual_values = []
    for tenant in TENANTS:
        tenant_id = int(tenant["id"])
        name = tenant["name"]
        mask = (
            tenant_ids[selected] == tenant_id
        )
        predicted_value = float(
            np.sum(predicted_total[mask])
        )
        actual_value = float(
            np.sum(
                selected_actual_total[mask]
            )
        )
        predicted_values.append(
            predicted_value
        )
        actual_values.append(actual_value)
        metrics[
            f"tenant_{name}_commitment_predicted_ms"
        ] = predicted_value
        metrics[
            f"tenant_{name}_commitment_actual_ms"
        ] = actual_value
    predicted_sum = sum(predicted_values)
    actual_sum = sum(actual_values)
    predicted_shares = [
        (
            value / predicted_sum
            if predicted_sum > 0.0
            else 0.0
        )
        for value in predicted_values
    ]
    actual_shares = [
        value / actual_sum
        for value in actual_values
    ]
    metrics[
        "commitment_tenant_share_l1_error"
    ] = (
        0.5
        * sum(
            abs(left - right)
            for left, right in zip(
                predicted_shares,
                actual_shares,
            )
        )
        if predicted_sum > 0.0
        else 0.0
    )
    for index, tenant in enumerate(
        TENANTS
    ):
        name = tenant["name"]
        metrics[
            f"tenant_{name}_commitment_predicted_share"
        ] = predicted_shares[index]
        metrics[
            f"tenant_{name}_commitment_actual_share"
        ] = actual_shares[index]
    return metrics


def simulate_policy(
    *,
    arrivals: np.ndarray,
    latency_origin_arrivals: (
        np.ndarray | None
    ) = None,
    workload: dict[str, np.ndarray],
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    policy_name: str,
    kv_budget_tokens: int,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    configuration = POLICIES[policy_name]
    reservation = str(
        configuration["reservation"]
    )
    target = target_weights(reservation)
    standalone_lookup = (
        standalone_decode_lookup(
            decode_model=decode_model,
            decode_service_models=(
                decode_service_models
            ),
        )
    )
    commitment_lookup = None
    decode_batch = configuration.get(
        "decode_batch"
    )
    if decode_batch is not None:
        commitment_lookup = (
            fixed_batch_decode_lookup(
                decode_model=decode_model,
                decode_service_models=(
                    decode_service_models
                ),
                prompt_values=workload[
                    "prompt_tokens"
                ],
                output_values=workload[
                    "output_tokens"
                ],
                batch_size=int(decode_batch),
            )
        )
    (
        prefill_observer,
        decode_observer,
        commit_request,
        ledger,
    ) = make_commitment_ledger(
        tenant_ids=workload["tenant_ids"],
        uncached_tokens=workload[
            "uncached_tokens"
        ],
        output_tokens=workload[
            "output_tokens"
        ],
        decode_model=decode_model,
        decode_service_models=(
            decode_service_models
        ),
        target=target,
        commitment_mode=str(
            configuration["commitment"]
        ),
        decode_batch=(
            int(decode_batch)
            if decode_batch is not None
            else None
        ),
    )
    (
        admission_selector,
        admission_decisions,
    ) = make_commitment_selector(
        tenant_ids=workload["tenant_ids"],
        standalone_lookup=standalone_lookup,
        commitment_lookup=commitment_lookup,
        policy_name=policy_name,
        ledger=ledger,
        commit_request=commit_request,
        prefill_model=prefill_model,
    )
    chunk_selector, slice_decisions = (
        make_selector(
            policy=hard_selector_config(
                100.0
            ),
            prefill_model=prefill_model,
        )
    )
    metrics = simulate(
        arrivals=arrivals,
        workload=workload,
        decode_model=decode_model,
        decode_service_models=(
            decode_service_models
        ),
        prefill_model=prefill_model,
        policy_name=(
            "chunk_1024_prefill_first"
        ),
        kv_budget_tokens=kv_budget_tokens,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        chunk_selector=chunk_selector,
        admission_selector=(
            admission_selector
        ),
        decode_service_observer=(
            decode_observer
        ),
        prefill_service_observer=(
            prefill_observer
        ),
        latency_origin_arrivals=(
            latency_origin_arrivals
        ),
    )
    metrics.update(
        decision_metrics(slice_decisions)
    )
    metrics.update(
        safety_admission_metrics(
            admission_decisions,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    metrics.update(
        tenant_metrics(
            admission_decisions,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    metrics.update(
        causal_service_metrics(
            workload=workload,
            decisions=admission_decisions,
            ledger=ledger,
            reservation=reservation,
            base_metrics=metrics,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    metrics.update(
        commitment_metrics(
            workload=workload,
            decisions=admission_decisions,
            ledger=ledger,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    metrics.update(
        admission_batch_metrics(
            decisions=admission_decisions,
            workload=workload,
            ledger=ledger,
            reservation=reservation,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )
    return metrics


def e59_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "causal_service_fairness"
        / "causal_service_fairness_full.json"
    )
    result = json.loads(
        path.read_text(encoding="utf-8")
    )
    return {
        (
            record["prefix_mode"],
            record[
                "offered_load_vs_decode_capacity"
            ],
            record["policy"],
        ): record
        for record in result["records"]
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    decode_service_models = load_service_models(
        args.root
    )
    prefill_model = load_prefill_service_model(
        args.root
    )
    kv_budget_tokens = int(
        args.kv_budget_gib
        * (1024**3)
        / KV_BYTES_PER_TOKEN
    )
    offered_loads = [
        float(value)
        for value in args.offered_loads.split(",")
    ]
    arrival_modes = [
        value.strip()
        for value in args.arrival_modes.split(",")
        if value.strip()
    ]
    unknown_modes = set(
        arrival_modes
    ) - set(ARRIVAL_MODES)
    if unknown_modes:
        raise ValueError(
            f"Unknown arrival modes: "
            f"{sorted(unknown_modes)}"
        )
    seeds = [
        args.seed + index
        for index in range(args.seeds)
    ]
    decode_capacities = {}
    tenant_capacities: dict[
        str,
        dict[str, Any],
    ] = {}
    for prefix_mode in PREFIX_MODES:
        tenant_capacities[
            prefix_mode
        ] = {}
        for replicate_index, decode_model in enumerate(
            decode_service_models[
                "attention_replicates"
            ]
        ):
            source = decode_model["source"]
            decode_workload = make_workload(
                name="mixed_chat",
                num_requests=(
                    args.capacity_requests
                ),
                seed=(
                    args.seed
                    + 500_000
                    + replicate_index * 10_000
                    + 1_000
                ),
            )
            decode_metrics = simulate_decode_only(
                arrivals=np.zeros(
                    args.capacity_requests,
                    dtype=np.float64,
                ),
                workload=decode_workload,
                model=decode_model,
                service_models=(
                    decode_service_models
                ),
                policy_name="ragged_fcfs",
                kv_budget_tokens=(
                    kv_budget_tokens
                ),
                ttft_target_ms=250.0,
                analysis_start=0,
                analysis_end=(
                    args.capacity_requests
                ),
            )
            decode_capacities[source] = (
                args.capacity_requests
                * 1_000.0
                / decode_metrics["busy_ms"]
            )
            tenant_capacities[
                prefix_mode
            ][source] = (
                estimate_tenant_capacity(
                    num_requests=(
                        args.capacity_requests
                    ),
                    seed=(
                        args.seed
                        + 600_000
                        + replicate_index
                        * 10_000
                        + (
                            0
                            if prefix_mode == "none"
                            else 1_000
                        )
                    ),
                    prefix_mode=prefix_mode,
                    decode_model=decode_model,
                    decode_service_models=(
                        decode_service_models
                    ),
                    prefill_model=prefill_model,
                    kv_budget_tokens=(
                        kv_budget_tokens
                    ),
                )
            )

    records = []
    scenario_count = 0
    composition_reference = {}
    for prefix_mode in PREFIX_MODES:
        for offered_load in offered_loads:
            for arrival_mode in arrival_modes:
                for policy_name in POLICIES:
                    scenarios = []
                    for (
                        replicate_index,
                        decode_model,
                    ) in enumerate(
                        decode_service_models[
                            "attention_replicates"
                        ]
                    ):
                        source = decode_model[
                            "source"
                        ]
                        request_rate = (
                            decode_capacities[
                                source
                            ]
                            * offered_load
                        )
                        for seed in seeds:
                            workload_seed = (
                                seed
                                + replicate_index
                                * 100_000
                                + (
                                    0
                                    if prefix_mode
                                    == "none"
                                    else 10_000
                                )
                            )
                            (
                                workload,
                                composition,
                            ) = (
                                make_tenant_workload(
                                    num_requests=(
                                        args.num_requests
                                    ),
                                    seed=workload_seed,
                                    prefix_mode=(
                                        prefix_mode
                                    ),
                                )
                            )
                            if (
                                prefix_mode
                                not in composition_reference
                            ):
                                composition_reference[
                                    prefix_mode
                                ] = composition
                            elif (
                                composition_reference[
                                    prefix_mode
                                ]
                                != composition
                            ):
                                raise RuntimeError(
                                    "Tenant composition "
                                    "drifted"
                                )
                            arrivals = (
                                make_arrival_trace(
                                    mode=arrival_mode,
                                    num_requests=(
                                        args.num_requests
                                    ),
                                    requests_per_second=(
                                        request_rate
                                    ),
                                    seed=(
                                        workload_seed
                                        + 1
                                    ),
                                )
                            )
                            metrics = simulate_policy(
                                arrivals=arrivals,
                                workload=workload,
                                decode_model=(
                                    decode_model
                                ),
                                decode_service_models=(
                                    decode_service_models
                                ),
                                prefill_model=(
                                    prefill_model
                                ),
                                policy_name=(
                                    policy_name
                                ),
                                kv_budget_tokens=(
                                    kv_budget_tokens
                                ),
                                analysis_start=(
                                    args.warmup_requests
                                ),
                                analysis_end=(
                                    args.num_requests
                                    - args.tail_guard_requests
                                ),
                            )
                            if (
                                metrics[
                                    "slice_budget_"
                                    "violation_fraction"
                                ]
                                != 0.0
                            ):
                                raise RuntimeError(
                                    "Hard slice budget "
                                    "failed"
                                )
                            if (
                                metrics[
                                    "p99_accepted_"
                                    "safety_adjusted_"
                                    "completion_ms"
                                ]
                                > (
                                    NOMINAL_BACKLOG_LIMIT_MS
                                    + 1e-9
                                )
                            ):
                                raise RuntimeError(
                                    "Admission threshold "
                                    "failed"
                                )
                            scenarios.append(
                                {
                                    "service_replicate": (
                                        source
                                    ),
                                    "seed": seed,
                                    "arrival_mode": (
                                        arrival_mode
                                    ),
                                    "request_rate_per_second": (
                                        request_rate
                                    ),
                                    "system_capacity_requests_s": (
                                        tenant_capacities[
                                            prefix_mode
                                        ][source][
                                            "requests_per_second"
                                        ]
                                    ),
                                    "offered_load_vs_system_capacity": (
                                        request_rate
                                        / tenant_capacities[
                                            prefix_mode
                                        ][source][
                                            "requests_per_second"
                                        ]
                                    ),
                                    **metrics,
                                }
                            )
                            scenario_count += 1
                    records.append(
                        {
                            "prefix_mode": (
                                prefix_mode
                            ),
                            "offered_load_vs_decode_capacity": (
                                offered_load
                            ),
                            "arrival_mode": (
                                arrival_mode
                            ),
                            "policy": policy_name,
                            "configuration": (
                                POLICIES[
                                    policy_name
                                ]
                            ),
                            "metrics": (
                                aggregate_metrics(
                                    scenarios
                                )
                            ),
                            "scenarios": scenarios,
                        }
                    )

    reference = e59_records(args.root)
    equivalence = []
    equivalence_enabled = (
        args.num_requests == 2_000
        and args.warmup_requests == 200
        and args.tail_guard_requests == 100
        and args.capacity_requests == 1_500
        and args.seeds == 3
        and args.seed == 40_000
        and offered_loads == [0.4]
        and "poisson" in arrival_modes
    )
    if equivalence_enabled:
        for record in records:
            if (
                record["arrival_mode"]
                != "poisson"
                or record["policy"]
                not in ACTUAL_POLICY_REFERENCE
            ):
                continue
            reference_policy = (
                ACTUAL_POLICY_REFERENCE[
                    record["policy"]
                ]
            )
            expected_record = reference[
                (
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    reference_policy,
                )
            ]
            expected_scenarios = {
                (
                    scenario[
                        "service_replicate"
                    ],
                    scenario["seed"],
                ): scenario
                for scenario in expected_record[
                    "scenarios"
                ]
            }
            for scenario in record[
                "scenarios"
            ]:
                key = (
                    scenario[
                        "service_replicate"
                    ],
                    scenario["seed"],
                )
                expected = expected_scenarios[
                    key
                ]
                numeric_metrics = [
                    metric
                    for metric, value
                    in expected.items()
                    if isinstance(
                        value,
                        (int, float),
                    )
                    and not isinstance(
                        value,
                        bool,
                    )
                ]
                maximum_difference = max(
                    abs(
                        scenario[metric]
                        - expected[metric]
                    )
                    for metric
                    in numeric_metrics
                )
                if maximum_difference > 1e-9:
                    raise RuntimeError(
                        "E59 actual-only "
                        "equivalence failed: "
                        f"{record['policy']=} "
                        f"{key=} "
                        f"{maximum_difference=}"
                    )
                equivalence.append(
                    {
                        "prefix_mode": (
                            record[
                                "prefix_mode"
                            ]
                        ),
                        "offered_load": (
                            record[
                                "offered_load_vs_decode_capacity"
                            ]
                        ),
                        "policy": (
                            record["policy"]
                        ),
                        "reference_policy": (
                            reference_policy
                        ),
                        "service_replicate": (
                            key[0]
                        ),
                        "seed": key[1],
                        "maximum_absolute_difference": (
                            maximum_difference
                        ),
                    }
                )

    result = {
        "suite": (
            "causal_service_commitment_"
            "simulation"
        ),
        "model": (
            "E59 actual-service feedback "
            "augmented with admission-time "
            "pending service commitments under "
            "Poisson and fixed-size request "
            "bursts."
        ),
        "prefill_service_model": (
            prefill_model
        ),
        "decode_service_models": (
            decode_service_models
        ),
        "tenant_specifications": (
            tenant_specifications()
        ),
        "tenant_composition": (
            composition_reference
        ),
        "controller": {
            "causal_half_life_ms": (
                CAUSAL_HALF_LIFE_MS
            ),
            "prior_total_service_ms": (
                PRIOR_TOTAL_SERVICE_MS
            ),
            "base_safety_factor": (
                BASE_SAFETY_FACTOR
            ),
            "base_limit_ms": BASE_LIMIT_MS,
            "minimum_credit": MIN_CREDIT,
            "maximum_credit": MAX_CREDIT,
            "feedback_rule": (
                "credit = clip(exp(16 * "
                "(target share - (decayed "
                "actual service + pending "
                "commitments) share)), 0.5, 4.0)"
            ),
            "pending_decay": (
                "Pending future service does not "
                "decay. It is removed as predicted "
                "prefill/decode steps execute; "
                "actual observed service enters "
                "the 40-second service-decayed "
                "ledger."
            ),
        },
        "arrival_modes": ARRIVAL_MODES,
        "policies": POLICIES,
        "simulation": {
            "num_requests": args.num_requests,
            "warmup_requests": (
                args.warmup_requests
            ),
            "tail_guard_requests": (
                args.tail_guard_requests
            ),
            "capacity_requests": (
                args.capacity_requests
            ),
            "seeds": seeds,
            "arrival_modes": arrival_modes,
            "offered_loads_vs_decode_capacity": (
                offered_loads
            ),
            "max_batch": MAX_BATCH,
            "kv_budget_gib": (
                args.kv_budget_gib
            ),
            "kv_budget_tokens": (
                kv_budget_tokens
            ),
            "kv_bytes_per_token": (
                KV_BYTES_PER_TOKEN
            ),
        },
        "decode_only_capacities": (
            decode_capacities
        ),
        "tenant_system_capacities": (
            tenant_capacities
        ),
        "validation": {
            "scenario_count": (
                scenario_count
            ),
            "e59_equivalence_enabled": (
                equivalence_enabled
            ),
            "e59_equivalence_checks": (
                equivalence
            ),
            "e59_maximum_absolute_difference": (
                max(
                    item[
                        "maximum_absolute_difference"
                    ]
                    for item in equivalence
                )
                if equivalence
                else None
            ),
            "accounting": (
                "The E59 actual-only control must "
                "reproduce exactly. All pending "
                "commitments drain to zero; "
                "prefill commitment follows the "
                "same hard-slice predictor as "
                "execution; decode commitment uses "
                "the exact fitted per-request "
                "formula at an assumed batch of "
                "32 or 1. Base token, causal-pair, "
                "batch, KV, hard-slice, observer, "
                "and admission invariants remain "
                "active."
            ),
        },
        "records": records,
        "limitations": [
            "The study is a fitted-service simulation and inherits E59's synthetic tenant, immediate-rejection, and single-replica limitations.",
            "Fixed bursts preserve mean arrival rate but are deterministic and do not model production autocorrelation or retry storms.",
            "Pending prefill is predictable from the hard-slice model, while decode commitments assume a fixed batch of 32 or the deliberately pessimistic batch-1 control.",
            "Pending service is not decayed before execution because it represents future work; other reservation semantics may be preferable for cancellations or long queues.",
            "The controller still does not reserve KV capacity across replicas, model cancellations, or price scheduler CPU and communication.",
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
        "--offered-loads",
        default="0.4",
    )
    parser.add_argument(
        "--arrival-modes",
        default=(
            "poisson,burst_4,burst_16"
        ),
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--tail-guard-requests",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--capacity-requests",
        type=int,
        default=1_500,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=40_000,
    )
    parser.add_argument(
        "--kv-budget-gib",
        type=float,
        default=32.0,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
