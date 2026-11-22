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
from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    LAYERS,
    MAX_BATCH,
    aggregate_metrics,
    load_service_models,
    make_arrivals,
    make_workload,
    mlp_ms,
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
    TENANT_NAMES,
    estimate_tenant_capacity,
    make_tenant_selector,
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


CAUSAL_HALF_LIFE_MS = 40_000.0
PRIOR_TOTAL_SERVICE_MS = 4_000.0
POLICIES = {
    "tenant_equal_e2e_service320_gain8": {
        "kind": "e57_control",
    },
    "tenant_reserved_e2e_service320": {
        "kind": "e57_control",
    },
    "tenant_equal_causal_service40_gain4": {
        "kind": "causal_service",
        "reservation": "equal",
        "gain": 4.0,
    },
    "tenant_equal_causal_service40_gain8": {
        "kind": "causal_service",
        "reservation": "equal",
        "gain": 8.0,
    },
    "tenant_equal_causal_service40_gain16": {
        "kind": "causal_service",
        "reservation": "equal",
        "gain": 16.0,
    },
    "tenant_reserved_causal_service40_gain4": {
        "kind": "causal_service",
        "reservation": "weighted",
        "gain": 4.0,
    },
    "tenant_reserved_causal_service40_gain8": {
        "kind": "causal_service",
        "reservation": "weighted",
        "gain": 8.0,
    },
    "tenant_reserved_causal_service40_gain16": {
        "kind": "causal_service",
        "reservation": "weighted",
        "gain": 16.0,
    },
}

CONTROL_POLICIES = (
    "tenant_equal_e2e_service320_gain8",
    "tenant_reserved_e2e_service320",
)


def target_weights(
    reservation: str,
) -> np.ndarray:
    if reservation == "equal":
        weights = np.ones(
            len(TENANTS),
            dtype=np.float64,
        )
    elif reservation == "weighted":
        weights = np.asarray(
            [
                tenant[
                    "reservation_multiplier"
                ]
                for tenant in TENANTS
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(reservation)
    return weights / np.sum(weights)


def fitted_decode_request_costs(
    *,
    contexts: np.ndarray,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    allocation_batch_size: int | None = None,
) -> np.ndarray:
    batch = (
        int(contexts.size)
        if allocation_batch_size is None
        else allocation_batch_size
    )
    if contexts.size < 1 or batch < 1:
        raise ValueError(
            "Decode allocation requires a "
            "positive batch"
        )
    coefficients = decode_model[
        "coefficients"
    ]
    common_per_request = LAYERS * (
        coefficients["batch_ms"]
        + (
            coefficients["intercept_ms"]
            + mlp_ms(
                decode_service_models,
                batch,
            )
        )
        / batch
    )
    return (
        common_per_request
        + LAYERS
        * coefficients["context_token_ms"]
        * contexts
    )


def make_service_ledger(
    *,
    tenant_ids: np.ndarray,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    target: np.ndarray,
) -> tuple[
    Callable[[dict[str, Any]], None],
    Callable[[dict[str, Any]], None],
    dict[str, Any],
]:
    num_requests = int(tenant_ids.size)
    prefill_by_request = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    decode_by_request = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    state: dict[str, Any] = {
        "decayed_service_ms": (
            target * PRIOR_TOTAL_SERVICE_MS
        ),
        "prefill_by_request": (
            prefill_by_request
        ),
        "decode_by_request": (
            decode_by_request
        ),
        "prefill_observed_ms": 0.0,
        "decode_observed_ms": 0.0,
        "decode_step_count": 0,
        "decode_batch_sizes": [],
        "maximum_decode_step_error_ms": 0.0,
        "event_count": 0,
        "decay_clock_ms": 0.0,
    }

    def apply_service(
        requests: np.ndarray,
        allocations: np.ndarray,
        *,
        kind: str,
    ) -> None:
        total_ms = float(
            np.sum(allocations)
        )
        if (
            total_ms <= 0.0
            or requests.size
            != allocations.size
        ):
            raise RuntimeError(
                "Invalid causal service event"
            )
        decay = 2.0 ** (
            -total_ms / CAUSAL_HALF_LIFE_MS
        )
        state[
            "decayed_service_ms"
        ] *= decay
        for request, allocation in zip(
            requests,
            allocations,
        ):
            tenant_id = int(
                tenant_ids[int(request)]
            )
            state[
                "decayed_service_ms"
            ][tenant_id] += float(
                allocation
            )
            if kind == "prefill":
                prefill_by_request[
                    int(request)
                ] += float(allocation)
            elif kind == "decode":
                decode_by_request[
                    int(request)
                ] += float(allocation)
            else:
                raise ValueError(kind)
        state["event_count"] += 1
        state["decay_clock_ms"] += total_ms

    def observe_prefill(
        event: dict[str, Any],
    ) -> None:
        request = int(event["request"])
        service_ms = float(
            event["service_ms"]
        )
        apply_service(
            np.asarray(
                [request],
                dtype=np.int64,
            ),
            np.asarray(
                [service_ms],
                dtype=np.float64,
            ),
            kind="prefill",
        )
        state[
            "prefill_observed_ms"
        ] += service_ms

    def observe_decode(
        event: dict[str, Any],
    ) -> None:
        requests = np.asarray(
            event["active_requests"],
            dtype=np.int64,
        )
        contexts = np.asarray(
            event["contexts"],
            dtype=np.float64,
        )
        service_ms = float(
            event["service_ms"]
        )
        batch = int(requests.size)
        allocations = fitted_decode_request_costs(
            contexts=contexts,
            decode_model=decode_model,
            decode_service_models=(
                decode_service_models
            ),
        )
        step_error = abs(
            float(np.sum(allocations))
            - service_ms
        )
        if step_error > 1e-9:
            raise RuntimeError(
                "Causal decode allocation "
                f"failed: {step_error=}"
            )
        apply_service(
            requests,
            allocations,
            kind="decode",
        )
        state[
            "decode_observed_ms"
        ] += service_ms
        state["decode_step_count"] += 1
        state[
            "decode_batch_sizes"
        ].append(batch)
        state[
            "maximum_decode_step_error_ms"
        ] = max(
            state[
                "maximum_decode_step_error_ms"
            ],
            step_error,
        )

    return (
        observe_prefill,
        observe_decode,
        state,
    )


def make_causal_selector(
    *,
    tenant_ids: np.ndarray,
    decode_lookup: dict[
        tuple[int, int],
        float,
    ],
    policy_name: str,
    ledger: dict[str, Any],
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
        served = np.asarray(
            ledger["decayed_service_ms"],
            dtype=np.float64,
        )
        served_total = float(
            np.sum(served)
        )
        if served_total <= 0.0:
            raise RuntimeError(
                "Empty causal service ledger"
            )
        shares = served / served_total
        deficit = (
            float(target[tenant_id])
            - float(shares[tenant_id])
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
        decode_ms = decode_lookup[
            (prompt, output)
        ]
        record = {
            **decision,
            "request": float(request),
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
                decode_ms
            ),
            "candidate_total_service_ms": (
                decision[
                    "candidate_prefill_ms"
                ]
                + decode_ms
            ),
            "causal_service_target_share": (
                float(target[tenant_id])
            ),
            "causal_service_share_before": (
                float(shares[tenant_id])
            ),
            "causal_service_deficit_before": (
                deficit
            ),
            "causal_service_ledger_total_ms": (
                served_total
            ),
        }
        for member in range(len(TENANTS)):
            record[
                f"causal_service_share_tenant_{member}_before"
            ] = float(shares[member])
            record[
                f"causal_service_target_tenant_{member}"
            ] = float(target[member])
        decisions.append(record)
        return allow

    return select, decisions


def jain_index(
    values: list[float],
) -> float:
    array = np.asarray(
        values,
        dtype=np.float64,
    )
    denominator = (
        array.size
        * float(np.sum(array * array))
    )
    return (
        float(np.sum(array) ** 2 / denominator)
        if denominator > 0.0
        else 1.0
    )


def causal_service_metrics(
    *,
    workload: dict[str, np.ndarray],
    decisions: list[dict[str, float]],
    ledger: dict[str, Any],
    reservation: str,
    base_metrics: dict[str, float],
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    prefill_all = ledger[
        "prefill_by_request"
    ]
    decode_all = ledger[
        "decode_by_request"
    ]
    if abs(
        ledger["prefill_observed_ms"]
        - base_metrics["prefill_busy_ms"]
    ) > 1e-8:
        raise RuntimeError(
            "Prefill observer did not cover "
            "busy time"
        )
    if abs(
        ledger["decode_observed_ms"]
        - base_metrics["decode_busy_ms"]
    ) > 1e-8:
        raise RuntimeError(
            "Decode observer did not cover "
            "busy time"
        )
    window = np.arange(
        analysis_start,
        analysis_end,
        dtype=np.int64,
    )
    causal_total = (
        prefill_all[window]
        + decode_all[window]
    )
    accepted_mask = causal_total > 0.0
    selected = window[accepted_mask]
    selected_total = causal_total[
        accepted_mask
    ]
    if selected.size < 1:
        raise RuntimeError(
            "No causally served requests"
        )
    tenant_ids = workload["tenant_ids"]
    target = target_weights(reservation)
    service_values = []
    shares = []
    for tenant in TENANTS:
        tenant_id = int(tenant["id"])
        name = tenant["name"]
        mask = (
            tenant_ids[selected] == tenant_id
        )
        tenant_service = float(
            np.sum(selected_total[mask])
        )
        service_values.append(
            tenant_service
        )
        metrics_key = (
            f"tenant_{name}_causal_service_ms"
        )
        # Assigned below after total is known.
        if not np.any(mask):
            raise RuntimeError(
                f"No causal tenant {name}"
            )
    total_service = sum(service_values)
    metrics: dict[str, float] = {
        "causal_observed_prefill_ms": float(
            ledger["prefill_observed_ms"]
        ),
        "causal_observed_decode_ms": float(
            ledger["decode_observed_ms"]
        ),
        "causal_observed_total_ms": float(
            ledger["prefill_observed_ms"]
            + ledger["decode_observed_ms"]
        ),
        "causal_analysis_total_service_ms": (
            total_service
        ),
        "causal_analysis_mean_service_ms": (
            total_service / selected.size
        ),
        "causal_decode_fraction": float(
            np.sum(decode_all[selected])
            / total_service
        ),
        "causal_decode_step_count": float(
            ledger["decode_step_count"]
        ),
        "causal_decode_mean_batch_size": (
            statistics.mean(
                ledger[
                    "decode_batch_sizes"
                ]
            )
        ),
        "causal_decode_p95_batch_size": (
            float(
                np.percentile(
                    ledger[
                        "decode_batch_sizes"
                    ],
                    95,
                )
            )
        ),
        "causal_maximum_decode_step_error_ms": float(
            ledger[
                "maximum_decode_step_error_ms"
            ]
        ),
        "causal_event_count": float(
            ledger["event_count"]
        ),
        "causal_decay_clock_ms": float(
            ledger["decay_clock_ms"]
        ),
    }
    normalized_shares = []
    for index, tenant in enumerate(
        TENANTS
    ):
        name = tenant["name"]
        share = (
            service_values[index]
            / total_service
        )
        shares.append(share)
        normalized = (
            share / float(target[index])
        )
        normalized_shares.append(
            normalized
        )
        metrics[
            f"tenant_{name}_causal_service_ms"
        ] = service_values[index]
        metrics[
            f"tenant_{name}_causal_service_share"
        ] = share
        metrics[
            f"tenant_{name}_causal_target_share"
        ] = float(target[index])
        metrics[
            f"tenant_{name}_causal_share_error"
        ] = (
            share - float(target[index])
        )
        metrics[
            f"tenant_{name}_causal_normalized_share"
        ] = normalized
    metrics[
        "causal_service_share_jain_index"
    ] = jain_index(shares)
    metrics[
        "causal_reservation_normalized_jain_index"
    ] = jain_index(normalized_shares)
    metrics[
        "causal_target_l1_error"
    ] = 0.5 * sum(
        abs(
            share - float(target[index])
        )
        for index, share in enumerate(
            shares
        )
    )
    selected_decisions = [
        decision
        for decision in decisions
        if analysis_start
        <= int(decision["request"])
        < analysis_end
    ]
    if selected_decisions:
        metrics[
            "mean_causal_service_ledger_ms"
        ] = statistics.mean(
            decision.get(
                "causal_service_ledger_total_ms",
                0.0,
            )
            for decision
            in selected_decisions
        )
        metrics[
            "mean_causal_service_deficit_abs"
        ] = statistics.mean(
            abs(
                decision.get(
                    "causal_service_deficit_before",
                    0.0,
                )
            )
            for decision
            in selected_decisions
        )
    else:
        metrics[
            "mean_causal_service_ledger_ms"
        ] = 0.0
        metrics[
            "mean_causal_service_deficit_abs"
        ] = 0.0
    return metrics


def simulate_policy(
    *,
    arrivals: np.ndarray,
    workload: dict[str, np.ndarray],
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    policy_name: str,
    kv_budget_tokens: int,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    decode_lookup = standalone_decode_lookup(
        decode_model=decode_model,
        decode_service_models=(
            decode_service_models
        ),
    )
    configuration = POLICIES[policy_name]
    reservation = (
        "weighted"
        if "reserved" in policy_name
        else "equal"
    )
    target = target_weights(reservation)
    (
        prefill_observer,
        decode_observer,
        ledger,
    ) = make_service_ledger(
        tenant_ids=workload["tenant_ids"],
        decode_model=decode_model,
        decode_service_models=(
            decode_service_models
        ),
        target=target,
    )
    if configuration["kind"] == (
        "e57_control"
    ):
        (
            admission_selector,
            admission_decisions,
        ) = make_tenant_selector(
            tenant_ids=workload[
                "tenant_ids"
            ],
            decode_lookup=decode_lookup,
            policy_name=policy_name,
            prefill_model=prefill_model,
        )
    else:
        (
            admission_selector,
            admission_decisions,
        ) = make_causal_selector(
            tenant_ids=workload[
                "tenant_ids"
            ],
            decode_lookup=decode_lookup,
            policy_name=policy_name,
            ledger=ledger,
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
        admission_selector=admission_selector,
        decode_service_observer=(
            decode_observer
        ),
        prefill_service_observer=(
            prefill_observer
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
    if configuration["kind"] == "e57_control":
        selected = [
            decision
            for decision in admission_decisions
            if analysis_start
            <= int(decision["request"])
            < analysis_end
        ]
        metrics[
            "mean_fairness_clock_delta_ms"
        ] = statistics.mean(
            decision.get(
                "decay_clock_delta_ms",
                decision[
                    "candidate_prefill_ms"
                ],
            )
            for decision in selected
        )
        metrics[
            "mean_candidate_decode_ms"
        ] = statistics.mean(
            decision["candidate_decode_ms"]
            for decision in selected
        )
        metrics[
            "mean_candidate_total_service_ms"
        ] = statistics.mean(
            decision[
                "candidate_total_service_ms"
            ]
            for decision in selected
        )
        metrics[
            "mean_effective_history_weight"
        ] = statistics.mean(
            decision[
                "effective_history_weight_before"
            ]
            for decision in selected
        )
        metrics["mean_decay_factor"] = (
            statistics.mean(
                decision["decay_factor"]
                for decision in selected
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
    return metrics


def e57_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "tenant_reservation_fairness"
        / "tenant_reservation_fairness_full.json"
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
        if record["workload_mode"]
        == "tenant_stationary"
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
                    source = decode_model["source"]
                    request_rate = (
                        decode_capacities[source]
                        * offered_load
                    )
                    for seed in seeds:
                        workload_seed = (
                            seed
                            + replicate_index
                            * 100_000
                            + (
                                0
                                if prefix_mode == "none"
                                else 10_000
                            )
                        )
                        (
                            workload,
                            composition,
                        ) = make_tenant_workload(
                            num_requests=(
                                args.num_requests
                            ),
                            seed=workload_seed,
                            prefix_mode=prefix_mode,
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
                        arrivals = make_arrivals(
                            num_requests=(
                                args.num_requests
                            ),
                            requests_per_second=(
                                request_rate
                            ),
                            seed=workload_seed + 1,
                        )
                        metrics = simulate_policy(
                            arrivals=arrivals,
                            workload=workload,
                            decode_model=decode_model,
                            decode_service_models=(
                                decode_service_models
                            ),
                            prefill_model=prefill_model,
                            policy_name=policy_name,
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
                                "Hard slice budget failed"
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
                                "Admission threshold failed"
                            )
                        scenarios.append(
                            {
                                "service_replicate": (
                                    source
                                ),
                                "seed": seed,
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
                        "policy": policy_name,
                        "configuration": POLICIES[
                            policy_name
                        ],
                        "metrics": aggregate_metrics(
                            scenarios
                        ),
                        "scenarios": scenarios,
                    }
                )

    reference = e57_records(args.root)
    equivalence = []
    equivalence_enabled = (
        args.num_requests == 2_000
        and args.warmup_requests == 200
        and args.tail_guard_requests == 100
        and args.capacity_requests == 1_500
        and args.seeds == 3
        and args.seed == 40_000
        and offered_loads == [0.3, 0.4]
    )
    if equivalence_enabled:
        for record in records:
            if (
                record["policy"]
                not in CONTROL_POLICIES
            ):
                continue
            expected_record = reference[
                (
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    record["policy"],
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
            for scenario in record["scenarios"]:
                key = (
                    scenario["service_replicate"],
                    scenario["seed"],
                )
                expected = expected_scenarios[key]
                numeric_metrics = [
                    metric
                    for metric, value
                    in expected.items()
                    if isinstance(
                        value,
                        (int, float),
                    )
                    and not isinstance(value, bool)
                ]
                maximum_difference = max(
                    abs(
                        scenario[metric]
                        - expected[metric]
                    )
                    for metric in numeric_metrics
                )
                if maximum_difference > 1e-9:
                    raise RuntimeError(
                        "E57 control equivalence "
                        "failed: "
                        f"{record['policy']=} "
                        f"{key=} "
                        f"{maximum_difference=}"
                    )
                equivalence.append(
                    {
                        "prefix_mode": record[
                            "prefix_mode"
                        ],
                        "offered_load": record[
                            "offered_load_vs_decode_capacity"
                        ],
                        "policy": record[
                            "policy"
                        ],
                        "service_replicate": key[0],
                        "seed": key[1],
                        "maximum_absolute_difference": (
                            maximum_difference
                        ),
                    }
                )

    result = {
        "suite": (
            "causal_service_tenant_"
            "controller_simulation"
        ),
        "model": (
            "E57 tenant admission with a shared "
            "fairness ledger updated by actual "
            "prefill slices and exact fitted-model "
            "causal batched decode allocations."
        ),
        "prefill_service_model": prefill_model,
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
                "credit = clip(exp(gain * "
                "(target service share - "
                "current decayed causal service "
                "share)), 0.5, 4.0)"
            ),
        },
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
            "scenario_count": scenario_count,
            "e57_equivalence_enabled": (
                equivalence_enabled
            ),
            "e57_equivalence_checks": (
                equivalence
            ),
            "e57_maximum_absolute_difference": (
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
                "The v40 base asserts exact "
                "accepted prefill/decode tokens, "
                "causal pairs, finite completion "
                "only for accepted requests, max "
                "batch 64, and KV within budget. "
                "Prefill and decode observers cover "
                "their busy times exactly; every "
                "decode allocation sums to its "
                "step; the causal ledger decays only "
                "on observed GPU service; every "
                "admitted request satisfies its "
                "threshold; and hard slices report "
                "zero violations."
            ),
        },
        "records": records,
        "limitations": [
            "The study inherits all E57-E58 limitations and uses a fitted service model rather than a live serving engine.",
            "Causal decode allocation is exact only for the affine fitted attention model plus batch-dependent MLP term.",
            "Admission reacts to service already consumed; it does not reserve future causal service for accepted requests, so feedback can lag under bursts.",
            "The target controls accepted GPU service share, while request and output-token acceptance remain secondary consequences.",
            "Rejected requests leave immediately and are not retried, routed to another replica, or placed in a durable reservation queue.",
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
        default="0.3,0.4",
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
