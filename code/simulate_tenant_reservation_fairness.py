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
from simulate_decayed_fair_admission import (
    make_decayed_fair_selector,
)
from simulate_decayed_fair_admission import (
    simulate_decayed_policy as simulate_e55_policy,
)
from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    MAX_BATCH,
    aggregate_metrics,
    decode_step_ms,
    load_service_models,
    make_arrivals,
    make_workload,
)
from simulate_decode_continuous_batching import (
    simulate as simulate_decode_only,
)
from simulate_nonstationary_fair_admission import (
    exact_class_counts,
    make_nonstationary_workload,
)
from simulate_prefill_admission_safety import (
    NOMINAL_BACKLOG_LIMIT_MS,
    estimate_capacity,
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
from simulate_weighted_fair_admission import (
    BASE_LIMIT_MS,
    BASE_SAFETY_FACTOR,
    MAX_CREDIT,
    MIN_CREDIT,
    PRIOR_ACCEPTANCE,
    PRIOR_COUNT,
    PROMPT_CLASSES,
)


OUTPUT_CLASSES = (32, 128, 512)
TENANTS = (
    {
        "id": 0,
        "name": "interactive",
        "request_share": 0.50,
        "prompt_probabilities": (
            0.75,
            0.20,
            0.05,
        ),
        "output_probabilities": (
            0.75,
            0.20,
            0.05,
        ),
        "cache_hit_probability": 0.70,
        "reservation_multiplier": 1.00,
    },
    {
        "id": 1,
        "name": "long_context",
        "request_share": 0.25,
        "prompt_probabilities": (
            0.10,
            0.25,
            0.65,
        ),
        "output_probabilities": (
            0.55,
            0.35,
            0.10,
        ),
        "cache_hit_probability": 0.50,
        "reservation_multiplier": 0.85,
    },
    {
        "id": 2,
        "name": "generation_heavy",
        "request_share": 0.25,
        "prompt_probabilities": (
            0.35,
            0.55,
            0.10,
        ),
        "output_probabilities": (
            0.10,
            0.25,
            0.65,
        ),
        "cache_hit_probability": 0.10,
        "reservation_multiplier": 0.70,
    },
)
TENANT_NAMES = tuple(
    tenant["name"]
    for tenant in TENANTS
)
TENANT_BY_ID = {
    tenant["id"]: tenant
    for tenant in TENANTS
}
PREFILL_HALF_LIFE_MS = 40_000.0
END_TO_END_HALF_LIFE_MS = 320_000.0

POLICIES = {
    "prompt_equal_prefill_service40": {
        "identity": "prompt_class",
        "clock": "prefill_service",
        "half_life_ms": PREFILL_HALF_LIFE_MS,
        "reservation": "equal",
    },
    "tenant_equal_prefill_service40": {
        "identity": "tenant",
        "clock": "prefill_service",
        "half_life_ms": PREFILL_HALF_LIFE_MS,
        "reservation": "equal",
        "gain": 4.0,
    },
    "tenant_equal_e2e_service320": {
        "identity": "tenant",
        "clock": "end_to_end_service",
        "half_life_ms": END_TO_END_HALF_LIFE_MS,
        "reservation": "equal",
        "gain": 4.0,
    },
    "tenant_equal_e2e_service320_gain8": {
        "identity": "tenant",
        "clock": "end_to_end_service",
        "half_life_ms": END_TO_END_HALF_LIFE_MS,
        "reservation": "equal",
        "gain": 8.0,
    },
    "tenant_reserved_e2e_service320": {
        "identity": "tenant",
        "clock": "end_to_end_service",
        "half_life_ms": END_TO_END_HALF_LIFE_MS,
        "reservation": "weighted",
        "gain": 4.0,
    },
}


def tenant_specifications() -> list[dict[str, Any]]:
    return [
        {
            **tenant,
            "prompt_probabilities": list(
                tenant[
                    "prompt_probabilities"
                ]
            ),
            "output_probabilities": list(
                tenant[
                    "output_probabilities"
                ]
            ),
        }
        for tenant in TENANTS
    ]


def make_tenant_workload(
    *,
    num_requests: int,
    seed: int,
    prefix_mode: str,
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    request_counts = exact_class_counts(
        length=num_requests,
        probabilities=tuple(
            tenant["request_share"]
            for tenant in TENANTS
        ),
    )
    tenant_ids = np.concatenate(
        [
            np.full(
                count,
                tenant["id"],
                dtype=np.int64,
            )
            for tenant, count in zip(
                TENANTS,
                request_counts,
            )
        ]
    )
    rng = np.random.default_rng(seed + 101)
    rng.shuffle(tenant_ids)
    prompt_tokens = np.empty(
        num_requests,
        dtype=np.int64,
    )
    output_tokens = np.empty(
        num_requests,
        dtype=np.int64,
    )
    prefix_hits = np.zeros(
        num_requests,
        dtype=np.bool_,
    )
    composition = []
    for tenant in TENANTS:
        tenant_id = int(tenant["id"])
        indices = np.flatnonzero(
            tenant_ids == tenant_id
        )
        count = int(indices.size)
        prompt_counts = exact_class_counts(
            length=count,
            probabilities=tenant[
                "prompt_probabilities"
            ],
        )
        output_counts = exact_class_counts(
            length=count,
            probabilities=tenant[
                "output_probabilities"
            ],
        )
        prompt_values = np.concatenate(
            [
                np.full(
                    member_count,
                    member,
                    dtype=np.int64,
                )
                for member, member_count
                in zip(
                    PROMPT_CLASSES,
                    prompt_counts,
                )
            ]
        )
        output_values = np.concatenate(
            [
                np.full(
                    member_count,
                    member,
                    dtype=np.int64,
                )
                for member, member_count
                in zip(
                    OUTPUT_CLASSES,
                    output_counts,
                )
            ]
        )
        prompt_rng = np.random.default_rng(
            seed + 200 + tenant_id
        )
        output_rng = np.random.default_rng(
            seed + 300 + tenant_id
        )
        prompt_rng.shuffle(prompt_values)
        output_rng.shuffle(output_values)
        prompt_tokens[indices] = prompt_values
        output_tokens[indices] = (
            output_values
        )
        if prefix_mode == (
            "half_requests_75pct_cached"
        ):
            cache_count = int(
                round(
                    count
                    * tenant[
                        "cache_hit_probability"
                    ]
                )
            )
            cache_flags = np.zeros(
                count,
                dtype=np.bool_,
            )
            cache_flags[:cache_count] = True
            cache_rng = np.random.default_rng(
                seed + 400 + tenant_id
            )
            cache_rng.shuffle(cache_flags)
            prefix_hits[indices] = cache_flags
        composition.append(
            {
                "tenant": tenant["name"],
                "tenant_id": tenant_id,
                "request_count": count,
                "prompt_class_counts": {
                    str(member): member_count
                    for member, member_count
                    in zip(
                        PROMPT_CLASSES,
                        prompt_counts,
                    )
                },
                "output_class_counts": {
                    str(member): member_count
                    for member, member_count
                    in zip(
                        OUTPUT_CLASSES,
                        output_counts,
                    )
                },
                "cache_hit_count": int(
                    np.sum(
                        prefix_hits[indices]
                    )
                ),
            }
        )
    cached_tokens = np.zeros(
        num_requests,
        dtype=np.int64,
    )
    if prefix_mode == (
        "half_requests_75pct_cached"
    ):
        cached_values = (
            prompt_tokens * 0.75
        ).astype(np.int64)
        cached_values = (
            cached_values // 16
        ) * 16
        cached_tokens[prefix_hits] = (
            cached_values[prefix_hits]
        )
    elif prefix_mode != "none":
        raise ValueError(prefix_mode)
    uncached_tokens = (
        prompt_tokens - cached_tokens
    )
    if np.any(uncached_tokens < 1):
        raise RuntimeError(
            "Tenant workload cached full prompt"
        )
    workload = {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "reservation_tokens": (
            prompt_tokens + output_tokens
        ),
        "cached_tokens": cached_tokens,
        "uncached_tokens": uncached_tokens,
        "prefix_hits": prefix_hits,
        "tenant_ids": tenant_ids,
    }
    return workload, composition


def standalone_decode_lookup(
    *,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
) -> dict[tuple[int, int], float]:
    lookup = {}
    for prompt in PROMPT_CLASSES:
        for output in OUTPUT_CLASSES:
            total_ms = sum(
                decode_step_ms(
                    decode_model,
                    decode_service_models,
                    np.asarray(
                        [prompt + generated],
                        dtype=np.int64,
                    ),
                    kernel_mode="ragged",
                )
                for generated in range(output)
            )
            if total_ms <= 0.0:
                raise RuntimeError(
                    "Invalid decode service"
                )
            lookup[(prompt, output)] = (
                total_ms
            )
    return lookup


def annotate_prompt_selector(
    *,
    tenant_ids: np.ndarray,
    decode_lookup: dict[tuple[int, int], float],
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[dict[str, float]],
]:
    base_selector, decisions = (
        make_decayed_fair_selector(
            clock="prefill_service",
            half_life_ms=(
                PREFILL_HALF_LIFE_MS
            ),
            prefill_model=prefill_model,
        )
    )

    def select(state: dict[str, Any]) -> bool:
        request = int(state["request"])
        allow = base_selector(state)
        decision = decisions[-1]
        prompt = int(
            state["prompt_tokens"][request]
        )
        output = int(
            state["output_tokens"][request]
        )
        tenant_id = int(
            tenant_ids[request]
        )
        decode_ms = decode_lookup[
            (prompt, output)
        ]
        decision.update(
            {
                "tenant_id": float(
                    tenant_id
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
                "reservation_multiplier": (
                    1.0
                ),
            }
        )
        return allow

    return select, decisions


def make_tenant_selector(
    *,
    tenant_ids: np.ndarray,
    decode_lookup: dict[tuple[int, int], float],
    policy_name: str,
    prefill_model: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[dict[str, float]],
]:
    configuration = POLICIES[policy_name]
    offered = {
        tenant["id"]: 0.0
        for tenant in TENANTS
    }
    accepted = {
        tenant["id"]: 0.0
        for tenant in TENANTS
    }
    decisions: list[dict[str, float]] = []
    half_life_ms = float(
        configuration["half_life_ms"]
    )

    def select(state: dict[str, Any]) -> bool:
        request = int(state["request"])
        tenant_id = int(
            tenant_ids[request]
        )
        tenant = TENANT_BY_ID[tenant_id]
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
        decode_ms = decode_lookup[
            (prompt, output)
        ]
        total_service_ms = (
            decision["candidate_prefill_ms"]
            + decode_ms
        )
        clock_delta_ms = (
            decision["candidate_prefill_ms"]
            if configuration["clock"]
            == "prefill_service"
            else total_service_ms
        )
        decay_factor = 2.0 ** (
            -clock_delta_ms / half_life_ms
        )
        for member in offered:
            offered[member] *= decay_factor
            accepted[member] *= decay_factor
            if (
                accepted[member]
                > offered[member] + 1e-12
                or offered[member] < 0.0
            ):
                raise RuntimeError(
                    "Tenant decayed accounting "
                    "failed"
                )
        ratios = {
            member: (
                accepted[member]
                + PRIOR_ACCEPTANCE
                * PRIOR_COUNT
            )
            / (
                offered[member]
                + PRIOR_COUNT
            )
            for member in offered
        }
        target_multipliers = {
            member: (
                1.0
                if configuration[
                    "reservation"
                ]
                == "equal"
                else TENANT_BY_ID[
                    member
                ][
                    "reservation_multiplier"
                ]
            )
            for member in offered
        }
        normalized = {
            member: (
                ratios[member]
                / target_multipliers[member]
            )
            for member in offered
        }
        normalized_mean = (
            statistics.mean(
                normalized.values()
            )
        )
        desired_acceptance = (
            target_multipliers[tenant_id]
            * normalized_mean
        )
        deficit = (
            desired_acceptance
            - ratios[tenant_id]
        )
        credit = min(
            MAX_CREDIT,
            max(
                MIN_CREDIT,
                math.exp(
                    configuration.get(
                        "gain",
                        4.0,
                    )
                    * deficit
                ),
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
        history_weight_before = sum(
            offered.values()
        )
        offered[tenant_id] += 1.0
        accepted[tenant_id] += float(
            allow
        )
        decisions.append(
            {
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
                "controller_gain": (
                    configuration.get(
                        "gain",
                        4.0,
                    )
                ),
                "decay_half_life_ms": (
                    half_life_ms
                ),
                "decay_clock_delta_ms": (
                    clock_delta_ms
                ),
                "decay_factor": (
                    decay_factor
                ),
                "effective_history_weight_before": (
                    history_weight_before
                ),
                "candidate_decode_ms": (
                    decode_ms
                ),
                "candidate_total_service_ms": (
                    total_service_ms
                ),
                "reservation_multiplier": (
                    target_multipliers[
                        tenant_id
                    ]
                ),
                "smoothed_tenant_acceptance_before": (
                    ratios[tenant_id]
                ),
                "normalized_tenant_acceptance_before": (
                    normalized[tenant_id]
                ),
                "normalized_tenant_mean_before": (
                    normalized_mean
                ),
                "tenant_deficit_before": (
                    deficit
                ),
            }
        )
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


def tenant_metrics(
    decisions: list[dict[str, float]],
    *,
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
    if not selected:
        raise RuntimeError(
            "No tenant decisions"
        )
    metrics: dict[str, float] = {}
    acceptance_values = []
    service_acceptance_values = []
    normalized_values = []
    accepted_services = []
    offered_services = []
    for tenant in TENANTS:
        tenant_id = int(tenant["id"])
        name = tenant["name"]
        members = [
            decision
            for decision in selected
            if int(decision["tenant_id"])
            == tenant_id
        ]
        if not members:
            raise RuntimeError(
                f"Missing tenant {name}"
            )
        accepted_members = [
            decision
            for decision in members
            if decision["allowed"] == 1.0
        ]
        acceptance = (
            len(accepted_members)
            / len(members)
        )
        offered_service = sum(
            decision[
                "candidate_total_service_ms"
            ]
            for decision in members
        )
        accepted_service = sum(
            decision[
                "candidate_total_service_ms"
            ]
            for decision
            in accepted_members
        )
        service_acceptance = (
            accepted_service
            / offered_service
        )
        offered_output = sum(
            decision["output_tokens"]
            for decision in members
        )
        accepted_output = sum(
            decision["output_tokens"]
            for decision
            in accepted_members
        )
        output_acceptance = (
            accepted_output
            / offered_output
        )
        multiplier = float(
            tenant[
                "reservation_multiplier"
            ]
        )
        normalized = (
            acceptance / multiplier
        )
        metrics[
            f"tenant_{name}_offered_fraction"
        ] = len(members) / len(selected)
        metrics[
            f"tenant_{name}_acceptance"
        ] = acceptance
        metrics[
            f"tenant_{name}_output_token_acceptance"
        ] = output_acceptance
        metrics[
            f"tenant_{name}_service_acceptance"
        ] = service_acceptance
        metrics[
            f"tenant_{name}_mean_total_service_ms"
        ] = offered_service / len(members)
        metrics[
            f"tenant_{name}_mean_credit"
        ] = statistics.mean(
            decision[
                "prompt_credit_multiplier"
            ]
            for decision in members
        )
        metrics[
            f"tenant_{name}_normalized_acceptance"
        ] = normalized
        acceptance_values.append(
            acceptance
        )
        service_acceptance_values.append(
            service_acceptance
        )
        normalized_values.append(
            normalized
        )
        accepted_services.append(
            accepted_service
        )
        offered_services.append(
            offered_service
        )
    total_accepted_service = sum(
        accepted_services
    )
    total_offered_service = sum(
        offered_services
    )
    for index, tenant in enumerate(
        TENANTS
    ):
        name = tenant["name"]
        metrics[
            f"tenant_{name}_accepted_service_share"
        ] = (
            accepted_services[index]
            / total_accepted_service
        )
        metrics[
            f"tenant_{name}_offered_service_share"
        ] = (
            offered_services[index]
            / total_offered_service
        )
    metrics[
        "tenant_acceptance_jain_index"
    ] = jain_index(acceptance_values)
    metrics[
        "tenant_service_acceptance_jain_index"
    ] = jain_index(
        service_acceptance_values
    )
    metrics[
        "tenant_reservation_normalized_jain_index"
    ] = jain_index(
        normalized_values
    )
    metrics[
        "tenant_acceptance_range"
    ] = max(acceptance_values) - min(
        acceptance_values
    )
    metrics[
        "tenant_service_acceptance_range"
    ] = max(
        service_acceptance_values
    ) - min(service_acceptance_values)
    metrics[
        "generation_vs_interactive_acceptance"
    ] = (
        acceptance_values[2]
        / max(acceptance_values[0], 1e-12)
    )
    metrics[
        "long_vs_interactive_acceptance"
    ] = (
        acceptance_values[1]
        / max(acceptance_values[0], 1e-12)
    )
    return metrics


def simulate_tenant_policy(
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
    decode_service_observer: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    prefill_service_observer: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> dict[str, float]:
    decode_lookup = standalone_decode_lookup(
        decode_model=decode_model,
        decode_service_models=(
            decode_service_models
        ),
    )
    if (
        POLICIES[policy_name]["identity"]
        == "prompt_class"
    ):
        (
            admission_selector,
            admission_decisions,
        ) = annotate_prompt_selector(
            tenant_ids=workload[
                "tenant_ids"
            ],
            decode_lookup=decode_lookup,
            prefill_model=prefill_model,
        )
    else:
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
            decode_service_observer
        ),
        prefill_service_observer=(
            prefill_service_observer
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
    return metrics


def estimate_tenant_capacity(
    *,
    num_requests: int,
    seed: int,
    prefix_mode: str,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    kv_budget_tokens: int,
) -> dict[str, float]:
    workload, _ = make_tenant_workload(
        num_requests=num_requests,
        seed=seed,
        prefix_mode=prefix_mode,
    )
    selector, _ = make_selector(
        policy=hard_selector_config(100.0),
        prefill_model=prefill_model,
    )
    metrics = simulate(
        arrivals=np.zeros(
            num_requests,
            dtype=np.float64,
        ),
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
        analysis_start=0,
        analysis_end=num_requests,
        chunk_selector=selector,
    )
    busy_ms = (
        metrics["prefill_busy_ms"]
        + metrics["decode_busy_ms"]
    )
    return {
        "requests_per_second": (
            num_requests * 1_000.0 / busy_ms
        ),
        "output_tokens_per_second": (
            metrics["completed_output_tokens"]
            * 1_000.0
            / busy_ms
        ),
    }


def e55_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "decayed_fair_admission"
        / "decayed_fair_admission_full.json"
    )
    result = json.loads(
        path.read_text(encoding="utf-8")
    )
    return {
        (
            record["workload_mode"],
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
    seeds = [
        args.seed + index
        for index in range(args.seeds)
    ]
    decode_capacities = {}
    control_capacities: dict[
        str,
        dict[str, Any],
    ] = {}
    tenant_capacities: dict[
        str,
        dict[str, Any],
    ] = {}
    for prefix_mode in PREFIX_MODES:
        control_capacities[
            prefix_mode
        ] = {}
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
            control_capacities[
                prefix_mode
            ][source] = estimate_capacity(
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
    for workload_mode in (
        "e55_rotating_control",
        "tenant_stationary",
    ):
        for prefix_mode in PREFIX_MODES:
            policy_names = (
                (
                    "fair_gain_4_"
                    "service_half_40s",
                )
                if workload_mode
                == "e55_rotating_control"
                else tuple(POLICIES)
            )
            for offered_load in offered_loads:
                for policy_name in policy_names:
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
                                    if prefix_mode
                                    == "none"
                                    else 10_000
                                )
                            )
                            if (
                                workload_mode
                                == "e55_rotating_control"
                            ):
                                (
                                    workload,
                                    phase_ids,
                                    _,
                                ) = (
                                    make_nonstationary_workload(
                                        num_requests=(
                                            args.num_requests
                                        ),
                                        seed=workload_seed,
                                        prefix_mode=(
                                            prefix_mode
                                        ),
                                    )
                                )
                            else:
                                (
                                    workload,
                                    composition,
                                ) = make_tenant_workload(
                                    num_requests=(
                                        args.num_requests
                                    ),
                                    seed=workload_seed,
                                    prefix_mode=(
                                        prefix_mode
                                    ),
                                )
                                reference_key = (
                                    prefix_mode
                                )
                                if (
                                    reference_key
                                    not in composition_reference
                                ):
                                    composition_reference[
                                        reference_key
                                    ] = composition
                                elif (
                                    composition_reference[
                                        reference_key
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
                                seed=(
                                    workload_seed + 1
                                ),
                            )
                            if (
                                workload_mode
                                == "e55_rotating_control"
                            ):
                                metrics = (
                                    simulate_e55_policy(
                                        arrivals=arrivals,
                                        workload=workload,
                                        phase_ids=phase_ids,
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
                                )
                            else:
                                metrics = (
                                    simulate_tenant_policy(
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
                            system_capacity = (
                                tenant_capacities[
                                    prefix_mode
                                ][source][
                                    "requests_per_second"
                                ]
                                if workload_mode
                                == "tenant_stationary"
                                else control_capacities[
                                    prefix_mode
                                ][source][
                                    "requests_per_second"
                                ]
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
                                        system_capacity
                                    ),
                                    "offered_load_vs_system_capacity": (
                                        request_rate
                                        / system_capacity
                                    ),
                                    **metrics,
                                }
                            )
                            scenario_count += 1
                    records.append(
                        {
                            "workload_mode": (
                                workload_mode
                            ),
                            "prefix_mode": (
                                prefix_mode
                            ),
                            "offered_load_vs_decode_capacity": (
                                offered_load
                            ),
                            "policy": policy_name,
                            "configuration": (
                                {
                                    "kind": (
                                        "e55_control"
                                    )
                                }
                                if workload_mode
                                == "e55_rotating_control"
                                else POLICIES[
                                    policy_name
                                ]
                            ),
                            "metrics": aggregate_metrics(
                                scenarios
                            ),
                            "scenarios": scenarios,
                        }
                    )

    reference = e55_records(args.root)
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
                record["workload_mode"]
                != "e55_rotating_control"
            ):
                continue
            expected_record = reference[
                (
                    "rotating_prompt_mix",
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    (
                        "fair_gain_4_"
                        "service_half_40s"
                    ),
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
                        "E55 control equivalence "
                        "failed: "
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
                        "service_replicate": key[0],
                        "seed": key[1],
                        "maximum_absolute_difference": (
                            maximum_difference
                        ),
                    }
                )

    result = {
        "suite": (
            "tenant_reservation_fairness_"
            "simulation"
        ),
        "model": (
            "Three tenant classes with distinct "
            "prompt, output, and prefix-reuse "
            "mixes; fairness identity is prompt "
            "class or tenant, and the controller "
            "clock is prefill or measured-service "
            "prefill plus standalone decode."
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
            "gain": 4.0,
            "prefill_half_life_ms": (
                PREFILL_HALF_LIFE_MS
            ),
            "end_to_end_half_life_ms": (
                END_TO_END_HALF_LIFE_MS
            ),
            "base_safety_factor": (
                BASE_SAFETY_FACTOR
            ),
            "base_limit_ms": BASE_LIMIT_MS,
            "prior_count": PRIOR_COUNT,
            "prior_acceptance": (
                PRIOR_ACCEPTANCE
            ),
            "minimum_credit": MIN_CREDIT,
            "maximum_credit": MAX_CREDIT,
            "weighted_reservation_rule": (
                "equalize acceptance divided by "
                "tenant reservation multiplier"
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
            "e55_equivalence_enabled": (
                equivalence_enabled
            ),
            "e55_equivalence_checks": (
                equivalence
            ),
            "e55_maximum_absolute_difference": (
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
                "Tenant prompt/output/cache counts "
                "are exact, decoded service is "
                "positive for all nine shape "
                "classes, decayed accepted weight "
                "never exceeds offered weight, "
                "every admitted request satisfies "
                "its unchanged admission threshold, "
                "and hard slices report zero "
                "violations."
            ),
        },
        "records": records,
        "limitations": [
            "The service model inherits all E49-E56 limitations and is not a live serving-engine measurement.",
            "Standalone decode service sums batch-1 measured-service steps and does not allocate shared batched decode cost causally among tenants.",
            "Tenant identities and reservation multipliers are synthetic and chosen to expose prompt-, output-, and cache-correlated work.",
            "The end-to-end clock includes predicted prefill plus standalone decode, not CPU, ingress, sampling, networking, or realized queue occupancy.",
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
