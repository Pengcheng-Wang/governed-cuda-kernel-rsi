from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    MAX_BATCH,
    WORKLOADS,
    aggregate_metrics,
    decode_step_ms,
    estimate_capacity,
    load_service_models,
    make_arrivals,
    make_workload,
    percentile,
    predicted_ttft_ms,
)


POLICIES = {
    "unguarded": {
        "ttft_guard": False,
        "completion_guard": False,
    },
    "ttft_guard": {
        "ttft_guard": True,
        "completion_guard": False,
    },
    "dual_slo_guard": {
        "ttft_guard": True,
        "completion_guard": True,
    },
}


def class_key(
    prompt_tokens: int,
    output_tokens: int,
) -> str:
    return (
        f"prompt_{prompt_tokens}_"
        f"output_{output_tokens}"
    )


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
    if denominator <= 0.0:
        return 1.0
    return float(
        np.sum(array) ** 2 / denominator
    )


def weighted_acceptance(
    *,
    selected: np.ndarray,
    accepted: np.ndarray,
    mask: np.ndarray,
) -> float:
    members = selected[mask[selected]]
    if members.size < 1:
        raise RuntimeError(
            "Empty fairness class"
        )
    return float(
        np.mean(accepted[members])
    )


def predicted_completion_ms(
    *,
    request: int,
    predicted_ttft: float,
    active: list[int],
    generated: np.ndarray,
    prompt_tokens: np.ndarray,
    output_tokens: np.ndarray,
    model: dict[str, Any],
    service_models: dict[str, Any],
) -> float:
    current_contexts = [
        int(prompt_tokens[item])
        + int(generated[item])
        for item in active
    ]
    mean_request_context = (
        int(prompt_tokens[request])
        + (
            int(output_tokens[request]) - 1
        )
        / 2.0
    )
    if len(current_contexts) >= MAX_BATCH:
        representative = np.asarray(
            current_contexts,
            dtype=np.float64,
        )
        representative[-1] = (
            mean_request_context
        )
    else:
        representative = np.asarray(
            [
                *current_contexts,
                mean_request_context,
            ],
            dtype=np.float64,
        )
    step_ms = decode_step_ms(
        model,
        service_models,
        representative,
        kernel_mode="ragged",
    )
    return predicted_ttft + (
        max(
            int(output_tokens[request]) - 1,
            0,
        )
        * step_ms
    )


def aggregate_class_metrics(
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted(
        scenarios[0]["classes"]
    )
    result = {}
    for key in keys:
        metrics = scenarios[0]["classes"][key]
        result[key] = {}
        for metric, value in metrics.items():
            if not isinstance(
                value,
                (int, float),
            ) or isinstance(value, bool):
                continue
            values = [
                scenario["classes"][key][metric]
                for scenario in scenarios
                if isinstance(
                    scenario["classes"][key][metric],
                    (int, float),
                )
                and math.isfinite(
                    scenario["classes"][key][metric]
                )
            ]
            if values:
                result[key][metric] = {
                    "median": statistics.median(
                        values
                    ),
                    "minimum": min(values),
                    "maximum": max(values),
                }
    return result


def simulate(
    *,
    arrivals: np.ndarray,
    workload: dict[str, np.ndarray],
    model: dict[str, Any],
    service_models: dict[str, Any],
    policy_name: str,
    kv_budget_tokens: int,
    ttft_target_ms: float,
    completion_target_ms: float,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, Any]:
    policy = POLICIES[policy_name]
    prompt_tokens = workload["prompt_tokens"]
    output_tokens = workload["output_tokens"]
    reservation_tokens = workload[
        "reservation_tokens"
    ]
    num_requests = arrivals.size
    if not (
        prompt_tokens.size
        == output_tokens.size
        == reservation_tokens.size
        == num_requests
    ):
        raise ValueError("Workload length mismatch")

    accepted = np.zeros(
        num_requests,
        dtype=np.bool_,
    )
    rejected = np.zeros(
        num_requests,
        dtype=np.bool_,
    )
    generated = np.zeros(
        num_requests,
        dtype=np.int64,
    )
    active_start = np.full(
        num_requests,
        np.nan,
        dtype=np.float64,
    )
    first_token = np.full(
        num_requests,
        np.nan,
        dtype=np.float64,
    )
    completion = np.full(
        num_requests,
        np.nan,
        dtype=np.float64,
    )
    predicted_ttft = np.full(
        num_requests,
        np.nan,
        dtype=np.float64,
    )
    predicted_completion = np.full(
        num_requests,
        np.nan,
        dtype=np.float64,
    )

    pending: deque[int] = deque()
    active: list[int] = []
    next_arrival = 0
    now_ms = 0.0
    busy_ms = 0.0
    reserved_tokens = 0
    peak_reserved_tokens = 0
    step_batches: list[int] = []
    step_ms_values: list[float] = []

    while (
        next_arrival < num_requests
        or pending
        or active
    ):
        while (
            next_arrival < num_requests
            and arrivals[next_arrival] <= now_ms
        ):
            request = next_arrival
            allow = True
            if policy["ttft_guard"]:
                predicted_ttft[request] = (
                    predicted_ttft_ms(
                        request=request,
                        now_ms=now_ms,
                        arrivals=arrivals,
                        active=active,
                        pending=pending,
                        generated=generated,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        reservation_tokens=(
                            reservation_tokens
                        ),
                        reserved_tokens=(
                            reserved_tokens
                        ),
                        kv_budget_tokens=(
                            kv_budget_tokens
                        ),
                        model=model,
                        service_models=(
                            service_models
                        ),
                    )
                )
                allow = (
                    predicted_ttft[request]
                    <= ttft_target_ms
                )
            if (
                allow
                and policy["completion_guard"]
            ):
                predicted_completion[request] = (
                    predicted_completion_ms(
                        request=request,
                        predicted_ttft=(
                            predicted_ttft[request]
                        ),
                        active=active,
                        generated=generated,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        model=model,
                        service_models=(
                            service_models
                        ),
                    )
                )
                allow = (
                    predicted_completion[request]
                    <= completion_target_ms
                )
            if allow:
                accepted[request] = True
                pending.append(request)
            else:
                rejected[request] = True
            next_arrival += 1

        while pending and len(active) < MAX_BATCH:
            request = pending[0]
            reservation = int(
                reservation_tokens[request]
            )
            if (
                reserved_tokens + reservation
                > kv_budget_tokens
            ):
                break
            pending.popleft()
            active.append(request)
            active_start[request] = now_ms
            reserved_tokens += reservation
            peak_reserved_tokens = max(
                peak_reserved_tokens,
                reserved_tokens,
            )

        if not active:
            if pending:
                raise RuntimeError(
                    "FCFS request cannot fit the KV budget"
                )
            if next_arrival < num_requests:
                now_ms = max(
                    now_ms,
                    float(arrivals[next_arrival]),
                )
                continue
            break

        contexts = np.array(
            [
                int(prompt_tokens[request])
                + int(generated[request])
                for request in active
            ],
            dtype=np.int64,
        )
        step_ms = decode_step_ms(
            model,
            service_models,
            contexts,
            kernel_mode="ragged",
        )
        ended_ms = now_ms + step_ms
        step_batches.append(len(active))
        step_ms_values.append(step_ms)

        survivors = []
        for request in active:
            if generated[request] == 0:
                first_token[request] = ended_ms
            generated[request] += 1
            if generated[request] == (
                output_tokens[request]
            ):
                completion[request] = ended_ms
                reserved_tokens -= int(
                    reservation_tokens[request]
                )
            else:
                survivors.append(request)
        active = survivors
        now_ms = ended_ms
        busy_ms += step_ms

    if not np.all(accepted | rejected):
        raise RuntimeError(
            "Not every request was classified"
        )
    if np.any(accepted & rejected):
        raise RuntimeError(
            "Request both accepted and rejected"
        )
    if not np.array_equal(
        generated[accepted],
        output_tokens[accepted],
    ):
        raise RuntimeError(
            "Accepted token accounting failed"
        )
    if np.any(np.isfinite(completion[rejected])):
        raise RuntimeError(
            "Rejected request completed"
        )
    if peak_reserved_tokens > kv_budget_tokens:
        raise RuntimeError("KV budget exceeded")
    if reserved_tokens != 0:
        raise RuntimeError("KV reservation leak")
    if max(step_batches) > MAX_BATCH:
        raise RuntimeError(
            "Batch-size invariant failed"
        )

    selected = np.arange(
        analysis_start,
        analysis_end,
    )
    selected_accepted = selected[
        accepted[selected]
    ]
    selected_rejected = selected[
        rejected[selected]
    ]
    if selected_accepted.size < 1:
        raise RuntimeError(
            "No accepted requests in analysis window"
        )
    selected_arrivals = arrivals[
        selected_accepted
    ]
    selected_first = first_token[
        selected_accepted
    ]
    selected_completion = completion[
        selected_accepted
    ]
    ttft = selected_first - selected_arrivals
    e2e = (
        selected_completion
        - selected_arrivals
    )
    queue_wait = (
        active_start[selected_accepted]
        - selected_arrivals
    )
    tpot = (
        selected_completion
        - selected_first
    ) / (
        output_tokens[selected_accepted] - 1
    )
    elapsed_ms = (
        float(np.max(selected_completion))
        - float(arrivals[analysis_start])
    )
    selected_offered_outputs = int(
        np.sum(output_tokens[selected])
    )
    selected_accepted_outputs = int(
        np.sum(
            output_tokens[selected_accepted]
        )
    )
    analysis_count = (
        analysis_end - analysis_start
    )

    classes = {}
    acceptance_values = []
    token_acceptance_values = []
    all_accepted_outputs = max(
        selected_accepted_outputs,
        1,
    )
    for prompt in WORKLOADS["mixed_chat"][
        "prompt_tokens"
    ]:
        for output in WORKLOADS["mixed_chat"][
            "output_tokens"
        ]:
            mask = (
                (prompt_tokens == prompt)
                & (output_tokens == output)
            )
            members = selected[mask[selected]]
            accepted_members = members[
                accepted[members]
            ]
            rejected_members = members[
                rejected[members]
            ]
            offered_output_count = int(
                np.sum(output_tokens[members])
            )
            accepted_output_count = int(
                np.sum(
                    output_tokens[
                        accepted_members
                    ]
                )
            )
            acceptance_fraction = (
                accepted_members.size
                / members.size
            )
            token_acceptance_fraction = (
                accepted_output_count
                / offered_output_count
            )
            acceptance_values.append(
                acceptance_fraction
            )
            token_acceptance_values.append(
                token_acceptance_fraction
            )
            metrics: dict[str, float] = {
                "offered_requests": float(
                    members.size
                ),
                "accepted_requests": float(
                    accepted_members.size
                ),
                "rejected_requests": float(
                    rejected_members.size
                ),
                "acceptance_fraction": (
                    acceptance_fraction
                ),
                "offered_output_tokens": float(
                    offered_output_count
                ),
                "accepted_output_tokens": float(
                    accepted_output_count
                ),
                "output_token_acceptance_fraction": (
                    token_acceptance_fraction
                ),
                "accepted_request_share": (
                    accepted_members.size
                    / selected_accepted.size
                ),
                "accepted_output_token_share": (
                    accepted_output_count
                    / all_accepted_outputs
                ),
            }
            if accepted_members.size:
                class_arrivals = arrivals[
                    accepted_members
                ]
                class_first = first_token[
                    accepted_members
                ]
                class_completion = completion[
                    accepted_members
                ]
                class_ttft = (
                    class_first
                    - class_arrivals
                )
                class_e2e = (
                    class_completion
                    - class_arrivals
                )
                class_tpot = (
                    class_completion
                    - class_first
                ) / (
                    output_tokens[
                        accepted_members
                    ]
                    - 1
                )
                metrics.update(
                    {
                        "p50_ttft_ms": percentile(
                            class_ttft,
                            50,
                        ),
                        "p99_ttft_ms": percentile(
                            class_ttft,
                            99,
                        ),
                        "p50_tpot_ms": percentile(
                            class_tpot,
                            50,
                        ),
                        "p99_tpot_ms": percentile(
                            class_tpot,
                            99,
                        ),
                        "p50_e2e_ms": percentile(
                            class_e2e,
                            50,
                        ),
                        "p99_e2e_ms": percentile(
                            class_e2e,
                            99,
                        ),
                        "ttft_slo_attainment": float(
                            np.mean(
                                class_ttft
                                <= ttft_target_ms
                            )
                        ),
                        "completion_slo_attainment": (
                            float(
                                np.mean(
                                    class_e2e
                                    <= (
                                        completion_target_ms
                                    )
                                )
                            )
                        ),
                    }
                )
            classes[
                class_key(prompt, output)
            ] = metrics

    short_output_acceptance = (
        weighted_acceptance(
            selected=selected,
            accepted=accepted,
            mask=(output_tokens == 32),
        )
    )
    long_output_acceptance = (
        weighted_acceptance(
            selected=selected,
            accepted=accepted,
            mask=(output_tokens == 512),
        )
    )
    short_prompt_acceptance = (
        weighted_acceptance(
            selected=selected,
            accepted=accepted,
            mask=(prompt_tokens == 512),
        )
    )
    long_prompt_acceptance = (
        weighted_acceptance(
            selected=selected,
            accepted=accepted,
            mask=(prompt_tokens == 6144),
        )
    )
    return {
        "accepted_fraction": (
            selected_accepted.size
            / analysis_count
        ),
        "rejected_fraction": (
            selected_rejected.size
            / analysis_count
        ),
        "accepted_output_token_fraction": (
            selected_accepted_outputs
            / selected_offered_outputs
        ),
        "throughput_requests_s": (
            selected_accepted.size
            * 1_000.0
            / elapsed_ms
        ),
        "throughput_output_tokens_s": (
            selected_accepted_outputs
            * 1_000.0
            / elapsed_ms
        ),
        "mean_queue_ms": float(
            np.mean(queue_wait)
        ),
        "p99_queue_ms": percentile(
            queue_wait,
            99,
        ),
        "p50_ttft_ms": percentile(ttft, 50),
        "p99_ttft_ms": percentile(ttft, 99),
        "ttft_slo_attainment": float(
            np.mean(ttft <= ttft_target_ms)
        ),
        "p50_tpot_ms": percentile(tpot, 50),
        "p99_tpot_ms": percentile(tpot, 99),
        "p50_e2e_ms": percentile(e2e, 50),
        "p99_e2e_ms": percentile(e2e, 99),
        "completion_slo_attainment": float(
            np.mean(
                e2e <= completion_target_ms
            )
        ),
        "mean_accepted_output_tokens": float(
            np.mean(
                output_tokens[
                    selected_accepted
                ]
            )
        ),
        "mean_rejected_output_tokens": (
            float(
                np.mean(
                    output_tokens[
                        selected_rejected
                    ]
                )
            )
            if selected_rejected.size
            else 0.0
        ),
        "acceptance_jain_index": jain_index(
            acceptance_values
        ),
        "output_token_acceptance_jain_index": (
            jain_index(
                token_acceptance_values
            )
        ),
        "acceptance_range": (
            max(acceptance_values)
            - min(acceptance_values)
        ),
        "short_output_acceptance": (
            short_output_acceptance
        ),
        "long_output_acceptance": (
            long_output_acceptance
        ),
        "long_vs_short_output_acceptance": (
            long_output_acceptance
            / max(
                short_output_acceptance,
                1e-12,
            )
        ),
        "short_prompt_acceptance": (
            short_prompt_acceptance
        ),
        "long_prompt_acceptance": (
            long_prompt_acceptance
        ),
        "long_vs_short_prompt_acceptance": (
            long_prompt_acceptance
            / max(
                short_prompt_acceptance,
                1e-12,
            )
        ),
        "mean_batch_size": float(
            np.mean(step_batches)
        ),
        "mean_step_ms": float(
            np.mean(step_ms_values)
        ),
        "peak_kv_tokens": float(
            peak_reserved_tokens
        ),
        "peak_kv_gib": (
            peak_reserved_tokens
            * KV_BYTES_PER_TOKEN
            / (1024**3)
        ),
        "busy_ms": busy_ms,
        "completed_output_tokens": float(
            np.sum(output_tokens[accepted])
        ),
        "classes": classes,
    }


def v37_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "decode_continuous_batching"
        / "decode_continuous_batching_full.json"
    )
    result = json.loads(
        path.read_text(encoding="utf-8")
    )
    return {
        (
            record["offered_load_vs_ragged_capacity"],
            record["policy"],
        ): record
        for record in result["records"]
        if record["workload"] == "mixed_chat"
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    service_models = load_service_models(args.root)
    max_fit_error = max(
        model["max_relative_error"]
        for model in service_models[
            "attention_replicates"
        ]
    )
    if max_fit_error > args.max_fit_relative_error:
        raise RuntimeError(
            "Attention service-model gate failed: "
            f"{max_fit_error=}"
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
    capacities = {}
    for replicate_index, model in enumerate(
        service_models["attention_replicates"]
    ):
        capacities[model["source"]] = (
            estimate_capacity(
                workload_name="mixed_chat",
                num_requests=args.capacity_requests,
                seed=(
                    args.seed
                    + 500_000
                    + replicate_index * 10_000
                    + 1_000
                ),
                model=model,
                service_models=service_models,
                kv_budget_tokens=kv_budget_tokens,
                ttft_target_ms=args.ttft_target_ms,
            )
        )

    records = []
    scenario_count = 0
    for offered_load in offered_loads:
        for policy_name in POLICIES:
            scenarios = []
            for replicate_index, model in enumerate(
                service_models[
                    "attention_replicates"
                ]
            ):
                request_rate = (
                    capacities[model["source"]][
                        "requests_per_second"
                    ]
                    * offered_load
                )
                for seed in seeds:
                    workload_seed = (
                        seed
                        + replicate_index * 100_000
                        + 10_000
                    )
                    workload = make_workload(
                        name="mixed_chat",
                        num_requests=args.num_requests,
                        seed=workload_seed,
                    )
                    arrivals = make_arrivals(
                        num_requests=args.num_requests,
                        requests_per_second=(
                            request_rate
                        ),
                        seed=workload_seed + 1,
                    )
                    metrics = simulate(
                        arrivals=arrivals,
                        workload=workload,
                        model=model,
                        service_models=service_models,
                        policy_name=policy_name,
                        kv_budget_tokens=(
                            kv_budget_tokens
                        ),
                        ttft_target_ms=(
                            args.ttft_target_ms
                        ),
                        completion_target_ms=(
                            args.completion_target_ms
                        ),
                        analysis_start=(
                            args.warmup_requests
                        ),
                        analysis_end=(
                            args.num_requests
                            - args.tail_guard_requests
                        ),
                    )
                    scenarios.append(
                        {
                            "service_replicate": (
                                model["source"]
                            ),
                            "seed": seed,
                            "request_rate_per_second": (
                                request_rate
                            ),
                            **metrics,
                        }
                    )
                    scenario_count += 1
            records.append(
                {
                    "workload": "mixed_chat",
                    "offered_load_vs_ragged_capacity": (
                        offered_load
                    ),
                    "policy": policy_name,
                    "configuration": POLICIES[
                        policy_name
                    ],
                    "metrics": aggregate_metrics(
                        scenarios
                    ),
                    "class_metrics": (
                        aggregate_class_metrics(
                            scenarios
                        )
                    ),
                    "scenarios": scenarios,
                }
            )

    reference = v37_records(args.root)
    equivalence = []
    policy_mapping = {
        "unguarded": "ragged_fcfs",
        "ttft_guard": "ragged_ttft_guard",
    }
    for record in records:
        policy_name = record["policy"]
        if policy_name not in policy_mapping:
            continue
        reference_record = reference[
            (
                record[
                    "offered_load_vs_ragged_capacity"
                ],
                policy_mapping[policy_name],
            )
        ]
        reference_scenarios = {
            (
                item["service_replicate"],
                item["seed"],
            ): item
            for item in reference_record[
                "scenarios"
            ]
        }
        for scenario in record["scenarios"]:
            key = (
                scenario["service_replicate"],
                scenario["seed"],
            )
            expected = reference_scenarios[key]
            maximum_difference = max(
                abs(
                    scenario[metric]
                    - expected[metric]
                )
                for metric in (
                    "accepted_fraction",
                    "rejected_fraction",
                    "throughput_requests_s",
                    "throughput_output_tokens_s",
                    "mean_queue_ms",
                    "p99_queue_ms",
                    "p50_ttft_ms",
                    "p99_ttft_ms",
                    "p50_tpot_ms",
                    "p99_tpot_ms",
                    "p50_e2e_ms",
                    "p99_e2e_ms",
                    "mean_batch_size",
                    "mean_step_ms",
                    "peak_kv_gib",
                    "busy_ms",
                    "completed_output_tokens",
                )
            )
            if maximum_difference > 1e-9:
                raise RuntimeError(
                    "v37 equivalence failed: "
                    f"{policy_name=} {key=} "
                    f"{maximum_difference=}"
                )
            equivalence.append(
                {
                    "offered_load": record[
                        "offered_load_vs_ragged_capacity"
                    ],
                    "policy": policy_name,
                    "service_replicate": key[0],
                    "seed": key[1],
                    "maximum_absolute_difference": (
                        maximum_difference
                    ),
                }
            )

    result = {
        "suite": (
            "decode_admission_fairness_simulation"
        ),
        "model": (
            "The v37 measured-service mixed-length "
            "continuous-decode model with request-class "
            "fairness metrics and an additional "
            "completion-aware admission proxy."
        ),
        "service_models": service_models,
        "workload": WORKLOADS["mixed_chat"],
        "policies": POLICIES,
        "simulation": {
            "num_requests": args.num_requests,
            "warmup_requests": args.warmup_requests,
            "tail_guard_requests": (
                args.tail_guard_requests
            ),
            "capacity_requests": (
                args.capacity_requests
            ),
            "seeds": seeds,
            "offered_loads_vs_ragged_capacity": (
                offered_loads
            ),
            "max_batch": MAX_BATCH,
            "kv_budget_gib": args.kv_budget_gib,
            "kv_budget_tokens": kv_budget_tokens,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "ttft_target_ms": (
                args.ttft_target_ms
            ),
            "completion_target_ms": (
                args.completion_target_ms
            ),
        },
        "capacities": capacities,
        "validation": {
            "max_attention_fit_relative_error": (
                max_fit_error
            ),
            "max_allowed_fit_relative_error": (
                args.max_fit_relative_error
            ),
            "scenario_count": scenario_count,
            "v37_equivalence_checks": (
                equivalence
            ),
            "v37_maximum_absolute_difference": (
                max(
                    item[
                        "maximum_absolute_difference"
                    ]
                    for item in equivalence
                )
            ),
            "accounting": (
                "Every scenario asserts exact request "
                "classification and generated tokens, "
                "finite completion only for accepted "
                "requests, max batch 64, KV within "
                "budget, and nonempty prompt/output "
                "fairness classes."
            ),
        },
        "records": records,
        "limitations": [
            "The A100 attention and MLP service points are measured, but arrivals, request classes, scheduler behavior, and admission outcomes are simulated.",
            "TTFT starts at the decode scheduler and excludes prefill, prefix reuse, ingress, tokenization, and sampling.",
            "The dual guard predicts completion by holding the current active contexts fixed and charging the arriving request's mean generation context. It is a policy proxy, not a calibrated end-to-end latency predictor.",
            "Jain indices weight the nine prompt/output classes equally, regardless of offered traffic share. Aggregate acceptance and token fractions remain authoritative for capacity.",
            "The workload has only three prompt lengths and three output lengths. Production tenants, priorities, cancellation, and burst correlations are not modeled.",
            "Full prompt plus output KV is reserved at activation. There is no preemption, swap, or incremental page allocation.",
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
        default="0.8,1.0,1.1,1.25",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=3_000,
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--tail-guard-requests",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--capacity-requests",
        type=int,
        default=2_500,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=37_000,
    )
    parser.add_argument(
        "--kv-budget-gib",
        type=float,
        default=32.0,
    )
    parser.add_argument(
        "--ttft-target-ms",
        type=float,
        default=250.0,
    )
    parser.add_argument(
        "--completion-target-ms",
        type=float,
        default=10_000.0,
    )
    parser.add_argument(
        "--max-fit-relative-error",
        type=float,
        default=0.15,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
