from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    LAYERS,
    MAX_BATCH,
    WORKLOADS,
    aggregate_metrics,
    decode_step_ms,
    load_service_models,
    make_arrivals,
    make_workload,
    mlp_ms,
    percentile,
    sha256,
)


BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
CONTEXT_BUCKETS = {
    "two": (2048, 8192),
    "four": (1024, 2048, 4096, 8192),
    "eight": (
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
    ),
}
STRATEGIES = {
    "ragged_dynamic": {
        "mode": "ragged",
        "split_context": False,
        "context_buckets": None,
        "round_batch": False,
    },
    "global_graph_4ctx_pow2": {
        "mode": "static",
        "split_context": False,
        "context_buckets": "four",
        "round_batch": True,
    },
    "split_graph_2ctx_pow2": {
        "mode": "static",
        "split_context": True,
        "context_buckets": "two",
        "round_batch": True,
    },
    "split_graph_4ctx_pow2": {
        "mode": "static",
        "split_context": True,
        "context_buckets": "four",
        "round_batch": True,
    },
    "split_graph_8ctx_pow2": {
        "mode": "static",
        "split_context": True,
        "context_buckets": "eight",
        "round_batch": True,
    },
    "split_graph_4ctx_exact_batch": {
        "mode": "static",
        "split_context": True,
        "context_buckets": "four",
        "round_batch": False,
    },
}


def round_up(
    value: int,
    buckets: tuple[int, ...],
) -> int:
    for bucket in buckets:
        if value <= bucket:
            return bucket
    raise ValueError(
        f"{value} exceeds maximum bucket {buckets[-1]}"
    )


def load_graph_evidence(
    root: Path,
) -> dict[str, Any]:
    decode_dir = root / "results" / "raw" / "decode_kv"
    smoke_path = decode_dir / "decode_kv_smoke.json"
    full_paths = (
        decode_dir / "decode_kv_full.json",
        decode_dir / "decode_kv_full_rep2.json",
    )
    smoke = json.loads(
        smoke_path.read_text(encoding="utf-8")
    )
    smoke_records = {
        record["implementation"]: record
        for record in smoke["records"]
        if record.get("configuration")
        == "smoke_homogeneous"
        and record.get("supported")
    }
    tiny_direct = smoke_records[
        "flashinfer_paged_tensor_core"
    ]["median_ms"]
    tiny_graph = smoke_records[
        "flashinfer_tensor_core_cuda_graph"
    ]["median_ms"]

    saturated_direct = []
    saturated_graph = []
    sources = []
    for path in full_paths:
        run = json.loads(
            path.read_text(encoding="utf-8")
        )
        records = {
            record["implementation"]: record
            for record in run["records"]
            if record.get("configuration")
            == "gqa_b64_l2048"
            and record.get("supported")
        }
        saturated_direct.append(
            records[
                "flashinfer_paged_tensor_core"
            ]["median_ms"]
        )
        saturated_graph.append(
            records[
                "flashinfer_tensor_core_cuda_graph"
            ]["median_ms"]
        )
        sources.append(
            {
                "source": path.name,
                "sha256": sha256(path),
            }
        )

    return {
        "tiny_b2_c257": {
            "source": smoke_path.name,
            "sha256": sha256(smoke_path),
            "direct_ms": tiny_direct,
            "graph_ms": tiny_graph,
            "speedup": tiny_direct / tiny_graph,
        },
        "saturated_b64_c2048": {
            "sources": sources,
            "direct_replicates_ms": saturated_direct,
            "graph_replicates_ms": saturated_graph,
            "median_direct_ms": statistics.median(
                saturated_direct
            ),
            "median_graph_ms": statistics.median(
                saturated_graph
            ),
            "speedup": (
                statistics.median(saturated_direct)
                / statistics.median(saturated_graph)
            ),
        },
        "model_policy": (
            "The static-shape simulation receives no graph-latency "
            "credit. It conservatively measures padding and extra "
            "group launches; the measured tiny and saturated graph "
            "speedups are reported as external boundaries."
        ),
    }


def static_group_cost(
    model: dict[str, Any],
    service_models: dict[str, Any],
    *,
    actual_batch: int,
    rounded_batch: int,
    context_bucket: int,
    maximum_graph_credit_ms: float,
    graph_latency_floor_ms: float,
) -> dict[str, float]:
    coefficients = model["coefficients"]
    attention_ms = (
        coefficients["intercept_ms"]
        + coefficients["batch_ms"] * rounded_batch
        + coefficients["context_token_ms"]
        * rounded_batch
        * context_bucket
    )
    if attention_ms <= 0.0:
        raise RuntimeError(
            "Non-positive static attention prediction: "
            f"{attention_ms=}"
        )
    if rounded_batch < actual_batch:
        raise RuntimeError("Rounded batch shrank")
    attention_credit_ms = min(
        maximum_graph_credit_ms,
        max(
            0.0,
            attention_ms - graph_latency_floor_ms,
        ),
    )
    return {
        "step_ms": LAYERS
        * (
            attention_ms
            + mlp_ms(
                service_models,
                rounded_batch,
            )
        ),
        "maximum_graph_credit_ms": (
            LAYERS * attention_credit_ms
        ),
    }


def step_cost(
    model: dict[str, Any],
    service_models: dict[str, Any],
    graph_evidence: dict[str, Any],
    contexts: np.ndarray,
    *,
    strategy_name: str,
) -> dict[str, Any]:
    if contexts.size < 1:
        raise ValueError("A decode step requires active requests")
    strategy = STRATEGIES[strategy_name]
    actual_context_tokens = int(np.sum(contexts))
    if strategy["mode"] == "ragged":
        return {
            "step_ms": decode_step_ms(
                model,
                service_models,
                contexts,
                kernel_mode="ragged",
            ),
            "groups": 1,
            "effective_context_tokens": (
                actual_context_tokens
            ),
            "rounded_batch_slots": int(
                contexts.size
            ),
            "graph_shapes": (),
            "maximum_graph_credit_ms": 0.0,
        }

    bucket_name = strategy["context_buckets"]
    context_buckets = CONTEXT_BUCKETS[bucket_name]
    grouped: dict[int, list[int]] = defaultdict(list)
    if strategy["split_context"]:
        for context in contexts:
            bucket = round_up(
                int(context),
                context_buckets,
            )
            grouped[bucket].append(int(context))
    else:
        bucket = round_up(
            int(np.max(contexts)),
            context_buckets,
        )
        grouped[bucket].extend(
            int(context)
            for context in contexts
        )

    total_ms = 0.0
    effective_context_tokens = 0
    rounded_batch_slots = 0
    graph_shapes = []
    maximum_graph_credit_ms = 0.0
    tiny_evidence = graph_evidence[
        "tiny_b2_c257"
    ]
    maximum_attention_credit_ms = (
        tiny_evidence["direct_ms"]
        - tiny_evidence["graph_ms"]
    )
    for context_bucket, values in sorted(
        grouped.items()
    ):
        actual_batch = len(values)
        rounded_batch = (
            round_up(
                actual_batch,
                BATCH_BUCKETS,
            )
            if strategy["round_batch"]
            else actual_batch
        )
        group_cost = static_group_cost(
            model,
            service_models,
            actual_batch=actual_batch,
            rounded_batch=rounded_batch,
            context_bucket=context_bucket,
            maximum_graph_credit_ms=(
                maximum_attention_credit_ms
            ),
            graph_latency_floor_ms=(
                tiny_evidence["graph_ms"]
            ),
        )
        total_ms += group_cost["step_ms"]
        maximum_graph_credit_ms += group_cost[
            "maximum_graph_credit_ms"
        ]
        effective_context_tokens += (
            rounded_batch * context_bucket
        )
        rounded_batch_slots += rounded_batch
        graph_shapes.append(
            (
                context_bucket,
                rounded_batch,
            )
        )
    return {
        "step_ms": total_ms,
        "groups": len(grouped),
        "effective_context_tokens": (
            effective_context_tokens
        ),
        "rounded_batch_slots": (
            rounded_batch_slots
        ),
        "graph_shapes": tuple(graph_shapes),
        "maximum_graph_credit_ms": (
            maximum_graph_credit_ms
        ),
    }


def simulate(
    *,
    arrivals: np.ndarray,
    workload: dict[str, np.ndarray],
    model: dict[str, Any],
    service_models: dict[str, Any],
    graph_evidence: dict[str, Any],
    strategy_name: str,
    kv_budget_tokens: int,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    prompt_tokens = workload["prompt_tokens"]
    output_tokens = workload["output_tokens"]
    reservation_tokens = workload["reservation_tokens"]
    num_requests = arrivals.size
    if not (
        prompt_tokens.size
        == output_tokens.size
        == reservation_tokens.size
        == num_requests
    ):
        raise ValueError("Workload length mismatch")

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
    pending: deque[int] = deque()
    active: list[int] = []
    next_arrival = 0
    now_ms = 0.0
    busy_ms = 0.0
    reserved_tokens = 0
    peak_reserved_tokens = 0
    step_batches: list[int] = []
    step_groups: list[int] = []
    step_ms_values: list[float] = []
    actual_context_work = 0
    effective_context_work = 0
    actual_batch_slots = 0
    rounded_batch_slots = 0
    graph_shapes: set[tuple[int, int]] = set()
    maximum_graph_credit_ms = 0.0

    while (
        next_arrival < num_requests
        or pending
        or active
    ):
        while (
            next_arrival < num_requests
            and arrivals[next_arrival] <= now_ms
        ):
            pending.append(next_arrival)
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
        cost = step_cost(
            model,
            service_models,
            graph_evidence,
            contexts,
            strategy_name=strategy_name,
        )
        ended_ms = now_ms + cost["step_ms"]
        step_batches.append(len(active))
        step_groups.append(cost["groups"])
        step_ms_values.append(cost["step_ms"])
        actual_context_work += int(
            np.sum(contexts)
        )
        effective_context_work += cost[
            "effective_context_tokens"
        ]
        actual_batch_slots += int(contexts.size)
        rounded_batch_slots += cost[
            "rounded_batch_slots"
        ]
        graph_shapes.update(cost["graph_shapes"])
        maximum_graph_credit_ms += cost[
            "maximum_graph_credit_ms"
        ]

        survivors = []
        for request in active:
            if generated[request] == 0:
                first_token[request] = ended_ms
            generated[request] += 1
            if generated[request] == output_tokens[request]:
                completion[request] = ended_ms
                reserved_tokens -= int(
                    reservation_tokens[request]
                )
            else:
                survivors.append(request)
        active = survivors
        now_ms = ended_ms
        busy_ms += cost["step_ms"]

    if not np.array_equal(
        generated,
        output_tokens,
    ):
        raise RuntimeError(
            "Generated-token accounting failed"
        )
    if not np.all(np.isfinite(completion)):
        raise RuntimeError(
            "A request did not complete"
        )
    if peak_reserved_tokens > kv_budget_tokens:
        raise RuntimeError("KV budget exceeded")
    if reserved_tokens != 0:
        raise RuntimeError("KV reservation leak")
    if max(step_batches) > MAX_BATCH:
        raise RuntimeError("Batch-size invariant failed")

    selected = np.arange(
        analysis_start,
        analysis_end,
    )
    selected_arrivals = arrivals[selected]
    selected_first = first_token[selected]
    selected_completion = completion[selected]
    ttft = selected_first - selected_arrivals
    e2e = selected_completion - selected_arrivals
    queue_wait = (
        active_start[selected]
        - selected_arrivals
    )
    tpot = (
        selected_completion
        - selected_first
    ) / (
        output_tokens[selected] - 1
    )
    elapsed_ms = (
        float(np.max(selected_completion))
        - float(arrivals[analysis_start])
    )
    selected_outputs = int(
        np.sum(output_tokens[selected])
    )
    selected_count = analysis_end - analysis_start
    return {
        "throughput_requests_s": (
            selected_count
            * 1_000.0
            / elapsed_ms
        ),
        "throughput_output_tokens_s": (
            selected_outputs
            * 1_000.0
            / elapsed_ms
        ),
        "mean_queue_ms": float(
            np.mean(queue_wait)
        ),
        "p95_queue_ms": percentile(
            queue_wait,
            95,
        ),
        "p99_queue_ms": percentile(
            queue_wait,
            99,
        ),
        "p50_ttft_ms": percentile(ttft, 50),
        "p95_ttft_ms": percentile(ttft, 95),
        "p99_ttft_ms": percentile(ttft, 99),
        "p50_tpot_ms": percentile(tpot, 50),
        "p95_tpot_ms": percentile(tpot, 95),
        "p99_tpot_ms": percentile(tpot, 99),
        "p50_e2e_ms": percentile(e2e, 50),
        "p95_e2e_ms": percentile(e2e, 95),
        "p99_e2e_ms": percentile(e2e, 99),
        "mean_batch_size": float(
            np.mean(step_batches)
        ),
        "mean_groups_per_step": float(
            np.mean(step_groups)
        ),
        "p95_groups_per_step": percentile(
            np.asarray(step_groups),
            95,
        ),
        "mean_graph_replays_per_step": (
            float(np.mean(step_groups)) * LAYERS
            if STRATEGIES[strategy_name]["mode"]
            == "static"
            else 0.0
        ),
        "graph_shape_count": float(
            len(graph_shapes)
        ),
        "mean_step_ms": float(
            np.mean(step_ms_values)
        ),
        "p95_step_ms": percentile(
            np.asarray(step_ms_values),
            95,
        ),
        "context_work_inflation": (
            effective_context_work
            / actual_context_work
        ),
        "batch_slot_inflation": (
            rounded_batch_slots
            / actual_batch_slots
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
        "maximum_graph_credit_ms": (
            maximum_graph_credit_ms
        ),
        "optimistic_graph_busy_ms": max(
            busy_ms - maximum_graph_credit_ms,
            1e-12,
        ),
        "completed_output_tokens": float(
            np.sum(output_tokens)
        ),
    }


def estimate_capacity(
    *,
    workload_name: str,
    strategy_name: str,
    num_requests: int,
    seed: int,
    model: dict[str, Any],
    service_models: dict[str, Any],
    graph_evidence: dict[str, Any],
    kv_budget_tokens: int,
) -> dict[str, float]:
    workload = make_workload(
        name=workload_name,
        num_requests=num_requests,
        seed=seed,
    )
    metrics = simulate(
        arrivals=np.zeros(
            num_requests,
            dtype=np.float64,
        ),
        workload=workload,
        model=model,
        service_models=service_models,
        graph_evidence=graph_evidence,
        strategy_name=strategy_name,
        kv_budget_tokens=kv_budget_tokens,
        analysis_start=0,
        analysis_end=num_requests,
    )
    return {
        "requests_per_second": (
            num_requests
            * 1_000.0
            / metrics["busy_ms"]
        ),
        "output_tokens_per_second": (
            metrics["completed_output_tokens"]
            * 1_000.0
            / metrics["busy_ms"]
        ),
        "optimistic_graph_output_tokens_per_second": (
            metrics["completed_output_tokens"]
            * 1_000.0
            / metrics["optimistic_graph_busy_ms"]
        ),
        "maximum_graph_credit_fraction": (
            metrics["maximum_graph_credit_ms"]
            / metrics["busy_ms"]
        ),
        "context_work_inflation": metrics[
            "context_work_inflation"
        ],
        "batch_slot_inflation": metrics[
            "batch_slot_inflation"
        ],
        "mean_groups_per_step": metrics[
            "mean_groups_per_step"
        ],
        "graph_shape_count": metrics[
            "graph_shape_count"
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    service_models = load_service_models(args.root)
    graph_evidence = load_graph_evidence(args.root)
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
    if (
        graph_evidence["tiny_b2_c257"]["speedup"]
        < 1.5
    ):
        raise RuntimeError(
            "Tiny graph evidence is unexpectedly weak"
        )
    saturated_speedup = graph_evidence[
        "saturated_b64_c2048"
    ]["speedup"]
    if not 0.95 <= saturated_speedup <= 1.05:
        raise RuntimeError(
            "Saturated graph evidence changed materially"
        )

    max_context = max(
        max(specification["prompt_tokens"])
        + max(specification["output_tokens"])
        for specification in WORKLOADS.values()
    )
    if any(
        buckets[-1] < max_context
        for buckets in CONTEXT_BUCKETS.values()
    ):
        raise RuntimeError(
            "A context bucket set does not cover the workload"
        )
    equal_context_checks = []
    for model in service_models[
        "attention_replicates"
    ]:
        for batch_size in (1, 8, 16, 32, 64):
            contexts = np.full(
                batch_size,
                2048,
                dtype=np.int64,
            )
            reference = step_cost(
                model,
                service_models,
                graph_evidence,
                contexts,
                strategy_name="ragged_dynamic",
            )["step_ms"]
            for strategy_name in STRATEGIES:
                actual = step_cost(
                    model,
                    service_models,
                    graph_evidence,
                    contexts,
                    strategy_name=strategy_name,
                )["step_ms"]
                if actual != reference:
                    raise RuntimeError(
                        "Equal-context bucket equivalence "
                        f"failed for {strategy_name}, "
                        f"{batch_size=}: {actual=} "
                        f"{reference=}"
                    )
            equal_context_checks.append(
                {
                    "source": model["source"],
                    "batch_size": batch_size,
                    "context": 2048,
                    "step_ms": reference,
                }
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

    capacities: dict[str, dict[str, Any]] = {}
    for workload_name in WORKLOADS:
        capacities[workload_name] = {}
        for replicate_index, model in enumerate(
            service_models["attention_replicates"]
        ):
            capacities[workload_name][
                model["source"]
            ] = {}
            for strategy_name in STRATEGIES:
                capacities[workload_name][
                    model["source"]
                ][strategy_name] = estimate_capacity(
                    workload_name=workload_name,
                    strategy_name=strategy_name,
                    num_requests=args.capacity_requests,
                    seed=(
                        args.seed
                        + 500_000
                        + replicate_index * 10_000
                        + (
                            0
                            if workload_name
                            == "homogeneous"
                            else 1_000
                        )
                    ),
                    model=model,
                    service_models=service_models,
                    graph_evidence=graph_evidence,
                    kv_budget_tokens=kv_budget_tokens,
                )

    records = []
    scenario_count = 0
    for workload_name in WORKLOADS:
        for offered_load in offered_loads:
            for strategy_name in STRATEGIES:
                scenarios = []
                for replicate_index, model in enumerate(
                    service_models[
                        "attention_replicates"
                    ]
                ):
                    ragged_capacity = capacities[
                        workload_name
                    ][model["source"]][
                        "ragged_dynamic"
                    ]["requests_per_second"]
                    request_rate = (
                        ragged_capacity
                        * offered_load
                    )
                    for seed in seeds:
                        workload_seed = (
                            seed
                            + replicate_index * 100_000
                            + (
                                0
                                if workload_name
                                == "homogeneous"
                                else 10_000
                            )
                        )
                        workload = make_workload(
                            name=workload_name,
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
                            graph_evidence=graph_evidence,
                            strategy_name=strategy_name,
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
                        "workload": workload_name,
                        "offered_load_vs_ragged_capacity": (
                            offered_load
                        ),
                        "strategy": strategy_name,
                        "configuration": STRATEGIES[
                            strategy_name
                        ],
                        "metrics": aggregate_metrics(
                            scenarios
                        ),
                        "scenarios": scenarios,
                    }
                )

    result = {
        "suite": (
            "decode_static_graph_bucketing_simulation"
        ),
        "model": (
            "The v37 measured-service 32-layer decode "
            "model with conservative static context and "
            "batch shape bucketing. Static groups execute "
            "sequentially and receive no assumed CUDA "
            "Graph latency credit."
        ),
        "service_models": service_models,
        "graph_evidence": graph_evidence,
        "workloads": WORKLOADS,
        "strategies": STRATEGIES,
        "bucket_sets": {
            "batch": BATCH_BUCKETS,
            "context": CONTEXT_BUCKETS,
        },
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
            "layers": LAYERS,
            "kv_budget_gib": args.kv_budget_gib,
            "kv_budget_tokens": kv_budget_tokens,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        },
        "capacities": capacities,
        "validation": {
            "max_attention_fit_relative_error": (
                max_fit_error
            ),
            "max_allowed_fit_relative_error": (
                args.max_fit_relative_error
            ),
            "maximum_workload_context": max_context,
            "equal_context_checks": (
                equal_context_checks
            ),
            "scenario_count": scenario_count,
            "accounting": (
                "Every scenario asserts exact generated "
                "tokens, finite completion, max batch 64, "
                "and KV reservations within budget. "
                "Every static group rounds upward only."
            ),
        },
        "records": records,
        "limitations": [
            "The attention and MLP points are real A100 measurements; request arrivals, lengths, 32-layer composition, and scheduler behavior are simulated.",
            "Static graph strategies receive no latency credit from replay. The measured 2.02x tiny-shape and 1.001x saturated graph speedups are reported separately, so the static results isolate padding and launch multiplicity conservatively.",
            "Each context bucket executes sequentially at every layer. A fused multi-bucket scheduler, concurrent streams, or a megakernel could reduce this launch cost.",
            "The affine attention service fit has at most 12.4% calibration error on the measured points. It does not model shape-specific kernel switches outside that set.",
            "The MLP curve extrapolates the measured batch-8-to-32 slope through batch 64. Static batch padding charges the extrapolated rounded batch.",
            "TTFT starts at the decode scheduler and excludes prefill, prefix reuse, ingress, tokenization, and sampling.",
            "Full prompt plus output KV is reserved at activation. There is no preemption, swap, cancellation, or incremental page growth.",
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
        default="0.25,0.5,0.8,1.0",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=2_500,
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--tail-guard-requests",
        type=int,
        default=125,
    )
    parser.add_argument(
        "--capacity-requests",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=38_000,
    )
    parser.add_argument(
        "--kv-budget-gib",
        type=float,
        default=32.0,
    )
    parser.add_argument(
        "--max-fit-relative-error",
        type=float,
        default=0.15,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
