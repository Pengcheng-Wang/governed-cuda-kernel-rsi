from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import statistics
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


MAX_BATCH = 64
LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2
KV_BYTES_PER_TOKEN = (
    2
    * NUM_KV_HEADS
    * HEAD_DIM
    * DTYPE_BYTES
    * LAYERS
)
ATTENTION_CONFIGURATIONS = (
    "gqa_b1_l2048",
    "gqa_b16_l2048",
    "gqa_b64_l2048",
    "gqa_b16_l8192",
    "gqa_b64_l8192",
    "gqa_b64_heterogeneous",
)
POLICIES = {
    "ragged_fcfs": {
        "kernel_mode": "ragged",
        "ttft_guard": False,
    },
    "padded_fcfs": {
        "kernel_mode": "padded",
        "ttft_guard": False,
    },
    "ragged_ttft_guard": {
        "kernel_mode": "ragged",
        "ttft_guard": True,
    },
}
WORKLOADS = {
    "homogeneous": {
        "prompt_tokens": (2048,),
        "prompt_probabilities": (1.0,),
        "output_tokens": (128,),
        "output_probabilities": (1.0,),
    },
    "mixed_chat": {
        "prompt_tokens": (512, 2048, 6144),
        "prompt_probabilities": (0.55, 0.30, 0.15),
        "output_tokens": (32, 128, 512),
        "output_probabilities": (0.55, 0.30, 0.15),
    },
}


def percentile(
    values: np.ndarray,
    q: float,
) -> float:
    return float(np.percentile(values, q))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_metrics(
    metrics: list[dict[str, float]],
) -> dict[str, Any]:
    keys = tuple(
        key
        for key, value in metrics[0].items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    )
    return {
        key: {
            "median": statistics.median(
                item[key]
                for item in metrics
            ),
            "minimum": min(
                item[key]
                for item in metrics
            ),
            "maximum": max(
                item[key]
                for item in metrics
            ),
        }
        for key in keys
    }


def load_service_models(
    root: Path,
) -> dict[str, Any]:
    decode_dir = root / "results" / "raw" / "decode_kv"
    decode_paths = (
        decode_dir / "decode_kv_full.json",
        decode_dir / "decode_kv_full_rep2.json",
    )
    attention_replicates = []
    for path in decode_paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        points = []
        for record in run["records"]:
            if (
                record.get("experiment") == "batch_paged_decode"
                and record.get("implementation")
                == "flashinfer_paged_tensor_core"
                and record.get("configuration")
                in ATTENTION_CONFIGURATIONS
                and record.get("supported")
            ):
                points.append(
                    {
                        "configuration": record["configuration"],
                        "batch_size": record["batch_size"],
                        "context_sum": (
                            record["context_mean"]
                            * record["batch_size"]
                        ),
                        "context_min": record["context_min"],
                        "context_mean": record["context_mean"],
                        "context_max": record["context_max"],
                        "measured_ms": record["median_ms"],
                    }
                )
        if len(points) != len(ATTENTION_CONFIGURATIONS):
            raise ValueError(
                f"Incomplete attention calibration set: {path}"
            )
        design = np.array(
            [
                [
                    1.0,
                    point["batch_size"],
                    point["context_sum"],
                ]
                for point in points
            ],
            dtype=np.float64,
        )
        measured = np.array(
            [
                point["measured_ms"]
                for point in points
            ],
            dtype=np.float64,
        )
        coefficients = np.linalg.lstsq(
            design,
            measured,
            rcond=None,
        )[0]
        predicted = design @ coefficients
        relative_errors = np.abs(
            predicted - measured
        ) / measured
        for point, prediction, error in zip(
            points,
            predicted,
            relative_errors,
        ):
            point["predicted_ms"] = float(prediction)
            point["relative_error"] = float(error)
        attention_replicates.append(
            {
                "source": path.name,
                "sha256": sha256(path),
                "coefficients": {
                    "intercept_ms": float(coefficients[0]),
                    "batch_ms": float(coefficients[1]),
                    "context_token_ms": float(
                        coefficients[2]
                    ),
                },
                "max_relative_error": float(
                    np.max(relative_errors)
                ),
                "points": points,
            }
        )

    mlp_path = (
        root
        / "results"
        / "raw"
        / "full"
        / "decode_graphs_full.json"
    )
    mlp_run = json.loads(
        mlp_path.read_text(encoding="utf-8")
    )
    mlp_points = sorted(
        (
            {
                "batch_size": record["shape"][0],
                "measured_ms": record["median_ms"],
            }
            for record in mlp_run["records"]
            if record["implementation"]
            == "explicit_cuda_graph"
        ),
        key=lambda item: item["batch_size"],
    )
    if [
        point["batch_size"]
        for point in mlp_points
    ] != [1, 8, 32]:
        raise ValueError("Unexpected decode MLP calibration points")

    return {
        "attention_replicates": attention_replicates,
        "mlp": {
            "source": mlp_path.name,
            "sha256": sha256(mlp_path),
            "points": mlp_points,
            "extrapolation": (
                "Linear interpolation through batch 1/8/32; "
                "the 8-to-32 slope extends to batch 64."
            ),
        },
        "architecture": {
            "layers": LAYERS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "dtype_bytes": DTYPE_BYTES,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        },
    }


def mlp_ms(
    service_models: dict[str, Any],
    batch_size: int,
) -> float:
    points = service_models["mlp"]["points"]
    batches = np.array(
        [
            point["batch_size"]
            for point in points
        ],
        dtype=np.float64,
    )
    values = np.array(
        [
            point["measured_ms"]
            for point in points
        ],
        dtype=np.float64,
    )
    if batch_size <= batches[-1]:
        return float(
            np.interp(
                batch_size,
                batches,
                values,
            )
        )
    slope = (
        values[-1] - values[-2]
    ) / (
        batches[-1] - batches[-2]
    )
    return float(
        values[-1]
        + slope * (batch_size - batches[-1])
    )


def decode_step_ms(
    model: dict[str, Any],
    service_models: dict[str, Any],
    contexts: np.ndarray,
    *,
    kernel_mode: str,
) -> float:
    if contexts.size < 1:
        raise ValueError("A decode step requires active requests")
    if kernel_mode == "ragged":
        effective_context_tokens = float(
            np.sum(contexts)
        )
    elif kernel_mode == "padded":
        effective_context_tokens = float(
            contexts.size * np.max(contexts)
        )
    else:
        raise ValueError(kernel_mode)
    coefficients = model["coefficients"]
    attention_ms = (
        coefficients["intercept_ms"]
        + coefficients["batch_ms"] * contexts.size
        + coefficients["context_token_ms"]
        * effective_context_tokens
    )
    if attention_ms <= 0.0:
        raise RuntimeError(
            f"Non-positive attention prediction: {attention_ms}"
        )
    return LAYERS * (
        attention_ms
        + mlp_ms(
            service_models,
            int(contexts.size),
        )
    )


def make_workload(
    *,
    name: str,
    num_requests: int,
    seed: int,
) -> dict[str, np.ndarray]:
    specification = WORKLOADS[name]
    rng = np.random.default_rng(seed)
    prompt_tokens = rng.choice(
        specification["prompt_tokens"],
        size=num_requests,
        p=specification["prompt_probabilities"],
    ).astype(np.int64)
    output_tokens = rng.choice(
        specification["output_tokens"],
        size=num_requests,
        p=specification["output_probabilities"],
    ).astype(np.int64)
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "reservation_tokens": (
            prompt_tokens + output_tokens
        ),
    }


def make_arrivals(
    *,
    num_requests: int,
    requests_per_second: float,
    seed: int,
) -> np.ndarray:
    if requests_per_second <= 0.0:
        raise ValueError(requests_per_second)
    rng = np.random.default_rng(seed)
    intervals = rng.exponential(
        1_000.0 / requests_per_second,
        size=num_requests,
    )
    intervals[0] = 0.0
    return np.cumsum(intervals)


def predicted_ttft_ms(
    *,
    request: int,
    now_ms: float,
    arrivals: np.ndarray,
    active: list[int],
    pending: deque[int],
    generated: np.ndarray,
    prompt_tokens: np.ndarray,
    output_tokens: np.ndarray,
    reservation_tokens: np.ndarray,
    reserved_tokens: int,
    kv_budget_tokens: int,
    model: dict[str, Any],
    service_models: dict[str, Any],
) -> float:
    elapsed_ms = max(
        0.0,
        now_ms - float(arrivals[request]),
    )
    candidate_fits_now = (
        len(active) < MAX_BATCH
        and reserved_tokens
        + int(reservation_tokens[request])
        <= kv_budget_tokens
        and not pending
    )
    current_contexts = np.array(
        [
            int(prompt_tokens[item])
            + int(generated[item])
            for item in active
        ],
        dtype=np.int64,
    )
    if candidate_fits_now:
        candidate_contexts = np.append(
            current_contexts,
            int(prompt_tokens[request]),
        )
        return elapsed_ms + decode_step_ms(
            model,
            service_models,
            candidate_contexts,
            kernel_mode="ragged",
        )

    release_steps = [
        int(output_tokens[item] - generated[item])
        for item in active
    ]
    free_slots = MAX_BATCH - len(active)
    release_steps.extend(
        0
        for _ in range(free_slots)
    )
    if not release_steps:
        release_steps.append(0)
    heapq.heapify(release_steps)
    for queued in pending:
        release = heapq.heappop(release_steps)
        heapq.heappush(
            release_steps,
            release + int(output_tokens[queued]),
        )
    start_steps = heapq.heappop(release_steps)
    representative_contexts = (
        current_contexts
        if current_contexts.size
        else np.array(
            [int(prompt_tokens[request])],
            dtype=np.int64,
        )
    )
    step_ms = decode_step_ms(
        model,
        service_models,
        representative_contexts,
        kernel_mode="ragged",
    )
    return elapsed_ms + (
        start_steps + 1
    ) * step_ms


def simulate(
    *,
    arrivals: np.ndarray,
    workload: dict[str, np.ndarray],
    model: dict[str, Any],
    service_models: dict[str, Any],
    policy_name: str,
    kv_budget_tokens: int,
    ttft_target_ms: float,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    policy = POLICIES[policy_name]
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

    pending: deque[int] = deque()
    active: list[int] = []
    next_arrival = 0
    now_ms = 0.0
    busy_ms = 0.0
    reserved_tokens = 0
    peak_reserved_tokens = 0
    step_batches: list[int] = []
    step_ms_values: list[float] = []
    actual_context_work = 0
    padded_context_work = 0

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
                predicted = predicted_ttft_ms(
                    request=request,
                    now_ms=now_ms,
                    arrivals=arrivals,
                    active=active,
                    pending=pending,
                    generated=generated,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    reservation_tokens=reservation_tokens,
                    reserved_tokens=reserved_tokens,
                    kv_budget_tokens=kv_budget_tokens,
                    model=model,
                    service_models=service_models,
                )
                allow = predicted <= ttft_target_ms
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
            kernel_mode=policy["kernel_mode"],
        )
        ended_ms = now_ms + step_ms
        step_batches.append(len(active))
        step_ms_values.append(step_ms)
        actual_context_work += int(np.sum(contexts))
        padded_context_work += int(
            contexts.size * np.max(contexts)
        )

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
        busy_ms += step_ms

    if not np.all(accepted | rejected):
        raise RuntimeError("Not every request was classified")
    if np.any(accepted & rejected):
        raise RuntimeError("Request both accepted and rejected")
    if not np.array_equal(
        generated[accepted],
        output_tokens[accepted],
    ):
        raise RuntimeError("Accepted token accounting failed")
    if np.any(np.isfinite(completion[rejected])):
        raise RuntimeError("Rejected request completed")
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
    selected_accepted = selected[accepted[selected]]
    selected_rejected = selected[rejected[selected]]
    if selected_accepted.size < 1:
        raise RuntimeError("No accepted requests in analysis window")
    selected_arrivals = arrivals[selected_accepted]
    selected_first = first_token[selected_accepted]
    selected_completion = completion[selected_accepted]
    ttft = selected_first - selected_arrivals
    e2e = selected_completion - selected_arrivals
    queue_wait = (
        active_start[selected_accepted]
        - selected_arrivals
    )
    multi_token = (
        output_tokens[selected_accepted] > 1
    )
    tpot = (
        selected_completion[multi_token]
        - selected_first[multi_token]
    ) / (
        output_tokens[selected_accepted][multi_token]
        - 1
    )
    elapsed_ms = (
        float(np.max(selected_completion))
        - float(arrivals[analysis_start])
    )
    accepted_outputs = int(
        np.sum(output_tokens[selected_accepted])
    )
    analysis_count = analysis_end - analysis_start
    return {
        "accepted_fraction": (
            selected_accepted.size / analysis_count
        ),
        "rejected_fraction": (
            selected_rejected.size / analysis_count
        ),
        "throughput_requests_s": (
            selected_accepted.size
            * 1_000.0
            / elapsed_ms
        ),
        "throughput_output_tokens_s": (
            accepted_outputs
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
        "mean_ttft_ms": float(np.mean(ttft)),
        "p50_ttft_ms": percentile(ttft, 50),
        "p95_ttft_ms": percentile(ttft, 95),
        "p99_ttft_ms": percentile(ttft, 99),
        "ttft_slo_attainment": float(
            np.mean(ttft <= ttft_target_ms)
        ),
        "mean_tpot_ms": float(np.mean(tpot)),
        "p50_tpot_ms": percentile(tpot, 50),
        "p95_tpot_ms": percentile(tpot, 95),
        "p99_tpot_ms": percentile(tpot, 99),
        "p50_e2e_ms": percentile(e2e, 50),
        "p95_e2e_ms": percentile(e2e, 95),
        "p99_e2e_ms": percentile(e2e, 99),
        "mean_batch_size": float(
            np.mean(step_batches)
        ),
        "p50_batch_size": percentile(
            np.asarray(step_batches),
            50,
        ),
        "p95_batch_size": percentile(
            np.asarray(step_batches),
            95,
        ),
        "mean_step_ms": float(
            np.mean(step_ms_values)
        ),
        "p95_step_ms": percentile(
            np.asarray(step_ms_values),
            95,
        ),
        "padding_inflation": (
            padded_context_work
            / actual_context_work
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
    }


def estimate_capacity(
    *,
    workload_name: str,
    num_requests: int,
    seed: int,
    model: dict[str, Any],
    service_models: dict[str, Any],
    kv_budget_tokens: int,
    ttft_target_ms: float,
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
        policy_name="ragged_fcfs",
        kv_budget_tokens=kv_budget_tokens,
        ttft_target_ms=ttft_target_ms,
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
        "mean_output_tokens": float(
            np.mean(workload["output_tokens"])
        ),
        "mean_prompt_tokens": float(
            np.mean(workload["prompt_tokens"])
        ),
        "padding_inflation": metrics[
            "padding_inflation"
        ],
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

    equal_context_checks = []
    for model in service_models[
        "attention_replicates"
    ]:
        contexts = np.full(
            MAX_BATCH,
            2048,
            dtype=np.int64,
        )
        ragged = decode_step_ms(
            model,
            service_models,
            contexts,
            kernel_mode="ragged",
        )
        padded = decode_step_ms(
            model,
            service_models,
            contexts,
            kernel_mode="padded",
        )
        if ragged != padded:
            raise RuntimeError(
                "Equal-context service models diverged"
            )
        equal_context_checks.append(
            {
                "source": model["source"],
                "ragged_ms": ragged,
                "padded_ms": padded,
            }
        )

    kv_budget_tokens = int(
        args.kv_budget_gib
        * (1024**3)
        / KV_BYTES_PER_TOKEN
    )
    if max(
        max(specification["prompt_tokens"])
        + max(specification["output_tokens"])
        for specification in WORKLOADS.values()
    ) > kv_budget_tokens:
        raise ValueError(
            "One request exceeds the KV budget"
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
            ] = estimate_capacity(
                workload_name=workload_name,
                num_requests=args.capacity_requests,
                seed=(
                    args.seed
                    + 500_000
                    + replicate_index * 10_000
                    + (
                        0
                        if workload_name == "homogeneous"
                        else 1_000
                    )
                ),
                model=model,
                service_models=service_models,
                kv_budget_tokens=kv_budget_tokens,
                ttft_target_ms=args.ttft_target_ms,
            )

    records = []
    scenario_count = 0
    for workload_name in WORKLOADS:
        for offered_load in offered_loads:
            for policy_name in POLICIES:
                scenarios = []
                for replicate_index, model in enumerate(
                    service_models[
                        "attention_replicates"
                    ]
                ):
                    capacity = capacities[
                        workload_name
                    ][model["source"]]
                    request_rate = (
                        capacity["requests_per_second"]
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
                            policy_name=policy_name,
                            kv_budget_tokens=(
                                kv_budget_tokens
                            ),
                            ttft_target_ms=(
                                args.ttft_target_ms
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
                        "policy": policy_name,
                        "kernel_mode": POLICIES[
                            policy_name
                        ]["kernel_mode"],
                        "ttft_guard": POLICIES[
                            policy_name
                        ]["ttft_guard"],
                        "metrics": aggregate_metrics(
                            scenarios
                        ),
                        "scenarios": scenarios,
                    }
                )

    result = {
        "suite": (
            "decode_continuous_batching_simulation"
        ),
        "model": (
            "A 32-layer Llama-class decode-stage model. "
            "Each iteration combines a fitted FlashInfer "
            "paged-GQA attention latency with a measured "
            "CUDA-Graph MLP latency for every layer."
        ),
        "service_models": service_models,
        "workloads": WORKLOADS,
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
            "ttft_target_ms": (
                args.ttft_target_ms
            ),
            "policies": POLICIES,
        },
        "capacities": capacities,
        "validation": {
            "max_attention_fit_relative_error": (
                max_fit_error
            ),
            "max_allowed_fit_relative_error": (
                args.max_fit_relative_error
            ),
            "equal_context_checks": (
                equal_context_checks
            ),
            "scenario_count": scenario_count,
            "accounting": (
                "Every scenario asserts request "
                "classification, exact generated-token "
                "counts, finite completion only for "
                "accepted requests, max batch 64, and "
                "KV reservations within budget."
            ),
        },
        "records": records,
        "limitations": [
            "Attention and MLP service points are real A100 measurements, but the 32-layer composition, arrivals, request lengths, scheduler, and admission outcomes are simulated.",
            "TTFT starts when a request reaches the decode scheduler. It excludes network ingress, tokenization, prefill, prefix-cache lookup, and sampling.",
            "TPOT is the request-level mean gap after the first token. The model executes one token per active request per iteration and does not model speculative decoding.",
            "The attention fit uses measured batch-1/16/64, context-2048/8192, and heterogeneous batch-64 points. The MLP curve extrapolates the measured batch-8-to-32 slope through batch 64.",
            "The 32-layer model omits RMSNorm, residual, projection, sampling, CPU scheduling, tensor parallelism, and GPU jitter. It is a kernel-stage systems bound, not an end-to-end serving benchmark.",
            "The admission guard predicts slot release from remaining output lengths and current measured step cost. It is not a production scheduler or a guarantee for future arrivals.",
            "Full prompt plus requested output KV is reserved on activation. Production paged allocators grow blocks incrementally and may preempt or swap requests.",
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
        default="0.5,0.8,1.0,1.1,1.25",
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
        "--max-fit-relative-error",
        type=float,
        default=0.15,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
