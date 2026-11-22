from __future__ import annotations

import argparse
import json
import statistics
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    LAYERS,
    MAX_BATCH,
    aggregate_metrics,
    decode_step_ms,
    load_service_models,
    make_arrivals,
    make_workload,
    percentile,
    sha256,
)


HIDDEN = 4096
INTERMEDIATE = 11008
NUM_QO_HEADS = 32
MEASURED_ATTENTION_HEADS = 16
NUM_KV_HEADS = 8
KV_PROJECTION = (
    HIDDEN * NUM_KV_HEADS // NUM_QO_HEADS
)
GEMM_LAUNCHES_PER_LAYER = 7
POLICIES = {
    "monolithic_prefill_first": {
        "chunk_tokens": None,
        "order": "prefill_first",
        "decode_steps_per_prefill": 1,
    },
    "chunk_2048_prefill_first": {
        "chunk_tokens": 2048,
        "order": "prefill_first",
        "decode_steps_per_prefill": 1,
    },
    "chunk_1024_prefill_first": {
        "chunk_tokens": 1024,
        "order": "prefill_first",
        "decode_steps_per_prefill": 1,
    },
    "chunk_512_prefill_first": {
        "chunk_tokens": 512,
        "order": "prefill_first",
        "decode_steps_per_prefill": 1,
    },
    "chunk_512_decode_priority_2to1": {
        "chunk_tokens": 512,
        "order": "decode_first",
        "decode_steps_per_prefill": 2,
    },
    "chunk_256_decode_priority_4to1": {
        "chunk_tokens": 256,
        "order": "decode_first",
        "decode_steps_per_prefill": 4,
    },
}
PREFIX_MODES = {
    "none": {
        "request_hit_probability": 0.0,
        "cached_prompt_fraction": 0.0,
    },
    "half_requests_75pct_cached": {
        "request_hit_probability": 0.5,
        "cached_prompt_fraction": 0.75,
    },
}


def load_prefill_service_model(
    root: Path,
) -> dict[str, Any]:
    attention_path = (
        root
        / "results"
        / "raw"
        / "full"
        / "attention_full.json"
    )
    attention_run = json.loads(
        attention_path.read_text(encoding="utf-8")
    )
    attention_points = []
    for record in attention_run["records"]:
        if record["implementation"] != "flash_attn":
            continue
        batch, sequence, heads, head_dim = record[
            "shape_bshd"
        ]
        causal_pairs = (
            batch
            * sequence
            * (sequence + 1)
            / 2.0
        )
        attention_points.append(
            {
                "batch": batch,
                "sequence": sequence,
                "heads": heads,
                "head_dim": head_dim,
                "causal_pairs": causal_pairs,
                "measured_ms": record["median_ms"],
            }
        )
    if len(attention_points) != 3:
        raise ValueError(
            "Unexpected FlashAttention calibration set"
        )
    design = np.array(
        [
            [
                1.0,
                point["causal_pairs"],
            ]
            for point in attention_points
        ],
        dtype=np.float64,
    )
    measured = np.array(
        [
            point["measured_ms"]
            for point in attention_points
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
        attention_points,
        predicted,
        relative_errors,
    ):
        point["predicted_ms"] = float(prediction)
        point["relative_error"] = float(error)

    matmul_path = (
        root
        / "results"
        / "raw"
        / "full"
        / "matmul_full.json"
    )
    matmul_run = json.loads(
        matmul_path.read_text(encoding="utf-8")
    )
    cublas_records = {
        tuple(record["shape_mnk"]): record
        for record in matmul_run["records"]
        if record["implementation"]
        == "torch_cublas"
    }
    square_small = cublas_records[
        (512, HIDDEN, HIDDEN)
    ]["median_ms"]
    square_large = cublas_records[
        (4096, HIDDEN, HIDDEN)
    ]["median_ms"]
    per_token_ms = (
        square_large - square_small
    ) / (
        4096 - 512
    )
    launch_ms = (
        square_small
        - per_token_ms * 512
    )
    throughput_tflop_s = (
        2.0
        * HIDDEN
        * HIDDEN
        / (per_token_ms / 1_000.0)
        / 1e12
    )
    validation_points = []
    for shape in (
        (4096, INTERMEDIATE, HIDDEN),
        (8192, HIDDEN, INTERMEDIATE),
    ):
        measured_ms = cublas_records[
            shape
        ]["median_ms"]
        m, n, k = shape
        predicted_ms = (
            launch_ms
            + (
                2.0
                * m
                * n
                * k
                / throughput_tflop_s
                / 1e9
            )
        )
        validation_points.append(
            {
                "shape_mnk": shape,
                "measured_ms": measured_ms,
                "predicted_ms": predicted_ms,
                "relative_error": abs(
                    predicted_ms - measured_ms
                )
                / measured_ms,
            }
        )

    dense_flops_per_token = (
        2 * HIDDEN * HIDDEN
        + 2 * HIDDEN * HIDDEN
        + 2 * HIDDEN * KV_PROJECTION
        + 2 * HIDDEN * KV_PROJECTION
        + 2 * HIDDEN * INTERMEDIATE
        + 2 * HIDDEN * INTERMEDIATE
        + 2 * INTERMEDIATE * HIDDEN
    )
    return {
        "attention": {
            "source": attention_path.name,
            "sha256": sha256(attention_path),
            "measured_heads": (
                MEASURED_ATTENTION_HEADS
            ),
            "target_query_heads": NUM_QO_HEADS,
            "head_scale": (
                NUM_QO_HEADS
                / MEASURED_ATTENTION_HEADS
            ),
            "intercept_ms": float(
                coefficients[0]
            ),
            "causal_pair_ms": float(
                coefficients[1]
            ),
            "max_relative_error": float(
                np.max(relative_errors)
            ),
            "points": attention_points,
        },
        "dense": {
            "source": matmul_path.name,
            "sha256": sha256(matmul_path),
            "launch_ms": launch_ms,
            "throughput_tflop_s": (
                throughput_tflop_s
            ),
            "gemm_launches_per_layer": (
                GEMM_LAUNCHES_PER_LAYER
            ),
            "dense_flops_per_token": (
                dense_flops_per_token
            ),
            "fit_points": [
                {
                    "shape_mnk": (
                        512,
                        HIDDEN,
                        HIDDEN,
                    ),
                    "measured_ms": square_small,
                },
                {
                    "shape_mnk": (
                        4096,
                        HIDDEN,
                        HIDDEN,
                    ),
                    "measured_ms": square_large,
                },
            ],
            "validation_points": (
                validation_points
            ),
            "max_validation_relative_error": (
                max(
                    point["relative_error"]
                    for point in validation_points
                )
            ),
        },
        "architecture": {
            "layers": LAYERS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "num_query_heads": NUM_QO_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "kv_projection": KV_PROJECTION,
        },
    }


def prefill_chunk_service(
    model: dict[str, Any],
    *,
    query_tokens: int,
    prefix_tokens: int,
) -> dict[str, float]:
    if query_tokens < 1:
        raise ValueError(query_tokens)
    if prefix_tokens < 0:
        raise ValueError(prefix_tokens)
    causal_pairs = (
        query_tokens * prefix_tokens
        + query_tokens
        * (query_tokens + 1)
        / 2.0
    )
    attention = model["attention"]
    attention_ms_per_layer = (
        attention["intercept_ms"]
        + attention["causal_pair_ms"]
        * causal_pairs
        * attention["head_scale"]
    )
    dense = model["dense"]
    dense_ms_per_layer = (
        dense["gemm_launches_per_layer"]
        * dense["launch_ms"]
        + (
            query_tokens
            * dense["dense_flops_per_token"]
            / dense["throughput_tflop_s"]
            / 1e9
        )
    )
    if (
        attention_ms_per_layer <= 0.0
        or dense_ms_per_layer <= 0.0
    ):
        raise RuntimeError(
            "Non-positive prefill service prediction"
        )
    return {
        "total_ms": LAYERS
        * (
            attention_ms_per_layer
            + dense_ms_per_layer
        ),
        "attention_ms": (
            LAYERS * attention_ms_per_layer
        ),
        "dense_ms": (
            LAYERS * dense_ms_per_layer
        ),
        "causal_pairs": causal_pairs,
    }


def exact_suffix_pairs(
    *,
    prompt_tokens: int,
    cached_tokens: int,
) -> float:
    return (
        prompt_tokens
        * (prompt_tokens + 1)
        / 2.0
        - cached_tokens
        * (cached_tokens + 1)
        / 2.0
    )


def chunk_pair_check(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    chunk_tokens: int,
) -> float:
    processed = 0
    total = 0.0
    remaining = prompt_tokens - cached_tokens
    while processed < remaining:
        query = min(
            chunk_tokens,
            remaining - processed,
        )
        total += (
            query * (cached_tokens + processed)
            + query * (query + 1) / 2.0
        )
        processed += query
    return total


def make_prefill_workload(
    *,
    num_requests: int,
    seed: int,
    prefix_mode: str,
) -> dict[str, np.ndarray]:
    workload = make_workload(
        name="mixed_chat",
        num_requests=num_requests,
        seed=seed,
    )
    specification = PREFIX_MODES[prefix_mode]
    rng = np.random.default_rng(seed + 47)
    hits = (
        rng.random(num_requests)
        < specification[
            "request_hit_probability"
        ]
    )
    cached_tokens = np.zeros(
        num_requests,
        dtype=np.int64,
    )
    if specification[
        "cached_prompt_fraction"
    ] > 0.0:
        values = (
            workload["prompt_tokens"]
            * specification[
                "cached_prompt_fraction"
            ]
        ).astype(np.int64)
        values = (values // 16) * 16
        cached_tokens[hits] = values[hits]
    uncached_tokens = (
        workload["prompt_tokens"]
        - cached_tokens
    )
    if np.any(uncached_tokens < 1):
        raise RuntimeError(
            "Prefix mode cached the full prompt"
        )
    return {
        **workload,
        "cached_tokens": cached_tokens,
        "uncached_tokens": uncached_tokens,
        "prefix_hits": hits,
    }


def simulate(
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
    chunk_selector: (
        Callable[[dict[str, Any]], int] | None
    ) = None,
    admission_selector: (
        Callable[[dict[str, Any]], bool] | None
    ) = None,
    decode_service_observer: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    prefill_service_observer: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    latency_origin_arrivals: (
        np.ndarray | None
    ) = None,
) -> dict[str, float]:
    policy = POLICIES[policy_name]
    prompt_tokens = workload["prompt_tokens"]
    output_tokens = workload["output_tokens"]
    reservation_tokens = workload[
        "reservation_tokens"
    ]
    cached_tokens = workload["cached_tokens"]
    uncached_tokens = workload[
        "uncached_tokens"
    ]
    prefix_hits = workload["prefix_hits"]
    num_requests = arrivals.size
    if latency_origin_arrivals is not None:
        if (
            latency_origin_arrivals.size
            != num_requests
        ):
            raise ValueError(
                "Latency-origin length mismatch"
            )
        if np.any(
            latency_origin_arrivals
            > arrivals + 1e-12
        ):
            raise ValueError(
                "Latency origin occurs after "
                "scheduled arrival"
            )
    if not all(
        array.size == num_requests
        for array in (
            prompt_tokens,
            output_tokens,
            reservation_tokens,
            cached_tokens,
            uncached_tokens,
            prefix_hits,
        )
    ):
        raise ValueError("Workload length mismatch")

    prefill_processed = np.zeros(
        num_requests,
        dtype=np.int64,
    )
    generated = np.zeros(
        num_requests,
        dtype=np.int64,
    )
    accepted = (
        np.ones(
            num_requests,
            dtype=np.bool_,
        )
        if admission_selector is None
        else np.zeros(
            num_requests,
            dtype=np.bool_,
        )
    )
    rejected = np.zeros(
        num_requests,
        dtype=np.bool_,
    )
    reserved = np.zeros(
        num_requests,
        dtype=np.bool_,
    )
    prefill_start = np.full(
        num_requests,
        np.nan,
        dtype=np.float64,
    )
    prefill_completion = np.full(
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
    last_token_time = np.full(
        num_requests,
        np.nan,
        dtype=np.float64,
    )
    maximum_token_gap = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    prefill_chunks = np.zeros(
        num_requests,
        dtype=np.int64,
    )

    prefill_queue: deque[int] = deque()
    decode_ready: deque[int] = deque()
    active: list[int] = []
    next_arrival = 0
    now_ms = 0.0
    reserved_tokens = 0
    peak_reserved_tokens = 0
    prefill_busy_ms = 0.0
    prefill_attention_ms = 0.0
    prefill_dense_ms = 0.0
    decode_busy_ms = 0.0
    actual_prefill_pairs = 0.0
    exact_prefill_pairs = 0.0
    actual_prefill_service_ms = 0.0
    monolithic_prefill_service_ms = 0.0
    no_cache_monolithic_service_ms = 0.0
    step_batches: list[int] = []
    decode_step_ms_values: list[float] = []

    if admission_selector is None:
        for request in range(num_requests):
            exact_prefill_pairs += (
                exact_suffix_pairs(
                    prompt_tokens=int(
                        prompt_tokens[request]
                    ),
                    cached_tokens=int(
                        cached_tokens[request]
                    ),
                )
            )
            monolithic_prefill_service_ms += (
                prefill_chunk_service(
                    prefill_model,
                    query_tokens=int(
                        uncached_tokens[request]
                    ),
                    prefix_tokens=int(
                        cached_tokens[request]
                    ),
                )["total_ms"]
            )
            no_cache_monolithic_service_ms += (
                prefill_chunk_service(
                    prefill_model,
                    query_tokens=int(
                        prompt_tokens[request]
                    ),
                    prefix_tokens=0,
                )["total_ms"]
            )

    def add_arrivals() -> None:
        nonlocal next_arrival
        nonlocal exact_prefill_pairs
        nonlocal monolithic_prefill_service_ms
        nonlocal no_cache_monolithic_service_ms
        while (
            next_arrival < num_requests
            and arrivals[next_arrival] <= now_ms
        ):
            request = next_arrival
            allow = (
                True
                if admission_selector is None
                else bool(
                    admission_selector(
                        {
                            "request": request,
                            "now_ms": now_ms,
                            "arrival_ms": float(
                                arrivals[request]
                            ),
                            "prefill_queue": tuple(
                                prefill_queue
                            ),
                            "decode_ready": tuple(
                                decode_ready
                            ),
                            "active_decode": tuple(
                                active
                            ),
                            "prefill_processed": (
                                prefill_processed
                            ),
                            "cached_tokens": (
                                cached_tokens
                            ),
                            "uncached_tokens": (
                                uncached_tokens
                            ),
                            "prompt_tokens": (
                                prompt_tokens
                            ),
                            "output_tokens": (
                                output_tokens
                            ),
                            "reservation_tokens": (
                                reservation_tokens
                            ),
                            "reserved_tokens": (
                                reserved_tokens
                            ),
                            "kv_budget_tokens": (
                                kv_budget_tokens
                            ),
                        }
                    )
                )
            )
            if allow:
                accepted[request] = True
                prefill_queue.append(request)
                if admission_selector is not None:
                    exact_prefill_pairs += (
                        exact_suffix_pairs(
                            prompt_tokens=int(
                                prompt_tokens[
                                    request
                                ]
                            ),
                            cached_tokens=int(
                                cached_tokens[
                                    request
                                ]
                            ),
                        )
                    )
                    monolithic_prefill_service_ms += (
                        prefill_chunk_service(
                            prefill_model,
                            query_tokens=int(
                                uncached_tokens[
                                    request
                                ]
                            ),
                            prefix_tokens=int(
                                cached_tokens[
                                    request
                                ]
                            ),
                        )["total_ms"]
                    )
                    no_cache_monolithic_service_ms += (
                        prefill_chunk_service(
                            prefill_model,
                            query_tokens=int(
                                prompt_tokens[
                                    request
                                ]
                            ),
                            prefix_tokens=0,
                        )["total_ms"]
                    )
            else:
                rejected[request] = True
            next_arrival += 1

    def admit_decode_ready() -> None:
        while (
            decode_ready
            and len(active) < MAX_BATCH
        ):
            active.append(
                decode_ready.popleft()
            )

    def run_prefill_stage() -> bool:
        nonlocal now_ms
        nonlocal reserved_tokens
        nonlocal peak_reserved_tokens
        nonlocal prefill_busy_ms
        nonlocal prefill_attention_ms
        nonlocal prefill_dense_ms
        nonlocal actual_prefill_pairs
        nonlocal actual_prefill_service_ms
        if not prefill_queue:
            return False
        request = prefill_queue[0]
        if not reserved[request]:
            reservation = int(
                reservation_tokens[request]
            )
            if (
                reserved_tokens + reservation
                > kv_budget_tokens
            ):
                return False
            reserved[request] = True
            reserved_tokens += reservation
            peak_reserved_tokens = max(
                peak_reserved_tokens,
                reserved_tokens,
            )
            prefill_start[request] = now_ms

        remaining = (
            int(uncached_tokens[request])
            - int(prefill_processed[request])
        )
        chunk_limit = policy["chunk_tokens"]
        prefix = (
            int(cached_tokens[request])
            + int(prefill_processed[request])
        )
        query = (
            int(
                chunk_selector(
                    {
                        "request": request,
                        "now_ms": now_ms,
                        "arrival_ms": float(
                            arrivals[request]
                        ),
                        "queue_age_ms": (
                            now_ms
                            - float(
                                arrivals[request]
                            )
                        ),
                        "prefill_queue_depth": len(
                            prefill_queue
                        ),
                        "active_decode_requests": len(
                            active
                        ),
                        "decode_ready_requests": len(
                            decode_ready
                        ),
                        "remaining_tokens": remaining,
                        "prefix_tokens": prefix,
                    }
                )
            )
            if chunk_selector is not None
            else (
                remaining
                if chunk_limit is None
                else min(
                    remaining,
                    int(chunk_limit),
                )
            )
        )
        if not 1 <= query <= remaining:
            raise RuntimeError(
                "Invalid selected prefill chunk: "
                f"{query=} {remaining=}"
            )
        service = prefill_chunk_service(
            prefill_model,
            query_tokens=query,
            prefix_tokens=prefix,
        )
        started_ms = now_ms
        now_ms += service["total_ms"]
        if prefill_service_observer is not None:
            prefill_service_observer(
                {
                    "request": request,
                    "started_ms": started_ms,
                    "ended_ms": now_ms,
                    "service_ms": service[
                        "total_ms"
                    ],
                    "query_tokens": query,
                    "prefix_tokens": prefix,
                }
            )
        prefill_busy_ms += service["total_ms"]
        prefill_attention_ms += service[
            "attention_ms"
        ]
        prefill_dense_ms += service[
            "dense_ms"
        ]
        actual_prefill_pairs += service[
            "causal_pairs"
        ]
        actual_prefill_service_ms += service[
            "total_ms"
        ]
        prefill_processed[request] += query
        prefill_chunks[request] += 1
        if (
            prefill_processed[request]
            == uncached_tokens[request]
        ):
            prefill_queue.popleft()
            prefill_completion[request] = now_ms
            decode_ready.append(request)
        return True

    def run_decode_stage() -> bool:
        nonlocal now_ms
        nonlocal reserved_tokens
        nonlocal decode_busy_ms
        admit_decode_ready()
        if not active:
            return False
        contexts = np.array(
            [
                int(prompt_tokens[request])
                + int(generated[request])
                for request in active
            ],
            dtype=np.int64,
        )
        service_ms = decode_step_ms(
            decode_model,
            decode_service_models,
            contexts,
            kernel_mode="ragged",
        )
        ended_ms = now_ms + service_ms
        if decode_service_observer is not None:
            decode_service_observer(
                {
                    "started_ms": now_ms,
                    "ended_ms": ended_ms,
                    "service_ms": service_ms,
                    "active_requests": tuple(
                        active
                    ),
                    "contexts": tuple(
                        int(value)
                        for value in contexts
                    ),
                }
            )
        step_batches.append(len(active))
        decode_step_ms_values.append(service_ms)
        survivors = []
        for request in active:
            if generated[request] == 0:
                first_token[request] = ended_ms
            else:
                gap = (
                    ended_ms
                    - last_token_time[request]
                )
                maximum_token_gap[request] = max(
                    maximum_token_gap[request],
                    gap,
                )
            last_token_time[request] = ended_ms
            generated[request] += 1
            if (
                generated[request]
                == output_tokens[request]
            ):
                completion[request] = ended_ms
                reserved_tokens -= int(
                    reservation_tokens[request]
                )
                reserved[request] = False
            else:
                survivors.append(request)
        active[:] = survivors
        now_ms = ended_ms
        decode_busy_ms += service_ms
        return True

    while (
        next_arrival < num_requests
        or prefill_queue
        or decode_ready
        or active
    ):
        add_arrivals()
        progressed = False
        if policy["order"] == "prefill_first":
            progressed = (
                run_prefill_stage()
                or progressed
            )
            add_arrivals()
            progressed = (
                run_decode_stage()
                or progressed
            )
        elif policy["order"] == "decode_first":
            for _ in range(
                int(
                    policy[
                        "decode_steps_per_prefill"
                    ]
                )
            ):
                decode_progress = (
                    run_decode_stage()
                )
                progressed = (
                    decode_progress or progressed
                )
                add_arrivals()
                if not decode_progress:
                    break
            progressed = (
                run_prefill_stage()
                or progressed
            )
        else:
            raise ValueError(policy["order"])
        if progressed:
            continue
        if next_arrival < num_requests:
            now_ms = max(
                now_ms,
                float(arrivals[next_arrival]),
            )
            continue
        if prefill_queue:
            request = prefill_queue[0]
            if (
                int(reservation_tokens[request])
                > kv_budget_tokens
            ):
                raise RuntimeError(
                    "One request exceeds KV budget"
                )
        raise RuntimeError(
            "Scheduler made no progress"
        )

    if not np.all(accepted | rejected):
        raise RuntimeError(
            "Not every request was classified"
        )
    if np.any(accepted & rejected):
        raise RuntimeError(
            "Request both accepted and rejected"
        )
    if not np.array_equal(
        prefill_processed,
        np.where(
            accepted,
            uncached_tokens,
            0,
        ),
    ):
        raise RuntimeError(
            "Prefill token accounting failed"
        )
    if not np.array_equal(
        generated,
        np.where(
            accepted,
            output_tokens,
            0,
        ),
    ):
        raise RuntimeError(
            "Decode token accounting failed"
        )
    if not np.all(
        np.isfinite(
            prefill_completion[accepted]
        )
    ) or not np.all(
        np.isfinite(completion[accepted])
    ):
        raise RuntimeError(
            "A request did not complete"
        )
    if np.any(
        np.isfinite(
            prefill_completion[rejected]
        )
    ) or np.any(
        np.isfinite(completion[rejected])
    ):
        raise RuntimeError(
            "Rejected request completed"
        )
    if reserved_tokens != 0:
        raise RuntimeError("KV reservation leak")
    if peak_reserved_tokens > kv_budget_tokens:
        raise RuntimeError("KV budget exceeded")
    if step_batches and max(step_batches) > MAX_BATCH:
        raise RuntimeError(
            "Batch-size invariant failed"
        )
    if abs(
        actual_prefill_pairs
        - exact_prefill_pairs
    ) > 1e-6:
        raise RuntimeError(
            "Chunked attention-pair accounting failed"
        )

    selected_window = np.arange(
        analysis_start,
        analysis_end,
    )
    selected = selected_window[
        accepted[selected_window]
    ]
    selected_rejected = selected_window[
        rejected[selected_window]
    ]
    if selected.size < 1:
        raise RuntimeError(
            "No accepted requests in analysis window"
        )
    latency_arrivals = (
        arrivals
        if latency_origin_arrivals is None
        else latency_origin_arrivals
    )
    selected_arrivals = (
        latency_arrivals[selected]
    )
    selected_scheduled_arrivals = (
        arrivals[selected]
    )
    selected_prefill_start = prefill_start[
        selected
    ]
    selected_prefill_completion = (
        prefill_completion[selected]
    )
    selected_first = first_token[selected]
    selected_completion = completion[selected]
    prefill_queue_wait = (
        selected_prefill_start
        - selected_arrivals
    )
    prefill_latency = (
        selected_prefill_completion
        - selected_arrivals
    )
    ttft = (
        selected_first
        - selected_arrivals
    )
    decode_activation_wait = (
        selected_first
        - selected_prefill_completion
    )
    tpot = (
        selected_completion
        - selected_first
    ) / (
        output_tokens[selected] - 1
    )
    e2e = (
        selected_completion
        - selected_arrivals
    )
    server_prefill_queue_wait = (
        selected_prefill_start
        - selected_scheduled_arrivals
    )
    server_prefill_latency = (
        selected_prefill_completion
        - selected_scheduled_arrivals
    )
    server_ttft = (
        selected_first
        - selected_scheduled_arrivals
    )
    server_e2e = (
        selected_completion
        - selected_scheduled_arrivals
    )
    elapsed_ms = (
        float(np.max(selected_completion))
        - float(
            latency_arrivals[
                analysis_start
            ]
        )
    )
    selected_outputs = int(
        np.sum(output_tokens[selected])
    )
    selected_count = selected.size
    selected_cached = cached_tokens[selected]
    selected_prompts = prompt_tokens[selected]
    result = {
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
        "p50_prefill_queue_ms": percentile(
            prefill_queue_wait,
            50,
        ),
        "p99_prefill_queue_ms": percentile(
            prefill_queue_wait,
            99,
        ),
        "p50_prefill_latency_ms": percentile(
            prefill_latency,
            50,
        ),
        "p99_prefill_latency_ms": percentile(
            prefill_latency,
            99,
        ),
        "p50_ttft_ms": percentile(ttft, 50),
        "p95_ttft_ms": percentile(ttft, 95),
        "p99_ttft_ms": percentile(ttft, 99),
        "p50_decode_activation_wait_ms": (
            percentile(
                decode_activation_wait,
                50,
            )
        ),
        "p99_decode_activation_wait_ms": (
            percentile(
                decode_activation_wait,
                99,
            )
        ),
        "p50_tpot_ms": percentile(tpot, 50),
        "p95_tpot_ms": percentile(tpot, 95),
        "p99_tpot_ms": percentile(tpot, 99),
        "p99_max_intertoken_gap_ms": (
            percentile(
                maximum_token_gap[selected],
                99,
            )
        ),
        "p50_e2e_ms": percentile(e2e, 50),
        "p99_e2e_ms": percentile(e2e, 99),
        "mean_batch_size": float(
            np.mean(step_batches)
        ),
        "mean_decode_step_ms": float(
            np.mean(decode_step_ms_values)
        ),
        "mean_prefill_chunks": float(
            np.mean(prefill_chunks[selected])
        ),
        "p95_prefill_chunks": percentile(
            prefill_chunks[selected],
            95,
        ),
        "prefix_hit_request_fraction": float(
            np.mean(prefix_hits[selected])
        ),
        "cached_prompt_token_fraction": float(
            np.sum(selected_cached)
            / np.sum(selected_prompts)
        ),
        "prefill_chunk_overhead": (
            actual_prefill_service_ms
            / monolithic_prefill_service_ms
        ),
        "prefix_compute_ratio_vs_no_cache": (
            monolithic_prefill_service_ms
            / no_cache_monolithic_service_ms
        ),
        "prefill_attention_fraction": (
            prefill_attention_ms
            / prefill_busy_ms
        ),
        "prefill_busy_fraction": (
            prefill_busy_ms
            / (
                prefill_busy_ms
                + decode_busy_ms
            )
        ),
        "prefill_busy_ms": prefill_busy_ms,
        "decode_busy_ms": decode_busy_ms,
        "peak_kv_tokens": float(
            peak_reserved_tokens
        ),
        "peak_kv_gib": (
            peak_reserved_tokens
            * KV_BYTES_PER_TOKEN
            / (1024**3)
        ),
        "completed_output_tokens": float(
            np.sum(output_tokens[accepted])
        ),
    }
    if latency_origin_arrivals is not None:
        selected_ingress_delay = (
            selected_scheduled_arrivals
            - selected_arrivals
        )
        result.update(
            {
                "mean_ingress_delay_ms": (
                    float(
                        np.mean(
                            selected_ingress_delay
                        )
                    )
                ),
                "p50_ingress_delay_ms": (
                    percentile(
                        selected_ingress_delay,
                        50,
                    )
                ),
                "p95_ingress_delay_ms": (
                    percentile(
                        selected_ingress_delay,
                        95,
                    )
                ),
                "p99_ingress_delay_ms": (
                    percentile(
                        selected_ingress_delay,
                        99,
                    )
                ),
                "maximum_ingress_delay_ms": (
                    float(
                        np.max(
                            selected_ingress_delay
                        )
                    )
                ),
                "p99_server_prefill_queue_ms": (
                    percentile(
                        server_prefill_queue_wait,
                        99,
                    )
                ),
                "p99_server_prefill_latency_ms": (
                    percentile(
                        server_prefill_latency,
                        99,
                    )
                ),
                "p99_server_ttft_ms": (
                    percentile(
                        server_ttft,
                        99,
                    )
                ),
                "p99_server_e2e_ms": (
                    percentile(
                        server_e2e,
                        99,
                    )
                ),
            }
        )
    if admission_selector is not None:
        offered_outputs = int(
            np.sum(output_tokens[selected_window])
        )
        result.update(
            {
                "accepted_fraction": (
                    selected_count
                    / selected_window.size
                ),
                "rejected_fraction": (
                    selected_rejected.size
                    / selected_window.size
                ),
                "accepted_output_token_fraction": (
                    selected_outputs
                    / offered_outputs
                ),
                "mean_accepted_prompt_tokens": (
                    float(
                        np.mean(
                            prompt_tokens[selected]
                        )
                    )
                ),
                "mean_rejected_prompt_tokens": (
                    float(
                        np.mean(
                            prompt_tokens[
                                selected_rejected
                            ]
                        )
                    )
                    if selected_rejected.size
                    else 0.0
                ),
                "mean_accepted_output_tokens": (
                    float(
                        np.mean(
                            output_tokens[selected]
                        )
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
            }
        )
    return result


def estimate_capacity(
    *,
    num_requests: int,
    seed: int,
    prefix_mode: str,
    policy_name: str,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
    prefill_model: dict[str, Any],
    kv_budget_tokens: int,
) -> dict[str, float]:
    workload = make_prefill_workload(
        num_requests=num_requests,
        seed=seed,
        prefix_mode=prefix_mode,
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
        policy_name=policy_name,
        kv_budget_tokens=kv_budget_tokens,
        analysis_start=0,
        analysis_end=num_requests,
    )
    busy_ms = (
        metrics["prefill_busy_ms"]
        + metrics["decode_busy_ms"]
    )
    return {
        "requests_per_second": (
            num_requests
            * 1_000.0
            / busy_ms
        ),
        "output_tokens_per_second": (
            metrics["completed_output_tokens"]
            * 1_000.0
            / busy_ms
        ),
        "prefill_chunk_overhead": metrics[
            "prefill_chunk_overhead"
        ],
        "prefix_compute_ratio_vs_no_cache": (
            metrics[
                "prefix_compute_ratio_vs_no_cache"
            ]
        ),
        "prefill_busy_fraction": metrics[
            "prefill_busy_fraction"
        ],
        "mean_prefill_chunks": metrics[
            "mean_prefill_chunks"
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    decode_service_models = load_service_models(
        args.root
    )
    prefill_model = load_prefill_service_model(
        args.root
    )
    max_attention_fit_error = prefill_model[
        "attention"
    ]["max_relative_error"]
    max_dense_validation_error = prefill_model[
        "dense"
    ]["max_validation_relative_error"]
    if (
        max_attention_fit_error
        > args.max_attention_fit_relative_error
    ):
        raise RuntimeError(
            "Prefill attention fit gate failed: "
            f"{max_attention_fit_error=}"
        )
    if (
        max_dense_validation_error
        > args.max_dense_relative_error
    ):
        raise RuntimeError(
            "Dense GEMM fit gate failed: "
            f"{max_dense_validation_error=}"
        )

    pair_checks = []
    for prompt in (512, 2048, 6144):
        for cached_fraction in (0.0, 0.75):
            cached = int(
                prompt * cached_fraction
            )
            cached = (cached // 16) * 16
            expected = exact_suffix_pairs(
                prompt_tokens=prompt,
                cached_tokens=cached,
            )
            for chunk in (
                128,
                256,
                512,
                1024,
                2048,
            ):
                actual = chunk_pair_check(
                    prompt_tokens=prompt,
                    cached_tokens=cached,
                    chunk_tokens=chunk,
                )
                if actual != expected:
                    raise RuntimeError(
                        "Chunk pair validation failed"
                    )
                pair_checks.append(
                    {
                        "prompt_tokens": prompt,
                        "cached_tokens": cached,
                        "chunk_tokens": chunk,
                        "causal_pairs": expected,
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
    decode_capacities = {}
    capacities: dict[str, dict[str, Any]] = {}
    for prefix_mode in PREFIX_MODES:
        capacities[prefix_mode] = {}
        for replicate_index, decode_model in enumerate(
            decode_service_models[
                "attention_replicates"
            ]
        ):
            source = decode_model["source"]
            decode_only = (
                decode_service_models[
                    "attention_replicates"
                ][replicate_index]
            )
            decode_workload = make_workload(
                name="mixed_chat",
                num_requests=args.capacity_requests,
                seed=(
                    args.seed
                    + 500_000
                    + replicate_index * 10_000
                    + 1_000
                ),
            )
            from simulate_decode_continuous_batching import (
                simulate as simulate_decode_only,
            )

            decode_metrics = simulate_decode_only(
                arrivals=np.zeros(
                    args.capacity_requests,
                    dtype=np.float64,
                ),
                workload=decode_workload,
                model=decode_only,
                service_models=(
                    decode_service_models
                ),
                policy_name="ragged_fcfs",
                kv_budget_tokens=kv_budget_tokens,
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
            capacities[prefix_mode][source] = {}
            for policy_name in POLICIES:
                capacities[prefix_mode][source][
                    policy_name
                ] = estimate_capacity(
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
                    policy_name=policy_name,
                    decode_model=decode_model,
                    decode_service_models=(
                        decode_service_models
                    ),
                    prefill_model=prefill_model,
                    kv_budget_tokens=(
                        kv_budget_tokens
                    ),
                )

    records = []
    scenario_count = 0
    for prefix_mode in PREFIX_MODES:
        for offered_load in offered_loads:
            for policy_name in POLICIES:
                scenarios = []
                for replicate_index, decode_model in enumerate(
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
                        workload = (
                            make_prefill_workload(
                                num_requests=(
                                    args.num_requests
                                ),
                                seed=workload_seed,
                                prefix_mode=(
                                    prefix_mode
                                ),
                            )
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
                        metrics = simulate(
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
                                    capacities[
                                        prefix_mode
                                    ][source][
                                        policy_name
                                    ][
                                        "requests_per_second"
                                    ]
                                ),
                                "offered_load_vs_system_capacity": (
                                    request_rate
                                    / capacities[
                                        prefix_mode
                                    ][source][
                                        policy_name
                                    ][
                                        "requests_per_second"
                                    ]
                                ),
                                **metrics,
                            }
                        )
                        scenario_count += 1
                records.append(
                    {
                        "prefix_mode": prefix_mode,
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

    result = {
        "suite": (
            "prefill_decode_scheduling_simulation"
        ),
        "model": (
            "A measured-service 32-layer Llama-class "
            "prefill/decode scheduler. Prefill combines "
            "a FlashAttention causal-pair fit with a "
            "cuBLAS dense-projection/MLP model; decode "
            "reuses the v37 paged-GQA and MLP model."
        ),
        "prefill_service_model": prefill_model,
        "decode_service_models": (
            decode_service_models
        ),
        "workload": {
            "base": "mixed_chat",
            "prefix_modes": PREFIX_MODES,
        },
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
            "offered_loads_vs_decode_capacity": (
                offered_loads
            ),
            "max_batch": MAX_BATCH,
            "layers": LAYERS,
            "kv_budget_gib": args.kv_budget_gib,
            "kv_budget_tokens": kv_budget_tokens,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        },
        "decode_only_capacities": (
            decode_capacities
        ),
        "system_capacities": capacities,
        "validation": {
            "max_attention_fit_relative_error": (
                max_attention_fit_error
            ),
            "max_allowed_attention_fit_relative_error": (
                args.max_attention_fit_relative_error
            ),
            "max_dense_validation_relative_error": (
                max_dense_validation_error
            ),
            "max_allowed_dense_relative_error": (
                args.max_dense_relative_error
            ),
            "chunk_pair_checks": pair_checks,
            "scenario_count": scenario_count,
            "accounting": (
                "Every scenario asserts exact uncached "
                "prefill and output-token counts, exact "
                "causal attention-pair conservation "
                "across chunks, finite completion, max "
                "batch 64, and KV reservations within "
                "budget."
            ),
        },
        "records": records,
        "limitations": [
            "The FlashAttention and cuBLAS calibration points are real A100 measurements; arrivals, request lengths, prefix hits, 32-layer composition, and scheduler behavior are simulated.",
            "The attention fit uses three MHA causal-prefill points and scales causal-pair work from 16 to 32 query heads. It does not directly measure GQA prefill, KV append, or cross-attention with a cached prefix.",
            "The dense model fits launch cost and throughput from square cuBLAS M=512/4096 and validates within 0.7% on measured up/down projections. It assumes the same efficiency for the smaller K/V projections.",
            "Prefill and decode stages execute sequentially. There is no fused mixed prefill/decode batch, concurrent streams, tensor parallelism, CPU scheduling cost, or GPU jitter.",
            "The prefix mode gives half the requests a deterministic 75% cached prompt. It does not model cache construction, lookup, eviction, block sharing, or cache memory.",
            "Full prompt plus output KV is reserved when prefill starts, even for cached tokens. Production paged allocators and shared prefix blocks can use memory more efficiently.",
            "TTFT includes simulated prefill and decode scheduling but excludes ingress, tokenization, sampling, and network output.",
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
        default="0.1,0.2,0.3,0.4",
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
    parser.add_argument(
        "--max-attention-fit-relative-error",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-dense-relative-error",
        type=float,
        default=0.01,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
