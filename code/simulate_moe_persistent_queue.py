from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


WORLD_SIZES = (8, 24)
POLICIES = ("additive", "adaptive")
MAX_BATCH = 8


def percentile(
    values: np.ndarray,
    q: float,
) -> float:
    return float(np.percentile(values, q))


def focused_files(
    raw_dir: Path,
    world_size: int,
    policy: str,
) -> list[Path]:
    if policy == "additive":
        names = (
            f"moe_persistent_fit_focused_{world_size}gpu_rep3.json",
            (
                "moe_persistent_fit_focused_"
                f"{world_size}gpu_reverse_rep4.json"
            ),
        )
    else:
        names = (
            (
                "moe_persistent_fit_adaptive_focused_"
                f"{world_size}gpu_rep1.json"
            ),
            (
                "moe_persistent_fit_adaptive_focused_"
                f"{world_size}gpu_reverse_rep2.json"
            ),
        )
    return [raw_dir / name for name in names]


def load_service_models(
    raw_dir: Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for world_size in WORLD_SIZES:
        result[world_size] = {}
        for policy in POLICIES:
            paths = focused_files(
                raw_dir,
                world_size,
                policy,
            )
            runs = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in paths
            ]
            replicates = []
            for path, run in zip(paths, runs):
                record = run["records"][0]
                if record["queue_depth"] != MAX_BATCH:
                    raise ValueError(
                        f"Expected queue depth {MAX_BATCH}: {path}"
                    )
                replicates.append(
                    {
                        "source": path.name,
                        "sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                        "scalar_request_ms": (
                            record["timings"]["scalar_host"][
                                "critical_path_ms"
                            ]
                            / MAX_BATCH
                        ),
                        "vector_batch_ms": record["timings"][
                            "vector_host"
                        ]["critical_path_ms"],
                        "ideal_batch_ms": record["timings"][
                            "ideal_precomputed"
                        ]["critical_path_ms"],
                        "dynamic_request_ms": (
                            record["timings"]["dynamic"][
                                "critical_path_ms"
                            ]
                            / MAX_BATCH
                        ),
                        "vector_controller_ms": record[
                            "controller_timings"
                        ]["vector_controller"][
                            "critical_path_ms"
                        ],
                        "hit_fraction": record["hit_fraction"],
                    }
                )
            result[world_size][policy] = {
                "observed_hit_rate": runs[0][
                    "profile_statistics"
                ]["observed_hit_rate"],
                "replicates": replicates,
                "median_scalar_request_ms": statistics.median(
                    item["scalar_request_ms"]
                    for item in replicates
                ),
                "median_vector_batch_ms": statistics.median(
                    item["vector_batch_ms"]
                    for item in replicates
                ),
                "median_vector_capacity_speedup": (
                    statistics.median(
                        item["scalar_request_ms"]
                        for item in replicates
                    )
                    * MAX_BATCH
                    / statistics.median(
                        item["vector_batch_ms"]
                        for item in replicates
                    )
                ),
            }
    return result


def make_arrivals(
    *,
    num_requests: int,
    mean_interarrival_ms: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    intervals = rng.exponential(
        mean_interarrival_ms,
        size=num_requests,
    )
    intervals[0] = 0.0
    return np.cumsum(intervals)


def simulate_queue(
    arrivals: np.ndarray,
    *,
    scalar_service_ms: float,
    vector_batch_ms: float,
    max_wait_ms: float | None,
    completion_model: str,
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    if completion_model not in {
        "streaming",
        "fenced",
    }:
        raise ValueError(completion_model)

    num_requests = arrivals.size
    queue: deque[int] = deque()
    completion = np.empty(
        num_requests,
        dtype=np.float64,
    )
    service_start = np.empty(
        num_requests,
        dtype=np.float64,
    )
    server_time = 0.0
    next_arrival = 0
    vector_requests = 0
    vector_batches = 0
    scalar_requests = 0
    busy_time = 0.0

    while next_arrival < num_requests or queue:
        while (
            next_arrival < num_requests
            and arrivals[next_arrival] <= server_time
        ):
            queue.append(next_arrival)
            next_arrival += 1

        if not queue:
            server_time = max(
                server_time,
                float(arrivals[next_arrival]),
            )
            continue

        if max_wait_ms == 0.0:
            request = queue.popleft()
            started = server_time
            ended = started + scalar_service_ms
            service_start[request] = started
            completion[request] = ended
            server_time = ended
            busy_time += scalar_service_ms
            scalar_requests += 1
            continue

        if len(queue) < MAX_BATCH:
            deadline = (
                math.inf
                if max_wait_ms is None
                else (
                    float(arrivals[queue[0]])
                    + max_wait_ms
                )
            )
            if (
                max_wait_ms != 0.0
                and next_arrival < num_requests
                and float(arrivals[next_arrival]) <= deadline
            ):
                server_time = max(
                    server_time,
                    float(arrivals[next_arrival]),
                )
                continue
            if (
                max_wait_ms is None
                and next_arrival >= num_requests
            ):
                deadline = server_time
            if deadline > server_time:
                server_time = deadline
                while (
                    next_arrival < num_requests
                    and arrivals[next_arrival] <= server_time
                ):
                    queue.append(next_arrival)
                    next_arrival += 1

        if len(queue) >= MAX_BATCH:
            batch = [
                queue.popleft()
                for _ in range(MAX_BATCH)
            ]
            started = server_time
            ended = started + vector_batch_ms
            offsets = (
                np.full(
                    MAX_BATCH,
                    vector_batch_ms,
                    dtype=np.float64,
                )
                if completion_model == "fenced"
                else np.linspace(
                    vector_batch_ms / MAX_BATCH,
                    vector_batch_ms,
                    MAX_BATCH,
                )
            )
            for request, offset in zip(batch, offsets):
                service_start[request] = started
                completion[request] = started + offset
            server_time = ended
            busy_time += vector_batch_ms
            vector_requests += MAX_BATCH
            vector_batches += 1
            continue

        request = queue.popleft()
        started = server_time
        ended = started + scalar_service_ms
        service_start[request] = started
        completion[request] = ended
        server_time = ended
        busy_time += scalar_service_ms
        scalar_requests += 1

    selected = slice(analysis_start, analysis_end)
    selected_arrivals = arrivals[selected]
    selected_starts = service_start[selected]
    selected_completion = completion[selected]
    queue_wait = selected_starts - selected_arrivals
    response = selected_completion - selected_arrivals
    elapsed = (
        float(selected_completion[-1])
        - float(selected_arrivals[0])
    )
    selected_count = analysis_end - analysis_start
    return {
        "throughput_requests_s": (
            selected_count * 1_000.0 / elapsed
        ),
        "mean_queue_ms": float(np.mean(queue_wait)),
        "p50_queue_ms": percentile(queue_wait, 50),
        "p95_queue_ms": percentile(queue_wait, 95),
        "p99_queue_ms": percentile(queue_wait, 99),
        "mean_response_ms": float(np.mean(response)),
        "p50_response_ms": percentile(response, 50),
        "p95_response_ms": percentile(response, 95),
        "p99_response_ms": percentile(response, 99),
        "max_response_ms": float(np.max(response)),
        "response_over_5ms": float(
            np.mean(response > 5.0)
        ),
        "response_over_10ms": float(
            np.mean(response > 10.0)
        ),
        "response_over_20ms": float(
            np.mean(response > 20.0)
        ),
        "vector_request_fraction": (
            vector_requests / num_requests
        ),
        "vector_batches": float(vector_batches),
        "scalar_requests": float(scalar_requests),
        "busy_fraction": (
            busy_time
            / max(server_time - float(arrivals[0]), 1e-12)
        ),
    }


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


def md1_mean_response_ms(
    service_ms: float,
    offered_load: float,
) -> float:
    return service_ms + (
        offered_load
        * service_ms
        / (2.0 * (1.0 - offered_load))
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = args.root / "results" / "raw" / "moe_persistent_fit"
    service_models = load_service_models(raw_dir)
    offered_loads = [
        float(value)
        for value in args.offered_loads.split(",")
    ]
    max_waits: list[float | None] = []
    for value in args.max_waits.split(","):
        max_waits.append(
            None
            if value == "unbounded"
            else float(value)
        )
    seeds = [
        args.seed + index
        for index in range(args.seeds)
    ]
    records = []
    md1_checks = []

    for world_size in WORLD_SIZES:
        for policy in POLICIES:
            scalar_ms = service_models[world_size][policy][
                "median_scalar_request_ms"
            ]
            for offered_load in (0.25, 0.5, 0.75):
                arrivals = make_arrivals(
                    num_requests=args.validation_requests,
                    mean_interarrival_ms=(
                        scalar_ms / offered_load
                    ),
                    seed=(
                        args.seed
                        + world_size * 100
                        + (0 if policy == "additive" else 1)
                    ),
                )
                metrics = simulate_queue(
                    arrivals,
                    scalar_service_ms=scalar_ms,
                    vector_batch_ms=scalar_ms * MAX_BATCH,
                    max_wait_ms=0.0,
                    completion_model="streaming",
                    analysis_start=(
                        args.validation_requests // 10
                    ),
                    analysis_end=(
                        args.validation_requests
                        - args.validation_requests // 20
                    ),
                )
                theoretical = md1_mean_response_ms(
                    scalar_ms,
                    offered_load,
                )
                md1_checks.append(
                    {
                        "world_size": world_size,
                        "policy": policy,
                        "offered_load": offered_load,
                        "simulated_ms": metrics[
                            "mean_response_ms"
                        ],
                        "theoretical_ms": theoretical,
                        "relative_error": abs(
                            metrics["mean_response_ms"]
                            - theoretical
                        )
                        / theoretical,
                    }
                )

    for world_size in WORLD_SIZES:
        for policy in POLICIES:
            model = service_models[world_size][policy]
            for offered_load in offered_loads:
                for max_wait_ms in max_waits:
                    for completion_model in (
                        "streaming",
                        "fenced",
                    ):
                        scenarios = []
                        for replicate_index, replicate in enumerate(
                            model["replicates"]
                        ):
                            scalar_ms = replicate[
                                "scalar_request_ms"
                            ]
                            mean_interarrival_ms = (
                                scalar_ms / offered_load
                            )
                            for seed in seeds:
                                arrivals = make_arrivals(
                                    num_requests=args.num_requests,
                                    mean_interarrival_ms=(
                                        mean_interarrival_ms
                                    ),
                                    seed=(
                                        seed
                                        + world_size * 10_000
                                        + replicate_index * 1_000
                                    ),
                                )
                                metrics = simulate_queue(
                                    arrivals,
                                    scalar_service_ms=scalar_ms,
                                    vector_batch_ms=replicate[
                                        "vector_batch_ms"
                                    ],
                                    max_wait_ms=max_wait_ms,
                                    completion_model=completion_model,
                                    analysis_start=args.warmup_requests,
                                    analysis_end=(
                                        args.num_requests
                                        - args.tail_guard_requests
                                    ),
                                )
                                scenarios.append(
                                    {
                                        "replicate": replicate[
                                            "source"
                                        ],
                                        "seed": seed,
                                        **metrics,
                                    }
                                )
                        records.append(
                            {
                                "world_size": world_size,
                                "policy": policy,
                                "offered_load_vs_scalar": (
                                    offered_load
                                ),
                                "max_wait_ms": max_wait_ms,
                                "completion_model": (
                                    completion_model
                                ),
                                "metrics": aggregate_metrics(
                                    scenarios
                                ),
                                "scenarios": scenarios,
                            }
                        )

    max_md1_error = max(
        item["relative_error"]
        for item in md1_checks
    )
    if max_md1_error > args.max_md1_relative_error:
        raise RuntimeError(
            "M/D/1 validation failed: "
            f"{max_md1_error=}"
        )

    result = {
        "suite": "moe_persistent_queue_simulation",
        "model": (
            "Single-server FCFS with Poisson arrivals. Full queues of eight "
            "use the measured vector path; timeout expiration serves the "
            "oldest request with the measured scalar path."
        ),
        "service_models": service_models,
        "simulation": {
            "num_requests": args.num_requests,
            "warmup_requests": args.warmup_requests,
            "tail_guard_requests": args.tail_guard_requests,
            "seeds": seeds,
            "validation_requests": (
                args.validation_requests
            ),
            "offered_loads_vs_scalar": offered_loads,
            "max_waits_ms": max_waits,
            "max_batch": MAX_BATCH,
            "completion_models": [
                "streaming",
                "fenced",
            ],
        },
        "validation": {
            "max_md1_relative_error": max_md1_error,
            "max_allowed_md1_relative_error": (
                args.max_md1_relative_error
            ),
            "md1_checks": md1_checks,
        },
        "records": records,
        "limitations": [
            "GPU service times come from real v35 depth-8 measurements, but arrivals and queueing are simulated.",
            "The streaming completion model evenly spaces request completions across the measured batch service time; the fenced model releases all requests at batch end. They bound the unmeasured per-request completion schedule.",
            "Service times are deterministic within each measured replicate. Tail latency therefore comes from arrivals and batching rather than modeled GPU jitter.",
            "The timeout policy forms only full vector batches. Partial queues fall back one request at a time to the measured scalar path.",
            "The model excludes scheduler CPU cost, network ingress, other model layers, fairness, cancellation, and request-length heterogeneity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(
        args.output.suffix + ".tmp"
    )
    temp.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(args.output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--offered-loads",
        default="0.25,0.5,0.75,0.9,1.0,1.05,1.1,1.2",
    )
    parser.add_argument(
        "--max-waits",
        default="0,0.5,1,2,4,8,16,unbounded",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=30_000,
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=3_000,
    )
    parser.add_argument(
        "--tail-guard-requests",
        type=int,
        default=1_000,
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=35_000)
    parser.add_argument(
        "--validation-requests",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--max-md1-relative-error",
        type=float,
        default=0.05,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
