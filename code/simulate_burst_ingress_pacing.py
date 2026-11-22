from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulate_causal_service_commitments import (
    ARRIVAL_MODES,
    make_arrival_trace,
    simulate_policy,
)
from simulate_decode_continuous_batching import (
    KV_BYTES_PER_TOKEN,
    MAX_BATCH,
    aggregate_metrics,
    load_service_models,
    make_workload,
)
from simulate_decode_continuous_batching import (
    simulate as simulate_decode_only,
)
from simulate_prefill_decode_scheduling import (
    PREFIX_MODES,
    load_prefill_service_model,
)
from simulate_tenant_reservation_fairness import (
    estimate_tenant_capacity,
    make_tenant_workload,
    tenant_specifications,
)


POLICIES = (
    "tenant_equal_actual_service40_gain16",
    "tenant_equal_batch32_commit40_gain16",
)

BURST_MODES = (
    "burst_4",
    "burst_16",
)


def pace_arrivals(
    *,
    original_arrivals: np.ndarray,
    pacing_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if not 0.0 <= pacing_fraction < 1.0:
        raise ValueError(
            "Pacing fraction must be in "
            "[0, 1)"
        )
    scheduled = np.array(
        original_arrivals,
        dtype=np.float64,
        copy=True,
    )
    origins, starts = np.unique(
        original_arrivals,
        return_index=True,
    )
    order = np.argsort(starts)
    origins = origins[order]
    starts = starts[order]
    if origins.size < 2:
        raise RuntimeError(
            "Pacing requires multiple bursts"
        )
    periods = np.diff(origins)
    if np.any(periods <= 0.0):
        raise RuntimeError(
            "Invalid burst periods"
        )
    windows = []
    sizes = []
    for index, start in enumerate(starts):
        end = (
            int(starts[index + 1])
            if index + 1 < starts.size
            else original_arrivals.size
        )
        size = end - int(start)
        period_ms = float(
            periods[
                min(
                    index,
                    periods.size - 1,
                )
            ]
        )
        window_ms = (
            pacing_fraction * period_ms
        )
        offsets = (
            np.linspace(
                0.0,
                window_ms,
                size,
                dtype=np.float64,
            )
            if size > 1
            else np.zeros(
                1,
                dtype=np.float64,
            )
        )
        scheduled[
            int(start):end
        ] += offsets
        windows.append(window_ms)
        sizes.append(size)
    if np.any(
        scheduled < original_arrivals
    ) or np.any(
        np.diff(scheduled) < -1e-12
    ):
        raise RuntimeError(
            "Paced arrivals are invalid"
        )
    return scheduled, {
        "pacing_fraction": (
            pacing_fraction
        ),
        "mean_burst_period_ms": float(
            np.mean(periods)
        ),
        "mean_pacing_window_ms": float(
            np.mean(windows)
        ),
        "mean_burst_size": float(
            np.mean(sizes)
        ),
        "maximum_burst_size": float(
            max(sizes)
        ),
    }


def e60_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "causal_service_commitments"
        / "causal_service_commitments_full.json"
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
            record["arrival_mode"],
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
    pacing_fractions = [
        float(value)
        for value
        in args.pacing_fractions.split(",")
    ]
    burst_modes = [
        value.strip()
        for value in args.burst_modes.split(",")
        if value.strip()
    ]
    unknown_modes = set(
        burst_modes
    ) - set(BURST_MODES)
    if unknown_modes:
        raise ValueError(
            f"Unknown burst modes: "
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
            for burst_mode in burst_modes:
                for pacing_fraction in (
                    pacing_fractions
                ):
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
                                        seed=(
                                            workload_seed
                                        ),
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
                                original_arrivals = (
                                    make_arrival_trace(
                                        mode=burst_mode,
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
                                (
                                    scheduled_arrivals,
                                    pacing,
                                ) = pace_arrivals(
                                    original_arrivals=(
                                        original_arrivals
                                    ),
                                    pacing_fraction=(
                                        pacing_fraction
                                    ),
                                )
                                metrics = (
                                    simulate_policy(
                                        arrivals=(
                                            scheduled_arrivals
                                        ),
                                        latency_origin_arrivals=(
                                            original_arrivals
                                        ),
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
                                window = np.arange(
                                    args.warmup_requests,
                                    args.num_requests
                                    - args.tail_guard_requests,
                                    dtype=np.int64,
                                )
                                ingress_delay = (
                                    scheduled_arrivals[
                                        window
                                    ]
                                    - original_arrivals[
                                        window
                                    ]
                                )
                                metrics.update(
                                    {
                                        **pacing,
                                        "mean_offered_ingress_delay_ms": float(
                                            np.mean(
                                                ingress_delay
                                            )
                                        ),
                                        "p95_offered_ingress_delay_ms": float(
                                            np.percentile(
                                                ingress_delay,
                                                95,
                                            )
                                        ),
                                        "p99_offered_ingress_delay_ms": float(
                                            np.percentile(
                                                ingress_delay,
                                                99,
                                            )
                                        ),
                                        "maximum_offered_ingress_delay_ms": float(
                                            np.max(
                                                ingress_delay
                                            )
                                        ),
                                    }
                                )
                                scenarios.append(
                                    {
                                        "service_replicate": (
                                            source
                                        ),
                                        "seed": seed,
                                        "burst_mode": (
                                            burst_mode
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
                                "burst_mode": (
                                    burst_mode
                                ),
                                "pacing_fraction": (
                                    pacing_fraction
                                ),
                                "policy": (
                                    policy_name
                                ),
                                "metrics": (
                                    aggregate_metrics(
                                        scenarios
                                    )
                                ),
                                "scenarios": (
                                    scenarios
                                ),
                            }
                        )

    reference = e60_records(args.root)
    equivalence = []
    equivalence_enabled = (
        args.num_requests == 2_000
        and args.warmup_requests == 200
        and args.tail_guard_requests == 100
        and args.capacity_requests == 1_500
        and args.seeds == 3
        and args.seed == 40_000
        and offered_loads == [0.4]
        and 0.0 in pacing_fractions
        and set(burst_modes)
        == set(BURST_MODES)
    )
    if equivalence_enabled:
        for record in records:
            if record["pacing_fraction"] != 0.0:
                continue
            expected_record = reference[
                (
                    record["prefix_mode"],
                    record[
                        "offered_load_vs_decode_capacity"
                    ],
                    record["burst_mode"],
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
                        "E60 zero-pacing "
                        "equivalence failed: "
                        f"{record['policy']=} "
                        f"{record['burst_mode']=} "
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
                        "burst_mode": (
                            record[
                                "burst_mode"
                            ]
                        ),
                        "policy": (
                            record["policy"]
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
            "burst_ingress_pacing_simulation"
        ),
        "model": (
            "E60 fixed request bursts spread "
            "across a fraction of their natural "
            "inter-burst period while end-to-end "
            "latency remains anchored to the "
            "original client arrival."
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
        "policies": POLICIES,
        "burst_modes": {
            mode: ARRIVAL_MODES[mode]
            for mode in BURST_MODES
        },
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
            "burst_modes": burst_modes,
            "pacing_fractions": (
                pacing_fractions
            ),
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
            "e60_equivalence_enabled": (
                equivalence_enabled
            ),
            "e60_equivalence_checks": (
                equivalence
            ),
            "e60_maximum_absolute_difference": (
                max(
                    item[
                        "maximum_absolute_difference"
                    ]
                    for item in equivalence
                )
                if equivalence
                else None
            ),
            "latency_accounting": (
                "Scheduling uses paced arrivals, "
                "but queue latency, TTFT, E2E, "
                "and throughput elapsed time use "
                "the original client arrivals. "
                "Server-only latency and ingress "
                "delay are reported separately."
            ),
        },
        "records": records,
        "limitations": [
            "The study inherits E60's fitted-service, synthetic-tenant, immediate-rejection, and single-replica limitations.",
            "Pacing spreads each fixed burst uniformly; it is an ingress shaping control rather than an optimized token bucket or tenant-aware queue.",
            "Original client arrivals remain simultaneous, so reported end-to-end TTFT includes intentional ingress delay.",
            "The burst period is deterministic and preserves mean rate but does not model production autocorrelation, retry storms, or cancellation.",
            "Only actual-only and batch-32 equal-share controllers are tested; the objective is burst shaping rather than a new reservation-weight sweep.",
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
        "--burst-modes",
        default="burst_4,burst_16",
    )
    parser.add_argument(
        "--pacing-fractions",
        default="0,0.1,0.25,0.5,0.9",
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
